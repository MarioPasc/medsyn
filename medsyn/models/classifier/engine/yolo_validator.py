from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
from ultralytics.models.yolo.classify.val import ClassificationValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.torch_utils import select_device
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import RANK
from ultralytics.engine.validator import smart_inference_mode, Profile, TQDM, callbacks
from sklearn.metrics import roc_auc_score
from medsyn.models.classifier.dataloaders import build_npz_loader

logger = logging.getLogger(__name__)

class MedsynClassificationValidator(ClassificationValidator):
    """Classification validator that supports NPZ-based datasets without relying on check_cls_dataset()."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None,
                 data_meta: Optional[Dict[str, Any]] = None):
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.medsyn_npz_path: Optional[str] = None              # set from CLI
        self.medsyn_training_images: Optional[str] = None       # set from CLI
        self._data_meta: Dict[str, Any] = data_meta or {}       # {"nc": int, "channels": int, "names": list}
        self.all_targets, self.all_probs = [], []

    def get_dataloader(self, dataset_path, batch_size: int = 16, rank: int = 0, mode: str = "val"):
        """Build NPZ dataloader ignoring dataset_path when we are in NPZ mode."""
        if str(self.args.data) == "__npz__":
            # NPZ mode: use our custom loader
            return build_npz_loader(
                npz_path=self.medsyn_npz_path,
                split={"train": "train", "val": "val", "test": "test"}[mode],
                imgsz=self.args.imgsz,
                batch=batch_size,
                workers=self.args.workers,
                training_images=self.medsyn_training_images,
                augment=False,
            )
        # fallback to default behavior for folder-based datasets
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Run validation. In non-training mode with data='__npz__' skip check_cls_dataset and use NPZ loader."""
        # Reset accumulators
        self.all_targets, self.all_probs = [], []

        self.training = trainer is not None
        if self.training:
            # Training branch: defer to base path which uses trainer.data and never downloads
            return super().__call__(trainer=trainer, model=model)

        # Non-training branch (final_eval, manual val_only)
        if str(self.args.data) == "__npz__":
            logger.info("Running validation in NPZ mode (bypassing check_cls_dataset)")

            # 1) Backend/model setup (copied from BaseValidator, but without check_cls_dataset)
            callbacks.add_integration_callbacks(self)
            backend = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device, batch=self.args.batch) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                fp16=self.args.half,
            )
            self.device = backend.device
            self.args.half = backend.fp16
            imgsz = check_imgsz(self.args.imgsz, stride=backend.stride)

            # 2) Data meta and dataloader without calling check_cls_dataset()
            # self._data_meta must contain {"nc","channels","names"}
            if not self._data_meta:
                raise ValueError(
                    "data_meta must be provided when using NPZ mode. "
                    "Pass data_meta={'nc': ..., 'channels': ..., 'names': ...} to validator constructor."
                )

            self.data = {
                **self._data_meta,
                self.args.split: "__npz__",
                "path": "__npz__"
            }

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0

            self.stride = backend.stride
            self.dataloader = self.dataloader or self.get_dataloader("__npz__", self.args.batch)

            backend.eval()
            model = backend  # for the rest of the validation loop

            # Warmup
            model.warmup(imgsz=(1 if backend.pt else self.args.batch,
                                self.data["channels"], imgsz, imgsz))

            # 3) Run the same loop as base __call__()
            self.run_callbacks("on_val_start")
            dt = (
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
                Profile(device=self.device),
            )
            bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
            self.init_metrics(model)  # sets self.names, self.nc, clears buffers
            self.jdict = []

            for batch_i, batch in enumerate(bar):
                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i

                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch)

                # Inference
                with dt[1]:
                    preds = model(batch["img"])

                # Loss (skip in val-only path)
                with dt[2]:
                    pass

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)

                if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                    self.plot_val_samples(batch, batch_i)
                    self.plot_predictions(batch, preds, batch_i)

                self.run_callbacks("on_val_batch_end")

            # Gather and finalize stats
            stats = {}
            self.gather_stats()
            if RANK in {-1, 0}:
                stats = self.get_stats()
                self.speed = dict(zip(
                    self.speed.keys(),
                    (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)
                ))
                self.finalize_metrics()
                self.print_results()
                self.run_callbacks("on_val_end")

            logger.info("Validation completed successfully in NPZ mode")
            return stats

        # Default behavior for folder/YAML datasets
        logger.info("Running validation in standard mode")
        return super().__call__(trainer=trainer, model=model)

    def update_metrics(self, preds, batch):
        """Accumulate predictions and targets for AUC computation."""
        # Store predictions (probabilities) and targets
        if isinstance(preds, torch.Tensor):
            probs = torch.softmax(preds, dim=1).cpu().numpy()
        else:
            probs = preds

        targets = batch["cls"].cpu().numpy()

        self.all_targets.append(targets)
        self.all_probs.append(probs)

        # Call parent to update standard metrics
        return super().update_metrics(preds, batch)

    def finalize_metrics(self, *args, **kwargs):
        """Compute and save per-class AUC metrics after validation."""
        results = super().finalize_metrics(*args, **kwargs)

        # Compute per-class AUC if we have predictions
        if self.all_targets and self.all_probs:
            try:
                all_targets = np.concatenate(self.all_targets, axis=0)
                all_probs = np.concatenate(self.all_probs, axis=0)

                num_classes = all_probs.shape[1]
                per_class_auc = {}

                # Compute AUC for each class (one-vs-rest)
                for class_idx in range(num_classes):
                    # Binary labels: 1 if this class, 0 otherwise
                    binary_targets = (all_targets == class_idx).astype(int)
                    class_probs = all_probs[:, class_idx]

                    # Only compute AUC if we have both positive and negative samples
                    if len(np.unique(binary_targets)) > 1:
                        auc = roc_auc_score(binary_targets, class_probs)
                        per_class_auc[f"class_{class_idx}"] = float(auc)
                    else:
                        per_class_auc[f"class_{class_idx}"] = None
                        logger.warning(f"Class {class_idx} has only one unique value in targets, cannot compute AUC")

                # Compute macro-average AUC (excluding None values)
                valid_aucs = [v for v in per_class_auc.values() if v is not None]
                if valid_aucs:
                    per_class_auc["macro_avg_auc"] = float(np.mean(valid_aucs))
                else:
                    per_class_auc["macro_avg_auc"] = None

                # Save metrics to file
                save_dir = Path(self.save_dir) if hasattr(self, 'save_dir') else Path('.')
                metrics_file = save_dir / "per_class_auc_metrics.json"

                with open(metrics_file, 'w') as f:
                    json.dump(per_class_auc, f, indent=2)

                # Check if we should print detailed per-class metrics (every 10 epochs)
                current_epoch = getattr(self.trainer, 'epoch', -1) if hasattr(self, 'trainer') else -1
                print_detailed = (current_epoch + 1) % 10 == 0 or current_epoch == -1

                if print_detailed:
                    logger.info("="*80)
                    logger.info(f"Per-class ROC-AUC (Epoch {current_epoch + 1})")
                    logger.info("="*80)
                    logger.info(f"Macro-average AUC: {per_class_auc.get('macro_avg_auc', 'N/A'):.4f}"
                               if per_class_auc.get('macro_avg_auc') is not None
                               else f"Macro-average AUC: N/A")
                    logger.info("-"*80)

                    # Print individual class AUCs in a formatted table
                    for class_idx in range(num_classes):
                        auc_val = per_class_auc.get(f"class_{class_idx}")
                        if auc_val is not None:
                            logger.info(f"  Class {class_idx:2d} AUC: {auc_val:.4f}")
                        else:
                            logger.info(f"  Class {class_idx:2d} AUC: N/A")
                    logger.info("="*80)
                else:
                    # For non-milestone epochs, just log macro-average
                    logger.info(f"Macro-average AUC: {per_class_auc.get('macro_avg_auc', 'N/A'):.4f}"
                               if per_class_auc.get('macro_avg_auc') is not None
                               else f"Macro-average AUC: N/A")

            except Exception as e:
                logger.error(f"Failed to compute per-class AUC: {e}")

        return results
