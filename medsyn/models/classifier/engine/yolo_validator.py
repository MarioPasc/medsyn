from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
import torch
from ultralytics.models.yolo.classify.val import ClassificationValidator
from sklearn.metrics import roc_auc_score
from medsyn.models.classifier.dataloaders import build_npz_loader

logger = logging.getLogger(__name__)

class MedsynClassificationValidator(ClassificationValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_targets = []
        self.all_probs = []

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="val"):
        args = self.args or self.trainer.args
        return build_npz_loader(
            npz_path=args.medsyn_npz_path,
            split={"train":"train","val":"val","test":"test"}[mode],
            imgsz=args.imgsz,
            batch=batch_size,
            workers=args.workers,
            training_images=args.medsyn_training_images,
            augment=False,
        )

    def __call__(self, trainer=None, model=None):
        """Override to reset accumulators before validation."""
        self.all_targets = []
        self.all_probs = []
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
