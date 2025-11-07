from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import logging
import numpy as np
import torch
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.cfg import get_cfg, DEFAULT_CFG_DICT  # used to discover valid keys
from medsyn.models.classifier.utils import select_indices_by_training_images

LOG = logging.getLogger(__name__)


def _split_overrides_for_yolo(overrides_in: Dict) -> tuple[Dict, Dict]:
    """
    Return (yolo_overrides, custom_overrides).
    YOLO set = keys present in Ultralytics default cfg for task=classify.
    """
    LOG.info("Splitting overrides for YOLO compatibility...")
    LOG.debug(f"Input overrides keys: {list(overrides_in.keys()) if overrides_in else []}")
    
    # Discover allowed keys from Ultralytics defaults
    # get_cfg expects cfg (can be None) and overrides (dict with task)
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT
        # Use the default config directly
        allowed = set(DEFAULT_CFG_DICT.keys())
        LOG.info(f"Found {len(allowed)} allowed YOLO keys from DEFAULT_CFG_DICT")
    except ImportError:
        # Fallback: try get_cfg with proper arguments
        try:
            default_cfg = get_cfg(cfg=DEFAULT_CFG_DICT.copy(), overrides={"task": "classify"})
            allowed = set(vars(default_cfg).keys()) if default_cfg else set()
            LOG.info(f"Found {len(allowed)} allowed YOLO keys from get_cfg")
        except Exception as e:
            LOG.warning(f"Could not get default cfg from Ultralytics: {e}")
            # Hardcoded fallback of common YOLO classify keys
            allowed = {
                'task', 'mode', 'model', 'data', 'epochs', 'time', 'patience', 'batch', 
                'imgsz', 'save', 'save_period', 'cache', 'device', 'workers', 'project', 
                'name', 'exist_ok', 'pretrained', 'optimizer', 'verbose', 'seed', 'deterministic',
                'single_cls', 'rect', 'cos_lr', 'close_mosaic', 'resume', 'amp', 'fraction',
                'profile', 'freeze', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
                'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl', 'pose', 'kobj',
                'label_smoothing', 'nbs', 'overlap_mask', 'mask_ratio', 'dropout', 'val',
                'split', 'save_json', 'save_hybrid', 'conf', 'iou', 'max_det', 'half',
                'dnn', 'plots', 'source', 'vid_stride', 'stream_buffer', 'visualize',
                'augment', 'agnostic_nms', 'classes', 'retina_masks', 'embed', 'show',
                'save_frames', 'save_txt', 'save_conf', 'save_crop', 'show_labels',
                'show_conf', 'show_boxes', 'line_width', 'format', 'keras', 'optimize',
                'int8', 'dynamic', 'simplify', 'opset', 'workspace', 'nms', 'lr0',
                'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum',
                'warmup_bias_lr', 'box', 'cls', 'dfl', 'pose', 'kobj', 'label_smoothing',
                'nbs', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
                'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
                'auto_augment', 'erasing', 'crop_fraction', 'names'
            }
            LOG.info(f"Using hardcoded fallback with {len(allowed)} keys")

    # Some keys look CLI-like and can cause rejections; drop them proactively
    # 'mode' is handled by CLI, not needed in overrides during init
    forbidden = {"mode"}

    yolo, custom = {}, {}
    for k, v in (overrides_in or {}).items():
        if k in forbidden:
            custom[k] = v
            LOG.debug(f"  {k}: forbidden -> custom")
        elif k in allowed:
            yolo[k] = v
            LOG.debug(f"  {k}: allowed -> yolo")
        else:
            custom[k] = v
            LOG.debug(f"  {k}: unknown -> custom")
    
    # Ensure required sentinel keys exist
    yolo.setdefault("task", "classify")
    yolo.setdefault("data", "__npz__")
    
    LOG.info(f"Split complete: {len(yolo)} YOLO keys, {len(custom)} custom keys")
    LOG.info(f"YOLO keys: {list(yolo.keys())}")
    LOG.info(f"Custom keys: {list(custom.keys())}")
    
    return yolo, custom


def _infer_meta_from_train(npz_path: str | Path, training_images: str) -> Tuple[int, int]:
    """
    Compute (nc, channels) from the TRAIN split using the same filtering.
    """
    z = np.load(npz_path)
    y = z["train_labels"].reshape(-1).astype(np.int64)
    syn = z.get("train_is_synth", np.zeros_like(y, dtype=np.uint8))
    keep = select_indices_by_training_images(syn, training_images)
    y = y[keep]
    # Reindex like the dataset would do
    unique_ids = np.unique(y)
    nc = int(unique_ids.size)

    X = z["train_images"][keep]
    channels = X.shape[-1] if X.ndim == 4 else 1
    if channels == 1:
        channels = 3  # we stack to RGB in the dataset
    return nc, channels


class MedsynClassificationTrainer(ClassificationTrainer):
    """
    Ultralytics trainer that consumes an NPZ via a custom DataLoader.
    We pass NPZ parameters at construction so they exist during BaseTrainer.__init__.
    """

    # Accept NPZ args BEFORE calling super().__init__()
    def __init__(self, cfg=None, overrides: dict | None = None, _callbacks=None,
                 npz_path: str | Path | None = None, training_images: str = "PathMNIST"):
        LOG.info("=" * 80)
        LOG.info("Initializing MedsynClassificationTrainer")
        LOG.info(f"  NPZ path: {npz_path}")
        LOG.info(f"  Training images mode: {training_images}")
        LOG.info(f"  Config (cfg) type: {type(cfg)}")
        LOG.info(f"  Overrides keys: {list(overrides.keys()) if overrides else 'None'}")
        
        self.medsyn_npz_path = str(npz_path) if npz_path is not None else None
        self.medsyn_training_images = training_images

        # Split overrides: pass only YOLO-known keys upstream
        yolo_overrides, self.medsyn_cfg = _split_overrides_for_yolo(overrides)

        # Ultralytics expects a dict-like cfg. Supply defaults if None.
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
            base_cfg = DEFAULT_CFG_DICT.copy()
        except Exception:
            base_cfg = {}
        # Minimal fields to satisfy cfg checks and make behavior explicit
        base_cfg.update({"task": "classify", "mode": "train", "save_dir": ""})

        LOG.info("Calling parent ClassificationTrainer.__init__ with a dict cfg...")
        super().__init__(base_cfg, yolo_overrides, _callbacks)
        LOG.info("MedsynClassificationTrainer initialized successfully")
        LOG.info("=" * 80)

    def get_dataset(self) -> dict:
        """
        Return dict with required keys plus split placeholders so Ultralytics' base loop works.
        """
        if not self.medsyn_npz_path:
            raise ValueError("NPZ path is not set; pass npz_path=... to MedsynClassificationTrainer(...)")
        nc, channels = _infer_meta_from_train(self.medsyn_npz_path, self.medsyn_training_images)

        names = {i: f"class_{i}" for i in range(nc)}
        try:
            if isinstance(self.args.names, (list, tuple)) and len(self.args.names) == nc:
                names = {i: str(self.args.names[i]) for i in range(nc)}
        except AttributeError:
            pass

        s = "__npz__"
        return {"nc": nc, "channels": channels, "names": names, "train": s, "val": s, "test": s, "path": s}


    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Return an Ultralytics InfiniteDataLoader so BaseTrainer can call .reset().
        We ignore dataset_path and build directly from the NPZ.
        """
        from ultralytics.data.build import build_dataloader  # returns InfiniteDataLoader
        from medsyn.models.classifier.dataloaders import NpzClassificationDataset

        ds = NpzClassificationDataset(
            npz_path=Path(self.medsyn_npz_path),
            split={"train": "train", "val": "val", "test": "test"}[mode],
            imgsz=int(self.args.imgsz),
            training_images=self.medsyn_training_images,
            augment=(mode == "train"),
        )
        # build_dataloader provides InfiniteDataLoader with .reset(), proper seeding, and worker reuse
        return build_dataloader(
            dataset=ds,
            batch=batch_size,
            workers=int(self.args.workers),
            shuffle=(mode == "train"),
            rank=rank,
            pin_memory=True,
        )

    def set_model_attributes(self) -> None:
        """
        Keep parent behavior; it sets model.names from self.data['names'].
        """
        super().set_model_attributes()

    def get_validator(self):
        """
        Override to always return our NPZ-aware validator with proper metadata.
        This ensures the validator works correctly even in final_eval() and val_only modes.
        """
        from copy import copy
        from medsyn.models.classifier.engine.yolo_validator import MedsynClassificationValidator

        LOG.info("Creating MedsynClassificationValidator with NPZ metadata")

        # Create validator with data metadata to bypass check_cls_dataset()
        v = MedsynClassificationValidator(
            dataloader=None,  # Will be created on-demand via get_dataloader
            save_dir=self.save_dir,
            args=copy(self.args),
            data_meta={
                "nc": self.data["nc"],
                "channels": self.data["channels"],
                "names": self.data["names"]
            }
        )
        v.medsyn_npz_path = self.medsyn_npz_path
        v.medsyn_training_images = self.medsyn_training_images

        LOG.info(f"Validator configured: nc={self.data['nc']}, channels={self.data['channels']}")
        return v

    def train_step(self, batch):
        """
        Runtime guard to check batch labels before training step.
        """
        y = batch["cls"]
        if y.dtype != torch.long:
            raise TypeError(f"Targets dtype must be torch.long, got {y.dtype}")
        nc = self.data["nc"]
        if (y < 0).any() or (y >= nc).any():
            bad = y[(y < 0) | (y >= nc)]
            raise ValueError(f"Out-of-range labels in batch: min={int(y.min())}, max={int(y.max())}, nc={nc}; sample={bad[:8].tolist()}")
        return super().train_step(batch)
