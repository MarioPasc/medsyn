from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import logging
import numpy as np
import torch
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.cfg import get_cfg  # used only to discover valid keys
from medsyn.models.classifier.utils import select_indices_by_training_images

LOG = logging.getLogger(__name__)


def _split_overrides_for_yolo(overrides_in: Dict) -> tuple[Dict, Dict]:
    """
    Return (yolo_overrides, custom_overrides).
    YOLO set = keys present in Ultralytics default cfg for task=classify.
    """
    # Discover allowed keys from Ultralytics defaults
    default_ns = get_cfg(None, {"task": "classify"})  # merged defaults
    allowed = set(vars(default_ns).keys())

    # Some keys look CLI-like and can cause rejections; drop them proactively
    forbidden = {"mode"}  # 'mode' is positional in CLI, not a runtime arg

    yolo, custom = {}, {}
    for k, v in (overrides_in or {}).items():
        if k in forbidden:
            custom[k] = v
        elif k in allowed:
            yolo[k] = v
        else:
            custom[k] = v
    # Ensure required sentinel keys exist
    yolo.setdefault("task", "classify")
    yolo.setdefault("data", "__npz__")
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
        self.medsyn_npz_path = str(npz_path) if npz_path is not None else None
        self.medsyn_training_images = training_images

        # Split overrides: pass only YOLO-known keys upstream
        yolo_overrides, self.medsyn_cfg = _split_overrides_for_yolo(overrides)

        super().__init__(cfg, yolo_overrides, _callbacks)

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
        Build the NPZ-backed DataLoader. Ultralytics still calls this with a dataset_path,
        which we ignore because our data comes from NPZ.
        """
        from medsyn.models.classifier.dataloaders import build_npz_loader
        return build_npz_loader(
            npz_path=self.medsyn_npz_path,
            split={"train": "train", "val": "val", "test": "test"}[mode],
            imgsz=self.args.imgsz,
            batch=batch_size,
            workers=self.args.workers,
            training_images=self.medsyn_training_images,
            augment=(mode == "train"),
        )

    def set_model_attributes(self) -> None:
        """
        Keep parent behavior; it sets model.names from self.data['names'].
        """
        super().set_model_attributes()

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
