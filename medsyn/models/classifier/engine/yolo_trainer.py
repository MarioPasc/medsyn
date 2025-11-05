from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import logging
import numpy as np
import torch
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from medsyn.models.classifier.utils import select_indices_by_training_images

LOG = logging.getLogger(__name__)


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
    Ultralytics trainer that consumes NPZ via a custom DataLoader.
    It bypasses the default path-based dataset checks.
    """

    def get_dataset(self) -> dict:
        """
        Return dict with required keys plus split placeholders so Ultralytics' base loop works.
        """
        npz_path = getattr(self, "medsyn_npz_path", None)
        training_images = getattr(self, "medsyn_training_images", "PathMNIST")
        nc, channels = _infer_meta_from_train(npz_path, training_images)

        names = {i: f"class_{i}" for i in range(nc)}  # can be overridden by args.names if provided and length==nc
        try:
            if isinstance(self.args.names, (list, tuple)) and len(self.args.names) == nc:
                names = {i: str(self.args.names[i]) for i in range(nc)}
        except AttributeError:
            pass

        sentinel = "__npz__"
        return {"nc": nc, "channels": channels, "names": names,
                "train": sentinel, "val": sentinel, "test": sentinel, "path": sentinel}


    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Build the NPZ-backed DataLoader. Ultralytics still calls this with a dataset_path,
        which we ignore because our data comes from NPZ.
        """
        from medsyn.models.classifier.dataloaders import build_npz_loader
        return build_npz_loader(
            npz_path=getattr(self, "medsyn_npz_path", None),
            split={"train": "train", "val": "val", "test": "test"}[mode],
            imgsz=self.args.imgsz,
            batch=batch_size,
            workers=self.args.workers,
            training_images=getattr(self, "medsyn_training_images", "PathMNIST"),
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
        # batch["cls"] must be Long in [0, nc-1]
        y = batch["cls"]
        if y.dtype != torch.long:
            raise TypeError(f"Targets dtype must be torch.long, got {y.dtype}")
        nc = self.data["nc"]
        if (y < 0).any() or (y >= nc).any():
            bad = y[(y < 0) | (y >= nc)]
            raise ValueError(f"Found out-of-range labels in batch: min={int(y.min())}, max={int(y.max())}, nc={nc}; bad={bad[:8].tolist()}")
        return super().train_step(batch)
