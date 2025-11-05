from __future__ import annotations
import numpy as np
from pathlib import Path
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.utils import RANK
from medsyn.models.classifier.dataloaders import build_npz_loader


class DummyDataset:
    """Dummy dataset to satisfy YOLO's initialization requirements."""
    def __init__(self, nc: int = 9):
        self.nc = nc  # number of classes
        self.names = {i: f"class_{i}" for i in range(nc)}  # class names


class MedsynClassificationTrainer(ClassificationTrainer):
    """Ultralytics trainer that feeds from NPZ instead of folders."""

    def get_dataset(self):
        """Provide a minimal dataset object; infer nc from NPZ if present."""
        nc = 9
        npz_str = getattr(self, "medsyn_npz_path", "")
        npz_path = Path(npz_str) if npz_str else None
        if npz_path and npz_path.exists():
            try:
                z = np.load(npz_path)
                train_labels = z["train_labels"].reshape(-1)
                nc = int(np.max(train_labels)) + 1
            except Exception as e:
                print(f"Warning: Could not load NPZ for class count: {e}")
        return DummyDataset(nc=nc)

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        # Not used; training/validation will call our get_dataloader
        return super().build_dataset(img_path, mode, batch)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        imgsz = self.args.imgsz
        workers = self.args.workers
        npz_path = getattr(self, "medsyn_npz_path", None)
        training_images = getattr(self, "medsyn_training_images", "PathMNIST")
        return build_npz_loader(
            npz_path=npz_path,
            split={"train": "train", "val": "val", "test": "test"}[mode],
            imgsz=imgsz,
            batch=batch_size,
            workers=workers,
            training_images=training_images,
            augment=(mode == "train"),
        )

    def set_model_attributes(self):
        # Let parent set class names from dataset if available
        super().set_model_attributes()
        # Attach transforms to model for export/val
        if RANK in (-1, 0):
            # Get the underlying model (unwrap DDP/DP if needed)
            m = self.model.module if hasattr(self.model, 'module') else self.model
            m.args = self.args
