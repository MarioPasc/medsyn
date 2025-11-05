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
        """Override to provide a dummy dataset since we use NPZ dataloaders."""
        # Extract number of classes from NPZ file if available
        nc = 9  # Default to 9 classes (PathMNIST)

        # Check if medsyn_npz_path is available (it might not be set yet during init)
        if hasattr(self.args, 'medsyn_npz_path'):
            npz_path = Path(self.args.medsyn_npz_path)
            if npz_path.exists():
                try:
                    z = np.load(npz_path)
                    # Get number of unique classes from training labels
                    train_labels = z["train_labels"].reshape(-1)
                    nc = int(np.max(train_labels)) + 1
                except Exception as e:
                    print(f"Warning: Could not load NPZ file for class count: {e}")

        return DummyDataset(nc=nc)

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        # Not used; training/validation will call our get_dataloader
        return super().build_dataset(img_path, mode, batch)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        args = self.args
        return build_npz_loader(
            npz_path=args.medsyn_npz_path,
            split={"train":"train","val":"val","test":"test"}[mode],
            imgsz=args.imgsz,
            batch=batch_size,
            workers=args.workers,
            training_images=args.medsyn_training_images,
            augment=(mode=="train"),
        )

    def set_model_attributes(self):
        # Let parent set class names from dataset if available
        super().set_model_attributes()
        # Attach transforms to model for export/val
        if RANK in (-1, 0):
            # Get the underlying model (unwrap DDP/DP if needed)
            m = self.model.module if hasattr(self.model, 'module') else self.model
            m.args = self.args
