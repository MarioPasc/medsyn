from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from medsyn.models.classifier.dataloaders import build_npz_loader

LOG = logging.getLogger(__name__)


def _infer_npz_meta(npz_path: str | Path) -> tuple[int, int, dict[int, str]]:
    """
    Infer dataset meta from an NPZ.
    Returns (nc, channels, names).
    - nc: max(label)+1 if labels present, else 1
    - channels: last-dim of images if 4D, else 1 for (N,H,W)
    - names: {i: f"class_{i}"} if not found elsewhere
    """
    nc, channels = 1, 3
    names: dict[int, str] = {}
    try:
        z = np.load(npz_path) if npz_path else None
        if z is not None:
            # infer classes
            for k in ("train_labels", "labels", "y_train"):
                if k in z:
                    y = z[k].reshape(-1)
                    nc = int(y.max()) + 1
                    break
            # infer channels
            for k in ("train_images", "images", "x_train"):
                if k in z:
                    x = z[k]
                    if x.ndim == 4:
                        channels = int(x.shape[-1])
                    elif x.ndim == 3:
                        channels = 1
                    break
    except Exception as e:
        LOG.warning("NPZ meta inference failed: %s", e)
    if not names:
        names = {i: f"class_{i}" for i in range(nc)}
    return nc, channels, names


class MedsynClassificationTrainer(ClassificationTrainer):
    """Ultralytics trainer that consumes an NPZ via a custom DataLoader."""

    def setup_model(self):
        """
        Ensure self.data is a dict with required keys BEFORE parent setup.
        Parent will call get_model() which uses self.data['nc'] and ['channels'],
        then reshape outputs and set model.names from self.data['names'].
        """
        npz_path = getattr(self, "medsyn_npz_path", None)
        training_images = getattr(self, "medsyn_training_images", "PathMNIST")
        nc, channels, names = _infer_npz_meta(npz_path)

        # Optional: override names if you pass them in overrides as a list
        try:
            if isinstance(self.args.names, (list, tuple)) and len(self.args.names) == nc:
                names = {i: str(self.args.names[i]) for i in range(nc)}
        except AttributeError:
            pass

        # Critical: self.data MUST be a dict with these keys
        self.data = {"nc": nc, "channels": channels, "names": names, "training_images": training_images}

        # Proceed with normal model setup (will read self.data)
        return super().setup_model()

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Return your NPZ-backed DataLoader. Unchanged from your previous version,
        but now it can also read self.data['channels'] if you need augment rules.
        """
        return build_npz_loader(
            npz_path=getattr(self, "medsyn_npz_path", None),
            split={"train": "train", "val": "val", "test": "test"}[mode],
            imgsz=self.args.imgsz,
            batch=batch_size,
            workers=self.args.workers,
            training_images=getattr(self, "medsyn_training_images", "PathMNIST"),
            augment=(mode == "train"),
        )

    def set_model_attributes(self):
        """Keep parent behavior: will use self.data['names']."""
        super().set_model_attributes()
