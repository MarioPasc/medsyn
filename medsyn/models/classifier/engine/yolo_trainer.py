from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import logging
import numpy as np
from ultralytics.models.yolo.classify.train import ClassificationTrainer

LOG = logging.getLogger(__name__)


def _infer_npz_meta(npz_path: str | Path | None) -> Tuple[int, int, Dict[int, str]]:
    """
    Infer (nc, channels, names) from an NPZ file.
    nc: max(label)+1 if labels present, else 1
    channels: last dim of images if 4D, else 1 for (N,H,W)
    names: {i: f"class_{i}"} by default
    """
    nc, channels = 1, 3
    names: Dict[int, str] = {}
    try:
        if npz_path:
            z = np.load(npz_path)
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
                    channels = int(x.shape[-1]) if x.ndim == 4 else 1
                    break
    except Exception as e:
        LOG.warning("NPZ meta inference failed: %s", e)
    if not names:
        names = {i: f"class_{i}" for i in range(nc)}
    return nc, channels, names


class MedsynClassificationTrainer(ClassificationTrainer):
    """
    Ultralytics trainer that consumes NPZ via a custom DataLoader.
    It bypasses the default path-based dataset checks.
    """

    def get_dataset(self) -> dict:
        """
        Return the minimal dataset dict Ultralytics expects for classification, plus
        split keys so BaseTrainer._setup_train can index ['train'/'val'/'test'].

        We pass a sentinel string for each split because our custom dataloader
        reads from NPZ and ignores the path argument.
        """
        npz_path = getattr(self, "medsyn_npz_path", None)
        nc, channels, names = _infer_npz_meta(npz_path)

        # Optional override from YAML if provided and length matches
        try:
            if isinstance(self.args.names, (list, tuple)) and len(self.args.names) == nc:
                names = {i: str(self.args.names[i]) for i in range(nc)}
        except AttributeError:
            pass

        sentinel = "__npz__"  # harmless placeholder for logs and validators
        return {
            "nc": nc,
            "channels": channels,
            "names": names,
            "train": sentinel,
            "val": sentinel,
            "test": sentinel,
            "path": sentinel,  # some tools log/pretty-print this
        }


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
