from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Ultralytics classify transforms with ImageNet mean/std
from ultralytics.data.augment import classify_transforms, DEFAULT_MEAN, DEFAULT_STD  # docs reference covers defaults

from .utils import select_indices_by_training_images

Split = Literal["train","val","test"]

def _to_pil_rgb(x: np.ndarray) -> Image.Image:
    """
    Convert HxW or HxWxC numpy array in [0,255] or [0,1] to PIL RGB.
    Grayscale will be stacked to 3 channels for compatibility with YOLO backbones.
    """
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x, 0, 255)
        x = (x * 255).astype(np.uint8) if x.max() <= 1 else x.astype(np.uint8)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return Image.fromarray(x[:, :, :3], mode="RGB")

@dataclass
class NpzClassificationDataset(Dataset):
    npz_path: Path
    split: Split
    imgsz: int
    training_images: str  # PathMNIST | PathMNIST_and_synth | synth
    augment: bool = False
    # Optional fold data for k-fold CV (overrides NPZ loading if provided)
    fold_images: Optional[np.ndarray] = None
    fold_labels: Optional[np.ndarray] = None
    fold_is_synth: Optional[np.ndarray] = None

    def __post_init__(self):
        # Use fold data if provided (k-fold CV mode), otherwise load from NPZ
        if self.fold_images is not None:
            X = self.fold_images
            y = self.fold_labels.reshape(-1).astype(np.int64)
            syn = self.fold_is_synth if self.fold_is_synth is not None else np.zeros_like(y, dtype=np.uint8)
        else:
            z = np.load(self.npz_path)
            X = z[f"{self.split}_images"]
            y = z[f"{self.split}_labels"].reshape(-1).astype(np.int64)
            syn = z.get(f"{self.split}_is_synth", np.zeros_like(y, dtype=np.uint8))

        # Apply your selection mask first
        keep = select_indices_by_training_images(syn, self.training_images)
        X = X[keep]
        y = y[keep]

        # Reindex to contiguous 0..K-1 to satisfy CrossEntropyLoss
        unique_ids = np.unique(y)
        id2new = {oid: i for i, oid in enumerate(unique_ids.tolist())}
        y = np.vectorize(id2new.__getitem__)(y).astype(np.int64)
        self.num_classes = len(unique_ids)   # for optional checks

        # Basic sanity checks
        if y.min() < 0:
            raise ValueError(f"Negative labels found after mapping: min={y.min()}")
        if y.max() >= self.num_classes:
            raise ValueError(f"Label out of range after mapping: max={y.max()} >= {self.num_classes}")

        self.X, self.y = X, y
        self.tx = classify_transforms(size=self.imgsz, mean=DEFAULT_MEAN, std=DEFAULT_STD)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        pil = _to_pil_rgb(self.X[i])
        timg = self.tx(pil)  # torch.float32 [C,H,W], normalized
        return {"img": timg, "cls": torch.tensor(self.y[i], dtype=torch.long)}

def build_npz_loader(
    npz_path: str | Path, split: Split, imgsz: int, batch: int, workers: int, training_images: str, augment: bool
) -> DataLoader:
    ds = NpzClassificationDataset(
        npz_path=Path(npz_path),
        split=split,
        imgsz=imgsz,
        training_images=training_images,
        augment=augment,
    )
    return DataLoader(ds, batch_size=batch, shuffle=(split=="train"), num_workers=workers, pin_memory=True)
