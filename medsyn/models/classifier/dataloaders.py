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

def _to_pil_rgb(img: np.ndarray) -> Image.Image:
    # Accept HWC or CHW, 1 or 3 channels; output RGB PIL
    if img.ndim != 3:
        raise ValueError(f"Expected 3D image, got {img.shape}")
    if img.shape[0] in (1, 3) and img.shape[0] <= img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))  # CHW->HWC
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return Image.fromarray(img, mode="RGB")

@dataclass
class NpzClassificationDataset(Dataset):
    npz_path: Path
    split: Split
    imgsz: int
    training_images: str  # PathMNIST | PathMNIST_and_synth | synth
    augment: bool = False

    def __post_init__(self):
        z = np.load(self.npz_path)
        X = z[f"{self.split}_images"]
        y = z[f"{self.split}_labels"].reshape(-1).astype(np.int64)
        syn = z.get(f"{self.split}_is_synth", np.zeros_like(y, dtype=np.uint8))
        keep = select_indices_by_training_images(syn, self.training_images)
        self.X = X[keep]
        self.y = y[keep]
        # Ultralytics transforms for classify
        self.tx = classify_transforms(size=self.imgsz, mean=DEFAULT_MEAN, std=DEFAULT_STD, hflip=0.5 if self.augment else 0.0)

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
