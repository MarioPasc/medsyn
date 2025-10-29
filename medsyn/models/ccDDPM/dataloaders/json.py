# medsyn/models/ccDDPM/dataloader.py
# Purpose: Dataset from the JSON index produced by medsyn/cli/data.py
# Output tensors: pixel_values in [-1,1], labels as int64, and metadata.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import json
import logging
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

logger = logging.getLogger(__name__)

@dataclass
class Sample:
    path: Path
    label: int
    is_synth: bool

class PathMNISTIndexDataset(Dataset):
    """
    Dataset backed by a MedSyn JSON index:
      { "PathMNIST": { "train": { "0": {"image": "...", "label": int, "is_synth": bool}, ... } } }
    """
    def __init__(
        self,
        index_json: Path,
        split: str = "train",
        image_size: int = 128,
        normalize: bool = True,
        augmentation_pipeline: Optional[Any] = None
    ):
        with open(index_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)["PathMNIST"][split]
        self.items: List[Sample] = []
        for _, rec in data.items():
            self.items.append(Sample(Path(rec["image"]).resolve(), int(rec["label"]), bool(rec.get("is_synth", False))))
        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) if normalize else (lambda x: x),
        ])
        self.augmentation_pipeline = augmentation_pipeline if split == "train" else None
        logger.info("IndexDataset split=%s size=%d", split, len(self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.items[i]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)

        # Apply augmentation (only for training split)
        applied_transforms = []
        if self.augmentation_pipeline is not None:
            x, applied_transforms = self.augmentation_pipeline(x, return_applied_transforms=True)

        return {
            "pixel_values": x,
            "labels": torch.tensor(s.label, dtype=torch.long),
            "path": str(s.path),
            "is_synth": s.is_synth,
            "applied_transforms": applied_transforms
        }

def build_json_loader(
    index_json: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
    augmentation_pipeline: Optional[Any] = None
) -> DataLoader:
    """
    Build torch DataLoader from JSON index with sane defaults.

    Args:
        index_json: Path to JSON index file
        split: 'train', 'val', or 'test'
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize to [-1, 1]
        augmentation_pipeline: Optional augmentation pipeline (only applied to training split)

    Returns:
        DataLoader instance
    """
    ds = PathMNISTIndexDataset(
        index_json,
        split=split,
        image_size=image_size,
        normalize=normalize,
        augmentation_pipeline=augmentation_pipeline
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")
    )
