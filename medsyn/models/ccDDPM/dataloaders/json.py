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
import os

logger = logging.getLogger(__name__)

# Debug logging control - set via environment variable
# Usage: export MEDSYN_DEBUG_DATALOADER=1
DEBUG_DATALOADER = os.getenv("MEDSYN_DEBUG_DATALOADER", "0") == "1"

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

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that handles variable-length lists in batch.

    CRITICAL FIX: PyTorch's default collate_fn cannot handle dictionaries containing
    variable-length lists (like "applied_transforms"). This custom collate function:
    1. Stacks tensors normally (pixel_values, labels)
    2. Keeps variable-length lists as list-of-lists (applied_transforms)
    3. Collects scalar/string values (is_synth, path)

    This is essential for multi-worker DataLoader with augmentation enabled.

    Args:
        batch: List of dictionaries from dataset __getitem__

    Returns:
        Collated batch dictionary
    """
    if DEBUG_DATALOADER:
        logger.debug(f"[COLLATE] Collating batch of size {len(batch)}")
        for i, sample in enumerate(batch):
            logger.debug(f"  Sample {i}: pixel_values.shape={sample['pixel_values'].shape}, "
                        f"labels={sample['labels'].item()}, "
                        f"applied_transforms={sample.get('applied_transforms', [])}")

    # Separate handling for different field types
    collated = {}

    # Stack tensors: pixel_values, labels
    collated["pixel_values"] = torch.stack([item["pixel_values"] for item in batch])
    collated["labels"] = torch.stack([item["labels"] for item in batch])

    # Keep lists as list-of-lists: applied_transforms
    # This cannot be stacked because each sample may have different number of transforms
    collated["applied_transforms"] = [item.get("applied_transforms", []) for item in batch]

    # Collect scalar/string values
    collated["is_synth"] = [item["is_synth"] for item in batch]
    collated["path"] = [item["path"] for item in batch]

    if DEBUG_DATALOADER:
        logger.debug(f"[COLLATE] Result: pixel_values.shape={collated['pixel_values'].shape}, "
                    f"labels.shape={collated['labels'].shape}, "
                    f"num_samples={len(collated['applied_transforms'])}")

    return collated


def build_json_loader(
    index_json: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
    augmentation_pipeline: Optional[Any] = None,
    sampler: Optional[Any] = None
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
        sampler: Optional sampler (e.g., DistributedSampler for DDP). If provided, shuffle is disabled.

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

    # If sampler is provided, disable shuffle (samplers handle shuffling)
    shuffle = (split == "train") and (sampler is None)

    if DEBUG_DATALOADER:
        logger.debug(f"[BUILD_LOADER] Creating DataLoader: split={split}, batch_size={batch_size}, "
                    f"num_workers={num_workers}, shuffle={shuffle}, sampler={sampler is not None}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),  # Improves performance when using workers
        collate_fn=custom_collate_fn  # CRITICAL FIX: Use custom collate for variable-length lists
    )
