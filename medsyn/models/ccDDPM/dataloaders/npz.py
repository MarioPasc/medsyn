# medsyn/models/ccDDPM/dataloaders/npz.py
# Purpose: NPZ-based dataloader for compressed datasets on supercomputers
# Loads all data from a single .npz file containing images and metadata
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os

logger = logging.getLogger(__name__)

# Debug logging control - set via environment variable
# Usage: export MEDSYN_DEBUG_DATALOADER=1
DEBUG_DATALOADER = os.getenv("MEDSYN_DEBUG_DATALOADER", "0") == "1"

# medsyn/models/ccDDPM/data/npz_dataset.py

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

logger = logging.getLogger(__name__)


class NPZDataset(Dataset):
    """
    Dataset backed by a compressed NPZ file containing:
      - Images: arrays by split (train_images, val_images, test_images)
      - Labels: arrays by split (train_labels, val_labels, test_labels)
      - Metadata: is_synth flags by split (train_is_synth, val_is_synth, test_is_synth)

    Normalization policy:
      - Images are loaded as uint8 RGB, scaled to [0, 1].
      - If normalize=True  → mapped to [-1, 1] (recommended for diffusion).
      - If an augmentation_pipeline is provided, images are always passed
        to it in [-1, 1], and its output is also in [-1, 1].

    This makes the dataset compatible with:
      - DDPM noise schedulers defined on x ∈ [-1, 1],
      - AugmentationPipeline, which expects input in [-1, 1] and internally
        converts to [0, 1] for Albumentations and back to [-1, 1].
    """

    def __init__(
        self,
        npz_path: Path,
        split: str = "train",
        image_size: int = 64,
        normalize: bool = True,
        augmentation_pipeline: Optional[Any] = None,
    ):
        """
        Args:
            npz_path: Path to the .npz file.
            split: One of "train", "val", or "test".
            image_size: Target image size (will resize if needed).
            normalize: If True, output pixel_values in [-1, 1] (recommended for DDPM).
                       If False, output pixel_values in [0, 1] and augmentation is disabled.
            augmentation_pipeline: Optional Albumentations-based pipeline that expects
                                   and returns tensors in [-1, 1].
        """
        self.split = split
        self.image_size = image_size
        self.normalize = normalize

        # If augmentation is provided, enforce normalize=True so that the
        # pipeline sees and returns images in [-1, 1], as designed.
        # See AugmentationPipeline.__call__ and _tensor_to_numpy / _numpy_to_tensor. :contentReference[oaicite:1]{index=1}
        if augmentation_pipeline is not None and not normalize:
            logger.warning(
                "Augmentation pipeline expects images in [-1, 1], but normalize=False was passed. "
                "Forcing normalize=True to maintain consistency with the augmentation pipeline."
            )
            self.normalize = True

        # Augmentation only for training split
        self.augmentation_pipeline = augmentation_pipeline if split == "train" else None

        logger.info(f"Loading NPZ dataset from {npz_path} (split={split})...")
        data = np.load(npz_path)

        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        is_synth_key = f"{split}_is_synth"

        if images_key not in data:
            raise KeyError(f"NPZ file missing key '{images_key}'. Available keys: {list(data.keys())}")

        # [N, H, W, C] uint8, 0..255
        self.images = data[images_key]
        self.labels = data[labels_key]
        self.is_synth = data[is_synth_key] if is_synth_key in data else np.zeros(len(self.images), dtype=bool)

        assert self.images.ndim == 4, f"Expected 4D images array, got shape {self.images.shape}"
        assert len(self.images) == len(self.labels), "Images and labels length mismatch"

        logger.info(f"Loaded {len(self.images)} images for split '{split}'")
        logger.info(f"  Image shape: {self.images.shape}")
        logger.info(f"  Labels shape: {self.labels.shape}")
        logger.info(f"  Unique labels: {np.unique(self.labels).tolist()}")

        # Build geometric transform pipeline (resize only; keep range unchanged)
        resize_transforms = []
        if (self.images.shape[1] != image_size) or (self.images.shape[2] != image_size):
            resize_transforms.append(
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR)
            )
            logger.info(
                f"  Will resize from {self.images.shape[1]}x{self.images.shape[2]} to "
                f"{image_size}x{image_size}"
            )

        self.resize_transform = T.Compose(resize_transforms) if resize_transforms else None

        if self.normalize:
            logger.info("  Output pixel_values will be normalized to [-1, 1]")
        else:
            logger.info("  Output pixel_values will be in [0, 1] (no diffusion-style scaling)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
              - pixel_values: Tensor [C, H, W] in [-1,1] if normalize=True, else [0,1].
              - labels: LongTensor scalar.
              - is_synth: bool.
              - idx: int.
              - applied_transforms: list[str] (augmentation names; empty if none).
        """
        # 1) Load raw image [H, W, C] uint8
        img = self.images[idx]

        # 2) Convert to [C, H, W] float32 in [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 3) Resize if needed (still [0, 1])
        if self.resize_transform is not None:
            img_tensor = self.resize_transform(img_tensor)

        applied_transforms: list[str] = []

        # 4) Diffusion-style normalization to [-1, 1] if requested
        if self.normalize:
            img_tensor = img_tensor * 2.0 - 1.0  # [0,1] → [-1,1]

        # 5) Augmentation (only for training split, only if provided)
        if self.augmentation_pipeline is not None:
            # AugmentationPipeline expects [C, H, W] in [-1, 1] and internally maps
            # to [0,1] for Albumentations, then returns [-1,1]. :contentReference[oaicite:2]{index=2}
            img_tensor, applied_transforms = self.augmentation_pipeline(
                img_tensor,
                return_applied_transforms=True
            )

        # 6) If normalize=False, ensure final range is [0,1]
        #    (This branch will only happen if no augmentation_pipeline was used,
        #     because we force normalize=True when augmentation is present.)
        if not self.normalize:
            # Keep a consistent contract: when normalize=False, always return [0,1]
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        return {
            "pixel_values": img_tensor,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "is_synth": bool(self.is_synth[idx]),
            "idx": idx,
            "applied_transforms": applied_transforms,
        }



def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that handles variable-length lists in batch.

    CRITICAL FIX: PyTorch's default collate_fn cannot handle dictionaries containing
    variable-length lists (like "applied_transforms"). This custom collate function:
    1. Stacks tensors normally (pixel_values, labels)
    2. Keeps variable-length lists as list-of-lists (applied_transforms)
    3. Collects scalar values (is_synth, idx)

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

    # Collect scalar values
    collated["is_synth"] = [item["is_synth"] for item in batch]
    collated["idx"] = [item["idx"] for item in batch]

    if DEBUG_DATALOADER:
        logger.debug(f"[COLLATE] Result: pixel_values.shape={collated['pixel_values'].shape}, "
                    f"labels.shape={collated['labels'].shape}, "
                    f"num_samples={len(collated['applied_transforms'])}")

    return collated


def build_npz_loader(
    npz_path: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
    augmentation_pipeline: Optional[Any] = None,
    sampler: Optional[Any] = None
) -> DataLoader:
    """
    Build DataLoader from NPZ file.

    Args:
        npz_path: Path to .npz file
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
    ds = NPZDataset(
        npz_path,
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
