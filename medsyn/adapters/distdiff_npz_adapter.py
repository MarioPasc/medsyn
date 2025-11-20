"""
NPZ Dataset Adapter for DistDiff Integration

This module provides a minimal adapter to load MedSyn's custom NPZ format
into DistDiff's training pipeline without modifying DistDiff's core code.

Custom NPZ Structure:
    - {split}_images: [N, H, W, C] uint8
    - {split}_labels: [N] int64
    - {split}_is_synth: [N] bool

Author: MedSyn Project
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List

# Import PathMNIST class names from medsyn
try:
    from medsyn.data.yolo_dataset import build_pathmnist_class_map
    HAS_MEDSYN = True
except ImportError:
    HAS_MEDSYN = False


class NPZDataset(Dataset):
    """
    Dataset for loading from custom NPZ files (MedSyn format).

    Compatible with DistDiff's training pipeline - returns (image, label) tuples.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        is_synth: Optional[np.ndarray] = None,
        transform=None,
        filter_synthetic: bool = True,
    ):
        """
        Args:
            images: numpy array [N, H, W, C] uint8
            labels: numpy array [N] int64
            is_synth: numpy array [N] bool (optional)
            transform: torchvision transforms
            filter_synthetic: If True and is_synth is provided, exclude synthetic images
        """
        # Filter synthetic images if requested
        if filter_synthetic and is_synth is not None:
            # Keep only real images (is_synth == False)
            mask = ~is_synth
            images = images[mask]
            labels = labels[mask]
            print(f"NPZDataset: Filtered {(~mask).sum()} synthetic images, kept {mask.sum()} real images")

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and label
        img = self.images[idx]  # [H, W, C] uint8
        label = int(self.labels[idx])

        # Convert to PIL Image for transform compatibility
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def load_npz_dataset(
    npz_path: str,
    split: str,
    transform,
    filter_synthetic: bool = True,
) -> NPZDataset:
    """
    Load a single split from an NPZ file.

    Args:
        npz_path: Path to NPZ file
        split: One of 'train', 'val', 'test'
        transform: torchvision transforms to apply
        filter_synthetic: Whether to filter out synthetic images (for guide model training)

    Returns:
        NPZDataset instance
    """
    print(f"Loading NPZ dataset from: {npz_path}")
    data = np.load(npz_path)

    # Load split data
    images = data[f'{split}_images']  # [N, H, W, C]
    labels = data[f'{split}_labels']  # [N]

    # Load is_synth if available
    is_synth_key = f'{split}_is_synth'
    is_synth = data[is_synth_key] if is_synth_key in data else None

    print(f"Loaded {split} split: {len(images)} samples, shape {images.shape}")

    return NPZDataset(
        images=images,
        labels=labels,
        is_synth=is_synth,
        transform=transform,
        filter_synthetic=filter_synthetic,
    )


def get_pathmnist_class_names() -> List[str]:
    """
    Get PathMNIST class names in the correct order.

    Returns:
        List of 9 class names for PathMNIST
    """
    if HAS_MEDSYN:
        # Use medsyn's official mapping
        class_map = build_pathmnist_class_map()
        # Sort by label index to get ordered list
        return [class_map[i] for i in sorted(class_map.keys())]
    else:
        # Fallback if medsyn is not available
        return [
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth_muscle",
            "normal_colon_mucosa",
            "cancer_associated_stroma",
            "colorectal_adenocarcinoma_epithelium",
        ]


def create_npz_dataloaders(
    npz_path: str,
    train_transform,
    test_transform,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int = 8,
    filter_synthetic_train: bool = True,
    filter_synthetic_val: bool = True,
):
    """
    Create DistDiff-compatible dataloaders from NPZ file.

    Args:
        npz_path: Path to NPZ file
        train_transform: Training transforms
        test_transform: Val/test transforms
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation/testing
        num_workers: Number of dataloader workers
        filter_synthetic_train: Filter synthetics from train split (recommended True for guide model)
        filter_synthetic_val: Filter synthetics from val split (recommended True)

    Returns:
        Tuple of (train_dataset, train_loader, train_loader_shuffle,
                  val_dataset, val_loader, test_dataset, test_loader,
                  num_classes, string_classnames)
    """
    # Load datasets
    train_dataset = load_npz_dataset(npz_path, 'train', train_transform, filter_synthetic_train)
    val_dataset = load_npz_dataset(npz_path, 'val', test_transform, filter_synthetic_val)
    test_dataset = load_npz_dataset(npz_path, 'test', test_transform, False)  # Never filter test

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    train_loader_shuffle = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Get class names
    string_classnames = get_pathmnist_class_names()
    num_classes = len(string_classnames)

    print(f'Loaded NPZ dataloaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}')
    print(f'Number of classes: {num_classes}')

    return (
        train_dataset, train_loader, train_loader_shuffle,
        val_dataset, val_loader, test_dataset, test_loader,
        num_classes, string_classnames
    )
