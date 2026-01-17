# medsyn/adapters/distdiff/dataset.py
"""
NPZ Dataset Adapter for DistDiff Integration.

This module provides a standardized adapter to load MedSyn's unified NPZ format
into DistDiff's training and generation pipelines without modifying DistDiff's core code.

Supports all MedMNIST datasets: PathMNIST, BloodMNIST, DermaMNIST.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple
import logging

# Import dataset registry for class names
from medsyn.data.registry import DATASET_INFO, get_num_classes

logger = logging.getLogger(__name__)


# Prompt templates for each dataset (matching DistDiff's format)
DATASET_TEMPLATES = {
    "pathmnist": "A COLON PATHOLOGICAL IMAGE OF {}.",
    "bloodmnist": "a photo of {}, a type of cell.",
    "dermamnist": "a dermoscopy image of {}.",
    "breastmnist": "a photo of {} ultrasound image.",
}


class DistDiffNPZDataset(Dataset):
    """
    Dataset for loading from MedSyn unified NPZ files.

    Compatible with DistDiff's training pipeline - returns (image, label) tuples.
    Also provides `image_paths` attribute for compatibility with SDDataset,
    using virtual paths in the format "npz:<instance_id>:<index>".

    Args:
        images: numpy array [N, H, W, C] uint8
        labels: numpy array [N] int64
        is_synth: numpy array [N] bool (optional)
        transform: torchvision transforms
        filter_synthetic: If True and is_synth is provided, exclude synthetic images
    """

    # Class-level registry to allow SDDataset to retrieve images by virtual path
    _instances: dict = {}

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        is_synth: Optional[np.ndarray] = None,
        transform=None,
        filter_synthetic: bool = True,
    ):
        # Filter synthetic images if requested
        if filter_synthetic and is_synth is not None:
            mask = ~is_synth
            images = images[mask]
            labels = labels[mask]
            logger.info(f"Filtered {(~mask).sum()} synthetic images, kept {mask.sum()} real images")

        self.images = images
        self.labels = labels
        self.transform = transform

        # Generate unique instance ID and register for virtual path resolution
        self._instance_id = id(self)
        DistDiffNPZDataset._instances[self._instance_id] = self

        # Create virtual image paths for SDDataset compatibility
        self._image_paths = [f"npz:{self._instance_id}:{i}" for i in range(len(self.images))]

    @property
    def image_paths(self) -> List[str]:
        """Virtual paths for SDDataset compatibility."""
        return self._image_paths

    def get_pil_image(self, idx: int) -> Image.Image:
        """Get PIL Image by index (for SDDataset compatibility)."""
        img = self.images[idx]  # [H, W, C] uint8
        return Image.fromarray(img)

    @classmethod
    def get_image_from_virtual_path(cls, virtual_path: str) -> Image.Image:
        """
        Resolve a virtual path to a PIL Image.

        Args:
            virtual_path: Path in format "npz:<instance_id>:<index>"

        Returns:
            PIL Image
        """
        parts = virtual_path.split(":")
        if len(parts) != 3 or parts[0] != "npz":
            raise ValueError(f"Invalid virtual path format: {virtual_path}")

        instance_id = int(parts[1])
        idx = int(parts[2])

        if instance_id not in cls._instances:
            raise ValueError(
                f"NPZDataset instance {instance_id} not found. "
                "Dataset may have been garbage collected."
            )

        return cls._instances[instance_id].get_pil_image(idx)

    @classmethod
    def is_virtual_path(cls, path: str) -> bool:
        """Check if a path is a virtual NPZ path."""
        return isinstance(path, str) and path.startswith("npz:")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # [H, W, C] uint8
        label = int(self.labels[idx])

        # Convert to PIL Image for transform compatibility
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def __del__(self):
        """Cleanup: remove from class registry when garbage collected."""
        if hasattr(self, '_instance_id') and self._instance_id in DistDiffNPZDataset._instances:
            del DistDiffNPZDataset._instances[self._instance_id]


def load_npz_dataset(
    npz_path: str,
    split: str,
    transform=None,
    filter_synthetic: bool = True,
) -> DistDiffNPZDataset:
    """
    Load a single split from an NPZ file.

    Args:
        npz_path: Path to NPZ file
        split: One of 'train', 'val', 'test'
        transform: torchvision transforms to apply
        filter_synthetic: Whether to filter out synthetic images

    Returns:
        DistDiffNPZDataset instance
    """
    logger.info(f"Loading NPZ dataset from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    # Load split data
    images = data[f'{split}_images']  # [N, H, W, C]
    labels = data[f'{split}_labels']  # [N]

    # Load is_synth if available
    is_synth_key = f'{split}_is_synth'
    is_synth = data[is_synth_key] if is_synth_key in data else None

    logger.info(f"Loaded {split} split: {len(images)} samples, shape {images.shape}")

    return DistDiffNPZDataset(
        images=images,
        labels=labels,
        is_synth=is_synth,
        transform=transform,
        filter_synthetic=filter_synthetic,
    )


def get_class_names(dataset_name: str) -> List[str]:
    """
    Get class names for a dataset.

    Uses the MedSyn registry to get standardized class names.

    Args:
        dataset_name: One of 'pathmnist', 'bloodmnist', 'dermamnist'

    Returns:
        List of class names in order
    """
    if dataset_name not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(DATASET_INFO.keys())}"
        )

    return list(DATASET_INFO[dataset_name]["class_names"])


def get_prompt_template(dataset_name: str) -> str:
    """
    Get the prompt template for a dataset.

    Args:
        dataset_name: One of 'pathmnist', 'bloodmnist', 'dermamnist'

    Returns:
        Prompt template string with {} placeholder for class name
    """
    if dataset_name not in DATASET_TEMPLATES:
        logger.warning(
            f"No template for dataset {dataset_name}, using generic template"
        )
        return "a photo of a {}."

    return DATASET_TEMPLATES[dataset_name]


def create_npz_dataloaders(
    npz_path: str,
    train_transform,
    test_transform,
    train_batch_size: int,
    val_batch_size: int,
    dataset_name: str = "pathmnist",
    num_workers: int = 8,
    filter_synthetic_train: bool = True,
    filter_synthetic_val: bool = True,
) -> Tuple:
    """
    Create DistDiff-compatible dataloaders from NPZ file.

    Args:
        npz_path: Path to NPZ file
        train_transform: Training transforms
        test_transform: Val/test transforms
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation/testing
        dataset_name: Name of dataset for class names
        num_workers: Number of dataloader workers
        filter_synthetic_train: Filter synthetics from train split
        filter_synthetic_val: Filter synthetics from val split

    Returns:
        Tuple of (train_dataset, train_loader, train_loader_shuffle,
                  val_dataset, val_loader, test_dataset, test_loader,
                  num_classes, string_classnames)
    """
    # Load datasets
    train_dataset = load_npz_dataset(
        npz_path, 'train', train_transform, filter_synthetic_train
    )
    val_dataset = load_npz_dataset(
        npz_path, 'val', test_transform, filter_synthetic_val
    )
    test_dataset = load_npz_dataset(
        npz_path, 'test', test_transform, False  # Never filter test
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    train_loader_shuffle = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    # Get class names from registry
    string_classnames = get_class_names(dataset_name)
    num_classes = len(string_classnames)

    logger.info(
        f'Loaded NPZ dataloaders: train={len(train_dataset)}, '
        f'val={len(val_dataset)}, test={len(test_dataset)}'
    )
    logger.info(f'Dataset: {dataset_name}, num_classes: {num_classes}')

    return (
        train_dataset, train_loader, train_loader_shuffle,
        val_dataset, val_loader, test_dataset, test_loader,
        num_classes, string_classnames
    )


# Backwards compatibility alias for existing code
NPZDataset = DistDiffNPZDataset
