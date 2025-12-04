"""
NPZ-based dataset and dataloader for classification with regime-aware training.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

Split = Literal["train", "val", "test"]
TrainingRegime = Literal["real_only", "real_plus_synth", "real_plus_trad_aug"]

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def select_indices_by_regime(is_synth: np.ndarray, regime: TrainingRegime) -> np.ndarray:
    """
    Select indices based on training regime.

    Args:
        is_synth: Boolean array indicating synthetic samples
        regime: Training regime

    Returns:
        Boolean mask of samples to keep
    """
    if regime == "real_only":
        return ~is_synth.astype(bool)
    elif regime == "real_plus_synth":
        return np.ones_like(is_synth, dtype=bool)
    elif regime == "real_plus_trad_aug":
        # Only real data, augmentation applied in __getitem__
        return ~is_synth.astype(bool)
    else:
        raise ValueError(f"Unknown regime: {regime}")


def _numpy_to_albumentations_format(x: np.ndarray) -> np.ndarray:
    """
    Convert numpy array to albumentations format [H, W, C] uint8 [0, 255].
    Handles various input formats and conversions.
    """
    # Convert dtype to uint8 if needed
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = (x * 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)

    # Convert from [C, H, W] to [H, W, C] if needed
    if x.ndim == 3 and x.shape[0] in [1, 3]:
        x = np.transpose(x, (1, 2, 0))

    # Grayscale to RGB
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x


@dataclass
class ClassificationDataset(Dataset):
    """
    NPZ-based classification dataset with regime-aware data selection.

    Supports three training regimes:
    1. real_only: Only real images
    2. real_plus_synth: Real + synthetic images
    3. real_plus_trad_aug: Real images + traditional augmentation

    For k-fold CV, provide fold_images/fold_labels/fold_is_synth to override NPZ loading.
    """
    npz_path: Path
    split: Split
    regime: TrainingRegime
    image_size: int = 64
    augmentation_pipeline: Optional[Any] = None  # AugmentationPipeline from ccDDPM

    # Optional fold data for k-fold CV (overrides NPZ loading)
    fold_images: Optional[np.ndarray] = None
    fold_labels: Optional[np.ndarray] = None
    fold_is_synth: Optional[np.ndarray] = None

    def __post_init__(self):
        # Load data from NPZ or use fold data
        if self.fold_images is not None:
            X = self.fold_images
            y = self.fold_labels.reshape(-1).astype(np.int64)
            is_synth = self.fold_is_synth if self.fold_is_synth is not None else np.zeros_like(y, dtype=bool)
        else:
            with np.load(self.npz_path) as data:
                X = data[f"{self.split}_images"]
                y = data[f"{self.split}_labels"].reshape(-1).astype(np.int64)
                is_synth = data.get(f"{self.split}_is_synth", np.zeros(len(y), dtype=bool))

        # IMPORTANT: Test and validation sets should NEVER contain synthetic images
        # They are used for evaluation and must only contain real data
        if self.split in ["test", "val"]:
            import logging
            logger = logging.getLogger(__name__)

            n_synth = is_synth.sum()
            if n_synth > 0:
                logger.warning(
                    f"Found {n_synth} synthetic images in {self.split} set! "
                    f"These will be excluded for proper evaluation."
                )
            # Force exclusion of synthetic images for test/val splits
            keep = ~is_synth.astype(bool)
            X = X[keep]
            y = y[keep]

            if n_synth > 0:
                logger.info(
                    f"{self.split.capitalize()} set: {len(X)} real images "
                    f"(excluded {n_synth} synthetic images)"
                )
        else:
            # For training split, apply regime-based filtering
            keep = select_indices_by_regime(is_synth, self.regime)
            X = X[keep]
            y = y[keep]

        # Remap labels to contiguous [0, K-1]
        unique_labels = np.unique(y)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.vectorize(label_map.__getitem__)(y).astype(np.int64)

        self.num_classes = len(unique_labels)
        self.X = X
        self.y = y

        # Build transforms
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> A.Compose:
        """Build albumentations transform pipeline."""
        transforms = []

        # Resize if needed
        transforms.append(A.Resize(self.image_size, self.image_size))

        # Normalize and convert to tensor
        transforms.extend([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2()
        ])

        return A.Compose(transforms)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict:
        image = self.X[idx]
        label = self.y[idx]

        # Convert to albumentations format [H, W, C] uint8
        image = _numpy_to_albumentations_format(image)

        # Apply augmentation if regime is real_plus_trad_aug and we're training
        if (self.regime == "real_plus_trad_aug" and
            self.split == "train" and
            self.augmentation_pipeline is not None):
            # Convert to tensor for augmentation pipeline (expects [C, H, W])
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # Normalize to [-1, 1] for augmentation pipeline
            image_tensor = (image_tensor - 0.5) / 0.5

            # Apply augmentation
            augmented_tensor, _ = self.augmentation_pipeline(image_tensor)

            # Convert back to [H, W, C] uint8 for albumentations normalization
            augmented_tensor = (augmented_tensor * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
            image = (augmented_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Apply transforms (resize, normalize, to tensor)
        transformed = self.transforms(image=image)
        image_tensor = transformed["image"]

        return {
            "img": image_tensor,
            "cls": torch.tensor(label, dtype=torch.long)
        }


def build_classification_dataloader(
    npz_path: Path,
    split: Split,
    regime: TrainingRegime,
    batch_size: int,
    num_workers: int,
    image_size: int = 64,
    augmentation_pipeline: Optional[Any] = None,
    fold_images: Optional[np.ndarray] = None,
    fold_labels: Optional[np.ndarray] = None,
    fold_is_synth: Optional[np.ndarray] = None
) -> DataLoader:
    """
    Build DataLoader for classification.

    Args:
        npz_path: Path to NPZ file
        split: Data split (train/val/test)
        regime: Training regime
        batch_size: Batch size
        num_workers: Number of dataloader workers
        image_size: Target image size
        augmentation_pipeline: Optional augmentation pipeline (for real_plus_trad_aug)
        fold_images: Optional fold images (for k-fold CV)
        fold_labels: Optional fold labels (for k-fold CV)
        fold_is_synth: Optional fold synthetic flags (for k-fold CV)

    Returns:
        DataLoader instance
    """
    dataset = ClassificationDataset(
        npz_path=npz_path,
        split=split,
        regime=regime,
        image_size=image_size,
        augmentation_pipeline=augmentation_pipeline,
        fold_images=fold_images,
        fold_labels=fold_labels,
        fold_is_synth=fold_is_synth
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train")  # Drop last incomplete batch in training
    )
