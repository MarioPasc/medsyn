# medsyn/adapters/cbdm/dataset.py
"""
CBDM dataset adapter for MedSyn's unified NPZ format.

This module provides a PyTorch Dataset that wraps the unified NPZ format
for use with CBDM training and generation.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from medsyn.data.npz_format import load_npz_with_metadata, get_split_data


class CBDMNPZDataset(Dataset):
    """
    PyTorch Dataset wrapping MedSyn's unified NPZ format for CBDM.

    CBDM expects:
    - Images normalized to [-1, 1]
    - Images in [C, H, W] format
    - Integer labels

    This dataset also provides a `targets` property for class weighting,
    which CBDM uses for class-balancing.

    Args:
        npz_path: Path to unified NPZ file
        split: Which split to use ('train', 'val', 'test')
        image_size: Target image size (will resize if different)
        augment: Whether to apply data augmentation (flips)
    """

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        image_size: int = 64,
        augment: bool = True,
    ):
        super().__init__()

        # Load data from NPZ
        data, metadata = load_npz_with_metadata(npz_path)
        self.images, self.labels, self.is_synth = get_split_data(data, split)

        self.split = split
        self.image_size = image_size
        self.num_classes = metadata.get("num_classes", len(np.unique(self.labels)))
        self.dataset_name = metadata.get("dataset_name", "unknown")

        # Build transform pipeline
        # CBDM expects images in [-1, 1] range
        transform_list = [transforms.ToPILImage()]

        # Resize if needed
        if self.images.shape[1] != image_size or self.images.shape[2] != image_size:
            transform_list.append(transforms.Resize((image_size, image_size)))

        # Data augmentation (only for training)
        if augment and split == "train":
            transform_list.append(transforms.RandomHorizontalFlip())

        # Convert to tensor and normalize to [-1, 1]
        transform_list.extend([
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),  # [-1, 1]
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.

        Returns:
            Tuple of (image_tensor, label):
            - image_tensor: [C, H, W] in [-1, 1] range
            - label: integer class label
        """
        img = self.images[idx]  # [H, W, C] uint8

        # Handle grayscale images by converting to RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        label = int(self.labels[idx])
        img_tensor = self.transform(img)
        return img_tensor, label

    @property
    def targets(self) -> List[int]:
        """
        Return labels as a list for class weighting.

        CBDM uses this property to compute class weights for balancing.
        """
        return self.labels.tolist()

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency class weights for balanced sampling.

        Returns:
            Tensor of shape [num_classes] with weights
        """
        labels = np.array(self.targets)
        class_counts = np.bincount(labels, minlength=self.num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        weights = weights / weights.sum()  # Normalize
        return torch.from_numpy(weights).float()

    def get_sample_weights(self) -> torch.Tensor:
        """
        Compute per-sample weights for WeightedRandomSampler.

        Returns:
            Tensor of shape [num_samples] with weights
        """
        class_weights = self.get_class_weights().numpy()
        sample_weights = class_weights[self.labels]
        return torch.from_numpy(sample_weights).float()


class CBDMGenerationDataset(Dataset):
    """
    Dataset for generating samples with specified class labels.

    This is used during generation to specify which classes to generate.

    Args:
        num_samples: Total number of samples to generate
        num_classes: Number of classes
        samples_per_class: If specified, generate equal samples per class
        class_weights: If specified, sample classes according to weights
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        samples_per_class: Optional[int] = None,
        class_weights: Optional[np.ndarray] = None,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes

        if samples_per_class is not None:
            # Generate balanced labels
            self.labels = np.repeat(
                np.arange(num_classes),
                samples_per_class,
            )[:num_samples]
        elif class_weights is not None:
            # Sample according to weights
            self.labels = np.random.choice(
                num_classes,
                size=num_samples,
                p=class_weights / class_weights.sum(),
            )
        else:
            # Uniform sampling
            self.labels = np.random.randint(0, num_classes, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> int:
        return int(self.labels[idx])
