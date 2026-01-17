# medsyn/adapters/medfusion/dataset.py
"""
MedFusion dataset adapter for MedSyn's unified NPZ format.

This module provides PyTorch Dataset and Lightning DataModule that wrap
the unified NPZ format for use with MedFusion training.
"""
from __future__ import annotations
from typing import Dict, Optional, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import pytorch_lightning as pl
except ImportError:
    pl = None  # Will error at runtime if Lightning is needed

from medsyn.data.npz_format import load_npz_with_metadata, get_split_data


class MedFusionNPZDataset(Dataset):
    """
    PyTorch Dataset wrapping MedSyn's unified NPZ format for MedFusion.

    MedFusion expects samples as dictionaries with:
    - 'source': Image tensor [C, H, W] normalized to [-1, 1]
    - 'target': Integer class label (optional, for conditional generation)

    Args:
        npz_path: Path to unified NPZ file
        split: Which split to use ('train', 'val', 'test')
        image_size: Target image size (will resize if different)
        augment_horizontal_flip: Apply random horizontal flip
        augment_vertical_flip: Apply random vertical flip
    """

    def __init__(
        self,
        npz_path: str,
        split: str = "train",
        image_size: int = 64,
        augment_horizontal_flip: bool = True,
        augment_vertical_flip: bool = False,
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
        transform_list = [transforms.ToPILImage()]

        # Resize if needed
        if self.images.shape[1] != image_size or self.images.shape[2] != image_size:
            transform_list.append(transforms.Resize((image_size, image_size)))

        # Data augmentation (only for training)
        if split == "train":
            if augment_horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            if augment_vertical_flip:
                transform_list.append(transforms.RandomVerticalFlip())

        # Convert to tensor and normalize to [-1, 1]
        # MedFusion expects input in [-1, 1] with do_input_centering=False
        # or [0, 1] with do_input_centering=True
        transform_list.extend([
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=0.5, std=0.5),  # [-1, 1]
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample in MedFusion's expected format.

        Returns:
            Dictionary with:
            - 'source': [C, H, W] tensor in [-1, 1] range
            - 'target': integer class label
            - 'uid': sample identifier
        """
        img = self.images[idx]  # [H, W, C] uint8

        # Handle grayscale images by converting to RGB
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        label = int(self.labels[idx])
        img_tensor = self.transform(img)

        return {
            "source": img_tensor,
            "target": label,
            "uid": f"{self.split}_{idx}",
        }

    def get_weights(self) -> Optional[List[float]]:
        """
        Return per-sample weights for WeightedRandomSampler.

        MedFusion's SimpleDataModule can use this for class balancing.
        """
        labels = np.array(self.labels)
        class_counts = np.bincount(labels, minlength=self.num_classes)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weight_per_class = 1.0 / class_counts

        weights = [weight_per_class[label] for label in labels]
        return weights


class MedFusionDataModule:
    """
    Data module wrapping NPZ datasets for MedFusion training.

    Compatible with PyTorch Lightning's LightningDataModule interface
    but doesn't require Lightning as a dependency.

    Args:
        npz_path: Path to unified NPZ file
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        image_size: Target image size
        seed: Random seed for reproducibility
        pin_memory: Pin memory for faster GPU transfer
        use_weighted_sampling: Use class-balanced sampling for training
    """

    def __init__(
        self,
        npz_path: str,
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int = 64,
        seed: int = 0,
        pin_memory: bool = True,
        use_weighted_sampling: bool = False,
    ):
        self.npz_path = npz_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.seed = seed
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling

        # Create datasets
        self.train_ds = MedFusionNPZDataset(
            npz_path, "train", image_size,
            augment_horizontal_flip=True,
        )
        self.val_ds = MedFusionNPZDataset(
            npz_path, "val", image_size,
            augment_horizontal_flip=False,
        )
        self.test_ds = MedFusionNPZDataset(
            npz_path, "test", image_size,
            augment_horizontal_flip=False,
        )

        # Get weights for weighted sampling
        self.weights = self.train_ds.get_weights() if use_weighted_sampling else None

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        return self.train_ds.num_classes

    @property
    def dataset_name(self) -> str:
        """Name of the dataset."""
        return self.train_ds.dataset_name

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        if self.weights is not None:
            from torch.utils.data.sampler import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                self.weights, len(self.weights), generator=generator
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            generator=generator if sampler is None else None,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            generator=generator,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            generator=generator,
        )


# Lightning-compatible DataModule (if pytorch_lightning is available)
if pl is not None:
    class MedFusionLightningDataModule(pl.LightningDataModule):
        """
        PyTorch Lightning DataModule wrapping NPZ datasets for MedFusion.

        Args:
            npz_path: Path to unified NPZ file
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            image_size: Target image size
            seed: Random seed for reproducibility
            pin_memory: Pin memory for faster GPU transfer
            use_weighted_sampling: Use class-balanced sampling for training
        """

        def __init__(
            self,
            npz_path: str,
            batch_size: int = 32,
            num_workers: int = 8,
            image_size: int = 64,
            seed: int = 0,
            pin_memory: bool = True,
            use_weighted_sampling: bool = False,
        ):
            super().__init__()
            self._dm = MedFusionDataModule(
                npz_path=npz_path,
                batch_size=batch_size,
                num_workers=num_workers,
                image_size=image_size,
                seed=seed,
                pin_memory=pin_memory,
                use_weighted_sampling=use_weighted_sampling,
            )

        @property
        def num_classes(self) -> int:
            return self._dm.num_classes

        @property
        def dataset_name(self) -> str:
            return self._dm.dataset_name

        def train_dataloader(self) -> DataLoader:
            return self._dm.train_dataloader()

        def val_dataloader(self) -> DataLoader:
            return self._dm.val_dataloader()

        def test_dataloader(self) -> DataLoader:
            return self._dm.test_dataloader()
