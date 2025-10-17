# medsyn/models/ccDDPM/dataloaders/npz.py
# Purpose: NPZ-based dataloader for compressed datasets on supercomputers
# Loads all data from a single .npz file containing images and metadata
from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

logger = logging.getLogger(__name__)

class NPZDataset(Dataset):
    """
    Dataset backed by a compressed NPZ file containing:
    - Images: arrays by split (train_images, val_images, test_images)
    - Labels: arrays by split (train_labels, val_labels, test_labels)
    - Metadata: is_synth flags by split (train_is_synth, val_is_synth, test_is_synth)
    
    NPZ structure:
    {
        'train_images': ndarray [N, H, W, C] uint8,
        'train_labels': ndarray [N] int64,
        'train_is_synth': ndarray [N] bool,
        'val_images': ndarray [M, H, W, C] uint8,
        'val_labels': ndarray [M] int64,
        'val_is_synth': ndarray [M] bool,
        'test_images': ndarray [K, H, W, C] uint8,
        'test_labels': ndarray [K] int64,
        'test_is_synth': ndarray [K] bool
    }
    """
    def __init__(
        self, 
        npz_path: Path, 
        split: str = "train", 
        image_size: int = 64, 
        normalize: bool = True
    ):
        """
        Args:
            npz_path: Path to the .npz file
            split: One of 'train', 'val', 'test'
            image_size: Target image size (will resize if needed)
            normalize: If True, normalize to [-1, 1], else [0, 1]
        """
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        
        # Load NPZ file
        logger.info(f"Loading NPZ dataset from {npz_path} (split={split})...")
        data = np.load(npz_path)
        
        # Extract split-specific data
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        is_synth_key = f"{split}_is_synth"
        
        if images_key not in data:
            raise KeyError(f"NPZ file missing key '{images_key}'. Available keys: {list(data.keys())}")
        
        self.images = data[images_key]  # [N, H, W, C] uint8
        self.labels = data[labels_key]  # [N] int64
        self.is_synth = data[is_synth_key] if is_synth_key in data else np.zeros(len(self.images), dtype=bool)
        
        # Validate shapes
        assert self.images.ndim == 4, f"Expected 4D images array, got shape {self.images.shape}"
        assert len(self.images) == len(self.labels), "Images and labels length mismatch"
        
        logger.info(f"Loaded {len(self.images)} images for split '{split}'")
        logger.info(f"  Image shape: {self.images.shape}")
        logger.info(f"  Labels shape: {self.labels.shape}")
        logger.info(f"  Unique labels: {np.unique(self.labels).tolist()}")
        
        # Build transform pipeline
        transforms_list = []
        
        # Resize if needed
        if self.images.shape[1] != image_size or self.images.shape[2] != image_size:
            transforms_list.append(T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR))
            logger.info(f"  Will resize from {self.images.shape[1]}x{self.images.shape[2]} to {image_size}x{image_size}")
        
        # Normalize
        if normalize:
            transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            logger.info("  Will normalize to [-1, 1]")
        
        self.transform = T.Compose(transforms_list) if transforms_list else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            Dictionary with keys:
            - pixel_values: Tensor [C, H, W] in [0,1] or [-1,1]
            - labels: Tensor scalar int64
            - is_synth: bool
            - idx: int (for debugging)
        """
        # Get image [H, W, C] uint8
        img = self.images[idx]
        
        # Convert to torch tensor [C, H, W] float32 in [0, 1]
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply transforms (resize, normalize)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        return {
            "pixel_values": img_tensor,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "is_synth": bool(self.is_synth[idx]),
            "idx": idx
        }


def build_npz_loader(
    npz_path: Path, 
    split: str, 
    image_size: int, 
    batch_size: int, 
    num_workers: int, 
    normalize: bool = True
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
    
    Returns:
        DataLoader instance
    """
    ds = NPZDataset(npz_path, split=split, image_size=image_size, normalize=normalize)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=(split == "train"), 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=(split == "train")
    )
