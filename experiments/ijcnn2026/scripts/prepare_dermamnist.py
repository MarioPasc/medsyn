#!/usr/bin/env python3
"""
IJCNN 2026 - DermaMNIST Data Preparation Script

Downloads DermaMNIST using the medmnist package and creates an NPZ file
in the same format as PathMNIST for use with the ccDDPM training pipeline.

Usage:
    python prepare_dermamnist.py --output /path/to/output/DermaMNIST.npz --size 64
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def resize_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """Resize images to target size using PIL."""
    if images.shape[1] == target_size and images.shape[2] == target_size:
        return images

    logger.info(f"Resizing images from {images.shape[1]}x{images.shape[2]} to {target_size}x{target_size}")
    resized = []
    for img in tqdm(images, desc="Resizing"):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size), Image.LANCZOS)
        resized.append(np.array(pil_img))
    return np.array(resized)


def main():
    parser = argparse.ArgumentParser(description="Prepare DermaMNIST dataset for ccDDPM training")
    parser.add_argument("--output", type=str, required=True, help="Output NPZ file path")
    parser.add_argument("--size", type=int, default=64, help="Target image size (default: 64)")
    parser.add_argument("--download_dir", type=str, default="./data_raw", help="Download directory")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Import medmnist
    try:
        from medmnist import DermaMNIST
    except ImportError:
        logger.error("medmnist not installed. Run: pip install medmnist")
        return 1

    logger.info("Loading DermaMNIST dataset...")

    # Download and load dataset
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DermaMNIST(split="train", download=True, size=28, root=str(download_dir))
    val_ds = DermaMNIST(split="val", download=True, size=28, root=str(download_dir))
    test_ds = DermaMNIST(split="test", download=True, size=28, root=str(download_dir))

    # Extract images and labels
    train_images = train_ds.imgs  # (N, 28, 28, 3) uint8
    train_labels = train_ds.labels.squeeze()  # (N,)
    val_images = val_ds.imgs
    val_labels = val_ds.labels.squeeze()
    test_images = test_ds.imgs
    test_labels = test_ds.labels.squeeze()

    logger.info(f"Loaded: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    # Print class distribution
    num_classes = 7
    logger.info("Class distribution:")
    for c in range(num_classes):
        train_count = (train_labels == c).sum()
        val_count = (val_labels == c).sum()
        test_count = (test_labels == c).sum()
        logger.info(f"  Class {c}: train={train_count}, val={val_count}, test={test_count}")

    # Resize if needed
    if args.size != 28:
        train_images = resize_images(train_images, args.size)
        val_images = resize_images(val_images, args.size)
        test_images = resize_images(test_images, args.size)

    # Create is_synth flags (all False for real data)
    train_is_synth = np.zeros(len(train_images), dtype=bool)
    val_is_synth = np.zeros(len(val_images), dtype=bool)
    test_is_synth = np.zeros(len(test_images), dtype=bool)

    # Save to NPZ
    logger.info(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        train_images=train_images,
        train_labels=train_labels.astype(np.int64),
        train_is_synth=train_is_synth,
        val_images=val_images,
        val_labels=val_labels.astype(np.int64),
        val_is_synth=val_is_synth,
        test_images=test_images,
        test_labels=test_labels.astype(np.int64),
        test_is_synth=test_is_synth,
    )

    # Verify
    data = np.load(output_path)
    logger.info("Verification:")
    logger.info(f"  train_images: {data['train_images'].shape}, dtype={data['train_images'].dtype}")
    logger.info(f"  train_labels: {data['train_labels'].shape}, dtype={data['train_labels'].dtype}")
    logger.info(f"  val_images: {data['val_images'].shape}")
    logger.info(f"  test_images: {data['test_images'].shape}")

    logger.info(f"DermaMNIST NPZ created successfully at {output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
