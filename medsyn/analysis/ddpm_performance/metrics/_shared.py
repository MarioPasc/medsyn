#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for FID and KID metric computation scripts.

This module provides common functionality used by both fid.py and kid.py:
- NPZ data loading
- Validation of sample counts per class
- Image preprocessing for torchmetrics
- CSV output formatting
- Inception weights configuration
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def load_npz_data(npz_path: Path, split: str = "train") -> Dict[str, np.ndarray]:
    """
    Load NPZ data and return images, labels, and is_synth arrays.

    This function loads data from standardized NPZ files with keys:
    - {split}_images: uint8 array [N, H, W, C]
    - {split}_labels: int64 array [N]
    - {split}_is_synth: bool array [N] (optional, defaults to False)

    Args:
        npz_path: Path to NPZ file
        split: Data split to load (default: "train")

    Returns:
        Dictionary with keys: 'images', 'labels', 'is_synth'

    Raises:
        ValueError: If required keys are missing from NPZ file
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    logger.debug(f"Loading NPZ data from: {npz_path}")
    data = np.load(npz_path)

    images_key = f"{split}_images"
    labels_key = f"{split}_labels"
    is_synth_key = f"{split}_is_synth"

    if images_key not in data:
        raise ValueError(
            f"Required key '{images_key}' not found in NPZ file. "
            f"Available keys: {list(data.keys())}"
        )

    if labels_key not in data:
        raise ValueError(
            f"Required key '{labels_key}' not found in NPZ file. "
            f"Available keys: {list(data.keys())}"
        )

    images = data[images_key]
    labels = data[labels_key]
    is_synth = data.get(is_synth_key, np.zeros(len(images), dtype=bool))

    logger.info(f"Loaded {len(images)} images from {npz_path}")
    logger.debug(f"  Image shape: {images.shape}, dtype: {images.dtype}")
    logger.debug(f"  Label shape: {labels.shape}, dtype: {labels.dtype}")
    logger.debug(f"  Synth count: {np.sum(is_synth)}/{len(is_synth)}")

    return {
        "images": images,
        "labels": labels,
        "is_synth": is_synth,
    }


def get_class_distribution(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute per-class sample counts.

    Args:
        labels: Label array [N]
        num_classes: Total number of classes

    Returns:
        Array of shape [num_classes] with sample count per class
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    unique_labels, unique_counts = np.unique(labels, return_counts=True)

    for label, count in zip(unique_labels, unique_counts):
        if 0 <= label < num_classes:
            counts[label] = count

    return counts


def validate_class_counts(
    labels: np.ndarray,
    num_classes: int,
    min_samples_per_class: int,
    dataset_name: str,
    allow_skip: bool = False,
) -> List[int]:
    """
    Validate that all classes have sufficient samples.

    This function checks that every class has at least min_samples_per_class samples.
    If any class has insufficient samples, it raises a ValueError with a detailed
    error message showing which classes are problematic.

    Args:
        labels: Label array [N]
        num_classes: Total number of classes
        min_samples_per_class: Minimum required samples per class
        dataset_name: Name of dataset (for error messages)
        allow_skip: If True, return list of insufficient classes instead of raising

    Returns:
        List of class IDs with insufficient samples (empty if all valid)

    Raises:
        ValueError: If any class has insufficient samples and allow_skip=False
    """
    counts = get_class_distribution(labels, num_classes)
    insufficient_classes = []

    logger.debug(f"Validating class counts for '{dataset_name}':")
    for class_id in range(num_classes):
        count = counts[class_id]
        status = "✓" if count >= min_samples_per_class else "✗ (INSUFFICIENT)"
        logger.debug(f"  Class {class_id}: {count} samples {status}")

        if count < min_samples_per_class:
            insufficient_classes.append(class_id)

    if insufficient_classes:
        error_msg = (
            f"\nERROR: Insufficient samples in dataset '{dataset_name}'\n"
            f"Required: {min_samples_per_class} samples per class\n"
            f"Found:\n"
        )
        for class_id in range(num_classes):
            count = counts[class_id]
            status = "✓" if count >= min_samples_per_class else "✗ (INSUFFICIENT)"
            error_msg += f"  Class {class_id}: {count} {status}\n"
        error_msg += f"Classes with insufficient samples: {insufficient_classes}\n"
        error_msg += f"Solution: Either lower --counts parameter or generate more samples."

        if not allow_skip:
            raise ValueError(error_msg)
        else:
            logger.warning(error_msg)

    return insufficient_classes


def preprocess_images_for_torchmetrics(images: np.ndarray) -> torch.Tensor:
    """
    Convert NPZ images to torchmetrics format.

    torchmetrics FID/KID expects images in uint8 [N, C, H, W] format in range [0, 255].

    Args:
        images: NumPy array of shape [N, H, W, C] in uint8 format [0, 255]

    Returns:
        Torch tensor of shape [N, C, H, W] in uint8 format [0, 255]
    """
    # Convert to torch tensor
    images_torch = torch.from_numpy(images)  # [N, H, W, C]

    # Permute to [N, C, H, W]
    images_torch = images_torch.permute(0, 3, 1, 2)  # [N, C, H, W]

    # Ensure uint8 dtype
    if images_torch.dtype != torch.uint8:
        logger.warning(
            f"Images are not uint8 (found {images_torch.dtype}). "
            f"Converting to uint8..."
        )
        images_torch = images_torch.to(torch.uint8)

    logger.debug(f"Preprocessed images: shape={images_torch.shape}, dtype={images_torch.dtype}")

    return images_torch


def create_subsets(
    images: torch.Tensor,
    labels: torch.Tensor,
    class_id: int,
    subset_size: int,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Create non-overlapping subsets for a specific class.

    This function:
    1. Extracts all samples for the given class
    2. Shuffles them with a fixed seed for reproducibility
    3. Splits into non-overlapping chunks of size subset_size
    4. Discards remaining samples that don't form a full subset

    Args:
        images: All images [N, C, H, W]
        labels: All labels [N]
        class_id: Class to extract
        subset_size: Size of each subset
        seed: Random seed for shuffling

    Returns:
        List of image tensors, each of shape [subset_size, C, H, W]
        Only full subsets are returned; partial subsets are discarded.
    """
    # Extract class samples
    mask = labels == class_id
    class_images = images[mask]

    if len(class_images) == 0:
        return []

    # Shuffle with fixed seed
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(class_images), generator=generator)
    shuffled_images = class_images[indices]

    # Calculate how many full subsets we can create
    num_full_subsets = len(shuffled_images) // subset_size
    samples_used = num_full_subsets * subset_size
    samples_discarded = len(shuffled_images) - samples_used

    # Split into non-overlapping subsets
    subsets = []
    for i in range(0, len(shuffled_images), subset_size):
        subset = shuffled_images[i:i + subset_size]
        if len(subset) == subset_size:  # Only include full subsets
            subsets.append(subset)

    logger.info(
        f"Class {class_id}: {len(class_images)} total samples -> "
        f"{len(subsets)} subsets × {subset_size} = {samples_used} used, "
        f"{samples_discarded} discarded"
    )

    return subsets


def save_results_to_csv(
    results: Dict[str, Dict[str, float]],
    output_path: Path,
    num_classes: int,
) -> None:
    """
    Save metric results to CSV with proper formatting.

    The CSV will have columns:
    - model_name
    - class_0_mean, class_0_std, class_1_mean, class_1_std, ...
    - average_mean, average_std

    Args:
        results: Dict mapping model_name -> {metric_name: value}
                 Expected keys per model: class_0_mean, class_0_std, ..., average_mean, average_std
        output_path: Path to save CSV
        num_classes: Number of classes
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build column order
    columns = ["model_name"]
    for class_id in range(num_classes):
        columns.append(f"class_{class_id}_mean")
        columns.append(f"class_{class_id}_std")
    columns.extend(["average_mean", "average_std"])

    # Build DataFrame rows
    rows = []
    for model_name, metrics in results.items():
        row = {"model_name": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns to match expected format
    available_cols = [col for col in columns if col in df.columns]
    df = df[available_cols]

    # Save to CSV
    df.to_csv(output_path, index=False, float_format="%.4f")
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print(df.to_string(index=False))


def setup_inception_weights_path(weights_dir: Path) -> None:
    """
    Configure Inception weights path for offline usage.

    This function sets the TORCH_HOME environment variable to point to the
    directory containing pre-downloaded Inception weights. This is useful
    for offline HPC environments where internet access is restricted.

    Args:
        weights_dir: Directory containing the 'hub/checkpoints/' subdirectory
                     with Inception weight files (.pth)
    """
    weights_dir = Path(weights_dir)

    if not weights_dir.exists():
        logger.warning(
            f"Weights directory does not exist: {weights_dir}. "
            f"torchmetrics will attempt to download weights from the internet."
        )
        return

    # Set TORCH_HOME environment variable
    os.environ["TORCH_HOME"] = str(weights_dir)
    logger.info(f"Set TORCH_HOME={weights_dir} for Inception weights")

    # Verify expected structure
    checkpoint_dir = weights_dir / "hub" / "checkpoints"
    if checkpoint_dir.exists():
        pth_files = list(checkpoint_dir.glob("*.pth"))
        logger.debug(f"Found {len(pth_files)} .pth files in {checkpoint_dir}")
    else:
        logger.warning(
            f"Expected checkpoint directory not found: {checkpoint_dir}. "
            f"Weights may need to be downloaded."
        )


def warn_if_few_subsets(num_subsets: int, class_id: int, threshold: int = 3) -> None:
    """
    Warn if number of subsets is too small for reliable std estimation.

    Args:
        num_subsets: Number of subsets available
        class_id: Class ID (for warning message)
        threshold: Minimum recommended number of subsets
    """
    if num_subsets < threshold:
        logger.warning(
            f"Class {class_id}: Only {num_subsets} subset(s) available. "
            f"Standard deviation may be unreliable with fewer than {threshold} subsets. "
            f"Consider lowering --counts for more reliable std estimation."
        )


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: If True, set log level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
