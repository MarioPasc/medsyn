# medsyn/data/npz_format.py
"""
Unified NPZ format utilities for MedSyn datasets.

This module provides functions for creating and loading NPZ files
with standardized metadata for reproducibility and interoperability.

Standard NPZ Keys:
- {split}_images: [N, H, W, C] uint8
- {split}_labels: [N] int64
- {split}_is_synth: [N] bool

Metadata Key:
- metadata: JSON string containing dataset info
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from medsyn.data.registry import get_dataset_info

logger = logging.getLogger(__name__)


def create_metadata(
    dataset_name: str,
    image_size: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create metadata dictionary for a dataset.

    Args:
        dataset_name: Dataset flag (e.g., 'pathmnist')
        image_size: Image dimension (e.g., 28, 64)
        extra: Optional extra metadata fields

    Returns:
        Metadata dictionary with dataset info
    """
    info = get_dataset_info(dataset_name)
    metadata = {
        "dataset_name": dataset_name,
        "num_classes": info["num_classes"],
        "class_names": info["class_names"],
        "image_size": image_size,
        "format_version": "1.0",
    }
    if extra:
        metadata.update(extra)
    return metadata


def encode_metadata(metadata: Dict[str, Any]) -> str:
    """Encode metadata dict as JSON string for NPZ storage."""
    return json.dumps(metadata, ensure_ascii=False, indent=None)


def decode_metadata(metadata_str: Union[str, np.ndarray]) -> Dict[str, Any]:
    """
    Decode metadata from NPZ storage.

    Args:
        metadata_str: JSON string or numpy array containing JSON

    Returns:
        Metadata dictionary
    """
    if isinstance(metadata_str, np.ndarray):
        # NPZ stores strings as 0-d arrays
        metadata_str = str(metadata_str)
    return json.loads(metadata_str)


def create_unified_npz(
    splits: Dict[str, Dict[str, np.ndarray]],
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
    compress: bool = True,
) -> Path:
    """
    Create a unified NPZ file with standardized format and metadata.

    Args:
        splits: Dictionary mapping split names to data dicts:
            {
                "train": {"images": np.array, "labels": np.array, "is_synth": np.array},
                "val": {...},
                "test": {...}
            }
        metadata: Metadata dictionary (will be JSON-encoded)
        output_path: Path for output NPZ file
        compress: Whether to use compressed NPZ (default: True)

    Returns:
        Path to created NPZ file

    Example:
        >>> splits = {
        ...     "train": {"images": train_imgs, "labels": train_labels, "is_synth": train_synth},
        ...     "val": {"images": val_imgs, "labels": val_labels, "is_synth": val_synth},
        ... }
        >>> metadata = create_metadata("pathmnist", 64)
        >>> create_unified_npz(splits, metadata, "pathmnist_64.npz")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build output dictionary
    output_dict: Dict[str, np.ndarray] = {}

    for split_name, data in splits.items():
        images = data["images"]
        labels = data["labels"]
        is_synth = data.get("is_synth")

        # Validate and normalize image format: [N, H, W, C]
        if images.ndim == 3:
            # Add channel dimension if missing
            images = images[..., np.newaxis]
        if images.ndim != 4:
            raise ValueError(
                f"Images must be 3D or 4D, got shape {images.shape}"
            )

        # Ensure proper dtypes
        images = images.astype(np.uint8)
        labels = labels.squeeze().astype(np.int64)

        # Create is_synth array if not provided
        if is_synth is None:
            is_synth = np.zeros(len(images), dtype=bool)
        else:
            is_synth = is_synth.astype(bool)

        # Store in output dict
        output_dict[f"{split_name}_images"] = images
        output_dict[f"{split_name}_labels"] = labels
        output_dict[f"{split_name}_is_synth"] = is_synth

        logger.info(
            "Split %s: %d samples, shape %s",
            split_name,
            len(images),
            images.shape,
        )

    # Add metadata as JSON string
    output_dict["metadata"] = np.array(encode_metadata(metadata))

    # Save NPZ
    if compress:
        np.savez_compressed(str(output_path), **output_dict)
    else:
        np.savez(str(output_path), **output_dict)

    logger.info("Unified NPZ saved to: %s", output_path)
    return output_path


def load_npz_with_metadata(
    npz_path: Union[str, Path],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load a unified NPZ file and extract data with metadata.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Tuple of (data_dict, metadata_dict):
        - data_dict: Contains all arrays from NPZ (e.g., train_images, train_labels, etc.)
        - metadata_dict: Decoded metadata dictionary

    Example:
        >>> data, meta = load_npz_with_metadata("pathmnist_64.npz")
        >>> print(meta["dataset_name"], meta["num_classes"])
        pathmnist 9
        >>> print(data["train_images"].shape)
        (89996, 64, 64, 3)
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = dict(np.load(str(npz_path), allow_pickle=True))

    # Extract and decode metadata
    metadata = {}
    if "metadata" in data:
        metadata_raw = data.pop("metadata")
        metadata = decode_metadata(metadata_raw)
    else:
        logger.warning("NPZ file does not contain metadata key")

    return data, metadata


def get_split_data(
    data_dict: Dict[str, np.ndarray],
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract split data from loaded NPZ dictionary.

    Args:
        data_dict: Dictionary loaded from NPZ
        split: Split name ('train', 'val', 'test')

    Returns:
        Tuple of (images, labels, is_synth) arrays
    """
    images = data_dict[f"{split}_images"]
    labels = data_dict[f"{split}_labels"]
    is_synth = data_dict.get(f"{split}_is_synth", np.zeros(len(images), dtype=bool))
    return images, labels, is_synth


def merge_synthetic_data(
    original_npz_path: Union[str, Path],
    synthetic_images: np.ndarray,
    synthetic_labels: np.ndarray,
    output_path: Union[str, Path],
    split: str = "train",
) -> Path:
    """
    Merge synthetic data with original dataset.

    Args:
        original_npz_path: Path to original NPZ file
        synthetic_images: Synthetic images [N, H, W, C]
        synthetic_labels: Synthetic labels [N]
        output_path: Path for merged NPZ
        split: Which split to augment (default: 'train')

    Returns:
        Path to merged NPZ file
    """
    data, metadata = load_npz_with_metadata(original_npz_path)

    # Get original split data
    orig_images = data[f"{split}_images"]
    orig_labels = data[f"{split}_labels"]
    orig_is_synth = data.get(
        f"{split}_is_synth",
        np.zeros(len(orig_images), dtype=bool),
    )

    # Validate synthetic data
    if synthetic_images.ndim == 3:
        synthetic_images = synthetic_images[..., np.newaxis]

    # Create is_synth flags for synthetic data (all True)
    synth_is_synth = np.ones(len(synthetic_images), dtype=bool)

    # Merge
    merged_images = np.concatenate([orig_images, synthetic_images], axis=0)
    merged_labels = np.concatenate([orig_labels, synthetic_labels], axis=0)
    merged_is_synth = np.concatenate([orig_is_synth, synth_is_synth], axis=0)

    # Update data dict
    data[f"{split}_images"] = merged_images
    data[f"{split}_labels"] = merged_labels
    data[f"{split}_is_synth"] = merged_is_synth

    # Update metadata
    metadata["augmented_split"] = split
    metadata["original_samples"] = len(orig_images)
    metadata["synthetic_samples"] = len(synthetic_images)
    metadata["total_samples"] = len(merged_images)

    # Rebuild splits dict for create_unified_npz
    splits = {}
    for key in list(data.keys()):
        if key.endswith("_images"):
            split_name = key.replace("_images", "")
            splits[split_name] = {
                "images": data[f"{split_name}_images"],
                "labels": data[f"{split_name}_labels"],
                "is_synth": data.get(f"{split_name}_is_synth"),
            }

    return create_unified_npz(splits, metadata, output_path)


def validate_npz_format(npz_path: Union[str, Path]) -> List[str]:
    """
    Validate that an NPZ file conforms to the unified format.

    Args:
        npz_path: Path to NPZ file

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: List[str] = []
    npz_path = Path(npz_path)

    if not npz_path.exists():
        return [f"File not found: {npz_path}"]

    try:
        data = dict(np.load(str(npz_path), allow_pickle=True))
    except Exception as e:
        return [f"Failed to load NPZ: {e}"]

    # Check for required splits
    required_splits = ["train"]  # At minimum need train
    for split in required_splits:
        if f"{split}_images" not in data:
            errors.append(f"Missing required key: {split}_images")
        if f"{split}_labels" not in data:
            errors.append(f"Missing required key: {split}_labels")

    # Check metadata
    if "metadata" not in data:
        errors.append("Missing metadata key")
    else:
        try:
            metadata = decode_metadata(data["metadata"])
            required_meta = ["dataset_name", "num_classes", "image_size"]
            for key in required_meta:
                if key not in metadata:
                    errors.append(f"Missing metadata field: {key}")
        except Exception as e:
            errors.append(f"Invalid metadata format: {e}")

    # Validate array shapes
    for key in data.keys():
        if key.endswith("_images"):
            arr = data[key]
            if arr.ndim != 4:
                errors.append(f"{key} should be 4D [N,H,W,C], got {arr.ndim}D")
        elif key.endswith("_labels"):
            arr = data[key]
            if arr.ndim != 1:
                errors.append(f"{key} should be 1D, got {arr.ndim}D")
        elif key.endswith("_is_synth"):
            arr = data[key]
            if arr.ndim != 1:
                errors.append(f"{key} should be 1D, got {arr.ndim}D")
            if arr.dtype != bool:
                errors.append(f"{key} should be bool, got {arr.dtype}")

    return errors
