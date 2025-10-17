#!/usr/bin/env python3
# medsyn/cli/json_to_npz.py
# Purpose: Convert JSON index dataset to compressed NPZ format
# Usage: python -m medsyn.cli.json_to_npz --json path/to/index.json --output path/to/output.npz
"""
Convert a JSON-indexed dataset to NPZ format for efficient loading on supercomputers.

The NPZ file will contain:
- {split}_images: [N, H, W, C] uint8 arrays
- {split}_labels: [N] int64 arrays
- {split}_is_synth: [N] bool arrays

for each split (train, val, test).
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from tqdm import tqdm

# Pillow 10+ moved resampling filters to Image.Resampling; keep compatibility with older versions
try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9.1
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR  # Pillow < 10

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_split_from_json(
    json_data: Dict,
    dataset_name: str,
    split: str,
    target_size: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load images and metadata for a single split.
    
    Args:
        json_data: Loaded JSON index
        dataset_name: Dataset name (e.g., "PathMNIST")
        split: Split name ("train", "val", "test")
        target_size: Optional (H, W) to resize all images to
    
    Returns:
        images: [N, H, W, C] uint8
        labels: [N] int64
        is_synth: [N] bool
    """
    if dataset_name not in json_data:
        raise KeyError(f"Dataset '{dataset_name}' not found in JSON. Available: {list(json_data.keys())}")
    
    if split not in json_data[dataset_name]:
        raise KeyError(f"Split '{split}' not found in dataset '{dataset_name}'. Available: {list(json_data[dataset_name].keys())}")
    
    split_data = json_data[dataset_name][split]
    logger.info(f"Loading split '{split}' with {len(split_data)} samples...")
    
    images_list = []
    labels_list = []
    is_synth_list = []
    
    for idx, (key, record) in enumerate(tqdm(split_data.items(), desc=f"Loading {split}")):
        img_path = Path(record["image"])
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}, skipping...")
            continue

        # Load and convert image
        img = Image.open(img_path).convert("RGB")

        # Resize if needed
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, RESAMPLE_BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)  # [H, W, C]
        # Resize if needed
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.uint8)  # [H, W, C]
        
        images_list.append(img_array)
        labels_list.append(int(record["label"]))
        is_synth_list.append(bool(record.get("is_synth", False)))
    
    # Stack into arrays
    images = np.stack(images_list, axis=0)  # [N, H, W, C]
    labels = np.array(labels_list, dtype=np.int64)
    is_synth = np.array(is_synth_list, dtype=bool)
    
    logger.info(f"  Loaded {len(images)} images with shape {images.shape}")
    logger.info(f"  Labels: {np.unique(labels).tolist()}")
    logger.info(f"  Synthetic samples: {is_synth.sum()}/{len(is_synth)}")
    
    return images, labels, is_synth


def json_to_npz(
    json_path: Path,
    output_path: Path,
    dataset_name: str = "PathMNIST",
    splits: List[str] = ["train", "val", "test"],
    target_size: tuple[int, int] | None = None
) -> None:
    """
    Convert JSON index to NPZ format.
    
    Args:
        json_path: Path to JSON index file
        output_path: Path to output .npz file
        dataset_name: Name of dataset in JSON (default: "PathMNIST")
        splits: List of splits to include
        target_size: Optional (H, W) to resize images to
    """
    logger.info(f"Loading JSON index from: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Prepare NPZ data dictionary
    npz_data = {}
    
    for split in splits:
        try:
            images, labels, is_synth = load_split_from_json(
                json_data, dataset_name, split, target_size
            )
            
            npz_data[f"{split}_images"] = images
            npz_data[f"{split}_labels"] = labels
            npz_data[f"{split}_is_synth"] = is_synth
            
        except KeyError as e:
            logger.warning(f"Split '{split}' not found, skipping: {e}")
            continue
    
    # Save to NPZ
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving NPZ to: {output_path}")
    np.savez_compressed(output_path, **npz_data)
    
    # Print summary
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Successfully created NPZ file ({file_size_mb:.2f} MB)")
    logger.info(f"  Splits included: {[k.replace('_images', '') for k in npz_data.keys() if k.endswith('_images')]}")
    
    # Verify by loading
    logger.info("Verifying NPZ file...")
    loaded = np.load(output_path)
    logger.info(f"  Keys in NPZ: {list(loaded.keys())}")
    for key in loaded.keys():
        if key.endswith("_images"):
            logger.info(f"  {key}: {loaded[key].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON-indexed dataset to NPZ format for efficient loading"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to JSON index file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output NPZ file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PathMNIST",
        help="Dataset name in JSON (default: PathMNIST)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to include (default: train val test)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        help="Optional: resize all images to (H, W)"
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json)
    output_path = Path(args.output)
    target_size = tuple(args.resize) if args.resize else None
    
    if not json_path.exists():
        logger.error(f"JSON file not found: {json_path}")
        return 1
    
    json_to_npz(
        json_path=json_path,
        output_path=output_path,
        dataset_name=args.dataset,
        splits=args.splits,
        target_size=target_size
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
