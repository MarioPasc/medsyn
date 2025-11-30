#!/usr/bin/env python3
"""
Compress generated PNG images from a DistDiff split directory into NPZ format.

This script converts the directory-based PNG output from DistDiff generation into
a compressed NPZ file matching the medsyn data format (medsyn/cli/data.py).

Output NPZ structure:
    - images: [N, H, W, C] uint8 array
    - labels: [N] int64 array
    - is_synth: [N] bool array (all True for synthetic data)
    - class_names: [num_classes] array of class name strings
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compress_split_to_npz(
    split_dir: Path,
    output_npz: Path,
    delete_pngs: bool = False
) -> Dict[str, int]:
    """
    Compress PNG files from a split directory into NPZ format.

    Args:
        split_dir: Directory containing class subdirectories with PNG files
        output_npz: Output NPZ file path
        delete_pngs: If True, delete PNG files after successful compression

    Returns:
        Dictionary with statistics (num_images, num_classes, etc.)
    """
    logger.info("="*80)
    logger.info(f"Compressing split directory to NPZ")
    logger.info("="*80)
    logger.info(f"Input directory: {split_dir}")
    logger.info(f"Output NPZ: {output_npz}")
    logger.info("="*80)

    # Validate input directory exists
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Find all class directories
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found in {split_dir}")

    logger.info(f"Found {len(class_dirs)} class directories:")
    for class_dir in class_dirs:
        num_pngs = len(list(class_dir.glob("*.png")))
        logger.info(f"  • {class_dir.name}: {num_pngs} images")

    # Create class name to index mapping
    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}
    class_names = [d.name for d in class_dirs]

    # Collect all images, labels, and is_synth flags
    images_list: List[np.ndarray] = []
    labels_list: List[int] = []
    is_synth_list: List[bool] = []

    total_pngs = sum(len(list(d.glob("*.png"))) for d in class_dirs)
    logger.info(f"Processing {total_pngs} images...")

    with tqdm(total=total_pngs, desc="Loading images") as pbar:
        for class_dir in class_dirs:
            label = class_to_idx[class_dir.name]
            png_files = sorted(class_dir.glob("*.png"))

            for png_file in png_files:
                try:
                    # Load image
                    img = Image.open(png_file)
                    img_array = np.array(img)  # [H, W, C] or [H, W]

                    # Ensure [H, W, C] format
                    if img_array.ndim == 2:
                        # Grayscale -> add channel dimension
                        img_array = img_array[..., np.newaxis]

                    images_list.append(img_array)
                    labels_list.append(label)
                    is_synth_list.append(True)  # All synthetic

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Failed to load {png_file}: {e}")
                    raise

    # Convert to numpy arrays
    logger.info("Converting to numpy arrays...")
    images = np.stack(images_list, axis=0).astype(np.uint8)  # [N, H, W, C]
    labels = np.array(labels_list, dtype=np.int64)  # [N]
    is_synth = np.array(is_synth_list, dtype=bool)  # [N]
    class_names_array = np.array(class_names, dtype=object)

    logger.info(f"Data shapes:")
    logger.info(f"  images: {images.shape} {images.dtype}")
    logger.info(f"  labels: {labels.shape} {labels.dtype}")
    logger.info(f"  is_synth: {is_synth.shape} {is_synth.dtype}")
    logger.info(f"  class_names: {class_names_array.shape}")

    # Save NPZ file
    logger.info(f"Saving compressed NPZ to {output_npz}...")
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(output_npz),
        images=images,
        labels=labels,
        is_synth=is_synth,
        class_names=class_names_array
    )

    # Verify saved file
    logger.info("Verifying saved NPZ file...")
    verification = np.load(str(output_npz))
    logger.info(f"NPZ keys: {list(verification.keys())}")
    for key in verification.keys():
        logger.info(f"  {key}: {verification[key].shape} {verification[key].dtype}")
    verification.close()

    # Get file size
    file_size_mb = output_npz.stat().st_size / (1024 * 1024)
    logger.info(f"NPZ file size: {file_size_mb:.2f} MB")

    # Delete PNGs if requested
    if delete_pngs:
        logger.info("Deleting original PNG files...")
        try:
            shutil.rmtree(split_dir)
            logger.info(f"✓ Deleted directory: {split_dir}")
        except Exception as e:
            logger.warning(f"Failed to delete {split_dir}: {e}")

    stats = {
        'num_images': len(images),
        'num_classes': len(class_names),
        'image_shape': images.shape[1:],
        'file_size_mb': file_size_mb
    }

    logger.info("="*80)
    logger.info("✅ Compression complete")
    logger.info("="*80)
    logger.info(f"Total images: {stats['num_images']}")
    logger.info(f"Total classes: {stats['num_classes']}")
    logger.info(f"Image shape: {stats['image_shape']}")
    logger.info(f"File size: {stats['file_size_mb']:.2f} MB")
    logger.info("="*80)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compress DistDiff PNG output to NPZ format"
    )
    parser.add_argument(
        '--split_dir',
        type=str,
        required=True,
        help='Directory containing class subdirectories with PNG files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output NPZ file path'
    )
    parser.add_argument(
        '--delete_pngs',
        action='store_true',
        help='Delete PNG files after successful compression'
    )

    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    output_npz = Path(args.output)

    try:
        stats = compress_split_to_npz(split_dir, output_npz, args.delete_pngs)
        return 0
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
