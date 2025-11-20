#!/usr/bin/env python3
"""
Visualization script for data augmentations applied to PathMNIST dataset.

This script loads augmentation configuration from config/medsyn_cfg.yaml and
visualizes each augmentation type applied to a random sample from a user-specified class.

Usage:
    python -m medsyn.utils.visualize_augmentations --class_id 0 --config config/medsyn_cfg.yaml
    python -m medsyn.utils.visualize_augmentations --class_id 5 --config config/medsyn_cfg.yaml --seed 42
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# Import config and augmentation modules
from medsyn.models.ccDDPM.config import load_cfg
from medsyn.models.ccDDPM.augmentation import AugmentationConfig, TransformConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PathMNISTClass:
    """PathMNIST class names for visualization."""
    LABELS = {
        0: "adipose",
        1: "background",
        2: "debris",
        3: "lymphocytes",
        4: "mucus",
        5: "smooth muscle",
        6: "normal colon mucosa",
        7: "cancer-associated stroma",
        8: "colorectal adenocarcinoma epithelium"
    }


def load_pathmnist_sample(npz_path: Path, class_id: int, seed: int = None) -> Tuple[np.ndarray, int]:
    """
    Load a random sample from PathMNIST dataset for a specific class.

    Args:
        npz_path: Path to the PathMNIST NPZ file
        class_id: Class ID to sample from (0-8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (image, label) where image is [H, W, C] uint8
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    logger.info(f"Loading PathMNIST from {npz_path}")
    data = np.load(npz_path)

    # Load training split
    images = data['train_images']  # [N, H, W, C]
    labels = data['train_labels'].reshape(-1)  # [N]

    logger.info(f"Loaded {len(images)} training images")
    logger.info(f"Image shape: {images.shape}, dtype: {images.dtype}")
    logger.info(f"Unique classes: {np.unique(labels)}")

    # Find samples for the specified class
    class_indices = np.where(labels == class_id)[0]

    if len(class_indices) == 0:
        raise ValueError(f"No samples found for class {class_id}")

    logger.info(f"Found {len(class_indices)} samples for class {class_id} ({PathMNISTClass.LABELS.get(class_id, 'unknown')})")

    # Sample randomly from the class
    if seed is not None:
        np.random.seed(seed)

    sample_idx = np.random.choice(class_indices)
    image = images[sample_idx]
    label = labels[sample_idx]

    logger.info(f"Selected sample index {sample_idx} with label {label}")

    return image, int(label)


def create_single_transform_pipeline(transform_config: TransformConfig) -> A.Compose:
    """
    Create an albumentations pipeline with a single transform forced to apply (p=1.0).

    Args:
        transform_config: Configuration for the transform

    Returns:
        Albumentations Compose pipeline
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required. Install with: pip install albumentations")

    transform_class = getattr(A, transform_config.name, None)
    if transform_class is None:
        raise ValueError(f"Unknown transform: {transform_config.name}")

    # Create transform with p=1.0 to force application
    import inspect
    valid_params = inspect.signature(transform_class.__init__).parameters
    filtered_params = {k: v for k, v in transform_config.params.items() if k in valid_params}

    # For Rotate, ensure dimensions are preserved
    if transform_config.name == 'Rotate':
        import cv2
        if 'border_mode' not in filtered_params:
            filtered_params['border_mode'] = cv2.BORDER_CONSTANT

    transform = transform_class(p=1.0, **filtered_params)

    return A.Compose([transform])


def apply_augmentation(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """
    Apply augmentation pipeline to an image.

    Args:
        image: Input image [H, W, C] in range [0, 1] (float32)
        pipeline: Albumentations pipeline

    Returns:
        Augmented image [H, W, C] in range [0, 1]
    """
    # Albumentations expects float32 in [0, 1]
    augmented = pipeline(image=image)
    return augmented['image']


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor [C, H, W] in [-1, 1] to numpy [H, W, C] in [0, 1]."""
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np + 1.0) / 2.0
    return np.clip(image_np, 0.0, 1.0).astype(np.float32)


def numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert numpy [H, W, C] in [0, 1] to tensor [C, H, W] in [-1, 1]."""
    tensor = torch.from_numpy(array).float()
    tensor = tensor.permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def visualize_augmentations(
    image: np.ndarray,
    label: int,
    augmentation_config: AugmentationConfig,
    save_path: Path = None
) -> None:
    """
    Visualize all augmentation types applied to a single image.

    Args:
        image: Input image [H, W, C] uint8
        label: Class label
        augmentation_config: Augmentation configuration with transforms
        save_path: Optional path to save the visualization
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations is required. Install with: pip install albumentations")

    # Convert to float32 in [0, 1]
    image_float = image.astype(np.float32) / 255.0

    # Prepare visualizations
    num_transforms = len(augmentation_config.transforms)
    num_cols = min(5, num_transforms + 1)  # +1 for original
    num_rows = (num_transforms + 1 + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))

    # Flatten axes for easier indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Display original image
    axes[0].imshow(image_float)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Apply and display each augmentation
    for idx, transform_config in enumerate(augmentation_config.transforms, start=1):
        try:
            # Create pipeline with single transform (p=1.0)
            pipeline = create_single_transform_pipeline(transform_config)

            # Apply augmentation
            augmented = apply_augmentation(image_float.copy(), pipeline)

            # Display augmented image
            axes[idx].imshow(augmented)

            # Create title with transform name and parameters
            title = f"{transform_config.name}"
            if transform_config.params:
                # Show key parameters
                param_str = ", ".join([f"{k}={v}" for k, v in list(transform_config.params.items())[:2]])
                title += f"\n({param_str})"

            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')

        except Exception as e:
            logger.error(f"Failed to apply {transform_config.name}: {e}")
            axes[idx].text(0.5, 0.5, f"Error:\n{transform_config.name}",
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_transforms + 1, len(axes)):
        axes[idx].axis('off')

    # Add overall title
    class_name = PathMNISTClass.LABELS.get(label, f"Class {label}")
    fig.suptitle(
        f'PathMNIST Augmentations - {class_name}',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # Add config info
    info_text = (
        f"Config probability: {augmentation_config.probability:.2f} "
        f"| Transforms: {num_transforms} "
        f"| Note: Each transform shown with p=1.0 for visualization"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize data augmentations on PathMNIST samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/medsyn_cfg.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--class_id',
        type=int,
        required=True,
        choices=range(9),
        help='Class ID to sample from (0-8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the visualization (e.g., outputs/augmentations_class0.png)'
    )

    args = parser.parse_args()

    # Display class information
    logger.info("\nPathMNIST Classes:")
    for cid, cname in PathMNISTClass.LABELS.items():
        marker = " <-- Selected" if cid == args.class_id else ""
        logger.info(f"  {cid}: {cname}{marker}")

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    logger.info(f"\nLoading config from {config_path}")
    cfg = load_cfg(config_path)

    # Check if augmentation is configured
    if cfg.ccddpm.augmentation is None or not cfg.ccddpm.augmentation.enabled:
        logger.error("Augmentation is not enabled in the configuration!")
        logger.info("Please set ccddpm.augmentation.enabled: true in your config file")
        return

    aug_config = cfg.ccddpm.augmentation
    logger.info(f"Augmentation config loaded: {len(aug_config.transforms)} transforms")
    for tf in aug_config.transforms:
        logger.info(f"  - {tf.name} (p={tf.p}, params={tf.params})")

    # Load NPZ file path
    npz_path = cfg.ccddpm.dataloader.npz_path
    if npz_path is None:
        logger.error("NPZ path not found in configuration!")
        logger.info("Please set data.postprocess_npz.npz_path in your config file")
        return

    # Load sample
    logger.info(f"\nLoading sample from class {args.class_id}")
    image, label = load_pathmnist_sample(npz_path, args.class_id, args.seed)

    # Visualize augmentations
    logger.info("\nGenerating visualization...")
    save_path = Path(args.save) if args.save else None
    visualize_augmentations(image, label, aug_config, save_path)

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
