# medsyn/adapters/medfusion/generate.py
"""
MedFusion generation wrapper for MedSyn.

This module provides generation capabilities using trained MedFusion models
to create synthetic images that can be saved in MedSyn's unified NPZ format.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from medsyn.adapters.medfusion.config import MedFusionConfig, load_medfusion_config
from medsyn.adapters.medfusion.train import build_pipeline
from medsyn.data.npz_format import (
    create_unified_npz,
    create_metadata,
    merge_synthetic_data,
)

# Import MedFusion components
from medsyn.models.medfusion.medical_diffusion.models.pipelines.diffusion_pipeline import DiffusionPipeline

logger = logging.getLogger(__name__)


def load_pipeline(
    checkpoint_path: str,
    cfg: MedFusionConfig,
    device: torch.device,
) -> DiffusionPipeline:
    """
    Load trained MedFusion pipeline from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: MedFusion configuration
        device: Device to load model on

    Returns:
        DiffusionPipeline loaded from checkpoint
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)

    # Build pipeline with same config
    pipeline = build_pipeline(cfg)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        pipeline.load_state_dict(checkpoint["state_dict"])
    else:
        pipeline.load_state_dict(checkpoint)

    pipeline = pipeline.to(device)
    pipeline.eval()

    return pipeline


def generate_samples(
    pipeline: DiffusionPipeline,
    num_samples: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
    guidance_scale: float = 1.0,
    samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic images using MedFusion pipeline.

    Args:
        pipeline: Trained MedFusion pipeline
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        image_size: Image size
        num_classes: Number of classes
        device: Device to use
        guidance_scale: Classifier-free guidance scale
        samples_per_class: If specified, generate balanced samples per class

    Returns:
        Tuple of (images, labels):
        - images: [N, H, W, C] uint8 array
        - labels: [N] int64 array
    """
    all_images = []
    all_labels = []

    # Calculate samples per class for balanced generation
    if samples_per_class is not None:
        total_balanced = samples_per_class * num_classes
        if total_balanced < num_samples:
            logger.warning(
                "Requested %d samples but samples_per_class=%d * num_classes=%d = %d",
                num_samples, samples_per_class, num_classes, total_balanced,
            )
            num_samples = total_balanced

    num_batches = (num_samples + batch_size - 1) // batch_size

    logger.info("Generating %d samples in %d batches...", num_samples, num_batches)

    with torch.no_grad():
        generated = 0
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            current_batch_size = min(batch_size, num_samples - generated)

            # Generate class labels for this batch
            if samples_per_class is not None:
                # Balanced generation
                labels = torch.tensor([
                    (generated + i) // samples_per_class % num_classes
                    for i in range(current_batch_size)
                ]).to(device)
            else:
                # Random labels
                labels = torch.randint(0, num_classes, (current_batch_size,)).to(device)

            # Generate samples using pipeline's sample method
            img_size = (3, image_size, image_size)
            samples = pipeline.sample(
                num_samples=current_batch_size,
                img_size=img_size,
                condition=labels,
                guidance_scale=guidance_scale,
            )

            # Convert from [-1, 1] to [0, 255] uint8
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
            samples = samples.clamp(0, 1)
            samples = (samples * 255).to(torch.uint8)

            # Move to CPU and convert to [N, H, W, C]
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_images.append(samples)
            all_labels.append(labels_np)
            generated += current_batch_size

    images = np.concatenate(all_images, axis=0)[:num_samples]
    labels = np.concatenate(all_labels)[:num_samples]

    return images, labels


def generate_and_save(
    config_path: str,
    output_path: Optional[str] = None,
    num_samples: int = 1000,
    samples_per_class: Optional[int] = None,
    save_grid: bool = True,
) -> Path:
    """
    Generate synthetic images and save in unified NPZ format.

    Args:
        config_path: Path to YAML config with checkpoint_path
        output_path: Path for output NPZ (default: {output_dir}/synthetic.npz)
        num_samples: Number of samples to generate
        samples_per_class: If specified, generate balanced samples per class
        save_grid: Whether to save a grid of samples

    Returns:
        Path to output NPZ file
    """
    # Load config
    cfg = load_medfusion_config(config_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be specified in config")

    pipeline = load_pipeline(cfg.checkpoint_path, cfg, device)

    # Generate samples
    images, labels = generate_samples(
        pipeline=pipeline,
        num_samples=num_samples,
        batch_size=cfg.training.batch_size,
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
        device=device,
        guidance_scale=cfg.guidance_scale,
        samples_per_class=samples_per_class,
    )

    logger.info("Generated %d samples", len(images))

    # Save grid of samples
    if save_grid:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "generated_grid.png"

        # Convert to tensor for save_image
        grid_images = torch.from_numpy(images[:64]).permute(0, 3, 1, 2).float() / 255
        save_image(grid_images, grid_path, nrow=8, normalize=False)
        logger.info("Saved sample grid: %s", grid_path)

    # Save as NPZ
    if output_path is None:
        output_path = Path(cfg.output_dir) / "synthetic.npz"
    else:
        output_path = Path(output_path)

    # Create synthetic-only NPZ
    splits = {
        "train": {
            "images": images,
            "labels": labels,
            "is_synth": np.ones(len(images), dtype=bool),
        }
    }

    metadata = create_metadata(
        dataset_name=cfg.dataset_name,
        image_size=cfg.image_size,
        extra={
            "source": "medfusion",
            "checkpoint": str(cfg.checkpoint_path),
            "guidance_scale": cfg.guidance_scale,
            "num_samples": num_samples,
        },
    )

    create_unified_npz(splits, metadata, output_path)
    logger.info("Saved synthetic NPZ: %s", output_path)

    return output_path


def augment_dataset(
    config_path: str,
    original_npz_path: str,
    output_path: str,
    num_samples: int = 1000,
    samples_per_class: Optional[int] = None,
) -> Path:
    """
    Augment an existing dataset with synthetic samples.

    Args:
        config_path: Path to YAML config with checkpoint_path
        original_npz_path: Path to original dataset NPZ
        output_path: Path for augmented NPZ
        num_samples: Number of synthetic samples to add
        samples_per_class: If specified, generate balanced samples per class

    Returns:
        Path to augmented NPZ file
    """
    # Load config
    cfg = load_medfusion_config(config_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be specified in config")

    pipeline = load_pipeline(cfg.checkpoint_path, cfg, device)

    # Generate samples
    images, labels = generate_samples(
        pipeline=pipeline,
        num_samples=num_samples,
        batch_size=cfg.training.batch_size,
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
        device=device,
        guidance_scale=cfg.guidance_scale,
        samples_per_class=samples_per_class,
    )

    logger.info("Generated %d synthetic samples", len(images))

    # Merge with original dataset
    output = merge_synthetic_data(
        original_npz_path=original_npz_path,
        synthetic_images=images,
        synthetic_labels=labels,
        output_path=output_path,
        split="train",
    )

    logger.info("Created augmented dataset: %s", output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate images with trained MedFusion")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output", type=str, default=None, help="Output NPZ path")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--samples-per-class", type=int, default=None, help="Samples per class")
    parser.add_argument("--augment", type=str, default=None, help="Original NPZ to augment")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    if args.augment:
        augment_dataset(
            args.config,
            args.augment,
            args.output or "augmented.npz",
            args.num_samples,
            args.samples_per_class,
        )
    else:
        generate_and_save(
            args.config,
            args.output,
            args.num_samples,
            args.samples_per_class,
        )
