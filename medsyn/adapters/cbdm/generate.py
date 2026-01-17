# medsyn/adapters/cbdm/generate.py
"""
CBDM generation wrapper for MedSyn.

This module provides generation capabilities using trained CBDM models
to create synthetic images that can be saved in MedSyn's unified NPZ format.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from medsyn.adapters.cbdm.config import CBDMConfig, load_cbdm_config
from medsyn.adapters.cbdm.dataset import CBDMGenerationDataset
from medsyn.data.npz_format import (
    load_npz_with_metadata,
    create_unified_npz,
    create_metadata,
    merge_synthetic_data,
)

# Import CBDM components
from medsyn.models.cbdm.model.model import UNet
from medsyn.models.cbdm.diffusion import GaussianDiffusionSampler

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[UNet, GaussianDiffusionSampler, CBDMConfig]:
    """
    Load trained CBDM model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, sampler, config)
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    cfg = checkpoint.get("config")
    if cfg is None:
        raise ValueError("Checkpoint does not contain config")

    # Build model
    model = UNet(
        T=cfg.T,
        ch=cfg.ch,
        ch_mult=cfg.ch_mult,
        attn=cfg.attn,
        num_res_blocks=cfg.num_res_blocks,
        dropout=cfg.dropout,
        cond=True,
        augm=False,
        num_class=cfg.num_classes,
    ).to(device)

    # Load EMA weights if available, otherwise use model weights
    if "ema" in checkpoint:
        logger.info("Loading EMA weights")
        for name, param in model.named_parameters():
            if name in checkpoint["ema"]:
                param.data = checkpoint["ema"][name].to(device)
    else:
        model.load_state_dict(checkpoint["model"])

    model.eval()

    # Build sampler
    sampler = GaussianDiffusionSampler(
        model=model,
        beta_1=cfg.beta_1,
        beta_T=cfg.beta_T,
        T=cfg.T,
        num_class=cfg.num_classes,
        img_size=cfg.image_size,
    ).to(device)

    return model, sampler, cfg


def generate_samples(
    sampler: GaussianDiffusionSampler,
    num_samples: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    device: torch.device,
    omega: float = 0.0,
    method: str = "cfg",
    samples_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic images using CBDM sampler.

    Args:
        sampler: Trained CBDM sampler
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        image_size: Image size
        num_classes: Number of classes
        device: Device to use
        omega: Classifier-free guidance strength
        method: Generation method ('cfg' or 'uncond')
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

            # Generate noise
            noise = torch.randn(
                current_batch_size, 3, image_size, image_size
            ).to(device)

            # Generate samples
            samples, labels = sampler(noise, omega=omega, method=method)

            # Convert to numpy and rescale to [0, 255]
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
            samples = samples.clamp(0, 1)
            samples = (samples * 255).to(torch.uint8)

            # Move to CPU and convert to [N, H, W, C]
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_images.append(samples)
            all_labels.append(labels)
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
    cfg = load_cbdm_config(config_path)

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load model
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be specified in config")

    model, sampler, model_cfg = load_model(cfg.checkpoint_path, device)

    # Generate samples
    images, labels = generate_samples(
        sampler=sampler,
        num_samples=num_samples,
        batch_size=cfg.batch_size,
        image_size=model_cfg.image_size,
        num_classes=model_cfg.num_classes,
        device=device,
        omega=cfg.omega,
        method=cfg.method,
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
        dataset_name=model_cfg.dataset_name,
        image_size=model_cfg.image_size,
        extra={
            "source": "cbdm",
            "checkpoint": str(cfg.checkpoint_path),
            "omega": cfg.omega,
            "method": cfg.method,
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
    cfg = load_cbdm_config(config_path)

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Load model
    if not cfg.checkpoint_path:
        raise ValueError("checkpoint_path must be specified in config")

    model, sampler, model_cfg = load_model(cfg.checkpoint_path, device)

    # Generate samples
    images, labels = generate_samples(
        sampler=sampler,
        num_samples=num_samples,
        batch_size=cfg.batch_size,
        image_size=model_cfg.image_size,
        num_classes=model_cfg.num_classes,
        device=device,
        omega=cfg.omega,
        method=cfg.method,
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

    parser = argparse.ArgumentParser(description="Generate images with trained CBDM")
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
