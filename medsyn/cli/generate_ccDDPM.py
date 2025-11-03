#!/usr/bin/env python3
"""
medsyn/cli/generate_ccDDPM.py

CLI for generating synthetic images using trained ccDDPM models.

Features:
- Per-class image generation based on config
- Organized output (one folder per class)
- Proper naming: synth_<uuid>_class<N>.png
- JSON index generation (compatible with _build_index_structure format)
- Denoising process visualization for random samples per class
- EMA weights support for higher quality
- Batch processing with progress tracking

Author: M.Pascual-González
Date: 2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.figure import Figure
from torchvision.utils import save_image
from tqdm import tqdm

from medsyn.models.ccDDPM.config import load_cfg, ProjectCfg
from medsyn.models.ccDDPM.model import CCDDPM, CCDDPMInit
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_generation_config(config_path: Path) -> tuple[dict[str, dict[int, int]], Path, Path]:
    """
    Parse the generation configuration from YAML.

    Expected format:
        generate:
          checkpoint: /absolute/path/to/best.pt
          npz_with_synth_images:
            save_to: /path/to/output
            train:
              classes:
                0: 100
                1: 50
                ...
            val:
              classes:
                0: 50
                1: 25
                ...

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (split_to_class_to_samples_dict, checkpoint_path, output_base_path)

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If checkpoint doesn't exist
    """
    logger.info("Parsing generation configuration from: %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "generate" not in config:
        raise ValueError("'generate' section not found in configuration file")

    gen_config = config["generate"]

    # Parse checkpoint path
    if "checkpoint" not in gen_config:
        raise ValueError("'checkpoint' path not specified in generate section")

    checkpoint_path = Path(gen_config["checkpoint"]).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    logger.info("Found checkpoint file: %s", checkpoint_path)

    # Parse npz_with_synth_images section
    if "npz_with_synth_images" not in gen_config:
        raise ValueError("'npz_with_synth_images' section not found in generate section")

    npz_config = gen_config["npz_with_synth_images"]

    # Parse save_to path
    if "save_to" not in npz_config:
        raise ValueError("'save_to' path not specified in npz_with_synth_images section")

    output_base_path = Path(npz_config["save_to"]).expanduser().resolve()
    
    logger.info("Output base path: %s", output_base_path)

    # Parse splits (train, val, test)
    split_to_class_to_samples: dict[str, dict[int, int]] = {}

    for split_name in ["train", "val", "test"]:
        if split_name not in npz_config:
            logger.debug(f"Split '{split_name}' not found in config, skipping...")
            continue

        split_config = npz_config[split_name]
        if "classes" not in split_config:
            logger.warning(f"'classes' not found in {split_name} section, skipping...")
            continue

        classes_dict = split_config["classes"]
        if not isinstance(classes_dict, dict):
            raise ValueError(f"'classes' in {split_name} must be a dictionary mapping class_id -> num_samples")

        # Convert to int keys and validate
        class_to_samples: dict[int, int] = {}
        for class_id, num_samples in classes_dict.items():
            class_id_int = int(class_id)
            num_samples_int = int(num_samples)
            if num_samples_int <= 0:
                raise ValueError(
                    f"Number of samples must be positive, got {num_samples_int} for class {class_id_int} in split {split_name}"
                )
            class_to_samples[class_id_int] = num_samples_int

        if class_to_samples:
            split_to_class_to_samples[split_name] = class_to_samples
            logger.info("Loaded split '%s' with %d classes and %d total samples", 
                       split_name, len(class_to_samples), sum(class_to_samples.values()))

    if not split_to_class_to_samples:
        raise ValueError("No valid splits/classes specified in generate.npz_with_synth_images")
    
    logger.info("Configuration parsed successfully: %d splits, %d total classes",
               len(split_to_class_to_samples),
               sum(len(classes) for classes in split_to_class_to_samples.values()))

    return split_to_class_to_samples, checkpoint_path, output_base_path


def load_model_and_scheduler(
    config_path: str | Path, checkpoint_path: Path, device: torch.device
) -> tuple[CCDDPM, DDPMScheduler, ProjectCfg]:
    """
    Load the ccDDPM model and scheduler from checkpoint.

    Args:
        config_path: Path to YAML config
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on

    Returns:
        Tuple of (model, scheduler, config)
    """
    logger.info("Loading configuration from %s", config_path)
    cfg: ProjectCfg = load_cfg(str(config_path), split="train")
    tcfg, scfg, icfg = cfg.ccddpm.train, cfg.ccddpm.sched, cfg.ccddpm.infer
    
    logger.info("Configuration loaded - Model: %d classes, Image size: %dx%d, Channels: %d",
               tcfg.num_classes, tcfg.image_size, tcfg.image_size, tcfg.in_channels)

    # Initialize model
    mcfg = CCDDPMInit(
        in_channels=tcfg.in_channels,
        class_embed_dim=tcfg.class_embed_dim,
        num_classes=tcfg.num_classes,
    )
    model = CCDDPM(mcfg).to(device)

    # Load checkpoint
    logger.info("Loading checkpoint from %s", checkpoint_path)
    logger.debug("Checkpoint file size: %.2f MB", checkpoint_path.stat().st_size / (1024 * 1024))
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if EMA weights are available (better quality)
    # EMA dict must exist, not be None, and contain weights (not be empty)
    if "ema" in state and state["ema"] is not None and len(state["ema"]) > 0:
        logger.info("Using EMA weights for generation (higher quality)")
        logger.debug("EMA weights available - Number of parameters: %d", len(state["ema"]))
        missing_keys, unexpected_keys = model.load_state_dict(state["ema"], strict=False)
        if missing_keys:
            logger.warning("Missing keys when loading EMA weights: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading EMA weights: %s", unexpected_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch detected! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}. "
                "Ensure you're loading a ccDDPM checkpoint (not bVAE or other model type)."
            )
        logger.info("✓ EMA weights loaded successfully with no key mismatches")
    else:
        if "ema" in state and len(state["ema"]) == 0:
            logger.info("EMA dict exists but is empty, falling back to standard model weights")
        else:
            logger.info("Using standard model weights")
        logger.debug("Standard model weights - Number of parameters: %d", len(state["model"]))
        missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=False)
        if missing_keys:
            logger.warning("Missing keys when loading model weights: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected keys when loading model weights: %s", unexpected_keys)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch detected! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}. "
                "Ensure you're loading a ccDDPM checkpoint (not bVAE or other model type)."
            )
        logger.info("✓ Model weights loaded successfully with no key mismatches")

    model.eval()
    logger.debug("Model set to evaluation mode")

    # Initialize scheduler for inference
    scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=True,            # <- critical
        clip_sample_range=1.0,
        thresholding=False,
    )
    scheduler.set_timesteps(icfg.num_inference_steps, device=device)
    
    logger.debug("Scheduler initialized with %d timesteps", len(scheduler.timesteps))

    # Log scheduler configuration to verify it matches training
    logger.info("=" * 60)
    logger.info("Scheduler Configuration (must match training):")
    logger.info("  num_train_timesteps: %d", scfg.num_train_timesteps)
    logger.info("  beta_schedule: %s", scfg.beta_schedule)
    logger.info("  beta_start: %.6f", scfg.beta_start)
    logger.info("  beta_end: %.6f", scfg.beta_end)
    logger.info("  prediction_type: %s", scfg.prediction_type)
    logger.info("  num_inference_steps: %d", icfg.num_inference_steps)
    logger.info("  guidance_scale: %.2f", icfg.guidance_scale)
    logger.info("=" * 60)

    logger.info(
        "Model loaded: %d classes, %dx%d images",
        tcfg.num_classes,
        tcfg.image_size,
        tcfg.image_size,
    )

    return model, scheduler, cfg


@torch.no_grad()
def generate_with_cfg(
    model: CCDDPM,
    scheduler: DDPMScheduler,
    class_label: int,
    image_size: int,
    in_channels: int,
    guidance_scale: float,
    device: torch.device,
    debug: bool = False,
) -> torch.Tensor:
    """
    Generate a single image using classifier-free guidance.

    Args:
        model: The ccDDPM model
        scheduler: DDPM scheduler
        class_label: Class label for conditional generation
        image_size: Image size (H=W)
        in_channels: Number of input channels
        guidance_scale: CFG scale (1.0 = pure conditional, >1.0 = enhanced conditional)
        device: Device
        debug: Enable debug logging

    Returns:
        Generated image tensor [1, C, H, W] in range [-1, 1]
    """
    # Start from pure noise
    x_t = torch.randn((1, in_channels, image_size, image_size), device=device)
    labels = torch.tensor([class_label], device=device, dtype=torch.long)

    if debug:
        logger.debug(f"Starting generation for class {class_label}")
        logger.debug(f"  Initial noise: shape={x_t.shape}, range=[{x_t.min():.3f}, {x_t.max():.3f}]")
        logger.debug(f"  Scheduler timesteps: {len(scheduler.timesteps)} steps")
        logger.debug(f"  Guidance scale: {guidance_scale}")

    # Denoising loop
    for step_idx, t in enumerate(scheduler.timesteps):
        t_batch = t.unsqueeze(0) if t.dim() == 0 else t

        # Classifier-free guidance
        if guidance_scale != 1.0:
            eps_cond = model(x_t, t_batch, labels)
            eps_uncond = model(x_t, t_batch, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            if debug and step_idx % 200 == 0:
                logger.debug(f"  Step {step_idx}/{len(scheduler.timesteps)}: t={t.item()}, " +
                           f"eps_cond_range=[{eps_cond.min():.3f}, {eps_cond.max():.3f}], " +
                           f"eps_uncond_range=[{eps_uncond.min():.3f}, {eps_uncond.max():.3f}]")
        else:
            eps = model(x_t, t_batch, labels)

            if debug and (step_idx < 5 or step_idx % 200 == 0):
                logger.debug(f"  Step {step_idx}/{len(scheduler.timesteps)}: t={t.item()}, " +
                           f"x_t_range=[{x_t.min():.3f}, {x_t.max():.3f}], " +
                           f"eps_range=[{eps.min():.3f}, {eps.max():.3f}]")

        # Denoising step
        x_t = scheduler.step(model_output=eps, timestep=t, sample=x_t).prev_sample

        # Extra debug for first few steps
        if debug and step_idx < 5:
            logger.debug(f"    After step: x_t_range=[{x_t.min():.3f}, {x_t.max():.3f}]")

    if debug:
        logger.debug(f"  Final image: range=[{x_t.min():.3f}, {x_t.max():.3f}]")

    return x_t


@torch.no_grad()
def generate_with_denoising_steps(
    model: CCDDPM,
    scheduler: DDPMScheduler,
    class_label: int,
    image_size: int,
    in_channels: int,
    guidance_scale: float,
    device: torch.device,
    num_vis_steps: int = 10,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Generate an image while capturing intermediate denoising steps for visualization.

    Args:
        model: The ccDDPM model
        scheduler: DDPM scheduler
        class_label: Class label
        image_size: Image size
        in_channels: Number of channels
        guidance_scale: CFG scale
        device: Device
        num_vis_steps: Number of intermediate steps to capture

    Returns:
        Tuple of (final_image, list_of_intermediate_images)
    """
    x_t = torch.randn((1, in_channels, image_size, image_size), device=device)
    labels = torch.tensor([class_label], device=device, dtype=torch.long)
    
    logger.debug("Starting denoising visualization generation for class %d", class_label)

    # Determine which steps to save
    total_steps = len(scheduler.timesteps)
    save_indices = set(
        np.linspace(0, total_steps - 1, num_vis_steps, dtype=int).tolist()
    )
    save_indices.add(total_steps - 1)  # Always save the last step

    intermediate_steps: list[torch.Tensor] = [x_t.cpu().clone()]

    for i, t in enumerate(scheduler.timesteps):
        t_batch = t.unsqueeze(0) if t.dim() == 0 else t

        # Consistent with main generation function: skip second pass when scale=1.0
        if guidance_scale != 1.0:
            eps_cond = model(x_t, t_batch, labels)
            eps_uncond = model(x_t, t_batch, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x_t, t_batch, labels)

        x_t = scheduler.step(model_output=eps, timestep=t, sample=x_t).prev_sample

        if i in save_indices:
            intermediate_steps.append(x_t.cpu().clone())
    
    logger.debug("Denoising visualization completed - captured %d intermediate steps", len(intermediate_steps))

    return x_t, intermediate_steps


def create_denoising_visualization(
    intermediate_steps: list[torch.Tensor],
    class_label: int,
    output_path: Path,
) -> None:
    """
    Create a scientific visualization of the denoising process.

    Shows the progression from noise to final image in a clean, publication-ready format.

    Args:
        intermediate_steps: List of tensors [1, C, H, W] in range [-1, 1]
        class_label: The class label
        output_path: Path to save the visualization
    """
    num_steps = len(intermediate_steps)

    # Create figure with subplots
    fig: Figure = plt.figure(figsize=(20, 4))

    for idx, img_tensor in enumerate(intermediate_steps):
        # Convert from [-1, 1] to [0, 1]
        img = (img_tensor[0].clamp(-1, 1) + 1.0) / 2.0
        img_np = img.permute(1, 2, 0).numpy()

        ax = fig.add_subplot(1, num_steps, idx + 1)
        ax.imshow(img_np)
        ax.axis("off")

        # Add step label
        if idx == 0:
            ax.set_title("Pure Noise", fontsize=10, pad=5)
        elif idx == num_steps - 1:
            ax.set_title("Final Image", fontsize=10, pad=5, fontweight="bold")
        else:
            progress = int((idx / (num_steps - 1)) * 100)
            ax.set_title(f"{progress}%", fontsize=9, pad=5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Saved denoising visualization to %s", output_path)


def generate_images_for_class(
    model: CCDDPM,
    scheduler: DDPMScheduler,
    class_id: int,
    num_samples: int,
    output_dir: Path,
    config: ProjectCfg,
    device: torch.device,
    create_visualization: bool = False,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Generate synthetic images for a specific class.

    Args:
        model: The ccDDPM model
        scheduler: DDPM scheduler
        class_id: Class ID to generate
        num_samples: Number of samples to generate
        output_dir: Base output directory
        config: Project configuration
        device: Device
        create_visualization: Whether to create a denoising visualization

    Returns:
        Tuple of:
        - Dictionary mapping sample IDs to metadata (for JSON index)
        - NumPy array of images [N, H, W, C] uint8
        - NumPy array of labels [N] int64
    """
    tcfg = config.ccddpm.train
    icfg = config.ccddpm.infer

    # Create class-specific output directory
    class_dir = output_dir / f"class_{class_id}"
    class_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting generation for class %d: %d samples to %s", class_id, num_samples, class_dir)

    # Dictionary to store metadata for JSON index
    samples_metadata: dict[str, dict[str, Any]] = {}

    # Lists to collect images and labels for NPZ
    images_list: list[np.ndarray] = []
    labels_list: list[int] = []

    logger.debug("Generating %d images for class %d...", num_samples, class_id)

    # Randomly select one sample for visualization
    vis_sample_idx = np.random.randint(0, num_samples) if create_visualization else -1
    if create_visualization:
        logger.debug("Will create denoising visualization for sample index %d", vis_sample_idx)

    for idx in tqdm(range(num_samples), desc=f"Class {class_id}", unit="img"):
        # Generate unique ID
        sample_uuid = uuid.uuid4().hex[:12]
        filename = f"synth_{sample_uuid}_class{class_id}.png"
        file_path = class_dir / filename

        # Enable debug logging for first sample of first class
        enable_debug = (class_id == 0 and idx == 0)

        # Generate image (with or without visualization)
        if idx == vis_sample_idx:
            img, intermediate_steps = generate_with_denoising_steps(
                model=model,
                scheduler=scheduler,
                class_label=class_id,
                image_size=tcfg.image_size,
                in_channels=tcfg.in_channels,
                guidance_scale=icfg.guidance_scale,
                device=device,
                num_vis_steps=10,
            )

            # Create visualization
            vis_path = class_dir / f"denoising_process_class{class_id}_{sample_uuid}.png"
            create_denoising_visualization(intermediate_steps, class_id, vis_path)
            logger.info("Created denoising visualization for class %d sample %d: %s", class_id, idx, vis_path)
        else:
            img = generate_with_cfg(
                model=model,
                scheduler=scheduler,
                class_label=class_id,
                image_size=tcfg.image_size,
                in_channels=tcfg.in_channels,
                guidance_scale=icfg.guidance_scale,
                device=device,
                debug=enable_debug,
            )

        # Convert from [-1, 1] to [0, 1] and save PNG
        img_normalized = (img.clamp(-1, 1) + 1.0) / 2.0
        save_image(img_normalized, file_path)

        # Log first sample of each class
        if idx == 0:
            logger.info(f"  First sample for class {class_id}: range=[{img.min():.3f}, {img.max():.3f}], " +
                       f"normalized range=[{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
        
        # Log progress every 10% of samples
        if (idx + 1) % max(1, num_samples // 10) == 0:
            progress_pct = ((idx + 1) / num_samples) * 100
            logger.debug(f"  Class {class_id} progress: {idx + 1}/{num_samples} ({progress_pct:.0f}%)")

        # Store metadata for JSON index
        # Use relative path from output_dir
        relative_path = file_path.relative_to(output_dir)
        samples_metadata[str(idx)] = {
            "image": str(relative_path),
            "label": class_id,
            "is_synth": True,
            "uuid": sample_uuid,
        }

        # Convert to uint8 format [H, W, C] for NPZ storage
        # img_normalized is [1, C, H, W] in [0, 1]
        img_np = (img_normalized[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        images_list.append(img_np)
        labels_list.append(class_id)

    # Convert lists to arrays
    images_array = np.stack(images_list, axis=0)  # [N, H, W, C]
    labels_array = np.array(labels_list, dtype=np.int64)  # [N]

    logger.info("Completed generation for class %d: %d images generated", class_id, len(images_array))
    logger.debug("  Images array shape: %s, dtype: %s", images_array.shape, images_array.dtype)
    logger.debug("  Labels array shape: %s, dtype: %s", labels_array.shape, labels_array.dtype)
    
    return samples_metadata, images_array, labels_array


def build_json_index(
    all_samples: dict[int, dict[str, dict[str, Any]]],
    dataset_name: str = "PathMNIST",
    split_name: str = "synth",
) -> dict[str, Any]:
    """
    Build JSON index structure matching _build_index_structure format.

    Args:
        all_samples: Nested dict mapping class_id -> sample_id -> metadata
        dataset_name: Name of the dataset (e.g., "PathMNIST")
        split_name: Name of the split (e.g., "synth", "train", "val", "test")

    Returns:
        JSON-serializable dictionary with the structure:
        {
            "PathMNIST": {
                "synth": {
                    "0": {"image": "...", "label": 0, "is_synth": true, ...},
                    "1": {"image": "...", "label": 0, "is_synth": true, ...},
                    ...
                }
            }
        }
    """
    logger.debug("Building JSON index for dataset '%s', split '%s'", dataset_name, split_name)
    
    # Flatten all samples into a single dictionary with sequential indices
    flat_samples: dict[str, dict[str, Any]] = {}
    global_idx = 0

    for class_id in sorted(all_samples.keys()):
        for sample_id in sorted(all_samples[class_id].keys(), key=int):
            flat_samples[str(global_idx)] = all_samples[class_id][sample_id]
            global_idx += 1
    
    logger.debug("JSON index built: %d total samples across %d classes", 
                global_idx, len(all_samples))

    # Build the structure matching _build_index_structure
    return {dataset_name: {split_name: flat_samples}}


def save_split_npz(
    all_images: dict[int, np.ndarray],
    all_labels: dict[int, np.ndarray],
    split_name: str,
    output_path: Path,
) -> None:
    """
    Save generated images for a split to NPZ file.

    Creates NPZ file with the same format as the custom NPZ files created by medsyn/cli/data.py:
    - {split}_images: [N, H, W, C] uint8
    - {split}_labels: [N] int64
    - {split}_is_synth: [N] bool (all True for synthetic data)

    Args:
        all_images: Dict mapping class_id -> images array [N_class, H, W, C]
        all_labels: Dict mapping class_id -> labels array [N_class]
        split_name: Split name (train, val, test)
        output_path: Path where to save the NPZ file
    """
    logger.info(f"Saving NPZ file for split '{split_name}'...")
    logger.debug(f"NPZ output path: {output_path}")

    # Concatenate all classes
    images_list = []
    labels_list = []

    for class_id in sorted(all_images.keys()):
        images_list.append(all_images[class_id])
        labels_list.append(all_labels[class_id])
    
    logger.debug(f"Concatenating data from {len(images_list)} classes")

    # Concatenate
    images = np.concatenate(images_list, axis=0)  # [N_total, H, W, C]
    labels = np.concatenate(labels_list, axis=0)  # [N_total]
    
    logger.debug(f"Concatenated images shape: {images.shape}, dtype: {images.dtype}")
    logger.debug(f"Concatenated labels shape: {labels.shape}, dtype: {labels.dtype}")

    # Create is_synth array (all True for synthetic data)
    is_synth = np.ones(len(images), dtype=bool)

    # Build output dictionary
    output_dict = {
        f"{split_name}_images": images.astype(np.uint8),
        f"{split_name}_labels": labels.astype(np.int64),
        f"{split_name}_is_synth": is_synth,
    }

    # Save NPZ file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving NPZ with keys: {list(output_dict.keys())}")
    np.savez_compressed(str(output_path), **output_dict)

    logger.info(f"NPZ file saved to: {output_path}")
    logger.info(f"  Total samples: {len(images)}")
    logger.info(f"  Images shape: {images.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    logger.info(f"  All samples marked as synthetic (is_synth=True)")

    # Verify the saved file
    verification = np.load(str(output_path))
    logger.info("Verification - Keys in NPZ: %s", list(verification.keys()))
    for key in verification.keys():
        logger.debug("  %s: shape=%s, dtype=%s", key, verification[key].shape, verification[key].dtype)
    
    logger.info(f"NPZ file verified successfully for split '{split_name}'")


def main() -> None:
    """
    Entry point for ccddpm-generate CLI command.
    Generates synthetic images using trained class-conditioned DDPM.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with trained class-conditioned DDPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ccddpm-generate config/medsyn_cfg.yaml
  ccddpm-generate --config config.yaml --output /path/to/output
  ccddpm-generate --config config.yaml --no-visualizations

The YAML config should contain a 'generate' section:
  generate:
    checkpoint: /absolute/path/to/best.pt
    classes:
      0: 100
      1: 50
      2: 190
      ...
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="config/medsyn_cfg.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--config",
        type=str,
        dest="config_alt",
        help="Alternative way to specify config path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory (default: from config ccddpm.infer.out_dir)",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable denoising process visualizations",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PathMNIST",
        help="Dataset name for JSON index (default: PathMNIST)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="synth",
        help="Split name for JSON index (default: synth)",
    )
    args = parser.parse_args()

    # Determine config path
    config_path = Path(args.config_alt if args.config_alt else args.config)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    print("=" * 80)
    print("   Class-Conditioned DDPM - Synthetic Image Generation")
    print("=" * 80)
    logger.info("Configuration file: %s", config_path.absolute())
    logger.info("Starting ccDDPM image generation pipeline")

    try:
        # Parse generation configuration
        logger.info("Step 1: Parsing generation configuration")
        split_to_class_to_samples, checkpoint_path, output_base_path = parse_generation_config(config_path)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        if device.type == "cuda":
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            logger.info("GPU Memory: %.2f GB total, %.2f GB available", 
                       torch.cuda.get_device_properties(0).total_memory / 1e9,
                       (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9)

        # Load model and scheduler
        logger.info("Step 2: Loading model and scheduler")
        model, scheduler, cfg = load_model_and_scheduler(
            config_path, checkpoint_path, device
        )

        # Determine output directory
        if args.output:
            output_base_path = Path(args.output).resolve()
            logger.info("Output directory overridden via CLI argument: %s", output_base_path)

        output_base_path.mkdir(parents=True, exist_ok=True)
        logger.info("Output base directory: %s", output_base_path)

        # Display configuration
        print("\nGeneration Configuration:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output base directory: {output_base_path}")
        print(f"  Device: {device}")
        print(f"  Guidance scale: {cfg.ccddpm.infer.guidance_scale}")
        print(f"  Inference steps: {cfg.ccddpm.infer.num_inference_steps}")
        print(f"  Visualizations: {'Disabled' if args.no_visualizations else 'Enabled'}")

        print("\nSamples per split and class:")
        total_samples = 0
        for split_name in sorted(split_to_class_to_samples.keys()):
            class_to_samples = split_to_class_to_samples[split_name]
            split_total = sum(class_to_samples.values())
            print(f"  {split_name.capitalize()} split: {split_total} samples")
            for class_id in sorted(class_to_samples.keys()):
                num_samples = class_to_samples[class_id]
                print(f"    Class {class_id}: {num_samples} samples")
            total_samples += split_total
        print(f"  Grand total: {total_samples} images")

        print("\n" + "=" * 80)
        print("Starting generation...")
        print("=" * 80 + "\n")
        
        logger.info("Step 3: Beginning image generation for %d splits", len(split_to_class_to_samples))

        # Generate images for each split
        for split_name in sorted(split_to_class_to_samples.keys()):
            class_to_samples = split_to_class_to_samples[split_name]

            print(f"\n{'=' * 80}")
            print(f"Processing {split_name.upper()} split")
            print(f"{'=' * 80}\n")
            
            logger.info("Processing split '%s' with %d classes", split_name, len(class_to_samples))

            # Create split-specific output directory
            split_output_dir = output_base_path / split_name
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Split output directory: %s", split_output_dir)

            # Generate images for each class in this split
            all_samples: dict[int, dict[str, dict[str, Any]]] = {}
            all_images: dict[int, np.ndarray] = {}
            all_labels: dict[int, np.ndarray] = {}

            for class_id in sorted(class_to_samples.keys()):
                num_samples = class_to_samples[class_id]
                
                logger.info("Generating class %d: %d samples", class_id, num_samples)

                samples_metadata, images_array, labels_array = generate_images_for_class(
                    model=model,
                    scheduler=scheduler,
                    class_id=class_id,
                    num_samples=num_samples,
                    output_dir=split_output_dir,
                    config=cfg,
                    device=device,
                    create_visualization=not args.no_visualizations,
                )

                all_samples[class_id] = samples_metadata
                all_images[class_id] = images_array
                all_labels[class_id] = labels_array

            # Build and save JSON index for this split
            print(f"\nBuilding JSON index for {split_name} split...")
            logger.info("Building JSON index for split '%s'", split_name)
            json_index = build_json_index(
                all_samples,
                dataset_name=args.dataset_name,
                split_name=split_name,
            )

            json_path = split_output_dir / f"{args.dataset_name.lower()}_{split_name}_index.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_index, f, ensure_ascii=False, indent=2)

            logger.info("JSON index saved to: %s", json_path)
            logger.debug("JSON index contains %d samples", 
                        len(json_index.get(args.dataset_name, {}).get(split_name, {})))

            # Save NPZ file for this split
            print(f"\nSaving NPZ file for {split_name} split...")
            logger.info("Saving NPZ file for split '%s'", split_name)
            npz_path = split_output_dir / f"{args.dataset_name.lower()}_{split_name}_synth.npz"
            save_split_npz(
                all_images=all_images,
                all_labels=all_labels,
                split_name=split_name,
                output_path=npz_path,
            )

            print(f"\n{split_name.capitalize()} split completed!")
            print(f"  Images: {sum(len(arr) for arr in all_images.values())}")
            print(f"  JSON index: {json_path}")
            print(f"  NPZ file: {npz_path}")
            
            logger.info("Split '%s' processing completed successfully", split_name)

        # Summary
        print("\n" + "=" * 80)
        print("Generation completed successfully!")
        print("=" * 80)
        logger.info("=" * 60)
        logger.info("GENERATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Total images generated: %d", total_samples)
        logger.info("Splits processed: %s", ', '.join(sorted(split_to_class_to_samples.keys())))
        logger.info("Output directory: %s", output_base_path)
        
        print(f"  Total images generated: {total_samples}")
        print(f"  Output base directory: {output_base_path}")
        print(f"  Splits processed: {', '.join(sorted(split_to_class_to_samples.keys()))}")

        if not args.no_visualizations:
            num_vis = sum(1 for _ in output_base_path.rglob("denoising_process_*.png"))
            print(f"  Denoising visualizations: {num_vis}")
            logger.info("Denoising visualizations created: %d", num_vis)

        print("=" * 80 + "\n")
        logger.info("=" * 60)

    except Exception as e:
        print("\n" + "=" * 80)
        print("Generation failed!")
        print("=" * 80)
        logger.error("=" * 60)
        logger.error("GENERATION PIPELINE FAILED")
        logger.error("=" * 60)
        logger.error("Error: %s", e, exc_info=True)
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
