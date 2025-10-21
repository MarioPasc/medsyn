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


def parse_generation_config(config_path: Path) -> tuple[dict[int, int], Path]:
    """
    Parse the generation configuration from YAML.

    Expected format:
        generate:
          checkpoint: /absolute/path/to/best.pt
          classes:
            0: 100
            1: 50
            ...

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (class_to_samples_dict, checkpoint_path)

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If checkpoint doesn't exist
    """
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

    # Parse classes dictionary
    if "classes" not in gen_config:
        raise ValueError("'classes' dictionary not found in generate section")

    classes_dict = gen_config["classes"]
    if not isinstance(classes_dict, dict):
        raise ValueError("'classes' must be a dictionary mapping class_id -> num_samples")

    # Convert to int keys and validate
    class_to_samples: dict[int, int] = {}
    for class_id, num_samples in classes_dict.items():
        class_id_int = int(class_id)
        num_samples_int = int(num_samples)
        if num_samples_int <= 0:
            raise ValueError(
                f"Number of samples must be positive, got {num_samples_int} for class {class_id_int}"
            )
        class_to_samples[class_id_int] = num_samples_int

    if not class_to_samples:
        raise ValueError("No valid classes specified in generate.classes")

    return class_to_samples, checkpoint_path


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

    # Initialize model
    mcfg = CCDDPMInit(
        in_channels=tcfg.in_channels,
        class_embed_dim=tcfg.class_embed_dim,
        num_classes=tcfg.num_classes,
    )
    model = CCDDPM(mcfg).to(device)

    # Load checkpoint
    logger.info("Loading checkpoint from %s", checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Check if EMA weights are available (better quality)
    # EMA dict must exist, not be None, and contain weights (not be empty)
    if "ema" in state and state["ema"] is not None and len(state["ema"]) > 0:
        logger.info("Using EMA weights for generation (higher quality)")
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

    # Initialize scheduler for inference
    scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=False,
    )
    scheduler.set_timesteps(icfg.num_inference_steps, device=device)

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
        logger.info(f"Starting generation for class {class_label}")
        logger.info(f"  Initial noise: shape={x_t.shape}, range=[{x_t.min():.3f}, {x_t.max():.3f}]")
        logger.info(f"  Scheduler timesteps: {len(scheduler.timesteps)} steps")
        logger.info(f"  Guidance scale: {guidance_scale}")

    # Denoising loop
    for step_idx, t in enumerate(scheduler.timesteps):
        t_batch = t.unsqueeze(0) if t.dim() == 0 else t

        # Classifier-free guidance
        if guidance_scale != 1.0:
            eps_cond = model(x_t, t_batch, labels)
            eps_uncond = model(x_t, t_batch, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            if debug and step_idx % 200 == 0:
                logger.info(f"  Step {step_idx}/{len(scheduler.timesteps)}: t={t.item()}, " +
                           f"eps_cond_range=[{eps_cond.min():.3f}, {eps_cond.max():.3f}], " +
                           f"eps_uncond_range=[{eps_uncond.min():.3f}, {eps_uncond.max():.3f}]")
        else:
            eps = model(x_t, t_batch, labels)

            if debug and (step_idx < 5 or step_idx % 200 == 0):
                logger.info(f"  Step {step_idx}/{len(scheduler.timesteps)}: t={t.item()}, " +
                           f"x_t_range=[{x_t.min():.3f}, {x_t.max():.3f}], " +
                           f"eps_range=[{eps.min():.3f}, {eps.max():.3f}]")

        # Denoising step
        x_t = scheduler.step(model_output=eps, timestep=t, sample=x_t).prev_sample

        # Extra debug for first few steps
        if debug and step_idx < 5:
            logger.info(f"    After step: x_t_range=[{x_t.min():.3f}, {x_t.max():.3f}]")

    if debug:
        logger.info(f"  Final image: range=[{x_t.min():.3f}, {x_t.max():.3f}]")

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

    # Determine which steps to save
    total_steps = len(scheduler.timesteps)
    save_indices = set(
        np.linspace(0, total_steps - 1, num_vis_steps, dtype=int).tolist()
    )
    save_indices.add(total_steps - 1)  # Always save the last step

    intermediate_steps: list[torch.Tensor] = [x_t.cpu().clone()]

    for i, t in enumerate(scheduler.timesteps):
        t_batch = t.unsqueeze(0) if t.dim() == 0 else t

        if guidance_scale > 0:
            eps_cond = model(x_t, t_batch, labels)
            eps_uncond = model(x_t, t_batch, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x_t, t_batch, labels)

        x_t = scheduler.step(model_output=eps, timestep=t, sample=x_t).prev_sample

        if i in save_indices:
            intermediate_steps.append(x_t.cpu().clone())

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

    # Add overall title
    fig.suptitle(
        f"Denoising Process Visualization - Class {class_label}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved denoising visualization to %s", output_path)


def generate_images_for_class(
    model: CCDDPM,
    scheduler: DDPMScheduler,
    class_id: int,
    num_samples: int,
    output_dir: Path,
    config: ProjectCfg,
    device: torch.device,
    create_visualization: bool = False,
) -> dict[str, dict[str, Any]]:
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
        Dictionary mapping sample IDs to metadata (for JSON index)
    """
    tcfg = config.ccddpm.train
    icfg = config.ccddpm.infer

    # Create class-specific output directory
    class_dir = output_dir / f"class_{class_id}"
    class_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store metadata for JSON index
    samples_metadata: dict[str, dict[str, Any]] = {}

    logger.info("Generating %d images for class %d...", num_samples, class_id)

    # Randomly select one sample for visualization
    vis_sample_idx = np.random.randint(0, num_samples) if create_visualization else -1

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

        # Convert from [-1, 1] to [0, 1] and save
        img_normalized = (img.clamp(-1, 1) + 1.0) / 2.0
        save_image(img_normalized, file_path)

        # Log first sample of each class
        if idx == 0:
            logger.info(f"  First sample for class {class_id}: range=[{img.min():.3f}, {img.max():.3f}], " +
                       f"normalized range=[{img_normalized.min():.3f}, {img_normalized.max():.3f}]")

        # Store metadata for JSON index
        # Use relative path from output_dir
        relative_path = file_path.relative_to(output_dir)
        samples_metadata[str(idx)] = {
            "image": str(relative_path),
            "label": class_id,
            "is_synth": True,
            "uuid": sample_uuid,
        }

    logger.info("Completed generation for class %d", class_id)
    return samples_metadata


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
    # Flatten all samples into a single dictionary with sequential indices
    flat_samples: dict[str, dict[str, Any]] = {}
    global_idx = 0

    for class_id in sorted(all_samples.keys()):
        for sample_id in sorted(all_samples[class_id].keys(), key=int):
            flat_samples[str(global_idx)] = all_samples[class_id][sample_id]
            global_idx += 1

    # Build the structure matching _build_index_structure
    return {dataset_name: {split_name: flat_samples}}


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

    try:
        # Parse generation configuration
        class_to_samples, checkpoint_path = parse_generation_config(config_path)

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Load model and scheduler
        model, scheduler, cfg = load_model_and_scheduler(
            config_path, checkpoint_path, device
        )

        # Determine output directory
        if args.output:
            output_dir = Path(args.output).resolve()
        else:
            output_dir = Path(cfg.ccddpm.infer.out_dir).resolve()

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", output_dir)

        # Display configuration
        print("\nGeneration Configuration:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Device: {device}")
        print(f"  Guidance scale: {cfg.ccddpm.infer.guidance_scale}")
        print(f"  Inference steps: {cfg.ccddpm.infer.num_inference_steps}")
        print(f"  Visualizations: {'Disabled' if args.no_visualizations else 'Enabled'}")

        print("\nSamples per class:")
        total_samples = 0
        for class_id in sorted(class_to_samples.keys()):
            num_samples = class_to_samples[class_id]
            print(f"  Class {class_id}: {num_samples} samples")
            total_samples += num_samples
        print(f"  Total: {total_samples} images")

        print("\n" + "=" * 80)
        print("Starting generation...")
        print("=" * 80 + "\n")

        # Generate images for each class
        all_samples: dict[int, dict[str, dict[str, Any]]] = {}

        for class_id in sorted(class_to_samples.keys()):
            num_samples = class_to_samples[class_id]

            samples_metadata = generate_images_for_class(
                model=model,
                scheduler=scheduler,
                class_id=class_id,
                num_samples=num_samples,
                output_dir=output_dir,
                config=cfg,
                device=device,
                create_visualization=not args.no_visualizations,
            )

            all_samples[class_id] = samples_metadata

        # Build and save JSON index
        print("\nBuilding JSON index...")
        json_index = build_json_index(
            all_samples,
            dataset_name=args.dataset_name,
            split_name=args.split_name,
        )

        json_path = output_dir / f"{args.dataset_name.lower()}_synth_index.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_index, f, ensure_ascii=False, indent=2)

        logger.info("JSON index saved to: %s", json_path)

        # Summary
        print("\n" + "=" * 80)
        print("Generation completed successfully!")
        print("=" * 80)
        print(f"  Total images generated: {total_samples}")
        print(f"  Output directory: {output_dir}")
        print(f"  JSON index: {json_path}")

        if not args.no_visualizations:
            num_vis = sum(1 for _ in output_dir.rglob("denoising_process_*.png"))
            print(f"  Denoising visualizations: {num_vis}")

        print("=" * 80 + "\n")

    except Exception as e:
        print("\n" + "=" * 80)
        print("Generation failed!")
        print("=" * 80)
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
