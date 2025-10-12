#!/usr/bin/env python3
# medsyn/cli/generate_bVAE.py
# Purpose: CLI entry point for conditional β-VAE synthetic image generation.

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_generation_config(config_path: Path) -> tuple[Dict[int, int], Path]:
    """
    Parse the generation configuration from YAML.

    The new format expects:
    generate:
      checkpoint: /absolute/path/to/best.pt
      classes:
        0: 100
        1: 50
        2: 190
        ...

    Returns:
        tuple: (class_to_samples_dict, checkpoint_path)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'generate' not in config:
        raise ValueError("'generate' section not found in configuration file")

    gen_config = config['generate']

    # Parse checkpoint path (must be absolute)
    if 'checkpoint' not in gen_config:
        raise ValueError("'checkpoint' path not specified in generate section")

    checkpoint_path = Path(gen_config['checkpoint'])
    if not checkpoint_path.is_absolute():
        raise ValueError(f"Checkpoint path must be absolute, got: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Parse classes dictionary
    if 'classes' not in gen_config:
        raise ValueError("'classes' dictionary not found in generate section")

    classes_dict = gen_config['classes']
    if not isinstance(classes_dict, dict):
        raise ValueError("'classes' must be a dictionary mapping class_id -> num_samples")

    # Convert to int keys and validate
    class_to_samples = {}
    for class_id, num_samples in classes_dict.items():
        try:
            class_id_int = int(class_id)
            num_samples_int = int(num_samples)
            if num_samples_int <= 0:
                raise ValueError(f"Number of samples must be positive, got {num_samples_int} for class {class_id_int}")
            class_to_samples[class_id_int] = num_samples_int
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid class/samples specification: {class_id}:{num_samples} - {e}")

    if not class_to_samples:
        raise ValueError("No valid classes specified in generate.classes")

    return class_to_samples, checkpoint_path


def main():
    """
    Entry point for bvae-generate CLI command.
    Generates synthetic images using trained conditional β-VAE.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with trained conditional β-VAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bvae-generate medsyn_config.yaml
  bvae-generate --config config.yaml

The YAML config should contain a 'generate' section with:
  generate:
    checkpoint: /absolute/path/to/best.pt
    classes:
      0: 100
      1: 50
      2: 190
        """
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="medsyn_config.yaml",
        help="Path to YAML configuration file (default: medsyn_config.yaml)"
    )
    parser.add_argument(
        "--config",
        type=str,
        dest="config_alt",
        help="Alternative way to specify config path"
    )
    args = parser.parse_args()

    # Use --config if provided, otherwise use positional argument
    config_path = args.config_alt if args.config_alt else args.config
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error("❌ Config file not found: %s", config_path)
        sys.exit(1)

    print("=" * 80)
    print("🎨 Starting Conditional β-VAE Generation")
    print("=" * 80)
    logger.info("📄 Configuration file: %s", config_path.absolute())

    try:
        # Parse generation configuration
        class_to_samples, checkpoint_path = parse_generation_config(config_path)

        print("\n📋 Generation Configuration:")
        print(f"  • Checkpoint: {checkpoint_path}")
        print(f"  • Total classes: {len(class_to_samples)}")
        print(f"  • Total images to generate: {sum(class_to_samples.values())}")

        print("\n🎯 Samples per class:")
        for class_id in sorted(class_to_samples.keys()):
            num_samples = class_to_samples[class_id]
            print(f"  • Class {class_id}: {num_samples} samples")

        # Import generation utilities
        from medsyn.models.bVAE.engine.predict import load_model_from_ckpt
        import torch
        from torchvision.utils import save_image

        print("\n" + "=" * 80)
        print("⏳ Loading model and generating images...")
        print("=" * 80 + "\n")

        # Load model
        logger.info("🔄 Loading model from checkpoint: %s", checkpoint_path)
        model, device, out_dir = load_model_from_ckpt(str(config_path), str(checkpoint_path))
        logger.info("✅ Model loaded successfully on device: %s", device)

        # Generate images for each class
        total_generated = 0
        for class_id in sorted(class_to_samples.keys()):
            num_samples = class_to_samples[class_id]

            print(f"🖼️  Generating {num_samples} images for class {class_id}...", end=" ")

            # Generate samples for this class
            y = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
            imgs = model.sample(num_samples, y=y, device=device)

            # Save images
            for i in range(num_samples):
                img_path = out_dir / f"class{class_id}_synth_{total_generated:06d}.png"
                save_image(imgs[i], img_path)
                total_generated += 1

            print("✅")
            logger.info("Class %d: Generated %d images", class_id, num_samples)

        print("\n" + "=" * 80)
        print("✅ Generation completed successfully!")
        print("=" * 80)
        print(f"📁 Output directory: {out_dir}")
        print(f"📊 Total images generated: {total_generated}")
        logger.info("🎉 All images saved to: %s", out_dir)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Generation failed!")
        print("=" * 80)
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
