#!/usr/bin/env python3
# medsyn/cli/generate_ccDDPM.py
# Purpose: CLI entry point for class-conditioned DDPM synthetic image generation.

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict
import yaml # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_generation_config(config_path: Path) -> tuple[Dict[int, int], Path]:
    """
    Parse the generation configuration from YAML.

    The format expects:
    generate:
      checkpoint: /absolute/path/to/ccddpm_epX.pt
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
    Entry point for ccddpm-generate CLI command.
    Generates synthetic images using trained class-conditioned DDPM.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with trained class-conditioned DDPM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ccddpm-generate medsyn_config.yaml
  ccddpm-generate --config config.yaml

The YAML config should contain a 'generate' section with:
  generate:
    checkpoint: /absolute/path/to/ccddpm_epX.pt
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
    print("🎨 Starting Class-Conditioned DDPM Generation")
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
        from medsyn.models.ccDDPM.engine.predict import generate
        from medsyn.models.ccDDPM.config import load_cfg

        # Load configuration to display inference settings
        cfg = load_cfg(str(config_path))
        
        print("\n⚙️  Inference Configuration:")
        print(f"  • Inference steps: {cfg.ccddpm.infer.num_inference_steps}")
        print(f"  • Guidance scale: {cfg.ccddpm.infer.guidance_scale}")
        print(f"  • Save grid: {'✅' if cfg.ccddpm.infer.save_grid else '❌'}")
        print(f"  • Output directory: {cfg.ccddpm.infer.out_dir}")

        print("\n" + "=" * 80)
        print("⏳ Generating images...")
        print("=" * 80 + "\n")

        # Generate images for each class
        total_generated = 0
        all_saved_paths = []

        for class_id in sorted(class_to_samples.keys()):
            num_samples = class_to_samples[class_id]

            print(f"🖼️  Generating {num_samples} images for class {class_id}...", end=" ")
            logger.info("Starting generation for class %d (%d samples)", class_id, num_samples)

            # Generate samples for this class using the predict module
            saved_paths = generate(
                yaml_path=str(config_path),
                checkpoint=checkpoint_path,
                class_id=class_id,
                k=num_samples
            )

            total_generated += len(saved_paths)
            all_saved_paths.extend(saved_paths)
            
            print("✅")
            logger.info("Class %d: Generated %d images", class_id, len(saved_paths))

        print("\n" + "=" * 80)
        print("✅ Generation completed successfully!")
        print("=" * 80)
        print(f"📁 Output directory: {cfg.ccddpm.infer.out_dir}")
        print(f"📊 Total images generated: {total_generated}")
        logger.info("🎉 All images saved to: %s", cfg.ccddpm.infer.out_dir)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Generation failed!")
        print("=" * 80)
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
