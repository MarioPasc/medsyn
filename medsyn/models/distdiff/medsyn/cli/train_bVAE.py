#!/usr/bin/env python3
# medsyn/cli/train_bVAE.py
# Purpose: CLI entry point for conditional β-VAE training.

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """
    Entry point for bvae-train CLI command.
    Trains the conditional β-VAE model.
    """
    parser = argparse.ArgumentParser(
        description="Train conditional β-VAE on PathMNIST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bvae-train medsyn_config.yaml
  bvae-train --config path/to/config.yaml
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
    print("🚀 Starting Conditional β-VAE Training")
    print("=" * 80)
    logger.info("📄 Configuration file: %s", config_path.absolute())

    try:
        from medsyn.models.bVAE.config import load_bvae_config
        from medsyn.models.bVAE.engine.train import train

        # Load and display configuration
        cfg = load_bvae_config(str(config_path))

        print("\n📋 Training Configuration:")
        print(f"  • Epochs: {cfg.train.epochs}")
        print(f"  • Batch size: {cfg.train.batch_size}")
        print(f"  • Device: {cfg.train.device}")
        print(f"  • Mixed precision: {cfg.train.mixed_precision}")
        print(f"  • Learning rate: {cfg.optim.lr_init}")
        print(f"  • Output directory: {cfg.train.output_dir}")

        print("\n🏗️  Model Configuration:")
        print(f"  • Image size: {cfg.model.img_size}x{cfg.model.img_size}")
        print(f"  • Latent dimension: {cfg.model.latent_dim}")
        print(f"  • Number of classes: {cfg.model.num_classes}")
        print(f"  • Conditioning type: {cfg.model.conditioning}")
        print(f"  • Beta value: {cfg.loss.beta}")

        print("\n" + "=" * 80)
        print("⏳ Training in progress...")
        print("=" * 80 + "\n")

        # Run training
        train(str(config_path))

        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        logger.info("📊 Model checkpoints saved to: %s/ckpts", cfg.train.output_dir)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Training failed!")
        print("=" * 80)
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
