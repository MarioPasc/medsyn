#!/usr/bin/env python3
# medsyn/cli/train_ccDDPM.py
# Purpose: CLI entry point for class-conditioned DDPM training.

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
    Entry point for ccddpm-train CLI command.
    Trains the class-conditioned DDPM model.
    """
    parser = argparse.ArgumentParser(
        description="Train class-conditioned DDPM on medical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ccddpm-train medsyn_config.yaml
  ccddpm-train --config path/to/config.yaml
  ccddpm-train --config config.yaml --dataset /path/to/data.npz --outdir /path/to/output
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
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Absolute path to NPZ dataset. Overrides YAML config (data.postprocess_npz.npz_path)."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Absolute output directory. Overrides YAML config (ccddpm.train.output_dir)."
    )
    args = parser.parse_args()

    # Use --config if provided, otherwise use positional argument
    config_path = args.config_alt if args.config_alt else args.config
    config_path = Path(config_path)

    if not config_path.exists():
        logger.error("❌ Config file not found: %s", config_path)
        sys.exit(1)

    print("=" * 80)
    print("🚀 Starting Class-Conditioned DDPM Training")
    print("=" * 80)
    logger.info("📄 Configuration file: %s", config_path.absolute())

    try:
        from medsyn.models.ccDDPM.config import load_cfg
        from medsyn.models.ccDDPM.engine.train import train

        # Load and display configuration
        cfg = load_cfg(str(config_path))
        
        # Apply CLI overrides (takes precedence over YAML and env vars)
        if args.dataset:
            logger.info("🔧 Overriding dataset path from CLI: %s", args.dataset)
            # Try to set npz_path in dataloader config
            if hasattr(cfg.ccddpm, "dataloader") and hasattr(cfg.ccddpm.dataloader, "npz_path"):
                cfg.ccddpm.dataloader.npz_path = Path(args.dataset)
            else:
                logger.warning("⚠️  Config has no ccddpm.dataloader.npz_path field. Override may not take effect.")
        
        if args.outdir:
            logger.info("🔧 Overriding output directory from CLI: %s", args.outdir)
            # Try to set output_dir in train config
            if hasattr(cfg.ccddpm, "train") and hasattr(cfg.ccddpm.train, "output_dir"):
                cfg.ccddpm.train.output_dir = Path(args.outdir)
            else:
                logger.warning("⚠️  Config has no ccddpm.train.output_dir field. Override may not take effect.")

        print("\n📋 Training Configuration:")
        print(f"  • Epochs: {cfg.ccddpm.train.epochs}")
        print(f"  • Batch size: {cfg.ccddpm.train.batch_size}")
        print(f"  • Image size: {cfg.ccddpm.train.image_size}x{cfg.ccddpm.train.image_size}")
        print(f"  • Mixed precision: {cfg.ccddpm.train.mixed_precision}")
        print(f"  • Learning rate: {cfg.ccddpm.optim.lr}")
        print(f"  • Output directory: {cfg.ccddpm.train.output_dir}")

        print("\n🏗️  Model Configuration:")
        print(f"  • Number of classes: {cfg.ccddpm.train.num_classes}")
        print(f"  • In channels: {cfg.ccddpm.train.in_channels}")
        print(f"  • Class embed dim: {cfg.ccddpm.train.class_embed_dim}")

        print("\n⚙️  Scheduler Configuration:")
        print(f"  • Training timesteps: {cfg.ccddpm.sched.num_train_timesteps}")
        print(f"  • Beta schedule: {cfg.ccddpm.sched.beta_schedule}")
        print(f"  • Beta start: {cfg.ccddpm.sched.beta_start}")
        print(f"  • Beta end: {cfg.ccddpm.sched.beta_end}")
        print(f"  • Prediction type: {cfg.ccddpm.sched.prediction_type}")

        print("\n🎯 Training Features:")
        print(f"  • EMA: {'✅ enabled' if cfg.ccddpm.train.ema_use else '❌ disabled'}")
        if cfg.ccddpm.train.ema_use:
            print(f"    - Decay: {cfg.ccddpm.train.ema_decay}")
        print(f"  • Gradient clipping: {'✅ enabled' if cfg.ccddpm.train.grad_clip_norm else '❌ disabled'}")
        if cfg.ccddpm.train.grad_clip_norm:
            print(f"    - Norm: {cfg.ccddpm.train.grad_clip_norm}")
        print(f"  • Classifier-free guidance: {'✅ enabled' if cfg.ccddpm.train.guidance_p_uncond > 0 else '❌ disabled'}")
        if cfg.ccddpm.train.guidance_p_uncond > 0:
            print(f"    - Unconditional prob: {cfg.ccddpm.train.guidance_p_uncond}")

        print("\n" + "=" * 80)
        print("⏳ Training in progress...")
        print("=" * 80 + "\n")

        # Run training
        train(str(config_path))

        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        logger.info("📊 Model checkpoints saved to: %s", cfg.ccddpm.train.output_dir)

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Training failed!")
        print("=" * 80)
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
