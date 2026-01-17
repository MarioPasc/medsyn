# medsyn/cli/train_distdiff.py
"""
CLI entry point for DistDiff guide model training.

Usage:
    distdiff-train config/adapters/distdiff_pathmnist.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys


def main():
    """Main entry point for DistDiff guide model training."""
    parser = argparse.ArgumentParser(
        description="Train DistDiff guide model for prototype-guided generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train guide model for PathMNIST
    distdiff-train config/adapters/distdiff_pathmnist.yaml

    # Train guide model for BloodMNIST
    distdiff-train config/adapters/distdiff_bloodmnist.yaml

    # Train guide model for DermaMNIST
    distdiff-train config/adapters/distdiff_dermamnist.yaml

Environment Variables:
    DATASET_PATH: Root directory for NPZ datasets
    OUTPUT_DIR: Directory for model outputs
        """
    )
    parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from medsyn.adapters.distdiff.train import train
        result = train(args.config)
        print(f"\nTraining completed!")
        print(f"Best accuracy: {result['best_accuracy']:.2f}%")
        print(f"Last accuracy: {result['last_accuracy']:.2f}%")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
