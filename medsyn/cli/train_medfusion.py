# medsyn/cli/train_medfusion.py
"""
CLI for training MedFusion with MedSyn datasets.

Usage:
    medfusion-train config/adapters/medfusion_pathmnist.yaml
"""
from __future__ import annotations
import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MedFusion with MedSyn unified dataset format"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    # Import and run training
    from medsyn.adapters.medfusion.train import train
    train(args.config)


if __name__ == "__main__":
    main()
