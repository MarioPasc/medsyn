# medsyn/cli/train_cbdm.py
"""
CLI for training CBDM (Class-Balancing Diffusion Models) with MedSyn datasets.

Usage:
    cbdm-train config/adapters/cbdm_pathmnist.yaml
"""
from __future__ import annotations
import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CBDM with MedSyn unified dataset format"
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
    from medsyn.adapters.cbdm.train import train
    train(args.config)


if __name__ == "__main__":
    main()
