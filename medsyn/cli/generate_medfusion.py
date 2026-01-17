# medsyn/cli/generate_medfusion.py
"""
CLI for generating synthetic images with trained MedFusion models.

Usage:
    medfusion-generate config/adapters/medfusion_pathmnist.yaml --num-samples 1000
    medfusion-generate config/adapters/medfusion_pathmnist.yaml --augment original.npz --output augmented.npz
"""
from __future__ import annotations
import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with trained MedFusion"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (must include checkpoint_path)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for NPZ file (default: {output_dir}/synthetic.npz)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Generate balanced samples per class (overrides --num-samples)",
    )
    parser.add_argument(
        "--augment",
        type=str,
        default=None,
        help="Path to original NPZ to augment with synthetic samples",
    )
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Don't save sample grid image",
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

    # Import generation functions
    from medsyn.adapters.medfusion.generate import generate_and_save, augment_dataset

    if args.augment:
        # Augment existing dataset
        output_path = args.output or "augmented.npz"
        augment_dataset(
            config_path=args.config,
            original_npz_path=args.augment,
            output_path=output_path,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
        )
    else:
        # Generate new synthetic dataset
        generate_and_save(
            config_path=args.config,
            output_path=args.output,
            num_samples=args.num_samples,
            samples_per_class=args.samples_per_class,
            save_grid=not args.no_grid,
        )


if __name__ == "__main__":
    main()
