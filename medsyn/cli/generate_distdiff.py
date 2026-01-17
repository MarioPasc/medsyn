# medsyn/cli/generate_distdiff.py
"""
CLI entry point for DistDiff synthetic image generation.

Usage:
    distdiff-generate config/adapters/distdiff_pathmnist.yaml
    distdiff-generate config/adapters/distdiff_pathmnist.yaml --split 0 --total-split 4
"""
from __future__ import annotations

import argparse
import logging
import sys


def main():
    """Main entry point for DistDiff synthetic image generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic medical images using DistDiff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate synthetic PathMNIST images
    distdiff-generate config/adapters/distdiff_pathmnist.yaml

    # Generate with parallel splits (for multi-GPU)
    distdiff-generate config/adapters/distdiff_pathmnist.yaml --split 0 --total-split 4

    # Generate in parallel across all GPUs
    distdiff-generate config/adapters/distdiff_pathmnist.yaml --parallel 4

    # Generate in parallel on specific GPUs
    distdiff-generate config/adapters/distdiff_pathmnist.yaml --parallel 4 --gpus 0,1,2,3

Environment Variables:
    DATASET_PATH: Root directory for NPZ datasets
    OUTPUT_DIR: Directory for outputs
    CUDA_VISIBLE_DEVICES: GPU(s) to use
        """
    )
    parser.add_argument(
        "config",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Split index for parallel execution (0-indexed)"
    )
    parser.add_argument(
        "--total-split",
        type=int,
        default=None,
        help="Total number of splits for parallel execution"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel processes (auto-distributes across GPUs)"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs (e.g., '0,1,2,3')"
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
        from medsyn.adapters.distdiff.generate import generate, generate_parallel
        from medsyn.adapters.distdiff.config import load_distdiff_generate_config

        if args.parallel:
            # Parallel mode
            gpus = None
            if args.gpus:
                gpus = [int(g) for g in args.gpus.split(",")]
            generate_parallel(args.config, args.parallel, gpus)
        else:
            # Single process mode
            # Load config to potentially override split settings
            cfg = load_distdiff_generate_config(args.config)

            if args.total_split is not None:
                cfg.total_split = args.total_split
            if args.split is not None:
                cfg.split = args.split

            # Call generate with override
            generate(args.config, override_split=args.split)

        print("\nGeneration completed!")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logging.exception("Generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
