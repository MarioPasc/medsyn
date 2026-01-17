# medsyn/adapters/distdiff/generate.py
"""
DistDiff synthetic image generation wrapper.

This module provides a standardized generation interface for DistDiff,
accepting YAML configuration and calling the original generation script.

DistDiff uses Stable Diffusion with prototype-guided sampling to generate
synthetic medical images.
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List
import subprocess

from medsyn.adapters.distdiff.config import (
    DistDiffGenerateConfig,
    load_distdiff_generate_config,
)

logger = logging.getLogger(__name__)


def build_generate_args(cfg: DistDiffGenerateConfig) -> List[str]:
    """
    Build command-line arguments for the DistDiff generate_data.py script.

    Args:
        cfg: DistDiffGenerateConfig instance

    Returns:
        List of command-line arguments
    """
    args = []

    # Dataset arguments
    # Use pathmnist_npz to trigger NPZ loading path
    args.extend(["--dataset", "pathmnist_npz"])
    args.extend(["--data_dir", cfg.npz_path])

    # Guide model arguments
    args.extend(["--arch", cfg.guide_model_arch])
    if cfg.guide_model_path:
        args.extend(["--encoder_weight_path", cfg.guide_model_path])

    # Stable Diffusion arguments
    args.extend(["--pretrained_model_name_or_path", cfg.pretrained_model_name_or_path])
    args.extend(["--resolution", str(cfg.resolution)])
    args.extend(["--guidance_scale", str(cfg.guidance_scale)])

    # Guidance arguments
    if cfg.guidance_type:
        args.extend(["--guidance_type", cfg.guidance_type])
    if cfg.optimize_targets:
        args.extend(["--optimize_targets", cfg.optimize_targets])
    args.extend(["--K", str(cfg.K)])
    args.extend(["--rho", str(cfg.rho)])
    args.extend(["--gs", str(cfg.gs)])
    args.extend(["--ls", str(cfg.ls)])
    args.extend(["--constraint_value", str(cfg.constraint_value)])
    args.extend(["--guidance_step", str(cfg.guidance_step)])
    args.extend(["--guidance_period", str(cfg.guidance_period)])
    args.extend(["--strength", str(cfg.strength)])

    # Generation arguments
    args.extend(["--num_images_per_prompt", str(cfg.num_images_per_sample)])
    args.extend(["--steps", str(cfg.steps)])
    if cfg.seed is not None:
        args.extend(["--seed", str(cfg.seed)])

    # Split arguments (for parallel generation)
    args.extend(["--total_split", str(cfg.total_split)])
    args.extend(["--split", str(cfg.split)])

    # Output arguments
    args.extend(["--output_dir", cfg.output_dir])
    args.extend(["--logging_dir", str(Path(cfg.output_dir) / "logs")])

    # Hardware arguments
    args.extend(["--train_batch_size", str(cfg.train_batch_size)])
    args.extend(["--val_batch_size", str(cfg.val_batch_size)])
    if cfg.mixed_precision:
        args.extend(["--mixed_precision", cfg.mixed_precision])

    # Cache and offline arguments
    if cfg.cache_dir:
        args.extend(["--cache_dir", cfg.cache_dir])
    if cfg.local_files_only:
        args.append("--local_files_only")

    # Language enhancement
    if cfg.language_enhance:
        args.append("--language_enhance")

    return args


def generate(config_path: str, override_split: Optional[int] = None) -> None:
    """
    Generate synthetic images using DistDiff.

    This function calls the original DistDiff generate_data.py script
    with arguments built from the YAML configuration.

    Args:
        config_path: Path to YAML configuration file
        override_split: Optional split index override (for parallel execution)
    """
    # Load configuration
    cfg = load_distdiff_generate_config(config_path)

    # Override split if specified
    if override_split is not None:
        cfg.split = override_split

    logger.info(f"Starting DistDiff generation")
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"NPZ path: {cfg.npz_path}")
    logger.info(f"Guide model: {cfg.guide_model_path}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Split: {cfg.split}/{cfg.total_split}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build arguments
    args = build_generate_args(cfg)
    logger.info(f"Generated arguments: {' '.join(args)}")

    # Find the generate_data.py script
    distdiff_path = Path(__file__).parent.parent.parent / "models" / "distdiff"
    generate_script = distdiff_path / "generate_data.py"

    if not generate_script.exists():
        raise FileNotFoundError(
            f"DistDiff generate_data.py not found at: {generate_script}"
        )

    # Change to DistDiff directory for proper imports
    original_dir = os.getcwd()
    os.chdir(distdiff_path)

    try:
        # Build the command
        cmd = [sys.executable, str(generate_script)] + args
        logger.info(f"Running: {' '.join(cmd)}")

        # Run the generation script
        result = subprocess.run(
            cmd,
            capture_output=False,  # Let output go to stdout/stderr
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"DistDiff generation failed with return code {result.returncode}"
            )

        logger.info("Generation completed successfully")

    finally:
        os.chdir(original_dir)


def generate_parallel(
    config_path: str,
    num_splits: int,
    gpus: Optional[List[int]] = None,
) -> None:
    """
    Generate synthetic images in parallel across multiple GPUs.

    This function spawns multiple generation processes, each handling
    a different split of the data.

    Args:
        config_path: Path to YAML configuration file
        num_splits: Number of parallel splits
        gpus: Optional list of GPU IDs to use (defaults to all available)
    """
    import subprocess
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Load configuration
    cfg = load_distdiff_generate_config(config_path)

    # Update total_split in config
    cfg.total_split = num_splits

    logger.info(f"Starting parallel generation with {num_splits} splits")

    # Determine GPUs to use
    if gpus is None:
        import torch
        gpus = list(range(torch.cuda.device_count()))

    if not gpus:
        raise RuntimeError("No GPUs available for parallel generation")

    logger.info(f"Using GPUs: {gpus}")

    # Find the CLI script
    cli_module = "medsyn.cli.generate_distdiff"

    def run_split(split_idx: int, gpu_id: int):
        """Run generation for a single split on a specific GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            sys.executable, "-m", cli_module,
            config_path,
            "--split", str(split_idx),
            "--total-split", str(num_splits),
        ]

        logger.info(f"Starting split {split_idx} on GPU {gpu_id}")

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"Split {split_idx} failed: {result.stderr}")
            raise RuntimeError(f"Split {split_idx} failed")

        logger.info(f"Split {split_idx} completed")
        return split_idx

    # Run splits in parallel
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for split_idx in range(num_splits):
            gpu_id = gpus[split_idx % len(gpus)]
            future = executor.submit(run_split, split_idx, gpu_id)
            futures[future] = split_idx

        for future in as_completed(futures):
            split_idx = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Split {split_idx} raised exception: {e}")
                raise

    logger.info("All parallel generation completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic images with DistDiff")
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--split", type=int, default=None, help="Split index for parallel execution")
    parser.add_argument("--parallel", type=int, default=None, help="Number of parallel splits")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs")
    args = parser.parse_args()

    if args.parallel:
        gpus = None
        if args.gpus:
            gpus = [int(g) for g in args.gpus.split(",")]
        generate_parallel(args.config, args.parallel, gpus)
    else:
        generate(args.config, args.split)
