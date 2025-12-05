#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FID (Frechet Inception Distance) Metric Computation Script

This script computes FID metrics comparing synthetic images against ground truth:
- Per-class FID with mean and standard deviation
- Global FID across all classes
- Comparison of multiple models

The standard deviation is computed by:
1. Keeping ALL ground truth images for every comparison
2. Splitting synthetic samples into non-overlapping subsets
3. Computing FID for each synthetic subset vs. full ground truth
4. Taking mean and std across subset FID values

Usage:
    python fid.py \\
        --ground-truth /path/to/ground_truth.npz \\
        --model ModelA /path/to/model_a.npz \\
        --model ModelB /path/to/model_b.npz \\
        --output fid_results.csv \\
        --counts 500 \\
        --weights-dir /path/to/pretrained/fid/

  python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth /media/mpascual/Sandisk2TB/research/medsyn/PathMNIST/merged.npz \
    --model CFG_MedMNIST /media/mpascual/Sandisk2TB/research/medsyn/synthetic_samples/AdamW_Batch64_lowt_enhancement_gamma5_generated/train/pathmnist_train_synth.npz \
    --model DistDiff /media/mpascual/Sandisk2TB/research/medsyn/synthetic_samples/DistDiff/split_0_64x64_balanced.npz \
    --output /media/mpascual/Sandisk2TB/research/medsyn/results/fid_results.csv \
    --counts 100 \
    --weights-dir /media/mpascual/Sandisk2TB/research/medsyn/pretrained/fid/ \
    --verbose \
    --device cuda:1
    
  python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth /media/hddb/mario/data/medsyn/merged.npz \
    --model CFG_MedMNIST /media/hddb/mario/data/medsyn/pathmnist_train_synth.npz \
    --model DistDiff /media/hddb/mario/data/medsyn/split_0_64x64_balanced.npz \
    --output /media/hddb/mario/results/medsyn/fid_results.csv \
    --counts 1000 \
    --weights-dir /media/hddb/mario/data/medsyn/pretrained/fid/ \
    --verbose \
    --device cuda:1




Output CSV format:
    model_name,class_0_mean,class_0_std,...,average_mean,average_std
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Import shared utilities
from medsyn.analysis.ddpm_performance.metrics._shared import (
    load_npz_data,
    validate_class_counts,
    preprocess_images_for_torchmetrics,
    create_subsets,
    save_results_to_csv,
    setup_inception_weights_path,
    warn_if_few_subsets,
    setup_logging,
    get_class_distribution,
)

logger = logging.getLogger(__name__)


def compute_fid_per_class(
    real_images_all: torch.Tensor,
    real_labels_all: torch.Tensor,
    synth_images_all: torch.Tensor,
    synth_labels_all: torch.Tensor,
    class_id: int,
    counts_per_computation: int,
    device: str,
    weights_path: Optional[str],
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute FID mean and std for a single class.

    Strategy:
    1. Extract all real samples for this class
    2. Split synthetic samples into non-overlapping subsets
    3. For each synthetic subset:
       - Create new FID computer
       - Add all real samples
       - Add synthetic subset
       - Compute FID
    4. Return mean and std across subsets

    Args:
        real_images_all: All real images [N, C, H, W] uint8
        real_labels_all: All real labels [N]
        synth_images_all: All synthetic images [M, C, H, W] uint8
        synth_labels_all: All synthetic labels [M]
        class_id: Class to compute FID for
        counts_per_computation: Number of synthetic samples per subset
        device: Device for computation ('cuda' or 'cpu')
        weights_path: Path to Inception weights (optional)
        seed: Random seed for subset creation

    Returns:
        Tuple of (mean_fid, std_fid)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.error(
            "torchmetrics FID not available. "
            "Install with: pip install torchmetrics[image]"
        )
        return float("nan"), float("nan")

    # Extract real samples for this class
    real_mask = real_labels_all == class_id
    real_images_class = real_images_all[real_mask]

    if len(real_images_class) == 0:
        logger.warning(f"Class {class_id}: No real samples found")
        return float("nan"), float("nan")

    # Create synthetic subsets
    synth_subsets = create_subsets(
        synth_images_all,
        synth_labels_all,
        class_id,
        counts_per_computation,
        seed,
    )

    if len(synth_subsets) == 0:
        logger.warning(f"Class {class_id}: No synthetic subsets could be created")
        return float("nan"), float("nan")

    # Warn if too few subsets
    warn_if_few_subsets(len(synth_subsets), class_id)

    # Compute FID for each subset
    fid_values = []
    for subset_idx, synth_subset in enumerate(synth_subsets):
        try:
            # Create FID computer
            fid_computer = FrechetInceptionDistance(
                feature=2048,
                normalize=True,
            )
            if weights_path:
                # Note: torchmetrics will use TORCH_HOME env var set earlier
                pass

            fid_computer = fid_computer.to(device)

            # Add real images (all of them)
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 256
                for i in range(0, len(real_images_class), chunk_size):
                    real_batch = real_images_class[i:i + chunk_size].to(device)
                    fid_computer.update(real_batch, real=True)
                    del real_batch
                    if "cuda" in device:
                        torch.cuda.empty_cache()

            # Add synthetic subset
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 256
                for i in range(0, len(synth_subset), chunk_size):
                    synth_batch = synth_subset[i:i + chunk_size].to(device)
                    fid_computer.update(synth_batch, real=False)
                    del synth_batch
                    if "cuda" in device:
                        torch.cuda.empty_cache()

            # Compute FID
            fid_value = float(fid_computer.compute().cpu().item())
            fid_values.append(fid_value)

            logger.debug(
                f"Class {class_id}, subset {subset_idx + 1}/{len(synth_subsets)}: "
                f"FID = {fid_value:.2f}"
            )

            # Clean up
            del fid_computer
            if "cuda" in device:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Class {class_id}, subset {subset_idx}: FID computation failed: {e}")
            continue

    if len(fid_values) == 0:
        logger.warning(f"Class {class_id}: No valid FID values computed")
        return float("nan"), float("nan")

    # Compute mean and std
    mean_fid = float(np.mean(fid_values))
    std_fid = float(np.std(fid_values, ddof=1) if len(fid_values) > 1 else 0.0)

    logger.info(
        f"Class {class_id}: FID = {mean_fid:.2f} +- {std_fid:.2f} "
        f"(n={len(fid_values)} subsets)"
    )

    return mean_fid, std_fid


def compute_fid_global(
    real_images: torch.Tensor,
    real_labels: torch.Tensor,
    synth_images: torch.Tensor,
    synth_labels: torch.Tensor,
    counts_per_computation: int,
    num_classes: int,
    device: str,
    weights_path: Optional[str],
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute global FID mean and std across all classes.

    Strategy:
    1. Keep all real images (across all classes)
    2. Split synthetic images into non-overlapping subsets (across all classes)
    3. For each synthetic subset:
       - Create new FID computer
       - Add all real images
       - Add synthetic subset
       - Compute FID
    4. Return mean and std across subsets

    Args:
        real_images: All real images [N, C, H, W] uint8
        real_labels: All real labels [N]
        synth_images: All synthetic images [M, C, H, W] uint8
        synth_labels: All synthetic labels [M]
        counts_per_computation: Total number of synthetic samples per subset
        num_classes: Number of classes
        device: Device for computation
        weights_path: Path to Inception weights (optional)
        seed: Random seed for shuffling

    Returns:
        Tuple of (mean_fid, std_fid)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.error(
            "torchmetrics FID not available. "
            "Install with: pip install torchmetrics[image]"
        )
        return float("nan"), float("nan")

    # Shuffle synthetic images and split into subsets
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(synth_images), generator=generator)
    shuffled_synth = synth_images[indices]
    shuffled_labels = synth_labels[indices]

    # Create non-overlapping subsets
    num_subsets = len(shuffled_synth) // counts_per_computation
    if num_subsets == 0:
        logger.warning("Global: Insufficient synthetic samples for even one subset")
        return float("nan"), float("nan")

    # Warn if too few subsets
    if num_subsets < 3:
        logger.warning(
            f"Global: Only {num_subsets} subset(s) available. "
            f"Consider lowering --counts for more reliable std estimation."
        )

    fid_values = []
    for subset_idx in range(num_subsets):
        start_idx = subset_idx * counts_per_computation
        end_idx = start_idx + counts_per_computation
        synth_subset = shuffled_synth[start_idx:end_idx]

        try:
            # Create FID computer
            fid_computer = FrechetInceptionDistance(
                feature=2048,
                normalize=True,
            )
            fid_computer = fid_computer.to(device)

            # Add all real images
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 256
                for i in range(0, len(real_images), chunk_size):
                    real_chunk = real_images[i:i + chunk_size].to(device)
                    fid_computer.update(real_chunk, real=True)
                    del real_chunk
                    if "cuda" in device:
                        torch.cuda.empty_cache()

            # Add synthetic subset
            with torch.no_grad():
                # Process in chunks to avoid OOM
                chunk_size = 256
                for i in range(0, len(synth_subset), chunk_size):
                    synth_batch = synth_subset[i:i + chunk_size].to(device)
                    fid_computer.update(synth_batch, real=False)
                    del synth_batch
                    if "cuda" in device:
                        torch.cuda.empty_cache()

            # Compute FID
            fid_value = float(fid_computer.compute().cpu().item())
            fid_values.append(fid_value)

            logger.debug(
                f"Global subset {subset_idx + 1}/{num_subsets}: FID = {fid_value:.2f}"
            )

            # Clean up
            del fid_computer
            if "cuda" in device:
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Global subset {subset_idx}: FID computation failed: {e}")
            continue

    if len(fid_values) == 0:
        logger.warning("Global: No valid FID values computed")
        return float("nan"), float("nan")

    # Compute mean and std
    mean_fid = float(np.mean(fid_values))
    std_fid = float(np.std(fid_values, ddof=1) if len(fid_values) > 1 else 0.0)

    logger.info(
        f"Global: FID = {mean_fid:.2f} +- {std_fid:.2f} "
        f"(n={len(fid_values)} subsets)"
    )

    return mean_fid, std_fid


def compute_fid_for_model(
    ground_truth_images: np.ndarray,
    ground_truth_labels: np.ndarray,
    synthetic_images: np.ndarray,
    synthetic_labels: np.ndarray,
    counts_per_computation: int,
    num_classes: int,
    device: str,
    weights_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute FID metrics with std for a single model.

    This is the main computation function that:
    1. Preprocesses images to torchmetrics format
    2. Computes per-class FID with mean/std
    3. Computes global FID with mean/std
    4. Returns dictionary with all metrics

    Args:
        ground_truth_images: Ground truth images uint8 [N, H, W, C]
        ground_truth_labels: Ground truth labels int64 [N]
        synthetic_images: Synthetic images uint8 [M, H, W, C]
        synthetic_labels: Synthetic labels int64 [M]
        counts_per_computation: Samples per class per computation
        num_classes: Number of classes
        device: Device for computation
        weights_path: Path to Inception weights (optional)
        seed: Random seed

    Returns:
        Dictionary with keys:
        - class_0_mean, class_0_std, ..., class_{num_classes-1}_mean, class_{num_classes-1}_std
        - average_mean, average_std (average across classes)
    """
    logger.info("Preprocessing images for torchmetrics...")

    # Preprocess images
    real_images = preprocess_images_for_torchmetrics(ground_truth_images)
    real_labels = torch.from_numpy(ground_truth_labels).long()

    synth_images = preprocess_images_for_torchmetrics(synthetic_images)
    synth_labels = torch.from_numpy(synthetic_labels).long()

    results = {}

    # Compute per-class FID
    logger.info("Computing per-class FID metrics...")
    class_means = []
    class_stds = []

    for class_id in tqdm(range(num_classes), desc="Computing per-class FID"):
        mean_fid, std_fid = compute_fid_per_class(
            real_images,
            real_labels,
            synth_images,
            synth_labels,
            class_id,
            counts_per_computation,
            device,
            weights_path,
            seed,
        )

        results[f"class_{class_id}_mean"] = mean_fid
        results[f"class_{class_id}_std"] = std_fid

        if not np.isnan(mean_fid):
            class_means.append(mean_fid)
            class_stds.append(std_fid)

    # Compute average across classes (excluding NaN)
    if class_means:
        results["average_mean"] = float(np.mean(class_means))
        results["average_std"] = float(np.mean(class_stds))
    else:
        results["average_mean"] = float("nan")
        results["average_std"] = float("nan")

    # Compute global FID (optional - commented out by default to save compute)
    # Uncomment if you want global FID in addition to per-class
    # logger.info("Computing global FID metric...")
    # global_mean, global_std = compute_fid_global(
    #     real_images,
    #     real_labels,
    #     synth_images,
    #     synth_labels,
    #     counts_per_computation * num_classes,  # Total samples across all classes
    #     num_classes,
    #     device,
    #     weights_path,
    #     seed,
    # )
    # results["global_mean"] = global_mean
    # results["global_std"] = global_std

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute FID metrics comparing synthetic images to ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python fid.py \\
      --ground-truth /media/.../merged.npz \\
      --model CFG /media/.../cfg_model.npz \\
      --model DistDiff /media/.../distdiff.npz \\
      --output fid_results.csv \\
      --counts 500 \\
      --weights-dir /media/.../pretrained/fid/ \\
      --verbose
        """,
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground truth NPZ file",
    )

    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="Model name and NPZ path (repeatable for multiple models)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )

    parser.add_argument(
        "--counts",
        type=int,
        default=500,
        help="Samples per class per computation (default: 500)",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=9,
        help="Number of classes (default: 9)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation (default: cuda if available, e.g. cuda:0, cuda:1)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)",
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing Inception weights for offline usage",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.ground_truth.exists():
        parser.error(f"Ground truth file not found: {args.ground_truth}")

    for model_name, model_path in args.model:
        model_path = Path(model_path)
        if not model_path.exists():
            parser.error(f"Model file not found ({model_name}): {model_path}")

    if args.counts < 1:
        parser.error("--counts must be at least 1")

    return args


def main():
    """Main entry point for FID computation script."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger.info("Starting FID computation")
    logger.info(f"Device: {args.device}")
    logger.info(f"Counts per computation: {args.counts}")
    logger.info(f"Number of classes: {args.num_classes}")

    # Setup weights path if provided
    if args.weights_dir:
        setup_inception_weights_path(args.weights_dir)

    # Load ground truth
    logger.info(f"Loading ground truth from: {args.ground_truth}")
    gt_data = load_npz_data(args.ground_truth, split="train")
    gt_images = gt_data["images"]
    gt_labels = gt_data["labels"]
    gt_is_synth = gt_data["is_synth"]

    # Filter to real samples only
    real_mask = ~gt_is_synth
    gt_images = gt_images[real_mask]
    gt_labels = gt_labels[real_mask]
    logger.info(f"Using {len(gt_images)} real ground truth images")

    # Validate ground truth
    logger.info("Validating ground truth class counts...")
    try:
        validate_class_counts(
            gt_labels,
            args.num_classes,
            args.counts,
            "Ground Truth",
            allow_skip=False,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Process each model
    all_results = {}

    for model_name, model_path_str in args.model:
        model_path = Path(model_path_str)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'=' * 60}")

        # Load synthetic data
        logger.info(f"Loading synthetic data from: {model_path}")
        synth_data = load_npz_data(model_path, split="train")
        synth_images = synth_data["images"]
        synth_labels = synth_data["labels"]

        # Validate synthetic data
        logger.info(f"Validating synthetic class counts for {model_name}...")
        try:
            validate_class_counts(
                synth_labels,
                args.num_classes,
                args.counts,
                f"Model: {model_name}",
                allow_skip=False,
            )
        except ValueError as e:
            logger.error(str(e))
            logger.error(f"Skipping model {model_name} due to insufficient samples")
            continue

        # Compute FID metrics
        logger.info(f"Computing FID metrics for {model_name}...")
        model_results = compute_fid_for_model(
            gt_images,
            gt_labels,
            synth_images,
            synth_labels,
            args.counts,
            args.num_classes,
            args.device,
            weights_path=str(args.weights_dir) if args.weights_dir else None,
            seed=args.seed,
        )

        all_results[model_name] = model_results

        # Print summary for this model
        logger.info(f"\nResults for {model_name}:")
        logger.info(f"  Average FID: {model_results['average_mean']:.2f} +- {model_results['average_std']:.2f}")

    # Save results to CSV
    if all_results:
        logger.info(f"\n{'=' * 60}")
        logger.info("Saving results to CSV")
        logger.info(f"{'=' * 60}")
        save_results_to_csv(all_results, args.output, args.num_classes)
        logger.info("FID computation completed successfully!")
        return 0
    else:
        logger.error("No results to save (all models failed validation)")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
