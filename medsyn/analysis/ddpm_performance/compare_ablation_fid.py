#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ablation Study FID Comparison Visualization

This script compares FID results from ablation study experiments (exp1, exp2, exp3)
with a selected hyperparameter configuration (e.g., gamma1_temp20).

The visualization shows:
- X-axis: Class labels + Global (average) 
- Y-axis: FID score
- 4 boxplots per class: 3 ablation experiments + 1 comparison study

Usage:
    python -m medsyn.analysis.ddpm_performance.compare_ablation_fid \
        --results-dir /path/to/results \
        --compare-study fid_gamma1_temp20.csv \
        --output /path/to/output/ablation_comparison.png

Example:
    python -m medsyn.analysis.ddpm_performance.compare_ablation_fid \
        --results-dir /media/mpascual/Sandisk2TB/research/medsyn/results/fid_results \
        --compare-study fid_gamma1_temp20.csv \
        --output /media/mpascual/Sandisk2TB/research/medsyn/results/ablation_fid_comparison.png

Expected directory structure:
    results_dir/
    ├── ablation_study/
    │   ├── fid_exp1_no_snr_classweight_temp2.csv
    │   ├── fid_exp2_snr_no_classweight.csv
    │   └── fid_exp3_baseline_no_weighting.csv
    ├── fid_gamma1_temp10.csv
    ├── fid_gamma1_temp15.csv
    └── ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Loading Functions (adapted from abstract_figures.py)
# ============================================================================

def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_npz_path(cfg: dict) -> Path:
    """
    Extract NPZ path from configuration.
    
    Priority:
      1. data.postprocess_npz.npz_path (if enabled)
      2. Fallback to download_dir/pathmnist.npz
    """
    data_cfg = cfg.get("data", {})
    
    # Check postprocess_npz first
    postprocess = data_cfg.get("postprocess_npz", {})
    if postprocess.get("enabled", False):
        npz_path = postprocess.get("npz_path")
        if npz_path and Path(npz_path).exists():
            return Path(npz_path)
    
    # Fallback to download_dir
    download_dir = data_cfg.get("download_dir", "./data_raw")
    flag = data_cfg.get("flag", "pathmnist")
    return Path(download_dir) / f"{flag}.npz"


def load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from NPZ file.
    
    Returns:
        Tuple of (all_images, all_labels)
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    logger.info(f"Loading dataset from {npz_path}")
    data = np.load(str(npz_path))
    
    all_images = []
    all_labels = []
    
    for split in ["train", "val", "test"]:
        images_key = f"{split}_images"
        labels_key = f"{split}_labels"
        
        if images_key in data and labels_key in data:
            imgs = data[images_key]
            lbls = data[labels_key].reshape(-1)
            all_images.append(imgs)
            all_labels.append(lbls)
            logger.info(f"  {split}: {len(imgs)} images")
    
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_images, all_labels


def get_representative_image_per_class(
    images: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    seed: Optional[int] = None
) -> Dict[int, np.ndarray]:
    """
    Get one representative image per class.
    
    Returns:
        Dictionary mapping class_id -> image [H, W, C]
    """
    if seed is not None:
        np.random.seed(seed)
    
    class_images = {}
    for class_id in range(num_classes):
        class_mask = labels == class_id
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) > 0:
            selected_idx = np.random.choice(class_indices)
            class_images[class_id] = images[selected_idx]
    
    return class_images


def get_class_counts(labels: np.ndarray) -> Dict[int, int]:
    """
    Get sample counts per class.
    
    Returns:
        Dictionary mapping class_id -> count
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    return {int(cls): int(cnt) for cls, cnt in zip(unique_classes, counts)}

# ============================================================================
# Plot Settings (matching temperature_weight_visualization.py style)
# ============================================================================

PLOT_SETTINGS = {
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif"],
    "font_size": 9,
    "axes_labelsize": 8,
    "axes_titlesize": 9,
    "axes_spine_width": 0.8,
    "axes_spine_color": "0.2",
    "tick_labelsize": 12,
    "tick_major_width": 0.6,
    "tick_minor_width": 0.4,
    "tick_direction": "in",
    "tick_length_major": 3.5,
    "tick_length_minor": 2.0,
    "legend_fontsize": 10,
    "legend_framealpha": 0.9,
    "legend_frameon": False,
    "legend_edgecolor": "0.8",
    "grid_linestyle": ":",
    "grid_alpha": 0.7,
    "grid_linewidth": 0.6,
    "line_width": 2.0,
    "axis_labelsize": 14,
    "xtick_fontsize": 12,
    "ytick_fontsize": 12,
    "xlabel_fontsize": 14,
    "ylabel_fontsize": 14,
    "title_fontsize": 16,
}

# Default class labels for PathMNIST
LABELS_SHORT = {
    0: "Adipose",
    1: "Background",
    2: "Debris",
    3: "Lymphocytes",
    4: "Mucus",
    5: "Smooth Muscle",
    6: "Normal Mucosa",
    7: "Cancer Stroma",
    8: "Colorectal Epi."
}

# Experiment display names
EXPERIMENT_NAMES = {
    "exp1": "Exp1: No SNR + ClassWeight τ=2",
    "exp2": "Exp2: SNR + No ClassWeight",
    "exp3": "Exp3: Baseline (No Weighting)",
}

# Colors for experiments (colorblind-friendly palette)
EXPERIMENT_COLORS = {
    "exp1": "#E69F00",  # Orange
    "exp2": "#56B4E9",  # Sky Blue
    "exp3": "#009E73",  # Teal
    "compare": "#CC79A7",  # Pink/Magenta
}


def apply_plot_settings():
    """Apply global matplotlib settings for consistent styling."""
    plt.rcParams.update({
        "font.family": PLOT_SETTINGS["font_family"],
        "font.serif": PLOT_SETTINGS["font_serif"],
        "font.size": PLOT_SETTINGS["font_size"],
        "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
        "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
        "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
        "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "xtick.direction": PLOT_SETTINGS["tick_direction"],
        "ytick.direction": PLOT_SETTINGS["tick_direction"],
        "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
        "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
        "legend.frameon": PLOT_SETTINGS["legend_frameon"],
        "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
        "axes.grid": True,
    })


def load_fid_csv(csv_path: Path) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Load a FID CSV file and extract per-class mean and std values.

    Args:
        csv_path: Path to CSV file

    Returns:
        Dictionary mapping class/metric name to (mean, std) tuple, or None if error
    """
    try:
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            logger.warning(f"Empty CSV file: {csv_path}")
            return None

        # Get the first row
        row = df.iloc[0]

        # Extract per-class metrics
        metrics = {}
        num_classes = 9  # PathMNIST has 9 classes

        for class_idx in range(num_classes):
            mean_col = f"class_{class_idx}_mean"
            std_col = f"class_{class_idx}_std"

            if mean_col in df.columns and std_col in df.columns:
                metrics[f"class_{class_idx}"] = (float(row[mean_col]), float(row[std_col]))

        # Extract average/global metrics
        if "average_mean" in df.columns and "average_std" in df.columns:
            metrics["global"] = (float(row["average_mean"]), float(row["average_std"]))

        return metrics

    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def find_ablation_files(results_dir: Path) -> Dict[str, Path]:
    """
    Find ablation study CSV files in the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary mapping experiment key to file path
    """
    ablation_dir = results_dir / "ablation_study"

    files = {}

    # Expected file patterns
    patterns = {
        "exp1": "fid_exp1_*.csv",
        "exp2": "fid_exp2_*.csv",
        "exp3": "fid_exp3_*.csv",
    }

    for exp_key, pattern in patterns.items():
        matches = list(ablation_dir.glob(pattern))
        if matches:
            files[exp_key] = matches[0]
            logger.info(f"Found {exp_key}: {matches[0].name}")
        else:
            logger.warning(f"No file found for {exp_key} with pattern {pattern}")

    return files


def extract_study_name(filename: str) -> str:
    """
    Extract a display name from the comparison study filename.

    Args:
        filename: Filename like 'fid_gamma1_temp20.csv'

    Returns:
        Display name like 'γ=1, τ=20'
    """
    # Remove 'fid_' prefix and '.csv' suffix
    name = filename
    if name.startswith("fid_"):
        name = name[4:]
    if name.endswith(".csv"):
        name = name[:-4]

    # Parse gamma and temp values
    parts = name.split("_")
    gamma = None
    temp = None

    for part in parts:
        if part.startswith("gamma"):
            gamma = part[5:]
        elif part.startswith("temp"):
            temp = part[4:]

    if gamma and temp:
        return f"γ={gamma}, τ={temp}"

    return name


def plot_ablation_comparison(
    ablation_data: Dict[str, Dict[str, Tuple[float, float]]],
    compare_data: Dict[str, Tuple[float, float]],
    compare_name: str,
    output_path: Optional[Path] = None,
    class_labels: Optional[Dict[int, str]] = None,
    class_images: Optional[Dict[int, np.ndarray]] = None,
    class_counts: Optional[Dict[int, int]] = None,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Create boxplot visualization comparing ablation experiments with a selected study.

    Args:
        ablation_data: Dictionary mapping exp key to metrics dict
        compare_data: Metrics dict for comparison study
        compare_name: Display name for comparison study
        output_path: Path to save the figure (optional)
        class_labels: Dictionary mapping class index to label name
        class_images: Dictionary mapping class index to representative image
        class_counts: Dictionary mapping class index to sample count (for ordering)
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    apply_plot_settings()

    if class_labels is None:
        class_labels = LABELS_SHORT

    num_classes = len(class_labels)

    # Determine class ordering (by count, descending - most popular first)
    if class_counts is not None:
        sorted_classes = sorted(class_counts.keys(), key=lambda c: class_counts[c], reverse=True)
    else:
        sorted_classes = list(range(num_classes))

    # Number of categories: sorted classes + Global
    num_categories = num_classes + 1

    # Prepare data for plotting
    experiments = ["exp1", "exp2", "exp3", "compare"]
    exp_names = {
        "exp1": EXPERIMENT_NAMES["exp1"],
        "exp2": EXPERIMENT_NAMES["exp2"],
        "exp3": EXPERIMENT_NAMES["exp3"],
        "compare": compare_name,
    }
    exp_colors = {
        "exp1": EXPERIMENT_COLORS["exp1"],
        "exp2": EXPERIMENT_COLORS["exp2"],
        "exp3": EXPERIMENT_COLORS["exp3"],
        "compare": EXPERIMENT_COLORS["compare"],
    }

    # Collect all data: for each category, we have 4 experiments
    # Order by sorted_classes
    all_data = {exp: [] for exp in experiments}

    for exp in experiments:
        if exp == "compare":
            data = compare_data
        else:
            data = ablation_data.get(exp, {})

        # Iterate in sorted order
        for class_idx in sorted_classes:
            key = f"class_{class_idx}"
            if key in data:
                mean, std = data[key]
                # Generate simulated data points for boxplot (mean ± std)
                points = [
                    mean - std,
                    mean - std / 2,
                    mean,
                    mean + std / 2,
                    mean + std,
                ]
                all_data[exp].append(points)
            else:
                all_data[exp].append([np.nan] * 5)

        # Global/Average (always last)
        if "global" in data:
            mean, std = data["global"]
            points = [
                mean - std,
                mean - std / 2,
                mean,
                mean + std / 2,
                mean + std,
            ]
            all_data[exp].append(points)
        else:
            all_data[exp].append([np.nan] * 5)

    # Create figure with extra bottom space for images
    fig, ax = plt.subplots(figsize=figsize)

    # Position setup for grouped boxplots with increased spacing
    num_experiments = len(experiments)
    box_width = 0.16
    spacing = 0.04
    group_spacing = 1.5  # Increased spacing between groups
    group_width = num_experiments * box_width + (num_experiments - 1) * spacing

    positions_by_exp = {exp: [] for exp in experiments}

    for cat_idx in range(num_categories):
        group_center = cat_idx * group_spacing
        group_start = group_center - group_width / 2 + box_width / 2

        for exp_idx, exp in enumerate(experiments):
            pos = group_start + exp_idx * (box_width + spacing)
            positions_by_exp[exp].append(pos)

    # Plot boxplots for each experiment
    box_plots = []
    for exp in experiments:
        positions = positions_by_exp[exp]
        data_for_exp = all_data[exp]

        bp = ax.boxplot(
            data_for_exp,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color='gray', linewidth=1.0),
            capprops=dict(color='gray', linewidth=1.0),
        )

        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor(exp_colors[exp])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)

        box_plots.append(bp)

    # X-axis tick positions (group centers)
    x_tick_positions = [i * group_spacing for i in range(num_categories)]

    # Remove text labels from x-axis (we'll use images instead)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([''] * num_categories)

    # Set labels
    ax.set_ylabel("FID Score", fontsize=PLOT_SETTINGS["ylabel_fontsize"])
    ax.set_title(
        "Ablation Study: FID Comparison by Class",
        fontsize=PLOT_SETTINGS["title_fontsize"],
        fontweight='bold',
    )

    # Configure y-axis
    ax.tick_params(axis='y', labelsize=PLOT_SETTINGS["ytick_fontsize"])

    # Set x limits with padding
    x_min = -group_spacing / 2
    x_max = (num_categories - 1) * group_spacing + group_spacing / 2
    ax.set_xlim(x_min, x_max)

    # Add legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=exp_colors[exp], alpha=0.7, edgecolor='black')
        for exp in experiments
    ]
    legend_labels = [exp_names[exp] for exp in experiments]

    ax.legend(
        legend_handles,
        legend_labels,
        loc='upper right',
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=True,
        framealpha=0.9,
        edgecolor='0.8',
    )

    # Add grid (only horizontal)
    ax.yaxis.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linewidth=PLOT_SETTINGS["grid_linewidth"])
    ax.xaxis.grid(False)

    # Add vertical separator before Global
    separator_x = (num_classes - 0.5) * group_spacing
    ax.axvline(
        x=separator_x,
        color='gray',
        linestyle='--',
        linewidth=1.0,
        alpha=0.5,
    )

    # Adjust layout to make room for images at bottom
    plt.subplots_adjust(bottom=0.18)

    # Add representative images below each class group
    if class_images is not None:
        # Get axis bounds in figure coordinates
        bbox = ax.get_position()
        ax_left = bbox.x0
        ax_right = bbox.x1
        ax_width = ax_right - ax_left

        # Image size as fraction of figure
        img_size = 0.065

        for i, class_idx in enumerate(sorted_classes):
            if class_idx in class_images:
                img = class_images[class_idx]

                # Calculate x position in figure coordinates
                x_data = i * group_spacing
                x_normalized = (x_data - x_min) / (x_max - x_min)
                x_fig = ax_left + x_normalized * ax_width - img_size / 2

                # Y position below the axes
                y_fig = 0.07

                # Create inset axes for the image
                img_ax = fig.add_axes([x_fig, y_fig, img_size, img_size])
                img_ax.imshow(img)
                img_ax.axis('off')

                # Add colored border
                for spine in img_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color('gray')
                    spine.set_linewidth(1.5)

        # Add "Global" text label for the last category
        global_x = num_classes * group_spacing
        global_x_normalized = (global_x - x_min) / (x_max - x_min)
        global_x_fig = ax_left + global_x_normalized * ax_width

        fig.text(
            global_x_fig,
            0.07,
            "Global",
            ha='center',
            va='center',
            fontsize=PLOT_SETTINGS["xtick_fontsize"],
            fontweight='bold',
        )
    else:
        # Fallback: use text labels if no images provided
        x_labels = [class_labels[c] for c in sorted_classes] + ["Global"]
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=PLOT_SETTINGS["xtick_fontsize"])

    # Save figure if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save PNG
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")

        # Save PDF
        pdf_path = output_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        logger.info(f"Saved figure to {pdf_path}")

    return fig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare ablation study FID results with a selected configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m medsyn.analysis.ddpm_performance.compare_ablation_fid \\
      --config config/medsyn_cfg.yaml \\
      --results-dir /path/to/fid_results \\
      --compare-study fid_gamma1_temp20.csv \\
      --output /path/to/ablation_comparison.png

Expected directory structure:
  results_dir/
  ├── ablation_study/
  │   ├── fid_exp1_no_snr_classweight_temp2.csv
  │   ├── fid_exp2_snr_no_classweight.csv
  │   └── fid_exp3_baseline_no_weighting.csv
  ├── fid_gamma1_temp10.csv
  └── ...
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("config/medsyn_cfg.yaml"),
        help="Path to medsyn configuration YAML (for loading dataset images)",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing FID result CSV files",
    )

    parser.add_argument(
        "--compare-study",
        type=str,
        required=True,
        help="Filename of the study to compare (e.g., fid_gamma1_temp20.csv)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the figure (PNG, will also save PDF)",
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[16, 8],
        help="Figure size (width height) in inches (default: 16 8)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting representative images (default: 42)",
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Use text labels instead of images on x-axis",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate config file
    if not args.no_images and not args.config.exists():
        parser.error(f"Config file does not exist: {args.config}. Use --no-images to skip image loading.")

    # Validate results directory
    if not args.results_dir.exists():
        parser.error(f"Results directory does not exist: {args.results_dir}")

    # Validate comparison study file
    compare_path = args.results_dir / args.compare_study
    if not compare_path.exists():
        parser.error(f"Comparison study file does not exist: {compare_path}")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger.info("Starting ablation study FID comparison")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Comparison study: {args.compare_study}")

    # Find ablation study files
    ablation_files = find_ablation_files(args.results_dir)

    if not ablation_files:
        logger.error("No ablation study files found!")
        return 1

    # Load ablation study data
    ablation_data = {}
    for exp_key, file_path in ablation_files.items():
        data = load_fid_csv(file_path)
        if data:
            ablation_data[exp_key] = data
            logger.info(f"Loaded {exp_key}: {len(data)} metrics")
        else:
            logger.warning(f"Failed to load {exp_key}")

    # Load comparison study data
    compare_path = args.results_dir / args.compare_study
    compare_data = load_fid_csv(compare_path)

    if not compare_data:
        logger.error(f"Failed to load comparison study: {compare_path}")
        return 1

    logger.info(f"Loaded comparison study: {len(compare_data)} metrics")

    # Extract display name for comparison study
    compare_name = extract_study_name(args.compare_study)
    logger.info(f"Comparison study display name: {compare_name}")

    # Load dataset for images and class counts (if not disabled)
    class_images = None
    class_counts = None

    if not args.no_images:
        try:
            logger.info(f"Loading dataset from config: {args.config}")
            cfg = load_config(args.config)
            npz_path = get_npz_path(cfg)
            all_images, all_labels = load_dataset(npz_path)

            num_classes = len(np.unique(all_labels))
            logger.info(f"Dataset loaded: {len(all_images)} images, {num_classes} classes")

            # Get representative images per class
            class_images = get_representative_image_per_class(
                all_images, all_labels, num_classes, seed=args.seed
            )
            logger.info(f"Got representative images for {len(class_images)} classes")

            # Get class counts for ordering
            class_counts = get_class_counts(all_labels)
            logger.info(f"Class counts: {class_counts}")

        except Exception as e:
            logger.warning(f"Failed to load dataset for images: {e}")
            logger.warning("Falling back to text labels")
            class_images = None
            class_counts = None

    # Create visualization
    fig = plot_ablation_comparison(
        ablation_data=ablation_data,
        compare_data=compare_data,
        compare_name=compare_name,
        output_path=args.output,
        class_images=class_images,
        class_counts=class_counts,
        figsize=tuple(args.figsize),
    )

    logger.info("Visualization complete!")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
