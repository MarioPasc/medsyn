#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temperature-Weighted Loss Visualization

This script visualizes how the temperature (τ) parameter affects per-class loss
weighting in classifier-free guidance diffusion models.

The visualization shows:
- X-axis: Temperature (τ) values from min to max
- Y-axis: Weight assigned to each class
- Each line represents one class with different colors
- Vertical lines mark user-specified τ points with intersection dots

Usage:
    python -m medsyn.analysis.ddpm_performance.temperature_weight_visualization \
        --class-counts 10407 10566 8006 6324 8509 13536 11557 14317 6590 \
        --tau-min 0.1 \
        --tau-max 3.0 \
        --tau-points 1.0 1.5 2.0 \
        --output /path/to/output/temperature_weights.png

Example with PathMNIST class counts:
    python -m medsyn.analysis.ddpm_performance.temperature_weight_visualization \
        --class-counts 10407 10566 8006 6324 8509 13536 11557 14317 6590 \
        --tau-min 0.1 \
        --tau-max 3.0 \
        --tau-points 1.0 1.5 2.0 \
        --output /media/mpascual/Sandisk2TB/research/medsyn/results/temperature_weight_analysis.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# ============================================================================
# Plot Settings (matching downstream_significance.py style)
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
    8: "Colorectal Epithelium"
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


def compute_class_weights_from_counts(
    counts: np.ndarray,
    temperature: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute per-class loss weights from class sample counts with temperature scaling.

    The algorithm:
    1. Compute frequency: freq[c] = count[c] / total_samples
    2. Compute inverse frequency: inv_freq[c] = 1 / freq[c]
    3. Apply temperature scaling: weight[c] = inv_freq[c] ^ temperature
    4. Normalize weights to mean=1.0 (optional, recommended for training stability)

    Temperature effects:
    - temperature = 1.0: Standard inverse frequency weighting
    - temperature > 1.0: More extreme weights (higher boost for minority classes)
    - temperature < 1.0: More uniform weights (less aggressive reweighting)

    Args:
        counts: Array of sample counts per class [num_classes]
        temperature: Temperature scaling factor (>1: more extreme, <1: more uniform)
        normalize: If True, normalize weights to mean=1.0 for gradient stability

    Returns:
        Array of class weights [num_classes]
    """
    counts = np.asarray(counts, dtype=np.float64)

    if np.any(counts < 0):
        raise ValueError(f"counts must be non-negative, got: {counts}")

    if np.all(counts == 0):
        raise ValueError("All class counts are zero, cannot compute weights")

    # Compute frequency
    total = float(counts.sum())
    freq = counts / total

    # Compute inverse frequency (with epsilon for zero-count classes)
    inv_freq = 1.0 / np.maximum(freq, 1e-8)

    # Apply temperature scaling: weight = inv_freq ^ temperature
    if temperature != 1.0:
        weights = np.power(inv_freq, temperature)
    else:
        weights = inv_freq

    # Normalize to mean=1.0 for training stability
    if normalize:
        weights = weights / weights.mean()

    return weights


def plot_temperature_weights(
    class_counts: List[int],
    tau_min: float,
    tau_max: float,
    tau_points: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
    class_labels: Optional[Dict[int, str]] = None,
    num_tau_samples: int = 200,
    normalize: bool = True,
    figsize: tuple = (12, 7),
) -> plt.Figure:
    """
    Create visualization of temperature effect on per-class loss weights.

    Args:
        class_counts: List of sample counts per class
        tau_min: Minimum temperature value
        tau_max: Maximum temperature value
        tau_points: List of tau values to mark with vertical lines
        output_path: Path to save the figure (optional)
        class_labels: Dictionary mapping class index to label name
        num_tau_samples: Number of tau samples for smooth curves
        normalize: Whether to normalize weights to mean=1.0
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    apply_plot_settings()

    # Use default labels if not provided
    if class_labels is None:
        class_labels = LABELS_SHORT

    counts = np.array(class_counts)
    num_classes = len(counts)

    # Generate tau range
    tau_values = np.linspace(tau_min, tau_max, num_tau_samples)

    # Compute weights for each tau value
    weights_matrix = np.zeros((num_tau_samples, num_classes))
    for i, tau in enumerate(tau_values):
        weights_matrix[i] = compute_class_weights_from_counts(counts, tau, normalize)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette (colorblind-friendly)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Plot each class line
    lines = []
    for class_idx in range(num_classes):
        label = class_labels.get(class_idx, f"Class {class_idx}")
        line, = ax.plot(
            tau_values,
            weights_matrix[:, class_idx],
            color=colors[class_idx],
            linewidth=PLOT_SETTINGS["line_width"],
            label=label,
        )
        lines.append(line)

    # Mark tau points with vertical lines and intersection dots
    if tau_points:
        for tau_point in tau_points:
            if tau_min <= tau_point <= tau_max:
                # Draw vertical line
                ax.axvline(
                    x=tau_point,
                    color='gray',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=1,
                )

                # Compute weights at this tau point
                weights_at_point = compute_class_weights_from_counts(counts, tau_point, normalize)

                # Plot intersection dots
                for class_idx in range(num_classes):
                    ax.scatter(
                        tau_point,
                        weights_at_point[class_idx],
                        color=colors[class_idx],
                        s=60,
                        zorder=5,
                        edgecolors='white',
                        linewidths=1.5,
                    )

                # Add tau label at the top
                y_max = ax.get_ylim()[1]
                ax.text(
                    tau_point,
                    y_max * 0.98,
                    rf"$\tau={tau_point}$",
                    ha='center',
                    va='top',
                    fontsize=PLOT_SETTINGS["legend_fontsize"],
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='white',
                        edgecolor='gray',
                        alpha=0.9,
                    ),
                )

    # Draw horizontal line at y=1.0 (normalized mean)
    if normalize:
        ax.axhline(
            y=1.0,
            color='black',
            linestyle='-',
            linewidth=1.0,
            alpha=0.5,
            zorder=0,
        )

    # Set labels and title
    ax.set_xlabel(r"Temperature $\tau$", fontsize=PLOT_SETTINGS["xlabel_fontsize"])
    ax.set_ylabel("Class Weight", fontsize=PLOT_SETTINGS["ylabel_fontsize"])
    ax.set_title(
        r"Effect of Temperature $\tau$ on Per-Class Loss Weights",
        fontsize=PLOT_SETTINGS["title_fontsize"],
        fontweight='bold',
    )

    # Configure ticks
    ax.tick_params(axis='x', labelsize=PLOT_SETTINGS["xtick_fontsize"])
    ax.tick_params(axis='y', labelsize=PLOT_SETTINGS["ytick_fontsize"])

    # Set x limits
    ax.set_xlim(tau_min, tau_max)

    # Add legend inside the plot (top-left, no box)
    ax.legend(
        loc='upper left',
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        frameon=False,
    )

    # Add grid
    ax.grid(True, alpha=PLOT_SETTINGS["grid_alpha"], linewidth=PLOT_SETTINGS["grid_linewidth"])

    # Tight layout
    plt.tight_layout()

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
        description="Visualize temperature effect on per-class loss weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m medsyn.analysis.ddpm_performance.temperature_weight_visualization \\
      --class-counts 10407 10566 8006 6324 8509 13536 11557 14317 6590 \\
      --tau-min 0.1 \\
      --tau-max 3.0 \\
      --tau-points 1.0 1.5 2.0 \\
      --output /path/to/temperature_weights.png
        """,
    )

    parser.add_argument(
        "--class-counts",
        type=int,
        nargs="+",
        required=True,
        help="Sample counts per class (space-separated integers)",
    )

    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.1,
        help="Minimum temperature value (default: 0.1)",
    )

    parser.add_argument(
        "--tau-max",
        type=float,
        default=3.0,
        help="Maximum temperature value (default: 3.0)",
    )

    parser.add_argument(
        "--tau-points",
        type=float,
        nargs="*",
        default=None,
        help="Temperature values to mark with vertical lines (space-separated)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the figure (PNG, will also save PDF)",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize weights to mean=1.0",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of tau samples for smooth curves (default: 200)",
    )

    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 7],
        help="Figure size (width height) in inches (default: 12 7)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate
    if args.tau_min >= args.tau_max:
        parser.error("--tau-min must be less than --tau-max")

    if args.tau_min <= 0:
        parser.error("--tau-min must be positive")

    if len(args.class_counts) < 2:
        parser.error("At least 2 class counts are required")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger.info("Starting temperature weight visualization")
    logger.info(f"Class counts: {args.class_counts}")
    logger.info(f"Temperature range: [{args.tau_min}, {args.tau_max}]")
    if args.tau_points:
        logger.info(f"Tau points to mark: {args.tau_points}")

    # Create visualization
    fig = plot_temperature_weights(
        class_counts=args.class_counts,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_points=args.tau_points,
        output_path=args.output,
        num_tau_samples=args.num_samples,
        normalize=not args.no_normalize,
        figsize=tuple(args.figsize),
    )

    logger.info("Visualization complete!")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
