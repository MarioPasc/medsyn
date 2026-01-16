#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pareto Frontier Plot: FID vs Macro-F1

Creates publication-quality Pareto frontier visualization comparing generative
fidelity (FID) against downstream utility (Macro-F1 classification performance).

Usage:
    # Global Pareto frontier (default)
    python -m medsyn.analysis.pareto_fid_f1 \
        --classification-results-dir /media/mpascual/Sandisk2TB/research/medsyn/results/downstream_analysis_test \
        --fid-csv /media/mpascual/Sandisk2TB/research/medsyn/results/aggregated_fid_comparison.csv \
        --output-dir /media/mpascual/Sandisk2TB/research/medsyn/results

    # With per-class Pareto frontiers (3×3 grid)
    python -m medsyn.analysis.pareto_fid_f1 \
        --classification-results-dir /media/mpascual/Sandisk2TB/research/medsyn/results/downstream_analysis_test \
        --fid-csv /media/mpascual/Sandisk2TB/research/medsyn/results/aggregated_fid_comparison.csv \
        --output-dir /media/mpascual/Sandisk2TB/research/medsyn/results \
        --per-class
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Visual design settings (adapted from time_sensitive_anisotropy_evolution.py)
PLOT_SETTINGS = {
    # Font and general figure settings
    "font_family": "serif",
    "font_serif": ["Times New Roman", "DejaVu Serif"],
    "font_size": 9,

    # Axis label / title sizes
    "axes_labelsize": 8,
    "axes_titlesize": 9,

    # Axis spines
    "axes_spine_width": 0.8,
    "axes_spine_color": "0.2",

    # Tick settings
    "tick_labelsize": 7,
    "tick_major_width": 0.6,
    "tick_minor_width": 0.4,
    "tick_direction": "in",
    "tick_length_major": 3.5,
    "tick_length_minor": 2.0,

    # Legend settings
    "legend_fontsize": 6.5,
    "legend_framealpha": 0.9,
    "legend_frameon": False,
    "legend_edgecolor": "0.8",

    # Grid settings
    "grid_linestyle": ":",
    "grid_alpha": 0.15,
    "grid_linewidth": 0.4,

    # Line settings
    "line_width": 0.9,

    # Scatter settings
    "scatter_size_outside": 3,
    "scatter_size_inside": 8,
    "scatter_alpha_outside": 0.15,
    "scatter_alpha_inside": 0.7,
    "scatter_edgecolor": "none",

    # Figure-level settings
    "figure_facecolor": "white",
    "axes_facecolor": "white",
    "figure_dpi": 300,
}


def apply_plot_settings():
    """Apply global matplotlib settings from PLOT_SETTINGS dictionary."""
    plt.rcParams.update({
        "font.family": PLOT_SETTINGS["font_family"],
        "font.serif": PLOT_SETTINGS["font_serif"],
        "font.size": PLOT_SETTINGS["font_size"],
        "axes.labelsize": PLOT_SETTINGS["axes_labelsize"],
        "axes.titlesize": PLOT_SETTINGS["axes_titlesize"],
        "axes.facecolor": PLOT_SETTINGS["axes_facecolor"],
        "xtick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "ytick.labelsize": PLOT_SETTINGS["tick_labelsize"],
        "xtick.major.width": PLOT_SETTINGS["tick_major_width"],
        "ytick.major.width": PLOT_SETTINGS["tick_major_width"],
        "xtick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "ytick.minor.width": PLOT_SETTINGS["tick_minor_width"],
        "xtick.direction": PLOT_SETTINGS["tick_direction"],
        "ytick.direction": PLOT_SETTINGS["tick_direction"],
        "xtick.major.size": PLOT_SETTINGS["tick_length_major"],
        "ytick.major.size": PLOT_SETTINGS["tick_length_major"],
        "xtick.minor.size": PLOT_SETTINGS["tick_length_minor"],
        "ytick.minor.size": PLOT_SETTINGS["tick_length_minor"],
        "legend.fontsize": PLOT_SETTINGS["legend_fontsize"],
        "legend.framealpha": PLOT_SETTINGS["legend_framealpha"],
        "legend.frameon": PLOT_SETTINGS["legend_frameon"],
        "legend.edgecolor": PLOT_SETTINGS["legend_edgecolor"],
        "grid.linestyle": PLOT_SETTINGS["grid_linestyle"],
        "grid.alpha": PLOT_SETTINGS["grid_alpha"],
        "grid.linewidth": PLOT_SETTINGS["grid_linewidth"],
        "lines.linewidth": PLOT_SETTINGS["line_width"],
        "figure.facecolor": PLOT_SETTINGS["figure_facecolor"],
        "figure.dpi": PLOT_SETTINGS["figure_dpi"],
    })

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Class labels (PathMNIST)
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

# Color mapping for models
MODEL_COLORS = {
    'cfg-MedSyn': plt.cm.tab10(0),  # Blue
    'DistDiff': plt.cm.tab10(1)      # Orange
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ParetoPoint:
    """Single point on the Pareto frontier plot."""
    # Model metadata
    model_name: str  # "cfg-MedSyn" or "DistDiff"
    config_name: str  # "gamma1_temp20" or "DistDiff"

    # Global metrics
    fid_mean: float
    fid_std: float
    f1_mean: float
    f1_std: float

    # Per-class metrics (only for per-class mode)
    class_id: Optional[int] = None
    per_class_fid_mean: Optional[float] = None
    per_class_fid_std: Optional[float] = None
    per_class_f1_mean: Optional[float] = None
    per_class_f1_std: Optional[float] = None

    # Pareto status (set by compute_pareto_frontier)
    is_pareto_optimal: bool = False


# ============================================================================
# Data Loading Module
# ============================================================================

def load_fid_data(fid_csv_path: Path) -> Dict[str, Dict]:
    """
    Load FID results from aggregated CSV.

    Args:
        fid_csv_path: Path to aggregated_fid_comparison.csv

    Returns:
        Dictionary mapping config names to FID metrics:
        {
            'DistDiff': {
                'avg_fid_mean': float,
                'avg_fid_std': float,
                'per_class_fid_mean': [9 values],
                'per_class_fid_std': [9 values]
            },
            'gamma1_temp20': {...},
            ...
        }

    Raises:
        FileNotFoundError: If FID CSV doesn't exist
        ValueError: If CSV format is invalid
    """
    if not fid_csv_path.exists():
        raise FileNotFoundError(f"FID CSV not found: {fid_csv_path}")

    logger.info(f"Loading FID data from {fid_csv_path}")
    df = pd.read_csv(fid_csv_path)

    # Validate required columns
    if 'Configuration' not in df.columns:
        raise ValueError("FID CSV missing 'Configuration' column")
    if 'Average_Mean' not in df.columns:
        raise ValueError("FID CSV missing 'Average_Mean' column")

    fid_data = {}

    for _, row in df.iterrows():
        config = row['Configuration']

        # Extract global metrics
        avg_fid_mean = row['Average_Mean']
        avg_fid_std = row['Average_Std']

        # Extract per-class metrics
        per_class_fid_mean = []
        per_class_fid_std = []

        for class_idx in range(9):
            mean_col = f'Class_{class_idx}_Mean'
            std_col = f'Class_{class_idx}_Std'

            if mean_col not in row or std_col not in row:
                raise ValueError(f"Missing per-class columns for class {class_idx}")

            per_class_fid_mean.append(row[mean_col])
            per_class_fid_std.append(row[std_col])

        fid_data[config] = {
            'avg_fid_mean': avg_fid_mean,
            'avg_fid_std': avg_fid_std,
            'per_class_fid_mean': per_class_fid_mean,
            'per_class_fid_std': per_class_fid_std
        }

    logger.info(f"Loaded FID data for {len(fid_data)} configurations")
    return fid_data


def load_classification_data(results_dir: Path) -> pd.DataFrame:
    """
    Load global classification metrics (Macro-F1).

    Args:
        results_dir: Directory containing global_metrics_raw.csv

    Returns:
        DataFrame with columns:
            dataset, backbone, experiment, fold, metric_name, metric_value
        Filtered for metric_name == 'f1'

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    csv_path = results_dir / "global_metrics_raw.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Classification CSV not found: {csv_path}")

    logger.info(f"Loading classification data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter for F1 metric
    df = df[df['metric_name'] == 'f1'].copy()

    logger.info(f"Loaded {len(df)} F1 measurements")
    return df


def load_per_class_f1(results_dir: Path) -> pd.DataFrame:
    """
    Load per-class F1 scores.

    Args:
        results_dir: Directory containing per_class_f1_raw.csv

    Returns:
        DataFrame with columns:
            dataset, backbone, experiment, fold, class_index, metric_name, metric_value
        Filtered for metric_name == 'f1'

    Raises:
        FileNotFoundError: If CSV doesn't exist
    """
    csv_path = results_dir / "per_class_f1_raw.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Per-class CSV not found: {csv_path}")

    logger.info(f"Loading per-class F1 data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Filter for F1 metric
    df = df[df['metric_name'] == 'f1'].copy()

    logger.info(f"Loaded {len(df)} per-class F1 measurements")
    return df


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_macro_f1_global(
    df: pd.DataFrame,
    experiment: str
) -> Tuple[float, float]:
    """
    Aggregate Macro-F1 across ALL folds and backbones.

    Args:
        df: DataFrame from load_classification_data()
        experiment: 'real_plus_synth_distdiff' or 'real_plus_synth_cfgmedsyn'

    Returns:
        (mean_f1, std_f1) where:
        - mean_f1 = mean over (5 folds × 3 backbones) = 15 values
        - std_f1 = std dev over 15 values

    Raises:
        ValueError: If experiment not found or wrong number of values
    """
    subset = df[df['experiment'] == experiment]
    values = subset['metric_value'].values

    if len(values) == 0:
        raise ValueError(f"No data found for experiment '{experiment}'")

    if len(values) != 15:
        logger.warning(
            f"Expected 15 values (5 folds × 3 backbones) for {experiment}, "
            f"got {len(values)}"
        )

    mean_f1 = np.mean(values)
    std_f1 = np.std(values, ddof=1)  # Sample std dev

    logger.info(
        f"Aggregated {experiment}: F1 = {mean_f1:.4f} ± {std_f1:.4f} "
        f"(n={len(values)})"
    )

    return mean_f1, std_f1


def aggregate_macro_f1_per_class(
    df: pd.DataFrame,
    experiment: str,
    class_idx: int
) -> Tuple[float, float]:
    """
    Aggregate per-class F1 across all folds and backbones.

    Args:
        df: DataFrame from load_per_class_f1()
        experiment: Experiment name
        class_idx: Class index (0-8)

    Returns:
        (mean_f1, std_f1) for the given class
        - mean_f1 = mean over (5 folds × 3 backbones) = 15 values
        - std_f1 = std dev over 15 values

    Raises:
        ValueError: If no data found
    """
    subset = df[
        (df['experiment'] == experiment) &
        (df['class_index'] == class_idx)
    ]
    values = subset['metric_value'].values

    if len(values) == 0:
        raise ValueError(
            f"No data found for experiment '{experiment}', class {class_idx}"
        )

    if len(values) != 15:
        logger.warning(
            f"Expected 15 values for {experiment}, class {class_idx}, "
            f"got {len(values)}"
        )

    mean_f1 = np.mean(values)
    std_f1 = np.std(values, ddof=1)

    return mean_f1, std_f1


# ============================================================================
# Pareto Frontier Computation
# ============================================================================

def compute_pareto_frontier(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """
    Compute Pareto frontier for FID vs F1.

    A point is Pareto-optimal if no other point has:
    - Lower FID AND higher F1

    Args:
        points: List of ParetoPoint objects

    Returns:
        Updated points with is_pareto_optimal flag set
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            # For per-class mode, use per_class metrics
            if points[i].class_id is not None:
                fid_i = points[i].per_class_fid_mean
                fid_j = points[j].per_class_fid_mean
                f1_i = points[i].per_class_f1_mean
                f1_j = points[j].per_class_f1_mean
            else:
                fid_i = points[i].fid_mean
                fid_j = points[j].fid_mean
                f1_i = points[i].f1_mean
                f1_j = points[j].f1_mean

            better_fid = fid_j < fid_i
            better_f1 = f1_j > f1_i

            if better_fid and better_f1:
                # j strictly dominates i
                is_pareto[i] = False
                break

    for i, point in enumerate(points):
        point.is_pareto_optimal = is_pareto[i]

    num_pareto = sum(is_pareto)
    logger.info(f"Computed Pareto frontier: {num_pareto}/{n} points are optimal")

    return points


# ============================================================================
# Visualization Functions
# ============================================================================

def create_global_pareto_plot(
    points: List[ParetoPoint],
    output_path: Path
):
    """
    Create global Pareto frontier plot.

    X-axis: Mean per-class FID (left = better, right = worse)
    Y-axis: Macro-F1 (bottom = worse, top = better)

    Args:
        points: List of ParetoPoint objects
        output_path: Output path (without extension, will save .png and .pdf)
    """
    logger.info("Creating global Pareto frontier plot")

    apply_plot_settings()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Plot each point
    plotted_labels = set()

    for point in points:
        color = MODEL_COLORS[point.model_name]

        # Marker size and edge based on Pareto status
        if point.is_pareto_optimal:
            size = 120
            edge = 'black'
            linewidth = 2.0
            alpha = 1.0
            zorder = 10
        else:
            size = 60
            edge = 'none'
            linewidth = 0
            alpha = 0.6
            zorder = 5

        # Only add label once per model
        label = point.model_name if point.model_name not in plotted_labels else None
        if label:
            plotted_labels.add(point.model_name)

        # Scatter plot with error bars
        ax.errorbar(
            point.fid_mean,
            point.f1_mean,
            xerr=point.fid_std,
            yerr=point.f1_std,
            fmt='o',
            color=color,
            markersize=np.sqrt(size),
            markeredgecolor=edge,
            markeredgewidth=linewidth,
            alpha=alpha,
            capsize=3,
            capthick=1,
            elinewidth=1,
            zorder=zorder
        )

    # Axis labels
    ax.set_xlabel('Mean per-class FID ↓', fontsize=10)
    ax.set_ylabel('Downstream Macro-F1 ↑', fontsize=10)

    # Grid
    ax.grid(True, linestyle=':', alpha=0.3)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODEL_COLORS['cfg-MedSyn'], markersize=8,
               label='cfg-MedSyn', markeredgecolor='black', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODEL_COLORS['DistDiff'], markersize=8,
               label='DistDiff', markeredgecolor='black', markeredgewidth=1)
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=False, fontsize=9)

    plt.tight_layout()

    # Save PNG and PDF
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    logger.info(f"Saved global plot: {png_path}")
    logger.info(f"Saved global plot: {pdf_path}")

    plt.close(fig)


def create_per_class_pareto_plot(
    points_per_class: Dict[int, List[ParetoPoint]],
    output_path: Path
):
    """
    Create 3×3 grid of per-class Pareto frontiers.

    Args:
        points_per_class: Dictionary mapping class_idx to list of ParetoPoint
        output_path: Output path (without extension)
    """
    logger.info("Creating per-class Pareto frontier plot (3×3 grid)")

    apply_plot_settings()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=300)
    axes = axes.flatten()

    for class_idx in range(9):
        ax = axes[class_idx]
        points = points_per_class[class_idx]

        # Plot each point
        for point in points:
            color = MODEL_COLORS[point.model_name]

            if point.is_pareto_optimal:
                size = 100
                edge = 'black'
                linewidth = 1.5
                alpha = 1.0
                zorder = 10
            else:
                size = 50
                edge = 'none'
                linewidth = 0
                alpha = 0.6
                zorder = 5

            ax.errorbar(
                point.per_class_fid_mean,
                point.per_class_f1_mean,
                xerr=point.per_class_fid_std,
                yerr=point.per_class_f1_std,
                fmt='o',
                color=color,
                markersize=np.sqrt(size),
                markeredgecolor=edge,
                markeredgewidth=linewidth,
                alpha=alpha,
                capsize=2,
                capthick=0.8,
                elinewidth=0.8,
                zorder=zorder
            )

        # Subplot title
        ax.set_title(f'Class {class_idx}: {LABELS_SHORT[class_idx]}', fontsize=8)

        # Labels
        ax.set_xlabel('FID ↓', fontsize=7)
        ax.set_ylabel('F1 ↑', fontsize=7)

        # Grid
        ax.grid(True, linestyle=':', alpha=0.2)

        # Tick size
        ax.tick_params(labelsize=6)

    # Shared legend at bottom center
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODEL_COLORS['cfg-MedSyn'], markersize=6,
               label='cfg-MedSyn', markeredgecolor='black', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=MODEL_COLORS['DistDiff'], markersize=6,
               label='DistDiff', markeredgecolor='black', markeredgewidth=1)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.05)

    # Save
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    logger.info(f"Saved per-class plot: {png_path}")
    logger.info(f"Saved per-class plot: {pdf_path}")

    plt.close(fig)


# ============================================================================
# Output Management
# ============================================================================

def export_data_csv(points: List[ParetoPoint], output_path: Path):
    """
    Export Pareto points to CSV.

    Args:
        points: List of ParetoPoint objects
        output_path: Output CSV path
    """
    logger.info(f"Exporting data to {output_path}")

    rows = []
    for point in points:
        rows.append({
            'model_name': point.model_name,
            'config_name': point.config_name,
            'fid_mean': point.fid_mean,
            'fid_std': point.fid_std,
            'f1_mean': point.f1_mean,
            'f1_std': point.f1_std,
            'is_pareto_optimal': point.is_pareto_optimal
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved {len(rows)} data points to CSV")


def export_metadata(
    points: List[ParetoPoint],
    args: argparse.Namespace,
    output_path: Path
):
    """
    Export metadata to JSON.

    Args:
        points: List of ParetoPoint objects
        args: CLI arguments
        output_path: Output JSON path
    """
    logger.info(f"Exporting metadata to {output_path}")

    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    metadata = {
        'created_at': datetime.now().isoformat(),
        'script': __file__,
        'arguments': {
            'classification_results_dir': str(args.classification_results_dir),
            'fid_csv': str(args.fid_csv),
            'output_dir': str(args.output_dir),
            'per_class': args.per_class,
            'cfgmedsyn_config': args.cfgmedsyn_config
        },
        'num_points': int(len(points)),
        'num_pareto_optimal': int(sum(p.is_pareto_optimal for p in points)),
        'points': [convert_to_json_serializable(asdict(p)) for p in points]
    }

    # Convert entire metadata dict to ensure all numpy types are handled
    metadata = convert_to_json_serializable(metadata)

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to JSON")


# ============================================================================
# CLI and Main
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create Pareto frontier plot: FID vs Macro-F1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Global Pareto frontier
    python -m medsyn.analysis.pareto_fid_f1

    # With per-class plots
    python -m medsyn.analysis.pareto_fid_f1 --per-class

    # Custom config
    python -m medsyn.analysis.pareto_fid_f1 --cfgmedsyn-config gamma5_temp15
        """
    )

    parser.add_argument(
        '--classification-results-dir',
        type=Path,
        default=Path('/media/mpascual/Sandisk2TB/research/medsyn/results/downstream_analysis_test'),
        help='Directory with classification results CSVs (default: %(default)s)'
    )

    parser.add_argument(
        '--fid-csv',
        type=Path,
        default=Path('/media/mpascual/Sandisk2TB/research/medsyn/results/aggregated_fid_comparison.csv'),
        help='Path to aggregated FID CSV (default: %(default)s)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/media/mpascual/Sandisk2TB/research/medsyn/results'),
        help='Output directory for plots (default: %(default)s)'
    )

    parser.add_argument(
        '--per-class',
        action='store_true',
        help='Create per-class Pareto frontier plots (3×3 grid)'
    )

    parser.add_argument(
        '--cfgmedsyn-config',
        type=str,
        default='gamma1_temp20',
        help='Which cfg-MedSyn config to use (default: %(default)s)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()

    logger.info("="*70)
    logger.info("Pareto Frontier Plot: FID vs Macro-F1")
    logger.info("="*70)

    try:
        # ====================================================================
        # STEP 1: Load FID Data
        # ====================================================================
        logger.info("\n### STEP 1: Load FID Data ###")
        fid_data = load_fid_data(args.fid_csv)

        # Validate that required configs exist
        if 'DistDiff' not in fid_data:
            raise ValueError("FID CSV missing 'DistDiff' configuration")
        if args.cfgmedsyn_config not in fid_data:
            raise ValueError(
                f"FID CSV missing '{args.cfgmedsyn_config}' configuration. "
                f"Available configs: {list(fid_data.keys())}"
            )

        # ====================================================================
        # STEP 2: Load Classification Data
        # ====================================================================
        logger.info("\n### STEP 2: Load Classification Data ###")
        global_f1_df = load_classification_data(args.classification_results_dir)
        per_class_f1_df = load_per_class_f1(args.classification_results_dir)

        # ====================================================================
        # STEP 3: Aggregate Global Metrics
        # ====================================================================
        logger.info("\n### STEP 3: Aggregate Global Metrics ###")

        # DistDiff
        distdiff_f1_mean, distdiff_f1_std = aggregate_macro_f1_global(
            global_f1_df, 'real_plus_synth_distdiff'
        )
        distdiff_fid_mean = fid_data['DistDiff']['avg_fid_mean']
        distdiff_fid_std = fid_data['DistDiff']['avg_fid_std']

        # cfg-MedSyn
        cfgmedsyn_f1_mean, cfgmedsyn_f1_std = aggregate_macro_f1_global(
            global_f1_df, 'real_plus_synth_cfgmedsyn'
        )
        cfgmedsyn_config = args.cfgmedsyn_config
        cfgmedsyn_fid_mean = fid_data[cfgmedsyn_config]['avg_fid_mean']
        cfgmedsyn_fid_std = fid_data[cfgmedsyn_config]['avg_fid_std']

        # ====================================================================
        # STEP 4: Create Data Points
        # ====================================================================
        logger.info("\n### STEP 4: Create Data Points ###")

        points = [
            ParetoPoint(
                model_name='DistDiff',
                config_name='DistDiff',
                fid_mean=distdiff_fid_mean,
                fid_std=distdiff_fid_std,
                f1_mean=distdiff_f1_mean,
                f1_std=distdiff_f1_std
            ),
            ParetoPoint(
                model_name='cfg-MedSyn',
                config_name=cfgmedsyn_config,
                fid_mean=cfgmedsyn_fid_mean,
                fid_std=cfgmedsyn_fid_std,
                f1_mean=cfgmedsyn_f1_mean,
                f1_std=cfgmedsyn_f1_std
            )
        ]

        logger.info(f"DistDiff: FID={distdiff_fid_mean:.2f}±{distdiff_fid_std:.2f}, "
                   f"F1={distdiff_f1_mean:.4f}±{distdiff_f1_std:.4f}")
        logger.info(f"cfg-MedSyn ({cfgmedsyn_config}): "
                   f"FID={cfgmedsyn_fid_mean:.2f}±{cfgmedsyn_fid_std:.2f}, "
                   f"F1={cfgmedsyn_f1_mean:.4f}±{cfgmedsyn_f1_std:.4f}")

        # ====================================================================
        # STEP 5: Compute Pareto Frontier
        # ====================================================================
        logger.info("\n### STEP 5: Compute Pareto Frontier ###")
        points = compute_pareto_frontier(points)

        # ====================================================================
        # STEP 6: Create Global Plot
        # ====================================================================
        logger.info("\n### STEP 6: Create Global Plot ###")
        output_path = args.output_dir / 'pareto_fid_f1_global'
        create_global_pareto_plot(points, output_path)

        # ====================================================================
        # STEP 7: Create Per-Class Plots (Optional)
        # ====================================================================
        if args.per_class:
            logger.info("\n### STEP 7: Create Per-Class Plots ###")

            # Aggregate per-class metrics for each class
            points_per_class = {}

            for class_idx in range(9):
                logger.info(f"Processing class {class_idx}: {LABELS_SHORT[class_idx]}")

                # DistDiff
                distdiff_f1_mean_c, distdiff_f1_std_c = aggregate_macro_f1_per_class(
                    per_class_f1_df, 'real_plus_synth_distdiff', class_idx
                )

                # cfg-MedSyn
                cfgmedsyn_f1_mean_c, cfgmedsyn_f1_std_c = aggregate_macro_f1_per_class(
                    per_class_f1_df, 'real_plus_synth_cfgmedsyn', class_idx
                )

                points_per_class[class_idx] = [
                    ParetoPoint(
                        model_name='DistDiff',
                        config_name='DistDiff',
                        fid_mean=0, f1_mean=0, fid_std=0, f1_std=0,
                        class_id=class_idx,
                        per_class_fid_mean=fid_data['DistDiff']['per_class_fid_mean'][class_idx],
                        per_class_fid_std=fid_data['DistDiff']['per_class_fid_std'][class_idx],
                        per_class_f1_mean=distdiff_f1_mean_c,
                        per_class_f1_std=distdiff_f1_std_c
                    ),
                    ParetoPoint(
                        model_name='cfg-MedSyn',
                        config_name=cfgmedsyn_config,
                        fid_mean=0, f1_mean=0, fid_std=0, f1_std=0,
                        class_id=class_idx,
                        per_class_fid_mean=fid_data[cfgmedsyn_config]['per_class_fid_mean'][class_idx],
                        per_class_fid_std=fid_data[cfgmedsyn_config]['per_class_fid_std'][class_idx],
                        per_class_f1_mean=cfgmedsyn_f1_mean_c,
                        per_class_f1_std=cfgmedsyn_f1_std_c
                    )
                ]

                # Compute Pareto for this class
                points_per_class[class_idx] = compute_pareto_frontier(
                    points_per_class[class_idx]
                )

            output_path_per_class = args.output_dir / 'pareto_fid_f1_per_class'
            create_per_class_pareto_plot(points_per_class, output_path_per_class)

        # ====================================================================
        # STEP 8: Export Data
        # ====================================================================
        logger.info("\n### STEP 8: Export Data ###")
        export_data_csv(points, args.output_dir / 'pareto_fid_f1_data.csv')
        export_metadata(points, args, args.output_dir / 'pareto_fid_f1_metadata.json')

        # ====================================================================
        # DONE!
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("DONE! All plots and data exported successfully.")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
