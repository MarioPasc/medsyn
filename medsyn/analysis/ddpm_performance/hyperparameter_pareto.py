#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ccDDPM Hyperparameter Performance Comparison with Pareto Frontier Visualization

Creates two publication-quality Pareto frontier plots for ccDDPM hyperparameter search:
1. SSIM vs PSNR (Max-Max optimization)
2. Weighted Loss vs Final Epoch (Min-Min optimization)

Usage:
    # Global Pareto frontier (default)
    python -m medsyn.analysis.ddpm_performance.hyperparameter_pareto \
        --experiments-dir /media/mpascual/Sandisk2TB/research/medsyn/experiments/hyperparameter_search \
        --output-dir /media/mpascual/Sandisk2TB/research/medsyn/experiments/hyperparameter_search/analysis

    # Per-class Pareto frontiers
    python -m medsyn.analysis.ddpm_performance.hyperparameter_pareto \
        --experiments-dir /media/mpascual/Sandisk2TB/research/medsyn/experiments/hyperparameter_search \
        --output-dir /media/mpascual/Sandisk2TB/research/medsyn/experiments/hyperparameter_search/analysis \
        --per-class-pareto
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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

# Import visual design settings from existing codebase
from medsyn.analysis.embeddings.time_sensitive_anisotropy_evolution import (
    PLOT_SETTINGS,
    apply_plot_settings,
)

# Setup logger
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

# Symbol mapping (10 classes + global)
CLASS_SYMBOLS = {
    None: 'o',   # Global: circle
    0: 's',      # Adipose: square
    1: '^',      # Background: triangle up
    2: 'v',      # Debris: triangle down
    3: 'D',      # Lymphocytes: diamond
    4: 'p',      # Mucus: pentagon
    5: '*',      # Smooth Muscle: star
    6: 'X',      # Normal Mucosa: X
    7: 'P',      # Cancer Stroma: plus
    8: 'h',      # Colorectal Epithelium: hexagon
}

# Experiment configurations for consistent color mapping
EXPERIMENT_CONFIGS = [
    (1, 1.0), (1, 1.5), (1, 2.0),
    (5, 1.0), (5, 1.5), (5, 2.0),
    (10, 1.0), (10, 1.5), (10, 2.0)
]


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DataPoint:
    """Single data point for plotting (70 total: 7 experiments × 10 points)."""
    # Experiment metadata
    experiment_name: str  # "γ=1, T=1.5"
    gamma: float
    temperature: float

    # Class metadata
    class_id: Optional[int]  # None for global, 0-8 for classes
    class_label: str  # "Global", "Adipose", etc.

    # Metrics for Plot 1 (SSIM vs PSNR)
    psnr: float
    ssim: float

    # Metrics for Plot 2 (Loss vs Epoch)
    weighted_loss: float
    final_epoch: int

    # Pareto status (set by assign_pareto_optimality)
    is_pareto_optimal: bool = False


# ============================================================================
# Data Loading Module
# ============================================================================

def discover_experiments(base_dir: Path) -> List[Tuple[Path, float, float]]:
    """
    Scan directory for experiment folders matching pattern:
    gamma{1,5,10}_temp{10,15,20}_training

    Args:
        base_dir: Directory containing experiment folders

    Returns:
        List of (folder_path, gamma, temperature) tuples sorted by (gamma, temp)

    Raises:
        ValueError: If no experiments found
    """
    pattern = re.compile(r'gamma(\d+)_temp(\d+)_training')
    experiments = []

    for folder in base_dir.iterdir():
        if not folder.is_dir():
            continue
        match = pattern.match(folder.name)
        if match:
            gamma = int(match.group(1))
            temp_code = int(match.group(2))
            temperature = temp_code / 10.0  # temp15 -> 1.5
            experiments.append((folder, gamma, temperature))

    # Validation
    if len(experiments) == 0:
        raise ValueError(f"No experiments found in {base_dir}")

    if len(experiments) != 7:
        logger.warning(f"Expected 7 experiments, found {len(experiments)}")

    # Sort by (gamma, temperature)
    experiments = sorted(experiments, key=lambda x: (x[1], x[2]))

    logger.info(f"Discovered {len(experiments)} experiments:")
    for exp_dir, gamma, temp in experiments:
        logger.info(f"  - γ={gamma}, T={temp}: {exp_dir.name}")

    return experiments


def load_experiment_data(exp_dir: Path, gamma: float, temp: float) -> pd.DataFrame:
    """
    Load training_metrics.csv and validate required columns.

    Args:
        exp_dir: Experiment directory path
        gamma: MinSNR gamma value
        temp: Temperature value for per-class loss weighting

    Returns:
        DataFrame with metadata columns added: gamma, temperature, experiment_name

    Raises:
        FileNotFoundError: If training_metrics.csv doesn't exist
        ValueError: If required columns are missing
    """
    csv_path = exp_dir / "training_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate required columns
    required = [
        'epoch', 'split', 'psnr', 'ssim', 'loss', 'best_val_score',
        # Per-class PSNR: psnr_c0...psnr_c8
        *[f'psnr_c{i}' for i in range(9)],
        # Per-class SSIM: ssim_c0...ssim_c8
        *[f'ssim_c{i}' for i in range(9)],
        # Per-class weighted loss: loss_weighted_c0...loss_weighted_c8
        *[f'loss_weighted_c{i}' for i in range(9)]
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    # Add metadata
    df['gamma'] = gamma
    df['temperature'] = temp
    df['experiment_name'] = f"γ={gamma}, T={temp}"

    logger.debug(f"Loaded {len(df)} rows from {exp_dir.name}")

    return df


def extract_best_epoch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract metrics from best validation epoch for test split.

    Strategy:
    1. Find epoch with max best_val_score from validation split
    2. Return test split metrics for that epoch

    Args:
        df: DataFrame with training metrics

    Returns:
        Single-row DataFrame with test split metrics for best epoch

    Raises:
        ValueError: If no test split found for best epoch
    """
    val_df = df[df['split'] == 'val'].copy()

    if 'best_val_score' in val_df.columns and not val_df['best_val_score'].isna().all():
        # Find best epoch using validation score
        best_epoch = val_df.loc[val_df['best_val_score'].idxmax(), 'epoch']
        logger.debug(f"Best epoch: {best_epoch} (max best_val_score)")
    else:
        # Fallback: use epoch with lowest validation loss
        logger.warning("best_val_score not found or all NaN, using min validation loss")
        best_epoch = val_df.loc[val_df['loss'].idxmin(), 'epoch']
        logger.debug(f"Best epoch: {best_epoch} (min validation loss)")

    # Extract test split for best epoch
    test_row = df[(df['split'] == 'test') & (df['epoch'] == best_epoch)]

    if len(test_row) == 0:
        raise ValueError(f"No test split found for best epoch {best_epoch}")

    return test_row.iloc[0:1]


def build_dataset(experiments: List[Tuple[Path, float, float]]) -> List[DataPoint]:
    """
    Load all experiments and create 70 data points.

    For each experiment:
    - 1 global point (using psnr, ssim, loss from best test epoch)
    - 9 class points (using psnr_c*, ssim_c*, loss_weighted_c*)

    Args:
        experiments: List of (folder_path, gamma, temperature) tuples

    Returns:
        List of 70 DataPoint objects (7 experiments × 10 points)
    """
    data_points = []

    for exp_dir, gamma, temp in experiments:
        # Load and extract best epoch
        df = load_experiment_data(exp_dir, gamma, temp)
        best_row = extract_best_epoch_metrics(df)

        exp_name = f"γ={gamma}, T={temp}"
        final_epoch = int(best_row['epoch'].values[0])

        # Global point
        global_pt = DataPoint(
            experiment_name=exp_name,
            gamma=gamma,
            temperature=temp,
            class_id=None,
            class_label="Global",
            psnr=float(best_row['psnr'].values[0]),
            ssim=float(best_row['ssim'].values[0]),
            weighted_loss=float(best_row['loss'].values[0]),
            final_epoch=final_epoch
        )
        data_points.append(global_pt)

        # Per-class points (c0-c8)
        for class_idx in range(9):
            class_pt = DataPoint(
                experiment_name=exp_name,
                gamma=gamma,
                temperature=temp,
                class_id=class_idx,
                class_label=LABELS_SHORT[class_idx],
                psnr=float(best_row[f'psnr_c{class_idx}'].values[0]),
                ssim=float(best_row[f'ssim_c{class_idx}'].values[0]),
                weighted_loss=float(best_row[f'loss_weighted_c{class_idx}'].values[0]),
                final_epoch=final_epoch
            )
            data_points.append(class_pt)

    logger.info(f"Built {len(data_points)} data points from {len(experiments)} experiments")
    return data_points


# ============================================================================
# Pareto Frontier Computation Module
# ============================================================================

def compute_pareto_frontier(
    points: np.ndarray,
    maximize: Tuple[bool, bool] = (True, True)
) -> np.ndarray:
    """
    Compute Pareto-optimal indices for 2D points.

    Args:
        points: (N, 2) array of (x, y) coordinates
        maximize: (bool, bool) for each dimension
                  (True, True) = max-max (SSIM vs PSNR)
                  (False, False) = min-min (Loss vs Epoch)

    Returns:
        Boolean array of length N, True for Pareto-optimal points

    Pareto Definition (max-max):
        Point P is optimal if no other point Q satisfies:
        Q.x >= P.x AND Q.y >= P.y AND (Q.x > P.x OR Q.y > P.y)
        (i.e., Q dominates or equals P in both dimensions, and strictly dominates in at least one)

    Complexity: O(N²) brute force (acceptable for N=70)
    """
    N = len(points)
    is_pareto = np.ones(N, dtype=bool)

    # Transform for minimization if needed
    adjusted = points.copy()
    if not maximize[0]:
        adjusted[:, 0] = -adjusted[:, 0]
    if not maximize[1]:
        adjusted[:, 1] = -adjusted[:, 1]

    # Check each point against all others
    for i in range(N):
        if not is_pareto[i]:
            continue

        for j in range(N):
            if i == j or not is_pareto[j]:
                continue

            # Does j dominate i?
            if (adjusted[j, 0] >= adjusted[i, 0] and
                adjusted[j, 1] >= adjusted[i, 1] and
                (adjusted[j, 0] > adjusted[i, 0] or
                 adjusted[j, 1] > adjusted[i, 1])):
                is_pareto[i] = False
                break

    return is_pareto


def assign_pareto_optimality(
    data_points: List[DataPoint],
    per_class: bool = False
) -> None:
    """
    Assign is_pareto_optimal flag to each DataPoint (modifies in-place).

    Two modes:
    1. per_class=False: Frontier computed ONLY from global points (7 experiments)
                        Classes are shown for information but with alpha=0.5
    2. per_class=True: 10 separate frontiers (one per class + global)

    A point is marked optimal if it's on the frontier for EITHER plot.

    Args:
        data_points: List of DataPoint objects
        per_class: If True, compute separate frontiers per class
    """
    if per_class:
        # Group by class_label
        groups = {}
        for pt in data_points:
            key = pt.class_label
            groups.setdefault(key, []).append(pt)

        # Compute frontier per group
        for class_label, points in groups.items():
            # Plot 1: SSIM vs PSNR (max-max)
            coords_1 = np.array([[pt.psnr, pt.ssim] for pt in points])
            pareto_1 = compute_pareto_frontier(coords_1, maximize=(True, True))

            # Plot 2: Weighted Loss vs Epoch (min-min)
            coords_2 = np.array([[pt.final_epoch, pt.weighted_loss] for pt in points])
            pareto_2 = compute_pareto_frontier(coords_2, maximize=(False, False))

            # Mark points on either frontier
            for i, pt in enumerate(points):
                pt.is_pareto_optimal = bool(pareto_1[i] or pareto_2[i])

    else:
        # Global mode: compute frontier ONLY from global points
        global_points = [pt for pt in data_points if pt.class_id is None]

        # Plot 1: SSIM vs PSNR (max-max)
        coords_1 = np.array([[pt.psnr, pt.ssim] for pt in global_points])
        pareto_1 = compute_pareto_frontier(coords_1, maximize=(True, True))

        # Plot 2: Weighted Loss vs Epoch (min-min)
        coords_2 = np.array([[pt.final_epoch, pt.weighted_loss] for pt in global_points])
        pareto_2 = compute_pareto_frontier(coords_2, maximize=(False, False))

        # Mark global points on either frontier
        for i, pt in enumerate(global_points):
            pt.is_pareto_optimal = bool(pareto_1[i] or pareto_2[i])

        # All class points are NOT on Pareto frontier in global mode
        for pt in data_points:
            if pt.class_id is not None:
                pt.is_pareto_optimal = False

    num_pareto = sum(1 for pt in data_points if pt.is_pareto_optimal)
    logger.info(f"Marked {num_pareto}/{len(data_points)} Pareto-optimal points")


# ============================================================================
# Visualization Module
# ============================================================================

def get_experiment_color(gamma: float, temp: float) -> tuple:
    """
    Map (gamma, temp) to consistent color from tab10.

    Args:
        gamma: MinSNR gamma value
        temp: Temperature value

    Returns:
        RGBA color tuple
    """
    try:
        idx = EXPERIMENT_CONFIGS.index((gamma, temp))
        return plt.cm.tab10(idx)
    except ValueError:
        logger.warning(f"Unknown experiment config: γ={gamma}, T={temp}, using gray")
        return (0.5, 0.5, 0.5, 1.0)  # Gray fallback


def create_shared_legend(
    fig: plt.Figure,
    data_points: List[DataPoint],
    per_class_mode: bool
) -> None:
    """
    Create shared legend at bottom center of figure.

    Args:
        fig: Figure object
        data_points: List of DataPoint objects
        per_class_mode: Whether per-class Pareto mode is active
    """
    # Part 1: Experiment colors
    exp_configs = sorted(set((pt.gamma, pt.temperature) for pt in data_points))
    exp_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=get_experiment_color(g, t),
               markersize=8, label=f'γ={int(g)}, T={t}')
        for g, t in exp_configs
    ]

    # Part 2: Class symbols
    class_configs = sorted(
        set((pt.class_id, pt.class_label) for pt in data_points),
        key=lambda x: (x[0] is not None, x[0])  # Global first
    )
    class_handles = [
        Line2D([0], [0], marker=CLASS_SYMBOLS[cid], color='w',
               markerfacecolor='gray', markersize=8, label=label)
        for cid, label in class_configs
    ]

    # Combine handles and labels
    all_handles = exp_handles + [Line2D([0], [0], color='w', label='')] + class_handles
    all_labels = [h.get_label() for h in all_handles]

    # Create shared legend at bottom
    fig.legend(
        all_handles,
        all_labels,
        loc='lower center',
        ncol=min(len(all_labels), 9),
        bbox_to_anchor=(0.5, -0.05),
        frameon=PLOT_SETTINGS["legend_frameon"],
        fontsize=PLOT_SETTINGS["legend_fontsize"] + 2,
        columnspacing=1.0
    )

    # Add mode annotation
    mode_text = "Per-class Pareto" if per_class_mode else "Global Pareto (classes shown for reference)"
    fig.text(0.99, 0.01, mode_text, ha='right', va='bottom',
             fontsize=9, style='italic', color='gray')


def create_combined_plot(
    data_points: List[DataPoint],
    output_path: Path,
    per_class_mode: bool
) -> None:
    """
    Generate combined Pareto frontier plot with two subplots.

    Visual encoding:
    - Color: experiment (7 colors)
    - Symbol: class (10 symbols)
    - Alpha: Pareto status (1.0) vs classes/non-Pareto (0.5)
    - Edge: black outline for Pareto points

    Args:
        data_points: List of DataPoint objects
        output_path: Output path (without extension)
        per_class_mode: Whether per-class Pareto mode is active
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Filter valid points
    valid_points_plot1 = [pt for pt in data_points
                          if (np.isfinite(pt.psnr) and np.isfinite(pt.ssim))]
    valid_points_plot2 = [pt for pt in data_points
                          if (np.isfinite(pt.final_epoch) and np.isfinite(pt.weighted_loss))]

    logger.info(f"Plotting {len(valid_points_plot1)}/{len(data_points)} valid points for SSIM vs PSNR")
    logger.info(f"Plotting {len(valid_points_plot2)}/{len(data_points)} valid points for Loss vs Epoch")

    # Plot 1: SSIM vs PSNR
    for pt in valid_points_plot1:
        color = get_experiment_color(pt.gamma, pt.temperature)
        symbol = CLASS_SYMBOLS[pt.class_id]

        # In global mode, classes always have alpha=0.5
        if pt.is_pareto_optimal:
            size = PLOT_SETTINGS["scatter_size_pareto"]
            alpha = PLOT_SETTINGS["scatter_alpha_pareto"]
            edge = 'black'
            linewidth = 1.5
        else:
            size = PLOT_SETTINGS["scatter_size_normal"]
            alpha = PLOT_SETTINGS["scatter_alpha_normal"]
            edge = 'none'
            linewidth = 0

        ax1.scatter(pt.psnr, pt.ssim, marker=symbol, color=color, s=size,
                    alpha=alpha, edgecolors=edge, linewidths=linewidth)

    ax1.set_xlabel('PSNR (dB)', fontsize=PLOT_SETTINGS["axis_labelsize"])
    ax1.set_ylabel('SSIM', fontsize=PLOT_SETTINGS["axis_labelsize"])
    ax1.set_title('SSIM vs PSNR (Max-Max)', fontsize=PLOT_SETTINGS["title_fontsize"], fontweight='bold')
    ax1.grid(True, alpha=PLOT_SETTINGS["grid_alpha"],
             linestyle=PLOT_SETTINGS["grid_linestyle"],
             linewidth=PLOT_SETTINGS["grid_linewidth"])
    ax1.tick_params(labelsize=PLOT_SETTINGS["tick_labelsize"])

    # Plot 2: Loss vs Epoch
    for pt in valid_points_plot2:
        color = get_experiment_color(pt.gamma, pt.temperature)
        symbol = CLASS_SYMBOLS[pt.class_id]

        if pt.is_pareto_optimal:
            size = PLOT_SETTINGS["scatter_size_pareto"]
            alpha = PLOT_SETTINGS["scatter_alpha_pareto"]
            edge = 'black'
            linewidth = 1.5
        else:
            size = PLOT_SETTINGS["scatter_size_normal"]
            alpha = PLOT_SETTINGS["scatter_alpha_normal"]
            edge = 'none'
            linewidth = 0

        ax2.scatter(pt.final_epoch, pt.weighted_loss, marker=symbol, color=color, s=size,
                    alpha=alpha, edgecolors=edge, linewidths=linewidth)

    ax2.set_xlabel('Final Epoch', fontsize=PLOT_SETTINGS["axis_labelsize"])
    ax2.set_ylabel('Weighted Loss', fontsize=PLOT_SETTINGS["axis_labelsize"])
    ax2.set_title('Weighted Loss vs Convergence (Min-Min)', fontsize=PLOT_SETTINGS["title_fontsize"], fontweight='bold')
    ax2.grid(True, alpha=PLOT_SETTINGS["grid_alpha"],
             linestyle=PLOT_SETTINGS["grid_linestyle"],
             linewidth=PLOT_SETTINGS["grid_linewidth"])
    ax2.tick_params(labelsize=PLOT_SETTINGS["tick_labelsize"])

    # Add shared legend at bottom
    create_shared_legend(fig, data_points, per_class_mode)

    # Save both PNG and PDF
    plt.tight_layout()
    fig.savefig(str(output_path) + '.png', dpi=PLOT_SETTINGS["figure_dpi"],
                bbox_inches='tight')
    fig.savefig(str(output_path) + '.pdf', bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Saved: {output_path}.png and {output_path}.pdf")


# ============================================================================
# Ranking Generation
# ============================================================================

def generate_ranking_csv(
    data_points: List[DataPoint],
    output_path: Path
) -> None:
    """
    Generate ranking CSV with experiment-level statistics.

    For each experiment (gamma, temperature combination):
    - Count times on global Pareto frontier (for both plots)
    - Count times on per-class Pareto frontier (for both plots)
    - Global metrics: SSIM, PSNR, loss, final_epoch
    - Per-class averages: avg SSIM, PSNR, loss, final_epoch across 9 classes

    Args:
        data_points: List of DataPoint objects (with Pareto assignments computed for both modes)
        output_path: Output CSV path
    """
    # Re-compute Pareto frontiers for both modes to get counts
    # Global mode counts
    global_points = [pt for pt in data_points if pt.class_id is None]

    # SSIM vs PSNR
    coords_1 = np.array([[pt.psnr, pt.ssim] for pt in global_points])
    pareto_1_global = compute_pareto_frontier(coords_1, maximize=(True, True))

    # Loss vs Epoch
    coords_2 = np.array([[pt.final_epoch, pt.weighted_loss] for pt in global_points])
    pareto_2_global = compute_pareto_frontier(coords_2, maximize=(False, False))

    # Create dict mapping (gamma, temp) to global Pareto counts
    global_pareto_counts = {}
    for i, pt in enumerate(global_points):
        key = (pt.gamma, pt.temperature)
        if key not in global_pareto_counts:
            global_pareto_counts[key] = {'ssim_psnr': 0, 'loss_epoch': 0}
        if pareto_1_global[i]:
            global_pareto_counts[key]['ssim_psnr'] += 1
        if pareto_2_global[i]:
            global_pareto_counts[key]['loss_epoch'] += 1

    # Per-class mode counts
    per_class_pareto_counts = {}
    groups = {}
    for pt in data_points:
        groups.setdefault(pt.class_label, []).append(pt)

    for class_label, points in groups.items():
        # SSIM vs PSNR
        coords_1 = np.array([[pt.psnr, pt.ssim] for pt in points])
        pareto_1 = compute_pareto_frontier(coords_1, maximize=(True, True))

        # Loss vs Epoch
        coords_2 = np.array([[pt.final_epoch, pt.weighted_loss] for pt in points])
        pareto_2 = compute_pareto_frontier(coords_2, maximize=(False, False))

        for i, pt in enumerate(points):
            key = (pt.gamma, pt.temperature)
            if key not in per_class_pareto_counts:
                per_class_pareto_counts[key] = {'ssim_psnr': 0, 'loss_epoch': 0}
            if pareto_1[i]:
                per_class_pareto_counts[key]['ssim_psnr'] += 1
            if pareto_2[i]:
                per_class_pareto_counts[key]['loss_epoch'] += 1

    # Build ranking data
    ranking_data = []
    experiments = sorted(set((pt.gamma, pt.temperature) for pt in data_points))

    for gamma, temp in experiments:
        # Get global point
        global_pt = next(pt for pt in data_points
                         if pt.gamma == gamma and pt.temperature == temp and pt.class_id is None)

        # Get class points
        class_pts = [pt for pt in data_points
                     if pt.gamma == gamma and pt.temperature == temp and pt.class_id is not None]

        # Compute averages across classes
        avg_ssim = np.mean([pt.ssim for pt in class_pts])
        avg_psnr = np.mean([pt.psnr for pt in class_pts])
        avg_loss = np.mean([pt.weighted_loss for pt in class_pts])
        avg_epoch = class_pts[0].final_epoch if class_pts else global_pt.final_epoch  # Same for all

        ranking_data.append({
            'gamma': gamma,
            'tau': temp,
            'count_global_pareto_ssim_psnr': global_pareto_counts.get((gamma, temp), {}).get('ssim_psnr', 0),
            'count_global_pareto_loss_epoch': global_pareto_counts.get((gamma, temp), {}).get('loss_epoch', 0),
            'count_per_class_pareto_ssim_psnr': per_class_pareto_counts.get((gamma, temp), {}).get('ssim_psnr', 0),
            'count_per_class_pareto_loss_epoch': per_class_pareto_counts.get((gamma, temp), {}).get('loss_epoch', 0),
            'global_ssim': global_pt.ssim,
            'global_psnr': global_pt.psnr,
            'global_loss': global_pt.weighted_loss,
            'global_final_epoch': global_pt.final_epoch,
            'avg_class_ssim': avg_ssim,
            'avg_class_psnr': avg_psnr,
            'avg_class_loss': avg_loss,
            'avg_class_final_epoch': avg_epoch
        })

    # Convert to DataFrame and save
    ranking_df = pd.DataFrame(ranking_data)

    # Sort by total Pareto appearances (both global counts)
    ranking_df['total_global_pareto'] = (ranking_df['count_global_pareto_ssim_psnr'] +
                                          ranking_df['count_global_pareto_loss_epoch'])
    ranking_df = ranking_df.sort_values('total_global_pareto', ascending=False)
    ranking_df = ranking_df.drop('total_global_pareto', axis=1)

    ranking_df.to_csv(output_path, index=False)
    logger.info(f"Saved ranking: {output_path}")


# ============================================================================
# CLI and Output Management
# ============================================================================

def get_output_paths(output_dir: Path, per_class_mode: bool) -> Dict[str, Path]:
    """
    Generate output file paths.

    Args:
        output_dir: Output directory
        per_class_mode: Whether per-class Pareto mode is active

    Returns:
        Dictionary mapping output types to paths
    """
    suffix = "_per_class" if per_class_mode else "_global"

    return {
        "data_csv": output_dir / f"hyperparameter_data{suffix}.csv",
        "data_json": output_dir / f"hyperparameter_data{suffix}.json",
        "combined_plot": output_dir / f"pareto_combined{suffix}",  # .png/.pdf added later
        "ranking_csv": output_dir / "hyperparameter_ranking.csv",  # Same for both modes
        "metadata": output_dir / f"analysis_metadata{suffix}.json",
    }


def export_data(data_points: List[DataPoint], output_paths: Dict[str, Path]) -> None:
    """
    Export processed data to CSV and JSON.

    Args:
        data_points: List of DataPoint objects
        output_paths: Dictionary of output paths
    """
    # Convert to DataFrame
    data_dicts = [asdict(pt) for pt in data_points]
    df = pd.DataFrame(data_dicts)

    # Save CSV
    df.to_csv(output_paths["data_csv"], index=False)
    logger.info(f"Saved: {output_paths['data_csv']}")

    # Save JSON
    with open(output_paths["data_json"], 'w') as f:
        json.dump(data_dicts, f, indent=2)
    logger.info(f"Saved: {output_paths['data_json']}")


def write_metadata(
    experiments: List[Tuple[Path, float, float]],
    per_class_mode: bool,
    output_paths: Dict[str, Path]
) -> None:
    """
    Write analysis metadata.

    Args:
        experiments: List of (folder_path, gamma, temperature) tuples
        per_class_mode: Whether per-class Pareto mode is active
        output_paths: Dictionary of output paths
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "pareto_mode": "per_class" if per_class_mode else "global",
        "pareto_computation": (
            "10 separate frontiers (one per class + global)"
            if per_class_mode else
            "Frontier computed from 7 global points only (classes shown for reference with alpha=0.5)"
        ),
        "num_experiments": len(experiments),
        "num_data_points": len(experiments) * 10,
        "experiments": [
            {"gamma": g, "temperature": t, "folder": str(p.name)}
            for p, g, t in experiments
        ],
        "combined_plot": {
            "subplot1": {
                "title": "SSIM vs PSNR",
                "x_axis": "PSNR (dB)",
                "y_axis": "SSIM",
                "optimization": "max-max"
            },
            "subplot2": {
                "title": "Weighted Loss vs Final Epoch",
                "x_axis": "Final Epoch",
                "y_axis": "Weighted Loss",
                "optimization": "min-min (faster convergence, lower loss)"
            },
            "legend": "Shared at bottom center"
        },
        "visual_encoding": {
            "colors": "7 experiments (tab10 colormap)",
            "symbols": "10 classes + global (various markers)",
            "alpha": "1.0 (Pareto optimal) vs 0.5 (non-Pareto or classes in global mode)",
            "edge": "black outline for Pareto points"
        },
        "outputs": {
            "combined_plot": str(output_paths["combined_plot"]) + ".png/.pdf",
            "ranking_csv": str(output_paths["ranking_csv"]),
            "data_export": str(output_paths["data_csv"])
        },
        "library_versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__
        }
    }

    with open(output_paths["metadata"], 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {output_paths['metadata']}")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="ccDDPM Hyperparameter Pareto Frontier Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Global Pareto frontier (default)
  python -m medsyn.analysis.ddpm_performance.hyperparameter_pareto \\
      --experiments-dir /media/.../hyperparameter_search \\
      --output-dir /media/.../hyperparameter_search/analysis

  # Per-class Pareto frontiers
  python -m medsyn.analysis.ddpm_performance.hyperparameter_pareto \\
      --experiments-dir /media/.../hyperparameter_search \\
      --output-dir /media/.../hyperparameter_search/analysis \\
      --per-class-pareto
        """
    )

    parser.add_argument(
        "--experiments-dir", type=Path, required=True,
        help="Directory containing gamma*_temp*_training folders"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for plots and data"
    )
    parser.add_argument(
        "--per-class-pareto", action="store_true",
        help="Compute separate frontiers per class (default: global)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity"
    )

    return parser.parse_args()


# ============================================================================
# Main Orchestration
# ============================================================================

def main() -> None:
    """CLI entry point."""
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    logger.info("="*80)
    logger.info("ccDDPM Hyperparameter Pareto Frontier Analysis")
    logger.info("="*80)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover experiments
    logger.info("\n### STEP 1: Discovering Experiments ###")
    experiments = discover_experiments(args.experiments_dir)
    logger.info(f"Found {len(experiments)} experiments")

    # Step 2: Load and process data
    logger.info("\n### STEP 2: Loading Data ###")
    data_points = build_dataset(experiments)

    # Step 3: Compute Pareto frontiers
    logger.info("\n### STEP 3: Computing Pareto Frontiers ###")
    assign_pareto_optimality(data_points, per_class=args.per_class_pareto)

    # Step 4: Generate outputs
    output_paths = get_output_paths(args.output_dir, args.per_class_pareto)

    logger.info("\n### STEP 4: Exporting Data ###")
    export_data(data_points, output_paths)

    logger.info("\n### STEP 5: Creating Visualizations ###")
    # Apply plot settings with customizations
    PLOT_SETTINGS["tick_labelsize"] = 12
    PLOT_SETTINGS["axis_labelsize"] = 14
    PLOT_SETTINGS["grid_alpha"] = 0.7
    PLOT_SETTINGS["title_fontsize"] = 16
    PLOT_SETTINGS["scatter_size_pareto"] = 80
    PLOT_SETTINGS["scatter_size_normal"] = 40
    PLOT_SETTINGS["scatter_alpha_pareto"] = 1.0
    PLOT_SETTINGS["scatter_alpha_normal"] = 0.5
    apply_plot_settings()

    create_combined_plot(data_points, output_paths["combined_plot"], args.per_class_pareto)

    logger.info("\n### STEP 6: Generating Ranking CSV ###")
    generate_ranking_csv(data_points, output_paths["ranking_csv"])

    logger.info("\n### STEP 7: Writing Metadata ###")
    write_metadata(experiments, args.per_class_pareto, output_paths)

    logger.info("\n" + "="*80)
    logger.info(f"Analysis Complete! Outputs: {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
