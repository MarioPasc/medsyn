#!/usr/bin/env python3
"""
Generate comprehensive training evolution visualizations for ccDDPM.

Creates:
- Multi-page PDF with all figures
- Individual PNG files for each visualization
- Summary statistics in JSON and text format

Usage:
    python -m medsyn.utils.visualize_training_evolution /path/to/output_dir \
        [--use-scienceplots-latex] [--dpi 300]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# PathMNIST class names (can be overridden)
PATHMNIST_CLASS_NAMES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth Muscle",
    "Normal Colon Mucosa",
    "Cancer-Assoc. Stroma",
    "Colorectal Adenocarc.",
]


def setup_style(use_latex: bool = False) -> None:
    """Configure matplotlib style for publication-quality figures."""
    if use_latex:
        try:
            import scienceplots
            plt.style.use(["science", "ieee"])
        except ImportError:
            print("Warning: scienceplots not installed. Using default style.")
            plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    # Common settings
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def get_viridis_colors(n: int) -> List[str]:
    """Get n colors from viridis colormap."""
    cmap = plt.cm.viridis
    return [mcolors.rgb2hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def get_split_colors() -> Dict[str, str]:
    """Get colors for train/val/test splits using viridis-derived palette."""
    return {
        "train": "#440154",  # Dark purple (viridis start)
        "val": "#21918c",    # Teal (viridis middle)
        "test": "#fde725",   # Yellow (viridis end)
    }


def load_data(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and diagnostics CSVs."""
    training_csv = output_dir / "training_metrics.csv"
    diag_csv = output_dir / "diagnostics_metrics.csv"

    if not training_csv.exists():
        raise FileNotFoundError(f"training_metrics.csv not found in {output_dir}")

    training_df = pd.read_csv(training_csv)

    if diag_csv.exists():
        diag_df = pd.read_csv(diag_csv)
    else:
        # Fallback: extract diag rows from combined CSV (old format)
        if "split" in training_df.columns and "diag" in training_df["split"].values:
            diag_df = training_df[training_df["split"] == "diag"].copy()
            training_df = training_df[training_df["split"] != "diag"].copy()
        else:
            diag_df = pd.DataFrame()

    return training_df, diag_df


def detect_class_columns(df: pd.DataFrame, prefix: str = "loss_raw_c") -> List[str]:
    """
    Detect per-class columns with given prefix.

    Args:
        df: DataFrame to search
        prefix: Column prefix to look for (e.g., 'loss_raw_c', 'loss_weighted_c', 'fid_c')

    Returns:
        Sorted list of matching column names
    """
    # Filter columns that match the prefix and have a digit after it
    cols = []
    for c in df.columns:
        if c.startswith(prefix):
            suffix = c[len(prefix):]
            if suffix.isdigit():
                cols.append(c)
    return sorted(cols, key=lambda x: int(x[len(prefix):]))


def get_class_names(n_classes: int) -> List[str]:
    """Get class names, using PathMNIST defaults if matching."""
    if n_classes == len(PATHMNIST_CLASS_NAMES):
        return PATHMNIST_CLASS_NAMES
    return [f"Class {i}" for i in range(n_classes)]


def plot_loss_curves(training_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot train/val/test loss curves with CI bands."""
    colors = get_split_colors()

    for split in ["train", "val", "test"]:
        split_data = training_df[training_df["split"] == split].sort_values("epoch")
        if split_data.empty:
            continue

        epochs = split_data["epoch"].values
        loss = split_data["loss"].values
        loss_std = split_data.get("loss_std", pd.Series([0] * len(split_data))).fillna(0).values

        # 95% CI approximation (assuming ~normal distribution)
        ci_95 = 1.96 * loss_std

        ax.plot(epochs, loss, color=colors[split], label=split.capitalize(), linewidth=2)
        ax.fill_between(epochs, loss - ci_95, loss + ci_95, color=colors[split], alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Evolution", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)

    # Add learning rate on secondary axis
    train_data = training_df[training_df["split"] == "train"].sort_values("epoch")
    if "lr" in train_data.columns and not train_data["lr"].isna().all():
        ax2 = ax.twinx()
        ax2.plot(train_data["epoch"], train_data["lr"], color="gray", linestyle="--",
                 alpha=0.5, linewidth=1, label="LR")
        ax2.set_ylabel("Learning Rate", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        ax2.set_yscale("log")


def plot_quality_metrics(training_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot PSNR and SSIM evolution with dual y-axis."""
    colors = get_split_colors()

    # PSNR on left axis
    for split in ["train", "val", "test"]:
        split_data = training_df[training_df["split"] == split].sort_values("epoch")
        if split_data.empty or "psnr" not in split_data.columns:
            continue

        epochs = split_data["epoch"].values
        psnr = split_data["psnr"].values

        ax.plot(epochs, psnr, color=colors[split], label=f"{split.capitalize()} PSNR",
                linewidth=2, linestyle="-")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Reconstruction Quality Metrics", fontweight="bold")

    # SSIM on right axis
    ax2 = ax.twinx()
    for split in ["train", "val", "test"]:
        split_data = training_df[training_df["split"] == split].sort_values("epoch")
        if split_data.empty or "ssim" not in split_data.columns:
            continue

        epochs = split_data["epoch"].values
        ssim = split_data["ssim"].values

        ax2.plot(epochs, ssim, color=colors[split], label=f"{split.capitalize()} SSIM",
                 linewidth=2, linestyle="--")

    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0, 1)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9, ncol=2)


def get_split_linestyles() -> Dict[str, str]:
    """Get linestyles for train/val/test splits."""
    return {
        "train": "-",      # solid
        "val": "--",       # dashed
        "test": ":",       # dotted
    }


def plot_perclass_loss_by_type(
    training_df: pd.DataFrame,
    ax: plt.Axes,
    loss_type: str = "raw",
    title: Optional[str] = None
) -> None:
    """
    Plot per-class loss evolution with one color per class, different linestyles per split.

    Args:
        training_df: DataFrame with training metrics
        ax: Matplotlib axes to plot on
        loss_type: 'raw' for loss_raw_c* or 'weighted' for loss_weighted_c*
        title: Optional custom title
    """
    prefix = f"loss_{loss_type}_c"
    class_cols = detect_class_columns(training_df, prefix=prefix)

    if not class_cols:
        ax.text(0.5, 0.5, f"No {loss_type} per-class data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    n_classes = len(class_cols)
    class_names = get_class_names(n_classes)
    colors = get_viridis_colors(n_classes)
    linestyles = get_split_linestyles()

    # Plot for each split and class
    for split in ["train", "val", "test"]:
        split_data = training_df[training_df["split"] == split].sort_values("epoch")
        if split_data.empty:
            continue

        epochs = split_data["epoch"].values

        for i, (col, name) in enumerate(zip(class_cols, class_names)):
            if col in split_data.columns:
                loss = split_data[col].values
                # Only add label for first split to avoid duplicate legend entries
                label = f"{name}" if split == "train" else None
                ax.plot(epochs, loss, color=colors[i], linestyle=linestyles[split],
                       linewidth=1.5, label=label, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    default_title = f"Per-Class {loss_type.capitalize()} Loss Evolution"
    ax.set_title(title if title else default_title, fontweight="bold")

    # Create custom legend with both class colors and split linestyles
    # First, add class legend
    handles1 = [plt.Line2D([0], [0], color=colors[i], linewidth=2, label=name)
                for i, name in enumerate(class_names)]

    # Then add split legend
    handles2 = [plt.Line2D([0], [0], color='gray', linestyle=ls, linewidth=2, label=split.capitalize())
                for split, ls in linestyles.items()]

    # Combine legends
    all_handles = handles1 + handles2
    ax.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=min(6, len(all_handles)), fontsize=8, framealpha=0.9)


def plot_perclass_loss_comparative(training_df: pd.DataFrame, axes: List[plt.Axes]) -> None:
    """
    Plot comparative visualization: raw vs weighted loss.

    Creates 3 subplots (train/val/test) with:
    - Different colors for classes
    - Different linestyles for raw vs weighted

    Args:
        training_df: DataFrame with training metrics
        axes: List of 3 axes [train_ax, val_ax, test_ax]
    """
    raw_cols = detect_class_columns(training_df, prefix="loss_raw_c")
    weighted_cols = detect_class_columns(training_df, prefix="loss_weighted_c")

    if not raw_cols and not weighted_cols:
        for ax in axes:
            ax.text(0.5, 0.5, "No per-class loss data available", ha="center", va="center",
                    transform=ax.transAxes)
        return

    n_classes = len(raw_cols) if raw_cols else len(weighted_cols)
    class_names = get_class_names(n_classes)
    colors = get_viridis_colors(n_classes)

    split_names = ["train", "val", "test"]
    loss_linestyles = {
        "raw": "-",       # solid
        "weighted": "--"  # dashed
    }

    for ax_idx, (ax, split) in enumerate(zip(axes, split_names)):
        split_data = training_df[training_df["split"] == split].sort_values("epoch")

        if split_data.empty:
            ax.text(0.5, 0.5, f"No {split} data available", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        epochs = split_data["epoch"].values

        # Plot raw loss
        for i, (col, name) in enumerate(zip(raw_cols, class_names)):
            if col in split_data.columns:
                loss = split_data[col].dropna().values
                if len(loss) > 0:
                    ep = epochs[:len(loss)]
                    ax.plot(ep, loss, color=colors[i], linestyle=loss_linestyles["raw"],
                           linewidth=1.5, alpha=0.8)

        # Plot weighted loss
        for i, (col, name) in enumerate(zip(weighted_cols, class_names)):
            if col in split_data.columns:
                loss = split_data[col].dropna().values
                if len(loss) > 0:
                    ep = epochs[:len(loss)]
                    ax.plot(ep, loss, color=colors[i], linestyle=loss_linestyles["weighted"],
                           linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{split.capitalize()} Split", fontweight="bold")

    # Add legend to last subplot only
    handles_class = [plt.Line2D([0], [0], color=colors[i], linewidth=2, label=name)
                     for i, name in enumerate(class_names)]
    handles_type = [plt.Line2D([0], [0], color='gray', linestyle=ls, linewidth=2, label=lt.capitalize())
                    for lt, ls in loss_linestyles.items()]

    axes[-1].legend(handles=handles_class + handles_type, loc='upper center',
                    bbox_to_anchor=(0.5, -0.15), ncol=min(6, n_classes + 2),
                    fontsize=7, framealpha=0.9)


def plot_fid_evolution(training_df: pd.DataFrame, axes: List[plt.Axes]) -> None:
    """
    Plot FID evolution with heatmap + line.

    Args:
        training_df: DataFrame with training metrics
        axes: List of 2 axes [line_ax, heatmap_ax]
    """
    line_ax, heatmap_ax = axes

    fid_cols = detect_class_columns(training_df, prefix="fid_c")
    has_fid_global = "fid_global" in training_df.columns

    if not fid_cols and not has_fid_global:
        for ax in axes:
            ax.text(0.5, 0.5, "No FID data available", ha="center", va="center",
                    transform=ax.transAxes)
        return

    split_colors = get_split_colors()
    n_classes = len(fid_cols) if fid_cols else 0
    class_names = get_class_names(n_classes) if n_classes > 0 else []

    # Top panel: Global FID line plot for val and test
    if has_fid_global:
        for split in ["val", "test"]:
            split_data = training_df[training_df["split"] == split].sort_values("epoch")
            if split_data.empty:
                continue

            epochs = split_data["epoch"].values
            fid_global = split_data["fid_global"].values

            # Filter out NaN values
            mask = ~np.isnan(fid_global)
            if mask.sum() > 0:
                line_ax.plot(epochs[mask], fid_global[mask], color=split_colors[split],
                            label=f"{split.capitalize()} FID", linewidth=2, marker='o', markersize=4)

        line_ax.set_xlabel("Epoch")
        line_ax.set_ylabel("FID (lower is better)")
        line_ax.set_title("Global FID Evolution", fontweight="bold")
        line_ax.legend(loc="upper right", framealpha=0.9)
        line_ax.grid(True, alpha=0.3)
    else:
        line_ax.text(0.5, 0.5, "No global FID data", ha="center", va="center",
                    transform=line_ax.transAxes)

    # Bottom panel: Per-class FID heatmap (using validation data)
    if fid_cols:
        val_data = training_df[training_df["split"] == "val"].sort_values("epoch")
        if val_data.empty:
            val_data = training_df[training_df["split"] == "test"].sort_values("epoch")

        if not val_data.empty:
            epochs = val_data["epoch"].values
            heatmap_data = val_data[fid_cols].values.T  # Classes x Epochs

            # Handle NaN values for visualization
            heatmap_data_masked = np.ma.masked_invalid(heatmap_data)

            im = heatmap_ax.imshow(heatmap_data_masked, aspect="auto", cmap="viridis_r",
                                   origin="upper")

            heatmap_ax.set_yticks(range(n_classes))
            heatmap_ax.set_yticklabels(class_names, fontsize=8)

            # Show every 5th epoch on x-axis
            n_epochs = len(epochs)
            tick_step = max(1, n_epochs // 10)
            heatmap_ax.set_xticks(range(0, n_epochs, tick_step))
            heatmap_ax.set_xticklabels(epochs[::tick_step])

            heatmap_ax.set_xlabel("Epoch")
            heatmap_ax.set_ylabel("Class")
            heatmap_ax.set_title("Per-Class FID Heatmap (Validation)", fontweight="bold")

            # Colorbar
            cbar = plt.colorbar(im, ax=heatmap_ax, shrink=0.8)
            cbar.set_label("FID (lower is better)")
        else:
            heatmap_ax.text(0.5, 0.5, "No per-class FID data", ha="center", va="center",
                           transform=heatmap_ax.transAxes)
    else:
        heatmap_ax.text(0.5, 0.5, "No per-class FID data", ha="center", va="center",
                       transform=heatmap_ax.transAxes)


def plot_perclass_loss_lines(training_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot per-class loss evolution with individual lines (legacy function)."""
    # Use the new function with raw loss
    plot_perclass_loss_by_type(training_df, ax, loss_type="raw",
                               title="Per-Class Raw Loss Evolution")


def plot_perclass_heatmap(
    training_df: pd.DataFrame,
    ax: plt.Axes,
    loss_type: str = "raw"
) -> None:
    """
    Plot epochs x classes loss heatmap.

    Args:
        training_df: DataFrame with training metrics
        ax: Matplotlib axes
        loss_type: 'raw' or 'weighted'
    """
    prefix = f"loss_{loss_type}_c"
    class_cols = detect_class_columns(training_df, prefix=prefix)

    if not class_cols:
        ax.text(0.5, 0.5, f"No per-class {loss_type} data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    n_classes = len(class_cols)
    class_names = get_class_names(n_classes)

    # Use validation data
    val_data = training_df[training_df["split"] == "val"].sort_values("epoch")
    if val_data.empty:
        val_data = training_df[training_df["split"] == "train"].sort_values("epoch")

    if val_data.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    epochs = val_data["epoch"].values
    heatmap_data = val_data[class_cols].values.T  # Classes x Epochs

    # Handle NaN values
    heatmap_data_masked = np.ma.masked_invalid(heatmap_data)

    im = ax.imshow(heatmap_data_masked, aspect="auto", cmap="viridis", origin="upper")

    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=8)

    # Show every 5th epoch on x-axis
    n_epochs = len(epochs)
    tick_step = max(1, n_epochs // 10)
    ax.set_xticks(range(0, n_epochs, tick_step))
    ax.set_xticklabels(epochs[::tick_step])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Class")
    ax.set_title(f"Per-Class {loss_type.capitalize()} Loss Heatmap (Validation)", fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Loss")


def plot_elbo_diagnostics(diag_df: pd.DataFrame, axes: List[plt.Axes]) -> None:
    """Plot ELBO diagnostics in 3 panels."""
    if diag_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No diagnostic data available", ha="center", va="center",
                    transform=ax.transAxes)
        return

    diag_df = diag_df.sort_values("epoch")
    epochs = diag_df["epoch"].values

    # Panel 1: L_simple vs L_weighted
    ax1 = axes[0]
    if "elbo_L_simple_mean" in diag_df.columns:
        ax1.plot(epochs, diag_df["elbo_L_simple_mean"], color="#440154",
                 label="L_simple (raw)", linewidth=2)
    if "elbo_L_weighted_mean" in diag_df.columns:
        ax1.plot(epochs, diag_df["elbo_L_weighted_mean"], color="#fde725",
                 label="L_weighted (KL)", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("ELBO Components", fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_yscale("log")

    # Panel 2: SNR by timestep region
    ax2 = axes[1]
    region_colors = {"low_t": "#440154", "mid_t": "#21918c", "high_t": "#fde725"}
    region_labels = {"low_t": "Low t (<333)", "mid_t": "Mid t (333-666)", "high_t": "High t (>666)"}

    for region, color in region_colors.items():
        col = f"elbo_snr_{region}"
        if col in diag_df.columns:
            ax2.plot(epochs, diag_df[col], color=color, label=region_labels[region], linewidth=2)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SNR")
    ax2.set_title("SNR by Timestep Region", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)

    # Panel 3: Weight ratios
    ax3 = axes[2]
    for region, color in region_colors.items():
        col = f"elbo_weight_ratio_{region}"
        if col in diag_df.columns:
            ax3.plot(epochs, diag_df[col], color=color, label=region_labels[region], linewidth=2)

    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Weight Ratio (L_weighted/L_simple)")
    ax3.set_title("Min-SNR Weighting Effect", fontweight="bold")
    ax3.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)


def plot_reconstruction_quality(diag_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot single-step and full-chain reconstruction metrics."""
    if diag_df.empty:
        ax.text(0.5, 0.5, "No diagnostic data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    diag_df = diag_df.sort_values("epoch")
    epochs = diag_df["epoch"].values

    # PSNR metrics
    metrics = [
        ("recon_psnr_t100", "#440154", "PSNR @ t=100", "-"),
        ("recon_psnr_t500", "#21918c", "PSNR @ t=500", "-"),
        ("full_chain_psnr", "#fde725", "Full Chain PSNR", "--"),
    ]

    for col, color, label, ls in metrics:
        if col in diag_df.columns:
            ax.plot(epochs, diag_df[col], color=color, label=label, linewidth=2, linestyle=ls)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Reconstruction Quality at Different Timesteps", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9)


def plot_noise_prediction_health(diag_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot noise prediction correlation and std metrics."""
    if diag_df.empty:
        ax.text(0.5, 0.5, "No diagnostic data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    diag_df = diag_df.sort_values("epoch")
    epochs = diag_df["epoch"].values

    # Correlation metrics on primary axis
    corr_metrics = [
        ("noise_pred_corr", "#440154", "Mean Correlation"),
        ("noise_pred_corr_t100", "#21918c", "Corr @ t=100"),
        ("noise_pred_corr_t500", "#fde725", "Corr @ t=500"),
    ]

    for col, color, label in corr_metrics:
        if col in diag_df.columns:
            ax.plot(epochs, diag_df[col], color=color, label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Correlation")
    ax.set_title("Noise Prediction Health Metrics", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold (0.5)")

    # Prediction std on secondary axis
    ax2 = ax.twinx()
    if "pred_std" in diag_df.columns:
        ax2.plot(epochs, diag_df["pred_std"], color="red", linestyle=":", linewidth=2,
                 label="Pred Std")
        ax2.set_ylabel("Prediction Std", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)


def compute_summary_stats(training_df: pd.DataFrame, diag_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for the training run."""
    stats = {
        "training": {},
        "best_epoch": {},
        "final_epoch": {},
        "convergence": {},
    }

    # Get final epoch data
    final_epoch = training_df["epoch"].max()
    stats["training"]["total_epochs"] = int(final_epoch)
    stats["training"]["splits"] = list(training_df["split"].unique())

    # Per-split final metrics
    for split in ["train", "val", "test"]:
        split_data = training_df[training_df["split"] == split]
        if split_data.empty:
            continue

        final_data = split_data[split_data["epoch"] == final_epoch].iloc[0]
        stats["final_epoch"][split] = {
            "loss": float(final_data.get("loss", np.nan)),
            "psnr": float(final_data.get("psnr", np.nan)),
            "ssim": float(final_data.get("ssim", np.nan)),
        }

    # Best epoch (based on validation loss)
    val_data = training_df[training_df["split"] == "val"]
    if not val_data.empty:
        best_idx = val_data["loss"].idxmin()
        best_row = val_data.loc[best_idx]
        stats["best_epoch"] = {
            "epoch": int(best_row["epoch"]),
            "val_loss": float(best_row["loss"]),
            "val_psnr": float(best_row.get("psnr", np.nan)),
            "val_ssim": float(best_row.get("ssim", np.nan)),
        }

    # Convergence info
    if not val_data.empty:
        losses = val_data.sort_values("epoch")["loss"].values
        if len(losses) > 5:
            # Check if loss is still decreasing in last 5 epochs
            recent_trend = np.polyfit(range(5), losses[-5:], 1)[0]
            stats["convergence"]["recent_trend"] = float(recent_trend)
            stats["convergence"]["converged"] = recent_trend > -1e-5  # Nearly flat
            stats["convergence"]["final_loss_std"] = float(np.std(losses[-5:]))

    # Diagnostic stats
    if not diag_df.empty:
        final_diag = diag_df[diag_df["epoch"] == final_epoch]
        if not final_diag.empty:
            final_diag = final_diag.iloc[0]
            stats["diagnostics"] = {
                "noise_pred_corr": float(final_diag.get("noise_pred_corr", np.nan)),
                "pred_std": float(final_diag.get("pred_std", np.nan)),
                "full_chain_psnr": float(final_diag.get("full_chain_psnr", np.nan)),
                "full_chain_ssim": float(final_diag.get("full_chain_ssim", np.nan)),
            }

    return stats


def generate_text_report(stats: Dict[str, Any]) -> str:
    """Generate human-readable summary report."""
    lines = [
        "=" * 70,
        "CCDDPM TRAINING SUMMARY REPORT",
        "=" * 70,
        "",
        "TRAINING OVERVIEW",
        "-" * 40,
        f"Total Epochs: {stats['training'].get('total_epochs', 'N/A')}",
        f"Splits: {', '.join(stats['training'].get('splits', []))}",
        "",
    ]

    if stats.get("best_epoch"):
        lines.extend([
            "BEST EPOCH (by validation loss)",
            "-" * 40,
            f"Epoch: {stats['best_epoch'].get('epoch', 'N/A')}",
            f"Validation Loss: {stats['best_epoch'].get('val_loss', 'N/A'):.6f}",
            f"Validation PSNR: {stats['best_epoch'].get('val_psnr', 'N/A'):.2f} dB",
            f"Validation SSIM: {stats['best_epoch'].get('val_ssim', 'N/A'):.4f}",
            "",
        ])

    lines.append("FINAL EPOCH METRICS")
    lines.append("-" * 40)
    for split, metrics in stats.get("final_epoch", {}).items():
        lines.append(f"{split.upper()}:")
        lines.append(f"  Loss: {metrics.get('loss', 'N/A'):.6f}")
        lines.append(f"  PSNR: {metrics.get('psnr', 'N/A'):.2f} dB")
        lines.append(f"  SSIM: {metrics.get('ssim', 'N/A'):.4f}")
    lines.append("")

    if stats.get("convergence"):
        lines.extend([
            "CONVERGENCE ANALYSIS",
            "-" * 40,
            f"Recent Trend: {stats['convergence'].get('recent_trend', 'N/A'):.2e}",
            f"Converged: {'Yes' if stats['convergence'].get('converged') else 'No'}",
            f"Final Loss Std (last 5): {stats['convergence'].get('final_loss_std', 'N/A'):.6f}",
            "",
        ])

    if stats.get("diagnostics"):
        lines.extend([
            "DIAGNOSTIC METRICS (Final Epoch)",
            "-" * 40,
            f"Noise Pred Correlation: {stats['diagnostics'].get('noise_pred_corr', 'N/A'):.4f}",
            f"Prediction Std: {stats['diagnostics'].get('pred_std', 'N/A'):.4f}",
            f"Full Chain PSNR: {stats['diagnostics'].get('full_chain_psnr', 'N/A'):.2f} dB",
            f"Full Chain SSIM: {stats['diagnostics'].get('full_chain_ssim', 'N/A'):.4f}",
            "",
        ])

    lines.append("=" * 70)
    return "\n".join(lines)


def generate_training_visualizations(
    output_dir: Path | str,
    use_latex: bool = False,
    dpi: int = 300,
) -> Path:
    """
    Generate all training visualizations and save to output_dir/vis/.

    Args:
        output_dir: Directory containing training_metrics.csv and diagnostics_metrics.csv
        use_latex: Use scienceplots with LaTeX rendering
        dpi: DPI for saved figures

    Returns:
        Path to the vis/ directory containing all outputs
    """
    output_dir = Path(output_dir)
    vis_dir = output_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Setup style
    setup_style(use_latex)

    # Load data
    print(f"Loading data from {output_dir}...")
    training_df, diag_df = load_data(output_dir)

    print(f"  Training rows: {len(training_df)}")
    print(f"  Diagnostic rows: {len(diag_df)}")

    # Compute summary stats
    stats = compute_summary_stats(training_df, diag_df)

    # Save summary stats
    with open(vis_dir / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved: summary_stats.json")

    report = generate_text_report(stats)
    with open(vis_dir / "summary_report.txt", "w") as f:
        f.write(report)
    print(f"  Saved: summary_report.txt")

    # Generate figures
    figures = []
    fig_num = 1

    # Figure 1: Loss Curves
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_loss_curves(training_df, ax1)
    fig1.tight_layout()
    fig1.savefig(vis_dir / f"{fig_num:02d}_loss_curves.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Loss Curves", fig1))
    print(f"  Saved: {fig_num:02d}_loss_curves.png")
    fig_num += 1

    # Figure 2: Quality Metrics
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_quality_metrics(training_df, ax2)
    fig2.tight_layout()
    fig2.savefig(vis_dir / f"{fig_num:02d}_quality_metrics.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Quality Metrics", fig2))
    print(f"  Saved: {fig_num:02d}_quality_metrics.png")
    fig_num += 1

    # Figure 3: Per-Class Raw Loss (NEW)
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    plot_perclass_loss_by_type(training_df, ax3, loss_type="raw",
                               title="Per-Class Raw Loss Evolution")
    fig3.tight_layout()
    fig3.subplots_adjust(bottom=0.2)  # Make room for legend
    fig3.savefig(vis_dir / f"{fig_num:02d}_perclass_raw_loss.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Per-Class Raw Loss", fig3))
    print(f"  Saved: {fig_num:02d}_perclass_raw_loss.png")
    fig_num += 1

    # Figure 4: Per-Class Weighted Loss (NEW)
    fig4, ax4 = plt.subplots(figsize=(14, 8))
    plot_perclass_loss_by_type(training_df, ax4, loss_type="weighted",
                               title="Per-Class Weighted Loss Evolution")
    fig4.tight_layout()
    fig4.subplots_adjust(bottom=0.2)  # Make room for legend
    fig4.savefig(vis_dir / f"{fig_num:02d}_perclass_weighted_loss.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Per-Class Weighted Loss", fig4))
    print(f"  Saved: {fig_num:02d}_perclass_weighted_loss.png")
    fig_num += 1

    # Figure 5: Comparative Plot - Raw vs Weighted Loss (NEW)
    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))
    plot_perclass_loss_comparative(training_df, list(axes5))
    fig5.suptitle("Per-Class Loss: Raw vs Weighted Comparison", fontweight="bold", y=1.02)
    fig5.tight_layout()
    fig5.subplots_adjust(bottom=0.18)  # Make room for legend
    fig5.savefig(vis_dir / f"{fig_num:02d}_perclass_loss_comparative.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Per-Class Loss Comparison", fig5))
    print(f"  Saved: {fig_num:02d}_perclass_loss_comparative.png")
    fig_num += 1

    # Figure 6: Per-Class Raw Loss Heatmap
    fig6, ax6 = plt.subplots(figsize=(14, 6))
    plot_perclass_heatmap(training_df, ax6, loss_type="raw")
    fig6.tight_layout()
    fig6.savefig(vis_dir / f"{fig_num:02d}_perclass_loss_heatmap.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Per-Class Loss Heatmap", fig6))
    print(f"  Saved: {fig_num:02d}_perclass_loss_heatmap.png")
    fig_num += 1

    # Figure 7: FID Evolution (NEW - heatmap + line)
    fig7, axes7 = plt.subplots(2, 1, figsize=(12, 10))
    plot_fid_evolution(training_df, list(axes7))
    fig7.tight_layout()
    fig7.savefig(vis_dir / f"{fig_num:02d}_fid_evolution.png", dpi=dpi, bbox_inches="tight")
    figures.append(("FID Evolution", fig7))
    print(f"  Saved: {fig_num:02d}_fid_evolution.png")
    fig_num += 1

    # Figure 8: ELBO Diagnostics (3 panels)
    fig8, axes8 = plt.subplots(1, 3, figsize=(15, 5))
    plot_elbo_diagnostics(diag_df, list(axes8))
    fig8.tight_layout()
    fig8.savefig(vis_dir / f"{fig_num:02d}_elbo_diagnostics.png", dpi=dpi, bbox_inches="tight")
    figures.append(("ELBO Diagnostics", fig8))
    print(f"  Saved: {fig_num:02d}_elbo_diagnostics.png")
    fig_num += 1

    # Figure 9: Reconstruction Quality
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    plot_reconstruction_quality(diag_df, ax9)
    fig9.tight_layout()
    fig9.savefig(vis_dir / f"{fig_num:02d}_reconstruction_quality.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Reconstruction Quality", fig9))
    print(f"  Saved: {fig_num:02d}_reconstruction_quality.png")
    fig_num += 1

    # Figure 10: Noise Prediction Health
    fig10, ax10 = plt.subplots(figsize=(10, 6))
    plot_noise_prediction_health(diag_df, ax10)
    fig10.tight_layout()
    fig10.savefig(vis_dir / f"{fig_num:02d}_noise_prediction_health.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Noise Prediction Health", fig10))
    print(f"  Saved: {fig_num:02d}_noise_prediction_health.png")

    # Generate multi-page PDF
    pdf_path = vis_dir / "training_dashboard.pdf"
    with PdfPages(pdf_path) as pdf:
        for title, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")

    print(f"  Saved: training_dashboard.pdf")

    # Close all figures
    for _, fig in figures:
        plt.close(fig)

    print(f"\nAll visualizations saved to: {vis_dir}")
    return vis_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate training evolution visualizations for ccDDPM"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory containing training_metrics.csv and diagnostics_metrics.csv"
    )
    parser.add_argument(
        "--use-scienceplots-latex",
        action="store_true",
        help="Use scienceplots with LaTeX rendering (requires scienceplots package)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures (default: 300)"
    )

    args = parser.parse_args()

    generate_training_visualizations(
        args.output_dir,
        use_latex=args.use_scienceplots_latex,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
