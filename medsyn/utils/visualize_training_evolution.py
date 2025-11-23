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


def detect_class_columns(df: pd.DataFrame) -> List[str]:
    """Detect loss_c* columns."""
    return sorted([c for c in df.columns if c.startswith("loss_c")])


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


def plot_perclass_loss_lines(training_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot per-class loss evolution with individual lines."""
    class_cols = detect_class_columns(training_df)
    if not class_cols:
        ax.text(0.5, 0.5, "No per-class data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    n_classes = len(class_cols)
    class_names = get_class_names(n_classes)
    colors = get_viridis_colors(n_classes)

    # Use validation data for cleaner visualization
    val_data = training_df[training_df["split"] == "val"].sort_values("epoch")
    if val_data.empty:
        val_data = training_df[training_df["split"] == "train"].sort_values("epoch")

    epochs = val_data["epoch"].values

    for i, (col, name) in enumerate(zip(class_cols, class_names)):
        if col in val_data.columns:
            loss = val_data[col].values
            ax.plot(epochs, loss, color=colors[i], label=name, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Per-Class Loss Evolution (Validation)", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8, ncol=2)


def plot_perclass_heatmap(training_df: pd.DataFrame, ax: plt.Axes) -> None:
    """Plot epochs x classes loss heatmap."""
    class_cols = detect_class_columns(training_df)
    if not class_cols:
        ax.text(0.5, 0.5, "No per-class data available", ha="center", va="center",
                transform=ax.transAxes)
        return

    n_classes = len(class_cols)
    class_names = get_class_names(n_classes)

    # Use validation data
    val_data = training_df[training_df["split"] == "val"].sort_values("epoch")
    if val_data.empty:
        val_data = training_df[training_df["split"] == "train"].sort_values("epoch")

    epochs = val_data["epoch"].values
    heatmap_data = val_data[class_cols].values.T  # Classes x Epochs

    im = ax.imshow(heatmap_data, aspect="auto", cmap="viridis", origin="upper")

    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=8)

    # Show every 5th epoch on x-axis
    n_epochs = len(epochs)
    tick_step = max(1, n_epochs // 10)
    ax.set_xticks(range(0, n_epochs, tick_step))
    ax.set_xticklabels(epochs[::tick_step])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Class")
    ax.set_title("Per-Class Loss Heatmap (Validation)", fontweight="bold")

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

    # Figure 1: Loss Curves
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_loss_curves(training_df, ax1)
    fig1.tight_layout()
    fig1.savefig(vis_dir / "01_loss_curves.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Loss Curves", fig1))
    print(f"  Saved: 01_loss_curves.png")

    # Figure 2: Quality Metrics
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_quality_metrics(training_df, ax2)
    fig2.tight_layout()
    fig2.savefig(vis_dir / "02_quality_metrics.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Quality Metrics", fig2))
    print(f"  Saved: 02_quality_metrics.png")

    # Figure 3: Per-Class Loss Lines
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plot_perclass_loss_lines(training_df, ax3)
    fig3.tight_layout()
    fig3.savefig(vis_dir / "03_perclass_loss_lines.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Per-Class Loss (Lines)", fig3))
    print(f"  Saved: 03_perclass_loss_lines.png")

    # Figure 4: Per-Class Heatmap
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    plot_perclass_heatmap(training_df, ax4)
    fig4.tight_layout()
    fig4.savefig(vis_dir / "04_perclass_loss_heatmap.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Per-Class Loss (Heatmap)", fig4))
    print(f"  Saved: 04_perclass_loss_heatmap.png")

    # Figure 5: ELBO Diagnostics (3 panels)
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 5))
    plot_elbo_diagnostics(diag_df, list(axes5))
    fig5.tight_layout()
    fig5.savefig(vis_dir / "05_elbo_diagnostics.png", dpi=dpi, bbox_inches="tight")
    figures.append(("ELBO Diagnostics", fig5))
    print(f"  Saved: 05_elbo_diagnostics.png")

    # Figure 6: Reconstruction Quality
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    plot_reconstruction_quality(diag_df, ax6)
    fig6.tight_layout()
    fig6.savefig(vis_dir / "06_reconstruction_quality.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Reconstruction Quality", fig6))
    print(f"  Saved: 06_reconstruction_quality.png")

    # Figure 7: Noise Prediction Health
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    plot_noise_prediction_health(diag_df, ax7)
    fig7.tight_layout()
    fig7.savefig(vis_dir / "07_noise_prediction_health.png", dpi=dpi, bbox_inches="tight")
    figures.append(("Noise Prediction Health", fig7))
    print(f"  Saved: 07_noise_prediction_health.png")

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
