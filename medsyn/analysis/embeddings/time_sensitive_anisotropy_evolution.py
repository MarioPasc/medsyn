#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time-sensitive anisotropy evolution visualization for ccDDPM latent spaces.

This script creates multi-panel visualizations showing how local anisotropy
evolves across training epochs and diffusion timesteps for different branches
(unconditional, conditional, both).

For each diffusion timestep T (e.g., 100, 500, 900):
  - Creates a grid of E columns x 3 rows
  - E = number of epochs to plot (based on max_epoch / num_epochs_to_plot)
  - 3 rows = uncond, cond, both branches
  - Each subplot shows the anisotropy visualization for that epoch/branch/timestep
  - All subplots share a common legend placed at the bottom center

Usage example:
  python time_sensitive_anisotropy_evolution.py \
      --features_path /path/to/embeddings \
      --output_path /path/to/output \
      --feature_layer unet.mid_block \
      --max_epoch 100 \
      --num_epochs_to_plot 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde

# Import functions from the single visualization script
from medsyn.analysis.embeddings.anisotropy_single_visualization import (
    compute_local_anisotropy,
)
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_available_epochs(features_path: Path) -> List[int]:
    """
    Scan the features_path directory for probe_features_epoch_XXXX.npz files
    and extract the list of available epochs.

    Parameters
    ----------
    features_path : Path
        Directory containing probe_features_epoch_*.npz files.

    Returns
    -------
    epochs : List[int]
        Sorted list of available epoch numbers.
    """
    npz_files = list(features_path.glob("probe_features_epoch_*.npz"))
    epochs = []
    for f in npz_files:
        try:
            # Extract epoch number from filename: probe_features_epoch_0031.npz -> 31
            epoch_str = f.stem.split("_")[-1]
            epochs.append(int(epoch_str))
        except (ValueError, IndexError):
            logging.warning(f"Could not parse epoch from filename: {f.name}")
            continue
    return sorted(epochs)


def get_available_timesteps(npz_path: Path) -> List[int]:
    """
    Load an NPZ file and extract the unique timesteps present.

    Parameters
    ----------
    npz_path : Path
        Path to a probe_features_epoch_*.npz file.

    Returns
    -------
    timesteps : List[int]
        Sorted list of unique timesteps.
    """
    data = np.load(npz_path, allow_pickle=True)
    timesteps = data["timesteps"]
    return sorted(np.unique(timesteps).tolist())


def load_probe_features_2d_with_variance(
    npz_path: str,
    feature_layer: str,
    feature_timestep: Optional[int],
    feature_branch: str = "cond",
    random_state: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load probe feature vectors from NPZ and reduce to 2D with PCA.
    Returns PCA variance explained ratios.

    Parameters
    ----------
    npz_path : str
        Path to probe_features_epoch_XXXX.npz.
    feature_layer : str
        Layer name to select.
    feature_timestep : int or None
        Timestep to select. If None, use median timestep.
    feature_branch : {"cond", "uncond", "both"}, default "cond"
        Which CFG branch to include.
    random_state : int
        Random seed for PCA reproducibility.
    max_samples : int or None
        Optional maximum number of feature vectors to subsample.

    Returns
    -------
    Z : ndarray of shape (N, 2)
        2D PCA projection of selected features.
    class_ids : ndarray of shape (N,)
        Class IDs corresponding to each feature vector.
    variance_ratio : ndarray of shape (2,)
        PCA explained variance ratio for each component.
    """
    data = np.load(npz_path, allow_pickle=True)

    features = data["features"]
    class_ids = data["class_ids"]
    timesteps = data["timesteps"]
    layer_names = data["layer_names"]
    branches = data["branches"]
    feature_lengths = data.get("feature_lengths", None)

    # Ensure string comparison works
    layer_names = layer_names.astype(str)
    branches = branches.astype(str)

    # Decide timestep if not provided
    unique_t = np.unique(timesteps)
    if feature_timestep is None:
        feature_timestep = int(np.median(unique_t))

    # Build mask
    mask = (layer_names == feature_layer) & (timesteps == feature_timestep)
    if feature_branch.lower() in {"cond", "uncond"}:
        mask &= (branches == feature_branch.lower())

    if not np.any(mask):
        raise ValueError(
            f"No features found in {npz_path} for "
            f"layer={feature_layer}, timestep={feature_timestep}, branch={feature_branch}."
        )

    sel_features = features[mask]
    sel_class_ids = class_ids[mask]

    # Recover original dimensionality if padded
    if feature_lengths is not None:
        sel_lengths = feature_lengths[mask]
        d_orig = int(sel_lengths.max())
        sel_features = sel_features[:, :d_orig]

    # Optional subsampling
    N = sel_features.shape[0]
    if max_samples is not None and N > max_samples:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(N, size=max_samples, replace=False)
        sel_features = sel_features[idx]
        sel_class_ids = sel_class_ids[idx]

    # PCA -> 2D
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(sel_features)

    return Z, sel_class_ids, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Plotting function with marginal distributions
# ---------------------------------------------------------------------------

def plot_anisotropy_with_marginals(
    fig: plt.Figure,
    position: gridspec.SubplotSpec,
    Z: np.ndarray,
    anisotropy_result: dict[str, Any],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    pca_var: np.ndarray,
    conf_level: float = 0.95,
    class_ids: Optional[np.ndarray] = None,
    grid_res: int = 80,
    show_legend: bool = False,
) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
    """
    Plot anisotropy visualization with marginal distributions in a jointplot style.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure to plot on.
    position : gridspec.SubplotSpec
        The position in the parent gridspec.
    Z : ndarray of shape (N, 2)
        2D latent coordinates.
    anisotropy_result : dict
        Output of compute_local_anisotropy.
    xlim : tuple of (xmin, xmax)
        X-axis limits (fixed for all subplots).
    ylim : tuple of (ymin, ymax)
        Y-axis limits (fixed for all subplots).
    pca_var : ndarray of shape (2,)
        PCA explained variance ratios.
    conf_level : float, default 0.95
        Confidence level for high-density region.
    class_ids : ndarray or None
        Optional class labels for coloring.
    grid_res : int, default 80
        Resolution of KDE grid for contours.
    show_legend : bool, default False
        Whether to show legend on this subplot.

    Returns
    -------
    ax_main, ax_top, ax_right : matplotlib Axes
        The three axes created.
    """
    # Create sub-gridspec for this subplot
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=position,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.02,
        wspace=0.02,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.axis("off")

    kde = anisotropy_result["kde"]
    density = anisotropy_result["density"]
    thr = anisotropy_result["threshold"]
    high_idx = anisotropy_result["high_idx"]
    regions = anisotropy_result["regions"]

    x = Z[:, 0]
    y = Z[:, 1]

    # Use fixed limits
    x_min, x_max = xlim
    y_min, y_max = ylim

    # Grid for contours
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid_points).reshape(xx.shape)

    # Main plot: Scatter
    if class_ids is None:
        ax_main.scatter(
            x[~high_idx], y[~high_idx],
            s=3, color="black", alpha=0.3, label="Sample",
        )
        ax_main.scatter(
            x[high_idx], y[high_idx],
            s=3, color="blue", alpha=0.5,
            label=f"{int(100 * conf_level)}% high-density",
        )
    else:
        ax_main.scatter(
            x[~high_idx], y[~high_idx],
            s=2, color="lightgray", alpha=0.2,
            label="Outside high-density",
        )
        cmap = plt.get_cmap("tab10")
        unique_classes = np.unique(class_ids)
        for i, cls in enumerate(unique_classes):
            mask = (class_ids == cls) & high_idx
            if not np.any(mask):
                continue
            ax_main.scatter(
                x[mask], y[mask],
                s=5, alpha=0.6,
                color=cmap(i % 10),
                label=f"class {cls}",
            )

    # KDE contours
    levels = np.linspace(zz.min(), zz.max(), 4)[1:]
    ax_main.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.4, alpha=0.4)

    # High-density contour
    ax_main.contour(
        xx, yy, zz,
        levels=[thr],
        colors="blue",
        linestyles="--",
        linewidths=0.8,
    )

    # Anisotropy ellipses and arrows
    for i, reg in enumerate(regions, start=1):
        center = reg["center"]
        eigvals = reg["eigvals"]
        eigvecs = reg["eigvecs"]
        beta = reg["beta"]

        width = 2.0 * np.sqrt(eigvals[0])
        height = 2.0 * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            xy=center, width=width, height=height, angle=angle,
            edgecolor="red", facecolor="none", lw=0.8,
        )
        ax_main.add_patch(ell)

        direction = eigvecs[:, 0]
        ax_main.arrow(
            center[0], center[1],
            direction[0] * width * 0.5, direction[1] * width * 0.5,
            head_width=0.06 * width, head_length=0.06 * width,
            fc="red", ec="red", length_includes_head=True,
        )
        ax_main.text(
            center[0], center[1],
            rf"$\beta_{{{i}}}={beta:.1f}$",
            color="red", fontsize=5,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
        )

    # Set axis labels with variance explained
    ax_main.set_xlabel(f"PC1 ({pca_var[0]*100:.1f}%)", fontsize=6)
    ax_main.set_ylabel(f"PC2 ({pca_var[1]*100:.1f}%)", fontsize=6)
    ax_main.tick_params(labelsize=5)
    ax_main.grid(True, linestyle="--", alpha=0.2)
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    if show_legend:
        ax_main.legend(loc="upper right", fontsize=4, ncol=2)

    # Top histogram (x marginal)
    ax_top.hist(x, bins=20, density=False, color="lightgray", edgecolor="black", linewidth=0.5)
    kde_x = gaussian_kde(x)
    xs = np.linspace(x_min, x_max, 100)
    bin_width = (x_max - x_min) / 20
    ax_top.plot(xs, kde_x(xs) * len(x) * bin_width, color="black", linewidth=0.8)
    ax_top.set_ylabel("Count", fontsize=5)
    ax_top.tick_params(labelsize=5, labelbottom=False)
    ax_top.grid(True, linestyle="--", alpha=0.2)

    # Right histogram (y marginal)
    ax_right.hist(
        y, bins=20, density=False,
        orientation="horizontal",
        color="lightgray", edgecolor="black", linewidth=0.5,
    )
    kde_y = gaussian_kde(y)
    ys = np.linspace(y_min, y_max, 100)
    bin_width = (y_max - y_min) / 20
    ax_right.plot(
        kde_y(ys) * len(y) * bin_width, ys,
        color="black", linewidth=0.8,
    )
    ax_right.set_xlabel("Count", fontsize=5)
    ax_right.tick_params(labelsize=5, labelleft=False)
    ax_right.grid(True, linestyle="--", alpha=0.2)

    return ax_main, ax_top, ax_right


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_timestep_evolution(
    features_path: Path,
    output_path: Path,
    feature_layer: str,
    timestep: int,
    epochs_to_plot: List[int],
    conf_level: float = 0.95,
    eps: float = 0.2,
    min_samples: int = 10,
    random_state: int = 42,
    max_features: Optional[int] = None,
) -> None:
    """
    Create a multi-panel figure for a single timestep showing evolution across epochs.

    Parameters
    ----------
    features_path : Path
        Directory containing probe_features_epoch_*.npz files.
    output_path : Path
        Output directory for the figure.
    feature_layer : str
        Layer name to visualize.
    timestep : int
        Diffusion timestep to visualize.
    epochs_to_plot : List[int]
        List of epoch numbers to plot (columns).
    conf_level : float, default 0.95
        High-density probability mass.
    eps : float, default 0.2
        DBSCAN eps parameter.
    min_samples : int, default 10
        DBSCAN min_samples parameter.
    random_state : int, default 42
        Random seed.
    max_features : int or None
        Optional cap on number of feature vectors.
    """
    branches = ["uncond", "cond", "both"]
    n_epochs = len(epochs_to_plot)
    n_branches = len(branches)

    # FIRST PASS: Load all data and compute global limits
    logging.info(f"First pass: loading all data to compute global limits...")
    all_data = {}
    x_min_global, x_max_global = np.inf, -np.inf
    y_min_global, y_max_global = np.inf, -np.inf

    for branch in branches:
        for epoch in epochs_to_plot:
            npz_file = features_path / f"probe_features_epoch_{epoch:04d}.npz"
            if not npz_file.exists():
                logging.warning(f"File not found: {npz_file}, skipping...")
                continue

            try:
                Z, class_ids, pca_var = load_probe_features_2d_with_variance(
                    npz_path=str(npz_file),
                    feature_layer=feature_layer,
                    feature_timestep=timestep,
                    feature_branch=branch,
                    random_state=random_state,
                    max_samples=max_features,
                )

                result = compute_local_anisotropy(
                    Z,
                    conf_level=conf_level,
                    eps=eps,
                    min_samples=min_samples,
                )

                # Store for later
                all_data[(branch, epoch)] = (Z, class_ids, pca_var, result)

                # Update global limits
                x_min_global = min(x_min_global, Z[:, 0].min())
                x_max_global = max(x_max_global, Z[:, 0].max())
                y_min_global = min(y_min_global, Z[:, 1].min())
                y_max_global = max(y_max_global, Z[:, 1].max())

            except Exception as e:
                logging.error(
                    f"Error loading epoch={epoch}, branch={branch}, timestep={timestep}: {e}"
                )
                continue

    # Add margin to global limits
    margin = 0.1
    dx = x_max_global - x_min_global
    dy = y_max_global - y_min_global
    x_min_global -= margin * dx
    x_max_global += margin * dx
    y_min_global -= margin * dy
    y_max_global += margin * dy

    logging.info(f"Global limits: x=[{x_min_global:.2f}, {x_max_global:.2f}], y=[{y_min_global:.2f}, {y_max_global:.2f}]")

    # SECOND PASS: Create figure with consistent limits
    fig = plt.figure(figsize=(4 * n_epochs, 4 * n_branches))
    gs = gridspec.GridSpec(
        n_branches,
        n_epochs,
        figure=fig,
        hspace=0.35,
        wspace=0.25,
    )

    # Track handles and labels for shared legend
    all_handles = []
    all_labels = []

    for row_idx, branch in enumerate(branches):
        for col_idx, epoch in enumerate(epochs_to_plot):
            key = (branch, epoch)
            if key not in all_data:
                continue

            Z, class_ids, pca_var, result = all_data[key]

            # Create subplot with marginals
            try:
                ax_main, ax_top, ax_right = plot_anisotropy_with_marginals(
                    fig=fig,
                    position=gs[row_idx, col_idx],
                    Z=Z,
                    anisotropy_result=result,
                    xlim=(x_min_global, x_max_global),
                    ylim=(y_min_global, y_max_global),
                    pca_var=pca_var,
                    conf_level=conf_level,
                    class_ids=class_ids,
                    show_legend=False,
                )

                # Collect legend info from first subplot only
                if row_idx == 0 and col_idx == 0 and not all_handles:
                    handles, labels = ax_main.get_legend_handles_labels()
                    all_handles = handles
                    all_labels = labels

                # Set titles: epoch on top row, branch on left column
                if row_idx == 0:
                    ax_top.set_title(f"Epoch {epoch}", fontsize=10, fontweight='bold', pad=5)
                if col_idx == 0:
                    ax_main.set_ylabel(
                        f"{branch.capitalize()}\n{ax_main.get_ylabel()}",
                        fontsize=8,
                        fontweight='bold',
                    )

            except Exception as e:
                logging.error(
                    f"Error plotting epoch={epoch}, branch={branch}, timestep={timestep}: {e}"
                )
                continue

    # Add shared legend at the bottom center
    if all_handles:
        fig.legend(
            all_handles,
            all_labels,
            loc='lower center',
            ncol=min(len(all_labels), 6),
            fontsize=7,
            bbox_to_anchor=(0.5, -0.01),
            frameon=True,
        )

    # Overall title
    fig.suptitle(
        f"Anisotropy Evolution: Layer={feature_layer}, Timestep={timestep}",
        fontsize=13,
        fontweight='bold',
        y=0.995,
    )

    # Save figure
    output_file = output_path / f"anisotropy_evolution_t{timestep}_{feature_layer.replace('.', '_')}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved: {output_file}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def main():
    #  python medsyn/analysis/embeddings/time_sensitive_anisotropy_evolution.py --features_path /media/mpascual/PortableSSD/medsyn/experiments/AdamW_Batch64_lowt_enhancement_gamma5/embeddings --output_path /media/mpascual/PortableSSD/medsyn/experiments/AdamW_Batch64_lowt_enhancement_gamma5/analysis/embeddings --max_epoch 10 --num_epochs_to_plot 10 
    parser = argparse.ArgumentParser(
        description="Time-sensitive anisotropy evolution visualization."
    )
    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Directory containing probe_features_epoch_*.npz files.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--feature_layer",
        type=str,
        default="unet.mid_block",
        help="Layer name to visualize (e.g., 'class_embed', 'unet.mid_block').",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=None,
        help="Maximum epoch to consider. If None, use the highest available epoch.",
    )
    parser.add_argument(
        "--num_epochs_to_plot",
        type=int,
        default=10,
        help="Number of epochs to plot (will be evenly spaced from 0 to max_epoch).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Specific timesteps to plot (e.g., 100 500 900). If None, auto-detect from first NPZ.",
    )
    parser.add_argument(
        "--conf_level",
        type=float,
        default=0.95,
        help="High-density probability mass (default 0.95).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.2,
        help="DBSCAN eps parameter.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="DBSCAN min_samples parameter.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Optional cap on number of feature vectors (for speed).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    features_path = Path(args.features_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get available epochs
    available_epochs = get_available_epochs(features_path)
    if not available_epochs:
        raise ValueError(f"No probe_features_epoch_*.npz files found in {features_path}")

    logging.info(f"Found {len(available_epochs)} epochs: {available_epochs}")

    # Determine max epoch
    if args.max_epoch is None:
        max_epoch = max(available_epochs)
    else:
        max_epoch = args.max_epoch

    # Select epochs to plot (evenly spaced from 0 to max_epoch)
    epochs_to_plot = np.linspace(0, max_epoch, args.num_epochs_to_plot, dtype=int).tolist()
    # Filter to only include available epochs
    epochs_to_plot = [e for e in epochs_to_plot if e in available_epochs]

    if not epochs_to_plot:
        raise ValueError("No valid epochs to plot after filtering.")

    logging.info(f"Epochs to plot: {epochs_to_plot}")

    # Determine timesteps
    if args.timesteps is None:
        # Auto-detect from first available NPZ
        first_npz = features_path / f"probe_features_epoch_{available_epochs[0]:04d}.npz"
        timesteps = get_available_timesteps(first_npz)
        logging.info(f"Auto-detected timesteps: {timesteps}")
    else:
        timesteps = args.timesteps
        logging.info(f"Using specified timesteps: {timesteps}")

    # Generate plots for each timestep
    for timestep in timesteps:
        logging.info(f"Processing timestep {timestep}...")
        plot_timestep_evolution(
            features_path=features_path,
            output_path=output_path,
            feature_layer=args.feature_layer,
            timestep=timestep,
            epochs_to_plot=epochs_to_plot,
            conf_level=args.conf_level,
            eps=args.eps,
            min_samples=args.min_samples,
            random_state=args.random_state,
            max_features=args.max_features,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
