#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local anisotropic ratio visualization for ccDDPM latent spaces.

Two modes are supported:

1) Class mode ("class"):
   - Input: class_embeddings_trajectory.pt
   - Uses parametric class embeddings only.

2) Features mode ("features"):
   - Input: probe_features_epoch_XXXX.npz
   - Uses probe feature vectors (conditional/unconditional, layer, timestep)
     as the latent cloud, which is much closer to the setup of Mabadeje et al.

In both cases the script:
  1) Selects a point cloud in some latent space R^d.
  2) Reduces it to 2D with PCA.
  3) Computes a Gaussian KDE in 2D latent space.
  4) Identifies a high-density region (e.g. 95% highest-density mass).
  5) Clusters that region and computes a covariance-based local anisotropic
     ratio β_local for each cluster (via eigenvalues of the covariance).
  6) Generates a jointplot-style figure with:
       - Scatter + KDE contours,
       - 95% contour,
       - Local anisotropy ellipses and arrows,
       - Marginal histograms.

Usage examples:

  # Original behaviour: class embeddings only
  python anisotropy_visualization.py \\
      --mode class \\
      --trajectory_path class_embeddings_trajectory.pt \\
      --output_path anisotropy_class.png \\
      --use_all_epochs

  # New behaviour: probe features from one epoch
  python anisotropy_visualization.py \\
      --mode features \\
      --features_path probe_features_epoch_0005.npz \\
      --feature_layer class_embed \\
      --feature_branch cond \\
      --feature_timestep 500 \\
      --output_path anisotropy_features_t500.png

Interpretation follows Mabadeje et al. (2024):
larger β_local indicates stronger directional anisotropy of the latent
point cloud in that high-density region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_class_embeddings_2d(
    trajectory_path: str,
    use_all_epochs: bool = True,
    epoch: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Load class embedding trajectory and reduce to 2D with PCA.

    Parameters
    ----------
    trajectory_path : str
        Path to .pt file with keys "epochs" and "embeddings" [E, C, D].
    use_all_epochs : bool, default True
        If True, use embeddings from all epochs.
        If False and `epoch` is not None, use only embeddings from that epoch.
    epoch : int or None
        Epoch number to select when use_all_epochs is False.
        If None and use_all_epochs is False, the last epoch is used.
    random_state : int
        Random state for PCA reproducibility.

    Returns
    -------
    Z : ndarray of shape (N, 2)
        2D PCA projection of selected embeddings.
        N = E * C if use_all_epochs, else N = C (one epoch).
    """
    state: dict[str, Any] = torch.load(trajectory_path, map_location="cpu")
    emb = state["embeddings"]  # [E, C, D]
    epochs = state["epochs"]
    E, C, D = emb.shape

    if use_all_epochs:
        X = emb.reshape(E * C, D)
    else:
        if epoch is None:
            e_idx = -1  # last epoch
        else:
            if epoch in epochs:
                e_idx = list(epochs).index(epoch)
            else:
                raise ValueError(f"Requested epoch {epoch} not in stored epochs {epochs}.")
        X = emb[e_idx]  # [C, D]

    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(X)
    return Z


def load_probe_features_2d(
    npz_path: str,
    feature_layer: str,
    feature_timestep: Optional[int],
    feature_branch: str = "cond",
    random_state: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load probe feature vectors from NPZ and reduce to 2D with PCA.

    Parameters
    ----------
    npz_path : str
        Path to probe_features_epoch_XXXX.npz.
    feature_layer : str
        Layer name to select (e.g. "class_embed" or "down_blocks.1").
    feature_timestep : int or None
        Timestep to select. If None, use the median timestep present in the file.
    feature_branch : {"cond", "uncond", "both"}, default "cond"
        Which CFG branch to include.
    random_state : int
        Random seed for PCA reproducibility.
    max_samples : int or None
        Optional maximum number of feature vectors to subsample (for speed).

    Returns
    -------
    Z : ndarray of shape (N, 2)
        2D PCA projection of selected features.
    class_ids : ndarray of shape (N,)
        Class IDs corresponding to each feature vector (for coloring).
    """
    data = np.load(npz_path, allow_pickle=True)

    features = data["features"]          # [N, D_pad]
    class_ids = data["class_ids"]       # [N]
    timesteps = data["timesteps"]       # [N]
    layer_names = data["layer_names"]   # [N]
    branches = data["branches"]         # [N]
    feature_lengths = data.get("feature_lengths", None)  # [N] or None

    # Ensure string comparison works (npz may give bytes or unicode)
    layer_names = layer_names.astype(str)
    branches = branches.astype(str)

    # Decide timestep if not provided: use the median available
    unique_t = np.unique(timesteps)
    if feature_timestep is None:
        feature_timestep = int(np.median(unique_t))

    # Build mask: layer, timestep, branch
    mask = (layer_names == feature_layer) & (timesteps == feature_timestep)
    if feature_branch.lower() in {"cond", "uncond"}:
        mask &= (branches == feature_branch.lower())
    # "both" keeps both branches

    if not np.any(mask):
        raise ValueError(
            f"No features found in {npz_path} for "
            f"layer={feature_layer}, timestep={feature_timestep}, branch={feature_branch}."
        )

    sel_features = features[mask]
    sel_class_ids = class_ids[mask]

    # If features were padded, recover the original dimensionality for this layer
    if feature_lengths is not None:
        sel_lengths = feature_lengths[mask]
        # All records for a single layer should share the same original length
        d_orig = int(sel_lengths.max())
        sel_features = sel_features[:, :d_orig]

    # Optional subsampling for speed
    N = sel_features.shape[0]
    if max_samples is not None and N > max_samples:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(N, size=max_samples, replace=False)
        sel_features = sel_features[idx]
        sel_class_ids = sel_class_ids[idx]

    # PCA -> 2D
    pca = PCA(n_components=2, random_state=random_state)
    Z = pca.fit_transform(sel_features)

    return Z, sel_class_ids


# ---------------------------------------------------------------------------
# KDE and local anisotropy
# ---------------------------------------------------------------------------

def compute_local_anisotropy(
    Z: np.ndarray,
    conf_level: float = 0.95,
    bandwidth: float | None = None,
    eps: float = 0.2,
    min_samples: int = 10,
) -> dict[str, Any]:
    """
    Compute KDE-based high-density regions and local anisotropy per region.

    Parameters
    ----------
    Z : ndarray of shape (N, 2)
        2D latent coordinates.
    conf_level : float, default 0.95
        Probability mass for high-density region (e.g. 0.95).
    bandwidth : float or None
        Optional manual bandwidth for gaussian_kde. If None, use Scott's rule.
    eps : float, default 0.2
        DBSCAN eps parameter for clustering high-density points (latent units).
    min_samples : int, default 10
        DBSCAN min_samples parameter.

    Returns
    -------
    result : dict
        Dictionary with keys:
            'kde' : gaussian_kde object.
            'density' : KDE values at sample points, shape (N,).
            'threshold' : density threshold for high-density region.
            'labels' : cluster labels for high-density points (shape (Nh,)).
            'high_idx' : boolean mask for high-density points (shape (N,)).
            'regions' : list of dicts with keys
                        {'points', 'center', 'eigvals', 'eigvecs', 'beta'}.
    """
    if Z.shape[1] != 2:
        raise ValueError("Z must be of shape (N, 2).")

    kde = gaussian_kde(Z.T, bw_method=bandwidth)
    density = kde(Z.T)

    # Approximate highest-density region of mass `conf_level`:
    # choose threshold so that ~conf_level fraction of points have density >= thr.
    thr = np.quantile(density, 1.0 - conf_level)
    high_idx = density >= thr
    Z_high = Z[high_idx]

    if Z_high.shape[0] < min_samples:
        labels = np.full(Z_high.shape[0], -1, dtype=int)
        regions: list[dict[str, Any]] = []
        return {
            "kde": kde,
            "density": density,
            "threshold": thr,
            "labels": labels,
            "high_idx": high_idx,
            "regions": regions,
        }

    # Cluster the high-density region to separate distinct lobes.
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(Z_high)

    regions: list[dict[str, Any]] = []
    for lab in sorted(set(labels)):
        if lab == -1:
            continue  # noise
        pts = Z_high[labels == lab]
        if pts.shape[0] < 2:
            continue

        center = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Anisotropy ratio β = sqrt(lambda1 / lambda2)
        if eigvals[1] > 0:
            beta = float(np.sqrt(eigvals[0] / eigvals[1]))
        else:
            beta = float("inf")

        regions.append(
            {
                "points": pts,
                "center": center,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "beta": beta,
            }
        )

    return {
        "kde": kde,
        "density": density,
        "threshold": thr,
        "labels": labels,
        "high_idx": high_idx,
        "regions": regions,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_local_anisotropy_joint(
    Z: np.ndarray,
    anisotropy_result: dict[str, Any],
    out_path: str | Path,
    title: str | None = None,
    grid_res: int = 200,
    conf_level: float = 0.95,
    class_ids: Optional[np.ndarray] = None,
) -> None:
    """
    Generate a joint-plot-style figure with KDE contours, marginal histograms,
    and local anisotropy vectors/ellipses.

    Parameters
    ----------
    Z : ndarray of shape (N, 2)
        2D latent coordinates.
    anisotropy_result : dict
        Output of `compute_local_anisotropy`.
    out_path : str or Path
        Output image path (PNG).
    title : str or None
        Optional figure title.
    grid_res : int, default 200
        Resolution of KDE grid for contours.
    conf_level : float, default 0.95
        Confidence level used when labelling the high-density region.
    class_ids : ndarray or None
        Optional class labels for coloring high-density points.
    """
    Z = np.asarray(Z)
    kde = anisotropy_result["kde"]
    density = anisotropy_result["density"]
    thr = anisotropy_result["threshold"]
    high_idx = anisotropy_result["high_idx"]
    regions = anisotropy_result["regions"]

    x = Z[:, 0]
    y = Z[:, 1]

    # Grid for contours
    margin = 0.1
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    dx = x_max - x_min
    dy = y_max - y_min
    x_min -= margin * dx
    x_max += margin * dx
    y_min -= margin * dy
    y_max += margin * dy

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid_points).reshape(xx.shape)

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[4, 1.2],
        height_ratios=[1.2, 4],
        hspace=0.0,
        wspace=0.0,
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    fig.add_subplot(gs[0, 1]).axis("off")  # empty corner

    # Main scatter and KDE contours
    if class_ids is None:
        # Original behavior: no class coloring, just samples vs high-density region
        ax_main.scatter(
            x[~high_idx],
            y[~high_idx],
            s=10,
            color="black",
            alpha=0.5,
            label="Sample",
        )
        ax_main.scatter(
            x[high_idx],
            y[high_idx],
            s=10,
            color="blue",
            alpha=0.7,
            label=f"{int(100 * conf_level)}% high-density region",
        )
    else:
        # Outside high-density region: gray
        ax_main.scatter(
            x[~high_idx],
            y[~high_idx],
            s=8,
            color="lightgray",
            alpha=0.4,
            label="Outside high-density",
        )
        # Inside high-density: color-coded by class
        cmap = plt.get_cmap("tab10")
        unique_classes = np.unique(class_ids)
        for i, cls in enumerate(unique_classes):
            mask = (class_ids == cls) & high_idx
            if not np.any(mask):
                continue
            ax_main.scatter(
                x[mask],
                y[mask],
                s=18,
                alpha=0.8,
                color=cmap(i % 10),
                label=f"class {cls} (high-density)",
            )

    levels = np.linspace(zz.min(), zz.max(), 7)[1:]
    cs = ax_main.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.8)
    ax_main.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    # Contour close to the high-density threshold
    ax_main.contour(
        xx,
        yy,
        zz,
        levels=[thr],
        colors="blue",
        linestyles="--",
        linewidths=1.2,
    )

    # Local anisotropy vectors/ellipses
    for i, reg in enumerate(regions, start=1):
        center = reg["center"]
        eigvals = reg["eigvals"]
        eigvecs = reg["eigvecs"]
        beta = reg["beta"]

        # Ellipse parameters: 2 std along each axis
        width = 2.0 * np.sqrt(eigvals[0])
        height = 2.0 * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ell = Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            edgecolor="red",
            facecolor="none",
            lw=1.5,
        )
        ax_main.add_patch(ell)

        # Arrow along major axis
        direction = eigvecs[:, 0]
        ax_main.arrow(
            center[0],
            center[1],
            direction[0] * width * 0.5,
            direction[1] * width * 0.5,
            head_width=0.1 * width,
            head_length=0.1 * width,
            fc="red",
            ec="red",
            length_includes_head=True,
        )
        ax_main.text(
            center[0],
            center[1],
            rf"$\beta_{{local}}^{({i})}={beta:.2f}$",
            color="red",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
        )

    ax_main.set_xlabel("LS 1")
    ax_main.set_ylabel("LS 2")
    ax_main.legend(loc="upper right", fontsize=8)
    ax_main.grid(True, linestyle="--", alpha=0.3)

    # Top histogram (x marginal)
    ax_top.hist(x, bins=30, density=False, color="lightgray", edgecolor="black")
    kde_x = gaussian_kde(x)
    xs = np.linspace(x_min, x_max, 200)
    ax_top.plot(xs, kde_x(xs) * len(x) * (xs[1] - xs[0]), color="black")
    ax_top.set_ylabel("Frequency")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.grid(True, linestyle="--", alpha=0.3)

    # Right histogram (y marginal)
    ax_right.hist(
        y,
        bins=30,
        density=False,
        orientation="horizontal",
        color="lightgray",
        edgecolor="black",
    )
    kde_y = gaussian_kde(y)
    ys = np.linspace(y_min, y_max, 200)
    ax_right.plot(
        kde_y(ys) * len(y) * (ys[1] - ys[0]),
        ys,
        color="black",
    )
    ax_right.set_xlabel("Frequency")
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.grid(True, linestyle="--", alpha=0.3)

    if title is not None:
        fig.suptitle(title, y=0.98)

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    python medsyn/analysis/embeddings/anisotropy_visualization.py --mode features --trajectory_path /media/mpascual/PortableSSD/medsyn/experiments/AdamW_Batch64_lowt_enhancement_gamma5/embeddings/class_embeddings_trajectory.pt --features_path /media/mpascual/PortableSSD/medsyn/experiments/AdamW_Batch64_lowt_enhancement_gamma5/embeddings/probe_features_epoch_0031.npz --output_path /media/mpascual/PortableSSD/medsyn/experiments/AdamW_Batch64_lowt_enhancement_gamma5/analysis/embeddings/anisotropy_visualization.png --feature_timestep 100 --feature_branch cond
    """
    
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Local anisotropy visualization for ccDDPM latent space."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["class", "features"],
        default="class",
        help="Whether to use class embeddings or probe features.",
    )
    parser.add_argument(
        "--trajectory_path",
        type=str,
        default=None,
        help="Path to class_embeddings_trajectory.pt (used in 'class' mode).",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Path to probe_features_epoch_XXXX.npz (used in 'features' mode).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--use_all_epochs",
        action="store_true",
        help="[class mode] If set, use embeddings from all epochs; "
             "otherwise use a single epoch.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="[class mode] Specific epoch to plot if --use_all_epochs is not set.",
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
        help="DBSCAN eps parameter in latent units.",
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
        help="Random seed for PCA and subsampling.",
    )
    # Feature-specific options
    parser.add_argument(
        "--feature_layer",
        type=str,
        default="unet.mid_block",
        help="[features mode] Layer name to select (e.g. 'class_embed', 'down_blocks.1').",
    )
    parser.add_argument(
        "--feature_timestep",
        type=int,
        default=None,
        help="[features mode] Timestep to select. If omitted, median timestep is used.",
    )
    parser.add_argument(
        "--feature_branch",
        type=str,
        default="cond",
        choices=["cond", "uncond", "both"],
        help="[features mode] Which CFG branch to include.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="[features mode] Optional cap on number of feature vectors (for speed).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    if args.mode == "class":
        if args.trajectory_path is None:
            parser.error("--trajectory_path is required in 'class' mode.")
        Z = load_class_embeddings_2d(
            trajectory_path=args.trajectory_path,
            use_all_epochs=args.use_all_epochs,
            epoch=args.epoch,
            random_state=args.random_state,
        )
        class_ids = None
        title = "Local anisotropic ratio from class embedding latent space"
    else:
        if args.features_path is None:
            parser.error("--features_path is required in 'features' mode.")
        Z, class_ids = load_probe_features_2d(
            npz_path=args.features_path,
            feature_layer=args.feature_layer,
            feature_timestep=args.feature_timestep,
            feature_branch=args.feature_branch,
            random_state=args.random_state,
            max_samples=args.max_features,
        )
        title = (
            f"Local anisotropy from probe features "
            f"(layer={args.feature_layer}, t={args.feature_timestep}, "
            f"branch={args.feature_branch})"
        )

    result = compute_local_anisotropy(
        Z,
        conf_level=args.conf_level,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    out_path = Path(args.output_path)
    plot_local_anisotropy_joint(
        Z,
        result,
        out_path=out_path,
        title=title,
        conf_level=args.conf_level,
        class_ids=class_ids,
    )
