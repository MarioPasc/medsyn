#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local anisotropic ratio visualization for class embedding latent space.

Given a trajectory file with:
  - "epochs": list[int], length E
  - "embeddings": tensor [E, num_classes, emb_dim]

this script:
  1) Selects embeddings (all epochs or a single epoch).
  2) Reduces them to 2D with PCA.
  3) Computes a Gaussian KDE in 2D latent space.
  4) Identifies a high-density region (e.g., 95% highest density mass).
  5) Clusters that region and computes a covariance-based local anisotropic
     ratio β_local for each cluster.
  6) Generates a jointplot-style figure with:
       - Scatter + KDE contours,
       - 95% contour,
       - Local anisotropy ellipses and arrows,
       - Marginal histograms.

Usage example:

  python local_anisotropy_plot.py \\
      --trajectory_path class_embeddings_trajectory.pt \\
      --output_path local_anisotropy.png \\
      --use_all_epochs

The interpretation is analogous to Mabadeje et al. (2024):
larger β_local indicates stronger directional anisotropy of the latent
point cloud in that high-density region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


def load_embeddings_2d(
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
        DBSCAN eps parameter for clustering high-density points (in latent units).
    min_samples : int, default 10
        DBSCAN min_samples parameter.

    Returns
    -------
    result : dict
        Dictionary with keys:
            'kde' : gaussian_kde object.
            'density' : KDE values at sample points, shape (N,).
            'threshold' : density threshold for high-density region.
            'labels' : cluster labels for high-density points (array of shape (Nh,)).
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


def plot_local_anisotropy_joint(
    Z: np.ndarray,
    anisotropy_result: dict[str, Any],
    out_path: str | Path,
    title: str | None = None,
    grid_res: int = 200,
    conf_level: float = 0.95,
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

    levels = np.linspace(zz.min(), zz.max(), 7)[1:]
    cs = ax_main.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.8)
    ax_main.clabel(cs, inline=True, fontsize=6, fmt="%.2f")

    # Contour close to the 95% high-density threshold
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


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Local anisotropy visualization for class embedding latent space."
    )
    parser.add_argument(
        "--trajectory_path",
        type=str,
        required=True,
        help="Path to class_embeddings_trajectory.pt",
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
        help="If set, use embeddings from all epochs; otherwise use last epoch.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Specific epoch to plot (used only if --use_all_epochs is not set).",
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
        help="DBSCAN eps parameter in latent units (tune if clusters merge/split).",
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
        help="Random seed for PCA.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    Z = load_embeddings_2d(
        trajectory_path=args.trajectory_path,
        use_all_epochs=args.use_all_epochs,
        epoch=args.epoch,
        random_state=args.random_state,
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
        title="Local anisotropic ratio from class embedding latent space",
        conf_level=args.conf_level,
    )
