#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze evolution of class-embedding trajectories across epochs.

Given a trajectory .pt file with:
  - "epochs": list[int], length E
  - "embeddings": tensor [E, num_classes, emb_dim]
  - "num_classes": int
  - "emb_dim": int

the script:
  1) Computes per-epoch and per-class metrics (norms, displacements, pairwise distances).
  2) Performs PCA, t-SNE, and UMAP dimensionality reduction on all snapshots jointly.
  3) Creates trajectory plots (2D) with arrows per class across epochs.
  4) Creates line plots for stability metrics vs. epoch.
  5) Writes a Markdown report summarizing the outputs.

Usage (example):
  python analyze_class_embeddings_trajectory.py \\
      --trajectory_path class_embeddings_trajectory.pt \\
      --output_dir ./class_embed_analysis

External dependencies:
  - torch
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - umap-learn
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# UMAP is optional but strongly recommended.
# Install via: pip install umap-learn
try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    umap = None
    UMAP_AVAILABLE = False

# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logger = logging.getLogger("class_embedding_analysis")


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass
class TrajectoryData:
    """Container for class embedding trajectory."""

    epochs: np.ndarray          # shape [E]
    embeddings: np.ndarray      # shape [E, C, D]
    num_classes: int
    emb_dim: int


@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization."""

    trajectory_path: Path
    output_dir: Path
    random_state: int = 42
    tsne_perplexity: float = 20.0
    tsne_n_iter: int = 2000
    umap_n_neighbors: int = 10
    umap_min_dist: float = 0.1


# -------------------------------------------------------------------
# Utility: loading trajectory
# -------------------------------------------------------------------

# Load the trajectory .pt file into a TrajectoryData structure.
def load_trajectory(path: Path) -> TrajectoryData:
    state = torch.load(path, map_location="cpu")
    if "epochs" not in state or "embeddings" not in state:
        raise RuntimeError(
            f"Trajectory file {path} does not contain required keys "
            "'epochs' and 'embeddings'."
        )

    epochs = np.array(state["epochs"], dtype=int)
    embeddings_t = state["embeddings"]

    if not torch.is_tensor(embeddings_t):
        raise RuntimeError("Expected 'embeddings' in .pt file to be a torch.Tensor.")

    embeddings = embeddings_t.detach().cpu().numpy()
    if embeddings.ndim != 3:
        raise RuntimeError(
            f"Expected embeddings to have shape [E, C, D], got {embeddings.shape}."
        )

    E, C, D = embeddings.shape
    num_classes = int(state.get("num_classes", C))
    emb_dim = int(state.get("emb_dim", D))

    if num_classes != C or emb_dim != D:
        logger.warning(
            "Inconsistent metadata: (num_classes, emb_dim)=(%d,%d) "
            "but embeddings.shape[1:]=(C=%d,D=%d). Using embeddings.shape.",
            num_classes, emb_dim, C, D
        )
        num_classes, emb_dim = C, D

    if len(epochs) != E:
        raise RuntimeError(
            f"Length of epochs ({len(epochs)}) does not match embeddings.shape[0] ({E})."
        )

    logger.info(
        "Loaded trajectory: epochs=%d, num_classes=%d, emb_dim=%d",
        E, num_classes, emb_dim
    )

    return TrajectoryData(
        epochs=epochs,
        embeddings=embeddings,
        num_classes=num_classes,
        emb_dim=emb_dim,
    )


# -------------------------------------------------------------------
# Metric computation
# -------------------------------------------------------------------

# Compute per-epoch summary metrics (norm statistics, pairwise distances, step size).
def compute_epoch_metrics(traj: TrajectoryData) -> pd.DataFrame:
    E, C, D = traj.embeddings.shape
    records = []

    for e_idx in range(E):
        epoch_id = int(traj.epochs[e_idx])
        emb_e = traj.embeddings[e_idx]  # [C, D]

        norms = np.linalg.norm(emb_e, axis=1)  # [C]
        mean_norm = float(norms.mean())
        min_norm = float(norms.min())
        max_norm = float(norms.max())

        center = emb_e.mean(axis=0, keepdims=True)  # [1, D]
        dist_to_center = np.linalg.norm(emb_e - center, axis=1)
        mean_radius = float(dist_to_center.mean())
        max_radius = float(dist_to_center.max())

        # Pairwise distances (upper triangle)
        diffs = emb_e[:, None, :] - emb_e[None, :, :]  # [C, C, D]
        iu = np.triu_indices(C, k=1)
        pairwise = np.linalg.norm(diffs[iu], axis=1)   # [C*(C-1)/2]
        mean_pairwise = float(pairwise.mean())
        min_pairwise = float(pairwise.min())
        max_pairwise = float(pairwise.max())

        # Per-epoch step size (mean displacement vs previous epoch)
        if e_idx == 0:
            mean_step = np.nan
        else:
            emb_prev = traj.embeddings[e_idx - 1]
            step = np.linalg.norm(emb_e - emb_prev, axis=1)
            mean_step = float(step.mean())

        # Anisotropy via singular values of centered embeddings
        emb_centered = emb_e - emb_e.mean(axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(emb_centered, full_matrices=False)
        if np.any(s > 0):
            sv_ratio = float(s.max() / s[s > 0].min())
        else:
            sv_ratio = np.nan

        records.append(
            {
                "epoch": epoch_id,
                "mean_norm": mean_norm,
                "min_norm": min_norm,
                "max_norm": max_norm,
                "mean_radius": mean_radius,
                "max_radius": max_radius,
                "mean_pairwise_dist": mean_pairwise,
                "min_pairwise_dist": min_pairwise,
                "max_pairwise_dist": max_pairwise,
                "mean_step_norm": mean_step,
                "singular_value_ratio": sv_ratio,
            }
        )

    df = pd.DataFrame.from_records(records)
    return df


# Compute per-class displacement and norms across epochs.
def compute_per_class_metrics(traj: TrajectoryData) -> pd.DataFrame:
    E, C, D = traj.embeddings.shape
    records = []

    for e_idx in range(E):
        epoch_id = int(traj.epochs[e_idx])
        emb_e = traj.embeddings[e_idx]  # [C, D]
        norms = np.linalg.norm(emb_e, axis=1)  # [C]

        if e_idx == 0:
            step = np.full(C, np.nan, dtype=float)
        else:
            emb_prev = traj.embeddings[e_idx - 1]
            step = np.linalg.norm(emb_e - emb_prev, axis=1)

        for c_idx in range(C):
            records.append(
                {
                    "epoch": epoch_id,
                    "class_id": int(c_idx),
                    "embedding_norm": float(norms[c_idx]),
                    "step_norm": float(step[c_idx]),
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


# -------------------------------------------------------------------
# Dimensionality reduction
# -------------------------------------------------------------------

# Flatten [E, C, D] into [E*C, D] and produce corresponding index arrays.
def flatten_embeddings(traj: TrajectoryData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    E, C, D = traj.embeddings.shape
    X = traj.embeddings.reshape(E * C, D)
    epoch_idx = np.repeat(np.arange(E), C)
    class_idx = np.tile(np.arange(C), E)
    return X, epoch_idx, class_idx


# Run PCA on flattened embeddings.
def run_pca_2d(X: np.ndarray, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    return X_2d, explained


# Run t-SNE on flattened embeddings.
def run_tsne_2d(
    X: np.ndarray,
    random_state: int,
    perplexity: float,
    n_iter: int,
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(X)
    return X_2d


# Run UMAP on flattened embeddings.
def run_umap_2d(
    X: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
) -> np.ndarray:
    if not UMAP_AVAILABLE:
        raise RuntimeError(
            "UMAP is not available. Please install 'umap-learn' to use this feature."
        )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    X_2d = reducer.fit_transform(X)
    return X_2d


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

# Create a trajectory plot in 2D (points + arrows), colored by class.
def plot_trajectories_2d(
    coords_2d: np.ndarray,
    traj: TrajectoryData,
    epoch_idx: np.ndarray,
    class_idx: np.ndarray,
    method_name: str,
    output_dir: Path,
) -> Path:
    E, C, _ = traj.embeddings.shape
    coords_reshaped = coords_2d.reshape(E, C, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(C)]

    for c in range(C):
        pts = coords_reshaped[:, c, :]  # [E, 2]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            color=colors[c],
            s=25,
            alpha=0.8,
            label=f"class {c}",
        )

        # Draw arrows between successive epochs for this class.
        ax.quiver(
            pts[:-1, 0],
            pts[:-1, 1],
            pts[1:, 0] - pts[:-1, 0],
            pts[1:, 1] - pts[:-1, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=colors[c],
            alpha=0.6,
            width=0.003,
        )

        # Optional: annotate first and last epoch for this class.
        ax.text(
            pts[0, 0],
            pts[0, 1],
            f"{traj.epochs[0]}",
            fontsize=6,
            color=colors[c],
        )
        ax.text(
            pts[-1, 0],
            pts[-1, 1],
            f"{traj.epochs[-1]}",
            fontsize=6,
            color=colors[c],
        )

    ax.set_title(f"Class embedding trajectories ({method_name})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path = output_dir / f"class_embeddings_{method_name.lower()}_trajectories.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info("Saved %s trajectory plot to %s", method_name, out_path)
    return out_path


# Plot per-epoch global metrics.
def plot_epoch_metrics(df_epoch: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}

    # Mean norm and radius
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_epoch["epoch"], df_epoch["mean_norm"], label="mean_norm")
    ax.plot(df_epoch["epoch"], df_epoch["mean_radius"], label="mean_radius")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Embedding norm and radius vs. epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path = output_dir / "epoch_norm_radius.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    paths["epoch_norm_radius"] = out_path

    # Pairwise distances and step size
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_epoch["epoch"], df_epoch["mean_pairwise_dist"], label="mean_pairwise_dist")
    ax.plot(df_epoch["epoch"], df_epoch["mean_step_norm"], label="mean_step_norm")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Pairwise distance and mean step vs. epoch")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path = output_dir / "epoch_pairwise_step.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    paths["epoch_pairwise_step"] = out_path

    # Singular value ratio
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_epoch["epoch"], df_epoch["singular_value_ratio"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max / Min singular value")
    ax.set_title("Anisotropy in embedding geometry vs. epoch")
    ax.grid(True, linestyle="--", alpha=0.3)
    out_path = output_dir / "epoch_singular_value_ratio.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    paths["epoch_singular_value_ratio"] = out_path

    logger.info("Saved epoch-level metric plots.")
    return paths


# Plot per-class step norms (movement) vs epoch.
def plot_per_class_steps(df_class: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))

    for class_id, df_sub in df_class.groupby("class_id"):
        ax.plot(
            df_sub["epoch"],
            df_sub["step_norm"],
            label=f"class {class_id}",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Step norm")
    ax.set_title("Per-class embedding displacement vs. epoch")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    out_path = output_dir / "per_class_step_norm.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info("Saved per-class step norm plot to %s", out_path)
    return out_path


# Plot PCA explained variance.
def plot_pca_explained_variance(explained: np.ndarray, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(len(explained))
    ax.bar(idx, explained)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA explained variance ratio (global)")
    ax.grid(True, linestyle="--", alpha=0.3)

    out_path = output_dir / "pca_explained_variance.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info("Saved PCA explained variance plot to %s", out_path)
    return out_path


# -------------------------------------------------------------------
# Report generation
# -------------------------------------------------------------------

# Write a simple Markdown report summarizing metrics and plot locations.
def write_markdown_report(
    cfg: AnalysisConfig,
    traj: TrajectoryData,
    df_epoch: pd.DataFrame,
    df_class: pd.DataFrame,
    plot_paths: Dict[str, Path],
    dr_plot_paths: Dict[str, Path],
    output_dir: Path,
) -> Path:
    report_path = output_dir / "class_embedding_report.md"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Class Embedding Trajectory Analysis\n\n")
        f.write(f"- Trajectory file: `{cfg.trajectory_path}`\n")
        f.write(f"- Output directory: `{output_dir}`\n")
        f.write(f"- Epochs: {int(traj.epochs[0])} → {int(traj.epochs[-1])} "
                f"(total snapshots: {len(traj.epochs)})\n")
        f.write(f"- Num classes: {traj.num_classes}\n")
        f.write(f"- Embedding dimension: {traj.emb_dim}\n\n")

        f.write("## Global metrics over epochs\n\n")
        f.write("See the following plots:\n\n")
        for key, path in plot_paths.items():
            f.write(f"- {key}: `./{path.name}`\n")

        f.write("\nSummary statistics (epoch-level):\n\n")
        f.write(df_epoch.describe().to_markdown(index=True))
        f.write("\n\n")

        f.write("## Per-class displacement metrics\n\n")
        f.write("Per-class step norms plot:\n\n")
        if "per_class_step_norm" in plot_paths:
            f.write(f"- `./{plot_paths['per_class_step_norm'].name}`\n\n")

        f.write("Per-class summary statistics:\n\n")
        f.write(df_class.groupby("class_id")[["embedding_norm", "step_norm"]]
                .describe().to_markdown())
        f.write("\n\n")

        f.write("## Dimensionality reduction and trajectories\n\n")
        for method, path in dr_plot_paths.items():
            f.write(f"- {method} trajectory: `./{path.name}`\n")

        f.write(
            "\nAll DR plots use a joint embedding of all epochs and classes "
            "so that trajectories are comparable across training.\n"
        )

    logger.info("Wrote Markdown report to %s", report_path)
    return report_path


# -------------------------------------------------------------------
# Main analysis pipeline
# -------------------------------------------------------------------

# Execute full analysis and visualization pipeline.
def run_analysis(cfg: AnalysisConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectory
    traj = load_trajectory(cfg.trajectory_path)

    # Metrics
    df_epoch = compute_epoch_metrics(traj)
    df_class = compute_per_class_metrics(traj)

    # Save metrics as CSV
    df_epoch_path = cfg.output_dir / "epoch_metrics.csv"
    df_class_path = cfg.output_dir / "per_class_metrics.csv"
    df_epoch.to_csv(df_epoch_path, index=False)
    df_class.to_csv(df_class_path, index=False)
    logger.info("Saved epoch metrics to %s", df_epoch_path)
    logger.info("Saved per-class metrics to %s", df_class_path)

    # Flatten embeddings for DR
    X, epoch_idx, class_idx = flatten_embeddings(traj)

    # PCA
    X_pca_2d, pca_explained = run_pca_2d(X, cfg.random_state)
    pca_traj_plot = plot_trajectories_2d(
        coords_2d=X_pca_2d,
        traj=traj,
        epoch_idx=epoch_idx,
        class_idx=class_idx,
        method_name="PCA",
        output_dir=cfg.output_dir,
    )
    pca_var_plot = plot_pca_explained_variance(pca_explained, cfg.output_dir)

    dr_plots: Dict[str, Path] = {"PCA": pca_traj_plot, "PCA_variance": pca_var_plot}

    # t-SNE
    X_tsne_2d = run_tsne_2d(
        X,
        random_state=cfg.random_state,
        perplexity=cfg.tsne_perplexity,
        n_iter=cfg.tsne_n_iter,
    )
    tsne_traj_plot = plot_trajectories_2d(
        coords_2d=X_tsne_2d,
        traj=traj,
        epoch_idx=epoch_idx,
        class_idx=class_idx,
        method_name="TSNE",
        output_dir=cfg.output_dir,
    )
    dr_plots["TSNE"] = tsne_traj_plot

    # UMAP (if available)
    if UMAP_AVAILABLE:
        X_umap_2d = run_umap_2d(
            X,
            random_state=cfg.random_state,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
        )
        umap_traj_plot = plot_trajectories_2d(
            coords_2d=X_umap_2d,
            traj=traj,
            epoch_idx=epoch_idx,
            class_idx=class_idx,
            method_name="UMAP",
            output_dir=cfg.output_dir,
        )
        dr_plots["UMAP"] = umap_traj_plot
    else:
        logger.warning(
            "UMAP is not installed; skipping UMAP trajectory plots. "
            "Install with 'pip install umap-learn' to enable."
        )

    # Epoch-level plots
    epoch_plot_paths = plot_epoch_metrics(df_epoch, cfg.output_dir)
    per_class_plot_path = plot_per_class_steps(df_class, cfg.output_dir)
    epoch_plot_paths["per_class_step_norm"] = per_class_plot_path

    # Report
    write_markdown_report(
        cfg=cfg,
        traj=traj,
        df_epoch=df_epoch,
        df_class=df_class,
        plot_paths=epoch_plot_paths,
        dr_plot_paths=dr_plots,
        output_dir=cfg.output_dir,
    )


# -------------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------------

def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(
        description="Analyze class embedding trajectory file."
    )
    parser.add_argument(
        "--trajectory_path",
        type=str,
        required=True,
        help="Path to .pt trajectory file produced by save_class_embeddings_trajectory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where plots, CSVs, and report will be written.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for PCA/t-SNE/UMAP.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=20.0,
        help="Perplexity parameter for t-SNE.",
    )
    parser.add_argument(
        "--tsne_n_iter",
        type=int,
        default=2000,
        help="Number of iterations for t-SNE optimization.",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=10,
        help="Number of neighbors for UMAP.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="min_dist parameter for UMAP.",
    )

    args = parser.parse_args()
    cfg = AnalysisConfig(
        trajectory_path=Path(args.trajectory_path),
        output_dir=Path(args.output_dir),
        random_state=args.random_state,
        tsne_perplexity=args.tsne_perplexity,
        tsne_n_iter=args.tsne_n_iter,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
    )
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )
    cfg = parse_args()
    run_analysis(cfg)


if __name__ == "__main__":
    main()
