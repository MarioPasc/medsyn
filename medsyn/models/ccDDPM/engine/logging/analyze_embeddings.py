"""
Analysis and visualization utilities for enhanced embeddings data.

This script demonstrates how to load and visualize the data collected
by the enhanced embeddings logging system.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

# Optional imports
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some visualizations disabled.")

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


class EmbeddingsAnalyzer:
    """Helper class for analyzing saved embeddings data."""

    def __init__(self, output_dir: Path):
        """
        Initialize analyzer with path to embeddings output directory.

        Args:
            output_dir: Path where embeddings were saved
        """
        self.output_dir = Path(output_dir)

        # Load metadata
        meta_path = self.output_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
            print(f"Warning: No metadata found at {meta_path}")

        # Extract class names
        if self.metadata and "dataset" in self.metadata:
            self.class_names = self.metadata["dataset"].get("class_names", None)
            self.num_classes = self.metadata["dataset"].get("num_classes", None)
        else:
            self.class_names = None
            self.num_classes = None

    def load_class_embeddings(self) -> Dict:
        """Load class embedding trajectory.

        Returns:
            Dictionary with epochs, embeddings, num_classes, emb_dim
        """
        path = self.output_dir / "class_embeddings_trajectory.pt"
        if not path.exists():
            raise FileNotFoundError(f"Class embeddings not found at {path}")

        return torch.load(path, map_location="cpu")

    def load_probe_features(self, epoch: int) -> Dict[str, np.ndarray]:
        """Load probe features for a specific epoch.

        Args:
            epoch: Epoch number

        Returns:
            Dictionary with features, sample_ids, class_ids, etc.
        """
        path = self.output_dir / f"probe_features_epoch_{epoch:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Probe features not found at {path}")

        return dict(np.load(path))

    def load_clustering_metrics(self) -> Dict:
        """Load clustering metrics trajectory.

        Returns:
            Dictionary with epochs, scores, cluster_centers, assignments
        """
        path = self.output_dir / "clustering_metrics.pt"
        if not path.exists():
            raise FileNotFoundError(f"Clustering metrics not found at {path}")

        return torch.load(path, map_location="cpu")

    def get_available_probe_epochs(self) -> List[int]:
        """Get list of epochs with saved probe features.

        Returns:
            List of epoch numbers
        """
        probe_files = sorted(self.output_dir.glob("probe_features_epoch_*.npz"))
        epochs = []
        for f in probe_files:
            # Extract epoch from filename
            epoch_str = f.stem.split("_")[-1]
            epochs.append(int(epoch_str))
        return epochs

    def filter_probe_features(
        self,
        probe_data: Dict[str, np.ndarray],
        branch: Optional[str] = "cond",
        timestep: Optional[int] = None,
        layer_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter probe features by conditions.

        Args:
            probe_data: Data from load_probe_features()
            branch: "cond" or "uncond" (None = both)
            timestep: Specific timestep (None = all)
            layer_name: Specific layer (None = all)

        Returns:
            (features, class_ids) arrays after filtering
        """
        mask = np.ones(len(probe_data["features"]), dtype=bool)

        if branch is not None:
            mask &= (probe_data["branches"] == branch)

        if timestep is not None:
            mask &= (probe_data["timesteps"] == timestep)

        if layer_name is not None:
            mask &= (probe_data["layer_names"] == layer_name)

        features = probe_data["features"][mask]
        class_ids = probe_data["class_ids"][mask]

        return features, class_ids

    def plot_class_embedding_trajectory(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None
    ):
        """Plot how class embeddings evolve over training.

        Creates two plots:
        1. Embedding norm per class over epochs
        2. PCA visualization of embedding space

        Args:
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn required for this visualization")
            return

        data = self.load_class_embeddings()
        epochs = data["epochs"]
        embeddings = data["embeddings"]  # [E, C, D]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Embedding norms
        for c in range(embeddings.shape[1]):
            norms = torch.norm(embeddings[:, c, :], dim=1).numpy()
            label = self.class_names[c] if self.class_names else f"Class {c}"
            axes[0].plot(epochs, norms, label=label, marker='o', markersize=3)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Embedding Norm")
        axes[0].set_title("Class Embedding Norms Over Training")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: PCA visualization (final epoch)
        final_emb = embeddings[-1].numpy()  # [C, D]
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(final_emb)

        for c in range(len(final_emb)):
            label = self.class_names[c] if self.class_names else f"Class {c}"
            axes[1].scatter(emb_2d[c, 0], emb_2d[c, 1],
                          s=100, label=label, alpha=0.7)
            axes[1].annotate(str(c), (emb_2d[c, 0], emb_2d[c, 1]),
                           fontsize=10, ha='center', va='center')

        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axes[1].set_title(f"Class Embedding Space (Epoch {epochs[-1]})")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_latent_space(
        self,
        epoch: int,
        timestep: int = 500,
        branch: str = "cond",
        method: str = "tsne",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ):
        """Plot latent space using dimensionality reduction.

        Args:
            epoch: Which epoch to visualize
            timestep: Which diffusion timestep
            branch: "cond" or "uncond"
            method: "tsne" or "pca"
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn required for this visualization")
            return

        # Load data
        probe_data = self.load_probe_features(epoch)
        features, class_ids = self.filter_probe_features(
            probe_data, branch=branch, timestep=timestep
        )

        # Dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//2))
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")

        X_2d = reducer.fit_transform(features)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        for c in range(self.num_classes or 10):
            mask = (class_ids == c)
            if not mask.any():
                continue

            label = self.class_names[c] if self.class_names else f"Class {c}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      label=label, alpha=0.6, s=50)

        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Latent Space at t={timestep}, Epoch {epoch} ({branch} branch)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_clustering_metrics(
        self,
        figsize: Tuple[int, int] = (15, 4),
        save_path: Optional[Path] = None
    ):
        """Plot clustering quality metrics over training.

        Args:
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        data = self.load_clustering_metrics()
        epochs = data["epochs"]
        scores = data["scores"]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Extract metrics
        silhouette = [s["silhouette"] for s in scores]
        ch = [s["calinski_harabasz"] for s in scores]
        db = [s["davies_bouldin"] for s in scores]

        # Plot 1: Silhouette
        axes[0].plot(epochs, silhouette, marker='o', linewidth=2)
        axes[0].axhline(0.5, color='r', linestyle='--', alpha=0.3, label='Good threshold')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Silhouette Score")
        axes[0].set_title("Cluster Separation\n(higher = better)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Calinski-Harabasz
        axes[1].plot(epochs, ch, marker='o', linewidth=2, color='green')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Calinski-Harabasz Index")
        axes[1].set_title("Cluster Density\n(higher = better)")
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Davies-Bouldin
        axes[2].plot(epochs, db, marker='o', linewidth=2, color='orange')
        axes[2].axhline(1.0, color='r', linestyle='--', alpha=0.3, label='Acceptable threshold')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Davies-Bouldin Index")
        axes[2].set_title("Cluster Compactness\n(lower = better)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_timestep_comparison(
        self,
        epoch: int,
        timesteps: List[int],
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[Path] = None
    ):
        """Compare latent space across different diffusion timesteps.

        Args:
            epoch: Which epoch to visualize
            timesteps: List of timesteps to compare
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn required for this visualization")
            return

        probe_data = self.load_probe_features(epoch)

        n_plots = len(timesteps)
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        for i, t in enumerate(timesteps):
            features, class_ids = self.filter_probe_features(
                probe_data, branch="cond", timestep=t
            )

            # PCA for speed
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(features)

            for c in range(self.num_classes or 10):
                mask = (class_ids == c)
                if not mask.any():
                    continue

                label = self.class_names[c] if self.class_names else f"C{c}"
                axes[i].scatter(X_2d[mask, 0], X_2d[mask, 1],
                              label=label if i == 0 else None,
                              alpha=0.6, s=30)

            axes[i].set_title(f"t={t}")
            axes[i].set_xlabel("PC1")
            if i == 0:
                axes[i].set_ylabel("PC2")
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[i].grid(True, alpha=0.3)

        fig.suptitle(f"Latent Space Evolution Across Diffusion Timesteps (Epoch {epoch})", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def plot_cfg_comparison(
        self,
        epoch: int,
        timestep: int = 500,
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[Path] = None
    ):
        """Compare conditional vs unconditional branches (CFG analysis).

        Args:
            epoch: Which epoch to visualize
            timestep: Which diffusion timestep
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn required for this visualization")
            return

        probe_data = self.load_probe_features(epoch)

        # Get both branches
        feat_cond, class_cond = self.filter_probe_features(
            probe_data, branch="cond", timestep=timestep
        )
        feat_uncond, class_uncond = self.filter_probe_features(
            probe_data, branch="uncond", timestep=timestep
        )

        # Combine for joint PCA
        features_all = np.vstack([feat_cond, feat_uncond])
        pca = PCA(n_components=2)
        X_2d_all = pca.fit_transform(features_all)

        cond_2d = X_2d_all[:len(feat_cond)]
        uncond_2d = X_2d_all[len(feat_cond):]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Conditional
        for c in range(self.num_classes or 10):
            mask = (class_cond == c)
            if not mask.any():
                continue
            label = self.class_names[c] if self.class_names else f"C{c}"
            axes[0].scatter(cond_2d[mask, 0], cond_2d[mask, 1],
                          label=label, alpha=0.6, s=50)

        axes[0].set_title("Conditional Branch")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Unconditional
        for c in range(self.num_classes or 10):
            mask = (class_uncond == c)
            if not mask.any():
                continue
            label = self.class_names[c] if self.class_names else f"C{c}"
            axes[1].scatter(uncond_2d[mask, 0], uncond_2d[mask, 1],
                          label=label, alpha=0.6, s=50)

        axes[1].set_title("Unconditional Branch")
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"CFG Branch Comparison at t={timestep}, Epoch {epoch}", y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        plt.show()

    def generate_full_report(self, save_dir: Optional[Path] = None):
        """Generate complete analysis report with all visualizations.

        Args:
            save_dir: Directory to save figures (optional)
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("GENERATING FULL EMBEDDINGS ANALYSIS REPORT")
        print("=" * 80)

        # 1. Class embedding trajectory
        print("\n[1/5] Class embedding trajectory...")
        save_path = save_dir / "class_embeddings.png" if save_dir else None
        self.plot_class_embedding_trajectory(save_path=save_path)

        # 2. Clustering metrics
        print("\n[2/5] Clustering quality metrics...")
        save_path = save_dir / "clustering_metrics.png" if save_dir else None
        try:
            self.plot_clustering_metrics(save_path=save_path)
        except FileNotFoundError:
            print("  (Clustering metrics not available)")

        # 3. Latest latent space
        print("\n[3/5] Latest latent space visualization...")
        available_epochs = self.get_available_probe_epochs()
        if available_epochs:
            latest_epoch = max(available_epochs)
            save_path = save_dir / "latent_space.png" if save_dir else None
            self.plot_latent_space(latest_epoch, save_path=save_path)
        else:
            print("  (No probe features available)")

        # 4. Timestep comparison
        print("\n[4/5] Timestep comparison...")
        if available_epochs:
            save_path = save_dir / "timestep_comparison.png" if save_dir else None
            self.plot_timestep_comparison(
                latest_epoch,
                timesteps=[100, 500, 900],
                save_path=save_path
            )

        # 5. CFG comparison
        print("\n[5/5] CFG branch comparison...")
        if available_epochs:
            save_path = save_dir / "cfg_comparison.png" if save_dir else None
            try:
                self.plot_cfg_comparison(latest_epoch, save_path=save_path)
            except Exception as e:
                print(f"  (CFG comparison failed: {e})")

        print("\n" + "=" * 80)
        print("REPORT GENERATION COMPLETE")
        print("=" * 80)


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze enhanced embeddings data")
    parser.add_argument("embeddings_dir", type=str, help="Path to embeddings output directory")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--save-dir", type=str, help="Directory to save figures")

    args = parser.parse_args()

    analyzer = EmbeddingsAnalyzer(Path(args.embeddings_dir))

    save_dir = Path(args.save_dir) if args.save_dir else None

    if args.report:
        analyzer.generate_full_report(save_dir=save_dir)
    else:
        # Interactive mode - show individual plots
        print("\nAvailable probe epochs:", analyzer.get_available_probe_epochs())
        print("\nGenerating sample visualizations...")

        # Class embeddings
        analyzer.plot_class_embedding_trajectory()

        # Clustering (if available)
        try:
            analyzer.plot_clustering_metrics()
        except FileNotFoundError:
            pass

        # Latent space (if probe data available)
        available_epochs = analyzer.get_available_probe_epochs()
        if available_epochs:
            latest = max(available_epochs)
            analyzer.plot_latent_space(latest)
