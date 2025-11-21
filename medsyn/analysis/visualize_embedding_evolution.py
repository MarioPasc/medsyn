#!/usr/bin/env python3
"""
Visualization and Analysis of Class Embedding Evolution Over Time

This script analyzes how class embeddings evolve across training epochs,
with temporal analysis as a core dimension.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import argparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_embeddings(file_path):
    """Load embedding trajectory data."""
    data = torch.load(file_path, map_location='cpu')
    return {
        'epochs': data['epochs'],
        'embeddings': data['embeddings'].numpy(),  # [num_epochs, num_classes, emb_dim]
        'num_classes': data['num_classes'],
        'emb_dim': data['emb_dim']
    }


def compute_embedding_norms(embeddings):
    """Compute L2 norms of embeddings over time.

    Args:
        embeddings: [num_epochs, num_classes, emb_dim]
    Returns:
        norms: [num_epochs, num_classes]
    """
    return np.linalg.norm(embeddings, axis=2)


def compute_pairwise_distances(embeddings):
    """Compute pairwise distances between classes at each epoch.

    Args:
        embeddings: [num_epochs, num_classes, emb_dim]
    Returns:
        distances: [num_epochs, num_classes, num_classes]
    """
    num_epochs, num_classes, _ = embeddings.shape
    distances = np.zeros((num_epochs, num_classes, num_classes))

    for epoch_idx in range(num_epochs):
        distances[epoch_idx] = squareform(pdist(embeddings[epoch_idx], metric='euclidean'))

    return distances


def compute_embedding_variance(embeddings):
    """Compute variance across classes at each epoch.

    Args:
        embeddings: [num_epochs, num_classes, emb_dim]
    Returns:
        variance_per_dim: [num_epochs, emb_dim]
        total_variance: [num_epochs]
    """
    variance_per_dim = np.var(embeddings, axis=1)  # Variance across classes
    total_variance = np.mean(variance_per_dim, axis=1)
    return variance_per_dim, total_variance


def compute_trajectory_velocity(embeddings):
    """Compute how fast embeddings are changing (velocity).

    Args:
        embeddings: [num_epochs, num_classes, emb_dim]
    Returns:
        velocities: [num_epochs-1, num_classes] - L2 distance between consecutive epochs
    """
    diff = np.diff(embeddings, axis=0)  # [num_epochs-1, num_classes, emb_dim]
    velocities = np.linalg.norm(diff, axis=2)  # [num_epochs-1, num_classes]
    return velocities


def compute_centroid_trajectory(embeddings):
    """Compute the centroid (mean) embedding across all classes over time.

    Args:
        embeddings: [num_epochs, num_classes, emb_dim]
    Returns:
        centroids: [num_epochs, emb_dim]
    """
    return np.mean(embeddings, axis=1)


def plot_embedding_norms(epochs, norms, num_classes, save_path):
    """Plot L2 norms of embeddings over time."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for class_idx in range(num_classes):
        ax.plot(epochs, norms[:, class_idx], marker='o', label=f'Class {class_idx}', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm', fontsize=12, fontweight='bold')
    ax.set_title('Embedding L2 Norms Evolution Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_trajectory_velocity(epochs, velocities, num_classes, save_path):
    """Plot embedding change velocity over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Individual class velocities
    for class_idx in range(num_classes):
        ax1.plot(epochs[:-1], velocities[:, class_idx], marker='o',
                label=f'Class {class_idx}', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Embedding Change (L2 Distance)', fontsize=12, fontweight='bold')
    ax1.set_title('Embedding Velocity: Per-Class', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Average velocity across all classes
    mean_velocity = np.mean(velocities, axis=1)
    std_velocity = np.std(velocities, axis=1)

    ax2.plot(epochs[:-1], mean_velocity, marker='o', linewidth=2, color='navy', label='Mean')
    ax2.fill_between(epochs[:-1], mean_velocity - std_velocity, mean_velocity + std_velocity,
                     alpha=0.3, color='navy', label='±1 STD')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Embedding Change', fontsize=12, fontweight='bold')
    ax2.set_title('Embedding Velocity: Average Across Classes', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_variance_evolution(epochs, variance_per_dim, total_variance, save_path):
    """Plot variance evolution over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Total variance over time
    ax1.plot(epochs, total_variance, marker='o', linewidth=2, color='crimson')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Variance', fontsize=12, fontweight='bold')
    ax1.set_title('Total Embedding Variance Across Classes Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Variance heatmap per dimension over time
    im = ax2.imshow(variance_per_dim.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax2.set_xlabel('Epoch Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Embedding Dimension', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Dimension Variance Over Time', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(epochs)))
    ax2.set_xticklabels(epochs)
    plt.colorbar(im, ax=ax2, label='Variance')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pairwise_distances_evolution(epochs, distances, num_classes, save_path):
    """Plot evolution of pairwise distances between classes."""
    num_epochs = len(epochs)

    # Create subplots for selected epochs
    selected_epochs = [0, num_epochs//3, 2*num_epochs//3, num_epochs-1]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    vmin = distances.min()
    vmax = distances.max()

    for idx, epoch_idx in enumerate(selected_epochs):
        im = axes[idx].imshow(distances[epoch_idx], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'Epoch {epochs[epoch_idx]}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Class', fontsize=10)
        axes[idx].set_ylabel('Class', fontsize=10)

        # Add colorbar
        plt.colorbar(im, ax=axes[idx], label='Euclidean Distance')

        # Add values
        for i in range(num_classes):
            for j in range(num_classes):
                text = axes[idx].text(j, i, f'{distances[epoch_idx, i, j]:.1f}',
                                    ha="center", va="center", color="white", fontsize=8)

    fig.suptitle('Pairwise Class Distances Evolution', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_mean_interclass_distance(epochs, distances, save_path):
    """Plot mean inter-class distance over time."""
    num_epochs, num_classes, _ = distances.shape

    # Compute mean distance (excluding diagonal)
    mean_distances = np.zeros(num_epochs)
    for epoch_idx in range(num_epochs):
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(num_classes, k=1)
        mean_distances[epoch_idx] = np.mean(distances[epoch_idx][triu_indices])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, mean_distances, marker='o', linewidth=2, color='darkgreen')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Inter-Class Distance', fontsize=12, fontweight='bold')
    ax.set_title('Average Separation Between Classes Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pca_trajectories(embeddings, epochs, num_classes, save_path):
    """Project embeddings to 2D using PCA and plot trajectories over time."""
    num_epochs = embeddings.shape[0]

    # Reshape for PCA: [num_epochs * num_classes, emb_dim]
    reshaped = embeddings.reshape(-1, embeddings.shape[2])

    # Apply PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(reshaped)

    # Reshape back: [num_epochs, num_classes, 2]
    projected = projected.reshape(num_epochs, num_classes, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for class_idx in range(num_classes):
        trajectory = projected[:, class_idx, :]

        # Plot trajectory line
        ax.plot(trajectory[:, 0], trajectory[:, 1],
               alpha=0.6, linewidth=2, color=colors[class_idx])

        # Plot points with color gradient for time
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1],
                           c=epochs, cmap='coolwarm',
                           s=100, edgecolors='black', linewidth=1.5,
                           alpha=0.8, zorder=5)

        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                  marker='s', s=200, color=colors[class_idx],
                  edgecolors='black', linewidth=2, label=f'Class {class_idx} start', zorder=10)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                  marker='*', s=300, color=colors[class_idx],
                  edgecolors='black', linewidth=2, zorder=10)

        # Annotate class at final position
        ax.annotate(f'C{class_idx}',
                   xy=(trajectory[-1, 0], trajectory[-1, 1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[class_idx], alpha=0.7))

    # Colorbar for time
    cbar = plt.colorbar(scatter, ax=ax, label='Epoch')
    cbar.set_label('Epoch', fontsize=12, fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Class Embedding Trajectories in PCA Space\n(Square=Start, Star=End)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return pca.explained_variance_ratio_


def plot_tsne_trajectories(embeddings, epochs, num_classes, save_path):
    """Project embeddings to 2D using t-SNE and plot trajectories."""
    num_epochs = embeddings.shape[0]

    # Reshape for t-SNE: [num_epochs * num_classes, emb_dim]
    reshaped = embeddings.reshape(-1, embeddings.shape[2])

    # Apply t-SNE
    print("Computing t-SNE projection (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, reshaped.shape[0]-1))
    projected = tsne.fit_transform(reshaped)

    # Reshape back: [num_epochs, num_classes, 2]
    projected = projected.reshape(num_epochs, num_classes, 2)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for class_idx in range(num_classes):
        trajectory = projected[:, class_idx, :]

        # Plot trajectory line
        ax.plot(trajectory[:, 0], trajectory[:, 1],
               alpha=0.6, linewidth=2, color=colors[class_idx])

        # Plot points with color gradient for time
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1],
                           c=epochs, cmap='coolwarm',
                           s=100, edgecolors='black', linewidth=1.5,
                           alpha=0.8, zorder=5)

        # Mark start and end
        ax.scatter(trajectory[0, 0], trajectory[0, 1],
                  marker='s', s=200, color=colors[class_idx],
                  edgecolors='black', linewidth=2, zorder=10)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                  marker='*', s=300, color=colors[class_idx],
                  edgecolors='black', linewidth=2, zorder=10)

        # Annotate class at final position
        ax.annotate(f'C{class_idx}',
                   xy=(trajectory[-1, 0], trajectory[-1, 1]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[class_idx], alpha=0.7))

    # Colorbar for time
    cbar = plt.colorbar(scatter, ax=ax, label='Epoch')
    cbar.set_label('Epoch', fontsize=12, fontweight='bold')

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title('Class Embedding Trajectories in t-SNE Space\n(Square=Start, Star=End)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_centroid_movement(centroids, epochs, save_path):
    """Plot how the centroid of all class embeddings moves over time."""
    # Apply PCA to visualize centroid movement
    pca = PCA(n_components=2)
    projected = pca.fit_transform(centroids)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot trajectory
    ax.plot(projected[:, 0], projected[:, 1],
           linewidth=2, color='purple', alpha=0.7, marker='o', markersize=8)

    # Color points by epoch
    scatter = ax.scatter(projected[:, 0], projected[:, 1],
                        c=epochs, cmap='viridis',
                        s=150, edgecolors='black', linewidth=2, zorder=5)

    # Annotate epochs
    for i, epoch in enumerate(epochs):
        ax.annotate(f'{epoch}',
                   xy=(projected[i, 0], projected[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')

    # Mark start and end
    ax.scatter(projected[0, 0], projected[0, 1],
              marker='s', s=300, color='green',
              edgecolors='black', linewidth=2, label='Start', zorder=10)
    ax.scatter(projected[-1, 0], projected[-1, 1],
              marker='*', s=400, color='red',
              edgecolors='black', linewidth=2, label='End', zorder=10)

    plt.colorbar(scatter, ax=ax, label='Epoch')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)',
                 fontsize=12, fontweight='bold')
    ax.set_title('Centroid Movement Over Time (PCA Projection)',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_report(data, output_dir):
    """Generate a text summary of key findings."""
    embeddings = data['embeddings']
    epochs = data['epochs']
    num_classes = data['num_classes']

    # Compute statistics
    norms = compute_embedding_norms(embeddings)
    velocities = compute_trajectory_velocity(embeddings)
    distances = compute_pairwise_distances(embeddings)
    _, total_variance = compute_embedding_variance(embeddings)

    report_path = output_dir / 'analysis_summary.txt'

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EMBEDDING EVOLUTION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Dataset Information:\n")
        f.write(f"  - Number of epochs: {len(epochs)}\n")
        f.write(f"  - Epoch range: {epochs[0]} to {epochs[-1]}\n")
        f.write(f"  - Number of classes: {num_classes}\n")
        f.write(f"  - Embedding dimension: {data['emb_dim']}\n\n")

        f.write("-"*80 + "\n")
        f.write("EMBEDDING NORMS (L2)\n")
        f.write("-"*80 + "\n")
        for class_idx in range(num_classes):
            initial_norm = norms[0, class_idx]
            final_norm = norms[-1, class_idx]
            change = final_norm - initial_norm
            f.write(f"  Class {class_idx}: {initial_norm:.3f} → {final_norm:.3f} "
                   f"(change: {change:+.3f}, {change/initial_norm*100:+.1f}%)\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("EMBEDDING VELOCITY (Average Change Per Epoch)\n")
        f.write("-"*80 + "\n")
        for class_idx in range(num_classes):
            avg_velocity = np.mean(velocities[:, class_idx])
            max_velocity = np.max(velocities[:, class_idx])
            min_velocity = np.min(velocities[:, class_idx])
            f.write(f"  Class {class_idx}: avg={avg_velocity:.3f}, "
                   f"max={max_velocity:.3f}, min={min_velocity:.3f}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("INTER-CLASS DISTANCES\n")
        f.write("-"*80 + "\n")

        # Mean inter-class distance over time
        mean_distances = []
        for epoch_idx in range(len(epochs)):
            triu_indices = np.triu_indices(num_classes, k=1)
            mean_dist = np.mean(distances[epoch_idx][triu_indices])
            mean_distances.append(mean_dist)

        f.write(f"  Initial mean distance (epoch {epochs[0]}): {mean_distances[0]:.3f}\n")
        f.write(f"  Final mean distance (epoch {epochs[-1]}): {mean_distances[-1]:.3f}\n")
        f.write(f"  Change: {mean_distances[-1] - mean_distances[0]:+.3f}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("EMBEDDING SPACE VARIANCE\n")
        f.write("-"*80 + "\n")
        f.write(f"  Initial variance (epoch {epochs[0]}): {total_variance[0]:.3f}\n")
        f.write(f"  Final variance (epoch {epochs[-1]}): {total_variance[-1]:.3f}\n")
        f.write(f"  Change: {total_variance[-1] - total_variance[0]:+.3f}\n")

        f.write("\n" + "-"*80 + "\n")
        f.write("TRAJECTORY ANALYSIS\n")
        f.write("-"*80 + "\n")

        # Compute total distance traveled by each class
        for class_idx in range(num_classes):
            total_distance = np.sum(velocities[:, class_idx])
            f.write(f"  Class {class_idx} total distance traveled: {total_distance:.3f}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Saved: {report_path}")

    # Print to console as well
    with open(report_path, 'r') as f:
        print("\n" + f.read())


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and analyze embedding evolution over training epochs'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/media/mpascual/PortableSSD/medsyn/PathMNIST_ccDDPM_parallel/embeddings/class_embeddings_trajectory.pt',
        help='Path to the embeddings .pt file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./embedding_analysis',
        help='Directory to save output visualizations'
    )
    parser.add_argument(
        '--skip-tsne',
        action='store_true',
        help='Skip t-SNE visualization (can be slow)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EMBEDDING EVOLUTION VISUALIZATION AND ANALYSIS")
    print("="*80)
    print(f"\nLoading embeddings from: {args.input}")

    # Load data
    data = load_embeddings(args.input)
    embeddings = data['embeddings']
    epochs = data['epochs']
    num_classes = data['num_classes']

    print(f"Loaded: {len(epochs)} epochs, {num_classes} classes, {data['emb_dim']}-dim embeddings")
    print(f"Output directory: {output_dir}")
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    # 1. Embedding norms over time
    print("[1/10] Computing and plotting embedding norms...")
    norms = compute_embedding_norms(embeddings)
    plot_embedding_norms(epochs, norms, num_classes, output_dir / 'embedding_norms.png')

    # 2. Trajectory velocities
    print("[2/10] Computing and plotting trajectory velocities...")
    velocities = compute_trajectory_velocity(embeddings)
    plot_trajectory_velocity(epochs, velocities, num_classes, output_dir / 'trajectory_velocity.png')

    # 3. Variance evolution
    print("[3/10] Computing and plotting variance evolution...")
    variance_per_dim, total_variance = compute_embedding_variance(embeddings)
    plot_variance_evolution(epochs, variance_per_dim, total_variance, output_dir / 'variance_evolution.png')

    # 4. Pairwise distances
    print("[4/10] Computing pairwise distances...")
    distances = compute_pairwise_distances(embeddings)

    print("[5/10] Plotting pairwise distance evolution...")
    plot_pairwise_distances_evolution(epochs, distances, num_classes,
                                     output_dir / 'pairwise_distances.png')

    print("[6/10] Plotting mean inter-class distance...")
    plot_mean_interclass_distance(epochs, distances, output_dir / 'mean_interclass_distance.png')

    # 5. PCA trajectories
    print("[7/10] Computing PCA trajectories...")
    variance_ratio = plot_pca_trajectories(embeddings, epochs, num_classes,
                                          output_dir / 'pca_trajectories.png')
    print(f"         PCA explained variance: PC1={variance_ratio[0]*100:.1f}%, PC2={variance_ratio[1]*100:.1f}%")

    # 6. t-SNE trajectories (optional, can be slow)
    if not args.skip_tsne:
        print("[8/10] Computing t-SNE trajectories...")
        plot_tsne_trajectories(embeddings, epochs, num_classes,
                              output_dir / 'tsne_trajectories.png')
    else:
        print("[8/10] Skipping t-SNE trajectories (--skip-tsne flag set)")

    # 7. Centroid movement
    print("[9/10] Computing and plotting centroid movement...")
    centroids = compute_centroid_trajectory(embeddings)
    plot_centroid_movement(centroids, epochs, output_dir / 'centroid_movement.png')

    # 8. Generate summary report
    print("[10/10] Generating summary report...")
    generate_summary_report(data, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")
    print()


if __name__ == '__main__':
    main()
