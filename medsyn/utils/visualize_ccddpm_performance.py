#!/usr/bin/env python3
"""
Performance visualization script for ccDDPM model.

Creates a comprehensive performance panel showing:
- Training/validation loss curves
- Quality metrics (PSNR, SSIM) over epochs
- Per-class performance at final epoch
- Generated synthetic samples from all 9 PathMNIST classes

Usage:
    python -m medsyn.utils.visualize_ccddpm_performance --data_dir /media/mpascual/PortableSSD/medsyn/PathMNIST_ccDDPM_parallel --output performance_panel.png
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PathMNISTClass:
    """PathMNIST class names for visualization."""
    LABELS = {
        0: "Adipose",
        1: "Background",
        2: "Debris",
        3: "Lymphocytes",
        4: "Mucus",
        5: "Smooth Muscle",
        6: "Normal Colon\nMucosa",
        7: "Cancer-Assoc.\nStroma",
        8: "Colorectal Adeno.\nEpithelium"
    }

    # Short labels for bar chart
    LABELS_SHORT = {
        0: "Adipose",
        1: "Background",
        2: "Debris",
        3: "Lymphocytes",
        4: "Mucus",
        5: "Smooth\nMuscle",
        6: "Normal\nMucosa",
        7: "Cancer\nStroma",
        8: "Colorectal\nEpithelium"
    }


def load_training_metrics(csv_path: Path) -> pd.DataFrame:
    """Load training metrics CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Training metrics CSV not found: {csv_path}")

    logger.info(f"Loading training metrics from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    return df


def load_composite_class_image(data_dir: Path) -> np.ndarray:
    """
    Load the composite class visualization image showing all 9 classes.

    Args:
        data_dir: Root directory containing the composite image

    Returns:
        Composite image as numpy array, or None if not found
    """
    # Look for epoch_*_classes.png files
    logger.info(f"Searching for composite class images in {data_dir}")
    data_dir = Path(data_dir) / "samples"
    composite_files = sorted(data_dir.glob("epoch_*_classes.png"))

    if not composite_files:
        logger.warning("No composite class visualization found")
        return None

    # Use the most recent one (last in sorted order)
    composite_path = composite_files[-1]
    logger.info(f"Loading composite class image: {composite_path.name}")
    composite_path = [file for file in composite_files if "10" in file.name][0]
    img = np.array(Image.open(composite_path))
    logger.info(f"Loaded composite image with shape: {img.shape}")

    return img


def plot_loss_curves(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot training, validation, and test loss curves."""
    # Filter train, val, and test splits
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # Plot loss curves
    ax.plot(train_df['epoch'], train_df['loss'],
            label='Train Loss', linewidth=2, color='#2E86AB', marker='o', markersize=3)
    ax.plot(val_df['epoch'], val_df['loss'],
            label='Val Loss', linewidth=2, color='#A23B72', marker='s', markersize=3)
    ax.plot(test_df['epoch'], test_df['loss'],
            label='Test Loss', linewidth=2, color='#F18F01', marker='^', markersize=3)

    # Optionally add std deviation bands
    if 'loss_std' in train_df.columns:
        ax.fill_between(train_df['epoch'],
                        train_df['loss'] - train_df['loss_std'],
                        train_df['loss'] + train_df['loss_std'],
                        alpha=0.2, color='#2E86AB')

    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training, Validation & Test Loss', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)


def plot_quality_metrics(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot PSNR and SSIM metrics over epochs for validation and test sets."""
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # Create twin axis for SSIM
    ax2 = ax.twinx()

    # Plot PSNR on primary axis (Val and Test)
    line1 = ax.plot(val_df['epoch'], val_df['psnr'],
                    label='Val PSNR', linewidth=2, color='#F18F01', marker='o', markersize=3)
    line2 = ax.plot(test_df['epoch'], test_df['psnr'],
                    label='Test PSNR', linewidth=2, color='#F18F01', marker='s', markersize=3, linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontsize=11, fontweight='bold', color='#F18F01')
    ax.tick_params(axis='y', labelcolor='#F18F01')

    # Plot SSIM on secondary axis (Val and Test)
    line3 = ax2.plot(val_df['epoch'], val_df['ssim'],
                     label='Val SSIM', linewidth=2, color='#6A994E', marker='o', markersize=3)
    line4 = ax2.plot(test_df['epoch'], test_df['ssim'],
                     label='Test SSIM', linewidth=2, color='#6A994E', marker='s', markersize=3, linestyle='--', alpha=0.7)
    ax2.set_ylabel('SSIM', fontsize=11, fontweight='bold', color='#6A994E')
    ax2.tick_params(axis='y', labelcolor='#6A994E')

    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right', framealpha=0.9, fontsize=9)

    ax.set_title('Quality Metrics (Val & Test)', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)


def plot_per_class_loss(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot per-class validation and test loss at final epoch."""
    # Get final epoch
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    final_epoch = val_df['epoch'].max()

    final_val_data = val_df[val_df['epoch'] == final_epoch].iloc[0]
    final_test_data = test_df[test_df['epoch'] == final_epoch].iloc[0]

    # Extract per-class losses for val and test
    val_losses = []
    test_losses = []
    for class_id in range(9):
        col_name = f'loss_c{class_id}'
        if col_name in final_val_data:
            val_losses.append(final_val_data[col_name])
            test_losses.append(final_test_data[col_name])
        else:
            val_losses.append(np.nan)
            test_losses.append(np.nan)

    # Create grouped bar chart
    class_ids = np.arange(9)
    bar_width = 0.35
    x_pos_val = class_ids - bar_width/2
    x_pos_test = class_ids + bar_width/2

    # Create bars
    bars_val = ax.bar(x_pos_val, val_losses, bar_width,
                      label='Validation', color='#A23B72', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars_test = ax.bar(x_pos_test, test_losses, bar_width,
                       label='Test', color='#F18F01', edgecolor='black', linewidth=1.2, alpha=0.8)

    # Add value labels on bars (smaller font to fit both)
    for i, (bar_val, bar_test, loss_val, loss_test) in enumerate(zip(bars_val, bars_test, val_losses, test_losses)):
        # Val labels
        height_val = bar_val.get_height()
        ax.text(bar_val.get_x() + bar_val.get_width()/2., height_val * 1.01,
                f'{loss_val:.3f}',
                ha='center', va='bottom', fontsize=6.5, fontweight='bold')
        # Test labels
        height_test = bar_test.get_height()
        ax.text(bar_test.get_x() + bar_test.get_width()/2., height_test * 1.01,
                f'{loss_test:.3f}',
                ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.set_xlabel('Class', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title(f'Per-Class Loss - Val vs Test (Epoch {final_epoch})', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(class_ids)
    ax.set_xticklabels([PathMNISTClass.LABELS_SHORT.get(i, str(i)) for i in class_ids],
                       fontsize=7.5, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    # Add some top margin for labels
    max_loss = max(max(val_losses), max(test_losses))
    ax.set_ylim(top=max_loss * 1.15)


def plot_synthetic_samples(fig: plt.Figure, gs: gridspec.GridSpec,
                           composite_image: np.ndarray,
                           start_row: int = 2) -> None:
    """
    Plot composite class visualization showing all 9 classes.

    Args:
        fig: Figure object
        gs: GridSpec object
        composite_image: Composite image showing all classes
        start_row: Starting row in GridSpec for the samples
    """
    # Create axis for the composite image
    ax = fig.add_subplot(gs[start_row, :])

    if composite_image is not None:
        ax.imshow(composite_image)
        ax.axis('off')

        # Add subtle border
        for spine in ax.spines.values():
            spine.set_edgecolor('#2E86AB')
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # Add class labels below the image
        # Assuming 9 classes evenly distributed
        img_width = composite_image.shape[1]
        class_width = img_width / 9

        for class_id in range(9):
            x_pos = (class_id + 0.5) * class_width
            class_name = PathMNISTClass.LABELS_SHORT.get(class_id, f"Class {class_id}").replace('\n', ' ')
            ax.text(x_pos, composite_image.shape[0] + 15,
                   class_name,
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   color='#2E86AB')

    else:
        ax.text(0.5, 0.5, 'Composite class visualization not available',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='#D32F2F', fontweight='bold')
        ax.axis('off')


def create_performance_panel(data_dir: Path, output_path: Path) -> None:
    """
    Create comprehensive performance panel visualization.

    Args:
        data_dir: Directory containing training metrics and synthetic images
        output_path: Path to save the output visualization
    """
    logger.info("Creating performance panel...")

    # Load data
    metrics_csv = data_dir / "training_metrics.csv"
    df = load_training_metrics(metrics_csv)
    composite_image = load_composite_class_image(data_dir)

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[0.75, 0.01, 1],
                          hspace=0.14, wspace=0.3, top=0.95, bottom=0.02,
                          left=0.06, right=0.96)

    # Top row: Loss curves, Quality metrics, Per-class loss
    ax1 = fig.add_subplot(gs[0, 0])
    plot_loss_curves(ax1, df)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_quality_metrics(ax2, df)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_per_class_loss(ax3, df)

    # Bottom section: Generated synthetic samples (pushed lower)
    plot_synthetic_samples(fig, gs, composite_image, start_row=2)

    # Add section title for synthetic samples (adjusted position)
    fig.text(0.5, 0.42, 'Generated Synthetic Samples - All 9 PathMNIST Classes',
            fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#2E86AB', linewidth=2))

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved performance panel to {output_path}")

    # Also display
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create performance panel for ccDDPM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='docs/performance',
        help='Directory containing training metrics and synthetic images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='docs/performance/performance_panel.png',
        help='Output path for the performance panel visualization'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    create_performance_panel(data_dir, output_path)
    logger.info("Done!")


if __name__ == '__main__':
    main()
