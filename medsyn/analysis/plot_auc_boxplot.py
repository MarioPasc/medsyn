#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Render a single figure of per-class AUC boxplots comparing Real vs Real+Synth,
with per-class significance stars above the Real+Synth box based on the precomputed
p-values from the consolidated CSV.

Style: scienceplots with ['science', 'ieee'].
No title. x-axis: classes + overall. y-axis: AUC.

CLI:
  python plot_auc_boxplot.py \
      --data-csv /path/auc_boxplot_data.csv \
      --out-png /path/auc_boxplot.png \
      --width 8 --height 3 --dpi 300
"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Short labels for x-axis
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

# Style manager
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science'])
except Exception as e:
    # Fall back silently if style is unavailable
    logging.warning(f"scienceplots not available: {e}")


@dataclass(frozen=True)
class PlotArgs:
    data_csv: Path
    out_png: Path
    width: float
    height: float
    dpi: int
    label_spacing: float
    npz_path: Optional[Path]
    img_height_frac: float
    seed: int


# Function: metric_sort_key
# Purpose: Order metrics by classes 0..N, then 'overall'.
# Arguments:
#   label (str): value in data['label'].
# Returns:
#   tuple: sort key.
def metric_sort_key(label: str):
    if label == "overall":
        return (10**6, 0)
    try:
        return (int(label), 0)
    except ValueError:
        return (10**6 - 1, 0)


# Function: load_npz_images
# Purpose: Load images from NPZ file for thumbnail display.
# Arguments:
#   npz_path (Path): path to NPZ file containing train/val/test images and labels.
# Returns:
#   Dict with 'train', 'val', 'test' keys, each containing {'images': ndarray, 'labels': ndarray}
def load_npz_images(npz_path: Path) -> Dict:
    """Load images and labels from custom NPZ."""
    data = np.load(str(npz_path))
    result = {}
    for split in ['train', 'val', 'test']:
        imgs = data[f"{split}_images"]
        lbls = data[f"{split}_labels"].astype(np.int64).reshape(-1)
        if imgs.ndim == 3:  # [N,H,W] -> [N,H,W,1]
            imgs = imgs[..., np.newaxis]
        result[split] = {'images': imgs, 'labels': lbls}
    return result


# Function: ensure_rgb
# Purpose: Convert grayscale to RGB.
# Arguments:
#   img (np.ndarray): image array [H,W,C].
# Returns:
#   np.ndarray: RGB image [H,W,3]
def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert [H,W,1] to RGB by replication; pass [H,W,3] through."""
    if img.ndim != 3:
        raise ValueError(f"Expected [H,W,C], got {img.shape}")
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported channels: {img.shape[2]}")


# Function: pick_random_image
# Purpose: Select a random image for a given class.
# Arguments:
#   npz_data (dict): loaded NPZ data.
#   cls (int): class index.
#   rng (np.random.RandomState): random state.
# Returns:
#   np.ndarray: RGB image [H,W,3]
def pick_random_image(npz_data: Dict, cls: int, rng: np.random.RandomState) -> np.ndarray:
    """Pick a random image belonging to class cls from train, then val, then test."""
    pools = []
    for split in ['train', 'val', 'test']:
        split_data = npz_data[split]
        idx = np.where(split_data['labels'] == cls)[0]
        if idx.size:
            pools.append(split_data['images'][idx])
    if not pools:
        logging.warning(f"No images found for class {cls}. Using zero placeholder.")
        return np.zeros((28, 28, 3), dtype=np.uint8)
    pool = np.concatenate(pools, axis=0)
    img = pool[rng.randint(0, pool.shape[0])]
    return ensure_rgb(img)


# Function: prepare_wide_for_plot
# Purpose: Split the long table into {metric -> [real_values], [synth_values]}.
# Arguments:
#   df (pd.DataFrame): consolidated long dataframe.
# Returns:
#   Dict[str, Dict[str, np.ndarray]]: mapping label -> {'Real': vals, 'Real+Synth': vals}
#   Dict[str, str]: mapping label -> stars annotation for synth.
def prepare_wide_for_plot(
    df: pd.DataFrame
) -> tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, str]]:
    metrics = sorted(df["label"].unique(), key=metric_sort_key)
    data_map: Dict[str, Dict[str, np.ndarray]] = {}
    star_map: Dict[str, str] = {}

    for m in metrics:
        sub = df[df["label"] == m]
        real_vals = sub[sub["dataset"] == "Real"]["auc"].to_numpy(dtype=float)
        synth_vals = sub[sub["dataset"] == "Real+Synth"]["auc"].to_numpy(dtype=float)
        data_map[m] = {"Real": real_vals, "Real+Synth": synth_vals}
        # Grab one non-null stars string on synth rows
        synth_rows = sub[sub["dataset"] == "Real+Synth"]
        star = synth_rows["stars"].dropna().astype(str)
        star_map[m] = (star.iloc[0] if len(star) > 0 else "")
    return data_map, star_map


# Function: add_star
# Purpose: Draw a star annotation above the synth box for a metric.
# Arguments:
#   ax (plt.Axes): axes.
#   x (float): x-position of the synth box.
#   y_values (np.ndarray): auc values for synth box.
#   star (str): annotation string.
# Returns:
#   None.
def add_star(ax: plt.Axes, x: float, y_values: np.ndarray, star: str) -> None:
    if not star:
        return
    ymax = float(np.max(y_values))
    ymin = float(np.min(y_values))
    spread = max(1e-4, ymax - ymin)
    y = ymax + 0.25 * spread
    ax.text(x, y, star, ha='center', va='bottom', fontsize=9)


# Function: plot_boxes
# Purpose: Construct two boxplots per metric with x-offsets and add stars.
# Arguments:
#   ax (plt.Axes): axes.
#   data_map (dict): per-metric data.
#   star_map (dict): per-metric stars.
#   label_spacing (float): horizontal spacing between x-axis positions.
#   npz_data (Optional[Dict]): loaded NPZ data for thumbnails.
#   img_height_frac (float): height of image as fraction of axes height.
#   seed (int): random seed for image selection.
# Returns:
#   None.
def plot_boxes(ax: plt.Axes, data_map, star_map, label_spacing: float = 1.0,
               npz_data: Optional[Dict] = None, img_height_frac: float = 0.15, 
               seed: int = 42) -> None:
    metrics = list(sorted(data_map.keys(), key=metric_sort_key))
    n = len(metrics)
    x_base = np.arange(n, dtype=float) * label_spacing
    offset = 0.18
    
    rng = np.random.RandomState(seed) if npz_data else None

    # Collect plot elements per group for legend
    bp_real = None
    bp_synth = None

    for i, m in enumerate(metrics):
        vals_real = data_map[m]["Real"]
        vals_synth = data_map[m]["Real+Synth"]

        # positions
        x_real = x_base[i] - offset
        x_synth = x_base[i] + offset

        # Real
        bp_real = ax.boxplot(
            vals_real,
            positions=[x_real],
            widths=0.25,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp_real['boxes']:
            patch.set_facecolor('#009988')

        # Synth
        bp_synth = ax.boxplot(
            vals_synth,
            positions=[x_synth],
            widths=0.25,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp_synth['boxes']:
            patch.set_facecolor('#EE7733')

        # Significance stars above synth
        add_star(ax, x_synth, vals_synth, star_map.get(m, ""))

    # Axes cosmetics
    ax.set_xticks(x_base)
    # Convert metric labels to short names
    xticklabels = []
    for m in metrics:
        if m == "overall":
            xticklabels.append("Overall")
        else:
            try:
                class_idx = int(m)
                xticklabels.append(LABELS_SHORT.get(class_idx, m))
            except ValueError:
                xticklabels.append(m)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set_xlabel(r'$\mathrm{Class}$')
    ax.set_ylabel(r'$\mathrm{AUC}$')
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # Legend using custom handles from last created boxplots
    if bp_real is not None and bp_synth is not None:
        handles = [
            bp_real["boxes"][0],
            bp_synth["boxes"][0],
        ]
        labels = ["Real", "Real+Synth"]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=2)
    
    # Add thumbnail images if NPZ data provided
    if npz_data is not None and rng is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax_width_data = xlim[1] - xlim[0]
        ax_height_data = ylim[1] - ylim[0]
        
        # Image width: use a fraction of label spacing
        img_width_data = label_spacing * 0.4
        img_width_frac = img_width_data / ax_width_data
        
        # Place images just above x-axis (at y=0)
        y0_frac = 0.02  # Small offset above x-axis
        
        for i, m in enumerate(metrics):
            # Skip "overall" metric
            if m == "overall":
                continue
            
            try:
                cls = int(m)
            except ValueError:
                continue
            
            # Center position
            x_center = x_base[i]
            x_center_frac = (x_center - xlim[0]) / ax_width_data
            x_left_frac = x_center_frac - img_width_frac / 2.0
            
            # Create inset axes for image
            iax = inset_axes(
                ax,
                width=f"{img_width_frac*3000:.4f}%",
                height=f"{img_height_frac*3000:.4f}%",
                bbox_transform=ax.transAxes,
                bbox_to_anchor=(x_left_frac-0.004, y0_frac-0.08, img_width_frac, img_height_frac),
                loc="lower left",
                borderpad=0.0
            )
            iax.axis("off")
            
            # Load and display image
            img = pick_random_image(npz_data, cls, rng)
            iax.imshow(img.astype(np.float32) / 255.0)
            iax.set_xticks([])
            iax.set_yticks([])


def main(a: PlotArgs) -> None:
    logging.info("Reading consolidated CSV")
    df = pd.read_csv(a.data_csv)

    # Basic checks
    needed = {"metric", "label", "dataset", "fold", "auc"}
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in data CSV: {missing}")

    data_map, star_map = prepare_wide_for_plot(df)
    
    # Load NPZ data if provided
    npz_data = None
    if a.npz_path is not None and a.npz_path.exists():
        logging.info(f"Loading NPZ data from {a.npz_path}")
        npz_data = load_npz_images(a.npz_path)

    logging.info("Rendering figure")
    fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=a.dpi, constrained_layout=True)
    plot_boxes(ax, data_map, star_map, label_spacing=a.label_spacing, 
               npz_data=npz_data, img_height_frac=a.img_height_frac, seed=a.seed)

    # y-limits: snug fit near [min-ε, 1+ε]
    all_vals = df["auc"].to_numpy(dtype=float)
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    eps = max(1e-3, 0.03 * (y_max - y_min))
    ax.set_ylim(max(0.0, y_min - eps), min(1.0, y_max + 2*eps))

    a.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out_png, dpi=a.dpi)
    logging.info(f"Wrote figure: {a.out_png}")


def parse_args() -> PlotArgs:
    p = argparse.ArgumentParser(description="Plot per-class AUC boxplots with significance annotations.")
    p.add_argument("--data-csv", type=Path, required=True, help="Consolidated CSV created by prepare_auc_boxplot_data.py")
    p.add_argument("--out-png", type=Path, required=True, help="Output figure path (PNG).")
    p.add_argument("--npz", type=Path, default=None, help="Optional NPZ file with train/val/test images for thumbnails.")
    p.add_argument("--width", type=float, default=8.0, help="Figure width in inches.")
    p.add_argument("--height", type=float, default=3.0, help="Figure height in inches.")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    p.add_argument("--label-spacing", type=float, default=2, help="Horizontal spacing between x-axis labels (default: 2.0).")
    p.add_argument("--img-height-frac", type=float, default=0.15, help="Height of thumbnail images as fraction of axes height (default: 0.15).")
    p.add_argument("--seed", type=int, default=123, help="Random seed for image selection (default: 123).")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG).")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")
    return PlotArgs(data_csv=a.data_csv, out_png=a.out_png, width=a.width, height=a.height, 
                    dpi=a.dpi, label_spacing=a.label_spacing, npz_path=a.npz,
                    img_height_frac=a.img_height_frac, seed=a.seed)


if __name__ == "__main__":
    main(parse_args())
