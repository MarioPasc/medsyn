#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked class-distribution barplot with centered thumbnails from a custom NPZ.

Input NPZ keys (required):
  train_images [N,H,W,C] uint8, train_labels [N] int
  val_images   [N,H,W,C] uint8, val_labels   [N] int
  test_images  [N,H,W,C] uint8, test_labels  [N] int

Figure:
  - X: class indices
  - Y: total images per class (train+val+test)
  - Stacked bars with Train/Validation/Test using fixed colors
  - Bars ordered by total samples (desc)
  - Thumbnail image centered above each bar; class name centered above image

Usage:
  python plot_npz_bars.py --npz path/to/data.npz --out fig.png
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
# Style: scienceplots with IEEE
import scienceplots  # noqa: F401  # import activates style availability
plt.style.use(['science', 'ieee'])

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------- Config -----------------------------------------

# Thumbnail sizing controls
IMG_HEIGHT_FRAC: float = 0.28   # fraction of the axes height reserved for each image box within the headroom
IMG_WIDTH_SCALE: float = 1.6    # multiple of the bar width for image width
HEADROOM_FRAC: float = 0.28     # fraction of max(total) reserved above bars for images + padding
IMAGE_GAP_FRAC: float = 0.04    # vertical gap fraction (of max total) between bar top and image box
IMG_X_OFFSET: float = 5.0       # horizontal offset for images and class names (in bar-width units)

# Colors (Train / Val / Test)
COLOR_TRAIN = '#004488'
COLOR_VAL   = '#DDAA33'
COLOR_TEST  = '#BB5566'

# ----------------------------- Logging ----------------------------------------

logger = logging.getLogger("npz_barplot")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --------------------------- Data Structures ----------------------------------

@dataclass(frozen=True)
class SplitArrays:
    """Images and labels of a split."""
    images: np.ndarray  # [N,H,W,C] uint8
    labels: np.ndarray  # [N] int64


@dataclass(frozen=True)
class DatasetNPZ:
    """Three splits and class names."""
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    class_names: Dict[int, str]


# ------------------------------ I/O -------------------------------------------

def load_custom_npz(npz_path: Path,
                    class_names: Optional[Dict[int, str]] = None) -> DatasetNPZ:
    """
    Load custom NPZ with train/val/test images+labels.

    Raises:
        FileNotFoundError, KeyError, ValueError
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(str(npz_path))
    required = ["train_images", "train_labels",
                "val_images", "val_labels",
                "test_images", "test_labels"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}")

    def _split(name: str) -> SplitArrays:
        imgs = data[f"{name}_images"]
        lbls = data[f"{name}_labels"].astype(np.int64).reshape(-1)
        if imgs.ndim == 3:  # [N,H,W] -> [N,H,W,1]
            imgs = imgs[..., np.newaxis]
        if imgs.ndim != 4 or imgs.shape[-1] not in (1, 3):
            raise ValueError(f"Unexpected {name}_images shape: {imgs.shape}")
        return SplitArrays(images=imgs, labels=lbls)

    default_names = {
        0: "Adipose", 1: "Background", 2: "Debris", 3: "Lymphocytes",
        4: "Mucus", 5: "Normal mucosa", 6: "CA stroma",
        7: "CRC epithelium", 8: "Smooth muscle",
    }
    names = default_names if class_names is None else class_names

    ds = DatasetNPZ(train=_split("train"),
                    val=_split("val"),
                    test=_split("test"),
                    class_names=names)

    for tag, sp in (("train", ds.train), ("val", ds.val), ("test", ds.test)):
        logger.info("%s: %d samples, img=%s, labels=%s",
                    tag, sp.labels.size, sp.images.shape, sp.labels.shape)
    return ds


# --------------------------- Computation --------------------------------------

def aggregate_counts(ds: DatasetNPZ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return classes, train_counts, val_counts, test_counts, totals.
    """
    all_labels = np.concatenate([ds.train.labels, ds.val.labels, ds.test.labels], axis=0)
    classes = np.unique(all_labels).astype(int)
    classes.sort()

    def count(labels: np.ndarray) -> np.ndarray:
        if labels.size == 0:
            return np.zeros_like(classes, dtype=int)
        vals, freqs = np.unique(labels, return_counts=True)
        lookup = dict(zip(vals.astype(int), freqs.astype(int)))
        return np.array([lookup.get(int(c), 0) for c in classes], dtype=int)

    tr = count(ds.train.labels)
    va = count(ds.val.labels)
    te = count(ds.test.labels)
    tot = tr + va + te
    return classes, tr, va, te, tot


def order_by_total_desc(classes: np.ndarray,
                        tr: np.ndarray, va: np.ndarray, te: np.ndarray, tot: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reorder arrays by total counts descending.
    """
    order = np.argsort(-tot)  # descending
    return classes[order], tr[order], va[order], te[order], tot[order]


# ----------------------------- Plotting ---------------------------------------

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert [H,W,1] to RGB by replication; pass [H,W,3] through."""
    if img.ndim != 3:
        raise ValueError(f"Expected [H,W,C], got {img.shape}")
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported channels: {img.shape[2]}")


def pick_random_image(ds: DatasetNPZ, cls: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Pick a random image belonging to class cls from train, then val, then test.
    """
    pools = []
    for sp in (ds.train, ds.val, ds.test):
        idx = np.where(sp.labels == cls)[0]
        if idx.size:
            pools.append(sp.images[idx])
    if not pools:
        logger.warning("No images found for class %d. Using a zero placeholder.", cls)
        return np.zeros((28, 28, 3), dtype=np.uint8)
    pool = np.concatenate(pools, axis=0)
    img = pool[rng.randint(0, pool.shape[0])]
    return _ensure_rgb(img)


def plot_bars_with_thumbnails(classes: np.ndarray,
                              tr: np.ndarray, va: np.ndarray, te: np.ndarray, tot: np.ndarray,
                              ds: DatasetNPZ,
                              seed: int = 123,
                              figsize: Tuple[float, float] = (12.0, 7.0),
                              dpi: int = 150,
                              img_x_offset: float = IMG_X_OFFSET) -> plt.Figure:
    """
    Stacked bars with thumbnails centered above each bar. IEEE scienceplots style.
    
    Args:
        img_x_offset: Horizontal offset for images and class names (in bar-width units).
                     Positive values shift right, negative values shift left.
    """
    rng = np.random.RandomState(seed)
    K = classes.size
    x = np.arange(K, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Bars (stacked). Explicit colors.
    b_train = ax.bar(x, tr, color=COLOR_TRAIN, label="Train")
    b_val   = ax.bar(x, va, bottom=tr, color=COLOR_VAL, label="Validation")
    b_test  = ax.bar(x, te, bottom=tr + va, color=COLOR_TEST, label="Test")

    # Axes labels and ticks
    ax.set_xlabel("Class index")
    ax.set_ylabel("Number of images")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in classes])

    # Disable top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend below
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)

    # Headroom for images
    ymax = float(np.max(tot)) if tot.size else 1.0
    ax.set_ylim(0.0, ymax * (1.0 + HEADROOM_FRAC))

    # Precise centering: compute bar centers from patches
    # All three stacks have identical x and width; use top stack patches.
    for i, cls in enumerate(classes):
        top_patch = b_test.patches[i]
        bar_x = top_patch.get_x()
        bar_w = top_patch.get_width()
        bar_center = bar_x + bar_w / 2.0
        
        # Apply horizontal offset (in bar-width units)
        bar_center_with_offset = bar_center + (img_x_offset * bar_w)
        bar_top = float(tot[i])

        # Convert bar width to axis-fraction for inset sizing
        xlim = ax.get_xlim()
        ax_width_data = xlim[1] - xlim[0]
        img_width_frac = (IMG_WIDTH_SCALE * bar_w) / ax_width_data

        # Image height as fraction of axes height reserved within headroom
        img_height_frac = min(max(IMG_HEIGHT_FRAC, 0.05), HEADROOM_FRAC * 0.9)

        # Convert center position (with offset) to axes fraction
        bar_center_frac = (bar_center_with_offset - xlim[0]) / ax_width_data
        x_left_frac = bar_center_frac - img_width_frac / 2.0

        # Vertical placement: start above the bar by a small gap
        ylim = ax.get_ylim()
        ax_height_data = ylim[1] - ylim[0]
        y0_data = bar_top + IMAGE_GAP_FRAC * ymax
        # Map y0_data to axes fraction baseline, then place an inset with fixed fractional height
        y0_frac = (y0_data - ylim[0]) / ax_height_data

        iax = inset_axes(
            ax,
            width=f"{img_width_frac*100:.4f}%",
            height=f"{img_height_frac*100:.4f}%",
            bbox_transform=ax.transAxes,
            bbox_to_anchor=(x_left_frac, y0_frac, img_width_frac, img_height_frac),
            loc="lower left",
            borderpad=0.0
        )
        iax.axis("off")

        # Thumbnail
        img = pick_random_image(ds, int(cls), rng)
        iax.imshow(img.astype(np.float32) / 255.0)
        iax.set_xticks([])
        iax.set_yticks([])

        # Class name centered above the image
        name = ds.class_names.get(int(cls), f"class {int(cls)}")
        iax.text(0.5, 1.02, name, ha="center", va="bottom", transform=iax.transAxes,
                 fontsize=9, fontweight="bold")

    fig.tight_layout()
    return fig


# --------------------------- CLI Entrypoint ------------------------------------

def parse_args() -> argparse.Namespace:
    """Argument parser."""
    p = argparse.ArgumentParser(description="Stacked class barplot with centered thumbnails from custom NPZ.")
    p.add_argument("--npz", type=Path, required=True, help="Path to custom NPZ file.")
    p.add_argument("--out", type=Path, required=True, help="Output image path (.png/.pdf/.svg).")
    p.add_argument("--seed", type=int, default=123, help="RNG seed for thumbnails.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    p.add_argument("--width", type=float, default=12.0, help="Figure width (in).")
    p.add_argument("--height", type=float, default=7.0, help="Figure height (in).")
    p.add_argument("--img-x-offset", type=float, default=0.0, 
                   help="Horizontal offset for images and class names (in bar-width units). Positive=right, negative=left.")
    return p.parse_args()


def main() -> None:
    """Load NPZ, compute counts, order by total desc, render, save."""
    args = parse_args()
    ds = load_custom_npz(args.npz)
    classes, tr, va, te, tot = aggregate_counts(ds)
    classes, tr, va, te, tot = order_by_total_desc(classes, tr, va, te, tot)

    logger.info("Order by total (desc): %s", [int(c) for c in classes])
    fig = plot_bars_with_thumbnails(classes, tr, va, te, tot, ds,
                                    seed=args.seed,
                                    figsize=(args.width, args.height),
                                    dpi=args.dpi,
                                    img_x_offset=args.img_x_offset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", args.out)


if __name__ == "__main__":
    main()
