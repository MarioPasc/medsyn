#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a stacked bar plot from a custom PathMNIST-style NPZ.

- X axis: class indices (0..K-1)
- Y axis: total images per class (sum over train/val/test)
- Each bar is stacked by split (Train, Validation, Test)
- A small random image per class is shown above each bar
- The class name is written on top of that image

Assumes NPZ keys:
  train_images: [N,H,W,C] uint8
  train_labels: [N] int64
  train_is_synth: [N] bool
  val_images / val_labels / val_is_synth
  test_images / test_labels / test_is_synth

Best practices: typing, logging, clear functions, arg parsing, minimal state, deterministic RNG (seed).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ----------------------------- Logging ----------------------------------------

logger = logging.getLogger("npz_barplot")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --------------------------- Data Structures ----------------------------------

@dataclass(frozen=True)
class SplitArrays:
    """Container for images and labels of a split."""
    images: np.ndarray  # [N,H,W,C] uint8
    labels: np.ndarray  # [N] int64


@dataclass(frozen=True)
class DatasetNPZ:
    """Container for three splits and class names if provided."""
    train: SplitArrays
    val: SplitArrays
    test: SplitArrays
    class_names: Dict[int, str]


# ------------------------------ I/O -------------------------------------------

def load_custom_npz(
    npz_path: Path,
    class_names: Optional[Dict[int, str]] = None
) -> DatasetNPZ:
    """
    Load the custom NPZ and return split arrays with optional class names mapping.

    Args:
        npz_path: Path to the NPZ produced by your pipeline.
        class_names: Optional mapping from class index to human-readable name.

    Returns:
        DatasetNPZ with train/val/test arrays and class name mapping.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    logger.info("Loading NPZ: %s", npz_path)
    data = np.load(str(npz_path))

    required = [
        "train_images", "train_labels",
        "val_images", "val_labels",
        "test_images", "test_labels",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"NPZ missing keys: {missing}")

    def _split(split: str) -> SplitArrays:
        imgs = data[f"{split}_images"]
        lbls = data[f"{split}_labels"].astype(np.int64).reshape(-1)
        if imgs.ndim == 3:
            imgs = imgs[..., np.newaxis]
        return SplitArrays(images=imgs, labels=lbls)

    # Default class names if none provided (PathMNIST, 9 classes)
    default_names = {
        0: "Adipose",
        1: "Background",
        2: "Debris",
        3: "Lymphocytes",
        4: "Mucus",
        5: "Normal mucosa",
        6: "CA stroma",
        7: "CRC epithelium",
        8: "Smooth muscle",
    }
    names = class_names if class_names is not None else default_names

    ds = DatasetNPZ(
        train=_split("train"),
        val=_split("val"),
        test=_split("test"),
        class_names=names,
    )

    # Basic sanity logging
    for s, arr in (("train", ds.train), ("val", ds.val), ("test", ds.test)):
        logger.info(
            "%s: %d samples, img shape=%s, labels shape=%s",
            s, len(arr.labels), arr.images.shape, arr.labels.shape
        )
    return ds


# --------------------------- Computation --------------------------------------

def compute_class_counts(
    labels: np.ndarray,
    classes: np.ndarray
) -> Dict[int, int]:
    """
    Count labels per provided class set.

    Args:
        labels: 1D array of integer labels.
        classes: sorted unique class indices to count over.

    Returns:
        Dict[class_idx, count]
    """
    counts = {int(c): 0 for c in classes}
    if labels.size == 0:
        return counts
    vals, freqs = np.unique(labels, return_counts=True)
    for v, f in zip(vals, freqs):
        if int(v) in counts:
            counts[int(v)] = int(f)
    return counts


def aggregate_counts_by_split(ds: DatasetNPZ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-class counts for train/val/test and totals.

    Returns:
        classes: [K] class indices sorted ascending
        train_counts: [K]
        val_counts:   [K]
        test_counts:  [K]
        totals:       [K] = train+val+test
    """
    # Infer classes present across all splits
    all_labels = np.concatenate([ds.train.labels, ds.val.labels, ds.test.labels], axis=0)
    classes = np.unique(all_labels).astype(int)
    classes.sort()

    train_counts_d = compute_class_counts(ds.train.labels, classes)
    val_counts_d   = compute_class_counts(ds.val.labels, classes)
    test_counts_d  = compute_class_counts(ds.test.labels, classes)

    train_counts = np.array([train_counts_d[c] for c in classes], dtype=int)
    val_counts   = np.array([val_counts_d[c]   for c in classes], dtype=int)
    test_counts  = np.array([test_counts_d[c]  for c in classes], dtype=int)
    totals       = train_counts + val_counts + test_counts

    return classes, train_counts, val_counts, test_counts, totals


# ----------------------------- Plotting ---------------------------------------

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert [H,W,1] to RGB by channel replication. Leave [H,W,3] unchanged.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims [H,W,C], got shape={img.shape}")
    if img.shape[2] == 1:
        return np.repeat(img, 3, axis=2)
    if img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported channel count C={img.shape[2]}")


def choose_random_image_for_class(
    rng: np.random.RandomState,
    ds: DatasetNPZ,
    class_idx: int
) -> np.ndarray:
    """
    Pick a random image from any split for the given class. Falls back if empty.

    Selection priority: train, then val, then test.
    """
    imgs = []
    for split in (ds.train, ds.val, ds.test):
        idx = np.where(split.labels == class_idx)[0]
        if idx.size > 0:
            imgs.append(split.images[idx])
    if not imgs:
        # No image for this class; create a placeholder
        logger.warning("No images found for class %d; using a zero placeholder.", class_idx)
        return np.zeros((28, 28, 3), dtype=np.uint8)
    pool = np.concatenate(imgs, axis=0)
    sel = rng.randint(0, pool.shape[0])
    return _ensure_rgb(pool[sel])


def plot_stacked_bar_with_thumbnails(
    classes: np.ndarray,
    train_counts: np.ndarray,
    val_counts: np.ndarray,
    test_counts: np.ndarray,
    totals: np.ndarray,
    ds: DatasetNPZ,
    seed: int = 123,
    figsize: Tuple[float, float] = (12.0, 7.0),
    dpi: int = 150,
    top_image_frac_height: float = 0.18,
    image_pad_frac: float = 0.02
) -> plt.Figure:
    """
    Render the figure with stacked bars, thumbnails, and class labels.

    Args:
        classes: class indices, length K
        train_counts, val_counts, test_counts, totals: arrays length K
        ds: dataset container to sample thumbnails
        seed: RNG seed for reproducible random picks
        figsize: figure size in inches
        dpi: dots per inch
        top_image_frac_height: fraction of axis height reserved above tallest bar for images
        image_pad_frac: vertical gap fraction between bar tops and images

    Returns:
        Matplotlib Figure
    """
    rng = np.random.RandomState(seed)
    K = len(classes)
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Stacked bars. Heights are counts; the color proportions reflect per-split percentages.
    b1 = ax.bar(x, train_counts, label="Train")
    b2 = ax.bar(x, val_counts, bottom=train_counts, label="Validation")
    b3 = ax.bar(x, test_counts, bottom=(train_counts + val_counts), label="Test")

    ax.set_xlabel("Class index")
    ax.set_ylabel("Number of images")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in classes])

    # Legend below the plot
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    # Y limits with headroom for images
    ymax = float(max(totals)) if totals.size > 0 else 1.0
    pad = ymax * top_image_frac_height
    ax.set_ylim(0, ymax + pad)

    # For each class, place a small image above the bar and write the class name over it
    for i, c in enumerate(classes):
        # Position for this bar
        bar_top = totals[i]
        # Image axes as an inset above the bar
        # Width: slightly wider than bar width in axis coords
        bar = b3[i]  # any stack gives same width and x, pick top stack
        bar_width = bar.get_width()

        # Compute image box in axis coordinates
        # x0 in data coords -> axis fraction:
        x0_data = bar.get_x()
        x1_data = x0_data + bar_width
        y0_data = bar_top + ymax * image_pad_frac
        y1_data = ax.get_ylim()[1]

        # Convert desired image height to a fixed fraction to keep them small
        img_height_frac = top_image_frac_height * 0.8  # inside headroom
        img_width_frac = (1.2 * bar_width) / (ax.get_xlim()[1] - ax.get_xlim()[0])

        # Place inset centered on bar
        bar_center_frac = ( (x0_data + x1_data) / 2.0 - ax.get_xlim()[0] ) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        x_left_frac = bar_center_frac - img_width_frac / 2.0
        y_bottom_frac = 1.0 - img_height_frac  # from top region

        iax = inset_axes(
            ax,
            width=f"{img_width_frac*100:.2f}%",
            height=f"{img_height_frac*100:.2f}%",
            bbox_transform=ax.transAxes,
            bbox_to_anchor=(x_left_frac, y_bottom_frac, img_width_frac, img_height_frac),
            loc="lower left",
            borderpad=0.0,
        )
        iax.axis("off")

        # Pick thumbnail
        img = choose_random_image_for_class(rng, ds, int(c))
        # Normalize to [0,1] for imshow
        iax.imshow(img.astype(np.float32) / 255.0)
        iax.set_xticks([])
        iax.set_yticks([])

        # Class name centered over the image
        name = ds.class_names.get(int(c), f"class {int(c)}")
        iax.text(
            0.5, -0.08, name,
            ha="center", va="top", transform=iax.transAxes,
            fontsize=9, fontweight="bold"
        )

    fig.tight_layout()
    return fig


# --------------------------- CLI Entrypoint ------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    p = argparse.ArgumentParser(description="Stacked class distribution barplot with thumbnails from custom NPZ.")
    p.add_argument("--npz", type=Path, required=True, help="Path to custom NPZ.")
    p.add_argument("--out", type=Path, required=True, help="Output image file (.png/.pdf/.svg).")
    p.add_argument("--seed", type=int, default=123, help="Random seed for thumbnail selection.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    p.add_argument("--width", type=float, default=12.0, help="Figure width in inches.")
    p.add_argument("--height", type=float, default=7.0, help="Figure height in inches.")
    return p.parse_args()


def main() -> None:
    """
    CLI main: load NPZ, compute counts, render figure, save to disk.
    """
    args = parse_args()
    ds = load_custom_npz(args.npz)

    classes, tr, va, te, tot = aggregate_counts_by_split(ds)
    logger.info("Classes: %s", classes.tolist())
    logger.info("Totals per class: %s", tot.tolist())

    fig = plot_stacked_bar_with_thumbnails(
        classes, tr, va, te, tot, ds,
        seed=args.seed,
        figsize=(args.width, args.height),
        dpi=args.dpi
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", args.out)


if __name__ == "__main__":
    main()
