# medsyn/data/base.py
"""
Base loader for MedMNIST datasets.

This module provides reusable loading and preprocessing functions
that work across all supported MedMNIST datasets.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Type
from pathlib import Path
import logging
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms

import medmnist

from medsyn.data.config import ProjectCfg
from medsyn.data.registry import DATASET_INFO

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitDatasets:
    """Container for train/val/test datasets (possibly reduced Subset objects)."""
    train: Dataset
    val: Dataset
    test: Dataset
    indices: Dict[str, Optional[List[int]]]  # indices used for each split


def _get_medmnist_class(dataset_name: str) -> Type:
    """
    Dynamically get the MedMNIST dataset class.

    Args:
        dataset_name: Dataset flag (e.g., 'pathmnist')

    Returns:
        MedMNIST dataset class (e.g., PathMNIST)
    """
    info = DATASET_INFO[dataset_name]
    class_name = info["medmnist_class"]
    return getattr(medmnist, class_name)


def _expected_npz_path(download_dir: str, flag: str, size: int) -> Path:
    """
    Determine expected NPZ file path for a MedMNIST dataset.

    MedMNIST stores datasets as `<flag>.npz` for size=28,
    and `<flag>_<size>.npz` for other sizes.
    """
    if size == 28:
        return Path(download_dir) / f"{flag}.npz"
    return Path(download_dir) / f"{flag}_{size}.npz"


def _download_if_needed(dataset_name: str, cfg: ProjectCfg) -> None:
    """
    Download the MedMNIST dataset if not already present.

    Uses MedMNIST's internal downloader by instantiating a split.

    Args:
        dataset_name: Dataset flag
        cfg: Project configuration
    """
    npz_path = _expected_npz_path(cfg.data.download_dir, dataset_name, cfg.data.size)
    if npz_path.exists():
        logger.info("Found %s. Skipping download.", npz_path.name)
        return

    logger.info("Downloading %s to %s ...", dataset_name, cfg.data.download_dir)
    dataset_class = _get_medmnist_class(dataset_name)
    # Trigger MedMNIST's internal downloader via any split
    _ = dataset_class(
        split="train",
        download=True,
        size=cfg.data.size,
        root=cfg.data.download_dir,
    )
    logger.info("Download complete.")


def _base_transform(size: int) -> transforms.Compose:
    """
    Compose minimal transforms for MedMNIST datasets.

    MedMNIST images are standardized to size=28 by default.

    Args:
        size: Target image size

    Returns:
        Composed transforms
    """
    t: List = [transforms.ToTensor()]  # [0,1], CxHxW
    if size != 28:
        t.insert(0, transforms.Resize(size))
    return transforms.Compose(t)


def _load_base_splits(dataset_name: str, cfg: ProjectCfg) -> Dict[str, Dataset]:
    """
    Instantiate canonical MedMNIST splits with deterministic transforms.

    Args:
        dataset_name: Dataset flag
        cfg: Project configuration

    Returns:
        Dictionary with 'train', 'val', 'test' keys mapping to datasets
    """
    tfm = _base_transform(cfg.data.size)
    dataset_class = _get_medmnist_class(dataset_name)
    common = dict(
        download=False,
        size=cfg.data.size,
        root=cfg.data.download_dir,
        transform=tfm,
    )
    train = dataset_class(split="train", **common)
    val = dataset_class(split="val", **common)
    test = dataset_class(split="test", **common)
    return {"train": train, "val": val, "test": test}


def _stratified_indices(
    labels: np.ndarray,
    cfg: ProjectCfg,
    split: str,
) -> Optional[List[int]]:
    """
    Build stratified indices per split.

    If fraction==1.0, return None to keep full split.

    Strategy 'fraction': sample round(n_c * frac) per class c.
    Strategy 'max_per_class': sample min(n_c, K) per class c.

    Args:
        labels: Array of labels for the split
        cfg: Project configuration
        split: Split name ('train', 'val', 'test')

    Returns:
        List of indices to use, or None if keeping full split
    """
    red = cfg.data.reduction
    if split == "train":
        frac = float(red.train)
    elif split == "val":
        frac = float(red.val)
    else:
        frac = float(red.test)

    rng = np.random.RandomState(cfg.data.seed)
    y = labels.astype(int).reshape(-1)
    classes = np.unique(y)

    if red.strategy == "fraction":
        if frac >= 1.0:
            return None
        per_class = {c: max(1, int(round((y == c).sum() * frac))) for c in classes}
    else:  # max_per_class
        if red.max_per_class is None or red.max_per_class <= 0:
            raise ValueError("max_per_class must be >0 when strategy='max_per_class'.")
        per_class = {c: min((y == c).sum(), int(red.max_per_class)) for c in classes}

    indices: List[int] = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        sel = rng.choice(idx_c, size=per_class[c], replace=False)
        indices.extend(sel.tolist())

    rng.shuffle(indices)
    logger.info("Split=%s reduction -> %d/%d samples", split, len(indices), len(y))
    return indices


def prepare_medmnist(dataset_name: str, cfg: ProjectCfg) -> SplitDatasets:
    """
    Prepare a MedMNIST dataset with stratified reduction.

    This is the unified entry point for all MedMNIST datasets.

    Steps:
      1) Ensure local .npz presence, downloading once if absent.
      2) Instantiate canonical MedMNIST splits (train/val/test).
      3) Build stratified indices per split according to cfg.data.reduction.
      4) Wrap into torch.utils.data.Subset for deterministic reduced sets.

    Args:
        dataset_name: Dataset flag ('pathmnist', 'bloodmnist', 'dermamnist')
        cfg: Project configuration

    Returns:
        SplitDatasets with train/val/test and the index mapping.
    """
    _download_if_needed(dataset_name, cfg)
    base = _load_base_splits(dataset_name, cfg)
    idx = {
        "train": _stratified_indices(base["train"].labels.squeeze(), cfg, split="train"),
        "val": _stratified_indices(base["val"].labels.squeeze(), cfg, split="val"),
        "test": _stratified_indices(base["test"].labels.squeeze(), cfg, split="test"),
    }
    reduced = {
        k: Subset(base[k], idx[k]) if idx[k] is not None else base[k]
        for k in ("train", "val", "test")
    }
    logger.info(
        "Prepared %s: train=%d val=%d test=%d",
        dataset_name,
        len(reduced["train"]),
        len(reduced["val"]),
        len(reduced["test"]),
    )
    return SplitDatasets(
        train=reduced["train"],
        val=reduced["val"],
        test=reduced["test"],
        indices=idx,
    )
