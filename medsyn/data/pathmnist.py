# data/pathmnist.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from pathlib import Path
import logging
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from medmnist import PathMNIST  # API: PathMNIST(split=..., download=..., size=...)  :contentReference[oaicite:2]{index=2}
from medsyn.data.config import ProjectCfg

logger = logging.getLogger(__name__)

# ---- Public API ----------------------------------------------------------------

@dataclass(frozen=True)
class SplitDatasets:
    """Container for train/val/test datasets (possibly reduced Subset objects)."""
    train: Dataset
    val: Dataset
    test: Dataset
    indices: Dict[str, List[int]]  # indices used for each split

def prepare_pathmnist(cfg: ProjectCfg) -> SplitDatasets:
    """
    Prepare PathMNIST with a stratified reduction that preserves label distribution.

    Steps:
      1) Ensure local .npz presence, downloading once if absent.
      2) Instantiate canonical MedMNIST splits (train/val/test).
      3) Build stratified indices per split according to cfg.data.reduction.
      4) Wrap into torch.utils.data.Subset for deterministic reduced sets.

    Returns:
        SplitDatasets with train/val/test and the index mapping.
    """
    _ensure_download_once(cfg)
    base = _load_base_splits(cfg)
    idx = {
        "train": _stratified_indices(base["train"].labels.squeeze(), cfg, split="train"),
        "val":   _stratified_indices(base["val"].labels.squeeze(),   cfg, split="val"),
        "test":  _stratified_indices(base["test"].labels.squeeze(),  cfg, split="test"),
    }
    reduced = {
        k: Subset(base[k], idx[k]) if idx[k] is not None else base[k]
        for k in ("train", "val", "test")
    }
    logger.info("Prepared PathMNIST: train=%d val=%d test=%d",
                len(reduced["train"]), len(reduced["val"]), len(reduced["test"]))
    return SplitDatasets(train=reduced["train"], val=reduced["val"], test=reduced["test"], indices=idx)

# ---- Internals -----------------------------------------------------------------

def _expected_npz_path(cfg: ProjectCfg) -> Path:
    """
    MedMNIST stores each subset as `<flag>.npz` for size=28. We target 28 by default.  :contentReference[oaicite:3]{index=3}
    """
    return Path(cfg.data.download_dir) / f"{cfg.data.flag}.npz"

def _ensure_download_once(cfg: ProjectCfg) -> None:
    """
    Download the `.npz` only if missing. Uses MedMNIST's downloader.
    """
    npz = _expected_npz_path(cfg)
    if npz.exists():
        logger.info("Found %s. Skipping download.", npz.name)
        return
    logger.info("Downloading %s to %s ...", cfg.data.flag, cfg.data.download_dir)
    # Trigger MedMNIST's internal downloader once via any split
    _ = PathMNIST(split="train", download=True, size=cfg.data.size, root=cfg.data.download_dir)
    logger.info("Download complete.")

def _base_transform(size: int) -> transforms.Compose:
    """
    Compose minimal transforms. MedMNIST images are already standardized to size=28 by default.  :contentReference[oaicite:4]{index=4}
    """
    t: List[transforms.Transform] = [transforms.ToTensor()]  # [0,1], CxHxW
    if size != 28:
        t.insert(0, transforms.Resize(size))
    return transforms.Compose(t)

def _load_base_splits(cfg: ProjectCfg) -> Dict[str, PathMNIST]:
    """
    Instantiate canonical MedMNIST splits with deterministic transforms.
    """
    tfm = _base_transform(cfg.data.size)
    common = dict(download=False, size=cfg.data.size, root=cfg.data.download_dir, transform=tfm)
    train = PathMNIST(split="train", **common)
    val   = PathMNIST(split="val",   **common)
    test  = PathMNIST(split="test",  **common)
    return {"train": train, "val": val, "test": test}

def _stratified_indices(labels: np.ndarray, cfg: ProjectCfg, split: str) -> List[int] | None:
    """
    Build stratified indices per split. If fraction==1.0, return None to keep full split.

    Strategy 'fraction': sample round(n_c * frac) per class c.
    Strategy 'max_per_class': sample min(n_c, K) per class c.
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
