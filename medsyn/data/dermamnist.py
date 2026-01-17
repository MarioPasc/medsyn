# medsyn/data/dermamnist.py
"""
DermaMNIST dataset loader.

DermaMNIST is a dataset of dermatoscopic images with 7 classes:
- actinic_keratoses, basal_cell_carcinoma, benign_keratosis,
- dermatofibroma, melanoma, melanocytic_nevi, vascular_lesions

This module provides a convenience wrapper around the unified base loader.
"""
from __future__ import annotations
from medsyn.data.config import ProjectCfg
from medsyn.data.base import SplitDatasets, prepare_medmnist


def prepare_dermamnist(cfg: ProjectCfg) -> SplitDatasets:
    """
    Prepare DermaMNIST with a stratified reduction that preserves label distribution.

    This is a convenience wrapper around prepare_medmnist() for DermaMNIST.

    Steps:
      1) Ensure local .npz presence, downloading once if absent.
      2) Instantiate canonical MedMNIST splits (train/val/test).
      3) Build stratified indices per split according to cfg.data.reduction.
      4) Wrap into torch.utils.data.Subset for deterministic reduced sets.

    Args:
        cfg: Project configuration

    Returns:
        SplitDatasets with train/val/test and the index mapping.
    """
    return prepare_medmnist("dermamnist", cfg)
