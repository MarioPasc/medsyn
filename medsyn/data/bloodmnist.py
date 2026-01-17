# medsyn/data/bloodmnist.py
"""
BloodMNIST dataset loader.

BloodMNIST is a dataset of blood cell images with 8 classes:
- basophil, eosinophil, erythroblast, immature_granulocyte,
- lymphocyte, monocyte, neutrophil, platelet

This module provides a convenience wrapper around the unified base loader.
"""
from __future__ import annotations
from medsyn.data.config import ProjectCfg
from medsyn.data.base import SplitDatasets, prepare_medmnist


def prepare_bloodmnist(cfg: ProjectCfg) -> SplitDatasets:
    """
    Prepare BloodMNIST with a stratified reduction that preserves label distribution.

    This is a convenience wrapper around prepare_medmnist() for BloodMNIST.

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
    return prepare_medmnist("bloodmnist", cfg)
