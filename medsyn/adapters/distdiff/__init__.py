# medsyn/adapters/distdiff/__init__.py
"""
DistDiff adapter for MedSyn.

This package provides a standardized interface for DistDiff that matches
the CBDM and MedFusion adapter patterns.

DistDiff uses Stable Diffusion with prototype-guided generation to create
synthetic medical images.
"""
from medsyn.adapters.distdiff.config import DistDiffConfig, load_distdiff_config
from medsyn.adapters.distdiff.dataset import (
    DistDiffNPZDataset,
    load_npz_dataset,
    create_npz_dataloaders,
    get_class_names,
)

__all__ = [
    "DistDiffConfig",
    "load_distdiff_config",
    "DistDiffNPZDataset",
    "load_npz_dataset",
    "create_npz_dataloaders",
    "get_class_names",
]
