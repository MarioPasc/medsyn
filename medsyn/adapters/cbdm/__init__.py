# medsyn/adapters/cbdm/__init__.py
"""
CBDM (Class-Balancing Diffusion Models) adapter for MedSyn.

This package provides adapters that allow training and generating
with CBDM using MedSyn's unified NPZ dataset format.

CBDM Reference:
  Kim et al., "Class-Balancing Diffusion Models"
  Paper: https://arxiv.org/abs/2305.00562
"""
from medsyn.adapters.cbdm.config import CBDMConfig, load_cbdm_config
from medsyn.adapters.cbdm.dataset import CBDMNPZDataset

__all__ = [
    "CBDMConfig",
    "load_cbdm_config",
    "CBDMNPZDataset",
]
