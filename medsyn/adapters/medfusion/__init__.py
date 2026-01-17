# medsyn/adapters/medfusion/__init__.py
"""
MedFusion adapter for MedSyn.

This package provides adapters that allow training and generating
with MedFusion using MedSyn's unified NPZ dataset format.

MedFusion Reference:
  Müller-Franzes et al., "A multimodal comparison of latent denoising diffusion
  probabilistic models and generative adversarial networks for medical image synthesis"
  Paper: https://arxiv.org/abs/2212.11936
"""
from medsyn.adapters.medfusion.config import MedFusionConfig, load_medfusion_config
from medsyn.adapters.medfusion.dataset import MedFusionNPZDataset, MedFusionDataModule

__all__ = [
    "MedFusionConfig",
    "load_medfusion_config",
    "MedFusionNPZDataset",
    "MedFusionDataModule",
]
