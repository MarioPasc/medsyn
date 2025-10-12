# medsyn/models/bVAE/__init__.py
"""Conditional β-VAE model for PathMNIST synthesis."""

from .model import ConditionalBetaVAE
from .config import load_bvae_config, BVAEConfig

__all__ = ["ConditionalBetaVAE", "load_bvae_config", "BVAEConfig"]
