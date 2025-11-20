# medsyn/models/bVAE/engine/__init__.py
"""Training and prediction engines for conditional β-VAE."""

from .train import train
from .predict import generate_for_classes

__all__ = ["train", "generate_for_classes"]
