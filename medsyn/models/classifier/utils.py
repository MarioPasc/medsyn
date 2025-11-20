from __future__ import annotations
import numpy as np

def select_indices_by_training_images(is_synth: np.ndarray, mode: str) -> np.ndarray:
    """
    mode ∈ {"PathMNIST","PathMNIST_and_synth","synth"}
    Returns boolean mask of kept samples.
    """
    if mode == "PathMNIST":
        return ~is_synth.astype(bool)
    if mode == "synth":
        return is_synth.astype(bool)
    return np.ones_like(is_synth, dtype=bool)
