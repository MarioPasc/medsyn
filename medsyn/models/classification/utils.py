"""
Utility functions for classification data filtering.
"""

from __future__ import annotations
from typing import Literal
import numpy as np
import warnings

TrainingRegime = Literal["real_only", "real_plus_synth", "real_plus_trad_aug"]


def select_indices_by_regime(is_synth: np.ndarray, regime: TrainingRegime) -> np.ndarray:
    """
    Select data indices based on training regime.

    Args:
        is_synth: Boolean array indicating synthetic samples (N,)
        regime: Training regime

    Returns:
        Boolean mask of samples to keep (N,)
    """
    if regime == "real_only":
        return ~is_synth.astype(bool)
    elif regime == "real_plus_synth":
        return np.ones_like(is_synth, dtype=bool)
    elif regime == "real_plus_trad_aug":
        # Only real data (augmentation applied at runtime)
        return ~is_synth.astype(bool)
    else:
        raise ValueError(
            f"Unknown regime: {regime}. Must be one of: real_only, real_plus_synth, real_plus_trad_aug"
        )


# Legacy function for backward compatibility (deprecated)
def select_indices_by_training_images(is_synth: np.ndarray, mode: str) -> np.ndarray:
    """
    DEPRECATED: Use select_indices_by_regime() instead.

    Legacy mapping:
    - "PathMNIST" -> "real_only"
    - "PathMNIST_and_synth" -> "real_plus_synth"
    - "synth" -> N/A (not supported in new framework)

    Args:
        is_synth: Boolean array indicating synthetic samples (N,)
        mode: Training mode (legacy)

    Returns:
        Boolean mask of samples to keep (N,)
    """
    warnings.warn(
        "select_indices_by_training_images() is deprecated. "
        "Use select_indices_by_regime() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if mode == "PathMNIST":
        return ~is_synth.astype(bool)
    elif mode == "synth":
        return is_synth.astype(bool)
    else:  # PathMNIST_and_synth
        return np.ones_like(is_synth, dtype=bool)
