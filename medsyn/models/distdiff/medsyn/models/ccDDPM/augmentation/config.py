# medsyn/models/ccDDPM/augmentation/config.py
# Purpose: Configuration dataclasses for augmentation pipeline
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class TransformConfig:
    """
    Configuration for a single augmentation transform.

    Attributes:
        name: Name of the albumentations transform (e.g., 'HorizontalFlip')
        p: Probability of applying this transform (0.0 to 1.0)
        params: Additional parameters for the transform
    """
    name: str
    p: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"Transform probability must be in [0, 1], got {self.p}")


@dataclass
class StatisticsConfig:
    """
    Configuration for augmentation statistics tracking.

    Attributes:
        enabled: Whether to track and save statistics
        output_path: Path to save the CSV statistics file
        save_every_n_epochs: Save statistics every N epochs (0 = only at end)
    """
    enabled: bool = True
    output_path: Path = Path("outputs/ccddpm/augmentation_stats.csv")
    save_every_n_epochs: int = 0  # 0 = save only at end, 1 = every epoch, etc.


@dataclass
class AugmentationConfig:
    """
    Main configuration for data augmentation pipeline.

    Attributes:
        enabled: Master switch for augmentation
        probability: Overall probability of applying any augmentation (0.0-1.0)
        transforms: List of transform configurations
        statistics: Statistics tracking configuration
        normalize_after_augment: Whether to re-normalize after augmentation
        preserve_range: Keep pixel values in [-1, 1] range after augmentation
    """
    enabled: bool = False
    probability: float = 0.5
    transforms: List[TransformConfig] = field(default_factory=list)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    normalize_after_augment: bool = True
    preserve_range: bool = True  # Clamp to [-1, 1] after augmentation

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"Augmentation probability must be in [0, 1], got {self.probability}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> AugmentationConfig:
        """
        Create AugmentationConfig from dictionary (loaded from YAML).

        Args:
            config_dict: Dictionary with augmentation configuration

        Returns:
            AugmentationConfig instance
        """
        if not config_dict:
            return cls(enabled=False)

        # Parse transforms
        transforms = []
        for tf_dict in config_dict.get('transforms', []):
            name = tf_dict.get('name')
            p = tf_dict.get('p', 0.5)
            # All other keys are parameters
            params = {k: v for k, v in tf_dict.items() if k not in ['name', 'p']}
            transforms.append(TransformConfig(name=name, p=p, params=params))

        # Parse statistics config
        stats_dict = config_dict.get('statistics', {})
        statistics = StatisticsConfig(
            enabled=stats_dict.get('enabled', True),
            output_path=Path(stats_dict.get('output_path', 'outputs/ccddpm/augmentation_stats.csv')),
            save_every_n_epochs=stats_dict.get('save_every_n_epochs', 0)
        )

        return cls(
            enabled=config_dict.get('enabled', False),
            probability=config_dict.get('probability', 0.5),
            transforms=transforms,
            statistics=statistics,
            normalize_after_augment=config_dict.get('normalize_after_augment', True),
            preserve_range=config_dict.get('preserve_range', True)
        )


# Predefined augmentation presets for medical images
MEDICAL_PRESET_LIGHT = {
    'enabled': True,
    'probability': 0.5,
    'transforms': [
        {'name': 'HorizontalFlip', 'p': 0.5},
        {'name': 'VerticalFlip', 'p': 0.5},
        {'name': 'Rotate', 'p': 0.3, 'limit': 10},
    ],
    'statistics': {'enabled': True}
}

MEDICAL_PRESET_MODERATE = {
    'enabled': True,
    'probability': 0.7,
    'transforms': [
        {'name': 'HorizontalFlip', 'p': 0.5},
        {'name': 'VerticalFlip', 'p': 0.5},
        {'name': 'Rotate', 'p': 0.4, 'limit': 15},
        {'name': 'RandomBrightnessContrast', 'p': 0.3, 'brightness_limit': 0.2, 'contrast_limit': 0.2},
        {'name': 'GaussNoise', 'p': 0.2, 'var_limit': (10.0, 50.0)},
    ],
    'statistics': {'enabled': True}
}

MEDICAL_PRESET_AGGRESSIVE = {
    'enabled': True,
    'probability': 0.8,
    'transforms': [
        {'name': 'HorizontalFlip', 'p': 0.5},
        {'name': 'VerticalFlip', 'p': 0.5},
        {'name': 'Rotate', 'p': 0.5, 'limit': 20},
        {'name': 'ShiftScaleRotate', 'p': 0.3, 'shift_limit': 0.0625, 'scale_limit': 0.1, 'rotate_limit': 15},
        {'name': 'RandomBrightnessContrast', 'p': 0.4, 'brightness_limit': 0.3, 'contrast_limit': 0.3},
        {'name': 'GaussNoise', 'p': 0.3, 'var_limit': (10.0, 50.0)},
        {'name': 'ElasticTransform', 'p': 0.2, 'alpha': 1, 'sigma': 50, 'alpha_affine': 50},
    ],
    'statistics': {'enabled': True}
}
