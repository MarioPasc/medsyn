# medsyn/models/ccDDPM/augmentation/__init__.py
"""
Data Augmentation Module for ccDDPM Training

This module provides data augmentation capabilities using albumentations library.
Features:
- Configurable augmentation pipelines
- In-memory augmentation (no disk writes)
- Detailed statistics tracking to CSV
- Support for medical imaging augmentations

Example usage:
    ```python
    from medsyn.models.ccDDPM.augmentation import (
        create_augmentation_pipeline,
        AugmentationStatistics,
        AugmentationConfig
    )

    # Create pipeline from config
    pipeline = create_augmentation_pipeline(config)

    # Apply to image
    augmented_image, applied_transforms = pipeline(image)

    # Track statistics
    stats = AugmentationStatistics(output_path, transform_names)
    stats.record_batch(applied_transforms_list, epoch=1)
    stats.save_csv()
    ```
"""

from .config import (
    AugmentationConfig,
    TransformConfig,
    StatisticsConfig,
    MEDICAL_PRESET_LIGHT,
    MEDICAL_PRESET_MODERATE,
    MEDICAL_PRESET_AGGRESSIVE,
)
from .transforms import (
    AugmentationPipeline,
    ReplayAugmentationPipeline,
    create_augmentation_pipeline,
)
from .statistics import AugmentationStatistics

__all__ = [
    # Configuration
    'AugmentationConfig',
    'TransformConfig',
    'StatisticsConfig',
    # Presets
    'MEDICAL_PRESET_LIGHT',
    'MEDICAL_PRESET_MODERATE',
    'MEDICAL_PRESET_AGGRESSIVE',
    # Pipeline
    'AugmentationPipeline',
    'ReplayAugmentationPipeline',
    'create_augmentation_pipeline',
    # Statistics
    'AugmentationStatistics',
]

__version__ = '1.0.0'
