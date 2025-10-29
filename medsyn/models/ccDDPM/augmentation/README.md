# ccDDPM Data Augmentation Module

## Overview

The augmentation module provides in-memory data augmentation for training class-conditional DDPM models. It leverages the [albumentations](https://albumentations.ai/) library for efficient image transformations and includes detailed statistics tracking.

**Key Features:**
- ✅ In-memory augmentation (no disk writes)
- ✅ Configurable via YAML
- ✅ Automatic memory cleanup
- ✅ Detailed CSV statistics tracking
- ✅ Multiple preset configurations
- ✅ Support for any albumentations transform
- ✅ Applied only during training (not validation/testing)

## Installation

The augmentation module requires the albumentations library:

```bash
pip install albumentations
```

If albumentations is not installed, the module will be skipped gracefully and training will proceed without augmentation.

## Quick Start

### 1. Enable Augmentation in Config

Edit your `config/medsyn_cfg.yaml`:

```yaml
ccddpm:
  augmentation:
    enabled: true  # Enable augmentation
    probability: 0.5  # 50% chance to augment each image
    transforms:
      - name: HorizontalFlip
        p: 0.5
      - name: Rotate
        p: 0.3
        limit: 10
    statistics:
      enabled: true
      output_path: augmentation_stats.csv
```

### 2. Train Your Model

```bash
python -m medsyn.cli.train_ccDDPM config/medsyn_cfg.yaml
```

Augmentation will be automatically applied during training. Statistics will be saved to `{output_dir}/augmentation_stats.csv`.

## Configuration Reference

### Top-Level Settings

```yaml
augmentation:
  enabled: false  # Master switch
  probability: 0.5  # Overall probability of applying augmentation (0.0-1.0)
  preserve_range: true  # Clamp outputs to [-1, 1] after augmentation
  normalize_after_augment: true  # Re-normalize after augmentation
```

- **enabled**: Master switch for augmentation. If `false`, no augmentation is applied.
- **probability**: Overall probability that augmentation will be applied to an image. For example, `0.5` means 50% of training images will be augmented.
- **preserve_range**: If `true`, output values are clamped to `[-1, 1]` after augmentation to maintain consistency with the expected input range.
- **normalize_after_augment**: Whether to re-normalize images after augmentation.

### Transform Configuration

Each transform has the following structure:

```yaml
transforms:
  - name: TransformName  # Any albumentations transform
    p: 0.5  # Probability of applying this specific transform
    param1: value1  # Transform-specific parameters
    param2: value2
```

**Example:**
```yaml
transforms:
  - name: Rotate
    p: 0.3  # 30% chance to apply rotation
    limit: 10  # Rotate by up to ±10 degrees

  - name: GaussNoise
    p: 0.2  # 20% chance to apply Gaussian noise
    var_limit: [10.0, 50.0]  # Noise variance range
```

### Statistics Configuration

```yaml
statistics:
  enabled: true  # Enable statistics tracking
  output_path: augmentation_stats.csv  # Output CSV file path
  save_every_n_epochs: 0  # How often to save (0 = only at end)
```

- **enabled**: If `true`, the module tracks which augmentations were applied to each image.
- **output_path**: Path to the CSV file. Can be relative (to `output_dir`) or absolute.
- **save_every_n_epochs**:
  - `0`: Save only at the end of training
  - `1`: Save after every epoch
  - `N`: Save every N epochs

## Available Transforms

The augmentation module supports all [albumentations transforms](https://albumentations.ai/docs/api_reference/augmentations/). Common transforms for medical imaging:

### Geometric Transforms

```yaml
# Horizontal flip
- name: HorizontalFlip
  p: 0.5

# Vertical flip
- name: VerticalFlip
  p: 0.5

# Rotation
- name: Rotate
  p: 0.3
  limit: 10  # Degrees: [-10, 10]
  interpolation: 1  # 1=BILINEAR, 3=CUBIC
  border_mode: 0  # 0=CONSTANT

# Shift, scale, and rotate
- name: ShiftScaleRotate
  p: 0.3
  shift_limit: 0.0625  # Shift by up to ±6.25%
  scale_limit: 0.1  # Scale by ±10%
  rotate_limit: 15  # Rotate by ±15°
```

### Intensity Transforms

```yaml
# Brightness and contrast
- name: RandomBrightnessContrast
  p: 0.3
  brightness_limit: 0.2  # ±20%
  contrast_limit: 0.2  # ±20%

# Gamma correction
- name: RandomGamma
  p: 0.2
  gamma_limit: [80, 120]

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
- name: CLAHE
  p: 0.2
  clip_limit: 4.0
  tile_grid_size: [8, 8]
```

### Noise and Blur

```yaml
# Gaussian noise
- name: GaussNoise
  p: 0.2
  var_limit: [10.0, 50.0]

# Gaussian blur
- name: GaussianBlur
  p: 0.2
  blur_limit: [3, 7]

# Motion blur
- name: MotionBlur
  p: 0.1
  blur_limit: 7
```

### Advanced Transforms

```yaml
# Elastic deformation
- name: ElasticTransform
  p: 0.2
  alpha: 1
  sigma: 50
  alpha_affine: 50

# Grid distortion
- name: GridDistortion
  p: 0.2
  num_steps: 5
  distort_limit: 0.3
```

## Preset Configurations

The module includes three preset configurations optimized for medical imaging:

### Light (Safe for Medical Images)

Conservative augmentations suitable for most medical imaging tasks:

```yaml
augmentation:
  enabled: true
  probability: 0.5
  transforms:
    - name: HorizontalFlip
      p: 0.5
    - name: VerticalFlip
      p: 0.5
    - name: Rotate
      p: 0.3
      limit: 10
  statistics:
    enabled: true
```

**Use when:** You want minimal distortion of anatomical structures.

### Moderate (Balanced)

Balanced augmentation with intensity and geometric transforms:

```yaml
augmentation:
  enabled: true
  probability: 0.7
  transforms:
    - name: HorizontalFlip
      p: 0.5
    - name: VerticalFlip
      p: 0.5
    - name: Rotate
      p: 0.4
      limit: 15
    - name: RandomBrightnessContrast
      p: 0.3
      brightness_limit: 0.2
      contrast_limit: 0.2
    - name: GaussNoise
      p: 0.2
      var_limit: [10.0, 50.0]
  statistics:
    enabled: true
```

**Use when:** You have limited training data and need to improve generalization.

### Aggressive (Use with Caution)

Extensive augmentation for challenging scenarios:

```yaml
augmentation:
  enabled: true
  probability: 0.8
  transforms:
    - name: HorizontalFlip
      p: 0.5
    - name: VerticalFlip
      p: 0.5
    - name: Rotate
      p: 0.5
      limit: 20
    - name: ShiftScaleRotate
      p: 0.3
      shift_limit: 0.0625
      scale_limit: 0.1
      rotate_limit: 15
    - name: RandomBrightnessContrast
      p: 0.4
      brightness_limit: 0.3
      contrast_limit: 0.3
    - name: GaussNoise
      p: 0.3
      var_limit: [10.0, 50.0]
    - name: ElasticTransform
      p: 0.2
      alpha: 1
      sigma: 50
      alpha_affine: 50
  statistics:
    enabled: true
```

**Use when:** You have very limited data or need robust generalization across diverse conditions.

⚠️ **Warning:** Aggressive augmentation can alter diagnostic features. Always validate on a held-out test set.

## Statistics Output

The augmentation statistics are saved as a CSV file with the following structure:

| epoch | total_images | original | HorizontalFlip | Rotate | HorizontalFlip+Rotate | ... |
|-------|--------------|----------|----------------|--------|-----------------------|-----|
| 1     | 10000        | 5123     | 2456           | 1234   | 1187                  | ... |
| 2     | 10000        | 5089     | 2501           | 1198   | 1212                  | ... |

**Columns:**
- **epoch**: Training epoch number
- **total_images**: Total images processed in this epoch
- **original**: Number of images that were NOT augmented
- **Transform names**: Count of images that received each transform combination

**Summary Statistics:**

At the end of training, a summary is printed:

```
================================================================================
Augmentation Statistics Summary
================================================================================
Total images processed: 800,000
  Original (no augmentation): 400,123 (50.0%)
  Augmented: 399,877 (50.0%)

Unique augmentation combinations: 15
Most common augmentation: HorizontalFlip (156,234 images)
Epochs tracked: 80

Statistics saved to: /path/to/outputs/augmentation_stats.csv
================================================================================
```

## Best Practices for Medical Imaging

### 1. Start Conservative

Begin with light augmentation and gradually increase if needed:

```yaml
# Start here
probability: 0.3
transforms:
  - name: HorizontalFlip
    p: 0.5
```

### 2. Validate on Real Data

Always test your model on a real (non-augmented) validation set to ensure augmentation doesn't harm performance.

### 3. Avoid Unrealistic Transforms

Some transforms may create unrealistic images:

❌ **Avoid:**
- Extreme rotations (>30°) for orientation-sensitive anatomy
- Heavy elastic deformations for structured tissues
- Strong color shifts for stained histopathology

✅ **Safe:**
- Horizontal/vertical flips (if anatomy allows)
- Small rotations (±10-15°)
- Mild brightness/contrast adjustments
- Subtle noise

### 4. Monitor Statistics

Check the augmentation statistics CSV to ensure:
- Augmentation rate matches expectations
- Transform combinations are reasonable
- No single transform dominates

### 5. Memory Considerations

All augmentation is done in-memory. For very large images:
- Monitor GPU/RAM usage during training
- Reduce batch size if needed
- Disable augmentation for extremely high-resolution images

## Programmatic Usage

You can also use the augmentation module programmatically:

```python
from medsyn.models.ccDDPM.augmentation import (
    AugmentationConfig,
    TransformConfig,
    create_augmentation_pipeline,
    AugmentationStatistics,
    MEDICAL_PRESET_LIGHT,
    MEDICAL_PRESET_MODERATE,
    MEDICAL_PRESET_AGGRESSIVE,
)

# Create config from dictionary
aug_cfg = AugmentationConfig.from_dict(MEDICAL_PRESET_LIGHT)

# Create pipeline
pipeline = create_augmentation_pipeline(aug_cfg, use_replay=True)

# Apply to image (PyTorch tensor [C, H, W] in range [-1, 1])
augmented_image, applied_transforms = pipeline(image, return_applied_transforms=True)

# Track statistics
stats = AugmentationStatistics(
    output_path="/path/to/stats.csv",
    transform_names=pipeline.get_transform_names()
)
stats.record_batch([applied_transforms], epoch=1)
stats.save_csv(final=True)
stats.print_summary()
```

## Troubleshooting

### Augmentation not being applied

**Check:**
1. `enabled: true` in config
2. `probability > 0.0`
3. At least one transform with `p > 0.0`
4. Albumentations is installed: `pip list | grep albumentations`

**Debug:**
Check training logs for:
```
INFO: Augmentation enabled: True, probability: 0.50, transforms: 5
```

### Statistics file not created

**Check:**
1. `statistics.enabled: true`
2. Output directory exists and is writable
3. Training completed at least one epoch

### Poor model performance after enabling augmentation

**Solutions:**
1. Reduce `probability` (e.g., from 0.8 to 0.5)
2. Switch to a lighter preset
3. Remove intensity transforms (keep only geometric)
4. Validate that transforms are appropriate for your data

### High memory usage

**Solutions:**
1. Reduce batch size
2. Disable augmentation for very large images
3. Use simpler transforms (avoid ElasticTransform)

## FAQ

**Q: Does augmentation apply to validation/test data?**
A: No. Augmentation is only applied to the training split.

**Q: Are augmented images saved to disk?**
A: No. All augmentation is done in-memory and augmented images are discarded after the training step.

**Q: Can I use custom transforms?**
A: Yes. Any albumentations transform can be used by specifying its name and parameters in the config.

**Q: How does the overall probability work with individual transform probabilities?**
A: First, the overall `probability` determines if augmentation is applied to an image. If yes, each transform is applied independently based on its own `p` value.

**Q: What happens if multiple transforms are applied?**
A: Transforms are composed into a pipeline and applied sequentially. The statistics CSV tracks combinations (e.g., "HorizontalFlip+Rotate").

**Q: Can I disable statistics tracking?**
A: Yes. Set `statistics.enabled: false` in the config.

## Module Architecture

```
medsyn/models/ccDDPM/augmentation/
├── __init__.py                  # Module exports
├── config.py                    # Configuration classes and presets
├── transforms.py                # Augmentation pipelines
├── statistics.py                # Statistics tracking
└── README.md                    # This file
```

**Key Classes:**
- `AugmentationConfig`: Main configuration dataclass
- `TransformConfig`: Single transform configuration
- `StatisticsConfig`: Statistics tracking configuration
- `AugmentationPipeline`: Basic augmentation pipeline
- `ReplayAugmentationPipeline`: Pipeline with exact transform tracking
- `AugmentationStatistics`: Statistics tracker and CSV writer

## References

- [Albumentations Documentation](https://albumentations.ai/)
- [Albumentations API Reference](https://albumentations.ai/docs/api_reference/augmentations/)
- [Data Augmentation for Medical Imaging](https://doi.org/10.1016/j.media.2019.101535)

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify albumentations installation
3. Review the configuration against examples
4. Check the statistics CSV for unexpected behavior

---

**Version:** 1.0.0
**Last Updated:** 2025-10-29
