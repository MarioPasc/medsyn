# ccDDPM Synthetic Image Generation Guide

**Last Updated**: 2025-10-29

## Overview

The `ccddpm-generate` CLI tool provides a comprehensive solution for generating synthetic medical images using trained class-conditioned DDPM models. This guide explains how to use the tool and understand its outputs.

### Recent Updates (2025-10-29)

✅ **Fixed critical bugs**:
- Guidance scale logic corrected (was returning wrong predictions for `scale=0`)
- Consistent CFG implementation across all generation functions
- See [TRAINING_FIXES_AND_DIAGNOSTICS.md](TRAINING_FIXES_AND_DIAGNOSTICS.md) for details

## Features

- **Per-class Image Generation**: Generate a specified number of synthetic images for each class
- **Organized Output Structure**: Automatic creation of class-specific directories
- **Unique Image Naming**: Format `synth_<uuid>_class<N>.png` with 12-character UUIDs
- **JSON Index Generation**: Compatible with `_build_index_structure` format from `cli/data.py`
- **Denoising Visualizations**: Scientific visualization of the diffusion process for random samples per class
- **EMA Weights Support**: Automatically uses EMA weights if available for higher quality
- **Classifier-Free Guidance**: Configurable guidance scale for controlled generation
- **Progress Tracking**: Real-time progress bars using tqdm

## Installation

Ensure you have installed the package with all dependencies:

```bash
cd /path/to/medsyn
pip install -e .
```

This will install matplotlib (required for visualizations) and all other dependencies.

## Configuration

### YAML Configuration File

Your configuration file (e.g., `config/medsyn_cfg.yaml`) must include a `generate` section:

```yaml
generate:
  checkpoint: /absolute/path/to/ccddpm/ckpts/best.pt
  classes:
    0: 100  # Generate 100 images for class 0
    1: 150  # Generate 150 images for class 1
    2: 200  # Generate 200 images for class 2
    3: 100
    4: 100
    5: 100
    6: 100
    7: 100
    8: 100

ccddpm:
  infer:
    guidance_scale: 2.0          # CFG scale (0=unconditional, >0=conditional)
    num_inference_steps: 1000    # Number of denoising steps
    out_dir: /path/to/output     # Output directory for generated images
```

### Key Parameters

- **checkpoint**: Absolute path to the trained model checkpoint (typically `best.pt`)
- **classes**: Dictionary mapping class IDs to the number of samples to generate
- **guidance_scale**: Classifier-free guidance strength (higher = more class-specific)
- **num_inference_steps**: Number of denoising steps (higher = better quality, slower)
- **out_dir**: Base directory for output files

## Usage

### Basic Usage

```bash
ccddpm-generate config/medsyn_cfg.yaml
```

### Advanced Usage

```bash
# Specify config explicitly
ccddpm-generate --config config/medsyn_cfg.yaml

# Override output directory
ccddpm-generate config/medsyn_cfg.yaml --output /custom/output/path

# Disable denoising visualizations (faster)
ccddpm-generate config/medsyn_cfg.yaml --no-visualizations

# Custom dataset and split names for JSON index
ccddpm-generate config/medsyn_cfg.yaml --dataset-name PathMNIST --split-name synthetic
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `config` | positional | `config/medsyn_cfg.yaml` | Path to YAML configuration |
| `--config` | optional | - | Alternative way to specify config |
| `--output` | optional | From config | Override output directory |
| `--no-visualizations` | flag | False | Disable denoising process visualizations |
| `--dataset-name` | optional | `PathMNIST` | Dataset name for JSON index |
| `--split-name` | optional | `synth` | Split name for JSON index |

## Output Structure

The script creates the following directory structure:

```
output_dir/
├── class_0/
│   ├── synth_a1b2c3d4e5f6_class0.png
│   ├── synth_b2c3d4e5f6a1_class0.png
│   ├── denoising_process_class0_a1b2c3d4e5f6.png  # Visualization
│   └── ...
├── class_1/
│   ├── synth_c3d4e5f6a1b2_class1.png
│   └── ...
├── class_2/
│   └── ...
└── pathmnist_synth_index.json  # JSON index file
```

### File Naming Convention

- **Synthetic Images**: `synth_<uuid>_class<N>.png`
  - `uuid`: 12-character hexadecimal unique identifier
  - `N`: Class ID (0-8 for PathMNIST)

- **Denoising Visualizations**: `denoising_process_class<N>_<uuid>.png`
  - One per class, showing the progression from noise to final image

### JSON Index Format

The generated JSON file follows the same structure as `_build_index_structure`:

```json
{
  "PathMNIST": {
    "synth": {
      "0": {
        "image": "class_0/synth_a1b2c3d4e5f6_class0.png",
        "label": 0,
        "is_synth": true,
        "uuid": "a1b2c3d4e5f6"
      },
      "1": {
        "image": "class_0/synth_b2c3d4e5f6a1_class0.png",
        "label": 0,
        "is_synth": true,
        "uuid": "b2c3d4e5f6a1"
      },
      ...
    }
  }
}
```

This format is compatible with the dataset indexing system used throughout the medsyn project.

## Denoising Process Visualization

For each class, the script generates a visualization showing the denoising process:

- **Pure Noise** → **Intermediate Steps** → **Final Image**
- Shows 10 evenly-spaced intermediate steps
- Saved as high-resolution PNG (150 DPI)
- Publication-ready format

This visualization helps understand how the diffusion model gradually removes noise to create the final synthetic image.

## Technical Details

### Model Loading

The script automatically:
1. Loads the configuration from YAML
2. Initializes the ccDDPM model architecture
3. Loads weights from the checkpoint
4. **Prefers EMA weights** if available (higher quality)
5. Sets up the DDPM scheduler with correct parameters

### Generation Process

For each image:
1. Start from pure Gaussian noise: `x_T ~ N(0, I)`
2. Apply `num_inference_steps` denoising steps
3. Use classifier-free guidance if `guidance_scale > 0`:
   - `ε = ε_uncond + scale * (ε_cond - ε_uncond)`
4. Clamp output to [-1, 1] range
5. Normalize to [0, 1] for saving

### Python 3.10+ Features Used

- **PEP 604 Union Types**: `str | Path` instead of `Union[str, Path]`
- **Type Hints**: Comprehensive type annotations throughout
- **Modern dict syntax**: `dict[int, int]` instead of `Dict[int, int]`
- **f-strings**: For all string formatting
- **Dataclasses**: Configuration structures
- **Context managers**: Proper resource management

## Performance Considerations

### Memory Usage

- **Batch size**: Images generated one at a time to minimize memory
- **GPU Memory**: Typically requires 4-8GB VRAM depending on image size
- **CPU Fallback**: Automatically uses CPU if CUDA unavailable

### Speed

Approximate generation times (64x64 images, 1000 steps, RTX 3090):
- **Per image**: ~5-10 seconds
- **100 images**: ~8-15 minutes
- **With visualization**: +2-3 seconds per visualized image

### Optimization Tips

1. **Reduce inference steps** for faster generation (trade-off: quality)
2. **Disable visualizations** with `--no-visualizations`
3. **Use EMA weights** (automatic if available)
4. **Reduce guidance scale** to 0 for unconditional (faster)

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Checkpoint file not found`
- **Solution**: Ensure the checkpoint path in config is absolute and exists

**Issue**: `CUDA out of memory`
- **Solution**: The script uses batch size 1; check if model fits on GPU

**Issue**: `ImportError: No module named 'matplotlib'`
- **Solution**: Reinstall with `pip install -e .` to get matplotlib

**Issue**: Images look poor quality
- **Solution**:
  - Check if EMA weights are being used (look for log message)
  - Increase `num_inference_steps` (e.g., 1000)
  - Adjust `guidance_scale` (try 1.5-3.0)

### Debug Mode

For detailed logging:

```bash
PYTHONPATH=. python -m medsyn.cli.generate_ccDDPM config/medsyn_cfg.yaml
```

## Integration with Existing Pipeline

The generated images and JSON index can be directly integrated with:

1. **Training Pipeline**: Use JSON index for augmented training
2. **YOLO Dataset**: Convert index to YOLO format using existing tools
3. **NPZ Pipeline**: Use `json_to_npz.py` to create NPZ from JSON index
4. **Evaluation**: Compare synthetic vs. real image distributions

## Example Workflow

```bash
# 1. Train ccDDPM model
ccddpm-train config/medsyn_cfg.yaml

# 2. Generate synthetic images
ccddpm-generate config/medsyn_cfg.yaml

# 3. Verify output
ls -lh /path/to/output/class_*/
cat /path/to/output/pathmnist_synth_index.json | jq '.PathMNIST.synth | length'

# 4. (Optional) Convert to NPZ for faster loading
python -m medsyn.cli.json_to_npz --index /path/to/output/pathmnist_synth_index.json
```

## Best Practices

1. **Always use best.pt**: Use the best checkpoint from training, not last.pt
2. **Validate config**: Ensure YAML is properly formatted before generation
3. **Monitor disk space**: Each 64x64 PNG is ~10-20KB
4. **Keep visualizations**: Useful for quality control and presentations
5. **Document parameters**: Note guidance_scale and steps used for reproducibility

## Citation

If you use this generation tool in your research, please cite:

```bibtex
@software{medsyn2024,
  author = {Pascual-González, M. and Cebolla Salas, Martina},
  title = {MedSyn: Medical Image Synthesis Framework},
  year = {2024},
  url = {https://github.com/MarioPasc/medsyn}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/MarioPasc/medsyn/issues
- Email: mpascual@uma.es
