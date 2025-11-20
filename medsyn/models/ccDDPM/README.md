# Class-Conditional Denoising Diffusion Probabilistic Model (ccDDPM)

**A robust implementation of class-conditional diffusion models for medical image synthesis**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation & Setup](#installation--setup)
5. [Quick Start](#quick-start)
6. [Configuration](#configuration)
7. [Training](#training)
8. [Generation](#generation)
9. [Module Structure](#module-structure)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Overview

This module implements a class-conditional DDPM for generating synthetic medical images. It supports:

- **Classifier-Free Guidance (CFG)** for enhanced conditional generation
- **Min-SNR Loss Weighting** for stable training across timesteps
- **EMA (Exponential Moving Average)** for improved sample quality
- **Comprehensive diagnostics** to detect and prevent training failures
- **Multiple dataloaders** (JSON-indexed or NPZ-compressed)

### Key Papers Implemented

1. **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
2. **Classifier-Free Guidance**: Ho & Salimans (2022)
3. **Min-SNR Weighting**: Hang et al. (2023)
4. **Improved Diffusion**: Dhariwal & Nichol (2021)

---

## Architecture

### Model Components

```
ccDDPM
├── ClassEmbedder          # Maps class labels → spatial conditioning
│   └── Embedding(num_classes, emb_dim)
│
└── UNet2DModel           # Diffusers' U-Net with attention
    ├── in_channels: 3 + class_embed_dim (concatenated)
    ├── out_channels: 3
    ├── block_out_channels: [128, 256, 256, 512]  # 4 levels
    ├── layers_per_block: 2
    └── attention: AttnDownBlock2D at level 4 (32×32)
```

### Class Conditioning

- **Method**: Channel concatenation (not time embedding)
- **Process**:
  1. Label `y` → Embedding → `[B, C_emb]`
  2. Broadcast to `[B, C_emb, H, W]`
  3. Concatenate with noisy image: `[x_t, c(y)]`
  4. Feed to U-Net

- **Classifier-Free Guidance**:
  - During training: 10% of samples use `y = -1` (unconditional)
  - At inference: `ε = ε_uncond + scale * (ε_cond - ε_uncond)`

---

## Features

### ✅ Production-Ready Training

- **Robust gradient handling**: Skip steps with NaN/Inf gradients
- **Mixed precision** (bfloat16 on Ampere+, fp16 fallback)
- **Early stopping** with configurable patience
- **Periodic checkpointing** every N epochs
- **EMA weights** properly tracked and saved

### 📊 Comprehensive Diagnostics

Computed every epoch to detect training failures:

| Metric | Healthy Range | Meaning |
|--------|---------------|---------|
| Input-Output Correlation | < 0.5 | Detects if model is echoing input |
| Prediction Std | 0.8-1.2 | Checks output distribution |
| Reconstruction PSNR@t500 | > 20 dB | Sample quality indicator |
| Conditioning Gap | > 0, growing | Verifies class labels affect outputs |
| Gradient Norm | < clip value | Optimization health check |

### 🎯 Flexible Generation

- Batch generation with progress bars
- Denoising process visualization (optional)
- Per-class output organization
- JSON index generation for downstream tasks
- Guidance scale sweep support

---

## Installation & Setup

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision diffusers
pip install tqdm pyyaml numpy pillow

# Optional: for visualization
pip install matplotlib
```

### Data Preparation

**Option 1: JSON Dataloader**
```bash
# Build index from PNG images
python -m medsyn.cli.data config/medsyn_cfg.yaml --build-index
```

**Option 2: NPZ Dataloader** (Recommended for HPC)
```bash
# Process dataset to compressed NPZ
python -m medsyn.cli.data config/medsyn_cfg.yaml --postprocess-npz
```

---

## Quick Start

### 1. Training

```bash
# Train from scratch
ccddpm-train config/medsyn_cfg.yaml

# Resume from checkpoint
ccddpm-train config/medsyn_cfg.yaml --resume /path/to/last.pt
```

### 2. Generation

```bash
# Generate synthetic images per class
ccddpm-generate config/medsyn_cfg.yaml

# Custom output directory
ccddpm-generate config/medsyn_cfg.yaml --output /path/to/output

# Disable visualizations for faster generation
ccddpm-generate config/medsyn_cfg.yaml --no-visualizations
```

### 3. Monitor Training

```bash
# View metrics in CSV
tail -f outputs/ccddpm/training_metrics.csv

# Check diagnostic warnings in console output
# Training will show ⚠️ warnings if issues detected
```

---

## Configuration

### Essential Settings

**Location**: `config/medsyn_cfg.yaml`

```yaml
ccddpm:
  dataloader:
    type: npz  # or "json"

  train:
    image_size: 64
    in_channels: 3
    class_embed_dim: 32        # MUST match checkpoint if resuming!
    num_classes: 9
    batch_size: 4
    epochs: 80

    # CRITICAL: Training stability settings
    grad_clip_norm: 10.0       # Was 1.0 (too aggressive)
    use_min_snr: true          # Enable Min-SNR weighting
    min_snr_gamma: 5.0

    # Classifier-free guidance setup
    guidance_p_uncond: 0.1     # 10% unconditional during training

    # EMA for better quality
    ema_use: true
    ema_decay: 0.999

  sched:
    num_train_timesteps: 1000
    beta_schedule: squaredcos_cap_v2
    prediction_type: epsilon   # ⚠️ DO NOT change without updating loss.py!

  infer:
    guidance_scale: 2.0        # 1.0=pure conditional, >1.0=enhanced
    num_inference_steps: 1000
```

### Critical Configuration Notes

⚠️ **DO NOT CHANGE** without code modifications:
- `prediction_type: epsilon` - Loss function assumes epsilon prediction
- `class_embed_dim` - Must match checkpoint architecture when resuming

✅ **Safe to modify**:
- `guidance_scale` (1.5-3.0 for enhanced quality)
- `batch_size`, `epochs`, `grad_clip_norm`
- `beta_schedule` (try `linear` or `squaredcos_cap_v2`)

---

## Training

### Training Loop Overview

```
For each epoch:
  1. Training phase
     ├─ Sample batch (x₀, y)
     ├─ Sample timestep t ~ Uniform(0, T)
     ├─ Add noise: x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
     ├─ Randomly drop 10% of labels (y → -1)
     ├─ Predict: ε̂ = model(x_t, t, y)
     ├─ Loss: MSE(ε̂, ε) with Min-SNR weighting
     └─ Update EMA weights

  2. Validation phase
     └─ Same as training (no label dropout)

  3. Diagnostics
     ├─ Input-output correlation
     ├─ Reconstruction quality metrics
     ├─ Conditioning gap check
     └─ ⚠️ Print warnings if issues detected

  4. Checkpointing
     ├─ Save best.pt (lowest val loss)
     ├─ Save last.pt (every epoch)
     └─ Save epoch_XXXX.pt (every 10 epochs)

  5. Visualizations (every 10 epochs)
     ├─ epoch_XXXX_recon.png
     ├─ epoch_XXXX_denoising.png
     └─ epoch_XXXX_classes.png (one sample per class)
```

### Monitoring Training Health

**Healthy Training**:
```
🔍 Training Diagnostics (detecting issues):
  Input-Output Correlation: 0.1234 ✓ (healthy)
  Prediction Std: 1.0234 ✓
  Reconstruction PSNR@t500: 23.45 dB ✓
```

**Failed Training**:
```
🔍 Training Diagnostics (detecting issues):
  Input-Output Correlation: 0.8234 ⚠️  WARNING: Model is echoing input!
  Prediction Std: 0.3234 ⚠️  Unusual (should be ~0.8-1.2)
  Reconstruction PSNR@t500: 12.34 dB ⚠️  Low quality
```

**Action**: If you see warnings for multiple epochs → STOP training and check configuration

### Checkpoint Format

```python
checkpoint = {
    "model": state_dict,              # Model weights
    "ema": ema_shadow_dict,          # EMA weights (331 params)
    "opt": optimizer_state,
    "epoch": int,
    "val_loss": float,
    "diagnostics": {                  # Health metrics
        "input_output_correlation": float,
        "reconstruction_psnr_t500": float,
        ...
    },
    "train_metrics": {...},
    "val_metrics": {...},
}
```

---

## Generation

### Classifier-Free Guidance (CFG)

**How it works**:
```python
# Two forward passes per timestep
ε_cond = model(x_t, t, y)        # With class label
ε_uncond = model(x_t, t, None)   # Without class label

# Blend predictions
ε = ε_uncond + guidance_scale * (ε_cond - ε_uncond)
```

**Guidance Scale Effects**:
- `scale = 1.0`: Pure conditional (single forward pass)
- `scale = 2.0`: Recommended default (enhanced conditioning)
- `scale > 3.0`: Stronger conditioning (may reduce diversity)
- `scale = 0.0`: Pure unconditional (NOT recommended)

### Generation Output Structure

```
output_dir/
├── class_0/
│   ├── synth_<uuid>_class0.png
│   ├── synth_<uuid>_class0.png
│   └── denoising_process_class0_<uuid>.png
├── class_1/
│   └── ...
└── pathmnist_synth_index.json
```

### JSON Index Format

Compatible with medsyn's training pipeline:

```json
{
  "PathMNIST": {
    "synth": {
      "0": {
        "image": "class_0/synth_abc123_class0.png",
        "label": 0,
        "is_synth": true,
        "uuid": "abc123"
      },
      ...
    }
  }
}
```

---

## Module Structure

```
medsyn/models/ccDDPM/
├── __init__.py
├── README.md                    # ← You are here
│
├── model.py                     # CCDDPM and ClassEmbedder
├── config.py                    # Configuration loader
├── loss.py                      # DDPMNoiseMSE with Min-SNR
├── metrics.py                   # PSNR, SSIM
├── training_logging.py          # CSV logger, EpochAverager
│
├── engine/
│   ├── train.py                 # Main training loop
│   └── predict.py               # Inference utilities
│
├── dataloaders/
│   ├── json.py                  # JSON-indexed dataloader
│   └── npz.py                   # NPZ-compressed dataloader
│
└── cli/
    ├── train_ccDDPM.py          # Training CLI entry point
    └── generate_ccDDPM.py       # Generation CLI entry point
```

### Key Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `model.py` | Model architecture | `CCDDPM`, `ClassEmbedder` |
| `engine/train.py` | Training loop | `train()`, diagnostics |
| `cli/generate_ccDDPM.py` | Generation | `generate_with_cfg()` |
| `loss.py` | Loss function | `DDPMNoiseMSE` with Min-SNR |
| `config.py` | Config parsing | `load_cfg()` |

---

## Troubleshooting

### Common Issues

#### 1. **All generated images are noise**

**Possible causes**:
- ❌ `guidance_scale: 0.0` in config → Set to `2.0`
- ❌ Model trained incorrectly → Check diagnostics in checkpoint
- ❌ Wrong `prediction_type` → Must be `epsilon`

**Solution**:
```bash
# Verify config
grep "guidance_scale" config/medsyn_cfg.yaml
# Should show: guidance_scale: 2.0

# Check checkpoint
python -c "import torch; print(torch.load('best.pt')['diagnostics'])"
```

#### 2. **Training loss not decreasing**

**Symptoms**: Loss stuck at ~0.1-0.2, correlation > 0.7

**Possible causes**:
- ❌ Gradient clipping too aggressive (`grad_clip_norm: 1.0`)
- ❌ Model architecture mismatch when resuming
- ❌ Learning rate too low

**Solution**:
```yaml
# In config
grad_clip_norm: 10.0  # Not 1.0!
```

#### 3. **CUDA Out of Memory**

**Solution**:
```yaml
# Reduce batch size
batch_size: 2  # or 1

# Or use gradient accumulation (not implemented)
# Or train on CPU (very slow)
```

#### 4. **All classes generate identical images**

**Cause**: Model not learning class conditioning

**Diagnostics to check**:
- Conditioning gap = 0 → Labels not affecting model
- All class losses equal → Class embedding not being used

**Solution**: Retrain from scratch with correct config

#### 5. **Checkpoint loading fails**

**Error**: `Missing keys` or `Unexpected keys`

**Cause**: Architecture mismatch

**Solution**:
```python
# Check saved config
ckpt = torch.load('checkpoint.pt')
print(ckpt['cfg']['class_embed_dim'])  # Must match current config!
```

---

## Advanced Topics

### Custom Prediction Types (Advanced)

⚠️ **Warning**: Changing `prediction_type` requires code modifications!

To use `v_prediction` instead of `epsilon`:

1. **Update loss function** (`loss.py`):
```python
# Compute v target: v = √ᾱ_t ε - √(1-ᾱ_t) x₀
v_true = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * x0
loss = F.mse_loss(pred, v_true)  # NOT F.mse_loss(pred, noise)!
```

2. **Update diagnostics** (`train.py`):
   - Modify x0 reconstruction formulas
   - Update variable names (`v_pred` instead of `eps_pred`)

3. **Thorough testing** required!

### Hyperparameter Tuning

**Key hyperparameters** to tune for your dataset:

| Parameter | Default | Tuning Range | Impact |
|-----------|---------|--------------|--------|
| `guidance_scale` | 2.0 | 1.5-3.0 | Quality vs diversity |
| `grad_clip_norm` | 10.0 | 5.0-20.0 | Training stability |
| `min_snr_gamma` | 5.0 | 2.0-10.0 | Timestep balance |
| `class_embed_dim` | 32 | 16-64 | Conditioning capacity |
| `lr` | 2e-4 | 1e-4 to 5e-4 | Convergence speed |

---

## References

### Papers

1. **Ho et al. (2020)**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. **Ho & Salimans (2022)**: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
3. **Hang et al. (2023)**: [Efficient Diffusion Training via Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556)
4. **Dhariwal & Nichol (2021)**: [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
5. **Karras et al. (2022)**: [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)

### Documentation

- [Training Fixes & Diagnostics](../../../docs/TRAINING_FIXES_AND_DIAGNOSTICS.md)
- [Generation Guide](../../../docs/CCDDPM_GENERATION_GUIDE.md)
- [Output Structure](../../../docs/CCDDPM_OUTPUT_STRUCTURE.md)
- [Bug Fix Log](../../../docs/CCDDPM_CODE_PATCHES_APPLIED.md)

### External Resources

- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion)
- [DDPM Tutorial](https://nn.labml.ai/diffusion/ddpm/index.html)

---

## Credits

**Implementation**: M. Pascual-González
**Framework**: PyTorch + HuggingFace Diffusers
**Dataset**: PathMNIST from MedMNIST

---

## License

See main repository LICENSE

---

**Last Updated**: 2025-10-29
**Version**: 2.0 (All critical bugs fixed)
**Status**: ✅ Production Ready
