# Quick Reference: medsyn_cfg.yaml Structure

## Complete Example Configuration

```yaml
# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
data:
  # Dataset selection and basic parameters
  flag: pathmnist                    # Dataset type (currently only pathmnist)
  size: 64                           # Image size (will resize from 28 if needed)
  download_dir: /path/to/data_raw   # Where original MedMNIST NPZ is stored
  seed: 23102003                     # Random seed for stratification
  num_workers: 8                     # Number of workers for data loading
  
  # Data reduction/stratification settings
  reduction:
    strategy: fraction               # "fraction" or "max_per_class"
    train: 1.00                      # Fraction of train data to use (1.0 = all)
    val: 1.00                        # Fraction of val data to use
    test: 1.00                       # Fraction of test data to use
    max_per_class: null              # Max samples per class (if strategy="max_per_class")
  
  # PNG extraction settings (for JSON dataloader and YOLO dataset)
  save_png:
    enabled: true                    # Enable/disable PNG extraction
    processed_dir: /path/to/pngs    # Where PNG files are saved
    index_json: /path/to/index.json # Where JSON index is saved
    yolo_folder_dataset: /path/to/yolo  # Optional: YOLO classification dataset path
  
  # NPZ postprocessing settings (for NPZ dataloader)
  postprocess_npz:
    enabled: true                    # Enable/disable custom NPZ creation
    npz_path: /path/to/custom.npz   # Output path for custom NPZ with splits

# ==============================================================================
# bVAE MODEL CONFIGURATION
# ==============================================================================
bVAE:
  model:
    in_channels: 3
    img_size: 64
    latent_dim: 32
    base_channels: 64
    num_down: 4
    decoder_sigmoid: true
    num_classes: 9
    conditioning: "film"
    decoder_conditioning: false
    class_embed_dim: 64
    use_class_prior: false
  
  train:
    epochs: 100
    batch_size: 16
    num_workers: 4
    device: "cuda"
    mixed_precision: true
    grad_clip_norm: 5.0
    seed: 17
    output_dir: /path/to/bVAE/outputs
    save_every_epoch: 10
    encoder_extra_steps: 2
    encoder_heavy_epochs: 5
  
  optim:
    optimizer: "adamw"
    lr_init: 0.0001
    weight_decay: 0.0001
    betas: [0.9, 0.999]
    eps: 1.0e-8
  
  sched:
    use_onecycle: true
    max_lr: 0.0001
    pct_start: 0.5
    div_factor: 10.0
    final_div_factor: 1000.0
  
  loss:
    recon_type: "l1"
    l1_weight: 1.0
    ssim_weight: 0.85
    beta: 1.0
    kld_weight: 1.0
    recon_weight: 1.0
    free_bits_nats: 0.5
    beta_schedule:
      type: "cyclical"
      beta_min: 0.0
      beta_max: 0.3
      cycles: 4
      ratio_increase: 0.7
    capacity:
      use: false
      C_max: 1.0
      steps_to_max: 70000
      gamma: 50.0
    prior_reg_w: 0.0001
    infovae_mmd:
      use: false
      weight: 0.5
      kernel: "rbf"
    per_class_mmd_use: false
    per_class_mmd_w: 1.0

# ==============================================================================
# ccDDPM MODEL CONFIGURATION
# ==============================================================================
ccddpm:
  # Dataloader selection (reads NPZ path from data.postprocess_npz.npz_path)
  dataloader:
    type: npz                        # "json" or "npz"
  
  train:
    image_size: 64
    in_channels: 3
    class_embed_dim: 16
    num_classes: 9
    batch_size: 4
    epochs: 80
    mixed_precision: true
    guidance_p_uncond: 0.1           # Classifier-free guidance dropout probability
    ema_use: true
    ema_decay: 0.999
    ckpt_every_epochs: 10            # Save checkpoint every N epochs
    patience: 15                     # Early stopping patience
    output_dir: /path/to/ccddpm/outputs
  
  optim:
    lr: 2.0e-4
    wd: 0.0
  
  sched:
    num_train_timesteps: 1000
    beta_start: 1.0e-4
    beta_end: 2.0e-2
    beta_schedule: squaredcos_cap_v2  # "linear" or "squaredcos_cap_v2"
    prediction_type: epsilon          # "epsilon" or "v_prediction"
  
  infer:
    guidance_scale: 2.0              # Classifier-free guidance scale (0=unconditional)
    num_inference_steps: 1000        # Number of denoising steps
    out_dir: /path/to/ccddpm/samples

# ==============================================================================
# GENERATION CONFIGURATION (for bVAE generation)
# ==============================================================================
generate:
  checkpoint: /path/to/bVAE/best.pt  # Checkpoint to use for generation
  classes:                            # Number of samples per class
    0: 100
    1: 100
    2: 100
    3: 100
    4: 100
    5: 100
    6: 100
    7: 100
    8: 100
```

## Common Configuration Patterns

### Pattern 1: Development (Local Machine, Full Features)
```yaml
data:
  size: 64
  reduction:
    train: 0.2  # Use 20% for faster testing
  save_png:
    enabled: true
  postprocess_npz:
    enabled: true

ccddpm:
  dataloader:
    type: json  # Use JSON for flexibility
  train:
    batch_size: 4
    epochs: 10  # Quick test
```

### Pattern 2: Supercomputer (NPZ Only, Full Dataset)
```yaml
data:
  size: 64
  reduction:
    train: 1.0  # Use all data
  save_png:
    enabled: false  # Don't need PNGs
  postprocess_npz:
    enabled: true   # Only NPZ

ccddpm:
  dataloader:
    type: npz      # Use NPZ for performance
  train:
    batch_size: 32
    epochs: 100
```

### Pattern 3: YOLO Training (PNG Only)
```yaml
data:
  save_png:
    enabled: true
    yolo_folder_dataset: /path/to/yolo
  postprocess_npz:
    enabled: false  # Don't need NPZ for YOLO

# Don't train ccDDPM, just use for YOLO classification
```

### Pattern 4: Quick Config Test (No Data Processing)
```yaml
data:
  save_png:
    enabled: false
  postprocess_npz:
    enabled: false

# Just downloads original data, good for config validation
```

## Field-by-Field Reference

### data.reduction.strategy
- `"fraction"`: Use `train/val/test` fractions (0.0-1.0)
- `"max_per_class"`: Use `max_per_class` limit per class

### data.save_png.enabled
- `true`: Export PNGs and create JSON index (required for JSON dataloader)
- `false`: Skip PNG extraction (use for NPZ-only or testing)

### data.postprocess_npz.enabled
- `true`: Create custom NPZ with splits (required for NPZ dataloader)
- `false`: Skip NPZ creation (use for JSON-only or testing)

### ccddpm.dataloader.type
- `"json"`: Load from PNG files using JSON index (slower, more flexible)
- `"npz"`: Load from compressed NPZ file (faster, supercomputer-friendly)
  - Automatically reads path from `data.postprocess_npz.npz_path`

### ccddpm.sched.beta_schedule
- `"linear"`: Linear noise schedule (DDPM original)
- `"squaredcos_cap_v2"`: Cosine schedule (better for high-res images)

### ccddpm.infer.guidance_scale
- `0.0`: Unconditional generation (no class guidance)
- `1.0-3.0`: Recommended range for class-conditional generation
- Higher values = stronger class conditioning

## Validation Checklist

Before running `medsyn-prepare-data`:
- [ ] `data.download_dir` exists or can be created
- [ ] If `save_png.enabled=true`: paths for `processed_dir` and `index_json` are valid
- [ ] If `postprocess_npz.enabled=true`: path for `npz_path` is valid
- [ ] `reduction` strategy matches your needs (fraction vs max_per_class)

Before running `ccddpm-train`:
- [ ] If `dataloader.type=json`: `save_png.enabled=true` was used in data prep
- [ ] If `dataloader.type=npz`: `postprocess_npz.enabled=true` was used in data prep
- [ ] Batch size fits in GPU memory
- [ ] Output directory is writable

## Tips

1. **Start Small**: Use `reduction.train: 0.1` for quick testing
2. **Choose One Dataloader**: Enable only the data format you need (save space)
3. **NPZ for Production**: Use NPZ dataloader on supercomputers for best I/O performance
4. **JSON for Development**: Use JSON dataloader for flexibility during development
5. **YOLO Symlinks**: YOLO dataset uses symlinks (fast, no duplication)
