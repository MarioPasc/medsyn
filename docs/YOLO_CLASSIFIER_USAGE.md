# YOLO Classifier Training Guide

This document explains how to use the YOLO classifier system for training on PathMNIST data with NPZ files.

## Overview

The YOLO classifier system supports:
- ✅ Training on NPZ files containing real and synthetic images
- ✅ Three training modes: real-only, real+synth (balanced), and synth-only
- ✅ Per-class AUC metrics computation and saving
- ✅ Professional logging to files and console
- ✅ Parallel training on multiple GPUs using SLURM
- ✅ Integration with ultralytics YOLO framework

## System Architecture

### Key Components

1. **Data Loader** (`medsyn/models/classifier/dataloaders.py`)
   - Loads images from NPZ files
   - Filters data based on `is_synth` flag
   - Supports train/val/test splits
   - Applies ultralytics transforms for augmentation

2. **Custom Trainer** (`medsyn/models/classifier/engine/yolo_trainer.py`)
   - Extends ultralytics ClassificationTrainer
   - Uses NPZ-backed dataloaders instead of folder-based

3. **Custom Validator** (`medsyn/models/classifier/engine/yolo_validator.py`)
   - Extends ultralytics ClassificationValidator
   - Computes per-class AUC metrics
   - Saves metrics to JSON file: `per_class_auc_metrics.json`
   - Logs AUC scores for each class and macro-average

4. **CLI Entry Point** (`medsyn/cli/classify.py`)
   - Professional logging setup
   - Command-line interface for training/validation
   - Support for training mode overrides

## Training Modes

The system supports three training modes controlled by the `training_images` parameter:

| Mode | CLI Flag | Config Value | Description |
|------|----------|--------------|-------------|
| Real Only | `--training_mode real` | `PathMNIST` | Uses only real images (is_synth=0) |
| Real + Synth | `--training_mode real_synth` | `PathMNIST_and_synth` | Uses both real and synthetic images (balanced) |
| Synth Only | `--training_mode synth` | `synth` | Uses only synthetic images (is_synth=1) |

## NPZ File Format

The NPZ file should contain the following arrays:

```
train_images: (N_train, C, H, W) or (N_train, H, W, C) - Training images
train_labels: (N_train,) - Training labels (class indices)
train_is_synth: (N_train,) - Binary flag: 0=real, 1=synthetic

val_images: (N_val, C, H, W) or (N_val, H, W, C) - Validation images
val_labels: (N_val,) - Validation labels
val_is_synth: (N_val,) - Binary flag: 0=real, 1=synthetic

test_images: (N_test, C, H, W) or (N_test, H, W, C) - Test images
test_labels: (N_test,) - Test labels
test_is_synth: (N_test,) - Binary flag: 0=real, 1=synthetic
```

## Configuration Files

### 1. Main Config (`config/medsyn_cfg.yaml`)

Key YOLO-related settings:

```yaml
data:
  postprocess_npz:
    npz_path: /path/to/your/PathMNIST.npz

yolo:
  model: yolo11n-cls.pt
  project: /home/mpascual/research/medsyn/yolo_classifier/runs
  name: pathmnist_npz
  imgsz: 64
  epochs: 100
  batch: 128
  workers: 8
  training_images: PathMNIST  # Default mode
```

### 2. Hyperparameters (`config/yolo_hyperparameters.yaml`)

Contains all YOLO training hyperparameters (learning rate, optimizer, augmentation, etc.)

## Usage

### Single Training Run

```bash
# Train on real data only
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real \
    --name pathmnist_real_only \
    --device 0

# Train on real + synthetic data
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real_synth \
    --name pathmnist_real_synth \
    --device 0

# Validation only
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --val_only \
    --device 0
```

### Parallel Training with SLURM

The SLURM script `scripts/train_yolo_parallel.sh` trains two models in parallel on 2 GPUs:

1. **Real-only model** (GPU 0): Trained on real PathMNIST images only
2. **Real+Synth model** (GPU 1): Trained on balanced real + synthetic images

**Submit the job:**

```bash
sbatch scripts/train_yolo_parallel.sh
```

**What it does:**

1. Requests 2 GPUs from SLURM
2. Creates timestamped experiment names
3. Launches two training processes in parallel
4. Each process writes to separate log files
5. Waits for both to complete
6. Reports success/failure status

**Output locations:**

```
logs/
  yolo_parallel_<job_id>.out          # SLURM stdout
  yolo_parallel_<job_id>.err          # SLURM stderr
  yolo_real_only_<timestamp>.log      # Real-only training log
  yolo_real_synth_<timestamp>.log     # Real+Synth training log

/home/mpascual/research/medsyn/yolo_classifier/runs/
  pathmnist_real_only_<timestamp>/
    weights/
      best.pt                          # Best checkpoint
      last.pt                          # Last checkpoint
    training_<timestamp>.log           # Training log from CLI
    per_class_auc_metrics.json         # Per-class AUC scores
    results.csv                        # Training metrics per epoch
    confusion_matrix.png               # Confusion matrix

  pathmnist_real_synth_<timestamp>/
    weights/
      best.pt
      last.pt
    training_<timestamp>.log
    per_class_auc_metrics.json
    results.csv
    confusion_matrix.png
```

## Per-Class AUC Metrics

After training/validation, the validator computes and saves per-class AUC metrics:

**File:** `per_class_auc_metrics.json`

```json
{
  "class_0": 0.9512,
  "class_1": 0.9234,
  "class_2": 0.8876,
  ...
  "class_8": 0.9101,
  "macro_avg_auc": 0.9187
}
```

**Metric Computation:**
- One-vs-rest AUC for each class
- Uses predicted probabilities (softmax outputs)
- Macro-average across all classes
- Handles missing classes gracefully

## Logging

The system uses professional logging with:

- **Timestamp** for each log entry
- **Logger name** to identify source
- **Log level** (INFO, WARNING, ERROR)
- **Dual output** to console and file

**Log format:**
```
2025-11-03 14:30:45 | medsyn.cli.classify | INFO | Starting training...
```

**Log files:**
- CLI creates: `<project>/<name>/training_<timestamp>.log`
- SLURM creates: `logs/yolo_real_only_<timestamp>.log`

## Command-Line Arguments

```
--config CONFIG         Path to medsyn_cfg.yaml (required)
--hparams HPARAMS       Path to yolo_hyperparameters.yaml (required)
--training_mode MODE    Override training mode: real, real_synth, synth
--name NAME             Override experiment name
--device DEVICE         Override device (e.g., '0', '1', 'cpu')
--val_only              Run validation only (no training)
--log_level LEVEL       Logging level: DEBUG, INFO, WARNING, ERROR
```

## Monitoring Training

**During training:**

```bash
# Watch SLURM output
tail -f logs/yolo_parallel_<job_id>.out

# Watch individual training logs
tail -f logs/yolo_real_only_<timestamp>.log
tail -f logs/yolo_real_synth_<timestamp>.log
```

**Check SLURM job status:**

```bash
squeue -u $USER
```

**Cancel job if needed:**

```bash
scancel <job_id>
```

## Troubleshooting

### Issue: "File does not exist: yolo_hyperparameters.yaml"
**Solution:** Check that `config/yolo_hyperparameters.yaml` exists

### Issue: "NPZ file not found"
**Solution:** Update `npz_path` in `config/medsyn_cfg.yaml`

### Issue: "CUDA out of memory"
**Solution:** Reduce `batch` size in `config/yolo_hyperparameters.yaml` or `config/medsyn_cfg.yaml`

### Issue: "No module named sklearn"
**Solution:** Install scikit-learn: `pip install scikit-learn`

### Issue: SLURM job fails immediately
**Solution:** Check logs in `logs/yolo_parallel_<job_id>.err`

## Performance Tips

1. **Batch Size:** Increase for faster training (if GPU memory allows)
2. **Workers:** Set to number of CPU cores available
3. **Mixed Precision:** Enabled by default (`amp: True`), speeds up training
4. **Patience:** Adjust early stopping patience in hyperparameters
5. **Learning Rate:** Tune `lr0` and `lrf` for better convergence

## Model Evaluation

After training, you can evaluate the model on test set:

```bash
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --val_only \
    --device 0
```

Or modify the validator to use test split instead of val split.

## Citation

If you use this code, please cite the Ultralytics YOLO framework:

```bibtex
@software{yolo,
  title = {Ultralytics YOLO},
  author = {Glenn Jocher and others},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}
```

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify configuration files are correct
3. Ensure NPZ file format matches expected structure
4. Check GPU availability with `nvidia-smi`
