# YOLO Classifier - Quick Start Guide

## TL;DR

Train two YOLO classifiers in parallel (real-only vs real+synth) with per-class AUC metrics:

```bash
# 1. Verify setup
python scripts/verify_yolo_setup.py

# 2. Submit parallel training job (2 GPUs)
sbatch scripts/train_yolo_parallel.sh

# 3. Monitor progress
tail -f logs/yolo_parallel_*.out
```

## What This Does

Trains **two YOLO classifiers simultaneously** on 2 GPUs:

1. **Model 1 (GPU 0)**: Real PathMNIST images only (`is_synth=0`)
2. **Model 2 (GPU 1)**: Real + Synthetic images balanced (`is_synth=0 OR is_synth=1`)

Both models will:
- ✅ Train for 100 epochs (configurable)
- ✅ Save best model checkpoint
- ✅ Compute per-class AUC metrics
- ✅ Generate comprehensive logs
- ✅ Save results to separate directories

## Results Location

```
/home/mpascual/research/medsyn/yolo_classifier/runs/
├── pathmnist_real_only_<timestamp>/
│   ├── weights/best.pt                    # ← Use this for inference
│   ├── per_class_auc_metrics.json         # ← Per-class AUC scores
│   └── training_<timestamp>.log           # ← Training log
└── pathmnist_real_synth_<timestamp>/
    └── (same structure)
```

## Single Training (Without SLURM)

```bash
# Train on real data only
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real \
    --name my_experiment \
    --device 0

# Train on real + synthetic data
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real_synth \
    --name my_experiment_synth \
    --device 0
```

## Key Files

- **SLURM script**: `scripts/train_yolo_parallel.sh`
- **CLI entry point**: `medsyn/cli/classify.py`
- **Config**: `config/medsyn_cfg.yaml`
- **Hyperparameters**: `config/yolo_hyperparameters.yaml`
- **Full documentation**: `docs/YOLO_CLASSIFIER_USAGE.md`

## Per-Class AUC Metrics

Automatically computed and saved to `per_class_auc_metrics.json`:

```json
{
  "class_0": 0.9512,
  "class_1": 0.9234,
  "class_2": 0.8876,
  ...
  "macro_avg_auc": 0.9187
}
```

## SLURM Job Management

```bash
# Submit job
sbatch scripts/train_yolo_parallel.sh

# Check job status
squeue -u $USER

# View output (live)
tail -f logs/yolo_parallel_<job_id>.out

# Cancel job
scancel <job_id>
```

## Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `real` | Real images only (is_synth=0) | Baseline performance |
| `real_synth` | Real + Synthetic (balanced) | Augmented training |
| `synth` | Synthetic only (is_synth=1) | Test synthetic quality |

## Configuration

Adjust training parameters in `config/yolo_hyperparameters.yaml`:

- `epochs: 100` - Number of training epochs
- `batch: 128` - Batch size (reduce if OOM)
- `lr0: 0.01` - Initial learning rate
- `patience: 100` - Early stopping patience

## Troubleshooting

```bash
# Verify setup
python scripts/verify_yolo_setup.py

# Check NPZ file path
grep "npz_path" config/medsyn_cfg.yaml

# Check SLURM logs for errors
cat logs/yolo_parallel_<job_id>.err

# Test single GPU training first
python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real \
    --device 0
```

## Expected Training Time

- **Per model**: ~2-4 hours (depending on GPU)
- **Parallel (2 GPUs)**: Same as single model (runs simultaneously)

## Next Steps After Training

1. **Check results**:
   ```bash
   ls -lh /home/mpascual/research/medsyn/yolo_classifier/runs/pathmnist_*
   ```

2. **View AUC metrics**:
   ```bash
   cat /home/mpascual/research/medsyn/yolo_classifier/runs/pathmnist_real_only_*/per_class_auc_metrics.json
   ```

3. **Load best model** (Python):
   ```python
   from ultralytics import YOLO
   model = YOLO('/path/to/runs/pathmnist_real_only_*/weights/best.pt')
   results = model.val()  # Validate
   ```

4. **Compare models**: Compare AUC metrics between real-only and real+synth

---

**For detailed documentation, see: `docs/YOLO_CLASSIFIER_USAGE.md`**
