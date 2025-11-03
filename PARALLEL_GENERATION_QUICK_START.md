# Parallel Image Generation - Quick Start

## TL;DR

Generate images 6-8x faster using multiple GPUs in parallel:

```bash
# Submit parallel generation job with 8 GPUs
sbatch scripts/picasso_generate_parallel_sbatch.sh
```

## What This Does

Instead of generating images sequentially on 1 GPU:
```
GPU 0: [class 0] → [class 1] → [class 2] → ... → [class 8]
Time: ~45 minutes for 2700 images
```

Parallel generation uses N GPUs simultaneously:
```
GPU 0: [class 0] → [class 3] → [class 6] → ...
GPU 1: [class 1] → [class 4] → [class 7] → ...
GPU 2: [class 2] → [class 5] → [class 8] → ...
...
Time: ~6 minutes for 2700 images (8 GPUs)
```

## Performance

| GPUs | Time (2700 images) | Speedup |
|------|-------------------|---------|
| 1    | ~45 min           | 1x      |
| 2    | ~23 min           | ~2x     |
| 4    | ~12 min           | ~3.8x   |
| 8    | ~6 min            | ~7.5x   |

## Quick Commands

```bash
# ===== SLURM (Recommended) =====

# 8 GPUs (default)
sbatch scripts/picasso_generate_parallel_sbatch.sh

# 4 GPUs (edit script first: #SBATCH --gres=gpu:4)
sbatch scripts/picasso_generate_parallel_sbatch.sh

# Monitor progress
tail -f ccddpm_generate_parallel.<job_id>.out

# ===== Manual (Local Machine) =====

# Use all available GPUs
python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml

# Use specific number of GPUs
python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 4

# Faster: disable visualizations
python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 4 --no-visualizations

# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,3 python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 3
```

## How It Works

1. **Creates task queue**: All (split, class) combinations become independent tasks
2. **Spawns N workers**: One worker process per GPU
3. **Dynamic allocation**: Workers pull tasks from queue as they complete
4. **Parallel generation**: All GPUs generate simultaneously
5. **Automatic aggregation**: Results merged into JSON + NPZ files

**Key advantage**: No idle GPUs! Workers immediately pick up next task when done.

## Configuration

Uses the same config as single-GPU version (`generate` section):

```yaml
generate:
  checkpoint: /path/to/best.pt
  npz_with_synth_images:
    save_to: /path/to/output
    train:
      classes:
        0: 100
        1: 50
        # ... more classes
    val:
      classes:
        0: 50
        # ... more classes
```

**No config changes needed!** The parallel version automatically distributes tasks.

## Output

Identical to single-GPU version:

```
output_directory/
├── train/
│   ├── class_0/, class_1/, ..., class_8/
│   ├── pathmnist_train_index.json
│   └── pathmnist_train_synth.npz
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

## Monitoring

```bash
# Watch progress
tail -f ccddpm_generate_parallel.<job_id>.out

# Check GPU usage
watch -n 1 nvidia-smi

# SLURM job status
squeue -u $USER
```

**Log output shows**:
- Number of GPUs being used
- Progress: X/Y tasks completed (%)
- Per-worker activity: which GPU is working on which class
- Throughput: samples/second
- Final stats: total time, speedup estimate

## SLURM Script Configuration

Edit `scripts/picasso_generate_parallel_sbatch.sh`:

```bash
# For 4 GPUs
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16  # 4 per GPU
#SBATCH --mem=64G            # 16GB per GPU

# For 8 GPUs
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32  # 4 per GPU
#SBATCH --mem=128G           # 16GB per GPU

# For 2 GPUs
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8   # 4 per GPU
#SBATCH --mem=32G            # 16GB per GPU
```

**Rule of thumb**:
- CPUs: 4 × num_gpus
- RAM: 16 GB × num_gpus
- Time: Divide single-GPU time by (num_gpus × 0.85)

## Troubleshooting

### Workers dying (OOM)
```bash
# Reduce number of GPUs
--num-gpus 4
```

### Slow despite multiple GPUs
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Should see ~90-100% utilization on all GPUs
# If low, check I/O (use local scratch)
```

### Import errors
```bash
# Ensure medsyn is installed
cd /path/to/medsyn && pip install -e .
```

### Results not found
```bash
# Check SLURM output for errors
cat ccddpm_generate_parallel.<job_id>.err

# Manually sync if needed
rsync -av /localscratch/.../generated/ /destination/
```

## Optimization Tips

1. **Disable visualizations**: Add `--no-visualizations` (10-20% faster)
2. **Use local scratch**: Fast I/O critical for multi-GPU
3. **Right-size GPU count**: More GPUs ≠ always faster
   - If 27 tasks total, using >27 GPUs wastes resources
   - Sweet spot: 4-8 GPUs for typical workloads
4. **Balance task sizes**: Avoid one huge class + many tiny classes

## When To Use

**Use Parallel:**
- ✅ Generating >1000 images
- ✅ Multiple splits (train/val/test)
- ✅ Multiple GPUs available
- ✅ Need results quickly

**Use Single-GPU:**
- ✅ Generating <500 images
- ✅ Only 1 GPU available
- ✅ Prototyping/testing
- ✅ Debugging generation issues

## Comparison: Single vs Parallel Scripts

| Feature | Single GPU | Parallel (8 GPUs) |
|---------|------------|-------------------|
| **Script** | `generate_ccDDPM.py` | `generate_ccDDPM_parallel.py` |
| **SLURM** | `picasso_generate_sbatch.sh` | `picasso_generate_parallel_sbatch.sh` |
| **Time** | ~45 min | ~6 min |
| **Config** | Same format | Same format |
| **Output** | Same format | Same format |
| **Complexity** | Simple | Moderate |

## Examples

### Example 1: Quick Test (2 GPUs)

```bash
# Edit config to generate small subset
# generate.npz_with_synth_images.train.classes: {0: 10, 1: 10}

python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 2
```

### Example 2: Full Generation (8 GPUs, SLURM)

```bash
# Submit job
sbatch scripts/picasso_generate_parallel_sbatch.sh

# Get job ID
JOB_ID=$(squeue -u $USER -o "%.18i" -h | head -1)

# Monitor
tail -f ccddpm_generate_parallel.${JOB_ID}.out
```

### Example 3: Specific GPUs

```bash
# Use GPUs 0, 2, 4, 6 (skip odd-numbered)
CUDA_VISIBLE_DEVICES=0,2,4,6 python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 4
```

## Next Steps

After generation completes:

1. **Verify outputs**:
   ```bash
   ls -lh /path/to/output/{train,val,test}/*.npz
   ```

2. **Check sample counts**:
   ```python
   import numpy as np
   data = np.load('/path/to/output/train/pathmnist_train_synth.npz')
   print(f"Train samples: {len(data['train_images'])}")
   print(f"All synthetic: {data['train_is_synth'].all()}")
   ```

3. **Use for training**: Update YOLO config to use real+synth mode

---

**For full documentation**: See `docs/PARALLEL_GENERATION_GUIDE.md`

**Key takeaway**: Same config, same output, 6-8x faster with multiple GPUs! 🚀
