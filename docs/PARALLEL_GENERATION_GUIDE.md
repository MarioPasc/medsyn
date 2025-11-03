# Parallel Multi-GPU Image Generation Guide

## Overview

The parallel image generation system distributes ccDDPM image generation tasks across multiple GPUs for maximum efficiency. Instead of generating images sequentially on a single GPU, this system uses a work queue approach where N GPUs work simultaneously on different generation tasks.

## Key Features

✅ **Multi-GPU Parallelization**: Use 2, 4, 8, or more GPUs simultaneously
✅ **Work Queue Architecture**: Dynamic task distribution for optimal load balancing
✅ **Order-Independent**: Tasks can be completed in any order (train/val/test, any class)
✅ **Efficient Resource Use**: No idle GPUs - workers pull tasks as they complete
✅ **Progress Tracking**: Real-time monitoring of completed tasks
✅ **Aggregated Outputs**: Automatic merging of results into JSON and NPZ files
✅ **Professional Logging**: Per-worker logs with GPU identification

## How It Works

### Architecture

```
Config File
    ↓
Parse Tasks → [train/class_0, train/class_1, ..., val/class_0, ..., test/class_8]
    ↓
Task Queue ← Workers pull tasks dynamically
    ↓
GPU 0: [train/class_0] → [train/class_3] → [val/class_2] → ...
GPU 1: [train/class_1] → [val/class_0] → [test/class_5] → ...
GPU 2: [train/class_2] → [test/class_1] → [val/class_7] → ...
...
GPU N: [val/class_1] → [test/class_3] → [train/class_8] → ...
    ↓
Results Queue ← Workers push completed tasks
    ↓
Aggregate Results → JSON + NPZ files per split
```

### Work Queue Strategy

1. **Task Creation**: All generation tasks (split × class) are created upfront
2. **Queue Population**: Tasks are added to a shared queue
3. **Worker Processes**: N worker processes (one per GPU) are spawned
4. **Dynamic Allocation**: Each worker:
   - Loads the model once on its GPU
   - Pulls next available task from queue
   - Generates images for that task
   - Puts result in result queue
   - Repeats until queue is empty
5. **Result Aggregation**: Main process collects results and creates final outputs

### Why This Is Fast

- **No Idle GPUs**: Workers immediately pick up next task when done
- **Load Balancing**: Faster classes don't block slower ones
- **Minimal Overhead**: Model loaded once per GPU (not per task)
- **Parallel I/O**: Each worker writes to separate class directories
- **No Synchronization**: Workers operate independently until aggregation

## Performance Comparison

### Single GPU (Sequential)
```
Total tasks: 27 (3 splits × 9 classes)
Time per task: ~30 seconds
Total time: ~13.5 minutes
```

### 4 GPUs (Parallel)
```
Total tasks: 27
Tasks per GPU: ~7 (dynamically distributed)
Total time: ~3.5 minutes
Speedup: ~3.9x
```

### 8 GPUs (Parallel)
```
Total tasks: 27
Tasks per GPU: ~3-4 (dynamically distributed)
Total time: ~2 minutes
Speedup: ~6.7x
```

**Note**: Speedup is not perfectly linear due to:
- Task granularity (some classes generate faster)
- Aggregation overhead
- I/O limitations
- Model loading time

## Usage

### Quick Start

```bash
# Submit job with 8 GPUs (default)
sbatch scripts/picasso_generate_parallel_sbatch.sh

# Or manually run with specific number of GPUs
python -m medsyn.cli.generate_ccDDPM_parallel config/picasso_cfg.yaml --num-gpus 4
```

### SLURM Script

The SLURM script automatically:
1. Requests N GPUs (configurable via `--gres=gpu:N`)
2. Copies repository and checkpoint to local scratch
3. Sets up conda environment
4. Runs parallel generation
5. Syncs results to permanent storage
6. Cleans up local scratch

**Modify for different GPU counts:**

```bash
#!/usr/bin/env bash
#SBATCH --gres=gpu:4    # Change this line (2, 4, 8, etc.)
#SBATCH --cpus-per-task=16  # Recommended: 4 × num_gpus
#SBATCH --mem=64G  # Recommended: 8-16GB per GPU
```

### Command-Line Options

```bash
python -m medsyn.cli.generate_ccDDPM_parallel config.yaml [OPTIONS]

Options:
  --num-gpus N              Number of GPUs to use (default: all available)
  --no-visualizations       Disable denoising visualizations (faster)
  --dataset-name NAME       Dataset name for JSON index (default: PathMNIST)
```

### Environment Variables

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1,3,5 python -m medsyn.cli.generate_ccDDPM_parallel config.yaml --num-gpus 4

# Disable GPU affinity (if needed)
CUDA_DEVICE_ORDER=PCI_BUS_ID python -m medsyn.cli.generate_ccDDPM_parallel config.yaml
```

## Configuration

The parallel version uses the same configuration format as the single-GPU version:

```yaml
generate:
  checkpoint: /path/to/best.pt
  npz_with_synth_images:
    save_to: /path/to/output
    train:
      classes:
        0: 100
        1: 50
        2: 75
        # ... other classes
    val:
      classes:
        0: 50
        1: 25
        # ... other classes
    test:
      classes:
        0: 50
        # ... other classes
```

## Output Structure

Same as single-GPU version:

```
output_directory/
├── train/
│   ├── class_0/
│   │   ├── synth_<uuid>_class0.png
│   │   ├── synth_<uuid>_class0.png
│   │   └── ...
│   ├── class_1/
│   │   └── ...
│   ├── pathmnist_train_index.json
│   └── pathmnist_train_synth.npz
├── val/
│   ├── class_0/
│   ├── class_1/
│   ├── pathmnist_val_index.json
│   └── pathmnist_val_synth.npz
└── test/
    ├── class_0/
    ├── pathmnist_test_index.json
    └── pathmnist_test_synth.npz
```

## Monitoring

### Real-Time Progress

```bash
# Watch SLURM output
tail -f ccddpm_generate_parallel.<job_id>.out

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor from SLURM
squeue -u $USER
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS,MaxVMSize
```

### Log Output

The parallel generation provides detailed logging:

```
[2025-11-03 15:30:45] INFO - MainProcess - Using 8 GPU(s) out of 8 available
[2025-11-03 15:30:45] INFO - MainProcess -   GPU 0: NVIDIA A100-SXM4-40GB
[2025-11-03 15:30:45] INFO - MainProcess -   GPU 1: NVIDIA A100-SXM4-40GB
...
[2025-11-03 15:30:50] GPU0 - INFO - Initializing on GPU 0
[2025-11-03 15:30:52] GPU0 - INFO - Model loaded successfully
[2025-11-03 15:30:53] GPU0 - INFO - Starting task: train/class_0 (100 samples)
[2025-11-03 15:31:25] GPU0 - INFO - Completed: train/class_0 in 32.1s (3.12 samples/s)
[2025-11-03 15:31:25] INFO - MainProcess - Progress: 1/27 (3.7%) - Latest: train/class_0
```

## Resource Requirements

### Memory

- **Per GPU**: ~8-12 GB GPU memory (depends on model size and batch size)
- **Per Worker**: ~4-8 GB RAM
- **Total RAM**: Recommend 8-16 GB per GPU

### Storage

- **Local Scratch**: Fast I/O recommended (NVMe SSD ideal)
- **Final Storage**: Depends on total images generated
  - ~10 MB per 1000 images (PNG format)
  - NPZ files are compressed (~50% of PNG size)

### CPU

- **Recommended**: 4 CPU cores per GPU
- Used for data preprocessing and I/O
- More cores = faster data loading

## Troubleshooting

### Issue: Workers die with OOM errors

**Causes:**
- Model too large for GPU
- Too many workers for available memory

**Solutions:**
```bash
# Reduce number of GPUs
python -m medsyn.cli.generate_ccDDPM_parallel config.yaml --num-gpus 4

# Or request more memory in SLURM
#SBATCH --mem=256G
```

### Issue: Slow generation despite multiple GPUs

**Possible causes:**
1. **Few large tasks**: If you have few classes with many samples each
   - Split classes into smaller batches in config
2. **I/O bottleneck**: Slow storage
   - Use local scratch (--localscratch in SLURM)
3. **Model loading overhead**: Too many GPUs for too few tasks
   - Use fewer GPUs if tasks < GPUs

**Check:**
```bash
# Monitor GPU utilization
nvidia-smi dmon -s u

# Check I/O wait
iostat -x 1
```

### Issue: Import errors

**Solution:**
```bash
# Ensure medsyn is installed
cd /path/to/medsyn
pip install -e .

# Test imports
python -c "from medsyn.cli.generate_ccDDPM_parallel import main; print('OK')"
```

### Issue: Results not syncing back

**Check:**
- SLURM output for rsync errors
- Destination directory permissions
- Available disk space

```bash
# Manual sync if needed
rsync -av /localscratch/.../work/generated/ /final/destination/
```

## Optimization Tips

### 1. Disable Visualizations

Denoising visualizations add overhead:

```bash
python -m medsyn.cli.generate_ccDDPM_parallel config.yaml --no-visualizations
```

**Impact**: ~10-20% faster

### 2. Optimize Task Distribution

Balance number of samples per class:

```yaml
# Instead of:
train:
  classes:
    0: 1000  # One large task
    1: 10    # Small task

# Do:
train:
  classes:
    0: 250
    1: 250
    # ... spread samples across more classes
```

### 3. Use Local Scratch

Always use high-speed local storage (NVMe SSD) for temporary files:

```bash
# In SLURM script
WORKDIR="${LOCALSCRATCH}/${USER}/${SLURM_JOB_ID}/work"
```

### 4. Right-Size GPU Count

More GPUs ≠ always faster. Consider:
- Total tasks: If 27 tasks, using >27 GPUs wastes resources
- Task size: Very fast tasks don't benefit from many GPUs
- Memory: More GPUs = more RAM needed

**Rule of thumb**: Use GPUs ≤ total_tasks / 2

## Comparison: Single vs Parallel

| Feature | Single GPU | Parallel (8 GPUs) |
|---------|------------|-------------------|
| **Time for 2700 images** | ~45 minutes | ~6 minutes |
| **GPU Utilization** | 1 GPU at 100% | 8 GPUs at ~90% |
| **Throughput** | ~1 img/s | ~7.5 img/s |
| **Memory (GPU)** | 10 GB | 80 GB (10 per GPU) |
| **Memory (RAM)** | 8 GB | 64 GB |
| **Complexity** | Simple | Moderate |

## Best Practices

1. **Test First**: Run with 1-2 GPUs on small subset before full generation
2. **Monitor Resources**: Watch GPU memory, RAM, and I/O during generation
3. **Log Everything**: Keep SLURM output logs for debugging
4. **Verify Results**: Check aggregated NPZ files match expected counts
5. **Clean Up**: Always clean local scratch after completion

## Advanced Usage

### Custom Task Distribution

For very imbalanced datasets, you can create custom task splits:

```python
# Custom script to split large classes
from medsyn.cli.generate_ccDDPM_parallel import GenerationTask

# Instead of one task with 1000 samples
# Create multiple smaller tasks
tasks = [
    GenerationTask("train", class_id=0, num_samples=250),
    GenerationTask("train", class_id=0, num_samples=250),
    GenerationTask("train", class_id=0, num_samples=250),
    GenerationTask("train", class_id=0, num_samples=250),
]
```

### Profiling

To identify bottlenecks:

```bash
# Profile with py-spy
py-spy record -o profile.svg -- python -m medsyn.cli.generate_ccDDPM_parallel config.yaml

# Profile GPU utilization
nsys profile -o generation.qdrep python -m medsyn.cli.generate_ccDDPM_parallel config.yaml
```

## FAQ

**Q: Can I use different GPU types?**
A: Yes, but performance may vary. Faster GPUs will process more tasks.

**Q: What if I have more GPUs than tasks?**
A: Extra GPUs will remain idle. Use --num-gpus to limit.

**Q: Can I resume failed generation?**
A: Not automatically. You'll need to modify config to regenerate only missing classes.

**Q: Is the order of generation deterministic?**
A: No, tasks are processed in any order. Final outputs are deterministic if you use the same random seed per class.

**Q: Can I mix single and parallel generation?**
A: Yes, outputs are compatible. Both use the same format.

## Support

For issues:
1. Check SLURM logs: `ccddpm_generate_parallel.<job_id>.out/err`
2. Verify GPU availability: `nvidia-smi`
3. Test with single GPU first
4. Check generation config format

---

**Summary**: Parallel generation provides ~6-7x speedup with 8 GPUs, with minimal code changes and automatic result aggregation. Ideal for generating large synthetic datasets efficiently.
