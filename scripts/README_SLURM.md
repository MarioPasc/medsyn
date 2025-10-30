# SLURM Scripts for ccDDPM Training

This directory contains SLURM job scripts for training the class-conditioned DDPM model on HPC clusters.

## Available Scripts

### 1. `picasso_sbatch.sh` - Single GPU Training
**Description**: Original script for single-GPU training.

**Resources**:
- 1 GPU
- 8 CPUs
- 64GB RAM
- 8 hour time limit

**Usage**:
```bash
sbatch scripts/picasso_sbatch.sh
```

---

### 2. `picasso_parallel_job.sh` - Multi-GPU Distributed Training ⚡
**Description**: Distributed training script using PyTorch DDP (DistributedDataParallel).

**Resources**:
- **2 GPUs** (configurable)
- 16 CPUs
- 128GB RAM
- 8 hour time limit

**Usage**:
```bash
sbatch scripts/picasso_parallel_job.sh
```

## Key Differences: Single-GPU vs Multi-GPU

| Feature | Single-GPU | Multi-GPU (DDP) |
|---------|------------|-----------------|
| **Launch command** | `srun python -m medsyn.cli.train_ccDDPM` | `torchrun --nproc_per_node=N -m medsyn.cli.train_ccDDPM` |
| **Config modification** | None | Automatically enables `dist.enabled: true` |
| **Speedup** | 1x | ~1.8x with 2 GPUs, ~3.5x with 4 GPUs |
| **Memory per GPU** | Full batch | Batch split across GPUs |
| **Results location** | `.../PathMNIST_ccDDPM` | `.../PathMNIST_ccDDPM_parallel` |

## Customizing for More GPUs

To use a different number of GPUs (e.g., 4 GPUs), edit `picasso_parallel_job.sh`:

### Step 1: Change SLURM directives
```bash
#SBATCH --gres=gpu:4          # Request 4 GPUs
#SBATCH --cpus-per-task=32    # Scale CPUs accordingly (16 per GPU)
#SBATCH --mem=256G             # Scale memory accordingly (64GB per GPU)
```

### Step 2: Change the hardcoded variable
```bash
# DISTRIBUTED TRAINING CONFIGURATION
NUM_GPUS=4  # Change from 2 to 4
```

### Step 3: Update CUDA_VISIBLE_DEVICES (optional)
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Add all GPU indices
```

That's it! The script will automatically:
- ✅ Configure the YAML to use 4 GPUs
- ✅ Launch torchrun with `--nproc_per_node=4`
- ✅ Sync data properly across all processes

## How the Multi-GPU Script Works

### 1. **Config Modification**
Before training, the script modifies `config/picasso_cfg.yaml`:
```yaml
ccddpm:
  dist:
    enabled: true    # Changed from false
    num_gpus: 2      # Set to NUM_GPUS variable
```

The original config is backed up and restored after training.

### 2. **Distributed Launch with torchrun**
```bash
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  -m medsyn.cli.train_ccDDPM \
  --config config/picasso_cfg.yaml \
  --dataset ${DATA_DST} \
  --outdir ${OUT_DIR}
```

`torchrun` automatically sets up:
- `RANK` (process ID: 0, 1)
- `WORLD_SIZE` (total processes: 2)
- `LOCAL_RANK` (GPU ID: 0, 1)
- `MASTER_ADDR` and `MASTER_PORT` (for communication)

### 3. **Automatic Synchronization**
The training code automatically:
- ✅ Splits batches across GPUs
- ✅ Synchronizes gradients
- ✅ Averages metrics across GPUs
- ✅ Saves checkpoints only on rank-0
- ✅ Coordinates early stopping

## Monitoring Training

### View logs in real-time:
```bash
# Standard output
tail -f ccddpm_parallel.<job_id>.out

# Error output
tail -f ccddpm_parallel.<job_id>.err
```

### Check GPU utilization:
```bash
# SSH to the compute node (get node name from squeue)
squeue -u $USER
ssh <node_name>
watch -n 1 nvidia-smi
```

## Performance Expectations

### Expected Speedup (batch_size=4 per GPU)

| GPUs | Effective Batch | Speedup | Training Time (100 epochs) |
|------|----------------|---------|---------------------------|
| 1    | 4              | 1.0x    | ~8 hours                  |
| 2    | 8              | 1.8x    | ~4.5 hours                |
| 4    | 16             | 3.5x    | ~2.3 hours                |
| 8    | 32             | 6.5x    | ~1.2 hours                |

*Note: Speedup is sub-linear due to communication overhead between GPUs.*

### Batch Size Considerations

The `batch_size` in the config is **per-GPU**:
- Single-GPU: `batch_size=4` → effective batch = 4
- 2-GPU DDP: `batch_size=4` → effective batch = 8
- 4-GPU DDP: `batch_size=4` → effective batch = 16

**Important**: Larger effective batch sizes may require learning rate adjustments!

## Troubleshooting

### Issue: Training hangs at initialization
**Cause**: NCCL communication issues between GPUs

**Solution**:
1. Check that NCCL is working:
   ```bash
   python -c "import torch; torch.distributed.init_process_group(backend='nccl')"
   ```
2. Try adding to the script:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_SOCKET_IFNAME=eth0  # Or your network interface
   ```

### Issue: Out of memory errors
**Cause**: Batch size too large for GPU memory

**Solution**: Reduce `batch_size` in `config/picasso_cfg.yaml`:
```yaml
ccddpm:
  train:
    batch_size: 2  # Reduce from 4 to 2
```

### Issue: Results not found after job completes
**Cause**: Output directory mismatch

**Solution**: Check the script's output log for the actual output path:
```bash
grep "Output directory:" ccddpm_parallel.<job_id>.out
```

### Issue: Different metrics between single-GPU and multi-GPU
**Cause**: This is expected due to different data shuffling and batch compositions

**Solution**: This is normal. Validation loss should converge to similar values, but exact numbers will differ.

## Advanced: Multi-Node Training (8+ GPUs)

For training across multiple nodes, you'll need to modify the script:

```bash
#SBATCH --nodes=2              # 2 nodes
#SBATCH --ntasks-per-node=1    # 1 task per node
#SBATCH --gres=gpu:4           # 4 GPUs per node = 8 total

# Get MASTER_ADDR from first node
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Launch with srun (not torchrun standalone)
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m medsyn.cli.train_ccDDPM \
  --config config/picasso_cfg.yaml
```

## Questions?

- **Single-GPU training**: Use `picasso_sbatch.sh` (simple, tested)
- **2-4 GPU training**: Use `picasso_parallel_job.sh` (faster, automatic)
- **8+ GPU training**: Contact your HPC admin for multi-node setup

For more details on distributed training, see:
- PyTorch DDP tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- torchrun documentation: https://pytorch.org/docs/stable/elastic/run.html
