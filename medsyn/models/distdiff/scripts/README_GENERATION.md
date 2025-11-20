# SLURM Generation Script Guide

## Overview

The `picasso_generate_sbatch.sh` script runs the ccDDPM image generation process on a SLURM cluster. It follows the same structure as the training script but is optimized for the generation workflow.

## Prerequisites

1. **Trained checkpoint**: You need a trained ccDDPM model checkpoint (`best.pt`)
2. **Configuration file**: `config/picasso_cfg.yaml` with the `generate` section configured
3. **Conda environment**: The `medsyn` environment must be pre-created on the cluster
4. **Sufficient resources**: Generation requires 1 GPU, ~32GB RAM, and ~2 hours

## Configuration

### 1. Update Script Paths

Edit `scripts/picasso_generate_sbatch.sh` and modify these paths:

```bash
# Path to the repository
REPO_SRC="/path/to/your/medsyn/repo"

# Path to trained checkpoint
CHECKPOINT_SRC="/path/to/your/results/PathMNIST_ccDDPM/ckpts/best.pt"

# Where to store generated images
RESULTS_DST="/path/to/save/generated/images"
```

### 2. Configure Generation in YAML

Edit `config/picasso_cfg.yaml` to specify how many images to generate per class and split:

```yaml
generate:
  model_type: ccddpm
  checkpoint: ${OUTPUT_DIR}/ckpts/best.pt  # Will be overridden by script
  npz_with_synth_images:
    save_to: ${OUTPUT_DIR}/synth
    train:
      classes:
        0: 100  # Generate 100 images for class 0 in train split
        1: 100
        2: 100
        # ... for all classes
    val:
      classes:
        0: 50   # Generate 50 images for class 0 in val split
        1: 50
        # ... for all classes
```

## Usage

### Submit the Job

```bash
cd /path/to/medsyn
sbatch scripts/picasso_generate_sbatch.sh
```

### Monitor the Job

```bash
# Check job status
squeue -u $USER

# Watch the output log in real-time
tail -f ccddpm_generate.<JOB_ID>.out

# Check for errors
tail -f ccddpm_generate.<JOB_ID>.err
```

### Cancel the Job

```bash
scancel <JOB_ID>
```

## Script Workflow

The script performs the following steps:

1. **Setup**: Creates directories in LOCALSCRATCH for fast I/O
2. **Copy Repository**: Syncs the medsyn repo to LOCALSCRATCH
3. **Copy Checkpoint**: Syncs the trained checkpoint to LOCALSCRATCH
4. **Load Environment**: Activates the conda `medsyn` environment
5. **Generate Images**: Runs the generation process with GPU acceleration
6. **Sync Results**: Copies generated images back to permanent storage
7. **Cleanup**: Removes temporary files from LOCALSCRATCH

## Output Structure

After successful completion, your results directory will contain:

```
RESULTS_DST/
├── train/
│   ├── class_0/
│   │   ├── synth_<uuid>_class0.png
│   │   ├── synth_<uuid>_class0.png
│   │   └── denoising_process_class0_<uuid>.png
│   ├── class_1/
│   │   └── ...
│   └── pathmnist_train_synth.npz          # NPZ with is_synth=1
├── val/
│   ├── class_0/
│   │   └── ...
│   └── pathmnist_val_synth.npz            # NPZ with is_synth=1
└── test/                                   # (if configured)
    └── ...
```

### File Formats

1. **PNG Images**: Individual synthetic images saved as PNG files
   - Standard images: `synth_<uuid>_class<N>.png`
   - Denoising visualizations: `denoising_process_class<N>_<uuid>.png`

2. **NPZ Files**: Compressed NumPy archives with batch data
   - `{split}_images`: [N, H, W, C] uint8 array
   - `{split}_labels`: [N] int64 array
   - `{split}_is_synth`: [N] bool array (all `True` for synthetic data)

3. **JSON Indices**: Metadata files mapping sample IDs to file paths
   - `pathmnist_{split}_index.json`

## Merging with Original Data

After generation, merge synthetic images with the original dataset:

```bash
# Activate conda environment
conda activate medsyn

# Merge train split
python -m medsyn.utils.merge_synth_with_original \
  --original /path/to/original/pathmnist_custom.npz \
  --synthetic /path/to/generated/train/pathmnist_train_synth.npz \
  --output /path/to/merged/pathmnist_train_merged.npz \
  --splits train

# Merge val split
python -m medsyn.utils.merge_synth_with_original \
  --original /path/to/original/pathmnist_custom.npz \
  --synthetic /path/to/generated/val/pathmnist_val_synth.npz \
  --output /path/to/merged/pathmnist_val_merged.npz \
  --splits val
```

The merged NPZ will contain both real and synthetic images, distinguishable by the `is_synth` flag:
- `is_synth=0` (False): Real/original images
- `is_synth=1` (True): Synthetic/generated images

## Resource Requirements

The script is configured with:

- **Time**: 2 hours (adjust with `#SBATCH --time=HH:MM:SS`)
- **GPUs**: 1 GPU (adjust with `#SBATCH --gres=gpu:N`)
- **CPUs**: 4 cores (adjust with `#SBATCH --cpus-per-task=N`)
- **Memory**: 32GB RAM (adjust with `#SBATCH --mem=XG`)
- **Constraint**: DGX nodes (modify or remove for your cluster)

### Adjusting Resources

For generating more images, you may need to increase:

```bash
#SBATCH --time=04:00:00      # More time for large batches
#SBATCH --mem=64G            # More memory for larger batches
```

## Troubleshooting

### No outputs found

Check the job output file for errors:
```bash
cat ccddpm_generate.<JOB_ID>.out | grep -i error
cat ccddpm_generate.<JOB_ID>.err
```

Common issues:
- Checkpoint path is incorrect
- Config file `generate` section not properly configured
- Insufficient GPU memory
- Conda environment not activated

### Generation is slow

- Check GPU utilization: `nvidia-smi` in the output logs
- Reduce `num_inference_steps` in config (trades quality for speed)
- Reduce batch generation per class

### Out of memory errors

- Reduce number of samples generated per class
- Reduce image size in config
- Request more GPU memory: `#SBATCH --gres=gpu:a100:1` (if available)

## Advanced Usage

### Generate specific splits only

Edit the YAML config to include only the splits you need:

```yaml
generate:
  npz_with_synth_images:
    train:      # Only generate train split
      classes:
        0: 100
        # ...
    # Comment out or remove val/test sections
```

### Disable denoising visualizations

Add `--no-visualizations` flag to the generation command in the script:

```bash
srun python -m medsyn.cli.generate_ccDDPM "${CONFIG_TEMP}" --no-visualizations
```

### Custom output directory

Modify the `OUTPUT_DIR` variable in the script:

```bash
export OUTPUT_DIR="${WORKDIR}/my_custom_output"
```

## See Also

- Training script: `scripts/picasso_sbatch.sh`
- Generation CLI: `medsyn/cli/generate_ccDDPM.py`
- Merge utility: `medsyn/utils/merge_synth_with_original.py`
- Main config: `config/picasso_cfg.yaml`
