# MedSyn Environment Management

MedSyn supports multiple synthesis methods with different dependency requirements. This guide explains how to set up and manage separate conda environments for each method.

## Overview

| Environment | Primary Use | Key Dependencies | Python Version |
|-------------|-------------|------------------|----------------|
| `medsyn` | ccDDPM, bVAE, data preparation | `diffusers>=0.21`, `transformers` (latest), `albumentationsx` | 3.9+ |
| `medsyn-distdiff` | DistDiff competitor integration | `diffusers>=0.21`, `transformers==4.19.2`, `timm`, `open-clip-torch` | 3.10 |

## Why Separate Environments?

**Version Conflicts**: DistDiff requires `transformers==4.19.2` (released 2022), while MedSyn's ccDDPM works best with newer versions. Installing both in the same environment would cause downgrades and potential breakage.

**Solution**: Use separate conda environments, both installing from the same MedSyn repository with different optional dependency groups.

## Environment Setup

### Option 1: MedSyn Core (ccDDPM + bVAE)

```bash
# Create environment
conda create -n medsyn python=3.10 -y
conda activate medsyn

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MedSyn with core dependencies
cd /path/to/medsyn
pip install -e .

# Verify
python -c "import medsyn; import torch; print('MedSyn + PyTorch OK')"
```

**Use for**:
- Data preparation: `medsyn-prepare-data`
- bVAE training: `bvae-train`
- ccDDPM training: `ccddpm-train`
- ccDDPM generation: `ccddpm-generate`
- Privacy auditing: `ccddpm-audit`

### Option 2: DistDiff Integration

```bash
# Create separate environment
conda create -n medsyn-distdiff python=3.10 -y
conda activate medsyn-distdiff

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MedSyn with DistDiff dependencies
cd /path/to/medsyn
pip install -e ".[distdiff]"

# Verify
python -c "import medsyn; import transformers; print(f'Transformers: {transformers.__version__}')"
```

**Use for**:
- DistDiff training: `python medsyn/models/distdiff/train.py`
- DistDiff generation: `python medsyn/models/distdiff/generate_data.py`
- DistDiff pipeline: `bash scripts/distdiff_local_job.sh`

## Quick Switching Guide

### Scenario 1: Prepare Data, Then Run DistDiff

```bash
# Step 1: Prepare NPZ dataset (medsyn environment)
conda activate medsyn
medsyn-prepare-data --config config/medsyn_cfg.yaml

# Step 2: Run DistDiff (medsyn-distdiff environment)
conda activate medsyn-distdiff
bash scripts/distdiff_local_job.sh config/distdiff_pathmnist.yaml 4
```

### Scenario 2: Compare ccDDPM vs DistDiff

```bash
# Train ccDDPM
conda activate medsyn
ccddpm-train --config config/medsyn_cfg.yaml --dataset data.npz --outdir outputs/ccddpm

# Train DistDiff
conda activate medsyn-distdiff
bash scripts/distdiff_local_job.sh config/distdiff_pathmnist.yaml 4

# Compare results (either environment works)
conda activate medsyn
python -m medsyn.analysis.compare_methods outputs/ccddpm outputs/distdiff
```

## Cluster/SLURM Usage

When using SLURM scripts, activate the appropriate environment in your job script:

### For ccDDPM (picasso_parallel_job.sh)
```bash
#!/usr/bin/env bash
#SBATCH ...

# Load conda
module load miniconda3
conda activate medsyn  # ← Use main environment

# Run ccDDPM training
torchrun -m medsyn.cli.train_ccDDPM ...
```

### For DistDiff (distdiff_slurm_job.sh)
```bash
#!/usr/bin/env bash
#SBATCH ...

# Load conda
module load miniconda3
conda activate medsyn-distdiff  # ← Use DistDiff environment

# Run DistDiff pipeline
python medsyn/models/distdiff/train.py ...
```

**Note**: Both scripts already activate `medsyn` by default. Update them to activate `medsyn-distdiff` when running DistDiff.

## Dependency Lists

### MedSyn Core Dependencies
```
numpy
torch>=2.0.0
torchvision>=0.15.0
medmnist>=2.2.0
pyyaml>=6.0
pillow>=9.0.0
tqdm>=4.64.0
pytorch_msssim
diffusers>=0.21.0
torchmetrics>=0.11.0
matplotlib>=3.5.0
ultralytics>=8.0.0
scienceplots>=1.0.0
albumentationsx
```

### DistDiff Additional Dependencies
```
scipy>=1.9.0
scikit-image>=0.19.0
timm>=0.6.12
regex>=2022.1.18
packaging>=21.3
transformers==4.19.2  # ← Specific version!
accelerate>=0.12.0
huggingface-hub>=0.8.1
open-clip-torch>=2.0.0
```

## Troubleshooting

### Issue: "transformers version conflict"

**Symptom**: Installing DistDiff deps in `medsyn` environment downgrades transformers

**Solution**: Use separate `medsyn-distdiff` environment as shown above

### Issue: "CUDA out of memory" when switching between methods

**Solution**: Each environment uses the same PyTorch, so clear cache between runs:
```python
import torch
torch.cuda.empty_cache()
```

### Issue: Can't find conda command

**Solution**: Initialize conda in your shell:
```bash
conda init bash  # or zsh, fish, etc.
source ~/.bashrc
```

### Issue: Scripts activate wrong environment

**Solution**: Update the conda activation line in your scripts:

**picasso_parallel_job.sh** (line ~76):
```bash
conda activate medsyn  # For ccDDPM
```

**distdiff_slurm_job.sh** (line ~76):
```bash
conda activate medsyn-distdiff  # For DistDiff
```

**distdiff_local_job.sh** (check script header):
```bash
# Run with: conda activate medsyn-distdiff && bash scripts/distdiff_local_job.sh
```

## Best Practices

1. **Always activate before running**: Don't rely on default environment
   ```bash
   conda activate medsyn-distdiff
   bash scripts/distdiff_local_job.sh config.yaml
   ```

2. **Check environment before long jobs**:
   ```bash
   conda info | grep "active environment"
   python -c "import transformers; print(transformers.__version__)"
   ```

3. **Document which environment in scripts**:
   ```bash
   #!/usr/bin/env bash
   # Environment: medsyn-distdiff
   # Usage: conda activate medsyn-distdiff && bash this_script.sh
   ```

4. **Use environment-specific output directories**:
   ```
   outputs/
   ├── ccddpm/      # From medsyn environment
   ├── bvae/        # From medsyn environment
   └── distdiff/    # From medsyn-distdiff environment
   ```

## Installing Both Environments (Quick Reference)

```bash
# Create base environment
conda create -n medsyn python=3.10 -y
conda activate medsyn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd /path/to/medsyn
pip install -e .

# Create DistDiff environment
conda create -n medsyn-distdiff python=3.10 -y
conda activate medsyn-distdiff
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd /path/to/medsyn
pip install -e ".[distdiff]"

# Verify both
conda activate medsyn
python -c "import medsyn, torch; print('medsyn OK')"

conda activate medsyn-distdiff
python -c "import medsyn, transformers; print(f'medsyn-distdiff OK, transformers={transformers.__version__}')"
```

## Summary

- **Two environments**: `medsyn` (core) and `medsyn-distdiff` (DistDiff)
- **Same repository**: Both install from the same MedSyn codebase
- **Different dependencies**: Use `pip install -e .` vs `pip install -e ".[distdiff]"`
- **Activation required**: Always activate correct environment before running
- **SLURM scripts**: Update conda activation line as needed

For more details:
- MedSyn core: See main `README.md`
- DistDiff integration: See `medsyn/models/distdiff/README_NPZ_INTEGRATION.md`
