# Environment Variables and CLI Overrides Guide

## Overview

The configuration system now supports **three levels of configuration** with clear precedence:

1. **YAML config** (base configuration)
2. **Environment variables** (expanded in YAML, e.g., `${DATASET_PATH}`)
3. **CLI arguments** (highest precedence, overrides everything)

This provides maximum flexibility for local development and HPC/SLURM deployment.

## Precedence Rules

```
CLI Arguments > Environment Variables > YAML Values
```

Example:
```yaml
# config.yaml
data:
  postprocess_npz:
    npz_path: ${DATASET_PATH}  # Will expand to env var

ccddpm:
  train:
    output_dir: ${OUTPUT_DIR}  # Will expand to env var
```

```bash
# Environment variables set
export DATASET_PATH=/path/from/env/data.npz
export OUTPUT_DIR=/path/from/env/output

# CLI overrides (takes precedence)
ccddpm-train --config config.yaml \
  --dataset /path/from/cli/data.npz \
  --outdir /path/from/cli/output

# Result: Uses CLI paths, not env vars or YAML
```

## Feature 1: Environment Variable Expansion

### How It Works

The config loaders automatically expand environment variables in YAML string values:

- `${VAR_NAME}` - Standard bash-style expansion
- `$VAR_NAME` - Short form expansion
- Absolute paths - Backward compatible, no expansion needed

### Implementation

Both config loaders (`medsyn.models.ccDDPM.config` and `medsyn.data.config`) now include:

```python
def _expand_env(obj: Any) -> Any:
    """
    Recursively expand ${ENV} and $ENV in strings within mappings/lists.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        expanded = [_expand_env(v) for v in obj]
        return type(obj)(expanded) if isinstance(obj, tuple) else expanded
    return obj
```

Called automatically in `load_cfg()` and `load_config()` after YAML parsing.

### Example Usage

**Config file (picasso_cfg.yaml):**
```yaml
data:
  postprocess_npz:
    enabled: true
    npz_path: ${DATASET_PATH}

ccddpm:
  train:
    output_dir: ${OUTPUT_DIR}
```

**SLURM script:**
```bash
export DATASET_PATH="/scratch/data/PathMNIST.npz"
export OUTPUT_DIR="/scratch/results/run_001"

ccddpm-train --config config/picasso_cfg.yaml
```

**Result:** Paths are automatically expanded to absolute paths.

### Backward Compatibility

Absolute paths in YAML work unchanged:

```yaml
data:
  postprocess_npz:
    npz_path: /absolute/path/to/data.npz  # ✅ Works as before

ccddpm:
  train:
    output_dir: /absolute/path/to/output  # ✅ Works as before
```

## Feature 2: CLI Argument Overrides

### New CLI Arguments

**ccddpm-train:**
```bash
ccddpm-train [config] [--config CONFIG] [--dataset DATASET] [--outdir OUTDIR]

Arguments:
  config              Path to YAML config (positional, default: medsyn_config.yaml)
  --config CONFIG     Alternative way to specify config path
  --dataset DATASET   Absolute path to NPZ dataset (overrides YAML)
  --outdir OUTDIR     Absolute output directory (overrides YAML)
```

### How Overrides Work

After loading the config, CLI arguments are applied:

```python
# Load base config (with env var expansion)
cfg = load_cfg(str(config_path))

# Apply CLI overrides (highest precedence)
if args.dataset:
    cfg.ccddpm.dataloader.npz_path = Path(args.dataset)

if args.outdir:
    cfg.ccddpm.train.output_dir = Path(args.outdir)
```

### Example Usage

**1. Basic usage (YAML only):**
```bash
ccddpm-train --config config/medsyn_cfg.yaml
```

**2. Override dataset path:**
```bash
ccddpm-train --config config.yaml --dataset /custom/path/data.npz
```

**3. Override both dataset and output:**
```bash
ccddpm-train --config config.yaml \
  --dataset /scratch/data.npz \
  --outdir /scratch/results/experiment_42
```

**4. SLURM with both env vars and CLI (recommended):**
```bash
# Set env vars for fallback
export DATASET_PATH="/scratch/data.npz"
export OUTPUT_DIR="/scratch/results"

# CLI overrides for explicit control
srun ccddpm-train \
  --config config/picasso_cfg.yaml \
  --dataset "${DATA_DST}" \
  --outdir  "${OUT_DIR}"
```

## SLURM/HPC Integration

### Complete Example: picasso_sbatch.sh

```bash
#!/usr/bin/env bash
#SBATCH -J ccddpm_pathmnist
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -euo pipefail

# Define paths
DATA_SRC="/mnt/fscratch/datasets/PathMNIST.npz"
REPO_SRC="/mnt/fscratch/repos/medsyn"
RESULTS_DST="/mnt/fscratch/results/PathMNIST_ccDDPM"

# Setup localscratch
MYLOCALSCRATCH="${LOCALSCRATCH}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/results"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"

# Copy repo and data to localscratch
rsync -a "${REPO_SRC}/" "${WORKDIR}/medsyn/"
rsync -a "${DATA_SRC}" "${DATA_DIR}/PathMNIST.npz"

# Set paths
DATA_DST="${DATA_DIR}/PathMNIST.npz"
cd "${WORKDIR}/medsyn"

# Activate conda environment
conda activate medsyn

# Export env vars (for YAML expansion)
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"

# Run training with CLI overrides (explicit, guaranteed)
srun ccddpm-train \
  --config config/picasso_cfg.yaml \
  --dataset "${DATA_DST}" \
  --outdir  "${OUT_DIR}"

# Sync results back
rsync -a "${OUT_DIR}/" "${RESULTS_DST}/"

# Cleanup
rm -rf "${MYLOCALSCRATCH}"
```

### Why Both Env Vars AND CLI Args?

**Environment variables (`export`):**
- Expand `${DATASET_PATH}` placeholders in YAML
- Useful for scripts/tools that read YAML directly
- Fallback if CLI args not used

**CLI arguments (`--dataset`, `--outdir`):**
- **Explicit and guaranteed** - no YAML parsing ambiguity
- **Highest precedence** - overrides everything
- **Easier to debug** - clear in job logs
- **Recommended for HPC** - eliminates "literal ${...}" bugs

Both methods work independently, but using both provides defense-in-depth.

## Configuration Patterns

### Pattern 1: Local Development (Absolute Paths)
```yaml
# config/local_cfg.yaml
data:
  postprocess_npz:
    npz_path: /home/user/datasets/PathMNIST.npz

ccddpm:
  train:
    output_dir: /home/user/experiments/run_001
```

```bash
ccddpm-train --config config/local_cfg.yaml
```

**Pros:** Simple, no env vars needed
**Cons:** Not portable across machines

### Pattern 2: Environment Variables (Portable)
```yaml
# config/env_cfg.yaml
data:
  postprocess_npz:
    npz_path: ${DATASET_PATH}

ccddpm:
  train:
    output_dir: ${OUTPUT_DIR}
```

```bash
export DATASET_PATH=/path/to/data.npz
export OUTPUT_DIR=/path/to/output
ccddpm-train --config config/env_cfg.yaml
```

**Pros:** Portable, reusable config
**Cons:** Must remember to set env vars

### Pattern 3: CLI Overrides (Most Flexible)
```yaml
# config/base_cfg.yaml - reasonable defaults
data:
  postprocess_npz:
    npz_path: /default/path/data.npz

ccddpm:
  train:
    output_dir: /default/output
```

```bash
# Override per-run
ccddpm-train --config config/base_cfg.yaml \
  --dataset /run1/data.npz \
  --outdir /run1/output

ccddpm-train --config config/base_cfg.yaml \
  --dataset /run2/data.npz \
  --outdir /run2/output
```

**Pros:** Maximum flexibility, clear intent
**Cons:** Longer command lines

### Pattern 4: HPC Best Practice (All Three)
```yaml
# config/picasso_cfg.yaml
data:
  postprocess_npz:
    npz_path: ${DATASET_PATH}  # Fallback from env

ccddpm:
  train:
    output_dir: ${OUTPUT_DIR}  # Fallback from env
```

```bash
# SLURM script
export DATASET_PATH="${DATA_DST}"  # Env var fallback
export OUTPUT_DIR="${OUT_DIR}"

srun ccddpm-train \
  --config config/picasso_cfg.yaml \
  --dataset "${DATA_DST}" \          # Explicit CLI override
  --outdir  "${OUT_DIR}"             # Explicit CLI override
```

**Pros:** Defense-in-depth, maximum robustness
**Cons:** Slight redundancy (acceptable for production)

## Troubleshooting

### Issue: Literal `${DATASET_PATH}` in logs

**Symptom:**
```
FileNotFoundError: /path/to/${DATASET_PATH}
```

**Cause:** Environment variable not set

**Solutions:**
1. Set the environment variable:
   ```bash
   export DATASET_PATH=/actual/path/data.npz
   ```

2. Use absolute path in YAML:
   ```yaml
   npz_path: /actual/path/data.npz
   ```

3. Use CLI override (recommended):
   ```bash
   ccddpm-train --config cfg.yaml --dataset /actual/path/data.npz
   ```

### Issue: CLI override not working

**Symptom:**
```
Training using /wrong/path/data.npz instead of /correct/path/data.npz
```

**Solution:**
- CLI overrides only work for `--dataset` and `--outdir`
- Make sure you're using the correct argument names
- Check that the config has the expected structure:
  ```python
  cfg.ccddpm.dataloader.npz_path  # For --dataset
  cfg.ccddpm.train.output_dir     # For --outdir
  ```

### Issue: Permission denied on output directory

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: '${OUTPUT_DIR}/checkpoints'
```

**Cause:** `${OUTPUT_DIR}` not expanded (env var not set)

**Solution:**
```bash
# Option 1: Set env var
export OUTPUT_DIR=/writable/path

# Option 2: Use CLI override
ccddpm-train --config cfg.yaml --outdir /writable/path

# Option 3: Use absolute path in YAML
```

## Testing the Configuration

### Test 1: Environment Variable Expansion

```bash
# Create test config
cat > test_cfg.yaml <<EOF
data:
  postprocess_npz:
    npz_path: \${TEST_DATA_PATH}
ccddpm:
  train:
    output_dir: \${TEST_OUTPUT_DIR}
EOF

# Set env vars
export TEST_DATA_PATH=/tmp/test_data.npz
export TEST_OUTPUT_DIR=/tmp/test_output

# Verify expansion (should NOT see literal ${...})
ccddpm-train --config test_cfg.yaml --help
# Check logs for expanded paths
```

### Test 2: CLI Overrides

```bash
# Should use CLI paths, ignoring YAML
ccddpm-train --config test_cfg.yaml \
  --dataset /override/data.npz \
  --outdir /override/output

# Check logs: should show "/override/..." paths
```

### Test 3: Backward Compatibility

```bash
# Old-style config with absolute paths
cat > old_cfg.yaml <<EOF
data:
  postprocess_npz:
    npz_path: /absolute/path/data.npz
ccddpm:
  train:
    output_dir: /absolute/path/output
EOF

# Should work without any env vars or CLI args
ccddpm-train --config old_cfg.yaml
```

## Migration Checklist

If you have existing configs, follow these steps:

- [ ] **Option 1: No changes needed** - Absolute paths work as before
- [ ] **Option 2: Add env vars** - Replace paths with `${VAR_NAME}`, set vars before running
- [ ] **Option 3: Use CLI overrides** - Keep YAML as-is, pass `--dataset` and `--outdir`
- [ ] **Update SLURM scripts** - Add `export` statements and/or CLI arguments
- [ ] **Test on local machine** - Verify paths resolve correctly
- [ ] **Test on HPC** - Submit test job before production runs

## Summary

### What Changed

1. ✅ **Environment variable expansion** in both config loaders
2. ✅ **CLI arguments** `--dataset` and `--outdir` for ccddpm-train
3. ✅ **Backward compatible** - absolute paths still work
4. ✅ **Updated sbatch script** - includes both env vars and CLI args

### Best Practices

1. **Local development:** Use absolute paths in YAML (simple)
2. **Shared configs:** Use environment variables (portable)
3. **HPC/SLURM:** Use CLI overrides (explicit, robust)
4. **Production:** Use all three for defense-in-depth

### Quick Reference

```bash
# Method 1: Absolute paths (unchanged)
ccddpm-train --config config.yaml

# Method 2: Environment variables
export DATASET_PATH=/path/to/data.npz
export OUTPUT_DIR=/path/to/output
ccddpm-train --config config.yaml

# Method 3: CLI overrides (recommended for HPC)
ccddpm-train --config config.yaml \
  --dataset /path/to/data.npz \
  --outdir /path/to/output
```
