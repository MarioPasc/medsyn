# DistDiff Offline Execution Guide for HPC Clusters

**Problem**: HPC compute nodes often have no internet access, but DistDiff needs to download:
- Stable Diffusion v1-4 (~4-5 GB)
- OpenCLIP ViT-B/32 (~350 MB)
- Transformers components

**Solution**: Pre-download all models on a machine with internet, then copy to cluster.

---

## Quick Start

### Step 1: Pre-download Models (on machine with internet)

```bash
# Activate DistDiff environment
conda activate medsyn-distdiff

# Download all models to a directory (~5 GB total)
python scripts/predownload_distdiff_models.py \
  --cache_dir /path/to/local/cache

# Example: Download to current directory
python scripts/predownload_distdiff_models.py \
  --cache_dir ./distdiff_models
```

### Step 2: Copy to HPC Cluster

```bash
# Copy models to cluster scratch space
rsync -avP ./distdiff_models/ \
  your_username@cluster:/scratch/$USER/models/distdiff/
```

### Step 3: Update Config

```yaml
# config/distdiff_pathmnist.yaml
expansion:
  cache_dir: /scratch/$USER/models/distdiff  # Set to your cache location
```

### Step 4: Run on Cluster

```bash
# Submit job (will use offline cache)
sbatch scripts/distdiff_slurm_job.sh config/distdiff_pathmnist.yaml
```

---

## Detailed Instructions

### What Models Are Downloaded?

| Model | Size | Components | Downloaded From |
|-------|------|------------|-----------------|
| **Stable Diffusion v1-4** | ~4-5 GB | UNet, VAE, Text Encoder, Tokenizer, Scheduler | HuggingFace |
| **OpenCLIP ViT-B/32** | ~350 MB | Vision encoder, pretrained weights | OpenCLIP |
| **Total** | ~5 GB | All required for DistDiff | - |

### Pre-download Script Options

```bash
python scripts/predownload_distdiff_models.py \
  --cache_dir /path/to/cache \
  --sd_model "CompVis/stable-diffusion-v1-4" \
  --clip_model "ViT-B-32" \
  --clip_pretrained "laion2b_s34b_b79k"
```

**Parameters**:
- `--cache_dir`: **Required**. Where to save models
- `--sd_model`: Stable Diffusion model (default: v1-4)
- `--clip_model`: OpenCLIP architecture (default: ViT-B/32)
- `--clip_pretrained`: OpenCLIP weights (default: laion2b_s34b_b79k)

### Where to Store Models on HPC

**Recommended locations** (in order of preference):

1. **User scratch space** (fast, node-accessible):
   ```
   /scratch/$USER/models/distdiff/
   ```

2. **Shared scratch space** (if available):
   ```
   /shared/scratch/models/distdiff/
   ```

3. **Project space** (if persistent and fast):
   ```
   /projects/your_project/models/distdiff/
   ```

**DO NOT use**:
- ❌ Home directory (too slow, size limits)
- ❌ Network drives (slow I/O)
- ❌ Compute node local storage (not accessible during staging)

### Verification

After pre-downloading, verify the cache:

```bash
# Check downloaded files
ls -lh /path/to/cache/

# Expected structure:
# models--CompVis--stable-diffusion-v1-4/
# ├── snapshots/
# │   └── [hash]/
# │       ├── unet/
# │       ├── vae/
# │       ├── text_encoder/
# │       ├── tokenizer/
# │       └── scheduler/
# └── ...
#
# models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/
# └── ...

# Check total size
du -sh /path/to/cache/
# Expected: ~5 GB
```

---

## Configuration

### Config File Setup

**config/distdiff_pathmnist.yaml**:

```yaml
expansion:
  # Set to your cluster cache location
  cache_dir: /scratch/$USER/models/distdiff

  # Or use environment variables
  cache_dir: ${SCRATCH}/models/distdiff

  # Or set to null to use default HuggingFace cache
  cache_dir: null  # Only works if home dir has internet!
```

### Manual Override

You can also pass `--cache_dir` directly:

```bash
python medsyn/models/distdiff/generate_data.py \
  --cache_dir /scratch/$USER/models/distdiff \
  ... other args ...
```

But using config is recommended for consistency.

---

## Troubleshooting

### Issue: "Connection timeout" or "Unable to download"

**Cause**: Compute node has no internet, models not cached

**Solution**:
1. Pre-download models with internet access
2. Copy to cluster
3. Set `cache_dir` in config or via `--cache_dir`

### Issue: "No space left on device"

**Cause**: Cache directory quota exceeded (~5 GB needed)

**Solution**:
```bash
# Check quota
quota -s

# Use different location with more space
# Update cache_dir in config
```

### Issue: "Permission denied" when accessing cache

**Cause**: Cache directory not readable from compute nodes

**Solution**:
```bash
# Make cache readable
chmod -R a+rX /path/to/cache/

# Or use user-only permissions
chmod -R u+rwX,go-rwx /path/to/cache/
```

### Issue: Models download despite cache_dir set

**Possible causes**:

1. **Cache path mismatch**:
   ```bash
   # Check what's being used
   grep "cache_dir" logs/*.log

   # Verify in config
   grep "cache_dir:" config/distdiff_pathmnist.yaml
   ```

2. **Cache structure incorrect**:
   - Models must be in HuggingFace cache format
   - Use `predownload_distdiff_models.py` (don't manually copy)

3. **Environment variable override**:
   ```bash
   # Check if HF_HOME is set
   echo $HF_HOME

   # Unset if conflicting
   unset HF_HOME
   ```

### Issue: "Model not found" in cache

**Cause**: Incomplete download or wrong model name

**Solution**:
```bash
# Re-download specific model
python scripts/predownload_distdiff_models.py \
  --cache_dir /path/to/cache \
  --sd_model "CompVis/stable-diffusion-v1-4"

# Verify model exists
ls /path/to/cache/models--CompVis--stable-diffusion-v1-4/
```

---

## Advanced: Multiple Experiments

If running multiple DistDiff experiments with different models:

### Option 1: Separate Cache Directories

```yaml
# Experiment 1: SD v1-4
cache_dir: /scratch/$USER/models/distdiff_v14

# Experiment 2: SD v2-1
cache_dir: /scratch/$USER/models/distdiff_v21
```

### Option 2: Shared Cache (Recommended)

Use single cache for all models (HuggingFace automatically manages):

```yaml
# All experiments share cache
cache_dir: /scratch/$USER/models/huggingface_cache
```

Models with same name/version are reused automatically.

---

## Testing Offline Setup

### Test 1: Verify Pre-download

```bash
# On login node (with internet):
python scripts/predownload_distdiff_models.py --cache_dir ./test_cache

# Check download succeeded
ls -R ./test_cache/ | grep -E "(unet|vae|text_encoder)"
# Should show model directories
```

### Test 2: Verify Offline Loading

```bash
# On compute node (without internet):
conda activate medsyn-distdiff

python -c "
from diffusers import StableDiffusionPipeline
import os

cache_dir = '/path/to/cache'
model_id = 'CompVis/stable-diffusion-v1-4'

# This should work offline if cache is correct
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    local_files_only=True  # Force offline mode
)
print('✓ Offline loading successful!')
"
```

### Test 3: End-to-End Smoke Test

```bash
# Generate 1 image offline
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/generate_data.py \
  -a open_clip_vit_b32 \
  -d pathmnist_npz \
  --data_dir /path/to/pathmnist.npz \
  --output_dir test_output \
  --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
  --cache_dir /path/to/cache \
  --encoder_weight_path guide_model.pth.tar \
  --num_images_per_prompt 1 \
  --K 5

# Should complete without internet
```

---

## Example: Complete Workflow

### On Local Machine (with internet)

```bash
# 1. Activate environment
conda activate medsyn-distdiff

# 2. Pre-download models
python scripts/predownload_distdiff_models.py \
  --cache_dir ~/distdiff_models

# 3. Verify download
ls -lh ~/distdiff_models/
du -sh ~/distdiff_models/
# Should show ~5 GB

# 4. Copy to cluster
rsync -avP ~/distdiff_models/ \
  user@cluster:/scratch/$USER/models/distdiff/
```

### On HPC Cluster (no internet)

```bash
# 1. Update config
nano config/distdiff_pathmnist.yaml
# Set: cache_dir: /scratch/$USER/models/distdiff

# 2. Submit job
sbatch scripts/distdiff_slurm_job.sh config/distdiff_pathmnist.yaml

# 3. Monitor (should not see download messages)
tail -f distdiff_pathmnist.*.out

# Look for:
# ✓ "Loading pretrained model from cache"
# ✗ "Downloading..." (means cache not working)
```

---

## Best Practices

1. **Pre-download on login node**: Most clusters allow internet from login nodes
   ```bash
   # On login node (has internet)
   python scripts/predownload_distdiff_models.py \
     --cache_dir /scratch/$USER/models/distdiff
   ```

2. **Use scratch space**: Faster than home, accessible from compute nodes
   ```
   cache_dir: /scratch/$USER/models/distdiff
   ```

3. **Verify before submitting**: Test offline loading before big jobs
   ```bash
   # Quick test
   python -c "from diffusers import StableDiffusionPipeline; \
     StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', \
     cache_dir='/scratch/$USER/models/distdiff', local_files_only=True)"
   ```

4. **Share cache (optional)**: If running same models, team can share cache
   ```bash
   # Set group permissions
   chmod -R g+rX /shared/scratch/models/distdiff/
   ```

5. **Document cache location**: Add to job output for debugging
   ```bash
   echo "Using cache: /scratch/$USER/models/distdiff"
   ```

---

## Summary

**Models that need pre-downloading**:
- ✅ Stable Diffusion v1-4 (~4-5 GB)
- ✅ OpenCLIP ViT-B/32 (~350 MB)
- ✅ Transformers tokenizer & text encoder

**What WON'T be downloaded**:
- ❌ Guide model weights (trained by you)
- ❌ NPZ dataset (provided by you)
- ❌ Python packages (installed in conda env)

**Workflow**:
1. Pre-download with `scripts/predownload_distdiff_models.py`
2. Copy cache to cluster scratch space
3. Set `cache_dir` in config
4. Run jobs normally (uses offline cache)

---

## See Also

- **Pre-download script**: `scripts/predownload_distdiff_models.py`
- **Config example**: `config/distdiff_pathmnist.yaml`
- **Execution scripts**: `scripts/distdiff_slurm_job.sh`, `scripts/distdiff_local_job.sh`
- **Environment setup**: `ENVIRONMENTS.md`
- **Integration guide**: `medsyn/models/distdiff/README_NPZ_INTEGRATION.md`

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Status**: Production Ready
