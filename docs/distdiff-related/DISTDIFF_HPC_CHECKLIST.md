# DistDiff HPC Offline Execution - Quick Reference

## ⚠️ CRITICAL: Models Will Download Without Internet Setup

**Problem**: HPC compute nodes have no internet → downloads will FAIL

**What gets downloaded at runtime**:
- ✅ Stable Diffusion v1-4: **~4-5 GB** (UNet, VAE, Text Encoder, Tokenizer)
- ✅ OpenCLIP ViT-B/32: **~350 MB** (Vision encoder)
- **Total**: ~5 GB downloaded from HuggingFace/OpenCLIP repos

**Solution**: Pre-download before submitting jobs

---

## Quick Setup (3 Steps)

### 1️⃣ Pre-download (on machine WITH internet)

```bash
conda activate medsyn-distdiff
python scripts/predownload_distdiff_models.py --cache_dir ./distdiff_models
```

### 2️⃣ Copy to Cluster

```bash
rsync -avP ./distdiff_models/ cluster:/scratch/$USER/models/distdiff/
```

### 3️⃣ Update Config

```yaml
# config/distdiff_pathmnist.yaml
expansion:
  cache_dir: /scratch/$USER/models/distdiff  # Change from null
```

---

## Detailed Checklist

### ☑️ Before First Run

- [ ] **Install environment** (on cluster):
  ```bash
  conda create -n medsyn-distdiff python=3.10 -y
  conda activate medsyn-distdiff
  pip install -e ".[distdiff]"
  ```

- [ ] **Pre-download models** (WITH internet - can be local machine):
  ```bash
  python scripts/predownload_distdiff_models.py \
    --cache_dir /path/to/local/cache
  ```

- [ ] **Copy models to cluster**:
  ```bash
  rsync -avP /path/to/local/cache/ \
    cluster:/scratch/$USER/models/distdiff/
  ```

- [ ] **Verify cache on cluster**:
  ```bash
  ssh cluster
  ls -lh /scratch/$USER/models/distdiff/
  du -sh /scratch/$USER/models/distdiff/  # Should show ~5 GB
  ```

- [ ] **Update config** (`config/distdiff_pathmnist.yaml`):
  ```yaml
  expansion:
    cache_dir: /scratch/$USER/models/distdiff  # Set this!
  ```

- [ ] **Test offline loading** (on cluster):
  ```bash
  conda activate medsyn-distdiff
  python -c "
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained(
      'CompVis/stable-diffusion-v1-4',
      cache_dir='/scratch/\$USER/models/distdiff',
      local_files_only=True
  )
  print('✓ Offline test passed!')
  "
  ```

### ☑️ Before Each Job Submission

- [ ] **Config cache_dir is set** (not null)
- [ ] **Cache directory exists and is accessible**:
  ```bash
  ls /scratch/$USER/models/distdiff/
  ```
- [ ] **Enough disk quota** (~5 GB for cache + output space)
- [ ] **Correct environment activated** (`medsyn-distdiff`)

### ☑️ After Job Starts

- [ ] **Check logs for download attempts**:
  ```bash
  tail -f distdiff_pathmnist.*.out

  # Good: "Loading from cache"
  # Bad: "Downloading..." means cache not working!
  ```

- [ ] **Monitor job doesn't fail early**:
  ```bash
  squeue -u $USER
  # Should run for hours, not fail in minutes
  ```

---

## What Gets Downloaded vs What You Provide

### Downloaded from Internet (need pre-download)

| Component | Size | Source | Pre-download? |
|-----------|------|--------|---------------|
| Stable Diffusion v1-4 | ~4-5 GB | HuggingFace | ✅ Required |
| OpenCLIP ViT-B/32 | ~350 MB | OpenCLIP | ✅ Required |
| **Total** | **~5 GB** | - | - |

### You Must Provide (not downloaded)

| Component | Source | Location |
|-----------|--------|----------|
| Guide model weights | Train yourself | Generated in Stage 1 |
| NPZ dataset | MedSyn data prep | Your data directory |
| Python packages | pip install | Conda environment |
| DistDiff code | Git subtree | `medsyn/models/distdiff/` |

---

## Common Failure Patterns

### ❌ Job fails in Stage 2 (Generation)

**Symptom**: Job fails ~10-20 min after starting, during generation phase

**Likely cause**: Models not cached, download fails without internet

**Check**:
```bash
grep -i "download\|timeout\|connection" logs/*.log
```

**Fix**: Pre-download and set `cache_dir`

### ❌ "OSError: Can't load tokenizer"

**Cause**: Tokenizer not in cache

**Fix**: Re-run pre-download script, verify cache structure

### ❌ Job uses default HF cache (`~/.cache/huggingface`)

**Symptom**: Downloads work on login node but fail on compute node

**Cause**: `cache_dir: null` in config

**Fix**: Set explicit cache_dir path

---

## Storage Requirements

### Minimum Space Needed

| Location | Purpose | Size | Notes |
|----------|---------|------|-------|
| Model cache | Pre-downloaded models | ~5 GB | One-time |
| Output dir | Generated images | Variable | ~10 GB for 5x expansion |
| Checkpoints | Model weights | ~1 GB | Guide + expanded classifier |
| **Total** | - | **~16 GB** | Per experiment |

### Recommended Locations

```
/scratch/$USER/
├── models/
│   └── distdiff/          # ~5 GB (shared across experiments)
└── experiments/
    └── pathmnist_distdiff/
        ├── checkpoints/   # ~1 GB
        ├── synthetic/     # ~10 GB
        └── logs/          # ~100 MB
```

---

## Quick Verification Commands

```bash
# 1. Check cache exists
test -d /scratch/$USER/models/distdiff && echo "✓ Cache exists" || echo "✗ No cache"

# 2. Check cache size
du -sh /scratch/$USER/models/distdiff/
# Expected: ~5 GB

# 3. Check config
grep "cache_dir:" config/distdiff_pathmnist.yaml
# Should NOT show "null"

# 4. Test offline loading
conda activate medsyn-distdiff
python -c "from diffusers import StableDiffusionPipeline; \
  StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', \
  cache_dir='/scratch/\$USER/models/distdiff', local_files_only=True); \
  print('✓ Works offline')"

# 5. Check environment
conda info | grep "active environment"
# Should show: medsyn-distdiff
```

---

## Emergency: Job Already Running Without Cache

If you submitted a job without pre-downloading:

### Option 1: Cancel and Resubmit

```bash
scancel JOBID  # Cancel current job
# Follow checklist above to setup cache
sbatch scripts/distdiff_slurm_job.sh config/distdiff_pathmnist.yaml
```

### Option 2: Download on Login Node (if allowed)

Some clusters allow downloads from login nodes:

```bash
# On login node (may have internet)
conda activate medsyn-distdiff
python scripts/predownload_distdiff_models.py \
  --cache_dir /scratch/$USER/models/distdiff

# Update config
nano config/distdiff_pathmnist.yaml  # Set cache_dir

# Cancel and resubmit
scancel JOBID
sbatch scripts/distdiff_slurm_job.sh config/distdiff_pathmnist.yaml
```

### Option 3: Use Different Cluster with Internet

If your cluster has nodes with internet access, request those nodes.

---

## Summary

**Must do before running DistDiff on HPC**:
1. ✅ Pre-download ~5 GB of models (WITH internet)
2. ✅ Copy to cluster scratch space
3. ✅ Set `cache_dir` in config
4. ✅ Test offline loading works

**Files to check**:
- `config/distdiff_pathmnist.yaml` - cache_dir set?
- `/scratch/$USER/models/distdiff/` - exists? ~5 GB?
- Logs - no "Downloading..." messages?

**If stuck, see**: `DISTDIFF_OFFLINE_SETUP.md` (full guide)

---

**Status**: Ready to use ✅
**Script**: `scripts/predownload_distdiff_models.py`
**Doc**: `DISTDIFF_OFFLINE_SETUP.md`
