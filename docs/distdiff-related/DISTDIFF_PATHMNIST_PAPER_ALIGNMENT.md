# DistDiff PathMNIST Paper Alignment Check

This document analyzes our DistDiff integration against the paper's specifications for PathMNIST experiments.

## Paper Reference

**Paper**: "Distribution-Aware Data Expansion with Diffusion Models"
**ArXiv**: [2403.06741](https://arxiv.org/abs/2403.06741)
**PathMNIST Settings**: Appendix / Table references

---

## Current vs Paper Settings

| Parameter | Current Setting | Paper Setting | Status |
|-----------|----------------|---------------|--------|
| **Prompt Template** | `"a colon pathological image of {}."` | `"A COLON PATHOLOGICAL IMAGE OF [CLASS]."` | ❌ **Lowercase vs Uppercase** |
| **Noise Strength** | `0.5` | `0.2` | ❌ **Wrong value** |
| **Guidance Step (M)** | `20` | `10` | ❌ **Wrong value** |
| **Stable Diffusion** | `CompVis/stable-diffusion-v1-4` | `stable-diffusion-v1-4` | ✅ **Correct** |
| **Baseline Model** | `open_clip_vit_b32` | `CLIP-ViT-B/32` | ✅ **Correct** |
| **K (sub-prototypes)** | `5` | Not specified for PathMNIST | ⚠️ **Need to verify** |

---

## Issue 1: Prompt Template Case

### Current Implementation
```python
# medsyn/models/distdiff/dataloader.py line 62
"pathmnist_npz": "a colon pathological image of {}.",
```

### Paper Specification
> "For PathMNIST, we use the prompt: **'A COLON PATHOLOGICAL IMAGE OF [CLASS].'**"

### Issue
- Current: lowercase
- Paper: UPPERCASE
- This affects CLIP text embeddings and may impact generation quality

### Fix Required
```python
"pathmnist_npz": "A COLON PATHOLOGICAL IMAGE OF {}.",
```

---

## Issue 2: Noise Strength

### Current Implementation
```yaml
# config/distdiff_pathmnist.yaml line 80
strength: 0.5  # Noise strength for img2img (0-1, higher = more noise)
```

### Paper Specification
> "PathMNIST uses a noise strength of **0.2**"

### Issue
- Current: 0.5 (50% noise)
- Paper: 0.2 (20% noise)
- Higher noise = more variation but less fidelity to original

### Impact
- 0.5 strength may generate images too different from originals
- 0.2 is more conservative, maintaining more structure from real images

### Fix Required
```yaml
strength: 0.2  # PathMNIST uses 0.2 per paper (not 0.5)
```

**Note**: DistDiff's default script uses 0.5 for Caltech-101, but PathMNIST requires 0.2

---

## Issue 3: Guidance Step (M)

### Current Implementation
```yaml
# config/distdiff_pathmnist.yaml line 86
guidance_step: 20  # Start guidance at timestep 20
```

### Paper Specification
> "PathMNIST uses **M=10** as guidance step"

### Issue
- Current: M=20 (start guidance at timestep 20/50)
- Paper: M=10 (start guidance at timestep 10/50)
- Earlier guidance = more influence on generation

### Impact
- M=10 starts guidance earlier in denoising process
- More aggressive prototype alignment
- Paper found M=10 optimal for PathMNIST

### Fix Required
```yaml
guidance_step: 10  # PathMNIST uses M=10 per paper (not 20)
```

---

## Issue 4: Baseline Model ✅

### Current Implementation
```yaml
# config/distdiff_pathmnist.yaml line 30
arch: open_clip_vit_b32
```

### Paper Specification
> "For the PathMNIST dataset, we fine-tune using the stronger **CLIP-ViT-B/32** baseline"

### Status: ✅ **CORRECT**

The config already uses `open_clip_vit_b32` which corresponds to OpenCLIP's ViT-B/32 architecture.

**Verification**:
- OpenCLIP model name: `ViT-B-32`
- Our config: `open_clip_vit_b32`
- ✅ Matches paper specification

---

## Proposed Fixes

### Fix 1: Update Prompt Template to Uppercase

**File**: `medsyn/models/distdiff/dataloader.py`

```python
# Line 62
CUSTOM_TEMPLATES = {
    # ... other templates ...
    "pathmnist": "A COLON PATHOLOGICAL IMAGE OF {}.",  # Changed to uppercase (paper spec)
    "pathmnist_npz": "A COLON PATHOLOGICAL IMAGE OF {}.",  # Changed to uppercase (paper spec)
    # ... other templates ...
}
```

**Reasoning**: CLIP models are sensitive to text case. Paper explicitly uses uppercase.

---

### Fix 2: Update Config to Paper Settings

**File**: `config/distdiff_pathmnist.yaml`

```yaml
expansion:
  # Generation parameters - PATHMNIST PAPER SETTINGS
  num_images_per_prompt: 5
  strength: 0.2             # PathMNIST paper specification (not 0.5)
  guidance_scale: 7.5       # Classifier-free guidance scale

  # Guidance configuration - PATHMNIST PAPER SETTINGS
  guidance_type: transform_guidance
  optimize_targets: global_prototype-local_prototype
  guidance_step: 10         # PathMNIST paper: M=10 (not 20)
  guidance_period: 2        # Guide for 2 consecutive steps
```

**Changes**:
- `strength: 0.5` → `strength: 0.2`
- `guidance_step: 20` → `guidance_step: 10`

---

### Fix 3: Update Execution Scripts

**File**: `scripts/distdiff_slurm_job.sh` (lines ~188-202)

```bash
# Stage 2: Generate synthetic data (4 GPUs)
CUDA_VISIBLE_DEVICES=${split} python medsyn/models/distdiff/generate_data.py \
    --guidance_type=transform_guidance \
    -a open_clip_vit_b32 \  # Changed from resnet50
    -d pathmnist_npz \
    --data_dir "${DATA_DST}" \
    --output_dir "${SYNTH_DATA_DIR}/split_${split}" \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --gradient_checkpointing \
    --K 5 \
    --train_batch_size 1 \
    --optimize_targets "global_prototype-local_prototype" \
    --strength 0.2 \  # CHANGED: PathMNIST paper value
    --num_images_per_prompt 5 \
    --guidance_step 10 \  # CHANGED: PathMNIST paper value (M=10)
    --guidance_period 2 \
    --encoder_weight_path "${GUIDE_MODEL_PATH}" \
    --guidance_scale 7.5 \
    --constraint_value 0.2 \
    --rho 10.0 \
    --total_split 4 \
    --split ${split}
```

**Same changes needed in**: `scripts/distdiff_local_job.sh`

---

## Summary of Required Changes

### 1. Code Changes (1 file)
- **File**: `medsyn/models/distdiff/dataloader.py`
- **Lines**: 61-62
- **Change**: Uppercase prompt template

### 2. Config Changes (1 file)
- **File**: `config/distdiff_pathmnist.yaml`
- **Lines**: 80, 86
- **Changes**:
  - `strength: 0.5` → `0.2`
  - `guidance_step: 20` → `10`

### 3. Script Changes (2 files)
- **Files**:
  - `scripts/distdiff_slurm_job.sh` (line ~196, ~200)
  - `scripts/distdiff_local_job.sh` (line ~223, ~227)
- **Changes**:
  - `--strength 0.5` → `--strength 0.2`
  - `--guidance_step 20` → `--guidance_step 10`
  - `-a resnet50` → `-a open_clip_vit_b32` (if not already changed)

---

## Verification After Fixes

Run these checks after applying fixes:

```bash
# 1. Check prompt template
python -c "from medsyn.models.distdiff.dataloader import CUSTOM_TEMPLATES; print(CUSTOM_TEMPLATES['pathmnist_npz'])"
# Expected: "A COLON PATHOLOGICAL IMAGE OF {}."

# 2. Check config values
grep "strength:" config/distdiff_pathmnist.yaml
# Expected: strength: 0.2

grep "guidance_step:" config/distdiff_pathmnist.yaml
# Expected: guidance_step: 10

# 3. Check script values
grep "strength" scripts/distdiff_slurm_job.sh | grep 0.2
grep "guidance_step" scripts/distdiff_slurm_job.sh | grep 10
```

---

## Impact Analysis

### Prompt Template Change (lowercase → UPPERCASE)
**Impact**: Medium
- CLIP text encoder is case-sensitive
- Uppercase may provide stronger semantic signal
- Paper specifically uses uppercase

**Risk**: Low (text encoding change only)

### Noise Strength (0.5 → 0.2)
**Impact**: High
- Less noise = closer to original images
- More fidelity, potentially less diversity
- Paper found 0.2 optimal for PathMNIST

**Risk**: Low (follows paper specification)

### Guidance Step (20 → 10)
**Impact**: High
- Earlier guidance = more prototype influence
- Starts at 10/50 steps instead of 20/50
- More aggressive distribution alignment

**Risk**: Low (follows paper specification)

---

## Additional Considerations

### K Value (Sub-prototypes)
**Current**: K=5
**Paper**: Not explicitly stated for PathMNIST

**Action**: Keep K=5 unless paper appendix specifies otherwise. The paper uses K=3 for some datasets but this varies.

### Expansion Factor
**Current**: 5x expansion
**Paper**: Varies by experiment

**Action**: Keep 5x as default, but note paper experiments with 3x, 5x, 10x.

---

## Testing Plan

After applying fixes:

1. **Smoke Test**: Generate 1 batch on single GPU
   ```bash
   conda activate medsyn-distdiff
   CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/generate_data.py \
     -d pathmnist_npz \
     --data_dir /path/to/pathmnist.npz \
     --output_dir test_output \
     --encoder_weight_path model.pth.tar \
     --strength 0.2 \
     --guidance_step 10 \
     --num_images_per_prompt 1
   ```

2. **Verify Prompt**: Check generation logs for correct prompt format

3. **Visual Inspection**: Compare generated images with paper figures

4. **Full Pipeline**: Run complete 3-stage pipeline

---

## References

- **Paper**: [DistDiff arXiv:2403.06741](https://arxiv.org/abs/2403.06741)
- **DistDiff Code**: `medsyn/models/distdiff/`
- **Default Script**: `medsyn/models/distdiff/scripts/exps/expand_diff.sh` (Caltech-101 defaults)
- **PathMNIST Config**: `config/distdiff_pathmnist.yaml`

---

**Document Version**: 1.0
**Date**: 2025-01-XX
**Status**: Fixes Proposed - Awaiting Implementation
