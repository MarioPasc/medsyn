# ccDDPM Bug Fix - Testing Guide

## Quick Overview

**Problem:** Class-conditional generation produced pure noise instead of images
**Root Cause:** Bug in `visualize_denoising_process()` - returned initial noise instead of final denoised image
**Status:** ✅ **FIXED**

## What Was Changed

### 1. Core Fix: `medsyn/models/ccDDPM/engine/train.py`

**Line 137:** Added final image save
```python
# CRITICAL FIX: Always append final denoised image
denoised_images.append(x_t.cpu())
```

**Line 115:** Fixed empty save_indices handling
```python
save_indices = set(torch.linspace(0, len(scheduler.timesteps) - 1, num_steps, dtype=torch.long).tolist()) if num_steps > 0 else set()
```

**Lines 615-637:** Updated class-conditional sampling with debug logging
- Now correctly extracts final denoised image from the returned tensor
- Logs shape and value range for each class

**Line 600:** Fixed denoising visualization grid (nrow=11 → nrow=12)

### 2. Enhanced Logging: `medsyn/cli/generate_ccDDPM.py`

**Lines 185, 207-238:** Added debug parameter to `generate_with_cfg()`
- Logs initial noise, scheduler setup
- Logs denoising progress every 200 steps
- Logs final image statistics

**Lines 400-401, 436-438:** Added per-class generation logging
- Debug enabled for first sample of first class
- Value range logged for first sample of each class

## How to Test

### Option 1: Quick Visual Test (Recommended)

Run generation with your existing checkpoint:

```bash
# Make sure you're in the medsyn environment
cd /home/mpascual/research/code/medsyn

# Generate 1 sample per class (already configured)
ccddpm-generate config/medsyn_cfg.yaml
```

**Expected output:**
- Directory: `/media/mpascual/PortableSSD/medsyn/PathMNIST_ccDDPM/synth/`
- 9 folders: `class_0/` through `class_8/`
- Each folder contains 1 PNG image
- **Images should show medical tissue, NOT pure noise**

**Debug output to look for:**
```
[INFO] Starting generation for class 0
[INFO]   Initial noise: shape=torch.Size([1, 3, 64, 64]), range=[-2.xxx, 2.xxx]
[INFO]   Scheduler timesteps: 1000 steps
[INFO]   Guidance scale: 2.0
[INFO]   Step 0/1000: t=999, eps_cond_range=[...], eps_uncond_range=[...]
...
[INFO]   Final image: range=[-0.xxx, 0.xxx]
[INFO]   First sample for class 0: range=[-0.xxx, 0.xxx], normalized range=[0.xxx, 0.xxx]
```

### Option 2: Resume Training

If you want to see the fix during training:

```bash
ccddpm-train config/medsyn_cfg.yaml
```

Check the generated visualizations in:
`/home/mpascual/research/medsyn/ccddpm/outputs/samples/`

**Look for:**
- `epoch_XXXX_classes.png` - Should show 9 distinct images (one per class)
- `epoch_XXXX_denoising.png` - Should show 12 steps (noise → image)
- Console logs showing class sample generation

## Interpreting Results

### ✅ Success Indicators

1. **Generated images are NOT pure noise**
   - Should see tissue structures, cell patterns
   - Colors should vary (not random RGB pixels)
   - Images should look like PathMNIST samples

2. **Value ranges are reasonable**
   ```
   Final image: range=[-0.8, 0.9]  # Good - bounded values
   ```

3. **Class samples differ visually**
   - Different tissue types should have different appearances
   - Not all identical noise

### ❌ Failure Indicators (If Still Broken)

1. **Still seeing pure noise**
   ```
   Final image: range=[-2.5, 2.8]  # Bad - unbounded noise
   ```
   → Model may not have learned properly, consider retraining

2. **All classes look identical**
   → Class conditioning may not have worked during training

3. **Values are all zeros or NaN**
   → Model weights may be corrupted

## What If It Still Doesn't Work?

### Scenario 1: Generation still produces noise

**Possible causes:**
- The trained model may not have learned class conditioning properly
- Checkpoint may be corrupted

**Solution:**
```bash
# Check if model weights are valid
python -c "
import torch
ckpt = torch.load('/media/mpascual/PortableSSD/medsyn/PathMNIST_ccDDPM/ckpts/best.pt', weights_only=False)
print('Keys:', ckpt.keys())
print('Has EMA:', 'ema' in ckpt and ckpt['ema'] is not None)
print('Epoch:', ckpt.get('epoch', 'N/A'))
print('Val loss:', ckpt.get('val_loss', 'N/A'))
"
```

If checkpoint is valid but still generating noise, you may need to:
1. Check if training converged properly (review loss curves)
2. Try generating with `guidance_scale: 1.0` instead of 2.0
3. Consider retraining with the fixed code

### Scenario 2: Syntax or import errors

**Solution:**
```bash
# Verify Python syntax
python -m py_compile medsyn/models/ccDDPM/engine/train.py
python -m py_compile medsyn/cli/generate_ccDDPM.py

# If errors, revert and reapply changes manually
```

### Scenario 3: Training visualizations fixed but generation still broken

**Diagnosis:** This would indicate the generation script has a separate issue

**Solution:**
```bash
# Run generation with debug enabled for ALL samples
# Modify generate_ccDDPM.py line 401:
# enable_debug = (class_id == 0 and idx == 0)
# →
# enable_debug = True

# Then run generation and examine the full log
```

## Next Steps

1. **Immediate:** Run generation test (Option 1 above)
2. **Verify:** Check that generated images look like medical tissue
3. **Document:** Save example generated images for comparison
4. **Optional:** If satisfied, run longer generation (update config classes to 100+ per class)
5. **Consider:** If model didn't learn well, retrain with fixed code

## Key Insights from Debugging

1. **Reconstruction worked** → Model CAN denoise
2. **Training metrics good** → Model learned something
3. **Class visualization was noise** → Bug in visualization code
4. **Bug was subtle** → Function ran completely but returned wrong tensor

The model likely learned correctly, but we couldn't see/use the results due to the visualization bug.

## Files to Review

- ✅ `docs/CCDDPM_GENERATION_BUG_FIX.md` - Detailed technical explanation
- ✅ `medsyn/models/ccDDPM/engine/train.py` - Fixed training code
- ✅ `medsyn/cli/generate_ccDDPM.py` - Enhanced generation with logging

## Contact

If issues persist after testing, review:
1. Training loss curves (should be decreasing)
2. PSNR metrics (should be > 15-20 dB)
3. Reconstruction images (should look good)
4. Compare with known working DDPM implementations

Good luck! The fix should resolve your noise generation issue. 🎉
