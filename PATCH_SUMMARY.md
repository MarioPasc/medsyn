# VAE Collapse Fix - Patch Summary

## Overview
Applied critical fixes to address posterior collapse in the conditional β-VAE implementation. These patches align the implementation with PyTorch-VAE best practices while preserving advanced features for later use.

## Root Causes Identified

1. **Data range vs loss mismatch**: SSIM with `data_range=1.0` on potentially [-1,1] normalized data
2. **Over-constraint early KL**: Capacity control with large γ while warming β pins KL≈0
3. **Encoder-extra steps update prior**: "Encoder-only" steps were updating class prior embeddings
4. **Param-group bug**: Using `set(prior_params)` for filtering can fail due to hashing issues
5. **Over-conditioning**: FiLM + learned class-prior stronger than needed for initial training

## Files Modified

### 1. `medsyn/models/bVAE/loss.py`
- **Added data range auto-detection**: Rescales [-1,1] inputs to [0,1] for BCE/SSIM
- **Clarified capacity vs free-bits**: Updated comment to reflect mutual exclusivity
- **Hardened SSIM**: Ensures proper data range handling

### 2. `medsyn/models/bVAE/engine/train.py`
- **Fixed param groups**: Use id-based filtering instead of set() for prior params
- **Added encoder-only optimizer**: Separate `enc_opt` for true encoder-only updates
- **Detached prior in encoder steps**: Prevents class prior updates during lagging-inference mitigation
- **Updated gradient clipping**: Clip only `enc_params` during encoder-only steps
- **Updated scheduler comment**: Clarified safety cap purpose

### 3. `medsyn/models/bVAE/config.py`
- **Disabled class prior by default**: `use_class_prior: bool = False`
- **Enabled encoder extra steps**: `encoder_extra_steps: int = 2`
- **Reduced encoder heavy epochs**: `encoder_heavy_epochs: int = 5`
- **Changed default recon loss**: `recon_type = "l1"` (SSIM added after stability)
- **Disabled capacity control**: `use: bool = False` (avoid early KL pinning)
- **Reduced capacity gamma**: `gamma: float = 50.0` (gentler pull when enabled)
- **Increased capacity ramp**: `steps_to_max: int = 30000`

### 4. `medsyn/models/bVAE/model.py`
- **Added geometry assertion**: Validates `img_size` divisibility by `2**num_down`

### 5. `config/medsyn_cfg.yaml`
- **Disabled class prior**: `use_class_prior: false`
- **Updated encoder extra steps**: `encoder_extra_steps: 2`
- **Reduced encoder heavy epochs**: `encoder_heavy_epochs: 5`
- **Changed recon loss**: `recon_type: "l1"`
- **Disabled capacity control**: `use: false`
- **Reduced capacity gamma**: `gamma: 50.0`

## Training Strategy

### Phase 1: Vanilla c-VAE (Current Configuration)
- `conditioning="film"` (keep)
- `use_class_prior=False` (disabled)
- `recon_type="l1"` (simple)
- `capacity.use=False` (disabled)
- **Goal**: Achieve stable reconstruction (PSNR > 20 dB)

### Phase 2: Add Capacity Control (After Phase 1 Success)
- Enable `capacity.use=True`
- Keep other settings from Phase 1
- **Goal**: Improve latent organization without collapse

### Phase 3: Advanced Features (After Phase 2 Success)
- Enable `use_class_prior=True` with 0.1× LR
- Switch to `recon_type="l1_ssim"`
- Enable InfoVAE-MMD if needed
- **Goal**: Class-conditional generation with high quality

## Key Improvements

1. **Lagging inference mitigation**: True encoder-only updates with detached prior
2. **Robust data handling**: Auto-rescaling for range mismatches
3. **Safe optimization**: ID-based param filtering, separate optimizers
4. **Progressive complexity**: Start simple, add features incrementally
5. **Numerical stability**: Geometry checks, clamping, NaN guards

## Next Steps

1. **Train with current config**: Should see improving recon/PSNR within 2-3 epochs
2. **Monitor KL**: Should gradually rise from ~0 (not stay pinned at 0)
3. **Check encoder extra steps**: Logs should show they're active in first 5 epochs
4. **After stable recon**: Enable capacity control progressively
5. **Finally**: Add class prior and SSIM when baseline is solid

## References
- Lagging inference: He et al. "Lagging Inference Networks" (arXiv)
- Capacity control: Burgess et al. "Understanding disentangling in β-VAE" (arXiv)
- Free-bits: Kingma et al. "Improved Variational Inference" (arXiv)
- InfoVAE: Zhao et al. "InfoVAE: Balancing Learning and Inference" (CVF)

---
**Applied**: October 14, 2025
**Status**: Ready for training
