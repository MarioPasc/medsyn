# Second experiment with AdamW and Batch 64

Changes:




## 1. Training Dynamics Analysis

### 1.1 Loss Convergence

| Metric | Epoch 1 | Epoch 10 | Epoch 50 (Best) | Epoch 58 (Final) |
|--------|---------|----------|-----------------|------------------|
| Train Loss | 0.193 | 0.018 | 0.016 | 0.016 |
| Val Loss | 0.038 | 0.016 | 0.016 | 0.016 |
| Val PSNR | 14.2 dB | 19.8 dB | 21.0 dB | 21.2 dB |
| Val SSIM | 0.346 | 0.526 | 0.571 | 0.578 |

**Observations:**
1. **Rapid initial learning** (epochs 1-10): Loss drops 10× during warmup—excellent warmup behavior
2. **Plateau after warmup** (epochs 10-58): Only marginal improvement (~0.5 dB PSNR gain in 48 epochs)
3. **No overfitting**: Train and Val metrics are nearly identical (Val slightly higher, which is unusual)

### 1.2 Learning Rate Schedule Analysis

From the step-level log, the LR schedule follows:
$$\text{LR}(t) = \begin{cases} 
\text{LR}_{\text{start}} + \frac{t}{T_{\text{warmup}}} \cdot (\text{LR}_{\text{max}} - \text{LR}_{\text{start}}) & t \leq T_{\text{warmup}} \\
\eta_{\text{min}} + \frac{1}{2}(\text{LR}_{\text{max}} - \eta_{\text{min}})(1 + \cos(\pi \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}})) & t > T_{\text{warmup}}
\end{cases}$$

With your configuration:
- `warmup_epochs=10` → $T_{\text{warmup}}$ = 7,000 steps (700 steps/epoch × 10)
- `lr=1e-4` → $\text{LR}_{\text{max}} = 10^{-4}$
- `warmup_start_factor=0.01` → $\text{LR}_{\text{start}} = 10^{-6}$
- `eta_min=1e-7`

**Issue:** The 10-epoch warmup (10% of 100 epochs) may be too short. The plateau at epoch 10-15 suggests the model hadn't fully "warmed up" before aggressive cosine decay began.

---

## 2. Critical Issues Identified

### 2.1 🔴 Full-Chain PSNR Instability (HIGH PRIORITY)

Looking at your **Reconstruction Quality at Different Timesteps** plot (Image 5):

| Metric | Min | Max | Std Dev |
|--------|-----|-----|---------|
| PSNR @ t=100 | 23.0 dB | 24.5 dB | ~0.5 dB (stable) |
| PSNR @ t=500 | 15.9 dB | 16.3 dB | ~0.2 dB (stable) |
| Full Chain PSNR | 16.8 dB | 23.3 dB | **~2.5 dB (unstable!)** |

**Diagnosis:** The full denoising chain (1000 steps from pure noise to image) has high variance. This indicates:

1. **Stochastic sampling instability**: Small errors at high-noise timesteps compound during the 1000-step reverse process
2. **Mid-timestep weakness**: The model may not be accurately predicting noise at $t \in [200, 800]$

**Root cause from ELBO diagnostics:**
```
SNR by Timestep Region (middle panel, Image 4):
- Low t (<333):  SNR varies wildly 100-800 (unstable!)
- Mid t (333-666): SNR ≈ 1 (stable)
- High t (>666):  SNR ≈ 0 (stable, as expected)
```

The erratic SNR at low-t suggests the model's noise prediction at low-noise regimes has high variance, which cascades through the entire sampling chain.

### 2.2 🟡 Early Training Plateau (MEDIUM PRIORITY)

From the quality metrics plot (Image 2), the PSNR/SSIM curves **flatten around epoch 20-25**, with only marginal gains afterward. The early stopping triggered at epoch 58 (patience=8 after best at epoch 50).

**Mathematical perspective:** The loss function
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ w(t) \cdot \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) \|^2 \right]$$

has plateaued at ~0.016, suggesting the model has found a local minimum in the noise prediction objective but hasn't achieved optimal sample quality.

### 2.3 🟡 Per-Class Performance Gap (MEDIUM PRIORITY)

From your per-class loss plot (Image 3) and metrics:

| Class | Name | Val PSNR (dB) | Val SSIM | Val Loss (Raw) |
|-------|------|---------------|----------|----------------|
| 1 | Debris | **32.9** | **0.80** | 0.017 |
| 0 | Adipose | 26.4 | 0.73 | 0.024 |
| 5 | Lymphocytes | 24.4 | 0.55 | 0.056 |
| 3 | Mucus | **21.3** | **0.53** | 0.077 |

**Gap:** Class 1 achieves 32.9 dB while Class 3 only reaches 21.3 dB—a **11.6 dB difference!**

Despite per-class weighting (temperature=1.2), the gap persists. This suggests:
1. **Intrinsic class complexity difference**: Mucus textures may be inherently harder to model
2. **Class imbalance not fully addressed**: Weight temperature may need to be higher
3. **Class embedding optimization**: Some classes may have sub-optimal embedding representations

---

## 3. Recommended Fixes

### 3.1 Address Full-Chain Instability

**Option A: Increase inference steps with DDIM**
```yaml
infer:
  num_inference_steps: 100  # Use DDIM instead of full 1000-step DDPM
  # Requires implementing DDIMScheduler for inference
```

**Option B: Add variance reduction during sampling**
```python
# In inference: use deterministic sampling at low-noise timesteps
if t < 100:
    variance = 0.0  # No stochastic noise for t < 100
```

**Option C: Train with loss on more timestep regions**
Ensure uniform timestep sampling during training (you already have this, but verify with diagnostics).

### 3.2 Address Training Plateau

**Recommendation 1: Extend warmup and use cyclic LR**
```yaml
lr_scheduler:
  type: "linear_warmup_cosine_annealing_lr"
  warmup_epochs: 20        # Increased from 10 (20% of 100 epochs)
  warmup_start_factor: 0.001  # Start even lower
  eta_min: 1.0e-6          # Slightly higher minimum
```

**Recommendation 2: Increase training budget**
Your model plateaued at epoch 25 of 100. With 1000 epochs (your original target), you'd have more room to escape local minima. For a quick test:
```yaml
train:
  epochs: 200
early_stopping:
  patience: 20  # Increase patience for longer training
```

**Recommendation 3: Try different optimizer**
Lion optimizer has shown faster convergence in some diffusion settings:
```yaml
optimizer:
  type: "lion"
  lr: 1.0e-5        # 10× smaller than AdamW
  wd: 0.1           # 10× larger than AdamW
  betas: [0.9, 0.99]
```

### 3.3 Address Per-Class Gap

**Recommendation 1: Increase class weight temperature**
```yaml
train:
  per_class_weight_temperature: 1.5  # More aggressive (was 1.2)
```

**Recommendation 2: Implement focal-style weighting**
Add loss focusing on hard examples:
```python
# In loss.py, add focal weighting
gamma_focal = 2.0
p_t = 1.0 - torch.sqrt(per_example_mse)  # "confidence"
focal_weight = (1 - p_t) ** gamma_focal
weighted_loss = per_example_mse * focal_weight
```

**Recommendation 3: Class-stratified sampling**
Ensure each batch has balanced class representation:
```python
from torch.utils.data import WeightedRandomSampler
# Create sampler that ensures equal class probability per batch
```

---

## 4. Configuration Optimizations

### 4.1 Verified Correct Settings ✅

| Setting | Value | Status |
|---------|-------|--------|
| Symmetric attention | ✅ Fixed | Correct |
| attention_head_dim | 32 | Correct (4-8 heads per block) |
| resnet_time_scale_shift | scale_shift | Correct |
| beta_schedule | squaredcos_cap_v2 | Correct |
| min_snr_gamma | 3.0 | Good (conservative) |
| guidance_p_uncond | 0.1 | Optimal |
| class_embed_dim | 64 | Generous (simplex needs only 8) |

### 4.2 Suggested Changes

```yaml
# ===== CHANGES TO config_AdamW_Batch64.yaml =====

train:
  epochs: 200  # Increase from 100 for more optimization time
  
  early_stopping:
    patience: 15  # Increase from 8
    
  # Optional: increase class temperature
  per_class_weight_temperature: 1.5  # Was 1.2

lr_scheduler:
  warmup_epochs: 25  # Was 10 (now 12.5% of 200 epochs)
  warmup_start_factor: 0.001  # Was 0.01 (gentler start)
  eta_min: 5.0e-7  # Was 1e-7 (slightly higher floor)

# Optional: Try gamma=5 for Min-SNR
train:
  min_snr_gamma: 5.0  # Was 3.0 (original recommendation)
```

### 4.3 Gradient Accumulation Note

Your config shows `grad_accum_steps: 4` with `batch_size: 64` on 2 GPUs. This gives:
$$\text{Effective batch size} = 64 \times 2 \times 4 = 512$$

This is quite large for ~90K samples (175 steps/epoch). Consider:
- Reducing to `grad_accum_steps: 2` for 256 effective batch size (more frequent updates)
- Or keeping 512 but increasing epochs to compensate for fewer updates per epoch



## 6. Priority Action Items

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| 🔴 HIGH | Investigate full-chain PSNR instability—add DDIM sampler for inference | Stable, higher-quality samples |
| 🟡 MEDIUM | Extend warmup to 20-25 epochs | Better exploration before decay |
| 🟡 MEDIUM | Increase training epochs to 200 with patience=15 | Escape local minima |
| 🟢 LOW | Increase per-class weight temperature to 1.5 | Reduce per-class gap |
| 🟢 LOW | Test Lion optimizer | Potentially faster convergence |

