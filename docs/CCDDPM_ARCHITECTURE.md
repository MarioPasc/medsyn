# ccDDPM Architecture Documentation

**Technical Deep Dive into Class-Conditional DDPM Implementation**

---

## Table of Contents

1. [Mathematical Foundation](#mathematical-foundation)
2. [Model Architecture](#model-architecture)
3. [Training Algorithm](#training-algorithm)
4. [Sampling Algorithm](#sampling-algorithm)
5. [Classifier-Free Guidance](#classifier-free-guidance)
6. [Min-SNR Loss Weighting](#min-snr-loss-weighting)
7. [Implementation Details](#implementation-details)
8. [Design Decisions](#design-decisions)

---

## Mathematical Foundation

### Denoising Diffusion Probabilistic Models (DDPM)

#### Forward Process (Noising)

The forward process gradually adds Gaussian noise to data over T timesteps:

```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1 - ᾱ_t)I)
```

Where:
- `x_0`: Clean image
- `x_t`: Noisy image at timestep t
- `ᾱ_t = ∏_{i=1}^t α_i`: Cumulative product of (1 - β_i)
- `β_t`: Variance schedule

**Closed-form sampling**:
```
x_t = √ᾱ_t x_0 + √(1 - ᾱ_t) ε,  where ε ~ N(0, I)
```

#### Reverse Process (Denoising)

The reverse process learns to remove noise:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**Reparameterization**: Instead of predicting mean directly, predict noise ε:
```
ε_θ(x_t, t) ≈ ε
```

**Training objective** (simplified):
```
L_simple = E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]
```

### Class Conditioning

For class-conditional generation, we condition on class label y:

```
ε_θ(x_t, t, y) ≈ ε
```

**Implementation choice**: Channel concatenation
```
ε_θ([x_t, c(y)], t)
```

Where `c(y)` is the class embedding broadcast to spatial dimensions.

---

## Model Architecture

### Overall Structure

```
Input: x_t [B, 3, H, W], t [B], y [B]
       ↓
  [ClassEmbedder]
       ↓
  y → [B, C_emb, H, W]
       ↓
  Concat: [x_t, c(y)] → [B, 3+C_emb, H, W]
       ↓
    [UNet2D]
       ↓
Output: ε̂ [B, 3, H, W]
```

### ClassEmbedder

**Purpose**: Map discrete class labels to spatial conditioning

```python
class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, emb_channels: int):
        self.emb = nn.Embedding(num_classes, emb_channels)
        # Initialized with N(0, 0.02²)

    def forward(self, labels, shape_hw, device):
        # labels: [B] or None
        # Returns: [B, C_emb, H, W]
        if labels is None:
            return zeros([1, emb_channels, H, W])  # Unconditional

        v = self.emb(labels)  # [B, C_emb]
        v[labels == -1] = 0   # Handle sentinel value
        return v[..., None, None].expand(-1, -1, H, W)
```

**Key features**:
- Handles unconditional generation (`labels=None` or `-1`)
- Broadcast to full spatial resolution
- Small initialization (0.02 std) for stability

### U-Net Architecture

**Based on**: HuggingFace Diffusers `UNet2DModel`

```
Encoder:
  DownBlock2D (128 channels, 2 res blocks)  # 64×64 → 32×32
  DownBlock2D (256 channels, 2 res blocks)  # 32×32 → 16×16
  DownBlock2D (256 channels, 2 res blocks)  # 16×16 → 8×8
  AttnDownBlock2D (512 channels, 2 res blocks, self-attention)  # 8×8 → 4×4

Middle:
  ResNet Block + Attention + ResNet Block

Decoder:
  AttnUpBlock2D (512 channels, 2 res blocks, self-attention)  # 4×4 → 8×8
  UpBlock2D (256 channels, 2 res blocks)   # 8×8 → 16×16
  UpBlock2D (256 channels, 2 res blocks)   # 16×16 → 32×32
  UpBlock2D (128 channels, 2 res blocks)   # 32×32 → 64×64
```

**Architecture hyperparameters**:
- Input channels: `3 + class_embed_dim` (19 for embed_dim=16, 35 for embed_dim=32)
- Output channels: `3`
- Block multipliers: `[1, 2, 2, 4]`
- Layers per block: `2`
- Attention: Only at lowest resolution (32×32 feature map)

**Why attention at lowest resolution?**
- Computational efficiency (attention is O(N²) in spatial size)
- Captures global structure
- Sufficient for 64×64 images (larger images may need attention at multiple scales)

---

## Training Algorithm

### Main Training Loop

```python
for epoch in range(epochs):
    for batch in train_loader:
        x_0, y = batch['pixel_values'], batch['labels']  # x_0 ∈ [-1, 1]

        # 1. Sample timestep uniformly
        t = torch.randint(0, T, (B,))

        # 2. Sample noise
        ε = torch.randn_like(x_0)

        # 3. Add noise (forward process)
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

        # 4. Classifier-free guidance: randomly drop labels
        if rand() < p_uncond:  # p_uncond = 0.1
            y = -1  # Sentinel for unconditional

        # 5. Predict noise
        ε̂ = model(x_t, t, y)

        # 6. Compute loss
        loss = MSE(ε̂, ε)

        # 7. Apply Min-SNR weighting (optional)
        if use_min_snr:
            snr = ᾱ_t / (1 - ᾱ_t)
            weight = min(snr, gamma) / snr
            loss = loss * weight

        # 8. Backprop and update
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        # 9. Update EMA
        ema.update(model)
```

### Key Training Components

#### 1. Noise Schedule

**Default**: `squaredcos_cap_v2` (Karras et al., 2022)

Improved over linear schedule:
- Better SNR (signal-to-noise ratio) distribution
- More balanced training across timesteps
- Better sample quality empirically

**Formula**:
```
f(t) = cos((t/T + s) / (1 + s) * π/2)²
ᾱ_t = f(t) / f(0)
β_t = 1 - ᾱ_t / ᾱ_{t-1}
```

#### 2. Gradient Clipping

**Value**: `10.0` (was `1.0` - too aggressive)

**Why it matters**:
- Too low (1.0): Model gets stuck, learns identity mapping
- Too high (>20): Training instability, divergence
- Sweet spot (5.0-10.0): Allows learning while preventing explosions

**Verification**: Monitor gradient norms
- Should be < clip value most of the time
- If constantly hitting limit → increase clip norm

#### 3. Mixed Precision Training

```python
autocast_dtype = torch.bfloat16 if is_ampere_plus else torch.float16
with autocast(device_type='cuda', dtype=autocast_dtype):
    pred = model(x_t, t, y)
    loss = loss_fn(pred, noise, ...)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
```

**bfloat16 advantages** (Ampere+):
- Better numerical stability than fp16
- Avoids loss scaling issues
- Wider dynamic range

#### 4. EMA (Exponential Moving Average)

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = decay * self.shadow[k] + (1 - decay) * v
```

**Why EMA?**
- Smooths out training noise
- Better sample quality at inference
- Standard practice in diffusion models (Karras et al.)
- Typical improvement: +2-5 FID points

**Usage**: Always use EMA weights for generation

---

## Sampling Algorithm

### Ancestral DDPM Sampling

```python
def sample(model, y, shape, num_steps=1000):
    # Start from pure noise
    x_t = torch.randn(shape)

    # Reverse diffusion
    for t in reversed(range(num_steps)):
        # Predict noise
        ε̂ = model(x_t, t, y)

        # Compute mean
        α_t = alphas[t]
        ᾱ_t = alphas_cumprod[t]
        β_t = betas[t]

        μ = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - ᾱ_t)) * ε̂)

        # Add noise (except at t=0)
        if t > 0:
            σ_t = sqrt(β_t)  # or posterior variance
            x_t = μ + σ_t * torch.randn_like(x_t)
        else:
            x_t = μ

    return x_t  # x_0
```

**Implemented via**: `scheduler.step(ε̂, t, x_t).prev_sample`

### Deterministic vs Stochastic

- **Training**: Uses stochastic formulation (variance term)
- **Sampling**: Can be deterministic (DDIM) or stochastic (DDPM)
- **Current**: Uses stochastic DDPM sampling
- **Future**: Could implement DDIM for faster sampling (20-50 steps)

---

## Classifier-Free Guidance

### Theory

**Idea**: Use both conditional and unconditional models to enhance conditioning

**Training**:
```python
# 90% conditional
ε̂_cond = model(x_t, t, y)

# 10% unconditional
ε̂_uncond = model(x_t, t, None)
```

**Inference**:
```python
ε̂_cond = model(x_t, t, y)
ε̂_uncond = model(x_t, t, None)
ε̂ = ε̂_uncond + w * (ε̂_cond - ε̂_uncond)
```

Where `w` is the guidance scale.

### Guidance Scale Effects

**Derivation**: CFG modifies the score function:
```
∇ log p(x_t | y) ≈ -ε̂_cond / σ_t
∇ log p(x_t) ≈ -ε̂_uncond / σ_t

# Guided score (w > 1 amplifies conditioning):
∇ log p(x_t | y)^w ∝ -[ε̂_uncond + w(ε̂_cond - ε̂_uncond)] / σ_t
```

**Practical effects**:
- `w = 0.0`: Pure unconditional (ignores y)
- `w = 1.0`: Pure conditional (standard model)
- `w > 1.0`: Enhanced conditional (sharper class features)
- `w >> 1.0`: Over-conditioning (artifacts, less diversity)

**Recommended range**: 1.5 - 3.0
- PathMNIST: `2.0` works well
- ImageNet: `3.0-4.0` common
- Text-to-image: `7.5-15.0` typical

### Implementation Optimization

```python
# ❌ WRONG (old code)
if guidance_scale <= 0:
    return eps_cond  # Returns conditional when scale=0!

# ✅ CORRECT (fixed)
if guidance_scale == 1.0:
    return eps_cond  # Pure conditional, skip second pass
eps_uncond = model(x_t, t, None)
return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

**Why check for 1.0?**
- Saves 50% compute (one forward pass instead of two)
- Numerically equivalent to the formula
- Common case in ablations

---

## Min-SNR Loss Weighting

### Motivation

**Problem**: Different timesteps have vastly different noise levels
- Early timesteps (t→T): Almost pure noise, low SNR
- Late timesteps (t→0): Almost clean, high SNR

**Issue**: Model can "cheat" by focusing on easy (high SNR) steps

### Algorithm

```python
def min_snr_loss(pred, target, t, alphas_cumprod, gamma=5.0):
    # Compute SNR
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])

    # Clamp to max value
    snr_clamped = torch.clamp(snr, max=gamma)

    # Compute weight
    weight = snr_clamped / snr

    # Apply to loss
    loss = F.mse_loss(pred, target, reduction='none')
    return (loss * weight.view(-1, 1, 1, 1)).mean()
```

### Effect

**Without Min-SNR**:
```
t=0   (SNR=1000): weight = 1.0, heavily weighted
t=500 (SNR=1):    weight = 1.0, moderately weighted
t=999 (SNR=0.01): weight = 1.0, barely contributes
```

**With Min-SNR (gamma=5)**:
```
t=0   (SNR=1000): weight = 5/1000 = 0.005, downweighted!
t=500 (SNR=1):    weight = 1/1 = 1.0, normal
t=999 (SNR=0.01): weight = 0.01/0.01 = 1.0, upweighted!
```

**Result**: More balanced training across timesteps

### Hyperparameter: gamma

- `gamma = 1.0`: Extreme balancing (may over-correct)
- `gamma = 5.0`: **Recommended** (Hang et al., 2023)
- `gamma = 10.0`: Mild balancing
- `gamma = ∞`: No balancing (standard MSE)

**Impact on training**:
- Faster convergence (fewer epochs to good quality)
- Better sample fidelity (especially in high-noise regions)
- More stable training (less variance in loss)

---

## Implementation Details

### Prediction Types

**Current**: `epsilon` (noise prediction)

```python
# Model predicts:
ε̂ = model(x_t, t, y)

# Loss:
L = ||ε̂ - ε||²
```

**Alternative**: `v_prediction` (velocity prediction)

```python
# Model predicts:
v̂ = model(x_t, t, y)

# Where v is defined as:
v = sqrt(ᾱ_t) * ε - sqrt(1 - ᾱ_t) * x_0

# Loss:
L = ||v̂ - v||²
```

**⚠️ Critical**: Config `prediction_type` MUST match loss computation!
- Current code assumes `epsilon`
- Changing to `v_prediction` requires updating loss.py and diagnostics

**Why v_prediction?**
- Better numerical stability for high resolution
- Used in Stable Diffusion 2.0
- More complex to implement correctly

### Data Normalization

**Images**: Normalized to `[-1, 1]`

```python
# Loading
img = img.float() / 255.0  # [0, 1]
img = (img - 0.5) / 0.5    # [-1, 1]

# Or with transforms
T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# Saving
img = (img + 1.0) / 2.0  # [-1, 1] → [0, 1]
save_image(img, path)
```

**Why [-1, 1]?**
- Symmetric around 0
- Better gradient flow
- Standard in diffusion literature

### Timestep Encoding

**Method**: Sinusoidal positional encoding (built into UNet2D)

```python
def timestep_embedding(t, dim):
    half_dim = dim // 2
    emb = log(10000) / (half_dim - 1)
    emb = exp(torch.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    return torch.cat([sin(emb), cos(emb)], dim=-1)
```

**Properties**:
- Provides smooth encoding of discrete timesteps
- Allows model to distinguish different noise levels
- Injected into ResNet blocks via AdaGN or similar

---

## Design Decisions

### 1. Why Channel Concatenation for Class Conditioning?

**Alternatives considered**:
1. **AdaGN** (Adaptive Group Normalization): Modulate activations
2. **Cross-attention**: Attend to class embedding
3. **Concatenation**: Our choice

**Why concatenation?**
- ✅ Simple and robust
- ✅ Works well for discrete labels
- ✅ Easy to implement unconditional (just zeros)
- ❌ Increases input channels (minor overhead)

**When to use alternatives?**
- AdaGN: Continuous conditioning (e.g., text embeddings)
- Cross-attention: Complex multi-modal inputs
- Concatenation: Discrete labels (our case)

### 2. Why Attention at Lowest Resolution Only?

**Trade-off**: Computation vs Expressiveness

**Attention complexity**: O(N²) where N = H × W

For 64×64 image with 4 downsampling levels:
- Level 1 (32×32): 1024² = 1M operations
- Level 2 (16×16): 256² = 65K operations
- Level 3 (8×8): 64² = 4K operations
- Level 4 (4×4): 16² = 256 operations ← We use this

**Decision**: Attention at 8×8 feature map (4×4 after downsample)
- Good enough for capturing global structure
- Minimal computational overhead
- Proven effective for 64-128px images

**For larger images** (256+):
- Add attention at multiple resolutions
- See Stable Diffusion architecture

### 3. Why EMA Decay = 0.999?

**EMA update**: `θ'_ema = 0.999 * θ_ema + 0.001 * θ_current`

**Effective window**: ~1000 steps = 1/(1-decay)

**For batch_size=4, dataset_size=100K**:
- Steps per epoch: 25K
- EMA window: ~1 epoch

**Why 0.999?**
- ✅ Standard in literature
- ✅ Balances smoothness vs recency
- ✅ Proven to work across datasets

**Alternatives**:
- 0.9999: More smoothing (very large datasets)
- 0.99: Less smoothing (small datasets, faster adaptation)

### 4. Why 1000 Timesteps?

**DDPM default**: T = 1000

**Trade-offs**:
- **More timesteps** (T=4000):
  - ✅ Finer noise schedule
  - ❌ Slower sampling (4x slower)
  - ❌ Diminishing returns

- **Fewer timesteps** (T=100-250):
  - ✅ Faster sampling
  - ❌ Quality degradation
  - ❌ Requires careful schedule tuning

**Best practice**:
- Train with T=1000
- Sample with T=1000 for quality
- Use DDIM/DPM-Solver for faster sampling (same model)

---

## Performance Considerations

### Memory Usage

**Model size**:
- Parameters: ~330K (very small compared to modern models)
- Memory: ~10MB for weights
- Activation memory (batch=4, 64×64):
  - Forward pass: ~500MB
  - Backward pass: ~1.5GB
  - Total: ~2GB VRAM

**Scaling**:
- Batch size × 2 → Memory × ~2
- Image size × 2 → Memory × ~4 (due to U-Net architecture)

### Training Speed

**On RTX 3090**:
- Batch size 4, 64×64: ~100 images/sec
- Epoch (100K images): ~15-20 minutes
- Full training (80 epochs): ~20-24 hours

**Bottlenecks**:
1. Data loading (mitigated by num_workers=8, pin_memory=True)
2. Forward/backward pass (use mixed precision)
3. EMA update (negligible)

### Generation Speed

**On RTX 3090, 1000 steps**:
- Single image: ~5-10 seconds
- With CFG (scale=2.0): ~10-15 seconds (2x forward passes)
- Batch generation (100 images): ~10-15 minutes

**Optimization strategies**:
1. Use DDIM (20-50 steps, same quality)
2. Distillation (train student to match in fewer steps)
3. Consistency models (single-step generation)

---

## Future Improvements

### Short-term

1. **DDIM Sampler**: 20-50 steps for faster generation
2. **Guidance scale sweep**: Automatic quality/diversity trade-off
3. **FID tracking**: Monitor generation quality during training

### Medium-term

1. **Latent diffusion**: Compress to latent space (like Stable Diffusion)
2. **v_prediction**: Better stability for high resolution
3. **Multi-resolution attention**: For larger images (128+)

### Long-term

1. **Consistency models**: Single-step generation
2. **Flow matching**: Alternative to diffusion (simpler training)
3. **Conditional augmentation**: Text or attribute conditioning

---

## References

### Core Papers

1. Ho et al. (2020) - DDPM: https://arxiv.org/abs/2006.11239
2. Ho & Salimans (2022) - CFG: https://arxiv.org/abs/2207.12598
3. Hang et al. (2023) - Min-SNR: https://arxiv.org/abs/2303.09556
4. Dhariwal & Nichol (2021) - Improved DDPM: https://arxiv.org/abs/2105.05233
5. Karras et al. (2022) - Design Space: https://arxiv.org/abs/2206.00364

### Implementation References

- HuggingFace Diffusers: https://github.com/huggingface/diffusers
- OpenAI Guided Diffusion: https://github.com/openai/guided-diffusion
- Annotated DDPM: https://nn.labml.ai/diffusion/ddpm/index.html

---

**Author**: M. Pascual-González
**Last Updated**: 2025-10-29
**Version**: 2.0
