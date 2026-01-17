# IJCNN IEEE 2026 Action Plan: MedSyn (CFG-MedMNIST)

**Title**: Utility-Aware Diffusion Augmentation for Imbalanced Medical Datasets
**Target Conference**: IJCNN IEEE 2026
**Document Date**: January 2026
**Status**: Pre-submission preparation

---

## Executive Summary

This document provides a comprehensive action plan to prepare the MedSyn project for IJCNN IEEE 2026 submission. It identifies:

1. **Methodological Flaws** - Scientific issues requiring theoretical justification or experimental validation
2. **Code Flaws** - Bugs, code smells, and implementation concerns
3. **Must-Dos for Acceptance** - Critical experiments, analyses, and paper requirements

Based on the preliminary paper analysis (FID table showing improvements over DistDiff, F1 classification results), the core contribution is solid but requires additional rigor for a top venue.

---

## Part 1: Methodological Flaws (Scientific Issues)

### 1.1 Class Conditioning via Channel Concatenation

**Current Implementation**: Class embeddings are spatially broadcast and concatenated to the noisy image input (model.py:107-108).

**Issue**: This is a simpler approach compared to cross-attention conditioning used in state-of-the-art diffusion models (Stable Diffusion, DALL-E). Channel concatenation:
- Only provides class information at the input level
- Does not allow class information to modulate attention at different resolutions
- May limit the model's expressiveness for fine-grained class distinctions

**Severity**: MEDIUM

**Action Required**:
- [ ] **Option A (Minimal)**: Justify this design choice explicitly in the paper. Cite Ho et al. (2020) original DDPM which uses similar conditioning.
- [ ] **Option B (Stronger)**: Implement cross-attention conditioning and compare empirically in ablation study.
- [ ] **Option C (Recommended)**: Acknowledge limitation in discussion, argue that for small image sizes (64x64) the simpler method suffices.

**References**:
- Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models" (cross-attention)
- Nichol & Dhariwal (2021), "Improved Denoising Diffusion Probabilistic Models" (AdaGN)

---

### 1.2 Temperature-Based Class Weighting Formulation

**Current Implementation** (math.py:268-334):
```python
# weight = inv_freq ^ temperature
weights = np.power(inv_freq, temperature)
weights = weights / weights.mean()  # normalize to mean=1
```

**Issues**:
1. **No theoretical justification** for why power-law smoothing of inverse frequency is optimal
2. **Temperature interpretation** differs from typical softmax temperature (which operates on logits, not frequencies)
3. **Normalization to mean=1** may not preserve intended relative magnitudes across different temperatures

**Severity**: HIGH (core contribution of the paper)

**Action Required**:
- [ ] Provide mathematical derivation connecting temperature parameter to expected gradient magnitudes
- [ ] Clarify relationship (or lack thereof) to softmax temperature
- [ ] Compare with alternative weighting schemes:
  - Effective number of samples (Cui et al., 2019)
  - Focal loss reweighting (Lin et al., 2017)
  - Square-root frequency weighting
- [ ] Justify normalization choice

**Recommended Addition to Paper**:
> "The temperature parameter $\tau$ controls the trade-off between uniform weighting ($\tau \to 0$) and aggressive inverse-frequency weighting ($\tau > 1$). We empirically observe that $\tau \in [1.0, 2.0]$ provides optimal balance between minority class enhancement and majority class preservation."

---

### 1.3 Min-SNR + Per-Class Weighting Interaction

**Current Implementation**: Both Min-SNR timestep weighting AND per-class loss weighting are applied simultaneously (loss.py:240):
```python
total_weight = snr_weight * class_weight_per_example
```

**Issues**:
1. **No analysis** of how these two weighting schemes interact
2. **Potential compounding effects**: A minority class sample at a high-SNR timestep gets double down-weighting
3. **Hyperparameter coupling**: Optimal temperature may depend on whether Min-SNR is enabled

**Severity**: MEDIUM-HIGH

**Action Required**:
- [ ] **Critical Ablation**: Run 2x2 grid (Min-SNR on/off) x (Class weighting on/off)
- [ ] Analyze loss landscape under each configuration
- [ ] Report per-class loss distribution under each setting
- [ ] Recommend best combination with justification

**Expected Results Table**:
| Min-SNR | Class Weight | Avg FID | Minority FID | F1 |
|---------|--------------|---------|--------------|-----|
| No      | No           | ?       | ?            | ?   |
| Yes     | No           | ?       | ?            | ?   |
| No      | Yes          | ?       | ?            | ?   |
| Yes     | Yes          | Current | Current      | Current |

---

### 1.4 FID Computation Methodology

**Current Implementation** (metrics/fid.py):
- Per-class FID with bootstrap subsets
- Uses torchmetrics InceptionV3
- Default min_samples=50 per class

**Issues**:
1. **Sample size sensitivity**: FID is known to be biased with small sample sizes (<10k)
2. **Feature extractor mismatch**: InceptionV3 trained on ImageNet may not capture medical image features well
3. **Reference distribution**: Paper should explicitly state FID is computed against training set
4. **Minimum samples**: 50 samples per class is too low for reliable FID

**Severity**: MEDIUM

**Action Required**:
- [ ] Increase minimum samples to at least 500 per class
- [ ] Report sample sizes used for each class FID
- [ ] Consider adding FID with medical-domain feature extractor (MedCLIP, RadImageNet)
- [ ] Add KID (Kernel Inception Distance) which is unbiased for small samples
- [ ] Explicitly state in paper: "FID computed between N generated samples and M training set samples"

**Reference**:
- Jayasumana et al. (2024), "Rethinking FID: Towards a Better Evaluation Metric for Image Generation"

---

### 1.5 Statistical Testing Methodology

**Current Implementation** (from PDF Figure 2):
- One-sided paired t-tests with Holm-Bonferroni correction
- Cohen's d reported for effect sizes
- 95% confidence intervals shown

**Issues**:
1. **Number of folds/runs not specified**: How many seeds/folds? 5? 10?
2. **One-sided test assumption**: Assumes direction of improvement is known a priori
3. **Independence assumption**: Paired t-test assumes paired observations (same val/test splits?)

**Severity**: MEDIUM

**Action Required**:
- [ ] Explicitly state number of random seeds (recommend N>=5)
- [ ] Report all seeds used for reproducibility
- [ ] Justify one-sided vs two-sided test choice
- [ ] Consider Wilcoxon signed-rank test as non-parametric alternative
- [ ] Report raw p-values before and after Bonferroni correction

---

### 1.6 Missing Critical Ablations

**Currently Missing**:

| Ablation | Status | Priority |
|----------|--------|----------|
| Guidance scale during generation | MISSING | **CRITICAL** |
| Class embedding dimension | MISSING | HIGH |
| Number of diffusion timesteps | MISSING | MEDIUM |
| Beta schedule (linear vs cosine) | MISSING | MEDIUM |
| Comparison with other conditioning methods | MISSING | HIGH |
| UNet architecture variations | MISSING | LOW |

**Severity**: HIGH (blocking for acceptance)

**Action Required**:
- [ ] **Guidance Scale Ablation** (CRITICAL):
  - Test scales: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
  - Report: FID, per-class FID, downstream F1
  - Current config shows guidance_scale=2.0 but no justification

- [ ] **Class Embedding Dimension Ablation**:
  - Test dimensions: [8, 16, 32, 64]
  - Current: 16 (no justification)

- [ ] **Temperature + Guidance Scale Interaction**:
  - 2D grid search: temperature x guidance_scale
  - Find Pareto-optimal configurations

---

### 1.7 Evaluation Gaps

**Current Evaluation**:
- Single dataset: PathMNIST
- Metrics: FID (global and per-class), F1 score
- Classifiers: ResNet-50, CLIP-ViT-B16, DINOv3

**Missing**:

| Gap | Impact | Action |
|-----|--------|--------|
| Single dataset | HIGH | Add DermaMNIST or BloodMNIST |
| No diversity metric | HIGH | Add LPIPS variance, sample diversity |
| No mode collapse check | HIGH | Analyze generated sample clusters |
| No perceptual metrics | MEDIUM | Add LPIPS, MS-SSIM |
| No human evaluation | MEDIUM | Small-scale expert evaluation |
| No inference time | LOW | Report generation speed |

**Severity**: HIGH

**Action Required**:
- [ ] **Second Dataset** (CRITICAL): Run full pipeline on DermaMNIST or BloodMNIST
- [ ] **Mode Collapse Analysis**:
  - t-SNE/UMAP of generated vs real samples per class
  - Compute intra-class diversity (average pairwise distance)
  - Check for duplicate/near-duplicate generations
- [ ] **Add LPIPS**: Measure perceptual quality
- [ ] **Inception Score**: Standard generative model metric

---

### 1.8 Classifier-Free Guidance Specification

**Issue**: The paper and PDF do not clearly specify the guidance scale used for generating synthetic samples.

**Current Config** (config.py:333):
```python
guidance_scale: float = 2.0
```

**Impact**: This is a critical hyperparameter. Without specification:
- Results are not reproducible
- Reviewers will question robustness

**Severity**: HIGH

**Action Required**:
- [ ] Explicitly state guidance scale in paper (Methods section)
- [ ] Perform guidance scale ablation (see 1.6)
- [ ] Report guidance scale in all result tables

---

### 1.9 Reproducibility Gaps

**Missing Information**:
- Total training epochs used for final model
- Number of generated samples per class for FID
- Validation set used for hyperparameter selection
- Complete hyperparameter table

**Action Required**:
- [ ] Create Table: "Training Hyperparameters"
- [ ] Create Table: "Generation Hyperparameters"
- [ ] Specify: number of epochs, early stopping configuration
- [ ] Provide code/config files as supplementary material

---

## Part 2: Code Flaws (Bugs and Code Smells)

### 2.1 Critical Bugs

#### Bug 1: PSNR/SSIM Clamping Range (train.py:762)

**Location**: `medsyn/models/ccDDPM/engine/train.py:762`

**Code**:
```python
x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
```

**Issue**: Images are normalized to [-1, 1], but clamping to [-10, 10] allows extreme values that will distort PSNR/SSIM metrics.

**Fix**:
```python
x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
```

**Severity**: MEDIUM (affects logged metrics but not training)

---

#### Bug 2: Gradient Clipping Default (config.py:311)

**Current**:
```python
grad_clip_norm: Optional[float] = 1.0
```

**Issue**: Gradient clipping at 1.0 may be too aggressive for diffusion models. Nichol & Dhariwal (2021) use no clipping; Hang et al. (2023) recommend 10.0.

**Recommendation**: Change default to `10.0` or `None`, and ablate.

**Severity**: LOW-MEDIUM

---

#### Bug 3: SSIM Fallback Accuracy (metrics.py:56-69)

**Code**:
```python
# Fallback: simple correlation-based approximation
logger.warning("torchmetrics not available, using simplified SSIM approximation")
```

**Issue**: The fallback SSIM implementation is a rough approximation that does not implement the full SSIM formula (no Gaussian window, no local computation).

**Recommendation**:
- Make torchmetrics a hard dependency, OR
- Implement proper SSIM fallback with Gaussian weighting

**Severity**: MEDIUM (if torchmetrics not installed, metrics are incorrect)

---

### 2.2 Code Smells

#### Smell 1: Magic Numbers

**Locations**:
- `math.py:69`: `for t_val in [100, 500]` - hardcoded timesteps
- `train.py:1337`: `torch.randint(100, scfg.num_train_timesteps // 2, ...)` - magic constants
- `fid.py:479`: `min_samples: int = 50` - too low for reliable FID

**Recommendation**: Move all magic numbers to configuration or constants file.

---

#### Smell 2: Duplicate Code in FID Computation

**Files**: `metrics.py` and `analysis/ddpm_performance/metrics/fid.py`

**Issue**: FID computation logic is duplicated with slight variations.

**Recommendation**: Consolidate into single FID utility module.

---

#### Smell 3: Inconsistent Tensor Range Handling

**Issue**: Different parts of code expect different ranges:
- Training: [-1, 1]
- PSNR/SSIM: [0, 1]
- FID (torchmetrics): [0, 255] uint8

**Current Handling**: Scattered conversion code.

**Recommendation**: Create explicit `normalize_for_fid()`, `normalize_for_metrics()` utilities.

---

### 2.3 Missing Error Handling

#### Issue 1: Empty Class in FID Computation

**Location**: `metrics.py:180-188`

**Current**:
```python
if mask.sum() > 0:
    # compute metrics
else:
    result[f"psnr_c{k}"] = float("nan")
```

**Issue**: Silently returns NaN without logging which class is missing.

**Recommendation**: Add explicit warning with class index.

---

#### Issue 2: Non-finite Loss Handling

**Location**: `train.py:669-681`

**Current**: Logs warning and either skips or raises error.

**Issue**: No summary of how many batches were skipped at epoch end.

**Recommendation**: Track and log `num_skipped_batches` per epoch.

---

### 2.4 Performance Issues

#### Issue 1: FID Memory Usage (metrics.py:543-547)

**Code**:
```python
for i in range(images.size(0)):
    img = images[i]
    save_image(img, out_dir / f"{i:06d}.png")
```

**Issue**: Saves each image individually to disk for FID computation. Very slow for large sample sets.

**Recommendation**: Use torchmetrics in-memory FID computation (already implemented but not always used).

---

#### Issue 2: Sequential Class-Conditional Sampling (train.py:1398-1421)

**Current**: Generates class samples one at a time in a loop.

**Recommendation**: Batch all classes together for parallel generation (already supported by generation scripts, but not in training visualizations).

---

## Part 3: Must-Dos for IJCNN IEEE 2026 Acceptance

### Priority Legend
- **P0 (Blocking)**: Without this, paper will likely be rejected
- **P1 (Critical)**: Strong expectation from reviewers
- **P2 (Important)**: Significantly strengthens paper
- **P3 (Nice-to-have)**: Differentiates from competitors

---

### 3.1 Experiments: P0 (Blocking)

| # | Experiment | Status | Effort | Description |
|---|------------|--------|--------|-------------|
| E1 | Guidance scale ablation | TODO | Medium | Test scales 1.0-4.0, report FID + F1 |
| E2 | Second MedMNIST dataset | TODO | High | Full pipeline on DermaMNIST or BloodMNIST |
| E3 | Min-SNR + Class weight interaction | TODO | Medium | 2x2 ablation grid |
| E4 | Temperature ablation with theoretical motivation | PARTIAL | Low | Add justification to existing results |

---

### 3.2 Experiments: P1 (Critical)

| # | Experiment | Status | Effort | Description |
|---|------------|--------|--------|-------------|
| E5 | Class embedding dimension ablation | TODO | Medium | Test [8, 16, 32, 64] |
| E6 | Mode collapse / diversity analysis | TODO | Medium | Intra-class diversity metrics |
| E7 | Compare with additional competitor | TODO | High | Add SMOTE, basic GAN, or augmentation baseline |
| E8 | Multiple random seeds | PARTIAL | Medium | Run with N>=5 seeds, report variance |

---

### 3.3 Experiments: P2 (Important)

| # | Experiment | Status | Effort | Description |
|---|------------|--------|--------|-------------|
| E9 | LPIPS metric | TODO | Low | Add perceptual quality metric |
| E10 | KID (unbiased FID alternative) | TODO | Low | More reliable for small samples |
| E11 | Beta schedule comparison | TODO | Medium | Linear vs cosine vs squared_cosine |
| E12 | Inference time benchmarking | TODO | Low | Report samples/second |

---

### 3.4 Paper Writing: P0 (Blocking)

| # | Section | Status | Description |
|---|---------|--------|-------------|
| W1 | Complete Methods section | TODO | Full mathematical formulation of temperature weighting |
| W2 | Hyperparameter table | TODO | All training/generation hyperparameters |
| W3 | Statistical testing details | TODO | Seeds, folds, test type justification |
| W4 | Limitations section | TODO | Acknowledge single dataset, conditioning approach |

---

### 3.5 Paper Writing: P1 (Critical)

| # | Section | Status | Description |
|---|---------|--------|-------------|
| W5 | Related work expansion | TODO | Position against data augmentation literature |
| W6 | Ablation study section | TODO | Present all ablations systematically |
| W7 | Theoretical motivation | TODO | Why temperature weighting works |
| W8 | Future work | TODO | Multi-dataset, larger images, other modalities |

---

### 3.6 Code/Reproducibility: P1 (Critical)

| # | Task | Status | Description |
|---|------|--------|-------------|
| C1 | Fix PSNR clamping bug | TODO | Change to [-1, 1] range |
| C2 | Configuration file for paper experiments | TODO | Single YAML reproducing all results |
| C3 | Seed specification | TODO | Document all seeds used |
| C4 | Supplementary code package | TODO | Clean, documented code for review |

---

## Part 4: Implementation Roadmap

### Phase 1: Critical Experiments (Week 1-2)

```
Day 1-3:   E1 - Guidance scale ablation (6 experiments)
Day 4-6:   E3 - Min-SNR + class weight grid (4 experiments)
Day 7-10:  E2 - Second dataset (DermaMNIST) full pipeline
Day 11-14: E4 - Complete temperature ablation analysis
```

### Phase 2: Supporting Experiments (Week 3)

```
Day 15-17: E5 - Class embedding dimension ablation (4 experiments)
Day 18-19: E6 - Mode collapse analysis
Day 20-21: E7 - Additional competitor comparison
```

### Phase 3: Polish and Metrics (Week 4)

```
Day 22-23: E8 - Multi-seed runs for statistical robustness
Day 24-25: E9, E10 - LPIPS and KID metrics
Day 26-28: Bug fixes (C1-C4)
```

### Phase 4: Paper Writing (Week 5-6)

```
Week 5: Methods, Experiments, Results sections
Week 6: Introduction, Related Work, Discussion, Abstract
```

---

## Part 5: Risk Assessment

### High Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Second dataset shows poor results | Paper rejection | Test on DermaMNIST first (smaller, easier) |
| Temperature has no optimal range | Core contribution weakened | Reframe as dataset-dependent hyperparameter |
| Guidance scale ablation shows instability | Reproducibility concerns | Report mean+-std across seeds |

### Medium Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mode collapse in minority classes | Reviewer concern | Early detection via diversity metrics |
| FID computation inconsistencies | Results questioned | Standardize methodology, report details |
| Training time on second dataset | Schedule slip | Use smaller model or fewer epochs |

---

## Part 6: Checklist for Submission

### Paper Components
- [ ] Abstract (250 words max for IEEE)
- [ ] Introduction with clear contribution statement
- [ ] Related work (data augmentation, diffusion models, medical imaging)
- [ ] Methods with mathematical formulation
- [ ] Experiments with ablation studies
- [ ] Results with statistical significance
- [ ] Discussion with limitations
- [ ] Conclusion
- [ ] References (IEEE format)

### Supplementary Material
- [ ] Extended ablation results
- [ ] Per-class detailed results
- [ ] Configuration files
- [ ] Code availability statement

### IEEE IJCNN Specific
- [ ] Page limit compliance (8 pages + references)
- [ ] IEEE template formatting
- [ ] No author information in submission (double-blind)
- [ ] Keywords selection

---

## Appendix A: Recommended Configuration for Paper Experiments

```yaml
# Recommended configuration for IJCNN 2026 paper experiments
ccddpm:
  train:
    image_size: 64
    batch_size: 64
    epochs: 100
    class_embed_dim: 16  # Ablate: [8, 16, 32, 64]
    num_classes: 9
    guidance_p_uncond: 0.1
    use_min_snr: true  # Ablate: [true, false]
    min_snr_gamma: 5.0
    per_class_loss_weighting: true  # Ablate: [true, false]
    per_class_weight_temperature: 1.5  # Ablate: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

  optimizer:
    type: adamw
    lr: 2e-4

  sched:
    num_train_timesteps: 1000
    beta_schedule: squaredcos_cap_v2

  infer:
    guidance_scale: 2.0  # Ablate: [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    sampler: ddim
    num_inference_steps: 100
```

---

## Appendix B: Competitor Methods to Include

| Method | Type | Citation | Available Code |
|--------|------|----------|----------------|
| DistDiff | Diffusion | You et al., 2024 | Yes (already compared) |
| SMOTE | Oversampling | Chawla et al., 2002 | scikit-learn |
| RandAugment | Traditional Aug | Cubuk et al., 2020 | torchvision |
| AutoAugment | Learned Aug | Cubuk et al., 2019 | timm |
| StyleGAN-ADA | GAN | Karras et al., 2020 | NVIDIA |

---

## Appendix C: Metrics to Report

### Primary Metrics
- FID (global and per-class) with std
- Downstream F1 (macro and per-class) with 95% CI
- Cohen's d effect size

### Secondary Metrics
- KID (unbiased alternative to FID)
- LPIPS (perceptual quality)
- Intra-class diversity (average pairwise LPIPS)
- Training time (GPU-hours)
- Inference throughput (samples/second)

---

*Document generated: January 2026*
*Author: Deep Learning Scientist Review*
