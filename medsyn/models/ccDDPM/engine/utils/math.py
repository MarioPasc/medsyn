
from typing import Dict, Optional, Union
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from medsyn.models.ccDDPM.metrics import compute_psnr, compute_ssim
from medsyn.models.ccDDPM.loss import estimate_elbo_terms

from medsyn.models.ccDDPM.config import InferenceCfg, SchedCfg

logger = logging.getLogger("medsyn.ccddpm.train")


@torch.no_grad()
def compute_training_diagnostics(
    model: nn.Module,
    x0_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    num_samples: int = 16,
) -> Dict[str, float]:
    """
    Compute comprehensive diagnostic metrics to detect training issues.

    These diagnostics are logged to the CSV with split="diag" to enable
    correlation of failure modes or instabilities with epoch and configuration.

    Returns:
        Dictionary with diagnostic metrics:

        Noise-Prediction Correlation (correctly interpreted):
        - noise_pred_corr: Mean correlation between predicted and true noise
        - noise_pred_corr_t100: Correlation at t=100 (high noise level)
        - noise_pred_corr_t500: Correlation at t=500 (medium noise level)
          * High correlation (>0.5) = model is learning noise pattern correctly
          * Low/negative = potential training issues

        Prediction Statistics:
        - pred_std: Mean std of model predictions
        - pred_std_t100: Std at t=100
        - pred_std_t500: Std at t=500
          * Should be ~0.8-1.2 for normalized data
          * Much lower = model collapsed to constant
          * Much higher = unstable predictions

        Single-Step Reconstruction (x0 estimated from single denoising step):
        - recon_mse_t100, recon_mse_t500: MSE of x0 reconstruction
        - recon_psnr_t100, recon_psnr_t500: PSNR of x0 reconstruction
        - recon_ssim_t100, recon_ssim_t500: SSIM of x0 reconstruction

        Legacy fields (for backwards compatibility):
        - input_output_correlation, reconstruction_mse_t100/t500, etc.
    """
    model.eval()

    # Take subset of batch
    x0 = x0_batch[:num_samples]
    labels = labels_batch[:num_samples]

    # Per-timestep metrics storage
    metrics_per_t: Dict[int, Dict[str, float]] = {}

    for t_val in [100, 500]:
        t = torch.full((len(x0),), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Model prediction
        eps_pred = model(x_t, t, labels)

        # ================================================================
        # Noise-Prediction Correlation (correctly interpreted)
        # ================================================================
        # We correlate predicted noise (eps_pred) with true noise (eps_true).
        # A well-trained model should have HIGH positive correlation here,
        # indicating it correctly predicts the noise that was added.
        eps_true = noise
        eps_pred_flat = eps_pred.flatten()
        eps_true_flat = eps_true.flatten()
        eps_pred_std = eps_pred_flat.std()
        eps_true_std = eps_true_flat.std()

        if eps_pred_std > 1e-8 and eps_true_std > 1e-8:
            corr = torch.corrcoef(torch.stack([eps_pred_flat, eps_true_flat]))[0, 1].item()
        else:
            corr = 0.0

        # ================================================================
        # Prediction Statistics
        # ================================================================
        pred_std = eps_pred.std().item()
        pred_mean = eps_pred.mean().item()

        # ================================================================
        # Single-Step x0 Reconstruction
        # ================================================================
        # Use the DDPM posterior mean formula to estimate x0 from x_t and eps_pred
        sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x0_pred = (x_t - sqrt_one_minus_alpha_prod * eps_pred) / sqrt_alpha_prod
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # MSE
        mse = F.mse_loss(x0_pred, x0).item()

        # PSNR
        psnr = compute_psnr(x0_pred, x0)

        # SSIM (convert from [-1, 1] to [0, 1] range for SSIM computation)
        x0_pred_01 = (x0_pred + 1.0) / 2.0
        x0_01 = (x0 + 1.0) / 2.0
        ssim = compute_ssim(x0_pred_01, x0_01, max_val=1.0)

        # Store metrics for this timestep
        metrics_per_t[t_val] = {
            "corr": corr,
            "pred_std": pred_std,
            "pred_mean": pred_mean,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
        }

    model.train()

    # ================================================================
    # Build output dictionary with all diagnostic fields
    # ================================================================
    t100 = metrics_per_t[100]
    t500 = metrics_per_t[500]

    # Average correlation across timesteps
    mean_corr = (t100["corr"] + t500["corr"]) / 2.0
    mean_pred_std = (t100["pred_std"] + t500["pred_std"]) / 2.0

    result = {
        # ---- New diagnostic field names ----
        # Noise-prediction correlation (correctly interpreted: high = good)
        "noise_pred_corr": float(mean_corr),
        "noise_pred_corr_t100": float(t100["corr"]),
        "noise_pred_corr_t500": float(t500["corr"]),

        # Prediction statistics
        "pred_std": float(mean_pred_std),
        "pred_std_t100": float(t100["pred_std"]),
        "pred_std_t500": float(t500["pred_std"]),

        # Single-step reconstruction metrics
        "recon_mse_t100": float(t100["mse"]),
        "recon_mse_t500": float(t500["mse"]),
        "recon_psnr_t100": float(t100["psnr"]),
        "recon_psnr_t500": float(t500["psnr"]),
        "recon_ssim_t100": float(t100["ssim"]),
        "recon_ssim_t500": float(t500["ssim"]),

        # ---- Legacy field names (for backwards compatibility) ----
        "input_output_correlation": float(mean_corr),
        "reconstruction_mse_t100": float(t100["mse"]),
        "reconstruction_mse_t500": float(t500["mse"]),
        "reconstruction_psnr_t500": float(t500["psnr"]),
        "prediction_std": float(mean_pred_std),
    }

    return result

@torch.no_grad()
def full_chain_reconstruction_metrics(
    inference_config: InferenceCfg,
    training_scheduler_config: SchedCfg,
    model: nn.Module,
    ddpm_scheduler: Union[DDPMScheduler, DDIMScheduler],
    x0: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    seed: Optional[int] = 3012022,
) -> Dict[str, float]:
    """
    Compute full-chain reconstruction that gives the option of
    using deterministic DDIM sampling.
    
    DDIM (Song et al., 2020) uses a non-Markovian reverse process that is
    deterministic when η=0, eliminating sampling variance:
    
        x_{t-1} = √(ᾱ_{t-1}) * x̂_0 + √(1-ᾱ_{t-1}) * ε_θ(x_t, t)
    
    where x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √(ᾱ_t)
    
    References:
        Song et al. (2020), "Denoising Diffusion Implicit Models"
    """
    
    # Save current RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    
    # Set fixed seed for deterministic forward/reverse process
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        
    model.eval()

    # Forward: noise to x_T
    noise = torch.randn_like(x0)
    T = training_scheduler_config.num_train_timesteps - 1
    x_t = ddpm_scheduler.add_noise(x0, noise, torch.tensor([T], device=device))
    

    if inference_config.sampler == "ddim":
        # Create DDIM scheduler from DDPM config
        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=training_scheduler_config.num_train_timesteps,
            beta_start=training_scheduler_config.beta_start,
            beta_end=training_scheduler_config.beta_end,
            beta_schedule=training_scheduler_config.beta_schedule,
            prediction_type=training_scheduler_config.prediction_type,
            clip_sample=True,
            set_alpha_to_one=False,
        )
        ddim_scheduler.set_timesteps(inference_config.num_inference_steps, device=device)
        timesteps = ddim_scheduler.timesteps
        scheduler_step = ddim_scheduler
    else:
        # Use DDPM reverse process instead of DDIM     
        # Set timesteps
        ddpm_scheduler.set_timesteps(inference_config.num_inference_steps, device=device)
        timesteps = ddpm_scheduler.timesteps
        scheduler_step = ddpm_scheduler

    # Reverse: x_T to x_0
    for t in timesteps:
        t_batch = t.expand(x0.size(0)).to(device)
        noise_pred = model(x_t, t_batch, y)
        
        if inference_config.sampler == "ddim":
            # DDIM step (eta=0 for deterministic)
            x_t = scheduler_step.step(
                noise_pred, t, x_t, eta=inference_config.eta
            ).prev_sample
        else:
            # DDPM step
            x_t = scheduler_step.step(noise_pred, t, x_t).prev_sample
        
    x0_recon = x_t
    
    # Rescale to [0, 1] for metrics
    x0_01 = (x0.clamp(-1, 1) + 1) / 2
    x0_recon_01 = (x0_recon.clamp(-1, 1) + 1) / 2
    
    # Compute PSNR and SSIM
    from medsyn.models.ccDDPM.metrics import compute_psnr, compute_ssim
    psnr = compute_psnr(x0_recon_01, x0_01, max_val=1.0)
    ssim = compute_ssim(x0_recon_01, x0_01, max_val=1.0)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, device)

    return {"psnr": psnr, "ssim": ssim}


def compute_class_weights_from_counts(
    counts,  # np.ndarray
    temperature: float = 1.0,
    normalize: bool = True,
):
    """
    Compute per-class loss weights from class sample counts with temperature scaling.

    The algorithm:
    1. Compute frequency: freq[c] = count[c] / total_samples
    2. Compute inverse frequency: inv_freq[c] = 1 / freq[c]
    3. Apply temperature scaling: weight[c] = inv_freq[c] ^ temperature
    4. Normalize weights to mean=1.0 (optional, recommended for training stability)

    Temperature effects:
    - temperature = 1.0: Standard inverse frequency weighting
    - temperature > 1.0: More extreme weights (higher boost for minority classes)
      * e.g., temp=2.0 squares the inverse frequency ratio
      * Minority classes get exponentially more weight
    - temperature < 1.0: More uniform weights (less aggressive reweighting)
      * e.g., temp=0.5 takes square root of inverse frequency ratio
      * Approaches uniform weighting as temp → 0

    Example with 3 classes [900, 90, 10] samples:
    - temp=0.5: weights ≈ [0.87, 0.92, 1.57]  (mild reweighting)
    - temp=1.0: weights ≈ [0.33, 1.00, 3.00]  (standard inverse freq)
    - temp=2.0: weights ≈ [0.11, 1.00, 9.00]  (aggressive reweighting)

    Args:
        counts: Array of sample counts per class [num_classes]
        temperature: Temperature scaling factor (>1: more extreme, <1: more uniform)
        normalize: If True, normalize weights to mean=1.0 for gradient stability

    Returns:
        Array of class weights [num_classes]

    Raises:
        ValueError: If counts contain negative values or all zeros
    """
    import numpy as np
    
    counts = np.asarray(counts)
    
    if np.any(counts < 0):
        raise ValueError(f"counts must be non-negative, got: {counts}")

    if np.all(counts == 0):
        raise ValueError("All class counts are zero, cannot compute weights")

    # Compute frequency (with small epsilon to avoid division by zero)
    total = float(counts.sum())
    freq = counts.astype("float64") / total

    # Compute inverse frequency (with epsilon for zero-count classes)
    inv_freq = 1.0 / np.maximum(freq, 1e-8)

    # Apply temperature scaling: weight = inv_freq ^ temperature
    if temperature != 1.0:
        weights = np.power(inv_freq, temperature)
    else:
        weights = inv_freq

    # Normalize to mean=1.0 for training stability
    if normalize:
        weights = weights / weights.mean()

    return weights

@torch.no_grad()
def compute_elbo_diagnostics(
    model: nn.Module,
    x0_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    num_samples: int = 32,
    num_timesteps_per_sample: int = 8,
    use_min_snr: bool = False,
    min_snr_gamma: float = 5.0,
) -> Dict[str, float]:
    """
    Compute ELBO-related diagnostic metrics using estimate_elbo_terms.

    This function samples random timesteps across the batch and computes:
    - L_simple: Unweighted ε-MSE (standard training loss before any weighting)
    - L_weighted: ε-MSE weighted by Min-SNR factor (matches training loss when enabled)
    - SNR: Signal-to-noise ratio at each timestep
    - min_snr_weight: The Min-SNR weight applied (for analysis)

    These metrics help analyze whether Min-SNR weighting is aligned with the
    parts of the loss that matter. By binning results by timestep range, you
    can see which regions dominate the loss and how weighting affects them.

    When use_min_snr=False, L_weighted == L_simple and weight_ratio ≈ 1.0.
    When use_min_snr=True, L_weighted < L_simple at high SNR timesteps.

    Args:
        model: DDPM model
        x0_batch: Clean images [B, C, H, W]
        labels_batch: Class labels [B]
        noise_scheduler: DDPM scheduler with alphas_cumprod and betas
        device: Compute device
        num_samples: Number of samples from batch to use
        num_timesteps_per_sample: Number of random timesteps to sample per image
        use_min_snr: If True, compute L_weighted with Min-SNR weighting
        min_snr_gamma: Clipping parameter γ for Min-SNR (typically 3–5)

    Returns:
        Dictionary with ELBO diagnostic metrics:
        - elbo_L_simple_mean: Mean L_simple across all samples
        - elbo_L_weighted_mean: Mean L_weighted across all samples
        - elbo_snr_mean: Mean SNR across all samples
        - elbo_L_simple_low_t, elbo_L_weighted_low_t, elbo_snr_low_t: Metrics for t < 333
        - elbo_L_simple_mid_t, elbo_L_weighted_mid_t, elbo_snr_mid_t: Metrics for 333 <= t < 666
        - elbo_L_simple_high_t, elbo_L_weighted_high_t, elbo_snr_high_t: Metrics for t >= 666
        - elbo_weight_ratio_*: Ratio of L_weighted/L_simple per timestep bin
        - elbo_min_snr_weight_mean: Mean of Min-SNR weights (1.0 when disabled)
    """
    model.eval()

    # Prepare scheduler config for estimate_elbo_terms
    T = noise_scheduler.config.num_train_timesteps
    scheduler_cfg = {
        "alphas_cumprod": noise_scheduler.alphas_cumprod,
    }

    # Take subset of batch
    x0 = x0_batch[:num_samples].to(device)
    labels = labels_batch[:num_samples].to(device)
    n = x0.size(0)

    # Accumulators for overall and binned statistics
    all_L_simple = []
    all_L_weighted = []
    all_snr = []
    all_min_snr_weight = []
    all_t = []

    # Sample multiple random timesteps per image to get coverage across t
    for _ in range(num_timesteps_per_sample):
        # Random timesteps uniformly from [0, T-1]
        t = torch.randint(0, T, (n,), device=device, dtype=torch.long)

        # Add noise
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Model prediction
        eps_pred = model(x_t, t, labels)

        # Compute ELBO terms with Min-SNR settings matching training
        elbo_result = estimate_elbo_terms(
            x0=x0,
            x_t=x_t,
            t=t,
            eps_pred=eps_pred,
            scheduler_cfg=scheduler_cfg,
            use_min_snr=use_min_snr,
            min_snr_gamma=min_snr_gamma,
        )

        # Accumulate results
        all_L_simple.append(elbo_result["L_simple"])
        all_L_weighted.append(elbo_result["L_weighted"])
        all_snr.append(elbo_result["snr"])
        all_min_snr_weight.append(elbo_result["min_snr_weight"])
        all_t.append(elbo_result["t"])

    # Concatenate all results
    all_L_simple = torch.cat(all_L_simple, dim=0)
    all_L_weighted = torch.cat(all_L_weighted, dim=0)
    all_snr = torch.cat(all_snr, dim=0)
    all_min_snr_weight = torch.cat(all_min_snr_weight, dim=0)
    all_t = torch.cat(all_t, dim=0)

    # Overall means
    elbo_L_simple_mean = all_L_simple.mean().item()
    elbo_L_weighted_mean = all_L_weighted.mean().item()
    elbo_snr_mean = all_snr.mean().item()
    elbo_min_snr_weight_mean = all_min_snr_weight.mean().item()

    # Bin by timestep: low (t < T/3), mid (T/3 <= t < 2T/3), high (t >= 2T/3)
    t_low_threshold = T // 3      # 333 for T=1000
    t_high_threshold = 2 * T // 3  # 666 for T=1000

    mask_low = all_t < t_low_threshold
    mask_mid = (all_t >= t_low_threshold) & (all_t < t_high_threshold)
    mask_high = all_t >= t_high_threshold

    def safe_mean(tensor, mask):
        """Compute mean only over masked elements, return NaN if no elements."""
        if mask.sum() == 0:
            return float("nan")
        return tensor[mask].mean().item()

    def safe_ratio(num, denom):
        """Compute ratio, avoiding division by zero."""
        if abs(denom) < 1e-10:
            return float("nan")
        return num / denom

    # Compute binned statistics
    L_simple_low = safe_mean(all_L_simple, mask_low)
    L_simple_mid = safe_mean(all_L_simple, mask_mid)
    L_simple_high = safe_mean(all_L_simple, mask_high)

    L_weighted_low = safe_mean(all_L_weighted, mask_low)
    L_weighted_mid = safe_mean(all_L_weighted, mask_mid)
    L_weighted_high = safe_mean(all_L_weighted, mask_high)

    snr_low = safe_mean(all_snr, mask_low)
    snr_mid = safe_mean(all_snr, mask_mid)
    snr_high = safe_mean(all_snr, mask_high)

    # Min-SNR weights per region (should be ~1.0 when disabled)
    min_snr_weight_low = safe_mean(all_min_snr_weight, mask_low)
    min_snr_weight_mid = safe_mean(all_min_snr_weight, mask_mid)
    min_snr_weight_high = safe_mean(all_min_snr_weight, mask_high)

    # Weight ratios show how much the Min-SNR weighting affects each region
    # When use_min_snr=False, these should all be ~1.0
    weight_ratio_low = safe_ratio(L_weighted_low, L_simple_low)
    weight_ratio_mid = safe_ratio(L_weighted_mid, L_simple_mid)
    weight_ratio_high = safe_ratio(L_weighted_high, L_simple_high)

    model.train()

    return {
        # Overall means
        "elbo_L_simple_mean": float(elbo_L_simple_mean),
        "elbo_L_weighted_mean": float(elbo_L_weighted_mean),
        "elbo_snr_mean": float(elbo_snr_mean),
        "elbo_min_snr_weight_mean": float(elbo_min_snr_weight_mean),

        # Low timesteps (t < T/3): low noise, high SNR
        "elbo_L_simple_low_t": float(L_simple_low),
        "elbo_L_weighted_low_t": float(L_weighted_low),
        "elbo_snr_low_t": float(snr_low),
        "elbo_min_snr_weight_low_t": float(min_snr_weight_low),

        # Mid timesteps (T/3 <= t < 2T/3)
        "elbo_L_simple_mid_t": float(L_simple_mid),
        "elbo_L_weighted_mid_t": float(L_weighted_mid),
        "elbo_snr_mid_t": float(snr_mid),
        "elbo_min_snr_weight_mid_t": float(min_snr_weight_mid),

        # High timesteps (t >= 2T/3): high noise, low SNR
        "elbo_L_simple_high_t": float(L_simple_high),
        "elbo_L_weighted_high_t": float(L_weighted_high),
        "elbo_snr_high_t": float(snr_high),
        "elbo_min_snr_weight_high_t": float(min_snr_weight_high),

        # Weight ratios (L_weighted / L_simple)
        "elbo_weight_ratio_low_t": float(weight_ratio_low),
        "elbo_weight_ratio_mid_t": float(weight_ratio_mid),
        "elbo_weight_ratio_high_t": float(weight_ratio_high),
    }

@torch.no_grad()
def conditioning_sanity_check(
    model: nn.Module,
    scheduler: DDPMScheduler,
    num_classes: int,
    image_shape: tuple,
    device: torch.device,
    num_samples: int = 3
) -> Dict[str, float]:
    """
    Verify that class conditioning is working by checking that different labels
    produce different noise predictions for the same input.

    Args:
        model: DDPM model
        scheduler: DDPM scheduler
        num_classes: Number of classes
        image_shape: (C, H, W)
        device: Device
        num_samples: Number of random samples to test

    Returns:
        Dictionary with conditioning gap statistics
    """
    model.eval()
    gaps = []

    for _ in range(num_samples):
        # Sample fixed noise and random timestep
        x_t = torch.randn((1, *image_shape), device=device)
        t = torch.randint(100, scheduler.config.num_train_timesteps // 2, (1,), device=device)

        # Get predictions for two different classes
        class_0 = torch.tensor([0], device=device)
        class_1 = torch.tensor([min(1, num_classes - 1)], device=device)

        eps_0 = model(x_t, t, class_0)
        eps_1 = model(x_t, t, class_1)

        # Compute L2 distance between predictions
        gap = torch.norm(eps_0 - eps_1, p=2).item()
        gaps.append(gap)

    return {
        "conditioning_gap_mean": float(torch.tensor(gaps).mean()),
        "conditioning_gap_std": float(torch.tensor(gaps).std()),
        "conditioning_gap_min": float(torch.tensor(gaps).min()),
        "conditioning_gap_max": float(torch.tensor(gaps).max()),
    }

def compute_batch_metrics(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    x0_reconstructed: torch.Tensor,
    x0_original: torch.Tensor,
    loss: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for a single batch during training/validation.
    
    Args:
        pred_noise: Predicted noise from model
        true_noise: Ground truth noise
        x0_reconstructed: Reconstructed x0 from predicted noise
        x0_original: Original clean images
        loss: Total loss value
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "loss": float(loss.detach().cpu()),
        "noise_mse": float(F.mse_loss(pred_noise, true_noise).detach().cpu()),
        "noise_mae": float(F.l1_loss(pred_noise, true_noise).detach().cpu()),
    }
    
    # Compute PSNR and SSIM on reconstructed x0
    # Assuming images are in [-1, 1] range, convert to [0, 1] for metrics
    x0_recon_01 = (x0_reconstructed.detach() + 1.0) / 2.0
    x0_orig_01 = (x0_original.detach() + 1.0) / 2.0
    x0_recon_01 = torch.clamp(x0_recon_01, 0.0, 1.0)
    x0_orig_01 = torch.clamp(x0_orig_01, 0.0, 1.0)
    
    metrics["psnr"] = compute_psnr(x0_recon_01, x0_orig_01, max_val=1.0)
    metrics["ssim"] = compute_ssim(x0_recon_01, x0_orig_01, max_val=1.0)

    # Guard against non-finite PSNR/SSIM values (e.g., inf from zero MSE)
    for key in ["psnr", "ssim"]:
        if key in metrics and not math.isfinite(float(metrics[key])):
            metrics[key] = 0.0

    return metrics
