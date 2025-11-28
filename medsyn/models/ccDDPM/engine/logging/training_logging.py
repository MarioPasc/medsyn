# medsyn/models/ccDDPM/training_logging.py
# Purpose: CSV epoch-level logger for ccDDPM training metrics.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
import math
import time
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("medsyn.ccddpm.train")

# Number of classes in PathMNIST (used for generating per-class column names)
NUM_CLASSES = 9

# ============================================================================
# TRAINING METRICS (split="train", "val", "test")
# ============================================================================
# These fields are logged to training_metrics.csv for standard train/val/test splits.
TRAINING_FIELDS: List[str] = [
    # Basic training info
    "epoch", "split", "time_s", "lr",

    # Core loss
    "loss", "loss_std",

    # Reconstruction quality metrics (computed on denoised samples)
    "psnr", "ssim",

    # Noise prediction quality
    "noise_mse", "noise_mae",

    # Per-class statistics
    "loss_per_class",  # JSON string or class-wise breakdown

    # Gradient & model health
    "grad_norm",

    # EMA tracking
    "ema_enabled",

    # Total samples processed
    "total_count",

    # ========================================================================
    # PER-CLASS RECONSTRUCTION METRICS
    # ========================================================================
    # Per-class PSNR (dB) - higher is better
    # These help identify which classes have better/worse reconstruction quality.
] + [f"psnr_c{k}" for k in range(NUM_CLASSES)] + [
    # Per-class SSIM (0-1) - higher is better
] + [f"ssim_c{k}" for k in range(NUM_CLASSES)] + [
    # Per-class raw loss (unweighted ε-MSE)
] + [f"loss_raw_c{k}" for k in range(NUM_CLASSES)] + [
    # Per-class weighted loss (with Min-SNR and class weighting)
] + [f"loss_weighted_c{k}" for k in range(NUM_CLASSES)] + [
    # ========================================================================
    # FID METRICS (computed on validation/test sets)
    # ========================================================================
    # Global FID (across all classes)
    "fid_global",
    # Per-class FID
] + [f"fid_c{k}" for k in range(NUM_CLASSES)] + [
    # ========================================================================
    # CLASS WEIGHT CORRELATION DIAGNOSTICS
    # ========================================================================
    # Correlation between class weights and per-class PSNR improvements
    # Positive correlation suggests class weighting is helping underrepresented classes
    "weight_psnr_corr",
    # Correlation between class weights and per-class SSIM
    "weight_ssim_corr",

    # ========================================================================
    # EARLY STOPPING DIAGNOSTICS
    # ========================================================================
    # These fields track the early stopping state at each epoch.
    # They help understand when and why training was stopped.
    #
    # Composite validation score: S = (psnr_weight * PSNR/20) + (ssim_weight * SSIM)
    # The PSNR is divided by 20 to put it on a similar scale to SSIM (~0-1).
    #
    # EMA smoothing: S_ema_t = ema_alpha * S_ema_{t-1} + (1 - ema_alpha) * S_t
    # Higher ema_alpha means more smoothing (less sensitive to fluctuations).

    # Raw composite validation score (before EMA smoothing)
    "val_score_raw",
    # EMA-smoothed composite validation score
    "val_score_ema",
    # Best validation score seen so far (used for model saving)
    "best_val_score",
    # Epoch that achieved the best validation score
    "best_val_epoch",
    # Number of epochs without improvement
    "epochs_without_improvement",
    # Early stopping flag: 1.0 if early stopping triggered, 0.0 otherwise
    "early_stop_flag",
    # Whether this epoch's model was saved as the new best
    "is_best_epoch",
]

# ============================================================================
# DIAGNOSTIC METRICS (split="diag")
# ============================================================================
# These fields are logged to diagnostics_metrics.csv for detailed training health analysis.
# Kept separate from training_metrics.csv for cleaner debugging.
DIAGNOSTIC_FIELDS: List[str] = [
    # Basic info (repeated for standalone CSV)
    "epoch", "split", "time_s", "lr",

    # Noise-prediction correlation: measures how well the model predicts noise
    # A high correlation (>0.5) indicates the model is learning the noise pattern.
    # A value close to 0 or negative may indicate training issues.
    "noise_pred_corr",          # Correlation between predicted and true noise (ideally high >0.5)
    "noise_pred_corr_t100",     # Correlation at timestep t=100 (high noise)
    "noise_pred_corr_t500",     # Correlation at timestep t=500 (medium noise)

    # Prediction statistics
    "pred_std",                 # Std of model predictions (should be ~0.8-1.2 for normalized data)
    "pred_std_t100",            # Std at t=100
    "pred_std_t500",            # Std at t=500

    # Single-step reconstruction metrics (x0 estimated from single denoising step)
    "recon_mse_t100",           # MSE of x0 reconstruction at t=100
    "recon_mse_t500",           # MSE of x0 reconstruction at t=500
    "recon_psnr_t100",          # PSNR of x0 reconstruction at t=100
    "recon_psnr_t500",          # PSNR of x0 reconstruction at t=500
    "recon_ssim_t100",          # SSIM of x0 reconstruction at t=100
    "recon_ssim_t500",          # SSIM of x0 reconstruction at t=500

    # Full-chain reconstruction (complete diffusion + denoising)
    "full_chain_psnr",          # PSNR after full denoising chain
    "full_chain_ssim",          # SSIM after full denoising chain

    # ========================================================================
    # ELBO / MIN-SNR DIAGNOSTICS
    # ========================================================================
    # These fields track the Min-SNR weighted loss vs the unweighted loss.
    # Used to verify that Min-SNR weighting is working correctly and to
    # analyze which timestep regions dominate the training loss.
    #
    # L_simple:  Unweighted ε-MSE (standard training loss before weighting)
    # L_weighted: ε-MSE * Min-SNR weight (matches actual training loss when enabled)
    # SNR: Signal-to-noise ratio at each timestep, SNR(t) = ᾱ_t / (1 - ᾱ_t)
    #
    # When use_min_snr=False:
    #   - L_weighted == L_simple
    #   - weight_ratio ≈ 1.0 across all timestep bins
    #   - min_snr_weight == 1.0
    #
    # When use_min_snr=True:
    #   - L_weighted <= L_simple (down-weighting high SNR / low noise timesteps)
    #   - weight_ratio < 1.0 at low timesteps (high SNR, low noise)
    #   - weight_ratio ≈ 1.0 at high timesteps (low SNR, high noise)
    #   - min_snr_weight in (0, 1]
    #
    # Overall means (averaged across random timesteps in diagnostic batch):
    "elbo_L_simple_mean",           # Mean L_simple across batch (unweighted MSE)
    "elbo_L_weighted_mean",         # Mean L_weighted across batch (Min-SNR weighted MSE)
    "elbo_snr_mean",                # Mean SNR across batch
    "elbo_min_snr_weight_mean",     # Mean Min-SNR weight (1.0 when disabled)

    # Timestep-binned metrics (to see which t regions dominate the loss):
    # Low timesteps (t < 333): low noise, high SNR, fine details matter
    "elbo_L_simple_low_t",          # Mean L_simple for t < 333
    "elbo_L_weighted_low_t",        # Mean L_weighted for t < 333
    "elbo_snr_low_t",               # Mean SNR for t < 333
    "elbo_min_snr_weight_low_t",    # Mean Min-SNR weight for t < 333 (< 1.0 when enabled)

    # Mid timesteps (333 <= t < 666): medium noise, balanced
    "elbo_L_simple_mid_t",          # Mean L_simple for 333 <= t < 666
    "elbo_L_weighted_mid_t",        # Mean L_weighted for 333 <= t < 666
    "elbo_snr_mid_t",               # Mean SNR for 333 <= t < 666
    "elbo_min_snr_weight_mid_t",    # Mean Min-SNR weight for 333 <= t < 666

    # High timesteps (t >= 666): high noise, low SNR, coarse structure
    "elbo_L_simple_high_t",         # Mean L_simple for t >= 666
    "elbo_L_weighted_high_t",       # Mean L_weighted for t >= 666
    "elbo_snr_high_t",              # Mean SNR for t >= 666
    "elbo_min_snr_weight_high_t",   # Mean Min-SNR weight for t >= 666 (~1.0 when enabled)

    # Weight ratios: L_weighted / L_simple shows Min-SNR weighting effect
    # These should be ~1.0 when use_min_snr=False (sanity check)
    "elbo_weight_ratio_low_t",      # L_weighted/L_simple for low t
    "elbo_weight_ratio_mid_t",      # L_weighted/L_simple for mid t
    "elbo_weight_ratio_high_t",     # L_weighted/L_simple for high t

    # Legacy field names (for backwards compatibility)
    "input_output_correlation", # Deprecated: use noise_pred_corr instead
    "reconstruction_mse_t100",  # Deprecated: use recon_mse_t100 instead
    "reconstruction_mse_t500",  # Deprecated: use recon_mse_t500 instead
    "reconstruction_psnr_t500", # Deprecated: use recon_psnr_t500 instead
    "prediction_std",           # Deprecated: use pred_std instead
]

# Combined fields for backwards compatibility (deprecated - use separate loggers)
DEFAULT_FIELDS: List[str] = TRAINING_FIELDS + [
    f for f in DIAGNOSTIC_FIELDS if f not in TRAINING_FIELDS
]


# ============================================================================
# FID CONFIGURATION
# ============================================================================
# Default FID computation settings (can be overridden in config)
DEFAULT_FID_CONFIG = {
    "samples_per_class": 100,  # Number of samples to generate per class
    "batch_size": 32,          # Batch size for generation/feature extraction
    "min_samples": 50,         # Minimum samples needed per class for FID
}

@dataclass
class CSVTrainingLogger:
    """
    CSV epoch-level logger for ccDDPM with optional extra fields (e.g., per-class metrics).
    Similar to bVAE logger but adapted for DDPM training metrics.
    """
    csv_path: str
    fieldnames: Optional[List[str]] = None
    extra_fields: Optional[List[str]] = None

    def __post_init__(self) -> None:
        fields = list(DEFAULT_FIELDS)
        if self.extra_fields:
            fields.extend(self.extra_fields)
        self.fieldnames = fields if self.fieldnames is None else self.fieldnames
        p = Path(self.csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with p.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=self.fieldnames)
                w.writeheader()

    def log_epoch(self, epoch: int, split: str, lr: float, metrics: Dict[str, float]) -> None:
        """Log metrics for a single epoch. Updates CSV immediately."""
        assert self.fieldnames is not None, "fieldnames must be initialized"
        
        row = {k: "" for k in self.fieldnames}
        row.update({
            "epoch": str(epoch),
            "split": split,
            "lr": str(lr),
            "time_s": str(time.time())
        })
        for k, v in metrics.items():
            if k in row:
                row[k] = str(v)  # Convert to string
        
        with Path(self.csv_path).open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames)
            w.writerow(row)


class EpochAverager:
    """
    Accumulate metrics across batches and compute weighted averages and standard deviations.

    Handles both global metrics (loss, psnr, ssim) and per-class metrics
    (psnr_c0, ssim_c0, loss_raw_c0, etc.) dynamically.
    """
    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.sums: Dict[str, float] = {}
        self.sums_sq: Dict[str, float] = {}  # For computing std
        self.counts: Dict[str, int] = {}
        self.total_samples = 0
        # Per-class sample counts (for correct averaging)
        self.per_class_counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Update running sums with batch metrics. Skips non-finite values."""
        self.total_samples += batch_size
        for k, v in metrics.items():
            # Skip inf/NaN values to prevent poisoning epoch averages
            v_float = float(v)
            if not math.isfinite(v_float):
                continue
            if k not in self.sums:
                self.sums[k] = 0.0
                self.sums_sq[k] = 0.0
                self.counts[k] = 0
            self.sums[k] += v_float * batch_size
            self.sums_sq[k] += v_float ** 2 * batch_size
            self.counts[k] += batch_size

    def update_per_class(
        self,
        per_class_metrics: Dict[str, float],
        per_class_counts: Dict[int, int]
    ) -> None:
        """
        Update running sums for per-class metrics with class-specific sample counts.

        This method properly weights per-class metrics by the actual number of
        samples in each class rather than total batch size.

        Args:
            per_class_metrics: Dict with keys like 'psnr_c0', 'ssim_c1', etc.
            per_class_counts: Dict mapping class index -> sample count
        """
        for k, v in per_class_metrics.items():
            v_float = float(v)
            if not math.isfinite(v_float):
                continue

            # Extract class index from key (e.g., 'psnr_c0' -> 0)
            class_idx = None
            for c in range(self.num_classes):
                if k.endswith(f"_c{c}"):
                    class_idx = c
                    break

            if class_idx is not None and class_idx in per_class_counts:
                n = per_class_counts[class_idx]
                if n > 0:
                    if k not in self.sums:
                        self.sums[k] = 0.0
                        self.sums_sq[k] = 0.0
                        self.counts[k] = 0
                    self.sums[k] += v_float * n
                    self.sums_sq[k] += v_float ** 2 * n
                    self.counts[k] += n
            else:
                # Non-per-class metric, use standard update
                if k not in self.sums:
                    self.sums[k] = 0.0
                    self.sums_sq[k] = 0.0
                    self.counts[k] = 0
                self.sums[k] += v_float
                self.sums_sq[k] += v_float ** 2
                self.counts[k] += 1

    def means(self) -> Dict[str, float]:
        """Compute weighted means and standard deviations."""
        out: Dict[str, float] = {}
        for k in self.sums:
            n = self.counts.get(k, 0)
            if n > 0:
                mean = self.sums[k] / n
                out[k] = mean

                # Compute standard deviation for 'loss' metric
                if k == "loss" and n > 1:
                    mean_sq = self.sums_sq[k] / n
                    variance = mean_sq - mean ** 2
                    out["loss_std"] = max(0.0, variance) ** 0.5
            else:
                out[k] = float("nan")

        out["total_count"] = float(self.total_samples)

        # Ensure all per-class columns are present (with NaN for missing classes)
        for prefix in ["psnr_c", "ssim_c", "loss_raw_c", "loss_weighted_c", "fid_c"]:
            for c in range(self.num_classes):
                key = f"{prefix}{c}"
                if key not in out:
                    out[key] = float("nan")

        return out

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.sums.clear()
        self.sums_sq.clear()
        self.counts.clear()
        self.per_class_counts.clear()
        self.total_samples = 0



