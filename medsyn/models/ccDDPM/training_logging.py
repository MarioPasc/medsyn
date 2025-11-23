# medsyn/models/ccDDPM/training_logging.py
# Purpose: CSV epoch-level logger for ccDDPM training metrics.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
import math
import time

DEFAULT_FIELDS: List[str] = [
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
    # DIAGNOSTIC METRICS (split="diag")
    # ========================================================================
    # These fields are populated for split="diag" to track training health.
    # For train/val/test splits, these will be empty or NaN.
    #
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
    # ELBO DIAGNOSTICS (split="diag")
    # ========================================================================
    # These fields track the approximate ELBO decomposition from estimate_elbo_terms.
    # Used to analyze whether Min-SNR weighting is aligned with important timesteps.
    #
    # L_simple: Unweighted ε-MSE (standard training loss before weighting)
    # L_t_weighted: KL-like term from ELBO (Ho et al. Eq. 12)
    # SNR: Signal-to-noise ratio at each timestep
    #
    # Overall means (averaged across random timesteps in diagnostic batch):
    "elbo_L_simple_mean",       # Mean L_simple across batch
    "elbo_L_weighted_mean",     # Mean L_t_weighted (KL term) across batch
    "elbo_snr_mean",            # Mean SNR across batch

    # Timestep-binned metrics (to see which t regions dominate the ELBO):
    # Low timesteps (t < 333): low noise, high SNR, fine details matter
    "elbo_L_simple_low_t",      # Mean L_simple for t < 333
    "elbo_L_weighted_low_t",    # Mean L_t_weighted for t < 333
    "elbo_snr_low_t",           # Mean SNR for t < 333

    # Mid timesteps (333 <= t < 666): medium noise, balanced
    "elbo_L_simple_mid_t",      # Mean L_simple for 333 <= t < 666
    "elbo_L_weighted_mid_t",    # Mean L_t_weighted for 333 <= t < 666
    "elbo_snr_mid_t",           # Mean SNR for 333 <= t < 666

    # High timesteps (t >= 666): high noise, low SNR, coarse structure
    "elbo_L_simple_high_t",     # Mean L_simple for t >= 666
    "elbo_L_weighted_high_t",   # Mean L_t_weighted for t >= 666
    "elbo_snr_high_t",          # Mean SNR for t >= 666

    # Min-SNR analysis: ratio of weighted to simple loss shows weighting effect
    "elbo_weight_ratio_low_t",  # L_weighted/L_simple for low t (should be ~1 if balanced)
    "elbo_weight_ratio_mid_t",  # L_weighted/L_simple for mid t
    "elbo_weight_ratio_high_t", # L_weighted/L_simple for high t

    # Legacy field names (for backwards compatibility)
    "input_output_correlation", # Deprecated: use noise_pred_corr instead
    "reconstruction_mse_t100",  # Deprecated: use recon_mse_t100 instead
    "reconstruction_mse_t500",  # Deprecated: use recon_mse_t500 instead
    "reconstruction_psnr_t500", # Deprecated: use recon_psnr_t500 instead
    "prediction_std",           # Deprecated: use pred_std instead
]

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
        """
        Log metrics for a single epoch. Updates CSV immediately.
        
        Args:
            epoch: Current epoch number
            split: 'train' or 'val' or 'test'
            lr: Current learning rate
            metrics: Dictionary of metric name -> value
        """
        row = {k: "" for k in self.fieldnames}
        row.update({
            "epoch": int(epoch),
            "split": split,
            "lr": float(lr),
            "time_s": float(time.time())
        })
        for k, v in metrics.items():
            if k in row:
                row[k] = v
        with Path(self.csv_path).open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames)
            w.writerow(row)


class EpochAverager:
    """
    Accumulate metrics across batches and compute weighted averages and standard deviations.
    """
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.sums_sq: Dict[str, float] = {}  # For computing std
        self.counts: Dict[str, int] = {}
        self.total_samples = 0

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

    def means(self) -> Dict[str, float]:
        """Compute weighted means and standard deviations."""
        out = {}
        for k in self.sums:
            n = self.counts.get(k, 0)
            mean = self.sums[k] / max(n, 1)
            out[k] = mean
            
            # Compute standard deviation for 'loss' metric
            if k == "loss" and n > 1:
                mean_sq = self.sums_sq[k] / n
                variance = mean_sq - mean ** 2
                out["loss_std"] = max(0.0, variance) ** 0.5  # Avoid negative due to numerical errors
        
        out["total_count"] = self.total_samples
        return out

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.sums.clear()
        self.sums_sq.clear()
        self.counts.clear()
        self.total_samples = 0
