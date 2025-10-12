# medsyn/models/bVAE/metrics.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import math
import torch

def psnr(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio for images scaled to [0,1].

    Args:
        x_hat: reconstructed images, float tensor in [0,1], shape (B,C,H,W)
        x:     target images, same shape and scale as x_hat
        max_val: dynamic range maximum; 1.0 for normalized images
    Returns:
        Tensor scalar with batch-wise PSNR (dB), averaged over batch.
    """
    mse = torch.mean((x_hat - x) ** 2, dtype=torch.float32)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)

def latent_stats(mu: torch.Tensor, logv: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Summaries of the approximate posterior q(z|x) for monitoring.

    Returns:
        dict of scalar tensors: mu_abs_mean, logv_mean, logv_std, z_var_mean
    """
    mu_abs_mean = mu.abs().mean()
    logv_mean = logv.mean()
    logv_std = logv.std(unbiased=False)
    z_var_mean = torch.mean(torch.exp(logv))
    return {
        "mu_abs_mean": mu_abs_mean,
        "logv_mean": logv_mean,
        "logv_std": logv_std,
        "z_var_mean": z_var_mean,
    }

@dataclass
class EpochAverager:
    """
    Accumulates per-batch metrics and returns epoch means.

    Methods:
        update(batch_metrics, batch_size): add weighted by batch size
        means(): return dict with averaged metrics and total_count
        reset(): clear accumulators
    """
    sums: Dict[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, batch_metrics: Dict[str, float], batch_size: int) -> None:
        for k, v in batch_metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * batch_size
        self.count += int(batch_size)

    def means(self) -> Dict[str, float]:
        if self.count == 0:
            return {k: 0.0 for k in self.sums} | {"total_count": 0}
        return {k: self.sums[k] / self.count for k in self.sums} | {"total_count": self.count}

    def reset(self) -> None:
        self.sums.clear()
        self.count = 0

def make_batch_metrics_dict(loss_total: torch.Tensor,
                            loss_recon: torch.Tensor,
                            loss_kld: torch.Tensor,
                            x_hat: torch.Tensor,
                            x: torch.Tensor,
                            mu: torch.Tensor,
                            logv: torch.Tensor,
                            latent_dim: int) -> Dict[str, float]:
    """
    Produce a flat metrics dict for logging from tensors.

    Returns:
        dict with keys: loss, recon, kld, kld_per_dim, psnr, mu_abs_mean, logv_mean, logv_std, z_var_mean
    """
    ls = latent_stats(mu, logv)
    return {
        "loss": float(loss_total.detach().cpu()),
        "recon": float(loss_recon.detach().cpu()),
        "kld": float(loss_kld.detach().cpu()),
        "kld_per_dim": float(loss_kld.detach().cpu() / max(1, latent_dim)),
        "psnr": float(psnr(x_hat.detach(), x.detach()).cpu()),
        "mu_abs_mean": float(ls["mu_abs_mean"].detach().cpu()),
        "logv_mean": float(ls["logv_mean"].detach().cpu()),
        "logv_std": float(ls["logv_std"].detach().cpu()),
        "z_var_mean": float(ls["z_var_mean"].detach().cpu()),
    }
