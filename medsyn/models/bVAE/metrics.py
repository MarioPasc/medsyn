from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn.functional as F

# --------------------------- Scalar metrics -----------------------------------

def psnr_batch(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((x_hat - x) ** 2, dtype=torch.float32)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)

def ssim_batch(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute SSIM if pytorch_msssim is available, else return 0."""
    try:
        from pytorch_msssim import ssim
        s = ssim(x_hat, x, data_range=1.0, size_average=True)
        return torch.clamp(s, min=0.0, max=1.0)
    except Exception:
        return torch.tensor(0.0, device=x_hat.device)

def output_health_stats(x_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Statistics about decoder output distribution and saturation."""
    # Saturated pixels: those very close to 0 or 1
    saturated = ((x_hat < 0.01) | (x_hat > 0.99)).float().mean()
    return {
        "output_mean": x_hat.mean(),
        "output_std": x_hat.std(unbiased=False),
        "output_saturated_ratio": saturated,
    }

def reconstruction_quality_stats(x_hat: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Multiple reconstruction quality metrics."""
    mse = F.mse_loss(x_hat, x, reduction="mean")
    mae = F.l1_loss(x_hat, x, reduction="mean")
    return {
        "mse": mse,
        "mae": mae,
        "ssim": ssim_batch(x_hat, x),
    }

def kld_standard_normal(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
    return F.binary_cross_entropy(x_hat, x, reduction="mean")

def prior_drift_stats(mu_p: torch.Tensor, logv_p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Statistics about learned class prior drift from N(0,I)."""
    # KLD between learned prior N(mu_p, exp(logv_p)) and standard N(0,I)
    # KLD = 0.5 * (mu_p^2 + exp(logv_p) - 1 - logv_p)
    var_p = torch.exp(torch.clamp(logv_p, min=-20.0, max=20.0))
    kld_per_class = 0.5 * (mu_p.pow(2).sum(dim=1) + var_p.sum(dim=1) - mu_p.size(1) - logv_p.sum(dim=1))
    return {
        "prior_mu_max": mu_p.abs().max(),
        "prior_mu_std": mu_p.std(unbiased=False),
        "prior_logv_min": logv_p.min(),
        "prior_logv_max": logv_p.max(),
        "prior_kld_avg": kld_per_class.mean(),
    }

def latent_stats(mu: torch.Tensor, logv: torch.Tensor) -> Dict[str, torch.Tensor]:
    # clamp exp for numerical safety
    logv_clamped = torch.clamp(logv, min=-20.0, max=20.0)
    z_var = torch.clamp(torch.exp(logv_clamped), min=1e-6, max=1e6)

    # Compute per-dimension KLD to detect posterior collapse
    # KLD per dim = 0.5 * (mu^2 + var - 1 - logv)
    kld_per_dim = 0.5 * (mu.pow(2) + z_var - 1.0 - logv_clamped)
    kld_per_dim_mean = torch.mean(kld_per_dim, dim=0)  # (D,)

    # Active units: dimensions with KLD > threshold (0.1 nats)
    active_units = (kld_per_dim_mean > 0.1).sum()

    return {
        "mu_abs_mean": mu.abs().mean(),
        "mu_max": mu.abs().max(),
        "mu_std": mu.std(unbiased=False),
        "logv_mean": logv_clamped.mean(),
        "logv_std": logv.std(unbiased=False),
        "logv_min": logv.min(),
        "logv_max": logv.max(),
        "z_var_mean": torch.mean(z_var),
        "active_units": active_units.float(),
        "kld_max": kld_per_dim_mean.max(),
        "kld_min": kld_per_dim_mean.min(),
    }

def make_batch_metrics_dict(loss_total: torch.Tensor,
                            loss_recon: torch.Tensor,
                            loss_kld: torch.Tensor,
                            x_hat: torch.Tensor,
                            x: torch.Tensor,
                            mu: torch.Tensor,
                            logv: torch.Tensor,
                            latent_dim: int,
                            mu_p: Optional[torch.Tensor] = None,
                            logv_p: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Enhanced batch metrics with collapse detection and health indicators.

    Args:
        loss_total, loss_recon, loss_kld: loss components
        x_hat, x: reconstruction and original
        mu, logv: posterior parameters
        latent_dim: dimensionality of latent space
        mu_p, logv_p: class-conditional prior parameters (from model.prior_params)
    """
    ls = latent_stats(mu, logv)
    rq = reconstruction_quality_stats(x_hat.detach(), x.detach())
    oh = output_health_stats(x_hat.detach())

    metrics = {
        # Core losses
        "loss": float(loss_total.detach().cpu()),
        "recon": float(loss_recon.detach().cpu()),
        "kld": float(loss_kld.detach().cpu()),
        "kld_per_dim": float(loss_kld.detach().cpu() / max(1, latent_dim)),

        # Reconstruction quality
        "psnr": float(psnr_batch(x_hat.detach(), x.detach()).cpu()),
        "ssim": float(rq["ssim"].detach().cpu()),
        "mse": float(rq["mse"].detach().cpu()),
        "mae": float(rq["mae"].detach().cpu()),

        # Decoder output health
        "output_mean": float(oh["output_mean"].detach().cpu()),
        "output_std": float(oh["output_std"].detach().cpu()),
        "output_saturated_ratio": float(oh["output_saturated_ratio"].detach().cpu()),

        # Posterior statistics
        "mu_abs_mean": float(ls["mu_abs_mean"].detach().cpu()),
        "mu_max": float(ls["mu_max"].detach().cpu()),
        "mu_std": float(ls["mu_std"].detach().cpu()),
        "logv_mean": float(ls["logv_mean"].detach().cpu()),
        "logv_std": float(ls["logv_std"].detach().cpu()),
        "logv_min": float(ls["logv_min"].detach().cpu()),
        "logv_max": float(ls["logv_max"].detach().cpu()),
        "z_var_mean": float(ls["z_var_mean"].detach().cpu()),

        # Posterior collapse detection
        "active_units": float(ls["active_units"].detach().cpu()),
        "kld_max": float(ls["kld_max"].detach().cpu()),
        "kld_min": float(ls["kld_min"].detach().cpu()),
    }

    # Prior drift stats (if class-conditional prior is used)
    if mu_p is not None and logv_p is not None:
        pd = prior_drift_stats(mu_p.detach(), logv_p.detach())
        metrics.update({
            "prior_mu_max": float(pd["prior_mu_max"].detach().cpu()),
            "prior_mu_std": float(pd["prior_mu_std"].detach().cpu()),
            "prior_logv_min": float(pd["prior_logv_min"].detach().cpu()),
            "prior_logv_max": float(pd["prior_logv_max"].detach().cpu()),
            "prior_kld_avg": float(pd["prior_kld_avg"].detach().cpu()),
        })
    else:
        metrics.update({
            "prior_mu_max": 0.0,
            "prior_mu_std": 0.0,
            "prior_logv_min": 0.0,
            "prior_logv_max": 0.0,
            "prior_kld_avg": 0.0,
        })

    return metrics

# --------------------------- Class-wise metrics --------------------------------

def _psnr_per_sample(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((x_hat - x) ** 2, dim=(1,2,3))
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)

def _kld_per_sample(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
    # mean over latent dims per sample
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp(), dim=1)

def _recon_mse_per_sample(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean((x_hat - x) ** 2, dim=(1,2,3))

def make_classwise_metrics_dict(x_hat: torch.Tensor, x: torch.Tensor,
                                mu: torch.Tensor, logv: torch.Tensor,
                                y: torch.Tensor, num_classes: int,
                                mu_p: Optional[torch.Tensor] = None,
                                logv_p: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Returns per-class SUMS and COUNTS so that an averager can compute means safely.
    Keys: psnr_c{k}, recon_c{k}, kld_c{k}, count_c{k}, kld_prior_c{k}
    """
    ps = _psnr_per_sample(x_hat, x)
    rs = _recon_mse_per_sample(x_hat, x)
    ks = _kld_per_sample(mu, logv)

    # KLD between posterior and class prior (if provided)
    kld_prior = None
    if mu_p is not None and logv_p is not None:
        # KLD(q(z|x,y) || p(z|y)) per sample
        logv_q_clamped = torch.clamp(logv, min=-20.0, max=20.0)
        logv_p_clamped = torch.clamp(logv_p, min=-20.0, max=20.0)
        var_q = torch.exp(logv_q_clamped)
        var_p = torch.exp(logv_p_clamped)
        kld_prior = 0.5 * torch.mean(
            logv_p_clamped - logv_q_clamped + (var_q + (mu - mu_p) ** 2) / (var_p + 1e-8) - 1.0,
            dim=1
        )

    out: Dict[str, float] = {}
    for k in range(num_classes):
        mask = (y == k)
        c = int(mask.sum().item())
        if c == 0:
            out[f"psnr_c{k}"] = 0.0
            out[f"recon_c{k}"] = 0.0
            out[f"kld_c{k}"] = 0.0
            out[f"kld_prior_c{k}"] = 0.0
            out[f"count_c{k}"] = 0.0
        else:
            out[f"psnr_c{k}"] = float(ps[mask].sum().item())
            out[f"recon_c{k}"] = float(rs[mask].sum().item())
            out[f"kld_c{k}"] = float(ks[mask].sum().item())
            if kld_prior is not None:
                out[f"kld_prior_c{k}"] = float(kld_prior[mask].sum().item())
            else:
                out[f"kld_prior_c{k}"] = 0.0
            out[f"count_c{k}"] = float(c)
    return out

# --------------------------- Averagers ----------------------------------------

@dataclass
class EpochAverager:
    sums: Dict[str, float] = field(default_factory=dict)
    count: int = 0
    batch_losses: list = field(default_factory=list)  # Track all batch losses for min/max
    batch_recons: list = field(default_factory=list)
    grad_norms: list = field(default_factory=list)

    def update(self, batch_metrics: Dict[str, float], batch_size: int) -> None:
        for k, v in batch_metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v) * batch_size
        self.count += int(batch_size)

        # Track batch-level statistics for outlier detection
        if "loss" in batch_metrics:
            self.batch_losses.append(float(batch_metrics["loss"]))
        if "recon" in batch_metrics:
            self.batch_recons.append(float(batch_metrics["recon"]))
        if "grad_norm" in batch_metrics:
            self.grad_norms.append(float(batch_metrics["grad_norm"]))

    def means(self) -> Dict[str, float]:
        if self.count == 0:
            return {k: 0.0 for k in self.sums} | {"total_count": 0}

        result = {k: self.sums[k] / self.count for k in self.sums} | {"total_count": self.count}

        # Add batch-level statistics
        if self.batch_losses:
            result["loss_max_batch"] = max(self.batch_losses)
            result["loss_min_batch"] = min(self.batch_losses)
        else:
            result["loss_max_batch"] = 0.0
            result["loss_min_batch"] = 0.0

        if self.batch_recons:
            result["recon_max_batch"] = max(self.batch_recons)
        else:
            result["recon_max_batch"] = 0.0

        if self.grad_norms:
            import statistics
            result["grad_norm_max"] = max(self.grad_norms)
            result["grad_norm_std"] = statistics.stdev(self.grad_norms) if len(self.grad_norms) > 1 else 0.0
        else:
            result["grad_norm_max"] = 0.0
            result["grad_norm_std"] = 0.0

        return result

    def reset(self) -> None:
        self.sums.clear()
        self.count = 0
        self.batch_losses.clear()
        self.batch_recons.clear()
        self.grad_norms.clear()

@dataclass
class ClasswiseAverager:
    sums: Dict[str, float] = field(default_factory=dict)

    def update(self, classwise_sums_and_counts: Dict[str, float]) -> None:
        for k, v in classwise_sums_and_counts.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)

    def means(self, num_classes: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in range(num_classes):
            c = self.sums.get(f"count_c{k}", 0.0)
            for m in ("psnr","recon","kld","kld_prior"):
                s = self.sums.get(f"{m}_c{k}", 0.0)
                out[f"{m}_c{k}"] = float(s / c) if c > 0 else 0.0
            out[f"count_c{k}"] = float(c)
        return out

    def reset(self) -> None:
        self.sums.clear()
