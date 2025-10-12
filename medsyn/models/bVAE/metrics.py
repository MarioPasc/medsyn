from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.nn.functional as F

# --------------------------- Scalar metrics -----------------------------------

def psnr_batch(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = torch.mean((x_hat - x) ** 2, dtype=torch.float32)
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10((max_val ** 2) / mse)

def kld_standard_normal(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
    return F.binary_cross_entropy(x_hat, x, reduction="mean")

def latent_stats(mu: torch.Tensor, logv: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "mu_abs_mean": mu.abs().mean(),
        "logv_mean": logv.mean(),
        "logv_std": logv.std(unbiased=False),
        "z_var_mean": torch.mean(torch.exp(logv)),
    }

def make_batch_metrics_dict(loss_total: torch.Tensor,
                            loss_recon: torch.Tensor,
                            loss_kld: torch.Tensor,
                            x_hat: torch.Tensor,
                            x: torch.Tensor,
                            mu: torch.Tensor,
                            logv: torch.Tensor,
                            latent_dim: int) -> Dict[str, float]:
    ls = latent_stats(mu, logv)
    return {
        "loss": float(loss_total.detach().cpu()),
        "recon": float(loss_recon.detach().cpu()),
        "kld": float(loss_kld.detach().cpu()),
        "kld_per_dim": float(loss_kld.detach().cpu() / max(1, latent_dim)),
        "psnr": float(psnr_batch(x_hat.detach(), x.detach()).cpu()),
        "mu_abs_mean": float(ls["mu_abs_mean"].detach().cpu()),
        "logv_mean": float(ls["logv_mean"].detach().cpu()),
        "logv_std": float(ls["logv_std"].detach().cpu()),
        "z_var_mean": float(ls["z_var_mean"].detach().cpu()),
    }

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
                                y: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Returns per-class SUMS and COUNTS so that an averager can compute means safely.
    Keys: psnr_c{k}, recon_c{k}, kld_c{k}, count_c{k}
    """
    ps = _psnr_per_sample(x_hat, x)
    rs = _recon_mse_per_sample(x_hat, x)
    ks = _kld_per_sample(mu, logv)
    out: Dict[str, float] = {}
    for k in range(num_classes):
        mask = (y == k)
        c = int(mask.sum().item())
        if c == 0:
            out[f"psnr_c{k}"] = 0.0
            out[f"recon_c{k}"] = 0.0
            out[f"kld_c{k}"] = 0.0
            out[f"count_c{k}"] = 0.0
        else:
            out[f"psnr_c{k}"] = float(ps[mask].sum().item())
            out[f"recon_c{k}"] = float(rs[mask].sum().item())
            out[f"kld_c{k}"] = float(ks[mask].sum().item())
            out[f"count_c{k}"] = float(c)
    return out

# --------------------------- Averagers ----------------------------------------

@dataclass
class EpochAverager:
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
            for m in ("psnr","recon","kld"):
                s = self.sums.get(f"{m}_c{k}", 0.0)
                out[f"{m}_c{k}"] = float(s / c) if c > 0 else 0.0
            out[f"count_c{k}"] = float(c)
        return out

    def reset(self) -> None:
        self.sums.clear()
