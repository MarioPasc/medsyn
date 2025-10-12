# medsyn/models/bVAE/loss.py
# Purpose: β-VAE losses (reconstruction + β·KLD) with selectable recon criterion.

from __future__ import annotations
from typing import Literal, Dict
import torch
import torch.nn.functional as F

def kld_standard_normal(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
    # KLD(q(z|x) || N(0,I)) = -0.5 * sum(1 + logσ^2 − μ^2 − σ^2)
    return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())

def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, kind: Literal["mse","bce"]) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
    # BCE expects probabilities if x_hat in [0,1]
    return F.binary_cross_entropy(x_hat, x, reduction="mean")

def bvae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logv: torch.Tensor,
              recon_kind: Literal["mse","bce"], beta: float, recon_w: float, kld_w: float) -> Dict[str, torch.Tensor]:
    r = recon_loss(x_hat, x, recon_kind)
    k = kld_standard_normal(mu, logv)
    total = recon_w * r + kld_w * beta * k
    return {"loss": total, "recon": r, "kld": k}
