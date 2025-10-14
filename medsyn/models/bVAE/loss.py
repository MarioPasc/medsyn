# medsyn/models/bVAE/loss.py
# Purpose: β-VAE losses con free-bits, L1+SSIM, y MMD opcional.

from __future__ import annotations
from typing import Dict, Literal, Optional
import torch
import torch.nn.functional as F

# ---- Recon ----
def recon_loss(x_hat: torch.Tensor, x: torch.Tensor,
               kind: Literal["mse","bce","l1","l1_ssim"],
               l1_w: float = 1.0, ssim_w: float = 0.85) -> torch.Tensor:
    # Ensure inputs are in [0,1] for BCE/SSIM. Autodetect [-1,1] and rescale.
    def _to_01(t: torch.Tensor) -> torch.Tensor:
        tmin, tmax = float(t.min()), float(t.max())
        if tmin < -0.01 or tmax > 1.01:
            # assume roughly [-1,1], map to [0,1]
            return (t.clamp(-1, 1) + 1.0) * 0.5
        return t.clamp(0.0, 1.0)
    if kind in ("bce", "l1_ssim"):
        x_hat = _to_01(x_hat); x = _to_01(x)
    if kind == "mse":
        return F.mse_loss(x_hat, x, reduction="mean")
    if kind == "bce":
        # Clamp to prevent log(0)
        x_hat_clamped = torch.clamp(x_hat, min=1e-7, max=1.0 - 1e-7)
        return F.binary_cross_entropy(x_hat_clamped, x, reduction="mean")
    if kind == "l1":
        return F.l1_loss(x_hat, x, reduction="mean") * l1_w
    # l1_ssim = L1 + (1-SSIM)
    try:
        from pytorch_msssim import ssim
    except Exception as e:
        raise RuntimeError("Instala pytorch-msssim para usar 'l1_ssim'") from e
    l1 = F.l1_loss(x_hat, x, reduction="mean") * l1_w
    s  = ssim(x_hat, x, data_range=1.0, size_average=True)
    # Clamp SSIM to [0, 1] to prevent negative values
    s = torch.clamp(s, min=0.0, max=1.0)
    loss = l1 + ssim_w * (1.0 - s)
    
    # Safeguard
    if torch.isnan(loss) or torch.isinf(loss):
        loss = l1  # fallback to L1 only
    
    return loss

# ---- KL entre gaussianas con free-bits ----
def kl_gaussians_freebits(mu_q: torch.Tensor, logv_q: torch.Tensor,
                          mu_p: torch.Tensor, logv_p: torch.Tensor,
                          free_bits_nats: float, reduce: bool = True) -> torch.Tensor:
    """
    KL(q||p) con free-bits por dimensión.

    Args:
        mu_q, logv_q: posterior q(z|x)
        mu_p, logv_p: prior p(z) o p(z|y)
        free_bits_nats: threshold en nats por dimensión
        reduce: si True, devuelve escalar promedio por dim
    """
    # Clamp logv to prevent numerical instability
    logv_q = torch.clamp(logv_q, min=-20.0, max=20.0)
    logv_p = torch.clamp(logv_p, min=-20.0, max=20.0)
    
    # KL(q||p) por dimensión
    var_q = torch.exp(logv_q)
    var_p = torch.exp(logv_p)
    
    # Add small epsilon to prevent division by zero
    eps = 1e-8
    kl_dim = 0.5 * (logv_p - logv_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + eps) - 1.0)  # (B,D)

    # free-bits por dim: media en batch y umbral
    kl_dim_mean = torch.mean(kl_dim, dim=0)  # (D,)
    kl_dim_adj = torch.clamp(kl_dim_mean, min=free_bits_nats)  # (D,)
    kl = torch.sum(kl_dim_adj)  # suma sobre dims
    
    # Safeguard: replace NaN/Inf with 0
    if torch.isnan(kl) or torch.isinf(kl):
        kl = torch.tensor(0.0, device=kl.device, dtype=kl.dtype)
    
    return kl / mu_q.size(1) if reduce else kl_dim  # escalar promedio por dim

# ---- InfoVAE-MMD (opcional) ----
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy con kernel RBF."""
    def _k(a, b):
        a2 = (a*a).sum(1, keepdim=True)
        b2 = (b*b).sum(1, keepdim=True)
        dist2 = a2 + b2.t() - 2*a@b.t()
        return torch.exp(-dist2 / (2*sigma**2))
    k_xx = _k(x, x)
    k_yy = _k(y, y)
    k_xy = _k(x, y)
    return k_xx.mean() + k_yy.mean() - 2*k_xy.mean()

def mmd_per_class(z_samp: torch.Tensor, mu_p: torch.Tensor, logv_p: torch.Tensor,
                  y: torch.Tensor) -> torch.Tensor:
    """
    MMD por clase: compara q(z|x,y) contra p(z|y) para cada clase.

    Args:
        z_samp: muestras del posterior q(z|x) (B, D)
        mu_p, logv_p: parámetros del prior condicional p(z|y) (B, D)
        y: etiquetas de clase (B,)

    Returns:
        MMD promediado sobre las clases presentes en el batch
    """
    mmd_total = 0.0
    n_classes = 0
    for yk in y.unique():
        mask = (y == yk)
        if mask.sum() < 2:  # necesitamos al menos 2 samples para MMD
            continue
        # Muestras del posterior para esta clase
        z_q = z_samp[mask]
        # Muestras del prior condicional para esta clase
        mu_p_k = mu_p[mask]
        logv_p_k = logv_p[mask]
        eps = torch.randn_like(mu_p_k)
        z_p = mu_p_k + torch.exp(0.5 * logv_p_k) * eps
        # MMD entre q(z|y) y p(z|y) para esta clase
        mmd_k = mmd_rbf(z_q, z_p)
        mmd_total = mmd_total + mmd_k
        n_classes += 1
    return mmd_total / max(1, n_classes) if n_classes > 0 else torch.tensor(0.0, device=z_samp.device)

def bvae_loss(x_hat: torch.Tensor, x: torch.Tensor,
              mu_q: torch.Tensor, logv_q: torch.Tensor,
              recon_kind: Literal["mse","bce","l1","l1_ssim"],
              beta: float, recon_w: float, kld_w: float,
              # NEW:
              mu_p: Optional[torch.Tensor] = None,
              logv_p: Optional[torch.Tensor] = None,
              free_bits_nats: float = 0.0,
              l1_weight: float = 1.0,
              ssim_weight: float = 0.85,
              infommd_use: bool = False,
              infommd_w: float = 10.0,
              capacity: Optional[float] = None,
              capacity_gamma: float = 1000.0,
              prior_reg_w: float = 1e-4,
              per_class_mmd_use: bool = False,
              per_class_mmd_w: float = 1.0,
              y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """
    β-VAE loss con prior condicional, free-bits y MMD opcional.

    Args:
        x_hat, x: reconstrucción y original
        mu_q, logv_q: posterior q(z|x)
        recon_kind: tipo de pérdida de reconstrucción
        beta: peso de KLD (modulado por scheduler)
        recon_w, kld_w: pesos globales
        mu_p, logv_p: prior p(z|y) (si None, usa N(0,I))
        free_bits_nats: umbral de free-bits
        l1_weight, ssim_weight: pesos para l1_ssim
        infommd_use, infommd_w: MMD opcional
    """
    r = recon_loss(x_hat, x, recon_kind, l1_w=l1_weight, ssim_w=ssim_weight)

    if mu_p is None or logv_p is None:
        # prior estándar N(0,I)
        mu_p = torch.zeros_like(mu_q)
        logv_p = torch.zeros_like(logv_q)

    # Capacity control vs free-bits must be mutually exclusive
    if capacity is None:
        # Standard β-VAE: use free-bits to prevent tiny-KL collapse
        k = kl_gaussians_freebits(mu_q, logv_q, mu_p, logv_p, free_bits_nats)
        total = recon_w * r + kld_w * beta * k
    else:
        # Capacity-controlled β-VAE: compute raw KL without free-bits
        # This allows capacity term to shape the rate properly
        k = kl_gaussians_freebits(mu_q, logv_q, mu_p, logv_p, free_bits_nats=0.0)
        total = recon_w * r + capacity_gamma * torch.abs(k - capacity)

    # small L2 on class prior to keep it near N(0,I)
    if mu_p is not None and logv_p is not None and prior_reg_w > 0:
        total = total + prior_reg_w * (mu_p.pow(2).mean() + logv_p.pow(2).mean())

    # InfoVAE-MMD: empuja q(z) hacia p(z|y) condicional a nivel agregado [Zhao+]
    if infommd_use:
        # Comparar contra p(z|y) del propio minibatch
        with torch.no_grad():
            z_p = mu_p + torch.exp(0.5 * logv_p) * torch.randn_like(mu_p)
        z_q = mu_q + torch.exp(0.5 * logv_q) * torch.randn_like(mu_q)
        mmd_loss = mmd_rbf(z_q, z_p)
        # Safeguard MMD
        if torch.isnan(mmd_loss) or torch.isinf(mmd_loss):
            mmd_loss = torch.tensor(0.0, device=mmd_loss.device, dtype=mmd_loss.dtype)
        total = total + infommd_w * mmd_loss

    # Per-class MMD: compara q(z|x,y) contra p(z|y) por cada clase
    if per_class_mmd_use and y is not None and mu_p is not None and logv_p is not None:
        z_q = mu_q + torch.exp(0.5 * logv_q) * torch.randn_like(mu_q)
        mmd_class = mmd_per_class(z_q, mu_p, logv_p, y)
        # Safeguard MMD
        if torch.isnan(mmd_class) or torch.isinf(mmd_class):
            mmd_class = torch.tensor(0.0, device=mmd_class.device, dtype=mmd_class.dtype)
        total = total + per_class_mmd_w * mmd_class

    # Final safeguard: check for NaN/Inf in total loss
    if torch.isnan(total) or torch.isinf(total):
        # Log warning and return a safe loss
        import logging
        logging.getLogger(__name__).warning(
            f"NaN/Inf detected in loss! recon={r.item():.4f}, kld={k.item():.4f}, total={total.item()}"
        )
        total = recon_w * r  # fallback to reconstruction only

    return {"loss": total, "recon": r, "kld": k}
