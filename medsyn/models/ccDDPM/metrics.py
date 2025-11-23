# medsyn/models/ccDDPM/metrics.py
# Purpose: Generation and training metrics for synthesis and interpretability.
# Includes: FID via clean-fid (optional), LPIPS (optional), per-class loss export, PSNR, SSIM.
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def compute_psnr(x_hat: torch.Tensor, x: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between reconstructed and original images.
    Assumes inputs are in [0, max_val] range.
    
    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        max_val: Maximum possible pixel value (1.0 for normalized, 255 for uint8)
    
    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(x_hat, x, reduction="mean")
    if mse < 1e-10:
        return 100.0  # Perfect reconstruction
    psnr = 20 * torch.log10(torch.tensor(max_val, device=mse.device) / torch.sqrt(mse))
    return float(psnr.cpu())

def compute_ssim(x_hat: torch.Tensor, x: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    """
    Compute Structural Similarity Index Measure between images.
    Simplified implementation for batch processing.
    
    Args:
        x_hat: Reconstructed images [B, C, H, W]
        x: Original images [B, C, H, W]
        window_size: Size of the Gaussian window
        max_val: Maximum possible pixel value
    
    Returns:
        Mean SSIM across batch
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure
        ssim_val = structural_similarity_index_measure(x_hat, x, data_range=max_val)
        return float(ssim_val.cpu())
    except ImportError:
        # Fallback: simple correlation-based approximation
        logger.warning("torchmetrics not available, using simplified SSIM approximation")
        x_mean = x.mean(dim=(2, 3), keepdim=True)
        x_hat_mean = x_hat.mean(dim=(2, 3), keepdim=True)
        x_var = ((x - x_mean) ** 2).mean(dim=(2, 3), keepdim=True)
        x_hat_var = ((x_hat - x_hat_mean) ** 2).mean(dim=(2, 3), keepdim=True)
        covar = ((x - x_mean) * (x_hat - x_hat_mean)).mean(dim=(2, 3), keepdim=True)
        
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        
        ssim = ((2 * x_mean * x_hat_mean + c1) * (2 * covar + c2)) / \
               ((x_mean ** 2 + x_hat_mean ** 2 + c1) * (x_var + x_hat_var + c2))
        return float(ssim.mean().cpu())

def try_fid(real_dir: str | Path, fake_dir: str | Path, device: str = "cuda") -> Optional[float]:
    """
    Compute FID with clean-fid if installed. Returns None if not available.
    """
    try:
        from cleanfid import fid # type: ignore
    except Exception as e:
        logger.warning("clean-fid not available: %s", e)
        return None
    score = fid.compute_fid(str(real_dir), str(fake_dir), device=device)
    return float(score)

def try_lpips(x: torch.Tensor, y: torch.Tensor, net: str = "alex", device: str = "cuda") -> Optional[torch.Tensor]:
    """
    Compute LPIPS if installed. x,y in [0,1]. Returns mean score or None.
    """
    try:
        import lpips  # type: ignore
    except Exception as e:
        logger.warning("lpips not available: %s", e)
        return None
    loss_fn = lpips.LPIPS(net=net).to(device)
    with torch.no_grad():
        s = loss_fn(x.to(device), y.to(device))
    return s.mean()

def export_per_class_loss(loss_tracker, out_csv: Path) -> None:
    """
    Dump per-class training loss means to CSV with columns [class, raw_loss, weighted_loss].

    Uses the loss_tracker's per_class_table() method which returns a pandas DataFrame
    with both raw and weighted per-class losses.
    """
    table = loss_tracker.per_class_table()
    table.to_csv(out_csv, index=False)
