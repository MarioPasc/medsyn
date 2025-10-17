# medsyn/models/ccDDPM/metrics.py
# Purpose: Generation and training metrics for synthesis and interpretability.
# Includes: FID via clean-fid (optional), LPIPS (optional), per-class loss export.
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import torch

logger = logging.getLogger(__name__)

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
    Dump per-class training loss means to CSV.
    """
    import csv
    stats = loss_tracker.per_class()
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class","mse_loss"])
        for c, v in stats.items():
            w.writerow([c, v])
