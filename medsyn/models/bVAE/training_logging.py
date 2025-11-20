from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
import time

DEFAULT_FIELDS: List[str] = [
    # Basic training info
    "epoch", "split", "time_s", "lr", "beta",

    # Core losses
    "loss", "recon", "kld", "kld_per_dim",

    # Reconstruction quality
    "psnr", "ssim", "mse", "mae",

    # Decoder output health
    "output_mean", "output_std", "output_saturated_ratio",

    # Posterior statistics
    "mu_abs_mean", "mu_max", "mu_std",
    "logv_mean", "logv_std", "logv_min", "logv_max",
    "z_var_mean",

    # Posterior collapse detection
    "active_units", "kld_max", "kld_min",

    # Prior drift (class-conditional)
    "prior_mu_max", "prior_mu_std",
    "prior_logv_min", "prior_logv_max",
    "prior_kld_avg",

    # Gradient & weight health
    "grad_norm", "grad_norm_max", "grad_norm_std",
    "enc_weight_norm", "dec_weight_norm",
    "weight_updates_ratio",

    # Batch statistics (outlier detection)
    "loss_max_batch", "loss_min_batch", "recon_max_batch",

    # Capacity & telemetry
    "capacity_t", "prior_mu_l2", "prior_logv_mean",

    "total_count"
]

@dataclass
class CSVTrainingLogger:
    """
    CSV epoch-level logger with optional extra fields (e.g., per-class metrics).
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
        row = {k: "" for k in self.fieldnames}
        row.update({"epoch": int(epoch), "split": split, "lr": float(lr), "time_s": float(time.time())})
        for k, v in metrics.items():
            if k in row:
                row[k] = v
        with Path(self.csv_path).open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames)
            w.writerow(row)
