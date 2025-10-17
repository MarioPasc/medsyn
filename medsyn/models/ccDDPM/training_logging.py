# medsyn/models/ccDDPM/training_logging.py
# Purpose: CSV epoch-level logger for ccDDPM training metrics.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
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
    "total_count"
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
    Accumulate metrics across batches and compute weighted averages.
    """
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.total_samples = 0

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Update running sums with batch metrics."""
        self.total_samples += batch_size
        for k, v in metrics.items():
            if k not in self.sums:
                self.sums[k] = 0.0
                self.counts[k] = 0
            self.sums[k] += float(v) * batch_size
            self.counts[k] += batch_size

    def means(self) -> Dict[str, float]:
        """Compute weighted means."""
        out = {}
        for k in self.sums:
            n = self.counts.get(k, 0)
            out[k] = self.sums[k] / max(n, 1)
        out["total_count"] = self.total_samples
        return out

    def reset(self) -> None:
        """Clear all accumulated values."""
        self.sums.clear()
        self.counts.clear()
        self.total_samples = 0
