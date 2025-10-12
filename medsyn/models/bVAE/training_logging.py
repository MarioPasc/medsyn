# medsyn/models/bVAE/training_logging.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import csv
import time

DEFAULT_FIELDS: List[str] = [
    "epoch", "split", "time_s", "lr",
    "loss", "recon", "kld", "kld_per_dim",
    "psnr", "mu_abs_mean", "logv_mean", "logv_std", "z_var_mean",
    "total_count"
]

@dataclass
class CSVTrainingLogger:
    """
    CSV epoch-level logger.

    Args:
        csv_path: target CSV file path
        fieldnames: ordered column names; defaults include losses and latent stats

    Methods:
        log_epoch(epoch, split, lr, metrics): append one row
    """
    csv_path: str
    fieldnames: List[str] = None

    def __post_init__(self) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(DEFAULT_FIELDS)
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
