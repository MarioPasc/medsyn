"""
CSV logging and checkpoint management for classification training.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import csv
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class CSVLogger:
    """
    CSV logger for training metrics.
    """

    def __init__(self, csv_path: Path):
        """
        Initialize CSV logger.

        Args:
            csv_path: Path to CSV file
        """
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "epoch",
            "train_loss", "train_accuracy", "train_precision", "train_recall", "train_f1", "train_auc_macro",
            "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1", "val_auc_macro",
            "lr"
        ]

        # Create CSV file with header
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float
    ):
        """
        Log metrics for one epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            lr: Current learning rate
        """
        row = {
            "epoch": epoch,
            "lr": lr
        }

        # Add training metrics
        for key, value in train_metrics.items():
            row[f"train_{key}"] = value

        # Add validation metrics
        for key, value in val_metrics.items():
            row[f"val_{key}"] = value

        # Write to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics: Dict[str, float],
    path: Path
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state (optional)
        epoch: Current epoch
        metrics: Metrics dictionary
        path: Path to save checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    logger.debug(f"Checkpoint saved to: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None
) -> int:
    """
    Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    logger.info(f"Checkpoint loaded from: {path} (epoch {epoch})")

    return epoch
