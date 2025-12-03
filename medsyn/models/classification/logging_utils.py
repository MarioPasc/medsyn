"""
CSV logging and checkpoint management for classification training.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List
import csv
import logging
import yaml
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


class PerClassCSVLogger:
    """
    CSV logger for per-class metrics across epochs.
    Saves per-class losses, F1 scores, and AUC-ROC values for each epoch.
    """

    def __init__(self, csv_path: Path, num_classes: int, split: str = "train"):
        """
        Initialize per-class CSV logger.

        Args:
            csv_path: Path to CSV file
            num_classes: Number of classes
            split: Data split name (train, val, or test)
        """
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.split = split

        # Create fieldnames for per-class metrics
        self.fieldnames = ["epoch"]
        for class_idx in range(num_classes):
            self.fieldnames.extend([
                f"class_{class_idx}_loss",
                f"class_{class_idx}_f1",
                f"class_{class_idx}_auc"
            ])

        # Create CSV file with header
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_epoch(self, epoch: int, per_class_metrics: Dict[str, any]):
        """
        Log per-class metrics for one epoch.

        Args:
            epoch: Epoch number
            per_class_metrics: Dictionary containing per-class metrics
                Expected keys: "loss", "f1", "auc"
                Each should be a dict/list indexed by class
        """
        row = {"epoch": epoch}

        # Extract per-class metrics
        per_class_loss = per_class_metrics.get("loss", {})
        per_class_f1 = per_class_metrics.get("f1", [])
        per_class_auc = per_class_metrics.get("auc", [])

        for class_idx in range(self.num_classes):
            # Loss (dict indexed by class_idx)
            if isinstance(per_class_loss, dict):
                row[f"class_{class_idx}_loss"] = per_class_loss.get(class_idx, 0.0)
            else:
                row[f"class_{class_idx}_loss"] = 0.0

            # F1 (list indexed by class_idx)
            if isinstance(per_class_f1, list) and class_idx < len(per_class_f1):
                row[f"class_{class_idx}_f1"] = per_class_f1[class_idx]
            else:
                row[f"class_{class_idx}_f1"] = 0.0

            # AUC (list indexed by class_idx)
            if isinstance(per_class_auc, list) and class_idx < len(per_class_auc):
                row[f"class_{class_idx}_auc"] = per_class_auc[class_idx]
            else:
                row[f"class_{class_idx}_auc"] = 0.0

        # Write to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class BestEpochMetricsSaver:
    """
    Saves the best epoch metrics (F1 and AUC-ROC) for final evaluation.
    """

    def __init__(self, save_path: Path, num_classes: int):
        """
        Initialize best epoch metrics saver.

        Args:
            save_path: Path to save best epoch metrics (YAML file)
            num_classes: Number of classes
        """
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.best_epoch = 0
        self.best_metrics = None

    def update(self, epoch: int, train_metrics: Dict, val_metrics: Dict, test_metrics: Dict = None):
        """
        Update best epoch metrics.

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics with per-class data
            val_metrics: Validation metrics with per-class data
            test_metrics: Optional test metrics with per-class data
        """
        self.best_epoch = epoch
        self.best_metrics = {
            "epoch": epoch,
            "train": self._extract_per_class_metrics(train_metrics),
            "val": self._extract_per_class_metrics(val_metrics)
        }
        if test_metrics is not None:
            self.best_metrics["test"] = self._extract_per_class_metrics(test_metrics)

    def _extract_per_class_metrics(self, metrics: Dict) -> Dict:
        """
        Extract per-class F1 and AUC metrics from metrics dictionary.

        Args:
            metrics: Metrics dictionary with per_class subdictionary

        Returns:
            Dictionary with extracted per-class F1 and AUC
        """
        if "per_class" not in metrics:
            return {}

        per_class = metrics["per_class"]
        return {
            "loss_overall": metrics.get("loss", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "f1_macro": metrics.get("f1", 0.0),
            "auc_macro": metrics.get("auc_macro", 0.0),
            "per_class_loss": per_class.get("loss", {}),
            "per_class_f1": per_class.get("f1", []),
            "per_class_auc": per_class.get("auc", []),
            "per_class_precision": per_class.get("precision", []),
            "per_class_recall": per_class.get("recall", []),
            "per_class_support": per_class.get("support", [])
        }

    def save(self):
        """Save the best epoch metrics to YAML file."""
        if self.best_metrics is None:
            logger.warning("No best metrics to save")
            return

        with open(self.save_path, "w") as f:
            yaml.safe_dump(self.best_metrics, f, default_flow_style=False)

        logger.info(f"Best epoch metrics (epoch {self.best_epoch}) saved to: {self.save_path}")


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
