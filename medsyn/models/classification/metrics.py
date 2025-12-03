"""
Classification metrics tracking and computation.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)


class ClassificationMetrics:
    """
    Tracks classification metrics during training/validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with a batch.

        Args:
            loss: Batch loss value
            logits: Model output logits (B, num_classes)
            labels: Ground truth labels (B,)
        """
        batch_size = labels.size(0)
        self.total_loss += loss * batch_size
        self.total_samples += batch_size

        # Convert to numpy
        preds = logits.argmax(dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        self.all_labels.append(labels_np)
        self.all_preds.append(preds)
        self.all_probs.append(probs)

    def accuracy(self) -> float:
        """Compute current accuracy."""
        if self.total_samples == 0:
            return 0.0
        labels = np.concatenate(self.all_labels)
        preds = np.concatenate(self.all_preds)
        return float(accuracy_score(labels, preds))

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric name -> value
        """
        if self.total_samples == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        labels = np.concatenate(self.all_labels)
        preds = np.concatenate(self.all_preds)
        probs = np.concatenate(self.all_probs)

        # Basic metrics
        accuracy = float(accuracy_score(labels, preds))
        avg_loss = float(self.total_loss / self.total_samples)

        # Precision, recall, F1 (macro-averaged)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )

        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

        # AUC (macro-averaged, one-vs-rest)
        try:
            num_classes = probs.shape[1]
            if num_classes == 2:
                # Binary classification
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                # Multiclass (one-vs-rest)
                auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            metrics["auc_macro"] = float(auc)
        except Exception:
            # Not enough data or other issues
            metrics["auc_macro"] = 0.0

        return metrics


def compute_per_class_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    num_classes: int
) -> Dict[str, any]:
    """
    Compute per-class metrics.

    Args:
        labels: Ground truth labels (N,)
        preds: Predicted labels (N,)
        probs: Class probabilities (N, num_classes)
        num_classes: Number of classes

    Returns:
        Dictionary with per-class metrics
    """
    # Per-class precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )

    # Per-class AUC (one-vs-rest)
    per_class_auc = []
    for class_idx in range(num_classes):
        try:
            binary_labels = (labels == class_idx).astype(int)
            if len(np.unique(binary_labels)) == 2:
                auc = roc_auc_score(binary_labels, probs[:, class_idx])
            else:
                auc = 0.0  # Only one class present
        except Exception:
            auc = 0.0
        per_class_auc.append(auc)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return {
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "auc_per_class": per_class_auc,
        "confusion_matrix": cm.tolist()
    }
