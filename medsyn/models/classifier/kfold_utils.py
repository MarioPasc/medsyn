"""
K-Fold Cross-Validation utilities for classification training.
"""

from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold


class StratifiedKFoldSplitter:
    """
    Creates stratified k-fold splits from training data while maintaining class balance.

    This splitter combines the original train and validation splits from the NPZ file
    and creates k stratified folds for cross-validation. The test set remains untouched
    for final evaluation.

    Args:
        k: Number of folds (default: 5)
        seed: Random seed for reproducibility (default: 42)
        shuffle: Whether to shuffle data before splitting (default: True)
    """

    def __init__(self, k: int = 5, seed: int = 42, shuffle: bool = True):
        self.k = k
        self.seed = seed
        self.shuffle = shuffle
        self.skf = StratifiedKFold(
            n_splits=k,
            shuffle=shuffle,
            random_state=seed
        )

    def create_folds(
        self,
        train_images: np.ndarray,
        train_labels: np.ndarray,
        val_images: np.ndarray,
        val_labels: np.ndarray,
        train_is_synth: np.ndarray,
        val_is_synth: np.ndarray
    ) -> List[Tuple[dict, dict]]:
        """
        Create k-fold splits from combined train and validation data.

        Args:
            train_images: Training images from NPZ
            train_labels: Training labels from NPZ
            val_images: Validation images from NPZ
            val_labels: Validation labels from NPZ
            train_is_synth: Training synthetic flags from NPZ
            val_is_synth: Validation synthetic flags from NPZ

        Returns:
            List of (train_data_dict, val_data_dict) tuples, one per fold.
            Each dict contains: 'images', 'labels', 'is_synth', 'indices'
        """
        # Combine train and validation data
        combined_images = np.concatenate([train_images, val_images], axis=0)
        combined_labels = np.concatenate([train_labels, val_labels], axis=0)
        combined_is_synth = np.concatenate([train_is_synth, val_is_synth], axis=0)

        # Create indices array for tracking
        n_total = len(combined_images)
        all_indices = np.arange(n_total)

        # Generate stratified folds
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.skf.split(combined_images, combined_labels)
        ):
            train_fold = {
                'images': combined_images[train_idx],
                'labels': combined_labels[train_idx],
                'is_synth': combined_is_synth[train_idx],
                'indices': train_idx
            }

            val_fold = {
                'images': combined_images[val_idx],
                'labels': combined_labels[val_idx],
                'is_synth': combined_is_synth[val_idx],
                'indices': val_idx
            }

            folds.append((train_fold, val_fold))

        return folds

    def get_fold_info(self, folds: List[Tuple[dict, dict]]) -> str:
        """
        Generate summary information about the created folds.

        Args:
            folds: List of fold tuples from create_folds()

        Returns:
            Formatted string with fold statistics
        """
        info_lines = [f"\nStratified {self.k}-Fold Cross-Validation (seed={self.seed})"]
        info_lines.append("=" * 60)

        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            n_train = len(train_fold['labels'])
            n_val = len(val_fold['labels'])

            # Class distribution in training fold
            train_classes, train_counts = np.unique(
                train_fold['labels'], return_counts=True
            )

            # Class distribution in validation fold
            val_classes, val_counts = np.unique(
                val_fold['labels'], return_counts=True
            )

            info_lines.append(f"\nFold {fold_idx}:")
            info_lines.append(f"  Train: {n_train} samples")
            info_lines.append(f"  Val:   {n_val} samples")

            # Check for synthetic data distribution
            if 'is_synth' in train_fold:
                n_train_synth = train_fold['is_synth'].sum()
                n_val_synth = val_fold['is_synth'].sum()
                info_lines.append(
                    f"  Synth: {n_train_synth}/{n_train} (train), "
                    f"{n_val_synth}/{n_val} (val)"
                )

        return "\n".join(info_lines)
