"""
K-Fold Cross-Validation utilities for classification training.
"""

from typing import List, Tuple, Optional
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
        val_images: Optional[np.ndarray],
        val_labels: Optional[np.ndarray],
        train_is_synth: np.ndarray,
        val_is_synth: Optional[np.ndarray]
    ) -> List[Tuple[dict, dict]]:
        """
        Create k-fold splits from train data (and optionally validation data).

        If validation data is provided, it will be combined with training data
        before splitting into k folds. Otherwise, only training data is used.

        Args:
            train_images: Training images from NPZ
            train_labels: Training labels from NPZ
            val_images: Validation images from NPZ (optional, can be None)
            val_labels: Validation labels from NPZ (optional, can be None)
            train_is_synth: Training synthetic flags from NPZ
            val_is_synth: Validation synthetic flags from NPZ (optional, can be None)

        Returns:
            List of (train_data_dict, val_data_dict) tuples, one per fold.
            Each dict contains: 'images', 'labels', 'is_synth', 'indices'
        """
        # Combine train and validation data (if validation data exists)
        if val_images is not None and val_labels is not None:
            combined_images = np.concatenate([train_images, val_images], axis=0)
            combined_labels = np.concatenate([train_labels, val_labels], axis=0)
            if val_is_synth is not None:
                combined_is_synth = np.concatenate([train_is_synth, val_is_synth], axis=0)
            else:
                # If val_is_synth not provided, assume all val images are real
                combined_is_synth = np.concatenate([
                    train_is_synth,
                    np.zeros(len(val_labels), dtype=bool)
                ], axis=0)
        else:
            # No validation data - use only training data for k-fold CV
            combined_images = train_images
            combined_labels = train_labels
            combined_is_synth = train_is_synth

        # Create indices array for tracking
        n_total = len(combined_images)
        all_indices = np.arange(n_total)

        # Generate stratified folds
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            self.skf.split(combined_images, combined_labels)
        ):
            # IMPORTANT: Filter validation fold to exclude synthetic images
            # Only real images should be in validation/test sets for proper evaluation
            val_is_synth_flags = combined_is_synth[val_idx]
            val_real_mask = ~val_is_synth_flags  # Keep only real images

            # Get indices of real images in validation fold
            val_real_indices = val_idx[val_real_mask]

            # Count synthetic images that were excluded from validation
            n_synth_excluded = val_is_synth_flags.sum()

            # Log warning if synthetic images were in validation fold
            if n_synth_excluded > 0:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Fold {fold_idx}: Excluded {n_synth_excluded} synthetic images "
                    f"from validation set (validation now has {len(val_real_indices)} real images only)"
                )

            train_fold = {
                'images': combined_images[train_idx],
                'labels': combined_labels[train_idx],
                'is_synth': combined_is_synth[train_idx],
                'indices': train_idx
            }

            val_fold = {
                'images': combined_images[val_real_indices],
                'labels': combined_labels[val_real_indices],
                'is_synth': combined_is_synth[val_real_indices],  # Should all be False
                'indices': val_real_indices
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
        info_lines.append("NOTE: Validation folds contain ONLY real images (is_synth=False)")
        info_lines.append("      Synthetic images are excluded from validation for proper evaluation")
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
                n_train_real = n_train - n_train_synth
                n_val_synth = val_fold['is_synth'].sum()
                n_val_real = n_val - n_val_synth

                info_lines.append(
                    f"  Train: {n_train_real} real, {n_train_synth} synthetic"
                )
                info_lines.append(
                    f"  Val:   {n_val_real} real, {n_val_synth} synthetic"
                )

                # Verification: validation should have NO synthetic images
                if n_val_synth > 0:
                    info_lines.append(
                        f"  ⚠ WARNING: Validation fold has {n_val_synth} synthetic images! "
                        f"This should not happen."
                    )
                else:
                    info_lines.append(
                        f"  ✓ Validation fold verified: contains only real images"
                    )

        return "\n".join(info_lines)
