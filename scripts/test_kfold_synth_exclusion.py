"""
Test script to verify that synthetic images are excluded from validation folds.
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from medsyn.models.classification.kfold_utils import StratifiedKFoldSplitter


def test_kfold_synthetic_exclusion():
    """Test that k-fold validation folds contain no synthetic images."""
    print("=" * 80)
    print("Testing K-Fold Synthetic Image Exclusion")
    print("=" * 80)

    # Create synthetic test data
    np.random.seed(42)

    # Create 100 real images (50 per class)
    n_real = 100
    real_images = np.random.randn(n_real, 3, 64, 64)
    real_labels = np.array([0] * 50 + [1] * 50)
    real_is_synth = np.zeros(n_real, dtype=bool)

    # Create 50 synthetic images (25 per class)
    n_synth = 50
    synth_images = np.random.randn(n_synth, 3, 64, 64)
    synth_labels = np.array([0] * 25 + [1] * 25)
    synth_is_synth = np.ones(n_synth, dtype=bool)

    # Combine real and synthetic data
    train_images = np.concatenate([real_images, synth_images], axis=0)
    train_labels = np.concatenate([real_labels, synth_labels], axis=0)
    train_is_synth = np.concatenate([real_is_synth, synth_is_synth], axis=0)

    print(f"\nTest data created:")
    print(f"  Total train images: {len(train_images)}")
    print(f"  Real images: {(~train_is_synth).sum()}")
    print(f"  Synthetic images: {train_is_synth.sum()}")
    print(f"  Class 0: {(train_labels == 0).sum()}")
    print(f"  Class 1: {(train_labels == 1).sum()}")

    # Create k-fold splitter
    k_folds = 5
    splitter = StratifiedKFoldSplitter(k=k_folds, seed=42, shuffle=True)

    # Create folds
    folds = splitter.create_folds(
        train_images=train_images,
        train_labels=train_labels,
        val_images=None,
        val_labels=None,
        train_is_synth=train_is_synth,
        val_is_synth=None
    )

    # Print fold info
    print(splitter.get_fold_info(folds))

    # Verify that no validation fold contains synthetic images
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)

    all_passed = True
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        n_val_synth = val_fold['is_synth'].sum()
        n_val_real = (~val_fold['is_synth']).sum()

        if n_val_synth > 0:
            print(f"❌ FAILED: Fold {fold_idx} has {n_val_synth} synthetic images in validation!")
            all_passed = False
        else:
            print(f"✓ PASSED: Fold {fold_idx} validation has 0 synthetic images ({n_val_real} real images)")

    # Also check that training folds can contain synthetic images
    print("\nTraining fold synthetic image counts:")
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        n_train_synth = train_fold['is_synth'].sum()
        n_train_real = (~train_fold['is_synth']).sum()
        print(f"  Fold {fold_idx}: {n_train_real} real, {n_train_synth} synthetic")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED: No synthetic images in validation folds")
    else:
        print("❌ TESTS FAILED: Some validation folds contain synthetic images")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = test_kfold_synthetic_exclusion()
    sys.exit(0 if success else 1)
