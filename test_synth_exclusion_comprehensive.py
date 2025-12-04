"""
Comprehensive test to verify synthetic images are excluded from validation and test sets.
"""
import numpy as np
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from medsyn.models.classification.kfold_utils import StratifiedKFoldSplitter
from medsyn.models.classification.dataloaders import ClassificationDataset


def test_kfold_exclusion():
    """Test that k-fold validation folds exclude synthetic images."""
    print("\n" + "=" * 80)
    print("TEST 1: K-Fold Validation Exclusion")
    print("=" * 80)

    # Create test data with synthetic images
    np.random.seed(42)
    n_real, n_synth = 100, 50

    real_images = np.random.randn(n_real, 3, 64, 64)
    real_labels = np.array([0] * 50 + [1] * 50)
    real_is_synth = np.zeros(n_real, dtype=bool)

    synth_images = np.random.randn(n_synth, 3, 64, 64)
    synth_labels = np.array([0] * 25 + [1] * 25)
    synth_is_synth = np.ones(n_synth, dtype=bool)

    train_images = np.concatenate([real_images, synth_images], axis=0)
    train_labels = np.concatenate([real_labels, synth_labels], axis=0)
    train_is_synth = np.concatenate([real_is_synth, synth_is_synth], axis=0)

    print(f"Test data: {n_real} real + {n_synth} synthetic = {len(train_images)} total")

    # Create k-fold splits
    splitter = StratifiedKFoldSplitter(k=5, seed=42, shuffle=True)
    folds = splitter.create_folds(
        train_images=train_images,
        train_labels=train_labels,
        val_images=None,
        val_labels=None,
        train_is_synth=train_is_synth,
        val_is_synth=None
    )

    # Verify no synthetic images in validation folds
    test1_passed = True
    for fold_idx, (train_fold, val_fold) in enumerate(folds):
        n_val_synth = val_fold['is_synth'].sum()
        if n_val_synth > 0:
            print(f"  ❌ Fold {fold_idx}: {n_val_synth} synthetic images in validation")
            test1_passed = False
        else:
            print(f"  ✓ Fold {fold_idx}: validation has only real images ({len(val_fold['labels'])} samples)")

    return test1_passed


def test_dataloader_test_exclusion():
    """Test that test set dataloader excludes synthetic images."""
    print("\n" + "=" * 80)
    print("TEST 2: Test Set Dataloader Exclusion")
    print("=" * 80)

    # Create temporary NPZ file with test set containing synthetic images
    np.random.seed(42)

    # Test set: 20 real + 5 synthetic (which should be excluded)
    test_images = np.random.randn(25, 3, 64, 64).astype(np.float32)
    test_labels = np.array([0] * 12 + [1] * 13)
    test_is_synth = np.array([False] * 20 + [True] * 5)  # Last 5 are synthetic

    # Create minimal train data (required for NPZ)
    train_images = np.random.randn(10, 3, 64, 64).astype(np.float32)
    train_labels = np.array([0] * 5 + [1] * 5)
    train_is_synth = np.zeros(10, dtype=bool)

    print(f"Test set: {(~test_is_synth).sum()} real + {test_is_synth.sum()} synthetic")

    # Create temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        np.savez(
            tmp_path,
            train_images=train_images,
            train_labels=train_labels,
            train_is_synth=train_is_synth,
            test_images=test_images,
            test_labels=test_labels,
            test_is_synth=test_is_synth
        )

    try:
        # Create dataset for test split
        test_dataset = ClassificationDataset(
            npz_path=tmp_path,
            split="test",
            regime="real_only",
            image_size=64
        )

        # Verify that synthetic images were excluded
        n_expected_real = (~test_is_synth).sum()
        n_actual = len(test_dataset)

        if n_actual == n_expected_real:
            print(f"  ✓ Test dataset has {n_actual} samples (expected {n_expected_real} real images)")
            test2_passed = True
        else:
            print(f"  ❌ Test dataset has {n_actual} samples (expected {n_expected_real} real images)")
            test2_passed = False

    finally:
        # Clean up temporary file
        tmp_path.unlink()

    return test2_passed


def test_dataloader_val_exclusion():
    """Test that validation set dataloader excludes synthetic images."""
    print("\n" + "=" * 80)
    print("TEST 3: Validation Set Dataloader Exclusion")
    print("=" * 80)

    # Create temporary NPZ file with val set containing synthetic images
    np.random.seed(42)

    # Val set: 15 real + 3 synthetic (which should be excluded)
    val_images = np.random.randn(18, 3, 64, 64).astype(np.float32)
    val_labels = np.array([0] * 9 + [1] * 9)
    val_is_synth = np.array([False] * 15 + [True] * 3)  # Last 3 are synthetic

    # Create minimal train data (required for NPZ)
    train_images = np.random.randn(10, 3, 64, 64).astype(np.float32)
    train_labels = np.array([0] * 5 + [1] * 5)
    train_is_synth = np.zeros(10, dtype=bool)

    print(f"Val set: {(~val_is_synth).sum()} real + {val_is_synth.sum()} synthetic")

    # Create temporary NPZ file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        np.savez(
            tmp_path,
            train_images=train_images,
            train_labels=train_labels,
            train_is_synth=train_is_synth,
            val_images=val_images,
            val_labels=val_labels,
            val_is_synth=val_is_synth
        )

    try:
        # Create dataset for val split
        val_dataset = ClassificationDataset(
            npz_path=tmp_path,
            split="val",
            regime="real_plus_synth",  # Even with this regime, val should exclude synth
            image_size=64
        )

        # Verify that synthetic images were excluded
        n_expected_real = (~val_is_synth).sum()
        n_actual = len(val_dataset)

        if n_actual == n_expected_real:
            print(f"  ✓ Val dataset has {n_actual} samples (expected {n_expected_real} real images)")
            test3_passed = True
        else:
            print(f"  ❌ Val dataset has {n_actual} samples (expected {n_expected_real} real images)")
            test3_passed = False

    finally:
        # Clean up temporary file
        tmp_path.unlink()

    return test3_passed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SYNTHETIC IMAGE EXCLUSION TESTS")
    print("=" * 80)

    test1 = test_kfold_exclusion()
    test2 = test_dataloader_test_exclusion()
    test3 = test_dataloader_val_exclusion()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  K-Fold Validation:     {'✓ PASSED' if test1 else '❌ FAILED'}")
    print(f"  Test Set Dataloader:   {'✓ PASSED' if test2 else '❌ FAILED'}")
    print(f"  Val Set Dataloader:    {'✓ PASSED' if test3 else '❌ FAILED'}")

    all_passed = test1 and test2 and test3
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80 + "\n")

    sys.exit(0 if all_passed else 1)
