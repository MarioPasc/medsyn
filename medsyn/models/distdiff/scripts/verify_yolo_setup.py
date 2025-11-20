#!/usr/bin/env python
"""
Verification script to check YOLO classifier setup.
Tests NPZ loading, data filtering, and basic functionality.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def verify_imports():
    """Verify all required imports work."""
    logger.info("Verifying imports...")
    try:
        import torch
        import ultralytics
        import sklearn
        from medsyn.cli.classify import main
        from medsyn.models.classifier.engine.yolo_validator import MedsynClassificationValidator
        from medsyn.models.classifier.engine.yolo_trainer import MedsynClassificationTrainer
        from medsyn.models.classifier.dataloaders import build_npz_loader
        from medsyn.models.classifier.utils import select_indices_by_training_images
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def verify_npz_file(npz_path: Path):
    """Verify NPZ file structure and contents."""
    logger.info(f"Verifying NPZ file: {npz_path}")

    if not npz_path.exists():
        logger.error(f"✗ NPZ file not found: {npz_path}")
        return False

    try:
        z = np.load(npz_path)

        # Check required keys
        required_keys = [
            'train_images', 'train_labels',
            'val_images', 'val_labels',
            'test_images', 'test_labels'
        ]

        optional_keys = [
            'train_is_synth', 'val_is_synth', 'test_is_synth'
        ]

        for key in required_keys:
            if key not in z:
                logger.error(f"✗ Missing required key: {key}")
                return False
            logger.info(f"  {key}: shape={z[key].shape}, dtype={z[key].dtype}")

        for key in optional_keys:
            if key in z:
                logger.info(f"  {key}: shape={z[key].shape}, dtype={z[key].dtype}")
            else:
                logger.warning(f"  {key}: NOT FOUND (will use zeros)")

        # Verify data consistency
        splits = ['train', 'val', 'test']
        for split in splits:
            n_images = z[f'{split}_images'].shape[0]
            n_labels = z[f'{split}_labels'].shape[0]

            if n_images != n_labels:
                logger.error(f"✗ {split}: images and labels have different sizes ({n_images} vs {n_labels})")
                return False

            if f'{split}_is_synth' in z:
                n_synth = z[f'{split}_is_synth'].shape[0]
                if n_synth != n_images:
                    logger.error(f"✗ {split}: is_synth has different size ({n_synth} vs {n_images})")
                    return False

        logger.info("✓ NPZ file structure is valid")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to load NPZ file: {e}")
        return False

def verify_data_filtering(npz_path: Path):
    """Verify data filtering by training mode."""
    logger.info("Verifying data filtering by training mode...")

    try:
        from medsyn.models.classifier.utils import select_indices_by_training_images

        z = np.load(npz_path)

        # Get train data
        train_labels = z['train_labels']
        train_is_synth = z.get('train_is_synth', np.zeros_like(train_labels, dtype=np.uint8))

        n_total = len(train_labels)
        n_real = np.sum(~train_is_synth.astype(bool))
        n_synth = np.sum(train_is_synth.astype(bool))

        logger.info(f"Train set composition:")
        logger.info(f"  Total: {n_total}")
        logger.info(f"  Real (is_synth=0): {n_real} ({100*n_real/n_total:.1f}%)")
        logger.info(f"  Synth (is_synth=1): {n_synth} ({100*n_synth/n_total:.1f}%)")

        # Test filtering modes
        modes = {
            'PathMNIST': 'real only',
            'PathMNIST_and_synth': 'real + synth',
            'synth': 'synth only'
        }

        for mode, desc in modes.items():
            mask = select_indices_by_training_images(train_is_synth, mode)
            n_selected = np.sum(mask)
            logger.info(f"  Mode '{mode}' ({desc}): {n_selected} samples")

        logger.info("✓ Data filtering works correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Data filtering failed: {e}")
        return False

def verify_dataloader(npz_path: Path):
    """Verify dataloader can be created."""
    logger.info("Verifying dataloader creation...")

    try:
        from medsyn.models.classifier.dataloaders import build_npz_loader

        # Try to build a train loader
        loader = build_npz_loader(
            npz_path=npz_path,
            split='train',
            imgsz=64,
            batch=8,
            workers=0,  # Use 0 workers for testing
            training_images='PathMNIST',
            augment=True
        )

        # Get one batch
        batch = next(iter(loader))
        logger.info(f"  Sample batch: img shape={batch['img'].shape}, cls shape={batch['cls'].shape}")

        logger.info("✓ Dataloader works correctly")
        return True

    except Exception as e:
        logger.error(f"✗ Dataloader failed: {e}")
        return False

def main():
    logger.info("="*80)
    logger.info("YOLO Classifier Setup Verification")
    logger.info("="*80)

    # Load config to get NPZ path
    config_path = Path("config/medsyn_cfg.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Please run this script from the project root directory")
        return False

    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        npz_path_str = cfg['data']['postprocess_npz']['npz_path']
        npz_path = Path(npz_path_str).expanduser().resolve()

    except Exception as e:
        logger.error(f"Failed to load NPZ path from config: {e}")
        return False

    # Run verification steps
    results = []

    results.append(("Imports", verify_imports()))
    results.append(("NPZ File Structure", verify_npz_file(npz_path)))

    if results[-1][1]:  # Only test these if NPZ file is valid
        results.append(("Data Filtering", verify_data_filtering(npz_path)))
        results.append(("Dataloader", verify_dataloader(npz_path)))

    # Summary
    logger.info("="*80)
    logger.info("Verification Summary:")
    logger.info("="*80)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status:8} {name}")
        if not passed:
            all_passed = False

    logger.info("="*80)

    if all_passed:
        logger.info("✓ All checks passed! System is ready for training.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Review config/medsyn_cfg.yaml and config/yolo_hyperparameters.yaml")
        logger.info("  2. Submit training job: sbatch scripts/train_yolo_parallel.sh")
        logger.info("  3. Or run single training: python -m medsyn.cli.classify --config config/medsyn_cfg.yaml --hparams config/yolo_hyperparameters.yaml --training_mode real")
        return True
    else:
        logger.error("✗ Some checks failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
