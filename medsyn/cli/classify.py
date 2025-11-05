#!/usr/bin/env python
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from ultralytics.models.yolo.classify import ClassificationTrainer
from medsyn.models.classifier.config import load_cfg
from medsyn.models.classifier.engine.yolo_trainer import MedsynClassificationTrainer
from medsyn.models.classifier.engine.yolo_validator import MedsynClassificationValidator

def setup_logging(log_file: Path = None, level: str = "INFO"):
    """Setup professional logging to console and file."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger

def main():
    ap = argparse.ArgumentParser(
        description="Train or validate YOLO classifier on PathMNIST with NPZ data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--config", required=True, help="Path to medsyn_cfg.yaml")
    ap.add_argument("--hparams", required=True, help="Path to yolo_hyperparameters.yaml")
    ap.add_argument("--val_only", action="store_true", help="Run validation only (no training)")
    ap.add_argument("--training_mode", type=str, choices=["real", "real_synth", "synth"],
                    help="Override training mode: 'real' (PathMNIST only), 'real_synth' (balanced mix), 'synth' (synthetic only)")
    ap.add_argument("--name", type=str, help="Override experiment name")
    ap.add_argument("--device", type=str, help="Override device (e.g., '0', '1', 'cpu')")
    ap.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging level")
    args = ap.parse_args()

    # Load configuration
    cfg = load_cfg(args.config, args.hparams)

    # Override training mode if specified
    if args.training_mode:
        mode_map = {
            "real": "PathMNIST",
            "real_synth": "PathMNIST_and_synth",
            "synth": "synth"
        }
        cfg.training_images = mode_map[args.training_mode]
        # Update the custom medsyn parameter
        cfg.medsyn_params["medsyn_training_images"] = cfg.training_images

    # Override name if specified
    if args.name:
        cfg.name = args.name
        cfg.overrides["name"] = args.name

    # Override device if specified
    if args.device:
        cfg.overrides["device"] = args.device
    cfg.overrides.setdefault("data", "__npz__")

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = cfg.project / cfg.name
    log_file = log_dir / f"training_{timestamp}.log"
    logger = setup_logging(log_file=log_file, level=args.log_level)

    # Log configuration
    logger.info("="*80)
    logger.info("YOLO Classification Training")
    logger.info("="*80)
    logger.info(f"Config file: {args.config}")
    logger.info(f"Hyperparameters file: {args.hparams}")
    logger.info(f"NPZ path: {cfg.npz_path}")
    logger.info(f"Training mode: {cfg.training_images}")
    logger.info(f"Model: {cfg.model}")
    logger.info(f"Image size: {cfg.imgsz}")
    logger.info(f"Epochs: {cfg.epochs}")
    logger.info(f"Batch size: {cfg.batch}")
    logger.info(f"Workers: {cfg.workers}")
    logger.info(f"Project: {cfg.project}")
    logger.info(f"Experiment name: {cfg.name}")
    logger.info(f"Device: {cfg.overrides.get('device', 'auto')}")
    logger.info(f"Validation only: {args.val_only}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)

    try:
        # Trainer with NPZ-backed dataloaders
        logger.info("Initializing trainer...")

        # Initialize trainer with only valid YOLO parameters
        trainer: ClassificationTrainer = MedsynClassificationTrainer(overrides=cfg.overrides)

        # Attach custom MedSyn runtime parameters OUTSIDE of YOLO args
        # (YOLO revalidates args and rejects unknown keys)
        trainer.medsyn_npz_path = str(cfg.npz_path)
        trainer.medsyn_training_images = cfg.training_images

        # Validator
        trainer.validator = MedsynClassificationValidator(args=trainer.args)
        trainer.validator.medsyn_npz_path = str(cfg.npz_path)
        trainer.validator.medsyn_training_images = cfg.training_images

        if args.val_only:
            logger.info("Starting validation...")
            results = trainer.validate()
            logger.info("Validation completed successfully")
            logger.info(f"Validation results: {results}")
        else:
            logger.info("Starting training...")
            results = trainer.train()
            logger.info("Training completed successfully")
            logger.info(f"Final results saved to: {trainer.save_dir}")

        logger.info("="*80)
        logger.info("Process completed successfully")
        logger.info("="*80)

    except Exception as e:
        logger.error("="*80)
        logger.error(f"Training/validation failed with error: {e}")
        logger.error("="*80)
        raise

if __name__ == "__main__":
    main()
