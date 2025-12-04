#!/usr/bin/env python3
"""
CLI entry point for classification training.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    """
    Entry point for classification training CLI.
    """
    parser = argparse.ArgumentParser(
        description="Train classification models on medical imaging datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with experiment config (references model config internally)
  train-classifier config/classification/experiments/cfg_medsyn/pathmnist_real_plus_synth.yaml

  # 5-fold cross-validation
  train-classifier config/classification/experiments/only_real/pathmnist_real_only.yaml --k_folds 5

  # Different training regime
  train-classifier config/classification/experiments/real_traditional_augmentation/pathmnist_real_plus_trad_aug.yaml

  # Override config parameters
  train-classifier config/classification/experiments/distdiff/pathmnist_real_plus_synth.yaml --batch_size 128 --lr 5e-5
        """
    )

    # Positional arguments
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )

    # Training mode
    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Run test evaluation only (skip training)"
    )

    # Config overrides
    parser.add_argument(
        "--regime",
        type=str,
        choices=["real_only", "real_plus_synth", "real_plus_trad_aug"],
        help="Override training regime"
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        help="Override number of folds for k-fold CV (1 = single split)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Override maximum epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override device (e.g., 'cuda', 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name (e.g., 'resnet50', 'clip_vit_b16')"
    )

    # Logging
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Check config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 80)
    print("Classification Training")
    print("=" * 80)
    logger.info(f"Configuration file: {config_path.absolute()}")

    try:
        # Load configuration
        from medsyn.models.classification.config import load_classification_config
        config = load_classification_config(str(config_path))

        # Apply CLI overrides
        import dataclasses

        # Override data config
        data_overrides = {}
        if args.regime:
            logger.info(f"Overriding regime: {config.data.regime} -> {args.regime}")
            data_overrides["regime"] = args.regime
        if args.k_folds:
            logger.info(f"Overriding k_folds: {config.data.num_folds} -> {args.k_folds}")
            data_overrides["num_folds"] = args.k_folds
        if args.batch_size:
            logger.info(f"Overriding batch_size: {config.data.batch_size} -> {args.batch_size}")
            data_overrides["batch_size"] = args.batch_size

        if data_overrides:
            config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, **data_overrides)
            )

        # Override optimization config
        if args.lr:
            logger.info(f"Overriding learning rate: {config.optimization.lr} -> {args.lr}")
            config = dataclasses.replace(
                config,
                optimization=dataclasses.replace(config.optimization, lr=args.lr)
            )

        # Override training config
        if args.max_epochs:
            logger.info(f"Overriding max_epochs: {config.training.max_epochs} -> {args.max_epochs}")
            config = dataclasses.replace(
                config,
                training=dataclasses.replace(config.training, max_epochs=args.max_epochs)
            )

        # Override experiment config
        experiment_overrides = {}
        if args.device:
            logger.info(f"Overriding device: {config.experiment.device} -> {args.device}")
            experiment_overrides["device"] = args.device
        if args.output_dir:
            logger.info(f"Overriding output_dir: {config.experiment.output_dir} -> {args.output_dir}")
            experiment_overrides["output_dir"] = Path(args.output_dir)

        if experiment_overrides:
            config = dataclasses.replace(
                config,
                experiment=dataclasses.replace(config.experiment, **experiment_overrides)
            )

        # Override model config
        if args.model:
            logger.info(f"Overriding model name: {config.model.name} -> {args.model}")
            config = dataclasses.replace(
                config,
                model=dataclasses.replace(config.model, name=args.model)
            )

        # Print configuration summary
        print("\nExperiment Configuration:")
        print(f"  Name: {config.experiment.name}")
        print(f"  Output directory: {config.experiment.output_dir}")
        print(f"  Seed: {config.experiment.seed}")
        print(f"  Device: {config.experiment.device}")

        print("\nData Configuration:")
        print(f"  NPZ path: {config.data.npz_path}")
        print(f"  Regime: {config.data.regime}")
        print(f"  K-folds: {config.data.num_folds}")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Num workers: {config.data.num_workers}")
        print(f"  Image size: {config.data.image_size}")

        print("\nModel Configuration:")
        print(f"  Model: {config.model.name}")
        print(f"  Pretrained: {config.model.pretrained}")
        print(f"  Num classes: {config.model.num_classes}")
        print(f"  Freeze backbone: {config.model.freeze_backbone}")
        print(f"  Dropout: {config.model.dropout}")

        print("\nOptimization Configuration:")
        print(f"  Optimizer: {config.optimization.optimizer}")
        print(f"  Learning rate: {config.optimization.lr}")
        print(f"  Weight decay: {config.optimization.weight_decay}")

        print("\nTraining Configuration:")
        print(f"  Max epochs: {config.training.max_epochs}")
        print(f"  Early stopping patience: {config.training.early_stopping_patience}")
        print(f"  Mixed precision: {config.training.mixed_precision}")
        print(f"  Gradient clipping: {config.training.grad_clip_norm}")

        print("\n" + "=" * 80)

        # Instantiate classifier based on model name
        from medsyn.models.classification.engine.resnet_classifier import ResNet50Classifier

        if config.model.name == "resnet50":
            classifier = ResNet50Classifier(config)
        elif config.model.name == "clip_vit_b16":
            from medsyn.models.classification.engine.clip_classifier import CLIPViTB16Classifier
            classifier = CLIPViTB16Classifier(config)
        else:
            logger.error(f"Unknown model: {config.model.name}")
            sys.exit(1)

        # Run training or test evaluation
        if args.test_only:
            print("\nRunning test evaluation...")
            print("=" * 80 + "\n")
            test_results = classifier.test()
            print("\n" + "=" * 80)
            print("Test evaluation completed!")
            print("=" * 80)
        else:
            if config.data.num_folds > 1:
                print(f"\nStarting {config.data.num_folds}-fold cross-validation...")
            else:
                print("\nStarting training (single train/val split)...")
            print("=" * 80 + "\n")

            classifier.run_kfold_cv()

            print("\n" + "=" * 80)
            print("Training completed successfully!")
            print("=" * 80)
            logger.info(f"Results saved to: {config.experiment.output_dir}")

    except Exception as e:
        print("\n" + "=" * 80)
        print("Training failed!")
        print("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
