"""
Base classifier abstract class for classification models.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import shutil
import yaml
import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from ..config import ClassificationConfig
from ..dataloaders import build_classification_dataloader
from ..metrics import ClassificationMetrics, compute_per_class_metrics
from ..logging_utils import (
    CSVLogger, PerClassCSVLogger, BestEpochMetricsSaver,
    save_checkpoint, load_checkpoint
)
from ..kfold_utils import StratifiedKFoldSplitter
from ..cv_aggregator import CrossValidationAggregator

logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """
    Abstract base class for classification models.

    Provides common training loop, validation, checkpointing, and k-fold CV orchestration.
    Subclasses implement model-specific components:
    - build_model(): Construct the model architecture
    - build_optimizer(): Create optimizer with model-specific parameter groups
    """

    def __init__(self, config: ClassificationConfig):
        """
        Initialize classifier with configuration.

        Args:
            config: ClassificationConfig instance
        """
        self.config = config
        self.device = torch.device(config.experiment.device)

        # Setup output directories
        self.output_dir = config.experiment.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ckpts").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Save config backup
        config_backup = self.output_dir / "config_backup.yaml"
        if not config_backup.exists():
            self._save_config_backup(config_backup)

        # Set random seed
        self._set_seed(config.experiment.seed)

        # Initialize components (to be built later)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.augmentation_pipeline: Optional[Any] = None

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

        # Metrics tracking (with per-class support)
        self.train_metrics = ClassificationMetrics(num_classes=config.model.num_classes)
        self.val_metrics = ClassificationMetrics(num_classes=config.model.num_classes)

        # CSV loggers
        self.csv_logger: Optional[CSVLogger] = None
        self.train_per_class_logger: Optional[PerClassCSVLogger] = None
        self.val_per_class_logger: Optional[PerClassCSVLogger] = None
        self.best_epoch_saver: Optional[BestEpochMetricsSaver] = None

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _save_config_backup(self, path: Path):
        """Save configuration to YAML file."""
        # Convert dataclass config to dict
        config_dict = dataclasses.asdict(self.config)

        # Convert Path objects to strings
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        config_dict = convert_paths(config_dict)

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

        logger.info(f"Configuration backup saved to: {path}")

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Build and return the model architecture.

        Returns:
            nn.Module: The classification model
        """
        pass

    @abstractmethod
    def build_optimizer(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Build optimizer and optionally learning rate scheduler.

        Returns:
            Tuple of (optimizer, scheduler). Scheduler can be None.
        """
        pass

    def setup(self):
        """
        Setup model, optimizer, scheduler, and augmentation pipeline.
        Called before training.
        """
        logger.info("Setting up model...")
        self.model = self.build_model()
        self.model = self.model.to(self.device)

        logger.info(f"Model architecture: {self.config.model.name}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        logger.info("Setting up optimizer and scheduler...")
        self.optimizer, self.scheduler = self.build_optimizer()

        # Setup mixed precision training
        if self.config.training.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")

        # Setup augmentation pipeline for real_plus_trad_aug regime
        if self.config.data.regime == "real_plus_trad_aug" and self.config.augmentation.enabled:
            logger.info("Setting up augmentation pipeline...")
            from medsyn.models.ccDDPM.augmentation.transforms import AugmentationPipeline
            from medsyn.models.ccDDPM.augmentation.config import AugmentationConfig

            # Convert AugmentationCfg to ccDDPM AugmentationConfig
            aug_config = AugmentationConfig(
                enabled=self.config.augmentation.enabled,
                probability=self.config.augmentation.probability,
                transforms=self.config.augmentation.transforms,
                preserve_range=self.config.augmentation.preserve_range,
                normalize_after_augment=self.config.augmentation.normalize_after_augment,
                statistics={"enabled": False}  # Disable statistics tracking
            )
            self.augmentation_pipeline = AugmentationPipeline(aug_config)

        # Setup CSV loggers
        log_file = self.output_dir / "logs" / "training_log.csv"
        self.csv_logger = CSVLogger(log_file)

        # Setup per-class CSV loggers
        train_per_class_log = self.output_dir / "logs" / "train_per_class_metrics.csv"
        self.train_per_class_logger = PerClassCSVLogger(
            train_per_class_log,
            num_classes=self.config.model.num_classes,
            split="train"
        )

        val_per_class_log = self.output_dir / "logs" / "val_per_class_metrics.csv"
        self.val_per_class_logger = PerClassCSVLogger(
            val_per_class_log,
            num_classes=self.config.model.num_classes,
            split="val"
        )

        # Setup best epoch metrics saver
        best_epoch_path = self.output_dir / "logs" / "best_epoch_metrics.yaml"
        self.best_epoch_saver = BestEpochMetricsSaver(
            best_epoch_path,
            num_classes=self.config.model.num_classes
        )

    def build_dataloaders(
        self,
        fold_train_data: Optional[Dict[str, np.ndarray]] = None,
        fold_val_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Build train and validation dataloaders.

        Args:
            fold_train_data: Optional fold training data (for k-fold CV)
            fold_val_data: Optional fold validation data (for k-fold CV)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Extract fold data if provided
        train_fold_images = fold_train_data["images"] if fold_train_data else None
        train_fold_labels = fold_train_data["labels"] if fold_train_data else None
        train_fold_is_synth = fold_train_data["is_synth"] if fold_train_data else None

        val_fold_images = fold_val_data["images"] if fold_val_data else None
        val_fold_labels = fold_val_data["labels"] if fold_val_data else None
        val_fold_is_synth = fold_val_data["is_synth"] if fold_val_data else None

        train_loader = build_classification_dataloader(
            npz_path=self.config.data.npz_path,
            split="train",
            regime=self.config.data.regime,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            augmentation_pipeline=self.augmentation_pipeline,
            fold_images=train_fold_images,
            fold_labels=train_fold_labels,
            fold_is_synth=train_fold_is_synth
        )

        val_loader = build_classification_dataloader(
            npz_path=self.config.data.npz_path,
            split="val",
            regime=self.config.data.regime,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            augmentation_pipeline=None,  # No augmentation during validation
            fold_images=val_fold_images,
            fold_labels=val_fold_labels,
            fold_is_synth=val_fold_is_synth
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run one training epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config.training.max_epochs} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            images = batch["img"].to(self.device)
            labels = batch["cls"].to(self.device)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logits = self.model(images)
                loss = nn.functional.cross_entropy(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.config.training.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.training.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip_norm
                    )
                self.optimizer.step()

            # Update metrics
            self.train_metrics.update(loss.item(), logits, labels)

            # Update progress bar
            if batch_idx % self.config.training.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{self.train_metrics.accuracy():.4f}"
                })

        return self.train_metrics.compute()

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run one validation epoch.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{self.config.training.max_epochs} [Val]", leave=False)

        with torch.no_grad():
            for batch in pbar:
                images = batch["img"].to(self.device)
                labels = batch["cls"].to(self.device)

                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    logits = self.model(images)
                    loss = nn.functional.cross_entropy(logits, labels)

                # Update metrics
                self.val_metrics.update(loss.item(), logits, labels)

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{self.val_metrics.accuracy():.4f}"
                })

        return self.val_metrics.compute()

    def fit(self):
        """
        Main training loop.
        Trains the model for the specified number of epochs.
        """
        logger.info("Starting training...")
        logger.info(f"Epochs: {self.config.training.max_epochs}")
        logger.info(f"Regime: {self.config.data.regime}")
        logger.info(f"Batch size: {self.config.data.batch_size}")
        logger.info(f"Output directory: {self.output_dir}")

        # Setup model, optimizer, scheduler
        self.setup()

        # Build dataloaders
        train_loader, val_loader = self.build_dataloaders()

        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Number of classes: {self.config.model.num_classes}")

        # Training loop
        for epoch in range(1, self.config.training.max_epochs + 1):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate epoch
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Step scheduler (epoch-level schedulers)
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Log to CSV
            self.csv_logger.log_epoch(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr
            )

            # Log per-class metrics
            if "per_class" in train_metrics:
                self.train_per_class_logger.log_epoch(epoch, train_metrics["per_class"])
            if "per_class" in val_metrics:
                self.val_per_class_logger.log_epoch(epoch, val_metrics["per_class"])

            # Save checkpoint
            val_acc = val_metrics["accuracy"]
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0

                # Save best checkpoint
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=self.output_dir / "ckpts" / "best_model.pt"
                )

                # Save best epoch metrics
                self.best_epoch_saver.update(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics
                )
                self.best_epoch_saver.save()

                logger.info(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.epochs_no_improve += 1

            # Periodic checkpoint
            if epoch % self.config.training.checkpoint_every == 0:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=self.output_dir / "ckpts" / f"checkpoint_epoch_{epoch:03d}.pt"
                )

            # Early stopping
            if (self.config.training.early_stopping_patience > 0 and
                self.epochs_no_improve >= self.config.training.early_stopping_patience):
                logger.info(
                    f"Early stopping triggered! No improvement for "
                    f"{self.config.training.early_stopping_patience} epochs."
                )
                break

        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch}")

    def test(self) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Returns:
            Dictionary of test metrics
        """
        logger.info("Evaluating on test set...")

        # Build test loader
        test_loader = build_classification_dataloader(
            npz_path=self.config.data.npz_path,
            split="test",
            regime=self.config.data.regime,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            image_size=self.config.data.image_size,
            augmentation_pipeline=None  # No augmentation during testing
        )

        # Load best checkpoint
        best_ckpt = self.output_dir / "ckpts" / "best_model.pt"
        if best_ckpt.exists():
            load_checkpoint(best_ckpt, self.model, self.optimizer, self.scheduler)
            logger.info(f"Loaded best checkpoint from: {best_ckpt}")
        else:
            logger.warning("No best checkpoint found, using current model weights")

        # Evaluate
        self.model.eval()
        test_metrics = ClassificationMetrics(num_classes=self.config.model.num_classes)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch["img"].to(self.device)
                labels = batch["cls"].to(self.device)

                logits = self.model(images)
                loss = nn.functional.cross_entropy(logits, labels)

                test_metrics.update(loss.item(), logits, labels)

        # Compute final metrics (includes per-class metrics)
        test_results = test_metrics.compute()

        # Save test results
        test_results_path = self.output_dir / "logs" / "test_results.yaml"
        with open(test_results_path, "w") as f:
            yaml.safe_dump(test_results, f, default_flow_style=False)

        logger.info(f"Test results saved to: {test_results_path}")
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"Test Loss: {test_results['loss']:.4f}")

        # Log per-class metrics summary
        if "per_class" in test_results:
            logger.info("\nPer-Class Test Metrics:")
            per_class = test_results["per_class"]
            for class_idx in range(self.config.model.num_classes):
                f1 = per_class["f1"][class_idx] if class_idx < len(per_class["f1"]) else 0.0
                auc = per_class["auc"][class_idx] if class_idx < len(per_class["auc"]) else 0.0
                logger.info(f"  Class {class_idx}: F1={f1:.4f}, AUC={auc:.4f}")

        return test_results

    def run_kfold_cv(self):
        """
        Run k-fold cross-validation.
        Trains the model on each fold and aggregates results.
        """
        k_folds = self.config.data.num_folds
        if k_folds == 1:
            logger.info("Single train/val split (no cross-validation)")
            self.fit()
            return

        logger.info(f"Starting {k_folds}-fold cross-validation...")

        # Load data for k-fold splitting
        with np.load(self.config.data.npz_path) as data:
            train_images = data["train_images"]
            train_labels = data["train_labels"]
            train_is_synth = data.get("train_is_synth", np.zeros(len(train_labels), dtype=bool))

            val_images = data["val_images"]
            val_labels = data["val_labels"]
            val_is_synth = data.get("val_is_synth", np.zeros(len(val_labels), dtype=bool))

        # Create k-fold splits
        splitter = StratifiedKFoldSplitter(
            k=k_folds,
            seed=self.config.experiment.seed,
            shuffle=True
        )

        folds = splitter.create_folds(
            train_images, train_labels,
            val_images, val_labels,
            train_is_synth, val_is_synth
        )

        logger.info(splitter.get_fold_info(folds))

        # Train on each fold
        fold_dirs = []
        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            logger.info(f"\n{'='*80}")
            logger.info(f"Training Fold {fold_idx + 1}/{k_folds}")
            logger.info(f"{'='*80}\n")

            # Create fold-specific output directory
            fold_output_dir = self.output_dir / f"fold_{fold_idx}"
            fold_output_dir.mkdir(exist_ok=True)

            # Create new classifier instance for this fold
            fold_config = dataclasses.replace(
                self.config,
                experiment=dataclasses.replace(
                    self.config.experiment,
                    output_dir=fold_output_dir
                )
            )

            fold_classifier = type(self)(fold_config)
            fold_classifier.setup()

            # Build fold-specific dataloaders
            train_loader, val_loader = fold_classifier.build_dataloaders(
                fold_train_data=train_fold,
                fold_val_data=val_fold
            )

            # Train on this fold
            for epoch in range(1, fold_config.training.max_epochs + 1):
                fold_classifier.current_epoch = epoch

                train_metrics = fold_classifier.train_epoch(train_loader, epoch)
                val_metrics = fold_classifier.validate_epoch(val_loader, epoch)

                # Step scheduler
                if fold_classifier.scheduler is not None:
                    fold_classifier.scheduler.step()

                current_lr = fold_classifier.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Fold {fold_idx + 1} | Epoch {epoch:03d} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                )

                fold_classifier.csv_logger.log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=current_lr
                )

                # Log per-class metrics
                if "per_class" in train_metrics:
                    fold_classifier.train_per_class_logger.log_epoch(epoch, train_metrics["per_class"])
                if "per_class" in val_metrics:
                    fold_classifier.val_per_class_logger.log_epoch(epoch, val_metrics["per_class"])

                # Save best checkpoint for this fold
                val_acc = val_metrics["accuracy"]
                if val_acc > fold_classifier.best_val_acc:
                    fold_classifier.best_val_acc = val_acc
                    fold_classifier.best_epoch = epoch
                    save_checkpoint(
                        model=fold_classifier.model,
                        optimizer=fold_classifier.optimizer,
                        scheduler=fold_classifier.scheduler,
                        epoch=epoch,
                        metrics=val_metrics,
                        path=fold_output_dir / "ckpts" / "best_model.pt"
                    )

                    # Save best epoch metrics for this fold
                    fold_classifier.best_epoch_saver.update(
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics
                    )
                    fold_classifier.best_epoch_saver.save()

            fold_dirs.append(fold_output_dir)
            logger.info(f"Fold {fold_idx + 1} completed! Best Val Acc: {fold_classifier.best_val_acc:.4f}")

        # Aggregate results across folds
        logger.info(f"\n{'='*80}")
        logger.info("Aggregating cross-validation results...")
        logger.info(f"{'='*80}\n")

        aggregator = CrossValidationAggregator(
            fold_dirs=fold_dirs,
            output_dir=self.output_dir,
            k_folds=k_folds
        )
        aggregator.aggregate_and_save()

        logger.info("Cross-validation completed!")
