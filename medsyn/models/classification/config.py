"""
Configuration dataclasses and YAML loading for classification training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Dict, Any, List
import yaml
import os
import logging

logger = logging.getLogger(__name__)

# Training regime types
TrainingRegime = Literal["real_only", "real_plus_synth", "real_plus_trad_aug"]


@dataclass(frozen=True)
class ExperimentCfg:
    """Experiment metadata and output configuration."""
    name: str = "classification_exp"
    output_dir: Path = Path("experiments/classification")
    seed: int = 42
    device: str = "cuda"  # "cuda", "cpu", or device index like "cuda:0"


@dataclass(frozen=True)
class DataCfg:
    """Data loading and regime configuration."""
    npz_path: Path  # Required
    regime: TrainingRegime = "real_only"
    num_folds: int = 5  # K-fold CV, 1 = single train/val split
    num_workers: int = 4
    batch_size: int = 64
    image_size: int = 64  # Expected image size (H, W)


@dataclass
class AugmentationCfg:
    """Traditional augmentation config (for real_plus_trad_aug regime)."""
    enabled: bool = False
    probability: float = 0.5
    transforms: List[Dict[str, Any]] = field(default_factory=list)
    preserve_range: bool = True
    normalize_after_augment: bool = True


@dataclass(frozen=True)
class ModelCfg:
    """Model architecture configuration."""
    name: str = "resnet50"  # ["resnet50", "clip_vit_b16", "dino_v2", ...]
    pretrained: bool = True
    num_classes: int = 9
    freeze_backbone: bool = False  # Freeze backbone, only train classifier head
    dropout: float = 0.0  # Dropout probability before final classifier


@dataclass(frozen=True)
class OptimizerCfg:
    """Optimizer configuration."""
    optimizer: str = "adamw"  # ["adamw", "adam", "sgd"]
    lr: float = 1e-4
    weight_decay: float = 1e-2
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW


@dataclass(frozen=True)
class SchedulerCfg:
    """Learning rate scheduler configuration."""
    scheduler: str = "cosine"  # ["cosine", "step", "constant", "onecycle"]
    warmup_epochs: int = 5
    eta_min: float = 1e-7  # Minimum LR for cosine
    step_size: int = 30  # For step scheduler
    gamma: float = 0.1  # LR decay factor for step


@dataclass(frozen=True)
class TrainingCfg:
    """Training loop configuration."""
    max_epochs: int = 100
    early_stopping_patience: int = 15  # 0 = disabled
    grad_clip_norm: float = 0.0  # 0 = disabled
    mixed_precision: bool = True  # Use torch.cuda.amp
    checkpoint_every: int = 10  # Save checkpoint every N epochs
    log_every: int = 10  # Log every N batches


@dataclass
class ClassificationConfig:
    """Top-level configuration for classification training."""
    experiment: ExperimentCfg
    data: DataCfg
    model: ModelCfg
    optimization: OptimizerCfg
    scheduler: SchedulerCfg
    training: TrainingCfg
    augmentation: AugmentationCfg = field(default_factory=lambda: AugmentationCfg(enabled=False))


def _expand_env(obj):
    """Recursively expand environment variables in strings."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    return obj


def load_classification_config(config_path: str | Path) -> ClassificationConfig:
    """
    Load classification configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ClassificationConfig instance with all settings

    Raises:
        ValueError: If required fields are missing or invalid
        FileNotFoundError: If config file does not exist
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Expand environment variables
    raw = _expand_env(raw)

    # Parse experiment config
    exp_dict = raw.get("experiment", {})
    experiment = ExperimentCfg(
        name=exp_dict.get("name", "classification_exp"),
        output_dir=Path(exp_dict.get("output_dir", "experiments/classification")).expanduser().resolve(),
        seed=exp_dict.get("seed", 42),
        device=exp_dict.get("device", "cuda")
    )

    # Parse data config
    data_dict = raw.get("data", {})
    if "npz_path" not in data_dict:
        raise ValueError("data.npz_path is required in config")

    data = DataCfg(
        npz_path=Path(data_dict["npz_path"]).expanduser().resolve(),
        regime=data_dict.get("regime", "real_only"),
        num_folds=data_dict.get("num_folds", 5),
        num_workers=data_dict.get("num_workers", 4),
        batch_size=data_dict.get("batch_size", 64),
        image_size=data_dict.get("image_size", 64)
    )

    # Validate regime
    valid_regimes = ["real_only", "real_plus_synth", "real_plus_trad_aug"]
    if data.regime not in valid_regimes:
        raise ValueError(
            f"Invalid regime: {data.regime}. Must be one of: {', '.join(valid_regimes)}"
        )

    # Validate npz_path exists
    if not data.npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {data.npz_path}")

    # Parse model config
    model_dict = raw.get("model", {})
    model = ModelCfg(
        name=model_dict.get("name", "resnet50"),
        pretrained=model_dict.get("pretrained", True),
        num_classes=model_dict.get("num_classes", 9),
        freeze_backbone=model_dict.get("freeze_backbone", False),
        dropout=model_dict.get("dropout", 0.0)
    )

    # Parse optimization config
    opt_dict = raw.get("optimization", {})
    betas_list = opt_dict.get("betas", [0.9, 0.999])
    optimization = OptimizerCfg(
        optimizer=opt_dict.get("optimizer", "adamw"),
        lr=opt_dict.get("lr", 1e-4),
        weight_decay=opt_dict.get("weight_decay", 1e-2),
        momentum=opt_dict.get("momentum", 0.9),
        betas=tuple(betas_list)
    )

    # Parse scheduler config
    sched_dict = raw.get("scheduler", {})
    scheduler = SchedulerCfg(
        scheduler=sched_dict.get("scheduler", "cosine"),
        warmup_epochs=sched_dict.get("warmup_epochs", 5),
        eta_min=sched_dict.get("eta_min", 1e-7),
        step_size=sched_dict.get("step_size", 30),
        gamma=sched_dict.get("gamma", 0.1)
    )

    # Parse training config
    train_dict = raw.get("training", {})
    training = TrainingCfg(
        max_epochs=train_dict.get("max_epochs", 100),
        early_stopping_patience=train_dict.get("early_stopping_patience", 15),
        grad_clip_norm=train_dict.get("grad_clip_norm", 0.0),
        mixed_precision=train_dict.get("mixed_precision", True),
        checkpoint_every=train_dict.get("checkpoint_every", 10),
        log_every=train_dict.get("log_every", 10)
    )

    # Parse augmentation config (for real_plus_trad_aug regime)
    aug_dict = raw.get("augmentation", {})
    augmentation = AugmentationCfg(
        enabled=aug_dict.get("enabled", False),
        probability=aug_dict.get("probability", 0.5),
        transforms=aug_dict.get("transforms", []),
        preserve_range=aug_dict.get("preserve_range", True),
        normalize_after_augment=aug_dict.get("normalize_after_augment", True)
    )

    # Validate: real_plus_trad_aug REQUIRES augmentation.enabled=True
    if data.regime == "real_plus_trad_aug":
        if not augmentation.enabled:
            logger.warning(
                "regime=real_plus_trad_aug but augmentation.enabled=False. "
                "Enabling augmentation automatically."
            )
            augmentation.enabled = True
        if not augmentation.transforms:
            logger.warning(
                "regime=real_plus_trad_aug but no augmentation transforms specified. "
                "Please add transforms in the augmentation section."
            )

    # Validate: other regimes should NOT use augmentation
    if data.regime in ["real_only", "real_plus_synth"] and augmentation.enabled:
        logger.warning(
            f"regime={data.regime} but augmentation.enabled=True. "
            "Augmentation will be applied during training."
        )

    config = ClassificationConfig(
        experiment=experiment,
        data=data,
        model=model,
        optimization=optimization,
        scheduler=scheduler,
        training=training,
        augmentation=augmentation
    )

    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Regime: {config.data.regime}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Device: {config.experiment.device}")

    return config
