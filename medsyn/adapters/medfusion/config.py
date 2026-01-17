# medsyn/adapters/medfusion/config.py
"""
MedFusion configuration adapter.

Converts YAML configuration to dataclasses for MedFusion training
using PyTorch Lightning.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Mapping, Dict
import os
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class NoiseEstimatorConfig:
    """Configuration for the UNet noise estimator."""
    in_ch: int = 3
    out_ch: int = 3
    spatial_dims: int = 2
    hid_chs: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    strides: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    num_res_blocks: int = 2
    act_name: str = "swish"
    norm_name: str = "group"
    attention_levels: List[int] = field(default_factory=lambda: [1])
    dropout: float = 0.0
    use_self_conditioning: bool = False
    estimate_variance: bool = False
    num_classes: Optional[int] = None  # For conditional generation


@dataclass
class NoiseSchedulerConfig:
    """Configuration for the Gaussian noise scheduler."""
    timesteps: int = 1000
    beta_start: float = 0.002
    beta_end: float = 0.02
    schedule_strategy: str = "scaled_linear"


@dataclass
class TrainingConfig:
    """Configuration for training."""
    max_epochs: int = 200
    batch_size: int = 32
    num_workers: int = 8
    lr: float = 1e-4
    accelerator: str = "auto"
    precision: str = "32"
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 50
    sample_every_n_steps: int = 1000


@dataclass
class MedFusionConfig:
    """
    Configuration for MedFusion training and generation.

    This dataclass provides configuration for training MedFusion models
    using PyTorch Lightning with MedSyn's unified dataset format.
    """

    # Data settings
    dataset_name: str = "pathmnist"
    npz_path: str = ""
    image_size: int = 64

    # Model configuration
    noise_estimator: NoiseEstimatorConfig = field(default_factory=NoiseEstimatorConfig)
    noise_scheduler: NoiseSchedulerConfig = field(default_factory=NoiseSchedulerConfig)

    # Training configuration
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Pipeline settings
    estimator_objective: str = "x_T"  # 'x_T' or 'x_0'
    classifier_free_guidance_dropout: float = 0.5
    do_input_centering: bool = True
    clip_x0: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Output and paths
    output_dir: str = "./medfusion_output"
    checkpoint_path: Optional[str] = None

    # Generation settings
    num_samples: int = 64
    guidance_scale: float = 1.0

    # Number of classes (auto-detected from dataset if not specified)
    num_classes: int = 9

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.npz_path and not Path(self.npz_path).exists():
            logger.warning(f"NPZ path does not exist: {self.npz_path}")

        # Auto-detect num_classes from dataset_name if using default
        if self.num_classes == 9 and self.dataset_name != "pathmnist":
            from medsyn.data.registry import get_num_classes
            self.num_classes = get_num_classes(self.dataset_name)

        # Update noise_estimator num_classes
        if self.noise_estimator.num_classes is None:
            self.noise_estimator.num_classes = self.num_classes


def _expand_env(obj: Any) -> Any:
    """Recursively expand ${ENV} and $ENV in strings within mappings/lists."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _dict_to_noise_estimator(d: Dict) -> NoiseEstimatorConfig:
    """Convert dict to NoiseEstimatorConfig."""
    return NoiseEstimatorConfig(
        in_ch=d.get("in_ch", 3),
        out_ch=d.get("out_ch", 3),
        spatial_dims=d.get("spatial_dims", 2),
        hid_chs=d.get("hid_chs", [64, 128, 256, 512]),
        kernel_sizes=d.get("kernel_sizes", [3, 3, 3, 3]),
        strides=d.get("strides", [1, 2, 2, 2]),
        num_res_blocks=d.get("num_res_blocks", 2),
        act_name=d.get("act_name", "swish"),
        norm_name=d.get("norm_name", "group"),
        attention_levels=d.get("attention_levels", [1]),
        dropout=d.get("dropout", 0.0),
        use_self_conditioning=d.get("use_self_conditioning", False),
        estimate_variance=d.get("estimate_variance", False),
        num_classes=d.get("num_classes", None),
    )


def _dict_to_noise_scheduler(d: Dict) -> NoiseSchedulerConfig:
    """Convert dict to NoiseSchedulerConfig."""
    return NoiseSchedulerConfig(
        timesteps=d.get("timesteps", 1000),
        beta_start=d.get("beta_start", 0.002),
        beta_end=d.get("beta_end", 0.02),
        schedule_strategy=d.get("schedule_strategy", "scaled_linear"),
    )


def _dict_to_training(d: Dict) -> TrainingConfig:
    """Convert dict to TrainingConfig."""
    return TrainingConfig(
        max_epochs=d.get("max_epochs", 200),
        batch_size=d.get("batch_size", 32),
        num_workers=d.get("num_workers", 8),
        lr=d.get("lr", 1e-4),
        accelerator=d.get("accelerator", "auto"),
        precision=d.get("precision", "32"),
        gradient_clip_val=d.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=d.get("accumulate_grad_batches", 1),
        val_check_interval=d.get("val_check_interval", 1.0),
        check_val_every_n_epoch=d.get("check_val_every_n_epoch", 1),
        log_every_n_steps=d.get("log_every_n_steps", 50),
        sample_every_n_steps=d.get("sample_every_n_steps", 1000),
    )


def load_medfusion_config(yaml_path: str) -> MedFusionConfig:
    """
    Load MedFusion configuration from a YAML file.

    Supports environment variable expansion (${VAR} or $VAR) in YAML values.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        MedFusionConfig dataclass instance
    """
    p = Path(yaml_path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Expand environment variables
    raw = _expand_env(raw)

    # Build config
    config_kwargs = {}

    # Data section
    if "data" in raw:
        data = raw["data"]
        config_kwargs["dataset_name"] = data.get("dataset_name", "pathmnist")
        config_kwargs["npz_path"] = data.get("npz_path", "")
        config_kwargs["image_size"] = data.get("image_size", 64)

    # Model section
    if "model" in raw:
        model = raw["model"]
        if "noise_estimator" in model:
            config_kwargs["noise_estimator"] = _dict_to_noise_estimator(model["noise_estimator"])
        if "noise_scheduler" in model:
            config_kwargs["noise_scheduler"] = _dict_to_noise_scheduler(model["noise_scheduler"])

    # Training section
    if "training" in raw:
        training = raw["training"]
        config_kwargs["training"] = _dict_to_training(training)
        config_kwargs["output_dir"] = training.get("output_dir", "./medfusion_output")

    # Pipeline section
    if "pipeline" in raw:
        pipe = raw["pipeline"]
        config_kwargs["estimator_objective"] = pipe.get("estimator_objective", "x_T")
        config_kwargs["classifier_free_guidance_dropout"] = pipe.get("classifier_free_guidance_dropout", 0.5)
        config_kwargs["do_input_centering"] = pipe.get("do_input_centering", True)
        config_kwargs["clip_x0"] = pipe.get("clip_x0", True)
        config_kwargs["use_ema"] = pipe.get("use_ema", True)
        config_kwargs["ema_decay"] = pipe.get("ema_decay", 0.9999)

    # Generation section
    if "generation" in raw:
        gen = raw["generation"]
        config_kwargs["checkpoint_path"] = gen.get("checkpoint_path", None)
        config_kwargs["num_samples"] = gen.get("num_samples", 64)
        config_kwargs["guidance_scale"] = gen.get("guidance_scale", 1.0)

    logger.info("Loaded MedFusion config from %s", yaml_path)
    return MedFusionConfig(**config_kwargs)
