# medsyn/adapters/distdiff/config.py
"""
DistDiff configuration adapter.

Converts YAML configuration to dataclasses that can be used
for DistDiff guide model training and synthetic image generation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Mapping
import os
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DistDiffTrainConfig:
    """
    Configuration for DistDiff guide model training.

    This dataclass provides configuration via YAML files with
    environment variable expansion, replacing argparse flags.
    """

    # Data settings
    dataset_name: str = "pathmnist"
    npz_path: str = ""
    image_size: int = 64

    # Model architecture
    arch: str = "open_clip_vit_b32"  # or resnet50, resnext50, mobilenetv2, wideresnet50
    pretrained: bool = True

    # Training settings
    epochs: int = 100
    train_batch_size: int = 128
    val_batch_size: int = 100
    lr: float = 1e-4  # Use lower lr for ViT models
    momentum: float = 0.9
    weight_decay: float = 1e-4
    dropout: float = 0.0
    train_fc: bool = False  # Only train final fc layer

    # Learning rate schedule
    schedule: List[int] = field(default_factory=lambda: [41, 81])
    gamma: float = 0.1

    # Output and logging
    output_dir: str = "./distdiff_guide_output"

    # Hardware
    num_workers: int = 4
    seed: Optional[int] = 42

    # Cache directory for pretrained weights (for offline HPC)
    cache_dir: Optional[str] = None

    # Number of classes (auto-detected from dataset if not specified)
    num_classes: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.npz_path and not Path(self.npz_path).exists():
            logger.warning(f"NPZ path does not exist: {self.npz_path}")

        # Auto-detect num_classes from dataset_name if not specified
        if self.num_classes is None:
            from medsyn.data.registry import get_num_classes
            try:
                self.num_classes = get_num_classes(self.dataset_name)
            except KeyError:
                logger.warning(f"Unknown dataset: {self.dataset_name}, num_classes not auto-detected")


@dataclass
class DistDiffGenerateConfig:
    """
    Configuration for DistDiff synthetic image generation.

    This dataclass provides configuration for Stable Diffusion-based
    generation with prototype-guided sampling.
    """

    # Data settings
    dataset_name: str = "pathmnist"
    npz_path: str = ""
    image_size: int = 64

    # Guide model settings
    guide_model_arch: str = "open_clip_vit_b32"
    guide_model_path: str = ""  # Path to trained guide model checkpoint

    # Stable Diffusion settings
    pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
    resolution: int = 512
    guidance_scale: float = 7.5

    # Guidance settings
    guidance_type: str = "transform_guidance"  # or "direct_guidance"
    optimize_targets: str = "global_prototype-local_prototype"  # or just "global_prototype"
    K: int = 3  # Number of local prototypes per class
    rho: float = 10.0  # Learning rate for optimization
    gs: float = 1.0  # Scale for global score
    ls: float = 1.0  # Scale for local score
    constraint_value: float = 0.8
    guidance_step: int = 1  # Index for start guidance
    guidance_period: int = 1  # Guidance period
    strength: float = 0.9  # Noising/unnoising strength

    # Generation settings
    num_images_per_sample: int = 4
    steps: int = 50
    seed: Optional[int] = 42

    # Split settings (for parallel generation)
    total_split: int = 1
    split: int = 0

    # Output settings
    output_dir: str = "./distdiff_synthetic"

    # Hardware
    train_batch_size: int = 2
    val_batch_size: int = 1
    num_workers: int = 0
    mixed_precision: Optional[str] = None  # "fp16", "bf16", or None

    # Cache directory for models (for offline HPC)
    cache_dir: Optional[str] = None
    local_files_only: bool = False

    # Language enhancement
    language_enhance: bool = False

    # Number of classes (auto-detected from dataset if not specified)
    num_classes: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.npz_path and not Path(self.npz_path).exists():
            logger.warning(f"NPZ path does not exist: {self.npz_path}")

        if self.guide_model_path and not Path(self.guide_model_path).exists():
            logger.warning(f"Guide model path does not exist: {self.guide_model_path}")

        # Auto-detect num_classes from dataset_name if not specified
        if self.num_classes is None:
            from medsyn.data.registry import get_num_classes
            try:
                self.num_classes = get_num_classes(self.dataset_name)
            except KeyError:
                logger.warning(f"Unknown dataset: {self.dataset_name}, num_classes not auto-detected")


@dataclass
class DistDiffConfig:
    """
    Combined configuration for DistDiff (both training and generation).

    This unified config can be loaded from a single YAML file that contains
    both guide model training and generation settings.
    """

    train: DistDiffTrainConfig = field(default_factory=DistDiffTrainConfig)
    generate: DistDiffGenerateConfig = field(default_factory=DistDiffGenerateConfig)


def _expand_env(obj: Any) -> Any:
    """
    Recursively expand ${ENV} and $ENV in strings within mappings/lists.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def load_distdiff_train_config(yaml_path: str) -> DistDiffTrainConfig:
    """
    Load DistDiff guide model training configuration from a YAML file.

    Supports environment variable expansion (${VAR} or $VAR) in YAML values.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        DistDiffTrainConfig dataclass instance
    """
    p = Path(yaml_path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Expand environment variables
    raw = _expand_env(raw)

    # Flatten nested config structure
    flat = {}

    # Data section
    if "data" in raw:
        data = raw["data"]
        flat["dataset_name"] = data.get("dataset_name", "pathmnist")
        flat["npz_path"] = data.get("npz_path", "")
        flat["image_size"] = data.get("image_size", 64)
        flat["num_workers"] = data.get("num_workers", 4)

    # Model section
    if "model" in raw:
        model = raw["model"]
        flat["arch"] = model.get("arch", "open_clip_vit_b32")
        flat["pretrained"] = model.get("pretrained", True)

    # Training section
    if "training" in raw:
        train = raw["training"]
        flat["epochs"] = train.get("epochs", 100)
        flat["train_batch_size"] = train.get("train_batch_size", 128)
        flat["val_batch_size"] = train.get("val_batch_size", 100)
        flat["lr"] = train.get("lr", 1e-4)
        flat["momentum"] = train.get("momentum", 0.9)
        flat["weight_decay"] = train.get("weight_decay", 1e-4)
        flat["dropout"] = train.get("dropout", 0.0)
        flat["train_fc"] = train.get("train_fc", False)
        flat["schedule"] = train.get("schedule", [41, 81])
        flat["gamma"] = train.get("gamma", 0.1)
        flat["output_dir"] = train.get("output_dir", "./distdiff_guide_output")
        flat["seed"] = train.get("seed", 42)
        flat["num_classes"] = train.get("num_classes", None)

    # Hardware section
    if "hardware" in raw:
        hw = raw["hardware"]
        flat["cache_dir"] = hw.get("cache_dir", None)

    logger.info("Loaded DistDiff train config from %s", yaml_path)
    return DistDiffTrainConfig(**flat)


def load_distdiff_generate_config(yaml_path: str) -> DistDiffGenerateConfig:
    """
    Load DistDiff generation configuration from a YAML file.

    Supports environment variable expansion (${VAR} or $VAR) in YAML values.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        DistDiffGenerateConfig dataclass instance
    """
    p = Path(yaml_path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Expand environment variables
    raw = _expand_env(raw)

    # Flatten nested config structure
    flat = {}

    # Data section
    if "data" in raw:
        data = raw["data"]
        flat["dataset_name"] = data.get("dataset_name", "pathmnist")
        flat["npz_path"] = data.get("npz_path", "")
        flat["image_size"] = data.get("image_size", 64)
        flat["num_workers"] = data.get("num_workers", 0)

    # Guide model section
    if "guide_model" in raw:
        guide = raw["guide_model"]
        flat["guide_model_arch"] = guide.get("arch", "open_clip_vit_b32")
        flat["guide_model_path"] = guide.get("checkpoint_path", "")

    # Stable Diffusion section
    if "stable_diffusion" in raw:
        sd = raw["stable_diffusion"]
        flat["pretrained_model_name_or_path"] = sd.get("pretrained_model_name_or_path", "CompVis/stable-diffusion-v1-4")
        flat["resolution"] = sd.get("resolution", 512)
        flat["guidance_scale"] = sd.get("guidance_scale", 7.5)

    # Guidance section
    if "guidance" in raw:
        guidance = raw["guidance"]
        flat["guidance_type"] = guidance.get("type", "transform_guidance")
        flat["optimize_targets"] = guidance.get("optimize_targets", "global_prototype-local_prototype")
        flat["K"] = guidance.get("K", 3)
        flat["rho"] = guidance.get("rho", 10.0)
        flat["gs"] = guidance.get("gs", 1.0)
        flat["ls"] = guidance.get("ls", 1.0)
        flat["constraint_value"] = guidance.get("constraint_value", 0.8)
        flat["guidance_step"] = guidance.get("guidance_step", 1)
        flat["guidance_period"] = guidance.get("guidance_period", 1)
        flat["strength"] = guidance.get("strength", 0.9)

    # Generation section
    if "generation" in raw:
        gen = raw["generation"]
        flat["num_images_per_sample"] = gen.get("num_images_per_sample", 4)
        flat["steps"] = gen.get("steps", 50)
        flat["seed"] = gen.get("seed", 42)
        flat["total_split"] = gen.get("total_split", 1)
        flat["split"] = gen.get("split", 0)
        flat["output_dir"] = gen.get("output_dir", "./distdiff_synthetic")
        flat["num_classes"] = gen.get("num_classes", None)

    # Hardware section
    if "hardware" in raw:
        hw = raw["hardware"]
        flat["train_batch_size"] = hw.get("train_batch_size", 2)
        flat["val_batch_size"] = hw.get("val_batch_size", 1)
        flat["mixed_precision"] = hw.get("mixed_precision", None)
        flat["cache_dir"] = hw.get("cache_dir", None)
        flat["local_files_only"] = hw.get("local_files_only", False)

    # Language enhancement
    if "language_enhance" in raw:
        flat["language_enhance"] = raw.get("language_enhance", False)

    logger.info("Loaded DistDiff generate config from %s", yaml_path)
    return DistDiffGenerateConfig(**flat)


def load_distdiff_config(yaml_path: str) -> DistDiffConfig:
    """
    Load combined DistDiff configuration from a YAML file.

    The YAML file can contain both training and generation sections.
    Sections that are not present will use default values.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        DistDiffConfig dataclass instance with train and generate configs
    """
    p = Path(yaml_path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    # Expand environment variables
    raw = _expand_env(raw)

    # Build train config if training section exists
    train_config = DistDiffTrainConfig()
    if "training" in raw or "data" in raw:
        train_config = load_distdiff_train_config(yaml_path)

    # Build generate config if generation section exists
    generate_config = DistDiffGenerateConfig()
    if "generation" in raw or "guidance" in raw:
        generate_config = load_distdiff_generate_config(yaml_path)

    logger.info("Loaded combined DistDiff config from %s", yaml_path)
    return DistDiffConfig(train=train_config, generate=generate_config)
