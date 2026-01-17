# medsyn/adapters/cbdm/config.py
"""
CBDM configuration adapter.

Converts YAML configuration to a dataclass that can be used
in place of ABSL flags for CBDM training and generation.
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
class CBDMConfig:
    """
    Configuration for CBDM training and generation.

    This dataclass replaces ABSL FLAGS for CBDM, allowing configuration
    via YAML files with environment variable expansion.
    """

    # Data settings
    dataset_name: str = "pathmnist"
    npz_path: str = ""
    image_size: int = 64

    # Model architecture
    ch: int = 128
    ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    attn: List[int] = field(default_factory=lambda: [1])
    num_res_blocks: int = 2
    dropout: float = 0.1

    # Diffusion settings
    T: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 0.02

    # Training settings
    lr: float = 2e-4
    total_steps: int = 200000
    batch_size: int = 64
    warmup: int = 5000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0

    # CBDM-specific settings
    cb: bool = True  # Enable class-balancing
    tau: float = 1.0  # Class-balancing temperature
    cfg: bool = True  # Enable classifier-free guidance
    finetune: bool = False  # Finetune mode

    # Number of classes (auto-detected from dataset if not specified)
    num_classes: int = 9

    # Output and logging
    output_dir: str = "./cbdm_output"
    log_interval: int = 100
    sample_interval: int = 10000
    save_interval: int = 50000
    num_samples: int = 64

    # Generation settings
    omega: float = 0.0  # Classifier-free guidance strength
    method: str = "cfg"  # 'cfg' or 'uncond'

    # Checkpoint
    checkpoint_path: Optional[str] = None

    # Hardware
    device: str = "cuda"
    num_workers: int = 4

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.npz_path and not Path(self.npz_path).exists():
            logger.warning(f"NPZ path does not exist: {self.npz_path}")

        # Auto-detect num_classes from dataset_name if using default
        if self.num_classes == 9 and self.dataset_name != "pathmnist":
            from medsyn.data.registry import get_num_classes
            self.num_classes = get_num_classes(self.dataset_name)


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


def load_cbdm_config(yaml_path: str) -> CBDMConfig:
    """
    Load CBDM configuration from a YAML file.

    Supports environment variable expansion (${VAR} or $VAR) in YAML values.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        CBDMConfig dataclass instance

    Example YAML:
        data:
          dataset_name: pathmnist
          npz_path: "${DATASET_PATH}/pathmnist_64.npz"
          image_size: 64

        model:
          ch: 128
          ch_mult: [1, 2, 2, 2]
          num_res_blocks: 2
          dropout: 0.1

        diffusion:
          T: 1000
          beta_1: 1.0e-4
          beta_T: 0.02

        training:
          lr: 2.0e-4
          total_steps: 200000
          batch_size: 64
          warmup: 5000
          ema_decay: 0.9999
          grad_clip: 1.0
          cb: true
          tau: 1.0
          output_dir: "${OUTPUT_DIR}"
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
        flat["ch"] = model.get("ch", 128)
        flat["ch_mult"] = model.get("ch_mult", [1, 2, 2, 2])
        flat["attn"] = model.get("attn", [1])
        flat["num_res_blocks"] = model.get("num_res_blocks", 2)
        flat["dropout"] = model.get("dropout", 0.1)

    # Diffusion section
    if "diffusion" in raw:
        diff = raw["diffusion"]
        flat["T"] = diff.get("T", 1000)
        flat["beta_1"] = diff.get("beta_1", 1e-4)
        flat["beta_T"] = diff.get("beta_T", 0.02)

    # Training section
    if "training" in raw:
        train = raw["training"]
        flat["lr"] = train.get("lr", 2e-4)
        flat["total_steps"] = train.get("total_steps", 200000)
        flat["batch_size"] = train.get("batch_size", 64)
        flat["warmup"] = train.get("warmup", 5000)
        flat["ema_decay"] = train.get("ema_decay", 0.9999)
        flat["grad_clip"] = train.get("grad_clip", 1.0)
        flat["cb"] = train.get("cb", True)
        flat["tau"] = train.get("tau", 1.0)
        flat["cfg"] = train.get("cfg", True)
        flat["finetune"] = train.get("finetune", False)
        flat["output_dir"] = train.get("output_dir", "./cbdm_output")
        flat["log_interval"] = train.get("log_interval", 100)
        flat["sample_interval"] = train.get("sample_interval", 10000)
        flat["save_interval"] = train.get("save_interval", 50000)
        flat["num_classes"] = train.get("num_classes", 9)

    # Generation section
    if "generation" in raw:
        gen = raw["generation"]
        flat["omega"] = gen.get("omega", 0.0)
        flat["method"] = gen.get("method", "cfg")
        flat["num_samples"] = gen.get("num_samples", 64)
        flat["checkpoint_path"] = gen.get("checkpoint_path", None)

    # Hardware section
    if "hardware" in raw:
        hw = raw["hardware"]
        flat["device"] = hw.get("device", "cuda")

    logger.info("Loaded CBDM config from %s", yaml_path)
    return CBDMConfig(**flat)
