# medsyn/models/bVAE/config.py
# Purpose: Parse medsyn_config.yaml and expose typed configs for the β-VAE stack.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import yaml
import logging

logger = logging.getLogger(__name__)

# --------------------------- Dataclasses --------------------------------------

@dataclass(frozen=True)
class BVAEModelCfg:
    """β-VAE model hyperparameters (conditional-ready)."""
    in_channels: int = 3
    img_size: int = 28
    latent_dim: int = 16
    base_channels: int = 32          # width of the first conv stage
    num_down: int = 3                # downsampling stages (×2 per stage)
    decoder_sigmoid: bool = True     # apply sigmoid at output
    # --- conditional params ---
    num_classes: int = 9
    conditioning: Literal["film","none"] = "film"
    class_embed_dim: int = 32

@dataclass(frozen=True)
class BVAETrainCfg:
    """Training loop hyperparameters."""
    epochs: int = 50
    batch_size: int = 128
    num_workers: int = 4
    device: Literal["cpu","cuda","mps"] = "cuda"
    mixed_precision: bool = True
    grad_clip_norm: float = 1.0
    seed: int = 17
    output_dir: str = "./outputs/bvae"

@dataclass(frozen=True)
class BVAEOptimCfg:
    """Optimizer configuration."""
    optimizer: Literal["adamw","adam"] = "adamw"
    lr_init: float = 1e-4             # initial LR of optimizer
    weight_decay: float = 1e-4
    betas: tuple[float,float] = (0.9, 0.999)
    eps: float = 1e-8

@dataclass(frozen=True)
class BVAESchedCfg:
    """LR scheduler configuration (OneCycleLR)."""
    use_onecycle: bool = True
    max_lr: float = 3e-4              # peak LR in the cycle
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4

@dataclass(frozen=True)
class BVAELossCfg:
    """Loss weights and types."""
    beta: float = 2.0                 # KLD weight
    recon_type: Literal["mse","bce"] = "mse"
    recon_weight: float = 1.0
    kld_weight: float = 1.0           # multiplies beta*KLD

@dataclass(frozen=True)
class BVAEGenerateCfg:
    """Generation configuration."""
    class_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    samples_per_class: int = 100
    checkpoint: str = "best.pt"

@dataclass(frozen=True)
class DataPathsCfg:
    """Data paths from the global config."""
    index_json: str                   # path to the PathMNIST JSON index
    processed_dir: Optional[str] = None

@dataclass(frozen=True)
class BVAEConfig:
    """Aggregate config for the β-VAE stack."""
    model: BVAEModelCfg
    train: BVAETrainCfg
    optim: BVAEOptimCfg
    sched: BVAESchedCfg
    loss:  BVAELossCfg
    generate: BVAEGenerateCfg
    data:  DataPathsCfg

# --------------------------- Loader -------------------------------------------

def load_bvae_config(yaml_path: str | Path) -> BVAEConfig:
    # Load YAML and map keys to dataclasses under the "bVAE" section and "data".
    p = Path(yaml_path)
    with p.open("r", encoding="utf-8") as fh:
        y = yaml.safe_load(fh)

    b = y.get("bVAE", {})
    d = y.get("data", {})

    model = BVAEModelCfg(**b.get("model", {}))
    train = BVAETrainCfg(**b.get("train", {}))
    optim = BVAEOptimCfg(**b.get("optim", {}))
    sched = BVAESchedCfg(**b.get("sched", {}))
    loss  = BVAELossCfg(**b.get("loss",  {}))
    gen_cfg = b.get("generate", {})
    # Convert class_ids list to tuple for frozen dataclass
    if "class_ids" in gen_cfg and isinstance(gen_cfg["class_ids"], list):
        gen_cfg["class_ids"] = tuple(gen_cfg["class_ids"])
    generate = BVAEGenerateCfg(**gen_cfg)
    data  = DataPathsCfg(index_json=d["index_json"], processed_dir=d.get("processed_dir"))
    cfg = BVAEConfig(model=model, train=train, optim=optim, sched=sched, loss=loss, generate=generate, data=data)

    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded β-VAE config from %s", p)
    return cfg
