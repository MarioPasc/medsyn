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
    img_size: int = 64
    latent_dim: int = 32
    base_channels: int = 64          # width of the first conv stage
    num_down: int = 4                # downsampling stages (×2 per stage)
    decoder_sigmoid: bool = True     # apply sigmoid at output
    # --- conditional params ---
    num_classes: int = 9
    conditioning: Literal["film","none"] = "film"
    class_embed_dim: int = 64
    use_class_prior: bool = False    # start vanilla; enable after recon stabilizes
    decoder_conditioning: bool = False  # off in decoder until KL>0

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
    save_every_epoch: int = 10
    # Lagging-inference mitigation: extra encoder updates to prevent collapse
    encoder_extra_steps: int = 2      # mitigate lagging inference
    encoder_heavy_epochs: int = 5     # first epochs only

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
class BetaSchedCfg:
    type: Literal["monotonic","cyclical"] = "monotonic"
    cycles: int = 4
    ratio_increase: float = 0.5
    beta_min: float = 0.0
    beta_max: float = 1.0

@dataclass(frozen=True)
class CapacityCfg:
    use: bool = False               # start disabled to avoid early KL pinning
    C_max: float = 5.0              # target KL per-dim scalar (nats averaged)
    steps_to_max: int = 30000       # steps to reach C_max
    gamma: float = 50.0             # gentler pull when enabled

@dataclass(frozen=True)
class InfoVAECfg:
    use: bool = False
    weight: float = 10.0
    kernel: Literal["rbf","imq"] = "rbf"

@dataclass(frozen=True)
class BVAELossCfg:
    """Loss weights and types."""
    beta: float = 1.0                 # KLD weight (techo)
    recon_type: Literal["mse","bce","l1","l1_ssim"] = "l1"  # SSIM later after stability
    recon_weight: float = 1.0
    kld_weight: float = 1.0           # multiplies beta*KLD
    l1_weight: float = 1.0
    ssim_weight: float = 0.85
    free_bits_nats: float = 0.03
    beta_schedule: Optional[BetaSchedCfg] = None
    infovae_mmd: Optional[InfoVAECfg] = None
    prior_reg_w: float = 1e-4
    capacity: Optional[CapacityCfg] = None
    per_class_mmd_use: bool = False   # per-class MMD regularization
    per_class_mmd_w: float = 1.0      # weight for per-class MMD

    def __post_init__(self):
        # Provide defaults for nested configs if None
        if self.beta_schedule is None:
            object.__setattr__(self, 'beta_schedule', BetaSchedCfg())
        if self.infovae_mmd is None:
            object.__setattr__(self, 'infovae_mmd', InfoVAECfg())
        if self.capacity is None:
            object.__setattr__(self, 'capacity', CapacityCfg())

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

    # Parse nested configs for loss
    loss_dict = b.get("loss", {})
    sched_loss = loss_dict.pop("beta_schedule", {})
    infommd = loss_dict.pop("infovae_mmd", {})
    capacity_dict = loss_dict.pop("capacity", {})
    loss = BVAELossCfg(
        **loss_dict,
        beta_schedule=BetaSchedCfg(**sched_loss),
        infovae_mmd=InfoVAECfg(**infommd),
        capacity=CapacityCfg(**capacity_dict)
    )

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
