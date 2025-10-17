# medsyn/models/ccDDPM/config.py
# Purpose: Typed config loader for ccDDPM training/inference.
# Notes:
# - Reads the global YAML, expects `data.index_json` and a `ccddpm` section.
# - Provides defaults if keys are missing.
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Mapping, Sequence
import yaml
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimCfg:
    lr: float = 2e-4
    wd: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

@dataclass
class SchedCfg:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"  # ["linear","squaredcos_cap_v2",...]
    prediction_type: str = "epsilon"  # DDPM default

@dataclass
class TrainCfg:
    image_size: int = 128
    in_channels: int = 3
    class_embed_dim: int = 16
    num_classes: int = 9
    batch_size: int = 64
    epochs: int = 50
    num_workers: int = 8
    seed: int = 1337
    mixed_precision: bool = True
    grad_clip_norm: Optional[float] = 1.0
    ema_use: bool = True
    ema_decay: float = 0.999
    guidance_p_uncond: float = 0.1  # classifier-free label drop prob
    log_every: int = 100
    ckpt_every_epochs: int = 1
    patience: int = 15  # Early stopping: number of epochs without improvement
    use_tqdm: bool = True  # Enable/disable tqdm progress bars (useful for HPC logs)
    output_dir: Path = Path("./outputs/ccddpm")

@dataclass
class InferenceCfg:
    guidance_scale: float = 0.0  # 0 = unconditional; >0 for cfg
    num_inference_steps: int = 1000
    save_grid: bool = True
    out_dir: Path = Path("./samples/ccddpm")

@dataclass
class DataloaderCfg:
    type: str = "json"  # "json" or "npz"
    npz_path: Optional[Path] = None  # Path to NPZ file (required if type="npz")

@dataclass
class DataCfg:
    index_json: Path
    processed_root: Optional[Path] = None  # optional; paths already absolute in JSON
    split: str = "train"
    normalize: bool = True  # scale to [-1,1]

@dataclass
class CCDDPmCfg:
    train: TrainCfg = field(default_factory=TrainCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    sched: SchedCfg = field(default_factory=SchedCfg)
    infer: InferenceCfg = field(default_factory=InferenceCfg)
    dataloader: DataloaderCfg = field(default_factory=DataloaderCfg)
    data: DataCfg | None = None  # set after reading YAML

@dataclass
class ProjectCfg:
    data_index_json: Path
    ccddpm: CCDDPmCfg

def _as_path(p: Optional[str]) -> Optional[Path]:
    return None if p is None else Path(p).expanduser().resolve()

def _expand_env(obj: Any) -> Any:
    """
    Recursively expand ${ENV} and $ENV in strings within mappings/lists.
    Ensures backward compatibility with absolute paths.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, Mapping):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        expanded = [_expand_env(v) for v in obj]
        return type(obj)(expanded) if isinstance(obj, tuple) else expanded
    return obj

def load_cfg(yaml_path: str | Path, split: str = "train") -> ProjectCfg:
    """
    Load YAML config and build a ProjectCfg with ccDDPM fields.
    Supports environment variable expansion (${VAR} or $VAR) in YAML values.
    Expects:
      data.save_png.index_json: path to JSON index built by medsyn/cli/data.py (for JSON dataloader)
      data.postprocess_npz.npz_path: path to custom NPZ file (for NPZ dataloader)
      ccddpm: optional section overriding defaults.
    """
    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    
    # Expand environment variables in all string values
    raw = _expand_env(raw)

    # data section
    data_dict = raw.get("data", {})
    
    # Get index_json from save_png section (for JSON dataloader compatibility)
    save_png = data_dict.get("save_png", {})
    idx = save_png.get("index_json")
    if idx:
        index_json_path = _as_path(idx)
    else:
        # Fallback for backward compatibility
        idx = data_dict.get("index_json")
        if idx:
            index_json_path = _as_path(idx)
        else:
            logger.warning("No index_json found in YAML. JSON dataloader may not work.")
            index_json_path = None
    
    data_cfg = DataCfg(
        index_json=index_json_path if index_json_path else Path("./dummy.json"),
        processed_root=_as_path(save_png.get("processed_dir")) if save_png.get("processed_dir") else None,
        split=split
    )

    # ccddpm section with deep defaults
    cc = raw.get("ccddpm", {}) or {}
    train = TrainCfg(**{**TrainCfg().__dict__, **{k: cc.get("train", {}).get(k, v) for k, v in TrainCfg().__dict__.items()}})
    optim = OptimCfg(**{**OptimCfg().__dict__, **cc.get("optim", {})})
    sched = SchedCfg(**{**SchedCfg().__dict__, **cc.get("sched", {})})
    infer = InferenceCfg(**{**InferenceCfg().__dict__, **cc.get("infer", {})})
    
    # dataloader configuration - read NPZ path from data.postprocess_npz section
    dl_cfg_dict = cc.get("dataloader", {})
    dataloader_type = dl_cfg_dict.get("type", "json")
    
    # Get NPZ path from data.postprocess_npz if type is npz
    npz_path = None
    if dataloader_type.lower() == "npz":
        postprocess_npz = data_dict.get("postprocess_npz", {})
        npz_path_str = postprocess_npz.get("npz_path")
        if npz_path_str:
            npz_path = _as_path(npz_path_str)
        else:
            raise ValueError("NPZ dataloader selected but data.postprocess_npz.npz_path not found in YAML")
    
    dataloader_cfg = DataloaderCfg(
        type=dataloader_type,
        npz_path=npz_path
    )
    
    cc_cfg = CCDDPmCfg(train=train, optim=optim, sched=sched, infer=infer, dataloader=dataloader_cfg, data=data_cfg)

    proj = ProjectCfg(
        data_index_json=index_json_path if index_json_path else Path("./dummy.json"),
        ccddpm=cc_cfg
    )
    logger.info("Loaded ccDDPM config. image_size=%d, classes=%d, timesteps=%d, dataloader=%s",
                cc_cfg.train.image_size, cc_cfg.train.num_classes, cc_cfg.sched.num_train_timesteps, dataloader_type)
    if npz_path:
        logger.info("NPZ dataloader path: %s", npz_path)
    return proj
