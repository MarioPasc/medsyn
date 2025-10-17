# data/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Any, Mapping
import yaml # type: ignore
import os
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ReductionCfg:
    """Reduction parameters to create a smaller, stratified subset per split."""
    strategy: Literal["fraction", "max_per_class"] = "fraction"
    train: float = 0.2
    val: float = 1.0
    test: float = 1.0
    max_per_class: Optional[int] = None

@dataclass(frozen=True)
class SavePngCfg:
    """Configuration for PNG extraction."""
    enabled: bool = True
    processed_dir: str = "./data_processed"  # where PNGs are stored by split
    index_json: str = "./indexes/pathmnist_index.json"  # index file path
    yolo_folder_dataset: Optional[str] = None  # root for YOLO classification symlink dataset

@dataclass(frozen=True)
class PostprocessNpzCfg:
    """Configuration for NPZ postprocessing."""
    enabled: bool = False
    npz_path: str = "./PathMNIST.npz"  # output path for custom NPZ with splits

@dataclass(frozen=True)
class DataCfg:
    """Data configuration for MedMNIST PathMNIST dataset."""
    flag: Literal["pathmnist"] = "pathmnist"
    size: int = 28                    # 28 is default MedMNIST size
    download_dir: str = "./data_raw"  # where original .npz will live
    seed: int = 17
    num_workers: int = 4
    reduction: ReductionCfg = ReductionCfg()
    save_png: SavePngCfg = SavePngCfg()
    postprocess_npz: PostprocessNpzCfg = PostprocessNpzCfg()

@dataclass(frozen=True)
class ProjectCfg:
    data: DataCfg

def _to_dataclass(d: dict) -> ProjectCfg:
    """Recursively cast dict -> dataclasses for strong typing."""
    red = ReductionCfg(**d["data"].get("reduction", {}))
    save_png = SavePngCfg(**d["data"].get("save_png", {}))
    postprocess_npz = PostprocessNpzCfg(**d["data"].get("postprocess_npz", {}))
    data = DataCfg(**{**d["data"], "reduction": red, "save_png": save_png, "postprocess_npz": postprocess_npz})
    return ProjectCfg(data=data)

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

def load_config(path: str | Path) -> ProjectCfg:
    """
    Load YAML configuration (medsyn_config.yaml) into typed dataclasses.
    Supports environment variable expansion (${VAR} or $VAR) in YAML values.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    # Expand environment variables in all string values
    raw = _expand_env(raw)
    cfg = _to_dataclass(raw)
    logger.info("Loaded config from %s", p)
    return cfg

def ensure_dirs(cfg: ProjectCfg) -> None:
    """
    Create expected directories if missing.
    """
    Path(cfg.data.download_dir).mkdir(parents=True, exist_ok=True)
    if cfg.data.save_png.enabled:
        Path(cfg.data.save_png.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.data.save_png.index_json).parent.mkdir(parents=True, exist_ok=True)
        if cfg.data.save_png.yolo_folder_dataset:
            Path(cfg.data.save_png.yolo_folder_dataset).mkdir(parents=True, exist_ok=True)
    if cfg.data.postprocess_npz.enabled:
        Path(cfg.data.postprocess_npz.npz_path).parent.mkdir(parents=True, exist_ok=True)