# data/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import yaml # type: ignore
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
class DataCfg:
    """Data configuration for MedMNIST PathMNIST dataset."""
    flag: Literal["pathmnist"] = "pathmnist"
    size: int = 28                    # 28 is default MedMNIST size
    download_dir: str = "./data_raw"  # where .npz will live
    processed_dir: str = "./data_processed"  # where PNGs are stored by split
    index_json: str = "./indexes/pathmnist_index.json"  # index file path
    seed: int = 17
    num_workers: int = 4
    reduction: ReductionCfg = ReductionCfg()

@dataclass(frozen=True)
class ProjectCfg:
    data: DataCfg

def _to_dataclass(d: dict) -> ProjectCfg:
    """Recursively cast dict -> dataclasses for strong typing."""
    red = ReductionCfg(**d["data"].get("reduction", {}))
    data = DataCfg(**{**d["data"], "reduction": red})
    return ProjectCfg(data=data)

def load_config(path: str | Path) -> ProjectCfg:
    """
    Load YAML configuration (medsyn_config.yaml) into typed dataclasses.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    cfg = _to_dataclass(raw)
    logger.info("Loaded config from %s", p)
    return cfg

def ensure_dirs(cfg: ProjectCfg) -> None:
    """
    Create expected directories if missing.
    """
    Path(cfg.data.download_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.data.index_json).parent.mkdir(parents=True, exist_ok=True)
