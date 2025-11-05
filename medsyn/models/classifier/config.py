from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

@dataclass
class MedsynYOLOCfg:
    npz_path: Path
    model: str
    project: Path
    name: str
    imgsz: int
    epochs: int
    batch: int
    workers: int
    training_images: str  # PathMNIST | PathMNIST_and_synth | synth
    overrides: Dict[str, Any]  # full hyperparameters

def _exp(s: str | None) -> str | None:
    import os
    return None if s is None else os.path.expandvars(s)

def load_cfg(medsyn_cfg: str | Path, hparams_yaml: str | Path) -> MedsynYOLOCfg:
    with open(medsyn_cfg, "r", encoding="utf-8") as fh:
        m = yaml.safe_load(fh)
    with open(hparams_yaml, "r", encoding="utf-8") as fh:
        hp = yaml.safe_load(fh) or {}

    y = m.get("yolo", {}) or {}
    data = m.get("data", {})
    pp = (data.get("postprocess_npz") or {})
    npz_path = Path(_exp(pp.get("npz_path"))).expanduser().resolve()

    cfg = MedsynYOLOCfg(
        npz_path=npz_path,
        model=y.get("model", "yolo11n-cls.pt"),
        project=Path(_exp(y.get("project", "./runs"))).expanduser().resolve(),
        name=y.get("name", "exp"),
        imgsz=int(y.get("imgsz", 64)),
        epochs=int(y.get("epochs", 100)),
        batch=int(y.get("batch", 128)),
        workers=int(y.get("workers", 8)),
        training_images=str(y.get("training_images", "PathMNIST")),
        overrides={}
    )

    # Build Ultralytics overrides
    o: Dict[str, Any] = dict(hp or {})
    o.update({
        "task": "classify",
        "model": cfg.model,
        "data": "dummy",  # Dummy value to satisfy YOLO's argument parser
        "imgsz": cfg.imgsz,
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "workers": cfg.workers,
        "project": str(cfg.project),
        "name": cfg.name,
        # custom keys consumed by our trainer/validator
        "medsyn_npz_path": str(cfg.npz_path),
        "medsyn_training_images": cfg.training_images,
    })
    cfg.overrides = o
    return cfg
