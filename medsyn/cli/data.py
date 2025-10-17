# medsyn/cli/data.py
"""
CLI: preparar PathMNIST y generar índice JSON por split para entrenar VAE y clasificadores.

Comandos:
  medsyn-prepare-data --config path/a/medsyn_config.yaml
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from medsyn.data.config import load_config, ensure_dirs, ProjectCfg
from medsyn.data.pathmnist import prepare_pathmnist, SplitDatasets
from medsyn.data.export import export_split_to_pngs_and_index
from medsyn.data.yolo_dataset import (
    generate_yolo_classification_from_index,
    build_pathmnist_class_map,
)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("medsyn.cli.data")


def _build_index_structure(name: str, per_split_indices: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Estructura JSON de salida:
    {
      "PathMNIST": {
        "train": { "0": {"image": "...", "label": 3, "is_synth": false}, ... },
        "val":   { ... },
        "test":  { ... }
      }
    }
    """
    return {name: {k: {str(i): per_split_indices[k][i] for i in sorted(per_split_indices[k].keys())}
                   for k in ("train", "val", "test")}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PathMNIST splits and JSON index")
    parser.add_argument("--config", type=str, required=True, help="Ruta a medsyn_config.yaml")
    args = parser.parse_args()

    cfg: ProjectCfg = load_config(args.config)
    ensure_dirs(cfg)

    # 1) Construir datasets reducidos y estratificados según YAML
    ds: SplitDatasets = prepare_pathmnist(cfg)

    # 2) Exportar cada split a PNGs y construir índices locales
    out_root = Path(cfg.data.processed_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    per_split_indices: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for split_name, dataset in (("train", ds.train), ("val", ds.val), ("test", ds.test)):
        split_dir = out_root / "pathmnist" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Exporting %s images -> %s", split_name, split_dir)
        per_split_indices[split_name] = export_split_to_pngs_and_index(dataset, split_dir)

    # 3) Empaquetar índice global y escribir JSON
    final_idx = _build_index_structure("PathMNIST", per_split_indices)
    index_path = Path(cfg.data.index_json).resolve()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump(final_idx, fh, ensure_ascii=False, indent=2)
    logger.info("Index JSON written to: %s", index_path)

    if cfg.data.yolo_folder_dataset:
        # 4) Generar dataset de clasificación YOLO con symlinks
        yolo_root = Path(args.yolo_root) if args.yolo_root else Path(cfg.data.yolo_folder_dataset)
        yolo_root = yolo_root.resolve()
        yolo_root.mkdir(parents=True, exist_ok=True)
        class_map = build_pathmnist_class_map()
        rep = generate_yolo_classification_from_index(index_path, yolo_root, class_map)
        logger.info("YOLO dataset at %s", yolo_root)
        logger.info("Counts: %s", rep.counts)

if __name__ == "__main__":
    main()
