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
import numpy as np

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


def _create_custom_npz(cfg: ProjectCfg, ds: SplitDatasets) -> None:
    """
    Create custom NPZ file from the prepared datasets with custom splits.
    
    This reads the original MedMNIST NPZ file and creates a new NPZ with:
    - Custom split indices (after reduction/stratification)
    - {split}_images: [N, H, W, C] uint8
    - {split}_labels: [N] int64
    - {split}_is_synth: [N] bool (all False for original data)
    
    Args:
        cfg: Project configuration
        ds: SplitDatasets with train/val/test
    """
    logger.info("Creating custom NPZ file from prepared datasets...")
    
    # Load original NPZ to get image data
    original_npz_path = Path(cfg.data.download_dir) / f"{cfg.data.flag}_{cfg.data.size}.npz"
    if not original_npz_path.exists():
        raise FileNotFoundError(f"Original NPZ not found: {original_npz_path}")
    
    original_data = np.load(str(original_npz_path))
    
    # Prepare output dictionary
    output_dict = {}
    
    for split_name, dataset in [("train", ds.train), ("val", ds.val), ("test", ds.test)]:
        logger.info(f"Processing {split_name} split...")
        
        # Get the indices used for this split
        indices = ds.indices[split_name]
        if indices is None:
            # Full split (no reduction)
            original_split_key = f"{split_name}_images"
            images = original_data[original_split_key]
            labels = original_data[f"{split_name}_labels"]
        else:
            # Reduced split - need to extract specific indices
            original_split_key = f"{split_name}_images"
            all_images = original_data[original_split_key]
            all_labels = original_data[f"{split_name}_labels"]
            
            images = all_images[indices]
            labels = all_labels[indices]
        
        # Ensure proper format: [N, H, W, C] for images
        if images.ndim == 3:
            # Add channel dimension if missing
            images = images[..., np.newaxis]
        
        # Create is_synth array (all False for original data)
        is_synth = np.zeros(len(images), dtype=bool)
        
        # Store in output dict
        output_dict[f"{split_name}_images"] = images.astype(np.uint8)
        output_dict[f"{split_name}_labels"] = labels.squeeze().astype(np.int64)
        output_dict[f"{split_name}_is_synth"] = is_synth
        
        logger.info(f"  {split_name}: {len(images)} samples, shape {images.shape}")
    
    # Save the custom NPZ
    output_path = Path(cfg.data.postprocess_npz.npz_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(str(output_path), **output_dict)
    logger.info(f"Custom NPZ saved to: {output_path}")
    
    # Verify the saved file
    verification = np.load(str(output_path))
    logger.info("Verification - Keys in custom NPZ: %s", list(verification.keys()))
    for key in verification.keys():
        logger.info("  %s: %s", key, verification[key].shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PathMNIST splits and JSON index")
    parser.add_argument("--config", type=str, required=True, help="Ruta a medsyn_config.yaml")
    args = parser.parse_args()

    cfg: ProjectCfg = load_config(args.config)
    ensure_dirs(cfg)

    # 1) Construir datasets reducidos y estratificados según YAML
    ds: SplitDatasets = prepare_pathmnist(cfg)

    # 2) PNG extraction (conditional based on save_png.enabled)
    if cfg.data.save_png.enabled:
        logger.info("PNG extraction enabled - exporting images and creating JSON index...")
        
        # Exportar cada split a PNGs y construir índices locales
        out_root = Path(cfg.data.save_png.processed_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        per_split_indices: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for split_name, dataset in (("train", ds.train), ("val", ds.val), ("test", ds.test)):
            split_dir = out_root / "pathmnist" / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Exporting %s images -> %s", split_name, split_dir)
            per_split_indices[split_name] = export_split_to_pngs_and_index(dataset, split_dir)

        # 3) Empaquetar índice global y escribir JSON
        final_idx = _build_index_structure("PathMNIST", per_split_indices)
        index_path = Path(cfg.data.save_png.index_json).resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(final_idx, fh, ensure_ascii=False, indent=2)
        logger.info("Index JSON written to: %s", index_path)

        # 4) Generar dataset de clasificación YOLO con symlinks (if configured)
        if cfg.data.save_png.yolo_folder_dataset:
            yolo_root = Path(cfg.data.save_png.yolo_folder_dataset)
            yolo_root = yolo_root.resolve()
            yolo_root.mkdir(parents=True, exist_ok=True)
            class_map = build_pathmnist_class_map()
            rep = generate_yolo_classification_from_index(index_path, yolo_root, class_map)
            logger.info("YOLO dataset at %s", yolo_root)
            logger.info("Counts: %s", rep.counts)
    else:
        logger.info("PNG extraction disabled (save_png.enabled=false) - skipping PNG export and JSON index")

    # 5) NPZ postprocessing (conditional based on postprocess_npz.enabled)
    if cfg.data.postprocess_npz.enabled:
        logger.info("NPZ postprocessing enabled - creating custom NPZ file...")
        _create_custom_npz(cfg, ds)
    else:
        logger.info("NPZ postprocessing disabled (postprocess_npz.enabled=false) - skipping custom NPZ creation")

    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
