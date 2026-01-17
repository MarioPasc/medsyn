# medsyn/cli/data.py
"""
CLI: Prepare MedMNIST datasets (PathMNIST, BloodMNIST, DermaMNIST) and generate JSON index.

Supports unified preprocessing for multiple datasets with stratified sampling
and optional PNG export for YOLO classification training.

Commands:
  medsyn-prepare-data --config path/to/medsyn_config.yaml
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

from medsyn.data.config import load_config, ensure_dirs, ProjectCfg
from medsyn.data.registry import prepare_dataset, get_class_map
from medsyn.data.base import SplitDatasets
from medsyn.data.export import export_split_to_pngs_and_index
from medsyn.data.yolo_dataset import generate_yolo_classification_from_index
from medsyn.data.npz_format import create_unified_npz, create_metadata
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("medsyn.cli.data")


def _build_index_structure(name: str, per_split_indices: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Build JSON index structure for a dataset.

    Output format:
    {
      "<DatasetName>": {
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
    - metadata: JSON string with dataset info

    Args:
        cfg: Project configuration
        ds: SplitDatasets with train/val/test
    """
    logger.info("Creating custom NPZ file from prepared datasets...")

    # Determine original NPZ path based on size
    if cfg.data.size == 28:
        original_npz_path = Path(cfg.data.download_dir) / f"{cfg.data.flag}.npz"
    else:
        original_npz_path = Path(cfg.data.download_dir) / f"{cfg.data.flag}_{cfg.data.size}.npz"

    if not original_npz_path.exists():
        raise FileNotFoundError(f"Original NPZ not found: {original_npz_path}")

    original_data = np.load(str(original_npz_path))

    # Prepare splits dictionary for unified NPZ format
    splits = {}

    for split_name, dataset in [("train", ds.train), ("val", ds.val), ("test", ds.test)]:
        logger.info(f"Processing {split_name} split...")

        # Get the indices used for this split
        indices = ds.indices[split_name]
        if indices is None:
            # Full split (no reduction)
            images = original_data[f"{split_name}_images"]
            labels = original_data[f"{split_name}_labels"]
        else:
            # Reduced split - need to extract specific indices
            all_images = original_data[f"{split_name}_images"]
            all_labels = original_data[f"{split_name}_labels"]
            images = all_images[indices]
            labels = all_labels[indices]

        # Ensure proper format: [N, H, W, C] for images
        if images.ndim == 3:
            images = images[..., np.newaxis]

        # Create is_synth array (all False for original data)
        is_synth = np.zeros(len(images), dtype=bool)

        splits[split_name] = {
            "images": images,
            "labels": labels,
            "is_synth": is_synth,
        }

        logger.info(f"  {split_name}: {len(images)} samples, shape {images.shape}")

    # Create metadata
    metadata = create_metadata(
        dataset_name=cfg.data.flag,
        image_size=cfg.data.size,
        extra={
            "reduction_strategy": cfg.data.reduction.strategy,
            "reduction_train": cfg.data.reduction.train,
            "reduction_val": cfg.data.reduction.val,
            "reduction_test": cfg.data.reduction.test,
            "seed": cfg.data.seed,
        },
    )

    # Create unified NPZ with metadata
    output_path = Path(cfg.data.postprocess_npz.npz_path)
    create_unified_npz(splits, metadata, output_path)

    # Verify the saved file
    verification = np.load(str(output_path), allow_pickle=True)
    logger.info("Verification - Keys in custom NPZ: %s", list(verification.keys()))
    for key in verification.keys():
        if key != "metadata":
            logger.info("  %s: %s", key, verification[key].shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MedMNIST datasets (PathMNIST, BloodMNIST, DermaMNIST)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to medsyn_config.yaml")
    args = parser.parse_args()

    cfg: ProjectCfg = load_config(args.config)
    ensure_dirs(cfg)

    dataset_flag = cfg.data.flag
    dataset_name = dataset_flag.replace("mnist", "MNIST").replace("path", "Path").replace("blood", "Blood").replace("derma", "Derma")
    logger.info("Preparing dataset: %s", dataset_flag)

    # 1) Build reduced and stratified datasets using the registry
    ds: SplitDatasets = prepare_dataset(dataset_flag, cfg)

    # 2) PNG extraction (conditional based on save_png.enabled)
    if cfg.data.save_png.enabled:
        logger.info("PNG extraction enabled - exporting images and creating JSON index...")

        # Export each split to PNGs and build local indices
        out_root = Path(cfg.data.save_png.processed_dir).resolve()
        out_root.mkdir(parents=True, exist_ok=True)
        per_split_indices: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for split_name, dataset in (("train", ds.train), ("val", ds.val), ("test", ds.test)):
            split_dir = out_root / dataset_flag / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Exporting %s images -> %s", split_name, split_dir)
            per_split_indices[split_name] = export_split_to_pngs_and_index(dataset, split_dir)

        # 3) Package global index and write JSON
        final_idx = _build_index_structure(dataset_name, per_split_indices)
        index_path = Path(cfg.data.save_png.index_json).resolve()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as fh:
            json.dump(final_idx, fh, ensure_ascii=False, indent=2)
        logger.info("Index JSON written to: %s", index_path)

        # 4) Generate YOLO classification dataset (if configured)
        if cfg.data.save_png.yolo_folder_dataset:
            yolo_root = Path(cfg.data.save_png.yolo_folder_dataset)
            yolo_root = yolo_root.resolve()
            yolo_root.mkdir(parents=True, exist_ok=True)

            # Use registry to get class map for any supported dataset
            class_map = get_class_map(dataset_flag)

            # Determine whether to use symlinks or copies based on config
            prefer_copy = not cfg.data.save_png.yolo_use_symlinks
            mode_str = "copying files" if prefer_copy else "creating symlinks"
            logger.info("Building YOLO dataset by %s...", mode_str)

            rep = generate_yolo_classification_from_index(
                index_path,
                yolo_root,
                class_map,
                use_relative_symlinks=True,
                allow_copy_fallback=cfg.data.save_png.yolo_allow_copy_fallback,
                prefer_copy=prefer_copy,
            )
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

    logger.info("Data preparation complete for %s!", dataset_flag)


if __name__ == "__main__":
    main()
