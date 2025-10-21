from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple
import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)


class YOLOSymlinkError(RuntimeError):
    """Raised when creating a required symlink fails."""


@dataclass(frozen=True)
class YOLOBuildReport:
    """Summary counters by split and class."""
    counts: Dict[str, Dict[str, int]]


def build_pathmnist_class_map() -> Dict[int, str]:
    """
    Return the official PathMNIST 9-class mapping (label -> folder name).
    """
    return {
        0: "adipose",
        1: "background",
        2: "debris",
        3: "lymphocytes",
        4: "mucus",
        5: "smooth_muscle",
        6: "normal_colon_mucosa",
        7: "cancer_associated_stroma",
        8: "colorectal_adenocarcinoma_epithelium",
    }


def _safe_symlink(src: Path, dst: Path, use_relative: bool = True, allow_copy: bool = True, prefer_copy: bool = False) -> None:
    """
    Create a symlink at dst pointing to src. Parents are created. Idempotent if link exists.
    If symlink creation fails and allow_copy is True, falls back to copying the file.
    If prefer_copy is True, directly copies the file without attempting to create a symlink.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If prefer_copy is True, skip symlink attempt and go straight to copying
    if prefer_copy:
        # Remove existing file/link if present
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except Exception as e:
                raise YOLOSymlinkError(f"Cannot replace existing path: {dst} -> {e}")

        try:
            shutil.copy2(src, dst)
            return
        except Exception as copy_err:
            raise YOLOSymlinkError(f"Copy failed: {src} -> {dst}: {copy_err}")

    # Remove broken or wrong existing links/files
    if dst.exists() or dst.is_symlink():
        try:
            if dst.is_symlink():
                target = dst.resolve(strict=False)
                if target == src:
                    return
                dst.unlink()
            else:
                # A file exists where link should be; remove and replace
                dst.unlink()
        except Exception as e:
            raise YOLOSymlinkError(f"Cannot replace existing path: {dst} -> {e}")

    try:
        link_target = os.path.relpath(src, dst.parent) if use_relative else str(src)
        os.symlink(link_target, dst)
    except (OSError, NotImplementedError) as e:
        if allow_copy:
            # Filesystem doesn't support symlinks; fall back to copying
            logger.warning("Symlink failed (%s), falling back to copying: %s -> %s", e, src, dst)
            try:
                shutil.copy2(src, dst)
            except Exception as copy_err:
                raise YOLOSymlinkError(f"Both symlink and copy failed: {src} -> {dst}: {copy_err}")
        else:
            raise YOLOSymlinkError(f"symlink failed: {src} -> {dst}: {e}")

def _iter_index_entries(index_split: Dict[str, Dict[str, object]]) -> Iterable[Tuple[int, Path, int]]:
    """
    Yield (idx, image_path, label) from the JSON split dict.
    """
    for k, meta in index_split.items():
        idx = int(k)
        img = Path(str(meta["image"]))  # absolute path
        lab = int(meta["label"])  # type: ignore
        yield idx, img, lab


def generate_yolo_classification_from_index(
    index_json: Path,
    out_root: Path,
    class_map: Dict[int, str],
    splits: Tuple[str, ...] = ("train", "val", "test"),
    use_relative_symlinks: bool = True,
    allow_copy_fallback: bool = True,
    prefer_copy: bool = False,
) -> YOLOBuildReport:
    """
    Build a Ultralytics-compatible classification dataset using symlinks or copies.
    Reads the MedSyn index JSON and places links/files at:
      out_root/{train,val,test}/{class_name}/*.png

    Args:
        index_json: Path to the classification index JSON file
        out_root: Root directory for the YOLO dataset
        class_map: Mapping from label integers to class names
        splits: Tuple of split names to process
        use_relative_symlinks: If True, use relative symlinks; if False, use absolute
        allow_copy_fallback: If True and symlink fails, fall back to copying
        prefer_copy: If True, directly copy files without attempting symlinks
    """
    with Path(index_json).open("r", encoding="utf-8") as fh:
        idx = json.load(fh)

    # Top-level dataset key, e.g., "PathMNIST"
    if not isinstance(idx, dict) or len(idx) != 1:
        raise ValueError("Unexpected index JSON structure.")
    ds_name = next(iter(idx.keys()))
    splits_dict = idx[ds_name]

    counts: Dict[str, Dict[str, int]] = {s: {} for s in splits}
    for split in splits:
        split_dict = splits_dict.get(split, {})
        for i, src, lab in _iter_index_entries(split_dict):
            if not src.exists():
                logger.warning("Missing source image, skipping: %s", src)
                continue
            if lab not in class_map:
                raise KeyError(f"Label {lab} missing in class_map.")
            cls = class_map[lab]
            dst_dir = out_root / split / cls
            # Name ensures uniqueness even if basenames repeat across splits
            dst = dst_dir / f"{i:06d}_{src.name}"
            _safe_symlink(src, dst, use_relative=use_relative_symlinks, allow_copy=allow_copy_fallback, prefer_copy=prefer_copy)
            counts[split][cls] = counts[split].get(cls, 0) + 1

        logger.info("Split %s -> %d total", split, sum(counts[split].values()))

    # Also write a names.txt for convenience (one class per line in label order)
    names_txt = out_root / "names.txt"
    ordered = [class_map[i] for i in sorted(class_map.keys())]
    names_txt.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return YOLOBuildReport(counts=counts)
