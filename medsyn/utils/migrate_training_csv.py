#!/usr/bin/env python3
"""
Migrate old combined training_metrics.csv to new split format:
- training_metrics.csv: train/val/test splits only (TRAINING_FIELDS + per-class)
- diagnostics_metrics.csv: diag split only (DIAGNOSTIC_FIELDS)

Usage:
    python -m medsyn.utils.migrate_training_csv /path/to/output_dir [--backup]
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Any

from medsyn.models.ccDDPM.engine.logging.training_logging import TRAINING_FIELDS, DIAGNOSTIC_FIELDS


def detect_per_class_columns(fieldnames: List[str]) -> List[str]:
    """Detect loss_c* columns from CSV header."""
    return [f for f in fieldnames if f.startswith("loss_c")]


def migrate_csv(input_dir: Path, backup: bool = True) -> Dict[str, Any]:
    """
    Migrate old training_metrics.csv to new split format.

    Args:
        input_dir: Directory containing training_metrics.csv
        backup: If True, backup original file before migration

    Returns:
        Dictionary with migration statistics
    """
    old_csv = input_dir / "training_metrics.csv"

    if not old_csv.exists():
        raise FileNotFoundError(f"training_metrics.csv not found in {input_dir}")

    # Check if already migrated (diagnostics_metrics.csv exists)
    diag_csv = input_dir / "diagnostics_metrics.csv"
    if diag_csv.exists():
        print(f"Warning: diagnostics_metrics.csv already exists in {input_dir}")
        print("Skipping migration to avoid data loss. Delete it manually to re-migrate.")
        return {"status": "skipped", "reason": "already_migrated"}

    # Read old CSV
    with open(old_csv, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        old_fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        return {"status": "skipped", "reason": "empty_csv"}

    # Detect per-class columns
    per_class_cols = detect_per_class_columns(old_fieldnames)

    # Check if this is already new format (no diag rows)
    splits = set(r.get("split", "") for r in rows)
    if "diag" not in splits:
        print("CSV appears to be in new format already (no 'diag' split found)")
        return {"status": "skipped", "reason": "new_format"}

    # Backup original if requested
    if backup:
        backup_path = input_dir / "training_metrics.csv.backup"
        shutil.copy2(old_csv, backup_path)
        print(f"Backed up original to: {backup_path}")

    # Separate rows by split type
    training_rows = [r for r in rows if r.get("split") in ("train", "val", "test")]
    diag_rows = [r for r in rows if r.get("split") == "diag"]

    # Prepare fieldnames for new CSVs
    training_fieldnames = list(TRAINING_FIELDS) + per_class_cols
    diagnostic_fieldnames = list(DIAGNOSTIC_FIELDS)

    # Write new training_metrics.csv (overwrite old one)
    with open(old_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=training_fieldnames)
        writer.writeheader()
        for row in training_rows:
            # Filter to only include relevant fields
            filtered_row = {k: row.get(k, "") for k in training_fieldnames}
            writer.writerow(filtered_row)

    # Write new diagnostics_metrics.csv
    with open(diag_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=diagnostic_fieldnames)
        writer.writeheader()
        for row in diag_rows:
            # Filter to only include relevant fields
            filtered_row = {k: row.get(k, "") for k in diagnostic_fieldnames}
            writer.writerow(filtered_row)

    stats = {
        "status": "success",
        "input_dir": str(input_dir),
        "total_rows": len(rows),
        "training_rows": len(training_rows),
        "diagnostic_rows": len(diag_rows),
        "per_class_columns": per_class_cols,
        "training_fields": len(training_fieldnames),
        "diagnostic_fields": len(diagnostic_fieldnames),
        "backup_created": backup,
    }

    print(f"Migration complete:")
    print(f"  - Training rows: {stats['training_rows']} -> training_metrics.csv")
    print(f"  - Diagnostic rows: {stats['diagnostic_rows']} -> diagnostics_metrics.csv")
    print(f"  - Per-class columns preserved: {per_class_cols}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate old training_metrics.csv to new split format"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing training_metrics.csv"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Backup original CSV before migration (default: True)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip backup of original CSV"
    )

    args = parser.parse_args()

    backup = not args.no_backup

    try:
        stats = migrate_csv(args.input_dir, backup=backup)
        if stats["status"] == "success":
            print("\nMigration successful!")
        else:
            print(f"\nMigration skipped: {stats.get('reason', 'unknown')}")
    except Exception as e:
        print(f"Error during migration: {e}")
        raise


if __name__ == "__main__":
    main()
