#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate FID LaTeX Table from Aggregated CSV Results

This script reads the aggregated FID CSV file and generates a LaTeX table
by filling in placeholders in a template file.

Usage:
    python generate_fid_latex_table.py \\
        --csv /path/to/aggregated_fid.csv \\
        --template /path/to/fid_table_template.tex \\
        --output /path/to/fid_table_filled.tex

Example:
    python medsyn/analysis/ddpm_performance/generate_fid_latex_table.py \\
        --csv /media/mpascual/Sandisk2TB/research/medsyn/results/aggregated_fid_comparison.csv \\
        --template docs/paper-related/fid_table_template.tex \\
        --output docs/paper-related/fid_table_generated.tex

Template placeholders format:
    - {{gamma10_temp10_class0}} -> \fidcell{mean}{std}{color}{arrow}{delta}
    - {{distdiff_class0}} -> \fidbase{mean}{std} or \textbf{\fidbase{mean}{std}}
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the aggregated FID CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with FID results
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} configurations")
    logger.debug(f"Columns: {list(df.columns)}")
    return df


def normalize_column_name(col: str) -> str:
    """
    Normalize column name to lowercase with underscores.
    
    Args:
        col: Column name (e.g., 'Class_0_Mean' or 'class_0_mean')
        
    Returns:
        Normalized column name (e.g., 'class_0_mean')
    """
    return col.lower().replace(" ", "_")


def get_column_value(row: pd.Series, col_pattern: str, df_columns: list) -> Optional[float]:
    """
    Get column value handling different column naming conventions.
    
    Args:
        row: DataFrame row
        col_pattern: Normalized column pattern (e.g., 'class_0_mean')
        df_columns: List of actual DataFrame columns
        
    Returns:
        Column value or None if not found
    """
    for col in df_columns:
        if normalize_column_name(col) == col_pattern:
            return row[col]
    return None


def format_fidcell(
    mean: float,
    std: float,
    delta: float,
    mean_precision: int = 2,
    std_precision: int = 2,
    delta_precision: int = 2,
) -> str:
    """
    Format a FID cell with colored arrow for CFG-MedMNIST-Syn configurations.
    
    Args:
        mean: FID mean value
        std: FID standard deviation
        delta: Delta from DistDiff (negative = better)
        mean_precision: Decimal precision for mean
        std_precision: Decimal precision for std
        delta_precision: Decimal precision for delta
        
    Returns:
        LaTeX string: \\fidcell{mean}{std}{color}{arrow}{delta}
    """
    # Determine color and arrow based on delta
    # Negative delta = our method is better (lower FID) = green down arrow
    # Positive delta = our method is worse (higher FID) = red up arrow
    if delta < 0:
        color = "ForestGreen"
        arrow = r"\downarrow"
        abs_delta = abs(delta)
    else:
        color = "Red"
        arrow = r"\uparrow"
        abs_delta = abs(delta)
    
    return (
        f"\\fidcell{{{mean:.{mean_precision}f}}}"
        f"{{{std:.{std_precision}f}}}"
        f"{{{color}}}"
        f"{{{arrow}}}"
        f"{{{abs_delta:.{delta_precision}f}}}"
    )


def format_fidbase(
    mean: float,
    std: float,
    bold: bool = False,
    mean_precision: int = 2,
    std_precision: int = 2,
) -> str:
    """
    Format a FID base cell for DistDiff (no arrow/delta).
    
    Args:
        mean: FID mean value
        std: FID standard deviation
        bold: Whether to make the cell bold
        mean_precision: Decimal precision for mean
        std_precision: Decimal precision for std
        
    Returns:
        LaTeX string: \\fidbase{mean}{std} or \\textbf{\\fidbase{mean}{std}}
    """
    base = f"\\fidbase{{{mean:.{mean_precision}f}}}{{{std:.{std_precision}f}}}"
    
    if bold:
        return f"\\textbf{{{base}}}"
    return base


def find_best_per_class(df: pd.DataFrame, num_classes: int = 9) -> Dict[str, str]:
    """
    Find which configuration has the best (lowest) FID for each class.
    
    Args:
        df: DataFrame with FID results
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping class to best configuration name
    """
    best_configs = {}
    columns = list(df.columns)
    
    for i in range(num_classes):
        col_pattern = f"class_{i}_mean"
        
        # Find the actual column name
        for col in columns:
            if normalize_column_name(col) == col_pattern:
                best_idx = df[col].idxmin()
                best_configs[f"class_{i}"] = df.loc[best_idx, "Configuration"]
                break
    
    # Also check average
    for col in columns:
        if normalize_column_name(col) == "average_mean":
            best_idx = df[col].idxmin()
            best_configs["avg"] = df.loc[best_idx, "Configuration"]
            break
    
    return best_configs


def generate_placeholder_mapping(
    df: pd.DataFrame,
    num_classes: int = 9,
    bold_best: bool = True,
) -> Dict[str, str]:
    """
    Generate mapping from placeholders to LaTeX cell content.
    
    Args:
        df: DataFrame with FID results
        num_classes: Number of classes (default 9)
        bold_best: Whether to bold the best value per class
        
    Returns:
        Dictionary mapping placeholder names to LaTeX content
    """
    placeholders = {}
    columns = list(df.columns)
    
    # Find best configs per class (for bolding)
    best_per_class = find_best_per_class(df, num_classes) if bold_best else {}
    logger.debug(f"Best per class: {best_per_class}")
    
    # Get DistDiff row for reference
    distdiff_row = df[df["Configuration"] == "DistDiff"].iloc[0] if "DistDiff" in df["Configuration"].values else None
    
    for _, row in df.iterrows():
        config = row["Configuration"]
        
        if config == "DistDiff":
            # DistDiff uses fidbase format
            for i in range(num_classes):
                mean = get_column_value(row, f"class_{i}_mean", columns)
                std = get_column_value(row, f"class_{i}_std", columns)
                
                if mean is not None and std is not None:
                    # Bold if DistDiff is best for this class
                    is_best = best_per_class.get(f"class_{i}") == "DistDiff"
                    placeholder = f"distdiff_class{i}"
                    placeholders[placeholder] = format_fidbase(mean, std, bold=is_best)
            
            # Average column
            mean = get_column_value(row, "average_mean", columns)
            std = get_column_value(row, "average_std", columns)
            
            if mean is not None and std is not None:
                is_best = best_per_class.get("avg") == "DistDiff"
                placeholders["distdiff_avg"] = format_fidbase(mean, std, bold=is_best)
        
        else:
            # CFG-MedMNIST-Syn configurations use fidcell format
            # Parse gamma and temp from config name (e.g., gamma1_temp10)
            match = re.match(r"gamma(\d+)_temp(\d+)", config)
            if not match:
                logger.warning(f"Could not parse config name: {config}")
                continue
            
            gamma = match.group(1)
            temp = match.group(2)
            
            for i in range(num_classes):
                mean = get_column_value(row, f"class_{i}_mean", columns)
                std = get_column_value(row, f"class_{i}_std", columns)
                delta = get_column_value(row, f"class_{i}_delta", columns)
                
                if mean is not None and std is not None and delta is not None:
                    placeholder = f"gamma{gamma}_temp{temp}_class{i}"
                    placeholders[placeholder] = format_fidcell(mean, std, delta)
            
            # Average column
            mean = get_column_value(row, "average_mean", columns)
            std = get_column_value(row, "average_std", columns)
            delta = get_column_value(row, "average_delta", columns)
            
            if mean is not None and std is not None and delta is not None:
                placeholder = f"gamma{gamma}_temp{temp}_avg"
                placeholders[placeholder] = format_fidcell(mean, std, delta)
    
    logger.info(f"Generated {len(placeholders)} placeholder mappings")
    return placeholders


def fill_template(template_content: str, placeholders: Dict[str, str]) -> str:
    """
    Fill template placeholders with actual values.
    
    Args:
        template_content: Template file content with {{placeholder}} markers
        placeholders: Dictionary mapping placeholder names to values
        
    Returns:
        Filled template content
    """
    result = template_content
    
    # Find all placeholders in template
    template_placeholders = set(re.findall(r'\{\{(\w+)\}\}', template_content))
    logger.info(f"Found {len(template_placeholders)} placeholders in template")
    
    # Check for missing placeholders
    missing = template_placeholders - set(placeholders.keys())
    if missing:
        logger.warning(f"Missing placeholders in CSV data: {missing}")
    
    # Check for unused placeholders from CSV
    unused = set(placeholders.keys()) - template_placeholders
    if unused:
        logger.debug(f"Unused placeholders from CSV: {unused}")
    
    # Replace placeholders
    for placeholder, value in placeholders.items():
        pattern = f"{{{{{placeholder}}}}}"
        if pattern in result:
            result = result.replace(pattern, value)
            logger.debug(f"Replaced {placeholder}")
    
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FID LaTeX table from aggregated CSV results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate_fid_latex_table.py \\
      --csv /path/to/aggregated_fid.csv \\
      --template docs/paper-related/fid_table_template.tex \\
      --output docs/paper-related/fid_table_generated.tex
        """,
    )
    
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to aggregated FID CSV file",
    )
    
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Path to LaTeX template file with placeholders",
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for generated LaTeX file",
    )
    
    parser.add_argument(
        "--num-classes",
        type=int,
        default=9,
        help="Number of classes (default: 9)",
    )
    
    parser.add_argument(
        "--no-bold-best",
        action="store_true",
        help="Don't bold the best value per class",
    )
    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.csv.exists():
        parser.error(f"CSV file not found: {args.csv}")
    
    if not args.template.exists():
        parser.error(f"Template file not found: {args.template}")
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("FID LaTeX Table Generator")
    logger.info("=" * 60)
    logger.info(f"CSV file:      {args.csv}")
    logger.info(f"Template file: {args.template}")
    logger.info(f"Output file:   {args.output}")
    logger.info(f"Num classes:   {args.num_classes}")
    
    # Load CSV
    df = load_csv(args.csv)
    
    # Load template
    logger.info("Loading template...")
    with open(args.template, 'r') as f:
        template_content = f.read()
    
    # Generate placeholder mapping
    logger.info("Generating placeholder mappings...")
    placeholders = generate_placeholder_mapping(
        df,
        num_classes=args.num_classes,
        bold_best=not args.no_bold_best,
    )
    
    # Fill template
    logger.info("Filling template...")
    filled_content = fill_template(template_content, placeholders)
    
    # Save output
    logger.info(f"Saving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(filled_content)
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
