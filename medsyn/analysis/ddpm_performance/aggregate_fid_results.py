#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate FID Results from Multiple Hyperparameter Configurations

This script aggregates FID results from multiple CSV files into a single
comprehensive CSV file for comparison between DistDiff and MedSYN configurations.

Usage:
    python aggregate_fid_results.py \\
        --input-dir /path/to/fid_results \\
        --output /path/to/aggregated_fid.csv \\
        --distdiff-path /path/to/distdiff_fid.csv

Example:
    python medsyn/analysis/ddpm_performance/aggregate_fid_results.py \\
        --input-dir /media/mpascual/Sandisk2TB/research/medsyn/results/not_considering_minSNR/fid \\
        --output /media/mpascual/Sandisk2TB/research/medsyn/results/not_considering_minSNR/fidaggregated_fid_comparison.csv \\
        --distdiff-path /media/mpascual/Sandisk2TB/research/medsyn/results/fid_distdiff.csv

Output CSV format:
    Configuration,Class_0_Mean,Class_0_Std,...,Average_Mean,Average_Std
    gamma1_temp10,45.67,1.11,...,42.61,0.92
    ...
    DistDiff,50.23,1.45,...,48.12,1.23
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

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


def extract_config_name(filename: str) -> Optional[str]:
    """
    Extract configuration name from filename.
    
    Expected format:
        1. fid_gamma{X}_temp{Y}.csv -> gamma{X}_temp{Y}
        2. {config}_fid.csv -> {config}
    
    Args:
        filename: Filename to extract config from
        
    Returns:
        Configuration name or None if not matching pattern
    """
    if not filename.endswith(".csv"):
        return None
        
    # Check for new format: {config}_fid.csv
    if filename.endswith("_fid.csv"):
        return filename[:-8]
    
    # Check for old format: fid_{config}.csv
    if filename.startswith("fid_"):
        return filename[4:-4]
        
    return None


def load_fid_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    """
    Load a single FID CSV file and extract metrics.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary with metric names and values, or None if error
    """
    try:
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            logger.warning(f"Empty CSV file: {csv_path}")
            return None
        
        # Get the first (and should be only) row
        row = df.iloc[0]
        
        # Extract all columns except model_name
        metrics = {}
        for col in df.columns:
            if col != "model_name":
                metrics[col] = row[col]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        return None


def aggregate_fid_results(
    input_dir: Path,
    distdiff_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate all FID results from input directory.
    
    Args:
        input_dir: Directory containing FID CSV files
        distdiff_path: Optional path to DistDiff FID CSV file
        
    Returns:
        DataFrame with aggregated results
    """
    all_results = []
    
    # Process all FID CSV files in input directory
    # Support both fid_*.csv and *_fid.csv
    csv_files = sorted(list(input_dir.glob("fid_*.csv")) + list(input_dir.glob("*_fid.csv")))
    # Remove potentially duplicates if patterns overlap
    csv_files = sorted(list(set(csv_files)))
    
    if len(csv_files) == 0:
        logger.warning(f"No FID CSV files found in {input_dir}")
    
    for csv_path in csv_files:
        config_name = extract_config_name(csv_path.name)
        
        if config_name is None:
            logger.warning(f"Skipping file with unexpected name: {csv_path.name}")
            continue
        
        logger.info(f"Processing: {csv_path.name} -> {config_name}")
        
        metrics = load_fid_csv(csv_path)
        
        if metrics is None:
            logger.warning(f"Skipping {csv_path.name} due to loading error")
            continue
        
        # Add configuration name
        metrics["Configuration"] = config_name
        all_results.append(metrics)
    
    # Process DistDiff if provided
    if distdiff_path and distdiff_path.exists():
        logger.info(f"Processing DistDiff: {distdiff_path}")
        
        metrics = load_fid_csv(distdiff_path)
        
        if metrics is not None:
            metrics["Configuration"] = "DistDiff"
            all_results.append(metrics)
        else:
            logger.warning("Could not load DistDiff results")
    elif distdiff_path:
        logger.warning(f"DistDiff file not found: {distdiff_path}")
    
    if len(all_results) == 0:
        raise ValueError("No valid FID results found")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Calculate Deltas if DistDiff is present
    has_distdiff = "DistDiff" in df["Configuration"].values
    if has_distdiff:
        distdiff_row = df[df["Configuration"] == "DistDiff"].iloc[0]
        
        # Find all mean columns
        mean_cols = [col for col in df.columns if col.endswith("_mean")]
        
        for col in mean_cols:
            # Create delta column name: class_0_mean -> class_0_delta
            delta_col = col.replace("_mean", "_delta")
            
            # Calculate delta: Config - DistDiff
            # Negative means Config is better (lower FID)
            df[delta_col] = df[col] - distdiff_row[col]
            
            # Set DistDiff's own delta to 0
            df.loc[df["Configuration"] == "DistDiff", delta_col] = 0.0

    # Reorder columns: Configuration first, then class metrics, then average
    config_col = ["Configuration"]
    
    # Get class columns in order (class_0_mean, class_0_std, class_0_delta, ...)
    class_cols = []
    num_classes = 0
    for col in df.columns:
        if col.startswith("class_") and col.endswith("_mean"):
            class_num = int(col.split("_")[1])
            num_classes = max(num_classes, class_num + 1)
    
    for i in range(num_classes):
        base = f"class_{i}"
        cols_to_add = [f"{base}_mean", f"{base}_std"]
        if has_distdiff:
            cols_to_add.append(f"{base}_delta")
        class_cols.extend(cols_to_add)
    
    # Add average columns
    avg_cols = ["average_mean", "average_std"]
    if has_distdiff:
        avg_cols.append("average_delta")
    
    # Reorder
    # Filter out columns that might not exist
    ordered_cols = config_col + [c for c in class_cols if c in df.columns] + [c for c in avg_cols if c in df.columns]
    
    # Add any other columns that might be there
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining_cols)
    
    df = df[ordered_cols]
    
    # Sort by configuration name (DistDiff last if present)
    df = df.sort_values("Configuration", key=lambda x: x.map(lambda v: "zzz_" + v if v == "DistDiff" else v))
    df = df.reset_index(drop=True)
    
    return df


def format_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format column names for better readability.
    
    Args:
        df: DataFrame with raw column names
        
    Returns:
        DataFrame with formatted column names
    """
    rename_map = {}
    
    for col in df.columns:
        if col == "Configuration":
            continue
        
        # Convert class_0_mean -> Class_0_Mean, class_0_std -> Class_0_Std
        parts = col.split("_")
        if len(parts) >= 2:
            formatted = "_".join([p.capitalize() for p in parts])
            rename_map[col] = formatted
    
    df = df.rename(columns=rename_map)
    return df


def print_summary(df: pd.DataFrame):
    """
    Print summary statistics of aggregated results.
    
    Args:
        df: Aggregated DataFrame
    """
    logger.info("\n" + "=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal configurations: {len(df)}")
    
    # Find average_mean column (might be renamed)
    avg_col = [col for col in df.columns if "average" in col.lower() and "mean" in col.lower()][0]
    
    # Sort by average FID
    df_sorted = df.sort_values(avg_col)
    
    logger.info("\nConfigurations ranked by Average FID:")
    for idx, row in df_sorted.iterrows():
        config = row["Configuration"]
        avg_mean = row[avg_col]
        
        # Find std column
        avg_std_col = [col for col in df.columns if "average" in col.lower() and "std" in col.lower()][0]
        avg_std = row[avg_std_col]
        
        logger.info(f"  {config:20s}: {avg_mean:6.2f} ± {avg_std:.2f}")
    
    # If DistDiff present, compare
    if "DistDiff" in df["Configuration"].values:
        distdiff_row = df[df["Configuration"] == "DistDiff"].iloc[0]
        distdiff_avg = distdiff_row[avg_col]
        
        logger.info("\nComparison to DistDiff:")
        for idx, row in df_sorted.iterrows():
            if row["Configuration"] == "DistDiff":
                continue
            
            config = row["Configuration"]
            avg_mean = row[avg_col]
            diff = avg_mean - distdiff_avg
            pct_change = (diff / distdiff_avg) * 100
            
            if diff < 0:
                logger.info(f"  {config:20s}: {diff:+7.2f} ({pct_change:+6.2f}%) - BETTER")
            else:
                logger.info(f"  {config:20s}: {diff:+7.2f} ({pct_change:+6.2f}%)")


def generate_analysis_report(df: pd.DataFrame, output_path: Path) -> str:
    """
    Generate a detailed analysis report as a text file.
    
    Args:
        df: Aggregated DataFrame with FID results
        output_path: Path to save the analysis report
        
    Returns:
        The report content as a string
    """
    import datetime
    
    lines = []
    lines.append("=" * 80)
    lines.append("FID RESULTS ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Find column names (handle formatted/unformatted)
    avg_mean_col = [col for col in df.columns if "average" in col.lower() and "mean" in col.lower()][0]
    avg_std_col = [col for col in df.columns if "average" in col.lower() and "std" in col.lower()][0]
    
    # Check for delta column
    avg_delta_col = None
    delta_cols = [col for col in df.columns if "average" in col.lower() and "delta" in col.lower()]
    if delta_cols:
        avg_delta_col = delta_cols[0]
    
    # Sort by average FID (lower is better)
    df_sorted = df.sort_values(avg_mean_col).reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # Section 1: Overall Rankings
    # -------------------------------------------------------------------------
    lines.append("-" * 80)
    lines.append("1. OVERALL RANKINGS (by Average FID, lower is better)")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"{'Rank':<6}{'Configuration':<25}{'Avg FID':<15}{'Std':<10}")
    lines.append("-" * 56)
    
    for rank, (idx, row) in enumerate(df_sorted.iterrows(), 1):
        config = row["Configuration"]
        avg_mean = row[avg_mean_col]
        avg_std = row[avg_std_col]
        lines.append(f"{rank:<6}{config:<25}{avg_mean:<15.4f}{avg_std:<10.4f}")
    
    lines.append("")
    
    # -------------------------------------------------------------------------
    # Section 2: Best Configuration
    # -------------------------------------------------------------------------
    lines.append("-" * 80)
    lines.append("2. BEST CONFIGURATION")
    lines.append("-" * 80)
    lines.append("")
    
    best_row = df_sorted.iloc[0]
    best_config = best_row["Configuration"]
    best_avg = best_row[avg_mean_col]
    best_std = best_row[avg_std_col]
    
    lines.append(f"  Configuration: {best_config}")
    lines.append(f"  Average FID:   {best_avg:.4f} ± {best_std:.4f}")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # Section 3: Comparison to DistDiff (if present)
    # -------------------------------------------------------------------------
    has_distdiff = "DistDiff" in df["Configuration"].values
    
    if has_distdiff:
        lines.append("-" * 80)
        lines.append("3. COMPARISON TO DISTDIFF (BASELINE)")
        lines.append("-" * 80)
        lines.append("")
        
        distdiff_row = df[df["Configuration"] == "DistDiff"].iloc[0]
        distdiff_avg = distdiff_row[avg_mean_col]
        distdiff_std = distdiff_row[avg_std_col]
        
        lines.append(f"  DistDiff Average FID: {distdiff_avg:.4f} ± {distdiff_std:.4f}")
        lines.append("")
        
        # Count configurations better than DistDiff
        better_configs = df_sorted[df_sorted[avg_mean_col] < distdiff_avg]
        worse_configs = df_sorted[df_sorted[avg_mean_col] > distdiff_avg]
        
        lines.append(f"  Configurations BETTER than DistDiff: {len(better_configs)}")
        lines.append(f"  Configurations WORSE than DistDiff:  {len(worse_configs)}")
        lines.append("")
        
        lines.append(f"  {'Configuration':<25}{'Delta FID':<15}{'% Change':<15}{'Status':<10}")
        lines.append("  " + "-" * 65)
        
        for idx, row in df_sorted.iterrows():
            config = row["Configuration"]
            if config == "DistDiff":
                continue
            
            avg_mean = row[avg_mean_col]
            delta = avg_mean - distdiff_avg
            pct_change = (delta / distdiff_avg) * 100
            status = "BETTER" if delta < 0 else "WORSE"
            
            lines.append(f"  {config:<25}{delta:+<15.4f}{pct_change:+<15.2f}%{status:<10}")
        
        lines.append("")
        
        # Best improvement over DistDiff
        if len(better_configs) > 0:
            best_improvement_row = better_configs.iloc[0]
            best_improvement_config = best_improvement_row["Configuration"]
            best_improvement_delta = best_improvement_row[avg_mean_col] - distdiff_avg
            best_improvement_pct = (best_improvement_delta / distdiff_avg) * 100
            
            lines.append(f"  BEST IMPROVEMENT over DistDiff:")
            lines.append(f"    Configuration: {best_improvement_config}")
            lines.append(f"    Delta FID:     {best_improvement_delta:+.4f}")
            lines.append(f"    % Change:      {best_improvement_pct:+.2f}%")
        else:
            lines.append("  No configuration outperforms DistDiff.")
        
        lines.append("")
    
    # -------------------------------------------------------------------------
    # Section 4: Per-Class Analysis
    # -------------------------------------------------------------------------
    lines.append("-" * 80)
    lines.append("4. PER-CLASS ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    
    # Find number of classes
    class_mean_cols = [col for col in df.columns if "class" in col.lower() and "mean" in col.lower()]
    num_classes = len(class_mean_cols)
    
    lines.append(f"  Number of classes: {num_classes}")
    lines.append("")
    
    # Best configuration per class
    lines.append("  Best Configuration per Class:")
    lines.append(f"  {'Class':<10}{'Best Config':<25}{'FID':<15}")
    lines.append("  " + "-" * 50)
    
    for i in range(num_classes):
        # Find class column (handle formatted/unformatted names)
        class_col = None
        for col in df.columns:
            if f"class_{i}_mean" in col.lower() or f"class_{i}_mean" == col.lower():
                class_col = col
                break
        
        if class_col is None:
            continue
        
        best_class_idx = df[class_col].idxmin()
        best_class_config = df.loc[best_class_idx, "Configuration"]
        best_class_fid = df.loc[best_class_idx, class_col]
        
        lines.append(f"  {i:<10}{best_class_config:<25}{best_class_fid:<15.4f}")
    
    lines.append("")
    
    # -------------------------------------------------------------------------
    # Section 5: Hyperparameter Insights (gamma and temperature)
    # -------------------------------------------------------------------------
    lines.append("-" * 80)
    lines.append("5. HYPERPARAMETER INSIGHTS")
    lines.append("-" * 80)
    lines.append("")
    
    # Extract gamma and temp from config names
    gamma_results = {}
    temp_results = {}
    
    for idx, row in df.iterrows():
        config = row["Configuration"]
        avg_mean = row[avg_mean_col]
        
        if config == "DistDiff":
            continue
        
        # Parse gamma and temp from config name (e.g., gamma1_temp10)
        try:
            parts = config.split("_")
            gamma = None
            temp = None
            for part in parts:
                if part.startswith("gamma"):
                    gamma = int(part.replace("gamma", ""))
                elif part.startswith("temp"):
                    temp = int(part.replace("temp", ""))
            
            if gamma is not None:
                if gamma not in gamma_results:
                    gamma_results[gamma] = []
                gamma_results[gamma].append(avg_mean)
            
            if temp is not None:
                if temp not in temp_results:
                    temp_results[temp] = []
                temp_results[temp].append(avg_mean)
        except:
            continue
    
    # Average FID by gamma
    if gamma_results:
        lines.append("  Average FID by Gamma (CFG scale):")
        lines.append(f"  {'Gamma':<10}{'Avg FID':<15}{'Num Configs':<15}")
        lines.append("  " + "-" * 40)
        
        for gamma in sorted(gamma_results.keys()):
            fids = gamma_results[gamma]
            avg_fid = sum(fids) / len(fids)
            lines.append(f"  {gamma:<10}{avg_fid:<15.4f}{len(fids):<15}")
        
        best_gamma = min(gamma_results.keys(), key=lambda g: sum(gamma_results[g]) / len(gamma_results[g]))
        lines.append(f"\n  Best Gamma: {best_gamma}")
        lines.append("")
    
    # Average FID by temperature
    if temp_results:
        lines.append("  Average FID by Temperature (τ):")
        lines.append(f"  {'Temp':<10}{'Avg FID':<15}{'Num Configs':<15}")
        lines.append("  " + "-" * 40)
        
        for temp in sorted(temp_results.keys()):
            fids = temp_results[temp]
            avg_fid = sum(fids) / len(fids)
            lines.append(f"  {temp:<10}{avg_fid:<15.4f}{len(fids):<15}")
        
        best_temp = min(temp_results.keys(), key=lambda t: sum(temp_results[t]) / len(temp_results[t]))
        lines.append(f"\n  Best Temperature: {best_temp}")
        lines.append("")
    
    # -------------------------------------------------------------------------
    # Section 6: Summary
    # -------------------------------------------------------------------------
    lines.append("-" * 80)
    lines.append("6. SUMMARY")
    lines.append("-" * 80)
    lines.append("")
    
    lines.append(f"  Total configurations evaluated: {len(df)}")
    lines.append(f"  Best overall configuration:     {best_config}")
    lines.append(f"  Best overall Average FID:       {best_avg:.4f} ± {best_std:.4f}")
    
    if has_distdiff and len(better_configs) > 0:
        lines.append(f"  Improvement over DistDiff:      {best_improvement_delta:+.4f} ({best_improvement_pct:+.2f}%)")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    # Join lines and save
    report_content = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Saved analysis report to {output_path}")
    
    return report_content


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate FID results from multiple hyperparameter configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python aggregate_fid_results.py \\
      --input-dir /media/.../fid_results \\
      --output /media/.../aggregated_fid.csv \\
      --distdiff-path /media/.../fid_distdiff.csv \\
      --verbose
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing FID CSV files (fid_gamma*_temp*.csv)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for aggregated results",
    )

    parser.add_argument(
        "--distdiff-path",
        type=Path,
        default=None,
        help="Optional path to DistDiff FID CSV file",
    )

    parser.add_argument(
        "--no-format",
        action="store_true",
        help="Keep original column names (don't capitalize)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.input_dir.exists():
        parser.error(f"Input directory not found: {args.input_dir}")

    if not args.input_dir.is_dir():
        parser.error(f"Input path is not a directory: {args.input_dir}")

    return args


def main():
    """Main entry point for FID aggregation script."""
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger.info("Starting FID results aggregation")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output}")

    # Aggregate results
    logger.info("Aggregating FID results...")
    df = aggregate_fid_results(
        args.input_dir,
        args.distdiff_path,
    )

    logger.info(f"Successfully aggregated {len(df)} configurations")

    # Format column names if requested
    if not args.no_format:
        logger.info("Formatting column names...")
        df = format_column_names(df)

    # Print summary
    print_summary(df)

    # Save to CSV
    logger.info(f"\nSaving results to: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.4f")
    
    # Generate analysis report
    report_path = args.output.with_suffix('.txt')
    generate_analysis_report(df, report_path)
    
    logger.info("Aggregation completed successfully!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
