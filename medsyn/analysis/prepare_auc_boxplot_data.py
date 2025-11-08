#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consolidate per-class AUCs from two cross-validation summaries (Real vs Real+Synth),
reshape them to long format per fold, and compute one-sided Mann–Whitney U tests
(Synth > Real) per class and for overall ("auc_macro"). Output a single CSV
ready for plotting.

CLI:
  python prepare_auc_boxplot_data.py \
      --real-csv /path/cv_per_class_auc_summary_real_only.csv \
      --synth-csv /path/cv_per_class_auc_summary_synth_and_real.csv \
      --out-csv /path/auc_boxplot_data.csv
"""

from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


# Function: significance_from_p
# Purpose: Convert a p-value into significance stars using thresholds [0.05, 0.01, 0.001].
# Arguments:
#   p (float): p-value from a statistical test.
# Returns:
#   str: '', '*', '**', or '***'.
def significance_from_p(p: float) -> str:
    if p <= 0.001:
        return '***'
    if p <= 0.01:
        return '**'
    if p <= 0.05:
        return '*'
    return ''


# Function: find_fold_columns
# Purpose: Identify fold columns in a DataFrame by prefix "fold_".
# Arguments:
#   df (pd.DataFrame): input dataframe with columns like fold_0, fold_1, ...
# Returns:
#   List[str]: ordered list of fold column names.
def find_fold_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("fold_")]
    # Sort folds by their numeric suffix to enforce order
    cols_sorted = sorted(cols, key=lambda c: int(c.split("_")[1]))
    return cols_sorted


# Function: normalize_metric_label
# Purpose: Map metric names to x-axis labels: auc_class_k -> str(k); auc_macro -> 'overall'.
# Arguments:
#   metric (str): raw metric name from CSV.
# Returns:
#   str: normalized label.
def normalize_metric_label(metric: str) -> str:
    if metric == "auc_macro":
        return "overall"
    if metric.startswith("auc_class_"):
        return metric.replace("auc_class_", "")
    return metric


# Function: metric_order_key
# Purpose: Provide numeric ordering for metrics: classes 0..N, then 'overall' last.
# Arguments:
#   label (str): normalized label from normalize_metric_label.
# Returns:
#   Tuple[int, int]: sort key.
def metric_order_key(label: str) -> Tuple[int, int]:
    if label == "overall":
        return (10**6, 0)  # send to the end
    try:
        return (int(label), 0)
    except ValueError:
        return (10**6 - 1, 0)  # unknowns before overall, but after numeric


@dataclass(frozen=True)
class Inputs:
    real_csv: Path
    synth_csv: Path
    out_csv: Path


# Function: compute_mwu_per_metric
# Purpose: Compute one-sided Mann–Whitney U test (Synth > Real) for each metric.
# Arguments:
#   real_df (pd.DataFrame): Real AUC table.
#   synth_df (pd.DataFrame): Synth AUC table.
#   fold_cols (List[str]): columns containing fold-wise AUCs.
# Returns:
#   Dict[str, Tuple[float, str]]: metric -> (p-value, stars).
def compute_mwu_per_metric(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    fold_cols: List[str]
) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    # Align by "metric"
    merged = pd.merge(
        real_df[["metric"] + fold_cols],
        synth_df[["metric"] + fold_cols],
        on="metric",
        suffixes=("_real", "_synth"),
        how="inner",
    )

    for _, row in merged.iterrows():
        metric = row["metric"]
        real_vals = row[[f + "_real" for f in fold_cols]].to_numpy(dtype=float)
        synth_vals = row[[f + "_synth" for f in fold_cols]].to_numpy(dtype=float)

        # Mann–Whitney U test, one-sided alternative: Synth > Real.
        # U = min(Ux, Uy); p is computed under H0 that the distributions are equal.
        # We set alternative="greater" meaning P(X > Y) larger for Synth.
        stat, p = mannwhitneyu(synth_vals, real_vals, alternative="greater")
        out[metric] = (float(p), significance_from_p(float(p)))
    return out


# Function: build_long_table
# Purpose: Reshape the two summary CSVs to a single long table per fold and dataset.
# Arguments:
#   real_df, synth_df (pd.DataFrame): input tables.
#   fold_cols (List[str]): fold columns.
#   mwu (Dict[str, Tuple[float, str]]): per-metric p-values and stars.
# Returns:
#   pd.DataFrame: long table with columns
#       ['metric','label','dataset','fold','auc','pvalue','stars','n'].
def build_long_table(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    fold_cols: List[str],
    mwu: Dict[str, Tuple[float, str]]
) -> pd.DataFrame:

    def melt_one(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        tmp = df[["metric"] + fold_cols].copy()
        tmp = tmp.melt(id_vars="metric", var_name="fold", value_name="auc")
        tmp["dataset"] = dataset_name
        return tmp

    long_real = melt_one(real_df, "Real")
    long_synth = melt_one(synth_df, "Real+Synth")
    long_df = pd.concat([long_real, long_synth], ignore_index=True)

    long_df["label"] = long_df["metric"].map(normalize_metric_label)
    long_df["fold"] = long_df["fold"].str.replace("fold_", "", regex=False).astype(int)

    # Attach per-metric p-values and stars only once per metric; repeat on Synth rows.
    long_df["pvalue"] = np.nan
    long_df["stars"] = ""

    pv_map = {m: pv for m, (pv, _) in mwu.items()}
    st_map = {m: st for m, (_, st) in mwu.items()}

    synth_mask = long_df["dataset"].eq("Real+Synth")
    long_df.loc[synth_mask, "pvalue"] = long_df.loc[synth_mask, "metric"].map(pv_map)
    long_df.loc[synth_mask, "stars"] = long_df.loc[synth_mask, "metric"].map(st_map)

    # n per metric-dataset for bookkeeping
    n_map = (
        long_df.groupby(["metric", "dataset"])["auc"]
        .size()
        .rename("n")
        .reset_index()
    )
    long_df = long_df.merge(n_map, on=["metric", "dataset"], how="left")

    # Sort by metric order, then dataset (Real first)
    long_df["__order__"] = long_df["label"].map(metric_order_key)
    long_df["__ds__"] = long_df["dataset"].map({"Real": 0, "Real+Synth": 1})
    long_df = long_df.sort_values(["__order__", "__ds__", "fold"]).drop(columns=["__order__", "__ds__"])

    return long_df


# Function: read_summary_csv
# Purpose: Read one summary CSV and keep only AUC metrics.
# Arguments:
#   path (Path): input path.
# Returns:
#   pd.DataFrame: filtered and validated dataframe.
def read_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"metric"}
    assert needed.issubset(df.columns), f"Missing required columns {needed} in {path}"
    # Keep auc_class_* and auc_macro
    df = df[df["metric"].str.startswith("auc_class_") | df["metric"].eq("auc_macro")].copy()
    return df


# Function: main
# Purpose: Orchestrate I/O, stats, and CSV export.
# Arguments:
#   args (Inputs): dataclass with paths.
# Returns:
#   None.
def main(args: Inputs) -> None:
    logging.info("Reading inputs")
    real_df = read_summary_csv(args.real_csv)
    synth_df = read_summary_csv(args.synth_csv)

    fold_cols_real = find_fold_columns(real_df)
    fold_cols_synth = find_fold_columns(synth_df)
    assert fold_cols_real == fold_cols_synth, "Fold columns differ between inputs."
    fold_cols = fold_cols_real

    logging.info("Computing one-sided Mann–Whitney U tests (Synth > Real)")
    mwu = compute_mwu_per_metric(real_df, synth_df, fold_cols)

    logging.info("Building long table")
    out_df = build_long_table(real_df, synth_df, fold_cols, mwu)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    logging.info(f"Wrote {args.out_csv} with {len(out_df)} rows")


def parse_args() -> Inputs:
    p = argparse.ArgumentParser(description="Prepare AUC boxplot data with Mann–Whitney U tests.")
    p.add_argument("--real-csv", type=Path, required=True, help="Path to cv_per_class_auc_summary.csv for Real.")
    p.add_argument("--synth-csv", type=Path, required=True, help="Path to cv_per_class_auc_summary.csv for Real+Synth.")
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (e.g., INFO, DEBUG).")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level.upper(), logging.INFO), format="%(levelname)s | %(message)s")
    return Inputs(real_csv=a.real_csv, synth_csv=a.synth_csv, out_csv=a.out_csv)


if __name__ == "__main__":
    main(parse_args())
