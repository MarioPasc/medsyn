#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Downstream Classification Significance Analysis Module

This module performs statistical analysis to answer two key questions:
1. Q1 (Global): Does cfg-MedSyn significantly outperform other regimes on a primary metric?
2. Q2 (Per-class): For which classes does cfg-MedSyn significantly improve F1-score?

Statistical methods:
- Q1: Paired t-tests per backbone with Holm-Bonferroni correction
- Q2: One-sample t-tests aggregated across backbones with Benjamini-Hochberg FDR correction

Usage:
    python -m medsyn.analysis.classification.downstream_significance \\
        --root-results-dir /path/to/results/classification \\
        --dataset pathmnist \\
        --primary-metric f1 \\
        --backbones resnet50 clip_vit_b16 wideresnet28_10 \\
        --experiments real_only real_plus_trad_aug real_plus_synth_cfgmedsyn real_plus_synth_distdiff \\
        --our-experiment real_plus_synth_cfgmedsyn \\
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel, ttest_1samp, t as t_dist
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class ExperimentNotFoundError(Exception):
    """Raised when an experiment directory cannot be located or is ambiguous."""
    pass


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DownstreamAnalysisConfig:
    """Configuration for downstream classification analysis.

    Attributes:
        root_results_dir: Root directory containing all classification experiments
        dataset_name: Dataset name (e.g., 'pathmnist')
        primary_metric: Primary metric for Q1 analysis (f1, auc, loss, precision, recall)
        backbones: List of backbone/model names
        experiments: List of experiment names (augmentation regimes)
        our_experiment_name: Name of cfg-MedSyn experiment (usually 'real_plus_synth_cfgmedsyn')
        output_dir: Directory for saving outputs
    """
    root_results_dir: Path
    dataset_name: str
    primary_metric: str
    backbones: List[str]
    experiments: List[str]
    our_experiment_name: str
    output_dir: Path

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.root_results_dir = Path(self.root_results_dir)
        self.output_dir = Path(self.output_dir)

        valid_metrics = {"f1", "auc", "loss", "precision", "recall"}
        if self.primary_metric not in valid_metrics:
            raise ValueError(
                f"primary_metric must be one of {valid_metrics}, got '{self.primary_metric}'"
            )

        if self.our_experiment_name not in self.experiments:
            raise ValueError(
                f"our_experiment_name '{self.our_experiment_name}' not found in experiments list"
            )

        if not self.root_results_dir.exists():
            raise FileNotFoundError(f"root_results_dir does not exist: {self.root_results_dir}")


# ============================================================================
# Experiment Discovery
# ============================================================================

class ExperimentLocator:
    """Locates experiment directories based on dataset, backbone, and experiment name.

    Handles naming variations:
    1. Try exact: {dataset}_{backbone}_{experiment}
    2. Try without dataset: {backbone}_{experiment}
    3. Fall back to substring matching
    """

    def __init__(self, root_dir: Path):
        """Initialize the locator.

        Args:
            root_dir: Root directory containing experiment subdirectories
        """
        self.root_dir = Path(root_dir)
        self.logger = logging.getLogger(f"{__name__}.ExperimentLocator")

    def locate(
        self,
        dataset: str,
        backbone: str,
        experiment: str
    ) -> Path:
        """Locate the experiment directory.

        Args:
            dataset: Dataset name
            backbone: Backbone/model name
            experiment: Experiment/regime name

        Returns:
            Path to the experiment directory

        Raises:
            ExperimentNotFoundError: If no unique match is found
        """
        # Strategy 1: Exact match with dataset
        exact_with_dataset = f"{dataset}_{backbone}_{experiment}"
        candidate = self.root_dir / exact_with_dataset
        if candidate.exists() and candidate.is_dir():
            self.logger.info(f"Found experiment: {exact_with_dataset}")
            return candidate

        # Strategy 2: Exact match without dataset
        exact_without_dataset = f"{backbone}_{experiment}"
        candidate = self.root_dir / exact_without_dataset
        if candidate.exists() and candidate.is_dir():
            self.logger.info(f"Found experiment: {exact_without_dataset}")
            return candidate

        # Strategy 3: Substring matching
        self.logger.debug(
            f"Exact matches failed for {dataset}/{backbone}/{experiment}, "
            "trying substring matching..."
        )
        matches = []
        for subdir in self.root_dir.iterdir():
            if not subdir.is_dir():
                continue
            name_lower = subdir.name.lower()
            if (dataset.lower() in name_lower and
                backbone.lower() in name_lower and
                experiment.lower() in name_lower):
                matches.append(subdir)

        if len(matches) == 0:
            raise ExperimentNotFoundError(
                f"No experiment directory found for dataset='{dataset}', "
                f"backbone='{backbone}', experiment='{experiment}' in {self.root_dir}"
            )
        elif len(matches) > 1:
            match_names = [m.name for m in matches]
            raise ExperimentNotFoundError(
                f"Multiple experiment directories found for dataset='{dataset}', "
                f"backbone='{backbone}', experiment='{experiment}': {match_names}"
            )

        self.logger.info(f"Found experiment via substring match: {matches[0].name}")
        return matches[0]

    def locate_all(
        self,
        dataset: str,
        backbones: List[str],
        experiments: List[str]
    ) -> Dict[Tuple[str, str], Path]:
        """Locate all experiment directories for given backbones and experiments.

        Args:
            dataset: Dataset name
            backbones: List of backbone names
            experiments: List of experiment names

        Returns:
            Dictionary mapping (backbone, experiment) tuples to directory paths
        """
        mapping = {}
        for backbone in backbones:
            for experiment in experiments:
                try:
                    path = self.locate(dataset, backbone, experiment)
                    mapping[(backbone, experiment)] = path
                except ExperimentNotFoundError as e:
                    self.logger.error(str(e))
                    raise

        self.logger.info(
            f"Located {len(mapping)} experiment directories "
            f"({len(backbones)} backbones × {len(experiments)} experiments)"
        )
        return mapping


# ============================================================================
# Data Collection
# ============================================================================

class ClassificationResultsCollector:
    """Collects and organizes classification results from test_results.yaml files."""

    METRIC_KEY_MAPPING = {
        "f1": "f1",
        "auc": "auc_macro",
        "loss": "loss",
        "precision": "precision",
        "recall": "recall"
    }

    def __init__(self):
        """Initialize the collector."""
        self.logger = logging.getLogger(f"{__name__}.ClassificationResultsCollector")

    def _find_folds(self, exp_dir: Path) -> List[int]:
        """Find all fold subdirectories in an experiment directory.

        Args:
            exp_dir: Experiment directory path

        Returns:
            List of fold indices (sorted)
        """
        folds = []
        for subdir in exp_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("fold_"):
                try:
                    fold_idx = int(subdir.name.split("_")[1])
                    folds.append(fold_idx)
                except (IndexError, ValueError):
                    self.logger.warning(f"Skipping invalid fold directory: {subdir.name}")
        return sorted(folds)

    def _read_test_results(self, fold_dir: Path) -> dict:
        """Read test_results.yaml from a fold directory.

        Args:
            fold_dir: Fold directory path

        Returns:
            Parsed YAML content as dictionary

        Raises:
            FileNotFoundError: If test_results.yaml doesn't exist
        """
        yaml_path = fold_dir / "logs" / "test_results.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"test_results.yaml not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def collect_global_metrics(
        self,
        config: DownstreamAnalysisConfig
    ) -> pd.DataFrame:
        """Collect global metrics for all experiments, backbones, and folds.

        Args:
            config: Analysis configuration

        Returns:
            DataFrame with columns: dataset, backbone, experiment, fold,
                                   metric_name, metric_value
        """
        self.logger.info("Collecting global metrics...")

        locator = ExperimentLocator(config.root_results_dir)
        exp_mapping = locator.locate_all(
            config.dataset_name,
            config.backbones,
            config.experiments
        )

        # Determine the YAML key for the primary metric
        metric_key = self.METRIC_KEY_MAPPING[config.primary_metric]

        rows = []
        for (backbone, experiment), exp_dir in exp_mapping.items():
            folds = self._find_folds(exp_dir)

            for fold_idx in folds:
                fold_dir = exp_dir / f"fold_{fold_idx}"
                try:
                    results = self._read_test_results(fold_dir)

                    if metric_key not in results:
                        self.logger.warning(
                            f"Metric '{metric_key}' not found in {fold_dir}/"
                            f"logs/test_results.yaml, skipping"
                        )
                        continue

                    metric_value = float(results[metric_key])

                    rows.append({
                        "dataset": config.dataset_name,
                        "backbone": backbone,
                        "experiment": experiment,
                        "fold": fold_idx,
                        "metric_name": config.primary_metric,
                        "metric_value": metric_value
                    })

                except Exception as e:
                    self.logger.error(
                        f"Error reading results for {backbone}/{experiment}/fold_{fold_idx}: {e}"
                    )
                    raise

        df = pd.DataFrame(rows)
        self.logger.info(
            f"Collected {len(df)} global metric records "
            f"({len(config.backbones)} backbones × {len(config.experiments)} experiments × "
            f"~{len(df) // (len(config.backbones) * len(config.experiments))} folds)"
        )

        # Save raw data
        config.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = config.output_dir / "global_metrics_raw.csv"
        json_path = config.output_dir / "global_metrics_raw.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        self.logger.info(f"Saved global metrics to {csv_path} and {json_path}")

        return df

    def collect_per_class_f1(
        self,
        config: DownstreamAnalysisConfig
    ) -> pd.DataFrame:
        """Collect per-class F1 scores for all experiments, backbones, and folds.

        Args:
            config: Analysis configuration

        Returns:
            DataFrame with columns: dataset, backbone, experiment, fold,
                                   class_index, metric_name, metric_value
        """
        self.logger.info("Collecting per-class F1 scores...")

        locator = ExperimentLocator(config.root_results_dir)
        exp_mapping = locator.locate_all(
            config.dataset_name,
            config.backbones,
            config.experiments
        )

        rows = []
        for (backbone, experiment), exp_dir in exp_mapping.items():
            folds = self._find_folds(exp_dir)

            for fold_idx in folds:
                fold_dir = exp_dir / f"fold_{fold_idx}"
                try:
                    results = self._read_test_results(fold_dir)

                    if "per_class" not in results or "f1" not in results["per_class"]:
                        self.logger.warning(
                            f"per_class.f1 not found in {fold_dir}/logs/test_results.yaml, skipping"
                        )
                        continue

                    f1_scores = results["per_class"]["f1"]
                    if not isinstance(f1_scores, list):
                        self.logger.warning(
                            f"per_class.f1 is not a list in {fold_dir}/logs/test_results.yaml, skipping"
                        )
                        continue

                    for class_idx, f1_value in enumerate(f1_scores):
                        rows.append({
                            "dataset": config.dataset_name,
                            "backbone": backbone,
                            "experiment": experiment,
                            "fold": fold_idx,
                            "class_index": class_idx,
                            "metric_name": "f1",
                            "metric_value": float(f1_value)
                        })

                except Exception as e:
                    self.logger.error(
                        f"Error reading per-class results for {backbone}/{experiment}/"
                        f"fold_{fold_idx}: {e}"
                    )
                    raise

        df = pd.DataFrame(rows)
        num_classes = df["class_index"].nunique() if len(df) > 0 else 0
        self.logger.info(
            f"Collected {len(df)} per-class F1 records "
            f"({len(config.backbones)} backbones × {len(config.experiments)} experiments × "
            f"~{len(df) // (len(config.backbones) * len(config.experiments) * num_classes)} folds × "
            f"{num_classes} classes)"
        )

        # Save raw data
        config.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = config.output_dir / "per_class_f1_raw.csv"
        json_path = config.output_dir / "per_class_f1_raw.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        self.logger.info(f"Saved per-class F1 to {csv_path} and {json_path}")

        return df


# ============================================================================
# Statistical Analysis - Q1 (Global Metric)
# ============================================================================

def analyse_global_metric(
    global_df: pd.DataFrame,
    config: DownstreamAnalysisConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze global metric differences between cfg-MedSyn and baselines (Q1).

    For each backbone:
    - Compute summary statistics per experiment
    - Perform paired t-tests comparing cfg-MedSyn vs each baseline
    - Apply Holm-Bonferroni correction per backbone

    Args:
        global_df: DataFrame from collect_global_metrics
        config: Analysis configuration

    Returns:
        Tuple of (experiment_summary_df, pairwise_diff_df)
    """
    logger.info("Analyzing global metric (Q1)...")

    # Compute experiment summaries
    experiment_summaries = []

    for backbone in config.backbones:
        for experiment in config.experiments:
            subset = global_df[
                (global_df["backbone"] == backbone) &
                (global_df["experiment"] == experiment)
            ]

            if len(subset) == 0:
                logger.warning(f"No data for {backbone}/{experiment}")
                continue

            values = subset["metric_value"].values
            n = len(values)
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)

            # 95% CI
            t_crit = t_dist.ppf(0.975, n - 1)
            se = std_val / np.sqrt(n)
            ci_lower = mean_val - t_crit * se
            ci_upper = mean_val + t_crit * se

            experiment_summaries.append({
                "dataset": config.dataset_name,
                "backbone": backbone,
                "experiment": experiment,
                "primary_metric": config.primary_metric,
                "mean_value": mean_val,
                "std_value": std_val,
                "n_folds": n,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            })

    experiment_summary_df = pd.DataFrame(experiment_summaries)

    # Pairwise comparisons
    baselines = [exp for exp in config.experiments if exp != config.our_experiment_name]
    pairwise_diffs = []

    for backbone in config.backbones:
        # Extract cfg-MedSyn values
        our_subset = global_df[
            (global_df["backbone"] == backbone) &
            (global_df["experiment"] == config.our_experiment_name)
        ].sort_values("fold")

        if len(our_subset) == 0:
            logger.warning(f"No data for {backbone}/{config.our_experiment_name}")
            continue

        our_values = our_subset["metric_value"].values

        # Compare against each baseline
        comparisons = []
        for baseline_exp in baselines:
            baseline_subset = global_df[
                (global_df["backbone"] == backbone) &
                (global_df["experiment"] == baseline_exp)
            ].sort_values("fold")

            if len(baseline_subset) == 0:
                logger.warning(f"No data for {backbone}/{baseline_exp}")
                continue

            baseline_values = baseline_subset["metric_value"].values

            if len(our_values) != len(baseline_values):
                logger.warning(
                    f"Fold count mismatch for {backbone}: "
                    f"{config.our_experiment_name}={len(our_values)} vs "
                    f"{baseline_exp}={len(baseline_values)}, skipping"
                )
                continue

            # Paired t-test
            differences = our_values - baseline_values
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)

            t_stat, p_twosided = ttest_rel(our_values, baseline_values)
            # Convert to one-sided: H1: our > baseline
            p_onesided = p_twosided / 2 if t_stat > 0 else 1 - p_twosided / 2

            # 95% CI for the difference
            t_crit = t_dist.ppf(0.975, n - 1)
            se = std_diff / np.sqrt(n)
            ci_lower = mean_diff - t_crit * se
            ci_upper = mean_diff + t_crit * se

            # Cohen's d
            cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan

            comparisons.append({
                "dataset": config.dataset_name,
                "backbone": backbone,
                "primary_metric": config.primary_metric,
                "our_experiment": config.our_experiment_name,
                "baseline_experiment": baseline_exp,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "n_folds": n,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "t_statistic": t_stat,
                "p_value_raw": p_onesided,
                "cohens_d": cohens_d,
                "test_type": "paired_t_one_sided"
            })

        # Apply Holm-Bonferroni correction per backbone
        if len(comparisons) > 0:
            p_values = [c["p_value_raw"] for c in comparisons]
            _, p_adj, _, _ = multipletests(p_values, method='holm')

            for i, comp in enumerate(comparisons):
                comp["p_value_holm"] = p_adj[i]
                pairwise_diffs.append(comp)

    pairwise_diff_df = pd.DataFrame(pairwise_diffs)

    logger.info(
        f"Computed {len(experiment_summary_df)} experiment summaries and "
        f"{len(pairwise_diff_df)} pairwise comparisons"
    )

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = config.output_dir / "global_experiment_summary.csv"
    summary_json = config.output_dir / "global_experiment_summary.json"
    experiment_summary_df.to_csv(summary_csv, index=False)
    experiment_summary_df.to_json(summary_json, orient="records", indent=2)
    logger.info(f"Saved experiment summaries to {summary_csv}")

    diff_csv = config.output_dir / "global_pairwise_differences.csv"
    diff_json = config.output_dir / "global_pairwise_differences.json"
    pairwise_diff_df.to_csv(diff_csv, index=False)
    pairwise_diff_df.to_json(diff_json, orient="records", indent=2)
    logger.info(f"Saved pairwise differences to {diff_csv}")

    return experiment_summary_df, pairwise_diff_df


# ============================================================================
# Statistical Analysis - Q2 (Per-class F1)
# ============================================================================

def analyse_per_class_f1(
    per_class_df: pd.DataFrame,
    config: DownstreamAnalysisConfig
) -> pd.DataFrame:
    """Analyze per-class F1 differences between cfg-MedSyn and baselines (Q2).

    For each class and baseline:
    - Aggregate differences across all backbone×fold pairs
    - Perform one-sample t-test against 0
    - Apply Benjamini-Hochberg FDR correction across all tests

    Args:
        per_class_df: DataFrame from collect_per_class_f1
        config: Analysis configuration

    Returns:
        DataFrame with per-class statistics
    """
    logger.info("Analyzing per-class F1 scores (Q2)...")

    baselines = [exp for exp in config.experiments if exp != config.our_experiment_name]
    num_classes = per_class_df["class_index"].nunique()

    # Compute per-class summaries (for context)
    class_summaries = []
    for class_idx in sorted(per_class_df["class_index"].unique()):
        for experiment in config.experiments:
            subset = per_class_df[
                (per_class_df["class_index"] == class_idx) &
                (per_class_df["experiment"] == experiment)
            ]

            if len(subset) > 0:
                values = subset["metric_value"].values
                class_summaries.append({
                    "dataset": config.dataset_name,
                    "class_index": class_idx,
                    "experiment": experiment,
                    "mean_f1": np.mean(values),
                    "std_f1": np.std(values, ddof=1),
                    "n_samples": len(values)
                })

    class_summary_df = pd.DataFrame(class_summaries)

    # Per-class pairwise differences
    class_diffs = []

    for class_idx in sorted(per_class_df["class_index"].unique()):
        # Extract cfg-MedSyn values
        our_subset = per_class_df[
            (per_class_df["class_index"] == class_idx) &
            (per_class_df["experiment"] == config.our_experiment_name)
        ]

        if len(our_subset) == 0:
            logger.warning(
                f"No data for class {class_idx}/{config.our_experiment_name}"
            )
            continue

        for baseline_exp in baselines:
            baseline_subset = per_class_df[
                (per_class_df["class_index"] == class_idx) &
                (per_class_df["experiment"] == baseline_exp)
            ]

            if len(baseline_subset) == 0:
                logger.warning(f"No data for class {class_idx}/{baseline_exp}")
                continue

            # Merge on backbone and fold to get paired differences
            merged = pd.merge(
                our_subset[["backbone", "fold", "metric_value"]],
                baseline_subset[["backbone", "fold", "metric_value"]],
                on=["backbone", "fold"],
                suffixes=("_our", "_baseline")
            )

            if len(merged) == 0:
                logger.warning(
                    f"No paired data for class {class_idx}: "
                    f"{config.our_experiment_name} vs {baseline_exp}"
                )
                continue

            differences = (merged["metric_value_our"] - merged["metric_value_baseline"]).values
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)

            # One-sample t-test: H0: mean_diff = 0, H1: mean_diff > 0
            t_stat, p_twosided = ttest_1samp(differences, 0)
            p_onesided = p_twosided / 2 if t_stat > 0 else 1 - p_twosided / 2

            # 95% CI for the difference
            t_crit = t_dist.ppf(0.975, n - 1)
            se = std_diff / np.sqrt(n)
            ci_lower = mean_diff - t_crit * se
            ci_upper = mean_diff + t_crit * se

            # Cohen's d
            cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan

            class_diffs.append({
                "dataset": config.dataset_name,
                "class_index": class_idx,
                "our_experiment": config.our_experiment_name,
                "baseline_experiment": baseline_exp,
                "n_pairs": n,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "t_statistic": t_stat,
                "p_value_raw": p_onesided,
                "cohens_d": cohens_d,
                "test_type": "one_sample_t_one_sided"
            })

    per_class_stats_df = pd.DataFrame(class_diffs)

    # Apply Benjamini-Hochberg FDR correction across all tests
    if len(per_class_stats_df) > 0:
        p_values = per_class_stats_df["p_value_raw"].values
        _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
        per_class_stats_df["p_value_fdr"] = p_adj
    else:
        per_class_stats_df["p_value_fdr"] = []

    logger.info(
        f"Computed {len(class_summary_df)} per-class summaries and "
        f"{len(per_class_stats_df)} per-class comparisons "
        f"({num_classes} classes × {len(baselines)} baselines)"
    )

    # Save results
    config.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = config.output_dir / "per_class_f1_summary.csv"
    summary_json = config.output_dir / "per_class_f1_summary.json"
    class_summary_df.to_csv(summary_csv, index=False)
    class_summary_df.to_json(summary_json, orient="records", indent=2)
    logger.info(f"Saved per-class F1 summaries to {summary_csv}")

    stats_csv = config.output_dir / "per_class_f1_differences.csv"
    stats_json = config.output_dir / "per_class_f1_differences.json"
    per_class_stats_df.to_csv(stats_csv, index=False)
    per_class_stats_df.to_json(stats_json, orient="records", indent=2)
    logger.info(f"Saved per-class F1 differences to {stats_csv}")

    return per_class_stats_df


# ============================================================================
# Visualization - Figure 1 (Q1: Global Metric)
# ============================================================================

def plot_global_metric(
    experiment_summary_df: pd.DataFrame,
    pairwise_diff_df: pd.DataFrame,
    config: DownstreamAnalysisConfig
) -> None:
    """Create Figure 1: Global metric comparison across experiments and backbones.

    Args:
        experiment_summary_df: Output from analyse_global_metric
        pairwise_diff_df: Output from analyse_global_metric
        config: Analysis configuration
    """
    logger.info("Creating Figure 1 (global metric)...")

    backbones = config.backbones
    experiments = config.experiments

    fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), 5))
    if len(backbones) == 1:
        axes = [axes]

    # Define experiment display order and labels
    exp_order = experiments
    exp_labels = {
        "real_only": "M0: Real only",
        "real_plus_trad_aug": "M1: Real+TradAug",
        "real_plus_synth_cfgmedsyn": "M2: Real+cfg-MedSyn",
        "real_plus_synth_distdiff": "M3: Real+DistDiff"
    }

    for ax_idx, backbone in enumerate(backbones):
        ax = axes[ax_idx]

        # Extract data for this backbone
        summary_subset = experiment_summary_df[
            experiment_summary_df["backbone"] == backbone
        ]

        # Plot each experiment
        x_positions = []
        means = []
        cis_lower = []
        cis_upper = []
        colors = []

        for i, exp in enumerate(exp_order):
            exp_data = summary_subset[summary_subset["experiment"] == exp]
            if len(exp_data) == 0:
                continue

            x_positions.append(i)
            means.append(exp_data["mean_value"].values[0])
            cis_lower.append(exp_data["ci_lower"].values[0])
            cis_upper.append(exp_data["ci_upper"].values[0])

            # Highlight cfg-MedSyn
            if exp == config.our_experiment_name:
                colors.append('red')
            else:
                colors.append('steelblue')

        # Plot points with error bars
        for i, (x, mean, ci_l, ci_u, color) in enumerate(
            zip(x_positions, means, cis_lower, cis_upper, colors)
        ):
            marker = 'D' if exp_order[x] == config.our_experiment_name else 'o'
            markersize = 10 if exp_order[x] == config.our_experiment_name else 7
            ax.errorbar(
                x, mean,
                yerr=[[mean - ci_l], [ci_u - mean]],
                fmt=marker,
                color=color,
                markersize=markersize,
                capsize=5,
                capthick=2,
                linewidth=2
            )

        # Annotate significance
        diff_subset = pairwise_diff_df[pairwise_diff_df["backbone"] == backbone]

        y_max = max(cis_upper) if cis_upper else 1.0
        y_offset = (y_max - min(cis_lower if cis_lower else [0])) * 0.05
        y_text = y_max + y_offset

        annotation_lines = []
        for _, row in diff_subset.iterrows():
            baseline = row["baseline_experiment"]
            mean_diff = row["mean_diff"]
            ci_l = row["ci_lower"]
            ci_u = row["ci_upper"]
            p_holm = row["p_value_holm"]
            cohens_d = row["cohens_d"]

            sig_marker = "***" if p_holm < 0.001 else "**" if p_holm < 0.01 else "*" if p_holm < 0.05 else "ns"

            baseline_label = exp_labels.get(baseline, baseline)
            annotation_lines.append(
                f"vs {baseline_label}: Δ={mean_diff:.3f} [{ci_l:.3f}, {ci_u:.3f}], "
                f"p={p_holm:.4f} ({sig_marker}), d={cohens_d:.2f}"
            )

        if annotation_lines:
            annotation_text = "\n".join(annotation_lines)
            ax.text(
                0.5, 0.98,
                annotation_text,
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
            )

        # Formatting
        ax.set_xticks(range(len(exp_order)))
        ax.set_xticklabels([exp_labels.get(e, e) for e in exp_order], rotation=45, ha='right')
        ax.set_ylabel(f"Test {config.primary_metric.upper()} (mean ± 95% CI)")
        ax.set_title(f"{backbone}")
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f"Q1: Global {config.primary_metric.upper()} Comparison\n"
        f"(Paired t-tests with Holm correction per backbone)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()

    # Save figure
    fig_path_png = config.output_dir / f"figure_Q1_global_{config.primary_metric}.png"
    fig_path_pdf = config.output_dir / f"figure_Q1_global_{config.primary_metric}.pdf"
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Figure 1 to {fig_path_png} and {fig_path_pdf}")


# ============================================================================
# Visualization - Figure 2 (Q2: Per-class F1)
# ============================================================================

def plot_per_class_improvement(
    per_class_stats_df: pd.DataFrame,
    config: DownstreamAnalysisConfig
) -> None:
    """Create Figure 2: Per-class F1 improvement heatmap.

    Args:
        per_class_stats_df: Output from analyse_per_class_f1
        config: Analysis configuration
    """
    logger.info("Creating Figure 2 (per-class F1 improvement)...")

    baselines = [exp for exp in config.experiments if exp != config.our_experiment_name]
    num_classes = per_class_stats_df["class_index"].nunique()

    # Create pivot table: rows = classes, columns = baselines, values = mean_diff
    pivot_data = per_class_stats_df.pivot_table(
        index="class_index",
        columns="baseline_experiment",
        values="mean_diff",
        aggfunc="first"
    )

    pivot_sig = per_class_stats_df.pivot_table(
        index="class_index",
        columns="baseline_experiment",
        values="p_value_fdr",
        aggfunc="first"
    )

    # Reorder columns
    pivot_data = pivot_data[baselines]
    pivot_sig = pivot_sig[baselines]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine color scale limits (symmetric around 0)
    vmax = np.abs(pivot_data.values).max()
    vmin = -vmax

    im = ax.imshow(
        pivot_data.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean ΔF1 (cfg-MedSyn - baseline)", rotation=270, labelpad=20)

    # Set ticks and labels
    ax.set_xticks(range(len(baselines)))
    baseline_labels = {
        "real_only": "M0: Real only",
        "real_plus_trad_aug": "M1: Real+TradAug",
        "real_plus_synth_distdiff": "M3: Real+DistDiff"
    }
    ax.set_xticklabels([baseline_labels.get(b, b) for b in baselines], rotation=45, ha='right')

    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([f"Class {i}" for i in pivot_data.index])

    # Annotate cells with mean_diff and significance stars
    for i, class_idx in enumerate(pivot_data.index):
        for j, baseline in enumerate(baselines):
            mean_diff = pivot_data.loc[class_idx, baseline]
            p_fdr = pivot_sig.loc[class_idx, baseline]

            if pd.notna(mean_diff):
                # Determine text color based on background
                text_color = 'white' if abs(mean_diff) > vmax * 0.5 else 'black'

                # Add value
                ax.text(
                    j, i, f"{mean_diff:.3f}",
                    ha='center', va='center',
                    color=text_color,
                    fontsize=9
                )

                # Add significance marker
                if pd.notna(p_fdr) and p_fdr < 0.05:
                    ax.text(
                        j, i - 0.3, '*',
                        ha='center', va='center',
                        color='yellow',
                        fontsize=16,
                        fontweight='bold'
                    )

    ax.set_xlabel("Baseline regime compared to cfg-MedSyn")
    ax.set_ylabel("PathMNIST class index")
    ax.set_title(
        "Q2: Per-class F1 Improvement (cfg-MedSyn vs Baselines)\n"
        "* indicates p_FDR < 0.05 (one-sample t-test, BH correction)",
        fontsize=12,
        fontweight='bold'
    )

    plt.tight_layout()

    # Save figure
    fig_path_png = config.output_dir / "figure_Q2_per_class_f1_improvement.png"
    fig_path_pdf = config.output_dir / "figure_Q2_per_class_f1_improvement.pdf"
    plt.savefig(fig_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved Figure 2 to {fig_path_png} and {fig_path_pdf}")


# ============================================================================
# Metadata
# ============================================================================

def write_metadata(config: DownstreamAnalysisConfig) -> None:
    """Write analysis metadata to JSON file.

    Args:
        config: Analysis configuration
    """
    logger.info("Writing metadata...")

    import sys

    metadata = {
        "dataset": config.dataset_name,
        "primary_metric": config.primary_metric,
        "backbones": config.backbones,
        "experiments": config.experiments,
        "our_experiment_name": config.our_experiment_name,
        "root_results_dir": str(config.root_results_dir),
        "output_dir": str(config.output_dir),
        "library_versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": __import__('scipy').__version__,
            "statsmodels": __import__('statsmodels').__version__,
            "matplotlib": matplotlib.__version__,
        },
        "statistical_methods": {
            "Q1": {
                "description": "Paired t-test per backbone, one-sided (cfg-MedSyn > baseline)",
                "correction": "Holm-Bonferroni per backbone (3 comparisons)",
                "effect_size": "Cohen's d = mean_diff / std_diff"
            },
            "Q2": {
                "description": "One-sample t-test per class, one-sided (mean_diff > 0)",
                "aggregation": "Pooled across backbones and folds",
                "correction": "Benjamini-Hochberg FDR across all 27 tests (9 classes × 3 baselines)",
                "effect_size": "Cohen's d = mean_diff / std_diff"
            },
            "confidence_intervals": "95% CI using t-distribution"
        },
        "output_files": {
            "global_metrics_raw": "global_metrics_raw.csv/.json",
            "per_class_f1_raw": "per_class_f1_raw.csv/.json",
            "global_experiment_summary": "global_experiment_summary.csv/.json",
            "global_pairwise_differences": "global_pairwise_differences.csv/.json",
            "per_class_f1_summary": "per_class_f1_summary.csv/.json",
            "per_class_f1_differences": "per_class_f1_differences.csv/.json",
            "figure_Q1": f"figure_Q1_global_{config.primary_metric}.png/.pdf",
            "figure_Q2": "figure_Q2_per_class_f1_improvement.png/.pdf"
        }
    }

    metadata_path = config.output_dir / "analysis_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved metadata to {metadata_path}")


# ============================================================================
# Orchestration
# ============================================================================

def run_full_analysis(config: DownstreamAnalysisConfig) -> None:
    """Run the complete downstream classification analysis pipeline.

    Args:
        config: Analysis configuration
    """
    logger.info("="*80)
    logger.info("Starting Downstream Classification Significance Analysis")
    logger.info("="*80)
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Primary metric: {config.primary_metric}")
    logger.info(f"Backbones: {config.backbones}")
    logger.info(f"Experiments: {config.experiments}")
    logger.info(f"Our experiment: {config.our_experiment_name}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info("="*80)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect data
    logger.info("\n### STEP 1: Data Collection ###")
    collector = ClassificationResultsCollector()
    global_df = collector.collect_global_metrics(config)
    per_class_df = collector.collect_per_class_f1(config)

    # Step 2: Analyze Q1 (Global metric)
    logger.info("\n### STEP 2: Analyze Q1 (Global Metric) ###")
    experiment_summary_df, pairwise_diff_df = analyse_global_metric(global_df, config)

    # Step 3: Visualize Q1
    logger.info("\n### STEP 3: Visualize Q1 ###")
    plot_global_metric(experiment_summary_df, pairwise_diff_df, config)

    # Step 4: Analyze Q2 (Per-class F1)
    logger.info("\n### STEP 4: Analyze Q2 (Per-class F1) ###")
    per_class_stats_df = analyse_per_class_f1(per_class_df, config)

    # Step 5: Visualize Q2
    logger.info("\n### STEP 5: Visualize Q2 ###")
    plot_per_class_improvement(per_class_stats_df, config)

    # Step 6: Write metadata
    logger.info("\n### STEP 6: Write Metadata ###")
    write_metadata(config)

    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info(f"All outputs saved to: {config.output_dir}")
    logger.info("="*80)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Downstream Classification Significance Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m medsyn.analysis.classification.downstream_significance \\
      --root-results-dir /media/mpascual/Sandisk2TB/research/medsyn/results/classification \\
      --dataset pathmnist \\
      --primary-metric f1 \\
      --backbones resnet50 clip_vit_b16 wideresnet28_10 \\
      --experiments real_only real_plus_trad_aug real_plus_synth_cfgmedsyn real_plus_synth_distdiff \\
      --our-experiment real_plus_synth_cfgmedsyn \\
      --output-dir ./downstream_analysis_output
        """
    )

    parser.add_argument(
        "--root-results-dir",
        type=Path,
        required=True,
        help="Root directory containing all classification experiment results"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., 'pathmnist')"
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        required=True,
        choices=["f1", "auc", "loss", "precision", "recall"],
        help="Primary metric for Q1 analysis"
    )
    parser.add_argument(
        "--backbones",
        type=str,
        nargs="+",
        required=True,
        help="List of backbone/model names"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="List of experiment/regime names"
    )
    parser.add_argument(
        "--our-experiment",
        type=str,
        required=True,
        help="Name of cfg-MedSyn experiment (usually 'real_plus_synth_cfgmedsyn')"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for saving results"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create config
    config = DownstreamAnalysisConfig(
        root_results_dir=args.root_results_dir,
        dataset_name=args.dataset,
        primary_metric=args.primary_metric,
        backbones=args.backbones,
        experiments=args.experiments,
        our_experiment_name=args.our_experiment,
        output_dir=args.output_dir
    )

    # Run analysis
    run_full_analysis(config)


if __name__ == "__main__":
    main()
