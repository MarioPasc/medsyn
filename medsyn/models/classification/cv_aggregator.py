"""
Cross-validation results aggregator for k-fold experiments.
"""

from pathlib import Path
from typing import List, Dict
import logging
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class CrossValidationAggregator:
    """
    Aggregates results from multiple k-fold cross-validation runs.

    This class reads results.csv and per_class_auc.csv from each fold directory,
    computes mean ± std statistics across folds, and saves comprehensive
    cross-validation summary.

    Args:
        fold_dirs: List of paths to fold result directories
        output_dir: Directory to save aggregated results
        k_folds: Number of folds
    """

    def __init__(self, fold_dirs: List[Path], output_dir: Path, k_folds: int):
        self.fold_dirs = [Path(d) for d in fold_dirs]
        self.output_dir = Path(output_dir)
        self.k_folds = k_folds
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_and_save(self):
        """
        Aggregate results from all folds and save summary statistics.
        """
        logger.info(f"Aggregating results from {self.k_folds} folds...")

        # Aggregate standard metrics (from results.csv)
        logger.info("Aggregating standard metrics (accuracy, loss)...")
        standard_metrics_df = self._aggregate_standard_metrics()

        if standard_metrics_df is not None:
            standard_output = self.output_dir / "cv_standard_metrics_summary.csv"
            standard_metrics_df.to_csv(standard_output, index=False)
            logger.info(f"Standard metrics summary saved to: {standard_output}")

        # Aggregate per-class AUC metrics
        logger.info("Aggregating per-class AUC metrics...")
        auc_metrics_df = self._aggregate_per_class_auc()

        if auc_metrics_df is not None:
            auc_output = self.output_dir / "cv_per_class_auc_summary.csv"
            auc_metrics_df.to_csv(auc_output, index=False)
            logger.info(f"Per-class AUC summary saved to: {auc_output}")

        # Aggregate best-epoch per-class metrics (F1 and AUC from best epochs)
        logger.info("Aggregating best-epoch per-class metrics (F1 and AUC)...")
        best_epoch_summary = self._aggregate_best_epoch_metrics()

        if best_epoch_summary is not None:
            best_epoch_output = self.output_dir / "cv_best_epoch_per_class_summary.yaml"
            with open(best_epoch_output, "w") as f:
                yaml.safe_dump(best_epoch_summary, f, default_flow_style=False)
            logger.info(f"Best-epoch per-class metrics summary saved to: {best_epoch_output}")

            # Also save as CSV for easier analysis
            self._save_best_epoch_as_csv(best_epoch_summary)

        # Create combined summary of best epoch results
        logger.info("Creating combined cross-validation summary...")
        cv_summary_df = self._create_cv_summary(standard_metrics_df, auc_metrics_df)

        if cv_summary_df is not None:
            summary_output = self.output_dir / "cv_summary.csv"
            cv_summary_df.to_csv(summary_output, index=False)
            logger.info(f"Cross-validation summary saved to: {summary_output}")

            # Log summary to console
            self._log_summary(cv_summary_df)

        # Log best-epoch summary
        if best_epoch_summary is not None:
            self._log_best_epoch_summary(best_epoch_summary)

    def _aggregate_standard_metrics(self) -> pd.DataFrame:
        """
        Aggregate standard metrics from results.csv across all folds.

        Returns:
            DataFrame with metrics, mean, std, and individual fold values
        """
        fold_results = []

        for fold_idx, fold_dir in enumerate(self.fold_dirs):
            results_file = fold_dir / "results.csv"

            if not results_file.exists():
                logger.warning(f"results.csv not found in {fold_dir}")
                continue

            try:
                df = pd.read_csv(results_file)
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()

                # Get best epoch (lowest validation loss or highest accuracy)
                if 'metrics/accuracy_top1' in df.columns:
                    best_idx = df['metrics/accuracy_top1'].idxmax()
                else:
                    best_idx = len(df) - 1  # Use last epoch as fallback

                best_epoch_data = df.iloc[best_idx].to_dict()
                best_epoch_data['fold'] = fold_idx
                fold_results.append(best_epoch_data)

            except Exception as e:
                logger.error(f"Error reading results from {results_file}: {e}")
                continue

        if not fold_results:
            logger.warning("No valid results.csv files found")
            return None

        # Convert to DataFrame
        results_df = pd.DataFrame(fold_results)

        # Select metrics to aggregate (exclude epoch, fold columns)
        metric_cols = [col for col in results_df.columns
                      if col not in ['epoch', 'fold'] and not col.startswith('lr/')]

        # Compute statistics
        summary_data = []
        for metric in metric_cols:
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    row = {
                        'metric': metric,
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                    }
                    # Add individual fold values
                    for fold_idx in range(self.k_folds):
                        if fold_idx < len(values):
                            row[f'fold_{fold_idx}'] = values.iloc[fold_idx]
                        else:
                            row[f'fold_{fold_idx}'] = np.nan

                    summary_data.append(row)

        return pd.DataFrame(summary_data)

    def _aggregate_per_class_auc(self) -> pd.DataFrame:
        """
        Aggregate per-class AUC metrics across all folds.

        Returns:
            DataFrame with per-class AUC statistics
        """
        fold_auc_results = []

        for fold_idx, fold_dir in enumerate(self.fold_dirs):
            auc_file = fold_dir / "per_class_auc.csv"

            if not auc_file.exists():
                logger.warning(f"per_class_auc.csv not found in {fold_dir}")
                continue

            try:
                df = pd.read_csv(auc_file)
                # Strip whitespace from column names
                df.columns = df.columns.str.strip()

                # Get last epoch (final AUC values)
                if len(df) > 0:
                    last_epoch_data = df.iloc[-1].to_dict()
                    last_epoch_data['fold'] = fold_idx
                    fold_auc_results.append(last_epoch_data)

            except Exception as e:
                logger.error(f"Error reading AUC results from {auc_file}: {e}")
                continue

        if not fold_auc_results:
            logger.warning("No valid per_class_auc.csv files found")
            return None

        # Convert to DataFrame
        auc_df = pd.DataFrame(fold_auc_results)

        # Select AUC columns
        auc_cols = [col for col in auc_df.columns
                   if col.startswith('auc_') and col != 'fold']

        # Compute statistics
        summary_data = []
        for auc_col in auc_cols:
            if auc_col in auc_df.columns:
                values = auc_df[auc_col].dropna()
                if len(values) > 0:
                    row = {
                        'metric': auc_col,
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                    }
                    # Add individual fold values
                    for fold_idx in range(self.k_folds):
                        if fold_idx < len(values):
                            row[f'fold_{fold_idx}'] = values.iloc[fold_idx]
                        else:
                            row[f'fold_{fold_idx}'] = np.nan

                    summary_data.append(row)

        return pd.DataFrame(summary_data)

    def _create_cv_summary(self, standard_df: pd.DataFrame,
                           auc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a combined cross-validation summary.

        Args:
            standard_df: Standard metrics summary
            auc_df: Per-class AUC summary

        Returns:
            Combined summary DataFrame
        """
        summary_rows = []

        # Add key standard metrics
        if standard_df is not None:
            key_metrics = ['metrics/accuracy_top1', 'metrics/accuracy_top5']
            for metric in key_metrics:
                row_data = standard_df[standard_df['metric'] == metric]
                if not row_data.empty:
                    summary_rows.append(row_data.iloc[0].to_dict())

        # Add macro-average AUC
        if auc_df is not None:
            macro_auc = auc_df[auc_df['metric'] == 'auc_macro']
            if not macro_auc.empty:
                summary_rows.append(macro_auc.iloc[0].to_dict())

        if not summary_rows:
            return None

        return pd.DataFrame(summary_rows)

    def _log_summary(self, summary_df: pd.DataFrame):
        """
        Log cross-validation summary to console.

        Args:
            summary_df: Summary DataFrame to log
        """
        logger.info("="*80)
        logger.info("Cross-Validation Results Summary")
        logger.info("="*80)

        for _, row in summary_df.iterrows():
            metric_name = row['metric']
            mean_val = row['mean']
            std_val = row['std']

            logger.info(f"{metric_name:30s}: {mean_val:.4f} ± {std_val:.4f}")

            # Log individual fold values
            fold_values = []
            for fold_idx in range(self.k_folds):
                fold_col = f'fold_{fold_idx}'
                if fold_col in row and not pd.isna(row[fold_col]):
                    fold_values.append(f"{row[fold_col]:.4f}")

            if fold_values:
                logger.info(f"{'  Individual folds':30s}: [{', '.join(fold_values)}]")

        logger.info("="*80)

    def _aggregate_best_epoch_metrics(self) -> Dict:
        """
        Aggregate best-epoch per-class metrics (F1 and AUC) across all folds.

        Returns:
            Dictionary with aggregated statistics for train and val splits
        """
        fold_best_metrics = []

        for fold_idx, fold_dir in enumerate(self.fold_dirs):
            best_epoch_file = fold_dir / "logs" / "best_epoch_metrics.yaml"

            if not best_epoch_file.exists():
                logger.warning(f"best_epoch_metrics.yaml not found in {fold_dir}")
                continue

            try:
                with open(best_epoch_file, "r") as f:
                    best_metrics = yaml.safe_load(f)
                fold_best_metrics.append({
                    "fold": fold_idx,
                    "data": best_metrics
                })
            except Exception as e:
                logger.error(f"Error reading best epoch metrics from {best_epoch_file}: {e}")
                continue

        if not fold_best_metrics:
            logger.warning("No valid best_epoch_metrics.yaml files found")
            return None

        # Aggregate across folds for each split (train, val)
        summary = {}

        for split in ["train", "val"]:
            split_data = []
            for fold_data in fold_best_metrics:
                if split in fold_data["data"]:
                    split_data.append(fold_data["data"][split])

            if split_data:
                summary[split] = self._compute_per_class_statistics(split_data)

        # Add metadata
        summary["metadata"] = {
            "num_folds": self.k_folds,
            "num_folds_with_data": len(fold_best_metrics),
            "description": "Aggregated per-class F1 and AUC-ROC from best epochs across k-fold CV"
        }

        return summary

    def _compute_per_class_statistics(self, split_data: List[Dict]) -> Dict:
        """
        Compute mean and std of per-class metrics across folds.

        Args:
            split_data: List of metrics dictionaries from each fold

        Returns:
            Dictionary with mean and std for each metric
        """
        if not split_data:
            return {}

        # Extract overall metrics
        overall_metrics = {}
        for metric_name in ["loss_overall", "accuracy", "f1_macro", "auc_macro"]:
            values = [d.get(metric_name, 0.0) for d in split_data]
            if values:
                overall_metrics[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values
                }

        # Extract per-class metrics
        per_class_metrics = {}

        # Get number of classes from first fold
        if split_data and "per_class_f1" in split_data[0]:
            num_classes = len(split_data[0]["per_class_f1"])

            for class_idx in range(num_classes):
                class_metrics = {}

                # F1 scores
                f1_values = [d["per_class_f1"][class_idx] for d in split_data
                            if "per_class_f1" in d and class_idx < len(d["per_class_f1"])]
                if f1_values:
                    class_metrics["f1"] = {
                        "mean": float(np.mean(f1_values)),
                        "std": float(np.std(f1_values)),
                        "values": f1_values
                    }

                # AUC scores
                auc_values = [d["per_class_auc"][class_idx] for d in split_data
                             if "per_class_auc" in d and class_idx < len(d["per_class_auc"])]
                if auc_values:
                    class_metrics["auc"] = {
                        "mean": float(np.mean(auc_values)),
                        "std": float(np.std(auc_values)),
                        "values": auc_values
                    }

                # Loss
                loss_values = [d["per_class_loss"].get(class_idx, 0.0) for d in split_data
                              if "per_class_loss" in d]
                if loss_values:
                    class_metrics["loss"] = {
                        "mean": float(np.mean(loss_values)),
                        "std": float(np.std(loss_values)),
                        "values": loss_values
                    }

                # Precision
                precision_values = [d["per_class_precision"][class_idx] for d in split_data
                                   if "per_class_precision" in d and class_idx < len(d["per_class_precision"])]
                if precision_values:
                    class_metrics["precision"] = {
                        "mean": float(np.mean(precision_values)),
                        "std": float(np.std(precision_values)),
                        "values": precision_values
                    }

                # Recall
                recall_values = [d["per_class_recall"][class_idx] for d in split_data
                                if "per_class_recall" in d and class_idx < len(d["per_class_recall"])]
                if recall_values:
                    class_metrics["recall"] = {
                        "mean": float(np.mean(recall_values)),
                        "std": float(np.std(recall_values)),
                        "values": recall_values
                    }

                per_class_metrics[f"class_{class_idx}"] = class_metrics

        return {
            "overall": overall_metrics,
            "per_class": per_class_metrics
        }

    def _save_best_epoch_as_csv(self, best_epoch_summary: Dict):
        """
        Save best-epoch summary as CSV files for easier analysis.

        Args:
            best_epoch_summary: Summary dictionary from _aggregate_best_epoch_metrics
        """
        for split in ["train", "val"]:
            if split not in best_epoch_summary:
                continue

            split_data = best_epoch_summary[split]
            if "per_class" not in split_data:
                continue

            # Create CSV rows
            rows = []
            for class_name, metrics in split_data["per_class"].items():
                row = {"class": class_name}
                for metric_name, metric_data in metrics.items():
                    row[f"{metric_name}_mean"] = metric_data.get("mean", 0.0)
                    row[f"{metric_name}_std"] = metric_data.get("std", 0.0)
                rows.append(row)

            # Save to CSV
            if rows:
                df = pd.DataFrame(rows)
                output_path = self.output_dir / f"cv_best_epoch_{split}_per_class.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"Best-epoch {split} per-class metrics saved to: {output_path}")

    def _log_best_epoch_summary(self, best_epoch_summary: Dict):
        """
        Log best-epoch summary to console.

        Args:
            best_epoch_summary: Summary dictionary from _aggregate_best_epoch_metrics
        """
        logger.info("="*80)
        logger.info("Best Epoch Per-Class Metrics Summary (Representative F1 and AUC-ROC)")
        logger.info("="*80)

        for split in ["train", "val"]:
            if split not in best_epoch_summary:
                continue

            logger.info(f"\n{split.upper()} Split:")
            logger.info("-" * 60)

            split_data = best_epoch_summary[split]

            # Log overall metrics
            if "overall" in split_data:
                logger.info("Overall Metrics:")
                for metric_name, metric_data in split_data["overall"].items():
                    mean = metric_data.get("mean", 0.0)
                    std = metric_data.get("std", 0.0)
                    logger.info(f"  {metric_name:20s}: {mean:.4f} ± {std:.4f}")

            # Log per-class metrics
            if "per_class" in split_data:
                logger.info("\nPer-Class Metrics:")
                for class_name, metrics in sorted(split_data["per_class"].items()):
                    logger.info(f"  {class_name}:")
                    for metric_name in ["f1", "auc", "precision", "recall", "loss"]:
                        if metric_name in metrics:
                            mean = metrics[metric_name].get("mean", 0.0)
                            std = metrics[metric_name].get("std", 0.0)
                            logger.info(f"    {metric_name:15s}: {mean:.4f} ± {std:.4f}")

        logger.info("="*80)
