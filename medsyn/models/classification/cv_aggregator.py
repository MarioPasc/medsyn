"""
Cross-validation results aggregator for k-fold experiments.
"""

from pathlib import Path
from typing import List
import logging
import pandas as pd
import numpy as np

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

        # Create combined summary of best epoch results
        logger.info("Creating combined cross-validation summary...")
        cv_summary_df = self._create_cv_summary(standard_metrics_df, auc_metrics_df)

        if cv_summary_df is not None:
            summary_output = self.output_dir / "cv_summary.csv"
            cv_summary_df.to_csv(summary_output, index=False)
            logger.info(f"Cross-validation summary saved to: {summary_output}")

            # Log summary to console
            self._log_summary(cv_summary_df)

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
