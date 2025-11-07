# K-Fold Cross-Validation Implementation Guide

## Overview
This implementation adds 5-fold cross-validation support to the classification training pipeline while maintaining full backward compatibility with the original single train/val split approach.

## What Was Implemented

### 1. **KFoldSplitter Utility** (`medsyn/models/classifier/kfold_utils.py`)
- Creates stratified k-fold splits from training data
- Maintains class balance across folds
- Uses fixed seed (42) for reproducibility
- Combines original train+val splits for cross-validation

### 2. **Dataset Modifications** (`medsyn/models/classifier/dataloaders.py`)
- Added optional `fold_images`, `fold_labels`, `fold_is_synth` parameters
- When provided, uses fold data instead of loading from NPZ
- Fully backward compatible (no changes when fold data is None)

### 3. **Per-Class AUC CSV Export** (`medsyn/models/classifier/engine/yolo_validator.py`)
- Validator now saves per-class AUC to `per_class_auc.csv` each epoch
- Columns: `epoch, auc_class_0, auc_class_1, ..., auc_class_K, auc_macro`
- CSV format allows easy analysis across epochs and folds
- Keeps existing JSON output for backward compatibility

### 4. **Trainer Fold Support** (`medsyn/models/classifier/engine/yolo_trainer.py`)
- Added `fold_idx` and `fold_data` parameters to trainer
- Passes fold data to dataset when creating dataloaders
- Experiment names automatically include fold index (e.g., `experiment_fold0`)

### 5. **CLI K-Fold Integration** (`medsyn/cli/classify.py`)
- Added `--k_folds` argument (default: 1)
- k_folds=1: Uses original train/val split (backward compatible)
- k_folds=5: Runs 5-fold cross-validation automatically
- Sequential execution: trains all folds one after another

### 6. **Cross-Validation Aggregator** (`medsyn/models/classifier/cv_aggregator.py`)
- Aggregates results from all folds after training completes
- Computes mean ± std for all metrics
- Outputs:
  - `cv_standard_metrics_summary.csv`: Accuracy, loss statistics
  - `cv_per_class_auc_summary.csv`: Per-class AUC statistics
  - `cv_summary.csv`: Combined summary of key metrics

## Usage

### Single Train/Val Split (Original Behavior)
```bash
~/.conda/envs/medsyn/bin/python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real
```

This runs with `--k_folds 1` by default, preserving the original behavior exactly.

### 5-Fold Cross-Validation
```bash
~/.conda/envs/medsyn/bin/python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real \
    --k_folds 5
```

This will:
1. Combine train+val splits into 100,000 samples
2. Create 5 stratified folds (80k train, 20k val each)
3. Train 5 separate models sequentially
4. Save results for each fold in separate directories:
   - `{project}/{name}_fold0/`
   - `{project}/{name}_fold1/`
   - ...
   - `{project}/{name}_fold4/`
5. Aggregate results and save summary to `{project}/{name}/cv_summary.csv`

### Testing with Fewer Epochs
For quick testing, you can modify `config/yolo_hyperparameters.yaml` to set `epochs: 1` or `epochs: 5`:

```bash
# Test k_folds=1 (backward compatibility)
~/.conda/envs/medsyn/bin/python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real \
    --name test_single_split \
    --k_folds 1

# Test k_folds=5 (full CV)
~/.conda/envs/medsyn/bin/python -m medsyn.cli.classify \
    --config config/medsyn_cfg.yaml \
    --hparams config/yolo_hyperparameters.yaml \
    --training_mode real \
    --name test_5fold_cv \
    --k_folds 5
```

## Output Structure

### Single Split (k_folds=1)
```
{project}/{name}/
├── results.csv                    # Standard YOLO metrics per epoch
├── per_class_auc.csv             # Per-class AUC per epoch (NEW)
├── per_class_auc_metrics.json    # Latest per-class AUC (existing)
├── best.pt                        # Best model checkpoint
├── last.pt                        # Latest model checkpoint
└── training_*.log                 # Training logs
```

### 5-Fold CV (k_folds=5)
```
{project}/{name}/
├── cv_summary.csv                           # Aggregated CV summary (NEW)
├── cv_standard_metrics_summary.csv         # Detailed metric statistics (NEW)
├── cv_per_class_auc_summary.csv           # Detailed AUC statistics (NEW)
└── training_*.log                          # Main training log

{project}/{name}_fold0/
├── results.csv
├── per_class_auc.csv              # Per-class AUC per epoch (NEW)
├── per_class_auc_metrics.json
├── best.pt
└── last.pt

{project}/{name}_fold1/
├── ...

... (folds 2, 3, 4 similarly)
```

## Key Features

### ✅ Backward Compatibility
- `--k_folds 1` (default) behaves exactly like the original implementation
- No changes to existing workflows unless explicitly requested
- All existing files and outputs remain the same

### ✅ Per-Class AUC Tracking
- Now saved to CSV format for easy epoch-wise analysis
- Works for both single split and k-fold CV
- Includes macro-average AUC

### ✅ Stratified Folds
- Class distribution maintained across all folds
- Fixed seed (42) ensures reproducibility
- Balanced train/val splits (80k/20k samples)

### ✅ Comprehensive Aggregation
- Mean ± std computed for all metrics
- Individual fold values preserved in summary
- Separate files for standard metrics and AUC

## Testing Status

All components have been tested:
- ✅ Module imports successful
- ✅ KFoldSplitter creates balanced folds correctly
- ✅ Dataset accepts fold data properly
- ✅ CLI syntax verified
- ✅ Integration tested with real NPZ data

To run full end-to-end tests with actual training, use the commands above with `epochs: 1` in the hyperparameters file.

## Design Choices

1. **Fixed seed (42)**: Ensures reproducibility across runs
2. **Sequential execution**: Simpler implementation, easier debugging
3. **Separate CSV for AUC**: Cleaner than modifying existing results.csv
4. **Fold naming**: `{name}_fold{idx}` pattern keeps organization clear
5. **Test set untouched**: K-fold only operates on train+val, test remains for final evaluation

## Notes

- The test set remains separate and is not used in cross-validation
- Each fold trains independently with its own checkpoints
- Training time increases linearly with k_folds (5x for k_folds=5)
- GPU memory usage is the same per fold as single training
- All folds use the same hyperparameters from the config files
