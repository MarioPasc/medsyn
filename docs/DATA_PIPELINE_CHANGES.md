# Data Pipeline Configuration Changes

## Overview

The data pipeline has been restructured to support flexible workflows with conditional PNG extraction and NPZ postprocessing. This allows you to:

1. **Enable/disable PNG extraction** for JSON-based dataloaders and YOLO datasets
2. **Enable/disable NPZ postprocessing** to create a custom compressed dataset from the original MedMNIST NPZ
3. **Automatically read NPZ paths** from the unified data configuration

## Configuration Structure

### Old Structure (deprecated)
```yaml
data:
  download_dir: /path/to/raw
  processed_dir: /path/to/processed  # ❌ Removed
  index_json: /path/to/index.json    # ❌ Removed
  yolo_folder_dataset: /path/to/yolo # ❌ Removed

ccddpm:
  dataloader:
    type: npz
    npz_path: /path/to/custom.npz    # ❌ Removed from here
```

### New Structure (current)
```yaml
data:
  flag: pathmnist
  size: 64
  download_dir: /path/to/raw         # Original MedMNIST NPZ location
  
  save_png:                          # ✅ NEW: PNG extraction control
    enabled: true                    # Set to false to skip PNG extraction
    processed_dir: /path/to/processed
    index_json: /path/to/index.json
    yolo_folder_dataset: /path/to/yolo  # Optional
  
  postprocess_npz:                   # ✅ NEW: NPZ postprocessing control
    enabled: true                    # Set to false to skip NPZ creation
    npz_path: /path/to/custom.npz   # Custom NPZ with splits
  
  seed: 23102003
  reduction:
    strategy: fraction
    train: 1.00
    val: 1.00
    test: 1.00

ccddpm:
  dataloader:
    type: npz                        # ✅ Type selection (json or npz)
    # npz_path is now read from data.postprocess_npz.npz_path
```

## Workflow Scenarios

### Scenario 1: Full Pipeline (PNG + NPZ)
```yaml
data:
  save_png:
    enabled: true      # Extract PNGs and create JSON index
  postprocess_npz:
    enabled: true      # Create custom NPZ from original

# Result: Both PNG files + JSON index AND custom NPZ created
```

### Scenario 2: JSON Dataloader Only
```yaml
data:
  save_png:
    enabled: true      # Extract PNGs and create JSON index
  postprocess_npz:
    enabled: false     # Skip NPZ creation

ccddpm:
  dataloader:
    type: json         # Use JSON-based dataloader

# Result: Only PNG files + JSON index created
```

### Scenario 3: NPZ Dataloader Only
```yaml
data:
  save_png:
    enabled: false     # Skip PNG extraction
  postprocess_npz:
    enabled: true      # Create custom NPZ from original

ccddpm:
  dataloader:
    type: npz          # Use NPZ-based dataloader

# Result: Only custom NPZ created (faster, supercomputer-friendly)
```

### Scenario 4: Development Mode (No Processing)
```yaml
data:
  save_png:
    enabled: false     # Skip PNG extraction
  postprocess_npz:
    enabled: false     # Skip NPZ creation

# Result: Only downloads original MedMNIST data, no processing
# Useful for testing configuration without data processing
```

## Implementation Details

### 1. Data Configuration (`medsyn/data/config.py`)

**New Dataclasses:**
```python
@dataclass(frozen=True)
class SavePngCfg:
    enabled: bool = True
    processed_dir: str = "./data_processed"
    index_json: str = "./indexes/pathmnist_index.json"
    yolo_folder_dataset: Optional[str] = None

@dataclass(frozen=True)
class PostprocessNpzCfg:
    enabled: bool = False
    npz_path: str = "./PathMNIST.npz"

@dataclass(frozen=True)
class DataCfg:
    # ... other fields ...
    save_png: SavePngCfg = SavePngCfg()
    postprocess_npz: PostprocessNpzCfg = PostprocessNpzCfg()
```

### 2. Data Preparation CLI (`medsyn/cli/data.py`)

**Flow:**
1. Load configuration and ensure directories
2. Prepare PathMNIST datasets with stratified reduction
3. **[Conditional]** PNG Extraction (if `save_png.enabled`):
   - Export images to PNG files
   - Create JSON index
   - Generate YOLO dataset (if configured)
4. **[Conditional]** NPZ Postprocessing (if `postprocess_npz.enabled`):
   - Read original MedMNIST NPZ
   - Extract custom split indices
   - Create new NPZ with `{split}_images`, `{split}_labels`, `{split}_is_synth`

**New Function:**
```python
def _create_custom_npz(cfg: ProjectCfg, ds: SplitDatasets) -> None:
    """
    Create custom NPZ from original MedMNIST NPZ with:
    - Custom stratified splits
    - Proper format: {split}_images [N,H,W,C], {split}_labels [N], {split}_is_synth [N]
    """
```

### 3. ccDDPM Configuration (`medsyn/models/ccDDPM/config.py`)

**NPZ Path Resolution:**
```python
# Old way (deprecated):
npz_path = cc.get("dataloader", {}).get("npz_path")

# New way (current):
if dataloader_type == "npz":
    postprocess_npz = data_dict.get("postprocess_npz", {})
    npz_path = postprocess_npz.get("npz_path")
```

**Benefits:**
- Single source of truth for NPZ path
- Automatically validated when NPZ dataloader selected
- Clearer separation between data processing and model config

## Usage Examples

### Example 1: Initial Setup (Everything)
```bash
# config/medsyn_cfg.yaml
data:
  save_png:
    enabled: true
  postprocess_npz:
    enabled: true

# Run data preparation
medsyn-prepare-data --config config/medsyn_cfg.yaml

# Train with JSON dataloader
ccddpm-train --config config/medsyn_cfg.yaml  # Uses JSON

# Or train with NPZ dataloader (faster)
# Change config: ccddpm.dataloader.type: npz
ccddpm-train --config config/medsyn_cfg.yaml  # Uses NPZ
```

### Example 2: Supercomputer Workflow
```bash
# Step 1: On local machine - create custom NPZ only
data:
  save_png:
    enabled: false       # Don't need PNGs for supercomputer
  postprocess_npz:
    enabled: true        # Create compressed NPZ

medsyn-prepare-data --config config/medsyn_cfg.yaml

# Step 2: Transfer only the NPZ file to supercomputer
scp /path/to/PathMNIST.npz supercomputer:/scratch/user/

# Step 3: Train on supercomputer with NPZ
ccddpm:
  dataloader:
    type: npz

ccddpm-train --config config/medsyn_cfg.yaml
```

### Example 3: YOLO Classification Dataset
```bash
# Enable PNG extraction with YOLO dataset
data:
  save_png:
    enabled: true
    yolo_folder_dataset: /path/to/yolo_dataset
  postprocess_npz:
    enabled: false       # Don't need NPZ for YOLO

medsyn-prepare-data --config config/medsyn_cfg.yaml
# Creates PNG files, JSON index, AND YOLO symlink dataset
```

## Migration Guide

### From Old Config to New Config

**Step 1: Update YAML structure**
```yaml
# OLD
data:
  processed_dir: /path/to/processed
  index_json: /path/to/index.json
  yolo_folder_dataset: /path/to/yolo

# NEW
data:
  save_png:
    enabled: true
    processed_dir: /path/to/processed
    index_json: /path/to/index.json
    yolo_folder_dataset: /path/to/yolo
  postprocess_npz:
    enabled: false  # Add this section
```

**Step 2: Move NPZ path (if using NPZ dataloader)**
```yaml
# OLD
ccddpm:
  dataloader:
    type: npz
    npz_path: /path/to/custom.npz

# NEW
data:
  postprocess_npz:
    enabled: true
    npz_path: /path/to/custom.npz

ccddpm:
  dataloader:
    type: npz
    # npz_path removed from here
```

**Step 3: Run data preparation**
```bash
# This will work with new config structure
medsyn-prepare-data --config config/medsyn_cfg.yaml
```

## Technical Notes

### NPZ Format Specification
The custom NPZ file contains:
```python
{
    "train_images": np.ndarray,  # [N_train, H, W, C] uint8
    "train_labels": np.ndarray,  # [N_train] int64
    "train_is_synth": np.ndarray,  # [N_train] bool
    "val_images": np.ndarray,    # [N_val, H, W, C] uint8
    "val_labels": np.ndarray,    # [N_val] int64
    "val_is_synth": np.ndarray,  # [N_val] bool
    "test_images": np.ndarray,   # [N_test, H, W, C] uint8
    "test_labels": np.ndarray,   # [N_test] int64
    "test_is_synth": np.ndarray  # [N_test] bool
}
```

### Backward Compatibility
- Old configs will show warnings but still work for JSON dataloader
- NPZ dataloader requires new config structure
- Migration is non-breaking for existing JSON workflows

## Troubleshooting

### "NPZ dataloader selected but data.postprocess_npz.npz_path not found"
**Solution:** Add the postprocess_npz section to your config:
```yaml
data:
  postprocess_npz:
    enabled: true
    npz_path: /path/to/output.npz
```

### "No index_json found in YAML"
**Solution:** Add the save_png section if using JSON dataloader:
```yaml
data:
  save_png:
    enabled: true
    index_json: /path/to/index.json
```

### Both PNG and NPZ enabled but only need one
**Solution:** Disable the one you don't need:
```yaml
data:
  save_png:
    enabled: false  # If using NPZ
  postprocess_npz:
    enabled: true
```
