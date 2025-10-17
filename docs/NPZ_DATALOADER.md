# NPZ Dataloader for ccDDPM

## Overview

The NPZ dataloader allows you to use a single compressed `.npz` file instead of loading individual PNG images. This is particularly useful when:
- Training on supercomputers with limited I/O
- Working with network-mounted storage
- Wanting faster data loading
- Needing to transfer datasets efficiently

## Setup

### 1. Convert JSON Index to NPZ

First, convert your existing JSON-indexed dataset to NPZ format:

```bash
python -m medsyn.cli.json_to_npz \
    --json /path/to/pathmnist_index.json \
    --output /path/to/PathMNIST.npz \
    --dataset PathMNIST \
    --splits train val test
```

**Optional: Resize during conversion**
```bash
python -m medsyn.cli.json_to_npz \
    --json /path/to/pathmnist_index.json \
    --output /path/to/PathMNIST_64x64.npz \
    --resize 64 64
```

### 2. Update Configuration

In your `medsyn_cfg.yaml`, configure the dataloader:

```yaml
ccddpm:
  dataloader:
    type: npz  # or "json" for file-based loading
    npz_path: /path/to/PathMNIST.npz
  
  train:
    image_size: 64
    # ... rest of config
```

**For JSON dataloader (default):**
```yaml
ccddpm:
  dataloader:
    type: json  # Uses individual PNG files
    npz_path: null
```

## NPZ File Structure

The NPZ file contains separate arrays for each split:

```
PathMNIST.npz:
  ├── train_images: [N_train, H, W, 3] uint8
  ├── train_labels: [N_train] int64
  ├── train_is_synth: [N_train] bool
  ├── val_images: [N_val, H, W, 3] uint8
  ├── val_labels: [N_val] int64
  ├── val_is_synth: [N_val] bool
  ├── test_images: [N_test, H, W, 3] uint8
  ├── test_labels: [N_test] int64
  └── test_is_synth: [N_test] bool
```

## Usage

### Training

```bash
# With NPZ dataloader
ccddpm-train --config config/medsyn_cfg.yaml

# The training script automatically selects the dataloader based on config
```

### Advantages of NPZ Dataloader

1. **Faster Loading**: Single file read vs. thousands of image reads
2. **Better I/O Performance**: Reduces file system overhead
3. **Compressed Storage**: NPZ uses compression (typical 30-50% size reduction)
4. **Atomic Operations**: No missing file errors
5. **Supercomputer Friendly**: Reduces strain on parallel filesystems

### Performance Comparison

| Metric | JSON Dataloader | NPZ Dataloader |
|--------|----------------|----------------|
| File I/O ops | ~10,000s | 1 |
| Load time (epoch 1) | ~30s | ~3s |
| Storage size | 100% | ~35-50% |
| num_workers | 4-8 | 0-2 (not needed) |

## Verification

After creating the NPZ file, verify it:

```python
import numpy as np

data = np.load('PathMNIST.npz')
print("Keys:", list(data.keys()))
print("Train images:", data['train_images'].shape)
print("Train labels:", data['train_labels'].shape)
print("Unique labels:", np.unique(data['train_labels']))
```

## Troubleshooting

### Issue: "NPZ file missing key 'train_images'"
**Solution**: Ensure the JSON conversion completed successfully and all splits are present.

### Issue: "Memory error when loading NPZ"
**Solution**: The NPZ file is loaded into memory. For very large datasets, consider:
- Using memory-mapped arrays (requires code modification)
- Reducing image resolution during conversion
- Using the JSON dataloader instead

### Issue: "Different image sizes in NPZ"
**Solution**: Use `--resize H W` during conversion to ensure uniform sizes.

## Migration Guide

### From JSON to NPZ

1. **Create NPZ file** (one-time operation):
   ```bash
   python -m medsyn.cli.json_to_npz \
       --json $JSON_INDEX \
       --output $NPZ_PATH
   ```

2. **Update config**:
   ```yaml
   ccddpm:
     dataloader:
       type: npz
       npz_path: $NPZ_PATH
   ```

3. **Train as usual**:
   ```bash
   ccddpm-train --config config/medsyn_cfg.yaml
   ```

### Back to JSON

Simply change the config:
```yaml
ccddpm:
  dataloader:
    type: json  # Switch back to file-based loading
```

No other changes needed!
