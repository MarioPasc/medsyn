# Enhanced Latent Space Diagnostics for ccDDPM

Comprehensive logging system for analyzing class embeddings and latent space geometry in diffusion models.

## Overview

This module extends the basic class embedding trajectory logging with three layers of diagnostics:

1. **Probe Bank**: Feature vectors from a balanced sample set across classes, timesteps, and layers
2. **Clustering Metrics**: Quantitative measures of latent space organization (silhouette, Calinski-Harabasz, Davies-Bouldin)
3. **Rich Metadata**: Complete experimental context for reproducibility and cross-run comparison

## Motivation

The original logging saved only parametric class embeddings (`class_embed.emb.weight`), which shows how the model *learns* class representations but not how they *interact* with actual data during diffusion.

This enhanced system enables:
- **Latent space visualization**: KDE plots, t-SNE/UMAP with thousands of points (not just 9 class centers)
- **Temporal analysis**: How class separation evolves across diffusion timesteps
- **Depth analysis**: How features transform through network layers
- **CFG analysis**: How classifier-free guidance affects conditional vs unconditional features
- **Quality monitoring**: Quantitative scores to detect mode collapse, poor separation, or overfitting

## Architecture

```
medsyn/models/ccDDPM/engine/logging/
├── embeddings.py              # Main implementation
├── embeddings_example.py      # Usage examples
└── EMBEDDINGS_README.md       # This file
```

### Key Components

#### 1. Configuration Classes

```python
@dataclass
class ProbeConfig:
    """Configure feature probe bank collection."""
    enabled: bool = True
    samples_per_class: int = 10          # Balanced sample size
    timesteps: List[int] = [100, 500, 900]  # Which diffusion steps
    layer_names: List[str] = []          # Which model layers to extract
    save_both_branches: bool = True      # Cond + uncond (for CFG)
    pooling_strategy: str = "mean"       # Spatial pooling

@dataclass
class ClusteringConfig:
    """Configure clustering quality metrics."""
    enabled: bool = True
    n_clusters: Optional[int] = None     # None = use num_classes
    algorithm: str = "kmeans"            # kmeans or gmm
    compute_scores: bool = True          # Silhouette, CH, DB

@dataclass
class MetadataConfig:
    """Configure metadata logging."""
    save_metadata: bool = True
    class_names: Optional[List[str]] = None
    dataset_name: str = "unknown"

@dataclass
class EmbeddingLogConfig:
    """Master configuration."""
    log_every_n_epochs: int = 5
    output_dir: Path = Path("embeddings")
    probe: ProbeConfig = None
    clustering: ClusteringConfig = None
    metadata: MetadataConfig = None
```

#### 2. Main Orchestrator

```python
def log_enhanced_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    epoch: int,
    config: EmbeddingLogConfig,
    probe_set: Optional[Dict[int, List[int]]] = None,
    training_config: Any = None,
    model_config: Any = None,
    optimizer_config: Any = None,
    scheduler_config: Any = None,
) -> Optional[Dict[int, List[int]]]:
    """
    Main function to call from training loop.

    Returns probe_set for reuse across epochs.
    """
```

## Data Saved

### File Structure

```
{output_dir}/
├── class_embeddings_trajectory.pt      # Parametric embeddings [E, C, D]
├── run_metadata.json                   # Experimental metadata
├── probe_set.json                      # Fixed probe sample indices
├── probe_features_epoch_0005.npz       # Feature bank (epoch 5)
├── probe_features_epoch_0010.npz       # Feature bank (epoch 10)
├── ...
└── clustering_metrics.pt               # Clustering scores trajectory
```

### 1. Class Embeddings Trajectory

**File**: `class_embeddings_trajectory.pt`

```python
{
    "epochs": [1, 2, 3, ...],                    # List[int]
    "embeddings": torch.Tensor[E, C, D],         # E=epochs, C=classes, D=emb_dim
    "num_classes": int,
    "emb_dim": int,
}
```

**What it captures**: How the learned class embedding vectors evolve during training.

**Use for**:
- Tracking embedding drift over time
- Visualizing parameter space geometry (9 points in D-dimensional space)
- Basic class separation analysis

### 2. Probe Features

**Files**: `probe_features_epoch_XXXX.npz`

```python
{
    "features": np.ndarray[N, D],        # N feature vectors, D-dimensional
    "sample_ids": np.ndarray[N],         # Which probe sample
    "class_ids": np.ndarray[N],          # Ground truth class
    "epochs": np.ndarray[N],             # Epoch number
    "timesteps": np.ndarray[N],          # Diffusion timestep (t)
    "layer_names": np.ndarray[N],        # Which layer extracted from
    "branches": np.ndarray[N],           # "cond" or "uncond"
}
```

**What it captures**: Feature representations from actual data passing through the model.

**Example**:
- 10 samples/class × 9 classes = 90 samples
- 3 timesteps (early=100, mid=500, late=900)
- 2 branches (cond, uncond)
- 1 layer (e.g., "down_blocks.1")
- **Total**: 90 × 3 × 2 × 1 = 540 feature vectors per epoch

**Use for**:
- Rich latent space visualization (hundreds/thousands of points)
- CFG analysis: compare conditional vs unconditional features
- Temporal dynamics: how features change across diffusion timesteps
- Layer analysis: how features transform through network depth

### 3. Clustering Metrics

**File**: `clustering_metrics.pt`

```python
{
    "epochs": [5, 10, 15, ...],                  # List[int]
    "scores": [                                   # List[Dict]
        {
            "epoch": int,
            "silhouette": float,                  # [-1, 1], higher = better
            "calinski_harabasz": float,           # [0, ∞), higher = better
            "davies_bouldin": float,              # [0, ∞), lower = better
            "n_clusters": int,
            "n_samples": int,
        },
        ...
    ],
    "cluster_centers": np.ndarray[K, D],         # Latest epoch centers
    "cluster_assignments": np.ndarray[N],        # Latest epoch assignments
}
```

**What it captures**: Quantitative quality of latent space organization.

**Metrics**:
- **Silhouette Score**: Measures how similar points are to their own cluster vs other clusters
  - Range: [-1, 1]
  - >0.5 = Good separation
  - <0.2 = Overlapping clusters

- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
  - Range: [0, ∞)
  - Higher = denser, better-separated clusters

- **Davies-Bouldin Index**: Average similarity between each cluster and its most similar cluster
  - Range: [0, ∞)
  - Lower = better separation (0 = perfect)

**Use for**:
- Detecting mode collapse (low silhouette, high DB)
- Monitoring training stability (fluctuating scores)
- Validating visualizations (good figure + good scores = trustworthy)
- Cross-run comparison (which hyperparameters give better geometry?)

### 4. Run Metadata

**File**: `run_metadata.json`

```json
{
  "dataset": {
    "name": "PathMNIST",
    "class_names": ["Adipose", "Background", ...],
    "num_classes": 9
  },
  "training": {
    "guidance_scale": 1.0,
    "guidance_p_uncond": 0.1,
    "epochs": 200,
    "batch_size": 64,
    "seed": 42,
    "ema_use": true,
    "use_min_snr": true,
    "min_snr_gamma": 5.0
  },
  "optimizer": {
    "type": "AdamW",
    "lr": 0.0001,
    "weight_decay": 0.01
  },
  "scheduler": {
    "num_train_timesteps": 1000,
    "beta_schedule": "linear",
    "prediction_type": "epsilon"
  },
  "model": {
    "model_channels": 128,
    "channel_mult": [1, 2, 2, 4],
    "class_embed_dim": 128
  }
}
```

**What it captures**: Complete experimental context.

**Use for**:
- Reproducibility (can you recreate the exact run?)
- Cross-run comparison (how does CFG=1.0 vs 2.0 affect geometry?)
- Publication documentation
- Debugging (what settings produced this result?)

## Usage

### Quick Start (YAML Configuration)

The recommended way to configure embeddings logging is through your experiment's YAML file:

**1. Add logging section to your YAML config:**

```yaml
ccddpm:
  # ... other configs ...

  logging:
    embeddings:
      enabled: true
      log_every_n_epochs: 5
      preset: custom  # Options: minimal, custom, research
```

**2. Run training normally:**

```bash
python -m medsyn.cli.train_ccDDPM experiments/my_experiment/config.yaml
```

That's it! The enhanced embeddings logging will automatically activate based on your YAML configuration.

### Configuration Presets

Three presets are available for quick configuration:

#### Minimal (Fast, Low Disk Usage)
```yaml
logging:
  embeddings:
    enabled: true
    preset: minimal  # 5 samples/class, 1 timestep, no clustering
```

#### Custom (Balanced, Recommended)
```yaml
logging:
  embeddings:
    enabled: true
    preset: custom  # 10 samples/class, 3 timesteps, clustering enabled
```

#### Research (Comprehensive)
```yaml
logging:
  embeddings:
    enabled: true
    preset: research  # 50 samples/class, 9 timesteps, all layers
```

### Manual Configuration

For fine-grained control, specify parameters explicitly (overrides preset):

```yaml
logging:
  embeddings:
    enabled: true
    log_every_n_epochs: 5
    output_dir: embeddings
    save_class_embeddings: true

    probe:
      enabled: true
      samples_per_class: 20
      timesteps: [50, 250, 500, 750, 950]
      layer_names:
        - down_blocks.1
        - down_blocks.3
        - up_blocks.1
      save_both_branches: true
      pooling_strategy: mean

    clustering:
      enabled: true
      n_clusters: null  # auto-detect from num_classes
      algorithm: kmeans
      random_state: 42

    metadata:
      save_metadata: true
      dataset_name: PathMNIST
      class_names:  # Optional
        - Class0
        - Class1
        # ...
```

### Programmatic Configuration (Advanced)

For custom integration or testing, you can also configure programmatically:

```python
from medsyn.models.ccDDPM.config import EmbeddingLogConfig
from medsyn.models.ccDDPM.engine.logging.embeddings import log_enhanced_embeddings

# Configuration is loaded from YAML by default
# But you can override if needed
cfg = load_cfg("config.yaml")
emb_cfg = cfg.ccddpm.logging.embeddings

# Modify if needed
if emb_cfg:
    emb_cfg.log_every_n_epochs = 10

# Use in training loop (handled automatically by train.py)
```

### Integration Status

✅ **Automatic Integration**: If using the standard `medsyn.cli.train_ccDDPM` entry point, embeddings logging is automatically integrated. Just add the YAML configuration and run training.

✅ **DDP Compatible**: Logging only runs on rank 0 to avoid conflicts.

✅ **Backward Compatible**: Existing experiments without logging config continue to work unchanged.

## Analysis Workflow

### 1. Load Data

```python
import torch
import numpy as np
import json
from pathlib import Path

output_dir = Path("outputs/embeddings")

# Class embeddings
emb_traj = torch.load(output_dir / "class_embeddings_trajectory.pt")
epochs = emb_traj["epochs"]
embeddings = emb_traj["embeddings"]  # [E, C, D]

# Metadata
with open(output_dir / "run_metadata.json") as f:
    metadata = json.load(f)
class_names = metadata["dataset"]["class_names"]

# Probe features (specific epoch)
probe = np.load(output_dir / "probe_features_epoch_0010.npz")
features = probe["features"]
class_ids = probe["class_ids"]
timesteps = probe["timesteps"]
branches = probe["branches"]

# Clustering metrics
cluster_traj = torch.load(output_dir / "clustering_metrics.pt")
scores = cluster_traj["scores"]
```

### 2. Filter Features

```python
# Get conditional features at timestep 500
mask = (branches == "cond") & (timesteps == 500)
X = features[mask]
y = class_ids[mask]

# Or compare early vs late diffusion
early_mask = (branches == "cond") & (timesteps == 100)
late_mask = (branches == "cond") & (timesteps == 900)
X_early = features[early_mask]
X_late = features[late_mask]
```

### 3. Visualize

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10, 8))
for c in range(num_classes):
    mask = (y == c)
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                label=class_names[c], alpha=0.6)
plt.legend()
plt.title("Latent Space at t=500 (Epoch 10)")
plt.show()
```

### 4. Analyze Clustering Quality

```python
import pandas as pd

# Convert to DataFrame
df = pd.DataFrame(scores)

# Plot metrics over time
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(df["epoch"], df["silhouette"])
axes[0].set_title("Silhouette Score (higher = better)")
axes[0].axhline(0.5, color='r', linestyle='--', alpha=0.3, label='Good threshold')
axes[0].legend()

axes[1].plot(df["epoch"], df["calinski_harabasz"])
axes[1].set_title("Calinski-Harabasz Index (higher = better)")

axes[2].plot(df["epoch"], df["davies_bouldin"])
axes[2].set_title("Davies-Bouldin Index (lower = better)")
axes[2].axhline(1.0, color='r', linestyle='--', alpha=0.3, label='Acceptable threshold')
axes[2].legend()

plt.tight_layout()
plt.show()
```

### 5. CFG Analysis

```python
# Compare conditional vs unconditional at same timestep
cond_mask = (branches == "cond") & (timesteps == 500)
uncond_mask = (branches == "uncond") & (timesteps == 500)

X_cond = features[cond_mask]
X_uncond = features[uncond_mask]

# Compute CFG direction per class
cfg_directions = []
for c in range(num_classes):
    class_mask_cond = class_ids[cond_mask] == c
    class_mask_uncond = class_ids[uncond_mask] == c

    mean_cond = X_cond[class_mask_cond].mean(axis=0)
    mean_uncond = X_uncond[class_mask_uncond].mean(axis=0)

    cfg_dir = mean_cond - mean_uncond  # Direction CFG pushes
    cfg_directions.append(cfg_dir)

# Visualize CFG effect
# ...
```

## Integration with Training Loop

### ✅ Already Integrated!

If you're using the standard `medsyn.cli.train_ccDDPM` training script, embeddings logging is **already integrated**. Simply add the configuration to your YAML file and run training as normal.

**No code changes needed!**

### What Happens Automatically

When you add the logging configuration to your YAML:

1. **Config Loading**: `config.py` parses the `logging.embeddings` section
2. **Initialization**: `train.py` initializes `probe_set = None`
3. **Logging**: During each epoch, enhanced embeddings are logged (rank 0 only)
4. **File Creation**: Outputs saved to `{output_dir}/embeddings/`

### Implementation Details

For reference, here's how it's integrated in `train.py`:

**Imports** (lines 74-75):
```python
from medsyn.models.ccDDPM.engine.logging.embeddings import log_enhanced_embeddings
from medsyn.models.ccDDPM.config import EmbeddingLogConfig
```

**Initialization** (after config loading):
```python
probe_set = None  # Will be created on first log
```

**Logging Call** (after validation/test):
```python
if is_main_process() and cfg.ccddpm.logging and cfg.ccddpm.logging.embeddings:
    emb_cfg = cfg.ccddpm.logging.embeddings
    if emb_cfg.enabled:
        # Resolve paths, auto-detect metadata, call log_enhanced_embeddings
        probe_set = log_enhanced_embeddings(
            model=base_model,
            dataloader=train_loader,
            noise_scheduler=noise_scheduler,
            device=device,
            epoch=epoch,
            config=emb_cfg_resolved,
            probe_set=probe_set,
            training_config=tcfg,
            model_config=mcfg,
            optimizer_config=optimizer_cfg,
            scheduler_config=scfg,
        )
```

### Custom Integration

If you're using a custom training script, you can integrate manually:

1. Import from config:
   ```python
   from medsyn.models.ccDDPM.config import EmbeddingLogConfig
   from medsyn.models.ccDDPM.engine.logging.embeddings import log_enhanced_embeddings
   ```

2. Load config from YAML:
   ```python
   cfg = load_cfg("config.yaml")
   emb_cfg = cfg.ccddpm.logging.embeddings if cfg.ccddpm.logging else None
   ```

3. Call in training loop (see above)

## Dependencies

Required:
- `torch`
- `numpy`
- `diffusers`

Optional (for clustering metrics):
- `scikit-learn` (for clustering quality scores)

Install:
```bash
pip install scikit-learn
```

If scikit-learn is not available, clustering metrics will be disabled gracefully.

## Performance Considerations

### Computational Cost

| Configuration | Time/Epoch | Disk/Epoch | Notes |
|--------------|-----------|-----------|-------|
| Class embeddings only | ~1s | ~1KB | Legacy mode (always runs) |
| Minimal probe (10 samples/class, 3 timesteps) | ~10s | ~1MB | Recommended default |
| Custom (20 samples/class, 5 timesteps, 3 layers) | ~30s | ~5MB | Good balance |
| Research (50 samples/class, 9 timesteps, 9 layers) | ~2min | ~50MB | Very detailed |

**Tips**:
- Use `log_every_n_epochs` to control frequency (e.g., 5 or 10)
- Set `probe.enabled = False` for quick experiments
- Use `save_both_branches = False` to halve probe size
- Reduce `samples_per_class` if memory is tight

### DDP Considerations

**Important**: Only call from rank 0 (main process):

```python
if is_main_process():
    probe_set = log_enhanced_embeddings(...)
```

**Why**:
- Probe collection runs inference on the model
- If called from all ranks, will trigger collective operations (ALLREDUCE)
- This can cause NCCL timeouts (same issue as visualizations)

**Best practice**:
- Place after all DDP synchronization (after validation/test metrics sync)
- Before barrier if you have one
- Use `base_model` (unwrapped), not DDP wrapper

## Comparison to Original Implementation

| Feature | Original | Enhanced |
|---------|----------|----------|
| Class embeddings | ✅ | ✅ |
| Probe features | ❌ | ✅ |
| Clustering metrics | ❌ | ✅ |
| Metadata | ❌ | ✅ |
| CFG analysis | ❌ | ✅ (cond/uncond branches) |
| Timestep analysis | ❌ | ✅ (multiple timesteps) |
| Layer analysis | ❌ | ✅ (intermediate layers) |
| Points per epoch | 9 | 90-900+ (configurable) |
| Backward compatible | N/A | ✅ (legacy function preserved) |

## References

This implementation is inspired by:

1. **Mabadeje et al.** "Stability and Geometry of Diffusion Latent Spaces"
   - Framework for analyzing latent space stability
   - Clustering metrics for convergence

2. **Latentverse** and similar latent space analysis toolkits
   - Best practices for feature extraction
   - Visualization techniques

3. **Classifier-Free Guidance** (Ho & Salimans, 2022)
   - Conditional vs unconditional branch analysis

## Future Enhancements

Potential additions:
- [ ] GMM clustering (currently only k-means)
- [ ] Automatic layer detection (no manual specification)
- [ ] Anisotropy metrics (eccentricity, aspect ratio)
- [ ] Trajectory smoothness (embedding drift speed)
- [ ] Cross-epoch feature matching (track same samples)
- [ ] Automatic visualization generation
- [ ] YAML config support

## Troubleshooting

**Q**: Clustering metrics show all zeros
**A**: scikit-learn not installed. Run `pip install scikit-learn`

**Q**: Probe collection is slow
**A**: Reduce `samples_per_class`, `timesteps`, or `layer_names`

**Q**: NCCL timeout during embedding logging
**A**: Ensure you're only calling from rank 0 (`if is_main_process()`)

**Q**: Layer extraction fails
**A**: Check layer name format (e.g., "down_blocks.1" not "down_blocks[1]")

**Q**: Out of memory
**A**: Reduce probe size or disable intermediate layers (use only class embeddings)

## License

Same as parent project (medsyn).
