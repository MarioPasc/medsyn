# DistDiff NPZ Integration for MedSyn

This document describes the integration of DistDiff (competitor's data augmentation method) with MedSyn's custom NPZ dataset format.

## Overview

**DistDiff** is a diffusion-based data augmentation method that generates synthetic training data by:
1. Training a guide model on original data
2. Extracting class and sub-class prototypes
3. Using Stable Diffusion with prototype-guided generation
4. Training a final classifier on original + synthetic data

**Integration Goal**: Run DistDiff on MedSyn's PathMNIST NPZ dataset without heavily modifying DistDiff's source code.

## Architecture

### Design Principle: Minimal Invasiveness

We achieved integration with **only ~15 lines of changes** to DistDiff's code:

```
medsyn/
├── adapters/
│   └── distdiff_npz_adapter.py          # Standalone NPZ dataloader (~200 lines)
├── models/distdiff/
│   ├── dataloader.py                     # Modified: +15 lines
│   ├── train.py                          # Unchanged
│   ├── generate_data.py                  # Unchanged
│   └── train_expanded_data_concat_original.py  # Unchanged
config/
└── distdiff_pathmnist.yaml               # Experiment configuration
scripts/
├── distdiff_slurm_job.sh                 # SLURM multi-GPU execution
└── distdiff_local_job.sh                 # Local multi-GPU execution
```

### Key Components

#### 1. NPZ Adapter (`medsyn/adapters/distdiff_npz_adapter.py`)

**Purpose**: Translate MedSyn's NPZ format to DistDiff's expected dataset interface.

**NPZ Structure**:
```python
{
    "train_images": np.ndarray,  # [N, H, W, C] uint8
    "train_labels": np.ndarray,  # [N] int64
    "train_is_synth": np.ndarray,  # [N] bool
    # ... same for val and test
}
```

**Key Features**:
- `NPZDataset`: PyTorch Dataset compatible with DistDiff's training loop
- `load_npz_dataset()`: Loads a single split with optional synthetic filtering
- `create_npz_dataloaders()`: Creates all dataloaders (train/val/test)
- `get_pathmnist_class_names()`: Returns PathMNIST class names in correct order
- **Synthetic Filtering**: Filters `is_synth=True` samples for guide model training

**Why Filter Synthetics?**
- DistDiff's guide model should train on **real data only**
- The generated synthetics are DistDiff's output, not input
- Validation/test sets are always filtered (real data only)

#### 2. DistDiff Modifications (`medsyn/models/distdiff/dataloader.py`)

**Line 11**: Import NPZ adapter
```python
from medsyn.adapters.distdiff_npz_adapter import create_npz_dataloaders
```

**Line 62**: Add PathMNIST NPZ template
```python
"pathmnist_npz": "a colon pathological image of {}.",  # MedSyn NPZ format
```

**Lines 111-122**: NPZ detection and loading
```python
if self.args.dataset == 'pathmnist_npz' or (hasattr(self.args, 'data_dir') and self.args.data_dir.endswith('.npz')):
    outputs = create_npz_dataloaders(
        npz_path=self.args.data_dir if hasattr(self.args, 'data_dir') else self.dataset_path,
        train_transform=self.train_preprocess,
        test_transform=self.test_preprocess,
        train_batch_size=self.args.train_batch_size,
        val_batch_size=self.args.val_batch_size,
        num_workers=8,
        filter_synthetic_train=True,
        filter_synthetic_val=True,
    )
```

**Total Changes**: 15 lines added (3 imports + 1 template + 11 conditional)

## Installation

### Option 1: Separate Conda Environment (Recommended)

DistDiff has specific dependency requirements (especially `transformers==4.19.2`) that may conflict with MedSyn's ccDDPM dependencies. We recommend creating a separate conda environment:

```bash
# Create new environment for DistDiff
conda create -n medsyn-distdiff python=3.10 -y
conda activate medsyn-distdiff

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MedSyn with DistDiff dependencies
cd /path/to/medsyn
pip install -e ".[distdiff]"

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "import accelerate; print('Accelerate:', accelerate.__version__)"
python -c "import open_clip; print('OpenCLIP: OK')"
```

**Important Notes**:
- DistDiff requires `transformers==4.19.2` (older version)
- MedSyn's ccDDPM may use newer `transformers` versions
- Using separate environments avoids version conflicts

### Option 2: Install in Existing MedSyn Environment (May Conflict)

```bash
# Activate your existing medsyn environment
conda activate medsyn

# Install DistDiff dependencies (may downgrade transformers!)
pip install -e ".[distdiff]"
```

**Warning**: This will downgrade `transformers` to 4.19.2, which may break other MedSyn components that require newer versions.

### Option 3: Manual Diffusers Installation (From Git)

If you need the latest diffusers features, install from source:

```bash
conda activate medsyn-distdiff

# Clone and install diffusers from source
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
cd ..

# Install MedSyn with DistDiff deps (without diffusers since we just installed it)
pip install -e ".[distdiff]"
```

## Usage

### Prerequisites

After installation, verify your environment:

```bash
# Activate DistDiff environment
conda activate medsyn-distdiff

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import diffusers; print('Diffusers:', diffusers.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import timm; print('TIMM:', timm.__version__)"
python -c "import open_clip; print('OpenCLIP: OK')"
```

**Expected versions**:
- PyTorch: >= 2.0.0
- Transformers: 4.19.2 (DistDiff requirement)
- Diffusers: >= 0.21.0
- Accelerate: >= 0.12.0

### Quick Start: Local Execution

**1. Prepare Configuration**

Edit `config/distdiff_pathmnist.yaml`:
```yaml
data:
  npz_path: /path/to/your/pathmnist.npz

expansion:
  output_dir: /path/to/output
  num_gpus: 4  # Number of GPUs available
```

**2. Run Pipeline**

```bash
# Full 3-stage pipeline (guide → generate → train)
bash scripts/distdiff_local_job.sh config/distdiff_pathmnist.yaml 4

# Or run stages individually:

# Stage 1: Train guide model
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train.py \
  -a resnet50 \
  -d pathmnist_npz \
  --data_dir /path/to/pathmnist.npz \
  --checkpoint outputs/distdiff/guide_model \
  --train-batch-size 64 \
  --epochs 100

# Stage 2: Generate synthetic data (parallel)
for split in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$split python medsyn/models/distdiff/generate_data.py \
    --guidance_type=transform_guidance \
    -a resnet50 \
    -d pathmnist_npz \
    --data_dir /path/to/pathmnist.npz \
    --output_dir outputs/distdiff/synth/split_$split \
    --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
    --K 5 \
    --num_images_per_prompt 5 \
    --encoder_weight_path outputs/distdiff/guide_model/model_best.pth.tar \
    --total_split 4 \
    --split $split &
done
wait

# Stage 3: Train on expanded data
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train_expanded_data_concat_original.py \
  -d pathmnist_npz \
  --data_dir /path/to/pathmnist.npz \
  --data_expanded_dir outputs/distdiff/synth/split_{0,1,2,3} \
  --checkpoint outputs/distdiff/expanded_classifier \
  --epochs 100
```

### SLURM Cluster Execution

**Submit Job**:
```bash
sbatch scripts/distdiff_slurm_job.sh config/distdiff_pathmnist.yaml
```

**Monitor Progress**:
```bash
# Check job status
squeue -u $USER

# View logs
tail -f distdiff_pathmnist.JOBID.out
tail -f distdiff_pathmnist.JOBID.err
```

**Key Features**:
- LocalScratch staging for fast I/O
- Automatic result syncing
- Parallel generation across 4 GPUs
- Follows picasso_parallel_job.sh patterns

## Configuration Reference

### Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.npz_path` | - | Path to custom NPZ file (required) |
| `expansion.output_dir` | - | Output directory for checkpoints and synthetics |
| `expansion.expand_factor` | 5 | Synthetic images per original image |
| `expansion.num_gpus` | 4 | GPUs for parallel generation |
| `expansion.K` | 5 | Sub-prototypes per class |
| `expansion.pretrained_model` | `CompVis/stable-diffusion-v1-4` | Stable Diffusion model |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expansion.guidance_scale` | 7.5 | Classifier-free guidance scale |
| `expansion.strength` | 0.5 | Noise strength for img2img (0-1) |
| `expansion.guidance_step` | 20 | Timestep to start guidance |
| `expansion.guidance_period` | 2 | Number of guidance steps |
| `expansion.constraint_value` | 0.2 | L-infinity constraint for optimization |
| `expansion.rho` | 10.0 | Learning rate for prototype optimization |

## Pipeline Stages Explained

### Stage 1: Guide Model Training

**Purpose**: Train a ResNet50 classifier to extract feature representations.

**Input**: Original NPZ dataset (real images only, `is_synth=False`)

**Output**:
- Checkpoint: `checkpoints/guide_model/model_best.pth.tar`
- Used for prototype extraction in Stage 2

**Duration**: ~2-3 hours on single V100 GPU

**Key Point**: Only trains on real data (synthetic samples filtered out)

### Stage 2: Synthetic Data Generation

**Purpose**: Generate synthetic images using Stable Diffusion + prototype guidance.

**Process**:
1. Extract global and local prototypes from guide model
2. For each original image:
   - Encode with VAE
   - Add noise (strength=0.5)
   - Denoise with prototype-guided optimization
   - Generate 5 variants
3. Save synthetic images to disk

**Parallelization**:
- Dataset split into N parts (one per GPU)
- Each GPU processes 1/N of images
- Results saved to `synthetic_data/split_{0,1,...,N-1}/`

**Duration**: ~4-8 hours on 4x V100 GPUs

**Output**: 5x expanded dataset (5 synthetic per original)

### Stage 3: Classifier Training on Expanded Data

**Purpose**: Train final classifier on original + synthetic data.

**Input**:
- Original NPZ dataset
- Synthetic data from all splits

**Process**:
1. Concatenate original and synthetic datasets
2. Train ResNet50 from scratch
3. Evaluate on test set

**Duration**: ~3-4 hours on single V100 GPU

**Output**: Final classifier at `checkpoints/classifier_on_expanded/model_best.pth.tar`

## Expected Outputs

```
outputs/distdiff_pathmnist/
├── checkpoints/
│   ├── guide_model/
│   │   ├── model_best.pth.tar
│   │   └── checkpoint.pth.tar
│   └── classifier_on_expanded/
│       ├── model_best.pth.tar
│       └── checkpoint.pth.tar
├── synthetic_data/
│   ├── split_0/
│   │   ├── class_0/
│   │   │   └── *.png
│   │   ├── class_1/
│   │   └── ...
│   ├── split_1/
│   ├── split_2/
│   └── split_3/
└── logs/
    ├── stage1_train_guide.log
    ├── generation_split_0.log
    ├── generation_split_1.log
    ├── generation_split_2.log
    ├── generation_split_3.log
    └── stage3_train_expanded.log
```

## Troubleshooting

### Issue: "Dataset not supported" error

**Solution**: Verify dataset name is `pathmnist_npz` or `--data_dir` ends with `.npz`

```bash
# Correct
python train.py -d pathmnist_npz --data_dir /path/to/file.npz

# Also correct (auto-detected)
python train.py -d anything --data_dir /path/to/file.npz
```

### Issue: OOM during generation

**Solutions**:
1. Reduce batch size: `--train_batch_size 1` (default)
2. Enable gradient checkpointing: `--gradient_checkpointing` (default)
3. Use smaller Stable Diffusion model
4. Increase `--total_split` to process smaller chunks

### Issue: Slow generation

**Expected**: Generation is I/O and compute intensive
- ~10-30 seconds per image (depends on GPU)
- With 5x expansion and 10K images: ~14-42 hours total
- Use multiple GPUs to parallelize

**Optimization**:
- Use faster GPUs (A100 > V100 > RTX 3090)
- Increase `--total_split` for better parallelization
- Use SD v1.4 instead of v2.1 (faster but lower quality)

### Issue: Class names mismatch

**Verify PathMNIST class map**:
```python
from medsyn.data.yolo_dataset import build_pathmnist_class_map
print(build_pathmnist_class_map())
```

Should output:
```
{0: 'adipose', 1: 'background', 2: 'debris', ...}
```

## Comparison with Copilot's Proposal

| Aspect | Copilot's Proposal | Our Implementation |
|--------|-------------------|-------------------|
| DistDiff Changes | ~100+ lines to dataloader.py | ~15 lines |
| Architecture | Modified DistDiff directly | Standalone adapter |
| Script Pattern | Custom Python wrappers | Follows picasso_parallel_job.sh |
| Config | New YAML + wrapper classes | Simple YAML parsing |
| Synthetics Handling | Not addressed | Filters during guide training |
| Execution | Complex wrapper system | Direct DistDiff script calls |
| Maintenance | Tightly coupled | Loosely coupled |

**Key Improvement**: Clean separation of concerns - adapter lives in medsyn codebase, not DistDiff.

## Citation

If you use this integration in your research:

```bibtex
@software{medsyn_distdiff_integration,
  title={DistDiff NPZ Integration for MedSyn},
  author={MedSyn Project},
  year={2025},
  note={Integration of DistDiff with MedSyn's custom NPZ dataset format}
}
```

DistDiff paper:
```bibtex
@article{zhu2024distdiff,
  title={DistDiff: Prototypical Distribution-Aware Diffusion for Data Augmentation},
  author={Zhu, Haowei and ...},
  journal={arXiv preprint},
  year={2024}
}
```

## Support

For issues or questions:
1. Check this README first
2. Review logs in `outputs/distdiff_pathmnist/logs/`
3. Verify NPZ structure matches expected format
4. Open an issue in the MedSyn repository

## Appendix: NPZ Format Specification

### Structure

```python
import numpy as np

# Loading
data = np.load('pathmnist.npz')

# Keys (required)
train_images: np.ndarray  # [N_train, H, W, C] uint8, values 0-255
train_labels: np.ndarray  # [N_train] int64, values 0 to num_classes-1
train_is_synth: np.ndarray  # [N_train] bool, True for synthetic samples

val_images: np.ndarray    # [N_val, H, W, C] uint8
val_labels: np.ndarray    # [N_val] int64
val_is_synth: np.ndarray  # [N_val] bool

test_images: np.ndarray   # [N_test, H, W, C] uint8
test_labels: np.ndarray   # [N_test] int64
test_is_synth: np.ndarray  # [N_test] bool
```

### Validation

```python
# Verify NPZ structure
data = np.load('pathmnist.npz')

required_keys = [
    'train_images', 'train_labels', 'train_is_synth',
    'val_images', 'val_labels', 'val_is_synth',
    'test_images', 'test_labels', 'test_is_synth',
]

for key in required_keys:
    assert key in data, f"Missing key: {key}"

# Verify shapes
assert data['train_images'].ndim == 4  # [N, H, W, C]
assert data['train_labels'].ndim == 1  # [N]
assert len(data['train_images']) == len(data['train_labels'])
assert data['train_images'].dtype == np.uint8
assert data['train_labels'].dtype == np.int64
assert data['train_is_synth'].dtype == bool

print("✓ NPZ structure valid")
```

## Changelog

### 2025-01-XX - Initial Integration
- Created NPZ adapter for DistDiff
- Minimal modifications to DistDiff dataloader (~15 lines)
- Added SLURM and local execution scripts
- Implemented synthetic image filtering for guide model training
- Created comprehensive documentation
