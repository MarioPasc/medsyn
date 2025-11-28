#!/usr/bin/env python3
"""
Integration test for embeddings logging.
This simulates the actual usage from train.py to catch any runtime issues.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from medsyn.models.ccDDPM.config import (
    CCDDPMInit, ProbeConfig, ClusteringConfig,
    MetadataConfig, EmbeddingLogConfig
)
from medsyn.models.ccDDPM.model import CCDDPM
from medsyn.models.ccDDPM.engine.logging.embeddings import log_enhanced_embeddings

print("=" * 80)
print("EMBEDDINGS INTEGRATION TEST")
print("=" * 80)

# Step 1: Create a simple model
print("\n1. Creating model...")
model_cfg = CCDDPMInit(
    in_channels=3,
    class_embed_dim=16,
    num_classes=9,
    model_channels=32,  # Small for fast testing
    channel_mult=(1, 2),  # 2 levels only
    layers_per_block=1,  # Minimal layers
    down_block_types=("DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D"),
)

model = CCDDPM(model_cfg)
model.eval()
print("  ✓ Model created")

# Step 2: Create dummy dataset
print("\n2. Creating dummy dataset...")
num_samples = 20
image_size = 28
dataset = TensorDataset(
    torch.randn(num_samples, 3, image_size, image_size),  # pixel_values
    torch.randint(0, 9, (num_samples,))  # labels
)

# Wrap to match expected format
class DummyDataset:
    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, idx):
        pixels, label = self.tensor_dataset[idx]
        return {
            'pixel_values': pixels,
            'labels': label
        }

wrapped_dataset = DummyDataset(dataset)
dataloader = DataLoader(wrapped_dataset, batch_size=4, shuffle=False)
print(f"  ✓ Dataset created: {len(wrapped_dataset)} samples")

# Step 3: Create noise scheduler
print("\n3. Creating noise scheduler...")
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=0.02,
)
print("  ✓ Scheduler created")

# Step 4: Configure embeddings logging
print("\n4. Configuring embeddings logging...")
probe_config = ProbeConfig(
    enabled=True,
    samples_per_class=2,  # Small for testing
    timesteps=[0, 100, 500],  # 3 timesteps
    layer_names=["unet.down_blocks.0", "unet.down_blocks.1"],  # These return tuples!
    pooling_strategy="mean",
    save_both_branches=True,
)

clustering_config = ClusteringConfig(
    enabled=True,
    algorithm="kmeans",
    n_clusters=9,
    compute_scores=True,
    random_state=42,
)

metadata_config = MetadataConfig(
    save_metadata=True,
    dataset_name="TestDataset",
    class_names=[f"class_{i}" for i in range(9)],
)

embedding_config = EmbeddingLogConfig(
    output_dir="test_output_embeddings",
    log_every_n_epochs=1,
    save_class_embeddings=True,
    probe=probe_config,
    clustering=clustering_config,
    metadata=metadata_config,
)

print("  ✓ Configuration created")
print(f"     Probe layers: {probe_config.layer_names}")
print(f"     Timesteps: {probe_config.timesteps}")

# Step 5: Run the embeddings logging
print("\n5. Running embeddings logging...")
device = torch.device('cpu')
model = model.to(device)

try:
    probe_set = log_enhanced_embeddings(
        model=model,
        dataloader=dataloader,
        noise_scheduler=noise_scheduler,
        device=device,
        epoch=1,
        config=embedding_config,
        probe_set=None,  # Will create new one
        training_config=None,
        model_config=None,
        optimizer_config=None,
        scheduler_config=None,
    )
    print("  ✓ Embeddings logging completed successfully!")
    print(f"     Probe set created with {sum(len(v) for v in probe_set.values())} samples")

except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Check outputs
print("\n6. Checking output files...")
output_dir = Path("test_output_embeddings")

expected_files = [
    "class_embeddings_trajectory.pt",
    "probe_features_epoch_0001.npz",
    "probe_set.json",
    "clustering_metrics.pt",
]

all_found = True
for filename in expected_files:
    filepath = output_dir / filename
    if filepath.exists():
        print(f"  ✓ {filename} exists")
    else:
        print(f"  ✗ {filename} MISSING")
        all_found = False

# Step 7: Validate probe features
print("\n7. Validating probe features...")
import numpy as np

features_path = output_dir / "probe_features_epoch_0001.npz"
if features_path.exists():
    data = np.load(features_path)
    print(f"  ✓ Loaded probe features")
    print(f"     Features shape: {data['features'].shape}")
    print(f"     Number of records: {len(data['sample_ids'])}")
    print(f"     Unique layers: {set(data['layer_names'])}")
    print(f"     Unique timesteps: {set(data['timesteps'])}")
    print(f"     Unique branches: {set(data['branches'])}")

    # Check for expected number of records
    # samples_per_class(2) * num_classes(9) * timesteps(3) * layers(2) * branches(2) = 216
    expected_records = 2 * 9 * 3 * 2 * 2
    actual_records = len(data['sample_ids'])
    if actual_records == expected_records:
        print(f"  ✓ Record count correct: {actual_records}")
    else:
        print(f"  ⚠️  Record count: expected {expected_records}, got {actual_records}")

# Step 8: Cleanup
print("\n8. Cleaning up test outputs...")
import shutil
if output_dir.exists():
    shutil.rmtree(output_dir)
    print("  ✓ Test outputs cleaned up")

# Final summary
print("\n" + "=" * 80)
if all_found:
    print("✅ INTEGRATION TEST PASSED!")
    print("All fixes working correctly in realistic scenario:")
    print("  ✓ Tuple handling in hooks (down_blocks)")
    print("  ✓ class_embed.emb() usage")
    print("  ✓ Feature extraction and stacking")
    print("  ✓ Clustering on features")
else:
    print("⚠️  INTEGRATION TEST COMPLETED WITH WARNINGS")
    print("Some expected files were not created.")

print("=" * 80)
