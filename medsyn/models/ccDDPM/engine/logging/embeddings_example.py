"""
Example usage of enhanced embeddings logging.

This demonstrates how to integrate the enhanced logging system into your training loop.
"""

from pathlib import Path
from medsyn.models.ccDDPM.engine.logging.embeddings import (
    EmbeddingLogConfig,
    ProbeConfig,
    ClusteringConfig,
    MetadataConfig,
    log_enhanced_embeddings,
)


# ============================================================================
# Example 1: Minimal Configuration (uses sensible defaults)
# ============================================================================

def example_minimal():
    """
    Minimal setup - logs everything with defaults.

    This will:
    - Save class embeddings every epoch
    - Collect probe features every 5 epochs (10 samples/class, 3 timesteps)
    - Compute clustering metrics every 5 epochs
    - Save metadata on first epoch
    """
    config = EmbeddingLogConfig(
        output_dir=Path("outputs/embeddings"),
        log_every_n_epochs=5,
    )

    # In your training loop:
    # probe_set = None
    # for epoch in range(1, max_epochs + 1):
    #     # ... training code ...
    #
    #     probe_set = log_enhanced_embeddings(
    #         model=model,
    #         dataloader=train_loader,
    #         noise_scheduler=noise_scheduler,
    #         device=device,
    #         epoch=epoch,
    #         config=config,
    #         probe_set=probe_set,  # Reuse across epochs
    #         training_config=tcfg,
    #         model_config=mcfg,
    #         optimizer_config=optimizer_cfg,
    #         scheduler_config=scfg,
    #     )


# ============================================================================
# Example 2: Custom Configuration
# ============================================================================

def example_custom():
    """
    Custom configuration with specific settings.

    This demonstrates:
    - More probe samples per class
    - Specific timesteps to analyze (early, mid, late diffusion)
    - Specific layers to extract features from
    - Custom clustering settings
    - Dataset metadata
    """

    # Configure probe bank collection
    probe_config = ProbeConfig(
        enabled=True,
        samples_per_class=20,  # More samples for better statistics
        timesteps=[50, 250, 500, 750, 950],  # 5 timesteps across diffusion process
        layer_names=["down_blocks.0", "down_blocks.2", "up_blocks.1"],  # Extract from 3 layers
        save_both_branches=True,  # Save both conditional and unconditional
        pooling_strategy="mean",  # Global average pooling for spatial dimensions
    )

    # Configure clustering analysis
    clustering_config = ClusteringConfig(
        enabled=True,
        n_clusters=None,  # Auto-detect from num_classes
        algorithm="kmeans",
        random_state=42,
        compute_scores=True,
    )

    # Configure metadata
    metadata_config = MetadataConfig(
        save_metadata=True,
        class_names=[
            "Adipose", "Background", "Debris", "Lymphocytes",
            "Mucus", "Smooth Muscle", "Normal Colon Mucosa",
            "Cancer-Associated Stroma", "Colorectal Adenocarcinoma Epithelium"
        ],  # PathMNIST class names
        dataset_name="PathMNIST",
    )

    # Master configuration
    config = EmbeddingLogConfig(
        output_dir=Path("outputs/embeddings_detailed"),
        log_every_n_epochs=10,  # Log detailed diagnostics every 10 epochs
        probe=probe_config,
        clustering=clustering_config,
        metadata=metadata_config,
        save_class_embeddings=True,
    )

    return config


# ============================================================================
# Example 3: Lightweight Configuration (fast, minimal disk usage)
# ============================================================================

def example_lightweight():
    """
    Lightweight configuration for quick experiments.

    This is useful when:
    - Disk space is limited
    - Training time is critical
    - You only need basic embedding trajectories
    """

    probe_config = ProbeConfig(
        enabled=True,
        samples_per_class=5,  # Fewer samples
        timesteps=[500],  # Only middle timestep
        layer_names=[],  # Only class embeddings (no intermediate layers)
        save_both_branches=False,  # Only conditional branch
        pooling_strategy="mean",
    )

    clustering_config = ClusteringConfig(
        enabled=False,  # Disable clustering to save time
    )

    config = EmbeddingLogConfig(
        output_dir=Path("outputs/embeddings_light"),
        log_every_n_epochs=20,  # Log less frequently
        probe=probe_config,
        clustering=clustering_config,
        save_class_embeddings=True,
    )

    return config


# ============================================================================
# Example 4: Research Configuration (maximum detail)
# ============================================================================

def example_research():
    """
    Comprehensive configuration for research/publication.

    This captures everything needed for:
    - Detailed latent space analysis
    - Cross-run comparison
    - Reproducibility
    - Publication-quality figures
    """

    probe_config = ProbeConfig(
        enabled=True,
        samples_per_class=50,  # Large probe bank
        timesteps=[10, 50, 100, 250, 500, 750, 900, 950, 990],  # Dense sampling
        layer_names=[
            "down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
            "mid_block",
            "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"
        ],  # All major layers
        save_both_branches=True,
        pooling_strategy="mean",
    )

    clustering_config = ClusteringConfig(
        enabled=True,
        n_clusters=None,
        algorithm="kmeans",
        random_state=42,
        compute_scores=True,
    )

    metadata_config = MetadataConfig(
        save_metadata=True,
        class_names=[
            "Adipose", "Background", "Debris", "Lymphocytes",
            "Mucus", "Smooth Muscle", "Normal Colon Mucosa",
            "Cancer-Associated Stroma", "Colorectal Adenocarcinoma Epithelium"
        ],
        dataset_name="PathMNIST",
    )

    config = EmbeddingLogConfig(
        output_dir=Path("outputs/embeddings_research"),
        log_every_n_epochs=5,  # Frequent logging
        probe=probe_config,
        clustering=clustering_config,
        metadata=metadata_config,
        save_class_embeddings=True,
    )

    return config


# ============================================================================
# Example 5: Integration into existing training loop
# ============================================================================

def example_integration():
    """
    Shows how to integrate into train.py
    """

    # At the top of train.py, import:
    # from medsyn.models.ccDDPM.engine.logging.embeddings import (
    #     EmbeddingLogConfig, ProbeConfig, ClusteringConfig, MetadataConfig,
    #     log_enhanced_embeddings
    # )

    # After creating model, optimizer, etc., create embedding config:
    embedding_config = EmbeddingLogConfig(
        output_dir=Path("outputs/my_experiment/embeddings"),
        log_every_n_epochs=5,
        # Use defaults for probe, clustering, metadata
    )

    # Initialize probe set (will be created on first call)
    probe_set = None

    # In the training loop (after validation/test):
    # for epoch in range(1, max_epochs + 1):
    #     # ... training code ...
    #     # ... validation code ...
    #     # ... test code ...
    #
    #     # Enhanced embeddings logging (only on rank 0 in DDP)
    #     if is_main_process():
    #         probe_set = log_enhanced_embeddings(
    #             model=base_model,  # Use unwrapped model (not DDP wrapper)
    #             dataloader=train_loader,
    #             noise_scheduler=noise_scheduler,
    #             device=device,
    #             epoch=epoch,
    #             config=embedding_config,
    #             probe_set=probe_set,  # Reuse across epochs
    #             training_config=tcfg,
    #             model_config=mcfg,
    #             optimizer_config=optimizer_cfg,
    #             scheduler_config=scfg,
    #         )

    # This will create the following files:
    # outputs/my_experiment/embeddings/
    #   ├── class_embeddings_trajectory.pt      # Every epoch
    #   ├── run_metadata.json                   # Once (epoch 1)
    #   ├── probe_set.json                      # Once (first detailed log)
    #   ├── probe_features_epoch_0005.npz       # Every log_every_n_epochs
    #   ├── probe_features_epoch_0010.npz
    #   ├── probe_features_epoch_0015.npz
    #   ├── ...
    #   └── clustering_metrics.pt               # Every log_every_n_epochs


# ============================================================================
# Example 6: Analyzing saved data
# ============================================================================

def example_analysis():
    """
    Shows how to load and analyze saved embeddings data.
    """
    import torch
    import numpy as np
    import json

    output_dir = Path("outputs/embeddings")

    # Load class embedding trajectory
    emb_traj = torch.load(output_dir / "class_embeddings_trajectory.pt")
    epochs = emb_traj["epochs"]  # [E]
    embeddings = emb_traj["embeddings"]  # [E, num_classes, emb_dim]

    print(f"Logged {len(epochs)} epochs")
    print(f"Embedding shape: {embeddings.shape}")

    # Load metadata
    with open(output_dir / "run_metadata.json") as f:
        metadata = json.load(f)

    print(f"Dataset: {metadata['dataset']['name']}")
    print(f"Classes: {metadata['dataset']['class_names']}")

    # Load probe features for a specific epoch
    probe_data = np.load(output_dir / "probe_features_epoch_0010.npz")
    features = probe_data["features"]  # [n_records, feature_dim]
    class_ids = probe_data["class_ids"]  # [n_records]
    timesteps = probe_data["timesteps"]  # [n_records]
    layer_names = probe_data["layer_names"]  # [n_records]
    branches = probe_data["branches"]  # [n_records]

    print(f"Probe features shape: {features.shape}")

    # Filter to specific conditions
    mask = (
        (branches == "cond") &  # Conditional branch
        (timesteps == 500) &    # Middle timestep
        (layer_names == "class_embed")  # Class embedding layer
    )

    filtered_features = features[mask]
    filtered_classes = class_ids[mask]

    print(f"Filtered features: {filtered_features.shape}")

    # Load clustering metrics
    cluster_traj = torch.load(output_dir / "clustering_metrics.pt")
    cluster_epochs = cluster_traj["epochs"]
    cluster_scores = cluster_traj["scores"]

    for i, (ep, scores) in enumerate(zip(cluster_epochs, cluster_scores)):
        print(f"Epoch {ep}: "
              f"Silhouette={scores['silhouette']:.4f}, "
              f"CH={scores['calinski_harabasz']:.2f}, "
              f"DB={scores['davies_bouldin']:.4f}")


if __name__ == "__main__":
    # Generate example configurations
    print("=" * 80)
    print("Example 1: Minimal Configuration")
    print("=" * 80)
    example_minimal()

    print("\n" + "=" * 80)
    print("Example 2: Custom Configuration")
    print("=" * 80)
    config = example_custom()
    print(f"Output dir: {config.output_dir}")
    print(f"Log frequency: every {config.log_every_n_epochs} epochs")
    print(f"Probe samples/class: {config.probe.samples_per_class}")
    print(f"Probe timesteps: {config.probe.timesteps}")
    print(f"Probe layers: {config.probe.layer_names}")

    print("\n" + "=" * 80)
    print("Example 3: Lightweight Configuration")
    print("=" * 80)
    config_light = example_lightweight()
    print(f"Output dir: {config_light.output_dir}")
    print(f"Clustering enabled: {config_light.clustering.enabled}")

    print("\n" + "=" * 80)
    print("Example 4: Research Configuration")
    print("=" * 80)
    config_research = example_research()
    print(f"Probe timesteps: {len(config_research.probe.timesteps)} timesteps")
    print(f"Probe layers: {len(config_research.probe.layer_names)} layers")
    print(f"Total probe records per epoch: ~{config_research.probe.samples_per_class * 9 * len(config_research.probe.timesteps) * len(config_research.probe.layer_names) * 2}")
