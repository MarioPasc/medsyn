"""
Enhanced latent space diagnostics for ccDDPM.

This module provides comprehensive logging of:
  1. Class embedding trajectories (parametric class space)
  2. Feature probe banks (how embeddings interact with data and diffusion time)
  3. Clustering quality metrics (latent space organization)
  4. Rich metadata (for interpretability and reproducibility)

Based on analysis frameworks from:
  - Mabadeje et al. "Stability and Geometry of Diffusion Latent Spaces"
  - Modern latent space analysis toolkits (Latentverse, etc.)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Literal
from dataclasses import dataclass, asdict
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Clustering and dimensionality reduction
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("medsyn.ccddpm.embeddings")

# Import configuration dataclasses from centralized config module
from medsyn.models.ccDDPM.config import (
    ProbeConfig,
    ClusteringConfig,
    MetadataConfig,
    EmbeddingLogConfig
)


# ============================================================================
# Probe Bank: Feature Extraction
# ============================================================================

@dataclass
class FeatureRecord:
    """Single feature vector record with metadata.

    Stores a single extracted feature vector along with all context needed
    to interpret it: which sample, class, epoch, timestep, layer, and branch.
    """
    sample_id: int
    class_id: int
    epoch: int
    timestep: int
    layer_name: str
    branch: Literal["cond", "uncond"]
    feature: np.ndarray  # Shape: [d] - flattened/pooled feature vector

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        d["feature"] = d["feature"].tolist()  # Convert numpy to list for JSON
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureRecord":
        """Reconstruct from dict."""
        d["feature"] = np.array(d["feature"])
        return cls(**d)


def create_balanced_probe_set(
    dataloader: DataLoader,
    samples_per_class: int,
    num_classes: int,
    seed: int = 42
) -> Dict[int, List[int]]:
    """Create a balanced probe set with fixed samples per class.

    Selects a stratified sample of indices from the dataset, ensuring equal
    representation from each class. This probe set is reused across epochs
    for consistent feature tracking.

    Args:
        dataloader: Training dataloader
        samples_per_class: Number of samples to select per class
        num_classes: Total number of classes
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping class_id -> list of dataset indices

    Saves:
        - Probe set indices organized by class
        - Total: samples_per_class × num_classes samples
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect all indices by class
    class_to_indices: Dict[int, List[int]] = {c: [] for c in range(num_classes)}

    # Iterate through dataset to find samples of each class
    dataset = dataloader.dataset
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample["labels"].item() if isinstance(sample["labels"], torch.Tensor) else sample["labels"]

        # Skip unconditional samples (label = -1)
        if label >= 0:
            class_to_indices[label].append(idx)

    # Sample balanced set
    probe_set = {}
    for class_id, indices in class_to_indices.items():
        if len(indices) < samples_per_class:
            logger.warning(
                f"Class {class_id} has only {len(indices)} samples, "
                f"requested {samples_per_class}. Using all available."
            )
            probe_set[class_id] = indices
        else:
            # Random sample without replacement
            sampled = np.random.choice(indices, samples_per_class, replace=False).tolist()
            probe_set[class_id] = sampled

    total_samples = sum(len(indices) for indices in probe_set.values())
    logger.info(
        f"Created balanced probe set: {total_samples} samples "
        f"({samples_per_class} per class × {num_classes} classes)"
    )

    return probe_set


def extract_features_from_layer(
    model: nn.Module,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: Optional[torch.Tensor],
    layer_name: str,
    pooling: Literal["mean", "max", "flatten"] = "mean"
) -> torch.Tensor:
    """Extract intermediate features from a specific layer.

    Uses forward hooks to capture activations from the specified layer during
    a forward pass. Applies spatial pooling to reduce dimensionality.

    Args:
        model: The ccDDPM model
        x_t: Noisy input [B, C, H, W]
        timesteps: Timestep tensor [B]
        class_labels: Class labels [B] or None for unconditional
        layer_name: Name of layer to extract (e.g., "down_blocks.1")
        pooling: Spatial pooling strategy

    Returns:
        Pooled features [B, D] where D depends on layer and pooling

    Saves:
        - Intermediate layer activations after spatial pooling
        - Both conditional (with class_labels) and unconditional (class_labels=None) branches
    """
    features = {}

    def hook_fn(module, input, output):
        # Store output activation
        # Handle case where output is a tuple (e.g., from attention layers)
        if isinstance(output, tuple):
            features['activation'] = output[0]
        else:
            features['activation'] = output

    # Register hook
    try:
        # Navigate to the layer (e.g., "down_blocks.1" -> model.down_blocks[1])
        parts = layer_name.split('.')
        target_module = model
        for part in parts:
            if part.isdigit():
                target_module = target_module[int(part)]
            else:
                target_module = getattr(target_module, part)

        handle = target_module.register_forward_hook(hook_fn)
    except (AttributeError, IndexError, TypeError) as e:
        logger.error(f"Failed to register hook on layer '{layer_name}': {e}")
        # Return zero features as fallback
        return torch.zeros(x_t.size(0), 1, device=x_t.device)

    # Forward pass to capture features
    try:
        with torch.no_grad():
            _ = model(x_t, timesteps, class_labels)

        # Extract activation
        if 'activation' not in features:
            logger.warning(f"No activation captured from layer '{layer_name}'")
            return torch.zeros(x_t.size(0), 1, device=x_t.device)

        act = features['activation']

        # Apply spatial pooling if activation is 4D [B, C, H, W]
        if act.dim() == 4:
            if pooling == "mean":
                feat = act.mean(dim=[2, 3])  # [B, C]
            elif pooling == "max":
                feat = act.max(dim=3)[0].max(dim=2)[0]  # [B, C]
            elif pooling == "flatten":
                feat = act.flatten(1)  # [B, C*H*W]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
        elif act.dim() == 3:
            # Sequence output [B, T, C] - pool over sequence
            feat = act.mean(dim=1)
        elif act.dim() == 2:
            # Already flat [B, C]
            feat = act
        elif act.dim() == 1:
            # 1D tensor - add batch dimension
            logger.warning(f"1D activation shape: {act.shape}, adding batch dimension")
            feat = act.unsqueeze(0)
        else:
            # 5D or higher - flatten everything except batch
            logger.warning(f"Unexpected activation shape: {act.shape}, flattening")
            feat = act.flatten(1) if act.dim() > 1 else act.unsqueeze(0)

        return feat

    finally:
        handle.remove()


def collect_probe_features(
    model: nn.Module,
    dataloader: DataLoader,
    probe_set: Dict[int, List[int]],
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    epoch: int,
    config: ProbeConfig,
) -> List[FeatureRecord]:
    """Collect feature vectors from the probe set at current epoch.

    For each probe sample, extracts features from specified layers at specified
    timesteps, for both conditional and unconditional branches (if CFG enabled).

    Args:
        model: The ccDDPM model
        dataloader: Training dataloader
        probe_set: Dict of class_id -> list of dataset indices
        noise_scheduler: DDPM noise scheduler
        device: Compute device
        epoch: Current epoch number
        config: Probe configuration

    Returns:
        List of FeatureRecord objects

    Saves:
        - Feature vectors for each (sample, class, timestep, layer, branch) combination
        - Metadata: sample_id, class_id, epoch, timestep, layer_name, branch
        - Features are spatially pooled to fixed-size vectors
    """
    model.eval()
    records = []

    dataset = dataloader.dataset

    # Flatten probe set to list of (class_id, sample_idx)
    probe_samples = []
    for class_id, indices in probe_set.items():
        for idx in indices:
            probe_samples.append((class_id, idx))

    logger.info(f"Collecting features from {len(probe_samples)} probe samples...")

    with torch.no_grad():
        for sample_id, (class_id, dataset_idx) in enumerate(probe_samples):
            # Load sample
            sample = dataset[dataset_idx]
            x0 = sample["pixel_values"].unsqueeze(0).to(device)  # [1, C, H, W]
            label = torch.tensor([class_id], device=device)

            # For each timestep
            for t_val in config.timesteps:
                t = torch.tensor([t_val], device=device, dtype=torch.long)

                # Add noise to create x_t
                noise = torch.randn_like(x0)
                x_t = noise_scheduler.add_noise(x0, noise, t)

                # Extract from each layer
                layer_names = config.layer_names if config.layer_names else [""]

                for layer_name in layer_names:
                    # Conditional branch
                    if layer_name:
                        feat_cond = extract_features_from_layer(
                            model, x_t, t, label, layer_name, config.pooling_strategy
                        )
                    else:
                        # No layer specified, use class embedding directly
                        feat_cond = model.class_embed.emb(label)  # [1, D]

                    records.append(FeatureRecord(
                        sample_id=sample_id,
                        class_id=class_id,
                        epoch=epoch,
                        timestep=t_val,
                        layer_name=layer_name if layer_name else "class_embed",
                        branch="cond",
                        feature=feat_cond.cpu().numpy().flatten()
                    ))

                    # Unconditional branch (if enabled)
                    if config.save_both_branches:
                        if layer_name:
                            feat_uncond = extract_features_from_layer(
                                model, x_t, t, None, layer_name, config.pooling_strategy
                            )
                        else:
                            # Unconditional uses -1 or special token
                            uncond_label = torch.tensor([-1], device=device)
                            feat_uncond = model.class_embed.emb(uncond_label)

                        records.append(FeatureRecord(
                            sample_id=sample_id,
                            class_id=class_id,
                            epoch=epoch,
                            timestep=t_val,
                            layer_name=layer_name if layer_name else "class_embed",
                            branch="uncond",
                            feature=feat_uncond.cpu().numpy().flatten()
                        ))

    logger.info(f"Collected {len(records)} feature records for epoch {epoch}")
    return records


# ============================================================================
# Clustering Quality Metrics
# ============================================================================

@dataclass
class ClusteringMetrics:
    """Clustering quality metrics for latent space organization analysis.

    Quantifies how well-separated and compact the class clusters are in the
    latent space. Higher silhouette and CH scores indicate better separation;
    lower DB score indicates better separation.
    """
    epoch: int
    silhouette: float  # [-1, 1], higher is better (well-separated clusters)
    calinski_harabasz: float  # [0, inf), higher is better (dense, well-separated)
    davies_bouldin: float  # [0, inf), lower is better (compact, well-separated)
    cluster_centers: np.ndarray  # [n_clusters, d]
    cluster_assignments: np.ndarray  # [n_samples]
    n_clusters: int
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "epoch": self.epoch,
            "silhouette": float(self.silhouette),
            "calinski_harabasz": float(self.calinski_harabasz),
            "davies_bouldin": float(self.davies_bouldin),
            "n_clusters": self.n_clusters,
            "n_samples": self.n_samples,
        }


def compute_clustering_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    epoch: int,
    config: ClusteringConfig,
) -> ClusteringMetrics:
    """Compute clustering quality metrics on feature vectors.

    Fits a clustering algorithm (k-means or GMM) to the features and evaluates
    how well the discovered clusters align with the true class labels.

    Args:
        features: Feature matrix [n_samples, d]
        labels: True class labels [n_samples]
        epoch: Current epoch
        config: Clustering configuration

    Returns:
        ClusteringMetrics object with scores and cluster info

    Saves:
        - Silhouette score (cluster cohesion and separation)
        - Calinski-Harabasz score (variance ratio criterion)
        - Davies-Bouldin score (average similarity between clusters)
        - Cluster centers and assignments
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not available, skipping clustering metrics")
        return ClusteringMetrics(
            epoch=epoch,
            silhouette=0.0,
            calinski_harabasz=0.0,
            davies_bouldin=0.0,
            cluster_centers=np.zeros((1, features.shape[1])),
            cluster_assignments=np.zeros(len(features)),
            n_clusters=0,
            n_samples=len(features)
        )

    n_clusters = config.n_clusters if config.n_clusters else len(np.unique(labels))

    # Fit clustering
    if config.algorithm == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)
        cluster_assignments = clusterer.fit_predict(features)
        cluster_centers = clusterer.cluster_centers_
    else:
        # GMM not implemented yet, fall back to kmeans
        logger.warning(f"Algorithm '{config.algorithm}' not implemented, using kmeans")
        clusterer = KMeans(n_clusters=n_clusters, random_state=config.random_state, n_init=10)
        cluster_assignments = clusterer.fit_predict(features)
        cluster_centers = clusterer.cluster_centers_

    # Compute quality scores
    if config.compute_scores and len(np.unique(cluster_assignments)) > 1:
        sil = silhouette_score(features, cluster_assignments)
        ch = calinski_harabasz_score(features, cluster_assignments)
        db = davies_bouldin_score(features, cluster_assignments)
    else:
        sil = ch = db = 0.0

    return ClusteringMetrics(
        epoch=epoch,
        silhouette=sil,
        calinski_harabasz=ch,
        davies_bouldin=db,
        cluster_centers=cluster_centers,
        cluster_assignments=cluster_assignments,
        n_clusters=n_clusters,
        n_samples=len(features)
    )


# ============================================================================
# Metadata Management
# ============================================================================

def create_run_metadata(
    config: MetadataConfig,
    training_config: Any,
    model_config: Any,
    optimizer_config: Any,
    scheduler_config: Any,
) -> Dict[str, Any]:
    """Create comprehensive metadata for the training run.

    Captures all experimental context needed for reproducibility and
    cross-run comparison.

    Args:
        config: Metadata configuration
        training_config: Training configuration object
        model_config: Model architecture configuration
        optimizer_config: Optimizer configuration
        scheduler_config: Noise scheduler configuration

    Returns:
        Dictionary with complete run metadata

    Saves:
        - Dataset information (name, classes)
        - Model architecture details
        - Training hyperparameters (CFG scale, LR, optimizer, etc.)
        - Noise schedule parameters
        - Random seed for reproducibility
        - Code version/commit (if available)
    """
    metadata = {
        "dataset": {
            "name": config.dataset_name,
            "class_names": config.class_names,
            "num_classes": len(config.class_names) if config.class_names else None,
        },
        "training": {
            "guidance_scale": getattr(training_config, "guidance_scale", None),
            "guidance_p_uncond": getattr(training_config, "guidance_p_uncond", None),
            "epochs": getattr(training_config, "epochs", None),
            "batch_size": getattr(training_config, "batch_size", None),
            "image_size": getattr(training_config, "image_size", None),
            "seed": getattr(training_config, "seed", None),
            "ema_decay": getattr(training_config, "ema_decay", None),
            "ema_use": getattr(training_config, "ema_use", None),
            "mixed_precision": getattr(training_config, "mixed_precision", None),
            "use_min_snr": getattr(training_config, "use_min_snr", None),
            "min_snr_gamma": getattr(training_config, "min_snr_gamma", None),
        },
        "optimizer": {
            "type": getattr(optimizer_config, "type", None),
            "lr": getattr(optimizer_config, "lr", None),
            "weight_decay": getattr(optimizer_config, "wd", None),
            "betas": getattr(optimizer_config, "betas", None),
        },
        "scheduler": {
            "num_train_timesteps": getattr(scheduler_config, "num_train_timesteps", None),
            "beta_schedule": getattr(scheduler_config, "beta_schedule", None),
            "beta_start": getattr(scheduler_config, "beta_start", None),
            "beta_end": getattr(scheduler_config, "beta_end", None),
            "prediction_type": getattr(scheduler_config, "prediction_type", None),
        },
        "model": {
            "model_channels": getattr(model_config, "model_channels", None),
            "channel_mult": getattr(model_config, "channel_mult", None),
            "layers_per_block": getattr(model_config, "layers_per_block", None),
            "class_embed_dim": getattr(model_config, "class_embed_dim", None),
            "dropout": getattr(model_config, "dropout", None),
        }
    }

    return metadata


# ============================================================================
# Enhanced Trajectory Saving (Legacy + New Features)
# ============================================================================

def save_class_embeddings_trajectory(
    model: nn.Module,
    epoch: int,
    output_path: Path,
) -> None:
    """Append current epoch's class-embedding matrix into a single .pt trajectory file.

    Legacy function for backward compatibility. Saves the parametric class
    embedding weights (model.class_embed.emb.weight) at each epoch.

    Args:
        model: The model containing class_embed.emb.weight
        epoch: Current epoch number
        output_path: Path to save the trajectory file

    Saves:
        - Parametric class embedding matrix [num_classes, emb_dim] per epoch
        - Accumulated trajectory over all logged epochs
        - Shape: [num_epochs, num_classes, emb_dim]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Current embedding matrix [num_classes, emb_dim]
    emb_weight = model.class_embed.emb.weight.detach().cpu().clone()  # type: ignore

    if output_path.exists():
        # Load existing trajectory and append
        state = torch.load(output_path, map_location="cpu", weights_only=False)
        epochs = state.get("epochs", [])
        prev_emb = state.get("embeddings", None)

        # Ensure shapes are consistent
        if prev_emb is not None:
            if prev_emb.shape[1:] != emb_weight.shape:
                raise RuntimeError(
                    f"Embedding shape changed from {tuple(prev_emb.shape[1:])} "
                    f"to {tuple(emb_weight.shape)}; cannot append trajectory."
                )
            embeddings = torch.cat([prev_emb, emb_weight.unsqueeze(0)], dim=0)
        else:
            embeddings = emb_weight.unsqueeze(0)
        epochs.append(int(epoch))
    else:
        # First snapshot
        epochs = [int(epoch)]
        embeddings = emb_weight.unsqueeze(0)

    state_out = {
        "epochs": epochs,
        "embeddings": embeddings,              # [E, num_classes, emb_dim]
        "num_classes": embeddings.shape[1],
        "emb_dim": embeddings.shape[2],
    }

    torch.save(state_out, output_path)
    logger.info(
        "Saved class embedding snapshot for epoch %d to %s "
        "(total snapshots: %d)",
        epoch, str(output_path), len(epochs)
    )


def save_probe_features(
    records: List[FeatureRecord],
    output_path: Path,
) -> None:
    """Save probe feature records to disk.

    Saves all feature vectors collected from the probe set, organized by
    sample, class, timestep, layer, and branch (cond/uncond).

    Args:
        records: List of FeatureRecord objects
        output_path: Path to save (will use .npz format for efficiency)

    Saves:
        - Feature vectors: [n_records, feature_dim]
        - Metadata arrays: sample_ids, class_ids, epochs, timesteps, layer_names, branches
        - Allows efficient querying (e.g., "all features from layer X at timestep Y")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        logger.warning("No probe features to save")
        return

    # Convert to structured arrays
    sample_ids = np.array([r.sample_id for r in records])
    class_ids = np.array([r.class_id for r in records])
    epochs = np.array([r.epoch for r in records])
    timesteps = np.array([r.timestep for r in records])
    layer_names = np.array([r.layer_name for r in records])
    branches = np.array([r.branch for r in records])
    features = np.stack([r.feature for r in records])  # [n_records, d]

    # Save as npz for efficiency
    np.savez_compressed(
        output_path,
        features=features,
        sample_ids=sample_ids,
        class_ids=class_ids,
        epochs=epochs,
        timesteps=timesteps,
        layer_names=layer_names,
        branches=branches,
    )

    logger.info(
        f"Saved {len(records)} probe features to {output_path} "
        f"(shape: {features.shape})"
    )


def save_clustering_metrics(
    metrics: ClusteringMetrics,
    output_path: Path,
) -> None:
    """Save clustering metrics to disk.

    Appends current epoch's clustering quality scores to a trajectory file.

    Args:
        metrics: ClusteringMetrics object
        output_path: Path to save (.pt format)

    Saves:
        - Per-epoch clustering scores (silhouette, Calinski-Harabasz, Davies-Bouldin)
        - Cluster centers and assignments
        - Trajectory over all logged epochs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        state = torch.load(output_path, map_location="cpu", weights_only=False)
        epochs = state.get("epochs", [])
        scores = state.get("scores", [])
    else:
        epochs = []
        scores = []

    epochs.append(metrics.epoch)
    scores.append(metrics.to_dict())

    state_out = {
        "epochs": epochs,
        "scores": scores,
        "cluster_centers": metrics.cluster_centers,
        "cluster_assignments": metrics.cluster_assignments,
    }

    torch.save(state_out, output_path)
    logger.info(
        f"Saved clustering metrics for epoch {metrics.epoch} to {output_path}"
    )


def save_metadata(
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save run metadata to JSON file.

    Args:
        metadata: Dictionary with run metadata
        output_path: Path to save (.json format)

    Saves:
        - Complete experimental context (dataset, model, training config)
        - Enables reproducibility and cross-run comparison
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved run metadata to {output_path}")


# ============================================================================
# Main Orchestrator
# ============================================================================

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
    """Main orchestrator for enhanced embedding logging.

    Coordinates all logging components:
      1. Class embedding trajectories (always)
      2. Feature probe bank (if enabled and epoch matches)
      3. Clustering metrics (if enabled and epoch matches)
      4. Run metadata (once, on first call)

    Args:
        model: The ccDDPM model
        dataloader: Training dataloader
        noise_scheduler: DDPM scheduler
        device: Compute device
        epoch: Current epoch
        config: Master logging configuration
        probe_set: Existing probe set (or None to create new)
        training_config: Training configuration (for metadata)
        model_config: Model configuration (for metadata)
        optimizer_config: Optimizer configuration (for metadata)
        scheduler_config: Scheduler configuration (for metadata)

    Returns:
        probe_set (for reuse across epochs)

    Saves:
        - Class embeddings: {output_dir}/class_embeddings_trajectory.pt
        - Probe features: {output_dir}/probe_features_epoch_{epoch:04d}.npz
        - Clustering: {output_dir}/clustering_metrics.pt
        - Metadata: {output_dir}/run_metadata.json
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Always save class embeddings (legacy)
    if config.save_class_embeddings:
        emb_path = output_dir / "class_embeddings_trajectory.pt"
        save_class_embeddings_trajectory(model, epoch, emb_path)

    # 2. Save metadata (once, on first epoch)
    if config.metadata.save_metadata and epoch == 1:
        if all(c is not None for c in [training_config, model_config, optimizer_config, scheduler_config]):
            metadata = create_run_metadata(
                config.metadata,
                training_config,
                model_config,
                optimizer_config,
                scheduler_config
            )
            meta_path = output_dir / "run_metadata.json"
            save_metadata(metadata, meta_path)

    # 3. Check if we should log detailed diagnostics this epoch
    should_log_details = (epoch % config.log_every_n_epochs) == 0

    if not should_log_details:
        return probe_set

    # 4. Create or reuse probe set
    if config.probe.enabled:
        if probe_set is None:
            num_classes = training_config.num_classes if training_config else 10
            probe_set = create_balanced_probe_set(
                dataloader,
                config.probe.samples_per_class,
                num_classes,
                seed=training_config.seed if training_config else 42
            )
            # Save probe set indices for reproducibility
            probe_path = output_dir / "probe_set.json"
            with open(probe_path, 'w') as f:
                json.dump(probe_set, f, indent=2)
            logger.info(f"Saved probe set to {probe_path}")

        # Collect features
        features = collect_probe_features(
            model, dataloader, probe_set, noise_scheduler,
            device, epoch, config.probe
        )

        # Save features
        feat_path = output_dir / f"probe_features_epoch_{epoch:04d}.npz"
        save_probe_features(features, feat_path)

        # 5. Compute clustering metrics (if enabled)
        if config.clustering.enabled and len(features) > 0:
            # Extract features and labels for clustering
            # Use only conditional branch, one timestep, one layer for simplicity
            if not config.probe.timesteps:
                logger.warning("No timesteps configured for probe, skipping clustering")
                return probe_set

            target_timestep = config.probe.timesteps[len(config.probe.timesteps) // 2]  # Middle timestep
            target_layer = config.probe.layer_names[0] if config.probe.layer_names else "class_embed"

            filtered = [
                f for f in features
                if f.branch == "cond" and f.timestep == target_timestep and f.layer_name == target_layer
            ]

            if len(filtered) > 10:  # Need minimum samples for clustering
                feat_matrix = np.stack([f.feature for f in filtered])
                labels = np.array([f.class_id for f in filtered])

                metrics = compute_clustering_metrics(
                    feat_matrix, labels, epoch, config.clustering
                )

                # Save clustering metrics
                cluster_path = output_dir / "clustering_metrics.pt"
                save_clustering_metrics(metrics, cluster_path)

                logger.info(
                    f"Epoch {epoch} clustering: "
                    f"silhouette={metrics.silhouette:.4f}, "
                    f"CH={metrics.calinski_harabasz:.2f}, "
                    f"DB={metrics.davies_bouldin:.4f}"
                )

    return probe_set
