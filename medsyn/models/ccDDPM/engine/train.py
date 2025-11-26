# medsyn/models/ccDDPM/engine/train.py
# Purpose: Training loop for class-conditioned DDPM with Diffusers' DDPMScheduler.
# Features: mixed precision, EMA, classifier-free guidance (label drop), checkpointing, CSV logs, DDP support.
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import os
import csv
import time
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from medsyn.models.ccDDPM.config import ProjectCfg, load_cfg
from medsyn.models.ccDDPM.dataloaders.json import build_json_loader
from medsyn.models.ccDDPM.dataloaders.npz import build_npz_loader
from medsyn.models.ccDDPM.model import CCDDPM, CCDDPMInit
from medsyn.models.ccDDPM.loss import DDPMNoiseMSE, estimate_elbo_terms
from medsyn.models.ccDDPM.metrics import (
    compute_psnr, compute_ssim,
    compute_per_class_metrics,
    PerClassMetricsAccumulator,
    compute_class_weight_correlation,
    TorchmetricsFIDComputer,
    setup_fid_weights_cache,
)
from medsyn.models.ccDDPM.vis.training_visualizations import (
    visualize_denoising_process, 
    visualize_multistep_reconstruction, 
    visualize_noising_process
)
from medsyn.models.ccDDPM.engine.utils.early_stopping import EarlyStoppingTracker
from medsyn.models.ccDDPM.engine.factories.optimizer import create_optimizer
from medsyn.models.ccDDPM.engine.factories.lr_scheduler import create_lr_scheduler
from medsyn.models.ccDDPM.engine.utils.ema import EMA
from medsyn.models.ccDDPM.engine.eval import evaluate_split
from medsyn.models.ccDDPM.engine.utils.time_formatting import format_time
from medsyn.models.ccDDPM.engine.utils.math import (
    compute_batch_metrics,
    compute_elbo_diagnostics,
    compute_training_diagnostics,
    conditioning_sanity_check,
    compute_class_weights_from_counts,
    full_chain_reconstruction_metrics,
    full_chain_reconstruction_psnr
)
from medsyn.models.ccDDPM.engine.utils.fid import compute_fid_on_split
from medsyn.models.ccDDPM.engine.logging.logging import (
    log_checkpoint_saved,
    log_early_stopping_triggered,
    log_enhanced_epoch_summary,
    log_epoch_start,
    log_epoch_summary,
    log_model_summary,
    log_per_class_summary,
    log_phase_complete,
    log_phase_start,
    log_training_complete,
    log_training_config,
    log_training_progress,
    log_validation_progress,
    should_log_step
)
from medsyn.models.ccDDPM.engine.logging.training_logging import (
    CSVTrainingLogger, EpochAverager, TRAINING_FIELDS, DIAGNOSTIC_FIELDS, NUM_CLASSES,
    DEFAULT_FID_CONFIG, save_class_embeddings_trajectory
)
from medsyn.models.ccDDPM.engine.utils.ddp_utils import (
    ddp_is_enabled, ddp_init, is_main_process,
    barrier, cleanup, all_reduce_mean, broadcast_bool, get_state_dict_for_save, sync_metrics_dict
)
from medsyn.models.ccDDPM.vis.visualize_training_evolution import generate_training_visualizations

logger = logging.getLogger("medsyn.ccddpm.train")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

# Supercomputer mode detection - disable tqdm, enable structured logging
IS_SUPERCOMPUTER = os.getenv("IS_SUPERCOMPUTER", "0") == "1"

# Augmentation imports (optional, will only be used if augmentation is enabled in config)
try:
    from medsyn.models.ccDDPM.augmentation import (
        create_augmentation_pipeline,
        AugmentationStatistics
    )
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logger.warning("Augmentation module not available. Install albumentations to enable augmentation.")

def train(yaml_path: str, split: str = "train") -> None:
    """
    Train ccDDPM using config at yaml_path. Saves checkpoints and CSV log.
    Supports both single-GPU and multi-GPU (DDP) training.

    For multi-GPU training, launch with:
        torchrun --standalone --nnodes=1 --nproc_per_node=N -m medsyn.cli.train_ccDDPM config.yaml
    """
    cfg: ProjectCfg = load_cfg(yaml_path, split=split)
    tcfg = cfg.ccddpm.train
    scfg = cfg.ccddpm.sched
    optimizer_cfg = cfg.ccddpm.optimizer  # Renamed from ocfg

    # ========================================================================
    # FID WEIGHTS SETUP (Before any FID computation)
    # ========================================================================
    # Configure FID weights cache for offline HPC usage
    # This must happen before any torchmetrics FID initialization
    if hasattr(cfg, 'fid') and cfg.fid and cfg.fid.get("enabled"):
        weights_path = cfg.fid.get('weights_path', None)
        if weights_path:
            setup_fid_weights_cache(weights_path)
            logger.info(f"FID weights cache configured: {weights_path}")
        else:
            logger.warning(
                "FID enabled but no weights_path specified. "
                "Will attempt automatic download (requires internet)."
            )

    # ========================================================================
    # DDP/NCCL DEBUGGING: Set environment variables for better error reporting
    # ========================================================================
    # Enable detailed distributed debugging (helps identify collective operation failures)
    os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
    # Enable async error handling in NCCL (reports errors immediately rather than timeout)
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

    # ========================================================================
    # PROTECTION: Detect misconfiguration
    # ========================================================================
    world_size_env = int(os.getenv("WORLD_SIZE", "1"))
    if world_size_env > 1 and not cfg.ccddpm.dist.enabled:
        raise RuntimeError(
            f"Misconfiguration detected: Script launched with torchrun (WORLD_SIZE={world_size_env}) "
            f"but dist.enabled=false in config. Either:\n"
            f"  1. Set 'ccddpm.dist.enabled: true' in your config, OR\n"
            f"  2. Launch with single process (no torchrun)"
        )

    # ========================================================================
    # BIFURCATION: Decide DDP vs Legacy path (once, early)
    # ========================================================================
    use_ddp = ddp_is_enabled(cfg)

    if use_ddp:
        # ====================================================================
        # DDP PATH: Initialize distributed training
        # ====================================================================
        ddp_info = ddp_init(cfg)
        device = ddp_info["device"]
        rank = ddp_info["rank"]
        world_size = ddp_info["world_size"]
        local_rank = ddp_info["local_rank"]
        logger.info(f"[Rank {rank}/{world_size}] DDP training on device {device}")
    else:
        # ====================================================================
        # LEGACY PATH: Single-GPU training
        # ====================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        local_rank = 0
        logger.info(f"Single-GPU training on device: {device}")

    # Augmentation setup
    augmentation_pipeline = None
    augmentation_stats = None
    if cfg.ccddpm.augmentation and AUGMENTATION_AVAILABLE:
        aug_cfg = cfg.ccddpm.augmentation
        if aug_cfg.enabled:
            logger.info("Creating augmentation pipeline...")
            augmentation_pipeline = create_augmentation_pipeline(aug_cfg, use_replay=True)
            logger.info(f"  Enabled: {aug_cfg.enabled}")
            logger.info(f"  Probability: {aug_cfg.probability}")
            logger.info(f"  Transforms: {len(aug_cfg.transforms)}")

            # Initialize statistics tracker if enabled (only on rank-0 in DDP)
            # Note: In DDP, each GPU sees different batches, so we only track on rank-0
            if aug_cfg.statistics.enabled and is_main_process():
                stats_output_path = Path(aug_cfg.statistics.output_path)
                # If relative path, resolve relative to output_dir
                if not stats_output_path.is_absolute():
                    stats_output_path = (Path(tcfg.output_dir) / stats_output_path.name).resolve()

                transform_names = augmentation_pipeline.get_transform_names()
                augmentation_stats = AugmentationStatistics(stats_output_path, transform_names)
                logger.info(f"  Statistics tracking enabled: {stats_output_path} (rank-0 only)")
        else:
            logger.info("Augmentation is disabled in config")
    elif cfg.ccddpm.augmentation and not AUGMENTATION_AVAILABLE:
        logger.warning("Augmentation requested in config but module not available. Proceeding without augmentation.")

    # ========================================================================
    # DATALOADERS: Create datasets and loaders (with DDP samplers if enabled)
    # ========================================================================
    # NOTE: batch_size is interpreted as per-GPU batch size
    dl_cfg = cfg.ccddpm.dataloader
    train_sampler = None
    val_sampler = None
    test_sampler = None
    # Optional per-class loss weights (computed from training set if enabled)
    class_weights = None

    if dl_cfg.type.lower() == "npz":
        if is_main_process():
            logger.info("Using NPZ dataloader from: %s", dl_cfg.npz_path)
        if dl_cfg.npz_path is None:
            raise ValueError("NPZ dataloader selected but npz_path is not specified in config")

        # Build datasets first to create samplers
        from medsyn.models.ccDDPM.dataloaders.npz import NPZDataset
        train_dataset = NPZDataset(
            dl_cfg.npz_path, "train", tcfg.image_size, normalize=True,
            augmentation_pipeline=augmentation_pipeline
        )
        val_dataset = NPZDataset(
            dl_cfg.npz_path, "val", tcfg.image_size, normalize=True,
            augmentation_pipeline=None
        )
        test_dataset = NPZDataset(
            dl_cfg.npz_path, "test", tcfg.image_size, normalize=True,
            augmentation_pipeline=None
        )

        # Optional: compute per-class loss weights from training-set imbalance.
        # Enabled by setting cfg.ccddpm.train.per_class_loss_weighting = True.
        if getattr(tcfg, "per_class_loss_weighting", False):
            import numpy as np  

            if not hasattr(train_dataset, "labels"):
                raise AttributeError(
                    "per_class_loss_weighting=True but train_dataset has no 'labels' attribute."
                )

            labels_np = np.asarray(train_dataset.labels, dtype=np.int64)
            if labels_np.ndim != 1:
                raise ValueError(
                    f"Expected 1D labels array in train_dataset, got shape {labels_np.shape}."
                )

            # Ignore any unconditional label -1 when computing frequencies
            if labels_np.min() < 0:
                labels_for_weights = labels_np[labels_np >= 0]
            else:
                labels_for_weights = labels_np

            if labels_for_weights.size == 0:
                raise ValueError("No valid labels available to compute class weights.")

            num_classes = int(tcfg.num_classes)
            counts = np.bincount(labels_for_weights, minlength=num_classes)
            if (counts == 0).any() and is_main_process():
                logger.warning(
                    "Some classes have zero samples in the training set when "
                    "computing class weights: %s",
                    np.where(counts == 0)[0].tolist(),
                )

            # Compute class weights with optional temperature scaling
            temperature = getattr(tcfg, "per_class_weight_temperature", 1.0)
            weights_np = compute_class_weights_from_counts(
                counts, temperature=temperature, normalize=True
            )

            # Stored on CPU; DDPMNoiseMSE will move to the correct device internally
            class_weights = torch.from_numpy(weights_np.astype("float32"))

            if is_main_process():
                logger.info(
                    "Class weights computed from training set (inverse frequency, temperature=%.2f):",
                    temperature
                )
                for c in range(num_classes):
                    logger.info(f"  Class {c}: count={counts[c]}, weight={class_weights[c]:.4f}")

        if use_ddp:
            # DDP: Create DistributedSamplers (shuffle via sampler, not DataLoader)
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            test_sampler = DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if is_main_process():
                logger.info(f"DDP: DistributedSampler created for {len(train_dataset)} train, "
                           f"{len(val_dataset)} val, {len(test_dataset)} test samples across {world_size} GPUs")
        # else: Legacy path uses shuffle=True in DataLoader, no sampler

        # Build loaders (DDP uses sampler with shuffle=False, Legacy uses shuffle=True)
        train_loader = build_npz_loader(
            dl_cfg.npz_path, "train", tcfg.image_size, tcfg.batch_size, tcfg.num_workers,
            normalize=True, augmentation_pipeline=augmentation_pipeline, sampler=train_sampler
        )
        val_loader = build_npz_loader(
            dl_cfg.npz_path, "val", tcfg.image_size, tcfg.batch_size, tcfg.num_workers,
            normalize=True, augmentation_pipeline=None, sampler=val_sampler
        )
        test_loader = build_npz_loader(
            dl_cfg.npz_path, "test", tcfg.image_size, tcfg.batch_size, tcfg.num_workers,
            normalize=True, augmentation_pipeline=None, sampler=test_sampler
        )

    else:  # default to JSON
        if is_main_process():
            logger.info("Using JSON dataloader from: %s", cfg.data_index_json)

        # Build datasets first to create samplers
        from medsyn.models.ccDDPM.dataloaders.json import PathMNISTIndexDataset
        train_dataset = PathMNISTIndexDataset(
            cfg.data_index_json, "train", tcfg.image_size, normalize=True,
            augmentation_pipeline=augmentation_pipeline
        )
        val_dataset = PathMNISTIndexDataset(
            cfg.data_index_json, "val", tcfg.image_size, normalize=True,
            augmentation_pipeline=None
        )
        test_dataset = PathMNISTIndexDataset(
            cfg.data_index_json, "test", tcfg.image_size, normalize=True,
            augmentation_pipeline=None
        )

        if use_ddp:
            # DDP: Create DistributedSamplers (shuffle via sampler, not DataLoader)
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            test_sampler = DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if is_main_process():
                logger.info(f"DDP: DistributedSampler created for {len(train_dataset)} train, "
                           f"{len(val_dataset)} val, {len(test_dataset)} test samples across {world_size} GPUs")
        # else: Legacy path uses shuffle=True in DataLoader, no sampler

        # Build loaders (DDP uses sampler with shuffle=False, Legacy uses shuffle=True)
        train_loader = build_json_loader(
            cfg.data_index_json, "train", tcfg.image_size, tcfg.batch_size, tcfg.num_workers,
            normalize=True, augmentation_pipeline=augmentation_pipeline, sampler=train_sampler
        )
        val_loader = build_json_loader(
            cfg.data_index_json, "val", tcfg.image_size, tcfg.batch_size, tcfg.num_workers,
            normalize=True, augmentation_pipeline=None, sampler=val_sampler
        )
        test_loader = build_json_loader(
            cfg.data_index_json, "test", tcfg.image_size, tcfg.batch_size, tcfg.num_workers,
            normalize=True, augmentation_pipeline=None, sampler=test_sampler
        )

    # ========================================================================
    # DATASET SIZE LOGGING
    # ========================================================================
    # Log comprehensive dataset information (only rank-0 to avoid clutter)
    if is_main_process():
        logger.info("=" * 80)
        logger.info("DATASET SIZES (Pre-Augmentation)")
        logger.info("=" * 80)
        logger.info(f"Train: {len(train_dataset):,} samples")
        if augmentation_pipeline is not None:
            aug_cfg = cfg.ccddpm.augmentation
            logger.info(f"  Augmentation: ENABLED (probability={aug_cfg.probability})")
            logger.info(f"  Transforms: {len(aug_cfg.transforms)} configured")
            logger.info(f"  Note: Effective sample diversity increases due to augmentation")
        else:
            logger.info(f"  Augmentation: DISABLED")
        logger.info(f"Val:   {len(val_dataset):,} samples (no augmentation)")
        logger.info(f"Test:  {len(test_dataset):,} samples (no augmentation)")
        logger.info("=" * 80)

    # ========================================================================
    # MODEL, SCHEDULER, OPTIMIZER, EMA
    # ========================================================================
    # Build model with full UNet configuration
    ucfg = cfg.ccddpm.unet  # UNet architecture config from YAML
    mcfg = CCDDPMInit(
        # Input/Output (from train config)
        in_channels=tcfg.in_channels,
        class_embed_dim=tcfg.class_embed_dim,
        num_classes=tcfg.num_classes,
        # Core UNet architecture (from unet config)
        model_channels=ucfg.model_channels,
        channel_mult=ucfg.channel_mult,
        layers_per_block=ucfg.layers_per_block,
        down_block_types=ucfg.down_block_types,
        up_block_types=ucfg.up_block_types,
        # Attention
        add_attention=ucfg.add_attention,
        attention_head_dim=ucfg.attention_head_dim,
        # Normalization
        norm_num_groups=ucfg.norm_num_groups,
        attn_norm_num_groups=ucfg.attn_norm_num_groups,
        norm_eps=ucfg.norm_eps,
        # Dropout
        dropout=ucfg.dropout,
        # Time embedding
        time_embedding_type=ucfg.time_embedding_type,
        freq_shift=ucfg.freq_shift,
        flip_sin_to_cos=ucfg.flip_sin_to_cos,
        resnet_time_scale_shift=ucfg.resnet_time_scale_shift,
        # Sampling
        center_input_sample=ucfg.center_input_sample,
        mid_block_scale_factor=ucfg.mid_block_scale_factor,
        # Down/Up sampling
        downsample_padding=ucfg.downsample_padding,
        downsample_type=ucfg.downsample_type,
        upsample_type=ucfg.upsample_type,
        # Activation
        act_fn=ucfg.act_fn,
        # Class embedding
        class_embed_type=ucfg.class_embed_type,
        num_class_embeds=ucfg.num_class_embeds,
        num_train_timesteps=ucfg.num_train_timesteps,
    )
    model = CCDDPM(mcfg).to(device)

    # Log model architecture summary
    if is_main_process():
        logger.info("=" * 80)
        logger.info("MODEL ARCHITECTURE")
        logger.info("=" * 80)
        logger.info(f"UNet block_out_channels: {mcfg.get_block_out_channels()}")
        logger.info(f"UNet down_block_types: {mcfg.down_block_types}")
        logger.info(f"UNet up_block_types: {mcfg.up_block_types}")
        logger.info(f"UNet layers_per_block: {mcfg.layers_per_block}")
        logger.info(f"Total parameters: {model.get_num_params():,}")
        logger.info(f"Trainable parameters: {model.get_num_trainable_params():,}")
        logger.info("=" * 80)

    # Build scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
        thresholding=False,
    )

    # CRITICAL: Keep reference to base_model for EMA
    base_model = model

    if use_ddp:
        # DDP: Wrap model in DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.ccddpm.dist.find_unused_parameters,
            broadcast_buffers=cfg.ccddpm.dist.broadcast_buffers,
            static_graph=True  # DDPM graph is static, enables optimizations
        )
        if is_main_process():
            logger.info("DDP: Model wrapped in DistributedDataParallel")
    # else: Legacy path uses model directly, no wrapper

    # Build optimizer using the factory function (supports multiple optimizer types)
    if is_main_process():
        logger.info(f"Optimizer config: type={optimizer_cfg.type}, lr={optimizer_cfg.lr:.2e}, "
                   f"weight_decay={optimizer_cfg.wd:.2e}")

    opt = create_optimizer(model.parameters(), optimizer_cfg)
    scaler = GradScaler(device='cuda', enabled=tcfg.mixed_precision)

    # Build EMA (CRITICAL: always tracks base_model, not DDP wrapper)
    ema = EMA(base_model, decay=tcfg.ema_decay) if tcfg.ema_use else None

    loss_fn = DDPMNoiseMSE(
        num_classes=tcfg.num_classes,
        use_min_snr=tcfg.use_min_snr,
        min_snr_gamma=tcfg.min_snr_gamma
    )

    out_dir = Path(tcfg.output_dir).resolve()

    # Only rank-0 creates directories and loggers
    csv_logger = None
    diag_logger = None
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        # Initialize CSV logger for train/val/test metrics
        # Note: loss_raw_c* and loss_weighted_c* are already in TRAINING_FIELDS
        csv_logger = CSVTrainingLogger(
            str(out_dir / "training_metrics.csv"),
            fieldnames=list(TRAINING_FIELDS)
        )

        # Initialize separate CSV logger for diagnostic metrics
        diag_logger = CSVTrainingLogger(
            str(out_dir / "diagnostics_metrics.csv"),
            fieldnames=list(DIAGNOSTIC_FIELDS)
        )

        # Keep old simple CSV for backward compatibility
        csv_path = out_dir / "train_log.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["epoch","step","loss","lr","time_s"])

    # Synchronize all processes after directory creation
    if use_ddp:
        barrier()

    # ========================================================================
    # EARLY STOPPING TRACKER
    # ========================================================================
    # Uses composite validation score with EMA smoothing
    early_stop_tracker = EarlyStoppingTracker.from_config(tcfg)
    if is_main_process():
        logger.info("Early stopping configuration:")
        logger.info(f"  Patience: {early_stop_tracker.patience}")
        logger.info(f"  Metric: {early_stop_tracker.metric}")
        logger.info(f"  PSNR weight: {early_stop_tracker.psnr_weight}")
        logger.info(f"  SSIM weight: {early_stop_tracker.ssim_weight}")
        logger.info(f"  EMA alpha: {early_stop_tracker.ema_alpha}")
        logger.info(f"  Use EMA for stopping: {early_stop_tracker.use_ema_for_stopping}")
        logger.info(f"  Min delta: {early_stop_tracker.min_delta}")

    # ========================================================================
    # LEARNING RATE SCHEDULER
    # ========================================================================
    steps_per_epoch = len(train_loader)
    lr_scheduler = None
    lr_step_per_batch = False

    if hasattr(tcfg, 'lr_scheduler') and tcfg.lr_scheduler is not None:
        lr_scheduler, lr_step_per_batch = create_lr_scheduler(
            optimizer=opt,
            scheduler_cfg=tcfg.lr_scheduler,
            steps_per_epoch=steps_per_epoch,
            total_epochs=tcfg.epochs,
            base_lr=optimizer_cfg.lr,  # Use lr from optimizer config
        )
    else:
        if is_main_process():
            logger.info("No LR scheduler configured, using constant learning rate")

    global_step = 0
    model.train()

    # ========================================================================
    # COMPREHENSIVE STARTUP LOGGING (for supercomputer batch jobs)
    # ========================================================================
    if is_main_process():
        log_training_config(
            tcfg=tcfg,
            optimizer_cfg=optimizer_cfg,
            scfg=scfg,
            dist_cfg=cfg.ccddpm.dist,
            augmentation_cfg=cfg.ccddpm.augmentation,
            num_train_samples=len(train_dataset),
            num_val_samples=len(val_dataset),
            num_test_samples=len(test_dataset),
            steps_per_epoch=steps_per_epoch,
        )
        log_model_summary(base_model, ucfg)

    # Track total training time for final summary
    training_start_time = time.time()
    best_val_metrics_snapshot = {}

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    for epoch in range(1, tcfg.epochs + 1):
        # DDP: Set epoch for DistributedSampler to ensure different shuffling per epoch
        # Legacy: No sampler, shuffling handled by DataLoader
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        running_loss = 0.0
        train_avg = EpochAverager()

        # Per-class metrics accumulator for training (PSNR/SSIM per class)
        num_classes = getattr(tcfg, 'num_classes', NUM_CLASSES)
        train_per_class_acc = PerClassMetricsAccumulator(num_classes=num_classes)

        # Reset per-class loss statistics at the start of each epoch
        loss_fn.reset()

        # Collect augmentation statistics for this epoch
        epoch_augmentation_transforms = []

        # Progress tracking
        total_train_steps = len(train_loader)
        use_progress_bar = tcfg.use_tqdm and is_main_process() and not IS_SUPERCOMPUTER

        # Epoch start logging (enhanced for supercomputer)
        if is_main_process():
            log_epoch_start(epoch, tcfg.epochs, opt.param_groups[0]['lr'])
            if IS_SUPERCOMPUTER:
                log_phase_start("training", total_train_steps)

        # Progress bar for the training epoch (only on rank-0 if using tqdm AND not supercomputer)
        train_iter = enumerate(train_loader, 1)
        if use_progress_bar:
            pbar = tqdm(train_iter, total=total_train_steps,
                        desc=f"Epoch {epoch}/{tcfg.epochs}",
                        unit="batch", leave=True)
        else:
            pbar = train_iter

        for step, batch in pbar:
            x0 = batch["pixel_values"].to(device)  # [-1,1]
            labels = batch["labels"].to(device)

            # Collect augmentation statistics if tracking is enabled
            if augmentation_stats is not None and "applied_transforms" in batch:
                epoch_augmentation_transforms.extend(batch["applied_transforms"])

            # sample t and noise
            # NOTE: aquí ocurre la magia jeje... 
            bsz = x0.size(0)
            t = torch.randint(0, scfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = noise_scheduler.add_noise(x0, noise, t) # type: ignore

            # classifier-free: drop labels with prob p (set to sentinel -1 for unconditional)
            class_labels = labels.clone()
            if tcfg.guidance_p_uncond > 0:
                drop = torch.rand(bsz, device=device) < tcfg.guidance_p_uncond
                class_labels[drop] = -1  # sentinel for uncond
            # forward
            # prefer bf16 on Ampere+ (more stable than fp16); falls back if unavailable
            autocast_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with autocast(device_type='cuda', enabled=tcfg.mixed_precision, dtype=autocast_dtype):
                pred = model(x_t, t, class_labels)
                loss = loss_fn(
                    pred, noise, labels,
                    timesteps=t,
                    alphas_cumprod=noise_scheduler.alphas_cumprod,
                    class_weights=class_weights,
                )
            # Guard: check for non-finite loss
            skip_step = False
            grad_norm_val = 0.0  # Initialize gradient norm
            if not torch.isfinite(loss):
                if tcfg.skip_nonfinite_grads:
                    logger.warning(
                        f"Non-finite loss at global_step={global_step}, epoch={epoch}, "
                        f"loss={loss.item()} - skipping batch"
                    )
                    skip_step = True
                else:
                    logger.error(
                        f"Non-finite loss at global_step={global_step}, epoch={epoch}, "
                        f"loss={loss.item()}"
                    )
                    raise FloatingPointError("Non-finite loss; aborting training.")

            # Skip backward pass and optimizer step if loss is non-finite
            if not skip_step:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()

                # Compute gradient norm and check for NaN/Inf BEFORE optimizer step
                if tcfg.grad_clip_norm:
                    scaler.unscale_(opt)
                    # Clip gradients; error_if_nonfinite=False prevents crashes on overflow
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), tcfg.grad_clip_norm, error_if_nonfinite=False
                    )
                    grad_norm_val = float(grad_norm)
                    # Check if gradients are non-finite
                    if not math.isfinite(grad_norm_val):
                        if tcfg.skip_nonfinite_grads:
                            logger.warning(
                                f"Non-finite gradients at global_step={global_step}, epoch={epoch}, "
                                f"grad_norm={grad_norm} - skipping batch"
                            )
                            skip_step = True
                        else:
                            logger.error(
                                f"Non-finite gradients at global_step={global_step}, epoch={epoch}, "
                                f"grad_norm={grad_norm}"
                            )
                            raise FloatingPointError("Non-finite gradients; aborting training.")
                else:
                    # Compute gradient norm for logging
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm_val = total_norm ** 0.5
                    if not math.isfinite(grad_norm_val):
                        if tcfg.skip_nonfinite_grads:
                            logger.warning(
                                f"Non-finite gradients at global_step={global_step}, epoch={epoch}, "
                                f"grad_norm={grad_norm_val} - skipping batch"
                            )
                            skip_step = True
                        else:
                            logger.error(
                                f"Non-finite gradients at global_step={global_step}, epoch={epoch}, "
                                f"grad_norm={grad_norm_val}"
                            )
                            raise FloatingPointError("Non-finite gradients; aborting training.")

            # Only update model if step is not skipped
            if not skip_step:
                scaler.step(opt)
                scaler.update()
                if ema:
                    # CRITICAL: Update EMA with base_model (unwrapped), not DDP wrapper
                    ema.update(base_model)

                # Step LR scheduler per batch (if configured)
                if lr_scheduler is not None and lr_step_per_batch:
                    lr_scheduler.step()

                global_step += 1
            else:
                # Reset gradients and scaler state on skipped step
                opt.zero_grad(set_to_none=True)
                # Update scaler to reset internal state (prevents scale from growing indefinitely)
                scaler.update()

            # Compute x0 reconstruction for metrics (predict x0 from predicted noise)
            with torch.no_grad():
                # Use scheduler to predict x0 from noise
                sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t].sqrt()
                sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t]).sqrt()

                # Add epsilon to prevent division by zero
                sqrt_alpha_prod = torch.clamp(sqrt_alpha_prod, min=1e-6)
                x0_pred = (x_t - sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1) * pred) / sqrt_alpha_prod.view(-1, 1, 1, 1)

                # Clamp to prevent extreme values
                x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

                # Compute batch metrics
                batch_metrics = compute_batch_metrics(pred, noise, x0_pred, x0, loss)
                batch_metrics["grad_norm"] = grad_norm_val if not skip_step else 0.0
                batch_metrics["ema_enabled"] = 1.0 if ema else 0.0
                batch_metrics["skipped_step"] = 1.0 if skip_step else 0.0

                # Update epoch averager
                train_avg.update(batch_metrics, batch_size=bsz)

                # Accumulate per-class PSNR/SSIM metrics for training
                # Rescale x0 and x0_pred from [-1, 1] to [0, 1] for metrics computation
                x0_01 = (x0 + 1.0) / 2.0
                x0_pred_01 = (x0_pred.clamp(-1, 1) + 1.0) / 2.0
                train_per_class_acc.update(
                    x_hat=x0_pred_01,
                    x=x0_01,
                    labels=labels,
                    max_val=1.0,
                )

            # Update running loss
            # Accumulate only finite losses
            li = float(loss.detach().cpu())
            if math.isfinite(li):
                running_loss += li

            # Update progress bar with current loss (only if tqdm enabled AND on main process)
            if use_progress_bar:
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                "avg_loss": f"{running_loss/step:.4f}",
                                "psnr": f"{batch_metrics['psnr']:.2f}dB"})

            # Supercomputer mode: periodic logging (10 times per epoch)
            if IS_SUPERCOMPUTER and should_log_step(step, total_train_steps, log_frequency=10):
                # Calculate time metrics
                elapsed = time.time() - epoch_start_time
                time_per_step = elapsed / step
                eta = time_per_step * (total_train_steps - step)

                # Prepare metrics for logging
                log_metrics = {
                    "loss": loss.item(),
                    "avg_loss": running_loss / step,
                    "psnr": batch_metrics['psnr'],
                    "ssim": batch_metrics['ssim'],
                    "grad_norm": grad_norm_val
                }

                # Log progress
                log_training_progress(
                    rank=rank,
                    epoch=epoch,
                    step=step,
                    total_steps=total_train_steps,
                    metrics=log_metrics,
                    elapsed=elapsed,
                    eta=eta,
                    lr=opt.param_groups[0]['lr']
                )
            
            if global_step % tcfg.log_every == 0 and is_main_process():
                csv_path = out_dir / "train_log.csv"
                with open(csv_path, "a", newline="") as fh:
                    csv.writer(fh).writerow([epoch, global_step, float(loss.item()), opt.param_groups[0]["lr"], round(time.time()-epoch_start_time, 2)])

        # Close progress bar if used
        if use_progress_bar:
            pbar.close()

        # Get per-class losses from loss_fn (local to this GPU)
        per_class_losses_weighted = loss_fn.per_class(weighted=True)
        per_class_losses_raw = loss_fn.per_class(weighted=False)
        train_metrics = train_avg.means()

        # Add per-class losses (use same naming as val/test: loss_raw_c* and loss_weighted_c*)
        for c, loss_c in per_class_losses_weighted.items():
            train_metrics[f"loss_weighted_c{c}"] = loss_c
        for c, loss_c in per_class_losses_raw.items():
            train_metrics[f"loss_raw_c{c}"] = loss_c

        # Add per-class PSNR/SSIM metrics from the accumulator
        train_per_class_metrics = train_per_class_acc.compute()
        for key, value in train_per_class_metrics.items():
            # Skip global psnr/ssim since they're already in train_metrics
            if key in ("psnr", "ssim"):
                continue
            train_metrics[key] = value

        # CRITICAL: Synchronize training metrics across all GPUs
        train_metrics = sync_metrics_dict(train_metrics, device, use_ddp)

        # Calculate training time
        train_time = time.time() - epoch_start_time

        # Log training metrics (only rank-0, but now with globally averaged values)
        curr_lr = opt.param_groups[0]["lr"]
        if csv_logger is not None:
            csv_logger.log_epoch(epoch=epoch, split="train", lr=curr_lr, metrics=train_metrics)

        # Log training summary (only rank-0)
        if is_main_process():
            avg_loss = running_loss / len(train_loader)
            if IS_SUPERCOMPUTER:
                # Supercomputer mode: structured phase logging
                log_phase_complete("training", train_metrics, train_time)
            else:
                # Interactive mode: use print for cleaner output
                print("\n" + "="*80)
                print(f"Epoch {epoch}/{tcfg.epochs} Training Summary:")
                print("="*80)
                print(f"  Average Loss: {avg_loss:.4f}")
                print(f"  PSNR: {train_metrics.get('psnr', 0.0):.2f} dB")
                print(f"  SSIM: {train_metrics.get('ssim', 0.0):.4f}")
                print(f"  Noise MSE: {train_metrics.get('noise_mse', 0.0):.4f}")
                print(f"  Learning Rate: {curr_lr:.6f}")
                print(f"  Time: {format_time(train_time)}")

        # ---- Validation ----
        if IS_SUPERCOMPUTER and is_main_process():
            log_phase_start("validation", len(val_loader))

        val_start_time = time.time()
        val_metrics = evaluate_split(
            model=model,
            dataloader=val_loader,
            split_name="val",
            noise_scheduler=noise_scheduler,
            device=device,
            scfg=scfg,
            tcfg=tcfg,
            use_ddp=use_ddp,
            rank=rank
        )
        val_time = time.time() - val_start_time

        # Log validation phase completion
        if IS_SUPERCOMPUTER and is_main_process():
            log_phase_complete("validation", val_metrics, val_time)

        # Extract loss for reference logging
        val_loss = val_metrics.get("loss", float("inf"))

        # ---- Update Early Stopping Tracker ----
        # This computes composite score and updates EMA smoothing
        if is_main_process():
            early_stop_info = early_stop_tracker.update(epoch, val_metrics)
        else:
            early_stop_info = {
                "val_score_raw": 0.0,
                "val_score_ema": 0.0,
                "is_best": False,
                "should_stop": False,
                "epochs_without_improvement": 0,
                "best_score": 0.0,
                "best_epoch": 0,
                "early_stop_flag": 0.0,
            }
        # For backward compatibility with logging
        val_score = early_stop_info["val_score_raw"]

        # ---- Test Evaluation ----
        if IS_SUPERCOMPUTER and is_main_process():
            log_phase_start("test", len(test_loader))

        test_start_time = time.time()
        test_metrics = evaluate_split(
            model=model,
            dataloader=test_loader,
            split_name="test",
            noise_scheduler=noise_scheduler,
            device=device,
            scfg=scfg,
            tcfg=tcfg,
            use_ddp=use_ddp,
            rank=rank
        )
        test_time = time.time() - test_start_time

        # Log test phase completion
        if IS_SUPERCOMPUTER and is_main_process():
            log_phase_complete("test", test_metrics, test_time)

        # Extract test loss for logging
        test_loss = test_metrics.get("loss", float("inf"))

        # ---- FID Computation (val and test) ----
        # Compute FID on validation and test sets (only on main process)
        # FID config is at the root level (cfg.fid), not in train section
        fid_config = cfg.fid if hasattr(cfg, 'fid') and cfg.fid else DEFAULT_FID_CONFIG
        if isinstance(fid_config, bool):
            # If fid is just True/False in config, use defaults
            fid_config = DEFAULT_FID_CONFIG if fid_config else None
        # Only proceed if FID is enabled
        if fid_config and not fid_config.get("enabled", True):
            fid_config = None

        fid_time = 0.0
        if fid_config is not None and is_main_process():
            fid_start_time = time.time()

            # Setup FID weights cache for offline HPC usage
            fid_weights_path = fid_config.get("weights_path") if isinstance(fid_config, dict) else None
            setup_fid_weights_cache(fid_weights_path)

            if IS_SUPERCOMPUTER:
                log_phase_start("FID (validation)", 0)

            val_fid_metrics = compute_fid_on_split(
                model=model,
                dataloader=val_loader,
                noise_scheduler=noise_scheduler,
                device=device,
                tcfg=tcfg,
                scfg=scfg,
                fid_config=fid_config,
                rank=rank,
            )
            # Merge FID metrics into val_metrics
            val_metrics.update(val_fid_metrics)

            if IS_SUPERCOMPUTER:
                log_phase_start("FID (test)", 0)

            test_fid_metrics = compute_fid_on_split(
                model=model,
                dataloader=test_loader,
                noise_scheduler=noise_scheduler,
                device=device,
                tcfg=tcfg,
                scfg=scfg,
                fid_config=fid_config,
                rank=rank,
            )
            # Merge FID metrics into test_metrics
            test_metrics.update(test_fid_metrics)

            fid_time = time.time() - fid_start_time

        # ---- CRITICAL: Compute Training Diagnostics ----
        # These metrics detect common training failures
        # NOTE: In DDP, each rank computes diagnostics on its local validation batch.
        # This is acceptable since diagnostics are rough indicators. Only rank-0's diagnostics are logged.
        val_batch_for_diag = next(iter(val_loader))
        x0_diag = val_batch_for_diag["pixel_values"].to(device)
        labels_diag = val_batch_for_diag["labels"].to(device)

        diagnostics = compute_training_diagnostics(
            model=model,
            x0_batch=x0_diag,
            labels_batch=labels_diag,
            noise_scheduler=noise_scheduler,
            device=device,
            num_samples=min(16, len(x0_diag))
        )

        # Full-chain reconstruction test (includes both PSNR and SSIM)
        full_chain_metrics = full_chain_reconstruction_metrics(
            model=model,
            scheduler=noise_scheduler,
            x0=x0_diag,
            y=labels_diag,
            device=device
        )
        diagnostics["full_chain_psnr"] = full_chain_metrics["psnr"]
        diagnostics["full_chain_ssim"] = full_chain_metrics["ssim"]

        # ---- ELBO Diagnostics ----
        # Compute approximate ELBO decomposition to analyze Min-SNR weighting alignment.
        # This helps identify which timestep regions dominate the loss.
        # Pass the same use_min_snr and min_snr_gamma settings as training so that
        # L_weighted in diagnostics matches the actual training loss.
        elbo_diagnostics = compute_elbo_diagnostics(
            model=model,
            x0_batch=x0_diag,
            labels_batch=labels_diag,
            noise_scheduler=noise_scheduler,
            device=device,
            num_samples=min(32, len(x0_diag)),
            num_timesteps_per_sample=8,
            use_min_snr=getattr(tcfg, 'use_min_snr', False),
            min_snr_gamma=getattr(tcfg, 'min_snr_gamma', 5.0),
        )
        diagnostics.update(elbo_diagnostics)

        # ---- Save class embedding trajectory (rank-0 only) ----
        if is_main_process() and hasattr(tcfg, 'snapshot_class_embedding_every'):
            if tcfg.snapshot_class_embedding_every > 0 and (epoch % tcfg.snapshot_class_embedding_every) == 0:
                emb_out_path = out_dir / "embeddings" / "class_embeddings_trajectory.pt"
                # Use base_model to avoid DDP wrappers; EMA is not active here
                save_class_embeddings_trajectory(base_model, epoch, emb_out_path)

        # Log validation metrics with early stopping diagnostics (only rank-0)
        if csv_logger is not None:
            # Merge early stopping diagnostics into validation metrics
            val_metrics_with_es = dict(val_metrics)
            val_metrics_with_es.update({
                "val_score_raw": early_stop_info["val_score_raw"],
                "val_score_ema": early_stop_info["val_score_ema"],
                "best_val_score": early_stop_info["best_score"],
                "best_val_epoch": early_stop_info["best_epoch"],
                "epochs_without_improvement": early_stop_info["epochs_without_improvement"],
                "early_stop_flag": early_stop_info["early_stop_flag"],
                "is_best_epoch": 1.0 if early_stop_info["is_best"] else 0.0,
            })
            csv_logger.log_epoch(epoch=epoch, split="val", lr=curr_lr, metrics=val_metrics_with_es)

        # Log test metrics (only rank-0)
        if csv_logger is not None:
            csv_logger.log_epoch(epoch=epoch, split="test", lr=curr_lr, metrics=test_metrics)

        # Log diagnostics to separate CSV file (only rank-0)
        if diag_logger is not None:
            diag_logger.log_epoch(epoch=epoch, split="diag", lr=curr_lr, metrics=diagnostics)

        # Validation summary logging (only rank-0)
        if is_main_process():
            if not IS_SUPERCOMPUTER:
                # Interactive mode: detailed print output
                print("\nValidation:")
                print(f"  Average Loss: {val_metrics.get('loss', 0.0):.4f} (Global: {val_loss:.4f})")
                print(f"  PSNR: {val_metrics.get('psnr', 0.0):.2f} dB")
                print(f"  SSIM: {val_metrics.get('ssim', 0.0):.4f}")
                print(f"  Noise MSE: {val_metrics.get('noise_mse', 0.0):.4f}")
                print(f"  Time: {format_time(val_time)}")

                print("\nTest:")
                print(f"  Average Loss: {test_metrics.get('loss', 0.0):.4f} (Global: {test_loss:.4f})")
                print(f"  PSNR: {test_metrics.get('psnr', 0.0):.2f} dB")
                print(f"  SSIM: {test_metrics.get('ssim', 0.0):.4f}")
                print(f"  Noise MSE: {test_metrics.get('noise_mse', 0.0):.4f}")
                print(f"  Time: {format_time(test_time)}")

                print("\n🔍 Training Diagnostics (detecting issues):")
                corr = diagnostics['input_output_correlation']
                pred_std = diagnostics['prediction_std']
                recon_psnr = diagnostics['reconstruction_psnr_t500']

                print(f"  Reconstruction MSE@t500: {diagnostics['reconstruction_mse_t500']:.4f}")
                print(f"  Gradient Norm (mean): {train_metrics.get('grad_norm', 0.0):.4f}")
                print("="*80 + "\n")
            else:
                # Supercomputer mode: structured logger output
                logger.info(f"Validation complete | loss={val_loss:.4f} | "
                           f"psnr={val_metrics.get('psnr', 0.0):.2f}dB | "
                           f"ssim={val_metrics.get('ssim', 0.0):.4f} | "
                           f"time={format_time(val_time)}")
                logger.info(f"Test complete | loss={test_loss:.4f} | "
                           f"psnr={test_metrics.get('psnr', 0.0):.2f}dB | "
                           f"ssim={test_metrics.get('ssim', 0.0):.4f} | "
                           f"time={format_time(test_time)}")

                # Log diagnostics with warnings
                corr = diagnostics['input_output_correlation']
                pred_std = diagnostics['prediction_std']
                recon_psnr = diagnostics['reconstruction_psnr_t500']

                diag_msg = (f"Diagnostics | corr={corr:.4f} | pred_std={pred_std:.4f} | "
                           f"recon_psnr@t500={recon_psnr:.2f}dB")
                logger.info(diag_msg)

        model.train()

        # ---- Record Augmentation Statistics ----
        if augmentation_stats is not None and len(epoch_augmentation_transforms) > 0:
            augmentation_stats.record_batch(epoch_augmentation_transforms, epoch=epoch)

            # Save statistics periodically if configured
            aug_cfg = cfg.ccddpm.augmentation
            if aug_cfg.statistics.save_every_n_epochs > 0:
                if epoch % aug_cfg.statistics.save_every_n_epochs == 0:
                    augmentation_stats.save_csv(final=False)
                    logger.info(f"Saved augmentation statistics (epoch {epoch})")

        # ---- LR Scheduler (per-epoch stepping) ----
        # Step the scheduler at end of epoch (if not per-batch)
        if lr_scheduler is not None and not lr_step_per_batch:
            lr_scheduler.step()

        # ---- Checkpointing & Early Stopping ----
        # Note: early_stop_info was computed earlier with the EarlyStoppingTracker

        # Extract early stopping decisions from tracker (rank-0 computed earlier)
        if is_main_process():
            is_best = early_stop_info["is_best"]
            should_stop = early_stop_info["should_stop"]
            best_val_score = early_stop_info["best_score"]
            best_epoch = early_stop_info["best_epoch"]
            epochs_without_improvement = early_stop_info["epochs_without_improvement"]
        else:
            is_best = False
            should_stop = False
            best_val_score = 0.0
            best_epoch = 0
            epochs_without_improvement = 0

        # Broadcast early stopping decision to all processes
        if use_ddp:
            should_stop = broadcast_bool(should_stop)

        # Save checkpoints (only rank-0)
        if is_main_process():
            checkpoint_data = {
                "model": get_state_dict_for_save(model),  # Handles DDP wrapper
                "opt": opt.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "ema": (ema.shadow if ema else None),
                "cfg": tcfg.__dict__,
                # Save diagnostics for post-training analysis
                "diagnostics": diagnostics,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                # EMA verification
                "ema_enabled": ema is not None,
                "ema_num_params": len(ema.shadow) if ema else 0,
                # Early stopping state
                "early_stop_info": early_stop_info,
                # LR scheduler state (for resume)
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }

            # Always save last.pt
            torch.save(checkpoint_data, out_dir / "ckpts" / "last.pt")

            # Save best.pt if this is the best validation score so far
            if is_best:
                torch.save(checkpoint_data, out_dir / "ckpts" / "best.pt")
                log_checkpoint_saved(str(out_dir / "ckpts" / "best.pt"), is_best=True, epoch=epoch)
                # Save snapshot of best metrics for final summary
                best_val_metrics_snapshot = dict(val_metrics)
            else:
                if not IS_SUPERCOMPUTER:
                    logger.info(f"No improvement for {epochs_without_improvement}/{early_stop_tracker.patience} epochs "
                               f"(best: {best_val_score:.4f} at epoch {best_epoch})")

            # Save periodic checkpoint every X epochs
            if (epoch % tcfg.ckpt_every_epochs) == 0:
                ck = out_dir / "ckpts" / f"epoch_{epoch:04d}.pt"
                torch.save(checkpoint_data, ck)
                log_checkpoint_saved(str(ck), is_best=False, epoch=epoch)

        # Calculate total epoch time
        total_epoch_time = train_time + val_time + test_time

        # Epoch summary logging (enhanced for supercomputer)
        if is_main_process():
            log_enhanced_epoch_summary(
                epoch=epoch,
                total_epochs=tcfg.epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                train_time=train_time,
                val_time=val_time,
                test_time=test_time,
                fid_time=fid_time,
                lr=curr_lr,
                early_stop_info=early_stop_info,
                num_classes=tcfg.num_classes,
            )

        # Early stopping check (synchronized across all GPUs)
        if should_stop:
            if is_main_process():
                log_early_stopping_triggered(
                    epoch=epoch,
                    patience=early_stop_tracker.patience,
                    best_epoch=best_epoch,
                    best_score=best_val_score,
                    ema_score=early_stop_info['val_score_ema'],
                )
                # Log training completion
                total_training_time = time.time() - training_start_time
                log_training_complete(
                    total_epochs_run=epoch,
                    total_time=total_training_time,
                    best_epoch=best_epoch,
                    best_val_metrics=best_val_metrics_snapshot,
                    output_dir=str(out_dir),
                )
            break
        
        # ---- Visualizations (only rank-0) ----
        # Skip visualizations on non-main processes to save computation
        if is_main_process():
            # Use EMA weights for better quality if available
            original_state = None
            if ema:
                # Save original state and copy EMA to base_model (unwrapped)
                original_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                ema.copy_to(base_model)

            model.eval()

            # Every epoch: Save reconstruction comparisons (x0 predictions vs ground truth)
            with torch.no_grad():
                # Get a batch from validation set
                val_iter = iter(val_loader)
                vis_batch = next(val_iter)
                # Take up to 5 images, but handle smaller batches
                num_vis = min(5, vis_batch["pixel_values"].size(0))
                x0_vis = vis_batch["pixel_values"][:num_vis].to(device)
                y_vis = vis_batch["labels"][:num_vis].to(device)

                # Sample random timesteps for reconstruction visualization
                torch.manual_seed(epoch)  # Reproducible per epoch
                t_vis = torch.randint(100, scfg.num_train_timesteps // 2, (num_vis,), device=device, dtype=torch.long)
                noise_vis = torch.randn_like(x0_vis)
                x_t_vis = noise_scheduler.add_noise(x0_vis, noise_vis, t_vis)

                # Predict noise and reconstruct x0
                noise_pred_vis = model(x_t_vis, t_vis, y_vis)
                sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t_vis].sqrt().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t_vis]).sqrt().view(-1, 1, 1, 1)
                x0_pred_vis = (x_t_vis - sqrt_one_minus_alpha_prod * noise_pred_vis) / sqrt_alpha_prod
                x0_pred_vis = torch.clamp(x0_pred_vis, -1.0, 1.0)

                # Convert to [0, 1] for saving
                x0_vis_01 = (x0_vis + 1.0) / 2.0
                x0_pred_vis_01 = (x0_pred_vis + 1.0) / 2.0

                # Interleave: original, noisy, reconstruction
                comparison = torch.stack([x0_vis_01, (x_t_vis + 1.0) / 2.0, x0_pred_vis_01], dim=1).flatten(0, 1)
                save_image(comparison, out_dir / "samples" / f"epoch_{epoch:04d}_recon.png",
                          nrow=3, normalize=False, value_range=(0, 1))

            # Every epoch: Save DDPM-specific visualizations
            if epoch % 1 == 0 or epoch == 1:
                with torch.no_grad():
                    # Get a single image for detailed visualization
                    single_img = x0_vis[0:1]
                    single_label = y_vis[0:1]

                    # 1. Noising process visualization
                    noising_steps = visualize_noising_process(
                        single_img, noise_scheduler, num_steps=10, device=device
                    )
                    noising_steps_01 = (noising_steps + 1.0) / 2.0
                    save_image(noising_steps_01, out_dir / "samples" / f"epoch_{epoch:04d}_noising.png",
                              nrow=11, normalize=False, value_range=(0, 1))

                    # 2. Denoising process visualization (full sampling)
                    denoising_steps = visualize_denoising_process(
                        model, noise_scheduler,
                        shape=(tcfg.in_channels, tcfg.image_size, tcfg.image_size),
                        class_label=single_label,
                        num_steps=10,
                        device=device,
                        guidance_scale=cfg.ccddpm.infer.guidance_scale
                    )
                    # Now returns initial + 10 intermediate + final = 12 images
                    denoising_steps_01 = (denoising_steps + 1.0) / 2.0
                    save_image(denoising_steps_01, out_dir / "samples" / f"epoch_{epoch:04d}_denoising.png",
                              nrow=12, normalize=False, value_range=(0, 1))
                    logger.info(f"Denoising visualization: generated {denoising_steps.shape[0]} steps")

                    # 3. Multi-timestep reconstruction visualization
                    timesteps_to_vis = [50, 150, 300, 500, 700, 900]
                    multistep_recons = visualize_multistep_reconstruction(
                        model, single_img, noise_scheduler, single_label,
                        timesteps=timesteps_to_vis, device=device
                    )
                    multistep_recons_01 = (multistep_recons + 1.0) / 2.0
                    save_image(multistep_recons_01, out_dir / "samples" / f"epoch_{epoch:04d}_multistep.png",
                              nrow=len(timesteps_to_vis) + 1, normalize=False, value_range=(0, 1))

                    # 4. Class-conditional samples (one per class)
                    class_samples = []
                    logger.info(f"Generating class-conditional samples for {tcfg.num_classes} classes...")
                    for c in range(tcfg.num_classes):
                        class_label_sample = torch.tensor([c], device=device)
                        sample = visualize_denoising_process(
                            model, noise_scheduler,
                            shape=(tcfg.in_channels, tcfg.image_size, tcfg.image_size),
                            class_label=class_label_sample,
                            num_steps=10,  # Minimal intermediate step (returns 3 images: initial + 1 intermediate + final)
                            device=device,
                            guidance_scale=cfg.ccddpm.infer.guidance_scale 
                        )
                        # sample has shape [3, C, H, W]: [initial_noise, intermediate, final_image]
                        # Take only the final denoised image
                        final_image = sample[-1:]
                        class_samples.append(final_image)
                        logger.info("  Class {}: sample shape={}, final shape={}, value range=[{:.3f}, {:.3f}]".format(
                            c, sample.shape, final_image.shape, final_image.min(), final_image.max()))

                    class_samples_tensor = torch.cat(class_samples, dim=0)
                    class_samples_01 = (class_samples_tensor + 1.0) / 2.0
                    save_image(class_samples_01, out_dir / "samples" / f"epoch_{epoch:04d}_classes.png",
                              nrow=tcfg.num_classes, normalize=False, value_range=(0, 1))
                    logger.info("Saved class-conditional samples: {}".format(class_samples_tensor.shape))

                    # 5. Conditioning sanity check - verify class labels affect predictions
                    cond_stats = conditioning_sanity_check(
                        model, noise_scheduler,
                        num_classes=tcfg.num_classes,
                        image_shape=(tcfg.in_channels, tcfg.image_size, tcfg.image_size),
                        device=device,
                        num_samples=10
                    )
                    logger.info("Conditioning check: gap_mean={:.4f}, gap_std={:.4f} (should be >0 and growing over epochs)".format(
                        cond_stats['conditioning_gap_mean'], cond_stats['conditioning_gap_std']))

                    logger.info("Saved detailed visualizations for epoch {}".format(epoch))

            # Restore training weights if we used EMA for visualization (still within rank-0 block)
            if ema and original_state is not None:
                base_model.load_state_dict(original_state)

        # All processes set model back to training mode
        model.train()

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================
    # Calculate total training time
    total_training_time = time.time() - training_start_time

    # Save final augmentation statistics (rank-0 only)
    if augmentation_stats is not None:
        augmentation_stats.save_csv(final=True)
        augmentation_stats.print_summary()
        logger.info("Final augmentation statistics saved")

    # Print summary (rank-0 only)
    if is_main_process():
        # Use enhanced logging for training completion
        log_training_complete(
            total_epochs_run=epoch,
            total_time=total_training_time,
            best_epoch=best_epoch,
            best_val_metrics=best_val_metrics_snapshot,
            output_dir=str(out_dir),
        )

        # Additional details (non-supercomputer mode)
        if not IS_SUPERCOMPUTER:
            print(f"\nCheckpoints saved in: {out_dir / 'ckpts'}")
            print(f"  - best.pt: Best model (epoch {best_epoch}, val_score={best_val_score:.4f})")
            print(f"  - last.pt: Final epoch model (epoch {epoch})")
            print(f"  - epoch_XXXX.pt: Periodic checkpoints every {tcfg.ckpt_every_epochs} epochs")
            print(f"\nVisualizations saved in: {out_dir / 'samples'}")
            print(f"Metrics logged in: {out_dir / 'training_metrics.csv'}")
            print("="*80 + "\n")

        # Generate training evolution visualizations (only rank-0)
        try:
            logger.info("Generating training evolution visualizations...")
            vis_dir = generate_training_visualizations(out_dir, use_latex=False, dpi=300)
            logger.info(f"Training visualizations saved to: {vis_dir}")
            print(f"\nTraining evolution plots saved to: {vis_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate training visualizations: {e}")
            print(f"\nWarning: Could not generate training visualizations: {e}")

    # ========================================================================
    # CLEANUP
    # ========================================================================
    if use_ddp:
        # DDP: Clean up process group
        cleanup()
        logger.info("DDP cleanup completed")
    # Legacy: No cleanup needed
