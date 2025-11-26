from typing import Dict, Any
import logging
import math
import torch
import torch.nn as nn

from medsyn.models.ccDDPM.engine.utils.time_formatting import format_time

logger = logging.getLogger("medsyn.ccddpm.train")


def log_training_progress(rank: int, epoch: int, step: int, total_steps: int,
                          metrics: Dict[str, float], elapsed: float,
                          eta: float, lr: float) -> None:
    """
    Log training progress with metrics and time estimates.

    Args:
        rank: Process rank
        epoch: Current epoch
        step: Current step within epoch
        total_steps: Total steps in epoch
        metrics: Dictionary of metrics (loss, psnr, etc.)
        elapsed: Time elapsed since epoch start (seconds)
        eta: Estimated time remaining for epoch (seconds)
        lr: Current learning rate
    """
    progress_pct = (step / total_steps) * 100

    # Format metrics string
    metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])

    # Main process gets detailed logs
    if rank == 0:
        logger.info(
            f"Epoch {epoch} | Step {step}/{total_steps} ({progress_pct:.1f}%) | "
            f"{metrics_str} | lr={lr:.2e} | "
            f"Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}"
        )
    else:
        # Non-main processes log less frequently and with rank prefix
        logger.debug(
            f"[Rank {rank}] Epoch {epoch} | Step {step}/{total_steps} ({progress_pct:.1f}%) | "
            f"{metrics_str}"
        )


def log_epoch_summary(rank: int, epoch: int, train_metrics: Dict[str, float],
                     val_metrics: Dict[str, float], test_metrics: Dict[str, float],
                     epoch_time: float, best_val_score: float, is_best: bool) -> None:
    """
    Log end-of-epoch summary with training, validation, and test metrics.

    Args:
        rank: Process rank
        epoch: Current epoch
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        test_metrics: Test metrics dictionary
        epoch_time: Total time for epoch (seconds)
        best_val_score: Best validation score (PSNR + 50*SSIM) so far
        is_best: Whether this epoch achieved best validation score
    """
    if rank == 0:
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Training   | loss={train_metrics.get('loss', 0):.4f} | "
                   f"psnr={train_metrics.get('psnr', 0):.2f}dB | "
                   f"ssim={train_metrics.get('ssim', 0):.4f}")
        logger.info(f"Validation | loss={val_metrics.get('loss', 0):.4f} | "
                   f"psnr={val_metrics.get('psnr', 0):.2f}dB | "
                   f"ssim={val_metrics.get('ssim', 0):.4f}")
        logger.info(f"Test       | loss={test_metrics.get('loss', 0):.4f} | "
                   f"psnr={test_metrics.get('psnr', 0):.2f}dB | "
                   f"ssim={test_metrics.get('ssim', 0):.4f}")

        # Compute current val_score for display
        curr_val_score = val_metrics.get('psnr', 0) + 50.0 * val_metrics.get('ssim', 0)
        if is_best:
            logger.info(f"🌟 NEW BEST MODEL! Val score: {curr_val_score:.2f} "
                       f"(PSNR + 50*SSIM)")
        else:
            logger.info(f"Best val score: {best_val_score:.2f} (PSNR + 50*SSIM)")

        logger.info(f"Epoch time: {format_time(epoch_time)}")
        logger.info("=" * 80)


def log_validation_progress(rank: int, step: int, total_steps: int,
                            loss: float, main_process_only: bool = True) -> None:
    """
    Log validation progress.

    Args:
        rank: Process rank
        step: Current validation step
        total_steps: Total validation steps
        loss: Current validation loss
        main_process_only: If True, only log on rank 0
    """
    if main_process_only and rank != 0:
        return

    progress_pct = (step / total_steps) * 100

    if rank == 0:
        logger.info(f"Validation | Step {step}/{total_steps} ({progress_pct:.1f}%) | "
                   f"loss={loss:.4f}")


def should_log_step(step: int, total_steps: int, log_frequency: int = 10) -> bool:
    """
    Determine if we should log at this step based on epoch progress.

    Logs at regular intervals throughout the epoch (default: 10 times per epoch).
    Always logs first and last step.

    Args:
        step: Current step (1-indexed)
        total_steps: Total steps in epoch
        log_frequency: Number of times to log per epoch

    Returns:
        True if should log at this step
    """
    # Always log first and last step
    if step == 1 or step == total_steps:
        return True

    # Log at regular intervals
    log_interval = max(1, total_steps // log_frequency)
    return step % log_interval == 0


# =============================================================================
# ENHANCED LOGGING FOR SUPERCOMPUTER ENVIRONMENT
# =============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information in GB."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def log_training_config(
    tcfg: Any,
    optimizer_cfg: Any,
    scfg: Any,
    dist_cfg: Any,
    augmentation_cfg: Any,
    num_train_samples: int,
    num_val_samples: int,
    num_test_samples: int,
    steps_per_epoch: int,
) -> None:
    """
    Log comprehensive training configuration at startup.

    This provides a complete overview of all training parameters for reproducibility
    and debugging in supercomputer batch job logs.
    """
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 80)

    # Training parameters
    logger.info("TRAINING PARAMETERS:")
    logger.info(f"  Epochs: {tcfg.epochs}")
    logger.info(f"  Batch size: {tcfg.batch_size}")
    logger.info(f"  Image size: {tcfg.image_size}x{tcfg.image_size}")
    logger.info(f"  Input channels: {tcfg.in_channels}")
    logger.info(f"  Num classes: {tcfg.num_classes}")
    logger.info(f"  Class embed dim: {tcfg.class_embed_dim}")
    logger.info(f"  Mixed precision: {tcfg.mixed_precision}")
    logger.info(f"  Gradient clip norm: {tcfg.grad_clip_norm}")
    logger.info(f"  Skip non-finite grads: {tcfg.skip_nonfinite_grads}")

    # Optimizer configuration
    logger.info("")
    logger.info("OPTIMIZER CONFIGURATION:")
    logger.info(f"  Type: {optimizer_cfg.type}")
    logger.info(f"  Learning rate: {optimizer_cfg.lr:.2e}")
    logger.info(f"  Weight decay: {optimizer_cfg.wd:.2e}")
    if optimizer_cfg.type.lower() in ["adam", "adamw"]:
        logger.info(f"  Betas: {optimizer_cfg.betas}")
        logger.info(f"  Epsilon: {optimizer_cfg.eps}")
        logger.info(f"  AMSGrad: {getattr(optimizer_cfg, 'amsgrad', False)}")
    elif optimizer_cfg.type.lower() == "sgd":
        logger.info(f"  Momentum: {getattr(optimizer_cfg, 'momentum', 0.9)}")
        logger.info(f"  Nesterov: {getattr(optimizer_cfg, 'nesterov', False)}")

    # LR Scheduler configuration
    if hasattr(tcfg, 'lr_scheduler') and tcfg.lr_scheduler is not None:
        lr_cfg = tcfg.lr_scheduler
        logger.info("")
        logger.info("LR SCHEDULER CONFIGURATION:")
        logger.info(f"  Type: {lr_cfg.type}")
        if lr_cfg.type.lower() == "onecycle":
            logger.info(f"  Max LR: {lr_cfg.max_lr}")
            logger.info(f"  Pct start: {lr_cfg.pct_start}")
            logger.info(f"  Div factor: {lr_cfg.div_factor}")
        elif lr_cfg.type.lower() in ["cosine", "cosine_warmup"]:
            logger.info(f"  Eta min: {lr_cfg.eta_min}")
            logger.info(f"  Warmup epochs: {lr_cfg.warmup_epochs}")

    # Diffusion scheduler configuration
    logger.info("")
    logger.info("DIFFUSION SCHEDULER:")
    logger.info(f"  Train timesteps: {scfg.num_train_timesteps}")
    logger.info(f"  Beta schedule: {scfg.beta_schedule}")
    logger.info(f"  Beta range: [{scfg.beta_start:.2e}, {scfg.beta_end:.2e}]")
    logger.info(f"  Prediction type: {scfg.prediction_type}")

    # Loss configuration
    logger.info("")
    logger.info("LOSS CONFIGURATION:")
    logger.info(f"  Min-SNR weighting: {tcfg.use_min_snr}")
    if tcfg.use_min_snr:
        logger.info(f"  Min-SNR gamma: {tcfg.min_snr_gamma}")
    logger.info(f"  Per-class loss weighting: {tcfg.per_class_loss_weighting}")
    logger.info(f"  Guidance p_uncond: {tcfg.guidance_p_uncond}")

    # EMA configuration
    logger.info("")
    logger.info("EMA CONFIGURATION:")
    logger.info(f"  Enabled: {tcfg.ema_use}")
    if tcfg.ema_use:
        logger.info(f"  Decay: {tcfg.ema_decay}")

    # Early stopping configuration
    if hasattr(tcfg, 'early_stopping') and tcfg.early_stopping is not None:
        es_cfg = tcfg.early_stopping
        logger.info("")
        logger.info("EARLY STOPPING:")
        logger.info(f"  Patience: {es_cfg.patience}")
        logger.info(f"  Metric: {es_cfg.metric}")
        logger.info(f"  PSNR weight: {es_cfg.psnr_weight}")
        logger.info(f"  SSIM weight: {es_cfg.ssim_weight}")
        logger.info(f"  EMA alpha: {es_cfg.ema_alpha}")
        logger.info(f"  Use EMA for stopping: {es_cfg.use_ema_for_stopping}")
        logger.info(f"  Min delta: {es_cfg.min_delta}")

    # Distributed configuration
    logger.info("")
    logger.info("DISTRIBUTED TRAINING:")
    logger.info(f"  Enabled: {dist_cfg.enabled}")
    if dist_cfg.enabled:
        logger.info(f"  Backend: {dist_cfg.backend}")
        logger.info(f"  Num GPUs: {dist_cfg.num_gpus}")
        logger.info(f"  Grad accum steps: {dist_cfg.grad_accum_steps}")

    # Augmentation configuration
    logger.info("")
    logger.info("AUGMENTATION:")
    if augmentation_cfg is not None and hasattr(augmentation_cfg, 'enabled'):
        logger.info(f"  Enabled: {augmentation_cfg.enabled}")
        if augmentation_cfg.enabled:
            logger.info(f"  Probability: {augmentation_cfg.probability}")
            logger.info(f"  Num transforms: {len(augmentation_cfg.transforms)}")
    else:
        logger.info("  Enabled: False")

    # Dataset statistics
    logger.info("")
    logger.info("DATASET STATISTICS:")
    logger.info(f"  Train samples: {num_train_samples:,}")
    logger.info(f"  Val samples: {num_val_samples:,}")
    logger.info(f"  Test samples: {num_test_samples:,}")
    logger.info(f"  Steps per epoch: {steps_per_epoch:,}")
    total_steps = steps_per_epoch * tcfg.epochs
    logger.info(f"  Total training steps: {total_steps:,}")

    # Estimated training time (rough estimate based on typical step time)
    # This will be updated with actual timings after first epoch
    logger.info("")
    logger.info("=" * 80)


def log_model_summary(model: nn.Module, ucfg: Any) -> None:
    """Log model architecture summary with parameter counts."""
    logger.info("")
    logger.info("MODEL ARCHITECTURE:")
    logger.info(f"  Model channels: {ucfg.model_channels}")
    logger.info(f"  Channel multipliers: {ucfg.channel_mult}")
    logger.info(f"  Block out channels: {ucfg.get_block_out_channels()}")
    logger.info(f"  Layers per block: {ucfg.layers_per_block}")
    logger.info(f"  Down blocks: {ucfg.down_block_types}")
    logger.info(f"  Up blocks: {ucfg.up_block_types}")
    logger.info(f"  Attention head dim: {ucfg.attention_head_dim}")
    logger.info(f"  Dropout: {ucfg.dropout}")
    logger.info(f"  Norm groups: {ucfg.norm_num_groups}")

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("")
    logger.info("PARAMETER COUNTS:")
    logger.info(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # Memory estimate (rough)
    param_memory_mb = total_params * 4 / 1e6  # 4 bytes per float32 param
    logger.info(f"  Estimated param memory: {param_memory_mb:.1f} MB")


def log_enhanced_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    train_time: float,
    val_time: float,
    test_time: float,
    fid_time: float,
    lr: float,
    early_stop_info: Dict[str, Any],
    num_classes: int = 9,
) -> None:
    """
    Log comprehensive epoch summary optimized for supercomputer batch logs.

    Includes all metrics, per-class analysis, timing breakdown, and training health indicators.
    """
    total_time = train_time + val_time + test_time + fid_time

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"EPOCH {epoch}/{total_epochs} COMPLETE")
    logger.info("=" * 80)

    # Core metrics comparison table
    logger.info("")
    logger.info("CORE METRICS:")
    logger.info(f"  {'Split':<10} | {'Loss':>10} | {'PSNR (dB)':>10} | {'SSIM':>10}")
    logger.info(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    logger.info(f"  {'Train':<10} | {train_metrics.get('loss', 0):>10.4f} | "
               f"{train_metrics.get('psnr', 0):>10.2f} | {train_metrics.get('ssim', 0):>10.4f}")
    logger.info(f"  {'Val':<10} | {val_metrics.get('loss', 0):>10.4f} | "
               f"{val_metrics.get('psnr', 0):>10.2f} | {val_metrics.get('ssim', 0):>10.4f}")
    logger.info(f"  {'Test':<10} | {test_metrics.get('loss', 0):>10.4f} | "
               f"{test_metrics.get('psnr', 0):>10.2f} | {test_metrics.get('ssim', 0):>10.4f}")

    # FID metrics (if available)
    fid_global_val = val_metrics.get('fid_global', float('nan'))
    fid_global_test = test_metrics.get('fid_global', float('nan'))
    if not (math.isnan(fid_global_val) and math.isnan(fid_global_test)):
        logger.info("")
        logger.info("FID METRICS (lower is better):")
        if not math.isnan(fid_global_val):
            logger.info(f"  Val FID: {fid_global_val:.2f}")
        if not math.isnan(fid_global_test):
            logger.info(f"  Test FID: {fid_global_test:.2f}")

    # Per-class metrics summary (validation) - show best and worst
    logger.info("")
    log_per_class_summary(val_metrics, "Validation", num_classes)

    # Early stopping status
    logger.info("")
    logger.info("EARLY STOPPING STATUS:")
    val_score_raw = early_stop_info.get('val_score_raw', 0)
    val_score_ema = early_stop_info.get('val_score_ema', 0)
    best_score = early_stop_info.get('best_score', 0)
    best_epoch = early_stop_info.get('best_epoch', 0)
    epochs_without_improvement = early_stop_info.get('epochs_without_improvement', 0)
    is_best = early_stop_info.get('is_best', False)

    logger.info(f"  Current score (raw): {val_score_raw:.4f}")
    logger.info(f"  Current score (EMA): {val_score_ema:.4f}")
    logger.info(f"  Best score: {best_score:.4f} (epoch {best_epoch})")
    logger.info(f"  Epochs without improvement: {epochs_without_improvement}")

    if is_best:
        logger.info("  >>> NEW BEST MODEL SAVED <<<")

    # Timing breakdown
    logger.info("")
    logger.info("TIMING BREAKDOWN:")
    logger.info(f"  Training: {format_time(train_time)} ({train_time/total_time*100:.1f}%)")
    logger.info(f"  Validation: {format_time(val_time)} ({val_time/total_time*100:.1f}%)")
    logger.info(f"  Test: {format_time(test_time)} ({test_time/total_time*100:.1f}%)")
    if fid_time > 0:
        logger.info(f"  FID computation: {format_time(fid_time)} ({fid_time/total_time*100:.1f}%)")
    logger.info(f"  Total: {format_time(total_time)}")

    # Training health indicators
    logger.info("")
    logger.info("TRAINING HEALTH:")
    logger.info(f"  Learning rate: {lr:.2e}")
    grad_norm = train_metrics.get('grad_norm', 0)
    logger.info(f"  Avg gradient norm: {grad_norm:.4f}")

    # GPU memory (if available)
    mem_info = get_gpu_memory_info()
    if mem_info['allocated_gb'] > 0:
        logger.info(f"  GPU memory: {mem_info['allocated_gb']:.2f} GB allocated, "
                   f"{mem_info['max_allocated_gb']:.2f} GB peak")

    # ETA for training completion
    epochs_remaining = total_epochs - epoch
    if epochs_remaining > 0:
        eta_seconds = epochs_remaining * total_time
        logger.info("")
        logger.info(f"ETA for training completion: {format_time(eta_seconds)} ({epochs_remaining} epochs remaining)")

    logger.info("=" * 80)


def log_per_class_summary(
    metrics: Dict[str, float],
    split_name: str,
    num_classes: int = 9,
) -> None:
    """
    Log per-class metrics summary showing best and worst performing classes.
    """
    # Collect per-class PSNR and SSIM
    psnr_per_class = {}
    ssim_per_class = {}
    loss_per_class = {}

    for k in range(num_classes):
        psnr_key = f"psnr_c{k}"
        ssim_key = f"ssim_c{k}"
        loss_key = f"loss_raw_c{k}"

        if psnr_key in metrics and not math.isnan(metrics[psnr_key]):
            psnr_per_class[k] = metrics[psnr_key]
        if ssim_key in metrics and not math.isnan(metrics[ssim_key]):
            ssim_per_class[k] = metrics[ssim_key]
        if loss_key in metrics and not math.isnan(metrics[loss_key]):
            loss_per_class[k] = metrics[loss_key]

    if not psnr_per_class:
        logger.info(f"PER-CLASS {split_name.upper()} METRICS: Not available")
        return

    logger.info(f"PER-CLASS {split_name.upper()} METRICS:")

    # Sort by PSNR to find best/worst
    sorted_by_psnr = sorted(psnr_per_class.items(), key=lambda x: x[1], reverse=True)

    # Show all classes in a compact format
    psnr_str = " | ".join([f"c{k}:{v:.1f}" for k, v in sorted_by_psnr])
    logger.info(f"  PSNR: {psnr_str}")

    if ssim_per_class:
        sorted_by_ssim = sorted(ssim_per_class.items(), key=lambda x: x[1], reverse=True)
        ssim_str = " | ".join([f"c{k}:{v:.3f}" for k, v in sorted_by_ssim])
        logger.info(f"  SSIM: {ssim_str}")

    # Highlight best and worst
    if len(sorted_by_psnr) >= 2:
        best_class, best_psnr = sorted_by_psnr[0]
        worst_class, worst_psnr = sorted_by_psnr[-1]
        logger.info(f"  Best: class {best_class} (PSNR={best_psnr:.2f}dB) | "
                   f"Worst: class {worst_class} (PSNR={worst_psnr:.2f}dB)")


def log_epoch_start(epoch: int, total_epochs: int, lr: float) -> None:
    """Log epoch start marker with current learning rate."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"STARTING EPOCH {epoch}/{total_epochs} | LR: {lr:.2e}")
    logger.info("=" * 80)


def log_phase_start(phase_name: str, num_batches: int) -> None:
    """Log the start of a training phase (train/val/test)."""
    logger.info(f"[{phase_name.upper()}] Starting {num_batches} batches...")


def log_phase_complete(
    phase_name: str,
    metrics: Dict[str, float],
    elapsed_time: float,
) -> None:
    """Log the completion of a training phase with key metrics."""
    loss = metrics.get('loss', 0)
    psnr = metrics.get('psnr', 0)
    ssim = metrics.get('ssim', 0)

    logger.info(f"[{phase_name.upper()}] Complete | "
               f"loss={loss:.4f} | psnr={psnr:.2f}dB | ssim={ssim:.4f} | "
               f"time={format_time(elapsed_time)}")


def log_checkpoint_saved(checkpoint_path: str, is_best: bool, epoch: int) -> None:
    """Log checkpoint save event."""
    if is_best:
        logger.info(f"[CHECKPOINT] Saved BEST model: {checkpoint_path} (epoch {epoch})")
    else:
        logger.info(f"[CHECKPOINT] Saved: {checkpoint_path}")


def log_early_stopping_triggered(
    epoch: int,
    patience: int,
    best_epoch: int,
    best_score: float,
    ema_score: float,
) -> None:
    """Log early stopping trigger event."""
    logger.warning("")
    logger.warning("=" * 80)
    logger.warning(f"EARLY STOPPING TRIGGERED at Epoch {epoch}")
    logger.warning("=" * 80)
    logger.warning(f"  No improvement for {patience} consecutive epochs")
    logger.warning(f"  Best model: epoch {best_epoch} with score {best_score:.4f}")
    logger.warning(f"  Final EMA score: {ema_score:.4f}")
    logger.warning("=" * 80)


def log_training_complete(
    total_epochs_run: int,
    total_time: float,
    best_epoch: int,
    best_val_metrics: Dict[str, float],
    output_dir: str,
) -> None:
    """Log training completion summary."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total epochs: {total_epochs_run}")
    logger.info(f"  Total time: {format_time(total_time)}")
    logger.info(f"  Best epoch: {best_epoch}")
    logger.info(f"  Best val PSNR: {best_val_metrics.get('psnr', 0):.2f} dB")
    logger.info(f"  Best val SSIM: {best_val_metrics.get('ssim', 0):.4f}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 80)