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
import torch.optim as optim
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
)
from medsyn.models.ccDDPM.training_logging import (
    CSVTrainingLogger, EpochAverager, TRAINING_FIELDS, DIAGNOSTIC_FIELDS, NUM_CLASSES,
    DEFAULT_FID_CONFIG,
)
from medsyn.models.ccDDPM.engine.ddp_utils import (
    ddp_is_enabled, ddp_init, is_main_process,
    barrier, cleanup, all_reduce_mean, broadcast_bool, get_state_dict_for_save
)
from medsyn.models.ccDDPM.vis.visualize_training_evolution import generate_training_visualizations
import numpy as np

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


# =============================================================================
# EARLY STOPPING TRACKER WITH EMA SMOOTHING
# =============================================================================

class EarlyStoppingTracker:
    """
    Early stopping tracker with EMA smoothing of the validation score.

    Computes a composite validation score:
        S = (psnr_weight * PSNR / 20) + (ssim_weight * SSIM)

    where PSNR is divided by 20 to put it on a similar scale to SSIM (~0-1).

    The score is optionally smoothed with an exponential moving average:
        S_ema_t = ema_alpha * S_ema_{t-1} + (1 - ema_alpha) * S_t

    Higher ema_alpha means more smoothing (less sensitive to fluctuations).

    Attributes:
        patience: Number of epochs without improvement before stopping
        psnr_weight: Weight for PSNR in composite score
        ssim_weight: Weight for SSIM in composite score
        ema_alpha: EMA smoothing factor (0 = no smoothing)
        use_ema_for_stopping: Whether to use EMA score for stopping decisions
        min_delta: Minimum improvement to count as "better"
    """

    def __init__(
        self,
        patience: int = 5,
        psnr_weight: float = 1.0,
        ssim_weight: float = 1.0,
        ema_alpha: float = 0.7,
        use_ema_for_stopping: bool = True,
        min_delta: float = 0.001,
        metric: str = "psnr_ssim_composite",
    ):
        self.patience = patience
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.ema_alpha = ema_alpha
        self.use_ema_for_stopping = use_ema_for_stopping
        self.min_delta = min_delta
        self.metric = metric

        # State
        self.best_score: float = float("-inf")
        self.best_ema_score: float = float("-inf")
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0
        self.score_ema: Optional[float] = None
        self._history: list = []

    def compute_score(self, val_metrics: Dict[str, float]) -> float:
        """
        Compute the validation score from metrics.

        Args:
            val_metrics: Dictionary with 'psnr', 'ssim', and/or 'loss' keys

        Returns:
            Composite validation score (higher is better)
        """
        if self.metric == "psnr_ssim_composite":
            psnr = val_metrics.get("psnr", 0.0)
            ssim = val_metrics.get("ssim", 0.0)
            # Divide PSNR by 20 to put it on similar scale to SSIM (~0-1)
            score = (self.psnr_weight * psnr / 20.0) + (self.ssim_weight * ssim)
        elif self.metric == "psnr":
            score = val_metrics.get("psnr", 0.0)
        elif self.metric == "ssim":
            score = val_metrics.get("ssim", 0.0)
        elif self.metric == "loss":
            # Negate loss so higher is still better
            score = -val_metrics.get("loss", float("inf"))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return score

    def update(self, epoch: int, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Update tracker with new validation metrics.

        Args:
            epoch: Current epoch number
            val_metrics: Dictionary with validation metrics

        Returns:
            Dictionary with early stopping diagnostics:
                - val_score_raw: Raw composite validation score
                - val_score_ema: EMA-smoothed validation score
                - is_best: Whether this epoch is the best so far
                - should_stop: Whether early stopping should trigger
                - epochs_without_improvement: Number of epochs since last improvement
                - best_score: Best score seen so far
                - best_epoch: Epoch that achieved best score
        """
        # Compute raw score
        score_raw = self.compute_score(val_metrics)

        # Update EMA
        if self.score_ema is None:
            # First epoch: initialize EMA to raw score
            self.score_ema = score_raw
        else:
            self.score_ema = (
                self.ema_alpha * self.score_ema + (1 - self.ema_alpha) * score_raw
            )

        # Determine which score to use for early stopping
        score_for_stopping = self.score_ema if self.use_ema_for_stopping else score_raw

        # Check if improved
        is_best = score_for_stopping > (self.best_score + self.min_delta)

        if is_best:
            self.best_score = score_for_stopping
            self.best_ema_score = self.score_ema
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Check if should stop
        should_stop = self.epochs_without_improvement >= self.patience

        # Store history for analysis
        self._history.append({
            "epoch": epoch,
            "score_raw": score_raw,
            "score_ema": self.score_ema,
            "is_best": is_best,
            "epochs_without_improvement": self.epochs_without_improvement,
        })

        return {
            "val_score_raw": score_raw,
            "val_score_ema": self.score_ema,
            "is_best": is_best,
            "should_stop": should_stop,
            "epochs_without_improvement": self.epochs_without_improvement,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop_flag": 1.0 if should_stop else 0.0,
        }

    @classmethod
    def from_config(cls, cfg: Any) -> "EarlyStoppingTracker":
        """
        Create an EarlyStoppingTracker from a config object.

        Handles both old-style configs (tcfg.patience) and new-style configs
        (tcfg.early_stopping.patience).
        """
        # Check for new-style config
        if hasattr(cfg, 'early_stopping') and cfg.early_stopping is not None:
            es_cfg = cfg.early_stopping
            return cls(
                patience=getattr(es_cfg, 'patience', 5),
                psnr_weight=getattr(es_cfg, 'psnr_weight', 1.0),
                ssim_weight=getattr(es_cfg, 'ssim_weight', 1.0),
                ema_alpha=getattr(es_cfg, 'ema_alpha', 0.7),
                use_ema_for_stopping=getattr(es_cfg, 'use_ema_for_stopping', True),
                min_delta=getattr(es_cfg, 'min_delta', 0.001),
                metric=getattr(es_cfg, 'metric', 'psnr_ssim_composite'),
            )
        else:
            # Fallback to old-style config
            return cls(
                patience=getattr(cfg, 'patience', 5),
                psnr_weight=1.0,
                ssim_weight=50.0,  # Match old behavior: PSNR + 50*SSIM
                ema_alpha=0.0,  # No smoothing in old behavior
                use_ema_for_stopping=False,
                min_delta=0.0,
                metric='psnr_ssim_composite',
            )


# =============================================================================
# OPTIMIZER FACTORY
# =============================================================================

def create_optimizer(
    model_parameters,
    optimizer_cfg: Any,
) -> optim.Optimizer:
    """
    Create an optimizer based on configuration.

    Args:
        model_parameters: Model parameters to optimize
        optimizer_cfg: Configuration object with optimizer parameters

    Returns:
        Configured optimizer instance

    Supported optimizer types:
        - adamw: AdamW (default) - Adam with decoupled weight decay
        - adam: Adam - standard Adam optimizer
        - sgd: SGD - stochastic gradient descent with optional momentum
        - rmsprop: RMSprop - adaptive learning rate method
    """
    opt_type = getattr(optimizer_cfg, 'type', 'adamw').lower()
    lr = optimizer_cfg.lr
    wd = optimizer_cfg.wd

    if opt_type == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=lr,
            betas=optimizer_cfg.betas,
            eps=optimizer_cfg.eps,
            weight_decay=wd,
            amsgrad=getattr(optimizer_cfg, 'amsgrad', False),
        )
    elif opt_type == "adam":
        return optim.Adam(
            model_parameters,
            lr=lr,
            betas=optimizer_cfg.betas,
            eps=optimizer_cfg.eps,
            weight_decay=wd,
            amsgrad=getattr(optimizer_cfg, 'amsgrad', False),
        )
    elif opt_type == "sgd":
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=getattr(optimizer_cfg, 'momentum', 0.9),
            weight_decay=wd,
            nesterov=getattr(optimizer_cfg, 'nesterov', False),
        )
    elif opt_type == "rmsprop":
        return optim.RMSprop(
            model_parameters,
            lr=lr,
            alpha=getattr(optimizer_cfg, 'alpha', 0.99),
            eps=optimizer_cfg.eps,
            weight_decay=wd,
            momentum=getattr(optimizer_cfg, 'momentum', 0.0),
            centered=getattr(optimizer_cfg, 'centered', False),
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: '{opt_type}'. "
            f"Supported types: adamw, adam, sgd, rmsprop"
        )


# =============================================================================
# LEARNING RATE SCHEDULER FACTORY
# =============================================================================

def create_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_cfg: Any,
    steps_per_epoch: int,
    total_epochs: int,
    base_lr: float,
) -> tuple:
    """
    Create a learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to schedule
        scheduler_cfg: Configuration object with scheduler parameters
        steps_per_epoch: Number of training steps per epoch
        total_epochs: Total number of training epochs
        base_lr: Base learning rate from optimizer config

    Returns:
        Tuple of (scheduler, step_per_batch: bool)
        step_per_batch indicates whether to step the scheduler after each batch (True)
        or after each epoch (False).

    Supported scheduler types:
        - "onecycle": One-Cycle LR (steps per batch)
        - "cosine": Cosine annealing (steps per epoch)
        - "cosine_warmup": Cosine with linear warmup (steps per batch)
        - "constant": No scheduling
        - "step": Step decay (steps per epoch)
    """
    total_steps = steps_per_epoch * total_epochs
    scheduler_type = getattr(scheduler_cfg, 'type', 'constant').lower()

    if scheduler_type == "onecycle":
        max_lr = getattr(scheduler_cfg, 'max_lr', base_lr * 2)
        pct_start = getattr(scheduler_cfg, 'pct_start', 0.3)
        div_factor = getattr(scheduler_cfg, 'div_factor', 25.0)
        final_div_factor = getattr(scheduler_cfg, 'final_div_factor', 1000.0)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        logger.info(f"Using OneCycleLR scheduler: max_lr={max_lr}, pct_start={pct_start}, "
                   f"total_steps={total_steps}")
        return scheduler, True  # Step per batch

    elif scheduler_type == "cosine":
        T_max = getattr(scheduler_cfg, 'T_max', None) or total_epochs
        eta_min = getattr(scheduler_cfg, 'eta_min', 1e-7)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
        logger.info(f"Using CosineAnnealingLR scheduler: T_max={T_max}, eta_min={eta_min}")
        return scheduler, False  # Step per epoch

    elif scheduler_type == "cosine_warmup":
        warmup_epochs = getattr(scheduler_cfg, 'warmup_epochs', 5)
        eta_min = getattr(scheduler_cfg, 'eta_min', 1e-7)
        warmup_steps = warmup_epochs * steps_per_epoch

        # Use SequentialLR with linear warmup + cosine decay
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=eta_min,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
        logger.info(f"Using Cosine with Warmup scheduler: warmup_epochs={warmup_epochs}, "
                   f"eta_min={eta_min}")
        return scheduler, True  # Step per batch

    elif scheduler_type == "step":
        step_size = getattr(scheduler_cfg, 'step_size', 30)
        gamma = getattr(scheduler_cfg, 'gamma', 0.1)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        logger.info(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
        return scheduler, False  # Step per epoch

    elif scheduler_type in ("constant", "none"):
        logger.info("Using constant learning rate (no scheduler)")
        return None, False

    else:
        logger.warning(f"Unknown scheduler type '{scheduler_type}', using constant LR")
        return None, False


def save_class_embeddings_trajectory(
    model: nn.Module,
    epoch: int,
    output_path: Path,
) -> None:
    """
    Append current epoch's class-embedding matrix into a single .pt trajectory file.

    The function expects `model` to have `class_embed.emb.weight`.
    It accumulates snapshots over epochs in a single file with:
      - "epochs": list of epoch numbers
      - "embeddings": tensor [E, num_classes, emb_dim]
      - "num_classes": int
      - "emb_dim": int

    Args:
        model: The model containing class_embed.emb.weight
        epoch: Current epoch number
        output_path: Path to save the trajectory file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Current embedding matrix [num_classes, emb_dim]
    emb_weight = model.class_embed.emb.weight.detach().cpu().clone()

    if output_path.exists():
        # Load existing trajectory and append
        state = torch.load(output_path, map_location="cpu")
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


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "2h 34m 12s" or "45m 23s" or "12s"
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {mins}m {secs}s"


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


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        # Don't filter by requires_grad - state_dict() tensors are always detached
        # We want all parameters, not just trainable ones (they're all trainable anyway)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)


def sync_metrics_dict(metrics_dict: Dict[str, float], device: torch.device, use_ddp: bool) -> Dict[str, float]:
    """
    Synchronize a dictionary of metrics across all DDP processes by averaging.

    Args:
        metrics_dict: Dictionary of metric name -> value
        device: Device tensors are on
        use_ddp: Whether DDP is enabled

    Returns:
        Dictionary with globally averaged metrics
    """
    if not use_ddp:
        return metrics_dict

    synced_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            # Convert to tensor and synchronize
            tensor = torch.tensor([float(value)], device=device, dtype=torch.float32)
            synced_tensor = all_reduce_mean(tensor)
            synced_metrics[key] = synced_tensor.item()
        else:
            # Non-numeric values (shouldn't happen for metrics, but just in case)
            synced_metrics[key] = value

    return synced_metrics


@torch.no_grad()
def compute_training_diagnostics(
    model: nn.Module,
    x0_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    num_samples: int = 16,
) -> Dict[str, float]:
    """
    Compute comprehensive diagnostic metrics to detect training issues.

    These diagnostics are logged to the CSV with split="diag" to enable
    correlation of failure modes or instabilities with epoch and configuration.

    Returns:
        Dictionary with diagnostic metrics:

        Noise-Prediction Correlation (correctly interpreted):
        - noise_pred_corr: Mean correlation between predicted and true noise
        - noise_pred_corr_t100: Correlation at t=100 (high noise level)
        - noise_pred_corr_t500: Correlation at t=500 (medium noise level)
          * High correlation (>0.5) = model is learning noise pattern correctly
          * Low/negative = potential training issues

        Prediction Statistics:
        - pred_std: Mean std of model predictions
        - pred_std_t100: Std at t=100
        - pred_std_t500: Std at t=500
          * Should be ~0.8-1.2 for normalized data
          * Much lower = model collapsed to constant
          * Much higher = unstable predictions

        Single-Step Reconstruction (x0 estimated from single denoising step):
        - recon_mse_t100, recon_mse_t500: MSE of x0 reconstruction
        - recon_psnr_t100, recon_psnr_t500: PSNR of x0 reconstruction
        - recon_ssim_t100, recon_ssim_t500: SSIM of x0 reconstruction

        Legacy fields (for backwards compatibility):
        - input_output_correlation, reconstruction_mse_t100/t500, etc.
    """
    model.eval()

    # Take subset of batch
    x0 = x0_batch[:num_samples]
    labels = labels_batch[:num_samples]

    # Per-timestep metrics storage
    metrics_per_t: Dict[int, Dict[str, float]] = {}

    for t_val in [100, 500]:
        t = torch.full((len(x0),), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Model prediction
        eps_pred = model(x_t, t, labels)

        # ================================================================
        # Noise-Prediction Correlation (correctly interpreted)
        # ================================================================
        # We correlate predicted noise (eps_pred) with true noise (eps_true).
        # A well-trained model should have HIGH positive correlation here,
        # indicating it correctly predicts the noise that was added.
        eps_true = noise
        eps_pred_flat = eps_pred.flatten()
        eps_true_flat = eps_true.flatten()
        eps_pred_std = eps_pred_flat.std()
        eps_true_std = eps_true_flat.std()

        if eps_pred_std > 1e-8 and eps_true_std > 1e-8:
            corr = torch.corrcoef(torch.stack([eps_pred_flat, eps_true_flat]))[0, 1].item()
        else:
            corr = 0.0

        # ================================================================
        # Prediction Statistics
        # ================================================================
        pred_std = eps_pred.std().item()
        pred_mean = eps_pred.mean().item()

        # ================================================================
        # Single-Step x0 Reconstruction
        # ================================================================
        # Use the DDPM posterior mean formula to estimate x0 from x_t and eps_pred
        sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x0_pred = (x_t - sqrt_one_minus_alpha_prod * eps_pred) / sqrt_alpha_prod
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # MSE
        mse = F.mse_loss(x0_pred, x0).item()

        # PSNR
        psnr = compute_psnr(x0_pred, x0)

        # SSIM (convert from [-1, 1] to [0, 1] range for SSIM computation)
        x0_pred_01 = (x0_pred + 1.0) / 2.0
        x0_01 = (x0 + 1.0) / 2.0
        ssim = compute_ssim(x0_pred_01, x0_01, max_val=1.0)

        # Store metrics for this timestep
        metrics_per_t[t_val] = {
            "corr": corr,
            "pred_std": pred_std,
            "pred_mean": pred_mean,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
        }

    model.train()

    # ================================================================
    # Build output dictionary with all diagnostic fields
    # ================================================================
    t100 = metrics_per_t[100]
    t500 = metrics_per_t[500]

    # Average correlation across timesteps
    mean_corr = (t100["corr"] + t500["corr"]) / 2.0
    mean_pred_std = (t100["pred_std"] + t500["pred_std"]) / 2.0

    result = {
        # ---- New diagnostic field names ----
        # Noise-prediction correlation (correctly interpreted: high = good)
        "noise_pred_corr": float(mean_corr),
        "noise_pred_corr_t100": float(t100["corr"]),
        "noise_pred_corr_t500": float(t500["corr"]),

        # Prediction statistics
        "pred_std": float(mean_pred_std),
        "pred_std_t100": float(t100["pred_std"]),
        "pred_std_t500": float(t500["pred_std"]),

        # Single-step reconstruction metrics
        "recon_mse_t100": float(t100["mse"]),
        "recon_mse_t500": float(t500["mse"]),
        "recon_psnr_t100": float(t100["psnr"]),
        "recon_psnr_t500": float(t500["psnr"]),
        "recon_ssim_t100": float(t100["ssim"]),
        "recon_ssim_t500": float(t500["ssim"]),

        # ---- Legacy field names (for backwards compatibility) ----
        "input_output_correlation": float(mean_corr),
        "reconstruction_mse_t100": float(t100["mse"]),
        "reconstruction_mse_t500": float(t500["mse"]),
        "reconstruction_psnr_t500": float(t500["psnr"]),
        "prediction_std": float(mean_pred_std),
    }

    return result

@torch.no_grad()
def full_chain_reconstruction_metrics(model, scheduler, x0, y, device) -> Dict[str, float]:
    """
    Full-chain reconstruction test: add noise at random t, sample back to t=0, compute metrics.
    This catches multi-step drift that single-step x̂₀ formulas miss.

    Args:
        model: DDPM model
        scheduler: DDPM scheduler
        x0: Clean image [N, C, H, W]
        y: Class labels [N]
        device: Device

    Returns:
        Dictionary with:
        - psnr: PSNR of reconstructed image
        - ssim: SSIM of reconstructed image
    """
    model.eval()
    x0 = x0[:1].to(device)
    y = y[:1].to(device)
    T = scheduler.config.num_train_timesteps
    t = torch.randint(T // 4, 3 * T // 4, (1,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, t)

    # Run reverse from current t to 0
    scheduler.set_timesteps(T)
    # Find index of closest scheduler timestep to t
    start_idx = int((scheduler.timesteps - t.cpu()).abs().argmin().item())
    # Move scheduler timesteps to device to avoid device mismatch in scheduler.step()
    scheduler.timesteps = scheduler.timesteps.to(device)

    x = x_t
    for i in range(start_idx, len(scheduler.timesteps)):
        tt = scheduler.timesteps[i].unsqueeze(0)
        eps = model(x, tt, y)
        x = scheduler.step(eps, tt, x).prev_sample

    x_rec = torch.clamp(x, -1, 1)

    # Convert to [0, 1] range for metrics
    x_rec_01 = (x_rec + 1) / 2
    x0_01 = (x0 + 1) / 2

    psnr = compute_psnr(x_rec_01, x0_01, max_val=1.0)
    ssim = compute_ssim(x_rec_01, x0_01, max_val=1.0)

    return {
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


# Legacy function for backwards compatibility
@torch.no_grad()
def full_chain_reconstruction_psnr(model, scheduler, x0, y, device) -> float:
    """
    Legacy function that returns only PSNR.
    Use full_chain_reconstruction_metrics() for both PSNR and SSIM.
    """
    metrics = full_chain_reconstruction_metrics(model, scheduler, x0, y, device)
    return metrics["psnr"]


@torch.no_grad()
def compute_elbo_diagnostics(
    model: nn.Module,
    x0_batch: torch.Tensor,
    labels_batch: torch.Tensor,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    num_samples: int = 32,
    num_timesteps_per_sample: int = 8,
    use_min_snr: bool = False,
    min_snr_gamma: float = 5.0,
) -> Dict[str, float]:
    """
    Compute ELBO-related diagnostic metrics using estimate_elbo_terms.

    This function samples random timesteps across the batch and computes:
    - L_simple: Unweighted ε-MSE (standard training loss before any weighting)
    - L_weighted: ε-MSE weighted by Min-SNR factor (matches training loss when enabled)
    - SNR: Signal-to-noise ratio at each timestep
    - min_snr_weight: The Min-SNR weight applied (for analysis)

    These metrics help analyze whether Min-SNR weighting is aligned with the
    parts of the loss that matter. By binning results by timestep range, you
    can see which regions dominate the loss and how weighting affects them.

    When use_min_snr=False, L_weighted == L_simple and weight_ratio ≈ 1.0.
    When use_min_snr=True, L_weighted < L_simple at high SNR timesteps.

    Args:
        model: DDPM model
        x0_batch: Clean images [B, C, H, W]
        labels_batch: Class labels [B]
        noise_scheduler: DDPM scheduler with alphas_cumprod and betas
        device: Compute device
        num_samples: Number of samples from batch to use
        num_timesteps_per_sample: Number of random timesteps to sample per image
        use_min_snr: If True, compute L_weighted with Min-SNR weighting
        min_snr_gamma: Clipping parameter γ for Min-SNR (typically 3–5)

    Returns:
        Dictionary with ELBO diagnostic metrics:
        - elbo_L_simple_mean: Mean L_simple across all samples
        - elbo_L_weighted_mean: Mean L_weighted across all samples
        - elbo_snr_mean: Mean SNR across all samples
        - elbo_L_simple_low_t, elbo_L_weighted_low_t, elbo_snr_low_t: Metrics for t < 333
        - elbo_L_simple_mid_t, elbo_L_weighted_mid_t, elbo_snr_mid_t: Metrics for 333 <= t < 666
        - elbo_L_simple_high_t, elbo_L_weighted_high_t, elbo_snr_high_t: Metrics for t >= 666
        - elbo_weight_ratio_*: Ratio of L_weighted/L_simple per timestep bin
        - elbo_min_snr_weight_mean: Mean of Min-SNR weights (1.0 when disabled)
    """
    model.eval()

    # Prepare scheduler config for estimate_elbo_terms
    T = noise_scheduler.config.num_train_timesteps
    scheduler_cfg = {
        "alphas_cumprod": noise_scheduler.alphas_cumprod,
    }

    # Take subset of batch
    x0 = x0_batch[:num_samples].to(device)
    labels = labels_batch[:num_samples].to(device)
    n = x0.size(0)

    # Accumulators for overall and binned statistics
    all_L_simple = []
    all_L_weighted = []
    all_snr = []
    all_min_snr_weight = []
    all_t = []

    # Sample multiple random timesteps per image to get coverage across t
    for _ in range(num_timesteps_per_sample):
        # Random timesteps uniformly from [0, T-1]
        t = torch.randint(0, T, (n,), device=device, dtype=torch.long)

        # Add noise
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Model prediction
        eps_pred = model(x_t, t, labels)

        # Compute ELBO terms with Min-SNR settings matching training
        elbo_result = estimate_elbo_terms(
            x0=x0,
            x_t=x_t,
            t=t,
            eps_pred=eps_pred,
            scheduler_cfg=scheduler_cfg,
            use_min_snr=use_min_snr,
            min_snr_gamma=min_snr_gamma,
        )

        # Accumulate results
        all_L_simple.append(elbo_result["L_simple"])
        all_L_weighted.append(elbo_result["L_weighted"])
        all_snr.append(elbo_result["snr"])
        all_min_snr_weight.append(elbo_result["min_snr_weight"])
        all_t.append(elbo_result["t"])

    # Concatenate all results
    all_L_simple = torch.cat(all_L_simple, dim=0)
    all_L_weighted = torch.cat(all_L_weighted, dim=0)
    all_snr = torch.cat(all_snr, dim=0)
    all_min_snr_weight = torch.cat(all_min_snr_weight, dim=0)
    all_t = torch.cat(all_t, dim=0)

    # Overall means
    elbo_L_simple_mean = all_L_simple.mean().item()
    elbo_L_weighted_mean = all_L_weighted.mean().item()
    elbo_snr_mean = all_snr.mean().item()
    elbo_min_snr_weight_mean = all_min_snr_weight.mean().item()

    # Bin by timestep: low (t < T/3), mid (T/3 <= t < 2T/3), high (t >= 2T/3)
    t_low_threshold = T // 3      # 333 for T=1000
    t_high_threshold = 2 * T // 3  # 666 for T=1000

    mask_low = all_t < t_low_threshold
    mask_mid = (all_t >= t_low_threshold) & (all_t < t_high_threshold)
    mask_high = all_t >= t_high_threshold

    def safe_mean(tensor, mask):
        """Compute mean only over masked elements, return NaN if no elements."""
        if mask.sum() == 0:
            return float("nan")
        return tensor[mask].mean().item()

    def safe_ratio(num, denom):
        """Compute ratio, avoiding division by zero."""
        if abs(denom) < 1e-10:
            return float("nan")
        return num / denom

    # Compute binned statistics
    L_simple_low = safe_mean(all_L_simple, mask_low)
    L_simple_mid = safe_mean(all_L_simple, mask_mid)
    L_simple_high = safe_mean(all_L_simple, mask_high)

    L_weighted_low = safe_mean(all_L_weighted, mask_low)
    L_weighted_mid = safe_mean(all_L_weighted, mask_mid)
    L_weighted_high = safe_mean(all_L_weighted, mask_high)

    snr_low = safe_mean(all_snr, mask_low)
    snr_mid = safe_mean(all_snr, mask_mid)
    snr_high = safe_mean(all_snr, mask_high)

    # Min-SNR weights per region (should be ~1.0 when disabled)
    min_snr_weight_low = safe_mean(all_min_snr_weight, mask_low)
    min_snr_weight_mid = safe_mean(all_min_snr_weight, mask_mid)
    min_snr_weight_high = safe_mean(all_min_snr_weight, mask_high)

    # Weight ratios show how much the Min-SNR weighting affects each region
    # When use_min_snr=False, these should all be ~1.0
    weight_ratio_low = safe_ratio(L_weighted_low, L_simple_low)
    weight_ratio_mid = safe_ratio(L_weighted_mid, L_simple_mid)
    weight_ratio_high = safe_ratio(L_weighted_high, L_simple_high)

    model.train()

    return {
        # Overall means
        "elbo_L_simple_mean": float(elbo_L_simple_mean),
        "elbo_L_weighted_mean": float(elbo_L_weighted_mean),
        "elbo_snr_mean": float(elbo_snr_mean),
        "elbo_min_snr_weight_mean": float(elbo_min_snr_weight_mean),

        # Low timesteps (t < T/3): low noise, high SNR
        "elbo_L_simple_low_t": float(L_simple_low),
        "elbo_L_weighted_low_t": float(L_weighted_low),
        "elbo_snr_low_t": float(snr_low),
        "elbo_min_snr_weight_low_t": float(min_snr_weight_low),

        # Mid timesteps (T/3 <= t < 2T/3)
        "elbo_L_simple_mid_t": float(L_simple_mid),
        "elbo_L_weighted_mid_t": float(L_weighted_mid),
        "elbo_snr_mid_t": float(snr_mid),
        "elbo_min_snr_weight_mid_t": float(min_snr_weight_mid),

        # High timesteps (t >= 2T/3): high noise, low SNR
        "elbo_L_simple_high_t": float(L_simple_high),
        "elbo_L_weighted_high_t": float(L_weighted_high),
        "elbo_snr_high_t": float(snr_high),
        "elbo_min_snr_weight_high_t": float(min_snr_weight_high),

        # Weight ratios (L_weighted / L_simple)
        "elbo_weight_ratio_low_t": float(weight_ratio_low),
        "elbo_weight_ratio_mid_t": float(weight_ratio_mid),
        "elbo_weight_ratio_high_t": float(weight_ratio_high),
    }


@torch.no_grad()
def visualize_noising_process(
    x0: torch.Tensor,
    scheduler: DDPMScheduler,
    num_steps: int = 10,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Visualize the forward noising process at evenly spaced timesteps.
    
    Args:
        x0: Clean image [1, C, H, W]
        scheduler: DDPM scheduler
        num_steps: Number of intermediate steps to visualize
        device: Device
    
    Returns:
        Tensor of shape [num_steps+1, C, H, W] showing progressive noising
    """
    x0 = x0.to(device)
    total_timesteps = scheduler.config.num_train_timesteps
    timesteps = torch.linspace(0, total_timesteps - 1, num_steps, dtype=torch.long, device=device)
    
    noised_images = [x0.cpu()]
    noise = torch.randn_like(x0)
    
    for t in timesteps:
        t_batch = t.unsqueeze(0)
        x_t = scheduler.add_noise(x0, noise, t_batch)
        noised_images.append(x_t.cpu())
    
    return torch.cat(noised_images, dim=0)

@torch.no_grad()
def visualize_denoising_process(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple,
    class_label: torch.Tensor,
    num_steps: int = 10,
    device: torch.device = torch.device("cuda"),
    guidance_scale: float = 1.0
) -> torch.Tensor:
    """
    Visualize the reverse denoising process from pure noise to clean image.

    Args:
        model: DDPM model
        scheduler: DDPM scheduler
        shape: Image shape (C, H, W)
        class_label: Class label for conditional generation
        num_steps: Number of intermediate steps to visualize (>= 1)
        device: Device
        guidance_scale: Classifier-free guidance scale

    Returns:
        Tensor showing progressive denoising: [initial_noise, *intermediate_steps, final_image]
        Shape: [num_steps+2, C, H, W] with initial noise + num_steps + final frame
    """
    model.eval()

    # Start from pure noise
    x_t = torch.randn((1, *shape), device=device)
    class_label = class_label.to(device)

    total_timesteps = scheduler.config.num_train_timesteps
    scheduler.set_timesteps(total_timesteps)
    # Move scheduler timesteps to device to avoid device mismatch in scheduler.step()
    scheduler.timesteps = scheduler.timesteps.to(device)

    # Ensure at least 1 intermediate step to maintain schedule compatibility
    save_indices = set(torch.linspace(
        0, len(scheduler.timesteps) - 1,
        max(1, num_steps),  # ensure >= 1 for proper step scheduling
        dtype=torch.long
    ).tolist())

    frames = [x_t.cpu()]  # keep initial noise for context

    for i, t in enumerate(scheduler.timesteps):
        # Predict noise
        t_batch = t.unsqueeze(0)

        if guidance_scale == 1.0:
            noise_pred = model(x_t, t_batch, class_label)
        else:
            # Classifier-free guidance
            noise_pred_cond = model(x_t, t_batch, class_label)
            noise_pred_uncond = model(x_t, t_batch, None)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Denoise one step
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample

        if i in save_indices:
            frames.append(x_t.detach().cpu())

    # Safeguard: ensure we always have the final denoised image
    if len(frames) == 1 or frames[-1] is not x_t.cpu():
        frames.append(x_t.detach().cpu())

    return torch.cat(frames, dim=0)

@torch.no_grad()
def visualize_multistep_reconstruction(
    model: nn.Module,
    x0: torch.Tensor,
    scheduler: DDPMScheduler,
    class_label: torch.Tensor,
    timesteps: list[int],
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Show x0 reconstruction quality at different timesteps.
    
    Args:
        model: DDPM model
        x0: Clean image [1, C, H, W]
        scheduler: DDPM scheduler
        class_label: Class label
        timesteps: List of timesteps to visualize
        device: Device
    
    Returns:
        Tensor [len(timesteps)+1, C, H, W] with original + reconstructions
    """
    model.eval()
    x0 = x0.to(device)
    class_label = class_label.to(device)
    
    reconstructions = [x0.cpu()]
    noise = torch.randn_like(x0)
    
    for t_val in timesteps:
        t = torch.tensor([t_val], dtype=torch.long, device=device)
        
        # Add noise
        x_t = scheduler.add_noise(x0, noise, t)
        
        # Predict noise
        noise_pred = model(x_t, t, class_label)
        
        # Reconstruct x0
        sqrt_alpha_prod = scheduler.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha_prod = (1 - scheduler.alphas_cumprod[t]).sqrt()
        x0_pred = (x_t - sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1) * noise_pred) / sqrt_alpha_prod.view(-1, 1, 1, 1)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        reconstructions.append(x0_pred.cpu())
    
    return torch.cat(reconstructions, dim=0)

def _maybe_drop_labels(labels: torch.Tensor, p: float) -> Optional[torch.Tensor]:
    if p <= 0:
        return labels
    mask = torch.rand_like(labels.float()) < p
    out = labels.clone()
    out[mask] = 0  # value unused
    # Return None for unconditional; model handles None as zeros
    return None if mask.all() else out

@torch.no_grad()
def conditioning_sanity_check(
    model: nn.Module,
    scheduler: DDPMScheduler,
    num_classes: int,
    image_shape: tuple,
    device: torch.device,
    num_samples: int = 3
) -> Dict[str, float]:
    """
    Verify that class conditioning is working by checking that different labels
    produce different noise predictions for the same input.

    Args:
        model: DDPM model
        scheduler: DDPM scheduler
        num_classes: Number of classes
        image_shape: (C, H, W)
        device: Device
        num_samples: Number of random samples to test

    Returns:
        Dictionary with conditioning gap statistics
    """
    model.eval()
    gaps = []

    for _ in range(num_samples):
        # Sample fixed noise and random timestep
        x_t = torch.randn((1, *image_shape), device=device)
        t = torch.randint(100, scheduler.config.num_train_timesteps // 2, (1,), device=device)

        # Get predictions for two different classes
        class_0 = torch.tensor([0], device=device)
        class_1 = torch.tensor([min(1, num_classes - 1)], device=device)

        eps_0 = model(x_t, t, class_0)
        eps_1 = model(x_t, t, class_1)

        # Compute L2 distance between predictions
        gap = torch.norm(eps_0 - eps_1, p=2).item()
        gaps.append(gap)

    return {
        "conditioning_gap_mean": float(torch.tensor(gaps).mean()),
        "conditioning_gap_std": float(torch.tensor(gaps).std()),
        "conditioning_gap_min": float(torch.tensor(gaps).min()),
        "conditioning_gap_max": float(torch.tensor(gaps).max()),
    }

def compute_batch_metrics(
    pred_noise: torch.Tensor,
    true_noise: torch.Tensor,
    x0_reconstructed: torch.Tensor,
    x0_original: torch.Tensor,
    loss: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for a single batch during training/validation.
    
    Args:
        pred_noise: Predicted noise from model
        true_noise: Ground truth noise
        x0_reconstructed: Reconstructed x0 from predicted noise
        x0_original: Original clean images
        loss: Total loss value
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "loss": float(loss.detach().cpu()),
        "noise_mse": float(F.mse_loss(pred_noise, true_noise).detach().cpu()),
        "noise_mae": float(F.l1_loss(pred_noise, true_noise).detach().cpu()),
    }
    
    # Compute PSNR and SSIM on reconstructed x0
    # Assuming images are in [-1, 1] range, convert to [0, 1] for metrics
    x0_recon_01 = (x0_reconstructed.detach() + 1.0) / 2.0
    x0_orig_01 = (x0_original.detach() + 1.0) / 2.0
    x0_recon_01 = torch.clamp(x0_recon_01, 0.0, 1.0)
    x0_orig_01 = torch.clamp(x0_orig_01, 0.0, 1.0)
    
    metrics["psnr"] = compute_psnr(x0_recon_01, x0_orig_01, max_val=1.0)
    metrics["ssim"] = compute_ssim(x0_recon_01, x0_orig_01, max_val=1.0)

    # Guard against non-finite PSNR/SSIM values (e.g., inf from zero MSE)
    for key in ["psnr", "ssim"]:
        if key in metrics and not math.isfinite(float(metrics[key])):
            metrics[key] = 0.0

    return metrics

@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    split_name: str,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    scfg: Any,
    tcfg: Any,
    use_ddp: bool,
    rank: int
) -> Dict[str, float]:
    """
    Evaluate model on a given split (val or test).

    Args:
        model: The model to evaluate
        dataloader: DataLoader for the split
        split_name: Name of the split ("val" or "test")
        noise_scheduler: DDPM scheduler
        device: Device to run on
        scfg: Scheduler config
        tcfg: Training config
        use_ddp: Whether DDP is enabled
        rank: Process rank

    Returns:
        Dictionary of metrics (loss, psnr, ssim, per-class losses, etc.)
    """
    model.eval()
    num_classes = getattr(tcfg, 'num_classes', NUM_CLASSES)
    split_avg = EpochAverager(num_classes=num_classes)
    per_class_acc = PerClassMetricsAccumulator(num_classes=num_classes)
    split_loss_fn = DDPMNoiseMSE(
        num_classes=num_classes,
        use_min_snr=tcfg.use_min_snr,
        min_snr_gamma=tcfg.min_snr_gamma
    )

    total_steps = len(dataloader)
    use_progress_bar = tcfg.use_tqdm and is_main_process() and not IS_SUPERCOMPUTER
    n_bad_batches = 0  # Track batches with non-finite loss

    # Progress bar (only on rank-0 if using tqdm AND not supercomputer)
    if use_progress_bar:
        pbar = tqdm(dataloader, desc=f"{split_name.capitalize()} Eval", unit="batch", leave=False)
    else:
        pbar = dataloader

    for step, batch in enumerate(pbar, 1):
        x0 = batch["pixel_values"].to(device)
        logger.debug(f"Pixel range ({split_name}): min={x0.min().item():.3f}, max={x0.max().item():.3f}")
        labels = batch["labels"].to(device)
        bsz = x0.size(0)

        # Sample t and noise
        t = torch.randint(0, scfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Forward with mixed precision (for model efficiency)
        with torch.autocast(device_type=device.type, enabled=tcfg.mixed_precision):
            pred = model(x_t, t, labels)

        # Compute loss in full precision (float32) for stability
        pred_fp32 = pred.float()
        noise_fp32 = noise.float()
        loss = split_loss_fn(
            pred_fp32, noise_fp32, labels,
            timesteps=t,
            alphas_cumprod=noise_scheduler.alphas_cumprod
        )

        # Guard: skip batches with non-finite loss
        if not torch.isfinite(loss):
            logger.warning(
                f"[{split_name}] Non-finite loss at batch_idx={step}; "
                f"skipping batch from metrics."
            )
            n_bad_batches += 1
            continue

        # Reconstruct x0 for metrics
        sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t]).sqrt()

        # Add epsilon to prevent division by zero
        sqrt_alpha_prod = torch.clamp(sqrt_alpha_prod, min=1e-6)
        x0_pred = (x_t - sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1) * pred) / sqrt_alpha_prod.view(-1, 1, 1, 1)

        # Clamp to prevent extreme values
        x0_pred = torch.clamp(x0_pred, -10.0, 10.0)

        # Compute global metrics
        batch_metrics = compute_batch_metrics(pred, noise, x0_pred, x0, loss)
        batch_metrics["grad_norm"] = 0.0  # No gradients in evaluation
        batch_metrics["ema_enabled"] = 1.0 if hasattr(model, '_ema') else 0.0
        split_avg.update(batch_metrics, batch_size=bsz)

        # Compute per-class PSNR/SSIM metrics
        # Rescale x0 and x0_pred from [-1, 1] to [0, 1] for metrics computation
        x0_01 = (x0 + 1.0) / 2.0
        x0_pred_01 = (x0_pred.clamp(-1, 1) + 1.0) / 2.0
        per_class_acc.update(
            x_hat=x0_pred_01,
            x=x0_01,
            labels=labels,
            max_val=1.0,
        )

        # Update progress bar (only if tqdm enabled on rank-0)
        if use_progress_bar:
            pbar.set_postfix({f"{split_name}_loss": f"{loss.item():.4f}"})

        # Supercomputer mode: periodic logging (3 times per evaluation)
        if IS_SUPERCOMPUTER and should_log_step(step, total_steps, log_frequency=3):
            log_validation_progress(
                rank=rank,
                step=step,
                total_steps=total_steps,
                loss=loss.item(),
                main_process_only=True
            )

    # Close progress bar if used
    if use_progress_bar:
        pbar.close()

    # Get per-class losses (local to this GPU)
    per_class_losses_weighted = split_loss_fn.per_class(weighted=True)
    per_class_losses_raw = split_loss_fn.per_class(weighted=False)
    metrics = split_avg.means()

    # Add per-class losses to metrics
    for c, loss_c in per_class_losses_weighted.items():
        metrics[f"loss_weighted_c{c}"] = loss_c
    for c, loss_c in per_class_losses_raw.items():
        metrics[f"loss_raw_c{c}"] = loss_c

    # Add per-class PSNR/SSIM metrics
    per_class_metrics = per_class_acc.compute()
    for key, value in per_class_metrics.items():
        # Skip global psnr/ssim since they're already in metrics
        if key in ("psnr", "ssim"):
            continue
        metrics[key] = value

    # Add number of bad batches to metrics (for monitoring numerical stability)
    metrics["n_bad_batches"] = float(n_bad_batches)

    # CRITICAL: Synchronize metrics across all GPUs
    metrics = sync_metrics_dict(metrics, device, use_ddp)

    # Log warning if validation had many bad batches (indicates training divergence)
    if metrics["n_bad_batches"] > 0:
        logger.warning(
            f"[{split_name}] Encountered {metrics['n_bad_batches']:.0f} batches with non-finite loss. "
            "This indicates potential training divergence."
        )

    return metrics


@torch.no_grad()
def compute_fid_on_split(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    noise_scheduler: DDPMScheduler,
    device: torch.device,
    tcfg: Any,
    scfg: Any,
    fid_config: Dict[str, Any],
    rank: int,
) -> Dict[str, float]:
    """
    Compute FID metrics for a given split (val or test).

    Generates samples from the model and compares against real data from the dataloader.

    Args:
        model: The trained model
        dataloader: DataLoader for the split (to get real images)
        noise_scheduler: DDPM scheduler for generation
        device: Device to run on
        tcfg: Training config
        scfg: Scheduler config
        fid_config: FID computation configuration (samples_per_class, batch_size, min_samples)
        rank: Process rank

    Returns:
        Dictionary with fid_global and fid_c0, fid_c1, ..., fid_c{num_classes-1}
    """
    num_classes = getattr(tcfg, 'num_classes', NUM_CLASSES)
    samples_per_class = fid_config.get("samples_per_class", 100)
    batch_size = fid_config.get("batch_size", 32)
    min_samples = fid_config.get("min_samples", 50)

    # Initialize FID computer
    fid_computer = TorchmetricsFIDComputer(
        num_classes=num_classes,
        device=str(device),
        normalize=True,
    )

    if not fid_computer.available:
        logger.warning("FID computation not available (torchmetrics.image.fid not installed)")
        result = {"fid_global": float("nan")}
        result.update({f"fid_c{k}": float("nan") for k in range(num_classes)})
        return result

    # Get inference config for generation parameters
    icfg = getattr(tcfg, 'infer', None)
    num_inference_steps = getattr(icfg, 'num_inference_steps', 50) if icfg else 50
    guidance_scale = getattr(icfg, 'guidance_scale', 1.0) if icfg else 1.0

    model.eval()

    # 1. Collect real images from dataloader
    logger.info(f"[FID] Collecting real images from dataloader (target: {samples_per_class * num_classes} total)...")
    max_real_samples = samples_per_class * num_classes
    real_count = 0

    for batch in dataloader:
        if real_count >= max_real_samples:
            break
        x0 = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Rescale from [-1, 1] to [0, 1]
        x0_01 = (x0 + 1.0) / 2.0

        fid_computer.add_real(x0_01, labels)
        real_count += x0.size(0)

    logger.info(f"[FID] Collected {fid_computer.real_count} real images")

    # 2. Generate fake samples (samples_per_class per class)
    logger.info(f"[FID] Generating {samples_per_class} samples per class ({samples_per_class * num_classes} total)...")

    # Create scheduler for generation
    gen_scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
        thresholding=False,
    )
    gen_scheduler.set_timesteps(num_inference_steps, device=device)

    for class_id in range(num_classes):
        n_generated = 0
        while n_generated < samples_per_class:
            cur_batch = min(batch_size, samples_per_class - n_generated)

            # Start from random noise
            x = torch.randn(
                (cur_batch, tcfg.in_channels, tcfg.image_size, tcfg.image_size),
                device=device
            )
            labels = torch.full((cur_batch,), class_id, device=device, dtype=torch.long)

            # Denoise using scheduler
            for t in gen_scheduler.timesteps:
                t_batch = t.expand(cur_batch).to(device)

                # Forward pass with optional CFG
                with torch.autocast(device_type=device.type, enabled=tcfg.mixed_precision):
                    eps_cond = model(x, t_batch, labels)

                if guidance_scale != 1.0:
                    eps_uncond = model(x, t_batch, None)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond

                x = gen_scheduler.step(model_output=eps, timestep=t, sample=x).prev_sample

            # Clamp and rescale to [0, 1]
            fake_images = (x.clamp(-1, 1) + 1.0) / 2.0

            fid_computer.add_fake(fake_images, labels)
            n_generated += cur_batch

    logger.info(f"[FID] Generated {fid_computer.fake_count} fake images")

    # 3. Compute FID
    logger.info("[FID] Computing FID scores...")
    fid_metrics = fid_computer.compute(min_samples_per_class=min_samples)

    logger.info(f"[FID] Global FID: {fid_metrics.get('fid_global', 'N/A'):.2f}")

    return fid_metrics


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

            freq = counts.astype("float64") / float(counts.sum())
            inv_freq = 1.0 / np.maximum(freq, 1e-8)
            inv_freq /= inv_freq.mean()  # normalize around 1.0 to keep gradients stable

            # Stored on CPU; DDPMNoiseMSE will move to the correct device internally
            class_weights = torch.from_numpy(inv_freq.astype("float32"))

            if is_main_process():
                logger.info("Class weights computed from training set (inverse frequency):")
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
        fid_config = getattr(tcfg, 'fid', DEFAULT_FID_CONFIG)
        if isinstance(fid_config, bool):
            # If fid is just True/False in config, use defaults
            fid_config = DEFAULT_FID_CONFIG if fid_config else None

        fid_time = 0.0
        if fid_config is not None and is_main_process():
            fid_start_time = time.time()
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
