from typing import Any, Optional, Tuple, Union

import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import logging

# Optional import of Flash scheduler
try:
    # Signature: (optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1)
    # warmup_epochs / max_epochs are *iterations* (steps), not necessarily "epochs" as in your training loop. 
    from flash.core.optimizers import LinearWarmupCosineAnnealingLR  # type: ignore[attr-defined]
except Exception:  # flash not installed or misconfigured
    LinearWarmupCosineAnnealingLR = None  # type: ignore[assignment]


logger = logging.getLogger("medsyn.ccddpm.train")

# =============================================================================
# LEARNING RATE SCHEDULER FACTORY
# =============================================================================
def create_lr_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Any,
    steps_per_epoch: int,
    total_epochs: int,
    base_lr: float,
) -> Tuple[Union[Optional[_LRScheduler], optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CosineAnnealingLR], bool]:
    """
    Create a learning rate scheduler based on configuration.

    Args:
        optimizer: Wrapped optimizer whose LR will be scheduled.
        scheduler_cfg: Configuration object with scheduler parameters (typically a simple namespace/YAML node).
        steps_per_epoch: Number of training steps (batches) per epoch.
        total_epochs: Total number of training epochs.
        base_lr: Base learning rate from optimizer config.

    Returns:
        (scheduler, step_per_batch)

        - scheduler: Instantiated lr_scheduler or None (for constant LR).
        - step_per_batch: True if the scheduler must be stepped after each batch,
          False if it should be stepped once per epoch.

    Supported scheduler types:
        - "onecycle": One-Cycle LR (steps per batch)
        - "cosine": Cosine annealing (steps per epoch)
        - "cosine_warmup" / "linear_warmup_cosine_annealing_lr":
              Linear warmup + cosine decay (steps per batch)
        - "constant": No scheduling
        - "step": Step decay (steps per epoch)
    """
    total_steps = steps_per_epoch * total_epochs
    scheduler_type = getattr(scheduler_cfg, "type", "constant").lower()

    # -------------------------------------------------------------------------
    # OneCycle
    # -------------------------------------------------------------------------
    if scheduler_type == "onecycle":
        max_lr = getattr(scheduler_cfg, "max_lr", base_lr * 2.0)
        pct_start = getattr(scheduler_cfg, "pct_start", 0.3)
        div_factor = getattr(scheduler_cfg, "div_factor", 25.0)
        final_div_factor = getattr(scheduler_cfg, "final_div_factor", 1000.0)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        logger.info(
            "Using OneCycleLR scheduler: "
            f"max_lr={max_lr}, pct_start={pct_start}, div_factor={div_factor}, "
            f"final_div_factor={final_div_factor}, total_steps={total_steps}"
        )
        return scheduler, True  # step per batch

    # -------------------------------------------------------------------------
    # Cosine (epoch-level)
    # -------------------------------------------------------------------------
    elif scheduler_type == "cosine":
        T_max = getattr(scheduler_cfg, "T_max", None) or total_epochs
        eta_min = getattr(scheduler_cfg, "eta_min", 1e-7)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
        logger.info(
            "Using CosineAnnealingLR scheduler: "
            f"T_max={T_max}, eta_min={eta_min}"
        )
        return scheduler, False  # dont step per epoch

    # -------------------------------------------------------------------------
    # Linear warmup + cosine decay
    # -------------------------------------------------------------------------
    elif scheduler_type in ("linear_warmup_cosine_annealing_lr", "cosine_warmup"):
        # User-facing: warmup_epochs is in "epochs"
        warmup_epochs = getattr(scheduler_cfg, "warmup_epochs", 5)
        eta_min = getattr(scheduler_cfg, "eta_min", 1e-7)

        # Convert to iterations (steps) — this matches the Flash implementation,
        # which interprets warmup_epochs and max_epochs as "max number of iterations". :contentReference[oaicite:1]{index=1}
        warmup_steps = warmup_epochs * steps_per_epoch
        max_steps = total_steps

        # Two ways to specify the warmup start LR:
        # 1) Absolute: warmup_start_lr (overrides everything)
        # 2) Relative: warmup_start_factor * base_lr (default: 1% of base_lr)
        warmup_start_factor = getattr(scheduler_cfg, "warmup_start_factor", 0.01)
        warmup_start_lr = getattr(
            scheduler_cfg,
            "warmup_start_lr",
            warmup_start_factor * base_lr,
        )

        if LinearWarmupCosineAnnealingLR is not None:
            # Use Flash scheduler if available
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=warmup_steps,  # actually "iterations"
                max_epochs=max_steps,        # actually "iterations"
                warmup_start_lr=warmup_start_lr,
                eta_min=eta_min,
            )
            logger.info(
                "Using Flash LinearWarmupCosineAnnealingLR: "
                f"warmup_steps={warmup_steps}, max_steps={max_steps}, "
                f"warmup_start_lr={warmup_start_lr}, eta_min={eta_min}"
            )
        else:
            # Fallback: compose LinearLR + CosineAnnealingLR via SequentialLR
            if base_lr <= 0.0:
                logger.warning(
                    "Base LR <= 0 detected when building warmup scheduler; "
                    "falling back to start_factor=0.01."
                )
                start_factor = 0.01
            else:
                start_factor = warmup_start_lr / base_lr

            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=eta_min,
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            logger.info(
                "Using SequentialLR (LinearLR + CosineAnnealingLR) warmup scheduler: "
                f"warmup_steps={warmup_steps}, max_steps={max_steps}, "
                f"start_factor={start_factor:.6f}, eta_min={eta_min}"
            )

        # Flash docs explicitly recommend stepping this scheduler *per iteration*. :contentReference[oaicite:2]{index=2}
        return scheduler, True  # step per batch

    # -------------------------------------------------------------------------
    # Step decay
    # -------------------------------------------------------------------------
    elif scheduler_type == "step":
        step_size = getattr(scheduler_cfg, "step_size", 30)
        gamma = getattr(scheduler_cfg, "gamma", 0.1)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        logger.info(
            "Using StepLR scheduler: "
            f"step_size={step_size}, gamma={gamma}"
        )
        return scheduler, False  # dont step per epoch

    # -------------------------------------------------------------------------
    # Constant LR / no scheduler
    # -------------------------------------------------------------------------
    elif scheduler_type in ("constant", "none"):
        logger.info("Using constant learning rate (no scheduler)")
        return None, False

    # -------------------------------------------------------------------------
    # Unknown type -> constant
    # -------------------------------------------------------------------------
    else:
        logger.warning(
            "Unknown scheduler type '%s', using constant LR (no scheduler)",
            scheduler_type,
        )
        return None, False
