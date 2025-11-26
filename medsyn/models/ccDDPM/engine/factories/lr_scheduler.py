from typing import Any
import logging
import torch.optim as optim

logger = logging.getLogger("medsyn.ccddpm.train")

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
        ) # type: ignore
        logger.info(f"Using CosineAnnealingLR scheduler: T_max={T_max}, eta_min={eta_min}")
        return scheduler, False  # Step per epoch

    elif scheduler_type == "linear_warmup_cosine_annealing_lr" or scheduler_type == "cosine_warmup":
        warmup_epochs = getattr(scheduler_cfg, 'warmup_epochs', 5)
        eta_min = getattr(scheduler_cfg, 'eta_min', 1e-7)
        warmup_steps = warmup_epochs * steps_per_epoch

        # Compose warmup + cosine scheduler
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
        ) # type: ignore
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
        ) # type: ignore
        logger.info(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
        return scheduler, False  # Step per epoch        

    elif scheduler_type in ("constant", "none"):
        logger.info("Using constant learning rate (no scheduler)")
        return None, False

    else:
        logger.warning(f"Unknown scheduler type '{scheduler_type}', using constant LR")
        return None, False