from typing import Dict, Any
import logging
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from tqdm import tqdm

from medsyn.models.ccDDPM.loss import DDPMNoiseMSE
from medsyn.models.ccDDPM.metrics import PerClassMetricsAccumulator
from medsyn.models.ccDDPM.engine.logging.training_logging import EpochAverager, NUM_CLASSES
from medsyn.models.ccDDPM.engine.utils.math import compute_batch_metrics
from medsyn.models.ccDDPM.engine.utils.ddp_utils import sync_metrics_dict   
from medsyn.models.ccDDPM.engine.utils.ddp_utils import is_main_process
from medsyn.models.ccDDPM.engine.logging.logging import log_validation_progress, should_log_step

logger = logging.getLogger("medsyn.ccddpm.train")
IS_SUPERCOMPUTER = False  # Will be set by train.py


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