from typing import Dict, Any
import logging
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from medsyn.models.ccDDPM.metrics import TorchmetricsFIDComputer
from medsyn.models.ccDDPM.engine.logging.training_logging import NUM_CLASSES

logger = logging.getLogger("medsyn.ccddpm.train")


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
    weights_path = fid_config.get("weights_path", None)

    # Initialize FID computer
    fid_computer = TorchmetricsFIDComputer(
        num_classes=num_classes,
        device=str(device),
        normalize=True,
        feature_extractor_weights_path=weights_path,
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