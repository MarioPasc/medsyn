# medsyn/models/ccDDPM/engine/train.py
# Purpose: Training loop for class-conditioned DDPM with Diffusers' DDPMScheduler.
# Features: mixed precision, EMA, classifier-free guidance (label drop), checkpointing, CSV logs, DDP support.
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
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
from medsyn.models.ccDDPM.loss import DDPMNoiseMSE
from medsyn.models.ccDDPM.metrics import compute_psnr, compute_ssim
from medsyn.models.ccDDPM.training_logging import CSVTrainingLogger, EpochAverager
from medsyn.models.ccDDPM.engine.ddp_utils import (
    ddp_is_enabled, ddp_init, is_main_process,
    barrier, cleanup, all_reduce_mean, broadcast_bool, get_state_dict_for_save
)
import numpy as np

logger = logging.getLogger("medsyn.ccddpm.train")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

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
    Compute diagnostic metrics to detect training issues.

    Returns:
        Dictionary with:
        - input_output_correlation: Correlation between noisy input and model prediction (should be LOW)
        - reconstruction_mse_t100: MSE at early timestep (high noise)
        - reconstruction_mse_t500: MSE at mid timestep
        - reconstruction_psnr_t500: PSNR at mid timestep
        - prediction_std: Std of model outputs (should be ~1.0 for normalized data)
    """
    model.eval()

    # Take subset of batch
    x0 = x0_batch[:num_samples]
    labels = labels_batch[:num_samples]

    # Test at different timesteps
    correlations = []
    recon_mse_t100 = []
    recon_mse_t500 = []
    recon_psnr_t500 = []
    pred_stds = []

    for t_val in [100, 500]:
        t = torch.full((len(x0),), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = noise_scheduler.add_noise(x0, noise, t)

        # Model prediction
        eps_pred = model(x_t, t, labels)

        # Correlate ε̂ with true ε, not with x_t (avoids false "echoing" alarms)
        eps_true = noise
        x_vec = eps_pred.flatten()
        y_vec = eps_true.flatten()
        x_std = x_vec.std(); y_std = y_vec.std()
        if x_std > 1e-8 and y_std > 1e-8:
            corr = torch.corrcoef(torch.stack([x_vec, y_vec]))[0,1].item()
        else:
            corr = 0.0
        correlations.append(corr)

        # Prediction std
        pred_stds.append(eps_pred.std().item())

        # Reconstruct x0
        sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        x0_pred = (x_t - sqrt_one_minus_alpha_prod * eps_pred) / sqrt_alpha_prod
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # Reconstruction metrics
        mse = F.mse_loss(x0_pred, x0).item()
        psnr = compute_psnr(x0_pred, x0)

        if t_val == 100:
            recon_mse_t100.append(mse)
        else:
            recon_mse_t500.append(mse)
            recon_psnr_t500.append(psnr)

    model.train()

    return {
        "input_output_correlation": float(np.mean(correlations)),  # Should be << 0.5, ideally negative
        "reconstruction_mse_t100": float(np.mean(recon_mse_t100)),
        "reconstruction_mse_t500": float(np.mean(recon_mse_t500)),
        "reconstruction_psnr_t500": float(np.mean(recon_psnr_t500)),
        "prediction_std": float(np.mean(pred_stds)),  # Should be ~0.8-1.2 for normalized data
    }

@torch.no_grad()
def full_chain_reconstruction_psnr(model, scheduler, x0, y, device):
    """
    Full-chain reconstruction test: add noise at random t, sample back to t=0, compute PSNR.
    This catches multi-step drift that single-step x̂₀ formulas miss.

    Args:
        model: DDPM model
        scheduler: DDPM scheduler
        x0: Clean image [N, C, H, W]
        y: Class labels [N]
        device: Device

    Returns:
        PSNR of reconstructed image
    """
    model.eval()
    x0 = x0[:1].to(device); y = y[:1].to(device)
    T = scheduler.config.num_train_timesteps
    t = torch.randint(T//4, 3*T//4, (1,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, t)
    # run reverse from current t to 0
    scheduler.set_timesteps(T)
    # find index of closest scheduler timestep to t
    start_idx = int((scheduler.timesteps - t.cpu()).abs().argmin().item())
    # Move scheduler timesteps to device to avoid device mismatch in scheduler.step()
    scheduler.timesteps = scheduler.timesteps.to(device)
    x = x_t
    for i in range(start_idx, len(scheduler.timesteps)):
        tt = scheduler.timesteps[i].unsqueeze(0)
        eps = model(x, tt, y)
        x = scheduler.step(eps, tt, x).prev_sample
    x_rec = torch.clamp(x, -1, 1)
    psnr = compute_psnr((x_rec+1)/2, (x0+1)/2, max_val=1.0)
    return float(psnr)

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
    
    return metrics

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
    ocfg = cfg.ccddpm.optim

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

        if use_ddp:
            # DDP: Create DistributedSamplers (shuffle via sampler, not DataLoader)
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if is_main_process():
                logger.info(f"DDP: DistributedSampler created for {len(train_dataset)} train, "
                           f"{len(val_dataset)} val samples across {world_size} GPUs")
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

        if use_ddp:
            # DDP: Create DistributedSamplers (shuffle via sampler, not DataLoader)
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=tcfg.seed
            )
            val_sampler = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if is_main_process():
                logger.info(f"DDP: DistributedSampler created for {len(train_dataset)} train, "
                           f"{len(val_dataset)} val samples across {world_size} GPUs")
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

    # ========================================================================
    # MODEL, SCHEDULER, OPTIMIZER, EMA
    # ========================================================================
    # Build model
    mcfg = CCDDPMInit(
        in_channels=tcfg.in_channels,
        class_embed_dim=tcfg.class_embed_dim,
        num_classes=tcfg.num_classes,
    )
    model = CCDDPM(mcfg).to(device)

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

    # Build optimizer (works on DDP-wrapped model if use_ddp=True, unwrapped otherwise)
    opt = optim.AdamW(model.parameters(), lr=ocfg.lr, betas=ocfg.betas, eps=ocfg.eps, weight_decay=ocfg.wd)
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
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "samples").mkdir(parents=True, exist_ok=True)
        (out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

        # Initialize CSV logger with per-class metrics
        extra_fields = []
        for k in range(tcfg.num_classes):
            extra_fields.append(f"loss_c{k}")
        csv_logger = CSVTrainingLogger(str(out_dir / "training_metrics.csv"), extra_fields=extra_fields)

        # Keep old simple CSV for backward compatibility
        csv_path = out_dir / "train_log.csv"
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["epoch","step","loss","lr","time_s"])

    # Synchronize all processes after directory creation
    if use_ddp:
        barrier()

    # Track best validation loss for best.pt and early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    
    global_step = 0
    model.train()

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    for epoch in range(1, tcfg.epochs + 1):
        # DDP: Set epoch for DistributedSampler to ensure different shuffling per epoch
        # Legacy: No sampler, shuffling handled by DataLoader
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        running_loss = 0.0
        train_avg = EpochAverager()

        # Collect augmentation statistics for this epoch
        epoch_augmentation_transforms = []

        # Progress bar for the training epoch (only on rank-0 if using tqdm)
        train_iter = enumerate(train_loader, 1)
        if tcfg.use_tqdm and is_main_process():
            pbar = tqdm(train_iter, total=len(train_loader),
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
                    alphas_cumprod=noise_scheduler.alphas_cumprod
                )
            # Guard: skip non-finite loss early
            if not torch.isfinite(loss):
                logger.warning(f"⚠️  Skipping step {global_step} due to non-finite loss")
                opt.zero_grad(set_to_none=True); scaler.update(); global_step += 1; continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Compute gradient norm and check for NaN/Inf BEFORE optimizer step
            skip_step = False
            if tcfg.grad_clip_norm:
                scaler.unscale_(opt)
                # Clip gradients; error_if_nonfinite=False prevents crashes on overflow
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tcfg.grad_clip_norm, error_if_nonfinite=False
                )
                grad_norm_val = float(grad_norm)
                # Check if gradients are non-finite
                if not math.isfinite(grad_norm_val):
                    skip_step = True
                    grad_norm_val = float("nan")
                    logger.warning(f"⚠️  Skipping step {global_step} due to non-finite gradients (grad_norm={grad_norm})")
            else:
                # Compute gradient norm for logging
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm_val = total_norm ** 0.5
                if not math.isfinite(grad_norm_val):
                    skip_step = True
                    grad_norm_val = float("nan")
                    logger.warning(f"⚠️  Skipping step {global_step} due to non-finite gradients (grad_norm={grad_norm_val})")

            # Only update model if gradients are finite
            if not skip_step:
                scaler.step(opt)
                scaler.update()
                if ema:
                    # CRITICAL: Update EMA with base_model (unwrapped), not DDP wrapper
                    ema.update(base_model)
            else:
                # Still update scaler state even when skipping
                scaler.update()
                # Zero out gradients to prevent accumulation
                opt.zero_grad(set_to_none=True)

            global_step += 1
            
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
                batch_metrics["grad_norm"] = grad_norm_val
                batch_metrics["ema_enabled"] = 1.0 if ema else 0.0
                batch_metrics["skipped_step"] = 1.0 if skip_step else 0.0

                # Update epoch averager
                train_avg.update(batch_metrics, batch_size=bsz)

            # Update running loss
            # Accumulate only finite losses
            li = float(loss.detach().cpu())
            if math.isfinite(li):
                running_loss += li

            # Update progress bar with current loss (only if tqdm enabled AND on main process)
            if tcfg.use_tqdm and is_main_process():
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                "avg_loss": f"{running_loss/step:.4f}",
                                "psnr": f"{batch_metrics['psnr']:.2f}dB"})
            
            if global_step % tcfg.log_every == 0 and is_main_process():
                csv_path = out_dir / "train_log.csv"
                with open(csv_path, "a", newline="") as fh:
                    csv.writer(fh).writerow([epoch, global_step, float(loss.item()), opt.param_groups[0]["lr"], round(time.time()-t0, 2)])
                logger.info("ep=%d step=%d loss=%.4f", epoch, global_step, float(loss.item()))

        if tcfg.use_tqdm and is_main_process():
            pbar.close()

        # Get per-class losses from loss_fn (local to this GPU)
        per_class_losses = loss_fn.per_class()
        train_metrics = train_avg.means()
        for c, loss_c in per_class_losses.items():
            train_metrics[f"loss_c{c}"] = loss_c

        # CRITICAL: Synchronize training metrics across all GPUs
        train_metrics = sync_metrics_dict(train_metrics, device, use_ddp)

        # Log training metrics (only rank-0, but now with globally averaged values)
        curr_lr = opt.param_groups[0]["lr"]
        if csv_logger is not None:
            csv_logger.log_epoch(epoch=epoch, split="train", lr=curr_lr, metrics=train_metrics)

        # Print epoch summary (only rank-0)
        if is_main_process():
            avg_loss = running_loss / len(train_loader)
            print("\n" + "="*80)
            print(f"Epoch {epoch}/{tcfg.epochs} Summary:")
            print("="*80)
            print("Training:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  PSNR: {train_metrics.get('psnr', 0.0):.2f} dB")
            print(f"  SSIM: {train_metrics.get('ssim', 0.0):.4f}")
            print(f"  Noise MSE: {train_metrics.get('noise_mse', 0.0):.4f}")
            print(f"  Learning Rate: {curr_lr:.6f}")
            print(f"  Time: {time.time()-t0:.2f}s")
        
        # ---- Validation ----
        model.eval()
        val_avg = EpochAverager()
        val_loss_fn = DDPMNoiseMSE(
            num_classes=tcfg.num_classes,
            use_min_snr=tcfg.use_min_snr,
            min_snr_gamma=tcfg.min_snr_gamma
        )
        
        # Validation progress bar (only on rank-0 if using tqdm)
        if tcfg.use_tqdm and is_main_process():
            val_pbar = tqdm(val_loader, desc="Validation", unit="batch", leave=False)
        else:
            val_pbar = val_loader
        
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=tcfg.mixed_precision):
            for batch in val_pbar:
                x0 = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                bsz = x0.size(0)
                
                # Sample t and noise
                t = torch.randint(0, scfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                x_t = noise_scheduler.add_noise(x0, noise, t)
                
                # Forward
                pred = model(x_t, t, labels)
                loss = val_loss_fn(
                    pred, noise, labels,
                    timesteps=t,
                    alphas_cumprod=noise_scheduler.alphas_cumprod
                )
                
                # Reconstruct x0 for metrics
                sqrt_alpha_prod = noise_scheduler.alphas_cumprod[t].sqrt()
                sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod[t]).sqrt()

                # Add epsilon to prevent division by zero
                sqrt_alpha_prod = torch.clamp(sqrt_alpha_prod, min=1e-6)
                x0_pred = (x_t - sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1) * pred) / sqrt_alpha_prod.view(-1, 1, 1, 1)

                # Clamp to prevent extreme values
                x0_pred = torch.clamp(x0_pred, -10.0, 10.0)
                
                # Compute metrics
                val_batch_metrics = compute_batch_metrics(pred, noise, x0_pred, x0, loss)
                val_batch_metrics["grad_norm"] = 0.0  # No gradients in validation
                val_batch_metrics["ema_enabled"] = 1.0 if ema else 0.0
                val_avg.update(val_batch_metrics, batch_size=bsz)
                
                # Update progress bar (only if tqdm enabled on rank-0)
                if tcfg.use_tqdm and is_main_process():
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

        if tcfg.use_tqdm and is_main_process():
            val_pbar.close()

        # Get per-class validation losses (local to this GPU)
        val_per_class_losses = val_loss_fn.per_class()
        val_metrics = val_avg.means()
        for c, loss_c in val_per_class_losses.items():
            val_metrics[f"loss_c{c}"] = loss_c

        # CRITICAL: Synchronize ALL validation metrics across GPUs (not just overall loss)
        val_metrics = sync_metrics_dict(val_metrics, device, use_ddp)

        # Extract the globally synchronized validation loss for early stopping
        val_loss = val_metrics.get("loss", float("inf"))
        
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

        # Full-chain reconstruction test
        full_chain_psnr = full_chain_reconstruction_psnr(
            model=model,
            scheduler=noise_scheduler,
            x0=x0_diag,
            y=labels_diag,
            device=device
        )
        diagnostics["full_chain_psnr"] = full_chain_psnr

        # Log validation metrics (only rank-0)
        if csv_logger is not None:
            csv_logger.log_epoch(epoch=epoch, split="val", lr=curr_lr, metrics=val_metrics)

        # Log diagnostics separately (only rank-0)
        if csv_logger is not None:
            csv_logger.log_epoch(epoch=epoch, split="diag", lr=curr_lr, metrics=diagnostics)

        # Print validation summary (only rank-0)
        if is_main_process():
            print("\nValidation:")
            print(f"  Average Loss: {val_metrics.get('loss', 0.0):.4f} (Global: {val_loss:.4f})")
            print(f"  PSNR: {val_metrics.get('psnr', 0.0):.2f} dB")
            print(f"  SSIM: {val_metrics.get('ssim', 0.0):.4f}")
            print(f"  Noise MSE: {val_metrics.get('noise_mse', 0.0):.4f}")

            print("\n🔍 Training Diagnostics (detecting issues):")
            corr = diagnostics['input_output_correlation']
            pred_std = diagnostics['prediction_std']
            recon_psnr = diagnostics['reconstruction_psnr_t500']

            # Color-coded warnings
            print(f"  Input-Output Correlation: {corr:.4f}", end="")
            if corr > 0.7:
                print(" ⚠️  WARNING: Model is echoing input! (should be < 0.5)")
            elif corr > 0.5:
                print(" ⚠️  High correlation, model may not be learning properly")
            else:
                print(" ✓ (healthy)")

            print(f"  Prediction Std: {pred_std:.4f}", end="")
            if pred_std < 0.5 or pred_std > 1.5:
                print(" ⚠️  Unusual (should be ~0.8-1.2)")
            else:
                print(" ✓")

            print(f"  Reconstruction PSNR@t500: {recon_psnr:.2f} dB", end="")
            if recon_psnr < 15.0:
                print(" ⚠️  Low quality (should improve over epochs)")
            else:
                print(" ✓")

            print(f"  Reconstruction MSE@t500: {diagnostics['reconstruction_mse_t500']:.4f}")
            print(f"  Gradient Norm (mean): {train_metrics.get('grad_norm', 0.0):.4f}")
            print("="*80 + "\n")

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

        # ---- Checkpointing & Early Stopping ----
        # Note: val_loss is already synchronized across GPUs above

        # Determine if this is the best model (on rank-0, then broadcast)
        if is_main_process():
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            should_stop = epochs_without_improvement >= tcfg.patience
        else:
            is_best = False
            should_stop = False

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
                "ema": (ema.shadow if ema else None),
                "cfg": tcfg.__dict__,
                # Save diagnostics for post-training analysis
                "diagnostics": diagnostics,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                # EMA verification
                "ema_enabled": ema is not None,
                "ema_num_params": len(ema.shadow) if ema else 0,
            }

            # Always save last.pt
            torch.save(checkpoint_data, out_dir / "ckpts" / "last.pt")

            # Save best.pt if this is the best validation loss so far
            if is_best:
                torch.save(checkpoint_data, out_dir / "ckpts" / "best.pt")
                logger.info(f"✓ New best model at epoch {epoch} with val_loss={val_loss:.4f}")
            else:
                logger.info(f"No improvement for {epochs_without_improvement}/{tcfg.patience} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")

            # Save periodic checkpoint every X epochs
            if (epoch % tcfg.ckpt_every_epochs) == 0:
                ck = out_dir / "ckpts" / f"epoch_{epoch:04d}.pt"
                torch.save(checkpoint_data, ck)
                logger.info(f"Saved periodic checkpoint: {ck.name}")

        # Early stopping check (synchronized across all GPUs)
        if should_stop:
            if is_main_process():
                logger.info(f"⚠ Early stopping triggered! No improvement for {tcfg.patience} epochs.")
                logger.info(f"Best model was at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
                print(f"\n{'='*80}")
                print(f"⚠ Early Stopping at Epoch {epoch}")
                print(f"{'='*80}")
                print(f"No improvement in validation loss for {tcfg.patience} consecutive epochs.")
                print(f"Best model saved at epoch {best_epoch} with val_loss={best_val_loss:.4f}")
                print(f"{'='*80}\n")
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

            # Every 10 epochs: Save DDPM-specific visualizations
            if epoch % 10 == 0 or epoch == 1:
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
                        guidance_scale=1.0
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
                            num_steps=1,  # Minimal intermediate step (returns 3 images: initial + 1 intermediate + final)
                            device=device,
                            guidance_scale=1.0
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
                        num_samples=5
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
    # Save final augmentation statistics (rank-0 only)
    if augmentation_stats is not None:
        augmentation_stats.save_csv(final=True)
        augmentation_stats.print_summary()
        logger.info("Final augmentation statistics saved")

    # Print summary (rank-0 only)
    if is_main_process():
        print(f"\n{'='*80}")
        print("🎉 Training Completed!")
        print("="*80)
        print("Best Validation Loss: {:.4f} (Epoch {})".format(best_val_loss, best_epoch))
        print("Completed Epochs: {}/{}".format(epoch, tcfg.epochs))
        if epochs_without_improvement >= tcfg.patience:
            print("Stopped early: No improvement for {} epochs".format(tcfg.patience))
        print("\nCheckpoints saved in: {}".format(out_dir / 'ckpts'))
        print("  - best.pt: Best model (epoch {}, val_loss={:.4f})".format(best_epoch, best_val_loss))
        print("  - last.pt: Final epoch model (epoch {})".format(epoch))
        print("  - epoch_XXXX.pt: Periodic checkpoints every {} epochs".format(tcfg.ckpt_every_epochs))
        print("\nVisualizations saved in: {}".format(out_dir / 'samples'))
        print("Metrics logged in: {}".format(out_dir / 'training_metrics.csv'))
        print("="*80 + "\n")
        logger.info(f"Training completed! Best model at epoch {best_epoch} with val_loss={best_val_loss:.4f}")

    # ========================================================================
    # CLEANUP
    # ========================================================================
    if use_ddp:
        # DDP: Clean up process group
        cleanup()
        logger.info("DDP cleanup completed")
    # Legacy: No cleanup needed
