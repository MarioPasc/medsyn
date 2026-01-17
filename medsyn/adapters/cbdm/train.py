# medsyn/adapters/cbdm/train.py
"""
CBDM training wrapper for MedSyn.

This module provides a training loop that uses CBDM model components
with MedSyn's unified dataset format, without modifying CBDM source code.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image
from tqdm import tqdm

from medsyn.adapters.cbdm.config import CBDMConfig, load_cbdm_config
from medsyn.adapters.cbdm.dataset import CBDMNPZDataset

# Import CBDM components (no modifications needed)
from medsyn.models.cbdm.model.model import UNet
from medsyn.models.cbdm.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler

logger = logging.getLogger(__name__)


def warmup_lr_scheduler(warmup_steps: int):
    """Create a warmup learning rate scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    return lr_lambda


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train(config_path: str) -> None:
    """
    Train CBDM model using MedSyn's unified NPZ dataset.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    cfg = load_cbdm_config(config_path)
    logger.info("Training CBDM with config: %s", config_path)

    # Setup device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Build dataset
    logger.info("Loading dataset from: %s", cfg.npz_path)
    dataset = CBDMNPZDataset(
        cfg.npz_path,
        split="train",
        image_size=cfg.image_size,
        augment=True,
    )
    logger.info("Dataset size: %d samples, %d classes", len(dataset), dataset.num_classes)

    # Update num_classes from dataset
    cfg.num_classes = dataset.num_classes

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Build model
    logger.info("Building UNet model...")
    model = UNet(
        T=cfg.T,
        ch=cfg.ch,
        ch_mult=cfg.ch_mult,
        attn=cfg.attn,
        num_res_blocks=cfg.num_res_blocks,
        dropout=cfg.dropout,
        cond=True,  # Conditional generation
        augm=False,  # No augmentation embedding
        num_class=cfg.num_classes,
    ).to(device)

    # Compute class weights for CBDM
    class_weights = dataset.get_class_weights()
    logger.info("Class weights: %s", class_weights)

    # Build trainer
    trainer = GaussianDiffusionTrainer(
        model=model,
        beta_1=cfg.beta_1,
        beta_T=cfg.beta_T,
        T=cfg.T,
        dataset=dataset,
        num_class=cfg.num_classes,
        cfg=cfg.cfg,
        cb=cfg.cb,
        tau=cfg.tau,
        weight=class_weights,
        finetune=cfg.finetune,
    ).to(device)

    # Build sampler for generating samples during training
    sampler = GaussianDiffusionSampler(
        model=model,
        beta_1=cfg.beta_1,
        beta_T=cfg.beta_T,
        T=cfg.T,
        num_class=cfg.num_classes,
        img_size=cfg.image_size,
    ).to(device)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_scheduler(cfg.warmup))

    # Setup EMA
    ema = EMA(model, decay=cfg.ema_decay)

    # Load checkpoint if specified
    start_step = 0
    if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
        logger.info("Loading checkpoint: %s", cfg.checkpoint_path)
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.shadow = checkpoint["ema"]
        start_step = checkpoint.get("step", 0)
        logger.info("Resumed from step %d", start_step)

    # Training loop
    logger.info("Starting training for %d steps...", cfg.total_steps)
    model.train()

    data_iter = iter(dataloader)
    for step in tqdm(range(start_step, cfg.total_steps), initial=start_step, total=cfg.total_steps):
        # Get batch (cycle through dataset)
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, labels = next(data_iter)

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        loss, loss_cb = trainer(images, labels)
        total_loss = loss.mean() + loss_cb.mean()

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Update
        optimizer.step()
        scheduler.step()
        ema.update()

        # Logging
        if step % cfg.log_interval == 0:
            logger.info(
                "Step %d/%d - Loss: %.4f, CB Loss: %.4f, LR: %.6f",
                step, cfg.total_steps, loss.mean().item(), loss_cb.mean().item(),
                scheduler.get_last_lr()[0],
            )

        # Generate samples
        if step > 0 and step % cfg.sample_interval == 0:
            logger.info("Generating samples at step %d...", step)
            model.eval()
            ema.apply_shadow()

            with torch.no_grad():
                # Generate samples
                noise = torch.randn(cfg.num_samples, 3, cfg.image_size, cfg.image_size).to(device)
                samples, labels_gen = sampler(noise, omega=cfg.omega, method=cfg.method)
                # Rescale from [-1, 1] to [0, 1]
                samples = (samples + 1) / 2

            save_image(
                samples,
                output_dir / "samples" / f"step_{step:06d}.png",
                nrow=8,
                normalize=False,
            )

            ema.restore()
            model.train()

        # Save checkpoint
        if step > 0 and step % cfg.save_interval == 0:
            checkpoint_path = output_dir / "checkpoints" / f"step_{step:06d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "ema": ema.shadow,
                "config": cfg,
            }, checkpoint_path)
            logger.info("Saved checkpoint: %s", checkpoint_path)

    # Save final checkpoint
    final_path = output_dir / "checkpoints" / "final.pt"
    torch.save({
        "step": cfg.total_steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.shadow,
        "config": cfg,
    }, final_path)
    logger.info("Training complete! Final checkpoint: %s", final_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CBDM with MedSyn dataset")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    train(args.config)
