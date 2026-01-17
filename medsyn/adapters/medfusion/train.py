# medsyn/adapters/medfusion/train.py
"""
MedFusion training wrapper for MedSyn.

This module provides a training function that uses MedFusion's diffusion pipeline
with MedSyn's unified dataset format.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger
    HAS_LIGHTNING = True
except ImportError:
    HAS_LIGHTNING = False

from medsyn.adapters.medfusion.config import MedFusionConfig, load_medfusion_config
from medsyn.adapters.medfusion.dataset import MedFusionDataModule

# Import MedFusion components
from medsyn.models.medfusion.medical_diffusion.models.pipelines.diffusion_pipeline import DiffusionPipeline
from medsyn.models.medfusion.medical_diffusion.models.estimators.unet import UNet
from medsyn.models.medfusion.medical_diffusion.models.noise_schedulers import GaussianNoiseScheduler
from medsyn.models.medfusion.medical_diffusion.models.embedders import LabelEmbedder

logger = logging.getLogger(__name__)


def build_noise_estimator(cfg: MedFusionConfig) -> UNet:
    """
    Build UNet noise estimator from config.

    Args:
        cfg: MedFusion configuration

    Returns:
        UNet model
    """
    ne_cfg = cfg.noise_estimator
    return UNet(
        in_ch=ne_cfg.in_ch,
        out_ch=ne_cfg.out_ch,
        spatial_dims=ne_cfg.spatial_dims,
        hid_chs=ne_cfg.hid_chs,
        kernel_sizes=ne_cfg.kernel_sizes,
        strides=ne_cfg.strides,
        num_res_blocks=ne_cfg.num_res_blocks,
        act_name=(ne_cfg.act_name, {}),
        norm_name=(ne_cfg.norm_name, {}),
        attention_levels=ne_cfg.attention_levels,
        dropout=ne_cfg.dropout,
        use_self_conditioning=ne_cfg.use_self_conditioning,
        estimate_variance=ne_cfg.estimate_variance,
        cond_embedder=LabelEmbedder,
        cond_embedder_kwargs={
            "num_classes": cfg.num_classes,
            "emb_dim": ne_cfg.hid_chs[0] * 4,
        },
    )


def build_noise_scheduler(cfg: MedFusionConfig) -> GaussianNoiseScheduler:
    """
    Build Gaussian noise scheduler from config.

    Args:
        cfg: MedFusion configuration

    Returns:
        GaussianNoiseScheduler instance
    """
    ns_cfg = cfg.noise_scheduler
    return GaussianNoiseScheduler(
        timesteps=ns_cfg.timesteps,
        beta_start=ns_cfg.beta_start,
        beta_end=ns_cfg.beta_end,
        schedule_strategy=ns_cfg.schedule_strategy,
    )


def build_pipeline(cfg: MedFusionConfig) -> DiffusionPipeline:
    """
    Build MedFusion diffusion pipeline from config.

    Args:
        cfg: MedFusion configuration

    Returns:
        DiffusionPipeline Lightning module
    """
    # Build noise estimator class and kwargs
    ne_cfg = cfg.noise_estimator
    noise_estimator_kwargs = {
        "in_ch": ne_cfg.in_ch,
        "out_ch": ne_cfg.out_ch,
        "spatial_dims": ne_cfg.spatial_dims,
        "hid_chs": ne_cfg.hid_chs,
        "kernel_sizes": ne_cfg.kernel_sizes,
        "strides": ne_cfg.strides,
        "num_res_blocks": ne_cfg.num_res_blocks,
        "act_name": (ne_cfg.act_name, {}),
        "norm_name": (ne_cfg.norm_name, {}),
        "attention_levels": ne_cfg.attention_levels,
        "dropout": ne_cfg.dropout,
        "use_self_conditioning": ne_cfg.use_self_conditioning,
        "estimate_variance": ne_cfg.estimate_variance,
        "cond_embedder": LabelEmbedder,
        "cond_embedder_kwargs": {
            "num_classes": cfg.num_classes,
            "emb_dim": ne_cfg.hid_chs[0] * 4,
        },
    }

    # Build noise scheduler kwargs
    ns_cfg = cfg.noise_scheduler
    noise_scheduler_kwargs = {
        "timesteps": ns_cfg.timesteps,
        "beta_start": ns_cfg.beta_start,
        "beta_end": ns_cfg.beta_end,
        "schedule_strategy": ns_cfg.schedule_strategy,
    }

    # Build pipeline
    pipeline = DiffusionPipeline(
        noise_scheduler=GaussianNoiseScheduler,
        noise_estimator=UNet,
        noise_scheduler_kwargs=noise_scheduler_kwargs,
        noise_estimator_kwargs=noise_estimator_kwargs,
        latent_embedder=None,
        estimator_objective=cfg.estimator_objective,
        estimate_variance=ne_cfg.estimate_variance,
        use_self_conditioning=ne_cfg.use_self_conditioning,
        classifier_free_guidance_dropout=cfg.classifier_free_guidance_dropout,
        do_input_centering=cfg.do_input_centering,
        clip_x0=cfg.clip_x0,
        use_ema=cfg.use_ema,
        ema_kwargs={"decay": cfg.ema_decay} if cfg.use_ema else {},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={"lr": cfg.training.lr},
        sample_every_n_steps=cfg.training.sample_every_n_steps,
    )

    return pipeline


def train(config_path: str) -> None:
    """
    Train MedFusion model using MedSyn's unified NPZ dataset.

    Args:
        config_path: Path to YAML configuration file
    """
    if not HAS_LIGHTNING:
        raise ImportError(
            "pytorch_lightning is required for MedFusion training. "
            "Install with: pip install pytorch-lightning"
        )

    # Load configuration
    cfg = load_medfusion_config(config_path)
    logger.info("Training MedFusion with config: %s", config_path)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data module
    logger.info("Loading dataset from: %s", cfg.npz_path)
    dm = MedFusionDataModule(
        npz_path=cfg.npz_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        image_size=cfg.image_size,
    )
    logger.info("Dataset: %s, %d classes", dm.dataset_name, dm.num_classes)

    # Update num_classes from dataset
    cfg.num_classes = dm.num_classes
    cfg.noise_estimator.num_classes = dm.num_classes

    # Build pipeline
    logger.info("Building diffusion pipeline...")
    pipeline = build_pipeline(cfg)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:03d}-{val/loss:.4f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Setup logger
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name="logs",
    )

    # Build trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        logger=tb_logger,
        default_root_dir=output_dir,
    )

    # Load checkpoint if specified
    ckpt_path = None
    if cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
        ckpt_path = cfg.checkpoint_path
        logger.info("Resuming from checkpoint: %s", ckpt_path)

    # Train
    logger.info("Starting training for %d epochs...", cfg.training.max_epochs)
    trainer.fit(pipeline, dm, ckpt_path=ckpt_path)

    logger.info("Training complete! Checkpoints saved to: %s", output_dir / "checkpoints")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MedFusion with MedSyn dataset")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    train(args.config)
