# medsyn/models/bVAE/engine/train.py
# Purpose: Training loop with AMP, OneCycleLR, checkpointing, and periodic sampling.

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import logging
import torch
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

from ..config import load_bvae_config
from ..dataloader import make_loaders
from ..model import BetaVAE
from ..loss import bvae_loss
from ..metrics import EpochAverager, make_batch_metrics_dict
from ..training_logging import CSVTrainingLogger

logger = logging.getLogger(__name__)

def _build_model(cfg) -> BetaVAE:
    mcfg = cfg.model
    model = BetaVAE(
        in_channels=mcfg.in_channels,
        img_size=mcfg.img_size,
        latent_dim=mcfg.latent_dim,
        base_channels=mcfg.base_channels,
        num_down=mcfg.num_down,
        decoder_sigmoid=(cfg.loss.recon_type == "bce") or mcfg.decoder_sigmoid,
    )
    return model

def _build_optimizer(cfg, model: torch.nn.Module):
    ocfg = cfg.optim
    if ocfg.optimizer == "adamw":
        opt = optim.AdamW(model.parameters(), lr=ocfg.lr_init, betas=ocfg.betas, eps=ocfg.eps, weight_decay=ocfg.weight_decay)
    else:
        opt = optim.Adam(model.parameters(), lr=ocfg.lr_init, betas=ocfg.betas, eps=ocfg.eps, weight_decay=ocfg.weight_decay)
    return opt

def _build_scheduler(cfg, optimizer, steps_per_epoch: int):
    sc = cfg.sched
    if not sc.use_onecycle:
        return None
    return OneCycleLR(
        optimizer,
        max_lr=sc.max_lr,
        epochs=cfg.train.epochs,
        steps_per_epoch=max(1, steps_per_epoch),
        pct_start=sc.pct_start,
        div_factor=sc.div_factor,
        final_div_factor=sc.final_div_factor,
        anneal_strategy="cos",
        three_phase=False,
    )

def _save_ckpt(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def train(cfg_path: str) -> None:
    cfg = load_bvae_config(cfg_path)
    out = Path(cfg.train.output_dir)
    (out / "ckpts").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    # Data
    loaders = make_loaders(cfg.data.index_json, cfg.model.img_size, cfg.train.batch_size, cfg.train.num_workers, cfg.train.seed)

    # Model / Opt / Sched
    device = torch.device(cfg.train.device)
    model = _build_model(cfg).to(device)
    opt = _build_optimizer(cfg, model)
    sched = _build_scheduler(cfg, opt, steps_per_epoch=len(loaders.train))

    # AMP
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")

    csv_logger = CSVTrainingLogger(str(out / "training_metrics.csv"))
    for epoch in range(cfg.train.epochs):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
        train_avg = EpochAverager()
        for step, (x, _) in enumerate(loaders.train, 1):
            x = x.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                losses = bvae_loss(out["x_hat"], x, out["mu"], out["logv"],
                                   cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
            bm = make_batch_metrics_dict(
                loss_total=losses["loss"], loss_recon=losses["recon"], loss_kld=losses["kld"],
                x_hat=out["x_hat"], x=x, mu=out["mu"], logv=out["logv"], latent_dim=cfg.model.latent_dim
            )
            train_avg.update(bm, batch_size=x.size(0))
            scaler.scale(losses["loss"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

            for k in running:
                running[k] += float(losses[k])

        n = len(loaders.train)
        logger.info(f"epoch={epoch} loss={running['loss']/n:.4f} recon={running['recon']/n:.4f} kld={running['kld']/n:.4f}")

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for x, _ in loaders.val:
                x = x.to(device, non_blocking=True)
                out = model(x)
                l = bvae_loss(out["x_hat"], x, out["mu"], out["logv"],
                              cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
                val_loss += float(l["loss"])
        val_loss /= max(1, len(loaders.val))
        logger.info(f"val_loss={val_loss:.4f}")

        # Aggregate validation metrics too
        val_avg = EpochAverager()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for x, _ in loaders.val:
                x = x.to(device, non_blocking=True)
                o = model(x)
                l = bvae_loss(o["x_hat"], x, o["mu"], o["logv"],
                            cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
                bm = make_batch_metrics_dict(
                    loss_total=l["loss"], loss_recon=l["recon"], loss_kld=l["kld"],
                    x_hat=o["x_hat"], x=x, mu=o["mu"], logv=o["logv"], latent_dim=cfg.model.latent_dim
                )
                val_avg.update(bm, batch_size=x.size(0))

        # LR for logging
        curr_lr = next(iter(opt.param_groups))["lr"]

        # Write CSV rows
        csv_logger.log_epoch(epoch=epoch, split="train", lr=curr_lr, metrics=train_avg.means())
        csv_logger.log_epoch(epoch=epoch, split="val",   lr=curr_lr, metrics=val_avg.means())


        # ---- Checkpointing ----
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            _save_ckpt({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "val_loss": val_loss},
                       out / "ckpts" / "best.pt")
        _save_ckpt({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "val_loss": val_loss},
                   out / "ckpts" / f"last.pt")

        # ---- Periodic sampling ----
        with torch.no_grad():
            samples = model.sample(n=64, device=device)
            save_image(samples, out / "samples" / f"epoch_{epoch:04d}.png", nrow=8)

    logger.info("Training completed. Best val loss: %.4f", best_val)
