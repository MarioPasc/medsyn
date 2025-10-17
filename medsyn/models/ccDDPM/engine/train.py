# medsyn/models/ccDDPM/engine/train.py
# Purpose: Training loop for class-conditioned DDPM with Diffusers' DDPMScheduler.
# Features: mixed precision, EMA, classifier-free guidance (label drop), checkpointing, CSV logs.
from __future__ import annotations
from pathlib import Path
from typing import Optional
import csv
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from medsyn.models.ccDDPM.config import ProjectCfg, load_cfg
from medsyn.models.ccDDPM.dataloader import build_loader
from medsyn.models.ccDDPM.model import CCDDPM, CCDDPMInit
from medsyn.models.ccDDPM.loss import DDPMNoiseMSE

logger = logging.getLogger("medsyn.ccddpm.train")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

def _maybe_drop_labels(labels: torch.Tensor, p: float) -> Optional[torch.Tensor]:
    if p <= 0:
        return labels
    mask = torch.rand_like(labels.float()) < p
    out = labels.clone()
    out[mask] = 0  # value unused
    # Return None for unconditional; model handles None as zeros
    return None if mask.all() else out

def train(yaml_path: str, split: str = "train") -> None:
    """
    Train ccDDPM using config at yaml_path. Saves checkpoints and CSV log.
    """
    cfg: ProjectCfg = load_cfg(yaml_path, split=split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg.ccddpm.train
    scfg = cfg.ccddpm.sched
    ocfg = cfg.ccddpm.optim

    # Data
    train_loader = build_loader(cfg.data_index_json, "train", tcfg.image_size, tcfg.batch_size, tcfg.num_workers, normalize=True)

    # Model
    mcfg = CCDDPMInit(
        in_channels=tcfg.in_channels,
        class_embed_dim=tcfg.class_embed_dim,
        num_classes=tcfg.num_classes,
    )
    model = CCDDPM(mcfg).to(device)

    # Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=False,
    )

    # Optim
    opt = optim.AdamW(model.parameters(), lr=ocfg.lr, betas=ocfg.betas, eps=ocfg.eps, weight_decay=ocfg.wd)
    scaler = GradScaler(enabled=tcfg.mixed_precision)
    ema = EMA(model, decay=tcfg.ema_decay) if tcfg.ema_use else None
    loss_fn = DDPMNoiseMSE(num_classes=tcfg.num_classes)

    out_dir = Path(tcfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "train_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch","step","loss","lr","time_s"])

    global_step = 0
    model.train()
    for epoch in range(1, tcfg.epochs + 1):
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            x0 = batch["pixel_values"].to(device)  # [-1,1]
            labels = batch["labels"].to(device)

            # sample t and noise
            bsz = x0.size(0)
            t = torch.randint(0, scfg.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = noise_scheduler.add_noise(x0, noise, t) # type: ignore

            # classifier-free: drop labels with prob p
            class_labels = labels.clone()
            if tcfg.guidance_p_uncond > 0:
                drop_mask = torch.rand(bsz, device=device) < tcfg.guidance_p_uncond
                if drop_mask.any():
                    class_labels[drop_mask] = 0  # value unused
            # forward
            with autocast(enabled=tcfg.mixed_precision):
                pred = model(x_t, t, class_labels if tcfg.guidance_p_uncond < 1.0 else None)
                loss = loss_fn(pred, noise, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg.grad_clip_norm:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            if ema:
                ema.update(model)

            global_step += 1
            if global_step % tcfg.log_every == 0:
                with open(csv_path, "a", newline="") as fh:
                    csv.writer(fh).writerow([epoch, global_step, float(loss.item()), opt.param_groups[0]["lr"], round(time.time()-t0, 2)])
                logger.info("ep=%d step=%d loss=%.4f", epoch, global_step, float(loss.item()))

        # checkpoint
        if (epoch % tcfg.ckpt_every_epochs) == 0:
            ck = out_dir / f"ccddpm_ep{epoch}.pt"
            to_save = {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "ema": (ema.shadow if ema else None), "cfg": tcfg.__dict__}
            torch.save(to_save, ck)
            logger.info("Saved %s", ck)
