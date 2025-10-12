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
from tqdm import tqdm

from ..config import load_bvae_config
from ..dataloader import make_loaders
from ..model import ConditionalBetaVAE
from ..loss import bvae_loss
from ..metrics import EpochAverager, ClasswiseAverager, make_batch_metrics_dict, make_classwise_metrics_dict
from ..training_logging import CSVTrainingLogger

logger = logging.getLogger(__name__)

def _build_model(cfg) -> ConditionalBetaVAE:
    m = cfg.model
    return ConditionalBetaVAE(
        in_channels=m.in_channels,
        img_size=m.img_size,
        latent_dim=m.latent_dim,
        base_channels=m.base_channels,
        num_down=m.num_down,
        num_classes=m.num_classes,
        conditioning=m.conditioning,
        class_embed_dim=m.class_embed_dim,
        decoder_sigmoid=(cfg.loss.recon_type == "bce") or m.decoder_sigmoid,
    )

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
    output_dir = Path(cfg.train.output_dir)
    (output_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)

    # Data
    loaders = make_loaders(cfg.data.index_json, cfg.model.img_size, cfg.train.batch_size, cfg.train.num_workers, cfg.train.seed)

    # Model / Opt / Sched
    device = torch.device(cfg.train.device)
    model = _build_model(cfg).to(device)
    opt = _build_optimizer(cfg, model)
    sched = _build_scheduler(cfg, opt, steps_per_epoch=len(loaders.train))

    # AMP - use the new torch.amp.GradScaler API (torch>=2.0)
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val = float("inf")

    # Prepare extra fields for per-class metrics
    extra_fields = []
    for k in range(cfg.model.num_classes):
        for mname in ("psnr","recon","kld","count"):
            extra_fields.append(f"{mname}_c{k}")

    csv_logger = CSVTrainingLogger(str(output_dir / "training_metrics.csv"), extra_fields=extra_fields)
    for epoch in range(cfg.train.epochs):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kld": 0.0}
        train_avg = EpochAverager()
        train_cavg = ClasswiseAverager()

        # Progress bar for the training epoch
        pbar = tqdm(enumerate(loaders.train, 1), total=len(loaders.train),
                    desc=f"Epoch {epoch+1}/{cfg.train.epochs}",
                    unit="batch", leave=True)

        for step, (x, y) in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x, y)
                losses = bvae_loss(out["x_hat"], x, out["mu"], out["logv"],
                                   cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
            bm = make_batch_metrics_dict(
                loss_total=losses["loss"], loss_recon=losses["recon"], loss_kld=losses["kld"],
                x_hat=out["x_hat"], x=x, mu=out["mu"], logv=out["logv"], latent_dim=cfg.model.latent_dim
            )
            train_avg.update(bm, batch_size=x.size(0))

            cm = make_classwise_metrics_dict(out["x_hat"], x, out["mu"], out["logv"], y, cfg.model.num_classes)
            train_cavg.update(cm)

            scaler.scale(losses["loss"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

            for k in running:
                running[k] += float(losses[k])

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{losses['loss']:.4f}",
                            "recon": f"{losses['recon']:.4f}",
                            "kld": f"{losses['kld']:.4f}"})

        pbar.close()
        n = len(loaders.train)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for x, y in loaders.val:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x, y)
                l = bvae_loss(out["x_hat"], x, out["mu"], out["logv"],
                              cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
                val_loss += float(l["loss"])
        val_loss /= max(1, len(loaders.val))
        logger.info(f"val_loss={val_loss:.4f}")

        # Aggregate validation metrics too
        val_avg = EpochAverager()
        val_cavg = ClasswiseAverager()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for x, y in loaders.val:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                o = model(x, y)
                l = bvae_loss(o["x_hat"], x, o["mu"], o["logv"],
                            cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
                bm = make_batch_metrics_dict(
                    loss_total=l["loss"], loss_recon=l["recon"], loss_kld=l["kld"],
                    x_hat=o["x_hat"], x=x, mu=o["mu"], logv=o["logv"], latent_dim=cfg.model.latent_dim
                )
                val_avg.update(bm, batch_size=x.size(0))

                cm = make_classwise_metrics_dict(o["x_hat"], x, o["mu"], o["logv"], y, cfg.model.num_classes)
                val_cavg.update(cm)

        # LR for logging
        curr_lr = next(iter(opt.param_groups))["lr"]

        # Write CSV rows with per-class metrics
        train_row = train_avg.means() | train_cavg.means(cfg.model.num_classes)
        val_row = val_avg.means() | val_cavg.means(cfg.model.num_classes)

        csv_logger.log_epoch(epoch=epoch, split="train", lr=curr_lr, metrics=train_row)
        csv_logger.log_epoch(epoch=epoch, split="val", lr=curr_lr, metrics=val_row)

        # Print performance metrics at the end of each epoch
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{cfg.train.epochs} Summary:")
        print(f"{'='*80}")
        print(f"Training:")
        print(f"  Total Loss: {running['loss']/n:.4f}")
        print(f"  Recon Loss: {running['recon']/n:.4f}")
        print(f"  KLD Loss:   {running['kld']/n:.4f}")
        print(f"  PSNR:       {train_row.get('psnr', 0.0):.2f} dB")
        print(f"\nValidation:")
        print(f"  Total Loss: {val_loss:.4f}")
        print(f"  PSNR:       {val_row.get('psnr', 0.0):.2f} dB")
        print(f"  Learning Rate: {curr_lr:.6f}")
        print(f"{'='*80}\n")


        # ---- Checkpointing ----
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            _save_ckpt({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "val_loss": val_loss},
                       output_dir / "ckpts" / "best.pt")
        _save_ckpt({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "val_loss": val_loss},
                   output_dir / "ckpts" / f"last.pt")

        # ---- Periodic checkpoint saving every X epochs ----
        if (epoch + 1) % cfg.train.save_every_epoch == 0:
            _save_ckpt({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "val_loss": val_loss},
                       output_dir / "ckpts" / f"epoch_{epoch+1:04d}.pt")
            logger.info(f"Saved checkpoint at epoch {epoch+1}")

        # ---- Generate 1 image per label for tracking progress ----
        with torch.no_grad():
            n_per_class = 1
            ys = torch.arange(cfg.model.num_classes, device=device)
            samples = model.sample(n=ys.numel(), y=ys, device=device)
            save_image(samples, output_dir / "samples" / f"epoch_{epoch+1:04d}_progress.png", nrow=cfg.model.num_classes)

    logger.info("Training completed. Best val loss: %.4f", best_val)

    # ---- Final evaluation on test set with best model ----
    logger.info("Loading best checkpoint for final test evaluation...")
    best_ckpt = torch.load(output_dir / "ckpts" / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()

    test_avg = EpochAverager()
    test_cavg = ClasswiseAverager()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for x, y in loaders.test:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            o = model(x, y)
            l = bvae_loss(o["x_hat"], x, o["mu"], o["logv"],
                        cfg.loss.recon_type, cfg.loss.beta, cfg.loss.recon_weight, cfg.loss.kld_weight)
            bm = make_batch_metrics_dict(
                loss_total=l["loss"], loss_recon=l["recon"], loss_kld=l["kld"],
                x_hat=o["x_hat"], x=x, mu=o["mu"], logv=o["logv"], latent_dim=cfg.model.latent_dim
            )
            test_avg.update(bm, batch_size=x.size(0))
            cm = make_classwise_metrics_dict(o["x_hat"], x, o["mu"], o["logv"], y, cfg.model.num_classes)
            test_cavg.update(cm)

    # Write test metrics to CSV
    test_row = test_avg.means() | test_cavg.means(cfg.model.num_classes)
    csv_logger.log_epoch(epoch=cfg.train.epochs, split="test", lr=0.0, metrics=test_row)

    test_loss = test_row.get("loss", 0.0)
    test_psnr = test_row.get("psnr", 0.0)
    logger.info("Test evaluation: loss=%.4f psnr=%.2f", test_loss, test_psnr)
