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
from ..loss import bvae_loss, kl_gaussians_freebits
from ..metrics import EpochAverager, ClasswiseAverager, make_batch_metrics_dict, make_classwise_metrics_dict
from ..training_logging import CSVTrainingLogger

logger = logging.getLogger(__name__)

# --------------------------- Weight tracking helpers --------------------------

def compute_weight_norms(model: torch.nn.Module) -> dict:
    """Compute weight norms for encoder vs decoder for tracking imbalance."""
    enc_norm = 0.0
    dec_norm = 0.0

    # Encoder: all conv/linear before fc_mu and fc_logv
    for name, param in model.named_parameters():
        if "enc" in name or "fc_mu" in name or "fc_logv" in name:
            enc_norm += float(param.detach().norm(2).cpu()) ** 2
        elif "dec" in name or "out" in name or "fc_dec" in name:
            dec_norm += float(param.detach().norm(2).cpu()) ** 2

    return {
        "enc_weight_norm": enc_norm ** 0.5,
        "dec_weight_norm": dec_norm ** 0.5,
    }

def compute_weight_updates_ratio(model: torch.nn.Module, old_params: dict) -> float:
    """Compute ||Δw|| / ||w|| to check if learning is effective."""
    delta_norm = 0.0
    w_norm = 0.0
    for name, param in model.named_parameters():
        if name in old_params:
            delta = param.detach() - old_params[name]
            delta_norm += float(delta.norm(2).cpu()) ** 2
            w_norm += float(param.detach().norm(2).cpu()) ** 2
    delta_norm = delta_norm ** 0.5
    w_norm = w_norm ** 0.5
    return delta_norm / max(w_norm, 1e-8)

# --------------------------- BetaScheduler ------------------------------------

class BetaScheduler:
    """Scheduler para β con warm-up monotónico o cíclico."""
    def __init__(self, cfg, steps_total: int):
        s = cfg.loss.beta_schedule
        self.typ = s.type
        self.cycles = s.cycles
        self.ratio = s.ratio_increase
        self.bmin = s.beta_min
        self.bmax = s.beta_max
        self.steps_total = steps_total
        self.steps_cycle = max(1, steps_total // max(1, self.cycles)) if self.typ == "cyclical" else steps_total

    def value(self, step: int) -> float:
        if self.typ == "monotonic":
            # 40% warm-up por defecto
            inc = int(0.4 * self.steps_total)
            t = min(step, inc)
            return self.bmin + (self.bmax - self.bmin) * (t / max(1, inc))
        # cyclical: sube en la fracción ratio de cada ciclo y se mantiene
        in_cycle = step % self.steps_cycle
        inc = int(self.ratio * self.steps_cycle)
        if in_cycle < inc:
            return self.bmin + (self.bmax - self.bmin) * (in_cycle / max(1, inc))
        return self.bmax

class CapacityScheduler:
    """Linear C(t) from 0 to C_max over `steps_to_max`."""
    def __init__(self, c_max: float, steps_to_max: int):
        self.c_max = float(c_max)
        self.steps_to_max = max(1, int(steps_to_max))
    def value(self, step: int) -> float:
        t = min(step, self.steps_to_max)
        return self.c_max * (t / self.steps_to_max)

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
        use_class_prior=m.use_class_prior,
        decoder_conditioning=getattr(m, "decoder_conditioning", False),
    )

def _build_optimizer(cfg, model: torch.nn.Module):
    ocfg = cfg.optim
    # group params: encoder, decoder, prior
    prior_params = []
    if getattr(model, "use_class_prior", False):
        if hasattr(model, "prior_mu"):   prior_params += list(model.prior_mu.parameters())
        if hasattr(model, "prior_logv"): prior_params += list(model.prior_logv.parameters())
    # Use id-based filtering to avoid hashing issues
    prior_ids = set(id(p) for p in prior_params)
    enc_params = [p for n,p in model.named_parameters()
                  if (n.startswith("enc") or n.startswith("fc_mu") or n.startswith("fc_logv")) and id(p) not in prior_ids]
    dec_params = [p for n,p in model.named_parameters()
                  if (n.startswith("dec") or n.startswith("out") or n.startswith("fc_dec")) and id(p) not in prior_ids]
    groups = [
        {"params": enc_params, "lr": ocfg.lr_init,        "weight_decay": ocfg.weight_decay},
        {"params": dec_params, "lr": ocfg.lr_init * 0.5,  "weight_decay": ocfg.weight_decay},  # slower decoder
        {"params": prior_params, "lr": ocfg.lr_init * 0.1,"weight_decay": ocfg.weight_decay},
    ]
    if ocfg.optimizer == "adamw":
        return optim.AdamW(groups, betas=ocfg.betas, eps=ocfg.eps)
    return optim.Adam(groups, betas=ocfg.betas, eps=ocfg.eps)

def _build_scheduler(cfg, optimizer, steps_per_epoch: int):
    sc = cfg.sched
    if not sc.use_onecycle:
        return None
    # Safety cap for OneCycleLR peak to avoid early KL blow-up
    safe_max = min(sc.max_lr, 3e-4)
    if safe_max < sc.max_lr:
        max_lr_str = f"{sc.max_lr:.2e}" if sc.max_lr is not None else "N/A"
        safe_max_str = f"{safe_max:.2e}" if safe_max is not None else "N/A"
        logger.warning(f"OneCycleLR max_lr capped from {max_lr_str} to {safe_max_str} for stability")
    return OneCycleLR(
        optimizer,
        max_lr=safe_max,
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
    # Separate encoder-only optimizer for lagging-inference mitigation
    enc_params = [p for n,p in model.named_parameters()
                  if n.startswith("enc") or n.startswith("fc_mu") or n.startswith("fc_logv")]
    enc_opt = optim.Adam(enc_params, lr=cfg.optim.lr_init, betas=cfg.optim.betas, eps=cfg.optim.eps)
    sched = _build_scheduler(cfg, opt, steps_per_epoch=len(loaders.train))

    # AMP - use the new torch.amp.GradScaler API (torch>=2.0)
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val = float("inf")

    # Prepare extra fields for per-class metrics + telemetry
    extra_fields = ["capacity_t", "grad_norm", "prior_mu_l2", "prior_logv_mean",
                    "mu_abs_mean", "post_var_mean"]
    for k in range(cfg.model.num_classes):
        for mname in ("psnr","recon","kld","kld_prior","count"):
            extra_fields.append(f"{mname}_c{k}")

    csv_logger = CSVTrainingLogger(str(output_dir / "training_metrics.csv"), extra_fields=extra_fields)

    # Initialize BetaScheduler
    steps_total = cfg.train.epochs * len(loaders.train)
    beta_sched = BetaScheduler(cfg, steps_total)
    cap_cfg = cfg.loss.capacity
    cap_sched = CapacityScheduler(cap_cfg.C_max, cap_cfg.steps_to_max) if cap_cfg.use else None
    global_step = 0

    # Flag to suppress scheduler warning on first step
    first_step = True

    # Save initial weights for tracking updates
    old_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Two-phase training: freeze class prior initially to stabilize decoder
    freeze_prior_epochs = 5

    for epoch in range(cfg.train.epochs):
        # Congelar prior por clase al inicio para estabilizar el decoder
        if getattr(model, "use_class_prior", False):
            req = epoch >= freeze_prior_epochs
            for p in list(model.prior_mu.parameters()) + list(model.prior_logv.parameters()):
                p.requires_grad = req
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

            # Get current beta
            curr_beta = beta_sched.value(global_step)
            global_step += 1

            opt.zero_grad(set_to_none=True)

            # Get current capacity for logging
            C_t = (cap_sched.value(global_step) if cap_sched is not None else 0.0)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x, y)
                mu_p, logv_p = model.prior_params(y)
                losses = bvae_loss(
                    out["x_hat"], x, out["mu"], out["logv"],
                    cfg.loss.recon_type, curr_beta, cfg.loss.recon_weight, cfg.loss.kld_weight,
                    mu_p=mu_p, logv_p=logv_p,
                    free_bits_nats=cfg.loss.free_bits_nats,
                    l1_weight=cfg.loss.l1_weight,
                    ssim_weight=cfg.loss.ssim_weight,
                    infommd_use=cfg.loss.infovae_mmd.use,
                    infommd_w=cfg.loss.infovae_mmd.weight,
                    capacity=(C_t if cap_sched is not None else None),
                    capacity_gamma=cap_cfg.gamma if cap_sched is not None else 0.0,
                    prior_reg_w=cfg.loss.prior_reg_w,
                    per_class_mmd_use=getattr(cfg.loss, 'per_class_mmd_use', False),
                    per_class_mmd_w=getattr(cfg.loss, 'per_class_mmd_w', 1.0),
                    y=y
                )
            # NaN/Inf guard
            if torch.isnan(losses["loss"]) or torch.isinf(losses["loss"]):
                logger.warning("NaN/Inf loss detected. Skipping step.")
                opt.zero_grad(set_to_none=True)
                continue
            bm = make_batch_metrics_dict(
                loss_total=losses["loss"], loss_recon=losses["recon"], loss_kld=losses["kld"],
                x_hat=out["x_hat"], x=x, mu=out["mu"], logv=out["logv"], latent_dim=cfg.model.latent_dim,
                mu_p=mu_p, logv_p=logv_p
            )
            bm["beta"] = float(curr_beta)
            bm["capacity_t"] = float(C_t)
            bm["mu_abs_mean"] = float(out["mu"].detach().abs().mean().cpu())
            bm["post_var_mean"] = float(out["logv"].detach().exp().mean().cpu())

            cm = make_classwise_metrics_dict(out["x_hat"], x, out["mu"], out["logv"], y, cfg.model.num_classes, mu_p, logv_p)

            scaler.scale(losses["loss"]).backward()

            # FIXED: Compute gradient norm and prior stats BEFORE updating averager
            # This ensures telemetry is included in the logged metrics
            total_gn = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = float(p.grad.detach().norm(2))
                    total_gn += param_norm * param_norm
            total_gn = total_gn ** 0.5
            bm["grad_norm"] = total_gn

            # Prior statistics
            if getattr(model, "use_class_prior", False):
                bm["prior_mu_l2"] = float(model.prior_mu.weight.detach().norm(2).cpu())
                bm["prior_logv_mean"] = float(model.prior_logv.weight.detach().mean().cpu())
            else:
                bm["prior_mu_l2"] = 0.0
                bm["prior_logv_mean"] = 0.0

            # Now update averagers with complete metrics
            train_avg.update(bm, batch_size=x.size(0))
            train_cavg.update(cm)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(opt)
            scaler.update()

            # Lagging-inference mitigation: extra encoder-only updates on RECON loss
            # Do NOT minimize KL here; that collapses q(z|x,y) -> p(z|y)
            if (cfg.train.encoder_extra_steps > 0 and
                cfg.train.encoder_heavy_epochs > 0 and
                epoch < cfg.train.encoder_heavy_epochs):
                for _ in range(cfg.train.encoder_extra_steps):
                    enc_opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        out_enc = model(x, y)
                        # Recon-only objective encourages informative codes without pushing to the prior
                        from ..loss import recon_loss
                        enc_loss = recon_loss(out_enc["x_hat"], x, cfg.loss.recon_type,
                                              l1_w=cfg.loss.l1_weight, ssim_w=cfg.loss.ssim_weight)
                    scaler.scale(enc_loss).backward()
                    torch.nn.utils.clip_grad_norm_(enc_params, cfg.train.grad_clip_norm)
                    scaler.step(enc_opt)
                    scaler.update()

            for k in running:
                running[k] += float(losses[k].detach())

            # Step scheduler after optimizer (skip warning on first iteration)
            if sched is not None:
                if first_step:
                    # On first step, step the scheduler to initialize properly
                    first_step = False
                sched.step()

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{losses['loss']:.4f}",
                            "recon": f"{losses['recon']:.4f}",
                            "kld": f"{losses['kld']:.4f}"})

        pbar.close()
        n = len(loaders.train)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_beta = cfg.loss.beta_schedule.beta_max  # β used only when capacity disabled
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for x, y in loaders.val:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x, y)
                mu_p, logv_p = model.prior_params(y)
                l = bvae_loss(
                    out["x_hat"], x, out["mu"], out["logv"],
                    cfg.loss.recon_type, val_beta, cfg.loss.recon_weight, cfg.loss.kld_weight,
                    mu_p=mu_p, logv_p=logv_p,
                    free_bits_nats=cfg.loss.free_bits_nats,
                    l1_weight=cfg.loss.l1_weight,
                    ssim_weight=cfg.loss.ssim_weight,
                    infommd_use=cfg.loss.infovae_mmd.use,
                    infommd_w=cfg.loss.infovae_mmd.weight,
                    capacity=(cap_cfg.C_max if cap_cfg.use else None),
                    capacity_gamma=cap_cfg.gamma if cap_cfg.use else 0.0,
                    prior_reg_w=cfg.loss.prior_reg_w,
                    per_class_mmd_use=getattr(cfg.loss, 'per_class_mmd_use', False),
                    per_class_mmd_w=getattr(cfg.loss, 'per_class_mmd_w', 1.0),
                    y=y
                )
                loss_val = float(l["loss"])
                # Safeguard against NaN/Inf
                if not (torch.isnan(l["loss"]) or torch.isinf(l["loss"])):
                    val_loss += loss_val
                else:
                    logger.warning(f"NaN/Inf detected in validation batch, skipping")
        val_loss /= max(1, len(loaders.val))
        logger.info(f"val_loss={val_loss:.4f}")

        # Aggregate validation metrics too
        val_avg = EpochAverager()
        val_cavg = ClasswiseAverager()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for x, y in loaders.val:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                o = model(x, y)
                mup, lvp = model.prior_params(y)
                l = bvae_loss(
                    o["x_hat"], x, o["mu"], o["logv"],
                    cfg.loss.recon_type, val_beta, cfg.loss.recon_weight, cfg.loss.kld_weight,
                    mu_p=mup, logv_p=lvp,
                    free_bits_nats=cfg.loss.free_bits_nats,
                    l1_weight=cfg.loss.l1_weight,
                    ssim_weight=cfg.loss.ssim_weight,
                    infommd_use=cfg.loss.infovae_mmd.use,
                    infommd_w=cfg.loss.infovae_mmd.weight,
                    capacity=(cap_cfg.C_max if cap_cfg.use else None),
                    capacity_gamma=cap_cfg.gamma if cap_cfg.use else 0.0,
                    prior_reg_w=cfg.loss.prior_reg_w,
                    per_class_mmd_use=getattr(cfg.loss, 'per_class_mmd_use', False),
                    per_class_mmd_w=getattr(cfg.loss, 'per_class_mmd_w', 1.0),
                    y=y
                )
                bm = make_batch_metrics_dict(
                    loss_total=l["loss"], loss_recon=l["recon"], loss_kld=l["kld"],
                    x_hat=o["x_hat"], x=x, mu=o["mu"], logv=o["logv"], latent_dim=cfg.model.latent_dim,
                    mu_p=mup, logv_p=lvp
                )
                bm["beta"] = float(val_beta)
                bm["capacity_t"] = float(cap_cfg.C_max if cap_cfg.use else 0.0)
                bm["grad_norm"] = 0.0  # Not computed in validation
                # Prior statistics
                if getattr(model, "use_class_prior", False):
                    bm["prior_mu_l2"] = float(model.prior_mu.weight.detach().norm(2).cpu())
                    bm["prior_logv_mean"] = float(model.prior_logv.weight.detach().mean().cpu())
                else:
                    bm["prior_mu_l2"] = 0.0
                    bm["prior_logv_mean"] = 0.0
                val_avg.update(bm, batch_size=x.size(0))

                cm = make_classwise_metrics_dict(o["x_hat"], x, o["mu"], o["logv"], y, cfg.model.num_classes, mup, lvp)
                val_cavg.update(cm)

        # LR for logging
        curr_lr = next(iter(opt.param_groups))["lr"]

        # Compute weight norms and update ratio
        weight_norms = compute_weight_norms(model)
        weight_updates = compute_weight_updates_ratio(model, old_params)

        # Update old params for next epoch
        old_params = {name: param.detach().clone() for name, param in model.named_parameters()}

        # Write CSV rows with per-class metrics + weight tracking
        train_row = train_avg.means() | train_cavg.means(cfg.model.num_classes) | weight_norms | {"weight_updates_ratio": weight_updates}
        val_row = val_avg.means() | val_cavg.means(cfg.model.num_classes) | weight_norms | {"weight_updates_ratio": weight_updates}

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

        # ---- Generate samples for tracking progress ----
        # Save 3 types of samples every 5 epochs: prior samples, reconstructions, posterior samples
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                # 1. Prior samples (4 per class with fixed seed for reproducibility)
                torch.manual_seed(42)
                ys_prior = torch.repeat_interleave(torch.arange(cfg.model.num_classes, device=device), 4)
                prior_samples = model.sample(n=ys_prior.numel(), y=ys_prior, device=device, tau=0.7)
                save_image(prior_samples, output_dir / "samples" / f"epoch_{epoch+1:04d}_prior.png",
                          nrow=4, normalize=False)

                # 2. Reconstructions from first validation batch
                val_iter = iter(loaders.val)
                x_val, y_val = next(val_iter)
                x_val, y_val = x_val[:cfg.model.num_classes*4].to(device), y_val[:cfg.model.num_classes*4].to(device)
                out_val = model(x_val, y_val)
                x_recon = out_val["x_hat"]
                # Interleave originals and reconstructions
                comparison = torch.stack([x_val, x_recon], dim=1).flatten(0, 1)
                save_image(comparison, output_dir / "samples" / f"epoch_{epoch+1:04d}_recon.png",
                          nrow=8, normalize=False)

                # 3. Posterior samples (sample from q(z|x) then decode)
                mu_q, logv_q = model.encode(x_val, y_val)
                z_posterior = model.reparameterize(mu_q, logv_q)
                x_posterior = model.decode(z_posterior, y_val)
                if model.decoder_sigmoid:
                    x_posterior = x_posterior.clamp(0, 1)
                save_image(x_posterior, output_dir / "samples" / f"epoch_{epoch+1:04d}_posterior.png",
                          nrow=8, normalize=False)
            model.train()
        else:
            # Quick single sample per class for other epochs
            with torch.no_grad():
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
    test_beta = cfg.loss.beta_schedule.beta_max
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for x, y in loaders.test:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            o = model(x, y)
            mup, lvp = model.prior_params(y)
            l = bvae_loss(
                o["x_hat"], x, o["mu"], o["logv"],
                cfg.loss.recon_type, test_beta, cfg.loss.recon_weight, cfg.loss.kld_weight,
                mu_p=mup, logv_p=lvp,
                free_bits_nats=cfg.loss.free_bits_nats,
                l1_weight=cfg.loss.l1_weight,
                ssim_weight=cfg.loss.ssim_weight,
                infommd_use=cfg.loss.infovae_mmd.use,
                infommd_w=cfg.loss.infovae_mmd.weight,
                capacity=(cap_cfg.C_max if cap_cfg.use else None),
                capacity_gamma=cap_cfg.gamma if cap_cfg.use else 0.0,
                prior_reg_w=cfg.loss.prior_reg_w,
                per_class_mmd_use=getattr(cfg.loss, 'per_class_mmd_use', False),
                per_class_mmd_w=getattr(cfg.loss, 'per_class_mmd_w', 1.0),
                y=y
            )
            bm = make_batch_metrics_dict(
                loss_total=l["loss"], loss_recon=l["recon"], loss_kld=l["kld"],
                x_hat=o["x_hat"], x=x, mu=o["mu"], logv=o["logv"], latent_dim=cfg.model.latent_dim,
                mu_p=mup, logv_p=lvp
            )
            bm["beta"] = float(test_beta)
            bm["capacity_t"] = float(cap_cfg.C_max if cap_cfg.use else 0.0)
            bm["grad_norm"] = 0.0  # Not computed in test
            # Prior statistics
            if getattr(model, "use_class_prior", False):
                bm["prior_mu_l2"] = float(model.prior_mu.weight.detach().norm(2).cpu())
                bm["prior_logv_mean"] = float(model.prior_logv.weight.detach().mean().cpu())
            else:
                bm["prior_mu_l2"] = 0.0
                bm["prior_logv_mean"] = 0.0
            test_avg.update(bm, batch_size=x.size(0))
            cm = make_classwise_metrics_dict(o["x_hat"], x, o["mu"], o["logv"], y, cfg.model.num_classes, mup, lvp)
            test_cavg.update(cm)

    # Write test metrics to CSV (with final weight norms)
    final_weight_norms = compute_weight_norms(model)
    test_row = test_avg.means() | test_cavg.means(cfg.model.num_classes) | final_weight_norms | {"weight_updates_ratio": 0.0}
    csv_logger.log_epoch(epoch=cfg.train.epochs, split="test", lr=0.0, metrics=test_row)

    test_loss = test_row.get("loss", 0.0)
    test_psnr = test_row.get("psnr", 0.0)
    logger.info("Test evaluation: loss=%.4f psnr=%.2f", test_loss, test_psnr)
