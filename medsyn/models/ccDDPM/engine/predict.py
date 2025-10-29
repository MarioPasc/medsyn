# medsyn/models/ccDDPM/engine/predict.py
# Purpose: Generate k samples for a target class i using a trained ccDDPM.
# Supports classifier-free guidance at inference via two forward passes.
from __future__ import annotations
from pathlib import Path
from typing import List
import math
import logging
import torch
from torchvision.utils import save_image, make_grid
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from medsyn.models.ccDDPM.config import load_cfg, ProjectCfg
from medsyn.models.ccDDPM.model import CCDDPM, CCDDPMInit

logger = logging.getLogger("medsyn.ccddpm.predict")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

@torch.no_grad()
def _cfg_step(model: CCDDPM, scheduler: DDPMScheduler, x: torch.Tensor, t: torch.Tensor, class_labels: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """
    Classifier-free guidance step.

    Args:
        guidance_scale:
            - 1.0: pure conditional (single forward pass)
            - >1.0: enhanced conditional (CFG formula with two passes)
            - <1.0: blend toward unconditional (rarely used)
    """
    eps_cond = model(x, t, class_labels)
    # Optimization: skip unconditional pass when scale=1.0 (pure conditional)
    if guidance_scale == 1.0:
        return eps_cond
    eps_uncond = model(x, t, None)
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

@torch.no_grad()
def generate(yaml_path: str, checkpoint: Path, class_id: int, k: int) -> List[Path]:
    """
    Load model and generate k images for class `class_id`. Returns list of file paths.
    """
    cfg: ProjectCfg = load_cfg(yaml_path, split="train")
    tcfg, scfg, icfg = cfg.ccddpm.train, cfg.ccddpm.sched, cfg.ccddpm.infer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    mcfg = CCDDPMInit(in_channels=tcfg.in_channels, class_embed_dim=tcfg.class_embed_dim, num_classes=tcfg.num_classes)
    model = CCDDPM(mcfg).to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])

    # scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=scfg.num_train_timesteps,
        beta_start=scfg.beta_start,
        beta_end=scfg.beta_end,
        beta_schedule=scfg.beta_schedule,
        prediction_type=scfg.prediction_type,
        clip_sample=True,
        clip_sample_range=1.0,
        thresholding=False,
    )
    scheduler.set_timesteps(icfg.num_inference_steps, device=device)

    bsz = min(k, 64)
    steps = math.ceil(k / bsz)
    out_dir = Path(icfg.out_dir).resolve() / f"class_{class_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for s in range(steps):
        cur = min(bsz, k - s * bsz)
        x = torch.randn((cur, tcfg.in_channels, tcfg.image_size, tcfg.image_size), device=device)
        labels = torch.full((cur,), int(class_id), device=device, dtype=torch.long)

        for t in scheduler.timesteps:
            # Move timestep to device to avoid device mismatch
            t_device = t.to(device)
            eps = _cfg_step(model, scheduler, x, t_device, labels, icfg.guidance_scale)
            x = scheduler.step(model_output=eps, timestep=t_device, sample=x).prev_sample # type: ignore

        # map from [-1,1] to [0,1]
        imgs = (x.clamp(-1, 1) + 1.0) / 2.0
        for i in range(cur):
            p = out_dir / f"class{class_id:02d}_{s*bsz+i:05d}.png"
            save_image(imgs[i], p)
            saved.append(p)

        if icfg.save_grid:
            grid = make_grid(imgs, nrow=int(math.sqrt(cur)))
            save_image(grid, out_dir / f"_grid_step{s}.png")

    logger.info("Generated %d images for class %d -> %s", k, class_id, out_dir)
    return saved
