# medsyn/models/bVAE/engine/predict.py
# Purpose: Load a trained β-VAE and generate synthetic images to disk.

from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import torch
from torchvision.utils import save_image

from ..config import load_bvae_config
from ..model import BetaVAE

logger = logging.getLogger(__name__)

def load_model_from_ckpt(cfg_path: str, ckpt_path: str) -> tuple[BetaVAE, torch.device, Path]:
    cfg = load_bvae_config(cfg_path)
    device = torch.device(cfg.train.device)
    model = BetaVAE(
        in_channels=cfg.model.in_channels,
        img_size=cfg.model.img_size,
        latent_dim=cfg.model.latent_dim,
        base_channels=cfg.model.base_channels,
        num_down=cfg.model.num_down,
        decoder_sigmoid=(cfg.loss.recon_type == "bce") or cfg.model.decoder_sigmoid,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    out_dir = Path(cfg.train.output_dir) / "samples_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    return model, device, out_dir

@torch.no_grad()
def generate(cfg_path: str, ckpt_path: Optional[str] = None, n: int = 1000, grid: bool = True) -> Path:
    cfg = load_bvae_config(cfg_path)
    ckpt = ckpt_path or str((Path(cfg.train.output_dir) / "ckpts" / "best.pt").resolve())
    model, device, out_dir = load_model_from_ckpt(cfg_path, ckpt)

    bs = 128
    saved = 0
    k = 0
    while saved < n:
        b = min(bs, n - saved)
        imgs = model.sample(b, device=device)
        for i in range(b):
            save_image(imgs[i], out_dir / f"synth_{k:06d}.png")
            k += 1
        saved += b

    if grid:
        grid_img = (out_dir / "grid.png")
        save_image(model.sample(64, device=device), grid_img, nrow=8)
    logger.info("Generated %d images -> %s", n, out_dir)
    return out_dir
