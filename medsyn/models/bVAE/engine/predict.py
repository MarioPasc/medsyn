from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import logging
import torch
from torchvision.utils import save_image

from ..config import load_bvae_config
from ..model import ConditionalBetaVAE

logger = logging.getLogger(__name__)

def load_model_from_ckpt(cfg_path: str, ckpt_path: str) -> tuple[ConditionalBetaVAE, torch.device, Path]:
    cfg = load_bvae_config(cfg_path)
    device = torch.device(cfg.train.device)
    model = ConditionalBetaVAE(
        in_channels=cfg.model.in_channels, img_size=cfg.model.img_size, latent_dim=cfg.model.latent_dim,
        base_channels=cfg.model.base_channels, num_down=cfg.model.num_down,
        num_classes=cfg.model.num_classes, conditioning=cfg.model.conditioning,
        class_embed_dim=cfg.model.class_embed_dim,
        decoder_sigmoid=(cfg.loss.recon_type == "bce") or cfg.model.decoder_sigmoid,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    out_dir = Path(cfg.train.output_dir) / "samples_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    return model, device, out_dir

@torch.no_grad()
def generate_for_classes(cfg_path: str, class_ids: Sequence[int], per_class: int = 100, ckpt_path: Optional[str] = None) -> Path:
    cfg = load_bvae_config(cfg_path)
    ckpt = ckpt_path or str((Path(cfg.train.output_dir) / "ckpts" / "best.pt").resolve())
    model, device, out_dir = load_model_from_ckpt(cfg_path, ckpt)

    k = 0
    for c in class_ids:
        y = torch.full((per_class,), int(c), dtype=torch.long, device=device)
        imgs = model.sample(per_class, y=y, device=device)
        for i in range(per_class):
            save_image(imgs[i], out_dir / f"class{c}_synth_{k:06d}.png")
            k += 1
    logger.info("Generated %d classes x %d imgs -> %s", len(class_ids), per_class, out_dir)
    return out_dir
