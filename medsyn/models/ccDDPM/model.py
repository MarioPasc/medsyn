# medsyn/models/ccDDPM/model.py
# Purpose: Class-conditional DDPM using Diffusers' UNet2DModel.
# Conditioning: learned class embedding broadcast to HxW and concatenated to input.
# Classifier-free guidance: support null label -> zero embedding during training.
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel

class ClassEmbedder(nn.Module):
    """
    Map integer class labels to a channel vector that is broadcast spatially.
    If labels is None, returns zeros for unconditional path (cfg).
    """
    def __init__(self, num_classes: int, emb_channels: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_channels)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.emb_channels = emb_channels

    def forward(self, labels: Optional[torch.Tensor], shape_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
        if labels is None:
            return torch.zeros((1, self.emb_channels, *shape_hw), device=device)
        v = self.emb(labels)  # [B, C_emb]
        v = v[..., None, None]  # [B, C_emb, 1, 1]
        return v.expand(-1, -1, shape_hw[0], shape_hw[1])

@dataclass
class CCDDPMInit:
    in_channels: int = 3
    class_embed_dim: int = 16
    num_classes: int = 9
    model_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0

class CCDDPM(nn.Module):
    """
    Wrapper around UNet2DModel with class-channel concatenation.
    """
    def __init__(self, cfg: CCDDPMInit):
        super().__init__()
        self.class_embed = ClassEmbedder(cfg.num_classes, cfg.class_embed_dim)
        self.unet = UNet2DModel(
            sample_size=None,  # flexible
            in_channels=cfg.in_channels + cfg.class_embed_dim,
            out_channels=cfg.in_channels,
            block_out_channels=tuple(cfg.model_channels * m for m in cfg.channel_mult),
            layers_per_block=cfg.num_res_blocks,
            down_block_types=("DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D","UpBlock2D","UpBlock2D","UpBlock2D"),
            add_attention=True,
            dropout=cfg.dropout,
        )

    def forward(self, x_noisy: torch.Tensor, timesteps: torch.Tensor, class_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass predicting noise ε_θ(x_t, t, y).
        """
        B, _, H, W = x_noisy.shape
        cond = self.class_embed(class_labels, (H, W), x_noisy.device)
        x_in = torch.cat([x_noisy, cond], dim=1)
        return self.unet(x_in, timesteps).sample
