# medsyn/models/ccDDPM/model.py
# Purpose: Class-conditional DDPM using Diffusers' UNet2DModel.
# Conditioning: learned class embedding broadcast to HxW and concatenated to input.
# Classifier-free guidance: support null label -> zero embedding during training.
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel
from medsyn.models.ccDDPM.config import CCDDPMInit

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
        # labels may contain -1 to indicate unconditional tokens (per-sample classifier-free guidance)
        mask_uncond = (labels == -1)
        # clamp to valid range for embedding lookup
        labels_clamped = labels.clamp_min(0)
        v = self.emb(labels_clamped)  # [B, C_emb]
        v[mask_uncond] = 0  # zero vectors for uncond rows
        v = v[..., None, None]  # [B, C_emb, 1, 1]
        return v.expand(-1, -1, shape_hw[0], shape_hw[1])

class CCDDPM(nn.Module):
    """
    Class-conditional DDPM wrapper around UNet2DModel.

    Class conditioning is implemented via channel concatenation:
    - Class labels are embedded to a vector of size `class_embed_dim`
    - The embedding is broadcast spatially and concatenated to the noisy image
    - UNet receives input of shape [B, in_channels + class_embed_dim, H, W]

    For classifier-free guidance during training, labels can be set to -1
    to indicate unconditional generation (zero embedding).
    """
    def __init__(self, cfg: CCDDPMInit):
        super().__init__()
        self.cfg = cfg
        self.class_embed = ClassEmbedder(cfg.num_classes, cfg.class_embed_dim)

        # Compute effective in_channels (image channels + class embedding channels)
        effective_in_channels = cfg.in_channels + cfg.class_embed_dim

        # Build UNet2DModel with full configuration
        self.unet = UNet2DModel(
            sample_size=None,  # Flexible spatial size
            in_channels=effective_in_channels,
            out_channels=cfg.in_channels,  # Output matches original image channels
            # Core architecture
            block_out_channels=cfg.get_block_out_channels(),
            layers_per_block=cfg.layers_per_block,
            down_block_types=cfg.down_block_types,
            up_block_types=cfg.up_block_types,
            # Attention
            add_attention=cfg.add_attention,
            attention_head_dim=cfg.attention_head_dim,
            # Normalization
            norm_num_groups=cfg.norm_num_groups,
            attn_norm_num_groups=cfg.attn_norm_num_groups,
            norm_eps=cfg.norm_eps,
            # Dropout
            dropout=cfg.dropout,
            # Time embedding
            time_embedding_type=cfg.time_embedding_type,
            freq_shift=cfg.freq_shift,
            flip_sin_to_cos=cfg.flip_sin_to_cos,
            resnet_time_scale_shift=cfg.resnet_time_scale_shift,
            # Sampling
            center_input_sample=cfg.center_input_sample,
            mid_block_scale_factor=cfg.mid_block_scale_factor,
            # Down/Up sampling
            downsample_padding=cfg.downsample_padding,
            downsample_type=cfg.downsample_type,
            upsample_type=cfg.upsample_type,
            # Activation
            act_fn=cfg.act_fn,
            # Class embedding (we handle this externally via ClassEmbedder)
            class_embed_type=cfg.class_embed_type,
            num_class_embeds=cfg.num_class_embeds,
            num_train_timesteps=cfg.num_train_timesteps,
        )

    def forward(self, x_noisy: torch.Tensor, timesteps: torch.Tensor, class_labels: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass predicting noise ε_θ(x_t, t, y).

        Args:
            x_noisy: Noisy image tensor [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            class_labels: Class labels [B] or None for unconditional

        Returns:
            Predicted noise tensor [B, C, H, W]
        """
        B, _, H, W = x_noisy.shape
        cond = self.class_embed(class_labels, (H, W), x_noisy.device)
        x_in = torch.cat([x_noisy, cond], dim=1)
        return self.unet(x_in, timesteps).sample

    def get_num_params(self) -> int:
        """Return total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
