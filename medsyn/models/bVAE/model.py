# medsyn/models/bVAE/model.py
# Purpose: Simple convolutional β-VAE with modular encoder/decoder and sampling.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Blocks -------------------------------------------

class ConvBlock(nn.Module):
    # Conv -> BN -> ReLU
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

# --------------------------- Model --------------------------------------------

class BetaVAE(nn.Module):
    """
    β-VAE with Gaussian encoder and Bernoulli/Gaussian decoder head.

    Args:
        in_channels: input channels.
        img_size: input size (assumed square).
        latent_dim: dimensionality of latent z.
        base_channels: channels of first stage; doubles per downsampling stage.
        num_down: number of stride-2 downsamples.
        decoder_sigmoid: if True, applies sigmoid to output in [0,1].
    """
    def __init__(self, in_channels: int, img_size: int, latent_dim: int,
                 base_channels: int, num_down: int, decoder_sigmoid: bool = True):
        super().__init__()
        C, S, D, B = in_channels, img_size, latent_dim, base_channels

        # Encoder
        feats = []
        c = C
        for i in range(num_down):
            feats += [ConvBlock(c, B*(2**i)), nn.Conv2d(B*(2**i), B*(2**i), 4, stride=2, padding=1)]
            c = B*(2**i)
        self.enc = nn.Sequential(*feats, nn.ReLU(inplace=True))
        s_out = S // (2**num_down)
        self.flat_dim = c * s_out * s_out
        self.fc_mu   = nn.Linear(self.flat_dim, D)
        self.fc_logv = nn.Linear(self.flat_dim, D)

        # Decoder
        self.fc_dec = nn.Linear(D, self.flat_dim)
        dec = []
        for i in reversed(range(num_down)):
            in_c = B*(2**max(i,0))
            out_c = B*(2**(i-1)) if i > 0 else B
            dec += [nn.ConvTranspose2d(in_c, in_c, 4, stride=2, padding=1), nn.BatchNorm2d(in_c), nn.ReLU(inplace=True), ConvBlock(in_c, out_c)]
        self.dec = nn.Sequential(*dec)
        self.out = nn.Conv2d(B, C, 3, padding=1)
        self.use_sigmoid = decoder_sigmoid

    @staticmethod
    def reparameterize(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x).flatten(1)
        mu, logv = self.fc_mu(h), self.fc_logv(h)
        return mu, logv

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), -1, int((self.flat_dim // (z.size(1)))**0.5), int((self.flat_dim // (z.size(1)))**0.5))
        x = self.dec(h)
        x = self.out(x)
        return torch.sigmoid(x) if self.use_sigmoid else x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "mu": mu, "logv": logv}

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.fc_mu.out_features, device=device)
        return self.decode(z).clamp(0, 1)
