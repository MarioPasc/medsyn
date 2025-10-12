from __future__ import annotations
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Conditioning blocks ------------------------------

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation from a class embedding."""
    def __init__(self, num_features: int, num_classes: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.to_gamma = nn.Linear(embed_dim, num_features)
        self.to_beta  = nn.Linear(embed_dim, num_features)
        nn.init.zeros_(self.to_beta.weight); nn.init.zeros_(self.to_beta.bias)
        nn.init.zeros_(self.to_gamma.bias);  nn.init.normal_(self.to_gamma.weight, 0, 0.02)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        e = self.embed(y)                       # (B, E)
        gamma = self.to_gamma(e).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        beta  = self.to_beta(e).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU, optional FiLM cond."""
    def __init__(self, c_in: int, c_out: int, use_film: bool, num_classes: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)
        self.use_film = use_film
        self.film = FiLMLayer(c_out, num_classes, embed_dim) if use_film else None

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        if self.use_film and y is not None:
            x = self.film(x, y)
        return x

# --------------------------- Conditional β-VAE --------------------------------

class ConditionalBetaVAE(nn.Module):
    """
    Conditional β-VAE with FiLM in encoder blocks and label concatenation in the decoder.

    forward(x,y) returns dict: {"x_hat","mu","logv"}.
    sample(n, y, device) draws class-conditional samples.
    """
    def __init__(self,
                 in_channels: int, img_size: int, latent_dim: int,
                 base_channels: int, num_down: int,
                 num_classes: int, conditioning: str = "film",
                 class_embed_dim: int = 32,
                 decoder_sigmoid: bool = True):
        super().__init__()
        C, S, D, B = in_channels, img_size, latent_dim, base_channels
        self.num_classes = int(num_classes)
        self.decoder_sigmoid = decoder_sigmoid
        use_film = (conditioning == "film")
        self.s_out = S // (2 ** num_down)

        # Encoder
        enc = []
        c = C
        for i in range(num_down):
            out_c = B * (2 ** i)
            enc += [ConvBlock(c, out_c, use_film, self.num_classes, class_embed_dim),
                    nn.Conv2d(out_c, out_c, 4, stride=2, padding=1)]
            c = out_c
        enc += [nn.ReLU(inplace=True)]
        self.enc = nn.Sequential(*enc)

        self.flat_dim = c * self.s_out * self.s_out
        self.fc_mu   = nn.Linear(self.flat_dim, D)
        self.fc_logv = nn.Linear(self.flat_dim, D)

        # Decoder: concatenate one-hot(y) to z
        self.fc_dec = nn.Linear(D + self.num_classes, self.flat_dim)
        dec_blocks = []
        for i in reversed(range(num_down)):
            in_c = B * (2 ** max(i, 0))
            out_c = B * (2 ** (i - 1)) if i > 0 else B
            dec_blocks += [
                nn.ConvTranspose2d(in_c, in_c, 4, stride=2, padding=1),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
        self.dec = nn.Sequential(*dec_blocks)
        self.out = nn.Conv2d(B, C, 3, padding=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        # pass y to FiLM-enabled blocks
        for m in self.enc:
            if isinstance(m, ConvBlock):
                h = m(h, y)
            else:
                h = m(h)
        h = h.view(h.size(0), -1)
        mu, logv = self.fc_mu(h), self.fc_logv(h)
        return mu, logv

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        zy = torch.cat([z, y_onehot], dim=1)
        h = self.fc_dec(zy).view(z.size(0), -1, self.s_out, self.s_out)
        x = self.dec(h)
        x = self.out(x)
        return torch.sigmoid(x) if self.decoder_sigmoid else x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logv = self.encode(x, y)
        z = self.reparameterize(mu, logv)
        x_hat = self.decode(z, y)
        return {"x_hat": x_hat, "mu": mu, "logv": logv}

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.fc_mu.out_features, device=device)
        return self.decode(z, y.to(device)).clamp(0, 1)
