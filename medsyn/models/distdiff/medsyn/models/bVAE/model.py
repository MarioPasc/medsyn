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

class ResBlock(nn.Module):
    """Residual block with optional FiLM conditioning."""
    def __init__(self, c: int, use_film: bool, num_classes: int, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1   = nn.GroupNorm(8, c)
        self.act1  = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2   = nn.GroupNorm(8, c)
        self.act2  = nn.SiLU(inplace=True)
        self.use_film = use_film
        self.film1 = FiLMLayer(c, num_classes, embed_dim) if use_film else None
        self.film2 = FiLMLayer(c, num_classes, embed_dim) if use_film else None

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        h = self.act1(self.bn1(self.conv1(x)))
        if self.use_film and y is not None:
            h = self.film1(h, y)
        h = self.bn2(self.conv2(h))
        if self.use_film and y is not None:
            h = self.film2(h, y)
        return self.act2(x + h)

# --------------------------- Conditional β-VAE --------------------------------

class ConditionalBetaVAE(nn.Module):
    """
    c-βVAE con FiLM en encoder, concat[y] en decoder, bloques residuales
    y prior condicional p(z|y)=N(μ_y,σ_y^2) opcional.
    """
    def __init__(self,
                 in_channels: int, img_size: int, latent_dim: int,
                 base_channels: int, num_down: int,
                 num_classes: int, conditioning: str = "film",
                 class_embed_dim: int = 64,
                 decoder_sigmoid: bool = True,
                 use_class_prior: bool = True,
                 decoder_conditioning: bool = False):
        super().__init__()
        C, S, D, B = in_channels, img_size, latent_dim, base_channels
        # Ensure geometry is valid: S must be divisible by 2**num_down
        assert (S % (2 ** num_down)) == 0, "img_size must be divisible by 2**num_down"
        self.num_classes = int(num_classes)
        self.decoder_sigmoid = decoder_sigmoid
        use_film = (conditioning == "film")
        self.s_out = S // (2 ** num_down)

        # --- Encoder ---
        enc = []
        c = C
        for i in range(num_down):
            out_c = B * (2 ** i)
            enc += [nn.Conv2d(c, out_c, 3, padding=1),
                    nn.GroupNorm(8, out_c),
                    nn.SiLU(inplace=True),
                    ResBlock(out_c, use_film, self.num_classes, class_embed_dim),
                    nn.Conv2d(out_c, out_c, 4, stride=2, padding=1)]
            c = out_c
        self.enc = nn.Sequential(*enc)
        self.flat_dim = c * self.s_out * self.s_out
        self.fc_mu   = nn.Linear(self.flat_dim, D)
        self.fc_logv = nn.Linear(self.flat_dim, D)

        # --- Prior condicional por clase (opcional) ---
        self.use_class_prior = bool(use_class_prior)
        if self.use_class_prior:
            self.prior_mu   = nn.Embedding(self.num_classes, D)
            self.prior_logv = nn.Embedding(self.num_classes, D)  # log σ²
            nn.init.zeros_(self.prior_mu.weight)
            nn.init.zeros_(self.prior_logv.weight)
            # bounds for numerical stability (tighter for better stability)
            self._prior_mu_max = 3.0          # |μ_y| ≤ 3
            self._prior_logv_min = -2.0       # σ² ≥ e^-2 ≈ 0.14
            self._prior_logv_max =  2.0       # σ² ≤ e^2  ≈ 7.39

        # --- Decoder ---
        self.fc_dec = nn.Linear(D + self.num_classes, self.flat_dim)
        dec = []
        for i in reversed(range(num_down)):
            in_c = B * (2 ** max(i, 0))
            out_c = B * (2 ** (i - 1)) if i > 0 else B
            dec += [nn.ConvTranspose2d(in_c, in_c, 4, stride=2, padding=1),
                    nn.GroupNorm(8, in_c),
                    nn.SiLU(inplace=True),
                    ResBlock(in_c, decoder_conditioning, self.num_classes, class_embed_dim),
                    nn.Conv2d(in_c, out_c, 3, padding=1),
                    nn.GroupNorm(8, out_c),
                    nn.SiLU(inplace=True)]
        self.dec = nn.Sequential(*dec)
        self.out = nn.Conv2d(B, C, 3, padding=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logv: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * torch.clamp(logv, -10.0, 10.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for m in self.enc:
            if isinstance(m, ResBlock):
                h = m(h, y)
            else:
                h = m(h)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logv(h)

    def decode(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        zy = torch.cat([z, y_onehot], dim=1)
        h = self.fc_dec(zy).view(z.size(0), -1, self.s_out, self.s_out)
        # Inyecta y en los ResBlock del decoder
        h2 = h
        for m in self.dec:
            h2 = m(h2, y) if isinstance(m, ResBlock) else m(h2)
        x = self.out(h2)
        return torch.sigmoid(x) if self.decoder_sigmoid else x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logv = self.encode(x, y)
        z = self.reparameterize(mu, logv)
        x_hat = self.decode(z, y)
        return {"x_hat": x_hat, "mu": mu, "logv": logv}

    def prior_params(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return class-conditional prior params (mu_p, logv_p) or N(0,I)."""
        if not self.use_class_prior:  # N(0,I)
            D = self.fc_mu.out_features
            return (torch.zeros(y.size(0), D, device=y.device),
                    torch.zeros(y.size(0), D, device=y.device))
        mu   = torch.tanh(self.prior_mu(y)) * self._prior_mu_max
        logv = torch.clamp(self.prior_logv(y),
                           min=self._prior_logv_min,
                           max=self._prior_logv_max)
        return mu, logv

    @torch.no_grad()
    def sample(self, n: int, y: torch.Tensor, device: torch.device, tau: float = 0.7) -> torch.Tensor:
        """Sample from the model by drawing z from the learned class-conditional prior p(z|y).

        Args:
            n: number of samples
            y: class labels (B,)
            device: torch device
            tau: temperature parameter to scale variance (reduces blanks from over-sampling)
        """
        y = y.to(device)
        if self.use_class_prior:
            # Draw from learned class prior: z ~ N(μ_y, (τ·σ_y)²)
            mu_p, logv_p = self.prior_params(y)
            eps = torch.randn_like(mu_p)
            z = mu_p + (torch.exp(0.5 * logv_p) * tau) * eps
        else:
            # Fallback to standard Gaussian prior: z ~ N(0, I)
            z = torch.randn(n, self.fc_mu.out_features, device=device)
        return self.decode(z, y).clamp(0, 1)
