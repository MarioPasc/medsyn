# medsyn/models/ccDDPM/loss.py
# Purpose: Training losses and ELBO/NLL estimators for interpretability in ccDDPM.
# Implements: simplified ε-MSE training loss with optional Min-SNR weighting,
# per-class aggregation (raw + weighted), and diagnostic ELBO term estimates.
from __future__ import annotations

from typing import Dict, Any, Optional
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DDPMNoiseMSE:
    """
    Simplified DDPM training objective operating on predicted noise ε_θ.

    The core objective is the per-example mean squared error

        L_simple = E[ || ε - ε_θ(x_t, t, y) ||_2^2 ],

    optionally reweighted by a Min-SNR schedule (Hang et al., 2023) and
    user-supplied per-class weights. This class also keeps running per-class
    aggregates for both the raw and the weighted loss, for interpretability
    and logging.
    """

    # Constructor documentation (comment only, no usage examples):
    # Args:
    #     num_classes: Number of semantic classes (excluding the "null" label -1).
    #     use_min_snr: If True, apply Min-SNR loss weighting across timesteps.
    #     min_snr_gamma: Clipping parameter γ for Min-SNR (typically 3–5).
    def __init__(
        self,
        num_classes: int,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
    ) -> None:
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        self.num_classes = int(num_classes)
        self.use_min_snr = bool(use_min_snr)
        self.min_snr_gamma = float(min_snr_gamma)

        # Running aggregates (on CPU) for per-class losses
        self.reset()

    # Method documentation:
    # Resets all running per-class statistics (raw + weighted losses and counts).
    def reset(self) -> None:
        self.sum_per_class_raw = torch.zeros(self.num_classes, dtype=torch.float64)
        self.sum_per_class_weighted = torch.zeros(self.num_classes, dtype=torch.float64)
        self.count_per_class = torch.zeros(self.num_classes, dtype=torch.long)

    # Call documentation:
    # Args:
    #     pred_noise: Model prediction ε_θ(x_t, t, y) with shape [B, C, H, W].
    #     true_noise: Ground-truth Gaussian noise ε with shape [B, C, H, W].
    #     labels: Class indices for conditioning, shape [B], with possible value -1
    #             for classifier-free "unconditional" samples.
    #     timesteps: Integer timesteps t ∈ {0, ..., T-1} with shape [B].
    #     alphas_cumprod: 1D tensor of ᾱ_t = ∏_{s=1}^t α_s, length T.
    #     class_weights: Optional tensor of shape [num_classes] with per-class
    #                    multiplicative weights computed outside the loss from
    #                    dataset class frequencies.
    #
    # Returns:
    #     Scalar tensor with the batch-averaged, fully weighted loss.
    def __call__(
        self,
        pred_noise: torch.Tensor,
        true_noise: torch.Tensor,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Basic shape checks
        if pred_noise.shape != true_noise.shape:
            raise ValueError(
                f"pred_noise and true_noise must have the same shape, "
                f"got {pred_noise.shape} vs {true_noise.shape}"
            )
        if pred_noise.dim() != 4:
            raise ValueError(
                f"Expected 4D tensors [B, C, H, W], got dim={pred_noise.dim()}"
            )

        bsz = pred_noise.size(0)

        if labels.shape[0] != bsz:
            raise ValueError(
                f"labels batch size {labels.shape[0]} does not match "
                f"pred batch size {bsz}"
            )
        if timesteps.shape[0] != bsz:
            raise ValueError(
                f"timesteps batch size {timesteps.shape[0]} does not match "
                f"pred batch size {bsz}"
            )
        if alphas_cumprod.dim() != 1:
            raise ValueError(
                "alphas_cumprod must be a 1D tensor of cumulative products ᾱ_t."
            )

        device = pred_noise.device
        labels = labels.to(device)
        timesteps = timesteps.to(device)

        # ---------------------------------------------------------------------
        # Raw ε-MSE per example (no weighting yet)
        # ---------------------------------------------------------------------
        per_example_mse = F.mse_loss(pred_noise, true_noise, reduction="none")
        per_example_mse = per_example_mse.view(bsz, -1).mean(dim=1)  # [B]

        # ---------------------------------------------------------------------
        # Min-SNR loss weighting across timesteps (optional)
        # ---------------------------------------------------------------------
        if self.use_min_snr:
            alphas_cumprod = alphas_cumprod.to(device)
            if timesteps.max() >= alphas_cumprod.shape[0]:
                raise ValueError(
                    f"Max timestep {timesteps.max().item()} exceeds "
                    f"alphas_cumprod length {alphas_cumprod.shape[0]}"
                )

            alpha_bar_t = alphas_cumprod[timesteps]  # [B]
            # SNR_t = ᾱ_t / (1 - ᾱ_t)
            snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)
            gamma = self.min_snr_gamma
            snr_clipped = torch.clamp(snr, max=gamma)
            snr_weight = snr_clipped / (snr + 1e-8)
            # Fallback to 1.0 where numerical issues occur
            snr_weight = torch.where(
                torch.isfinite(snr_weight), snr_weight, torch.ones_like(snr_weight)
            )
        else:
            snr_weight = torch.ones_like(per_example_mse, device=device)

        # ---------------------------------------------------------------------
        # Optional per-class weighting (pre-computed outside the loss)
        # ---------------------------------------------------------------------
        if class_weights is not None:
            if class_weights.dim() != 1:
                raise ValueError(
                    f"class_weights must be 1D of shape [num_classes], "
                    f"got shape {tuple(class_weights.shape)}"
                )
            if class_weights.numel() < self.num_classes:
                raise ValueError(
                    f"class_weights has {class_weights.numel()} entries, "
                    f"but num_classes={self.num_classes}"
                )
            if (labels >= 0).any() and labels.max() >= self.num_classes:
                raise ValueError(
                    f"Found label >= num_classes while applying class weights: "
                    f"max label={labels.max().item()}, num_classes={self.num_classes}"
                )

            class_weights = class_weights.to(device).float()
            class_weight_per_example = torch.ones_like(per_example_mse, device=device)

            # Only apply class weights to conditional samples (labels >= 0)
            cond_mask = labels >= 0
            if cond_mask.any():
                class_weight_per_example[cond_mask] = class_weights[
                    labels[cond_mask]
                ]
        else:
            class_weight_per_example = torch.ones_like(per_example_mse, device=device)

        # ---------------------------------------------------------------------
        # Combine all weighting factors and accumulate per-class statistics
        # ---------------------------------------------------------------------
        total_weight = snr_weight * class_weight_per_example
        weighted_loss_per_example = per_example_mse * total_weight  # [B]

        # Update running per-class aggregates on CPU (ignore label = -1)
        with torch.no_grad():
            labels_cpu = labels.detach().cpu()
            raw_cpu = per_example_mse.detach().cpu().to(torch.float64)
            weighted_cpu = weighted_loss_per_example.detach().cpu().to(torch.float64)

            for c in range(self.num_classes):
                mask_c = labels_cpu == c
                if mask_c.any():
                    idxs = mask_c.nonzero(as_tuple=False).view(-1)
                    self.sum_per_class_raw[c] += raw_cpu[idxs].sum()
                    self.sum_per_class_weighted[c] += weighted_cpu[idxs].sum()
                    self.count_per_class[c] += int(idxs.numel())

        # Final scalar loss used for backpropagation
        return weighted_loss_per_example.mean()

    # Per-class statistics documentation:
    # Args:
    #     weighted: If True, return means of fully weighted losses (Min-SNR and
    #               class weights). If False, return raw ε-MSE means.
    #
    # Returns:
    #     Dict mapping class index -> mean loss for that class (float).
    def per_class(self, weighted: bool = True) -> Dict[int, float]:
        if self.count_per_class is None:
            return {}

        stats: Dict[int, float] = {}
        sums = self.sum_per_class_weighted if weighted else self.sum_per_class_raw

        for c in range(self.num_classes):
            n = int(self.count_per_class[c].item())
            if n > 0:
                mean_c = (sums[c] / n).item()
                stats[c] = float(mean_c)
        return stats

    # Tabular per-class statistics documentation:
    # Returns:
    #     A pandas.DataFrame with columns ['class', 'raw_loss', 'weighted_loss'].
    #     This is intended for CSV export or rich logging.
    def per_class_table(self):
        try:
            import pandas as pd  
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "pandas is required to build the per-class loss table."
            ) from exc

        classes = list(range(self.num_classes))
        raw_means = []
        weighted_means = []

        for c in classes:
            n = int(self.count_per_class[c].item())
            if n > 0:
                raw_means.append(float((self.sum_per_class_raw[c] / n).item()))
                weighted_means.append(float((self.sum_per_class_weighted[c] / n).item()))
            else:
                raw_means.append(float("nan"))
                weighted_means.append(float("nan"))

        df = pd.DataFrame(
            {
                "class": classes,
                "raw_loss": raw_means,
                "weighted_loss": weighted_means,
            }
        )
        return df


def estimate_elbo_terms(
    x0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    scheduler_cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Approximate per-sample decomposition of the DDPM variational bound.

    This diagnostic function follows Ho et al. (2020, Sec. 3.2–3.4).
    It assumes an ε-parameterized model and a fixed-variance reverse
    process, and reports:

        L_simple:      Unweighted ε-MSE for each sample.
        L_t_weighted:  KL-like term proportional to the weighting in Eq. (12),
                       using σ_t^2 = β_t (the "fixed small" variance choice).
        snr:           Signal-to-noise ratio at each timestep.
        t:             Timestep indices as provided.

    Note:
        • L_0 and L_T are not computed because the decoder p_θ(x_0 | x_1) and
          the prior p(x_T) are not parameterized in this codebase.
        • This function is intended for offline analysis and does not affect
          training. It assumes images are scaled to [-1, 1], consistent with
          the main training pipeline.

    Required scheduler_cfg keys:
        'alphas_cumprod' : 1D tensor ᾱ_t of length T.
        'betas'          : 1D tensor β_t of length T.
    """
    if "alphas_cumprod" not in scheduler_cfg:
        raise KeyError("scheduler_cfg must contain key 'alphas_cumprod'.")
    if "betas" not in scheduler_cfg:
        raise KeyError("scheduler_cfg must contain key 'betas'.")

    alphas_cumprod: torch.Tensor = scheduler_cfg["alphas_cumprod"].to(x0.device).float()
    betas: torch.Tensor = scheduler_cfg["betas"].to(x0.device).float()

    if alphas_cumprod.dim() != 1 or betas.dim() != 1:
        raise ValueError("alphas_cumprod and betas must be 1D tensors.")
    if alphas_cumprod.shape[0] != betas.shape[0]:
        raise ValueError(
            f"alphas_cumprod and betas must have the same length, got "
            f"{alphas_cumprod.shape[0]} and {betas.shape[0]}"
        )

    bsz = x0.size(0)
    if t.shape[0] != bsz:
        raise ValueError(
            f"Timestep batch size {t.shape[0]} does not match x0 batch size {bsz}."
        )

    t = t.to(x0.device).long()
    if t.max() >= alphas_cumprod.shape[0]:
        raise ValueError(
            f"Max timestep {t.max().item()} exceeds schedule length "
            f"{alphas_cumprod.shape[0]}."
        )

    # Gather per-sample schedule values
    alpha_bar_t = alphas_cumprod[t]  # [B]
    beta_t = betas[t]  # [B]
    alpha_t = 1.0 - beta_t  # [B]

    # Recover ground-truth noise ε from (x0, x_t, t) using q(x_t | x_0)
    # x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
    eps_true = (x_t - sqrt_alpha_bar_t * x0) / (sqrt_one_minus_alpha_bar_t + 1e-8)

    # L_simple: standard ε-MSE term per sample
    L_simple = F.mse_loss(eps_pred, eps_true, reduction="none")
    L_simple = L_simple.view(bsz, -1).mean(dim=1)  # [B]

    # Eq. (12) in Ho et al. shows that the KL term Lt−1 is proportional to
    #
    #   (β_t^2 / (2 σ_t^2 α_t (1 - ᾱ_t))) ||ε - ε_θ||^2
    #
    # For the common fixed-variance choice σ_t^2 = β_t ("fixed small"), this
    # simplifies to β_t / (2 α_t (1 - ᾱ_t)) * ||ε - ε_θ||^2.
    sigma_t_sq = beta_t  # fixed small variance
    # Avoid division by zero in pathological schedules
    denom = 2.0 * sigma_t_sq * alpha_t * (1.0 - alpha_bar_t) + 1e-8
    weight_t = (beta_t**2) / denom  # [B]

    L_t_weighted = weight_t * L_simple  # [B]

    # For completeness, also report the forward-process SNR
    snr = alpha_bar_t / (1.0 - alpha_bar_t + 1e-8)

    # All outputs are detached to emphasize that this is diagnostic only
    return {
        "t": t.detach(),
        "snr": snr.detach(),
        "L_simple": L_simple.detach(),
        "L_t_weighted": L_t_weighted.detach(),
    }
