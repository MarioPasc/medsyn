# medsyn/models/ccDDPM/loss.py
# Purpose: Training losses and ELBO/NLL estimators for interpretability in ccDDPM.
# Implements: simplified ε-MSE training loss with optional Min-SNR weighting,
# per-class aggregation (raw + weighted), and diagnostic ELBO term estimates.
#
# References:
#   - Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
#   - Hang et al. (2023), "Efficient Diffusion Training via Min-SNR Weighting Strategy"
#   - Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models"
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# SNR UTILITY FUNCTIONS
# =============================================================================

def _compute_snr(alpha_bar_t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute signal-to-noise ratio (SNR) from cumulative product of alphas.

    The SNR at timestep t is defined as:

        SNR(t) = ᾱ_t / (1 - ᾱ_t)

    where ᾱ_t = ∏_{s=1}^t α_s is the cumulative product of (1 - β_s).

    Args:
        alpha_bar_t: Tensor of cumulative alpha values ᾱ_t, shape [B] or [T].
        eps: Small constant to prevent division by zero.

    Returns:
        Tensor of SNR values with the same shape as input.
    """
    return alpha_bar_t / (1.0 - alpha_bar_t + eps)


def _compute_min_snr_weight(
    snr: torch.Tensor,
    gamma: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Min-SNR loss weights from SNR values.

    The Min-SNR weighting strategy (Hang et al., 2023) assigns weights:

        w_snr(t) = min(SNR(t), γ) / SNR(t)

    This effectively:
      - For high-SNR (low noise) timesteps: w ≈ γ/SNR < 1 (down-weight)
      - For low-SNR (high noise) timesteps where SNR < γ: w = 1 (no change)

    The weight is always in (0, 1] and clips large SNR contributions while
    preserving contributions from low-SNR (high noise) timesteps.

    Args:
        snr: Signal-to-noise ratio values, shape [B].
        gamma: Clipping parameter γ (typically 3–5).
        eps: Small constant to prevent division by zero.

    Returns:
        Min-SNR weights in (0, 1], shape [B].
    """
    snr_clipped = torch.clamp(snr, max=gamma)
    weight = snr_clipped / (snr + eps)
    # Handle numerical issues (inf/nan) by falling back to 1.0
    weight = torch.where(torch.isfinite(weight), weight, torch.ones_like(weight))
    return weight


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
        # Reference: Hang et al. (2023), "Efficient Diffusion Training via Min-SNR"
        #
        # w_snr(t) = min(SNR(t), γ) / SNR(t)
        #
        # where SNR(t) = ᾱ_t / (1 - ᾱ_t). This weight is in (0, 1]:
        #   - High SNR (low noise, early t): w < 1 (down-weight easy samples)
        #   - Low SNR (high noise, late t) where SNR < γ: w = 1 (preserve hard samples)
        # ---------------------------------------------------------------------
        if self.use_min_snr:
            alphas_cumprod = alphas_cumprod.to(device)
            if timesteps.max() >= alphas_cumprod.shape[0]:
                raise ValueError(
                    f"Max timestep {timesteps.max().item()} exceeds "
                    f"alphas_cumprod length {alphas_cumprod.shape[0]}"
                )

            alpha_bar_t = alphas_cumprod[timesteps]  # [B]
            snr = _compute_snr(alpha_bar_t)
            snr_weight = _compute_min_snr_weight(snr, self.min_snr_gamma)
        else:
            snr_weight = torch.ones(bsz, device=device, dtype=per_example_mse.dtype)

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
    use_min_snr: bool = False,
    min_snr_gamma: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """
    Approximate per-sample decomposition of the DDPM loss with Min-SNR diagnostics.

    This diagnostic function computes:

        L_simple:  Unweighted ε-MSE for each sample (standard training loss
                   before any weighting). This matches DDPMNoiseMSE when
                   use_min_snr=False and class_weights=None.

        L_weighted: ε-MSE weighted by Min-SNR factor:
                    L_weighted = w_snr(t) * L_simple
                    where w_snr(t) = min(SNR(t), γ) / SNR(t)
                    This matches DDPMNoiseMSE when use_min_snr=True.

        snr:        Signal-to-noise ratio SNR(t) = ᾱ_t / (1 - ᾱ_t)

        min_snr_weight: The Min-SNR weight w_snr(t) applied (for analysis)

    When use_min_snr=False, L_weighted equals L_simple and min_snr_weight=1.

    References:
        - Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
        - Hang et al. (2023), "Efficient Diffusion Training via Min-SNR"

    Note:
        • This function is intended for offline analysis and does not affect
          training. It assumes images are scaled to [-1, 1].
        • The L_simple computed here exactly matches the per-example MSE in
          DDPMNoiseMSE.__call__ before any weighting is applied.

    Required scheduler_cfg keys:
        'alphas_cumprod' : 1D tensor ᾱ_t of length T.

    Optional scheduler_cfg keys:
        'betas'          : 1D tensor β_t of length T (not used in current impl).

    Args:
        x0: Clean images [B, C, H, W]
        x_t: Noised images at timestep t [B, C, H, W]
        t: Timestep indices [B]
        eps_pred: Predicted noise ε_θ(x_t, t) [B, C, H, W]
        scheduler_cfg: Dict with 'alphas_cumprod' tensor
        use_min_snr: If True, compute L_weighted with Min-SNR weighting.
                     If False, L_weighted == L_simple.
        min_snr_gamma: Clipping parameter γ for Min-SNR (typically 3–5).

    Returns:
        Dictionary with:
            t: Timestep indices [B]
            snr: Signal-to-noise ratio at each timestep [B]
            L_simple: Unweighted ε-MSE [B]
            L_weighted: Min-SNR weighted ε-MSE [B]
            min_snr_weight: The weight factor applied [B]
    """
    if "alphas_cumprod" not in scheduler_cfg:
        raise KeyError("scheduler_cfg must contain key 'alphas_cumprod'.")

    alphas_cumprod: torch.Tensor = scheduler_cfg["alphas_cumprod"].to(x0.device).float()

    if alphas_cumprod.dim() != 1:
        raise ValueError("alphas_cumprod must be a 1D tensor.")

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

    # Recover ground-truth noise ε from (x0, x_t, t) using q(x_t | x_0)
    # x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
    eps_true = (x_t - sqrt_alpha_bar_t * x0) / (sqrt_one_minus_alpha_bar_t + 1e-8)

    # L_simple: standard ε-MSE term per sample (no weighting)
    # This matches DDPMNoiseMSE.forward() before any weighting is applied.
    L_simple = F.mse_loss(eps_pred, eps_true, reduction="none")
    L_simple = L_simple.view(bsz, -1).mean(dim=1)  # [B]

    # Compute SNR using shared utility
    snr = _compute_snr(alpha_bar_t)

    # Compute Min-SNR weight using shared utility
    if use_min_snr:
        min_snr_weight = _compute_min_snr_weight(snr, min_snr_gamma)
    else:
        min_snr_weight = torch.ones_like(L_simple)

    # L_weighted: ε-MSE with Min-SNR weighting (matches training loss)
    L_weighted = min_snr_weight * L_simple  # [B]

    # All outputs are detached to emphasize that this is diagnostic only
    return {
        "t": t.detach(),
        "snr": snr.detach(),
        "L_simple": L_simple.detach(),
        "L_weighted": L_weighted.detach(),
        "min_snr_weight": min_snr_weight.detach(),
    }

