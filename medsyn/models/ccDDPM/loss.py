# medsyn/models/ccDDPM/loss.py
# Purpose: Training losses and ELBO/NLL estimators for interpretability.
# Implements: simple MSE on ε, per-class aggregation, optional ELBO term estimates.
from __future__ import annotations
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

class DDPMNoiseMSE:
    """
    Compute the simplified DDPM training objective: E[ ||ε - ε_θ(x_t, t, y)||^2 ].
    Keeps running means per class for interpretability.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.sum_per_class = torch.zeros(num_classes, dtype=torch.float64)
        self.count_per_class = torch.zeros(num_classes, dtype=torch.long)

    def __call__(self, pred_eps: torch.Tensor, true_eps: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(pred_eps, true_eps, reduction="none").mean(dim=(1,2,3))
        # update stats
        with torch.no_grad():
            # Move accumulator tensors to the same device as loss if needed
            if self.sum_per_class.device != loss.device:
                self.sum_per_class = self.sum_per_class.to(loss.device)
                self.count_per_class = self.count_per_class.to(loss.device)
            
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.any():
                    self.sum_per_class[c] += loss[mask].sum().double()
                    self.count_per_class[c] += mask.sum().long()
        return loss.mean()

    def per_class(self) -> Dict[int, float]:
        out = {}
        for c in range(self.num_classes):
            n = int(self.count_per_class[c])
            out[c] = float(self.sum_per_class[c] / max(n, 1))
        return out

def estimate_elbo_terms(
    x0_pred: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    scheduler_cfg: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Rough ELBO term estimates following Ho et al. (2020) appendix.
    Returns dict with L_simple (MSE), and L_0, L_t, L_T approximations in nats per-sample.
    Assumes scheduler_cfg provides alphas_cumprod[t], betas[t], and posterior variance terms.
    """
    alphas_cumprod = scheduler_cfg["alphas_cumprod"]    # [T]
    betas = scheduler_cfg["betas"]                      # [T]
    sqrt_one_minus_ac = torch.sqrt(1 - alphas_cumprod)

    # L_simple already available as MSE(ε, ε̂)
    L_simple = F.mse_loss(eps_pred, (x_t - torch.sqrt(alphas_cumprod[t])[:,None,None,None]*x0_pred)/sqrt_one_minus_ac[t][:,None,None,None], reduction="none").mean(dim=(1,2,3))

    # L_0: KL(q(x_{t-1}|x_0,x_t) || p_θ(x_{t-1}|x_t)) at t=1 -> NLL of x0 under p_θ
    # Approximate using predicted mean and var from model through scheduler equations.
    # Here we use a proxy: mse in x0 space scaled; exact closed form requires mean/var from scheduler.step.
    L_0 = F.mse_loss(x0_pred.clamp(-1,1), x0_pred.detach(), reduction="none").mean(dim=(1,2,3)) * 0.0  # placeholder zero (use scheduler step outputs during training if needed)

    # L_t: average KL for intermediate steps (proxy via ε MSE per-step)
    L_t = L_simple.detach()

    # L_T: prior term KL(q(x_T|x_0) || p(x_T)) closed form; for large T with β small -> near constant
    L_T = torch.zeros_like(L_simple)

    return {"L_simple": L_simple, "L_0": L_0, "L_t": L_t, "L_T": L_T}
