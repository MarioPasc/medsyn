from typing import Optional, Dict, Any

# =============================================================================
# EARLY STOPPING TRACKER WITH EMA SMOOTHING
# =============================================================================

class EarlyStoppingTracker:
    """
    Early stopping tracker with EMA smoothing of the validation score.

    Computes a composite validation score:
        S = (psnr_weight * PSNR / 20) + (ssim_weight * SSIM)

    where PSNR is divided by 20 to put it on a similar scale to SSIM (~0-1).

    The score is optionally smoothed with an exponential moving average:
        S_ema_t = ema_alpha * S_ema_{t-1} + (1 - ema_alpha) * S_t

    Higher ema_alpha means more smoothing (less sensitive to fluctuations).

    Attributes:
        patience: Number of epochs without improvement before stopping
        psnr_weight: Weight for PSNR in composite score
        ssim_weight: Weight for SSIM in composite score
        ema_alpha: EMA smoothing factor (0 = no smoothing)
        use_ema_for_stopping: Whether to use EMA score for stopping decisions
        min_delta: Minimum improvement to count as "better"
    """

    def __init__(
        self,
        patience: int = 5,
        psnr_weight: float = 1.0,
        ssim_weight: float = 1.0,
        ema_alpha: float = 0.7,
        use_ema_for_stopping: bool = True,
        min_delta: float = 0.001,
        metric: str = "psnr_ssim_composite",
    ):
        self.patience = patience
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.ema_alpha = ema_alpha
        self.use_ema_for_stopping = use_ema_for_stopping
        self.min_delta = min_delta
        self.metric = metric

        # State
        self.best_score: float = float("-inf")
        self.best_ema_score: float = float("-inf")
        self.best_epoch: int = 0
        self.epochs_without_improvement: int = 0
        self.score_ema: Optional[float] = None
        self._history: list = []

    def compute_score(self, val_metrics: Dict[str, float]) -> float:
        """
        Compute the validation score from metrics.

        Args:
            val_metrics: Dictionary with 'psnr', 'ssim', and/or 'loss' keys

        Returns:
            Composite validation score (higher is better)
        """
        if self.metric == "psnr_ssim_composite":
            psnr = val_metrics.get("psnr", 0.0)
            ssim = val_metrics.get("ssim", 0.0)
            # Divide PSNR by 20 to put it on similar scale to SSIM (~0-1)
            score = (self.psnr_weight * psnr / 20.0) + (self.ssim_weight * ssim)
        elif self.metric == "psnr":
            score = val_metrics.get("psnr", 0.0)
        elif self.metric == "ssim":
            score = val_metrics.get("ssim", 0.0)
        elif self.metric == "loss":
            # Negate loss so higher is still better
            score = -val_metrics.get("loss", float("inf"))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return score

    def update(self, epoch: int, val_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Update tracker with new validation metrics.

        Args:
            epoch: Current epoch number
            val_metrics: Dictionary with validation metrics

        Returns:
            Dictionary with early stopping diagnostics:
                - val_score_raw: Raw composite validation score
                - val_score_ema: EMA-smoothed validation score
                - is_best: Whether this epoch is the best so far
                - should_stop: Whether early stopping should trigger
                - epochs_without_improvement: Number of epochs since last improvement
                - best_score: Best score seen so far
                - best_epoch: Epoch that achieved best score
        """
        # Compute raw score
        score_raw = self.compute_score(val_metrics)

        # Update EMA
        if self.score_ema is None:
            # First epoch: initialize EMA to raw score
            self.score_ema = score_raw
        else:
            self.score_ema = (
                self.ema_alpha * self.score_ema + (1 - self.ema_alpha) * score_raw
            )

        # Determine which score to use for early stopping
        score_for_stopping = self.score_ema if self.use_ema_for_stopping else score_raw

        # Check if improved
        is_best = score_for_stopping > (self.best_score + self.min_delta)

        if is_best:
            self.best_score = score_for_stopping
            self.best_ema_score = self.score_ema
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Check if should stop
        should_stop = self.epochs_without_improvement >= self.patience

        # Store history for analysis
        self._history.append({
            "epoch": epoch,
            "score_raw": score_raw,
            "score_ema": self.score_ema,
            "is_best": is_best,
            "epochs_without_improvement": self.epochs_without_improvement,
        })

        return {
            "val_score_raw": score_raw,
            "val_score_ema": self.score_ema,
            "is_best": is_best,
            "should_stop": should_stop,
            "epochs_without_improvement": self.epochs_without_improvement,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "early_stop_flag": 1.0 if should_stop else 0.0,
        }

    @classmethod
    def from_config(cls, cfg: Any) -> "EarlyStoppingTracker":
        """
        Create an EarlyStoppingTracker from a config object.

        Handles both old-style configs (tcfg.patience) and new-style configs
        (tcfg.early_stopping.patience).
        """
        # Check for new-style config
        if hasattr(cfg, 'early_stopping') and cfg.early_stopping is not None:
            es_cfg = cfg.early_stopping
            return cls(
                patience=getattr(es_cfg, 'patience', 5),
                psnr_weight=getattr(es_cfg, 'psnr_weight', 1.0),
                ssim_weight=getattr(es_cfg, 'ssim_weight', 1.0),
                ema_alpha=getattr(es_cfg, 'ema_alpha', 0.7),
                use_ema_for_stopping=getattr(es_cfg, 'use_ema_for_stopping', True),
                min_delta=getattr(es_cfg, 'min_delta', 0.001),
                metric=getattr(es_cfg, 'metric', 'psnr_ssim_composite'),
            )
        else:
            # Fallback to old-style config
            return cls(
                patience=getattr(cfg, 'patience', 5),
                psnr_weight=1.0,
                ssim_weight=50.0,  # Match old behavior: PSNR + 50*SSIM
                ema_alpha=0.0,  # No smoothing in old behavior
                use_ema_for_stopping=False,
                min_delta=0.0,
                metric='psnr_ssim_composite',
            )
