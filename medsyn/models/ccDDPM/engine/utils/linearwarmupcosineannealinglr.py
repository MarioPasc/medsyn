# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0

import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warmup from warmup_start_lr → base_lr for `warmup_epochs`,
    followed by cosine annealing from base_lr → eta_min for (max_epochs - warmup_epochs).

    Note:
        Call `scheduler.step()` *every iteration*, not every epoch.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute the learning rate for this iteration.
        """

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last computed learning rate, use `get_last_lr()`.",
                UserWarning,
            )

        # Case 0: beginning of training
        if self.last_epoch == 0:
            return [self.warmup_start_lr for _ in self.base_lrs]

        # Case 1: warmup finished exactly
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        # Case 2: warmup phase (linear increase)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        # Case 3: cosine restart boundary (rare edge case)
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        # Case 4: cosine annealing phase
        return [
            (
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs)
                              / (self.max_epochs - self.warmup_epochs)))
                / (
                    1
                    + math.cos(
                        math.pi * (self.last_epoch - self.warmup_epochs - 1)
                        / (self.max_epochs - self.warmup_epochs)
                    )
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        If someone calls scheduler.step(epoch), PyTorch uses this closed-form version.
        Same behavior, no iteration-based increment.
        """

        # Warmup
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / max(1, self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        # Cosine decay
        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs)
                            / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
