"""
WideResNet-28-10 classifier for medical image classification.
"""

from __future__ import annotations
from typing import Tuple, Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    _LRScheduler, CosineAnnealingLR, StepLR, OneCycleLR
)

from .base_classifier import BaseClassifier
from ..config import ClassificationConfig

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """
    Basic block for WideResNet.
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.equal_inout = (in_planes == out_planes)
        self.convShortcut = (not self.equal_inout) and nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride,
            padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x if self.equal_inout else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    """
    Network block (group of basic blocks) for WideResNet.
    """

    def __init__(self, nb_layers: int, in_planes: int, out_planes: int,
                 block: nn.Module, stride: int, dropout: float = 0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout)

    def _make_layer(self, block, in_planes: int, out_planes: int, nb_layers: int,
                    stride: int, dropout: float):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(
                i == 0 and in_planes or out_planes,
                out_planes,
                i == 0 and stride or 1,
                dropout
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet model.

    Args:
        depth: Network depth (total layers)
        widen_factor: Width multiplier
        num_classes: Number of output classes
        dropout: Dropout rate
    """

    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0), 'Depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class WideResNet28_10Classifier(BaseClassifier):
    """
    WideResNet-28-10 classifier for medical image classification.

    WideResNet with depth=28 and widen_factor=10.
    """

    def __init__(self, config: ClassificationConfig):
        """
        Initialize WideResNet-28-10 classifier.

        Args:
            config: ClassificationConfig instance
        """
        super().__init__(config)

        # Validate model name
        if config.model.name != "wideresnet28_10":
            logger.warning(
                f"Model name is '{config.model.name}' but using WideResNet28_10Classifier. "
                "Consider using config.model.name='wideresnet28_10'"
            )

    def build_model(self) -> nn.Module:
        """
        Build WideResNet-28-10 model.

        Returns:
            WideResNet-28-10 model
        """
        logger.info("Building WideResNet-28-10 (depth=28, widen_factor=10)")

        model = WideResNet(
            depth=28,
            widen_factor=10,
            num_classes=self.config.model.num_classes,
            dropout=self.config.model.dropout
        )

        logger.info(
            f"WideResNet-28-10 classifier: "
            f"num_classes={self.config.model.num_classes}, "
            f"dropout={self.config.model.dropout}"
        )

        return model

    def build_optimizer(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Build optimizer and learning rate scheduler.

        Returns:
            Tuple of (optimizer, scheduler)
        """
        opt_config = self.config.optimization
        sched_config = self.config.scheduler

        # Build optimizer
        if opt_config.optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
        elif opt_config.optimizer == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
        elif opt_config.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")

        logger.info(f"Optimizer: {opt_config.optimizer} (lr={opt_config.lr}, wd={opt_config.weight_decay})")

        # Build scheduler
        scheduler = None
        if sched_config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs - sched_config.warmup_epochs,
                eta_min=sched_config.eta_min
            )
            logger.info(f"Scheduler: CosineAnnealingLR (T_max={scheduler.T_max}, eta_min={sched_config.eta_min})")
        elif sched_config.scheduler == "step":
            scheduler = StepLR(
                optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma
            )
            logger.info(f"Scheduler: StepLR (step_size={sched_config.step_size}, gamma={sched_config.gamma})")
        elif sched_config.scheduler == "constant":
            logger.info("Scheduler: None (constant learning rate)")
        else:
            logger.warning(f"Unknown scheduler: {sched_config.scheduler}, using constant LR")

        return optimizer, scheduler
