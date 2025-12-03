"""
ResNet50 classifier for medical image classification.
"""

from __future__ import annotations
from typing import Tuple, Optional
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    _LRScheduler, CosineAnnealingLR, StepLR, OneCycleLR
)
import torchvision.models as models

from .base_classifier import BaseClassifier
from ..config import ClassificationConfig

logger = logging.getLogger(__name__)


class ResNet50Classifier(BaseClassifier):
    """
    ResNet50 classifier for medical image classification.

    Uses pretrained ImageNet weights and replaces the final FC layer
    with a new classifier head.
    """

    def __init__(self, config: ClassificationConfig):
        """
        Initialize ResNet50 classifier.

        Args:
            config: ClassificationConfig instance
        """
        super().__init__(config)

        # Validate model name
        if config.model.name != "resnet50":
            logger.warning(
                f"Model name is '{config.model.name}' but using ResNet50Classifier. "
                "Consider using config.model.name='resnet50'"
            )

    def build_model(self) -> nn.Module:
        """
        Build ResNet50 model with custom classifier head.

        Returns:
            ResNet50 model
        """
        # Load pretrained ResNet50
        if self.config.model.pretrained:
            logger.info("Loading pretrained ResNet50 (ImageNet weights)")
            model = models.resnet50(pretrained=True)
        else:
            logger.info("Initializing ResNet50 from scratch")
            model = models.resnet50(pretrained=False)

        # Freeze backbone if requested
        if self.config.model.freeze_backbone:
            logger.info("Freezing ResNet50 backbone (only training classifier head)")
            for param in model.parameters():
                param.requires_grad = False

        # Replace final FC layer
        num_features = model.fc.in_features

        # Build classifier head
        classifier_layers = []
        if self.config.model.dropout > 0:
            classifier_layers.append(nn.Dropout(p=self.config.model.dropout))
        classifier_layers.append(nn.Linear(num_features, self.config.model.num_classes))

        model.fc = nn.Sequential(*classifier_layers)

        # Ensure classifier head is trainable
        for param in model.fc.parameters():
            param.requires_grad = True

        logger.info(f"ResNet50 classifier head: {num_features} -> {self.config.model.num_classes}")

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
