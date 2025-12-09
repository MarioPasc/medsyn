"""
DINOv3 ViT-B/16 classifier for medical image classification.

Uses Meta AI's DINOv3 with self-supervised pretraining.
Classification is performed via a custom head on top of pretrained ViT backbone.
"""

from __future__ import annotations
from typing import Tuple, Optional
import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam, SGD
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, StepLR

from .base_classifier import BaseClassifier
from ..config import ClassificationConfig

logger = logging.getLogger(__name__)


class DINOv3ClassificationWrapper(nn.Module):
    """
    DINOv3 model wrapper for classification.

    Architecture:
    - Backbone: DINOv3 ViT-B/16 extracts image features [B, embed_dim]
    - Classification head: Optional dropout + Linear layer [embed_dim, num_classes]

    Forward pass:
    1. Encode image with vision encoder -> [B, embed_dim]
    2. Pass through classification head -> [B, num_classes]
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.0
    ):
        """
        Initialize DINOv3 classification wrapper.

        Args:
            backbone: DINOv3 vision encoder (ViT-B/16)
            embed_dim: Embedding dimension from backbone
            num_classes: Number of output classes
            dropout: Dropout probability before classification layer
        """
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Build classification head
        classifier_layers = []
        if dropout > 0:
            classifier_layers.append(nn.Dropout(p=dropout))
        classifier_layers.append(nn.Linear(embed_dim, num_classes))
        self.head = nn.Sequential(*classifier_layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute classification logits.

        Args:
            images: [B, 3, H, W] normalized images (ImageNet normalization)

        Returns:
            logits: [B, num_classes] classification logits
        """
        # Extract features from backbone
        features = self.backbone(images)

        # Pass through classification head
        logits = self.head(features)

        return logits


class DINOv3ViTB16Classifier(BaseClassifier):
    """
    DINOv3 ViT-B/16 classifier for medical image classification.

    Uses Meta AI's DINOv3 with self-supervised pretraining. Classification is performed
    via a custom head on top of the pretrained ViT backbone.

    Key features:
    - DINOv3 ViT-B/16 pretrained backbone (86M parameters)
    - Custom classification head for medical imaging
    - Differential learning rates (backbone: 0.1x, head: 1.0x)
    - 224x224 input size (resized from PathMNIST 64x64)
    - Inherits training loop, validation, k-fold CV, metrics from BaseClassifier

    Architecture:
    - Backbone: DINOv3 ViT-B/16 (pretrained on web data - LVD-1689M)
    - Embedding dimension: 768 (standard ViT-B)
    - Classification head: Optional dropout + Linear(768 -> num_classes)
    """

    def __init__(self, config: ClassificationConfig):
        """
        Initialize DINOv3 ViT-B/16 classifier.

        Args:
            config: ClassificationConfig instance
        """
        super().__init__(config)

        # Validate config
        if config.model.name != "dinov3_vitb16":
            logger.warning(
                f"Model name is '{config.model.name}' but using DINOv3ViTB16Classifier. "
                "Consider using config.model.name='dinov3_vitb16'"
            )

    def build_model(self) -> nn.Module:
        """
        Build DINOv3 ViT-B/16 model with custom classification head.

        Steps:
        1. Load pretrained DINOv3 backbone from PyTorch Hub
        2. Extract embedding dimension
        3. Build custom classification head
        4. Wrap in classification wrapper
        5. Apply backbone freezing if requested

        Returns:
            nn.Module: DINOv3ClassificationWrapper model
        """
        logger.info("=" * 60)
        logger.info("Building DINOv3 ViT-B/16 classifier")
        logger.info("=" * 60)

        # Get DINOv3 configuration
        dinov3_variant = getattr(self.config.model, 'dinov3_variant', 'dinov3_vitb16')
        dinov3_repo = getattr(self.config.model, 'dinov3_repo', 'facebookresearch/dinov3')

        logger.info(f"Loading DINOv3 from PyTorch Hub...")
        logger.info(f"  Repository: {dinov3_repo}")
        logger.info(f"  Variant: {dinov3_variant}")
        logger.info(f"  Pretrained: {self.config.model.pretrained}")

        # Load pretrained DINOv3 model from PyTorch Hub
        try:
            model = torch.hub.load(
                dinov3_repo,
                dinov3_variant,
                pretrained=self.config.model.pretrained,
                trust_repo=True
            )
            logger.info("DINOv3 model loaded successfully from PyTorch Hub")
        except Exception as e:
            logger.error(f"Failed to load DINOv3 from PyTorch Hub: {e}")
            raise RuntimeError(
                f"Failed to load DINOv3 from PyTorch Hub.\n"
                f"Repository: {dinov3_repo}\n"
                f"Variant: {dinov3_variant}\n"
                f"Error: {e}\n\n"
                f"Troubleshooting:\n"
                f"1. Check internet connection (PyTorch Hub downloads from GitHub)\n"
                f"2. Try: torch.hub.list('{dinov3_repo}') to see available models\n"
                f"3. Ensure timm is installed: pip install -e '.[distdiff]'\n"
                f"4. Check https://github.com/facebookresearch/dinov3 for latest info"
            )

        # Inspect model structure to find embedding dimension
        logger.info("\nInspecting DINOv3 model structure...")

        # DINOv3 models typically have embed_dim attribute or can be inferred
        if hasattr(model, 'embed_dim'):
            embed_dim = model.embed_dim
            logger.info(f"  Found embed_dim attribute: {embed_dim}")
        elif hasattr(model, 'num_features'):
            embed_dim = model.num_features
            logger.info(f"  Found num_features attribute: {embed_dim}")
        else:
            # Default to 768 for ViT-B
            embed_dim = 768
            logger.warning(
                f"Could not find embed_dim in model, using default: {embed_dim}"
            )

        logger.info(f"\nModel architecture:")
        logger.info(f"  Backbone: DINOv3 {dinov3_variant}")
        logger.info(f"  Embedding dimension: {embed_dim}")
        logger.info(f"  Number of classes: {self.config.model.num_classes}")
        logger.info(f"  Dropout: {self.config.model.dropout}")

        # Build classification wrapper
        dinov3_model = DINOv3ClassificationWrapper(
            backbone=model,
            embed_dim=embed_dim,
            num_classes=self.config.model.num_classes,
            dropout=self.config.model.dropout
        )

        # Handle backbone freezing
        if self.config.model.freeze_backbone:
            logger.info("\nFreezing DINOv3 backbone (only training classification head)")
            for param in dinov3_model.backbone.parameters():
                param.requires_grad = False

            # Count frozen vs trainable params
            frozen_params = sum(p.numel() for p in dinov3_model.backbone.parameters())
            trainable_params = sum(
                p.numel() for p in dinov3_model.parameters() if p.requires_grad
            )
            logger.info(f"  Frozen params: {frozen_params:,}")
            logger.info(f"  Trainable params: {trainable_params:,}")
        else:
            total_params = sum(p.numel() for p in dinov3_model.parameters())
            trainable_params = sum(
                p.numel() for p in dinov3_model.parameters() if p.requires_grad
            )
            logger.info("\nTraining full model (backbone + classification head)")
            logger.info(f"  Total params: {total_params:,}")
            logger.info(f"  Trainable params: {trainable_params:,}")

        logger.info("=" * 60)
        logger.info("DINOv3 model built successfully")
        logger.info("=" * 60)

        return dinov3_model

    def build_optimizer(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Build optimizer with differential learning rates for DINOv3 components.

        Parameter groups (with differential learning rates):
        - Backbone: base_lr * 0.1 (pretrained, needs smaller updates)
        - Classification head: base_lr * 1.0 (new layer, needs larger updates)

        Returns:
            Tuple of (optimizer, scheduler). Scheduler can be None.
        """
        opt_config = self.config.optimization
        sched_config = self.config.scheduler

        logger.info("Building optimizer with differential learning rates...")

        # Separate parameters into groups
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'backbone' in name:
                backbone_params.append(param)
            elif 'head' in name:
                head_params.append(param)

        base_lr = opt_config.lr

        # Build parameter groups with differential LRs
        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * 0.1,
                'name': 'backbone'
            })

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr,
                'name': 'classification_head'
            })

        # Log parameter group info
        logger.info("Parameter groups:")
        for group in param_groups:
            n_params = sum(p.numel() for p in group['params'])
            logger.info(f"  {group['name']}: {n_params:,} params, lr={group['lr']:.6f}")

        # Build optimizer (same options as other classifiers)
        if opt_config.optimizer == "adamw":
            optimizer = AdamW(
                param_groups,
                lr=base_lr,  # Default, overridden by group LRs
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
            logger.info(f"Optimizer: AdamW (base_lr={base_lr:.6f}, wd={opt_config.weight_decay})")

        elif opt_config.optimizer == "adam":
            optimizer = Adam(
                param_groups,
                lr=base_lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
            logger.info(f"Optimizer: Adam (base_lr={base_lr:.6f}, wd={opt_config.weight_decay})")

        elif opt_config.optimizer == "sgd":
            optimizer = SGD(
                param_groups,
                lr=base_lr,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum,
                nesterov=True
            )
            logger.info(f"Optimizer: SGD (base_lr={base_lr:.6f}, wd={opt_config.weight_decay}, momentum={opt_config.momentum})")

        else:
            raise ValueError(f"Unknown optimizer: {opt_config.optimizer}")

        # Build scheduler (same options as other classifiers)
        scheduler = None

        if sched_config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs - sched_config.warmup_epochs,
                eta_min=sched_config.eta_min
            )
            logger.info(
                f"Scheduler: CosineAnnealingLR "
                f"(T_max={self.config.training.max_epochs - sched_config.warmup_epochs}, "
                f"eta_min={sched_config.eta_min:.2e})"
            )
            logger.info(
                f"  Note: warmup_epochs={sched_config.warmup_epochs} is in config but "
                "not implemented (scheduler starts from epoch 0)"
            )

        elif sched_config.scheduler == "step":
            scheduler = StepLR(
                optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma
            )
            logger.info(
                f"Scheduler: StepLR "
                f"(step_size={sched_config.step_size}, gamma={sched_config.gamma})"
            )

        elif sched_config.scheduler == "constant":
            logger.info("Scheduler: None (constant learning rate)")

        else:
            logger.warning(
                f"Unknown scheduler '{sched_config.scheduler}', using constant LR"
            )

        return optimizer, scheduler
