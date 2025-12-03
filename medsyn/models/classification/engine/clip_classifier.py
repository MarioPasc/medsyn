"""
CLIP ViT-B/16 classifier for medical image classification (STUB).
"""

from __future__ import annotations
from typing import Tuple, Optional
import logging
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .base_classifier import BaseClassifier
from ..config import ClassificationConfig

logger = logging.getLogger(__name__)


class CLIPViTB16Classifier(BaseClassifier):
    """
    CLIP ViT-B/16 classifier for medical image classification.

    NOTE: This is a stub implementation. Full implementation requires:
    - Install transformers or open_clip
    - Load CLIP model and freeze vision encoder
    - Add classification head on top of CLIP features

    TODO for future implementation:
    1. Install dependencies:
       pip install transformers  # or open-clip-torch
    2. Load CLIP model:
       from transformers import CLIPVisionModel
       model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
    3. Extract vision encoder and add classification head
    4. Configure separate learning rates for encoder vs head
    5. Add CLIP-specific preprocessing transforms
    """

    def __init__(self, config: ClassificationConfig):
        """
        Initialize CLIP ViT-B/16 classifier.

        Args:
            config: ClassificationConfig instance

        Raises:
            NotImplementedError: This classifier is not yet implemented
        """
        super().__init__(config)
        raise NotImplementedError(
            "CLIPViTB16Classifier is not yet implemented. "
            "Please use ResNet50Classifier for now.\n\n"
            "To implement CLIP support:\n"
            "1. Install: pip install transformers  # or open-clip-torch\n"
            "2. Implement build_model() to load CLIP vision encoder\n"
            "3. Implement build_optimizer() with differential LR for encoder/head\n"
            "4. Update dataloaders.py to use CLIP preprocessing (different from ImageNet)\n"
        )

    def build_model(self) -> nn.Module:
        """
        Build CLIP ViT-B/16 model with custom classifier head.

        TODO:
        1. Load CLIP model from transformers or open_clip
        2. Extract vision encoder
        3. Add classification head
        4. Return composed model

        Example implementation:
        ```python
        from transformers import CLIPVisionModel
        import torch.nn as nn

        # Load pretrained CLIP vision encoder
        clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        # Freeze CLIP encoder if requested
        if self.config.model.freeze_backbone:
            for param in clip_model.parameters():
                param.requires_grad = False

        # Add classification head
        clip_hidden_size = clip_model.config.hidden_size  # 768 for ViT-B/16
        classifier = nn.Linear(clip_hidden_size, self.config.model.num_classes)

        # Compose model
        class CLIPClassifier(nn.Module):
            def __init__(self, clip_encoder, classifier):
                super().__init__()
                self.clip_encoder = clip_encoder
                self.classifier = classifier

            def forward(self, x):
                # x: [B, 3, H, W] images
                outputs = self.clip_encoder(pixel_values=x)
                pooled_output = outputs.pooler_output  # [B, hidden_size]
                logits = self.classifier(pooled_output)  # [B, num_classes]
                return logits

        model = CLIPClassifier(clip_model, classifier)
        return model
        ```

        Raises:
            NotImplementedError: CLIP model implementation pending
        """
        raise NotImplementedError("CLIP model implementation pending")

    def build_optimizer(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Build optimizer and scheduler for CLIP model.

        TODO:
        1. Use different LR for CLIP encoder vs classifier head
        2. Consider lower LR for frozen/pretrained components

        Example implementation:
        ```python
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Separate parameter groups for differential learning rates
        clip_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'clip_encoder' in name:
                    clip_params.append(param)
                else:
                    head_params.append(param)

        # Lower LR for pretrained encoder, higher LR for new head
        optimizer = AdamW([
            {'params': clip_params, 'lr': self.config.optimization.lr * 0.1},
            {'params': head_params, 'lr': self.config.optimization.lr}
        ], weight_decay=self.config.optimization.weight_decay)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.max_epochs,
            eta_min=self.config.scheduler.eta_min
        )

        return optimizer, scheduler
        ```

        Raises:
            NotImplementedError: CLIP optimizer implementation pending
        """
        raise NotImplementedError("CLIP optimizer implementation pending")
