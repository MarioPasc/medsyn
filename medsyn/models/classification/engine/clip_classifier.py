"""
CLIP ViT-B/16 classifier for medical image classification.

Uses OpenCLIP for vision-language pretraining with text prompts for classification.
Classification is performed via image-text similarity with learnable text embeddings.
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


class CLIPClassificationWrapper(nn.Module):
    """
    CLIP model wrapper for classification via image-text similarity.

    Architecture:
    - Vision encoder: CLIP ViT-B/16 extracts image features [B, embed_dim]
    - Text embeddings: Learnable class text features [num_classes, embed_dim]
    - Classification: Cosine similarity with temperature scaling

    Forward pass:
    1. Encode image with vision encoder -> [B, embed_dim]
    2. Normalize image and text features
    3. Compute similarity: image_features @ text_features.T -> [B, num_classes]
    4. Scale by temperature -> logits
    """

    def __init__(
        self,
        visual: nn.Module,
        text_encoder: nn.Module,
        initial_text_features: torch.Tensor,
        logit_scale: float
    ):
        """
        Initialize CLIP classification wrapper.

        Args:
            visual: CLIP vision encoder (ViT-B/16)
            text_encoder: CLIP text encoder (kept for potential use, not in forward)
            initial_text_features: Initial text embeddings [num_classes, embed_dim]
            logit_scale: Temperature parameter for scaling logits
        """
        super().__init__()
        self.visual = visual
        self.text_encoder = text_encoder  # Kept for reference, not used in forward

        # Make text features learnable (allows fine-tuning to medical domain)
        self.text_features = nn.Parameter(initial_text_features.clone())

        # Logit scale (temperature parameter) - learnable
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale).log())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute classification logits via image-text similarity.

        Args:
            images: [B, 3, H, W] normalized images (ImageNet normalization)

        Returns:
            logits: [B, num_classes] classification logits
        """
        # Encode image
        image_features = self.visual(images)

        # Normalize features to unit vectors (required for cosine similarity)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        # Compute similarity and scale by temperature
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)

        return logits


class CLIPViTB16Classifier(BaseClassifier):
    """
    CLIP ViT-B/16 classifier for medical image classification.

    Uses OpenCLIP with vision-language pretraining. Classification is performed
    via image-text similarity using prompts: "A histopathological image of {class_name}".

    Key features:
    - Full CLIP with vision + text encoders
    - Text prompts for class-aware classification
    - Learnable text embeddings (fine-tuned for medical domain)
    - Differential learning rates (encoder: 0.1x, text: 0.5x, scale: 1.0x)
    - Inherits training loop, validation, k-fold CV, metrics from BaseClassifier
    """

    def __init__(self, config: ClassificationConfig):
        """
        Initialize CLIP ViT-B/16 classifier.

        Args:
            config: ClassificationConfig instance

        Raises:
            ImportError: If open-clip-torch is not installed
        """
        # Check dependencies first
        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open-clip-torch is required for CLIP classifier.\n"
                "Install with: pip install open-clip-torch>=2.0.0\n"
                "Or install distdiff extra: pip install -e .[distdiff]"
            )

        super().__init__(config)

        # Validate config
        if config.model.name != "clip_vit_b16":
            logger.warning(f"Model name is '{config.model.name}' but using CLIPViTB16Classifier")

    def build_model(self) -> nn.Module:
        """
        Build CLIP ViT-B/16 model with text-based classification.

        Steps:
        1. Load pretrained CLIP from OpenCLIP
        2. Get class names (PathMNIST by default or from config)
        3. Generate text embeddings for all classes using prompts
        4. Build classification wrapper with trainable components
        5. Apply backbone freezing if requested

        Returns:
            nn.Module: CLIPClassificationWrapper model
        """
        import open_clip
        from medsyn.data.yolo_dataset import build_pathmnist_class_map

        logger.info("="*60)
        logger.info("Building CLIP ViT-B/16 classifier")
        logger.info("="*60)

        # Load pretrained CLIP model
        clip_pretrained = getattr(self.config.model, 'clip_pretrained', 'openai')
        logger.info(f"Loading CLIP model (pretrained={self.config.model.pretrained}, source={clip_pretrained})")

        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16',
            pretrained=clip_pretrained if self.config.model.pretrained else None
        )
        model.eval()  # Set to eval mode for text encoding

        # Get class names
        if getattr(self.config.model, 'class_names', None):
            class_names = self.config.model.class_names
            logger.info(f"Using custom class names from config: {len(class_names)} classes")
        else:
            # Default to PathMNIST
            class_map = build_pathmnist_class_map()
            class_names = [class_map[i] for i in range(self.config.model.num_classes)]
            logger.info(f"Using PathMNIST class names: {self.config.model.num_classes} classes")

        logger.info(f"Classes: {class_names}")

        # Generate text prompts
        prompt_template = getattr(
            self.config.model,
            'prompt_template',
            'A histopathological image of {}'
        )
        prompts = [prompt_template.format(name) for name in class_names]
        logger.info(f"Prompt template: '{prompt_template}'")
        logger.info(f"Example prompt: '{prompts[0]}'")

        # Tokenize and encode text
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        text_tokens = tokenizer(prompts).to(self.device)

        logger.info("Generating text embeddings...")
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
            # Normalize to unit vectors
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logger.info(f"Text embeddings shape: {text_features.shape}")
        logger.info(f"Embedding dimension: {text_features.shape[1]}")

        # Build classification wrapper
        clip_model = CLIPClassificationWrapper(
            visual=model.visual,
            text_encoder=model,
            initial_text_features=text_features,
            logit_scale=model.logit_scale.exp().item()
        )

        # Handle backbone freezing
        if self.config.model.freeze_backbone:
            logger.info("Freezing vision encoder (only training text embeddings + logit scale)")
            for param in clip_model.visual.parameters():
                param.requires_grad = False

            # Count frozen vs trainable params
            frozen_params = sum(p.numel() for p in clip_model.visual.parameters())
            trainable_params = sum(
                p.numel() for p in clip_model.parameters() if p.requires_grad
            )
            logger.info(f"  Frozen params: {frozen_params:,}")
            logger.info(f"  Trainable params: {trainable_params:,}")
        else:
            total_params = sum(p.numel() for p in clip_model.parameters())
            trainable_params = sum(
                p.numel() for p in clip_model.parameters() if p.requires_grad
            )
            logger.info(f"Training full model (vision + text embeddings + logit scale)")
            logger.info(f"  Total params: {total_params:,}")
            logger.info(f"  Trainable params: {trainable_params:,}")

        logger.info("="*60)
        logger.info("CLIP model built successfully")
        logger.info("="*60)

        return clip_model

    def build_optimizer(self) -> Tuple[Optimizer, Optional[_LRScheduler]]:
        """
        Build optimizer with differential learning rates for CLIP components.

        Parameter groups (with differential learning rates):
        - Vision encoder: base_lr * 0.1 (pretrained, needs smaller updates)
        - Text embeddings: base_lr * 0.5 (pretrained but class-specific)
        - Logit scale: base_lr * 1.0 (small parameter, quick adaptation)

        Returns:
            Tuple of (optimizer, scheduler). Scheduler can be None.
        """
        opt_config = self.config.optimization
        sched_config = self.config.scheduler

        logger.info("Building optimizer with differential learning rates...")

        # Separate parameters into groups
        vision_params = []
        text_params = []
        scale_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'visual' in name:
                vision_params.append(param)
            elif 'text_features' in name:
                text_params.append(param)
            elif 'logit_scale' in name:
                scale_params.append(param)

        base_lr = opt_config.lr

        # Build parameter groups with differential LRs
        param_groups = []

        if vision_params:
            param_groups.append({
                'params': vision_params,
                'lr': base_lr * 0.1,
                'name': 'vision_encoder'
            })

        if text_params:
            param_groups.append({
                'params': text_params,
                'lr': base_lr * 0.5,
                'name': 'text_embeddings'
            })

        if scale_params:
            param_groups.append({
                'params': scale_params,
                'lr': base_lr,
                'name': 'logit_scale'
            })

        # Log parameter group info
        logger.info("Parameter groups:")
        for group in param_groups:
            n_params = sum(p.numel() for p in group['params'])
            logger.info(f"  {group['name']}: {n_params:,} params, lr={group['lr']:.6f}")

        # Build optimizer (same options as ResNet50)
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

        # Build scheduler (same options as ResNet50)
        scheduler = None

        if sched_config.scheduler == "cosine":
            # Note: warmup_epochs is in config but not implemented in base class
            # Scheduler starts from epoch 0
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
