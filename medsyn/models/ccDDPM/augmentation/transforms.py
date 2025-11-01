# medsyn/models/ccDDPM/augmentation/transforms.py
# Purpose: Augmentation pipeline using albumentations library
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np
import torch

try:
    import albumentations as A
    import cv2  # Import cv2 at module level for worker process compatibility
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None
    cv2 = None

from .config import AugmentationConfig, TransformConfig

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """
    Manages augmentation pipeline using albumentations.
    Applies transforms to images and tracks which transforms were applied.
    """

    def __init__(self, config: AugmentationConfig):
        """
        Initialize augmentation pipeline.

        Args:
            config: Augmentation configuration

        Raises:
            ImportError: If albumentations is not installed
        """
        if not ALBUMENTATIONS_AVAILABLE:
            raise ImportError(
                "albumentations is required for data augmentation. "
                "Install it with: pip install albumentations"
            )

        self.config = config
        self.enabled = config.enabled

        if not self.enabled:
            logger.info("Augmentation is disabled")
            self.pipeline = None
            self.transform_names = []
            return

        # Build augmentation pipeline
        self.pipeline, self.transform_names = self._build_pipeline()

        logger.info(f"Augmentation pipeline initialized with {len(self.transform_names)} transforms")
        logger.info(f"  Transforms: {', '.join(self.transform_names)}")
        logger.info(f"  Overall probability: {config.probability}")

    def _build_pipeline(self) -> Tuple[Optional[A.Compose], List[str]]:
        """
        Build albumentations pipeline from configuration.

        Returns:
            Tuple of (albumentations Compose object, list of transform names)
        """
        if not self.config.transforms:
            logger.warning("No transforms configured, augmentation will be skipped")
            return None, []

        transforms = []
        transform_names = []

        for tf_config in self.config.transforms:
            try:
                transform = self._create_transform(tf_config)
                transforms.append(transform)
                transform_names.append(tf_config.name)
            except Exception as e:
                logger.error(f"Failed to create transform '{tf_config.name}': {e}")
                logger.warning(f"Skipping transform '{tf_config.name}'")

        if not transforms:
            logger.warning("No valid transforms created")
            return None, []

        # Create Compose pipeline
        # Note: We use additional_targets for handling multiple images if needed
        pipeline = A.Compose(
            transforms,
            # Track which transforms were applied
            # This is handled by our custom tracking in __call__
        )

        return pipeline, transform_names

    def _create_transform(self, tf_config: TransformConfig) -> A.BasicTransform:
        """
        Create a single albumentations transform from configuration.

        Args:
            tf_config: Transform configuration

        Returns:
            Albumentations transform object
        """
        transform_class = getattr(A, tf_config.name, None)

        if transform_class is None:
            raise ValueError(f"Unknown transform: {tf_config.name}")

        # Filter out invalid parameters for this transform
        import inspect
        valid_params = inspect.signature(transform_class.__init__).parameters
        filtered_params = {}
        invalid_params = []

        for key, value in tf_config.params.items():
            if key in valid_params:
                filtered_params[key] = value
            else:
                invalid_params.append(key)

        if invalid_params:
            logger.warning(
                f"Ignoring invalid parameter(s) {invalid_params} for transform {tf_config.name}. "
                f"Valid parameters: {list(valid_params.keys())}"
            )

        # CRITICAL FIX: For Rotate transform, ensure it never changes dimensions
        # This prevents batch collation errors in multi-worker DataLoader
        if tf_config.name == 'Rotate':
            # Set border_mode to ensure no dimension changes
            # cv2.BORDER_REFLECT_101 is the default and should preserve dimensions
            if 'border_mode' not in filtered_params and cv2 is not None:
                filtered_params['border_mode'] = cv2.BORDER_CONSTANT
                logger.debug(f"Added border_mode=BORDER_CONSTANT to Rotate transform for dimension stability")

        # Create transform with probability and filtered parameters
        return transform_class(p=tf_config.p, **filtered_params)

    def __call__(
        self,
        image: torch.Tensor,
        return_applied_transforms: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Apply augmentation pipeline to an image.

        Args:
            image: Input image tensor [C, H, W] in range [-1, 1]
            return_applied_transforms: If True, return list of applied transform names

        Returns:
            Tuple of (augmented image tensor, list of applied transform names)
            If augmentation is disabled or not applied, returns (original image, [])
        """
        if not self.enabled or self.pipeline is None:
            return image, []

        # Random check: should we apply augmentation to this image?
        if np.random.random() > self.config.probability:
            return image, []  # Skip augmentation for this image

        # Convert tensor to numpy for albumentations
        # Albumentations expects [H, W, C] in range [0, 255] for uint8 or [0, 1] for float32
        image_np = self._tensor_to_numpy(image)
        original_shape = image_np.shape

        # Apply augmentation
        try:
            augmented = self.pipeline(image=image_np)
            augmented_image = augmented['image']

            # CRITICAL: ALWAYS validate and enforce dimension preservation
            # This prevents batch collation errors in DataLoader (especially with multi-GPU + multi-worker)
            # Note: We check EVERY time, not just when dimensions change, as a safety measure
            if augmented_image.shape != original_shape:
                logger.warning(
                    f"Augmentation changed image dimensions from {original_shape} to {augmented_image.shape}! "
                    f"Reshaping back to original dimensions using bicubic interpolation. "
                    f"This can happen with transforms like Rotate. Consider using fixed-size transforms."
                )
                if cv2 is None:
                    raise RuntimeError(
                        "cv2 (opencv-python) is required for dimension correction but not available. "
                        "Install it with: pip install opencv-python"
                    )

                # Resize to match original H, W (shape is [H, W, C])
                augmented_image = cv2.resize(
                    augmented_image,
                    (original_shape[1], original_shape[0]),  # (width, height)
                    interpolation=cv2.INTER_CUBIC
                )

                # Ensure channel dimension is preserved (cv2.resize can drop channel dim for grayscale)
                if len(original_shape) == 3 and len(augmented_image.shape) == 2:
                    augmented_image = augmented_image[:, :, np.newaxis]

            # CRITICAL: Final safety check - ensure shape matches EXACTLY
            # This is essential for multi-worker DataLoader to avoid collation errors
            if augmented_image.shape != original_shape:
                raise RuntimeError(
                    f"Failed to restore original image dimensions! "
                    f"Expected {original_shape}, got {augmented_image.shape} after resize. "
                    f"This indicates a critical bug in dimension handling."
                )

            # Track which transforms were actually applied
            # Note: Albumentations doesn't directly expose this, so we track via replay
            applied_transforms = self._get_applied_transforms(image_np, augmented_image)

        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, using original image")
            return image, []

        # Convert back to tensor
        augmented_tensor = self._numpy_to_tensor(augmented_image)

        # Ensure output is in correct range
        if self.config.preserve_range:
            augmented_tensor = torch.clamp(augmented_tensor, -1.0, 1.0)

        return augmented_tensor, applied_transforms

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array for albumentations.

        Args:
            tensor: Image tensor [C, H, W] in range [-1, 1]

        Returns:
            Numpy array [H, W, C] in range [0, 1] (float32)
        """
        # Convert from [C, H, W] to [H, W, C]
        image_np = tensor.permute(1, 2, 0).cpu().numpy()

        # Convert from [-1, 1] to [0, 1]
        image_np = (image_np + 1.0) / 2.0

        # Clamp to valid range
        image_np = np.clip(image_np, 0.0, 1.0).astype(np.float32)

        return image_np

    def _numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        Convert numpy array back to PyTorch tensor.

        Args:
            array: Numpy array [H, W, C] in range [0, 1]

        Returns:
            Tensor [C, H, W] in range [-1, 1]
        """
        # Convert to tensor
        tensor = torch.from_numpy(array).float()

        # Convert from [H, W, C] to [C, H, W]
        tensor = tensor.permute(2, 0, 1)

        # Convert from [0, 1] to [-1, 1]
        tensor = tensor * 2.0 - 1.0

        return tensor

    def _get_applied_transforms(
        self,
        original: np.ndarray,
        augmented: np.ndarray
    ) -> List[str]:
        """
        Determine which transforms were applied by comparing images.

        Note: This is an approximation since albumentations doesn't expose
        which transforms were actually applied in a Compose pipeline.

        For better tracking, we could use ReplayCompose, but this adds overhead.
        For now, we use a simple heuristic: if image changed, we assume
        augmentation was applied.

        Args:
            original: Original image
            augmented: Augmented image

        Returns:
            List of transform names (empty if no change detected)
        """
        # Simple check: did the image change?
        if np.allclose(original, augmented, atol=1e-6):
            return []  # No change, no transforms applied

        # Image changed, so some augmentation was applied
        # We can't easily determine which specific transforms without ReplayCompose
        # For statistics, we'll mark this as "augmented"
        # To get more detailed tracking, we'd need to modify the approach

        # For now, return all transform names (since we can't determine which)
        # This is conservative: if ANY augmentation happened, we list all possible
        # Better solution: Use ReplayCompose for exact tracking
        # TODO: Consider implementing ReplayCompose for more accurate statistics

        return self.transform_names  # Conservative: mark all as potentially applied

    def get_transform_names(self) -> List[str]:
        """Get list of all configured transform names."""
        return self.transform_names.copy()


class ReplayAugmentationPipeline(AugmentationPipeline):
    """
    Enhanced pipeline that uses ReplayCompose for exact transform tracking.
    This provides accurate statistics but adds slight overhead.
    """

    def _build_pipeline(self) -> Tuple[Optional[A.ReplayCompose], List[str]]:
        """Build pipeline with ReplayCompose for accurate tracking."""
        if not self.config.transforms:
            return None, []

        transforms = []
        transform_names = []

        for tf_config in self.config.transforms:
            try:
                transform = self._create_transform(tf_config)
                transforms.append(transform)
                transform_names.append(tf_config.name)
            except Exception as e:
                logger.error(f"Failed to create transform '{tf_config.name}': {e}")

        if not transforms:
            return None, []

        # Use ReplayCompose instead of Compose
        pipeline = A.ReplayCompose(transforms)

        return pipeline, transform_names

    def __call__(
        self,
        image: torch.Tensor,
        return_applied_transforms: bool = True
    ) -> Tuple[torch.Tensor, List[str]]:
        """Apply augmentation and track exactly which transforms were used."""
        if not self.enabled or self.pipeline is None:
            return image, []

        if np.random.random() > self.config.probability:
            return image, []

        image_np = self._tensor_to_numpy(image)
        original_shape = image_np.shape

        try:
            augmented = self.pipeline(image=image_np)
            augmented_image = augmented['image']

            # CRITICAL: ALWAYS validate and enforce dimension preservation
            # This prevents batch collation errors in DataLoader (especially with multi-GPU + multi-worker)
            if augmented_image.shape != original_shape:
                logger.warning(
                    f"Augmentation changed image dimensions from {original_shape} to {augmented_image.shape}! "
                    f"Reshaping back to original dimensions using bicubic interpolation. "
                    f"This can happen with transforms like Rotate. Consider using fixed-size transforms."
                )
                if cv2 is None:
                    raise RuntimeError(
                        "cv2 (opencv-python) is required for dimension correction but not available. "
                        "Install it with: pip install opencv-python"
                    )

                # Resize to match original H, W (shape is [H, W, C])
                augmented_image = cv2.resize(
                    augmented_image,
                    (original_shape[1], original_shape[0]),  # (width, height)
                    interpolation=cv2.INTER_CUBIC
                )

                # Ensure channel dimension is preserved (cv2.resize can drop channel dim for grayscale)
                if len(original_shape) == 3 and len(augmented_image.shape) == 2:
                    augmented_image = augmented_image[:, :, np.newaxis]

            # CRITICAL: Final safety check - ensure shape matches EXACTLY
            # This is essential for multi-worker DataLoader to avoid collation errors
            if augmented_image.shape != original_shape:
                raise RuntimeError(
                    f"Failed to restore original image dimensions! "
                    f"Expected {original_shape}, got {augmented_image.shape} after resize. "
                    f"This indicates a critical bug in dimension handling."
                )

            # Extract applied transforms from replay data
            applied_transforms = []
            if 'replay' in augmented:
                for transform_dict in augmented['replay']['transforms']:
                    # Check if transform was actually applied
                    if transform_dict.get('applied', False):
                        transform_name = transform_dict.get('__class_fullname__', '').split('.')[-1]
                        if transform_name:
                            applied_transforms.append(transform_name)

        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, using original image")
            return image, []

        augmented_tensor = self._numpy_to_tensor(augmented_image)

        if self.config.preserve_range:
            augmented_tensor = torch.clamp(augmented_tensor, -1.0, 1.0)

        return augmented_tensor, applied_transforms


def create_augmentation_pipeline(
    config: AugmentationConfig,
    use_replay: bool = True
) -> AugmentationPipeline:
    """
    Factory function to create augmentation pipeline.

    Args:
        config: Augmentation configuration
        use_replay: If True, use ReplayCompose for accurate tracking (recommended)

    Returns:
        AugmentationPipeline instance
    """
    if not config.enabled:
        return AugmentationPipeline(config)

    if use_replay:
        return ReplayAugmentationPipeline(config)
    else:
        return AugmentationPipeline(config)
