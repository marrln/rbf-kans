import albumentations as A
from typing import List, Dict, Any
import torch

class Augmentor:
    """
    Handles all data transformations for training, validation and test.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Training configuration dict. Expects keys:
                - 'probability': float for random augmentations
                - 'resize': tuple (H, W) or 'None'
                - 'seed': int for reproducibility
        """
        self.config = config
        self._build_transforms()

    def _build_common(self) -> List[A.BasicTransform]:
        """
        Deterministic preprocessing shared across all pipelines.
        These do NOT introduce randomness.
        """
        transforms = []
        resize = self.config.get('resize')
        if resize is not None and resize != 'None':
            transforms.append(A.Resize(*resize))
        transforms.append(A.Normalize(normalization='min_max_per_channel'))
        transforms.append(A.ToTensorV2())
        return transforms

    def _build_transforms(self) -> None:
        # --- Training pipeline (with random augmentations) ---
        train_common = self._build_common()
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=self.config['probability']),
            A.VerticalFlip(p=self.config['probability']),
            A.SafeRotate(limit=(-90, 90), p=self.config['probability']),
            # Add more augmentations here if needed, e.g.:
            # A.ColorJitter(brightness=0.2, contrast=0.2, p=self.config['probability']),
            *train_common
        ], seed=self.config['seed'])

        # --- Validation pipeline (deterministic only) ---
        val_common = self._build_common()
        self.val_transform = A.Compose(
            val_common,
            seed=self.config['seed']
        )

        # --- Base transform for TTA (same as validation) ---
        self._base_tta_transform = A.Compose(
            val_common,
            seed=self.config['seed']
        )

    # Public methods for dataset preprocessing
    def train(self, image) -> torch.Tensor:
        """Apply training augmentations + preprocessing."""
        return self.train_transform(image=image)['image']

    def val(self, image) -> torch.Tensor:
        """Apply validation preprocessing (no augmentations)."""
        return self.val_transform(image=image)['image']

    def test(self, image) -> torch.Tensor:
        """Apply test-time preprocessing (same as validation)."""
        return self._base_tta_transform(image=image)['image']