import albumentations as A
from typing import List, Dict, Any
import torch

class Augmentor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._build_transforms()

    def _build_common(self) -> List[A.BasicTransform]:
        transforms = []
        resize = self.config.get('resize')
        if resize is not None and resize != 'None':
            transforms.append(A.Resize(*resize))
        transforms.append(A.Normalize(normalization='standard', mean=(0.5,), std=(0.5,)))
        transforms.append(A.ToTensorV2())
        return transforms

    def _build_transforms(self) -> None:
        train_common = self._build_common()
        dataset = self.config.get('dataset', '').lower()

        # --- MNIST: safe geometric transformations ---
        if dataset == 'mnist':
            self.train_transform = A.Compose([
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=self.config['probability']
                ),
                # Better parameters for small images (28x28)
                A.ElasticTransform(
                    alpha=1.0,        # Deformation intensity
                    sigma=4,          # Smoothness (small = local warps)
                    alpha_affine=5,   # Affine deformation strength
                    p=self.config['probability']
                ),
                *train_common
            ], seed=self.config['seed'])

        # --- CIFAR: standard augmentations ---
        elif dataset in ('cifar10', 'cifar100'):
            self.train_transform = A.Compose([
                # Always apply a random crop (with optional padding)
                A.PadIfNeeded(min_height=36, min_width=36, border_mode=0, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(p=self.config['probability']),
                A.RandomFog(fog_coef_lower=0.1, p=self.config['probability']),
                A.CoarseDropout(
                    num_holes_range=(1, 3), 
                    hole_height_range=(4, 8), 
                    hole_width_range=(4, 8), 
                    fill=0, 
                    p=self.config['probability']
                ),
                *train_common
            ], seed=self.config['seed'])

        # --- Fallback: no random augmentations ---
        else:
            self.train_transform = A.Compose(
                train_common,
                seed=self.config['seed']
            )

        # --- Validation & Test (deterministic) ---
        val_common = self._build_common()
        self.val_transform = A.Compose(val_common, seed=self.config['seed'])
        self.test_transform = A.Compose(val_common, seed=self.config['seed'])  # same as val

    def train(self, image) -> torch.Tensor:
        return self.train_transform(image=image)['image']

    def val(self, image) -> torch.Tensor:
        return self.val_transform(image=image)['image']

    def test(self, image) -> torch.Tensor:
        return self.test_transform(image=image)['image']