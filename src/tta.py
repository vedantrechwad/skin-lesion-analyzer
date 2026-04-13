"""
tta.py — Test-Time Augmentation (TTA)

Runs inference on multiple augmented versions of the input image and
averages the predictions.  Typically gives +1-3% AUC with zero retraining.

Works with any model (binary or multi-class).
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Optional
import numpy as np

from dataset import IMAGE_SIZE, MEAN, STD


# ── TTA augmentation transforms ─────────────────────────────────────────────
def _get_tta_transforms() -> list[transforms.Compose]:
    """Returns a list of augmentation pipelines for TTA."""
    base_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    normalize = transforms.Normalize(MEAN, STD)

    augments = [
        # 0: Original (no augmentation)
        transforms.Compose([base_resize, transforms.ToTensor(), normalize]),
        # 1: Horizontal flip
        transforms.Compose([base_resize, transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), normalize]),
        # 2: Vertical flip
        transforms.Compose([base_resize, transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), normalize]),
        # 3: Both flips
        transforms.Compose([
            base_resize,
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(), normalize,
        ]),
        # 4: 90° rotation
        transforms.Compose([
            base_resize,
            transforms.Lambda(lambda img: img.rotate(90)),
            transforms.ToTensor(), normalize,
        ]),
        # 5: 180° rotation
        transforms.Compose([
            base_resize,
            transforms.Lambda(lambda img: img.rotate(180)),
            transforms.ToTensor(), normalize,
        ]),
        # 6: 270° rotation
        transforms.Compose([
            base_resize,
            transforms.Lambda(lambda img: img.rotate(270)),
            transforms.ToTensor(), normalize,
        ]),
        # 7: Slight brightness/contrast
        transforms.Compose([
            base_resize,
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(), normalize,
        ]),
    ]
    return augments


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    pil_image,
    device: torch.device,
    n_augments: int = 8,
) -> np.ndarray:
    """
    Run TTA on a PIL image and return averaged class probabilities.

    Args:
        model:       trained model (binary or multi-class)
        pil_image:   PIL.Image in RGB
        device:      torch device
        n_augments:  number of augmentations to use (1-8, default=8)

    Returns:
        probs: numpy array of shape [num_classes] with averaged probabilities
    """
    model.eval()
    tta_transforms = _get_tta_transforms()[:n_augments]

    all_probs = []
    for t in tta_transforms:
        tensor = t(pil_image).unsqueeze(0).to(device)
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
        all_probs.append(probs)

    # Average across all augmentations
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs
