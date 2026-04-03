"""
dataset.py — ISIC dataset loading, augmentation, and class-balancing utilities
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ── Image constants ──────────────────────────────────────────────────────────
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet stats (EfficientNet was trained on these)
STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Benign", "Malignant"]


# ── Transforms ───────────────────────────────────────────────────────────────
def get_transforms(split: str = "train") -> transforms.Compose:
    """
    Returns augmentation pipeline.
    - train: heavy augmentation to reduce overfitting
    - val/test: only resize + normalize
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


# ── Dataset class ─────────────────────────────────────────────────────────────
class ISICDataset(Dataset):
    """
    Loads ISIC skin lesion images from a CSV manifest.

    Expected CSV columns:
        image_name  — filename without extension (e.g., ISIC_0024306)
        target      — 0 (benign) or 1 (malignant)

    Images should live in `image_dir` as .jpg files.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
        limit: Optional[int] = None,
    ):
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.sample(min(limit, len(self.df)), random_state=42).reset_index(drop=True)

        self.image_dir = Path(image_dir)
        self.transform = transform

        # Build label map
        self.labels = self.df["target"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            p = self.image_dir / f"{stem}{ext}"
            if p.is_file():
                return p
        return self.image_dir / f"{stem}.jpg"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(str(row["image_name"]))

        img = Image.open(image_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, int(row["target"])

    def class_counts(self) -> dict:
        counts = self.df["target"].value_counts().to_dict()
        return {CLASS_NAMES[k]: v for k, v in sorted(counts.items())}


# ── Weighted sampler for class imbalance ─────────────────────────────────────
def make_weighted_sampler(dataset: ISICDataset) -> WeightedRandomSampler:
    """
    Creates a sampler that oversamples the minority class (malignant)
    so that each training batch is roughly balanced.
    """
    labels = dataset.labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


# ── DataLoader factory ───────────────────────────────────────────────────────
def get_dataloaders(
    train_csv: str,
    val_csv: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = ISICDataset(train_csv, image_dir, transform=get_transforms("train"), limit=limit)
    val_ds   = ISICDataset(val_csv,   image_dir, transform=get_transforms("val"),   limit=limit)

    sampler = make_weighted_sampler(train_ds)

    print(f"Train class distribution: {train_ds.class_counts()}")
    print(f"Val   class distribution: {val_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ── Denormalize helper (for visualization) ───────────────────────────────────
def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Converts a normalized tensor back to a displayable numpy image."""
    mean = np.array(MEAN)
    std  = np.array(STD)
    img  = tensor.permute(1, 2, 0).numpy()
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
