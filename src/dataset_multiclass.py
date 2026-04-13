"""
dataset_multiclass.py — Multi-class dataset for 9 ISIC diagnostic categories

Extends the existing dataset infrastructure with multi-class labels
and Mixup/CutMix augmentation.

Classes:
    0: actinic keratosis      (Pre-cancerous)
    1: basal cell carcinoma    (Malignant)
    2: dermatofibroma          (Benign)
    3: melanoma                (Malignant)
    4: nevus                   (Benign)
    5: pigmented benign keratosis (Benign)
    6: seborrheic keratosis    (Benign)
    7: squamous cell carcinoma (Malignant)
    8: vascular lesion         (Benign)
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

from dataset import IMAGE_SIZE, MEAN, STD, get_transforms


# ── Multi-class constants ────────────────────────────────────────────────────

MULTICLASS_NAMES = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

MULTICLASS_SHORT = [
    "AK", "BCC", "DF", "MEL", "NV", "BKL", "SK", "SCC", "VASC"
]

RISK_LEVELS = {
    "actinic keratosis":          "Pre-cancerous",
    "basal cell carcinoma":       "Malignant",
    "dermatofibroma":             "Benign",
    "melanoma":                   "Malignant",
    "nevus":                      "Benign",
    "pigmented benign keratosis": "Benign",
    "seborrheic keratosis":       "Benign",
    "squamous cell carcinoma":    "Malignant",
    "vascular lesion":            "Benign",
}

RISK_COLORS = {
    "Malignant":      "#ef4444",
    "Pre-cancerous":  "#f59e0b",
    "Benign":         "#22c55e",
}

NUM_CLASSES = len(MULTICLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(MULTICLASS_NAMES)}


# ── Dataset class ────────────────────────────────────────────────────────────

class ISICMulticlassDataset(Dataset):
    """
    Loads ISIC skin lesion images with multi-class labels.

    Expected CSV columns:
        image_name   — filename without extension
        class_label  — one of MULTICLASS_NAMES
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

        # Map class labels to indices
        self.df["target"] = self.df["class_label"].map(CLASS_TO_IDX)
        if self.df["target"].isna().any():
            unknown = self.df[self.df["target"].isna()]["class_label"].unique()
            raise ValueError(f"Unknown class labels: {unknown}. Expected: {MULTICLASS_NAMES}")
        self.labels = self.df["target"].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, stem: str) -> Path:
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
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
        counts = self.df["class_label"].value_counts().to_dict()
        return counts


# ── Mixup / CutMix augmentation ─────────────────────────────────────────────

class MixupCutmixCollator:
    """
    Batch-level Mixup and CutMix augmentation.

    With probability `mixup_prob`, applies Mixup.
    With probability `cutmix_prob`, applies CutMix.
    Otherwise, returns the batch unchanged.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, mixup_prob: float = 0.3,
                 cutmix_prob: float = 0.3, alpha: float = 1.0):
        self.num_classes = num_classes
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.alpha = alpha

    def __call__(self, batch):
        images = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)

        r = random.random()
        if r < self.mixup_prob:
            images, targets_a, targets_b, lam = self._mixup(images, targets)
            return images, (targets_a, targets_b, lam)
        elif r < self.mixup_prob + self.cutmix_prob:
            images, targets_a, targets_b, lam = self._cutmix(images, targets)
            return images, (targets_a, targets_b, lam)
        else:
            return images, targets

    def _mixup(self, images, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed = lam * images + (1 - lam) * images[index]
        return mixed, targets, targets[index], lam

    def _cutmix(self, images, targets):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        _, _, h, w = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(w, cx + cut_w // 2)
        y2 = min(h, cy + cut_h // 2)

        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)
        return images, targets, targets[index], lam


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_multiclass_dataloaders(
    train_csv: str,
    val_csv: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    limit: Optional[int] = None,
    use_mixup: bool = True,
) -> Tuple[DataLoader, DataLoader]:

    train_ds = ISICMulticlassDataset(train_csv, image_dir, transform=get_transforms("train"), limit=limit)
    val_ds   = ISICMulticlassDataset(val_csv, image_dir, transform=get_transforms("val"), limit=limit)

    # Weighted sampler for class balance
    labels = train_ds.labels
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    collator = MixupCutmixCollator() if use_mixup else None

    print(f"Multi-class train distribution: {train_ds.class_counts()}")
    print(f"Multi-class val   distribution: {val_ds.class_counts()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
