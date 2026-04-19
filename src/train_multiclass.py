"""
train_multiclass.py — Multi-class training with Focal Loss, Mixup/CutMix,
                       and Progressive Resizing

Trains a 9-class skin lesion classifier using the ISIC class-folder dataset.
Saves to outputs/best_model_multiclass.pth (separate from binary model).

Usage:
    python src/train_multiclass.py
    python src/train_multiclass.py --epochs 25 --lr 3e-4 --limit 200
"""

import os
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report

from model import SkinLesionClassifier, save_model
from dataset import IMAGE_SIZE, MEAN, STD
from dataset_multiclass import (
    get_multiclass_dataloaders, NUM_CLASSES, MULTICLASS_NAMES,
    MixupCutmixCollator,
)
from focal_loss import FocalLoss


# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir":       "data",
    "output_dir":     "outputs",
    "models_dir":     "models",
    "batch_size":     32,
    "epochs":         20,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "dropout":        0.3,
    "patience":       6,
    "num_workers":    2,
    "freeze_epochs":  2,
    "limit":          None,
    "progressive":    True,     # progressive resizing: phase 1 = 224, phase 2 = 300
    "focal_gamma":    2.0,      # focal loss gamma
}


# ── Training loop with Mixup/CutMix support ─────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images = images.to(device)

        optimizer.zero_grad()

        # Handle Mixup/CutMix: targets may be a tuple (targets_a, targets_b, lam)
        if isinstance(targets, tuple):
            targets_a, targets_b, lam = targets
            targets_a = targets_a.to(device)
            targets_b = targets_b.to(device)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            # Accuracy approximation: use original labels
            preds = outputs.argmax(dim=1)
            correct += (lam * (preds == targets_a).float() + (1 - lam) * (preds == targets_b).float()).sum().item()
        else:
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += images.size(0)
        total_loss += loss.item() * images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    acc = correct / total
    return total_loss / total, acc, np.array(all_labels), np.array(all_preds)


# ── Main training function ────────────────────────────────────────────────────
def train_multiclass(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Training {NUM_CLASSES}-class model: {MULTICLASS_NAMES}\n")

    data_dir = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    models_dir = Path(config["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_csv = str(data_dir / "train_multiclass.csv")
    val_csv   = str(data_dir / "val_multiclass.csv")
    image_dir = str(data_dir / "images")

    # Check files exist
    for f in (train_csv, val_csv):
        if not Path(f).exists():
            raise FileNotFoundError(
                f"{f} not found. Run: python src/prepare_data.py "
                f"--isic-dir \"Skin cancer ISIC The International Skin Imaging Collaboration\" --multiclass"
            )

    # Data
    train_loader, val_loader = get_multiclass_dataloaders(
        train_csv=train_csv, val_csv=val_csv, image_dir=image_dir,
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        limit=config["limit"], use_mixup=True,
    )

    # Model — 9-class EfficientNet-B0
    model = SkinLesionClassifier(
        num_classes=NUM_CLASSES,
        dropout=config["dropout"],
        freeze_backbone=True,
    ).to(device)

    # Focal Loss
    criterion = FocalLoss(gamma=config["focal_gamma"], label_smoothing=0.1)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Training state
    best_acc = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    save_path = str(models_dir / "best_model_multiclass.pth")

    print(f"Starting multi-class training for {config['epochs']} epochs...\n")

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        # Unfreeze backbone after freeze_epochs
        if epoch == config["freeze_epochs"] + 1:
            print("\nUnfreezing backbone for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            for pg in optimizer.param_groups:
                pg["lr"] = config["lr"] / 10

        # Progressive resizing: switch to larger images in second half
        if config["progressive"] and epoch == config["epochs"] // 2 + 1:
            print(f"\nProgressive resizing: switching to 300x300...")
            large_train = transforms.Compose([
                transforms.Resize((332, 332)),
                transforms.RandomCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
            large_val = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
            from dataset_multiclass import ISICMulticlassDataset, MixupCutmixCollator
            from torch.utils.data import WeightedRandomSampler
            train_ds = ISICMulticlassDataset(train_csv, image_dir, transform=large_train, limit=config["limit"])
            val_ds = ISICMulticlassDataset(val_csv, image_dir, transform=large_val, limit=config["limit"])
            labels = train_ds.labels
            class_counts = np.bincount(labels, minlength=NUM_CLASSES)
            class_weights = 1.0 / (class_counts + 1e-8)
            sample_weights = class_weights[labels]
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).float(),
                num_samples=len(sample_weights), replacement=True,
            )
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=config["batch_size"], sampler=sampler,
                num_workers=config["num_workers"], pin_memory=True,
                collate_fn=MixupCutmixCollator(),
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False,
                num_workers=config["num_workers"], pin_memory=True,
            )

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        marker = " ★ BEST" if val_acc > best_acc else ""
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            save_model(
                model, optimizer, epoch, path=save_path,
                extra={
                    "val_acc": val_acc,
                    "num_classes": NUM_CLASSES,
                    "class_names": MULTICLASS_NAMES,
                    "config": config,
                },
            )
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save history
    with open(output_dir / "history_multiclass.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final classification report
    print(f"\nTraining complete. Best Val Accuracy: {best_acc:.4f}")
    print(f"Model saved to {save_path}")

    # Print per-class report from last validation
    print("\nPer-class validation report:")
    print(classification_report(val_labels, val_preds, target_names=MULTICLASS_NAMES, zero_division=0))

    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-class skin lesion classifier")
    parser.add_argument("--data-dir",    default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir",  default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--limit",       type=int,   default=DEFAULT_CONFIG["limit"],
                        help="Limit dataset size for debug runs")
    parser.add_argument("--no-progressive", action="store_true",
                        help="Disable progressive resizing")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    config["progressive"] = not args.no_progressive
    train_multiclass(config)
