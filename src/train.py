"""
train.py — Full training pipeline with early stopping, LR scheduling, and logging
"""

import os
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

from model import SkinLesionClassifier, save_model
from dataset import get_dataloaders
from evaluate import compute_metrics


# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir":     "data",
    "output_dir":   "outputs",
    "batch_size":   32,
    "epochs":       20,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "dropout":      0.3,
    "patience":     5,        # early stopping patience
    "num_workers":  2,
    "freeze_epochs": 2,       # freeze backbone for first N epochs (feature extraction phase)
    "limit":        None,     # set to e.g. 2000 for quick debug runs
}


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc="Validating", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


# ── Main training function ────────────────────────────────────────────────────
def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Paths
    data_dir   = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        train_csv    = str(data_dir / "train.csv"),
        val_csv      = str(data_dir / "val.csv"),
        image_dir    = str(data_dir / "images"),
        batch_size   = config["batch_size"],
        num_workers  = config["num_workers"],
        limit        = config["limit"],
    )

    # Model — start with frozen backbone (feature extraction)
    model = SkinLesionClassifier(
        dropout=config["dropout"],
        freeze_backbone=True,
    ).to(device)

    # Loss — use label smoothing to reduce overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer (only train classifier head at first)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Training state
    best_auc    = 0.0
    no_improve  = 0
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    print(f"\nStarting training for {config['epochs']} epochs...\n")

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        # Unfreeze backbone after freeze_epochs
        if epoch == config["freeze_epochs"] + 1:
            print("\nUnfreezing backbone for full fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Lower LR for fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = config["lr"] / 10

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            save_model(
                model, optimizer, epoch,
                path=str(output_dir / "best_model.pth"),
                extra={"val_auc": val_auc, "val_acc": val_acc, "config": config},
            )
            print(f"  New best AUC: {best_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no AUC improvement for {config['patience']} epochs)")
                break

    # Save training history
    import json
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Val AUC: {best_auc:.4f}")
    print(f"   Model saved to {output_dir / 'best_model.pth'}")
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
    parser.add_argument("--data-dir",      default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir",    default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",            type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--limit",         type=int,   default=DEFAULT_CONFIG["limit"],
                        help="Limit dataset size for quick debug runs")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    train(config)
