"""
train_finetune.py — Fine-tune existing model on expanded dataset

Loads the best checkpoint and continues training on the (potentially larger)
merged dataset.  Uses discriminative learning rates: backbone gets 10x lower LR
than the classifier head.

The original model is backed up before overwriting so you can always roll back.

Usage:
    python src/train_finetune.py
    python src/train_finetune.py --epochs 20 --lr 1e-4 --patience 7
"""

import os
import argparse
import shutil
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

from model import SkinLesionClassifier, load_model, save_model
from dataset import get_dataloaders
from evaluate import compute_metrics


# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir":      "data",
    "output_dir":    "outputs",
    "batch_size":    32,
    "epochs":        15,
    "lr":            1e-4,         # head LR; backbone gets lr/10
    "weight_decay":  1e-4,
    "dropout":       0.3,
    "patience":      5,
    "num_workers":   2,
    "limit":         None,
    "checkpoint":    "outputs/best_model.pth",
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


# ── Main fine-tuning function ─────────────────────────────────────────────────
def finetune(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Paths
    data_dir   = Path(config["data_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config["checkpoint"])
    best_model_path = output_dir / "best_model.pth"
    backup_path     = output_dir / "best_model_backup.pth"
    finetune_path   = output_dir / "best_model_finetuned.pth"

    # Data
    train_loader, val_loader = get_dataloaders(
        train_csv    = str(data_dir / "train.csv"),
        val_csv      = str(data_dir / "val.csv"),
        image_dir    = str(data_dir / "images"),
        batch_size   = config["batch_size"],
        num_workers  = config["num_workers"],
        limit        = config["limit"],
    )

    # ── Load existing model ──────────────────────────────────────────────
    if checkpoint_path.exists():
        print(f"\nLoading existing checkpoint: {checkpoint_path}")
        model = SkinLesionClassifier(dropout=config["dropout"]).to(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        baseline_auc = checkpoint.get("val_auc", 0.0)
        baseline_epoch = checkpoint.get("epoch", "?")
        print(f"   Loaded model from epoch {baseline_epoch} with val AUC: {baseline_auc:.4f}")
    else:
        print(f"\nNo checkpoint found at {checkpoint_path} — training from scratch.")
        model = SkinLesionClassifier(dropout=config["dropout"]).to(device)
        baseline_auc = 0.0

    # ── All layers unfrozen with discriminative LR ───────────────────────
    # Backbone layers get lr/10, classifier head gets full lr
    for param in model.parameters():
        param.requires_grad = True

    backbone_params = list(model.backbone.features.parameters())
    head_params     = list(model.backbone.classifier.parameters())

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": config["lr"] / 10},
        {"params": head_params,     "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

    # Loss + scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Training state
    best_auc    = baseline_auc
    no_improve  = 0
    history     = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    print(f"\nStarting fine-tuning for {config['epochs']} epochs...")
    print(f"   Baseline AUC to beat: {baseline_auc:.4f}")
    print(f"   Backbone LR: {config['lr']/10:.1e} | Head LR: {config['lr']:.1e}\n")

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

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
        marker = ""
        if val_auc > best_auc:
            marker = " ★ NEW BEST"
        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} AUC: {val_auc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            save_model(
                model, optimizer, epoch,
                path=str(finetune_path),
                extra={"val_auc": val_auc, "val_acc": val_acc, "config": config},
            )
            print(f"  → Saved fine-tuned model (AUC: {best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= config["patience"]:
                print(f"\nEarly stopping at epoch {epoch} (no AUC improvement for {config['patience']} epochs)")
                break

    # Save training history
    history_path = output_dir / "history_finetune.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # ── Promote fine-tuned model if it beats the baseline ────────────────
    print(f"\n{'='*60}")
    print(f"  Fine-tuning complete.")
    print(f"  Baseline AUC:    {baseline_auc:.4f}")
    print(f"  Best new AUC:    {best_auc:.4f}")

    if best_auc > baseline_auc and finetune_path.exists():
        # Back up original
        if best_model_path.exists():
            shutil.copy2(str(best_model_path), str(backup_path))
            print(f"  Backed up original -> {backup_path}")

        # Promote
        shutil.copy2(str(finetune_path), str(best_model_path))
        print(f"  ✓ Promoted fine-tuned model -> {best_model_path}")
        print(f"  The app will now use the improved model automatically.")
    else:
        print(f"  ✗ Fine-tuned model did not beat baseline. Original model kept.")
        print(f"    Fine-tuned model still available at: {finetune_path}")

    print(f"{'='*60}")
    return history


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune skin lesion classifier on expanded dataset")
    parser.add_argument("--data-dir",    default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--output-dir",  default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--checkpoint",  default=DEFAULT_CONFIG["checkpoint"],
                        help="Path to existing model checkpoint to fine-tune from")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--limit",       type=int,   default=DEFAULT_CONFIG["limit"],
                        help="Limit dataset size for quick debug runs")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG, **vars(args)}
    finetune(config)
