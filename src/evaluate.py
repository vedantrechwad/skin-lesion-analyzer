"""
evaluate.py — Evaluation metrics, confusion matrix, ROC curve, and full test report
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from dataset import CLASS_NAMES


# ── Core metric computation ───────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute key metrics for binary classification.
    In medical settings, sensitivity (recall for malignant) is critical.
    """
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)   # True Positive Rate (recall for malignant)
    specificity = tn / (tn + fp + 1e-8)   # True Negative Rate
    precision   = tp / (tp + fp + 1e-8)
    f1          = 2 * precision * sensitivity / (precision + sensitivity + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    auc         = roc_auc_score(labels, probs)
    avg_prec    = average_precision_score(labels, probs)

    return {
        "accuracy":    round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),   # How often we catch malignant
        "specificity": round(specificity, 4),   # How often we correctly clear benign
        "precision":   round(precision, 4),
        "f1":          round(f1, 4),
        "auc":         round(auc, 4),
        "avg_precision": round(avg_prec, 4),
        "confusion_matrix": cm.tolist(),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ── Full evaluation pass ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, loader: DataLoader, device: torch.device) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    metrics    = compute_metrics(all_labels, all_probs)

    return metrics, all_labels, all_probs


# ── Plot: ROC Curve ───────────────────────────────────────────────────────────
def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: str = None):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    # Random baseline
    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1.5, label="Random (AUC=0.50)")

    # ROC curve with gradient-like coloring
    ax.plot(fpr, tpr, color="#00d2a0", lw=2.5, label=f"Model (AUC={auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#00d2a0")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", color="#ccc", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", color="#ccc", fontsize=12)
    ax.set_title("ROC Curve — Skin Lesion Classifier", color="#fff", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1a1a1a", labelcolor="#fff", fontsize=11)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 ROC curve saved → {save_path}")
    plt.show()
    plt.close()


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5, save_path: str = None):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=18, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES, color="#ccc", fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, color="#ccc", fontsize=12)
    ax.set_xlabel("Predicted", color="#ccc", fontsize=12)
    ax.set_ylabel("Actual", color="#ccc", fontsize=12)
    ax.set_title("Confusion Matrix", color="#fff", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved -> {save_path}")
    plt.show()
    plt.close()


# ── Plot: Training History ────────────────────────────────────────────────────
def plot_training_history(history_path: str, save_path: str = None):
    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0f0f0f")
    fig.suptitle("Training History", color="#fff", fontsize=15, fontweight="bold")

    specs = [
        ("Loss",     "train_loss", "val_loss"),
        ("Accuracy", "train_acc",  "val_acc"),
        ("Val AUC",  None,         "val_auc"),
    ]

    colors = {"train": "#4fa8ff", "val": "#00d2a0"}

    for ax, (title, train_key, val_key) in zip(axes, specs):
        ax.set_facecolor("#0f0f0f")
        if train_key:
            ax.plot(epochs, history[train_key], color=colors["train"], lw=2, label="Train")
        ax.plot(epochs, history[val_key], color=colors["val"], lw=2, label="Val")
        ax.set_title(title, color="#fff", fontsize=12)
        ax.set_xlabel("Epoch", color="#aaa")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.legend(facecolor="#1a1a1a", labelcolor="#fff")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📉 Training history saved → {save_path}")
    plt.show()
    plt.close()


# ── CLI: run full evaluation ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from model import load_model
    from dataset import ISICDataset, get_transforms
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--test-csv",   default="data/val.csv")
    parser.add_argument("--image-dir",  default="data/images")
    parser.add_argument("--output-dir", default="dashboards")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    ds     = ISICDataset(args.test_csv, args.image_dir, transform=get_transforms("val"))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    metrics, labels, probs = evaluate_model(model, loader, device)

    print("\n── Evaluation Results ──────────────────────────────")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:<20} {v}")
    print("────────────────────────────────────────────────────\n")

    out = Path(args.output_dir)
    plot_roc_curve(labels, probs, save_path=str(out / "roc_curve.png"))
    plot_confusion_matrix(labels, probs, save_path=str(out / "confusion_matrix.png"))

    if (out / "history.json").exists():
        plot_training_history(str(out / "history.json"), save_path=str(out / "training_history.png"))
