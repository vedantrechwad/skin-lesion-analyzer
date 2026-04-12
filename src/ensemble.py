"""
ensemble.py — Ensemble prediction module

Loads multiple model checkpoints and averages their predictions for
more robust and accurate classification.

Works with any model architecture / number of classes.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional

from model import SkinLesionClassifier


class EnsemblePredictor:
    """
    Ensemble predictor that loads N model checkpoints and averages
    their softmax outputs.

    Args:
        model_paths:  list of checkpoint file paths
        device:       torch device
        num_classes:  number of output classes (default 2 for binary)
        weights:      optional per-model weights (e.g., by validation AUC)
    """

    def __init__(
        self,
        model_paths: list[str],
        device: torch.device,
        num_classes: int = 2,
        weights: Optional[list[float]] = None,
    ):
        self.device = device
        self.models = []
        self.weights = weights

        for path in model_paths:
            p = Path(path)
            if not p.exists():
                print(f"  Warning: checkpoint not found, skipping: {path}")
                continue

            model = SkinLesionClassifier(num_classes=num_classes).to(device)
            checkpoint = torch.load(str(p), map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            val_auc = checkpoint.get("val_auc", None)
            print(f"  Loaded: {p.name} (AUC: {val_auc:.4f})" if val_auc else f"  Loaded: {p.name}")
            self.models.append(model)

        if not self.models:
            raise RuntimeError("No valid model checkpoints found for ensemble.")

        # Normalize weights
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            w_sum = sum(self.weights[:len(self.models)])
            self.weights = [w / w_sum for w in self.weights[:len(self.models)]]

        print(f"  Ensemble ready: {len(self.models)} models")

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Run ensemble prediction on a single image tensor.

        Args:
            image_tensor: preprocessed tensor [C, H, W] or [1, C, H, W]

        Returns:
            probs: numpy array [num_classes] of averaged probabilities
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        all_probs = []
        for model, weight in zip(self.models, self.weights):
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)[0].cpu().numpy()
            all_probs.append(probs * weight)

        # Weighted average
        avg_probs = np.sum(all_probs, axis=0)
        return avg_probs

    @torch.no_grad()
    def predict_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Run ensemble prediction on a batch.

        Args:
            batch_tensor: [B, C, H, W]

        Returns:
            probs: numpy array [B, num_classes]
        """
        batch_tensor = batch_tensor.to(self.device)
        all_probs = []

        for model, weight in zip(self.models, self.weights):
            output = model(batch_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()
            all_probs.append(probs * weight)

        avg_probs = np.sum(all_probs, axis=0)
        return avg_probs

    def __len__(self) -> int:
        return len(self.models)


def discover_checkpoints(output_dir: str = "outputs", pattern: str = "best_model*.pth") -> list[str]:
    """Find all model checkpoints matching the pattern."""
    out = Path(output_dir)
    paths = sorted(out.glob(pattern))
    return [str(p) for p in paths if p.is_file()]
