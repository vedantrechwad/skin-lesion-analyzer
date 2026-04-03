"""
model.py — EfficientNet-B0 fine-tuned for skin lesion binary classification
"""

import torch
import torch.nn as nn
from torchvision import models


class SkinLesionClassifier(nn.Module):
    """
    EfficientNet-B0 with a custom classification head.
    Pretrained on ImageNet, fine-tuned for benign vs malignant classification.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Optionally freeze backbone for feature extraction only
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_feature_extractor(self):
        """Returns the feature extraction part (used by Grad-CAM)."""
        return self.backbone.features


def load_model(checkpoint_path: str, device: torch.device) -> SkinLesionClassifier:
    """Load a trained model from a checkpoint."""
    model = SkinLesionClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def save_model(model: SkinLesionClassifier, optimizer, epoch: int, path: str, extra: dict = None):
    """Save model checkpoint."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"Checkpoint saved -> {path}")
