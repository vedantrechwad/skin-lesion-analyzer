"""
model.py — EfficientNet-B0 fine-tuned for skin lesion classification

Contains:
    - SkinLesionClassifier:         standard image-only classifier (binary or multi-class)
    - MetadataFusionClassifier:     dual-input model (image + patient metadata)
    - load_model / load_model_auto: checkpoint loading utilities
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


# ── Metadata Fusion Model (dual-input: image + tabular) ─────────────────────

class MetadataFusionClassifier(nn.Module):
    """
    Dual-input model: CNN (EfficientNet-B0) for images + MLP for patient metadata.

    Metadata features (expected order):
        - age (normalized 0-1)
        - sex_male (0 or 1)
        - sex_female (0 or 1)
        - location_* (one-hot encoded body locations, variable length)

    The metadata dimension is configurable via `metadata_dim`.
    """

    def __init__(
        self,
        num_classes: int = 2,
        metadata_dim: int = 10,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Image branch — same EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Extract image features (before the original classifier)
        img_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()  # Remove original head

        # Metadata branch — small MLP
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combined classifier head
        combined_dim = img_features + 32
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(image)         # [B, 1280]
        meta_feat = self.metadata_mlp(metadata)  # [B, 32]
        combined = torch.cat([img_feat, meta_feat], dim=1)  # [B, 1312]
        return self.classifier(combined)

    def get_feature_extractor(self):
        """Returns the feature extraction part (used by Grad-CAM)."""
        return self.backbone.features


# ── Auto-detect model loader ────────────────────────────────────────────────

def load_model_auto(checkpoint_path: str, device: torch.device):
    """
    Auto-detect model type from checkpoint and load accordingly.

    Returns:
        (model, model_type, num_classes, class_names)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint.get("num_classes", 2)
    class_names = checkpoint.get("class_names", ["Benign", "Malignant"])

    # Detect if it's a fusion model by checking for metadata_mlp keys
    state_keys = set(checkpoint["model_state_dict"].keys())
    is_fusion = any("metadata_mlp" in k for k in state_keys)

    if is_fusion:
        metadata_dim = checkpoint.get("metadata_dim", 10)
        model = MetadataFusionClassifier(
            num_classes=num_classes, metadata_dim=metadata_dim
        ).to(device)
        model_type = "fusion"
    else:
        model = SkinLesionClassifier(num_classes=num_classes).to(device)
        model_type = "standard"

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded {model_type} model: {checkpoint_path} "
          f"({num_classes} classes, epoch {checkpoint.get('epoch', '?')})")

    return model, model_type, num_classes, class_names

