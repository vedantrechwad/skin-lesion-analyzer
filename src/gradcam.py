"""
gradcam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)
Visualizes WHERE in the image the model is paying attention.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple

from dataset import get_transforms, denormalize, IMAGE_SIZE


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0.

    Computes gradient of the class score with respect to the last
    convolutional feature map, then creates a heatmap highlighting
    the most influential image regions.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()

        # Hook into the last conv block of EfficientNet
        self.target_layer = target_layer or model.backbone.features[-1]

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for a single image tensor.

        Returns:
            heatmap     — numpy array [H, W] in [0, 1]
            pred_class  — predicted class index
            confidence  — predicted probability for the predicted class
        """
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
        image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)
        probs  = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        # Use predicted class if no target specified
        if target_class is None:
            target_class = pred_class

        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Pool gradients across spatial dimensions [C]
        pooled_grads = self.gradients.mean(dim=[0, 2, 3])

        # Weight activations by pooled gradients
        activations = self.activations[0]           # [C, H, W]
        for i, w in enumerate(pooled_grads):
            activations[i] *= w

        # Create heatmap
        heatmap = activations.mean(dim=0).numpy()   # [H, W]
        heatmap = np.maximum(heatmap, 0)            # ReLU
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap, pred_class, confidence


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image   — original image as numpy uint8 [H, W, 3] (RGB)
        heatmap — float array [H, W] in [0, 1]
        alpha   — heatmap opacity

    Returns:
        blended — numpy uint8 [H, W, 3] (RGB)
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap (OpenCV uses BGR → convert to RGB)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended


def explain_prediction(
    model,
    image_path: str,
    device: torch.device,
    save_path: Optional[str] = None,
) -> dict:
    """
    Full explanation pipeline: load image → predict → visualize Grad-CAM.

    Returns a dict with prediction info.
    """
    from dataset import CLASS_NAMES

    # Load and preprocess image
    pil_image = Image.open(image_path).convert("RGB")
    pil_image_resized = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    original_np = np.array(pil_image_resized)

    transform     = get_transforms("val")
    image_tensor  = transform(pil_image_resized).to(device)

    # Grad-CAM
    gradcam = GradCAM(model)
    heatmap, pred_class, confidence = gradcam.generate(image_tensor)

    # Overlay
    blended = overlay_heatmap(original_np, heatmap)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0f0f0f")
    fig.suptitle(
        f"Prediction: {CLASS_NAMES[pred_class]}  |  Confidence: {confidence:.1%}",
        color="#fff", fontsize=15, fontweight="bold",
    )

    titles  = ["Original Image", "Grad-CAM Heatmap", "Overlaid Attention"]
    images  = [original_np, cm.jet(heatmap)[..., :3], blended]

    for ax, title, img in zip(axes, titles, images):
        ax.set_facecolor("#0f0f0f")
        ax.imshow(img if img.dtype != np.float64 else (img * 255).astype(np.uint8))
        ax.set_title(title, color="#ccc", fontsize=11)
        ax.axis("off")

    # Color bar for heatmap
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046)
    cbar.set_label("Attention", color="#ccc")
    cbar.ax.yaxis.set_tick_params(color="#ccc")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccc")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"🔍 Explanation saved → {save_path}")
    plt.show()
    plt.close()

    return {
        "prediction":  CLASS_NAMES[pred_class],
        "confidence":  confidence,
        "class_index": pred_class,
    }


if __name__ == "__main__":
    import argparse
    from model import load_model

    parser = argparse.ArgumentParser(description="Explain a prediction with Grad-CAM")
    parser.add_argument("--image",      required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth")
    parser.add_argument("--save",       default=None,  help="Save explanation image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    result = explain_prediction(model, args.image, device, save_path=args.save)
    print(f"\n🔍 Prediction: {result['prediction']}  (confidence: {result['confidence']:.1%})")
