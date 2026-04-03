"""
predict.py — Single-image prediction from command line

Usage:
    python src/predict.py --image path/to/lesion.jpg
    python src/predict.py --image path/to/lesion.jpg --gradcam --save output.png
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from model import load_model
from dataset import get_transforms, CLASS_NAMES, IMAGE_SIZE
from gradcam import GradCAM, overlay_heatmap, explain_prediction


def predict_single(image_path: str, checkpoint: str, device: torch.device, show_gradcam: bool = True, save: str = None):
    """Run prediction on a single image."""
    model = load_model(checkpoint, device)

    if show_gradcam:
        result = explain_prediction(model, image_path, device, save_path=save)
    else:
        # Simple prediction without visualization
        transform = get_transforms("val")
        img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        pred_class  = probs.argmax()
        confidence  = probs[pred_class]
        result = {
            "prediction": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "class_index": pred_class,
        }

    print("\n--- Prediction ---")
    print(f"  Image:       {image_path}")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Confidence:  {result['confidence']:.1%}")
    print("------------------")

    if result["class_index"] == 1:
        print("Warning: potentially malignant features detected; consult a dermatologist.")
    else:
        print("Benign features detected; continue regular skin monitoring.")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict on a single image")
    parser.add_argument("--image",      required=True,                   help="Path to input image")
    parser.add_argument("--checkpoint", default="outputs/best_model.pth", help="Model checkpoint")
    parser.add_argument("--no-gradcam", action="store_true",             help="Skip Grad-CAM visualization")
    parser.add_argument("--save",       default=None,                    help="Save visualization to path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_single(
        image_path=args.image,
        checkpoint=args.checkpoint,
        device=device,
        show_gradcam=not args.no_gradcam,
        save=args.save,
    )
