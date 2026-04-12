"""
abcde.py — Automated ABCDE Dermatological Rule Analysis

Computes the classic ABCDE criteria used by dermatologists:
    A — Asymmetry
    B — Border irregularity
    C — Color variation
    D — Diameter (relative size)
    E — Evolution (requires multiple images; flagged as "monitor")

Uses OpenCV for segmentation and analysis.  Independent of any ML model.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional


# ── Lesion segmentation ──────────────────────────────────────────────────────

def _segment_lesion(image_np: np.ndarray) -> np.ndarray:
    """
    Segment the lesion from background using adaptive thresholding.
    Returns a binary mask (uint8, 0/255).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup: close small holes, open small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only the largest connected component (the lesion)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    # Find largest component (skip background = label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)
    lesion_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return lesion_mask


# ── A: Asymmetry ─────────────────────────────────────────────────────────────

def _compute_asymmetry(mask: np.ndarray) -> dict:
    """
    Measure asymmetry by comparing the mask to its horizontal and vertical flips.
    Returns asymmetry score (0 = perfect symmetry, 1 = completely asymmetric).
    """
    h, w = mask.shape
    if mask.sum() == 0:
        return {"score": 0.0, "rating": "N/A", "detail": "No lesion detected"}

    # Normalize mask to 0-1
    m = (mask > 0).astype(np.float32)

    # Horizontal symmetry
    flipped_h = np.flip(m, axis=1)
    diff_h = np.abs(m - flipped_h).sum() / (m.sum() + 1e-8)

    # Vertical symmetry
    flipped_v = np.flip(m, axis=0)
    diff_v = np.abs(m - flipped_v).sum() / (m.sum() + 1e-8)

    asymmetry = (diff_h + diff_v) / 2.0
    asymmetry = min(asymmetry, 1.0)

    if asymmetry < 0.25:
        rating, risk = "Low", 0
    elif asymmetry < 0.50:
        rating, risk = "Moderate", 1
    else:
        rating, risk = "High", 2

    return {
        "score": round(float(asymmetry), 3),
        "rating": rating,
        "risk": risk,
        "detail": f"Asymmetry index: {asymmetry:.1%} (H: {diff_h:.1%}, V: {diff_v:.1%})",
    }


# ── B: Border irregularity ──────────────────────────────────────────────────

def _compute_border(mask: np.ndarray) -> dict:
    """
    Measure border irregularity using the compactness ratio:
    compactness = perimeter² / (4π × area).
    A perfect circle = 1.0; higher = more irregular.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"score": 0.0, "rating": "N/A", "detail": "No contour found", "risk": 0}

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    if area < 10:
        return {"score": 0.0, "rating": "N/A", "detail": "Lesion too small", "risk": 0}

    compactness = (perimeter ** 2) / (4 * np.pi * area)
    # Normalize: circle=1.0, typical lesion 1.0-3.0, very irregular 3.0+
    irregularity = min((compactness - 1.0) / 3.0, 1.0)
    irregularity = max(irregularity, 0.0)

    if irregularity < 0.25:
        rating, risk = "Smooth", 0
    elif irregularity < 0.50:
        rating, risk = "Slightly irregular", 1
    else:
        rating, risk = "Highly irregular", 2

    return {
        "score": round(float(irregularity), 3),
        "rating": rating,
        "risk": risk,
        "detail": f"Compactness: {compactness:.2f} | Irregularity: {irregularity:.1%}",
    }


# ── C: Color variation ──────────────────────────────────────────────────────

def _compute_color(image_np: np.ndarray, mask: np.ndarray) -> dict:
    """
    Measure color variation within the lesion using K-means clustering.
    More distinct color clusters = higher concern.
    """
    # Extract pixels within the lesion mask
    lesion_pixels = image_np[mask > 0]
    if len(lesion_pixels) < 50:
        return {"score": 0.0, "rating": "N/A", "detail": "Too few lesion pixels", "risk": 0, "n_colors": 0}

    # Convert to float32 for K-means
    pixels = lesion_pixels.astype(np.float32)

    # Run K-means with different K values and find optimal
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    best_k = 1
    for k in range(2, 7):
        if len(pixels) < k * 10:
            break
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        # Check if clusters are meaningfully different (distance > threshold)
        min_dist = float("inf")
        for i in range(k):
            for j in range(i + 1, k):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = min(min_dist, dist)
        if min_dist > 30:  # Colors are meaningfully different
            best_k = k
        else:
            break

    n_colors = best_k
    # Score: 1 color = 0, 2 = low, 3+ = moderate, 5+ = high
    if n_colors <= 2:
        score, rating, risk = 0.2, "Uniform", 0
    elif n_colors <= 3:
        score, rating, risk = 0.4, "Moderate variation", 1
    else:
        score, rating, risk = 0.8, "High variation", 2

    return {
        "score": round(score, 3),
        "rating": rating,
        "risk": risk,
        "n_colors": n_colors,
        "detail": f"{n_colors} distinct color clusters detected",
    }


# ── D: Diameter (relative size) ─────────────────────────────────────────────

def _compute_diameter(mask: np.ndarray) -> dict:
    """
    Estimate relative lesion diameter as percentage of image diagonal.
    Without a reference scale, absolute size is unknown, but relative size matters.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"score": 0.0, "rating": "N/A", "detail": "No contour found", "risk": 0}

    contour = max(contours, key=cv2.contourArea)
    _, (w, h), _ = cv2.minAreaRect(contour)
    diameter = max(w, h)

    img_h, img_w = mask.shape
    img_diag = np.sqrt(img_h**2 + img_w**2)
    relative_size = diameter / img_diag

    if relative_size < 0.15:
        rating, risk = "Small", 0
    elif relative_size < 0.35:
        rating, risk = "Medium", 1
    else:
        rating, risk = "Large", 2

    return {
        "score": round(float(relative_size), 3),
        "rating": rating,
        "risk": risk,
        "detail": f"Relative diameter: {relative_size:.1%} of image | ~{diameter:.0f}px",
    }


# ── E: Evolution ─────────────────────────────────────────────────────────────

def _compute_evolution() -> dict:
    """
    Evolution cannot be determined from a single image.
    Returns a placeholder prompting the user to monitor.
    """
    return {
        "score": 0.0,
        "rating": "Requires monitoring",
        "risk": 0,
        "detail": "Evolution requires comparing images over time. Take regular photos and watch for changes.",
    }


# ── Full ABCDE analysis ─────────────────────────────────────────────────────

def analyze_abcde(image, return_mask: bool = False) -> dict:
    """
    Run full ABCDE analysis on a skin lesion image.

    Args:
        image: PIL Image or numpy array (RGB)
        return_mask: if True, include the segmentation mask in the result

    Returns:
        dict with keys: asymmetry, border, color, diameter, evolution,
                        overall_score, overall_risk, mask (optional)
    """
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    # Resize for consistent analysis
    target_size = 256
    image_resized = cv2.resize(image_np, (target_size, target_size))

    # Segment the lesion
    mask = _segment_lesion(image_resized)

    # Run each criterion
    asymmetry = _compute_asymmetry(mask)
    border = _compute_border(mask)
    color = _compute_color(image_resized, mask)
    diameter = _compute_diameter(mask)
    evolution = _compute_evolution()

    # Overall ABCDE score (sum of risk scores, 0-8 possible)
    total_risk = asymmetry["risk"] + border["risk"] + color["risk"] + diameter["risk"]
    if total_risk <= 2:
        overall = "Low concern"
    elif total_risk <= 4:
        overall = "Moderate concern"
    else:
        overall = "High concern - dermatology review recommended"

    result = {
        "asymmetry": asymmetry,
        "border": border,
        "color": color,
        "diameter": diameter,
        "evolution": evolution,
        "overall_score": total_risk,
        "overall_risk": overall,
    }

    if return_mask:
        # Scale mask back to original image size
        orig_h, orig_w = image_np.shape[:2]
        result["mask"] = cv2.resize(mask, (orig_w, orig_h))

    return result


def create_abcde_visualization(image, abcde_result: dict) -> np.ndarray:
    """
    Create a visual summary of the ABCDE analysis with the segmentation overlay.

    Returns a numpy RGB image.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = image.copy()

    # Re-segment at display size
    display = cv2.resize(image_np, (256, 256))
    mask = _segment_lesion(display)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#09111d")
    fig.suptitle("ABCDE Dermatological Analysis", color="#e8f2ff", fontsize=14, fontweight="bold")

    # Original
    axes[0].imshow(display)
    axes[0].set_title("Original", color="#cbd7e6", fontsize=11)
    axes[0].axis("off")

    # Segmentation overlay
    overlay = display.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 128), 2)
    # Tint lesion area
    tint = overlay.copy()
    tint[mask > 0] = (tint[mask > 0] * 0.6 + np.array([0, 100, 80]) * 0.4).astype(np.uint8)
    axes[1].imshow(tint)
    axes[1].set_title("Lesion Segmentation", color="#cbd7e6", fontsize=11)
    axes[1].axis("off")

    # Scores summary
    axes[2].set_facecolor("#0b1628")
    axes[2].axis("off")

    criteria = [
        ("A — Asymmetry", abcde_result["asymmetry"]),
        ("B — Border", abcde_result["border"]),
        ("C — Color", abcde_result["color"]),
        ("D — Diameter", abcde_result["diameter"]),
        ("E — Evolution", abcde_result["evolution"]),
    ]

    risk_colors = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
    y_start = 0.92
    for i, (name, data) in enumerate(criteria):
        color = risk_colors.get(data["risk"], "#9ca3af")
        axes[2].text(0.05, y_start - i * 0.18, name, color="#e8f2ff", fontsize=11,
                     fontweight="bold", transform=axes[2].transAxes)
        axes[2].text(0.05, y_start - i * 0.18 - 0.08, f'{data["rating"]}',
                     color=color, fontsize=10, transform=axes[2].transAxes)

    axes[2].text(0.05, 0.02, f'Overall: {abcde_result["overall_risk"]}',
                 color="#38bdf8", fontsize=11, fontweight="bold", transform=axes[2].transAxes)

    plt.tight_layout()

    # Render to numpy array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    # We must make a copy so the array is writable later if needed
    buf_rgb = np.copy(buf)
    plt.close(fig)
    return buf_rgb
