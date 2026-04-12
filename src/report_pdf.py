"""
report_pdf.py — Professional PDF clinical report generation

Generates a multi-page clinical report with:
  - Original image + Grad-CAM side-by-side
  - Binary classification result
  - Multi-class probability distribution
  - ABCDE analysis table
  - Disclaimer

Uses fpdf2 for PDF generation.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None


class SkinLesionReport(FPDF if FPDF else object):
    """Custom PDF report for skin lesion analysis."""

    def __init__(self):
        if FPDF is None:
            raise ImportError("fpdf2 is required for PDF reports. Install with: pip install fpdf2")
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(9, 17, 31)
        self.rect(0, 0, 210, 30, "F")
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(232, 242, 255)
        self.set_y(8)
        self.cell(0, 12, "Skin Lesion Analysis Report", align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 160, 180)
        self.cell(0, 10,
                  "DISCLAIMER: This report is for educational/research purposes only. "
                  "Not a substitute for professional medical diagnosis.",
                  align="C")

    def add_section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(45, 212, 191)
        self.cell(0, 10, title)
        self.ln(8)

    def add_key_value(self, key: str, value: str, indent: int = 10):
        self.set_x(indent)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(55, 7, f"{key}:")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, str(value))
        self.ln(7)


def generate_report(
    original_image: Image.Image,
    gradcam_image: Optional[np.ndarray],
    binary_result: Optional[dict],
    multiclass_result: Optional[dict],
    abcde_result: Optional[dict],
    metadata: Optional[dict] = None,
) -> str:
    """
    Generate a full PDF clinical report.

    Args:
        original_image:   PIL Image of the lesion
        gradcam_image:    numpy array of Grad-CAM overlay (or None)
        binary_result:    dict with keys: decision, confidence, malignant_prob, benign_prob, threshold
        multiclass_result: dict with keys: top_class, probabilities (dict), risk_level
        abcde_result:     dict from abcde.analyze_abcde()
        metadata:         optional dict with patient info (age, sex, etc.)

    Returns:
        path to generated PDF file
    """
    if FPDF is None:
        raise ImportError("fpdf2 is required. Install: pip install fpdf2")

    pdf = SkinLesionReport()
    pdf.add_page()

    # ── Report metadata ──────────────────────────────────────────────────
    pdf.add_section_title("Report Information")
    pdf.add_key_value("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    pdf.add_key_value("Model", "EfficientNet-B0 (Skin Lesion Classifier)")
    if metadata:
        for k, v in metadata.items():
            pdf.add_key_value(k, str(v))
    pdf.ln(5)

    # ── Images ───────────────────────────────────────────────────────────
    pdf.add_section_title("Visual Analysis")

    # Save original to temp
    tmp_orig = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_orig.close()
    original_image.resize((300, 300)).save(tmp_orig.name)

    img_width = 80
    x_start = 15
    pdf.image(tmp_orig.name, x=x_start, y=pdf.get_y(), w=img_width)

    if gradcam_image is not None:
        tmp_gc = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_gc.close()
        Image.fromarray(gradcam_image).resize((300, 300)).save(tmp_gc.name)
        pdf.image(tmp_gc.name, x=x_start + img_width + 10, y=pdf.get_y(), w=img_width)

    pdf.ln(img_width + 5)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(img_width, 5, "Original Image", align="C")
    if gradcam_image is not None:
        pdf.set_x(x_start + img_width + 10)
        pdf.cell(img_width, 5, "Grad-CAM Attention", align="C")
    pdf.ln(10)

    # ── Binary Classification ────────────────────────────────────────────
    if binary_result:
        pdf.add_section_title("Binary Classification (Benign / Malignant)")
        pdf.add_key_value("Decision", binary_result.get("decision", "N/A"))
        pdf.add_key_value("Confidence", f"{binary_result.get('confidence', 0) * 100:.1f}%")
        pdf.add_key_value("Benign Probability", f"{binary_result.get('benign_prob', 0) * 100:.1f}%")
        pdf.add_key_value("Malignant Probability", f"{binary_result.get('malignant_prob', 0) * 100:.1f}%")
        pdf.add_key_value("Screening Threshold", f"{binary_result.get('threshold', 0.5) * 100:.0f}%")

        risk_band = binary_result.get("risk_band", "N/A")
        pdf.add_key_value("Risk Band", risk_band)
        pdf.ln(5)

    # ── Multi-class Classification ───────────────────────────────────────
    if multiclass_result:
        pdf.add_section_title("Multi-Class Diagnosis (9 Categories)")
        top = multiclass_result.get("top_class", "N/A")
        risk = multiclass_result.get("risk_level", "N/A")
        pdf.add_key_value("Top Diagnosis", f"{top} ({risk})")

        probs = multiclass_result.get("probabilities", {})
        if probs:
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.set_x(10)
            pdf.cell(0, 7, "Class Probability Distribution:")
            pdf.ln(7)

            # Sort by probability descending
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            for cls_name, prob in sorted_probs:
                pdf.set_x(15)
                pdf.set_font("Helvetica", "", 9)
                pdf.set_text_color(50, 50, 50)
                label = f"{cls_name}: {prob * 100:.1f}%"
                pdf.cell(60, 6, label)

                # Draw progress bar
                bar_x = pdf.get_x()
                bar_y = pdf.get_y() + 1
                bar_w = 80
                bar_h = 4
                # Background
                pdf.set_fill_color(230, 230, 230)
                pdf.rect(bar_x, bar_y, bar_w, bar_h, "F")
                # Fill
                fill_w = max(bar_w * prob, 1)
                pdf.set_fill_color(45, 212, 191)
                pdf.rect(bar_x, bar_y, fill_w, bar_h, "F")
                pdf.ln(6)
        pdf.ln(5)

    # ── ABCDE Analysis ───────────────────────────────────────────────────
    if abcde_result:
        pdf.add_section_title("ABCDE Dermatological Analysis")

        criteria_map = {
            "A - Asymmetry": abcde_result.get("asymmetry", {}),
            "B - Border": abcde_result.get("border", {}),
            "C - Color": abcde_result.get("color", {}),
            "D - Diameter": abcde_result.get("diameter", {}),
            "E - Evolution": abcde_result.get("evolution", {}),
        }

        for name, data in criteria_map.items():
            pdf.add_key_value(name, f'{data.get("rating", "N/A")} - {data.get("detail", "")}')

        pdf.ln(3)
        overall = abcde_result.get("overall_risk", "N/A")
        pdf.add_key_value("Overall ABCDE", overall)
        pdf.ln(5)

    # ── Save PDF ─────────────────────────────────────────────────────────
    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_pdf.close()
    pdf.output(tmp_pdf.name)
    return tmp_pdf.name
