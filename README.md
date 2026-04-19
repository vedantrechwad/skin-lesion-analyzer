# 🔬 Skin Lesion Analyzer (Multi-Task Diagnostic Toolkit)

A professional-grade dermatological diagnostic platform that integrates sequential deep learning models for lesion segmentation, attribute presence analysis, and disease classification with **logit-level calibration** and **Grad-CAM explainability**.

Built with PyTorch + Gradio. Trained on the **ISIC 2020** and **HAM10000** datasets.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Multi-Stage Inference** | Sequential pipeline: Lesion Segmentation (Task 1) → Attribute Analysis (Task 2) → Disease Classification (Task 3) |
| **Disease Classification** | 7-class diagnostic categorization (Melanoma, BCC, NV, etc.) with logit-level calibration to reduce class bias |
| **Attribute Presence** | Multi-label EfficientNet-B0 classifier detecting 5 key morphological structures (Globules, Milia, Networks, etc.) |
| **Developer Diagnostics** | Real-time logit/probability audit panel for mathematical transparency and calibration validation |
| **Evidence Tiering** | Confidence-based reporting (High, Moderate, Low Evidence) for morphological findings |
| **PDF Reporting Engine** | Professional clinical PDF export containing attention maps, probability distributions, and ABCDE metrics |
| **Test-Time Augmentation** | Toggleable TTA (flips, rotations, jitter) to boost prediction robustness live |
| **Grad-CAM** | Visual explanation of model attention for the malignant screening layer |

---

## 🏗️ Architecture Upgrades (v3.0)

The system has matured into a multi-task diagnostic suite. The inference engine now coordinates three specialized neural networks:

### 1. Lesion Segmentation (Task 1)
Utilizes a Deep Learning **U-Net** to isolate the lesion boundary. This mask drives the automated **ABCDE criteria analysis**, calculating asymmetry, border irregularity, and diameter metrics via computer vision.

### 2. Dermoscopic Attribute Presence (Task 2)
Replaced legacy segmentation with a high-fidelity **EfficientNet-B0 Multi-Label Classifier**. It detects the presence of clinically significant structures with tiered evidence labels:
- **High Evidence** (≥65%) | **Moderate Evidence** (35-64%) | **Low / Not Detected** (<35%)
- Attributes: Globules, Milia-like cysts, Negative network, Pigment network, Streaks.

### 3. Calibrated Disease Classification (Task 3)
A fine-tuned EfficientNet-B0 predicting 7 precise diagnostic categories.
- **Logit Calibration**: Implements an inference-time `NV_PENALTY` to counteract the common "Melanocytic Nevus" bias.
- **Uncertainty Logic**: Automatically flags predictions as "Uncertain" if the top-2 probability margin is <15%.

---

## 🗂️ Project Structure

```text
skin-lesion-analyzer/
├── src/
│   ├── model.py                ← EfficientNet-B0 (Binary, Multi-Class, & Attribute Classifier)
│   ├── dataset_multiclass.py   ← ISIC/HAM10000 formatting + Multi-label logic
│   ├── tta.py                  ← Test-Time Augmentation algorithms
│   ├── abcde.py                ← OpenCV visual morphological algorithms
│   ├── report_pdf.py           ← PDF clinical report generator
│   └── app.py                  ← Gradio Web Interface (Single Analysis / Segmentation / Classification)
├── data/                       ← Generated Datasets and manifests
├── outputs/                    ← Trained checkpoints & technical figures
├── task3_best_classifier.pth   ← 7-class weight file
├── task2_attribute_classifier_best.pth ← Attribute presence weight file
├── best_unet_model.pth         ← Task 1 segmentation weights
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Model Requirements
Ensure the following weights are in the project root:
- `task3_best_classifier.pth`
- `task2_attribute_classifier_best.pth`
- `best_unet_model.pth`

### 3. Launch the web app
```bash
python src/app.py
# Open http://localhost:7860
```

---

## 🧠 Diagnostic Transparency

### Developer Debug Panel
The **Disease Classification** tab contains an "Advanced Developer Diagnostics" panel. This allows clinicians and researchers to audit:
- **Raw Logits**: The direct numerical output from the neural network before calibration.
- **Calibrated Logits**: The adjusted values after the `NV` penalty is applied.
- **Probability Calibration**: The final Softmax distribution used for the diagnostic decision.

### Automated Clinical Notes
- **"Prediction highly confident"**: Triggered if the primary class exceeds 95% probability.
- **"Prediction uncertain"**: Triggered if the model cannot clearly distinguish between the top two categories (margin < 15%).

---

## ⚕️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute medical advice and should not be used for clinical diagnosis. High-risk predictions should always be reviewed by a board-certified dermatologist.

---

## 📚 Datasets

- **ISIC Archive**: Comprehensive dermoscopy images for Task 1 and Task 3.
- **HAM10000**: Used for fine-tuning multi-class diagnostic categories.
