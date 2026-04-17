# 🔬 Skin Lesion Classifier

A production-grade binary image classifier that detects potentially malignant skin lesions from dermoscopy images using **EfficientNet-B0** with **Grad-CAM explainability**.

Built with PyTorch + Gradio. Trained on the **ISIC 2020** dataset.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Multi-Class Architecture** | 7-class diagnostic categorization built on EfficientNet-B0 (Task 3) |
| **Attribute Segmentation** | Multi-channel Deep Learning U-Net extracting 5 dermoscopic morphological attributes (Task 2) |
| **Test-Time Augmentation** | Toggleable TTA (flips, rotations, jitter) to boost prediction accuracy live |
| **PDF Reporting Engine** | Professional fpdf2 PDF export containing Grad-CAM maps, probabilities, and patient metadata |
| **Transfer Learning** | EfficientNet-B0 pretrained on ImageNet, fine-tuned on ISIC |
| **Class imbalance handling** | Focal Loss, CutMix/Mixup augmentation, and Weighted Random Sampling |
| **Grad-CAM** | Visual explanation of model attention overlay |
| **Gradio Web Console** | Fully segmented UI for Binary, Multi-Class, ABCDE rules, and Symptom Review |

---

## 🏗️ Architecture Upgrades (v2.0)

This system has been upgraded from a simple binary classifier into a comprehensive Multi-Modal Diagnostic toolkit. The following core features have been natively integrated without breaking the baseline classifier:

### 1. 7-Class Multi-Class Architecture (Task 3 Disease Classification)
We integrated a fine-tuned EfficientNet-B0 directly predicting 7 precise dermatological categories: Melanoma, Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, and Vascular lesions. It outputs exact probability distributions alongside confidence metrics.

### 2. Clinical Dermoscopic Attribute Segmentation (Task 2)
Moving beyond rule-based computer vision (formerly ABCDE), the toolkit now utilizes a lightweight 5-channel Deep Learning **Tiny U-Net**. This extracts and explicitly maps clinically significant spatial features from the lesion:
- **Globules**
- **Milia-like cysts**
- **Negative networks**
- **Pigment networks**
- **Streaks**

### 3. Inference Amplification (TTA & Ensembling)
To squeeze maximum accuracy out of deployed models, we introduced the `tta.py` module. **Test-Time Augmentation (TTA)** flips and jitters the input image at inference time, running the model multiple times and averaging the probabilities for a noticeably robust prediction.

### 4. PDF Reporting Engine (`report_pdf.py`)
To mimic clinical workflows, the system dynamically compiles all inference outputs into a downloadable PDF. Built on `fpdf2`, the exporter packages the patient metadata, Grad-CAM attention heatmap, Multi-Class probability bars, and the ABCDE morphological scores into a unified, printable document.

---

## 🗂️ Project Structure

```text
skin-lesion-classifier/
├── src/
│   ├── model.py                ← EfficientNet-B0 (Binary & MetadataFusionClassifier)
│   ├── dataset_multiclass.py   ← 9-Class ISIC formatting + CutMix/Mixup
│   ├── focal_loss.py           ← Focal Loss module for class imbalance
│   ├── tta.py                  ← Test-Time Augmentation algorithms
│   ├── abcde.py                ← OpenCV visual morphological algorithms
│   ├── ensemble.py             ← Multi-checkpoint prediction averager
│   ├── report_pdf.py           ← PDF clinical report generator
│   ├── train_multiclass.py     ← Multi-class specific training loop
│   ├── train.py                ← Binary baseline training loop
│   ├── prepare_data.py         ← Universal ISIC/Kaggle dataset constructor
│   └── app.py                  ← Full Gradio Multi-Tab Web Interface
├── data/                       ← Generated Datasets
│   ├── images/  
│   ├── train.csv / val.csv 
│   └── train_multiclass.csv / val_multiclass.csv 
├── outputs/                    ← Trained checkpoints & figures
│   ├── best_model.pth                  ← Binary weights
│   ├── best_model_multiclass.pth       ← 9-class weights
│   └── ...
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (Binary & Multi-Class)
Download and format your folder datasets. By passing `--multiclass`, the engine will output files ready for 9-class deep analysis. 
```bash
python src/prepare_data.py --isic-dir "path/to/isic/directory" --multiclass
```

### 3. Train the models
Train the binary classification baseline, or train the 9-class system directly using Focal Loss and Progressive Resizing:
```bash
# Train Multi-Class Pipeline
python src/train_multiclass.py --epochs 30 --lr 3e-4

# Train Binary System Baseline
python src/train.py --epochs 20
```
- Val AUC: **~0.84–0.88**
- Sensitivity: ~0.75
- Specificity: ~0.85

### 4. Evaluate
```bash
python src/evaluate.py --checkpoint outputs/best_model.pth --test-csv data/val.csv --image-dir data/images
```
Saves ROC curve, confusion matrix, and training history plots to `outputs/`.

### 5. Launch the web app
```bash
python src/app.py
# Open http://localhost:7860
```

### 6. CLI prediction on a single image
```bash
python src/predict.py --image path/to/lesion.jpg
python src/predict.py --image path/to/lesion.jpg --save gradcam_output.png
```

---

## 🧠 ML Concepts Covered

### Transfer Learning
`EfficientNet-B0` was pretrained on 1.2M ImageNet images. It already knows textures, edges, and shapes. We replace only the final classification head and fine-tune on ISIC data.

### Two-Phase Training
1. **Freeze phase** (epochs 1–2): Only the new classifier head is trained. This warms up the head without corrupting pretrained weights.
2. **Fine-tune phase** (epoch 3+): Entire network is unfrozen with a lower LR (÷10). All layers adapt to dermoscopy images.

### Class Imbalance
ISIC data is ~97% benign / ~3% malignant. We fix this two ways:
- **WeightedRandomSampler**: Oversamples malignant in each batch
- **Label smoothing**: Prevents overconfident predictions

### Grad-CAM
Computes the gradient of the malignant class score with respect to the last convolutional feature map. High-gradient regions = areas the model used most for its decision. Overlaid as a heatmap on the original image.

### Metrics That Matter in Medical ML
| Metric | Why it matters |
|---|---|
| **AUC** | Threshold-independent overall quality |
| **Sensitivity** | How often we catch real malignant cases (missing one = bad) |
| **Specificity** | How often we correctly clear benign cases |
| **F1** | Balance between precision and recall |

---

## 📊 Expected Results

After full training on 5000 images for 20 epochs:

| Metric | Expected |
|---|---|
| Val AUC | 0.84 – 0.88 |
| Sensitivity | 0.72 – 0.80 |
| Specificity | 0.82 – 0.90 |
| Val Accuracy | 0.78 – 0.85 |

Results improve significantly with more data (20k+ images → AUC ~0.90+).

---

## ⚗️ Tips for Better Results

- **More data**: Use `--n-images 20000` in `prepare_data.py`
- **GPU**: Use Google Colab (free T4) for 5–10x faster training
- **Longer training**: Increase `--epochs 30` with `--patience 8`
- **Larger model**: Change to `efficientnet_b3` in `model.py` for +2% AUC

---

## ⚕️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute medical advice and should not be used for clinical diagnosis. Always consult a qualified dermatologist for any skin concern.

---

## 📚 Dataset

**ISIC 2020 Challenge Dataset**
- Source: [isic-archive.com](https://www.isic-archive.com)
- 33,126 dermoscopy images
- Binary labels: benign (0) / malignant (1)
- Used in the SIIM-ISIC Melanoma Classification Kaggle competition
