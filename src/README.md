# 🔬 Skin Lesion Classifier

A production-grade binary image classifier that detects potentially malignant skin lesions from dermoscopy images using **EfficientNet-B0** with **Grad-CAM explainability**.

Built with PyTorch + Gradio. Trained on the **ISIC 2020** dataset.

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Transfer Learning** | EfficientNet-B0 pretrained on ImageNet, fine-tuned on ISIC |
| **Two-phase training** | Feature extraction → full fine-tuning for better convergence |
| **Class imbalance handling** | Weighted sampler + label smoothing |
| **Data augmentation** | Flips, rotations, color jitter, grayscale dropout |
| **Grad-CAM** | Visual explanation of model attention |
| **ROC-AUC evaluation** | Sensitivity, specificity, F1, confusion matrix |
| **Early stopping** | Prevents overfitting automatically |
| **Gradio web UI** | Drag-and-drop interface with live Grad-CAM visualization |

---

## 🗂️ Project Structure

```
skin-lesion-classifier/
├── src/
│   ├── model.py          ← EfficientNet-B0 model definition
│   ├── dataset.py        ← Dataset, augmentation, weighted sampler
│   ├── train.py          ← Full training loop
│   ├── evaluate.py       ← Metrics, ROC curve, confusion matrix
│   ├── gradcam.py        ← Grad-CAM explainability
│   ├── predict.py        ← Single-image prediction CLI
│   └── app.py            ← Gradio web interface
├── data/                 ← Dataset lives here (created by prepare_data.py)
│   ├── images/           ← Downloaded .jpg files
│   ├── train.csv
│   └── val.csv
├── outputs/              ← Trained model + plots saved here
│   ├── best_model.pth
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   └── training_history.png
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & prepare data
```bash
# Downloads 5000 images (~2GB) from ISIC 2020
python src/prepare_data.py --n-images 5000

# For a quick test run (small download):
python src/prepare_data.py --n-images 500
```

### 3. Train the model
```bash
python src/train.py

# Quick debug run (small data limit):
python src/train.py --limit 200 --epochs 3
```

Expected output (with 5k images, ~20 epochs):
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
