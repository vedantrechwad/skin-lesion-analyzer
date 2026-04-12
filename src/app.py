"""
app.py - Gradio web interface for skin lesion classification
Run: python app.py
"""

import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset import CLASS_NAMES, IMAGE_SIZE, get_transforms
from gradcam import GradCAM, overlay_heatmap
from model import SkinLesionClassifier, load_model, load_model_auto
from dataset_multiclass import MULTICLASS_NAMES, RISK_LEVELS, RISK_COLORS
from tta import predict_with_tta
from abcde import analyze_abcde, create_abcde_visualization
from report_pdf import generate_report

matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = OUTPUTS_DIR / "best_model.pth"
MULTICLASS_CHECKPOINT = OUTPUTS_DIR / "best_model_multiclass.pth"
DEFAULT_THRESHOLD = 0.50
HISTORY_COLUMNS = [
    "Time",
    "Source",
    "Decision",
    "Risk Band",
    "Confidence %",
    "Malignant %",
    "Threshold %",
]
SYMPTOM_SOURCE_URLS = {
    "AAD Melanoma Warning Signs": "https://www.aad.org/public/spot-skin-cancer/learn-about-skin-cancer/detect/how-to-spot-skin-cancer/how-to-spot-skin-cancer",
    "American Cancer Society Melanoma Signs": "https://www.cancer.org/cancer/types/melanoma-skin-cancer/detection-diagnosis-staging/signs-and-symptoms.html",
}
SKIN_CARE_SOURCES = {
    "American Academy of Dermatology": "https://www.aad.org/public/everyday-care/skin-care-basics",
    "CDC Sun Safety": "https://www.cdc.gov/skin-cancer/sunsafeselfie/index.html",
    "MedlinePlus Skin Care": "https://medlineplus.gov/skinconditions.html",
}
PLOT_FILES = {
    "Confusion Matrix": OUTPUTS_DIR / "confusion_matrix.png",
    "ROC Curve": OUTPUTS_DIR / "roc_curve.png",
    "Training History": OUTPUTS_DIR / "training_history.png",
}

model = None
multiclass_model = None

def get_multiclass_model():
    global multiclass_model
    if multiclass_model is None:
        if MULTICLASS_CHECKPOINT.exists():
            multiclass_model, _, _, _ = load_model_auto(str(MULTICLASS_CHECKPOINT), DEVICE)
            multiclass_model.eval()
    return multiclass_model


def get_model():
    global model
    if model is None:
        if CHECKPOINT.exists():
            model = load_model(str(CHECKPOINT), DEVICE)
        else:
            print("No checkpoint found. Running in demo mode with untrained weights.")
            model = SkinLesionClassifier().to(DEVICE)
            model.eval()
    return model


def build_placeholder_html() -> str:
    return """
    <div class="result-shell">
        <div class="result-head">
            <div>
                <p class="eyebrow">Awaiting Analysis</p>
                <h2>Upload a lesion image to begin</h2>
            </div>
            <span class="status-pill neutral">Idle</span>
        </div>
        <p class="result-copy">
            The app will generate a classification result, confidence distribution,
            Grad-CAM attention map, and exportable report for the uploaded image.
        </p>
        <div class="metric-grid">
            <div class="metric-card">
                <span>Prediction</span>
                <strong>Not analyzed yet</strong>
            </div>
            <div class="metric-card">
                <span>Confidence</span>
                <strong>--</strong>
            </div>
            <div class="metric-card">
                <span>Resolution</span>
                <strong>224 x 224</strong>
            </div>
        </div>
    </div>
    """


def empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def discover_example_images(limit: int = 6) -> list[list[str]]:
    val_csv = DATA_DIR / "val.csv"
    image_dir = IMAGE_DIR
    if not val_csv.exists() or not image_dir.exists():
        return []

    try:
        df = pd.read_csv(val_csv)
    except Exception:
        return []

    examples = []
    for stem in df["image_name"].astype(str).tolist():
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
            path = image_dir / f"{stem}{ext}"
            if path.exists():
                examples.append([str(path)])
                break
        if len(examples) >= limit:
            break
    return examples


def resolve_risk_band(malignant_prob: float) -> str:
    if malignant_prob < 0.25:
        return "Low"
    if malignant_prob < 0.5:
        return "Guarded"
    if malignant_prob < 0.75:
        return "Elevated"
    return "High"


def source_link_list(source_map: dict[str, str]) -> str:
    links = [f"<li><a href='{url}' target='_blank'>{name}</a></li>" for name, url in source_map.items()]
    return "<ul class='source-list'>" + "".join(links) + "</ul>"


def compute_symptom_review(
    evolving: bool,
    asymmetry: bool,
    irregular_border: bool,
    multiple_colors: bool,
    diameter_large: bool,
    itching_or_tender: bool,
    bleeding_or_oozing: bool,
    personal_or_family_history: bool,
) -> dict:
    score = 0
    reasons = []

    # Evidence-informed weights inferred from official warning signs pages.
    if evolving:
        score += 3
        reasons.append("recent change/evolution reported")
    if bleeding_or_oozing:
        score += 3
        reasons.append("bleeding or oozing reported")
    if asymmetry:
        score += 2
        reasons.append("asymmetry reported")
    if irregular_border:
        score += 2
        reasons.append("irregular border reported")
    if multiple_colors:
        score += 2
        reasons.append("multiple colors reported")
    if diameter_large:
        score += 1
        reasons.append("diameter over about 6 mm reported")
    if itching_or_tender:
        score += 1
        reasons.append("itching or tenderness reported")
    if personal_or_family_history:
        score += 1
        reasons.append("personal or family history reported")

    if score >= 8:
        level = "High symptom concern"
    elif score >= 4:
        level = "Moderate symptom concern"
    else:
        level = "Lower symptom concern"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
    }


def build_symptom_html(symptom_review: dict, combined_guidance: str, image_context: str) -> str:
    reasons = symptom_review["reasons"] or ["no major official warning-sign boxes were selected"]
    reason_items = "".join(f"<li>{reason}</li>" for reason in reasons)
    return f"""
    <div class="result-shell">
        <div class="result-head">
            <div>
                <p class="eyebrow">Symptom Review</p>
                <h2>{symptom_review['level']}</h2>
            </div>
            <span class="status-pill {'warn' if symptom_review['score'] >= 4 else 'safe'}">Score {symptom_review['score']}</span>
        </div>
        <p class="result-copy">
            This is an evidence-informed triage layer derived from official warning-sign guidance.
            It does not retrain or modify the image model.
        </p>
        <div class="metric-grid">
            <div class="metric-card">
                <span>Symptom Concern</span>
                <strong>{symptom_review['level']}</strong>
            </div>
            <div class="metric-card">
                <span>Image Context</span>
                <strong>{image_context}</strong>
            </div>
            <div class="metric-card">
                <span>Combined Guidance</span>
                <strong>{combined_guidance}</strong>
            </div>
        </div>
        <div class="advice-box">
            <h3>Signals selected</h3>
            <ul class="reason-list">{reason_items}</ul>
        </div>
        <p class="medical-note">
            Source basis: official AAD and American Cancer Society warning-sign pages listed below.
            Weighting is an app-side inference from those signs, not an official scoring tool.
        </p>
    </div>
    """


def combine_guidance(symptom_level: str, malignant_prob: float | None, threshold: float) -> tuple[str, str]:
    if malignant_prob is None:
        if symptom_level == "High symptom concern":
            return "Dermatology review recommended", "No image result yet; symptom concern alone is enough to justify prompt professional review."
        if symptom_level == "Moderate symptom concern":
            return "Monitor closely and consider review", "Symptoms include recognized warning signs. If possible, add an image analysis and seek review if changes continue."
        return "Low immediate concern", "No strong symptom cluster was selected, but any changing lesion should still be monitored."

    if malignant_prob >= threshold or symptom_level == "High symptom concern":
        return "Dermatology review recommended", "Either the image analysis crossed the alert threshold or the symptom review is high concern."
    if malignant_prob >= max(0.35, threshold - 0.15) or symptom_level == "Moderate symptom concern":
        return "Monitor closely and consider review", "There is some concern from the image result or the symptom review, so short-interval monitoring or professional review is reasonable."
    return "Low immediate concern", "Both the current image signal and the symptom review are relatively low, though ongoing monitoring is still important."


def display_plot(path: Path):
    return str(path) if path.exists() else None


def build_skin_care_fun_html() -> str:
    return f"""
    <div class="guide-list">
        <div class="guide-item">
            <strong>Sun-smart is skin-smart</strong>
            <span>Shade, protective clothing, sunglasses, and sunscreen are the unglamorous superheroes of healthy skin. The CDC specifically recommends sun protection habits when outdoors.</span>
        </div>
        <div class="guide-item">
            <strong>Be gentle, not dramatic</strong>
            <span>Choose gentle cleansing and avoid over-scrubbing. Healthy skin routines usually win by consistency, not intensity.</span>
        </div>
        <div class="guide-item">
            <strong>Moisturize like you mean it</strong>
            <span>Simple moisturizing helps support the skin barrier, especially after washing and when skin feels dry or irritated.</span>
        </div>
        <div class="guide-item">
            <strong>Watch for change, not perfection</strong>
            <span>Skin does not need to look flawless to be healthy. What matters is noticing new, evolving, bleeding, or unusual lesions and getting them checked.</span>
        </div>
        <div class="guide-item">
            <strong>Do regular self-checks</strong>
            <span>Making skin checks a monthly habit is more powerful than trying to remember every tiny detail from memory.</span>
        </div>
    </div>
    <div class="info-grid">
        <div class="info-card">
            <span>Advice Sources</span>
            <strong>AAD, CDC, MedlinePlus</strong>
        </div>
        <div class="info-card">
            <span>Style</span>
            <strong>Fun wording, conservative guidance</strong>
        </div>
    </div>
    <div class="advice-box" style="margin-top:18px;">
        <h3>Trusted source links</h3>
        {source_link_list(SKIN_CARE_SOURCES)}
    </div>
    """


def build_result_html(
    decision: str,
    risk_band: str,
    confidence_pct: float,
    benign_prob: float,
    malignant_prob: float,
    threshold_pct: float,
    advice: str,
) -> str:
    risk_class = "safe" if decision == "Benign" else "warn"
    subtitle = "Low Concern" if decision == "Benign" else "Review Recommended"

    return f"""
    <div class="result-shell">
        <div class="result-head">
            <div>
                <p class="eyebrow">AI Screening Result</p>
                <h2>{decision}</h2>
            </div>
            <span class="status-pill {risk_class}">{subtitle}</span>
        </div>

        <p class="result-copy">
            This decision uses a malignant screening threshold of <strong>{threshold_pct:.0f}%</strong>.
            Current risk band is <strong>{risk_band}</strong> with decision confidence
            <strong>{confidence_pct:.1f}%</strong>.
        </p>

        <div class="metric-grid">
            <div class="metric-card">
                <span>Decision</span>
                <strong>{decision}</strong>
            </div>
            <div class="metric-card">
                <span>Risk Band</span>
                <strong>{risk_band}</strong>
            </div>
            <div class="metric-card">
                <span>Threshold</span>
                <strong>{threshold_pct:.0f}% malignant</strong>
            </div>
        </div>

        <div class="probability-section">
            <div class="probability-row">
                <div class="probability-labels">
                    <span>Benign probability</span>
                    <strong>{benign_prob * 100:.1f}%</strong>
                </div>
                <div class="probability-track">
                    <div class="probability-fill probability-benign" style="width: {benign_prob * 100:.1f}%"></div>
                </div>
            </div>
            <div class="probability-row">
                <div class="probability-labels">
                    <span>Malignant probability</span>
                    <strong>{malignant_prob * 100:.1f}%</strong>
                </div>
                <div class="probability-track">
                    <div class="probability-fill probability-malignant" style="width: {malignant_prob * 100:.1f}%"></div>
                </div>
            </div>
        </div>

        <div class="advice-box">
            <h3>What this means</h3>
            <p>{advice}</p>
        </div>

        <p class="medical-note">
            Educational and research use only. Always consult a qualified dermatologist for diagnosis.
        </p>
    </div>
    """


def create_visualization_figure(original_np: np.ndarray, blended: np.ndarray) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#09111d")
    fig.suptitle("Grad-CAM Attention Map", color="#e8f2ff", fontsize=13, fontweight="bold")

    axes[0].imshow(original_np)
    axes[0].set_title("Original", color="#cbd7e6", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("Attention Heatmap", color="#cbd7e6", fontsize=11)
    axes[1].axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.02)
    cbar.set_label("Attention", color="#cbd7e6", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cbd7e6")

    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#09111d")
    plt.close(fig)
    return tmp.name


def create_report_card(original_np: np.ndarray, blended: np.ndarray, record: dict) -> str:
    fig = plt.figure(figsize=(12, 7), facecolor="#07111f")
    gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.9], hspace=0.18, wspace=0.08)

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])

    for ax in (ax_orig, ax_heat, ax_text):
        ax.set_facecolor("#0b1628")

    ax_orig.imshow(original_np)
    ax_orig.set_title("Original Image", color="#e6eef8", fontsize=12, fontweight="bold")
    ax_orig.axis("off")

    ax_heat.imshow(blended)
    ax_heat.set_title("Grad-CAM Overlay", color="#e6eef8", fontsize=12, fontweight="bold")
    ax_heat.axis("off")

    ax_text.axis("off")
    report_lines = [
        "Skin Lesion Analysis Report",
        f"Time: {record['Time']}",
        f"Source: {record['Source']}",
        f"Decision: {record['Decision']}",
        f"Risk Band: {record['Risk Band']}",
        f"Confidence: {record['Confidence %']}",
        f"Malignant Probability: {record['Malignant %']}",
        f"Screening Threshold: {record['Threshold %']}",
        "",
        "Disclaimer: This report is for educational and research use only.",
        "It is not a substitute for professional medical diagnosis.",
    ]
    ax_text.text(
        0.02,
        0.96,
        "\n".join(report_lines),
        va="top",
        ha="left",
        color="#dbe7f5",
        fontsize=12,
        linespacing=1.5,
        family="sans-serif",
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    plt.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#07111f")
    plt.close(fig)
    return tmp.name


def history_to_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return empty_history_df()
    return pd.DataFrame(records, columns=HISTORY_COLUMNS)


def write_records_csv(records: list[dict]) -> str | None:
    if not records:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8")
    tmp.close()
    history_to_dataframe(records).to_csv(tmp.name, index=False)
    return tmp.name


def load_pil_image(file_obj) -> Image.Image:
    path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    return Image.open(path).convert("RGB")


def analyze_image(image: Image.Image, threshold: float, source_name: str, use_tta: bool = False, run_abcde: bool = False) -> dict:
    model = get_model()
    transform = get_transforms("val")

    pil_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    tensor = transform(pil_resized).to(DEVICE)

    # ── Grad-CAM ──
    gradcam = GradCAM(model)
    heatmap, _, _ = gradcam.generate(tensor)

    original_np = np.array(pil_resized)
    blended = overlay_heatmap(original_np, heatmap, alpha=0.45)
    visual_path = create_visualization_figure(original_np, blended)

    # ── Binary Inference ──
    if use_tta:
        probs = predict_with_tta(model, pil_resized, DEVICE)
    else:
        with torch.no_grad():
            outputs = model(tensor.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    benign_prob = float(probs[0])
    malignant_prob = float(probs[1])
    decision_idx = 1 if malignant_prob >= threshold else 0
    decision = CLASS_NAMES[decision_idx]
    confidence = malignant_prob if decision_idx == 1 else benign_prob
    risk_band = resolve_risk_band(malignant_prob)

    binary_result = {
        "decision": decision, 
        "confidence": confidence, 
        "benign_prob": benign_prob,
        "malignant_prob": malignant_prob, 
        "threshold": threshold, 
        "risk_band": risk_band
    }

    # ── Multi-class Inference ──
    mc_model = get_multiclass_model()
    multiclass_result = None
    mc_visual_path = None
    if mc_model is not None:
        if use_tta:
            mc_probs = predict_with_tta(mc_model, pil_resized, DEVICE)
        else:
            with torch.no_grad():
                outputs = mc_model(tensor.unsqueeze(0))
                mc_probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
        top_idx = int(np.argmax(mc_probs))
        top_class = MULTICLASS_NAMES[top_idx]
        risk_level = RISK_LEVELS[top_class]
        
        # Build multi-class output dict
        mc_prob_dict = {name: float(prob) for name, prob in zip(MULTICLASS_NAMES, mc_probs)}
        multiclass_result = {
            "top_class": top_class,
            "risk_level": risk_level,
            "probabilities": mc_prob_dict,
        }
        
        # Build bar chart for multi-class
        mc_visual_path = create_multiclass_figure(mc_prob_dict, top_class, risk_level)

    # ── ABCDE Analysis ──
    abcde_res = None
    abcde_visual_path = None
    if run_abcde:
        abcde_res = analyze_abcde(pil_resized)
        abcde_visual_path = create_abcde_visualization_wrapper(pil_resized, abcde_res)

    if decision_idx == 0:
        advice = (
            "Lower-risk indicators were detected in this image. Keep monitoring for visible changes "
            "and seek dermatology review for any persistent concern."
        )
    else:
        advice = (
            "This screening crossed the malignant alert threshold. A dermatologist should review the "
            "lesion promptly, especially if it is changing, bleeding, or itching."
        )

    record = {
        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source_name,
        "Decision": decision,
        "Risk Band": risk_band,
        "Confidence %": f"{confidence * 100:.1f}%",
        "Malignant %": f"{malignant_prob * 100:.1f}%",
        "Threshold %": f"{threshold * 100:.0f}%",
    }
    
    # Generate integrated PDF report
    report_path = generate_report(
        original_image=image,
        gradcam_image=blended,
        binary_result=binary_result,
        multiclass_result=multiclass_result,
        abcde_result=abcde_res,
        metadata=None,
    )
    
    result_html = build_result_html(
        decision=decision,
        risk_band=risk_band,
        confidence_pct=confidence * 100,
        benign_prob=benign_prob,
        malignant_prob=malignant_prob,
        threshold_pct=threshold * 100,
        advice=advice,
    )

    return {
        "visual_path": visual_path,
        "report_path": report_path,
        "result_html": result_html,
        "record": record,
        "malignant_prob": malignant_prob,
        "benign_prob": benign_prob,
        "decision": decision,
        "threshold": threshold,
        "mc_visual_path": mc_visual_path,
        "abcde_visual_path": abcde_visual_path,
        "multiclass_result": multiclass_result,
        "abcde_result": abcde_res,
    }



def create_multiclass_figure(probs: dict, top_class: str, risk_level: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#09111d")
    ax.set_facecolor="#09111d"
    
    names = list(probs.keys())
    values = [probs[n] * 100 for n in names]
    
    # Sort for bar chart
    sorted_pairs = sorted(zip(values, names), reverse=False)
    values_s = [v for v, n in sorted_pairs]
    names_s = [n for v, n in sorted_pairs]
    
    colors = [RISK_COLORS.get(RISK_LEVELS[n], "#38bdf8") for n in names_s]
    bars = ax.barh(names_s, values_s, color=colors, alpha=0.8)
    
    for bar in bars:
        width = bar.get_width()
        if width > 1:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                    va='center', color="#cbd7e6", fontsize=9)
    
    ax.set_title(f"Multi-Class Probabilities\nTop: {top_class} ({risk_level})", color="#e8f2ff", fontsize=12, fontweight="bold")
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(colors='#94a3b8')
    ax.set_xlabel("Probability (%)", color="#94a3b8")
    
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#09111d")
    plt.close(fig)
    return tmp.name

def create_abcde_visualization_wrapper(pil_resized, abcde_res):
    import numpy as np
    from abcde import create_abcde_visualization
    vis_np = create_abcde_visualization(pil_resized, abcde_res)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#09111d")
    ax.imshow(vis_np)
    ax.axis('off')
    plt.tight_layout()
    
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    plt.savefig(tmp.name, dpi=130, bbox_inches="tight", facecolor="#09111d")
    plt.close(fig)
    return tmp.name

def run_single_analysis(image: Image.Image, threshold: float, use_tta: bool, run_abcde: bool, history_state: list[dict] | None):
    if image is None:
        return None, build_placeholder_html(), None, None, empty_history_df(), None, history_state or [], None, None, None

    history_state = list(history_state or [])
    analysis = analyze_image(image, threshold, "Single Upload", use_tta, run_abcde)
    history_state.append(analysis["record"])
    history_csv = write_records_csv(history_state)

    return (
        analysis["visual_path"],
        analysis["result_html"],
        analysis["visual_path"],
        analysis["report_path"],
        history_to_dataframe(history_state),
        history_csv,
        history_state,
        analysis,
        analysis["mc_visual_path"],
        analysis["abcde_visual_path"]
    )


def run_batch_analysis(files, threshold: float, history_state: list[dict] | None):
    history_state = list(history_state or [])
    if not files:
        return (
            "<div class='mini-summary'><strong>No files selected.</strong><span>Add one or more images to run batch analysis.</span></div>",
            empty_history_df(),
            None,
            history_to_dataframe(history_state),
            write_records_csv(history_state),
            history_state,
        )

    batch_records = []
    for file_obj in files:
        try:
            image = load_pil_image(file_obj)
            source_name = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj)).name
            analysis = analyze_image(image, threshold, source_name)
            batch_records.append(analysis["record"])
        except Exception as exc:
            source_name = Path(file_obj.name if hasattr(file_obj, "name") else str(file_obj)).name
            batch_records.append(
                {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Source": source_name,
                    "Decision": "Error",
                    "Risk Band": "Unavailable",
                    "Confidence %": "--",
                    "Malignant %": "--",
                    "Threshold %": f"{threshold * 100:.0f}%",
                }
            )
            print(f"Batch analysis failed for {source_name}: {exc}")

    history_state.extend(batch_records)
    batch_csv = write_records_csv(batch_records)
    history_csv = write_records_csv(history_state)

    decision_counts = pd.DataFrame(batch_records)["Decision"].value_counts().to_dict()
    summary_parts = [f"{key}: {value}" for key, value in decision_counts.items()]
    summary_html = (
        "<div class='mini-summary'>"
        "<strong>Batch analysis complete.</strong>"
        f"<span>{len(batch_records)} files processed | {' | '.join(summary_parts)}</span>"
        "</div>"
    )

    return (
        summary_html,
        pd.DataFrame(batch_records, columns=HISTORY_COLUMNS),
        batch_csv,
        history_to_dataframe(history_state),
        history_csv,
        history_state,
    )


def clear_history():
    return empty_history_df(), None, []


def run_symptom_review(
    latest_analysis: dict | None,
    threshold: float,
    evolving: bool,
    asymmetry: bool,
    irregular_border: bool,
    multiple_colors: bool,
    diameter_large: bool,
    itching_or_tender: bool,
    bleeding_or_oozing: bool,
    personal_or_family_history: bool,
):
    symptom_review = compute_symptom_review(
        evolving=evolving,
        asymmetry=asymmetry,
        irregular_border=irregular_border,
        multiple_colors=multiple_colors,
        diameter_large=diameter_large,
        itching_or_tender=itching_or_tender,
        bleeding_or_oozing=bleeding_or_oozing,
        personal_or_family_history=personal_or_family_history,
    )

    malignant_prob = None if not latest_analysis else latest_analysis.get("malignant_prob")
    image_context = "No image analysis yet"
    if latest_analysis:
        image_context = (
            f"{latest_analysis.get('decision', 'Unknown')} "
            f"({latest_analysis.get('malignant_prob', 0.0) * 100:.1f}% malignant)"
        )

    combined_guidance, helper_text = combine_guidance(symptom_review["level"], malignant_prob, threshold)
    html = build_symptom_html(symptom_review, combined_guidance, image_context)
    summary = (
        "<div class='mini-summary'>"
        f"<strong>{combined_guidance}</strong>"
        f"<span>{helper_text}</span>"
        "</div>"
    )
    sources_html = (
        "<div class='mini-summary'>"
        "<strong>Official warning-sign sources used</strong>"
        f"{source_link_list(SYMPTOM_SOURCE_URLS)}"
        "</div>"
    )
    return html, summary, sources_html


def build_ui():
    examples = discover_example_images()

    with gr.Blocks(
        title="Skin Lesion Classifier",
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Plus Jakarta Sans"),
        ),
        css="""
        :root {
            --bg: #07111f;
            --panel: rgba(8, 20, 36, 0.78);
            --line: rgba(148, 163, 184, 0.16);
            --text: #e6eef8;
            --muted: #9eb0c7;
            --teal: #2dd4bf;
            --cyan: #38bdf8;
            --green: #22c55e;
            --orange: #f97316;
        }
        .gradio-container {
            background:
                radial-gradient(circle at top left, rgba(45, 212, 191, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(56, 189, 248, 0.16), transparent 25%),
                linear-gradient(135deg, #07111f 0%, #0b1628 50%, #07111f 100%);
            color: var(--text);
        }
        .block-wrap {
            max-width: 1240px !important;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 30px 32px;
            border: 1px solid var(--line);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(17, 34, 57, 0.92), rgba(7, 17, 31, 0.95)),
                radial-gradient(circle at 15% 15%, rgba(45, 212, 191, 0.25), transparent 25%);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
            margin-bottom: 20px;
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: auto -60px -60px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.28), transparent 62%);
            filter: blur(6px);
        }
        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(260px, 0.85fr);
            gap: 24px;
            align-items: center;
        }
        .hero-copy h1 {
            margin: 0 0 12px;
            font-size: 3rem;
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #f7fbff;
        }
        .hero-copy p {
            margin: 0;
            max-width: 760px;
            color: var(--muted);
            font-size: 1.03rem;
            line-height: 1.7;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 18px;
        }
        .hero-badge {
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.09);
            background: rgba(255, 255, 255, 0.05);
            color: #d8e6f6;
            font-size: 0.88rem;
            font-weight: 600;
            backdrop-filter: blur(10px);
        }
        .hero-stats {
            display: grid;
            gap: 12px;
        }
        .hero-stat {
            padding: 16px 18px;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        .hero-stat span {
            display: block;
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .hero-stat strong {
            font-size: 1.2rem;
            color: #f8fcff;
        }
        .panel {
            border: 1px solid var(--line);
            border-radius: 24px;
            background: var(--panel);
            backdrop-filter: blur(14px);
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
            min-height: 100%;
        }
        .panel-header {
            padding: 22px 24px 0;
        }
        .panel-title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #f4f8ff;
        }
        .panel-subtitle {
            margin: 6px 0 0;
            color: var(--muted);
            font-size: 0.95rem;
        }
        .panel-body {
            padding: 20px 24px 24px;
        }
        .upload-box, .output-box, .download-box, .dataframe, .file-preview {
            border: 1px solid var(--line) !important;
            background: rgba(255, 255, 255, 0.03) !important;
            border-radius: 18px !important;
            overflow: hidden;
        }
        .analyze-btn, .soft-btn {
            height: 52px;
            border: none !important;
            border-radius: 16px !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }
        .analyze-btn {
            background: linear-gradient(135deg, var(--teal), var(--cyan)) !important;
            color: #062234 !important;
            box-shadow: 0 16px 40px rgba(45, 212, 191, 0.25);
        }
        .analyze-btn:hover, .soft-btn:hover {
            transform: translateY(-1px);
        }
        .soft-btn {
            background: rgba(255, 255, 255, 0.09) !important;
            color: #e6eef8 !important;
            border: 1px solid var(--line) !important;
        }
        .result-shell {
            color: var(--text);
        }
        .result-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 14px;
        }
        .result-head h2 {
            margin: 2px 0 0;
            font-size: 1.9rem;
            line-height: 1.1;
            color: #f8fbff;
        }
        .eyebrow {
            margin: 0;
            color: var(--cyan);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 700;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            padding: 10px 14px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .status-pill.safe {
            background: rgba(34, 197, 94, 0.14);
            border: 1px solid rgba(34, 197, 94, 0.28);
            color: #9ff0b8;
        }
        .status-pill.warn {
            background: rgba(249, 115, 22, 0.14);
            border: 1px solid rgba(249, 115, 22, 0.26);
            color: #ffc48c;
        }
        .status-pill.neutral {
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #d1dbe7;
        }
        .result-copy, .guide-item span, .mini-summary span {
            color: var(--muted);
            line-height: 1.7;
        }
        .metric-grid, .info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }
        .info-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 18px;
            margin-bottom: 0;
        }
        .metric-card, .info-card, .guide-item, .mini-summary {
            padding: 16px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.04);
        }
        .metric-card span, .info-card span {
            display: block;
            font-size: 0.8rem;
            color: var(--muted);
            margin-bottom: 8px;
        }
        .metric-card strong, .info-card strong, .mini-summary strong {
            color: #f4f8fd;
        }
        .probability-section {
            display: grid;
            gap: 14px;
            margin-bottom: 18px;
        }
        .probability-labels {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            margin-bottom: 7px;
            color: #dce8f6;
            font-size: 0.95rem;
        }
        .probability-track {
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            overflow: hidden;
        }
        .probability-fill {
            height: 100%;
            border-radius: inherit;
            transition: width 0.35s ease;
        }
        .probability-benign {
            background: linear-gradient(90deg, #34d399, #22c55e);
        }
        .probability-malignant {
            background: linear-gradient(90deg, #fb923c, #f97316);
        }
        .advice-box {
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid rgba(56, 189, 248, 0.18);
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(45, 212, 191, 0.05));
        }
        .advice-box h3 {
            margin: 0 0 8px;
            font-size: 1rem;
        }
        .advice-box p, .medical-note, .footer-note {
            margin: 0;
            color: #dce6f2;
        }
        .medical-note, .footer-note {
            color: #8ea2ba;
            font-size: 0.84rem;
            margin-top: 16px;
        }
        .control-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: -4px;
            margin-bottom: 14px;
        }
        .footer-note {
            text-align: center;
            margin-top: 18px;
        }
        .source-list, .reason-list {
            margin: 8px 0 0 18px;
            color: #dce6f2;
            line-height: 1.7;
        }
        .source-list a {
            color: #89e0ff;
        }
        @media (max-width: 980px) {
            .hero-grid, .metric-grid, .info-grid {
                grid-template-columns: 1fr;
            }
            .result-head {
                flex-direction: column;
            }
            .hero-copy h1 {
                font-size: 2.35rem;
            }
        }
        """,
    ) as demo:
        history_state = gr.State([])
        latest_analysis_state = gr.State(None)

        gr.HTML(
            f"""
            <div class="hero">
                <div class="hero-grid">
                    <div class="hero-copy">
                        <div class="badge-row">
                            <span class="hero-badge">Skin Health Screening</span>
                            <span class="hero-badge">Grad-CAM Explainability</span>
                            <span class="hero-badge">Batch Ready</span>
                        </div>
                        <h1>AI-Powered Skin Lesion Analysis</h1>
                        <p>
                            Upload a dermoscopy image to generate a benign-versus-malignant screening
                            result, threshold-aware decision, visual attention heatmap, session history,
                            and exportable reports in one interface.
                        </p>
                    </div>
                    <div class="hero-stats">
                        <div class="hero-stat">
                            <span>Model Backbone</span>
                            <strong>EfficientNet-B0</strong>
                        </div>
                        <div class="hero-stat">
                            <span>Inference</span>
                            <strong>Binary Classification + Grad-CAM</strong>
                        </div>
                        <div class="hero-stat">
                            <span>Input Resolution</span>
                            <strong>{IMAGE_SIZE} x {IMAGE_SIZE}</strong>
                        </div>
                    </div>
                </div>
            </div>
            """
        )

        threshold = gr.Slider(
            minimum=0.10,
            maximum=0.90,
            value=DEFAULT_THRESHOLD,
            step=0.05,
            label="Malignant Screening Threshold",
            info="Lower values flag more images as malignant. Default is 50%.",
        )

        with gr.Tabs():
            with gr.Tab("Single Analysis"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Upload Image</h3>
                                    <p class="panel-subtitle">
                                        Use a clear dermoscopy image with the lesion centered for best results.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                image_input = gr.Image(
                                    type="pil",
                                    label="Lesion Image",
                                    height=360,
                                    elem_classes=["upload-box"],
                                )
                                with gr.Row():
                                    use_tta = gr.Checkbox(label="Enable Test-Time Augmentation (TTA)", value=True, info="Boosts accuracy via averaging augmentations")
                                    run_abcde_chk = gr.Checkbox(label="Run ABCDE Auto-Vision Analysis", value=True)
                                predict_btn = gr.Button(
                                    "Analyze Image",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["analyze-btn"],
                                )
                                gr.Markdown(
                                    "Use the threshold slider above to make the screening decision more sensitive or more conservative.",
                                    elem_classes=["control-note"],
                                )
                                if examples:
                                    gr.Examples(examples=examples, inputs=image_input, label="Example images")

                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Analysis Summary</h3>
                                    <p class="panel-subtitle">
                                        Threshold-aware decision, confidence distribution, and screening guidance.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                result_html = gr.HTML(value=build_placeholder_html())

                with gr.Row(equal_height=True):
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Visual Explanation</h3>
                                    <p class="panel-subtitle">
                                        Grad-CAM highlights the regions most influential to the model.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                gradcam_out = gr.Image(
                                    label="Grad-CAM Visualization",
                                    height=380,
                                    elem_classes=["output-box"],
                                )
                                with gr.Row():
                                    snapshot_file = gr.File(
                                        label="Download Visualization",
                                        elem_classes=["download-box"],
                                    )
                                    report_file = gr.File(
                                        label="Download Report Card",
                                        elem_classes=["download-box"],
                                    )

                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">How To Use</h3>
                                    <p class="panel-subtitle">
                                        A few simple notes to keep the workflow clear and safe.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                gr.HTML(
                                    f"""
                                    <div class="guide-list">
                                        <div class="guide-item">
                                            <strong>1. Upload a focused lesion image</strong>
                                            <span>Use a well-lit image with minimal blur and the lesion near the center.</span>
                                        </div>
                                        <div class="guide-item">
                                            <strong>2. Review the attention map</strong>
                                            <span>The heatmap shows where the model looked, not a definitive medical explanation.</span>
                                        </div>
                                        <div class="guide-item">
                                            <strong>3. Use results as screening only</strong>
                                            <span>Any concerning lesion should still be assessed by a qualified dermatologist.</span>
                                        </div>
                                    </div>
                                    <div class="info-grid">
                                        <div class="info-card">
                                            <span>Checkpoint</span>
                                            <strong>{CHECKPOINT.name if CHECKPOINT.exists() else "Demo mode"}</strong>
                                        </div>
                                        <div class="info-card">
                                            <span>Device</span>
                                            <strong>{DEVICE.type.upper()}</strong>
                                        </div>
                                        <div class="info-card">
                                            <span>Classes</span>
                                            <strong>{" / ".join(CLASS_NAMES)}</strong>
                                        </div>
                                        <div class="info-card">
                                            <span>Image Size</span>
                                            <strong>{IMAGE_SIZE} x {IMAGE_SIZE}</strong>
                                        </div>
                                    </div>
                                    """
                                )

            with gr.Tab("Multi-Class Diagnosis"):
                with gr.Group(elem_classes=["panel"]):
                    gr.HTML(
                        '''
                        <div class="panel-header">
                            <h3 class="panel-title">9-Class Probability Distribution</h3>
                            <p class="panel-subtitle">
                                The multi-class model estimates probabilities across 9 diagnostic categories.
                            </p>
                        </div>
                        '''
                    )
                    with gr.Group(elem_classes=["panel-body"]):
                        mc_output_img = gr.Image(label="Multi-Class Predictions", interactive=False, elem_classes=["output-box"])

            with gr.Tab("ABCDE Auto-Vision Details"):
                with gr.Group(elem_classes=["panel"]):
                    gr.HTML(
                        '''
                        <div class="panel-header">
                            <h3 class="panel-title">ABCDE Dermatological Features</h3>
                            <p class="panel-subtitle">
                                Automated lesion segmentation and rule-based computer vision analysis of the classic ABCDE criteria.
                            </p>
                        </div>
                        '''
                    )
                    with gr.Group(elem_classes=["panel-body"]):
                        abcde_output_img = gr.Image(label="ABCDE Analysis Details", interactive=False, elem_classes=["output-box"])

            with gr.Tab("Symptom Review"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Official Warning-Sign Intake</h3>
                                    <p class="panel-subtitle">
                                        This uses official warning-sign criteria and produces a transparent triage summary.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                evolving = gr.Checkbox(label="The spot is changing or evolving")
                                asymmetry = gr.Checkbox(label="The lesion looks asymmetric")
                                irregular_border = gr.Checkbox(label="The border looks irregular")
                                multiple_colors = gr.Checkbox(label="There are multiple colors in the lesion")
                                diameter_large = gr.Checkbox(label="The lesion seems larger than about 6 mm")
                                itching_or_tender = gr.Checkbox(label="Itching or tenderness is present")
                                bleeding_or_oozing = gr.Checkbox(label="Bleeding or oozing is present")
                                personal_or_family_history = gr.Checkbox(label="Personal or family history of skin cancer")
                                symptom_btn = gr.Button(
                                    "Review Symptoms",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["analyze-btn"],
                                )
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Symptom Triage Summary</h3>
                                    <p class="panel-subtitle">
                                        Combined guidance keeps symptom review separate from the image model and cites the official source pages.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                symptom_html = gr.HTML(value=build_placeholder_html())
                                combined_summary = gr.HTML(
                                    value="""
                                    <div class="mini-summary">
                                        <strong>Ready for symptom review.</strong>
                                        <span>Select any warning signs that apply, then generate the triage summary.</span>
                                    </div>
                                    """
                                )
                                symptom_sources = gr.HTML(
                                    value=(
                                        "<div class='mini-summary'><strong>Official source pages</strong>"
                                        f"{source_link_list(SYMPTOM_SOURCE_URLS)}</div>"
                                    )
                                )

            with gr.Tab("Batch Analysis"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Batch Upload</h3>
                                    <p class="panel-subtitle">
                                        Analyze multiple images at once and export the batch results to CSV.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                batch_files = gr.Files(label="Upload multiple lesion images")
                                batch_btn = gr.Button(
                                    "Run Batch Analysis",
                                    variant="primary",
                                    size="lg",
                                    elem_classes=["analyze-btn"],
                                )
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Batch Results</h3>
                                    <p class="panel-subtitle">
                                        Summary, per-image screening decisions, and a downloadable CSV export.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                batch_summary = gr.HTML(
                                    value="""
                                    <div class="mini-summary">
                                        <strong>Batch mode is ready.</strong>
                                        <span>Upload one or more images to generate a results table.</span>
                                    </div>
                                    """
                                )
                                batch_table = gr.Dataframe(
                                    value=empty_history_df(),
                                    headers=HISTORY_COLUMNS,
                                    interactive=False,
                                    wrap=True,
                                )
                                batch_csv = gr.File(label="Download Batch CSV", elem_classes=["download-box"])

            with gr.Tab("History"):
                with gr.Group(elem_classes=["panel"]):
                    gr.HTML(
                        """
                        <div class="panel-header">
                            <h3 class="panel-title">Session History</h3>
                            <p class="panel-subtitle">
                                Every single and batch analysis performed in this session appears here.
                            </p>
                        </div>
                        """
                    )
                    with gr.Group(elem_classes=["panel-body"]):
                        history_table = gr.Dataframe(
                            value=empty_history_df(),
                            headers=HISTORY_COLUMNS,
                            interactive=False,
                            wrap=True,
                        )
                        with gr.Row():
                            history_csv = gr.File(label="Download Session CSV", elem_classes=["download-box"])
                            clear_btn = gr.Button("Clear History", elem_classes=["soft-btn"])

            with gr.Tab("Model Dashboard"):
                with gr.Group(elem_classes=["panel"]):
                    gr.HTML(
                        """
                        <div class="panel-header">
                            <h3 class="panel-title">Saved Training Visuals</h3>
                            <p class="panel-subtitle">
                                These panels display the latest plots already generated by your evaluation pipeline.
                            </p>
                        </div>
                        """
                    )
                    with gr.Group(elem_classes=["panel-body"]):
                        gr.HTML(
                            """
                            <div class="mini-summary">
                                <strong>Display-only dashboard.</strong>
                                <span>No retraining is happening here; the app is simply showing the saved output artifacts from <code>outputs/</code>.</span>
                            </div>
                            """
                        )
                        with gr.Row():
                            confusion_img = gr.Image(value=display_plot(PLOT_FILES["Confusion Matrix"]), label="Confusion Matrix", interactive=False, elem_classes=["output-box"])
                            roc_img = gr.Image(value=display_plot(PLOT_FILES["ROC Curve"]), label="ROC Curve", interactive=False, elem_classes=["output-box"])
                        training_img = gr.Image(value=display_plot(PLOT_FILES["Training History"]), label="Training History", interactive=False, elem_classes=["output-box"])

            with gr.Tab("Healthy Skin Guide"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=7):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Healthy Skin, Happy Skin</h3>
                                    <p class="panel-subtitle">
                                        Fun, conservative skin-care advice grounded in trustworthy public-health and dermatology sources.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                gr.HTML(build_skin_care_fun_html())
                    with gr.Column(scale=5):
                        with gr.Group(elem_classes=["panel"]):
                            gr.HTML(
                                """
                                <div class="panel-header">
                                    <h3 class="panel-title">Quick Habits That Actually Matter</h3>
                                    <p class="panel-subtitle">
                                        The boring basics are secretly elite.
                                    </p>
                                </div>
                                """
                            )
                            with gr.Group(elem_classes=["panel-body"]):
                                gr.HTML(
                                    """
                                    <div class="guide-list">
                                        <div class="guide-item">
                                            <strong>SPF is not optional side content</strong>
                                            <span>Make sun protection a routine, not a panic button.</span>
                                        </div>
                                        <div class="guide-item">
                                            <strong>Patch test your chaos</strong>
                                            <span>New products are exciting, but skin prefers drama-free introductions.</span>
                                        </div>
                                        <div class="guide-item">
                                            <strong>Hydrate, sleep, repeat</strong>
                                            <span>Skin loves consistency more than miracle claims.</span>
                                        </div>
                                        <div class="guide-item">
                                            <strong>If a spot changes, do not negotiate with it</strong>
                                            <span>Get it checked instead of trying to out-stubborn biology.</span>
                                        </div>
                                    </div>
                                """
                                )

        predict_btn.click(
            fn=run_single_analysis,
            inputs=[image_input, threshold, use_tta, run_abcde_chk, history_state],
            outputs=[
                gradcam_out,
                result_html,
                snapshot_file,
                report_file,
                history_table,
                history_csv,
                history_state,
                latest_analysis_state,
                mc_output_img,
                abcde_output_img
            ],
        )

        symptom_btn.click(
            fn=run_symptom_review,
            inputs=[
                latest_analysis_state,
                threshold,
                evolving,
                asymmetry,
                irregular_border,
                multiple_colors,
                diameter_large,
                itching_or_tender,
                bleeding_or_oozing,
                personal_or_family_history,
            ],
            outputs=[symptom_html, combined_summary, symptom_sources],
        )

        batch_btn.click(
            fn=run_batch_analysis,
            inputs=[batch_files, threshold, history_state],
            outputs=[batch_summary, batch_table, batch_csv, history_table, history_csv, history_state],
        )

        clear_btn.click(
            fn=clear_history,
            outputs=[history_table, history_csv, history_state],
        )

        gr.HTML(
            """
            <p class="footer-note">
                This interface is intended for educational and research purposes only and is not a
                substitute for professional medical diagnosis.
            </p>
            """
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_port=7860, show_error=True)
