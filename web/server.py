"""
AnanthiX AI - Flask backend
Sert l'interface web et expose une API /api/predict
"""
import json
import io
import base64
from pathlib import Path
from flask import send_file
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================
# CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "class_names.json"
DISEASE_CARDS_PATH = Path(__file__).parent.parent / "app" / "disease_cards.json"

MODEL_CATALOG = {
    "v2": {
        "file": "resnet50_v2.pth",
        "label": "Model V2 — PlantVillage + PlantDoc",
        "short": "V2",
        "accuracy_lab": 99.77,
        "accuracy_terrain": 69.84,
        "tag": "RECOMMENDED",
    },
    "baseline": {
        "file": "resnet50_baseline.pth",
        "label": "Baseline — PlantVillage only",
        "short": "Baseline",
        "accuracy_lab": 99.72,
        "accuracy_terrain": 26.98,
        "tag": "REFERENCE",
    },
}

DEVICE = torch.device("cpu")
IMG_SIZE = 256
NUM_CLASSES = 38

# ============================================================
# CHARGEMENT MODÈLES + DONNÉES (1 fois au démarrage)
# ============================================================
def build_model(weights_path):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    state = torch.load(weights_path, map_location=DEVICE)
    m.load_state_dict(state)
    m.eval().to(DEVICE)
    return m

print("Loading models...")
MODELS = {}
CAMS = {}
for key, info in MODEL_CATALOG.items():
    path = MODELS_DIR / info["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    m = build_model(path)
    MODELS[key] = m
    CAMS[key] = GradCAM(model=m, target_layers=[m.layer4[-1]])
    print(f"  ✓ {key}")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)
    if isinstance(CLASS_NAMES, dict):
        CLASS_NAMES = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES, key=lambda x: int(x))]

with open(DISEASE_CARDS_PATH, "r", encoding="utf-8") as f:
    DISEASE_CARDS = json.load(f)

print(f"Loaded {len(CLASS_NAMES)} classes, {len(DISEASE_CARDS)} disease cards")

# ============================================================
# PRÉTRAITEMENT
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def image_to_base64(pil_or_np):
    """Convertit une image (PIL ou numpy uint8) en data URL base64."""
    if isinstance(pil_or_np, np.ndarray):
        pil = Image.fromarray(pil_or_np)
    else:
        pil = pil_or_np
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html",
        models=MODEL_CATALOG,
        disease_cards=DISEASE_CARDS,
        class_names=CLASS_NAMES,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "no_image"}), 400

    model_key = request.form.get("model", "v2")
    if model_key not in MODELS:
        return jsonify({"error": "invalid_model"}), 400

    use_gradcam = request.form.get("gradcam", "true").lower() == "true"

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "invalid_image", "detail": str(e)}), 400

    # Prétraitement
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    rgb_float = np.array(image_resized).astype(np.float32) / 255.0

    # Prédiction
    model = MODELS[model_key]
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(3)
    predictions = [
        {"class": CLASS_NAMES[i.item()], "prob": float(p.item())}
        for i, p in zip(top_idx, top_probs)
    ]

    # Grad-CAM (optionnel)
    gradcam_b64 = None
    if use_gradcam:
        cam = CAMS[model_key]
        grayscale = cam(input_tensor=tensor, targets=None)[0, :]
        overlay = show_cam_on_image(rgb_float, grayscale, use_rgb=True)
        gradcam_b64 = image_to_base64(overlay)

    # Image d'entrée renvoyée aussi (pour affichage propre)
    input_b64 = image_to_base64(image_resized)

    # Fiche maladie
    top_class = predictions[0]["class"]
    card = DISEASE_CARDS.get(top_class)

    return jsonify({
        "predictions": predictions,
        "input_image": input_b64,
        "gradcam_image": gradcam_b64,
        "card": card,
        "model_used": model_key,
    })
@app.route("/api/report", methods=["POST"])
def api_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from flask import send_file
 
    data = request.get_json()
    if not data:
        return jsonify({"error": "no_data"}), 400
 
    lang        = data.get("lang", "fr")
    report_data = data.get("data", {})
    predictions = report_data.get("predictions", [])
    card        = report_data.get("card")
    model_used  = report_data.get("model_used", "v2")
    input_img   = report_data.get("input_image", "")
    gradcam_img = report_data.get("gradcam_image", "")
 
    top = predictions[0] if predictions else {"class": "—", "prob": 0}
    confidence_pct = round(top["prob"] * 100, 2)
 
    # Labels bilingues
    if lang == "fr":
        L = {
            "title":      "Rapport de diagnostic — AnanthiX AI",
            "date_label": "Date",
            "model_label":"Moteur de diagnostic",
            "disease":    "Maladie détectée",
            "plant":      "Plante",
            "severity":   "Gravité",
            "confidence": "Score de confiance",
            "input_img":  "Image analysée",
            "gradcam_img":"Zones analysées (Grad-CAM)",
            "top3":       "Top 3 des hypothèses",
            "symptoms":   "Symptômes",
            "treatment":  "Traitement",
            "prevention": "Prévention",
            "footer":     "AnanthiX AI — Outil d'aide au diagnostic. Ne remplace pas l'avis d'un agronome.",
            "conf_high":  "Diagnostic fiable",
            "conf_med":   "Diagnostic à vérifier",
            "conf_low":   "Résultat non concluant",
        }
    else:
        L = {
            "title":      "Diagnosis Report — AnanthiX AI",
            "date_label": "Date",
            "model_label":"Diagnostic engine",
            "disease":    "Detected disease",
            "plant":      "Plant",
            "severity":   "Severity",
            "confidence": "Confidence score",
            "input_img":  "Analysed image",
            "gradcam_img":"Analysis zones (Grad-CAM)",
            "top3":       "Top 3 hypotheses",
            "symptoms":   "Symptoms",
            "treatment":  "Treatment",
            "prevention": "Prevention",
            "footer":     "AnanthiX AI — Decision-support tool. Not a substitute for agronomic expertise.",
            "conf_high":  "Reliable diagnosis",
            "conf_med":   "Verify with an expert",
            "conf_low":   "Inconclusive result",
        }
 
    # Confidence label
    if top["prob"] >= 0.85:
        conf_label = L["conf_high"]
        conf_color = (0.36, 0.55, 0.36)  # vert
    elif top["prob"] >= 0.50:
        conf_label = L["conf_med"]
        conf_color = (0.78, 0.66, 0.29)  # or
    else:
        conf_label = L["conf_low"]
        conf_color = (0.72, 0.36, 0.36)  # rouge
 
    # Decode images base64 → PIL
    def b64_to_pil(b64str):
        if not b64str or "," not in b64str:
            return None
        raw = base64.b64decode(b64str.split(",")[1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
 
    img_input   = b64_to_pil(input_img)
    img_gradcam = b64_to_pil(gradcam_img)
 
    def pil_to_tmp(pil_img):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf
 
    # ── Build PDF ──────────────────────────────────────────────
    buf = io.BytesIO()
    W, H = A4  # 595 x 842 pt
    c = rl_canvas.Canvas(buf, pagesize=A4)
 
    # Palette (sombre simulée en gris foncé sur blanc)
    BG   = (0.97, 0.97, 0.96)
    DARK = (0.06, 0.08, 0.07)
    GRN  = (0.29, 0.55, 0.36)
    MUT  = (0.61, 0.66, 0.61)
 
    def setfill(rgb): c.setFillColorRGB(*rgb)
    def setstroke(rgb): c.setStrokeColorRGB(*rgb)
 
    # Fond
    setfill(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
 
    # Header band
    setfill(DARK)
    c.rect(0, H - 48*mm, W, 48*mm, fill=1, stroke=0)
 
    # Barre verte
    setfill(GRN)
    c.rect(0, H - 48*mm, 4, 48*mm, fill=1, stroke=0)
 
    # Titre
    setfill((1,1,1))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(10*mm, H - 18*mm, "AnanthiX AI")
    c.setFont("Helvetica", 11)
    c.drawString(10*mm, H - 27*mm, L["title"])
 
    from datetime import datetime
    c.setFont("Helvetica", 9)
    setfill((0.7, 0.7, 0.7))
    c.drawString(10*mm, H - 36*mm, f"{L['date_label']} : {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.drawString(10*mm, H - 43*mm, f"{L['model_label']} : {model_used.upper()}")
 
    y = H - 58*mm  # curseur Y courant
 
    # ── Section : Résultat principal ──────────────────────────
    def section_title(label, y_pos):
        setfill(DARK)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(10*mm, y_pos, label.upper())
        setstroke(GRN)
        c.setLineWidth(1.5)
        c.line(10*mm, y_pos - 2, W - 10*mm, y_pos - 2)
        return y_pos - 8*mm
 
    def field_row(label, value, y_pos, value_color=DARK):
        setfill(MUT)
        c.setFont("Helvetica", 8)
        c.drawString(10*mm, y_pos, label)
        setfill(value_color)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(55*mm, y_pos, str(value))
        return y_pos - 7*mm
 
    disease_name = card[f"disease_{lang}"] if card else top["class"]
    plant_name   = card[f"plant_{lang}"] if card else "—"
    severity_raw = card.get("severity", "—") if card else "—"
    severity_map_fr = {"saine": "Saine", "légère": "Légère", "modérée": "Modérée", "critique": "Critique"}
    severity_map_en = {"saine": "Healthy", "légère": "Mild", "modérée": "Moderate", "critique": "Critical"}
    sev_map = severity_map_fr if lang == "fr" else severity_map_en
    severity_label = sev_map.get(severity_raw, severity_raw)
 
    y = section_title(L["disease"] if lang == "fr" else "Diagnosis", y)
    y = field_row(L["disease"],    disease_name, y)
    y = field_row(L["plant"],      plant_name, y)
    y = field_row(L["severity"],   severity_label, y)
    y = field_row(L["confidence"], f"{confidence_pct}% — {conf_label}", y, conf_color)
    y -= 4*mm
 
    # ── Images côte à côte ────────────────────────────────────
    if img_input or img_gradcam:
        img_w = 80*mm
        img_h = 80*mm
 
        if img_input:
            tmp = pil_to_tmp(img_input)
            c.drawImage(
                rl_canvas.ImageReader(tmp),
                10*mm, y - img_h, img_w, img_h,
                preserveAspectRatio=True
            )
            setfill(MUT); c.setFont("Helvetica", 8)
            c.drawString(10*mm, y - img_h - 5*mm, L["input_img"])
 
        if img_gradcam:
            tmp2 = pil_to_tmp(img_gradcam)
            c.drawImage(
                rl_canvas.ImageReader(tmp2),
                100*mm, y - img_h, img_w, img_h,
                preserveAspectRatio=True
            )
            setfill(MUT); c.setFont("Helvetica", 8)
            c.drawString(100*mm, y - img_h - 5*mm, L["gradcam_img"])
 
        y -= img_h + 12*mm
 
    # ── Top 3 ─────────────────────────────────────────────────
    if predictions:
        y = section_title(L["top3"], y)
        for i, p in enumerate(predictions[:3]):
            pct = round(p["prob"] * 100, 2)
            bar_w = (pct / 100) * 80*mm
            # Barre
            setfill((0.9, 0.9, 0.9)); setstroke((0.9,0.9,0.9))
            c.rect(55*mm, y - 1*mm, 80*mm, 4, fill=1, stroke=0)
            setfill(GRN)
            c.rect(55*mm, y - 1*mm, bar_w, 4, fill=1, stroke=0)
            # Texte
            setfill(DARK); c.setFont("Helvetica", 9)
            c.drawString(10*mm, y, f"{i+1}. {p['class']}")
            c.drawString(138*mm, y, f"{pct}%")
            y -= 6*mm
        y -= 4*mm
 
    # ── Fiche maladie ─────────────────────────────────────────
    if card and not card.get("is_healthy"):
        def list_section(title, items, y_pos):
            if not items: return y_pos
            y_pos = section_title(title, y_pos)
            for item in items:
                # Wrapping manuel simple
                words = item.split()
                line = ""
                for word in words:
                    test = line + (" " if line else "") + word
                    if c.stringWidth(test, "Helvetica", 9) < 170*mm:
                        line = test
                    else:
                        setfill(DARK); c.setFont("Helvetica", 9)
                        c.drawString(14*mm, y_pos, "— " + line)
                        y_pos -= 5*mm
                        line = word
                if line:
                    setfill(DARK); c.setFont("Helvetica", 9)
                    c.drawString(14*mm, y_pos, "— " + line)
                    y_pos -= 5*mm
                # Saut de page si nécessaire
                if y_pos < 30*mm:
                    c.showPage()
                    setfill(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
                    y_pos = H - 20*mm
            return y_pos - 3*mm
 
        symptoms  = card.get(f"symptoms_{lang}", [])
        treatment = card.get(f"treatment_{lang}", [])
        prevention= card.get(f"prevention_{lang}", [])
 
        if y < 80*mm:
            c.showPage()
            setfill(BG); c.rect(0, 0, W, H, fill=1, stroke=0)
            y = H - 20*mm
 
        y = list_section(L["symptoms"],   symptoms,   y)
        y = list_section(L["treatment"],  treatment,  y)
        y = list_section(L["prevention"], prevention, y)
 
    # ── Footer ────────────────────────────────────────────────
    setfill(DARK)
    c.rect(0, 0, W, 14*mm, fill=1, stroke=0)
    setfill((0.7, 0.7, 0.7)); c.setFont("Helvetica", 8)
    c.drawString(10*mm, 5*mm, L["footer"])
    setfill(GRN); c.setFont("Helvetica-Bold", 8)
    c.drawRightString(W - 10*mm, 5*mm, "AnanthiX AI")
 
    c.save()
    buf.seek(0)
 
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="rapport_ananthix.pdf"
    )

if __name__ == "__main__":
    print("\nServer running on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)