# """
# AnanthiX AI - Streamlit MVP v4
# - Upload image → prédiction
# - Confiance colorée (🟢🟡🔴)
# - Fiche maladie bilingue détaillée
# - Message générique si confiance faible
# - Grad-CAM : visualisation des zones de décision du modèle
# """

# import json
# from pathlib import Path

# import numpy as np
# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # ============================================================
# # CONFIGURATION
# # ============================================================
# PROJECT_ROOT = Path(__file__).parent.parent
# MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_baseline.pth"
# CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "class_names.json"
# DISEASE_CARDS_PATH = Path(__file__).parent / "disease_cards.json"

# DEVICE = torch.device("cpu")
# IMG_SIZE = 256
# NUM_CLASSES = 38

# CONF_HIGH = 0.85
# CONF_LOW = 0.50

# # ============================================================
# # CHARGEMENTS (cache)
# # ============================================================
# @st.cache_resource
# def load_model():
#     model = models.resnet50(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
#     state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
#     model.load_state_dict(state_dict)
#     model.eval()
#     model.to(DEVICE)
#     return model

# @st.cache_resource
# def load_class_names():
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, dict):
#         data = [data[k] for k in sorted(data, key=lambda x: int(x))]
#     return data

# @st.cache_resource
# def load_disease_cards():
#     with open(DISEASE_CARDS_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# @st.cache_resource
# def get_gradcam(_model):
#     """Instancie GradCAM une seule fois (lourd à initialiser)."""
#     target_layers = [_model.layer4[-1]]
#     return GradCAM(model=_model, target_layers=target_layers)

# # ============================================================
# # PRÉTRAITEMENT
# # ============================================================
# preprocess = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

# def prepare_image(image):
#     """Prépare l'image et retourne (tensor, rgb_float pour overlay)."""
#     image = image.convert("RGB")
#     tensor = preprocess(image).unsqueeze(0).to(DEVICE)
#     image_resized = image.resize((IMG_SIZE, IMG_SIZE))
#     rgb_float = np.array(image_resized).astype(np.float32) / 255.0
#     return tensor, rgb_float

# def predict(tensor, model, class_names, top_k=3):
#     with torch.no_grad():
#         logits = model(tensor)
#         probs = F.softmax(logits, dim=1)[0]
#     top_probs, top_idx = probs.topk(top_k)
#     return [
#         (class_names[i.item()], p.item())
#         for i, p in zip(top_idx, top_probs)
#     ]

# def compute_gradcam(tensor, rgb_float, cam):
#     """Génère l'overlay Grad-CAM."""
#     grayscale_cam = cam(input_tensor=tensor, targets=None)
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
#     return visualization, grayscale_cam

# # ============================================================
# # HELPERS D'AFFICHAGE
# # ============================================================
# def confidence_badge(conf):
#     if conf >= CONF_HIGH:
#         return "🟢", "Diagnostic fiable", "#22c55e"
#     elif conf >= CONF_LOW:
#         return "🟡", "Diagnostic probable — à vérifier", "#eab308"
#     else:
#         return "🔴", "Faible confiance — consulter un expert", "#ef4444"

# def severity_badge(severity):
#     mapping = {
#         "saine":    ("", "#22c55e"),
#         "légère":   ("", "#84cc16"),
#         "modérée":  ("", "#eab308"),
#         "critique": ("", "#ef4444"),
#     }
#     return mapping.get(severity, ("ℹ️", "#6b7280"))

# def display_low_confidence_message(lang="fr"):
#     if lang == "fr":
#         st.warning(
#             " **Le modèle n'est pas suffisamment confiant pour poser un diagnostic fiable.**\n\n"
#             "Cela peut être dû à : une image floue ou mal cadrée, une plante "
#             "non couverte par le modèle, ou une condition d'éclairage atypique.\n\n"
#             "**Recommandations :**\n"
#             "- Reprenez la photo en lumière naturelle, avec une feuille bien visible\n"
#             "- Évitez les arrière-plans encombrés (terre, mains, autres feuilles)\n"
#             "- Consultez un expert agricole ou un agronome local pour confirmation"
#         )
#     else:
#         st.warning(
#             " **The model is not confident enough for a reliable diagnosis.**\n\n"
#             "This may be due to: a blurry or poorly framed image, a plant not "
#             "covered by the model, or unusual lighting conditions.\n\n"
#             "**Recommendations:**\n"
#             "- Retake the photo in natural light with the leaf clearly visible\n"
#             "- Avoid cluttered backgrounds (soil, hands, other leaves)\n"
#             "- Consult an agricultural expert or local agronomist to confirm"
#         )

# def display_disease_card(card, lang="fr"):
#     suffix = f"_{lang}"
#     plant = card[f"plant{suffix}"]
#     disease = card[f"disease{suffix}"]
#     emoji, color = severity_badge(card["severity"])

#     st.markdown(
#         f"<div style='background:{color}20; padding:1rem; border-radius:8px; "
#         f"border-left:4px solid {color}; margin-bottom:1rem;'>"
#         f"<h3 style='margin:0;'>{emoji} {plant} — {disease}</h3>"
#         f"<p style='margin:0.3rem 0 0 0; color:{color}; font-weight:600;'>"
#         f"Sévérité : {card['severity']}</p>"
#         f"</div>",
#         unsafe_allow_html=True,
#     )

#     if card["is_healthy"]:
#         msg = "Aucune maladie détectée. Continuez vos bonnes pratiques." if lang == "fr" \
#               else "No disease detected. Keep up your good practices."
#         st.success(msg)
#         exp_label = "💡 Conseils de prévention" if lang == "fr" else "💡 Prevention tips"
#         with st.expander(exp_label):
#             for tip in card[f"prevention{suffix}"]:
#                 st.write(f"- {tip}")
#         return

#     tabs_labels = [" Symptômes", " Traitement", " Prévention"] if lang == "fr" \
#                   else [" Symptoms", " Treatment", " Prevention"]
#     tab1, tab2, tab3 = st.tabs(tabs_labels)
#     with tab1:
#         for s in card[f"symptoms{suffix}"]:
#             st.write(f"- {s}")
#     with tab2:
#         for t in card[f"treatment{suffix}"]:
#             st.write(f"- {t}")
#     with tab3:
#         for p in card[f"prevention{suffix}"]:
#             st.write(f"- {p}")

# # ============================================================
# # UI
# # ============================================================
# st.set_page_config(page_title="AnanthiX AI", page_icon="", layout="wide")

# st.title(" AnanthiX AI")
# st.caption("Diagnostic des maladies des plantes ")

# # Vérifications
# for path, label in [
#     (MODEL_PATH, "Modèle"),
#     (CLASS_NAMES_PATH, "class_names.json"),
#     (DISEASE_CARDS_PATH, "disease_cards.json"),
# ]:
#     if not path.exists():
#         st.error(f" {label} introuvable : {path}")
#         st.stop()

# with st.spinner("Chargement du modèle..."):
#     model = load_model()
#     class_names = load_class_names()
#     disease_cards = load_disease_cards()
#     cam = get_gradcam(model)

# # Sidebar
# lang = st.sidebar.radio(
#     " Langue / Language",
#     ["fr", "en"],
#     format_func=lambda x: "Français" if x == "fr" else "English",
# )

# show_gradcam = st.sidebar.checkbox(
#     " Afficher Grad-CAM" if lang == "fr" else " Show Grad-CAM",
#     value=True,
#     help="Visualise les zones que le modèle utilise pour décider" if lang == "fr" \
#          else "Visualizes the areas the model uses to decide",
# )

# st.divider()

# uploaded = st.file_uploader(
#     "Téléverser une photo de feuille" if lang == "fr" else "Upload a leaf photo",
#     type=["jpg", "jpeg", "png"],
# )

# if uploaded is not None:
#     image = Image.open(uploaded)

#     # Préparation et prédiction
#     with st.spinner("Analyse en cours..." if lang == "fr" else "Analyzing..."):
#         tensor, rgb_float = prepare_image(image)
#         predictions = predict(tensor, model, class_names, top_k=3)

#     top_class, top_conf = predictions[0]

#     # Layout : image(s) en haut, infos en bas
#     if show_gradcam:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption="Image originale" if lang == "fr" else "Original image", width="stretch")
#         with col2:
#             with st.spinner("Génération Grad-CAM..." if lang == "fr" else "Generating Grad-CAM..."):
#                 overlay, _ = compute_gradcam(tensor, rgb_float, cam)
#             cap = " Grad-CAM : zones d'attention du modèle" if lang == "fr" \
#                   else " Grad-CAM: model attention areas"
#             st.image(overlay, caption=cap, width="stretch")
        
#         # Légende explicative
#         with st.expander(" Comment lire Grad-CAM ?" if lang == "fr" else " How to read Grad-CAM?"):
#             if lang == "fr":
#                 st.markdown(
#                     "Les **zones rouges/jaunes** indiquent les régions de l'image qui ont "
#                     "le plus influencé la décision du modèle.\n\n"
#                     "-  **Heatmap sur la feuille / les symptômes** : le modèle a appris les bonnes features\n"
#                     "-  **Heatmap sur le fond / les bords** : le modèle utilise des indices non pertinents → diagnostic à vérifier"
#                 )
#             else:
#                 st.markdown(
#                     "**Red/yellow areas** indicate the regions of the image that most "
#                     "influenced the model's decision.\n\n"
#                     "-  **Heatmap on the leaf / symptoms**: the model learned good features\n"
#                     "-  **Heatmap on background / edges**: the model uses irrelevant cues → diagnosis to verify"
#                 )
#     else:
#         st.image(image, caption="Image envoyée" if lang == "fr" else "Submitted image", width="stretch")

#     # Badge de confiance
#     emoji, label, color = confidence_badge(top_conf)
#     st.markdown(
#         f"<div style='background:{color}20; padding:0.8rem 1rem; border-radius:8px; "
#         f"border-left:4px solid {color}; margin:1rem 0;'>"
#         f"<strong>{emoji} {label}</strong><br>"
#         f"Confiance : <strong>{top_conf*100:.2f}%</strong>"
#         f"</div>",
#         unsafe_allow_html=True,
#     )

#     # Top 3
#     exp_label = "Voir le top 3 des prédictions" if lang == "fr" else "View top 3 predictions"
#     with st.expander(exp_label):
#         for i, (cls, p) in enumerate(predictions, 1):
#             st.markdown(f"**{i}.** `{cls}` — **{p*100:.2f}%**")

#     st.divider()

#     # Fiche maladie ou message faible confiance
#     if top_conf < CONF_LOW:
#         display_low_confidence_message(lang=lang)
#     elif top_class in disease_cards:
#         display_disease_card(disease_cards[top_class], lang=lang)
#     else:
#         st.error(f" Erreur technique : fiche manquante pour `{top_class}`")
# else:
#     msg = " Téléverse une image pour lancer le diagnostic." if lang == "fr" \
#           else " Upload an image to start the diagnosis."
#     st.info(msg)


#2nd version
# """
# AnanthiX AI - Streamlit MVP v5
# - Sélecteur de modèle (V2 par défaut, baseline en option)
# - Affichage de la version du modèle utilisé
# - Tout le reste : confiance colorée, fiches bilingues, Grad-CAM
# """

# import json
# from pathlib import Path

# import numpy as np
# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models, transforms
# from PIL import Image

# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image

# # ============================================================
# # CONFIGURATION
# # ============================================================
# PROJECT_ROOT = Path(__file__).parent.parent
# MODELS_DIR = PROJECT_ROOT / "models"
# CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "class_names.json"
# DISEASE_CARDS_PATH = Path(__file__).parent / "disease_cards.json"

# # Catalogue des modèles disponibles
# MODEL_CATALOG = {
#     "v2": {
#         "file": "resnet50_v2.pth",
#         "label": "V2 (recommandé) — entraîné sur PlantVillage + PlantDoc",
#         "accuracy_lab": 99.77,
#         "accuracy_terrain": 69.84,
#         "badge": "🌿 V2",
#     },
#     "baseline": {
#         "file": "resnet50_baseline.pth",
#         "label": "Baseline — entraîné uniquement sur PlantVillage",
#         "accuracy_lab": 99.72,
#         "accuracy_terrain": 26.98,
#         "badge": "📊 Baseline",
#     },
# }

# DEVICE = torch.device("cpu")
# IMG_SIZE = 256
# NUM_CLASSES = 38

# CONF_HIGH = 0.85
# CONF_LOW = 0.50

# # ============================================================
# # CHARGEMENTS (cache)
# # ============================================================
# @st.cache_resource
# def load_model(model_key: str):
#     """Charge un modèle selon sa clé (v2 ou baseline)."""
#     model_info = MODEL_CATALOG[model_key]
#     model_path = MODELS_DIR / model_info["file"]
    
#     model = models.resnet50(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
#     state_dict = torch.load(model_path, map_location=DEVICE)
#     model.load_state_dict(state_dict)
#     model.eval()
#     model.to(DEVICE)
#     return model

# @st.cache_resource
# def load_class_names():
#     with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     if isinstance(data, dict):
#         data = [data[k] for k in sorted(data, key=lambda x: int(x))]
#     return data

# @st.cache_resource
# def load_disease_cards():
#     with open(DISEASE_CARDS_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# @st.cache_resource
# def get_gradcam(_model, model_key: str):
#     """Cache séparé par model_key pour ne pas mélanger les CAM."""
#     target_layers = [_model.layer4[-1]]
#     return GradCAM(model=_model, target_layers=target_layers)

# # ============================================================
# # PRÉTRAITEMENT
# # ============================================================
# preprocess = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

# def prepare_image(image):
#     image = image.convert("RGB")
#     tensor = preprocess(image).unsqueeze(0).to(DEVICE)
#     image_resized = image.resize((IMG_SIZE, IMG_SIZE))
#     rgb_float = np.array(image_resized).astype(np.float32) / 255.0
#     return tensor, rgb_float

# def predict(tensor, model, class_names, top_k=3):
#     with torch.no_grad():
#         logits = model(tensor)
#         probs = F.softmax(logits, dim=1)[0]
#     top_probs, top_idx = probs.topk(top_k)
#     return [
#         (class_names[i.item()], p.item())
#         for i, p in zip(top_idx, top_probs)
#     ]

# def compute_gradcam(tensor, rgb_float, cam):
#     grayscale_cam = cam(input_tensor=tensor, targets=None)
#     grayscale_cam = grayscale_cam[0, :]
#     visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
#     return visualization, grayscale_cam

# # ============================================================
# # HELPERS D'AFFICHAGE
# # ============================================================
# def confidence_badge(conf):
#     if conf >= CONF_HIGH:
#         return "🟢", "Diagnostic fiable", "#22c55e"
#     elif conf >= CONF_LOW:
#         return "🟡", "Diagnostic probable — à vérifier", "#eab308"
#     else:
#         return "🔴", "Faible confiance — consulter un expert", "#ef4444"

# def severity_badge(severity):
#     mapping = {
#         "saine":    ("✅", "#22c55e"),
#         "légère":   ("🟢", "#84cc16"),
#         "modérée":  ("⚠️", "#eab308"),
#         "critique": ("🚨", "#ef4444"),
#     }
#     return mapping.get(severity, ("ℹ️", "#6b7280"))

# def display_model_badge(model_key, lang="fr"):
#     """Affiche le badge du modèle utilisé."""
#     info = MODEL_CATALOG[model_key]
#     color = "#2C5F2D" if model_key == "v2" else "#6b7280"
#     perf_label = "Performance terrain" if lang == "fr" else "Field accuracy"
#     st.markdown(
#         f"<div style='background:{color}15; padding:0.5rem 1rem; border-radius:6px; "
#         f"border-left:3px solid {color}; margin-bottom:1rem; font-size:0.9em;'>"
#         f"<strong style='color:{color};'>{info['badge']}</strong> · "
#         f"{perf_label} : <strong>{info['accuracy_terrain']:.1f}%</strong>"
#         f"</div>",
#         unsafe_allow_html=True,
#     )

# def display_low_confidence_message(lang="fr"):
#     if lang == "fr":
#         st.warning(
#             "⚠️ **Le modèle n'est pas suffisamment confiant pour poser un diagnostic fiable.**\n\n"
#             "Cela peut être dû à : une image floue ou mal cadrée, une plante "
#             "non couverte par le modèle, ou une condition d'éclairage atypique.\n\n"
#             "**Recommandations :**\n"
#             "- Reprenez la photo en lumière naturelle, avec une feuille bien visible\n"
#             "- Évitez les arrière-plans encombrés (terre, mains, autres feuilles)\n"
#             "- Consultez un expert agricole ou un agronome local pour confirmation"
#         )
#     else:
#         st.warning(
#             "⚠️ **The model is not confident enough for a reliable diagnosis.**\n\n"
#             "This may be due to: a blurry or poorly framed image, a plant not "
#             "covered by the model, or unusual lighting conditions.\n\n"
#             "**Recommendations:**\n"
#             "- Retake the photo in natural light with the leaf clearly visible\n"
#             "- Avoid cluttered backgrounds (soil, hands, other leaves)\n"
#             "- Consult an agricultural expert or local agronomist to confirm"
#         )

# def display_disease_card(card, lang="fr"):
#     suffix = f"_{lang}"
#     plant = card[f"plant{suffix}"]
#     disease = card[f"disease{suffix}"]
#     emoji, color = severity_badge(card["severity"])

#     st.markdown(
#         f"<div style='background:{color}20; padding:1rem; border-radius:8px; "
#         f"border-left:4px solid {color}; margin-bottom:1rem;'>"
#         f"<h3 style='margin:0;'>{emoji} {plant} — {disease}</h3>"
#         f"<p style='margin:0.3rem 0 0 0; color:{color}; font-weight:600;'>"
#         f"Sévérité : {card['severity']}</p>"
#         f"</div>",
#         unsafe_allow_html=True,
#     )

#     if card["is_healthy"]:
#         msg = "Aucune maladie détectée. Continuez vos bonnes pratiques." if lang == "fr" \
#               else "No disease detected. Keep up your good practices."
#         st.success(msg)
#         exp_label = "💡 Conseils de prévention" if lang == "fr" else "💡 Prevention tips"
#         with st.expander(exp_label):
#             for tip in card[f"prevention{suffix}"]:
#                 st.write(f"- {tip}")
#         return

#     tabs_labels = ["🔍 Symptômes", "💊 Traitement", "🛡️ Prévention"] if lang == "fr" \
#                   else ["🔍 Symptoms", "💊 Treatment", "🛡️ Prevention"]
#     tab1, tab2, tab3 = st.tabs(tabs_labels)
#     with tab1:
#         for s in card[f"symptoms{suffix}"]:
#             st.write(f"- {s}")
#     with tab2:
#         for t in card[f"treatment{suffix}"]:
#             st.write(f"- {t}")
#     with tab3:
#         for p in card[f"prevention{suffix}"]:
#             st.write(f"- {p}")

# # ============================================================
# # UI
# # ============================================================
# st.set_page_config(page_title="AnanthiX AI", page_icon="🌿", layout="wide")

# st.title("🌿 AnanthiX AI")
# st.caption("Diagnostic des maladies des plantes — Jalon 4")

# # Vérifications de fichiers
# for path, label in [
#     (MODELS_DIR / MODEL_CATALOG["v2"]["file"], "Modèle V2"),
#     (MODELS_DIR / MODEL_CATALOG["baseline"]["file"], "Modèle baseline"),
#     (CLASS_NAMES_PATH, "class_names.json"),
#     (DISEASE_CARDS_PATH, "disease_cards.json"),
# ]:
#     if not path.exists():
#         st.error(f"❌ {label} introuvable : {path}")
#         st.stop()

# # ============================================================
# # SIDEBAR
# # ============================================================
# st.sidebar.header("⚙️ Configuration")

# # Langue
# lang = st.sidebar.radio(
#     "🌍 Langue / Language",
#     ["fr", "en"],
#     format_func=lambda x: "Français" if x == "fr" else "English",
# )

# # Sélecteur de modèle
# st.sidebar.divider()
# st.sidebar.subheader("🧠 Modèle" if lang == "fr" else "🧠 Model")

# model_key = st.sidebar.radio(
#     "Choix du modèle" if lang == "fr" else "Model choice",
#     options=["v2", "baseline"],
#     format_func=lambda k: MODEL_CATALOG[k]["label"],
#     index=0,  # V2 par défaut
# )

# # Comparatif performances
# with st.sidebar.expander("📊 Comparatif performances" if lang == "fr" else "📊 Performance comparison"):
#     st.markdown(
#         f"""
# | Modèle | Lab | Terrain |
# |--------|----:|--------:|
# | Baseline | 99.72% | 26.98% |
# | **V2** | **99.77%** | **69.84%** |

# *Gain V2 sur terrain : +42.86 pts*
# """
#     )

# # Grad-CAM
# st.sidebar.divider()
# show_gradcam = st.sidebar.checkbox(
#     "🔥 Afficher Grad-CAM" if lang == "fr" else "🔥 Show Grad-CAM",
#     value=True,
# )

# # ============================================================
# # CHARGEMENT DU MODÈLE SÉLECTIONNÉ
# # ============================================================
# with st.spinner(f"Chargement de {MODEL_CATALOG[model_key]['badge']}..."):
#     model = load_model(model_key)
#     class_names = load_class_names()
#     disease_cards = load_disease_cards()
#     cam = get_gradcam(model, model_key)

# # Badge du modèle actif
# display_model_badge(model_key, lang)

# st.divider()

# # ============================================================
# # UPLOAD ET PRÉDICTION
# # ============================================================
# uploaded = st.file_uploader(
#     "Téléverser une photo de feuille" if lang == "fr" else "Upload a leaf photo",
#     type=["jpg", "jpeg", "png"],
# )

# if uploaded is not None:
#     image = Image.open(uploaded)

#     with st.spinner("Analyse en cours..." if lang == "fr" else "Analyzing..."):
#         tensor, rgb_float = prepare_image(image)
#         predictions = predict(tensor, model, class_names, top_k=3)

#     top_class, top_conf = predictions[0]

#     # Layout : image(s) en haut, infos en bas
#     if show_gradcam:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(image, caption="Image originale" if lang == "fr" else "Original image", width="stretch")
#         with col2:
#             with st.spinner("Génération Grad-CAM..." if lang == "fr" else "Generating Grad-CAM..."):
#                 overlay, _ = compute_gradcam(tensor, rgb_float, cam)
#             cap = "🔥 Grad-CAM : zones d'attention du modèle" if lang == "fr" \
#                   else "🔥 Grad-CAM: model attention areas"
#             st.image(overlay, caption=cap, width="stretch")
        
#         with st.expander("ℹ️ Comment lire Grad-CAM ?" if lang == "fr" else "ℹ️ How to read Grad-CAM?"):
#             if lang == "fr":
#                 st.markdown(
#                     "Les **zones rouges/jaunes** indiquent les régions de l'image qui ont "
#                     "le plus influencé la décision du modèle.\n\n"
#                     "- ✅ **Heatmap sur la feuille / les symptômes** : le modèle a appris les bonnes features\n"
#                     "- ⚠️ **Heatmap sur le fond / les bords** : le modèle utilise des indices non pertinents → diagnostic à vérifier"
#                 )
#             else:
#                 st.markdown(
#                     "**Red/yellow areas** indicate the regions of the image that most "
#                     "influenced the model's decision.\n\n"
#                     "- ✅ **Heatmap on the leaf / symptoms**: the model learned good features\n"
#                     "- ⚠️ **Heatmap on background / edges**: the model uses irrelevant cues → diagnosis to verify"
#                 )
#     else:
#         st.image(image, caption="Image envoyée" if lang == "fr" else "Submitted image", width="stretch")

#     # Badge de confiance
#     emoji, label, color = confidence_badge(top_conf)
#     st.markdown(
#         f"<div style='background:{color}20; padding:0.8rem 1rem; border-radius:8px; "
#         f"border-left:4px solid {color}; margin:1rem 0;'>"
#         f"<strong>{emoji} {label}</strong><br>"
#         f"Confiance : <strong>{top_conf*100:.2f}%</strong>"
#         f"</div>",
#         unsafe_allow_html=True,
#     )

#     # Top 3
#     exp_label = "Voir le top 3 des prédictions" if lang == "fr" else "View top 3 predictions"
#     with st.expander(exp_label):
#         for i, (cls, p) in enumerate(predictions, 1):
#             st.markdown(f"**{i}.** `{cls}` — **{p*100:.2f}%**")

#     st.divider()

#     # Fiche maladie ou message faible confiance
#     if top_conf < CONF_LOW:
#         display_low_confidence_message(lang=lang)
#     elif top_class in disease_cards:
#         display_disease_card(disease_cards[top_class], lang=lang)
#     else:
#         st.error(f"⚠️ Erreur technique : fiche manquante pour `{top_class}`")
# else:
#     msg = "👆 Téléverse une image pour lancer le diagnostic." if lang == "fr" \
#           else "👆 Upload an image to start the diagnosis."
#     st.info(msg)

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ============================================================
# CONFIGURATION (inchangé)
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CLASS_NAMES_PATH = PROJECT_ROOT / "results" / "class_names.json"
DISEASE_CARDS_PATH = Path(__file__).parent / "disease_cards.json"

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
CONF_HIGH = 0.85
CONF_LOW = 0.50

# ============================================================
# CHARGEMENTS (inchangé)
# ============================================================
@st.cache_resource
def load_model(model_key: str):
    model_info = MODEL_CATALOG[model_key]
    model_path = MODELS_DIR / model_info["file"]
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data[k] for k in sorted(data, key=lambda x: int(x))]
    return data

@st.cache_resource
def load_disease_cards():
    with open(DISEASE_CARDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def get_gradcam(_model, model_key: str):
    target_layers = [_model.layer4[-1]]
    return GradCAM(model=_model, target_layers=target_layers)

# ============================================================
# PRÉTRAITEMENT (inchangé)
# ============================================================
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def prepare_image(image):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(image_resized).astype(np.float32) / 255.0
    return tensor, rgb_float

def predict(tensor, model, class_names, top_k=3):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(top_k)
    return [
        (class_names[i.item()], p.item())
        for i, p in zip(top_idx, top_probs)
    ]

def compute_gradcam(tensor, rgb_float, cam):
    grayscale_cam = cam(input_tensor=tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam

# ============================================================
# THÈME — palettes clair / sombre
# ============================================================
THEMES = {
    "light": {
        "bg":          "#FAFAF7",
        "surface":     "#FFFFFF",
        "border":      "#E5E5E5",
        "text":        "#1A1A1A",
        "text_muted":  "#6B7280",
        "accent":      "#1B4332",
        "accent_soft": "#F0F4EE",
        "sage":        "#6B7F5C",
        "ok":          "#15803D",
        "ok_soft":     "#DCFCE7",
        "warn":        "#B45309",
        "warn_soft":   "#FEF3C7",
        "err":         "#B91C1C",
        "err_soft":    "#FEE2E2",
        "neutral":     "#F3F4F6",
    },
    "dark": {
        "bg":          "#0F1410",
        "surface":     "#1A201C",
        "border":      "#2D332F",
        "text":        "#F5F5F0",
        "text_muted":  "#9CA3AF",
        "accent":      "#7FB069",
        "accent_soft": "#1F2A22",
        "sage":        "#A4B89D",
        "ok":          "#4ADE80",
        "ok_soft":     "#14271C",
        "warn":        "#FBBF24",
        "warn_soft":   "#2A2114",
        "err":         "#F87171",
        "err_soft":    "#2A1818",
        "neutral":     "#252B27",
    },
}

def inject_css(theme: dict):
    """Injecte la feuille de style globale selon le thème."""
    css = f"""
    <style>
    /* Imports police */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Variables globales */
    :root {{
        --bg: {theme['bg']};
        --surface: {theme['surface']};
        --border: {theme['border']};
        --text: {theme['text']};
        --text-muted: {theme['text_muted']};
        --accent: {theme['accent']};
        --accent-soft: {theme['accent_soft']};
        --sage: {theme['sage']};
        --ok: {theme['ok']};
        --ok-soft: {theme['ok_soft']};
        --warn: {theme['warn']};
        --warn-soft: {theme['warn_soft']};
        --err: {theme['err']};
        --err-soft: {theme['err_soft']};
        --neutral: {theme['neutral']};
    }}

    /* Fond global */
    .stApp {{
        background-color: var(--bg);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text);
    }}

    /* Titre principal et caption Streamlit */
    h1 {{
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.02em !important;
        color: var(--text) !important;
        margin-bottom: 0.25rem !important;
    }}
    h2, h3 {{
        font-weight: 600 !important;
        color: var(--text) !important;
        letter-spacing: -0.01em !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {{
        color: var(--text) !important;
    }}

    /* Composants Streamlit habillés */
    .stRadio > label, .stCheckbox > label {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text) !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] section {{
        background-color: var(--surface);
        border: 1.5px dashed var(--border);
        border-radius: 6px;
        padding: 1.25rem;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent);
    }}

    /* Tabs Streamlit */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
        padding: 0.75rem 1.25rem !important;
        background: transparent !important;
        border-bottom: 2px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }}

    /* Suppression decoration de page Streamlit */
    [data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2rem; }}

    /* Cards et composants custom */
    .ax-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }}

    .ax-section-label {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }}

    .ax-model-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem 1.25rem;
        margin-bottom: 1.5rem;
    }}
    .ax-model-bar .left {{ display: flex; align-items: center; gap: 0.75rem; }}
    .ax-model-bar .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--accent);
    }}
    .ax-model-bar .name {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8125rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: 0.04em;
    }}
    .ax-model-bar .tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.625rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: var(--accent);
        background: var(--accent-soft);
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
    }}
    .ax-model-bar .perf {{
        font-size: 0.8125rem;
        color: var(--text-muted);
    }}
    .ax-model-bar .perf strong {{ color: var(--text); }}

    .ax-confidence {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }}
    .ax-confidence-row {{
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 0.75rem;
    }}
    .ax-confidence-label {{
        display: flex; align-items: center; gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text);
    }}
    .ax-confidence-pill {{
        width: 10px; height: 10px; border-radius: 50%;
    }}
    .ax-confidence-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text);
    }}
    .ax-confidence-bar {{
        height: 4px;
        background: var(--neutral);
        border-radius: 2px;
        overflow: hidden;
    }}
    .ax-confidence-fill {{ height: 100%; border-radius: 2px; }}

    .ax-diagnosis {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-left-width: 3px;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .ax-diagnosis .plant {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }}
    .ax-diagnosis .disease {{
        font-size: 1.375rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
        margin-bottom: 0.5rem;
    }}
    .ax-diagnosis .severity {{
        display: inline-flex; align-items: center; gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.25rem 0.6rem;
        border-radius: 3px;
    }}

    .ax-warning {{
        background: var(--warn-soft);
        border: 1px solid var(--warn);
        border-radius: 6px;
        padding: 1rem 1.25rem;
        color: var(--text);
        font-size: 0.875rem;
        line-height: 1.5;
    }}
    .ax-warning strong {{ color: var(--warn); }}
    .ax-warning ul {{ margin: 0.5rem 0 0 1.25rem; padding: 0; }}
    .ax-warning li {{ margin-bottom: 0.25rem; }}

    .ax-empty {{
        background: var(--surface);
        border: 1px dashed var(--border);
        border-radius: 6px;
        padding: 2rem;
        text-align: center;
        color: var(--text-muted);
        font-size: 0.875rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================
# HELPERS AFFICHAGE — sans emoji, basés sur classes CSS
# ============================================================
def confidence_state(conf: float, theme: dict):
    if conf >= CONF_HIGH:
        return {
            "label_fr": "Confiance élevée", "label_en": "High confidence",
            "color": theme["ok"], "soft": theme["ok_soft"],
        }
    elif conf >= CONF_LOW:
        return {
            "label_fr": "Confiance modérée", "label_en": "Moderate confidence",
            "color": theme["warn"], "soft": theme["warn_soft"],
        }
    else:
        return {
            "label_fr": "Confiance faible", "label_en": "Low confidence",
            "color": theme["err"], "soft": theme["err_soft"],
        }

def severity_state(severity: str, theme: dict):
    mapping = {
        "saine":    {"color": theme["ok"],   "label_fr": "Saine",    "label_en": "Healthy"},
        "légère":   {"color": theme["sage"], "label_fr": "Légère",   "label_en": "Mild"},
        "modérée":  {"color": theme["warn"], "label_fr": "Modérée",  "label_en": "Moderate"},
        "critique": {"color": theme["err"],  "label_fr": "Critique", "label_en": "Critical"},
    }
    return mapping.get(severity, {"color": theme["text_muted"], "label_fr": severity, "label_en": severity})

def display_model_bar(model_key: str, lang: str):
    info = MODEL_CATALOG[model_key]
    perf_label = "Performance terrain" if lang == "fr" else "Field performance"
    st.markdown(
        f"""
        <div class="ax-model-bar">
          <div class="left">
            <span class="dot"></span>
            <span class="name">MODEL {info['short'].upper()}</span>
            <span class="tag">{info['tag']}</span>
          </div>
          <div class="perf">{perf_label} <strong>{info['accuracy_terrain']:.2f}%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def display_confidence(conf: float, theme: dict, lang: str):
    state = confidence_state(conf, theme)
    label = state[f"label_{lang}"]
    pct = conf * 100
    width = max(2, min(100, pct))
    cf_label = "Confiance" if lang == "fr" else "Confidence"
    st.markdown(
        f"""
        <div class="ax-confidence">
          <div class="ax-confidence-row">
            <div class="ax-confidence-label">
              <span class="ax-confidence-pill" style="background:{state['color']};"></span>
              <span>{label}</span>
            </div>
            <div class="ax-confidence-value">{pct:.2f}%</div>
          </div>
          <div class="ax-confidence-bar">
            <div class="ax-confidence-fill" style="width:{width}%; background:{state['color']};"></div>
          </div>
          <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.5rem; font-family:'JetBrains Mono', monospace;">
            {cf_label.upper()} · TOP-1
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def display_low_confidence(theme: dict, lang: str):
    if lang == "fr":
        body = """
        <strong>Le modèle n'a pas suffisamment de certitude pour proposer un diagnostic fiable.</strong><br><br>
        Cela peut provenir d'une image floue, d'un cadrage inadapté, d'une plante non couverte par le modèle, ou de conditions d'éclairage atypiques.
        <ul>
          <li>Reprenez la photo en lumière naturelle, feuille bien visible</li>
          <li>Évitez les arrière-plans encombrés (terre, mains, autres feuilles)</li>
          <li>Consultez un expert agricole ou un agronome local pour confirmation</li>
        </ul>
        """
    else:
        body = """
        <strong>The model lacks sufficient confidence to provide a reliable diagnosis.</strong><br><br>
        This may stem from a blurry image, poor framing, a plant outside the model's scope, or unusual lighting conditions.
        <ul>
          <li>Retake the photo in natural light with the leaf clearly visible</li>
          <li>Avoid cluttered backgrounds (soil, hands, other leaves)</li>
          <li>Consult an agricultural expert or local agronomist for confirmation</li>
        </ul>
        """
    st.markdown(f'<div class="ax-warning">{body}</div>', unsafe_allow_html=True)

def display_disease_card(card: dict, theme: dict, lang: str):
    suffix = f"_{lang}"
    plant = card[f"plant{suffix}"]
    disease = card[f"disease{suffix}"]
    sev = severity_state(card["severity"], theme)
    sev_label = sev[f"label_{lang}"]
    sev_color = sev["color"]
    sev_word = "SÉVÉRITÉ" if lang == "fr" else "SEVERITY"

    st.markdown(
        f"""
        <div class="ax-diagnosis" style="border-left-color:{sev_color};">
          <div class="plant">{plant}</div>
          <div class="disease">{disease}</div>
          <span class="severity" style="background:{sev_color}1A; color:{sev_color};">
            {sev_word} · {sev_label}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if card["is_healthy"]:
        ok_msg = "Aucune maladie détectée. Maintenez les bonnes pratiques culturales." if lang == "fr" \
                 else "No disease detected. Keep up the good cultural practices."
        st.markdown(
            f"""
            <div class="ax-card" style="border-left: 3px solid {theme['ok']};">
              <div class="ax-section-label">{'État' if lang == 'fr' else 'Status'}</div>
              <div style="font-size:0.9375rem;">{ok_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        prev_label = "Conseils de prévention" if lang == "fr" else "Prevention tips"
        with st.expander(prev_label):
            for tip in card[f"prevention{suffix}"]:
                st.markdown(f"- {tip}")
        return

    tab_labels = ["SYMPTÔMES", "TRAITEMENT", "PRÉVENTION"] if lang == "fr" \
                 else ["SYMPTOMS", "TREATMENT", "PREVENTION"]
    tab1, tab2, tab3 = st.tabs(tab_labels)
    with tab1:
        for s in card[f"symptoms{suffix}"]:
            st.markdown(f"- {s}")
    with tab2:
        for t in card[f"treatment{suffix}"]:
            st.markdown(f"- {t}")
    with tab3:
        for p in card[f"prevention{suffix}"]:
            st.markdown(f"- {p}")

# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="AnanthiX AI", page_icon="·", layout="wide")

# Vérifications fichiers (silencieux si OK)
for path, label in [
    (MODELS_DIR / MODEL_CATALOG["v2"]["file"], "Model V2"),
    (MODELS_DIR / MODEL_CATALOG["baseline"]["file"], "Baseline"),
    (CLASS_NAMES_PATH, "class_names.json"),
    (DISEASE_CARDS_PATH, "disease_cards.json"),
]:
    if not path.exists():
        st.error(f"File not found: {label} — {path}")
        st.stop()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 1rem 0;">
          <div style="font-family:'Inter',sans-serif; font-size:1.125rem; font-weight:700; letter-spacing:-0.01em;">
            AnanthiX AI
          </div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:0.6875rem; color:var(--text-muted); letter-spacing:0.08em; margin-top:0.15rem;">
            PLANT DISEASE DIAGNOSIS · v2.0
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="ax-section-label">Theme</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme",
        options=["light", "dark"],
        format_func=lambda x: "Light" if x == "light" else "Dark",
        label_visibility="collapsed",
        horizontal=True,
    )
    theme = THEMES[theme_choice]

    st.markdown('<div class="ax-section-label" style="margin-top:1rem;">Language</div>', unsafe_allow_html=True)
    lang = st.radio(
        "Language",
        ["fr", "en"],
        format_func=lambda x: "Français" if x == "fr" else "English",
        label_visibility="collapsed",
        horizontal=True,
    )

    st.markdown('<div class="ax-section-label" style="margin-top:1rem;">Model</div>', unsafe_allow_html=True)
    model_key = st.radio(
        "Model",
        options=["v2", "baseline"],
        format_func=lambda k: MODEL_CATALOG[k]["label"],
        index=0,
        label_visibility="collapsed",
    )

    with st.expander("Performance comparison"):
        st.markdown(
            """
| Model    | Lab        | Field      |
|----------|-----------:|-----------:|
| Baseline | 99.72%     | 26.98%     |
| **V2**   | **99.77%** | **69.84%** |

Field gain: **+42.86 pts**
            """
        )

    st.markdown('<div class="ax-section-label" style="margin-top:1rem;">Explainability</div>', unsafe_allow_html=True)
    show_gradcam = st.checkbox(
        "Grad-CAM heatmap",
        value=True,
        help="Visualise les zones utilisées par le modèle pour décider" if lang == "fr"
             else "Visualize the regions used by the model to decide",
    )

# Injecter le CSS APRÈS avoir choisi le thème
inject_css(theme)

# ============================================================
# CHARGEMENT
# ============================================================
with st.spinner(f"Loading model {MODEL_CATALOG[model_key]['short']}..."):
    model = load_model(model_key)
    class_names = load_class_names()
    disease_cards = load_disease_cards()
    cam = get_gradcam(model, model_key)

# ============================================================
# CONTENU PRINCIPAL
# ============================================================
# En-tête épuré
st.markdown(
    f"""
    <div style="margin-bottom:1.5rem;">
      <h1 style="margin:0;">AnanthiX AI</h1>
      <div style="color:var(--text-muted); font-size:0.9375rem; margin-top:0.25rem;">
        {'Diagnostic visuel des maladies des plantes' if lang == 'fr' else 'Visual diagnosis of plant diseases'}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Barre du modèle actif
display_model_bar(model_key, lang)

# Zone d'upload
upload_label = "Téléversez une photo de feuille" if lang == "fr" else "Upload a leaf photo"
uploaded = st.file_uploader(
    upload_label,
    type=["jpg", "jpeg", "png"],
    label_visibility="visible",
)

if uploaded is not None:
    image = Image.open(uploaded)

    spin = "Analyse en cours" if lang == "fr" else "Analyzing"
    with st.spinner(spin):
        tensor, rgb_float = prepare_image(image)
        predictions = predict(tensor, model, class_names, top_k=3)

    top_class, top_conf = predictions[0]

    # Section : visualisation
    st.markdown(
        f'<div class="ax-section-label" style="margin-top:1rem;">'
        f'{"Visualisation" if lang == "fr" else "Visualization"}</div>',
        unsafe_allow_html=True,
    )

    if show_gradcam:
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                image,
                caption="Input" if lang == "en" else "Image d'entrée",
                width="stretch",
            )
        with col2:
            with st.spinner("Grad-CAM"):
                overlay, _ = compute_gradcam(tensor, rgb_float, cam)
            st.image(
                overlay,
                caption="Grad-CAM · attention map",
                width="stretch",
            )

        gradcam_help = "Comment lire la heatmap" if lang == "fr" else "How to read the heatmap"
        with st.expander(gradcam_help):
            if lang == "fr":
                st.markdown(
                    "Les zones rouges et jaunes correspondent aux régions ayant le plus "
                    "influencé la prédiction du modèle. Une attention centrée sur les "
                    "symptômes de la feuille indique un raisonnement pertinent. "
                    "Une attention dispersée sur l'arrière-plan suggère un biais à vérifier."
                )
            else:
                st.markdown(
                    "Red and yellow areas correspond to the regions that most influenced "
                    "the model's prediction. Attention centered on leaf symptoms indicates "
                    "sound reasoning. Attention scattered on the background suggests a bias "
                    "to verify."
                )
    else:
        st.image(image, caption="Input" if lang == "en" else "Image d'entrée", width="stretch")

    # Section : diagnostic
    st.markdown(
        f'<div class="ax-section-label" style="margin-top:1.5rem;">'
        f'{"Diagnostic" if lang == "fr" else "Diagnosis"}</div>',
        unsafe_allow_html=True,
    )

    # Confiance
    display_confidence(top_conf, theme, lang)

    # Top-3
    top3_label = "Top 3 prédictions" if lang == "fr" else "Top 3 predictions"
    with st.expander(top3_label):
        for i, (cls, p) in enumerate(predictions, 1):
            bar_w = max(2, min(100, p * 100))
            st.markdown(
                f"""
                <div style="margin-bottom:0.75rem;">
                  <div style="display:flex; justify-content:space-between; font-size:0.8125rem; margin-bottom:0.3rem;">
                    <span style="font-family:'JetBrains Mono',monospace; color:var(--text);">{i}. {cls}</span>
                    <span style="font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--text);">{p*100:.2f}%</span>
                  </div>
                  <div style="height:3px; background:var(--neutral); border-radius:2px;">
                    <div style="width:{bar_w}%; height:100%; background:var(--accent); border-radius:2px;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Fiche maladie ou avertissement faible confiance
    st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)

    if top_conf < CONF_LOW:
        display_low_confidence(theme, lang)
    elif top_class in disease_cards:
        display_disease_card(disease_cards[top_class], theme, lang)
    else:
        st.error(f"Missing card for class: {top_class}")
else:
    msg = (
        "Téléversez une image de feuille pour démarrer l'analyse."
        if lang == "fr"
        else "Upload a leaf image to start the analysis."
    )
    st.markdown(f'<div class="ax-empty">{msg}</div>', unsafe_allow_html=True)