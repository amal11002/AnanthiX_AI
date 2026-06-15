# AnanthiX AI

Plateforme de diagnostic des maladies des plantes par deep learning. Soumet une image de feuille, obtient un diagnostic instantané avec explicabilité visuelle (zones analysées) et rapport PDF téléchargeable.

## Table des matières

- Prérequis
- Installation
- Lancement de l'application
- Structure du projet
- Utilisation
- API
- Modèles
- Classes supportées
- Notebooks
- Dépendances
- Dépannage

## Prérequis

- Python 3.9 ou supérieur
- pip
- ~2 Go d'espace disque (modèles + dépendances PyTorch)
- CPU suffisant pour l'inférence (pas de GPU requis)

## Installation

```bash
git clone https://github.com/amal11002/AnanthiX_AI
cd AnanthiX_AI
python -m venv venv
```

Activer l'environnement virtuel :

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Lancement de l'application

```bash
cd web
python server.py
```

Au démarrage, le serveur charge les deux modèles (baseline et V2) ainsi que les fiches maladies. Le chargement prend quelques secondes. Une fois prêt, l'application est accessible sur :

```
http://localhost:5000
```

## Structure du projet

```
AnanthiX_AI/
├── app/                          # Application Streamlit (prototype initial, conservé)
│   ├── app_streamlit.py
│   └── disease_cards.json
├── web/                          # Application Flask (principale)
│   ├── server.py                 # Backend Flask + API
│   ├── templates/
│   │   └── index.html            # Interface (layout sidebar)
│   └── static/
│       ├── style.css             # Thème light + motif nervures
│       ├── app.js                # Logique frontend
│       └── leaf-pattern.svg      # Motif d'arrière-plan
├── models/
│   ├── resnet50_baseline.pth     # Modèle entraîné sur PlantVillage seul
│   └── resnet50_v2.pth           # Modèle entraîné sur PlantVillage + PlantDoc x10
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_training.ipynb
│   ├── 04_gradcam.ipynb
│   ├── 05_plantdoc_exploration.ipynb
│   ├── 06_training_v2.ipynb
│   └── 07_comparison_baseline_v2.ipynb
├── results/
│   ├── class_names.json          # Mapping index -> nom de classe
│   ├── plantdoc_mapping.json     # Mapping classes PlantDoc -> PlantVillage
│   ├── comparison_baseline_v2.json # Metriques detaillees baseline vs V2
│   ├── training_history_v2.json  # Historique d'entrainement V2
│   └── comparison_confusion_matrices.png
├── reports/
│   └── Rapport_Final_AnanthiX_AI.docx
├── requirements.txt
└── README.md
```

## Utilisation

1. Ouvrir `http://localhost:5000`
2. Dans la sidebar, choisir la langue (français ou anglais)
3. Choisir le moteur de diagnostic (V2 recommandé pour les conditions terrain, Baseline pour comparaison)
4. Déposer ou sélectionner une image de feuille (PNG ou JPG, max 10 Mo)
5. Le diagnostic s'affiche automatiquement :
   - Maladie détectée et plante concernée
   - Gravité (saine / légère / modérée / critique)
   - Niveau de confiance (Diagnostic fiable / À vérifier / Non concluant), avec le score exact disponible dans le panneau de détails
   - Top 3 des hypothèses (repliable)
   - Fiche maladie complète : symptômes, traitement, prévention
6. Le toggle "Zones analysées" dans la sidebar active ou désactive l'explication visuelle (Grad-CAM). Survoler l'image affiche un aperçu, cliquer sur le bouton fige la superposition
7. Le bouton "Télécharger le rapport" génère un PDF complet du diagnostic courant

## API

L'application expose deux endpoints principaux.

### POST /api/predict

Reçoit une image et retourne le diagnostic.

**Paramètres (form-data) :**

| Champ | Type | Description |
|---|---|---|
| `image` | fichier | Image PNG/JPG, max 10 Mo |
| `model` | string | `v2` ou `baseline` |
| `gradcam` | string | `true` ou `false` |

**Réponse (JSON) :**

```json
{
  "predictions": [
    {"class": "Tomato_Late_blight", "prob": 0.91},
    {"class": "Tomato_Early_blight", "prob": 0.05},
    {"class": "Tomato_healthy", "prob": 0.02}
  ],
  "input_image": "data:image/png;base64,...",
  "gradcam_image": "data:image/png;base64,...",
  "card": { },
  "model_used": "v2"
}
```

### POST /api/report

Génère un rapport PDF à partir d'une réponse de `/api/predict`.

**Corps (JSON) :**

```json
{
  "data": { },
  "lang": "fr"
}
```

`data` correspond à l'objet JSON retourné par `/api/predict`. `lang` accepte `fr` ou `en`.

**Réponse :** fichier PDF (`application/pdf`), téléchargé sous le nom `rapport_ananthix.pdf`.

## Modèles

| Modèle | Données d'entraînement | Accuracy laboratoire | Accuracy terrain | Taille |
|---|---|---:|---:|---:|
| Baseline | PlantVillage uniquement | 99.72% | 26.98% | ~102 Mo |
| V2 | PlantVillage + PlantDoc x10 | 99.77% | 69.84% | ~102 Mo |

Les deux modèles sont des ResNet-50 fine-tunés, architecture identique. Le modèle V2 est recommandé pour un usage réel — il a été spécifiquement amélioré pour généraliser aux conditions terrain (éclairage variable, arrière-plans complexes). Le modèle Baseline est conservé pour comparaison et démonstration de l'impact de la stratégie de données.

Les fichiers `.pth` doivent être placés dans `models/`. Ils ne sont pas inclus dans le dépôt Git standard en raison de leur taille — utiliser Git LFS ou un lien de téléchargement séparé si nécessaire.

## Classes supportées

38 classes de maladies sur 14 espèces végétales :

Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato (10 classes).

La liste complète, avec l'index numérique correspondant à chaque classe, est disponible dans `results/class_names.json`.

## Notebooks

| Notebook | Contenu |
|---|---|
| `01_data_exploration.ipynb` | Exploration initiale de PlantVillage |
| `02_preprocessing.ipynb` | Prétraitement et séparation train/val/test |
| `03_baseline_training.ipynb` | Entraînement du modèle baseline |
| `04_gradcam.ipynb` | Implémentation et validation de Grad-CAM |
| `05_plantdoc_exploration.ipynb` | Exploration et mapping de PlantDoc |
| `06_training_v2.ipynb` | Fine-tuning du modèle V2 (PlantVillage + PlantDoc x10) |
| `07_comparison_baseline_v2.ipynb` | Calcul des métriques comparatives et matrices de confusion |

Les notebooks 03 et 06 sont conçus pour Google Colab avec GPU (type G4 ou équivalent).

## Dépendances principales

```
flask
torch
torchvision
pillow
numpy
pytorch-grad-cam
reportlab
```

Liste complète et versions exactes dans `requirements.txt`.

## Dépannage

**`ModuleNotFoundError: No module named 'reportlab'`**
Installer reportlab dans l'environnement virtuel actif : `pip install reportlab`

**`FileNotFoundError: Missing model`**
Vérifier que `resnet50_baseline.pth` et `resnet50_v2.pth` sont présents dans `models/`. Ces fichiers ne sont pas versionnés directement en raison de leur taille.

**Le serveur met du temps à démarrer**
Normal — les deux modèles ResNet-50 sont chargés en mémoire au démarrage. Le démarrage prend généralement entre 10 et 30 secondes selon le matériel.

**Erreur 404 sur `/api/report`**
Vérifier que l'endpoint `/api/report` est bien présent dans `server.py` et que `reportlab` est installé.
