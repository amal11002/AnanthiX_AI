# AnanthiX AI

**Diagnostic intelligent des maladies des plantes via Deep Learning**

##  Résultats Jalon 2

- **Model**: ResNet-50 (pré-entraîné ImageNet + fine-tuning)
- **Dataset**: PlantVillage (54,303 images, 38 classes)
- **Test Accuracy**: 99.43%
- **F1 Macro**: 0.9908
- **Epochs**: 10

##  Structure
## Objectif

Créer une solution IA pour diagnostiquer rapidement les maladies des plantes 
destinée aux petits exploitants agricoles en Afrique de l'Ouest.

##  Jalons

- **Jalon 1** (12 mai) : Cadrage + Maquette ✓
- **Jalon 2** (20 mai) : Pipeline + Baseline ✓
- **Jalon 3** (26 mai) : Prototype Streamlit + Grad-CAM
- **Jalon 4** (5 juin) : Prototype bêta
- **Jalon 5** (9 juin) : Version finale

##  Usage

### Exécuter les notebooks

Tous les notebooks sont autonomes et exécutables sur Google Colab :

1. `01_exploration_plantvillage.ipynb` : Explorer le dataset
2. `02_preprocessing_training.ipynb` : Entraîner le modèle
3. `03_evaluation.ipynb` : Évaluer les résultats

### Charger le modèle

```python
import torch
import torchvision.models as models

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(2048, 38)
model.load_state_dict(torch.load('models/resnet50_baseline.pth'))
model.eval()
```

##  Performance par classe

**Top 5 classes avec meilleure F1** :
- Apple_healthy : 1.0000
- Blueberry_healthy : 1.0000
- Orange_Haunglongbing : 0.9977

**Classes nécessitant amélioration** :
- Corn_Cercospora : 0.9054
- Corn_Northern_Leaf_Blight : 0.9577
- Tomato_Early_blight : 0.9585

##  Limitations

1. Dataset d'entraînement = lab contrôlé (fonds uniformes)
2. Performance sur terrain réel probablement 60-75% (vs 99% lab)
3. Limité à 38 classes (pas universel)
4. Pas un remplacement d'expert agronomique

##  Références

- Dataset: PlantVillage (Hughes & Salathé, 2015)
- Model: ResNet-50 (He et al., 2015)
- Framework: PyTorch

##  Auteur

Amal Ouedraogo | UQAC | Atelier pratique IA

##  Dates clés

- 12 mai : Jalon 1 ✓
- 20 mai : Jalon 2 ✓
- 26 mai : Jalon 3
- 16 juin : Présentation finale
