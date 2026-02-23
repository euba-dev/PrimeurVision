# 🥕 PrimeurVision

Détection d'objets appliquée aux fruits et légumes par fine-tuning de **YOLOv8** sur un dataset constitué et annoté manuellement.

Projet réalisé dans le cadre du Master 2 SISE — Statistiques et Informatique pour la Science des données.

---

## 🎯 Objectif

Entraîner un modèle de détection capable d'identifier 6 classes de fruits et légumes dans des images en contexte culinaire :

| ID | Classe |
|----|--------|
| 0 | Carotte |
| 1 | Aubergine |
| 2 | Citron |
| 3 | Pomme de terre |
| 4 | Radis |
| 5 | Tomate |

---

## 🗃️ Dataset

**238 images** constituées à partir de sources hétérogènes, remappées et curées manuellement.

| Source | Description |
|--------|-------------|
| LVIS | Dataset open-source (classes ciblées extraites) |
| Kaggle / Roboflow | Datasets publics de fruits et légumes |
| GitHub | Datasets annotés YOLO disponibles publiquement |
| Photos personnelles | Images collectées manuellement |

### Distribution par classe

| Classe | Total | Train | Val | Test |
|--------|-------|-------|-----|------|
| carotte | 51 | ~36 | ~8 | ~7 |
| aubergine | 50* | ~35 | ~8 | ~7 |
| citron | 50 | ~35 | ~8 | ~7 |
| pomme_de_terre | 51 | ~36 | ~8 | ~7 |
| radis | 51 | ~36 | ~8 | ~7 |
| tomate | 52 | ~37 | ~8 | ~7 |
| **Total** | **238** | **166** | **36** | **36** |

*dont 1 image générée par augmentation (bruit gaussien)

**Split stratifié 70/15/15** (seed=42) — chaque classe est représentée proportionnellement dans les 3 ensembles.

### Outil de curation — Streamlit

Un outil interactif de revue du dataset a été développé (`scripts/review_grid.py`). Il affiche les images par classe avec leurs bounding boxes annotées et permet de supprimer une image d'un clic.

```bash
streamlit run scripts/review_grid.py
```

---

## 🏋️ Entraînement

Fine-tuning de YOLOv8 pré-entraîné sur COCO en **2 phases** :

1. **Phase 1 — Backbone gelé** : seule la tête de détection apprend (10 epochs, LR=1e-2). Permet d'adapter rapidement le modèle sans perturber les features génériques.
2. **Phase 2 — Fine-tuning complet** : toutes les couches sont libérées (40 epochs, LR=1e-3). Affinage fin sur notre domaine.

### Comparaison des versions

| | v1 — YOLOv8n ✅ | v2 — YOLOv8s |
|---|---|---|
| Paramètres | 3M | 11M |
| Epochs phase 2 | 40 | 80 |
| Augmentation | Défauts YOLO | + rotation, flip, mixup |
| **mAP@50 (test)** | **0.455** | 0.355 |

> Le modèle v2 (plus grand) s'est révélé moins performant sur le test malgré de meilleures métriques de validation — signe d'overfitting sur un dataset de 166 images. **Le v1 (YOLOv8n) est retenu comme modèle final.**

---

## 📊 Résultats (modèle final v1 — jeu de test)

| Métrique | Score |
|----------|-------|
| mAP@50 | **0.455** |
| mAP@50-95 | 0.311 |
| Précision | 0.502 |
| Recall | 0.431 |

| Classe | AP@50 |
|--------|-------|
| pomme_de_terre | 0.695 |
| citron | 0.532 |
| aubergine | 0.482 |
| tomate | 0.411 |
| carotte | 0.330 |
| radis | 0.281 |

---

## 📁 Structure du projet

```
PrimeurVision/
├── dataset/
│   ├── images/
│   │   ├── train/       # 166 images
│   │   ├── val/         # 36 images
│   │   └── test/        # 36 images
│   ├── labels/          # Annotations YOLO (class cx cy w h)
│   └── data.yaml        # Configuration YOLOv8
├── models/
│   ├── best_yolov8n_primeurvision.pt   # Modèle final (v1)
│   ├── results.png                      # Courbes d'entraînement v1
│   ├── confusion_matrix_normalized.png  # Matrice de confusion v1
│   ├── v2_results.png                   # Courbes v2 (comparaison)
│   └── v2_confusion_matrix.png          # Matrice v2 (comparaison)
├── notebooks/
│   ├── train_yolov8.ipynb     # Entraînement (config v2 documentée)
│   └── evaluate_yolov8.ipynb  # Évaluation sur le jeu de test (modèle v1)
├── scripts/
│   └── review_grid.py         # Interface Streamlit de curation
└── docs/
    └── resultats.md            # Analyse complète des résultats
```

---

## ⚙️ Installation

```bash
# Créer un environnement conda (Python 3.11 recommandé)
conda create -n primeurvision python=3.11
conda activate primeurvision

# Installer les dépendances
pip install torch torchvision torchaudio
pip install ultralytics
pip install jupyter streamlit
```

> **Note Apple Silicon (M1/M2/M3)** : MPS n'est pas compatible avec le calcul de loss de YOLO lors de l'entraînement. Les notebooks utilisent `device='cpu'` par défaut.

---

## 🚀 Utilisation

### Entraînement

Ouvrir `notebooks/train_yolov8.ipynb` avec le kernel `primeurvision` et exécuter les cellules. Fonctionne en local et sur Google Colab.

### Évaluation

Ouvrir `notebooks/evaluate_yolov8.ipynb` — évalue le modèle v1 sur le jeu de test et produit métriques, matrice de confusion et analyses qualitatives.

### Curation du dataset

```bash
streamlit run scripts/review_grid.py
```

---

## 👥 Auteurs

Eugénie Barlet & Perrine Ibouroi
