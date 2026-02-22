# PrimeurVision — Résultats des modèles

## Dataset

- **238 images** au total (6 classes : carotte, aubergine, citron, pomme_de_terre, radis, tomate)
- Split stratifié 70/15/15 (seed=42) : **166 train / 36 val / 36 test**
- Sources multiples : LVIS, Kaggle, Roboflow, GitHub, images collectées manuellement
- Augmentation manuelle : 1 image aubergine générée par bruit gaussien (σ=12) pour équilibrer les classes

| Classe | Train | Val | Test |
|--------|-------|-----|------|
| carotte | ~28 | ~6 | ~6 |
| aubergine | ~28 | ~6 | ~6 |
| citron | ~28 | ~6 | ~6 |
| pomme_de_terre | ~28 | ~6 | ~6 |
| radis | ~28 | ~6 | ~6 |
| tomate | ~28 | ~6 | ~6 |

---

## Modèle v1 — YOLOv8n (baseline)

**Configuration :**
- Modèle : YOLOv8n pré-entraîné COCO (3M paramètres)
- Phase 1 : 10 epochs, LR=1e-2, backbone gelé (10 couches), patience=10
- Phase 2 : 40 epochs, LR=1e-3, fine-tuning complet, patience=15
- Batch=16, imgsz=640, conf_threshold=0.25
- Augmentation : défauts YOLO uniquement (mosaic, fliplr=0.5, scale=0.5, translate=0.1)

### Résultats sur la validation (phase 2)

| Métrique | Score |
|----------|-------|
| mAP@50 | 0.4935 |
| mAP@50-95 | 0.3241 |
| Précision | 0.5331 |
| Recall | 0.4619 |

| Classe | AP@50 val |
|--------|-----------|
| carotte | 0.2479 |
| aubergine | 0.1192 |
| citron | 0.8462 |
| pomme_de_terre | 0.8537 |
| radis | 0.2134 |
| tomate | 0.6805 |

### Résultats sur le jeu de test (métriques officielles)

| Métrique | Score |
|----------|-------|
| **mAP@50** | **0.4552** |
| mAP@50-95 | 0.3112 |
| Précision | 0.5015 |
| Recall | 0.4313 |

| Classe | AP@50 test |
|--------|------------|
| carotte | 0.3301 |
| aubergine | 0.4823 |
| citron | 0.5321 |
| pomme_de_terre | 0.6951 |
| radis | 0.2806 |
| tomate | 0.4109 |

### Analyse de la matrice de confusion (test)

Principaux enseignements :

- **Pomme_de_terre** : meilleure détection (~83% recall) — visuellement distinctive
- **Citron** : bon recall (~61%) — forme et couleur caractéristiques
- **Tomate** : recall modéré (~59%) mais 65 instances manquées (fond sombre confondu avec background)
- **Carotte** : recall faible (~33%) + **124 faux positifs** (le modèle "voit" des carottes là où il n'y en a pas)
- **Radis** : recall très faible (~18%) — 45 instances sur ~55 ratées, petit objet peu distinctif
- **Aubergine** : recall quasi nul (~9%) — 52 instances sur ~57 ratées, visuellement complexe

**Problème principal :** la ligne `background` de la matrice est très chargée → le modèle rate beaucoup d'objets (faux négatifs). La colonne `background` montre aussi des faux positifs significatifs sur carotte (124) et tomate (23).

**Statistiques de détection (test) :**
- Total détections : 361 (seuil conf ≥ 0.25) — ~10 boîtes/image, beaucoup de bruit à faible confiance
- 1 image sur 36 sans aucune détection
- Confiance médiane de toutes les détections : 0.48 (beaucoup de détections peu sûres)
- Confiance max par image (moyenne) : 0.77 — quand le modèle trouve, sa meilleure détection est assez confiante

### Diagnostic

Le mAP@50 de 0.46 s'explique par :
1. **Dataset trop petit** : ~28 images/classe pour de la détection d'objets est insuffisant
2. **Déséquilibre de difficulté** : citron et pomme_de_terre sont visuellement faciles ; aubergine et radis sont difficiles même pour un humain hors contexte
3. **Qualité des annotations variables** : données issues de sources hétérogènes (LVIS, Kaggle, Roboflow, GitHub)
4. **Modèle trop petit** : YOLOv8n (nano) a une capacité limitée à apprendre 6 classes distinctes

---

## Modèle v2 — YOLOv8s (améliorations)

**Changements par rapport à v1 :**

| Paramètre | v1 (nano) | v2 (small) | Justification |
|-----------|-----------|------------|---------------|
| Modèle | YOLOv8n (3M) | **YOLOv8s (11M)** | Plus de capacité représentationnelle |
| Phase 2 epochs | 40 | **80** | Plus de temps pour converger (early stopping) |
| Patience phase 2 | 15 | **20** | Laisser le modèle sortir des plateaux |
| Rotation | ✗ | **±15°** | Robustesse aux orientations (légumes photographiés sous différents angles) |
| Flip vertical | ✗ | **30%** | Augmentation supplémentaire |
| Mixup | ✗ | **10%** | Régularisation : mélange léger d'images |
| Label smoothing | ✗ | **0.1** | Réduit la sur-confiance sur des annotations potentiellement bruitées |

*(Résultats v2 à compléter après entraînement)*

---

## Pistes d'amélioration futures

- **Plus de données** : doubler le dataset (300 → 600 images) ciblant les classes faibles (aubergine, radis, carotte)
- **Augmentation ciblée** : copy-paste augmentation pour augmenter la variété des contextes
- **YOLOv8m** : passer au modèle "medium" si le dataset augmente
- **Réentraînement avec pseudo-labels** : utiliser le modèle v2 pour annoter de nouvelles images non labellisées
