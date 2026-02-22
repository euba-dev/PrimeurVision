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

### Résultats sur la validation (phase 2)

| Métrique | Score |
|----------|-------|
| mAP@50 | 0.5650 |
| mAP@50-95 | 0.3652 |
| Précision | 0.6142 |
| Recall | 0.5265 |

| Classe | AP@50 val |
|--------|-----------|
| carotte | 0.2864 |
| aubergine | 0.2551 |
| citron | 0.8854 |
| pomme_de_terre | 0.9053 |
| radis | 0.2711 |
| tomate | 0.7863 |

### Résultats sur le jeu de test (métriques officielles)

| Métrique | Score |
|----------|-------|
| **mAP@50** | **0.3546** |
| mAP@50-95 | 0.1982 |
| Précision | 0.3307 |
| Recall | 0.4579 |

| Classe | AP@50 test |
|--------|------------|
| carotte | 0.2381 |
| aubergine | 0.3198 |
| citron | 0.3283 |
| pomme_de_terre | 0.6908 |
| radis | 0.1029 |
| tomate | 0.4476 |

### Analyse comparative v1 vs v2

| Métrique | v1 (nano) test | v2 (small) test | Évolution |
|----------|---------------|-----------------|-----------|
| mAP@50 | 0.4552 | 0.3546 | **-11 pts** |
| mAP@50-95 | 0.3112 | 0.1982 | -11 pts |
| Précision | 0.5015 | 0.3307 | -17 pts |
| Recall | 0.4313 | 0.4579 | +3 pts |

**Constat : le v2 est moins bon sur le test malgré de meilleures métriques de validation (0.49→0.57).**

### Diagnostic de l'échec du v2 : overfitting

L'écart val/test du v2 est révélateur :
- **Val mAP** : 0.4935 (v1) → 0.5650 (v2) = **+7 pts** ✓
- **Test mAP** : 0.4552 (v1) → 0.3546 (v2) = **-11 pts** ✗

Avec seulement 166 images d'entraînement et 36 images de validation, YOLOv8s (11M paramètres) est trop grand pour ce dataset. Le modèle a sur-appris les patterns spécifiques des 36 images de validation utilisées pour le monitoring (early stopping) et ne généralise pas aussi bien au jeu de test.

La leçon : **la taille du modèle doit être proportionnelle à la taille du dataset**. YOLOv8n (3M paramètres) était mieux adapté à nos 166 images.

**Modèle retenu : v1 (YOLOv8n), mAP@50 = 0.4552 sur le test.**

---

## Bilan et pistes d'amélioration futures

Le v1 reste le meilleur modèle. Pour améliorer significativement les performances, il faudrait :

- **Plus de données** : passer à 600+ images, en ciblant les classes faibles (aubergine, radis, carotte)
- **Rester sur YOLOv8n** (ou tester YOLOv8s seulement avec >400 images)
- **Augmentation ciblée** : copy-paste augmentation pour augmenter la variété des contextes
- **Annotations de meilleure qualité** : homogénéiser les sources du dataset
- **Réentraînement avec pseudo-labels** : utiliser le modèle v1 pour annoter de nouvelles images non labellisées
