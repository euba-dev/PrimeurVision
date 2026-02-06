#!/usr/bin/env python3
"""
Script pour extraire un sous-ensemble du dataset LVIS pour le projet Computer Vision.
6 classes : Carotte, Tomate, Aubergine, Citron, Radis, Pomme de terre
50 images par classe (ou le maximum disponible)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
SOURCE_DIR = Path.home() / "Downloads" / "LVIS_Fruits_And_Vegetables"
DEST_DIR = Path(__file__).parent.parent / "dataset"

# Mapping des classes source vers les nouvelles classes
# Format: {id_source: (id_dest, nom_classe)}
CLASS_MAPPING = {
    14: (0, "carotte"),
    26: (1, "aubergine"),
    37: (2, "citron"),
    52: (3, "pomme_de_terre"),
    55: (4, "radis"),
    59: (5, "tomate"),
}

IMAGES_PER_CLASS = 50
TRAIN_RATIO = 0.8  # 80% train, 20% val

def find_images_per_class(labels_dir: Path) -> dict:
    """Trouve toutes les images contenant chaque classe cible."""
    images_by_class = defaultdict(list)

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Trouver les classes présentes dans cette image
        classes_in_image = set()
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                if class_id in CLASS_MAPPING:
                    classes_in_image.add(class_id)

        # Ajouter l'image à chaque classe trouvée
        for class_id in classes_in_image:
            images_by_class[class_id].append(label_file.stem)

    return images_by_class

def create_new_label(source_label: Path, dest_label: Path):
    """Crée un nouveau fichier label avec uniquement les classes cibles et les IDs remappés."""
    with open(source_label, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:
            class_id = int(parts[0])
            if class_id in CLASS_MAPPING:
                new_class_id = CLASS_MAPPING[class_id][0]
                new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                new_lines.append(new_line)

    with open(dest_label, 'w') as f:
        f.writelines(new_lines)

def main():
    print("=" * 60)
    print("Creation du dataset pour le projet Computer Vision")
    print("=" * 60)

    # Chemins source
    source_images_train = SOURCE_DIR / "images" / "train" / "train"
    source_labels_train = SOURCE_DIR / "labels" / "train" / "train"

    # Vérification des chemins
    if not source_images_train.exists():
        print(f"Erreur: Dossier source introuvable: {source_images_train}")
        return

    # Trouver les images par classe
    print("\nRecherche des images par classe...")
    images_by_class = find_images_per_class(source_labels_train)

    print("\nImages disponibles par classe:")
    for class_id, (new_id, name) in CLASS_MAPPING.items():
        count = len(images_by_class[class_id])
        print(f"  {name}: {count} images")

    # Sélectionner les images pour chaque classe
    selected_images = {}
    for class_id, (new_id, name) in CLASS_MAPPING.items():
        available = images_by_class[class_id]
        n_select = min(IMAGES_PER_CLASS, len(available))
        random.seed(42)  # Pour reproductibilité
        selected = random.sample(available, n_select)
        selected_images[class_id] = selected
        print(f"\n{name}: {n_select} images selectionnees")

    # Créer la structure du dataset de destination
    dest_train_images = DEST_DIR / "images" / "train"
    dest_val_images = DEST_DIR / "images" / "val"
    dest_train_labels = DEST_DIR / "labels" / "train"
    dest_val_labels = DEST_DIR / "labels" / "val"

    for d in [dest_train_images, dest_val_images, dest_train_labels, dest_val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Copier les images et créer les labels
    print("\n" + "=" * 60)
    print("Copie des images et creation des labels...")
    print("=" * 60)

    total_train = 0
    total_val = 0

    for class_id, images in selected_images.items():
        name = CLASS_MAPPING[class_id][1]
        n_train = int(len(images) * TRAIN_RATIO)

        random.seed(42)
        random.shuffle(images)

        train_images = images[:n_train]
        val_images = images[n_train:]

        # Copier les images de train
        for img_name in train_images:
            # Trouver l'extension de l'image
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                src_img = source_images_train / f"{img_name}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, dest_train_images / f"{img_name}{ext}")
                    break

            # Créer le label
            src_label = source_labels_train / f"{img_name}.txt"
            if src_label.exists():
                create_new_label(src_label, dest_train_labels / f"{img_name}.txt")

        # Copier les images de val
        for img_name in val_images:
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                src_img = source_images_train / f"{img_name}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, dest_val_images / f"{img_name}{ext}")
                    break

            src_label = source_labels_train / f"{img_name}.txt"
            if src_label.exists():
                create_new_label(src_label, dest_val_labels / f"{img_name}.txt")

        total_train += len(train_images)
        total_val += len(val_images)
        print(f"  {name}: {len(train_images)} train, {len(val_images)} val")

    # Créer le fichier data.yaml
    data_yaml_content = f"""# Dataset Fruits et Legumes - Projet Computer Vision
# Cree automatiquement par create_dataset.py

path: {DEST_DIR}
train: images/train
val: images/val

names:
  0: carotte
  1: aubergine
  2: citron
  3: pomme_de_terre
  4: radis
  5: tomate

# Statistiques
# Train: {total_train} images
# Val: {total_val} images
# Total: {total_train + total_val} images
"""

    with open(DEST_DIR / "data.yaml", 'w') as f:
        f.write(data_yaml_content)

    print("\n" + "=" * 60)
    print("Dataset cree avec succes!")
    print("=" * 60)
    print(f"\nEmplacement: {DEST_DIR}")
    print(f"Train: {total_train} images")
    print(f"Val: {total_val} images")
    print(f"Total: {total_train + total_val} images")
    print(f"\nFichier de configuration: {DEST_DIR / 'data.yaml'}")

if __name__ == "__main__":
    main()
