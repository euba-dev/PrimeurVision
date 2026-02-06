"""
Outil de curation du dataset YOLO - Projet Computer Vision
Perrine IBOUROI & Eugénie BARLET - M2 SISE

Parcourir les images, accepter ou rejeter chacune.
Objectif : 50 images acceptées par classe, 300 images max au total.
L'état est sauvegardé dans curation_state.json (persistent entre les sessions).

Lancer avec : streamlit run validate_dataset.py
"""

import json
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

# --- Configuration ---
DATASET_DIR = Path(__file__).parent.parent / "dataset"
STATE_FILE = Path(__file__).parent.parent / "curation_state.json"
MAX_PER_CLASS = 50
MAX_TOTAL = 300

CLASS_NAMES = {
    0: "carotte",
    1: "aubergine",
    2: "citron",
    3: "pomme_de_terre",
    4: "radis",
    5: "tomate",
}
CLASS_COLORS = {
    0: "#FF6B35",
    1: "#7B2D8E",
    2: "#FFD700",
    3: "#C4A35A",
    4: "#E84855",
    5: "#DC2626",
}


# --- Persistence ---
def load_state() -> dict:
    """Charge l'état de curation depuis le fichier JSON."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"accepted": [], "rejected": []}


def save_state(state: dict):
    """Sauvegarde l'état de curation."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# --- Labels ---
def load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Charge les annotations YOLO depuis un fichier .txt."""
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                annotations.append((cls_id, cx, cy, w, h))
    return annotations


def get_classes_in_image(img_stem: str, labels_dir: Path) -> set[int]:
    """Retourne les classes présentes dans une image."""
    label_path = labels_dir / (img_stem + ".txt")
    annotations = load_labels(label_path)
    return {cls_id for cls_id, *_ in annotations}


def draw_boxes(image: Image.Image, annotations: list, line_width: int = 3) -> Image.Image:
    """Dessine les bounding boxes YOLO sur l'image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for cls_id, cx, cy, w, h in annotations:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        color = CLASS_COLORS.get(cls_id, "#FFFFFF")
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        label = f"{name}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_h - 3), label, fill="white", font=font)

    return img


# --- Comptage ---
def count_accepted_per_class(state: dict, all_labels_dirs: list[Path]) -> Counter:
    """Compte les images acceptées par classe."""
    counts = Counter()
    for img_key in state["accepted"]:
        # img_key = "train/000000001234" ou "val/000000001234"
        split, stem = img_key.split("/", 1)
        for labels_dir in all_labels_dirs:
            if labels_dir.parent.name == split or split in str(labels_dir):
                classes = get_classes_in_image(stem, labels_dir)
                for cls_id in classes:
                    counts[cls_id] += 1
                break
        else:
            # Chercher dans tous les labels_dirs
            for labels_dir in all_labels_dirs:
                label_path = labels_dir / (stem + ".txt")
                if label_path.exists():
                    classes = get_classes_in_image(stem, labels_dir)
                    for cls_id in classes:
                        counts[cls_id] += 1
                    break
    return counts


def build_image_list(all_labels_dirs: list[Path]) -> list[tuple[str, Path, Path]]:
    """Construit la liste complète des images avec leur clé unique."""
    images = []
    for labels_dir in all_labels_dirs:
        split = labels_dir.name  # "train" ou "val"
        images_dir = DATASET_DIR / "images" / split
        for img_path in sorted(images_dir.glob("*.jpg")):
            key = f"{split}/{img_path.stem}"
            images.append((key, img_path, labels_dir))
        for img_path in sorted(images_dir.glob("*.png")):
            key = f"{split}/{img_path.stem}"
            images.append((key, img_path, labels_dir))
    return images


def should_skip(img_key: str, labels_dir: Path, state: dict, class_counts: Counter) -> bool:
    """Vérifie si une image doit être sautée (toutes ses classes sont complètes)."""
    split, stem = img_key.split("/", 1)
    classes = get_classes_in_image(stem, labels_dir)
    if not classes:
        return True
    # Sauter si TOUTES les classes de l'image ont déjà atteint 50
    return all(class_counts.get(cls_id, 0) >= MAX_PER_CLASS for cls_id in classes)


# --- Interface Streamlit ---
st.set_page_config(page_title="Curation Dataset YOLO", page_icon="🎯", layout="wide")

# Charger l'état
if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

# Répertoires labels
all_labels_dirs = [
    DATASET_DIR / "labels" / "train",
    DATASET_DIR / "labels" / "val",
]

# Toutes les images
all_images = build_image_list(all_labels_dirs)

# Comptage actuel
class_counts = count_accepted_per_class(state, all_labels_dirs)
total_accepted = len(state["accepted"])

# --- Vérifier si objectif atteint ---
all_classes_full = all(class_counts.get(i, 0) >= MAX_PER_CLASS for i in CLASS_NAMES)

if total_accepted >= MAX_TOTAL or all_classes_full:
    st.title("Dataset complet !")
    st.balloons()
    st.success(f"Objectif atteint : {total_accepted} images acceptées.")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = class_counts.get(cls_id, 0)
        st.write(f"**{CLASS_NAMES[cls_id]}** : {count}/{MAX_PER_CLASS}")
    if st.button("Exporter le dataset final"):
        st.json(state)
    if st.button("Remettre a zero la curation"):
        st.session_state.state = {"accepted": [], "rejected": []}
        save_state(st.session_state.state)
        st.rerun()
    st.stop()

# --- Filtrer les images à traiter ---
pending_images = []
for key, img_path, labels_dir in all_images:
    if key in state["accepted"] or key in state["rejected"]:
        continue
    if should_skip(key, labels_dir, state, class_counts):
        continue
    pending_images.append((key, img_path, labels_dir))

if not pending_images:
    st.title("Plus d'images a traiter")
    st.warning(
        "Toutes les images disponibles ont été traitées mais l'objectif de 300 "
        "n'est pas atteint. Il faut ajouter des images sources supplementaires."
    )
    st.stop()

# Index courant
if "cursor" not in st.session_state:
    st.session_state.cursor = 0
cursor = min(st.session_state.cursor, len(pending_images) - 1)

img_key, img_path, labels_dir = pending_images[cursor]
split, stem = img_key.split("/", 1)
label_path = labels_dir / (stem + ".txt")
annotations = load_labels(label_path)
classes_in_img = {cls_id for cls_id, *_ in annotations}

# === SIDEBAR ===
with st.sidebar:
    st.header("Progression")

    # Barre globale
    pct_total = min(total_accepted / MAX_TOTAL, 1.0)
    st.metric("Images acceptées", f"{total_accepted} / {MAX_TOTAL}")
    st.progress(pct_total)

    st.divider()

    # Par classe
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = class_counts.get(cls_id, 0)
        color = CLASS_COLORS[cls_id]
        full = count >= MAX_PER_CLASS
        marker = " ✅" if full else ""
        st.markdown(
            f'<span style="color:{color}; font-weight:bold;">'
            f"{CLASS_NAMES[cls_id]}</span> : {count}/{MAX_PER_CLASS}{marker}",
            unsafe_allow_html=True,
        )
        st.progress(min(count / MAX_PER_CLASS, 1.0))

    st.divider()
    st.metric("Rejetées", len(state["rejected"]))
    st.metric("Restantes a traiter", len(pending_images))

    st.divider()
    if st.button("Remettre a zero", type="secondary"):
        st.session_state.state = {"accepted": [], "rejected": []}
        st.session_state.cursor = 0
        save_state(st.session_state.state)
        st.rerun()

# === ZONE PRINCIPALE ===
st.title("Curation du Dataset")

# Info image
st.markdown(
    f"**Image {cursor + 1} / {len(pending_images)} restantes** — "
    f"`{split}/{img_path.name}` — "
    f"**{len(annotations)} annotation(s)**"
)

# Classes présentes avec badge
badges = ""
for cls_id in sorted(classes_in_img):
    color = CLASS_COLORS.get(cls_id, "#666")
    name = CLASS_NAMES.get(cls_id, "?")
    count = class_counts.get(cls_id, 0)
    badges += (
        f'<span style="background-color:{color}; color:white; '
        f'padding:2px 10px; border-radius:12px; margin-right:6px; '
        f'font-weight:bold;">{name} ({count}/{MAX_PER_CLASS})</span>'
    )
st.markdown(badges, unsafe_allow_html=True)
st.write("")

# Image avec bounding boxes
col_img, col_details = st.columns([3, 1])

with col_img:
    image = Image.open(img_path)
    display_img = draw_boxes(image, annotations)
    st.image(display_img, use_container_width=True)

with col_details:
    st.subheader("Annotations")
    if annotations:
        class_summary = Counter(cls_id for cls_id, *_ in annotations)
        for cls_id, count in sorted(class_summary.items()):
            color = CLASS_COLORS.get(cls_id, "#FFF")
            st.markdown(
                f'<span style="color:{color}; font-weight:bold;">'
                f"{CLASS_NAMES.get(cls_id, '?')}</span> : {count} boite(s)",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Aucune annotation")

    with st.expander("Label brut"):
        if label_path.exists():
            st.code(label_path.read_text(), language="text")

# === BOUTONS ACCEPT / REJECT ===
st.write("")
col_reject, col_skip, col_accept = st.columns([1, 1, 1])

with col_accept:
    if st.button("✅ Accepter", use_container_width=True, type="primary"):
        state["accepted"].append(img_key)
        save_state(state)
        st.session_state.cursor = cursor  # reste au même index (la liste se décale)
        st.rerun()

with col_reject:
    if st.button("❌ Rejeter", use_container_width=True):
        state["rejected"].append(img_key)
        save_state(state)
        st.session_state.cursor = cursor
        st.rerun()

with col_skip:
    if st.button("⏭ Passer", use_container_width=True):
        st.session_state.cursor = min(cursor + 1, len(pending_images) - 1)
        st.rerun()
