"""
Grille de revue du dataset - Projet PrimeurVision
Affiche toutes les images acceptées par classe en grille.
Permet de retirer des images du dataset d'un clic.

Lancer avec : streamlit run scripts/review_grid.py
"""

import json
import os
import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import Counter, defaultdict

# --- Configuration ---
PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
STATE_FILE = PROJECT_DIR / "curation_state.json"

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

SOURCE_REMAPS = {
    "selected_images_lemon_tomato": {8: 2, 17: 5},
    "lvis_to_curate": {14: 0, 26: 1, 37: 2, 52: 3, 55: 4, 59: 5},
}


# --- Persistence ---
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"accepted": [], "rejected": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# --- Labels ---
def load_labels(label_path: Path, remap: dict | None = None):
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                if remap:
                    if cls_id not in remap:
                        continue
                    cls_id = remap[cls_id]
                cx, cy, w, h = map(float, parts[1:])
                annotations.append((cls_id, cx, cy, w, h))
    return annotations


def draw_boxes(image: Image.Image, annotations: list, line_width: int = 2) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for cls_id, cx, cy, w, h in annotations:
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        color = CLASS_COLORS.get(cls_id, "#FFFFFF")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    return img


def resolve_image_info(img_key: str) -> tuple[Path | None, Path | None, dict | None]:
    """Retourne (img_path, label_path, remap) pour une clé acceptée."""
    prefix, stem = img_key.split("/", 1)
    if prefix in ("train", "val"):
        img_dir = DATASET_DIR / "images" / prefix
        lbl_dir = DATASET_DIR / "labels" / prefix
        remap = None
    else:
        img_dir = DATASET_DIR / prefix / "images"
        lbl_dir = DATASET_DIR / prefix / "labels"
        remap = SOURCE_REMAPS.get(prefix)

    # Trouver l'image
    for ext in (".jpg", ".jpeg", ".png"):
        img_path = img_dir / (stem + ext)
        if img_path.exists():
            return img_path, lbl_dir / (stem + ".txt"), remap

    # Fallback : chercher dans train/ (les images externes y sont copiées)
    for ext in (".jpg", ".jpeg", ".png"):
        img_path = DATASET_DIR / "images" / "train" / (stem + ext)
        if img_path.exists():
            return img_path, DATASET_DIR / "labels" / "train" / (stem + ".txt"), None

    return None, None, None


# --- Interface ---
st.set_page_config(page_title="Revue par classe", page_icon="🔍", layout="wide")
st.title("Revue du dataset par classe")

if "state" not in st.session_state:
    st.session_state.state = load_state()
state = st.session_state.state

# Indexer les images acceptées par classe (dédupliquées par stem)
seen_stems = set()
images_by_class = defaultdict(list)  # {cls_id: [(img_key, img_path, label_path, remap)]}

for img_key in state["accepted"]:
    _, stem = img_key.split("/", 1)
    if stem in seen_stems:
        continue
    seen_stems.add(stem)

    img_path, label_path, remap = resolve_image_info(img_key)
    if img_path is None:
        continue

    annotations = load_labels(label_path, remap=remap)
    classes = {cls_id for cls_id, *_ in annotations}
    for cls_id in classes:
        if cls_id in CLASS_NAMES:
            images_by_class[cls_id].append((img_key, stem, img_path, label_path, remap))

# Sidebar : choix de classe
with st.sidebar:
    st.header("Classe")
    class_options = []
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = len(images_by_class.get(cls_id, []))
        class_options.append(f"{CLASS_NAMES[cls_id]} ({count})")

    selected_idx = st.radio(
        "Filtrer par classe",
        range(len(class_options)),
        format_func=lambda i: class_options[i],
    )

    st.divider()
    st.header("Progression")
    for cls_id in sorted(CLASS_NAMES.keys()):
        count = len(images_by_class.get(cls_id, []))
        color = CLASS_COLORS[cls_id]
        st.markdown(
            f'<span style="color:{color}; font-weight:bold;">'
            f"{CLASS_NAMES[cls_id]}</span> : {count}/50",
            unsafe_allow_html=True,
        )
        st.progress(min(count / 50, 1.0))

    total = len(seen_stems)
    st.divider()
    st.metric("Total images uniques", total)

# Zone principale : grille
cls_id = selected_idx
cls_name = CLASS_NAMES[cls_id]
cls_images = images_by_class.get(cls_id, [])

st.subheader(f"{cls_name} — {len(cls_images)} images")

if not cls_images:
    st.warning("Aucune image acceptée pour cette classe.")
    st.stop()

# Grille 4 colonnes
COLS = 4
rows = [cls_images[i : i + COLS] for i in range(0, len(cls_images), COLS)]

# Gestion des suppressions en batch
if "to_remove" not in st.session_state:
    st.session_state.to_remove = set()

for row_idx, row in enumerate(rows):
    cols = st.columns(COLS)
    for col_idx, (img_key, stem, img_path, label_path, remap) in enumerate(row):
        with cols[col_idx]:
            # Charger et dessiner
            img = Image.open(img_path)
            annotations = load_labels(label_path, remap=remap)
            display = draw_boxes(img, annotations)
            st.image(display, use_container_width=True)

            # Info compacte
            marked = stem in st.session_state.to_remove
            btn_label = "↩ Annuler" if marked else "✕ Retirer"
            btn_type = "secondary" if marked else "primary"

            if st.button(btn_label, key=f"btn_{stem}", use_container_width=True, type=btn_type):
                if marked:
                    st.session_state.to_remove.discard(stem)
                else:
                    st.session_state.to_remove.add(stem)
                st.rerun()

            if marked:
                st.markdown(
                    '<div style="text-align:center; color:#DC2626; font-weight:bold;">RETIRE</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption(stem[:20] + "...")

# Bouton de confirmation
st.divider()
n_remove = len(st.session_state.to_remove)
if n_remove > 0:
    st.warning(f"{n_remove} image(s) marquée(s) pour retrait.")
    if st.button(f"Confirmer le retrait de {n_remove} image(s)", type="primary"):
        # Déplacer de accepted vers rejected
        new_accepted = []
        for key in state["accepted"]:
            _, stem = key.split("/", 1)
            if stem in st.session_state.to_remove:
                state["rejected"].append(key)
            else:
                new_accepted.append(key)
        state["accepted"] = new_accepted

        # Supprimer les fichiers physiques de train/
        for stem in st.session_state.to_remove:
            for ext in (".jpg", ".jpeg", ".png"):
                p = DATASET_DIR / "images" / "train" / (stem + ext)
                if p.exists():
                    p.unlink()
            p = DATASET_DIR / "labels" / "train" / (stem + ".txt")
            if p.exists():
                p.unlink()

        save_state(state)
        st.session_state.state = state
        st.session_state.to_remove = set()
        st.rerun()
