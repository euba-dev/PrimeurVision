"""
Script d'augmentation de dataset d'images.

"""

import os
import glob
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = r"c:\Users\perri\Documents\MASTER SISE\Cours\Deep Learning\photos\troisieme_dataset\images"

# {prefixe: (nb_originaux, cible)}
# nb_originaux = images a conserver, cible = total final voulu
CATEGORIES = {
    "pdt":        (14, 33),
    "radis":      (6,  18),
    "aubergines": (4,   8),
}

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Nettoyage des images precedemment generees ───────────────────────────────

def cleanup(prefix, nb_originaux):
    """Supprime toutes les images au-dela des originaux (generees precedemment)."""
    pattern = os.path.join(BASE_DIR, f"{prefix} (*).jpg")
    paths = sorted(glob.glob(pattern))
    removed = 0
    for p in paths:
        # Extrait le numero depuis le nom de fichier
        name = os.path.basename(p)
        try:
            num = int(name.replace(f"{prefix} (", "").replace(").jpg", ""))
            if num > nb_originaux:
                os.remove(p)
                removed += 1
        except ValueError:
            pass
    if removed:
        print(f"  [nettoyage] {removed} image(s) precedente(s) supprimee(s) pour '{prefix}'")

# ─── Utilitaires ──────────────────────────────────────────────────────────────

def load_originals(prefix, nb_originaux):
    """Charge uniquement les images originales d'une categorie."""
    images = []
    for i in range(1, nb_originaux + 1):
        p = os.path.join(BASE_DIR, f"{prefix} ({i}).jpg")
        if os.path.exists(p):
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  [WARN] Impossible de lire {p} : {e}")
    return images

# ─── Blocs de transformation (chacun tres visible) ───────────────────────────

def apply_rotation(img):
    """Rotation moderee : entre -40 et 40 degres."""
    angle = random.uniform(-40, 40)
    # Couleur de remplissage proche de la moyenne de l'image (moins intrusif)
    arr = np.array(img)
    fill = tuple(arr.mean(axis=(0, 1)).astype(int).tolist())
    return img.rotate(angle, expand=False, resample=Image.Resampling.BICUBIC, fillcolor=fill)


def apply_flip(img):
    """Flip horizontal uniquement (le vertical peut trop denaturer)."""
    if random.random() > 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return img


def apply_brightness(img):
    """
    Luminosite variable mais reconnaissable : 0.5 (assez sombre) a 1.7 (clair).
    Evite les extremes qui rendent l'image illisible.
    """
    factor = random.uniform(0.5, 1.7)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_contrast(img):
    """Contraste modere : 0.65 (doux) a 1.9 (marque)."""
    factor = random.uniform(0.65, 1.9)
    return ImageEnhance.Contrast(img).enhance(factor)


def apply_saturation(img):
    """Saturation : entre 0.5 (pastel) et 2.0 (vif) — jamais en niveaux de gris."""
    factor = random.uniform(0.5, 2.0)
    return ImageEnhance.Color(img).enhance(factor)


def apply_crop_zoom(img):
    """Zoom leger a modere : recadrage de 5 a 20% sur chaque cote."""
    w, h = img.size
    pct = random.uniform(0.05, 0.20)
    left   = int(w * random.uniform(0, pct))
    top    = int(h * random.uniform(0, pct))
    right  = int(w * (1 - random.uniform(0, pct)))
    bottom = int(h * (1 - random.uniform(0, pct)))
    left, right = min(left, right - 10), max(right, left + 10)
    top, bottom = min(top, bottom - 10), max(bottom, top + 10)
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.Resampling.BICUBIC)


def apply_color_shift(img):
    """
    Leger decalage de teinte par canal R/G/B (effet chaud/froid discret).
    Plage reduite pour rester naturel.
    """
    arr = np.array(img, dtype=np.float32)
    shifts = np.random.uniform(-25, 25, 3)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + shifts[0], 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + shifts[1], 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + shifts[2], 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_gamma(img):
    """Correction gamma moderee : 0.55 a 1.8."""
    gamma = random.uniform(0.55, 1.8)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_blur_or_sharpen(img):
    """Flou doux OU legere surnettetee."""
    if random.random() > 0.5:
        radius = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        factor = random.uniform(1.5, 3.0)
        return ImageEnhance.Sharpness(img).enhance(factor)


def apply_noise(img):
    """Bruit gaussien modere."""
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(5, 18)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ─── Fonction principale d'augmentation ──────────────────────────────────────

# Pool de transformations "optionnelles" (en plus des obligatoires)
OPTIONAL_TRANSFORMS = [
    apply_crop_zoom,
    apply_color_shift,
    apply_gamma,
    apply_blur_or_sharpen,
    apply_noise,
]

def augment(img):
    """
    Applique une combinaison garantissant une image visuellement differente :
    - Transformations geometriques toujours appliquees (rotation forte, flip)
    - Luminosite + contraste + saturation toujours appliques avec plages larges
    - 2 ou 3 transformations supplementaires piochees au hasard dans le pool
    """
    # Geometrique (toujours)
    img = apply_rotation(img)
    img = apply_flip(img)

    # Photometrique (toujours, plages larges)
    img = apply_brightness(img)
    img = apply_contrast(img)
    img = apply_saturation(img)

    # Supplementaires : on en tire 2 ou 3 parmi les 5 disponibles
    nb_extra = random.randint(2, 3)
    extras = random.sample(OPTIONAL_TRANSFORMS, nb_extra)
    for fn in extras:
        img = fn(img)

    return img

# ─── Generation ──────────────────────────────────────────────────────────────

def generate_for_category(prefix, nb_originaux, target):
    print(f"\n[{prefix}]")

    # 1. Nettoyage des images precedemment generees
    cleanup(prefix, nb_originaux)

    needed = target - nb_originaux
    print(f"  {nb_originaux} originales, cible {target} -> {needed} a generer")

    sources = load_originals(prefix, nb_originaux)
    if not sources:
        print(f"  [ERREUR] Aucune image originale trouvee pour '{prefix}'.")
        return

    idx = nb_originaux + 1

    for i in range(needed):
        src = random.choice(sources)
        aug = augment(src.copy())

        out_name = f"{prefix} ({idx}).jpg"
        out_path = os.path.join(BASE_DIR, out_name)
        aug.save(out_path, "JPEG", quality=95)
        print(f"  [{i+1}/{needed}] Genere : {out_name}")
        idx += 1

    print(f"  OK : {needed} images generees pour '{prefix}'.")


def main():
    print("=" * 55)
    print("   Augmentation du dataset (mode intensif)")
    print("=" * 55)
    print(f"Dossier : {BASE_DIR}\n")

    for prefix, (nb_originaux, target) in CATEGORIES.items():
        generate_for_category(prefix, nb_originaux, target)

    print("\n" + "=" * 55)
    print("   Termine !")
    print("\nRecapitulatif :")
    for prefix, (nb_originaux, target) in CATEGORIES.items():
        pattern = os.path.join(BASE_DIR, f"{prefix} (*).jpg")
        final = len(glob.glob(pattern))
        status = "OK" if final >= target else f"MANQUE {target - final}"
        print(f"  {prefix:12s} : {final}/{target}  [{status}]")
    print("=" * 55)


if __name__ == "__main__":
    main()
