"""
ocr_position.py — Lecture de la position dans le TEXTE incrusté sur la photo.

POURQUOI
--------
Certaines applications de terrain (bandeau « Work Progress » sur iPhone) n'écrivent
PAS les coordonnées GPS dans l'EXIF : la position n'existe que sous forme de texte
incrusté dans l'image (« Lat 48.508951° Long 1.232363° »). Ce module lit ce texte
par OCR (Tesseract) puis l'interprète. Il n'intervient qu'en SECOURS de l'EXIF.

EXTENSIBILITÉ — POINT CLÉ
-------------------------
Les formats de coordonnées ne sont pas figés dans une regex unique : ils sont
déclarés dans FORMATS_COORD, une liste de (nom, fonction). Chaque fonction reçoit
le texte OCR et retourne (lat, lon) ou None. Les formats sont essayés dans l'ordre
et le premier résultat qui passe le garde-fou géographique est retenu.

Prendre en charge un nouveau format d'application = écrire une fonction et ajouter
une ligne à FORMATS_COORD. Aucun autre fichier n'est à modifier.

GARDE-FOU
---------
Une coordonnée lue par OCR n'est acceptée que si elle tombe en France
métropolitaine (voir BORNES_FRANCE). L'OCR confond facilement des chiffres :
mieux vaut écarter une photo que la placer à un endroit faux.
"""

import os
import re
import shutil

from PIL import Image, ImageOps

import formats_images  # noqa: F401  (enregistre le décodeur HEIC avant Image.open)

try:
    import pytesseract
    OCR_IMPORTE = True
except Exception:                                             # pragma: no cover
    pytesseract = None
    OCR_IMPORTE = False

# Langue Tesseract : « eng » suffit et est le seul pack requis. Les coordonnées
# sont numériques ; ajouter « fra » n'apporte rien et alourdit le déploiement.
LANGUE_OCR = "eng"

# Garde-fou géographique — France métropolitaine (lat_min, lat_max, lon_min, lon_max).
BORNES_FRANCE = (41.0, 51.6, -5.5, 9.8)

# Emplacements où l'installeur Windows dépose tesseract.exe. L'ajout au PATH y
# est facultatif (et absent d'une installation « pour l'utilisateur courant ») :
# sans cette recherche, l'OCR passerait pour indisponible alors qu'il est là.
CHEMINS_TESSERACT_WINDOWS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
)


def _localiser_tesseract():
    """Indique à pytesseract où trouver le binaire s'il n'est pas dans le PATH."""
    if not OCR_IMPORTE or shutil.which("tesseract"):
        return
    for chemin in CHEMINS_TESSERACT_WINDOWS:
        if os.path.isfile(chemin):
            pytesseract.pytesseract.tesseract_cmd = chemin
            return


_localiser_tesseract()

# Message d'installation, réutilisé par l'interface quand Tesseract manque.
AIDE_INSTALLATION = (
    "Tesseract n'est pas installé sur cette machine : la lecture des coordonnées "
    "par OCR est désactivée (les photos géolocalisées par EXIF restent traitées). "
    "Installation — Windows : installeur UB-Mannheim ; macOS : « brew install "
    "tesseract » ; Linux : « sudo apt-get install tesseract-ocr »."
)


# --------------------------------------------------------------------------
# Formats de coordonnées — liste extensible
# --------------------------------------------------------------------------

def _parse_decimal(texte):
    """Degrés décimaux étiquetés, ex. « Lat 48.508951° Long 1.232363° ».

    Couvre aussi « Latitude: 48,4459 Longitude: 4,5031 » (virgule décimale,
    deux-points, mot complet). Au moins trois décimales sont exigées : en
    dessous, il s'agit plus probablement d'un autre nombre du bandeau
    (altitude, vitesse, heure) que d'une coordonnée.
    """
    m = re.compile(
        r'lat\w*\s*[:=]?\s*(-?\d{1,2}[.,]\d{3,})\s*[°º]?.*?'
        r'long\w*\s*[:=]?\s*(-?\d{1,3}[.,]\d{3,})',
        re.IGNORECASE | re.DOTALL).search(texte)
    if m:
        return (float(m.group(1).replace(',', '.')),
                float(m.group(2).replace(',', '.')))


def _parse_dms(texte):
    """Degrés / minutes / secondes suffixés, ex. « 48°30'32.1"N 1°13'56"E ».

    Les secondes et leurs décimales sont facultatives.

    À REVALIDER SUR CAS RÉEL : ce format n'a été éprouvé que sur du texte
    synthétique, aucune photo de terrain en DMS n'était disponible.
    Cas connu non couvert : cardinal placé AVANT les chiffres (« N 48° 26' 45" ») ;
    ce sera précisément l'objet d'une future entrée de FORMATS_COORD.
    """
    motif = re.compile(
        r"(\d{1,3})\s*[°º]\s*(\d{1,2})\s*['′]\s*([\d.]+)?\s*[\"″]?\s*([NSEW])",
        re.IGNORECASE)
    valeurs = []
    for degres, minutes, secondes, reference in motif.findall(texte):
        v = int(degres) + int(minutes) / 60 + (float(secondes) if secondes else 0) / 3600
        if reference.upper() in "SW":
            v = -v
        valeurs.append((reference.upper(), v))

    lat = [v for r, v in valeurs if r in "NS"]
    lon = [v for r, v in valeurs if r in "EW"]
    if lat and lon:
        return lat[0], lon[0]


# Formats essayés dans l'ordre. AJOUTER UN FORMAT = AJOUTER UNE LIGNE ICI.
FORMATS_COORD = [
    ("decimal", _parse_decimal),
    ("dms",     _parse_dms),
]


# --------------------------------------------------------------------------
# Garde-fou et interprétation du texte
# --------------------------------------------------------------------------

def position_valide(lat, lon):
    """Vrai si la position tombe en France métropolitaine.

    Garde-fou obligatoire : aucune coordonnée hors de ces bornes n'est portée
    sur la carte, qu'elle vienne de l'OCR ou d'un EXIF douteux.
    """
    if lat is None or lon is None:
        return False
    lat_min, lat_max, lon_min, lon_max = BORNES_FRANCE
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def interpreter_texte(texte):
    """Cherche une position dans un texte, en essayant tous les formats connus.

    Retourne (lat, lon, nom_du_format) ou None. Un format qui produit une
    coordonnée hors bornes est ignoré : le suivant a sa chance.
    """
    if not texte:
        return None

    for nom, analyser in FORMATS_COORD:
        try:
            resultat = analyser(texte)
        except Exception:
            resultat = None            # un format défaillant ne bloque pas les autres
        if resultat:
            lat, lon = resultat
            if position_valide(lat, lon):
                return lat, lon, nom
    return None


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------

def tesseract_disponible():
    """Vrai si le moteur Tesseract est utilisable (module Python ET binaire)."""
    if not OCR_IMPORTE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _passes(image):
    """Les deux passes d'OCR, dans l'ordre d'essai.

    1. l'image entière ;
    2. sa moitié basse agrandie ×2 — le bandeau de coordonnées y est le plus
       souvent incrusté, et l'agrandissement rattrape les petits caractères.
    """
    yield image

    largeur, hauteur = image.size
    bas = image.crop((0, hauteur // 2, largeur, hauteur))
    yield bas.resize((bas.width * 2, bas.height * 2), Image.LANCZOS)


def lire_position_ocr(chemin_image):
    """Lit la position dans le texte incrusté d'une photo.

    Retourne un dictionnaire : lat, lon, format (nom du format reconnu) et
    message (explication en clair en cas d'échec).
    """
    resultat = {"lat": None, "lon": None, "format": None, "message": ""}

    if not tesseract_disponible():
        resultat["message"] = "OCR indisponible (Tesseract non installé)"
        return resultat

    try:
        with Image.open(chemin_image) as brut:
            # exif_transpose : sans redressement, la moitié « basse » de la
            # deuxième passe ne serait pas celle que voit l'utilisateur.
            image = ImageOps.exif_transpose(brut).convert("RGB")
    except Exception as erreur:
        resultat["message"] = f"Image illisible pour l'OCR ({erreur})"
        return resultat

    textes = []
    for image_passe in _passes(image):
        try:
            textes.append(pytesseract.image_to_string(image_passe, lang=LANGUE_OCR))
        except Exception as erreur:
            resultat["message"] = f"OCR en échec ({erreur})"
            return resultat

        lu = interpreter_texte(textes[-1])
        if lu:
            resultat["lat"], resultat["lon"], resultat["format"] = lu
            return resultat

    # Dernière chance : les deux passes réunies, au cas où la latitude et la
    # longitude auraient été lues chacune par une passe différente.
    lu = interpreter_texte("\n".join(textes))
    if lu:
        resultat["lat"], resultat["lon"], resultat["format"] = lu
        return resultat

    resultat["message"] = "Aucune coordonnée reconnue dans le texte de l'image"
    return resultat
