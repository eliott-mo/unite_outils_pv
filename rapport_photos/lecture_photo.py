"""
lecture_photo.py — Cascades de lecture d'une photo : d'abord la position, puis le cap.

Ce module n'invente rien : il APPELLE dans un ordre fixé les briques existantes
(`lecture_exif`, `ocr_position`, `detection_cap`) et rend un enregistrement
unique, prêt à alimenter le tableau et la carte.

CASCADE DE POSITION (ordre impératif)
  1. position EXIF valide            → source « EXIF » ;
  2. sinon, texte incrusté lu en OCR → source « OCR » ;
  3. sinon, photo écartée avec le motif « position introuvable ».
Dans les deux premiers cas, la position doit passer le garde-fou géographique
(France métropolitaine) : une coordonnée hors bornes est rejetée, jamais placée.

CASCADE DE CAP
  1. cap EXIF (GPSImgDirection), écrit par les iPhone et Open Camera → « EXIF » ;
  2. sinon, cône bleu de la vignette GPS Map Camera                  → « Vignette » ;
  3. sinon, pas de cap → le marqueur sera affiché sans cône.
L'absence de cap n'est PAS une erreur : un drone ou un bandeau « Work Progress »
n'en fournit aucun, et la carte sait déjà l'afficher.
"""

import os
import shutil
import tempfile

from PIL import Image, ImageOps

import formats_images
from detection_cap import detecter_cap
from lecture_exif import lire_metadonnees
from ocr_position import lire_position_ocr, position_valide

SANS_CAP = "—"


def _chemin_lisible_par_opencv(chemin_image):
    """Retourne un chemin qu'OpenCV sait ouvrir, et True si c'est un fichier temporaire.

    `detection_cap` travaille avec cv2.imread, qui bute sur deux cas :
      - le HEIC, qu'il ne sait pas décoder → copie ré-encodée en JPEG ;
      - un chemin contenant des accents, qu'il n'ouvre pas sous Windows (il rend
        None sans message explicite, donnant à tort « image illisible ») → simple
        recopie sous un nom neutre, sans ré-encodage.

    Passer par une copie évite de modifier le module de détection, dont le
    fonctionnement reste inchangé.
    """
    heic = formats_images.est_heic(chemin_image)
    chemin_accentue = not str(chemin_image).isascii()
    if not (heic or chemin_accentue):
        return chemin_image, False

    extension = ".jpg" if heic else os.path.splitext(chemin_image)[1]
    descripteur, temporaire = tempfile.mkstemp(suffix=extension, prefix="cap_")
    os.close(descripteur)

    if heic:
        with Image.open(chemin_image) as brut:
            ImageOps.exif_transpose(brut).convert("RGB").save(temporaire, "JPEG", quality=92)
    else:
        # Recopie à l'identique : la détection doit voir les pixels d'origine.
        shutil.copyfile(chemin_image, temporaire)
    return temporaire, True


def _lire_cap(chemin_image, cap_exif):
    """Applique la cascade de cap. Retourne (cap, confiance, source, message)."""
    # 1. Le cap EXIF vient directement de la boussole : plus fiable que l'analyse
    #    d'image, qui passe par le rendu de la vignette.
    if cap_exif is not None:
        return cap_exif, 1.0, "EXIF", ""

    # 2. Repli sur le cône bleu de la vignette GPS Map Camera.
    chemin, temporaire = _chemin_lisible_par_opencv(chemin_image)
    try:
        detection = detecter_cap(chemin)
    finally:
        if temporaire:
            os.remove(chemin)

    if detection["cap"] is not None:
        return detection["cap"], detection["confiance"], "Vignette", ""

    # 3. Pas de cône (drone, bandeau sans boussole) : simple absence de cap.
    return None, None, SANS_CAP, detection["message"]


def lire_photo(chemin_image):
    """Lit une photo de bout en bout : position, cap, date et provenances.

    Retourne un dictionnaire :
        lat, lon         : position en degrés décimaux, ou None si photo à écarter
        source_position  : « EXIF » ou « OCR »
        format_position  : nom du format de coordonnées reconnu par l'OCR, sinon None
        precision_m      : incertitude annoncée par l'appareil, ou None si inconnue
        cap              : cap brut en degrés, ou None
        confiance        : indice de confiance de la détection du cap, ou None
        source_cap       : « EXIF », « Vignette » ou « — »
        date             : datetime de prise de vue, ou None
        message          : motif du rejet, ou explication de l'absence de cap
    """
    resultat = {
        "lat": None, "lon": None, "source_position": None, "format_position": None,
        "precision_m": None, "cap": None, "confiance": None, "source_cap": SANS_CAP,
        "date": None, "message": "",
    }

    meta = lire_metadonnees(chemin_image)
    resultat["date"] = meta["date"]

    # --- Cascade de position ---------------------------------------------
    if position_valide(meta["lat"], meta["lon"]):
        resultat["lat"], resultat["lon"] = meta["lat"], meta["lon"]
        resultat["source_position"] = "EXIF"
        # L'incertitude EXIF décrit la fixation GPS de l'appareil : elle n'a de
        # sens que pour une position issue de cette même fixation, donc pas pour
        # une position relue par OCR dans le bandeau.
        resultat["precision_m"] = meta["precision_m"]
    else:
        # Position EXIF absente, illisible ou hors France : on tente le texte
        # incrusté. Le motif EXIF est conservé pour l'expliquer en cas d'échec.
        if meta["lat"] is not None:
            motif_exif = "position EXIF hors France métropolitaine"
        else:
            motif_exif = meta["message"] or "pas de position EXIF"

        ocr = lire_position_ocr(chemin_image)
        if ocr["lat"] is not None:
            resultat["lat"], resultat["lon"] = ocr["lat"], ocr["lon"]
            resultat["source_position"] = "OCR"
            resultat["format_position"] = ocr["format"]
        else:
            resultat["message"] = (
                f"Position introuvable (ni EXIF ni texte lisible) — "
                f"{motif_exif} ; {ocr['message'].lower()}"
            )
            return resultat

    # --- Cascade de cap ---------------------------------------------------
    (resultat["cap"], resultat["confiance"],
     resultat["source_cap"], message_cap) = _lire_cap(chemin_image, meta["cap_exif"])
    if resultat["cap"] is None:
        resultat["message"] = message_cap

    return resultat
