"""
detection_cap.py — Détection automatique du cap (direction de prise de vue)
à partir de la vignette Google Maps incrustée par l'application GPS Map Camera.

PRINCIPE
--------
GPS Map Camera incruste en bas à gauche de la photo une mini-carte contenant :
  - un marqueur rouge Google (#EA4335) qui localise le photographe ;
  - un cône bleu Google (#4285F4) qui indique la direction visée.
La mini-carte est orientée nord en haut (vérifié sur photos de référence).

On repère donc le marqueur rouge, on prend sa pointe (base du marqueur) comme
sommet du cône, puis on calcule la direction moyenne des pixels bleus situés
autour de ce sommet.

LIMITE CONNUE
-------------
La précision est celle de la boussole du téléphone, soit ±10 à 20°.
C'est suffisant pour comprendre l'orientation d'une photo, pas pour du relevé.
"""

import math
import cv2
import numpy as np

# Couleur BGR du marqueur Google (#EA4335) — sert à départager les blobs rouges.
GOOGLE_RED_BGR = np.array([53, 67, 234])

# Rayon d'analyse du cône, exprimé en multiples de la hauteur du marqueur.
# Testé de 1,5 à 6 : stable jusqu'à 4, dérive au-delà car le masque déborde
# hors de la vignette et capte des éléments bleutés de la photo (asphalte, eau).
RAYON_CONE = 3.0

# Zone de recherche du marqueur, en fraction de l'image (x_min, y_min, x_max, y_max).
# Volontairement large pour tolérer les variations de mise en page de l'appli.
ZONE_RECHERCHE = (0.0, 0.55, 0.55, 1.0)

# Sous-échantillonnage avant traitement : divise par 4 la mémoire et le temps
# de calcul sans dégrader le résultat (écart mesuré < 1,5°).
ECHELLE = 0.5

# Seuil de concentration angulaire au-delà duquel la détection est jugée fiable.
SEUIL_CONFIANCE = 0.70


def _masque_rouge(hsv):
    """Masque binaire des pixels rouges saturés (marqueur Google)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = (((h <= 8) | (h >= 172)) & (s > 140) & (v > 90)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _masque_bleu(hsv):
    """Masque binaire des pixels du bleu Google (#4285F4).

    La plage de teinte est étroite : elle exclut le turquoise de l'eau en vue
    satellite, qui est la principale source de confusion possible.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((h >= 100) & (h <= 122) & (s > 70) & (v > 110)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _trouver_marqueur(zone_bgr, hsv):
    """Retourne (x_pointe, y_pointe, hauteur, aire) du marqueur, ou None.

    Parmi tous les blobs rouges, on retient celui qui maximise
    aire / (1 + écart de couleur au rouge Google), en ne gardant que ceux dont
    le rapport hauteur/largeur correspond à une goutte de marqueur (0,9 à 2,0).
    """
    masque = _masque_rouge(hsv)
    nb, labels, stats, centroides = cv2.connectedComponentsWithStats(masque)

    nb_pixels = zone_bgr.shape[0] * zone_bgr.shape[1]
    aire_min, aire_max = 0.0002 * nb_pixels, 0.05 * nb_pixels

    meilleur = None
    for i in range(1, nb):
        aire = stats[i, cv2.CC_STAT_AREA]
        larg = stats[i, cv2.CC_STAT_WIDTH]
        haut = stats[i, cv2.CC_STAT_HEIGHT]

        if not (aire_min <= aire <= aire_max):
            continue
        if not (0.9 <= haut / max(larg, 1) <= 2.0):
            continue

        ecart = np.linalg.norm(zone_bgr[labels == i].mean(axis=0) - GOOGLE_RED_BGR)
        score = aire / (1.0 + ecart)
        if meilleur is None or score > meilleur[0]:
            meilleur = (score, centroides[i], haut, aire)

    if meilleur is None:
        return None

    _, (cx, cy), haut, aire = meilleur
    # La pointe du marqueur (position réelle) est en bas de la goutte ;
    # le centroïde est à peu près à mi-hauteur.
    return cx, cy + haut * 0.5, haut, aire


def detecter_cap(chemin_image):
    """Détecte le cap de prise de vue d'une photo GPS Map Camera.

    Retourne un dictionnaire :
        cap        : cap en degrés (0 = nord, 90 = est), ou None si indétectable
        confiance  : concentration angulaire de 0 à 1, ou None
        fiable     : True si confiance >= SEUIL_CONFIANCE
        message    : explication en clair si la détection a échoué
    """
    echec = lambda msg: {"cap": None, "confiance": None, "fiable": False, "message": msg}

    image = cv2.imread(str(chemin_image))
    if image is None:
        return echec("Image illisible")

    if ECHELLE != 1.0:
        image = cv2.resize(image, None, fx=ECHELLE, fy=ECHELLE,
                           interpolation=cv2.INTER_AREA)

    H, W = image.shape[:2]
    x0, y0 = int(ZONE_RECHERCHE[0] * W), int(ZONE_RECHERCHE[1] * H)
    x1, y1 = int(ZONE_RECHERCHE[2] * W), int(ZONE_RECHERCHE[3] * H)
    zone = image[y0:y1, x0:x1]

    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV).astype(int)

    marqueur = _trouver_marqueur(zone, hsv)
    if marqueur is None:
        return echec("Marqueur Google introuvable (vignette absente ?)")

    ax, ay, haut, aire = marqueur

    ys, xs = np.nonzero(_masque_bleu(hsv))
    if len(xs) == 0:
        return echec("Aucun pixel bleu (boussole désactivée ?)")

    # On ne garde que les pixels bleus proches du marqueur : au-delà, le masque
    # sortirait de la vignette et capterait des éléments de la photo elle-même.
    proches = np.hypot(xs - ax, ys - ay) < haut * RAYON_CONE
    xs, ys = xs[proches], ys[proches]

    # Un cône représente typiquement plusieurs fois l'aire du marqueur.
    # En dessous de 10 % on considère qu'il n'y a pas de cône exploitable.
    if len(xs) < 0.10 * aire:
        return echec("Cône de direction absent (boussole désactivée ?)")

    # Angle de chaque pixel : 0° au nord (haut de l'image), sens horaire.
    angles = np.arctan2(xs - ax, -(ys - ay))

    # Moyenne circulaire : la moyenne arithmétique serait fausse autour de 0°/360°.
    sin_moy, cos_moy = np.sin(angles).mean(), np.cos(angles).mean()
    cap = math.degrees(math.atan2(sin_moy, cos_moy)) % 360.0

    # Longueur du vecteur résultant : 1 = tous les pixels alignés (cône net),
    # proche de 0 = pixels dispersés (ce n'était pas un cône).
    confiance = float(math.hypot(sin_moy, cos_moy))

    return {
        "cap": round(cap, 1),
        "confiance": round(confiance, 2),
        "fiable": confiance >= SEUIL_CONFIANCE,
        "message": "",
    }
