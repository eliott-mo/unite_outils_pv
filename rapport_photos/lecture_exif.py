"""
lecture_exif.py — Extraction de la position GPS, de la date et du cap EXIF.

Le cap EXIF (GPSImgDirection) n'est PAS écrit par GPS Map Camera : il est lu
ici uniquement pour le cas où l'équipe utiliserait une autre application
(iPhone, Open Camera, Solocator...). Quand il est présent, il est prioritaire sur
la détection par analyse d'image, qui reste la solution de repli.

L'import de `formats_images` enregistre le décodeur HEIC auprès de Pillow avant
tout Image.open : ce module lit donc les HEIC d'iPhone sans code spécifique.
"""

from datetime import datetime
from PIL import Image, ExifTags

import formats_images  # noqa: F401  (enregistre le décodeur HEIC — à garder en tête de fichier)

# Identifiants des sous-répertoires EXIF (constants du standard).
IFD_GPS = 0x8825
IFD_EXIF = 0x8769

# Codes des champs GPS utilisés.
GPS_LAT_REF, GPS_LAT = 1, 2
GPS_LON_REF, GPS_LON = 3, 4
GPS_DIR_REF, GPS_DIR = 16, 17
GPS_PRECISION = 31          # GPSHPositioningError : incertitude horizontale, en mètres

# Au-delà de cette incertitude, la position est signalée comme peu fiable dans le
# tableau de résultats. Elle n'est jamais écartée pour autant : elle reste souvent
# utile, et c'est au CDP de juger.
# Valeur calée sur l'usage : à l'échelle d'une visite de site, 100 m ne voulait
# plus rien dire (on change de parcelle). Un iPhone en bonne réception annonce
# environ 5 m, une fixation dégradée plusieurs milliers ; 20 m laisse donc passer
# une réception normale tout en signalant tout ce qui commence à dériver.
SEUIL_PRECISION_M = 20


def _dms_vers_degres(dms, reference):
    """Convertit une coordonnée EXIF (degrés, minutes, secondes) en degrés décimaux."""
    degres, minutes, secondes = [float(x) for x in dms]
    valeur = degres + minutes / 60.0 + secondes / 3600.0
    if reference in ("S", "W"):
        valeur = -valeur
    return valeur


def lire_metadonnees(chemin_image):
    """Retourne un dictionnaire : lat, lon, cap_exif, precision_m, date, message.

    lat/lon valent None si la photo n'est pas géolocalisée : elle sera alors
    écartée de la carte, avec un message explicite pour l'utilisateur.

    precision_m est l'incertitude annoncée par le téléphone lui-même. La plupart
    des applications ne l'écrivent pas : None signifie « inconnue », surtout pas
    « mauvaise ».
    """
    resultat = {"lat": None, "lon": None, "cap_exif": None,
                "precision_m": None, "date": None, "message": ""}

    try:
        with Image.open(chemin_image) as image:
            exif = image.getexif()
            gps = exif.get_ifd(IFD_GPS)
            bloc_exif = exif.get_ifd(IFD_EXIF)
    except Exception as erreur:
        resultat["message"] = f"Lecture impossible ({erreur})"
        return resultat

    # Incertitude horizontale annoncée par l'appareil (GPSHPositioningError).
    # Lue avant les coordonnées : le champ existe indépendamment d'elles.
    if gps and gps.get(GPS_PRECISION) is not None:
        try:
            resultat["precision_m"] = float(gps[GPS_PRECISION])
        except Exception:
            pass                        # champ illisible : on reste sur « inconnue »

    if not gps or GPS_LAT not in gps or GPS_LON not in gps:
        resultat["message"] = "Photo non géolocalisée (pas de coordonnées GPS)"
        return resultat

    try:
        resultat["lat"] = _dms_vers_degres(gps[GPS_LAT], gps.get(GPS_LAT_REF, "N"))
        resultat["lon"] = _dms_vers_degres(gps[GPS_LON], gps.get(GPS_LON_REF, "E"))
    except Exception:
        resultat["message"] = "Coordonnées GPS illisibles"
        return resultat

    # Cap EXIF, si l'application photo l'a écrit.
    if GPS_DIR in gps:
        try:
            cap = float(gps[GPS_DIR]) % 360.0
            reference = gps.get(GPS_DIR_REF, "T")  # T = nord géographique, M = magnétique
            resultat["cap_exif"] = round(cap, 1)
            resultat["cap_exif_ref"] = "magnétique" if reference == "M" else "géographique"
        except Exception:
            pass

    # Date de prise de vue : on privilégie DateTimeOriginal (36867).
    brut = bloc_exif.get(36867) or exif.get(306)
    if brut:
        try:
            resultat["date"] = datetime.strptime(str(brut), "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass

    return resultat
