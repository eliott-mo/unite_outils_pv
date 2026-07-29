"""
formats_images.py — Formats d'image acceptés et prise en charge du HEIC.

Ce module doit être importé AVANT tout appel à Image.open : il enregistre
auprès de Pillow le décodeur HEIC des iPhone (pillow-heif). Une fois cet
enregistrement fait, `lecture_exif.py` et `generation_html.py` lisent les .HEIC
sans aucune autre modification, métadonnées EXIF comprises (position ET cap).

Il centralise aussi la liste des extensions acceptées, pour que l'uploader, le
filtre du ZIP et le reste de l'application ne puissent pas diverger.
"""

# Extensions traitées en entrée. TIFF et WEBP sont lus nativement par Pillow et
# par OpenCV ; le HEIC ne l'est que grâce à l'enregistrement ci-dessous.
EXTENSIONS_IMAGE = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
                    ".heic", ".heif")

# Formats qu'OpenCV ne sait pas ouvrir : la détection du cap par la vignette
# passe alors par une conversion temporaire (voir lecture_photo.py).
EXTENSIONS_HEIC = (".heic", ".heif")

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_DISPONIBLE = True
    MESSAGE_HEIC = ""
except Exception as erreur:                                   # pragma: no cover
    HEIC_DISPONIBLE = False
    MESSAGE_HEIC = (
        "Le module « pillow-heif » n'est pas installé : les photos .HEIC "
        f"(iPhone) ne pourront pas être lues. Installer avec "
        f"« pip install pillow-heif ». ({erreur})"
    )


def est_image(nom_fichier):
    """Vrai si le nom de fichier porte une extension d'image prise en charge."""
    return nom_fichier.lower().endswith(EXTENSIONS_IMAGE)


def est_heic(nom_fichier):
    """Vrai si le fichier est un HEIC/HEIF (illisible par OpenCV)."""
    return nom_fichier.lower().endswith(EXTENSIONS_HEIC)
