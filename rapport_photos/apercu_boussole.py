"""
apercu_boussole.py — Aperçu visuel de la direction de prise de vue.

Dessine une rose des vents montrant, superposées :
  - la direction brute détectée (cône gris, en arrière-plan) ;
  - la direction après calibration (cône orange, identique à celui de la carte) ;
  - la rotation appliquée, matérialisée par un arc fléché.

Objectif : permettre de régler l'offset en regardant tourner le cône, sans avoir
à raisonner sur une convention d'angles ni à générer la carte pour vérifier.

CHOIX TECHNIQUE
---------------
L'image est produite avec Pillow plutôt qu'en SVG : Streamlit filtre les balises
<svg> transmises à st.html(), alors que st.image() affiche un objet Pillow de
façon garantie. Pillow est par ailleurs déjà une dépendance du projet.

Le dessin est réalisé à trois fois la taille finale puis réduit : ce
suréchantillonnage donne des bords lisses, Pillow ne gérant pas l'anticrénelage.

Les couleurs reprennent volontairement celles de generation_html.py pour que
l'aperçu et la carte finale soient visuellement cohérents.
"""

import math

from PIL import Image, ImageDraw, ImageFont

ORANGE = (255, 138, 0)          # cône de direction, identique à la carte
ROUGE = (211, 47, 47)           # pastille de position, identique à la carte
GRIS = (154, 167, 180)          # direction brute, avant calibration
FOND_CADRAN = (15, 23, 32)
BORD_CADRAN = (58, 74, 92)
GRADUATION = (74, 92, 110)
TEXTE = (232, 238, 244)
ARC = (110, 128, 148)

OUVERTURE_CONE = 50             # angle d'ouverture du cône, en degrés
SUREECHANTILLONNAGE = 3

# Polices habituellement présentes sur Linux (Streamlit Cloud) et Windows.
POLICES_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)


def _police(taille):
    """Charge une police de caractères, avec repli sur la police intégrée."""
    for chemin in POLICES_CANDIDATES:
        try:
            return ImageFont.truetype(chemin, taille)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=taille)
    except TypeError:      # Pillow antérieur à 10.1
        return ImageFont.load_default()


def _point(centre, rayon, cap):
    """Coordonnées d'un point à `rayon` du centre, dans la direction `cap`.

    Convention : 0° = nord = haut de l'image, sens horaire (90° = est = droite).
    """
    radians = math.radians(cap)
    return (centre + rayon * math.sin(radians),
            centre - rayon * math.cos(radians))


def _boite(centre, rayon):
    """Rectangle englobant un cercle, au format attendu par Pillow."""
    return [centre - rayon, centre - rayon, centre + rayon, centre + rayon]


def _angles_pillow(cap, ouverture=OUVERTURE_CONE):
    """Convertit un cap en angles Pillow.

    Pillow compte les angles à partir de 3 heures, dans le sens horaire :
    un cap de 0° (nord) correspond donc à -90° chez Pillow.
    """
    return cap - ouverture / 2 - 90, cap + ouverture / 2 - 90


def _dessiner_cone(dessin, centre, rayon, cap, couleur, opacite, epaisseur):
    """Dessine un secteur angulaire (le cône), sommet au centre."""
    debut, fin = _angles_pillow(cap)
    dessin.pieslice(_boite(centre, rayon), debut, fin,
                    fill=couleur + (opacite,), outline=couleur + (255,),
                    width=epaisseur)


def _dessiner_arc_rotation(dessin, centre, rayon, depart, arrivee, echelle):
    """Arc fléché figurant la rotation appliquée, du cap brut vers le cap corrigé."""
    ecart = (arrivee - depart + 180.0) % 360.0 - 180.0
    if abs(ecart) < 2:
        return

    # Pillow dessine toujours dans le sens horaire : on inverse les bornes
    # lorsque la rotation se fait vers la gauche.
    if ecart > 0:
        debut, fin = depart - 90, arrivee - 90
    else:
        debut, fin = arrivee - 90, depart - 90
    dessin.arc(_boite(centre, rayon), debut, fin,
               fill=ARC + (255,), width=int(2 * echelle))

    # Pointe de flèche : petit triangle tangent à l'arc, à son extrémité.
    tangente = arrivee + (90 if ecart > 0 else -90)
    sommet = _point(centre, rayon, arrivee)
    pointe = (sommet[0] + 7 * echelle * math.sin(math.radians(tangente)),
              sommet[1] - 7 * echelle * math.cos(math.radians(tangente)))
    dessin.polygon([pointe,
                    _point(centre, rayon - 4.5 * echelle, arrivee),
                    _point(centre, rayon + 4.5 * echelle, arrivee)],
                   fill=ARC + (255,))


def _dessiner_cadran(dessin, centre, rayon, echelle):
    """Cadran : cercle, graduations tous les 30° et lettres cardinales."""
    dessin.ellipse(_boite(centre, rayon), fill=FOND_CADRAN + (255,),
                   outline=BORD_CADRAN + (255,), width=int(1.5 * echelle))

    for angle in range(0, 360, 30):
        majeure = angle % 90 == 0
        longueur = (11 if majeure else 6) * echelle
        dessin.line([_point(centre, rayon, angle),
                     _point(centre, rayon - longueur, angle)],
                    fill=GRADUATION + (255,), width=int((2 if majeure else 1) * echelle))

    police = _police(int(14 * echelle))
    for lettre, angle in (("N", 0), ("E", 90), ("S", 180), ("O", 270)):
        x, y = _point(centre, rayon + 13 * echelle, angle)
        dessin.text((x, y), lettre, font=police, anchor="mm",
                    fill=(ORANGE if lettre == "N" else TEXTE) + (255,))


def boussole(cap_brut, cap_corrige, taille=250, afficher_brut=True):
    """Construit l'image de l'aperçu et la retourne (objet Pillow, mode RGBA).

    cap_brut      : direction détectée avant calibration (peut valoir None)
    cap_corrige   : direction après application de l'offset (peut valoir None)
    afficher_brut : si False, seul le cône corrigé est dessiné
    """
    echelle = SUREECHANTILLONNAGE
    cote = taille * echelle
    centre = cote / 2
    rayon_cadran = cote / 2 - 20 * echelle
    rayon_cone = rayon_cadran - 6 * echelle

    image = Image.new("RGBA", (cote, cote), (0, 0, 0, 0))
    dessin = ImageDraw.Draw(image, "RGBA")

    _dessiner_cadran(dessin, centre, rayon_cadran, echelle)

    if cap_brut is None and cap_corrige is None:
        dessin.text((centre, centre + 20 * echelle), "direction inconnue",
                    font=_police(int(12 * echelle)), anchor="mm",
                    fill=(122, 138, 154, 255))
    else:
        ecart_significatif = (
            cap_brut is not None and cap_corrige is not None
            and abs((cap_corrige - cap_brut + 180) % 360 - 180) >= 2
        )
        # Direction brute, en arrière-plan.
        if afficher_brut and ecart_significatif:
            _dessiner_cone(dessin, centre, rayon_cone * 0.80, cap_brut,
                           GRIS, 55, int(1.5 * echelle))
            _dessiner_arc_rotation(dessin, centre, rayon_cone * 0.42,
                                   cap_brut, cap_corrige, echelle)

        # Direction retenue, telle qu'elle apparaîtra sur la carte.
        cap = cap_corrige if cap_corrige is not None else cap_brut
        _dessiner_cone(dessin, centre, rayon_cone, cap, ORANGE, 140, int(2 * echelle))

    # Pastille de position : le point où se tient le photographe.
    rayon_pastille = 9 * echelle
    dessin.ellipse(_boite(centre, rayon_pastille), fill=ROUGE + (255,),
                   outline=(255, 255, 255, 255), width=int(2.5 * echelle))

    return image.resize((taille, taille), Image.LANCZOS)


def legende_html(cap_brut, cap_corrige, offset):
    """Légende texte accompagnant l'aperçu, en HTML."""
    def rose(cap):
        if cap is None:
            return "—"
        noms = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        return f"{noms[round(cap / 22.5) % 16]} ({cap:.0f}°)"

    sens = ""
    if abs(offset) >= 1:
        sens = " vers la droite" if offset > 0 else " vers la gauche"

    gris = "#%02x%02x%02x" % GRIS
    orange = "#%02x%02x%02x" % ORANGE

    return (
        f'<div style="font-size:12.5px;line-height:1.7">'
        f'<span style="color:{gris}">Détecté : <b>{rose(cap_brut)}</b></span><br>'
        f'<span style="color:{orange}">Après correction : <b>{rose(cap_corrige)}</b></span><br>'
        f'<span style="color:#8fa0b0">Rotation de <b>{offset:+.0f}°</b>{sens}</span>'
        f'</div>'
    )
