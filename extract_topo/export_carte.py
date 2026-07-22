"""
export_carte.py — Génération image statique PNG/PDF de la carte de pentes.

Approche : contextily (fond satellite ESRI) + matplotlib (toutes les couches).
Ne dépend pas de Folium/Leaflet ni d'un navigateur headless.

Utilisation standalone :
    python export_carte.py
    → écrit test_export.png et test_export.pdf dans le dossier courant.
"""

import io
import os
import time
import math
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif, compatible serveur
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds as rt_from_bounds
from rasterio.crs import CRS
import rasterio.warp
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.ndimage import binary_opening, label as ndi_label
from skimage import measure as skimage_measure
from rasterio.features import shapes as rio_shapes
import geopandas as gpd
from shapely.geometry import box, shape as shapely_shape

try:
    import contextily as cx
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    _HAS_SCALEBAR = True
except ImportError:
    _HAS_SCALEBAR = False


# ── Constantes partagées avec app-2.py ────────────────────────────────────────

_PALETTE = {
    0: (76, 175, 80),    # vert   — Favorable
    1: (255, 193, 7),    # jaune  — Acceptable
    2: (255, 152, 0),    # orange — Contraignant
    3: (244, 67, 54),    # rouge  — Exclusion possible
}

# ── Mode topographie : pente seule, seuils universels, palette séquentielle ocre
_SEUILS_TOPO = [5, 10, 15]  # %
_PALETTE_TOPO = ["#f2e6c9", "#d9b166", "#a8763a", "#6b3f1d"]
_LABELS_TOPO = ["< 5 %", "5 – 10 %", "10 – 15 %", "> 15 %"]

# Modes de rendu acceptés par generer_image_carte() / creer_carte()
MODE_CONSTRUCTIBILITE = "constructibilite"
MODE_TOPOGRAPHIE = "topographie"

# Couches optionnelles, communes au rendu Folium et à l'export matplotlib.
# La couche "zones" n'existe qu'en mode topographie.
COUCHES = ("pentes", "courbes", "contour", "zones")
LIBELLES_COUCHES = {
    "pentes":  "Pentes",
    "courbes": "Courbes de niveau",
    "contour": "Contour du site",
    "zones":   "Zones planes",
}


def couches_par_defaut() -> dict:
    """Toutes les couches actives — état initial des deux cartes."""
    return {nom: True for nom in COUCHES}

_SEUILS = {
    ("FIXE", "<5MWc"): {
        "N":  {"seuils": [3,  5,  15], "note_contraignant": "Ombrage possible"},
        "S":  {"seuils": [5,  10, 15], "note_contraignant": None},
        "EO": {"seuils": [5,  10, 15], "note_contraignant": None},
        "note_globale": None,
    },
    ("FIXE", "≥5MWc"): {
        "N":  {"seuils": [3,  5,  15], "note_contraignant": "Ombrage possible"},
        "S":  {"seuils": [5,  10, 20], "note_contraignant": None},
        "EO": {"seuils": [5,  10, 15], "note_contraignant": None},
        "note_globale": (
            "Surcout significatif à prévoir à partir du stade « Contraignant » "
            "selon la proportion de terrain concernée. Au-delà du stade "
            "« Contraignant », étude de préfaisabilité spécifique au projet nécessaire."
        ),
    },
    ("TRACKERS", "<5MWc"): {
        "N":  {"seuils": [3, 5, 10], "note_contraignant": None},
        "S":  {"seuils": [3, 5, 10], "note_contraignant": None},
        "EO": {"seuils": [3, 5, 10], "note_contraignant": None},
        "note_globale": (
            "Le BE interne UNITe étudie actuellement les différentes solutions "
            "de trackers disponibles sur le marché afin de déterminer le degré "
            "de pente max admissible. Dans l'attente des résultats, considérer "
            "en phase ESQ une pente max admissible de 10 %."
        ),
    },
    ("TRACKERS", "≥5MWc"): {
        "N":  {"seuils": [3, 5, 10], "note_contraignant": None},
        "S":  {"seuils": [3, 5, 10], "note_contraignant": None},
        "EO": {"seuils": [3, 5, 10], "note_contraignant": None},
        "note_globale": (
            "Le BE interne UNITe étudie actuellement les différentes solutions "
            "de trackers disponibles sur le marché afin de déterminer le degré "
            "de pente max admissible. Dans l'attente des résultats, considérer "
            "en phase ESQ une pente max admissible de 10 %."
        ),
    },
    ("OMBRIÈRES", "<5MWc"): {
        "N":  {"seuils": [3, 5, 10], "note_contraignant": "Ombrage possible"},
        "S":  {"seuils": [3, 5, 10], "note_contraignant": None},
        "EO": {"seuils": [2, 3,  5], "note_contraignant": None},
        "note_globale": None,
    },
    ("OMBRIÈRES", "≥5MWc"): {
        "N":  {"seuils": [3, 5, 10], "note_contraignant": "Ombrage possible"},
        "S":  {"seuils": [3, 5, 10], "note_contraignant": None},
        "EO": {"seuils": [3, 5, 10], "note_contraignant": None},
        "note_globale": (
            "Surcout significatif à prévoir à partir du stade « Contraignant » "
            "selon la proportion de terrain concernée. Au-delà du stade "
            "« Contraignant », étude de préfaisabilité spécifique au projet nécessaire."
        ),
    },
}


# ── Helpers internes ──────────────────────────────────────────────────────────

def _classifier_pente_orientee(
    pentes: np.ndarray,
    orientations: np.ndarray,
    technologie: str,
    puissance: str,
) -> np.ndarray:
    """Même logique que app-2.py — retourne array uint8 0–3 / 255 pour NaN."""
    table = _SEUILS[(technologie, puissance)]
    cls = np.full(pentes.shape, 255, dtype=np.uint8)
    masks_orient = {
        "N":  np.isin(orientations, ["N", "NE", "NO"]),
        "S":  np.isin(orientations, ["S", "SE", "SO"]),
        "EO": np.isin(orientations, ["E", "O"]),
    }
    for orient, mask_o in masks_orient.items():
        s1, s2, s3 = table[orient]["seuils"]
        valid = mask_o & ~np.isnan(pentes)
        cls[valid & (pentes <  s1)]                  = 0
        cls[valid & (pentes >= s1) & (pentes <  s2)] = 1
        cls[valid & (pentes >= s2) & (pentes <  s3)] = 2
        cls[valid & (pentes >= s3)]                  = 3
    return cls


def classifier_pente_topo(pentes: np.ndarray) -> np.ndarray:
    """
    Classification par pente seule — seuils universels _SEUILS_TOPO.
    Ignore orientation / technologie / puissance.
    Retourne array uint8 0–3 / 255 pour NaN.

    Publique : app-3.py l'utilise pour la carte Folium en mode topographie,
    afin que les deux rendus (Folium et matplotlib) partagent la classification.
    """
    cls = np.full(pentes.shape, 255, dtype=np.uint8)
    valid = ~np.isnan(pentes)
    cls[valid] = np.digitize(pentes[valid], _SEUILS_TOPO).astype(np.uint8)
    return cls


def _hex_to_rgb(couleur_hex: str) -> tuple[int, int, int]:
    h = couleur_hex.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _to_rgba(
    classes: np.ndarray,
    palette: dict | None = None,
    alpha: int = 166,
) -> np.ndarray:
    """
    alpha=166 ≈ 0.65 × 255.
    palette : dict {classe: (r, g, b)} — _PALETTE (constructibilité) par défaut.
    """
    if palette is None:
        palette = _PALETTE
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for val, (r, g, b) in palette.items():
        rgba[classes == val] = [r, g, b, alpha]
    return rgba


# ── Détection des zones planes exploitables ──────────────────────────────────

# Couleurs de la couche « zones planes » — partagées Folium / matplotlib
COULEUR_ZONE_FILL = "#00BFFF"
COULEUR_ZONE_TRAIT = "#0080FF"

# Surface minimale par défaut (m²) — constante, non exposée en slider
SURFACE_MIN_ZONE = 1000.0


def detecter_zones_planes(
    pentes: np.ndarray,
    transform_l93,
    seuil_pente: float = 3.0,      # %
    largeur_min: float = 20.0,     # m
    surface_min: float = SURFACE_MIN_ZONE,   # m²
) -> list:
    """
    Détecte les zones suffisamment planes ET suffisamment larges pour
    accueillir une base vie ou une aire de stockage.

    Retourne une liste de dicts :
        {"geom": <shapely Polygon en L93>,
         "surface": <float m², surface RÉELLE après ouverture>}

    Méthode :
      1. Masque binaire pentes < seuil
      2. Ouverture morphologique raster (nettoie les pixels isolés)
      3. Labelling des composantes connexes
      4. Vectorisation
      5. Ouverture morphologique géométrique : buffer(-l/2).buffer(+l/2)
         → ne conserve que les parties où un disque de `largeur_min`
           de diamètre tient. Élimine les rubans et liserés, quelle que
           soit la forme globale de la zone (contrairement à un critère
           de rectangle englobant, qui laisse passer les croissants et
           rejette à tort les zones en L).
      6. Filtre sur la surface de la zone OUVERTE (pas la zone brute)

    IMPORTANT : appeler sur les pentes calculées depuis le MNT LISSÉ.
    Sur MNT brut, le bruit fragmente les zones et rend le résultat instable.
    """
    # 1-2. Masque + ouverture raster
    # errstate : la comparaison NaN < seuil est volontaire (donne False),
    # on évite juste le RuntimeWarning associé.
    with np.errstate(invalid="ignore"):
        masque = (~np.isnan(pentes)) & (pentes < seuil_pente)
    # structure 3x3 : retire les pixels isolés et les fils d'un pixel
    masque = binary_opening(masque, structure=np.ones((3, 3)))

    # 3. Composantes connexes
    labels, n = ndi_label(masque)
    if n == 0:
        return []

    # 4. Vectorisation (une passe sur le raster labellisé)
    zones = []
    r = largeur_min / 2.0
    for geom_json, val in rio_shapes(
        labels.astype(np.int32),
        mask=(labels > 0),
        transform=transform_l93,
    ):
        poly = shapely_shape(geom_json)
        if poly.area < surface_min:
            continue  # pré-filtre rapide avant l'opération coûteuse

        # 5. Ouverture géométrique
        ouverte = poly.buffer(-r).buffer(r)
        if ouverte.is_empty:
            continue

        # 6. Filtre surface sur la géométrie ouverte
        # (multipolygon possible : une zone en haltère peut donner 2 lobes)
        parts = ouverte.geoms if ouverte.geom_type == "MultiPolygon" else [ouverte]
        for part in parts:
            if part.area >= surface_min:
                zones.append({"geom": part, "surface": float(part.area)})

    # Tri par surface décroissante (les plus grandes d'abord)
    zones.sort(key=lambda z: -z["surface"])
    return zones


def libelle_zones_planes(
    seuil_pente: float,
    largeur_min: float,
    surface_min: float = SURFACE_MIN_ZONE,
) -> str:
    """Ligne de légende décrivant les paramètres actifs de détection."""
    return (
        f"Zones planes : pente < {seuil_pente:.1f} % · "
        f"largeur ≥ {largeur_min:.0f} m · "
        f"surface ≥ {surface_min:,.0f} m²".replace(",", " ")
    )


def resume_zones_planes(zones: list) -> str:
    """
    Récapitulatif texte des zones détectées, pour affichage sous les sliders.
    """
    if not zones:
        return (
            "Aucune zone ne satisfait ces critères. Essayer d'augmenter la pente "
            "maximale ou de réduire la largeur minimale."
        )
    surfaces = ", ".join(
        f"{z['surface']:,.0f} m²".replace(",", " ") for z in zones
    )
    pluriel = "s" if len(zones) > 1 else ""
    return f"{len(zones)} zone{pluriel} détectée{pluriel} : {surfaces}"


def _reproject_rgba_to_3857(
    rgba: np.ndarray,
    transform_l93,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Reprojette le raster RGBA de L93 (2154) vers Web Mercator (3857).
    Retourne (rgba_3857, (xmin, ymin, xmax, ymax) en 3857).
    """
    h, w = rgba.shape[:2]
    src_crs = CRS.from_epsg(2154)
    dst_crs = CRS.from_epsg(3857)

    x_min = transform_l93.c
    y_max = transform_l93.f
    x_max = x_min + w * transform_l93.a
    y_min = y_max + h * transform_l93.e

    t_dst, w_dst, h_dst = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, w, h,
        left=x_min, bottom=y_min, right=x_max, top=y_max,
    )
    dst = np.zeros((h_dst, w_dst, 4), dtype=np.uint8)
    for i in range(4):
        rasterio.warp.reproject(
            source=rgba[:, :, i],
            destination=dst[:, :, i],
            src_transform=transform_l93,
            src_crs=src_crs,
            dst_transform=t_dst,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.nearest,
        )

    bounds_3857 = rasterio.warp.transform_bounds(
        src_crs, dst_crs, x_min, y_min, x_max, y_max, densify_pts=21
    )
    return dst, bounds_3857


def _generer_courbes_l93(mnt: np.ndarray, transform_l93) -> tuple[list, list, int]:
    """
    Calcule les courbes de niveau en coordonnées L93 (pas WGS84 — évite double reprojection).
    Retourne (features_l93, labels_l93, intervalle).

    features_l93 : liste de dict {"coords_l93": [(x, y), ...], "niveau": float}
    labels_l93   : liste de dict {"x": float, "y": float, "niveau": float}
    """
    z_min = np.nanmin(mnt)
    z_max = np.nanmax(mnt)
    if np.isnan(z_min) or z_max - z_min < 1:
        return [], [], 0

    intervalle = max(1, round((z_max - z_min) / 20))
    niveaux = np.arange(
        np.ceil(z_min / intervalle) * intervalle,
        z_max,
        intervalle,
    )
    pas_label = 5 if intervalle <= 2 else 1
    niveaux_labellises = {float(n) for i, n in enumerate(niveaux) if i % pas_label == 0}

    raw_contours = []
    for niveau in niveaux:
        for contour in skimage_measure.find_contours(mnt, level=float(niveau)):
            if len(contour) >= 2:
                raw_contours.append((float(niveau), contour))

    if not raw_contours:
        return [], [], intervalle

    step = 3 if len(raw_contours) > 5000 else 1
    features = []
    labels = []

    for niveau, contour in raw_contours:
        coords_c = contour[::step]
        if len(coords_c) < 2:
            continue
        rows_c = gaussian_filter1d(coords_c[:, 0], sigma=2)
        cols_c = gaussian_filter1d(coords_c[:, 1], sigma=2)
        x_l93 = transform_l93.c + cols_c * transform_l93.a
        y_l93 = transform_l93.f + rows_c * transform_l93.e
        coords = list(zip(x_l93.tolist(), y_l93.tolist()))
        features.append({"coords_l93": coords, "niveau": niveau})
        if len(coords) >= 20 and niveau in niveaux_labellises:
            mid = len(coords) // 2
            labels.append({"x": coords[mid][0], "y": coords[mid][1], "niveau": niveau})

    return features, labels, intervalle


def _ajouter_echelle(ax, bounds_3857: tuple[float, float, float, float]) -> None:
    """
    Ajoute une barre d'échelle.
    Utilise matplotlib-scalebar si disponible, sinon dessine manuellement
    une barre calibrée depuis la largeur de la vue en mètres.
    """
    if _HAS_SCALEBAR:
        sb = ScaleBar(1.0, units="m", location="lower left",
                      box_alpha=0.6, font_properties={"size": 8})
        ax.add_artist(sb)
        return

    # Barre manuelle — longueur cible ~15 % de la largeur de la vue
    xmin, ymin, xmax, ymax = bounds_3857
    largeur_m = xmax - xmin  # EPSG:3857 : unité = mètre (approximativement)

    cibles = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    cible_m = min(cibles, key=lambda v: abs(v - largeur_m * 0.15))

    frac = cible_m / largeur_m
    x0 = xmin + 0.03 * largeur_m
    x1 = x0 + cible_m
    y0 = ymin + 0.04 * (ymax - ymin)

    ax.plot([x0, x1], [y0, y0], color="white", linewidth=3, solid_capstyle="butt",
            transform=ax.transData, zorder=6)
    ax.plot([x0, x1], [y0, y0], color="black", linewidth=1.5, solid_capstyle="butt",
            transform=ax.transData, zorder=7)

    label = f"{cible_m:,} m".replace(",", " ")
    ax.text((x0 + x1) / 2, y0 + (ymax - ymin) * 0.015, label,
            ha="center", va="bottom", fontsize=7, color="white",
            fontweight="bold",
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=2, foreground="black")
            ],
            zorder=8)


def _ajouter_logo(fig, logo_path: str | None) -> None:
    """Insère le logo UNITe en coin supérieur droit via fig.figimage()."""
    if logo_path is None or not os.path.exists(logo_path):
        return
    try:
        logo = Image.open(logo_path).convert("RGBA")
        # Redimensionne à ~70 px de haut pour le PNG 200 dpi
        target_h = 70
        ratio = target_h / logo.height
        logo = logo.resize(
            (int(logo.width * ratio), target_h), Image.LANCZOS
        )
        arr = np.array(logo)
        fig_w_px = int(fig.get_figwidth() * fig.dpi)
        fig_h_px = int(fig.get_figheight() * fig.dpi)
        x0 = fig_w_px - arr.shape[1] - 20
        y0 = fig_h_px - arr.shape[0] - 20
        fig.figimage(arr, xo=x0, yo=y0, zorder=10, origin="upper")
    except Exception:
        pass  # logo non critique — on continue sans


# ── Tableau des seuils matplotlib ────────────────────────────────────────────

def _construire_donnees_tableau(technologie: str, puissance: str) -> tuple[list, list]:
    """Retourne (col_labels, rows) pour ax.table()."""
    table = _SEUILS[(technologie, puissance)]
    col_labels = ["Orientation", "Favorable", "Acceptable", "Contraignant", "Exclusion"]
    rows = []
    for orient_label, orient_key in [("Nord", "N"), ("Sud", "S"), ("Est / Ouest", "EO")]:
        s = table[orient_key]["seuils"]
        note = table[orient_key]["note_contraignant"]
        cont = f"{s[1]}–{s[2]} %"
        if note:
            cont += f"\n({note})"
        rows.append([
            orient_label,
            f"< {s[0]} %",
            f"{s[0]}–{s[1]} %",
            cont,
            f"> {s[2]} %",
        ])
    return col_labels, rows


_COULEURS_TABLEAU = {
    "Favorable":    "#57bb57",
    "Acceptable":   "#f5e642",
    "Contraignant": "#f5a623",
    "Exclusion":    "#e8342a",
}

# Colonnes indexées 1-4 correspondent aux classes
_COL_COULEURS = [None, "#57bb57", "#f5e642", "#f5a623", "#e8342a"]


# ── Fonction principale ───────────────────────────────────────────────────────

def generer_image_carte(
    pentes: np.ndarray,
    orientations: np.ndarray,
    transform_l93,
    gdf_site: gpd.GeoDataFrame,
    mnt_brut: np.ndarray | None = None,
    technologie: str = "FIXE",
    puissance: str = "<5MWc",
    logo_path: str | None = None,
    dpi: int = 200,
    avec_courbes: bool = True,
    nom_fichier: str | None = None,
    mode: str = MODE_CONSTRUCTIBILITE,
    zones_planes: list | None = None,
    zones_params: tuple | None = None,
    couches: dict | None = None,
) -> bytes:
    """
    Génère la carte statique PNG.

    mode="constructibilite" : classification par (pente, orientation) selon
        les seuils projet — palette vert/jaune/orange/rouge. Le tableau des
        seuils par orientation est intégré sous la carte (même figure).
    mode="topographie" : classification par pente seule, seuils universels
        <5 / 5-10 / 10-15 / >15 % — palette séquentielle ocre. Pas de tableau :
        la légende suffit (les seuils par orientation n'ont pas de sens ici).

    zones_planes / zones_params : couche « zones planes exploitables », rendue
        en mode topographie uniquement. zones_params = (seuil_pente, largeur_min)
        alimente la ligne de légende décrivant les critères actifs.

    couches : dict {nom: bool} parmi COUCHES. Une couche désactivée n'est ni
        dessinée ni mentionnée en légende — l'export reflète alors exactement
        ce que l'utilisateur a coché à l'écran. Toutes actives par défaut.

    Travaille entièrement en EPSG:3857 (Web Mercator) car contextily l'exige
    pour le fond satellite.

    Retourne les bytes PNG.
    """
    import matplotlib.patheffects as pe

    if mode not in (MODE_CONSTRUCTIBILITE, MODE_TOPOGRAPHIE):
        raise ValueError(
            f"mode inconnu : {mode!r} "
            f"(attendu {MODE_CONSTRUCTIBILITE!r} ou {MODE_TOPOGRAPHIE!r})"
        )
    avec_tableau = mode == MODE_CONSTRUCTIBILITE

    couches = {**couches_par_defaut(), **(couches or {})}

    # ── 1. Reprojections en EPSG:3857 ────────────────────────────────────────
    gdf_3857 = gdf_site.to_crs("EPSG:3857")

    # Raster RGBA pentes → 3857 — la classification dépend du mode
    if mode == MODE_CONSTRUCTIBILITE:
        classes = _classifier_pente_orientee(
            pentes, orientations, technologie, puissance
        )
        palette = _PALETTE
    else:
        classes = classifier_pente_topo(pentes)
        palette = {i: _hex_to_rgb(c) for i, c in enumerate(_PALETTE_TOPO)}

    rgba = _to_rgba(classes, palette=palette)
    rgba_3857, raster_bounds_3857 = _reproject_rgba_to_3857(rgba, transform_l93)
    rx0, ry0, rx1, ry1 = raster_bounds_3857

    # ── 1b. Emprise affichée : raster élargi d'une marge, pour que le contour
    #        du site ne touche jamais les bords de l'image ──────────────────
    MARGE = 0.07  # 7 % de la plus grande dimension
    delta = max(rx1 - rx0, ry1 - ry0) * MARGE
    vx0, vy0, vx1, vy1 = rx0 - delta, ry0 - delta, rx1 + delta, ry1 + delta

    # Courbes de niveau en L93, puis reprojection des coordonnées vers 3857
    courbes_features: list = []
    courbes_labels: list = []
    if avec_courbes and couches["courbes"] and mnt_brut is not None:
        courbes_l93, labels_l93, _ = _generer_courbes_l93(mnt_brut, transform_l93)
        tr_l93_to_3857 = Transformer.from_crs("EPSG:2154", "EPSG:3857", always_xy=True)
        for feat in courbes_l93:
            xs, ys = zip(*feat["coords_l93"]) if feat["coords_l93"] else ([], [])
            if not xs:
                continue
            x3857, y3857 = tr_l93_to_3857.transform(list(xs), list(ys))
            courbes_features.append({"coords": list(zip(x3857, y3857)), "niveau": feat["niveau"]})
        for lbl in labels_l93:
            x3857, y3857 = tr_l93_to_3857.transform(lbl["x"], lbl["y"])
            courbes_labels.append({"x": x3857, "y": y3857, "niveau": lbl["niveau"]})

    # Contour du site en 3857
    site_geoms_3857 = list(gdf_3857.geometry)

    # ── 2. Figure : carte (haut) + tableau (bas, mode constructibilité seul) ─
    # La largeur de la carte est fixe (12 po) ; la hauteur est calculée depuis
    # le ratio géographique de l'emprise affichée pour éviter toute déformation.
    MAP_W_IN = 12.0
    TABLE_H_IN = 2.5 if avec_tableau else 0.0
    TITLE_H_IN = 0.35

    geo_ratio = (vx1 - vx0) / max(vy1 - vy0, 1)  # largeur / hauteur en mètres
    map_h_in = MAP_W_IN / geo_ratio
    fig_h_in = map_h_in + TABLE_H_IN + TITLE_H_IN

    fig = plt.figure(figsize=(MAP_W_IN, fig_h_in), dpi=dpi)

    # Titre en haut (nom du fichier source)
    if nom_fichier:
        nom_base = os.path.splitext(os.path.basename(nom_fichier))[0]
        titre_fig = f"Extraction Topo RGE ALTI IGN — {nom_base}"
        fig.suptitle(titre_fig, fontsize=11, fontweight="bold", y=0.995, va="top")

    # Marges exprimées en fraction de la hauteur totale
    top_margin = 1.0 - TITLE_H_IN / fig_h_in
    bottom_margin = 0.01

    if avec_tableau:
        gs = fig.add_gridspec(
            2, 1,
            height_ratios=[map_h_in, TABLE_H_IN],
            hspace=0.04,
            left=0.01, right=0.99,
            top=top_margin, bottom=bottom_margin,
        )
        ax_carte = fig.add_subplot(gs[0])
        ax_table = fig.add_subplot(gs[1])
    else:
        gs = fig.add_gridspec(
            1, 1,
            left=0.01, right=0.99,
            top=top_margin, bottom=bottom_margin,
        )
        ax_carte = fig.add_subplot(gs[0])
        ax_table = None

    # ── 3. Fond satellite ESRI (contextily) ───────────────────────────────────
    # Le fond est ajouté AVANT le raster pour que les pentes soient au-dessus.
    # Les limites sont posées sur l'emprise élargie AVANT add_basemap : contextily
    # lit les limites courantes pour choisir les tuiles à télécharger. Elles sont
    # réappliquées ensuite car add_basemap peut les ajuster au bord des tuiles.
    ax_carte.set_xlim(vx0, vx1)
    ax_carte.set_ylim(vy0, vy1)

    if _HAS_CTX:
        try:
            cx.add_basemap(
                ax_carte,
                source=cx.providers.Esri.WorldImagery,
                crs="EPSG:3857",
                zorder=0,
                attribution=False,
            )
        except Exception as exc:
            # Connexion réseau indisponible → fond gris avec message
            ax_carte.set_facecolor("#555555")
            ax_carte.text(
                0.5, 0.5,
                f"Fond satellite indisponible\n(vérifier la connexion internet)\n{exc}",
                transform=ax_carte.transAxes,
                ha="center", va="center",
                fontsize=9, color="white",
                bbox=dict(boxstyle="round", facecolor="#333", alpha=0.7),
            )
    else:
        ax_carte.set_facecolor("#555555")
        ax_carte.text(
            0.5, 0.5,
            "contextily non installé — pip install contextily",
            transform=ax_carte.transAxes,
            ha="center", va="center",
            fontsize=9, color="white",
        )

    # ── 4. Overlay pentes ─────────────────────────────────────────────────────
    if couches["pentes"]:
        ax_carte.imshow(
            rgba_3857,
            extent=[rx0, rx1, ry0, ry1],
            origin="upper",
            aspect="auto",
            zorder=2,
            interpolation="nearest",
        )

    # Réapplication de l'emprise élargie : add_basemap et imshow ont pu la modifier
    ax_carte.set_xlim(vx0, vx1)
    ax_carte.set_ylim(vy0, vy1)

    # ── 5. Courbes de niveau ──────────────────────────────────────────────────
    for feat in courbes_features:
        if len(feat["coords"]) < 2:
            continue
        xs, ys = zip(*feat["coords"])
        ax_carte.plot(xs, ys, color="#000000", linewidth=0.5, alpha=0.55, zorder=3)

    for lbl in courbes_labels:
        ax_carte.text(
            lbl["x"], lbl["y"],
            str(int(lbl["niveau"])),
            fontsize=6,
            ha="center", va="center",
            color="#111111",
            fontweight="bold",
            zorder=4,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="white")],
        )

    # ── 6. Contour du site — jaune pointillé ─────────────────────────────────
    for geom in site_geoms_3857 if couches["contour"] else []:
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            _polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            _polys = list(geom.geoms)
        else:
            _polys = []
        for poly in _polys:
            x_ext, y_ext = poly.exterior.xy
            ax_carte.plot(
                x_ext, y_ext,
                color="#FFE600", linewidth=2.0, linestyle="--",
                zorder=5,
            )

    # ── 6b. Zones planes exploitables — mode topographie uniquement ──────────
    zones_affichees = []
    if mode == MODE_TOPOGRAPHIE and zones_planes and couches["zones"]:
        gdf_zones = gpd.GeoDataFrame(
            geometry=[z["geom"] for z in zones_planes], crs="EPSG:2154"
        ).to_crs("EPSG:3857")
        for geom in gdf_zones.geometry:
            if geom is None or geom.is_empty:
                continue
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
            for poly in polys:
                x_ext, y_ext = poly.exterior.xy
                ax_carte.fill(
                    x_ext, y_ext,
                    facecolor=COULEUR_ZONE_FILL, alpha=0.35,
                    edgecolor=COULEUR_ZONE_TRAIT, linewidth=1.6,
                    zorder=6,
                )
        zones_affichees = zones_planes

    # ── 7. Légende ────────────────────────────────────────────────────────────
    if mode == MODE_CONSTRUCTIBILITE:
        labels_legende = ["Favorable", "Acceptable", "Contraignant", "Exclusion possible"]
        couleurs_legende = ["#%02x%02x%02x" % _PALETTE[i] for i in range(4)]
        titre_legende = "Pente — seuils projet"
    else:
        labels_legende = _LABELS_TOPO
        couleurs_legende = _PALETTE_TOPO
        titre_legende = "Pente du terrain"

    # Ne listent que les couches effectivement dessinées
    legend_elements = [
        mpatches.Patch(facecolor=c, label=lbl)
        for c, lbl in zip(couleurs_legende, labels_legende)
    ] if couches["pentes"] else []

    if couches["contour"]:
        legend_elements.append(
            Line2D([0], [0], color="#FFE600", linewidth=2, linestyle="--",
                   label="Contour du site")
        )
    if zones_affichees:
        seuil_z, largeur_z = zones_params or (3.0, 20.0)
        legend_elements.append(
            mpatches.Patch(
                facecolor=COULEUR_ZONE_FILL, alpha=0.35,
                edgecolor=COULEUR_ZONE_TRAIT, linewidth=1.6,
                label=libelle_zones_planes(seuil_z, largeur_z),
            )
        )
    # Aucune couche cochée : pas de boîte de légende vide
    if legend_elements:
        legend = ax_carte.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=8,
            framealpha=0.85,
            edgecolor="#cccccc",
            title=titre_legende if couches["pentes"] else None,
            title_fontsize=8,
        )
        legend.get_frame().set_linewidth(0.8)
        # Au-dessus des couches (zones planes en zorder 6) — sinon recouverte
        legend.set_zorder(11)

    # ── 7b. Bandeau de titre en surimpression — lève l'ambiguïté entre modes ──
    if mode == MODE_CONSTRUCTIBILITE:
        titre_bandeau = f"Constructibilité — {technologie} · {puissance}"
    else:
        titre_bandeau = "Pente du terrain — toutes orientations"

    ax_carte.text(
        0.5, 0.985,
        titre_bandeau,
        transform=ax_carte.transAxes,
        ha="center", va="top",
        fontsize=10, fontweight="bold", color="#111111",
        zorder=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white", alpha=0.82,
            edgecolor="#bbbbbb", linewidth=0.8,
        ),
    )

    # ── 8. Barre d'échelle ────────────────────────────────────────────────────
    _ajouter_echelle(ax_carte, (vx0, vy0, vx1, vy1))

    # ── 9. Logo UNITe ─────────────────────────────────────────────────────────
    # fig.figimage() positionne après le rendu donc on passe fig
    # (appel différé ci-dessous, après fig.savefig qui flush)

    # ── 10. Axes carte off ────────────────────────────────────────────────────
    ax_carte.set_axis_off()

    # ── 11. Tableau des seuils — mode constructibilité uniquement ────────────
    # En mode topographie les seuils sont universels : la légende suffit.
    if not avec_tableau:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, facecolor="white")
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    ax_table.set_axis_off()
    col_labels, rows = _construire_donnees_tableau(technologie, puissance)

    tbl = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Hauteur de ligne adaptée au contenu : on détecte les cellules multilignes
    # (présence de "\n") et on leur applique une hauteur proportionnelle
    BASE_H = 0.22   # hauteur de ligne standard (fraction de l'axe)
    LINE_H = 0.13   # hauteur supplémentaire par ligne additionnelle
    for row_idx in range(len(rows)):
        max_lines = max(
            cell_text.count("\n") + 1 for cell_text in rows[row_idx]
        )
        cell_h = BASE_H + (max_lines - 1) * LINE_H
        for col_idx in range(len(col_labels)):
            tbl[row_idx + 1, col_idx].set_height(cell_h)

    # Couleurs d'en-tête : gris pour "Orientation", couleur de classe pour les 4 suivantes
    for col_idx, label in enumerate(col_labels):
        cell = tbl[0, col_idx]
        couleur_header = _COL_COULEURS[col_idx] if col_idx >= 1 else None
        cell.set_facecolor(couleur_header if couleur_header else "#DDDDDD")
        cell.set_alpha(0.85)
        cell.set_text_props(fontweight="bold")

    # Titre du tableau
    note_globale = _SEUILS[(technologie, puissance)].get("note_globale")
    titre = f"Seuils de pente — {technologie} · {puissance}"
    ax_table.set_title(titre, fontsize=9, fontweight="bold", pad=4)
    if note_globale:
        fig.text(
            0.5, 0.005,
            f"ℹ {note_globale}",
            ha="center", va="bottom",
            fontsize=6.5, color="#555555",
            wrap=True,
        )

    # ── 12. Export PNG en bytes ───────────────────────────────────────────────
    buf = io.BytesIO()
    # On sauvegarde d'abord pour connaître les dimensions en pixels
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, facecolor="white")
    buf.seek(0)
    plt.close(fig)

    return buf.getvalue()


def png_vers_pdf(png_bytes: bytes) -> bytes:
    """Convertit un PNG (bytes) en PDF mono-page via PIL."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=200)
    return buf.getvalue()


# ── Script de test autonome ───────────────────────────────────────────────────

def _generer_donnees_test() -> tuple:
    """
    Génère un MNT synthétique (colline gaussienne) centré sur Ancy-le-Libre
    (approx. L93 : x=742 000, y=6 715 000) pour tester le rendu sans réseau IGN.

    Retourne (pentes, orientations, transform_l93, gdf_site, mnt_brut).
    """
    from scipy.ndimage import gaussian_filter as gf

    # Zone de ~500 × 500 m
    xmin, ymin = 742_000.0, 6_715_000.0
    xmax, ymax = xmin + 500, ymin + 500
    resolution = 5  # m
    cols = int((xmax - xmin) / resolution)
    rows = int((ymax - ymin) / resolution)

    transform = rt_from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    # MNT synthétique : fond plat à 200 m + colline gaussienne + légère crête
    xx, yy = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    mnt = (
        200
        + 25 * np.exp(-(xx**2 + yy**2) / 0.3)        # colline centrale
        + 8 * np.exp(-((xx - 0.5)**2 + (yy + 0.3)**2) / 0.1)  # crête secondaire
        + np.random.default_rng(42).normal(0, 0.3, (rows, cols))
    ).astype(np.float32)

    # Lissage pour des pentes réalistes
    mnt = gf(mnt, sigma=3.0)

    # Calcul pentes/orientations (Horn)
    res_x = abs(transform.a)
    res_y = abs(transform.e)
    pad = np.pad(mnt, 1, mode="edge")
    dzdx = (
        (pad[:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:])
        - (pad[:-2, :-2] + 2 * pad[1:-1, :-2] + pad[2:, :-2])
    ) / (8 * res_x)
    dzdy = (
        (pad[2:, :-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:])
        - (pad[:-2, :-2] + 2 * pad[:-2, 1:-1] + pad[:-2, 2:])
    ) / (8 * res_y)

    pentes = np.tan(np.arctan(np.sqrt(dzdx**2 + dzdy**2))) * 100
    azimut = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360
    _CARDS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    idx = np.round(azimut / 45).astype(int) % 8
    orientations = np.array(_CARDS)[idx]

    # GeoDataFrame site : carré central 300 × 300 m
    cx_l93 = (xmin + xmax) / 2
    cy_l93 = (ymin + ymax) / 2
    site_poly = box(cx_l93 - 150, cy_l93 - 150, cx_l93 + 150, cy_l93 + 150)
    gdf_site = gpd.GeoDataFrame(geometry=[site_poly], crs="EPSG:2154")

    return pentes.astype(np.float32), orientations, transform, gdf_site, mnt


if __name__ == "__main__":
    import matplotlib.patheffects  # noqa: F401 — assure que le module est chargé

    print("Generation des donnees de test (MNT synthetique - Ancy-le-Libre)...")
    pentes, orientations, transform, gdf_site, mnt_brut = _generer_donnees_test()

    logo_path = os.path.join(os.path.dirname(__file__), "logo_unite.png")

    technologie = "FIXE"
    puissance = "≥5MWc"  # ≥5MWc

    print("Parametres : %s / %s" % (technologie, puissance.encode("ascii", "replace").decode()))

    # ── Detection des zones planes : plusieurs jeux de parametres ────────────
    print("\nZones planes exploitables :")
    for seuil_zp, largeur_zp in ((3.0, 20.0), (5.0, 20.0), (5.0, 30.0)):
        t0 = time.time()
        zones_test = detecter_zones_planes(
            pentes, transform, seuil_pente=seuil_zp, largeur_min=largeur_zp
        )
        dt = time.time() - t0
        detail = (
            ", ".join("%.0f m2" % z["surface"] for z in zones_test)
            if zones_test else "aucune"
        )
        print(
            "  pente < %.1f %% / largeur >= %2.0f m -> %d zone(s) en %.2f s : %s"
            % (seuil_zp, largeur_zp, len(zones_test), dt, detail)
        )

    # Parametres retenus pour l'export de test
    SEUIL_ZP, LARGEUR_ZP = 3.0, 20.0
    zones = detecter_zones_planes(
        pentes, transform, seuil_pente=SEUIL_ZP, largeur_min=LARGEUR_ZP
    )

    print("\nGeneration des cartes (fond satellite ESRI - connexion requise)...")

    # Ordre d'affichage : topographie d'abord, constructibilite ensuite.
    # Les deux modes partagent MNT, transform et contour : seule la
    # classification et la palette changent.
    for mode, base in (
        (MODE_TOPOGRAPHIE, "test_topographie"),
        (MODE_CONSTRUCTIBILITE, "test_constructibilite"),
    ):
        t0 = time.time()
        png_bytes = generer_image_carte(
            pentes=pentes,
            orientations=orientations,
            transform_l93=transform,
            gdf_site=gdf_site,
            mnt_brut=mnt_brut,
            technologie=technologie,
            puissance=puissance,
            logo_path=logo_path,
            dpi=200,
            avec_courbes=True,
            nom_fichier="Ancy-le-Libre-contours.zip",
            mode=mode,
            zones_planes=zones,
            zones_params=(SEUIL_ZP, LARGEUR_ZP),
        )
        duree = time.time() - t0

        with open(base + ".png", "wb") as f:
            f.write(png_bytes)
        pdf_bytes = png_vers_pdf(png_bytes)
        with open(base + ".pdf", "wb") as f:
            f.write(pdf_bytes)

        print(
            "  %-18s -> %s.png (%.0f Ko) + %s.pdf (%.0f Ko) en %.1f s"
            % (mode, base, len(png_bytes) / 1024, base, len(pdf_bytes) / 1024, duree)
        )
        if duree > 5:
            print("    [!] > 5 s - prevoir un st.spinner() cote Streamlit.")

    # Cas limite : courbes de niveau desactivees
    print("Test sans courbes de niveau...")
    png_no_curves = generer_image_carte(
        pentes=pentes,
        orientations=orientations,
        transform_l93=transform,
        gdf_site=gdf_site,
        mnt_brut=None,
        technologie=technologie,
        puissance=puissance,
        logo_path=logo_path,
        dpi=200,
        avec_courbes=False,
        nom_fichier="Ancy-le-Libre-contours.zip",
    )
    with open("test_sans_courbes.png", "wb") as f:
        f.write(png_no_curves)
    print("  test_sans_courbes.png ecrit")

    print(
        "\nTermine - ouvrir test_constructibilite.png/.pdf et "
        "test_topographie.png/.pdf pour inspection visuelle."
    )
