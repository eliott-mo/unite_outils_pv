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
from skimage import measure as skimage_measure
import geopandas as gpd
from shapely.geometry import box

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


def _to_rgba(classes: np.ndarray, alpha: int = 166) -> np.ndarray:
    """alpha=166 ≈ 0.65 × 255."""
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for val, (r, g, b) in _PALETTE.items():
        rgba[classes == val] = [r, g, b, alpha]
    return rgba


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
) -> bytes:
    """
    Génère la carte statique PNG (carte + tableau des seuils).

    Travaille entièrement en EPSG:3857 (Web Mercator) car contextily l'exige
    pour le fond satellite. Le tableau des seuils est intégré dans la même figure
    (Option A recommandée : un seul fichier, idéal pour le PDF).

    Retourne les bytes PNG.
    """
    import matplotlib.patheffects as pe

    # ── 1. Reprojections en EPSG:3857 ────────────────────────────────────────
    gdf_3857 = gdf_site.to_crs("EPSG:3857")
    union_3857 = gdf_3857.union_all()
    b = union_3857.bounds                  # (xmin, ymin, xmax, ymax) en 3857
    site_bounds_3857 = b

    # Raster RGBA pentes → 3857
    classes = _classifier_pente_orientee(pentes, orientations, technologie, puissance)
    rgba = _to_rgba(classes)
    rgba_3857, raster_bounds_3857 = _reproject_rgba_to_3857(rgba, transform_l93)
    rx0, ry0, rx1, ry1 = raster_bounds_3857

    # Courbes de niveau en L93, puis reprojection des coordonnées vers 3857
    courbes_features: list = []
    courbes_labels: list = []
    if avec_courbes and mnt_brut is not None:
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

    # ── 2. Figure avec deux sous-graphes : carte (haut) + tableau (bas) ──────
    # Proportions : carte occupe ~80 % de la hauteur, tableau ~20 %
    fig = plt.figure(figsize=(12, 11), dpi=dpi)

    # Titre en haut (nom du fichier source)
    if nom_fichier:
        nom_base = os.path.splitext(os.path.basename(nom_fichier))[0]
        titre_fig = f"Extraction Topo RGE ALTI IGN — {nom_base}"
        fig.suptitle(titre_fig, fontsize=11, fontweight="bold", y=0.995, va="top")
        top_margin = 0.965
    else:
        top_margin = 0.97

    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[8, 2],
        hspace=0.05,
        left=0.01, right=0.99,
        top=top_margin, bottom=0.01,
    )
    ax_carte = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # ── 3. Fond satellite ESRI (contextily) ───────────────────────────────────
    # Le fond est ajouté AVANT le raster pour que les pentes soient au-dessus.
    # On fixe les limites de l'axe sur les bounds du raster (qui inclut le buffer).
    ax_carte.set_xlim(rx0, rx1)
    ax_carte.set_ylim(ry0, ry1)

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
    ax_carte.imshow(
        rgba_3857,
        extent=[rx0, rx1, ry0, ry1],
        origin="upper",
        aspect="auto",
        zorder=2,
        interpolation="nearest",
    )

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
    for geom in site_geoms_3857:
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

    # ── 7. Légende ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#%02x%02x%02x" % _PALETTE[0], label="Favorable"),
        mpatches.Patch(facecolor="#%02x%02x%02x" % _PALETTE[1], label="Acceptable"),
        mpatches.Patch(facecolor="#%02x%02x%02x" % _PALETTE[2], label="Contraignant"),
        mpatches.Patch(facecolor="#%02x%02x%02x" % _PALETTE[3], label="Exclusion possible"),
        Line2D([0], [0], color="#FFE600", linewidth=2, linestyle="--", label="Contour du site"),
    ]
    legend = ax_carte.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=8,
        framealpha=0.85,
        edgecolor="#cccccc",
        title="Pente — seuils projet",
        title_fontsize=8,
    )
    legend.get_frame().set_linewidth(0.8)

    # ── 8. Barre d'échelle ────────────────────────────────────────────────────
    _ajouter_echelle(ax_carte, (rx0, ry0, rx1, ry1))

    # ── 9. Logo UNITe ─────────────────────────────────────────────────────────
    # fig.figimage() positionne après le rendu donc on passe fig
    # (appel différé ci-dessous, après fig.savefig qui flush)

    # ── 10. Axes carte off ────────────────────────────────────────────────────
    ax_carte.set_axis_off()

    # ── 11. Tableau des seuils ────────────────────────────────────────────────
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
    print("Generation carte PNG (fond satellite ESRI - connexion requise)...")

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
    )

    t1 = time.time()
    duree = t1 - t0
    if duree > 5:
        print("[!] Rendu en %.1f s - prevoir un st.spinner() cote Streamlit." % duree)
    else:
        print("Rendu en %.1f s." % duree)

    png_path = "test_export.png"
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    print("PNG ecrit : %s (%.0f Ko)" % (png_path, len(png_bytes) / 1024))

    print("Conversion PDF...")
    pdf_bytes = png_vers_pdf(png_bytes)
    pdf_path = "test_export.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print("PDF ecrit : %s (%.0f Ko)" % (pdf_path, len(pdf_bytes) / 1024))

    # Test sans courbes de niveau
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
    )
    with open("test_export_sans_courbes.png", "wb") as f:
        f.write(png_no_curves)
    print("PNG sans courbes ecrit : test_export_sans_courbes.png")

    print("\nTermine - ouvrir test_export.png et test_export.pdf pour inspection visuelle.")
