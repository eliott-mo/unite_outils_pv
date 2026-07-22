"""Données Topo ESQ/APS — Extraction RGE ALTI IGN + Conversion données drone UNITe."""

import io
import os
import base64
import time
import zipfile
import tempfile
import json
import math
import inspect
import hashlib

import numpy as np
import geopandas as gpd
import folium
import requests
import streamlit as st
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds as rt_from_bounds
from rasterio.crs import CRS
from rasterio.features import rasterize as rio_rasterize
import rasterio.warp
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import griddata as scipy_griddata
from shapely.geometry import MultiPoint
from skimage import measure as skimage_measure
from streamlit_folium import st_folium
from export_carte import (
    generer_image_carte,
    png_vers_pdf,
    classifier_pente_topo,
    detecter_zones_planes,
    resume_zones_planes,
    MODE_CONSTRUCTIBILITE,
    MODE_TOPOGRAPHIE,
    COUCHES,
    LIBELLES_COUCHES,
    couches_par_defaut,
    COULEUR_ZONE_FILL,
    COULEUR_ZONE_TRAIT,
    SURFACE_MIN_ZONE,
    _PALETTE_TOPO,
    _LABELS_TOPO,
)
from affine import Affine

# ── Constantes ────────────────────────────────────────────────────────────────
_URL_ALTI = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
_BATCH_SIZE = 5000
_RATE_LIMIT = 0.2  # secondes entre batches (≤ 5 req/s)

# Palette pente — seuils Fixe (identiques à PV Topo Analyzer)
_PALETTE = {
    0: (76, 175, 80),   # vert   — Favorable
    1: (255, 193, 7),   # jaune  — Acceptable
    2: (255, 152, 0),   # orange — Contraignant
    3: (244, 67, 54),   # rouge  — Exclusion possible
}

# Seuils de pente par (technologie, puissance, orientation)
# Format seuils : [favorable_max, acceptable_max, contraignant_max]
# Au-delà de contraignant_max → Exclusion
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


# ── 1. Chargement shapefile ZIP ───────────────────────────────────────────────

def load_shapefile_from_zip(uploaded_file) -> gpd.GeoDataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as zf:
            shp_files = [n for n in zf.namelist() if n.lower().endswith(".shp")]
            if not shp_files:
                raise ValueError("Aucun fichier .shp dans le ZIP.")
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
    except zipfile.BadZipFile:
        raise ValueError("Le fichier n'est pas un ZIP valide.")

    if gdf.crs is None:
        st.warning("CRS absent dans le shapefile — WGS84 (EPSG:4326) supposé.")
        gdf = gdf.set_crs("EPSG:4326")
    if gdf.empty:
        raise ValueError("Le shapefile est vide.")
    return gdf


# ── 2. Bbox Lambert 93 avec buffer ───────────────────────────────────────────

def calculer_bbox_l93(gdf: gpd.GeoDataFrame, buffer_m: int) -> tuple:
    """Retourne (xmin, ymin, xmax, ymax) en Lambert 93 avec buffer."""
    gdf_l93 = gdf.to_crs("EPSG:2154")
    union = gdf_l93.union_all()
    buffered = union.buffer(buffer_m) if buffer_m > 0 else union
    return buffered.bounds  # (xmin, ymin, xmax, ymax)


# ── 3. Téléchargement MNT via API IGN REST ───────────────────────────────────

def _alti_batch(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    payload = {
        "lon":       "|".join(f"{v:.7f}" for v in lons),
        "lat":       "|".join(f"{v:.7f}" for v in lats),
        "resource":  "ign_rge_alti_wld",
        "delimiter": "|",
        "zonly":     "true",
    }
    resp = requests.post(_URL_ALTI, json=payload, timeout=30)
    resp.raise_for_status()
    elevations = resp.json().get("elevations", [])
    if not elevations:
        raise RuntimeError(f"Réponse vide de l'API IGN : {resp.text[:200]}")
    z = np.array(elevations, dtype=np.float32)
    z[z <= -99000] = np.nan
    return z


def fetch_mnt(bbox_l93: tuple, resolution: int) -> tuple:
    """
    Télécharge le MNT RGE ALTI via l'API altimétrique REST IGN.
    Retourne (mnt_array float32, transform_l93 Affine).
    """
    xmin, ymin, xmax, ymax = bbox_l93
    cols = max(2, int((xmax - xmin) / resolution))
    rows = max(2, int((ymax - ymin) / resolution))

    # Sécurité : limiter à 2000×2000 points
    max_dim = 2000
    if cols > max_dim or rows > max_dim:
        facteur = max(cols, rows) / max_dim
        resolution = resolution * facteur
        cols = max(2, int((xmax - xmin) / resolution))
        rows = max(2, int((ymax - ymin) / resolution))
        st.warning(
            f"Zone trop grande — résolution ajustée à {resolution:.1f} m "
            f"({rows}×{cols} = {rows * cols:,} points)."
        )

    # Transform L93 — origine au coin SW, axe Y décroissant (nord→sud)
    transform = rt_from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    # Centres de pixels en L93
    col_idx = np.tile(np.arange(cols), rows)
    row_idx = np.repeat(np.arange(rows), cols)
    x_l93 = transform.c + (col_idx + 0.5) * transform.a
    y_l93 = transform.f + (row_idx + 0.5) * transform.e  # transform.e < 0

    # Conversion L93 → WGS84 pour l'API IGN
    _tr = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lons, lats = _tr.transform(x_l93, y_l93)

    n_points = rows * cols
    n_batches = (n_points + _BATCH_SIZE - 1) // _BATCH_SIZE
    z_all = np.full(n_points, np.nan, dtype=np.float32)

    bar = st.progress(
        0.0,
        text=f"Altimétrie IGN — batch 1/{n_batches} ({n_points:,} points à {resolution:.0f} m)…",
    )
    try:
        for i in range(n_batches):
            debut, fin = i * _BATCH_SIZE, min((i + 1) * _BATCH_SIZE, n_points)
            bar.progress(
                i / n_batches,
                text=f"Altimétrie IGN — batch {i + 1}/{n_batches} ({fin:,}/{n_points:,} points)…",
            )
            z_all[debut:fin] = _alti_batch(lons[debut:fin], lats[debut:fin])
            if i < n_batches - 1:
                time.sleep(_RATE_LIMIT)

        bar.progress(1.0, text=f"MNT téléchargé — {n_points:,} points à {resolution:.0f} m")
    except requests.exceptions.Timeout:
        bar.empty()
        raise RuntimeError("Délai dépassé — essayez la résolution 5 m ou réduisez le buffer.")
    except requests.exceptions.RequestException as e:
        bar.empty()
        raise RuntimeError(f"Erreur réseau API IGN : {e}")
    except Exception:
        bar.empty()
        raise

    return z_all.reshape(rows, cols), transform


# ── 4. Détection résolution réelle (1m natif vs 5m rééchantillonné) ──────────

def detecter_resolution_reelle(mnt: np.ndarray) -> bool:
    """
    Retourne True si les données sont du vrai 1 m, False si interpolé depuis 5 m.

    Signal : quand l'API IGN renvoie du 5 m rééchantillonné en nearest-neighbour,
    ~75 % des transitions entre pixels consécutifs sont nulles (blocs de 5 pixels
    identiques). Sur de vraies données 1 m, ce taux est proche de 0 %.

    Teste sur la ligne centrale du raster pour éviter les bords NaN.
    Seuil calibré empiriquement sur test Guadeloupe (hors couverture 1 m).
    """
    ligne = mnt[mnt.shape[0] // 2, :]
    ligne = ligne[~np.isnan(ligne)]
    if len(ligne) < 20:
        return True  # pas assez de points → confiance par défaut
    taux_repetition = np.sum(np.diff(ligne) == 0) / (len(ligne) - 1)
    return taux_repetition < 0.60  # seuil : 60 % → interpolé si ≥ 60 %


# ── 5. Calcul des pentes et orientations (algorithme Horn) ───────────────────

def calculer_pentes_et_orientation(mnt: np.ndarray, transform) -> tuple:
    """
    Calcule pente (%) et orientation cardinale via l'algorithme Horn 3×3.
    Transform doit être en Lambert 93 (mètres).
    Retourne (pentes float32, orientations array d'objets str).
    """
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

    pente = np.tan(np.arctan(np.sqrt(dzdx**2 + dzdy**2))) * 100
    pente[np.isnan(mnt)] = np.nan

    # Azimut géographique → 8 directions cardinales
    azimut_deg = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360
    _CARDS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]
    orientations = np.full(mnt.shape, "", dtype=object)
    valid = ~np.isnan(mnt)
    if valid.any():
        idx = np.round(azimut_deg[valid] / 45).astype(int) % 8
        orientations[valid] = np.array(_CARDS)[idx]

    return pente.astype(np.float32), orientations


# ── 5b. Classification et rendu RGBA ─────────────────────────────────────────

def classifier_pente_orientee(
    pentes: np.ndarray,
    orientations: np.ndarray,
    technologie: str,
    puissance: str,
) -> np.ndarray:
    """
    Classifie chaque pixel selon sa pente ET son orientation terrain.
    Retourne un array uint8 : 0=Favorable 1=Acceptable 2=Contraignant 3=Exclusion 255=NaN

    Orientation terrain (groupes N/S/EO depuis les 8 directions cardinales) :
      N  : N, NE, NO  (secteur nord)
      S  : S, SE, SO  (secteur sud)
      EO : E, O       (secteur est / ouest)
    """
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
        cls[valid & (pentes <  s1)]                        = 0
        cls[valid & (pentes >= s1) & (pentes <  s2)]       = 1
        cls[valid & (pentes >= s2) & (pentes <  s3)]       = 2
        cls[valid & (pentes >= s3)]                        = 3

    return cls


def _hex_to_rgb(couleur_hex: str) -> tuple:
    h = couleur_hex.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# Palette du mode topographie, convertie une fois au format {classe: (r, g, b)}
_PALETTE_TOPO_RGB = {i: _hex_to_rgb(c) for i, c in enumerate(_PALETTE_TOPO)}


def _to_rgba(classes: np.ndarray, opacite: float = 0.65, palette: dict = None) -> np.ndarray:
    if palette is None:
        palette = _PALETTE
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    alpha = int(opacite * 255)
    for val, (r, g, b) in palette.items():
        rgba[classes == val] = [r, g, b, alpha]
    return rgba


def _rgba_l93_to_wgs84(rgba: np.ndarray, transform_l93) -> tuple:
    """Reprojette RGBA de Lambert 93 vers WGS84 (rasterio.warp, pixel-parfait)."""
    h, w = rgba.shape[:2]
    src_crs = CRS.from_epsg(2154)
    dst_crs = CRS.from_epsg(4326)

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

    # warp_transform_bounds donne les bounds WGS84 exactes de la reprojection
    # (tient compte de la courbure L93→WGS84 — évite le décalage est/ouest)
    lon_min, lat_min, lon_max, lat_max = rasterio.warp.transform_bounds(
        src_crs, dst_crs, x_min, y_min, x_max, y_max, densify_pts=21
    )
    return dst, [[lat_min, lon_min], [lat_max, lon_max]]


def _img_b64(rgba: np.ndarray) -> str:
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── 6. Génération carte Folium ────────────────────────────────────────────────

def creer_carte(
    pentes: np.ndarray,
    orientations: np.ndarray,
    transform_l93,
    gdf_site: gpd.GeoDataFrame,
    mnt_brut: np.ndarray = None,
    technologie: str = "FIXE",
    puissance: str = "<5MWc",
    mode: str = MODE_CONSTRUCTIBILITE,
    zones: list = None,
    couches: dict = None,
) -> folium.Map:
    """
    Carte Folium avec :
    - Fond satellite ESRI (défaut)
    - Overlay couleur de pentes (zone + buffer)
    - Courbes de niveau calculées depuis mnt_brut (si fourni)
    - Contour jaune pointillé du site (toggleable)
    - Légende fixe des classes

    mode="constructibilite" : classification par (pente, orientation) selon
        les seuils projet — palette vert/jaune/orange/rouge.
    mode="topographie" : classification par pente seule, seuils universels
        <5 / 5-10 / 10-15 / >15 % — palette séquentielle ocre.

    Même sémantique que export_carte.generer_image_carte(). Les deux modes
    partagent MNT et transform : seule la classification et la palette changent.

    `zones` — liste issue de detecter_zones_planes() (géométries L93). Rendue
    en couche toggleable, mode topographie uniquement.

    `couches` — dict {nom: bool}. Une couche décochée n'est pas ajoutée à la
    carte : le même dict est passé à generer_image_carte(), si bien que
    l'export PNG/PDF montre exactement les couches visibles à l'écran.
    """
    couches = {**couches_par_defaut(), **(couches or {})}
    gdf_wgs84 = gdf_site.to_crs("EPSG:4326")
    union = gdf_wgs84.union_all()
    centre = [union.centroid.y, union.centroid.x]
    b = union.bounds
    site_bounds = [[b[1], b[0]], [b[3], b[2]]]

    carte = folium.Map(
        location=centre,
        zoom_start=15,
        tiles=None,
        control_scale=True,
        zoom_snap=0.25,
        zoom_delta=0.25,
    )

    # Fond OSM puis ESRI — le dernier ajouté est actif par défaut
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(carte)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, DigitalGlobe, GeoEye, i-cubed, USDA FSA, USGS, AEX",
        name="Satellite (ESRI)",
        overlay=False,
        control=True,
    ).add_to(carte)

    # Overlay pentes (zone + buffer) — la classification dépend du mode
    if mode == MODE_TOPOGRAPHIE:
        classes = classifier_pente_topo(pentes)
        rgba = _to_rgba(classes, palette=_PALETTE_TOPO_RGB)
    else:
        classes = classifier_pente_orientee(pentes, orientations, technologie, puissance)
        rgba = _to_rgba(classes)
    rgba_wgs84, bounds_pente = _rgba_l93_to_wgs84(rgba, transform_l93)

    if couches["pentes"]:
        fg = folium.FeatureGroup(name="Pentes", show=True)
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{_img_b64(rgba_wgs84)}",
            bounds=bounds_pente,
            opacity=1.0,
            interactive=False,
        ).add_to(fg)
        fg.add_to(carte)

    # Courbes de niveau (calculées depuis le MNT brut si fourni)
    if couches["courbes"] and mnt_brut is not None:
        courbes, labels_courbes = generer_courbes_niveau(mnt_brut, transform_l93)
        if courbes:
            fg_courbes = folium.FeatureGroup(name="Courbes de niveau", show=True)
            folium.GeoJson(
                {"type": "FeatureCollection", "features": courbes},
                style_function=lambda _: {
                    "color": "#000000",
                    "weight": 0.8,
                    "opacity": 0.5,
                },
            ).add_to(fg_courbes)
            for lbl in labels_courbes:
                folium.Marker(
                    location=[lbl["lat"], lbl["lon"]],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="'
                            f'font-size:9px;font-weight:bold;color:#111;'
                            f'text-shadow:1px 1px 0 white,-1px -1px 0 white,'
                            f'1px -1px 0 white,-1px 1px 0 white;'
                            f'white-space:nowrap;pointer-events:none;'
                            f'">{int(lbl["niveau"])}</div>'
                        ),
                        icon_size=(28, 12),
                        icon_anchor=(14, 6),
                    ),
                ).add_to(fg_courbes)
            fg_courbes.add_to(carte)

    # Contour du shapefile initial — jaune pointillé (toggleable via LayerControl)
    if couches["contour"]:
        fg_site = folium.FeatureGroup(name="Contour du site", control=True, show=True)
        folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            style_function=lambda _: {
                "fillColor": "none",
                "color": "#FFE600",
                "weight": 2,
                "dashArray": "6,4",
            },
        ).add_to(fg_site)
        fg_site.add_to(carte)

    # Zones planes exploitables — mode topographie uniquement.
    # ATTENTION : doit être ajoutée AVANT LayerControl, sinon la couche
    # n'apparaît pas dans le panneau de contrôle.
    if mode == MODE_TOPOGRAPHIE and zones and couches["zones"]:
        fg_zones = folium.FeatureGroup(name="Zones planes exploitables", show=True)
        gdf_zones_wgs84 = gpd.GeoDataFrame(
            geometry=[z["geom"] for z in zones], crs="EPSG:2154"
        ).to_crs("EPSG:4326")
        for geom_wgs84, z in zip(gdf_zones_wgs84.geometry, zones):
            surface_ha = z["surface"] / 10000
            libelle = (
                f"{z['surface']:,.0f} m² ({surface_ha:.2f} ha)".replace(",", " ")
            )
            # Popup au CLIC, pas tooltip au survol : le survol est déjà occupé
            # par la boîte pente/orientation (_ajouter_tooltip_hover), et deux
            # bulles suivant le curseur se superposaient.
            folium.GeoJson(
                geom_wgs84.__geo_interface__,
                style_function=lambda _: {
                    "fillColor": COULEUR_ZONE_FILL,
                    "color": COULEUR_ZONE_TRAIT,
                    "weight": 2,
                    "fillOpacity": 0.35,
                },
                popup=folium.Popup(f"Zone plane — {libelle}", max_width=250),
            ).add_to(fg_zones)

            # Étiquette permanente au centroïde : lisible sans interaction,
            # et statique donc jamais en conflit avec la boîte de survol.
            centre_z = geom_wgs84.representative_point()
            folium.Marker(
                location=[centre_z.y, centre_z.x],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:10px;font-weight:bold;color:#004a75;'
                        f'text-shadow:1px 1px 0 white,-1px -1px 0 white,'
                        f'1px -1px 0 white,-1px 1px 0 white;'
                        f'white-space:nowrap;pointer-events:none;text-align:center;'
                        f'">{libelle}</div>'
                    ),
                    icon_size=(120, 14),
                    icon_anchor=(60, 7),
                ),
            ).add_to(fg_zones)
        fg_zones.add_to(carte)

    folium.LayerControl(position="bottomright", collapsed=False).add_to(carte)
    carte.fit_bounds(site_bounds)
    _ajouter_legende(
        carte,
        mode=mode,
        avec_zones=bool(zones) and couches["zones"],
        couches=couches,
    )
    _ajouter_tooltip_hover(carte, pentes, orientations, transform_l93)

    # Logo UNITe — coin supérieur droit de la carte
    if _logo_b64:
        carte.get_root().html.add_child(folium.Element(
            f'<div style="position:absolute;top:10px;right:10px;z-index:1000;pointer-events:none;">'
            f'<img src="data:image/png;base64,{_logo_b64}" style="height:34px;opacity:0.88;"></div>'
        ))

    return carte


def _ajouter_legende(
    carte: folium.Map,
    mode: str = MODE_CONSTRUCTIBILITE,
    avec_zones: bool = False,
    couches: dict = None,
) -> None:
    couches = {**couches_par_defaut(), **(couches or {})}
    sw = "display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:5px"

    if mode == MODE_TOPOGRAPHIE:
        titre_leg = "Pente du terrain"
        entrees = list(zip(_PALETTE_TOPO, _LABELS_TOPO))
    else:
        titre_leg = "Pente — seuils projet"
        entrees = [
            ("#4CAF50", "Favorable"),
            ("#FFC107", "Acceptable"),
            ("#FF9800", "Contraignant"),
            ("#F44336", "Exclusion possible"),
        ]

    # La légende ne liste que les couches réellement présentes sur la carte
    if not couches["pentes"]:
        entrees = []
        titre_leg = "Légende"
    lignes = "".join(
        f'<span style="background:{couleur};{sw}"></span>{libelle}<br>'
        for couleur, libelle in entrees
    )
    if avec_zones:
        lignes += (
            f'<span style="background:{COULEUR_ZONE_FILL};opacity:.55;'
            f'border:1px solid {COULEUR_ZONE_TRAIT};{sw}"></span>'
            f'Zone plane exploitable<br>'
        )

    # Séparateur + contour : uniquement si la couche est affichée
    bloc_contour = (
        '<hr style="margin:7px 0;border:none;border-top:1px solid #e0e0e0">'
        '<div style="display:flex;align-items:center;gap:6px">'
        '<span style="display:inline-block;width:20px;height:0;'
        'border-top:2px dashed #FFE600;flex-shrink:0"></span>'
        '<span>Contour du site</span></div>'
        if couches["contour"] else ""
    )

    html = f"""
<style>
/* Empilement en bas à DROITE, de bas en haut :
     bandeau d'attribution — légende (la plus large) — panneau de couches.
   Le centre de la carte reste dégagé, et la barre d'échelle en bas à gauche
   n'est pas recouverte. Les décalages exacts sont posés par le script
   ci-dessous, qui mesure les hauteurs réelles (elles varient selon le mode). */
#leg {{
  position:fixed; right:10px; bottom:22px; z-index:1000;
  background:white; padding:9px 11px; border-radius:6px;
  font-size:11px; min-width:195px;
  box-shadow:0 1px 5px rgba(0,0,0,.3); color:#212121;
}}
.leaflet-control-attribution {{ font-size:8px !important; opacity:.7; }}
</style>
<div id="leg">
  <b>{titre_leg}</b><br><br>
  {lignes}
  {bloc_contour}
</div>
<script>
// Empile bandeau d'attribution / légende / panneau de couches sans
// chevauchement. Les hauteurs sont MESURÉES et non codées en dur : la légende
// varie selon le mode (4 ou 5 entrées) et la présence de la ligne
// « Zone plane exploitable », et le bandeau se replie sur deux lignes
// lorsque la carte est étroite.
// La marge est posée sur le seul panneau de couches, pas sur le conteneur
// bas-droite : sinon le bandeau d'attribution, qui y vit aussi, remonterait
// avec lui et viendrait s'intercaler entre les deux boîtes.
(function() {{
  function placer() {{
    var leg = document.getElementById('leg');
    var couches = document.querySelector('.leaflet-control-layers');
    if (!leg || !couches) {{ setTimeout(placer, 100); return; }}
    var attribution = document.querySelector('.leaflet-control-attribution');
    var hAttribution = attribution ? attribution.offsetHeight : 16;
    // Légende posée juste au-dessus du bandeau, laissé tout en bas
    leg.style.bottom = (hAttribution + 5) + 'px';
    // Panneau de couches remonté au-dessus de la légende
    couches.style.marginBottom = (leg.offsetHeight + 12) + 'px';
  }}
  placer();
  window.addEventListener('resize', placer);
}})();
</script>
"""
    carte.get_root().html.add_child(folium.Element(html))


def afficher_tableau_seuils(technologie: str, puissance: str) -> None:
    """Affiche un tableau HTML des seuils de pente par orientation avec pastilles colorées."""
    table = _SEUILS[(technologie, puissance)]

    _COULEURS = {
        "Favorable":    "#57bb57",
        "Acceptable":   "#f5e642",
        "Contraignant": "#f5a623",
        "Exclusion":    "#e8342a",
    }

    def _pastille(couleur: str) -> str:
        return (
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{couleur};'
            f'margin-right:5px;vertical-align:middle;"></span>'
        )

    headers = ["Orientation"] + [
        f'{_pastille(_COULEURS[col])}{col}'
        for col in ["Favorable", "Acceptable", "Contraignant", "Exclusion"]
    ]

    rows_html = ""
    for orient_label, orient_key in [("Nord", "N"), ("Sud", "S"), ("Est / Ouest", "EO")]:
        s = table[orient_key]["seuils"]
        note = table[orient_key]["note_contraignant"]
        cont_str = f"{s[1]}–{s[2]} %"
        if note:
            cont_str += f'<br><span style="font-size:11px;color:#888">({note})</span>'
        cells = [
            f"<b>{orient_label}</b>",
            f"&lt; {s[0]} %",
            f"{s[0]}–{s[1]} %",
            cont_str,
            f"&gt; {s[2]} %",
        ]
        rows_html += "<tr>" + "".join(f'<td style="padding:5px 8px;">{c}</td>' for c in cells) + "</tr>"

    html = (
        '<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        "<thead>"
        '<tr style="border-bottom:2px solid #ddd;">'
        + "".join(
            f'<th style="text-align:left;padding:6px 8px;">{h}</th>'
            for h in headers
        )
        + "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
    )

    st.caption(f"Seuils de pente — **{technologie}** · **{puissance}**")
    st.markdown(html, unsafe_allow_html=True)
    note_globale = table.get("note_globale")
    if note_globale:
        st.caption(f"ℹ️ {note_globale}")


# ── 6b. Sections de résultats : carte + exports PNG/PDF à la demande ─────────

def _slug(texte: str) -> str:
    """
    Nom de fichier sûr : accents et caractères interdits sous Windows remplacés.
    Ex. '≥5MWc' → 'sup5MWc', 'OMBRIÈRES' → 'OMBRIERES'.
    """
    remplacements = {
        "≥": "sup", "≤": "inf", "<": "inf", ">": "sup",
        "È": "E", "É": "E", "Ê": "E", "À": "A", "Ç": "C",
        "è": "e", "é": "e", "ê": "e", "à": "a", "ç": "c",
        "·": "-", "/": "-", "\\": "-", ":": "-",
        " ": "_", "*": "", "?": "", '"': "", "|": "",
    }
    for avant, apres in remplacements.items():
        texte = texte.replace(avant, apres)
    return texte


# ── Zones planes : détection et carte topo mises en cache ────────────────────
#
# Coûts mesurés sur grille 2000×2000 à 1 m (pire cas réaliste) :
#   détection zones ....... 4,2 à 8,6 s selon les paramètres
#   courbes de niveau ..... 0,68 s
#   sérialisation tooltip . 0,42 s (2 Mo de JSON)
# → ~10 s par recalcul : trop lent pour des sliders en temps réel, d'où le
#   st.form (bouton « Actualiser les zones ») côté interface.
#
# Le hash porte sur le CONTENU des pentes + les deux paramètres : revenir à un
# couple de valeurs déjà testé est instantané (pas de recalcul, pas de rerender).

@st.cache_data(show_spinner=False, max_entries=8)
def _zones_planes_cached(
    pentes_bytes: bytes,
    shape: tuple,
    dtype_str: str,
    transform_tuple: tuple,
    seuil: float,
    largeur: float,
) -> list:
    """Enveloppe cacheable de detecter_zones_planes (arrays → bytes hashables)."""
    pentes = np.frombuffer(pentes_bytes, dtype=dtype_str).reshape(shape)
    return detecter_zones_planes(
        pentes,
        Affine(*transform_tuple),
        seuil_pente=seuil,
        largeur_min=largeur,
    )


def _cle_cache(export: dict) -> tuple:
    """Fragment de clé de cache commun : contenu des pentes + géoréférencement."""
    pentes = np.ascontiguousarray(export["pentes"])
    return (
        pentes.tobytes(),
        pentes.shape,
        pentes.dtype.str,
        tuple(export["transform"])[:6],
    )


def calculer_zones_planes(export: dict, seuil: float, largeur: float) -> list:
    """Détection des zones planes pour un jeu de paramètres, avec cache."""
    pentes_bytes, shape, dtype_str, transform_tuple = _cle_cache(export)
    return _zones_planes_cached(
        pentes_bytes, shape, dtype_str, transform_tuple, seuil, largeur
    )


def _version_rendu_carte() -> str:
    """
    Empreinte du code qui produit le HTML de la carte.

    Sans elle, @st.cache_data resservirait le HTML rendu par une version
    antérieure des fonctions de rendu après modification du code : les
    changements de mise en page (position de la légende, couches…) resteraient
    invisibles tant que les paramètres de zones ne changent pas.
    """
    source = "".join(
        inspect.getsource(f) for f in (creer_carte, _ajouter_legende)
    )
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:12]


@st.cache_data(show_spinner=False, max_entries=8)
def _carte_html_cached(
    _export: dict,
    cle_contenu: tuple,
    mode: str,
    seuil: float,
    largeur: float,
    version_rendu: str,
) -> str:
    """
    HTML d'une carte Folium pour un mode et un jeu de paramètres donnés.

    La carte affichée porte TOUJOURS toutes les couches : leur affichage se
    règle dans le panneau Leaflet, côté navigateur. Le tri des couches pour
    l'export est une décision distincte, prise au moment du téléchargement.

    `_export` est préfixé d'un underscore : Streamlit ne le hache pas (il
    contient des arrays non hashables). La clé repose sur `cle_contenu`
    (contenu des pentes + géoréférencement), le mode et les paramètres de
    zones — revenir à une combinaison déjà vue est instantané.
    """
    zones = (
        calculer_zones_planes(_export, seuil, largeur)
        if mode == MODE_TOPOGRAPHIE else None
    )
    carte = creer_carte(
        _export["pentes"], _export["orientations"],
        _export["transform"], _export["gdf"],
        mnt_brut=_export["mnt"],
        technologie=_export["technologie"],
        puissance=_export["puissance"],
        mode=mode,
        zones=zones,
    )
    return carte.get_root().render()


# Valeurs par défaut des sliders — doivent correspondre à celles des widgets
SEUIL_ZP_DEFAUT = 3.0
LARGEUR_ZP_DEFAUT = 20


def reinitialiser_exports(volet: str) -> None:
    """Purge l'état des blocs d'export d'un volet (nouvelle extraction)."""
    for mode in (MODE_CONSTRUCTIBILITE, MODE_TOPOGRAPHIE):
        for prefixe in ("png", "panneau", "resume", "sig"):
            st.session_state[f"{volet}_{prefixe}_{mode}"] = None


def preparer_carte(volet: str, mode: str, export: dict) -> tuple:
    """
    Assemble une carte pour l'affichage : lit les paramètres de zones depuis
    session_state, calcule les zones, récupère le HTML (avec cache).

    Retourne (html, zones, (seuil, largeur)).
    """
    if mode == MODE_TOPOGRAPHIE:
        seuil = float(st.session_state.get(f"{volet}_zp_seuil", SEUIL_ZP_DEFAUT))
        largeur = float(st.session_state.get(f"{volet}_zp_largeur", LARGEUR_ZP_DEFAUT))
        zones = calculer_zones_planes(export, seuil, largeur)
    else:
        seuil, largeur = SEUIL_ZP_DEFAUT, float(LARGEUR_ZP_DEFAUT)
        zones = None

    html = _carte_html_cached(
        export, _cle_cache(export), mode, seuil, largeur, _version_rendu_carte()
    )
    return html, zones, (seuil, largeur)


def afficher_section_carte(
    volet: str,
    mode: str,
    carte_html: str,
    export: dict,
    zones: list = None,
    zones_params: tuple = None,
) -> None:
    """
    Affiche une section de résultat : titre, bloc d'export, carte Folium,
    et (mode constructibilité seulement) le tableau des seuils.

    `volet` — "a" ou "b", préfixe des clés session_state et des clés widget.
    `export` — dict portant tout ce qu'il faut pour régénérer l'image :
        pentes, orientations, transform, mnt, gdf, technologie, puissance,
        nom_fichier, nom_site.

    L'image PNG/PDF n'est PAS générée à l'extraction : le téléchargement des
    tuiles contextily prend plusieurs secondes. Elle l'est au clic, puis
    mémorisée en session_state pour que le second clic soit instantané.

    `zones` / `zones_params` — couche zones planes, mode topographie seulement.
    """
    est_constructibilite = mode == MODE_CONSTRUCTIBILITE

    if est_constructibilite:
        st.markdown("##### 2 · 🏗️ Carte constructibilité")
        st.caption(
            "Classification croisée pente × orientation selon les seuils projet."
        )
        nom_base = (
            f"{export['nom_site']}_constructibilite_"
            f"{_slug(export['technologie'])}_{_slug(export['puissance'])}"
        )
    else:
        st.markdown("##### 1 · ⛰️ Carte topographique")
        st.caption(
            "Pente brute du terrain, toutes orientations confondues — "
            "seuils universels < 5 / 5–10 / 10–15 / > 15 %."
        )
        nom_base = f"{export['nom_site']}_topographie"

    _afficher_bloc_export(volet, mode, export, nom_base, zones, zones_params)

    if carte_html:
        st.components.v1.html(carte_html, height=580, scrolling=False)

    if est_constructibilite:
        afficher_tableau_seuils(export["technologie"], export["puissance"])
    else:
        _afficher_sliders_zones(volet, zones)


def _afficher_bloc_export(
    volet: str,
    mode: str,
    export: dict,
    nom_base: str,
    zones: list,
    zones_params: tuple,
) -> None:
    """
    Bloc d'export en trois états successifs :

      1. bouton « Générer l'export PNG / PDF »
      2. choix des couches à conserver, puis confirmation
      3. boutons de téléchargement + retour possible à l'étape 2

    Le choix des couches n'apparaît qu'à l'étape 2 : il ne concerne que le
    fichier produit. L'affichage à l'écran, lui, se règle dans le panneau de
    couches de la carte — deux jeux de cases visibles en permanence prêtaient
    à confusion.

    Ces cases sont indispensables côté Streamlit : le panneau Leaflet vit dans
    l'iframe et son état n'est pas lisible par le serveur, qui ne saurait donc
    pas quoi dessiner dans le PNG/PDF.
    """
    cle_png = f"{volet}_png_{mode}"
    cle_panneau = f"{volet}_panneau_{mode}"
    cle_resume = f"{volet}_resume_{mode}"
    for cle in (cle_png, cle_panneau, cle_resume):
        if cle not in st.session_state:
            st.session_state[cle] = None

    # Un changement des paramètres de zones périme le fichier déjà produit :
    # sans ça, le téléchargement livrerait les zones de la version précédente.
    cle_sig = f"{volet}_sig_{mode}"
    if st.session_state.get(cle_sig) != zones_params:
        st.session_state[cle_png] = None
        st.session_state[cle_resume] = None
        st.session_state[cle_sig] = zones_params

    noms = [n for n in COUCHES if n != "zones" or mode == MODE_TOPOGRAPHIE]

    # ── État 2 : sélection des couches ───────────────────────────────────────
    if st.session_state[cle_panneau]:
        with st.container(border=True):
            st.markdown("**Couches à conserver pour le téléchargement**")
            # La valeur retournée par le widget est lue directement : elle est
            # à jour dès le run déclenché par le clic sur « Générer ».
            choix = {}
            colonnes = st.columns(len(noms))
            for colonne, nom in zip(colonnes, noms):
                with colonne:
                    choix[nom] = st.checkbox(
                        LIBELLES_COUCHES[nom],
                        value=True,
                        key=f"{volet}_exp_{nom}_{mode}",
                    )
            col_ok, col_annul = st.columns([2, 1])
            with col_ok:
                valider = st.button(
                    "✅ Générer le fichier",
                    key=f"{volet}_ok_{mode}",
                    use_container_width=True,
                    type="primary",
                )
            with col_annul:
                annuler = st.button(
                    "Annuler",
                    key=f"{volet}_annul_{mode}",
                    use_container_width=True,
                )

        if annuler:
            st.session_state[cle_panneau] = False
            st.rerun()

        if valider:
            # Les couches absentes du mode (zones hors topographie) restent à
            # True : elles ne sont de toute façon pas dessinées.
            couches = {nom: bool(choix.get(nom, True)) for nom in COUCHES}
            try:
                with st.spinner(
                    "Génération de l'image — téléchargement des tuiles satellite…"
                ):
                    st.session_state[cle_png] = generer_image_carte(
                        pentes=export["pentes"],
                        orientations=export["orientations"],
                        transform_l93=export["transform"],
                        gdf_site=export["gdf"],
                        mnt_brut=export["mnt"],
                        technologie=export["technologie"],
                        puissance=export["puissance"],
                        dpi=200,
                        nom_fichier=export["nom_fichier"],
                        mode=mode,
                        zones_planes=zones,
                        zones_params=zones_params,
                        couches=couches,
                    )
                retenues = [LIBELLES_COUCHES[n] for n in noms if couches[n]]
                st.session_state[cle_resume] = (
                    ", ".join(retenues) if retenues else "fond satellite seul"
                )
                st.session_state[cle_panneau] = False
                st.rerun()
            except Exception as e:
                st.warning(f"Export image indisponible : {e}")
        return

    # ── État 3 : fichier prêt ────────────────────────────────────────────────
    if st.session_state[cle_png]:
        col_png, col_pdf = st.columns(2)
        with col_png:
            st.download_button(
                label="🖼️ Télécharger PNG",
                data=st.session_state[cle_png],
                file_name=f"{nom_base}.png",
                mime="image/png",
                use_container_width=True,
                key=f"{volet}_dl_png_{mode}",
            )
        with col_pdf:
            st.download_button(
                label="📄 Télécharger PDF",
                data=png_vers_pdf(st.session_state[cle_png]),
                file_name=f"{nom_base}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"{volet}_dl_pdf_{mode}",
            )
        st.caption(f"Couches incluses : {st.session_state[cle_resume]}.")
        if st.button(
            "↻ Régénérer avec d'autres couches",
            key=f"{volet}_regen_{mode}",
            use_container_width=True,
        ):
            st.session_state[cle_panneau] = True
            st.rerun()
        return

    # ── État 1 : point de départ ─────────────────────────────────────────────
    if st.button(
        "🖼️ Générer l'export PNG / PDF",
        key=f"{volet}_gen_{mode}",
        use_container_width=True,
    ):
        st.session_state[cle_panneau] = True
        st.rerun()


def _afficher_sliders_zones(volet: str, zones: list) -> None:
    """
    Sliders de paramétrage des zones planes, sous la carte topographique.

    Dans un st.form : le recalcul complet (détection + reconstruction de la
    carte Folium) atteint ~10 s sur un site 1 m de grande emprise, ce qui rend
    des sliders en temps réel inutilisables. La validation explicite déclenche
    un seul rerun au lieu d'un par cran de slider.
    """
    st.markdown("**Zones planes exploitables** — base vie, stockage")

    with st.form(key=f"{volet}_zp_form"):
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.slider(
                "Pente maximale", 1.0, 8.0, 3.0, 0.5,
                format="%.1f %%",
                key=f"{volet}_zp_seuil",
                help=(
                    "3 % : base vie, modulaires, parking. "
                    "5 % : stockage, circulation PL. "
                    "Au-delà, terrassement nécessaire."
                ),
            )
        with col_s2:
            st.slider(
                "Largeur minimale", 10, 40, 20, 5,
                format="%d m",
                key=f"{volet}_zp_largeur",
                help=(
                    "Largeur utile en tout point de la zone. Élimine les bandes "
                    "étroites qui suivent les courbes de niveau."
                ),
            )
        st.form_submit_button(
            "🔄 Actualiser les zones", use_container_width=True
        )

    if zones:
        st.success(resume_zones_planes(zones))
    else:
        st.info(resume_zones_planes(zones or []))
    st.caption(
        f"Surface minimale retenue : {SURFACE_MIN_ZONE:,.0f} m².".replace(",", " ")
        + " Détection sur le MNT lissé — les zones sont indicatives, à confirmer "
        "par un relevé terrain."
    )


# ── 7. Tooltip hover pente / orientation ─────────────────────────────────────

def _pentes_to_js(arr: np.ndarray) -> str:
    """Sérialise un array 2D float en JSON compact (NaN → null, 1 décimale)."""
    rows = []
    for row in arr:
        parts = ["null" if np.isnan(v) else str(round(float(v), 1)) for v in row]
        rows.append("[" + ",".join(parts) + "]")
    return "[" + ",".join(rows) + "]"


def _orientations_to_js(arr: np.ndarray) -> str:
    """Sérialise un array 2D de strings d'orientation en JSON compact."""
    rows = []
    for row in arr:
        parts = [f'"{v}"' if v else "null" for v in row]
        rows.append("[" + ",".join(parts) + "]")
    return "[" + ",".join(rows) + "]"


def _ajouter_tooltip_hover(
    carte: folium.Map,
    pentes: np.ndarray,
    orientations: np.ndarray,
    transform_l93,
) -> None:
    """
    Injecte un handler mousemove Leaflet affichant pente et orientation au survol.
    Downsample automatique si rows×cols > 250 000 pour limiter le JSON à ~2 Mo.
    Les arrays originaux (ImageOverlay, export TXT) restent inchangés.
    """
    h, w = pentes.shape

    # Downsample uniquement pour le JSON du tooltip
    factor = max(1, math.ceil(math.sqrt(h * w / 250_000)))
    pentes_js = pentes[::factor, ::factor]
    orientations_js = orientations[::factor, ::factor]
    rows_js, cols_js = pentes_js.shape

    # Bounds WGS84 exactes via rasterio.warp.transform_bounds
    # (densify_pts=21 tient compte de la courbure L93→WGS84 sur les bords
    #  → corrige le décalage est/ouest du tooltip vs l'ImageOverlay)
    x_min = transform_l93.c
    y_max_l93 = transform_l93.f
    x_max = x_min + w * transform_l93.a
    y_min = y_max_l93 + h * transform_l93.e  # transform.e < 0
    lon_w, lat_s, lon_e, lat_n = rasterio.warp.transform_bounds(
        "EPSG:2154", "EPSG:4326",
        x_min, y_min, x_max, y_max_l93,
        densify_pts=21,
    )

    map_var = carte.get_name()

    html = f"""<div id="hover-info" style="
  position:fixed;top:0;left:0;z-index:1000;
  background:rgba(0,0,0,0.65);color:white;
  padding:6px 12px;border-radius:4px;
  font-size:13px;font-family:monospace;
  pointer-events:none;display:none;
  transform:translate(14px,-50%);
  white-space:nowrap;"></div>
<script>
(function() {{
  var PENTES = {_pentes_to_js(pentes_js)};
  var ORIENTATIONS = {_orientations_to_js(orientations_js)};
  var BOUNDS = {{north:{lat_n:.8f},south:{lat_s:.8f},east:{lon_e:.8f},west:{lon_w:.8f}}};
  var ROWS = {rows_js};
  var COLS = {cols_js};

  // {map_var} est cree APRES ce script — on attend qu'il soit disponible.
  function init() {{
    if (typeof {map_var} === 'undefined') {{ setTimeout(init, 100); return; }}
    var mapObj = {map_var};

    mapObj.on('mousemove', function(e) {{
      var pt = e.containerPoint;
      var row = Math.floor((BOUNDS.north - e.latlng.lat) / (BOUNDS.north - BOUNDS.south) * ROWS);
      var col = Math.floor((e.latlng.lng - BOUNDS.west) / (BOUNDS.east - BOUNDS.west) * COLS);
      var el = document.getElementById('hover-info');
      if (!el) return;
      if (row < 0 || row >= ROWS || col < 0 || col >= COLS) {{
        el.style.display = 'none'; return;
      }}
      var pente = PENTES[row][col];
      var orient = ORIENTATIONS[row][col];
      if (pente !== null) {{
        el.style.top  = pt.y + 'px';
        el.style.left = pt.x + 'px';
        el.innerHTML = 'Pente : <b>' + pente + ' %</b> &mdash; Orientation : <b>' + orient + '</b>';
        el.style.display = 'block';
      }} else {{
        el.style.display = 'none';
      }}
    }});

    mapObj.on('mouseout', function() {{
      var el = document.getElementById('hover-info');
      if (el) el.style.display = 'none';
    }});
  }}

  init();
}})();
</script>"""
    carte.get_root().html.add_child(folium.Element(html))


# ── 8. Export TXT PVCase ──────────────────────────────────────────────────────

def exporter_txt(mnt: np.ndarray, transform_l93, pas: int = 1) -> bytes:
    """
    Format PVCase Ground Mount : _MULTIPLE _POINT + X,Y,Z (L93, virgule, 2 déc.)

    pas=1 : tous les pixels — résolution native
    pas>1 : un pixel sur pas×pas — sous-échantillonnage
            (ex. pas=5 pour obtenir un export 5 m depuis un raster 1 m)
    Les coordonnées X,Y sont calculées dans le repère du raster original.
    """
    mnt_exp = mnt[::pas, ::pas]
    rows, cols = mnt_exp.shape

    col_idx = np.tile(np.arange(cols), rows)
    row_idx = np.repeat(np.arange(rows), cols)
    z_flat = mnt_exp.ravel()
    valid = ~np.isnan(z_flat)

    # Indices originaux (avant sous-échantillonnage) pour coordonnées exactes
    x = transform_l93.c + (col_idx[valid] * pas + 0.5) * transform_l93.a
    y = transform_l93.f + (row_idx[valid] * pas + 0.5) * transform_l93.e
    z = z_flat[valid]

    if len(z) == 0:
        raise ValueError("Aucun point valide à exporter.")

    buf = io.BytesIO()
    buf.write(b"_MULTIPLE _POINT\n")
    for xi, yi, zi in zip(x, y, z):
        buf.write(f"{xi:.2f},{yi:.2f},{zi:.2f}\n".encode())
    return buf.getvalue()


# ── 9. Courbes de niveau (skimage) ───────────────────────────────────────────

def generer_courbes_niveau(mnt: np.ndarray, transform_l93) -> list:
    """
    Retourne une liste de features GeoJSON (LineString WGS84) représentant
    les courbes de niveau du MNT.

    Intervalle automatique : ~20 courbes sur la plage d'altitude.
    Si > 5 000 contours bruts, sous-échantillonne les coordonnées (1 sur 3)
    avant la conversion WGS84 pour limiter la taille du GeoJSON.
    """
    z_min = np.nanmin(mnt)
    z_max = np.nanmax(mnt)
    if np.isnan(z_min) or z_max - z_min < 1:
        return [], []

    # Intervalle arrondi au mètre le plus proche, ~20 courbes sur la plage
    intervalle = max(1, round((z_max - z_min) / 20))
    niveaux = np.arange(
        np.ceil(z_min / intervalle) * intervalle,
        z_max,
        intervalle,
    )

    # Niveaux à labelliser : 1 sur 5 si intervalle fin (≤ 2 m), tous sinon
    pas_label = 5 if intervalle <= 2 else 1
    niveaux_labellises = {float(n) for i, n in enumerate(niveaux) if i % pas_label == 0}

    # Première passe : collecte des contours en coordonnées pixel (row, col)
    raw_contours = []
    for niveau in niveaux:
        for contour in skimage_measure.find_contours(mnt, level=float(niveau)):
            if len(contour) >= 2:
                raw_contours.append((float(niveau), contour))

    if not raw_contours:
        return []

    # Sous-échantillonnage si trop de contours (avant conversion WGS84)
    step = 3 if len(raw_contours) > 5000 else 1

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    features = []
    labels = []  # {"lat": ..., "lon": ..., "niveau": ...} pour les DivIcon
    for niveau, contour in raw_contours:
        coords_c = contour[::step]
        if len(coords_c) < 2:
            continue
        # Lissage gaussien sur les coordonnées pixel avant conversion L93
        # sigma=2 : bon compromis angularité / fidélité topographique
        rows_c = gaussian_filter1d(coords_c[:, 0], sigma=2)
        cols_c = gaussian_filter1d(coords_c[:, 1], sigma=2)
        x_l93 = transform_l93.c + cols_c * transform_l93.a
        y_l93 = transform_l93.f + rows_c * transform_l93.e
        lons, lats = transformer.transform(x_l93, y_l93)
        coords = list(zip(lons.tolist(), lats.tolist()))
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"niveau": niveau},
        })
        # Label au point médian — seulement courbes assez longues + niveaux principaux
        if len(coords) >= 20 and niveau in niveaux_labellises:
            lon_mid, lat_mid = coords[len(coords) // 2]
            labels.append({"lat": lat_mid, "lon": lon_mid, "niveau": niveau})

    return features, labels


# ── B1. Chargement courbes de niveau drone ────────────────────────────────────

def charger_courbes(raw_bytes: bytes, ext: str) -> gpd.GeoDataFrame:
    """
    Charge un fichier de courbes de niveau (LineString Z) depuis des bytes bruts.

    Formats supportés :
      - zip    : doit contenir un GeoJSON (.geojson / .json) ou un Shapefile (.shp)
      - geojson / json : GeoJSON direct

    Reprojette en Lambert 93 (EPSG:2154).
    Vérifie la présence de coordonnées Z sur la première géométrie non nulle.
    """
    ext = ext.lower().lstrip(".")

    if ext == "zip":
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            geojson_files = [
                n for n in zf.namelist()
                if n.lower().endswith((".geojson", ".json"))
            ]
            shp_files = [n for n in zf.namelist() if n.lower().endswith(".shp")]
            if not geojson_files and not shp_files:
                raise ValueError("Le ZIP ne contient ni .shp ni .geojson.")
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                if shp_files:
                    gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                else:
                    gdf = gpd.read_file(os.path.join(tmpdir, geojson_files[0]))
    elif ext in ("geojson", "json"):
        gdf = gpd.read_file(io.BytesIO(raw_bytes))
    else:
        raise ValueError(f"Format non supporté : .{ext}")

    # Détection CRS robuste :
    # certains GeoJSON portent le CRS en entête mais geopandas/fiona l'ignore
    # dans les versions anciennes. Si les coordonnées ressemblent à de l'UTM
    # (X > 1000, Y > 1000), on force EPSG:32631 (UTM 31N — CRS drone UNITe).
    if gdf.crs is None:
        sample_geom = next((g for g in gdf.geometry if g is not None), None)
        if sample_geom is not None:
            coords = list(sample_geom.coords)
            if coords:
                x0, y0 = coords[0][0], coords[0][1]
                if abs(x0) > 1000 or abs(y0) > 1000:
                    # Coordonnées métriques détectées → UTM 31N
                    st.warning(
                        "CRS absent du fichier — EPSG:32631 (WGS84 / UTM zone 31N) "
                        "supposé d'après la plage des coordonnées."
                    )
                    gdf = gdf.set_crs("EPSG:32631")
                else:
                    gdf = gdf.set_crs("EPSG:4326")
            else:
                gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.set_crs("EPSG:4326")

    if gdf.empty:
        raise ValueError("Le fichier de courbes est vide.")

    # Vérification Z sur la première géométrie non nulle
    sample = next((g for g in gdf.geometry if g is not None), None)
    if sample is None:
        raise ValueError("Toutes les géométries sont nulles.")
    if not sample.has_z:
        raise ValueError(
            "Les géométries ne contiennent pas de coordonnée Z. "
            "Le fichier doit provenir d'un relevé drone avec altitudes "
            "(courbes de niveau 3D)."
        )

    # Reprojection en Lambert 93
    gdf = gdf.to_crs("EPSG:2154")
    return gdf


# ── B2. Extraction des sommets XYZ ───────────────────────────────────────────

def extraire_points_xyz(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Extrait tous les sommets (X, Y, Z) des géométries LineString Z.
    Retourne un tableau NumPy (N, 3) en coordonnées L93.
    Les types Point, Polygon, etc. sont ignorés.
    """
    points = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "LineString":
            for coord in geom.coords:
                if len(coord) >= 3:
                    points.append(coord[:3])
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                for coord in line.coords:
                    if len(coord) >= 3:
                        points.append(coord[:3])

    if not points:
        raise ValueError(
            "Aucun sommet Z extrait. Vérifiez que le fichier contient "
            "des LineString ou MultiLineString avec coordonnées Z."
        )

    return np.array(points, dtype=np.float64)


# ── B3. Interpolation MNT (scipy griddata linéaire) ──────────────────────────

def interpoler_mnt(points_xyz: np.ndarray, resolution: float) -> tuple:
    """
    Interpole un MNT régulier (grille L93) depuis un nuage de points XYZ.

    Utilise scipy.interpolate.griddata avec méthode 'linear'.
    Les points hors de l'enveloppe convexe des données sources resteront NaN.
    Retourne (mnt float32, transform_l93 Affine).
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]

    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())

    cols = max(2, int((xmax - xmin) / resolution))
    rows = max(2, int((ymax - ymin) / resolution))

    # Sécurité : limiter à 2000×2000
    max_dim = 2000
    if cols > max_dim or rows > max_dim:
        facteur = max(cols, rows) / max_dim
        resolution = resolution * facteur
        cols = max(2, int((xmax - xmin) / resolution))
        rows = max(2, int((ymax - ymin) / resolution))
        st.warning(
            f"Zone trop grande — résolution ajustée à {resolution:.2f} m "
            f"({rows}×{cols} = {rows * cols:,} points)."
        )

    transform = rt_from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    # Grille de destination (centres de pixels, L93)
    col_idx = np.tile(np.arange(cols), rows)
    row_idx = np.repeat(np.arange(rows), cols)
    x_grid = transform.c + (col_idx + 0.5) * transform.a
    y_grid = transform.f + (row_idx + 0.5) * transform.e  # transform.e < 0

    # Interpolation linéaire (Delaunay + interpolation barycentrique)
    z_interp = scipy_griddata(
        np.column_stack([x, y]),
        z,
        np.column_stack([x_grid, y_grid]),
        method="linear",
    )

    mnt = z_interp.reshape(rows, cols).astype(np.float32)
    return mnt, transform


# ── Interface Streamlit ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Données Topo ESQ/APS",
    page_icon="🗺️",
    layout="wide",
)

# ── Logo UNITe — chargé une fois, utilisé dans le header ET les cartes Folium ─
_logo_b64 = None
try:
    _logo_path = os.path.join(os.path.dirname(__file__), "logo_unite.png")
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
except FileNotFoundError:
    pass

# Header : titre à gauche, logo à droite dans la même grille de contenu
_col_titre, _col_logo = st.columns([8, 1], vertical_alignment="center")
with _col_titre:
    st.title("🗺️ Données Topo ESQ/APS")
    st.caption(
        "MNT RGE ALTI IGN (phase ESQ) · "
        "Conversion données drone UNITe (phase APS) · "
        "Export TXT PVCase Ground Mount"
    )
with _col_logo:
    if _logo_b64:
        st.markdown(
            f'<div style="text-align:right;">'
            f'<img src="data:image/png;base64,{_logo_b64}" '
            f'style="height:85px;max-width:100%;"></div>',
            unsafe_allow_html=True,
        )
st.divider()

# ── Session state — volet A ───────────────────────────────────────────────────
for _k in (
    "a_txt_bytes", "a_nom_fichier", "a_intervalle_courbes",
    "a_txt_bytes_interp", "a_nom_fichier_interp",
    "a_resolution_effective", "a_warning_resolution",
    # Données de réexport ; les HTML de cartes viennent de _carte_html_cached
    "a_export",
    f"a_png_{MODE_CONSTRUCTIBILITE}", f"a_png_{MODE_TOPOGRAPHIE}",
):
    if _k not in st.session_state:
        st.session_state[_k] = None

if "a_extracting" not in st.session_state:
    st.session_state.a_extracting = False

for _k in ("a__uploaded_bytes", "a__uploaded_name", "a__resolution", "a__buffer_m"):
    if _k not in st.session_state:
        st.session_state[_k] = None

if "a__technologie" not in st.session_state:
    st.session_state.a__technologie = None
if "a__puissance" not in st.session_state:
    st.session_state.a__puissance = None

# ── Session state — volet B ───────────────────────────────────────────────────
for _k in (
    "b_txt_bytes", "b_nom_fichier", "b_n_points", "b_shape",
    "b_intervalle_courbes", "b_export",
    f"b_png_{MODE_CONSTRUCTIBILITE}", f"b_png_{MODE_TOPOGRAPHIE}",
):
    if _k not in st.session_state:
        st.session_state[_k] = None

if "b_extracting" not in st.session_state:
    st.session_state.b_extracting = False

for _k in (
    "b__uploaded_bytes", "b__uploaded_name", "b__resolution_drone",
    "b__zone_bytes", "b__zone_name", "b__buffer_m",
):
    if _k not in st.session_state:
        st.session_state[_k] = None

if "b__mode" not in st.session_state:
    st.session_state.b__mode = "Convertir les données"
if "b__technologie" not in st.session_state:
    st.session_state.b__technologie = None
if "b__puissance" not in st.session_state:
    st.session_state.b__puissance = None

# ── Onglets ───────────────────────────────────────────────────────────────────
onglet_a, onglet_b = st.tabs([
    "A · Phase ESQ — Extraction RGE ALTI IGN",
    "B · Phase APS — Conversion données drone UNITe",
])


# ══════════════════════════════════════════════════════════════════════════════
#  VOLET A — Extraction RGE ALTI IGN
# ══════════════════════════════════════════════════════════════════════════════

with onglet_a:

    st.caption(
        "🗺️ Ce volet extrait les données altimétriques IGN (MNT RGE ALTI) "
        "sur la zone d'implantation et les exporte au format PVCase Ground Mount "
        "(_MULTIPLE _POINT, X,Y,Z en L93)."
    )


    # Raccourci : True pendant tout le calcul → désactive les widgets
    a_locked = st.session_state.a_extracting

    col_a_params, col_a_result = st.columns([1, 2], gap="large")

    # ── Colonne gauche : saisie des paramètres ────────────────────────────────
    with col_a_params:

        # ── 1. Shapefile ──────────────────────────────────────────────────────
        st.subheader("1 · Shapefile")
        a_uploaded = st.file_uploader(
            "Déposer le fichier .zip contenant le shapefile de la zone d'implantation",
            type=["zip"],
            disabled=a_locked,
            key="a_uploader",
            help="Le zip doit contenir les fichiers .shp, .dbf et .shx de la zone de projet.",
        )

        # ── 2. Résolution ─────────────────────────────────────────────────────
        st.subheader("2 · Résolution")
        a_resolution = st.radio(
            "Résolution du MNT",
            options=[5, 1],
            format_func=lambda x: (
                "5 m — rapide"
                if x == 5 else
                "1 m — précis (~30–60 s pour 10 ha)"
            ),
            disabled=a_locked,
            key="a_resolution_radio",
            help="La résolution 5 m est suffisante pour le pré-design. Passer à 1 m pour une analyse fine en phase APS/APD.",
        )

        # ── 3. Buffer ─────────────────────────────────────────────────────────
        st.subheader("3 · Buffer")
        a_buffer_m = st.select_slider(
            "Emprise à récupérer autour de la zone d'implantation",
            options=list(range(0, 31, 5)),
            value=10,
            format_func=lambda x: f"{x} m",
            disabled=a_locked,
            key="a_buffer_slider",
            help="Zone étendue autour du shapefile couverte par le MNT et visible sur la carte.",
        )

        # ── 4. Caractéristiques projet ────────────────────────────────────────
        st.subheader("4 · Caractéristiques projet")
        technologie_a = st.radio(
	    "Type de structure",
            options=["FIXE", "TRACKERS", "OMBRIÈRES"],
            horizontal=True,
            index=None,
            disabled=a_locked,
            key="a_technologie_radio",
        )
        puissance_a = st.radio(
            "Puissance",
            options=["<5MWc", "≥5MWc"],
            horizontal=True,
            index=None,
            disabled=a_locked,
            key="a_puissance_radio",
        )

        st.divider()

        # ── Validation ────────────────────────────────────────────────────────
        a_manquants = []
        if a_uploaded is None:
            a_manquants.append("shapefile ZIP")
        if technologie_a is None:
            a_manquants.append("type de structure")
        if puissance_a is None:
            a_manquants.append("puissance du projet")

        a_pret = len(a_manquants) == 0
        if not a_pret and not a_locked:
            st.info("En attente : {}".format(", ".join(a_manquants)))
        if a_locked:
            st.info("⏳ Extraction en cours — paramètres verrouillés.")

        a_lancer = st.button(
            "🔍 Lancer l'extraction",
            disabled=not a_pret or a_locked,
            use_container_width=True,
            type="primary",
            key="a_lancer_btn",
        )

        st.caption(
            "⚠️ MNT RGE ALTI IGN — sol nu, végétation et bâtiments non modélisés. "
            "Précision adaptée au pré-design. Relevé terrain recommandé en phase APD."
        )

    # ── Déclenchement : double-rerun pour verrouiller avant de calculer ───────
    if a_lancer and a_uploaded is not None:
        st.session_state.a__uploaded_bytes = a_uploaded.read()
        st.session_state.a__uploaded_name = a_uploaded.name
        st.session_state.a__resolution = a_resolution
        st.session_state.a__buffer_m = a_buffer_m
        st.session_state.a__technologie = technologie_a
        st.session_state.a__puissance = puissance_a
        st.session_state.a_extracting = True
        st.session_state.a_txt_bytes = None
        st.session_state.a_export = None
        reinitialiser_exports("a")
        st.rerun()

    # ── Colonne droite : rendu (calcul + résultats) ───────────────────────────
    with col_a_result:

        if not st.session_state.a_extracting and st.session_state.a_txt_bytes is None:
            # ── État initial : placeholder centré ──
            st.markdown(
                """
                <div style='text-align:center; padding: 80px 40px; color: #999;'>
                    <div style='font-size:60px'>&#128506;</div>
                    <p style='margin-top:16px; font-size:15px; line-height:1.8'>
                    Renseignez les paramètres à gauche<br>
                    et cliquez sur <strong>Lancer l'extraction</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif st.session_state.a_extracting:
            # ── Traitement ──
            _res = st.session_state.a__resolution
            _buf = st.session_state.a__buffer_m

            try:
                with st.spinner("Chargement du shapefile…"):
                    gdf = load_shapefile_from_zip(
                        io.BytesIO(st.session_state.a__uploaded_bytes)
                    )
                    nom_shp = os.path.splitext(st.session_state.a__uploaded_name)[0]

                with st.spinner("Calcul de l'emprise…"):
                    bbox_l93 = calculer_bbox_l93(gdf, _buf)

                # Téléchargement MNT (barre de progression intégrée dans fetch_mnt)
                mnt, transform = fetch_mnt(bbox_l93, _res)

                # Masquage du MNT par le polygone bufférisé (forme réelle, pas rectangle)
                with st.spinner("Application du masque de zone…"):
                    gdf_l93 = gdf.to_crs("EPSG:2154")
                    geom_buffered = gdf_l93.union_all().buffer(_buf if _buf > 0 else 0)
                    masque = rio_rasterize(
                        [(geom_buffered.__geo_interface__, 1)],
                        out_shape=mnt.shape,
                        transform=transform,
                        fill=0,
                        dtype=np.uint8,
                    )
                    mnt = mnt.astype(np.float32)
                    mnt[masque == 0] = np.nan

                # Détection 1m réel vs 5m rééchantillonné par l'API IGN
                resolution_effective = _res
                if _res == 1:
                    if not detecter_resolution_reelle(mnt):
                        resolution_effective = 5
                        st.session_state.a_warning_resolution = (
                            "Données 1 m non disponibles sur cette zone — "
                            "l'API IGN a retourné du 5 m rééchantillonné (nearest-neighbour). "
                            "La carte affichée correspond à une interpolation de ces données "
                            "à la granularité 1 m. "
                            "Deux exports disponibles ci-dessous : fichier 5 m non interpolé "
                            "et fichier 1 m interpolé."
                        )
                    else:
                        st.session_state.a_warning_resolution = None
                else:
                    st.session_state.a_warning_resolution = None
                st.session_state.a_resolution_effective = resolution_effective

                with st.spinner("Calcul des pentes et orientations…"):
                    # Lissage gaussien pour l'affichage carte (sigma=5 px) si résolution 1m
                    # L'export TXT utilise le MNT brut non lissé
                    if _res == 1:
                        mask_nan = np.isnan(mnt)
                        mnt_tmp = mnt.copy()
                        mnt_tmp[mask_nan] = float(np.nanmean(mnt))
                        mnt_affichage = gaussian_filter(mnt_tmp.astype(np.float32), sigma=5.0)
                        mnt_affichage[mask_nan] = np.nan
                    else:
                        mnt_affichage = mnt
                    pentes, orientations = calculer_pentes_et_orientation(mnt_affichage, transform)

                # mnt = MNT brut (non lissé) — utilisé pour les courbes de niveau
                _z_min_a = float(np.nanmin(mnt))
                _z_max_a = float(np.nanmax(mnt))
                st.session_state.a_intervalle_courbes = max(
                    1, round((_z_max_a - _z_min_a) / 20)
                )
                # Les deux cartes dépendent de l'état des couches (et des
                # paramètres de zones pour la topographique) : elles sont
                # produites à l'affichage par _carte_html_cached(), qui les
                # mémorise pour chaque combinaison.

                # Données conservées pour régénérer les PNG/PDF à la demande
                st.session_state.a_export = {
                    "pentes": pentes,
                    "orientations": orientations,
                    "transform": transform,
                    "mnt": mnt,
                    "gdf": gdf,
                    "technologie": st.session_state.a__technologie,
                    "puissance": st.session_state.a__puissance,
                    "nom_fichier": st.session_state.a__uploaded_name,
                    "nom_site": nom_shp,
                }

                with st.spinner("Préparation de l'export TXT…"):
                    if _res == 1 and resolution_effective == 5:
                        # Cas interpolé : deux exports distincts
                        st.session_state.a_txt_bytes = exporter_txt(mnt, transform, pas=5)
                        st.session_state.a_nom_fichier = f"{nom_shp}_5m.txt"
                        st.session_state.a_txt_bytes_interp = exporter_txt(mnt, transform, pas=1)
                        st.session_state.a_nom_fichier_interp = f"{nom_shp}_1m_interpole.txt"
                    else:
                        # Cas normal : un seul export à la résolution effective
                        st.session_state.a_txt_bytes = exporter_txt(mnt, transform, pas=1)
                        st.session_state.a_nom_fichier = f"{nom_shp}_{resolution_effective}m.txt"
                        st.session_state.a_txt_bytes_interp = None
                        st.session_state.a_nom_fichier_interp = None

                st.session_state.a_extracting = False
                st.rerun()

            except Exception as e:
                st.session_state.a_extracting = False
                st.error(f"❌ Erreur lors de l'extraction : {e}")

        else:
            # ── Résultats : téléchargement en priorité, carte en dessous ──
            if st.session_state.a_warning_resolution:
                st.warning(st.session_state.a_warning_resolution)

            res_eff = st.session_state.a_resolution_effective or st.session_state.a__resolution
            req_res = st.session_state.a__resolution
            if req_res == 1 and res_eff == 5:
                st.success(
                    "✅ Extraction terminée — résolution effective : 5 m "
                    "(données 1 m indisponibles sur cette zone)"
                )
            else:
                st.success(f"✅ Extraction terminée — résolution effective : {res_eff} m")

            if st.session_state.a_intervalle_courbes:
                st.caption(
                    f"Courbes de niveau tous les "
                    f"**{st.session_state.a_intervalle_courbes} m** "
                    f"(calculé automatiquement selon la plage d'altitude du site)."
                )

            if st.session_state.a_txt_bytes_interp:
                # Cas interpolé : deux boutons côte à côte
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label=f"⬇️ {st.session_state.a_nom_fichier} — 5 m non interpolé",
                        data=st.session_state.a_txt_bytes,
                        file_name=st.session_state.a_nom_fichier,
                        mime="text/plain",
                        use_container_width=True,
                        type="primary",
                        key="a_dl_5m",
                    )
                with col_dl2:
                    st.download_button(
                        label=f"⬇️ {st.session_state.a_nom_fichier_interp} — 1 m interpolé (nearest-neighbour)",
                        data=st.session_state.a_txt_bytes_interp,
                        file_name=st.session_state.a_nom_fichier_interp,
                        mime="text/plain",
                        use_container_width=True,
                        key="a_dl_1m_interp",
                    )
            else:
                # Cas normal : un seul bouton
                st.download_button(
                    label=f"⬇️ Télécharger le fichier TXT PVCase ({st.session_state.a_nom_fichier})",
                    data=st.session_state.a_txt_bytes,
                    file_name=st.session_state.a_nom_fichier,
                    mime="text/plain",
                    use_container_width=True,
                    type="primary",
                    key="a_dl_main",
                )

            if st.session_state.a_export:
                # 1 · topographie (avec zones planes paramétrables)
                st.divider()
                _html_a, _zones_a, _params_a = preparer_carte(
                    "a", MODE_TOPOGRAPHIE, st.session_state.a_export
                )
                afficher_section_carte(
                    volet="a",
                    mode=MODE_TOPOGRAPHIE,
                    carte_html=_html_a,
                    export=st.session_state.a_export,
                    zones=_zones_a,
                    zones_params=_params_a,
                )
                # 2 · constructibilité
                st.divider()
                _html_ca, _, _ = preparer_carte(
                    "a", MODE_CONSTRUCTIBILITE, st.session_state.a_export
                )
                afficher_section_carte(
                    volet="a",
                    mode=MODE_CONSTRUCTIBILITE,
                    carte_html=_html_ca,
                    export=st.session_state.a_export,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  VOLET B — Conversion données drone UNITe
# ══════════════════════════════════════════════════════════════════════════════

with onglet_b:

    st.caption(
        "🚁 Ce volet convertit les courbes de niveau issues de relevés drone "
        "(GeoJSON, Shapefile, GeoTIFF ou DXF) au format PVCase Ground Mount "
        "(_MULTIPLE _POINT, X,Y,Z en L93)."
    )

    b_locked = st.session_state.b_extracting

    col_b_params, col_b_result = st.columns([1, 2], gap="large")

    # ── Colonne gauche : saisie des paramètres ────────────────────────────────
    with col_b_params:

        # ── 1. Fichier de données drone ───────────────────────────────────────
        st.subheader("1 · Fichier de données drone")
        b_uploaded = st.file_uploader(
            "Déposer le fichier de courbes de niveau (ZIP ou GeoJSON)",
            type=["zip", "geojson", "json"],
            disabled=b_locked,
            key="b_uploader",
            help=(
                "Formats acceptés :\n"
                "- ZIP contenant un .geojson ou un shapefile (.shp)\n"
                "- GeoJSON direct (.geojson ou .json)\n\n"
                "Les géométries doivent être de type LineString ou MultiLineString "
                "avec coordonnées Z (courbes de niveau 3D issues d'un relevé drone)."
            ),
        )

        # ── 2. Résolution d'interpolation ─────────────────────────────────────
        st.subheader("2 · Résolution d'interpolation")
        b_resolution = st.select_slider(
            "Résolution de la grille MNT interpolée",
            options=[round(i * 0.25, 2) for i in range(1, 21)],
            value=1.0,
            format_func=lambda x: f"{x:.2f} m",
            disabled=b_locked,
            key="b_resolution_slider",
            help=(
                "Résolution de la grille régulière résultante :\n"
                "0.25 m — très haute résolution (grille dense, calcul long)\n"
                "1.00 m — recommandé (bon compromis précision / taille)\n"
                "5.00 m — grille légère, compatible avec la résolution ESQ"
            ),
        )

        # ── 3. Mode ───────────────────────────────────────────────────────────
        st.subheader("3 · Mode")
        b_mode = st.radio(
            "Mode de traitement",
            options=["Convertir les données", "Convertir les données et afficher la carte"],
            disabled=b_locked,
            key="b_mode_radio",
            help=(
                "Convertir les données : pipeline court — export TXT uniquement, rapide.\n"
                "Convertir les données et afficher la carte : pipeline complet — carte "
                "interactive avec pentes, courbes de niveau et options projet."
            ),
        )

        if b_mode == "Convertir les données et afficher la carte":
            # ── Zone d'implantation (optionnel) ──────────────────────────────
            st.subheader("Zone d'implantation — optionnel")
            b_zone = st.file_uploader(
                "ZIP shapefile de la zone (optionnel)",
                type=["zip"],
                disabled=b_locked,
                key="b_zone_uploader",
                help=(
                    "Si fourni : le contour du site s'affiche en jaune pointillé "
                    "et l'overlay de pentes est masqué au polygone.\n"
                    "Si absent : l'enveloppe convexe des courbes est utilisée."
                ),
            )
            if b_zone is not None:
                st.caption("✅ Zone fournie — le masque sera appliqué au MNT interpolé.")
                b_buffer_m = st.select_slider(
                    "Buffer autour de la zone d'implantation",
                    options=list(range(0, 31, 5)),
                    value=10,
                    format_func=lambda x: f"{x} m",
                    disabled=b_locked,
                    key="b_buffer_slider",
                    help="Zone étendue autour du shapefile visible sur la carte et incluse dans l'export TXT.",
                )
            else:
                b_buffer_m = 0

            # ── Caractéristiques projet ───────────────────────────────────────
            st.subheader("Caractéristiques projet")
            technologie_b = st.radio(
                "Type de structure",
                options=["FIXE", "TRACKERS", "OMBRIÈRES"],
                horizontal=True,
                index=None,
                disabled=b_locked,
                key="b_technologie_radio",
            )
            puissance_b = st.radio(
                "Puissance",
                options=["<5MWc", "≥5MWc"],
                horizontal=True,
                index=None,
                disabled=b_locked,
                key="b_puissance_radio",
            )
        else:
            b_zone = None
            b_buffer_m = 0
            technologie_b = None
            puissance_b = None

        st.divider()

        # ── Validation ────────────────────────────────────────────────────────
        b_manquants = []
        if b_uploaded is None:
            b_manquants.append("fichier de courbes")
        if b_mode == "Convertir les données et afficher la carte":
            if technologie_b is None:
                b_manquants.append("type de structure")
            if puissance_b is None:
                b_manquants.append("puissance du projet")

        b_pret = len(b_manquants) == 0
        if not b_pret and not b_locked:
            st.info("En attente : {}".format(", ".join(b_manquants)))
        if b_locked:
            st.info("⏳ Conversion en cours — paramètres verrouillés.")

        b_lancer = st.button(
            "⚙️ Lancer la conversion",
            disabled=not b_pret or b_locked,
            use_container_width=True,
            type="primary",
            key="b_lancer_btn",
        )

        st.caption(
            "⚠️ L'interpolation linéaire retourne NaN hors de l'enveloppe convexe "
            "des courbes sources. Vérifiez la carte avant export."
        )

    # ── Déclenchement : double-rerun pour verrouiller avant de calculer ───────
    if b_lancer and b_uploaded is not None:
        st.session_state.b__uploaded_bytes = b_uploaded.read()
        st.session_state.b__uploaded_name = b_uploaded.name
        st.session_state.b__resolution_drone = b_resolution
        st.session_state.b__zone_bytes = b_zone.read() if b_zone is not None else None
        st.session_state.b__zone_name = b_zone.name if b_zone is not None else None
        st.session_state.b__buffer_m = b_buffer_m
        st.session_state.b__mode = b_mode
        st.session_state.b__technologie = technologie_b
        st.session_state.b__puissance = puissance_b
        st.session_state.b_extracting = True
        st.session_state.b_txt_bytes = None
        st.session_state.b_export = None
        reinitialiser_exports("b")
        st.rerun()

    # ── Colonne droite : rendu (calcul + résultats) ───────────────────────────
    with col_b_result:

        if not st.session_state.b_extracting and st.session_state.b_txt_bytes is None:
            # ── État initial : placeholder centré ──
            st.markdown(
                """
                <div style='text-align:center; padding: 80px 40px; color: #999;'>
                    <div style='font-size:60px'>🚁</div>
                    <p style='margin-top:16px; font-size:15px; line-height:1.8'>
                    Déposez un fichier de courbes de niveau drone<br>
                    et cliquez sur <strong>Lancer la conversion</strong>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif st.session_state.b_extracting:
            # ── Traitement ──
            _fname = st.session_state.b__uploaded_name
            _raw = st.session_state.b__uploaded_bytes
            _res_drone = st.session_state.b__resolution_drone
            _ext = os.path.splitext(_fname)[1].lower().lstrip(".")
            nom_source = os.path.splitext(_fname)[0]
            _mode = st.session_state.b__mode

            try:
                with st.spinner("Chargement des courbes de niveau…"):
                    gdf_courbes = charger_courbes(_raw, _ext)

                with st.spinner("Extraction des points XYZ…"):
                    points_xyz = extraire_points_xyz(gdf_courbes)
                    n_pts = len(points_xyz)

                with st.spinner(
                    f"Interpolation du MNT à {_res_drone:.2f} m "
                    f"({n_pts:,} points sources)…"
                ):
                    mnt_b, transform_b = interpoler_mnt(points_xyz, _res_drone)

                if _mode == "Convertir les données et afficher la carte":
                    # Zone d'implantation : shapefile fourni → masque + contour
                    #                       absent         → enveloppe convexe du nuage
                    _zone_bytes = st.session_state.b__zone_bytes
                    if _zone_bytes is not None:
                        with st.spinner("Chargement de la zone d'implantation…"):
                            gdf_zone_b = load_shapefile_from_zip(io.BytesIO(_zone_bytes))
                            gdf_zone_b_l93 = gdf_zone_b.to_crs("EPSG:2154")
                            _buf_b = st.session_state.b__buffer_m or 0
                            geom_zone = (
                                gdf_zone_b_l93.union_all().buffer(_buf_b)
                                if _buf_b > 0
                                else gdf_zone_b_l93.union_all()
                            )
                            masque_b = rio_rasterize(
                                [(geom_zone.__geo_interface__, 1)],
                                out_shape=mnt_b.shape,
                                transform=transform_b,
                                fill=0,
                                dtype=np.uint8,
                            )
                            mnt_b[masque_b == 0] = np.nan
                        gdf_site_b = gdf_zone_b
                    else:
                        hull = MultiPoint(
                            list(zip(points_xyz[:, 0], points_xyz[:, 1]))
                        ).convex_hull
                        gdf_site_b = gpd.GeoDataFrame(geometry=[hull], crs="EPSG:2154")

                    with st.spinner("Calcul des pentes et orientations…"):
                        # Lissage gaussien pour les résolutions sub-métriques ou 1 m
                        if _res_drone <= 1.0:
                            mask_nan_b = np.isnan(mnt_b)
                            mnt_tmp_b = mnt_b.copy()
                            valid_vals = mnt_b[~mask_nan_b]
                            fill_val = float(valid_vals.mean()) if len(valid_vals) > 0 else 0.0
                            mnt_tmp_b[mask_nan_b] = fill_val
                            mnt_affichage_b = gaussian_filter(
                                mnt_tmp_b.astype(np.float32), sigma=5.0
                            )
                            mnt_affichage_b[mask_nan_b] = np.nan
                        else:
                            mnt_affichage_b = mnt_b
                        pentes_b, orientations_b = calculer_pentes_et_orientation(
                            mnt_affichage_b, transform_b
                        )

                    _z_min_b = float(np.nanmin(mnt_b))
                    _z_max_b = float(np.nanmax(mnt_b))
                    st.session_state.b_intervalle_courbes = max(
                        1, round((_z_max_b - _z_min_b) / 20)
                    )
                    # Cartes produites à l'affichage (avec cache) : elles
                    # dépendent de l'état des couches et des zones.

                    # Données conservées pour régénérer les PNG/PDF à la demande
                    st.session_state.b_export = {
                        "pentes": pentes_b,
                        "orientations": orientations_b,
                        "transform": transform_b,
                        "mnt": mnt_b,
                        "gdf": gdf_site_b,
                        "technologie": st.session_state.b__technologie,
                        "puissance": st.session_state.b__puissance,
                        "nom_fichier": st.session_state.b__uploaded_name,
                        "nom_site": nom_source,
                    }
                else:
                    # Mode convert : pipeline court, pas de génération de carte
                    st.session_state.b_export = None
                    st.session_state.b_intervalle_courbes = None

                with st.spinner("Préparation de l'export TXT…"):
                    res_str = f"{_res_drone:g}"
                    nom_fichier_b = f"{nom_source}_drone_{res_str}m.txt"
                    st.session_state.b_txt_bytes = exporter_txt(mnt_b, transform_b, pas=1)
                    st.session_state.b_nom_fichier = nom_fichier_b
                    st.session_state.b_n_points = n_pts
                    st.session_state.b_shape = f"{mnt_b.shape[0]}×{mnt_b.shape[1]}"

                st.session_state.b_extracting = False
                st.rerun()

            except Exception as e:
                st.session_state.b_extracting = False
                st.error(f"❌ Erreur lors de la conversion : {e}")

        else:
            # ── Résultats : téléchargement en priorité, carte en dessous ──
            n_pts = st.session_state.b_n_points
            shape = st.session_state.b_shape
            res_used = st.session_state.b__resolution_drone
            if n_pts and shape and res_used is not None:
                st.success(
                    f"✅ Conversion terminée — "
                    f"{n_pts:,} points sources · "
                    f"grille {shape} · "
                    f"résolution {res_used:g} m"
                )
            else:
                st.success("✅ Conversion terminée")

            if st.session_state.b_intervalle_courbes:
                st.caption(
                    f"Courbes de niveau tous les "
                    f"**{st.session_state.b_intervalle_courbes} m** "
                    f"(calculé automatiquement selon la plage d'altitude du site)."
                )

            st.download_button(
                label=(
                    f"⬇️ Télécharger le fichier TXT PVCase "
                    f"({st.session_state.b_nom_fichier})"
                ),
                data=st.session_state.b_txt_bytes,
                file_name=st.session_state.b_nom_fichier,
                mime="text/plain",
                use_container_width=True,
                type="primary",
                key="b_dl_main",
            )

            if st.session_state.b_export:
                # 1 · topographie (avec zones planes paramétrables)
                st.divider()
                _html_b, _zones_b, _params_b = preparer_carte(
                    "b", MODE_TOPOGRAPHIE, st.session_state.b_export
                )
                afficher_section_carte(
                    volet="b",
                    mode=MODE_TOPOGRAPHIE,
                    carte_html=_html_b,
                    export=st.session_state.b_export,
                    zones=_zones_b,
                    zones_params=_params_b,
                )
                # 2 · constructibilité
                st.divider()
                _html_cb, _, _ = preparer_carte(
                    "b", MODE_CONSTRUCTIBILITE, st.session_state.b_export
                )
                afficher_section_carte(
                    volet="b",
                    mode=MODE_CONSTRUCTIBILITE,
                    carte_html=_html_cb,
                    export=st.session_state.b_export,
                )
