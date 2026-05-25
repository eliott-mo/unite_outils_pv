"""Extraction Topo RGE ALTI IGN — export TXT PVCase Ground Mount."""

import io
import os
import base64
import time
import zipfile
import tempfile
import json
import math

import numpy as np
import geopandas as gpd
import folium
import requests
import streamlit as st
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds as rt_from_bounds
from rasterio.crs import CRS
import rasterio.warp
from scipy.ndimage import gaussian_filter
from streamlit_folium import st_folium

# ── Constantes ────────────────────────────────────────────────────────────────
_URL_ALTI = "https://data.geopf.fr/altimetrie/1.0/calcul/alti/rest/elevation.json"
_BATCH_SIZE = 5000
_RATE_LIMIT = 0.2  # secondes entre batches (≤ 5 req/s)

# Palette pente — seuils Fixe (identiques à PV Topo Analyzer)
_PALETTE = {
    0: (76, 175, 80),   # vert   — Favorable    < 5 %
    1: (255, 193, 7),   # jaune  — Acceptable   5–10 %
    2: (255, 152, 0),   # orange — Contraignant 10–15 %
    3: (244, 67, 54),   # rouge  — Exclusion    > 15 %
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


# ── 4. Calcul des pentes et orientations (algorithme Horn) ───────────────────

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


# ── 5. Classification et rendu RGBA ──────────────────────────────────────────

def _classifier(pente: np.ndarray) -> np.ndarray:
    cls = np.zeros_like(pente, dtype=np.uint8)
    cls[pente >= 5] = 1
    cls[pente >= 10] = 2
    cls[pente >= 15] = 3
    cls[np.isnan(pente)] = 255
    return cls


def _to_rgba(classes: np.ndarray, opacite: float = 0.65) -> np.ndarray:
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    alpha = int(opacite * 255)
    for val, (r, g, b) in _PALETTE.items():
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

    lon_min = t_dst.c
    lat_max = t_dst.f
    lon_max = lon_min + w_dst * t_dst.a
    lat_min = lat_max + h_dst * t_dst.e
    return dst, [[lat_min, lon_min], [lat_max, lon_max]]


def _img_b64(rgba: np.ndarray) -> str:
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── 6. Génération carte Folium ────────────────────────────────────────────────

def creer_carte(pentes: np.ndarray, orientations: np.ndarray, transform_l93, gdf_site: gpd.GeoDataFrame) -> folium.Map:
    """
    Carte Folium avec :
    - Fond satellite ESRI (défaut)
    - Overlay couleur de pentes (zone + buffer)
    - Contour jaune pointillé du shapefile original (sans buffer)
    - Légende fixe des classes
    """
    gdf_wgs84 = gdf_site.to_crs("EPSG:4326")
    union = gdf_wgs84.union_all()
    centre = [union.centroid.y, union.centroid.x]
    b = union.bounds
    site_bounds = [[b[1], b[0]], [b[3], b[2]]]

    carte = folium.Map(location=centre, zoom_start=15, tiles=None, control_scale=True)

    # Fond OSM puis ESRI — le dernier ajouté est actif par défaut
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(carte)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, DigitalGlobe, GeoEye, i-cubed, USDA FSA, USGS, AEX",
        name="Satellite (ESRI)",
        overlay=False,
        control=True,
    ).add_to(carte)

    # Overlay pentes (zone + buffer)
    classes = _classifier(pentes)
    rgba = _to_rgba(classes)
    rgba_wgs84, bounds_pente = _rgba_l93_to_wgs84(rgba, transform_l93)

    fg = folium.FeatureGroup(name="Pentes", show=True)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{_img_b64(rgba_wgs84)}",
        bounds=bounds_pente,
        opacity=1.0,
        interactive=False,
    ).add_to(fg)
    fg.add_to(carte)

    # Contour du shapefile initial — jaune pointillé (hors LayerControl : toujours visible)
    fg_site = folium.FeatureGroup(name="Contour du site", control=False, show=True)
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

    folium.LayerControl(position="bottomright", collapsed=False).add_to(carte)
    carte.fit_bounds(site_bounds)
    _ajouter_legende(carte)
    _ajouter_tooltip_hover(carte, pentes, orientations, transform_l93)
    return carte


def _ajouter_legende(carte: folium.Map) -> None:
    sw = "display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle;margin-right:5px"
    html = f"""
<style>
#leg {{
  position:fixed; right:10px; bottom:160px; z-index:1000;
  background:white; padding:9px 11px; border-radius:6px;
  font-size:11px; min-width:195px;
  box-shadow:0 1px 5px rgba(0,0,0,.3); color:#212121;
}}
.leaflet-control-attribution {{ font-size:8px !important; opacity:.7; }}
</style>
<div id="leg">
  <b>Pente — seuils Fixe</b><br><br>
  <span style="background:#4CAF50;{sw}"></span>Favorable (&lt; 5 %)<br>
  <span style="background:#FFC107;{sw}"></span>Acceptable (5 – 10 %)<br>
  <span style="background:#FF9800;{sw}"></span>Contraignant (10 – 15 %)<br>
  <span style="background:#F44336;{sw}"></span>Exclusion (&gt; 15 %)
  <hr style="margin:7px 0;border:none;border-top:1px solid #e0e0e0">
  <div style="display:flex;align-items:center;gap:6px">
    <span style="display:inline-block;width:20px;height:0;border-top:2px dashed #FFE600;flex-shrink:0"></span>
    <span>Contour du site</span>
  </div>
</div>
"""
    carte.get_root().html.add_child(folium.Element(html))


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

    # Bounds WGS84 du raster L93 (conversion pyproj — approximation linéaire suffisante)
    _tr = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    x_min, y_max_l93 = transform_l93.c, transform_l93.f
    x_max = x_min + w * transform_l93.a
    y_min = y_max_l93 + h * transform_l93.e  # transform.e < 0
    lon_w, lat_s = _tr.transform(x_min, y_min)
    lon_e, lat_n = _tr.transform(x_max, y_max_l93)

    map_var = carte.get_name()

    html = f"""<div id="hover-info" style="
  position:fixed;bottom:24px;right:12px;z-index:1000;
  background:rgba(0,0,0,0.65);color:white;
  padding:6px 12px;border-radius:4px;
  font-size:13px;font-family:monospace;
  pointer-events:none;display:none;"></div>
<script>
(function() {{
  var mapObj = {map_var};
  var PENTES = {_pentes_to_js(pentes_js)};
  var ORIENTATIONS = {_orientations_to_js(orientations_js)};
  var BOUNDS = {{north:{lat_n:.8f},south:{lat_s:.8f},east:{lon_e:.8f},west:{lon_w:.8f}}};
  var ROWS = {rows_js};
  var COLS = {cols_js};

  mapObj.on('mousemove', function(e) {{
    var row = Math.floor((BOUNDS.north - e.latlng.lat) / (BOUNDS.north - BOUNDS.south) * ROWS);
    var col = Math.floor((e.latlng.lng - BOUNDS.west) / (BOUNDS.east - BOUNDS.west) * COLS);
    var el = document.getElementById('hover-info');
    if (!el) return;
    if (row < 0 || row >= ROWS || col < 0 || col >= COLS) {{
      el.style.display = 'none';
      return;
    }}
    var pente = PENTES[row][col];
    var orient = ORIENTATIONS[row][col];
    if (pente !== null) {{
      el.innerHTML = 'Pente : <b>' + pente + ' %</b> — Orientation : <b>' + orient + '</b>';
      el.style.display = 'block';
    }} else {{
      el.style.display = 'none';
    }}
  }});

  mapObj.on('mouseout', function() {{
    var el = document.getElementById('hover-info');
    if (el) el.style.display = 'none';
  }});
}})();
</script>"""
    carte.get_root().html.add_child(folium.Element(html))


# ── 8. Export TXT PVCase ──────────────────────────────────────────────────────

def exporter_txt(mnt: np.ndarray, transform_l93) -> bytes:
    """
    Format PVCase Ground Mount :
        _MULTIPLE _POINT
        X,Y,Z  (Lambert 93, virgule, 2 décimales)
    Couvre tous les pixels valides du MNT (zone + buffer).
    """
    rows, cols = mnt.shape
    col_idx = np.tile(np.arange(cols), rows)
    row_idx = np.repeat(np.arange(rows), cols)

    z_flat = mnt.ravel()
    valid = ~np.isnan(z_flat)

    x = transform_l93.c + (col_idx[valid] + 0.5) * transform_l93.a
    y = transform_l93.f + (row_idx[valid] + 0.5) * transform_l93.e
    z = z_flat[valid]

    if len(z) == 0:
        raise ValueError("Aucun point valide à exporter.")

    buf = io.BytesIO()
    buf.write(b"_MULTIPLE _POINT\n")
    for xi, yi, zi in zip(x, y, z):
        buf.write(f"{xi:.2f},{yi:.2f},{zi:.2f}\n".encode())
    return buf.getvalue()


# ── Interface Streamlit ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Extraction Topo RGE ALTI IGN",
    page_icon="⛰️",
    layout="wide",
)

st.title("⛰️ Extraction Topo RGE ALTI IGN")
st.caption("MNT RGE ALTI IGN · Export TXT PVCase Ground Mount · UNITe")
st.divider()

# ── Session state ─────────────────────────────────────────────────────────────
# Résultats affichage
for _k in ("carte_html", "txt_bytes", "nom_fichier"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# État de l'extraction
if "extracting" not in st.session_state:
    st.session_state.extracting = False

# Paramètres sauvegardés pour survivre au st.rerun() de verrouillage
for _k in ("_uploaded_bytes", "_uploaded_name", "_resolution", "_buffer_m"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# Raccourci : True pendant tout le calcul → désactive les widgets
locked = st.session_state.extracting

col_params, col_result = st.columns([1, 2], gap="large")

# ── Colonne gauche : saisie des paramètres ────────────────────────────────────
with col_params:

    # ── 1. Shapefile ──────────────────────────────────────────────────────────
    st.subheader("1 · Shapefile")
    uploaded = st.file_uploader(
        "Déposer le fichier .zip contenant le shapefile de la zone d'implantation",
        type=["zip"],
        disabled=locked,
        help="Le zip doit contenir les fichiers .shp, .dbf et .shx de la zone de projet.",
    )

    # ── 2. Résolution ─────────────────────────────────────────────────────────
    st.subheader("2 · Résolution")
    resolution = st.radio(
        "Résolution du MNT",
        options=[5, 1],
        format_func=lambda x: (
            "5 m — rapide, recommandé pré-design"
            if x == 5 else
            "1 m — précis, ~30–60 s pour 10 ha"
        ),
        disabled=locked,
        help="La résolution 5 m est suffisante pour le pré-design. Passer à 1 m pour une analyse fine en phase APS/APD.",
    )

    # ── 3. Buffer ─────────────────────────────────────────────────────────────
    st.subheader("3 · Buffer")
    buffer_m = st.select_slider(
        "Emprise autour de la zone d'implantation",
        options=list(range(0, 301, 50)),
        value=100,
        format_func=lambda x: f"{x} m",
        disabled=locked,
        help="Zone étendue autour du shapefile couverte par le MNT et visible sur la carte.",
    )

    st.divider()

    # ── Validation ────────────────────────────────────────────────────────────
    manquants = []
    if uploaded is None:
        manquants.append("shapefile ZIP")

    pret = len(manquants) == 0
    if not pret and not locked:
        st.info("En attente : {}".format(", ".join(manquants)))
    if locked:
        st.info("⏳ Extraction en cours — paramètres verrouillés.")

    lancer = st.button(
        "🔍 Lancer l'extraction",
        disabled=not pret or locked,
        use_container_width=True,
        type="primary",
    )

    st.caption(
        "⚠️ MNT RGE ALTI IGN — sol nu, végétation et bâtiments non modélisés. "
        "Précision adaptée au pré-design. Relevé terrain recommandé en phase APD."
    )

# ── Déclenchement : double-run pour verrouiller avant de calculer ─────────────
# Run 1 (clic bouton) : on sauvegarde les paramètres et on bascule extracting=True,
#   puis st.rerun() → les widgets s'affichent disabled sur le run suivant.
# Run 2 (extraction) : extracting=True, lancer=False → le calcul s'exécute.
if lancer and uploaded is not None:
    st.session_state._uploaded_bytes = uploaded.read()   # lire avant le rerun
    st.session_state._uploaded_name = uploaded.name
    st.session_state._resolution = resolution
    st.session_state._buffer_m = buffer_m
    st.session_state.extracting = True
    st.session_state.carte_html = None
    st.session_state.txt_bytes = None
    st.rerun()

# ── Colonne droite : rendu (calcul + résultats) ───────────────────────────────
with col_result:

    if not st.session_state.extracting and st.session_state.txt_bytes is None:
        # ── État initial : placeholder centré ──
        st.markdown(
            """
            <div style='text-align:center; padding: 80px 40px; color: #999;'>
                <div style='font-size:60px'>🗺️</div>
                <p style='margin-top:16px; font-size:15px; line-height:1.8'>
                Renseignez les paramètres à gauche<br>
                et cliquez sur <strong>Lancer l'extraction</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif st.session_state.extracting:
        # ── Traitement ──
        _res = st.session_state._resolution
        _buf = st.session_state._buffer_m

        try:
            with st.spinner("Chargement du shapefile…"):
                # On passe un BytesIO car l'UploadedFile d'origine n'existe plus
                # après le st.rerun() de verrouillage
                gdf = load_shapefile_from_zip(io.BytesIO(st.session_state._uploaded_bytes))
                nom_shp = os.path.splitext(st.session_state._uploaded_name)[0]

            with st.spinner("Calcul de l'emprise…"):
                bbox_l93 = calculer_bbox_l93(gdf, _buf)

            # Téléchargement MNT (barre de progression intégrée dans fetch_mnt)
            mnt, transform = fetch_mnt(bbox_l93, _res)

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

            with st.spinner("Génération de la carte…"):
                carte = creer_carte(pentes, orientations, transform, gdf)
                # Correction bug folium : render() évite le I/O on closed file
                st.session_state.carte_html = carte.get_root().render()

            with st.spinner("Préparation de l'export TXT…"):
                st.session_state.txt_bytes = exporter_txt(mnt, transform)
                st.session_state.nom_fichier = f"{nom_shp}_{_res}m.txt"

            st.session_state.extracting = False
            st.rerun()  # rerun final pour déverrouiller les widgets

        except Exception as e:
            st.session_state.extracting = False
            st.error(f"❌ Erreur lors de l'extraction : {e}")

    else:
        # ── Résultats : téléchargement en priorité, carte en dessous ──
        st.success("✅ Extraction terminée !")
        st.download_button(
            label=f"⬇️ Télécharger le fichier TXT PVCase ({st.session_state.nom_fichier})",
            data=st.session_state.txt_bytes,
            file_name=st.session_state.nom_fichier,
            mime="text/plain",
            use_container_width=True,
            type="primary",
        )
        if st.session_state.carte_html:
            st.components.v1.html(st.session_state.carte_html, height=580, scrolling=False)