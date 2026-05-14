"""
ceti_generate_map.py — Générateur de carte de situation CETI
UNITe PV — AO CRE Sol Période 9

Expose generer_carte() appelée par app.py.
Peut aussi être lancé directement depuis un terminal.
"""

import os, re, io, math, zipfile, tempfile
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.ops import unary_union
from pyproj import Transformer


# ════════════════════════════════════════════════════════════════
# COULEURS CONVENTIONNELLES PLU
# ════════════════════════════════════════════════════════════════
COULEURS_PLU = {
    "U":  {"fc": "#FF6B6B", "ec": "#CC2200", "label": "Zone U — Urbaine"},
    "AU": {"fc": "#FFB347", "ec": "#CC6600", "label": "Zone AU — À urbaniser"},
    "A":  {"fc": "#F9E04B", "ec": "#CC9900", "label": "Zone A — Agricole"},
    "N":  {"fc": "#74C476", "ec": "#2D7A2D", "label": "Zone N — Naturelle"},
}
COULEUR_PLU_DEFAUT = {"fc": "#CCCCCC", "ec": "#888888", "label": "Zone — Autre"}


def charger_zones_urbanisme(x0, y0, x1, y1):
    """
    Interroge le Geoportail de l'Urbanisme (GPU) via API Carto IGN.
    Endpoint : https://apicarto.ign.fr/api/gpu/zone-urba

    Parametres : emprise en Lambert-93 (x0,y0 = coin SO, x1,y1 = coin NE)
    Retourne
    --------
    GeoDataFrame (n > 0)   : zones PLU trouvees
    GeoDataFrame vide      : API OK mais aucune zone = commune sous RNU
    None                   : echec de l'API (reseau, timeout...)
    """
    import json, requests
    from pyproj import Transformer as _Tr

    tr = _Tr.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon0, lat0 = tr.transform(x0, y0)
    lon1, lat1 = tr.transform(x1, y1)

    geom_bbox = {
        "type": "Polygon",
        "coordinates": [[[lon0, lat0], [lon1, lat0],
                          [lon1, lat1], [lon0, lat1],
                          [lon0, lat0]]],
    }
    params = {
        "geom":   json.dumps(geom_bbox, separators=(",", ":")),
        "_limit": 1000,
    }
    try:
        r = requests.get(
            "https://apicarto.ign.fr/api/gpu/zone-urba",
            params=params, timeout=30,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        features = r.json().get("features", [])
        if not features:
            print("GPU : aucune zone PLU (commune sous RNU probable)")
            # GeoDataFrame vide = signal RNU
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                                    crs="EPSG:4326")
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=2154)
        print("GPU : {} zones PLU chargees".format(len(gdf)))
        return gdf
    except Exception as e:
        print("GPU : echec ({}) : {}".format(type(e).__name__, str(e)[:80]))
        return None   # None = erreur reseau


def type_zone(libelle):
    """Retourne la clé de couleur (U/AU/A/N) depuis le libellé de zone."""
    if libelle is None:
        return None
    lib = str(libelle).upper().strip()
    for key in ["AU", "U", "A", "N"]:   # AU avant U pour éviter les faux positifs
        if lib.startswith(key):
            return key
    return None


# ════════════════════════════════════════════════════════════════
# PARAMETRES — lancement direct terminal uniquement
# ════════════════════════════════════════════════════════════════
SHP_PATH       = ""   # chemin local uniquement
NOM_PROJET     = "EMO 21 — Gissey / Darcey"
RECUL_CAPTEURS = 10
BUFFER_CARTE   = 650
ECHELLE        = 5000
URBANISME      = ""
FOND_AERIEN    = True
OUTPUT_DIR     = ""   # chemin local uniquement
DPI            = 150
ZH_PATH        = None   # chemin couche zones humides (optionnel)
ELEMENTS_PATH  = None   # chemin couche éléments techniques KML (optionnel)


# ════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════

def to_dms(deg, is_lat):
    hemi = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "O")
    deg  = abs(deg)
    d = int(deg); m = int((deg - d) * 60); s = (deg - d - m / 60) * 3600
    return "{:02d}\u00b0{:02d}'{:05.2f}'' {}".format(d, m, s, hemi)


def calcul_extremaux(terrain, tr, echelle=5000):
    """
    Selectionne jusqu'a 4 points cardinaux (N/S/E/O) sur l'enveloppe convexe
    du terrain avec deduplication maximin et offsets d'annotation radiaux.

    Algorithme
    ----------
    1. Calculer les 4 extrema cardinaux (N=max Y, S=min Y, E=max X, O=min X).
    2. Selectionner dans l'ordre N, S, E, O.
       Si le candidat est a moins de MIN_DIST d'un point deja retenu,
       le remplacer par le sommet du hull non encore selectionne qui maximise
       la distance minimale a l'ensemble des points retenus.
       Si aucun remplacement valide -> point omis.
       MIN_DIST = max(5 % diagonale terrain, 1 m).
    3. Offset radial depuis le centroide (8 octants).
    4. Anti-chevauchement iteratif des encarts (push perpendiculaire).
    """
    hull_pts = np.array(terrain.convex_hull.exterior.coords[:-1])
    n        = len(hull_pts)
    cx, cy   = terrain.centroid.x, terrain.centroid.y

    minx, miny, maxx, maxy = terrain.bounds
    diag     = math.hypot(maxx - minx, maxy - miny)
    MIN_DIST = max(diag * 0.05, 1.0)

    def hdist(i, j):
        return math.hypot(hull_pts[i][0] - hull_pts[j][0],
                          hull_pts[i][1] - hull_pts[j][1])

    def min_dist_to_sel(k, sel):
        return min(hdist(k, s) for s in sel)

    cardinal = [
        ("A", int(np.argmax(hull_pts[:, 1]))),   # nord
        ("B", int(np.argmin(hull_pts[:, 1]))),   # sud
        ("C", int(np.argmax(hull_pts[:, 0]))),   # est
        ("D", int(np.argmin(hull_pts[:, 0]))),   # ouest
    ]

    sel_idx = []
    sel_lbl = []

    for lbl, idx in cardinal:
        ok = (not sel_idx) or all(hdist(idx, si) >= MIN_DIST for si in sel_idx)
        if ok:
            sel_idx.append(idx)
            sel_lbl.append(lbl)
        else:
            # Remplacement maximin : sommet non retenu le plus eloigne des selectionnes
            remaining = [k for k in range(n) if k not in sel_idx]
            if remaining:
                best = max(remaining, key=lambda k: min_dist_to_sel(k, sel_idx))
                if min_dist_to_sel(best, sel_idx) >= MIN_DIST:
                    sel_idx.append(best)
                    sel_lbl.append(lbl)
                else:
                    print("  [extremaux] Pt {} omis (terrain trop compact)".format(lbl))
            else:
                print("  [extremaux] Pt {} omis (pas de sommet disponible)".format(lbl))

    # \u2500\u2500 Offsets radiaux (8 octants depuis centroide) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    M_PER_PT = echelle * 0.0254 / 72
    BOX_W    = 92
    BOX_H    = 52
    PUSH     = 8

    result = []
    for lbl, idx in zip(sel_lbl, sel_idx):
        px, py = hull_pts[idx]
        lon, lat = tr.transform(px, py)

        ux = px - cx; uy = py - cy
        norm  = max(math.hypot(ux, uy), 1e-6)
        ux   /= norm; uy /= norm
        angle = math.degrees(math.atan2(uy, ux))

        if   -22.5 <= angle <  22.5:   adx, ady = PUSH,             -BOX_H // 2
        elif  22.5 <= angle <  67.5:   adx, ady = PUSH,              PUSH
        elif  67.5 <= angle < 112.5:   adx, ady = -BOX_W // 2,       PUSH
        elif 112.5 <= angle < 157.5:   adx, ady = -(BOX_W + PUSH),   PUSH
        elif angle >= 157.5 or angle < -157.5:
                                       adx, ady = -(BOX_W + PUSH),  -BOX_H // 2
        elif -157.5 <= angle < -112.5: adx, ady = -(BOX_W + PUSH), -(BOX_H + PUSH)
        elif -112.5 <= angle <  -67.5: adx, ady = -BOX_W // 2,     -(BOX_H + PUSH)
        else:                          adx, ady = PUSH,             -(BOX_H + PUSH)

        result.append({
            "label":  "Pt {}".format(lbl),
            "x": px, "y": py,
            "lat":    to_dms(lat, True),
            "lon":    to_dms(lon, False),
            "ann_dx": adx, "ann_dy": ady,
            "_ux":    ux, "_uy": uy,
        })

    # \u2500\u2500 Anti-chevauchement perpendiculaire iteratif \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    GAP      = 6
    STEP     = 10
    MAX_ITER = 30
    THRESH_X = (BOX_W + GAP) * M_PER_PT
    THRESH_Y = (BOX_H + GAP) * M_PER_PT

    for _ in range(MAX_ITER):
        moved = False
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                ri, rj = result[i], result[j]
                cxi = ri["x"] + (ri["ann_dx"] + BOX_W / 2) * M_PER_PT
                cyi = ri["y"] + (ri["ann_dy"] + BOX_H / 2) * M_PER_PT
                cxj = rj["x"] + (rj["ann_dx"] + BOX_W / 2) * M_PER_PT
                cyj = rj["y"] + (rj["ann_dy"] + BOX_H / 2) * M_PER_PT
                if abs(cxi - cxj) < THRESH_X and abs(cyi - cyj) < THRESH_Y:
                    sx = cxj - cxi; sy = cyj - cyi
                    snorm = max(math.hypot(sx, sy), 1e-6)
                    sx /= snorm; sy /= snorm
                    pi_x, pi_y = -ri["_uy"],  ri["_ux"]
                    pj_x, pj_y = -rj["_uy"],  rj["_ux"]
                    dot_i = pi_x * sx + pi_y * sy
                    dot_j = pj_x * sx + pj_y * sy
                    si = -math.copysign(1.0, dot_i) if abs(dot_i) > 1e-9 else  1.0
                    sj =  math.copysign(1.0, dot_j) if abs(dot_j) > 1e-9 else -1.0
                    result[i]["ann_dx"] = int(ri["ann_dx"] + si * pi_x * STEP)
                    result[i]["ann_dy"] = int(ri["ann_dy"] + si * pi_y * STEP)
                    result[j]["ann_dx"] = int(rj["ann_dx"] + sj * pj_x * STEP)
                    result[j]["ann_dy"] = int(rj["ann_dy"] + sj * pj_y * STEP)
                    moved = True
        if not moved:
            break

    for pt in result:
        pt.pop("_ux", None)
        pt.pop("_uy", None)

    return result


def charger_geodata(path):
    """
    Charge un fichier géographique quel que soit son format :
    - .zip  → shapefile extrait dans un dossier temporaire
    - .kml  → lu directement par geopandas
    - .geojson / .json → lu directement
    Retourne un GeoDataFrame en EPSG:2154 (Lambert-93).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".zip":
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmpdir)
        shp = None
        for root, _, files in os.walk(tmpdir):
            for f in files:
                if f.endswith(".shp"):
                    shp = os.path.join(root, f)
                    break
        if shp is None:
            raise ValueError("Aucun .shp trouvé dans le zip : {}".format(path))
        gdf = gpd.read_file(shp)

    elif ext == ".kml":
        import fiona
        # Activer le driver KML (désactivé par défaut dans fiona)
        fiona.drvsupport.supported_drivers["KML"]  = "rw"
        fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
        gdf = gpd.read_file(path, driver="KML")

    elif ext in (".geojson", ".json"):
        gdf = gpd.read_file(path)

    else:
        # Tentative générique
        gdf = gpd.read_file(path)

    # Reprojection en Lambert-93 si nécessaire
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    if gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs(epsg=2154)

    return gdf


def draw_geom(ax, geom, fc="none", ec="black", lw=1.5,
              alpha_fill=0.3, ls="-", zorder=2):
    """Trace une géométrie Shapely sur un axe matplotlib."""
    geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for g in geoms:
        if g.geom_type == "Polygon":
            xs, ys = g.exterior.xy
            if fc != "none":
                ax.fill(xs, ys, color=fc, alpha=alpha_fill, zorder=zorder)
            ax.plot(xs, ys, color=ec, linewidth=lw, linestyle=ls, zorder=zorder + 1)
        elif g.geom_type == "LineString":
            xs, ys = g.xy
            ax.plot(xs, ys, color=ec, linewidth=lw, linestyle=ls, zorder=zorder)
        elif g.geom_type == "Point":
            ax.plot(g.x, g.y, "o", color=ec, markersize=4, zorder=zorder)


def draw_hatch(ax, geom, ec="#0077BB", fc="#AEE4FF", hatch="////",
               alpha_fill=0.25, lw=1.5, zorder=6):
    """
    Trace un polygone hachuré (zones humides).
    Le hachure est appliqué via matplotlib directement.
    """
    geoms = list(geom.geoms) if geom.geom_type.startswith("Multi") else [geom]
    for g in geoms:
        if g.geom_type != "Polygon":
            continue
        xs, ys = g.exterior.xy
        ax.fill(xs, ys, fc=fc, alpha=alpha_fill, zorder=zorder)
        ax.fill(xs, ys, fc="none", hatch=hatch, ec=ec,
                linewidth=0.3, alpha=0.6, zorder=zorder + 1)
        ax.plot(xs, ys, color=ec, linewidth=lw, zorder=zorder + 2)


# ════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ════════════════════════════════════════════════════════════════

def generer_carte(shp_path, nom_projet, recul_capteurs=10, urbanisme="",
                  echelle=5000, fond_aerien=True, dpi=150, buffer_carte=650,
                  tick_deg=0.005, zh_path=None, elements_path=None,
                  urba_terrain=False, urba_buffer=True):
    """
    Génère la carte de situation CETI et retourne les bytes PNG.

    Paramètres
    ----------
    shp_path        : chemin .shp terrain d'implantation
    nom_projet      : nom affiché dans le titre et le nom de fichier
    recul_capteurs  : recul zone capteurs en mètres (défaut 10)
    urbanisme       : texte libre encart urbanisme
    echelle         : dénominateur échelle (défaut 5000)
    fond_aerien     : True = IGN Géoportail
    dpi             : résolution image (défaut 150)
    buffer_carte    : rayon emprise carte en mètres (défaut 650)
    tick_deg        : intervalle ticks WGS84 en degrés (défaut 0.005)
    zh_path         : chemin couche zones humides (.zip, .kml, .geojson) — optionnel
    elements_path   : chemin couche éléments techniques (.kml) — optionnel

    Retourne : bytes PNG
    """

    # ── Géométries terrain ────────────────────────────────────────────────────
    gdf      = gpd.read_file(shp_path)
    terrain  = unary_union(gdf.geometry)
    capteurs = terrain.buffer(-recul_capteurs)
    buf600   = terrain.buffer(600)

    minx, miny, maxx, maxy = terrain.bounds
    pad = buffer_carte
    x0, x1 = minx - pad, maxx + pad
    y0, y1 = miny - pad, maxy + pad
    geo_w, geo_h = x1 - x0, y1 - y0

    # ── Chargement couches optionnelles ───────────────────────────────────────
    gdf_zh  = charger_geodata(zh_path)       if zh_path       else None
    gdf_elts = charger_geodata(elements_path) if elements_path else None

    # Séparer les éléments techniques par type de géométrie
    elts_poly       = None  # panneaux (polygones si dispo)
    elts_lines      = None  # pistes et autres lignes
    elts_pts        = None  # locaux techniques (points)
    capteurs_depuis_kml = None  # zone capteurs reconstruite depuis les lignes du KML

    if gdf_elts is not None and len(gdf_elts) > 0:
        from shapely.ops import polygonize as _polygonize
        mask_poly  = gdf_elts.geometry.geom_type.isin(["Polygon","MultiPolygon"])
        mask_lines = gdf_elts.geometry.geom_type.isin(["LineString","MultiLineString"])
        mask_pts   = gdf_elts.geometry.geom_type.isin(["Point","MultiPoint"])

        if mask_poly.any():
            elts_poly = unary_union(gdf_elts[mask_poly].geometry)

        if mask_lines.any():
            all_lines = unary_union(gdf_elts[mask_lines].geometry)
            elts_lines = all_lines
            # Reconstruction de la zone capteurs par polygonize des lignes
            polys_from_lines = list(_polygonize(all_lines))
            if polys_from_lines:
                # Buffer +5m / -5m pour fusionner les rangées proches en zones
                capteurs_depuis_kml = unary_union(polys_from_lines).buffer(5).buffer(-5)
                print("Zone capteurs reconstruite depuis KML : {:.2f} ha".format(
                    capteurs_depuis_kml.area / 10000))

        if mask_pts.any():
            elts_pts = unary_union(gdf_elts[mask_pts].geometry)

    # ZH : on decoupe a l emprise du terrain d implantation
    # (la couche ZH peut etre tres etendue)
    if gdf_zh is not None:
        zh_raw  = unary_union(gdf_zh.geometry)
        zh_geom = zh_raw.intersection(terrain)
        if zh_geom.is_empty:
            print("Avertissement : la couche ZH n intersecte pas le terrain — ignoree")
            zh_geom = None
        else:
            print("ZH decoupee au terrain : {:.2f} ha".format(zh_geom.area / 10000))
    else:
        zh_geom = None

    # ── Taille figure à l'échelle exacte ──────────────────────────────────────
    MARGIN_L, MARGIN_R     = 0.90, 0.15
    MARGIN_TOP, MARGIN_BOT = 0.95, 0.65

    ax_w_in  = geo_w / echelle / 0.0254
    ax_h_in  = geo_h / echelle / 0.0254
    fig_w_in = ax_w_in + MARGIN_L + MARGIN_R
    fig_h_in = ax_h_in + MARGIN_TOP + MARGIN_BOT
    echelle_lbl = "1 / {:,}".format(echelle).replace(",", " ")

    print("Axe : {:.1f}\u00d7{:.1f} cm | Figure : {}\u00d7{} px".format(
        ax_w_in * 2.54, ax_h_in * 2.54,
        int(fig_w_in * dpi), int(fig_h_in * dpi)))

    # ── Points extrémaux WGS84 DMS ────────────────────────────────────────────
    tr       = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    extremal = calcul_extremaux(terrain, tr, echelle)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax  = fig.add_axes([
        MARGIN_L   / fig_w_in,
        MARGIN_BOT / fig_h_in,
        ax_w_in    / fig_w_in,
        ax_h_in    / fig_h_in,
    ])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_autoscale_on(False)

    # ── Fond IGN ──────────────────────────────────────────────────────────────
    fond_ok = False
    ign_url = (
        "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile"
        "&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS"
        "&STYLE=normal&TILEMATRIXSET=PM"
        "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg"
    )
    if fond_aerien:
        try:
            import contextily as ctx
            from rasterio.transform import from_bounds as _rio_bounds
            from rasterio.warp import (reproject as _rio_proj, Resampling,
                                       calculate_default_transform as _rio_cdt)
            from rasterio.crs import CRS as _CRS

            _to3857 = Transformer.from_crs("EPSG:2154", "EPSG:3857", always_xy=True)
            _bx0, _by0 = _to3857.transform(x0, y0)
            _bx1, _by1 = _to3857.transform(x1, y1)
            _img_wm, _ext = ctx.bounds2img(_bx0, _by0, _bx1, _by1,
                                            zoom="auto", source=ign_url, ll=False)
            _H, _W, _nb = _img_wm.shape
            _src_crs = _CRS.from_epsg(3857)
            _dst_crs = _CRS.from_epsg(2154)
            _west, _east, _south, _north = _ext
            _src_tf  = _rio_bounds(_west, _south, _east, _north, _W, _H)
            _dst_tf, _dw, _dh = _rio_cdt(_src_crs, _dst_crs, _W, _H,
                                           left=_west, bottom=_south,
                                           right=_east, top=_north)
            _img_l93 = np.zeros((_dh, _dw, _nb), dtype=_img_wm.dtype)
            for _b in range(_nb):
                _rio_proj(_img_wm[:, :, _b], _img_l93[:, :, _b],
                          src_transform=_src_tf, src_crs=_src_crs,
                          dst_transform=_dst_tf, dst_crs=_dst_crs,
                          resampling=Resampling.bilinear)
            _el = _dst_tf.c; _et = _dst_tf.f
            _er = _el + _dst_tf.a * _dw; _eb = _et + _dst_tf.e * _dh
            ax.imshow(_img_l93, extent=[_el, _er, _eb, _et],
                      zorder=0, alpha=0.85, aspect="auto")
            fond_ok = True
            print("\u2705 Fond IGN charg\u00e9")
        except Exception as e:
            print("\u26a0\ufe0f  Fond IGN indisponible ({}) \u2014 fond neutre".format(type(e).__name__))

    if not fond_ok:
        ax.set_facecolor("#f0ede8")
        ax.grid(True, linestyle=":", linewidth=0.5, color="#aaaaaa", alpha=0.6)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_autoscale_on(False)

    # ── Zones PLU (Géoportail de l'Urbanisme — API Carto IGN) ────────────────
    gdf_plu     = None
    legende_plu = {}      # clé tz -> couleur (pour légende, sans doublons)
    rnu_detecte = False

    if urba_terrain or urba_buffer:
        if urba_buffer:
            bx0, by0, bx1, by1 = buf600.bounds
        else:
            bx0, by0, bx1, by1 = terrain.bounds
        gdf_plu = charger_zones_urbanisme(bx0, by0, bx1, by1)

    # Déterminer la géométrie de clipping une seule fois
    if urba_terrain or urba_buffer:
        if urba_terrain and urba_buffer:
            clip_geom = buf600
        elif urba_terrain:
            clip_geom = terrain
        else:
            clip_geom = buf600
    else:
        clip_geom = None

    if gdf_plu is None and clip_geom is not None:
        # ── API indisponible : pas de couleur, pas de hachure ───────────────
        # Conformément à la demande : "pas d'info = pas de couleur"
        # On laisse la zone sans remplissage, juste la note en légende
        pass   # rien à tracer, la note "Sans couleur = info non disponible" suffira

    if gdf_plu is not None and clip_geom is not None:
        if len(gdf_plu) == 0:
            # ── Commune sous RNU : hachure grise ────────────────────────────
            rnu_detecte = True
            draw_hatch(ax, clip_geom,
                       ec="#999999", fc="#CCCCCC", hatch="////",
                       alpha_fill=0.25, lw=0.8, zorder=1)
            rnu_pt = clip_geom.representative_point()
            ax.text(rnu_pt.x, rnu_pt.y, "RNU",
                    fontsize=13, fontweight="bold", color="#555555",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.45", fc="white",
                              alpha=0.90, ec="#999999", lw=1.2),
                    zorder=20)

        elif len(gdf_plu) > 0:
            # ── PLU numerise : dessiner chaque zone ─────────────────────────
            col_type = "typezone" if "typezone" in gdf_plu.columns else None
            col_lib  = "libelle"  if "libelle"  in gdf_plu.columns else None
            SEUIL_LABEL_M2 = 5000   # zone < 5 000 m² = pas de label

            for _, row in gdf_plu.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    geom = geom.intersection(clip_geom)
                except Exception:
                    continue
                if geom is None or geom.is_empty:
                    continue

                # Determiner la categorie (U / AU / A / N)
                # type_zone() normalise AUs->AU, UB->U, etc.
                tz = None
                if col_type:
                    raw = str(row[col_type]).strip() if row[col_type] else ""
                    tz = type_zone(raw)
                if tz is None and col_lib:
                    raw = str(row[col_lib]).strip() if row[col_lib] else ""
                    tz = type_zone(raw)

                if tz is not None:
                    couleur = COULEURS_PLU[tz]
                    draw_geom(ax, geom, fc=couleur["fc"], ec=couleur["ec"],
                              lw=0.8, alpha_fill=0.30, ls="-", zorder=1)
                    if tz not in legende_plu:
                        legende_plu[tz] = couleur
                else:
                    # Zone sans categorie reconnue : bord gris, aucun remplissage
                    draw_geom(ax, geom, fc="none", ec="#AAAAAA",
                              lw=0.5, alpha_fill=0, ls="--", zorder=1)
                    legende_plu["?"] = None   # signale la presence de zones inconnues

                # ── Label libelle court (ex : Ns, AUx, A) ──────────────────
                lbl_txt = ""
                if col_lib:
                    v = str(row[col_lib]).strip() if row[col_lib] else ""
                    if v and v.lower() not in ("none", "nan"):
                        lbl_txt = v
                if not lbl_txt and col_type:
                    v = str(row[col_type]).strip() if row[col_type] else ""
                    if v and v.lower() not in ("none", "nan"):
                        lbl_txt = v

                if lbl_txt and geom.area >= SEUIL_LABEL_M2:
                    try:
                        rp = geom.representative_point()
                        rx, ry = float(rp.x), float(rp.y)
                    except Exception:
                        rx = None
                    if rx is not None and x0 <= rx <= x1 and y0 <= ry <= y1:
                        txt_color = (COULEURS_PLU[tz]["ec"]
                                     if tz else "#888888")
                        ax.text(rx, ry, lbl_txt,
                                fontsize=8, fontweight="bold",
                                color=txt_color,
                                ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.2",
                                          fc="white", alpha=0.80, ec="none"),
                                zorder=20, clip_on=True)


    # ── Zones non couvertes par le PLU = RNU ou hors périmètre GPU ─────────
    # On calcule la différence entre l'emprise clippée et l'union des zones PLU tracées
    if clip_geom is not None and gdf_plu is not None and len(gdf_plu) > 0:
        try:
            zones_plu_tracees = []
            for _, row in gdf_plu.iterrows():
                g = row.geometry
                if g is None or g.is_empty:
                    continue
                try:
                    g = g.intersection(clip_geom)
                except Exception:
                    continue
                if g and not g.is_empty:
                    zones_plu_tracees.append(g)

            if zones_plu_tracees:
                couverture_plu = unary_union(zones_plu_tracees)
                zone_non_couverte = clip_geom.difference(couverture_plu)
            else:
                zone_non_couverte = clip_geom

            # N'afficher le gris RNU que si la zone non couverte est significative
            SEUIL_RNU_M2 = 10000   # 1 ha minimum pour afficher
            if not zone_non_couverte.is_empty and zone_non_couverte.area > SEUIL_RNU_M2:
                rnu_detecte = True
                draw_hatch(ax, zone_non_couverte,
                           ec="#999999", fc="#CCCCCC", hatch="////",
                           alpha_fill=0.25, lw=0.8, zorder=1)
                # Label RNU centré sur la zone non couverte
                rnu_pt = zone_non_couverte.representative_point()
                rx, ry = float(rnu_pt.x), float(rnu_pt.y)
                if x0 <= rx <= x1 and y0 <= ry <= y1:
                    ax.text(rx, ry, "RNU",
                            fontsize=11, fontweight="bold", color="#555555",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.40", fc="white",
                                      alpha=0.90, ec="#999999", lw=1.0),
                            zorder=20)
        except Exception as e:
            print("Calcul zone RNU échoué : {}".format(e))

        # ── Couches de base ───────────────────────────────────────────────────────
    # Périmètre 600m — halo sombre + trait blanc pour lisibilité sur fond aérien
    draw_geom(ax, buf600, fc="none", ec="#333333", lw=3.5, alpha_fill=0, ls=(0,(4,5)), zorder=2)
    draw_geom(ax, buf600, fc="none", ec="#FFFFFF", lw=1.8, alpha_fill=0, ls=(0,(4,5)), zorder=2)
    draw_geom(ax, terrain,  fc="none",    ec="#CC0000", lw=2.5, alpha_fill=0,    ls="-",       zorder=10)
    # Zone capteurs : buffer 10m autour des éléments techniques si fournis,
    # sinon buffer négatif standard sur le terrain
    if gdf_elts is not None and len(gdf_elts) > 0:
        zone_capteurs = unary_union(gdf_elts.geometry).buffer(5)
    else:
        zone_capteurs = capteurs
    draw_geom(ax, zone_capteurs, fc="none", ec="#d94701",
              lw=1.2, alpha_fill=0, ls=(0,(4,3)), zorder=8)

    # ── Zones humides (si présentes) ──────────────────────────────────────────
    if zh_geom is not None:
        draw_hatch(ax, zh_geom,
                   ec="#005B9F",    # bleu foncé contour
                   fc="#AEE4FF",    # bleu clair remplissage
                   hatch="///",
                   alpha_fill=0.20,
                   lw=0.8,
                   zorder=2)

    # ── Éléments techniques (si présents) — formes du KML ────────────────────
    # Éléments techniques centrale PV — couche unique toutes géométries confondues
    if gdf_elts is not None and len(gdf_elts) > 0:
        for geom in gdf_elts.geometry:
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype in ("Polygon", "MultiPolygon"):
                draw_geom(ax, geom, fc="#E8A020", ec="#B87000",
                          lw=0.7, alpha_fill=0.35, ls="-", zorder=7)
            elif gtype in ("LineString", "MultiLineString"):
                draw_geom(ax, geom, fc="none", ec="#CC6600",
                          lw=0.7, alpha_fill=0, ls="-", zorder=7)
            # Points ignorés (locaux techniques non différenciables dans le KML)

    # ── Points extrémaux ──────────────────────────────────────────────────────
    c_txt = "white" if fond_ok else "#222"
    for pt in extremal:
        ax.plot(pt["x"], pt["y"], "o", color="#990000", markersize=8,
                zorder=9, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate("{}\n{}\n{}".format(pt["label"], pt["lat"], pt["lon"]),
                    xy=(pt["x"], pt["y"]),
                    xytext=(pt["ann_dx"], pt["ann_dy"]),
                    textcoords="offset points",
                    fontsize=7.5, color="#660000", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              alpha=0.92, ec="#990000", lw=0.8), zorder=15)

    # ── Barre d'échelle ───────────────────────────────────────────────────────
    sb_x = x1 - geo_w * 0.04 - 500
    sb_y = y0 + geo_h * 0.035
    ax.annotate("", xy=(sb_x + 500, sb_y), xytext=(sb_x, sb_y),
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", color=c_txt, lw=2))
    ax.text(sb_x + 250, sb_y + geo_h * 0.013, "500 m", ha="center",
            fontsize=9, fontweight="bold", color=c_txt,
            bbox=dict(fc="#00000066" if fond_ok else "white", alpha=0.7, ec="none"))

    # ── Flèche Nord ───────────────────────────────────────────────────────────
    arr_h = geo_h * 0.07
    nx = x0 + geo_w * 0.07
    ny = y1 - geo_h * 0.18
    ax.annotate("", xy=(nx, ny + arr_h), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color=c_txt, lw=4.5))
    ax.text(nx, ny + arr_h * 1.18, "N", ha="center", fontsize=14,
            fontweight="bold", color=c_txt)

    # ── Légende ───────────────────────────────────────────────────────────────
    legend_items = [
        mlines.Line2D([], [], color="#CC0000", linewidth=2.5,
                     label="Terrain d'implantation"),
    ]
    legend_items.append(
        mlines.Line2D([], [], color="#d94701", linewidth=2, linestyle=(0,(6,4)),
                      label="Zone d'implantation des capteurs")
    )
    legend_items += [
        mlines.Line2D([], [], color="#FFFFFF", linewidth=1.8, linestyle=(0,(4,5)),
                     markeredgecolor="#333333",
                     label="P\u00e9rim\u00e8tre de 600 m"),
        mlines.Line2D([], [], color="#990000", marker="o", linestyle="None",
                      markersize=8, label="Points de coordonn\u00e9es WGS84"),
    ]

    # Entrées légende conditionnelles ZH
    if zh_geom is not None:
        legend_items.append(
            mpatches.Patch(facecolor="#AEE4FF", edgecolor="#005B9F",
                           alpha=0.6, linewidth=1.8, hatch="////",
                           label="Zone(s) humide(s)")
        )

    # Entrées légende conditionnelles éléments techniques
    if gdf_elts is not None and len(gdf_elts) > 0:
        legend_items.append(
            mlines.Line2D([], [], color="#CC6600", linewidth=0.7,
                          label="\u00c9l\u00e9ments techniques centrale PV")
        )
    # Zones PLU — RNU en premier si détecté, puis une entrée par catégorie
    if rnu_detecte:
        legend_items.append(
            mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999999",
                           alpha=0.55, linewidth=0.8, hatch="////",
                           label="Commune sous RNU")
        )
    for tz, couleur in legende_plu.items():
        if tz == "?" or couleur is None:
            continue  # zones inconnues gérées par la note
        lbl = COULEURS_PLU[tz]["label"] if tz in COULEURS_PLU else "Zone urbanisme"
        legend_items.append(
            mpatches.Patch(facecolor=couleur["fc"], edgecolor=couleur["ec"],
                           alpha=0.4, linewidth=0.5, label=lbl)
        )
    # Note en bas si PLU demandé (disclaimer données GPU)
    _NOTE = "Sans couleur = info non disponible (Géoportail de l'Urbanisme)"
    _show_note = (urba_terrain or urba_buffer)
    if _show_note:
        legend_items.append(
            mlines.Line2D([], [], color="none", linewidth=0, label=_NOTE)
        )

    leg = ax.legend(handles=legend_items, loc="lower left",
                    fontsize=9, framealpha=0.93, edgecolor="#cccccc")

    # Mise en forme de la note (italique gris, handle invisible)
    if _show_note:
        for txt in leg.get_texts():
            if txt.get_text() == _NOTE:
                txt.set_color("#888888")
                txt.set_style("italic")
                txt.set_fontsize(7.5)
        try:
            handles = leg.legend_handles
        except AttributeError:
            handles = leg.legendHandles
        for handle, txt in zip(handles, leg.get_texts()):
            if txt.get_text() == _NOTE:
                handle.set_visible(False)

    # ── Encart urbanisme ──────────────────────────────────────────────────────
    if urbanisme.strip():
        ax.text(x1 - geo_w * 0.02, y1 - geo_h * 0.02,
                "Document d'urbanisme applicable\nau terrain d'implantation{}\n{}".format("\u2500" * 22, urbanisme),
                ha="right", va="top", fontsize=10, weight="bold", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.6", fc="white",
                          alpha=0.93, ec="#CC0000", lw=2.5), zorder=11)

    # ── Axes WGS84 ────────────────────────────────────────────────────────────
    _tr_wgs = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    _tr_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    _cx_mid, _cy_mid = (x0 + x1) / 2, (y0 + y1) / 2
    _lon_mid, _lat_mid = _tr_wgs.transform(_cx_mid, _cy_mid)

    def _fmt_lon(v, _):
        lon, _ = _tr_wgs.transform(v, _cy_mid)
        return "{:.3f}\u00b0 {}".format(abs(lon), "E" if lon >= 0 else "O")

    def _fmt_lat(v, _):
        _, lat = _tr_wgs.transform(_cx_mid, v)
        return "{:.3f}\u00b0 {}".format(abs(lat), "N" if lat >= 0 else "S")

    lon0, lat0 = _tr_wgs.transform(x0, y0)
    lon1, lat1 = _tr_wgs.transform(x1, y1)
    lon_ticks = np.arange(math.ceil(lon0 / tick_deg) * tick_deg,
                          math.floor(lon1 / tick_deg) * tick_deg + tick_deg / 2, tick_deg)
    lat_ticks = np.arange(math.ceil(lat0 / tick_deg) * tick_deg,
                          math.floor(lat1 / tick_deg) * tick_deg + tick_deg / 2, tick_deg)
    ax.set_xticks([_tr_l93.transform(lon, _lat_mid)[0] for lon in lon_ticks])
    ax.set_yticks([_tr_l93.transform(_lon_mid, lat)[1] for lat in lat_ticks])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude (WGS84)", fontsize=9)
    ax.set_ylabel("Latitude (WGS84)", fontsize=9)
    ax.tick_params(labelsize=8)

    # ── Titre ─────────────────────────────────────────────────────────────────
    src = "\u00a9 IGN G\u00e9oportail" if fond_ok else "fond neutre"
    ax.set_title(
        "UNITe PV \u2014 {}\nPlan de situation  |  \u00c9chelle\u202f: {}  |  {}".format(
            nom_projet, echelle_lbl, src),
        fontsize=15, fontweight="bold", pad=16)

    # ── Logo UNITe (haut-droite, meme hauteur que le titre) ───────────────────
    _LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo unite.png")
    if os.path.exists(_LOGO):
        from PIL import Image as _PILImg
        _logo = _PILImg.open(_LOGO).convert("RGBA")
        _target_h = max(int(MARGIN_TOP * dpi * 0.72), 10)
        _target_w = int(_target_h * _logo.width / _logo.height)
        _logo = _logo.resize((_target_w, _target_h), _PILImg.LANCZOS)
        _logo_arr = np.array(_logo)
        _axes_top_px = int((MARGIN_BOT + ax_h_in) * dpi)
        _yo = _axes_top_px + (int(MARGIN_TOP * dpi) - _target_h) // 2
        _fig_w_px = int(fig_w_in * dpi)
        _xo = _fig_w_px - _target_w - int(0.10 * dpi)
        fig.figimage(_logo_arr, xo=_xo, yo=_yo, origin="upper")

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)

    # ── Export bytes ──────────────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, facecolor="white", format="png")
    plt.close()
    buf.seek(0)

    print("Surface : {:.2f} ha  |  \u00c9chelle : {}".format(
        terrain.area / 10000, echelle_lbl))
    for pt in extremal:
        print("  {}  {}   {}".format(pt["label"], pt["lat"], pt["lon"]))

    return buf.read()


# ════════════════════════════════════════════════════════════════
# LANCEMENT DIRECT (terminal)
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    png_bytes = generer_carte(
        shp_path=SHP_PATH, nom_projet=NOM_PROJET,
        recul_capteurs=RECUL_CAPTEURS, urbanisme=URBANISME,
        echelle=ECHELLE, fond_aerien=FOND_AERIEN,
        dpi=DPI, buffer_carte=BUFFER_CARTE,
        zh_path=ZH_PATH, elements_path=ELEMENTS_PATH,
    )
    _slug = re.sub(r"[^\w]+", "_", NOM_PROJET).strip("_")
    output_png = os.path.join(OUTPUT_DIR, "UNITe_CETI_PV_{}.png".format(_slug))
    with open(output_png, "wb") as f:
        f.write(png_bytes)
    print("\n\u2705 Carte : {}".format(output_png))
