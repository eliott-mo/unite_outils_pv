"""
generate_map.py — Générateur de carte de situation CETI
UNITe PV — AO CRE Sol Période 9

Ce fichier expose une fonction principale : generer_carte()
appelée par l'interface Streamlit (app.py).
Il peut aussi être lancé directement depuis un terminal
en renseignant les paramètres dans le bloc PARAMETRES ci-dessous.
"""

import os
import re
import io
import math
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from shapely.ops import unary_union
from pyproj import Transformer


# ════════════════════════════════════════════════════════════════
# PARAMETRES — utilisés uniquement en lancement direct (terminal)
# ════════════════════════════════════════════════════════════════
SHP_PATH       = ""  # chemin local uniquement
NOM_PROJET     = "EMO 21 — Gissey / Darcey"
RECUL_CAPTEURS = 10
BUFFER_CARTE   = 650
ECHELLE        = 5000
URBANISME      = ""
FOND_AERIEN    = True
OUTPUT_DIR     = ""  # chemin local uniquement
DPI            = 150


# ════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ════════════════════════════════════════════════════════════════

def to_dms(deg, is_lat):
    hemi = ("N" if deg >= 0 else "S") if is_lat else ("E" if deg >= 0 else "O")
    deg  = abs(deg)
    d = int(deg); m = int((deg - d) * 60); s = (deg - d - m / 60) * 3600
    return "{:02d}\u00b0{:02d}'{:05.2f}'' {}".format(d, m, s, hemi)


def draw_geom(ax, geom, fc="none", ec="black", lw=1.5,
              alpha_fill=0.3, ls="-", zorder=2):
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


# ════════════════════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ════════════════════════════════════════════════════════════════

def generer_carte(shp_path, nom_projet, recul_capteurs=10, urbanisme="",
                  echelle=5000, fond_aerien=True, dpi=150,
                  buffer_carte=650, tick_deg=0.005):
    """
    Génère la carte de situation CETI et retourne les bytes PNG.

    Paramètres
    ----------
    shp_path        : chemin vers le fichier .shp du terrain d'implantation
    nom_projet      : nom du projet (affiché dans le titre)
    recul_capteurs  : recul en mètres pour la zone capteurs (défaut : 10 m)
    urbanisme       : texte libre pour l'encart urbanisme (défaut : vide)
    echelle         : dénominateur de l'échelle (défaut : 5000)
    fond_aerien     : True = fond IGN Géoportail, False = fond neutre
    dpi             : résolution de l'image (défaut : 150)
    buffer_carte    : rayon autour du terrain pour l'emprise carte (défaut : 650 m)
    tick_deg        : intervalle des ticks WGS84 en degrés (défaut : 0.005)

    Retourne
    --------
    bytes PNG de la carte générée
    """

    # Géométries
    gdf      = gpd.read_file(shp_path)
    terrain  = unary_union(gdf.geometry)
    capteurs = terrain.buffer(-recul_capteurs)
    buf600   = terrain.buffer(600)

    minx, miny, maxx, maxy = terrain.bounds
    pad = buffer_carte
    x0, x1 = minx - pad, maxx + pad
    y0, y1 = miny - pad, maxy + pad
    geo_w, geo_h = x1 - x0, y1 - y0

    # Taille figure à l'échelle exacte
    MARGIN_L, MARGIN_R   = 0.90, 0.15
    MARGIN_TOP, MARGIN_BOT = 0.95, 0.65

    ax_w_in  = geo_w / echelle / 0.0254
    ax_h_in  = geo_h / echelle / 0.0254
    fig_w_in = ax_w_in + MARGIN_L + MARGIN_R
    fig_h_in = ax_h_in + MARGIN_TOP + MARGIN_BOT
    echelle_lbl = "1 / {:,}".format(echelle).replace(",", "\u202f")

    print("Axe : {:.1f}\u00d7{:.1f} cm  |  Figure : {}\u00d7{} px".format(
        ax_w_in * 2.54, ax_h_in * 2.54,
        int(fig_w_in * dpi), int(fig_h_in * dpi)))

    # Points extrémaux WGS84 DMS
    tr  = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    pts = np.array(terrain.convex_hull.exterior.coords)
    extremal = []
    for lbl, fn in [("N", lambda p: np.argmax(p[:, 1])),
                    ("S", lambda p: np.argmin(p[:, 1])),
                    ("E", lambda p: np.argmax(p[:, 0])),
                    ("O", lambda p: np.argmin(p[:, 0]))]:
        x, y = pts[fn(pts)]
        lon, lat = tr.transform(x, y)
        extremal.append({"label": "Pt {}".format(lbl), "x": x, "y": y,
                         "lat": to_dms(lat, True), "lon": to_dms(lon, False)})

    # Figure
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

    # Fond IGN
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
            _el = _dst_tf.c
            _et = _dst_tf.f
            _er = _el + _dst_tf.a * _dw
            _eb = _et + _dst_tf.e * _dh
            ax.imshow(_img_l93, extent=[_el, _er, _eb, _et],
                      zorder=0, alpha=0.85, aspect="auto")
            fond_ok = True
            print("\u2705 Fond IGN charg\u00e9")
        except Exception as e:
            print("\u26a0\ufe0f  Fond IGN indisponible ({}) \u2014 fond neutre".format(type(e).__name__))

    if not fond_ok:
        ax.set_facecolor("#f0ede8")
        ax.grid(True, linestyle=":", linewidth=0.5, color="#aaaaaa", alpha=0.6)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_autoscale_on(False)

    # Couches
    draw_geom(ax, buf600,   fc="#9970ab", ec="#762a83", lw=1.8, alpha_fill=0.10, ls=(0,(4,5)), zorder=2)
    draw_geom(ax, terrain,  fc="#6baed6", ec="#08519c", lw=2.5, alpha_fill=0.30, ls="-",       zorder=3)
    draw_geom(ax, capteurs, fc="none",    ec="#d94701", lw=2.0, alpha_fill=0,    ls=(0,(6,4)), zorder=5)

    # Points extrémaux
    c_txt = "white" if fond_ok else "#222"
    off = {"N": (10, 12), "S": (10, -65), "E": (10, 5), "O": (-148, 5)}
    for pt in extremal:
        dx, dy = off.get(pt["label"][-1], (10, 10))
        ax.plot(pt["x"], pt["y"], "o", color="#990000", markersize=8,
                zorder=7, markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate("{}\n{}\n{}".format(pt["label"], pt["lat"], pt["lon"]),
                    xy=(pt["x"], pt["y"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, color="#660000", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              alpha=0.92, ec="#990000", lw=0.8), zorder=8)

    # Barre d'échelle
    sb_x = x1 - geo_w * 0.04 - 500
    sb_y = y0 + geo_h * 0.035
    ax.annotate("", xy=(sb_x + 500, sb_y), xytext=(sb_x, sb_y),
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5", color=c_txt, lw=2))
    ax.text(sb_x + 250, sb_y + geo_h * 0.013, "500 m", ha="center",
            fontsize=9, fontweight="bold", color=c_txt,
            bbox=dict(fc="#00000066" if fond_ok else "white", alpha=0.7, ec="none"))

    # Flèche Nord
    arr_h = geo_h * 0.07
    nx = x0 + geo_w * 0.07
    ny = y1 - geo_h * 0.18
    ax.annotate("", xy=(nx, ny + arr_h), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color=c_txt, lw=4.5))
    ax.text(nx, ny + arr_h * 1.18, "N", ha="center", fontsize=14, fontweight="bold", color=c_txt)

    # Légende
    h1 = mpatches.Patch(facecolor="#6baed6", edgecolor="#08519c", alpha=0.55, linewidth=2, label="Terrain d'implantation")
    h2 = mlines.Line2D([], [], color="#d94701", linewidth=2, linestyle=(0,(6,4)), label="Zone d'implantation des capteurs")
    h3 = mlines.Line2D([], [], color="#762a83", linewidth=1.8, linestyle=(0,(4,5)), label="P\u00e9rim\u00e8tre de 600 m")
    h4 = mlines.Line2D([], [], color="#990000", marker="o", linestyle="None", markersize=8, label="Points extr\u00e9maux WGS84")
    ax.legend(handles=[h1,h2,h3,h4], loc="lower left", fontsize=9, framealpha=0.93, edgecolor="#cccccc")

    # Encart urbanisme
    if urbanisme.strip():
        ax.text(x1 - geo_w * 0.02, y1 - geo_h * 0.02,
                "Urbanisme\n{}\n{}".format("\u2500" * 22, urbanisme),
                ha="right", va="top", fontsize=8, linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.6", fc="white", alpha=0.93, ec="#aaaaaa", lw=0.8), zorder=9)

    # Axes WGS84
    _tr_wgs = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    _tr_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    _cx_mid, _cy_mid = (x0+x1)/2, (y0+y1)/2
    _lon_mid, _lat_mid = _tr_wgs.transform(_cx_mid, _cy_mid)

    def _fmt_lon(v, _): lon, _ = _tr_wgs.transform(v, _cy_mid); return "{:.3f}\u00b0 {}".format(abs(lon), "E" if lon>=0 else "O")
    def _fmt_lat(v, _): _, lat = _tr_wgs.transform(_cx_mid, v); return "{:.3f}\u00b0 {}".format(abs(lat), "N" if lat>=0 else "S")

    lon0, lat0 = _tr_wgs.transform(x0, y0)
    lon1, lat1 = _tr_wgs.transform(x1, y1)
    lon_ticks = np.arange(math.ceil(lon0/tick_deg)*tick_deg, math.floor(lon1/tick_deg)*tick_deg+tick_deg/2, tick_deg)
    lat_ticks = np.arange(math.ceil(lat0/tick_deg)*tick_deg, math.floor(lat1/tick_deg)*tick_deg+tick_deg/2, tick_deg)
    ax.set_xticks([_tr_l93.transform(lon, _lat_mid)[0] for lon in lon_ticks])
    ax.set_yticks([_tr_l93.transform(_lon_mid, lat)[1] for lat in lat_ticks])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_lat))
    ax.set_xlabel("Longitude (WGS84)", fontsize=9)
    ax.set_ylabel("Latitude (WGS84)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Titre
    src = "\u00a9 IGN G\u00e9oportail" if fond_ok else "fond neutre"
    ax.set_title("UNITe PV \u2014 {}\nPlan de situation  |  \u00c9chelle : {}  |  {}".format(
        nom_projet, echelle_lbl, src), fontsize=15, fontweight="bold", pad=16)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # Export bytes
    buf = io.BytesIO()
    plt.savefig(buf, dpi=dpi, facecolor="white", format="png")
    plt.close()
    buf.seek(0)

    print("Surface : {:.2f} ha  |  \u00c9chelle : {}".format(terrain.area/10000, echelle_lbl))
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
    )
    _slug = re.sub(r"[^\w]+", "_", NOM_PROJET).strip("_")
    output_png = os.path.join(OUTPUT_DIR, "UNITe_CETI_PV_{}.png".format(_slug))
    with open(output_png, "wb") as f:
        f.write(png_bytes)
    print("\n\u2705 Carte : {}".format(output_png))
