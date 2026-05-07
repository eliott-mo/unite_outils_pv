"""
app.py — Interface Streamlit — Générateur de carte CETI
UNITe PV — AO CRE Sol Période 9

Lancement local : streamlit run app.py
"""

import os, re, zipfile, tempfile
import streamlit as st
from ceti_generate_map import generer_carte

st.set_page_config(
    page_title="Générateur de carte CETI — UNITe PV",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Générateur de carte de situation CETI")
st.caption("AO CRE PV Sol · Période 9 · UNITe")
st.divider()

col_params, col_result = st.columns([1, 2], gap="large")

with col_params:

    # ── 1. Shapefile terrain ──────────────────────────────────────────────────
    st.subheader("1 · Shapefile")
    zip_file = st.file_uploader(
        "Déposer le dossier .zip contenant le shapefile de la zone d'implantation (export de Géoperso)",
        type=["zip"],
        help="Le zip doit contenir les fichiers .shp, .dbf, .prj et .shx du terrain d'implantation."
    )

    # ── 2. Informations projet ────────────────────────────────────────────────
    st.subheader("2 · Informations projet")
    nom_projet = st.text_input(
        "Nom du projet",
        placeholder="Futur lauréat CRE",
        help="Sera repris dans le titre de la carte et dans le nom du fichier."
    )
    urbanisme = st.text_area(
        "Document d'urbanisme applicable. Bien rensigner les informations pour l'ensemble des parcelles du terrain d'implantation",
        placeholder="L'urbanisme c'est super",
        height=110,
        help="Texte libre affiché dans l'encart en haut à droite de la carte. Laisser vide si non renseigné."
    )

    # ── 3. Zones humides ──────────────────────────────────────────────────────
    st.subheader("3 · Zones humides")
    zh_presence = st.radio(
        "Présence de zones humides dans la zone d'implantation ?",
        options=["Non", "Oui"],
        horizontal=True,
        help="Si oui, la carte devra afficher les ZH et les éléments techniques du projet (requis par le CDC CRE)."
    )

    zh_file       = None
    elements_file = None

    if zh_presence == "Oui":
        st.info("ℹ️ Le CDC impose d'afficher les zones humides, les panneaux, les pistes et les locaux techniques sur la carte.")
        zh_file = st.file_uploader(
            "Couche zones humides (.zip shapefile, .kml ou .geojson) - fournie par le BE enviro",
            type=["zip", "kml", "geojson", "json"],
            help="Fichier fourni par le BE environnemental. Formats acceptés : shapefile zippé, KML, GeoJSON."
        )
        elements_file = st.file_uploader(
            "Couche éléments techniques — panneaux, pistes, locaux (.kml) - fournie en un KML par notre BE interne. Ne doit contenir QUE les trois éléments mentionnés.",
            type=["kml"],
            help="Fichier KML fourni par le BE technique, contenant panneaux (polygones), pistes (lignes) et locaux (points)."
        )

    # ── 4. Paramètres ─────────────────────────────────────────────────────────
    st.subheader("4 · Paramètres")
    if zh_presence == "Non":
        recul = st.slider(
            "Recul zone capteurs PV (m)",
            min_value=0, max_value=50, value=10, step=1,
            help="Buffer négatif appliqué au terrain d'implantation pour délimiter la zone des capteurs PV au sens du CdC de la CRE."
        )
    else:
        recul = 10  # non utilisé quand éléments techniques fournis
        st.caption("ℹ️ Le recul zone capteurs est calculé automatiquement depuis les éléments techniques (5 m).")
    fond_aerien = st.toggle(
        "Fond aérien IGN Géoportail",
        value=True,
        help="Désactiver si la connexion est lente ou si le fond ne se charge pas correctement."
    )

    st.divider()

    # ── Validation ────────────────────────────────────────────────────────────
    manquants = []
    if zip_file is None:           manquants.append("shapefile terrain")
    if not nom_projet.strip():     manquants.append("nom du projet")
    if zh_presence == "Oui":
        if zh_file is None:        manquants.append("couche zones humides")
        if elements_file is None:  manquants.append("couche éléments techniques")

    pret = len(manquants) == 0
    if not pret:
        st.info("En attente : {}".format(", ".join(manquants)))

    generer = st.button(
        "🗺️ Générer la carte",
        disabled=not pret,
        use_container_width=True,
        type="primary"
    )

# ── Zone principale ───────────────────────────────────────────────────────────
with col_result:

    if not generer:
        st.markdown(
            """
            <div style='text-align:center; padding: 80px 40px; color: #999;'>
                <div style='font-size:60px'>🧭</div>
                <p style='margin-top:16px; font-size:15px; line-height:1.8'>
                Renseignez les paramètres à gauche<br>
                et cliquez sur <strong>Générer la carte</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # ── Extraction shapefile terrain ──────────────────────────────
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(tmpdir)

                shp_path = None
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith(".shp"):
                            shp_path = os.path.join(root, f)
                            break
                    if shp_path:
                        break

                if shp_path is None:
                    st.error("❌ Aucun fichier .shp trouvé dans le zip.")
                    st.stop()

                # ── Sauvegarde couches optionnelles sur disque temporaire ──────
                zh_tmp_path = None
                if zh_file is not None:
                    ext_zh = os.path.splitext(zh_file.name)[1]
                    zh_tmp_path = os.path.join(tmpdir, "zh{}".format(ext_zh))
                    with open(zh_tmp_path, "wb") as f:
                        f.write(zh_file.read())

                elts_tmp_path = None
                if elements_file is not None:
                    elts_tmp_path = os.path.join(tmpdir, "elements.kml")
                    with open(elts_tmp_path, "wb") as f:
                        f.write(elements_file.read())

                # ── Génération ────────────────────────────────────────────────
                with st.spinner("Génération de la carte en cours…"):
                    png_bytes = generer_carte(
                        shp_path       = shp_path,
                        nom_projet     = nom_projet.strip(),
                        recul_capteurs = recul,
                        urbanisme      = urbanisme.strip(),
                        echelle        = 5000,
                        fond_aerien    = fond_aerien,
                        dpi            = 150,
                        zh_path        = zh_tmp_path,
                        elements_path  = elts_tmp_path,
                    )

                st.success("✅ Carte générée avec succès !")
                st.image(png_bytes, use_container_width=True)

                _slug     = re.sub(r"[^\w]+", "_", nom_projet.strip()).strip("_")
                file_name = "UNITe_CETI_PV_{}.png".format(_slug)

                st.download_button(
                    label            = "⬇️ Télécharger la carte (PNG)",
                    data             = png_bytes,
                    file_name        = file_name,
                    mime             = "image/png",
                    use_container_width=True,
                )

            except zipfile.BadZipFile:
                st.error("❌ Le fichier uploadé n'est pas un zip valide.")
            except Exception as e:
                st.error("❌ Erreur lors de la génération : {}".format(str(e)))
                st.exception(e)
