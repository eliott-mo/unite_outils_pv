"""
app.py — Interface Streamlit — Générateur de carte CETI
UNITe PV — AO CRE Sol Période 9

Lancement local : streamlit run app.py
"""

import os, re, zipfile, tempfile
import streamlit as st
from ceti_generate_map import generer_carte


def _generer_carte_cached(
    zip_bytes, nom_projet, recul_capteurs, urbanisme,
    echelle, fond_aerien, dpi, fmt,
    zh_bytes=None, zh_ext=".zip", panneaux_bytes=None, pistes_bytes_list=None,
):
    """
    Wrapper autour de generer_carte().
    Les arguments sont des bytes (contenu des fichiers), pas des chemins :
    Streamlit hache le contenu pour la clé de cache → même fichier = cache hit.
    TTL implicite via le cycle de vie de la session Streamlit.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Terrain shapefile
        zip_path = os.path.join(tmpdir, "terrain.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        shp = None
        for root, _, files in os.walk(tmpdir):
            for fn in files:
                if fn.endswith(".shp"):
                    shp = os.path.join(root, fn)
                    break
            if shp:
                break

        # Couches optionnelles
        zh_path = None
        if zh_bytes and zh_ext:
            zh_path = os.path.join(tmpdir, "zh{}".format(zh_ext))
            with open(zh_path, "wb") as f:
                f.write(zh_bytes)

        panneaux_path = None
        if panneaux_bytes:
            panneaux_path = os.path.join(tmpdir, "panneaux.kml")
            with open(panneaux_path, "wb") as f:
                f.write(panneaux_bytes)

        pistes_paths = None
        if pistes_bytes_list:
            pistes_paths = []
            for i, pb in enumerate(pistes_bytes_list):
                p = os.path.join(tmpdir, "pistes_{}.kml".format(i))
                with open(p, "wb") as f:
                    f.write(pb)
                pistes_paths.append(p)

        return generer_carte(
            shp_path       = shp,
            nom_projet     = nom_projet,
            recul_capteurs = recul_capteurs,
            urbanisme      = urbanisme,
            echelle        = echelle,
            fond_aerien    = fond_aerien,
            dpi            = dpi,
            zh_path        = zh_path,
            kml_panneaux   = panneaux_path,
            kml_pistes     = pistes_paths,
            format         = fmt,
        )

st.set_page_config(
    page_title="CETI - Générateur plan de situation",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ CETI - Générateur plan de situation")
st.caption("AO CRE PV Sol · Compatible CdC Période 9 · UNITe")
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
        placeholder="Nom Commune (Département)",
        help="Sera repris dans le titre de la carte et dans le nom du fichier."
    )
    urbanisme = st.text_area(
        "Document d'urbanisme applicable",
        placeholder="RNU / PLU / PLUi / Carte Communale",
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

    zh_file        = None
    panneaux_file  = None
    pistes_files   = []

    if zh_presence == "Oui":
        st.info("ℹ️ Le CDC impose d'afficher les zones humides, les panneaux, les pistes et les locaux techniques sur la carte.")
        zh_file = st.file_uploader(
            "Couche zones humides (.zip shapefile, .kml ou .geojson)",
            type=["zip", "kml", "geojson", "json"],
            help="Fichier fourni par le BE environnemental. Formats acceptés : shapefile zippé, KML, GeoJSON."
        )
        panneaux_file = st.file_uploader(
            "KML rangées de panneaux — lignes et polygones des tables solaires (.kml)",
            type=["kml"],
            help="Fichier KML contenant uniquement les rangées de panneaux (LineStrings courtes et/ou polygones). Obligatoire si zones humides."
        )
        pistes_files = st.file_uploader(
            "KML pistes et postes — optionnel (1 ou 2 fichiers)",
            type=["kml"],
            accept_multiple_files=True,
            help="Fichier(s) KML contenant les pistes d'accès (LineStrings) et postes de transformation (points). Accepte 1 ou 2 fichiers KML. Facultatif : améliore la délimitation des clusters de panneaux."
        )

    st.subheader("4 · Paramètres")
    if zh_presence == "Non":
        recul = st.slider(
            "Recul zone capteurs PV (m)",
            min_value=0, max_value=100, value=10, step=1,
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
    format_export = st.radio(
        "Format d'export",
        options=["PDF", "PNG"],
        horizontal=True,
        help="PDF recommandé pour l'envoi à la DREAL."
    )

    st.divider()

    # ── Validation ────────────────────────────────────────────────────────────
    manquants = []
    if zip_file is None:           manquants.append("shapefile terrain")
    if not nom_projet.strip():     manquants.append("nom du projet")
    if not urbanisme.strip():      manquants.append("document d'urbanisme applicable")
    if zh_presence == "Oui":
        if zh_file is None:        manquants.append("couche zones humides")
        if panneaux_file is None:  manquants.append("KML rangées de panneaux")

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
        try:
            # ── Lecture des bytes (clés de cache, indépendant des chemins tmp) ─
            zip_bytes      = zip_file.read()
            zh_bytes       = zh_file.read()       if zh_file       else None
            panneaux_bytes = panneaux_file.read() if panneaux_file  else None
            pistes_blist   = [pf.read() for pf in (pistes_files or [])] or None

            fmt_lower = format_export.lower()

            # ── Génération (cachée) ────────────────────────────────────────────
            with st.spinner("Génération de la carte en cours… (chargement fond aérien si activé)"):
                carte_bytes = _generer_carte_cached(
                    zip_bytes      = zip_bytes,
                    nom_projet     = nom_projet.strip(),
                    recul_capteurs = recul,
                    urbanisme      = urbanisme.strip(),
                    echelle        = 5000,
                    fond_aerien    = fond_aerien,
                    dpi            = 150,
                    fmt            = fmt_lower,
                    zh_bytes       = zh_bytes,
                    zh_ext         = os.path.splitext(zh_file.name)[1] if zh_file else ".zip",
                    panneaux_bytes = panneaux_bytes,
                    pistes_bytes_list = pistes_blist,
                )

            st.success("✅ Carte générée avec succès !")

            # Aperçu toujours en PNG (le cache évite la 2e génération si déjà PNG)
            if fmt_lower == "pdf":
                with st.spinner("Aperçu PNG…"):
                    apercu_bytes = _generer_carte_cached(
                        zip_bytes      = zip_bytes,
                        nom_projet     = nom_projet.strip(),
                        recul_capteurs = recul,
                        urbanisme      = urbanisme.strip(),
                        echelle        = 5000,
                        fond_aerien    = fond_aerien,
                        dpi            = 100,
                        fmt            = "png",
                        zh_bytes       = zh_bytes,
                        zh_ext         = os.path.splitext(zh_file.name)[1] if zh_file else '.zip',
                        panneaux_bytes = panneaux_bytes,
                        pistes_bytes_list = pistes_blist,
                    )
                st.image(apercu_bytes, use_container_width=True)
            else:
                st.image(carte_bytes, use_container_width=True)

            _slug     = re.sub(r"[^\w]+", "_", nom_projet.strip()).strip("_")
            mime_type = "application/pdf" if fmt_lower == "pdf" else "image/png"
            st.download_button(
                label            = "⬇️ Télécharger la carte ({})".format(format_export),
                data             = carte_bytes,
                file_name        = "UNITe_CETI_PV_{}.{}".format(_slug, fmt_lower),
                mime             = mime_type,
                use_container_width=True,
            )

        except zipfile.BadZipFile:
            st.error("❌ Le fichier uploadé n'est pas un zip valide.")
        except Exception as e:
            st.error("❌ Erreur lors de la génération : {}".format(str(e)))
            st.exception(e)
