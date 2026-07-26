"""
app.py — Interface Streamlit — Générateur de carte CETI
UNITe PV — AO CRE PPE2 Neutre Période 5

Lancement local : streamlit run app.py
"""

import os, re, zipfile, tempfile, base64
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

# ── Logo UNITe — chargé une fois pour le header
_logo_b64 = None
try:
    _logo_path = os.path.join(os.path.dirname(__file__), "logo_unite.png")
    with open(_logo_path, "rb") as _f:
        _logo_b64 = base64.b64encode(_f.read()).decode()
except FileNotFoundError:
    pass

# Header : titre à gauche, logo à droite
_col_titre, _col_logo = st.columns([8, 1], vertical_alignment="center")
with _col_titre:
    st.title("🗺️ CETI - Générateur plan de situation")
    st.caption("AO CRE PPE2 Neutre · Compatible CdC Période 5 (juillet 2026) · UNITe")
    st.caption(
        "Le plan de situation est exigé en pièce n°2 du dossier de candidature, "
        "y compris pour les projets ne nécessitant pas de CETI (cas 4 et I du cas 2 bis) : "
        "cet outil s'applique donc à tous les projets PV au sol."
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
        help="Si oui, la carte devra afficher les ZH ainsi que les panneaux, pistes, locaux "
             "techniques, clôture et autres aménagements liés à l'installation (requis par le CdC CRE)."
    )

    zh_file = None
    if zh_presence == "Oui":
        zh_file = st.file_uploader(
            "Couche zones humides (.zip shapefile, .kml ou .geojson)",
            type=["zip", "kml", "geojson", "json"],
            help="Fichier fourni par le BE environnemental. Formats acceptés : shapefile zippé, KML, GeoJSON."
        )

    # ── 4. Zone d'implantation des capteurs ──────────────────────────────────
    st.subheader("4 · Zone d'implantation des capteurs")

    panneaux_file = None
    pistes_files  = []
    recul         = 10
    mode_capteurs = None

    if zh_presence == "Non":
        mode_capteurs = st.radio(
            "Mode de délimitation",
            options=["Recul automatique depuis le terrain", "Depuis le plan d'implantation (KML)"],
            help="Le recul applique un buffer négatif au shapefile terrain. Le mode KML trace la zone autour des clusters de panneaux fournis."
        )
        if mode_capteurs == "Recul automatique depuis le terrain":
            recul = st.slider(
                "Recul zone capteurs PV (m)",
                min_value=0, max_value=100, value=10, step=1,
                help="Buffer négatif appliqué au terrain d'implantation pour délimiter la zone des capteurs PV au sens du CdC de la CRE."
            )
        else:
            panneaux_file = st.file_uploader(
                "KML rangées de panneaux — lignes et polygones des tables solaires (.kml)",
                type=["kml"],
                help="Fichier KML contenant les rangées de panneaux (LineStrings courtes et/ou polygones)."
            )
            pistes_files = st.file_uploader(
                "KML pistes, postes, clôture et autres aménagements — optionnel",
                type=["kml"],
                accept_multiple_files=True,
                help="Fichier(s) KML contenant les pistes d'accès internes et externes, les postes "
                     "de transformation et locaux techniques, la clôture et tout autre aménagement "
                     "ou équipement lié à l'installation. Accepte plusieurs fichiers KML. Facultatif."
            )
    else:
        st.info(
            "ℹ️ Le CdC impose de faire apparaître les zones humides ainsi que les "
            "emplacements des panneaux, des pistes internes et externes, des locaux "
            "techniques, de la clôture et de tout autre aménagement lié à l'installation."
        )
        panneaux_file = st.file_uploader(
            "KML rangées de panneaux — lignes et polygones des tables solaires (.kml)",
            type=["kml"],
            help="Fichier KML contenant les rangées de panneaux (LineStrings courtes et/ou polygones). Obligatoire si zones humides."
        )
        pistes_files = st.file_uploader(
            "KML pistes, postes, clôture et autres aménagements — optionnel",
            type=["kml"],
            accept_multiple_files=True,
            help="Fichier(s) KML contenant les pistes d'accès internes et externes (LineStrings), "
                 "les postes de transformation et locaux techniques (points), la clôture et tout "
                 "autre aménagement ou équipement lié à l'installation. Accepte plusieurs fichiers KML."
        )

    # ── 5. Paramètres ────────────────────────────────────────────────────────
    st.subheader("5 · Paramètres")
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
    if zip_file is None:       manquants.append("shapefile terrain")
    if not nom_projet.strip(): manquants.append("nom du projet")
    if not urbanisme.strip():  manquants.append("document d'urbanisme applicable")
    if zh_presence == "Oui":
        if zh_file is None:       manquants.append("couche zones humides")
        if panneaux_file is None: manquants.append("KML rangées de panneaux")
    elif mode_capteurs == "Depuis le plan d'implantation (KML)":
        if panneaux_file is None: manquants.append("KML rangées de panneaux")

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
