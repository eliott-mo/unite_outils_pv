"""
app.py — Interface Streamlit — Générateur de carte CETI
UNITe PV — AO CRE Sol Période 9

Lancement local : streamlit run app.py
"""

import os
import re
import zipfile
import tempfile
import streamlit as st
from ceti_generate_map import generer_carte

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Générateur de carte CETI — UNITe PV",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ Générateur de carte de situation CETI")
st.caption("AO CRE PV Sol · Période 9 · UNITe")
st.divider()

# ── Mise en page : sidebar gauche + zone principale droite ────────────────────
col_params, col_result = st.columns([1, 2], gap="large")

with col_params:

    # 1. Upload shapefile
    st.subheader("1 · Shapefile")
    zip_file = st.file_uploader(
        "Déposer le dossier .zip contenant le shapefile de la zone d'implantation (export de Géoperso)",
        type=["zip"],
        help="Le zip doit contenir les fichiers .shp, .dbf, .prj et .shx du terrain d'implantation."
    )

    st.subheader("2 · Informations projet")
    nom_projet = st.text_input(
        "Nom du projet",
        placeholder="Exemple : Amillis",
        help="Sera repris dans le titre de la carte et dans le nom du fichier."
    )
    urbanisme = st.text_area(
        "Document d'urbanisme applicable",
        placeholder="Exemple :\nPLU d'Amillis \nZone : Ns (naturelle spéciale)",
        height=110,
        help="Texte libre affiché dans l'encart en haut à droite de la carte. Laisser vide si non renseigné."
    )

    st.subheader("3 · Paramètres")
    recul = st.slider(
        "Recul zone capteurs PV (m)",
        min_value=0, max_value=50, value=10, step=1,
        help="Buffer négatif appliqué au terrain d'implantation pour délimiter la zone des capteurs PV au sens du CdC de la CRE."
    )
    fond_aerien = st.toggle(
        "Fond aérien IGN Géoportail",
        value=True,
        help="Désactiver si la connexion est lente ou si le fond ne se charge pas correctement."
    )

    st.divider()

    # Validation des champs requis
    pret = zip_file is not None and nom_projet.strip() != ""
    if not pret:
        manquants = []
        if zip_file is None:    manquants.append("fichier shapefile")
        if not nom_projet.strip(): manquants.append("nom du projet")
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
                Renseignez les paramètres à gauche<br>et cliquez sur <strong>Générer la carte</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        # Extraction du zip dans un dossier temporaire
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Décompression
                with zipfile.ZipFile(zip_file, "r") as zf:
                    zf.extractall(tmpdir)

                # Recherche du .shp (à la racine ou dans un sous-dossier)
                shp_path = None
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith(".shp"):
                            shp_path = os.path.join(root, f)
                            break
                    if shp_path:
                        break

                if shp_path is None:
                    st.error("❌ Aucun fichier .shp trouvé dans le zip. Vérifiez le contenu de votre archive.")
                    st.stop()

                # Génération de la carte
                with st.spinner("Génération de la carte en cours…"):
                    png_bytes = generer_carte(
                        shp_path       = shp_path,
                        nom_projet     = nom_projet.strip(),
                        recul_capteurs = recul,
                        urbanisme      = urbanisme.strip(),
                        echelle        = 5000,
                        fond_aerien    = fond_aerien,
                        dpi            = 150,
                    )

                st.success("✅ Carte générée avec succès !")

                # Aperçu
                st.image(png_bytes, use_container_width=True)

                # Bouton de téléchargement
                _slug     = re.sub(r"[^\w]+", "_", nom_projet.strip()).strip("_")
                file_name = "UNITe_CETI_PV_{}.png".format(_slug)

                st.download_button(
                    label     = "⬇️ Télécharger la carte (PNG)",
                    data      = png_bytes,
                    file_name = file_name,
                    mime      = "image/png",
                    use_container_width=True,
                )

            except zipfile.BadZipFile:
                st.error("❌ Le fichier uploadé n'est pas un zip valide.")
            except Exception as e:
                st.error("❌ Erreur lors de la génération : {}".format(str(e)))
                st.exception(e)
