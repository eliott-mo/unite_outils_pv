"""
Carte des photos de terrain — outil UNITe
==========================================
À partir de photos de terrain (déposées directement ou dans un ZIP), génère une
carte HTML autonome où chaque photo est positionnée avec sa direction de prise
de vue.

La position vient de l'EXIF ou, à défaut, du texte incrusté dans l'image lu par
OCR ; la direction vient de l'EXIF ou, à défaut, du cône bleu de la vignette
GPS Map Camera (voir lecture_photo.py pour les deux cascades).

Cette application ne produit que des directions BRUTES, telles que détectées.
Tout ajustement — calibration de la boussole du téléphone, correction d'une
photo isolée — se fait ensuite dans la carte HTML, en mode édition : un décalage
de boussole ne se juge qu'en voyant les cônes sur le fond satellite, en
vérifiant s'ils pointent vers les bons éléments du paysage.

Lancement en local :  streamlit run app.py
"""

import base64
import os
import tempfile
import warnings
import zipfile

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import formats_images
import ocr_position
from detection_cap import SEUIL_CONFIANCE
from lecture_exif import SEUIL_PRECISION_M
from lecture_photo import lire_photo
from generation_html import (CarteIllisible, completer_carte, construire_carte,
                             extraire_donnees, photos_nouvelles)
from apercu_boussole import boussole

EXTENSIONS_IMAGE = formats_images.EXTENSIONS_IMAGE

# Types acceptés par le dépôt : les photos une par une ET l'archive ZIP.
TYPES_ACCEPTES = [extension.lstrip(".") for extension in EXTENSIONS_IMAGE] + ["zip"]

# Compromis résolution / poids du fichier final, mesuré sur photos 12 Mpx.
PRESETS_QUALITE = {
    "Léger — envoi par mail (~0,25 Mo/photo)":            (1024, 72),
    "Standard — recommandé (~0,4 Mo/photo)":              (1280, 75),
    "Haute qualité — zoom sur détails (~0,75 Mo/photo)":  (1600, 80),
}

st.set_page_config(
    page_title="Visite de site - Rapport photos",
    page_icon="📸",
    layout="wide",
)


# --------------------------------------------------------------------------
# Lecture des directions
# --------------------------------------------------------------------------

def position_peu_fiable(photo):
    """Vrai si l'appareil a lui-même annoncé une incertitude au-delà du seuil.

    Une précision inconnue (champ EXIF absent, cas le plus courant) n'est PAS
    considérée comme peu fiable : la plupart des applications n'écrivent pas ce
    champ, l'absence n'est donc pas un défaut.
    """
    return (photo["precision_m"] is not None
            and photo["precision_m"] > SEUIL_PRECISION_M)


def texte_precision(precision_m):
    """Met en forme l'incertitude GPS pour le tableau de vérification."""
    if precision_m is None:
        return "—"
    if precision_m > SEUIL_PRECISION_M:
        return f"⚠️ ±{precision_m:.0f} m — à vérifier"
    return f"±{precision_m:.0f} m"


def vers_rose(cap):
    """Traduit un cap en point cardinal lisible (N, NNE, NE...)."""
    if cap is None:
        return "—"
    noms = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return f"{noms[round(cap / 22.5) % 16]} ({cap:.0f}°)"


# --------------------------------------------------------------------------
# Traitement
# --------------------------------------------------------------------------

def _nom_libre(dossier, nom):
    """Chemin de destination dans `dossier`, suffixé si le nom est déjà pris."""
    destination = os.path.join(dossier, nom)
    base, extension = os.path.splitext(destination)
    compteur = 1
    while os.path.exists(destination):
        destination = f"{base}_{compteur}{extension}"
        compteur += 1
    return destination


def extraire_zip(fichier_zip, dossier):
    """Extrait les images du ZIP à plat. Retourne la liste des chemins extraits.

    Les sous-dossiers sont acceptés, mais les fichiers systèmes ajoutés par
    macOS (__MACOSX, fichiers commençant par un point) sont ignorés.
    """
    chemins = []
    with zipfile.ZipFile(fichier_zip) as archive:
        for membre in archive.namelist():
            nom = os.path.basename(membre)
            if not nom or not formats_images.est_image(nom):
                continue
            if membre.startswith("__MACOSX") or nom.startswith("."):
                continue
            destination = _nom_libre(dossier, nom)
            with archive.open(membre) as source, open(destination, "wb") as cible:
                cible.write(source.read())
            chemins.append(destination)
    return chemins


def preparer_fichiers(fichiers, dossier):
    """Écrit sur disque les fichiers déposés. Retourne la liste des chemins d'images.

    Les deux modes de dépôt cohabitent : photos déposées une à une (JPEG, PNG,
    WEBP, TIFF, HEIC) et archives ZIP, éventuellement mélangées.
    """
    chemins = []
    for fichier in fichiers:
        if fichier.name.lower().endswith(".zip"):
            chemins += extraire_zip(fichier, dossier)
        elif formats_images.est_image(fichier.name):
            destination = _nom_libre(dossier, os.path.basename(fichier.name))
            with open(destination, "wb") as cible:
                cible.write(fichier.getbuffer())
            chemins.append(destination)
    return sorted(chemins)


def analyser(chemins, barre_progression):
    """Lit la position et le cap de chaque photo (voir lecture_photo.lire_photo).

    Retourne (liste des photos exploitables, liste des photos écartées).
    Le cap obtenu est stocké dans « cap_brut » et n'est plus jamais modifié :
    calibration et corrections se font dans la carte HTML, par-dessus.
    """
    exploitables, ecartees = [], []

    for numero, chemin in enumerate(chemins, start=1):
        barre_progression.progress(
            numero / len(chemins),
            text=f"Analyse {numero}/{len(chemins)} — {os.path.basename(chemin)}"
        )

        lecture = lire_photo(chemin)
        if lecture["lat"] is None:
            ecartees.append((os.path.basename(chemin), lecture["message"]))
            continue

        exploitables.append({
            "chemin": chemin,
            "nom": os.path.basename(chemin),
            "lat": lecture["lat"],
            "lon": lecture["lon"],
            "source_position": lecture["source_position"],
            "precision_m": lecture["precision_m"],
            "cap_brut": lecture["cap"],
            "confiance": lecture["confiance"],
            "source_cap": lecture["source_cap"],
            "date": lecture["date"],
            "date_texte": lecture["date"].strftime("%d/%m/%Y %H:%M") if lecture["date"] else "",
        })

    # Tri chronologique : l'ordre de parcours du terrain est le plus parlant.
    exploitables.sort(key=lambda p: (p["date"] is None, p["date"], p["nom"]))
    return exploitables, ecartees


def compter_images(fichier):
    """Nombre d'images qu'apporte un fichier déposé (un ZIP en contient plusieurs).

    Le contenu du ZIP est seulement listé, jamais extrait : le compteur reste
    instantané même sur une grosse archive.
    """
    if not fichier.name.lower().endswith(".zip"):
        return 1 if formats_images.est_image(fichier.name) else 0
    try:
        with zipfile.ZipFile(fichier) as archive:
            return sum(1 for membre in archive.namelist()
                       if formats_images.est_image(os.path.basename(membre))
                       and not membre.startswith("__MACOSX")
                       and not os.path.basename(membre).startswith("."))
    except zipfile.BadZipFile:
        return 0
    finally:
        fichier.seek(0)          # le fichier sera relu lors de l'extraction


def dossier_de_travail():
    """Dossier temporaire du lot, créé une seule fois pour toute la session.

    Les photos y sont écrites une fois pour toutes : on n'en garde que le chemin.
    Les octets d'origine ne restent donc pas en mémoire, et l'image de la carte
    n'est encodée qu'au moment de la génération, quand la qualité est connue.
    """
    if "dossier" not in st.session_state:
        st.session_state["dossier"] = tempfile.mkdtemp(prefix="photos_carte_")
    return st.session_state["dossier"]


def reporter_saisies(anciennes_photos, nouvelles_photos, saisies):
    """Reporte des saisies repérées par rang sur une liste de photos réordonnée.

    Corrections et commentaires sont indexés par le rang de la photo dans le
    tableau. Après un ajout, la liste est refusionnée puis retriée par date et
    les rangs changent : sans ce report, une saisie glisserait d'une photo à une
    autre. Le chemin du fichier sert de repère stable.
    """
    rangs = {photo["chemin"]: rang for rang, photo in enumerate(nouvelles_photos)}
    reportees = {}
    for ancien_rang, valeur in saisies.items():
        if ancien_rang < len(anciennes_photos):
            chemin = anciennes_photos[ancien_rang]["chemin"]
            if chemin in rangs:
                reportees[rangs[chemin]] = valeur
    return reportees


def traiter(fichiers_a_traiter, remplacer):
    """Traite les fichiers en attente et met à jour l'état de la session.

    remplacer=False : les résultats s'ajoutent à ceux déjà obtenus, sans jamais
    relancer l'analyse des photos déjà traitées.
    remplacer=True  : on repart de zéro, seules les photos en attente comptent.
    """
    dossier = dossier_de_travail()
    with st.spinner("Préparation des fichiers…"):
        chemins = preparer_fichiers(fichiers_a_traiter, dossier)

    if not chemins:
        st.error("Aucune image exploitable dans ce dépôt "
                 "(formats acceptés : JPEG, PNG, WEBP, TIFF, HEIC).")
        return

    barre = st.progress(0.0, text="Analyse…")
    nouvelles, ecartees = analyser(chemins, barre)
    barre.empty()

    anciennes = [] if remplacer else st.session_state["photos"]
    photos = anciennes + nouvelles
    # Tri chronologique de l'ensemble : l'ordre de parcours du terrain reste
    # lisible même quand un lot est complété plus tard.
    photos.sort(key=lambda p: (p["date"] is None, p["date"], p["nom"]))

    if remplacer:
        # Repartir sans aucune saisie du lot précédent.
        st.session_state["ecartees"] = ecartees
        st.session_state["commentaires"] = {}
    else:
        st.session_state["ecartees"] = st.session_state["ecartees"] + ecartees
        st.session_state["commentaires"] = reporter_saisies(
            anciennes, photos, st.session_state["commentaires"])

    # Vider le déposoir, quel que soit le mode : ce qui vient d'être traité est
    # désormais dans le lot, le garder affiché mélangerait « déjà traité » et
    # « à traiter ». Un st.file_uploader ne se vide pas en effaçant son entrée de
    # session_state : il faut lui donner une nouvelle key pour que Streamlit le
    # recrée à neuf.
    st.session_state["version_deposoir"] += 1

    st.session_state["photos"] = photos
    st.rerun()          # repart sur un affichage propre (compteur remis à zéro)


@st.cache_data(show_spinner=False)
def apercu(chemin, largeur_max=1200):
    """Version affichable d'une photo, quel que soit son format d'origine.

    Un navigateur ne sait pas afficher un HEIC : on passe donc par Pillow, qui
    le décode (pillow-heif) et laisse Streamlit le servir dans un format
    universel. Le redimensionnement évite de transmettre 12 Mpx à chaque
    interaction avec l'interface.
    """
    with Image.open(chemin) as brut:
        image = ImageOps.exif_transpose(brut).convert("RGB")
    image.thumbnail((largeur_max, largeur_max), Image.LANCZOS)
    return image


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

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
    st.title("📸 Visite de site - Rapport photos")
    st.caption(
        "Transforme un lot de photos de terrain en carte interactive : "
        "position, direction de prise de vue et photo agrandissable."
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

with st.expander("ℹ️ Mode d'emploi et limites", expanded=False):
    st.markdown(f"""
**Comment faire**
1. Sur le terrain, photographier avec une application qui géolocalise
   (**GPS Map Camera** de préférence : la vignette carte en bas à gauche, avec le
   cône bleu, donne aussi la direction).
2. Récupérer les photos sur l'ordinateur.
3. Les déposer ci-dessous — **directement** (sélection multiple) ou dans un
   **.zip**, au choix —, vérifier le tableau, puis télécharger la carte.
4. Ouvrir la carte et cliquer sur ✏️ pour **ajuster les directions** : c'est là,
   sur le fond satellite, qu'un décalage de boussole se voit et se corrige.
5. Plus tard, pour **ajouter des photos à une carte déjà annotée** : choisir
   *Compléter une carte existante* tout en haut, déposer la carte `.html`, puis
   les nouvelles photos.

**Compléter une carte existante**
Déposer une carte `.html` déjà produite ajoute les nouvelles photos **sans rien
perdre** de ce qui y a été fait dans le navigateur : commentaires, noms, ordre,
corbeille, calibration de la boussole et directions figées à la main. Les photos
déjà présentes ne sont ni retraitées ni dupliquées, et la calibration de la carte
s'applique aussi aux directions des arrivantes. Le **titre** est pré-rempli avec
celui de la carte importée et reste modifiable, pour dater la nouvelle version.
Une photo **renommée** dans l'éditeur a perdu son nom de fichier : si vous la
redéposez, elle sera ajoutée une seconde fois.

**Formats acceptés**
JPEG, PNG, WEBP, TIFF et **HEIC** (format par défaut des iPhone), ainsi que les
ZIP contenant ces images.

**Comment la position est déterminée** (dans cet ordre)
1. Les **coordonnées GPS des métadonnées EXIF**, quand l'application les écrit.
2. Sinon, le **texte incrusté dans l'image** (bandeau « Lat … Long … »), lu par
   OCR : c'est le cas de certaines applications iPhone qui ne renseignent pas
   l'EXIF.
3. Sinon, la photo est écartée et signalée.
Une position hors de France métropolitaine est refusée par sécurité : mieux vaut
écarter une photo que la placer à un endroit faux.

**Comment la direction est déterminée** (dans cet ordre)
1. Le **cap GPS des métadonnées** (iPhone, Open Camera, Solocator…).
2. Sinon, le **cône bleu de la vignette Google** incrustée par GPS Map Camera.
3. Sinon, aucune direction : la photo est quand même placée, sans cône.

**Limites à connaître**
- La précision est celle de la boussole du téléphone : **± 10 à 20°**, et davantage
  si le téléphone était mal calibré. Les directions affichées ici sont **brutes** :
  elles se corrigent dans la carte, en mode édition.
- Quand l'application photo le renseigne, l'**incertitude GPS annoncée par
  l'appareil** est reprise dans la colonne *Précision*. Au-delà de
  {SEUIL_PRECISION_M} m, la photo est signalée : la fixation GPS était dégradée et
  l'emplacement peut être faux de plusieurs centaines de mètres. La photo reste
  placée — c'est un avertissement, pas une exclusion. Une colonne vide signifie
  simplement que l'application n'écrit pas cette information.
- Un indice de confiance inférieur à {SEUIL_CONFIANCE:.2f} signale une détection
  douteuse à vérifier à la main.
- La carte finale a besoin d'une connexion Internet pour afficher le fond
  satellite, mais les photos, elles, sont intégrées dans le fichier.

**Éviter le problème à la source**
Faire calibrer la boussole avant la visite : ouvrir Google Maps sur le téléphone,
toucher le point bleu, choisir *Étalonner la boussole* et dessiner un 8 en l'air.
Un téléphone dans une coque aimantée ou posé sur un tableau de bord dérive
systématiquement.
""")

# Avertissements de configuration : ils n'empêchent pas de travailler, les photos
# géolocalisées par EXIF restent traitables.
if not formats_images.HEIC_DISPONIBLE:
    st.warning(formats_images.MESSAGE_HEIC)
if not ocr_position.tesseract_disponible():
    st.warning(ocr_position.AIDE_INSTALLATION)

# État conservé d'une interaction à l'autre. Les photos déjà traitées y restent :
# aucun dépôt ultérieur ne peut provoquer leur réanalyse.
for cle, valeur_initiale in [("photos", []), ("ecartees", []),
                             ("commentaires", {}), ("version_deposoir", 0)]:
    st.session_state.setdefault(cle, valeur_initiale)

# Première question posée, avant même les photos : celui qui vient compléter une
# carte doit voir la fonctionnalité tout de suite, pas la découvrir en bas de
# page une fois ses photos traitées.
mode = st.radio(
    "Que voulez-vous faire ?",
    ["🆕 Nouvelle carte", "🔄 Compléter une carte existante"],
    horizontal=True,
    help="« Compléter » ajoute vos photos à une carte déjà produite par cet "
         "outil, en conservant tout ce qui y a été édité dans le navigateur.",
)
complement = mode.endswith("existante")

html_existant, donnees_existantes = None, None
if complement:
    carte_existante = st.file_uploader(
        "Carte à compléter (.html)",
        type=["html"],
        key="carte_existante",
        help="La carte HTML déjà produite par cet outil. Tout ce qui y a été "
             "édité — commentaires, noms, corbeille, ordre, calibration, "
             "directions figées — est conservé, et les photos déjà présentes "
             "ne sont ni retraitées ni dupliquées.",
    )
    if carte_existante is not None:
        html_existant = carte_existante.getvalue().decode("utf-8", errors="replace")
        try:
            # Lecture immédiate : mieux vaut signaler un fichier inexploitable
            # maintenant qu'après avoir traité puis encodé toutes les photos.
            with warnings.catch_warnings(record=True) as alertes:
                warnings.simplefilter("always")
                donnees_existantes = extraire_donnees(html_existant)
            for alerte in alertes:
                st.warning(str(alerte.message))
            st.success(
                f"Carte « {donnees_existantes['titre']} » lue — "
                f"{len(donnees_existantes['points'])} photo(s) déjà présente(s). "
                "Déposez maintenant les photos à ajouter."
            )
        except CarteIllisible as erreur:
            st.error(str(erreur))
            html_existant = None
    else:
        st.info("Déposez la carte à compléter, puis les photos à y ajouter.")

# La version fait partie de la key : l'incrémenter après un traitement recrée un
# déposoir vide (voir traiter()). Le déposoir ne montre donc jamais que ce qui
# reste à traiter.
fichiers = st.file_uploader(
    "Photos ou fichier ZIP",
    key=f"deposoir_{st.session_state['version_deposoir']}",
    type=TYPES_ACCEPTES,
    accept_multiple_files=True,
    help="Déposez les photos directement (sélection multiple) ou un ZIP qui les "
         "contient — les deux peuvent être mélangés. Formats : JPEG, PNG, WEBP, "
         "TIFF, HEIC. Les sous-dossiers d'un ZIP sont parcourus. "
         "Rien n'est traité tant que vous n'avez pas cliqué sur le bouton.",
) or []

photos = st.session_state["photos"]
ecartees = st.session_state["ecartees"]

# Le déposoir est vidé après chaque traitement : ce qu'il contient a donc
# forcément été déposé depuis, et reste entièrement à traiter. Aucun suivi de
# « déjà traité » n'est nécessaire — les résultats acquis vivent dans
# session_state["photos"] et ne sont jamais réanalysés.
en_attente = fichiers

if en_attente:
    nombre = sum(compter_images(f) for f in en_attente)
    st.info(f"**{nombre} photo(s) prête(s) à traiter.** "
            "Vous pouvez encore en déposer avant de lancer le traitement.")

    if not photos:
        if st.button("▶️ Traiter les photos", type="primary", use_container_width=True):
            traiter(en_attente, remplacer=False)
    else:
        # Un lot est déjà traité : à l'utilisateur de dire quoi faire des nouvelles.
        st.write("**Que faire de ces nouvelles photos ?**")
        gauche, droite = st.columns(2)
        if gauche.button(f"➕ Ajouter au lot ({len(photos)} photo(s) déjà traitée(s))",
                         type="primary", use_container_width=True,
                         help="Les photos déjà traitées sont conservées telles "
                              "quelles : elles ne sont pas réanalysées."):
            traiter(en_attente, remplacer=False)
        if droite.button("♻️ Remplacer le lot", use_container_width=True,
                         help="Le lot précédent est oublié, commentaires compris. "
                              "Seules les nouvelles photos sont traitées."):
            traiter(en_attente, remplacer=True)

if not photos:
    # Des écartées sans aucune photo placée : un traitement a bien eu lieu, mais
    # rien n'en est sorti. Le déposoir étant vidé entre-temps, c'est ce constat —
    # et non son contenu — qui dit à l'utilisateur ce qui s'est passé.
    if ecartees:
        st.error("Aucune photo géolocalisée dans ce dépôt.")
        st.dataframe(pd.DataFrame(ecartees, columns=["Fichier", "Motif"]),
                     hide_index=True, use_container_width=True)
    elif not fichiers:
        st.info("Déposez vos photos ou un fichier ZIP pour commencer.")
    st.stop()

# --- Synthèse ---
nb_fiables = sum(1 for p in photos
                 if p["confiance"] and p["confiance"] >= SEUIL_CONFIANCE)
nb_sans_cap = sum(1 for p in photos if p["cap_brut"] is None)

nb_position_ocr = sum(1 for p in photos if p["source_position"] == "OCR")

colonnes = st.columns(4)
colonnes[0].metric("Photos placées", len(photos))
colonnes[1].metric("Direction fiable", nb_fiables)
colonnes[2].metric("À vérifier", len(photos) - nb_fiables - nb_sans_cap)
colonnes[3].metric("Sans direction", nb_sans_cap)

if nb_position_ocr:
    st.caption(
        f"{nb_position_ocr} position(s) lue(s) dans le texte incrusté de l'image "
        "(pas de GPS dans les métadonnées) — colonne « Source pos. » du tableau."
    )

# Photos dont le téléphone a lui-même annoncé une fixation GPS dégradée. Elles
# restent sur la carte : c'est un signalement, pas une exclusion.
peu_fiables = [p for p in photos if position_peu_fiable(p)]
if peu_fiables:
    detail = ", ".join(f"{p['nom']} (±{p['precision_m']:.0f} m)" for p in peu_fiables)
    st.warning(
        f"⚠️ {len(peu_fiables)} photo(s) à position peu fiable : {detail}.\n\n"
        f"L'appareil a annoncé une incertitude supérieure à {SEUIL_PRECISION_M} m "
        "au moment du déclenchement (réception GPS dégradée). Ces photos restent "
        "placées sur la carte, mais leur emplacement est à vérifier avant diffusion."
    )

if ecartees:
    with st.expander(f"⚠️ {len(ecartees)} photo(s) écartée(s)"):
        st.dataframe(pd.DataFrame(ecartees, columns=["Fichier", "Motif"]),
                     hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------
# Vérification photo par photo
# --------------------------------------------------------------------------

st.subheader("Vérification des directions")
st.caption(
    "Ce tableau est en lecture seule, hormis les commentaires : les directions "
    "affichées sont celles détectées, sans retouche. Elles se calibrent et se "
    "corrigent dans la carte, en mode édition — c'est là, sur le fond satellite, "
    "qu'un décalage se juge."
)

commentaires = st.session_state["commentaires"]

tableau = pd.DataFrame([{
    "N°": index + 1,
    "Fichier": photo["nom"],
    "Date": photo["date_texte"],
    "Position": f"{photo['lat']:.6f}, {photo['lon']:.6f}",
    "Source pos.": photo["source_position"],
    "Précision (m)": texte_precision(photo["precision_m"]),
    "Direction": photo["cap_brut"],
    "Confiance": photo["confiance"],
    "Source cap": photo["source_cap"],
    "Commentaire": commentaires.get(index, ""),
} for index, photo in enumerate(photos)])

tableau_corrige = st.data_editor(
    tableau,
    hide_index=True,
    use_container_width=True,
    disabled=["N°", "Fichier", "Date", "Position", "Source pos.", "Précision (m)",
              "Direction", "Confiance", "Source cap"],
    column_config={
        "Position": st.column_config.TextColumn(
            "Position", help="Latitude, longitude en degrés décimaux."),
        "Source pos.": st.column_config.TextColumn(
            "Source pos.", width="small",
            help="EXIF = métadonnées de la photo ; OCR = texte incrusté dans l'image."),
        "Précision (m)": st.column_config.TextColumn(
            "Précision (m)", width="small",
            help="Incertitude annoncée par l'appareil (GPSHPositioningError). "
                 "« — » = non renseignée par l'application photo, ce qui est le cas "
                 f"le plus fréquent. Au-delà de {SEUIL_PRECISION_M} m, la position "
                 "est signalée mais la photo reste placée."),
        "Direction": st.column_config.NumberColumn(
            "Direction (°)", format="%.0f",
            help="Direction détectée, portée sur la carte. Ajustable ensuite "
                 "dans l'éditeur HTML."),
        "Confiance": st.column_config.ProgressColumn(
            "Confiance", min_value=0, max_value=1, format="%.2f"),
        "Source cap": st.column_config.TextColumn(
            "Source cap", width="small",
            help="EXIF = cap des métadonnées ; Vignette = cône bleu GPS Map Camera ; "
                 "— = aucune direction disponible."),
        "Commentaire": st.column_config.TextColumn("Commentaire", width="medium"),
    },
    key="editeur",
)

# Seuls les commentaires sont saisissables ici.
for index, ligne in tableau_corrige.iterrows():
    commentaires[index] = ligne["Commentaire"] or ""

# Directions portées sur la carte : les directions brutes, telles quelles.
# La carte reçoit une calibration neutre et aucune correction, à elle de les
# porter si le chargé de projet en pose.
for index, photo in enumerate(photos):
    photo["cap"] = photo["cap_brut"]
    photo["commentaire"] = commentaires.get(index, "")
    photo["cap_manuel"] = None

with st.expander("👁️ Vérifier visuellement une photo"):
    choix = st.selectbox(
        "Photo", range(len(photos)),
        format_func=lambda i: f"{i + 1}. {photos[i]['nom']}", key="photo_verif",
    )
    photo_verifiee = photos[choix]
    gauche, droite = st.columns([2, 1])
    gauche.image(apercu(photo_verifiee["chemin"]), use_container_width=True)

    # Sert à valider la DÉTECTION : le cône dessiné doit correspondre à celui de
    # la vignette incrustée. Les deux caps passés sont identiques (aucune
    # correction ici), l'aperçu ne montre donc qu'un seul cône.
    droite.image(boussole(photo_verifiee["cap_brut"], photo_verifiee["cap"], taille=210))
    droite.write(f"**Détecté :** {vers_rose(photo_verifiee['cap_brut'])}")
    droite.write(f"**Confiance :** {photo_verifiee['confiance']}")
    droite.write(f"**Source du cap :** {photo_verifiee['source_cap']}")
    droite.write(f"**Position :** {photo_verifiee['lat']:.6f}, {photo_verifiee['lon']:.6f} "
                 f"({photo_verifiee['source_position']})")
    droite.write(f"**Précision :** {texte_precision(photo_verifiee['precision_m'])}")
    droite.caption("Comparez avec le cône bleu de la vignette en bas à gauche.")

# --------------------------------------------------------------------------
# Génération
# --------------------------------------------------------------------------

st.subheader("Génération de la carte")

# La carte à compléter, elle, a été demandée tout en haut : ce qui reste à
# régler ici, c'est le titre et la qualité des photos ajoutées.
if donnees_existantes is not None:
    deja = len(donnees_existantes["points"])
    a_ajouter = photos_nouvelles(donnees_existantes, photos)
    st.info(
        f"**Carte « {donnees_existantes['titre']} » — {deja} photo(s) déjà "
        f"présente(s).** {len(a_ajouter)} photo(s) du lot seront ajoutées"
        + (f", {len(photos) - len(a_ajouter)} déjà présente(s) seront ignorée(s)."
           if len(a_ajouter) < len(photos) else ".")
    )

col_gauche, col_droite = st.columns(2)
if donnees_existantes is not None:
    # Titre pré-rempli avec celui de la carte importée, mais modifiable : c'est
    # l'occasion de dater la nouvelle version. La calibration, elle, reste celle
    # de la carte — elle ne se règle que dans l'éditeur, sur le fond satellite.
    titre = col_gauche.text_input(
        "Titre de la carte", value=donnees_existantes["titre"],
        help="Pré-rempli avec le titre de la carte importée. Modifiez-le pour "
             "dater cette nouvelle version, par exemple.",
    )
    col_gauche.caption("La calibration de la carte importée est conservée.")
else:
    titre = col_gauche.text_input("Titre de la carte", value="Visite de site")
preset = col_droite.selectbox("Qualité des photos", list(PRESETS_QUALITE), index=1)
largeur_max, qualite = PRESETS_QUALITE[preset]

# En complément, seules les photos ajoutées sont encodées ici : le poids de la
# carte importée s'ajoute tel quel, la qualité choisie ne s'applique qu'aux
# arrivantes.
nb_encodees = len(a_ajouter) if donnees_existantes is not None else len(photos)
poids_estime = nb_encodees * {1024: 0.25, 1280: 0.40, 1600: 0.75}[largeur_max]
if donnees_existantes is not None:
    st.caption(f"Poids ajouté au fichier importé : environ **{poids_estime:.1f} Mo** "
               f"({len(html_existant.encode('utf-8')) / 1e6:.1f} Mo actuellement).")
else:
    st.caption(f"Poids estimé du fichier : environ **{poids_estime:.1f} Mo**.")

bouton = "🗺️ Compléter la carte" if html_existant else "🗺️ Générer la carte"
if st.button(bouton, type="primary", use_container_width=True):
    with st.spinner("Génération en cours…"):
        if html_existant:
            html = completer_carte(html_existant, photos, largeur_max, qualite, titre)
        else:
            html = construire_carte(photos, titre, largeur_max, qualite)

    st.success("Carte complétée." if html_existant else "Carte générée.")
    nom_fichier = "".join(c if c.isalnum() or c in " -_" else "_" for c in titre).strip()
    st.download_button(
        "⬇️ Télécharger la carte HTML",
        data=html.encode("utf-8"),
        file_name=f"{nom_fichier or 'carte'}.html",
        mime="text/html",
        use_container_width=True,
    )
    st.caption(
        "Ouvrez le fichier avec un navigateur (double-clic). "
        "Il peut être envoyé par mail : les photos sont intégrées dedans. "
        "Le bouton ✏️ de la carte permet de la modifier directement dans le "
        "navigateur (commentaires, noms, ordre, titre, masquage) puis de "
        "l'enregistrer sous forme d'un nouveau fichier."
    )
