"""Tous les chemins de l'application, en un seul endroit.

L'application vit dans **un seul dossier** et n'écrit que dedans. Chaque sous-dossier
porte un nom qui dit ce qu'il contient, pour qu'on s'y retrouve sans lire le code :

    app_maelia/
      espaces_app/        les espaces de conception livrés (fichiers .json)
      donnees_app/        les données d'entrée fournies avec l'application
      terrains_app/       les terrains GAMA que l'application construit
      simulations_app/    les plans produits : dateDose, XML, DOE tiré
      results_app/        les résultats d'analyse exportés
      ui/ outils/ tests_app/

Deux choses restent à l'extérieur et ne peuvent pas en bouger : **l'installation de
MAELIA**, qui contient le modèle, et **GAMA** lui-même. Leurs emplacements varient d'une
machine à l'autre : ce sont donc des réglages, et jamais des chemins enfouis dans le
code.

Trois sources, dans cet ordre de priorité :

  1. ``reglages.json``, écrit par l'utilisateur depuis l'interface — il gagne toujours ;
  2. les variables d'environnement ``MAELIA_ROOT`` et ``GAMA_HEADLESS`` ;
  3. les emplacements d'installation habituels, essayés dans l'ordre.

Le réglage est donc relu à chaque appel, jamais figé au chargement du module : changer
le chemin depuis l'interface prend effet sans redémarrer l'application.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ── Le dossier de l'application ──────────────────────────────────────────────
APP = Path(__file__).resolve().parent

ESPACES = APP / "espaces_app"
DONNEES = APP / "donnees_app"
TERRAINS = APP / "terrains_app"
SIMULATIONS = APP / "simulations_app"
RESULTATS = APP / "results_app"
UI = APP / "ui"
OUTILS = APP / "outils"

CLIMATS_OBSERVEE = DONNEES / "climats_observee"

# ── Ce qui reste dehors ──────────────────────────────────────────────────────
REGLAGES = APP / "reglages.json"

MAELIA_CANDIDATS = (
    Path.home() / "Workspace_GAMA" / "MAELIA",
    Path.home() / "MAELIA",
    Path("/opt/MAELIA"),
)

GAMA_HEADLESS_CANDIDATS = (
    Path("/Applications/Gama.app/Contents/headless/gama-headless.sh"),
    Path.home() / "Applications/Gama.app/Contents/headless/gama-headless.sh",
    Path("/opt/gama/headless/gama-headless.sh"),
    Path("C:/Program Files/Gama/headless/gama-headless.bat"),
)


def lire_reglages() -> dict:
    """Réglages écrits par l'utilisateur, ou un dictionnaire vide."""
    if not REGLAGES.exists():
        return {}
    try:
        contenu = json.loads(REGLAGES.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Un fichier abîmé ne doit pas empêcher l'application de démarrer : on
        # retombe sur les autres sources, et l'interface le signalera.
        return {}
    return contenu if isinstance(contenu, dict) else {}


def ecrire_reglages(**valeurs: str | None) -> dict:
    """Enregistre les chemins choisis. Une valeur vide efface le réglage."""
    reglages = lire_reglages()
    for cle, valeur in valeurs.items():
        if valeur:
            reglages[cle] = str(Path(str(valeur)).expanduser())
        else:
            reglages.pop(cle, None)
    REGLAGES.write_text(json.dumps(reglages, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return reglages


def _choisir(cle: str, variable: str, candidats) -> tuple[Path | None, str]:
    """Premier chemin existant parmi réglage, variable d'environnement, candidats."""
    regle = lire_reglages().get(cle)
    if regle:
        chemin = Path(regle).expanduser()
        # Un réglage explicite est rendu même s'il n'existe pas : l'utilisateur doit
        # voir ce qu'il a demandé, et pourquoi cela ne marche pas.
        return chemin, "réglage de l'application"
    declare = os.environ.get(variable)
    if declare and Path(declare).expanduser().exists():
        return Path(declare).expanduser(), f"variable d'environnement {variable}"
    trouve = next((c for c in candidats if c.exists()), None)
    if trouve:
        return trouve, "emplacement d'installation habituel"
    return None, "introuvable"


def maelia_root() -> Path:
    """Racine de l'installation MAELIA, réglable ; premier candidat à défaut."""
    chemin, _ = _choisir("maelia_root", "MAELIA_ROOT", MAELIA_CANDIDATS)
    return chemin or MAELIA_CANDIDATS[0]


def launcher() -> Path:
    return maelia_root() / "models" / "main" / "launcherBase.gaml"


def sorties_gama() -> Path:
    return maelia_root() / "models" / "main" / "log"


def terrain_source() -> Path:
    """Terrain de référence livré avec MAELIA, dont l'application dérive les siens."""
    return maelia_root() / "includes" / "terraintest"


def gama_headless() -> Path | None:
    """Localise gama-headless, ou None si rien n'est réglé ni trouvé."""
    chemin, _ = _choisir("gama_headless", "GAMA_HEADLESS", GAMA_HEADLESS_CANDIDATS)
    return chemin


def preparer() -> None:
    """Crée les sous-dossiers d'écriture s'ils manquent."""
    for dossier in (TERRAINS, SIMULATIONS, RESULTATS):
        dossier.mkdir(parents=True, exist_ok=True)


def afficher(chemin: Path | str | None) -> str | None:
    """Écriture courte d'un chemin, relative au repère le plus proche.

    Un chemin absolu complet est illisible et propre à une machine : il change d'un
    utilisateur à l'autre, alors que la structure, elle, ne change jamais. On affiche
    donc **relativement** — au dossier de l'application pour ce qui lui appartient, au
    dossier personnel pour le reste. Une seule ancre absolue est donnée, celle de
    l'application, et tout se lit par rapport à elle.
    """
    if chemin is None:
        return None
    p = Path(chemin)
    for racine, prefixe in ((APP, ""), (Path.home(), "~/")):
        try:
            return prefixe + Path(p).relative_to(racine).as_posix()
        except ValueError:
            continue
    return str(p)


def etat() -> dict:
    """Ce que l'application trouve, pour l'afficher plutôt que de le supposer.

    Rendre ces chemins visibles évite le diagnostic à l'aveugle : quand GAMA ne
    démarre pas, la première question est toujours « où cherche-t-il ? ». Chaque entrée
    porte donc son écriture courte (``affiche``) et son chemin complet (``chemin``) :
    la première pour lire, la seconde pour copier dans un terminal.
    """
    headless = gama_headless()
    racine, source_maelia = _choisir("maelia_root", "MAELIA_ROOT", MAELIA_CANDIDATS)
    racine = racine or MAELIA_CANDIDATS[0]
    _, source_gama = _choisir("gama_headless", "GAMA_HEADLESS", GAMA_HEADLESS_CANDIDATS)
    return {
        "application": str(APP),
        "espaces": {"chemin": str(ESPACES), "affiche": afficher(ESPACES),
                    "fichiers": sorted(p.name for p in ESPACES.glob("*.json"))},
        "simulations": {"chemin": str(SIMULATIONS), "affiche": afficher(SIMULATIONS),
                        "plans": sorted(p.name for p in SIMULATIONS.iterdir() if p.is_dir())
                        if SIMULATIONS.exists() else []},
        "resultats": {"chemin": str(RESULTATS), "affiche": afficher(RESULTATS)},
        "terrains": {"chemin": str(TERRAINS), "affiche": afficher(TERRAINS),
                     "construits": sorted(p.name for p in TERRAINS.iterdir() if p.is_dir())
                     if TERRAINS.exists() else []},
        "maelia": {"chemin": str(racine), "affiche": afficher(racine),
                   "present": racine.exists(),
                   "launcher": launcher().exists(), "source": source_maelia,
                   "reglable": True},
        "gama": {"chemin": str(headless) if headless else None,
                 "affiche": afficher(headless),
                 "present": bool(headless and headless.exists()),
                 "source": source_gama, "reglable": True},
        "climats": {"chemin": str(CLIMATS_OBSERVEE), "affiche": afficher(CLIMATS_OBSERVEE),
                    "annees": len(list(CLIMATS_OBSERVEE.glob("*.csv")))
                    if CLIMATS_OBSERVEE.exists() else 0},
    }
