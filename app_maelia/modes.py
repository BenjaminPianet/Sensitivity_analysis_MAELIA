"""Les deux modes de l'application, et rien d'autre.

L'ancien catalogue proposait trois « espaces de référence » nommés v1, v2 et climats.
C'était clair pour qui avait suivi leur histoire, opaque pour tout le monde d'autre.
Il n'y a désormais que deux façons de travailler, distinguées par une seule question :
**les simulations existent-elles déjà ?**

── Mode « données » ──────────────────────────────────────────────────────────
On analyse des simulations déjà faites. L'espace n'est alors pas un choix : il est
**imposé par les données**, puisqu'elles ont été produites par un plan précis. Le
`space_spec.json` déposé auprès du jeu de données fait foi. L'utilisateur peut
restreindre à l'intérieur de cet espace — c'est le sens des bornes de l'arborescence —
mais pas l'élargir, faute de simulations pour le remplir.

── Mode « sur mesure » ───────────────────────────────────────────────────────
On construit un plan neuf. L'espace est entièrement modifiable, et le climat devient
un paramètre comme un autre. L'activer a une conséquence matérielle : le climat étant
porté par l'îlot, l'application doit **construire un terrain** à huit îlots, un par
type de climat. C'est annoncé avant, pas découvert après.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chemins import DONNEES, ESPACES
from .space import SpaceSpec

MODE_DONNEES = "donnees"
MODE_SUR_MESURE = "sur_mesure"

# Base des plans sur mesure : le plan corrigé, dont les apports sont séparés en
# tallage / montaison / épiaison. C'est celui qu'il faut utiliser pour tout nouveau
# plan — l'ancien confondait les apports (cf. SYNTHESE_v2.md).
BASE_SUR_MESURE = ESPACES / "spec_terrainSA_v2.json"

CLIMATS = ["montagne", "semi_continental", "oceanique_degrade", "oceanique_altere",
           "oceanique_franc", "mediterraneen_altere", "sud_ouest", "mediterraneen_franc"]

LIBELLES_CLIMATS = {
    "montagne": "Montagne", "semi_continental": "Semi-continental",
    "oceanique_degrade": "Océanique dégradé", "oceanique_altere": "Océanique altéré",
    "oceanique_franc": "Océanique franc", "mediterraneen_altere": "Méditerranéen altéré",
    "sud_ouest": "Bassin du Sud-Ouest", "mediterraneen_franc": "Méditerranéen franc",
}

# Types de sol du terrain, dans l'ordre du fichier. Contrairement au climat, le sol
# n'est pas géographique : l'îlot le désigne par son attribut ID_SOL.
SOLS = ["limono-sableux", "limoneux", "argilo-calcaire"]

LIBELLES_SOLS = {
    "limono-sableux": "Limono-sableux — 120 cm, 24,9 % d'argile, 3,1 % de MO",
    "limoneux": "Limoneux — 165 cm, 14,6 % d'argile, 1,7 % de MO",
    "argilo-calcaire": "Argilo-calcaire — 80 cm, 44,5 % d'argile, 5,5 % de MO",
}

DESCRIPTIONS = {
    MODE_DONNEES: (
        "Analyser des simulations déjà produites. L'espace est imposé par le jeu de "
        "données : on peut le restreindre, pas l'élargir."),
    MODE_SUR_MESURE: (
        "Construire un plan neuf, à simuler ensuite avec GAMA. L'espace est libre, et "
        "le climat peut y être ajouté — ce qui suppose de construire un terrain."),
}


@dataclass(frozen=True)
class JeuPrecharge:
    """Un jeu de simulations livré avec l'application."""

    cle: str
    chemin: Path
    nom: str
    n_points: int
    avec_climat: bool


def jeux_precharges() -> list[JeuPrecharge]:
    """Les jeux de données que l'application connaît, sans rien exiger de l'utilisateur."""
    trouves = []
    for dossier in sorted(DONNEES.iterdir()) if DONNEES.exists() else []:
        if not dossier.is_dir() or not (dossier / "dataset_metamodel.csv").exists():
            continue
        spec_path = dossier / "space_spec.json"
        nom, climat = dossier.name, False
        if spec_path.exists():
            brut = json.loads(spec_path.read_text(encoding="utf-8"))
            nom = brut.get("name", dossier.name)
            climat = any(v.get("name") == "climat" for v in brut.get("variables", []))
        # Compter les lignes sans charger le fichier entier.
        with open(dossier / "dataset_metamodel.csv") as handle:
            n = sum(1 for _ in handle) - 1
        trouves.append(JeuPrecharge(dossier.name, dossier, nom, n, climat))
    return trouves


def espace_sur_mesure(avec_climat: bool = False, avec_sol: bool = False) -> SpaceSpec:
    """Espace modifiable pour un plan neuf, climat et sol compris si demandés.

    Tous deux sont **stratifiés** : leur valeur n'est pas tirée par SMT mais imposée
    par l'îlot auquel appartient la parcelle. C'est pourquoi les activer oblige à
    construire un terrain — à huit emplacements pour le climat, et autant d'îlots
    empilés que de sols à chaque emplacement.
    """
    brut = json.loads(BASE_SUR_MESURE.read_text(encoding="utf-8"))
    ajouts = []
    if avec_sol:
        ajouts.append("sol")
        brut["variables"].insert(0, {
            "name": "sol", "label": "Type de sol", "kind": "categorical",
            "domain": SOLS, "always_active": True, "stratified": True,
        })
    if avec_climat:
        ajouts.append("climat")
        brut["variables"].insert(0, {
            "name": "climat", "label": "Type de climat", "kind": "categorical",
            "domain": CLIMATS, "always_active": True, "stratified": True,
        })
    brut["name"] = ("Plan sur mesure, avec " + " et ".join(ajouts) if ajouts
                    else "Plan sur mesure")
    return SpaceSpec.from_dict(brut)


def espace_du_jeu(dossier: str | Path) -> tuple[SpaceSpec, str]:
    """Espace décrivant un jeu de données, et d'où il vient.

    Un jeu produit par cette application porte sa spécification ; un jeu antérieur au
    format n'en a pas, et l'on retombe alors sur le plan historique à quatorze
    paramètres — le seul qui décrive ces données-là.
    """
    chemin = Path(dossier).expanduser()
    embarquee = chemin / "space_spec.json"
    if embarquee.exists():
        return SpaceSpec.load(embarquee), "jeu de données"
    return SpaceSpec.load(ESPACES / "spec_terrainSA_v1.json"), "plan historique (par défaut)"


def terrain_requis(spec: SpaceSpec) -> bool:
    """Un espace portant une variable stratifiée exige un terrain adapté."""
    return bool(spec.stratified_variables())
