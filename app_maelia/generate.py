"""Génération des fichiers d'entrée GAMA depuis une spécification d'espace.

Produit, dans un dossier de run isolé : les fichiers dateDose, les XML headless, le
DOE tiré, la spécification, un manifeste, et la commande GAMA à lancer.

Trois principes de sûreté, tous délibérés :

1. **Rien n'est écrit dans l'installation MAELIA.** La v1 écrit ses dateDose
   directement dans ``<terrain>/modeleAgricole/agriculteurs/variants_SMT``. Écraser
   ce dossier ferait perdre la correspondance entre les fichiers et le dataset déjà
   analysé. On produit donc dans un dossier de run, et l'installation dans le terrain
   est une étape distincte, explicite, qui refuse d'écraser par défaut.

2. **Le DOE est persisté.** Le tirage stratifié est reproductible à graine fixée,
   mais la matrice reste écrite à côté de la spécification : elle rend la
   correspondance point_idx -> paramètres lisible sans rejouer le tirage, et elle
   couvre les plans produits par l'ancien tirage ADSG, lui non reproductible.

3. **Aucune écriture avant validation.** Les calendriers sont vérifiés en mémoire ;
   un calendrier infaisable interrompt la génération avant le premier fichier.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import itinerary
from .plan import build_design_space, decode, sample
from .space import SpaceSpec

# Colonnes du fichier dateDose, dans l'ordre attendu par MAELIA.
COLS = ["Trait", "nom_MAELIA", "plant_sem", "annee_deb", "id_operation",
        "ordre_op", "DATE", "PROF", "TYPE", "DOSE", "TEMPS"]

from . import chemins
from .chemins import TERRAINS

# Le terrain que GAMA lit est celui que l'XML désigne par cheminModeleVersDonnees :
# l'application peut donc garder les siens chez elle, et n'a pas à écrire dans
# l'installation MAELIA.
DEFAULT_PROJECT_INCLUDES = TERRAINS
DEFAULT_TERRAIN = TERRAINS / "terrainSA"
VARIANTS_SUBPATH = Path("modeleAgricole/agriculteurs/variants_SMT")
CLONES_PER_RUN = 100


@dataclass
class GamaConfig:
    """Ce dont la génération XML a besoin, et rien de plus."""

    # Relu à la construction, non figé au chargement : le réglage peut changer
    # en cours de session depuis l'interface.
    maelia_root: Path = field(default_factory=chemins.maelia_root)
    terrain_dir: Path = DEFAULT_TERRAIN
    project_includes_root: Path | None = DEFAULT_PROJECT_INCLUDES
    nom_decoupage: str = "terrainSA"
    annee_debut: int = 2019
    nb_annees: int = 10
    parcelles: tuple[str, ...] = ()
    launcher_parameters: dict[str, str] = field(default_factory=dict)
    # Scénario climatique. None = ne pas émettre le paramètre, donc hériter du défaut
    # du launcher — qui vaut « rcp8.5 » : c'est sous ce scénario, et non sous la météo
    # observée, qu'ont tourné les simulations historiques, sans que rien ne le dise.
    # "" force la météo observée, "rcp4.5" un scénario projeté.
    scenario_climatique: str | None = None
    # Strates du terrain : parcelle -> modalité de la variable stratifiée. Le climat
    # étant porté par l'îlot, c'est la parcelle qui décide, pas le tirage. Avec des
    # îlots de taille égale, l'équilibre du plan est garanti par construction.
    # Strates : une table par variable stratifiée, parcelle -> modalité. Deux
    # variables peuvent l'être en même temps — le climat vient de l'emplacement de
    # l'îlot, le sol de son attribut — et chacune a sa propre correspondance.
    strata: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def model_path(self) -> Path:
        return self.maelia_root / "models/main/launcherBase.gaml"

    @property
    def final_step(self) -> int:
        return (date(self.annee_debut + self.nb_annees + 1, 1, 1)
                - date(self.annee_debut, 1, 1)).days

    @property
    def campaigns(self) -> tuple[int, ...]:
        return tuple(range(self.annee_debut, self.annee_debut + self.nb_annees))

    @property
    def gama_output_dir(self) -> Path:
        return self.maelia_root / "models/main/log"

    @property
    def chemin_racine(self) -> str:
        return str(self.maelia_root) + "/"

    @property
    def chemin_modele_vers_donnees(self) -> str:
        root = self.project_includes_root or (self.terrain_dir.parent)
        return str(root) + "/"


from .chemins import gama_headless as find_gama_headless  # noqa: E402


def read_launcher_parameters(model_path: Path) -> dict[str, str]:
    """Variables exposées comme paramètres headless par le launcher MAELIA."""
    text = Path(model_path).read_text(errors="replace")
    params: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = re.search(r"parameter\s+(['\"])(.*?)\1\s+var:\s*([A-Za-z0-9_]+)", line)
        if match:
            params[match.group(3)] = match.group(2)
    return params


def read_parcelles(terrain_dir: Path, limit: int = CLONES_PER_RUN) -> list[str]:
    """Parcelles clonées déclarées par le terrain, dans l'ordre attendu."""
    bloc = Path(terrain_dir) / "modeleAgricole" / "blocsDonnees.csv"
    found: list[str] = []
    with open(bloc) as handle:
        for line in handle:
            parts = [p.strip() for p in line.strip().split(";") if p.strip()]
            if len(parts) >= 2:
                found.extend(parts[1:])
    parcelles = sorted(p for p in found if p.startswith("beauce_5_sa_"))[:limit]
    if len(parcelles) != limit:
        raise ValueError(
            f"{terrain_dir} déclare {len(parcelles)} clones, {limit} attendus. "
            "Reconstruis terrainSA avant de générer un plan."
        )
    return parcelles


def build_xml(cfg: GamaConfig, nom_simu: str, datedose_path: Path) -> str:
    """XML d'expérience headless, identique à celui produit par run_terrainSA_batch.

    ``datedose_path`` est le chemin **une fois installé dans le terrain** : MAELIA
    attend un chemin relatif au terrain, préfixé d'un slash. Un fichier resté hors du
    terrain donnerait un chemin absolu que le modèle ne saurait pas résoudre.
    """
    datedose_path = Path(datedose_path)
    try:
        datedose_for_gama = "/" + datedose_path.relative_to(cfg.terrain_dir).as_posix()
    except ValueError:
        datedose_for_gama = datedose_path.as_posix()

    root = ET.Element("Experiment_plan")
    sim = ET.SubElement(root, "Simulation")
    sim.set("experiment", "simulationBase")
    sim.set("finalStep", str(cfg.final_step))
    sim.set("id", "0")
    sim.set("seed", "1.0")
    sim.set("sourcePath", str(cfg.model_path))

    params = ET.SubElement(sim, "Parameters")

    def p(name: str, ptype: str, value, var: str) -> None:
        # Un paramètre absent du launcher courant est ignoré, comme en v1 : les
        # versions de MAELIA n'exposent pas toutes les mêmes variables.
        if var not in cfg.launcher_parameters:
            return
        element = ET.SubElement(params, "Parameter")
        element.set("name", name)
        element.set("type", ptype)
        element.set("value", str(value))
        element.set("var", var)

    p("executerSurCluster: ", "BOOLEAN", "false", "executerSurCluster")
    p("cheminRacineMaelia", "STRING", cfg.chemin_racine, "cheminRacineMaelia")
    p("cheminModeleVersDonnees", "STRING", cfg.chemin_modele_vers_donnees, "cheminModeleVersDonnees")
    p("cheminSorties", "STRING", str(cfg.gama_output_dir), "cheminRelatifDuDossierDeSortieDeSimulation")
    p("anneeDebutSimulation : ", "INT", cfg.annee_debut, "anneeDebutSimulation")
    p("nbAnneesSimulation : ", "INT", cfg.nb_annees, "nbAnneesSimulation")
    p("nomSimulation : ", "STRING", nom_simu, "nomSimulation")
    p("nomDecoupageZonePourLectureFichiers : ", "STRING", cfg.nom_decoupage, "nomDecoupageZonePourLectureFichiers")
    p("modeVerbeux :", "BOOLEAN", "false", "verboseMode")
    p("simulationSurParcelle : ", "BOOLEAN", "false", "executerUneSeuleParcelle")
    p("idParcelleASimuler : ", "STRING", cfg.parcelles[0], "nomParcelleAffichee")
    p("executerModeleAgricole : ", "BOOLEAN", "true", "executerModeleAgricole")
    p("nomChoixAssolement : ", "STRING", "DateDose", "nomChoixAssolement")
    p("fichierDateDose : ", "STRING", datedose_for_gama, "fichierDateDose")
    p("avecContrainteDeMainOeuvre : ", "BOOLEAN", "false", "avecContrainteDeMainOeuvre")
    p("isIrrigationSimulee : ", "BOOLEAN", "false", "isIrrigationSimulee")
    p("executerModeleHydrographique : ", "BOOLEAN", "false", "executerModeleHydrographique")
    p("executerModeleNormatif : ", "BOOLEAN", "false", "executerModeleNormatif")
    p("executerEcritureFichiers : ", "BOOLEAN", "true", "executerEcritureFichiers")
    p("sorties eau", "BOOLEAN", "true", "sorties_eau")
    p("sorties azote", "BOOLEAN", "true", "sorties_azote")
    p("sorties carbone et GES", "BOOLEAN", "true", "sorties_carboneGES")
    p("Sortie N_Cstock_Parcelles", "BOOLEAN", "true", "N_Cstock_Parcelles")
    p("Sortie N_lixi_Parcelles", "BOOLEAN", "true", "N_lixi_Parcelles")
    # Émis uniquement s'il est explicitement demandé : ne rien émettre reproduit à
    # l'octet près l'XML de run_terrainSA_batch, défaut du launcher compris.
    #
    # Le libellé est repris tel quel du launcher, et non écrit en dur : celui-ci
    # déclare « nomScenarioClimatique : » sans espace final, là où tous les autres en
    # portent un. Un libellé qui ne correspond pas fait abandonner GAMA en silence,
    # sans la moindre ligne d'erreur.
    if cfg.scenario_climatique is not None:
        p(cfg.launcher_parameters.get("nomScenarioClimatique", "nomScenarioClimatique :"),
          "STRING", cfg.scenario_climatique, "nomScenarioClimatique")

    ET.SubElement(sim, "Outputs")
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding="UTF-8", xml_declaration=True)
    return buf.getvalue().decode("UTF-8").replace(
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
    )


@dataclass
class GeneratedPlan:
    output_dir: Path
    n_points: int
    n_runs: int
    parcelles: tuple[str, ...]
    datedose_files: list[Path]
    xml_files: list[Path]
    report: itinerary.CalendarReport
    manifest: dict

    def gama_command(self, gama_headless: str | Path, workspace: str | Path | None = None) -> str:
        """Commande à lancer une fois les dateDose installés dans le terrain.

        Le ``mkdir -p`` n'est pas une précaution de confort : ``gama-headless.sh``
        crée son dossier de sortie avec un ``mkdir`` **non récursif**, qui échoue si
        le dossier parent n'existe pas. GAMA s'arrête alors sur « Unable to create
        directory […] Check your file permission ! », message qui désigne les droits
        alors que le problème est l'arborescence.
        """
        ws = Path(workspace) if workspace else self.output_dir / "ws"
        return (
            f'for xml in "{self.output_dir / "xml"}"/*.xml; do\n'
            f'  ws="{ws}/$(basename "$xml" .xml)"\n'
            f'  mkdir -p "$ws"\n'
            f'  bash "{Path(gama_headless)}" "$xml" "$ws"\n'
            f'done'
        )

    def spec_command(self, log_dir: str | Path) -> str:
        """Dépose la spécification à côté du futur dataset.

        C'est ce qui referme la boucle : une fois GAMA passé et le dataset
        reconstruit, l'analyse retrouvera dans ce dossier l'espace exact qui a
        produit ces simulations, au lieu de supposer le plan historique.
        """
        return f'cp "{self.output_dir / "space_spec.json"}" "{Path(log_dir)}/"'

    def install_command(self) -> str:
        """Copie des dateDose vers le terrain, à exécuter en connaissance de cause.

        Affichée plutôt qu'exécutée : cette commande écrit dans l'installation MAELIA
        et peut écraser les fichiers d'un plan précédent.
        """
        target = Path(self.manifest["installation_cible"])
        return f'cp "{self.output_dir / "dateDose"}"/*.csv "{target}/"'


def generate(
    spec: SpaceSpec,
    output_dir: str | Path,
    n_points: int = 500,
    cfg: GamaConfig | None = None,
    seed: int = 42,
    allow_drift: bool = True,
) -> GeneratedPlan:
    """Produit les fichiers d'entrée GAMA correspondant à cet espace.

    Rien n'est écrit tant que les calendriers ne sont pas validés, et rien n'est
    écrit hors de ``output_dir``.
    """
    cfg = cfg or GamaConfig()
    if not cfg.parcelles:
        cfg = GamaConfig(**{**cfg.__dict__, "parcelles": tuple(read_parcelles(cfg.terrain_dir))})
    if not cfg.launcher_parameters:
        cfg = GamaConfig(**{**cfg.__dict__,
                            "launcher_parameters": read_launcher_parameters(cfg.model_path)})

    parcelles = list(cfg.parcelles)
    n_runs = max(1, -(-int(n_points) // len(parcelles)))
    n_doe = n_runs * len(parcelles)  # on remplit toujours les runs, comme la v1

    built = build_design_space(spec)
    xt, _ = sample(built, n_doe, seed=seed)
    points = decode(built, xt)

    # Injection des variables stratifiées : leur valeur ne vient pas du tirage mais de
    # la parcelle sur laquelle le point sera exécuté. Sans cette étape, le générateur
    # d'itinéraires ne verrait pas le climat, et l'analyse croirait qu'il ne varie pas.
    if built.stratified:
        for name in built.stratified:
            table = cfg.strata.get(name, {})
            manquantes = [p for p in parcelles if p not in table]
            if manquantes:
                raise ValueError(
                    f"Variable stratifiée « {name} » déclarée, mais {len(manquantes)} "
                    f"parcelle(s) sans strate (ex. {manquantes[:3]}). Fournis "
                    "GamaConfig.strata depuis affectation_strates.csv du terrain.")
        for index, point in enumerate(points):
            parcelle = parcelles[index % len(parcelles)]
            for name in built.stratified:
                point[name] = cfg.strata[name][parcelle]

    # Validation avant toute écriture : un calendrier infaisable arrête tout.
    report = itinerary.validate_points(points, campaigns=cfg.campaigns, spec=spec)
    if report.n_failed:
        raise ValueError(
            f"{report.n_failed}/{report.n_points} calendriers infaisables — aucun fichier écrit.\n"
            + "\n".join(f"  - {line}" for line in report.examples)
        )
    if report.drift and not allow_drift:
        raise ValueError(
            "Des dates réalisées sortent des fenêtres demandées — aucun fichier écrit.\n"
            + "\n".join(f"  - {line}" for line in report.summary())
        )

    out = Path(output_dir).expanduser()
    datedose_dir = out / "dateDose"
    xml_dir = out / "xml"
    datedose_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    installed_variants = cfg.terrain_dir / VARIANTS_SUBPATH
    datedose_files: list[Path] = []
    xml_files: list[Path] = []

    for run_index in range(n_runs):
        nom_simu = f"smt_{run_index:03d}"
        operations: list[dict] = []
        for offset, parcelle_id in enumerate(parcelles):
            point_idx = run_index * len(parcelles) + offset
            for year in cfg.campaigns:
                operations.extend(itinerary.build_operations(points[point_idx], year, parcelle_id))

        issues = itinerary.check_sequence(operations)
        if issues:
            raise ValueError(
                f"Séquence invalide pour {nom_simu} — génération interrompue.\n"
                + "\n".join(f"  - {i}" for i in issues[:10])
            )

        path = datedose_dir / f"dateDose_{nom_simu}.csv"
        pd.DataFrame(operations, columns=COLS).to_csv(path, sep=";", index=False)
        datedose_files.append(path)

        # Le XML pointe vers l'emplacement d'installation, pas vers le dossier de run.
        xml_path = xml_dir / f"{nom_simu}.xml"
        xml_path.write_text(build_xml(cfg, nom_simu, installed_variants / path.name))
        xml_files.append(xml_path)

    # Table des points : c'est la matrice X de l'analyse, strates comprises. Elle
    # relie chaque point à la parcelle et au run qui l'exécuteront.
    table = []
    for index, point in enumerate(points):
        run_index, offset = divmod(index, len(parcelles))
        table.append({"point_idx": index, "run": f"smt_{run_index:03d}",
                      "parcelle": parcelles[offset], **point})
    pd.DataFrame(table).to_csv(out / "points.csv", index=False)

    # La matrice reste écrite : elle donne la correspondance point_idx -> paramètres
    # sans avoir à rejouer le tirage.
    np.save(out / "doe.npy", xt)
    (out / "space_spec.json").write_text(
        json.dumps(_spec_to_dict(spec), ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "n_points": n_doe,
        "n_points_demandes": int(n_points),
        "n_runs": n_runs,
        "clones_per_run": len(parcelles),
        "seed": seed,
        "reproductible": True,
        "note_reproductibilite":
            "Plan stratifié par sous-espace, hypercube latin à l'intérieur : la même "
            "graine redonne le même plan. Les plans antérieurs au 14 août 2026 "
            "viennent du tirage ADSG, qui n'était pas reproductible.",
        "echantillonnage": "stratifie_lhs",
        "annee_debut": cfg.annee_debut,
        "nb_annees": cfg.nb_annees,
        "final_step": cfg.final_step,
        "maelia_root": str(cfg.maelia_root),
        "terrain_dir": str(cfg.terrain_dir),
        "installation_cible": str(installed_variants),
        "design_names": list(built.design_names),
        "constants": {k: str(v) for k, v in built.constants.items()},
        "drift": report.drift,
        "stratifiees": list(built.stratified),
        "equilibre_strates": (
            pd.DataFrame(table)[list(built.stratified)[0]].value_counts().to_dict()
            if built.stratified else {}),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return GeneratedPlan(
        output_dir=out, n_points=n_doe, n_runs=n_runs, parcelles=tuple(parcelles),
        datedose_files=datedose_files, xml_files=xml_files, report=report, manifest=manifest,
    )


def _spec_to_dict(spec: SpaceSpec) -> dict:
    return {
        "schema_version": 1,
        "name": spec.name,
        "sentinel_inactive": spec.sentinel_inactive,
        "meta_variables": [
            {"name": m.name, "label": m.label, "kind": m.kind,
             "levels": [{"value": lv.value, "tag": lv.tag, "label": lv.label,
                         "activates": list(lv.activates)} for lv in m.levels],
             "domain": list(m.domain), "window": list(m.window)}
            for m in spec.meta_variables
        ],
        "variables": [
            {k: v for k, v in {
                "name": v.name, "label": v.label, "kind": v.kind,
                "domain": list(v.domain),
                "window": list(v.window) if v.window is not None else None,
                "always_active": v.always_active or None,
                "scale": v.scale,
                # Sans ce drapeau, la spécification relue perdrait le fait que la
                # variable est imposée par le terrain : elle repasserait dans le
                # tirage SMT au plan suivant.
                "stratified": v.stratified or None,
            }.items() if v is not None}
            for v in spec.variables
        ],
    }


def install(plan: GeneratedPlan, terrain_dir: str | Path | None = None,
            force: bool = False) -> list[Path]:
    """Copie les dateDose générés dans le terrain, là où GAMA les lira.

    Étape séparée et explicite : elle écrit dans l'installation MAELIA. Par défaut
    elle refuse d'écraser un fichier existant — ces fichiers peuvent correspondre à
    un jeu de simulations déjà analysé.
    """
    target = Path(terrain_dir or plan.manifest["terrain_dir"]) / VARIANTS_SUBPATH
    target.mkdir(parents=True, exist_ok=True)

    existing = [source.name for source in plan.datedose_files if (target / source.name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{len(existing)} fichier(s) seraient écrasés dans {target} "
            f"(dont {', '.join(existing[:3])}). Sauvegarde-les, choisis un autre terrain, "
            "ou relance avec force=True en connaissance de cause."
        )

    copied = []
    for source in plan.datedose_files:
        destination = target / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied
