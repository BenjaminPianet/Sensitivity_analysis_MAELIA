"""API de l'application : choix du mode, couverture, plan, analyse.

Depuis le dossier qui contient ``app_maelia`` :

    python3 -m uvicorn app_maelia.api:app --reload --port 8000

Tout ce que l'application lit et écrit se trouve sous ``app_maelia/`` ; les seuls
chemins extérieurs sont l'installation MAELIA et GAMA, réglables par variables
d'environnement (voir ``chemins.py``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import analysis, chemins, data, generate, itinerary, modes
from .plan import build_design_space
from .space import CATEGORICAL, CONTINUOUS, ORDINAL, SpaceSpec, SpecError

UI_DIR = chemins.UI
PLAN_RUNS_ROOT = chemins.SIMULATIONS

app = FastAPI(
    title="MAELIA Sensitivity Studio v2",
    description="Sélection d'un espace de conception hiérarchique et analyse des simulations disponibles.",
    version="0.2.0",
)

app.mount("/ui/static", StaticFiles(directory=UI_DIR), name="maelia-ui-v2")


class SpaceRequest(BaseModel):
    """Ce que l'utilisateur demande : un mode, une source, et ses restrictions.

    Les fenêtres sont indexées par nom de variable — continue vers [min, max],
    ordinale vers la liste des niveaux retenus. Toute variable absente garde la
    fenêtre de l'espace courant.
    """

    mode: str = Field(
        modes.MODE_DONNEES,
        description="« donnees » pour analyser des simulations existantes, "
                    "« sur_mesure » pour construire un plan neuf.")
    dossier_donnees: str | None = Field(
        None,
        description="Mode « donnees » : dossier contenant dataset_metamodel.csv. "
                    "Un jeu préchargé peut être désigné par son nom court.")
    avec_climat: bool = Field(
        False,
        description="Mode « sur mesure » : ajouter le climat à l'espace. Impose de "
                    "construire un terrain à huit emplacements.")
    avec_sol: bool = Field(
        False,
        description="Mode « sur mesure » : ajouter le type de sol à l'espace. Impose "
                    "de construire un terrain portant les trois sols.")
    windows: dict[str, list] = Field(default_factory=dict)
    targets: list[str] | None = Field(None, description="Sorties à analyser.")
    analyses: list[str] | None = Field(
        None,
        description="Analyses à calculer parmi one_factor, metamodel, "
                    "metamodel_comparison, hsic. Par défaut toutes celles que la "
                    "couverture autorise ; HSIC est quadratique en nombre de points.")


def _config_terrain(terrain: Path, spec: SpaceSpec) -> generate.GamaConfig:
    """Configuration GAMA pour un terrain portant des strates.

    Climat et sol sont portés par l'îlot : chaque parcelle en hérite. Les
    correspondances sont lues dans le terrain et transmises variable par variable, ce
    qui garantit l'équilibre du plan par construction plutôt que par le tirage.
    """
    import pandas as pd

    from . import terrain as terrain_mod

    fichier = terrain / "affectation_strates.csv"
    if not fichier.exists():                       # terrains produits avant le sol
        fichier = terrain / "affectation_climats.csv"
    affectation = pd.read_csv(fichier).sort_values(["ilot", "parcelle"])

    strates: dict[str, dict[str, str]] = {}
    for var in spec.stratified_variables():
        if var.name == "climat" and "climat_code" in affectation:
            modalites = affectation.climat_code.map(lambda c: modes.CLIMATS[int(c) - 1])
        elif var.name in affectation.columns:
            modalites = affectation[var.name]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Le terrain {terrain.name} ne porte pas la strate « {var.name} ». "
                       f"Colonnes disponibles : {sorted(affectation.columns)}")
        # Le terrain peut porter la colonne sans porter la variation : un terrain
        # construit pour le climat seul écrit un sol constant. Le plan serait alors
        # muet sur cette strate, sans que rien ne le signale.
        observees = sorted(set(modalites.dropna()))
        if len(observees) < len(var.domain):
            raise HTTPException(
                status_code=400,
                detail=f"Le terrain {terrain.name} ne porte que {len(observees)} "
                       f"modalité(s) de « {var.name} » sur {len(var.domain)} "
                       f"attendues ({observees}). Reconstruis-le pour cette strate.")
        strates[var.name] = dict(zip(affectation.parcelle, modalites))

    return generate.GamaConfig(
        terrain_dir=terrain, nom_decoupage=terrain.name,
        parcelles=tuple(affectation.parcelle), strata=strates,
        scenario_climatique=terrain_mod.SCENARIO_NAME,
        launcher_parameters=generate.read_launcher_parameters(chemins.launcher()))


def _dossier_donnees(request: SpaceRequest | None) -> Path:
    """Dossier de données demandé, qu'il soit désigné par son nom court ou son chemin."""
    demande = getattr(request, "dossier_donnees", None)
    if not demande:
        return Path(data.DEFAULT_LOG_DIR)
    court = chemins.DONNEES / demande
    return court if court.is_dir() else Path(demande).expanduser()


def _resolve_spec(request: SpaceRequest | None = None) -> tuple[SpaceSpec, str]:
    """Espace courant, et d'où il vient — selon le mode demandé.

    En mode « données », l'espace n'est pas un choix : il est imposé par le jeu, qui
    a été produit par un plan précis. En mode « sur mesure », il est construit.
    """
    if getattr(request, "mode", modes.MODE_DONNEES) == modes.MODE_SUR_MESURE:
        spec = modes.espace_sur_mesure(bool(getattr(request, "avec_climat", False)),
                                       bool(getattr(request, "avec_sol", False)))
        return spec, "plan sur mesure"

    dossier = _dossier_donnees(request)
    if not dossier.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dossier de données introuvable : {dossier}. "
                   f"Jeux disponibles : {[j.cle for j in modes.jeux_precharges()]}")
    return modes.espace_du_jeu(dossier)


def _base_spec() -> SpaceSpec:
    return _resolve_spec()[0]


def _safe_slug(value: str) -> str:
    """Nom de dossier sûr sur tous les OS : alphanumérique, tiret, souligné."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value).strip("_")
    return cleaned[:40] or "plan"


def _apply(request: SpaceRequest) -> SpaceSpec:
    spec = _resolve_spec(request)[0]
    if not request.windows:
        return spec
    try:
        return spec.with_window(**{k: list(v) for k, v in request.windows.items()})
    except SpecError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Variable inconnue : {exc}") from exc


def _load(request: SpaceRequest, spec: SpaceSpec) -> data.Dataset:
    try:
        return data.load(spec, _dossier_donnees(request))
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(
        UI_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/modes")
def list_modes() -> dict[str, Any]:
    """Les deux modes, les jeux préchargés, et où l'application range ses fichiers."""
    return {
        "modes": [
            {"cle": modes.MODE_DONNEES, "libelle": "Données existantes",
             "description": modes.DESCRIPTIONS[modes.MODE_DONNEES]},
            {"cle": modes.MODE_SUR_MESURE, "libelle": "Plan sur mesure",
             "description": modes.DESCRIPTIONS[modes.MODE_SUR_MESURE]},
        ],
        "jeux_precharges": [
            {"cle": j.cle, "nom": j.nom, "n_points": j.n_points,
             "avec_climat": j.avec_climat, "chemin": str(j.chemin)}
            for j in modes.jeux_precharges()
        ],
        "climats": [{"cle": c, "libelle": modes.LIBELLES_CLIMATS[c]} for c in modes.CLIMATS],
        "sols": [{"cle": s, "libelle": modes.LIBELLES_SOLS[s]} for s in modes.SOLS],
        "chemins": chemins.etat(),
    }


class ReglageChemins(BaseModel):
    """Chemins que l'utilisateur choisit lui-même.

    Une chaîne vide efface le réglage et rend la main aux variables d'environnement,
    puis aux emplacements d'installation habituels.
    """

    maelia_root: str | None = Field(
        None, description="Racine de l'installation MAELIA (contient models/main).")
    gama_headless: str | None = Field(
        None, description="Chemin du script gama-headless.")


def _diagnostic(chemin: str | None, attendu: Path | None = None) -> dict:
    """Ce qu'on peut dire d'un chemin saisi, avant même de lancer quoi que ce soit."""
    if not chemin:
        return {"saisi": None, "existe": None, "message": "non réglé"}
    p = Path(chemin).expanduser()
    if not p.exists():
        return {"saisi": str(p), "existe": False, "message": "ce chemin n'existe pas"}
    if attendu is not None and not attendu.exists():
        return {"saisi": str(p), "existe": True,
                "message": f"le chemin existe, mais {attendu.name} est introuvable dedans"}
    return {"saisi": str(p), "existe": True, "message": "trouvé"}


@app.get("/reglages")
def get_reglages() -> dict[str, Any]:
    """Chemins réglés, ce qu'ils valent, et où l'application cherche à défaut."""
    reglages = chemins.lire_reglages()
    return {
        "reglages": reglages,
        "maelia": _diagnostic(reglages.get("maelia_root") or str(chemins.maelia_root()),
                              chemins.launcher()),
        "gama": _diagnostic(reglages.get("gama_headless")
                            or (str(chemins.gama_headless()) if chemins.gama_headless() else None)),
        "candidats": {
            "maelia": [str(c) for c in chemins.MAELIA_CANDIDATS],
            "gama": [str(c) for c in chemins.GAMA_HEADLESS_CANDIDATS],
        },
        "fichier": str(chemins.REGLAGES),
        "chemins": chemins.etat(),
    }


@app.post("/reglages")
def set_reglages(request: ReglageChemins) -> dict[str, Any]:
    """Enregistre les chemins choisis et rend le diagnostic aussitôt.

    Le réglage n'est pas refusé quand le chemin n'existe pas : l'utilisateur peut
    préparer sa configuration avant d'installer GAMA. Le diagnostic dit ce qui va.
    """
    chemins.ecrire_reglages(maelia_root=request.maelia_root,
                            gama_headless=request.gama_headless)
    return get_reglages()


@app.get("/spec")
def get_spec(mode: str = modes.MODE_DONNEES, dossier_donnees: str | None = None,
             avec_climat: bool = False, avec_sol: bool = False) -> dict[str, Any]:
    """Espace courant, sous la forme dont l'arborescence a besoin.

    L'interface a besoin de savoir, pour chaque variable, quels niveaux l'activent :
    c'est ce qui lui permet de griser en direct sans rappeler le serveur.
    """
    request = SpaceRequest(mode=mode, dossier_donnees=dossier_donnees,
                           avec_climat=avec_climat, avec_sol=avec_sol)
    spec, origin = _resolve_spec(request)
    dataset = None
    try:
        dataset = data.load(spec, _dossier_donnees(request))
    except (FileNotFoundError, ValueError):
        pass

    return {
        "origin": origin,
        "mode": mode,
        "modifiable": mode == modes.MODE_SUR_MESURE,
        "terrain_requis": modes.terrain_requis(spec),
        "dossier_donnees": str(_dossier_donnees(request)),
        "name": spec.name,
        "sentinel_inactive": spec.sentinel_inactive,
        "meta_variables": [
            {
                "name": m.name,
                "label": m.label,
                "domain": list(m.domain),
                "window": list(m.window),
                "levels": [
                    {"value": lv.value, "label": lv.label, "activates": list(lv.activates)}
                    for lv in m.levels
                ],
            }
            for m in spec.meta_variables
        ],
        "variables": [
            {
                "name": v.name,
                "label": v.label,
                "kind": v.kind,
                "domain": list(v.domain),
                "window": list(v.effective_window),
                "always_active": v.always_active,
                "scale": v.scale,
                "stratified": v.stratified,
                "unit": "kg N/ha" if v.scale == "log10" else None,
                # Niveaux qui activent la variable, par méta-variable.
                "activated_by": {
                    m.name: [lv.value for lv in m.levels if v.name in lv.activates]
                    for m in spec.meta_variables
                    if any(v.name in lv.activates for lv in m.levels)
                },
            }
            for v in spec.variables
        ],
        "targets": dataset.targets if dataset else [],
        "target_labels": data.TARGET_LABELS,
        "n_available": int(len(dataset.frame)) if dataset else 0,
        "anomalies": dataset.anomalies if dataset else [],
    }


@app.post("/coverage")
def get_coverage(request: SpaceRequest) -> dict[str, Any]:
    """Ce que les données disponibles permettent sous l'espace demandé.

    Appelé à chaque modification de l'arborescence : c'est le garde-fou qui dit,
    avant tout calcul, quelles analyses seront fiables.
    """
    spec = _apply(request)
    dataset = _load(request, spec)
    cov = analysis.coverage(spec, dataset.frame)
    built = build_design_space(spec)
    return {
        "n_points": cov.n_points,
        "n_available": cov.n_available,
        "subspaces": cov.subspaces,
        "verdicts": [v.__dict__ for v in cov.verdicts],
        "reachable": cov.reachable,
        "unconditional": cov.unconditional,
        "decreed": cov.decreed,
        "unreachable": cov.unreachable,
        "plan": {
            "design_names": list(built.design_names),
            "constants": {k: str(v) for k, v in built.constants.items()},
            "n_decreed": len(built.decreed),
            "n_subspaces": len(spec.subspaces()),
        },
        # Les écarts au domaine décrivent les données ; les avertissements de
        # filtrage décrivent ce que la sélection courante en a retenu.
        "anomalies": dataset.anomalies + spec.filter_warnings(dataset.frame),
    }


class PlanCheckRequest(SpaceRequest):
    n_points: int = Field(500, ge=20, le=5000, description="Points tirés pour le contrôle.")
    seed: int = Field(42, description="Graine du tirage de contrôle.")


@app.post("/plan/check")
def check_plan(request: PlanCheckRequest) -> dict[str, Any]:
    """Vérifie en mémoire les calendriers que cet espace produirait.

    À lancer avant toute génération : le contrôle est immédiat, là où découvrir le
    problème après coup impose de relancer les simulations. Rien n'est écrit.
    """
    spec = _apply(request)
    built = build_design_space(spec)
    report = itinerary.validate_space(spec, n_points=request.n_points,
                                      seed=request.seed, plan=built)
    n_runs = -(-request.n_points // 100)  # 100 parcelles clonées par run GAMA
    return {
        "n_points": report.n_points,
        "n_ok": report.n_ok,
        "n_failed": report.n_failed,
        "failure_rate": round(report.failure_rate, 4),
        "failures": report.failures,
        "examples": report.examples,
        "drift": report.drift,
        "degenerate": report.degenerate,
        "sequence_issues": report.sequence_issues,
        "summary": report.summary(),
        "ok": report.ok,
        "campaigns": list(report.campaigns),
        "cost": {"n_runs": n_runs},
        "plan": {
            "design_names": list(built.design_names),
            "constants": {k: str(v) for k, v in built.constants.items()},
        },
    }


class PlanGenerateRequest(PlanCheckRequest):
    label: str | None = Field(None, description="Nom lisible du run, repris dans le dossier.")


@app.post("/plan/generate")
def generate_plan(request: PlanGenerateRequest) -> dict[str, Any]:
    """Produit les fichiers d'entrée GAMA correspondant à l'espace sélectionné.

    Écrit uniquement dans ``analysis/plan_runs_v2/<run_id>/``. L'installation dans
    le terrain MAELIA et le lancement de GAMA restent à la main de l'utilisateur :
    les deux commandes sont renvoyées, pas exécutées.
    """
    spec = _apply(request)

    # Un espace portant une strate exige un terrain fait pour elle : on le construit
    # ici plutôt que de laisser l'utilisateur découvrir l'échec après coup. Un terrain
    # par combinaison de strates — les mélanger rendrait les analyses incomparables.
    terrain_construit = None
    if modes.terrain_requis(spec):
        from . import terrain as terrain_mod

        strates = {v.name for v in spec.stratified_variables()}
        avec_climat, avec_sol = "climat" in strates, "sol" in strates
        cible = chemins.TERRAINS / terrain_mod.NOMS_TERRAIN[(avec_climat, avec_sol)]
        if not cible.exists() or terrain_mod.validate(cible):
            try:
                bati = terrain_mod.build(force=True, avec_climat=avec_climat,
                                         avec_sol=avec_sol)
                terrain_construit = str(bati.path)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Construction du terrain {cible.name} impossible : {exc}") from exc
        cfg = _config_terrain(cible, spec)
    else:
        cfg = generate.GamaConfig()

    if not cfg.terrain_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Terrain introuvable : {cfg.terrain_dir}.",
        )
    if not cfg.model_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Launcher MAELIA introuvable : {cfg.model_path}.",
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _safe_slug(request.label) if request.label else "plan"
    output_dir = PLAN_RUNS_ROOT / f"{stamp}_{slug}"

    try:
        plan = generate.generate(spec, output_dir, n_points=request.n_points,
                                 cfg=cfg, seed=request.seed)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headless = generate.find_gama_headless()
    return {
        "run_id": output_dir.name,
        "terrain": str(cfg.terrain_dir),
        "terrain_construit": terrain_construit,
        "output_dir": str(output_dir),
        "n_points": plan.n_points,
        "n_points_demandes": request.n_points,
        "n_runs": plan.n_runs,
        "clones_per_run": len(plan.parcelles),
        "files": {
            "dateDose": [p.name for p in plan.datedose_files],
            "xml": [p.name for p in plan.xml_files],
            "doe": "doe.npy",
            "spec": "space_spec.json",
            "manifest": "manifest.json",
        },
        "validation": {
            "n_failed": plan.report.n_failed,
            "drift": plan.report.drift,
            "degenerate": plan.report.degenerate,
            "summary": plan.report.summary(),
        },
        "commands": {
            "install": plan.install_command(),
            "gama": plan.gama_command(headless or "<chemin vers gama-headless.sh>"),
            "spec": plan.spec_command(_dossier_donnees(request)),
            # La collecte referme la boucle : elle relie chaque point du plan aux
            # sorties de GAMA et écrit le dataset que le mode « données » attend,
            # à côté du plan. Sans elle, les simulations restent illisibles.
            "collecte": f'python3 -m app_maelia.collecte "{output_dir}"',
            "gama_headless_trouve": str(headless) if headless else None,
        },
        "manifest": plan.manifest,
    }


@app.post("/analyse")
def run_analysis(request: SpaceRequest) -> dict[str, Any]:
    """Applique l'espace au jeu de données puis lance les analyses autorisées."""
    spec = _apply(request)
    dataset = _load(request, spec)
    targets = request.targets or dataset.targets
    unknown = [t for t in targets if t not in dataset.frame.columns]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Sorties absentes du dataset : {unknown}")
    try:
        result = analysis.run(spec, dataset.frame, targets, analyses=request.analyses)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["anomalies"] = dataset.anomalies + spec.filter_warnings(dataset.frame)
    result["dataset_path"] = str(dataset.path)
    return result
