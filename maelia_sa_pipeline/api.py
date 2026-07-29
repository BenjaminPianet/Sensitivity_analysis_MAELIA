from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_OUTPUT_ROOT
from .pipeline import run_analysis

# L'app tourne en local : le navigateur et l'API sont sur la même machine, donc les
# chemins saisis dans l'interface désignent bien le disque de l'utilisateur.
UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(
    title="MAELIA Sensitivity Analysis API",
    description=(
        "Pipeline web pour ANOVA à un facteur, HSIC-ANOVA, indices de Sobol par PCE "
        "ordre 1 et total, et courbes PDP/ICE par sous-espace de l'espace SMT hiérarchique."
    ),
    version="0.1.0",
)


class AnalysisRequest(BaseModel):
    log_dir: str = Field(..., description="Répertoire contenant les logs MAELIA, par exemple simulations/log_terrainSA")
    dataset_path: str | None = Field(
        None,
        description="Chemin optionnel vers dataset_metamodel.csv exporté par le notebook de simulation.",
    )
    output_dir: str | None = Field(None, description="Dossier de sortie. Par défaut : analysis/web_runs/<run_id>")
    targets: list[str] | None = Field(None, description="Sorties à analyser, par défaut N_lixi, dCorg, rdt")
    features: list[str] | None = Field(None, description="Colonnes de paramètres à utiliser. Par défaut : les colonnes du plan SMT courant")
    analyses: list[str] | None = Field(None, description="Blocs d'analyse à exécuter. Par défaut : profil rapide recommandé")
    sample_size: int | None = Field(
        2000,
        ge=100,
        le=200000,
        description="Nombre de simulations tirées du dataset pour les analyses (sous-échantillonnage reproductible). Plafonné au nombre de lignes disponibles.",
    )
    n_bins: int = Field(4, ge=2, le=8, description="Nombre de classes pour discrétiser les variables continues en ANOVA")
    sobol_n_mc: int = Field(2000, ge=200, le=50000, description="Déprécié : non utilisé. L'app compare désormais des métamodèles (R²/Q²) au lieu d'estimer des indices de Sobol.")
    tree_max_depth: int = Field(4, ge=1, le=8, description="Profondeur maximale des arbres de décision")
    random_state: int = Field(42, description="Graine aléatoire")



app.mount("/ui/static", StaticFiles(directory=UI_DIR), name="maelia-ui")


@app.get("/", response_class=HTMLResponse)
def home() -> FileResponse:
    return FileResponse(
        UI_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyses")
def create_analysis(request: AnalysisRequest) -> dict[str, Any]:
    """Lance la pipeline et renvoie le manifeste, consommé tel quel par l'interface."""
    try:
        # model_dump() en pydantic v2, dict() en v1.
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return run_analysis(**payload)
    except Exception as exc:
        # Erreurs de saisie (log_dir invalide, plan SMT non conforme...) : on renvoie le
        # message d'origine en 400, il est affiché tel quel dans l'interface.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ExportRequest(BaseModel):
    run_id: str = Field(..., description="Identifiant du run dont provient le fichier.")
    relative_path: str = Field(..., description="Chemin du fichier relatif au dossier du run, ex. rdt/metamodel_comparison_rdt.png")
    dest_dir: str = Field(..., description="Répertoire local de destination choisi par l'utilisateur.")


@app.post("/export")
def export_asset(request: ExportRequest) -> dict[str, str]:
    """Copie une figure ou un CSV du run vers un répertoire local choisi par l'utilisateur.

    L'application étant lancée en local, la destination est un chemin de la machine
    de l'utilisateur. La source est contrainte au dossier de résultats du run.
    """
    # relative_path vient du navigateur : resolve() puis vérification d'appartenance à
    # root, sinon un "../.." donnerait accès à n'importe quel fichier de la machine.
    root = (DEFAULT_OUTPUT_ROOT / request.run_id).resolve()
    src = (root / request.relative_path).resolve()
    if root not in src.parents and src != root:
        raise HTTPException(status_code=400, detail="Chemin source hors du dossier de résultats.")
    if not src.exists() or not src.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {request.relative_path}")

    dest_dir_raw = request.dest_dir.strip()
    if not dest_dir_raw:
        raise HTTPException(status_code=400, detail="Précise un répertoire de destination.")
    dest_dir = Path(dest_dir_raw).expanduser()
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / src.name
        shutil.copy2(src, target)
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Impossible d'écrire dans {dest_dir} : {exc}") from exc
    return {"status": "ok", "path": str(target), "filename": src.name}


@app.get("/analyses/{run_id}/manifest")
def get_manifest(run_id: str) -> dict[str, Any]:
    path = DEFAULT_OUTPUT_ROOT / run_id / "manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest introuvable : {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/analyses/{run_id}/report", response_class=HTMLResponse)
def get_report(run_id: str) -> HTMLResponse:
    path = DEFAULT_OUTPUT_ROOT / run_id / "report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Rapport introuvable : {path}")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/analyses/{run_id}/file")
def get_file(run_id: str, relative_path: str):
    """Variante de get_generated_asset avec le chemin en paramètre de requête, pour les
    chemins contenant des caractères mal supportés directement dans l'URL."""
    root = (DEFAULT_OUTPUT_ROOT / run_id).resolve()
    path = (root / relative_path).resolve()
    # Même garde que dans export_asset.
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="Chemin hors du dossier de résultats.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {relative_path}")
    return FileResponse(path)


@app.get("/analyses/{run_id}/{relative_path:path}")
def get_generated_asset(run_id: str, relative_path: str):
    """Sert une figure ou un CSV d'un run : c'est la cible des balises <img> et des liens
    construits côté navigateur."""
    root = (DEFAULT_OUTPUT_ROOT / run_id).resolve()
    path = (root / relative_path).resolve()
    # Même garde que dans export_asset.
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="Chemin hors du dossier de résultats.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {relative_path}")
    return FileResponse(path)
