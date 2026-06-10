from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DEFAULT_OUTPUT_ROOT
from .pipeline import run_analysis

UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(
    title="MAELIA Sensitivity Analysis API",
    description=(
        "Pipeline web pour ANOVA, Sobol total, régions sensibles par arbres de décision, "
        "suivi temporel des sorties et courbes PDP/ICE finales."
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
    features: list[str] | None = Field(None, description="Colonnes de paramètres à utiliser. Par défaut : feat_0...feat_25")
    n_bins: int = Field(4, ge=2, le=8, description="Nombre de classes pour discrétiser les variables continues en ANOVA")
    sobol_n_mc: int = Field(2000, ge=200, le=50000, description="Taille Monte-Carlo pour les indices de Sobol total")
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
    try:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return run_analysis(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    root = (DEFAULT_OUTPUT_ROOT / run_id).resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="Chemin hors du dossier de résultats.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {relative_path}")
    return FileResponse(path)


@app.get("/analyses/{run_id}/{relative_path:path}")
def get_generated_asset(run_id: str, relative_path: str):
    root = (DEFAULT_OUTPUT_ROOT / run_id).resolve()
    path = (root / relative_path).resolve()
    if root not in path.parents and path != root:
        raise HTTPException(status_code=400, detail="Chemin hors du dossier de résultats.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {relative_path}")
    return FileResponse(path)
