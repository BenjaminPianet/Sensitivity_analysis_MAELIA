from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import DEFAULT_OUTPUT_ROOT, TARGET_LABELS
from .data import load_dataset
from .hsic import compute_hsic_anova
from .models import train_sparse_pce_sobol
from .pdp_subspace import compute_subspace_pdp_ice
from .stats import discretize_factors, one_factor_anova
from .viz import plot_one_factor, plot_hsic_order_decomposition, plot_pce_sobol


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


ANALYSIS_LABELS = {
    "anova_1factor": "ANOVA à un facteur",
    "hsic_anova": "HSIC-ANOVA hiérarchique",
    "sobol_indices": "Sobol par PCE ordre 1 et total",
    "pdp_subspace": "PDP/ICE par sous-espace",
}

DEFAULT_ANALYSES = [
    "anova_1factor",
    "hsic_anova",
    "sobol_indices",
    "pdp_subspace",
]


def _normalise_analyses(analyses: list[str] | None) -> set[str]:
    selected = set(analyses or DEFAULT_ANALYSES)
    unknown = sorted(selected - set(ANALYSIS_LABELS))
    if unknown:
        raise ValueError(f"Analyses inconnues demandées : {unknown}")
    if not selected:
        raise ValueError("Sélectionner au moins une analyse à exécuter.")
    return selected


def _write_html_report(manifest: dict, output_dir: Path) -> Path:
    cards = []
    for target, artifacts in manifest["targets"].items():
        title = TARGET_LABELS.get(target, target)
        cards.append(f"<h2>{title}</h2>")
        for key, label in [
            ("anova_1factor_png", "ANOVA à un facteur"),
            ("pce_sobol_png", "Sobol par PCE ordre 1 et total"),
            ("hsic_anova_order_png", "Principaux effets HSIC-ANOVA"),
        ]:
            path = artifacts.get(key)
            if not path:
                continue
            rel = Path(path).relative_to(output_dir)
            cards.append(
                f'<section class="figure-card"><h3>{label}</h3>'
                f'<img src="{rel.as_posix()}" alt="{label} - {title}"></section>'
            )
        subspaces = [s for s in artifacts.get("pdp_subspace", []) if s.get("path")]
        if subspaces:
            cards.append('<h3 class="section-sub">PDP/ICE par sous-espace</h3>')
            for item in subspaces:
                rel = Path(item["path"]).relative_to(output_dir)
                cards.append(
                    f'<section class="figure-card"><h4>{item["title"]} '
                    f'· {item["n_points"]} sim. · Q²={item["q2"]}</h4>'
                    f'<img src="{rel.as_posix()}" alt="PDP ICE - {title} - {item["subspace"]}"></section>'
                )
    warnings = "".join(f"<li>{w}</li>" for w in manifest.get("warnings", []))
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rapport d'analyse de sensibilité MAELIA</title>
  <style>
    :root {{ color-scheme: light; --ink:#263238; --muted:#667085; --line:#E5E7EB; --paper:#FAFBFC; --accent:#2A9D8F; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:white; }}
    header {{ padding:40px clamp(24px,6vw,80px) 24px; border-bottom:1px solid var(--line); background:var(--paper); }}
    main {{ padding:28px clamp(24px,6vw,80px) 56px; }}
    h1 {{ margin:0 0 8px; font-size:clamp(28px,4vw,44px); }}
    h2 {{ margin:44px 0 18px; font-size:28px; }}
    h3 {{ margin:0 0 12px; font-size:18px; }}
    h3.section-sub {{ margin:30px 0 12px; color:var(--accent); }}
    h4 {{ margin:0 0 10px; font-size:15px; color:var(--muted); font-weight:600; }}
    p, li {{ color:var(--muted); line-height:1.55; }}
    code {{ background:#EEF2F7; padding:2px 6px; border-radius:4px; }}
    .figure-card {{ margin:18px 0 30px; padding:18px; border:1px solid var(--line); border-radius:8px; background:white; }}
    img {{ max-width:100%; height:auto; display:block; border-radius:6px; }}
    .meta {{ display:grid; gap:6px; margin-top:14px; }}
    .pill {{ display:inline-block; background:#E6F4F1; color:#17685E; padding:5px 9px; border-radius:999px; font-size:13px; }}
  </style>
</head>
<body>
<header>
  <span class="pill">MAELIA Sensitivity Analysis</span>
  <h1>Rapport d'analyse de sensibilité</h1>
  <p>Dataset : <code>{manifest['dataset_path']}</code></p>
  <div class="meta">
    <p>Lignes analysées : {manifest['n_rows']} · Paramètres : {manifest['n_features']} · Run : <code>{manifest['run_id']}</code></p>
  </div>
  {f'<ul>{warnings}</ul>' if warnings else ''}
</header>
<main>
  {''.join(cards)}
</main>
</body>
</html>"""
    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def run_analysis(
    log_dir: str | Path,
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    targets: list[str] | None = None,
    features: list[str] | None = None,
    analyses: list[str] | None = None,
    sample_size: int | None = None,
    n_bins: int = 4,
    sobol_n_mc: int = 2000,
    tree_max_depth: int = 4,
    random_state: int = 42,
) -> dict:
    analysis_set = _normalise_analyses(analyses)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    out = Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(log_dir=log_dir, dataset_path=dataset_path, targets=targets, features=features)
    df = bundle.dataframe

    # Sous-échantillonnage reproductible : le curseur "Simulations utilisées" plafonne le
    # nombre de points alimentant TOUTES les analyses (y compris les plafonds internes HSIC/PCE).
    n_available = int(len(df))
    if sample_size and sample_size < n_available:
        df = df.sample(n=int(sample_size), random_state=random_state).reset_index(drop=True)
    n_used = int(len(df))

    manifest: dict = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(Path(log_dir).expanduser()),
        "dataset_path": str(bundle.dataset_path),
        "output_dir": str(out),
        "n_rows": n_used,
        "n_rows_available": n_available,
        "sample_size": int(sample_size) if sample_size else None,
        "output_source": "dataset_metamodel",
        "n_features": int(len(bundle.feature_columns)),
        "features": bundle.feature_columns,
        "analyses": sorted(analysis_set),
        "analysis_labels": [ANALYSIS_LABELS[key] for key in sorted(analysis_set)],
        "targets": {},
        "warnings": bundle.warnings,
        "tables": {},
    }

    anova = pd.DataFrame()
    if "anova_1factor" in analysis_set:
        factor_frame = discretize_factors(
            df,
            bundle.feature_columns,
            bundle.categorical_columns,
            bundle.continuous_columns,
            n_bins=n_bins,
        )
        anova = one_factor_anova(df, factor_frame, bundle.feature_columns, bundle.target_columns)
        anova_path = out / "anova_1facteur.csv"
        anova.to_csv(anova_path, index=False)
        manifest["tables"]["anova_1factor"] = str(anova_path)

    pce_metric_rows = []
    hsic_metric_rows = []
    subspace_rows = []
    for target in bundle.target_columns:
        target_dir = out / _safe_name(target)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_artifacts: dict[str, str | dict | list] = {}

        if "anova_1factor" in analysis_set:
            p = plot_one_factor(anova, target, target_dir / f"anova_1facteur_{target}.png")
            target_artifacts["anova_1factor_png"] = str(p)

        if "sobol_indices" in analysis_set:
            try:
                pce_sobol, pce_metrics = train_sparse_pce_sobol(
                    df,
                    target,
                    bundle.feature_columns,
                    bundle.categorical_columns,
                    bundle.continuous_columns,
                    max_terms=60,
                    max_train=n_used,
                    random_state=random_state + 1000 + bundle.target_columns.index(target),
                )
                pce_path = target_dir / f"pce_sobol_indices_{target}.csv"
                pce_sobol.to_csv(pce_path, index=False)
                target_artifacts["pce_sobol_csv"] = str(pce_path)
                target_artifacts["pce_metrics"] = pce_metrics
                pce_metric_rows.append({"sortie": target, **pce_metrics})
                p = plot_pce_sobol(pce_sobol, target, target_dir / f"pce_sobol_indices_{target}.png")
                target_artifacts["pce_sobol_png"] = str(p)
            except Exception as exc:
                bundle.warnings.append(f"PCE creux indisponible pour {target}: {exc}")
                target_artifacts["pce_metrics"] = {"model_name": "SparsePCE", "status": "error", "error": str(exc)}

        if "hsic_anova" in analysis_set:
            try:
                hsic_terms, hsic_metrics = compute_hsic_anova(
                    df,
                    target,
                    bundle.feature_columns,
                    bundle.categorical_columns,
                    bundle.continuous_columns,
                    max_order=3,
                    max_samples=n_used,
                    random_state=random_state + 3000 + bundle.target_columns.index(target),
                )
                hsic_path = target_dir / f"hsic_anova_terms_{target}.csv"
                hsic_terms.to_csv(hsic_path, index=False)
                target_artifacts["hsic_anova_terms_csv"] = str(hsic_path)
                target_artifacts["hsic_anova_metrics"] = hsic_metrics
                hsic_metric_rows.append({"sortie": target, **hsic_metrics})
                p = plot_hsic_order_decomposition(hsic_terms, target, target_dir / f"hsic_anova_decomposition_{target}.png")
                target_artifacts["hsic_anova_order_png"] = str(p)
            except Exception as exc:
                bundle.warnings.append(f"HSIC-ANOVA indisponible pour {target}: {exc}")
                target_artifacts["hsic_anova_metrics"] = {"model_name": "HSIC-ANOVA", "status": "error", "error": str(exc)}

        if "pdp_subspace" in analysis_set:
            try:
                subspaces = compute_subspace_pdp_ice(
                    df,
                    target,
                    target_dir / "pdp_ice_sous_espaces",
                    random_state=random_state,
                )
                target_artifacts["pdp_subspace"] = subspaces
                for item in subspaces:
                    subspace_rows.append({"sortie": target, **{k: item[k] for k in
                        ("subspace", "title", "n_points", "n_active_features", "q2", "status")}})
            except Exception as exc:
                bundle.warnings.append(f"PDP/ICE par sous-espace indisponible pour {target}: {exc}")
                target_artifacts["pdp_subspace"] = []

        manifest["targets"][target] = target_artifacts

    if pce_metric_rows:
        pce_metrics_path = out / "pce_metamodel_metrics.csv"
        pd.DataFrame(pce_metric_rows).to_csv(pce_metrics_path, index=False)
        manifest["tables"]["pce_metamodel_metrics"] = str(pce_metrics_path)

    if hsic_metric_rows:
        hsic_metrics_path = out / "hsic_anova_metrics.csv"
        pd.DataFrame(hsic_metric_rows).to_csv(hsic_metrics_path, index=False)
        manifest["tables"]["hsic_anova_metrics"] = str(hsic_metrics_path)

    if subspace_rows:
        subspace_path = out / "pdp_ice_sous_espaces_synthese.csv"
        pd.DataFrame(subspace_rows).to_csv(subspace_path, index=False)
        manifest["tables"]["pdp_subspace_summary"] = str(subspace_path)

    report_path = _write_html_report(manifest, out)
    manifest["report_html"] = str(report_path)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_json"] = str(manifest_path)
    return manifest
