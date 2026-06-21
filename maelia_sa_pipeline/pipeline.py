from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .config import DEFAULT_OUTPUT_ROOT, DEFAULT_PDP_ICE_FEATURES, FEATURE_LABELS, TARGET_LABELS
from .data import final_dynamic_dataset, load_dataset, load_dynamic_outputs
from .models import metamodel_feature_importance, select_best_metamodel, sobol_total_from_metamodel, train_decision_tree, train_sparse_pce_sobol
from .stats import discretize_factors, one_factor_anova, two_factor_interaction
from .viz import plot_interaction_heatmap, plot_metamodel_performance, plot_one_factor, plot_pce_sobol, plot_pdp_ice, plot_regions, plot_sobol_total, plot_temporal_trajectories, plot_tree_figure


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)




ANALYSIS_LABELS = {
    "temporal": "Évolution temporelle",
    "anova_1factor": "ANOVA à un facteur",
    "anova_2factor": "ANOVA à deux facteurs",
    "pce_sobol": "Sobol analytique par PCE",
    "metamodel": "Comparaison / sélection du métamodèle",
    "sobol_empirical": "Sobol total empirique",
    "pdp_ice": "PDP/ICE finales",
    "decision_tree": "Régions par arbre de décision",
}

DEFAULT_ANALYSES = [
    "anova_1factor",
    "anova_2factor",
    "pce_sobol",
    "decision_tree",
]

METAMODEL_DEPENDENT_ANALYSES = {"metamodel", "sobol_empirical", "pdp_ice"}


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
            ("temporal_trajectory_png", "Évolution temporelle observée"),
            ("anova_1factor_png", "ANOVA à un facteur"),
            ("anova_2factor_interaction_png", "Interactions à deux facteurs"),
            ("metamodel_performance_png", "Performance du métamodèle"),
            ("pce_sobol_png", "Sobol analytique par PCE"),
            ("sobol_total_png", "Sobol total empirique"),
            ("decision_tree_regions_png", "Régions sensibles"),
            ("decision_tree_png", "Arbre de décision"),
        ]:
            path = artifacts.get(key)
            if not path:
                continue
            rel = Path(path).relative_to(output_dir)
            cards.append(
                f'<section class="figure-card"><h3>{label}</h3>'
                f'<img src="{rel.as_posix()}" alt="{label} - {title}"></section>'
            )
        for item in artifacts.get("pdp_ice_pngs", []):
            path = item.get("path")
            if not path:
                continue
            rel = Path(path).relative_to(output_dir)
            feature = FEATURE_LABELS.get(item.get("feature", ""), item.get("feature", ""))
            cards.append(
                f'<section class="figure-card"><h3>PDP/ICE finale — {feature}</h3>'
                f'<img src="{rel.as_posix()}" alt="PDP ICE - {title} - {feature}"></section>'
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
    n_bins: int = 4,
    sobol_n_mc: int = 2000,
    tree_max_depth: int = 4,
    random_state: int = 42,
) -> dict:
    analysis_set = _normalise_analyses(analyses)
    model_needed = bool(analysis_set & METAMODEL_DEPENDENT_ANALYSES)
    dynamic_needed = "temporal" in analysis_set

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    out = Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_ROOT / run_id
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(log_dir=log_dir, dataset_path=dataset_path, targets=targets, features=features)
    source_df = bundle.dataframe
    dynamic_df = pd.DataFrame()
    dynamic_final_df = pd.DataFrame()

    if dynamic_needed:
        dynamic_df, dynamic_warnings = load_dynamic_outputs(log_dir=log_dir, metadata=source_df)
        bundle.warnings.extend(dynamic_warnings)
        dynamic_final_df, dynamic_final_warnings = final_dynamic_dataset(
            dynamic_df,
            source_df,
            bundle.feature_columns,
            bundle.target_columns,
        )
        bundle.warnings.extend(dynamic_final_warnings)

    df = dynamic_final_df if not dynamic_final_df.empty else source_df

    manifest: dict = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "log_dir": str(Path(log_dir).expanduser()),
        "dataset_path": str(bundle.dataset_path),
        "output_dir": str(out),
        "n_rows": int(len(df)),
        "output_source": "dynamic_final_logs" if not dynamic_final_df.empty else "dataset_metamodel",
        "n_features": int(len(bundle.feature_columns)),
        "features": bundle.feature_columns,
        "analyses": sorted(analysis_set),
        "analysis_labels": [ANALYSIS_LABELS[key] for key in sorted(analysis_set)],
        "targets": {},
        "warnings": bundle.warnings,
        "tables": {},
    }

    factor_frame = None
    anova = pd.DataFrame()
    matrices: dict[str, pd.DataFrame] = {}
    if analysis_set & {"anova_1factor", "anova_2factor"}:
        factor_frame = discretize_factors(
            df,
            bundle.feature_columns,
            bundle.categorical_columns,
            bundle.continuous_columns,
            n_bins=n_bins,
        )

    if "anova_1factor" in analysis_set:
        anova = one_factor_anova(df, factor_frame, bundle.feature_columns, bundle.target_columns)
        anova_path = out / "anova_1facteur.csv"
        anova.to_csv(anova_path, index=False)
        manifest["tables"]["anova_1factor"] = str(anova_path)

    if "anova_2factor" in analysis_set:
        interactions, matrices = two_factor_interaction(df, factor_frame, bundle.feature_columns, bundle.target_columns)
        interactions_path = out / "anova_2facteurs_interactions.csv"
        interactions.to_csv(interactions_path, index=False)
        manifest["tables"]["anova_2factor_interactions"] = str(interactions_path)

    metamodel_rows = []
    pce_metric_rows = []
    for target in bundle.target_columns:
        target_dir = out / _safe_name(target)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_artifacts: dict[str, str | dict | list] = {}

        if dynamic_needed and not dynamic_df.empty and target in dynamic_df.columns:
            p = plot_temporal_trajectories(
                dynamic_df,
                target,
                target_dir / f"trajectoires_temporelles_{target}.png",
                group_col="point_idx",
                random_state=random_state,
            )
            target_artifacts["temporal_trajectory_png"] = str(p)

        if "anova_1factor" in analysis_set:
            p = plot_one_factor(anova, target, target_dir / f"anova_1facteur_{target}.png")
            target_artifacts["anova_1factor_png"] = str(p)

        if "anova_2factor" in analysis_set:
            matrix = matrices[target]
            matrix_path = target_dir / f"anova_2facteurs_R2_interaction_{target}.csv"
            matrix.to_csv(matrix_path)
            target_artifacts["anova_2factor_interaction_csv"] = str(matrix_path)
            p = plot_interaction_heatmap(matrix, target, target_dir / f"anova_2facteurs_R2_interaction_{target}.png")
            target_artifacts["anova_2factor_interaction_png"] = str(p)

        model = None
        if model_needed:
            model, model_metrics, model_comparison = select_best_metamodel(
                df,
                target,
                bundle.feature_columns,
                bundle.categorical_columns,
                bundle.continuous_columns,
                random_state=random_state,
            )
            comparison_path = target_dir / f"metamodel_selection_{target}.csv"
            model_comparison.to_csv(comparison_path, index=False)
            target_artifacts["metamodel_selection_csv"] = str(comparison_path)
            model_metrics_row = {"sortie": target, **model_metrics}
            metamodel_rows.append(model_metrics_row)
            target_artifacts["metamodel_metrics"] = model_metrics
            p = plot_metamodel_performance(model_metrics, target, target_dir / f"metamodel_performance_{target}.png")
            target_artifacts["metamodel_performance_png"] = str(p)

            importances = metamodel_feature_importance(model, bundle.categorical_columns)
            importances_path = target_dir / f"metamodel_importances_{target}.csv"
            importances.to_csv(importances_path, index=False)
            target_artifacts["metamodel_importances_csv"] = str(importances_path)

        if "pce_sobol" in analysis_set:
            try:
                pce_sobol, pce_metrics = train_sparse_pce_sobol(
                    df,
                    target,
                    bundle.feature_columns,
                    bundle.categorical_columns,
                    bundle.continuous_columns,
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
                warning = f"PCE creux indisponible pour {target}: {exc}"
                bundle.warnings.append(warning)
                target_artifacts["pce_metrics"] = {"model_name": "SparsePCE", "status": "error", "error": str(exc)}

        if "pdp_ice" in analysis_set and model is not None:
            pdp_features = [
                feature for feature in DEFAULT_PDP_ICE_FEATURES
                if feature in bundle.feature_columns and feature in bundle.continuous_columns
            ]
            pdp_ice_pngs = []
            for feature in pdp_features:
                p = plot_pdp_ice(
                    model,
                    df,
                    target,
                    feature,
                    bundle.feature_columns,
                    bundle.categorical_columns,
                    bundle.continuous_columns,
                    target_dir / f"pdp_ice_final_{target}_{_safe_name(feature)}.png",
                    random_state=random_state,
                )
                pdp_ice_pngs.append({"feature": feature, "feature_label": FEATURE_LABELS.get(feature, feature), "path": str(p)})
            target_artifacts["pdp_ice_pngs"] = pdp_ice_pngs

        if "sobol_empirical" in analysis_set and model is not None:
            sobol = sobol_total_from_metamodel(
                model,
                df,
                bundle.feature_columns,
                bundle.categorical_columns,
                bundle.continuous_columns,
                target,
                n_mc=sobol_n_mc,
                random_state=random_state,
            )
            sobol_path = target_dir / f"sobol_total_{target}.csv"
            sobol.to_csv(sobol_path, index=False)
            target_artifacts["sobol_total_csv"] = str(sobol_path)
            p = plot_sobol_total(sobol, target, target_dir / f"sobol_total_{target}.png")
            target_artifacts["sobol_total_png"] = str(p)

        if "decision_tree" in analysis_set:
            tree_result = train_decision_tree(
                df,
                target,
                bundle.feature_columns,
                bundle.categorical_columns,
                bundle.continuous_columns,
                max_depth=tree_max_depth,
                random_state=random_state,
            )
            regions_path = target_dir / f"decision_tree_regions_{target}.csv"
            tree_result.regions.to_csv(regions_path, index=False)
            target_artifacts["decision_tree_regions_csv"] = str(regions_path)
            rules_path = target_dir / f"decision_tree_rules_{target}.txt"
            rules_path.write_text(tree_result.rules_text, encoding="utf-8")
            target_artifacts["decision_tree_rules_txt"] = str(rules_path)
            target_artifacts["decision_tree_metrics"] = tree_result.metrics
            p = plot_regions(tree_result.regions, target, target_dir / f"decision_tree_regions_{target}.png")
            target_artifacts["decision_tree_regions_png"] = str(p)
            p = plot_tree_figure(tree_result, target, target_dir / f"decision_tree_{target}.png")
            target_artifacts["decision_tree_png"] = str(p)

        manifest["targets"][target] = target_artifacts

    if metamodel_rows:
        metamodel_path = out / "metamodel_metrics.csv"
        pd.DataFrame(metamodel_rows).to_csv(metamodel_path, index=False)
        manifest["tables"]["metamodel_metrics"] = str(metamodel_path)

    if pce_metric_rows:
        pce_metrics_path = out / "pce_metamodel_metrics.csv"
        pd.DataFrame(pce_metric_rows).to_csv(pce_metrics_path, index=False)
        manifest["tables"]["pce_metamodel_metrics"] = str(pce_metrics_path)

    report_path = _write_html_report(manifest, out)
    manifest["report_html"] = str(report_path)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_json"] = str(manifest_path)
    return manifest
