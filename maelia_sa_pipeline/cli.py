from __future__ import annotations

import argparse
import json

from .pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Lance la pipeline d'analyse de sensibilité MAELIA.")
    parser.add_argument("--log-dir", required=True, help="Répertoire de logs MAELIA")
    parser.add_argument("--dataset-path", default=None, help="Chemin vers dataset_metamodel.csv")
    parser.add_argument("--output-dir", default=None, help="Dossier de sortie")
    parser.add_argument("--targets", nargs="*", default=None, help="Sorties à analyser")
    parser.add_argument("--n-bins", type=int, default=4, help="Nombre de classes ANOVA")
    parser.add_argument("--sobol-n-mc", type=int, default=2000, help="Paramètre conservé pour compatibilité API")
    parser.add_argument("--analyses", nargs="*", default=None, help="Analyses à exécuter : anova_1factor anova_2factor hsic_anova metamodel sobol_indices decision_tree pdp_ice")
    parser.add_argument("--tree-max-depth", type=int, default=4, help="Profondeur maximale des arbres")
    args = parser.parse_args()
    manifest = run_analysis(
        log_dir=args.log_dir,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        targets=args.targets,
        n_bins=args.n_bins,
        analyses=args.analyses,
        sobol_n_mc=args.sobol_n_mc,
        tree_max_depth=args.tree_max_depth,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
