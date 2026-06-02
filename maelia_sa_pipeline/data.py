from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import AGRI_CATEGORICAL, AGRI_FEATURES, DEFAULT_TARGETS


@dataclass
class DatasetBundle:
    dataframe: pd.DataFrame
    dataset_path: Path
    feature_columns: list[str]
    categorical_columns: list[str]
    continuous_columns: list[str]
    target_columns: list[str]
    warnings: list[str]


def _candidate_paths(log_dir: Path, dataset_path: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    if dataset_path:
        candidates.append(Path(dataset_path).expanduser())
    candidates.extend([
        log_dir / "dataset_metamodel.csv",
        log_dir / "dataset_metamodel_terrainSA.csv",
        log_dir.parent / "dataset_metamodel.csv",
        log_dir.parent / "dataset_metamodel_terrainSA.csv",
    ])
    return list(dict.fromkeys(candidates))


def count_log_runs(log_dir: Path) -> int:
    if not log_dir.exists():
        return 0
    return sum(1 for p in log_dir.iterdir() if p.is_dir() and (p / "sorties_CN.csv").exists())


def latest_log_mtime(log_dir: Path) -> float | None:
    if not log_dir.exists():
        return None
    mtimes = [p.stat().st_mtime for p in log_dir.iterdir() if p.is_dir() and (p / "sorties_CN.csv").exists()]
    return max(mtimes) if mtimes else None


def _validate_smt_feature_scale(df: pd.DataFrame, dataset_path: Path) -> None:
    if "feat_14" not in df.columns:
        return
    date_semis = pd.to_numeric(df["feat_14"], errors="coerce").dropna()
    if date_semis.empty:
        return
    vmin = float(date_semis.min())
    vmax = float(date_semis.max())
    if vmax > 150 or vmin < 35:
        raise ValueError(
            "Le dataset_metamodel.csv semble provenir de l'ancien plan d'expérience. "
            f"La colonne feat_14, maintenant interprétée comme Date_Semis, vaut [{vmin:.1f}, {vmax:.1f}], "
            "alors que le nouveau plan attend environ [45, 106] en jours de campagne. "
            "Régénère les fichiers dateDose, relance GAMA, puis ré-exporte dataset_metamodel.csv. "
            f"Dataset concerné : {dataset_path}"
        )


def infer_features(df: pd.DataFrame, requested_features: Iterable[str] | None = None) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    warnings: list[str] = []
    if requested_features:
        feature_columns = list(requested_features)
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Paramètres absents du dataset : {missing}")
        X = df[feature_columns].copy()
    else:
        feat_cols = [f"feat_{i}" for i in range(26)]
        if all(col in df.columns for col in feat_cols):
            X = df[feat_cols].copy()
            X.columns = AGRI_FEATURES
            feature_columns = AGRI_FEATURES.copy()
            warnings.append(
                "Les colonnes feat_0...feat_25 ont été renommées avec des libellés agronomiques "
                "pour l'affichage. Les valeurs restent celles du plan SMT exporté."
            )
        elif all(col in df.columns for col in AGRI_FEATURES):
            feature_columns = AGRI_FEATURES.copy()
            X = df[feature_columns].copy()
        else:
            raise ValueError(
                "Le dataset doit contenir soit feat_0...feat_25, soit les 26 colonnes agronomiques. "
                "Les logs MAELIA seuls ne suffisent pas : il faut le dataset exporté avec la matrice xt."
            )

    categorical = [c for c in feature_columns if c in AGRI_CATEGORICAL]
    continuous = [c for c in feature_columns if c not in categorical]
    return X, feature_columns, categorical, continuous, warnings


def load_dataset(
    log_dir: str | Path,
    dataset_path: str | Path | None = None,
    targets: Iterable[str] | None = None,
    features: Iterable[str] | None = None,
) -> DatasetBundle:
    log_path = Path(log_dir).expanduser()
    target_columns = list(targets or DEFAULT_TARGETS)

    tried = _candidate_paths(log_path, dataset_path)
    found_path = next((p for p in tried if p.exists()), None)
    if found_path is None:
        n_runs = count_log_runs(log_path)
        tried_text = "\n".join(f" - {p}" for p in tried)
        raise FileNotFoundError(
            "Aucun dataset_metamodel.csv n'a été trouvé.\n"
            f"Répertoire de logs : {log_path} ({n_runs} runs détectés).\n"
            "Pour calculer ANOVA, Sobol et arbres, il faut les paramètres du plan d'expérience, "
            "pas seulement les sorties MAELIA. Relance le notebook de simulation jusqu'à l'export "
            "du dataset, puis passe son chemin via dataset_path ou place-le dans le répertoire de logs.\n"
            f"Chemins testés :\n{tried_text}"
        )

    log_mtime = latest_log_mtime(log_path)
    if log_mtime is not None and found_path.stat().st_mtime + 60 < log_mtime:
        raise ValueError(
            "Le dataset_metamodel.csv est plus ancien que les derniers logs détectés. "
            "Il risque de ne pas correspondre aux simulations actuellement analysées. "
            "Relance la cellule d'export du dataset après la collecte des résultats. "
            f"Dataset : {found_path}"
        )

    df_raw = pd.read_csv(found_path)
    _validate_smt_feature_scale(df_raw, found_path)
    missing_targets = [c for c in target_columns if c not in df_raw.columns]
    if missing_targets:
        raise ValueError(f"Sorties absentes du dataset {found_path}: {missing_targets}")

    X, feature_columns, categorical, continuous, warnings = infer_features(df_raw, features)
    df = pd.concat([X, df_raw[target_columns]], axis=1)
    for extra in ["point_idx", "parcelle", "simulation", "sim_idx"]:
        if extra in df_raw.columns:
            df[extra] = df_raw[extra]

    df = df.dropna(subset=target_columns).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f"Dataset trop petit après nettoyage ({len(df)} lignes).")

    return DatasetBundle(
        dataframe=df,
        dataset_path=found_path,
        feature_columns=feature_columns,
        categorical_columns=categorical,
        continuous_columns=continuous,
        target_columns=target_columns,
        warnings=warnings,
    )
