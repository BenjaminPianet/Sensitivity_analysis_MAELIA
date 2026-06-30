from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

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
    date_semis_col = f"feat_{AGRI_FEATURES.index('Date_Semis')}"
    if date_semis_col not in df.columns:
        return
    date_semis = pd.to_numeric(df[date_semis_col], errors="coerce").dropna()
    if date_semis.empty:
        return
    vmin = float(date_semis.min())
    vmax = float(date_semis.max())
    if vmax > 150 or vmin < 35:
        raise ValueError(
            "Le dataset_metamodel.csv semble provenir de l'ancien plan d'expérience. "
            f"La colonne {date_semis_col}, interprétée comme Date_Semis, vaut [{vmin:.1f}, {vmax:.1f}], "
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
        feat_cols = [f"feat_{i}" for i in range(len(AGRI_FEATURES))]
        if all(col in df.columns for col in feat_cols):
            X = df[feat_cols].copy()
            X.columns = AGRI_FEATURES
            feature_columns = AGRI_FEATURES.copy()
            warnings.append(
                "Les colonnes feat_* ont été renommées avec des libellés agronomiques "
                "pour l'affichage. Les valeurs restent celles du plan SMT exporté."
            )
        elif all(col in df.columns for col in AGRI_FEATURES):
            feature_columns = AGRI_FEATURES.copy()
            X = df[feature_columns].copy()
        else:
            raise ValueError(
                "Le dataset doit contenir soit les colonnes feat_* du plan courant, soit les colonnes agronomiques nommées. "
                "Les logs MAELIA seuls ne suffisent pas : il faut le dataset exporté avec la matrice xt."
            )

    categorical = [c for c in feature_columns if c in AGRI_CATEGORICAL]
    continuous = [c for c in feature_columns if c not in categorical]
    return X, feature_columns, categorical, continuous, warnings



def parse_sim_idx(name: str) -> int | None:
    match = re.search(r"smt_(\d+)", name)
    return int(match.group(1)) if match else None


def first_dynamic_file(folder: Path) -> Path | None:
    matches = sorted(folder.glob("sorties_dynamiques_SA*.csv"))
    return matches[0] if matches else None


def load_dynamic_outputs(log_dir: str | Path, metadata: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load MAELIA daily outputs and align them with point_idx when possible."""
    warnings: list[str] = []
    log_path = Path(log_dir).expanduser()
    frames: list[pd.DataFrame] = []
    for folder in sorted(log_path.iterdir() if log_path.exists() else []):
        if not folder.is_dir():
            continue
        dynamic_file = first_dynamic_file(folder)
        sim_idx = parse_sim_idx(folder.name)
        if dynamic_file is None or sim_idx is None:
            continue
        df = pd.read_csv(dynamic_file, sep=";")
        df["sim_idx"] = sim_idx
        df["source_folder"] = folder.name
        frames.append(df)

    if not frames:
        return pd.DataFrame(), [
            f"Aucun fichier sorties_dynamiques_SA*.csv trouvé dans {log_path}. "
            "Les courbes temporelles et PDP/ICE dynamiques ne seront pas affichées."
        ]

    dyn = pd.concat(frames, ignore_index=True)
    rename = {}
    if "DeltaCorg" in dyn.columns:
        rename["DeltaCorg"] = "dCorg"
    if "Yield" in dyn.columns:
        rename["Yield"] = "rdt"
    dyn = dyn.rename(columns=rename)

    required = ["annee", "jour", "parcelle", "sim_idx"]
    missing = [c for c in required if c not in dyn.columns]
    if missing:
        return pd.DataFrame(), [f"Colonnes dynamiques manquantes: {missing}"]

    dyn["date"] = pd.to_datetime(
        dict(year=dyn["annee"].astype(int), month=1, day=1),
        errors="coerce",
    ) + pd.to_timedelta(dyn["jour"].astype(int) - 1, unit="D")
    dyn["time_index"] = (dyn["date"] - dyn["date"].min()).dt.days.astype(int) + 1

    if metadata is not None and {"point_idx", "parcelle"}.issubset(metadata.columns):
        meta_cols = ["point_idx", "parcelle"]
        meta = metadata.copy()
        if "sim_idx" not in meta.columns:
            clones_per_run = max(1, meta["parcelle"].nunique())
            meta["sim_idx"] = (pd.to_numeric(meta["point_idx"], errors="coerce") // clones_per_run).astype("Int64")
        meta = meta[["sim_idx", "parcelle", "point_idx"]].dropna().drop_duplicates(["sim_idx", "parcelle"])
        dyn = dyn.merge(meta, on=["sim_idx", "parcelle"], how="left")
        if dyn["point_idx"].isna().all():
            warnings.append("Les sorties dynamiques n'ont pas pu être alignées avec point_idx; affichage au niveau sim_idx.")
    else:
        warnings.append("Le dataset ne contient pas point_idx/parcelle; affichage temporel au niveau sim_idx seulement.")

    return dyn, warnings


def final_dynamic_dataset(
    dynamic: pd.DataFrame,
    metadata: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if dynamic.empty:
        return pd.DataFrame(), ["Aucune sortie dynamique disponible pour reconstruire les sorties finales."]
    if "point_idx" not in dynamic.columns or dynamic["point_idx"].isna().all():
        return pd.DataFrame(), ["Les sorties dynamiques ne sont pas alignées avec point_idx; impossible de reconstruire le dataset final dynamique."]
    missing_targets = [target for target in target_columns if target not in dynamic.columns]
    if missing_targets:
        return pd.DataFrame(), [f"Sorties absentes des logs dynamiques: {missing_targets}"]
    missing_features = [feature for feature in feature_columns if feature not in metadata.columns]
    if missing_features:
        return pd.DataFrame(), [f"Paramètres absents du dataset de référence: {missing_features}"]

    id_cols = [col for col in ["point_idx", "sim_idx", "parcelle", "simulation"] if col in metadata.columns]
    if "point_idx" not in id_cols:
        return pd.DataFrame(), ["Le dataset de référence ne contient pas point_idx; impossible de fusionner les sorties finales dynamiques."]

    final = metadata[id_cols + feature_columns].drop_duplicates("point_idx").copy()
    for target in target_columns:
        tmp = (
            dynamic[["point_idx", "time_index", target]]
            .replace([float("inf"), float("-inf")], pd.NA)
            .dropna(subset=["point_idx", "time_index", target])
            .sort_values(["point_idx", "time_index"])
            .groupby("point_idx", as_index=False)
            .tail(1)[["point_idx", target]]
        )
        final = final.merge(tmp, on="point_idx", how="inner")

    final = final.dropna(subset=target_columns).reset_index(drop=True)
    if final.empty:
        warnings.append("La reconstruction des sorties finales dynamiques a produit un dataset vide.")
    else:
        warnings.append(
            "Les sorties finales utilisées pour les métamodèles proviennent des derniers pas des logs dynamiques, "
            "afin d'aligner l'application avec le notebook PDP/ICE."
        )
    return final, warnings


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
