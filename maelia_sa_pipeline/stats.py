from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def discretize_factors(
    df: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    continuous: list[str],
    n_bins: int = 4,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in features:
        if col in categorical:
            out[col] = df[col].astype("object").where(df[col].notna(), "inactif").astype(str)
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        unique = values.dropna().nunique()
        if unique <= 1:
            out[col] = "constant"
        elif unique <= n_bins:
            out[col] = values.astype("object").where(values.notna(), "inactif").astype(str)
        else:
            # Bin the observed values themselves, not their row ranks. Ranking with
            # method="first" splits identical inactive/default values across several
            # groups and creates artificial ANOVA effects in hierarchical SMT plans.
            try:
                bins = pd.qcut(values, q=min(n_bins, unique), duplicates="drop")
            except ValueError:
                bins = pd.cut(values, bins=min(n_bins, unique), duplicates="drop")

            categories = list(bins.cat.categories)
            labels = []
            for idx, interval in enumerate(categories, start=1):
                mask = bins == interval
                lo = values[mask].min()
                hi = values[mask].max()
                labels.append(f"Q{idx}: {lo:.6g} – {hi:.6g}")

            out[col] = pd.Categorical.from_codes(bins.cat.codes, categories=labels).astype("object")
            out[col] = out[col].where(out[col].notna(), "inactif").astype(str)
    return out


def explained_r2_by_group(y: pd.Series, groups: pd.Series) -> float:
    data = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "g": groups}).dropna(subset=["y", "g"])
    if data["g"].nunique() < 2:
        return 0.0
    overall = data["y"].mean()
    sst = float(((data["y"] - overall) ** 2).sum())
    if sst <= 0:
        return 0.0
    means = data.groupby("g")["y"].transform("mean")
    return max(0.0, min(1.0, float(((means - overall) ** 2).sum() / sst)))


def one_factor_anova(df: pd.DataFrame, factors: pd.DataFrame, features: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        y = pd.to_numeric(df[target], errors="coerce")
        for feature in features:
            groups = factors[feature]
            data = pd.DataFrame({"y": y, "g": groups}).dropna(subset=["y", "g"])
            grouped = [g["y"].to_numpy() for _, g in data.groupby("g") if len(g) >= 2]
            p_anova = np.nan
            p_kruskal = np.nan
            if len(grouped) >= 2:
                try:
                    p_anova = float(stats.f_oneway(*grouped).pvalue)
                except Exception:
                    p_anova = np.nan
                try:
                    p_kruskal = float(stats.kruskal(*grouped).pvalue)
                except Exception:
                    p_kruskal = np.nan
            rows.append({
                "sortie": target,
                "parametre": feature,
                "R2": explained_r2_by_group(y, groups),
                "p_anova": p_anova,
                "p_kruskal": p_kruskal,
                "n_groupes": int(data["g"].nunique()),
                "n": int(len(data)),
            })
    return pd.DataFrame(rows).sort_values(["sortie", "R2"], ascending=[True, False]).reset_index(drop=True)


def two_factor_interaction(df: pd.DataFrame, factors: pd.DataFrame, features: list[str], targets: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    matrices: dict[str, pd.DataFrame] = {}
    for target in targets:
        y = pd.to_numeric(df[target], errors="coerce")
        matrix = pd.DataFrame(0.0, index=features, columns=features)
        for a, b in itertools.combinations(features, 2):
            data = pd.DataFrame({"y": y, "a": factors[a], "b": factors[b]}).dropna(subset=["y", "a", "b"])
            if data["a"].nunique() < 2 or data["b"].nunique() < 2:
                continue
            overall = data["y"].mean()
            sst = float(((data["y"] - overall) ** 2).sum())
            if sst <= 0:
                continue

            pred_cell = data.groupby(["a", "b"])["y"].transform("mean")
            pred_a = data.groupby("a")["y"].transform("mean")
            pred_b = data.groupby("b")["y"].transform("mean")
            pred_add = pred_a + pred_b - overall

            r2_cell = 1.0 - float(((data["y"] - pred_cell) ** 2).sum() / sst)
            r2_add = 1.0 - float(((data["y"] - pred_add) ** 2).sum() / sst)
            r2_interaction = max(0.0, min(1.0, r2_cell - r2_add))
            matrix.loc[a, b] = matrix.loc[b, a] = r2_interaction
            rows.append({
                "sortie": target,
                "parametre_1": a,
                "parametre_2": b,
                "R2_interaction": r2_interaction,
                "R2_cellules": max(0.0, min(1.0, r2_cell)),
                "R2_additif": max(0.0, min(1.0, r2_add)),
                "n": int(len(data)),
                "n_cellules": int(data.groupby(["a", "b"]).ngroups),
            })
        matrices[target] = matrix
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["sortie", "R2_interaction"], ascending=[True, False]).reset_index(drop=True)
    return table, matrices
