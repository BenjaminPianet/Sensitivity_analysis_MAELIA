from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from .models import prepare_X

TOOLS_DIR = Path(__file__).resolve().parents[1] / "analysis" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from hsic_methods import hsic_anova_hierarchical  # noqa: E402


DEFAULT_CONDITIONALLY_ACTING = {
    "nb_prepa", "Delta_PREPA_Semis", "Profondeur_Prepa_1", "Profondeur_Prepa_2",
    "Date_F1", "Dose_F1", "Date_F2", "Dose_F2", "Date_F3", "Dose_F3",
}


def _numeric_design_matrix(
    df: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    continuous: list[str],
) -> pd.DataFrame:
    X = prepare_X(df, features, categorical, continuous)
    numeric = pd.DataFrame(index=X.index)
    for feature in features:
        series = X[feature]
        as_numeric = pd.to_numeric(series, errors="coerce")
        if feature in categorical and as_numeric.notna().sum() < max(1, int(0.8 * series.notna().sum())):
            filled = series.astype("object").where(series.notna(), "inactif").astype(str)
            codes, _ = pd.factorize(filled, sort=True)
            numeric[feature] = codes.astype(float)
        else:
            numeric[feature] = as_numeric

    medians = numeric.median(numeric_only=True).fillna(0.0)
    numeric = numeric.fillna(medians)
    minima = numeric.min(axis=0)
    ranges = (numeric.max(axis=0) - minima).replace(0, 1.0)
    return ((numeric - minima) / ranges).clip(0.0, 1.0)


def _raw_numeric(df: pd.DataFrame, feature: str, default: float = 0.0) -> pd.Series:
    if feature not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    values = pd.to_numeric(df[feature], errors="coerce")
    if values.notna().any():
        return values.fillna(default).astype(float)
    lowered = df[feature].astype(str).str.strip().str.lower()
    mapped = lowered.map({
        "oui": 1.0, "yes": 1.0, "true": 1.0, "vrai": 1.0, "1": 1.0,
        "non": 0.0, "no": 0.0, "false": 0.0, "faux": 0.0, "0": 0.0,
        "inactif": 0.0, "nan": 0.0,
    })
    return mapped.fillna(default).astype(float)


def build_acting_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    acting = np.ones((len(df), len(features)), dtype=bool)
    index = {name: i for i, name in enumerate(features)}

    n_ferti = _raw_numeric(df, "n_ferti")
    nb_prepa = _raw_numeric(df, "nb_prepa")

    def set_active(feature: str, mask: pd.Series | np.ndarray) -> None:
        if feature in index:
            acting[:, index[feature]] = np.asarray(mask, dtype=bool)

    for feature in ["Delta_PREPA_Semis", "Profondeur_Prepa_1"]:
        set_active(feature, nb_prepa >= 1)
    set_active("Profondeur_Prepa_2", nb_prepa >= 2)

    for i in [1, 2, 3]:
        active = n_ferti >= i
        set_active(f"Date_F{i}", active)
        set_active(f"Dose_F{i}", active)

    return acting


def compute_hsic_anova(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    categorical: list[str],
    continuous: list[str],
    max_order: int = 3,
    max_samples: int = 1200,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    y = pd.to_numeric(df[target], errors="coerce")
    X_unit = _numeric_design_matrix(df, features, categorical, continuous)
    valid = y.notna() & X_unit.notna().all(axis=1)
    X_unit = X_unit.loc[valid]
    y = y.loc[valid]
    raw_df = df.loc[valid, features].copy()

    if len(X_unit) < 30:
        raise ValueError(f"Pas assez de points valides pour HSIC-ANOVA sur {target}.")
    if float(y.std()) <= 0.0:
        raise ValueError(f"La sortie {target} est constante; HSIC-ANOVA n'est pas informatif.")

    rng = np.random.default_rng(random_state)
    if len(X_unit) > max_samples:
        selected = np.sort(rng.choice(len(X_unit), size=max_samples, replace=False))
        X_unit = X_unit.iloc[selected]
        y = y.iloc[selected]
        raw_df = raw_df.iloc[selected]

    X_np = X_unit.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    x_is_acting = build_acting_matrix(raw_df, features)
    num_is_decreed = np.array([feature in DEFAULT_CONDITIONALLY_ACTING for feature in features], dtype=bool)
    is_categorical = np.array([feature in categorical for feature in features], dtype=bool)

    # ASTUCE : Arbres plus profonds pour capter les effets des sous-branches rares
    min_leaf = max(3, int(0.01 * len(X_np)))
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=min_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    # --- Gemini Trick: Random Marginal Imputation for RF ---
    # Resample from the empirical distribution of ACTIVE states.
    # This flawlessly handles continuous, ordinal, and categorical variables
    # (they retain their exact discrete values instead of continuous noise).
    X_rf = np.copy(X_np)
    for i in range(X_np.shape[1]):
        inactive_idx = np.where(~x_is_acting[:, i])[0]
        active_idx = np.where(x_is_acting[:, i])[0]
        if len(inactive_idx) > 0 and len(active_idx) > 0:
            X_rf[inactive_idx, i] = rng.choice(X_np[active_idx, i], size=len(inactive_idx))

    rf.fit(X_rf, y_np)
    
    # Conditional Permutation Importance (Intrinsic)
    n_features = X_np.shape[1]
    importances_mean = np.zeros(n_features)
    np.random.seed(random_state)
    
    for i in range(n_features):
        active_idx = np.where(x_is_acting[:, i])[0]
        if len(active_idx) < 2:
            importances_mean[i] = 0.0
            continue
        
        pred_base = rf.predict(X_np[active_idx])
        mse_base = np.mean((y_np[active_idx] - pred_base)**2)
        var_active = np.var(y_np[active_idx])
        
        mse_shuff = 0.0
        n_repeats = 6
        for _ in range(n_repeats):
            xt_shuff = np.copy(X_np)
            
            # Numpy advanced indexing returns a copy, so we must extract, shuffle, and re-assign
            shuffled_vals = np.copy(xt_shuff[active_idx, i])
            np.random.shuffle(shuffled_vals)
            xt_shuff[active_idx, i] = shuffled_vals
            
            pred_shuff = rf.predict(xt_shuff[active_idx])
            mse_shuff += np.mean((y_np[active_idx] - pred_shuff)**2)
        mse_shuff /= n_repeats
        
        if var_active > 1e-6:
            importances_mean[i] = max(0.0, (mse_shuff - mse_base) / var_active)
        else:
            importances_mean[i] = 0.0

    theta = importances_mean * 5.0
    theta[importances_mean < 0.001] = 0.0
    theta_scales = theta if np.any(theta > 0) else None

    filtered_results, global_hsic = hsic_anova_hierarchical(
        X=X_np,
        Y=y_np,
        x_is_acting=x_is_acting,
        num_is_decreed=num_is_decreed,
        is_categorical=is_categorical,
        theta_scales=theta_scales,
        var_names=features,
        max_order=max_order,
        use_smt_theta=theta_scales is not None,
        use_kta=False,
    )

    hsic_df = pd.DataFrame(filtered_results)
    if hsic_df.empty:
        hsic_df = pd.DataFrame(columns=[
            "sortie", "order", "variables", "contribution_hsic_globale_pct",
            "contribution_hsic_intrinseque_pct", "frequence_active", "trace", "adj_trace",
        ])
    else:
        hsic_df["sortie"] = target
        hsic_df["variables"] = hsic_df["combo"].apply(lambda combo: " & ".join(features[i] for i in combo))
        hsic_df["contribution_hsic_globale_pct"] = 100 * hsic_df["trace"] / global_hsic if global_hsic != 0 else np.nan
        adj_sum = hsic_df["adj_trace"].sum()
        hsic_df["contribution_hsic_intrinseque_pct"] = 100 * hsic_df["adj_trace"] / adj_sum if adj_sum != 0 else np.nan
        hsic_df["frequence_active"] = hsic_df["p_A"]
        hsic_df = hsic_df[[
            "sortie", "order", "variables", "contribution_hsic_globale_pct",
            "contribution_hsic_intrinseque_pct", "frequence_active", "trace", "adj_trace",
        ]].sort_values("contribution_hsic_globale_pct", ascending=False).reset_index(drop=True)

    metrics: dict[str, float | int | str] = {
        "model_name": "HSIC-ANOVA",
        "global_hsic": float(global_hsic),
        "n_samples": int(len(X_np)),
        "max_order": int(max_order),
        "n_terms": int(len(hsic_df)),
        "represented_global_pct": float(hsic_df["contribution_hsic_globale_pct"].sum()) if not hsic_df.empty else 0.0,
        "theta_source": "RF permutation" if theta_scales is not None else "median heuristic",
    }
    return hsic_df, metrics
