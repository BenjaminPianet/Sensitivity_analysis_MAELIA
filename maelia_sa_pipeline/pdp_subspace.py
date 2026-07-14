"""Courbes PDP/ICE par sous-espace de l'espace de conception hiérarchique SMT.

L'espace SMT est hiérarchique : deux variables de décision ordinales
(`n_ferti` in {0,1,2,3}, `nb_prepa` in {0,1,2}) définissent 12 sous-espaces. Dans
chaque sous-espace, ces variables sont fixées et seul un sous-ensemble des
variables continues est actif. On entraîne un métamodèle (forêt aléatoire) sur
les seules variables actives de chaque sous-espace, puis on trace la PDP moyenne
et le faisceau ICE pour chacune de ces variables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .config import FEATURE_LABELS, PALETTE, TARGET_LABELS

# Variables continues toujours actives, quel que soit le sous-espace.
_ALWAYS = ["Date_Semis", "Profondeur_Semis", "Date_Recolte"]

# Ordre stable des variables continues (aligné sur le plan SMT).
_CONTINUOUS_ORDER = [
    "Date_Semis", "Delta_PREPA_Semis", "Profondeur_Semis",
    "Profondeur_Prepa_1", "Profondeur_Prepa_2",
    "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
    "Dose_F1", "Dose_F2", "Dose_F3",
]


def active_continuous(n_ferti: int, nb_prepa: int) -> list[str]:
    """Variables continues actives pour un sous-espace donné."""
    feats = list(_ALWAYS)
    if nb_prepa >= 1:
        feats += ["Delta_PREPA_Semis", "Profondeur_Prepa_1"]
        if nb_prepa >= 2:
            feats += ["Profondeur_Prepa_2"]
    if n_ferti >= 1:
        feats += ["Date_F1", "Dose_F1"]
    if n_ferti >= 2:
        feats += ["Date_F2", "Dose_F2"]
    if n_ferti >= 3:
        feats += ["Date_F3", "Dose_F3"]
    return [f for f in _CONTINUOUS_ORDER if f in feats]


def subspace_label(n_ferti: int, nb_prepa: int) -> str:
    prep = "sansPrepa" if nb_prepa == 0 else f"prepa{nb_prepa}"
    return f"nferti{n_ferti}_{prep}"


def subspace_title(n_ferti: int, nb_prepa: int) -> str:
    if nb_prepa == 0:
        prep = "sans préparation"
    else:
        prep = f"préparation ({nb_prepa} reprise{'s' if nb_prepa == 2 else ''})"
    apports = "0 apport N" if n_ferti == 0 else f"{n_ferti} apport{'s' if n_ferti > 1 else ''} N"
    return f"{apports}, {prep}"


def enumerate_subspaces() -> list[tuple[int, int]]:
    """Les 12 sous-espaces valides (n_ferti, nb_prepa)."""
    subs: list[tuple[int, int]] = []
    for n_ferti in (0, 1, 2, 3):
        for nb_prepa in (0, 1, 2):
            subs.append((n_ferti, nb_prepa))
    return subs


def _decode_decision(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables de décision décodées (n_ferti, nb_prepa)."""
    out = df.copy()
    out["_n_ferti"] = pd.to_numeric(out["n_ferti"], errors="coerce").round().astype("Int64")
    out["_nb_prepa"] = pd.to_numeric(out["nb_prepa"], errors="coerce").round().astype("Int64")
    return out


def _pdp_ice_curve(model, X: pd.DataFrame, feature: str, grid_size=15, ice_sample=100, seed=42):
    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(X[feature], [0.02, 0.98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(X[feature].min()), float(X[feature].max())
    grid = np.linspace(lo, hi, grid_size)
    n = min(ice_sample, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_ref = X.iloc[idx].reset_index(drop=True)
    ice = np.empty((n, grid_size))
    for j, value in enumerate(grid):
        X_mod = X_ref.copy()
        X_mod[feature] = value
        ice[:, j] = model.predict(X_mod)
    return grid, ice, ice.mean(axis=0)


def _plot_subspace(sub, target, model, X, features, q2, n_points, path: Path) -> Path:
    ncol = min(3, len(features))
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow), squeeze=False)
    for k, feat in enumerate(features):
        ax = axes[k // ncol][k % ncol]
        grid, ice, pdp = _pdp_ice_curve(model, X, feat)
        for row in ice:
            ax.plot(grid, row, color=PALETTE["blue"], alpha=0.10, linewidth=0.8)
        ax.plot(grid, pdp, color=PALETTE["ink"], linewidth=2.6, label="PDP moyenne")
        ax.scatter(grid, pdp, color=PALETTE["ink"], s=14, zorder=3)
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=9)
        ax.set_ylabel(target, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
    for k in range(len(features), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(
        f"PDP/ICE — {TARGET_LABELS.get(target, target)}\n"
        f"Sous-espace : {subspace_title(*sub)}   |   {n_points} simulations   |   Q²(test) = {q2:.2f}",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def compute_subspace_pdp_ice(
    df: pd.DataFrame,
    target: str,
    out_dir: Path,
    random_state: int = 42,
    min_points: int = 40,
) -> list[dict]:
    """Génère une figure PDP/ICE par sous-espace pour une sortie donnée.

    Retourne la liste des sous-espaces avec leurs métadonnées et le chemin de la
    figure (ou None si le sous-espace a trop peu de points exploitables).
    """
    decoded = _decode_decision(df)
    results: list[dict] = []
    for sub in enumerate_subspaces():
        n_ferti, nb_prepa = sub
        sel = (decoded["_n_ferti"] == n_ferti) & (decoded["_nb_prepa"] == nb_prepa)
        sub_df = decoded[sel]
        feats = active_continuous(*sub)
        label = subspace_label(*sub)

        data = sub_df[feats + [target]].apply(pd.to_numeric, errors="coerce").dropna()
        entry = {
            "subspace": label,
            "title": subspace_title(*sub),
            "n_ferti": n_ferti,
            "nb_prepa": nb_prepa,
            "n_active_features": len(feats),
            "n_points": int(len(data)),
            "q2": None,
            "path": None,
            "status": "ok",
        }
        if len(data) < min_points:
            entry["status"] = "trop peu de points"
            results.append(entry)
            continue

        X, y = data[feats], data[target]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=random_state)
        model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=random_state, n_jobs=-1
        ).fit(X_tr, y_tr)
        q2 = float(r2_score(y_te, model.predict(X_te)))
        path = out_dir / f"pdp_ice_{label}_{target}.png"
        _plot_subspace(sub, target, model, X, feats, q2, len(data), path)
        entry["q2"] = round(q2, 3)
        entry["path"] = str(path)
        results.append(entry)
    return results
