"""Comparaison de métamodèles, pilotée par la spécification d'espace.

Portage de ``maelia_sa_pipeline/models.py``. Deux changements, et deux seulement :

  - les paramètres analysés viennent de la spécification, non d'``AGRI_FEATURES`` ;
  - les catégorielles sont encodées au lieu d'être éliminées par ``pd.to_numeric``.

Les familles de modèles, leurs hyperparamètres et la règle de sélection sont repris
tels quels : le score retenu est ``Q² − 0,05 × écart de surapprentissage``, ce qui
préfère un modèle un peu moins ajusté mais qui généralise mieux.

Comparer plusieurs familles n'est pas une coquetterie : cela dit si la relation
paramètres → sortie est robuste, ou l'artefact d'un modèle unique.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .analysis import encode
from .space import CATEGORICAL, SpaceSpec

MIN_POINTS = 40
PENALITE_SURAPPRENTISSAGE = 0.05


def analysable_features(spec: SpaceSpec, df: pd.DataFrame) -> list[str]:
    """Paramètres de l'espace présents dans les données et réellement variables."""
    noms = [n for n in spec.feature_names()
            if n in spec.reachable() or any(m.name == n for m in spec.meta_variables)]
    encoded = encode(spec, df)
    return [n for n in noms if n in encoded.columns and encoded[n].nunique(dropna=True) > 1]


def candidates(n_train: int, random_state: int) -> list[tuple[str, object]]:
    """Familles comparées. Identiques à celles de la v1, XGBoost si disponible."""
    min_leaf = max(5, int(0.01 * n_train))
    liste: list[tuple[str, object]] = [
        ("ExtraTrees", ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=min_leaf,
            random_state=random_state, n_jobs=-1)),
        ("RandomForest", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=min_leaf,
            random_state=random_state, n_jobs=-1)),
        ("HistGradientBoosting", HistGradientBoostingRegressor(
            max_iter=450, learning_rate=0.04, l2_regularization=0.05,
            min_samples_leaf=min_leaf, random_state=random_state)),
    ]
    try:
        from xgboost import XGBRegressor
        liste.append(("XGBoost", XGBRegressor(
            n_estimators=450, max_depth=4, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=random_state, n_jobs=-1)))
    except Exception:
        # XGBoost est optionnel : son absence retire une famille, pas l'analyse.
        pass
    return liste


def _metriques(y_train, pred_train, y_test, pred_test) -> dict:
    r2_train = float(r2_score(y_train, pred_train))
    q2_test = float(r2_score(y_test, pred_test))
    ecart = max(0.0, r2_train - q2_test)
    return {
        "R2_train": round(r2_train, 4),
        "Q2_test": round(q2_test, 4),
        "MAE_test": round(float(mean_absolute_error(y_test, pred_test)), 4),
        "RMSE_test": round(float(np.sqrt(mean_squared_error(y_test, pred_test))), 4),
        "overfit_gap": round(ecart, 4),
        "selection_score": round(q2_test - PENALITE_SURAPPRENTISSAGE * ecart, 4),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def compare(spec: SpaceSpec, df: pd.DataFrame, target: str,
            test_size: float = 0.25, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """Entraîne chaque famille sur le même partage et renvoie (comparaison, meilleur)."""
    features = analysable_features(spec, df)
    if not features:
        raise ValueError("Aucun paramètre ne varie dans cet espace : rien à modéliser.")

    frame = encode(spec, df)[features].copy()
    frame[target] = pd.to_numeric(df[target], errors="coerce")
    frame = frame.dropna()
    if len(frame) < MIN_POINTS:
        raise ValueError(f"Pas assez de points exploitables pour {target} "
                         f"({len(frame)} pour {MIN_POINTS} requis).")
    if float(frame[target].std()) <= 0.0:
        raise ValueError(f"La sortie {target} est constante ; aucun modèle n'apprendrait.")

    X, y = frame[features], frame[target]
    # Le même partage pour toutes les familles : sans cela, la comparaison mesurerait
    # la chance du découpage autant que la qualité du modèle.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    lignes = []
    modeles: dict[str, object] = {}
    for nom, modele in candidates(len(X_tr), random_state):
        try:
            modele.fit(X_tr, y_tr)
            entree = {"model": nom, "status": "ok",
                      **_metriques(y_tr, modele.predict(X_tr), y_te, modele.predict(X_te))}
            modeles[nom] = modele
        except Exception as exc:  # noqa: BLE001
            entree = {"model": nom, "status": "error", "error": str(exc)}
        lignes.append(entree)

    comparaison = pd.DataFrame(lignes)
    valides = comparaison[comparaison.status == "ok"]
    if valides.empty:
        erreurs = "; ".join(f"{r.model}: {r.error}" for r in comparaison.itertuples())
        raise ValueError(f"Aucun métamodèle n'a pu être entraîné. {erreurs}")

    meilleur = valides.sort_values("selection_score", ascending=False).iloc[0]
    nom = str(meilleur["model"])
    categorielles = {v.name for v in spec.variables if v.kind == CATEGORICAL}

    importances = []
    poids = getattr(modeles[nom], "feature_importances_", None)
    if poids is not None:
        importances = sorted(
            [{"parametre": f, "importance": round(float(w), 4)} for f, w in zip(features, poids)],
            key=lambda r: -r["importance"])

    resume = {
        "model_name": nom,
        "status": "ok",
        "n_features": len(features),
        "features": features,
        "categorielles": sorted(categorielles & set(features)),
        "n_candidats": int(len(valides)),
        "importances": importances,
        **{k: float(meilleur[k]) for k in
           ("R2_train", "Q2_test", "MAE_test", "RMSE_test", "overfit_gap", "selection_score")},
        "n_train": int(meilleur["n_train"]),
        "n_test": int(meilleur["n_test"]),
        # L'écart entre familles dit si la relation est robuste ou propre à un modèle.
        "ecart_Q2_familles": round(float(valides.Q2_test.max() - valides.Q2_test.min()), 4),
    }
    return comparaison.sort_values("selection_score", ascending=False).reset_index(drop=True), resume
