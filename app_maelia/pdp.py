"""Courbes PDP et faisceaux ICE, par sous-espace de l'espace hiérarchique.

Portage de ``maelia_sa_pipeline/pdp_subspace.py``, où les douze sous-espaces et la
liste des variables actives dans chacun étaient écrits en dur (``enumerate_subspaces``,
``active_continuous``). Les deux viennent maintenant de la spécification : les
sous-espaces sont le produit cartésien des fenêtres des méta-variables, et les
variables actives se dérivent de la hiérarchie.

Deux différences assumées avec la v1 :

  - **on renvoie les courbes, pas des images**. La v1 écrit des PNG ; ici la donnée
    est structurée, à charge de l'interface de la tracer. Cela évite matplotlib dans
    le chemin de requête et rend les courbes exploitables autrement qu'à l'œil.
  - **les catégorielles sont admises**. Une PDP sur une catégorielle n'est pas une
    courbe mais un effet par modalité : c'est ce qui donne l'effet du climat, à
    itinéraire moyen.

L'intérêt du découpage par sous-espace tient à ce que la structure de l'itinéraire y
est fixée : on y lit l'effet des réglages fins, sans que la variance due au nombre
d'apports ou de préparations vienne l'écraser.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .analysis import encode
from .space import CATEGORICAL, CONTINUOUS, SpaceSpec

MIN_POINTS = 40
GRID_SIZE = 15
ICE_SAMPLE = 60


def _subspace_label(combinaison: dict) -> str:
    return "_".join(f"{nom}{valeur}" for nom, valeur in sorted(combinaison.items()))


def _subspace_title(spec: SpaceSpec, combinaison: dict) -> str:
    morceaux = []
    for meta in spec.meta_variables:
        if meta.name not in combinaison:
            continue
        niveau = next((lv for lv in meta.levels if lv.value == combinaison[meta.name]), None)
        morceaux.append(niveau.label if niveau else f"{meta.name}={combinaison[meta.name]}")
    return ", ".join(morceaux)


def _curve(model, X: pd.DataFrame, feature: str, grid_size: int,
           ice_sample: int, rng) -> dict:
    """PDP moyenne et faisceau ICE d'une variable, par intervention sur la colonne.

    On remplace la colonne par chaque valeur de la grille et on repropage : c'est la
    définition même d'une PDP, et elle vaut aussi bien pour une catégorielle, dont la
    « grille » est l'ensemble de ses modalités observées.
    """
    valeurs = X[feature]
    distinctes = np.unique(valeurs.dropna())
    if len(distinctes) <= grid_size:
        grille = distinctes.astype(float)
    else:
        lo, hi = np.quantile(valeurs, [0.02, 0.98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(valeurs.min()), float(valeurs.max())
        grille = np.linspace(lo, hi, grid_size)

    n = min(ice_sample, len(X))
    echantillon = X.iloc[rng.choice(len(X), size=n, replace=False)].reset_index(drop=True)
    ice = np.empty((n, len(grille)))
    for j, valeur in enumerate(grille):
        modifie = echantillon.copy()
        modifie[feature] = valeur
        ice[:, j] = model.predict(modifie)

    return {
        "variable": feature,
        "grille": [round(float(v), 4) for v in grille],
        "pdp": [round(float(v), 4) for v in ice.mean(axis=0)],
        # Le faisceau ICE est tronqué : il sert à montrer la dispersion, pas à être lu
        # trajectoire par trajectoire.
        "ice": [[round(float(v), 4) for v in ligne] for ligne in ice[:20]],
        "amplitude": round(float(ice.mean(axis=0).max() - ice.mean(axis=0).min()), 4),
    }


def compute(spec: SpaceSpec, df: pd.DataFrame, target: str,
            min_points: int = MIN_POINTS, grid_size: int = GRID_SIZE,
            ice_sample: int = ICE_SAMPLE, random_state: int = 42) -> list[dict]:
    """Une entrée par sous-espace, avec ses courbes ou la raison de son absence."""
    rng = np.random.default_rng(random_state)
    resultats: list[dict] = []

    for combinaison in spec.subspaces():
        # Restreindre la spécification à ce sous-espace donne directement les
        # variables qui y sont actives : c'est la dérivation, pas une liste.
        restreint = spec.with_window(**{nom: [valeur] for nom, valeur in combinaison.items()})
        actives = [v for v in restreint.variables
                   if v.name in restreint.reachable() and v.name in df.columns
                   and v.kind in (CONTINUOUS, CATEGORICAL)]

        selection = pd.Series(True, index=df.index)
        for nom, valeur in combinaison.items():
            selection &= pd.to_numeric(df[nom], errors="coerce").round() == valeur
        sous = df[selection]

        entree = {
            "sous_espace": _subspace_label(combinaison),
            "titre": _subspace_title(spec, combinaison),
            "combinaison": combinaison,
            "n_points": int(len(sous)),
            "n_variables_actives": len(actives),
            "status": "ok",
            "q2": None,
            "courbes": [],
        }

        noms = [v.name for v in actives]
        donnees = encode(restreint, sous)[noms].copy() if noms else pd.DataFrame()
        y = pd.to_numeric(sous[target], errors="coerce") if len(sous) else pd.Series(dtype=float)
        if len(donnees):
            valides = donnees.notna().all(axis=1) & y.notna()
            donnees, y = donnees[valides], y[valides]

        if len(donnees) < min_points:
            entree["status"] = f"trop peu de points ({len(donnees)} pour {min_points})"
            resultats.append(entree)
            continue
        variables = [n for n in noms if donnees[n].nunique() > 1]
        if not variables:
            entree["status"] = "aucune variable ne varie dans ce sous-espace"
            resultats.append(entree)
            continue

        X, cible = donnees[variables], y
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, cible, test_size=0.25, random_state=random_state)
        modele = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3,
            random_state=random_state, n_jobs=-1).fit(X_tr, y_tr)

        entree["q2"] = round(float(r2_score(y_te, modele.predict(X_te))), 4)
        entree["n_variables_actives"] = len(variables)
        entree["courbes"] = [
            {**_curve(modele, X, nom, grid_size, ice_sample, rng),
             "nature": "catégorielle" if any(v.name == nom and v.kind == CATEGORICAL
                                             for v in actives) else "continue",
             "modalites": _modalites(spec, nom)}
            for nom in variables
        ]
        # L'amplitude de la PDP classe les variables par effet dans ce sous-espace.
        entree["courbes"].sort(key=lambda c: -c["amplitude"])
        resultats.append(entree)

    return resultats


def _modalites(spec: SpaceSpec, nom: str) -> list | None:
    """Étiquettes d'une catégorielle, pour rendre la grille encodée lisible."""
    for var in spec.variables:
        if var.name == nom and var.kind == CATEGORICAL:
            return list(var.domain)
    return None


def summary(resultats: list[dict]) -> dict:
    """Vue d'ensemble : ce qui a pu être calculé, et la variable la plus influente."""
    calcules = [r for r in resultats if r["status"] == "ok"]
    dominants = {r["sous_espace"]: r["courbes"][0]["variable"]
                 for r in calcules if r["courbes"]}
    return {
        "n_sous_espaces": len(resultats),
        "n_calcules": len(calcules),
        "n_ecartes": len(resultats) - len(calcules),
        "q2_min": round(min((r["q2"] for r in calcules if r["q2"] is not None), default=0.0), 4),
        "q2_max": round(max((r["q2"] for r in calcules if r["q2"] is not None), default=0.0), 4),
        "dominant_par_sous_espace": dominants,
        "dominants": sorted(set(dominants.values())),
        "dominant_stable": len(set(dominants.values())) <= 1,
    }
