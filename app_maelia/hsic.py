"""HSIC-ANOVA hiérarchique, piloté par la spécification d'espace.

Portage de ``maelia_sa_pipeline/hsic.py`` où trois choses étaient codées en dur : la
liste des variables décrétées (``DEFAULT_CONDITIONALLY_ACTING``), les règles
d'activité (``build_acting_matrix``, avec ses conditions sur ``n_ferti`` et
``nb_prepa``), et l'absence de catégorielles. Les trois viennent désormais de la
spécification, dont l'équivalence avec les règles d'origine est prouvée sur les 5000
points historiques.

Le noyau de calcul, ``hsic_anova_hierarchical``, est celui de la v1, recopié tel quel
dans ``outils/`` pour que l'application soit autonome — il n'est pas modifié. Il prend
déjà ``x_is_acting`` en argument, ce qui rend le branchement direct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import encode
from .space import CATEGORICAL, SpaceSpec

from .outils.hsic_methods import hsic_anova_hierarchical

MIN_POINTS = 30
DEFAULT_MAX_SAMPLES = 1200


def unit_matrix(spec: SpaceSpec, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Matrice de conception ramenée dans [0, 1], catégorielles encodées.

    Les noyaux HSIC comparent des distances : sans mise à l'échelle commune, une dose
    en kilos pèserait mécaniquement plus qu'une profondeur en centimètres.

    Une catégorielle conditionnelle reçoit une modalité dédiée là où elle est inactive.
    Sans elle, « inactive » serait codée comme une modalité réelle et le noyau
    d'égalité les confondrait.
    """
    numeric = encode(spec, df)[features].copy()

    decretees = spec.decreed()
    categorielles = {v.name for v in spec.variables if v.kind == CATEGORICAL}
    conditionnelles = [f for f in features if f in categorielles and f in decretees]
    if conditionnelles:
        acting = spec.acting_matrix(df)
        for nom in conditionnelles:
            numeric.loc[~acting[nom].to_numpy(), nom] = spec.sentinel_inactive

    # Une valeur manquante est remplacée par la médiane plutôt que d'écarter la ligne :
    # les variables inactives portent la sentinelle, pas du vide.
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    minima = numeric.min(axis=0)
    etendues = (numeric.max(axis=0) - minima).replace(0, 1.0)
    return ((numeric - minima) / etendues).clip(0.0, 1.0)


def _largeurs_de_noyau(X: np.ndarray, y: np.ndarray, acting: np.ndarray,
                       rng: np.random.Generator, random_state: int) -> np.ndarray | None:
    """Largeurs de noyau apprises, plutôt que l'heuristique de la médiane.

    Deux obstacles se dressent devant HSIC sur un espace hiérarchique, et la v1 les
    contourne tous les deux ici. On reprend sa méthode telle quelle.

    **Le bouchon des variables inactives trahit la structure.** Une dose inactive porte
    toujours la même valeur : la forêt y lit « moins de deux apports » et crédite la
    dose de ce qui revient au nombre d'apports. Les variables structurelles deviennent
    redondantes, leur importance tombe à zéro, et elles disparaissent à tort du HSIC.
    On rééchantillonne donc chaque cellule inactive dans la distribution empirique des
    lignes **actives** de la même colonne, avant l'ajustement. Tirer dans les valeurs
    observées plutôt qu'un bruit continu préserve les modalités discrètes.

    **Une largeur commune n'est pas comparable d'une nature à l'autre.** Une
    catégorielle est mesurée par une distance d'égalité, une continue par une distance
    ordinaire : à largeur fixe, leurs HSIC ne sont pas sur la même échelle et la
    catégorielle est sous-estimée. La largeur de chaque variable est donc dérivée de
    son importance par permutation **conditionnelle** — chaque variable n'étant permutée
    que parmi les lignes où elle est active, la hausse d'erreur étant rapportée à la
    variance de ce sous-ensemble.

    Renvoie ``None`` si aucune variable ne ressort, auquel cas le noyau retombe sur
    l'heuristique de la médiane.
    """
    from sklearn.ensemble import RandomForestRegressor

    # Des arbres profonds, pour capter les effets des branches rares.
    foret = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=max(3, int(0.01 * len(X))),
        random_state=random_state, n_jobs=-1)

    X_ajuste = np.copy(X)
    for j in range(X.shape[1]):
        inactives = np.where(~acting[:, j])[0]
        actives = np.where(acting[:, j])[0]
        if len(inactives) and len(actives):
            X_ajuste[inactives, j] = rng.choice(X[actives, j], size=len(inactives))
    foret.fit(X_ajuste, y)

    # L'importance est mesurée sur la matrice brute : c'est l'espace réel.
    importances = np.zeros(X.shape[1])
    melange = np.random.default_rng(random_state)
    for j in range(X.shape[1]):
        actives = np.where(acting[:, j])[0]
        if len(actives) < 2:
            continue
        reference = np.mean((y[actives] - foret.predict(X[actives])) ** 2)
        variance = np.var(y[actives])
        if variance <= 1e-6:
            continue
        erreur = 0.0
        repetitions = 6
        for _ in range(repetitions):
            permute = np.copy(X)
            valeurs = np.copy(permute[actives, j])
            melange.shuffle(valeurs)
            permute[actives, j] = valeurs
            erreur += np.mean((y[actives] - foret.predict(permute[actives])) ** 2)
        importances[j] = max(0.0, (erreur / repetitions - reference) / variance)

    theta = importances * 5.0
    theta[importances < 0.001] = 0.0
    return theta if np.any(theta > 0) else None


def compute(spec: SpaceSpec, df: pd.DataFrame, target: str, max_order: int = 3,
            max_samples: int = DEFAULT_MAX_SAMPLES,
            random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """Décomposition HSIC-ANOVA de la dépendance entre paramètres et sortie."""
    features = [n for n in spec.feature_names()
                if n in spec.reachable() or any(m.name == n for m in spec.meta_variables)]
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("Aucun paramètre analysable dans ce jeu de données.")

    y = pd.to_numeric(df[target], errors="coerce")
    valides = y.notna()
    y = y[valides]
    brut = df.loc[valides, features]

    if len(y) < MIN_POINTS:
        raise ValueError(f"Pas assez de points valides pour HSIC-ANOVA sur {target} "
                         f"({len(y)} pour {MIN_POINTS} requis).")
    if float(y.std()) <= 0.0:
        raise ValueError(f"La sortie {target} est constante ; HSIC-ANOVA n'apprendrait rien.")

    rng = np.random.default_rng(random_state)
    if len(y) > max_samples:
        choisis = np.sort(rng.choice(len(y), size=max_samples, replace=False))
        y = y.iloc[choisis]
        brut = brut.iloc[choisis]

    X = unit_matrix(spec, brut, features).to_numpy(dtype=float)

    # Les trois vecteurs qui étaient codés en dur dans la v1, désormais dérivés.
    acting = spec.acting_matrix(brut)[features].to_numpy(dtype=bool)
    decreed = spec.decreed()
    est_decretee = np.array([f in decreed for f in features], dtype=bool)
    categorielles = {v.name for v in spec.variables if v.kind == CATEGORICAL}
    est_categorielle = np.array([f in categorielles for f in features], dtype=bool)

    theta = _largeurs_de_noyau(X, y.to_numpy(dtype=float), acting, rng, random_state)

    # Le noyau renvoie (termes retenus, HSIC global) et écrit un tableau sur la sortie
    # standard : on le tait, l'appelant décide de ce qu'il affiche.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        retenus, hsic_global = hsic_anova_hierarchical(
            X, y.to_numpy(dtype=float),
            x_is_acting=acting,
            num_is_decreed=est_decretee,
            is_categorical=est_categorielle,
            theta_scales=theta,
            var_names=features,
            max_order=max_order,
            use_smt_theta=theta is not None,
        )

    termes = _normaliser(retenus, features, float(hsic_global))
    par_ordre = termes.groupby("ordre")["part_globale"].sum().to_dict() if len(termes) else {}
    metriques = {
        "model_name": "HSIC-ANOVA",
        "status": "ok",
        "n_points": int(len(y)),
        "n_features": len(features),
        "n_termes": int(len(termes)),
        "max_order": max_order,
        "hsic_global": round(float(hsic_global), 6),
        "part_ordre_1": round(float(par_ordre.get(1, 0.0)), 4),
        "part_ordre_2": round(float(par_ordre.get(2, 0.0)), 4),
        "part_ordre_3": round(float(par_ordre.get(3, 0.0)), 4),
        "part_representee": round(float(termes["part_globale"].sum()) if len(termes) else 0.0, 4),
        "categorielles": sorted(categorielles & set(features)),
        "decretees": sorted(set(decreed) & set(features)),
    }
    return termes, metriques


COLONNES = ["variables", "ordre", "part_globale", "part_intrinseque",
            "frequence_active", "trace", "adj_trace"]


def _normaliser(retenus: list[dict], features: list[str], hsic_global: float) -> pd.DataFrame:
    """Tableau des termes, aux conventions de la v1.

    Deux parts coexistent, et elles ne disent pas la même chose :
      - **globale** : contribution au HSIC total, donc à la dépendance observée ;
      - **intrinsèque** : contribution rapportée aux seules lignes où le terme est
        actif. Une variable rarement active peut peser peu globalement tout en
        gouvernant fortement son sous-espace.
    """
    frame = pd.DataFrame(retenus)
    if frame.empty:
        return pd.DataFrame(columns=COLONNES)

    frame["variables"] = frame["combo"].apply(
        lambda combo: " & ".join(features[int(i)] for i in combo))
    frame["ordre"] = frame["order"].astype(int)
    frame["part_globale"] = frame["trace"] / hsic_global if hsic_global else 0.0
    somme_ajustee = float(frame["adj_trace"].sum())
    frame["part_intrinseque"] = frame["adj_trace"] / somme_ajustee if somme_ajustee else 0.0
    frame["frequence_active"] = frame["p_A"]
    return (frame[COLONNES].sort_values("part_globale", ascending=False)
            .reset_index(drop=True))
