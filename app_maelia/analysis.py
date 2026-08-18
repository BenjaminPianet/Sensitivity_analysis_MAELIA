"""Couverture et analyse minimale sur l'espace sélectionné.

L'analyse est volontairement réduite : un R² descriptif par paramètre et un Q² de
métamodèle. L'objet de cette version est la **sélection de l'espace**, pas la richesse
des analyses — celles de la v1 viendront au lot 2.

La couverture, elle, est traitée sérieusement : c'est elle qui décide de ce qui est
calculable, et le point de conception qui distingue cette version.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .space import CATEGORICAL, CONTINUOUS, SpaceSpec

# Seuils. Le premier est celui déjà en vigueur dans la v1 (pdp_subspace.min_points) ;
# les autres sont des valeurs de travail, à calibrer.
MIN_POINTS_PER_SUBSPACE = 40
MIN_POINTS_METAMODEL = 300
MIN_POINTS_ANOVA_PER_BIN = 30
N_BINS = 4


@dataclass
class Verdict:
    name: str
    label: str
    ok: bool
    reason: str


@dataclass
class Coverage:
    n_points: int
    n_available: int
    subspaces: list[dict]
    verdicts: list[Verdict]
    reachable: list[str] = field(default_factory=list)
    unconditional: list[str] = field(default_factory=list)
    decreed: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)

    def allows(self, name: str) -> bool:
        return any(v.ok for v in self.verdicts if v.name == name)


def coverage(spec: SpaceSpec, df: pd.DataFrame) -> Coverage:
    """Mesure ce que les données disponibles permettent sous l'espace demandé."""
    table = spec.coverage(df)
    n_points = int(table.n_points.sum()) if len(table) else 0
    subspaces = table.to_dict("records") if len(table) else []

    starved = [s for s in subspaces if s["n_points"] < MIN_POINTS_PER_SUBSPACE]
    verdicts = [
        Verdict(
            "metamodel", "Métamodèle (Q²)",
            n_points >= MIN_POINTS_METAMODEL,
            "" if n_points >= MIN_POINTS_METAMODEL
            else f"{n_points} points pour {MIN_POINTS_METAMODEL} requis — Q² peu fiable",
        ),
        Verdict(
            "anova", "R² par paramètre",
            n_points >= MIN_POINTS_ANOVA_PER_BIN * N_BINS,
            "" if n_points >= MIN_POINTS_ANOVA_PER_BIN * N_BINS
            else f"{n_points} points pour {MIN_POINTS_ANOVA_PER_BIN * N_BINS} requis "
                 f"({MIN_POINTS_ANOVA_PER_BIN} par classe sur {N_BINS} classes)",
        ),
        Verdict(
            "subspace", "Analyse par sous-espace",
            not starved and bool(subspaces),
            "" if not starved and subspaces
            else f"{len(starved)} sous-espace(s) sous {MIN_POINTS_PER_SUBSPACE} points",
        ),
    ]

    reachable = spec.reachable()
    unconditional = spec.unconditional()
    return Coverage(
        n_points=n_points,
        n_available=int(len(df)),
        subspaces=subspaces,
        verdicts=verdicts,
        reachable=sorted(reachable),
        unconditional=sorted(unconditional & reachable),
        decreed=sorted(spec.decreed()),
        unreachable=sorted({v.name for v in spec.variables} - reachable),
    )


def encode(spec: SpaceSpec, df: pd.DataFrame) -> pd.DataFrame:
    """Copie numérique du jeu de données, catégorielles encodées.

    Une catégorielle — le climat, l'espèce — n'est pas un nombre : la passer à
    ``pd.to_numeric`` la transforme en NaN et l'élimine silencieusement de l'analyse.
    On l'encode donc par son rang dans le **domaine de la spécification**, et non par
    l'ordre d'apparition dans les données : le code d'une modalité reste ainsi le même
    d'un jeu de données à l'autre, ce qui rend les analyses comparables.

    L'encodage est ordinal alors que la variable ne l'est pas. C'est sans conséquence
    pour un modèle à arbres, qui partitionne sans supposer d'ordre, mais il faudrait
    un encodage disjonctif pour un modèle linéaire.
    """
    out = pd.DataFrame(index=df.index)
    for var in spec.variables:
        if var.name not in df.columns:
            continue
        if var.kind == CATEGORICAL:
            codes = {modalite: rang for rang, modalite in enumerate(var.domain)}
            out[var.name] = df[var.name].astype(str).map(codes).astype("float")
        else:
            out[var.name] = pd.to_numeric(df[var.name], errors="coerce")
    for meta in spec.meta_variables:
        if meta.name in df.columns:
            out[meta.name] = pd.to_numeric(df[meta.name], errors="coerce")
    return out


def _is_categorical(spec: SpaceSpec, name: str) -> bool:
    return any(v.name == name and v.kind == CATEGORICAL for v in spec.variables)


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Part de variance expliquée par les groupes (R² descriptif à un facteur)."""
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for _, idx in groups.groupby(groups, observed=True).groups.items():
        subset = values.loc[idx]
        if len(subset):
            ss_between += len(subset) * (subset.mean() - grand_mean) ** 2
    return max(0.0, min(1.0, ss_between / ss_total))


def one_factor(spec: SpaceSpec, df: pd.DataFrame, target: str) -> list[dict]:
    """R² descriptif de chaque paramètre analysable, calculé sur ses lignes actives."""
    acting = spec.acting_matrix(df)
    reachable = spec.reachable()
    rows = []

    for name in spec.feature_names():
        is_meta = any(m.name == name for m in spec.meta_variables)
        if not is_meta and name not in reachable:
            continue

        active = acting[name].to_numpy()
        values = pd.to_numeric(df[target], errors="coerce")[active]
        categorielle = _is_categorical(spec, name)
        # Une catégorielle est déjà un facteur : ses modalités sont les groupes, il n'y
        # a ni conversion numérique ni discrétisation à faire.
        factor = (df[name].astype(str)[active] if categorielle
                  else pd.to_numeric(df[name], errors="coerce")[active])
        usable = values.notna() & factor.notna()
        values, factor = values[usable], factor[usable]
        if len(values) < 8 or factor.nunique() < 2:
            continue

        if categorielle or is_meta or factor.nunique() <= N_BINS:
            groups = factor.astype(str)
        else:
            groups = pd.qcut(factor.rank(method="first"), N_BINS, labels=False).astype(str)

        rows.append({
            "parametre": name,
            "r2": round(_eta_squared(values, groups), 4),
            "n_points": int(len(values)),
            "statut": "inconditionnelle" if is_meta or name in spec.unconditional() else "décrétée",
            "nature": "catégorielle" if categorielle else ("méta" if is_meta else "continue"),
            "modalites": int(factor.nunique()) if categorielle else None,
        })

    return sorted(rows, key=lambda r: r["r2"], reverse=True)


def metamodel(spec: SpaceSpec, df: pd.DataFrame, target: str, random_state: int = 42) -> dict:
    """Q² d'une forêt aléatoire sur les paramètres atteignables.

    N'est plus une analyse offerte à l'écran : les quatre familles comparées mesurent
    déjà la prévisibilité, et deux Q² pour un même nom de modèle se lisaient comme une
    incohérence. La fonction demeure pour ``by_stratum`` et pour l'usage en
    bibliothèque.

    Les valeurs inactives conservent la sentinelle -1 : un modèle à arbres la traite
    comme une modalité à part, ce qui suffit pour cette version d'exemple. Le
    traitement hiérarchique complet (imputation marginale) est celui de la v1.
    """
    features = [n for n in spec.feature_names()
                if n in spec.reachable() or any(m.name == n for m in spec.meta_variables)]
    encoded = encode(spec, df)
    frame = encoded[[f for f in features if f in encoded.columns]].copy()
    frame[target] = pd.to_numeric(df[target], errors="coerce")
    frame = frame.dropna()
    if len(frame) < 20:
        return {"status": "error", "error": f"trop peu de points exploitables ({len(frame)})"}

    # Un paramètre sans variation (figé, ou constant dans ce sous-espace) n'apporte rien.
    varying = [f for f in features if frame[f].nunique() > 1]
    if not varying:
        return {"status": "error", "error": "aucun paramètre ne varie dans cet espace"}

    X, y = frame[varying], frame[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=random_state)
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=random_state, n_jobs=-1
    ).fit(X_tr, y_tr)

    return {
        "status": "ok",
        "model": "RandomForest",
        "R2_train": round(float(r2_score(y_tr, model.predict(X_tr))), 4),
        "Q2_test": round(float(r2_score(y_te, model.predict(X_te))), 4),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "features": varying,
        "categorielles": [f for f in varying if _is_categorical(spec, f)],
        "importances": sorted(
            [{"parametre": f, "importance": round(float(w), 4)}
             for f, w in zip(varying, model.feature_importances_)],
            key=lambda r: r["importance"], reverse=True,
        ),
    }


def _compare_models(spec: SpaceSpec, df: pd.DataFrame, target: str) -> dict:
    """Comparaison de familles de métamodèles ; une erreur est remontée, pas tue."""
    from . import models

    try:
        comparaison, resume = models.compare(spec, df, target)
        resume["scores"] = comparaison.replace({np.nan: None}).to_dict("records")
        return resume
    except (ValueError, ImportError) as exc:
        return {"model_name": "métamodèles", "status": "error", "error": str(exc)}


def _hsic(spec: SpaceSpec, df: pd.DataFrame, target: str) -> dict:
    """Décomposition HSIC-ANOVA ; les termes sont tronqués pour rester lisibles."""
    from . import hsic as hsic_module

    try:
        termes, metriques = hsic_module.compute(spec, df, target)
        metriques["termes"] = termes.head(20).replace({np.nan: None}).to_dict("records")
        return metriques
    except (ValueError, ImportError) as exc:
        return {"model_name": "HSIC-ANOVA", "status": "error", "error": str(exc)}


def _pdp(spec: SpaceSpec, df: pd.DataFrame, target: str) -> dict:
    """Courbes PDP/ICE par sous-espace ; une erreur est remontée, pas tue."""
    from . import pdp as pdp_module

    try:
        resultats = pdp_module.compute(spec, df, target)
        return {"status": "ok", "sous_espaces": resultats,
                **pdp_module.summary(resultats)}
    except (ValueError, KeyError) as exc:
        return {"status": "error", "error": str(exc)}


def grouping_values(spec: SpaceSpec, df: pd.DataFrame, group: str) -> list:
    """Modalités d'un regroupement, dans l'ordre du domaine quand il est déclaré.

    Un regroupement peut être une catégorielle (le climat), une méta-variable
    (n_ferti), ou n'importe quelle colonne du jeu de données.
    """
    for var in spec.variables:
        if var.name == group and var.kind == CATEGORICAL:
            presentes = set(df[group].astype(str))
            return [m for m in var.domain if m in presentes]
    return sorted(df[group].dropna().unique().tolist())


def by_stratum(spec: SpaceSpec, df: pd.DataFrame, target: str, group: str,
               random_state: int = 42) -> dict:
    """Refait l'analyse **à l'intérieur de chaque modalité** du regroupement.

    C'est ce qui permet de répondre à « le classement des facteurs change-t-il d'un
    climat à l'autre ». Chaque strate est traitée comme un jeu de données à part
    entière, avec ses propres seuils de couverture : une strate trop peu peuplée voit
    ses analyses refusées plutôt que produites à l'aveugle.

    Le regroupement est retiré de l'analyse interne : à climat fixé, le climat ne
    varie plus et son R² vaudrait zéro, ce qui n'apprendrait rien.
    """
    if group not in df.columns:
        raise ValueError(f"Regroupement absent du jeu de données : {group}")

    strates = []
    for modalite in grouping_values(spec, df, group):
        sous = df[df[group].astype(str) == str(modalite)]
        sous = sous.dropna(subset=[target])
        n_points = int(len(sous))
        assez_anova = n_points >= MIN_POINTS_ANOVA_PER_BIN * N_BINS
        assez_modele = n_points >= MIN_POINTS_METAMODEL

        entree: dict = {"modalite": str(modalite), "n_points": n_points,
                        "anova": assez_anova, "metamodele": assez_modele}
        if assez_anova:
            entree["one_factor"] = [r for r in one_factor(spec, sous, target)
                                    if r["parametre"] != group]
        if assez_modele:
            modele = metamodel(spec, sous, target, random_state=random_state)
            entree["metamodel"] = modele
        strates.append(entree)

    return {"group": group, "target": target, "strata": strates,
            "comparison": _ranking_comparison(strates)}


def _ranking_comparison(strates: list[dict]) -> dict:
    """Matrice paramètre × modalité des R², et stabilité du classement."""
    lignes: dict[str, dict] = {}
    premiers: dict[str, str] = {}
    for strate in strates:
        rows = strate.get("one_factor")
        if not rows:
            continue
        premiers[strate["modalite"]] = rows[0]["parametre"]
        for rang, row in enumerate(rows, 1):
            entree = lignes.setdefault(row["parametre"], {"parametre": row["parametre"],
                                                          "r2": {}, "rangs": {}})
            entree["r2"][strate["modalite"]] = row["r2"]
            entree["rangs"][strate["modalite"]] = rang

    for entree in lignes.values():
        valeurs = list(entree["r2"].values())
        rangs = list(entree["rangs"].values())
        entree["r2_moyen"] = round(sum(valeurs) / len(valeurs), 4) if valeurs else 0.0
        entree["r2_min"] = round(min(valeurs), 4) if valeurs else 0.0
        entree["r2_max"] = round(max(valeurs), 4) if valeurs else 0.0
        entree["rang_min"] = min(rangs) if rangs else None
        entree["rang_max"] = max(rangs) if rangs else None
        # Un paramètre dont le rang bouge est celui qui interagit avec le regroupement.
        entree["rang_stable"] = bool(rangs) and min(rangs) == max(rangs)

    ordre = sorted(lignes.values(), key=lambda e: -e["r2_moyen"])
    dominants = sorted(set(premiers.values()))
    return {
        "parametres": ordre,
        "premier_par_modalite": premiers,
        "premier_stable": len(dominants) <= 1,
        "dominants": dominants,
    }


def run(spec: SpaceSpec, df: pd.DataFrame, targets: list[str],
        analyses: list[str] | None = None) -> dict:
    """Applique l'espace au jeu de données puis lance les analyses autorisées.

    ``analyses`` restreint ce qui est calculé. Par défaut tout ce que la couverture
    autorise : les deux analyses portées de la v1 — comparaison de métamodèles et
    HSIC-ANOVA — sont coûteuses, HSIC étant quadratique en nombre de points.
    """
    demandees = set(analyses) if analyses else {"one_factor", "metamodel_comparison",
                                                "hsic", "pdp"}
    cov = coverage(spec, df)
    kept = spec.filter(df)

    results: dict[str, dict] = {}
    for target in targets:
        entry: dict = {}
        if cov.allows("anova") and "one_factor" in demandees:
            entry["one_factor"] = one_factor(spec, kept, target)
        if cov.allows("metamodel") and "metamodel_comparison" in demandees:
            entry["metamodel_comparison"] = _compare_models(spec, kept, target)
        if cov.allows("metamodel") and "hsic" in demandees:
            entry["hsic"] = _hsic(spec, kept, target)
        if cov.allows("subspace") and "pdp" in demandees:
            entry["pdp"] = _pdp(spec, kept, target)
        results[target] = entry

    return {
        "n_points": cov.n_points,
        "n_available": cov.n_available,
        "coverage": {
            "subspaces": cov.subspaces,
            "verdicts": [v.__dict__ for v in cov.verdicts],
            "reachable": cov.reachable,
            "unconditional": cov.unconditional,
            "decreed": cov.decreed,
            "unreachable": cov.unreachable,
        },
        "targets": results,
    }
