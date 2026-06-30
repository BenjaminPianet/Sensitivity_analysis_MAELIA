from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree

from .models import prepare_X
from .config import FEATURE_LABELS, PALETTE, TARGET_LABELS


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": PALETTE["paper"],
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "grid.color": PALETTE["grid"],
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
    })


def label(name: str) -> str:
    return FEATURE_LABELS.get(name, TARGET_LABELS.get(name, name))


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_one_factor(anova: pd.DataFrame, target: str, path: Path, top_n: int = 12) -> Path:
    setup_style()
    data = anova[anova["sortie"] == target].head(top_n).copy()
    data["label"] = data["parametre"].map(label)
    data = data.sort_values("R2", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.42 * len(data) + 1.8)))
    colors = sns.color_palette("crest", n_colors=max(3, len(data)))
    ax.barh(data["label"], data["R2"], color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Part de variance expliquée seule (R²)")
    ax.set_ylabel("")
    ax.set_title(f"Paramètres les plus influents — {label(target)}")
    ax.set_xlim(0, max(0.02, data["R2"].max() * 1.18))
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.2f}" for v in data["R2"]], padding=4, fontsize=9)
    ax.text(
        0.0, -0.18,
        "Lecture : plus la barre est longue, plus ce paramètre sépare des comportements différents.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(left=True, bottom=False)
    return _save(fig, path)


def plot_interaction_heatmap(matrix: pd.DataFrame, target: str, path: Path, top_n: int = 16) -> Path:
    setup_style()
    scores = matrix.sum(axis=1).sort_values(ascending=False)
    keep = list(scores.head(min(top_n, len(scores))).index)
    mat = matrix.loc[keep, keep].copy()
    mat.index = [label(c) for c in mat.index]
    mat.columns = [label(c) for c in mat.columns]
    mask = np.triu(np.ones_like(mat, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        mat,
        mask=mask,
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        vmin=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "R² d'interaction"},
        ax=ax,
    )
    ax.set_title(f"Couples de paramètres qui interagissent — {label(target)}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.text(
        0.0, -0.13,
        "Lecture : une case foncée indique que le rôle d'un paramètre dépend fortement de l'autre.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    return _save(fig, path)


def plot_sobol_total(sobol: pd.DataFrame, target: str, path: Path, top_n: int = 12) -> Path:
    setup_style()
    data = sobol[sobol["sortie"] == target].head(top_n).copy()
    data["label"] = data["parametre"].map(label)
    data = data.sort_values("ST", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.42 * len(data) + 1.8)))
    colors = sns.color_palette("mako", n_colors=max(3, len(data)))
    ax.barh(data["label"], data["ST"], color=colors, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Indice de Sobol total estimé")
    ax.set_ylabel("")
    ax.set_title(f"Influence globale avec interactions — {label(target)}")
    ax.set_xlim(0, max(0.02, data["ST"].max() * 1.18))
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.2f}" for v in data["ST"]], padding=4, fontsize=9)
    ax.text(
        0.0, -0.18,
        "Lecture : cet indice inclut l'effet direct du paramètre et ses interactions avec les autres.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(left=True, bottom=False)
    return _save(fig, path)




def plot_pce_sobol(sobol: pd.DataFrame, target: str, path: Path, top_n: int = 12) -> Path:
    setup_style()
    data = sobol[sobol["sortie"] == target].head(top_n).copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "PCE non disponible pour cette sortie", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return _save(fig, path)
    data["label"] = data["parametre"].map(label)
    data = data.sort_values("Sobol_ST", ascending=True)
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10.5, max(5, 0.46 * len(data) + 1.9)))
    ax.barh(y - 0.18, data["Sobol_S1"], height=0.34, color=PALETTE["blue"], label="Ordre 1", edgecolor="white", linewidth=1.0)
    ax.barh(y + 0.18, data["Sobol_ST"], height=0.34, color=PALETTE["teal"], label="Total", edgecolor="white", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("Indice de Sobol")
    ax.set_ylabel("")
    ax.set_title(f"Sobol ordre 1 et total — {label(target)}")
    xmax = max(0.02, float(data[["Sobol_S1", "Sobol_ST"]].max().max()) * 1.18)
    ax.set_xlim(0, xmax)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.0, -0.18,
        "Lecture : le PCE est entraîné sur les points faisables SMT; les indices sont calculés depuis ses coefficients, sans plan Saltelli artificiel.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(left=True, bottom=False)
    return _save(fig, path)


def plot_hsic_order_decomposition(hsic_terms: pd.DataFrame, target: str, path: Path) -> Path:
    setup_style()
    data = hsic_terms[hsic_terms["sortie"] == target].copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.text(0.5, 0.5, "HSIC-ANOVA indisponible pour cette sortie", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return _save(fig, path)

    order_summary = (
        data.groupby("order", as_index=False)["contribution_hsic_globale_pct"]
        .sum()
        .sort_values("order")
    )
    represented = float(order_summary["contribution_hsic_globale_pct"].sum())
    if represented < 99.5:
        order_summary = pd.concat([
            order_summary,
            pd.DataFrame([{"order": "Non représenté", "contribution_hsic_globale_pct": max(0.0, 100.0 - represented)}]),
        ], ignore_index=True)

    colors = {
        1: PALETTE["teal"],
        2: PALETTE["amber"],
        3: "#F4A261",
        4: PALETTE["coral"],
        "Non représenté": "#CBD5E1",
    }

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    left = 0.0
    y = [0]
    for row in order_summary.itertuples(index=False):
        order = row.order
        value = float(row.contribution_hsic_globale_pct)
        label_text = f"Ordre {order}" if isinstance(order, (int, np.integer)) else str(order)
        ax.barh(
            y,
            [value],
            left=left,
            height=0.42,
            color=colors.get(order, "#7C8DA6"),
            edgecolor="white",
            linewidth=1.2,
            label=label_text,
        )
        if value >= 4.0:
            ax.text(left + value / 2, 0, f"{value:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["ink"])
        left += value

    ax.set_xlim(0, max(100.0, left * 1.04))
    ax.set_yticks([])
    ax.set_xlabel("Contribution au HSIC global (%)")
    ax.set_title(f"Décomposition HSIC-ANOVA — {label(target)}")
    ax.legend(frameon=False, ncol=min(len(order_summary), 5), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", alpha=0.28)
    ax.grid(axis="y", visible=False)
    ax.text(
        0.0,
        -0.42,
        "Lecture : l'ordre 1 correspond aux effets simples; l'ordre 2 aux interactions deux à deux; "
        "les ordres supérieurs signalent des dépendances plus combinatoires. Le gris, s'il apparaît, "
        "regroupe les termes non retenus par le filtrage HSIC.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
        va="top",
    )
    sns.despine(left=True, bottom=False)
    return _save(fig, path)

def plot_metamodel_performance(metrics: dict, target: str, path: Path) -> Path:
    setup_style()
    values = [float(metrics.get("R2_train", 0.0)), float(metrics.get("Q2_test", 0.0))]
    names = ["R² entraînement", "Q² test"]
    colors = [PALETTE["blue"], PALETTE["amber"]]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.4, width=0.58)
    ax.axhline(0, color=PALETTE["ink"], linewidth=1)
    ax.set_ylim(min(-0.05, min(values) - 0.08), min(1.05, max(0.2, max(values) + 0.12)))
    ax.set_ylabel("Score de prédiction")
    ax.set_title(f"Qualité du métamodèle — {label(target)}")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=11, fontweight="bold")
    ax.text(
        0.0, -0.22,
        "Lecture : R² mesure l'ajustement sur les données vues ; Q² mesure la capacité à prédire des simulations mises de côté.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(left=False, bottom=True)
    return _save(fig, path)


def plot_regions(regions: pd.DataFrame, target: str, path: Path, top_n: int = 10) -> Path:
    setup_style()
    data = regions.head(top_n).copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "Aucune région sensible stable détectée", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return _save(fig, path)
    data = data.sort_values("ecart_a_la_moyenne", ascending=True)
    colors = [PALETTE["coral"] if v < 0 else PALETTE["teal"] for v in data["ecart_a_la_moyenne"]]
    labels = [f"{r}\n{p:.0%} des points" for r, p in zip(data["region"], data["part_des_points"])]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(data) + 2)))
    ax.barh(labels, data["ecart_a_la_moyenne"], color=colors, edgecolor="white", linewidth=1.2)
    ax.axvline(0, color=PALETTE["ink"], linewidth=1)
    ax.set_xlabel(f"Écart à la moyenne globale de {label(target)}")
    ax.set_ylabel("")
    ax.set_title(f"Régions locales mises en évidence par l'arbre — {label(target)}")
    for i, (_, row) in enumerate(data.iterrows()):
        x = row["ecart_a_la_moyenne"]
        ha = "left" if x >= 0 else "right"
        offset = max(abs(data["ecart_a_la_moyenne"]).max() * 0.02, 0.01)
        ax.text(x + (offset if x >= 0 else -offset), i, f"moy. {row['moyenne_observee']:.3g}", va="center", ha=ha, fontsize=9)
    ax.text(
        0.0, -0.18,
        "Lecture : chaque barre résume un ensemble de simulations partageant les mêmes règles de seuil.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(left=True, bottom=False)
    return _save(fig, path)


def plot_tree_figure(tree_result, target: str, path: Path) -> Path:
    setup_style()
    tree = tree_result.pipeline.named_steps["tree"]
    fig_height = max(6, 1.9 * (tree.get_depth() + 1))
    fig, ax = plt.subplots(figsize=(24, fig_height))
    plot_tree(
        tree,
        feature_names=tree_result.feature_names,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Arbre de décision — {label(target)} | Q² test = {tree_result.metrics['Q2_test']:.2f}")
    return _save(fig, path)



def plot_temporal_trajectories(
    dynamic: pd.DataFrame,
    target: str,
    path: Path,
    group_col: str = "point_idx",
    max_curves: int = 280,
    random_state: int = 42,
) -> Path:
    setup_style()
    if target not in dynamic.columns:
        raise ValueError(f"Sortie dynamique absente: {target}")
    if group_col not in dynamic.columns or dynamic[group_col].isna().all():
        group_col = "sim_idx"
    data = (
        dynamic[[group_col, "date", "time_index", target]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[group_col, "date", target])
        .groupby([group_col, "date", "time_index"], as_index=False)[target]
        .mean()
        .sort_values([group_col, "time_index"])
    )
    if data.empty:
        raise ValueError(f"Aucune trajectoire exploitable pour {target}")

    ids = np.array(sorted(data[group_col].unique()))
    rng = np.random.default_rng(random_state)
    if len(ids) > max_curves:
        ids = np.sort(rng.choice(ids, size=max_curves, replace=False))
    shown = data[data[group_col].isin(ids)]

    summary = data.groupby(["date", "time_index"], as_index=False)[target].agg(
        mean="mean",
        q10=lambda s: s.quantile(0.10),
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
        q90=lambda s: s.quantile(0.90),
    ).sort_values("time_index")

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.fill_between(summary["date"], summary["q10"], summary["q90"], color=PALETTE["blue"], alpha=0.12, linewidth=0, label="10-90 %")
    ax.fill_between(summary["date"], summary["q25"], summary["q75"], color=PALETTE["teal"], alpha=0.18, linewidth=0, label="25-75 %")
    for _, curve in shown.groupby(group_col, sort=True):
        ax.plot(curve["date"], curve[target], color=PALETTE["blue"], alpha=0.12, linewidth=0.8)
    ax.plot(summary["date"], summary["mean"], color="black", linewidth=3.0, label="Moyenne")
    ax.set_title(f"Évolution temporelle — {label(target)}")
    ax.set_xlabel("Date")
    ax.set_ylabel(label(target))
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(
        0.0, -0.20,
        f"Lecture : {len(ids)} trajectoires individuelles affichées sur {data[group_col].nunique()} groupes; la courbe noire est la moyenne globale.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    fig.autofmt_xdate()
    sns.despine(left=False, bottom=False)
    return _save(fig, path)


def _feature_grid(series: pd.Series, categorical: bool, n: int = 12) -> np.ndarray:
    values = series.dropna()
    if values.empty:
        return np.array([])
    if categorical:
        counts = values.astype(str).value_counts().head(n)
        return counts.index.to_numpy(dtype=object)
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    uniques = np.unique(numeric.to_numpy())
    if len(uniques) <= n:
        return np.sort(uniques)
    return np.unique(np.round(np.quantile(numeric, np.linspace(0.05, 0.95, n)), 6))


def plot_pdp_ice(
    model: Pipeline,
    df: pd.DataFrame,
    target: str,
    feature: str,
    features: list[str],
    categorical: list[str],
    continuous: list[str],
    path: Path,
    ice_sample: int = 80,
    grid_size: int = 12,
    random_state: int = 42,
) -> Path:
    setup_style()
    X = prepare_X(df, features, categorical, continuous)
    y = pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    if X.empty:
        raise ValueError(f"Dataset vide pour PDP/ICE {target}")
    grid = _feature_grid(X[feature], feature in categorical, n=grid_size)
    if len(grid) == 0:
        raise ValueError(f"Grille vide pour {feature}")
    n = min(ice_sample, len(X))
    X_ref = X.sample(n=n, random_state=random_state).reset_index(drop=True)

    ice = []
    for value in grid:
        X_mod = X_ref.copy()
        X_mod[feature] = value
        ice.append(model.predict(X_mod))
    ice_arr = np.asarray(ice).T
    pdp = ice_arr.mean(axis=0)

    is_cat = feature in categorical
    x = np.arange(len(grid)) if is_cat else grid.astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for row in ice_arr:
        ax.plot(x, row, color="#94A3B8", alpha=0.23, linewidth=0.9)
    ax.plot(x, pdp, color="black", linewidth=3.2, label="PDP moyenne")
    ax.scatter(x, pdp, color="black", s=28, zorder=3)
    if is_cat:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in grid], rotation=35, ha="right")
    ax.set_title(f"PDP/ICE finale — {label(target)}")
    ax.set_xlabel(label(feature))
    ax.set_ylabel(label(target))
    ax.legend(frameon=False)
    ax.text(
        0.0, -0.22,
        "Lecture : les lignes grises sont des scénarios individuels; la ligne noire donne l'effet moyen prédit par le métamodèle.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=9,
    )
    sns.despine(left=False, bottom=False)
    return _save(fig, path)
