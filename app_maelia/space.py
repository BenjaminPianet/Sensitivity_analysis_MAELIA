"""Spécification d'un espace de conception hiérarchique : domaine, fenêtre, dérivation.

Prototype du lot 1. Une seule structure déclarative décrit l'espace exploré ; tout le
reste — variables atteignables, inconditionnelles, décrétées, sous-espaces, matrice
d'activité, filtrage — s'en dérive au lieu d'être réécrit à la main.

Deux champs par variable, et seulement deux :
  - ``domain`` : ce que le modèle autorise, métadonnée fixe ;
  - ``window`` : ce que l'utilisateur explore, toujours inclus dans le domaine.

Trois natures, qui ne diffèrent que par la représentation de la fenêtre :
  - continue    : ``[min, max]``
  - ordinale    : liste des niveaux retenus
  - catégorielle : pas de fenêtre — l'exploration est exhaustive par construction.

Figer une variable n'est pas un cas particulier : c'est une fenêtre réduite à un point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd

CONTINUOUS = "continuous"
ORDINAL = "ordinal"
CATEGORICAL = "categorical"

KINDS = {CONTINUOUS, ORDINAL, CATEGORICAL}


class SpecError(ValueError):
    """Spécification d'espace invalide."""


@dataclass(frozen=True)
class Level:
    """Un niveau d'une méta-variable, et les variables que ce niveau rend actives.

    ``tag`` est l'étiquette vue par SMT et par le générateur d'itinéraires
    (« 2_ferti »), tandis que ``value`` est la valeur numérique stockée dans les
    données. Les deux coexistent parce que SMT indexe les ordinales par étiquette.
    """

    value: int
    label: str
    activates: tuple[str, ...]
    tag: str = ""

    def __post_init__(self):
        if not self.tag:
            object.__setattr__(self, "tag", str(self.value))


@dataclass(frozen=True)
class MetaVariable:
    """Variable de décision dont les niveaux gouvernent l'activité d'autres variables."""

    name: str
    label: str
    kind: str
    levels: tuple[Level, ...]
    domain: tuple
    window: tuple

    def levels_in_window(self) -> list[Level]:
        return [lv for lv in self.levels if lv.value in self.window]

    @property
    def is_frozen(self) -> bool:
        return len(self.window) == 1


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    kind: str
    domain: tuple
    window: tuple | None = None
    always_active: bool = False
    scale: str | None = None
    # Variable dont la valeur n'est pas tirée mais **imposée par le terrain** : le
    # climat est porté par l'îlot auquel appartient la parcelle. Elle reste une
    # variable de l'espace — elle est analysée, filtrée, décrite — mais elle sort du
    # plan SMT, et l'affectation des points aux parcelles doit la respecter.
    stratified: bool = False

    @property
    def effective_window(self) -> tuple:
        """La fenêtre réellement explorée. Absente chez les catégorielles : c'est le domaine."""
        return self.domain if self.window is None else self.window

    @property
    def is_frozen(self) -> bool:
        """Fenêtre réduite à un point : la variable est figée, pas explorée."""
        if self.kind == CATEGORICAL:
            return len(self.domain) == 1
        if self.kind == CONTINUOUS:
            lo, hi = self.effective_window
            return lo == hi
        return len(self.effective_window) == 1


@dataclass(frozen=True)
class SpaceSpec:
    name: str
    meta_variables: tuple[MetaVariable, ...]
    variables: tuple[Variable, ...]
    sentinel_inactive: float = -1.0

    # ── Chargement ────────────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, payload: dict) -> "SpaceSpec":
        metas = tuple(
            MetaVariable(
                name=m["name"],
                label=m.get("label", m["name"]),
                kind=m.get("kind", ORDINAL),
                levels=tuple(
                    Level(lv["value"], lv.get("label", str(lv["value"])),
                          tuple(lv.get("activates", [])), lv.get("tag", ""))
                    for lv in m["levels"]
                ),
                domain=tuple(m["domain"]),
                window=tuple(m.get("window", m["domain"])),
            )
            for m in payload.get("meta_variables", [])
        )
        variables = tuple(
            Variable(
                name=v["name"],
                label=v.get("label", v["name"]),
                kind=v.get("kind", CONTINUOUS),
                domain=tuple(v["domain"]),
                window=tuple(v["window"]) if "window" in v else None,
                always_active=bool(v.get("always_active", False)),
                scale=v.get("scale"),
                stratified=bool(v.get("stratified", False)),
            )
            for v in payload.get("variables", [])
        )
        spec = cls(
            name=payload.get("name", "sans nom"),
            meta_variables=metas,
            variables=variables,
            sentinel_inactive=float(payload.get("sentinel_inactive", -1.0)),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "SpaceSpec":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def for_dataset(cls, log_dir: str | Path, fallback: str | Path) -> tuple["SpaceSpec", str]:
        """Spécification décrivant les données d'un dossier de logs.

        Un jeu de données produit par cette application porte son ``space_spec.json``
        à côté du dataset : c'est lui qui fait foi, puisqu'il décrit exactement
        l'espace qui a produit ces simulations. À défaut — cas des données
        antérieures à ce format — on retombe sur la spécification de référence.

        Renvoie aussi l'origine retenue, pour que l'interface puisse le dire.
        """
        embedded = Path(log_dir).expanduser() / "space_spec.json"
        if embedded.exists():
            return cls.load(embedded), "dataset"
        return cls.load(fallback), "defaut"

    # ── Validation ────────────────────────────────────────────────────────────
    def validate(self) -> None:
        known = {v.name for v in self.variables}

        for var in self.variables:
            if var.kind not in KINDS:
                raise SpecError(f"{var.name} : nature inconnue « {var.kind} »")

            if var.stratified and var.kind != CATEGORICAL:
                raise SpecError(
                    f"{var.name} : seule une catégorielle peut être imposée par le "
                    "terrain ; une continue ne se stratifie pas.")

            if var.kind == CATEGORICAL:
                # L'exhaustivité n'est pas une règle ajoutée : c'est la seule nature
                # dont la fenêtre n'est pas réglable, donc le champ doit être absent.
                if var.window is not None:
                    raise SpecError(
                        f"{var.name} : une catégorielle est explorée exhaustivement, "
                        "elle ne peut pas porter de fenêtre."
                    )
                continue

            if var.kind == CONTINUOUS:
                lo, hi = var.effective_window
                dlo, dhi = var.domain
                if lo > hi:
                    raise SpecError(f"{var.name} : fenêtre vide [{lo}, {hi}]")
                if lo < dlo or hi > dhi:
                    raise SpecError(
                        f"{var.name} : fenêtre [{lo}, {hi}] hors du domaine [{dlo}, {dhi}]"
                    )
                # Une fenêtre englobant la sentinelle rendrait une valeur légitime
                # indiscernable d'une inactivité.
                if lo <= self.sentinel_inactive <= hi:
                    raise SpecError(
                        f"{var.name} : la fenêtre [{lo}, {hi}] contient la sentinelle "
                        f"d'inactivité ({self.sentinel_inactive}) ; les valeurs actives "
                        "ne seraient plus distinguables des inactives."
                    )
            else:  # ordinale
                if not set(var.effective_window) <= set(var.domain):
                    raise SpecError(f"{var.name} : fenêtre hors du domaine")
                if not var.effective_window:
                    raise SpecError(f"{var.name} : fenêtre vide")

        for meta in self.meta_variables:
            values = [lv.value for lv in meta.levels]
            if set(meta.domain) != set(values):
                raise SpecError(f"{meta.name} : domaine et niveaux incohérents")
            if not meta.window:
                raise SpecError(f"{meta.name} : fenêtre vide")
            if not set(meta.window) <= set(meta.domain):
                raise SpecError(f"{meta.name} : fenêtre hors du domaine {meta.domain}")
            for lv in meta.levels:
                unknown = set(lv.activates) - known
                if unknown:
                    raise SpecError(f"{meta.name} niveau {lv.value} : variables inconnues {sorted(unknown)}")

            # Emboîtement : le niveau k doit activer tout ce qu'active le niveau k-1.
            # Sans cela, « la borne basse fixe l'inconditionnel » n'a plus de sens.
            if meta.kind == ORDINAL:
                ordered = sorted(meta.levels, key=lambda lv: lv.value)
                for lower, upper in zip(ordered, ordered[1:]):
                    if not set(lower.activates) <= set(upper.activates):
                        raise SpecError(
                            f"{meta.name} : hiérarchie non emboîtée entre les niveaux "
                            f"{lower.value} et {upper.value} — "
                            f"{sorted(set(lower.activates) - set(upper.activates))} "
                            "disparaît en montant d'un niveau."
                        )

    # ── Dérivation ────────────────────────────────────────────────────────────
    def feature_names(self) -> list[str]:
        return [m.name for m in self.meta_variables] + [v.name for v in self.variables]

    def stratified_variables(self) -> list[Variable]:
        """Variables imposées par le terrain, hors du tirage."""
        return [v for v in self.variables if v.stratified]

    def _governing(self, var_name: str) -> list[MetaVariable]:
        """Méta-variables qui gouvernent l'activité de cette variable."""
        return [
            m for m in self.meta_variables
            if any(var_name in lv.activates for lv in m.levels)
        ]

    def reachable(self) -> set[str]:
        """Variables qui existent quelque part dans la fenêtre (⋃ des niveaux retenus)."""
        return self._derive(quantifier=any)

    def unconditional(self) -> set[str]:
        """Variables actives partout dans la fenêtre (⋂ des niveaux retenus)."""
        return self._derive(quantifier=all)

    def _derive(self, quantifier) -> set[str]:
        out: set[str] = set()
        for var in self.variables:
            metas = self._governing(var.name)
            if var.always_active or not metas:
                out.add(var.name)
                continue
            if all(
                quantifier(var.name in lv.activates for lv in m.levels_in_window())
                for m in metas
            ):
                out.add(var.name)
        return out

    def decreed(self) -> set[str]:
        """Variables présentes mais conditionnelles : c'est ce que la hiérarchie coûte."""
        return self.reachable() - self.unconditional()

    def subspaces(self) -> list[dict]:
        """Produit cartésien des fenêtres des méta-variables."""
        if not self.meta_variables:
            return [{}]
        combos = product(*[m.window for m in self.meta_variables])
        return [dict(zip([m.name for m in self.meta_variables], combo)) for combo in combos]

    # ── Confrontation aux données ─────────────────────────────────────────────
    def acting_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matrice booléenne lignes × features : la variable est-elle active sur cette ligne ?"""
        acting = pd.DataFrame(True, index=df.index, columns=self.feature_names())
        for var in self.variables:
            metas = self._governing(var.name)
            if var.always_active or not metas:
                continue
            mask = pd.Series(True, index=df.index)
            for meta in metas:
                activating = [lv.value for lv in meta.levels if var.name in lv.activates]
                levels = pd.to_numeric(df[meta.name], errors="coerce").round()
                mask &= levels.isin(activating)
            acting[var.name] = mask.to_numpy()
        return acting

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lignes du jeu de données qui tombent dans l'espace décrit.

        Une fenêtre laissée égale à son domaine n'est pas une restriction : elle
        n'impose aucun filtre. C'est ce qui permet à l'espace complet de rendre le jeu
        de données entier, alors même que l'itinéraire réalisé déborde parfois du
        domaine échantillonné (décalages de calendrier). Ces débordements relèvent du
        diagnostic ``check_data``, pas du filtrage.
        """
        mask = pd.Series(True, index=df.index)

        for meta in self.meta_variables:
            if set(meta.window) == set(meta.domain):
                continue
            levels = pd.to_numeric(df[meta.name], errors="coerce").round()
            mask &= levels.isin(list(meta.window))

        acting = self.acting_matrix(df)
        for var in self.variables:
            if var.kind != CONTINUOUS or var.name not in df.columns:
                continue
            lo, hi = var.effective_window
            if (lo, hi) == tuple(var.domain):
                continue
            values = pd.to_numeric(df[var.name], errors="coerce")
            inside = (values >= lo) & (values <= hi)
            # La contrainte ne vaut que là où la variable est active : sinon on
            # éliminerait toutes les lignes portant la sentinelle.
            mask &= inside | ~acting[var.name]

        return df[mask]

    def check_data(self, df: pd.DataFrame) -> list[str]:
        """Confronte le jeu de données au domaine déclaré, hors valeurs inactives.

        Sert de garde-fou quand une spécification est associée à un dataset : un
        domaine plus étroit que les données trahit soit un plan mal décrit (unité,
        échelle), soit des valeurs réalisées qui s'écartent des valeurs échantillonnées.
        Les deux se sont produits sur terrainSA — doses en physique plutôt qu'en log10,
        et dates d'apport décalées par les contraintes de calendrier.
        """
        anomalies: list[str] = []
        acting = self.acting_matrix(df)
        for var in self.variables:
            if var.kind != CONTINUOUS or var.name not in df.columns:
                continue
            values = pd.to_numeric(df[var.name], errors="coerce")[acting[var.name]].dropna()
            if values.empty:
                continue
            lo, hi = var.domain
            outside = values[(values < lo) | (values > hi)]
            if not outside.empty:
                anomalies.append(
                    f"{var.name} : {len(outside)} valeurs actives hors du domaine "
                    f"[{lo}, {hi}] (observé [{values.min():g}, {values.max():g}])"
                )
        for meta in self.meta_variables:
            if meta.name not in df.columns:
                continue
            levels = pd.to_numeric(df[meta.name], errors="coerce").round().dropna()
            unknown = sorted(set(levels.unique()) - set(meta.domain))
            if unknown:
                anomalies.append(f"{meta.name} : niveaux inconnus dans les données {unknown}")
        return anomalies

    def filter_warnings(self, df: pd.DataFrame) -> list[str]:
        """Ce que le filtrage a fait de contre-intuitif, dit plutôt que subi.

        Une contrainte ne s'applique que là où la variable est active — sinon les
        lignes où elle ne s'applique pas seraient toutes éliminées, ce qui n'aurait
        pas de sens. La conséquence surprend quand la fenêtre est **dégénérée** :
        aucune simulation ne porte exactement la valeur demandée pour une variable
        continue, si bien qu'il ne reste que les lignes où cette variable n'existe
        pas. On croit alors étudier « la dose fixée à 60 » et l'on étudie en réalité
        les itinéraires sans apport du tout.
        """
        avertissements: list[str] = []
        acting = self.acting_matrix(df)
        retenu = self.filter(df)
        if retenu.empty:
            return avertissements

        for var in self.variables:
            if var.kind != CONTINUOUS or var.name not in df.columns:
                continue
            lo, hi = var.effective_window
            if (lo, hi) == tuple(var.domain):
                continue
            actives = acting.loc[retenu.index, var.name]
            if actives.any():
                continue
            avertissements.append(
                f"{var.label} : aucune des {len(retenu)} simulations retenues ne porte "
                f"cette variable. La fenêtre [{lo:g}, {hi:g}] ne correspond à aucune "
                f"valeur simulée ; il ne reste que les itinéraires où {var.name} "
                f"n'existe pas."
            )
        return avertissements

    def coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Effectif disponible par sous-espace, une fois l'espace appliqué."""
        kept = self.filter(df)
        names = [m.name for m in self.meta_variables]
        rows = []
        for sub in self.subspaces():
            sel = pd.Series(True, index=kept.index)
            for name in names:
                sel &= pd.to_numeric(kept[name], errors="coerce").round() == sub[name]
            rows.append({**sub, "n_points": int(sel.sum())})
        return pd.DataFrame(rows)

    # ── Confort ───────────────────────────────────────────────────────────────
    def with_window(self, **windows) -> "SpaceSpec":
        """Copie de la spécification avec des fenêtres remplacées (nom -> fenêtre).

        Un nom inconnu est une erreur, jamais un silence : sinon une faute de frappe
        laisserait croire à une restriction qui n'a pas eu lieu.
        """
        known = {m.name for m in self.meta_variables} | {v.name for v in self.variables}
        unknown = sorted(set(windows) - known)
        if unknown:
            raise SpecError(f"Variables inconnues dans cet espace : {unknown}")

        metas = tuple(
            MetaVariable(m.name, m.label, m.kind, m.levels, m.domain, tuple(windows[m.name]))
            if m.name in windows else m
            for m in self.meta_variables
        )
        variables = tuple(
            Variable(v.name, v.label, v.kind, v.domain, tuple(windows[v.name]),
                     v.always_active, v.scale, v.stratified)
            if v.name in windows else v
            for v in self.variables
        )
        spec = SpaceSpec(self.name, metas, variables, self.sentinel_inactive)
        spec.validate()
        return spec
