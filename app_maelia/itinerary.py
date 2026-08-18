"""Génération et validation des calendriers d'itinéraire technique.

Reprend la logique de ``generate_ops_for_parcelle`` de ``run_terrainSA_batch.py``,
avec une différence décisive : **les valeurs sont lues par nom, pas par position**.

La version v1 collecte les flottants décodés dans l'ordre où ils arrivent puis les
indexe en dur (``fl[0]`` = date de semis, ``fl[5]`` = date d'apport 1, ``fl[8]`` =
date de récolte). Cela suppose que le plan contienne toujours les douze variables
continues, dans le même ordre. Dès que l'utilisateur restreint l'espace — une
préparation retirée, un apport figé — le plan se raccourcit, les positions glissent
et les dates seraient prises les unes pour les autres. Pas une erreur : des
itinéraires silencieusement faux.

Les valeurs attendues ici sont **agronomiques** : une dose est en kg N/ha, pas en
log10. La conversion a déjà eu lieu dans ``plan.decode``.

Deux niveaux de validation, comme en v1 :
  - la faisabilité du calendrier d'un point (exceptions levées à la construction) ;
  - la cohérence de la séquence d'opérations produite (semis sans récolte, doublons…).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from .plan import BuiltPlan, build_design_space, decode, sample
from .space import CONTINUOUS, SpaceSpec

# Constantes reprises de run_terrainSA_batch.py ; toute divergence ferait diverger
# les itinéraires produits de ceux que GAMA a réellement simulés.
HARVEST_BUFFER_DAYS = 7
MIN_DAYS_LAST_OP_TO_HARVEST = 1
CROP = "ble"
CROP_MAELIA = "BTH"
FERTILIZER_TYPE = "AN"
PREPA_TYPE_1 = "travail_sol_1"
PREPA_TYPE_2 = "une_reprise"
SEMIS_TYPE = "semis"

DEFAULT_CAMPAIGNS = tuple(range(2019, 2029))

# Valeurs de repli d'une variable absente du plan ou inactive, en unité agronomique.
# Reprend DEFAULT_FLOATS de la v1 ; les doses y sont en log10, converties ici.
DEFAULTS: dict[str, float] = {
    "Date_Semis": 75.0,
    "Delta_PREPA_Semis": -20.0,
    "Profondeur_Semis": 3.0,
    "Profondeur_Prepa_1": 15.0,
    "Profondeur_Prepa_2": 10.0,
    "Date_F1": 150.0,
    "Date_F2": 230.0,
    "Date_F3": 280.0,
    "Date_Recolte": 350.0,
    "Dose_F1": 10 ** 1.699,
    "Dose_F2": 10 ** 1.699,
    "Dose_F3": 10 ** 1.699,
}


class CalendarError(ValueError):
    """Calendrier impossible à satisfaire pour ce jeu de valeurs."""

    def __init__(self, constraint: str, message: str):
        super().__init__(message)
        self.constraint = constraint


def days_in_year(year: int) -> int:
    return (date(year + 1, 8, 1) - date(year, 8, 1)).days


def doy_to_date_str(year: int, doy: int) -> str:
    return (date(year, 8, 1) + timedelta(days=int(doy) - 1)).strftime("%d/%m/%Y")


def _value(values: dict, name: str) -> float:
    """Valeur agronomique, avec repli sur le défaut si absente, non numérique ou NaN."""
    raw = values.get(name)
    if raw is None:
        return DEFAULTS[name]
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return DEFAULTS[name]
    return DEFAULTS[name] if number != number else number  # NaN


def build_operations(values: dict, year: int, parcelle_id: str) -> list[dict]:
    """Opérations dateDose d'une parcelle pour une campagne, depuis des valeurs nommées."""
    n_ferti = int(round(float(values.get("n_ferti", 0) or 0)))
    nb_prepa = int(round(float(values.get("nb_prepa", 0) or 0)))

    semi_doy = max(1, int(round(_value(values, "Date_Semis"))))
    prepa_doy = (
        max(1, int(round(semi_doy + _value(values, "Delta_PREPA_Semis"))))
        if nb_prepa >= 1 else None
    )

    # Les apports sont réordonnés : chacun doit tomber au moins un jour après le précédent.
    ferti_doys: list[int | None] = [None, None, None]
    if n_ferti >= 1:
        ferti_doys[0] = int(round(_value(values, "Date_F1")))
    if n_ferti >= 2:
        ferti_doys[1] = max(int(round(_value(values, "Date_F2"))), ferti_doys[0] + 1)
    if n_ferti >= 3:
        ferti_doys[2] = max(int(round(_value(values, "Date_F3"))), ferti_doys[1] + 1)

    next_entry_doy = prepa_doy if prepa_doy is not None else semi_doy
    latest_rec_doy = days_in_year(year) + next_entry_doy - HARVEST_BUFFER_DAYS
    max_last_op_doy = latest_rec_doy - MIN_DAYS_LAST_OP_TO_HARVEST

    ferti_events = [
        (doy, _value(values, f"Dose_F{i + 1}"))
        for i, doy in enumerate(ferti_doys)
        if doy is not None and doy <= max_last_op_doy
    ]

    last_doy = ferti_events[-1][0] if ferti_events else semi_doy
    rec_doy = max(int(round(_value(values, "Date_Recolte"))), last_doy + MIN_DAYS_LAST_OP_TO_HARVEST)
    rec_doy = min(rec_doy, latest_rec_doy)
    if rec_doy <= last_doy:
        raise CalendarError(
            "recolte_apres_dernier_apport",
            f"Calendrier impossible pour {parcelle_id} campagne {year}: "
            f"dernière op DOY {last_doy}, récolte max DOY {latest_rec_doy}",
        )

    # L'espèce vient du point si le plan l'explore, sinon on garde le blé du plan
    # historique. MAELIA lit cette colonne et nulle part ailleurs : le mode DateDose
    # ignore la rotation déclarée sur la parcelle.
    espece = str(values.get("espece") or CROP_MAELIA)
    base = {"Trait": parcelle_id, "nom_MAELIA": espece,
            "plant_sem": CROP if espece == CROP_MAELIA else espece, "annee_deb": year}
    ops: list[dict] = []

    if nb_prepa >= 1:
        prepa_dates = [(prepa_doy, "", _value(values, "Profondeur_Prepa_1"), PREPA_TYPE_1)]
        if nb_prepa == 2:
            prepa2_doy = min(max(2, prepa_doy), semi_doy - 1)
            prepa1_doy = max(1, min(prepa_doy - 7, prepa2_doy - 1))
            if prepa1_doy >= prepa2_doy:
                raise CalendarError(
                    "deux_prepa_avant_semis",
                    f"Impossible de placer deux PREPA distinctes avant le semis "
                    f"pour {parcelle_id} campagne {year}: prepa={prepa_doy}, semis={semi_doy}",
                )
            prepa_dates = [
                (prepa1_doy, "", _value(values, "Profondeur_Prepa_1"), PREPA_TYPE_1),
                (prepa2_doy, "2", _value(values, "Profondeur_Prepa_2"), PREPA_TYPE_2),
            ]
        for op_doy, ordre_op, depth, op_type in prepa_dates:
            ops.append({**base, "id_operation": "PREPA", "ordre_op": ordre_op,
                        "DATE": doy_to_date_str(year, op_doy),
                        "PROF": str(round(depth, 1)), "TYPE": op_type, "DOSE": ""})

    ops.append({**base, "id_operation": "SEMIS", "ordre_op": "",
                "DATE": doy_to_date_str(year, semi_doy),
                "PROF": str(round(_value(values, "Profondeur_Semis"), 1)),
                "TYPE": SEMIS_TYPE, "DOSE": ""})

    for ordre, (doy, dose) in enumerate(ferti_events, 1):
        ops.append({**base, "id_operation": "FERTI_Min", "ordre_op": str(ordre),
                    "DATE": doy_to_date_str(year, doy), "PROF": "",
                    "TYPE": FERTILIZER_TYPE, "DOSE": str(round(dose, 1))})

    ops.append({**base, "id_operation": "RECOLTE", "ordre_op": "",
                "DATE": doy_to_date_str(year, rec_doy), "PROF": "", "TYPE": "", "DOSE": ""})

    for op in ops:
        op.setdefault("TEMPS", "")
    return ops


def check_sequence(rows: list[dict]) -> list[str]:
    """Cohérence de la séquence produite : doublons, semis ou apport hors culture."""
    import pandas as pd

    order = {"RECOLTE": 0, "PREPA": 1, "SEMIS": 2, "FERTI_Min": 3}
    by_parcelle: dict[str, list[dict]] = {}
    duplicates: dict[tuple, int] = {}
    issues: list[str] = []

    for row in rows:
        by_parcelle.setdefault(row["Trait"], []).append(row)
        key = (row["Trait"], row["annee_deb"], row["id_operation"], row["DATE"])
        duplicates[key] = duplicates.get(key, 0) + 1

    for (trait, campagne, op, day), count in duplicates.items():
        if count > 1:
            issues.append(f"{trait} campagne {campagne}: {count} opérations {op} le même jour ({day})")

    def as_date(value):
        return pd.to_datetime(value, dayfirst=True).date()

    for parcelle_id, ops in by_parcelle.items():
        culture_en_place = False
        for row in sorted(ops, key=lambda r: (as_date(r["DATE"]),
                                              order.get(r["id_operation"], 99),
                                              str(r.get("ordre_op", "")))):
            op, day, campagne = row["id_operation"], row["DATE"], int(row["annee_deb"])
            if as_date(day) < date(campagne, 8, 1):
                issues.append(f"{parcelle_id} campagne {campagne}: {op} le {day} avant démarrage campagne")
            if op == "SEMIS":
                if culture_en_place:
                    issues.append(f"{parcelle_id} campagne {campagne}: SEMIS le {day} alors qu'une culture est en place")
                culture_en_place = True
            elif op == "RECOLTE":
                if not culture_en_place:
                    issues.append(f"{parcelle_id} campagne {campagne}: RECOLTE le {day} sans culture en place")
                culture_en_place = False
            elif op == "FERTI_Min":
                if not culture_en_place:
                    issues.append(f"{parcelle_id} campagne {campagne}: FERTI_Min le {day} sans culture en place")
            elif op != "PREPA":
                issues.append(f"{parcelle_id} campagne {campagne}: opération inconnue {op}")
    return issues


def _doy(date_str: str, year: int) -> int:
    """Jour de campagne d'une date écrite au format dateDose."""
    day, month, yy = (int(part) for part in date_str.split("/"))
    return (date(yy, month, day) - date(year, 8, 1)).days + 1


# Opération -> variable de la spécification dont elle porte la date réalisée.
_DATE_OF = {("SEMIS", ""): "Date_Semis", ("RECOLTE", ""): "Date_Recolte",
            ("FERTI_Min", "1"): "Date_F1", ("FERTI_Min", "2"): "Date_F2",
            ("FERTI_Min", "3"): "Date_F3"}


def realised_dates(ops: list[dict], year: int) -> dict[str, int]:
    """Dates effectivement écrites dans l'itinéraire, en jour de campagne.

    Le générateur réordonne les apports (chacun au moins un jour après le précédent)
    et plafonne la récolte. La date obtenue peut donc sortir de la fenêtre demandée
    sans qu'aucune erreur ne soit levée — c'est ce décalage qui explique les
    Date_F2 à 324 et Date_F3 à 325 observés dans le dataset historique.
    """
    out: dict[str, int] = {}
    for op in ops:
        if int(op["annee_deb"]) != year:
            continue
        name = _DATE_OF.get((op["id_operation"], str(op.get("ordre_op", ""))))
        if name:
            out[name] = _doy(op["DATE"], year)
    return out


@dataclass
class CalendarReport:
    n_points: int
    n_ok: int
    campaigns: tuple[int, ...]
    failures: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)
    sequence_issues: list[str] = field(default_factory=list)
    # Variable -> décalage hors fenêtre : nombre de cas et amplitude observée.
    drift: dict[str, dict] = field(default_factory=dict)
    # Paire d'apports -> itinéraires où les deux dates sont quasi confondues.
    degenerate: dict[str, dict] = field(default_factory=dict)

    @property
    def n_failed(self) -> int:
        return self.n_points - self.n_ok

    @property
    def ok(self) -> bool:
        return self.n_failed == 0 and not self.sequence_issues and not self.drift

    @property
    def failure_rate(self) -> float:
        return 0.0 if not self.n_points else self.n_failed / self.n_points

    def summary(self) -> list[str]:
        lines = []
        if self.n_failed:
            lines.append(f"{self.n_failed}/{self.n_points} calendriers infaisables "
                         f"({', '.join(f'{k} × {v}' for k, v in self.failures.items())})")
        for name, info in sorted(self.drift.items()):
            lines.append(
                f"{name} : {info['n_out']} date(s) réalisée(s) hors de la fenêtre "
                f"{info['window']} — observé [{info['min']}, {info['max']}]"
            )
        for pair, info in sorted(self.degenerate.items()):
            lines.append(
                f"{pair} : {info['n_close']}/{info['n_total']} "
                f"({100 * info['share']:.0f} %) à {info['max_gap_days']} jour(s) ou moins "
                f"— apports quasi confondus, écart médian {info['median_gap']} j"
            )
        lines.extend(self.sequence_issues)
        return lines


def validate_points(points: list[dict], campaigns=DEFAULT_CAMPAIGNS,
                    max_examples: int = 5, spec: SpaceSpec | None = None,
                    max_gap_days: int = 7) -> CalendarReport:
    """Passe le générateur de calendrier sur des points déjà décodés.

    Purement en mémoire : rien n'est écrit, GAMA n'est pas sollicité. C'est ce qui
    permet de rejeter un espace problématique avant de lancer la moindre simulation.

    Deux natures de problème sont remontées :
      - un calendrier qu'aucun ordonnancement ne satisfait (exception à la
        construction) — rare, la géométrie du modèle le rend presque toujours
        satisfiable ;
      - un **décalage** : la date écrite sort de la fenêtre demandée parce que le
        générateur a réordonné ou plafonné. Silencieux, et de loin le cas fréquent.
      - une **dégénérescence** : deux apports successifs tombent à quelques jours
        d'intervalle. Le réordonnancement impose à chaque apport de suivre le
        précédent d'au moins un jour ; quand la date tirée est antérieure, elle est
        ramenée juste après. L'itinéraire reste valide et dans les fenêtres, mais
        « trois apports » y devient de fait un apport fractionné sur des jours
        consécutifs — ce qui n'est pas ce que le plan prétend explorer.
    """
    report = CalendarReport(n_points=len(points), n_ok=0, campaigns=tuple(campaigns))
    produced: list[dict] = []

    windows = {}
    if spec is not None:
        windows = {v.name: v.effective_window for v in spec.variables
                   if v.kind == CONTINUOUS and v.name in spec.reachable()}

    observed: dict[str, list[int]] = {}
    gaps: dict[str, list[int]] = {}

    for index, values in enumerate(points):
        parcelle_id = f"controle_{index:04d}"
        point_ops: list[dict] = []
        try:
            for year in campaigns:
                point_ops.extend(build_operations(values, year, parcelle_id))
        except CalendarError as exc:
            report.failures[exc.constraint] = report.failures.get(exc.constraint, 0) + 1
            if len(report.examples) < max_examples:
                report.examples.append(str(exc))
            continue

        produced.extend(point_ops)
        report.n_ok += 1

        for year in campaigns:
            realised = realised_dates(point_ops, year)
            for name, doy in realised.items():
                if name in windows:
                    observed.setdefault(name, []).append(doy)
            for first, second in (("Date_F1", "Date_F2"), ("Date_F2", "Date_F3")):
                if first in realised and second in realised:
                    gaps.setdefault(f"{first} → {second}", []).append(
                        realised[second] - realised[first])

    for name, values_seen in observed.items():
        lo, hi = windows[name]
        outside = [v for v in values_seen if v < lo or v > hi]
        if outside:
            report.drift[name] = {
                "n_out": len(outside),
                "n_total": len(values_seen),
                "window": [lo, hi],
                "min": min(values_seen),
                "max": max(values_seen),
            }

    for pair, values_seen in gaps.items():
        close = [g for g in values_seen if g <= max_gap_days]
        if close:
            ordered = sorted(values_seen)
            report.degenerate[pair] = {
                "n_close": len(close),
                "n_total": len(values_seen),
                "share": round(len(close) / len(values_seen), 4),
                "max_gap_days": max_gap_days,
                "median_gap": ordered[len(ordered) // 2],
                "min_gap": ordered[0],
            }

    if produced:
        report.sequence_issues = check_sequence(produced)[:max_examples]
    return report


def validate_space(spec: SpaceSpec, n_points: int = 500, seed: int = 42,
                   campaigns=DEFAULT_CAMPAIGNS, plan: BuiltPlan | None = None,
                   max_gap_days: int = 7) -> CalendarReport:
    """Échantillonne l'espace puis vérifie les calendriers qu'il produirait."""
    built = plan or build_design_space(spec)
    xt, _ = sample(built, n_points, seed=seed)
    return validate_points(decode(built, xt), campaigns=campaigns, spec=spec,
                           max_gap_days=max_gap_days)
