"""Réaligne dataset_metamodel.csv en reconstruisant X depuis les logs MAELIA.

Contexte : les colonnes feat_0..14 du dataset provenaient d'un DOE
(`doe_matrix_terrainSA.npy`) qui ne correspond PAS aux itinéraires réellement
simulés (dont proviennent les sorties). Les sorties (N_lixi, dCorg, rdt), elles,
sont correctement indexées par point_idx. Ce script reconstruit les 15 paramètres
SMT à partir des opérations réellement exécutées (suiviOTParParcelle.csv), campagne
par campagne, et remplace feat_0..14 — laissant les sorties intactes. Le résultat
est un dataset cohérent (chaque itinéraire correspond à ses sorties).

Convention de dates : jours de campagne, 1 = 1er août (doy 213). Une campagne va
d'août N à l'été N+1 ; les fertilisations de printemps sont donc en année N+1.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "simulations" / "log_terrainSA"

AGRI_FEATURES = [
    "n_ferti", "nb_prepa",
    "Date_Semis", "Delta_PREPA_Semis", "Profondeur_Semis",
    "Profondeur_Prepa_1", "Profondeur_Prepa_2",
    "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
    "Dose_F1", "Dose_F2", "Dose_F3",
]
FEAT_COLS = [f"feat_{i}" for i in range(len(AGRI_FEATURES))]

# Défauts pour les variables continues INACTIVES (ordre AGRI continu).
DEFAULTS = {
    "Delta_PREPA_Semis": -20.0, "Profondeur_Prepa_1": 15.0, "Profondeur_Prepa_2": 10.0,
    "Date_F1": 259.0, "Date_F2": 383.0, "Date_F3": 393.0,
    "Dose_F1": 10.0, "Dose_F2": 10.0, "Dose_F3": 10.0,
}


def sim_idx_of(name: str) -> int | None:
    m = re.search(r"smt_(\d+)", name)
    return int(m.group(1)) if m else None


def campaign_day(doy: float) -> float:
    """doy calendaire -> jour de campagne (1 = 1er août = doy 213)."""
    doy = float(doy)
    return doy - 212.0 if doy >= 213.0 else doy + 153.0


def reconstruct_parcelle(ops: pd.DataFrame) -> dict | None:
    """Reconstruit les 15 paramètres SMT (médiane sur les campagnes) pour une parcelle."""
    ops = ops.copy()
    ops["doy"] = pd.to_numeric(ops["date"], errors="coerce")
    ops["key"] = ops["annee"].astype(int) * 1000 + ops["doy"]
    ops = ops.dropna(subset=["doy"]).sort_values("key")

    semis = ops[ops["OT"] == "SEMIS"].sort_values("key")
    if semis.empty:
        return None
    semis_keys = semis["key"].to_numpy()

    campaigns: list[dict] = []
    for c, s_key in enumerate(semis_keys):
        next_key = semis_keys[c + 1] if c + 1 < len(semis_keys) else np.inf
        s_row = semis[semis["key"] == s_key].iloc[0]
        semis_doy = s_row["doy"]

        # Travail du sol : entre la récolte précédente et ce semis (même automne).
        prev_key = semis_keys[c - 1] if c >= 1 else -np.inf
        till = ops[(ops["OT"] == "TRAVAIL_SOL") & (ops["key"] < s_key) & (ops["key"] > prev_key)]
        till = till.sort_values("key")
        # Fertilisations : après ce semis, avant le semis suivant.
        ferti = ops[(ops["OT"] == "FERTI") & (ops["key"] >= s_key) & (ops["key"] < next_key)].sort_values("key")
        # Récolte : après ce semis, avant le semis suivant.
        rec = ops[(ops["OT"] == "RECOLTE") & (ops["key"] > s_key) & (ops["key"] < next_key)].sort_values("key")
        if rec.empty:
            continue

        n_prepa = len(till)
        till_depths = pd.to_numeric(till["profondeur[cm]"], errors="coerce").to_list()
        ferti_doys = ferti["doy"].to_list()
        ferti_doses = pd.to_numeric(ferti["FERTI_apportNminReel[kg/ha]"], errors="coerce").to_list()

        cd_semis = campaign_day(semis_doy)
        rec_c = campaign_day(rec.iloc[0]["doy"])
        # Delta = jour de campagne du travail le plus tardif (le plus proche du semis) - semis.
        delta = campaign_day(till["doy"].max()) - cd_semis if n_prepa >= 1 else np.nan

        campaigns.append({
            "n_ferti": len(ferti),
            "nb_prepa": n_prepa,
            "Date_Semis": cd_semis,
            "Delta_PREPA_Semis": delta,
            "Profondeur_Semis": pd.to_numeric(s_row["profondeur[cm]"], errors="coerce"),
            "Profondeur_Prepa_1": till_depths[0] if n_prepa >= 1 else np.nan,
            "Profondeur_Prepa_2": till_depths[1] if n_prepa >= 2 else np.nan,
            "Date_F1": campaign_day(ferti_doys[0]) if len(ferti_doys) >= 1 else np.nan,
            "Date_F2": campaign_day(ferti_doys[1]) if len(ferti_doys) >= 2 else np.nan,
            "Date_F3": campaign_day(ferti_doys[2]) if len(ferti_doys) >= 3 else np.nan,
            "Date_Recolte": rec_c,
            "Dose_F1": ferti_doses[0] if len(ferti_doses) >= 1 else np.nan,
            "Dose_F2": ferti_doses[1] if len(ferti_doses) >= 2 else np.nan,
            "Dose_F3": ferti_doses[2] if len(ferti_doses) >= 3 else np.nan,
        })

    if not campaigns:
        return None
    cdf = pd.DataFrame(campaigns)
    out = {}
    # Catégorielles : mode (valeur la plus fréquente sur les campagnes).
    for c in ["n_ferti", "nb_prepa"]:
        out[c] = int(cdf[c].mode().iloc[0])
    # Continues : médiane sur les campagnes où la variable est active.
    for c in AGRI_FEATURES[3:]:
        vals = cdf[c].dropna()
        out[c] = float(vals.median()) if len(vals) else np.nan
    return out


def encode_features(rec: dict) -> dict:
    """Vers le schéma feat_* : nb_prepa en index ordinal, défauts pour l'inactif."""
    n_ferti = rec["n_ferti"]
    nb_prepa = rec["nb_prepa"]

    vals = {
        "n_ferti": n_ferti,
        "nb_prepa": nb_prepa,
        "Date_Semis": rec["Date_Semis"],
        "Profondeur_Semis": rec["Profondeur_Semis"],
        "Date_Recolte": rec["Date_Recolte"],
    }
    # Préparation
    vals["Delta_PREPA_Semis"] = rec["Delta_PREPA_Semis"] if nb_prepa >= 1 else DEFAULTS["Delta_PREPA_Semis"]
    vals["Profondeur_Prepa_1"] = rec["Profondeur_Prepa_1"] if nb_prepa >= 1 else DEFAULTS["Profondeur_Prepa_1"]
    vals["Profondeur_Prepa_2"] = rec["Profondeur_Prepa_2"] if nb_prepa >= 2 else DEFAULTS["Profondeur_Prepa_2"]
    # Fertilisation
    for k, name in enumerate(["Date_F1", "Date_F2", "Date_F3"], start=1):
        vals[name] = rec[name] if n_ferti >= k else DEFAULTS[name]
    for k, name in enumerate(["Dose_F1", "Dose_F2", "Dose_F3"], start=1):
        vals[name] = rec[name] if n_ferti >= k else DEFAULTS[name]
    # Sécurité : remplacer tout NaN résiduel par le défaut de la variable.
    for name in AGRI_FEATURES[3:]:
        if pd.isna(vals.get(name, np.nan)):
            vals[name] = DEFAULTS.get(name, 0.0)
    return vals


def main() -> None:
    dataset_path = LOG_DIR / "dataset_metamodel.csv"
    df = pd.read_csv(dataset_path)
    df["si"] = (df["point_idx"].astype(int) // 100)

    folders = sorted(f for f in LOG_DIR.iterdir() if f.is_dir() and (f / "suiviOTParParcelle.csv").exists())
    print(f"Dossiers de runs : {len(folders)}")

    records = []
    for f in folders:
        si = sim_idx_of(f.name)
        ot = pd.read_csv(
            f / "suiviOTParParcelle.csv", sep=";",
            usecols=["annee", "date", "parcelle", "OT", "profondeur[cm]", "FERTI_apportNminReel[kg/ha]"],
        )
        for parcelle, g in ot.groupby("parcelle"):
            rec = reconstruct_parcelle(g)
            if rec is None:
                continue
            row = {"si": si, "parcelle": parcelle, **encode_features(rec)}
            records.append(row)

    recon = pd.DataFrame(records)
    print(f"Itinéraires reconstruits : {len(recon)}")

    merged = df.merge(recon, on=["si", "parcelle"], how="left", suffixes=("", "_new"))
    covered = merged[AGRI_FEATURES[0] + "_new" if (AGRI_FEATURES[0] + "_new") in merged else AGRI_FEATURES[0]]
    n_missing = merged["Date_Semis_new"].isna().sum() if "Date_Semis_new" in merged else 0
    print(f"Lignes sans reconstruction : {n_missing}")

    # Réécrit feat_0..14 à partir des valeurs reconstruites.
    for i, name in enumerate(AGRI_FEATURES):
        src = f"{name}_new" if f"{name}_new" in merged.columns else name
        merged[f"feat_{i}"] = merged[src]

    out = merged[df.columns.drop("si")].copy()
    # Supprime les lignes non reconstruites (itinéraire inconnu).
    out = out.dropna(subset=FEAT_COLS).reset_index(drop=True)

    backup = dataset_path.with_suffix(".csv.bak_doe")
    if not backup.exists():
        shutil.copy2(dataset_path, backup)
        print(f"Sauvegarde de l'original -> {backup.name}")
    out.to_csv(dataset_path, index=False)
    print(f"Dataset réaligné écrit : {dataset_path} ({len(out)} lignes)")


if __name__ == "__main__":
    main()
