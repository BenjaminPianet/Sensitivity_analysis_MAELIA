#!/usr/bin/env python3
"""Lancement serveur des simulations MAELIA/GAMA sur terrainSA (plan SMT).

Version script du notebook `simulations/batch_simulations_smt_terrainSA.ipynb`,
pensée pour tourner sans interface sur un serveur : arguments CLI, chemins GAMA
et MAELIA configurables, journalisation fichier, reprise après interruption, et
export final de `dataset_metamodel.csv`.

Placement recommandé : à la racine `simulations/` du dépôt
(`Sensitivity_analysis_MAELIA/simulations/run_terrainSA_batch.py`), à côté de
`build_terrainSA_project.py`.

Usage typique sur le serveur
----------------------------
    # 1. Vérifier la config et générer DOE + dateDose + XML sans lancer GAMA :
    python run_terrainSA_batch.py --dry-run \
        --gama-headless /opt/gama/headless/gama-headless.sh \
        --maelia-root ~/Workspace_GAMA/MAELIA

    # 2. Lancer réellement, en tâche de fond, avec reprise possible :
    nohup python run_terrainSA_batch.py \
        --gama-headless /opt/gama/headless/gama-headless.sh \
        --maelia-root ~/Workspace_GAMA/MAELIA \
        > batch_terrainSA.out 2>&1 &

    # 3. Si le job a été coupé, relancer la même commande : les runs déjà
    #    terminés sont détectés et sautés (reprise par défaut).

    # 4. Récupérer seulement les sorties, sans relancer GAMA :
    python run_terrainSA_batch.py --collect-only

Les sorties d'intérêt en fin de course :
    <repo>/simulations/log_terrainSA/dataset_metamodel.csv
    <repo>/simulations/log_terrainSA/dataset_metamodel_features.csv
plus le détail par run dans le même dossier.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("terrainSA")


def setup_logging(log_file: Path, verbose: bool) -> None:
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Localisation du projet
# ─────────────────────────────────────────────────────────────────────────────
def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "requirements.txt").exists() and (candidate / "maelia_sa_pipeline").exists():
            return candidate
    raise RuntimeError(
        "Racine du dépôt introuvable. Lance le script depuis le dépôt "
        "Sensitivity_analysis_MAELIA ou passe --project-root."
    )


def read_launcher_parameters(model_path: Path) -> dict[str, str]:
    """Variables exposées comme paramètres headless par le launcher MAELIA courant."""
    import re

    text = Path(model_path).read_text(errors="replace")
    params: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        match = re.search(r"parameter\s+(['\"])(.*?)\1\s+var:\s*([A-Za-z0-9_]+)", line)
        if match:
            params[match.group(3)] = match.group(2)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Espace de conception SMT (identique au notebook)
# ─────────────────────────────────────────────────────────────────────────────
FERTILIZER_TYPE = "AN"
PREPA_TYPE_1 = "travail_sol_1"
PREPA_TYPE_2 = "une_reprise"
SEMIS_TYPE = "semis"
PREPA_DEPTH_MIN_CM = 1.0
PREPA_DEPTH_MAX_CM = 25.0
SEMIS_DEPTH_MIN_CM = 1.0
SEMIS_DEPTH_MAX_CM = 4.0

SMT_FEATURES = [
    "n_ferti", "nb_prepa",
    "Date_Semis", "Delta_PREPA_Semis", "Profondeur_Semis",
    "Profondeur_Prepa_1", "Profondeur_Prepa_2",
    "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
    "Dose_F1", "Dose_F2", "Dose_F3",
]

DEFAULT_FLOATS = [
    75.0, -20.0, 3.0, 15.0, 10.0,
    150.0, 230.0, 280.0, 350.0,
    1.699, 1.699, 1.699,
]


def build_design_space():
    """Construit l'espace ADSG hiérarchique. Import local pour un --help sans SMT."""
    from smt.design_space import FloatVariable, OrdinalVariable
    from smt_design_space_ext import AdsgDesignSpaceImpl

    ds = AdsgDesignSpaceImpl(
        design_variables=[
            OrdinalVariable(["0_ferti", "1_ferti", "2_ferti", "3_ferti"]),  # 0: n_ferti
            OrdinalVariable(["0_prepa", "1_prepa", "2_prepa"]),             # 1: nb_prepa
            FloatVariable(45, 106),                                        # 3: Date_Semis
            FloatVariable(-44, -4),                                        # 4: Delta_PREPA_Semis
            FloatVariable(SEMIS_DEPTH_MIN_CM, SEMIS_DEPTH_MAX_CM),          # 5: Profondeur_Semis
            FloatVariable(PREPA_DEPTH_MIN_CM, PREPA_DEPTH_MAX_CM),          # 6: Profondeur_Prepa_1
            FloatVariable(PREPA_DEPTH_MIN_CM, PREPA_DEPTH_MAX_CM),          # 7: Profondeur_Prepa_2
            FloatVariable(106, 323),                                       # 8: Date_F1
            FloatVariable(210, 323),                                       # 9: Date_F2
            FloatVariable(240, 323),                                       # 10: Date_F3
            FloatVariable(323, 384),                                       # 11: Date_Recolte
            FloatVariable(0, 2),                                        # 12: Dose_F1
            FloatVariable(0, 2),                                        # 13: Dose_F2
            FloatVariable(0, 2),                                        # 14: Dose_F3
        ]
    )
    ds.declare_decreed_var(decreed_var=3, meta_var=1, meta_value=["1_prepa", "2_prepa"])
    ds.declare_decreed_var(decreed_var=5, meta_var=1, meta_value=["1_prepa", "2_prepa"])
    ds.declare_decreed_var(decreed_var=6, meta_var=1, meta_value=["2_prepa"])
    ds.declare_decreed_var(decreed_var=7, meta_var=0, meta_value=["1_ferti", "2_ferti", "3_ferti"])
    ds.declare_decreed_var(decreed_var=11, meta_var=0, meta_value=["1_ferti", "2_ferti", "3_ferti"])
    ds.declare_decreed_var(decreed_var=8, meta_var=0, meta_value=["2_ferti", "3_ferti"])
    ds.declare_decreed_var(decreed_var=12, meta_var=0, meta_value=["2_ferti", "3_ferti"])
    ds.declare_decreed_var(decreed_var=9, meta_var=0, meta_value=["3_ferti"])
    ds.declare_decreed_var(decreed_var=13, meta_var=0, meta_value=["3_ferti"])
    return ds


def parse_smt_row(row_data):
    parsed = {"n_ferti": 0, "nb_prepa": 0, "floats": []}
    values = list(row_data.values()) if isinstance(row_data, dict) else row_data
    for val in values:
        if isinstance(val, (float, int, np.floating, np.integer)):
            parsed["floats"].append(float(val))
        elif isinstance(val, str):
            if val in ["0_ferti", "1_ferti", "2_ferti", "3_ferti"]:
                parsed["n_ferti"] = int(val[0])
            elif val in ["0_prepa", "1_prepa", "2_prepa"]:
                parsed["nb_prepa"] = int(val[0])
    return parsed


def normalized_floats(parsed):
    fl = list(parsed["floats"])
    for default in DEFAULT_FLOATS[len(fl):]:
        fl.append(default)
    return fl[: len(DEFAULT_FLOATS)]


def ordered_fertilisation_dates(parsed, fl):
    dates = {"f1": None, "f2": None, "f3": None}
    if parsed["n_ferti"] >= 1:
        dates["f1"] = int(round(fl[5]))
    if parsed["n_ferti"] >= 2:
        dates["f2"] = max(int(round(fl[6])), dates["f1"] + 1)
    if parsed["n_ferti"] >= 3:
        dates["f3"] = max(int(round(fl[7])), dates["f2"] + 1)
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# Génération / persistance du plan DOE (reproductible)
# ─────────────────────────────────────────────────────────────────────────────
def get_or_build_doe(design_space, n_doe: int, seed: int, cache_path: Path) -> np.ndarray:
    """Charge le DOE persisté si compatible, sinon l'échantillonne et le sauvegarde.

    La persistance garantit que `point_idx -> paramètres` reste identique entre
    deux lancements (indispensable pour la reprise et pour l'analyse).
    """
    if cache_path.exists():
        xt = np.load(cache_path, allow_pickle=True)
        if xt.shape[0] == n_doe:
            log.info("DOE rechargé depuis %s (%d points).", cache_path.name, xt.shape[0])
            return xt
        log.warning(
            "DOE en cache (%d points) incompatible avec N_DOE=%d : régénération.",
            xt.shape[0], n_doe,
        )

    log.info("Génération de %d points LHS hiérarchiques (SMT ADSG), seed=%d…", n_doe, seed)
    np.random.seed(seed)  # fixe la graine : _sample_valid_x n'expose pas de seed
    result = design_space._sample_valid_x(n_doe)
    xt = result[0]
    np.save(cache_path, xt)
    log.info("DOE sauvegardé dans %s (matrice %d × %d).", cache_path.name, *xt.shape)
    return xt


# ─────────────────────────────────────────────────────────────────────────────
# Génération des fichiers dateDose (identique au notebook)
# ─────────────────────────────────────────────────────────────────────────────
COLS = ["Trait", "nom_MAELIA", "plant_sem", "annee_deb", "id_operation",
        "ordre_op", "DATE", "PROF", "TYPE", "DOSE", "TEMPS"]
HARVEST_BUFFER_DAYS = 7
MIN_DAYS_LAST_OP_TO_HARVEST = 1
CROP = "ble"
CROP_MAELIA = "BTH"


def days_in_year(year):
    return (date(year + 1, 8, 1) - date(year, 8, 1)).days


def doy_to_date_str(year, doy):
    return (date(year, 8, 1) + timedelta(days=int(doy) - 1)).strftime("%d/%m/%Y")


def parse_date_str(value):
    return pd.to_datetime(value, dayfirst=True).date()


def safe_f(val, default=50.0):
    try:
        v = float(val)
        return default if v != v else v
    except (TypeError, ValueError):
        return default


def validate_datedose_rows(rows):
    order = {"RECOLTE": 0, "PREPA": 1, "SEMIS": 2, "FERTI_Min": 3}
    by_parcelle: dict = {}
    duplicate_keys: dict = {}
    issues: list[str] = []
    for row in rows:
        by_parcelle.setdefault(row["Trait"], []).append(row)
        key = (row["Trait"], row["annee_deb"], row["id_operation"], row["DATE"])
        duplicate_keys[key] = duplicate_keys.get(key, 0) + 1

    for (trait, campagne, op, day), n in duplicate_keys.items():
        if n > 1:
            issues.append(f"{trait} campagne {campagne}: {n} opérations {op} le même jour ({day})")

    for parcelle_id, parcelle_ops in by_parcelle.items():
        culture_en_place = False
        sorted_ops = sorted(
            parcelle_ops,
            key=lambda r: (parse_date_str(r["DATE"]), order.get(r["id_operation"], 99), str(r.get("ordre_op", ""))),
        )
        for row in sorted_ops:
            op = row["id_operation"]
            day = row["DATE"]
            campagne = int(row["annee_deb"])
            op_date = parse_date_str(day)
            campaign_start = date(campagne, 8, 1)
            if op_date < campaign_start:
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


def generate_ops_for_parcelle(design_space, xt, point_idx, parcelle_id, years):
    decoded = design_space.decode_values(xt[point_idx: point_idx + 1])
    p = parse_smt_row(decoded[0])
    fl = [safe_f(v, default=d) for v, d in zip(normalized_floats(p), DEFAULT_FLOATS)]

    ops = []
    for year in years:
        semi_doy = max(1, int(round(fl[0])))
        prepa_doy = max(1, int(round(semi_doy + fl[1]))) if p["nb_prepa"] >= 1 else None
        ferti_dates = ordered_fertilisation_dates(p, fl)

        next_entry_doy = prepa_doy if prepa_doy is not None else semi_doy
        latest_rec_doy = days_in_year(year) + next_entry_doy - HARVEST_BUFFER_DAYS
        max_last_op_doy = latest_rec_doy - MIN_DAYS_LAST_OP_TO_HARVEST

        ferti_events = []
        if ferti_dates["f1"] is not None:
            ferti_events.append((ferti_dates["f1"], 10 ** fl[9]))
        if ferti_dates["f2"] is not None:
            ferti_events.append((ferti_dates["f2"], 10 ** fl[10]))
        if ferti_dates["f3"] is not None:
            ferti_events.append((ferti_dates["f3"], 10 ** fl[11]))
        ferti_events = [e for e in ferti_events if e[0] <= max_last_op_doy]

        last_doy = ferti_events[-1][0] if ferti_events else semi_doy
        rec_doy = max(int(round(fl[8])), last_doy + MIN_DAYS_LAST_OP_TO_HARVEST)
        rec_doy = min(rec_doy, latest_rec_doy)
        if rec_doy <= last_doy:
            raise ValueError(
                f"Calendrier impossible pour {parcelle_id} campagne {year}: "
                f"dernière op DOY {last_doy}, récolte max DOY {latest_rec_doy}"
            )

        base = {"Trait": parcelle_id, "nom_MAELIA": CROP_MAELIA, "plant_sem": CROP, "annee_deb": year}

        if p["nb_prepa"] >= 1:
            prepa_dates = [(prepa_doy, "", fl[3], PREPA_TYPE_1)]
            if p["nb_prepa"] == 2:
                prepa2_doy = min(max(2, prepa_doy), semi_doy - 1)
                prepa1_doy = max(1, min(prepa_doy - 7, prepa2_doy - 1))
                if prepa1_doy >= prepa2_doy:
                    raise ValueError(
                        f"Impossible de placer deux PREPA distinctes avant le semis "
                        f"pour {parcelle_id} campagne {year}: prepa={prepa_doy}, semis={semi_doy}"
                    )
                prepa_dates = [
                    (prepa1_doy, "", fl[3], PREPA_TYPE_1),
                    (prepa2_doy, "2", fl[4], PREPA_TYPE_2),
                ]
            for op_doy, ordre_op, depth, op_type in prepa_dates:
                ops.append({**base, "id_operation": "PREPA", "ordre_op": ordre_op,
                            "DATE": doy_to_date_str(year, op_doy),
                            "PROF": str(round(depth, 1)), "TYPE": op_type, "DOSE": ""})

        ops.append({**base, "id_operation": "SEMIS", "ordre_op": "",
                    "DATE": doy_to_date_str(year, semi_doy),
                    "PROF": str(round(fl[2], 1)), "TYPE": SEMIS_TYPE, "DOSE": ""})

        for ordre, (doy, dose) in enumerate(ferti_events, 1):
            ops.append({**base, "id_operation": "FERTI_Min", "ordre_op": str(ordre),
                        "DATE": doy_to_date_str(year, doy), "PROF": "",
                        "TYPE": FERTILIZER_TYPE, "DOSE": str(round(dose, 1))})

        ops.append({**base, "id_operation": "RECOLTE", "ordre_op": "",
                    "DATE": doy_to_date_str(year, rec_doy), "PROF": "", "TYPE": "", "DOSE": ""})

    for op in ops:
        op.setdefault("TEMPS", "")
    return ops


def generate_all_datedose(design_space, xt, variants_dir, all_parcelles, annees, n_simu):
    log.info("Génération de %d fichiers dateDose distribués…", n_simu)
    for i in range(n_simu):
        path = variants_dir / f"dateDose_smt_{i:03d}.csv"
        all_ops = []
        for j, parcelle_id in enumerate(all_parcelles):
            point_idx = i * len(all_parcelles) + j
            all_ops.extend(generate_ops_for_parcelle(design_space, xt, point_idx, parcelle_id, annees))
        issues = validate_datedose_rows(all_ops)
        if issues:
            preview = "\n  - ".join(issues[:10])
            raise ValueError(f"Calendrier dateDose invalide pour simulation {i:03d}:\n  - {preview}")
        pd.DataFrame(all_ops, columns=COLS).to_csv(path, sep=";", index=False)
        if i % 10 == 0 or i == n_simu - 1:
            log.info("  dateDose_smt_%03d : %d opérations", i, len(all_ops))

    # Contrôle anti-régression : semis jamais avant le 1er août de sa campagne.
    check = pd.read_csv(variants_dir / "dateDose_smt_000.csv", sep=";")
    semis = pd.to_datetime(check.loc[check["id_operation"] == "SEMIS", "DATE"], dayfirst=True, errors="coerce")
    camp = pd.to_datetime(check.loc[check["id_operation"] == "SEMIS", "annee_deb"].astype(int).astype(str) + "-08-01", errors="coerce")
    if (semis.reset_index(drop=True) < camp.reset_index(drop=True)).any():
        raise ValueError("Dates de semis invalides : au moins un semis avant le 1er août de sa campagne.")
    log.info("Fichiers dateDose écrits dans : %s", variants_dir)


# ─────────────────────────────────────────────────────────────────────────────
# XML headless + lancement GAMA (identique au notebook, paramétré)
# ─────────────────────────────────────────────────────────────────────────────
def make_xml(cfg, nom_simu, datedose_path):
    datedose_path = Path(datedose_path)
    try:
        datedose_for_gama = "/" + datedose_path.relative_to(cfg["terrain_dir"]).as_posix()
    except ValueError:
        datedose_for_gama = datedose_path.as_posix()

    root = ET.Element("Experiment_plan")
    sim = ET.SubElement(root, "Simulation")
    sim.set("experiment", "simulationBase")
    sim.set("finalStep", str(cfg["final_step"]))
    sim.set("id", "0")
    sim.set("seed", "1.0")
    sim.set("sourcePath", str(cfg["model_path"]))

    params = ET.SubElement(sim, "Parameters")
    launcher = cfg["launcher_parameters"]
    skipped = []

    def p(name, ptype, value, var):
        if var not in launcher:
            skipped.append(var)
            return
        el = ET.SubElement(params, "Parameter")
        el.set("name", name)
        el.set("type", ptype)
        el.set("value", str(value))
        el.set("var", var)

    p("executerSurCluster: ", "BOOLEAN", "false", "executerSurCluster")
    p("cheminRacineMaelia", "STRING", cfg["chemin_racine"], "cheminRacineMaelia")
    p("cheminModeleVersDonnees", "STRING", cfg["chemin_modele_vers_donnees"], "cheminModeleVersDonnees")
    p("cheminSorties", "STRING", str(cfg["gama_output_dir"]), "cheminRelatifDuDossierDeSortieDeSimulation")
    p("anneeDebutSimulation : ", "INT", cfg["annee_debut"], "anneeDebutSimulation")
    p("nbAnneesSimulation : ", "INT", cfg["nb_annees"], "nbAnneesSimulation")
    p("nomSimulation : ", "STRING", nom_simu, "nomSimulation")
    p("nomDecoupageZonePourLectureFichiers : ", "STRING", cfg["nom_decoupage"], "nomDecoupageZonePourLectureFichiers")
    p("modeVerbeux :", "BOOLEAN", "false", "verboseMode")
    p("simulationSurParcelle : ", "BOOLEAN", "false", "executerUneSeuleParcelle")
    p("idParcelleASimuler : ", "STRING", cfg["all_parcelles"][0], "nomParcelleAffichee")
    p("executerModeleAgricole : ", "BOOLEAN", "true", "executerModeleAgricole")
    p("nomChoixAssolement : ", "STRING", "DateDose", "nomChoixAssolement")
    p("fichierDateDose : ", "STRING", datedose_for_gama, "fichierDateDose")
    p("avecContrainteDeMainOeuvre : ", "BOOLEAN", "false", "avecContrainteDeMainOeuvre")
    p("isIrrigationSimulee : ", "BOOLEAN", "false", "isIrrigationSimulee")
    p("executerModeleHydrographique : ", "BOOLEAN", "false", "executerModeleHydrographique")
    p("executerModeleNormatif : ", "BOOLEAN", "false", "executerModeleNormatif")
    p("executerEcritureFichiers : ", "BOOLEAN", "true", "executerEcritureFichiers")
    p("sorties eau", "BOOLEAN", "true", "sorties_eau")
    p("sorties azote", "BOOLEAN", "true", "sorties_azote")
    p("sorties carbone et GES", "BOOLEAN", "true", "sorties_carboneGES")
    p("Sortie N_Cstock_Parcelles", "BOOLEAN", "true", "N_Cstock_Parcelles")
    p("Sortie N_lixi_Parcelles", "BOOLEAN", "true", "N_lixi_Parcelles")

    if skipped and nom_simu == "smt_000":
        log.warning("Paramètres XML ignorés (absents du launcher) : %s", ", ".join(sorted(set(skipped))))

    ET.SubElement(sim, "Outputs")
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding="UTF-8", xml_declaration=True)
    xml_str = buf.getvalue().decode("UTF-8")
    return xml_str.replace(
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
    )


def ensure_gama_buffering_preferences(cfg):
    """Fixe les préférences save/write de GAMA en per_simulation avant la série."""
    strategy = cfg["buffering_strategy"]
    xml_dir = cfg["xml_dir"]
    model_path = xml_dir / "set_gama_buffering_preferences.gaml"
    model_path.write_text(
        f'model set_gama_buffering_preferences\n\n'
        f'global {{\n    init {{\n'
        f'        gama.pref_save_buffering_strategy <- "{strategy}";\n'
        f'        gama.pref_write_buffering_strategy <- "{strategy}";\n'
        f'        write "GAMA buffering set to {strategy}" buffering: "{strategy}";\n'
        f'    }}\n}}\n\n'
        f'experiment set_buffering_preferences type: gui {{\n}}\n'
    )
    root = ET.Element("Experiment_plan")
    sim = ET.SubElement(root, "Simulation")
    sim.set("experiment", "set_buffering_preferences")
    sim.set("finalStep", "1")
    sim.set("id", "0")
    sim.set("seed", "1.0")
    sim.set("sourcePath", str(model_path))
    ET.SubElement(sim, "Parameters")
    ET.SubElement(sim, "Outputs")
    try:
        ET.indent(root, space="  ")
    except AttributeError:
        pass
    buf = io.BytesIO()
    ET.ElementTree(root).write(buf, encoding="UTF-8", xml_declaration=True)
    xml_path = xml_dir / "set_gama_buffering_preferences.xml"
    xml_path.write_text(
        buf.getvalue().decode("UTF-8").replace(
            "<?xml version='1.0' encoding='UTF-8'?>",
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        )
    )
    pref_ws = xml_dir / "ws_buffering_preferences"
    pref_ws.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(cfg["gama_headless"]), str(xml_path), str(pref_ws)]
    with open(cfg["gama_log"], "a") as flog:
        flog.write(f"\n{'=' * 62}\n[buffering-preferences] {time.strftime('%H:%M:%S')}\n")
        proc = subprocess.run(cmd, stdout=flog, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        log.warning("Pré-configuration du buffering GAMA échouée (code %d), on continue.", proc.returncode)
    else:
        log.info("Buffering GAMA demandé : %s", strategy)


# ─────────────────────────────────────────────────────────────────────────────
# Validation / collecte des logs
# ─────────────────────────────────────────────────────────────────────────────
# Fichiers exigés pour considérer un run GAMA *frais* comme réussi (contrôle
# complet sur la sortie brute, avant tout élagage).
REQUIRED_LOGS = ["sorties_CN.csv", "sorties_GES.csv", "suiviOTParParcelle.csv"]

# Fichiers réellement exploités pour construire dataset_metamodel.csv.
# sorties_CN.csv -> N_lixi, dCorg ; suiviOTParParcelle.csv -> rdt ;
# corresponsanceIlotZoneMeteo.csv -> zone météo.
DATASET_FILES = ["sorties_CN.csv", "suiviOTParParcelle.csv", "corresponsanceIlotZoneMeteo.csv"]

# Whitelist conservée par défaut dans les runs mirrorés : les fichiers du
# dataset + un petit fichier de provenance. Tout le reste est élagué.
DEFAULT_KEEP_FILES = DATASET_FILES + ["simulationParameters.txt"]

FAILURE_MARKERS = [
    "The model couldn't be compiled", "Model didn't compile",
    "ERREUR LORS DE L'INITIALISATION", "ERREUR LORS DE LA SIMULATION",
    "Fichier date/dose inexistant",
]


def output_dir_ok(path: Path | None) -> bool:
    return bool(path and path.exists() and all((path / n).exists() for n in REQUIRED_LOGS))


def missing_required_logs(path: Path | None):
    if path is None or not path.exists():
        return ["dossier de sortie introuvable"]
    return [n for n in REQUIRED_LOGS if not (path / n).exists()]


def find_output_dir(cfg, nom_simu, ts_before):
    prefix = f"{cfg['nom_decoupage']}_{nom_simu}_"
    log_dir = cfg["gama_output_dir"]
    for _ in range(20):
        matches = [
            p for p in log_dir.glob(f"{cfg['nom_decoupage']}_*")
            if p.is_dir() and p.stat().st_mtime >= ts_before
            and (p.name.startswith(prefix) or nom_simu in p.name)
        ]
        if matches:
            return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]
        time.sleep(1)
    return None


def mirrored_run_ok(path: Path | None) -> bool:
    """Un run *conservé* est exploitable s'il garde les fichiers du dataset."""
    return bool(path and path.exists() and all((path / n).exists() for n in DATASET_FILES))


def prune_run_dir(path: Path, keep_files) -> int:
    """Supprime d'un dossier de run tout ce qui n'est pas dans keep_files.

    keep_files=None => ne rien élaguer. Retourne le nombre d'octets libérés.
    Les fichiers du dataset sont toujours protégés.
    """
    if keep_files is None:
        return 0
    protected = set(keep_files) | set(DATASET_FILES)
    freed = 0
    for item in path.iterdir():
        if item.name in protected:
            continue
        if item.is_file():
            freed += item.stat().st_size
            item.unlink()
        elif item.is_dir():
            freed += sum(p.stat().st_size for p in item.rglob("*") if p.is_file())
            shutil.rmtree(item)
    return freed


def mirror_output_dir(cfg, out_dir: Path) -> Path:
    """Copie un run validé vers project_log_dir en ne gardant que la whitelist."""
    dest = cfg["project_log_dir"] / out_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    keep = cfg["keep_files"]
    if keep is None:  # --keep-all-outputs : copie intégrale
        shutil.copytree(out_dir, dest)
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    effective = list(dict.fromkeys(list(keep) + DATASET_FILES))
    for name in effective:
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    return dest


def existing_project_run(cfg, nom_simu) -> Path | None:
    """Cherche un run déjà mirroré et exploitable pour ce nom_simu (reprise)."""
    prefix = f"{cfg['nom_decoupage']}_{nom_simu}_"
    candidates = [
        d for d in cfg["project_log_dir"].glob(f"{cfg['nom_decoupage']}_*")
        if d.is_dir() and (d.name.startswith(prefix) or f"_{nom_simu}_" in d.name) and mirrored_run_ok(d)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def run_simulation(cfg, nom_simu, xml_path, gama_ws) -> Path | None:
    gama_ws.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(cfg["gama_headless"]), str(xml_path), str(gama_ws)]
    t0 = time.time()
    ts_before = t0 - 0.5
    with open(cfg["gama_log"], "a") as flog:
        flog.write(f"\n{'=' * 62}\n[{nom_simu}] {time.strftime('%H:%M:%S')}\n")
        flog.write(f"Commande: {' '.join(cmd)}\n{'=' * 62}\n")
        proc = subprocess.run(cmd, stdout=flog, stderr=subprocess.STDOUT, text=True)

    elapsed = time.time() - t0
    out_dir = find_output_dir(cfg, nom_simu, ts_before) if proc.returncode == 0 else None
    console = (gama_ws / "console-outputs-0.txt")
    ctext = console.read_text(errors="replace") if console.exists() else ""
    missing = missing_required_logs(out_dir)
    ok = proc.returncode == 0 and not any(m in ctext for m in FAILURE_MARKERS) and not missing
    if ok:
        project_dir = mirror_output_dir(cfg, out_dir)
        log.info("  [OK] %s en %.1fs → %s", nom_simu, elapsed, project_dir.name)
        if cfg["purge_gama_runs"]:
            # Élimine la copie brute côté GAMA (models/main/log) et le workspace
            # headless : on garde uniquement la copie élaguée côté projet.
            shutil.rmtree(out_dir, ignore_errors=True)
            shutil.rmtree(gama_ws, ignore_errors=True)
        return project_dir
    log.error("  [ERREUR code %d] %s en %.1fs — logs manquants: %s",
              proc.returncode, nom_simu, elapsed, ", ".join(missing))
    if any(m in ctext for m in FAILURE_MARKERS):
        log.error("  Console GAMA : erreur de compilation ou d'initialisation détectée.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Lecture des sorties + export metamodel (identique au notebook)
# ─────────────────────────────────────────────────────────────────────────────
def read_dbf_ilot_sol(dbf_path: Path) -> dict:
    import struct

    d = Path(dbf_path).read_bytes()
    n = struct.unpack("<I", d[4:8])[0]
    hs = struct.unpack("<H", d[8:10])[0]
    rs = struct.unpack("<H", d[10:12])[0]
    flds = []
    off = 32
    while d[off] != 0x0D:
        name = d[off:off + 11].split(b"\x00")[0].decode("latin1")
        flds.append((name, d[off + 16]))
        off += 32
    off = hs
    out = {}
    for _ in range(n):
        if d[off] == 0x2A:
            off += rs
            continue
        off += 1
        row = {}
        for nm, ln in flds:
            row[nm] = d[off:off + ln].decode("latin1").strip()
            off += ln
        out[row["ID_ILOT"]] = row["ID_SOL"]
    return out


def _zone(parcelle):
    for z in ("beauce", "oceanique", "sudouest"):
        if z in parcelle:
            return z
    return "autre"


def read_simulation_outputs(out_dir: Path, ilot_sol: dict) -> pd.DataFrame:
    f_cn = out_dir / "sorties_CN.csv"
    if not f_cn.exists():
        return pd.DataFrame()
    df = pd.read_csv(f_cn, sep=";",
                     usecols=["annee", "parcelle", "couvert", "N_lixivie[kgN/ha]", "delta_Corg[kgC/ha]"])
    df = df[df["couvert"].notna() & (df["couvert"] != "solnu")].drop(columns="couvert")
    df = df.rename(columns={"N_lixivie[kgN/ha]": "N_lixi", "delta_Corg[kgC/ha]": "dCorg"})

    f_ot = out_dir / "suiviOTParParcelle.csv"
    if f_ot.exists():
        df_ot = pd.read_csv(f_ot, sep=";", usecols=["annee", "parcelle", "OT", "RECOLTE_rendement[t/ha]"])
        df_ot = (df_ot[df_ot["OT"] == "RECOLTE"].drop(columns="OT")
                 .rename(columns={"RECOLTE_rendement[t/ha]": "rdt"}))
        df = df.merge(df_ot, on=["annee", "parcelle"], how="left")

    df["zone"] = df["parcelle"].apply(_zone)

    corr_path = out_dir / "corresponsanceIlotZoneMeteo.csv"
    corr = {}
    if corr_path.exists():
        cdf = pd.read_csv(corr_path, sep=";")
        corr = dict(zip(cdf["ilot"], cdf["zoneMeteo"].astype(int)))

    def ilot_from_parcelle(p):
        if str(p).startswith("beauce_5_sa_"):
            return "beauce_5"
        return "_".join(str(p).rsplit("_", 1)[:-1])

    df["sol_type"] = df["parcelle"].apply(lambda p: ilot_sol.get(ilot_from_parcelle(p)))
    df["zone_meteo"] = df["parcelle"].apply(lambda p: corr.get(ilot_from_parcelle(p)))
    return df


def collect_and_export(cfg, xt, results: dict[str, Path | None]):
    dbf_path = cfg["project_includes_root"] / cfg["nom_decoupage"] / "modeleAgricole/ilots/dansZone/ilots.dbf"
    ilot_sol = read_dbf_ilot_sol(dbf_path)

    all_parcelles = cfg["all_parcelles"]
    parc_to_idx = {p: k for k, p in enumerate(all_parcelles)}
    records = []
    n_ok = 0
    for nom_simu, out_dir in sorted(results.items()):
        if not (out_dir and out_dir.exists()):
            continue
        df = read_simulation_outputs(out_dir, ilot_sol)
        if df.empty:
            log.warning("%s : sorties_CN.csv absent ou sans couvert cultivé.", nom_simu)
            continue
        sim_idx = int(nom_simu.split("_")[-1])
        df["point_idx"] = sim_idx * len(all_parcelles) + df["parcelle"].map(parc_to_idx)
        df_parc = (df.groupby(["parcelle", "point_idx", "zone", "sol_type"])
                   .agg({"N_lixi": "mean", "dCorg": "mean", "rdt": "mean"}).reset_index())
        records.append(df_parc)
        n_ok += 1

    if not records:
        log.error("Aucune donnée exploitable collectée.")
        return None

    df_full = pd.concat(records, ignore_index=True)
    xt_subset = xt[df_full["point_idx"].values]
    for j in range(xt.shape[1]):
        df_full[f"feat_{j}"] = xt_subset[:, j]

    zone_map = {z: i for i, z in enumerate(sorted(df_full["zone"].unique()))}
    soil_map = {s: i for i, s in enumerate(sorted(df_full["sol_type"].unique()))}
    df_full["Milieu (Climat)"] = df_full["zone"].map(zone_map)
    df_full["Milieu (Sol)"] = df_full["sol_type"].map(soil_map)

    feat_names = [f"feat_{j}" for j in range(xt.shape[1])]
    export_cols = feat_names + ["Milieu (Climat)", "Milieu (Sol)", "N_lixi", "dCorg", "rdt", "point_idx", "parcelle"]
    df_export = df_full[export_cols]

    out_csv = cfg["project_log_dir"] / "dataset_metamodel.csv"
    df_export.to_csv(out_csv, index=False)
    pd.DataFrame({"colonne": feat_names, "parametre": SMT_FEATURES[: len(feat_names)]}).to_csv(
        cfg["project_log_dir"] / "dataset_metamodel_features.csv", index=False
    )
    log.info("Export : %s (%d points, %d runs OK)", out_csv, len(df_export), n_ok)
    return out_csv


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def _parse_keep_files(keep_outputs: str | None) -> list[str]:
    """Convertit la valeur CLI --keep-outputs en liste; None/'' => whitelist par défaut."""
    if not keep_outputs:
        return list(DEFAULT_KEEP_FILES)
    names = [n.strip() for n in keep_outputs.split(",") if n.strip()]
    # Les fichiers du dataset sont toujours conservés, même absents de la liste.
    return list(dict.fromkeys(names + DATASET_FILES))


def prune_existing_runs(cfg) -> None:
    """Élague tous les runs déjà présents dans project_log_dir selon la whitelist."""
    keep = cfg["keep_files"]
    if keep is None:
        log.info("--keep-all-outputs actif : aucun élagage effectué.")
        return
    runs = sorted(d for d in cfg["project_log_dir"].glob(f"{cfg['nom_decoupage']}_smt_*") if d.is_dir())
    if not runs:
        log.info("Aucun run à élaguer dans %s.", cfg["project_log_dir"])
        return
    total_freed = 0
    for d in runs:
        total_freed += prune_run_dir(d, keep)
    log.info("Élagage terminé : %d run(s), %.1f Mo libérés.", len(runs), total_freed / 1e6)
    log.info("Fichiers conservés par run : %s", ", ".join(_parse_keep_files(",".join(keep))))


def build_config(args) -> dict:
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root()
    simulations_dir = project_root / "simulations"
    sys.path.insert(0, str(simulations_dir))

    maelia_root = Path(args.maelia_root or os.environ.get("MAELIA_ROOT")
                       or (Path.home() / "Workspace_GAMA" / "MAELIA")).expanduser()
    gama_headless = Path(args.gama_headless or os.environ.get("GAMA_HEADLESS") or "").expanduser()

    model_path = maelia_root / "models/main/launcherBase.gaml"
    annee_debut = args.annee_debut
    nb_annees = args.nb_annees
    final_step = (date(annee_debut + nb_annees + 1, 1, 1) - date(annee_debut, 1, 1)).days

    cfg = {
        "project_root": project_root,
        "simulations_dir": simulations_dir,
        "maelia_root": maelia_root,
        "gama_headless": gama_headless,
        "model_path": model_path,
        "nom_decoupage": "terrainSA",
        "annee_debut": annee_debut,
        "nb_annees": nb_annees,
        "annees": list(range(annee_debut, annee_debut + nb_annees)),
        "final_step": final_step,
        "chemin_racine": str(maelia_root) + "/",
        "project_includes_root": simulations_dir / "gama_includes",
        "gama_output_dir": maelia_root / "models/main/log",
        "project_log_dir": simulations_dir / "log_terrainSA",
        "xml_dir": Path(args.xml_dir).expanduser(),
        "gama_log": Path(args.gama_log).expanduser(),
        "buffering_strategy": "per_simulation",
        "target_n_doe": args.n_doe,
        "clones_per_run": 100,
        "seed": args.seed,
        "doe_cache": simulations_dir / "doe_matrix_terrainSA.npy",
        "keep_files": None if args.keep_all_outputs else _parse_keep_files(args.keep_outputs),
        "purge_gama_runs": args.purge_gama_runs,
    }
    cfg["chemin_modele_vers_donnees"] = str(cfg["project_includes_root"]) + "/"
    return cfg


def prepare_terrain_and_plan(cfg, args):
    from build_terrainSA_project import ensure_terrain_sa, validate_terrain_sa

    log.info("Construction / vérification de terrainSA…")
    terrain_dir = ensure_terrain_sa(
        maelia_root=cfg["maelia_root"],
        project_includes_root=cfg["project_includes_root"],
        force=args.rebuild_terrain,
        n_clones=cfg["clones_per_run"],
    )
    validate_terrain_sa(terrain_dir, n_clones=cfg["clones_per_run"])
    cfg["terrain_dir"] = terrain_dir
    cfg["variants_dir"] = terrain_dir / "modeleAgricole/agriculteurs/variants_SMT"
    cfg["bloc_donnees"] = terrain_dir / "modeleAgricole/blocsDonnees.csv"

    # Liste des parcelles clonées
    parcs = []
    with open(cfg["bloc_donnees"]) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(";") if p.strip()]
            if len(parts) >= 2:
                parcs.extend(parts[1:])
    all_parcelles = sorted(p for p in parcs if p.startswith("beauce_5_sa_"))[: cfg["clones_per_run"]]
    if len(all_parcelles) != cfg["clones_per_run"]:
        raise ValueError(f"terrainSA doit contenir {cfg['clones_per_run']} clones, trouvé {len(all_parcelles)}")
    cfg["all_parcelles"] = all_parcelles

    n_simu = int(np.ceil(cfg["target_n_doe"] / len(all_parcelles)))
    cfg["n_simu"] = n_simu
    cfg["n_doe"] = n_simu * len(all_parcelles)

    cfg["launcher_parameters"] = read_launcher_parameters(cfg["model_path"])

    cfg["xml_dir"].mkdir(parents=True, exist_ok=True)
    cfg["variants_dir"].mkdir(parents=True, exist_ok=True)
    cfg["project_log_dir"].mkdir(parents=True, exist_ok=True)

    design_space = build_design_space()
    xt = get_or_build_doe(design_space, cfg["n_doe"], cfg["seed"], cfg["doe_cache"])
    return design_space, xt


def write_run_manifest(cfg):
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_simu": cfg["n_simu"], "n_doe": cfg["n_doe"],
        "clones_per_run": cfg["clones_per_run"], "seed": cfg["seed"],
        "annee_debut": cfg["annee_debut"], "nb_annees": cfg["nb_annees"],
        "final_step": cfg["final_step"], "maelia_root": str(cfg["maelia_root"]),
        "gama_headless": str(cfg["gama_headless"]),
        "smt_features": SMT_FEATURES,
    }
    (cfg["project_log_dir"] / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Batch MAELIA/GAMA terrainSA — version serveur.")
    parser.add_argument("--project-root", help="Racine du dépôt (auto-détectée par défaut).")
    parser.add_argument("--maelia-root", help="Racine du workspace MAELIA (ou variable MAELIA_ROOT).")
    parser.add_argument("--gama-headless", help="Chemin vers gama-headless.sh (ou variable GAMA_HEADLESS).")
    parser.add_argument("--n-doe", type=int, default=10000, help="Nombre de points DOE cible (défaut 10000).")
    parser.add_argument("--seed", type=int, default=42, help="Graine du plan LHS (défaut 42).")
    parser.add_argument("--annee-debut", type=int, default=2019)
    parser.add_argument("--nb-annees", type=int, default=10)
    parser.add_argument("--xml-dir", default="/tmp/maelia_smt_terrainSA_xml")
    parser.add_argument("--gama-log", default="/tmp/gama_smt_terrainSA.log")
    parser.add_argument("--script-log", default="batch_terrainSA.log", help="Journal du script.")
    parser.add_argument("--rebuild-terrain", action="store_true", help="Force la reconstruction de terrainSA.")
    parser.add_argument("--fresh", action="store_true",
                        help="Repart de zéro : supprime les runs terrainSA existants avant de lancer.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ne réutilise pas les runs déjà terminés (relance tout).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Génère DOE + dateDose + XML mais ne lance pas GAMA.")
    parser.add_argument("--collect-only", action="store_true",
                        help="Ne lance rien : reconstruit dataset_metamodel.csv depuis les logs présents.")
    parser.add_argument("--keep-outputs", default=None,
                        help="Liste (séparée par des virgules) des fichiers à conserver par run. "
                             f"Défaut : {','.join(DEFAULT_KEEP_FILES)}. Les fichiers du dataset sont "
                             "toujours gardés.")
    parser.add_argument("--keep-all-outputs", action="store_true",
                        help="Conserve tous les fichiers de sortie (désactive l'élagage).")
    parser.add_argument("--purge-gama-runs", action="store_true",
                        help="Après mirror, supprime la sortie brute côté GAMA (models/main/log) "
                             "et le workspace headless, pour éviter la double copie sur le serveur.")
    parser.add_argument("--prune-existing", action="store_true",
                        help="Action seule : élague les runs déjà présents dans log_terrainSA selon "
                             "la whitelist, puis quitte.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = build_config(args)
    setup_logging(Path(args.script_log).expanduser(), args.verbose)

    # Action autonome : élaguer les runs existants et quitter.
    if args.prune_existing:
        log.info("Élagage des runs existants dans %s…", cfg["project_log_dir"])
        prune_existing_runs(cfg)
        return

    # Validation du binaire GAMA (sauf en mode collect/dry-run)
    if not (args.dry_run or args.collect_only):
        if not cfg["gama_headless"] or not cfg["gama_headless"].exists():
            log.error("gama-headless.sh introuvable : %s", cfg["gama_headless"])
            log.error("Passe --gama-headless /chemin/vers/headless/gama-headless.sh ou GAMA_HEADLESS=...")
            sys.exit(2)
    if not cfg["model_path"].exists():
        log.error("launcherBase.gaml introuvable : %s (vérifie --maelia-root)", cfg["model_path"])
        sys.exit(2)

    design_space, xt = prepare_terrain_and_plan(cfg, args)

    log.info("Configuration :")
    log.info("  Repo            : %s", cfg["project_root"])
    log.info("  MAELIA          : %s", cfg["maelia_root"])
    log.info("  GAMA headless   : %s", cfg["gama_headless"])
    log.info("  Runs / clones   : %d × %d = %d points", cfg["n_simu"], len(cfg["all_parcelles"]), cfg["n_doe"])
    log.info("  Campagnes       : %d–%d (finalStep=%d)", cfg["annees"][0], cfg["annees"][-1], cfg["final_step"])
    log.info("  Sorties projet  : %s", cfg["project_log_dir"])
    log.info("  Log GAMA        : %s  (tail -f pour suivre)", cfg["gama_log"])
    if cfg["keep_files"] is None:
        log.info("  Conservation    : tous les fichiers (élagage désactivé)")
    else:
        log.info("  Conservation    : %s", ", ".join(cfg["keep_files"]))
    if cfg["purge_gama_runs"]:
        log.info("  Purge GAMA      : oui (sortie brute supprimée après mirror)")

    # --- Mode collecte seule ---
    if args.collect_only:
        results = {f"smt_{i:03d}": existing_project_run(cfg, f"smt_{i:03d}") for i in range(cfg["n_simu"])}
        n_found = sum(1 for v in results.values() if v)
        log.info("Runs valides trouvés : %d/%d", n_found, cfg["n_simu"])
        collect_and_export(cfg, xt, results)
        return

    # --- Nettoyage optionnel ---
    if args.fresh:
        for label, root in [("GAMA", cfg["gama_output_dir"]), ("projet", cfg["project_log_dir"])]:
            olds = sorted(root.glob(f"{cfg['nom_decoupage']}_smt_*")) if root.exists() else []
            for d in olds:
                shutil.rmtree(d)
            if olds:
                log.info("Nettoyage %s : %d ancien(s) run(s) supprimé(s).", label, len(olds))

    # --- Génération dateDose ---
    generate_all_datedose(design_space, xt, cfg["variants_dir"], cfg["all_parcelles"], cfg["annees"], cfg["n_simu"])
    write_run_manifest(cfg)

    if args.dry_run:
        # Écrit aussi les XML pour vérification, sans lancer GAMA.
        for i in range(cfg["n_simu"]):
            nom = f"smt_{i:03d}"
            (cfg["xml_dir"] / f"{nom}.xml").write_text(
                make_xml(cfg, nom, cfg["variants_dir"] / f"dateDose_{nom}.csv")
            )
        log.info("[dry-run] DOE + dateDose + XML générés. GAMA non lancé.")
        log.info("[dry-run] Relance sans --dry-run pour exécuter les %d simulations.", cfg["n_simu"])
        return

    # --- Boucle GAMA ---
    cfg["gama_log"].write_text("")
    ensure_gama_buffering_preferences(cfg)

    results: dict[str, Path | None] = {}
    t_global = time.time()
    n_skipped = 0
    for i in range(cfg["n_simu"]):
        nom = f"smt_{i:03d}"

        if not args.no_resume:
            existing = existing_project_run(cfg, nom)
            if existing is not None:
                results[nom] = existing
                n_skipped += 1
                log.info("  [SKIP] %s déjà présent (%s)", nom, existing.name)
                continue

        xml_path = cfg["xml_dir"] / f"{nom}.xml"
        xml_path.write_text(make_xml(cfg, nom, cfg["variants_dir"] / f"dateDose_{nom}.csv"))
        log.info("Lancement %s (%d/%d)…", nom, i + 1, cfg["n_simu"])
        results[nom] = run_simulation(cfg, nom, xml_path, cfg["xml_dir"] / f"ws_{nom}")

    n_ok = sum(1 for v in results.values() if v is not None)
    log.info("%s", "═" * 62)
    log.info("Terminé : %d/%d OK (%d repris) en %.0fs",
             n_ok, cfg["n_simu"], n_skipped, time.time() - t_global)
    log.info("%s", "═" * 62)

    collect_and_export(cfg, xt, results)


if __name__ == "__main__":
    main()
