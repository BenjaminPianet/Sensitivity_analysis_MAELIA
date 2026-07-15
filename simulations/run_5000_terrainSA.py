#!/usr/bin/env python3
"""Lance 5000 simulations MAELIA/GAMA sur terrainSA avec les modifications récentes.

Enchaîne deux étapes déjà en place dans le dépôt :

  1. `run_terrainSA_batch.py --n-doe 5000`
       - échantillonne le plan SMT (doses en **log10, FloatVariable(0, 2)**),
       - génère les fichiers dateDose (dose physique appliquée = **10 ** valeur**, log-uniforme dans [1, 100]),
       - lance GAMA headless sur terrainSA (reprise/`--dry-run`/`--collect-only` gérés par ce script),
       - écrit un premier `dataset_metamodel.csv` (sorties + point_idx).

  2. `rebuild_dataset_from_logs.py`
       - reconstruit les colonnes feat_* depuis les itinéraires réellement exécutés dans les logs :
         doses **physiques** [1, 100], **Date_Recolte corrigée** (conversion année-consciente),
         et **défaut d'inactivité = -1** (sentinelle hors plage).

Le dataset final est donc aligné (X reconstruit des mêmes logs que Y) et porte les trois
modifications : passage en log des doses, correctif Date_Recolte, sentinelle d'inactivité.

Usage serveur
-------------
    python run_5000_terrainSA.py \
        --gama-headless /opt/gama/headless/gama-headless.sh \
        --maelia-root ~/Workspace_GAMA/MAELIA

Toute option supplémentaire est transmise telle quelle à `run_terrainSA_batch.py`
(`--dry-run`, `--rebuild-terrain`, `--seed`, etc.). `--n-doe` vaut 5000 par défaut
mais peut être surchargé. En cas d'interruption, relancer la même commande : les runs
GAMA déjà terminés sont détectés et sautés.

Sortie finale :
    <repo>/simulations/log_terrainSA/dataset_metamodel.csv
    <repo>/simulations/log_terrainSA/dataset_metamodel_features.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = HERE / "run_terrainSA_batch.py"
REBUILD = HERE / "rebuild_dataset_from_logs.py"


def run(cmd: list[str]) -> None:
    print("\n>>> " + " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"Échec (code {rc}) : {' '.join(str(c) for c in cmd)}")


def main() -> None:
    extra = sys.argv[1:]

    # Étape 1 : DOE (log) + dateDose (10**) + GAMA + export initial.
    batch_cmd = [sys.executable, str(BATCH)]
    if "--n-doe" not in extra:
        batch_cmd += ["--n-doe", "5000"]
    batch_cmd += extra
    run(batch_cmd)

    # En dry-run, on ne reconstruit pas (pas de logs GAMA).
    if "--dry-run" in extra:
        print("\n[dry-run] Étape 2 (reconstruction) sautée : aucun log GAMA produit.")
        return

    # Étape 2 : reconstruction des feat_* depuis les logs (doses physiques,
    # Date_Recolte corrigée, inactifs = -1).
    run([sys.executable, str(REBUILD)])

    final = HERE / "log_terrainSA" / "dataset_metamodel.csv"
    print(f"\nTerminé. Dataset final aligné : {final}")


if __name__ == "__main__":
    main()
