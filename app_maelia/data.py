"""Chargement du dataset et confrontation à la spécification d'espace.

Volontairement minimal : la v2 lit le même `dataset_metamodel.csv` que la v1, mais
ne valide plus le plan contre une liste de 14 paramètres codée en dur — c'est la
spécification qui dit ce qui est attendu.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .space import SpaceSpec

from .chemins import DONNEES

# Jeu de données livré avec l'application. L'utilisateur peut en désigner un autre :
# c'est le champ « dossier de données » de l'interface.
DEFAULT_LOG_DIR = DONNEES / "simulations_prechargees"

TARGET_LABELS = {
    "N_lixi": "Azote lixivié (kg N/ha)",
    "dCorg": "Variation du carbone organique (kg C/ha)",
    "rdt": "Rendement (t/ha)",
}


@dataclass
class Dataset:
    frame: pd.DataFrame
    path: Path
    targets: list[str]
    anomalies: list[str]


def load(spec: SpaceSpec, log_dir: str | Path = DEFAULT_LOG_DIR) -> Dataset:
    """Charge le dataset du dossier de logs et renomme feat_* selon le manifeste."""
    log_path = Path(log_dir).expanduser()
    dataset_path = log_path / "dataset_metamodel.csv"
    manifest_path = log_path / "dataset_metamodel_features.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset_metamodel.csv introuvable dans {log_path}. "
            "Relance l'export du notebook de simulation, ou choisis un autre dossier de logs."
        )

    frame = pd.read_csv(dataset_path)
    if manifest_path.exists():
        mapping = (
            pd.read_csv(manifest_path)[["colonne", "parametre"]]
            .dropna().drop_duplicates("colonne")
            .set_index("colonne")["parametre"].astype(str).to_dict()
        )
        frame = frame.rename(columns=mapping)

    expected = spec.feature_names()
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Le dataset ne contient pas les paramètres décrits par l'espace « {spec.name} » : "
            f"{missing}. Vérifie que le dataset et la spécification décrivent le même plan."
        )

    targets = [t for t in TARGET_LABELS if t in frame.columns]
    if not targets:
        raise ValueError(f"Aucune sortie MAELIA trouvée dans {dataset_path}.")

    return Dataset(
        frame=frame,
        path=dataset_path,
        targets=targets,
        # Les écarts au domaine ne bloquent pas : ils sont remontés à l'utilisateur.
        anomalies=spec.check_data(frame),
    )
