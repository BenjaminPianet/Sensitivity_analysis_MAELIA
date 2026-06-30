from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "web_runs"
DEFAULT_TARGETS = ["N_lixi", "dCorg", "rdt"]
DEFAULT_PDP_ICE_FEATURES = ["Date_Semis", "Date_Recolte", "Delta_PREPA_Semis"]

AGRI_FEATURES = [
    "n_ferti", "has_prepa", "nb_prepa",
    "Date_Semis", "Delta_PREPA_Semis", "Profondeur_Semis",
    "Profondeur_Prepa_1", "Profondeur_Prepa_2",
    "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
    "Dose_F1", "Dose_F2", "Dose_F3",
]

AGRI_CATEGORICAL = ["n_ferti", "has_prepa", "nb_prepa"]

FEATURE_LABELS = {
    "n_ferti": "Nombre d'apports N",
    "has_prepa": "Préparation du sol",
    "nb_prepa": "Nombre de préparations",
    "Date_Semis": "Date de semis",
    "Delta_PREPA_Semis": "Décalage préparation-semis",
    "Profondeur_Semis": "Profondeur de semis",
    "Profondeur_Prepa_1": "Profondeur préparation 1",
    "Profondeur_Prepa_2": "Profondeur préparation 2",
    "Date_F1": "Date apport 1",
    "Date_F2": "Date apport 2",
    "Date_F3": "Date apport 3",
    "Date_Recolte": "Date de récolte",
    "Dose_F1": "Dose N apport 1",
    "Dose_F2": "Dose N apport 2",
    "Dose_F3": "Dose N apport 3",
    "N_lixi": "Azote lixivié",
    "dCorg": "Variation carbone organique",
    "rdt": "Rendement",
}


TARGET_LABELS = {
    "N_lixi": "Azote lixivié (kg N/ha)",
    "dCorg": "Variation du carbone organique (kg C/ha)",
    "rdt": "Rendement (t/ha)",
}

PALETTE = {
    "blue": "#2F6B9A",
    "teal": "#2A9D8F",
    "amber": "#E9A03F",
    "coral": "#D7655B",
    "ink": "#263238",
    "muted": "#6B7280",
    "grid": "#E7EAEE",
    "paper": "#FAFBFC",
}
