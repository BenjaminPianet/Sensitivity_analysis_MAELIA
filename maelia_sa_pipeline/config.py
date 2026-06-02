from pathlib import Path

PROJECT_ROOT = Path("/Users/benjamin/files/Repositories/Sensitivity_analysis_MAELIA")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "web_runs"
DEFAULT_TARGETS = ["N_lixi", "dCorg", "rdt"]

AGRI_FEATURES = ['n_ferti', 'has_prepa', 'nb_prepa', 'prepa_1', 'nb_f1', 'type_f1_1', 'nb_f2', 'type_f2_1', 'nb_f3', 'type_f3_1', 'prepa_2', 'type_f1_2', 'type_f2_2', 'type_f3_2', 'Date_Semis', 'Delta_PREPA_Semis', 'Date_F1', 'Date_F2', 'Date_F3', 'Date_Recolte', 'Dose_F1_1', 'Dose_F1_2', 'Dose_F2_1', 'Dose_F2_2', 'Dose_F3_1', 'Dose_F3_2']

AGRI_CATEGORICAL = ['n_ferti', 'has_prepa', 'nb_prepa', 'prepa_1', 'nb_f1', 'type_f1_1', 'nb_f2', 'type_f2_1', 'nb_f3', 'type_f3_1', 'prepa_2', 'type_f1_2', 'type_f2_2', 'type_f3_2']

FEATURE_LABELS = {
    "n_ferti": "Nombre d'apports N",
    "has_prepa": "Préparation du sol",
    "nb_prepa": "Nombre de préparations",
    "prepa_1": "Préparation 1",
    "prepa_2": "Préparation 2",
    "nb_f1": "Produits apport 1",
    "type_f1_1": "Type N 1.1",
    "type_f1_2": "Type N 1.2",
    "nb_f2": "Produits apport 2",
    "type_f2_1": "Type N 2.1",
    "type_f2_2": "Type N 2.2",
    "nb_f3": "Produits apport 3",
    "type_f3_1": "Type N 3.1",
    "type_f3_2": "Type N 3.2",
    "Date_Semis": "Date de semis",
    "Delta_PREPA_Semis": "Décalage préparation-semis",
    "Date_F1": "Date apport 1",
    "Date_F2": "Date apport 2",
    "Date_F3": "Date apport 3",
    "Date_Recolte": "Date de récolte",
    "Dose_F1_1": "Dose N 1.1",
    "Dose_F1_2": "Dose N 1.2",
    "Dose_F2_1": "Dose N 2.1",
    "Dose_F2_2": "Dose N 2.2",
    "Dose_F3_1": "Dose N 3.1",
    "Dose_F3_2": "Dose N 3.2",
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
