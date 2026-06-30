from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / 'analysis' / 'tools'
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# 1. Imports from smt_generation
from smt_generation import agri_design_space, xt, yt

# 2. Imports from our methods file
from hsic_methods import hsic_anova_hierarchical

# SMT imports for optimization
from smt.surrogate_models import KRG
from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
from smt.design_space import CategoricalVariable

from sklearn.inspection import permutation_importance
print(f"Data imported: {xt.shape[0]} samples, {xt.shape[1]} features.")

# Explicitly map the 15 MAELIA variables based on the current design space definition
var_names = [
    "n_ferti", "has_prepa", "nb_prepa",
    "Date_Semis", "Delta_PREPA_Semis", "Profondeur_Semis",
    "Profondeur_Prepa_1", "Profondeur_Prepa_2",
    "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
    "Dose_F1", "Dose_F2", "Dose_F3",
]


    # Get decreed status for features (boolean array)
print("\n[1] Normalizing data and Computing Permutation Importance (RF)...")

# IMPORTANT: Normalize continuous variables to [0, 1] for proper distance calculation
xt_normalized = np.copy(xt).astype(float)
for i, var in enumerate(agri_design_space.design_variables):
    if not isinstance(var, CategoricalVariable):
        lower = var.lower
        upper = var.upper
        if upper > lower:
            xt_normalized[:, i] = (xt[:, i] - lower) / (upper - lower)

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ASTUCE : En limitant la profondeur de l'arbre et en forçant des feuilles larges,
# le RF évite de surinterpréter les variables conditionnelles rarement actives.
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=15, random_state=42)
rf.fit(xt_normalized, yt.ravel())

# Extract conditional acting status for all samples
_, x_is_acting = agri_design_space.correct_get_acting(xt)

# Get decreed status for features (boolean array)
num_is_decreed = np.array(agri_design_space.is_conditionally_acting)
is_categorical = np.array([isinstance(v, CategoricalVariable) for v in agri_design_space.design_variables])

print("    Computing Permutation Importances...")
result = permutation_importance(rf, xt_normalized, yt.ravel(), n_repeats=10, random_state=42)

# We use the RAW permutation importance to set the RKHS length scale (theta).
# The physical sensitivity of a variable doesn't change based on its rarity!
theta = result.importances_mean * 5.0

# On filtre sur le score brut pour tuer le vrai bruit blanc
theta[result.importances_mean < 0.005] = 0.0

print("\n[2] Computing Hierarchical HSIC-ANOVA decomposition (Orders 1, 2, 3)...")
print("    Extracting components explaining up to 95% of total variance.")

filtered_results, global_hsic = hsic_anova_hierarchical(
    X=xt_normalized, 
    Y=yt, 
    x_is_acting=x_is_acting, 
    num_is_decreed=num_is_decreed,
    is_categorical=is_categorical,
    theta_scales=theta,
    var_names=var_names,
    max_order=4,
    use_smt_theta=True,
    use_kta=False
)

print("\nL'analyse est terminée. Le tableau 5 colonnes a été généré avec succès par hsic_methods.py !")
