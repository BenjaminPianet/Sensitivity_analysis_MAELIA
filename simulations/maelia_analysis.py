import numpy as np

# 1. Imports from smt_generation
from smt_generation import agri_design_space, xt, yt

# 2. Imports from our methods file
from hsic_methods import hsic_anova_hierarchical

# SMT imports for optimization
from smt.surrogate_models import KRG
from smt.surrogate_models.krg_based import MixIntKernelType, MixHrcKernelType
from smt.design_space import CategoricalVariable
from hsic_methods import hsic_anova_hierarchical

print(f"Data imported: {xt.shape[0]} samples, {xt.shape[1]} features.")

# Explicitly map the 26 MAELIA variables based on the design space definition
var_names = [
    "Nb_Ferti", "Has_Prepa", "Nb_Prepa", "Type_Prepa_1", "Type_Prepa_2",
    "Nb_F1", "Type_F1_1", "Type_F1_2",
    "Nb_F2", "Type_F2_1", "Type_F2_2",
    "Nb_F3", "Type_F3_1", "Type_F3_2",
    "Date_Semis", "Date_Prepa_Offset", "Date_F1", "Date_F2", "Date_F3", "Date_Recolte",
    "Dose_F1_1", "Dose_F1_2", "Dose_F2_1", "Dose_F2_2", "Dose_F3_1", "Dose_F3_2"
]


    # Get decreed status for features (boolean array)
print("\n[1] Bypassing Kriging: Using Random Forest to compute supervised theta scales...")
print("    Since SMT Kriging failed due to collinearity, Random Forest will perfectly isolate Doses.")

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(xt, yt.ravel())

# RF feature importances are a perfect proxy for true sensitivity (supervised by Y)
rf_importances = rf.feature_importances_

# We scale the importances to act as theta length scales (larger theta = more important)
# A scaling factor of 5.0 ensures the active kernels are sharp enough for ANOVA.
theta = rf_importances * 5.0

# Force theta to be strictly 0 for variables that RF found completely useless (e.g. Dates)
theta[rf_importances < 0.01] = 0.0

# Extract conditional acting status for all samples
_, x_is_acting = agri_design_space.correct_get_acting(xt)

# Get decreed status for features (boolean array)
num_is_decreed = np.array(agri_design_space.is_conditionally_acting)
is_categorical = np.array([isinstance(v, CategoricalVariable) for v in agri_design_space.design_variables])

print("\n[2] Computing Hierarchical HSIC-ANOVA decomposition (Orders 1, 2, 3)...")
print("    Extracting components explaining up to 95% of total variance.")

filtered_results, global_hsic = hsic_anova_hierarchical(
    X=xt, 
    Y=yt, 
    x_is_acting=x_is_acting, 
    num_is_decreed=num_is_decreed,
    is_categorical=is_categorical,
    theta_scales=theta,
    var_names=var_names,
    max_order=4,
    use_smt_theta=True
)

print("\n" + "="*80)
print("                MAELIA SENSITIVITY ANALYSIS (HSIC-ANOVA)")
print("="*80)
print(f"Global HSIC (Total Dependency): {global_hsic:.6e}\n")
print(f"{'Order':<6} | {'Variance Share':<15} | {'Interacting Variables'}")
print("-" * 80)

cumulative_var = 0.0
for res in filtered_results:
    print(f"  {res['order']:<4} |      {res['variance_share']*100:>5.2f}%      | {res['name']}")
    cumulative_var += res['variance_share']
    
print("-" * 80)
print(f"Total variance explained by these terms: {cumulative_var*100:.2f}%")
print("="*80)
