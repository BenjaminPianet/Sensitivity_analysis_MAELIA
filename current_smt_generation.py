#!/usr/bin/env python
# coding: utf-8

# <div class="jumbotron text-left"><b>
#     
# This tutorial describes how to use de DesignSpace within the SMT toolbox.
# <div>
#     
#     February 2026 - `SMT version 2.10.1`
#   
#      Paul Saves (IRIT/SMAC), Jasper Bussemaker (DLR), Rémi Lafage (ONERA/DTIS/MIDL) and Nathalie BARTOLI (ONERA/DTIS/M2CI)

# <div class="alert alert-info fade in" id="d110">
# <p>Some updates</p>
# <ol> -  Manipulation of mixed DOE (continuous, integer,  categorical and hierarchical variables) </ol>
# </div>

# <p class="alert alert-success" style="padding:1em">
# To use SMT models, please follow this link : https://github.com/SMTorg/SMT/blob/master/README.md. The documentation is available here: http://smt.readthedocs.io/en/latest/
# </p>
# 
# The reference paper is available
# here https://www.sciencedirect.com/science/article/pii/S096599782300162X
# 
# 

# For mixed integer with continuous relaxation, the reference paper is available here https://www.sciencedirect.com/science/article/pii/S0925231219315619

# In[18]:


# to have the latest version
get_ipython().system('pip install configspace')
get_ipython().system('pip install adsg-core')
get_ipython().system('pip install smt-design-space-ext')
get_ipython().system('pip install smt')
get_ipython().system('pip install "adsg-core[nb]"')


# <div class="alert alert-warning" >
# If you use hierarchical variables and the size of your doe greater than 30 points, you may leverage the `numba` JIT compiler to speed up the computation
# To do so:
#     
#  - install numba library
#     
#      `pip install numba`
#     
#     
#  - and define the environment variable `USE_NUMBA_JIT = 1` (unset or 0 if you do not want to use numba)
#     
#      - Linux: export USE_NUMBA_JIT = 1
#     
#      - Windows: set USE_NUMBA_JIT = 1
# 
# </div>

# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')

# to ignore warning messages
import warnings

import plotly.io as pio
from smt.sampling_methods import LHS
from smt_design_space_ext import (
    AdsgDesignSpaceImpl,
    CategoricalVariable,
    ConfigSpaceDesignSpaceImpl,
    DesignSpace,
    FloatVariable,
    IntegerVariable,
    OrdinalVariable,
)

warnings.filterwarnings("ignore")

pio.renderers.default = "notebook"


# # MAELIA

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from smt.design_space import FloatVariable, OrdinalVariable, CategoricalVariable
from smt_design_space_ext import AdsgDesignSpaceImpl
from smt.surrogate_models import KRG

# ==========================================
# 1. PARAMÈTRES EXPLORÉS
# ==========================================
FERTILIZER_TYPE = 'AN'
PREPA_TYPE_1 = 'travail_sol_1'
PREPA_TYPE_2 = 'une_reprise'
SEMIS_TYPE = 'semis'
PREPA_DEPTH_MIN_CM = 1.0
PREPA_DEPTH_MAX_CM = 25.0
SEMIS_DEPTH_MIN_CM = 1.0
SEMIS_DEPTH_MAX_CM = 4.0

SMT_FEATURES = [
    'n_ferti', 'has_prepa', 'nb_prepa',
    'Date_Semis', 'Delta_PREPA_Semis', 'Profondeur_Semis',
    'Profondeur_Prepa_1', 'Profondeur_Prepa_2',
    'Date_F1', 'Date_F2', 'Date_F3', 'Date_Recolte',
    'Dose_F1', 'Dose_F2', 'Dose_F3',
]
SMT_CATEGORICAL = ['n_ferti', 'has_prepa', 'nb_prepa']
SMT_CONTINUOUS = [feature for feature in SMT_FEATURES if feature not in SMT_CATEGORICAL]

agri_design_space = AdsgDesignSpaceImpl(
    design_variables=[
        OrdinalVariable(['0_ferti', '1_ferti', '2_ferti', '3_ferti']),
        OrdinalVariable(['Non_prepa', 'Oui_prepa']),
        OrdinalVariable(['1_prepa', '2_prepa']),
        FloatVariable(45, 106),
        FloatVariable(-44, -4),
        FloatVariable(SEMIS_DEPTH_MIN_CM, SEMIS_DEPTH_MAX_CM),
        FloatVariable(PREPA_DEPTH_MIN_CM, PREPA_DEPTH_MAX_CM),
        FloatVariable(PREPA_DEPTH_MIN_CM, PREPA_DEPTH_MAX_CM),
        FloatVariable(106, 323),
        FloatVariable(210, 323),
        FloatVariable(240, 323),
        FloatVariable(323, 384),
        FloatVariable(10, 100),
        FloatVariable(10, 100),
        FloatVariable(10, 100),
    ]
)

agri_design_space.declare_decreed_var(decreed_var=2, meta_var=1, meta_value=['Oui_prepa'])
agri_design_space.declare_decreed_var(decreed_var=4, meta_var=1, meta_value=['Oui_prepa'])
agri_design_space.declare_decreed_var(decreed_var=6, meta_var=1, meta_value=['Oui_prepa'])
agri_design_space.declare_decreed_var(decreed_var=7, meta_var=2, meta_value=['2_prepa'])

agri_design_space.declare_decreed_var(decreed_var=8,  meta_var=0, meta_value=['1_ferti', '2_ferti', '3_ferti'])
agri_design_space.declare_decreed_var(decreed_var=12, meta_var=0, meta_value=['1_ferti', '2_ferti', '3_ferti'])
agri_design_space.declare_decreed_var(decreed_var=9,  meta_var=0, meta_value=['2_ferti', '3_ferti'])
agri_design_space.declare_decreed_var(decreed_var=13, meta_var=0, meta_value=['2_ferti', '3_ferti'])
agri_design_space.declare_decreed_var(decreed_var=10, meta_var=0, meta_value=['3_ferti'])
agri_design_space.declare_decreed_var(decreed_var=14, meta_var=0, meta_value=['3_ferti'])

DEFAULT_FLOATS = [
    75.0, -20.0, 3.0, 15.0, 10.0,
    150.0, 230.0, 280.0, 350.0,
    50.0, 50.0, 50.0,
]


def parse_smt_row(row_data):
    """Analyse une ligne décodée par SMT avec le plan terrainSA compact."""
    parsed = {'n_ferti': 0, 'has_prepa': False, 'nb_prepa': 1, 'floats': []}
    values = list(row_data.values()) if isinstance(row_data, dict) else row_data
    for val in values:
        if isinstance(val, (float, int, np.floating, np.integer)):
            parsed['floats'].append(float(val))
        elif isinstance(val, str):
            if val in ['0_ferti', '1_ferti', '2_ferti', '3_ferti']:
                parsed['n_ferti'] = int(val[0])
            elif val in ['Oui_prepa', 'Non_prepa']:
                parsed['has_prepa'] = val == 'Oui_prepa'
            elif val in ['1_prepa', '2_prepa']:
                parsed['nb_prepa'] = int(val[0])
    return parsed


def normalized_floats(parsed):
    fl = list(parsed['floats'])
    for default in DEFAULT_FLOATS[len(fl):]:
        fl.append(default)
    return fl[:len(DEFAULT_FLOATS)]


def ordered_fertilisation_dates(parsed, fl):
    dates = {'f1': None, 'f2': None, 'f3': None}
    if parsed['n_ferti'] >= 1:
        dates['f1'] = int(round(fl[5]))
    if parsed['n_ferti'] >= 2:
        dates['f2'] = max(int(round(fl[6])), dates['f1'] + 1)
    if parsed['n_ferti'] >= 3:
        dates['f3'] = max(int(round(fl[7])), dates['f2'] + 1)
    return dates


def extract_table_from_smt(x_array, dsg_space):
    rows = []
    decoded_data = dsg_space.decode_values(x_array)
    for row_data in decoded_data:
        p = parse_smt_row(row_data)
        fl = normalized_floats(p)
        ferti_dates = ordered_fertilisation_dates(p, fl)
        semi = int(round(fl[0]))
        prepa_date = int(round(semi + fl[1])) if p['has_prepa'] else None
        rows.append({
            'n_ferti': p['n_ferti'],
            'has_prepa': 'Oui' if p['has_prepa'] else 'Non',
            'nb_prepa': p['nb_prepa'] if p['has_prepa'] else '-',
            'Semi': semi,
            'Delta_PREPA_Semis': round(fl[1], 1) if p['has_prepa'] else '-',
            'Profondeur_Semis': round(fl[2], 1),
            'PREPA_Date': prepa_date if prepa_date is not None else '-',
            'Profondeur_Prepa_1': round(fl[3], 1) if p['has_prepa'] else '-',
            'Profondeur_Prepa_2': round(fl[4], 1) if p['has_prepa'] and p['nb_prepa'] == 2 else '-',
            'Ferti_1': ferti_dates['f1'] if ferti_dates['f1'] is not None else '-',
            'Dose_F1': round(fl[9], 1) if p['n_ferti'] >= 1 else '-',
            'Ferti_2': ferti_dates['f2'] if ferti_dates['f2'] is not None else '-',
            'Dose_F2': round(fl[10], 1) if p['n_ferti'] >= 2 else '-',
            'Ferti_3': ferti_dates['f3'] if ferti_dates['f3'] is not None else '-',
            'Dose_F3': round(fl[11], 1) if p['n_ferti'] >= 3 else '-',
            'Recolte': int(round(fl[8])),
            'Engrais': FERTILIZER_TYPE,
        })
    return pd.DataFrame(rows)


# ==========================================
# 4. GESTION DES DONNÉES (MAELIA DUMMY)
# ==========================================
def evaluate_maelia_dummy(x_array, dsg_space):
    y = np.zeros((x_array.shape[0], 1))
    decoded_data = dsg_space.decode_values(x_array)

    for i, row_data in enumerate(decoded_data):
        p = parse_smt_row(row_data)
        fl = normalized_floats(p)
        dose_totale = 0

        if p["n_ferti"] >= 1:
            dose_totale += fl[9]
        if p["n_ferti"] >= 2:
            dose_totale += fl[10]
        if p["n_ferti"] >= 3:
            dose_totale += fl[11]

        y[i, 0] = 50 + (dose_totale * 0.25) - (0.0006 * dose_totale**2) + 0.1#np.random.normal(0, 1.5)
    return y




# In[23]:


N_DOE = 30
print(f"Génération de {N_DOE} points LHS hiérarchiques (SMT ADSG)...")

from smt.applications.mixed_integer import MixedIntegerSamplingMethod
from smt.sampling_methods import LHS

# On NE PASSE PLUS par xlimits_relaxed
# On donne directement le design_space (qui contient la nature des variables) au wrapper

# 1. Configuration du sampler
sampler = MixedIntegerSamplingMethod(
    LHS, 
    agri_design_space, 
    criterion="ese", 
    seed=42 # Note: SMT utilise souvent random_state au lieu de seed
)


xt_raw = sampler(N_DOE)
xt, _ = agri_design_space.correct_get_acting(xt_raw)

# --- Robust deduplication ---
def is_new(row, X, tol=1e-8):
    return not any(np.allclose(row, x, atol=tol) for x in X)

xt_unique = []
for row in xt:
    if is_new(row, xt_unique):
        xt_unique.append(row)

xt = np.array(xt_unique)

if xt.shape[0] < N_DOE:
    print("{N_DOE - xt.shape[0]} duplicates after projection")

# --- Resampling (batch, not 1-by-1) ---
while xt.shape[0] < N_DOE:
    n_missing = N_DOE - xt.shape[0]
    xt_new = sampler(n_missing * 2)
    xt_corr, _ = agri_design_space.correct_get_acting(xt_new)

    for row in xt_corr:
        if is_new(row, xt):
            xt = np.vstack([xt, row])
        if xt.shape[0] >= N_DOE:
            break

# --- Critical step ---
xt,_ = agri_design_space.correct_get_acting(xt)

print(f"Matrice DOE : {xt.shape[0]} × {xt.shape[1]} variables\n")


# In[24]:


doe_display = extract_table_from_smt(xt, agri_design_space)
display(doe_display.iloc[:, :10])
display(doe_display.iloc[:, 10:])

yt = evaluate_maelia_dummy(xt, agri_design_space)


# In[25]:


from graphviz import Source
from IPython.display import display

def render_to_image(dsg_obj, filename="temp_graph"):
    dot_str = dsg_obj.export_dot()

    # Injection des paramètres de layout dans le DOT
    layout_config = """
    graph [
        rankdir=LR,
        nodesep=0.3,
        ranksep=0.5,
        splines=true,
        margin=0
    ];
    """

    # On insère juste après "digraph ..."
    dot_str = dot_str.replace("{", "{\n" + layout_config, 1)

    src = Source(dot_str, filename=filename, format="png", engine="dot")

    display(src)

print("=== GRAPHE GLOBAL DE L'ESPACE DE CONCEPTION ===")
render_to_image(agri_design_space.adsg, "maelia_space")


# In[27]:


from adsg_core import BasicADSG, DesignVariableNode, NamedNode
from graphviz import Source
from IPython.display import display

def create_explicit_maelia_adsg():
    adsg = BasicADSG()

    # --- 1. ROOT NODES (Always Active) ---
    n_ferti = NamedNode("n_ferti")
    n_ferti_opts = [NamedNode("0"), NamedNode("1"), NamedNode("2"), NamedNode("3")]
    adsg.add_selection_choice("n_ferti_choice", n_ferti, n_ferti_opts)

    has_prepa = NamedNode("has_prepa")
    has_prepa_opts = [NamedNode("Non"), NamedNode("Oui")]
    adsg.add_selection_choice("has_prepa_choice", has_prepa, has_prepa_opts)

    semi_date = DesignVariableNode("Date_Semis", bounds=(45, 106))
    profondeur_semis = DesignVariableNode("Profondeur_Semis", bounds=(1, 4))
    jours_recolte = DesignVariableNode("Date_Recolte", bounds=(323, 384))

    adsg.add_node(semi_date)
    adsg.add_node(profondeur_semis)
    adsg.add_node(jours_recolte)

    # --- 2. PREPARATION HIERARCHY ---
    nb_prepa = NamedNode("nb_prepa")
    nb_prepa_opts = [NamedNode("1_prepa"), NamedNode("2_prepa")]
    adsg.add_selection_choice("nb_prepa_choice", nb_prepa, nb_prepa_opts)

    jours_prepa = DesignVariableNode("Delta_PREPA_Semis", bounds=(-44, -4))
    profondeur_prepa_1 = DesignVariableNode("Profondeur_Prepa_1", bounds=(1, 25))

    for node in [nb_prepa, jours_prepa, profondeur_prepa_1]:
        adsg.add_edge(has_prepa_opts[1], node)

    profondeur_prepa_2 = DesignVariableNode("Profondeur_Prepa_2", bounds=(1, 25))
    adsg.add_edge(nb_prepa_opts[1], profondeur_prepa_2)

    # --- 3. FERTILIZATION HIERARCHY ---
    # The fertilizer product is fixed to the representative mineral fertilizer AN.
    f1_dose = DesignVariableNode("Dose_F1", bounds=(10, 100))
    f1_date = DesignVariableNode("Date_F1", bounds=(106, 323))
    for opt_idx in [1, 2, 3]:
        for node in [f1_date, f1_dose]:
            adsg.add_edge(n_ferti_opts[opt_idx], node)

    f2_dose = DesignVariableNode("Dose_F2", bounds=(10, 100))
    f2_date = DesignVariableNode("Date_F2", bounds=(106, 323))
    for opt_idx in [2, 3]:
        for node in [f2_date, f2_dose]:
            adsg.add_edge(n_ferti_opts[opt_idx], node)

    f3_dose = DesignVariableNode("Dose_F3", bounds=(10, 100))
    f3_date = DesignVariableNode("Date_F3", bounds=(240, 323))
    for node in [f3_date, f3_dose]:
        adsg.add_edge(n_ferti_opts[3], node)

    # --- 4. INITIALIZE GRAPH START NODES ---
    adsg = adsg.set_start_nodes({n_ferti, has_prepa, semi_date, profondeur_semis, jours_recolte})

    return adsg

def render_to_image(dsg_obj, filename="temp_graph"):
    dot_str = dsg_obj.export_dot()

    layout_config = """
    graph [
        rankdir=LR,
        nodesep=0.3,
        ranksep=0.5,
        splines=true,
        margin=0
    ];
    """

    dot_str = dot_str.replace("{", "{\n" + layout_config, 1)

    src = Source(dot_str, filename=filename, format="png", engine="dot")
    display(src)


maelia_adsg = create_explicit_maelia_adsg()

print("=== GRAPHE GLOBAL DE L'ESPACE DE CONCEPTION MAELIA ===")
render_to_image(maelia_adsg, "maelia_adsg")


# In[ ]:


print("\nEntraînement du Krigeage Hiérarchique SMT (KRG)...")
sm = KRG(design_space=agri_design_space, print_global=False,noise0=[0.01],corr='pow_exp')
sm.set_training_values(xt, yt)
sm.train()
print("Modèle entraîné !")


# In[ ]:


xtest, is_acting_test = agri_design_space._sample_valid_x(30)
ytest_vrai = evaluate_maelia_dummy(xtest, agri_design_space)
ytest_predit = sm.predict_values(xtest)

plt.figure(figsize=(8, 6))
plt.scatter(ytest_vrai, ytest_predit, color='#2ca02c', edgecolors='k', s=60, alpha=0.8, label="Prédictions SMT")
plt.plot([ytest_vrai.min(), ytest_vrai.max()], [ytest_vrai.min(), ytest_vrai.max()], 'r--', lw=2, label="Parfait (Y = X)")
plt.xlabel("Vrai Rendement (MAELIA)")
plt.ylabel("Rendement Prédit (Surrogate)")
plt.title("Validation du Krigeage Hiérarchique SMT")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# In[ ]:


ytest_vrai = evaluate_maelia_dummy(xt, agri_design_space)
ytest_predit = sm.predict_values(xt)

plt.figure(figsize=(8, 6))
plt.scatter(ytest_vrai, ytest_predit, color='#2ca02c', edgecolors='k', s=60, alpha=0.8, label="Prédictions SMT")
plt.plot([ytest_vrai.min(), ytest_vrai.max()], [ytest_vrai.min(), ytest_vrai.max()], 'r--', lw=2, label="Parfait (Y = X)")
plt.xlabel("Vrai Rendement (MAELIA)")
plt.ylabel("Rendement Prédit (Surrogate)")
plt.title("Validation du Krigeage Hiérarchique SMT")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# In[ ]:





# In[ ]:




