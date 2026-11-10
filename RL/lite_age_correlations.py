"""
Correlate age with every fitted parameter of every model from saved fits,
without re-running any model fitting.

Run from the repo root (same as RL_fit_main.py):
    python RL/lite_age_correlations.py

Inputs:
    RL/modelling/fit_data_FIT.pkl          best-run fit per participant and model
    RL/data/processed_learning_data.csv    contains each participant's age

Outputs:
    RL/stats/age_parameter_correlations.csv                 all models and parameters, overall and within each pain group
    RL/plots/correlations/{model}_age_correlations.png/.svg  one figure per model
"""

import os
import sys
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.dont_write_bytecode = True

# Work from the repo root, with the RL folder importable (as in RL_fit_main.py)
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
sys.path.insert(0, RL_DIR)
os.chdir(REPO_ROOT)

import pandas as pd
from helpers.analyses import Analyses

# =========================================== #
# ================= INPUTS ================== #
# =========================================== #

# Note: the pickles in RL/fits/ are small test runs; the full fits are in RL/modelling/
FIT_DATA_FILE = 'RL/modelling/fit_data_FIT.pkl'
LEARNING_FILE = 'RL/data/processed_learning_data.csv'

MODELS = None        # None = every model in the pickle, or e.g. ['Advantage+novel']
METHOD = 'pearson'   # 'pearson' or 'spearman'
TRANSFORM = True     # log-transform skewed parameters before correlating


if __name__ == '__main__':

    print(f'Loading fits from {FIT_DATA_FILE}...')
    with open(FIT_DATA_FILE, 'rb') as f:
        fit_data = pickle.load(f)
    models = list(fit_data.keys()) if MODELS is None else MODELS
    fit_data = {model: fit_data[model] for model in models}
    print(f'  Models: {models}')

    print('Correlating age with model parameters...')
    results = Analyses().run_age_correlations(fit_data, LEARNING_FILE, method=METHOD, transform=TRANSFORM)

    print('')
    print('AGE CORRELATIONS (Holm-corrected across each model\'s parameters, separately for each group)')
    print('==========')
    print(results.drop(columns=['covariate', 'method']).to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('==========')
    print('Saved RL/stats/age_parameter_correlations.csv and RL/plots/correlations/*_age_correlations.png/.svg')
