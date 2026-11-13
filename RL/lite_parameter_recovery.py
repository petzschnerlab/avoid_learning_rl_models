"""
Re-plot the parameter recovery figure (true vs fitted parameters) from saved recovery fits,
without re-running any data generation or model fitting.

Run from the repo root (same as RL_validation_main.py):
    python RL/lite_parameter_recovery.py

Inputs (written by a previous parameter recovery run, RL_validation_main.py):
    RL/fits/fit_data_PARAMETER.pkl                                fitted parameters for each simulated participant
    RL/data/generated/{name}/{name}_generated_parameters.csv      true parameters for each simulated participant

Outputs:
    RL/plots/correlations/recovery_correlation_plot.png/.svg
    RL/stats/param_recovery_corrs.csv   (only if UPDATE_TABLE = True)
"""

import os
import sys
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.dont_write_bytecode = True

# Work from the repo root, with the RL folder importable (as in RL_validation_main.py)
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
sys.path.insert(0, RL_DIR)
os.chdir(REPO_ROOT)

from helpers.analyses import Analyses
from helpers.priors import fixed_priors

# =========================================== #
# ================= INPUTS ================== #
# =========================================== #

FIT_DATA_FILE = 'RL/fits/fit_data_PARAMETER.pkl'

MODELS = None          # None = every model in the pickle, or e.g. ['Advantage+novel']
BOUNDS = None          # None = model default bounds (as in RL_validation_main.py, where custom bounds are commented out)

PANEL_SIZE = (2.4, 2)  # width, height of each subplot in inches
MARKER_SIZE = 5        # scatter point size
ALPHA = 0.5            # scatter point transparency (as in the full pipeline)

UPDATE_TABLE = False   # also recompute RL/stats/param_recovery_corrs.csv


if __name__ == '__main__':

    print(f'Loading recovery fits from {FIT_DATA_FILE}...')
    with open(FIT_DATA_FILE, 'rb') as f:
        fit_data = pickle.load(f)
    models = list(fit_data.keys()) if MODELS is None else MODELS
    fit_data = {model: fit_data[model] for model in models}
    print(f'  Models: {models}')

    analyses = Analyses()
    os.makedirs('RL/plots/correlations', exist_ok=True)

    print('Plotting parameter recovery...')
    analyses.plot_parameter_fits(models=models,
                                 fit_data=fit_data,
                                 fixed=fixed_priors(models),
                                 bounds=BOUNDS,
                                 alpha=ALPHA,
                                 panel_size=PANEL_SIZE,
                                 marker_size=MARKER_SIZE)

    if UPDATE_TABLE:
        print('Computing recovery correlations...')
        os.makedirs('RL/stats', exist_ok=True)
        analyses.describe_param_recovery_corrs(fit_data=fit_data)

    print(f'Done. Figure saved to {os.path.join(REPO_ROOT, "RL", "plots", "correlations", "recovery_correlation_plot.png")}')
