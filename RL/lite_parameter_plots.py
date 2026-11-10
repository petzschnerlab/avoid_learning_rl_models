"""
Re-run the parameter group analyses and plots (run_fit_analyses) from saved fits, then rebuild the
combined behaviour + parameter figures, without re-running any model fitting.

Run from the repo root (same as RL_fit_main.py):
    python RL/lite_parameter_plots.py

Inputs:
    RL/modelling/fit_data_FIT.pkl                     best-run fit per participant and model
    RL/plots/model_behaviours/{model}_model_behaviours.png
        (needed only for the combined figures; make it first with RL/plot_model_behaviours.py)

Outputs:
    RL/stats/pain_fits_linear_results.csv, pain_fits_ttest_results.csv, pain_fits_posthoc_results.csv   (overwritten)
    RL/plots/fits/{model}-model-fits[-significant|-nonsignificant][_supplemental].png/.svg
        (bar versions include significance brackets)
    RL/plots/model_behaviours/{model}_combined_model_behaviours.png        (behaviours + all parameters, stacked)
    RL/plots/model_behaviours/{model}_combined_sig_model_behaviours.png    (behaviours + significant parameters, side by side)
"""

import os
import sys
import copy
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.dont_write_bytecode = True

# Work from the repo root, with the RL folder importable (as in RL_fit_main.py)
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
sys.path.insert(0, RL_DIR)
os.chdir(REPO_ROOT)

from helpers.analyses import Analyses

# =========================================== #
# ================= INPUTS ================== #
# =========================================== #

# Note: the pickles in RL/fits/ are small test runs; the full fits are in RL/modelling/
FIT_DATA_FILE = 'RL/modelling/fit_data_FIT.pkl'

MODELS = None    # None = every model in the pickle, or e.g. ['Advantage+novel']
COMBINE = True   # also rebuild the combined behaviour + parameter figures


if __name__ == '__main__':

    print(f'Loading fits from {FIT_DATA_FILE}...')
    with open(FIT_DATA_FILE, 'rb') as f:
        fit_data = pickle.load(f)
    models = list(fit_data.keys()) if MODELS is None else MODELS
    fit_data = {model: fit_data[model] for model in models}
    print(f'  Models: {models}')

    os.makedirs('RL/stats', exist_ok=True)
    analyses = Analyses()

    print('Running parameter group analyses and plots...')
    analyses.run_fit_analyses(copy.deepcopy(fit_data))

    # Same calls as the end of Analyses.run_fit_simulations
    if COMBINE:
        print('Combining with behaviour plots...')
        for model in models:
            analyses.plot_combined_simulations(f'{model}_combined_model_behaviours', f'{model}_model_behaviours', f'{model}-model-fits')
            analyses.plot_combined_simulations(f'{model}_combined_sig_model_behaviours', f'{model}_model_behaviours', f'{model}-model-fits-significant', orientation='horizontal')

    print(f'Done. Figures saved to {os.path.join(REPO_ROOT, "RL", "plots", "fits")}' +
          (f' and {os.path.join(REPO_ROOT, "RL", "plots", "model_behaviours")}' if COMBINE else ''))
