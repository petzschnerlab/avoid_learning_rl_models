"""
Re-plot the AIC/BIC model comparison figures (plot_model_comparisons) from saved fits,
without re-running any model fitting.

Run from the repo root (same as RL_fit_main.py):
    python RL/lite_model_comparisons.py

Inputs (written by a previous fit run):
    RL/modelling/fit_data_FIT.pkl          best-run fit per participant and model
    RL/data/processed_learning_data.csv    empirical data (used for the number of trials in the BIC)
    RL/data/processed_transfer_data.csv

Outputs (same as compute_criterions in the full pipeline):
    RL/plots/AIC_model_comparisons.png
    RL/plots/BIC_model_comparisons.png
    RL/fits/group_AIC*.csv, RL/fits/group_BIC*.csv   (overwritten)
    RL/stats/group_AIC_friedman.csv, RL/stats/group_AIC_posthoc.csv
    RL/stats/group_BIC_friedman.csv, RL/stats/group_BIC_posthoc.csv
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.dont_write_bytecode = True

# Work from the repo root, with the RL folder importable (as in RL_fit_main.py)
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
sys.path.insert(0, RL_DIR)
os.chdir(REPO_ROOT)

from helpers.analyses import Analyses
from helpers.dataloader import DataLoader

# =========================================== #
# ================= INPUTS ================== #
# =========================================== #

# Note: the pickles in RL/fits/ are small test runs; the full fits are in RL/modelling/
FIT_DATA_FILE = 'RL/modelling/fit_data_FIT.pkl'

LEARNING_FILE = 'RL/data/processed_learning_data.csv'
TRANSFER_FILE = 'RL/data/processed_transfer_data.csv'

MODELS = None  # None = every model in the pickle, or e.g. ['QLearning+novel', 'Advantage+novel']


def print_skew(fit_data: dict, dataloader: DataLoader) -> None:
    """
    Print the skew of each model's per-participant AIC/BIC, and of the differences from the best model
    (the values the paired t-tests are run on). Fit values are computed exactly as in compute_criterions.
    """
    learning, transfer = dataloader.get_data()
    number_trials = learning.groupby('participant').size().values[0] + transfer.groupby('participant').size().values[0]

    for criterion in ['AIC', 'BIC']:
        penalty = 2 if criterion == 'AIC' else np.log(number_trials)
        values = pd.DataFrame({model: penalty * (data.shape[1] - 4) + 2 * data.set_index('participant')['fit']
                               for model, data in fit_data.items()}).astype(float)
        best_model = values.mean().idxmin()
        differences = values.drop(columns=best_model).sub(values[best_model], axis=0)

        skew = pd.DataFrame({f'{criterion} skew': values.skew(),
                             f'Difference from {best_model} skew': differences.skew()}).reindex(values.columns)
        print('')
        print(f'{criterion} SKEW (|skew| > 1 = highly skewed)')
        print('==========')
        print(skew.round(2).to_string(na_rep='(reference)'))
        print('==========')


if __name__ == '__main__':

    print(f'Loading fits from {FIT_DATA_FILE}...')
    with open(FIT_DATA_FILE, 'rb') as f:
        fit_data = pickle.load(f)
    models = list(fit_data.keys()) if MODELS is None else MODELS
    fit_data = {model: fit_data[model] for model in models}
    print(f'  Models: {models}')
    print(f'  Participants: {fit_data[models[0]]["participant"].nunique()}')

    print('Loading empirical data...')
    dataloader = DataLoader(LEARNING_FILE, TRANSFER_FILE, number_of_participants=0)

    # recovery=None matches the empirical fit pipeline (also prints the fit report)
    print('Computing AIC/BIC and plotting...')
    os.makedirs('RL/plots', exist_ok=True)
    os.makedirs('RL/fits', exist_ok=True)
    Analyses().compute_criterions(dataloader, fit_data, models, dataloader.get_group_ids(), recovery=None)

    print_skew(fit_data, dataloader)

    print(f'Done. Figures saved to {os.path.join(REPO_ROOT, "RL", "plots")}')
