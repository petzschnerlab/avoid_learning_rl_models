"""
Re-plot the model-behaviour figures (plot_simulations_behaviours) from saved
results, without re-running any fits or simulations.

Run from the repo root (same as RL_fit_main.py):
    python RL/plot_model_behaviours.py

Inputs (written by a previous fit run):
    RL/fits/modelsimulation_accuracy_data.csv
    RL/fits/modelsimulation_choice_data.csv
    RL/data/processed_learning_data.csv   (empirical data, for the diamonds/dashed lines)
    RL/data/processed_transfer_data.csv

Outputs:
    RL/plots/model_behaviours/<model>_model_behaviours_supplemental.png/.svg  (raincloud)
    RL/plots/model_behaviours/<model>_model_behaviours.png/.svg               (bar)
"""

import os
import sys
import copy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.dont_write_bytecode = True

# Make the script work no matter where it is launched from:
# the repo code uses paths like 'RL/plots/...' relative to the repo root,
# and imports like 'helpers.analyses' relative to the RL folder.
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
sys.path.insert(0, RL_DIR)
os.chdir(REPO_ROOT)

import pandas as pd
from helpers.analyses import Analyses
from helpers.dataloader import DataLoader

# =========================================== #
# ================= INPUTS ================== #
# =========================================== #

SIM_DIR = 'RL/fits'  # use 'RL/modelling' for the older (June) results
ACCURACY_FILE = os.path.join(SIM_DIR, 'modelsimulation_accuracy_data.csv')
CHOICE_FILE = os.path.join(SIM_DIR, 'modelsimulation_choice_data.csv')

LEARNING_FILE = 'RL/data/processed_learning_data.csv'
TRANSFER_FILE = 'RL/data/processed_transfer_data.csv'

MODELS = None  # None = every model in the CSVs, or e.g. ['QLearning+novel']
PLOT_TYPES = ['raincloud', 'bar']
SUBPLOT_TITLE = ['a. Learning Phase', 'b. Transfer Phase']
BINNED_TRIAL = True


def load_simulations(accuracy_file: str, choice_file: str, models: list = None) -> tuple:
    """
    Rebuild the accuracy and choice-rate dicts ({model: {group: DataFrame}})
    exactly as Analyses.run_fit_simulations builds them before plotting.
    """
    accuracy_data = pd.read_csv(accuracy_file)
    choice_data = pd.read_csv(choice_file)

    if models is None:
        models = list(accuracy_data['model'].unique())
    groups = list(accuracy_data['group'].unique())

    accuracy, choice_rates = {}, {}
    for model in models:
        accuracy[model], choice_rates[model] = {}, {}
        for group in groups:
            acc = accuracy_data[(accuracy_data['model'] == model) & (accuracy_data['group'] == group)]
            acc = acc[['context', 'trial_total', 'accuracy', 'run']]
            # Average across states (as in run_fit_simulations); accuracy is already in %
            accuracy[model][group] = acc.groupby(['context', 'trial_total', 'run']).mean().reset_index()

            ch = choice_data[(choice_data['model'] == model) & (choice_data['group'] == group)]
            choice_rates[model][group] = ch[['A', 'B', 'E', 'F', 'N']].reset_index(drop=True)

    return accuracy, choice_rates, models, groups


if __name__ == '__main__':

    print('Loading simulation results...')
    accuracy, choice_rates, models, groups = load_simulations(ACCURACY_FILE, CHOICE_FILE, MODELS)
    print(f'  Models: {models}')
    print(f'  Groups: {groups}')

    print('Loading empirical data...')
    dataloader = DataLoader(LEARNING_FILE, TRANSFER_FILE, number_of_participants=0, reduced=False)

    analyses = Analyses()
    for plot_type in PLOT_TYPES:
        print(f'Plotting ({plot_type})...')
        # deepcopy: the plotting method adds columns to the dicts in place
        analyses.plot_simulations_behaviours(copy.deepcopy(accuracy),
                                             copy.deepcopy(choice_rates),
                                             models,
                                             groups,
                                             dataloader,
                                             binned_trial=BINNED_TRIAL,
                                             subplot_title=SUBPLOT_TITLE,
                                             plot_type=plot_type)

    # Combine the (bar) behaviour plot with the saved parameter plots in RL/plots/fits
    # (same calls as the end of Analyses.run_fit_simulations)
    print('Combining with parameter plots...')
    for model in models:
        analyses.plot_combined_simulations(f'{model}_combined_model_behaviours', f'{model}_model_behaviours', f'{model}-model-fits')
        analyses.plot_combined_simulations(f'{model}_combined_sig_model_behaviours', f'{model}_model_behaviours', f'{model}-model-fits-significant', orientation='horizontal')

    print(f'Done. Figures saved to {os.path.join(REPO_ROOT, "RL", "plots", "model_behaviours")}')
