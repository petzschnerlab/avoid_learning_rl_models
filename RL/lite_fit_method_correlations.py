"""
Correlate Advantage model parameter fits between two fitting methods, across the same participants.

Loads the Advantage+novel fits from two fit_data pickles (one per fitting method), matches them by
participant, and correlates each fitted parameter between the two methods.

Run from the repo root (same as RL_fit_main.py):
    python RL/lite_fit_method_correlations.py

Inputs:
    RL/modelling/fit_data_FIT.pkl        fits from the current method
    RL/modelling/fit_data_FIT_orig.pkl   fits from the original method

Output:
    Printed table of correlations for each Advantage parameter
    RL/stats/advantage_fit_method_correlations.csv
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.dont_write_bytecode = True

# Work from the repo root
RL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RL_DIR)
os.chdir(REPO_ROOT)

# =========================================== #
# ================= INPUTS ================== #
# =========================================== #

FIT_FILE = 'RL/modelling/fit_data_FIT.pkl'          # current method
FIT_FILE_ORIG = 'RL/modelling/fit_data_FIT_orig.pkl'  # original method

MODEL = 'Advantage+novel'
METADATA_COLUMNS = ['participant', 'pain_group', 'run', 'fit']  # non-parameter columns in fit_data
INCLUDE_FIT = True  # also correlate the negative log-likelihood ('fit') between methods

SAVE_FILE = 'RL/stats/advantage_fit_method_correlations.csv'


def load_model_fits(filename: str, model: str) -> pd.DataFrame:
    """Load one model's per-participant fits from a fit_data pickle."""
    with open(filename, 'rb') as f:
        fit_data = pickle.load(f)
    if model not in fit_data:
        raise KeyError(f"'{model}' not found in {filename}. Available models: {list(fit_data.keys())}")
    return fit_data[model].copy()


if __name__ == '__main__':

    # Load both sets of fits
    fits_new = load_model_fits(FIT_FILE, MODEL)
    fits_orig = load_model_fits(FIT_FILE_ORIG, MODEL)
    print(f'{MODEL}')
    print(f'  {FIT_FILE}: {len(fits_new)} participants')
    print(f'  {FIT_FILE_ORIG}: {len(fits_orig)} participants')

    # Parameters present in both files
    parameters = [c for c in fits_new.columns if c not in METADATA_COLUMNS and c in fits_orig.columns]
    missing = sorted(set(fits_new.columns).symmetric_difference(fits_orig.columns) - set(METADATA_COLUMNS))
    if missing:
        print(f'  Columns not in both files (skipped): {missing}')
    if INCLUDE_FIT:
        parameters = parameters + ['fit']

    # Match participants across methods
    merged = fits_new[['participant'] + parameters].merge(fits_orig[['participant'] + parameters],
                                                          on='participant', suffixes=('_new', '_orig'))
    print(f'  Participants in both files: {len(merged)}')
    if len(merged) < len(fits_new) or len(merged) < len(fits_orig):
        print('  Note: some participants are only in one file and were excluded.')

    # Correlate each parameter between methods
    results = []
    for parameter in parameters:
        x, y = merged[f'{parameter}_new'].astype(float), merged[f'{parameter}_orig'].astype(float)
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_rho, spearman_p = stats.spearmanr(x, y)
        results.append({'parameter': 'fit (NLL)' if parameter == 'fit' else parameter,
                        'n': len(x),
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_rho': spearman_rho,
                        'spearman_p': spearman_p,
                        'mean_new': x.mean(),
                        'mean_orig': y.mean(),
                        'mean_abs_difference': (x - y).abs().mean()})
    results = pd.DataFrame(results).set_index('parameter')

    # Save and print
    os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
    results.to_csv(SAVE_FILE)

    print('')
    print(f'{MODEL.upper()} PARAMETER CORRELATIONS BETWEEN FITTING METHODS')
    print('==========')
    print(results.to_string(float_format=lambda v: f'{v:.4f}'))
    print('==========')
    print(f'Saved to {SAVE_FILE}')
