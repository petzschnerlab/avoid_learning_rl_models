    
from helpers.dataloader import DataLoader
from helpers.analyses import run_fit_simulations, create_confusion_matrix, plot_model_fits, plot_parameter_fits
from helpers.priors import get_priors
from helpers.statistics import Statistics
import pandas as pd
from helpers.plotting import plot_parameter_rainclouds


import pickle
import copy
import numpy as np


if __name__ == "__main__":


    with open('SOMA_RL/model results/Final Modelling Results/fit_data_FIT.pkl', 'rb') as f:
        fit_data = pickle.load(f)

    for model in fit_data.keys():
        model_data = fit_data[model]
        print('***********************************')
        print('Model:', model)
        for group in model_data['pain_group'].unique():
            print('\n')
            print('Group:', group)
            group_data = model_data[model_data['pain_group'] == group]
            for parameter in list(group_data.keys())[4:]:
                if parameter not in ['novel_value', 'mixing_factor', 'valence_factor']: # Exclude parameters that are not to be log-transformed
                    if group_data[parameter].min() <= 0: 
                        group_data[parameter] = group_data[parameter] - group_data[parameter].min() + 1  # Shift the parameter to be positive if it has non-positive values
                    group_data[parameter] = np.log(group_data[parameter])  # Log-transform the parameter to reduce skewness
                print(parameter, np.round(group_data[parameter].mean(),2), np.round(group_data[parameter].std(),2))
    
    for model in fit_data:
        plot_parameter_rainclouds(f'{model}-model-fits', fit_data[model])