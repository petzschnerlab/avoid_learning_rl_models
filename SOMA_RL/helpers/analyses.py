import sys
sys.dont_write_bytecode = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import multiprocessing as mp
import copy
import pandas as pd
import numpy as np
import tqdm
import pickle

from models.rl_models import RLModel
from helpers.dataloader import DataLoader
from helpers.tasks import AvoidanceLearningTask
from helpers.pipeline import RLPipeline, mp_run_fit, mp_run_simulations, mp_progress
from helpers.plotting import plot_simulations

def run_fit_analysis(learning_filename, transfer_filename, models, number_of_participants, random_params, number_of_runs, multiprocessing):
    # =========================================== #
    # ================ LOAD DATA ================ #
    # =========================================== #

    dataloader = DataLoader(learning_filename, transfer_filename, number_of_participants)
    participant_ids = dataloader.get_participant_ids()
    group_ids = dataloader.get_group_ids()

    # =========================================== #
    # =================== FIT =================== #
    # =========================================== #

    #Delete any existing files
    for f in os.listdir('SOMA_RL/fits/temp'):
        os.remove(os.path.join('SOMA_RL','fits','temp',f))

    #Run fits
    print('\n')
    print(f"Number of participants: {len(participant_ids)}, Number of models: {len(models)}, Number of runs: {number_of_runs}, Total fits: {len(participant_ids)*len(models)*number_of_runs}")
    loop = tqdm.tqdm(range(len(participant_ids)*len(models)*number_of_runs)) if not multiprocessing else None
    inputs = []
    columns = {}
    for n, participant in enumerate(participant_ids):
        for model_name in models:  
            p_dataloader = copy.copy(dataloader)
            p_dataloader.filter_participant_data(participant)  
            for run in range(number_of_runs):
                model = RLModel(model_name, random_params=random_params)
                task = AvoidanceLearningTask()
                pipeline = RLPipeline(model, p_dataloader, task)
                columns[model_name] = ['participant', 'pain_group', 'run', 'fit',] + list(model.get_parameters())
                if multiprocessing:
                    inputs.append((pipeline, columns[model_name], participant, run))
                else:
                    pipeline.run_fit((columns[model_name], participant, run))
                    loop.update(1)

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(mp_run_fit, inputs)
        pool.close()

        #Progress bar checking how many have completed
        mp_progress(len(inputs), multiply_by=number_of_runs)
        print('\nMultiprocessing complete!')

    #Load all data in the fit data
    files = os.listdir('SOMA_RL/fits/temp')
    fit_data = {model: pd.DataFrame(columns=columns[model]) for model in models}
    for f in files:
        model_name, participant = f.split('_')[:2]
        if model_name in models:
            participant_data = pd.read_csv(os.path.join('SOMA_RL','fits','temp',f), header=None)
            participant_data.columns = columns[model_name]
            participant_data = participant_data.iloc[[participant_data['fit'].idxmin()]]
            nan_keys = [key for key in participant_data if participant_data[key].isnull().any()]
            for key in nan_keys:
                participant_data[key] = None

            if len(fit_data[model_name]) == 0:
                fit_data[model_name] = participant_data
            else:
                fit_data[model_name] = pd.concat((fit_data[model_name], participant_data), ignore_index=True)

    #Save fit data as a pickle file
    with open('SOMA_RL/fits/fit_data.pkl', 'wb') as f:
        pickle.dump(fit_data, f)

    #Delete all files
    for f in files:
        os.remove(os.path.join('SOMA_RL','fits','temp',f))

    # =========================================== #
    # =============== REPORT FIT ================ #
    # =========================================== #

    group_AIC = {m: {} for m in models}
    group_BIC = {m: {} for m in models}
    for model_name in models:

        #Compute AIC and BIC
        for group in group_ids:
            group_fit = fit_data[model_name][fit_data[model_name]['pain_group'] == group]
            total_NLL = np.sum(group_fit["fit"])
            number_params = len(group_fit.columns) - 3
            number_samples = dataloader.get_num_samples_by_group(group)
            group_AIC[model_name][group] = 2*number_params + 2*total_NLL
            group_BIC[model_name][group] = np.log(number_samples)*number_params + 2*total_NLL
                    
        total_NLL = np.sum(fit_data[model_name]["fit"])
        number_params = len(fit_data[model_name].columns) - 3
        number_samples = dataloader.get_num_samples()
        AIC = 2*number_params + 2*total_NLL
        BIC = np.log(number_samples)*number_params + 2*total_NLL
        group_AIC[model_name]['full'] = AIC
        group_BIC[model_name]['full'] = BIC

        print('')
        print(f'FIT REPORT: {model_name}')
        print('==========')
        print(f'AIC: {AIC.round(0)}')
        print(f'BIC: {BIC.round(0)}')
        for col in fit_data[model_name].columns[3:]:
            if fit_data[model_name][col][0] is not None:
                print(f'{col}: {fit_data[model_name][col].mean().round(4)}, {fit_data[model_name][col].std().round(4)}')
        print('==========')
        print('')
        
    #Turn nested dictionary into dataframe
    group_AIC = pd.DataFrame(group_AIC)
    group_AIC['best_model'] = group_AIC.idxmin(axis=1)
    group_AIC.to_csv('SOMA_RL/plots/group_AIC.csv')

    group_BIC = pd.DataFrame(group_BIC)
    group_BIC['best_model'] = group_BIC.idxmin(axis=1)
    group_BIC.to_csv('SOMA_RL/plots/group_BIC.csv')

    print('AIC REPORT')
    print('==========')
    print(group_AIC)
    print('==========')

    print('')
    print('BIC REPORT')
    print('==========')
    print(group_BIC)
    print('==========')

    # =========================================== #
    # =========== SIMULATE WITH FITS ============ #
    # =========================================== #

    dataloader = DataLoader(learning_filename, transfer_filename, number_of_participants, reduced=False)

    #Set up data columns
    accuracy_columns = ['context', 'trial_total', 'accuracy', 'run']
    pe_columns = ['context', 'trial_total', 'averaged_pe', 'run']
    values_columns = ['context', 'trial_total', 'values1', 'values2', 'run']
    transfer_columns = ['A', 'B', 'E', 'F', 'N']
    columns = {'accuracy': accuracy_columns, 'pe': pe_columns, 'values': values_columns, 'choice_rate': transfer_columns}

    #Run simulations
    number_of_metrics = len(columns)
    print(f"\nNumber of participants: {len(participant_ids)}, Number of models: {len(models)}, Total simulations: {len(participant_ids)*len(models)}")
    loop = tqdm.tqdm(range(fit_data[models[0]]['participant'].nunique()*len(models))) if not multiprocessing else None
    inputs = []
    for n, participant in enumerate(participant_ids):
        for model_name in models:

            #Participant data
            participant_params = fit_data[model_name][fit_data[model_name]['participant'] == participant].copy()
            group = participant_params['pain_group'].values[0]
            
            #Initialize task, model, and task design
            p_dataloader = copy.copy(dataloader)
            p_dataloader.filter_participant_data(participant) 
            model = RLModel(model_name, participant_params)
            task = AvoidanceLearningTask()
            pipeline = RLPipeline(model, p_dataloader, task)
            if multiprocessing:
                inputs.append((pipeline, columns, participant, group, n))
            else:
                loop.update(1)
                pipeline.run_simulations((columns, participant, group, n))

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(mp_run_simulations, inputs)
        pool.close()

        #Progress bar checking how many have completed
        mp_progress(len(inputs)*number_of_metrics, divide_by=number_of_metrics)
        print('\nMultiprocessing complete!')

    #Load all data in the fit data
    files = os.listdir('SOMA_RL/fits/temp')
    accuracy = {group: {model: pd.DataFrame(columns=accuracy_columns) for model in models} for group in group_ids}
    prediction_errors = {group: {model: pd.DataFrame(columns=pe_columns) for model in models} for group in group_ids}
    values = {group: {model: pd.DataFrame(columns=values_columns) for model in models} for group in group_ids}
    choice_rates = {group: {model: pd.DataFrame(columns=transfer_columns) for model in models} for group in group_ids}

    for f in files:
        model_name, group, participant, value_name = f.split('_')[:4]
        participant_data = pd.read_csv(os.path.join('SOMA_RL','fits','temp',f))

        if value_name == 'accuracy':
            if len(accuracy[group][model_name]) == 0:
                accuracy[group][model_name] = participant_data
            else:
                accuracy[group][model_name] = pd.concat((accuracy[group][model_name], participant_data), ignore_index=True)
        elif value_name == 'pe':
            if len(prediction_errors[group][model_name]) == 0:
                prediction_errors[group][model_name] = participant_data
            else:
                prediction_errors[group][model_name] = pd.concat((prediction_errors[group][model_name], participant_data), ignore_index=True)
        elif value_name == 'values':
            if len(values[group][model_name]) == 0:
                values[group][model_name] = participant_data
            else:
                values[group][model_name] = pd.concat((values[group][model_name], participant_data), ignore_index=True)
        elif value_name == 'choice':
            if len(choice_rates[group][model_name]) == 0:
                choice_rates[group][model_name] = participant_data
            else:
                choice_rates[group][model_name] = pd.concat((choice_rates[group][model_name], participant_data), ignore_index=True)

    #Delete all files
    for f in files:
        os.remove(os.path.join('SOMA_RL','fits','temp',f))
        
    #Plot simulations 
    for group in accuracy:
        plot_simulations(accuracy[group], prediction_errors[group], values[group], choice_rates[group], models, group, dataloader)

    #Debug print
    print('')