import sys
sys.dont_write_bytecode = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import multiprocessing as mp
import copy
import random as rnd
import pandas as pd
import numpy as np
import tqdm

from helpers.dataloader import DataLoader
from helpers.rl_models import RLModel
from helpers.tasks import AvoidanceLearningTask
from helpers.pipeline import RLPipeline, mp_run_fit, mp_run_simulations, mp_progress
from helpers.plotting import plot_simulations

if __name__ == "__main__":

    # =========================================== #
    # ================= INPUTS ================== #
    # =========================================== #

    #Seed random number generator
    rnd.seed(1251)

    #Debug parameters
    number_of_participants = 4 #Number of participants to keep, 0 = all

    #Parameters
    multiprocessing = True #Whether to run fits and simulations in parallel
    fit_transfer_phase = True #Whether to fit the transfer phase
    transfer_trials = 0 #Number of times to present each stimulus pair in the transfer phase for fitting, 0 = all
    number_of_fits = 1 #Number of times to fit the dataset for each participant

    #File names
    learning_filename = 'SOMA_RL/data/pain_learning_processed.csv'
    transfer_filename = 'SOMA_RL/data/pain_transfer_processed.csv'

    #Models
    '''
    Supported models: 
        QLearning, ActorCritic
        Relative, wRelative, QRelative
        Hybrid2012, Hybrid2021

    Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign
        +bias: Adds a valence bias to the model (e.g. wRelative+bias), only usable with wRelative, QRelative, Hybrid2012, and Hybrid2021
        +novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models
    '''

    models = ['QLearning', 
              'ActorCritic', 

              'Relative',
              'Relative+novel', 
              'wRelative+bias+novel', 
              'wRelative+novel',

              'Hybrid2012',
              'Hybrid2012+bias+novel', 
              'Hybrid2012+novel', 
              'Hybrid2021',
              'Hybrid2021+bias+novel', 
              'Hybrid2021+novel']
        
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
    for f in os.listdir('SOMA_RL/data/fits'):
        os.remove(os.path.join('SOMA_RL','data','fits',f))

    #Run fits
    loop = tqdm.tqdm(range(len(participant_ids)*len(models))) if not multiprocessing else None
    inputs = []
    columns = {}
    for n, participant in enumerate(participant_ids):
        for model_name in models:  
            p_dataloader = copy.copy(dataloader)
            p_dataloader.filter_participant_data(participant)  
            model = RLModel(model_name)
            task = AvoidanceLearningTask()
            pipeline = RLPipeline(model, p_dataloader, task)
            columns[model_name] = ['participant', 'pain_group', 'fit'] + list(model.get_parameters()) + ['skipped_parameters']
            if multiprocessing:
                inputs.append((pipeline, columns[model_name], participant))
            else:
                pipeline.run_fit((columns[model_name], participant))
                loop.update(1)

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')
        print(f"Number of participants: {len(participant_ids)}, Number of models: {len(models)}, Total fits: {len(participant_ids)*len(models)}")

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(mp_run_fit, inputs)
        pool.close()

        #Progress bar checking how many have completed
        mp_progress(len(inputs))
        print('\nMultiprocessing complete!')

    #Load all data in the fit data
    files = os.listdir('SOMA_RL/data/fits')
    fit_data = {model: pd.DataFrame(columns=columns[model]) for model in models}
    for f in files:
        model_name, participant = f.split('_')[:2]
        if model_name in models:
            participant_data = pd.read_csv(os.path.join('SOMA_RL','data','fits',f), header=None)
            participant_data.columns = columns[model_name]
            nan_keys = [key for key in participant_data if participant_data[key].isnull().any()]
            for key in nan_keys:
                participant_data[key] = None

            if len(fit_data[model_name]) == 0:
                fit_data[model_name] = participant_data
            else:
                fit_data[model_name] = pd.concat((fit_data[model_name], participant_data), ignore_index=True)
    
    #Delete all files
    for f in files:
        os.remove(os.path.join('SOMA_RL','data','fits',f))

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
            number_params = len(group_fit.columns) - 4 - group_fit['skipped_parameters'].values[0]
            number_samples = dataloader.get_num_samples_by_group(group)
            group_AIC[model_name][group] = 2*number_params + 2*total_NLL
            group_BIC[model_name][group] = np.log(number_samples)*number_params + 2*total_NLL
                    
        total_NLL = np.sum(fit_data[model_name]["fit"])
        number_params = len(fit_data[model_name].columns) - 4 - fit_data[model_name]['skipped_parameters'][0]
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
        for col in fit_data[model_name].columns[2:-1]:
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
        print(f"Number of participants: {fit_data[models[0]]['participant'].nunique()}, Number of models: {len(models)}, Total simulations: {fit_data[models[0]]['participant'].nunique()*len(models)}")

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(mp_run_simulations, inputs)
        pool.close()

        #Progress bar checking how many have completed
        mp_progress(len(inputs)*number_of_metrics, divide_by=number_of_metrics)
        print('\nMultiprocessing complete!')

    #Load all data in the fit data
    files = os.listdir('SOMA_RL/data/fits')
    accuracy = {group: {model: pd.DataFrame(columns=accuracy_columns) for model in models} for group in group_ids}
    prediction_errors = {group: {model: pd.DataFrame(columns=pe_columns) for model in models} for group in group_ids}
    values = {group: {model: pd.DataFrame(columns=values_columns) for model in models} for group in group_ids}
    choice_rates = {group: {model: pd.DataFrame(columns=transfer_columns) for model in models} for group in group_ids}

    for f in files:
        model_name, group, participant, value_name = f.split('_')[:4]
        participant_data = pd.read_csv(os.path.join('SOMA_RL','data','fits',f))

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
        os.remove(os.path.join('SOMA_RL','data','fits',f))
        
    #Plot simulations 
    for group in accuracy:
        plot_simulations(accuracy[group], prediction_errors[group], values[group], choice_rates[group], models, group, dataloader)

    #Debug print
    print('')