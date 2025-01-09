import sys
sys.dont_write_bytecode = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import multiprocessing as mp
import random as rnd
import pandas as pd
import numpy as np
import tqdm

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
    n = 0 #Number of participants to keep, 0 = all

    #Parameters
    multiprocessing = True
    fit_transfer_phase = True
    transfer_trials = 2
    number_of_fits = 5

    #Models and simulation task design
    models = ['QLearning', 'ActorCritic', 'Relative', 'Hybrid2012', 'wRelative', 'QRelative']
    task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                   'transfer_phase': {'times_repeated': 4}}

    # =========================================== #
    # ================ LOAD DATA ================ #
    # =========================================== #

    #Load data
    learning_filename = 'SOMA_RL/data/pain_learning_processed.csv'
    learning_data = pd.read_csv(learning_filename)
    transfer_filename = 'SOMA_RL/data/pain_transfer_processed.csv'
    transfer_data = pd.read_csv(transfer_filename)

    #Reorganize data so that left stim is always better
    learning_data['stim_order'] = learning_data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
    learning_data['reward_L'] = learning_data.apply(lambda x: x['feedback_L']/10 if x['stim_order'] else x['feedback_R']/10, axis=1)
    learning_data['reward_R'] = learning_data.apply(lambda x: x['feedback_R']/10 if x['stim_order'] else x['feedback_L']/10, axis=1)
    learning_data['action'] = learning_data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)

    learning_data = learning_data[['participant_id', 'group_code', 'symbol_names', 'reward_L', 'reward_R', 'action']]
    learning_data['symbol_names'] = learning_data['symbol_names'].replace({'Reward1': 'State AB', 'Reward2': 'State CD', 'Punish1': 'State EF', 'Punish2': 'State GH'})
    learning_data.columns = ['participant', 'pain_group', 'state', 'reward_L', 'reward_R', 'action']

    transfer_data['stim_order'] = transfer_data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
    transfer_data['symbol_L_name'] = transfer_data['symbol_L_name'].replace({'75R1': 'A', '25R1': 'B', '75R2': 'C', '25R2': 'D', '75P1': 'E', '25P1': 'F', '75P2': 'G', '25P2': 'H', 'Zero': 'N'})
    transfer_data['symbol_R_name'] = transfer_data['symbol_R_name'].replace({'75R1': 'A', '25R1': 'B', '75R2': 'C', '25R2': 'D', '75P1': 'E', '25P1': 'F', '75P2': 'G', '25P2': 'H', 'Zero': 'N'})
    transfer_data['state'] = transfer_data.apply(lambda x: f"State {x['symbol_L_name']}{x['symbol_R_name']}" if x['stim_order'] else f"State {x['symbol_R_name']}{x['symbol_L_name']}", axis=1)
    transfer_data['action'] = transfer_data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)
    
    transfer_data = transfer_data[['participant_id', 'group_code', 'state', 'action']]
    transfer_data.columns = ['participant', 'pain_group', 'state', 'action']

    #DEBUGGING CLEANUP
    #Cut the dataframe to n participants: #TODO:This is for debugging, remove it later
    if n > 0:
        p_indices = learning_data['participant'].unique()[:n]
        learning_data = learning_data[learning_data['participant'].isin(p_indices)]
        transfer_data = transfer_data[transfer_data['participant'].isin(p_indices)]
    
    # =========================================== #
    # =================== FIT =================== #
    # =========================================== #

    #Delete any existing files
    for f in os.listdir('SOMA_RL/data/fits'):
        os.remove(os.path.join('SOMA_RL','data','fits',f))

    #Run fits
    loop = tqdm.tqdm(range(len(learning_data['participant'].unique())*len(models))) if not multiprocessing else None
    inputs = []
    columns = {}
    for n, participant in enumerate(learning_data['participant'].unique()):
        for m in models:                
            participant_learning_data = learning_data[learning_data['participant'] == participant]
            participant_transfer_data = transfer_data[transfer_data['participant'] == participant]
            model = RLModel(m)
            participant_data = {'learning': participant_learning_data.copy(), 'transfer': participant_transfer_data.copy()}
            task = AvoidanceLearningTask(task_design, transfer_trials=transfer_trials)
            pipeline = RLPipeline(model, task, fit_transfer_phase=fit_transfer_phase, number_of_fits=number_of_fits)
            columns[m] = ['participant', 'pain_group', 'fit'] + list(model.get_model().parameters.keys())
            if multiprocessing:
                inputs.append((pipeline, participant_data, columns[m], participant))
            else:
                loop.update(1)
                pipeline.run_fit((participant_data, columns[m], participant))

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')
        print(f"Number of participants: {len(learning_data['participant'].unique())}, Number of models: {len(models)}, Total fits: {len(learning_data['participant'].unique())*len(models)}")

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
    for m in models:

        #Compute AIC and BIC
        for group in learning_data['pain_group'].unique():
            group_fit = fit_data[m][fit_data[m]['pain_group'] == group]
            total_NLL = np.sum(group_fit["fit"])
            number_params = len(group_fit.columns) - 3
            number_samples_learning = learning_data[learning_data['pain_group'] == group].shape[0]
            number_samples_transfer = transfer_data[transfer_data['pain_group'] == group].shape[0]
            number_samples = number_samples_learning + number_samples_transfer
            group_AIC[m][group] = 2*number_params - 2*total_NLL
            group_BIC[m][group] = np.log(number_samples)*number_params - 2*total_NLL
            
        total_NLL = np.sum(fit_data[m]["fit"])
        number_params = len(fit_data[m].columns) - 3
        number_samples = learning_data.shape[0] + transfer_data.shape[0]
        AIC = 2*number_params - 2*total_NLL
        BIC = np.log(number_samples)*number_params - 2*total_NLL

        print('')
        print(f'FIT REPORT: {m}')
        print('==========')
        print(f'AIC: {AIC.round(0)}')
        print(f'BIC: {BIC.round(0)}')
        for col in fit_data[m].columns[2:]:
            print(f'{col}: {fit_data[m][col].mean().round(4)}, {fit_data[m][col].std().round(4)}')
        print('==========')
        print('')
        
    #Turn nested dictionary into dataframe
    group_AIC = pd.DataFrame(group_AIC)
    group_AIC['best_model'] = group_AIC.idxmin(axis=1)
    group_BIC = pd.DataFrame(group_BIC)
    group_BIC['best_model'] = group_BIC.idxmin(axis=1)

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

    #Set up data columns
    accuracy_columns = ['context', 'trial_total', 'accuracy', 'run']
    pe_columns = ['context', 'trial_total', 'averaged_pe', 'run']
    values_columns = ['context', 'trial_total', 'values1', 'values2', 'run']
    transfer_columns = ['A', 'B', 'E', 'F', 'N']
    columns = {'accuracy': accuracy_columns, 'pe': pe_columns, 'values': values_columns, 'choice_rate': transfer_columns}

    #Run simulations
    number_of_metrics = len(columns)
    loop = tqdm.tqdm(range(fit_data[models[0]]['participant'].nunique()*len(models)*number_of_metrics)) if not multiprocessing else None
    inputs = []
    for n, participant in enumerate(fit_data[models[0]]['participant'].unique()):
        for m in models:

            #Participant data
            participant_params = fit_data[m][fit_data[m]['participant'] == participant].copy()
            group = participant_params['pain_group'].values[0]
            participant_learning_data = learning_data[learning_data['participant'] == participant]
            participant_transfer_data = transfer_data[transfer_data['participant'] == participant]
            participant_data = {'learning': participant_learning_data.copy(), 'transfer': participant_transfer_data.copy()}

            #Initialize task, model, and task design
            model = RLModel(m, participant_params)
            task = AvoidanceLearningTask(task_design)
            pipeline = RLPipeline(model, task, number_of_fits=number_of_fits)
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
    accuracy = {group: {model: pd.DataFrame(columns=accuracy_columns) for model in models} for group in learning_data['pain_group'].unique()}
    prediction_errors = {group: {model: pd.DataFrame(columns=pe_columns) for model in models} for group in learning_data['pain_group'].unique()}
    values = {group: {model: pd.DataFrame(columns=values_columns) for model in models} for group in learning_data['pain_group'].unique()}
    choice_rates = {group: {model: pd.DataFrame(columns=transfer_columns) for model in models} for group in learning_data['pain_group'].unique()}

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
        plot_simulations(accuracy[group], prediction_errors[group], values[group], choice_rates[group], models, group)

    print('debug')