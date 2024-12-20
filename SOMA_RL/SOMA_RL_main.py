import sys
sys.dont_write_bytecode = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import multiprocessing as mp
import time
import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats

from helpers.tasks import AvoidanceLearningTask
from helpers.rl_models import RLModel
from helpers.plotting import plot_simulations

class RLPipeline:
        
        """
        Reinforcement Learning Pipeline

        Parameters
        ----------
        task : object
            Task object
        model : object
            Reinforcement learning model object
        task_design : dict
            Dictionary containing task design parameters
        """

        def __init__(self, model, task=None):

            #Get parameters
            self.task_design = task.task_design
            self.task = task
            self.task.initiate_model(model)

        def simulate(self):

            #Run simulation and computations
            self.task.run_experiment()
            self.task.rl_model.run_computations()

            return self.task.rl_model
    
        def fit(self, data, bounds):

            return self.task.rl_model.fit(data, bounds)
        
class RLFit:

    """
    Reinforcement Learning Model Fitting

    """
    def run_fit(self, args):

        #Extract args
        pipeline, data, columns, participant = args
        model_name = pipeline.task.rl_model.__class__.__name__
        participant_id = data['learning']['participant'].values[0]
        pain_group = data['learning']['pain_group'].values[0]

        #Extract participant data
        learning_states = data['learning']['state'].values
        learning_actions = data['learning']['action'].values
        learning_rewards = data['learning'][['reward_L', 'reward_R']].values

        transfer_states = data['transfer']['state'].values
        transfer_actions = data['transfer']['action'].values

        data = [(learning_states, learning_actions, learning_rewards), (transfer_states, transfer_actions)]
        
        #Fit models
        fit_results, fitted_params = pipeline.fit(data, bounds=pipeline.task.rl_model.bounds)
        
        #Store fit results
        participant_fitted = [participant_id, pain_group, fit_results.fun]
        participant_fitted.extend([fitted_params[key] for key in columns[3:]])

        #Save to csv file
        with open(f'SOMA_RL/data/fits/{model_name}_{participant}_fit_results.csv', 'a') as f:
            f.write(','.join([str(x) for x in participant_fitted]) + '\n')

    def run_simulations(self, args):

        #Extract args
        pipeline, columns, participant, group, n = args
        
        #Run simulation and computations
        model = pipeline.simulate()
        model_name = model.__class__.__name__

        #Extract model data
        task_learning_data = model.task_learning_data
        task_learning_data['trial_total'] = task_learning_data.groupby('state_id').cumcount()+1
        if 'v_prediction_errors' in task_learning_data.columns:
            task_learning_data['prediction_errors'] = task_learning_data['v_prediction_errors'] + task_learning_data['q_prediction_errors']
        task_learning_data['averaged_pe'] = task_learning_data['prediction_errors'].apply(lambda x: sum(x)/len(x))
        learning_accuracy = task_learning_data.groupby(['context', 'trial_total'])['accuracy'].mean().reset_index()
        learning_prediction_errors = task_learning_data.groupby(['context', 'trial_total'])['averaged_pe'].mean().reset_index()

        value_labels = {'QLearning': 'q_values', 
                        'ActorCritic': 'w_values', 
                        'Relative': 'q_values', 
                        'Hybrid2012': 'h_values', 
                        'Hybrid2021': 'h_values', 
                        'QRelative': 'm_values', 
                        'wRelative': 'q_values'}
        value_label = value_labels[model_name]

        task_learning_data[f'{value_label}1'] = task_learning_data[value_label].apply(lambda x: x[0])
        task_learning_data[f'{value_label}2'] = task_learning_data[value_label].apply(lambda x: x[1])
        learning_values1 = task_learning_data.groupby(['context', 'trial_total'])[f'{value_label}1'].mean().reset_index()
        learning_values2 = task_learning_data.groupby(['context', 'trial_total'])[f'{value_label}2'].mean().reset_index()[f'{value_label}2']
        learning_values = pd.concat([learning_values1, learning_values2], axis=1)
        learning_values.columns = ['context', 'trial_total', 'values1', 'values2']
        
        learning_accuracy['run'] = n
        learning_prediction_errors['run'] = n
        learning_values['run'] = n

        #Store data
        accuracy = pd.DataFrame(learning_accuracy, columns=columns['accuracy'])
        prediction_errors = pd.DataFrame(learning_prediction_errors, columns=columns['pe'])
        values = pd.DataFrame(learning_values, columns=columns['values'])
        choice_rates = pd.DataFrame([model.choice_rate], columns=columns['choice_rate'])

        #Save to csv file
        accuracy.to_csv(f'SOMA_RL/data/fits/{model_name}_{group}_{participant}_accuracy_sim_results.csv', index=False)
        prediction_errors.to_csv(f'SOMA_RL/data/fits/{model_name}_{group}_{participant}_pe_sim_results.csv', index=False)
        values.to_csv(f'SOMA_RL/data/fits/{model_name}_{group}_{participant}_values_sim_results.csv', index=False)
        choice_rates.to_csv(f'SOMA_RL/data/fits/{model_name}_{group}_{participant}_choice_sim_results.csv', index=False)

    def multiprocessing_progress(self, num_files):
        last_count = 0
        loop = tqdm.tqdm(range(num_files))
        while len(os.listdir('SOMA_RL/data/fits')) < num_files:
            n_files = len(os.listdir('SOMA_RL/data/fits'))
            if n_files > last_count:
                loop.update(n_files-last_count)
                last_count = n_files
            time.sleep(1)
        loop.update(num_files-last_count)

if __name__ == "__main__":

    # =========================================== #
    # ================= SYSTEM ================== #
    # =========================================== #

    rnd.seed(1251)
    multiprocessing = True

    # =========================================== #
    # ============== MODEL FITTING ============== #
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
    #Cut the dataframe to 50 participants: #TODO:This is for debugging, remove it later
    p_indices = learning_data['participant'].unique()[:10]
    learning_data = learning_data[learning_data['participant'].isin(p_indices)]
    transfer_data = transfer_data[transfer_data['participant'].isin(p_indices)]
    
    #Setup fit dataframe
    models = ['QLearning', 'ActorCritic', 'Relative', 'Hybrid2012', 'wRelative', 'QRelative']

    #Task setup
    task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                   'transfer_phase': {'times_repeated': 4}}
    
    # =========================================== #
    # =================== FIT =================== #
    # =========================================== #

    #Delete any existing files
    for f in os.listdir('SOMA_RL/data/fits'):
        os.remove(os.path.join('SOMA_RL','data','fits',f))

    #Assign classes to inputs array
    loop = tqdm.tqdm(range(len(learning_data['participant'].unique())*len(models))) if not multiprocessing else None
    inputs = []
    columns = {}
    for n, participant in enumerate(learning_data['participant'].unique()):
        for m in models:                
            participant_learning_data = learning_data[learning_data['participant'] == participant]
            participant_transfer_data = transfer_data[transfer_data['participant'] == participant]
            participant_data = {'learning': participant_learning_data.copy(), 'transfer': participant_transfer_data.copy()}
            task = AvoidanceLearningTask(task_design)
            model = RLModel(m).model
            pipeline = RLPipeline(model, task)
            columns[m] = ['participant', 'pain_group', 'fit'] + list(model.parameters.keys())
            if multiprocessing:
                inputs.append((pipeline, participant_data, columns[m], participant))
            else:
                loop.update(1)
                RLFit().run_fit((pipeline, participant_data, columns[m], participant))

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(RLFit().run_fit, inputs)
        pool.close()

        #Progress bar checking how many have completed
        RLFit().multiprocessing_progress(len(inputs))
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

    #Set up task
    task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                    'transfer_phase': {'times_repeated': 4}}

    #Run simulations
    number_of_metrics = 4
    loop = tqdm.tqdm(range(fit_data[models[0]]['participant'].nunique()*len(models)*number_of_metrics)) if not multiprocessing else None
    inputs = []
    for n, participant in enumerate(fit_data[models[0]]['participant'].unique()):
        for m in models:

            #Participant data
            participant_params = fit_data[m][fit_data[m]['participant'] == participant].copy()
            group = participant_params['pain_group'].values[0]

            #Initialize task, model, and task design
            task = AvoidanceLearningTask(task_design)
            model = RLModel(m, participant_params).model
            pipeline = RLPipeline(model, task)
            if multiprocessing:
                inputs.append((pipeline, columns, participant, group, n))
            else:
                loop.update(1)
                RLFit().run_simulations((pipeline, columns, participant, group, n))

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(RLFit().run_simulations, inputs)
        pool.close()

        #Progress bar checking how many have completed
        RLFit().multiprocessing_progress(len(inputs)*number_of_metrics)
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