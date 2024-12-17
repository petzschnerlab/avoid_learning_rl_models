import sys
sys.dont_write_bytecode = True
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

        #Extract participant data
        states = data['state'].values
        actions = data['action'].values
        rewards = data[['reward_L', 'reward_R']].values
        
        #Fit models
        fit_results, fitted_params = pipeline.fit((states, actions, rewards), bounds=pipeline.task.rl_model.bounds)
        
        #Store fit results
        participant_fitted = [data['participant'].values[0], 
                            data['pain_group'].values[0], 
                            fit_results.fun]
        participant_fitted.extend([fitted_params[key] for key in columns[model_name][3:]])

        #Save to csv file
        with open(f'SOMA_RL/data/fits/{model_name}_{participant}_fit_results.csv', 'a') as f:
            f.write(','.join([str(x) for x in participant_fitted]) + '\n')

        
        '''
        if len(fit_data[m]) == 0:
            fit_data[m] = pd.DataFrame([participant_fitted], columns=fit_data[m].columns)
        else:
            fit_data[m] = pd.concat((fit_data[m], 
                                pd.DataFrame([participant_fitted], columns=fit_data[m].columns)),
                                ignore_index=True)
        '''

if __name__ == "__main__":

    rnd.seed(1251)
    multiprocessing = True

    # =========================================== #
    # ============== MODEL FITTING ============== #
    # =========================================== #

    #Load data
    filename = 'SOMA_RL/data/pain_learning_processed.csv'
    data = pd.read_csv(filename)

    #Reorganize data so that left stim is always better
    data['stim_order'] = data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
    data['reward_L'] = data.apply(lambda x: x['feedback_L']/10 if x['stim_order'] else x['feedback_R']/10, axis=1)
    data['reward_R'] = data.apply(lambda x: x['feedback_R']/10 if x['stim_order'] else x['feedback_L']/10, axis=1)
    data['action'] = data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)

    data = data[['participant_id', 'group_code', 'symbol_names', 'reward_L', 'reward_R', 'action']]
    data['symbol_names'] = data['symbol_names'].replace({'Reward1': 'State AB', 
                                                         'Reward2': 'State CD', 
                                                         'Punish1': 'State EF', 
                                                         'Punish2': 'State GH'})
    data.columns = ['participant', 'pain_group', 'state', 'reward_L', 'reward_R', 'action']

    #DEBUGGING CLEANUP
    #Cut the dataframe to 50 participants: #TODO:This is for debugging, remove it later
    #data = data[data['participant'].isin(data['participant'].unique()[:10])]
    
    #Setup fit dataframe
    models = ['QLearning', 'ActorCritic', 'Relative', 'Hybrid2012']
    
    columns = {'QLearning': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'temperature'],
               'ActorCritic': ['participant', 'pain_group', 'fit', 'factual_actor_lr', 'counterfactual_actor_lr', 'critic_lr', 'temperature', 'valence_factor'],
               'Relative': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'contextual_lr', 'temperature'],
               'Hybrid2012': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'factual_actor_lr', 'counterfactual_actor_lr', 'critic_lr', 'temperature', 'mixing_factor', 'valence_factor'],
               'Hybrid2021': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'factual_actor_lr', 'counterfactual_actor_lr', 'critic_lr', 'temperature', 'mixing_factor', 'valence_factor', 'noise_factor', 'decay_factor'],
               'QRelative': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'contextual_lr', 'temperature', 'mixing_factor']}

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
    loop = tqdm.tqdm(range(len(data['participant'].unique())*len(models))) if not multiprocessing else None
    inputs = []
    for n, participant in enumerate(data['participant'].unique()):
        for m in models:                
            participant_data = data[data['participant'] == participant]
            task = AvoidanceLearningTask(task_design)
            model = RLModel(m).model
            pipeline = RLPipeline(model, task)
            if multiprocessing:
                inputs.append((pipeline, participant_data.copy(), columns, participant))
            else:
                loop.update(1)
                RLFit().run_fit((pipeline, participant_data.copy(), columns, participant))

    #Run all models fits in parallel
    if multiprocessing:
        print('\nMultiprocessing initiated...')

        pool = mp.Pool(mp.cpu_count())
        pool.map_async(RLFit().run_fit, inputs)
        pool.close()

        #Progress bar checking how many have completed
        last_count = 0
        loop = tqdm.tqdm(range(len(inputs)))
        while len(os.listdir('SOMA_RL/data/fits')) < len(inputs):
            n_files = len(os.listdir('SOMA_RL/data/fits'))
            if n_files > last_count:
                loop.update(n_files-last_count)
                last_count = n_files
            time.sleep(1)
        loop.update(len(inputs)-last_count)
        
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

    for m in models:
        print('')
        print(f'FIT REPORT: {m}')
        print('==========')
        for col in fit_data[m].columns[2:]:
            print(f'{col}: {fit_data[m][col].mean().round(4)}, {fit_data[m][col].std().round(4)}')
        print('==========')
        print('')
    
    # =========================================== #
    # =========== SIMULATE WITH FITS ============ #
    # =========================================== #

    for group in fit_data[models[0]]['pain_group'].unique():

        #Group data
        group_data = {m: fit_data[m][fit_data[m]['pain_group'] == group] for m in models}

        #Metaparameters
        number_of_runs = len(group_data[models[0]])

        #Set up data columns
        accuracy_columns = ['context', 'trial_total', 'accuracy', 'run']
        pe_columns = ['context', 'trial_total', 'averaged_pe', 'run']
        values_columns = ['context', 'trial_total', 'values1', 'values2', 'run']
        transfer_columns = ['A', 'B', 'E', 'F', 'N']

        #Setup data tracking
        note = []
        accuracy = {m: [] for m in models}
        prediction_errors = {m: [] for m in models}
        values = {m: [] for m in models}
        choice_rates = {m: [] for m in models}

        #Run simulations
        loop = tqdm.tqdm(range(number_of_runs*len(models)))
        for n in range(number_of_runs):
            for m in models:
                loop.update(1)
                loop.set_description(f'Group {group.title()}, Participant {n+1}/{number_of_runs}')

                #Initialize task, model, and task design
                task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                    'transfer_phase': {'times_repeated': 4}}
                task = AvoidanceLearningTask(task_design)

                #Initialize model
                model = RLModel(m, group_data[m].iloc[n]).model

                #Initialize pipeline
                model = RLPipeline(model, task).simulate()

                #Extract model data
                params = ', '.join([f'{key}: {model.parameters[key]}'for key in model.parameters])
                note.append(f'{m} | {params}')
                task_learning_data = model.task_learning_data
                task_learning_data['trial_total'] = task_learning_data.groupby('state_id').cumcount()+1
                if 'v_prediction_errors' in task_learning_data.columns:
                    task_learning_data['prediction_errors'] = task_learning_data['v_prediction_errors'] + task_learning_data['q_prediction_errors']
                task_learning_data['averaged_pe'] = task_learning_data['prediction_errors'].apply(lambda x: sum(x)/len(x))
                learning_accuracy = task_learning_data.groupby(['context', 'trial_total'])['accuracy'].mean().reset_index()
                learning_prediction_errors = task_learning_data.groupby(['context', 'trial_total'])['averaged_pe'].mean().reset_index()

                value_labels = {'QLearning': 'q_values', 'ActorCritic': 'w_values', 'Relative': 'q_values', 'Hybrid2012': 'h_values', 'Hybrid2021': 'h_values', 'QRelative': 'm_values'}
                value_label = value_labels[m]

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
                if not type(choice_rates[m]) == pd.DataFrame:
                    accuracy[m] = pd.DataFrame(learning_accuracy, columns=accuracy_columns)
                    prediction_errors[m] = pd.DataFrame(learning_prediction_errors, columns=pe_columns)
                    values[m] = pd.DataFrame(learning_values, columns=values_columns)
                    choice_rates[m] = pd.DataFrame([model.choice_rate], columns=transfer_columns)
                else:
                    accuracy[m] = pd.concat([accuracy[m], pd.DataFrame(learning_accuracy)], ignore_index=True)
                    prediction_errors[m] = pd.concat([prediction_errors[m], pd.DataFrame(learning_prediction_errors)], ignore_index=True)
                    values[m] = pd.concat([values[m], pd.DataFrame(learning_values)], ignore_index=True)
                    choice_rates[m] = pd.concat([choice_rates[m], pd.DataFrame([model.choice_rate])], ignore_index=True)

        #Debug print
        print('')

        #Plot simulations    
        plot_simulations(accuracy, prediction_errors, values, choice_rates, models, group, number_of_runs)

    print('debug')