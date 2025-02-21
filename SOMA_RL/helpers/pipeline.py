import os
import time
import pandas as pd
import numpy as np
import tqdm

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

    def __init__(self, model, dataloader=None, task=None):

        #Get parameters
        self.dataloader = dataloader
        if dataloader is None:
            self.task_design = task.task_design
        else:
            num_learning_trials, num_transfer_trials = self.dataloader.get_num_trials()
            self.task_design = {'learning_phase': {'number_of_trials': num_learning_trials, 'number_of_blocks': 1}, 
                           'transfer_phase': {'number_of_trials': num_transfer_trials}}
            task.task_design = self.task_design
        self.task = task
        self.task.initiate_model(model.get_model())

    def simulate(self, data):

        #Run simulation and computations
        if data is None:
            self.task.run_experiment()
        else:
            self.task.rl_model.simulate(data)
        self.task.rl_model.run_computations()

        return self.task.rl_model

    def fit(self, data, bounds):

        return self.task.rl_model.fit(data, bounds)
        
    def run_fit(self, args):

        #Extract args
        columns, participant_id, run = args
        data = self.dataloader.get_data_dict()
        model_name = self.task.rl_model.__class__.__name__
        pain_group = data['learning']['pain_group'].values[0]

        #Extract participant data
        learning_states = data['learning']['state'].values
        learning_actions = data['learning']['action'].values
        learning_rewards = data['learning'][['reward_L', 'reward_R']].values

        transfer_states = data['transfer']['state'].values
        transfer_actions = data['transfer']['action'].values

        data = [(learning_states, learning_actions, learning_rewards), (transfer_states, transfer_actions)]
        
        #Fit models
        fit_results, fitted_params = self.fit(data, bounds=self.task.rl_model.bounds)

        #Store fit results
        participant_fitted = [participant_id, pain_group, run, float(fit_results.fun)]
        participant_fitted.extend([float(fitted_params[key]) for key in columns[4:] if key in fitted_params])

        #Save to csv file
        with open(f'SOMA_RL/fits/temp/{self.task.rl_model.model_name}_{participant_id}_Run{run}_fit_results.csv', 'a') as f:
            f.write(','.join([str(x) for x in participant_fitted]) + '\n')

    def run_simulations(self, args, generate_data=False):

        #Extract args
        columns, participant_id, group, run_number = args
        if self.dataloader is not None:
            data = self.dataloader.get_data_dict().copy()
        else:
            data = None
        model_name = self.task.rl_model.__class__.__name__

        #Run simulation and computations
        model = self.simulate(data)

        #Extract model data
        task_learning_data = model.task_learning_data
        task_transfer_data = model.task_transfer_data
        model_parameters = pd.DataFrame(model.parameters, index=[0])

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
        
        learning_accuracy['run'] = run_number
        learning_prediction_errors['run'] = run_number
        learning_values['run'] = run_number

        #Save task data
        if generate_data:
            unique_id = np.random.randint(0, 1000000)
            simulation_name = f"{model.model_name}_{unique_id}"
            os.makedirs(f'SOMA_RL/data/generated/{simulation_name}', exist_ok=True)
            model_parameters.to_csv(f'SOMA_RL/data/generated/{simulation_name}/{simulation_name}_generated_parameters.csv', header=True, index=False)
            task_learning_data.to_csv(f'SOMA_RL/data/generated/{simulation_name}/{simulation_name}_generated_learning.csv', header=True, index=False)
            task_transfer_data.to_csv(f'SOMA_RL/data/generated/{simulation_name}/{simulation_name}_generated_transfer.csv', header=True, index=False)
        else:
            #Store data
            accuracy = pd.DataFrame(learning_accuracy, columns=columns['accuracy'])
            prediction_errors = pd.DataFrame(learning_prediction_errors, columns=columns['pe'])
            values = pd.DataFrame(learning_values, columns=columns['values'])
            choice_rates = pd.DataFrame([model.choice_rate], columns=columns['choice_rate'])

            #Save to csv file
            accuracy.to_csv(f'SOMA_RL/fits/temp/{model.model_name}_{group}_{participant_id}_accuracy_sim_results.csv', index=False)
            prediction_errors.to_csv(f'SOMA_RL/fits/temp/{model.model_name}_{group}_{participant_id}_pe_sim_results.csv', index=False)
            values.to_csv(f'SOMA_RL/fits/temp/{model.model_name}_{group}_{participant_id}_values_sim_results.csv', index=False)
            choice_rates.to_csv(f'SOMA_RL/fits/temp/{model.model_name}_{group}_{participant_id}_choice_sim_results.csv', index=False)        
# Functions
def mp_run_fit(args):
    pipeline = args[0]
    pipeline.run_fit(args[1:])

def mp_run_simulations(args):
    pipeline = args[0]
    if 'generate_data' in str(args[-1]):
        pipeline.run_simulations(args[1:-1], generate_data=args[-1].split('=')[-1])
    else:
        pipeline.run_simulations(args[1:])

def mp_progress(num_files, filepath='SOMA_RL/fits/temp', divide_by=1, multiply_by=1, progress_bar=True):
    n_files = 0
    last_count = 0
    start_file_count = len(os.listdir(filepath))
    if progress_bar:
        loop = tqdm.tqdm(range(int((num_files-start_file_count)/divide_by))) if progress_bar else None
    while n_files*multiply_by < (num_files-start_file_count):
        if progress_bar:
            n_files = (np.floor(len(os.listdir(filepath))/divide_by)*multiply_by)-start_file_count
            if n_files > last_count:
                loop.update(int(n_files-last_count))
                last_count = n_files
        time.sleep(1)
    if progress_bar:
        loop.update(int((num_files-start_file_count)-last_count))
        
'''

def mp_progress(num_files, divide_by=1, multiply_by=1, filepath='SOMA_RL/fits/temp'):
    last_count = 0
    loop = tqdm.tqdm(range(int(num_files/divide_by)))
    while len(os.listdir(filepath))*multiply_by < num_files:
        n_files = np.floor(len(os.listdir(filepath))/divide_by)*multiply_by
        if n_files > last_count:
            loop.update(int(n_files-last_count))
            last_count = n_files
        time.sleep(1)
    loop.update(int((num_files/divide_by)-last_count))
'''