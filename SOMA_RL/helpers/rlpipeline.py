from typing import Optional
import os
import pandas as pd
import numpy as np
from .dataloader import DataLoader
from .tasks import AvoidanceLearningTask

class RLPipeline:
        
    """
    Reinforcement Learning Pipeline
    """

    def __init__(self, model: object,
                 dataloader: DataLoader = None,
                 task: Optional[AvoidanceLearningTask] = None,
                 training: str = 'scipy',
                 training_epochs: int = 1000,
                 optimizer_lr: float = 0.01,
                 multiprocessing: bool = False):

        """
        Initializes the Reinforcement Learning Pipeline with a model, dataloader, and task.

        Parameters
        ----------
        model : Model object
            The reinforcement learning model to be used in the pipeline.
        dataloader : DataLoader object, optional
            The dataloader to provide data for the model. If None, the task design will be used.
        task : Task object, optional
            The task object containing the task design and model. If None, it will be created from the dataloader.
        training : str, optional
            The training backend to use, either 'scipy' or 'torch'. Default is 'scipy'.
        training_epochs : int, optional
            The number of training epochs for the model. Default is 1000.
        optimizer_lr : float, optional
            The learning rate for the optimizer. Default is 0.01.
        multiprocessing : bool, optional
            Whether to use multiprocessing for fitting and simulations. Default is False.

        Returns
        -------
        None
        """

        #Set parameters
        self.training = training
    
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
        self.task.rl_model.training = training
        self.task.rl_model.training_epochs = training_epochs
        self.task.rl_model.optimizer_lr = optimizer_lr
        self.task.rl_model.multiprocessing = multiprocessing

    def simulate(self, data: pd.DataFrame = None) -> object:

        """
        Runs the simulation of the reinforcement learning model.
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to be used for the simulation. If None, the task design will be used.
        
        Returns
        -------
        object
            The reinforcement learning model after running the simulation.
        """

        #Run simulation and computations
        if data is None:
            self.task.run_experiment()
        else:
            self.task.rl_model.simulate(data)
        self.task.rl_model.run_computations()

        return self.task.rl_model

    def fit(self, data: pd.DataFrame, bounds: dict = None) -> tuple:

        """
        Fits the reinforcement learning model to the provided data.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit the model to, structured as a list of tuples containing states, actions, and rewards.
        bounds : dict, optional
            The bounds for the model parameters. If None, the model's default bounds will be used.

        Returns
        -------
        tuple
            A tuple containing the fit results and the fitted parameters.
        """

        if self.training == 'torch':
            fit_results, fitted_params = self.task.rl_model.fit_torch(data, bounds)
        else:
            fit_results, fitted_params = self.task.rl_model.fit(data, bounds)

        return fit_results, fitted_params
        
    def run_rl_fit(self, args: list) -> None:

        """
        Run the reinforcement learning fit for a participant.

        Parameters
        ----------
        args : list
            A list where the first element is a list of column names, followed by participant ID and run number.

        Returns
        -------
        None
        """

        #Extract args
        columns, participant_id, run = args
        self.task.rl_model.participant_id = participant_id
        self.task.rl_model.run = run
        data = self.dataloader.get_data_dict()
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
        fit_results = fit_results if self.training == 'torch' else fit_results.fun
        participant_fitted = [participant_id, pain_group, run, float(fit_results)]
        participant_fitted.extend([float(fitted_params[key]) for key in columns[4:] if key in fitted_params])

        #Save to csv file
        with open(f'SOMA_RL/fits/temp/{self.task.rl_model.model_name}_{participant_id}_Run{run}_fit_results.csv', 'a') as f:
            f.write(','.join([str(x) for x in participant_fitted]) + '\n')

    def run_simulations(self, args: list, generate_data: bool = False) -> None:

        """
        Run the reinforcement learning simulations for a participant.

        Parameters
        ----------
        args : list
            A list where the first element is a list of column names, followed by participant ID, group, and run number.
        generate_data : bool, optional
            Whether to generate new data or use existing data. Default is False.

        Returns (External)
        ------------------
        generated_parameters : csv
            A CSV file containing the generated parameters of the model.
        generated_learning : csv
            A CSV file containing the learning data from the simulation.
        generated_transfer : csv
            A CSV file containing the transfer data from the simulation.
        OR
        sim_results : csv
            CSV files containing the simulation results, including accuracy, prediction errors, values, and choice rates.
        """

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
        learning_accuracy = task_learning_data.groupby(['context', 'trial_total', 'state_id'])['accuracy'].mean().reset_index()
        learning_prediction_errors = task_learning_data.groupby(['context', 'trial_total', 'state_id'])['averaged_pe'].mean().reset_index()

        value_labels = {'QLearning': 'q_values', 
                        'ActorCritic': 'w_values', 
                        'Relative': 'q_values',
                        'wRelative': 'q_values',
                        'Hybrid2012': 'h_values',
                        'Hybrid2021': 'h_values',
                        'StandardHybrid2012': 'h_values',
                        'StandardHybrid2021': 'h_values'}
        
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
            unique_id = np.random.randint(0, 1000000) if participant_id == None else participant_id
            while f"{model.model_name}_{unique_id}" in os.listdir(f'SOMA_RL/data/generated/'):
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