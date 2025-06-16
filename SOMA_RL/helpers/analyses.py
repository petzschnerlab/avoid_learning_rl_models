from typing import Union
import sys
sys.dont_write_bytecode = True
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import shutil
import multiprocessing as mp
import copy
import pandas as pd
import numpy as np
import tqdm
import pickle

from models.rl_models import RLModel
from helpers.dataloader import DataLoader
from helpers.tasks import AvoidanceLearningTask
from helpers.rlpipeline import RLPipeline
from helpers.statistics import Statistics
from helpers.plotting import Plotting
from helpers.multiprocessing import mp_run_fit, mp_run_simulations, mp_progress

class Analyses(Plotting):

    """
    Class to hold all functions for the SOMA project analyses
    """

    def __init__(self):
        super().__init__()

    def run_fit(self) -> None:

        """
        Run the empirical fit analysis.

        Parameters
        ----------
        self.learning_filename : str
            Path to the learning dataset.
        self.transfer_filename : str
            Path to the transfer dataset.
        self.models : list
            List of model names to fit.
        self.number_of_participants : int
            Number of participants to include in the fit.
        self.fixed : dict, optional
            Dictionary of fixed parameters for the models.
        self.bounds : dict, optional
            Dictionary of bounds for the model parameters.
        self.random_params : bool, optional
            Whether to use random parameters for the models.
        self.number_of_runs : int, optional
            Number of runs for each model fit.
        self.generated : bool, optional
            Whether to use generated data for the fit.
        self.multiprocessing : bool, optional
            Whether to use multiprocessing for the fit.
        self.training : str, optional
            Training method to use ('torch' or 'scipy').
        self.training_epochs : int, optional
            Number of epochs for training the models.
        self.optimizer_lr : float, optional
            Learning rate for the optimizer.


        Returns
        -------

        """

        if self.fixed is not None:
            print('\nParameter Overwrites:')
            for model in self.fixed:
                print(f'       {model}:')
                for key in self.fixed[model]:
                    print(f'              {key} = {self.fixed[model][key]}')
        if self.bounds is not None:
            print('\nParameter Bound Overwrites:')
            for model in self.bounds:
                print(f'       {model}:')
                for key in self.bounds[model]:
                    print(f'              {key} = {self.bounds[model][key]}')
        print('--------------------------------------------------------')
        
        dataloader = self.run_model_fit(self.learning_filename, 
                            self.transfer_filename, 
                            self.models, 
                            number_of_participants=self.number_of_participants, 
                            fixed=self.fixed,
                            bounds=self.bounds,
                            random_params=self.random_params, 
                            number_of_runs=self.number_of_runs, 
                            generated=self.generated, 
                            multiprocessing=self.multiprocessing,
                            training=self.training,
                            training_epochs=self.training_epochs,
                            optimizer_lr=self.optimizer_lr)
        
        fit_data = self.run_fit_comparison(dataloader, 
                                    self.models, 
                                    dataloader.get_group_ids(),
                                    None)
        
        self.plot_fits_by_run_number(fit_data)
        self.describe_fits(fit_data)
        params_of_interest = {
            'wRelative+novel': 'weighting_factor',
            'Relative+novel': 'contextual_lr',
            'Hybrid2012+novel': 'mixing_factor'
        } 
        self.plot_model_parameter_correlations(fit_data, params_of_interest)
        
        self.run_fit_analyses(copy.deepcopy(fit_data))
        
        self.run_fit_simulations(self.learning_filename, 
                            self.transfer_filename, 
                            fit_data, 
                            self.models, 
                            dataloader.get_participant_ids(), 
                            dataloader.get_group_ids(), 
                            number_of_participants=self.number_of_participants, 
                            multiprocessing=self.multiprocessing)

    def run_validation(self) -> None:

        """
        Run the validation analysis.

        Parameters
        ----------
        self.recovery : str
            Recovery method to employ ('parameter' or 'model').
        self.models : list
            List of model names to validate.
        self.parameters : dict or str
            Dictionary of parameters for the models or 'random' to use random parameters.
        self.learning_filename : str
            Path to the learning dataset.
        self.transfer_filename : str
            Path to the transfer dataset.
        self.fit_filename : str, optional
            Path to the fit data file. If provided, it will override the parameters and bounds.
        self.fixed : dict, optional
            Dictionary of fixed parameters for the models.
        self.bounds : dict, optional
            Dictionary of bounds for the model parameters.
        self.number_of_participants : int, optional
            Number of participants to include in the validation.
        self.number_of_runs : int, optional
            Number of runs for each model validation.
        self.generate_data : bool, optional
            Whether to generate simulated data for the validation.
        self.task_design : str, optional
            Task design to use for the validation.
        self.multiprocessing : bool, optional
            Whether to use multiprocessing for the validation.
        self.clear_data : bool, optional
            Whether to clear existing data before running the validation.
        self.training : str, optional
            Training method to use ('torch' or 'scipy').
        self.training_epochs : int, optional
            Number of epochs for training the models.
        self.optimizer_lr : float, optional
            Learning rate for the optimizer.

        Returns
        -------
        None
        
        """

        if self.fit_filename is not None:

            if isinstance(self.parameters, dict):
                raise ValueError('Parameters and fit_filename will both provide parameters to the model. Please provide only one of them.')
            
            if self.bounds is not None:
                warnings.warn('\nBounds are being overridden by fit data.', stacklevel=2)
                self.bounds = None

            with open(self.fit_filename, 'rb') as f:
                fit_data = pickle.load(f)
            
            if set(fit_data.keys()) != set(self.models):
                warnings.warn(f'\nModels in fit data ({set(fit_data.keys())}) do not match models provided ({set(self.models)}). Overriding models parameter to match fit data.', stacklevel=2)
                self.models = list(fit_data.keys())
            
        if self.fixed is not None:
            print('\nParameter Overwrites:')
            for model in self.fixed:
                print(f'       {model}:')
                for key in self.fixed[model]:
                    print(f'              {key} = {self.fixed[model][key]}')
        if self.bounds is not None:
            print('\nParameter Bound Overwrites:')
            for model in self.bounds:
                print(f'       {model}:')
                for key in self.bounds[model]:
                    print(f'              {key} = {self.bounds[model][key]}')
        print('--------------------------------------------------------')

        if not os.path.exists('SOMA_RL/data/generated'):
            os.makedirs('SOMA_RL/data/generated')

        if [self.learning_filename, self.transfer_filename].count(None) == 0:
            datasets_to_generate = len(DataLoader(self.learning_filename, self.transfer_filename, number_of_participants=self.number_of_participants, reduced=False).get_participant_ids())

        if self.generate_data:
            self.generate_simulated_data(models=self.models, 
                                    parameters=self.parameters, 
                                    random_params=self.random_params,
                                    learning_filename=self.learning_filename, 
                                    transfer_filename=self.transfer_filename, 
                                    fit_filename=self.fit_filename,
                                    task_design=self.task_design, 
                                    fixed=self.fixed, 
                                    bounds=self.bounds, 
                                    datasets_to_generate=datasets_to_generate, 
                                    number_of_participants=self.number_of_participants, 
                                    multiprocessing=self.multiprocessing,
                                    clear_data=self.clear_data)
        
        dataloader = self.run_generative_fits(models=self.models, 
                                        number_of_runs=self.number_of_runs, 
                                        datasets_to_generate=datasets_to_generate, 
                                        fixed=self.fixed, 
                                        bounds=self.bounds, 
                                        multiprocessing=self.multiprocessing, 
                                        recovery=self.recovery,
                                        training=self.training,
                                        training_epochs=self.training_epochs,
                                        optimizer_lr=self.optimizer_lr)
        
        fit_data = self.run_fit_comparison(dataloader=dataloader, 
                                    models=self.models, 
                                    group_ids=['simulated'], 
                                    recovery=self.recovery)
        
        if self.recovery == 'model':
            confusion_matrix = self.create_confusion_matrix(dataloader=dataloader, 
                                                    fit_data=fit_data)
            
            self.plot_model_fits(confusion_matrix=confusion_matrix)
        else:
            self.plot_parameter_fits(models=self.models, 
                                fit_data=fit_data, 
                                fixed=self.fixed, 
                                bounds=self.bounds)

    def run_fit_analyses(self, fit_data: dict, transform: bool = True) -> None:

        """
        Run statistics on the fit data.
        
        Parameters
        ----------
        fit_data : dict
            Dictionary containing the fit data for each model.
        transform : bool, optional
            Whether to log-transform the parameters before running the analyses. Default is True.

        Returns (External)
        ------------------
        linear_results : csv
            CSV file containing the results of the glmm models.
        ttest_results : csv
            CSV file containing the results of the planned t-tests.
        posthoc_results : csv
            CSV file containing the results of the post-hoc tests.
        model-fits : png & svg
            Plots of the model fits for each parameter per model.
        """
        
        linear_results = None
        ttest_results = None
        posthoc_results = None
        for model in fit_data:
            model_data = fit_data[model]
            fit_parameters = model_data.columns[4:]
            for parameter in fit_parameters:
                statistics = Statistics()
                
                #turn pain_group in model_data into a category
                model_data['pain_group'] = pd.Categorical(model_data['pain_group'], categories=['no pain', 'acute pain', 'chronic pain'])

                #Log-Transform when needed
                if transform:
                    if parameter not in ['novel_factor', 'mixing_factor', 'valence_factor', 'weighting_factor']: # Exclude parameters that are not to be log-transformed
                        if model_data[parameter].min() <= 0: 
                            model_data[parameter] = model_data[parameter] - model_data[parameter].min() + 1  # Shift the parameter to be positive if it has non-positive values
                        model_data[parameter] = np.log(model_data[parameter])  # Log-transform the parameter to reduce skewness

                #Run linear model
                linear_model = statistics.linear_model_categorical(f'{parameter} ~ pain_group', model_data)
                linear_result = linear_model['model_summary']
                linear_result.insert(0, 'parameter', parameter)
                linear_result.insert(0, 'model', model)
                if linear_results is None:
                    linear_results = linear_result
                else:
                    linear_results = pd.concat((linear_results, linear_result), ignore_index=True)

                #Run planned ttest
                comparisons = [['no pain', 'acute pain'], ['no pain', 'chronic pain']]
                model_data = model_data.rename(columns={'participant': 'participant_id'})
                ttest_model = statistics.planned_ttests(parameter, 'pain_group', comparisons, model_data)
                posthoc_model = statistics.post_hoc_tests(parameter, 'pain_group', model_data)

                ttest_result = ttest_model['model_summary']
                ttest_result.insert(0, 'parameter', parameter)
                ttest_result.insert(0, 'model', model)
                if ttest_results is None:
                    ttest_results = ttest_result
                else:
                    ttest_results = pd.concat((ttest_results, ttest_result), ignore_index=True)

                posthoc_result = posthoc_model.reset_index()
                posthoc_result.insert(0, 'parameter', parameter)
                posthoc_result.insert(0, 'model', model)
                if posthoc_results is None:
                    posthoc_results = posthoc_result
                else:
                    posthoc_results = pd.concat((posthoc_results, posthoc_result), ignore_index=True)

        linear_results.to_csv('SOMA_RL/stats/pain_fits_linear_results.csv', index=False)
        ttest_results.to_csv('SOMA_RL/stats/pain_fits_ttest_results.csv', index=False)
        posthoc_results.to_csv('SOMA_RL/stats/pain_fits_posthoc_results.csv', index=False)

        for model in fit_data:
            #Plot all data
            self.plot_parameter_data(f'{model}-model-fits', copy.copy(fit_data[model]))
            self.plot_parameter_data(f'{model}-model-fits', copy.copy(fit_data[model]), plot_type='bar')

            #Plot data separately for significant and non-significant parameters
            model_results = linear_results[linear_results['model'] == model]
            significant_params = model_results['p_value'].values < .05
            nonsignificant_params = model_results['p_value'].values >= .05

            significant_data = copy.copy(fit_data[model])
            significant_data = significant_data.drop(columns=model_results['parameter'].values[nonsignificant_params])

            nonsignificant_data = copy.copy(fit_data[model])
            nonsignificant_data = nonsignificant_data.drop(columns=model_results['parameter'].values[significant_params])
            
            if significant_params.any():
                self.plot_parameter_data(f'{model}-model-fits-significant', copy.copy(significant_data))
                self.plot_parameter_data(f'{model}-model-fits-significant', copy.copy(significant_data), plot_type='bar')
            if nonsignificant_params.any():
                self.plot_parameter_data(f'{model}-model-fits-nonsignificant', copy.copy(nonsignificant_data))
                self.plot_parameter_data(f'{model}-model-fits-nonsignificant', copy.copy(nonsignificant_data), plot_type='bar')

    def create_confusion_matrix(self, dataloader: DataLoader, fit_data: dict) -> pd.DataFrame:
        
        """
        Create a confusion matrix for the model fits.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object containing the data.
        fit_data : dict
            Dictionary containing the fit data for each model.

        Returns
        -------
        confusion_matrix : pd.DataFrame
            DataFrame containing the confusion matrix with BIC values.
        """

        confusion_matrix = pd.DataFrame(index=fit_data.keys(), columns=fit_data.keys(), dtype=float) #Rows = model fit, Columns = generated model

        data = dataloader.get_data()
        number_trials_learning = data[0].groupby(['participant', 'pain_group']).size().reset_index(name='counts')
        number_trials_transfer = data[1].groupby(['participant', 'pain_group']).size().reset_index(name='counts')
        number_trials = number_trials_learning['counts'].values[0] + number_trials_transfer['counts'].values[0]
        
        for model in fit_data:
            number_params = RLModel(model).get_n_parameters()
            model_fit = fit_data[model].copy()
            for gen_model in fit_data:
                fit = model_fit[model_fit['model'] == gen_model]['fit']
                BIC = np.mean(np.log(number_trials)*number_params + 2*fit)
                confusion_matrix.loc[model, gen_model] = BIC
                
        return confusion_matrix
                
    def run_model_fit(self,
                learning_filename: str,
                transfer_filename: str,
                models: list,
                fixed: dict = None,
                bounds: dict = None,
                number_of_participants: int = 0,
                random_params: bool = False,
                number_of_runs: int = 1,
                number_of_files: int = None,
                generated: bool = False,
                clear_data: bool = True,
                progress_bar: bool = True,
                multiprocessing: bool = False,
                training: str = 'torch',
                training_epochs: int = 1000,
                optimizer_lr: float = 0.01) -> DataLoader:
        
        """
        Run the model fit analysis.

        Parameters
        ----------
        learning_filename : str
            Path to the learning dataset.
        transfer_filename : str
            Path to the transfer dataset.
        models : list
            List of model names to fit.
        fixed : dict, optional
            Dictionary of fixed parameters for the models.
        bounds : dict, optional
            Dictionary of bounds for the model parameters.
        number_of_participants : int, optional
            Number of participants to include in the fit. Default is 0, which means all participants.
        random_params : bool, optional
            Whether to use random parameters for the models. Default is False.
        number_of_runs : int, optional
            Number of runs for each model fit. Default is 1.
        number_of_files : int, optional
            Number of files to process. If None, it will process all files. Default is None.
        generated : bool, optional
            Whether to use generated data for the fit. Default is False.
        clear_data : bool, optional
            Whether to clear existing data before running the fit. Default is True.
        progress_bar : bool, optional
            Whether to show a progress bar during the fit. Default is True.
        multiprocessing : bool, optional
            Whether to use multiprocessing for the fit. Default is False.
        training : str, optional
            Training method to use ('torch' or 'scipy'). Default is 'torch'.
        training_epochs : int, optional
            Number of epochs for training the models. Default is 1000.
        optimizer_lr : float, optional
            Learning rate for the optimizer. Default is 0.01.

        Returns
        -------
        dataloader : DataLoader
            DataLoader object containing the data for the fit.
        """
        
        #Delete any existing files
        if clear_data:
            for f in os.listdir('SOMA_RL/fits/temp'):
                os.remove(os.path.join('SOMA_RL','fits','temp',f))

        #Run fits
        inputs = []
        columns = RLModel().get_model_columns()
        data_names = models if 'XX' in learning_filename else ['']
        for data_index, data_model_name in enumerate(data_names):  
            current_learning_filename = learning_filename.replace('XX', data_model_name)
            current_transfer_filename = transfer_filename.replace('XX', data_model_name)
            dataloader = DataLoader(current_learning_filename, current_transfer_filename, number_of_participants, generated=generated)
            participant_ids = dataloader.get_participant_ids()
            if progress_bar and data_index == 0:
                print('\n')
                model_runs = len(models)**2 if 'XX' in learning_filename else len(models)
                print(f"Number of participants: {len(participant_ids)}, Number of models: {model_runs}, Number of runs: {number_of_runs}, Total fits: {len(participant_ids)*model_runs*number_of_runs}")
                loop = tqdm.tqdm(range(len(participant_ids)*model_runs*number_of_runs)) if not multiprocessing else None
            for participant in participant_ids:
                p_dataloader = copy.copy(dataloader)
                p_dataloader.filter_participant_data(participant)  
                for model_name in models:
                    for run in range(number_of_runs):
                        model = RLModel(model_name, random_params=random_params, fixed=fixed, bounds=bounds)
                        task = AvoidanceLearningTask()
                        pipeline = RLPipeline(model, p_dataloader, task, training, training_epochs, optimizer_lr, multiprocessing)
                        if multiprocessing:
                            inputs.append((pipeline, columns[model_name.split('+')[0]], participant, run))
                        else:
                            pipeline.run_rl_fit((columns[model_name.split('+')[0]], participant, run))
                            if progress_bar:
                                loop.update(1)

        #Run all models fits in parallel
        if multiprocessing:
            if progress_bar:
                print('\nMultiprocessing initiated...')

            pool = mp.Pool(mp.cpu_count())
            pool.map_async(mp_run_fit, inputs)
            pool.close()

            num_ints = number_of_files if number_of_files is not None else len(inputs)
            #Progress bar checking how many have completed
            mp_progress(num_ints, progress_bar=progress_bar)
            if progress_bar:
                print('\nMultiprocessing complete!')

        #Combine all files that are the same except for the run number
        files = os.listdir('SOMA_RL/fits/temp')
        files_prefix = [f.split('_Run')[0] for f in files if '_Run' in f]
        files_prefix = list(set(files_prefix))
        for file_prefix in files_prefix:
            files_to_combine = [os.path.join('SOMA_RL','fits','temp',f'{file_prefix}_Run{i}_fit_results.csv') for i in range(number_of_runs)]
            combined_file = pd.concat([pd.read_csv(f, header=None) for f in files_to_combine], ignore_index=True)
            combined_file.to_csv(os.path.join('SOMA_RL','fits','temp',f'{file_prefix}_fit_results.csv'), index=False, header=False)
            for f in files_to_combine:
                os.remove(f)

        return dataloader

    def run_fit_comparison(self, dataloader: DataLoader, 
                           models: list, 
                           group_ids: list, 
                           recovery: str = 'parameter') -> dict:

        """
        Run the fit comparison analysis.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object containing the data for the fit.
        models : list
            List of model names to compare.
        group_ids : list
            List of group IDs to include in the fit comparison.
        recovery : str, optional
            Recovery method to employ ('parameter' or 'model'). Default is 'parameter'.

        Returns
        -------
        fit_data : dict
            Dictionary containing the fit data for each model, with keys as model names and values as DataFrames.
        """
        
        columns = RLModel().get_model_columns()

        if recovery == 'model':
            for m in models:
                columns[m.split('+')[0]].insert(2, 'model')

        #Load all data in the fit data
        files = os.listdir('SOMA_RL/fits/temp')
        files = [f for f in files if 'ERROR' not in f]
        fit_data = {model: pd.DataFrame(columns=columns[model.split('+')[0]]) for model in models}
        full_fit_data = {model: pd.DataFrame(columns=columns[model.split('+')[0]]) for model in models}
        for f in files:
            model_name = f.split('_')[0]
            if model_name in models:
                participant_data = pd.read_csv(os.path.join('SOMA_RL','fits','temp',f), header=None)
                data_model = participant_data.loc[0,0].split('_')[0][1:]
                model_columns = RLModel(model_name).get_model_columns(model_name.split('+')[0])
                if recovery == 'model':
                    participant_data.insert(2, 'model', data_model)
                    model_columns.insert(2, 'model')
                participant_data.columns = model_columns
                best_participant_data = participant_data.iloc[[participant_data['fit'].idxmin()]]
                nan_keys = [key for key in best_participant_data if best_participant_data[key].isnull().any()]
                full_nan_keys = [key for key in participant_data if participant_data[key].isnull().any()]
                for key in nan_keys:
                    best_participant_data[key] = None
                for key in full_nan_keys:
                    participant_data[key] = None

                if len(fit_data[model_name]) == 0:
                    fit_data[model_name] = best_participant_data
                    full_fit_data[model_name] = participant_data
                else:
                    fit_data[model_name] = pd.concat((fit_data[model_name], best_participant_data), ignore_index=True)
                    full_fit_data[model_name] = pd.concat((full_fit_data[model_name], participant_data), ignore_index=True)

        #Plots
        self.plot_fit_distributions(fit_data)

        #Determine participants with outlier fits
        self.determine_parameter_outliers(fit_data, dataloader)

        #Run analyses on the fit data
        self.compute_criterions(dataloader, fit_data, models, group_ids, recovery=recovery)

        #Save fit data as a pickle file
        if recovery is None: 
            with open(f'SOMA_RL/fits/full_fit_data.pkl', 'wb') as f:
                pickle.dump(full_fit_data, f)

            with open(f'SOMA_RL/fits/fit_data.pkl', 'wb') as f:
                pickle.dump(fit_data, f)  
        else: 
            with open(f'SOMA_RL/fits/full_fit_data_{recovery.upper()}.pkl', 'wb') as f:
                pickle.dump(full_fit_data, f)

            with open(f'SOMA_RL/fits/fit_data_{recovery.upper()}.pkl', 'wb') as f:
                pickle.dump(fit_data, f)

        #Delete all files
        for f in files:
            os.remove(os.path.join('SOMA_RL','fits','temp',f))

        return fit_data

    def compute_criterions(self, dataloader: DataLoader,
                           fit_data: dict, 
                           models: list, 
                           group_ids: list, 
                           recovery: str = 'parameter') -> None:

        """
        Compute AIC and BIC for the fit data and print a report.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object containing the data for the fit.
        fit_data : dict
            Dictionary containing the fit data for each model, with keys as model names and values as DataFrames.
        models : list
            List of model names to compare.
        group_ids : list
            List of group IDs to include in the fit comparison.
        recovery : str, optional
            Recovery method to employ ('parameter' or 'model'). Default is 'parameter'.

        Returns (External)
        ------------------
        group_AIC: csv
            CSV file containing the AIC values for each group and model.
        group_BIC: csv
            CSV file containing the BIC values for each group and model.
        group_AIC_percentages: csv
            CSV file containing the percentages of best fits based on AIC for each group and model.
        group_BIC_percentages: csv
            CSV file containing the percentages of best fits based on BIC for each group and model.
        AIC_model_comparison: png & svg
            Plots of the AIC values for each model and group.
        BIC_model_comparison: png & svg
            Plots of the BIC values for each model and group.
        
        """

        # =========================================== #
        # =============== REPORT FIT ================ #
        # =========================================== #

        # Get number of samples per participant
        data = dataloader.get_data()
        number_trials_learning = data[0].groupby(['participant', 'pain_group']).size().reset_index(name='counts')
        number_trials_transfer = data[1].groupby(['participant', 'pain_group']).size().reset_index(name='counts')
        number_trials = number_trials_learning['counts'].values[0] + number_trials_transfer['counts'].values[0]

        group_AIC = {m: {} for m in models}
        group_BIC = {m: {} for m in models}
        all_AIC = {m: {} for m in models}
        all_BIC = {m: {} for m in models}
        for model_name in models:

            #Compute AIC and BIC
            for group in group_ids:
                group_fit = fit_data[model_name][fit_data[model_name]['pain_group'] == group]
                participant_NLL = group_fit["fit"]
                number_params = len(group_fit.columns) - 4 # TODO: Check if this is always correct
                participant_AIC = 2*number_params + 2*participant_NLL
                participant_BIC = np.log(number_trials)*number_params + 2*participant_NLL
                group_AIC[model_name][group] = np.mean(participant_AIC)
                group_BIC[model_name][group] = np.mean(participant_BIC)
                all_AIC[model_name][group] = participant_AIC.reset_index(drop=True)
                all_BIC[model_name][group] = participant_BIC.reset_index(drop=True)

            number_params = len(fit_data[model_name].columns) - 4
            participant_NLL = fit_data[model_name]["fit"]
            participant_AIC = 2*number_params + 2*participant_NLL
            participant_BIC = np.log(number_trials)*number_params + 2*participant_NLL
            AIC = np.mean(participant_AIC)
            BIC = np.mean(participant_BIC)
            group_AIC[model_name]['full'] = AIC
            group_BIC[model_name]['full'] = BIC

            print('')
            print(f'FIT REPORT: {model_name}')
            print('==========')
            print(f'AIC: {AIC.round(0)}')
            print(f'BIC: {BIC.round(0)}')
            col_index = 4 if recovery == 'model' else 3
            for col in fit_data[model_name].columns[col_index:]:
                if fit_data[model_name][col].values[0] is not None:
                    print(f'{col}: {fit_data[model_name][col].mean().round(4)}, {fit_data[model_name][col].std().round(4)}')
            print('==========')
            print('')
            
        #Turn nested dictionary into dataframe
        group_AIC = pd.DataFrame(group_AIC)
        group_AIC['best_model'] = group_AIC.idxmin(axis=1)
        group_AIC.to_csv('SOMA_RL/fits/group_AIC.csv')

        group_BIC = pd.DataFrame(group_BIC)
        group_BIC['best_model'] = group_BIC.idxmin(axis=1)
        group_BIC.to_csv('SOMA_RL/fits/group_BIC.csv')

        print('AIC REPORT')
        print('==========')
        print(group_AIC)
        print('==========')

        print('')
        print('BIC REPORT')
        print('==========')
        print(group_BIC)
        print('==========')

        #Create a dataframe for each key within all_AIC['QLearning+novel']
        model_data_AIC = {m: {} for m in models}
        model_data_BIC = {m: {} for m in models}
        for model_name in all_AIC:
            model_data_AIC[model_name] = pd.DataFrame(pd.concat(all_AIC[model_name].values(), ignore_index=True))
            model_data_BIC[model_name] = pd.DataFrame(pd.concat(all_BIC[model_name].values(), ignore_index=True))
            
            group_names = []
            for group in all_AIC[model_name]:
                group_names.extend([group] * len(all_AIC[model_name][group]))

        best_fits_AIC = pd.concat(model_data_AIC.values(), ignore_index=True, axis=1)
        best_fits_AIC.columns = list(all_AIC.keys())
        best_fits_AIC['best_model'] = best_fits_AIC.idxmin(axis=1)
        best_fits_AIC.insert(0, 'group', group_names)

        best_fits_BIC = pd.concat(model_data_BIC.values(), ignore_index=True, axis=1)
        best_fits_BIC.columns = list(all_BIC.keys())
        best_fits_BIC['best_model'] = best_fits_BIC.idxmin(axis=1)
        best_fits_BIC.insert(0, 'group', group_names)

        #Create best fit percentages
        best_fits_AIC_summary = best_fits_AIC.groupby(['group', 'best_model']).size().unstack(fill_value=0)
        best_fits_AIC_summary.loc['full'] = best_fits_AIC_summary.sum(numeric_only=True)
        best_fits_AIC_summary = best_fits_AIC_summary.reindex(columns=list(all_AIC.keys()), fill_value=0)
        best_fits_AIC_summary = best_fits_AIC_summary.div(best_fits_AIC_summary.sum(axis=1), axis=0) * 100

        best_fits_BIC_summary = best_fits_BIC.groupby(['group', 'best_model']).size().unstack(fill_value=0)
        best_fits_BIC_summary.loc['full'] = best_fits_BIC_summary.sum(numeric_only=True)
        best_fits_BIC_summary = best_fits_BIC_summary.reindex(columns=list(all_BIC.keys()), fill_value=0)
        best_fits_BIC_summary = best_fits_BIC_summary.div(best_fits_BIC_summary.sum(axis=1), axis=0) * 100

        #Save as csv files
        best_fits_AIC_summary.to_csv('SOMA_RL/fits/group_AIC_percentages.csv')
        best_fits_BIC_summary.to_csv('SOMA_RL/fits/group_BIC_percentages.csv')

        #Plots
        self.plot_model_comparisons(group_AIC, 'AIC_model_comparisons')
        self.plot_model_comparisons(group_BIC, 'BIC_model_comparisons')

    def run_fit_simulations(self, 
                            learning_filename: str,
                            transfer_filename: str,
                            fit_data: dict,
                            models: list,
                            participant_ids: list,
                            group_ids: list,
                            number_of_participants: int = 0,
                            multiprocessing: bool = False) -> None:

        """
        Run simulations with the fitted models and save the results.

        Parameters
        ----------
        learning_filename : str
            Path to the learning dataset.
        transfer_filename : str
            Path to the transfer dataset.
        fit_data : dict
            Dictionary containing the fit data for each model, with keys as model names and values as DataFrames.
        models : list
            List of model names to simulate.
        participant_ids : list
            List of participant IDs to include in the simulations.
        group_ids : list
            List of group IDs to include in the simulations.
        number_of_participants : int, optional
            Number of participants to include in the simulations. Default is 0, which means all participants.
        multiprocessing : bool, optional
            Whether to use multiprocessing for the simulations. Default is False.

        Returns (External)
        ------------------
        modelsimulations_accuracy_data : csv
            CSV file containing the accuracy data from the simulations.
        modelsimulations_choice_data : csv
            CSV file containing the choice rate data from the simulations.
        model_simulations: png & svg
            Plots of the model simulations for accuracy and choice rates.
        """

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
                participant_params = fit_data[model_name][fit_data[model_name]['participant'] == participant].copy().reset_index()
                group = participant_params['pain_group'].values[0]
                
                #Initialize task, model, and task design
                p_dataloader = copy.copy(dataloader)
                p_dataloader.filter_participant_data(participant) 
                model = RLModel(model_name, participant_params)
                task = AvoidanceLearningTask()
                pipeline = RLPipeline(model, p_dataloader, task, training='None')
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
                participant_data['accuracy'] = participant_data['accuracy']*100
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

        #for accuracy, combine nested files (group, model_name)
        accuracy_data = None
        choice_data = None
        for group in accuracy:
            for model in accuracy[group]:

                group_model_accuracy = accuracy[group][model].copy()
                group_model_accuracy.insert(0, 'group', [group]*len(group_model_accuracy))
                group_model_accuracy.insert(0, 'model', [model]*len(group_model_accuracy))

                group_model_choice = choice_rates[group][model].copy()
                group_model_choice.insert(0, 'group', [group]*len(group_model_choice))
                group_model_choice.insert(0, 'model', [model]*len(group_model_choice))

                if accuracy_data is None:
                    accuracy_data = group_model_accuracy
                    choice_data = group_model_choice
                else:
                    accuracy_data = pd.concat((accuracy_data, group_model_accuracy), ignore_index=True)
                    choice_data = pd.concat((choice_data, group_model_choice), ignore_index=True)
        accuracy_model = {model: {group: accuracy[group][model] for group in accuracy} for model in models}
        choice_rates_model = {model: {group: choice_rates[group][model] for group in choice_rates} for model in models}
        accuracy_data.to_csv('SOMA_RL/fits/modelsimulation_accuracy_data.csv', index=False)
        choice_data.to_csv('SOMA_RL/fits/modelsimulation_choice_data.csv', index=False)

        #Delete all files
        for f in files:
            os.remove(os.path.join('SOMA_RL','fits','temp',f))

        #Average across states for plotting
        for model in accuracy_model:
            for group in accuracy_model[model]:
                accuracy_model[model][group] = accuracy_model[model][group].groupby(['context', 'trial_total', 'run']).mean().reset_index()
            
        #Plot simulations 
        for group in accuracy:
            self.plot_simulations(accuracy[group], prediction_errors[group], values[group], choice_rates[group], models, group, dataloader)
        self.plot_simulations_behaviours(accuracy_model, choice_rates_model, models, accuracy.keys(), dataloader, binned_trial=True, subplot_title=['A. Learning Phase', 'B. Transfer Phase'])
        self.plot_simulations_behaviours(accuracy_model, choice_rates_model, models, accuracy.keys(), dataloader, binned_trial=True, subplot_title=['A. Learning Phase', 'B. Transfer Phase'], plot_type = 'bar')

    def generate_simulated_data(self,
                                models: list,
                                parameters: dict = None,
                                random_params: Union[bool, str] = 'random',
                                learning_filename: str = None,
                                transfer_filename: str = None,
                                fit_filename: str = None,
                                task_design: dict = None,
                                fixed: dict = None,
                                bounds: dict = None,
                                datasets_to_generate: int = 1,
                                number_of_participants: int = 0,
                                multiprocessing: bool = False,
                                clear_data: bool = True) -> None:
        
        """
        Generate simulated data using the specified models and parameters.

        Parameters
        ----------
        models : list
            List of model names to generate data for.
        parameters : dict
            Dictionary of parameters for the models.
            If you want parameters from a fit file, use the fit_filename argument.
            If you want random (uniform or normal) parameters, use the random_params argument instead.
        random_params : bool or str, optional
            Whether to use random parameters for the models. If 'random' or 'normal', random parameters will be generated. Default is 'random'.
        learning_filename : str, optional
            Path to the learning dataset. If None, no learning data will be generated.
        transfer_filename : str, optional
            Path to the transfer dataset. If None, no transfer data will be generated.
        fit_filename : str, optional
            Path to the fit data file. If None, no fit data will be used.
        task_design : dict, optional
            Dictionary defining the task design for the simulations. If None, default task design will be used.
        fixed : dict, optional
            Dictionary of fixed parameters for the models. Default is None.
        bounds : dict, optional
            Dictionary of bounds for the model parameters. Default is None.
        datasets_to_generate : int, optional
            Number of datasets to generate for each model. Default is 1.
        number_of_participants : int, optional
            Number of participants to include in the generated data. Default is 0, which means all participants.
        multiprocessing : bool, optional
            Whether to use multiprocessing for generating data. Default is False.
        clear_data : bool, optional
            Whether to clear existing generated data before running the generation. Default is True.

        Returns (External)
        ------------------

        generated_learning : csv
            CSV file containing the generated learning data.
        generated_transfer : csv
            CSV file containing the generated transfer data.
        """

        #Checks
        if [learning_filename, transfer_filename].count(None) == 1:
            raise ValueError('Both learning and transfer filenames must be provided')
        
        if fit_filename is not None and parameters is not None:
            warnings.warn('fit_filename is set, but parameters is also set. Using fit_filename parameters instead of provided parameters.')
            parameters = None

        if fit_filename is not None and random_params:
            warnings.warn('fit_filename is set, but random_params is also set. Using fit_filename parameters instead of random parameters.')
            random_params = False

        if parameters is not None and random_params:
            warnings.warn('parameters is set, but random_params is also set. Using provided parameters instead of random parameters.')
            random_params = False
        
        if parameters is None and random_params is False and fit_filename is None:
            raise ValueError('Either parameters, random_params, or fit_filename must be provided.')

        #Delete all folders with generated data from previous run, if desired
        if clear_data:
            for f in os.listdir('SOMA_RL/data/generated'):
                if '.csv' in f:
                    os.remove(os.path.join('SOMA_RL','data','generated',f))
                else:
                    shutil.rmtree(os.path.join('SOMA_RL','data','generated',f))

        #Load fit data
        if fit_filename is not None:
            #Check to see if fit_filename is a file that exists
            if not os.path.exists(fit_filename):
                raise ValueError(f'Fit file {fit_filename} does not exist.')
            
            with open(fit_filename, 'rb') as f:
                fit_data = pickle.load(f)

        #Set up parameters
        if learning_filename is not None and transfer_filename is not None:
            dataloader = DataLoader(learning_filename, transfer_filename, reduced=False, generated=False, number_of_participants=number_of_participants)
            participant_ids = dataloader.get_participant_ids()
            datasets_to_generate = len(participant_ids)
        else:
            dataloader = None

        #Run simulations
        print(f'\nNumber of Models: {len(models)}, Number of Runs: {datasets_to_generate}, Total Simulations: {len(models)*datasets_to_generate}')
        loop = tqdm.tqdm(range(len(models)*datasets_to_generate)) if not multiprocessing else None
        inputs = []
        columns = RLModel().get_model_columns()
        for model_name in models:
            for run in range(datasets_to_generate):

                if dataloader is not None:
                    p_dataloader = copy.copy(dataloader)
                    p_dataloader.filter_participant_data(participant_ids[run])
                else:
                    p_dataloader = None

                if fit_filename is not None:
                    if dataloader is not None:
                        participant_id = fit_data[model_name]['participant'][fit_data[model_name]['participant']==participant_ids[run]].values[0]
                    else:
                        participant_id = fit_data[model_name]['participant'][run]
                    parameters = {model_name: fit_data[model_name][fit_data[model_name]['participant']==participant_id]}

                model_parameters = parameters[model_name] if parameters else None
                model = RLModel(model_name, model_parameters, random_params=random_params, fixed=fixed, bounds=bounds)
                task = AvoidanceLearningTask(task_design)
                pipeline = RLPipeline(model, dataloader=p_dataloader, task=task, training='scipy')
                if multiprocessing:
                    inputs.append((pipeline, columns, run, 'simulation', run, 'generate_data=True'))
                else:
                    pipeline.run_simulations((columns, run, 'simulation', run), generate_data=True)
                    loop.update(1)

        #Run all generations in parallel
        if multiprocessing:
            print('\nMultiprocessing initiated...')

            pool = mp.Pool(mp.cpu_count())
            pool.map_async(mp_run_simulations, inputs)
            pool.close()

            #Progress bar checking how many have completed
            mp_progress(len(inputs), filepath='SOMA_RL/data/generated')
            print('\nMultiprocessing complete!')

        #For each model, combine all files for each model
        files = os.listdir('SOMA_RL/data/generated')
        for model in models:
            #Find all files that start with model
            data_names = [filename for filename in files if model == filename.split('_')[0]]
            #Load and concatenate all files
            model_learning = None
            model_transfer = None
            for data_name in data_names:
                learning_data = pd.read_csv(f'SOMA_RL/data/generated/{data_name}/{data_name}_generated_learning.csv')
                transfer_data = pd.read_csv(f'SOMA_RL/data/generated/{data_name}/{data_name}_generated_transfer.csv')
                participant = f"[{data_name.split(']')[0].split('[')[-1]}]"
                learning_data.insert(0, 'participant_id', participant)
                transfer_data.insert(0, 'participant_id', participant)

                if model_learning is None:
                    model_learning = learning_data
                    model_transfer = transfer_data
                else:
                    model_learning = pd.concat((model_learning, learning_data), ignore_index=True)
                    model_transfer = pd.concat((model_transfer, transfer_data), ignore_index=True)
            #Save concatenated files
            model_learning.to_csv(f'SOMA_RL/data/generated/{model}_generated_learning.csv', index=False)
            model_transfer.to_csv(f'SOMA_RL/data/generated/{model}_generated_transfer.csv', index=False)

    def run_generative_fits(self,
                            models: Union[list, str],
                            number_of_runs: int = 1,
                            datasets_to_generate: int = 1,
                            fixed: dict = None,
                            bounds: dict = None,
                            multiprocessing: bool = False,
                            recovery: str = 'parameter',
                            training: str = 'torch',
                            training_epochs: int = 1000,
                            optimizer_lr: float = 0.01) -> DataLoader:
        
        """
        Run generative fits for the specified models.

        Parameters
        ----------
        models : list or str
            List of model names to fit or a single model name.
        number_of_runs : int, optional
            Number of runs to perform for each model. Default is 1.
        datasets_to_generate : int, optional
            Number of datasets to generate for each model. Default is 1.
        fixed : dict, optional
            Dictionary of fixed parameters for the models. Default is None.
        bounds : dict, optional
            Dictionary of bounds for the model parameters. Default is None.
        multiprocessing : bool, optional
            Whether to use multiprocessing for the fits. Default is False.
        recovery : str, optional
            Recovery method to employ ('parameter' or 'model'). Default is 'parameter'.
        training : str, optional
            Training method to use ('torch' or 'scipy'). Default is 'torch'.
        training_epochs : int, optional
            Number of epochs for training if using 'torch'. Default is 1000.
        optimizer_lr : float, optional
            Learning rate for the optimizer if using 'torch'. Default is 0.01.

        Returns
        -------
        dataloader : DataLoader
            DataLoader object containing the generated data for the fits.       

        """

        #Find all files in SOMA_RL/data/generated that are not folders
        for f in os.listdir('SOMA_RL/fits/temp'):
            os.remove(os.path.join('SOMA_RL','fits','temp',f))

        if recovery == 'model':
            models = [models]

        for mi, model in enumerate(models):
            #Create param dictionary
            num_files = (datasets_to_generate*number_of_runs)+(mi*datasets_to_generate) if recovery != 'model' else None
            fit_params = {'learning_filename':    f'SOMA_RL/data/generated/XX_generated_learning.csv',
                            'transfer_filename':  f'SOMA_RL/data/generated/XX_generated_transfer.csv',
                            'models':             model if recovery=='model' else [model],
                            'random_params':      True,
                            'fixed':              fixed,
                            'bounds':             bounds,
                            'number_of_runs':     number_of_runs,
                            'generated':          True, #TODO: This should be based on whether dataloader was used in last step
                            'clear_data':         False,
                            'progress_bar':       True,
                            'number_of_files':    num_files,
                            'multiprocessing':    multiprocessing,
                            'training':           training,
                            'training_epochs':    training_epochs,
                            'optimizer_lr':       optimizer_lr}

            print(f'\nFitting model: {model}')
            dataloader = self.run_model_fit(**fit_params)
        
        return dataloader

    def determine_parameter_outliers(self, fit_data: dict, dataloader: DataLoader) -> None:

        """
        Determine parameter outliers in the fit data.

        Parameters
        ----------
        fit_data : dict
            Dictionary containing the fit data for each model, with keys as model names and values as DataFrames.
        dataloader : DataLoader
            DataLoader object containing the data for the fit.

        Returns (External)
        ------------------
        outlier_results : pickle
            Dictionary containing the results of the outlier analysis, including participant outliers, model outliers, and a summary DataFrame.
        """

        outliers = {model: {} for model in fit_data}
        participant_outliers = {model: {} for model in fit_data}
        model_outliers = {model: {} for model in fit_data}
        model_params = {model: [] for model in fit_data}
        for model in fit_data:
            model_data = fit_data[model].copy()
            parameter_bounds = RLModel(model).bounds[model.split('+')[0]]
            outliers[model] = {param: [] for param in parameter_bounds}

            for param in parameter_bounds:
                parameter_data = model_data[['participant', 'fit', param]]
                parameter_outliers = parameter_data[(parameter_data[param] == parameter_bounds[param][0]) | (parameter_data[param] == parameter_bounds[param][1])]
                outliers[model][param] = parameter_outliers
            model_params[model] = len(parameter_bounds)
            participant_outliers[model] = pd.concat([outliers[model][param]['participant'] for param in outliers[model]]).value_counts().sort_values(ascending=False)
            model_outliers[model] = participant_outliers[model].value_counts().sort_index()

        #Convert to DataFrame for easier viewing
        outliers_summary = pd.DataFrame(model_outliers).T
        outliers_summary = outliers_summary.fillna(0).astype(int)
        outliers_summary['total'] = outliers_summary.sum(axis=1)

        #Join dataframes into a dictionary
        outlier_results = {'participant_outliers': participant_outliers, 'model_outliers': model_outliers, 'outliers_summary': outliers_summary, 'model_params': model_params}
        outlier_results = self.add_outlier_data(dataloader, outlier_results)

        #Save participant_outliers
        with open('SOMA_RL/fits/parameter_outlier_results.pkl', 'wb') as f:
            pickle.dump(outlier_results, f)
        
    def add_outlier_data(self, dataloader: DataLoader, outlier_results: dict) -> dict:

        """
        Organize outlier data into a DataFrame for each model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader object containing the data for the fit.
        outlier_results : dict
            Dictionary containing the results of the outlier analysis, including participant outliers, model outliers, and a summary DataFrame.

        Returns
        -------
        outlier_results : dict
            Updated dictionary containing the outlier data for each model, with keys as model names and values as DataFrames.

        """
        
        proportion_violations = 0.5
        model_params = outlier_results['model_params']
        participant_outliers = outlier_results['participant_outliers']
        outlier_data = {model: [] for model in participant_outliers}
        for model in participant_outliers:
            count_violations = int(np.round(model_params[model] * proportion_violations, 0))
            model_outliers = participant_outliers[model][participant_outliers[model] > count_violations]
            model_participant_data = pd.DataFrame(columns=['participant', 'accuracy', 'A', 'B', 'E', 'F', 'N'])
            for participant in model_outliers.index:
                participant_choice_rate = {}
                participant_dataloader = copy.copy(dataloader)
                participant_dataloader.filter_participant_data(participant)
                participant_learning, participant_transfer = participant_dataloader.get_data()

                #Compute choice rate
                participant_transfer.loc[:, 'stim1'] = participant_transfer['state'].apply(lambda x: x[-2])
                participant_transfer.loc[:, 'stim2'] = participant_transfer['state'].apply(lambda x: x[-1])
                stims = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']
                stim_pairs = [['A', 'C'], ['B', 'D'], ['E', 'G'], ['F', 'H']]
                for stim in stims:
                    stim_transfer = participant_transfer[(participant_transfer['stim1'] == stim) | (participant_transfer['stim2'] == stim)].copy()
                    stim_transfer.loc[:, 'stim_chosen'] = stim_transfer.apply(lambda x: x['stim1'] if x['action'] == 0 else x['stim2'], axis=1)
                    participant_choice_rate[stim] = stim_transfer['stim_chosen'].value_counts(normalize=True).get(stim, 0)
                reduced_choice_rate = {}
                for pair in stim_pairs:
                    reduced_choice_rate[pair[0]] = (participant_choice_rate[pair[0]] + participant_choice_rate[pair[1]])/2
                reduced_choice_rate['N'] = participant_choice_rate['N']

                #Create participant dataframe
                participant_data = pd.DataFrame({'participant': participant}, index=[0])
                participant_data['accuracy'] = float(1-participant_learning['action'].mean())
                for stim in reduced_choice_rate:
                    participant_data[stim] = reduced_choice_rate[stim]

                #Append to model dataframe
                model_participant_data = pd.concat((model_participant_data, participant_data))
            
            #Append to
            outlier_data[model] = model_participant_data
        
        outlier_results['outlier_data'] = outlier_data

        return outlier_results      

    def describe_fits(self, fit_data: dict) -> None:

        """
        Generate descriptive statistics for the fitted parameters across different pain groups.

        Parameters
        ----------
        fit_data : dict
            Dictionary containing the fit data for each model, with keys as model names and values as DataFrames.

        Returns (External)
        -------
        param_fit_descriptives : csv
            CSV file containing the descriptive statistics of the fitted parameters for each model across different pain groups.
        """

        groups = ['no pain', 'acute pain', 'chronic pain']
        describe_fit = []
        for model in fit_data:
            for param in fit_data[model].columns[4:]:
                group_params = []
                for group in groups:
                    group_indices = fit_data[model]['pain_group'] == group
                    group_data = fit_data[model].loc[group_indices]
                    mean_data = np.round(group_data[param].mean(),2)
                    sd_data = np.round(group_data[param].std(),2)
                    group_params.append(f"{mean_data} ({sd_data})")
                param_data = [model.split('+')[0], param] + group_params
                if not isinstance(describe_fit, pd.DataFrame):
                    describe_fit = pd.DataFrame([param_data], columns=['model', 'parameter'] + groups)
                else:
                    describe_fit = pd.concat((describe_fit, pd.DataFrame([param_data], columns=['model', 'parameter'] + groups)), ignore_index=True, axis=0)

        describe_fit.to_csv('SOMA_RL/stats/param_fit_descriptives.csv', index=False)