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
from helpers.pipeline import RLPipeline, mp_run_fit, mp_run_simulations, mp_progress
from helpers.plotting import plot_simulations, plot_simulations_behaviours, plot_parameter_fits, plot_model_fits, plot_rainclouds, plot_fits_by_run_number
from helpers.statistics import Statistics


def run_fit_empirical(learning_filename, 
                      transfer_filename, 
                      models, 
                      number_of_participants=0,
                      fixed=None, 
                      bounds=None,  
                      random_params=False, 
                      number_of_runs=1, 
                      generated=False, 
                      multiprocessing=False,
                      training='torch',
                      training_epochs=1000,
                      optimizer_lr=0.01):

    #Report each of the inputs
    print('Running Empirical Fit with the following parameters:')
    print('--------------------------------------------------------')
    print(f'Models: {models}')
    print(f'Learning Filename: {learning_filename}')
    print(f'Transfer Filename: {transfer_filename}')
    print(f'Number of Participants: {number_of_participants}')
    print(f'Random Parameters: {random_params}')
    print(f'Number of Runs: {number_of_runs}')
    print(f'Generated: {generated}')
    print(f'Multiprocessing: {multiprocessing}')
    print(f'Training: {training}')
    print(f'Training Epochs: {training_epochs}')
    print(f'Optimizer Learning Rate: {optimizer_lr}')

    if fixed is not None:
        print('\nParameter Overwrites:')
        for model in fixed:
            print(f'       {model}:')
            for key in fixed[model]:
                print(f'              {key} = {fixed[model][key]}')
    if bounds is not None:
        print('\nParameter Bound Overwrites:')
        for model in bounds:
            print(f'       {model}:')
            for key in bounds[model]:
                print(f'              {key} = {bounds[model][key]}')
    print('--------------------------------------------------------')
    
    dataloader = run_fit(learning_filename, 
                         transfer_filename, 
                         models, 
                         number_of_participants=number_of_participants, 
                         fixed=fixed,
                         bounds=bounds,
                         random_params=random_params, 
                         number_of_runs=number_of_runs, 
                         generated=generated, 
                         multiprocessing=multiprocessing,
                         training=training,
                         training_epochs=training_epochs,
                         optimizer_lr=optimizer_lr)
    
    fit_data = run_fit_comparison(dataloader, 
                                  models, 
                                  dataloader.get_group_ids())
    
    plot_fits_by_run_number('SOMA_RL/fits/full_fit_data.pkl')
    
    linear_results, ttest_results = run_fit_analyses(fit_data)
    
    run_fit_simulations(learning_filename, 
                        transfer_filename, 
                        fit_data, 
                        models, 
                        dataloader.get_participant_ids(), 
                        dataloader.get_group_ids(), 
                        number_of_participants=number_of_participants, 
                        multiprocessing=multiprocessing)

def run_recovery(models, 
                 parameters, 
                 learning_filename=None, 
                 transfer_filename=None, 
                 task_design=None, 
                 fixed=None, 
                 bounds=None, 
                 datasets_to_generate=1, 
                 number_of_runs=1, 
                 number_of_participants=0, 
                 multiprocessing=False, 
                 clear_data=True, 
                 recovery='parameter',
                 training='torch',
                 training_epochs=1000,
                 optimizer_lr=0.01):

    #Report each of the inputs
    print('Running Generation and Fit with the following parameters:')
    print('--------------------------------------------------------')
    print(f'Models: {models}')
    print(f'Parameters: {parameters}')
    print(f'Learning Filename: {learning_filename}')
    print(f'Transfer Filename: {transfer_filename}')
    print(f'Task Design: {task_design}')
    print(f'Datasets to Generate: {datasets_to_generate}')
    print(f'Number of Runs: {number_of_runs}')
    print(f'Number of Participants: {number_of_participants}')
    print(f'Multiprocessing: {multiprocessing}')
    print(f'Clear Data: {clear_data}')
    print(f'Recovery: {recovery}')
    print(f'Training: {training}')
    print(f'Training Epochs: {training_epochs}')
    print(f'Optimizer Learning Rate: {optimizer_lr}')

    if fixed is not None:
        print('\nParameter Overwrites:')
        for model in fixed:
            print(f'       {model}:')
            for key in fixed[model]:
                print(f'              {key} = {fixed[model][key]}')
    if bounds is not None:
        print('\nParameter Bound Overwrites:')
        for model in bounds:
            print(f'       {model}:')
            for key in bounds[model]:
                print(f'              {key} = {bounds[model][key]}')
    print('--------------------------------------------------------')

    if not os.path.exists('SOMA_RL/data/generated'):
        os.makedirs('SOMA_RL/data/generated')

    if [learning_filename, transfer_filename].count(None) == 0:
        datasets_to_generate = len(DataLoader(learning_filename, transfer_filename, number_of_participants=number_of_participants, reduced=False).get_participant_ids())

    generate_simulated_data(models=models, 
                            parameters=parameters, 
                            learning_filename=learning_filename, 
                            transfer_filename=transfer_filename, 
                            task_design=task_design, 
                            fixed=fixed, 
                            bounds=bounds, 
                            datasets_to_generate=datasets_to_generate, 
                            number_of_participants=number_of_participants, 
                            multiprocessing=False, # TODO: True fails on ODD for some reason 
                            clear_data=clear_data)
    
    dataloader = run_generative_fits(models=models, 
                                     number_of_runs=number_of_runs, 
                                     datasets_to_generate=datasets_to_generate, 
                                     fixed=fixed, 
                                     bounds=bounds, 
                                     multiprocessing=multiprocessing, 
                                     recovery=recovery,
                                     training=training,
                                     training_epochs=training_epochs,
                                     optimizer_lr=optimizer_lr)
    
    fit_data = run_fit_comparison(dataloader=dataloader, 
                                  models=models, 
                                  group_ids=['simulated'], 
                                  recovery=recovery)
    
    if recovery == 'model':
        confusion_matrix = create_confusion_matrix(dataloader=dataloader, 
                                                   fit_data=fit_data)
        
        plot_model_fits(confusion_matrix=confusion_matrix)
    else:
        plot_parameter_fits(models=models, 
                            fit_data=fit_data, 
                            fixed=fixed, 
                            bounds=bounds)

def run_fit_analyses(fit_data):

    linear_results = None
    ttest_results = None
    for model in fit_data:
        model_data = fit_data[model]
        fit_parameters = model_data.columns[4:]
        for parameter in fit_parameters:
            statistics = Statistics()
            
            #turn pain_group in model_data into a category
            model_data['pain_group'] = pd.Categorical(model_data['pain_group'], categories=['no pain', 'acute pain', 'chronic pain'])
            linear_model = statistics.linear_model_categorical(f'{parameter} ~ pain_group', model_data)
            linear_result = linear_model['model_summary']
            linear_result.insert(0, 'parameter', parameter)
            linear_result.insert(0, 'model', model)
            if linear_results is None:
                linear_results = linear_result
            else:
                linear_results = pd.concat((linear_results, linear_result), ignore_index=True)

            comparisons = [['no pain', 'acute pain'], ['no pain', 'chronic pain']]
            model_data = model_data.rename(columns={'participant': 'participant_id'})
            ttest_model = statistics.planned_ttests(parameter, 'pain_group', comparisons, model_data)
            ttest_result = ttest_model['model_summary']
            ttest_result.insert(0, 'parameter', parameter)
            ttest_result.insert(0, 'model', model)
            if ttest_results is None:
                ttest_results = ttest_result
            else:
                ttest_results = pd.concat((ttest_results, ttest_result), ignore_index=True)

    linear_results.to_csv('SOMA_RL/stats/pain_fits_linear_results.csv', index=False)
    ttest_results.to_csv('SOMA_RL/stats/pain_fits_ttest_results.csv', index=False)

    for model in fit_data:
        plot_rainclouds(f'{model}-model-fits', fit_data[model])

    return linear_results, ttest_results

def create_confusion_matrix(dataloader, fit_data):
    confusion_matrix = pd.DataFrame(index=fit_data.keys(), columns=fit_data.keys()) #Rows = model fit, Columns = generated model

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
            
def run_fit(learning_filename, 
            transfer_filename,
            models, 
            fixed=None, 
            bounds=None, 
            number_of_participants=0, 
            random_params=False, 
            number_of_runs=1, 
            number_of_files=None, 
            generated=False, 
            clear_data=True, 
            progress_bar=True, 
            multiprocessing=False,
            training='torch',
            training_epochs=1000,
            optimizer_lr=0.01):
    
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
                        pipeline.run_fit((columns[model_name.split('+')[0]], participant, run))
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

def run_fit_comparison(dataloader, models, group_ids, recovery='parameter'):
    
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

    #Save fit data as a pickle file
    with open('SOMA_RL/fits/full_fit_data.pkl', 'wb') as f:
        pickle.dump(full_fit_data, f)

    with open('SOMA_RL/fits/fit_data.pkl', 'wb') as f:
        pickle.dump(fit_data, f)

    #Delete all files
    for f in files:
        os.remove(os.path.join('SOMA_RL','fits','temp',f))

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

    return fit_data

def run_fit_simulations(learning_filename, transfer_filename, fit_data, models, participant_ids, group_ids, number_of_participants=0, multiprocessing=False):

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
        
    #Plot simulations 
    for group in accuracy:
        plot_simulations(accuracy[group], prediction_errors[group], values[group], choice_rates[group], models, group, dataloader)
    plot_simulations_behaviours(accuracy_model, choice_rates_model, models, accuracy.keys(), dataloader)

def generate_simulated_data(models, parameters, learning_filename=None, transfer_filename=None, task_design=None, fixed=None, bounds=None, datasets_to_generate=1, number_of_participants=0, multiprocessing=False, clear_data=True):
    '''
    Parameters
    ----------
    models : list
        List of models to simulate.
    parameters : dict | str
        Dictionary of model parameters or 'random'
    task_design : dict
        Dictionary of task design parameters
    number_of_runs : int
        Number of times to run the simulation for each model (only applicable when parameters='random')
    '''

    #Checks
    if [learning_filename, transfer_filename].count(None) == 1:
        raise ValueError('Both learning and transfer filenames must be provided')

    #Delete all folders with generated data from previous run, if desired
    if clear_data:
        for f in os.listdir('SOMA_RL/data/generated'):
            if '.csv' in f:
                os.remove(os.path.join('SOMA_RL','data','generated',f))
            else:
                shutil.rmtree(os.path.join('SOMA_RL','data','generated',f))

    #Set up parameters
    random_params = True if parameters == 'random' or parameters == 'normal' else False
    datasets_to_generate = datasets_to_generate if random_params else 1

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

            model_parameters = parameters[model_name] if not random_params else None
            model = RLModel(model_name, model_parameters, random_params=parameters, fixed=fixed, bounds=bounds)
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

def run_generative_fits(models, number_of_runs=1, datasets_to_generate=1, fixed=None, bounds=None, multiprocessing=False, recovery='parameter', training='torch', training_epochs=1000, optimizer_lr=0.01):

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
        dataloader = run_fit(**fit_params)
    
    return dataloader

    