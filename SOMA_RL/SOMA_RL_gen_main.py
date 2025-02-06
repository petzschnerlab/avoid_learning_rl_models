import os
import random as rnd
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from helpers.analyses import generate_simulated_data, run_fit, run_fit_comparison

if __name__ == "__main__":

    # =========================================== #
    # ================= INPUTS ================== #
    # =========================================== #

    #Seed random number generator
    rnd.seed(1251)

    #Parameters
    multiprocessing = True
    parameters = 'random'
    number_of_runs = 100

    #Models
    '''
    Supported models: 
        QLearning, ActorCritic
        Relative, wRelative, QRelative
        Hybrid2012, Hybrid2021

    Standard models:
        QLearning: Standard Q-Learning Model
        ActorCritic: Standard Actor-Critic Model
        Relative: Standard Relative Model (Palminteri et al., 2015)
        wRelative+bias+decay: Standard Weighted-Relative Model [Proposed model] (Williams et al., in prep)
        Hybrid2012+bias: Standard Hybrid 2012 Model (Gold et al., 2012)
        Hybrid2021+bias+decay: Standard Hybrid 2021 Model (Geana et al., 2021)

    Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign
        +bias: Adds a valence bias to the model (e.g. wRelative+bias), only usable with wRelative, QRelative, Hybrid2012, and Hybrid2021
        +novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models
        +decay: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models
    '''

    models = ['QLearning', #Standard
              'QLearning+novel', #Standard + novel
              
              'ActorCritic', #Standard
              'ActorCritic+novel', #Standard + novel

              'Relative', #Standard
              'Relative+novel', #Standard + novel
              'wRelative+bias+decay', #Standard
              'wRelative+bias+decay+novel', #Standard + novel

              'Hybrid2012+bias', #Standard w/o bias
              'Hybrid2012+bias+novel', #Standard + novel
              'Hybrid2021+bias+decay', #Standard w/o bias
              'Hybrid2021+bias+decay+novel'] #Standard + novel

    #Task Design
    task_design = {'learning_phase': {
                        'number_of_trials': 24,
                        'number_of_blocks': 4,},
                    'transfer_phase': {
                        'times_repeated': 4}}
    
    # =========================================== #
    # ============== RUN ANALYSES =============== #
    # =========================================== #

    generate_simulated_data(models, parameters, task_design, number_of_runs=number_of_runs, multiprocessing=multiprocessing)

    # =========================================== #
    # ================ RUN FITS ================= #
    # =========================================== #

    #Parameters
    multiprocessing = True
    random_params = True
    number_of_runs = 5

    #Find all files in SOMA_RL/data/generated
    generated_filenames = os.listdir('SOMA_RL/data/generated')
    for f in os.listdir('SOMA_RL/fits/temp'):
        os.remove(os.path.join('SOMA_RL','fits','temp',f))

    loop = tqdm.tqdm(range(len(generated_filenames)))
    columns = {model: [] for model in models}
    for mi, model in enumerate(models):
        #Find all files that start with model
        data_names = [filename for filename in generated_filenames if model == filename.split('_')[0]]
        for di, data_name in enumerate(data_names):
            learning_filename = f'SOMA_RL/data/generated/{data_name}/{data_name}_generated_learning.csv'
            transfer_filename = f'SOMA_RL/data/generated/{data_name}/{data_name}_generated_transfer.csv'
            metadata = pd.read_csv(f'SOMA_RL/data/generated/{data_name}/{data_name}_generated_parameters.csv')

            number_of_files = mi*len(data_names) + di + number_of_runs
            dataloader, cols = run_fit(learning_filename, 
                                        transfer_filename, 
                                        [model], 
                                        random_params=random_params, 
                                        number_of_runs=number_of_runs, 
                                        generated=True, 
                                        clear_data=False, 
                                        progress_bar=False, 
                                        number_of_files=number_of_files,
                                        multiprocessing=multiprocessing)
            columns[model] = cols[model]
            loop.update(1)

    fit_data = run_fit_comparison(dataloader, models, ['simulated'], columns)

    # =========================================== #
    # ======= CREATE CORRELATIONAL PLOTS ======== #
    # =========================================== #

    #Create a dictionary with model being keys and pd.dataframe empty as value
    fit_results = {model: [] for model in models}
    for model in models:
        model_data = fit_data[model]
        for run_params in model_data['participant']:
            
            true_parameters = pd.read_csv(f'SOMA_RL/data/generated/{model}_{run_params}/{model}_{run_params}_generated_parameters.csv')
            fit_parameters = pd.DataFrame(model_data[model_data['participant']==run_params].values[0][4:]).T
            fit_parameters.columns = true_parameters.columns

            true_parameters['Model'] = model
            fit_parameters['Model'] = model
            true_parameters['fit_type'] = 'True'
            fit_parameters['fit_type'] = 'Fit'
            combined_parameters = pd.concat([true_parameters, fit_parameters])
            combined_parameters = combined_parameters[['Model', 'fit_type'] + [col for col in combined_parameters.columns if col not in ['Model', 'fit_type']]]

            if isinstance(fit_results[model], pd.DataFrame):
                fit_results[model] = pd.concat([fit_results[model], combined_parameters])
            else:
                fit_results[model] = combined_parameters            

    for model in models:
        #Plot correlation plots, new figure for each model, subplot for each parameter
        fig, axs = plt.subplots(1, len(fit_results[model].columns)-2, figsize=(5*len(fit_results[model].columns)-2, 5))
        for i, parameter in enumerate(fit_results[model].columns[2:]):
            axs[i].scatter(fit_results[model][fit_results[model]['fit_type']=='True'][parameter], 
                           fit_results[model][fit_results[model]['fit_type']=='Fit'][parameter])
            axs[i].set_title(parameter)
            axs[i].set_xlabel('True')
            axs[i].set_ylabel('Fit')

        fig.suptitle(f'{model} Correlation Plot')
        plt.tight_layout()
        plt.savefig(f'SOMA_RL/plots/{model}_correlation_plot.png')