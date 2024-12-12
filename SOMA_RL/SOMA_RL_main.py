import sys
sys.dont_write_bytecode = True
import os
import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy import stats

from helpers.tasks import AvoidanceLearningTask
from helpers.rl_models import get_model
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
        trial_design : dict
            Dictionary containing trial design parameters
        """

        def __init__(self, model, task=None, trial_design=None):

            #Get parameters
            self.trial_design = trial_design
            self.task = task
            self.task.initiate_model(model)

        def simulate(self):

            #Run simulation and computations
            self.task.run_experiment(self.trial_design)
            self.task.rl_model.run_computations()

            return self.task.rl_model
    
        def fit(self, data, bounds):

            return self.task.rl_model.fit(data, bounds)

if __name__ == "__main__":

    rnd.seed(1251)

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
    data = data[data['participant'].isin(data['participant'].unique()[:5])]
    
    #Setup fit dataframe
    models = ['QRelative']#, 'QLearning', 'ActorCritic', 'Relative', 'Hybrid2012', 'QRelative'] #, 'Hybrid2021']
    
    columns = {'QLearning': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'temperature'],
               'ActorCritic': ['participant', 'pain_group', 'fit', 'factual_actor_lr', 'counterfactual_actor_lr', 'critic_lr', 'temperature', 'valence_factor'],
               'Relative': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'contextual_lr', 'temperature'],
               'Hybrid2012': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'factual_actor_lr', 'counterfactual_actor_lr', 'critic_lr', 'temperature', 'mixing_factor', 'valence_factor'],
               'Hybrid2021': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'factual_actor_lr', 'counterfactual_actor_lr', 'critic_lr', 'temperature', 'mixing_factor', 'valence_factor', 'noise_factor', 'decay_factor'],
               'QRelative': ['participant', 'pain_group', 'fit', 'factual_lr', 'counterfactual_lr', 'contextual_lr', 'temperature', 'mixing_factor']}

    fit_data = {model: pd.DataFrame(columns=columns[model]) for model in models}

    #Task setup
    task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                           'transfer_phase': {'times_repeated': 4}}
    
    # =========================================== #
    # =================== FIT =================== #
    # =========================================== #
    loop = tqdm.tqdm(range(data['participant'].nunique()*len(models)))
    for n, participant in enumerate(data['participant'].unique()):

        #Update loop
        loop.set_description(f"Participant {n+1}/{data['participant'].nunique()}")

        #Extract participant data
        participant_data = data[data['participant'] == participant]
        states = participant_data['state'].values
        actions = participant_data['action'].values
        rewards = participant_data[['reward_L', 'reward_R']].values

        for m in models:
            loop.update(1)
            #Fit models
            task = AvoidanceLearningTask()
            model = get_model(m)

            pipeline = RLPipeline(model, task, task_design)
            fit_results, fitted_params = pipeline.fit((states, actions, rewards), bounds=model.bounds)
            
            #Store fit results
            participant_fitted = [participant, 
                                participant_data['pain_group'].values[0], 
                                fit_results.fun]
            participant_fitted.extend([fitted_params[key] for key in columns[m][3:]])
            
            if len(fit_data[m]) == 0:
                fit_data[m] = pd.DataFrame([participant_fitted], columns=fit_data[m].columns)
            else:
                fit_data[m] = pd.concat((fit_data[m], 
                                    pd.DataFrame([participant_fitted], columns=fit_data[m].columns)),
                                    ignore_index=True)

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
        run_type = 'fit' #fit or simulate

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
                task = AvoidanceLearningTask()
                task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                            'transfer_phase': {'times_repeated': 4}}

                #Initialize model
                model = get_model(m, group_data[m].iloc[n])

                #Initialize pipeline
                model = RLPipeline(model, task, task_design).simulate()

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
    # ================================================================================================================ #
    # ================================================================================================================ #

    # =========================================== #
    # =============== SIMULATIONS =============== #
    # =========================================== #

    #Metaparameters
    number_of_runs = 100
    models = ['QLearning', 'ActorCritic', 'Relative', 'Hybrid2012', 'Hybrid2021']

    #Set up data columns
    accuracy_columns = ['context', 'trial_total', 'accuracy', 'run']
    pe_columns = ['context', 'trial_total', 'averaged_pe', 'run']
    values_columns = ['context', 'trial_total', 'values1', 'values2']
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
            loop.set_description(f'Run {n+1}/{number_of_runs}')

            #Initialize task, model, and task design
            task = AvoidanceLearningTask()
            task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                           'transfer_phase': {'times_repeated': 4}}
            model = get_model(m)
            model = RLPipeline(model, task, task_design).simulate()

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

            value_labels = {'QLearning': 'q_values', 
                           'ActorCritic': 'w_values', 
                           'Relative': 'q_values', 
                           'Hybrid2012': 'h_values', 
                           'Hybrid2021': 'h_values'}
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

    #Plot simulations    
    colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']
    bi_colors = ['#B2DF8A', '#FB9A99']
    val_colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C']
    fig, ax = plt.subplots(4, len(models), figsize=(4*len(models), 15))
    for i, m in enumerate(models):

        #Plot accuracy
        model_accuracy = accuracy[m].groupby(['context','trial_total','run']).mean().reset_index()
        model_accuracy['context'] = pd.Categorical(model_accuracy['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
        for ci, context in enumerate(['Reward', 'Loss Avoid']):
            CIs = model_accuracy.groupby(['context','trial_total'])['accuracy'].sem()*stats.t.ppf(0.975, number_of_runs-1)
            averaged_accuracy = model_accuracy.groupby(['context','trial_total']).mean().reset_index()
            context_accuracy = averaged_accuracy[averaged_accuracy['context'] == context]['accuracy'].reset_index(drop=True).astype(float)*100
            context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True)*100
            ax[0, i].fill_between(context_accuracy.index, context_accuracy - context_CIs, context_accuracy + context_CIs, alpha=0.2, color=bi_colors[ci], edgecolor='none')
            ax[0, i].plot(context_accuracy, color=bi_colors[ci], alpha = .8, label=context.replace('Loss Avoid', 'Punish'))
        ax[0, i].set_title(m)
        ax[0, i].set_ylim([25, 110])
        if i == 0:
            ax[0, i].set_ylabel('Accuracy (%)')
        ax[0, i].set_xlabel('Trial')
        if i == len(models)-1:
            ax[0, i].legend(loc='lower right', frameon=False)
        ax[0, i].spines['top'].set_visible(False)
        ax[0, i].spines['right'].set_visible(False)

        #Plot prediction errors
        model_pe = prediction_errors[m].groupby(['context','trial_total', 'run']).mean().reset_index()
        model_pe['context'] = pd.Categorical(model_pe['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
        for ci, context in enumerate(['Reward', 'Loss Avoid']):
            CIs = model_pe.groupby(['context','trial_total'])['averaged_pe'].sem()*stats.t.ppf(0.975, number_of_runs-1)
            averaged_pe = model_pe.groupby(['context','trial_total']).mean().reset_index()
            context_pe = averaged_pe[averaged_pe['context'] == context]['averaged_pe'].reset_index(drop=True)
            context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True)
            ax[1, i].fill_between(context_pe.index, context_pe - context_CIs, context_pe + context_CIs, alpha=0.2, color=bi_colors[ci], edgecolor='none')
            ax[1, i].plot(context_pe, color=bi_colors[ci], alpha = .8, label=context.replace('Loss Avoid', 'Punish'))
        ax[1, i].set_ylim([-.75, .75])
        if i == 0:
            ax[1, i].set_ylabel('Prediction Error')
        ax[1, i].set_xlabel('Trial')
        if i == len(models)-1:
            ax[1, i].legend(loc='lower right', frameon=False)
        ax[1, i].spines['top'].set_visible(False)
        ax[1, i].spines['right'].set_visible(False)
        ax[1, i].axhline(0, linestyle='--', color='grey', alpha=.5)

        #Plot values
        model_values = values[m].groupby(['context','trial_total', 'run']).mean().reset_index()
        model_values['context'] = pd.Categorical(model_values['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
        for ci, context in enumerate(['Reward', 'Loss Avoid']):
            for vi, val in enumerate(['values1', 'values2']):
                CIs = model_values.groupby(['context','trial_total'])[val].sem()*stats.t.ppf(0.975, number_of_runs-1)
                averaged_values = model_values.groupby(['context','trial_total']).mean().reset_index()
                context_values = averaged_values[averaged_values['context'] == context][val].reset_index(drop=True)
                context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True)
                ax[2, i].fill_between(context_values.index, context_values - context_CIs, context_values + context_CIs, alpha=0.2, color=val_colors[ci*2+vi], edgecolor='none')
                ax[2, i].plot(context_values, color=val_colors[ci*2+vi], alpha = .8, label=['High Reward', 'Low Reward', 'Low Punish', 'High Punish'][ci*2+vi])
        ax[2, i].set_ylim([-1, 1])
        if i == 0:
            ax[2, i].set_ylabel('q/w/h Value')
        ax[2, i].set_xlabel('Trial')
        if i == len(models)-1:
            ax[2, i].legend(loc='lower left', frameon=False, ncol=2)
        ax[2, i].spines['top'].set_visible(False)
        ax[2, i].spines['right'].set_visible(False)
        ax[2, i].axhline(0, linestyle='--', color='grey', alpha=.5)

        #Plot choice rates
        ax[3, i].bar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), color=colors, alpha = .5)
        ax[3, i].errorbar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), yerr=choice_rates[m].sem(), fmt='.', color='grey')
        ax[3, i].set_ylim([0, 100])
        if i == 0:
            ax[3, i].set_ylabel('Choice rate (%)')
        ax[3, i].spines['top'].set_visible(False)
        ax[3, i].spines['right'].set_visible(False)

    #Metaplot settings
    fig.tight_layout()
    fig.savefig(os.path.join('SOMA_RL','plots','model_simulations.png'))
    plt.show()
    
    #Debug stop
    print()