import sys
sys.dont_write_bytecode = True
import os
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from scipy import stats


from helpers.tasks import AvoidanceLearningTask
from helpers.rl_models import QLearning, ActorCritic, Relative, Hybrid, Hybrid2

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

        def __init__(self, task, model, trial_design):

            #Get parameters
            self.trial_design = trial_design
            self.task = task
            self.task.initiate_model(model)

        def simulate(self):

            #Run simulation and computations
            self.task.run_experiment(self.trial_design)
            self.task.rl_model.run_computations()

            return self.task.rl_model

if __name__ == "__main__":

    # =========================================== #
    # ======= EXAMPLE RUNNING SIMULATIONS ======= #
    # =========================================== #

    #Metaparameters
    rnd.seed(1251)
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

            if m == 'QLearning':
                model = QLearning(factual_lr=0.1, 
                                  counterfactual_lr=0.05, 
                                  temperature=0.1)

            elif m == 'ActorCritic':
                model = ActorCritic(factual_actor_lr=0.1, 
                                    counterfactual_actor_lr=0.05, 
                                    critic_lr=0.1, 
                                    temperature=0.1, 
                                    valence_factor=0.5)

            elif m == 'Relative':
                model = Relative(factual_lr=0.1, 
                                 counterfactual_lr=0.05, 
                                 contextual_lr=0.1, 
                                 temperature=0.1)

            elif m == 'Hybrid2012':
                model = Hybrid(factual_lr=0.1, 
                               counterfactual_lr=0.05, 
                               factual_actor_lr=0.1, 
                               counterfactual_actor_lr=0.05, 
                               critic_lr=0.1, 
                               temperature=0.1, 
                               mixing_factor=0.5, 
                               valence_factor=0.5)

            elif m == 'Hybrid2021':
                model = Hybrid2(factual_lr=0.1, 
                               counterfactual_lr=0.05, 
                               factual_actor_lr=0.1, 
                               counterfactual_actor_lr=0.05, 
                               critic_lr=0.1, 
                               temperature=0.1, 
                               mixing_factor=0.5, 
                               valence_factor=0.5,
                               noise_factor=0.1,
                               decay_factor=0.1)
            else:
                raise ValueError('Model not recognized.')

            #Initialize pipeline
            model = RLPipeline(task, model, task_design).simulate()

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

            match m:
                case 'QLearning':
                    value_label = 'q_values'
                case 'ActorCritic':
                    value_label = 'w_values'
                case 'Relative':
                    value_label = 'q_values'
                case 'Hybrid2012':
                    value_label = 'h_values'
                case 'Hybrid2021':
                    value_label = 'h_values'

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