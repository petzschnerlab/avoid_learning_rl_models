import sys
sys.dont_write_bytecode = True
import os
import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

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
    # === EXAMPLE RUNNING A SINGLE SIMULATION === #
    # =========================================== #
    number_of_runs = 100
    models = ['QLearning', 'ActorCritic', 'Relative', 'Hybrid2012', 'Hybrid2021']

    transfer_columns = ['A', 'B', 'E', 'F', 'N']
    accuracy_columns = ['context', 'trial_total', 'accuracy']
    pe_columns = ['context', 'trial_total', 'averaged_pe']
    accuracy = {m: [] for m in models}
    prediction_errors = {m: [] for m in models}
    choice_rates = {m: [] for m in models}
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
            task_learning_data = model.task_learning_data
            task_learning_data['trial_total'] = task_learning_data.groupby('state_id').cumcount()+1
            if 'v_prediction_errors' in task_learning_data.columns:
                task_learning_data['prediction_errors'] = task_learning_data['v_prediction_errors'] + task_learning_data['q_prediction_errors']
            task_learning_data['averaged_pe'] = task_learning_data['prediction_errors'].apply(lambda x: sum(x)/len(x))
            learning_accuracy = task_learning_data.groupby(['context', 'trial_total'])['accuracy'].mean().reset_index()
            learning_prediction_errors = task_learning_data.groupby(['context', 'trial_total'])['averaged_pe'].mean().reset_index()
            if not type(choice_rates[m]) == pd.DataFrame:
                accuracy[m] = pd.DataFrame(learning_accuracy, columns=accuracy_columns)
                prediction_errors[m] = pd.DataFrame(learning_prediction_errors, columns=pe_columns)
                choice_rates[m] = pd.DataFrame([model.choice_rate], columns=transfer_columns)
            else:
                accuracy[m] = pd.concat([accuracy[m], pd.DataFrame(learning_accuracy)], ignore_index=True)
                prediction_errors[m] = pd.concat([prediction_errors[m], pd.DataFrame(learning_prediction_errors)], ignore_index=True)
                choice_rates[m] = pd.concat([choice_rates[m], pd.DataFrame([model.choice_rate])], ignore_index=True)

    #Plot simulations    
    colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']
    bi_colors = ['#FB9A99', '#B2DF8A']
    fig, ax = plt.subplots(3, len(models), figsize=(4*len(models), 10))
    for i, m in enumerate(models):

        #Plot accuracy
        model_accuracy = accuracy[m].groupby(['context','trial_total']).mean().reset_index()
        model_accuracy['context'] = pd.Categorical(model_accuracy['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
        for ci, context in enumerate(model_accuracy['context'].unique()):
            context_accuracy = model_accuracy[model_accuracy['context'] == context]['accuracy'].reset_index(drop=True)
            ax[0, i].plot(context_accuracy*100, color=bi_colors[ci], alpha = .8, label=context)
        ax[0, i].set_title(m)
        ax[0, i].set_ylim([40, 105])
        if i == 0:
            ax[0, i].set_ylabel('Accuracy (%)')
        ax[0, i].set_xlabel('Trial')
        if i == len(models)-1:
            ax[0, i].legend(loc='lower right')
        ax[0, i].spines['top'].set_visible(False)
        ax[0, i].spines['right'].set_visible(False)

        #Plot prediction errors
        model_pe = prediction_errors[m].groupby(['context','trial_total']).mean().reset_index()
        model_pe['context'] = pd.Categorical(model_pe['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
        for ci, context in enumerate(model_pe['context'].unique()):
            context_pe = model_pe[model_pe['context'] == context]['averaged_pe'].reset_index(drop=True)
            ax[1, i].plot(context_pe, color=bi_colors[ci], alpha = .8, label=context)
        ax[1, i].set_ylim([-1, 1])
        if i == 0:
            ax[1, i].set_ylabel('Prediction Error')
        ax[1, i].set_xlabel('Trial')
        if i == len(models)-1:
            ax[1, i].legend(loc='lower right')
        ax[1, i].spines['top'].set_visible(False)
        ax[1, i].spines['right'].set_visible(False)

        #Plot choice rates
        ax[2, i].bar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), color=colors, alpha = .5)
        ax[2, i].errorbar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), yerr=choice_rates[m].sem(), fmt='.', color='grey')
        ax[2, i].set_ylim([0, 100])
        if i == 0:
            ax[2, i].set_ylabel('Choice rate (%)')
        ax[2, i].spines['top'].set_visible(False)
        ax[2, i].spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join('SOMA_RL','plots','model_simulations.png'))
    plt.show()
    
    #Debug stop
    print()