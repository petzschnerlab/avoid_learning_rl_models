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

    stims = ['A', 'B', 'E', 'F', 'N']
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
            if not type(choice_rates[m]) == pd.DataFrame:
                choice_rates[m] = pd.DataFrame([model.choice_rate], columns=stims)
            else:
                choice_rates[m] = pd.concat([choice_rates[m], pd.DataFrame([model.choice_rate])], ignore_index=True)

    #Compute SEM    
    colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']
    fig, ax = plt.subplots(1, len(models), figsize=(4*len(models), 5))
    for i, m in enumerate(models):
        ax[i].bar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), color=colors, alpha = .5)
        ax[i].errorbar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), yerr=choice_rates[m].sem(), fmt='.', color='grey')
        ax[i].set_title(m)
        ax[i].set_ylim([0, 100])
        if i == 0:
            ax[i].set_ylabel('Choice rate (%)')
    fig.tight_layout()
    fig.savefig(os.path.join('SOMA_RL','plots','model_simulations.png'))
    plt.show()

    print('debug')

    '''
    # ============================================== #
    # === EXAMPLE RUNNING SEQUENTIAL SIMULATIONS === #
    # ============================================== #

    parameters = []
    for i in range(100):
        print(f'\nRunning simulation {i}')

        task = AvoidanceLearningTask()
        model = QLearning(factual_lr=rnd.random(), #Randomize parameters
                          counterfactual_lr=rnd.random(), 
                          temperature=rnd.random())

        task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                       'transfer_phase': {'times_repeated': 4}}

        q_learning = RLPipeline(task, model, task_design).simulate()

        model_data = {'index': i, 
                      'factual_lr': model.factual_lr, 
                      'counterfactual_lr': model.counterfactual_lr, 
                      'temperature': model.temperature, 
                      '75R': model.choice_rate['A'],
                      '25R': model.choice_rate['B'],
                      '25P': model.choice_rate['E'],
                      '75P': model.choice_rate['F'],
                      'N': model.choice_rate['N']}
        
        if not type(parameters) == pd.DataFrame:
            parameters = pd.DataFrame([model_data])
        else:
            parameters = pd.concat([parameters, pd.DataFrame([model_data])], ignore_index=True)
    
    '''
    
    #Debug stop
    print()