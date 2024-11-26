import sys
sys.dont_write_bytecode = True
import random as rnd
import pandas as pd

from helpers.tasks import AvoidanceLearningTask
from helpers.rl_models import QLearning, ActorCritic, Relative, Hybrid

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

    #Initialize task, model, and task design
    task = AvoidanceLearningTask()
    #model = QLearning(factual_lr=0.1, counterfactual_lr=0.05, temperature=0.1)
    #model = Relative(factual_lr=0.1, counterfactual_lr=0.05, contextual_lr=0.1, temperature=0.1)
    #model = ActorCritic(factual_actor_lr=0.1, counterfactual_actor_lr=0.05, factual_critic_lr=0.1, counterfactual_critic_lr=0.05, temperature=0.1)
    model = Hybrid(factual_lr=0.1, counterfactual_lr=0.05, factual_actor_lr=0.1, counterfactual_actor_lr=0.05, 
                 factual_critic_lr=0.1, counterfactual_critic_lr=0.05, temperature=0.1, mixing_factor=0.5)
    task_design = {'learning_phase': {'number_of_trials': 24, 'number_of_blocks': 4},
                    'transfer_phase': {'times_repeated': 4}}
            
    #Initialize pipeline
    q_learning = RLPipeline(task, model, task_design).simulate()

    #Finalize and view model
    q_learning.plot_model()

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