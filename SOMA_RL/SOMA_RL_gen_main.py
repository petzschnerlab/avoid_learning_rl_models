import os
import random as rnd
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from helpers.analyses import run_generate_and_fit
from models.rl_models import RLModel

if __name__ == "__main__":

    # =========================================== #
    # ================= INPUTS ================== #
    # =========================================== #

    #Seed random number generator
    rnd.seed(1251)

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

    task_design = {'learning_phase': {
                        'number_of_trials': 24,
                        'number_of_blocks': 4,},
                    'transfer_phase': {
                        'times_repeated': 4}}
    
    generate_params = {'models':                models,
                       'task_design':           task_design,
                       'parameters':            'random',
                       'datasets_to_generate':  100,
                       'number_of_runs':        20,
                       'multiprocessing':       True
                       }

    run_generate_and_fit(**generate_params)
