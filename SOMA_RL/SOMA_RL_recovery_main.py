import os
import random as rnd
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from helpers.analyses import run_recovery
from helpers.priors import get_priors
from helpers.pipeline import export_recovery

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
        Relative
        Hybrid2012, Hybrid2021

    Standard models:
        QLearning: Standard Q-Learning Model
        ActorCritic: Standard Actor-Critic Model
        Relative: Standard Relative Model (Palminteri et al., 2015)
        Hybrid2012+bias: Standard Hybrid 2012 Model (Gold et al., 2012)
        Hybrid2021+bias+decay: Standard Hybrid 2021 Model (Geana et al., 2021)

    Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign
        +bias: Adds a valence bias to the model (e.g. Hybrid2012+bias), only usable with Hybrid2012, and Hybrid2021
        +novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models
        +decay: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models
    '''

    models = ['StandardHybrid2012+bias+novel', #Standard
              'StandardHybrid2021+bias+decay+novel'] #Standard

    fixed, _ = get_priors()
    bounds = {'StandardHybrid2012':     {'temperature': (0.1, .2),
                                         'valence_factor': (0.5, 0.5)},
              'StandardHybrid2021':     {'temperature': (0.1, .2),
                                         'noise_factor': (0, 0),
                                         'valence_factor': (0.5, 0.5)}}
    
    generate_params = {'learning_filename':         'SOMA_RL/data/pain_learning_processed.csv',
                       'transfer_filename':         'SOMA_RL/data/pain_transfer_processed.csv',
                       'models':                    models,
                       'parameters':                'random',
                       'fixed':                     fixed,
                       'bounds':                    bounds,
                       'number_of_runs':            10,
                       'multiprocessing':           True,
                       'number_of_participants':    0,
                       'training':                  'scipy',
                       }

    run_recovery(**generate_params, recovery='parameter')
    #run_recovery(**generate_params, recovery='model')
    export_recovery(path="SOMA_RL/reports")
