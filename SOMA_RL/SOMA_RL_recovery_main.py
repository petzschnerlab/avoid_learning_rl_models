import os
import random as rnd
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

from helpers.analyses import run_recovery
from helpers.priors import get_priors

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
    
    models = ['QLearning', 'ActorCritic', 'Relative']

    fixed = {
        'QLearning': {  # From Palminteri et al., 2015
            'factual_lr': 0.28,
            'counterfactual_lr': 0.18,
            'temperature': 0.06,
            # From Geana et al., 2021:
            'decay_factor': 0.08,
            # Custom
            'novel_value': .50,
        },

        'ActorCritic': {  # From Geana et al., 2021's Hybrid2021 model
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': .06,
            # From Geana et al., 2021:
            'valence_factor': .33,
            'decay_factor': .08,
            # Custom
            'novel_value': .50,
        },

        'Relative': {  # From Palminteri et al., 2015
            'factual_lr': 0.19,
            'counterfactual_lr': 0.15,
            'context_lr': 0.33,
            'temperature': 0.05,
            # From Geana et al., 2021:
            'decay_factor': 0.08,
            # Custom
            'novel_value': .50,
        },

        'Hybrid2012': {  # From Geana et al., 2021:
            'factual_lr': 0.49,
            'counterfactual_lr': 0.49,
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': 0.06,
            'mixing_factor': 0.7,  # From Gold et al., 2012
            # From Geana et al., 2021:
            'valence_bias': 0.33,
            'decay_factor': 0.08,
            # Custom
            'novel_value': .50,
        },

        'Hybrid2021': {  # From Geana et al., 2021
            'factual_lr': 0.49,
            'counterfactual_lr': 0.49,
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': 0.06,
            'mixing_factor': 0.69,
            # From Geana et al., 2021:
            'valence_bias': 0.33,
            'decay_factor': 0.08,
            # Custom
            'novel_value': .50,
        },
    }

    fixed, _ = get_priors()
    bounds = {'QLearning':      {'temperature': (0.1, 1)},
              'ActorCritic':    {'temperature': (0.1, 1)},
              'Relative':       {'temperature': (0.1, 1)},
              'wRelative':      {'temperature': (0.1, 1)},
              'Hybrid2012':     {'temperature': (0.1, 1)},
              'Hybrid2021':     {'temperature': (0.1, 1)}}
    
    generate_params = {'learning_filename':         'SOMA_RL/data/pain_learning_processed.csv',
                       'transfer_filename':         'SOMA_RL/data/pain_transfer_processed.csv',
                       'models':                    models,
                       'parameters':                'normal',
                       'fixed':                     fixed,
                       'bounds':                    bounds,
                       'number_of_runs':            10,
                       'multiprocessing':           True,
                       'number_of_participants':    0,
                       }

    run_recovery(**generate_params, recovery='parameter')
    run_recovery(**generate_params, recovery='model')
