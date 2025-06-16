from helpers.pipeline import Pipeline
from helpers.priors import fixed_priors

if __name__ == "__main__":

    # =========================================== #
    # ================= INPUTS ================== #
    # =========================================== #

    #Models and Parameters
    '''
    Supported models: 
        QLearning, ActorCritic
        Relative, wRelative
        Hybrid2012, Hybrid2021

    Standard models:
        QLearning: Standard Q-Learning Model
        ActorCritic: Standard Actor-Critic Model
        Relative: Standard Relative Model (Palminteri et al., 2015)
        wRelative: Simplified Relative Model (Williams et al., ...)
        Hybrid2012+bias: Standard Hybrid 2012 Model (Gold et al., 2012)
        Hybrid2021+bias+decay: Standard Hybrid 2021 Model (Geana et al., 2021)

    Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign
        +bias: Adds a valence bias to the model (e.g. Hybrid2012+bias), only usable with Hybrid2012, and Hybrid2021
        +novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models
        +decay: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models
    '''

    models = ['QLearning+novel',    #Standard + novel
              'ActorCritic+novel',  #Standard + novel
              'Relative+novel',     #Standard + novel
              'wRelative+novel',    #Standard + novel
              'Hybrid2012+novel',   #Standard - bias + novel
    ]
        
    fixed = fixed_priors(models)    
    fit_params = {'mode':                       'fit',
                  'learning_filename':          'RL/data/pain_learning_processed.csv',
                  'transfer_filename':          'RL/data/pain_transfer_processed.csv',
                  'models':                     models,
                  'random_params':              'normal',
                  'fixed':                      fixed,
                  'number_of_runs':             10,
                  'multiprocessing':            True,
    }

    pipeline = Pipeline(seed=1251)
    pipeline.run(**fit_params)