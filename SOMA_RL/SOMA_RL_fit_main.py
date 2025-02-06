import random as rnd

from helpers.analyses import run_fit_empirical

if __name__ == "__main__":

    # =========================================== #
    # ================= INPUTS ================== #
    # =========================================== #

    #Seed random number generator
    rnd.seed(1251)

    #Debug parameters
    number_of_participants = 0 #Number of participants to keep, 0 = all

    #Parameters
    multiprocessing = True #Whether to run fits and simulations in parallel
    random_params = True #Whether to use random initial parameters for the models
    number_of_runs = 5 #Number of times to fit the dataset for each participant and model (this is especially important when random_params=True). For these data/models 5 was quantitatively determined as best.

    #File names
    learning_filename = 'SOMA_RL/data/pain_learning_processed.csv'
    transfer_filename = 'SOMA_RL/data/pain_transfer_processed.csv'

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
                
    # =========================================== #
    # ============== RUN ANALYSES =============== #
    # =========================================== #

    run_fit_empirical(learning_filename, 
                      transfer_filename, 
                      models, 
                      number_of_participants=number_of_participants, 
                      random_params=random_params, 
                      number_of_runs=number_of_runs, 
                      multiprocessing=multiprocessing)
    print('done')
            