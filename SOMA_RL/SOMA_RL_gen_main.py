import random as rnd
import pandas as pd

from helpers.analyses import generate_simulated_data

if __name__ == "__main__":

    # =========================================== #
    # ================= INPUTS ================== #
    # =========================================== #

    #Seed random number generator
    rnd.seed(1251)

    #Parameters
    multiprocessing = True #Whether to run fits and simulations in parallel

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
        wRelative+decay: Standard Weighted-Relative Model [Proposed model] (Williams et al., in prep)
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
    
    QLearning_params = {"factual_lr": 0.1,"counterfactual_lr": 0.5,"temperature": 0.1,"novel_value": 0,"decay_factor": 0,}
    ActorCritic_params = {"factual_actor_lr": 0.1,"counterfactual_actor_lr": 0.05,"critic_lr": 0.1,"temperature": 0.1,"novel_value": 0,"decay_factor": 0,}
    Relative_params = {"factual_lr": 0.1,"counterfactual_lr": 0.05,"contextual_lr": 0.1,"temperature": 0.1,"novel_value": 0,"decay_factor": 0,}
    wRelative_params = {"factual_lr": 0.1,"counterfactual_lr": 0.05,"contextual_lr": 0.1,"temperature": 0.1,"mixing_factor": 0.5,"valence_factor": 0.5,"novel_value": 0,"decay_factor": 0,}
    Hybrid2012_params = {"factual_lr": 0.1,"counterfactual_lr": 0.05,"factual_actor_lr": 0.1,"counterfactual_actor_lr": 0.05,"critic_lr": 0.1,"temperature": 0.1,"mixing_factor": 0.5,"valence_factor": 0.5,"novel_value": 0,"decay_factor": 0,}
    Hybrid2021_params = {"factual_lr": 0.1,"counterfactual_lr": 0.05,"factual_actor_lr": 0.1,"counterfactual_actor_lr": 0.05,"critic_lr": 0.1,"temperature": 0.1,"mixing_factor": 0.5,"noise_factor":0.1,"valence_factor": 0.5,"novel_value": 0,"decay_factor": 0,}

    parameters = {'QLearning': QLearning_params,
                  'QLearning+novel': QLearning_params,
                  'ActorCritic': ActorCritic_params,
                  'ActorCritic+novel': ActorCritic_params,

                  'Relative': Relative_params,
                  'Relative+novel': Relative_params,
                  'wRelative+bias+decay': wRelative_params,
                  'wRelative+bias+decay+novel': wRelative_params,

                  'Hybrid2012+bias': Hybrid2012_params,
                  'Hybrid2012+bias+novel': Hybrid2012_params,
                  'Hybrid2021+bias+decay': Hybrid2021_params,
                  'Hybrid2021+bias+decay+novel': Hybrid2021_params}

    #Must convert each subdict into pandas dataframe
    for model in parameters:
        parameters[model] = pd.DataFrame(parameters[model], index=[0])

    #Task design
    task_design = {'learning_phase': {
                        'number_of_trials': 24,
                        'number_of_blocks': 4,},
                    'transfer_phase': {
                        'times_repeated': 4}}

    # =========================================== #
    # ============== RUN ANALYSES =============== #
    # =========================================== #

    generate_simulated_data(models, parameters, task_design)
    print('Done!')
            