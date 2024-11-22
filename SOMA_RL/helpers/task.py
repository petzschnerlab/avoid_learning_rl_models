import random as rnd
import pandas as pd

class AvoidanceLearningTask:

    def __init__(self, rl_model):

        self.rl_model = rl_model

        self.stimuli_ids = [['A', 'B'], 
                            ['C', 'D'],
                            ['E', 'F'],
                            ['G', 'H']]
        
        self.stimuli_context = [['Reward'],
                                ['Reward'],
                                ['Loss Avoid'],
                                ['Loss Avoid']]
        
        self.stimuli_feedback = [1, 1, -1, -1]
        
        self.stimuli_probabilities = [.25, .75]

        self.rl_model.q_values = {f'State AB': pd.DataFrame([[0,0]], columns=['Q1', 'Q2']),
                             f'State CD': pd.DataFrame([[0,0]], columns=['Q1', 'Q2']),
                             f'State EF': pd.DataFrame([[0,0]], columns=['Q1', 'Q2']),
                             f'State GH': pd.DataFrame([[0,0]], columns=['Q1', 'Q2'])}
    
    def run_experiment(self, number_of_trials = 100, number_of_blocks = 4):


        for block in range(number_of_blocks):

            for trial in range(number_of_trials):

                #Select stimuli
                stimuli_index = rnd.randint(0, len(self.stimuli_ids)-1)

                stimuli_id = self.stimuli_ids[stimuli_index]
                stimuli_context = self.stimuli_context[stimuli_index]
                stimuli_feedback = self.stimuli_feedback[stimuli_index]
                stimuli_probabilities = self.stimuli_probabilities
                state = {'state_index': stimuli_index, 
                         'state_id': f'State {stimuli_id[0]}{stimuli_id[1]}',
                         'stim_id': stimuli_id, 
                         'context': stimuli_context, 
                         'feedback': stimuli_feedback,
                         'probabilities': stimuli_probabilities,}

                #Run model
                self.rl_model.run_trial(state)
