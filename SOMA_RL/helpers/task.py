import random as rnd
import pandas as pd

class AvoidanceLearningTask:

    def __init__(self, rl_model):

        self.rl_model = rl_model

        self.stimuli_ids = [['A', 'B'], 
                            ['C', 'D'],
                            ['E', 'F'],
                            ['G', 'H']]
        
        self.stimuli_context = ['Reward',
                                'Reward',
                                'Loss Avoid',
                                'Loss Avoid']
        
        self.stimuli_feedback = [1, 1, -1, -1]
        
        self.stimuli_probabilities = [.25, .75]

        #Populate model with data matrices
        self.rl_model.task_data_columns = ['block_number', 'trial_number', 'state_index', 
                                           'state_id', 'stim_id', 'context', 'feedback', 
                                           'probabilities', 'rewards', 'q_values', 'action', 
                                           'prediction_errors', 'correct_action', 'accuracy']
        
        self.rl_model.create_matrices(states=['State AB', 'State CD', 'State EF', 'State GH'],
                                      number_actions=2)
        
    def run_learning_phase(self, trial_design):

        number_of_trials = trial_design['learning_phase']['number_of_trials']
        number_of_blocks = trial_design['learning_phase']['number_of_blocks']

        for block in range(number_of_blocks):

            #Determine trial order
            trial_order =[0, 1, 2, 3]*int(number_of_trials/4)
            rnd.shuffle(trial_order)

            for trial in range(number_of_trials):

                #Select stimuli
                stimuli_index = trial_order[trial]
                stimuli_id = self.stimuli_ids[stimuli_index]
                stimuli_context = self.stimuli_context[stimuli_index]
                stimuli_feedback = self.stimuli_feedback[stimuli_index]
                stimuli_probabilities = self.stimuli_probabilities
                if stimuli_context == 'Reward':
                    correct_action = stimuli_probabilities.index(max(stimuli_probabilities))
                else:
                    correct_action = stimuli_probabilities.index(min(stimuli_probabilities))
                state = {'block_number': block,
                         'trial_number': trial,
                         'state_index': stimuli_index, 
                         'state_id': f'State {"".join(stimuli_id)}',
                         'stim_id': stimuli_id,
                         'context': stimuli_context,
                         'feedback': stimuli_feedback,
                         'probabilities': stimuli_probabilities,
                         'correct_action': correct_action}

                #Run model
                self.rl_model.run_trial(state)
    
    def run_transfer_phase(self):
        pass

    def run_experiment(self, trial_design = {'learning_phase': {'number_of_trials': 100, 'number_of_blocks': 4}}):
        self.run_learning_phase(trial_design)
        self.run_transfer_phase()