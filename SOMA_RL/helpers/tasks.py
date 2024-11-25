import random as rnd
import pandas as pd

class AvoidanceLearningTask:

    def __init__(self):
        self.task = 'Avoidance Learning Task'

        self.stimuli_ids = [['A', 'B'], 
                            ['C', 'D'],
                            ['E', 'F'],
                            ['G', 'H']]
        
        self.stimuli_context = ['Reward',
                                'Reward',
                                'Loss Avoid',
                                'Loss Avoid']
        
        self.stimuli_feedback = [1, 1, -1, -1]
        
        self.stimuli_probabilities = [[.75, .25], [.25, .75]]
    
    def initiate_model(self, rl_model):

        #Initialize model
        self.rl_model = rl_model

        #Populate model with data matrices
        self.rl_model.task_learning_data_columns = ['block_number', 'trial_number', 'state_index', 
                                           'state_id', 'stim_id', 'context', 'feedback', 
                                           'probabilities', 'rewards', 'q_values', 'action', 
                                           'prediction_errors', 'correct_action', 'accuracy']
        
        self.rl_model.task_transfer_data_columns = ['block_number', 'trial_number', 'state_id',
                                                    'stim_id', 'q_values', 'action']
        
        self.rl_model.create_matrices(states=['State AB', 'State CD', 'State EF', 'State GH'],
                                      number_actions=2)
        
    def run_learning_phase(self, task_design):

        number_of_trials = task_design['learning_phase']['number_of_trials']
        number_of_blocks = task_design['learning_phase']['number_of_blocks']

        for block in range(number_of_blocks):

            #Determine trial order
            trial_order =[0, 1, 2, 3]*int(number_of_trials/4)
            rnd.shuffle(trial_order)

            for trial in range(number_of_trials):

                #Select stimuli
                stimuli_index = trial_order[trial]
                stimuli_id = self.stimuli_ids[stimuli_index]
                stimuli_context = self.stimuli_context[stimuli_index]
                context_index = 0 if stimuli_context == 'Reward' else 1
                stimuli_feedback = self.stimuli_feedback[stimuli_index]
                stimuli_probabilities = self.stimuli_probabilities[context_index]
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
                self.rl_model.run_trial(state, phase='learning')
    
    def run_transfer_phase(self, task_design):

        #Get arguments
        times_repeated = task_design['transfer_phase']['times_repeated']
        
        #Setup pairs
        self.rl_model.transfer_stimuli = ['A','B','C','D','E','F','G','H','N'] #N is a novel stimulus
        exclusion_pairs = [['A','C'], ['B','D'], ['E','G'], ['F','H']]
        self.transfer_pairs = [[self.rl_model.transfer_stimuli[i], self.rl_model.transfer_stimuli[j]] 
                               for i in range(len(self.rl_model.transfer_stimuli)) for j in range(i+1, len(self.rl_model.transfer_stimuli)) 
                               if [self.rl_model.transfer_stimuli[i], self.rl_model.transfer_stimuli[j]] not in exclusion_pairs]
        self.transfer_pairs = self.transfer_pairs * times_repeated
        rnd.shuffle(self.transfer_pairs)

        #Setup q-values
        self.rl_model.combine_q_values()

        #Run transfer phase
        for trial, pair in enumerate(self.transfer_pairs):

            #Select stimuli
            stimuli_id = pair
            
            state = {'block_number': 0,
                     'trial_number': trial,
                     'state_id': f'State {"".join(stimuli_id)}',
                     'stim_id': stimuli_id}
        
            #Run model
            self.rl_model.run_trial(state, phase='transfer')

    def run_experiment(self, task_design = {'learning_phase': {'number_of_trials': 100, 'number_of_blocks': 4},
                                             'transfer_phase': {'times_repeated': 4}}):
        self.run_learning_phase(task_design)
        self.run_transfer_phase(task_design)