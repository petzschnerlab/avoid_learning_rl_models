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
        self.rl_model.q_values = {f'State AB': pd.DataFrame([[0,0]], columns=['Q1', 'Q2']),
                             f'State CD': pd.DataFrame([[0,0]], columns=['Q1', 'Q2']),
                             f'State EF': pd.DataFrame([[0,0]], columns=['Q1', 'Q2']),
                             f'State GH': pd.DataFrame([[0,0]], columns=['Q1', 'Q2'])}
        
        self.rl_model.prediction_errors = {f'State AB': pd.DataFrame([[0,0]], columns=['PE1', 'PE2']),
                        f'State CD': pd.DataFrame([[0,0]], columns=['PE1', 'PE2']),
                        f'State EF': pd.DataFrame([[0,0]], columns=['PE1', 'PE2']),
                        f'State GH': pd.DataFrame([[0,0]], columns=['PE1', 'PE2'])}
        
        self.rl_model.task_data_columns = ['block_number', 'trial_number', 'state_index', 
                                           'state_id', 'stim_id', 'context', 'feedback', 
                                           'probabilities', 'rewards', 'q_values', 'action', 
                                           'prediction_errors']
        self.rl_model.task_data = pd.DataFrame(columns=self.rl_model.task_data_columns)

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
                state = {'block_number': block,
                         'trial_number': trial,
                         'state_index': stimuli_index, 
                         'state_id': f'State {"".join(stimuli_id)}',
                         'stim_id': stimuli_id,
                         'context': stimuli_context,
                         'feedback': stimuli_feedback,
                         'probabilities': stimuli_probabilities}

                #Run model
                self.rl_model.run_trial(state)
    
    def run_transfer_phase(self):
        pass

    def run_experiment(self, 
                       trial_design = {'learning_phase': {'number_of_trials': 100, 'number_of_blocks': 4}}):
        self.run_learning_phase(trial_design)
        self.run_transfer_phase()