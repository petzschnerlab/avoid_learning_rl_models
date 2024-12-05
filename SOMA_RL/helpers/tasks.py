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

        #Populate model with data matrices
        self.task_learning_data_columns = ['block_number', 'trial_number', 'state_index', 
                                           'state_id', 'stim_id', 'context', 'feedback', 
                                           'probabilities', 'rewards', 'q_values', 'action', 
                                           'prediction_errors', 'correct_action', 'accuracy']
        
        self.task_transfer_data_columns = ['block_number', 'trial_number', 'state_id',
                                                    'stim_id', 'q_values', 'action']
    
    def create_model_matrices(self, states, number_actions):
        self.rl_model.q_values = {state: pd.DataFrame([[0]*number_actions], columns=[f'Q{i+1}' for i in range(number_actions)]) for state in states}
        self.rl_model.prediction_errors = {state: pd.DataFrame([[0]*number_actions], columns=[f'PE{i+1}' for i in range(number_actions)]) for state in states}
        self.rl_model.task_learning_data = pd.DataFrame(columns=self.task_learning_data_columns)
        self.rl_model.task_transfer_data = pd.DataFrame(columns=self.task_transfer_data_columns)

        if self.rl_model.__class__.__name__ == 'Relative':
            self.rl_model.context_values = {state: pd.DataFrame([[0]], columns=['C']) for state in states}
            self.rl_model.context_prediction_errors = {state: pd.DataFrame([[0]], columns=['PE']) for state in states}

        if self.rl_model.__class__.__name__ == 'ActorCritic':
            self.rl_model.w_values = {state: pd.DataFrame([[0.01]*number_actions], columns=[f'Q{i+1}' for i in range(number_actions)]) for state in states}
            self.rl_model.v_values = {state: pd.DataFrame([[0]], columns=['V']) for state in states}
            delattr(self.rl_model, 'q_values')
        
        if 'Hybrid' in self.rl_model.__class__.__name__:
            self.rl_model.w_values = {state: pd.DataFrame([[0.01]*number_actions], columns=[f'Q{i+1}' for i in range(number_actions)]) for state in states}
            self.rl_model.v_values = {state: pd.DataFrame([[0]], columns=['V']) for state in states}
            self.rl_model.h_values = {state: pd.DataFrame([[0]*number_actions], columns=[f'Q{i+1}' for i in range(number_actions)]) for state in states}
            self.rl_model.q_prediction_errors = {state: pd.DataFrame([[0]*number_actions], columns=[f'PE{i+1}' for i in range(number_actions)]) for state in states}
            self.rl_model.v_prediction_errors = {state: pd.DataFrame([[0]*number_actions], columns=[f'PE{i+1}' for i in range(number_actions)]) for state in states}
            delattr(self.rl_model, 'prediction_errors')

    def update_task_data(self, state, phase='learning'):
        if phase == 'learning':
            self.rl_model.task_learning_data = pd.concat([self.rl_model.task_learning_data, 
                                        pd.DataFrame([[state[col_name] for col_name in self.task_learning_data_columns]], 
                                                     columns=self.task_learning_data_columns)], 
                                        ignore_index=True)
        else:
            self.rl_model.task_transfer_data = pd.concat([self.rl_model.task_transfer_data, 
                                        pd.DataFrame([[state[col_name] for col_name in self.task_transfer_data_columns]], 
                                                     columns=self.task_transfer_data_columns)],
                                        ignore_index=True)
            
    def compute_accuracy(self):

        self.rl_model.accuracy = {}
        for state in self.rl_model.task_learning_data['state_id'].unique():
            state_data = self.rl_model.task_learning_data[self.rl_model.task_learning_data['state_id'] == state]
            self.rl_model.accuracy[state] = state_data['accuracy']

    def compute_choice_rate(self):
    
        choice_rate = {}
        for stimulus in self.rl_model.transfer_stimuli:
            stimulus_data = self.rl_model.task_transfer_data.loc[self.rl_model.task_transfer_data['stim_id'].apply(lambda x: stimulus in x)].copy()
            stimulus_data.loc[:,'stim_index'] = stimulus_data.loc[:,'stim_id'].apply(lambda x: 0 if stimulus == x[0] else 1)
            stimulus_data['stim_chosen'] = stimulus_data.apply(lambda x: int(x['action'] == x['stim_index']), axis=1)
            choice_rate[stimulus] = int((stimulus_data['stim_chosen'].sum()/len(stimulus_data))*100)

        #Average column pairs:
        pairs = [['A','C'],['B','D'],['E','G'],['F','H']]
        self.rl_model.choice_rate = {}
        for pair in pairs:
            self.rl_model.choice_rate[pair[0]] = (choice_rate[pair[0]] + choice_rate[pair[1]])/2
        self.rl_model.choice_rate['N'] = choice_rate['N']
    
    def run_computations(self):
        self.compute_accuracy()
        self.compute_choice_rate()

    def combine_q_values(self):

        stimuli = ['A','B','C','D','E','F','G','H']
        self.rl_model.q_values_summary = pd.concat([self.rl_model.q_values[state] for state in self.rl_model.q_values.keys()], axis=1)
        self.rl_model.q_values_summary.columns = stimuli
        self.rl_model.final_q_values = self.rl_model.q_values_summary.iloc[-1].copy()
        self.rl_model.final_q_values['N'] = 0

        self.rl_model.initial_q_values = self.rl_model.q_values_summary.iloc[0].copy()
        self.rl_model.initial_q_values['N'] = 0

    def combine_v_values(self):

        stimuli = ['A','B','C','D','E','F','G','H']
        v_columns = [[self.rl_model.v_values[state], self.rl_model.v_values[state]] for state in self.rl_model.v_values.keys()] #Duplicate for compatibility
        v_columns = [item for sublist in v_columns for item in sublist]
        self.rl_model.v_values_summary = pd.concat(v_columns, axis=1)

        self.rl_model.v_values_summary.columns = stimuli
        self.rl_model.final_v_values = self.rl_model.v_values_summary.iloc[-1].copy()
        self.rl_model.final_v_values['N'] = 0

    def combine_w_values(self):

        stimuli = ['A','B','C','D','E','F','G','H']
        self.rl_model.w_values_summary = pd.concat([self.rl_model.w_values[state] for state in self.rl_model.w_values.keys()], axis=1)
        self.rl_model.w_values_summary.columns = stimuli
        self.rl_model.final_w_values = self.rl_model.w_values_summary.iloc[-1].copy()
        self.rl_model.final_w_values['N'] = 0

        self.rl_model.initial_w_values = self.rl_model.w_values_summary.iloc[0].copy()
        self.rl_model.initial_w_values['N'] = 0

    def initiate_model(self, rl_model):

        #Initialize model, create dataframes, and load methods
        self.rl_model = rl_model        
        self.create_model_matrices(states=['State AB', 'State CD', 'State EF', 'State GH'],
                                            number_actions=2)
        
        methods = {
            'update_task_data': self.update_task_data,
            'compute_accuracy': self.compute_accuracy,
            'compute_choice_rate': self.compute_choice_rate,
            'run_computations': self.run_computations,
            'combine_q_values': self.combine_q_values,
            'combine_v_values': self.combine_v_values,
            'combine_w_values': self.combine_w_values
        }
        self.rl_model.load_methods(methods)

        if self.rl_model.__class__.__name__ == 'Relative':
            self.task_learning_data_columns += ['context_value']
            self.task_learning_data_columns += ['context_prediction_errors']
        
        if self.rl_model.__class__.__name__ == 'ActorCritic':
            self.task_learning_data_columns += ['w_values']
            self.task_learning_data_columns.remove('q_values')
            self.task_learning_data_columns += ['v_values']
            self.task_transfer_data_columns.remove('q_values')
            self.task_transfer_data_columns += ['w_values']

        if 'Hybrid' in self.rl_model.__class__.__name__:
            self.task_learning_data_columns += ['v_values']
            self.task_learning_data_columns += ['w_values']
            self.task_learning_data_columns += ['h_values']
            self.task_learning_data_columns += ['q_prediction_errors']
            self.task_learning_data_columns += ['v_prediction_errors']
            self.task_learning_data_columns.remove('prediction_errors')
            self.task_transfer_data_columns += ['w_values']
        
    def run_learning_phase(self, task_design):

        number_of_trials = task_design['learning_phase']['number_of_trials']
        number_of_blocks = task_design['learning_phase']['number_of_blocks']

        for block in range(number_of_blocks):

            #Determine trial order
            trial_order = [0, 1, 2, 3]*int(number_of_trials/4)
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
                self.rl_model.forward(state, phase='learning')
    
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

        #Setup q-values & w-values
        
        if self.rl_model.__class__.__name__ == 'ActorCritic':
            self.rl_model.combine_v_values()
            self.rl_model.combine_w_values()
        elif 'Hybrid' in self.rl_model.__class__.__name__:
            self.rl_model.combine_q_values()
            self.rl_model.combine_v_values()
            self.rl_model.combine_w_values()
        else:
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
            self.rl_model.forward(state, phase='transfer')

    def run_experiment(self, task_design = {'learning_phase': {'number_of_trials': 100, 'number_of_blocks': 4},
                                             'transfer_phase': {'times_repeated': 4}}):
        self.run_learning_phase(task_design)
        self.run_transfer_phase(task_design)