import random as rnd
import numpy as np
import pandas as pd
import torch 


class AvoidanceLearningTask:

    def __init__(self, task_design: dict = None) -> None:

        """
        Initialize the Avoidance Learning Task with a specific design.

        Parameters
        ----------
        task_design : dict, optional
            A dictionary containing the design of the task, including parameters for learning and transfer phases.

        Returns
        -------
        None
        """

        self.task = 'Avoidance Learning Task'
        self.task_design = task_design

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
            
    def create_model_lists(self, states: list, learning_dimensions: list, transfer_dimensions: list) -> None:

        """
        Create and initialize lists for the reinforcement learning model to store various values.

        Parameters
        ----------
        states : list
            A list of state identifiers for the task.
        learning_dimensions : list
            A list containing the number of learning trials and the number of actions.
        transfer_dimensions : list
            A list containing the number of transfer trials and the number of actions.

        Returns
        -------
        None
        """

        self.rl_model.number_state_trials = learning_dimensions[0]
        self.rl_model.q_values = {state: [0]*learning_dimensions[1] for state in states}
        self.rl_model.prediction_errors = {state: [0]*learning_dimensions[1] for state in states}
        self.rl_model.task_learning_data = pd.DataFrame(np.zeros((learning_dimensions[0]*len(states), len(self.task_learning_data_columns))), columns=self.task_learning_data_columns)
        self.rl_model.task_transfer_data = pd.DataFrame(np.zeros((transfer_dimensions[0], len(self.task_transfer_data_columns))), columns=self.task_transfer_data_columns)

        self.rl_model.initial_q_values = pd.DataFrame([self.rl_model.q_values[self.rl_model.states[0]][0]]*9).T
        self.rl_model.initial_q_values.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']
        
        if self.rl_model.__class__.__name__ == 'Relative':
            self.rl_model.context_values = {state: [0] for state in states}
            self.rl_model.context_prediction_errors = {state: [0] for state in states}

        if self.rl_model.__class__.__name__ == 'ActorCritic':
            self.rl_model.w_values = {state: [0.01]*learning_dimensions[1] for state in states}
            self.rl_model.v_values = {state: [0] for state in states}
            delattr(self.rl_model, 'q_values')
            delattr(self.rl_model, 'initial_q_values')

            self.rl_model.initial_w_values = pd.DataFrame([self.rl_model.w_values[self.rl_model.states[0]][0]]*9).T
            self.rl_model.initial_w_values.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']
        
        if 'Hybrid' in self.rl_model.__class__.__name__:
            self.rl_model.w_values = {state: [0.01]*learning_dimensions[1] for state in states}
            self.rl_model.v_values = {state: [0] for state in states}
            self.rl_model.h_values = {state: [0]*learning_dimensions[1] for state in states}
            self.rl_model.q_prediction_errors = {state: [0]*learning_dimensions[1] for state in states}
            self.rl_model.v_prediction_errors = {state: [0]*learning_dimensions[1] for state in states}
            delattr(self.rl_model, 'prediction_errors')

            self.rl_model.initial_w_values = pd.DataFrame([self.rl_model.w_values[self.rl_model.states[0]][0]]*9).T
            self.rl_model.initial_w_values.columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']
    
    def update_task_data(self, state: dict, phase: str = 'learning') -> None:

        """
        Update the task data with the current state information.

        Parameters
        ----------
        state : dict
            A dictionary containing the current state information, including block number, trial number, state ID, stimulus ID, context, feedback, probabilities, rewards, action, and accuracy.
        phase : str, optional
            The phase of the task, either 'learning' or 'transfer'. Default is 'learning'.

        Returns
        -------
        None
        """
        
        if phase == 'learning':
            n = self.rl_model.task_counts['learning']
            series = pd.Series({col_name: state[col_name] for col_name in self.task_learning_data_columns})
            if n == 0:
                self.rl_model.task_learning_data = self.rl_model.task_learning_data.astype(series.dtypes)
            self.rl_model.task_learning_data.iloc[n] = series
            self.rl_model.task_counts['learning'] += 1
        else:
            series = pd.Series({col_name: state[col_name] for col_name in self.task_transfer_data_columns})
            n = self.rl_model.task_counts['transfer']
            if n == 0:
                self.rl_model.task_transfer_data = self.rl_model.task_transfer_data.astype(series.dtypes)
            self.rl_model.task_transfer_data.iloc[n] = series
            self.rl_model.task_counts['transfer'] += 1

    def compute_accuracy(self) -> None:

        """
        Compute the accuracy of the model based on the task learning data.
        This method calculates the accuracy for each state in the task learning data and stores it in the model's accuracy attribute.
        The accuracy is computed as the proportion of correct actions taken in each state.
        The accuracy is stored in a dictionary where the keys are state IDs and the values are lists of accuracy values for each trial in that state.

        Returns
        -------
        None
        """

        self.rl_model.accuracy = {}
        for state in self.rl_model.task_learning_data['state_id'].unique():
            state_data = self.rl_model.task_learning_data[self.rl_model.task_learning_data['state_id'] == state]
            self.rl_model.accuracy[state] = state_data['accuracy']

    def compute_choice_rate(self) -> None:

        """
        Compute the choice rate for each stimulus in the task transfer data.
        This method calculates the percentage of times each stimulus was chosen in the transfer phase.
        The choice rate is stored in the model's choice_rate attribute, which is a dictionary where the keys are stimulus IDs and the values are the percentage of times each stimulus was chosen.
        The choice rate is computed by filtering the task transfer data for each stimulus, counting the number of times the stimulus was chosen, and dividing by the total number of trials for that stimulus.

        Returns
        -------
        None
        """
    
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
    
    def run_computations(self) -> None:
        """
        Run the computations for the reinforcement learning model.

        Returns
        -------
        None
        """

        self.compute_accuracy()
        self.compute_choice_rate()

    def combine_q_values(self) -> None:
        """
        Combine the Q-values from the reinforcement learning model into a final format.
        This method processes the Q-values stored in the model and formats them into a dictionary or DataFrame for easy access.

        Returns
        -------
        None
        """

        if self.rl_model.training == 'torch':
            final_q_values = {state.replace('State ', '')[i]: values[i].detach()
                            for state, values in self.rl_model.q_values.items() 
                            for i in range(2)}
            final_q_values['N'] = torch.tensor(0.) if self.rl_model.novel_value is None else self.rl_model.novel_value
            self.rl_model.final_q_values = final_q_values
        else:
            stimuli = ['A','B','C','D','E','F','G','H']
            self.rl_model.final_q_values = pd.DataFrame(np.array([self.rl_model.q_values[state] for state in self.rl_model.q_values.keys()]).flatten()).T
            self.rl_model.final_q_values.columns = stimuli
            self.rl_model.final_q_values['N'] = 0 if self.rl_model.novel_value is None else self.rl_model.novel_value

    def combine_v_values(self) -> None:
        """
        Combine the V-values from the reinforcement learning model into a final format.
        This method processes the V-values stored in the model and formats them into a dictionary or DataFrame for easy access.

        Returns
        -------
        None
        """

        if self.rl_model.training == 'torch':
            final_v_values = {state.replace('State ', '')[i]: values.detach()
                            for state, values in self.rl_model.v_values.items()
                            for i in range(2)}
            final_v_values['N'] = torch.tensor(0.) if self.rl_model.novel_value is None else self.rl_model.novel_value
            self.rl_model.final_v_values = final_v_values
        else:
            stimuli = ['A','B','C','D','E','F','G','H']
            v_array = np.array([[self.rl_model.v_values[state]]*2 for state in self.rl_model.v_values.keys()])
            self.rl_model.final_v_values = pd.DataFrame(v_array.flatten()).T
            self.rl_model.final_v_values.columns = stimuli
            self.rl_model.final_v_values['N'] = 0 if self.rl_model.novel_value is None else self.rl_model.novel_value

    def combine_w_values(self) -> None:
        
        """
        Combine the W-values from the reinforcement learning model into a final format.
        This method processes the W-values stored in the model and formats them into a dictionary or DataFrame for easy access.
        
        Returns
        -------
        None
        """

        if self.rl_model.training == 'torch':
            final_w_values = {state.replace('State ', '')[i]: values[i].detach()
                            for state, values in self.rl_model.w_values.items() 
                            for i in range(2)}
            final_w_values['N'] = torch.tensor(0.) if self.rl_model.novel_value is None else self.rl_model.novel_value
            self.rl_model.final_w_values = final_w_values
        else:
            stimuli = ['A','B','C','D','E','F','G','H']
            self.rl_model.final_w_values = pd.DataFrame(np.array([self.rl_model.w_values[state] for state in self.rl_model.w_values.keys()]).flatten()).T
            self.rl_model.final_w_values.columns = stimuli
            self.rl_model.final_w_values['N'] = 0 if self.rl_model.novel_value is None else self.rl_model.novel_value

    def combine_c_values(self) -> None:
        
        """
        Combine the C-values from the reinforcement learning model into a final format.
        This method processes the C-values stored in the model and formats them into a dictionary or DataFrame for easy access.
        
        Returns
        -------
        None
        """
        
        if self.rl_model.training == 'torch':
            final_c_values = {state.replace('State ', '')[i]: values[i].detach()
                            for state, values in self.rl_model.c_values.items() 
                            for i in range(2)}
            final_c_values['N'] = torch.tensor(0.) if self.rl_model.novel_value is None else self.rl_model.novel_value
            self.rl_model.final_c_values = final_c_values
        else:
            stimuli = ['A','B','C','D','E','F','G','H']
            self.rl_model.final_c_values = pd.DataFrame(np.array([self.rl_model.c_values[state] for state in self.rl_model.c_values.keys()]).flatten()).T
            self.rl_model.final_c_values.columns = stimuli
            self.rl_model.final_c_values['N'] = 0 if self.rl_model.novel_value is None else self.rl_model.novel_value

    def initiate_model(self, rl_model: object) -> None:

        """
        Initialize the reinforcement learning model with the task design and set up necessary parameters.
        
        Parameters
        ----------
        rl_model : object
            An instance of a reinforcement learning model that will be used for the task.
        
        Returns
        -------
        None
        """

        #Initialize model, create dataframes, and load methods
        states = ['State AB', 'State CD', 'State EF', 'State GH']
        self.rl_model = rl_model     
        self.rl_model.states = states
        number_of_learning_trials = (self.task_design['learning_phase']['number_of_trials'] * self.task_design['learning_phase']['number_of_blocks'])//len(states)
        num_stim = (len(self.stimuli_ids)*2)+1
        num_pairs = num_stim*(num_stim-1)//2 - len(self.stimuli_ids)
        if 'number_of_trials' in self.task_design['transfer_phase']:
            number_of_transfer_trials = self.task_design['transfer_phase']['number_of_trials']
            self.rl_model.transfer_stimuli = ['A','B','C','D','E','F','G','H','N'] #N is a novel stimulus
        else:
            number_of_transfer_trials = num_pairs * self.task_design['transfer_phase']['times_repeated']
        self.rl_model.task_counts = {phase: 0 for phase in ['learning', 'transfer']}
        
        methods = {
            'update_task_data': self.update_task_data,
            'compute_accuracy': self.compute_accuracy,
            'compute_choice_rate': self.compute_choice_rate,
            'run_computations': self.run_computations,
            'combine_q_values': self.combine_q_values,
            'combine_v_values': self.combine_v_values,
            'combine_w_values': self.combine_w_values,
            'combine_c_values': self.combine_c_values
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
            
        self.create_model_lists(states=states,
                                learning_dimensions=[number_of_learning_trials, 2],
                                transfer_dimensions=[number_of_transfer_trials, 2])

    def run_learning_phase(self, task_design: dict) -> None:

        """
        Run the learning phase of the avoidance learning task.
        This method iterates through the specified number of blocks and trials, selecting stimuli based on a randomized order.
        It updates the state information for each trial and runs the reinforcement learning model's forward method.
        
        Parameters
        ----------
        task_design : dict
            A dictionary containing the design of the learning phase, including the number of trials and blocks.
        
        Returns
        -------
        None
        """

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
    
    def run_transfer_phase(self, task_design: dict) -> None:

        """
        Run the transfer phase of the avoidance learning task.
        This method sets up the transfer pairs based on the task design and runs the reinforcement learning model's forward method for each pair.

        Parameters
        ----------
        task_design : dict
            A dictionary containing the design of the transfer phase, including the number of times to repeat the transfer pairs.
        
        Returns
        -------
        None
        """

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
        self.rl_model.combine_values()

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

    def run_experiment(self) -> None:

        """
        Run the entire avoidance learning task experiment, including both learning and transfer phases.
        
        Returns
        -------
        None
        """

        self.run_learning_phase(self.task_design)
        self.run_transfer_phase(self.task_design)