import random as rnd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

class RLToolbox:

    def create_matrices(self, states, number_actions):
        self.q_values = {state: pd.DataFrame([[0]*number_actions], columns=[f'Q{i+1}' for i in range(number_actions)]) for state in states}
        self.prediction_errors = {state: pd.DataFrame([[0]*number_actions], columns=[f'PE{i+1}' for i in range(number_actions)]) for state in states}
        self.task_learning_data = pd.DataFrame(columns=self.task_learning_data_columns)
        self.task_transfer_data = pd.DataFrame(columns=self.task_transfer_data_columns)

    def get_q_value(self, state):
        state['q_values'] = list(self.q_values[state['state_id']].iloc[-1])
        return state

    def get_final_q_values(self, state):
        state['q_values'] = [float(self.final_q_values[stim]) for stim in state['stim_id']]
        return state
    
    def update_model(self, state, phase='learning'):
        if phase == 'learning':
            self.update_q_values(state)
            self.update_prediction_errors(state)
        self.update_task_data(state, phase=phase)

    def update_task_data(self, state, phase='learning'):
        if phase == 'learning':
            self.task_learning_data = pd.concat([self.task_learning_data, 
                                        pd.DataFrame([[state[col_name] for col_name in self.task_learning_data_columns]], 
                                                     columns=self.task_learning_data_columns)], 
                                        ignore_index=True)
        else:
            self.task_transfer_data = pd.concat([self.task_transfer_data, 
                                        pd.DataFrame([[state[col_name] for col_name in self.task_transfer_data_columns]], 
                                                     columns=self.task_transfer_data_columns)],
                                        ignore_index=True)

    def update_prediction_errors(self, state):
        self.prediction_errors[state['state_id']] = pd.concat([self.prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['prediction_errors']], columns=['PE1', 'PE2'])], 
                                                               ignore_index=True)

    def update_q_values(self, state):

        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]

        new_q_values = []
        for i in range(len(state['rewards'])):
            new_q_values.append(state['q_values'][i] + (learning_rates[i] * state['prediction_errors'][i]))
        new_q_values = pd.DataFrame([new_q_values], columns=['Q1', 'Q2'])

        self.q_values[state['state_id']] = pd.concat([self.q_values[state['state_id']], new_q_values], ignore_index=True)

    def plot_progress(self):
    
        fig, ax = plt.subplots(4, 4, figsize=(20,5))

        #Plot q-values
        for i, key in enumerate(self.q_values.keys()):
            for ci, col in enumerate(self.q_values[key].columns):
                rolling_q_values = self.q_values[key][col].reset_index(drop=True).rolling(window=2).mean()
                ax[0,i].plot(rolling_q_values, label=['Stim 1', 'Stim 2'][ci])
            ax[0,i].set_title(key)
            ax[0,i].set_ylim(-1, 1)
            if i == 0:
                ax[0,i].set_ylabel('Q-Value')
            ax[0,i].set_xlabel('')
            if i == len(self.q_values.keys())-1:
                ax[0,i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax[0,i].set_xticklabels([])
            if i > 0:
                ax[0,i].set_yticklabels([])
        
        #Plot prediction errors
        for i, key in enumerate(self.prediction_errors.keys()):
            for ci, col in enumerate(self.prediction_errors[key].columns):
                rolling_prediction_errors = self.prediction_errors[key][col].reset_index(drop=True).rolling(window=2).mean()
                ax[1,i].plot(rolling_prediction_errors, label=['Stim 1', 'Stim 2'][ci])
            ax[1,i].set_title('')
            ax[1,i].set_ylim(-1, 1)
            if i == 0:
                ax[1,i].set_ylabel('Prediction Error')
            ax[1,i].set_xlabel('')
            ax[1,i].set_xticklabels([])
            if i > 0:
                ax[1,i].set_yticklabels([])

        #Plot accuracy
        for i, key in enumerate(self.accuracy.keys()):
            rolling_accuracy = self.accuracy[key].reset_index(drop=True).rolling(window=2).mean()
            ax[2,i].plot(rolling_accuracy)
            ax[2,i].set_title('')
            ax[2,i].set_ylim(-.05, 1.05)
            if i == 0:
                ax[2,i].set_ylabel('Accuracy')
            ax[2,i].set_xlabel('Trial')
            if i > 0:
                ax[2,i].set_yticklabels([])

        x_tick_labels = ['75R', '25R', '25P', '75P', 'N']
        ax[3,0].bar(self.choice_rate.keys(), self.choice_rate.values())
        ax[3,0].set_ylabel('Choice Rate (%)')
        ax[3,0].set_xlabel('Stimulus')
        ax[3,0].set_xticks(range(len(self.choice_rate.keys())))
        ax[3,0].set_xticklabels(x_tick_labels)

        plt.show()
        
    def run_computations(self):
        self.compute_accuracy()
        self.compute_choice_rate()

    def compute_accuracy(self):

        self.accuracy = {}
        for state in self.task_learning_data['state_id'].unique():
            state_data = self.task_learning_data[self.task_learning_data['state_id'] == state]
            self.accuracy[state] = state_data['accuracy']

    def compute_choice_rate(self):
    
        choice_rate = {}
        for stimulus in self.transfer_stimuli:
            #find rows where stim_id contains stimulus
            stimulus_data = self.task_transfer_data.loc[self.task_transfer_data['stim_id'].apply(lambda x: stimulus in x)]
            #find whether stimulus is action 0 or 1
            stimulus_data.loc[:,'stim_index'] = stimulus_data['stim_id'].apply(lambda x: 0 if stimulus == x[0] else 1)
            stimulus_data.loc[:,'stim_chosen'] = stimulus_data.apply(lambda x: int(x['action'] == x['stim_index']), axis=1)
            choice_rate[stimulus] = int((stimulus_data['stim_chosen'].sum()/len(stimulus_data))*100)

        #Average column pairs:
        pairs = [['A','C'],['B','D'],['E','G'],['F','H']]
        self.choice_rate = {}
        for pair in pairs:
            self.choice_rate[pair[0]] = (choice_rate[pair[0]] + choice_rate[pair[1]])/2
        self.choice_rate['N'] = choice_rate['N']

    def combine_q_values(self):

        stimuli = ['A','B','C','D','E','F','G','H']
        self.q_values_summary = pd.concat([self.q_values[state] for state in self.q_values.keys()], axis=1)
        self.q_values_summary.columns = stimuli
        self.final_q_values = self.q_values_summary.iloc[-1]
        self.final_q_values['N'] = 0

#Q-Learning Model
class QLearning(RLToolbox):

    """
    Reinforcement Learning Model: Q-Learning

    Parameters:
    ------------
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    """

    def __init__(self, factual_lr, counterfactual_lr, temperature):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.temperature = temperature   

    def select_action(self, state):

        transformed_q_values = np.exp(np.divide(state['q_values'], self.temperature))
        probability_q_values = (transformed_q_values/np.sum(transformed_q_values)).cumsum()
        state['action'] = np.where(probability_q_values >= rnd.random())[0][0]
        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state
    
    def determine_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(rewards, state):
        state['prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        return state

    def run_trial(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.determine_reward(state)
            state = self.get_q_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state, phase=phase)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
            self.update_model(state, phase=phase)




