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
        self.task_data = pd.DataFrame(columns=self.task_data_columns)

    def get_q_value(self, state):
        state['q_values'] = list(self.q_values[state['state_id']].iloc[-1])
        return state
    
    def update_model(self, state):
        self.update_q_values(state)
        self.update_prediction_errors(state)
        self.update_task_data(state)

    def update_task_data(self, state):
        self.task_data = pd.concat([self.task_data, 
                                    pd.DataFrame([[state[col_name] for col_name in self.task_data_columns]], 
                                    columns=self.task_data_columns)], 
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
    
        fig, ax = plt.subplots(2, 4, figsize=(10,5))
        for i, key in enumerate(self.q_values.keys()):
            self.q_values[key][1:].plot(ax=ax[0,i])
            ax[0,i].set_title(key)
            ax[0,i].set_ylim(-1, 1)
            ax[0,i].set_ylabel('Q-Value')
            ax[0,i].set_xlabel('Trial')
        
        for i, key in enumerate(self.prediction_errors.keys()):
            self.prediction_errors[key][1:].plot(ax=ax[1,i])
            ax[1,i].set_title(key)
            ax[1,i].set_ylim(-1, 1)
            ax[1,i].set_ylabel('Prediction Error')
            ax[1,i].set_xlabel('Trial')
        
        plt.show()

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

    def run_trial(self, state):
        state = self.determine_reward(state)
        state = self.get_q_value(state)
        state = self.select_action(state)
        state = self.compute_prediction_error(state)
        self.update_model(state)



