import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RLToolbox:

    def get_q_value(self, state):
        state['q_values'] = list(self.q_values[state['state_id']].iloc[-1])
        return state

    def update_q_values(self, state):

        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]

        new_q_values = []
        for i in range(len(state['rewards'])):
            new_q_values.append(state['q_values'][i] + (learning_rates[i] * state['prediction_errors'][i]))
        new_q_values = pd.DataFrame([new_q_values], columns=['Q1', 'Q2'])

        self.q_values[state['state_id']] = pd.concat([self.q_values[state['state_id']], new_q_values], ignore_index=True)

    def plot_q_values(self):
    
        fig, ax = plt.subplots(1, 4, figsize=(10,10))
        for i, key in enumerate(self.q_values.keys()):
            self.q_values[key].plot(ax=ax[i])
            ax[i].set_title(key)
            ax[i].set_ylim(-1, 1)
            ax[i].set_ylabel('Q-Value')
            ax[i].set_xlabel('Trial')
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
        #TODO: So far this is just greedy selection
        max_index = np.argwhere(state['q_values'] == np.amax(state['q_values']))
        state['action'] = rnd.choice(max_index)[0]

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
        self.update_q_values(state)



