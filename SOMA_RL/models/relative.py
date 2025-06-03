import numpy as np
import random as rnd
import torch
import torch.nn as nn

from .rl_toolbox import RLToolbox

class Relative(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Relative (Palminteri et al., 2015)

    Parameters:
    ------------
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    """

    def __init__(self, factual_lr, counterfactual_lr, contextual_lr, temperature, novel_value, decay_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.contextual_lr = contextual_lr
        self.temperature = temperature
        self.novel_value = novel_value
        self.decay_factor = decay_factor
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'contextual_lr': self.contextual_lr,
                           'temperature': self.temperature,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    #RL functions
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward
        state['context_reward'] = np.average(state['rewards'])

        return state
    
    def compute_prediction_error(self, state):
        if self.training == 'torch':
            state['prediction_errors'] = state['rewards'] - state['context_value'] - state['q_values'].detach()
            state['context_prediction_errors'] = state['context_reward'] - state['context_value'].detach()
        else:
            state['prediction_errors'] = [state['rewards'][i] - state['context_value'][0] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['context_prediction_errors'] = state['context_reward'] - state['context_value']
        return state
    
    def select_action(self, state):

        if self.training == 'torch':
            transformed_q_values = torch.exp(torch.div(state['q_values'], self.temperature))
            probability_q_values = torch.cumsum(transformed_q_values/torch.sum(transformed_q_values), dim=0)
            state['action'] = self.torch_select_action(probability_q_values)
        else:
            transformed_q_values = np.exp(np.divide(state['q_values'], self.temperature))
            probability_q_values = (transformed_q_values/np.sum(transformed_q_values)).cumsum()
            state['action'] = np.where(probability_q_values >= rnd.random())[0][0]
            
        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state          

    #Run trial functions
    def forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_reward(state)
            state = self.get_q_value(state)
            state = self.get_context_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_model_update(self, state):
        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_context_value(state)
            if not self.training == 'torch':
                state = self.compute_prediction_error(state)
                self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        
        return state
    
    def sim_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_context_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Reset indices on succeeding fits
        self.reset_datalists()

        #Unpack free parameters
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature, *optionals = x
        self.unpack_optionals(optionals)

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'q_values', context_reward=True)

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, context_reward=True)
    
class wRelative(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Weighted-Relative Model
    Different from the Relative model in that it uses a weighted average of the rewards rather than learning the average reward

    Parameters:
    ------------
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    weighting_factor: float
            Weighting factor for contextual information
    """

    def __init__(self, factual_lr, counterfactual_lr, temperature, weighting_factor, novel_value, decay_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.temperature = temperature
        self.weighting_factor = weighting_factor
        self.novel_value = novel_value
        self.decay_factor = decay_factor
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'temperature': self.temperature,
                           'weighting_factor': self.weighting_factor,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}
    
    #RL functions
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward

        return state
    
    def select_action(self, state):

        if self.training == 'torch':
            transformed_q_values = torch.exp(torch.div(torch.tensor(state['q_values']), self.temperature))
            probability_q_values = torch.cumsum(transformed_q_values/torch.sum(transformed_q_values), 0)
            state['action'] = self.torch_select_action(probability_q_values)
        else:
            transformed_q_values = np.exp(np.divide(state['q_values'], self.temperature))
            probability_q_values = (transformed_q_values/np.sum(transformed_q_values)).cumsum()
            state['action'] = np.where(probability_q_values >= rnd.random())[0][0]
        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state
    
    def compute_prediction_error(self, state):
        if self.training == 'torch':
            state['prediction_errors'] = state['rewards'] - (torch.mean(state['rewards'])*self.weighting_factor) - state['q_values']
        else:
            state['prediction_errors'] = [state['rewards'][i] - (np.mean(state['rewards'])*self.weighting_factor) - state['q_values'][i] for i in range(len(state['rewards']))]
        return state
    
    #Run trial functions
    def forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_reward(state)
            state = self.get_q_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_model_update(self, state):
        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            if not self.training == 'torch':
                state = self.compute_prediction_error(state)
                self.update_model(state)
        else:
            state = self.get_final_q_values(state)
        
        return state

    def sim_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''
        
        #Reset indices on succeeding fits
        self.reset_datalists()
        
        #Unpack free parameters
        self.factual_lr, self.counterfactual_lr, self.temperature, self.weighting_factor, *optionals = x
        self.unpack_optionals(optionals)

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'q_values')

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args)