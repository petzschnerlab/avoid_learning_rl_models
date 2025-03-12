import random as rnd
import numpy as np
import torch.nn as nn
import torch

from .rl_toolbox import RLToolbox

class Hybrid2012(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Hybrid Actor-Critic-Q-Learning Model (Gold et al., 2012)
    
    Parameters:
    ------------
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    """

    def __init__(self, factual_lr, counterfactual_lr, factual_actor_lr, counterfactual_actor_lr, 
                 critic_lr, temperature, mixing_factor, valence_factor, novel_value, decay_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.factual_actor_lr = factual_actor_lr
        self.counterfactual_actor_lr = counterfactual_actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.mixing_factor = mixing_factor
        self.valence_factor = valence_factor
        self.novel_value = novel_value
        self.decay_factor = decay_factor

        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'factual_actor_lr': self.factual_actor_lr,
                           'counterfactual_actor_lr': self.counterfactual_actor_lr,
                           'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor,
                           'valence_factor': self.valence_factor,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    #RL functions    
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(self, state):
        if self.training == 'torch':
            state['q_prediction_errors'] = state['rewards'] - state['q_values'].detach()
            state['v_prediction_errors'] = state['rewards'][state['action']] - state['v_values']
        else:
            state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['v_prediction_errors'] = [state['rewards'][state['action']] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state):

        if self.training == 'torch':
            transformed_h_values = torch.exp(torch.div(state['h_values'], self.temperature))
            probability_h_values = torch.cumsum(transformed_h_values/torch.sum(transformed_h_values), dim=0)
            state['action'] = torch.where(probability_h_values >= rnd.random())[0][0]
        else:
            transformed_h_values = np.exp(np.divide(state['h_values'], self.temperature))
            probability_h_values = (transformed_h_values/np.sum(transformed_h_values)).cumsum()
            state['action'] = np.where(probability_h_values >= rnd.random())[0][0]

        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    #Run trial functions
    def forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_reward(state)
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.get_h_values(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.get_h_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_model_update(self, state):
        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.get_h_values(state)
            if not self.training == 'torch':
                state = self.compute_prediction_error(state)
                self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.get_h_values(state)
            state = self.select_action(state)

        return state
    
    def sim_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.get_h_values(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.get_h_values(state)
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

        #Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, *optionals = x
        self.unpack_optionals(optionals)

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'h_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])

class Hybrid2021(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Hybrid Actor-Critic-Q-Learning Model (Geana et al., 2021)
    Note: Applied the decay function to the q-values and w-values, but the publication seems to only apply it to the q-values.
    
    Parameters:
    ------------
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    """

    def __init__(self, factual_lr, counterfactual_lr, factual_actor_lr, counterfactual_actor_lr, 
                 critic_lr, temperature, mixing_factor, valence_factor, noise_factor, novel_value, decay_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.factual_actor_lr = factual_actor_lr
        self.counterfactual_actor_lr = counterfactual_actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.mixing_factor = mixing_factor
        self.noise_factor = noise_factor
        self.valence_factor = valence_factor
        self.novel_value = novel_value
        self.decay_factor = decay_factor

        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'factual_actor_lr': self.factual_actor_lr,
                           'counterfactual_actor_lr': self.counterfactual_actor_lr,
                           'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor,
                           'noise_factor': self.noise_factor,
                           'valence_factor': self.valence_factor,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    #RL functions    
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(self, state):
        if self.training == 'torch':
            state['q_prediction_errors'] = state['rewards'] - state['q_values']
            state['v_prediction_errors'] = state['rewards'][state['action']] - state['v_values']
        else:
            state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['v_prediction_errors'] = [state['rewards'][state['action']] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses selected reward

        return state

    def select_action(self, state):

        if self.training == 'torch':
            state['h_values'] = (state['w_values'] * (1-self.mixing_factor)) + (state['q_values'] * self.mixing_factor)
            transformed_h_values = torch.exp(torch.div(state['h_values'], self.temperature))
            probability_h_values = transformed_h_values/torch.sum(transformed_h_values)
            uniform_dist = torch.ones(len(probability_h_values))/len(probability_h_values)
            probability_h_values = torch.cumsum(((1-self.noise_factor)*probability_h_values) + (self.noise_factor*uniform_dist), dim=0)
            state['action'] = torch.where(probability_h_values >= rnd.random())[0][0]
        else:
            state['h_values'] = [(state['w_values'][i] * (1-self.mixing_factor)) + (state['q_values'][i] * self.mixing_factor) for i in range(len(state['w_values']))]
            transformed_h_values = np.exp(np.divide(state['h_values'], self.temperature))
            probability_h_values = (transformed_h_values/np.sum(transformed_h_values))
            uniform_dist = np.ones(len(probability_h_values))/len(probability_h_values)
            probability_h_values = (((1-self.noise_factor)*probability_h_values) + (self.noise_factor*uniform_dist)).cumsum()
            state['action'] = np.where(probability_h_values >= rnd.random())[0][0]

        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state
    
    #Run trial functions
    def forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_reward(state)
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.get_h_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_model_update(self, state):
        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.get_h_values(state)
            if not self.training == 'torch':
                state = self.compute_prediction_error(state)
                self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.get_h_values(state)
            state = self.select_action(state)

        return state
    
    def sim_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.get_h_values(state)
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

        #Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.noise_factor, *optionals = x
        self.unpack_optionals(optionals)

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'h_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])