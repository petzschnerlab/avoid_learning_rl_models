import random as rnd
import numpy as np
import torch.nn as nn
import torch

from .rl_toolbox import RLToolbox
class Hybrid2012(RLToolbox, nn.Module):
    
    """
    Hybrid reinforcement learning model combining factual and counterfactual learning
    with actor-critic dynamics and temperature-scaled action selection.
    """

    def __init__(self,
                 factual_lr: float,
                 counterfactual_lr: float,
                 factual_actor_lr: float,
                 counterfactual_actor_lr: float,
                 critic_lr: float,
                 temperature: float,
                 mixing_factor: float,
                 valence_factor: float,
                 novel_value: float,
                 decay_factor: float):
        
        """
        Initialize the Hybrid2012 RL model with the given parameters.

        Parameters
        ------------
        factual_lr : float
            Learning rate for factual critic updates.
        counterfactual_lr : float
            Learning rate for counterfactual critic updates.
        factual_actor_lr : float
            Learning rate for factual actor updates.
        counterfactual_actor_lr : float
            Learning rate for counterfactual actor updates.
        critic_lr : float
            Learning rate for general critic updates.
        temperature : float
            Softmax temperature for action selection.
        mixing_factor : float
            Weighting factor for mixing systems.
        valence_factor : float
            Factor to scale rewards by valence.
        novel_value : float
            Initial value for novel stimuli.
        decay_factor : float
            Value decay rate across trials.
        """
        
        super().__init__()
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

        self.parameters = {
            'factual_lr': self.factual_lr, 
            'counterfactual_lr': self.counterfactual_lr, 
            'factual_actor_lr': self.factual_actor_lr,
            'counterfactual_actor_lr': self.counterfactual_actor_lr,
            'critic_lr': self.critic_lr,
            'temperature': self.temperature,
            'mixing_factor': self.mixing_factor,
            'valence_factor': self.valence_factor,
            'novel_value': self.novel_value,
            'decay_factor': self.decay_factor
        }

    def get_reward(self, state: dict) -> dict:
        
        """
        Sample rewards based on stimulus probabilities and apply feedback and valence.

        Parameters
        ------------
        state : dict
            Current state information including stimulus IDs and feedback.

        Returns
        --------
        dict
            Updated state with computed rewards.
        """
        
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)

        state['rewards'] = reward
        return state
    
    def compute_prediction_error(self, state: dict) -> dict:
        
        """
        Compute prediction error signals for both Q-values and V-values.

        Parameters
        ------------
        state : dict
            State containing current values and rewards.

        Returns
        --------
        dict
            Updated state with prediction error values.
        """
        
        if self.training == 'torch':
            state['q_prediction_errors'] = state['rewards'] - state['q_values'].detach()
            state['v_prediction_errors'] = state['rewards'] - state['v_values']
        else:
            state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state: dict) -> dict:
        
        """
        Select an action using a softmax distribution over h-values.

        Parameters
        ------------
        state : dict
            State dictionary containing h-values for available actions.

        Returns
        --------
        dict
            Updated state with chosen action and optional accuracy.
        """
        
        if self.training == 'torch':
            transformed_h_values = torch.exp(torch.div(state['h_values'], self.temperature))
            probability_h_values = torch.cumsum(transformed_h_values / torch.sum(transformed_h_values), dim=0)
            state['action'] = self.torch_select_action(probability_h_values)
        else:
            transformed_h_values = np.exp(np.divide(state['h_values'], self.temperature))
            probability_h_values = (transformed_h_values / np.sum(transformed_h_values)).cumsum()
            state['action'] = np.where(probability_h_values >= rnd.random())[0][0]

        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    def forward(self, state: dict, phase: str = 'learning') -> None:
        
        """
        Execute a trial in learning or transfer/test phase.

        Parameters
        ------------
        state : dict
            Dictionary representing current trial state.
        phase : str, default='learning'
            Trial phase: 'learning' or any other string for testing phase.

        Returns
        --------
        None
        """
        
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

    def fit_model_update(self, state: dict) -> None:
        
        """
        Update the model using the computed prediction errors.

        Parameters
        ------------
        state : dict
            State dictionary with values needed for update.

        Returns
        --------
        None
        """
        
        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state: dict, phase: str = 'learning') -> dict:
        
        """
        Forward pass during fitting phase.

        Parameters
        ------------
        state : dict
            Current trial state.
        phase : str, default='learning'
            Phase of trial: 'learning' or other.

        Returns
        --------
        dict
            Updated trial state.
        """
        
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
    
    def sim_forward(self, state: dict, phase: str = 'learning') -> None:
        
        """
        Forward pass during simulation.

        Parameters
        ------------
        state : dict
            Current trial state.
        phase : str, default='learning'
            Trial phase: 'learning' or other.

        Returns
        --------
        None
        """
        
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

    def fit_func(self, x: list, *args) -> float:
        
        """
        Fit model parameters to data and return the negative log-likelihood.

        Parameters
        ------------
        x : list
            List of model parameters.
        *args : tuple
            Additional task-specific arguments (e.g., actions, rewards).

        Returns
        --------
        float
            Negative log-likelihood of observed behavior.
        """
        
        self.reset_datalists()

        # Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, *optionals = x
        self.unpack_optionals(optionals)

        return -self.fit_task(args, 'h_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args):
        
        """
        Simulate the model using current parameters and input data.

        Parameters
        ------------
        *args : tuple
            Task inputs for simulation.

        Returns
        --------
        Any
            Simulation output from `sim_task`.
        """
        
        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])
class Hybrid2021(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Hybrid Actor-Critic-Q-Learning Model (Geana et al., 2021)
    """

    def __init__(self, factual_lr, counterfactual_lr, factual_actor_lr, counterfactual_actor_lr, 
                 critic_lr, temperature, mixing_factor, valence_factor, noise_factor, novel_value, decay_factor):

        """
        Initialize Hybrid2021 model with specified learning and control parameters.

        Parameters
        ------------
        factual_lr : float
            Learning rate for factual Q-value updates
        counterfactual_lr : float
            Learning rate for counterfactual Q-value updates
        factual_actor_lr : float
            Learning rate for factual actor updates
        counterfactual_actor_lr : float
            Learning rate for counterfactual actor updates
        critic_lr : float
            Learning rate for value critic updates
        temperature : float
            Softmax temperature for stochastic action selection
        mixing_factor : float
            Weighting between actor and critic contributions
        valence_factor : float
            Factor for applying valence to outcomes
        noise_factor : float
            Amount of stochastic noise in action selection
        novel_value : float
            Initial value for unseen stimuli
        decay_factor : float
            Decay applied to Q-values (and W-values in this implementation)
        """

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

        """
        Sample and apply feedback-based reward using outcome probabilities.

        Parameters
        ------------
        state : dict
            Trial state containing stimulus IDs and feedback

        Returns
        --------
        dict
            Updated state with sampled rewards
        """

        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(self, state):

        """
        Compute prediction errors for critic and Q-learning values.

        Parameters
        ------------
        state : dict
            State including rewards and value estimates

        Returns
        --------
        dict
            Updated state with prediction errors
        """

        if self.training == 'torch':
            state['q_prediction_errors'] = state['rewards'] - state['q_values'].detach()
            state['v_prediction_errors'] = state['rewards'][state['action']] - state['v_values']
        else:
            state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses selected reward

        return state

    def select_action(self, state):

        """
        Select an action based on softmax over mixed actor-critic values.

        Parameters
        ------------
        state : dict
            State with Q-values and actor values

        Returns
        --------
        dict
            State with selected action and updated h-values
        """

        if self.training == 'torch':
            state['h_values'] = (state['w_values'] * (1-self.mixing_factor)) + (state['q_values'] * self.mixing_factor)
            transformed_h_values = torch.exp(torch.div(state['h_values'], self.temperature))
            probability_h_values = transformed_h_values/torch.sum(transformed_h_values)
            uniform_dist = torch.ones(len(probability_h_values))/len(probability_h_values)
            probability_h_values = torch.cumsum(((1-self.noise_factor)*probability_h_values) + (self.noise_factor*uniform_dist), dim=0)
            state['action'] = self.torch_select_action(probability_h_values)
            state['h_values'] = probability_h_values
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

        """
        Run a single trial of the model, learning or testing.

        Parameters
        ------------
        state : dict
            Current trial state
        phase : str, default='learning'
            Whether in learning phase or test phase

        Returns
        --------
        None
        """

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

        """
        Apply prediction error update during model fitting.

        Parameters
        ------------
        state : dict
            Trial state containing errors

        Returns
        --------
        None
        """

        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state, phase = 'learning'):

        """
        Forward pass during model fitting.

        Parameters
        ------------
        state : dict
            Current trial state
        phase : str, default='learning'
            Learning or transfer phase

        Returns
        --------
        dict
            Updated state
        """

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

        """
        Forward pass during simulation.

        Parameters
        ------------
        state : dict
            Current trial state
        phase : str, default='learning'
            Trial phase

        Returns
        --------
        None
        """

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

        """
        Fit the model to the data.

        Parameters
        ------------
        x : list
            Model parameter values
        args : tuple
            Data tuple (e.g., actions, rewards)

        Returns
        --------
        float
            Negative log-likelihood of the fit
        """

        #Reset indices on succeeding fits
        self.reset_datalists()

        #Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.noise_factor, *optionals = x
        self.unpack_optionals(optionals)

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'h_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args):

        """
        Simulate the model.

        Parameters
        ------------
        args : tuple
            Task input arguments

        Returns
        --------
        Any
            Simulation output
        """

        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])
    
class StandardHybrid2012(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Hybrid Actor-Critic-Q-Learning Model (Gold et al., 2012)
    """

    def __init__(self,
                 factual_lr: float,
                 factual_actor_lr: float,
                 critic_lr: float,
                 temperature: float,
                 mixing_factor: float,
                 valence_factor: float,
                 novel_value: float,
                 decay_factor: float) -> None:

        """
        
        Initialize the StandardHybrid2012 model with specified parameters.

        Parameters
        ----------
        factual_lr : float
            Learning rate for factual Q-value update.
        factual_actor_lr : float
            Learning rate for actor updates based on factual outcomes.
        critic_lr : float
            Learning rate for value function updates.
        temperature : float
            Temperature parameter for softmax action selection.
        mixing_factor : float
            Weighting between Q-values and V-values.
        valence_factor : float
            Factor to modulate reward valence.
        novel_value : float
            Initial value assigned to novel stimuli.
        decay_factor : float
            Rate of decay for Q-values and value estimates.

        Returns
        -------
        None
        """

        super().__init__()

        # Set parameters
        self.factual_lr = factual_lr
        self.factual_actor_lr = factual_actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.mixing_factor = mixing_factor
        self.valence_factor = valence_factor
        self.novel_value = novel_value
        self.decay_factor = decay_factor

        self.parameters = {'factual_lr': self.factual_lr, 
                           'factual_actor_lr': self.factual_actor_lr,
                           'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor,
                           'valence_factor': self.valence_factor,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    def get_reward(self, state: dict) -> dict:

        """
        
        Generate rewards based on stimulus probabilities and feedback.

        Parameters
        ----------
        state : dict
            Current state dictionary containing 'stim_id', 'probabilities', and 'feedback'.

        Returns
        -------
        dict
            Updated state dictionary with computed 'rewards'.
        """

        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)

        state['rewards'] = reward

        return state

    def compute_prediction_error(self, state: dict) -> dict:

        """
        
        Compute Q-value and value-function prediction errors.

        Parameters
        ----------
        state : dict
            Current state dictionary containing 'rewards', 'q_values', 'v_values', and 'action'.

        Returns
        -------
        dict
            Updated state dictionary with 'q_prediction_errors' and 'v_prediction_errors'.
        """

        if self.training == 'torch':
            state['q_prediction_errors'] = state['rewards'] - state['q_values'].detach()
            state['v_prediction_errors'] = state['rewards'][state['action']] - state['v_values']
        else:
            state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state: dict) -> dict:

        """
        
        Select an action using softmax policy over h-values.

        Parameters
        ----------
        state : dict
            Current state dictionary containing 'h_values', optionally 'correct_action'.

        Returns
        -------
        dict
            Updated state dictionary with chosen 'action' and optional 'accuracy'.
        """

        if self.training == 'torch':
            transformed_h_values = torch.exp(torch.div(state['h_values'], self.temperature))
            probability_h_values = torch.cumsum(transformed_h_values / torch.sum(transformed_h_values), dim=0)
            state['action'] = self.torch_select_action(probability_h_values)
        else:
            transformed_h_values = np.exp(np.divide(state['h_values'], self.temperature))
            probability_h_values = (transformed_h_values / np.sum(transformed_h_values)).cumsum()
            state['action'] = np.where(probability_h_values >= rnd.random())[0][0]

        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    def forward(self, state: dict, phase: str = 'learning') -> None:

        """
        
        Run model forward pass for a single trial.

        Parameters
        ----------
        state : dict
            Current state dictionary containing all required trial information.
        phase : str, optional
            Phase of the trial ('learning' or otherwise), by default 'learning'.

        Returns
        -------
        None
        """

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

    def fit_model_update(self, state: dict) -> None:

        """
        
        Apply model update after computing prediction error.

        Parameters
        ----------
        state : dict
            Current state dictionary containing 'rewards', 'q_values', 'v_values', and 'action'.

        Returns
        -------
        None
        """

        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state: dict, phase: str = 'learning') -> dict:

        """
        
        Run forward pass during model fitting.

        Parameters
        ----------
        state : dict
            Current state dictionary containing trial data.
        phase : str, optional
            Phase of the trial ('learning' or otherwise), by default 'learning'.

        Returns
        -------
        dict
            Updated state dictionary after forward pass.
        """

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

    def sim_forward(self, state: dict, phase: str = 'learning') -> None:

        """
        
        Run forward pass for simulation purposes.

        Parameters
        ----------
        state : dict
            Current state dictionary containing trial data.
        phase : str, optional
            Phase of the trial ('learning' or otherwise), by default 'learning'.

        Returns
        -------
        None
        """

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

    def fit_func(self, x: list, *args) -> float:

        """
        
        Fit the model to behavioral data.

        Parameters
        ----------
        x : list
            List of model parameters to fit (learning rates, temperature, etc.).
        *args : tuple
            Additional arguments, e.g., data.

        Returns
        -------
        float
            Negative log likelihood of observed actions for fitting.
        """

        self.reset_datalists()

        # Unpack parameters
        self.factual_lr, self.factual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, *optionals = x
        self.unpack_optionals(optionals)

        return -self.fit_task(args, 'h_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args) -> dict:

        """
        
        Simulate the model.

        Parameters
        ----------
        *args : tuple
            Arguments for the simulation.

        Returns
        -------
        dict
            Simulated data from the model.
        """

        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])

class StandardHybrid2021(RLToolbox, nn.Module):


    def __init__(self, factual_lr: float,
                 factual_actor_lr: float,
                 critic_lr: float,
                 temperature: float,
                 mixing_factor: float,
                 valence_factor: float,
                 noise_factor: float,
                 novel_value: float,
                 decay_factor: float) -> None:
        
        """
        Initialize the StandardHybrid2021 model with specified parameters.
        
        Parameters
        ----------
        factual_lr : float
            Learning rate for factual Q-value updates.
        factual_actor_lr : float
            Learning rate for actor updates based on factual outcomes.
        critic_lr : float
            Learning rate for value function updates.
        temperature : float
            Temperature parameter for softmax action selection.
        mixing_factor : float
            Weighting between Q-values and value estimates.
        valence_factor : float
            Factor modulating reward valence.
        noise_factor : float
            Noise parameter affecting action selection probabilities.
        novel_value : float
            Initial value assigned to novel stimuli.
        decay_factor : float
            Rate of decay for Q-values and value estimates.
        
        Returns
        -------
        None
        """
        
        super().__init__()

        # Set parameters
        self.factual_lr = factual_lr
        self.factual_actor_lr = factual_actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.mixing_factor = mixing_factor
        self.noise_factor = noise_factor
        self.valence_factor = valence_factor
        self.novel_value = novel_value
        self.decay_factor = decay_factor

        self.parameters = {'factual_lr': self.factual_lr, 
                           'factual_actor_lr': self.factual_actor_lr,
                           'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor,
                           'noise_factor': self.noise_factor,
                           'valence_factor': self.valence_factor,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    # RL functions    
    def get_reward(self, state: dict) -> dict:
        
        """
        Generate rewards based on stimulus probabilities and feedback.
        
        Parameters
        ----------
        state : dict
            Dictionary containing keys:
                - 'stim_id': List of stimulus IDs.
                - 'probabilities': List of reward probabilities for each stimulus.
                - 'feedback': Feedback multiplier (e.g., reward magnitude).
        
        Returns
        -------
        dict
            Updated state dictionary including 'rewards'.
        """
        
        random_numbers = [rnd.random() for _ in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(self, state: dict) -> dict:
        
        """
        Compute Q-value and value-function prediction errors.
        
        Parameters
        ----------
        state : dict
            Dictionary containing keys:
                - 'rewards': List of received rewards.
                - 'q_values': List or tensor of Q-values.
                - 'v_values': List or tensor of value function estimates.
                - 'action': Selected action index.
                - 'training': Training mode indicator.
        
        Returns
        -------
        dict
            Updated state dictionary including:
                - 'q_prediction_errors': Prediction errors for Q-values.
                - 'v_prediction_errors': Prediction errors for value estimates.
        """
        
        if self.training == 'torch':
            state['q_prediction_errors'] = state['rewards'] - state['q_values'].detach()
            state['v_prediction_errors'] = state['rewards'][state['action']] - state['v_values']
        else:
            state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state: dict) -> dict:
        
        """
        Select an action using a noisy softmax policy over combined values.
        
        Parameters
        ----------
        state : dict
            Dictionary containing keys:
                - 'w_values': List or tensor of weighted value estimates.
                - 'q_values': List or tensor of Q-values.
                - 'temperature': Temperature for softmax (from self).
                - 'mixing_factor': Weighting factor (from self).
                - 'noise_factor': Noise factor (from self).
                - 'correct_action' (optional): Index of correct action for accuracy.
        
        Returns
        -------
        dict
            Updated state dictionary including:
                - 'action': Selected action index.
                - 'h_values': Probabilities over actions.
                - 'accuracy' (if applicable): 1 if selected action matches correct, else 0.
        """
        
        if self.training == 'torch':
            state['h_values'] = (state['w_values'] * (1 - self.mixing_factor)) + (state['q_values'] * self.mixing_factor)
            transformed_h_values = torch.exp(torch.div(state['h_values'], self.temperature))
            probability_h_values = transformed_h_values / torch.sum(transformed_h_values)
            uniform_dist = torch.ones(len(probability_h_values)) / len(probability_h_values)
            probability_h_values = torch.cumsum(((1 - self.noise_factor) * probability_h_values) + (self.noise_factor * uniform_dist), dim=0)
            state['action'] = self.torch_select_action(probability_h_values)
            state['h_values'] = probability_h_values
        else:
            state['h_values'] = [(state['w_values'][i] * (1 - self.mixing_factor)) + (state['q_values'][i] * self.mixing_factor) for i in range(len(state['w_values']))]
            transformed_h_values = np.exp(np.divide(state['h_values'], self.temperature))
            probability_h_values = (transformed_h_values / np.sum(transformed_h_values))
            uniform_dist = np.ones(len(probability_h_values)) / len(probability_h_values)
            probability_h_values = (((1 - self.noise_factor) * probability_h_values) + (self.noise_factor * uniform_dist)).cumsum()
            state['action'] = np.where(probability_h_values >= rnd.random())[0][0]

        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state
    
    # Run trial functions
    def forward(self, state: dict, phase: str = 'learning') -> None:
        
        """
        Execute the forward pass for a single trial.
        
        Parameters
        ----------
        state : dict
            Trial state dictionary with necessary model inputs.
        phase : str, optional
            Phase of the trial ('learning' or other), by default 'learning'.
        
        Returns
        -------
        None
        """
        
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

    def fit_model_update(self, state: dict) -> None:
        
        """
        Update model parameters after computing prediction errors during fitting.
        
        Parameters
        ----------
        state : dict
            Current trial state with prediction errors computed.
        
        Returns
        -------
        None
        """
        
        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state: dict, phase: str = 'learning') -> dict:
        
        """
        Run the forward pass during model fitting.
        
        Parameters
        ----------
        state : dict
            Trial state dictionary with necessary inputs.
        phase : str, optional
            Phase of the trial ('learning' or other), by default 'learning'.
        
        Returns
        -------
        dict
            Updated state after forward pass.
        """
        
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
    
    def sim_forward(self, state: dict, phase: str = 'learning') -> None:
        
        """
        Run the forward pass for simulation purposes.
        
        Parameters
        ----------
        state : dict
            Trial state dictionary with necessary inputs.
        phase : str, optional
            Phase of the trial ('learning' or other), by default 'learning'.
        
        Returns
        -------
        None
        """
        
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

    def fit_func(self, x: list, *args) -> float:
        
        """
        Fit the model to behavioral data by optimizing parameters.
        
        Parameters
        ----------
        x : list
            List of parameter values to be fitted, including:
            factual_lr, factual_actor_lr, critic_lr, temperature, mixing_factor, noise_factor, and optionals.
        *args : tuple
            Additional arguments required for fitting (e.g., data).
        
        Returns
        -------
        float
            Negative log likelihood of observed actions under the model.
        """
        
        # Reset indices on succeeding fits
        self.reset_datalists()

        # Unpack parameters
        self.factual_lr, self.factual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.noise_factor, *optionals = x
        self.unpack_optionals(optionals)

        # Return negative log likelihood of all observed actions
        return -self.fit_task(args, 'h_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args):
        
        """
        Simulate the model behavior given input arguments.
        
        Parameters
        ----------
        *args : tuple
            Arguments for simulation.
        
        Returns
        -------
        Any
            Simulation results.
        """
        
        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])
