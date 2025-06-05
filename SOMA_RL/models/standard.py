import numpy as np
import random as rnd

from models.rl_toolbox import RLToolbox
import torch.nn as nn
import torch

class QLearning(RLToolbox, nn.Module):

    def __init__(self,
                 factual_lr: float,
                 counterfactual_lr: float,
                 temperature: float,
                 novel_value: float,
                 decay_factor: float):
            
        """
        Reinforcement Learning Model: Q-Learning

        Parameters
        ----------
        factual_lr : float
            Learning rate for factual Q-value update
        counterfactual_lr : float
            Learning rate for counterfactual Q-value update
        temperature : float
            Temperature parameter for softmax action selection
        novel_value : float
            Initial value assigned to novel stimuli
        decay_factor : float
            Rate of decay for Q-values
        """
                
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.temperature = temperature
        self.novel_value = novel_value
        self.decay_factor = decay_factor
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'temperature': self.temperature,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    def get_reward(self, state: dict) -> dict:

        """
        Generate a reward outcome based on probabilities and feedback.

        Parameters
        ----------
        state : dict
            Dictionary containing trial information, including 'stim_id', 'probabilities', and 'feedback'.

        Returns
        -------
        dict
            Updated state with computed 'rewards'.
        """

        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward

        return state

    def select_action(self, state: dict) -> dict:

        """
        Select an action based on softmax policy over Q-values.

        Parameters
        ----------
        state : dict
            Dictionary with current 'q_values' and optionally 'correct_action'.

        Returns
        -------
        dict
            Updated state with selected 'action' and optionally 'accuracy'.
        """

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

    def compute_prediction_error(self, state: dict) -> dict:

        """
        Compute prediction errors based on reward and current Q-values.

        Parameters
        ----------
        state : dict
            Dictionary containing 'rewards' and 'q_values'.

        Returns
        -------
        dict
            Updated state with computed 'prediction_errors'.
        """

        if self.training == 'torch':
            state['prediction_errors'] = state['rewards'] - state['q_values']
        else:
            state['prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        return state

    def forward(self, state: dict, phase: str = 'learning') -> None:

        """
        Perform forward pass of the model.

        Parameters
        ----------
        state : dict
            Trial-level data including task state.
        phase : str, optional
            Either 'learning' or 'transfer'.

        Returns
        -------
        None
        """

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

    def fit_model_update(self, state: dict) -> None:

        """
        Update model using prediction errors (for fitting phase).

        Parameters
        ----------
        state : dict
            Dictionary with trial data.

        Returns
        -------
        None
        """

        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state: dict, phase: str = 'learning') -> dict:

        """
        Perform forward pass for model fitting.

        Parameters
        ----------
        state : dict
            Trial-level input data.
        phase : str, optional
            Trial phase ('learning' or other).

        Returns
        -------
        dict
            Updated state.
        """

        if phase == 'learning':
            state = self.get_q_value(state)
            if not self.training == 'torch':
                state = self.compute_prediction_error(state)
                self.update_model(state)
        else:
            state = self.get_final_q_values(state)

        return state

    def sim_forward(self, state: dict, phase: str = 'learning') -> None:

        """
        Run forward pass for model simulations.

        Parameters
        ----------
        state : dict
            Trial state input.
        phase : str, optional
            Trial phase ('learning' or otherwise).

        Returns
        -------
        None
        """

        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_func(self, x: list, *args: tuple) -> float:

        """
        Fit the model to behavioral data.

        Parameters
        ----------
        x : list
            Model parameters to fit.
        *args : tuple
            Additional data needed for fitting.

        Returns
        -------
        float
            Negative log-likelihood of the observed actions.
        """

        self.reset_datalists()
        self.factual_lr, self.counterfactual_lr, self.temperature, *optionals = x
        self.unpack_optionals(optionals)
        return -self.fit_task(args, 'q_values')

    def sim_func(self, *args: tuple) -> any:

        """
        Simulate the model on input task data.

        Parameters
        ----------
        *args : tuple
            Input task structure for simulation.

        Returns
        -------
        any
            Output of self.sim_task.
        """

        return self.sim_task(args)
class ActorCritic(RLToolbox, nn.Module):

    """
    Reinforcement Learning Model: Actor-Critic
    """

    def __init__(self,
                 factual_actor_lr: float,
                 counterfactual_actor_lr: float,
                 critic_lr: float,
                 temperature: float,
                 valence_factor: float,
                 novel_value: float,
                 decay_factor: float):
        
        """
        Reinforcement Learning Model: Actor-Critic

        Parameters
        ----------
        factual_actor_lr : float
            Learning rate for factual actor update
        counterfactual_actor_lr : float
            Learning rate for counterfactual actor update
        critic_lr : float
            Learning rate for critic value updates
        temperature : float
            Temperature parameter for softmax action selection
        valence_factor : float
            Factor for adjusting reward valence
        novel_value : float
            Initial value assigned to novel stimuli
        decay_factor : float
            Rate of decay for Q-values
        """
        
        super().__init__()

        # Set parameters
        self.factual_actor_lr = factual_actor_lr
        self.counterfactual_actor_lr = counterfactual_actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.valence_factor = valence_factor
        self.novel_value = novel_value
        self.decay_factor = decay_factor

        self.parameters = {'factual_actor_lr': self.factual_actor_lr, 
                           'counterfactual_actor_lr': self.counterfactual_actor_lr, 
                           'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'valence_factor': self.valence_factor,
                           'novel_value': self.novel_value,
                           'decay_factor': self.decay_factor}

    def get_reward(self, state: dict) -> dict:

        """
        Generate a reward outcome based on probabilities, feedback, and valence.

        Parameters
        ----------
        state : dict
            Dictionary containing trial information, including 'stim_id', 'probabilities', and 'feedback'.

        Returns
        -------
        dict
            Updated state with computed 'rewards'.
        """

        random_numbers = [rnd.random() for _ in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        reward = self.reward_valence(reward)
        state['rewards'] = reward

        return state

    def compute_prediction_error(self, state: dict) -> dict:

        """
        Compute prediction errors using critic value.

        Parameters
        ----------
        state : dict
            Dictionary containing 'rewards' and 'v_values'.

        Returns
        -------
        dict
            Updated state with computed 'prediction_errors'.
        """

        if self.training == 'torch':
            state['prediction_errors'] = state['rewards'] - state['v_values']
        else:
            state['prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state: dict) -> dict:

        """
        Select an action based on softmax policy over W-values.

        Parameters
        ----------
        state : dict
            Dictionary with current 'w_values' and optionally 'correct_action'.

        Returns
        -------
        dict
            Updated state with selected 'action' and optionally 'accuracy'.
        """

        transformed_w_values = np.exp(np.divide(state['w_values'], self.temperature))
        probability_w_values = (transformed_w_values/np.sum(transformed_w_values)).cumsum()
        state['action'] = np.where(probability_w_values >= rnd.random())[0][0]
        if 'correct_action' in state:
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    def forward(self, state: dict, phase: str = 'learning') -> None:

        """
        Perform forward pass of the model.

        Parameters
        ----------
        state : dict
            Trial-level data including task state.
        phase : str, optional
            Either 'learning' or 'transfer'.

        Returns
        -------
        None
        """

        if phase == 'learning':
            state = self.get_reward(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_w_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_model_update(self, state: dict) -> None:

        """
        Update model using prediction errors (for fitting phase).

        Parameters
        ----------
        state : dict
            Dictionary with trial data.

        Returns
        -------
        None
        """

        state = self.compute_prediction_error(state)
        self.update_model(state)

    def fit_forward(self, state: dict, phase: str = 'learning') -> dict:

        """
        Perform forward pass for model fitting.

        Parameters
        ----------
        state : dict
            Trial-level input data.
        phase : str, optional
            Trial phase ('learning' or other).

        Returns
        -------
        dict
            Updated state.
        """

        if phase == 'learning':
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            if not self.training == 'torch':
                state = self.compute_prediction_error(state)
                self.update_model(state)
        else:
            state = self.get_final_w_values(state)

        return state

    def sim_forward(self, state: dict, phase: str = 'learning') -> None:

        """
        Run forward pass for model simulations.

        Parameters
        ----------
        state : dict
            Trial state input.
        phase : str, optional
            Trial phase ('learning' or otherwise).

        Returns
        -------
        None
        """

        if phase == 'learning':
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_w_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_func(self, x: list, *args: tuple) -> float:

        """
        Fit the model to behavioral data.

        Parameters
        ----------
        x : list
            Model parameters to fit.
        *args : tuple
            Additional data needed for fitting.

        Returns
        -------
        float
            Negative log-likelihood of the observed actions.
        """

        self.reset_datalists()
        self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, *optionals = x
        self.unpack_optionals(optionals)
        return -self.fit_task(args, 'w_values', transform_reward=self.optional_parameters['bias'])

    def sim_func(self, *args: tuple) -> any:

        """
        Simulate the model on input task data.

        Parameters
        ----------
        *args : tuple
            Input task structure for simulation.

        Returns
        -------
        any
            Output of self.sim_task.
        """

        return self.sim_task(args, transform_reward=self.optional_parameters['bias'])