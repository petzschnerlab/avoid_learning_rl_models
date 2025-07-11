import numpy as np
import random as rnd
import torch
import torch.nn as nn

from .rl_toolbox import RLToolbox

class Relative(RLToolbox, nn.Module):
    """
    Reinforcement Learning Model: Relative (Palminteri et al., 2015)
    """

    def __init__(self, factual_lr: float, counterfactual_lr: float, contextual_lr: float,
                 temperature: float, novel_value: float, decay_factor: float):
        
        
        """
        Initialize the Relative model with specified parameters.

        Parameters
        ----------
        factual_lr : float
            Learning rate for factual Q-value updates.
        counterfactual_lr : float
            Learning rate for counterfactual Q-value updates.
        contextual_lr : float
            Learning rate for contextual value updates.
        temperature : float
            Temperature parameter for softmax action selection.
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
        self.counterfactual_lr = counterfactual_lr
        self.contextual_lr = contextual_lr
        self.temperature = temperature
        self.novel_value = novel_value
        self.decay_factor = decay_factor

        self.parameters = {
            'factual_lr': self.factual_lr,
            'counterfactual_lr': self.counterfactual_lr,
            'contextual_lr': self.contextual_lr,
            'temperature': self.temperature,
            'novel_value': self.novel_value,
            'decay_factor': self.decay_factor
        }

    # RL functions
    def get_reward(self, state: dict) -> dict:
        
        """
        Generate rewards based on stimulus probabilities and feedback.

        Parameters
        ----------
        state : dict
            Dictionary with keys:
                - 'stim_id': list of stimulus IDs.
                - 'probabilities': list of reward probabilities.
                - 'feedback': feedback multiplier (e.g., reward magnitude).

        Returns
        -------
        dict
            Updated state dictionary including:
                - 'rewards': list of rewards for each stimulus.
                - 'context_reward': average reward across stimuli.
        """

        random_numbers = [rnd.random() for _ in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward
        state['context_reward'] = np.average(state['rewards'])

        return state

    def compute_prediction_error(self, state: dict) -> dict:
        
        """
        Compute Q-value and contextual prediction errors.

        Parameters
        ----------
        state : dict
            Dictionary with keys:
                - 'rewards': list of received rewards.
                - 'context_value': contextual value estimate.
                - 'q_values': list or tensor of Q-values.

        Returns
        -------
        dict
            Updated state dictionary including:
                - 'prediction_errors': prediction errors for Q-values.
                - 'context_prediction_errors': prediction error for contextual value.
        """

        if self.training == 'torch':
            state['prediction_errors'] = state['rewards'] - state['context_value'] - state['q_values'].detach()
            state['context_prediction_errors'] = state['context_reward'] - state['context_value'].detach()
        else:
            state['prediction_errors'] = [state['rewards'][i] - state['context_value'][0] - state['q_values'][i] for i in range(len(state['rewards']))]
            state['context_prediction_errors'] = state['context_reward'] - state['context_value']

        return state

    def select_action(self, state: dict) -> dict:
        
        """
        Select an action using softmax policy over Q-values.

        Parameters
        ----------
        state : dict
            Dictionary with keys:
                - 'q_values': list or tensor of Q-values.
                - 'temperature': temperature parameter (used from self).
                - optionally 'correct_action': index of correct action.

        Returns
        -------
        dict
            Updated state dictionary including:
                - 'action': selected action index.
                - 'accuracy': 1 if selected action matches correct action, else 0 (if applicable).
        """

        if self.training == 'torch':
            transformed_q_values = torch.exp(torch.div(state['q_values'], self.temperature))
            probability_q_values = torch.cumsum(transformed_q_values / torch.sum(transformed_q_values), dim=0)
            state['action'] = self.torch_select_action(probability_q_values)
        else:
            transformed_q_values = np.exp(np.divide(state['q_values'], self.temperature))
            probability_q_values = (transformed_q_values / np.sum(transformed_q_values)).cumsum()
            state['action'] = np.where(probability_q_values >= rnd.random())[0][0]

        if 'correct_action' in state:
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    # Run trial functions
    def forward(self, state: dict, phase: str = 'learning') -> None:
        
        """
        Execute the forward pass for a single trial.

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
            Trial state dictionary.
        phase : str, optional
            Phase of the trial ('learning' or other), by default 'learning'.

        Returns
        -------
        dict
            Updated state after forward pass.
        """

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

    def sim_forward(self, state: dict, phase: str = 'learning') -> None:
        
        """
        Run the forward pass for simulation purposes.

        Parameters
        ----------
        state : dict
            Trial state dictionary.
        phase : str, optional
            Phase of the trial ('learning' or other), by default 'learning'.

        Returns
        -------
        None
        """

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

    def fit_func(self, x: list, *args) -> float:
        
        """
        Fit the model to the data by optimizing parameters.

        Parameters
        ----------
        x : list
            List of parameters: factual_lr, counterfactual_lr, contextual_lr, temperature, plus optionals.
        *args : tuple
            Additional arguments required for fitting (e.g., data).

        Returns
        -------
        float
            Negative log likelihood of all observed actions.
        """

        # Reset indices on succeeding fits
        self.reset_datalists()

        # Unpack free parameters
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature, *optionals = x
        self.unpack_optionals(optionals)

        # Return negative log likelihood of all observed actions
        return -self.fit_task(args, 'q_values', context_reward=True)

    def sim_func(self, *args):
        
        """
        Simulate the model.

        Parameters
        ----------
        *args : tuple
            Arguments for simulation.

        Returns
        -------
        Any
            Simulation results.
        """

        return self.sim_task(args, context_reward=True)