from typing import Union
import numpy as np
import pandas as pd
import scipy
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import random as rnd

class RLToolbox:

    """
    Reinforcement Learning Toolbox contains general RL methods that are consistent across different RL models
    """

    #Setup functions
    def load_methods(self, methods: dict) -> None:

        """
        Load methods into the class instance

        Parameters
        ----------
        methods: dict
            Dictionary of methods to load into the class instance    

        Returns
        -------
        None
        """

        for key in methods:
            setattr(self, key, methods[key])

    def get_value_type(self, model_name: str) -> str:

        """
        Get the value type based on the model name
        
        Parameters
        ----------
        model_name: str
            Name of the model to determine the value type
        
        Returns
        -------
        str
            The value type corresponding to the model name
        """

        if model_name == 'QLearning' or model_name == 'Relative' or model_name == 'wRelative':
            return 'q_values'
        elif model_name == 'ActorCritic':
            return 'w_values'
        elif model_name == 'Hybrid2012' or model_name == 'Hybrid2021':
            return 'h_values'
        
    def get_context_reward(self, model_name: str) -> bool:

        """
        Check if the model uses context reward

        Parameters
        ----------
        model_name: str
            Name of the model to check for context reward

        Returns
        -------
        bool
            True if the model uses context reward, False otherwise
        """

        if model_name == 'Relative':
            return True
        else:
            return False

    def define_parameters(self) -> tuple:

        """
        Define the parameters of the model based on the class name and optional parameters

        Returns
        -------
        tuple
            A tuple containing:
            - value_type: str, the type of value used in the model (e.g., 'q_values', 'w_values', 'h_values')
            - context_reward: bool, whether the model uses context reward
            - transform_reward: bool, whether the model transforms rewards (e.g., applies valence bias)
        """

        # Re-Assign parameters
        self.free_params = self.parameters.copy()
        del self.parameters

        # Get model specifics
        value_type = self.get_value_type(self.__class__.__name__)
        context_reward = self.get_context_reward(self.__class__.__name__)
        transform_reward = self.optional_parameters['bias']

        # Assign parameters using torch
        self.unpack_parameters(self.free_params)

        return value_type, context_reward, transform_reward
    
    def unpack_parameters(self, free_params: dict) -> None:

        """ 
        Unpack the free parameters into the class instance
        
        Parameters
        ----------
        free_params: dict
            Dictionary of free parameters to unpack into the class instance

        Returns
        -------
        None
        """

        for key, value in free_params.items():
            value_torch = nn.Parameter(torch.tensor(value, dtype=torch.float32, requires_grad=True))
            setattr(self, key, value_torch)
            self.register_parameter(key, value_torch)

    def unpack_optionals(self, optionals: list) -> None:

        """
        Unpack optional parameters into the class instance
        
        Parameters
        ----------
        optionals: list
            List of optional parameters to unpack into the class instance
        
        Returns
        -------
        None
        """

        if self.optional_parameters['bias'] == True and 'valence_factor' in self.parameters:
            self.valence_factor = optionals[0]
            optionals = optionals[1:]
        if self.optional_parameters['novel'] == True and 'novel_value' in self.parameters:
            self.novel_value = optionals[0]
            optionals = optionals[1:]
        if self.optional_parameters['decay'] == True and 'decay_factor' in self.parameters:
            self.decay_factor = optionals[0]
            optionals = optionals[1:]

    def clamp_parameters(self, bounds: dict) -> None:

        """
        Clamp the free parameters to stay within the specified bounds

        Parameters
        ----------
        bounds: dict
            Dictionary containing the bounds for each free parameter
        """
        for param_name in self.free_params:
            param = getattr(self, param_name)
            param.data.clamp_(bounds[param_name][0], bounds[param_name][1])

    #Extraction functions
    def get_q_value(self, state: dict) -> dict:

        """
        Get the Q-values for the given state
        
        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'
        
        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'q_values' containing the Q-values for the state
        """
        
        state['q_values'] = self.q_values[state['state_id']]
        return state
    
    def get_v_value(self, state: dict) -> dict:

        """
        Get the V-values for the given state

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'
        
        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'v_values' containing the V-values for the state
        """

        state['v_values'] = self.v_values[state['state_id']]
        return state
    
    def get_w_value(self, state: dict) -> dict:

        """
        Get the W-values for the given state

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'

        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'w_values' containing the W-values for the state
        """

        state['w_values'] = self.w_values[state['state_id']]
        return state

    def get_c_value(self, state: dict) -> dict:

        """
        Get the C-values for the given state

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'
        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'c_values' containing the C-values for the state
        """

        state['c_values'] = self.c_values[state['state_id']]
        return state
    
    def get_m_value(self, state: dict) -> dict:

        """
        Get the M-values for the given state, which is a mix of Q-values and C-values

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'

        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'm_values' containing the M-values for the state
        """

        state['m_values'] = [(state['q_values'][i] * (1-self.mixing_factor)) + (state['c_values'][i] * self.mixing_factor) for i in range(len(state['q_values']))]
        return state
    
    def get_h_values(self, state: dict) -> dict:

        """
        Get the H-values for the given state, which is a mix of W-values and Q-values

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'

        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'h_values' containing the H-values for the state
        """

        if self.training == 'torch':
            state['h_values'] = state['w_values'] * (1-self.mixing_factor) + state['q_values'] * self.mixing_factor
        else:
            state['h_values'] = [(state['w_values'][i] * (1-self.mixing_factor)) + (state['q_values'][i] * self.mixing_factor) for i in range(len(state['w_values']))]
        return state

    def get_final_q_values(self, state: dict) -> dict:

        """
        Get the final Q-values for the given state, applying decay if specified

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'stim_id' and 'state_id'
        
        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'q_values' containing the final Q-values for the state
        """

        if self.optional_parameters['decay']:
            state['q_values'] = self.get_decayed_q_values(state)
        else:
            if self.training == 'torch':
                state['q_values'] = torch.stack([self.final_q_values[stim] for stim in state['stim_id']])
            else:
                state['q_values'] = [self.final_q_values[stim].values[0] for stim in state['stim_id']]
        return state

    def get_final_c_values(self, state: dict) -> dict:

        """
        Get the final C-values for the given state, applying decay if specified

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'stim_id' and 'state_id'

        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'c_values' containing the final C-values for the state
        """

        if self.training == 'torch':
            state['c_values'] = torch.stack([self.final_c_values[stim] for stim in state['stim_id']])
        else:
            state['c_values'] = [self.final_c_values[stim].values[0] for stim in state['stim_id']]
        return state

    def get_context_value(self, state: dict) -> dict:

        """
        Get the context value for the given state

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id'

        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'context_value' containing the context value for the state
        """

        state['context_value'] = self.context_values[state['state_id']]
        return state
    
    def get_final_w_values(self, state: dict) -> dict:

        """
        Get the final W-values for the given state, applying decay if specified

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'stim_id' and 'state_id'
        
        Returns
        -------
        state: dict
            The input state dictionary with an additional key 'w_values' containing the final W-values for the state
        """

        if self.training == 'torch':
            state['w_values'] = torch.stack([self.final_w_values[stim] for stim in state['stim_id']])
        else:
            state['w_values'] = [self.final_w_values[stim].values[0] for stim in state['stim_id']]
        return state
    
    def get_decayed_q_values(self, state: dict) -> list:

        """
        Get the decayed Q-values for the given state, combining final and initial Q-values based on decay factor

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'stim_id' and 'state_id'

        Returns
        -------
        list
            List of decayed Q-values for the stimuli in the state
        """

        if self.training == 'torch':
            q_final = torch.stack([self.final_q_values[stim] for stim in state['stim_id']])
            q_initial = torch.stack([torch.tensor(self.initial_q_values[stim][0]) for stim in state['stim_id']])
            q_values = q_final*(1-self.decay_factor) + q_initial*(self.decay_factor)
        else:
            q_final = [self.final_q_values[stim] for stim in state['stim_id']]
            q_initial = [self.initial_q_values[stim] for stim in state['stim_id']]
            q_final_decayed = [q*(1-self.decay_factor) for q in q_final]
            q_initial_decayed = [q*(self.decay_factor) for q in q_initial]
            q_values = [q_final_decayed[i].values[0] + q_initial_decayed[i].values[0] for i in range(len(q_final))]
        return q_values

    def torch_select_action(self, probabilities: torch.Tensor) -> int:

        """
        Select an action based on the given probabilities using a random number generator

        Parameters
        ----------
        probabilities: torch.Tensor
            Tensor containing the probabilities for each action
        
        Returns
        -------
        int
            The index of the selected action
        """

        number_of_tries = 0
        while True:
            random_number = rnd.random()
            action = torch.where(probabilities >= random_number)
            
            if len(action[0]) > 0:
                return action[0][0]
            else:
                #Debugging
                number_of_tries += 1
                print('*******************************')
                print(f'Action selection failed {number_of_tries} times, trying again...')
                print(f'Probabilities: {probabilities}')
                print(f'Random number: {random_number}')
                print(f'Action: {action}')

                if number_of_tries > 10:                
                    filename = f'RL/fits/temp/ERROR_{self.model_name}_{self.participant_id}_Run{self.run}_fit_results.csv'
                    pd.DataFrame(probabilities).to_csv(filename)
                    raise ValueError(f'Action selection failed too many times, check the error file, {filename}, for more information')

    #Update functions
    def update_prediction_errors(self, state: dict) -> None:

        """
        Update the prediction errors for the given state

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id' and prediction errors

        Returns
        -------
        None
        """

        if 'Hybrid' in self.__class__.__name__:
            self.q_prediction_errors[state['state_id']] = state['q_prediction_errors'].detach() if self.training == 'torch' else state['q_prediction_errors']
            self.v_prediction_errors[state['state_id']] = state['v_prediction_errors'].detach() if self.training == 'torch' else state['v_prediction_errors']
        else:
            self.prediction_errors[state['state_id']] = state['prediction_errors'].detach() if self.training == 'torch' else state['prediction_errors']

    def update_q_values(self, state: dict) -> None:

        """
        Update the Q-values for the given state based on the prediction errors and learning rates

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'q_values', 'action', and prediction errors

        Returns
        -------
        None
        """

        if self.training == 'torch':
            if 'Hybrid' in self.__class__.__name__:
                prediction_errors = state['q_prediction_errors']
            elif self.__class__.__name__ == 'Relative' or self.__class__.__name__ == 'wRelative':
                prediction_errors = state['prediction_errors']
            else:
                prediction_errors = state['prediction_errors'].detach()

            if 'Standard' in self.model_name:
                self.q_values[state['state_id']] = state['q_values'].detach()[state['action']] + (self.factual_lr * prediction_errors[state['action']])
            else:
                learning_rates = torch.stack([self.factual_lr, self.counterfactual_lr]) if state['action'] == 0 else torch.stack([self.counterfactual_lr, self.factual_lr])
                self.q_values[state['state_id']] = state['q_values'].detach() + (learning_rates * prediction_errors)
        else:
            if 'Standard' in self.model_name:
                prediction_errors = state['q_prediction_errors'][state['action']] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
                self.q_values[state['state_id']][state['action']] = state['q_values'][state['action']] + (self.factual_lr * prediction_errors)
            else:
                learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
                prediction_errors = state['q_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
                self.q_values[state['state_id']] = [state['q_values'][i] + (learning_rates[i] * prediction_errors[i]) for i in range(len(state['q_values']))]

    def update_w_values(self, state: dict) -> None:

        """
        Update the W-values for the given state based on the prediction errors and learning rates

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'w_values', 'action', and prediction errors

        Returns
        -------
        None
        """

        if self.training == 'torch':
            if 'Standard' in self.model_name:
                learning_rates = torch.stack([self.factual_actor_lr, self.factual_actor_lr])
                prediction_errors = state['v_prediction_errors'][state['action']] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
                new_w_values = state['w_values'].detach() + (self.factual_actor_lr * prediction_errors)
            else:
                learning_rates = torch.stack([self.factual_actor_lr, self.counterfactual_actor_lr]) if state['action'] == 0 else torch.stack([self.counterfactual_actor_lr, self.factual_actor_lr])
                prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
                new_w_values = state['w_values'].detach() + (learning_rates * prediction_errors)
        else:
            if 'Standard' in self.model_name:
                prediction_errors = state['v_prediction_errors'][state['action']] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
                new_w_values = state['w_values']
                new_w_values[state['action']] = state['w_values'][state['action']] + (self.factual_actor_lr * prediction_errors)
            else:
                learning_rates = [self.factual_actor_lr, self.counterfactual_actor_lr] if state['action'] == 0 else [self.counterfactual_actor_lr, self.factual_actor_lr]
                prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
                new_w_values = [state['w_values'][i] + (learning_rates[i] * prediction_errors[i]) for i in range(len(state['w_values']))]

        #Check if the new w values are all zeros, and adjust them to initial values if so
        if self.training == 'torch':
            if torch.sum(torch.abs(new_w_values)).item() == 0:
                new_w_values = torch.tensor([0.01]*len(new_w_values), dtype=torch.float32)
            new_w_values = new_w_values/torch.sum(torch.abs(new_w_values))
        else:
            if np.sum([np.abs(new_w_values[i]) for i in range(len(new_w_values))]) == 0:
                new_w_values = [0.01]*len(new_w_values)
            new_w_values = new_w_values/np.sum(np.abs(new_w_values))

        self.w_values[state['state_id']] = new_w_values
    
    def update_v_values(self, state: dict) -> None:

        """
        Update the V-values for the given state based on the prediction errors and learning rates

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'v_values', 'action', and prediction errors
        
        Returns
        -------
        None
        """

        if self.training == 'torch':
            prediction_errors = state['v_prediction_errors'].detach() if 'Hybrid' in self.__class__.__name__ else state['prediction_errors'].detach()
            self.v_values[state['state_id']] = state['v_values'].detach() + (self.critic_lr * prediction_errors)
        else:
            prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
            self.v_values[state['state_id']] = [state['v_values'][0] + (self.critic_lr * prediction_errors[state['action']])]
    
    def update_h_values(self, state: dict) -> None:

        """
        Update the H-values for the given state based on the W-values and Q-values

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'h_values', 'w_values', and 'q_values'
        
        Returns
        -------
        None
        """

        self.h_values[state['state_id']] = state['h_values'].detach() if self.training == 'torch' else state['h_values']

    def update_c_values(self, state: dict) -> None:

        """
        Update the C-values for the given state based on the prediction errors and learning rates

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'c_values', 'action', and prediction errors

        Returns
        -------
        None
        """

        if self.training == 'torch':
            learning_rates = torch.stack([self.factual_lr, self.counterfactual_lr]) if state['action'] == 0 else torch.stack([self.counterfactual_lr, self.factual_lr])
            self.c_values[state['state_id']] = state['c_values'].detach() + (learning_rates * state['c_prediction_errors'].detach())
        else:
            learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
            self.c_values[state['state_id']] = [state['c_values'][i] + (learning_rates[i] * state['c_prediction_errors'][i]) for i in range(len(state['c_values']))]

    def update_context_values(self, state: dict) -> dict:

        """
        Update the context values for the given state based on the context prediction errors and learning rate

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'context_value', and 'context_prediction_errors'
        
        Returns
        -------
        state: dict
            The input state dictionary with an updated key 'context_value' containing the updated context values for the state
        """

        if self.training == 'torch':
            self.context_values[state['state_id']] = state['context_value'].detach() + (self.contextual_lr * state['context_prediction_errors'])
        else:
            self.context_values[state['state_id']] = state['context_value'] + (self.contextual_lr * state['context_prediction_errors'])
        state = self.get_context_value(state)
        return state
    
    def update_context_prediction_errors(self, state: dict) -> None:

        """
        Update the context prediction errors for the given state
        
        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id' and 'context_prediction_errors'

        Returns
        -------
        None
        """

        self.context_prediction_errors[state['state_id']] = state['context_prediction_errors'].detach() if self.training == 'torch' else state['context_prediction_errors']

    def reward_valence(self, reward: list) -> list:

        """
        Apply valence bias to the rewards based on the valence factor

        Parameters
        ----------
        reward: list
            List of rewards to apply valence bias to

        Returns
        -------
        list
            List of rewards after applying valence bias
        """

        new_reward = []
        for ri, r in enumerate(reward):
            if r > 0:
                new_reward.append((1-self.valence_factor)*r)
            elif r < 0:
                new_reward.append(self.valence_factor*r)
            else:
                new_reward.append(r)
        return torch.stack(new_reward) if self.training == 'torch' else new_reward
    
    def update_model(self, state: dict) -> None:

        """
        Update the model based on the current state

        Parameters
        ----------
        state: dict
            Dictionary containing the state information, including 'state_id', 'q_values', 'v_values', 'w_values', and prediction errors

        Returns
        -------
        None
        """
        
        self.update_prediction_errors(state)

        if self.__class__.__name__ != 'ActorCritic':
            self.update_q_values(state)

        if self.__class__.__name__ == 'Relative':
            self.update_context_values(state)
            self.update_context_prediction_errors(state)

        if self.__class__.__name__ == 'ActorCritic' or 'Hybrid' in self.__class__.__name__:
            self.update_w_values(state)
            self.update_v_values(state)
        
        if 'Hybrid' in self.__class__.__name__:
            self.update_h_values(state)
    
    def reset_datalists(self) -> None:
        
        """
        Reset the data lists for all states in the model
        This method clears the prediction errors, Q-values, W-values, V-values, context values, and H-values for each state.

        Returns
        -------
        None
        """

        for s in self.states:

            if 'Hybrid' in self.__class__.__name__:
                self.q_prediction_errors[s] = [0]*len(self.q_prediction_errors[s])
                self.v_prediction_errors[s] = [0]*len(self.v_prediction_errors[s])
            else:
                self.prediction_errors[s] = [0]*len(self.prediction_errors[s])

            if self.__class__.__name__ != 'ActorCritic':
                self.q_values[s] = [0]*len(self.q_values[s])

            if self.__class__.__name__ == 'Relative':
                self.context_values[s] = [0]*len(self.context_values[s])
                self.context_prediction_errors[s] = [0]*len(self.context_prediction_errors[s])

            if self.__class__.__name__ == 'ActorCritic' or 'Hybrid' in self.__class__.__name__: 
                self.w_values[s] = [0.01]*len(self.w_values[s])
                self.v_values[s] = [0]*len(self.v_values[s])
                
            if 'Hybrid' in self.__class__.__name__:
                self.h_values[s] = [0]*len(self.h_values[s])

    def reset_datalists_torch(self) -> None:

        """
        Reset the data lists for all states in the model using torch tensors
        This method clears the prediction errors, Q-values, W-values, V-values, context values, and H-values for each state.

        Returns
        -------
        None
        """

        for s in self.states:
            if 'Hybrid' in self.__class__.__name__:
                self.q_prediction_errors[s] = torch.zeros(len(self.q_prediction_errors[s]))
                self.v_prediction_errors[s] = torch.zeros(len(self.v_prediction_errors[s]))
            else:
                self.prediction_errors[s] = torch.zeros(len(self.prediction_errors[s]))

            if self.__class__.__name__ != 'ActorCritic':
                self.q_values[s] = torch.zeros(len(self.q_values[s]))

            if self.__class__.__name__ == 'Relative':
                self.context_values[s] = torch.zeros(len(self.context_values[s]))
                self.context_prediction_errors[s] = torch.zeros(len(self.context_prediction_errors[s]))

            if self.__class__.__name__ == 'ActorCritic' or 'Hybrid' in self.__class__.__name__:
                self.w_values[s] = torch.full((len(self.w_values[s]),), 0.01)  # Initializes all values to 0.01
                self.v_values[s] = torch.zeros(len(self.v_values[s]))

            if 'Hybrid' in self.__class__.__name__:
                self.h_values[s] = torch.zeros(len(self.h_values[s]))

    def detach_values(self, values: dict) -> dict:
        """
        Detach the values from the computation graph and convert them to a list
        
        Parameters
        ----------
        values: dict
            Dictionary containing the values to detach and convert

        Returns
        -------
        dict
            Dictionary with the same keys as input, but values are detached and converted to lists
        """

        return {state: [x.item() for x in values[state]] for state in values.keys()}

    def attach_values(self, values: dict) -> dict:

        """
        Attach the values to the computation graph and convert them to torch tensors
        
        Parameters
        ----------
        values: dict
            Dictionary containing the values to attach and convert

        Returns
        -------
        dict
            Dictionary with the same keys as input, but values are converted to torch tensors
        """

        return {key: torch.tensor(value, dtype=torch.float32) for key, value in values.items()}
    
    def combine_values(self) -> None:

        """
        Combine the values of the model after each training epoch
        This method is called after the learning trials to combine the Q-values, W-values, V-values, and H-values into their final forms for the transfer trials.
        It ensures that the values are properly updated and ready for the next training phase or evaluation.

        Returns
        -------
        None
        """

        #Inter-phase processing
        if self.__class__.__name__ == 'ActorCritic':
            self.combine_v_values()
            self.combine_w_values()
        elif 'Hybrid' in self.__class__.__name__:
            self.combine_q_values()
            self.combine_v_values()
            self.combine_w_values()
        else:
            self.combine_q_values()

    def fit_log_likelihood(self, values: list) -> Union[torch.Tensor, np.ndarray]:

        """
        Calculate the log likelihood of the observed actions given the values
        
        Parameters
        ----------
        values: list
            List of values (Q-values, W-values, etc.) for the actions taken
            
        Returns
        -------
        torch.Tensor | np.ndarray
            The negative log likelihood of the observed actions given the values, transformed by the temperature parameter
        """

        if self.training == 'torch':
            transformed_values = torch.exp(torch.divide(values, self.temperature))
            probability_values = transformed_values / torch.sum(transformed_values)
            uniform_dist = torch.ones((len(probability_values)))/len(probability_values)
        else:
            transformed_values = np.exp(np.divide(values, self.temperature))
            probability_values = (transformed_values/np.sum(transformed_values))
            uniform_dist = np.ones((len(probability_values)))/len(probability_values)

        if 'Hybrid2021' in self.__class__.__name__:
            probability_values = (((1-self.noise_factor)*probability_values).T + (self.noise_factor*uniform_dist)).T

        return -torch.log(probability_values) if self.training == 'torch' else -np.log(probability_values)
        
    def fit_torch(self, data: tuple, bounds: dict) -> tuple:

        """
        Fit the model to the data using PyTorch
        
        Parameters
        ----------
        data: tuple
            A tuple containing the learning data and transfer data
        bounds: dict    
            Dictionary containing the bounds for each free parameter

        Returns
        -------
        best_fit: float
            The best fit value obtained during the training
        fitted_params: dict
            Dictionary containing the fitted parameters after training
        """

        # TODO:
        # Check scipy NLL (should I minus max like nn.CrossEntropyLoss?)

        # Define parameters
        value_type, context_reward, transform_reward = self.define_parameters()

        # Set data
        learning_data, transfer_data = data
        learning_states, learning_actions, learning_rewards = learning_data
        transfer_states, transfer_actions = transfer_data
        
        # Training loop
        if not self.multiprocessing:
            loop = tqdm.tqdm(range(self.training_epochs), leave=True)
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_lr)
        loss_fn = nn.CrossEntropyLoss()
        all_losses = []

        best_fit = np.inf
        fitted_params = None
        for epoch in range(self.training_epochs):
            # Reset data lists
            self.reset_datalists_torch()
            losses = []

            # Learning phase
            for trial, (state_id, action, reward) in enumerate(zip(learning_states.copy(), learning_actions.copy(), learning_rewards.copy())):
                action = torch.tensor(action, dtype=torch.long, requires_grad=False)
                reward = torch.tensor(reward, dtype=torch.float32, requires_grad=False)

                # Forward pass                
                state = {'rewards': reward, 'action': action, 'state_id': state_id}
                                
                if transform_reward:
                    state['rewards'] = self.reward_valence(state['rewards'])
                
                if context_reward:
                    state['context_reward'] = torch.mean(reward)

                # Forward
                state = self.fit_forward(state)

                # Compute loss
                action_values = torch.divide(state[value_type], self.temperature) 
                loss = loss_fn(action_values, action)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update model
                self.fit_model_update(state)

                # Clamp parameters to stay within bounds
                self.clamp_parameters(bounds)

                # Track loss
                losses.append(loss.detach().item())

            #Inter-phase processing  
            self.combine_values()
                        
            #Transfer phase
            for trial, (state_id, action) in enumerate(zip(transfer_states.copy(), transfer_actions.copy())):
                action = torch.tensor(action, dtype=torch.long, requires_grad=False)
                
                #Populate state
                state = {'action': action, 
                        'stim_id': [stim for stim in state_id.split(' ')[-1]]}
                
                # Forward
                state = self.fit_forward(state, phase='transfer')

                # Compute and store the log probability of the observed action
                action_values = torch.divide(state[value_type], self.temperature)
                loss = loss_fn(action_values, action)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clamp parameters to stay within bounds
                self.clamp_parameters(bounds)
            
                # Track loss
                losses.append(loss.detach().item())

            all_losses.append(np.sum(losses))
            if not self.multiprocessing:
                loop.update(1)
                loop.set_postfix_str(f'loss: {all_losses[-1]:.0f}')

            if np.sum(losses) < best_fit:
                best_fit = np.sum(losses)
                fitted_params = {name: param.item() for name, param in self.named_parameters()}
        
        return best_fit, fitted_params
    
    def fit(self, data: tuple, bounds: dict) -> tuple:

        """
        Fit the model to the data using scipy optimization
        
        Parameters
        ----------
        data: tuple
            A tuple containing the learning data and transfer data
        bounds: dict
            Dictionary containing the bounds for each free parameter
            
        Returns
        -------
        fit_results: OptimizeResult
            The result of the optimization process containing the fitted parameters and other information
        fitted_params: dict
            Dictionary containing the fitted parameters after optimization
        """

        #Extract the values from dict
        free_params = list(self.parameters.values())
        param_bounds = list(bounds.values())

        #Fit the model
        fit_results = scipy.optimize.minimize(self.fit_func, 
                                              free_params, 
                                              args=data,
                                              bounds=param_bounds, 
                                              method='L-BFGS-B')
        
        #Warning if fit failed
        if not fit_results.success:
            Warning(f'Fit failed for {self.model_name} model: {fit_results.message}')
        
        #Unpack the fitted params
        fitted_params = {}
        for fi, key in enumerate(self.parameters):
            fitted_params[key] = fit_results.x[fi]

        return fit_results, fitted_params

    def fit_task(self,
                 args: tuple,
                 value_type: str,
                 transform_reward: bool = False,
                 context_reward: bool = False) -> float:

        """
        Fit the model to the task data, including both learning and transfer phases
        
        Parameters
        ----------
        args: tuple
            A tuple containing the learning data and transfer data
        value_type: str
            The type of values to use for fitting (e.g., 'q_values', 'w_values', etc.)
        transform_reward: bool, optional
            A boolean as to whether rewards will be transformed
        context_reward: bool, optional
            A boolean as to whether the context_reward should be computed
        """
        
        #Unpack data
        learning_data, transfer_data = args[0]
        learning_states, learning_actions, learning_rewards = learning_data
        transfer_states, transfer_actions = transfer_data

        #Initialize values
        log_likelihood = 0

        #Learning phase
        for trial, (state_id, action, reward) in enumerate(zip(learning_states.copy(), learning_actions.copy(), learning_rewards.copy())):

            #Populate state
            if transform_reward:
                reward = self.reward_valence(reward)

            state = {'rewards': reward, 
                    'action': action, 
                    'state_id': state_id}
            
            if context_reward:
                state['context_reward'] = np.average(reward)
            
            #Forward
            state = self.fit_forward(state)

            #Compute and store the log probability of the observed action
            log_likelihood -= self.fit_log_likelihood(state[value_type])[action]
  
        #Inter-phase processing
        self.combine_values()

        #Transfer phase
        #Toggle to switch between methods for testing. Function method is slower than loop, so it's avoided
        if False:
            log_likelihood -= self.fit_transfer_forward(transfer_data, value_type, reduced = True)
        else:
            for trial, (state_id, action) in enumerate(zip(transfer_states.copy(), transfer_actions.copy())):
                
                #Populate state
                state = {'action': action, 
                        'stim_id': [stim for stim in state_id.split(' ')[-1]]}
                
                #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower 
                state = self.fit_forward(state, phase='transfer')

                #Compute and store the log probability of the observed action
                log_likelihood -= self.fit_log_likelihood(state[value_type])[action]

        return log_likelihood
    
    def simulate(self, data: tuple) -> None:

        """
        Simulate the model on the provided data
        
        Parameters
        ----------
        data: tuple
            A tuple containing the learning data and transfer data

        Returns
        -------
        None
        """

        self.sim_func(data)

    def sim_task(self,
                 args: tuple, 
                 transform_reward: bool = False, 
                 context_reward: bool = False) -> tuple:

        """
        Simulate the model on the task data, including both learning and transfer phases

        Parameters
        ----------
        args: tuple
            A tuple containing the learning data and transfer data
        transform_reward: bool, optional
            A boolean indicating whether rewards should be transformed
        context_reward: bool, optional
            A boolean indicating whether context rewards should be computed

        Returns
        -------
        task_learning_data: pd.DataFrame
            DataFrame containing the learning phase data with model predictions
        task_transfer_data: pd.DataFrame
            DataFrame containing the transfer phase data with model predictions
        """

        #Unpack data
        learning_data, transfer_data = args[0]['learning'], args[0]['transfer']

        #Learning phase
        for trial, trial_data in learning_data.copy().iterrows():

            #Populate rewards
            rewards = [int(trial_data['reward_L']), int(trial_data['reward_R'])]
            if transform_reward:
                rewards = self.reward_valence(rewards)

            #Populate state
            stims = ['symbol_L_name', 'symbol_R_name'] if trial_data['stim_order'] else ['symbol_R_name', 'symbol_L_name']
            stims_index = [0, 1] if trial_data['stim_order'] else [1, 0]
            state = {'block_number': 0, 
                     'trial_number': trial_data['trial_number'], 
                     'state_index': 0, 
                     'state_id': trial_data['state'], 
                     'stim_id': [trial_data['state'].split(' ')[-1][stims_index[0]], trial_data['state'].split(' ')[-1][stims_index[1]]], 
                     'context': trial_data['context_val_name'], 
                     'feedback': [trial_data['feedback_L'], trial_data['feedback_R']], 
                     'probabilities': [trial_data[stims[0]][:2], trial_data[stims[1]][:2]],
                     'rewards': rewards,
                     'correct_action': 0}
            
            if context_reward:
                state['context_reward'] = np.average(state['rewards'])
            
            #Forward
            self.sim_forward(state)

        #Inter-phase processing
        self.combine_values()

        #Transfer phase
        for trial, trial_data in transfer_data.copy().iterrows():
            
            #Populate state
            state = {'block_number': 0,
                     'trial_number': trial,
                     'state_id': trial_data['state'],
                     'stim_id': trial_data['state'].split('State ')[-1]}
            
            #Forward
            self.sim_forward(state, phase='transfer')
        
        return self.task_learning_data, self.task_transfer_data

    def get_choice_rates(self) -> pd.DataFrame:

        """ 
        Get the choice rates for each state in the model

        Returns
        -------
        pd.DataFrame
            DataFrame containing the choice rates for each state, with columns 'state_id', 'choice_rate', and 'choice_count'
        """

        return self.choice_rate
