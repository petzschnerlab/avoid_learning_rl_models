import random as rnd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

class RLToolbox:

    """
    Reinforcement Learning Toolbox contains general RL methods that are consistent across different RL models
    """

    #Setup functions
    def load_methods(self, methods):
        for key in methods:
            setattr(self, key, methods[key])

    #Extraction functions
    def get_q_value(self, state):
        state['q_values'] = list(self.q_values[state['state_id']].iloc[-1])
        return state
    
    def get_v_value(self, state):
        state['v_values'] = list(self.v_values[state['state_id']].iloc[-1])
        return state
    
    def get_w_value(self, state):
        state['w_values'] = list(self.w_values[state['state_id']].iloc[-1])
        return state

    def get_final_q_values(self, state):
        state['q_values'] = [float(self.final_q_values[stim]) for stim in state['stim_id']]
        return state
    
    def get_final_w_values(self, state):
        state['w_values'] = [float(self.final_w_values[stim]) for stim in state['stim_id']]
        return state
    
    #Update functions
    def update_prediction_errors(self, state):

        if self.__class__.__name__ == 'Hybrid':
            self.q_prediction_errors[state['state_id']] = pd.concat([self.q_prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['q_prediction_errors']], 
                                                                            columns=self.q_prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)
            self.v_prediction_errors[state['state_id']] = pd.concat([self.v_prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['v_prediction_errors']], 
                                                                            columns=self.v_prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)
        else:
            self.prediction_errors[state['state_id']] = pd.concat([self.prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['prediction_errors']], 
                                                                            columns=self.prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)

    def update_q_values(self, state):

        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
        prediction_errors = state['q_prediction_errors'] if self.__class__.__name__ == 'Hybrid' else state['prediction_errors']
            
        new_q_values = []
        for i in range(len(state['rewards'])):
            new_q_values.append(state['q_values'][i] + (learning_rates[i] * prediction_errors[i]))

        self.q_values[state['state_id']] = pd.concat([self.q_values[state['state_id']],
                                                      pd.DataFrame([new_q_values], 
                                                                   columns=self.q_values[state['state_id']].columns)], 
                                                    ignore_index=True)
        
    def update_w_values(self, state):

        learning_rates = [self.factual_actor_lr, self.counterfactual_actor_lr] if state['action'] == 0 else [self.counterfactual_actor_lr, self.factual_actor_lr]
        prediction_errors = state['v_prediction_errors'] if self.__class__.__name__ == 'Hybrid' else state['prediction_errors']
            
        new_w_values = []
        for i in range(len(state['rewards'])):
            new_w_values.append(state['w_values'][i] + (learning_rates[i] * prediction_errors[i]))
        new_w_values = [w_val/np.sum(np.abs(new_w_values)) for w_val in new_w_values]

        self.w_values[state['state_id']] = pd.concat([self.w_values[state['state_id']],
                                                    pd.DataFrame([new_w_values], 
                                                                columns=self.w_values[state['state_id']].columns)], 
                                                    ignore_index=True)
    
    def update_v_values(self, state):

        learning_rates = [self.factual_critic_lr, self.counterfactual_critic_lr] if state['action'] == 0 else [self.counterfactual_critic_lr, self.factual_critic_lr]
        prediction_errors = state['v_prediction_errors'] if self.__class__.__name__ == 'Hybrid' else state['prediction_errors']

        new_v_values = []
        for i in range(len(state['rewards'])):
            new_v_values.append(state['v_values'][i] + (learning_rates[i] * prediction_errors[i]))

        self.v_values[state['state_id']] = pd.concat([self.v_values[state['state_id']],
                                                      pd.DataFrame([new_v_values], 
                                                                   columns=self.v_values[state['state_id']].columns)], 
                                                    ignore_index=True)
    
    def update_h_values(self, state):

        self.h_values[state['state_id']] = pd.concat([self.h_values[state['state_id']],
                                                      pd.DataFrame([state['h_values']], 
                                                                   columns=self.h_values[state['state_id']].columns)], 
                                                    ignore_index=True)
        
    def update_context_values(self, state):

        new_context_value = state['context_value'] + (self.contextual_lr * state['context_prediction_error'])

        self.context_values[state['state_id']] = pd.concat([self.context_values[state['state_id']], 
                                                            pd.DataFrame([new_context_value], 
                                                                         columns=self.context_values[state['state_id']].columns)], 
                                                            ignore_index=True)
    
    def update_context_prediction_errors(self, state):
        self.context_prediction_errors[state['state_id']] = pd.concat([self.context_prediction_errors[state['state_id']], 
                                                                      pd.DataFrame([state['context_prediction_error']], 
                                                                                   columns=self.context_prediction_errors[state['state_id']].columns)], 
                                                                      ignore_index=True)

    def update_model(self, state):
        
        self.update_prediction_errors(state)

        if self.__class__.__name__ != 'ActorCritic':
            self.update_q_values(state)

        if self.__class__.__name__ == 'Relative':
            self.update_context_values(state)
            self.update_context_prediction_errors(state)

        if self.__class__.__name__ == 'ActorCritic' or self.__class__.__name__ == 'Hybrid':
            self.update_w_values(state)
            self.update_v_values(state)

    #Plotting functions
    def plot_model(self):
    
        fig, ax = plt.subplots(4, 4, figsize=(20,5))

        #Plot q-values
        if self.__class__.__name__ == 'Hybrid':
            values = self.h_values
            value_label = 'H-Value'
        elif self.__class__.__name__ == 'ActorCritic':
            values = self.v_values
            value_label = 'V-Value'
        else:
            values = self.q_values
            value_label = 'Q-Value'
        
        values = self.q_values if self.__class__.__name__ != 'ActorCritic' else self.w_values
        for i, key in enumerate(values.keys()):
            for ci, col in enumerate(values[key].columns):
                rolling_values = values[key][col].reset_index(drop=True).rolling(window=2).mean()
                ax[0,i].plot(rolling_values, label=['Stim 1', 'Stim 2'][ci])
            ax[0,i].set_title(key)
            ax[0,i].set_ylim(-1, 1)
            if i == 0:
                ax[0,i].set_ylabel(value_label)
            ax[0,i].set_xlabel('')
            if i == len(values.keys())-1:
                ax[0,i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax[0,i].set_xticklabels([])
            if i > 0:
                ax[0,i].set_yticklabels([])
        
        #Plot prediction errors
        prediction_errors = self.prediction_errors if self.__class__.__name__ != 'Hybrid' else self.q_prediction_errors
        for i, key in enumerate(prediction_errors.keys()):
            for ci, col in enumerate(prediction_errors[key].columns):
                rolling_prediction_errors = prediction_errors[key][col].reset_index(drop=True).rolling(window=2).mean()
                ax[1,i].plot(rolling_prediction_errors, label=['Stim 1', 'Stim 2'][ci], color=['C0', 'C1'][ci])
                if self.__class__.__name__ == 'Hybrid':
                    rolling_v_prediction_errors = self.v_prediction_errors[key][col].reset_index(drop=True).rolling(window=2).mean()
                    ax[1,i].plot(rolling_v_prediction_errors, label=['Stim 1', 'Stim 2'][ci], color=['C0', 'C1'][ci], linestyle='dashed')
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
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'temperature': self.temperature}

    #RL functions
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward

        return state
    
    def select_action(self, state):

        transformed_q_values = np.exp(np.divide(state['q_values'], self.temperature))
        probability_q_values = (transformed_q_values/np.sum(transformed_q_values)).cumsum()
        state['action'] = np.where(probability_q_values >= rnd.random())[0][0]
        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state
    
    def compute_prediction_error(rewards, state):
        state['prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
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

class ActorCritic(RLToolbox):

    """
    Reinforcement Learning Model: Actor-Critic

    Parameters:
    ------------
    factual_actor_lr: float
        Learning rate for factual actor update
    counterfactual_actor_lr: float
        Learning rate for counterfactual actor update
    factual_critic_lr: float
        Learning rate for factual critic update
    counterfactual_critic_lr: float
        Learning rate for counterfactual critic update
    temperature: float
        Temperature parameter for softmax action selection
    """

    def __init__(self, factual_actor_lr, counterfactual_actor_lr, factual_critic_lr, counterfactual_critic_lr, temperature, valence_factor):
        super().__init__()

        #Set parameters
        self.factual_actor_lr = factual_actor_lr
        self.counterfactual_actor_lr = counterfactual_actor_lr
        self.factual_critic_lr = factual_critic_lr
        self.counterfactual_critic_lr = counterfactual_critic_lr
        self.temperature = temperature
        self.valence_factor = valence_factor
        self.parameters = {'factual_actor_lr': self.factual_actor_lr, 
                           'counterfactual_actor_lr': self.counterfactual_actor_lr, 
                           'factual_critic_lr': self.factual_critic_lr,
                           'counterfactual_critic_lr': self.counterfactual_critic_lr,
                           'temperature': self.temperature,
                           'valence_factor': self.valence_factor}
        
    #RL functions
    def get_reward(self, state):

        """
        TODO:
        "In agreement with previous studies, we also allow positive and negative rewards to be weighed differently. Positive
        feedback at trial t was encoded as outcome(t) = 1-d, neutral feedback as outcome(t) = 0 and negative feedback as
        outcome(t) = -d. Thus the free parameter d indicates full neglect of negative outcomes if d = 0, full neglect of positive
        outcomes if d = 1, and equal weighing of positive and negative outcomes if d = 0.5"
        """
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]
        for ri, r in enumerate(reward):
            if r > 0:
                reward[ri] = 1-self.valence_factor
            elif r < 0:
                reward[ri] = -self.valence_factor
            else:
                reward[ri] = 0

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(rewards, state):
        state['prediction_errors'] = [state['rewards'][i] - state['v_values'][i] for i in range(len(state['rewards']))]
        return state

    def select_action(self, state):

        transformed_w_values = np.exp(np.divide(state['w_values'], self.temperature))
        probability_w_values = (transformed_w_values/np.sum(transformed_w_values)).cumsum()
        state['action'] = np.where(probability_w_values >= rnd.random())[0][0]
        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    #Run trial functions
    def forward(self, state, phase = 'learning'):
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

class Relative(RLToolbox):

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

    def __init__(self, factual_lr, counterfactual_lr, contextual_lr, temperature):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.contextual_lr = contextual_lr
        self.temperature = temperature
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'contextual_lr': self.contextual_lr,
                           'temperature': self.temperature}
        
        #Create context value dataframe
        
    #RL functions
    def get_context_value(self, state):
        state['context_value'] = list(self.context_values[state['state_id']].iloc[-1])
        return state
    
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward
        state['context_reward'] = np.average(state['rewards'])

        return state
    
    def compute_prediction_error(rewards, state):
        state['prediction_errors'] = [state['rewards'][i] - state['context_value'][0] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['context_prediction_error'] = state['context_reward'] - state['context_value']
        return state
    
    def select_action(self, state):

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

class Hybrid(RLToolbox):

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
                 factual_critic_lr, counterfactual_critic_lr, temperature, mixing_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.factual_actor_lr = factual_actor_lr
        self.counterfactual_actor_lr = counterfactual_actor_lr
        self.factual_critic_lr = factual_critic_lr
        self.counterfactual_critic_lr = counterfactual_critic_lr
        self.temperature = temperature
        self.mixing_factor = mixing_factor
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                            'factual_actor_lr': self.factual_actor_lr,
                            'counterfactual_actor_lr': self.counterfactual_actor_lr,
                            'factual_critic_lr': self.factual_critic_lr,
                            'counterfactual_critic_lr': self.counterfactual_critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor}

    #RL functions    
    def get_reward(self, state):

        """
        TODO:
        "In agreement with previous studies, we also allow positive and negative rewards to be weighed differently. Positive
        feedback at trial t was encoded as outcome(t) = 1-d, neutral feedback as outcome(t) = 0 and negative feedback as
        outcome(t) = -d. Thus the free parameter d indicates full neglect of negative outcomes if d = 0, full neglect of positive
        outcomes if d = 1, and equal weighing of positive and negative outcomes if d = 0.5"
        """
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(rewards, state):
        state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][i] for i in range(len(state['rewards']))]
        return state

    def select_action(self, state):

        state['h_values'] = [(state['w_values'][i] * (1-self.mixing_factor)) + (state['q_values'][i] * self.mixing_factor) for i in range(len(state['w_values']))]
        transformed_h_values = np.exp(np.divide(state['h_values'], self.temperature))
        probability_v_values = (transformed_h_values/np.sum(transformed_h_values)).cumsum()
        state['action'] = np.where(probability_v_values >= rnd.random())[0][0]
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
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)


