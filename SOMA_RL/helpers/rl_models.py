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
        self.prediction_errors[state['state_id']] = pd.concat([self.prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['prediction_errors']], 
                                                                            columns=self.prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)

    def update_q_values(self, state):

        if self.__class__.__name__ == 'QLearning' or self.__class__.__name__ == 'Relative':
            learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
        elif self.__class__.__name__ == 'ActorCritic':
            learning_rates = [self.critic_lr, self.critic_lr]

        new_q_values = []
        for i in range(len(state['rewards'])):
            new_q_values.append(state['q_values'][i] + (learning_rates[i] * state['prediction_errors'][i]))

        self.q_values[state['state_id']] = pd.concat([self.q_values[state['state_id']],
                                                      pd.DataFrame([new_q_values], 
                                                                   columns=self.q_values[state['state_id']].columns)], 
                                                    ignore_index=True)
        
    def update_w_values(self, state):
            
        new_w_values = []
        for i in range(len(state['rewards'])):
            new_w_values.append(state['w_values'][i] + (self.actor_lr * state['prediction_errors'][i]))
        new_w_values = [w_val/np.sum(np.abs(new_w_values)) for w_val in new_w_values]

        self.w_values[state['state_id']] = pd.concat([self.w_values[state['state_id']],
                                                    pd.DataFrame([new_w_values], 
                                                                columns=self.w_values[state['state_id']].columns)], 
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
        self.update_q_values(state)
        self.update_prediction_errors(state)

        if self.__class__.__name__ == 'Relative':
            self.update_context_values(state)
            self.update_context_prediction_errors(state)

        if self.__class__.__name__ == 'ActorCritic':
            self.update_w_values(state)

    #Plotting functions
    def plot_model(self):
    
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
    def run_trial(self, state, phase = 'learning'):
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
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    """

    def __init__(self, actor_lr, critic_lr, temperature):
        super().__init__()

        #Set parameters
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.parameters = {'actor_lr': self.actor_lr, 
                           'critic_lr': self.critic_lr, 
                           'temperature': self.temperature}
        
    #RL functions
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward

        return state
    
    def compute_prediction_error(rewards, state):
        state['prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        return state

    def select_action(self, state):

        transformed_w_values = np.exp(np.divide(state['w_values'], self.temperature))
        probability_w_values = (transformed_w_values/np.sum(transformed_w_values)).cumsum()
        state['action'] = np.where(probability_w_values >= rnd.random())[0][0]
        if 'correct_action' in state.keys():
            state['accuracy'] = int(state['action'] == state['correct_action'])

        return state

    #Run trial functions
    def run_trial(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_reward(state)
            state = self.get_q_value(state)
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
    def run_trial(self, state, phase = 'learning'):
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

