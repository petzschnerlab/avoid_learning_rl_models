import random as rnd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import scipy

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

    def get_c_value(self, state):
        state['c_values'] = list(self.c_values[state['state_id']].iloc[-1])
        return state
    
    def get_m_value(self, state):
        state['m_values'] = list(self.m_values[state['state_id']].iloc[-1])
        return state

    def get_final_q_values(self, state):
        state['q_values'] = [float(self.final_q_values[stim]) for stim in state['stim_id']]
        return state
    
    def get_final_m_values(self, state):
        state['m_values'] = [float(self.final_m_values[stim]) for stim in state['stim_id']]
        return state

    def get_context_value(self, state):
        state['context_value'] = list(self.context_values[state['state_id']].iloc[-1])
        return state
    
    def get_decayed_q_values(self, state):
        q_final = [float(self.final_q_values[stim]) for stim in state['stim_id']]
        q_initial = [float(self.initial_q_values[stim]) for stim in state['stim_id']]
        q_final_decayed = [q*(1-self.decay_factor) for q in q_final]
        q_initial_decayed = [q*(self.decay_factor) for q in q_initial]
        state['q_values'] = [q_final_decayed[i] + q_initial_decayed[i] for i in range(len(q_final))]
        return state
    
    def get_final_w_values(self, state):
        state['w_values'] = [float(self.final_w_values[stim]) for stim in state['stim_id']]
        return state
    
    def get_decayed_w_values(self, state):
        w_final = [float(self.final_w_values[stim]) for stim in state['stim_id']]
        w_initial = [float(self.initial_w_values[stim]) for stim in state['stim_id']]
        w_final_decayed = [w*(1-self.decay_factor) for w in w_final]
        w_initial_decayed = [w*(self.decay_factor) for w in w_initial]
        state['w_values'] = [w_final_decayed[i] + w_initial_decayed[i] for i in range(len(w_final))]
        return state

    def get_h_values(self, state):
        state['h_values'] = [(state['w_values'][i] * (1-self.mixing_factor)) + (state['q_values'][i] * self.mixing_factor) for i in range(len(state['w_values']))]
        return state

    #Update functions
    def update_prediction_errors(self, state):

        if 'Hybrid' in self.__class__.__name__:
            self.q_prediction_errors[state['state_id']] = pd.concat([self.q_prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['q_prediction_errors']], 
                                                                            columns=self.q_prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)
            self.v_prediction_errors[state['state_id']] = pd.concat([self.v_prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['v_prediction_errors']], 
                                                                            columns=self.v_prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)
        elif 'QRelative' == self.__class__.__name__:
            self.q_prediction_errors[state['state_id']] = pd.concat([self.q_prediction_errors[state['state_id']],
                                                                pd.DataFrame([state['q_prediction_errors']],
                                                                                columns=self.q_prediction_errors[state['state_id']].columns)],
                                                                ignore_index=True)
            self.c_prediction_errors[state['state_id']] = pd.concat([self.c_prediction_errors[state['state_id']],
                                                                pd.DataFrame([state['c_prediction_errors']],
                                                                                columns=self.c_prediction_errors[state['state_id']].columns)],
                                                                ignore_index=True)
        else:
            self.prediction_errors[state['state_id']] = pd.concat([self.prediction_errors[state['state_id']], 
                                                               pd.DataFrame([state['prediction_errors']], 
                                                                            columns=self.prediction_errors[state['state_id']].columns)], 
                                                               ignore_index=True)

    def update_q_values(self, state):

        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
        prediction_errors = state['q_prediction_errors'] if 'Hybrid' in self.__class__.__name__ or 'QRelative' == self.__class__.__name__ else state['prediction_errors']
            
        new_q_values = []
        for i in range(len(state['rewards'])):
            new_q_values.append(state['q_values'][i] + (learning_rates[i] * prediction_errors[i]))

        self.q_values[state['state_id']] = pd.concat([self.q_values[state['state_id']],
                                                      pd.DataFrame([new_q_values], 
                                                                   columns=self.q_values[state['state_id']].columns)], 
                                                    ignore_index=True)
        
    def update_w_values(self, state):

        learning_rates = [self.factual_actor_lr, self.counterfactual_actor_lr] if state['action'] == 0 else [self.counterfactual_actor_lr, self.factual_actor_lr]
        prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
            
        new_w_values = []
        for i in range(len(state['rewards'])):
            new_w_values.append(state['w_values'][i] + (learning_rates[i] * prediction_errors[i]))
        #check whether the new w values are nan
        try:
            new_w_values = [w_val/np.sum(np.abs(new_w_values)) for w_val in new_w_values]
        except:
            print('debug')

        self.w_values[state['state_id']] = pd.concat([self.w_values[state['state_id']],
                                                    pd.DataFrame([new_w_values], 
                                                                columns=self.w_values[state['state_id']].columns)], 
                                                    ignore_index=True)
    
    def update_v_values(self, state):

        prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']

        new_v_values = state['v_values'][0] + (self.critic_lr * prediction_errors[state['action']]) #TODO: Should we use the averaged prediction error or the selected action's prediction error?

        self.v_values[state['state_id']] = pd.concat([self.v_values[state['state_id']],
                                                      pd.DataFrame([new_v_values], 
                                                                   columns=self.v_values[state['state_id']].columns)], 
                                                    ignore_index=True)
    
    def update_h_values(self, state):

        self.h_values[state['state_id']] = pd.concat([self.h_values[state['state_id']],
                                                      pd.DataFrame([state['h_values']], 
                                                                   columns=self.h_values[state['state_id']].columns)], 
                                                    ignore_index=True)
    def update_c_values(self, state):

        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
            
        new_c_values = []
        for i in range(len(state['rewards'])):
            new_c_values.append(state['c_values'][i] + (learning_rates[i] * state['c_prediction_errors'][i]))

        self.c_values[state['state_id']] = pd.concat([self.c_values[state['state_id']],
                                                      pd.DataFrame([new_c_values], 
                                                                   columns=self.c_values[state['state_id']].columns)], 
                                                    ignore_index=True)
                                                
    def update_m_values(self, state):

        m_values = [(state['q_values'][i] * (1-self.mixing_factor)) + (state['c_values'][i] * self.mixing_factor) for i in range(len(state['q_values']))]
        
        self.m_values[state['state_id']] = pd.concat([self.m_values[state['state_id']], 
                                                                      pd.DataFrame([m_values], 
                                                                                   columns=self.m_values[state['state_id']].columns)], 
                                                                      ignore_index=True)

    def update_context_values(self, state):

        new_context_value = state['context_value'] + (self.contextual_lr * state['context_prediction_errors'])

        self.context_values[state['state_id']] = pd.concat([self.context_values[state['state_id']], 
                                                            pd.DataFrame([new_context_value], 
                                                                         columns=self.context_values[state['state_id']].columns)], 
                                                            ignore_index=True)
    
    def update_context_prediction_errors(self, state):
        self.context_prediction_errors[state['state_id']] = pd.concat([self.context_prediction_errors[state['state_id']], 
                                                                      pd.DataFrame([state['context_prediction_errors']], 
                                                                                   columns=self.context_prediction_errors[state['state_id']].columns)], 
                                                                      ignore_index=True)

    def reward_valence(self, reward):
        for ri, r in enumerate(reward):
            if r > 0:
                reward[ri] = 1-self.valence_factor
            elif r < 0:
                reward[ri] = -self.valence_factor
            else:
                reward[ri] = 0
        return reward
    
    def update_model(self, state):
        
        self.update_prediction_errors(state)

        if self.__class__.__name__ != 'ActorCritic':
            self.update_q_values(state)

        if self.__class__.__name__ == 'Relative':
            self.update_context_values(state)
            self.update_context_prediction_errors(state)

        if self.__class__.__name__ == 'QRelative':
            self.update_context_values(state)
            self.update_context_prediction_errors(state)
            self.update_c_values(state)
            self.update_m_values(state)
            
        if self.__class__.__name__ == 'ActorCritic' or 'Hybrid' in self.__class__.__name__:
            self.update_w_values(state)
            self.update_v_values(state)
        
        if 'Hybrid' in self.__class__.__name__:
            self.update_h_values(state)

    def fit_log_likelihood(self, values):

        '''
        Action selection function for the fitting procedure

        parameters
        ----------
        values: list[float]
            List of Q-values
        temperature: float
            Temperature parameter for softmax action selection
        '''
        
        transformed_values = np.divide(values, self.temperature)
        return transformed_values - scipy.special.logsumexp(transformed_values)
    
    def fit(self, data, bounds):

        '''
        Fit the model to the data
        '''

        #Fit the model
        free_params = [self.parameters[key] for key in self.parameters]
        fit_results = scipy.optimize.minimize(self.fit_func, 
                                              free_params, 
                                              args=data,
                                              bounds=bounds, 
                                              method='L-BFGS-B')
        
        #Unpack the fitted params
        fitted_params = {}
        for fi, key in enumerate(self.parameters):
            fitted_params[key] = fit_results.x[fi]

        return fit_results, fitted_params

    #Plotting functions
    def plot_model(self):
    
        fig, ax = plt.subplots(4, 4, figsize=(20,5))

        #Plot q-values
        if 'Hybrid' in self.__class__.__name__:
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
        prediction_errors = self.prediction_errors if 'Hybrid' in self.__class__.__name__ else self.q_prediction_errors
        for i, key in enumerate(prediction_errors.keys()):
            for ci, col in enumerate(prediction_errors[key].columns):
                rolling_prediction_errors = prediction_errors[key][col].reset_index(drop=True).rolling(window=2).mean()
                ax[1,i].plot(rolling_prediction_errors, label=['Stim 1', 'Stim 2'][ci], color=['C0', 'C1'][ci])
                if'Hybrid' in self.__class__.__name__:
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

        #Plot choice rate
        x_tick_labels = ['75R', '25R', '25P', '75P', 'N']
        ax[3,0].bar(self.choice_rate.keys(), self.choice_rate.values())
        ax[3,0].set_ylabel('Choice Rate (%)')
        ax[3,0].set_xlabel('Stimulus')
        ax[3,0].set_xticks(range(len(self.choice_rate.keys())))
        ax[3,0].set_xticklabels(x_tick_labels)

        plt.show()
    
    def get_choice_rates(self):
        return self.choice_rate


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

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        
        return state

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Unpack free parameters
        self.factual_lr, self.counterfactual_lr, self.temperature = x
        states, actions, rewards = args

        #Initialize values
        logp_actions = np.zeros((len(actions),1))

        for trial, (state_id, action, reward) in enumerate(zip(states, actions, rewards)):

            #Populate state
            state = {'rewards': reward, 
                     'action': action, 
                     'state_id': state_id}
            
            #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower 
            state = self.fit_forward(state)

            #Compute and store the log probability of the observed action
            logp_actions[trial] = self.fit_log_likelihood(state['q_values'])[action]

        #Return the negative log likelihood of all observed actions
        return -np.sum(logp_actions[1:])

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

    def __init__(self, factual_actor_lr, counterfactual_actor_lr, critic_lr, temperature, valence_factor):
        super().__init__()

        #Set parameters
        self.factual_actor_lr = factual_actor_lr
        self.counterfactual_actor_lr = counterfactual_actor_lr
        self.critic_lr = critic_lr
        self.temperature = temperature
        self.valence_factor = valence_factor
        self.parameters = {'factual_actor_lr': self.factual_actor_lr, 
                           'counterfactual_actor_lr': self.counterfactual_actor_lr, 
                           'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'valence_factor': self.valence_factor}
        
    #RL functions
    def get_reward(self, state):
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
        state['prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]
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

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_w_values(state)
            state = self.select_action
        
        return state

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Unpack free parameters
        self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.valence_factor = x    
        states, actions, rewards = args

        #Initialize values
        logp_actions = np.zeros((len(actions),1))

        for trial, (state_id, action, reward) in enumerate(zip(states, actions, rewards)):
            
            #Populate state
            state = {'rewards': self.reward_valence(reward),
                     'action': action,
                     'state_id': state_id}
            
            #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower 
            state = self.fit_forward(state)

            #Compute and store the log probability of the observed action
            logp_actions[trial] = self.fit_log_likelihood(state['w_values'])[action]
            
        #Return the negative log likelihood of all observed actions
        return -np.sum(logp_actions[1:])

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
                
    #RL functions
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward
        state['context_reward'] = np.average(state['rewards'])

        return state
    
    def compute_prediction_error(rewards, state):
        state['prediction_errors'] = [state['rewards'][i] - state['context_value'][0] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['context_prediction_errors'] = state['context_reward'] - state['context_value']
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

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_context_value(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.select_action(state)
        
        return state

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Unpack free parameters
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature = x
        states, actions, rewards = args

        #Initialize values
        logp_actions = np.zeros((len(actions),1))

        for trial, (state_id, action, reward) in enumerate(zip(states, actions, rewards)):
            
            #Populate state
            state = {'rewards': reward,
                     'context_reward': np.average(reward),
                     'action': action,
                     'state_id': state_id}

            #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower
            state = self.fit_forward(state)

            #Compute and store the log probability of the observed action
            logp_actions[trial] = self.fit_log_likelihood(state['q_values'])[action]

        #Return the negative log likelihood of all observed actions
        return -np.sum(logp_actions[1:])
    
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
                 critic_lr, temperature, mixing_factor, valence_factor):
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
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                            'factual_actor_lr': self.factual_actor_lr,
                            'counterfactual_actor_lr': self.counterfactual_actor_lr,
                            'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor,
                           'valence_factor': self.valence_factor}

    #RL functions    
    def get_reward(self, state):
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
        state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state):

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
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.get_h_values(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_w_values(state)
            state = self.select_action(state)

        return state

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.valence_factor = x
        states, actions, rewards = args

        #Initialize values
        logp_actions = np.zeros((len(actions),1))

        for trial, (state_id, action, reward) in enumerate(zip(states, actions, rewards)):
            
            #Populate state
            state = {'rewards': self.reward_valence(reward),
                     'action': action,
                     'state_id': state_id}

            #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower
            state = self.fit_forward(state)

            #Compute and store the log probability of the observed action
            logp_actions[trial] = self.fit_log_likelihood(state['h_values'])[action]

        #Return the negative log likelihood of all observed actions
        return -np.sum(logp_actions[1:])

class Hybrid2(RLToolbox):

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
                 critic_lr, temperature, mixing_factor, valence_factor, noise_factor, decay_factor):
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
        self.noise_factor = noise_factor
        self.decay_factor = decay_factor
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                            'factual_actor_lr': self.factual_actor_lr,
                            'counterfactual_actor_lr': self.counterfactual_actor_lr,
                            'critic_lr': self.critic_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor,
                           'valence_factor': self.valence_factor,
                           'noise_factor': self.noise_factor,
                           'decay_factor': self.decay_factor}

    #RL functions    
    def get_reward(self, state):
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
        state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]

        return state

    def select_action(self, state):

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
            state = self.get_decayed_q_values(state)
            state = self.get_decayed_w_values(state)
            state = self.get_h_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_v_value(state)
            state = self.get_w_value(state)
            state = self.get_h_values(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_decayed_q_values(state)
            state = self.get_decayed_w_values(state)
            state = self.get_h_values(state)
            state = self.select_action(state)

        return state

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.valence_factor, self.noise_factor, self.decay_factor = x
        states, actions, rewards = args

        #Initialize values
        logp_actions = np.zeros((len(actions),1))

        for trial, (state_id, action, reward) in enumerate(zip(states, actions, rewards)):

            #Populate state
            state = {'rewards': self.reward_valence(reward),
                     'action': action,
                     'state_id': state_id}

            #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower
            state = self.fit_forward(state)
            
            #Compute and store the log probability of the observed action
            logp_actions[trial] = self.fit_log_likelihood(state['h_values'])[action]

        #Return the negative log likelihood of all observed actions
        return -np.sum(logp_actions[1:])
    
class QRelative(RLToolbox):

    """
    Reinforcement Learning Model: Q-Relative
    This is a custom combination of Q-Learning and Relative (Palminteri et al., 2015)

    Parameters:
    ------------
    factual_lr: float
        Learning rate for factual Q-value update
    counterfactual_lr: float
        Learning rate for counterfactual Q-value update
    temperature: float
        Temperature parameter for softmax action selection
    mixing_factor: float
        Mixing factor for the Q-values
    """

    def __init__(self, factual_lr, counterfactual_lr, contextual_lr, temperature, mixing_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.contextual_lr =  contextual_lr
        self.temperature = temperature  
        self.mixing_factor = mixing_factor
        self.parameters = {'factual_lr': self.factual_lr, 
                           'counterfactual_lr': self.counterfactual_lr, 
                           'contextual_lr': self.contextual_lr,
                           'temperature': self.temperature,
                           'mixing_factor': self.mixing_factor}
    
    #RL functions    
    def get_reward(self, state):
        random_numbers = [rnd.random() for i in range(len(state['stim_id']))]
        reward = [int(random_numbers[i] < state['probabilities'][i]) for i in range(len(state['stim_id']))]
        reward = [reward[i] * state['feedback'] for i in range(len(state['stim_id']))]

        state['rewards'] = reward
        state['context_reward'] = np.average(state['rewards'])

        return state
    
    def compute_prediction_error(rewards, state):
        state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['c_prediction_errors'] = [state['rewards'][i] - state['context_value'][0] - state['c_values'][i] for i in range(len(state['rewards']))]
        state['context_prediction_errors'] = state['context_reward'] - state['context_value']
        state['prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        return state

    def select_action(self, state):
        transformed_q_values = np.exp(np.divide(state['m_values'], self.temperature))
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
            state = self.get_c_value(state)
            state = self.get_m_value(state)
            state = self.get_context_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_m_values(state)
            state = self.select_action(state)
        self.update_task_data(state, phase=phase)

    def fit_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_c_value(state)
            state = self.get_m_value(state)
            state = self.get_context_value(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_m_values(state)
            state = self.select_action(state)

        return state

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Unpack free parameters
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature, self.mixing_factor = x
        states, actions, rewards = args

        #Initialize values
        logp_actions = np.zeros((len(actions),1))

        for trial, (state_id, action, reward) in enumerate(zip(states, actions, rewards)):

            #Populate state
            state = {'rewards': reward,
                     'context_reward': np.average(reward),
                     'action': action,
                     'state_id': state_id}

            #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower
            state = self.fit_forward(state)
            
            #Compute and store the log probability of the observed action
            logp_actions[trial] = self.fit_log_likelihood(state['m_values'])[action]

        #Return the negative log likelihood of all observed actions
        return -np.sum(logp_actions[1:])
    
def get_model(model, fit_data=None):

    if model == 'QLearning':
        factual_lr = 0.1 if fit_data is None else fit_data['factual_lr']
        counterfactual_lr = 0.5 if fit_data is None else fit_data['counterfactual_lr']
        temperature = 0.1 if fit_data is None else fit_data['temperature']

        model = QLearning(factual_lr=factual_lr, 
                        counterfactual_lr=counterfactual_lr, 
                        temperature=temperature)

        model.bounds = [(0, .99), (0, .99), (0.01, 10)]

    elif model == 'ActorCritic':
        factual_actor_lr = 0.1 if fit_data is None else fit_data['factual_actor_lr']
        counterfactual_actor_lr = 0.5 if fit_data is None else fit_data['counterfactual_actor_lr']
        critic_lr = 0.1 if fit_data is None else fit_data['critic_lr']
        temperature = 0.1 if fit_data is None else fit_data['temperature']
        valence_factor = 0.5 if fit_data is None else fit_data['valence_factor']
        
        model = ActorCritic(factual_actor_lr=factual_actor_lr,
                            counterfactual_actor_lr=counterfactual_actor_lr,
                            critic_lr=critic_lr,
                            temperature=temperature,
                            valence_factor=valence_factor)
        
        model.bounds = [(0, .99), (0, .99), (0,.99), (0.01, 10), (-1, 1)]

    elif model == 'Relative':
        factual_lr = 0.1 if fit_data is None else fit_data['factual_lr']
        counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr']
        contextual_lr = 0.1 if fit_data is None else fit_data['contextual_lr']
        temperature = 0.1 if fit_data is None else fit_data['temperature']

        model = Relative(factual_lr=factual_lr,
                        counterfactual_lr=counterfactual_lr,
                        contextual_lr=contextual_lr,
                        temperature=temperature)
        
        model.bounds = [(0, .99), (0, .99), (0,.99), (0.01, 10)]

    elif model == 'Hybrid2012':
        factual_lr = 0.1 if fit_data is None else fit_data['factual_lr']
        counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr']
        factual_actor_lr = 0.1 if fit_data is None else fit_data['factual_actor_lr']
        counterfactual_actor_lr = 0.05 if fit_data is None else fit_data['counterfactual_actor_lr']
        critic_lr = 0.1 if fit_data is None else fit_data['critic_lr']
        temperature = 0.1 if fit_data is None else fit_data['temperature']
        mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor']
        valence_factor = 0.5 if fit_data is None else fit_data['valence_factor']

        model = Hybrid(factual_lr=factual_lr,
                    counterfactual_lr=counterfactual_lr,
                    factual_actor_lr=factual_actor_lr,
                    counterfactual_actor_lr=counterfactual_actor_lr,
                    critic_lr=critic_lr,
                    temperature=temperature,
                    mixing_factor=mixing_factor,
                    valence_factor=valence_factor)
    
        model.bounds = [(0, .99), (0, .99), (0, .99), (0, .99), (0, .99), (0.01, 10), (0, 1), (-1, 1)]

    elif model == 'Hybrid2021':
        factual_lr = 0.1 if fit_data is None else fit_data['factual_lr']
        counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr']
        factual_actor_lr = 0.1 if fit_data is None else fit_data['factual_actor_lr']
        counterfactual_actor_lr = 0.05 if fit_data is None else fit_data['counterfactual_actor_lr']
        critic_lr = 0.1 if fit_data is None else fit_data['critic_lr']
        temperature = 0.1 if fit_data is None else fit_data['temperature']
        mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor']
        valence_factor = 0.5 if fit_data is None else fit_data['valence_factor']
        noise_factor = 0.1 if fit_data is None else fit_data['noise_factor']
        decay_factor = 0.1 if fit_data is None else fit_data['decay_factor']

        model = Hybrid2(factual_lr=factual_lr,
                    counterfactual_lr=counterfactual_lr,
                    factual_actor_lr=factual_actor_lr,
                    counterfactual_actor_lr=counterfactual_actor_lr,
                    critic_lr=critic_lr,
                    temperature=temperature,
                    mixing_factor=mixing_factor,
                    valence_factor=valence_factor,
                    noise_factor=noise_factor,
                    decay_factor=decay_factor)
        
        model.bounds = [(0, .99), (0, .99), (0, .99), (0, .99), (0, .99), (0.01, 10), (0, 1), (-1, 1), (0, 1), (0, 1)]
        
    elif model == 'QRelative':
        factual_lr = 0.1 if fit_data is None else fit_data['factual_lr']
        counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr']
        contextual_lr = 0.1 if fit_data is None else fit_data['contextual_lr']
        temperature = 0.1 if fit_data is None else fit_data['temperature']
        mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor']

        model = QRelative(factual_lr=factual_lr,
                        counterfactual_lr=counterfactual_lr,
                        contextual_lr=contextual_lr,
                        temperature=temperature,
                        mixing_factor=mixing_factor)
        
        model.bounds = [(0, .99), (0, .99), (0, .99), (0.01, 10), (0, 1)]
        
    else:
        raise ValueError('Model not recognized.')
    
    return model