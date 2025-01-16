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
        state['q_values'] = self.q_values[state['state_id']]
        return state
    
    def get_v_value(self, state):
        state['v_values'] = self.v_values[state['state_id']]
        return state
    
    def get_w_value(self, state):
        state['w_values'] = self.w_values[state['state_id']]
        return state

    def get_c_value(self, state):
        state['c_values'] = self.c_values[state['state_id']]
        return state
    
    def get_m_value(self, state):
        state['m_values'] = [(state['q_values'][i] * (1-self.mixing_factor)) + (state['c_values'][i] * self.mixing_factor) for i in range(len(state['q_values']))]
        return state
    
    def get_h_values(self, state):
        state['h_values'] = [(state['w_values'][i] * (1-self.mixing_factor)) + (state['q_values'][i] * self.mixing_factor) for i in range(len(state['w_values']))]
        return state

    def get_final_q_values(self, state):
        state['q_values'] = [self.final_q_values[stim].values for stim in state['stim_id']]
        return state

    def get_final_c_values(self, state):
        state['c_values'] = [self.final_c_values[stim].values for stim in state['stim_id']]
        return state

    def get_context_value(self, state):
        state['context_value'] = self.context_values[state['state_id']]
        return state
    
    def get_decayed_q_values(self, state):
        q_final = [self.final_q_values[stim] for stim in state['stim_id']]
        q_initial = [self.initial_q_values[stim] for stim in state['stim_id']]
        q_final_decayed = [q*(1-self.decay_factor) for q in q_final]
        q_initial_decayed = [q*(self.decay_factor) for q in q_initial]
        state['q_values'] = [q_final_decayed[i].values[0] + q_initial_decayed[i].values[0] for i in range(len(q_final))]
        return state
    
    def get_final_w_values(self, state):
        state['w_values'] = [self.final_w_values[stim] for stim in state['stim_id']]
        return state
    
    def get_decayed_w_values(self, state):
        w_final = [self.final_w_values[stim] for stim in state['stim_id']]
        w_initial = [self.initial_w_values[stim] for stim in state['stim_id']]
        w_final_decayed = [w*(1-self.decay_factor) for w in w_final]
        w_initial_decayed = [w*(self.decay_factor) for w in w_initial]
        state['w_values'] = [w_final_decayed[i].values[0] + w_initial_decayed[i].values[0] for i in range(len(w_final))]
        return state

    #Update functions
    def update_prediction_errors(self, state):

        if 'Hybrid' in self.__class__.__name__:
            self.q_prediction_errors[state['state_id']] = state['q_prediction_errors']
            self.v_prediction_errors[state['state_id']] = state['v_prediction_errors']

        elif 'QRelative' == self.__class__.__name__:
            self.q_prediction_errors[state['state_id']] = state['q_prediction_errors']
            self.c_prediction_errors[state['state_id']] = state['c_prediction_errors']
        else:
            self.prediction_errors[state['state_id']] = state['prediction_errors']

    def update_q_values(self, state):
        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
        prediction_errors = state['q_prediction_errors'] if 'Hybrid' in self.__class__.__name__ or 'QRelative' == self.__class__.__name__ else state['prediction_errors']
        self.q_values[state['state_id']] = [state['q_values'][i] + (learning_rates[i] * prediction_errors[i]) for i in range(len(state['q_values']))]
        
    def update_w_values(self, state):
        learning_rates = [self.factual_actor_lr, self.counterfactual_actor_lr] if state['action'] == 0 else [self.counterfactual_actor_lr, self.factual_actor_lr]
        prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
        new_w_values = [state['w_values'][i] + (learning_rates[i] * prediction_errors[i]) for i in range(len(state['w_values']))]

        #Check if the new w values are all zeros, and adjust them to initial values if so
        if np.sum([np.abs(new_w_values[i]) for i in range(len(new_w_values))]) == 0:
            new_w_values = [0.01]*len(new_w_values)

        self.w_values[state['state_id']] = new_w_values/np.sum(np.abs(new_w_values))
    
    def update_v_values(self, state):
        prediction_errors = state['v_prediction_errors'] if 'Hybrid' in self.__class__.__name__ else state['prediction_errors']
        self.v_values[state['state_id']] = [state['v_values'][0] + (self.critic_lr * prediction_errors[state['action']])] #TODO: Should we use the averaged prediction error or the selected action's prediction error?
    
    def update_h_values(self, state):
        self.h_values[state['state_id']] = state['h_values']

    def update_c_values(self, state):
        learning_rates = [self.factual_lr, self.counterfactual_lr] if state['action'] == 0 else [self.counterfactual_lr, self.factual_lr]
        self.c_values[state['state_id']] = [state['c_values'][i] + (learning_rates[i] * state['c_prediction_errors'][i]) for i in range(len(state['c_values']))]

    def update_context_values(self, state):
        self.context_values[state['state_id']] = state['context_value'] + (self.contextual_lr * state['context_prediction_errors'])
    
    def update_context_prediction_errors(self, state):
        self.context_prediction_errors[state['state_id']] = state['context_prediction_errors']

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

        if 'Relative' in self.__class__.__name__:
            self.update_context_values(state)
            self.update_context_prediction_errors(state)

        if self.__class__.__name__ == 'QRelative':
            self.update_context_values(state)
            self.update_context_prediction_errors(state)
            self.update_c_values(state)
            
        if self.__class__.__name__ == 'ActorCritic' or 'Hybrid' in self.__class__.__name__:
            self.update_w_values(state)
            self.update_v_values(state)
        
        if 'Hybrid' in self.__class__.__name__:
            self.update_h_values(state)
    
    def reset_datalists(self):

        for s in self.states:

            if 'Hybrid' in self.__class__.__name__:
                self.q_prediction_errors[s] = [0]*len(self.q_prediction_errors[s])
                self.v_prediction_errors[s] = [0]*len(self.v_prediction_errors[s])
            elif 'QRelative' == self.__class__.__name__:
                self.q_prediction_errors[s] = [0]*len(self.q_prediction_errors[s])
                self.c_prediction_errors[s] = [0]*len(self.c_prediction_errors[s])
            else:
                self.prediction_errors[s] = [0]*len(self.prediction_errors[s])

            if self.__class__.__name__ != 'ActorCritic':
                self.q_values[s] = [0]*len(self.q_values[s])

            if 'Relative' in self.__class__.__name__:
                self.context_values[s] = [0]*len(self.context_values[s])
                self.context_prediction_errors[s] = [0]*len(self.context_prediction_errors[s])

            if self.__class__.__name__ == 'QRelative':
                self.context_values[s] = [0]*len(self.context_values[s])
                self.context_prediction_errors[s] = [0]*len(self.context_prediction_errors[s])
                self.c_values[s] = [0]*len(self.c_values[s])

            if self.__class__.__name__ == 'ActorCritic' or 'Hybrid' in self.__class__.__name__: 
                self.w_values[s] = [0.01]*len(self.w_values[s])
                self.v_values[s] = [0]*len(self.v_values[s])
                
            if 'Hybrid' in self.__class__.__name__:
                self.h_values[s] = [0]*len(self.h_values[s])

    def combine_values(self):
        #Inter-phase processing
        if self.__class__.__name__ == 'ActorCritic':
            self.combine_v_values()
            self.combine_w_values()
        elif 'Hybrid' in self.__class__.__name__:
            self.combine_q_values()
            self.combine_v_values()
            self.combine_w_values()
        elif 'QRelative' == self.__class__.__name__:
            self.combine_q_values()
            self.combine_c_values()
        else:
            self.combine_q_values()

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

        transformed_values = np.exp(np.divide(values, self.temperature))
        probability_values = (transformed_values/np.sum(transformed_values))

        if self.__class__.__name__ == 'Hybrid2021':
            uniform_dist = np.ones(len(probability_values))/len(probability_values)
            probability_values = (((1-self.noise_factor)*probability_values) + (self.noise_factor*uniform_dist))

        return -np.log(probability_values)

    def fit_transfer_forward(self, transfer_data, value_name):
        #Although more succint and verbose, takes much longer to run
        transfer_data = pd.DataFrame({'state_id': transfer_data[0], 'action': transfer_data[1]})

        transfer_data['stim_id'] = transfer_data['state_id'].apply(lambda x: x.split(' ')[-1])
        transfer_data['values'] = transfer_data['stim_id'].apply(lambda x: self.fit_forward({'stim_id': x}, phase='transfer')[value_name])
        transfer_data['NLL'] = transfer_data.apply(lambda x: self.fit_log_likelihood(x['values'])[x['action']][0], axis=1)

        #transfer_data['NLL'] = transfer_data.apply(lambda x: self.fit_log_likelihood(self.fit_forward({'stim_id': x['state_id'].split(' ')[-1]}, 
        #                                                                                              phase='transfer')[value_name])[x['action']], 
        #                                                                                              axis=1)
        
        return transfer_data['NLL'].sum()
    
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
    
    def fit_task(self, args, value_type, transform_reward=False, context_reward=False):
        
        #Unpack data
        learning_data, transfer_data = args[0]
        learning_states, learning_actions, learning_rewards = learning_data

        if self.transfer_trials > 0:
            #Reduce the transfer data to the first three presentations of each state_id
            transfer_data = pd.DataFrame({'state_id': transfer_data[0], 'action': transfer_data[1]})
            transfer_data = transfer_data.groupby('state_id').head(self.transfer_trials)
            transfer_data = transfer_data.values.T.tolist()
        transfer_states, transfer_actions = transfer_data

        #Initialize values
        log_likelihood = 0

        #Learning phase
        for fit_number in range(self.number_of_fits):
            for trial, (state_id, action, reward) in enumerate(zip(learning_states, learning_actions, learning_rewards)):

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

            if self.fit_transfer_phase:
                
                #Inter-phase processing
                self.combine_values()

                #Transfer phase
                if False: #Toggle to switch between methods for testing. Function method is slower than loop, so it's avoided
                    log_likelihood -= self.fit_transfer_forward(transfer_data, value_type, reduced = True)
                else:
                    for trial, (state_id, action) in enumerate(zip(transfer_states, transfer_actions)):
                        
                        #Populate state
                        state = {'action': action, 
                                'stim_id': [stim for stim in state_id.split(' ')[-1]]}
                        
                        #Forward: TODO: When using functions (e.g., self.compute_PE -> self.update model -> self.get_q_value), it's much slower 
                        state = self.fit_forward(state, phase='transfer')

                        #Compute and store the log probability of the observed action
                        log_likelihood -= self.fit_log_likelihood(state[value_type])[action]

        return log_likelihood
    
    def simulate(self, data):

        '''
        Simulate the model
        '''
        self.sim_func(data)

    def sim_task(self, args, transform_reward=False, context_reward=False):

        #Unpack data
        learning_data, transfer_data = args[0]['learning'], args[0]['transfer']

        #Learning phase
        for trial, trial_data in learning_data.iterrows():

            #Populate rewards
            rewards = [int(trial_data['reward_L']), int(trial_data['reward_R'])]
            if transform_reward:
                rewards = self.reward_valence(rewards)

            #Populate state
            state = {'block_number': 0, 
                     'trial_number': trial_data['trial_number'], 
                     'state_index': 0, 
                     'state_id': trial_data['state'], 
                     'stim_id': [trial_data['state'].split('State ')[-1][0],trial_data['state'].split('State ')[-1][1]], 
                     'context': trial_data['context_val_name'], 
                     'feedback': [trial_data['feedback_L'], trial_data['feedback_R']], 
                     'probabilities': [trial_data['symbol_L_name'][:2], trial_data['symbol_R_name'][:2]], 
                     'rewards': rewards,
                     'correct_action': 0}
            
            if context_reward:
                state['context_reward'] = np.average(state['rewards'])
            
            #Forward
            self.sim_forward(state)

        #Inter-phase processing
        self.combine_values()

        #Transfer phase
        for trial, trial_data in transfer_data.iterrows():
            
            #Populate state
            state = {'block_number': 0,
                     'trial_number': trial,
                     'state_id': trial_data['state'],
                     'stim_id': trial_data['state'].split('State ')[-1]}
            
            #Forward
            self.sim_forward(state, phase='transfer')
        
        return self.task_learning_data, self.task_transfer_data
    
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
    
    def compute_prediction_error(self, state):
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
        self.factual_lr, self.counterfactual_lr, self.temperature = x

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'q_values')

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args)

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
    
    def compute_prediction_error(self, state):
        state['prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses rewards independently
        #state['prediction_errors'] = [np.mean(state['rewards']) - state['v_values'][0] for i in range(len(state['rewards']))] #Uses averaged reward, do not use - values of actions within same state are equal
        #state['prediction_errors'] = [state['rewards'][state['action']] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses selected reward
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
        
        return state
    
    def sim_forward(self, state, phase = 'learning'):
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

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Reset indices on succeeding fits
        self.reset_datalists()
            
        #Unpack free parameters
        self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.valence_factor = x 

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'w_values', transform_reward=False)

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, transform_reward=False)
    
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
    
    def compute_prediction_error(self, state):
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
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature = x

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'q_values', context_reward=True)

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, context_reward=True)
    
class Hybrid2012(RLToolbox):

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
    
    def compute_prediction_error(self, state):
        state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses rewards independently
        #state['v_prediction_errors'] = [state['rewards'][state['action']] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses selected reward

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
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.valence_factor = x

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'h_values', transform_reward=False)


    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, transform_reward=False)

class Hybrid2021(RLToolbox):

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
    
    def compute_prediction_error(self, state):
        state['q_prediction_errors'] = [state['rewards'][i] - state['q_values'][i] for i in range(len(state['rewards']))]
        state['v_prediction_errors'] = [state['rewards'][i] - state['v_values'][0] for i in range(len(state['rewards']))]
        #state['v_prediction_errors'] = [state['rewards'][state['action']] - state['v_values'][0] for i in range(len(state['rewards']))] #Uses selected reward

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
    
    def sim_forward(self, state, phase = 'learning'):
        if phase == 'learning':
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

    def fit_func(self, x, *args):

        '''
        Fit the model to the data

        data: tuple
            Tuple of data to be fitted: actions, rewards
        '''

        #Reset indices on succeeding fits
        self.reset_datalists()

        #Unpack parameters
        self.factual_lr, self.counterfactual_lr, self.factual_actor_lr, self.counterfactual_actor_lr, self.critic_lr, self.temperature, self.mixing_factor, self.valence_factor, self.noise_factor, self.decay_factor = x
        
        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'h_values', transform_reward=False)
    

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, transform_reward=False)

class wRelative(RLToolbox):

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

    def __init__(self, factual_lr, counterfactual_lr, contextual_lr, temperature, mixing_factor):
        super().__init__()

        #Set parameters
        self.factual_lr = factual_lr
        self.counterfactual_lr = counterfactual_lr
        self.contextual_lr = contextual_lr
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
    
    def compute_prediction_error(self, state):
        state['prediction_errors'] = [state['rewards'][i] - (state['context_value'][0] * self.mixing_factor) - state['q_values'][i] for i in range(len(state['rewards']))]
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
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature, self.mixing_factor = x

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'q_values', context_reward=True)
    

    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, context_reward=True)
    
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
    
    def compute_prediction_error(self, state):
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
            state = self.get_final_q_values(state)
            state = self.get_final_c_values(state)
            state = self.get_m_value(state)
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
            state = self.get_final_q_values(state)
            state = self.get_final_c_values(state)
            state = self.get_m_value(state)
            state = self.select_action(state)

        return state
    
    def sim_forward(self, state, phase = 'learning'):
        if phase == 'learning':
            state = self.get_q_value(state)
            state = self.get_c_value(state)
            state = self.get_m_value(state)
            state = self.get_context_value(state)
            state = self.select_action(state)
            state = self.compute_prediction_error(state)
            self.update_model(state)
        else:
            state = self.get_final_q_values(state)
            state = self.get_final_c_values(state)
            state = self.get_m_value(state)
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
        self.factual_lr, self.counterfactual_lr, self.contextual_lr, self.temperature, self.mixing_factor = x

        #Return the negative log likelihood of all observed actions
        return -self.fit_task(args, 'm_values', context_reward=True)


    def sim_func(self, *args):
        
        '''
        Simulate the model
        '''

        return self.sim_task(args, context_reward=True)
    
class RLModel:

    def __init__(self, model, fit_data=None):
        self.model = self.define_model(model, fit_data)

    def get_model(self):
        return self.model
    
    def get_parameters(self):
        return self.model.parameters.keys()
    
    def define_model(self, model, fit_data=None):

        if model == 'QLearning':
            factual_lr = 0.1 if fit_data is None else fit_data['factual_lr'].values[0]
            counterfactual_lr = 0.5 if fit_data is None else fit_data['counterfactual_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]

            model = QLearning(factual_lr=factual_lr, 
                            counterfactual_lr=counterfactual_lr, 
                            temperature=temperature)

            model.bounds = [(0.01, .99), (0.01, .99), (0.01, 10)]

        elif model == 'ActorCritic':
            factual_actor_lr = 0.1 if fit_data is None else fit_data['factual_actor_lr'].values[0]
            counterfactual_actor_lr = 0.05 if fit_data is None else fit_data['counterfactual_actor_lr'].values[0]
            critic_lr = 0.1 if fit_data is None else fit_data['critic_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]
            valence_factor = 0.5 if fit_data is None else fit_data['valence_factor'].values[0]
            
            model = ActorCritic(factual_actor_lr=factual_actor_lr,
                                counterfactual_actor_lr=counterfactual_actor_lr,
                                critic_lr=critic_lr,
                                temperature=temperature,
                                valence_factor=valence_factor)
            
            model.bounds = [(0.01, .99), (0.01, .99), (0.01,.99), (0.01, 10), (-1, 1)]

        elif model == 'Relative':
            factual_lr = 0.1 if fit_data is None else fit_data['factual_lr'].values[0]
            counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr'].values[0]
            contextual_lr = 0.1 if fit_data is None else fit_data['contextual_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]

            model = Relative(factual_lr=factual_lr,
                            counterfactual_lr=counterfactual_lr,
                            contextual_lr=contextual_lr,
                            temperature=temperature)
            
            model.bounds = [(0.01, .99), (0.01, .99), (0.01,.99), (0.01, 10)]

        elif model == 'Hybrid2012':
            factual_lr = 0.1 if fit_data is None else fit_data['factual_lr'].values[0]
            counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr'].values[0]
            factual_actor_lr = 0.1 if fit_data is None else fit_data['factual_actor_lr'].values[0]
            counterfactual_actor_lr = 0.05 if fit_data is None else fit_data['counterfactual_actor_lr'].values[0]
            critic_lr = 0.1 if fit_data is None else fit_data['critic_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]
            mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor'].values[0]
            valence_factor = 0.5 if fit_data is None else fit_data['valence_factor'].values[0]

            model = Hybrid2012(factual_lr=factual_lr,
                        counterfactual_lr=counterfactual_lr,
                        factual_actor_lr=factual_actor_lr,
                        counterfactual_actor_lr=counterfactual_actor_lr,
                        critic_lr=critic_lr,
                        temperature=temperature,
                        mixing_factor=mixing_factor,
                        valence_factor=valence_factor)
        
            model.bounds = [(0.01, .99), (0.01, .99), (0.01, .99), (0.01, .99), (0.01, .99), (0.01, 10), (0, 1), (-1, 1)]

        elif model == 'Hybrid2021':
            factual_lr = 0.1 if fit_data is None else fit_data['factual_lr'].values[0]
            counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr'].values[0]
            factual_actor_lr = 0.1 if fit_data is None else fit_data['factual_actor_lr'].values[0]
            counterfactual_actor_lr = 0.05 if fit_data is None else fit_data['counterfactual_actor_lr'].values[0]
            critic_lr = 0.1 if fit_data is None else fit_data['critic_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]
            mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor'].values[0]
            valence_factor = 0.5 if fit_data is None else fit_data['valence_factor'].values[0]
            noise_factor = 0.1 if fit_data is None else fit_data['noise_factor'].values[0]
            decay_factor = 0.1 if fit_data is None else fit_data['decay_factor'].values[0]

            model = Hybrid2021(factual_lr=factual_lr,
                        counterfactual_lr=counterfactual_lr,
                        factual_actor_lr=factual_actor_lr,
                        counterfactual_actor_lr=counterfactual_actor_lr,
                        critic_lr=critic_lr,
                        temperature=temperature,
                        mixing_factor=mixing_factor,
                        valence_factor=valence_factor,
                        noise_factor=noise_factor,
                        decay_factor=decay_factor)
            
            model.bounds = [(0.01, .99), (0.01, .99), (0.01, .99), (0.01, .99), (0.01, .99), (0.01, 10), (0, 1), (-1, 1), (0, 1), (0, 1)]
            
        elif model == 'QRelative':
            factual_lr = 0.1 if fit_data is None else fit_data['factual_lr'].values[0]
            counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr'].values[0]
            contextual_lr = 0.1 if fit_data is None else fit_data['contextual_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]
            mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor'].values[0]

            model = QRelative(factual_lr=factual_lr,
                            counterfactual_lr=counterfactual_lr,
                            contextual_lr=contextual_lr,
                            temperature=temperature,
                            mixing_factor=mixing_factor)
            
            model.bounds = [(0.01, .99), (0.01, .99), (0.01, .99), (0.01, 10), (0, 1)]
        
        elif model == 'wRelative':
            factual_lr = 0.1 if fit_data is None else fit_data['factual_lr'].values[0]
            counterfactual_lr = 0.05 if fit_data is None else fit_data['counterfactual_lr'].values[0]
            contextual_lr = 0.1 if fit_data is None else fit_data['contextual_lr'].values[0]
            temperature = 0.1 if fit_data is None else fit_data['temperature'].values[0]
            mixing_factor = 0.5 if fit_data is None else fit_data['mixing_factor'].values[0]

            model = wRelative(factual_lr=factual_lr,
                            counterfactual_lr=counterfactual_lr,
                            contextual_lr=contextual_lr,
                            temperature=temperature,
                            mixing_factor=mixing_factor)
            
            model.bounds = [(0.01, .99), (0.01, .99), (0.01, .99), (0.01, 10), (-1, 1)]
            
        else:
            raise ValueError('Model not recognized.')
        
        return model