import numpy as np
import pandas as pd
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

    def unpack_parameters(self, x):
        #TODO: Implement this
        if self.__class__.__name__ == 'QLearning':
            self.factual_lr, self.counterfactual_lr, self.temperature, *optionals = x

    def unpack_optionals(self, optionals):
        if self.optional_parameters['bias'] == True and 'valence_factor' in self.parameters:
            self.valence_factor = optionals[0]
            optionals = optionals[1:]
        if self.optional_parameters['novel'] == True and 'novel_value' in self.parameters:
            self.novel_value = optionals[0]
            optionals = optionals[1:]
        if self.optional_parameters['decay'] == True and 'decay_factor' in self.parameters:
            self.decay_factor = optionals[0]
            optionals = optionals[1:]

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
        if self.optional_parameters['decay']:
            state['q_values'] = self.get_decayed_q_values(state)
        else:
            state['q_values'] = [self.final_q_values[stim].values[0] for stim in state['stim_id']]
        return state

    def get_final_c_values(self, state):
        state['c_values'] = [self.final_c_values[stim].values[0] for stim in state['stim_id']]
        return state

    def get_context_value(self, state):
        state['context_value'] = self.context_values[state['state_id']]
        return state
    
    def get_final_w_values(self, state):
        if self.optional_parameters['decay']:
            state['w_values'] = self.get_decayed_w_values(state)
        else:
            state['w_values'] = [self.final_w_values[stim].values[0] for stim in state['stim_id']]
        return state
    
    def get_decayed_q_values(self, state):
        q_final = [self.final_q_values[stim] for stim in state['stim_id']]
        q_initial = [self.initial_q_values[stim] for stim in state['stim_id']]
        q_final_decayed = [q*(1-self.decay_factor) for q in q_final]
        q_initial_decayed = [q*(self.decay_factor) for q in q_initial]
        q_values = [q_final_decayed[i].values[0] + q_initial_decayed[i].values[0] for i in range(len(q_final))]
        return q_values
    
    def get_decayed_w_values(self, state):
        w_final = [self.final_w_values[stim] for stim in state['stim_id']]
        w_initial = [self.initial_w_values[stim] for stim in state['stim_id']]
        w_final_decayed = [w*(1-self.decay_factor) for w in w_final]
        w_initial_decayed = [w*(self.decay_factor) for w in w_initial]
        w_values = [w_final_decayed[i].values[0] + w_initial_decayed[i].values[0] for i in range(len(w_final))]
        return w_values

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
                reward[ri] = (1-self.valence_factor)*r
            elif r < 0:
                reward[ri] = self.valence_factor*r
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
            if self.novel_value is not None:
                self.final_v_values['N'] = self.novel_value
                self.final_w_values['N'] = self.novel_value
        elif 'Hybrid' in self.__class__.__name__:
            self.combine_q_values()
            self.combine_v_values()
            self.combine_w_values()
            if self.novel_value is not None:
                self.final_q_values['N'] = self.novel_value
                self.final_v_values['N'] = self.novel_value
                self.final_w_values['N'] = self.novel_value
        elif 'QRelative' == self.__class__.__name__:
            self.combine_q_values()
            self.combine_c_values()
            if self.novel_value is not None:
                self.final_q_values['N'] = self.novel_value
                self.final_c_values['N'] = self.novel_value
        else:
            self.combine_q_values()
            if self.novel_value is not None:
                self.final_q_values['N'] = self.novel_value

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
            uniform_dist = np.ones((len(probability_values)))/len(probability_values)
            probability_values = (((1-self.noise_factor)*probability_values).T + (self.noise_factor*uniform_dist)).T

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
    
    def fit_task(self, args, value_type, transform_reward=False, context_reward=False):
        
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
        if False: #Toggle to switch between methods for testing. Function method is slower than loop, so it's avoided
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
    
    def simulate(self, data):

        '''
        Simulate the model
        '''
        self.sim_func(data)

    def sim_task(self, args, transform_reward=False, context_reward=False):

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
