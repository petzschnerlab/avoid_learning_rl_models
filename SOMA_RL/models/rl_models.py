import random as rnd
import numpy as np

from models.standard import QLearning, ActorCritic
from models.relative import Relative, wRelative, QRelative
from models.hybrid import Hybrid2012, Hybrid2021

class RLModel:

    def __init__(self, model, parameters=None, random_params=False):

        self.random_params = random_params
        
        model_name = model
        fit_bias = True if '+bias' in model else False
        fit_novel = True if '+novel' in model else False
        fit_decay = True if '+decay' in model else False

        model = model.replace('+bias', '')
        model = model.replace('+novel', '')
        model = model.replace('+decay', '')

        self.model = self.define_model(model, parameters, fit_bias, fit_novel, fit_decay)
        self.model.model_name = model_name
        self.model.optional_parameters = {'bias': fit_bias, 'novel': fit_novel, 'decay': fit_decay}

        #Remove any optional parameters that are not being used
        linked_keys = {'bias': 'valence_factor', 'novel': 'novel_value', 'decay': 'decay_factor'}
        for opt_key, param_key in linked_keys.items():
            if not self.model.optional_parameters.get(opt_key, True) and param_key in self.model.parameters:
                self.model.parameters.pop(param_key)
                self.model.bounds.pop(param_key)

        #Reorder optional parameters so that they are always in same order
        parameter_keys = [key for key in self.model.parameters.keys() if key not in linked_keys.values()]
        if self.model.optional_parameters['bias']:
            parameter_keys.append('valence_factor')
        if self.model.optional_parameters['novel']:
            parameter_keys.append('novel_value')
        if self.model.optional_parameters['decay']:
            parameter_keys.append('decay_factor')
        self.model.parameters = {key: self.model.parameters[key] for key in parameter_keys}
        self.model.bounds = {key: self.model.bounds[key] for key in parameter_keys}

    def get_model(self):
        return self.model
    
    def get_bounds(self):
        return self.model.bounds
    
    def get_parameters(self):
        return self.model.parameters.keys()
    
    def starting_param(self, fixed_param=None, bounds=None):

        if self.random_params:
            return np.round(rnd.uniform(bounds[0], bounds[1]),2)
        else:
            return fixed_param
    
    def define_model(self, model, parameters=None, fit_bias=False, fit_novel=False, fit_decay=False):

        if model == 'QLearning':
            bounds = {'factual_lr': (0.01, .99), 
                      'counterfactual_lr': (0.01, .99), 
                      'temperature': (0.01, 10), 
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
                        
            factual_lr = self.starting_param(0.1, bounds['factual_lr'])                     if parameters is None else parameters['factual_lr'].values[0]
            counterfactual_lr = self.starting_param(0.5, bounds['counterfactual_lr'])       if parameters is None else parameters['counterfactual_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])                   if parameters is None else parameters['temperature'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                     if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])                   if not fit_decay or parameters is None else parameters['decay_factor'].values[0]

            model = QLearning(factual_lr=factual_lr, 
                            counterfactual_lr=counterfactual_lr, 
                            temperature=temperature,
                            novel_value=novel_value,
                            decay_factor=decay_factor)
            model.bounds = bounds
            
        elif model == 'ActorCritic':

            bounds = {'factual_actor_lr': (0.01, .99),
                      'counterfactual_actor_lr': (0.01, .99),
                      'critic_lr': (0.01, .99),
                      'temperature': (0.01, 10),
                      'valence_factor': (0, 1),
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
            
            factual_actor_lr = self.starting_param(0.1, bounds['factual_actor_lr'])                     if parameters is None else parameters['factual_actor_lr'].values[0]
            counterfactual_actor_lr = self.starting_param(0.05, bounds['counterfactual_actor_lr'])      if parameters is None else parameters['counterfactual_actor_lr'].values[0]
            critic_lr = self.starting_param(0.1, bounds['critic_lr'])                                   if parameters is None else parameters['critic_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])                               if parameters is None else parameters['temperature'].values[0]
            valence_factor = self.starting_param(0.5, bounds['valence_factor'])                         if not fit_bias or parameters is None else parameters['valence_factor'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                                 if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])                               if not fit_decay or parameters is None else parameters['decay_factor'].values[0]
            
            model = ActorCritic(factual_actor_lr=factual_actor_lr,
                                counterfactual_actor_lr=counterfactual_actor_lr,
                                critic_lr=critic_lr,
                                temperature=temperature,
                                valence_factor=valence_factor,
                                novel_value=novel_value,
                                decay_factor=decay_factor)
            model.bounds = bounds

        elif model == 'Relative':

            bounds = {'factual_lr': (0.01, .99),
                      'counterfactual_lr': (0.01, .99),
                      'contextual_lr': (0.01, .99),
                      'temperature': (0.01, 10),
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
            
            factual_lr = self.starting_param(0.1, bounds['factual_lr'])                     if parameters is None else parameters['factual_lr'].values[0]
            counterfactual_lr = self.starting_param(0.05, bounds['counterfactual_lr'])      if parameters is None else parameters['counterfactual_lr'].values[0]
            contextual_lr = self.starting_param(0.1, bounds['contextual_lr'])               if parameters is None else parameters['contextual_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])                   if parameters is None else parameters['temperature'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                     if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])                   if not fit_decay or parameters is None else parameters['decay_factor'].values[0]

            model = Relative(factual_lr=factual_lr,
                            counterfactual_lr=counterfactual_lr,
                            contextual_lr=contextual_lr,
                            temperature=temperature,
                            novel_value=novel_value,
                            decay_factor=decay_factor)
            model.bounds = bounds

        elif model == 'Hybrid2012':

            bounds = {'factual_lr': (0.01, .99),
                      'counterfactual_lr': (0.01, .99),
                      'factual_actor_lr': (0.01, .99),
                      'counterfactual_actor_lr': (0.01, .99),
                      'critic_lr': (0.01, .99),
                      'temperature': (0.01, 10),
                      'mixing_factor': (0, 1),
                      'valence_factor': (0, 1),
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
            
            factual_lr = self.starting_param(0.1, bounds['factual_lr'])                                 if parameters is None else parameters['factual_lr'].values[0]
            counterfactual_lr = self.starting_param(0.05, bounds['counterfactual_lr'])                  if parameters is None else parameters['counterfactual_lr'].values[0]
            factual_actor_lr = self.starting_param(0.1, bounds['factual_actor_lr'])                     if parameters is None else parameters['factual_actor_lr'].values[0]
            counterfactual_actor_lr = self.starting_param(0.05, bounds['counterfactual_actor_lr'])      if parameters is None else parameters['counterfactual_actor_lr'].values[0]
            critic_lr = self.starting_param(0.1, bounds['critic_lr'])                                   if parameters is None else parameters['critic_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])                               if parameters is None else parameters['temperature'].values[0]
            mixing_factor = self.starting_param(0.5, bounds['mixing_factor'])                           if parameters is None else parameters['mixing_factor'].values[0]
            valence_factor = self.starting_param(0.5, bounds['valence_factor'])                         if not fit_bias or parameters is None else parameters['valence_factor'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                                 if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])                               if not fit_decay or parameters is None else parameters['decay_factor'].values[0]

            model = Hybrid2012(factual_lr=factual_lr,
                        counterfactual_lr=counterfactual_lr,
                        factual_actor_lr=factual_actor_lr,
                        counterfactual_actor_lr=counterfactual_actor_lr,
                        critic_lr=critic_lr,
                        temperature=temperature,
                        mixing_factor=mixing_factor,
                        valence_factor=valence_factor,
                        novel_value=novel_value,
                        decay_factor=decay_factor)
            model.bounds = bounds

        elif model == 'Hybrid2021':

            bounds = {'factual_lr': (0.01, .99),
                      'counterfactual_lr': (0.01, .99),
                      'factual_actor_lr': (0.01, .99),
                      'counterfactual_actor_lr': (0.01, .99),
                      'critic_lr': (0.01, .99),
                      'temperature': (0.01, 10),
                      'mixing_factor': (0, 1),
                      'noise_factor': (0, 1),
                      'valence_factor': (0, 1),
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
            
            factual_lr = self.starting_param(0.1, bounds['factual_lr'])                             if parameters is None else parameters['factual_lr'].values[0]
            counterfactual_lr = self.starting_param(0.05, bounds['counterfactual_lr'])              if parameters is None else parameters['counterfactual_lr'].values[0]
            factual_actor_lr = self.starting_param(0.1, bounds['factual_actor_lr'])                 if parameters is None else parameters['factual_actor_lr'].values[0]
            counterfactual_actor_lr = self.starting_param(0.05, bounds['counterfactual_actor_lr'])  if parameters is None else parameters['counterfactual_actor_lr'].values[0]
            critic_lr = self.starting_param(0.1, bounds['critic_lr'])                               if parameters is None else parameters['critic_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])                           if parameters is None else parameters['temperature'].values[0]
            mixing_factor = self.starting_param(0.5, bounds['mixing_factor'])                       if parameters is None else parameters['mixing_factor'].values[0]
            noise_factor = self.starting_param(0.1, bounds['noise_factor'])                         if parameters is None else parameters['noise_factor'].values[0]
            valence_factor = self.starting_param(0.5, bounds['valence_factor'])                     if not fit_bias or parameters is None else parameters['valence_factor'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                             if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])                           if not fit_decay or parameters is None else parameters['decay_factor'].values[0]

            model = Hybrid2021(factual_lr=factual_lr,
                        counterfactual_lr=counterfactual_lr,
                        factual_actor_lr=factual_actor_lr,
                        counterfactual_actor_lr=counterfactual_actor_lr,
                        critic_lr=critic_lr,
                        temperature=temperature,
                        mixing_factor=mixing_factor,
                        noise_factor=noise_factor,
                        valence_factor=valence_factor,
                        novel_value=novel_value,
                        decay_factor=decay_factor)
            model.bounds = bounds
            
        elif model == 'QRelative':

            bounds = {'factual_lr': (0.01, .99),
                      'counterfactual_lr': (0.01, .99),
                      'contextual_lr': (0.01, .99),
                      'temperature': (0.01, 10),
                      'mixing_factor': (0, 1),
                      'valence_reward': (0, 1),
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
            
            factual_lr = self.starting_param(0.1, bounds['factual_lr'])                 if parameters is None else parameters['factual_lr'].values[0]
            counterfactual_lr = self.starting_param(0.05, bounds['counterfactual_lr'])  if parameters is None else parameters['counterfactual_lr'].values[0]
            contextual_lr = self.starting_param(0.1, bounds['contextual_lr'])           if parameters is None else parameters['contextual_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])               if parameters is None else parameters['temperature'].values[0]
            mixing_factor = self.starting_param(0.5, bounds['mixing_factor'])           if parameters is None else parameters['mixing_factor'].values[0]
            valence_factor = self.starting_param(0.5, bounds['valence_reward'])         if not fit_bias or parameters is None else parameters['valence_factor'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                 if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])               if not fit_decay or parameters is None else parameters['decay_factor'].values[0]

            model = QRelative(factual_lr=factual_lr,
                            counterfactual_lr=counterfactual_lr,
                            contextual_lr=contextual_lr,
                            temperature=temperature,
                            mixing_factor=mixing_factor,
                            valence_reward=valence_factor,
                            novel_value=novel_value,
                            decay_factor=decay_factor)
            model.bounds = bounds

        elif model == 'wRelative':

            bounds = {'factual_lr': (0.01, .99),
                      'counterfactual_lr': (0.01, .99),
                      'contextual_lr': (0.01, .99),
                      'temperature': (0.01, 10),
                      'mixing_factor': (0, 1),
                      'valence_factor': (0, 1),
                      'novel_value': (-1, 1),
                      'decay_factor': (0, 1)}
            
            factual_lr = self.starting_param(0.1, bounds['factual_lr'])                 if parameters is None else parameters['factual_lr'].values[0]
            counterfactual_lr = self.starting_param(0.05, bounds['counterfactual_lr'])  if parameters is None else parameters['counterfactual_lr'].values[0]
            contextual_lr = self.starting_param(0.1, bounds['contextual_lr'])           if parameters is None else parameters['contextual_lr'].values[0]
            temperature = self.starting_param(0.1, bounds['temperature'])               if parameters is None else parameters['temperature'].values[0]
            mixing_factor = self.starting_param(0.5, bounds['mixing_factor'])           if parameters is None else parameters['mixing_factor'].values[0]
            valence_factor = self.starting_param(0.5, bounds['valence_factor'])         if not fit_bias or parameters is None else parameters['valence_factor'].values[0]
            novel_value = self.starting_param(0, bounds['novel_value'])                 if not fit_novel or parameters is None else parameters['novel_value'].values[0]
            decay_factor = self.starting_param(0, bounds['decay_factor'])               if not fit_decay or parameters is None else parameters['decay_factor'].values[0]
            
            model = wRelative(factual_lr=factual_lr,
                            counterfactual_lr=counterfactual_lr,
                            contextual_lr=contextual_lr,
                            temperature=temperature,
                            mixing_factor=mixing_factor,
                            valence_factor=valence_factor,
                            novel_value=novel_value,
                            decay_factor=decay_factor)
            model.bounds = bounds
            
        else:
            raise ValueError(f'Model {model} not recognized.')
                
        return model