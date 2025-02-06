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
        self.optional_parameters = {'bias': fit_bias, 'novel': fit_novel, 'decay': fit_decay}

        model = model.replace('+bias', '')
        model = model.replace('+novel', '')
        model = model.replace('+decay', '')

        self.model = self.define_model(model, parameters)
        self.model.model_name = model_name
        self.model.optional_parameters = self.optional_parameters

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
        
    def define_params(self, fixed, bounds, parameters=None):
        
        model_params = {}
        for param in fixed:
            if parameters is None or param == 'valence_factor' and not self.optional_parameters['bias'] or param == 'novel_value' and not self.optional_parameters['novel'] or param == 'decay_factor' and not self.optional_parameters['decay']:
                model_params[param] = self.starting_param(fixed[param], bounds[param]) 
            else:
                model_params[param] = parameters[param].values[0]

        return model_params
    
    def define_model(self, model, parameters=None):

        if model == 'QLearning':

            fixed =  {'factual_lr':             0.1,
                      'counterfactual_lr':      0.05,
                      'temperature':            0.1,
                      'novel_value':            0,
                      'decay_factor':           0}
             
            bounds = {'factual_lr':             (0.01, .99), 
                      'counterfactual_lr':      (0.01, .99), 
                      'temperature':            (0.01, 10), 
                      'novel_value':            (-1, 1),
                      'decay_factor':           (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = QLearning(**model_params)
            model.bounds = bounds
            
        elif model == 'ActorCritic':

            fixed =  {'factual_actor_lr':           0.1,
                      'counterfactual_actor_lr':    0.05,
                      'critic_lr':                  0.1,
                      'temperature':                0.1,
                      'valence_factor':             0.5,
                      'novel_value':                0,
                      'decay_factor':               0}

            bounds = {'factual_actor_lr':           (0.01, .99),
                      'counterfactual_actor_lr':    (0.01, .99),
                      'critic_lr':                  (0.01, .99),
                      'temperature':                (0.01, 10),
                      'valence_factor':             (0, 1),
                      'novel_value':                (-1, 1),
                      'decay_factor':               (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = ActorCritic(**model_params)
            model.bounds = bounds

        elif model == 'Relative':

            fixed =  {'factual_lr':             0.1,
                      'counterfactual_lr':      0.05,
                      'contextual_lr':          0.1,
                      'temperature':            0.1,
                      'novel_value':            0,
                      'decay_factor':           0}

            bounds = {'factual_lr':             (0.01, .99),
                      'counterfactual_lr':      (0.01, .99),
                      'contextual_lr':          (0.01, .99),
                      'temperature':            (0.01, 10),
                      'novel_value':            (-1, 1),
                      'decay_factor':           (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = Relative(**model_params)
            model.bounds = bounds

        elif model == 'Hybrid2012':

            fixed =  {'factual_lr':                 0.1,
                      'counterfactual_lr':          0.05,
                      'factual_actor_lr':           0.1,
                      'counterfactual_actor_lr':    0.05,
                      'critic_lr':                  0.1,
                      'temperature':                0.1,
                      'mixing_factor':              0.5,
                      'valence_factor':             0.5,
                      'novel_value':                0,
                      'decay_factor':               0}
            
            bounds = {'factual_lr':                 (0.01, .99),
                      'counterfactual_lr':          (0.01, .99),
                      'factual_actor_lr':           (0.01, .99),
                      'counterfactual_actor_lr':    (0.01, .99),
                      'critic_lr':                  (0.01, .99),
                      'temperature':                (0.01, 10),
                      'mixing_factor':              (0, 1),
                      'valence_factor':             (0, 1),
                      'novel_value':                (-1, 1),
                      'decay_factor':               (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = Hybrid2012(**model_params)
            model.bounds = bounds

        elif model == 'Hybrid2021':

            fixed =  {'factual_lr':                 0.1,
                      'counterfactual_lr':          0.05,
                      'factual_actor_lr':           0.1,
                      'counterfactual_actor_lr':    0.05,
                      'critic_lr':                  0.1,
                      'temperature':                0.1,
                      'mixing_factor':              0.5,
                      'noise_factor':               0.1,
                      'valence_factor':             0.5,
                      'novel_value':                0,
                      'decay_factor':               0}
            
            bounds = {'factual_lr':                 (0.01, .99),
                      'counterfactual_lr':          (0.01, .99),
                      'factual_actor_lr':           (0.01, .99),
                      'counterfactual_actor_lr':    (0.01, .99),
                      'critic_lr':                  (0.01, .99),
                      'temperature':                (0.01, 10),
                      'mixing_factor':              (0, 1),
                      'noise_factor':               (0, 1),
                      'valence_factor':             (0, 1),
                      'novel_value':                (-1, 1),
                      'decay_factor':               (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = Hybrid2021(**model_params)
            model.bounds = bounds
            
        elif model == 'QRelative':
 
            fixed =  {'factual_lr':             0.1,
                      'counterfactual_lr':      0.05,
                      'contextual_lr':          0.1,
                      'temperature':            0.1,
                      'mixing_factor':          0.5,
                      'valence_reward':         0.5,
                      'novel_value':            0,
                      'decay_factor':           0}

            bounds = {'factual_lr':             (0.01, .99),
                      'counterfactual_lr':      (0.01, .99),
                      'contextual_lr':          (0.01, .99),
                      'temperature':            (0.01, 10),
                      'mixing_factor':          (0, 1),
                      'valence_reward':         (0, 1),
                      'novel_value':            (-1, 1),
                      'decay_factor':           (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = QRelative(**model_params)
            model.bounds = bounds

        elif model == 'wRelative':

            fixed =  {'factual_lr':             0.1,
                      'counterfactual_lr':      0.05,
                      'contextual_lr':          0.1,
                      'temperature':            0.1,
                      'mixing_factor':          0.5,
                      'valence_factor':         0.5,
                      'novel_value':            0,
                      'decay_factor':           0}
            
            bounds = {'factual_lr':             (0.01, .99),
                      'counterfactual_lr':      (0.01, .99),
                      'contextual_lr':          (0.01, .99),
                      'temperature':            (0.01, 10),
                      'mixing_factor':          (0, 1),
                      'valence_factor':         (0, 1),
                      'novel_value':            (-1, 1),
                      'decay_factor':           (0, 1)}
            
            model_params = self.define_params(fixed, bounds, parameters)
            model = wRelative(**model_params)
            model.bounds = bounds
            
        else:
            raise ValueError(f'Model {model} not recognized.')
                
        return model