import random as rnd
import numpy as np

from models.standard import QLearning, ActorCritic
from models.relative import Relative, wRelative, QRelative
from models.hybrid import Hybrid2012, Hybrid2021

class RLModel:

    def __init__(self, model, parameters=None, random_params=False, fixed=None, bounds=None):

        self.random_params = random_params

        model_name = model

        #Determine optimal parameters, remove them from model name
        fit_bias = True if '+bias' in model else False
        fit_novel = True if '+novel' in model else False
        fit_decay = True if '+decay' in model else False
        self.optional_parameters = {'bias': fit_bias, 'novel': fit_novel, 'decay': fit_decay}
        model = model.replace('+bias', '')
        model = model.replace('+novel', '')
        model = model.replace('+decay', '')

        #Set fixed and bounds parameters
        self.fixed, self.bounds = self.get_default_parameters()
        if fixed is not None:
            if model in fixed:
                self.fixed[model].update(fixed[model])
        if bounds is not None:
            if model in bounds:
                self.bounds[model].update(bounds[model])

        #Define model
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
    
    def get_model_parameters(self):
        model_parameters = {}
        model_parameters['QLearning'] = {'factual_lr', 
                                         'counterfactual_lr',
                                         'temperature',
                                         'novel_value',
                                         'decay_factor'}
        
        model_parameters['ActorCritic'] = {'factual_actor_lr',
                                           'counterfactual_actor_lr',
                                           'critic_lr',
                                           'temperature',
                                           'valence_factor',
                                           'novel_value',
                                           'decay_factor'}
        
        model_parameters['Relative'] = {'factual_lr',
                                        'counterfactual_lr',
                                        'contextual_lr',
                                        'temperature',
                                        'novel_value',
                                        'decay_factor'}

        model_parameters['Hybrid2012'] = {'factual_lr',
                                            'counterfactual_lr',
                                            'factual_actor_lr',
                                            'counterfactual_actor_lr',
                                            'critic_lr',
                                            'temperature',
                                            'mixing_factor',
                                            'valence_factor',
                                            'novel_value',
                                            'decay_factor'}
        
        model_parameters['Hybrid2021'] = {'factual_lr',
                                            'counterfactual_lr',
                                            'factual_actor_lr',
                                            'counterfactual_actor_lr',
                                            'critic_lr',
                                            'temperature',
                                            'mixing_factor',
                                            'noise_factor',
                                            'valence_factor',
                                            'novel_value',
                                            'decay_factor'}
        
        model_parameters['QRelative'] = {'factual_lr',
                                        'counterfactual_lr',
                                        'contextual_lr',
                                        'temperature',
                                        'mixing_factor',
                                        'valence_reward',
                                        'novel_value',
                                        'decay_factor'}
        
        model_parameters['wRelative'] = {'factual_lr',
                                        'counterfactual_lr',
                                        'contextual_lr',
                                        'temperature',
                                        'mixing_factor',
                                        'valence_factor',
                                        'novel_value',
                                        'decay_factor'}
    
        return model_parameters

    def get_default_parameters(self):
        return self.get_default_fixed(), self.get_default_bounds()

    def get_default_fixed(self):

        parameters_fixed = {'factual_lr': 0.1,
                            'counterfactual_lr': 0.05,
                            'factual_actor_lr': 0.1,
                            'counterfactual_actor_lr': 0.05,
                            'critic_lr': 0.1,
                            'contextual_lr': 0.1,
                            'temperature': 0.1,
                            'mixing_factor': 0.5,
                            'noise_factor': 0.1,
                            'valence_factor': 0.5,
                            'valence_reward': 0.5,
                            'novel_value': 0,
                            'decay_factor': 0}
        
        model_parameters = self.get_model_parameters()
        default_fixed = {}
        for model in model_parameters:
            default_fixed[model] = {param: parameters_fixed[param] for param in model_parameters[model]}

        return default_fixed

    def get_default_bounds(self):

        parameters_bounds = {'factual_lr': (0.01, .99),
                             'counterfactual_lr': (0.01, .99),
                             'factual_actor_lr': (0.01, .99),
                             'counterfactual_actor_lr': (0.01, .99),
                             'critic_lr': (0.01, .99),
                             'contextual_lr': (0.01, .99),
                             'temperature': (0.01, 10),
                             'mixing_factor': (0, 1),
                             'noise_factor': (0, 1),
                             'valence_factor': (0, 1),
                             'valence_reward': (0, 1),
                             'novel_value': (-1, 1),
                             'decay_factor': (0, 1)}

        model_parameters = self.get_model_parameters()
        default_bounds = {}
        for model in model_parameters:
            default_bounds[model] = {param: parameters_bounds[param] for param in model_parameters[model]}
        
        return default_bounds
    
    def define_model(self, model, parameters=None):

        model_classes = {'QLearning': QLearning,
                         'ActorCritic': ActorCritic,
                         'Relative': Relative,
                         'wRelative': wRelative,
                         'QRelative': QRelative,
                         'Hybrid2012': Hybrid2012,
                         'Hybrid2021': Hybrid2021}
        
        if model not in model_classes:
            raise ValueError(f'Model {model} not recognized.')
        
        bounds = self.bounds[model]
        model_params = self.define_params(self.fixed[model], self.bounds[model], parameters)
        model = model_classes[model](**model_params)
        model.bounds = bounds
                
        return model