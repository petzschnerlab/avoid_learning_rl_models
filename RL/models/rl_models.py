from typing import Union
import random as rnd
import numpy as np

from helpers.priors import get_priors
from models.standard import QLearning, ActorCritic
from models.relative import Relative
from models.advantage import Advantage
from models.hybrid import Hybrid2012, Hybrid2021, StandardHybrid2012, StandardHybrid2021

class RLModel:

    def __init__(self,
                 model: str = None, 
                 parameters: dict = None, 
                 fixed: dict = None, 
                 bounds: dict = None, 
                 random_params: Union[bool, str] = False) -> None:

        """
        Initializes the RLModel class with a specified model and its parameters.
        
        Parameters
        ----------
        model : str, optional
            The name of the model to be used. If None, no model is set.
        parameters : dict, optional
            A dictionary of parameters for the model. If None, default parameters are used.
        random_params : bool or str, optional
            If False, fixed parameters are initialized randomly within bounds.
            If 'random', parameters are initialized randomly within bounds.
            If 'normal', parameters are initialized using a normal distribution around the fixed parameter.     
        """

        if model is not None:
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
            self.fixed, _ = get_priors()
            self.bounds = self.get_default_bounds()
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
            if not self.model.optional_parameters['novel']:
                self.model.novel_value = None
            
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

    def get_model(self) -> object:
        
        """
        Returns the model object.

        Returns
        -------
        object
            The model object initialized in the class.
        """

        return self.model
    
    def get_bounds(self) -> dict:
        
        """
        Returns the bounds of the model parameters.

        Returns
        -------
        dict
            A dictionary containing the bounds for each parameter of the model.
        """

        return self.model.bounds
    
    def get_parameters(self) -> dict:
        
        """
        Returns the parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the parameters of the model.
        """

        return self.model.parameters.keys()
    
    def get_n_parameters(self) -> int:
        """
        Returns the number of parameters in the model.

        Returns
        -------
        int
            The number of parameters in the model.
        """

        return len(self.model.parameters)
    
    def starting_param(self, fixed_param: float = None,
                       bounds: tuple = None) -> float:

        """
        Returns a starting parameter value based on the fixed parameter and bounds.

        Parameters
        ----------
        fixed_param : float, optional
            The fixed parameter value. If None, a random value is generated.
        bounds : tuple, optional
            A tuple containing the lower and upper bounds for the parameter. If None, defaults are used.

        Returns
        -------
        float
            A starting parameter value, either fixed or randomly generated within the specified bounds.
        """

        if self.random_params == 'random':
            return np.round(rnd.uniform(bounds[0], bounds[1]),2)
        elif self.random_params == 'normal':
            std = (bounds[1]-bounds[0])/15 # TODO: make this a parameter
            param = np.round(np.random.normal(fixed_param, std), 2)
            return np.clip(param, bounds[0], bounds[1])
        else:
            return fixed_param
        
    def define_params(self, fixed: dict,
                      bounds: dict,
                      parameters: dict = None) -> dict:

        """
        Defines the model parameters based on fixed values, bounds, and optional user parameters.
        
        Parameters
        ----------
        fixed : dict
            A dictionary of fixed parameters for the model.
        bounds : dict
            A dictionary of bounds for the parameters.
        parameters : dict, optional
            A dictionary of user-defined parameters. If None, fixed parameters are used.
        
        Returns
        -------
        dict
            A dictionary of model parameters, either from fixed values or user-defined parameters.
        """
        
        model_params = {}
        for param in fixed:

            #Check parameter type
            bool_parameter = parameters is None
            bool_valence = param == 'valence_factor' and not self.optional_parameters['bias']
            bool_novel = param == 'novel_value' and not self.optional_parameters['novel']
            bool_decay = param == 'decay_factor' and not self.optional_parameters['decay']

            if bool_parameter or bool_valence or bool_novel or bool_decay:
                model_params[param] = self.starting_param(fixed[param], bounds[param]) 
            else:
                model_params[param] = parameters[param].values[0]

        return model_params
    
    def get_model_parameters(self) -> dict:
        
        """
        Returns a dictionary of model parameters for each model type.
        The keys are model names and the values are lists of parameter names.
        This method defines the parameters used in different reinforcement learning models.
        
        Returns
        -------
        dict
            A dictionary where keys are model names and values are lists of parameter names.
        """

        model_parameters = {}
        model_parameters['QLearning'] = ['factual_lr', 
                                         'counterfactual_lr',
                                         'temperature',
                                         'novel_value',
                                         'decay_factor']
        
        model_parameters['ActorCritic'] = ['factual_actor_lr',
                                           'counterfactual_actor_lr',
                                           'critic_lr',
                                           'temperature',
                                           'valence_factor',
                                           'novel_value',
                                           'decay_factor']
        
        model_parameters['Relative'] = ['factual_lr',
                                        'counterfactual_lr',
                                        'contextual_lr',
                                        'temperature',
                                        'novel_value',
                                        'decay_factor']
        
        model_parameters['Advantage'] = ['factual_lr',
                                           'counterfactual_lr',
                                           'temperature',
                                           'weighting_factor',
                                           'novel_value',
                                           'decay_factor']

        model_parameters['Hybrid2012'] = ['factual_lr',
                                            'counterfactual_lr',
                                            'factual_actor_lr',
                                            'counterfactual_actor_lr',
                                            'critic_lr',
                                            'temperature',
                                            'mixing_factor',
                                            'valence_factor',
                                            'novel_value',
                                            'decay_factor']
        
        model_parameters['Hybrid2021'] = ['factual_lr',
                                            'counterfactual_lr',
                                            'factual_actor_lr',
                                            'counterfactual_actor_lr',
                                            'critic_lr',
                                            'temperature',
                                            'mixing_factor',
                                            'noise_factor',
                                            'valence_factor',
                                            'novel_value',
                                            'decay_factor']
        
        model_parameters['StandardHybrid2012'] = model_parameters['Hybrid2012'].copy()
        model_parameters['StandardHybrid2012'].remove('counterfactual_lr')
        model_parameters['StandardHybrid2012'].remove('counterfactual_actor_lr')

        model_parameters['StandardHybrid2021'] = model_parameters['Hybrid2021'].copy()
        model_parameters['StandardHybrid2021'].remove('counterfactual_lr')
        model_parameters['StandardHybrid2021'].remove('counterfactual_actor_lr')
    
        return model_parameters
    
    def get_model_columns(self, model: str = None) -> Union[dict, list]:

        """
        Returns the columns for the model parameters, including metadata and model-specific parameters.

        Parameters
        ----------
        model : str, optional
            The name of the model for which to get the columns. If None, returns columns for all models.
        
        Returns
        -------
        dict or list
            If model is specified, returns a list of columns for that model.
            If model is None, returns a dictionary with model names as keys and lists of columns as values.
        """

        metadata = ['participant', 'pain_group', 'run', 'fit']
        model_parameters = self.get_model_parameters()
        #check if class has attribute optional_parameters
        if model is not None and hasattr(self.model, 'optional_parameters'):
            linked_keys = {'bias': 'valence_factor', 'novel': 'novel_value', 'decay': 'decay_factor'}
            for key in self.model.optional_parameters:
                if not self.model.optional_parameters[key] and linked_keys[key] in model_parameters[model]:
                    model_parameters[model].remove(linked_keys[key])

        columns = {model: metadata + model_parameters[model] for model in model_parameters}
        if model is not None:
            return columns[model]
        else:
            return columns

    def get_default_parameters(self) -> tuple:

        """
        Returns the default fixed parameters and bounds for the models.
        
        Returns
        -------
        tuple
            A tuple containing two dictionaries: the default fixed parameters and the default bounds for the models.
        """
        return self.get_default_fixed(), self.get_default_bounds()

    def get_default_fixed(self) -> dict:

        """
        Returns the default fixed parameters for the models.
        
        Returns
        -------
        dict
            A dictionary containing fixed parameters for various models.
        """

        parameters_fixed = {'factual_lr': 0.1,
                            'counterfactual_lr': 0.05,
                            'factual_actor_lr': 0.1,
                            'counterfactual_actor_lr': 0.05,
                            'critic_lr': 0.1,
                            'contextual_lr': 0.1,
                            'temperature': 0.1,
                            'mixing_factor': 0.5,
                            'weighting_factor': 0.5,
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

    def get_default_bounds(self) -> dict:

        """
        Returns the default bounds for the model parameters.

        Returns
        -------
        dict
            A dictionary containing the bounds for various model parameters.
        """

        parameters_bounds = {'factual_lr': (0.01, .99),
                             'counterfactual_lr': (0.01, .99),
                             'factual_actor_lr': (0.01, .99),
                             'counterfactual_actor_lr': (0.01, .99),
                             'critic_lr': (0.01, .99),
                             'contextual_lr': (0.01, .99),
                             'temperature': (0.01, 1),
                             'mixing_factor': (0, 1),
                             'weighting_factor': (0, 1),
                             'noise_factor': (0, .2),
                             'valence_factor': (0, 1),
                             'valence_reward': (0, 1),
                             'novel_value': (-1, 1),
                             'decay_factor': (0, .2)}

        model_parameters = self.get_model_parameters()
        default_bounds = {}
        for model in model_parameters:
            default_bounds[model] = {param: parameters_bounds[param] for param in model_parameters[model]}
        
        return default_bounds
    
    def define_model(self, model, parameters=None):

        """
        Defines the model based on the specified model name and parameters.

        Parameters
        ----------
        model : str
            The name of the model to be defined.
        parameters : dict, optional
            A dictionary of parameters for the model. If None, parameters are defined by self.random_params.

        Returns
        -------
        object
            An instance of the specified model class with the defined parameters.
        """

        model_classes = {'QLearning': QLearning,
                         'ActorCritic': ActorCritic,
                         'Relative': Relative,
                         'Advantage': Advantage,
                         'Hybrid2012': Hybrid2012,
                         'Hybrid2021': Hybrid2021,
                         'StandardHybrid2012': StandardHybrid2012,
                         'StandardHybrid2021': StandardHybrid2021}
        
        if model not in model_classes:
            raise ValueError(f'Model {model} not recognized.')
        
        bounds = self.bounds[model]
        model_params = self.define_params(self.fixed[model], self.bounds[model], parameters)
        model = model_classes[model](**model_params)
        model.bounds = bounds
                
        return model