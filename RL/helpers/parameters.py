import os
import warnings

class Parameters:
    """
    Class to set parameters for model fitting and validation
    """

    def set_parameters(self, **kwargs: dict) -> None:
        
        """
        Assign parameters for FIT or validation mode.

        Parameters
        ----------
        kwargs : dict
            Parameter values.

        Raises
        ------
        ValueError
            If required parameters are missing or if mode is invalid.
        """

        # Skip setting parameters if 'help' is requested
        if 'help' in kwargs and kwargs['help']:
            self.help = help
            return None

        kwargs['mode'] = kwargs['mode'].upper()
        mode = kwargs['mode']
        if mode not in ['FIT', 'VALIDATION']:
            raise ValueError(f"Invalid mode '{mode}'. Mode must be 'fit' or 'validation'.")
        
        if mode == 'VALIDATION':
            if len(kwargs['models']) <= 1 and kwargs['recovery'] == 'model':
                raise ValueError("For model validation, at least two models must be specified.")

        # Define accepted and required parameters
        fit_params = [
            'help',
            'mode',
            'learning_filename',
            'transfer_filename',
            'models',
            'number_of_participants',
            'fixed',
            'bounds',
            'random_params',
            'number_of_runs',
            'generated',
            'multiprocessing',
            'training',
            'training_epochs',
            'optimizer_lr'
        ]
        
        validation_params = [
            'help',
            'mode',
            'models',
            'parameters',
            'learning_filename',
            'transfer_filename',
            'fit_filename',
            'task_design',
            'fixed',
            'bounds',
            'random_params',
            'datasets_to_generate',
            'number_of_runs',
            'number_of_participants',
            'multiprocessing',
            'generate_data',
            'clear_data',
            'recovery',
            'training',
            'training_epochs',
            'optimizer_lr'
        ]

        required_fit = ['mode', 'learning_filename', 'transfer_filename', 'models']
        required_validation = ['mode', 'models']

        accepted_params = fit_params if mode == 'FIT' else validation_params
        required_params = required_fit if mode == 'FIT' else required_validation

        # Warn about unknown parameters
        for key in kwargs:
            if key not in accepted_params:
                warnings.warn(f"Unknown parameter '{key}' is being ignored.", stacklevel=2)

        # Check for missing required parameters
        for param in required_params:
            if param not in kwargs:
                raise ValueError(f"Missing required parameter '{param}' for mode '{mode}'.")

        # Assign defaults and user values
        for param in accepted_params:
            default_value = {
                'help': False,
                'number_of_participants': 0,
                'fixed': None,
                'bounds': None,
                'parameters': None,
                'random_params': False,
                'number_of_runs': 1,
                'generated': False,
                'multiprocessing': False,
                'training': 'scipy',
                'training_epochs': 1000,
                'optimizer_lr': 0.01,
                'learning_filename': None,
                'transfer_filename': None,
                'parameters': None,
                'fit_filename': None,
                'task_design': None,
                'datasets_to_generate': 1,
                'generate_data': True,
                'clear_data': True,
                'recovery': 'parameter'
            }.get(param, None)
            setattr(self, param, kwargs.get(param, default_value))

        #Report all custom parameters (i.e. kwargs)
        print("User-defined parameters:")
        print('--------------------------\n')
        for key, value in kwargs.items():
            print(f"{key.title()}: {value}")
        print('--------------------------\n')

        