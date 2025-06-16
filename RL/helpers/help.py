class Help:
    
    """
    Class to display help information for the model fitting and validation package.
    """

    def __init__(self):
       
        """
        Initialize the Help class with parameter descriptions for both FIT and VALIDATION modes.
        """

        # TODO: ADD OPTIONAL/REQUIRED TAGS

        self.pipeline_parameters = [
            'seed'
        ]

        self.fit_parameters = [
            'learning_filename',
            'transfer_filename',
            'models',
            'fixed',
            'bounds',
            'random_params',
            'number_of_runs',
            'generated',
            'multiprocessing',
            'training',
            'training_epochs',
            'optimizer_lr',
            'number_of_participants'
        ]

        self.validation_parameters = [
            'models',
            'learning_filename',
            'transfer_filename',
            'parameters',
            'fit_filename',
            'random_params',
            'task_design',
            'fixed',
            'bounds',
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

        self.parameter_descriptions = {
            'seed':
                ['Seed for random number generation. If None, no seed is set.',
                'int | None',
                None],

            'learning_filename': 
                ['Filename (and path) of the learning task data.',
                'str | None',
                None],

            'transfer_filename':
                ['Filename (and path) of the transfer task data.',
                'str | None',
                None],

            'models':
                [
    '''
    List of models to fit.
            
    Supported models: 
        QLearning, ActorCritic
        Relative, wRelative
        Hybrid2012, Hybrid2021, StandardHybrid2012, StandardHybrid2021

    Standard models:
        QLearning: Standard Q-Learning Model
        ActorCritic: Standard Actor-Critic Model
        Relative: Standard Relative Model (Palminteri et al., 2015)
        wRelative: Simplified Relative Model (Williams et al., in prep)
        Hybrid2012+bias: Standard Hybrid 2012 Model (Gold et al., 2012)
        Hybrid2021+bias+decay: Standard Hybrid 2021 Model (Geana et al., 2021)

    Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign
        +bias: Adds a valence bias to the model (e.g. Hybrid2012+bias), only usable with Hybrid2012, and Hybrid2021
        +novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models
        +decay: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models
    ''', 
                'list[str] | None',
                None],

            'fixed': 
                [('Optional fixed parameter values. '
                    'For the FIT mode, these values will be used as the model parameters if random_params = False. '
                    'If random_params = \'normal\', these values will be used as the mean of the normal distribution. '
                    'If random_params = \'random\', these values will be ignored. '
                    'For the VALIDATION mode, these values will be ignored if the parameter or fit_filename parameters are used. '
                    'Otherwise, these values will be used in the same way as in the FIT mode.'),
                'dict | None',
                None],

            'bounds': 
                [('Bounds that each parameter is cutoff at. Operates in both fitting and simulating data. '
                    'It is a nested dictionairy in the form of {model: {parameter: tuple}}. '
                    'The tuple are two floats representing the bottom and top bounds, e.g., bounds[\'QLearning\'][\'factual_lr\'] = (0.1, 0.99)'),
                'dict | None',
                None],

            'random_params': 
                [('Mode of determining parameter starting points. '
                    'Can be \'normal\', \'random\' or False. ' #TODO: Change False to 'none' (or something)
                    'If \'normal\' starting parameter values will be drawn from a normal distribution with the means being defined in the fixed parameter. '
                    'The parameters will be cutoff at the bounds defined in the bounds parameter. '
                    'If \'random\' is selected, the parameters will be drawn from a uniform distribution between the bounds. '
                    'If no fixed or bound parameters are provided, the default values will be used.'),
                'bool | str',
                False],

            'number_of_runs': 
                [('How many times to fit each model per participant. Two outputs when fitting data is the fit_data.pkl and full_fit_data.pkl files. '
                    'This works well with randomized starting points (e.g., random_params = \'random\' or random_params = \'normal\') '
                    'because each run has a different set of starting parameters, which helps finding the best fit parameters. '
                    'The fit_data.pkl file contains the best run for each model and participant, while the full_fit_data.pkl file contains all runs. '),
                'int',
                1],

            'generated': 
                ['Use generated data instead of empirical.',
                'bool',
                False],

            'multiprocessing':
                ['Use multiprocessing for parallel computation.',
                'bool',
                False],
            
            'training': 
                [('Training backend to use [scipy, torch]. '
                    'The pytorch backend is on beta testing. It works, but performs worse than the scipy backend. '
                    'There has not yet been an investigation into why this is the case. '
                    'If using the torch backend, the training_epochs and optimizer_lr parameters are used. '
                    'These are ignored if the scipy backend is used. '),
                'str', 
                'scipy'], 

            'training_epochs': 
                ['If using torch backend (training = \'torch\'), this determines the number of training epochs.',
                'int',
                1000],

            'optimizer_lr':
                ['If using torch backend (training = \'torch\'), this is the learning rate for the ADAM optimizer (which is the only one implemented at this time).',
                'float',
                0.01],
            
            'number_of_participants': 
                [('TEST PARAM. This parameter is used to cut your provided dataset down. It will cut it down to this number of participants. '
                    'It will take the first N participants from the dataset, where N is the number of participants you inputted. '
                    'If 0 is inputted (default), it will keep all participants. This is designed mostly for testing.'),
                'int',
                0],

            # HERE:
            'parameters':['Parameters to use for simulation or evaluation.', 'dict | list[dict]', None],
            'fit_filename': ['Filename of pre-fit parameter results.', 'str | None', None],
            'task_design': ['Task structure or setup for generating new data.', 'dict | None', None],
            'datasets_to_generate': ['Number of datasets to generate.', 'int', 1],
            'generate_data': ['Whether to generate new data.', 'bool', True],
            'clear_data': ['Clear previous simulation data before generating new.', 'bool', True],
            'recovery': ['Type of recovery analysis [parameter, model].', 'str', 'parameter'],
        }

    def print_help(self) -> None:
        
        """
        Print the help information, including overview and parameter descriptions.

        Returns
        -------
        None
        """

        self.print_overview()
        self.print_parameters()

    def print_overview(self) -> None:
        
        """
        Print an overview of the package and usage instructions.

        Returns
        -------
        None
        """

        overview = """\nThe Avoid Learning RL Models package is a tool to deploy reinforcement learning computational models.
        The package is designed to be run as a function with parameters that can be set to customize the modeling procedure.
        This package is a standalone package but can also be used as a companion package to the SOMA_AL package.
        As a companion package, moves all relevant data, figures, etc. to a folder named 'report' that should be copy pasted into SOMA_AL/modelling
        This package supports two processing modes: FIT and VALIDATION.

        - FIT mode is used to fit computational models to empirical data.
        - VALIDATION mode is used to test parameter recovery and model recovery.

        The easiest implementation of this package is:
            from helpers.pipeline import Pipeline

            kwargs = {
                'mode: 'fit',
                'models': ['QLearning']',
                'learning_filename': 'data/learning_data.csv',
                'transfer_filename': 'data/transfer_data.csv'}

            pipeline = Pipeline()
            pipeline.run(**kwargs)

        Please note: The docstrings within the classes and methods of this package are mostly built using AI. They should be alright, but there may be errors
        However, this help function was written by hand and should be accurate. If there is any conflict between the two, believe the help function.
        If the help function is not clear, or wrong, sorry! Feel free to open an issue on the GitHub repository or make changes and submit a pull request.
        """

        print(overview)

    def print_parameters(self) -> None:
        
        """
        Print the parameters for both FIT and VALIDATION modes.

        Returns
        -------
        None
        """
        print('\n==============================================================')
        print('==============================================================')
        print('PIPELINE OBJECT PARAMETERS')
        print('-------------------------\n')
        print('from helpers.pipeline import Pipeline')
        print('pipeline = Pipeline(**params)')
        print('==============================================================\n')
        for param in self.pipeline_parameters:
            desc = self.parameter_descriptions[param]
            print(f'{param}\n  DESCRIPTION: {desc[0]}\n  TYPE: {desc[1]}\n  DEFAULT: {desc[2]}\n')

        print('\n==============================================================')
        print('FIT MODE PARAMETERS')
        print('-------------------\n')
        print('from helpers.pipeline import Pipeline')
        print('pipeline = Pipeline()')
        print('pipeline.run(mode=\'fit\', **params)')
        print('==============================================================\n')
        for param in self.fit_parameters:
            desc = self.parameter_descriptions[param]
            print(f'{param}\n  DESCRIPTION: {desc[0]}\n  TYPE: {desc[1]}\n  DEFAULT: {desc[2]}\n')

        print('\n==============================================================')
        print('VALIDATION MODE PARAMETERS')
        print('--------------------------\n')
        print('from helpers.pipeline import Pipeline')
        print('pipeline = Pipeline()')
        print('pipeline.run(mode=\'validation\', **params)')
        print('==============================================================\n')
        for param in self.validation_parameters:
            desc = self.parameter_descriptions[param]
            print(f'{param}\n  DESCRIPTION: {desc[0]}\n  TYPE: {desc[1]}\n  DEFAULT: {desc[2]}\n')

        print('==============================================================\n')
