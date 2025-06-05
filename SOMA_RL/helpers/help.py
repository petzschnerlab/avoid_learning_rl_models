class Help:
    
    """
    Class to display help information for the model fitting and validation package.
    """

    def __init__(self):
       
        """
        Initialize the Help class with parameter descriptions for both FIT and VALIDATION modes.
        """

        self.parameter_descriptions = {
            'FIT': {
                'learning_filename': ['Filename of the learning task data.', 'str', None],
                'transfer_filename': ['Filename of the transfer task data.', 'str', None],
                'models': ['List of models to fit.', 'list[str]', None],
                'number_of_participants': ['Number of participants to simulate or analyze.', 'int', 0],
                'fixed': ['Optional fixed parameter values.', 'dict | None', None],
                'bounds': ['Optional parameter bounds.', 'dict | None', None],
                'random_params': ['Whether to use random initialization for parameters.', 'bool', False],
                'number_of_runs': ['How many times to fit each participant/model.', 'int', 1],
                'generated': ['Use generated data instead of empirical.', 'bool', False],
                'multiprocessing': ['Use multiprocessing for parallel computation.', 'bool', False],
                'training': ['Training backend to use [scipy, torch].', 'str', 'scipy'],
                'training_epochs': ['Number of training epochs (if applicable).', 'int', 1000],
                'optimizer_lr': ['Learning rate for the optimizer (if applicable).', 'float', 0.01]
            },

            'VALIDATION': {
                'models': ['List of models to validate.', 'list[str]', None],
                'parameters': ['Parameters to use for simulation or evaluation.', 'dict | list[dict]', None],
                'learning_filename': ['Filename of the learning task data.', 'str | None', None],
                'transfer_filename': ['Filename of the transfer task data.', 'str | None', None],
                'fit_filename': ['Filename of pre-fit parameter results.', 'str | None', None],
                'task_design': ['Task structure or setup for generating new data.', 'dict | None', None],
                'fixed': ['Optional fixed parameter values.', 'dict | None', None],
                'bounds': ['Optional parameter bounds.', 'dict | None', None],
                'datasets_to_generate': ['Number of datasets to generate.', 'int', 1],
                'number_of_runs': ['How many times to simulate each model.', 'int', 1],
                'number_of_participants': ['Number of participants per dataset.', 'int', 0],
                'multiprocessing': ['Use multiprocessing for parallel computation.', 'bool', False],
                'generate_data': ['Whether to generate new data.', 'bool', True],
                'clear_data': ['Clear previous simulation data before generating new.', 'bool', True],
                'recovery': ['Type of recovery analysis [parameter, model].', 'str', 'parameter'],
                'training': ['Training backend to use [scipy, torch].', 'str', 'torch'],
                'training_epochs': ['Number of training epochs (if applicable).', 'int', 1000],
                'optimizer_lr': ['Learning rate for the optimizer (if applicable).', 'float', 0.01]
            }
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

        overview = """The SOMA_RL package is a tool to deploy reinforcement learning computational models.
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
        """

        print(overview)

    def print_parameters(self) -> None:
        
        """
        Print the parameters for both FIT and VALIDATION modes.

        Returns
        -------
        None
        """
        print('\nPipeline Object Parameter')
        print('-------------------')
        print('the pipeline object can take one parameter, which is the seed for random number generation.\n')

        
        print('\nFIT Mode Parameters')
        print('-------------------')
        for param, desc in self.parameter_descriptions['FIT'].items():
            print(f'{param}\n  Description: {desc[0]}\n  Type: {desc[1]}\n  Default: {desc[2]}\n')

        print('\nVALIDATION Mode Parameters')
        print('--------------------------')
        for param, desc in self.parameter_descriptions['VALIDATION'].items():
            print(f'{param}\n  Description: {desc[0]}\n  Type: {desc[1]}\n  Default: {desc[2]}\n')

        print('-------------------------------------------------------------')
