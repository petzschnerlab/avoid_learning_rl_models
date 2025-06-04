from typing import Optional
import random as rnd

from . import Master
from .help import Help


class Pipeline(Master):

    """
    Pipeline for running reinforcement learning model fitting and validation.
    This class inherits from the Master class, which contains all the necessary
    methods for fitting and validating models.
    """

    def __init__(self, seed: Optional[int] = None):

        """
        Initializes the Pipeline class.

        Parameters
        ----------
        seed : Optional[int]
            Seed for random number generation. If None, no seed is set.
            Default is None.
        """

        if seed is not None:
            rnd.seed(seed)
            
        super().__init__()

    def run(self, mode: str = None, **kwargs):

        """
        Runs the pipeline in the specified mode.

        Parameters
        ----------
        mode : str
            'FIT' or 'validation' to specify the processing mode.
        kwargs : dict
            Additional parameters for the fitting or validation process.
        
        Raises
        ------
        ValueError
            If the mode is invalid or if required parameters are missing.
        """

        # Setup the parameters
        self.set_parameters(mode=mode, **kwargs)

        #Run the help
        if self.help:
            help = Help()
            help.print_help()
            return None

        mode = mode.upper()
        if mode == 'FIT':
            self.run_fit()
            self.export_fits(path="SOMA_RL/reports")           
        elif mode == 'VALIDATION':
            self.run_validation()
            self.export_recovery(path="SOMA_RL/reports")
        else:
            raise ValueError(f"Invalid mode '{mode}'. Mode must be 'fit' or 'validation'.")