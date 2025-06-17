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

    def run(self, **kwargs) -> None:

        """
        Runs the pipeline in the specified mode.

        Parameters
        ----------
        kwargs : dict
            Additional parameters for the fitting or validation process.
        
        Raises
        ------
        ValueError
            If the mode is invalid or if required parameters are missing.
        """

        # Setup the parameters
        self.set_parameters(**kwargs)

        #Run the help
        if self.help:
            help = Help()
            help.print_help()
            return None

        # Run the fit or validation process based on the mode
        if self.mode == 'FIT':
            self.run_fit()
            self.export_fits(path="RL/modelling")           
        elif self.mode == 'VALIDATION':
            if len(self.models) <= 1:
                raise ValueError("For model validation, at least two models must be specified.")
            self.run_validation()
            self.export_recovery(path="RL/modelling")
        else:
            raise ValueError(f"Invalid mode '{self.mode}'. Mode must be 'fit' or 'validation'.")