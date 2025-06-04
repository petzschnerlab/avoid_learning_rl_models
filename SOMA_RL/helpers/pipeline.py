from typing import Optional
import random as rnd

from . import Master


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

    def run(self, mode: str, **kwargs):

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

        mode = mode.upper()
        self.set_parameters(mode=mode, **kwargs)

        if mode == 'FIT':
            self.run_fit_empirical()
            self.export_fits(path="SOMA_RL/reports")           
        elif mode == 'VALIDATION':
            self.run_recovery()
            self.export_recovery(path="SOMA_RL/reports")
        else:
            raise ValueError(f"Invalid mode '{mode}'. Mode must be 'fit' or 'validation'.")