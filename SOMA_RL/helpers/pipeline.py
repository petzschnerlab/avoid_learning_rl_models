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

    def run_fit(self, **kwargs):
        self.run_fit_empirical(**kwargs)
        self.export_fits(path="SOMA_RL/reports")           

    def run_validation(self, **kwargs):
        self.run_recovery(**kwargs)
        self.export_recovery(path="SOMA_RL/reports")