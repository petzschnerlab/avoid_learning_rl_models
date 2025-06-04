from .parameters import Parameters
from .analyses import Analyses
from .export import Export


class Master(Parameters,
             Analyses,
             Export):
    
    """
    Class to hold all functions for the SOMA project
    """

    def __init__(self):
        super().__init__()
