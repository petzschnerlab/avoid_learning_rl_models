def get_priors(models: list = None) -> tuple:

    """
    Returns fixed and bounds priors for the models used in SOMA-RL.
    Fixed priors are based on previous literature, while bounds priors are currently set to None.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - fixed: Fixed priors for the models.
        - bounds: Bounds priors for the models (currently None).
    """

    return fixed_priors(models=models), bounds_priors(models=models)

def fixed_priors(models: list = None) -> dict:

    """
    Returns fixed priors for the models used in RL.
    Fixed priors are based on previous literature and custom settings.

    Returns
    -------
    dict
        A dictionary containing fixed priors for various models.
    """

    fixed = {
        'QLearning': {  # From Palminteri et al., 2015
            'factual_lr': 0.28,
            'counterfactual_lr': 0.18,
            'temperature': 0.06,
            'decay_factor': 0.08, # From Geana et al., 2021:
            'novel_value': 0, # Custom
        },

        'ActorCritic': {  # From Geana et al., 2021's Hybrid2021 model
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': .06,
            'valence_factor': .33, # From Geana et al., 2021:
            'decay_factor': .08,
            'novel_value': 0, # Custom
        },

        'Relative': {  # From Palminteri et al., 2015
            'factual_lr': 0.19,
            'counterfactual_lr': 0.15,
            'contextual_lr': 0.33,
            'temperature': 0.05,
            'decay_factor': 0.08, # From Geana et al., 2021:
            'novel_value': 0, # Custom
        },

        'Hybrid2012': {  # From Geana et al., 2021:
            'factual_lr': 0.49,
            'counterfactual_lr': 0.49,
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': 0.06,
            'mixing_factor': 0.7, # From Gold et al., 2012
            'valence_factor': 0.33,
            'decay_factor': 0.08,
            'novel_value': 0, # Custom
        },

        'Hybrid2021': { # From Geana et al., 2021
            'factual_lr': 0.49,
            'counterfactual_lr': 0.49,
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': 0.06,
            'mixing_factor': 0.69,
            'noise_factor': 0.04,
            'valence_factor': 0.33, # From Geana et al., 2021:
            'decay_factor': 0.08, # From Geana et al., 2021:
            'novel_value': 0, # Custom
        },

        'Advantage': {  # From Palminteri et al., 2015
            'factual_lr': 0.28,
            'counterfactual_lr': 0.18,
            'temperature': 0.06,
            'weighting_factor': 0.5, # Custom
            'decay_factor': 0.08, # From Geana et al., 2021:
            'novel_value': 0, # Custom
        },
    }

    fixed['StandardHybrid2012'] = fixed['Hybrid2012'].copy()
    fixed['StandardHybrid2012'].pop('counterfactual_lr')
    fixed['StandardHybrid2012'].pop('counterfactual_actor_lr')

    fixed['StandardHybrid2021'] = fixed['Hybrid2021'].copy()
    fixed['StandardHybrid2021'].pop('counterfactual_lr')
    fixed['StandardHybrid2021'].pop('counterfactual_actor_lr')

    if models is not None:
        fixed = {model.split('+')[0]: fixed[model.split('+')[0]] for model in models if model.split('+')[0] in fixed}

    return fixed

def bounds_priors(models: list = None) -> dict:

    """
    Returns bounds priors for the models used in RL.
    Bounds priors are currently set to None, indicating no specific bounds are defined.
    
    Returns
    -------
    dict
        A dictionary containing bounds priors for various models (currently None).
    """

    return None