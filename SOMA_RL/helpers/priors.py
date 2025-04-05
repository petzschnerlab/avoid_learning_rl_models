def get_priors():
    return fixed_priors(), bounds_priors()

def fixed_priors():
    fixed = {
        'QLearning': {  # From Palminteri et al., 2015
            'factual_lr': 0.28,
            'counterfactual_lr': 0.18,
            'temperature': 0.06,
            'decay_factor': 0.08, # From Geana et al., 2021:
            'novel_value': .50, # Custom
        },

        'ActorCritic': {  # From Geana et al., 2021's Hybrid2021 model
            'factual_actor_lr': .33,
            'counterfactual_actor_lr': .33,
            'critic_lr': .48,
            'temperature': .06,
            'valence_factor': .33, # From Geana et al., 2021:
            'decay_factor': .08,
            'novel_value': .50, # Custom
        },

        'Relative': {  # From Palminteri et al., 2015
            'factual_lr': 0.19,
            'counterfactual_lr': 0.15,
            'contextual_lr': 0.33,
            'temperature': 0.05,
            'decay_factor': 0.08, # From Geana et al., 2021:
            'novel_value': .50, # Custom
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
            'novel_value': .50, # Custom
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
            'novel_value': .50, # Custom
        },
    }

    return fixed

def bounds_priors():
    return None