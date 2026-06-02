"""
Parameter configuration for 135-model 3-state Kalman filter framework.
NO LOG TRANSFORM VERSION - same parameter bounds as log version.
"""

from .config import C_AXIS, S_AXIS, B_AXIS

# =============================================================================
# BASE PARAMETERS (always present in all models)
# =============================================================================
BASE_PARAMS = {
    'names': ['q1', 'q2', 'q3', 'lambda', 'r_base', 'd0', 'alpha_d0'],
    'p0':    [0.6,  0.03, 1.3,  0.5,      1.7,      0.08, -0.1],
    'lower': [0.0,  0.0,  0.0,  0.0,      0.1,     -1.0, -2.0],
    'upper': [20.0, 20.0, 5.0,  1.0,      10.0,     1.0,  2.0],
}

# =============================================================================
# C-AXIS PARAMETERS (Coherence Modulation)
# =============================================================================
C_MODULATION_PARAMS = {
    'Q1': {'name': 'alpha_q1', 'p0': 0.0, 'lower': -5.0, 'upper': 5.0},
    'Q2': {'name': 'alpha_q2', 'p0': 0.0, 'lower': -5.0, 'upper': 5.0},
    'Q3': {'name': 'alpha_q3', 'p0': 0.0, 'lower': -5.0, 'upper': 5.0},
    'R':  {'name': 'r_low',    'p0': 1.1, 'lower': 0.1,  'upper': 10.0},
}

# =============================================================================
# S-AXIS PARAMETERS (Switch Effect)
# =============================================================================
S_PARAMS = {
    'S0': {
        'names': [],
        'p0': [],
        'lower': [],
        'upper': []
    },
    'S1': {
        'names': ['x_reset'],
        'p0': [0.3],
        'lower': [0.0],
        'upper': [1.0]
    },
    'S2': {
        'names': ['k_reset'],
        'p0': [0.3],
        'lower': [0.0],
        'upper': [1.0]
    },
}

# =============================================================================
# B-AXIS PARAMETERS (Bias Mechanism)
# =============================================================================
B_PARAMS = {
    'B1': {'names': [], 'p0': [], 'lower': [], 'upper': []},
    'B2': {'names': [], 'p0': [], 'lower': [], 'upper': []},
    'B3': {'names': [], 'p0': [], 'lower': [], 'upper': []},
}


def get_parameter_config(c_id, s_id, b_id):
    """Get full parameter configuration for a (C, S, B) model."""
    c_cfg = C_AXIS[c_id]

    names = BASE_PARAMS['names'].copy()
    p0 = BASE_PARAMS['p0'].copy()
    lower = BASE_PARAMS['lower'].copy()
    upper = BASE_PARAMS['upper'].copy()

    for mod_key in ['Q1', 'Q2', 'Q3', 'R']:
        if c_cfg[mod_key]:
            mod_param = C_MODULATION_PARAMS[mod_key]
            names.append(mod_param['name'])
            p0.append(mod_param['p0'])
            lower.append(mod_param['lower'])
            upper.append(mod_param['upper'])

    s_cfg = S_PARAMS[s_id]
    names.extend(s_cfg['names'])
    p0.extend(s_cfg['p0'])
    lower.extend(s_cfg['lower'])
    upper.extend(s_cfg['upper'])

    b_cfg = B_PARAMS[b_id]
    names.extend(b_cfg['names'])
    p0.extend(b_cfg['p0'])
    lower.extend(b_cfg['lower'])
    upper.extend(b_cfg['upper'])

    return {
        'names': names,
        'p0': p0,
        'lower': lower,
        'upper': upper,
        'n_params': len(names),
    }


def parse_parameters(par, c_id, s_id, b_id):
    """Parse parameter vector into named dictionary."""
    config = get_parameter_config(c_id, s_id, b_id)

    params = {
        'q1': 0.1,
        'q2': 0.01,
        'q3': 0.1,
        'lambda': 0.5,
        'r_base': 1.0,
        'd0': 0.0,
        'alpha_d0': 0.0,
        'alpha_q1': 0.0,
        'alpha_q2': 0.0,
        'alpha_q3': 0.0,
        'r_low': 1.0,
        'x_reset': 0.0,
        'k_reset': 0.0,
    }

    for i, name in enumerate(config['names']):
        params[name] = par[i]

    return params


def get_param_count_by_model(c_id, s_id, b_id):
    """Get number of free parameters for a model."""
    config = get_parameter_config(c_id, s_id, b_id)
    return config['n_params']


def print_parameter_table():
    """Print parameter counts for all 135 models."""
    print("Parameter counts by model (no-log version):")
    print("-" * 60)
    for c_id in C_AXIS:
        for s_id in S_AXIS:
            for b_id in B_AXIS:
                k = get_param_count_by_model(c_id, s_id, b_id)
                print(f"{c_id}_{s_id}_{b_id}: {k} params")


def get_parameter_bounds_summary():
    """Return summary of all parameter bounds."""
    return {
        'Base': BASE_PARAMS,
        'C_modulation': C_MODULATION_PARAMS,
        'S_params': S_PARAMS,
    }
