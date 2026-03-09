"""
Configuration for 135-model 3-state Kalman filter framework.
Compositional space: C(15) × S(3) × B(3) = 135 models.

C-Axis: Coherence modulation targets (Q1, Q2, Q3, R combinations)
S-Axis: Switch effects (baseline, x_reset, K_reset)
B-Axis: Bias mechanism (how bias state affects prediction/response)

Coherence Modulation:
    Q_i(c) = q_i_base × exp(α_qi × (1 - c))  for i ∈ {1, 2, 3}  (exponential)
    R(c) = 1 if c ≥ 0.5, r_low if c < 0.5  (binary step function)

Key difference from 54-model:
- Q3 (bias state process noise) is explicitly modeled
- B-axis has 3 levels for different bias mechanisms
- C0 (no coherence modulation) is excluded - all models have coherence effects
- R modulation uses binary step function (not exponential)
"""
import numpy as np
from itertools import product

# =============================================================================
# C-AXIS: Coherence Modulation Targets (15 levels)
# =============================================================================
# Each combination specifies which parameters are modulated by coherence
# Q modulations use exponential form:
#   Q_i(c) = q_i_base × exp(α_qi × (1 - c))  for i ∈ {1, 2, 3}
# R modulation uses binary step function:
#   R(c) = 1 if c ≥ 0.5, r_low if c < 0.5
# Note: C0 (no coherence modulation) is excluded

C_AXIS = {
    'C1':  {'name': 'Q1',         'Q1': True,  'Q2': False, 'Q3': False, 'R': False,
            'description': 'Fast state process noise modulated'},
    'C2':  {'name': 'Q2',         'Q1': False, 'Q2': True,  'Q3': False, 'R': False,
            'description': 'Slow state process noise modulated'},
    'C3':  {'name': 'Q3',         'Q1': False, 'Q2': False, 'Q3': True,  'R': False,
            'description': 'Bias state process noise modulated'},
    'C4':  {'name': 'R',          'Q1': False, 'Q2': False, 'Q3': False, 'R': True,
            'description': 'Measurement noise modulated'},
    'C5':  {'name': 'Q1_R',       'Q1': True,  'Q2': False, 'Q3': False, 'R': True,
            'description': 'Fast state + measurement noise'},
    'C6':  {'name': 'Q2_R',       'Q1': False, 'Q2': True,  'Q3': False, 'R': True,
            'description': 'Slow state + measurement noise'},
    'C7':  {'name': 'Q3_R',       'Q1': False, 'Q2': False, 'Q3': True,  'R': True,
            'description': 'Bias state + measurement noise'},
    'C8':  {'name': 'Q1_Q2',      'Q1': True,  'Q2': True,  'Q3': False, 'R': False,
            'description': 'Fast + slow state process noise'},
    'C9':  {'name': 'Q1_Q3',      'Q1': True,  'Q2': False, 'Q3': True,  'R': False,
            'description': 'Fast + bias state process noise'},
    'C10': {'name': 'Q2_Q3',      'Q1': False, 'Q2': True,  'Q3': True,  'R': False,
            'description': 'Slow + bias state process noise'},
    'C11': {'name': 'Q1_Q2_R',    'Q1': True,  'Q2': True,  'Q3': False, 'R': True,
            'description': 'Fast + slow + measurement'},
    'C12': {'name': 'Q1_Q3_R',    'Q1': True,  'Q2': False, 'Q3': True,  'R': True,
            'description': 'Fast + bias + measurement'},
    'C13': {'name': 'Q2_Q3_R',    'Q1': False, 'Q2': True,  'Q3': True,  'R': True,
            'description': 'Slow + bias + measurement'},
    'C14': {'name': 'Q1_Q2_Q3',   'Q1': True,  'Q2': True,  'Q3': True,  'R': False,
            'description': 'All process noise modulated'},
    'C15': {'name': 'Q1_Q2_Q3_R', 'Q1': True,  'Q2': True,  'Q3': True,  'R': True,
            'description': 'All parameters modulated'},
}

# =============================================================================
# S-AXIS: Task Switch Effect (3 levels)
# =============================================================================
S_AXIS = {
    'S0': {
        'name': 'baseline',
        'description': 'No task switching effect',
        'target': None
    },
    'S1': {
        'name': 'x_reset',
        'description': 'Switch resets slow state toward prior mean',
        'target': 'state'
    },
    'S2': {
        'name': 'K_reset',
        'description': 'Switch reduces Kalman gain',
        'target': 'gain'
    },
}

# =============================================================================
# B-AXIS: Bias Mechanism (3 levels)
# =============================================================================
# Controls how bias state b affects prediction and response:
#   F[0,2]: whether b affects prediction (μ⁻ = m vs μ⁻ = m + b)
#   response_bias: whether b is added to response (y = μ + d0 vs y = μ + b + d0)

B_AXIS = {
    'B1': {
        'name': 'B1',
        'description': 'Bias in prediction only (F[0,2]=0, response=μ+d0)',
        'F_02': 0.0,           # μ⁻ = m
        'response_bias': False  # y = μ + d0
    },
    'B2': {
        'name': 'B2',
        'description': 'Bias in prediction via F (F[0,2]=1, response=μ+d0)',
        'F_02': 1.0,           # μ⁻ = m + b
        'response_bias': False  # y = μ + d0
    },
    'B3': {
        'name': 'B3',
        'description': 'Bias in response only (F[0,2]=0, response=μ+b+d0)',
        'F_02': 0.0,           # μ⁻ = m
        'response_bias': True   # y = μ + b + d0
    },
}

# =============================================================================
# MODEL ENUMERATION
# =============================================================================

N_MODELS = len(C_AXIS) * len(S_AXIS) * len(B_AXIS)  # 15 × 3 × 3 = 135


def get_model_name(c_id, s_id, b_id):
    """Generate standardized model name."""
    c_name = C_AXIS[c_id]['name']
    s_name = S_AXIS[s_id]['name']
    b_name = B_AXIS[b_id]['name']
    return f"C_{c_name}__S_{s_name}__B_{b_name}"


def get_model_by_id(c_id, s_id, b_id):
    """Get specific model configuration by axis IDs."""
    return {
        'model_id': f"{c_id}_{s_id}_{b_id}",
        'model_name': get_model_name(c_id, s_id, b_id),
        'c_id': c_id,
        's_id': s_id,
        'b_id': b_id,
        'c_config': C_AXIS[c_id],
        's_config': S_AXIS[s_id],
        'b_config': B_AXIS[b_id],
    }


def get_all_models(include_switching=True):
    """
    Generate all 135 model combinations.

    Parameters
    ----------
    include_switching : bool
        If True, return all 135 models (15 C × 3 S × 3 B)
        If False, return only baseline switching (15 C × 1 S × 3 B = 45)

    Returns
    -------
    models : list of dict
    """
    c_ids = list(C_AXIS.keys())
    s_ids = list(S_AXIS.keys()) if include_switching else ['S0']
    b_ids = list(B_AXIS.keys())

    models = []
    for c_id, s_id, b_id in product(c_ids, s_ids, b_ids):
        models.append(get_model_by_id(c_id, s_id, b_id))

    return models


def get_models_by_axis(axis, value):
    """Get all models with a specific axis value."""
    all_models = get_all_models()
    return [m for m in all_models if m[axis] == value]


def print_model_space():
    """Print summary of the 135-model space."""
    print("=" * 70)
    print("THREE-STATE 135-MODEL COMPOSITIONAL SPACE")
    print("=" * 70)
    print(f"\nTotal: N = |C| × |S| × |B| = {len(C_AXIS)} × {len(S_AXIS)} × {len(B_AXIS)} = {N_MODELS}")

    print("\n" + "-" * 70)
    print("C-AXIS: Coherence Modulation Targets (15 levels)")
    print("-" * 70)
    print(f"{'ID':<5} {'Name':<12} {'Q1':<4} {'Q2':<4} {'Q3':<4} {'R':<4} Description")
    print("-" * 70)
    for cid, cfg in C_AXIS.items():
        q1 = '✓' if cfg['Q1'] else '-'
        q2 = '✓' if cfg['Q2'] else '-'
        q3 = '✓' if cfg['Q3'] else '-'
        r = '✓' if cfg['R'] else '-'
        print(f"{cid:<5} {cfg['name']:<12} {q1:<4} {q2:<4} {q3:<4} {r:<4} {cfg['description']}")

    print("\n" + "-" * 70)
    print("S-AXIS: Switch Effect (3 levels)")
    print("-" * 70)
    for sid, cfg in S_AXIS.items():
        print(f"  {sid}: {cfg['name']:10s} - {cfg['description']}")

    print("\n" + "-" * 70)
    print("B-AXIS: Bias Mechanism (3 levels)")
    print("-" * 70)
    for bid, cfg in B_AXIS.items():
        print(f"  {bid}: {cfg['name']:5s} - {cfg['description']}")

    print("\n" + "-" * 70)
    print("Coherence Modulation Formulas:")
    print("  Q_i(c) = q_i_base × exp(α_qi × (1 - c))  for i ∈ {1, 2, 3}")
    print("    c=0.7: Q = Q_base × exp(0.3α)")
    print("    c=0.3: Q = Q_base × exp(0.7α)")
    print("  R(c) = 1 if c ≥ 0.5, r_low if c < 0.5  (binary step)")
    print("=" * 70)


# Name mapping for convenience
NAME_TO_ID = {
    # C-axis (by modulation pattern) - C0 (None) excluded
    'Q1': 'C1', 'Q2': 'C2', 'Q3': 'C3', 'R': 'C4',
    'Q1_R': 'C5', 'Q2_R': 'C6', 'Q3_R': 'C7',
    'Q1_Q2': 'C8', 'Q1_Q3': 'C9', 'Q2_Q3': 'C10',
    'Q1_Q2_R': 'C11', 'Q1_Q3_R': 'C12', 'Q2_Q3_R': 'C13',
    'Q1_Q2_Q3': 'C14', 'Q1_Q2_Q3_R': 'C15',
    # S-axis
    'baseline': 'S0', 'x_reset': 'S1', 'K_reset': 'S2',
    # B-axis
    'B1': 'B1', 'B2': 'B2', 'B3': 'B3',
}


def get_model_by_names(c_name, s_name, b_name):
    """Get model by human-readable names."""
    c_id = NAME_TO_ID.get(c_name, c_name)
    s_id = NAME_TO_ID.get(s_name, s_name)
    b_id = NAME_TO_ID.get(b_name, b_name)
    return get_model_by_id(c_id, s_id, b_id)
