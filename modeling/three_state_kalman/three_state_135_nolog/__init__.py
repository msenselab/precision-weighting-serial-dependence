"""
135-Model Three-State Kalman Filter Framework
NO LOG TRANSFORM VERSION - operates in original duration space.

This version removes the log transform to avoid bias in d0 estimation.
Suitable for narrow duration ranges (e.g., 0.8-1.6s).
"""

from .config import (
    C_AXIS,
    S_AXIS,
    B_AXIS,
    N_MODELS,
    get_model_by_id,
    get_all_models,
    print_model_space,
)

from .parameters import (
    get_parameter_config,
    parse_parameters,
    get_param_count_by_model,
)

from .engine import (
    run_3state_kf,
    generate_predictions,
    simulate_responses,
)

from .fitting import (
    fit_single_model,
    fit_all_models_subject,
    fit_all_subjects,
    fit_all_subjects_incremental,
    get_best_model,
    rank_models,
    compare_axes,
    generate_ppc_single_subject,
    generate_ppc_all_subjects,
    compute_ppc_metrics,
)

__all__ = [
    # Config
    'C_AXIS', 'S_AXIS', 'B_AXIS', 'N_MODELS',
    'get_model_by_id', 'get_all_models', 'print_model_space',
    # Parameters
    'get_parameter_config', 'parse_parameters', 'get_param_count_by_model',
    # Engine
    'run_3state_kf', 'generate_predictions', 'simulate_responses',
    # Fitting & comparison
    'fit_single_model', 'fit_all_models_subject', 'fit_all_subjects',
    'fit_all_subjects_incremental',
    'get_best_model', 'rank_models', 'compare_axes',
    'generate_ppc_single_subject', 'generate_ppc_all_subjects',
    'compute_ppc_metrics',
]

__version__ = '1.0.0-nolog'
