"""
Core 3-state Kalman filter engine for 135-model framework.
NO LOG TRANSFORM VERSION - operates directly in original duration space.

State vector: x = [μ (fast), m (slow), b (bias)]^T

Key equations:
--------------
State Transition (B-axis controls F[0,2]):
    B1/B3: F = [[0, 1, 0], [0, 1, 0], [0, 0, λ]]  → μ⁻ = m
    B2:    F = [[0, 1, 1], [0, 1, 0], [0, 0, λ]]  → μ⁻ = m + b

Observation:
    H = [1, 0, 0]  → z = μ + noise

Coherence Modulation:
    Q_i(c) = q_i_base × exp(α_qi × (1 - c))  for i ∈ {1, 2, 3}
    R(c) = 1 if c ≥ 0.5, r_low if c < 0.5

Response:
    B1/B2: ŷ = μ⁺ + d0_eff  where d0_eff = d0 + α_d0×(1-coh)
    B3:    ŷ = μ⁺ + b⁺ + d0_eff
"""
import numpy as np
from .config import C_AXIS, S_AXIS, B_AXIS
from .parameters import parse_parameters


def run_3state_kf(par, stimrep, coherence, structure, c_id, s_id, b_id,
                  residual_space='orig', return_tracking=False):
    """
    Run 3-state Kalman filter with specified (C, S, B) configuration.
    NO LOG TRANSFORM - operates in original duration space.

    Parameters
    ----------
    par : array-like
        Parameter vector (order determined by parameters.py)
    stimrep : array (N, 2)
        [:, 0] = stimulus durations, [:, 1] = reproduced durations
    coherence : array (N,)
        Coherence values per trial (e.g., 0.3 or 0.7)
    structure : array (N,) or None
        'Switch' or 'Repeat' labels
    c_id : str
        C-axis ID ('C1' to 'C15')
    s_id : str
        S-axis ID ('S0', 'S1', 'S2')
    b_id : str
        B-axis ID ('B1', 'B2', 'B3')
    residual_space : str
        Ignored in this version (always original space)
    return_tracking : bool
        If True, return state trajectories for analysis

    Returns
    -------
    residuals : array
        Prediction errors (observed - predicted) in original space
    tracking : dict (only if return_tracking=True)
        State trajectories for PPC analysis
    """
    # Get axis configurations
    c_cfg = C_AXIS[c_id]
    s_cfg = S_AXIS[s_id]
    b_cfg = B_AXIS[b_id]

    # Parse parameters
    params = parse_parameters(par, c_id, s_id, b_id)
    q1_base = params['q1']
    q2_base = params['q2']
    q3_base = params['q3']
    lam = params['lambda']
    d0 = params['d0']
    alpha_d0 = params['alpha_d0']
    alpha_q1 = params['alpha_q1']
    alpha_q2 = params['alpha_q2']
    alpha_q3 = params['alpha_q3']
    r_base = params['r_base']
    r_low = params['r_low']
    x_reset = params['x_reset']
    k_reset = params['k_reset']

    # Validate parameters
    if not (0 <= lam <= 1) or q1_base < 0 or q2_base < 0 or q3_base < 0 or r_base < 0:
        return np.full(len(stimrep), 1e10)

    # NO LOG TRANSFORM - use raw durations
    stim = stimrep[:, 0]  # Stimulus durations (0.8 - 1.6)
    obs = stimrep[:, 1]   # Reproduced durations
    prior_mean = np.mean(stim)  # ~1.2
    N = len(stim)

    # =========================================================================
    # STATE-SPACE MATRICES
    # =========================================================================
    F_02 = b_cfg['F_02']  # 0.0 for B1/B3, 1.0 for B2
    response_bias = b_cfg['response_bias']  # True for B3

    F = np.array([
        [0.0, 1.0, F_02],   # μ⁻ = m + F_02 * b
        [0.0, 1.0, 0.0],    # m⁻ = m
        [0.0, 0.0, lam]     # b⁻ = λ * b
    ])

    # Observation matrix
    H = np.array([[1.0, 0.0, 0.0]])

    # Initialize state at first stimulus value
    x = np.array([[stim[0]], [stim[0]], [0.0]])
    P = np.eye(3)
    I3 = np.eye(3)

    # Output arrays
    pred = np.zeros(N)

    if return_tracking:
        tracking = {
            'mu_prior': np.zeros(N),
            'mu_post': np.zeros(N),
            'm_post': np.zeros(N),
            'b_prior': np.zeros(N),
            'b_post': np.zeros(N),
            'K_mu': np.zeros(N),
            'K_m': np.zeros(N),
            'K_b': np.zeros(N),
            'innovation': np.zeros(N),
            'q1_eff': np.zeros(N),
            'q2_eff': np.zeros(N),
            'q3_eff': np.zeros(N),
            'R_eff': np.zeros(N),
        }

    # Pre-allocate Q and R matrices
    Q = np.zeros((3, 3))
    R = np.zeros((1, 1))

    # Pre-check which modulations are active
    mod_q1 = c_cfg['Q1']
    mod_q2 = c_cfg['Q2']
    mod_q3 = c_cfg['Q3']
    mod_r = c_cfg['R']

    for t in range(N):
        coh = coherence[t]
        z = stim[t]  # Current stimulus (original scale, e.g., 1.2)

        # =====================================================================
        # C-AXIS: Coherence modulation of noise parameters
        # =====================================================================
        q1_eff = q1_base * np.exp(alpha_q1 * (1.0 - coh)) if mod_q1 else q1_base
        q2_eff = q2_base * np.exp(alpha_q2 * (1.0 - coh)) if mod_q2 else q2_base
        q3_eff = q3_base * np.exp(alpha_q3 * (1.0 - coh)) if mod_q3 else q3_base

        if mod_r:
            R_eff = 1.0 if coh >= 0.5 else r_low
        else:
            R_eff = r_base

        # Fill pre-allocated matrices (no new allocation)
        Q[0, 0] = q1_eff
        Q[1, 1] = q2_eff
        Q[2, 2] = q3_eff
        R[0, 0] = R_eff

        # =====================================================================
        # S-AXIS: Switch detection
        # =====================================================================
        is_switch = (structure is not None and
                     len(structure) > t and
                     structure[t] == 'Switch')

        # =====================================================================
        # PREDICT STEP
        # =====================================================================
        x_prior = F @ x
        P_prior = F @ P @ F.T + Q

        # S1: x_reset - reset slow state toward prior on switch
        if s_cfg['target'] == 'state' and is_switch and x_reset > 0:
            x_prior[1, 0] = (1.0 - x_reset) * x_prior[1, 0] + x_reset * prior_mean

        # =====================================================================
        # UPDATE STEP
        # =====================================================================
        S = H @ P_prior @ H.T + R
        S_val = float(S[0, 0])

        if S_val <= 0 or not np.isfinite(S_val):
            return np.full(N, 1e10)

        K = (P_prior @ H.T) / S_val

        # S2: K_reset - reduce gain on switch trials
        if s_cfg['target'] == 'gain' and is_switch and k_reset > 0:
            K = K * (1.0 - k_reset)

        y_hat_prior = float((H @ x_prior)[0, 0])
        innovation = z - y_hat_prior

        x_post = x_prior + K * innovation
        P = (I3 - K @ H) @ P_prior

        # =====================================================================
        # RESPONSE GENERATION (no transform needed)
        # =====================================================================
        mu_post = float(x_post[0, 0])
        b_post = float(x_post[2, 0])

        # Coherence-dependent d0
        d0_eff = d0 + alpha_d0 * (1 - coherence[t])

        # Response prediction (original scale)
        if response_bias:  # B3: y = μ + b + d0
            y_pred = mu_post + b_post + d0_eff
        else:  # B1/B2: y = μ + d0
            y_pred = mu_post + d0_eff
        
        pred[t] = y_pred

        # =====================================================================
        # STORE TRACKING DATA
        # =====================================================================
        if return_tracking:
            tracking['mu_prior'][t] = float(x_prior[0, 0])
            tracking['mu_post'][t] = mu_post
            tracking['m_post'][t] = float(x_post[1, 0])
            tracking['b_prior'][t] = float(x_prior[2, 0])
            tracking['b_post'][t] = float(x_post[2, 0])
            tracking['K_mu'][t] = float(K[0, 0])
            tracking['K_m'][t] = float(K[1, 0])
            tracking['K_b'][t] = float(K[2, 0])
            tracking['innovation'][t] = innovation
            tracking['q1_eff'][t] = q1_eff
            tracking['q2_eff'][t] = q2_eff
            tracking['q3_eff'][t] = q3_eff
            tracking['R_eff'][t] = R_eff

        x = x_post

        # Sanity check
        if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1e6):
            return np.full(N, 1e10)

    # =========================================================================
    # COMPUTE RESIDUALS (always in original space)
    # =========================================================================
    resid = obs - pred

    if return_tracking:
        tracking['pred_orig'] = pred
        tracking['obs_orig'] = obs
        tracking['stim_orig'] = stim
        return resid, tracking

    resid = resid[np.isfinite(resid)]
    return resid if resid.size > 0 else np.full(N, 1e10)


def generate_predictions(par, stimrep, coherence, structure, c_id, s_id, b_id):
    """
    Generate model predictions for PPC.
    """
    _, tracking = run_3state_kf(
        par, stimrep, coherence, structure,
        c_id, s_id, b_id,
        residual_space='orig',
        return_tracking=True
    )
    return tracking


def simulate_responses(par, stimrep, coherence, structure, c_id, s_id, b_id,
                       add_noise=True, noise_scale=0.1):
    """
    Simulate responses from the model (for model recovery).
    Noise scale adjusted for original duration space (~0.05s SD).
    """
    N = len(stimrep)
    stimrep_sim = stimrep.copy()
    stimrep_sim[:, 1] = stimrep[:, 0]

    tracking = generate_predictions(par, stimrep_sim, coherence, structure,
                                   c_id, s_id, b_id)

    simulated = tracking['pred_orig'].copy()

    if add_noise:
        simulated += np.random.normal(0, noise_scale, N)

    return simulated
