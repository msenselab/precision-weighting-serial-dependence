import numpy as np

from sdu.model import glasauer_shi as gs


def test_paper_transform_and_response_equation():
    duration = np.array([0.4, 1.0, 1.9])
    encoded = gs.encode(duration)
    assert np.allclose(encoded, np.log1p(duration / gs.D0))
    shift = -0.07
    assert np.allclose(gs.decode(encoded, shift), gs.D0 * np.expm1(encoded + shift))
    assert np.allclose(gs.decode(encoded, 0.0), duration)


def test_observer_matches_literal_paper_equations():
    duration = np.array([0.5, 1.3, 0.8, 1.7])
    v_over_r, q_over_r, shift = 0.7, 0.2, -0.1
    z = gs.encode(duration)
    state = np.array([z[0], z[0]])
    covariance = np.eye(2)
    states = []
    for measurement in z:
        state_minus = gs.F @ state
        covariance_minus = gs.F @ covariance @ gs.F.T + np.diag([v_over_r, q_over_r])
        gain = covariance_minus @ gs.H / (gs.H @ covariance_minus @ gs.H + 1.0)
        state = state_minus + gain * (measurement - gs.H @ state_minus)
        covariance = (np.eye(2) - np.outer(gain, gs.H)) @ covariance_minus
        states.append(state.copy())
    expected = gs.decode(np.asarray(states)[:, 0], shift)
    assert np.allclose(gs.observer(duration, v_over_r, q_over_r, shift), expected)


def test_static_and_iterative_are_full_model_boundaries():
    duration = np.array([0.4, 0.9, 1.5, 1.0, 1.8])
    shift = -0.1
    static = gs.observer(duration, 0.8, 0.0, shift)
    iterative = gs.observer(duration, 0.0, 0.3, shift)
    assert np.allclose(static, gs.observer(duration, 0.8, 0, shift))
    assert np.allclose(iterative, gs.observer(duration, 0, 0.3, shift))
    assert not np.allclose(static, iterative)


def test_fit_counts_only_valid_residuals_and_sigma_in_aic():
    duration = np.linspace(0.4, 1.9, 30)
    response = gs.observer(duration, 0.5, 0.1, -0.08)
    response += np.random.default_rng(4).normal(0, 0.02, len(response))
    response[[3, 11]] = np.nan
    result = gs.fit(duration, response, model="two-state", starts=3, seed=2)
    assert result.n == 28
    assert np.isfinite(result.aic)
    assert result.sigma > 0
    assert result.success


def test_absolute_serial_dependence_is_error_on_previous_stimulus():
    duration = np.array([0.5, 1.0, 1.5, 0.7, 1.2, 1.8])
    previous = np.r_[np.nan, duration[:-1]]
    response = duration + 0.2 * np.nan_to_num(previous, nan=duration[0])
    result = gs.statistics(duration, response, np.ones(len(duration), bool))
    expected = np.polyfit(previous[1:], (response - duration)[1:], 1)[0]
    assert np.isclose(result["sd"], expected)


def test_partial_correlations_control_the_other_stimulus():
    duration = np.array([0.5, 0.8, 1.1, 1.4, 1.0, 1.7, 1.3, 0.7])
    previous = np.r_[np.nan, duration[:-1]]
    error = -0.3 * duration + 0.2 * np.nan_to_num(previous, nan=duration[0]) + 0.1
    response = duration + error
    result = gs.statistics(duration, response, np.ones(len(duration), bool))
    assert np.isclose(result["partial_previous_r"], 1.0)
    assert np.isclose(result["partial_current_r"], -1.0)
