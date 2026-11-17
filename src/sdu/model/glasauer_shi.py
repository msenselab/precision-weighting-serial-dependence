"""The duration-reproduction data of Glasauer & Shi (2021) and their two-state observer (manuscript Section 5.5,
Appendix C).

Data: data/external/glasauer_shi_2021; one 400-trial sequence per participant and sequence type (random walk,
randomized). load() returns them as {"<participant>|<sequence>": (durations, responses, valid)}.

Two-state observer (Glasauer & Shi, 2022): Eq. 4 and the Kalman equations of the paper. The fitted parameters are
the variance ratios v/r and q/r and the response shift Delta x. Durations are encoded as log(1 + d/d0); the supplied
reference implementation decodes the estimate as d0 * (exp(x_hat + Delta x) - 1).

Three-state observer on these data (sdu.model.engine): there is no coherence manipulation (coherence = 1 on every
trial), so the C and S dimensions have no effect and the model reduces to one retention phi (phi_H) and the bias
level (THREE_STATE; Appendix C compares B2 with B0), FIXED_THREE_STATE the parameters without effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from sdu import paths as SP
from sdu.model import engine as G

DATA = SP.EXTERNAL / "glasauer_shi_2021" / "Glasauer_Shi_2021b_duration_reproduction.csv"
SEQUENCES = {1: "random walk", 2: "randomized"}                  # RandLevel
THREE_STATE = {f"B{b}": G.Model(0, 0, b) for b in G.B_LEVELS}
TRANSFER = ("B0", "B2")                                          # the models of the transfer test
FIXED_THREE_STATE = {"phi_L": 0.0, "d0_LH": 0.0}                 # no low-coherence trials


def load():
    """{"<participant>|<sequence>": (durations, responses, valid)} in trial order."""
    d = pd.read_csv(DATA).dropna(subset=["RandLevel"])
    # The column labels of the CSV are swapped: "condition" (1-14) is the participant and "NSub" (1-20) the
    # 40-trial block; NT (1-800) is the trial order within a participant.
    units = {}
    for (sub, level), g in d.groupby(["condition", "RandLevel"]):
        g = g.sort_values("NT")
        y = g.repDur.to_numpy(float)
        units[f"{int(sub)}|{SEQUENCES[int(level)]}"] = (g.dur.to_numpy(float), y,
                                                         (g.valid.to_numpy() == 1) & np.isfinite(y))
    return units


def participants(units):
    return sorted({int(key.split("|")[0]) for key in units})


def three_state_bounds(model):
    """Bounds of a three-state model on these data, with the parameters without effect fixed."""
    lo, hi = model.bounds()
    for name, value in FIXED_THREE_STATE.items():
        lo[G.IX[name]] = hi[G.IX[name]] = value
    return lo, hi


def three_state_arrays(duration, response, valid):
    """Engine arrays (durations, coherence = 1, responses, valid)."""
    return duration, np.ones(len(duration)), np.where(valid, response, 0.0), valid

D0 = 0.1
F = np.array([[0.0, 1.0], [0.0, 1.0]])
H = np.array([1.0, 0.0])


@dataclass(frozen=True)
class Fit:
    model: str
    v_over_r: float
    q_over_r: float
    shift: float
    sigma: float
    log_likelihood: float
    aic: float
    n: int
    success: bool


def encode(duration: np.ndarray, d0: float = D0) -> np.ndarray:
    """Paper's Weber-Fechner transform z = log(1 + d/d0)."""
    duration = np.asarray(duration, float)
    if np.any(duration <= 0):
        raise ValueError("durations must be positive")
    return np.log1p(duration / d0)


def decode(state: np.ndarray, shift: float, d0: float = D0) -> np.ndarray:
    """Reference response equation ``d_hat = d0 * (exp(x_hat + shift) - 1)``."""
    return d0 * np.expm1(np.asarray(state, float) + shift)


def observer(
    duration: np.ndarray,
    v_over_r: float,
    q_over_r: float,
    shift: float,
    *,
    d0: float = D0,
    return_state: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the full two-state Kalman observer on one stimulus sequence.

    The scale of the covariance is arbitrary, so r is fixed to one and Q contains
    the fitted ratios v/r and q/r.  Initialization follows the supplied translation:
    both states start at the first encoded duration and P starts as the identity.
    """
    if v_over_r < 0 or q_over_r < 0:
        raise ValueError("variance ratios must be non-negative")
    z = encode(duration, d0)
    state = np.array([z[0], z[0]], float)
    covariance = np.eye(2)
    process_noise = np.diag([v_over_r, q_over_r])
    states = np.empty((len(z), 2))
    variances = np.empty((len(z), 2))

    for trial, measurement in enumerate(z):
        predicted_state = F @ state
        predicted_covariance = F @ covariance @ F.T + process_noise
        gain = predicted_covariance @ H / (H @ predicted_covariance @ H + 1.0)
        innovation = measurement - H @ predicted_state
        state = predicted_state + gain * innovation
        covariance = (np.eye(2) - np.outer(gain, H)) @ predicted_covariance
        states[trial] = state
        variances[trial] = np.diag(covariance)

    response = decode(states[:, 0], shift, d0)
    if return_state:
        return response, states, variances
    return response


def statistics(duration: np.ndarray, response: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    """Paper-aligned participant statistics for within- and cross-condition prediction."""
    duration = np.asarray(duration, float)
    response = np.asarray(response, float)
    ok = np.asarray(valid, bool) & np.isfinite(response)
    error = response - duration
    previous = np.r_[np.nan, duration[:-1]]
    previous2 = np.r_[np.nan, np.nan, duration[:-2]]

    central_slope = np.polyfit(duration[ok], error[ok], 1)[0]
    use1 = ok & np.isfinite(previous)
    lag1 = np.polyfit(previous[use1], error[use1], 1)[0]

    def partial_correlation(x, y, control):
        design = np.column_stack([np.ones(len(control)), control])
        x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return float(np.corrcoef(x_residual, y_residual)[0, 1])

    use2 = ok & np.isfinite(previous2)
    lag2 = np.polyfit(previous2[use2], error[use2], 1)[0]

    return {
        "ct": central_slope,
        "sd": lag1,
        "partial_previous_r": partial_correlation(
            error[use1], previous[use1], duration[use1]
        ),
        "partial_current_r": partial_correlation(
            error[use1], duration[use1], previous[use1]
        ),
        "current_previous_r": np.corrcoef(duration[use1], previous[use1])[0, 1],
        "sd2": lag2,
        "bias": float(np.nanmean(error[ok])),
    }


def _unpack(model: str, free: np.ndarray) -> tuple[float, float, float]:
    if model == "two-state":
        return float(free[0]), float(free[1]), float(free[2])
    if model == "static":
        return float(free[0]), 0.0, float(free[1])
    if model == "iterative":
        return 0.0, float(free[0]), float(free[1])
    raise ValueError(f"unknown model: {model}")


def fit(
    duration: np.ndarray,
    response: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    model: str = "two-state",
    starts: int = 12,
    seed: int = 0,
) -> Fit:
    """Least-squares fit with Gaussian residual likelihood and multistart optimization."""
    duration = np.asarray(duration, float)
    response = np.asarray(response, float)
    if valid is None:
        valid = np.isfinite(response)
    valid = np.asarray(valid, bool) & np.isfinite(response)
    if duration.shape != response.shape or valid.shape != duration.shape:
        raise ValueError("duration, response, and valid must have the same shape")
    if valid.sum() < 4:
        raise ValueError("at least four valid trials are required")

    n_variances = 2 if model == "two-state" else 1
    lower = np.r_[np.zeros(n_variances), -2.0]
    upper = np.r_[np.full(n_variances, 100.0), 2.0]

    def residual(free):
        v_over_r, q_over_r, shift = _unpack(model, free)
        predicted = observer(duration, v_over_r, q_over_r, shift)
        return response[valid] - predicted[valid]

    rng = np.random.default_rng(seed)
    initial = [np.r_[np.ones(n_variances), 0.0]]
    for _ in range(starts - 1):
        ratios = np.exp(rng.uniform(np.log(1e-3), np.log(20.0), n_variances))
        initial.append(np.r_[ratios, rng.uniform(-0.4, 0.4)])
    results = [
        least_squares(residual, start, bounds=(lower, upper), method="trf") for start in initial
    ]
    best = min(results, key=lambda result: np.dot(result.fun, result.fun))

    v_over_r, q_over_r, shift = _unpack(model, best.x)
    n = len(best.fun)
    rss = float(np.dot(best.fun, best.fun))
    sigma = np.sqrt(rss / n)
    log_likelihood = -0.5 * n * (np.log(2 * np.pi * sigma**2) + 1.0)
    k = len(best.x) + 1  # fitted model parameters plus residual sigma
    return Fit(
        model=model,
        v_over_r=v_over_r,
        q_over_r=q_over_r,
        shift=shift,
        sigma=sigma,
        log_likelihood=log_likelihood,
        aic=2 * k - 2 * log_likelihood,
        n=n,
        success=bool(best.success),
    )
