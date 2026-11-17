"""Behavioural statistics of the main text, for observed and simulated responses (the one implementation).

Each participant: OLS of the reproduction error on the current and the previous duration (main text, Eq. 2 at the
participant level),
    error = b0 + b1 * current duration + b2 * previous duration,
within groups of current trials. b2 is the controlled previous-duration slope (serial dependence), b1 the
current-duration slope (central tendency).

  groups                                            minimum trials   source in the manuscript
  current coherence Low / High                      10               Section 4.1 (participant-level check)
  Same / Switch (previous = current coherence)      10               Section 4.1 (Experiment 2)
  transition LL, LH, HL, HH (previous, current)     5                Tables A1a-c, Appendix A.1

Contrasts (participant level, then averaged over participants):
  low_high        b2_Low - b2_High                      Experiment 1 core effect (main text .0477)
  low_high_cells  (b2_HL + b2_LL - b2_LH - b2_HH) / 2   equal-cell version shown in Figure 6
  LL_other_three  b2_LL - (b2_LH + b2_HL + b2_HH) / 3  Experiment 2 core effect
  LL_HL           b2_LL - b2_HL                         supporting pairwise description
  same_switch     b2_Same - b2_Switch                   supporting planned contrast (main text .0279)
Additional checks: mean_bias_Low, mean_bias_High and mean_bias_low_high are the conditional response
biases and Low - High shift at current and previous duration 1.2 s, averaged over previous coherence; rsd_Low
and rsd_High are the residual SDs of the Low/High regressions. var_HL_LL is the auxiliary within-duration
residual-SD contrast.

The previous duration is the physical duration of the preceding trial in the same block; the first trial of a block
has none. Valid current trials are those with a response that passed the exclusions (the model input's mask).
"""
import numpy as np
import pandas as pd

from sdu import paths as SP

CORE = {1: ["low_high"], 2: ["LL_other_three"]}             # experiment -> its focal contrast
MIN_GROUP, MIN_CELL = 10, 5
KEYS = ["b1_Low", "b1_High", "b2_Low", "b2_High", "rsd_Low", "rsd_High", "b2_Same", "b2_Switch",
        "b2_LL", "b2_LH", "b2_HL", "b2_HH", "low_high", "low_high_cells", "LL_other_three", "LL_HL",
        "same_switch", "mean_bias_Low", "mean_bias_High", "mean_bias_low_high", "var_HL_LL"]


def previous_duration(T, block_start):
    prev = np.r_[np.nan, np.asarray(T, float)[:-1]]
    prev[np.asarray(block_start, bool)] = np.nan
    return prev


def _ols(x1, x2, y, min_n):
    """(b1, b2, residual SD with ddof 3) of y ~ 1 + x1 + x2, NaN if fewer than min_n trials or a constant predictor."""
    if len(y) < min_n or np.unique(x1).size < 2 or np.unique(x2).size < 2:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), x1, x2])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1], b[2], (y - X @ b).std(ddof=3)


def _conditional_mean_bias(T, prev, err, previous_low, current_low, ok):
    """Participant estimand: conditional bias at T=previous T=1.2 s, averaged over previous coherence."""
    cur = T[ok] - 1.2
    pre = prev[ok] - 1.2
    pl = previous_low[ok] - 0.5
    cl = current_low[ok].astype(float) - 0.5
    X = np.column_stack([
        np.ones(ok.sum()), cur, pl, cur * pl, cl, cur * cl, pl * cl, cur * pl * cl,
        pre, pre * pl, pre * cl, pre * pl * cl,
    ])
    if len(X) < X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        return np.nan, np.nan, np.nan
    beta, *_ = np.linalg.lstsq(X, err[ok], rcond=None)
    high, low = beta[0] - .5 * beta[4], beta[0] + .5 * beta[4]
    return low, high, beta[4]


def participant_stats(T, y, coh, valid, block_start):
    """Statistics of one participant. T durations (s), y reproductions (s), coh coherence, valid mask, block_start."""
    T, y, coh = (np.asarray(a, float) for a in (T, y, coh))
    low = coh < 0.5
    prev = previous_duration(T, block_start)
    prev_low = np.r_[np.nan, low[:-1].astype(float)]
    prev_low[np.asarray(block_start, bool)] = np.nan
    ok = np.asarray(valid, bool) & np.isfinite(y) & np.isfinite(prev)
    err = y - T
    out = {}
    for lab, sel in (("Low", low), ("High", ~low)):
        s = ok & sel
        out[f"b1_{lab}"], out[f"b2_{lab}"], out[f"rsd_{lab}"] = _ols(T[s], prev[s], err[s], MIN_GROUP)
    same = prev_low == low.astype(float)
    for lab, sel in (("Same", same), ("Switch", ~same)):
        s = ok & sel
        out[f"b2_{lab}"] = _ols(T[s], prev[s], err[s], MIN_GROUP)[1]
    resid_sd = {}
    for a in "LH":
        for b in "LH":
            s = ok & (prev_low == float(a == "L")) & (low == (b == "L"))
            out[f"b2_{a}{b}"] = _ols(T[s], prev[s], err[s], MIN_CELL)[1]
            e = pd.Series(err[s])
            resid_sd[a + b] = (e - e.groupby(T[s]).transform("mean")).std(ddof=1) if s.sum() > 1 else np.nan
    out["low_high"] = out["b2_Low"] - out["b2_High"]
    out["low_high_cells"] = np.nanmean([out["b2_HL"], out["b2_LL"]]) - np.nanmean([out["b2_LH"], out["b2_HH"]])
    out["LL_other_three"] = out["b2_LL"] - np.nanmean([out["b2_LH"], out["b2_HL"], out["b2_HH"]])
    out["LL_HL"] = out["b2_LL"] - out["b2_HL"]
    out["same_switch"] = out["b2_Same"] - out["b2_Switch"]
    out["mean_bias_Low"], out["mean_bias_High"], out["mean_bias_low_high"] = _conditional_mean_bias(
        T, prev, err, prev_low, low, ok
    )
    out["var_HL_LL"] = resid_sd["HL"] - resid_sd["LL"]
    return out


def _sequence(exp):
    d = pd.read_csv(SP.MODEL_INPUT)
    d = d[d.exp == exp].sort_values(["Sub", "trial_num"])
    return {s: g for s, g in d.groupby("Sub")}


_SEQ = {}


def sequence(exp, sub):
    """The participant's trial sequence (model input): T, coherence, valid mask, block starts."""
    if exp not in _SEQ:
        _SEQ[exp] = _sequence(exp)
    g = _SEQ[exp][sub]
    return (g.stimulus.to_numpy(float), g.coherence.to_numpy(float), g.response.notna().to_numpy(),
            g.block_start.to_numpy(bool))


def participant_stats_arrays(exp, sub, y):
    """participant_stats for responses y (observed or simulated) on the participant's own sequence."""
    T, coh, valid, block_start = sequence(exp, sub)
    return participant_stats(T, np.where(valid, y, np.nan), coh, valid, block_start)


def observed(exp):
    """One row per participant: the observed statistics."""
    rows = []
    for sub, g in _sequence(exp).items():
        rows.append(dict(Sub=sub, **participant_stats_arrays(exp, sub, g.response.to_numpy(float))))
    return pd.DataFrame(rows)
