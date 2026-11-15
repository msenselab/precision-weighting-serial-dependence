"""Behavioural statistics of the main text applied to data and to simulated responses (2026-10-31).

The functions are the ones the predictive checks used from ppc_manuscript_figures.py and mechanism_knockout.py (now
archived with the models they were written for); the statistics are unchanged.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sdu import paths as SP

DATA = {1: SP.PREPROCESSED / "E1_reprocessed.csv", 2: SP.PREPROCESSED / "E2_reprocessed.csv"}   # data code -> file


def lab(c):
    return np.where(np.isclose(c, 0.3), "L", "H")


def trials(exp):
    """Trial table in the order used by the model fits, with the manuscript's derived columns."""
    d = pd.read_csv(DATA[exp], dtype={"subID": str})
    d["Sub"] = d.subID.astype(int)
    d = d.sort_values(["Sub", "trial_num"]).reset_index(drop=True)
    d["transition"] = pd.Series(lab(d.preCoherence1back), index=d.index).str.cat(lab(d.curCoherence))
    d.loc[d.preCoherence1back.isna(), "transition"] = np.nan
    d["cur_coh"] = np.where(np.isclose(d.curCoherence, 0.3), "Low", "High")
    d["same_switch"] = np.where(d.transition.isin(["LL", "HH"]), "Same", "Switch")
    d.loc[d.transition.isna(), "same_switch"] = np.nan
    return d


def valid_frame(d, rpr):
    """Valid trials with a response vector. Lag/lead columns are the data file's (full sequence, reset at block
    starts), as in the manuscript Figure 4."""
    v = d.loc[~d.is_outlier].copy()
    v["curBias"] = rpr[~d.is_outlier.to_numpy()] - v.curDur
    return v


def slope_ols2(g, x1, x2, y):
    """Coefficient of x2 in y ~ 1 + x1 + x2 (closed form)."""
    a, b, c = g[x1] - g[x1].mean(), g[x2] - g[x2].mean(), g[y] - g[y].mean()
    s11, s22, s12 = (a * a).sum(), (b * b).sum(), (a * b).sum()
    den = s11 * s22 - s12 ** 2
    return ((s11 * (b * c).sum() - s12 * (a * c).sum()) / den) if den > 1e-12 else np.nan


def contrasts(v):
    """Per-participant Low - High and Same - Switch contrasts of b2 (Figure 3)."""
    j = v.dropna(subset=["transition", "preDur1back"])
    b2 = j.groupby(["Sub", "transition"]).filter(lambda g: len(g) >= 5).groupby(["Sub", "transition"]).apply(
        lambda g: slope_ols2(g, "curDur", "preDur1back", "curBias")).unstack()
    low_high = b2[["HL", "LL"]].mean(axis=1) - b2[["LH", "HH"]].mean(axis=1)
    same_switch = b2[["LL", "HH"]].mean(axis=1) - b2[["LH", "HL"]].mean(axis=1)
    out = pd.DataFrame({"low_high": low_high, "same_switch": same_switch})
    for cell in ("LL", "LH", "HL", "HH"):                  # the four transitions (2026-11-03)
        out[f"tr_{cell}"] = b2[cell]
    return out
