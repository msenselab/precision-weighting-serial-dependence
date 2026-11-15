#!/usr/bin/env python3
"""Is the current-coherence effect on history a response-repetition effect? (2026-11-02)

Extends response_history_reanalysis.py (same block-safe lags and masking) with the current coherence. If participants
partly repeated their previous reproduction under low coherence, the previous response error would predict the
current bias more strongly under low coherence (response error lag 1 x current coherence), beyond the previous
duration. The stimulus-history effect of the manuscript is previous duration x current coherence.

    curBias ~ (curDur_c + preDur_c + response_error_lag1) * low + response_error_lag2..5,  random intercept
    low = +0.5 (current Low) / -0.5 (current High)

Participant-level OLS of the same model gives a sensitivity check (one-sample t tests of the interaction slopes).
Outputs: analysis/response_history/coherence_interaction_lmm.csv, coherence_interaction_participants.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import response_history_reanalysis as RH

FORMULA = ("curBias ~ (curDur_c + preDur_c + response_error_lag1) * low + response_error_lag2 + response_error_lag3"
           " + response_error_lag4 + response_error_lag5")
KEY = ["preDur_c", "response_error_lag1", "preDur_c:low", "response_error_lag1:low", "low"]


def main():
    rows, prow = [], []
    for exp, path in RH.DATA.items():
        d = RH.prepare(path)
        d["low"] = np.where(np.isclose(d["curCoherence"], 0.3), 0.5, -0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = smf.mixedlm(FORMULA, d, groups=d["subID"], re_formula="1").fit(reml=True, method="lbfgs",
                                                                               maxiter=3000)
        for term in KEY:
            rows.append(dict(experiment=exp, term=term, b=res.params[term], se=res.bse[term], z=res.tvalues[term],
                             p=res.pvalues[term], n=len(d), converged=res.converged))
        for sub, g in d.groupby("subID"):
            fit = smf.ols(FORMULA, g).fit()
            prow.append(dict(experiment=exp, subID=sub, **{t: fit.params[t] for t in KEY}))
    lmm = pd.DataFrame(rows)
    part = pd.DataFrame(prow)
    lmm.to_csv(RH.OUT / "coherence_interaction_lmm.csv", index=False)
    part.to_csv(RH.OUT / "coherence_interaction_participants.csv", index=False)
    print(lmm.round(4).to_string(index=False))
    for exp, g in part.groupby("experiment", sort=False):
        for term in ("preDur_c:low", "response_error_lag1:low"):
            t = stats.ttest_1samp(g[term], 0)
            print(f"{exp} participant {term}: mean {g[term].mean():+.4f}, t({len(g) - 1}) = {t.statistic:.2f}, "
                  f"p = {t.pvalue:.3f}, positive {int((g[term] > 0).sum())}/{len(g)}")


if __name__ == "__main__":
    main()
