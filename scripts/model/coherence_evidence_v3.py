"""Matched behavioral experiment × coherence test and existing S0/S3 model recovery.

The four-cell LMM is the project's joint current/previous-duration analysis, with both experiments in one model.
Previous and current Low coherence are effect coded (-0.5 High, +0.5 Low).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm, t, ttest_ind

from sdu import paths as SP
from sdu.behavior import trials as B

sys.path.insert(0, str(SP.ROOT / "scripts/behavior"))
from current_coherence_mean_shift import fit_mixed

OUT = SP.RESULTS / "model/minimal_qdiag/coherence_evidence_v3_20261113"
BASE = "curDur_c * previous_low * current_low + preDur_c * previous_low * current_low"
COMBINED = f"curBias ~ dynamic * ({BASE})"
WITHIN = f"curBias ~ {BASE}"
RANDOM = "~curDur_c + preDur_c"
EXPERIMENT = {2: "Experiment 1", 1: "Experiment 2"}


def observed_frame(exp):
    d = B.trials(exp)
    d = d.loc[~d.is_outlier & d.preDur1back.notna() & d.transition.notna()].copy()
    d["curDur_c"] = d.curDur - 1.2
    d["preDur_c"] = d.preDur1back - 1.2
    d["current_low"] = np.where(np.isclose(d.curCoherence, 0.3), 0.5, -0.5)
    d["previous_low"] = np.where(np.isclose(d.preCoherence1back, 0.3), 0.5, -0.5)
    d["dynamic"] = int(exp == 2)
    d["exp"] = exp
    d["participant"] = str(exp) + "_" + d.Sub.astype(str)
    return d


def sd_vector(names, dynamic=0, current=0, previous=0):
    """Derivative of the fixed-effect mean with respect to previous duration."""
    values = {"preDur_c": 1.0, "dynamic": dynamic, "current_low": current, "previous_low": previous}
    vector = np.zeros(len(names))
    for i, name in enumerate(names):
        parts = name.split(":")
        if "preDur_c" in parts:
            vector[i] = np.prod([values[p] for p in parts])
    return vector


def delta_vector(names, dynamic=0):
    # Averaging over previous coherence cancels terms with previous_low.
    return sd_vector(names, dynamic, 0.5) - sd_vector(names, dynamic, -0.5)


def contrast(result, label, vector):
    names = result.fe_params.index.tolist()
    covariance = result.cov_params().loc[names, names].to_numpy()
    estimate = float(vector @ result.fe_params.to_numpy())
    se = float(np.sqrt(vector @ covariance @ vector))
    return dict(contrast=label, estimate=estimate, se=se, ci_low=estimate - 1.9599639845 * se,
                ci_high=estimate + 1.9599639845 * se, z=estimate / se,
                p=float(2 * norm.sf(abs(estimate / se))))


def projection(frame):
    """OLS Low-minus-High controlled lag slope as one linear map of the response errors."""
    x = dmatrix(BASE, frame, return_type="dataframe")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Four-cell participant design is rank deficient")
    return delta_vector(x.columns.tolist()) @ np.linalg.pinv(x.to_numpy())


def participant_effects(frame):
    return pd.DataFrame([dict(exp=int(g.exp.iloc[0]), Sub=int(g.Sub.iloc[0]),
                             participant=key, effect=float(projection(g) @ g.curBias), n=len(g))
                         for key, g in frame.groupby("participant")])


def behavioral_analysis(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    d = pd.concat([observed_frame(e) for e in (2, 1)], ignore_index=True)
    result, optimizer, messages = fit_mixed(COMBINED, d, RANDOM)
    names = result.fe_params.index.tolist()
    rows = []
    for exp, dynamic in ((2, 1), (1, 0)):
        for name, current in (("High", -0.5), ("Low", 0.5)):
            rows.append(dict(exp=exp, **contrast(result, f"{EXPERIMENT[exp]}: {name} lag slope",
                                                sd_vector(names, dynamic, current))))
        rows.append(dict(exp=exp, **contrast(result, f"{EXPERIMENT[exp]}: Low - High",
                                            delta_vector(names, dynamic))))
    rows.append(dict(exp=0, **contrast(result, "Experiment 1 - Experiment 2: coherence interaction",
                                      delta_vector(names, 1) - delta_vector(names, 0))))
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(out / "behavioral_lmm_contrasts.csv", index=False)
    pd.DataFrame([dict(formula=COMBINED, random_formula=RANDOM, reml=True, optimizer=optimizer,
                       converged=result.converged, trials=len(d), participants=d.participant.nunique(),
                       log_likelihood=result.llf, warnings=messages)]).to_csv(out / "behavioral_fit_record.csv", index=False)
    pd.DataFrame({"term": names, "estimate": result.fe_params, "se": result.bse_fe}).to_csv(
        out / "behavioral_fixed_effects.csv", index=False)
    participant = participant_effects(d)
    participant.to_csv(out / "behavioral_participant_effects.csv", index=False)
    a = participant[participant.exp == 2].effect.to_numpy()
    b = participant[participant.exp == 1].effect.to_numpy()
    test = ttest_ind(a, b, equal_var=False)
    variance = a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)
    df = variance ** 2 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) +
                          (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    rng = np.random.default_rng(2026111301)
    boot = rng.choice(a, (10000, len(a))).mean(1) - rng.choice(b, (10000, len(b))).mean(1)
    pd.DataFrame([dict(contrast="Experiment 1 - Experiment 2: participant coherence effect",
                       estimate=a.mean() - b.mean(), se=np.sqrt(variance), df=df, t=test.statistic, p=test.pvalue,
                       ci_low=a.mean() - b.mean() - t.ppf(0.975, df) * np.sqrt(variance),
                       ci_high=a.mean() - b.mean() + t.ppf(0.975, df) * np.sqrt(variance),
                       bootstrap_low=np.quantile(boot, 0.025), bootstrap_high=np.quantile(boot, 0.975))]).to_csv(
                           out / "behavioral_participant_interaction.csv", index=False)
    print(contrasts.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    return contrasts


def model_recovery(out=OUT):
    source = SP.RESULTS / "model/minimal_qdiag/model_recovery_v3/r100_s12_n10"
    raw = pd.concat([pd.read_csv(p) for p in sorted(source.glob("switch_exp*_*.csv"))], ignore_index=True)
    keys = ["generator", "rep", "candidate"]
    counts = raw.groupby(keys).size()
    if len(raw) != 8800 or raw.duplicated(keys + ["Sub"]).any() or not counts.eq(22).all():
        raise ValueError("Incomplete or duplicated S0/S3 cross-fits")
    if not raw.k.eq(10).all():
        raise ValueError("S0/S3 parameter count has changed")
    aic = raw.groupby(keys).AIC.mean().unstack("candidate").reset_index()
    aic["delta_AIC_S3_minus_S0"] = aic.S3 - aic.S0
    aic["winner"] = np.where(aic.S0 < aic.S3, "S0", "S3")
    matrix = pd.crosstab(aic.generator, aic.winner).reindex(index=["S0", "S3"], columns=["S0", "S3"], fill_value=0)
    matrix.to_csv(out / "switch_confusion_counts.csv")
    aic.to_csv(out / "switch_cohorts.csv", index=False)
    pd.DataFrame([dict(source=str(source.relative_to(SP.ROOT)), generator=g, cohorts=len(d),
                       correct=(d.winner == g).mean(), median_delta_aic=d.delta_AIC_S3_minus_S0.median())
                  for g, d in aic.groupby("generator")]).to_csv(out / "switch_summary.csv", index=False)
    print(matrix.to_string(), flush=True)


if __name__ == "__main__":
    behavioral_analysis()
    model_recovery()
