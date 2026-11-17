"""Predictive checks of the selected models (Section 5.4, Figure 7, Appendix B.4).

For each experiment's selected model, N_COHORTS cohorts are simulated: every participant's fitted parameters
(results/model/fits/all.csv) on the participant's own sequence and validity mask. Each cohort gives
  * the summary statistics by current coherence (sdu.model.stats: b1 central-tendency slope, b2 serial-dependence
    slope, rsd residual SD), averaged over participants;
  * duration-resolved curves: bias by current duration, residual bias by previous duration, within-duration
    residual SD, each by current coherence;
  * the mean bias by previous duration, by current coherence and by Same/Switch transition.
The model value is the mean over cohorts; the range is the central 95% across cohorts.
Writes results/model/predictive_checks/{summary,duration_curves,history_curves}.csv.

Usage (repo root): python scripts/model/predictive_checks.py [n_cohorts]
"""
import sys

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.behavior import trials as BT
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import stats as S

MODELS = {1: "C3-S0-B2", 2: "C3-S3-B2"}
SEEDS = {1: 20261311, 2: 20261211}                      # random seed of each experiment's cohorts
N_COHORTS = 100
PANELS = ("central tendency", "serial dependence", "response variability")
GROUPINGS = ("cur_coh", "same_switch")                  # history curves: by current coherence, by Same/Switch
SUMMARY = ["b1_Low", "b1_High", "b2_Low", "b2_High", "rsd_Low", "rsd_High"]


def signatures(v):
    """Participant curves by duration: bias by current duration, residual bias by previous duration, and the
    within-duration residual SD, by current coherence."""
    v = v.dropna(subset=["curBias"]).copy()
    v["coh"] = np.where(np.isclose(v.curCoherence, .3), "Low", "High")
    v["residual_bias"] = v.curBias - v.groupby(["Sub", "coh", "curDur"]).curBias.transform("mean")
    central = v.groupby(["Sub", "coh", "curDur"]).curBias.mean()
    serial = v.dropna(subset=["preDur1back"]).groupby(["Sub", "coh", "preDur1back"]).residual_bias.mean()
    variability = v.groupby(["Sub", "coh", "curDur"]).residual_bias.std(ddof=1)
    return central, serial, variability


def history(v):
    """Participant mean bias by previous duration, separately by current coherence and by Same/Switch."""
    v = v.dropna(subset=["curBias", "preDur1back"])
    return {gr: v.groupby(["Sub", gr, "preDur1back"]).curBias.mean() for gr in GROUPINGS}


def group_curve(series, coherence):
    return series.xs(coherence, level="coh").unstack("Sub").mean(axis=1)


def mean_sem(by_participant):
    return (by_participant.mean(axis=1),
            by_participant.std(axis=1, ddof=1) / np.sqrt(by_participant.notna().sum(axis=1)))


def main():
    n_cohorts = int(sys.argv[1]) if len(sys.argv) > 1 else N_COHORTS
    fits_all = FF.read_fits()
    summary, curves, histories = [], [], []
    for exp, label in MODELS.items():
        model = G.Model.parse(label)
        table = BT.trials(exp)
        observed_frame = BT.valid_frame(table, table.rpr.to_numpy(float))
        observed, observed_h = signatures(observed_frame), history(observed_frame)
        data = FF.load(exp)
        subs = sorted(data)
        frames_by_sub = {s: table[table.Sub == s].copy() for s in subs}
        obs_stats = pd.DataFrame([S.participant_stats_arrays(exp, s, np.where(data[s][3], data[s][2], np.nan))
                                  for s in subs])[SUMMARY]
        fits = fits_all[(fits_all.exp == exp) & (fits_all.model == label)].set_index("Sub")
        assert set(fits.index) == set(subs), (exp, label)
        rng = np.random.default_rng(SEEDS[exp])
        cohort_stats, sims, sims_h = [], [], []
        for _ in range(n_cohorts):
            frames, stats = [], []
            for s in subs:
                T, coh, _, valid = data[s]
                y = G.simulate(FF.theta(fits.loc[s]), model, T, coh, valid, rng)[0]
                stats.append(S.participant_stats_arrays(exp, s, y))
                frames.append(BT.valid_frame(frames_by_sub[s], y))
            cohort_stats.append(pd.DataFrame(stats)[SUMMARY].mean())
            cohort = pd.concat(frames, ignore_index=True)
            sims.append(signatures(cohort))
            sims_h.append(history(cohort))
        info = dict(experiment=FF.EXPERIMENT[exp], exp=exp, model=label)
        cs = pd.DataFrame(cohort_stats)
        for st in SUMMARY:
            o = obs_stats[st]
            summary.append(dict(**info, statistic=st, observed=o.mean(), observed_sem=o.std(ddof=1) / np.sqrt(o.notna().sum()),
                                simulated=cs[st].mean(), lo=cs[st].quantile(.025), hi=cs[st].quantile(.975),
                                n_cohorts=n_cohorts))
        for pi, panel in enumerate(PANELS):
            for c in ("Low", "High"):
                om, se = mean_sem(observed[pi].xs(c, level="coh").unstack("Sub"))
                cur = pd.concat([group_curve(k[pi], c) for k in sims], axis=1)
                for dur in om.index:
                    v = cur.loc[dur]
                    curves.append(dict(**info, panel=panel, coherence=c, duration=float(dur), observed=om.loc[dur],
                                       observed_sem=se.loc[dur], simulated=v.mean(), lo=v.quantile(.025),
                                       hi=v.quantile(.975), n_cohorts=n_cohorts))
        for gr in GROUPINGS:
            for lev in observed_h[gr].index.get_level_values(gr).unique():
                om, se = mean_sem(observed_h[gr].xs(lev, level=gr).unstack("Sub"))
                cur = pd.concat([k[gr].xs(lev, level=gr).unstack("Sub").mean(axis=1) for k in sims_h], axis=1)
                for dur in om.index:
                    v = cur.loc[dur]
                    histories.append(dict(**info, grouping=gr, level=lev, previous_duration=float(dur),
                                          observed=om.loc[dur], observed_sem=se.loc[dur], simulated=v.mean(),
                                          lo=v.quantile(.025), hi=v.quantile(.975), n_cohorts=n_cohorts))
        print("done", FF.EXPERIMENT[exp], label, flush=True)
    out = SP.PREDICTIVE_CHECKS
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "duration_curves.csv", index=False)
    pd.DataFrame(histories).to_csv(out / "history_curves.csv", index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
