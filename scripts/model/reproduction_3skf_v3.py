"""Fixed-parameter predictive check for 3SKF v3 (readout phi), in the format of the 3SKF v1 reproduction (2026-11-11).

For each selected model, N_COHORTS cohorts are simulated: every participant's fitted parameters
(results/model/minimal_qdiag/readout_space) on the participant's own sequence and validity mask. Each cohort gives
  * the summary statistics by current coherence (sdu.model.stats: b1 central-tendency slope, b2 serial-dependence
    slope, rsd residual SD), averaged over participants;
  * the duration-resolved curves of the v1 figure (sdu.model.reproduction._signatures).
The model value is the mean over cohorts; the range is the central 95% across cohorts.
  * the mean bias by previous duration, by current coherence and by same/switch (the layout of the earlier Figure 7).
Writes results/model/minimal_qdiag/readout_space_ppc/{summary,duration_curves,history_curves}.csv.

Usage (repo root): GKF_SCALE=lin GKF_CORE=reduced GKF_BOUNDS=v3 GKF_INIT=gs GKF_NOISELESS=1 GKF_QDIAG=1 \\
                   GKF_READOUT=1 python scripts/model/reproduction_3skf_v3.py [n_cohorts]
"""
import os
import sys

import numpy as np
import pandas as pd

from sdu import names
from sdu import paths as SP
from sdu.behavior import trials as BT
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import stats as S
from sdu.model.reproduction import PANELS, _group_curve, _signatures

sys.path.insert(0, os.path.dirname(__file__))
import fit_minimal_space as M  # noqa: E402

MODELS = {2: ["C3-R-S0-B2"], 1: ["C3-R-S3-B2"]}      # data code 2 = Experiment 1, 1 = Experiment 2
GROUPINGS = ("cur_coh", "same_switch")                  # history curves: by current coherence, by same/switch


def _history(v):
    """Participant mean bias by previous duration, separately by current coherence and by same/switch."""
    v = v.dropna(subset=["curBias", "preDur1back"])
    return {gr: v.groupby(["Sub", gr, "preDur1back"]).curBias.mean() for gr in GROUPINGS}
SUMMARY = ["b1_Low", "b1_High", "b2_Low", "b2_High", "rsd_Low", "rsd_High"]
ROOT = SP.RESULTS / "model" / "minimal_qdiag"
OUT = ROOT / "readout_space_ppc"


def main():
    assert G.READOUT and G.LIN and G.NOISELESS and G.QDIAG
    n_cohorts = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    space = dict(M.space())
    fits_all = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "readout_space").glob("*.csv"))], ignore_index=True)
    summary, curves, history = [], [], []
    for exp, labels in MODELS.items():
        table = BT.trials(exp)
        observed = _signatures(BT.valid_frame(table, table.rpr.to_numpy(float)))
        observed_h = _history(BT.valid_frame(table, table.rpr.to_numpy(float)))
        data = FF.load(exp)
        subs = sorted(data)
        frames_by_sub = {s: table[table.Sub == s].copy() for s in subs}
        obs_stats = pd.DataFrame([S.participant_stats_arrays(exp, s, np.where(data[s][4], data[s][3], np.nan))
                                  for s in subs])[SUMMARY]
        for mi, label in enumerate(labels):
            fits = fits_all[(fits_all.exp == exp) & (fits_all.model == label)].set_index("Sub")
            assert set(fits.index) == set(subs), (exp, label)
            rng = np.random.default_rng(20261111 + 100 * exp + mi)
            cohort_stats, sims, sims_h = [], [], []
            for _ in range(n_cohorts):
                frames, stats = [], []
                for s in subs:
                    logT, coh, tau, _, valid = data[s]
                    y = G.simulate(fits.loc[s, G.NAMES].to_numpy(float), space[label], np.exp(logT), coh, tau, valid,
                                   rng)[0]
                    stats.append(S.participant_stats_arrays(exp, s, y))
                    frames.append(BT.valid_frame(frames_by_sub[s], y))
                cohort_stats.append(pd.DataFrame(stats)[SUMMARY].mean())
                cohort = pd.concat(frames, ignore_index=True)
                sims.append(_signatures(cohort))
                sims_h.append(_history(cohort))
            cs = pd.DataFrame(cohort_stats)
            for st in SUMMARY:
                o = obs_stats[st]
                summary.append(dict(experiment=names.EXPERIMENT[exp], exp=exp, model=label.replace("-R-", "-"),
                                    statistic=st, observed=o.mean(), observed_sem=o.std(ddof=1) / np.sqrt(o.notna().sum()),
                                    simulated=cs[st].mean(), lo=cs[st].quantile(.025), hi=cs[st].quantile(.975),
                                    n_cohorts=n_cohorts))
            for pi, panel in enumerate(PANELS):
                for c in ("Low", "High"):
                    ob = observed[pi].xs(c, level="coh").unstack("Sub")
                    om, se = ob.mean(axis=1), ob.std(axis=1, ddof=1) / np.sqrt(ob.notna().sum(axis=1))
                    cur = pd.concat([_group_curve(k[pi], c) for k in sims], axis=1)
                    for dur in om.index:
                        v = cur.loc[dur]
                        curves.append(dict(experiment=names.EXPERIMENT[exp], exp=exp, model=label.replace("-R-", "-"),
                                           panel=panel, coherence=c, duration=float(dur), observed=om.loc[dur],
                                           observed_sem=se.loc[dur], simulated=v.mean(), lo=v.quantile(.025),
                                           hi=v.quantile(.975), n_cohorts=n_cohorts))
            for gr in GROUPINGS:
                for lev in observed_h[gr].index.get_level_values(gr).unique():
                    ob = observed_h[gr].xs(lev, level=gr).unstack("Sub")
                    om, se = ob.mean(axis=1), ob.std(axis=1, ddof=1) / np.sqrt(ob.notna().sum(axis=1))
                    cur = pd.concat([k[gr].xs(lev, level=gr).unstack("Sub").mean(axis=1) for k in sims_h], axis=1)
                    for dur in om.index:
                        v = cur.loc[dur]
                        history.append(dict(experiment=names.EXPERIMENT[exp], exp=exp, model=label.replace("-R-", "-"),
                                            grouping=gr, level=lev, previous_duration=float(dur), observed=om.loc[dur],
                                            observed_sem=se.loc[dur], simulated=v.mean(), lo=v.quantile(.025),
                                            hi=v.quantile(.975), n_cohorts=n_cohorts))
            print("done", exp, label, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "duration_curves.csv", index=False)
    pd.DataFrame(history).to_csv(OUT / "history_curves.csv", index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
