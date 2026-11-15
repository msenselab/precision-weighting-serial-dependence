"""Revision diagnostic: duration-conditional within-person response dispersion.

Response-dispersion analysis (2026-09-25). This is response
precision, not an isolated measure of sensory precision. Inputs remain read-only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sdu import paths as SP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN = SP.PREPROCESSED
OUT = SP.RESULTS_BEHAVIOR / "analysis" / "dispersion"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [("Exp1_dynamic", "E2_reprocessed.pkl"), ("Exp2_fixed", "E1_reprocessed.pkl")]
records = []
participants = []
summary = []
for exp, filename in SOURCES:
    frame = pd.read_pickle(IN / filename)
    assert frame['subID'].nunique() == 22, (exp, frame['subID'].nunique())
    x = frame.loc[(~frame['is_outlier']) & frame['rpr'].notna(),
                  ['subID', 'curDur', 'curCoherence', 'rpr', 'curBias']].copy()
    assert set(x.curCoherence.unique()) == {0.3, 0.7}
    assert set(x.curDur.unique()) == {0.8, 1., 1.2, 1.4, 1.6}
    cells = (x.groupby(['subID', 'curDur', 'curCoherence'], observed=True)
               .agg(n=('rpr', 'size'), mean_rpr=('rpr', 'mean'),
                    sd=('rpr', 'std'), mad=('rpr',
                         lambda z: np.median(np.abs(z - np.median(z)))))
               .reset_index())
    assert cells['n'].min() >= 2
    assert (cells.groupby(['subID','curCoherence']).size() == 5).all()
    cells.insert(0, 'experiment', exp)
    records.append(cells)
    person = (cells.groupby(['subID', 'curCoherence'], observed=True)
                   .agg(mean_sd=('sd', 'mean'), mean_mad=('mad', 'mean'),
                        n=('n', 'sum')).reset_index())
    pivot = person.pivot(index='subID', columns='curCoherence', values='mean_sd')
    mad_pivot = person.pivot(index='subID', columns='curCoherence', values='mean_mad')
    d = pivot[0.3] - pivot[0.7]
    dmad = mad_pivot[0.3] - mad_pivot[0.7]
    n = len(d)
    ci = stats.t.interval(0.95, n-1, loc=d.mean(), scale=stats.sem(d))
    t = stats.ttest_rel(pivot[0.3], pivot[0.7])
    summary.append(dict(experiment=exp, all_rows=len(frame), valid_rows=len(x),
                        n_subjects=n, cell_min_n=int(cells.n.min()),
                        cell_max_n=int(cells.n.max()),
                        low_mean_sd=float(pivot[0.3].mean()),
                        high_mean_sd=float(pivot[0.7].mean()),
                        low_minus_high_sd=float(d.mean()),
                        ci95_low=float(ci[0]), ci95_high=float(ci[1]),
                        t=float(t.statistic), p=float(t.pvalue),
                        low_gt_high_n=int((d > 0).sum()),
                        low_minus_high_mad=float(dmad.mean()),
                        mad_p=float(stats.ttest_rel(mad_pivot[0.3],mad_pivot[0.7]).pvalue)))
    for sid in pivot.index:
        participants.append(dict(experiment=exp, subID=sid, low_sd=pivot.loc[sid,0.3],
                                 high_sd=pivot.loc[sid,0.7], difference=d.loc[sid],
                                 low_mad=mad_pivot.loc[sid,0.3],
                                 high_mad=mad_pivot.loc[sid,0.7]))

pd.concat(records, ignore_index=True).to_csv(OUT/'duration_cells.csv', index=False)
pd.DataFrame(participants).to_csv(OUT/'participant_contrasts.csv', index=False)
pd.DataFrame(summary).to_csv(OUT/'summary.csv', index=False)
(OUT/'provenance.json').write_text(json.dumps({
    'inputs': [str(IN/name) for _, name in SOURCES],
    'filter': '~is_outlier and nonmissing rpr',
    'estimand': 'equal-duration-weighted within-participant SD(rpr) in participant x duration x current-coherence cells; paired L-H',
    'sensitivity': 'within-cell median absolute deviation with same weighting',
    'packages': {'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': __import__('scipy').__version__}
}, indent=2))
print(pd.DataFrame(summary).to_string(index=False))
