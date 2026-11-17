"""Estimate participant-level slopes for the four coherence transitions.

Sensitivity analysis only; not a replacement for the primary trial-level interaction.
Uses frozen reprocessed rows and objective-coherence H/L labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sdu import paths as SP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = SP.BEHAVIOR_TRANSITIONS
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for exp, fname in [('Exp1_ramped','experiment1_ramped.pkl'),('Exp2_constant','experiment2_constant.pkl')]:
    df = pd.read_pickle(SP.PREPROCESSED/fname)
    x = df.loc[(~df.is_outlier) & df[['curBias','curDur','preDur1back','curCoherence','preCoherence1back']].notna().all(axis=1)].copy()
    assert x.subID.nunique() == 22
    for (sid, prev, current), cell in x.groupby(['subID','preCoherence1back','curCoherence']):
        assert len(cell) > 8
        X = np.column_stack([np.ones(len(cell)),cell.curDur.to_numpy(),cell.preDur1back.to_numpy()])
        assert np.linalg.matrix_rank(X) == 3
        beta = np.linalg.lstsq(X, cell.curBias.to_numpy(), rcond=None)[0]
        label = ('L' if prev == .3 else 'H') + ('L' if current == .3 else 'H')
        rows.append(dict(experiment=exp,subID=sid,cell=label,n=len(cell),
                         intercept=beta[0],current_duration_slope=beta[1],
                         previous_duration_slope=beta[2]))
slopes=pd.DataFrame(rows)
assert len(slopes)==176 and (slopes.groupby(['experiment','subID']).size()==4).all()
slopes.to_csv(OUT/'participant_cell_slopes.csv',index=False)
results=[]
for exp, d in slopes.groupby('experiment'):
    wide=d.pivot(index='subID',columns='cell',values='previous_duration_slope')
    contrasts={
      'current_L_minus_H':(wide['LL']+wide['HL']-wide['LH']-wide['HH'])/2,
      'previous_H_minus_L':(wide['HL']+wide['HH']-wide['LL']-wide['LH'])/2,
      'same_minus_switch':(wide['LL']+wide['HH']-wide['LH']-wide['HL'])/2,
      'four_cell_interaction':(wide['LL']+wide['HH']-wide['LH']-wide['HL']),
    }
    for cell in ['LL','LH','HL','HH']:
        contrasts[cell]=wide[cell]
    for name,v in contrasts.items():
        n=len(v); t=stats.ttest_1samp(v,0)
        ci=stats.t.interval(.95,n-1,loc=v.mean(),scale=stats.sem(v))
        results.append(dict(experiment=exp,contrast=name,n_subjects=n,
                            estimate=v.mean(),ci_low=ci[0],ci_high=ci[1],
                            t=t.statistic,p=t.pvalue))
report=pd.DataFrame(results)
report.to_csv(OUT/'participant_contrasts.csv',index=False)
print(report.to_string(index=False))
