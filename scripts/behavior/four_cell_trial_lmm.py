"""Matched four-cell trial-level LMM on frozen reprocessed trials.

New revision analysis, not a rerun of the submitted primary two-cell model.
H/L refer to objective previous/current coherence. REML and random slopes
match generate_manuscript_behavior_results.py:dual_lmms.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sdu import paths as SP  # noqa: E402

ROOT=Path(__file__).resolve().parents[1]
OUT=SP.RESULTS_BEHAVIOR / "analysis"/'four_cells'
OUT.mkdir(parents=True,exist_ok=True)
FORMULA='curBias ~ curDur_c * previous_low * current_low + preDur_c * previous_low * current_low'
results=[]; fits=[]
for exp,filename in [('Exp1_dynamic','E2_reprocessed.pkl'),('Exp2_fixed','E1_reprocessed.pkl')]:
    frame=pd.read_pickle(SP.PREPROCESSED/filename)
    x=frame.loc[~frame.is_outlier].dropna(subset=['subID','curBias','curDur','preDur1back','curCoherence','preCoherence1back']).copy()
    x['subID']=x.subID.astype(str)
    assert x.subID.nunique()==22
    x['curDur_c']=x.curDur-1.2
    x['preDur_c']=x.preDur1back-1.2
    x['previous_low']=np.isclose(x.preCoherence1back,.3).astype(int)
    x['current_low']=np.isclose(x.curCoherence,.3).astype(int)
    model=smf.mixedlm(FORMULA,x,groups=x.subID,re_formula='~curDur_c + preDur_c')
    errors=[]
    # Pin the optimizer recorded by the frozen analysis so convergence changes across
    # statsmodels versions do not move the published coefficients by small amounts.
    preferred={'Exp1_dynamic':'powell','Exp2_fixed':'lbfgs'}[exp]
    for method in [preferred]+[m for m in ['lbfgs','powell','cg'] if m != preferred]:
        try:
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always')
                fit=model.fit(reml=True,method=method,disp=False,maxiter=3000)
            if fit.converged:
                break
            errors.append(method+' not converged')
        except Exception as exc:
            errors.append(method+': '+str(exc))
    assert fit.converged,(exp,errors)
    print(exp, 'n',len(x),'fit',method,'warnings',list(set(str(w.message) for w in warn)))
    names=fit.fe_params.index.tolist()
    betas=fit.fe_params.to_numpy()
    cov=fit.cov_params().loc[names,names].to_numpy()
    fits.append(dict(experiment=exp,n_rows=len(x),n_subjects=x.subID.nunique(),formula=FORMULA,
                     random='~curDur_c + preDur_c',method=method,converged=fit.converged,
                     warnings=' | '.join(sorted(set(str(w.message) for w in warn))),
                     prior_low_terms='; '.join(names)))
    def vector(p,c):
        a=np.zeros(len(names))
        for term,value in [('preDur_c',1),('preDur_c:previous_low',p),
                           ('preDur_c:current_low',c),('preDur_c:previous_low:current_low',p*c)]:
            a[names.index(term)]=value
        return a
    V={'HH':vector(0,0),'HL':vector(0,1),'LH':vector(1,0),'LL':vector(1,1)}
    contrasts={**V,
      'current_L_minus_H':(V['HL']+V['LL']-V['HH']-V['LH'])/2,
      'previous_H_minus_L':(V['HH']+V['HL']-V['LH']-V['LL'])/2,
      'same_minus_switch':(V['HH']+V['LL']-V['HL']-V['LH'])/2,
      'four_cell_interaction':V['HH']+V['LL']-V['HL']-V['LH'],
      # 2026-11-04: the LL-dominant pattern of Experiment 2, and HL - LL (reliability weighting predicts
      # HL > LL, current uncertainty HL ~ LL)
      'LL_minus_other_three':V['LL']-(V['HH']+V['HL']+V['LH'])/3,
      'LL_minus_LH':V['LL']-V['LH'],
      'LL_minus_HL':V['LL']-V['HL'],
      'LL_minus_HH':V['LL']-V['HH'],
      'HL_minus_LL':V['HL']-V['LL']}
    for name,a in contrasts.items():
        estimate=float(a@betas); se=float(np.sqrt(a@cov@a)); z=estimate/se
        results.append(dict(experiment=exp,contrast=name,beta=estimate,se=se,
                            ci_low=estimate-1.96*se,ci_high=estimate+1.96*se,
                            z=z,p=2*norm.sf(abs(z))))
pd.DataFrame(results).to_csv(OUT/'trial_lmm_contrasts.csv',index=False)
pd.DataFrame(fits).to_csv(OUT/'trial_lmm_fit_record.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
