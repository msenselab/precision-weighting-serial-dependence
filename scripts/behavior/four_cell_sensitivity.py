"""Random-effects and cluster-robust sensitivity for matched four-cell contrasts."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sdu import paths as SP  # noqa: E402


ROOT=Path(__file__).resolve().parents[1]; OUT=SP.RESULTS_BEHAVIOR / "analysis"/"four_cells"; OUT.mkdir(parents=True,exist_ok=True)
FORMULA="curBias ~ curDur_c * previous_low * current_low + preDur_c * previous_low * current_low"
DATA={"Exp1_dynamic":"E2_reprocessed.csv","Exp2_fixed":"E1_reprocessed.csv"}


def vectors(names):
    def v(p,c):
        a=np.zeros(len(names))
        for term,value in [("preDur_c",1),("preDur_c:previous_low",p),("preDur_c:current_low",c),("preDur_c:previous_low:current_low",p*c)]: a[names.index(term)]=value
        return a
    cells={"HH":v(0,0),"HL":v(0,1),"LH":v(1,0),"LL":v(1,1)}
    return {"current_L_minus_H":(cells["HL"]+cells["LL"]-cells["HH"]-cells["LH"])/2,"previous_H_minus_L":(cells["HH"]+cells["HL"]-cells["LH"]-cells["LL"])/2,"same_minus_switch":(cells["HH"]+cells["LL"]-cells["HL"]-cells["LH"])/2}


rows=[]; records=[]
for exp,filename in DATA.items():
    d=pd.read_csv(SP.PREPROCESSED/filename,dtype={"subID":str})
    x=d.loc[~d.is_outlier].dropna(subset=["subID","curBias","curDur","preDur1back","curCoherence","preCoherence1back"]).copy()
    x["curDur_c"]=x.curDur-1.2; x["preDur_c"]=x.preDur1back-1.2; x["previous_low"]=np.isclose(x.preCoherence1back,.3).astype(int); x["current_low"]=np.isclose(x.curCoherence,.3).astype(int)
    fits=[]
    for label,re_formula in [("RI","1"),("RI_cur","~curDur_c"),("RI_pre","~preDur_c"),("RI_cur_pre","~curDur_c + preDur_c")]:
        model=smf.mixedlm(FORMULA,x,groups=x.subID,re_formula=re_formula)
        fit=None; used=None; messages=[]
        for method in ["lbfgs","powell","cg"]:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always"); candidate=model.fit(reml=True,method=method,maxiter=3000,disp=False)
                messages += [str(w.message) for w in caught]
                if candidate.converged: fit=candidate; used=method; break
            except Exception as exc: messages.append(f"{method}: {exc}")
        if fit is None: records.append({"experiment":exp,"specification":label,"converged":False,"warnings":" | ".join(sorted(set(messages)))}); continue
        names=fit.fe_params.index.tolist(); beta=fit.fe_params.to_numpy(); cov=fit.cov_params().loc[names,names].to_numpy(); fits.append((label,names,beta,cov))
        records.append({"experiment":exp,"specification":label,"converged":True,"method":used,"warnings":" | ".join(sorted(set(messages)))})
    ols=smf.ols(FORMULA,x).fit(cov_type="cluster",cov_kwds={"groups":x.subID}); names=ols.params.index.tolist(); fits.append(("OLS_cluster",names,ols.params.to_numpy(),ols.cov_params().loc[names,names].to_numpy())); records.append({"experiment":exp,"specification":"OLS_cluster","converged":True,"method":"OLS cluster-robust"})
    for label,names,beta,cov in fits:
        for contrast,a in vectors(names).items():
            est=float(a@beta); se=float(np.sqrt(a@cov@a)); z=est/se
            rows.append({"experiment":exp,"specification":label,"contrast":contrast,"beta":est,"se":se,"ci_low":est-1.96*se,"ci_high":est+1.96*se,"z":z,"p":2*norm.sf(abs(z))})

pd.DataFrame(rows).to_csv(OUT/"trial_lmm_sensitivity_contrasts.csv",index=False); pd.DataFrame(records).to_csv(OUT/"trial_lmm_sensitivity_fit_record.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
