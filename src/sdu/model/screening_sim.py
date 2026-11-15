"""Screening simulations: every model of the reported space, every participant (main-text statistics, sdu.model.stats).

For each participant and model, N data sets are simulated exactly (engine.simulate) from the participant's fitted
parameters (best LL over the rows of the fit file) on the participant's own trial sequence and validity mask; each
simulated data set goes through sdu.model.stats.participant_stats, and the mean over the N data sets is stored with
the observed statistics. One file per participant (resumable):
    <SDU_MODEL_RESULTS>/screening/sim_exp<code>_<Sub>.csv   columns exp, experiment, Sub, model (manuscript label),
                                                             k, LL, obs_<stat>, sim_<stat>

Usage (repo root):
    SDU_MODEL_RESULTS=results/model/v3 GKF_BOUNDS=v3 GKF_INIT=gs GKF_LIK=gh GKF_CORE=reduced_ase \\
        python -m sdu.model.screening_sim [workers] [--fits=fits] [--n=20] [--exp=2,1] [--set=core|prediction]
"""
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sdu import names
from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import space as KS
from sdu.model import stats as S

N_SIM = int(FF._opt("n", "20"))
FITS = SP.MODEL_RESULTS / FF._opt("fits", "fits")
SET = FF._opt("set", "core")                       # core: the 572 reported models; prediction: Bt and Btm (330)
OUTDIR = SP.MODEL_RESULTS / "screening"
EXPS = [int(e) for e in FF._opt("exp", "2,1").split(",")]
CORE_PUBLIC = set(KS.prediction_bias_names() if SET == "prediction" else KS.core_names())


def out_path(exp, sub):
    suffix = "" if SET == "core" else f"_{SET}"
    return OUTDIR / f"sim_exp{exp}_{sub}{suffix}.csv"


def run(job):
    exp, sub, arrs = job
    logT, coh, tau, y, mask = arrs
    T = np.exp(logT)
    obs = S.participant_stats_arrays(exp, sub, np.where(mask, y, np.nan))
    fits = FF.read_fits([FITS / f"exp{exp}_{sub}.csv"])
    fits = fits[fits.public.isin(CORE_PUBLIC)]
    rows = []
    for _, r in fits.iterrows():
        spec = G.Spec.parse(r.model)
        th = r[G.NAMES].to_numpy(float)
        rng = np.random.default_rng(zlib.crc32(f"screen|{exp}|{sub}|{r.public}".encode()))
        sims = pd.DataFrame([S.participant_stats_arrays(exp, sub, G.simulate(th, spec, T, coh, tau, mask, rng)[0])
                             for _ in range(N_SIM)])
        rows.append(dict(exp=exp, experiment=names.EXPERIMENT[exp], Sub=sub, model=names.label(r.public), k=r.k,
                         LL=r.LL, **{f"obs_{k}": v for k, v in obs.items()},
                         **{f"sim_{k}": v for k, v in sims.mean().items()}))
    d = pd.DataFrame(rows)
    d.to_csv(out_path(exp, sub), index=False)
    return exp, sub, len(d)


def main():
    workers = int(next((a for a in sys.argv[1:] if not a.startswith("--")), "8"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for exp in EXPS:
        data = FF.load(exp)
        jobs += [(exp, s, data[s]) for s in sorted(data) if not out_path(exp, s).exists()]
    print(f"{len(jobs)} participant jobs, {N_SIM} simulations per model, fits {FITS}", flush=True)
    with ProcessPoolExecutor(min(workers, max(len(jobs), 1))) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            exp, sub, n = f.result()
            print(f"done exp {exp} participant {sub}: {n} models", flush=True)
    print("===EXIT", flush=True)


if __name__ == "__main__":
    main()
