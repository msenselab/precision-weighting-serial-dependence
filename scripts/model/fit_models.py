"""Fit the 148 models of the manuscript (Appendix B.3) to every participant and simulate the fitted models.

Model space: C1-C15 x S0-S3 x B0-B2 without the C levels that modulate q3 under B0 (engine.space()). Every model is
fitted by maximum likelihood (L-BFGS-B; 13 starts: the default point, 11 random points within the bounds, and the
default point with zero retention) and simulated N_SIM = 20 times on the participant's own sequence and validity
mask; the behavioural statistics of the data (obs_<statistic>) and the mean over the simulations (sim_<statistic>,
sdu.model.stats) are stored with the fit. Model comparison and screening: python -m sdu.model.comparison.

Each work unit is one participant and one chunk of models. One CSV per unit is written to the requested work
directory. The public archive consolidates these files as results/model/fits/all.csv.

Usage:
    python scripts/model/fit_models.py --list                      # print the model space (name, k)
    python scripts/model/fit_models.py --task=<i> --ntasks=<n> [--chunks=11] [--only=C3-S0-B2,C3-S3-B2]
        [--out=results/model/fits/tasks]
        processes units i, i + n, i + 2n, ... (e.g. SLURM_ARRAY_TASK_ID and the array size)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import stats as S

N_SIM = int(FF.option("nsim", "20"))
N_CHUNKS = int(FF.option("chunks", "11"))
OUTDIR = Path(FF.option("out", str(SP.FITS / "tasks")))
# Seeds of the stored fits (see fitting.SEED_CODE): text keys with the experiment's collection code and the model
# name with the readout marker R (C3-R-S0-B2), the name used when the fits were run.
FIT_KEY = "minspace|{code}|{sub}|{name}"
SIM_KEY = "minspace-sim|{code}|{sub}|{name}"


def seed_name(model):
    return f"C{model.c}-R-S{model.s}-B{model.b}"


def units(only=None):
    space = G.space()
    if only:
        unknown = set(only) - {m.name for m in space}
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
        space = [m for m in space if m.name in only]
    chunks = [c for c in np.array_split(np.arange(len(space)), N_CHUNKS) if len(c)]
    return [(exp, sub, k, [space[i] for i in idx])
            for exp in FF.EXPERIMENT for sub in sorted(FF.load(exp)) for k, idx in enumerate(chunks)]


def fit_unit(exp, sub, models, J, arrays):
    T, coh, y, mask = arrays
    obs = S.participant_stats_arrays(exp, sub, np.where(mask, y, np.nan))
    rows = []
    for model in models:
        keys = dict(code=FF.SEED_CODE[exp], sub=sub, name=seed_name(model))
        best, lls = FF.fit(J, model, arrays, np.random.default_rng(FF.seed(FIT_KEY.format(**keys))))
        n_best = int(np.sum(np.abs(np.array(lls) + best.fun) < 0.01))   # starts that reached the optimum
        rng = np.random.default_rng(FF.seed(SIM_KEY.format(**keys)))
        sims = pd.DataFrame([S.participant_stats_arrays(exp, sub, G.simulate(best.x, model, T, coh, mask, rng,
                                                                              return_mean=False)[0])
                             for _ in range(N_SIM)])
        rows.append(dict(exp=exp, experiment=FF.EXPERIMENT[exp], Sub=sub, model=model.name, k=model.k(),
                         LL=-best.fun, n_at_best=n_best, n_starts=len(lls), **dict(zip(G.NAMES, best.x)),
                         **{f"obs_{q}": v for q, v in obs.items()},
                         **{f"sim_{q}": v for q, v in sims.mean().items()}))
    return pd.DataFrame(rows)


def main():
    if "--list" in sys.argv:
        for model in G.space():
            print(model.name, model.k())
        print(len(G.space()), "models", file=sys.stderr)
        return
    task = int(FF.option("task", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ntasks = int(FF.option("ntasks", "1"))
    only = tuple(filter(None, FF.option("only", "").split(",")))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    J = G.jax_funcs()
    data = {exp: FF.load(exp) for exp in FF.EXPERIMENT}
    for exp, sub, k, models in units(only)[task::ntasks]:
        out = OUTDIR / f"exp{exp}_{sub}_c{k:02d}.csv"
        if out.exists():
            continue
        t0 = time.time()
        d = fit_unit(exp, sub, models, J, data[exp][sub])
        d.to_csv(out.with_suffix(".tmp"), index=False)
        os.replace(out.with_suffix(".tmp"), out)
        print(f"done {FF.EXPERIMENT[exp]} {sub} chunk {k}: {len(models)} models, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
