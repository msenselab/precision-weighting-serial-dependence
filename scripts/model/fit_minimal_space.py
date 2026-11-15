"""Fit the 3SKF v2 model space (notes/model/3skf_v2.md; 2026-11-09).

Space: C1-C15 x T {T0, T1, T3} x S {S0, S1 memory reversion, S2 gain cut (first submission), S3 fast-state reset}
x B {B0 none, B1 prediction, B2 report (latent AR(1))}; duplicates removed by the rules checked
numerically on 2026-11-09 (T0 x S3 == T0 x S0; under B0 the q3 modulation is inert; C3 x B0 has no coherence
effect and is not admissible): 407 models. The exploratory measurement-bias B3 is not part of v2.

Each work unit is one participant x one chunk of models. Every model is fitted by maximum likelihood (L-BFGS-B,
N_STARTS starts: the default vector and random points within the bounds) and simulated N_SIM times on the
participant's own sequence; the main-text statistics (sdu.model.stats) of the simulations are averaged.
One CSV per unit (resumable):  <SDU_MODEL_RESULTS>/minimal_space/exp<code>_<Sub>_c<chunk>.csv

Configuration (required): GKF_SCALE=lin GKF_CORE=reduced GKF_BOUNDS=v3 GKF_INIT=gs GKF_QDIAG=1.
GKF_NOISELESS=1 reproduces the primary v2 analysis; GKF_NOISELESS=0 is the separate sensory-noise sensitivity analysis
in which R scales both the observer's gain and the trial-wise measurement noise. The two modes write to separate
folders. (GKF_QDIAG=1 since 2026-11-09: the first submission's diagonal process noise; LRZ job 5722649 ran without it.)
Encoding check: GKF_SCALE=log GKF_LIK=gh (log encoding; the response exp(u) needs the Gauss-Hermite likelihood).
Readout persistence (GKF_READOUT=1; 2026-11-11): the memory filter has phi = 0 and the persistence acts only on the
report (engine.READOUT); the memory filter's F has first row (0, 1, F13) in every model, so there is no T dimension.
The readout persistence has phi^L and phi^H free (engine T1s / T1sS, label R); with both at 0 the model equals the T0
models of the v2 space. 15 C x 4 S x 3 B minus the 32 inert B0 duplicates = 148 models, written to readout_space/.

Usage:
    python scripts/model/fit_minimal_space.py --list                      # print the space (label, engine, k)
    python scripts/model/fit_minimal_space.py --task=<i> --ntasks=<n> [--chunks=11] [--models=<n>] [--subs=<n>]
        [--only=C3-T3-S0-B2,C4-T3-S0-B2,C7-T3-S0-B2]
        processes units i, i + n, i + 2n, ... (SLURM_ARRAY_TASK_ID / array size)
"""
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

from sdu import names
from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import stats as S

N_STARTS = int(FF._opt("starts", "8"))
N_SIM = int(FF._opt("nsim", "20"))
N_CHUNKS = int(FF._opt("chunks", "11"))
OUTDIR = SP.MODEL_RESULTS / ("minimal_space" if G.NOISELESS else "minimal_space_sensory")

T_LV = {"T0": ("Tn", "Tn"), "T1": ("T0", "T0S"), "T3": ("T1n", "T1nS")}     # (engine T, engine T with fast reset)
if G.READOUT:                                                                # persistence at the readout only
    T_LV = {"R": ("T1s", "T1sS")}                 # phi^L and phi^H both free (beta, wuH); no T dimension
    OUTDIR = SP.MODEL_RESULTS / ("readout_space" if G.NOISELESS else "readout_space_sensory")
S_LV = {"S0": 0, "S1": 3, "S2": G.GAIN_CUT, "S3": "reset"}
B_LV = {"B0": 1, "B1": 2, "B2": 3}                       # B3 measurement alternative removed from v2
OLD_B = {"B1": "B0", "Bt": "B1"}                          # labels in the files of LRZ job 5722649
Q3_LOCI = {3, 7, 9, 10, 12, 13, 14, 15}                                      # C levels that modulate q3


def space():
    """[(label, Spec)] of the 407 distinct admissible models (148 under GKF_READOUT), in a fixed order."""
    out = []
    for c in range(1, 16):
        for t, (te, tr) in T_LV.items():
            for s, sc in S_LV.items():
                if t == "T0" and s == "S3":
                    continue                                                 # == T0 x S0
                for b, bc in B_LV.items():
                    if b == "B0" and c in Q3_LOCI:
                        continue                                             # q3 inert (duplicate or no C effect)
                    spec = G.Spec(c, tr, 0, bc) if sc == "reset" else G.Spec(c, te, sc, bc)
                    out.append((f"C{c}-{t}-{s}-{b}", spec))
    return out


def relabel(label):
    """Model label of an older result file (B1 -> B0, Bt -> B1) in the current naming."""
    c, t, sw, b = label.split("-")
    return f"{c}-{t}-{sw}-{OLD_B.get(b, b)}"


def units(n_models=None, n_subs=None, only=None):
    sp = space()
    if only:
        requested = set(only)
        available = {label for label, _ in sp}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown model labels: {sorted(unknown)}")
        sp = [(label, spec) for label, spec in sp if label in requested]
    if n_models:
        sp = sp[:n_models]
    chunks = [c for c in np.array_split(np.arange(len(sp)), N_CHUNKS) if len(c)]
    out = []
    for exp in (2, 1):
        subs = sorted(FF.load(exp))
        for sub in (subs[:n_subs] if n_subs else subs):
            for k, idx in enumerate(chunks):
                out.append((exp, sub, k, [sp[i] for i in idx]))
    return out


def fit_unit(exp, sub, k, models, J, data):
    from scipy.optimize import minimize
    jnp = J["jnp"]
    arrs = data[sub]
    logT, coh, tau, y, mask = arrs
    a = [jnp.asarray(v) for v in arrs]
    obs = S.participant_stats_arrays(exp, sub, np.where(mask, y, np.nan))
    rows = []
    for label, spec in models:
        code = jnp.asarray(spec.vec())
        lo, hi = spec.bounds()
        rng = np.random.default_rng(zlib.crc32(f"minspace|{exp}|{sub}|{label}".encode()))
        f = lambda th: tuple(np.asarray(v, float) for v in J["vg"](jnp.asarray(th), code, *a))
        starts = [np.clip(G.X0, lo, hi)] + [np.where(lo == hi, lo, lo + (hi - lo) * rng.uniform(.1, .9, len(lo)))
                                            for _ in range(N_STARTS - 1)]
        if G.READOUT:                             # a start at phi^L = phi^H = 0 (the nested T0 model)
            x00 = np.clip(G.X0, lo, hi)
            x00[G.IX["beta"]] = x00[G.IX["wuH"]] = 0.0
            starts.append(x00)
        best, lls = None, []
        for x0 in starts:
            r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=4000))
            if np.isfinite(r.fun):
                lls.append(-r.fun)
                if best is None or r.fun < best.fun:
                    best = r
        n_best = int(np.sum(np.abs(np.array(lls) - (-best.fun)) < 0.01))   # starts that reached the optimum
        srng = np.random.default_rng(zlib.crc32(f"minspace-sim|{exp}|{sub}|{label}".encode()))
        sims = pd.DataFrame([S.participant_stats_arrays(exp, sub, G.simulate(best.x, spec, np.exp(logT), coh, tau, mask,
                                                                              srng, return_mean=False)[0])
                             for _ in range(N_SIM)])
        rows.append(dict(exp=exp, experiment=names.EXPERIMENT[exp], Sub=sub, model=label, engine=spec.name,
                         sensory_noise=not G.NOISELESS,
                         k=spec.k(), LL=-best.fun, n_at_best=n_best, n_starts=len(lls),
                         **dict(zip(G.NAMES, best.x)), **{f"obs_{q}": v for q, v in obs.items()},
                         **{f"sim_{q}": v for q, v in sims.mean().items()}))
    return pd.DataFrame(rows)


def main():
    assert (G.LIN or G.LIK == "gh") and G.QDIAG and G.CORE == "reduced" and G.BOUNDS == "v3" \
        and G.INIT == "gs", "set GKF_SCALE=lin (or GKF_SCALE=log GKF_LIK=gh, the encoding check) GKF_CORE=reduced " \
        "GKF_BOUNDS=v3 GKF_INIT=gs GKF_QDIAG=1; set GKF_NOISELESS=1 for primary or 0 for sensory noise"
    if "--list" in sys.argv:
        for label, spec in space():
            print(label, spec.name, spec.k())
        print(len(space()), "models", file=sys.stderr)
        return
    task = int(FF._opt("task", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    ntasks = int(FF._opt("ntasks", "1"))
    nm = int(FF._opt("models", "0")) or None
    ns = int(FF._opt("subs", "0")) or None
    only = tuple(filter(None, FF._opt("only", "").split(",")))
    us = units(nm, ns, only)[task::ntasks]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    J = G.jax_funcs()
    data = {2: FF.load(2), 1: FF.load(1)}
    for exp, sub, k, models in us:
        out = OUTDIR / f"exp{exp}_{sub}_c{k:02d}.csv"
        if out.exists():
            continue
        t0 = time.time()
        d = fit_unit(exp, sub, k, models, J, data[exp])
        d.to_csv(out.with_suffix(".tmp"), index=False)
        os.replace(out.with_suffix(".tmp"), out)
        print(f"done exp{exp} {sub} chunk {k}: {len(models)} models, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
