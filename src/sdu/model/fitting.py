"""Helpers for the fits of the standard model space (fit_systematic.py, Appendix C.3; 2026-10-28, restricted to the
standard space on 2026-10-31).

  load        one experiment's trial arrays per participant (log duration, coherence, session position, response,
              validity mask)
  parents     the models with one mechanism less (one locus fewer in C, the nested T level, S0, B1)
  embed       a parent's solution embedded in a child's parameter space (exact for C and T, at the parameter bound
              for S and B)
  read_fits   fits files -> the best row per (exp, Sub, model)

The fitting loop of the earlier spaces (fit_models, fit_participant, main) is archived with them
(in the project's working repository).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from sdu.model import engine as G  # noqa: E402
from sdu import paths as SP  # noqa: E402

OUTDIR = SP.MODEL_RESULTS / ("framework_v2" if G.BOUNDS == "v2" else "framework")
DATA = SP.MODEL_INPUT / "input_full_sequence.csv"
T_PARENTS = {"Tn": [], "T0": ["Tn"], "T1s": ["T0"], "T1n": ["Tn"], "T0S": ["Tn", "T0"], "T1nS": ["Tn", "T1n"],
             "T1sS": ["T0S", "T1s"], "TnW": ["Tn"], "T0W": ["T0", "TnW", "T0S"], "T1nW": ["T1n", "TnW", "T1nS"],
             "T1sW": ["T1s", "T0W", "T1sS"]}
LOCI_INDEX = {frozenset(v): k for k, v in G.C_LOCI.items()}


def _opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def fit_folders():
    """Current fitted parameters; historical folders require explicit v1 selection."""
    default = "framework_v2" if G.BOUNDS == "v2" else "framework_std,framework_b9"
    return [SP.MODEL_RESULTS / f for f in _opt("fits", default).split(",") if f]


def fit_paths(exp=None, sub=None):
    pattern = f"exp{exp if exp is not None else '*'}_{sub if sub is not None else '*'}.csv"
    paths = [p for folder in fit_folders() for p in sorted(folder.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No {G.BOUNDS} fits: {fit_folders()} / {pattern}")
    return paths


def current_fits(exp=None, sub=None):
    return read_fits(fit_paths(exp, sub))


def result_path(name, suffix=".csv"):
    """Keep new summaries separate from historical results."""
    if G.BOUNDS == "v2" and not name.endswith("_v2"):
        name += "_v2"
    return SP.MODEL_RESULTS / (name + suffix)


def parents(spec):
    out = [G.Spec(spec.c, t, spec.s, spec.b) for t in T_PARENTS[spec.t]]
    loci = G.C_LOCI[spec.c]
    out += [G.Spec(LOCI_INDEX[frozenset(set(loci) - {q})], spec.t, spec.s, spec.b) for q in loci]
    if spec.s:
        out.append(G.Spec(spec.c, spec.t, 0, spec.b))
    if spec.b != 1:
        out.append(G.Spec(spec.c, spec.t, spec.s, 1))
    return out


def embed(th_parent, parent, child):
    """Child theta that reproduces the parent's model as closely as the child's bounds allow."""
    th = th_parent.copy()
    tilt = ("T1s", "T1sS")
    if G.BOUNDS == "v3":                                  # natural-scale parameters: the persistences themselves
        phL, phH = G.weights(th_parent, G.T_CODES[parent.t])
        th[G.IX["wuL"]], th[G.IX["wuH"]] = phL, phH
        if child.t in tilt:                               # T2: beta = phi^L, wuH = phi^H
            th[G.IX["beta"]] = phL
        if "wsw" in G.T_FREE[child.t] and "wsw" not in G.T_FREE[parent.t]:
            reset = G.T_CODES[parent.t] in G.RESET_CODES or parent.t == "Tn"
            th[G.IX["wsw"]] = 0.0 if reset else (phL if parent.t == "T1n" else phH)
        if child.s and not parent.s:
            th[G.IX["lsw"]] = G.LO[G.IX["lsw"]]
        if child.b != 1 and parent.b == 1:
            th[G.IX["lq3"]], th[G.IX["lam"]] = 0.0, 0.0
        lo, hi = child.bounds()
        return np.clip(np.where(lo == hi, lo, th), lo, hi)
    if parent.t not in tilt:                              # the parent's persistence at Low and High (a tilted parent
        w = np.clip(np.array(G.weights(th_parent, G.T_CODES[parent.t])), 1e-9, G.WMAX - 1e-9) / G.WMAX
        for nm, v in zip(("wuL", "wuH"), np.log(w / (1 - w))):        # keeps its base and tilt: 2026-10-31 fix)
            th[G.IX[nm]] = v
    if child.t in tilt and parent.t not in tilt:
        th[G.IX["beta"]] = 0.0
    if "wsw" in G.T_FREE[child.t] and "wsw" not in G.T_FREE[parent.t]:     # S5 child: phi_sw of the parent
        if G.T_CODES[parent.t] in G.RESET_CODES or parent.t == "Tn":
            th[G.IX["wsw"]] = G.LO[G.IX["wsw"]]                               # phi = 0 on switch trials
        else:
            th[G.IX["wsw"]] = th[G.IX["wuL"] if parent.t == "T1n" else G.IX["wuH"]]
    if child.s and not parent.s:
        th[G.IX["lsw"]] = G.LO[G.IX["lsw"]]
    if child.b != 1 and parent.b == 1:
        th[G.IX["lq3"]], th[G.IX["lam"]] = G.LO[G.IX["lq3"]], 0.0
    lo, hi = child.bounds()
    return np.clip(np.where(lo == hi, lo, th), lo, hi)


def load(exp):
    d = pd.read_csv(DATA)
    d = d[d.exp == exp].sort_values(["Sub", "trial_num"])
    out = {}
    for s, g in d.groupby("Sub"):
        tn = g.trial_num.to_numpy(float)
        out[s] = (np.log(g.stimulus.to_numpy(float)), g.coherence.to_numpy(float),
                  2 * (tn - tn.min()) / (tn.max() - tn.min()) - 1, np.nan_to_num(g.response.to_numpy(float)),
                  g.response.notna().to_numpy())
    return out


def read_fits(paths):
    """Fits files -> one row per (exp, Sub, model): the best LL over passes."""
    d = pd.concat([pd.read_csv(p) for p in paths])
    for n in G.NAMES:                                     # parameters added after a file was written
        if n not in d:
            d[n] = G.FIXED.get(n, G.X0[G.IX[n]])
    return d.sort_values("LL").drop_duplicates(["exp", "Sub", "model"], keep="last").reset_index(drop=True)
