"""Trial data for the model, the stored fits, and the maximum-likelihood fit of one model.

  load        one experiment's trial arrays per participant: durations (s), coherence, response, validity mask
  read_fits   the stored fits (results/model/fits/all.csv): one row per experiment, participant and model
  theta       the parameter vector of one fit row
  fit         maximum-likelihood fit of one model (L-BFGS-B from several starting points)
"""
import sys
import zlib

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G

EXPERIMENT = {1: "Exp 1", 2: "Exp 2"}
N_RANDOM = 11           # random starting points per fit; with the default and the zero-retention point, 13

# Reproducing the stored runs. Their random seeds come from text keys in which the experiments carry the codes of
# the data collection (Experiment 1 = 2, Experiment 2 = 1), and each random starting point was drawn as 25 uniform
# numbers, of which the entries of DRAW_INDEX are the parameters of NAMES. Keeping both makes reruns start from the
# same points as the stored fits.
SEED_CODE = {1: 2, 2: 1}
DRAW_SIZE = 25
DRAW_INDEX = [0, 1, 6, 5, 7, 8, 10, 11, 12, 17, 18, 19, 20, 21]


def option(name, default):
    """Value of a command-line option --name=value."""
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def load(exp):
    """Participant -> (durations, coherence, responses, valid) of experiment `exp`, in trial order. Responses of
    trials excluded in preprocessing are set to 0 and masked: these trials are presented to the observer but do not
    enter the likelihood."""
    d = pd.read_csv(SP.MODEL_INPUT)
    d = d[d.exp == exp].sort_values(["Sub", "trial_num"])
    return {s: (g.stimulus.to_numpy(float), g.coherence.to_numpy(float), np.nan_to_num(g.response.to_numpy(float)),
                g.response.notna().to_numpy())
            for s, g in d.groupby("Sub")}


def read_fits(paths=None):
    """Stored fits -> one row per (exp, Sub, model)."""
    paths = [SP.FITS / "all.csv"] if paths is None else paths
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert not d.duplicated(["exp", "Sub", "model"]).any()
    return d


def theta(row):
    """Parameter vector of one fit row."""
    return np.asarray(row[G.NAMES], float)


def random_point(lo, hi, rng, a=.1, b=.9):
    """Uniform point between the bounds, at fractions a-b of each parameter's range (fixed entries at their value)."""
    return np.where(lo == hi, lo, lo + (hi - lo) * rng.uniform(a, b, DRAW_SIZE)[DRAW_INDEX])


def starts(model, rng, n_random=N_RANDOM):
    """Starting points: the default point, n_random uniform points within the bounds, and the default point with
    zero retention (phi_L = phi_H = 0, the model without the trace of the preceding trial)."""
    lo, hi = model.bounds()
    out = [np.clip(G.X0, lo, hi)]
    out += [random_point(lo, hi, rng) for _ in range(n_random)]
    zero = np.clip(G.X0, lo, hi)
    zero[[G.IX["phi_L"], G.IX["phi_H"]]] = 0.0
    out.append(zero)
    return out


def fit(J, model, arrays, rng, n_random=N_RANDOM, return_all=False):
    """Maximum-likelihood fit of `model` to the trial arrays (durations, coherence, responses, valid) with the JAX
    functions J (engine.jax_funcs()). Returns the best scipy result and the log-likelihood of every start."""
    from scipy.optimize import minimize

    jnp = J["jnp"]
    code = jnp.asarray(model.code())
    a = [jnp.asarray(v) for v in arrays]
    lo, hi = model.bounds()

    def f(th):
        value, grad = J["vg"](jnp.asarray(th), code, *a)
        return float(value), np.asarray(grad, float)

    best, results = None, []
    for x0 in starts(model, rng, n_random):
        r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=4000))
        if np.isfinite(r.fun):
            results.append(r)
            if best is None or r.fun < best.fun:
                best = r
    return (best, results) if return_all else (best, [-r.fun for r in results])


def seed(*parts):
    """Reproducible seed from a text key."""
    return zlib.crc32("|".join(map(str, parts)).encode())
