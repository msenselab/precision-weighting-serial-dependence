"""Three-state observer on the Glasauer & Shi (2021) data, fitted separately to each sequence type (Appendix C).

For each participant and sequence (random walk, randomized), the models B0 (no bias state), B1 and B2 (latent response
bias) are fitted by maximum likelihood (L-BFGS-B; the default point, 4 random points and the zero-retention point).
Writes results/model/glasauer_shi/three_state_fits.csv.

Usage: python scripts/model/glasauer_shi/fit_three_state.py [--workers=8]
"""
import multiprocessing as mp

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import glasauer_shi as GS

N_RANDOM = 4
SEED_KEY = "ext|gs|{unit}"


def run(job):
    from scipy.optimize import minimize
    unit, (duration, response, valid) = job
    J = G.jax_funcs()
    jnp = J["jnp"]
    a = [jnp.asarray(v) for v in GS.three_state_arrays(duration, response, valid)]
    rng = np.random.default_rng(FF.seed(SEED_KEY.format(unit=unit)))
    rows = []
    for name, model in GS.THREE_STATE.items():
        lo, hi = GS.three_state_bounds(model)
        code = jnp.asarray(model.code())

        def f(th):
            value, grad = J["vg"](jnp.asarray(th), code, *a)
            return float(value), np.asarray(grad, float)

        starts = [np.clip(G.X0, lo, hi)] + [FF.random_point(lo, hi, rng, .15, .85) for _ in range(N_RANDOM)]
        zero = np.clip(G.X0, lo, hi)
        zero[G.IX["phi_H"]] = 0.0                     # the nested model without retention
        starts.append(zero)
        best = None
        for x0 in starts:
            r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=4000))
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        free = lo < hi
        at_bound = [G.NAMES[i] for i in np.where(free & ((np.abs(best.x - lo) < 1e-6) | (np.abs(best.x - hi) < 1e-6)))[0]]
        rows.append(dict(dataset="gs", unit=unit, model=name, k=int(free.sum()), LL=-best.fun, n=int(valid.sum()),
                         at_bound=";".join(at_bound), **dict(zip(G.NAMES, best.x))))
    return rows


def main():
    units = GS.load()
    with mp.get_context("spawn").Pool(int(FF.option("workers", "8"))) as pool:
        rows = sum(pool.map(run, list(units.items())), [])
    out = SP.GLASAUER_SHI / "three_state_fits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
