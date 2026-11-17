"""Fits to the randomized sequences for the transfer test of Appendix C (three-state B0 and B2, 24 random starts).

The random-walk responses are not used here; they are reserved for compare_transfer.py.
Writes the fits to the given file (results/model/glasauer_shi/transfer/fits.csv).

Usage: python scripts/model/glasauer_shi/fit_transfer.py results/model/glasauer_shi/transfer/fits.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import glasauer_shi as GS

SEED_KEY = "gs-transfer|{participant}"


def fit_participant(job):
    from scipy.optimize import minimize

    participant, (duration, response, valid), starts = job
    functions = G.jax_funcs()
    jnp = functions["jnp"]
    arrays = [jnp.asarray(v) for v in GS.three_state_arrays(duration, response, valid)]
    rng = np.random.default_rng(FF.seed(SEED_KEY.format(participant=participant)))
    rows = []
    for name in GS.TRANSFER:
        model = GS.THREE_STATE[name]
        lower, upper = GS.three_state_bounds(model)
        code = jnp.asarray(model.code())

        def objective(theta):
            value, gradient = functions["vg"](jnp.asarray(theta), code, *arrays)
            return float(value), np.asarray(gradient, float)

        initial = [np.clip(G.X0, lower, upper)] + [FF.random_point(lower, upper, rng) for _ in range(starts - 1)]
        zero = np.clip(G.X0, lower, upper)
        zero[G.IX["phi_H"]] = 0.0                     # the nested model without retention
        initial.append(zero)
        best = None
        for start in initial:
            result = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                              options={"maxiter": 4000})
            if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
                best = result
        if best is None:
            raise RuntimeError(f"no finite fit: participant {participant}, model {name}")
        free = lower < upper
        at_bound = [G.NAMES[i] for i in np.where(free & ((np.abs(best.x - lower) < 1e-6)
                                                          | (np.abs(best.x - upper) < 1e-6)))[0]]
        rows.append({"dataset": "gs", "unit": f"{participant}|randomized", "model": name, "k": int(free.sum()),
                     "LL": -best.fun, "n": int(valid.sum()), "starts": starts, "at_bound": ";".join(at_bound),
                     **dict(zip(G.NAMES, best.x))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    units = GS.load()
    jobs = [(p, units[f"{p}|randomized"], args.starts) for p in GS.participants(units)]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        rows = sum(pool.map(fit_participant, jobs), [])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["model", "unit"])
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} fits)")


if __name__ == "__main__":
    main()
