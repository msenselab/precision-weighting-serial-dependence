"""High-start randomized-condition fits for the Glasauer-Shi transfer comparison.

Only two matched 3SKF models are fitted:

* R-B0: 3SKF v3 without the bias state;
* R-B2: 3SKF v3 with the corrected response-bias state (C3-S0-B2 of our experiments reduces to it here).

Requires GKF_READOUT=1.

The random-walk responses are never used here.  They are reserved for evaluation by
``compare_transfer_gs.py``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G

sys.path.insert(0, str(SP.ROOT / "scripts" / "model" / "external"))
import fit_external as external  # noqa: E402


assert G.READOUT, "3SKF v3 only: set GKF_READOUT=1"
MODEL_KEYS = ("R-B0", "R-B2")


def fit_participant(job):
    from scipy.optimize import minimize

    participant, sequence, starts = job
    duration, response, valid = sequence
    trial = np.arange(len(duration), dtype=float)
    arrays = (
        np.log(duration),
        np.ones(len(duration)),
        2 * trial / trial.max() - 1,
        np.where(valid, response, 0.0),
        valid,
    )
    functions = G.jax_funcs()
    jnp = functions["jnp"]
    jax_arrays = [jnp.asarray(value) for value in arrays]
    rng = np.random.default_rng(zlib.crc32(f"gs-transfer|{participant}".encode()))
    rows = []

    for model in MODEL_KEYS:
        specification = external.MODELS[model]
        lower, upper = specification.bounds()
        for name, value in external.FIX.items():
            lower[G.IX[name]] = upper[G.IX[name]] = value
        code = jnp.asarray(specification.vec())

        def objective(theta):
            value, gradient = functions["vg"](jnp.asarray(theta), code, *jax_arrays)
            return float(value), np.asarray(gradient, float)

        initial = [np.clip(G.X0, lower, upper)]
        initial.extend(
            np.where(
                lower == upper,
                lower,
                lower + (upper - lower) * rng.uniform(0.1, 0.9, len(lower)),
            )
            for _ in range(starts - 1)
        )
        start0 = np.clip(G.X0, lower, upper)            # v3 nests phi = 0: make sure that point is reached
        start0[G.IX["wuH"]] = 0.0
        initial.append(start0)
        best = None
        for start in initial:
            result = minimize(
                objective,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": 4000},
            )
            if np.isfinite(result.fun) and (best is None or result.fun < best.fun):
                best = result
        if best is None:
            raise RuntimeError(f"no finite fit: participant {participant}, model {model}")

        free = lower < upper
        at_bound = [
            G.NAMES[index]
            for index in np.where(
                free
                & (
                    (np.abs(best.x - lower) < 1e-6)
                    | (np.abs(best.x - upper) < 1e-6)
                )
            )[0]
        ]
        rows.append(
            {
                "dataset": "gs",
                "unit": f"{participant}|randomized",
                "model": model,
                "k": int(free.sum()),
                "LL": -best.fun,
                "n": int(valid.sum()),
                "starts": starts,
                "at_bound": ";".join(at_bound),
                **dict(zip(G.NAMES, best.x)),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--participants", type=int, default=14)
    args = parser.parse_args()

    _, loaded = external.load("gs")
    units = dict(loaded)
    participants = sorted(
        {int(unit.split("|")[0]) for unit in units},
    )[: args.participants]
    jobs = [
        (participant, units[f"{participant}|randomized"][0], args.starts)
        for participant in participants
    ]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        rows = sum(pool.map(fit_participant, jobs), [])

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["model", "unit"])
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} fits)")


if __name__ == "__main__":
    main()
