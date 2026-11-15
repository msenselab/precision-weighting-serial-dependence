"""Shared fitting utilities for the targeted 3SKF v3 recovery analyses."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G


ROOT = SP.RESULTS / "model" / "minimal_qdiag"
FITS_DIR = ROOT / "readout_space"


def fitted_theta(exp: int, sub: int, label: str) -> np.ndarray:
    """Load one participant/model MLE from the chunked main-fit files."""
    rows = [
        frame[frame.model == label]
        for frame in (pd.read_csv(path) for path in sorted(FITS_DIR.glob(f"exp{exp}_{sub}_c*.csv")))
    ]
    rows = [row for row in rows if len(row)]
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError(f"expected one main-fit row for exp={exp}, Sub={sub}, model={label}")
    return rows[0][G.NAMES].iloc[0].to_numpy(float)


def seed(text: str) -> int:
    return zlib.crc32(text.encode())


def fit_theta(functions, specification, arrays, rng, n_starts: int):
    """Fit one model with the main analysis's default, random, and zero-persistence starts."""
    from scipy.optimize import minimize

    jnp = functions["jnp"]
    code = jnp.asarray(specification.vec())
    values = [jnp.asarray(value) for value in arrays]
    lower, upper = specification.bounds()

    def objective(theta):
        value, gradient = functions["vg"](jnp.asarray(theta), code, *values)
        return float(value), np.asarray(gradient, float)

    starts = [np.clip(G.X0, lower, upper)]
    starts.extend(
        np.where(lower == upper, lower, lower + (upper - lower) * rng.uniform(0.1, 0.9, len(lower)))
        for _ in range(n_starts - 1)
    )
    zero = np.clip(G.X0, lower, upper)
    for name in ("wuH", "beta"):
        index = G.IX[name]
        if lower[index] < upper[index]:
            zero[index] = 0.0
    starts.append(zero)

    best = None
    log_likelihoods = []
    diagnostics = []
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": 4000},
        )
        if np.isfinite(result.fun):
            log_likelihoods.append(-result.fun)
            diagnostics.append({
                "nll": float(result.fun), "success": bool(result.success),
                "status": int(result.status), "iterations": int(result.nit),
                "message": str(result.message),
            })
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError("all recovery starts returned a non-finite objective")

    best["start_diagnostics"] = diagnostics
    best["n_converged"] = sum(item["success"] for item in diagnostics)
    log_likelihoods = np.asarray(log_likelihoods)
    return best, int(np.sum(np.abs(log_likelihoods + best.fun) < 0.01)), len(log_likelihoods)
