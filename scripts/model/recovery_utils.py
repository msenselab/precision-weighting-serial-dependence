"""Shared utilities for the recovery analyses (Appendix B.5-B.6)."""
from __future__ import annotations

import zlib

import numpy as np

from sdu.model import engine as G
from sdu.model import fitting as FF

SELECTED = {1: "C3-S0-B2", 2: "C3-S3-B2"}           # the models reported for Experiment 1 and Experiment 2
SEED_CODE = FF.SEED_CODE                            # experiment codes in the seed keys of the stored runs


def fitted_theta(fits, exp, sub, label):
    """One participant's maximum-likelihood estimate from the stored fits."""
    row = fits[(fits.exp == exp) & (fits.Sub == sub) & (fits.model == label)]
    if len(row) != 1:
        raise ValueError(f"expected one fit for exp={exp}, Sub={sub}, model={label}")
    return FF.theta(row.iloc[0])


def seed(text: str) -> int:
    return zlib.crc32(text.encode())


def fit_theta(functions, model, arrays, rng, n_starts: int):
    """Fit one model from the default point, n_starts - 1 random points and the zero-retention point, recording
    every start. Returns (best result with start_diagnostics and n_converged, starts at the optimum, finite starts)."""
    best, results = FF.fit(functions, model, arrays, rng, n_random=n_starts - 1, return_all=True)
    if best is None:
        raise RuntimeError("all recovery starts returned a non-finite objective")
    diagnostics = [{"nll": float(r.fun), "success": bool(r.success), "status": int(r.status),
                    "iterations": int(r.nit), "message": str(r.message)} for r in results]
    best["start_diagnostics"] = diagnostics
    best["n_converged"] = sum(item["success"] for item in diagnostics)
    nll = np.array([r.fun for r in results])
    return best, int(np.sum(np.abs(nll - best.fun) < 0.01)), len(results)
