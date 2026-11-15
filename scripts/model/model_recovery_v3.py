"""Targeted model recovery for the two mechanistic claims of 3SKF v3.

``phi`` (Experiment 1)
    Generate from and fit C3-S0-B2 with either one shared persistence or separate Low/High persistence.
``switch`` (Experiment 2)
    Generate from and fit C3-B2 with either S0 or the S3 readout reset.

Each participant/repetition data set is fitted with both candidates using the main analysis's 13 starts. Ten fresh
simulations at the fitted parameters estimate the focal behavioral contrast. The companion summary script classifies
each 22-participant cohort by mean AIC and by the same |z| < 1.5 descriptive focal-effect band as the main screen.

Usage: python scripts/model/model_recovery_v3.py --task=<i> --ntasks=<n> [--reps=20] [--starts=12]
       [--nsim=10] [--analyses=phi,switch] [--subs=<n>]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import stats as S

sys.path.insert(0, str(SP.ROOT / "scripts" / "model"))
import fit_minimal_space as M  # noqa: E402
from recovery_v3_common import ROOT, fit_theta, fitted_theta, seed  # noqa: E402


N_REP = int(FF._opt("reps", "20"))
N_STARTS = int(FF._opt("starts", "12"))
N_SIM = int(FF._opt("nsim", "10"))
REQUESTED = tuple(filter(None, FF._opt("analyses", "phi,switch").split(",")))
SPACE = dict(M.space())

ANALYSES = {
    "phi": {
        "exp": 2,
        "focal": "low_high",
        "candidates": {
            "shared": G.Spec(3, "T0", 0, 3),
            "specific": SPACE["C3-R-S0-B2"],
        },
    },
    "switch": {
        "exp": 1,
        "focal": "LL_other_three",
        "candidates": {
            "S0": SPACE["C3-R-S0-B2"],
            "S3": SPACE["C3-R-S3-B2"],
        },
    },
}


def output_directory():
    return ROOT / "model_recovery_v3" / f"r{N_REP:03d}_s{N_STARTS:02d}_n{N_SIM:02d}"


def units(n_subs=None):
    unknown = set(REQUESTED) - set(ANALYSES)
    if unknown:
        raise ValueError(f"unknown analyses: {sorted(unknown)}")
    out = []
    for analysis in REQUESTED:
        exp = ANALYSES[analysis]["exp"]
        participants = sorted(FF.load(exp))
        out.extend((analysis, exp, sub) for sub in (participants[:n_subs] if n_subs else participants))
    return out


def observed_fit(analysis, candidate, exp, sub, specification, functions, arrays):
    """Generator MLE: main-space fit when available, otherwise the shared-phi fit to the observed data."""
    labels = {
        ("phi", "specific"): "C3-R-S0-B2",
        ("switch", "S0"): "C3-R-S0-B2",
        ("switch", "S3"): "C3-R-S3-B2",
    }
    if (analysis, candidate) in labels:
        return fitted_theta(exp, sub, labels[(analysis, candidate)])
    rng = np.random.default_rng(seed(f"recv3-observed|{analysis}|{candidate}|{exp}|{sub}"))
    best, _, _ = fit_theta(functions, specification, arrays, rng, N_STARTS)
    return best.x


def focal_stat(exp, sub, response, name):
    return S.participant_stats_arrays(exp, sub, response)[name]


def run_unit(analysis, exp, sub, functions, participant_data):
    definition = ANALYSES[analysis]
    candidates = definition["candidates"]
    focal = definition["focal"]
    log_duration, coherence, trial, response, valid = participant_data
    observed_arrays = (log_duration, coherence, trial, response, valid)
    generators = {
        name: observed_fit(analysis, name, exp, sub, specification, functions, observed_arrays)
        for name, specification in candidates.items()
    }
    rows = []
    for generator_name, generator_theta in generators.items():
        generator_specification = candidates[generator_name]
        for rep in range(N_REP):
            simulation_rng = np.random.default_rng(seed(f"recv3-model|{analysis}|{exp}|{sub}|{generator_name}|{rep}"))
            generated = G.simulate(
                generator_theta,
                generator_specification,
                np.exp(log_duration),
                coherence,
                trial,
                valid,
                simulation_rng,
                return_mean=False,
            )[0]
            observed_focal = focal_stat(exp, sub, generated, focal)
            fit_arrays = (log_duration, coherence, trial, np.nan_to_num(generated), valid)
            for candidate_name, candidate_specification in candidates.items():
                fit_rng = np.random.default_rng(
                    seed(f"recv3-fit|{analysis}|{exp}|{sub}|{generator_name}|{candidate_name}|{rep}")
                )
                best, n_best, n_finite = fit_theta(
                    functions, candidate_specification, fit_arrays, fit_rng, N_STARTS
                )
                simulated_focal = []
                check_rng = np.random.default_rng(
                    seed(f"recv3-check|{analysis}|{exp}|{sub}|{generator_name}|{candidate_name}|{rep}")
                )
                for _ in range(N_SIM):
                    simulated = G.simulate(
                        best.x,
                        candidate_specification,
                        np.exp(log_duration),
                        coherence,
                        trial,
                        valid,
                        check_rng,
                        return_mean=False,
                    )[0]
                    simulated_focal.append(focal_stat(exp, sub, simulated, focal))
                rows.append(
                    {
                        "analysis": analysis,
                        "exp": exp,
                        "Sub": sub,
                        "generator": generator_name,
                        "candidate": candidate_name,
                        "rep": rep,
                        "focal": focal,
                        "k": candidate_specification.k(),
                        "LL": -best.fun,
                        "AIC": 2 * candidate_specification.k() + 2 * best.fun,
                        "n_at_best": n_best,
                        "n_starts": n_finite,
                        "n_sim": N_SIM,
                        "obs_focal": observed_focal,
                        "sim_focal": np.mean(simulated_focal),
                        **dict(zip(G.NAMES, best.x)),
                        **{f"true_{name}": value for name, value in zip(G.NAMES, generator_theta)},
                    }
                )
    return pd.DataFrame(rows)


def validate_existing(path, analysis, exp, sub):
    frame = pd.read_csv(path)
    definition = ANALYSES[analysis]
    expected = 2 * 2 * N_REP
    if len(frame) != expected:
        raise ValueError(f"stale recovery file {path}: {len(frame)} rows, expected {expected}")
    if set(frame.generator) != set(definition["candidates"]) or set(frame.candidate) != set(definition["candidates"]):
        raise ValueError(f"stale candidate set in {path}")
    if frame.rep.nunique() != N_REP or not frame.n_sim.eq(N_SIM).all() or not frame.n_starts.eq(N_STARTS + 1).all():
        raise ValueError(f"stale recovery configuration in {path}")
    if frame.exp.nunique() != 1 or frame.exp.iloc[0] != exp or frame.Sub.nunique() != 1 or frame.Sub.iloc[0] != sub:
        raise ValueError(f"wrong recovery unit in {path}")


def main():
    assert G.LIN and G.NOISELESS and G.QDIAG and G.READOUT and G.CORE == "reduced"
    assert G.BOUNDS == "v3" and G.INIT == "gs"
    task = int(FF._opt("task", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    tasks = int(FF._opt("ntasks", "1"))
    n_subs = int(FF._opt("subs", "0")) or None
    directory = output_directory()
    directory.mkdir(parents=True, exist_ok=True)
    functions = G.jax_funcs()
    data = {exp: FF.load(exp) for exp in {definition["exp"] for definition in ANALYSES.values()}}
    for analysis, exp, sub in units(n_subs)[task::tasks]:
        path = directory / f"{analysis}_exp{exp}_{sub}.csv"
        if path.exists():
            validate_existing(path, analysis, exp, sub)
            continue
        started = time.time()
        frame = run_unit(analysis, exp, sub, functions, data[exp][sub])
        frame.to_csv(path.with_suffix(".tmp"), index=False)
        os.replace(path.with_suffix(".tmp"), path)
        print(f"done {analysis} exp{exp} {sub}: {len(frame)} fits, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
