"""Fresh v3 parameter recovery on real sequences and independently sampled parameters.

Prepare: python scripts/model/parameter_recovery_v3.py --prepare
Run:     python scripts/model/parameter_recovery_v3.py --task=0 --ntasks=64
Requires the primary v3 environment; see the saved protocol and Slurm runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF

sys.path.insert(0, str(SP.ROOT / "scripts/model"))
import fit_minimal_space as M
from recovery_v3_common import fit_theta, fitted_theta, seed

OUT = SP.RESULTS / "model/minimal_qdiag/recovery_v3_restart_20261113"
SELECTED = {2: "C3-R-S0-B2", 1: "C3-R-S3-B2"}
# Stored names are historical. These are the quantities actually optimized under v3 bounds.
PARAMETERS = {
    "lq1": (r"$\sigma_{q_1}$ (s)", "Fast innovation SD", "s"),
    "lq2": (r"$\sigma_{q_2}$ (s)", "Slow innovation SD", "s"),
    "beta": (r"$\phi^L$", "Low readout continuity", ""),
    "wuH": (r"$\phi^H$", "High readout continuity", ""),
    "lam": (r"$\lambda$", "Bias persistence", ""),
    "lq3": (r"$\sigma_{q_3}$ (s)", "Baseline bias innovation SD", "s"),
    "d0": (r"$d_0$ (s)", "Baseline response offset", "s"),
    "ad0": (r"$\Delta d_0$ (s)", "Low minus High response offset", "s"),
    "lsde": (r"$\sigma_e$ (s)", "Response noise SD", "s"),
    "aq3": (r"$\rho_{q_3}$", "Low/High bias variance ratio", "ratio"),
}
ENV = {
    "GKF_SCALE": "lin", "GKF_LIK": "gh", "GKF_CORE": "reduced", "GKF_BOUNDS": "v3",
    "GKF_INIT": "gs", "GKF_NOISELESS": "1", "GKF_QDIAG": "1", "GKF_READOUT": "1",
}
SOURCES = (
    "src/sdu/model/engine.py", "src/sdu/model/fitting.py", "src/sdu/paths.py",
    "scripts/model/fit_minimal_space.py", "scripts/model/recovery_v3_common.py",
    "scripts/model/parameter_recovery_v3.py", "data/processed/model_input/input_full_sequence.csv",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_configuration():
    actual = (G.SCALE, G.LIK, G.CORE, G.BOUNDS, G.INIT, G.NOISELESS, G.QDIAG, G.READOUT)
    expected = ("lin", "gh", "reduced", "v3", "gs", True, True, True)
    if actual != expected or G.EFF_GAMMA or G.NOISE_TREND:
        raise ValueError(f"Use the primary v3 environment: {ENV}; got {actual}")
    for label in SELECTED.values():
        if set(dict(M.space())[label].free()) != set(PARAMETERS):
            raise ValueError(f"Free parameters have changed: {label}")


def prepare(out=OUT, local_reps=20, range_n=200, master_seed=20261113, starts=12):
    check_configuration()
    out = Path(out)
    if (out / "design.json").exists():
        raise FileExistsError(f"Design already frozen: {out}; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    rows, ranges = [], []
    for exp, label in SELECTED.items():
        data = FF.load(exp)
        participants = sorted(data)
        spec = dict(M.space())[label]
        lower, upper = spec.bounds()
        true = np.array([fitted_theta(exp, sub, label) for sub in participants])
        if np.any(true < lower - 1e-10) or np.any(true > upper + 1e-10):
            raise ValueError(f"Generating fit outside current bounds: {label}")
        for j, sub in enumerate(participants):
            for rep in range(local_reps):
                rows.append(dict(design="fitted", exp=exp, Sub=sub, rep=rep, model=label,
                                 **dict(zip(G.NAMES, true[j]))))
        sample = qmc.LatinHypercube(d=len(PARAMETERS), seed=master_seed + exp).random(range_n)
        for j, name in enumerate(PARAMETERS):
            index = G.IX[name]
            # Full continuity range; all other ranges span the observed participant fits.
            lo, hi = (0.0, 1.0) if name in ("beta", "wuH") else (true[:, index].min(), true[:, index].max())
            if hi <= lo:
                raise ValueError(f"No fitted range for {name}")
            scale = "log" if name == "aq3" else "linear"
            sample[:, j] = (np.exp(np.log(lo) + sample[:, j] * np.log(hi / lo)) if scale == "log"
                            else lo + sample[:, j] * (hi - lo))
            ranges.append(dict(exp=exp, parameter=name, lower=lo, upper=hi, sampling=scale))
        # Balanced assignment to the observed sequences, independent of sampled parameter values.
        order = np.random.default_rng(master_seed + exp + 100).permutation(range_n)
        for j in range(range_n):
            th = np.clip(G.X0, lower, upper)
            th[[G.IX[n] for n in PARAMETERS]] = sample[j]
            sub = participants[order[j] % len(participants)]
            rows.append(dict(design="range", exp=exp, Sub=sub, rep=0, model=label,
                             **dict(zip(G.NAMES, th))))
    table = pd.DataFrame(rows)
    table.insert(0, "unit", np.arange(len(table)))
    table["simulation_seed"] = [seed(f"v3-fresh-{master_seed}|simulation|{u}") for u in table.unit]
    table["fit_seed"] = [seed(f"v3-fresh-{master_seed}|fit|{u}") for u in table.unit]
    table.to_csv(out / "design.csv", index=False)
    pd.DataFrame(ranges).to_csv(out / "sampling_ranges.csv", index=False)
    metadata = dict(
        seed=master_seed, local_reps=local_reps, range_n=range_n, starts=starts,
        total_starts=starts + 1, units=len(table), environment=ENV,
        parameters={n: dict(label=v[0], meaning=v[1], unit=v[2]) for n, v in PARAMETERS.items()},
        sources={name: digest(SP.ROOT / name) for name in SOURCES},
        design_sha256=digest(out / "design.csv"),
        sequence="Full observed sequence and response mask, identical to the main fits",
        correlations="Across 22 participants within each fitted-design repetition; across 200 range draws",
    )
    (out / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(table)} fresh data sets in {out}")
    return table


def verify_design(out):
    metadata = json.loads((out / "design.json").read_text())
    if digest(out / "design.csv") != metadata["design_sha256"]:
        raise ValueError("Recovery design was modified")
    for name, expected in metadata["sources"].items():
        if digest(SP.ROOT / name) != expected:
            raise ValueError(f"Source differs from the frozen recovery design: {name}")
    return metadata


def run_one(row, functions, arrays, starts):
    spec = dict(M.space())[row.model]
    theta = row[G.NAMES].to_numpy(float)
    log_t, coh, tau, _, mask = arrays
    rng = np.random.default_rng(int(row.simulation_seed))
    response = G.simulate(theta, spec, np.exp(log_t), coh, tau, mask, rng, return_mean=False)[0]
    values = (log_t, coh, tau, np.nan_to_num(response), mask)
    t0 = time.perf_counter()
    best, n_best, n_finite = fit_theta(
        functions, spec, values, np.random.default_rng(int(row.fit_seed)), starts,
    )
    true_ll = float(G.loglik(theta, spec, *values).sum())
    numpy_ll = float(G.loglik(best.x, spec, *values).sum())
    if abs(numpy_ll + best.fun) > 1e-5:
        raise ValueError("NumPy and JAX likelihoods disagree at the recovered fit")
    result = {name: row[name] for name in ("unit", "design", "exp", "Sub", "rep", "model")}
    result.update(
        simulation_seed=int(row.simulation_seed), fit_seed=int(row.fit_seed),
        n_trials=len(log_t), n_valid=int(mask.sum()), LL=-float(best.fun), LL_true=true_ll,
        LL_minus_true=-float(best.fun) - true_ll, success=bool(best.success),
        status=int(best.status), message=str(best.message), iterations=int(best.nit),
        n_starts=n_finite, n_converged=best.n_converged, n_at_best=n_best,
        elapsed_seconds=time.perf_counter() - t0,
        response_sha256=hashlib.sha256(np.nan_to_num(response).astype("<f8").tobytes()).hexdigest(),
        python=platform.python_version(),
        **dict(zip(G.NAMES, best.x)), **{f"true_{n}": v for n, v in zip(G.NAMES, theta)},
    )
    return result, best.start_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--local-reps", type=int, default=20)
    parser.add_argument("--range-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Pilot: maximum assigned units")
    args = parser.parse_args()
    if min(args.local_reps, args.range_n, args.starts, args.ntasks) < 1 or not 0 <= args.task < args.ntasks:
        parser.error("Counts must be positive and task must be in [0, ntasks)")
    if args.prepare:
        prepare(args.out, args.local_reps, args.range_n, args.seed, args.starts)
        return
    check_configuration()
    metadata = verify_design(args.out)
    table = pd.read_csv(args.out / "design.csv")
    assigned = table.iloc[args.task::args.ntasks]
    if args.limit:
        assigned = assigned.head(args.limit)
    data = {exp: FF.load(exp) for exp in SELECTED}
    functions = G.jax_funcs()
    raw = args.out / "raw"
    raw.mkdir(exist_ok=True)
    for _, row in assigned.iterrows():
        path = raw / f"unit_{int(row.unit):04d}.csv"
        if path.exists():
            old = pd.read_csv(path)
            if len(old) != 1 or old.iloc[0].design_sha256 != metadata["design_sha256"]:
                raise ValueError(f"Unexpected existing result: {path}")
            continue
        result, starts = run_one(row, functions, data[row.exp][row.Sub], metadata["starts"])
        result["design_sha256"] = metadata["design_sha256"]
        path.with_suffix(".json").write_text(json.dumps(starts, indent=2) + "\n")
        tmp = path.with_suffix(".tmp")
        pd.DataFrame([result]).to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"unit {row.unit}: {row.design} exp{row.exp} Sub={row.Sub}, "
              f"{result['elapsed_seconds']:.1f}s, converged={result['success']}, "
              f"LL gain={result['LL_minus_true']:.3f}", flush=True)


if __name__ == "__main__":
    main()
