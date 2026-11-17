"""Parameter recovery of the selected models on the observed sequences (Appendix B.5).

Design ("fitted"): each participant's fitted parameters generate 20 new response data sets on the participant's own
sequence and validity mask; every data set is refitted with the generating model (13 starts: the default point, 11
random points and the zero-retention point). Design ("range"): 200 further parameter vectors per experiment, sampled
by Latin hypercube over the range of the participant fits (retention over 0-1), each on one observed sequence.

Prepare: python scripts/model/parameter_recovery.py --prepare      (writes design.csv and design.json)
Run:     python scripts/model/parameter_recovery.py --task=0 --ntasks=64
The stored run is in results/model/parameter_recovery (PROTOCOL.md); the design is frozen there.
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
from recovery_utils import SELECTED, fit_theta, fitted_theta, seed  # noqa: E402

OUT = SP.PARAMETER_RECOVERY
PARAMETERS = {
    "sd_q1": (r"$\sigma_{q_1}$ (s)", "Fast innovation SD", "s"),
    "sd_q2": (r"$\sigma_{q_2}$ (s)", "Slow innovation SD", "s"),
    "phi_L": (r"$\phi^L$", "Low readout continuity", ""),
    "phi_H": (r"$\phi^H$", "High readout continuity", ""),
    "lambda": (r"$\lambda$", "Bias persistence", ""),
    "sd_q3": (r"$\sigma_{q_3}$ (s)", "Baseline bias innovation SD", "s"),
    "d0": (r"$d_0$ (s)", "Baseline response offset", "s"),
    "d0_LH": (r"$\Delta d_0$ (s)", "Low minus High response offset", "s"),
    "sigma_e": (r"$\sigma_e$ (s)", "Response noise SD", "s"),
    "ratio_q3": (r"$\rho_{q_3}$", "Low/High bias variance ratio", "ratio"),
}
# Seeds of the stored design: the Latin-hypercube draw and the assignment to sequences use master_seed + offset,
# and every unit's simulation and fit seeds come from these text keys.
RANGE_SEED_OFFSET = {1: 2, 2: 1}
SEED_KEY = "v3-fresh-{master}|{kind}|{unit}"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_configuration():
    for label in SELECTED.values():
        if set(G.Model.parse(label).free()) != set(PARAMETERS):
            raise ValueError(f"Free parameters have changed: {label}")


def prepare(out=OUT, local_reps=20, range_n=200, master_seed=20261113, starts=12):
    check_configuration()
    out = Path(out)
    if (out / "design.json").exists():
        raise FileExistsError(f"Design already frozen: {out}; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    fits = FF.read_fits()
    rows, ranges = [], []
    for exp, label in SELECTED.items():
        participants = sorted(FF.load(exp))
        model = G.Model.parse(label)
        lower, upper = model.bounds()
        true = np.array([fitted_theta(fits, exp, sub, label) for sub in participants])
        if np.any(true < lower - 1e-10) or np.any(true > upper + 1e-10):
            raise ValueError(f"Generating fit outside the bounds: {label}")
        for j, sub in enumerate(participants):
            for rep in range(local_reps):
                rows.append(dict(design="fitted", exp=exp, Sub=sub, rep=rep, model=label,
                                 **dict(zip(G.NAMES, true[j]))))
        offset = RANGE_SEED_OFFSET[exp]
        sample = qmc.LatinHypercube(d=len(PARAMETERS), seed=master_seed + offset).random(range_n)
        for j, name in enumerate(PARAMETERS):
            index = G.IX[name]
            # retention over its full range; every other parameter over the range of the participant fits
            lo, hi = (0.0, 1.0) if name in ("phi_L", "phi_H") else (true[:, index].min(), true[:, index].max())
            if hi <= lo:
                raise ValueError(f"No fitted range for {name}")
            scale = "log" if name == "ratio_q3" else "linear"
            sample[:, j] = (np.exp(np.log(lo) + sample[:, j] * np.log(hi / lo)) if scale == "log"
                            else lo + sample[:, j] * (hi - lo))
            ranges.append(dict(exp=exp, parameter=name, lower=lo, upper=hi, sampling=scale))
        # balanced assignment to the observed sequences, independent of the sampled parameter values
        order = np.random.default_rng(master_seed + offset + 100).permutation(range_n)
        for j in range(range_n):
            th = np.clip(G.X0, lower, upper)
            th[[G.IX[n] for n in PARAMETERS]] = sample[j]
            sub = participants[order[j] % len(participants)]
            rows.append(dict(design="range", exp=exp, Sub=sub, rep=0, model=label, **dict(zip(G.NAMES, th))))
    table = pd.DataFrame(rows)
    table.insert(0, "unit", np.arange(len(table)))
    for kind in ("simulation", "fit"):
        table[f"{kind}_seed"] = [seed(SEED_KEY.format(master=master_seed, kind=kind, unit=u)) for u in table.unit]
    table.to_csv(out / "design.csv", index=False)
    pd.DataFrame(ranges).to_csv(out / "sampling_ranges.csv", index=False)
    metadata = dict(
        seed=master_seed, local_reps=local_reps, range_n=range_n, starts=starts,
        total_starts=starts + 1, units=len(table),
        parameters={n: dict(label=v[0], meaning=v[1], unit=v[2]) for n, v in PARAMETERS.items()},
        design_sha256=digest(out / "design.csv"),
        sequence="Full observed sequence and response mask, identical to the main fits",
        correlations="Across 22 participants within each fitted-design repetition; across 200 range draws",
    )
    (out / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(table)} data sets in {out}")
    return table


def verify_design(out):
    metadata = json.loads((out / "design.json").read_text())
    if digest(out / "design.csv") != metadata["design_sha256"]:
        raise ValueError("Recovery design was modified")
    return metadata


def run_one(row, functions, arrays, starts):
    model = G.Model.parse(row.model)
    theta = FF.theta(row)
    T, coh, _, mask = arrays
    rng = np.random.default_rng(int(row.simulation_seed))
    response = G.simulate(theta, model, T, coh, mask, rng, return_mean=False)[0]
    values = (T, coh, np.nan_to_num(response), mask)
    t0 = time.perf_counter()
    best, n_best, n_finite = fit_theta(functions, model, values, np.random.default_rng(int(row.fit_seed)), starts)
    true_ll = float(G.loglik(theta, model, *values).sum())
    numpy_ll = float(G.loglik(best.x, model, *values).sum())
    if abs(numpy_ll + best.fun) > 1e-5:
        raise ValueError("NumPy and JAX likelihoods disagree at the recovered fit")
    result = {name: row[name] for name in ("unit", "design", "exp", "Sub", "rep", "model")}
    result.update(
        simulation_seed=int(row.simulation_seed), fit_seed=int(row.fit_seed),
        n_trials=len(T), n_valid=int(mask.sum()), LL=-float(best.fun), LL_true=true_ll,
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
    selected = args.out / "fit_records.csv"
    initial = args.out / "fit_records_initial.csv"
    consolidated = selected if selected.exists() else initial
    stored = pd.read_csv(consolidated) if consolidated.exists() else pd.DataFrame()
    if not stored.empty and (stored.unit.duplicated().any()
                             or not stored.design_sha256.eq(metadata["design_sha256"]).all()):
        raise ValueError(f"Invalid consolidated results: {consolidated}")
    stored_units = set(stored.unit) if not stored.empty else set()
    for _, row in assigned.iterrows():
        if row.unit in stored_units:
            continue
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
        print(f"unit {row.unit}: {row.design} {FF.EXPERIMENT[row.exp]} Sub={row.Sub}, "
              f"{result['elapsed_seconds']:.1f}s, converged={result['success']}, "
              f"LL gain={result['LL_minus_true']:.3f}", flush=True)


if __name__ == "__main__":
    main()
