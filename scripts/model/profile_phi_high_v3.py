"""Profile current-High readout continuity while reoptimizing the other nine v3 parameters.

Uses all 44 observed participant data sets. No new response simulations are generated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF

sys.path.insert(0, str(SP.ROOT / "scripts/model"))
import fit_minimal_space as M
from parameter_recovery_v3 import SOURCES, ENV, SELECTED, check_configuration, digest
from recovery_v3_common import fitted_theta, seed
from coherence_evidence_v3 import OUT as EVIDENCE

OUT = EVIDENCE / "profile_phi_high"
REFERENCE = 3.8414588207


def prepare(out=OUT):
    check_configuration()
    if (out / "design.json").exists():
        raise FileExistsError(f"Profile design already frozen: {out}")
    rows = []
    for exp, label in SELECTED.items():
        for sub in sorted(FF.load(exp)):
            theta = fitted_theta(exp, sub, label)
            rows.append(dict(unit=len(rows), exp=exp, Sub=sub, model=label, **dict(zip(G.NAMES, theta))))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "participants.csv", index=False)
    sources = list(SOURCES) + ["scripts/model/profile_phi_high_v3.py", "scripts/model/coherence_evidence_v3.py",
                              "src/sdu/behavior/trials.py", "scripts/behavior/current_coherence_mean_shift.py"]
    metadata = dict(environment=ENV, participants=len(rows), regular_grid=np.linspace(0, 1, 11).tolist(),
                    additional_grid="Each participant's original phi High MLE", reference_loss=REFERENCE,
                    forward_starts="original MLE, previous profile point, default, independent random point",
                    reverse_pass="one warm start from the neighboring profile point",
                    sources={p: digest(SP.ROOT / p) for p in sources},
                    participants_sha256=digest(out / "participants.csv"))
    (out / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared profiles for {len(rows)} participants", flush=True)


def fit_point(objective, lower, upper, starts):
    best = None
    logs = []
    for start in starts:
        result = minimize(objective, np.clip(start, lower, upper), jac=True, method="L-BFGS-B",
                          bounds=list(zip(lower, upper)), options={"maxiter": 4000})
        if np.isfinite(result.fun):
            logs.append(dict(LL=-float(result.fun), success=bool(result.success), iterations=int(result.nit)))
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError("No finite profile fit")
    return best, logs


def profile(row, functions, arrays, metadata):
    specification = dict(M.space())[row.model]
    lower, upper = specification.bounds()
    original = row[G.NAMES].to_numpy(float)
    jnp = functions["jnp"]
    code = jnp.asarray(specification.vec())
    values = [jnp.asarray(v) for v in arrays]
    def objective(theta):
        value, gradient = functions["vg"](jnp.asarray(theta), code, *values)
        return float(value), np.asarray(gradient, float)
    index = G.IX["wuH"]
    grid = np.unique(np.r_[metadata["regular_grid"], original[index]])
    original_ll = -objective(original)[0]
    rng = np.random.default_rng(seed(f"profile-phiH-v3|{row.exp}|{row.Sub}"))
    fits, logs = {}, {}
    warm = original.copy()
    for value in grid:
        lo, hi = lower.copy(), upper.copy()
        lo[index] = hi[index] = value
        random = lo + (hi - lo) * rng.uniform(0.1, 0.9, len(lo))
        best, trials = fit_point(objective, lo, hi, [original, warm, G.X0, random])
        fits[value], logs[value] = best, trials
        warm = best.x
    warm = fits[grid[-1]].x
    for value in grid[::-1]:
        lo, hi = lower.copy(), upper.copy()
        lo[index] = hi[index] = value
        best, trials = fit_point(objective, lo, hi, [warm])
        logs[value] += trials
        if best.fun < fits[value].fun:
            fits[value] = best
        warm = fits[value].x
    peak = min(fits.values(), key=lambda f: f.fun)
    full, full_logs = fit_point(objective, lower, upper, [peak.x, original, G.X0])
    best_ll = max(-full.fun, original_ll, -peak.fun)
    rows = []
    for value in grid:
        best = fits[value]
        rows.append(dict(exp=row.exp, Sub=row.Sub, model=row.model, phi_high=value,
                         LL=-float(best.fun), loss=2 * (best_ll + best.fun), original_LL=original_ll,
                         reference_LL=best_ll, success=bool(best.success),
                         **dict(zip(G.NAMES, best.x))))
    result = pd.DataFrame(rows)
    supported = result[result.loss <= REFERENCE].phi_high
    record = dict(exp=row.exp, Sub=row.Sub, model=row.model, original_phi_high=original[index],
                  refitted_phi_high=full.x[index], original_LL=original_ll, refitted_LL=-float(full.fun),
                  LL_improvement=-float(full.fun)-original_ll, grid_support_min=supported.min(),
                  grid_support_max=supported.max(), grid_support_width=supported.max()-supported.min(),
                  loss_at_zero=result.loc[result.phi_high == 0, "loss"].iloc[0],
                  loss_at_one=result.loc[result.phi_high == 1, "loss"].iloc[0],
                  profile_points=len(grid), full_fit_success=bool(full.success))
    return result, record, {"grid_starts": {str(k): v for k, v in logs.items()}, "full_starts": full_logs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--ntasks", type=int, default=1)
    args = parser.parse_args()
    check_configuration()
    if args.prepare:
        prepare(args.out)
        return
    metadata = json.loads((args.out / "design.json").read_text())
    if digest(args.out / "participants.csv") != metadata["participants_sha256"]:
        raise ValueError("Profile inputs changed")
    for name, expected in metadata["sources"].items():
        if digest(SP.ROOT / name) != expected:
            raise ValueError(f"Profile source changed: {name}")
    participants = pd.read_csv(args.out / "participants.csv")
    data = {e: FF.load(e) for e in SELECTED}
    functions = G.jax_funcs()
    directory = args.out / "curves"
    directory.mkdir(exist_ok=True)
    for _, row in participants.iloc[args.task::args.ntasks].iterrows():
        path = directory / f"exp{row.exp}_{row.Sub}.csv"
        if path.exists():
            continue
        curve, record, log = profile(row, functions, data[row.exp][row.Sub], metadata)
        pd.DataFrame([record]).to_csv(path.with_suffix(".summary.csv"), index=False)
        path.with_suffix(".json").write_text(json.dumps(log) + "\n")
        tmp = path.with_suffix(".tmp")
        curve.to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"exp{row.exp} Sub={row.Sub}: grid span={record['grid_support_width']:.2f}, "
              f"LL improvement={record['LL_improvement']:.4f}", flush=True)


if __name__ == "__main__":
    main()
