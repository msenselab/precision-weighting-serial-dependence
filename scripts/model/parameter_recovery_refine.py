"""Parameter recovery: refit the data sets whose fit fell below the generating likelihood or did not converge.

For an active run, original fits stay in raw/ and additional-start fits go in refined/. For the archived public
results, improvements are stored in fit_records_refined.csv and replace the selected row in fit_records.csv.
The simulated responses remain identical.
Run after the initial array completes (python scripts/model/parameter_recovery_refine.py). No generating value is
supplied as an optimizer start.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sdu.model import engine as G
from sdu.model import fitting as FF
from parameter_recovery import OUT, check_configuration, digest, run_one, verify_design
from recovery_utils import SELECTED, seed

REFINE_KEY = "v3-restart-refine|{master}|{unit}"        # text key of the refinement fit seeds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--starts", type=int, default=48)
    args = parser.parse_args()
    check_configuration()
    metadata = verify_design(args.out)
    consolidated = args.out / "fit_records.csv"
    paths = sorted((args.out / "raw").glob("unit_*.csv"))
    if consolidated.exists():
        raw = pd.read_csv(consolidated)
    elif paths:
        raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    else:
        raise FileNotFoundError("No parameter-recovery fits found")
    failed = raw[(raw.LL_minus_true < -0.01) | ~raw.success]
    design = pd.read_csv(args.out / "design.csv").set_index("unit", drop=False)
    target = args.out / "refined"
    if not consolidated.exists():
        target.mkdir(exist_ok=True)
    functions = G.jax_funcs()
    data = {e: FF.load(e) for e in SELECTED}
    consolidated_logs = []
    refined_results = []
    for _, original in failed.iterrows():
        path = target / f"unit_{int(original.unit):04d}.csv"
        if not consolidated.exists() and path.exists():
            continue
        row = design.loc[original.unit].copy()
        row.fit_seed = seed(REFINE_KEY.format(master=metadata["seed"], unit=row.unit))
        result, starts = run_one(row, functions, data[row.exp][row.Sub], args.starts)
        if result["response_sha256"] != original.response_sha256:
            raise ValueError(f"Simulated responses changed: unit {row.unit}")
        if result["LL"] > original.LL:
            initial_starts = [] if consolidated.exists() else json.loads(
                (args.out / "raw" / path.with_suffix(".json").name).read_text())
            retained_initial = int(original.n_at_best) if result["LL"] - original.LL < 0.01 else 0
            result.update(design_sha256=metadata["design_sha256"], initial_LL=original.LL,
                          initial_fit_seed=original.fit_seed, refinement_starts=args.starts + 1,
                          n_starts=int(original.n_starts) + len(starts),
                          n_converged=int(original.n_converged) + sum(s["success"] for s in starts),
                          n_at_best=retained_initial + sum(abs(s["nll"] + result["LL"]) < 0.01 for s in starts),
                          refinement_source_sha256=digest(__file__))
            if consolidated.exists():
                for key in result:
                    if key not in raw:
                        raw[key] = np.nan
                    raw.loc[raw.unit == original.unit, key] = result[key]
                consolidated_logs.append(json.dumps({"unit": int(original.unit), "starts": starts}))
                refined_results.append(result.copy())
            else:
                path.with_suffix(".json").write_text(json.dumps(starts, indent=2) + "\n")
                tmp = path.with_suffix(".tmp")
                pd.DataFrame([result]).to_csv(tmp, index=False)
                tmp.replace(path)
        print(f"unit {row.unit}: initial gap={original.LL_minus_true:.3f}, "
              f"retry gap={result['LL_minus_true']:.3f}", flush=True)
    if consolidated_logs:
        raw.to_csv(consolidated, index=False)
        archive = args.out / "fit_records_refined.csv"
        previous = pd.read_csv(archive) if archive.exists() else pd.DataFrame()
        additions = pd.DataFrame(refined_results)
        if not previous.empty:
            previous = previous[~previous.unit.isin(additions.unit)]
        pd.concat([previous, additions], ignore_index=True).to_csv(archive, index=False)
        (args.out / "refinement_logs.jsonl").write_text("\n".join(consolidated_logs) + "\n")
    print(f"Checked {len(failed)} fits with {args.starts + 1} additional starts each", flush=True)


if __name__ == "__main__":
    main()
