"""Retry recovery fits below the generating likelihood or without optimizer convergence.

Original fits stay in raw/. Additional-start fits go in refined/ and retain the same simulated responses.
Run after the initial array completes. No generating value is supplied as an optimizer start.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from sdu.model import engine as G
from sdu.model import fitting as FF
from parameter_recovery_v3 import OUT, check_configuration, digest, run_one, verify_design
from recovery_v3_common import seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--starts", type=int, default=48)
    args = parser.parse_args()
    check_configuration()
    metadata = verify_design(args.out)
    paths = sorted((args.out / "raw").glob("unit_*.csv"))
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    failed = raw[(raw.LL_minus_true < -0.01) | ~raw.success]
    design = pd.read_csv(args.out / "design.csv").set_index("unit", drop=False)
    target = args.out / "refined"
    target.mkdir(exist_ok=True)
    functions = G.jax_funcs()
    data = {e: FF.load(e) for e in (2, 1)}
    for _, original in failed.iterrows():
        path = target / f"unit_{int(original.unit):04d}.csv"
        if path.exists():
            continue
        row = design.loc[original.unit].copy()
        row.fit_seed = seed(f"v3-restart-refine|{metadata['seed']}|{row.unit}")
        result, starts = run_one(row, functions, data[row.exp][row.Sub], args.starts)
        if result["response_sha256"] != original.response_sha256:
            raise ValueError(f"Simulated responses changed: unit {row.unit}")
        if result["LL"] > original.LL:
            initial_path = args.out / "raw" / path.with_suffix(".json").name
            initial_starts = json.loads(initial_path.read_text())
            result.update(design_sha256=metadata["design_sha256"], initial_LL=original.LL,
                          initial_fit_seed=original.fit_seed, refinement_starts=args.starts + 1,
                          n_starts=int(original.n_starts) + len(starts),
                          n_converged=int(original.n_converged) + sum(s["success"] for s in starts),
                          n_at_best=sum(abs(s["nll"] + result["LL"]) < 0.01 for s in initial_starts + starts),
                          refinement_source_sha256=digest(__file__))
            path.with_suffix(".json").write_text(json.dumps(starts, indent=2) + "\n")
            tmp = path.with_suffix(".tmp")
            pd.DataFrame([result]).to_csv(tmp, index=False)
            tmp.replace(path)
        print(f"unit {row.unit}: initial gap={original.LL_minus_true:.3f}, "
              f"retry gap={result['LL_minus_true']:.3f}", flush=True)
    print(f"Checked {len(failed)} fits with {args.starts + 1} additional starts each", flush=True)


if __name__ == "__main__":
    main()
