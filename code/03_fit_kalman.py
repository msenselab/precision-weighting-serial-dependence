#!/usr/bin/env python3
"""Fit the full three-state Kalman model grid on the cleaned data.

The script checkpoints incrementally and writes canonical fit outputs to
`results/kalman_model_fits`. Full model fitting can take a long time.
"""

from __future__ import annotations

import hashlib
import argparse
import csv
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
sys.path.insert(0, str(MODEL_ROOT))

from three_state_kalman import fit_single_model, get_all_models  # noqa: E402
import three_state_kalman.parameters as P  # noqa: E402
from three_state_kalman.parameters import get_parameter_config  # noqa: E402


DATA_DIR = ROOT / "data"
OUTDIR = ROOT / "results" / "kalman_model_fits"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    1: DATA_DIR / "experiment1" / "E1.pkl",
    2: DATA_DIR / "experiment2" / "E2.pkl",
}

N_STARTS = 5
MAX_NFEV = 2000
SAVE_EVERY = 12


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alpha-q1-bound",
        type=float,
        default=5.0,
        help="Symmetric bound for alpha_q1; default matches the original +/-5 setting.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=OUTDIR,
        help="Output directory. Use a separate folder for sensitivity fits.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Default is 1 for exact serial behavior.",
    )
    return parser.parse_args()


def configure_run(args: argparse.Namespace) -> None:
    global OUTDIR
    OUTDIR = args.outdir
    if not OUTDIR.is_absolute():
        OUTDIR = ROOT / OUTDIR
    OUTDIR.mkdir(parents=True, exist_ok=True)

    P.C_MODULATION_PARAMS["Q1"]["lower"] = -args.alpha_q1_bound
    P.C_MODULATION_PARAMS["Q1"]["upper"] = args.alpha_q1_bound
    print(
        f"alpha_q1 bound set to [{P.C_MODULATION_PARAMS['Q1']['lower']}, "
        f"{P.C_MODULATION_PARAMS['Q1']['upper']}]",
        flush=True,
    )


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def load_kalman_data() -> pd.DataFrame:
    same_set = {"HH", "LL"}
    frames = []
    for exp_num, path in DATASETS.items():
        df = pd.read_pickle(path)
        df = df[df["is_outlier"] == False].copy()
        df = (
            df.assign(
                exp=exp_num,
                Structure=lambda d: np.where(d["TransitionType"].isin(same_set), "Repeat", "Switch"),
            )
            .rename(
                columns={
                    "curDur": "Duration",
                    "curBias": "Bias",
                    "rpr": "Reproduction",
                    "curCoherence": "coherence",
                    "subID": "Sub",
                }
            )
            .astype({"coherence": float, "Sub": int})
        )
        frames.append(
            df[
                [
                    "Sub",
                    "exp",
                    "trial_num",
                    "coherence",
                    "Structure",
                    "Duration",
                    "Bias",
                    "Reproduction",
                ]
            ]
        )
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUTDIR / "kalman_input_data.csv", index=False)
    return data


def completed_keys(output_file: Path) -> set[tuple[int, int, str]]:
    if not output_file.exists():
        return set()
    df = read_checkpoint_flexible(output_file)
    if df.empty:
        return set()
    deduped = df.drop_duplicates(["Sub", "exp", "model_id"], keep="last")
    return set(zip(deduped["Sub"].astype(int), deduped["exp"].astype(int), deduped["model_id"].astype(str)))


def append_checkpoint(rows: list[dict], output_file: Path, write_header: bool) -> bool:
    pd.DataFrame(rows).to_csv(output_file, mode="a", header=write_header, index=False)
    return False


def read_checkpoint_flexible(output_file: Path) -> pd.DataFrame:
    """Read checkpoint CSV even if appended batches had different parameter columns."""
    if not output_file.exists():
        return pd.DataFrame()

    fixed_cols = [
        "model_id",
        "model_name",
        "c_id",
        "s_id",
        "b_id",
        "n_params",
        "AIC",
        "BIC",
        "RMSE",
        "DW",
        "RSS",
        "N",
        "success",
        "Sub",
        "exp",
    ]
    rows = []
    model_re = re.compile(r"^C\d+_S[012]_B[123]$")

    with output_file.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header from the first appended batch
        for raw in reader:
            model_idx = next((i for i, value in enumerate(raw) if model_re.match(value)), None)
            if model_idx is None:
                continue

            fixed = raw[model_idx : model_idx + len(fixed_cols)]
            if len(fixed) < len(fixed_cols):
                continue
            record = dict(zip(fixed_cols, fixed))
            param_values = raw[:model_idx]
            try:
                param_names = get_parameter_config(
                    record["c_id"], record["s_id"], record["b_id"]
                )["names"]
            except Exception:
                continue
            for name, value in zip(param_names, param_values):
                record[name] = value
            rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    numeric_cols = [
        "n_params",
        "AIC",
        "BIC",
        "RMSE",
        "DW",
        "RSS",
        "N",
        "Sub",
        "exp",
        "q1",
        "q2",
        "q3",
        "lambda",
        "r_base",
        "d0",
        "alpha_d0",
        "alpha_q1",
        "alpha_q2",
        "alpha_q3",
        "r_low",
        "x_reset",
        "k_reset",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "success" in df.columns:
        df["success"] = df["success"].astype(str).str.lower().eq("true")
    return df


def fit_one_task(task: dict) -> dict:
    P.C_MODULATION_PARAMS["Q1"]["lower"] = -task["alpha_q1_bound"]
    P.C_MODULATION_PARAMS["Q1"]["upper"] = task["alpha_q1_bound"]
    model = task["model"]
    result = fit_single_model(
        task["stimrep"],
        task["coherence"],
        task["structure"],
        model["c_id"],
        model["s_id"],
        model["b_id"],
        n_starts=N_STARTS,
        max_nfev=MAX_NFEV,
        seed=task["seed"],
    )
    result["Sub"] = task["sub"]
    result["exp"] = task["exp_num"]
    return result


def build_pending_tasks(data: pd.DataFrame, completed: set[tuple[int, int, str]], alpha_q1_bound: float) -> list[dict]:
    models = get_all_models(include_switching=True)
    tasks = []
    skipped = 0

    for exp_num in sorted(data["exp"].unique()):
        exp_data = data[data["exp"].eq(exp_num)]
        subjects = sorted(exp_data["Sub"].unique())
        print(
            f"Exp{exp_num}: {len(subjects)} subjects, {len(models)} models, "
            f"{len(subjects) * len(models)} fits",
            flush=True,
        )

        for subject_index, sub in enumerate(subjects, start=1):
            sub_data = exp_data[exp_data["Sub"].eq(sub)]
            stimrep = sub_data[["Duration", "Reproduction"]].to_numpy()
            coherence = sub_data["coherence"].to_numpy()
            structure = sub_data["Structure"].to_numpy()

            sub_new = 0
            for model in models:
                key = (int(sub), int(exp_num), model["model_id"])
                if key in completed:
                    skipped += 1
                    continue

                tasks.append(
                    {
                        "sub": int(sub),
                        "exp_num": int(exp_num),
                        "model": model,
                        "stimrep": stimrep,
                        "coherence": coherence,
                        "structure": structure,
                        "seed": stable_seed("fit_cleaned", sub, exp_num, model["model_id"]),
                        "alpha_q1_bound": alpha_q1_bound,
                    }
                )
                sub_new += 1

            print(
                f"  queued Exp{exp_num} subject {subject_index}/{len(subjects)} "
                f"Sub={sub}, new fits={sub_new}",
                flush=True,
            )

    return tasks, skipped


def fit_full(data: pd.DataFrame, alpha_q1_bound: float, workers: int = 1) -> pd.DataFrame:
    output_file = OUTDIR / "model_fits.csv"
    completed = completed_keys(output_file)
    pending_rows: list[dict] = []
    write_header = not output_file.exists()

    tasks, skipped = build_pending_tasks(data, completed, alpha_q1_bound)
    total_possible = len(tasks) + skipped
    new_fits = 0

    if not tasks:
        print(f"New fits run: 0; skipped completed: {skipped}", flush=True)
        df = read_checkpoint_flexible(output_file)
        deduped = df.drop_duplicates(["Sub", "exp", "model_id"], keep="last")
        deduped.to_csv(OUTDIR / "model_fits_deduplicated.csv", index=False)
        return deduped

    if workers > 1:
        print(f"Running {len(tasks)} pending fits with {workers} workers", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fit_one_task, task) for task in tasks]
            for future in as_completed(futures):
                pending_rows.append(future.result())
                new_fits += 1
                if len(pending_rows) >= SAVE_EVERY:
                    write_header = append_checkpoint(pending_rows, output_file, write_header)
                    pending_rows = []
                    print(
                        f"  checkpoint: new={new_fits}, skipped={skipped}, "
                        f"done_total={new_fits + skipped}/{total_possible}",
                        flush=True,
                    )
    else:
        print(f"Running {len(tasks)} pending fits serially", flush=True)
        for task in tasks:
            pending_rows.append(fit_one_task(task))
            new_fits += 1
            if len(pending_rows) >= SAVE_EVERY:
                write_header = append_checkpoint(pending_rows, output_file, write_header)
                pending_rows = []
                print(
                    f"  checkpoint: new={new_fits}, skipped={skipped}, "
                    f"done_total={new_fits + skipped}/{total_possible}",
                    flush=True,
                )

    if pending_rows:
        write_header = append_checkpoint(pending_rows, output_file, write_header)

    print(f"New fits run: {new_fits}; skipped completed: {skipped}", flush=True)
    df = read_checkpoint_flexible(output_file)
    deduped = df.drop_duplicates(["Sub", "exp", "model_id"], keep="last")
    deduped.to_csv(OUTDIR / "model_fits_deduplicated.csv", index=False)
    return deduped


def summarize(results: pd.DataFrame) -> None:
    rank = (
        results.groupby(["exp", "model_id", "model_name"], as_index=False)
        .agg(
            mean_AIC=("AIC", "mean"),
            sem_AIC=("AIC", "sem"),
            mean_BIC=("BIC", "mean"),
            mean_RMSE=("RMSE", "mean"),
            n_subjects=("Sub", "nunique"),
            success_rate=("success", "mean"),
        )
        .sort_values(["exp", "mean_AIC"])
    )
    rank["delta_AIC"] = rank.groupby("exp")["mean_AIC"].transform(lambda x: x - x.min())
    rank.to_csv(OUTDIR / "model_ranking.csv", index=False)

    param_cols = [
        c
        for c in [
            "q1",
            "q2",
            "q3",
            "lambda",
            "d0",
            "alpha_d0",
            "alpha_q1",
            "alpha_q2",
            "alpha_q3",
            "r_base",
            "r_low",
            "x_reset",
            "k_reset",
        ]
        if c in results.columns
    ]
    params = (
        results.groupby(["exp", "model_id", "model_name"], as_index=False)[param_cols]
        .mean(numeric_only=True)
        .sort_values(["exp", "model_id"])
    )
    params.to_csv(OUTDIR / "model_parameter_means.csv", index=False)


def main() -> None:
    args = parse_args()
    configure_run(args)
    data = load_kalman_data()
    print(
        data.groupby("exp")
        .agg(n_trials=("Duration", "size"), n_subjects=("Sub", "nunique"))
        .to_string(),
        flush=True,
    )
    results = fit_full(data, alpha_q1_bound=args.alpha_q1_bound, workers=args.workers)
    summarize(results)
    print(f"\nOutputs written to: {display_path(OUTDIR)}", flush=True)


if __name__ == "__main__":
    main()
