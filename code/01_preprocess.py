#!/usr/bin/env python3
"""
Recompute preprocessing with a two-stage outlier rule.

Stage 1 screens participants using only objective trial criteria:
    - valid target duration, non-missing response, abs(response error) <= 0.6 s
    - participant valid-trial ratio > 0.8 of 240 trials

Stage 2 recomputes trial-level +/- 3 SD outliers only within the retained participants.
Previous-trial variables are constructed after participant selection and are reset at
block boundaries so they do not carry across blocks. Boundary trials are retained
in the output and marked as analysis exclusions rather than dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "preprocessing"

ERROR_THRESH = 0.6
TIME_BOUNDS = (0.8, 1.6)
EXPECTED_TRIALS = 240
MIN_VALID_RATIO = 0.8
MAX_N_BACK = 10
MAX_N_FUTURE = 2


@dataclass(frozen=True)
class ExperimentConfig:
    exp_num: int
    rawdata_dir: Path
    output_name: str
    max_n_future: int = 0


EXPERIMENTS = [
    ExperimentConfig(
        exp_num=1,
        rawdata_dir=ROOT / "raw" / "experiment1",
        output_name="E1.pkl",
        max_n_future=MAX_N_FUTURE,
    ),
    ExperimentConfig(
        exp_num=2,
        rawdata_dir=ROOT / "raw" / "experiment2",
        output_name="E2.pkl",
        max_n_future=MAX_N_FUTURE,
    ),
]


def read_experiment_raw(rawdata_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read PsychoPy CSV files using the same header + line-33 trimming as the notebooks."""
    rows = []
    skipped = []
    encodings = ["utf-8", "utf-8-sig", "latin1", "gbk", "windows-1252"]

    for file_path in sorted(rawdata_dir.glob("*.csv")):
        lines = None
        used_encoding = None
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as handle:
                    lines = handle.readlines()
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if lines is None or len(lines) < 34:
            skipped.append({"file": file_path.name, "reason": "too_short_or_encoding"})
            continue

        try:
            df = pd.read_csv(StringIO("".join([lines[0]] + lines[33:])))
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped.append({"file": file_path.name, "reason": f"parse_error: {exc}"})
            continue

        required = {"keyDuration", "TimeDur", "Coherence1", "trials.thisN", "trials.thisTrialN"}
        if not required.issubset(df.columns):
            skipped.append({"file": file_path.name, "reason": "missing_required_columns"})
            continue

        df["source_file"] = file_path.name
        df["source_encoding"] = used_encoding
        df["subID"] = file_path.name.split("_")[0]
        df["keyDuration"] = pd.to_numeric(df["keyDuration"], errors="coerce") / 1000.0
        df["TimeDur"] = pd.to_numeric(df["TimeDur"], errors="coerce")
        df["Coherence1"] = pd.to_numeric(df["Coherence1"], errors="coerce")
        df["trials.thisN"] = pd.to_numeric(df["trials.thisN"], errors="coerce")
        df["trials.thisTrialN"] = pd.to_numeric(df["trials.thisTrialN"], errors="coerce")
        df["ReproductionError"] = df["keyDuration"] - df["TimeDur"]
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No usable CSV files found in {rawdata_dir}")

    return pd.concat(rows, ignore_index=True), pd.DataFrame(skipped)


def add_objective_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add fixed, sample-independent trial validity flags."""
    df = df.copy()
    df["bad_time_bounds"] = ~df["TimeDur"].between(TIME_BOUNDS[0], TIME_BOUNDS[1])
    df["bad_missing_response"] = df[["TimeDur", "keyDuration", "ReproductionError"]].isna().any(axis=1)
    df["bad_abs_error"] = df["ReproductionError"].abs() > ERROR_THRESH
    df["objective_outlier"] = (
        df["bad_time_bounds"] | df["bad_missing_response"] | df["bad_abs_error"]
    )
    return df


def screen_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Screen participants using only objective trial criteria."""
    total = df.groupby("subID").size().rename("raw_trials")
    valid = (~df["objective_outlier"]).groupby(df["subID"]).sum().rename("objective_valid_trials")
    summary = pd.concat([total, valid], axis=1).fillna(0)
    summary["objective_valid_trials"] = summary["objective_valid_trials"].astype(int)
    summary["expected_trials"] = EXPECTED_TRIALS
    summary["objective_valid_ratio"] = summary["objective_valid_trials"] / EXPECTED_TRIALS
    summary["included"] = summary["objective_valid_ratio"] > MIN_VALID_RATIO
    return summary.reset_index()


def add_final_sample_outlier_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag analysis-stage outlier trials.

    Primary rule: a hard plausibility bound |reproduction error| > 0.6 s (carried in
    `objective_outlier`, also used for the >80% subject screening) PLUS a within
    participant x stimulus-duration +/- 3 SD distributional rule on reproduction error.
    Block-boundary first trials are removed downstream in `construct_trial_variables`.
    """
    df = df.copy()
    df["sd_outlier_final_sample"] = False
    usable = ~(df["bad_time_bounds"] | df["bad_missing_response"])
    thresholds = []
    for (sub, duration), group in df.groupby(["subID", "TimeDur"], dropna=False):
        idx = group.index[usable.loc[group.index]]
        if len(idx) < 4:
            continue
        err = df.loc[idx, "ReproductionError"]
        mu, sd = err.mean(), err.std()
        if pd.isna(sd) or sd == 0:
            continue
        df.loc[idx, "sd_outlier_final_sample"] = (err - mu).abs() > 3 * sd
        thresholds.append(
            {
                "subID": sub,
                "curDur": duration,
                "n": int(len(idx)),
                "mean_error": float(mu),
                "sd_error": float(sd),
                "lower": float(mu - 3 * sd),
                "upper": float(mu + 3 * sd),
            }
        )

    df["is_outlier"] = df["objective_outlier"] | df["sd_outlier_final_sample"]
    return df, pd.DataFrame(thresholds)


def transition_label(previous: pd.Series, current: pd.Series) -> pd.Series:
    """Return HH/HL/LH/LL labels where 0.3 is high uncertainty and 0.7 is low."""
    labels = pd.Series(pd.NA, index=current.index, dtype="object")
    ok = previous.notna() & current.notna()
    prev_label = np.where(np.isclose(previous[ok], 0.7), "L", "H")
    curr_label = np.where(np.isclose(current[ok], 0.7), "L", "H")
    labels.loc[ok] = prev_label + curr_label
    return labels


def construct_trial_variables(df: pd.DataFrame, max_n_future: int = 0) -> pd.DataFrame:
    """Construct analysis columns and reset history variables at block boundaries."""
    df = df.copy()
    df = df.rename(
        columns={
            "trials.thisN": "trial_num",
            "trials.thisTrialN": "trial_in_block",
            "TimeDur": "curDur",
            "keyDuration": "rpr",
            "ReproductionError": "curBias",
            "Coherence1": "curCoherence",
        }
    )

    boundary = pd.to_numeric(df["trial_in_block"], errors="coerce").eq(0)
    df["block_boundary_outlier"] = boundary
    df["is_outlier"] = df["is_outlier"] | df["block_boundary_outlier"]

    response_invalid = df["objective_outlier"] | df["sd_outlier_final_sample"]
    valid_response_for_mean = df["rpr"].where(~response_invalid)
    df["mean_rpr"] = valid_response_for_mean.groupby(df["subID"]).transform("mean")
    df["resp_type"] = np.where(df["rpr"] > df["mean_rpr"], "Long", "Short")
    df.loc[df["rpr"].isna() | response_invalid, "resp_type"] = np.nan
    df["preResp"] = df.groupby("subID")["resp_type"].shift(1)
    df.loc[boundary, "preResp"] = np.nan

    for n in range(1, MAX_N_BACK + 1):
        df[f"preDur{n}back"] = df.groupby("subID")["curDur"].shift(n)
        df[f"preCoherence{n}back"] = df.groupby("subID")["curCoherence"].shift(n)
        within_block_ok = pd.to_numeric(df["trial_in_block"], errors="coerce") >= n
        df.loc[~within_block_ok, [f"preDur{n}back", f"preCoherence{n}back"]] = np.nan

    for f in range(1, max_n_future + 1):
        df[f"postDur{f}"] = df.groupby("subID")["curDur"].shift(-f)
        df[f"postCoherence{f}"] = df.groupby("subID")["curCoherence"].shift(-f)
        within_block_ok = pd.to_numeric(df["trial_in_block"], errors="coerce") <= (29 - f)
        df.loc[~within_block_ok, [f"postDur{f}", f"postCoherence{f}"]] = np.nan

    df["TransitionType"] = transition_label(df["preCoherence1back"], df["curCoherence"])

    df["preDur1backc"] = df["preDur1back"] - 1.2
    df["curDurc"] = df["curDur"] - 1.2

    core_cols = [
        "subID",
        "trial_num",
        "trial_in_block",
        "curDur",
        "rpr",
        "curBias",
        "curCoherence",
        "preDur1back",
        "preCoherence1back",
        "TransitionType",
        "is_outlier",
        "objective_outlier",
        "sd_outlier_final_sample",
        "bad_abs_error",
        "bad_missing_response",
        "bad_time_bounds",
        "block_boundary_outlier",
    ]
    pre_cols = [f"preDur{n}back" for n in range(2, MAX_N_BACK + 1)] + [
        f"preCoherence{n}back" for n in range(2, MAX_N_BACK + 1)
    ]
    post_cols = [f"postDur{f}" for f in range(1, max_n_future + 1)] + [
        f"postCoherence{f}" for f in range(1, max_n_future + 1)
    ]
    derived_cols = ["preDur1backc", "curDurc", "mean_rpr", "resp_type", "preResp"]
    keep_cols = [col for col in core_cols + pre_cols + post_cols + derived_cols if col in df.columns]
    return df[keep_cols].copy()


def process_experiment(config: ExperimentConfig) -> dict[str, object]:
    raw, skipped = read_experiment_raw(config.rawdata_dir)
    raw = add_objective_flags(raw)
    subject_summary = screen_subjects(raw)

    included_subjects = set(subject_summary.loc[subject_summary["included"], "subID"].astype(str))
    retained_raw = raw[raw["subID"].astype(str).isin(included_subjects)].copy()
    retained_marked, thresholds = add_final_sample_outlier_flags(retained_raw)
    analysis_df = construct_trial_variables(retained_marked, max_n_future=config.max_n_future)

    output_path = ROOT / "data" / f"experiment{config.exp_num}" / config.output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_pickle(output_path)
    analysis_df.to_csv(output_path.with_suffix(".csv"), index=False)

    subject_summary.insert(0, "Experiment", config.exp_num)
    thresholds.insert(0, "Experiment", config.exp_num)
    skipped.insert(0, "Experiment", config.exp_num)

    valid = analysis_df[~analysis_df["is_outlier"]]
    return {
        "Experiment": config.exp_num,
        "raw_subjects": int(raw["subID"].nunique()),
        "raw_rows": int(len(raw)),
        "included_subjects": int(len(included_subjects)),
        "analysis_rows": int(len(analysis_df)),
        "final_outliers": int(analysis_df["is_outlier"].sum()),
        "valid_analysis_rows": int(len(valid)),
        "valid_trials_min": int(valid.groupby("subID").size().min()),
        "valid_trials_mean": float(valid.groupby("subID").size().mean()),
        "valid_trials_max": int(valid.groupby("subID").size().max()),
        "output": str(output_path.relative_to(ROOT)),
        "subject_summary": subject_summary,
        "thresholds": thresholds,
        "skipped": skipped,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    subject_tables = []
    threshold_tables = []
    skipped_tables = []

    for config in EXPERIMENTS:
        result = process_experiment(config)
        summaries.append({k: v for k, v in result.items() if not isinstance(v, pd.DataFrame)})
        subject_tables.append(result["subject_summary"])
        threshold_tables.append(result["thresholds"])
        skipped_tables.append(result["skipped"])

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTDIR / "preprocessing_summary.csv", index=False)
    pd.concat(subject_tables, ignore_index=True).to_csv(OUTDIR / "subject_screening.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(
        OUTDIR / "final_sample_sd_thresholds.csv", index=False
    )
    pd.concat(skipped_tables, ignore_index=True).to_csv(OUTDIR / "skipped_files.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"\nOutputs written to: {OUTDIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
