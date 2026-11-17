#!/usr/bin/env python3
"""Audit coherence sequences without treating random transitions as volatility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sdu import paths as SP  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = SP.BEHAVIOR_SEQUENCES
OUT.mkdir(parents=True, exist_ok=True)

DATA = {
    "Exp1_ramped": SP.PREPROCESSED_EXP[1],
    "Exp2_constant": SP.PREPROCESSED_EXP[2],
}


def add_runs(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["subID", "trial_num"]).copy()
    d["block"] = d.groupby("subID")["trial_in_block"].transform(lambda x: x.eq(0).cumsum())
    grouped = d.groupby(["subID", "block"], sort=False)
    d["previous_coherence"] = grouped["curCoherence"].shift(1)
    transition = np.where(
        np.isclose(d["previous_coherence"], 0.7),
        np.where(np.isclose(d["curCoherence"], 0.7), "HH", "HL"),
        np.where(np.isclose(d["curCoherence"], 0.7), "LH", "LL"),
    ).astype(object)
    transition[d["previous_coherence"].isna().to_numpy()] = None
    d["transition"] = transition
    d["switch"] = d["transition"].isin(["HL", "LH"])
    # Run lengths are descriptive only; randomized runs were not experimental regimes.
    change = grouped["curCoherence"].transform(lambda x: x.ne(x.shift()).cumsum())
    d["run_id"] = change
    return d


def main() -> None:
    cell_rows = []
    participant_rows = []
    run_rows = []

    for experiment, path in DATA.items():
        d = add_runs(pd.read_csv(path, dtype={"subID": str}))
        transitions = d.dropna(subset=["transition"]).copy()
        for (participant, transition), g in transitions.groupby(["subID", "transition"]):
            cell_rows.append({
                "experiment": experiment,
                "participant": participant,
                "transition": transition,
                "count": len(g),
            })
        for participant, g in transitions.groupby("subID"):
            counts = g["transition"].value_counts()
            participant_rows.append({
                "experiment": experiment,
                "participant": participant,
                "n_transitions": len(g),
                "p_low": float(np.mean(np.isclose(g["curCoherence"], 0.3))),
                "p_switch": float(g["switch"].mean()),
                "p_low_given_previous_high": float(np.mean(np.isclose(
                    g.loc[np.isclose(g["previous_coherence"], 0.7), "curCoherence"], 0.3
                ))),
                "p_low_given_previous_low": float(np.mean(np.isclose(
                    g.loc[np.isclose(g["previous_coherence"], 0.3), "curCoherence"], 0.3
                ))),
                **{f"n_{cell}": int(counts.get(cell, 0)) for cell in ["HH", "HL", "LH", "LL"]},
            })
        for (participant, block, run_id), g in d.groupby(["subID", "block", "run_id"]):
            run_rows.append({
                "experiment": experiment,
                "participant": participant,
                "block": int(block),
                "coherence": float(g["curCoherence"].iloc[0]),
                "run_length": len(g),
            })

    cells = pd.DataFrame(cell_rows)
    participants = pd.DataFrame(participant_rows)
    runs = pd.DataFrame(run_rows)
    cell_summary = cells.groupby(["experiment", "transition"])["count"].agg(
        total="sum", mean_per_participant="mean", sd_per_participant="std", min="min", max="max"
    ).reset_index()
    sequence_summary = participants.groupby("experiment").agg(
        n_participants=("participant", "nunique"),
        mean_p_low=("p_low", "mean"),
        mean_p_switch=("p_switch", "mean"),
        mean_p_low_given_previous_high=("p_low_given_previous_high", "mean"),
        mean_p_low_given_previous_low=("p_low_given_previous_low", "mean"),
    ).reset_index()
    run_summary = runs.groupby("experiment")["run_length"].agg(
        n_runs="count", mean="mean", median="median", max="max"
    ).reset_index()

    cells.to_csv(OUT / "participant_transition_counts.csv", index=False)
    participants.to_csv(OUT / "participant_sequence_metrics.csv", index=False)
    runs.to_csv(OUT / "coherence_runs.csv", index=False)
    cell_summary.to_csv(OUT / "transition_summary.csv", index=False)
    sequence_summary.to_csv(OUT / "sequence_summary.csv", index=False)
    run_summary.to_csv(OUT / "run_summary.csv", index=False)
    print(cell_summary.to_string(index=False))
    print("\nSequence summary")
    print(sequence_summary.to_string(index=False))
    print("\nRun summary")
    print(run_summary.to_string(index=False))


if __name__ == "__main__":
    main()
