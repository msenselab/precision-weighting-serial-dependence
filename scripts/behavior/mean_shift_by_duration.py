#!/usr/bin/env python3
"""Is the current-coherence mean shift additive or proportional to duration?

For each participant and physical duration, the mean reproduction under low
coherence minus the mean under high coherence. A boundary-detection account
(coherent motion registered late at onset and early at offset) predicts the
same shift at every duration. A change in the rate of timing predicts a shift
proportional to duration, that is, a slope of (mean shift) / (mean duration).

Trial selection follows ``current_coherence_mean_shift.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sdu import paths as SP

OUT = SP.RESULTS_BEHAVIOR / "analysis" / "current_coherence_mean_shift"
OUT.mkdir(parents=True, exist_ok=True)

# File names keep the collection order: E2 is Experiment 1 (ramped coherence).
DATA = {
    "Exp 1": SP.PREPROCESSED / "E2_reprocessed.csv",
    "Exp 2": SP.PREPROCESSED / "E1_reprocessed.csv",
}
MEAN_DURATION = 1.2


def shift_table(path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"subID": str})
    d = d.loc[~d["is_outlier"]].dropna(
        subset=["subID", "rpr", "curDur", "preDur1back", "curCoherence", "preCoherence1back"]
    )
    d["coh"] = np.where(np.isclose(d["curCoherence"], 0.3), "low", "high")
    cell = d.groupby(["subID", "curDur", "coh"])["rpr"].mean().unstack("coh")
    return (cell["low"] - cell["high"]).unstack("curDur")


def main() -> None:
    rows, tests = [], []
    for experiment, path in DATA.items():
        w = shift_table(path)
        durations = w.columns.to_numpy(dtype=float)
        for dur in w.columns:
            rows.append({"experiment": experiment, "duration_s": dur, "n": w[dur].notna().sum(),
                         "low_minus_high_s": w[dur].mean(), "sem_s": w[dur].sem()})
        slope = w.apply(lambda r: np.polyfit(durations, r.to_numpy(), 1)[0], axis=1)
        mean_shift = w.mean(axis=1).mean()
        proportional = mean_shift / MEAN_DURATION
        t0 = ttest_1samp(slope, 0.0)
        tp = ttest_1samp(slope, proportional)
        tests.append({"experiment": experiment, "n": len(slope), "mean_shift_s": mean_shift,
                      "slope_on_duration": slope.mean(), "slope_sem": slope.sem(),
                      "t_vs_zero": t0.statistic, "p_vs_zero": t0.pvalue,
                      "proportional_slope": proportional,
                      "t_vs_proportional": tp.statistic, "p_vs_proportional": tp.pvalue})
    by_duration, summary = pd.DataFrame(rows), pd.DataFrame(tests)
    by_duration.to_csv(OUT / "shift_by_duration.csv", index=False)
    summary.to_csv(OUT / "shift_by_duration_tests.csv", index=False)
    print(by_duration.round(4).to_string(index=False))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
