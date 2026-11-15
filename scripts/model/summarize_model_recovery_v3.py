"""Summarize targeted phi and switch model recovery over complete simulated cohorts."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import fitting as FF
from sdu.model.screening import Z_PASS

sys.path.insert(0, str(SP.ROOT / "scripts" / "model"))
from model_recovery_v3 import ANALYSES, N_REP, N_SIM, N_STARTS, output_directory  # noqa: E402


def load_results():
    directory = output_directory()
    paths = sorted(directory.glob("phi_exp*_*.csv")) + sorted(directory.glob("switch_exp*_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no model-recovery files in {directory}")
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected_participants = {name: len(FF.load(definition["exp"])) for name, definition in ANALYSES.items()}
    for analysis in data.analysis.unique():
        group = data[data.analysis == analysis]
        if group.Sub.nunique() != expected_participants[analysis]:
            raise ValueError(
                f"incomplete {analysis} recovery: {group.Sub.nunique()}/{expected_participants[analysis]} participants"
            )
    return directory, data


def cohort_table(data):
    rows = []
    keys = ["analysis", "generator", "rep"]
    for (analysis, generator, rep), cohort in data.groupby(keys, sort=False):
        candidates = sorted(cohort.candidate.unique())
        observed = cohort.drop_duplicates("Sub").obs_focal
        sem = observed.std(ddof=1) / np.sqrt(observed.notna().sum())
        result = {
            "analysis": analysis,
            "generator": generator,
            "rep": rep,
            "observed_focal": observed.mean(),
            "observed_focal_sem": sem,
        }
        aic = cohort.groupby("candidate").AIC.mean()
        result["aic_winner"] = aic.idxmin()
        result["aic_correct"] = result["aic_winner"] == generator
        for candidate in candidates:
            predicted = cohort[cohort.candidate == candidate].sim_focal.mean()
            z = (predicted - observed.mean()) / sem
            result[f"mean_AIC_{candidate}"] = aic[candidate]
            result[f"predicted_focal_{candidate}"] = predicted
            result[f"z_focal_{candidate}"] = z
            result[f"passes_focal_{candidate}"] = abs(z) < Z_PASS
        other = next(candidate for candidate in candidates if candidate != generator)
        result["generator_only_passes_focal"] = (
            result[f"passes_focal_{generator}"] and not result[f"passes_focal_{other}"]
        )
        rows.append(result)
    return pd.DataFrame(rows)


def summary_table(cohorts):
    rows = []
    for (analysis, generator), group in cohorts.groupby(["analysis", "generator"], sort=False):
        candidates = set(ANALYSES[analysis]["candidates"])
        other = next(candidate for candidate in candidates if candidate != generator)
        rows.append(
            {
                "analysis": analysis,
                "generator": generator,
                "cohorts": len(group),
                "AIC correct": group.aic_correct.mean(),
                "generator passes focal": group[f"passes_focal_{generator}"].to_numpy(dtype=bool).mean(),
                "other passes focal": group[f"passes_focal_{other}"].to_numpy(dtype=bool).mean(),
                "generator only passes focal": group.generator_only_passes_focal.mean(),
                "median focal z (generator)": group[f"z_focal_{generator}"].median(),
                "median focal z (other)": group[f"z_focal_{other}"].median(),
            }
        )
    return pd.DataFrame(rows)


def main():
    directory, data = load_results()
    cohorts = cohort_table(data)
    summary = summary_table(cohorts)
    cohorts.to_csv(directory / "cohort_results.csv", index=False)
    summary.to_csv(directory / "summary.csv", index=False)
    print(summary.set_index(["analysis", "generator"]).round(3).to_string())
    print(f"\nConfiguration: {N_REP} cohorts, {N_STARTS + 1} starts, {N_SIM} fitted-model simulations")
    print(f"Wrote {directory / 'cohort_results.csv'}")
    print(f"Wrote {directory / 'summary.csv'}")


if __name__ == "__main__":
    main()
