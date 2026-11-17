"""Summaries of the S0/S3 model recovery (Appendix B.6, Figure B5A) over the simulated 22-participant cohorts.

  switch_cohorts.csv           per generator and repetition: mean participant AIC of both candidates, their
                               difference and the selected candidate
  switch_confusion_counts.csv  generating x selected model, cohort counts
  cohort_results.csv           per cohort: AIC and the focal-contrast check of both candidates
  summary.csv                  per generator: cohort and dataset-level selection rates, focal-contrast checks

Usage: python scripts/model/model_recovery_summary.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import fitting as FF
from sdu.model.comparison import Z_PASS

sys.path.insert(0, str(SP.ROOT / "scripts" / "model"))
from model_recovery import CANDIDATES, EXP, N_REP, N_SIM, N_STARTS  # noqa: E402


def load_results():
    consolidated = SP.MODEL_RECOVERY / "participant_fits.csv"
    paths = sorted(SP.MODEL_RECOVERY.glob(f"switch_exp{EXP}_*.csv"))
    if consolidated.exists():
        data = pd.read_csv(consolidated)
    elif paths:
        data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    else:
        raise FileNotFoundError("No S0/S3 participant fits found")
    keys = ["generator", "rep", "candidate"]
    if data.Sub.nunique() != len(FF.load(EXP)) or data.duplicated(keys + ["Sub"]).any() \
            or not data.groupby(keys).size().eq(data.Sub.nunique()).all():
        raise ValueError("incomplete or duplicated S0/S3 cross-fits")
    return data


def switch_cohorts(data):
    aic = data.groupby(["generator", "rep", "candidate"]).AIC.mean().unstack("candidate").reset_index()
    aic.columns.name = None
    aic["delta_AIC_S3_minus_S0"] = aic.S3 - aic.S0
    aic["winner"] = np.where(aic.S0 < aic.S3, "S0", "S3")
    matrix = pd.crosstab(aic.generator, aic.winner).reindex(index=list(CANDIDATES), columns=list(CANDIDATES),
                                                            fill_value=0)
    return aic, matrix


def cohort_table(data):
    rows = []
    for (generator, rep), cohort in data.groupby(["generator", "rep"], sort=False):
        observed = cohort.drop_duplicates("Sub").obs_focal
        sem = observed.std(ddof=1) / np.sqrt(observed.notna().sum())
        aic = cohort.groupby("candidate").AIC.mean()
        result = dict(generator=generator, rep=rep, observed_focal=observed.mean(), observed_focal_sem=sem,
                      aic_winner=aic.idxmin(), aic_correct=aic.idxmin() == generator)
        for candidate in CANDIDATES:
            predicted = cohort[cohort.candidate == candidate].sim_focal.mean()
            z = (predicted - observed.mean()) / sem
            result.update({f"mean_AIC_{candidate}": aic[candidate], f"predicted_focal_{candidate}": predicted,
                           f"z_focal_{candidate}": z, f"passes_focal_{candidate}": abs(z) < Z_PASS})
        other = next(c for c in CANDIDATES if c != generator)
        result["generator_only_passes_focal"] = result[f"passes_focal_{generator}"] and not result[f"passes_focal_{other}"]
        rows.append(result)
    return pd.DataFrame(rows)


def summary_table(data, cohorts):
    rows = []
    per_dataset = data.pivot_table(index=["generator", "rep", "Sub"], columns="candidate", values="AIC")
    for generator, group in cohorts.groupby("generator", sort=False):
        other = next(c for c in CANDIDATES if c != generator)
        datasets = per_dataset.loc[generator]
        rows.append({
            "generator": generator, "cohorts": len(group), "AIC correct": group.aic_correct.mean(),
            "datasets": len(datasets), "datasets AIC correct": int((datasets[generator] < datasets[other]).sum()),
            "generator passes focal": group[f"passes_focal_{generator}"].to_numpy(dtype=bool).mean(),
            "other passes focal": group[f"passes_focal_{other}"].to_numpy(dtype=bool).mean(),
            "generator only passes focal": group.generator_only_passes_focal.mean(),
            "median focal z (generator)": group[f"z_focal_{generator}"].median(),
            "median focal z (other)": group[f"z_focal_{other}"].median()})
    return pd.DataFrame(rows)


def main():
    data = load_results()
    consolidated = SP.MODEL_RECOVERY / "participant_fits.csv"
    if not consolidated.exists():
        data.to_csv(consolidated, index=False)
    aic, matrix = switch_cohorts(data)
    cohorts = cohort_table(data)
    summary = summary_table(data, cohorts)
    out = SP.MODEL_RECOVERY
    aic.to_csv(out / "switch_cohorts.csv", index=False)
    matrix.to_csv(out / "switch_confusion_counts.csv")
    cohorts.to_csv(out / "cohort_results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    print(matrix.to_string())
    print(summary.set_index("generator").round(3).to_string())
    print(f"\nConfiguration: {N_REP} cohorts, {N_STARTS + 1} starts, {N_SIM} fitted-model simulations")


if __name__ == "__main__":
    main()
