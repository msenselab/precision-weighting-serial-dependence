#!/usr/bin/env python3
"""Estimate the current-coherence mean shift with matched models in both experiments.

Coherence predictors use effect coding (-0.5 high, +0.5 low), so the
``current_low`` coefficient is the Low-minus-High response-bias contrast,
averaged over previous coherence at centered current/previous durations.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm, ttest_1samp, ttest_ind
from sdu import paths as SP  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = SP.RESULTS_BEHAVIOR / "analysis" / "current_coherence_mean_shift"
OUT.mkdir(parents=True, exist_ok=True)

DATA = {
    "Exp1_dynamic": SP.PREPROCESSED / "E2_reprocessed.csv",
    "Exp2_fixed": SP.PREPROCESSED / "E1_reprocessed.csv",
}

BASE_FORMULA = (
    "curBias ~ curDur_c * previous_low * current_low "
    "+ preDur_c * previous_low * current_low"
)


def load_data() -> pd.DataFrame:
    frames = []
    for experiment, path in DATA.items():
        d = pd.read_csv(path, dtype={"subID": str})
        d = d.loc[~d["is_outlier"]].dropna(
            subset=[
                "subID", "curBias", "curDur", "preDur1back",
                "curCoherence", "preCoherence1back",
            ]
        ).copy()
        d["experiment"] = experiment
        d["participant"] = experiment + "_" + d["subID"].astype(str)
        d["curDur_c"] = d["curDur"] - 1.2
        d["preDur_c"] = d["preDur1back"] - 1.2
        d["current_low"] = np.where(np.isclose(d["curCoherence"], 0.3), 0.5, -0.5)
        d["previous_low"] = np.where(np.isclose(d["preCoherence1back"], 0.3), 0.5, -0.5)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fit_mixed(formula: str, data: pd.DataFrame, re_formula: str):
    model = smf.mixedlm(formula, data, groups=data["participant"], re_formula=re_formula)
    messages = []
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(reml=True, method=method, maxiter=3000, disp=False)
            messages.extend(str(w.message) for w in caught)
            if result.converged and np.all(np.isfinite(result.bse_fe)):
                return result, method, " | ".join(sorted(set(messages)))
        except Exception as exc:  # pragma: no cover - optimizer fallback
            messages.append(f"{method}: {exc}")
    raise RuntimeError("Mixed model did not converge: " + " | ".join(messages))


def contrast_row(result, experiment: str, label: str, vector: np.ndarray) -> dict:
    names = result.fe_params.index.tolist()
    beta = result.fe_params.to_numpy()
    covariance = result.cov_params().loc[names, names].to_numpy()
    estimate = float(vector @ beta)
    se = float(np.sqrt(vector @ covariance @ vector))
    z = estimate / se
    return {
        "experiment": experiment,
        "analysis": label,
        "contrast": "current coherence Low - High",
        "beta_seconds": estimate,
        "se": se,
        "ci_low": estimate - 1.96 * se,
        "ci_high": estimate + 1.96 * se,
        "z": z,
        "p": float(2 * norm.sf(abs(z))),
    }


def main() -> None:
    data = load_data()
    mixed_rows = []
    fit_rows = []

    # Matched within-experiment models.
    for experiment, d in data.groupby("experiment", sort=False):
        result, method, messages = fit_mixed(BASE_FORMULA, d, "~curDur_c + preDur_c")
        names = result.fe_params.index.tolist()
        vector = np.zeros(len(names))
        vector[names.index("current_low")] = 1.0
        mixed_rows.append(contrast_row(result, experiment, "within-experiment LMM", vector))
        fit_rows.append({
            "experiment": experiment,
            "analysis": "within-experiment LMM",
            "n_trials": int(result.nobs),
            "n_participants": int(d["participant"].nunique()),
            "optimizer": method,
            "converged": bool(result.converged),
            "warnings": messages,
        })

    # One combined model directly tests whether that contrast differs by experiment.
    combined = data.copy()
    combined["dynamic"] = (combined["experiment"] == "Exp1_dynamic").astype(int)
    combined_formula = (
        "curBias ~ dynamic * (curDur_c * previous_low * current_low "
        "+ preDur_c * previous_low * current_low)"
    )
    result, method, messages = fit_mixed(combined_formula, combined, "~curDur_c + preDur_c")
    names = result.fe_params.index.tolist()
    fixed_vector = np.zeros(len(names))
    fixed_vector[names.index("current_low")] = 1.0
    dynamic_vector = fixed_vector.copy()
    dynamic_vector[names.index("dynamic:current_low")] = 1.0
    difference_vector = np.zeros(len(names))
    difference_vector[names.index("dynamic:current_low")] = 1.0
    mixed_rows.extend([
        contrast_row(result, "Exp2_fixed", "combined LMM simple effect", fixed_vector),
        contrast_row(result, "Exp1_dynamic", "combined LMM simple effect", dynamic_vector),
        contrast_row(result, "Exp1_dynamic - Exp2_fixed", "combined LMM interaction", difference_vector),
    ])
    fit_rows.append({
        "experiment": "combined",
        "analysis": "combined LMM interaction",
        "n_trials": int(result.nobs),
        "n_participants": int(combined["participant"].nunique()),
        "optimizer": method,
        "converged": bool(result.converged),
        "warnings": messages,
    })

    # Participant-level OLS coefficients provide a distribution-free sensitivity check.
    participant_rows = []
    for (experiment, participant), d in data.groupby(["experiment", "participant"]):
        result = smf.ols(BASE_FORMULA, d).fit()
        participant_rows.append({
            "experiment": experiment,
            "participant": participant,
            "current_low_beta": float(result.params["current_low"]),
            "n_trials": int(result.nobs),
        })
    participant = pd.DataFrame(participant_rows)
    sensitivity_rows = []
    for experiment, d in participant.groupby("experiment", sort=False):
        test = ttest_1samp(d["current_low_beta"], 0.0)
        mean = float(d["current_low_beta"].mean())
        se = float(d["current_low_beta"].sem())
        sensitivity_rows.append({
            "contrast": f"{experiment}: Low - High",
            "n": len(d),
            "mean_beta_seconds": mean,
            "se": se,
            "ci_low": mean - 2.079614 * se,
            "ci_high": mean + 2.079614 * se,
            "t": float(test.statistic),
            "df": len(d) - 1,
            "p": float(test.pvalue),
        })
    dynamic = participant.loc[participant.experiment == "Exp1_dynamic", "current_low_beta"]
    fixed = participant.loc[participant.experiment == "Exp2_fixed", "current_low_beta"]
    test = ttest_ind(dynamic, fixed, equal_var=False)
    difference = float(dynamic.mean() - fixed.mean())
    se = float(np.sqrt(dynamic.var(ddof=1) / len(dynamic) + fixed.var(ddof=1) / len(fixed)))
    df = float((dynamic.var(ddof=1) / len(dynamic) + fixed.var(ddof=1) / len(fixed)) ** 2 /
               ((dynamic.var(ddof=1) / len(dynamic)) ** 2 / (len(dynamic) - 1) +
                (fixed.var(ddof=1) / len(fixed)) ** 2 / (len(fixed) - 1)))
    sensitivity_rows.append({
        "contrast": "Exp1_dynamic - Exp2_fixed",
        "n": len(dynamic) + len(fixed),
        "mean_beta_seconds": difference,
        "se": se,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "t": float(test.statistic),
        "df": df,
        "p": float(test.pvalue),
    })

    mixed = pd.DataFrame(mixed_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    mixed.to_csv(OUT / "lmm_contrasts.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(OUT / "lmm_fit_record.csv", index=False)
    participant.to_csv(OUT / "participant_coefficients.csv", index=False)
    sensitivity.to_csv(OUT / "participant_sensitivity.csv", index=False)
    print(mixed.to_string(index=False))
    print("\nParticipant sensitivity")
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
