#!/usr/bin/env python3
"""Block-safe response-error history analysis for the revision.

Lag variables are created on the complete trial sequence within participant and
block. Outlier/missing responses are then masked, never removed before shifting.
This prevents false adjacency across omitted trials or block boundaries.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sdu import paths as SP  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = SP.RESULTS_BEHAVIOR / "analysis" / "response_history"
OUT.mkdir(parents=True, exist_ok=True)

DATA = {
    "Exp1_dynamic": SP.PREPROCESSED / "E2_reprocessed.csv",
    "Exp2_fixed": SP.PREPROCESSED / "E1_reprocessed.csv",
}


def prepare(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"subID": str}).sort_values(["subID", "trial_num"]).copy()
    # Every trial_in_block==0 starts a new block; create an explicit grouping key.
    d["block"] = d.groupby("subID")["trial_in_block"].transform(lambda x: x.eq(0).cumsum())
    d["response_valid"] = d["rpr"].where(~d["is_outlier"])
    d["response_error"] = d["response_valid"] - d["curDur"]
    grouped = d.groupby(["subID", "block"], sort=False)
    for lag in range(1, 6):
        d[f"response_error_lag{lag}"] = grouped["response_error"].shift(lag)
        d[f"duration_lag{lag}"] = grouped["curDur"].shift(lag)
    d["curDur_c"] = d["curDur"] - 1.2
    d["preDur_c"] = d["duration_lag1"] - 1.2
    required = ["curBias", "curDur_c", "preDur_c"] + [f"response_error_lag{k}" for k in range(1, 6)]
    # The current outcome is invalid on outlier trials even if curBias is numeric.
    d.loc[d["is_outlier"], "curBias"] = np.nan
    return d.dropna(subset=required).copy()


def fit_lmm(d: pd.DataFrame):
    lag_terms = " + ".join(f"response_error_lag{k}" for k in range(1, 6))
    formula = f"curBias ~ curDur_c + preDur_c + {lag_terms}"
    model = smf.mixedlm(formula, d, groups=d["subID"], re_formula="1")
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
    raise RuntimeError("LMM failed: " + " | ".join(messages))


def vif_rows(d: pd.DataFrame, experiment: str) -> list[dict]:
    predictors = ["curDur_c", "preDur_c"] + [f"response_error_lag{k}" for k in range(1, 6)]
    x = d[predictors].copy()
    x = (x - x.mean()) / x.std(ddof=0)
    x.insert(0, "Intercept", 1.0)
    return [
        {
            "experiment": experiment,
            "predictor": predictor,
            "vif": float(variance_inflation_factor(x.to_numpy(), index)),
        }
        for index, predictor in enumerate(x.columns)
        if predictor != "Intercept"
    ]


def main() -> None:
    parameter_rows = []
    sensitivity_rows = []
    fit_rows = []
    vif = []
    correlation_rows = []
    prepared = []

    for experiment, path in DATA.items():
        d = prepare(path)
        d["experiment"] = experiment
        prepared.append(d)
        result, method, messages = fit_lmm(d)
        for parameter in result.fe_params.index:
            parameter_rows.append({
                "experiment": experiment,
                "parameter": parameter,
                "beta": float(result.fe_params[parameter]),
                "se": float(result.bse_fe[parameter]),
                "z": float(result.tvalues[parameter]),
                "p": float(result.pvalues[parameter]),
            })
        fit_rows.append({
            "experiment": experiment,
            "n_trials": int(result.nobs),
            "n_participants": int(d["subID"].nunique()),
            "optimizer": method,
            "converged": bool(result.converged),
            "warnings": messages,
        })
        lag_terms = " + ".join(f"response_error_lag{k}" for k in range(1, 6))
        formula = f"curBias ~ curDur_c + preDur_c + {lag_terms}"
        # Participant fixed intercepts plus participant-clustered SEs provide a
        # sensitivity check for the random-intercept specification.
        cluster = smf.ols(formula + " + C(subID)", d).fit(
            cov_type="cluster", cov_kwds={"groups": d["subID"]}
        )
        reported = ["curDur_c", "preDur_c"] + [f"response_error_lag{k}" for k in range(1, 6)]
        for parameter in reported:
            sensitivity_rows.append({
                "experiment": experiment,
                "specification": "participant fixed effects + cluster-robust SE",
                "parameter": parameter,
                "beta": float(cluster.params[parameter]),
                "se": float(cluster.bse[parameter]),
                "z": float(cluster.tvalues[parameter]),
                "p": float(cluster.pvalues[parameter]),
            })
        vif.extend(vif_rows(d, experiment))
        correlation_rows.extend([
            {
                "experiment": experiment,
                "variables": "previous duration vs previous reproduced duration",
                "r": float(d["duration_lag1"].corr(d["duration_lag1"] + d["response_error_lag1"])),
            },
            {
                "experiment": experiment,
                "variables": "previous duration vs previous response error",
                "r": float(d["duration_lag1"].corr(d["response_error_lag1"])),
            },
        ])

    parameters = pd.DataFrame(parameter_rows)
    fits = pd.DataFrame(fit_rows)
    vifs = pd.DataFrame(vif)
    correlations = pd.DataFrame(correlation_rows)
    parameters.to_csv(OUT / "lmm_parameters.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "cluster_robust_sensitivity.csv", index=False)
    fits.to_csv(OUT / "lmm_fit_record.csv", index=False)
    vifs.to_csv(OUT / "predictor_vif.csv", index=False)
    correlations.to_csv(OUT / "lag1_correlations.csv", index=False)
    pd.concat(prepared, ignore_index=True)[
        ["experiment", "subID", "trial_num", "trial_in_block", "curBias", "curDur_c", "preDur_c"]
        + [f"response_error_lag{k}" for k in range(1, 6)]
    ].to_csv(OUT / "analysis_rows.csv", index=False)
    print(parameters.to_string(index=False))
    print("\nFit records")
    print(fits.to_string(index=False))
    print("\nMaximum VIF by experiment")
    print(vifs.groupby("experiment")["vif"].max().to_string())
    print("\nLag-1 correlations")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
