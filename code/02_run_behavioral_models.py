#!/usr/bin/env python3
"""Run the primary behavioral mixed-effects analyses on cleaned data."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "experiment1_dynamic": ROOT / "data" / "experiment2" / "E2.pkl",
    "experiment2_fixed": ROOT / "data" / "experiment1" / "E1.pkl",
}
OUTDIR = ROOT / "results" / "behavioral_models"
OUTDIR.mkdir(parents=True, exist_ok=True)


def prep(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df[df["is_outlier"] == False].copy()
    df = df.dropna(
        subset=["curBias", "curDur", "preDur1back", "curCoherence", "preCoherence1back", "subID"]
    )
    df["subID"] = df["subID"].astype(str)
    df["curDur_c"] = df["curDur"] - 1.2
    df["preDur_c"] = df["preDur1back"] - 1.2
    df["current_high_uncertainty"] = np.where(np.isclose(df["curCoherence"], 0.3), 1.0, 0.0)
    df["prior_high_uncertainty"] = np.where(
        np.isclose(df["preCoherence1back"], 0.3), 1.0, 0.0
    )
    df["same_transition"] = (
        df["current_high_uncertainty"].eq(df["prior_high_uncertainty"]).astype(float)
    )
    df["SameSwitch"] = np.where(df["same_transition"].eq(1.0), "Same", "Switch")
    df["current_uncertainty"] = np.where(
        df["current_high_uncertainty"].eq(1.0), "HighUncertainty", "LowUncertainty"
    )
    df["prior_uncertainty"] = np.where(
        df["prior_high_uncertainty"].eq(1.0), "HighUncertainty", "LowUncertainty"
    )
    df["preResp_long"] = df["preResp"].map({"Short": 0.0, "Long": 1.0})
    return df


def fit_mixedlm(df: pd.DataFrame, formula: str, re_formula: str):
    model = smf.mixedlm(formula, df, groups=df["subID"], re_formula=re_formula)
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(reml=True, method=method, disp=False, maxiter=2000)
            return result, method
        except Exception:
            continue
    raise RuntimeError(f"MixedLM failed for formula: {formula}")


def param_table(result, dataset: str, model: str, formula: str, method: str) -> pd.DataFrame:
    rows = []
    for param in result.params.index:
        if param == "Group Var":
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "formula": formula,
                "fit_method": method,
                "parameter": param,
                "beta": float(result.params[param]),
                "se": float(result.bse[param]),
                "z": float(result.tvalues[param]),
                "p": float(result.pvalues[param]),
                "n_obs": int(result.nobs),
                "n_subjects": int(result.model.groups.shape[0]),
                "converged": bool(result.converged),
            }
        )
    return pd.DataFrame(rows)


def simple_slope(result, base: str, interaction: str) -> tuple[float, float]:
    cov = result.cov_params()
    beta = result.params[base] + result.params[interaction]
    se = np.sqrt(cov.loc[base, base] + cov.loc[interaction, interaction] + 2 * cov.loc[base, interaction])
    return float(beta), float(se)


def run_main_lmms(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = []
    key = []

    df1 = datasets["experiment1_dynamic"]
    formula1 = "curBias ~ curDur_c + preDur_c * current_high_uncertainty"
    res1, method1 = fit_mixedlm(df1, formula1, "~curDur_c + preDur_c")
    tables.append(param_table(res1, "experiment1_dynamic", "main_current_uncertainty", formula1, method1))
    high_slope, high_se = simple_slope(res1, "preDur_c", "preDur_c:current_high_uncertainty")
    key.append(
        {
            "dataset": "experiment1_dynamic",
            "question": "current uncertainty moderation",
            "n_obs": int(res1.nobs),
            "n_subjects": int(df1["subID"].nunique()),
            "baseline": "low uncertainty / 70% coherence",
            "baseline_preDur_slope": float(res1.params["preDur_c"]),
            "baseline_se": float(res1.bse["preDur_c"]),
            "moderated": "high uncertainty / 30% coherence",
            "moderated_preDur_slope": high_slope,
            "moderated_se": high_se,
            "interaction": "preDur_c:current_high_uncertainty",
            "interaction_beta": float(res1.params["preDur_c:current_high_uncertainty"]),
            "interaction_se": float(res1.bse["preDur_c:current_high_uncertainty"]),
            "interaction_z": float(res1.tvalues["preDur_c:current_high_uncertainty"]),
            "interaction_p": float(res1.pvalues["preDur_c:current_high_uncertainty"]),
            "converged": bool(res1.converged),
            "fit_method": method1,
        }
    )

    df2 = datasets["experiment2_fixed"]
    formula2 = "curBias ~ curDur_c + preDur_c * same_transition"
    res2, method2 = fit_mixedlm(df2, formula2, "~curDur_c + preDur_c")
    tables.append(param_table(res2, "experiment2_fixed", "main_same_switch", formula2, method2))
    same_slope, same_se = simple_slope(res2, "preDur_c", "preDur_c:same_transition")
    key.append(
        {
            "dataset": "experiment2_fixed",
            "question": "same/switch moderation",
            "n_obs": int(res2.nobs),
            "n_subjects": int(df2["subID"].nunique()),
            "baseline": "Switch",
            "baseline_preDur_slope": float(res2.params["preDur_c"]),
            "baseline_se": float(res2.bse["preDur_c"]),
            "moderated": "Same",
            "moderated_preDur_slope": same_slope,
            "moderated_se": same_se,
            "interaction": "preDur_c:same_transition",
            "interaction_beta": float(res2.params["preDur_c:same_transition"]),
            "interaction_se": float(res2.bse["preDur_c:same_transition"]),
            "interaction_z": float(res2.tvalues["preDur_c:same_transition"]),
            "interaction_p": float(res2.pvalues["preDur_c:same_transition"]),
            "converged": bool(res2.converged),
            "fit_method": method2,
        }
    )

    return pd.concat(tables, ignore_index=True), pd.DataFrame(key)


def fit_subject_ols(df: pd.DataFrame, predictors: list[str]) -> dict[str, float] | None:
    d = df.dropna(subset=["curBias"] + predictors)
    if len(d) < max(10, len(predictors) + 2):
        return None
    if any(d[p].nunique() < 2 for p in predictors):
        return None
    X = sm.add_constant(d[predictors], has_constant="add")
    res = sm.OLS(d["curBias"], X).fit()
    out = {"n_trials": int(len(d)), "r2": float(res.rsquared)}
    for predictor in predictors:
        out[f"b_{predictor}"] = float(res.params[predictor])
        out[f"p_{predictor}"] = float(res.pvalues[predictor])
    return out


def summarize_1sample(values: pd.Series) -> dict[str, float]:
    v = values.dropna().astype(float)
    t, p = stats.ttest_1samp(v, 0.0)
    return {
        "N": int(v.size),
        "mean": float(v.mean()),
        "sem": float(v.sem()),
        "t_vs_0": float(t),
        "p_vs_0": float(p),
    }


def paired_test(df: pd.DataFrame, condition_col: str, value_col: str, a: str, b: str):
    wide = df.pivot(index="subID", columns=condition_col, values=value_col)
    if a not in wide or b not in wide:
        return None
    wide = wide[[a, b]].dropna()
    if len(wide) < 2:
        return None
    diff = wide[a] - wide[b]
    t, p = stats.ttest_rel(wide[a], wide[b])
    return {
        "contrast": f"{a} - {b}",
        "N": int(len(wide)),
        "mean_diff": float(diff.mean()),
        "sem_diff": float(diff.sem()),
        "t": float(t),
        "p": float(p),
        "dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan,
    }


def run_subject_slope_checks(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    group_specs = [
        ("overall", []),
        ("current_uncertainty", ["current_uncertainty"]),
        ("SameSwitch", ["SameSwitch"]),
        ("TransitionType", ["TransitionType"]),
    ]

    for dataset, df in datasets.items():
        for grouping, group_cols in group_specs:
            gb_cols = ["subID"] + group_cols
            for keys, subdf in df.groupby(gb_cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                joint = fit_subject_ols(subdf, ["curDur", "preDur1back"])
                if joint is None:
                    continue
                row = {"dataset": dataset, "grouping": grouping, "subID": keys[0]}
                for col, val in zip(group_cols, keys[1:]):
                    row[col] = val
                row.update(
                    {
                        "n_trials": joint["n_trials"],
                        "ct_slope_curDur": joint["b_curDur"],
                        "controlled_sdi_preDur": joint["b_preDur1back"],
                        "p_curDur": joint["p_curDur"],
                        "p_preDur": joint["p_preDur1back"],
                        "r2_joint": joint["r2"],
                    }
                )
                rows.append(row)

    slopes = pd.DataFrame(rows)
    summaries = []
    for (dataset, grouping), d in slopes.groupby(["dataset", "grouping"]):
        cond_cols = [c for c in ["current_uncertainty", "SameSwitch", "TransitionType"] if c in d and d[c].notna().any()]
        if cond_cols:
            cond_col = cond_cols[0]
            for cond, dd in d.groupby(cond_col):
                summaries.append(
                    {
                        "dataset": dataset,
                        "grouping": grouping,
                        "condition": cond,
                        "coefficient": "controlled_sdi_preDur",
                        **summarize_1sample(dd["controlled_sdi_preDur"]),
                    }
                )
        else:
            summaries.append(
                {
                    "dataset": dataset,
                    "grouping": grouping,
                    "condition": "overall",
                    "coefficient": "controlled_sdi_preDur",
                    **summarize_1sample(d["controlled_sdi_preDur"]),
                }
            )

    contrasts = []
    checks = [
        ("experiment1_dynamic", "current_uncertainty", "HighUncertainty", "LowUncertainty"),
        ("experiment2_fixed", "SameSwitch", "Same", "Switch"),
    ]
    for dataset, grouping, a, b in checks:
        d = slopes[(slopes["dataset"].eq(dataset)) & (slopes["grouping"].eq(grouping))]
        res = paired_test(d, grouping, "controlled_sdi_preDur", a, b)
        if res:
            contrasts.append(
                {
                    "dataset": dataset,
                    "grouping": grouping,
                    "coefficient": "controlled_sdi_preDur",
                    **res,
                }
            )

    return slopes, pd.DataFrame(summaries), pd.DataFrame(contrasts)


def run_response_history_lmms(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    specs = {
        "experiment1_dynamic": "curBias ~ curDur_c + preDur_c * curCoherence + preResp_long",
        "experiment2_fixed": "curBias ~ curDur_c + preDur_c + preResp_long * same_transition + curCoherence",
    }
    tables = []
    for dataset, formula in specs.items():
        df = datasets[dataset].dropna(subset=["preResp_long"]).copy()
        res, method = fit_mixedlm(df, formula, "1")
        tables.append(param_table(res, dataset, "response_history_sensitivity", formula, method))
    return pd.concat(tables, ignore_index=True)


def descriptive_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in datasets.items():
        rows.append(
            {
                "dataset": dataset,
                "n_subjects": int(df["subID"].nunique()),
                "n_trials": int(len(df)),
                "trials_per_subject_min": int(df.groupby("subID").size().min()),
                "trials_per_subject_mean": float(df.groupby("subID").size().mean()),
                "trials_per_subject_max": int(df.groupby("subID").size().max()),
                "mean_bias": float(df["curBias"].mean()),
                "sd_bias": float(df["curBias"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    descriptives: pd.DataFrame,
    main_key: pd.DataFrame,
    slope_contrasts: pd.DataFrame,
    response_table: pd.DataFrame,
) -> None:
    lines = ["# Behavioral Model Results", ""]
    lines.append("## Descriptives")
    for _, r in descriptives.iterrows():
        lines.append(
            f"- {r['dataset']}: N={int(r['n_subjects'])}, trials={int(r['n_trials'])}, "
            f"trials/sub={r['trials_per_subject_mean']:.1f} "
            f"[{int(r['trials_per_subject_min'])}, {int(r['trials_per_subject_max'])}]"
        )
    lines.append("")
    lines.append("## Main LMMs")
    for _, r in main_key.iterrows():
        lines.append(
            f"- {r['dataset']} ({r['question']}): {r['baseline']} slope={r['baseline_preDur_slope']:.4f}; "
            f"{r['moderated']} slope={r['moderated_preDur_slope']:.4f}; "
            f"interaction beta={r['interaction_beta']:.4f}, SE={r['interaction_se']:.4f}, "
            f"z={r['interaction_z']:.2f}, p={r['interaction_p']:.4g}; converged={r['converged']}"
        )
    lines.append("")
    lines.append("## Subject-Level Controlled SDI Contrasts")
    for _, r in slope_contrasts.iterrows():
        lines.append(
            f"- {r['dataset']} / {r['contrast']}: mean diff={r['mean_diff']:.4f}, "
            f"t({int(r['N']) - 1})={r['t']:.2f}, p={r['p']:.4g}, dz={r['dz']:.3f}"
        )
    lines.append("")
    lines.append("## Response History Key Terms")
    key_terms = response_table[
        response_table["parameter"].isin(
            ["preResp_long", "preResp_long:same_transition", "preDur_c:curCoherence"]
        )
    ]
    for _, r in key_terms.iterrows():
        lines.append(
            f"- {r['dataset']} `{r['parameter']}`: beta={r['beta']:.4f}, SE={r['se']:.4f}, "
            f"z={r['z']:.2f}, p={r['p']:.4g}"
        )
    lines.append("")
    lines.append("All outputs are in `results/behavioral_models`.")
    (OUTDIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    datasets = {name: prep(path) for name, path in DATASETS.items()}

    descriptives = descriptive_summary(datasets)
    descriptives.to_csv(OUTDIR / "descriptive_summary.csv", index=False)

    main_params, main_key = run_main_lmms(datasets)
    main_params.to_csv(OUTDIR / "main_lmm_parameter_table.csv", index=False)
    main_key.to_csv(OUTDIR / "main_lmm_key_results.csv", index=False)

    slopes, slope_summary, slope_contrasts = run_subject_slope_checks(datasets)
    slopes.to_csv(OUTDIR / "subject_level_controlled_slopes.csv", index=False)
    slope_summary.to_csv(OUTDIR / "subject_level_controlled_slope_summary.csv", index=False)
    slope_contrasts.to_csv(OUTDIR / "subject_level_controlled_slope_contrasts.csv", index=False)

    response_table = run_response_history_lmms(datasets)
    response_table.to_csv(OUTDIR / "response_history_lmm_parameter_table.csv", index=False)

    write_summary(descriptives, main_key, slope_contrasts, response_table)

    print((OUTDIR / "summary.md").read_text())


if __name__ == "__main__":
    main()
