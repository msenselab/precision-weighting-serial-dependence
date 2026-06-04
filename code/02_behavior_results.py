#!/usr/bin/env python3
"""Prepare behavioral result tables and figure-source data from cleaned data."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "experiment1": ROOT / "data" / "experiment1" / "E1.pkl",
    "experiment2": ROOT / "data" / "experiment2" / "E2.pkl",
}
OUTDIR = ROOT / "results" / "figure_source_data"
OUTDIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "experiment1": "Experiment 1",
    "experiment2": "Experiment 2",
}


def fit_mixedlm(df: pd.DataFrame, formula: str, re_formula: str = "1"):
    model = smf.mixedlm(formula, df, groups=df["subID"], re_formula=re_formula)
    last_error = None
    for method in ("lbfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(reml=True, method=method, disp=False, maxiter=3000)
            return result, method
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
    raise RuntimeError(f"MixedLM failed: {formula}") from last_error


def param_rows(result, dataset: str, model_name: str, formula: str, method: str) -> list[dict[str, object]]:
    rows = []
    for param in result.params.index:
        if param == "Group Var":
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "formula": formula,
                "parameter": param,
                "beta": float(result.params[param]),
                "se": float(result.bse[param]),
                "z": float(result.tvalues[param]),
                "p": float(result.pvalues[param]),
                "n_obs": int(result.nobs),
                "n_subjects": int(pd.Series(result.model.groups).nunique()),
                "converged": bool(result.converged),
                "fit_method": method,
            }
        )
    return rows


def prep_valid(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df[df["is_outlier"] == False].copy()
    df = df.dropna(subset=["subID", "curBias", "curDur", "preDur1back", "curCoherence", "preCoherence1back"])
    df["subID"] = df["subID"].astype(str)
    df["curDur_c"] = df["curDur"] - 1.2
    df["preDur_c"] = df["preDur1back"] - 1.2
    df["current_low_coherence"] = np.where(np.isclose(df["curCoherence"], 0.3), 1.0, 0.0)
    df["prior_low_coherence"] = np.where(np.isclose(df["preCoherence1back"], 0.3), 1.0, 0.0)
    df["same_transition"] = df["current_low_coherence"].eq(df["prior_low_coherence"]).astype(float)
    df["current_coherence_label"] = np.where(df["current_low_coherence"].eq(1.0), "Low", "High")
    df["same_switch_label"] = np.where(df["same_transition"].eq(1.0), "Same", "Switch")
    return df


def full_marked(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path).copy()
    df["subID"] = df["subID"].astype(str)
    df = df.sort_values(["subID", "trial_num"]).reset_index(drop=True)
    df["curDur_c"] = df["curDur"] - 1.2
    df["preDur_c"] = df["preDur1back"] - 1.2
    df["current_low_coherence"] = np.where(np.isclose(df["curCoherence"], 0.3), 1.0, 0.0)
    df["prior_low_coherence"] = np.where(np.isclose(df["preCoherence1back"], 0.3), 1.0, 0.0)
    df["same_transition"] = df["current_low_coherence"].eq(df["prior_low_coherence"]).astype(float)
    return df


def simple_slope(result, base: str, interaction: str) -> tuple[float, float, float, float]:
    cov = result.cov_params()
    beta = result.params[base] + result.params[interaction]
    se = np.sqrt(cov.loc[base, base] + cov.loc[interaction, interaction] + 2 * cov.loc[base, interaction])
    z = beta / se
    p = 2 * stats.norm.sf(abs(z))
    return float(beta), float(se), float(z), float(p)


def dual_lmms(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    slope_rows = []

    specs = {
        "experiment1": {
            "formula": "curBias ~ curDur_c * current_low_coherence + preDur_c * current_low_coherence",
            "moderator": "current_low_coherence",
            "reference": "High coherence",
            "moderated": "Low coherence",
        },
        "experiment2": {
            "formula": "curBias ~ curDur_c * same_transition + preDur_c * same_transition",
            "moderator": "same_transition",
            "reference": "Switch",
            "moderated": "Same",
        },
    }

    for dataset, spec in specs.items():
        df = datasets[dataset]
        result, method = fit_mixedlm(df, spec["formula"], "~curDur_c + preDur_c")
        rows.extend(param_rows(result, dataset, "primary_dual_interaction_lmm", spec["formula"], method))

        for base, channel in [("curDur_c", "current_duration"), ("preDur_c", "previous_duration")]:
            interaction = f"{base}:{spec['moderator']}"
            mod_beta, mod_se, mod_z, mod_p = simple_slope(result, base, interaction)
            slope_rows.extend(
                [
                    {
                        "dataset": dataset,
                        "channel": channel,
                        "condition": spec["reference"],
                        "beta": float(result.params[base]),
                        "se": float(result.bse[base]),
                        "z": float(result.tvalues[base]),
                        "p": float(result.pvalues[base]),
                        "n_obs": int(result.nobs),
                    },
                    {
                        "dataset": dataset,
                        "channel": channel,
                        "condition": spec["moderated"],
                        "beta": mod_beta,
                        "se": mod_se,
                        "z": mod_z,
                        "p": mod_p,
                        "n_obs": int(result.nobs),
                    },
                    {
                        "dataset": dataset,
                        "channel": channel,
                        "condition": f"{spec['moderated']} - {spec['reference']}",
                        "beta": float(result.params[interaction]),
                        "se": float(result.bse[interaction]),
                        "z": float(result.tvalues[interaction]),
                        "p": float(result.pvalues[interaction]),
                        "n_obs": int(result.nobs),
                    },
                ]
            )

    return pd.DataFrame(rows), pd.DataFrame(slope_rows)


def subject_slopes(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    group_cols = ["current_coherence_label", "same_switch_label"]
    for dataset, df in datasets.items():
        for grouping in group_cols:
            for (sub_id, condition), dd in df.groupby(["subID", grouping]):
                if len(dd) < 10 or dd["curDur"].nunique() < 2 or dd["preDur1back"].nunique() < 2:
                    continue
                X = sm.add_constant(dd[["curDur", "preDur1back"]], has_constant="add")
                res = sm.OLS(dd["curBias"], X).fit()
                rows.append(
                    {
                        "dataset": dataset,
                        "grouping": grouping,
                        "subID": sub_id,
                        "condition": condition,
                        "n_trials": int(len(dd)),
                        "ct_slope": float(res.params["curDur"]),
                        "sdi_slope": float(res.params["preDur1back"]),
                    }
                )

    slopes = pd.DataFrame(rows)
    contrast_specs = [
        ("experiment1", "current_coherence_label", "Low", "High"),
        ("experiment2", "same_switch_label", "Same", "Switch"),
        ("experiment1", "same_switch_label", "Same", "Switch"),
        ("experiment2", "current_coherence_label", "Low", "High"),
    ]
    contrasts = []
    for dataset, grouping, a, b in contrast_specs:
        dd = slopes[(slopes["dataset"].eq(dataset)) & (slopes["grouping"].eq(grouping))]
        wide = dd.pivot(index="subID", columns="condition", values="sdi_slope")
        if a not in wide or b not in wide:
            continue
        wide = wide[[a, b]].dropna()
        diff = wide[a] - wide[b]
        t, p = stats.ttest_rel(wide[a], wide[b])
        contrasts.append(
            {
                "dataset": dataset,
                "grouping": grouping,
                "contrast": f"{a} - {b}",
                "N": int(len(wide)),
                "mean_a": float(wide[a].mean()),
                "sem_a": float(wide[a].sem()),
                "mean_b": float(wide[b].mean()),
                "sem_b": float(wide[b].sem()),
                "mean_diff": float(diff.mean()),
                "sem_diff": float(diff.sem()),
                "t": float(t),
                "df": int(len(wide) - 1),
                "p": float(p),
                "dz": float(diff.mean() / diff.std(ddof=1)),
            }
        )
    return slopes, pd.DataFrame(contrasts)


def add_response_error_lags(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    df = df.sort_values(["subID", "trial_num"]).copy()
    response_invalid = (
        df["objective_outlier"] | df["sd_outlier_final_sample"] | df["rpr"].isna()
    )
    df["response_error_for_lag"] = df["curBias"].where(~response_invalid)
    df["rpr_for_lag"] = df["rpr"].where(~response_invalid)
    for lag in range(1, max_lag + 1):
        df[f"response_error_lag{lag}"] = df.groupby("subID")["response_error_for_lag"].shift(lag)
        df[f"rpr_lag{lag}"] = df.groupby("subID")["rpr_for_lag"].shift(lag)
        within_block = pd.to_numeric(df["trial_in_block"], errors="coerce") >= lag
        df.loc[~within_block, [f"response_error_lag{lag}", f"rpr_lag{lag}"]] = np.nan
    return df


def response_error_models(marked_datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    param_tables = []
    vif_rows = []
    corr_rows = []
    predictors = [f"response_error_lag{i}" for i in range(1, 6)]
    formula = "curBias ~ curDur_c + preDur_c + " + " + ".join(predictors)

    for dataset, df in marked_datasets.items():
        df = add_response_error_lags(df)
        model_df = df[df["is_outlier"] == False].dropna(
            subset=["curBias", "curDur_c", "preDur_c"] + predictors
        ).copy()
        result, method = fit_mixedlm(model_df, formula, "1")
        param_tables.extend(param_rows(result, dataset, "response_error_lag_1_to_5", formula, method))

        corr_df = df[df["is_outlier"] == False].dropna(
            subset=["preDur1back", "rpr_lag1", "response_error_lag1"]
        )
        corr_rows.append(
            {
                "dataset": dataset,
                "n": int(len(corr_df)),
                "r_preDur_prevRpr": float(corr_df["preDur1back"].corr(corr_df["rpr_lag1"])),
                "r_preDur_prevResponseError": float(
                    corr_df["preDur1back"].corr(corr_df["response_error_lag1"])
                ),
            }
        )

        X = model_df[["curDur_c", "preDur_c"] + predictors].astype(float)
        X = sm.add_constant(X, has_constant="add")
        for i, col in enumerate(X.columns):
            if col == "const":
                continue
            vif_rows.append(
                {
                    "dataset": dataset,
                    "n": int(len(model_df)),
                    "predictor": col,
                    "vif": float(variance_inflation_factor(X.values, i)),
                }
            )

    return pd.DataFrame(param_tables), pd.DataFrame(corr_rows), pd.DataFrame(vif_rows)


def descriptives(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in datasets.items():
        per_sub = df.groupby("subID").size()
        rows.append(
            {
                "dataset": dataset,
                "n_subjects": int(df["subID"].nunique()),
                "n_trials": int(len(df)),
                "trials_min": int(per_sub.min()),
                "trials_mean": float(per_sub.mean()),
                "trials_sd": float(per_sub.std(ddof=1)),
                "trials_max": int(per_sub.max()),
            }
        )
    return pd.DataFrame(rows)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}".replace("0.", ".")


def write_markdown(
    desc: pd.DataFrame,
    lmm_slopes: pd.DataFrame,
    subject_contrasts: pd.DataFrame,
    response_params: pd.DataFrame,
    corrs: pd.DataFrame,
    vifs: pd.DataFrame,
) -> None:
    lines = ["# Behavioral Result Tables", ""]

    lines.append("## Descriptives")
    for _, r in desc.iterrows():
        lines.append(
            f"- {DISPLAY[r['dataset']]}: N = {int(r['n_subjects'])}, trials = {int(r['n_trials'])}, "
            f"trials/sub = {r['trials_mean']:.1f} [{int(r['trials_min'])}, {int(r['trials_max'])}]."
        )

    lines.extend(["", "## Primary Dual-Interaction LMMs"])
    for dataset in DATASETS:
        lines.append(f"### {DISPLAY[dataset]}")
        dd = lmm_slopes[lmm_slopes["dataset"].eq(dataset)]
        for _, r in dd.iterrows():
            lines.append(
                f"- {r['channel']} / {r['condition']}: beta = {r['beta']:.3f}, "
                f"SE = {r['se']:.3f}, z = {r['z']:.2f}, p {fmt_p(r['p'])}."
            )

    lines.extend(["", "## Participant-Level Controlled SDI Contrasts"])
    for _, r in subject_contrasts.iterrows():
        lines.append(
            f"- {DISPLAY[r['dataset']]} {r['contrast']}: mean diff = {r['mean_diff']:.4f}, "
            f"t({int(r['df'])}) = {r['t']:.2f}, p {fmt_p(r['p'])}, dz = {r['dz']:.3f}."
        )

    lines.extend(["", "## Response-Error Carryover"])
    response_terms = response_params[response_params["parameter"].str.startswith("response_error_lag")]
    for dataset in DATASETS:
        lines.append(f"### {DISPLAY[dataset]}")
        dd = response_terms[response_terms["dataset"].eq(dataset)]
        n_obs = int(dd["n_obs"].iloc[0])
        lines.append(f"- Model N = {n_obs} trials after requiring lag-5 availability.")
        for _, r in dd.iterrows():
            lag = r["parameter"].replace("response_error_lag", "lag ")
            lines.append(
                f"- {lag}: beta = {r['beta']:.3f}, SE = {r['se']:.3f}, "
                f"z = {r['z']:.2f}, p {fmt_p(r['p'])}."
            )

    lines.extend(["", "## Collinearity / VIF"])
    for _, r in corrs.iterrows():
        max_vif = vifs[vifs["dataset"].eq(r["dataset"])]["vif"].max()
        lines.append(
            f"- {DISPLAY[r['dataset']]}: r(previous duration, previous reproduced duration) = "
            f"{r['r_preDur_prevRpr']:.3f}; r(previous duration, previous response error) = "
            f"{r['r_preDur_prevResponseError']:.3f}; largest VIF = {max_vif:.2f}."
        )

    (OUTDIR / "behavior_results_summary.md").write_text("\n".join(lines))


def main() -> None:
    valid = {name: prep_valid(path) for name, path in DATASETS.items()}
    marked = {name: full_marked(path) for name, path in DATASETS.items()}

    desc = descriptives(valid)
    lmm_params, lmm_slopes = dual_lmms(valid)
    subj_slopes, subj_contrasts = subject_slopes(valid)
    response_params, response_corrs, response_vifs = response_error_models(marked)

    desc.to_csv(OUTDIR / "descriptives.csv", index=False, float_format="%.10g")
    lmm_params.to_csv(OUTDIR / "primary_dual_lmm_parameters.csv", index=False, float_format="%.10g")
    lmm_slopes.to_csv(OUTDIR / "primary_dual_lmm_slopes.csv", index=False, float_format="%.10g")
    subj_slopes.to_csv(OUTDIR / "participant_controlled_slopes.csv", index=False, float_format="%.10g")
    subj_contrasts.to_csv(OUTDIR / "participant_controlled_slope_contrasts.csv", index=False, float_format="%.10g")
    response_params.to_csv(OUTDIR / "response_error_lag_lmm_parameters.csv", index=False, float_format="%.10g")
    response_corrs.to_csv(OUTDIR / "response_error_collinearity_correlations.csv", index=False, float_format="%.10g")
    response_vifs.to_csv(OUTDIR / "response_error_vif.csv", index=False, float_format="%.10g")

    write_markdown(desc, lmm_slopes, subj_contrasts, response_params, response_corrs, response_vifs)
    print((OUTDIR / "behavior_results_summary.md").read_text())


if __name__ == "__main__":
    main()
