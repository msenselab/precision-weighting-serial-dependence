#!/usr/bin/env python3
"""Validation checks for the winning three-state Kalman model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models"
sys.path.insert(0, str(MODEL_ROOT))

from three_state_kalman.engine import generate_predictions  # noqa: E402
from three_state_kalman.parameters import get_parameter_config  # noqa: E402


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.nanstd(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan, np.nan
    r, p = pearsonr(x[ok], y[ok])
    return float(r), float(p)


def model_parts(model_id):
    c_id, s_id, b_id = model_id.split("_")
    return c_id, s_id, b_id


def load_inputs(fits_dir, winner_model_id=None):
    data = pd.read_csv(fits_dir / "kalman_input_data.csv")
    fits = pd.read_csv(fits_dir / "model_fits_deduplicated.csv")
    if winner_model_id:
        winner_models = {
            int(exp): {
                "model_id": winner_model_id,
                "model_name": fits.loc[fits["model_id"].eq(winner_model_id), "model_name"].dropna().iloc[0],
            }
            for exp in sorted(data["exp"].unique())
        }
    else:
        ranking = (
            fits.groupby(["exp", "model_id", "model_name"], as_index=False)
            .agg(mean_AIC=("AIC", "mean"))
            .sort_values(["exp", "mean_AIC"])
        )
        winner_models = {
            int(exp): {
                "model_id": d.iloc[0]["model_id"],
                "model_name": d.iloc[0]["model_name"],
            }
            for exp, d in ranking.groupby("exp", sort=True)
        }

    winner_rows = []
    for exp, info in winner_models.items():
        winner_rows.append(
            fits[(fits["exp"].eq(exp)) & (fits["model_id"].eq(info["model_id"])) & (fits["success"] == True)]
        )
    winner = pd.concat(winner_rows, ignore_index=True)
    missing = sorted(set(zip(data["exp"], data["Sub"])) - set(zip(winner["exp"], winner["Sub"])))
    if missing:
        raise RuntimeError(f"Missing winner fits for {len(missing)} subject-experiment cells")
    return data, winner, winner_models


def add_predictions(data, winner, winner_models):
    rows = []
    for (exp, sub), d in data.groupby(["exp", "Sub"], sort=True):
        fit = winner[(winner["exp"] == exp) & (winner["Sub"] == sub)].iloc[0]
        model_id = winner_models[int(exp)]["model_id"]
        c_id, s_id, b_id = model_parts(model_id)
        param_names = get_parameter_config(c_id, s_id, b_id)["names"]
        par = [fit[name] for name in param_names]
        stimrep = d[["Duration", "Reproduction"]].to_numpy(float)
        coherence = d["coherence"].to_numpy(float)
        structure = d["Structure"].to_numpy(str)
        tracking = generate_predictions(par, stimrep, coherence, structure, c_id, s_id, b_id)
        out = d.copy()
        out["pred"] = tracking["pred_orig"]
        out["pred_bias"] = out["pred"] - out["Duration"]
        out["obs_bias"] = out["Reproduction"] - out["Duration"]
        out["resid"] = out["Reproduction"] - out["pred"]
        out["mu_post"] = tracking["mu_post"]
        out["m_post"] = tracking["m_post"]
        out["b_post"] = tracking["b_post"]
        out["K_mu"] = tracking["K_mu"]
        out["q1_eff"] = tracking["q1_eff"]
        out["R_eff"] = tracking["R_eff"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def subject_metrics(ppc):
    rows = []
    for (exp, sub), d in ppc.groupby(["exp", "Sub"], sort=True):
        prev = d["Duration"].shift(1)
        rows.append({
            "exp": exp,
            "Sub": sub,
            "cti_obs": slope(d["Duration"], d["Reproduction"]),
            "cti_pred": slope(d["Duration"], d["pred"]),
            "sdi_obs": slope(prev, d["obs_bias"]),
            "sdi_pred": slope(prev, d["pred_bias"]),
            "rmse": float(np.sqrt(np.mean(np.square(d["resid"])))),
            "mae": float(np.mean(np.abs(d["resid"]))),
        })
    return pd.DataFrame(rows)


def controlled_slope_by_condition(frame, response_col):
    rows = []
    for (exp, sub), d in frame.groupby(["exp", "Sub"], sort=True):
        tmp = d.copy()
        tmp["bias"] = tmp[response_col] - tmp["Duration"]
        tmp["cur_c"] = tmp["Duration"] - tmp["Duration"].mean()
        tmp["pre_c"] = tmp["Duration"].shift(1) - tmp["Duration"].mean()
        tmp["high_uncertainty"] = np.where(tmp["coherence"] < 0.5, "HighUncertainty", "LowUncertainty")
        for grouping, col, conds in [
            ("current_uncertainty", "high_uncertainty", ["HighUncertainty", "LowUncertainty"]),
            ("SameSwitch", "Structure", ["Repeat", "Switch"]),
        ]:
            for cond in conds:
                dd = tmp[tmp[col] == cond]
                # Residualize current-duration central tendency by including cur_c
                ok = dd[["bias", "cur_c", "pre_c"]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(ok) < 12:
                    val = np.nan
                else:
                    X = np.column_stack([np.ones(len(ok)), ok["cur_c"], ok["pre_c"]])
                    beta = np.linalg.lstsq(X, ok["bias"], rcond=None)[0]
                    val = float(beta[2])
                rows.append({
                    "exp": exp,
                    "Sub": sub,
                    "source": response_col,
                    "grouping": grouping,
                    "condition": cond,
                    "controlled_sdi": val,
                })
    return pd.DataFrame(rows)


def summarize_effect_recovery(effects):
    rows = []
    for (exp, source, grouping, condition), d in effects.groupby(
        ["exp", "source", "grouping", "condition"], sort=True
    ):
        rows.append({
            "exp": exp,
            "source": source,
            "grouping": grouping,
            "condition": condition,
            "mean": d["controlled_sdi"].mean(),
            "sem": d["controlled_sdi"].sem(),
            "n_subjects": d["controlled_sdi"].notna().sum(),
        })
    summary = pd.DataFrame(rows)

    contrast_rows = []
    contrasts = [
        (1, "current_uncertainty", "HighUncertainty", "LowUncertainty", "Experiment 1 High-Low"),
        (2, "SameSwitch", "Repeat", "Switch", "Experiment 2 Same-Switch"),
    ]
    for exp, grouping, a, b, label in contrasts:
        wide = effects[(effects["exp"] == exp) & (effects["grouping"] == grouping)].pivot_table(
            index=["Sub", "source"], columns="condition", values="controlled_sdi"
        ).reset_index()
        for source, d in wide.groupby("source"):
            diff = d[a] - d[b]
            contrast_rows.append({
                "exp": exp,
                "source": source,
                "grouping": grouping,
                "contrast": label,
                "mean_diff": diff.mean(),
                "sem_diff": diff.sem(),
                "n_subjects": diff.notna().sum(),
            })
    contrasts_df = pd.DataFrame(contrast_rows)
    return summary, contrasts_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits-dir", type=Path, default=ROOT / "results" / "kalman_model_fits")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "kalman_model_checks")
    parser.add_argument(
        "--winner-model-id",
        default=None,
        help="Use one fixed winner model for all experiments. Default: best AIC model within each experiment.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fits_dir = args.fits_dir
    if not fits_dir.is_absolute():
        fits_dir = ROOT / fits_dir
    data, winner, winner_models = load_inputs(fits_dir, args.winner_model_id)
    ppc = add_predictions(data, winner, winner_models)
    ppc.to_csv(out_dir / "winner_ppc_trial_predictions.csv", index=False, float_format="%.12g")

    metrics = subject_metrics(ppc)
    metrics.to_csv(out_dir / "winner_ppc_subject_metrics.csv", index=False, float_format="%.12g")

    metric_summary_rows = []
    for exp, d in metrics.groupby("exp", sort=True):
        cti_r, cti_p = corr(d["cti_obs"], d["cti_pred"])
        sdi_r, sdi_p = corr(d["sdi_obs"], d["sdi_pred"])
        metric_summary_rows.append({
            "exp": exp,
            "model_id": winner_models[int(exp)]["model_id"],
            "model_name": winner_models[int(exp)]["model_name"],
            "cti_r": cti_r,
            "cti_p": cti_p,
            "sdi_r": sdi_r,
            "sdi_p": sdi_p,
            "cti_mean_obs": d["cti_obs"].mean(),
            "cti_mean_pred": d["cti_pred"].mean(),
            "sdi_mean_obs": d["sdi_obs"].mean(),
            "sdi_mean_pred": d["sdi_pred"].mean(),
            "rmse_mean": d["rmse"].mean(),
            "mae_mean": d["mae"].mean(),
            "n_subjects": d["Sub"].nunique(),
        })
    metric_summary = pd.DataFrame(metric_summary_rows)
    metric_summary.to_csv(out_dir / "winner_ppc_summary.csv", index=False, float_format="%.12g")

    obs_effects = controlled_slope_by_condition(data, "Reproduction")
    pred_effects = controlled_slope_by_condition(ppc, "pred")
    effects = pd.concat([obs_effects, pred_effects], ignore_index=True)
    effects.to_csv(out_dir / "winner_behavioral_effects_subject.csv", index=False, float_format="%.12g")
    effect_summary, effect_contrasts = summarize_effect_recovery(effects)
    effect_summary.to_csv(out_dir / "winner_behavioral_effect_summary.csv", index=False, float_format="%.12g")
    effect_contrasts.to_csv(out_dir / "winner_behavioral_effect_contrasts.csv", index=False, float_format="%.12g")

    # Console-only summary (no summary.md / summary.json are written; the CSV
    # outputs above are the canonical machine-readable artifacts).
    lines = ["# Kalman Model Validation", ""]
    lines.append("Winner models:")
    for exp, info in winner_models.items():
        label = f"Experiment {exp}"
        lines.append(f"- {label}: `{info['model_id']}` / `{info['model_name']}`")
    lines.append("")
    lines.append("## PPC subject-level CTI/SDI recovery")
    for row in metric_summary.to_dict(orient="records"):
        label = f"Experiment {int(row['exp'])}"
        lines.append(
            f"- {label}: CTI r={row['cti_r']:.3f}, SDI r={row['sdi_r']:.3f}, "
            f"mean RMSE={row['rmse_mean']:.4f}"
        )
    lines.append("")
    lines.append("## Behavioral effect recovery")
    for row in effect_contrasts.to_dict(orient="records"):
        label = "observed" if row["source"] == "Reproduction" else "predicted"
        lines.append(
            f"- {row['contrast']} ({label}): mean diff={row['mean_diff']:.4f}, SEM={row['sem_diff']:.4f}"
        )
    lines.append("")
    lines.append(f"All outputs are in `{display_path(out_dir)}`.")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
