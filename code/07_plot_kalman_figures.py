#!/usr/bin/env python3
"""Generate Kalman model figures from canonical fit and validation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FULL_DIR = ROOT / "results" / "kalman_model_fits"
VAL_DIR = ROOT / "results" / "kalman_model_checks"
DATA_DIR = ROOT / "data"
READY_DIR = ROOT / "results" / "figure_source_data"
FIG_DIR = ROOT / "figures"
READY_DIR.mkdir(parents=True, exist_ok=True)

WINNER = "C1_S0_B2"
CODE_TO_DISPLAY = {
    1: "Experiment 1",
    2: "Experiment 2",
}
FIGURE_ORDER = ["Experiment 1", "Experiment 2"]
COLORS = {
    "Experiment 1": "#4A9F79",
    "Experiment 2": "#5B7C99",
    "Observed": "#222222",
    "Predicted": "#8A6F3D",
}


def sem(x):
    x = pd.Series(x).dropna()
    return x.sem() if len(x) > 1 else np.nan


def add_display_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["display_experiment"] = df["exp"].map(CODE_TO_DISPLAY)
    return df


def load_valid_data(code_exp: int) -> pd.DataFrame:
    exp_dir = "experiment1" if code_exp == 1 else "experiment2"
    path = DATA_DIR / exp_dir / f"E{code_exp}.pkl"
    df = pd.read_pickle(path)
    df = df[df["is_outlier"] == False].copy()
    df["subID"] = df["subID"].astype(int)
    return df


def figure6_parameter_estimates():
    fits = pd.read_csv(FULL_DIR / "model_fits_deduplicated.csv")
    fits = add_display_label(fits[fits["model_id"].eq(WINNER)])
    params = ["q1", "q2", "q3", "lambda", "alpha_q1", "alpha_d0", "d0", "r_base"]
    labels = {
        "q1": "$q_1$",
        "q2": "$q_2$",
        "q3": "$q_3$",
        "lambda": "$\\lambda$",
        "alpha_q1": "$\\alpha_{q_1}$",
        "alpha_d0": "$\\alpha_{d_0}$",
        "d0": "$d_0$",
        "r_base": "$R_{base}$",
    }
    summary = (
        fits.groupby("display_experiment")[params]
        .agg(["mean", sem])
        .reindex(FIGURE_ORDER)
    )
    summary.to_csv(READY_DIR / "kalman_winner_parameter_summary.csv")

    fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.6), constrained_layout=True)
    for ax, param in zip(axes.flat, params):
        means = [summary.loc[exp, (param, "mean")] for exp in FIGURE_ORDER]
        errors = [summary.loc[exp, (param, "sem")] for exp in FIGURE_ORDER]
        x = np.arange(2)
        ax.bar(x, means, yerr=errors, color=[COLORS[e] for e in FIGURE_ORDER], width=0.65, alpha=0.9)
        ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
        ax.set_xticks(x, ["Exp. 1", "Exp. 2"])
        ax.set_title(labels[param], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Best-fitting model parameters (C1_S0_B2)", fontsize=11)
    for path in [FIG_DIR / "fig6_parameter_estimates.png", FIG_DIR / "fig6_parameter_estimates.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def participant_mean_summary(df: pd.DataFrame, group_cols: list[str], value: str) -> pd.DataFrame:
    by_sub = df.groupby(["Sub"] + group_cols, as_index=False)[value].mean()
    return by_sub.groupby(group_cols)[value].agg(mean="mean", sem=sem, n="count").reset_index()


def prepare_ppc_trials() -> pd.DataFrame:
    ppc = pd.read_csv(VAL_DIR / "winner_ppc_trial_predictions.csv")
    frames = []
    for code_exp in [1, 2]:
        raw = load_valid_data(code_exp)[["subID", "trial_num", "preDur1back", "curCoherence", "preCoherence1back"]]
        merged = ppc[ppc["exp"].eq(code_exp)].merge(
            raw,
            left_on=["Sub", "trial_num"],
            right_on=["subID", "trial_num"],
            how="left",
        )
        if code_exp == 2:
            merged["condition"] = np.where(np.isclose(merged["curCoherence"], 0.3), "Low", "High")
        else:
            merged["condition"] = np.where(
                np.isclose(merged["curCoherence"], merged["preCoherence1back"]), "Same", "Switch"
            )
        merged["display_experiment"] = CODE_TO_DISPLAY[code_exp]
        frames.append(merged)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(READY_DIR / "kalman_ppc_trials_with_conditions.csv", index=False)
    return out


def figure7_trial_level_ppc():
    df = prepare_ppc_trials()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True, sharey=False)
    specs = [
        ("Experiment 1", ["High", "Low"], "Current coherence"),
        ("Experiment 2", ["Switch", "Same"], "Transition"),
    ]
    for ax, (experiment, conditions, legend_title) in zip(axes, specs):
        dd = df[df["display_experiment"].eq(experiment)].dropna(subset=["preDur1back"])
        for condition in conditions:
            sub = dd[dd["condition"].eq(condition)]
            obs = participant_mean_summary(sub, ["preDur1back"], "obs_bias")
            pred = participant_mean_summary(sub, ["preDur1back"], "pred_bias")
            color = "#4A9F79" if condition in ("Low", "Same") else "#5B7C99" if condition == "High" else "#C77C5B"
            ax.errorbar(obs["preDur1back"], obs["mean"], yerr=obs["sem"], marker="o", lw=1.4, color=color, label=f"{condition} obs")
            ax.plot(pred["preDur1back"], pred["mean"], ls="--", lw=1.5, color=color, label=f"{condition} pred")
        ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
        ax.set_title(experiment.replace(" dynamic", "").replace(" fixed", ""), fontsize=10, loc="left")
        ax.set_xlabel("Previous duration (s)")
        ax.set_ylabel("Bias / predicted bias (s)")
        ax.legend(frameon=False, fontsize=7, title=legend_title, title_fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for path in [FIG_DIR / "fig7_trial_level_serial_dependence.png", FIG_DIR / "fig7_trial_level_serial_dependence.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_b1_model_comparison():
    ranking = pd.read_csv(FULL_DIR / "model_ranking.csv")
    ranking = add_display_label(ranking)
    top = ranking.sort_values(["display_experiment", "delta_AIC"]).groupby("display_experiment").head(12)
    top.to_csv(READY_DIR / "kalman_top_model_ranking_for_figure.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.4), constrained_layout=True)
    for ax, experiment in zip(axes, FIGURE_ORDER):
        dd = top[top["display_experiment"].eq(experiment)].sort_values("delta_AIC", ascending=True)
        labels = dd["model_id"].str.replace("_", "\n", regex=False)
        y = np.arange(len(dd))
        ax.barh(y, dd["delta_AIC"], color=COLORS[experiment], alpha=0.9)
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(2, color="#777777", lw=0.8, ls="--")
        ax.set_xlabel("$\\Delta$AIC")
        ax.set_title(experiment, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    for path in [FIG_DIR / "figB1_model_comparison.png", FIG_DIR / "figB1_model_comparison.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_b2_recovery():
    metrics = add_display_label(pd.read_csv(VAL_DIR / "winner_ppc_subject_metrics.csv"))
    metrics.to_csv(READY_DIR / "kalman_ppc_subject_metrics.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.0), constrained_layout=True)
    for row, metric in enumerate(["cti", "sdi"]):
        for col, experiment in enumerate(FIGURE_ORDER):
            ax = axes[row, col]
            dd = metrics[metrics["display_experiment"].eq(experiment)]
            x = dd[f"{metric}_obs"]
            y = dd[f"{metric}_pred"]
            r = x.corr(y)
            lim_min = min(x.min(), y.min()) - 0.03
            lim_max = max(x.max(), y.max()) + 0.03
            ax.scatter(x, y, s=24, color=COLORS[experiment], alpha=0.85)
            ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#777777", lw=0.9, ls="--")
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_title(f"{experiment}: {metric.upper()} r = {r:.3f}", fontsize=9)
            ax.set_xlabel(f"Observed {metric.upper()}")
            ax.set_ylabel(f"Predicted {metric.upper()}")
            ax.spines[["top", "right"]].set_visible(False)
    for path in [FIG_DIR / "figB2_cti_sdi_recovery.png", FIG_DIR / "figB2_cti_sdi_recovery.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_kalman_summary():
    ranking = add_display_label(pd.read_csv(FULL_DIR / "model_ranking.csv"))
    params = pd.read_csv(READY_DIR / "kalman_winner_parameter_summary.csv", header=[0, 1], index_col=0)
    metrics = add_display_label(pd.read_csv(VAL_DIR / "winner_ppc_subject_metrics.csv"))
    lines = ["# Kalman Results", ""]
    for experiment in FIGURE_ORDER:
        best = ranking[ranking["display_experiment"].eq(experiment)].sort_values("delta_AIC").iloc[0]
        lines.append(
            f"- {experiment}: winner `{best['model_id']}`, mean AIC = {best['mean_AIC']:.3f}, "
            f"delta AIC = {best['delta_AIC']:.3f}, RMSE = {best['mean_RMSE']:.4f}."
        )
    lines.append("")
    lines.append("## Winner Parameter Means")
    for experiment in FIGURE_ORDER:
        p = params.loc[experiment]
        lines.append(
            f"- {experiment}: q1 = {p[('q1', 'mean')]:.3f}, q2 = {p[('q2', 'mean')]:.3f}, "
            f"q3 = {p[('q3', 'mean')]:.3f}, lambda = {p[('lambda', 'mean')]:.3f}, "
            f"alpha_q1 = {p[('alpha_q1', 'mean')]:.3f}, r_base = {p[('r_base', 'mean')]:.3f}."
        )
    lines.append("")
    lines.append("## PPC Correlations")
    for experiment in FIGURE_ORDER:
        dd = metrics[metrics["display_experiment"].eq(experiment)]
        lines.append(
            f"- {experiment}: CTI r = {dd['cti_obs'].corr(dd['cti_pred']):.3f}, "
            f"SDI r = {dd['sdi_obs'].corr(dd['sdi_pred']):.3f}."
        )
    (READY_DIR / "kalman_results_summary.md").write_text("\n".join(lines))


def main() -> None:
    figure6_parameter_estimates()
    figure7_trial_level_ppc()
    figure_b1_model_comparison()
    figure_b2_recovery()
    write_kalman_summary()
    print((READY_DIR / "kalman_results_summary.md").read_text())
    print("Saved Figure 6, Figure 7, Figure B1, and Figure B2.")


if __name__ == "__main__":
    main()
