#!/usr/bin/env python3
"""Generate behavioral figures from cleaned data and prepared source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
READY_DIR = ROOT / "results" / "figure_source_data"
FIG_DIR = ROOT / "figures"
READY_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Low": "#4A9F79",
    "High": "#4E79A7",
    "Same": "#5B7C99",
    "Switch": "#C77C5B",
    "Exp1": "#4A9F79",
    "Exp2": "#5B7C99",
}

DATASETS = {
    "experiment1_dynamic": DATA_DIR / "experiment2" / "E2.pkl",
    "experiment2_fixed": DATA_DIR / "experiment1" / "E1.pkl",
}

DISPLAY = {
    "experiment1_dynamic": "Experiment 1",
    "experiment2_fixed": "Experiment 2",
}


def sem(x):
    x = pd.Series(x).dropna()
    return x.sem() if len(x) > 1 else np.nan


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def load_valid(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df[df["is_outlier"] == False].copy()
    df = df.dropna(subset=["subID", "curBias", "curDur", "preDur1back", "curCoherence", "preCoherence1back"])
    df["subID"] = df["subID"].astype(str)
    df["current_low"] = np.isclose(df["curCoherence"], 0.3)
    df["prior_low"] = np.isclose(df["preCoherence1back"], 0.3)
    df["current_coherence_label"] = np.where(df["current_low"], "Low", "High")
    df["same_switch_label"] = np.where(df["current_low"].eq(df["prior_low"]), "Same", "Switch")
    df["curDur_c"] = df["curDur"] - 1.2
    df["preDur_c"] = df["preDur1back"] - 1.2
    return df


def subject_condition_slopes(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    specs = [
        ("current_coherence_label", ["Low", "High"]),
        ("same_switch_label", ["Same", "Switch"]),
    ]
    for dataset, df in datasets.items():
        for grouping, conditions in specs:
            for (sub_id, condition), dd in df.groupby(["subID", grouping]):
                if condition not in conditions or len(dd) < 10:
                    continue
                if dd["curDur"].nunique() < 2 or dd["preDur1back"].nunique() < 2:
                    continue
                X = sm.add_constant(dd[["curDur", "preDur1back"]], has_constant="add")
                res = sm.OLS(dd["curBias"], X).fit()
                rows.append(
                    {
                        "dataset": dataset,
                        "grouping": grouping,
                        "subID": sub_id,
                        "condition": condition,
                        "ct_slope": res.params["curDur"],
                        "sdi_slope": res.params["preDur1back"],
                        "n_trials": len(dd),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(READY_DIR / "figure_subject_condition_slopes.csv", index=False)
    return out


def participant_mean_summary(df: pd.DataFrame, group_cols: list[str], value: str) -> pd.DataFrame:
    by_sub = df.groupby(["subID"] + group_cols, as_index=False)[value].mean()
    summary = (
        by_sub.groupby(group_cols)[value]
        .agg(mean="mean", sem=sem, n="count")
        .reset_index()
    )
    return summary


def draw_line_panel(ax, summary: pd.DataFrame, x: str, hue: str | None, title: str, xlabel: str):
    if hue is None:
        ax.errorbar(summary[x], summary["mean"], yerr=summary["sem"], marker="o", color="#333333", lw=1.5)
    else:
        for condition, dd in summary.groupby(hue):
            ax.errorbar(
                dd[x],
                dd["mean"],
                yerr=dd["sem"],
                marker="o",
                lw=1.5,
                color=COLORS.get(condition, "#333333"),
                label=condition,
            )
    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reproduction bias (s)")
    ax.spines[["top", "right"]].set_visible(False)


def draw_slope_bar(ax, slopes: pd.DataFrame, dataset: str, grouping: str, conditions: list[str], title: str):
    dd = slopes[(slopes["dataset"].eq(dataset)) & (slopes["grouping"].eq(grouping))]
    vals = []
    errs = []
    for cond in conditions:
        v = dd.loc[dd["condition"].eq(cond), "sdi_slope"]
        vals.append(v.mean())
        errs.append(v.sem())
    x = np.arange(len(conditions))
    ax.bar(x, vals, yerr=errs, color=[COLORS[c] for c in conditions], alpha=0.85, width=0.65)
    for i, cond in enumerate(conditions):
        y = dd.loc[dd["condition"].eq(cond), "sdi_slope"]
        jitter = np.linspace(-0.08, 0.08, len(y)) if len(y) else []
        ax.scatter(np.full(len(y), i) + jitter, y, s=14, color="#222222", alpha=0.55, zorder=3)
    if len(conditions) == 2:
        wide = dd.pivot(index="subID", columns="condition", values="sdi_slope")[conditions].dropna()
        _, p = stats.ttest_rel(wide[conditions[0]], wide[conditions[1]])
        y_star = max(np.array(vals) + np.array(errs)) + 0.025
        ax.plot([0, 1], [y_star, y_star], color="#333333", lw=1)
        ax.text(0.5, y_star + 0.006, p_to_stars(p), ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_xticks(x, conditions)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("Controlled serial-dependence slope")
    ax.spines[["top", "right"]].set_visible(False)


def figure2(datasets: dict[str, pd.DataFrame], slopes: pd.DataFrame):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4), constrained_layout=True)

    for col, dataset in enumerate(["experiment1_dynamic", "experiment2_fixed"]):
        df = datasets[dataset]
        cur_summary = participant_mean_summary(df, ["curDur"], "curBias")
        draw_line_panel(
            axes[0, col],
            cur_summary,
            "curDur",
            None,
            f"{chr(65 + col)}. {DISPLAY[dataset]}: current duration",
            "Current duration (s)",
        )

    exp1_prev = participant_mean_summary(
        datasets["experiment1_dynamic"], ["preDur1back", "current_coherence_label"], "curBias"
    )
    draw_line_panel(
        axes[1, 0],
        exp1_prev,
        "preDur1back",
        "current_coherence_label",
        "C. Experiment 1: previous duration",
        "Previous duration (s)",
    )
    axes[1, 0].legend(frameon=False, title="Current coherence", fontsize=8, title_fontsize=8, loc="lower right")

    exp2_prev = participant_mean_summary(
        datasets["experiment2_fixed"], ["preDur1back", "same_switch_label"], "curBias"
    )
    draw_line_panel(
        axes[1, 1],
        exp2_prev,
        "preDur1back",
        "same_switch_label",
        "D. Experiment 2: previous duration",
        "Previous duration (s)",
    )
    axes[1, 1].legend(frameon=False, title="Transition", fontsize=8, title_fontsize=8, loc="upper left")

    draw_slope_bar(
        axes[2, 0],
        slopes,
        "experiment1_dynamic",
        "current_coherence_label",
        ["High", "Low"],
        "E. Experiment 1 controlled slopes",
    )
    draw_slope_bar(
        axes[2, 1],
        slopes,
        "experiment2_fixed",
        "same_switch_label",
        ["Switch", "Same"],
        "F. Experiment 2 controlled slopes",
    )

    for path in [FIG_DIR / "fig2_behavioral_results_combined.png", FIG_DIR / "fig2_behavioral_results_combined.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure3(slopes: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    panel_specs = [
        ("current_coherence_label", ["High", "Low"], "A. Current coherence"),
        ("same_switch_label", ["Switch", "Same"], "B. Transition consistency"),
    ]
    for ax, (grouping, conditions, title) in zip(axes, panel_specs):
        x_centers = np.arange(2)
        width = 0.33
        for j, dataset in enumerate(["experiment1_dynamic", "experiment2_fixed"]):
            dd = slopes[(slopes["dataset"].eq(dataset)) & (slopes["grouping"].eq(grouping))]
            vals, errs = [], []
            for cond in conditions:
                y = dd.loc[dd["condition"].eq(cond), "sdi_slope"]
                vals.append(y.mean())
                errs.append(y.sem())
            x = x_centers + (j - 0.5) * width
            ax.bar(x, vals, width=width, yerr=errs, color=COLORS["Exp1" if j == 0 else "Exp2"], alpha=0.85, label=DISPLAY[dataset])
        ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
        ax.set_xticks(x_centers, conditions)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Controlled serial-dependence slope")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    for path in [
        FIG_DIR / "fig3_cross_experiment_serial_dependence.png",
        FIG_DIR / "fig3_cross_experiment_serial_dependence.pdf",
    ]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_slope(dd: pd.DataFrame, predictor: str) -> float | None:
    d = dd.dropna(subset=["curBias", "curDur", predictor])
    if len(d) < 10 or d[predictor].nunique() < 2 or d["curDur"].nunique() < 2:
        return None
    X = sm.add_constant(d[["curDur", predictor]], has_constant="add")
    return float(sm.OLS(d["curBias"], X).fit().params[predictor])


def temporal_window(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    lag_specs = [(-3, "preDur3back"), (-2, "preDur2back"), (-1, "preDur1back"), (1, "postDur1"), (2, "postDur2")]
    for dataset, df in datasets.items():
        for lag, predictor in lag_specs:
            if predictor not in df.columns:
                continue
            if dataset == "experiment1_dynamic":
                df["lag_condition"] = df["current_coherence_label"]
            else:
                if lag < 0:
                    coh_col = f"preCoherence{abs(lag)}back"
                else:
                    coh_col = f"postCoherence{lag}"
                df["lag_condition"] = np.where(np.isclose(df["curCoherence"], df[coh_col]), "Same", "Switch")
            for (sub_id, condition), dd in df.groupby(["subID", "lag_condition"]):
                slope = fit_slope(dd, predictor)
                if slope is None:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "lag": lag,
                        "predictor": predictor,
                        "condition": condition,
                        "subID": sub_id,
                        "slope": slope,
                    }
                )
    slopes = pd.DataFrame(rows)
    summary_rows = []
    for (dataset, condition, lag), dd in slopes.groupby(["dataset", "condition", "lag"]):
        t, p = stats.ttest_1samp(dd["slope"], 0.0)
        summary_rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "lag": lag,
                "mean": dd["slope"].mean(),
                "sem": dd["slope"].sem(),
                "N": len(dd),
                "t": t,
                "p": p,
                "p_bonferroni_5": min(p * 5, 1.0),
            }
        )
    slopes.to_csv(READY_DIR / "temporal_window_subject_slopes.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(READY_DIR / "temporal_window_summary.csv", index=False)
    return summary


def figure4(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True, sharey=True)
    specs = [
        ("experiment1_dynamic", ["High", "Low"], "A. Experiment 1"),
        ("experiment2_fixed", ["Switch", "Same"], "B. Experiment 2"),
    ]
    x_order = [-3, -2, -1, 1, 2]
    labels = ["n-3", "n-2", "n-1", "n+1", "n+2"]
    for ax, (dataset, conditions, title) in zip(axes, specs):
        for cond_idx, condition in enumerate(conditions):
            dd = summary[(summary["dataset"].eq(dataset)) & (summary["condition"].eq(condition))]
            dd = dd.set_index("lag").reindex(x_order).reset_index()
            ax.errorbar(
                np.arange(len(x_order)),
                dd["mean"],
                yerr=dd["sem"],
                marker="o",
                lw=1.5,
                color=COLORS[condition],
                label=condition,
            )
            for i, r in dd.iterrows():
                if pd.notna(r["p_bonferroni_5"]) and r["p_bonferroni_5"] < 0.05:
                    y_offset = 0.017 + cond_idx * 0.011
                    text_x = i + (-0.08 if cond_idx == 0 else 0.08)
                    ax.text(
                        text_x,
                        r["mean"] + np.sign(r["mean"] or 1) * y_offset,
                        p_to_stars(r["p_bonferroni_5"]),
                        ha="center",
                        fontsize=8,
                    )
        ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
        ax.axvline(2.5, color="#999999", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(np.arange(len(x_order)), labels)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("Relative trial")
        ax.set_ylabel("Serial-dependence slope")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for path in [FIG_DIR / "fig4_temporal_window.png", FIG_DIR / "fig4_temporal_window.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_a1():
    params = pd.read_csv(READY_DIR / "response_error_lag_lmm_parameters.csv")
    params = params[params["parameter"].str.startswith("response_error_lag")].copy()
    params["lag"] = params["parameter"].str.extract(r"(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    for dataset, color, x_offset in [
        ("experiment1_dynamic", COLORS["Exp1"], -0.10),
        ("experiment2_fixed", COLORS["Exp2"], 0.10),
    ]:
        dd = params[params["dataset"].eq(dataset)].sort_values("lag")
        x = dd["lag"].astype(float) + x_offset
        ax.errorbar(x, dd["beta"], yerr=dd["se"], marker="o", lw=1.5, color=color, label=DISPLAY[dataset])
        for _, r in dd.iterrows():
            ax.text(r["lag"] + x_offset, r["beta"] + r["se"] + 0.012, p_to_stars(r["p"]), ha="center", fontsize=8)
    ax.axhline(0, color="#777777", lw=0.8, alpha=0.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Response-error lag")
    ax.set_ylabel("Carryover coefficient")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for path in [FIG_DIR / "figA1_response_error_lag.png", FIG_DIR / "figA1_response_error_lag.pdf"]:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    datasets = {name: load_valid(path) for name, path in DATASETS.items()}
    slopes = subject_condition_slopes(datasets)
    figure2(datasets, slopes)
    figure3(slopes)
    temporal = temporal_window(datasets)
    figure4(temporal)
    figure_a1()
    print("Saved Figure 2, Figure 3, Figure 4, and Figure A1.")


if __name__ == "__main__":
    main()
