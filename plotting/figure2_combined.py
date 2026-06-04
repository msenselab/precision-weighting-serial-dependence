#!/usr/bin/env python3
"""Generate the combined behavioral-results layout used for Figure 2.

The 3 x 2 layout overlays observed group means with lines reconstructed from
participant-level joint current/previous-duration coefficients. Columns are
manuscript Experiment 1/2; rows show central tendency, serial dependence, and
coefficient summaries.

This module retains the original combined central-tendency and serial-dependence
implementation used by the public figure-generation entry point.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
import statsmodels.api as sm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_config import (  # noqa: E402
    COLORS,
    TRANSITION_ORDER,
    COHERENCE_TRANSITION_LABELS,
    set_nature_style,
    despine,
)

# Default output directories (callers override OUT_DIR / FIG_DIR before plotting).
OUT_DIR = PROJECT / "results" / "figure_source_data"
FIG_DIR = PROJECT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Slightly larger text for A4 readability.
set_nature_style()
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
})

X_JITTER = {t: (i - (len(TRANSITION_ORDER) - 1) / 2) * 0.012 for i, t in enumerate(TRANSITION_ORDER)}


def load_data():
    """Load public datasets in manuscript display order."""
    df1 = pd.read_pickle(PROJECT / "data" / "experiment1" / "E1.pkl")
    df2 = pd.read_pickle(PROJECT / "data" / "experiment2" / "E2.pkl")
    df1 = df1.loc[df1["is_outlier"].eq(False)].copy()
    df2 = df2.loc[df2["is_outlier"].eq(False)].copy()
    for df in (df1, df2):
        df["SameSwitch"] = df["TransitionType"].map({"HH": "Same", "LL": "Same", "HL": "Switch", "LH": "Switch"})
        df["curCoherenceLevel"] = df["curCoherence"].map({0.3: "Low", 0.7: "High"})
    return df1, df2


def fit_participant_transition_joint(df, exp_label):
    """Fit curBias ~ curDurc + preDur1backc per participant x transition."""
    rows = []
    for (sub_id, trans), g in df.dropna(subset=["curBias", "curDurc", "preDur1backc", "TransitionType"]).groupby(["subID", "TransitionType"]):
        if len(g) < 5:
            continue
        X = sm.add_constant(g[["curDurc", "preDur1backc"]])
        res = sm.OLS(g["curBias"], X).fit()
        rows.append({
            "experiment": exp_label,
            "subID": sub_id,
            "TransitionType": trans,
            "display_transition": COHERENCE_TRANSITION_LABELS.get(trans, trans),
            "intercept": res.params.get("const", np.nan),
            "central_slope": res.params.get("curDurc", np.nan),
            "serial_slope": res.params.get("preDur1backc", np.nan),
            "n_trials": len(g),
            "r2": res.rsquared,
        })
    return pd.DataFrame(rows)


def summarize_observed(df, x_col):
    """Subject-first observed means and SEM across subjects."""
    subj = (
        df.dropna(subset=["curBias", x_col, "TransitionType"])
        .groupby(["subID", "TransitionType", x_col], as_index=False)["curBias"]
        .mean()
    )
    summ = (
        subj.groupby(["TransitionType", x_col], as_index=False)["curBias"]
        .agg(mean="mean", sem="sem", n="count")
    )
    return summ


def coefficient_summary(coef_df):
    rows = []
    for trans, g in coef_df.groupby("TransitionType"):
        rows.append({
            "TransitionType": trans,
            "display_transition": COHERENCE_TRANSITION_LABELS.get(trans, trans),
            "central_mean": g["central_slope"].mean(),
            "central_sem": g["central_slope"].sem(),
            "serial_mean": g["serial_slope"].mean(),
            "serial_sem": g["serial_slope"].sem(),
            "intercept_mean": g["intercept"].mean(),
            "n": g["subID"].nunique(),
        })
    return pd.DataFrame(rows)


def plot_row_observed_plus_model(ax, df, coef, x_col, beta_col, title, xlabel, ylabel, ylim=None):
    obs = summarize_observed(df, x_col)
    x_values = sorted(obs[x_col].dropna().unique())
    x_center = 1.2  # curDurc/preDur1backc are centered around 1.2 s in both experiments.

    for trans in TRANSITION_ORDER:
        color = COLORS[trans]
        label = COHERENCE_TRANSITION_LABELS[trans]
        o = obs.loc[obs["TransitionType"].eq(trans)].sort_values(x_col)
        if o.empty:
            continue
        ax.errorbar(
            o[x_col] + X_JITTER[trans],
            o["mean"],
            yerr=o["sem"],
            marker="o",
            linestyle="none",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=4.5,
            alpha=0.85,
        )
        c = coef.loc[coef["TransitionType"].eq(trans)]
        if c.empty:
            continue
        intercept = c["intercept"].mean()
        beta = c[beta_col].mean()
        yhat = intercept + beta * (np.asarray(x_values) - x_center)
        ax.plot(x_values, yhat, color=color, lw=2.2, label=label)

    ax.axhline(0, color="0.55", lw=0.8, ls="--", zorder=0)
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    if ylim is not None:
        ax.set_ylim(*ylim)
    despine(ax)


def p_to_stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def add_group_bracket(ax, x1, x2, y, p_value, height=0.008):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=0.9, color="black")
    ax.text((x1 + x2) / 2, y + height + 0.002, p_to_stars(p_value), ha="center", va="bottom", fontsize=12)


def plot_coef_panel(ax, coef_df, title, group_defs):
    """Plot beta_2 by transition, visually grouped by planned contrast."""
    x_positions = []
    x_labels = []
    means = []
    sems = []
    colors = []
    group_centers = []
    group_spans = []

    cursor = 0.0
    within_gap = 0.68
    between_gap = 1.15
    for group_label, transitions in group_defs:
        group_x = []
        for i, trans in enumerate(transitions):
            x = cursor + i * within_gap
            vals = coef_df.loc[coef_df["TransitionType"].eq(trans), "serial_slope"]
            x_positions.append(x)
            x_labels.append(COHERENCE_TRANSITION_LABELS[trans])
            means.append(vals.mean())
            sems.append(vals.sem())
            colors.append(COLORS[trans])
            group_x.append(x)
        group_centers.append(np.mean(group_x))
        group_spans.append((min(group_x) - 0.32, max(group_x) + 0.32))
        cursor = max(group_x) + between_gap

    ax.bar(
        x_positions,
        means,
        yerr=sems,
        color=colors,
        alpha=0.82,
        capsize=3,
        width=0.48,
        edgecolor="white",
        linewidth=0.8,
    )

    # Paired planned contrast on subject-averaged beta_2 within each group.
    group_values = []
    contrast_rows = []
    for group_label, transitions in group_defs:
        g = coef_df.loc[coef_df["TransitionType"].isin(transitions)]
        sub_mean = g.groupby("subID", as_index=False)["serial_slope"].mean().rename(columns={"serial_slope": group_label})
        group_values.append(sub_mean)
    wide = group_values[0].merge(group_values[1], on="subID", how="inner")
    g1, g2 = group_defs[0][0], group_defs[1][0]
    t_stat, p_value = stats.ttest_rel(wide[g1], wide[g2])
    diff = wide[g1] - wide[g2]
    contrast_rows.append({
        "contrast": f"{g1} - {g2}",
        "n": len(wide),
        "mean_difference": diff.mean(),
        "t": t_stat,
        "p": p_value,
        "dz": diff.mean() / diff.std(ddof=1),
    })

    ax.axhline(0, color="0.55", lw=0.8, ls="--")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Coherence transition")
    ax.set_ylabel(r"SD coefficient $b_2$")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    # Leave extra headroom so the tallest error bars/brackets do not touch the axis boundary.
    ax.set_ylim(-0.03, 0.20)

    # Group labels under transition labels and light spans to make contrasts obvious.
    for (group_label, _), center, (span_left, span_right) in zip(group_defs, group_centers, group_spans):
        ax.axvspan(span_left, span_right, color="0.94", zorder=-2)
        ax.text(center, -0.20, group_label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11, fontweight="bold")

    add_group_bracket(ax, group_centers[0], group_centers[1], 0.160, p_value)
    despine(ax)
    return contrast_rows


def main():
    df1, df2 = load_data()
    coef1 = fit_participant_transition_joint(df1, "Experiment 1")
    coef2 = fit_participant_transition_joint(df2, "Experiment 2")
    coef_all = pd.concat([coef1, coef2], ignore_index=True)
    coef_all.to_csv(OUT_DIR / "participant_joint_coefficients_by_transition.csv", index=False)
    coefficient_summary(coef_all).to_csv(OUT_DIR / "coefficient_summary_by_transition.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 12.2), sharey="row")

    plot_row_observed_plus_model(
        axes[0, 0], df1, coef1, "curDur", "central_slope",
        "A  Exp. 1: central tendency", "Current duration (s)", "Reproduction bias (s)", (-0.25, 0.30)
    )
    plot_row_observed_plus_model(
        axes[0, 1], df2, coef2, "curDur", "central_slope",
        "B  Exp. 2: central tendency", "Current duration (s)", "Reproduction bias (s)", (-0.25, 0.30)
    )
    plot_row_observed_plus_model(
        axes[1, 0], df1, coef1, "preDur1back", "serial_slope",
        "C  Exp. 1: serial dependence", "Previous duration (s)", "Reproduction bias (s)", (-0.16, 0.21)
    )
    plot_row_observed_plus_model(
        axes[1, 1], df2, coef2, "preDur1back", "serial_slope",
        "D  Exp. 2: serial dependence", "Previous duration (s)", "Reproduction bias (s)", (-0.16, 0.21)
    )
    contrast_rows = []
    contrast_rows += plot_coef_panel(
        axes[2, 0],
        coef1,
        r"E  Exp. 1: serial dependence coefficient",
        [("Current low", ["HH", "LH"]), ("Current high", ["HL", "LL"])],
    )
    contrast_rows += plot_coef_panel(
        axes[2, 1],
        coef2,
        r"F  Exp. 2: serial dependence coefficient",
        [("Same", ["HH", "LL"]), ("Switch", ["HL", "LH"])],
    )
    pd.DataFrame(contrast_rows).to_csv(OUT_DIR / "beta2_grouped_planned_contrasts.csv", index=False)

    # Single shared legend: lines = model estimates, points/errorbars = observed means.
    transition_handles = [mlines.Line2D([], [], color=COLORS[t], lw=2.2, marker="o", markersize=4.5, label=COHERENCE_TRANSITION_LABELS[t]) for t in TRANSITION_ORDER]
    obs_handle = mlines.Line2D([], [], color="0.2", marker="o", linestyle="none", markersize=4.5, label="Observed mean ± SEM")
    model_handle = mlines.Line2D([], [], color="0.2", lw=2.2, label="Coefficient line")
    beta2_handle = mlines.Line2D([], [], color="0.2", lw=7, label=r"Bottom row: $b_2$ only")
    fig.legend(handles=[obs_handle, model_handle, beta2_handle, *transition_handles], loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.995))

    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.7, w_pad=1.4)

    png_out = FIG_DIR / "fig2_behavioral_results_combined.png"
    pdf_out = FIG_DIR / "fig2_behavioral_results_combined.pdf"
    fig.savefig(png_out, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 2: {png_out}")
    print(f"Coefficient table: {OUT_DIR / 'participant_joint_coefficients_by_transition.csv'}")


if __name__ == "__main__":
    main()
