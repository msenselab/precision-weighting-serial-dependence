"""Behavioural manuscript figures from the reprocessed data.

Each figure function returns the matplotlib figure; with save=True it also writes PNG and PDF under
results/figures. Run `python -m sdu.behavior.figures` to write the manuscript and audit figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from sdu import paths as SP


ROOT = SP.ROOT
DATA_DIR = SP.PREPROCESSED
READY_DIR = SP.BEHAVIOR_PRIMARY
FIG_DIR = SP.FIGURES_MAIN
APPENDIX_FIG_DIR = SP.FIGURES_APPENDIX

# Palette of the manuscript figures. The data column TransitionType keeps the
# notation of the experiment software (uncertainty levels), so display labels are
# remapped to objective coherence below.
COLORS = {
    "Low": "#E64B35", "High": "#4DBBD5", "Same": "#3C5488", "Switch": "#F39B7F",
    "Exp1": "#00A087", "Exp2": "#8491B4",
    "HH": "#E64B35", "HL": "#F39B7F", "LH": "#7E6148", "LL": "#4DBBD5",
}
TRANSITION_ORDER = ["HH", "HL", "LH", "LL"]
TRANSITION_LABELS = {"HH": "LL", "HL": "LH", "LH": "HL", "LL": "HH"}
TRANSITION_X_JITTER = {t: (i - 1.5) * 0.012 for i, t in enumerate(TRANSITION_ORDER)}

DATASETS = {
    "exp1_ramped": SP.PREPROCESSED_PKL[1],
    "exp2_constant": SP.PREPROCESSED_PKL[2],
}

DISPLAY = {
    "exp1_ramped": "Experiment 1",
    "exp2_constant": "Experiment 2",
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
    df["previous_coherence_label"] = np.where(df["prior_low"], "Low", "High")
    df["same_switch_label"] = np.where(df["current_low"].eq(df["prior_low"]), "Same", "Switch")
    df["SameSwitch"] = df["same_switch_label"]
    df["curCoherenceLevel"] = df["current_coherence_label"]
    df["curDur_c"] = df["curDur"] - 1.2
    df["preDur_c"] = df["preDur1back"] - 1.2
    df["curDurc"] = df.get("curDurc", df["curDur_c"])
    df["preDur1backc"] = df.get("preDur1backc", df["preDur_c"])
    return df


def transition_coefficients(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Joint Eq. 2 coefficients per participant and transition (original Figure 2)."""
    rows = []
    needed = ["curBias", "curDurc", "preDur1backc", "TransitionType"]
    for (sub_id, transition), group in df.dropna(subset=needed).groupby(["subID", "TransitionType"]):
        if len(group) < 5:
            continue
        X = sm.add_constant(group[["curDurc", "preDur1backc"]], has_constant="add")
        fit = sm.OLS(group["curBias"], X).fit()
        rows.append({
            "experiment": experiment, "subID": sub_id, "TransitionType": transition,
            "intercept": fit.params.get("const", np.nan),
            "central_slope": fit.params.get("curDurc", np.nan),
            "serial_slope": fit.params.get("preDur1backc", np.nan),
            "n_trials": len(group), "r2": fit.rsquared,
        })
    return pd.DataFrame(rows)


def _observed_transition_summary(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    by_sub = df.dropna(subset=["curBias", x_col, "TransitionType"]).groupby(
        ["subID", "TransitionType", x_col], as_index=False
    )["curBias"].mean()
    return by_sub.groupby(["TransitionType", x_col], as_index=False)["curBias"].agg(
        mean="mean", sem="sem", n="count"
    )


def _transition_line_panel(ax, df, coefficients, x_col, beta_col, title, xlabel, ylim):
    observed = _observed_transition_summary(df, x_col)
    x_values = sorted(observed[x_col].dropna().unique())
    for transition in TRANSITION_ORDER:
        color = COLORS[transition]
        points = observed[observed["TransitionType"].eq(transition)].sort_values(x_col)
        if points.empty:
            continue
        ax.errorbar(points[x_col] + TRANSITION_X_JITTER[transition], points["mean"], yerr=points["sem"],
                    marker="o", ls="none", color=color, ecolor=color, capsize=2.5, ms=4.5, alpha=.85)
        coef = coefficients[coefficients["TransitionType"].eq(transition)]
        if not coef.empty:
            yhat = coef["intercept"].mean() + coef[beta_col].mean() * (np.asarray(x_values) - 1.2)
            ax.plot(x_values, yhat, color=color, lw=2.2, label=TRANSITION_LABELS[transition])
    ax.axhline(0, color=".55", lw=.8, ls="--")
    ax.set(title=title, xlabel=xlabel, ylabel="Reproduction bias (s)", ylim=ylim)
    ax.title.set_fontweight("bold")
    ax.title.set_ha("left")
    ax.title.set_position((0, 1))
    ax.set_xticks(x_values)
    ax.spines[["top", "right"]].set_visible(False)


def _coefficient_panel(ax, coefficients, title, groups):
    positions, labels, means, errors, colors, centers = [], [], [], [], [], []
    cursor = 0.0
    for _, transitions in groups:
        xx = []
        for i, transition in enumerate(transitions):
            x = cursor + i * .68
            values = coefficients.loc[coefficients["TransitionType"].eq(transition), "serial_slope"]
            positions.append(x); labels.append(TRANSITION_LABELS[transition]); means.append(values.mean())
            errors.append(values.sem()); colors.append(COLORS[transition]); xx.append(x)
        centers.append(np.mean(xx)); cursor = max(xx) + 1.15
    ax.bar(positions, means, yerr=errors, color=colors, alpha=.82, capsize=3, width=.48,
           edgecolor="white", linewidth=.8)
    subject_groups = []
    for group_label, transitions in groups:
        subject_groups.append(coefficients[coefficients["TransitionType"].isin(transitions)].groupby("subID")[
            "serial_slope"].mean().rename(group_label))
    paired = pd.concat(subject_groups, axis=1).dropna()
    p = stats.ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1]).pvalue
    ax.plot([centers[0], centers[0], centers[1], centers[1]], [.16, .168, .168, .16], color="black", lw=.9)
    ax.text(np.mean(centers), .17, p_to_stars(p), ha="center", va="bottom", fontsize=12)
    ax.axhline(0, color=".55", lw=.8, ls="--")
    ax.set_xticks(positions, labels)
    ax.set(xlabel="Coherence transition", ylabel=r"SD coefficient $b_2$", ylim=(-.03, .20), title=title)
    ax.title.set_fontweight("bold"); ax.title.set_ha("left"); ax.title.set_position((0, 1))
    ax.spines[["top", "right"]].set_visible(False)


def subject_condition_slopes(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    specs = [
        ("current_coherence_label", ["Low", "High"]),
        ("previous_coherence_label", ["Low", "High"]),
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


def figure2(datasets: dict[str, pd.DataFrame], slopes: pd.DataFrame, save: bool = False):
    del slopes  # Figure 2 uses the original transition-wise Eq. 2 fits.
    df1, df2 = datasets["exp1_ramped"], datasets["exp2_constant"]
    coef1, coef2 = transition_coefficients(df1, "Experiment 1"), transition_coefficients(df2, "Experiment 2")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12.2), sharey="row")
    _transition_line_panel(axes[0, 0], df1, coef1, "curDur", "central_slope",
                           "A  Exp. 1: central tendency", "Current duration (s)", (-.25, .30))
    _transition_line_panel(axes[0, 1], df2, coef2, "curDur", "central_slope",
                           "B  Exp. 2: central tendency", "Current duration (s)", (-.25, .30))
    _transition_line_panel(axes[1, 0], df1, coef1, "preDur1back", "serial_slope",
                           "C  Exp. 1: serial dependence", "Previous duration (s)", (-.16, .21))
    _transition_line_panel(axes[1, 1], df2, coef2, "preDur1back", "serial_slope",
                           "D  Exp. 2: serial dependence", "Previous duration (s)", (-.16, .21))
    _coefficient_panel(axes[2, 0], coef1, r"E  Exp. 1: serial-dependence coefficient $b_2$",
                       [("Current low", ["HH", "LH"]), ("Current high", ["HL", "LL"])])
    _coefficient_panel(axes[2, 1], coef2, r"F  Exp. 2: serial-dependence coefficient $b_2$",
                       [("Same", ["HH", "LL"]), ("Switch", ["HL", "LH"])])
    handles = [mlines.Line2D([], [], color=".2", marker="o", ls="none", label="Observed mean ± SEM"),
               mlines.Line2D([], [], color=".2", lw=2.2, label="Coefficient line")]
    handles += [mlines.Line2D([], [], color=COLORS[t], lw=2.2, marker="o", label=TRANSITION_LABELS[t])
                for t in TRANSITION_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(.5, .995))
    fig.tight_layout(rect=(0, 0, 1, .955), h_pad=1.7, w_pad=1.4)

    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [FIG_DIR / "fig2_behavioral_results_combined.png", FIG_DIR / "fig2_behavioral_results_combined.pdf"]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure3(slopes: pd.DataFrame, save: bool = False):
    # Original cross-experiment layout: experiments on x, condition within experiment.
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    panel_specs = [
        ("current_coherence_label", ["Low", "High"], "A", {"Low": COLORS["Low"], "High": COLORS["High"]}),
        ("same_switch_label", ["Same", "Switch"], "B", {"Same": COLORS["Same"], "Switch": COLORS["Switch"]}),
    ]
    for ax, (grouping, conditions, panel, palette) in zip(axes, panel_specs):
        width = .35
        for j, dataset in enumerate(["exp1_ramped", "exp2_constant"]):
            dd = slopes[(slopes["dataset"].eq(dataset)) & (slopes["grouping"].eq(grouping))]
            values, errors = {}, {}
            for i, cond in enumerate(conditions):
                y = dd.loc[dd["condition"].eq(cond), "sdi_slope"]
                values[cond], errors[cond] = y.mean(), y.sem()
                ax.bar(j + width * (i - .5), values[cond], width, yerr=errors[cond], capsize=4,
                       color=palette[cond], alpha=.85, label=cond if j == 0 else None)
            wide = dd.pivot(index="subID", columns="condition", values="sdi_slope")[conditions].dropna()
            p = stats.ttest_rel(wide[conditions[0]], wide[conditions[1]]).pvalue
            if p < .05:
                top = max(values[c] + errors[c] for c in conditions) + .015
                ax.plot([j-.175, j-.175, j+.175, j+.175], [top, top+.01, top+.01, top], color="black", lw=1.2)
                ax.text(j, top+.01, p_to_stars(p), ha="center", va="bottom", fontsize=14)
        ax.axhline(0, color="gray", lw=.8, ls="--", alpha=.5)
        ax.set_xticks([0, 1], ["Exp 1", "Exp 2"], fontsize=13)
        ax.text(.02, .98, panel, transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight="bold")
        ax.set_ylabel("Serial Dependence", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend(title="Collapsed", frameon=False, fontsize=13, title_fontsize=12, loc="upper right")
        ax.set_ylim(0, .20)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [
        FIG_DIR / "fig3_cross_experiment_serial_dependence.png",
        FIG_DIR / "fig3_cross_experiment_serial_dependence.pdf",
    ]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


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
            if dataset == "exp1_ramped":
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


def figure4(summary: pd.DataFrame, save: bool = False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    specs = [
        ("exp1_ramped", ["Low", "High"], "A: Experiment 1",
         {"Low": "Current low coherence", "High": "Current high coherence"}),
        ("exp2_constant", ["Same", "Switch"], "B: Experiment 2",
         {"Same": "Current vs lag: same", "Switch": "Current vs lag: switch"}),
    ]
    x_order = [-3, -2, -1, 1, 2]
    labels = ["n-3", "n-2", "n-1", "n+1", "n+2"]
    bar_width = .35
    for ax, (dataset, conditions, title, legend_labels) in zip(axes, specs):
        for cond_idx, condition in enumerate(conditions):
            dd = summary[(summary["dataset"].eq(dataset)) & (summary["condition"].eq(condition))]
            dd = dd.set_index("lag").reindex(x_order).reset_index()
            offset = bar_width * (cond_idx - .5)
            ax.bar(np.arange(len(x_order)) + offset, dd["mean"], bar_width, yerr=dd["sem"], capsize=3,
                   color=COLORS[condition], alpha=.8, label=legend_labels[condition])
            for i, r in dd.iterrows():
                if pd.notna(r["p_bonferroni_5"]) and r["p_bonferroni_5"] < 0.05:
                    ax.text(i + offset, r["mean"] + r["sem"] + .005, p_to_stars(r["p_bonferroni_5"]),
                            ha="center", va="bottom", fontsize=14)
        ax.axhline(0, color="black", lw=.8, ls="--", alpha=.5)
        ax.set_xticks(np.arange(len(x_order)), labels, fontsize=13)
        ax.set_title(title, fontsize=16, loc="left", fontweight="bold", pad=8)
        ax.set_xlabel("Lag", fontsize=14)
        ax.set_ylabel("Serial Dependence", fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.16),
                  columnspacing=1.2, handletextpad=.6)
        ax.set_ylim(-.05, .22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if save:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [FIG_DIR / "fig4_temporal_window.png", FIG_DIR / "fig4_temporal_window.pdf"]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_a1(save: bool = False):
    params = pd.read_csv(READY_DIR / "response_error_lag_lmm_parameters.csv")
    params = params[params["parameter"].str.startswith("response_error_lag")].copy()
    params["lag"] = params["parameter"].str.extract(r"(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    for dataset, color, x_offset in [
        ("exp1_ramped", COLORS["Exp1"], -0.10),
        ("exp2_constant", COLORS["Exp2"], 0.10),
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
    if save:
        APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [APPENDIX_FIG_DIR / "figA1_response_error_lag.png",
                     APPENDIX_FIG_DIR / "figA1_response_error_lag.pdf"]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_four_transitions(
    observed: dict[int, pd.DataFrame], participant_contrasts: pd.DataFrame, save: bool = False
):
    """Participant slopes for LL/LH/HL/HH; brackets mark between-condition contrasts only."""
    cells = ["LL", "LH", "HL", "HH"]
    specs = [(1, "Exp1_ramped", "A  Experiment 1"), (2, "Exp2_constant", "B  Experiment 2")]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharey=True)
    for ax, (code, experiment, title) in zip(axes, specs):
        values = observed[code][[f"b2_{cell}" for cell in cells]]
        means, errors = values.mean().to_numpy(), values.sem().to_numpy()
        colors = [COLORS["Low"] if cell[1] == "L" else COLORS["High"] for cell in cells]
        x = np.arange(len(cells))
        ax.bar(x, means, yerr=errors, color=colors, alpha=.82, capsize=3.5, width=.68,
               edgecolor="white", linewidth=.8)
        for i, cell in enumerate(cells):
            y = values[f"b2_{cell}"].dropna().to_numpy()
            jitter = np.random.default_rng(100 * code + i).uniform(-.14, .14, len(y))
            ax.scatter(np.full(len(y), i) + jitter, y, color="black", s=13, alpha=.42, zorder=3)

        if code == 1:
            contrast = participant_contrasts[
                participant_contrasts["experiment"].eq(experiment)
                & participant_contrasts["contrast"].eq("current_L_minus_H")
            ]
            p = float(contrast.iloc[0]["p"])
            contrast_label = f"Current Low > High  {p_to_stars(p)}"
        else:
            ll_minus_others = values["b2_LL"] - values[["b2_LH", "b2_HL", "b2_HH"]].mean(axis=1)
            p = float(stats.ttest_1samp(ll_minus_others.dropna(), 0).pvalue)
            contrast_label = f"LL > mean(LH, HL, HH)  {p_to_stars(p)}"
        if p < .05:
            bracket_y = float(np.nanmax(values.to_numpy())) + .025
            bracket_h = .010
            ax.plot([0, 0, 3, 3], [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                    color="black", lw=1.1, clip_on=False)
            ax.text(1.5, bracket_y + bracket_h + .005, contrast_label,
                    ha="center", va="bottom", fontsize=9.5)
        ax.axhline(0, color=".55", lw=.8, ls="--")
        ax.set_xticks(x, cells)
        ax.set_xlabel("Previous → current coherence")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Controlled previous-duration slope")
    fig.text(.5, .01, "Brackets mark between-condition participant contrasts; no symbols test individual bars against zero.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1), w_pad=2)
    if save:
        APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [APPENDIX_FIG_DIR / "four_transition_slopes.png",
                     APPENDIX_FIG_DIR / "four_transition_slopes.pdf"]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_sequence_audit(
    sequence_summary: pd.DataFrame, transition_summary: pd.DataFrame, save: bool = False
):
    """Visual audit of coherence balance, switching, and the four transition counts."""
    experiments = list(sequence_summary["experiment"])
    exp_labels = {"Exp1_ramped": "Experiment 1", "Exp2_constant": "Experiment 2"}
    exp_colors = {"Exp1_ramped": COLORS["Exp1"], "Exp2_constant": COLORS["Exp2"]}
    metrics = [
        ("mean_p_low", "Low trials"),
        ("mean_p_switch", "Switches"),
        ("mean_p_low_given_previous_high", "Low | previous High"),
        ("mean_p_low_given_previous_low", "Low | previous Low"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    x, width = np.arange(len(metrics)), .34
    for j, experiment in enumerate(experiments):
        row = sequence_summary[sequence_summary["experiment"].eq(experiment)].iloc[0]
        values = np.array([row[column] for column, _ in metrics], float)
        xpos = x + (j - .5) * width
        ax.bar(xpos, values, width, color=exp_colors[experiment], alpha=.82,
               label=exp_labels[experiment])
        for xx, value in zip(xpos, values):
            ax.text(xx, value + .006, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(.5, color="black", lw=.9, ls="--", alpha=.7, label="Balanced = 0.50")
    ax.set_xticks(x, [label for _, label in metrics], rotation=16, ha="right")
    ax.set_ylim(.42, .58)
    ax.set_ylabel("Participant-mean proportion")
    ax.set_title("A  Were coherence and switches balanced?", loc="left", fontweight="bold")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.17))
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    transitions = ["LL", "LH", "HL", "HH"]
    x = np.arange(len(transitions))
    for j, experiment in enumerate(experiments):
        dd = transition_summary[transition_summary["experiment"].eq(experiment)].set_index("transition").loc[transitions]
        xpos = x + (j - .5) * width
        means = dd["mean_per_participant"].to_numpy(float)
        ax.bar(xpos, means, width, yerr=dd["sd_per_participant"], capsize=3,
               color=exp_colors[experiment], alpha=.82, label=exp_labels[experiment])
        ax.vlines(xpos, dd["min"], dd["max"], color=".2", lw=.8, alpha=.7)
        for xx, value in zip(xpos, means):
            ax.text(xx, value + 4.4, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, transitions)
    ax.set_ylim(0, 74)
    ax.set_xlabel("Previous → current coherence")
    ax.set_ylabel("Trials per participant")
    ax.set_title("B  How many trials informed each transition?", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2)
    if save:
        APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [APPENDIX_FIG_DIR / "sequence_audit.png", APPENDIX_FIG_DIR / "sequence_audit.pdf"]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def figure_ordering_bayes_factors(ordering: pd.DataFrame, save: bool = False):
    """Sensitivity of the four transition-ordering Bayes factors to prior tau and minimum gap."""
    patterns = ["current", "reliability", "reset", "flat"]
    labels = ["Current\nuncertainty", "Reliability\nweighting", "Reset", "Flat"]
    conventions = [("direction_only", "Order only"), ("meaningful_gap", "Order + minimum gap")]
    experiments = ["Exp 1", "Exp 2"]
    palette = dict(zip(patterns, [COLORS["Low"], "#7E6148", COLORS["Same"], ".45"]))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True, sharey=True)
    combinations = sorted(ordering[["prior_tau", "delta"]].drop_duplicates().itertuples(index=False, name=None))
    offsets = np.linspace(-.16, .16, len(combinations))
    offset_by_combo = dict(zip(combinations, offsets))
    for row, experiment in enumerate(experiments):
        for col, (convention, convention_label) in enumerate(conventions):
            ax = axes[row, col]
            dd = ordering[
                ordering["experiment"].eq(experiment) & ordering["convention"].eq(convention)
            ]
            for i, pattern in enumerate(patterns):
                points = dd[dd["pattern"].eq(pattern)].copy()
                point_x = [i + offset_by_combo[(r.prior_tau, r.delta)] for r in points.itertuples()]
                ax.scatter(point_x, points["BF_vs_unconstrained"], s=23, color=palette[pattern],
                           alpha=.58, edgecolor="none")
                median = points["BF_vs_unconstrained"].median()
                ax.scatter(i, median, s=67, marker="D", color=palette[pattern], edgecolor="white",
                           linewidth=.8, zorder=4)
            ax.axhline(1, color="black", lw=.9, ls="--")
            ax.set_yscale("log")
            ax.set_ylim(.0003, 250)
            ax.set_yticks([.001, .01, .1, 1, 10, 100])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.set_title(f"{chr(65 + row * 2 + col)}  {experiment}: {convention_label}",
                         loc="left", fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(range(len(patterns)), labels)
    for ax in axes[:, 0]:
        ax.set_ylabel("Bayes factor vs unconstrained")
    fig.text(.5, .015,
             "Order only: any positive difference counts. Order + minimum gap: the difference must be at least "
             "0.015, 0.020, or 0.030. Small points show all settings; diamonds show medians; dashed line: BF = 1.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, 1), h_pad=2, w_pad=1.5)
    if save:
        APPENDIX_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for path in [APPENDIX_FIG_DIR / "ordering_bayes_factors.png",
                     APPENDIX_FIG_DIR / "ordering_bayes_factors.pdf"]:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return fig


def main() -> None:
    datasets = {name: load_valid(path) for name, path in DATASETS.items()}
    slopes = subject_condition_slopes(datasets)
    figure2(datasets, slopes, save=True)
    figure3(slopes, save=True)
    temporal = temporal_window(datasets)
    figure4(temporal, save=True)
    figure_a1(save=True)
    from sdu.model import stats as model_stats
    participant_contrasts = pd.read_csv(SP.BEHAVIOR_TRANSITIONS / "participant_contrasts.csv")
    figure_four_transitions({code: model_stats.observed(code) for code in (1, 2)}, participant_contrasts, save=True)
    sequence_dir = SP.BEHAVIOR_SEQUENCES
    figure_sequence_audit(
        pd.read_csv(sequence_dir / "sequence_summary.csv"),
        pd.read_csv(sequence_dir / "transition_summary.csv"),
        save=True,
    )
    figure_ordering_bayes_factors(
        pd.read_csv(SP.BEHAVIOR_ORDERINGS / "ordering_bayes_factors.csv"),
        save=True,
    )
    print("Saved behavioural manuscript and audit figures.")


if __name__ == "__main__":
    main()
