#!/usr/bin/env python3
"""Regenerate the canonical behavioral manuscript figures (Fig 2, 3, 4, A1).

It uses the shared manuscript plotting modules under ``plotting/``:

  - ``generate_main_figures.py``        (cross-experiment + shared helpers)
  - ``figure2_combined.py``             (Figure 2 six-panel layout)
  - ``results/figure_source_data/``      (behavioral analysis tables used by
                                         Figure 2, Figure 3, and Figure A1)

Data mapping:
  Experiment 1 (dynamic) = data/experiment1/E1.pkl
  Experiment 2 (fixed)   = data/experiment2/E2.pkl

Figures are rendered in a temporary staging directory. Canonical exports under
``figures/`` are replaced only when the rendered PNG pixels change.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, ttest_1samp


ROOT = Path(__file__).resolve().parents[1]
PLOTTING = ROOT / "plotting"
FIG_DIR = ROOT / "figures"
SOURCE_DATA = ROOT / "results" / "figure_source_data"

RESP_LAG_PARAMS = SOURCE_DATA / "response_error_lag_lmm_parameters.csv"
CONTROLLED_SLOPES = SOURCE_DATA / "participant_controlled_slopes.csv"
CONTROLLED_CONTRASTS = SOURCE_DATA / "participant_controlled_slope_contrasts.csv"

CANONICAL_STEMS = [
    "fig2_behavioral_results_combined",
    "fig3_cross_experiment_serial_dependence",
    "fig4_temporal_window",
    "figA1_response_error_lag",
]


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def patched_fig2_contrast_tests(module):
    """Use the formal participant-controlled contrast p-values in Figure 2."""
    original = module.stats.ttest_rel
    formal = pd.read_csv(CONTROLLED_CONTRASTS)
    p_exp1 = float(
        formal[
            (formal["dataset"] == "experiment1")
            & (formal["grouping"] == "current_coherence_label")
        ]["p"].iloc[0]
    )
    t_exp1 = float(
        formal[
            (formal["dataset"] == "experiment1")
            & (formal["grouping"] == "current_coherence_label")
        ]["t"].iloc[0]
    )
    p_exp2 = float(
        formal[
            (formal["dataset"] == "experiment2")
            & (formal["grouping"] == "same_switch_label")
        ]["p"].iloc[0]
    )
    t_exp2 = float(
        formal[
            (formal["dataset"] == "experiment2")
            & (formal["grouping"] == "same_switch_label")
        ]["t"].iloc[0]
    )
    calls = {"n": 0}

    def ttest_rel(a, b, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return t_exp1, p_exp1
        if calls["n"] == 2:
            return t_exp2, p_exp2
        return original(a, b, *args, **kwargs)

    module.stats.ttest_rel = ttest_rel
    try:
        yield
    finally:
        module.stats.ttest_rel = original


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def add_sig_bracket(ax, x1: float, x2: float, y: float, label: str, h: float = 0.01) -> None:
    if label == "n.s.":  # do not annotate non-significant contrasts
        return
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=14)


def plot_fig3_from_formal_slopes(staging: Path, main_module) -> None:
    """Current Figure 3 style, using formal participant-controlled slopes."""
    slopes = pd.read_csv(CONTROLLED_SLOPES)
    contrasts = pd.read_csv(CONTROLLED_CONTRASTS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    width = 0.35
    x_exp = np.arange(2)

    def get_contrast(dataset: str, grouping: str):
        row = contrasts[(contrasts["dataset"] == dataset) & (contrasts["grouping"] == grouping)].iloc[0]
        return p_to_stars(float(row["p"])), float(row["p"])

    ax = axes[0]
    conds = ["Low", "High"]
    colors = {"Low": main_module.COLORS["low"], "High": main_module.COLORS["high"]}
    datasets = ["experiment1", "experiment2"]
    labels = ["Exp 1", "Exp 2"]
    means = {}
    sems = {}
    for j, dataset in enumerate(datasets):
        means[j], sems[j] = {}, {}
        d = slopes[(slopes["dataset"] == dataset) & (slopes["grouping"] == "current_coherence_label")]
        for i, cond in enumerate(conds):
            vals = d[d["condition"] == cond]["sdi_slope"]
            means[j][cond] = vals.mean()
            sems[j][cond] = vals.sem()
            ax.bar(
                j + width * (i - 0.5),
                means[j][cond],
                width,
                yerr=sems[j][cond],
                capsize=4,
                color=colors[cond],
                alpha=0.85,
                label=f"{cond} coherence" if j == 0 else None,
            )
    for j, dataset in enumerate(datasets):
        label, _ = get_contrast(dataset, "current_coherence_label")
        top = max(means[j]["Low"] + sems[j]["Low"], means[j]["High"] + sems[j]["High"]) + 0.015
        add_sig_bracket(ax, j - 0.175, j + 0.175, top, label)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x_exp)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Serial Dependence", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(title="Collapsed", frameon=False, fontsize=13, title_fontsize=12, loc="upper right")
    ax.text(0.02, 0.98, "A", transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight="bold")
    main_module.despine(ax)

    ax = axes[1]
    conds = ["Same", "Switch"]
    colors = {"Same": main_module.COLORS["same"], "Switch": main_module.COLORS["switch"]}
    means = {}
    sems = {}
    for j, dataset in enumerate(datasets):
        means[j], sems[j] = {}, {}
        d = slopes[(slopes["dataset"] == dataset) & (slopes["grouping"] == "same_switch_label")]
        for i, cond in enumerate(conds):
            vals = d[d["condition"] == cond]["sdi_slope"]
            means[j][cond] = vals.mean()
            sems[j][cond] = vals.sem()
            ax.bar(
                j + width * (i - 0.5),
                means[j][cond],
                width,
                yerr=sems[j][cond],
                capsize=4,
                color=colors[cond],
                alpha=0.85,
                label=cond if j == 0 else None,
            )
    for j, dataset in enumerate(datasets):
        label, _ = get_contrast(dataset, "same_switch_label")
        top = max(means[j]["Same"] + sems[j]["Same"], means[j]["Switch"] + sems[j]["Switch"]) + 0.015
        add_sig_bracket(ax, j - 0.175, j + 0.175, top, label)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x_exp)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("Serial Dependence", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(title="Collapsed", frameon=False, fontsize=13, title_fontsize=12, loc="upper right")
    ax.text(0.02, 0.98, "B", transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight="bold")
    main_module.despine(ax)

    for axis in axes:
        axis.set_ylim(0, 0.20)
    plt.tight_layout()
    fig.savefig(staging / "fig3_cross_experiment_serial_dependence.png", dpi=300, bbox_inches="tight")
    fig.savefig(staging / "fig3_cross_experiment_serial_dependence.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig4_temporal_window(staging: Path, main_module, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    """Preserve the current Figure 4 style but use block-reset lag/future columns."""

    lag_vars = ["preDur3back", "preDur2back", "preDur1back", "postDur1", "postDur2"]
    lag_labels = ["n-3", "n-2", "n-1", "n+1", "n+2"]

    def summarize(df: pd.DataFrame, group_cols_by_lag: list[str], group_values: list[str]):
        results = {group: {"mean": [], "sem": [], "stars": []} for group in group_values}
        for lag_var, group_col in zip(lag_vars, group_cols_by_lag):
            for group in group_values:
                slopes = []
                group_data = df[df[group_col] == group]
                for _, sub_df in group_data.groupby("subID"):
                    use = sub_df.dropna(subset=[lag_var, "curBias"])
                    if len(use) >= 5:
                        slopes.append(np.polyfit(use[lag_var], use["curBias"], 1)[0])
                mean_slope = np.mean(slopes) if slopes else 0
                sem_slope = sem(slopes) if len(slopes) > 1 else 0
                if len(slopes) >= 5:
                    _, p_raw = ttest_1samp(slopes, 0)
                    p_bonf = min(p_raw * len(lag_vars), 1.0)
                    star = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else ""
                else:
                    star = ""
                results[group]["mean"].append(mean_slope)
                results[group]["sem"].append(sem_slope)
                results[group]["stars"].append(star)
        return results

    df1_lag = df1.copy()
    df2_lag = df2.copy()

    results_exp1 = summarize(df1_lag, ["curCoherenceLevel"] * len(lag_vars), ["Low", "High"])

    lag_coherence_cols = [
        "preCoherence3back",
        "preCoherence2back",
        "preCoherence1back",
        "postCoherence1",
        "postCoherence2",
    ]
    same_switch_cols = []
    for label, coh_col in zip(lag_labels, lag_coherence_cols):
        col = f"SameSwitch_{label.replace('+', 'plus').replace('-', 'minus')}"
        df2_lag[col] = pd.Series(np.nan, index=df2_lag.index, dtype=object)
        valid = df2_lag[coh_col].notna()
        df2_lag.loc[valid, col] = np.where(
            df2_lag.loc[valid, "curCoherence"].eq(df2_lag.loc[valid, coh_col]),
            "Same",
            "Switch",
        )
        same_switch_cols.append(col)
    results_exp2 = summarize(df2_lag, same_switch_cols, ["Same", "Switch"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    x = np.arange(len(lag_labels))
    bar_width = 0.35

    ax = axes[0]
    colors_exp1 = {"High": main_module.COLORS["high"], "Low": main_module.COLORS["low"]}
    labels_exp1 = {"Low": "Current low coherence", "High": "Current high coherence"}
    for i, group in enumerate(["Low", "High"]):
        offset = bar_width * (i - 0.5)
        ax.bar(
            x + offset,
            results_exp1[group]["mean"],
            bar_width,
            yerr=results_exp1[group]["sem"],
            capsize=3,
            label=labels_exp1[group],
            color=colors_exp1[group],
            alpha=0.8,
        )
        for xi, (mean_val, star) in enumerate(zip(results_exp1[group]["mean"], results_exp1[group]["stars"])):
            if star:
                y_pos = mean_val + results_exp1[group]["sem"][xi] + 0.005
                ax.text(xi + offset, y_pos, star, ha="center", va="bottom", fontsize=14)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lag_labels, fontsize=13)
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel("Serial Dependence", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.set_title("A: Experiment 1", loc="left", fontweight="bold", fontsize=16, pad=8)
    main_module.despine(ax)

    ax = axes[1]
    colors_exp2 = {"Same": main_module.COLORS["same"], "Switch": main_module.COLORS["switch"]}
    labels_exp2 = {"Same": "Current vs lag: same", "Switch": "Current vs lag: switch"}
    for i, group in enumerate(["Same", "Switch"]):
        offset = bar_width * (i - 0.5)
        ax.bar(
            x + offset,
            results_exp2[group]["mean"],
            bar_width,
            yerr=results_exp2[group]["sem"],
            capsize=3,
            label=labels_exp2[group],
            color=colors_exp2[group],
            alpha=0.8,
        )
        for xi, (mean_val, star) in enumerate(zip(results_exp2[group]["mean"], results_exp2[group]["stars"])):
            if star:
                y_pos = mean_val + results_exp2[group]["sem"][xi] + 0.005
                ax.text(xi + offset, y_pos, star, ha="center", va="bottom", fontsize=14)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lag_labels, fontsize=13)
    ax.set_xlabel("Lag", fontsize=14)
    ax.set_ylabel("Serial Dependence", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16))
    ax.set_title("B: Experiment 2", loc="left", fontweight="bold", fontsize=16, pad=8)
    main_module.despine(ax)

    for axis in axes:
        axis.set_ylim(-0.07, 0.18)
    plt.tight_layout()
    fig.savefig(staging / "fig4_temporal_window.png", dpi=300, bbox_inches="tight")
    fig.savefig(staging / "fig4_temporal_window.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figA1_response_error_lag(staging: Path, main_module) -> None:
    params = pd.read_csv(RESP_LAG_PARAMS)
    params = params[params["parameter"].str.startswith("response_error_lag")].copy()
    params["lag"] = params["parameter"].str.extract(r"lag(\d+)").astype(int)

    order = ["experiment1", "experiment2"]
    labels = {"experiment1": "Experiment 1", "experiment2": "Experiment 2"}
    colors = {
        "experiment1": main_module.COLORS.get("high", "#2196F3"),
        "experiment2": main_module.COLORS.get("low", "#FF9800"),
    }

    def sig_label(p_value: float) -> str:
        if p_value < 0.001:
            return "***"
        if p_value < 0.01:
            return "**"
        if p_value < 0.05:
            return "*"
        return ""

    fig, ax = plt.subplots(figsize=(8, 4))
    lags = np.arange(1, 6)
    width = 0.35
    for idx, dataset in enumerate(order):
        subset = params[params["dataset"] == dataset].sort_values("lag")
        xpos = lags + width * (idx - 0.5)
        ax.bar(
            xpos,
            subset["beta"],
            width,
            yerr=subset["se"],
            capsize=4,
            color=colors[dataset],
            alpha=0.8,
            label=labels[dataset],
            error_kw=dict(lw=1.2),
        )
        for xval, beta, se_val, p_val in zip(xpos, subset["beta"], subset["se"], subset["p"]):
            ax.text(xval, beta + se_val + 0.002, sig_label(p_val), ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="gray", linestyle="--", alpha=0.5, lw=0.8)
    ax.set_xticks(lags)
    ax.set_xticklabels([f"Lag {i}" for i in lags])
    ax.set_xlabel("Lag")
    ax.set_ylabel(r"Response-error carryover ($b$)")
    ax.legend(frameon=False)
    main_module.despine(ax)
    plt.tight_layout()
    fig.savefig(staging / "figA1_response_error_lag.png", dpi=300, bbox_inches="tight")
    fig.savefig(staging / "figA1_response_error_lag.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    staging = Path(tempfile.mkdtemp(prefix="behavior_figs_"))

    main_figs = import_from_path(
        "current_main_figures", PLOTTING / "generate_main_figures.py"
    )
    main_figs.FIG_DIR = staging
    figure_io = import_from_path("figure_io", PLOTTING / "figure_io.py")

    # Figure 2 uses larger A4-oriented text settings. Keep those rcParams local
    # so the remaining manuscript figures retain the shared Nature style.
    with plt.rc_context():
        fig2 = import_from_path(
            "current_fig2_style", PLOTTING / "figure2_combined.py"
        )
        fig2.OUT_DIR = staging
        fig2.FIG_DIR = staging

        with patched_fig2_contrast_tests(fig2):
            fig2.main()

    df1, df2 = main_figs.load_data()
    df_sdi_1, df_sdi_2 = main_figs.compute_sdi_dataframes(df1, df2)
    main_figs.plot_fig5_cross_experiment(df1, df2, df_sdi_1, df_sdi_2)
    plot_fig3_from_formal_slopes(staging, main_figs)
    plot_fig4_temporal_window(staging, main_figs, df1, df2)
    plot_figA1_response_error_lag(staging, main_figs)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    updated = []
    for stem in CANONICAL_STEMS:
        if figure_io.install_figure_pair(staging, FIG_DIR, stem):
            updated.append(stem)
    shutil.rmtree(staging, ignore_errors=True)
    if updated:
        print(f"Updated behavioral figures in {FIG_DIR}: {', '.join(updated)}")
    else:
        print(f"Verified canonical behavioral figures unchanged: {', '.join(CANONICAL_STEMS)}")


if __name__ == "__main__":
    parse_args()
    main()
