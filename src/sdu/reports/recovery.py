"""Plots for the v3 parameter-recovery notebook. All annotations stay outside the data axes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

EXPERIMENT = {2: "Experiment 1", 1: "Experiment 2"}
COLORS = {2: "#287CA6", 1: "#617D71"}
DISPLAY_NAMES = {
    "lq1": "σq₁ (s)", "lq2": "σq₂ (s)", "beta": "φᴸ", "wuH": "φᴴ",
    "lam": "λ", "lq3": "σq₃ (s)", "d0": "d₀ (s)", "ad0": "Δd₀ (s)",
    "lsde": "σₑ (s)", "aq3": "ρq₃", "delta_phi": "Δφ", "delta_weight": "Δw",
}


def style():
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlepad": 10, "axes.axisbelow": True, "savefig.bbox": "tight"})


def number(value):
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def recovery_table(summary, parameters, *, show_means=False, boundary_markers=False):
    """Place the two experiments side by side using the saved recovery summaries."""
    local = summary[summary.design == "fitted"]
    rows = []
    for name in parameters:
        d = local[local.parameter == name].set_index("exp")
        marker = "†" if boundary_markers and ((d.lower_rate >= .2) | (d.upper_rate >= .2)).any() else ""
        row = {("Parameter", ""): DISPLAY_NAMES[name] + marker}
        for exp in (2, 1):
            value = d.loc[exp]
            label = EXPERIMENT[exp]
            row[label, "r: mean / cohort"] = f"{number(value.r_mean)} / {number(value.r)}"
            if show_means:
                row[label, "Generating mean"] = value.true_mean
                row[label, "Recovered mean"] = value.fitted_mean
            else:
                row[label, "Bias"] = value.bias
                row[label, "RMSE"] = value.mean_participant_rmse
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    numeric = [c for c in frame.columns if c[1] not in ("", "r: mean / cohort")]
    precision = 4 if show_means else 3
    formats = {c: lambda v: f"{0.0 if abs(v) < .5*10**-precision else v:.{precision}f}" for c in numeric}
    return (frame.style.format(formats, na_rep="—").hide(axis="index")
            .set_properties(**{"white-space": "nowrap", "padding": "6px 10px"})
            .set_table_styles([
                {"selector": "", "props": [("margin", "0 auto")]},
                {"selector": "th.col_heading.level0", "props": [("text-align", "center")]},
                {"selector": "th.col_heading.level1", "props": [("text-align", "right")]},
            ]))


def recovery_legend(fig):
    handles = [Line2D([], [], marker="o", ls="none", color="#526D7A", alpha=.35, markersize=5),
               Line2D([], [], marker="o", ls="none", color="#526D7A", markeredgecolor="white", markersize=7),
               Line2D([], [], color="#8B989E", ls="--", lw=1)]
    fig.legend(handles, ["Single recovery", "Participant mean", "Exact recovery"],
               loc="outside lower center", ncol=3, frameon=False)


def scatter_comparison(table, parameters):
    """Rows are parameters, columns are experiments; both experiments share each row's scale."""
    fig, axes = plt.subplots(len(parameters), 2, figsize=(8.2, 3.5 * len(parameters)),
                             squeeze=False, layout="compressed")
    fig.get_layout_engine().set(h_pad=.10, w_pad=.08, hspace=.06, wspace=.06)
    local = table[table.design == "fitted"]
    for row, (name, meta) in enumerate(parameters.items()):
        both = local[local.parameter == name]
        lo = min(both.true.min(), both.fitted.min())
        hi = max(both.true.max(), both.fitted.max())
        pad = max((hi-lo)*.06, .005)
        for col, exp in enumerate((2, 1)):
            ax = axes[row, col]
            d = both[both.exp == exp]
            mean = d.groupby("Sub")[["true", "fitted"]].mean()
            ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="#8B989E", ls="--", lw=1)
            ax.scatter(d.true, d.fitted, color=COLORS[exp], s=13, alpha=.14, edgecolor="none")
            ax.scatter(mean.true, mean.fitted, color=COLORS[exp], s=32,
                       edgecolor="white", linewidth=.5)
            ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad),
                   title=f"{EXPERIMENT[exp]} — {meta[0]}")
            if row == len(parameters)-1:
                ax.set_xlabel("Generating value")
            if col == 0:
                ax.set_ylabel("Recovered value")
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
            ax.grid(color="#E7EBEC", lw=.5)
            ax.set_aspect("equal", adjustable="box")
    recovery_legend(fig)
    return fig


def scatter_grid(table, summary, parameters, exp, design, columns=None):
    count = len(parameters)
    columns = columns or (5 if count > 5 else count)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.9, rows * 3.6), squeeze=False,
                             layout="constrained")
    fig.get_layout_engine().set(h_pad=0.10, w_pad=0.06, hspace=0.08, wspace=0.04)
    d = table[(table.exp == exp) & (table.design == design)]
    s = summary[(summary.exp == exp) & (summary.design == design)].set_index("parameter")
    for ax, (name, meta) in zip(axes.flat, parameters.items()):
        g = d[d.parameter == name]
        lo, hi = min(g.true.min(), g.fitted.min()), max(g.true.max(), g.fitted.max())
        pad = max((hi - lo) * 0.06, 0.0001)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#A1A9AD", ls="--", lw=1)
        ax.scatter(g.true, g.fitted, color=COLORS[exp], s=12 if design == "fitted" else 15,
                   alpha=0.12 if design == "fitted" else 0.55, edgecolor="none", label="Individual fit")
        if design == "fitted":
            mean = g.groupby("Sub")[["true", "fitted"]].mean()
            ax.scatter(mean.true, mean.fitted, color=COLORS[exp], s=24, edgecolor="white", lw=0.4,
                       label="Mean of 20 fits")
        ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad))
        caption = (f"mean r {number(s.loc[name, 'r_mean'])} · cohort r {number(s.loc[name, 'r'])}"
                   if design == "fitted" else f"r {number(s.loc[name, 'r'])} · slope {number(s.loc[name, 'slope'])}")
        ax.set_title(f"{meta[0]}\n{caption}", fontsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        fmt = FuncFormatter(lambda x, _: f"{x:.2f}")
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.grid(color="#E7EBEC", lw=0.5)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes.flat[count:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("Generating value", fontsize=10)
    fig.supylabel("Recovered value", fontsize=11)
    title = "Observed parameter values" if design == "fitted" else "Sampled parameter values"
    fig.suptitle(f"{EXPERIMENT[exp]} — {title}", fontsize=13, weight="bold")
    if design == "fitted":
        recovery_legend(fig)
    return fig


def condition_means(table):
    """Mean by condition, with the 5th–95th range over the 20 simulated cohort means."""
    pairs = [("beta", "wuH", r"Readout continuity $\phi$"),
             ("weight_low", "weight_high", r"Trace weight $(1-K^{(x)})\phi$"),
             ("bias_sd_low", "bias_sd_high", "Bias innovation SD (s)")]
    d = table[table.design == "fitted"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey="col", layout="constrained")
    for row, exp in enumerate((2, 1)):
        for ax, (low, high, title) in zip(axes[row], pairs):
            g = d[(d.exp == exp) & d.parameter.isin((low, high))]
            true = g.groupby("parameter").true.mean().reindex((high, low))
            cohort = g.groupby(["parameter", "rep"]).fitted.mean().unstack("parameter")[[high, low]]
            mean, lower, upper = cohort.mean(), cohort.quantile(0.05), cohort.quantile(0.95)
            x = np.arange(2)
            ax.bar(x - 0.17, true, 0.32, color="#CED5D8", label="True")
            ax.bar(x + 0.17, mean, 0.32, color=COLORS[exp], label="Recovered",
                   yerr=np.array([mean - lower, upper - mean]), capsize=3)
            ax.set_xticks(x, ["High", "Low"])
            ax.set_title(title if row == 0 else "", fontsize=11)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
            ax.grid(axis="y", color="#E7EBEC", lw=0.5)
        axes[row, 0].set_ylabel(EXPERIMENT[exp], labelpad=12, weight="bold")
    for col in range(3):
        upper = max(ax.dataLim.ymax for ax in axes[:, col])
        axes[0, col].set_ylim(0, upper * 1.10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    return fig


def contrast_distributions(contrasts):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2), layout="constrained")
    local = contrasts[contrasts.design == "fitted"]
    for row, exp in enumerate((2, 1)):
        for col, (parameter, label) in enumerate((("delta_phi", r"Mean $\phi^L-\phi^H$"),
                                                  ("delta_weight", r"Mean $w^L-w^H$"))):
            ax = axes[row, col]
            d = local[(local.exp == exp) & (local.parameter == parameter)]
            ax.hist(d.fitted_mean, bins=8, color=COLORS[exp], edgecolor="white", alpha=0.75)
            ax.axvline(d.true_mean.iloc[0], color="#252A2D", ls="--", lw=1.5, label="True mean")
            ax.axvline(0, color="#A9B0B2", lw=0.8)
            ax.set_title(f"{EXPERIMENT[exp]} — {label}", fontsize=11)
            ax.set_xlabel("Recovered cohort mean")
            ax.set_ylabel("Cohorts")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False)
    return fig


def error_matrix(errors, parameters):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), layout="constrained")
    short = [meta[0].replace(" (s)", "") for meta in parameters.values()]
    for ax, exp in zip(axes, (2, 1)):
        m = errors[errors.exp == exp].pivot(index="parameter_a", columns="parameter_b", values="r")
        matrix = m.reindex(index=parameters, columns=parameters).to_numpy()
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(parameters)), short, rotation=45, ha="right")
        ax.set_yticks(range(len(parameters)), short)
        ax.set_title(EXPERIMENT[exp])
    fig.colorbar(im, ax=axes, shrink=0.65, label="Correlation of estimation errors")
    return fig
