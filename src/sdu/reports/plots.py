"""Shared plotting style and figure primitives for the notebooks."""
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COH = {"Low": "#E8503A", "High": "#4BB8D3"}
EXP_COL = {1: "#26ae96", 2: "#98a3c1"}              # Experiment 1, Experiment 2
SAME_SWITCH = {"Same": "#33508a", "Switch": "#f08a6b"}


def b2_bias_sd(p, coh):
    """B2 response-bias drift SD in seconds at coherence coh: sqrt(q3s * exp(alpha_q3 (1 - coh))).

    B2's bias lives in the response, in seconds; the likelihood and the simulator use p["q3s"] (s^2). p["q3"] is the
    observer-normalised quantity of the bias levels that enter the observer and must not be reported for B2."""
    return np.sqrt(p["q3s"] * np.exp(p["aq3"] * (1 - coh)))


def use_style():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                         "axes.formatter.useoffset": False})


def grouped_bars(ax, groups, items, ylabel, log=False, legend_loc="upper right"):
    """Two (or more) bars per quantity on one x axis, mean ± SEM over rows, with the mean printed on each bar.

    groups: list of (label, frame, color); items: list of (column, tick label)."""
    xs = np.arange(len(items))
    width = 0.76 / len(groups)
    for j, (label, frame, color) in enumerate(groups):
        mean = np.array([frame[c].mean() for c, _ in items])
        sem = np.array([frame[c].std(ddof=1) / np.sqrt(frame[c].notna().sum()) for c, _ in items])
        x = xs + (j - (len(groups) - 1) / 2) * width
        ax.bar(x, mean, width, color=color, edgecolor=INK, lw=.8, label=label, zorder=2)
        ax.errorbar(x, mean, sem, fmt="none", ecolor=INK, capsize=4, lw=1.3, zorder=3)
        for xi, mi, si in zip(x, mean, sem):
            up = mi >= 0 or log
            ax.annotate(f"{mi:.2f}", (xi, mi + si if up else mi - si), xytext=(0, 4 if up else -4),
                        textcoords="offset points", ha="center", va="bottom" if up else "top", color=color,
                        fontsize=10)
    if log:
        ax.set_yscale("log")
    else:
        ax.axhline(0, color="#bdbdbd", lw=1, ls="--", zorder=1)
    ax.set_xticks(xs, [label for _, label in items])
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(loc=legend_loc)
    return ax


def model_line(ax, x, mean, lo, hi, color, label):
    """Model: dashed line with a band."""
    ax.fill_between(x, lo, hi, color=color, alpha=.18, lw=0)
    ax.plot(x, mean, "--", color=color, lw=2.2, label=label)


def observed_points(ax, x, mean, sem, color=INK, label="Observed", ms=7):
    """Data: points with error bars."""
    ax.errorbar(x, mean, sem, fmt="o", ms=ms, color=color, capsize=3, lw=1.4, zorder=5, label=label)


def zero_line(ax):
    ax.axhline(0, color="#9e9e9e", lw=.9, ls=":")
