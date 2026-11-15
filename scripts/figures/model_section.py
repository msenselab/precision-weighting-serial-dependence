"""Figures for the modeling section (Section 5) from frozen inputs: no fitting or simulation.

Run from the repository root:
  .venv/bin/python scripts/figures/model_section.py

Outputs (PNG + PDF) go to the figure folder (sdu.paths.FIGURES):
  fig6_parameters_sources   Figure 6: gains, retention, and sources of serial dependence
  fig7_predictive_checks    Figure 7: bias by previous duration and summary statistics, data against model
The remaining parameter estimates (Appendix Table B2) are written to the data folder as table_B2_parameters.csv.
Every plotted value is exported to sdu.paths.FIGURE_DATA / "model_figures".
"""
from pathlib import Path
import os
import sys

for key, value in dict(GKF_SCALE="lin", GKF_CORE="reduced", GKF_BOUNDS="v3", GKF_INIT="gs", GKF_NOISELESS="1",
                       GKF_QDIAG="1", GKF_READOUT="1").items():
    os.environ[key] = value

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "model"))

from sdu import paths as SP                           # noqa: E402
from sdu.model import engine as G                     # noqa: E402
from sdu.model import fitting as FF                   # noqa: E402
from sdu.reports import v3 as R                       # noqa: E402
import fit_minimal_space as M                         # noqa: E402

FIG, DATA = SP.FIGURES, SP.FIGURE_DATA / "model_figures"
RES = ROOT / "results" / "model" / "minimal_qdiag"
BEH = ROOT / "results" / "behavior"

EXPS = R.EXPS                                         # data code 2 = Experiment 1, 1 = Experiment 2
SEL = {2: "C3-R-S0-B2", 1: "C3-R-S3-B2"}
INK, MUTED, GRID, BAND = "#303030", "#7A7A7A", "#ECECEC", "#D9D9D9"
LOW, HIGH, EDGE_LOW, EDGE_HIGH = "#E8503A", "#4BB8D3", "#872C20", "#215664"
COH = {"Low": dict(color=LOW, edge=EDGE_LOW, marker="^"), "High": dict(color=HIGH, edge=EDGE_HIGH, marker="o")}
SS = {"Same": dict(color="#33508a", edge="#1d2f52", marker="o"), "Switch": dict(color="#f08a6b", edge="#8a4a36", marker="s")}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .65, "axes.edgecolor": INK, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
                     "mathtext.default": "regular"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=260, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    print("saved", name)


def export(name, frame):
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / f"{name}.csv", index=False)


def panel_title(ax, s):
    ax.set_title(s, loc="left", fontweight="bold", pad=6)


def sem(x):
    return x.std(ddof=1) / np.sqrt(len(x))


# ---------------------------------------------------------------------------- Figure 6: parameters and sources
def load_fits():
    data = {e: FF.load(e) for e in EXPS}
    ro = pd.concat([pd.read_csv(f) for f in sorted((RES / "readout_space").glob("*.csv"))], ignore_index=True)
    assert ro.groupby(["exp", "Sub"]).model.nunique().eq(148).all()
    return R.Fits(ro, data, dict(M.space()))


def decompose_subjects(fits):
    """R.decompose, kept at the participant level."""
    rows = []
    for e in EXPS:
        label = SEL[e]
        for sub, r in fits.rows(e, label).iterrows():
            th = r[G.NAMES].to_numpy(float)
            th0 = th.copy()
            th0[G.IX["beta"]] = th0[G.IX["wuH"]] = 0.0
            (_, c, kx, _, K, _), coh = fits.observer(e, sub, th, label)
            c0 = fits.observer(e, sub, th0, label)[0][1]
            tot, mem = (c[1:] * K[:-1]).sum(1), (c0[1:] * K[:-1]).sum(1)
            low = coh < 0.5
            for (pl, cl), name in R.TRANS.items():
                w = (low[:-1] == pl) & (low[1:] == cl)
                rows.append(dict(exp=e, Sub=sub, transition=name, mean_route=mem[w].mean(),
                                 trace_route=(tot - mem)[w].mean(), total=tot[w].mean()))
    return pd.DataFrame(rows)


def fig_parameters_sources(fits):
    par = R.participant_quantities(fits, SEL)
    dec = decompose_subjects(fits)
    obs = pd.read_csv(BEH / "analysis" / "four_cells" / "trial_lmm_contrasts.csv")
    obs = obs[obs.contrast.isin(["LL", "HL", "LH", "HH"])].copy()
    obs["exp"] = obs.experiment.map({"Exp1_dynamic": 2, "Exp2_fixed": 1})
    obs["transition"] = obs.contrast.map({"LL": "L→L", "HL": "H→L", "LH": "L→H", "HH": "H→H"})
    export("fig6_participant_quantities", par.assign(experiment=par.exp.map(EXPS)))
    export("fig6_sources_participants", dec.assign(experiment=dec.exp.map(EXPS)))
    export("fig6_observed_transition_slopes", obs)

    order = ["L→L", "H→L", "L→H", "H→H"]                              # previous → current; current Low first
    cur = {"L→L": "Low", "H→L": "Low", "L→H": "High", "H→H": "High"}
    rng = np.random.default_rng(3)
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(width_ratios=[1.0, 1.15, 1.5], wspace=.5,
                                                                      hspace=.86))
    for row, e in enumerate((2, 1)):
        p = par[par.exp == e]
        name = f"{EXPS[e]} ({R.v3(SEL[e])})"
        # --- gains
        ax = axs[row, 0]
        for yi, (stem, lab) in enumerate((("gm", "$K^{(m)}$\nlearned mean"), ("gx", "$K^{(x)}$\ncurrent duration"))):
            for c, off in (("Low", .16), ("High", -.16)):
                v = p[f"{stem}_{c[0]}"]
                ax.errorbar(v.mean(), yi + off, xerr=sem(v), fmt=COH[c]["marker"], ms=5.5, color=COH[c]["edge"],
                            mfc=COH[c]["color"], mec=COH[c]["edge"], lw=1.1, capsize=2, zorder=3)
        ax.set_yticks([0, 1], ["$K^{(m)}$\nlearned mean", "$K^{(x)}$\ncurrent\nduration"])
        ax.set_ylim(-.6, 1.6)
        ax.set_xlim(0, .75)
        ax.set_xticks([0, .25, .5, .75], ["0", ".25", ".5", ".75"])
        ax.set_xlabel("Kalman gain")
        panel_title(ax, f"{'AD'[row]}  Gains")
        ax.text(-.72, 1.2, name, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=INK,
                fontweight="bold", style="italic")
        # --- retention, participants
        ax = axs[row, 1]
        yL, yH = 1 + rng.uniform(-.13, .13, len(p)), 0 + rng.uniform(-.13, .13, len(p))
        ax.plot(np.vstack([p.phiL, p.phiH]), np.vstack([yL, yH]), color="#C8C8C8", lw=.5, zorder=1)
        ax.scatter(p.phiL, yL, s=11, color=LOW, alpha=.55, edgecolors="none", zorder=2)
        ax.scatter(p.phiH, yH, s=11, color=HIGH, alpha=.7, edgecolors="none", zorder=2)
        for c, y, v in (("Low", 1.36, p.phiL), ("High", -.36, p.phiH)):
            ax.errorbar(v.mean(), y, xerr=sem(v), fmt=COH[c]["marker"], ms=6.5, color=COH[c]["edge"],
                        mfc=COH[c]["color"], mec=COH[c]["edge"], lw=1.2, capsize=2.5, zorder=4)
        ax.set_yticks([0, 1], ["$\\phi^H$", "$\\phi^L$"])
        ax.set_ylim(-.75, 1.75)
        ax.set_xlim(-.03, 1.03)
        ax.set_xticks([0, .5, 1], ["0", ".5", "1"])
        ax.set_xlabel("Retention $\\phi$")
        panel_title(ax, f"{'BE'[row]}  Retention of the trace")
        # --- sources of serial dependence
        ax = axs[row, 2]
        d = dec[dec.exp == e].groupby("transition")[["mean_route", "trace_route", "total"]].mean()
        o = obs[obs.exp == e].set_index("transition")
        ys = dict(zip(order, [3.15, 2.25, .9, 0]))
        for t in order:
            y, c = ys[t], cur[t]
            ax.barh(y, d.loc[t, "mean_route"], height=.5, color=BAND, edgecolor="white", lw=.8, zorder=2)
            ax.barh(y, d.loc[t, "trace_route"], left=d.loc[t, "mean_route"], height=.5, color=COH[c]["color"],
                    edgecolor="white", lw=.8, zorder=2)
            ax.errorbar(o.loc[t, "beta"], y, xerr=[[o.loc[t, "beta"] - o.loc[t, "ci_low"]],
                                                    [o.loc[t, "ci_high"] - o.loc[t, "beta"]]],
                        fmt="D", ms=4.2, color=INK, mfc="white", mec=INK, lw=1.0, capsize=2, zorder=4)
        ax.axvline(0, color=MUTED, lw=.7, zorder=1)
        ax.set_yticks([ys[t] for t in order], [t.replace("→", " → ") for t in order])
        ax.set_ylim(-.6, 3.75)
        ax.set_xlim(-.04, .2)
        ax.set_xticks([0, .05, .1, .15, .2], ["0", ".05", ".1", ".15", ".2"])
        ax.set_xlabel("Slope on the previous duration")
        panel_title(ax, f"{'CF'[row]}  Sources of serial dependence")
        for a in axs[row]:
            a.tick_params(axis="y", length=0)
            a.grid(axis="x", color=GRID, lw=.6)
            a.set_axisbelow(True)
    handles = [Line2D([], [], marker="^", ls="none", mfc=LOW, mec=EDGE_LOW, ms=5.5, label="current Low"),
               Line2D([], [], marker="o", ls="none", mfc=HIGH, mec=EDGE_HIGH, ms=5.5, label="current High"),
               Patch(facecolor=BAND, label="through the learned mean"),
               Patch(facecolor=LOW, label="through the retained trace"), Patch(facecolor=HIGH, label=""),
               Line2D([], [], marker="D", ls="-", color=INK, mfc="white", mec=INK, ms=4.2, lw=1,
                      label="observed slope (95% CI)")]
    fig.legend(handles=[handles[0], handles[1], handles[2], (handles[3], handles[4]), handles[5]],
               labels=["current Low", "current High", "through the learned mean", "through the retained trace",
                       "observed slope (95% CI)"],
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None, pad=.2)},
               loc="upper center", bbox_to_anchor=(.5, 1.1), ncol=3, frameon=False, handletextpad=.4,
               columnspacing=1.6)
    save(fig, "fig6_parameters_sources")
    return par, dec, obs


# ---------------------------------------------------------------------------- Figure 7: predictive checks
def fig_predictive_checks():
    ppc = RES / "readout_space_ppc"
    s, h = pd.read_csv(ppc / "summary.csv"), pd.read_csv(ppc / "history_curves.csv")
    assert s.n_cohorts.eq(100).all() and h.n_cohorts.eq(100).all()
    export("fig7_summary", s)
    export("fig7_history_curves", h)

    fig = plt.figure(figsize=(7.2, 7.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.45, .8], hspace=.3)
    top = outer[0].subgridspec(2, 2, hspace=.55, wspace=.34)
    bottom = outer[1].subgridspec(1, 3, wspace=.16)
    ylim = {2: (-.04, .20), 1: (-.12, .12)}                           # equal spans: slopes are comparable
    for row, (grouping, styles, what) in enumerate((("cur_coh", COH, "current coherence"),
                                                    ("same_switch", SS, "coherence transition"))):
        for col, e in enumerate((2, 1)):
            ax = fig.add_subplot(top[row, col])
            ends = []
            d = h[(h.exp == e) & (h.grouping == grouping)]
            for i, (level, st) in enumerate(styles.items()):
                g = d[d.level == level].sort_values("previous_duration")
                x = g.previous_duration.to_numpy()
                ax.fill_between(x, g.lo, g.hi, color=st["color"], alpha=.22, lw=0, zorder=1)
                ax.plot(x, g.simulated, color=st["color"], lw=1.6, zorder=2)
                ax.errorbar(x + (-.012 if i == 0 else .012), g.observed, yerr=g.observed_sem, fmt=st["marker"],
                            ms=5, color=st["edge"], mfc=st["color"], mec=st["edge"], lw=1, capsize=2, zorder=3)
                ends.append((g.simulated.iloc[-1], level))
            ends.sort()
            gap = .024                                                # keep the two direct labels apart
            if ends[1][0] - ends[0][0] < gap:
                mid = (ends[0][0] + ends[1][0]) / 2
                ends = [(mid - gap / 2, ends[0][1]), (mid + gap / 2, ends[1][1])]
            for yend, level in ends:
                ax.annotate(level, (1.6, yend), xytext=(13, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK, annotation_clip=False)
            ax.axhline(0, color=MUTED, lw=.6, ls=":", zorder=0)
            ax.set_xlim(.74, 1.66)
            ax.spines["bottom"].set_bounds(.74, 1.66)
            ax.set_ylim(*ylim[e])
            ax.set_xticks([.8, 1.0, 1.2, 1.4, 1.6])
            ax.set_xlabel("Previous duration (s)")
            if col == 0:
                ax.set_ylabel("Reproduction bias (s)")
            ax.grid(axis="y", color=GRID, lw=.6)
            ax.set_axisbelow(True)
            panel_title(ax, f"{'ABCD'[row * 2 + col]}  {EXPS[e]}: by {what}")

    rows = [(2, "Low", "Exp 1  Low"), (2, "High", "Exp 1  High"), (1, "Low", "Exp 2  Low"), (1, "High", "Exp 2  High")]
    ypos = [3.2, 2.4, 1.0, .2]
    panels = [("b1", -1, "Central tendency", "−slope, current duration"),
              ("b2", 1, "Serial dependence", "slope, previous duration"),
              ("rsd", 1, "Response variability", "residual SD (s)")]
    for k, (stem, sign, title, xlab) in enumerate(panels):
        ax = fig.add_subplot(bottom[0, k])
        for (e, c, _), y in zip(rows, ypos):
            r = s[(s.exp == e) & (s.statistic == f"{stem}_{c}")].iloc[0]
            lo, hi = sorted((sign * r.lo, sign * r.hi))
            ax.barh(y, hi - lo, left=lo, height=.5, color=COH[c]["color"], alpha=.28, lw=0, zorder=1)
            ax.plot([sign * r.simulated] * 2, [y - .25, y + .25], color=COH[c]["color"], lw=1.8, zorder=2)
            ax.errorbar(sign * r.observed, y, xerr=r.observed_sem, fmt=COH[c]["marker"], ms=5.2,
                        color=COH[c]["edge"], mfc=COH[c]["color"], mec=COH[c]["edge"], lw=1.05, capsize=2, zorder=3)
        ax.set_yticks(ypos, [lab for *_, lab in rows] if k == 0 else [""] * 4)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-.4, 3.8)
        ax.set_xlabel(xlab)
        ax.grid(axis="x", color=GRID, lw=.6)
        ax.set_axisbelow(True)
        ax.margins(x=.12)
        panel_title(ax, f"{'EFG'[k]}  {title}")
    handles = [Line2D([], [], marker="o", ls="none", mfc=MUTED, mec=INK, ms=5, label="observed (mean ± SEM)"),
               (Patch(facecolor=MUTED, alpha=.3), Line2D([], [], color=MUTED, lw=1.8))]
    fig.legend(handles=handles, labels=["observed (mean ± SEM)", "model (mean and 95% range of 100 simulated cohorts)"],
               handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=1, pad=0)},
               loc="upper center", bbox_to_anchor=(.5, .95), ncol=2, frameon=False)
    save(fig, "fig7_predictive_checks")
    return s, h


def table_b2(par):
    """Parameters of the selected models that Figure 6 does not show (Appendix Table B2)."""
    names = [("q1", "q1 (fast state, R units)"), ("q2", "q2 (slow state, R units)"),
             ("sbL", "sigma_b, current Low (s)"), ("sbH", "sigma_b, current High (s)"),
             ("aq3", "alpha_q3"), ("lam", "lambda")]
    rows = []
    for col, label in names:
        row = dict(parameter=label)
        for e in (2, 1):
            v = par.loc[par.exp == e, col]
            row.update({f"{EXPS[e]} mean": v.mean(), f"{EXPS[e]} SEM": sem(v), f"{EXPS[e]} median": v.median()})
        rows.append(row)
    tab = pd.DataFrame(rows)
    export("table_B2_parameters", tab)
    return tab


if __name__ == "__main__":
    import matplotlib.legend_handler  # noqa: F401
    fig_predictive_checks()
    fits = load_fits()
    par, dec, obs = fig_parameters_sources(fits)
    pd.set_option("display.width", 200)
    print(dec.groupby(["exp", "transition"])[["mean_route", "trace_route", "total"]].mean().round(3))
    print(par.groupby("exp")[["gx_L", "gx_H", "gm_L", "gm_H", "phiL", "phiH"]].mean().round(3))
    tab = table_b2(par)
    print(tab.round(3).to_string(index=False))
