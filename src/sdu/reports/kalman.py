"""Helpers for notebooks/three_state_kalman.ipynb: the three-state Kalman observer on the two experiments
(Section 5, Appendix B)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

from sdu.model import engine as G
from sdu.model import comparison as SC
from sdu.model import fitting as FF
from sdu.reports import plots as P

EXPS = FF.EXPERIMENT
FOCAL = {1: "low_high", 2: "LL_other_three"}
TRANS = {(1, 1): "L→L", (0, 1): "H→L", (1, 0): "L→H", (0, 0): "H→H"}


class Fits:
    """Fitted parameters (results/model/fits/all.csv) and trial data (fitting.load)."""

    def __init__(self, fits, data):
        self.ro, self.data = fits, data

    def rows(self, e, label):
        return self.ro[(self.ro.exp == e) & (self.ro.model == label)].set_index("Sub")

    def observer(self, e, sub, theta, label):
        T, coh = self.data[e][sub][0], self.data[e][sub][1]
        return G.observer(theta, G.Model.parse(label), T, coh), coh


# ------------------------------------------------------------------ Part I: closed form
def closed_form(th, model, z, coh):
    """Readout of the observer (B0/B2, S0/S3) from the scalar recursion of the derivation."""
    p = G.unpack(th)
    phL, phH = p["phL"], p["phH"]
    reset, sw = model.s == 3, G.switch_flags(coh)
    m, M, f_prev = G.MU0, G.MEM0, 0.0
    u = np.empty(len(z))
    for i, c in enumerate(coh):
        q1, q2 = p["q1"] * np.exp(p["aq1"] * (1 - c)), p["q2"] * np.exp(p["aq2"] * (1 - c))
        S = M + q1 + (p["rlow"] if c < 0.5 else 1.0)
        kx, km = (M + q1) / S, M / S
        x_post = (1 - kx) * m + kx * z[i]                          # boxed x_i
        phi = 0.0 if (reset and sw[i]) else (phL if c < 0.5 else phH)
        u[i] = x_post + (1 - kx) * phi * f_prev                     # readout u_i
        f_prev = (kx - km) * (z[i] - m)                             # f_i
        m, M = m + km * (z[i] - m), M + q2 - M ** 2 / S             # m_i, P^(m)_i
    return u


def check_closed_form(fits, cases, n_subs=2):
    rows = []
    for exp, label in cases:
        for sub in sorted(fits.data[exp])[:n_subs]:
            th = FF.theta(fits.rows(exp, label).loc[sub])
            z, coh = fits.data[exp][sub][0], fits.data[exp][sub][1]
            model = G.Model.parse(label)
            u = G.observer(th, model, z, coh)[0]
            rows.append(dict(experiment=EXPS[exp], Sub=sub, model=label,
                             max_abs_diff=np.abs(u - closed_form(th, model, z, coh)).max()))
    return pd.DataFrame(rows)


def slope_formula(th, model, coh, g, K):
    """Slope of u_i on s_{i-1} for every level (entry k is trial i = k + 1):
    (1 - Kx_i) [(1 - gamma sw_i) Km_{i-1} + phi_i (Kx_{i-1} - Km_{i-1}) + F13 Kb_{i-1}].
    g and K are the engine's gains; K is in the coordinates (m, f = x - m, b), so K[:, 1] = Kx - Km."""
    p = G.unpack(th)
    sw = G.switch_flags(coh)
    phi = np.where(coh < 0.5, p["phL"], p["phH"]) * ~(sw & (model.s == 3))   # S3: phi = 0 on a switch
    keep = np.where(sw & (model.s == 1), 1 - p["sw"], 1.0)                   # S1: the mean reverts by gamma
    f13 = float(model.b == 1)
    return (1 - g[1:]) * (keep[1:] * K[:-1, 0] + phi[1:] * K[:-1, 1] + f13 * K[:-1, 2])


def check_slopes(fits, cases, n_subs=2, n_trials=25, h=0.1):
    """The two slopes against the engine: a finite difference of the engine's readout (one duration moved by h
    seconds; the readout is linear in the durations) and the engine's linear maps c_i K_{i-1}."""
    rows, rng = [], np.random.default_rng(0)
    for exp, label in cases:
        spec = G.Model.parse(label)
        for sub in sorted(fits.data[exp])[:n_subs]:
            th = FF.theta(fits.rows(exp, label).loc[sub])
            z, coh = fits.data[exp][sub][0], fits.data[exp][sub][1]
            u, c, g, _, K, _ = G.observer(th, spec, z, coh)
            sd = slope_formula(th, spec, coh, g, K)
            d_ct, d_sd = [], []
            for j in rng.choice(np.arange(1, len(z) - 1), n_trials, replace=False):
                zz = z.copy(); zz[j] += h
                du = (G.observer(th, spec, zz, coh)[0] - u) / h
                d_ct.append(du[j] - g[j]); d_sd.append(du[j + 1] - sd[j])
            rows.append(dict(experiment=EXPS[exp], Sub=sub, model=label,
                             central_tendency=np.abs(d_ct).max(), serial_dependence=np.abs(d_sd).max(),
                             linear_maps=np.abs(sd - (c[1:] * K[:-1]).sum(1)).max()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Part II: screening and AIC
def load_screening(path):
    T = pd.read_csv(path)
    T["label"] = T.model
    T["dAIC"] = T.AIC - T.groupby("exp").AIC.transform("min")
    T["rank"] = T.groupby("exp").AIC.rank(method="first").astype(int)
    stats = {e: SC.CENTRAL_TENDENCY + SC.SERIAL_DEPENDENCE[e] + SC.RESPONSE_VARIABILITY for e in EXPS}
    for e in EXPS:
        Z = T.loc[T.exp == e, [f"z_{s}" for s in stats[e]]].abs()
        T.loc[T.exp == e, "n_within"] = (Z < SC.Z_PASS).sum(axis=1)
        T.loc[T.exp == e, "sum_z2"] = (Z ** 2).sum(axis=1)
    return T, stats


def top_models(T, e, n=10):
    cols = ["rank", "label", "k", "dAIC", "passes_focal_contrasts", "passes_all_three", "n_within", "sum_z2"]
    return T[T.exp == e].sort_values("AIC")[cols].head(n).set_index("rank")


def by_dimension(T, dim):
    t = (T.groupby(["exp", dim])
         .agg(models=("label", "size"), best_dAIC=("dAIC", "min"), focal=("passes_focal_contrasts", "sum"),
              all_three=("passes_all_three", "sum"))
         .unstack("exp"))
    t.columns = [f"{EXPS[e]} {c}" for c, e in t.columns]
    return t.loc[[f"C{i}" for i in range(1, 16)]] if dim == "C" else t


def c_level_comparison(fits, T, selected, c_levels=("C3", "C1", "C4", "C9")):
    """C3 against C levels that add q1 or R, each with the selected S and B: AIC, checks, gain by coherence."""
    rows = []
    for e in EXPS:
        s_b = selected[e].split("-", 1)[1]
        for c in c_levels:
            label = f"{c}-{s_b}"
            kl, kh = [], []
            for sub, r in fits.rows(e, label).iterrows():
                (_, _, kx, _, _, _), coh = fits.observer(e, sub, FF.theta(r), label)
                kl.append(kx[coh < .5].mean()); kh.append(kx[coh >= .5].mean())
            t = T[(T.exp == e) & (T.model == label)].iloc[0]
            rows.append(dict(experiment=EXPS[e], model=label, rank=t["rank"], k=t.k, dAIC=t.dAIC,
                             z_focal=t[f"z_{FOCAL[e]}"], z_HH=t.z_b2_HH,
                             response_variability=t.passes_response_variability, all_three=t.passes_all_three,
                             Kx_Low=np.mean(kl), Kx_High=np.mean(kh),
                             gain_factor=(1 - np.mean(kl)) / (1 - np.mean(kh))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ §3 behavioural checks
SUMMARY_PANELS = [("Central tendency", "b1", "−slope on current duration", -1),
                  ("Serial dependence", "b2", "slope on previous duration", 1),
                  ("Response variability", "rsd", "residual SD (s)", 1)]
CURVE_PANELS = (("central tendency", "Current duration (s)", "Reproduction bias (s)"),
                ("serial dependence", "Previous duration (s)", "Residual bias (s)"),
                ("response variability", "Current duration (s)", "Within-duration residual SD (s)"))
HISTORY_STYLE = {"cur_coh": {"Low": ("Low coherence", P.COH["Low"]), "High": ("High coherence", P.COH["High"])},
                 "same_switch": {"Same": ("Same", P.SAME_SWITCH["Same"]), "Switch": ("Switch", P.SAME_SWITCH["Switch"])}}
HISTORY_TITLE = {"cur_coh": "Current coherence", "same_switch": "Same vs. Switch"}


def plot_summary_bars(ppc_s, models):
    """Data (filled, mean ± SEM) and model (hatched, mean and 95% cohort range) by current coherence."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for row, e in zip(axes, EXPS):
        d = ppc_s[(ppc_s.exp == e) & (ppc_s.model == models[e])].set_index("statistic")
        for ax, (title, stem, ylab, sign) in zip(row, SUMMARY_PANELS):
            for xi, c in enumerate(("Low", "High")):
                r = d.loc[f"{stem}_{c}"]
                ob, sim = sign * r.observed, sign * r.simulated
                lo, hi = sorted((sign * r.lo, sign * r.hi))
                ax.bar(xi - .19, ob, .36, color=P.COH[c], edgecolor=P.COH[c], lw=1.2, zorder=2)
                ax.errorbar(xi - .19, ob, r.observed_sem, fmt="none", ecolor=P.INK, capsize=4, lw=1.3, zorder=3)
                ax.bar(xi + .19, sim, .36, color="white", edgecolor=P.COH[c], hatch="////", lw=1.2, zorder=2)
                ax.errorbar(xi + .19, sim, [[sim - lo], [hi - sim]], fmt="none", ecolor=P.INK, capsize=4, lw=1.3,
                            zorder=3)
            ax.axhline(0, color=P.MUTED, lw=.8)
            ax.set_xticks([0, 1], ["current Low", "current High"])
            ax.set_xlim(-.6, 1.6)
            ax.grid(axis="x", visible=False)
            ax.set_ylabel(ylab)
            ax.set_title(f"{EXPS[e]} · {title}", loc="left", fontsize=11)
        row[0].text(-.32, .5, models[e], transform=row[0].transAxes, rotation=90, va="center", ha="center",
                    fontsize=10, color=P.MUTED, style="italic")
    fig.legend([Patch(facecolor=P.MUTED, edgecolor=P.MUTED), Patch(facecolor="white", edgecolor=P.MUTED, hatch="////")],
               ["Data (mean ± SEM)", "Model (mean, 95% cohort range)"], loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, 1))
    return fig


def plot_duration_curves(ppc_c, e, model):
    """Duration-resolved data and model, Low and High coherence in two columns."""
    sub = ppc_c[(ppc_c.exp == e) & (ppc_c.model == model)]
    fig, axes = plt.subplots(3, 2, figsize=(9.8, 9.0))
    for r_i, (panel, xlabel, ylabel) in enumerate(CURVE_PANELS):
        for c_i, c in enumerate(("Low", "High")):
            ax = axes[r_i, c_i]
            cv = sub[(sub.panel == panel) & (sub.coherence == c)].sort_values("duration")
            x = cv.duration.to_numpy(float)
            ax.fill_between(x, cv.lo, cv.hi, color=P.COH[c], alpha=.16, lw=0, label="95% simulated-cohort range")
            ax.plot(x, cv.simulated, color=P.COH[c], lw=2.0, label="Model simulation")
            ax.errorbar(x, cv.observed, cv.observed_sem, fmt="o", ms=5, capsize=2.5, mfc="white", mec=P.COH[c],
                        mew=1.3, color=P.COH[c], lw=1.0, zorder=5, label="Observed ± SEM")
            if r_i < 2:
                ax.axhline(0, color="#888888", lw=.7, ls=":")
            ax.set_xticks([.8, 1.0, 1.2, 1.4, 1.6])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if r_i == 0:
                ax.set_title(f"{c} coherence", color=P.COH[c])
        lims = [axes[r_i, k].get_ylim() for k in range(2)]
        for k in range(2):
            axes[r_i, k].set_ylim(min(v[0] for v in lims), max(v[1] for v in lims))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(.5, 0))
    fig.suptitle(f"{EXPS[e]}: {model}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, .03, 1, .97))
    return fig


def plot_history_curves(ppc_h, models):
    """Bias by previous duration, by current coherence (top) and by same/switch (bottom); points data, dashed model."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharey="row")
    for r_i, gr in enumerate(HISTORY_STYLE):
        for c_i, e in enumerate(EXPS):
            ax = axes[r_i, c_i]
            d = ppc_h[(ppc_h.exp == e) & (ppc_h.model == models[e]) & (ppc_h.grouping == gr)]
            for lev, (name, col) in HISTORY_STYLE[gr].items():
                cv = d[d.level == lev].sort_values("previous_duration")
                x = cv.previous_duration.to_numpy(float)
                P.model_line(ax, x, cv.simulated, cv.lo, cv.hi, col, f"{name} (Model)")
                P.observed_points(ax, x, cv.observed, cv.observed_sem, col, f"{name} (Observed)")
            P.zero_line(ax)
            ax.set_xticks([.8, 1.0, 1.2, 1.4, 1.6])
            ax.set_xlabel("Previous duration (s)")
            ax.set_ylabel("Bias (s)")
            ax.set_title(f"{'ABCD'[2 * r_i + c_i]}: {EXPS[e].replace('Exp', 'Experiment')} — {HISTORY_TITLE[gr]}",
                         loc="left", fontsize=12)
            ax.text(.98, .04, f"Model: {models[e]}", transform=ax.transAxes, ha="right", fontsize=10,
                    color=P.MUTED, style="italic")
            ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=.9)
            ax.tick_params(labelleft=True)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ §4 parameters and gains
def participant_quantities(fits, selected):
    rows = []
    for e in EXPS:
        label = selected[e]
        model = G.Model.parse(label)
        for sub, r in fits.rows(e, label).iterrows():
            th = FF.theta(r)
            p = G.unpack(th)
            phL, phH = p["phL"], p["phH"]
            (_, _, kx, _, K, _), coh = fits.observer(e, sub, th, label)
            reset, sw = model.s == 3, G.switch_flags(coh)
            assert model.b == 2, "B2 models only (the bias is reported in seconds)"
            row = dict(exp=e, Sub=sub, q1=p["q1"], q2=p["q2"], aq3=p["aq3"], lam=p["lam"], phiL=phL, phiH=phH,
                       sbL=P.b2_bias_sd(p, 0.3), sbH=P.b2_bias_sd(p, 0.7))
            for name, level in (("L", coh < 0.5), ("H", coh >= 0.5)):
                phi = phL if name == "L" else phH
                row[f"gx_{name}"] = kx[level].mean()
                row[f"gm_{name}"] = K[level, 0].mean()
                row[f"ro_{name}"] = ((1 - kx) * np.where(reset & sw, 0.0, phi))[level].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_parameters(par, selected):
    groups = [(f"{EXPS[e]} ({selected[e]})", par[par.exp == e], P.EXP_COL[e]) for e in EXPS]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    P.grouped_bars(ax, groups, [("q1", "$q_1$\n(fast state,\n$R$ units)"), ("q2", "$q_2$\n(slow state,\n$R$ units)"),
                                ("sbL", "$\\sigma_b$ (s)\nbias drift SD,\ncurrent Low"),
                                ("sbH", "$\\sigma_b$ (s)\nbias drift SD,\ncurrent High"),
                                ("aq3", "$\\alpha_{q_3}$\n(coherence\nmodulation)"),
                                ("lam", "$\\lambda$\n(bias\ndecay)"), ("phiL", "$\\phi^L$\n(readout,\ncurrent Low)"),
                                ("phiH", "$\\phi^H$\n(readout,\ncurrent High)")], "Parameter value")
    ax.set_title("Parameters", loc="left")
    fig.tight_layout()
    return fig


def plot_gains(par, selected):
    groups = [(f"{EXPS[e]} ({selected[e]})", par[par.exp == e], P.EXP_COL[e]) for e in EXPS]
    fig, ax = plt.subplots(figsize=(13, 5.2))
    P.grouped_bars(ax, groups, [("gx_L", "$K^{(x)}$\ncurrent Low"), ("gx_H", "$K^{(x)}$\ncurrent High"),
                                ("gm_L", "$K^{(m)}$\ncurrent Low"), ("gm_H", "$K^{(m)}$\ncurrent High"),
                                ("ro_L", "$(1-K^{(x)})\\phi$\ncurrent Low"),
                                ("ro_H", "$(1-K^{(x)})\\phi$\ncurrent High")], "Gain / weight")
    ax.set_title("Kalman gains and readout weight by current coherence", loc="left")
    fig.tight_layout()
    return fig


def low_high_tests(par, selected):
    rows = []
    for e in EXPS:
        d = par[par.exp == e]
        for a, b, name in (("phiL", "phiH", "φ"), ("gx_L", "gx_H", "K(x)"), ("gm_L", "gm_H", "K(m)"),
                           ("ro_L", "ro_H", "readout weight (1−K(x))φ")):
            diff = d[a] - d[b]
            nz = diff[np.abs(diff) > 1e-9]
            rows.append(dict(experiment=EXPS[e], model=selected[e], quantity=name, median_Low=d[a].median(),
                             median_High=d[b].median(), Low_gt_High=f"{int((diff > 1e-9).sum())}/{len(d)}",
                             wilcoxon_p=wilcoxon(nz).pvalue if len(nz) >= 5 else np.nan))
        a = d.aq3
        rows.append(dict(experiment=EXPS[e], model=selected[e], quantity="α_q3 (> 0: more bias noise under Low)",
                         median_Low=np.nan, median_High=np.nan, Low_gt_High=f"{int((a > 1e-9).sum())}/{len(d)}",
                         wilcoxon_p=wilcoxon(a[np.abs(a) > 1e-9]).pvalue))
    return pd.DataFrame(rows).set_index(["experiment", "model", "quantity"])


# ------------------------------------------------------------------ §5 memory path and readout path
def decompose(fits, selected):
    rows = []
    for e in EXPS:
        label = selected[e]
        for sub, r in fits.rows(e, label).iterrows():
            th = FF.theta(r)
            th0 = th.copy(); th0[G.IX["phi_L"]] = th0[G.IX["phi_H"]] = 0.0
            (_, c, kx, _, K, _), coh = fits.observer(e, sub, th, label)
            c0 = fits.observer(e, sub, th0, label)[0][1]
            tot, mem = (c[1:] * K[:-1]).sum(1), (c0[1:] * K[:-1]).sum(1)
            low = coh < 0.5
            for (pl, cl), name in TRANS.items():
                w = (low[:-1] == pl) & (low[1:] == cl)
                rows.append(dict(experiment=EXPS[e], model=label, Sub=sub, transition=name,
                                 Kx_current=kx[1:][w].mean(), memory=mem[w].mean(), readout=(tot - mem)[w].mean(),
                                 total=tot[w].mean()))
    return (pd.DataFrame(rows).groupby(["experiment", "model", "transition"], sort=False)
            [["Kx_current", "memory", "readout", "total"]].mean())
