"""Helpers for notebooks/glasauer_shi.ipynb: 3SKF C3-S0-B2 and the Glasauer-Shi two-state model on Glasauer & Shi (2021).

Requires GKF_READOUT=1 and the v3 engine configuration (set before sdu.model.engine is imported).
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import glasauer_shi as GS
from sdu.reports import plots as P

sys.path.insert(0, str(SP.ROOT / "scripts" / "model" / "external"))
import fit_external as X  # noqa: E402

RESULTS = SP.RESULTS / "model" / "minimal_qdiag" / "glasauer_shi"
V3 = RESULTS / "v3"
TRANSFER = V3 / "transfer"
CONDITIONS = ("randomized", "random walk")
THREE = "3SKF C3-S0-B2"
ABLATION = "3SKF B0 (no bias state)"
TWO = "Glasauer–Shi two-state"
MODELS = ("data", TWO, THREE)
COLORS = {"data": P.INK, TWO: P.ORANGE, THREE: P.BLUE}
SEQ_COL = {"randomized": P.BLUE, "random walk": "#98a3c1"}
NAME = {"3SKF v3 R-B2": THREE, "3SKF v3 R-B0 (no bias state)": ABLATION, "Glasauer-Shi two-state": TWO}


class Data:
    """Trials, the paper model's randomized fits and the 3SKF separate fits."""

    def __init__(self):
        _, loaded = X.load("gs")
        self.units = dict(loaded)
        self.participants = sorted({int(key.split("|")[0]) for key in self.units})
        self.two_state = pd.read_csv(RESULTS / "two_state_fits.csv").set_index("sub")
        fits = pd.read_csv(V3 / "fits_separate.csv")
        fits = fits[fits.model.isin(X.MODELS)].copy()
        fits["sub"] = fits.unit.str.split("|").str[0].astype(int)
        fits["condition"] = fits.unit.str.split("|").str[1]
        self.three_state = fits
        self.randomized_3skf = fits[(fits.model == "R-B2") & (fits.condition == "randomized")].set_index("sub")
        self.spec = X.MODELS["R-B2"]
        assert self.participants == list(range(1, 15))
        assert set(self.two_state.index) == set(self.participants) == set(self.randomized_3skf.index)

    def predictions(self, participant, condition):
        """Observed responses and the two models' deterministic predictions (parameters frozen from randomized)."""
        duration, observed, valid = self.units[f"{participant}|{condition}"][0]
        fit = self.two_state.loc[participant]
        paper = GS.observer(duration, fit.v_over_r, fit.q_over_r, fit["shift"])
        theta = self.randomized_3skf.loc[participant, G.NAMES].to_numpy(float)
        three = G.response_mean(theta, self.spec, np.log(duration), np.ones(len(duration)),
                                np.linspace(-1, 1, len(duration)))
        return duration, valid, {"data": observed, TWO: paper, THREE: three}


# ------------------------------------------------------------------ CT and SD
def paper_statistics(duration, response, valid):
    raw = GS.statistics(duration, response, valid)
    return {"CT": -raw["ct"], "SD": raw["sd"], "lag-2 SD": raw["sd2"],
            "partial r(error, previous | current)": raw["partial_previous_r"],
            "partial r(error, current | previous)": raw["partial_current_r"],
            "r(current, previous)": raw["current_previous_r"]}


def participant_statistics(d):
    rows = []
    for participant in d.participants:
        for condition in CONDITIONS:
            duration, valid, responses = d.predictions(participant, condition)
            for model, response in responses.items():
                rows.append({"sub": participant, "condition": condition, "model": model,
                             **paper_statistics(duration, response, valid)})
    return pd.DataFrame(rows)


def ct_sd_summary(ps):
    return ps.groupby(["condition", "model"])[["CT", "SD"]].agg(["mean", "sem"]).round(3)


def partial_correlations(ps):
    rows = []
    data = ps[(ps.condition == "randomized") & (ps.model == "data")]
    for measure in ("partial r(error, previous | current)", "partial r(error, current | previous)"):
        values = data[measure]
        test = ttest_1samp(values, 0)
        rows.append({"measure": measure, "mean": values.mean(), "std": values.std(), "t(13)": test.statistic,
                     "p": "<.0001" if test.pvalue < .0001 else f"{test.pvalue:.4f}"})
    return pd.DataFrame(rows).set_index("measure").round(4)


def predictor_correlation(ps):
    return ps[ps.model == "data"].groupby("condition")["r(current, previous)"].agg(["mean", "min", "max"]).round(3)


def random_walk_prediction_tests(ps):
    rows = []
    observed = ps[(ps.condition == "random walk") & (ps.model == "data")].set_index("sub")
    for model in MODELS[1:]:
        predicted = ps[(ps.condition == "random walk") & (ps.model == model)].set_index("sub")
        for measure in ("CT", "SD"):
            difference = predicted[measure] - observed[measure]
            test = ttest_rel(predicted[measure], observed[measure])
            rows.append({"model": model, "measure": measure, "model − data": difference.mean(),
                         "SEM of difference": difference.sem(), "t(13)": test.statistic, "p": test.pvalue})
    return pd.DataFrame(rows).set_index(["model", "measure"]).round(3)


def condition_tests(ps):
    rows = []
    data = ps[ps.model == "data"]
    for measure in ("CT", "SD"):
        wide = data.pivot(index="sub", columns="condition", values=measure)
        test = ttest_rel(wide["randomized"], wide["random walk"])
        rows.append({"measure": measure, "randomized": wide["randomized"].mean(),
                     "random walk": wide["random walk"].mean(), "t(13)": test.statistic, "p": test.pvalue,
                     "randomized > random walk": f"{int((wide['randomized'] > wide['random walk']).sum())}/14"})
    return pd.DataFrame(rows).set_index("measure").round(4)


def plot_ct_sd(ps):
    """CT and SD by sequence: observed points, both models overlaid (dashed line ± SEM band)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(CONDITIONS))
    for ax, (letter, measure) in zip(axes, (("A", "CT"), ("B", "SD"))):
        for model in (THREE, TWO):
            frame = ps[ps.model == model].groupby("condition")[measure]
            mean, sem = frame.mean().loc[list(CONDITIONS)], frame.sem().loc[list(CONDITIONS)]
            P.model_line(ax, x, mean, mean - sem, mean + sem, COLORS[model], f"{model} (Model)")
        frame = ps[ps.model == "data"].groupby("condition")[measure]
        P.observed_points(ax, x, frame.mean().loc[list(CONDITIONS)], frame.sem().loc[list(CONDITIONS)], ms=8)
        P.zero_line(ax)
        ax.set_xticks(x, CONDITIONS)
        ax.set_xlim(-.4, 1.4)
        ax.set_ylabel(measure + " (participant mean ± SEM)")
        ax.set_title(f"{letter}: {measure}, randomized fit → random-walk prediction", loc="left", fontsize=12)
        ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=.9)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ duration-resolved curves
def _duration_curve(duration, response, valid, lag):
    error = pd.Series(np.where(valid, response - duration, np.nan))
    grouping = duration if lag == 0 else duration[:-lag]
    if lag:
        error = error.iloc[lag:]
    return error.groupby(np.round(grouping, 1)).mean()


def duration_curves(d):
    rows = []
    for participant in d.participants:
        for condition in CONDITIONS:
            duration, valid, responses = d.predictions(participant, condition)
            for model, response in responses.items():
                for lag, panel in ((0, "current duration"), (1, "previous duration"), (2, "lag-2 duration")):
                    for bin_duration, value in _duration_curve(duration, response, valid, lag).items():
                        rows.append({"sub": participant, "condition": condition, "model": model, "panel": panel,
                                     "duration": bin_duration, "error": value})
    return (pd.DataFrame(rows).groupby(["condition", "model", "panel", "duration"]).error
            .agg(["mean", "sem", "count"]).reset_index())


def plot_duration_curves(curves):
    """Two columns: observed vs 3SKF (left), observed vs two-state (right); one row per sequence and duration."""
    rows = [(condition, panel) for panel in ("current duration", "previous duration") for condition in CONDITIONS]
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 17), sharey="row")
    for r_i, (condition, panel) in enumerate(rows):
        sel = (curves.condition == condition) & (curves.panel == panel)
        data = curves[sel & (curves.model == "data")].sort_values("duration")
        for c_i, model in enumerate((THREE, TWO)):
            ax = axes[r_i, c_i]
            frame = curves[sel & (curves.model == model)].sort_values("duration")
            x, y, sem = frame.duration.to_numpy(float), frame["mean"].to_numpy(float), frame["sem"].to_numpy(float)
            P.model_line(ax, x, y, y - sem, y + sem, COLORS[model], f"{model} (Model)")
            P.observed_points(ax, data.duration, data["mean"], data["sem"])
            P.zero_line(ax)
            ax.set_title(f"{'ABCDEFGH'[2 * r_i + c_i]}: {condition}, {panel} — Observed vs. {model}", loc="left",
                         fontsize=11.5)
            ax.set_xlabel(panel.capitalize() + " (s)")
            ax.set_ylabel("Reproduction error (s)")
            ax.tick_params(labelleft=True)
            ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=.9)
    fig.tight_layout()
    return fig


def r_squared_table(curves):
    def r2(data, model):
        joined = data[["duration", "mean"]].merge(model[["duration", "mean"]], on="duration",
                                                  suffixes=("_data", "_model"))
        return 1 - np.square(joined.mean_data - joined.mean_model).sum() / \
            np.square(joined.mean_data - joined.mean_data.mean()).sum()
    rows = []
    for condition in CONDITIONS:
        for model in MODELS[1:]:
            for panel in ("current duration", "previous duration", "lag-2 duration"):
                sel = (curves.condition == condition) & (curves.panel == panel)
                rows.append({"condition": condition, "model": model, "curve": panel,
                             "R²": r2(curves[sel & (curves.model == "data")], curves[sel & (curves.model == model)])})
    return pd.DataFrame(rows).pivot_table(index=["model", "curve"], columns="condition", values="R²").round(3)


# ------------------------------------------------------------------ held-out transfer and AIC
METRICS = {"rmse": "trial-level RMSE", "mean_log_score": "mean log score", "ct_absolute_error": "absolute CT error",
           "asd_absolute_error": "absolute SD error"}


def transfer_tables():
    scores = pd.read_csv(TRANSFER / "participant_scores.csv")
    summary = pd.read_csv(TRANSFER / "pairwise_summary.csv")
    assert scores.shape[0] == 14 * 3 and summary.isna().sum().sum() == 0
    summary["comparator"] = summary.comparator.map(NAME)
    scores["model"] = scores.model.map(NAME)
    diff, fav = f"{THREE} − comparator", f"participants favoring {THREE}"
    comparison = summary[summary.metric.isin(METRICS)].copy()
    comparison["metric"] = comparison.metric.map(METRICS)
    comparison = comparison.rename(columns={"mean_focal_minus_comparator": diff, "bootstrap_95_lo": "95% CI low",
                                            "bootstrap_95_hi": "95% CI high", "exact_sign_flip_p": "exact p",
                                            "participants_favouring_focal": fav})
    table = comparison.set_index(["comparator", "metric"])[["preferred", diff, "95% CI low", "95% CI high",
                                                            "exact p", fav]].round(4)
    means = scores.groupby("model")[list(METRICS)].mean().round(4).rename(columns=METRICS)
    return table, means


def aic_comparison():
    paper = pd.read_csv(TRANSFER / "paper_two_state_fits.csv").set_index("sub")
    fits = pd.read_csv(TRANSFER / "fits.csv")
    fits["sub"] = fits.unit.str.split("|").str[0].astype(int)
    aic = {TWO: paper.aic}
    for model, name in (("R-B0", ABLATION), ("R-B2", THREE)):
        frame = fits[fits.model == model].set_index("sub")
        assert (frame.n == paper.n.reindex(frame.index)).all()
        aic[name] = 2 * frame.k - 2 * frame.LL
    aic = pd.DataFrame(aic)
    difference = aic[THREE] - aic[TWO]
    test = pd.DataFrame({f"{THREE} − two-state (mean)": [difference.mean()],
                         f"participants favoring {THREE}": [f"{int((difference < 0).sum())}/14"],
                         "Wilcoxon p": [wilcoxon(difference).pvalue]})
    return aic.mean().rename("mean AIC per participant").to_frame(), test


# ------------------------------------------------------------------ parameters by sequence type
def sequence_parameters(d):
    rows = []
    for _, fit in d.three_state[d.three_state.model == "R-B2"].iterrows():
        theta = fit[G.NAMES].to_numpy(float)
        p = G.unpack(theta)
        duration, _, valid = d.units[f"{int(fit['sub'])}|{fit.condition}"][0]
        _, _, kx, _, K, _ = G.observer(theta, d.spec, duration, np.ones(len(duration)))
        rows.append({"sub": int(fit["sub"]), "condition": fit.condition, "q1": p["q1"], "q2": p["q2"],
                     "sb": P.b2_bias_sd(p, 1.0), "lam": p["lam"], "phi": fit.wuH, "sde": p["sde"], "offset": p["d0"],
                     "gx": kx[valid].mean(), "gm": K[valid, 0].mean()})
    out = pd.DataFrame(rows)
    assert out.groupby("condition")["sub"].nunique().eq(14).all()
    return out


def plot_sequence_parameters(sp):
    groups = [(c, sp[sp.condition == c], SEQ_COL[c]) for c in CONDITIONS]
    fig1, ax = plt.subplots(figsize=(9, 4.6))
    P.grouped_bars(ax, groups, [("q1", "$q_1$\n(fast state)"), ("q2", "$q_2$\n(slow state)")],
                   "Variance ($R$ units, log)", log=True)
    ax.set_title(f"{THREE}, separate fits: noise variances by sequence type", loc="left")
    fig1.tight_layout()
    fig2, ax = plt.subplots(figsize=(13, 4.8))
    P.grouped_bars(ax, groups, [("sb", "$\\sigma_b$ (s)\n(response-bias\ndrift SD)"), ("phi", "$\\phi$\n(readout)"), ("gx", "$K^{(x)}$\n(measurement gain)"),
                                ("gm", "$K^{(m)}$\n(slow-state gain)"), ("lam", "$\\lambda$\n(bias decay)"),
                                ("sde", "response\nnoise SD (s)"), ("offset", "$d_0$\n(response offset, s)")], "Value")
    ax.set_title(f"{THREE}, separate fits: persistence, gains and response parameters by sequence type", loc="left")
    fig2.tight_layout()
    return fig1, fig2


def sequence_tests(sp):
    rows = []
    for column, name in (("q1", "q1"), ("q2", "q2"), ("sb", "σ_b (s), B2 bias drift SD"), ("lam", "λ"), ("phi", "φ"), ("gx", "K(x)"),
                         ("gm", "K(m)"), ("sde", "response noise SD"), ("offset", "d0")):
        wide = sp.pivot(index="sub", columns="condition", values=column)
        difference = wide["randomized"] - wide["random walk"]
        nonzero = difference[np.abs(difference) > 1e-9]
        rows.append({"quantity": name, "randomized (median)": wide["randomized"].median(),
                     "random walk (median)": wide["random walk"].median(),
                     "randomized > random walk": f"{int((difference > 1e-9).sum())}/14",
                     "Wilcoxon p": wilcoxon(nonzero).pvalue if len(nonzero) >= 5 else np.nan})
    return pd.DataFrame(rows).set_index("quantity")
