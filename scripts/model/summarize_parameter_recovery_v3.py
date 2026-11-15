"""Summarize the fresh v3 recovery in the optimized parameter units."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF

sys.path.insert(0, str(SP.ROOT / "scripts/model"))
import fit_minimal_space as M
from parameter_recovery_v3 import OUT, PARAMETERS, check_configuration, verify_design

DERIVED = {
    "delta_phi": (r"$\phi^L-\phi^H$", "Low minus High continuity", ""),
    "weight_low": (r"$w^L$", "Low trace weight", ""),
    "weight_high": (r"$w^H$", "High trace weight", ""),
    "delta_weight": (r"$w^L-w^H$", "Low minus High trace weight", ""),
    "bias_sd_low": ("Bias SD — Low (s)", "Bias innovation SD at coherence 0.3", "s"),
    "bias_sd_high": ("Bias SD — High (s)", "Bias innovation SD at coherence 0.7", "s"),
}


def derived(theta, spec, arrays):
    log_t, coherence = arrays[:2]
    _, _, gain, *_ = G.observer(theta, spec, np.exp(log_t), coherence)
    phi_l, phi_h = G.weights(theta, G.T_CODES[spec.t])
    phi = np.where(coherence < 0.5, phi_l, phi_h)
    if G.T_CODES[spec.t] in G.RESET_CODES:
        phi = np.where(G.switch_flags(coherence), 0, phi)
    weight = (1 - gain) * phi
    low, high = weight[coherence < 0.5].mean(), weight[coherence >= 0.5].mean()
    p = G.unpack(theta)
    return dict(delta_phi=phi_l - phi_h, weight_low=low, weight_high=high, delta_weight=low - high,
                bias_sd_low=np.sqrt(p["q3s"] * np.exp(p["aq3"] * 0.7)),
                bias_sd_high=np.sqrt(p["q3s"] * np.exp(p["aq3"] * 0.3)))


def load_results(out=OUT):
    metadata = json.loads((out / "design.json").read_text())
    paths = sorted((out / "raw").glob("unit_*.csv"))
    if not paths:
        return pd.DataFrame()
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if raw.unit.duplicated().any() or not raw.design_sha256.eq(metadata["design_sha256"]).all():
        raise ValueError("Duplicate units or results from a different design")
    refinements = sorted((out / "refined").glob("unit_*.csv"))
    if refinements:
        extra = pd.concat([pd.read_csv(p) for p in refinements], ignore_index=True)
        if extra.unit.duplicated().any() or not extra.design_sha256.eq(metadata["design_sha256"]).all():
            raise ValueError("Invalid refinement design")
        original = raw.set_index("unit")
        for _, row in extra.iterrows():
            old = original.loc[row.unit]
            if row.response_sha256 != old.response_sha256 or row.LL < old.LL:
                raise ValueError(f"Invalid refinement: {row.unit}")
        raw = pd.concat([raw[~raw.unit.isin(extra.unit)], extra], ignore_index=True)
    design = pd.read_csv(out / "design.csv").set_index("unit")
    for _, row in raw.iterrows():
        expected = design.loc[row.unit]
        if any(row[n] != expected[n] for n in ("design", "model", "exp", "Sub", "rep")):
            raise ValueError(f"Unit metadata mismatch: {row.unit}")
        if not np.allclose(row[[f"true_{n}" for n in G.NAMES]].to_numpy(float),
                           expected[G.NAMES].to_numpy(float), rtol=0, atol=1e-12):
            raise ValueError(f"Generating parameters differ from design: {row.unit}")
    return raw


def long_table(raw):
    space = dict(M.space())
    data = {e: FF.load(e) for e in (2, 1)}
    rows = []
    for _, row in raw.iterrows():
        spec = space[row.model]
        lower, upper = spec.bounds()
        fit = row[G.NAMES].to_numpy(float)
        true = row[[f"true_{n}" for n in G.NAMES]].to_numpy(float)
        fv = {n: fit[G.IX[n]] for n in PARAMETERS}
        tv = {n: true[G.IX[n]] for n in PARAMETERS}
        fv.update(derived(fit, spec, data[row.exp][row.Sub]))
        tv.update(derived(true, spec, data[row.exp][row.Sub]))
        for name in fv:
            lo, hi = (lower[G.IX[name]], upper[G.IX[name]]) if name in PARAMETERS else (np.nan, np.nan)
            tol = (hi - lo) * 1e-6
            rows.append(dict(unit=row.unit, design=row.design, exp=row.exp, Sub=row.Sub, rep=row.rep,
                             parameter=name, true=tv[name], fitted=fv[name], error=fv[name] - tv[name],
                             lower=lo, upper=hi, at_lower=bool(fv[name] <= lo + tol) if name in PARAMETERS else False,
                             at_upper=bool(fv[name] >= hi - tol) if name in PARAMETERS else False))
    return pd.DataFrame(rows)


def metrics(group):
    true, fitted = group.true.to_numpy(float), group.fitted.to_numpy(float)
    varied = len(true) > 2 and np.ptp(true) > 1e-12
    r = np.corrcoef(true, fitted)[0, 1] if varied and np.ptp(fitted) > 1e-12 else np.nan
    slope = np.polyfit(true, fitted, 1)[0] if varied else np.nan
    return dict(n=len(group), r=r, slope=slope, true_mean=true.mean(), fitted_mean=fitted.mean(),
                bias=np.mean(fitted - true), rmse=np.sqrt(np.mean((fitted - true) ** 2)),
                lower_rate=group.at_lower.mean(), upper_rate=group.at_upper.mean())


def summarize(table):
    # Each local repetition supplies one independent noisy fit per generating participant.
    cohort_rows = []
    for (design, exp, rep, parameter), group in table.groupby(["design", "exp", "rep", "parameter"], sort=False):
        cohort_rows.append(dict(design=design, exp=exp, rep=rep, parameter=parameter, **metrics(group)))
    cohorts = pd.DataFrame(cohort_rows)
    rows = []
    for (design, exp, parameter), group in table.groupby(["design", "exp", "parameter"], sort=False):
        c = cohorts[(cohorts.design == design) & (cohorts.exp == exp) & (cohorts.parameter == parameter)]
        # Pool errors for bias/RMSE; summarize r and slope within cohort, without averaging fits.
        values = metrics(group)
        values.update(r=c.r.median(), r_q25=c.r.quantile(0.25), r_q75=c.r.quantile(0.75),
                      slope=c.slope.median(), cohorts=len(c))
        if design == "fitted":
            participant = group.groupby("Sub").agg(
                true=("true", "mean"), fitted=("fitted", "mean"),
                at_lower=("at_lower", "mean"), at_upper=("at_upper", "mean"))
            mean_metrics = metrics(participant)
            participant_rmse = group.groupby("Sub").error.apply(lambda e: np.sqrt(np.mean(e ** 2)))
            values.update(r_mean=mean_metrics["r"], slope_mean=mean_metrics["slope"],
                          mean_participant_rmse=participant_rmse.mean(),
                          bias=group.groupby("Sub").error.mean().mean(),
                          lower_rate=participant.at_lower.mean(), upper_rate=participant.at_upper.mean())
        else:
            values.update(r_mean=np.nan, slope_mean=np.nan, mean_participant_rmse=np.nan)
        rows.append(dict(design=design, exp=exp, parameter=parameter, **values))
    return pd.DataFrame(rows), cohorts


def participant_summary(table):
    rows = []
    local = table[table.design == "fitted"]
    for (exp, sub, parameter), g in local.groupby(["exp", "Sub", "parameter"]):
        rows.append(dict(exp=exp, Sub=sub, parameter=parameter, repetitions=len(g),
                         true=g.true.mean(), fitted_mean=g.fitted.mean(), fitted_sd=g.fitted.std(),
                         fitted_q05=g.fitted.quantile(0.05), fitted_q95=g.fitted.quantile(0.95),
                         bias=g.error.mean(), rmse=np.sqrt(np.mean(g.error ** 2)),
                         lower_rate=g.at_lower.mean(), upper_rate=g.at_upper.mean()))
    return pd.DataFrame(rows)


def contrasts(table):
    rows = []
    delta = table[table.parameter.isin(("delta_phi", "delta_weight"))]
    for (design, exp, rep, parameter), group in delta.groupby(["design", "exp", "rep", "parameter"]):
        tm, fm = group.true.mean(), group.fitted.mean()
        eligible = np.abs(group.true) > 1e-10
        rows.append(dict(design=design, exp=exp, rep=rep, parameter=parameter, n=len(group),
                         true_mean=tm, fitted_mean=fm, bias=fm - tm,
                         group_direction_correct=bool(np.sign(tm) == np.sign(fm)) if abs(tm) > 1e-10 else np.nan,
                         participant_direction_accuracy=np.mean(
                             np.sign(group.true[eligible]) == np.sign(group.fitted[eligible])) if eligible.any() else np.nan))
    return pd.DataFrame(rows)


def error_correlations(table):
    rows = []
    # Center within each generating participant to isolate estimation trade-offs from between-person differences.
    local = table[(table.design == "fitted") & table.parameter.isin(PARAMETERS)]
    for exp, group in local.groupby("exp"):
        wide = group.pivot(index=["Sub", "rep"], columns="parameter", values="error")
        centered = wide - wide.groupby(level="Sub").transform("mean")
        corr = centered[list(PARAMETERS)].corr()
        for a in PARAMETERS:
            for b in PARAMETERS:
                rows.append(dict(exp=exp, parameter_a=a, parameter_b=b, r=corr.loc[a, b]))
    return pd.DataFrame(rows)


def write_report(out, summary, condition, raw, progress):
    def md_table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |",
                          "|" + "---|" * len(headers)] + ["| " + " | ".join(r) + " |" for r in rows])
    def fmt(x):
        return "—" if not np.isfinite(x) else f"{0.0 if abs(x) < 0.0005 else x:.3f}"
    lines = ["# 3SKF v3 parameter recovery", "",
             f"Completed {len(raw)} of {int(progress.planned.sum())} planned fits.", "",
             "The primary design contains 22 generating participants per experiment, each with 20 new response data sets. "
             "Experiment 1 uses C3-S0-B2; Experiment 2 uses C3-S3-B2.", "",
             "## Ten optimized parameters", "",
             "Mean r compares the 22 true values with participant mean estimates over 20 fits. Cohort r is the median "
             "of 20 correlations across 22 participants. RMSE is calculated within participant, then averaged across "
             "participants. All values are on the optimizer's natural parameter scale.", ""]
    for exp in (2, 1):
        d = summary[(summary.design == "fitted") & (summary.exp == exp)].set_index("parameter")
        rows = []
        for n, meta in PARAMETERS.items():
            r = d.loc[n]
            rows.append([meta[1], fmt(r.r_mean), fmt(r.r), fmt(r.slope_mean), fmt(r.bias),
                         fmt(r.mean_participant_rmse), f"{r.lower_rate:.1%}", f"{r.upper_rate:.1%}"])
        lines += [f"### Experiment {1 if exp == 2 else 2}", "",
                  md_table(["Parameter", "Mean r", "Cohort r", "Mean slope", "Bias", "Participant RMSE",
                            "Lower bound", "Upper bound"], rows), ""]
    lines += ["## Coherence differences", "",
              "Trace weights are computed from the actual per-trial Kalman gain and the current condition's phi, "
              "then averaged within High and Low. S3 sets the weight to zero on switch trials. Direction percentages "
              "refer to cohort means across 20 repeated simulations.", ""]
    rows = []
    for exp in (2, 1):
        for name in ("delta_phi", "delta_weight"):
            d = condition[(condition.design == "fitted") & (condition.exp == exp) & (condition.parameter == name)]
            rows.append([str(1 if exp == 2 else 2), DERIVED[name][1], fmt(d.true_mean.mean()),
                         fmt(d.fitted_mean.mean()), f"{d.group_direction_correct.mean():.0%}"])
    lines += [md_table(["Experiment", "Difference", "True mean", "Recovered mean", "Direction accuracy"], rows), "",
              "## Exploratory parameter sampling", "",
              "A further 200 independent parameter vectors per experiment cover the observed fitted ranges, with "
              "continuity independently sampled from 0 to 1. Results are saved separately under design=range. "
              "Exact sampling bounds are in sampling_ranges.csv.", "",
              "## Optimizer checks", "",
              f"Best-fit convergence: {raw.success.mean():.1%}. "
              f"Fitted likelihood below the generating likelihood by more than 0.01: "
              f"{int((raw.LL_minus_true < -0.01).sum())} of {len(raw)} fits.", "",
              f"Additional-start fits retained: {len(list((out / 'refined').glob('unit_*.csv')))}. "
              "Original fits remain in raw/ and refinements in refined/.", "",
              "## Estimation error correlations", "",
              "Errors are centered within each generating participant across repetitions before computing the "
              "10 by 10 correlation matrix. This measures error covariance for repeated data at fixed generating values.", "",
              "## Files", "",
              "- PROTOCOL.md: design, parameter definitions and references.",
              "- participant_summary.csv: each participant's mean, spread, bias, RMSE and boundary rates.",
              "- parameter_summary.csv: parameter recovery metrics for the observed and sampled designs.",
              "- condition_contrasts.csv: Low minus High differences in each simulated cohort.",
              "- optimizer_diagnostics.csv: convergence and likelihood checks.",
              "- design.json: seeds, source hashes and fitting configuration.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def write_summary(out=OUT):
    check_configuration()
    verify_design(out)
    raw = load_results(out)
    design = pd.read_csv(out / "design.csv")
    progress = design.groupby(["design", "exp"]).size().rename("planned").to_frame()
    progress["complete"] = 0 if raw.empty else raw.groupby(["design", "exp"]).size()
    progress["complete"] = progress.complete.fillna(0).astype(int)
    progress.to_csv(out / "progress.csv")
    if raw.empty:
        print(progress.to_string())
        return
    table = long_table(raw)
    summary, cohorts = summarize(table)
    condition = contrasts(table)
    for name, frame in {"recovery_long": table, "parameter_summary": summary, "cohort_metrics": cohorts,
                        "participant_summary": participant_summary(table),
                        "condition_contrasts": condition, "error_correlations": error_correlations(table)}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    diagnostics = raw[["unit", "design", "exp", "Sub", "rep", "success", "message", "n_starts",
                       "n_converged", "n_at_best", "LL_minus_true", "elapsed_seconds"]]
    diagnostics.to_csv(out / "optimizer_diagnostics.csv", index=False)
    if (progress.complete == progress.planned).all():
        write_report(out, summary, condition, raw, progress)
    print(progress.to_string())
    print(f"Best-fit convergence: {raw.success.mean():.1%}; fitted LL below generating LL: "
          f"{int((raw.LL_minus_true < -0.01).sum())}/{len(raw)}")
    print(summary[summary.parameter.isin(PARAMETERS)].round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    write_summary(parser.parse_args().out)
