"""Behavioral-adequacy screening of the reported model space, judged per experiment.

Reads <SDU_MODEL_RESULTS>/screening/sim_exp*_*.csv (sdu.model.screening_sim). For each experiment and model:
  - mean AIC per participant (2k - 2 LL, best fit row), dAIC relative to the experiment's best model; only models
    fitted for every participant enter;
  - z of each statistic: (mean simulated - mean observed) / SE of the observed mean over participants;
  - three independent exploratory diagnostics at |z| < 1.5: central tendency, serial dependence (absolute slopes, four
    cells and the experiment-specific contrasts), and a broad response-variability check (Low- and
    High-coherence residual SD). These flags describe a reference band and do not select or reject models;
  - passes_all_three is retained only as a descriptive overlap. Experiments are summarized separately. AIC is reported
    separately.
Writes screening/screening.csv (one row per experiment x model) and screening/screening.log.

Usage (repo root):  SDU_MODEL_RESULTS=results/model/v3 python -m sdu.model.screening [--with-prediction]
"""
import glob

import numpy as np
import pandas as pd

from sdu import names
from sdu import paths as SP
from sdu.model import stats as S

Z_PASS = 1.5
CENTRAL_TENDENCY = ["b1_Low", "b1_High"]
SERIAL_BASE = ["b2_Low", "b2_High", "b2_Same", "b2_Switch", "b2_LL", "b2_LH", "b2_HL", "b2_HH"]
SERIAL_DEPENDENCE = {
    2: SERIAL_BASE + ["low_high", "low_high_cells"],
    1: SERIAL_BASE + ["LL_other_three"],
}
RESPONSE_VARIABILITY = ["rsd_Low", "rsd_High"]
# Retain the transition-specific dispersion contrast in the output for traceability, but do not use it to judge
# whether a model reproduces overall response variability.
VARIABILITY_AUXILIARY = ["var_HL_LL"]
REPORT = list(dict.fromkeys(CENTRAL_TENDENCY + SERIAL_BASE + ["low_high", "low_high_cells", "LL_other_three", "LL_HL", "same_switch"]
                            + RESPONSE_VARIABILITY + VARIABILITY_AUXILIARY))


def add_derived_contrasts(d):
    """Derive contrasts from stored cell slopes so historical simulation files remain readable."""
    d = d.copy()
    for prefix in ("obs_", "sim_"):
        name = prefix + "LL_other_three"
        derived = d[prefix + "b2_LL"] - d[[prefix + "b2_LH", prefix + "b2_HL", prefix + "b2_HH"]].mean(axis=1)
        d[name] = derived if name not in d else d[name].fillna(derived)     # files written before the column existed
    return d


def load(folder=None, with_prediction=False):
    """The 572 reported models; with_prediction adds the Bt/Btm simulations (sim_exp*_*_prediction.csv)."""
    folder = folder or SP.MODEL_RESULTS / "screening"
    files = sorted(glob.glob(str(folder / "sim_exp*_*.csv")))
    files = [f for f in files if with_prediction or not f.endswith("_prediction.csv")]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return add_derived_contrasts(d)


def family(label):
    c, t, s, b = label.split("-")
    return dict(C=c, T=t, S=s, B=b)


def screen(d):
    d = add_derived_contrasts(d)
    rows = []
    for exp, e in d.groupby("exp"):
        n_sub = e.Sub.nunique()
        obs = e.drop_duplicates("Sub")
        complete = e.groupby("model").Sub.nunique()
        e = e[e.model.isin(complete[complete == n_sub].index)].copy()
        e["AIC"] = 2 * e.k - 2 * e.LL
        g = e.groupby("model").agg(AIC=("AIC", "mean"), k=("k", "first"))
        g["dAIC"] = g.AIC - g.AIC.min()
        g["rank_AIC"] = g.AIC.rank(method="min").astype(int)
        sims = e.groupby("model")[[f"sim_{s}" for s in REPORT]].mean()
        for s in REPORT:
            o = obs[f"obs_{s}"]
            g[f"data_{s}"] = o.mean()
            g[f"pred_{s}"] = sims[f"sim_{s}"]
            g[f"z_{s}"] = (sims[f"sim_{s}"] - o.mean()) / (o.std(ddof=1) / np.sqrt(o.notna().sum()))
        gates = {
            "central_tendency": CENTRAL_TENDENCY,
            "serial_dependence": SERIAL_DEPENDENCE[exp],
            "response_variability": RESPONSE_VARIABILITY,
        }
        for label, metrics in gates.items():
            z = g[[f"z_{s}" for s in metrics]].abs()
            g[f"max_abs_z_{label}"] = z.max(axis=1)
            g[f"passes_{label}"] = (z < Z_PASS).all(axis=1)
        g["passes_focal_contrasts"] = np.logical_and.reduce(
            [g[f"z_{s}"].abs() < Z_PASS for s in S.CORE[exp]])
        g["passes_all_three"] = np.logical_and.reduce([g[f"passes_{label}"] for label in gates])
        g["experiment"] = names.EXPERIMENT[exp]
        g["exp"] = exp
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    fam = pd.DataFrame([family(m) for m in out.model])
    return pd.concat([out, fam], axis=1)


def report(t):
    lines = [f"=== Exploratory standardized discrepancies: {t.model.nunique()} models; |z| < {Z_PASS:g} is a "
             "reference band only, not a selection or rejection rule; AIC is separate ==="]
    gates = ["central_tendency", "serial_dependence", "response_variability"]
    show = (["dAIC", "rank_AIC"] + [f"max_abs_z_{g}" for g in gates]
            + [f"passes_{g}" for g in gates] + ["passes_all_three"])
    for exp, e in t.groupby("exp", sort=False):
        lab = names.EXPERIMENT[exp]
        shown = CENTRAL_TENDENCY + SERIAL_DEPENDENCE[exp] + RESPONSE_VARIABILITY
        lines.append(f"\n##### {lab} (statistics shown: {', '.join(shown)}); data: " + ", ".join(
            f"{s} {e[f'data_{s}'].iloc[0]:+.4f}"
            for s in ("low_high", "low_high_cells", "LL_other_three", "LL_HL", "same_switch")))
        overlap = e[e.passes_all_three]
        lines.append("gate counts: " + ", ".join(
            f"{g} {int(e[f'passes_{g}'].sum())}/{len(e)}" for g in gates)
            + f"; descriptive overlap {len(overlap)}/{len(e)} (not required)")
        lines.append("\n--- best 10 by AIC ---\n" + e.sort_values("AIC").set_index("model")[show].head(10).round(2)
                     .to_string())
        for gate in gates:
            passing = e[e[f"passes_{gate}"]]
            lines.append(f"\n--- models reproducing {gate} ({len(passing)}) ---\n" +
                         passing.sort_values("model").set_index("model")[show].round(2).to_string())
        for dim in ("T", "C", "S", "B"):
            f = e.groupby(dim).agg(models=("passes_all_three", "size"),
                                   central_tendency=("passes_central_tendency", "sum"),
                                   serial_dependence=("passes_serial_dependence", "sum"),
                                   response_variability=("passes_response_variability", "sum"),
                                   descriptive_overlap=("passes_all_three", "sum"), best_dAIC=("dAIC", "min"))
            lines.append(f"\n--- by {dim} ---\n" + f.round(2).to_string())
    return "\n".join(lines)


def main():
    import sys
    wp = "--with-prediction" in sys.argv                # add Bt/Btm; written to separate files
    t = screen(load(with_prediction=wp))
    out = SP.MODEL_RESULTS / "screening"
    stem = "screening_with_prediction" if wp else "screening"
    t.to_csv(out / f"{stem}.csv", index=False)
    txt = "\n".join(line.rstrip() for line in report(t).splitlines())
    (out / f"{stem}.log").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
