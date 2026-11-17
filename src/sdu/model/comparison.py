"""Model comparison and screening by simulation (Appendix B.3), per experiment.

Input: the stored fits (results/model/fits/all.csv). Each row holds a participant's fitted model, its log-likelihood, the
behavioural statistics of the participant's data (obs_<statistic>, sdu.model.stats) and their mean over 20
simulations of the fitted model on the participant's own sequence (sim_<statistic>).

For each experiment and model:
  - AIC = 2k - 2 LL averaged over participants, dAIC relative to the best of the 148 models, and its rank;
  - z of each statistic: (model mean - data mean) / SE of the data mean over participants;
  - a statistic counts as reproduced when |z| < 1.5. Three groups are summarized: central tendency, serial dependence
    (slopes by current coherence, by transition, and the experiment's focal contrast) and response variability
    (residual SD at low and high coherence). passes_focal_contrasts refers to the focal contrast of the experiment
    (Exp 1: Low - High serial dependence; Exp 2: LL - mean of the other three transitions).

Writes results/model/model_comparison.csv (one row per experiment and model).
Usage (repo root):  python -m sdu.model.comparison
"""
import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import fitting as FF
from sdu.model import stats as S

Z_PASS = 1.5
CENTRAL_TENDENCY = ["b1_Low", "b1_High"]
SERIAL_BASE = ["b2_Low", "b2_High", "b2_Same", "b2_Switch", "b2_LL", "b2_LH", "b2_HL", "b2_HH"]
SERIAL_DEPENDENCE = {
    1: SERIAL_BASE + ["low_high", "low_high_cells"],
    2: SERIAL_BASE + ["LL_other_three"],
}
RESPONSE_VARIABILITY = ["rsd_Low", "rsd_High"]
REPORT = list(dict.fromkeys(CENTRAL_TENDENCY + SERIAL_BASE + ["low_high", "low_high_cells", "LL_other_three",
                                                             "LL_HL", "same_switch"]
                            + RESPONSE_VARIABILITY + ["var_HL_LL"]))


def compare(fits):
    """Fits (one row per experiment, participant and model) -> one row per experiment and model."""
    rows = []
    for exp, e in fits.groupby("exp"):
        n_sub = e.Sub.nunique()
        obs = e.drop_duplicates("Sub")
        assert (e.groupby("model").Sub.nunique() == n_sub).all(), "every model is fitted to every participant"
        e = e.assign(AIC=2 * e.k - 2 * e.LL)
        g = e.groupby("model").agg(AIC=("AIC", "mean"), k=("k", "first"))
        g["dAIC"] = g.AIC - g.AIC.min()
        g["rank_AIC"] = g.AIC.rank(method="min").astype(int)
        sims = e.groupby("model")[[f"sim_{s}" for s in REPORT]].mean()
        for s in REPORT:
            o = obs[f"obs_{s}"]
            g[f"data_{s}"] = o.mean()
            g[f"pred_{s}"] = sims[f"sim_{s}"]
            g[f"z_{s}"] = (sims[f"sim_{s}"] - o.mean()) / (o.std(ddof=1) / np.sqrt(o.notna().sum()))
        groups = {"central_tendency": CENTRAL_TENDENCY, "serial_dependence": SERIAL_DEPENDENCE[exp],
                  "response_variability": RESPONSE_VARIABILITY}
        for label, metrics in groups.items():
            z = g[[f"z_{s}" for s in metrics]].abs()
            g[f"max_abs_z_{label}"] = z.max(axis=1)
            g[f"passes_{label}"] = (z < Z_PASS).all(axis=1)
        g["passes_focal_contrasts"] = np.logical_and.reduce([g[f"z_{s}"].abs() < Z_PASS for s in S.CORE[exp]])
        g["passes_all_three"] = np.logical_and.reduce([g[f"passes_{label}"] for label in groups])
        g["experiment"] = FF.EXPERIMENT[exp]
        g["exp"] = exp
        rows.append(g.reset_index())
    out = pd.concat(rows, ignore_index=True)
    levels = out.model.str.split("-", expand=True)
    return out.assign(C=levels[0], S=levels[1], B=levels[2])


def main():
    out = compare(FF.read_fits())
    out.to_csv(SP.MODEL_COMPARISON, index=False)
    print(f"wrote {SP.MODEL_COMPARISON} ({out.model.nunique()} models x {out.exp.nunique()} experiments)")


if __name__ == "__main__":
    main()
