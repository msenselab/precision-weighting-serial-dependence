"""Duration-resolved data/model curves for the current and comparison v3 models.

The output restores the informative line-plot check used in the manuscript: central tendency, serial dependence and
response variability, each resolved by duration and current coherence.  Observed points are participant means; model
curves and ranges come from fixed-parameter simulations on every participant's recorded sequence and validity mask.

Usage (repo root):
    SDU_MODEL_RESULTS=results/model/v3 GKF_BOUNDS=v3 PYTHONPATH=. python -m sdu.model.reproduction
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sdu import names
from sdu import paths as SP
from sdu.behavior import trials as BT
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model.selection import PRIMARY_MODELS, REPRODUCTION_MODELS


if G.BOUNDS != "v3":
    raise RuntimeError("duration-resolved reproduction curves require GKF_BOUNDS=v3")

N_COHORTS = 100
PANELS = ("central tendency", "serial dependence", "response variability")


def _signatures(v: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Participant curves in the same duration-resolved form as the observed data."""
    v = v.dropna(subset=["curBias"]).copy()
    v["coh"] = np.where(np.isclose(v.curCoherence, .3), "Low", "High")
    v["residual_bias"] = v.curBias - v.groupby(["Sub", "coh", "curDur"]).curBias.transform("mean")
    central = v.groupby(["Sub", "coh", "curDur"]).curBias.mean()
    serial = (v.dropna(subset=["preDur1back"])
              .groupby(["Sub", "coh", "preDur1back"]).residual_bias.mean())
    variability = v.groupby(["Sub", "coh", "curDur"]).residual_bias.std(ddof=1)
    return central, serial, variability


def _group_curve(series: pd.Series, coherence: str) -> pd.Series:
    by_participant = series.xs(coherence, level="coh").unstack("Sub")
    return by_participant.mean(axis=1)


def generate(n_cohorts: int = N_COHORTS) -> pd.DataFrame:
    rows: list[dict] = []
    for exp, model_labels in REPRODUCTION_MODELS.items():
        trial_table = BT.trials(exp)
        observed = _signatures(BT.valid_frame(trial_table, trial_table.rpr.to_numpy(float)))
        all_fits = FF.read_fits(sorted((SP.RESULTS_MODEL_V3 / "fits").glob(f"exp{exp}_*.csv")))
        data = FF.load(exp)
        participants = sorted(data)

        subjects = {sub: trial_table[trial_table.Sub == sub].copy() for sub in participants}
        for sub in participants:
            if not np.array_equal(~subjects[sub].is_outlier.to_numpy(), data[sub][4]):
                raise ValueError(f"validity-mask mismatch for experiment code {exp}, participant {sub}")

        for model_index, model_label in enumerate(model_labels):
            fits = all_fits[all_fits.public == names.public(model_label)].set_index("Sub")
            if set(fits.index) != set(participants):
                raise ValueError(f"incomplete model fits for {model_label}, experiment code {exp}")

            rng = np.random.default_rng(20261105 + 100 * exp + model_index)
            simulated: list[tuple[pd.Series, pd.Series, pd.Series]] = []
            for _ in range(n_cohorts):
                frames = []
                for sub in participants:
                    log_t, coherence, tau, _, valid = data[sub]
                    fit = fits.loc[sub]
                    response = G.simulate(
                        fit[G.NAMES].to_numpy(float), G.Spec.parse(fit.model), np.exp(log_t), coherence, tau, valid, rng
                    )[0]
                    frames.append(BT.valid_frame(subjects[sub], response))
                simulated.append(_signatures(pd.concat(frames, ignore_index=True)))

            for panel_index, panel in enumerate(PANELS):
                for coherence in ("Low", "High"):
                    obs_by_participant = observed[panel_index].xs(coherence, level="coh").unstack("Sub")
                    obs_mean = obs_by_participant.mean(axis=1)
                    obs_sem = obs_by_participant.std(axis=1, ddof=1) / np.sqrt(obs_by_participant.notna().sum(axis=1))
                    cohort_curves = pd.concat(
                        [_group_curve(cohort[panel_index], coherence) for cohort in simulated], axis=1
                    )
                    for duration in obs_mean.index:
                        values = cohort_curves.loc[duration]
                        rows.append({
                            "experiment": names.EXPERIMENT[exp], "exp": exp, "model": model_label,
                            "is_primary": model_label == PRIMARY_MODELS[exp],
                            "panel": panel, "coherence": coherence, "duration": float(duration),
                            "observed": obs_mean.loc[duration], "observed_sem": obs_sem.loc[duration],
                            "simulated": values.mean(), "lo": values.quantile(.025), "hi": values.quantile(.975),
                            "n_cohorts": n_cohorts,
                        })
    out = pd.DataFrame(rows)
    path = SP.RESULTS_MODEL_V3 / "screening" / "selected_model_duration_curves.csv"
    out.to_csv(path, index=False)
    print(f"saved {path}")
    return out


if __name__ == "__main__":
    generate()
