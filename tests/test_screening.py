"""Screening is judged per experiment with that experiment's core effects."""
import numpy as np
import pandas as pd

from sdu.model import screening as SC


def test_exploratory_reference_band_is_one_point_five():
    assert SC.Z_PASS == 1.5


def _frame():
    rows = []
    rng = np.random.default_rng(0)
    for exp in (2, 1):
        for sub in range(6):
            obs = {s: rng.normal(0.05, 0.02) for s in SC.REPORT}
            for model, k, ll, shift in (("C1-T3-S0-B4", 10, -10.0, 0.0), ("C1-T0-S0-B4", 9, -9.0, -0.2),
                                        ("C1-T3-S2-B4", 10, -10.5, 0.0)):
                sim = {s: v + (shift if (exp == 2 and s == "low_high") else 0.0) for s, v in obs.items()}
                if model == "C1-T3-S0-B4" and exp == 1:
                    sim["LL_other_three"] = obs["LL_other_three"] - 0.2  # fails the Experiment 2 core effect only
                rows.append(dict(exp=exp, Sub=sub, model=model, k=k, LL=ll,
                                 **{f"obs_{s}": v for s, v in obs.items()}, **{f"sim_{s}": v for s, v in sim.items()}))
    return pd.DataFrame(rows)


def test_pass_is_judged_per_experiment():
    t = SC.screen(_frame()).set_index(["experiment", "model"])
    assert t.loc[("Exp 1", "C1-T3-S0-B4"), "passes_all_three"]
    assert not t.loc[("Exp 1", "C1-T0-S0-B4"), "passes_serial_dependence"]  # misses low_high in Experiment 1
    assert not t.loc[("Exp 2", "C1-T3-S0-B4"), "passes_serial_dependence"]  # misses LL-other-three in Experiment 2
    assert t.loc[("Exp 2", "C1-T3-S2-B4"), "passes_all_three"]
    assert t.loc[("Exp 1", "C1-T3-S0-B4"), "passes_central_tendency"]
    assert t.loc[("Exp 1", "C1-T3-S0-B4"), "passes_response_variability"]


def test_aic_mean_per_participant_and_families():
    t = SC.screen(_frame()).set_index(["experiment", "model"])
    assert t.loc[("Exp 1", "C1-T0-S0-B4"), "dAIC"] == 0.0           # 2*9 + 18 = 36 is the smallest
    assert t.loc[("Exp 1", "C1-T3-S0-B4"), "dAIC"] == 4.0
    assert t.loc[("Exp 1", "C1-T3-S2-B4"), "T"] == "T3" and t.loc[("Exp 1", "C1-T3-S2-B4"), "S"] == "S2"


def test_incomplete_models_are_excluded():
    d = _frame()
    d = d[~((d.model == "C1-T3-S2-B4") & (d.Sub == 0) & (d.exp == 2))]
    t = SC.screen(d)
    assert "C1-T3-S2-B4" not in set(t[t.experiment == "Exp 1"].model)
    assert "C1-T3-S2-B4" in set(t[t.experiment == "Exp 2"].model)


def test_derived_contrast_filled_when_files_are_mixed():
    """Simulation files written before LL_other_three existed are combined with newer ones without losing rows."""
    d = _frame()
    old = d[d.model == "C1-T0-S0-B4"].drop(columns=["obs_LL_other_three", "sim_LL_other_three"])
    mixed = pd.concat([d[d.model != "C1-T0-S0-B4"], old], ignore_index=True)
    out = SC.add_derived_contrasts(mixed)
    assert out["sim_LL_other_three"].notna().all() and out["obs_LL_other_three"].notna().all()
