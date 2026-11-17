"""Model comparison and screening (sdu.model.comparison) are judged per experiment with its focal contrast."""
import numpy as np
import pandas as pd
import pytest

from sdu.model import comparison as SC


def test_exploratory_reference_band_is_one_point_five():
    assert SC.Z_PASS == 1.5


def _frame():
    rows = []
    rng = np.random.default_rng(0)
    for exp in (1, 2):
        for sub in range(6):
            obs = {s: rng.normal(0.05, 0.02) for s in SC.REPORT}
            for model, k, ll, shift in (("C1-S0-B2", 10, -10.0, 0.0), ("C1-S0-B0", 9, -9.0, -0.2),
                                        ("C1-S2-B2", 10, -10.5, 0.0)):
                sim = {s: v + (shift if (exp == 1 and s == "low_high") else 0.0) for s, v in obs.items()}
                if model == "C1-S0-B2" and exp == 2:
                    sim["LL_other_three"] = obs["LL_other_three"] - 0.2  # fails the Experiment 2 core effect only
                rows.append(dict(exp=exp, Sub=sub, model=model, k=k, LL=ll,
                                 **{f"obs_{s}": v for s, v in obs.items()}, **{f"sim_{s}": v for s, v in sim.items()}))
    return pd.DataFrame(rows)


def test_pass_is_judged_per_experiment():
    t = SC.compare(_frame()).set_index(["experiment", "model"])
    assert t.loc[("Exp 1", "C1-S0-B2"), "passes_all_three"]
    assert not t.loc[("Exp 1", "C1-S0-B0"), "passes_serial_dependence"]     # misses low_high in Experiment 1
    assert not t.loc[("Exp 2", "C1-S0-B2"), "passes_serial_dependence"]     # misses LL-other-three in Experiment 2
    assert t.loc[("Exp 2", "C1-S2-B2"), "passes_all_three"]
    assert t.loc[("Exp 1", "C1-S0-B2"), "passes_central_tendency"]
    assert t.loc[("Exp 1", "C1-S0-B2"), "passes_response_variability"]


def test_aic_mean_per_participant_and_levels():
    t = SC.compare(_frame()).set_index(["experiment", "model"])
    assert t.loc[("Exp 1", "C1-S0-B0"), "dAIC"] == 0.0                # 2*9 + 18 = 36 is the smallest
    assert t.loc[("Exp 1", "C1-S0-B2"), "dAIC"] == 4.0
    assert t.loc[("Exp 1", "C1-S2-B2"), "S"] == "S2" and t.loc[("Exp 1", "C1-S2-B2"), "B"] == "B2"


def test_every_model_must_be_fitted_to_every_participant():
    d = _frame()
    d = d[~((d.model == "C1-S2-B2") & (d.Sub == 0) & (d.exp == 2))]
    with pytest.raises(AssertionError):
        SC.compare(d)


def test_stored_comparison_is_reproduced():
    from sdu import paths as SP
    from sdu.model import fitting as FF
    stored = pd.read_csv(SP.MODEL_COMPARISON)
    new = SC.compare(FF.read_fits())
    assert len(stored) == len(new) == 296
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), new.reset_index(drop=True), check_exact=False,
                                  rtol=1e-12, atol=1e-12)
