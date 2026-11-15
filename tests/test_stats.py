"""Main-text statistics (sdu.model.stats) on the observed data reproduce the manuscript's participant-level results."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from sdu import paths as SP
from sdu.model import fitting as FF
from sdu.model import stats as S

READY = SP.RESULTS_BEHAVIOR / "manuscript_ready"
DATASET = {2: "manuscript_exp1_dynamic", 1: "manuscript_exp2_fixed"}


@pytest.fixture(scope="module")
def observed():
    return {exp: S.observed(exp) for exp in (2, 1)}


def test_core_contrasts_defined():
    assert S.CORE == {2: ["low_high"], 1: ["LL_other_three"]}


def test_exp1_low_high_matches_main_text(observed):
    d = observed[2]
    assert len(d) == 22
    assert d.low_high.mean() == pytest.approx(0.04771831269722826, abs=1e-9)
    t = st.ttest_1samp(d.low_high, 0).statistic
    assert t == pytest.approx(3.5216789756182165, abs=1e-6)


def test_exp2_same_switch_matches_main_text(observed):
    assert observed[1].same_switch.mean() == pytest.approx(0.02789754706534767, abs=1e-9)


def test_exp2_ll_other_three_matches_contrast(observed):
    d = observed[1]
    assert d.LL_other_three.mean() == pytest.approx(0.06071896441040959, abs=1e-9)
    np.testing.assert_allclose(
        d.LL_other_three,
        d.b2_LL - (d.b2_LH + d.b2_HL + d.b2_HH) / 3,
        atol=1e-12,
    )


def test_r25_conditional_mean_shifts_match_participant_analysis(observed):
    assert observed[2].mean_bias_low_high.mean() == pytest.approx(-0.08325131319475902, abs=1e-9)
    assert observed[1].mean_bias_low_high.mean() == pytest.approx(-0.01105428719785276, abs=1e-9)
    for exp in (2, 1):
        d = observed[exp]
        np.testing.assert_allclose(d.mean_bias_Low - d.mean_bias_High, d.mean_bias_low_high, atol=1e-12)


def test_participant_slopes_match_manuscript_file(observed):
    ref = pd.read_csv(READY / "participant_controlled_slopes.csv", dtype={"subID": str})
    for exp, d in observed.items():
        r = ref[(ref.dataset == DATASET[exp]) & (ref.grouping == "current_coherence_label")]
        for cond in ("Low", "High"):
            want = r[r.condition == cond].set_index(r[r.condition == cond].subID.astype(int)).sdi_slope.sort_index()
            got = d.set_index("Sub")[f"b2_{cond}"].sort_index()
            np.testing.assert_allclose(got.loc[want.index].to_numpy(), want.to_numpy(), atol=1e-9)
            want_ct = r[r.condition == cond].set_index(r[r.condition == cond].subID.astype(int)).ct_slope.sort_index()
            np.testing.assert_allclose(d.set_index("Sub")[f"b1_{cond}"].loc[want_ct.index].to_numpy(),
                                       want_ct.to_numpy(), atol=1e-9)


def test_transition_cells_match_behavioral_analysis(observed):
    """Four-transition slopes equal the participant cell slopes of the behavioural analysis (Tables A1a-c)."""
    ref = pd.read_csv(SP.RESULTS_BEHAVIOR / "analysis" / "four_cells" / "participant_cell_slopes.csv",
                      dtype={"subID": str})
    for exp, name in ((2, "Exp1_dynamic"), (1, "Exp2_fixed")):
        r = ref[ref.experiment == name].assign(Sub=lambda d: d.subID.astype(int)).pivot(index="Sub", columns="cell",
                                                                                         values="previous_duration_slope")
        d = observed[exp].set_index("Sub")
        for cell in ("LL", "LH", "HL", "HH"):
            np.testing.assert_allclose(d.loc[r.index, f"b2_{cell}"].to_numpy(), r[cell].to_numpy(), atol=1e-9)


def test_block_boundary():
    T = np.array([1.0, 1.2, 1.4, 0.8, 1.6, 1.0])
    block_start = np.array([True, False, False, True, False, False])
    prev = S.previous_duration(T, block_start)
    assert np.isnan(prev[0]) and np.isnan(prev[3])
    assert prev[4] == 0.8 and prev[1] == 1.0


def test_simulated_cohort_uses_same_function():
    """participant_stats on arrays gives the same values as observed() for one participant."""
    exp = 2
    data = FF.load(exp)
    sub = sorted(data)[0]
    obs = S.observed(exp).set_index("Sub").loc[sub]
    one = S.participant_stats_arrays(exp, sub, data[sub][3])
    for k in ("b2_Low", "b2_High", "b2_LL", "LL_other_three", "LL_HL", "same_switch", "rsd_Low",
              "mean_bias_low_high"):
        assert one[k] == pytest.approx(obs[k], abs=1e-12)
