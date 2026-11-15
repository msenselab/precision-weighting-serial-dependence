"""The naming layer: manuscript labels everywhere outside the engine."""
import re

import pytest

from sdu import names
from sdu.model import space


def test_key_labels():
    assert names.label_internal("C1-T1n-S0-B9") == "C1-T3-S0-B4"    # internal spec name (engine)
    assert names.label("C1-T3-S0-B9") == "C1-T3-S0-B4"               # public name of the result files
    assert names.label_internal("C1-T1nS-S0-B9") == "C1-T3-S2-B4"   # reset at a switch
    assert names.label_internal("C1-T0-S0-B9") == "C1-T1-S0-B4"     # internal T0 is the manuscript's T1
    assert names.label("C1-T3-S4-B9") == "C1-T3-S2-B4"
    assert names.label_internal("C4-Tn-S0-B9") == "C4-T0-S0-B4"
    assert names.label("C1-T0-S0-B9") == "C1-T0-S0-B4"
    assert names.label("C1-T0-S0-B3") == "C1-T0-S0-B2"


@pytest.mark.parametrize("public", space.core_names())
def test_round_trip_reported_space(public):
    lab = names.label(public)
    assert names.public(lab) == public
    assert names.label_internal(names.code(lab)) == lab
    assert not re.search(r"T1n|Tn-|T1s|T0S|B[5-9]$", lab)


def test_reported_space_size_and_uniqueness():
    labels = [names.label(p) for p in space.core_names()]
    assert len(labels) == 572 and len(set(labels)) == 572
    assert all(label.split("-")[0] != "C0" for label in labels)


def test_experiment_codes():
    assert names.EXPERIMENT == {2: "Exp 1", 1: "Exp 2"}


def test_relabel_frame():
    import pandas as pd
    d = pd.DataFrame({"model": ["C3-T3-S0-B9", "C1-T3-S4-B9"], "exp": [2, 1]})
    out = names.relabel(d)
    assert out.model.tolist() == ["C3-T3-S0-B4", "C1-T3-S2-B4"]
    assert out.experiment.tolist() == ["Exp 1", "Exp 2"]
