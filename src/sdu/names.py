"""Model names: the one place that translates between the engine's codes and the manuscript's labels.

Three spellings exist:
  internal   engine Spec names, e.g. C1-T1n-S0-B9 (engine.py; T0 there is the manuscript's T1)
  public     names stored in the result files of fitting, e.g. C1-T3-S0-B9 (space.py)
  label      the manuscript's labels, e.g. C1-T3-S0-B4

Everything a person reads (notebooks, tables, figures, new result files) uses labels.
"""
from functools import lru_cache

import pandas as pd

from sdu.model import engine as G
from sdu.model import space as KS

EXPERIMENT = {2: "Exp 1", 1: "Exp 2"}          # data code -> manuscript experiment


def label(public: str) -> str:
    """Public name (result files) -> manuscript label."""
    return KS.label(public)


def label_internal(internal: str) -> str:
    """Engine Spec name -> manuscript label."""
    return KS.label(KS.public(G.Spec.parse(internal)))


@lru_cache(maxsize=None)
def _by_label():
    return {KS.label(p): p for p in KS.distinct_names() + KS.extension_names()}


def public(lab: str) -> str:
    """Manuscript label -> public name."""
    return _by_label()[lab]


def code(lab: str) -> str:
    """Manuscript label -> engine Spec name."""
    return KS.internal(public(lab)).name


def relabel(d: pd.DataFrame, column: str = "model") -> pd.DataFrame:
    """Copy of a result table with the public model names in `column` replaced by labels and, when an `exp` column
    exists, an `experiment` column added."""
    out = d.copy()
    out[column] = out[column].map(label)
    if "exp" in out:
        out["experiment"] = out["exp"].map(EXPERIMENT)
    return out
