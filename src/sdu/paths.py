"""Every location of data and results, in one place.

Model scripts read and write below MODEL_RESULTS. It defaults to the results of the original configuration
(results/model/legacy); a run of the v3 configuration sets SDU_MODEL_RESULTS=results/model/v3 (see scripts/hpc).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"                                    # experiment1/, experiment2/ (PsychoPy output, anonymised)
DATA_PROCESSED = DATA / "processed"
PREPROCESSED = DATA_PROCESSED / "preprocessing"            # E1_reprocessed.csv (Exp 2), E2_reprocessed.csv (Exp 1), ...
MODEL_INPUT = DATA_PROCESSED / "model_input"               # input_full_sequence.csv: trial sequences for the model

RESULTS = ROOT / "results"
RESULTS_BEHAVIOR = RESULTS / "behavior"
RESULTS_MODEL_LEGACY = RESULTS / "model" / "legacy"
RESULTS_MODEL_V3 = RESULTS / "model" / "v3"
MODEL_RESULTS = Path(os.environ.get("SDU_MODEL_RESULTS", RESULTS_MODEL_LEGACY))
if not MODEL_RESULTS.is_absolute():
    MODEL_RESULTS = ROOT / MODEL_RESULTS

MANUSCRIPT = ROOT / "manuscript"
NOTEBOOKS = ROOT / "notebooks"

# Figure outputs and the values plotted in them. The working repository keeps them inside the manuscript package;
# the public release has no manuscript folder and uses figures/ and results/figure_data/.
_REVISION = MANUSCRIPT / "revision"
FIGURES = _REVISION / "figures" if _REVISION.exists() else ROOT / "figures"
FIGURE_DATA = _REVISION / "data" if _REVISION.exists() else RESULTS / "figure_data"

# Preprocessed trial files. The file names keep the original collection order: E2 is Experiment 1 (ramped
# coherence), E1 is Experiment 2 (fixed coherence); the model's data codes follow the same files (2 = Exp 1).
PREPROCESSED_EXP = {"Exp 1": PREPROCESSED / "E2_reprocessed.csv", "Exp 2": PREPROCESSED / "E1_reprocessed.csv"}
