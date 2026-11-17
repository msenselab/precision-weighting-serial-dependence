"""Every location of data and results, in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"                                    # raw experiment output (not distributed)
PREPROCESSED = DATA / "processed" / "preprocessing"        # cleaned trial data, participant screening
MODEL_INPUT = DATA / "processed" / "model_input" / "input_full_sequence.csv"   # complete trial sequences
EXTERNAL = DATA / "external"

# Cleaned trial data of the two experiments, by experiment number as in the manuscript.
PREPROCESSED_EXP = {1: PREPROCESSED / "experiment1_ramped.csv", 2: PREPROCESSED / "experiment2_constant.csv"}
PREPROCESSED_PKL = {e: p.with_suffix(".pkl") for e, p in PREPROCESSED_EXP.items()}

RESULTS = ROOT / "results"
RESULTS_BEHAVIOR = RESULTS / "behavior"
BEHAVIOR_PRIMARY = RESULTS_BEHAVIOR / "primary"
BEHAVIOR_ANALYSIS = RESULTS_BEHAVIOR / "analysis"
BEHAVIOR_TRANSITIONS = BEHAVIOR_ANALYSIS / "transitions"
BEHAVIOR_HISTORY = BEHAVIOR_ANALYSIS / "response_history"
BEHAVIOR_MEAN_SHIFT = BEHAVIOR_ANALYSIS / "mean_shift"
BEHAVIOR_VARIABILITY = BEHAVIOR_ANALYSIS / "variability"
BEHAVIOR_ORDERINGS = BEHAVIOR_ANALYSIS / "orderings"
BEHAVIOR_SEQUENCES = BEHAVIOR_ANALYSIS / "sequence_audit"
RESULTS_MODEL = RESULTS / "model"
FITS = RESULTS_MODEL / "fits"                              # maximum-likelihood fits of the 148 models
MODEL_COMPARISON = RESULTS_MODEL / "model_comparison.csv"  # AIC and simulation screening of every model
PREDICTIVE_CHECKS = RESULTS_MODEL / "predictive_checks"
PARAMETER_RECOVERY = RESULTS_MODEL / "parameter_recovery"
MODEL_RECOVERY = RESULTS_MODEL / "model_recovery"
HIGH_CONTINUITY_PROFILE = RESULTS_MODEL / "high_continuity_profile"
GLASAUER_SHI = RESULTS_MODEL / "glasauer_shi"

NOTEBOOKS = ROOT / "notebooks"
FIGURES = RESULTS / "figures"
FIGURES_MAIN = FIGURES / "main"                            # Figures 1-7 and behavioural checks
FIGURES_APPENDIX = FIGURES / "appendix"                    # Appendix figures
FIGURE_DATA = FIGURES / "data"                             # values plotted in the figures
