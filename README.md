# Precision-weighted updating explains serial dependence across sensory and contextual transitions 

Data, analysis code and stored results for the manuscript *Precision-Weighted Updating Explains Serial Dependence Across Sensory and Contextual Transitions* (Qu & Shi). Two duration-reproduction experiments varied the motion
coherence of the stimulus; a three-state Kalman filter accounts for central tendency and serial dependence.

Everything reported in the manuscript can be regenerated from the files in this repository. Model fitting and the
recovery simulations ran on a compute cluster; their outputs are stored under `results/`, so the figures and tables
can be reproduced in a few minutes on a laptop.

## Quick start

Install the dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Regenerate the figures:

```bash
uv run python -m sdu.behavior.figures                      # Figures 2-4      -> results/behavior/figures/
uv run python scripts/figures/fig5_v3.py figures/fig5_v3   # Figure 5 (model schematic)
uv run python scripts/figures/model_section.py             # Figures 6-7 and the values of Table B2
uv run python scripts/figures/appendix_compact.py          # Figures A1-A3, B1-B5, C1
```

Figures are written to `figures/` and the plotted values to `results/figure_data/`. Figure 1 is a static schematic
of the procedure (`figures/fig1_experimental_procedure.png`).

Run the tests with:

```bash
uv sync --extra dev
uv run pytest
```

## What reproduces what

| Manuscript | Code | Stored output |
|---|---|---|
| Primary mixed models, Equation 1 (Section 4.1) | `scripts/preprocessing/generate_manuscript_behavior_results.py` | `results/behavior/manuscript_ready/` |
| Figures 2-4 | `python -m sdu.behavior.figures` | `results/behavior/figures/` |
| Four coherence transitions (Appendix A.1) | `scripts/behavior/four_cell_trial_lmm.py`, `four_cell_participant_check.py`, `four_cell_sensitivity.py` | `results/behavior/analysis/four_cells/` |
| Response-error history (Appendix A.2) | `scripts/behavior/response_history_reanalysis.py`, `response_history_coherence.py` | `results/behavior/analysis/response_history/` |
| Mean shift with coherence (Appendix A.3) | `scripts/behavior/current_coherence_mean_shift.py`, `mean_shift_by_duration.py` | `results/behavior/analysis/current_coherence_mean_shift/` |
| Response variability (Appendix A.3) | `scripts/behavior/dispersion_check.py` | `results/behavior/analysis/dispersion/` |
| Predicted orderings, Bayes factors (Appendix A.4) | `scripts/behavior/behavior_orderings.py`, `report_behavior_orderings.py` | `results/behavior/analysis/behavior_extra/` |
| Model definition, model space, Table 1, parameters, sources of serial dependence (Section 5, Appendix B.1-B.3) | `notebooks/3skf_v3.ipynb` | `results/model/minimal_qdiag/readout_space/`, `readout_space_screening.csv` |
| Predictive checks (Section 5.4, Appendix B.4) | `scripts/model/reproduction_3skf_v3.py` | `results/model/minimal_qdiag/readout_space_ppc/` |
| Figures 6-7, Table B2 | `scripts/figures/model_section.py` | `figures/`, `results/figure_data/model_figures/` |
| Parameter and model recovery (Appendix B.5-B.6) | `notebooks/3skf_v3_recovery.ipynb`; `scripts/model/parameter_recovery_v3.py`, `model_recovery_v3.py`, `profile_phi_high_v3.py` and their `summarize_*`/`refine_*` companions | `results/model/minimal_qdiag/recovery_v3_restart_20261113/`, `model_recovery_v3/`, `coherence_evidence_v3_20261113/` |
| Sequence volatility, Glasauer & Shi data (Section 5.5, Appendix C) | `notebooks/glasauer_shi.ipynb`; `scripts/model/external/` | `results/model/minimal_qdiag/glasauer_shi/` |
| Appendix figures | `scripts/figures/appendix_compact.py` | `figures/`, `results/figure_data/compact_figures/` |

The behavioural scripts run in seconds, for example `uv run python scripts/behavior/four_cell_trial_lmm.py`.

## The model

`src/sdu/model/engine.py` implements the three-state Kalman observer and its exact likelihood; `space.py` defines
the model space, `fitting.py` the data loading and optimisation, `stats.py` the statistics shared by data and
simulations, and `screening_sim.py`/`screening.py` the comparison of every fitted model with the behavioural
effects. `scripts/model/fit_minimal_space.py` fits the model space (one participant and one chunk of models per work
unit; see its header).

The configuration reported in the manuscript is selected by environment variables:

```bash
GKF_SCALE=lin GKF_CORE=reduced GKF_BOUNDS=v3 GKF_INIT=gs GKF_NOISELESS=1 GKF_QDIAG=1 GKF_READOUT=1
```

The figure scripts and the notebooks set these themselves. Scripts under `scripts/model/` expect them to be set.
Model labels in the code carry an extra `R`: `C3-R-S0-B2` is the manuscript's `C3-S0-B2`.

## Data

- `data/processed/preprocessing/`: cleaned trial-level data of both experiments (CSV and pickle), with the
  participant screening and outlier flags.
- `data/processed/model_input/input_full_sequence.csv`: the complete trial sequences given to the model.
- `data/external/glasauer_shi_2021/`: the duration-reproduction data of Glasauer and Shi used in Section 5.5 and
  Appendix C, with the authors' model code. Source and licence (CC BY-SA 4.0) are in that folder.

**Experiment numbering.** File names keep the order of data collection. `E2_reprocessed` is Experiment 1 (ramped
coherence) and `E1_reprocessed` is Experiment 2 (constant coherence). The model's data code follows the files:
`exp = 2` is Experiment 1 and `exp = 1` is Experiment 2.

The raw experiment output is not included. `scripts/preprocessing/recompute_preprocessing.py` documents how the
cleaned files were produced from it.

## Layout

```text
data/       cleaned trial data, model input, external data
src/sdu/    Python package: behavioural analyses, the Kalman observer, reporting helpers
scripts/    entry points: preprocessing, behaviour, model, figures
notebooks/  model, recovery and transfer analyses with their outputs
results/    stored outputs of the analyses, fits, simulations and recovery runs
figures/    manuscript figures
tests/      unit tests
```

## Reproducibility notes

- Package versions are pinned in `pyproject.toml` to those that produced the stored results.
- Re-running the scripts reproduces the reported values. Stored files can still change in late decimals: the mixed
  models that fit at the boundary of the parameter space differ by up to about 1e-4 between runs, and the secondary
  Monte Carlo summaries of the ordering analysis (leave-one-out, probability sensitivity) vary with the sampling.
- The stored model fits (148 models, 22 participants per experiment) and the recovery simulations took many CPU
  hours. The protocol files in the recovery folders describe the designs.

## Licence and citation

Code is released under the MIT licence (`LICENSE`). The external data keep their own licence. If you use the data or
code, please cite the associated article.
