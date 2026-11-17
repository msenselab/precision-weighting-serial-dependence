# Precision-weighted updating explains serial dependence across sensory and contextual transitions

Data, analysis code and stored results for the manuscript *Precision-Weighted Updating Explains Serial Dependence
Across Sensory and Contextual Transitions* (Qu & Shi). Two duration-reproduction experiments varied the motion
coherence of the stimulus: **Experiment 1** with ramped coherence and **Experiment 2** with constant coherence. A
three-state Kalman observer accounts for central tendency and serial dependence.

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
uv run python -m sdu.behavior.figures            # Figures 2-4 and behavioural checks
uv run python scripts/figures/model_schematic.py # Figure 5
uv run python scripts/figures/model_results.py   # Figures 6-7 and Table B2 values
uv run python scripts/figures/appendix.py        # Figures A1-A3, B1-B5, C1
```

Figures are written under `results/figures/`; their plotted values are under `results/figures/data/`. Figure 1 is a
static schematic of the procedure (`results/figures/main/fig1_experimental_procedure.png`).

Run the tests with:

```bash
uv sync --extra dev --extra fit
uv run pytest
```

## Analysis map

Short links connect each manuscript result to its code and stored output.

### Behavioural analyses

| Manuscript result | Files |
|---|---|
| Primary mixed models · Section 4.1, Equation 1 | [code](scripts/behavior/primary.py) · [output](results/behavior/primary/) |
| Figures 2-4 | [code](src/sdu/behavior/figures.py) · [output](results/figures/main/) |
| Four coherence transitions · Appendix A.1 | [trial LMM](scripts/behavior/transitions.py) · [participant check](scripts/behavior/transition_participants.py) · [sensitivity](scripts/behavior/transition_sensitivity.py) · [output](results/behavior/analysis/transitions/) |
| Response-error history · Appendix A.2 | [main analysis](scripts/behavior/response_history.py) · [coherence](scripts/behavior/response_history_coherence.py) · [output](results/behavior/analysis/response_history/) |
| Mean shift with coherence · Appendix A.3 | [mean shift](scripts/behavior/mean_shift.py) · [by duration](scripts/behavior/mean_shift_duration.py) · [output](results/behavior/analysis/mean_shift/) |
| Response variability · Appendix A.3 | [code](scripts/behavior/variability.py) · [output](results/behavior/analysis/variability/) |
| Predicted orderings and Bayes factors · Appendix A.4 | [code](scripts/behavior/orderings.py) · [output](results/behavior/analysis/orderings/) |

Behavioural scripts run in seconds. For example:

```bash
uv run python scripts/behavior/transitions.py
```

### Model analyses

| Manuscript result | Files |
|---|---|
| Model definition, candidate space, and fits · Section 5, Appendices B.1-B.3 | [model](src/sdu/model/engine.py) · [fitting](scripts/model/fit_models.py) · [output](results/model/fits/all.csv) |
| Model comparison and screening · Table 1, Figure B1 | [comparison](src/sdu/model/comparison.py) · [notebook](notebooks/three_state_kalman.ipynb) · [output](results/model/model_comparison.csv) |
| Predictive checks · Section 5.4, Figure 7, Appendix B.4 | [code](scripts/model/predictive_checks.py) · [output](results/model/predictive_checks/) |
| Parameters and sources of serial dependence · Figure 6, Table B2 | [notebook](notebooks/three_state_kalman.ipynb) · [figure code](scripts/figures/model_results.py) · [output](results/figures/data/model/) |

### Recovery analyses

| Manuscript result | Files |
|---|---|
| Parameter recovery · Appendix B.5 | [simulate](scripts/model/parameter_recovery.py) · [refine](scripts/model/parameter_recovery_refine.py) · [summarize](scripts/model/parameter_recovery_summary.py) · [notebook](notebooks/recovery.ipynb) · [output](results/model/parameter_recovery/) |
| S0 versus S3 model recovery · Appendix B.6 | [simulate](scripts/model/model_recovery.py) · [summarize](scripts/model/model_recovery_summary.py) · [output](results/model/model_recovery/) |
| High-continuity profiles · Appendix B.6 | [code](scripts/model/continuity_profile.py) · [output](results/model/high_continuity_profile/) |

### External transfer and appendix figures

| Manuscript result | Files |
|---|---|
| Sequence volatility and Glasauer-Shi data · Section 5.5, Appendix C | [code](scripts/model/glasauer_shi/) · [notebook](notebooks/glasauer_shi.ipynb) · [output](results/model/glasauer_shi/) |
| Appendix figures | [code](scripts/figures/appendix.py) · [figures](results/figures/appendix/) · [plotted values](results/figures/data/appendix/) |

## The model

The code implements the manuscript's three-state Kalman observer. It separates a fast estimate, a learned mean, and
a slowly varying response-bias state. The preceding fast-state trace enters the current readout but is not written
back into the filter.

### Selected models

- Experiment 1: `C3-S0-B2`
- Experiment 2: `C3-S3-B2`

### Candidate space

- `C1-C15`: which noise terms depend on coherence
- `S0-S3`: how a coherence switch affects the preceding trace
- `B0-B2`: response-bias state structure

Invalid combinations are excluded, leaving 148 candidate models. Full parameter definitions, transformations, and
bounds are documented in Appendix B and the recovery protocols.

### Key files

- `src/sdu/model/engine.py`: observer, likelihood, simulation, and model space
- `src/sdu/model/fitting.py`: data loading and multistart fitting
- `src/sdu/model/stats.py`: behavioural summaries shared by data and simulations
- `src/sdu/model/comparison.py`: model comparison and behavioural screening
- `scripts/model/fit_models.py`: cluster entry point for fitting the model space

Model fitting requires JAX: `uv sync --extra fit`.

## Data

- `data/processed/preprocessing/`: cleaned trial-level data of Experiment 1 (`experiment1_ramped.csv`) and
  Experiment 2 (`experiment2_constant.csv`), as CSV and pickle, with the participant screening and outlier flags.
- `data/processed/model_input/input_full_sequence.csv`: the complete trial sequences given to the model; `exp` is the
  experiment number.
- `data/external/glasauer_shi_2021/`: the duration-reproduction data of Glasauer and Shi used in Section 5.5 and
  Appendix C, with the authors' model code. Source and licence (CC BY-SA 4.0) are in that folder.

The raw experiment output is not included. `scripts/preprocessing/preprocess.py` documents how the
cleaned files were produced from it.

## Layout

```text
data/       cleaned trial data, model input, external data
src/sdu/    Python package: behavioural analyses, the Kalman observer, reporting helpers
scripts/    entry points: preprocessing, behaviour, model, figures
notebooks/  model, recovery and Glasauer & Shi analyses with their outputs
results/    stored analyses, fitted models, figures and recovery runs
tests/      unit tests
```

## Reproducibility

- Dependency versions are pinned in `pyproject.toml` and `uv.lock`.
- Figures and tables use the stored outputs and regenerate on a laptop. Full model fitting and recovery require many
  CPU hours; stored fits are provided because refits can reach different local likelihood maxima.
- Recovery task files are consolidated under `results/model/parameter_recovery/`, `model_recovery/`, and
  `high_continuity_profile/`. Their protocol files document the designs and stored records.

## Licence and citation

Code is released under the MIT licence (`LICENSE`). The external data keep their own licence. If you use the data or
code, please cite the associated article.
