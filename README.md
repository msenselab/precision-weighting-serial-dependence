# Precision weighting and serial dependence under uncertainty

This repository contains the public analysis code, cleaned data, fitted model outputs, and figures for the serial-dependence uncertainty project.

## Repository layout

```text
.
├── data/                         # Cleaned trial-level data
│   ├── experiment1/E1.pkl
│   └── experiment2/E2.pkl
├── code/                         # Scripted reproduction pipeline
│   ├── 01_preprocess.py
│   ├── 02_behavior_models.py
│   ├── 03_behavior_results.py
│   ├── 04_fit_kalman.py
│   ├── 05_validate_kalman.py
│   ├── 06_plot_behavior.py
│   └── 07_plot_kalman.py
├── models/three_state_kalman/    # Three-state Kalman model implementation
├── notebooks/                    # Saved-output analysis records for inspection
│   └── shared/                   # Notebook-local plotting configuration
├── results/                      # Canonical output tables
│   ├── preprocessing/
│   ├── behavioral_models/
│   ├── figure_source_data/
│   ├── kalman_model_fits/
│   └── kalman_model_checks/
├── figures/                      # Final figures in PDF/PNG, with SVG where available
├── pyproject.toml
└── uv.lock
```

## Data

The cleaned data files are:

- `data/experiment1/E1.pkl`
- `data/experiment2/E2.pkl`

Both files contain trial-level behavioral data with participant-screening and trial-level outlier flags already applied. Scripts that run behavioral analyses load these files directly.

## Reproducing the analysis

Install dependencies with `uv`:

```bash
uv sync
```

Run the behavioral tables/source-data pipeline:

```bash
uv run python code/02_behavior_models.py
uv run python code/03_behavior_results.py
```

Run the Kalman-model consistency checks from the included canonical fit outputs:

```bash
uv run python code/05_validate_kalman.py --skip-parameter-recovery
```

The exported files in `figures/` are the canonical manuscript figures. The plotting scripts in `code/06_plot_behavior.py` and `code/07_plot_kalman.py` are retained for inspecting public source-data tables and saved model outputs, but the manuscript figures should be treated as the authoritative exported assets.

Full Kalman model fitting/filtering is computationally expensive and is **not required** for the public release. Canonical fit outputs are included in `results/kalman_model_fits/`, and saved PPC/trial-level prediction outputs are included in `results/kalman_model_checks/`, so readers and notebooks can inspect the final model results without rerunning the Kalman filter.

`code/04_fit_kalman.py` is retained only as methodological source code for the original fitting procedure; it is not part of the recommended public reproduction path.

`code/01_preprocess.py` documents the preprocessing logic used to create the cleaned data. It expects the original raw experiment CSV folders, which are not included in this public repository.

## Notebooks

The notebooks in `notebooks/` are retained as saved-output analysis records. They include the final manuscript and appendix figure coverage, plus additional QC/intermediate plots from the analysis history. The notebook-local plotting helper lives in `notebooks/shared/`. The canonical exported figures remain in `figures/`, and the reproducible pipeline is the numbered script sequence in `code/`.

## Main outputs

- `results/behavioral_models/`: mixed-effects model tables and behavioral summaries.
- `results/figure_source_data/`: compact tables used to generate figures.
- `results/kalman_model_fits/`: full model-fit tables, model ranking, and parameter summaries.
- `results/kalman_model_checks/`: posterior-predictive checks, effect recovery, and parameter-recovery summaries for the winning model.
- `figures/`: exported final figures.

## Citation

If you use this data or code, please cite the associated publication/preprint once available.
