# Precision-Weighting Governs Serial Dependence Under Uncertainty and Contextual Change

**Authors:** Chunyu Qu, Zhuanghua Shi  
**Affiliation:** Neuro-cognitive Psychology, Department of Psychology, Ludwig-Maximilians-Universität München

## Overview

This repository contains the anonymized data and analysis code for:

> Qu, C., & Shi, Z. (2026). How the past persists: Precision-weighting governs serial dependence under uncertainty and contextual change. *Communications Psychology* (submitted).

Serial dependence—the attraction of current percepts toward recent stimuli—is modulated by both sensory uncertainty and contextual continuity. Across two time-reproduction experiments (*N* = 44), we show that a single precision-weighting process parsimoniously accounts for both effects. A three-state Kalman filter model—comprising fast (serial dependence), slow (central tendency), and bias (decision carryover) states—captures both patterns through coherence-dependent modulation of Kalman gain. What behavioral studies have characterized as contextual gating thus emerges naturally from precision-weighted inference.

## Repository Structure

```
├── data/
│   ├── experiment1/          # Anonymized data, Exp 1: Ramped coherence (N = 44)
│   └── experiment2/          # Anonymized data, Exp 2: Constant coherence (N = 44)
├── analysis/
│   ├── experiment1/          # Preprocessing, QC, behavioral analysis, LMMs
│   ├── experiment2/          # Preprocessing, QC, behavioral analysis, LMMs
│   └── combined/             # Publication figures and supplementary figures
├── modeling/
│   ├── 1_135Model_ThreeState.ipynb   # Model fitting and comparison
│   └── three_state_kalman/           # Three-state Kalman filter package
│       ├── config.py                 # Model configurations (C0–C3, S0–S1, B0–B3)
│       ├── engine.py                 # Kalman filter engine
│       ├── fitting.py                # Parameter optimization (MLE)
│       └── parameters.py             # Parameter definitions
├── shared/
│   └── plot_config.py        # Shared plotting configuration
└── README.md
```

## Experiments

### Experiment 1: Ramped Coherence
Participants reproduced durations of random dot kinematograms (RDKs) whose motion coherence ramped up and then down within each trial, creating continuous within-trial uncertainty variation. Two coherence levels (30% and 70%) were randomly interleaved.

### Experiment 2: Constant Coherence
Coherence remained constant within each trial at 30% or 70%, randomly interleaved across trials. Green dots during encoding and white dots during masks provided a salient categorical boundary. Trial-by-trial feedback was provided.

## Analysis Pipeline

Each experiment follows the same analysis pipeline:

1. **Data QC** (`1_Data_QC.ipynb`): Exclusion criteria, outlier detection, data summary
2. **Reproduction Bias** (`2_Reproduction_Bias_Check.ipynb`): Central tendency and Vierordt's law
3. **CT & SD Analysis** (`3_CT_SD_Analysis.ipynb`): Serial dependence index (SDI) and central tendency index (CTI) by condition
4. **LMM Models** (`4_LMM_Models.ipynb`): Linear mixed-effects models with progressive predictor inclusion

### Combined Figures
- `1_Publication_Figures.ipynb`: Main manuscript figures (Figs 2–5)
- `3_Supplementary_Figures.ipynb`: Supplementary analyses

## Computational Modeling

The three-state Kalman filter (`modeling/three_state_kalman/`) implements:

- **Fast state** ($\hat{x}$): Tracks recent stimuli → serial dependence
- **Slow state** ($\hat{m}$): Tracks distributional mean → central tendency
- **Bias state** ($\hat{b}$): Tracks response tendency → decision carryover

Model variants are defined in `config.py`:
- **Coherence modulation** (C0–C3): How motion coherence affects process noise
- **State reset** (S0–S1): Whether category switches reset state estimates
- **Bias mechanism** (B0–B3): How prior responses influence the current estimate

Model comparison uses AIC across all configurations to identify the best-fitting model per experiment.

## Requirements

```
python >= 3.10
numpy
pandas
scipy
matplotlib
seaborn
statsmodels
jupyter
```

Install dependencies:
```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels jupyter
```

## Data Format

Each CSV file contains one participant's trial-level data with the following key columns:

| Column | Description |
|--------|-------------|
| `participant` | Anonymous participant ID (S01, S02, ...) |
| `curDur` | Current stimulus duration (s) |
| `workerResp` | Reproduction response (s) |
| `curCoh` | Current coherence level (0.3 or 0.7) |
| `preDur` | Previous stimulus duration (s) |
| `preResp` | Previous response (s) |
| `SameSwitch` | Same or Switch coherence transition |

## License

This project is shared for academic and research purposes. Please cite the paper if you use this data or code.

## Contact

- Chunyu Qu — chunyu.qu@psy.lmu.de
- Zhuanghua Shi — strongway@psy.lmu.de
