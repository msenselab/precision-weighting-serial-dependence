# Precision-Weighting Governs Serial Dependence Under Uncertainty and Contextual Change

**Authors:** Chunyu Qu, Zhuanghua Shi
**Affiliation:** Neuro-cognitive Psychology, Department of Psychology, Ludwig-Maximilians-Universität München

## Overview

This repository contains the anonymized data and analysis code for:

> Qu, C., & Shi, Z. (2026). How the past persists: Precision-weighting governs serial dependence under uncertainty and contextual change (submitted).

Serial dependence—the attraction of current percepts toward recent stimuli—is modulated by both sensory uncertainty and contextual continuity. Across two time-reproduction experiments (*N* = 44, 22 per experiment), we show that a single precision-weighting process parsimoniously accounts for both effects. A three-state Kalman filter model—comprising fast (serial dependence), slow (central tendency), and bias (decision carryover) states—captures both patterns through coherence-dependent modulation of Kalman gain. What behavioral studies have characterized as contextual gating thus emerges naturally from precision-weighted inference.

## Repository Structure

```
├── data/
│   ├── experiment1/          # Cleaned data, Exp 1: Dynamic coherence (N = 22, 5,280 trials)
│   │   └── E1.pkl
│   └── experiment2/          # Cleaned data, Exp 2: Constant coherence (N = 22, 5,104 trials)
│       └── E2.pkl
├── analysis/
│   ├── experiment1/          # Behavioral analysis, Exp 1
│   │   ├── 1_Reproduction_Bias_Check.ipynb
│   │   ├── 2_CT_SD_Analysis.ipynb
│   │   └── 3_LMM_Models.ipynb
│   ├── experiment2/          # Behavioral analysis, Exp 2
│   │   ├── 1_Reproduction_Bias_Check.ipynb
│   │   ├── 2_CT_SD_Analysis.ipynb
│   │   └── 3_LMM_Models.ipynb
│   └── combined/             # Publication and supplementary figures
│       ├── 1_Publication_Figures.ipynb
│       └── 2_Supplementary_Figures.ipynb
├── modeling/
│   ├── 1_135Model_ThreeState.ipynb   # Model fitting, comparison, and PPC
│   └── three_state_kalman/           # Three-state Kalman filter package
│       └── three_state_135_nolog/
│           ├── config.py             # 135-model space (C × S × B axes)
│           ├── engine.py             # Kalman filter equations
│           ├── fitting.py            # MLE parameter optimization
│           ├── ppc.py                # Posterior predictive checks
│           └── parameters.py        # Parameter bounds and definitions
├── shared/
│   └── plot_config.py        # Shared plotting configuration (Nature style)
├── figures/       # Generated publication figures
├── pyproject.toml
└── README.md
```

## Experiments

### Experiment 1: Dynamic Coherence
Participants reproduced durations of random dot kinematograms (RDKs) whose motion coherence ramped up and then down within each trial, creating continuous within-trial uncertainty variation. Two coherence levels (30% and 70%) were randomly interleaved across trials.

### Experiment 2: Constant Coherence
Coherence remained constant within each trial at either 30% or 70%, randomly interleaved across trials. This design isolates the effect of between-trial coherence transitions on serial dependence.

## Analysis Pipeline

Both experiments follow the same three-stage analysis pipeline. Data are provided as pre-processed `.pkl` files with outlier trials flagged (`is_outlier`).

1. **Reproduction Bias** (`1_Reproduction_Bias_Check.ipynb`): Central tendency effect and Vierordt's law verification
2. **CT & SD Analysis** (`2_CT_SD_Analysis.ipynb`): Serial dependence index (SDI) and central tendency index (CTI) as a function of coherence and transition type; RM-ANOVA and pairwise comparisons
3. **LMM Models** (`3_LMM_Models.ipynb`): Linear mixed-effects models with progressive predictor inclusion; model comparison via AIC/BIC

### Combined Figures
- `1_Publication_Figures.ipynb`: Main manuscript figures
- `2_Supplementary_Figures.ipynb`: Supplementary analyses

## Computational Modeling

The three-state Kalman filter (`modeling/three_state_kalman/`) implements a compositional model space of **135 configurations**:

- **Fast state** (*μ*): Tracks recent stimuli → serial dependence
- **Slow state** (*m*): Tracks distributional mean → central tendency
- **Bias state** (*b*): Tracks response tendency → decision carryover

The model space is defined along three independent axes:

| Axis | Levels | Description |
|------|--------|-------------|
| **C-axis** | 15 | Which Kalman filter parameters (Q₁, Q₂, Q₃, R) are modulated by coherence |
| **S-axis** | 3 | Whether and how category switches reset state estimates (none / state reset / gain reset) |
| **B-axis** | 3 | How prior responses influence the current estimate (update / prediction / response) |

Coherence modulation follows:
- Process noise: *Q(c)* = *Q*_base × exp(*α* × (1 − *c*))
- Measurement noise: *R(c)* = 1 if *c* ≥ 0.5, *r*_low if *c* < 0.5

Model comparison uses AIC across all 135 configurations. Posterior predictive checks (PPC) assess recovery of CTI and SDI.

## Data Format

Data are provided as pandas DataFrames (`.pkl`) with one row per trial. Key columns:

| Column | Description |
|--------|-------------|
| `subID` | Anonymous participant ID (numeric code) |
| `trial_num` | Trial number within session |
| `curDur` | Current stimulus duration (s) |
| `rpr` | Reproduction response (s) |
| `curBias` | Reproduction error: `rpr − curDur` (s) |
| `curCoherence` | Current motion coherence (0.3 or 0.7) |
| `preDur1back` | Previous stimulus duration (s) |
| `preCoherence1back` | Previous coherence level |
| `TransitionType` | Coherence transition type (HH / HL / LH / LL) |
| `is_outlier` | Outlier flag (IQR-based exclusion across participants) |
| `preDur2back` – `preDur10back` | Stimulus durations 2–10 trials back |
| `preCoherence2back` – `preCoherence10back` | Coherence levels 2–10 trials back |

## Requirements

```
python >= 3.10
numpy >= 1.24
pandas >= 2.0
scipy >= 1.10
statsmodels >= 0.14
matplotlib >= 3.7
seaborn >= 0.12
pingouin >= 0.5
jupyter
```

**Recommended:** Install with [uv](https://github.com/astral-sh/uv) for exact dependency reproduction:

```bash
uv sync
```

Or with pip:

```bash
pip install numpy pandas scipy matplotlib seaborn statsmodels pingouin jupyter
```

## License

This repository is licensed under the MIT License — see [LICENSE](LICENSE) for details.
If you use this data or code in your research, please cite the paper above.
