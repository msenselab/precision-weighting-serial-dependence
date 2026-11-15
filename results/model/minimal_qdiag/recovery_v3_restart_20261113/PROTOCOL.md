# 3SKF v3 parameter recovery

Fresh run prepared on 13 November 2026. The design and generating values are frozen in `design.csv`.
`design.json` records source/data hashes, random seeds and fitting settings.

## Models and parameters

Experiment 1 uses C3-S0-B2; Experiment 2 uses C3-S3-B2. The engine labels include `R` to select readout continuity.
Both models have these ten free parameters:

| Stored name | Reported quantity | Unit |
|---|---|---|
| lq1 | Fast innovation SD | s |
| lq2 | Slow innovation SD | s |
| beta | Low readout continuity, phi Low | proportion |
| wuH | High readout continuity, phi High | proportion |
| lam | Bias persistence, lambda | proportion |
| lq3 | Baseline bias innovation SD | s |
| d0 | Baseline response offset | s |
| ad0 | Low minus High response offset | s |
| lsde | Response noise SD | s |
| aq3 | Low/High bias innovation variance ratio | ratio |

The innovation SDs are the quantities optimized under v3 bounds. The engine also computes normalized variances
`(innovation SD / sigma_s)**2`; the primary recovery uses the optimized SDs. `sigma_s` is fixed at 0.1 in this
noise-free measurement configuration. R is fixed in the selected C3 models.

The baseline offset and bias SD are defined at coherence 1. The variance ratio is the ratio between coherence 0.3
and 0.7. The effective bias SD at each of those conditions is reported as a derived quantity.

## Simulate and refit

1. **Fitted-parameter design:** the 22 observed participants per experiment each generate 20 independent response
   data sets: 880 fits in total.
2. **Exploratory sampled-parameter design:** 200 independently varying parameter vectors per experiment are drawn with Latin
   hypercube sampling: 400 fits. Continuity spans 0 to 1. Other parameters span the observed fitted minimum to
   maximum for that experiment. The bias variance ratio is sampled on the log scale. Exact ranges are in
   `sampling_ranges.csv`.
3. Each simulation uses a real participant's 240-trial sequence, coherence sequence and response mask. The filter
   initialization and processing of the full sequence follow the main fits. The sampled design assigns vectors
   evenly across participant sequences.
4. Responses include newly drawn latent AR(1) bias and response noise. Simulation and fitting use separate seeds.
5. The same model is refitted using L-BFGS-B, the default start, 11 random starts and one zero-continuity start.
   All ten free parameters are estimated jointly. The generating values are used to evaluate the likelihood after
   fitting.
6. Each result includes generating and fitted parameters, every optimizer start's status, response hash and the
   likelihood difference between the fitted and generating parameters.
7. A best fit below the generating likelihood by more than 0.01, or an unconverged best fit, receives 49 additional
   independent starts. Original results stay in `raw/`; improvements are stored in `refined/`. Summaries use the
   highest-likelihood result for the identical simulated response sequence.

## Recovery summaries

For each free parameter: true versus recovered, Pearson correlation, recovery slope, bias, RMSE and the fraction
of fits at the lower/upper parameter bounds.

For the fitted-parameter design, two correlations and slopes are reported: the correspondence between the 22
generating values and each participant's mean recovered value, and the median correspondence across 22 participants
within each of the 20 repetitions. Bias, participant RMSE and boundary rates are calculated within each participant,
then averaged across the 22 participants. The usual pooled RMSE is also saved in the CSV. Scatter plots show both
individual fits and participant means. For the sampled design, all 200 distinct parameter vectors enter each
calculation. The sampled run is exploratory coverage of the selected parameter ranges.

Derived results include phi Low minus phi High, condition-specific mean trace weights `(1 - Kx) * phi`, their
Low minus High difference, and bias innovation SD by coherence. The S3 trace weight is zero on switch trials.
Direction accuracy is reported at participant and cohort-mean levels; zero true differences are excluded from
direction calculations. These are two different summaries.

The error correlation matrix uses fitted-parameter errors centered within each generating participant across
repetitions. This shows which estimates move together across repeated data sets.

## Reproduce

```bash
export PYTHONPATH=src
export GKF_SCALE=lin GKF_LIK=gh GKF_CORE=reduced GKF_BOUNDS=v3 GKF_INIT=gs
export GKF_NOISELESS=1 GKF_QDIAG=1 GKF_READOUT=1 GKF_NOISE_TREND=0 GKF_EFFENC=0
python -m pytest -q tests/test_parameter_recovery_v3.py
python scripts/model/parameter_recovery_v3.py --prepare
sbatch --array=0-31 scripts/hpc/parameter_recovery_v3_restart.sbatch
python scripts/model/summarize_parameter_recovery_v3.py
python scripts/model/refine_parameter_recovery_v3.py
python scripts/notebooks/build_3skf_v3_recovery.py
```

The preparation command creates a frozen design in a fresh output directory. Existing designs are reused by the
run and summary commands. Each completed unit has one CSV and a JSON list of optimizer-start diagnostics in `raw/`.

## References

- [Wilson & Collins (2019), Rule 5](https://elifesciences.org/articles/49547): simulate known parameters and refit.
- [CPM parameter recovery tutorial](https://devcompsy.github.io/cpm/tutorials/parameter-recovery.html): MLE recovery
  using individual parameters and observed task sequences.
- [CCN 2026 state-space tutorial](https://github.com/harrisonritz/DynamicsTutorial_CCN2026): LDS parameter learning
  and identifiability.
- [COSMOS recovery tutorial](https://cosmos-konstanz.github.io/notebooks/tutorial-3-model-comparisons.html):
  parameter recovery and parameter trade-offs.
