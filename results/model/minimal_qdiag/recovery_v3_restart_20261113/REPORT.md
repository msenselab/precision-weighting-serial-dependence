# 3SKF v3 parameter recovery

Completed 1280 of 1280 planned fits.

The primary design contains 22 generating participants per experiment, each with 20 new response data sets. Experiment 1 uses C3-S0-B2; Experiment 2 uses C3-S3-B2.

## Ten optimized parameters

Mean r compares the 22 true values with participant mean estimates over 20 fits. Cohort r is the median of 20 correlations across 22 participants. RMSE is calculated within participant, then averaged across participants. All values are on the optimizer's natural parameter scale.

### Experiment 1

| Parameter | Mean r | Cohort r | Mean slope | Bias | Participant RMSE | Lower bound | Upper bound |
|---|---|---|---|---|---|---|---|
| Fast innovation SD | 0.970 | 0.908 | 0.995 | -0.002 | 0.024 | 0.7% | 0.7% |
| Slow innovation SD | 0.847 | 0.360 | 0.511 | 0.000 | 0.030 | 42.7% | 0.0% |
| Low readout continuity | 0.922 | 0.575 | 0.643 | 0.031 | 0.263 | 32.3% | 7.0% |
| High readout continuity | 0.663 | 0.157 | 0.362 | 0.074 | 0.230 | 46.8% | 2.0% |
| Bias persistence | 0.912 | 0.699 | 0.717 | -0.079 | 0.218 | 3.0% | 2.7% |
| Baseline bias innovation SD | 0.662 | 0.349 | 0.576 | 0.015 | 0.062 | 2.0% | 0.0% |
| Baseline response offset | 0.996 | 0.964 | 0.963 | -0.004 | 0.043 | 0.0% | 1.8% |
| Low minus High response offset | 0.997 | 0.958 | 1.005 | 0.001 | 0.024 | 0.0% | 0.0% |
| Response noise SD | 0.737 | 0.468 | 0.520 | -0.008 | 0.044 | 11.8% | 0.0% |
| Low/High bias variance ratio | 0.730 | 0.287 | 0.311 | 0.138 | 4.198 | 12.0% | 20.7% |

### Experiment 2

| Parameter | Mean r | Cohort r | Mean slope | Bias | Participant RMSE | Lower bound | Upper bound |
|---|---|---|---|---|---|---|---|
| Fast innovation SD | 0.976 | 0.872 | 0.893 | 0.003 | 0.019 | 5.9% | 0.0% |
| Slow innovation SD | 0.966 | 0.812 | 0.836 | -0.001 | 0.016 | 24.1% | 0.0% |
| Low readout continuity | 0.850 | 0.508 | 0.475 | 0.060 | 0.383 | 32.7% | 17.5% |
| High readout continuity | 0.690 | 0.258 | 0.319 | 0.066 | 0.408 | 35.7% | 13.6% |
| Bias persistence | 0.690 | 0.434 | 0.591 | -0.138 | 0.299 | 7.5% | 1.1% |
| Baseline bias innovation SD | 0.667 | 0.381 | 0.504 | 0.016 | 0.080 | 5.0% | 0.0% |
| Baseline response offset | 0.998 | 0.893 | 1.010 | 0.001 | 0.046 | 0.0% | 0.0% |
| Low minus High response offset | 0.993 | 0.808 | 0.984 | 0.000 | 0.027 | 0.0% | 0.0% |
| Response noise SD | 0.730 | 0.461 | 0.477 | -0.016 | 0.051 | 8.2% | 0.0% |
| Low/High bias variance ratio | 0.781 | 0.342 | 0.247 | -0.548 | 3.869 | 24.8% | 13.9% |

## Coherence differences

Trace weights are computed from the actual per-trial Kalman gain and the current condition's phi, then averaged within High and Low. S3 sets the weight to zero on switch trials. Direction percentages refer to cohort means across 20 repeated simulations.

| Experiment | Difference | True mean | Recovered mean | Direction accuracy |
|---|---|---|---|---|
| 1 | Low minus High continuity | 0.174 | 0.132 | 100% |
| 1 | Low minus High trace weight | 0.056 | 0.047 | 95% |
| 2 | Low minus High continuity | 0.046 | 0.040 | 75% |
| 2 | Low minus High trace weight | 0.015 | 0.013 | 85% |

## Exploratory parameter sampling

A further 200 independent parameter vectors per experiment cover the observed fitted ranges, with continuity independently sampled from 0 to 1. Results are saved separately under design=range. Exact sampling bounds are in sampling_ranges.csv.

## Optimizer checks

Best-fit convergence: 100.0%. Fitted likelihood below the generating likelihood by more than 0.01: 0 of 1280 fits.

Additional-start fits retained: 2. Original fits remain in raw/ and refinements in refined/.

## Estimation error correlations

Errors are centered within each generating participant across repetitions before computing the 10 by 10 correlation matrix. This measures error covariance for repeated data at fixed generating values.

## Files

- PROTOCOL.md: design, parameter definitions and references.
- participant_summary.csv: each participant's mean, spread, bias, RMSE and boundary rates.
- parameter_summary.csv: parameter recovery metrics for the observed and sampled designs.
- condition_contrasts.csv: Low minus High differences in each simulated cohort.
- optimizer_diagnostics.csv: convergence and likelihood checks.
- design.json: seeds, source hashes and fitting configuration.
