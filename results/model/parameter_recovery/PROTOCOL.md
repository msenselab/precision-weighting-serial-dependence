# Parameter recovery (Appendix B.5)

The design and the generating values are frozen in `design.csv`; `design.json` records the random seeds, the
fitting settings, the parameter definitions and the hash of `design.csv`.

## Models and parameters

Experiment 1 uses C3-S0-B2; Experiment 2 uses C3-S3-B2. Both models have these ten free parameters:

| Name | Reported quantity | Unit |
|---|---|---|
| sd_q1 | Fast innovation SD | s |
| sd_q2 | Slow innovation SD | s |
| phi_L | Low readout continuity, phi Low | proportion |
| phi_H | High readout continuity, phi High | proportion |
| lambda | Bias persistence, lambda | proportion |
| sd_q3 | Baseline bias innovation SD | s |
| d0 | Baseline response offset | s |
| d0_LH | Low minus High response offset | s |
| sigma_e | Response noise SD | s |
| ratio_q3 | Low/High bias innovation variance ratio | ratio |

The innovation SDs are the fitted quantities. The engine also computes the normalized variances
`q_j = (innovation SD / SIGMA_S)**2` with SIGMA_S = 0.1 s (sdu.model.engine); the recovery reports the fitted SDs.
R is fixed in the selected C3 models.

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
   independent starts. During computation, original results stay in `raw/` and improvements in `refined/`.
   The public archive retains them separately in `fit_records_initial.csv` and `fit_records_refined.csv`;
   `fit_records.csv` stores the selected highest-likelihood row for each response sequence.

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
uv sync --extra fit
uv run python scripts/model/parameter_recovery.py --prepare --out <new directory>   # freezes a new design
uv run python scripts/model/parameter_recovery.py --task=<i> --ntasks=<n>           # one array task per i
uv run python scripts/model/parameter_recovery_refine.py
uv run python scripts/model/parameter_recovery_summary.py
```

`--prepare` refuses to overwrite a frozen design; with the default output directory the run and summary commands
use the stored design. The stored run had 32 array tasks; units 579 and 837 were refined with 49 additional starts
(`execution.json`). `fit_records_initial.csv` contains all 1,280 initial fits, `fit_records_refined.csv` the two
additional-start fits, and `fit_records.csv` the selected fits. `optimizer_diagnostics.csv` contains convergence
and likelihood checks. Per-unit working files are omitted from the public archive.

## References

- [Wilson & Collins (2019), Rule 5](https://elifesciences.org/articles/49547): simulate known parameters and refit.
- [CPM parameter recovery tutorial](https://devcompsy.github.io/cpm/tutorials/parameter-recovery.html): MLE recovery
  using individual parameters and observed task sequences.
- [CCN 2026 state-space tutorial](https://github.com/harrisonritz/DynamicsTutorial_CCN2026): LDS parameter learning
  and identifiability.
- [COSMOS recovery tutorial](https://cosmos-konstanz.github.io/notebooks/tutorial-3-model-comparisons.html):
  parameter recovery and parameter trade-offs.
