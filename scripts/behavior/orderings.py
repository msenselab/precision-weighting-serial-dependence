"""Compare predicted orderings of the four transition slopes (Appendix A.4, Figure A3C).

Primary: all within-group pairs differ by <= .02 and all ordered groups differ
by > .02. Sensitivity: .015/.03 equivalence bounds and directional gaps > 0.
Sampling compatibility, bootstrap frequency, and Bayesian evidence are distinct.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits
from sdu import paths as SP  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = SP.BEHAVIOR_ORDERINGS
CELLS = ("LL", "LH", "HL", "HH")
PATTERNS = ("reliability", "current", "reset", "flat")
LABELS = {
    "reliability": "HL > LL ~= HH > LH",
    "current": "HL ~= LL > HH ~= LH",
    "reset": "LL > HH ~= LH ~= HL",
    "flat": "LL ~= LH ~= HL ~= HH",
}
# Orthonormal contrasts; the unrestricted common slope drops out of every hypothesis.
TRANSFORM = np.array([[.5, -.5, .5, -.5], [.5, .5, -.5, -.5], [.5, -.5, -.5, .5]])
SETTINGS = tuple((delta, delta if convention == "meaningful_gap" else 0., convention)
                 for delta in (.015, .02, .03)
                 for convention in ("meaningful_gap", "direction_only"))
PRIMARY_TAU = .10
PRIOR_SCALES = (.05, .10, .20)
SEED = 20261103


def region_masks(values: np.ndarray, delta: float, gap: float) -> dict[str, np.ndarray]:
    """Full group constraints, including every between- and within-group pair."""
    if delta <= 0 or gap < 0:
        raise ValueError("delta must be positive and gap nonnegative")
    ll, lh, hl, hh = np.asarray(values).T
    baseline_min = np.minimum.reduce([hh, lh, hl])
    baseline_max = np.maximum.reduce([hh, lh, hl])
    return {
        "reliability": ((np.abs(ll - hh) <= delta) &
                        (hl - np.maximum(ll, hh) > gap) &
                        (np.minimum(ll, hh) - lh > gap)),
        "current": ((np.abs(hl - ll) <= delta) & (np.abs(hh - lh) <= delta) &
                    (np.minimum(hl, ll) - np.maximum(hh, lh) > gap)),
        "reset": ((baseline_max - baseline_min <= delta) & (ll - baseline_max > gap)),
        "flat": np.max(values, axis=1) - np.min(values, axis=1) <= delta,
    }


def component_masks(values, delta=.02, gap=.02):
    ll, lh, hl, hh = np.asarray(values).T
    return {
        "reliability: HL exceeds both middle cells": hl - np.maximum(ll, hh) > gap,
        "reliability: both middle cells exceed LH": np.minimum(ll, hh) - lh > gap,
        "reliability: LL and HH approximately equal": np.abs(ll - hh) <= delta,
        "current: HL and LL approximately equal": np.abs(hl - ll) <= delta,
        "current: HH and LH approximately equal": np.abs(hh - lh) <= delta,
        "current: both current-L cells exceed both current-H cells": np.minimum(hl, ll) - np.maximum(hh, lh) > gap,
        "reset: LL exceeds each other cell": ll - np.maximum.reduce([hh, lh, hl]) > gap,
        "reset: all other cells pairwise approximately equal": np.maximum.reduce([hh, lh, hl]) - np.minimum.reduce([hh, lh, hl]) <= delta,
        "flat: all four cells pairwise approximately equal": np.ptp(values, axis=1) <= delta,
    }


def niw_marginal(observations, tau, kappa0=1., nu0=5.):
    """Exact multivariate-t marginal for the mean under a proper NIW prior.

    d=3, Sigma ~ IW(nu0, Psi0), mu|Sigma ~ N(0,Sigma/kappa0).
    Psi0=kappa0*(nu0-d+1)*tau^2 I makes the prior mean's t scale tau.
    Returns mean, t scale matrix, degrees of freedom. No fixed covariance plug-in.
    """
    observations = np.asarray(observations, dtype=float).reshape(-1, 3)
    n, d = observations.shape
    df0 = nu0 - d + 1
    psi = kappa0 * df0 * tau**2 * np.eye(d)
    if n == 0:
        return np.zeros(d), psi / (kappa0 * df0), df0
    mean = observations.mean(axis=0)
    centered = observations - mean
    kappa = kappa0 + n
    psi_n = psi + centered.T @ centered + kappa0 * n / kappa * np.outer(mean, mean)
    df = nu0 + n - d + 1
    return n / kappa * mean, psi_n / (kappa * df), df


def multivariate_t(rng, location, scale, df, count):
    gaussian = rng.standard_normal((count, len(location))) @ np.linalg.cholesky(scale).T
    return location + gaussian / np.sqrt(rng.chisquare(df, count)[:, None] / df)


def interval(successes, draws, alpha=.025):
    """97.5% binomial MC interval; two such intervals give >=95% ratio bounds."""
    lo = 0. if successes == 0 else stats.beta.ppf(alpha / 2, successes, draws - successes + 1)
    hi = 1. if successes == draws else stats.beta.ppf(1 - alpha / 2, successes + 1, draws - successes)
    return float(lo), float(hi)


def count_samples(generator, experiment, method, tau=None):
    counts, components = Counter(), Counter()
    total = 0
    for contrasts in generator:
        cells = contrasts @ TRANSFORM
        total += len(cells)
        for delta, gap, convention in SETTINGS:
            masks = region_masks(cells, delta, gap)
            for pattern, mask in masks.items():
                counts[(delta, gap, convention, pattern)] += int(mask.sum())
            masks_array = np.stack(list(masks.values()))
            counts[(delta, gap, convention, "outside_all_patterns")] += int((~masks_array.any(axis=0)).sum())
            counts[(delta, gap, convention, "overlap")] += int((masks_array.sum(axis=0) > 1).sum())
        for component, mask in component_masks(cells).items():
            components[component] += int(mask.sum())
    rows = []
    for (delta, gap, convention, pattern), successes in counts.items():
        lo, hi = interval(successes, total)
        rows.append(dict(experiment=experiment, method=method, prior_tau=tau, delta=delta,
                         min_gap=gap, convention=convention, pattern=pattern, label=LABELS.get(pattern, pattern),
                         probability=successes / total, mc_successes=successes, mc_draws=total,
                         mc_se=np.sqrt((successes / total) * (1 - successes / total) / total),
                         mc_lower_97_5=lo, mc_upper_97_5=hi))
    component_rows = [dict(experiment=experiment, method=method, prior_tau=tau, delta=.02, min_gap=.02,
                           component=k, probability=v / total, mc_successes=v, mc_draws=total)
                      for k, v in components.items()]
    return rows, component_rows


def chunks_t(rng, location, scale, df, count, chunk=100000):
    for start in range(0, count, chunk):
        yield multivariate_t(rng, location, scale, df, min(chunk, count - start))


def load_slopes():
    matrices, rows, input_hashes = {}, [], {}
    for experiment, filename in [("Exp 1", "experiment1_ramped.csv"), ("Exp 2", "experiment2_constant.csv")]:
        path = SP.PREPROCESSED / filename
        input_hashes[str(path.relative_to(SP.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = pd.read_csv(path, dtype={"subID": str})
        valid = ~raw.is_outlier.astype(str).str.lower().isin(["true", "1"])
        frame = raw.loc[valid].dropna(subset=["subID", "curBias", "curDur", "preDur1back", "curCoherence", "preCoherence1back"]).copy()
        assert len(frame) == {"Exp 1": 4693, "Exp 2": 4654}[experiment]
        frame["cell"] = np.where(np.isclose(frame.preCoherence1back, .3), "L", "H") + np.where(np.isclose(frame.curCoherence, .3), "L", "H")
        for (participant, cell), group in frame.groupby(["subID", "cell"]):
            x = np.column_stack([np.ones(len(group)), group.curDur - 1.2, group.preDur1back - 1.2])
            assert np.linalg.matrix_rank(x) == 3
            beta = np.linalg.lstsq(x, group.curBias.to_numpy(float), rcond=None)[0]
            rows.append(dict(experiment=experiment, participant=participant, cell=cell, n_trials=len(group),
                             current_duration_slope=beta[1], previous_duration_slope=beta[2]))
        matrix = pd.DataFrame([r for r in rows if r["experiment"] == experiment]).pivot(index="participant", columns="cell", values="previous_duration_slope").loc[:, CELLS]
        assert matrix.shape == (22, 4) and matrix.notna().all().all()
        matrices[experiment] = matrix
    return matrices, pd.DataFrame(rows), input_hashes


def run_analysis(matrices=None, output=OUT, *, n_mc=1000000, n_prior=2000000,
                 n_boot=100000, n_loo=100000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loaded, participant_rows, input_hashes = load_slopes()
    if matrices is None:
        matrices = loaded

    summary, pairwise, mass_rows, component_rows = [], [], [], []
    for exp, matrix in matrices.items():
        values = matrix.loc[:, CELLS].to_numpy()
        n = len(values)
        for j, cell in enumerate(CELLS):
            mean, se = values[:, j].mean(), stats.sem(values[:, j])
            lo, hi = stats.t.interval(.95, n - 1, loc=mean, scale=se)
            summary.append(dict(experiment=exp, cell=cell, n=n, mean=mean, se=se, ci_low=lo, ci_high=hi))
        for a, b in itertools.combinations(range(4), 2):
            difference = values[:, a] - values[:, b]
            test = stats.ttest_1samp(difference, 0)
            lo, hi = stats.t.interval(.95, n - 1, loc=difference.mean(), scale=stats.sem(difference))
            pairwise.append(dict(experiment=exp, contrast=f"{CELLS[a]} - {CELLS[b]}", estimate=difference.mean(),
                                 se=stats.sem(difference), ci_low=lo, ci_high=hi, raw_p=test.pvalue))
    print("Four-cell participant means verified: 22 participants per experiment.", flush=True)

    with threadpool_limits(limits=1):
        for index, (exp, matrix) in enumerate(matrices.items()):
            z = matrix.loc[:, CELLS].to_numpy() @ TRANSFORM.T
            rng = np.random.default_rng(SEED + 10 + index)
            normal = (rng.multivariate_normal(z.mean(axis=0), np.cov(z.T) / len(z), min(100000, n_mc - start))
                      for start in range(0, n_mc, 100000))
            rows, components = count_samples(normal, exp, "normal_sampling")
            mass_rows.extend(rows)
            component_rows.extend(components)
            rng = np.random.default_rng(SEED + 20 + index)
            def bootstrap():
                for start in range(0, n_boot, 10000):
                    count = min(10000, n_boot - start)
                    weights = rng.multinomial(len(z), np.ones(len(z)) / len(z), size=count)
                    yield weights @ z / len(z)
            rows, components = count_samples(bootstrap(), exp, "participant_bootstrap")
            mass_rows.extend(rows)
            component_rows.extend(components)
        for index, tau in enumerate(PRIOR_SCALES):
            loc, scale, df = niw_marginal(np.empty((0, 3)), tau)
            rng = np.random.default_rng(SEED + 100 + index)
            rows, _ = count_samples(chunks_t(rng, loc, scale, df, n_prior), "prior", "niw_prior", tau)
            mass_rows.extend(rows)
            for exp_index, (exp, matrix) in enumerate(matrices.items()):
                z = matrix.loc[:, CELLS].to_numpy() @ TRANSFORM.T
                loc, scale, df = niw_marginal(z, tau)
                rng = np.random.default_rng(SEED + 200 + 10 * index + exp_index)
                rows, components = count_samples(chunks_t(rng, loc, scale, df, n_mc), exp, "niw_posterior", tau)
                mass_rows.extend(rows)
                component_rows.extend(components)
            print(f"Completed proper NIW prior/posterior integration, tau={tau:.2f}.", flush=True)

        masses = pd.DataFrame(mass_rows)
        bayes_rows = []
        prior = masses[(masses.method == "niw_prior") & masses.pattern.isin(PATTERNS)]
        for row in masses[(masses.method == "niw_posterior") & masses.pattern.isin(PATTERNS)].itertuples():
            p = prior[(prior.prior_tau == row.prior_tau) & (prior.delta == row.delta) &
                      (prior.min_gap == row.min_gap) & (prior.pattern == row.pattern)].iloc[0]
            estimate = row.probability / p.probability if p.probability > 0 else np.nan
            lower = row.mc_lower_97_5 / p.mc_upper_97_5
            upper = row.mc_upper_97_5 / p.mc_lower_97_5 if p.mc_lower_97_5 > 0 else np.inf
            bayes_rows.append(dict(experiment=row.experiment, prior_tau=row.prior_tau, delta=row.delta,
                                   min_gap=row.min_gap, convention=row.convention, pattern=row.pattern,
                                   posterior_mass=row.probability, prior_mass=p.probability, BF_vs_unconstrained=estimate,
                                   mc_bf_lower_conservative=lower, mc_bf_upper_conservative=upper,
                                   prior_events=int(p.mc_successes), posterior_events=row.mc_successes,
                                   mc_zero_events=bool(row.mc_successes == 0 or p.mc_successes == 0)))
        bayes = pd.DataFrame(bayes_rows)
        comparisons = []
        for key, group in bayes.groupby(["experiment", "prior_tau", "delta", "min_gap", "convention"]):
            for a, b in itertools.combinations(PATTERNS, 2):
                ra, rb = group.set_index("pattern").loc[[a, b]].itertuples()
                value = ra.BF_vs_unconstrained / rb.BF_vs_unconstrained if rb.BF_vs_unconstrained > 0 else np.nan
                comparisons.append(dict(zip(["experiment", "prior_tau", "delta", "min_gap", "convention"], key)) |
                                   dict(numerator=a, denominator=b, BF_ratio=value,
                                        denominator_mc_zero=bool(rb.mc_zero_events)))

        # Influence check for the primary definition and prior, without rerunning trial-level models.
        loo = []
        primary_prior = prior[(prior.prior_tau == PRIMARY_TAU) & (prior.delta == .02) & (prior.min_gap == .02)].set_index("pattern")
        for exp_index, (exp, matrix) in enumerate(matrices.items()):
            values = matrix.loc[:, CELLS].to_numpy()
            for dropped, participant in enumerate(matrix.index):
                z = np.delete(values, dropped, axis=0) @ TRANSFORM.T
                loc, scale, df = niw_marginal(z, PRIMARY_TAU)
                rng = np.random.default_rng(SEED + 1000 + exp_index * 100 + dropped)
                cells = multivariate_t(rng, loc, scale, df, n_loo) @ TRANSFORM
                masks = region_masks(cells, .02, .02)
                for pattern, mask in masks.items():
                    probability = mask.mean()
                    loo.append(dict(experiment=exp, dropped_participant=str(participant), pattern=pattern,
                                    posterior_mass=probability, mc_draws=n_loo,
                                    BF_vs_unconstrained=probability / primary_prior.loc[pattern, "probability"]))
        print("Completed participant bootstrap and leave-one-participant-out checks.", flush=True)

    participant_rows.to_csv(output / "ordering_participant_slopes.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "ordering_cell_summary.csv", index=False)
    pd.DataFrame(pairwise).to_csv(output / "ordering_pairwise_contrasts.csv", index=False)
    masses.to_csv(output / "ordering_probability_sensitivity.csv", index=False)
    pd.DataFrame(component_rows).to_csv(output / "ordering_constraint_components.csv", index=False)
    bayes.to_csv(output / "ordering_bayes_factors.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "ordering_bayes_comparisons.csv", index=False)
    pd.DataFrame(loo).to_csv(output / "ordering_leave_one_out.csv", index=False)
    metadata = dict(
        seed=SEED, cells=CELLS, participant_counts={k: len(v) for k, v in matrices.items()},
        estimand="Unweighted participant-average controlled previous-duration slopes; same frozen valid trials as the four-cell analyses.",
        primary=dict(delta=.02, min_gap=.02, prior_tau=PRIMARY_TAU),
        settings=SETTINGS, prior_scales=PRIOR_SCALES, kappa0=1., nu0=5.,
        statistical_model="Participant contrast vectors iid MVN(mu,Sigma); proper NIW prior with mean 0, Psi0=kappa0*(nu0-3+1)*tau^2 I.",
        contrast_transform=TRANSFORM.tolist(),
        posterior_calculation="Exact conjugate multivariate-t marginal mean, including covariance uncertainty; Monte Carlo region integration.",
        bayes_factor="P(H|data,U) / P(H|U); prior conditioned on the SAME complete constraint region. No normalization to mechanism probabilities.",
        mc_draws=n_mc, prior_draws=n_prior, bootstrap_draws=n_boot, leave_one_out_draws=n_loo,
        mc_intervals="97.5% binomial intervals for numerical integration; not data confidence intervals. Ratio bounds have >=95% joint MC coverage.",
        approximate_equalities="Every within-group pair must differ by <= delta; not a chain of adjacent comparisons.",
        ordering="Every member of a higher group must exceed every member of a lower group by > min_gap.",
        overlap="direction_only regions may overlap; probability masses must not be normalized or summed as competing mechanism probabilities.",
        limitations=["Post hoc analysis; equality/gap bounds and proper priors are sensitivity assumptions, not preregistered choices.",
                     "Three behavioral predictions are not uniquely identified cognitive mechanisms.",
                     "The contextual-reset hypothesis was prompted by the observed transition pattern.",
                     "Different within-experiment compatibility or BF rankings do not establish a cross-experiment mechanism difference.",
                     "The NIW population model treats participant slope vectors as observations; it is not a trial-level hierarchical Kalman analysis."],
        input_sha256=input_hashes, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "ordering_analysis_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    primary = bayes[(bayes.prior_tau == PRIMARY_TAU) & (bayes.delta == .02) & (bayes.min_gap == .02)]
    print(primary[["experiment", "pattern", "posterior_mass", "prior_mass", "BF_vs_unconstrained"]].to_string(index=False), flush=True)
    return masses, bayes


if __name__ == "__main__":
    run_analysis()
