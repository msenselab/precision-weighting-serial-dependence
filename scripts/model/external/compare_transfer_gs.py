"""Participant-level held-out comparison on the Glasauer-Shi random-walk condition.

All model parameters are estimated from randomized trials.  The primary outputs are
participant-level random-walk RMSE and mean log predictive density.  CT and lag-1 ASD
absolute errors are mechanism-focused secondary outcomes.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import glasauer_shi as GS

sys.path.insert(0, str(SP.ROOT / "scripts" / "model" / "external"))
import fit_external as external  # noqa: E402


MODEL_NAMES = {
    "two-state": "Glasauer-Shi two-state",
    "R-B0": "3SKF v3 R-B0 (no bias state)",
    "R-B2": "3SKF v3 R-B2",
}
assert G.READOUT, "3SKF v3 only: set GKF_READOUT=1"
ABLATION, FOCAL = "R-B0", "R-B2"
METRICS = {
    "rmse": "lower",
    "mean_log_score": "higher",
    "ct_absolute_error": "lower",
    "asd_absolute_error": "lower",
}


def exact_sign_flip_p(differences):
    differences = np.asarray(differences, float)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(differences))))
    null_means = (signs * differences[None, :]).mean(axis=1)
    observed = abs(differences.mean())
    return float(np.mean(np.abs(null_means) >= observed - 1e-15))


def bootstrap_interval(differences, seed, repetitions=50_000):
    differences = np.asarray(differences, float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    means = differences[indices].mean(axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)))


def model_statistics(duration, response, valid):
    statistics = GS.statistics(duration, response, valid)
    return {
        "ct": -statistics["ct"],
        "asd": statistics["sd"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fits", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(args.fits)
    fits = fits[fits.model.isin((ABLATION, FOCAL))].copy()
    fits["sub"] = fits.unit.str.split("|").str[0].astype(int)
    fits["condition"] = fits.unit.str.split("|").str[1]
    fits = fits[fits.condition == "randomized"].set_index(["sub", "model"]).sort_index()
    if fits.index.has_duplicates:
        raise ValueError("fits must contain one randomized row per participant and model")

    _, loaded = external.load("gs")
    units = dict(loaded)
    participants = sorted({int(unit.split("|")[0]) for unit in units})
    paper_fits = []
    rows = []

    for participant in participants:
        train_duration, train_response, train_valid = units[f"{participant}|randomized"][0]
        duration, response, valid = units[f"{participant}|random walk"][0]
        observed = model_statistics(duration, response, valid)

        seed = zlib.crc32(f"gs-paper-transfer|{participant}".encode())
        paper = GS.fit(
            train_duration,
            train_response,
            train_valid,
            model="two-state",
            starts=1,
            seed=seed,
        )
        paper_fits.append({"sub": participant, **paper.__dict__})
        paper_mean = GS.observer(
            duration,
            paper.v_over_r,
            paper.q_over_r,
            paper.shift,
        )
        error = response[valid] - paper_mean[valid]
        paper_statistics = model_statistics(duration, paper_mean, valid)
        paper_log_score = -0.5 * (
            np.log(2 * np.pi * paper.sigma**2) + (error / paper.sigma) ** 2
        )
        rows.append(
            {
                "sub": participant,
                "model": MODEL_NAMES["two-state"],
                "rmse": np.sqrt(np.mean(error**2)),
                "mean_log_score": paper_log_score.mean(),
                "data_ct": observed["ct"],
                "predicted_ct": paper_statistics["ct"],
                "ct_absolute_error": abs(paper_statistics["ct"] - observed["ct"]),
                "data_asd": observed["asd"],
                "predicted_asd": paper_statistics["asd"],
                "asd_absolute_error": abs(paper_statistics["asd"] - observed["asd"]),
                "n": int(valid.sum()),
            }
        )

        tau = np.linspace(-1, 1, len(duration))
        for label in (ABLATION, FOCAL):
            fit = fits.loc[(participant, label)]
            theta = fit[G.NAMES].to_numpy(float)
            specification = external.MODELS[label]
            predicted = G.response_mean(
                theta,
                specification,
                np.log(duration),
                np.ones(len(duration)),
                tau,
            )
            error = response[valid] - predicted[valid]
            predicted_statistics = model_statistics(duration, predicted, valid)
            log_score = G.loglik(
                theta,
                specification,
                np.log(duration),
                np.ones(len(duration)),
                tau,
                np.where(valid, response, 0.0),
                valid,
            )[valid]
            rows.append(
                {
                    "sub": participant,
                    "model": MODEL_NAMES[label],
                    "rmse": np.sqrt(np.mean(error**2)),
                    "mean_log_score": log_score.mean(),
                    "data_ct": observed["ct"],
                    "predicted_ct": predicted_statistics["ct"],
                    "ct_absolute_error": abs(predicted_statistics["ct"] - observed["ct"]),
                    "data_asd": observed["asd"],
                    "predicted_asd": predicted_statistics["asd"],
                    "asd_absolute_error": abs(predicted_statistics["asd"] - observed["asd"]),
                    "n": int(valid.sum()),
                }
            )

    scores = pd.DataFrame(rows).sort_values(["sub", "model"])
    scores.to_csv(output / "participant_scores.csv", index=False)
    pd.DataFrame(paper_fits).to_csv(output / "paper_two_state_fits.csv", index=False)

    wide = scores.pivot(index="sub", columns="model", values=list(METRICS))
    summaries = []
    focal = MODEL_NAMES[FOCAL]
    for comparator in (MODEL_NAMES["two-state"], MODEL_NAMES[ABLATION]):
        for metric, preferred in METRICS.items():
            difference = wide[(metric, focal)] - wide[(metric, comparator)]
            lower, upper = bootstrap_interval(
                difference,
                zlib.crc32(f"gs-bootstrap|{comparator}|{metric}".encode()),
            )
            favours = difference > 0 if preferred == "higher" else difference < 0
            summaries.append(
                {
                    "focal_model": focal,
                    "comparator": comparator,
                    "metric": metric,
                    "preferred": preferred,
                    "mean_focal_minus_comparator": difference.mean(),
                    "bootstrap_95_lo": lower,
                    "bootstrap_95_hi": upper,
                    "exact_sign_flip_p": exact_sign_flip_p(difference),
                    "participants_favouring_focal": int(favours.sum()),
                    "participants": len(difference),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "pairwise_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
