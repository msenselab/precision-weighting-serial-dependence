"""Fit the paper-faithful Glasauer-Shi models to randomized trials only.

The fitted parameters are then frozen and evaluated on both the randomized and random-walk
sequences.  Random-walk responses never enter parameter estimation.
"""

from __future__ import annotations

import sys
import zlib

import pandas as pd

from sdu import paths as SP
from sdu.model import glasauer_shi as GS

sys.path.insert(0, str(SP.ROOT / "scripts" / "model" / "external"))
import fit_external as external  # noqa: E402

MODELS = ("two-state",)


def main():
    output = SP.RESULTS / "model" / "minimal_qdiag" / "glasauer_shi"
    output.mkdir(parents=True, exist_ok=True)
    _, loaded = external.load("gs")
    units = dict(loaded)
    participants = sorted({unit.split("|")[0] for unit in units}, key=int)

    fits = []
    for participant in participants:
        duration, response, valid = units[f"{participant}|randomized"][0]
        for model in MODELS:
            seed = zlib.crc32(f"gs-two-state-fit|{participant}|{model}".encode())
            # Match the supplied Glasauer-Shi fitting code: p0 = (1, 1, 0), one
            # least-squares run.  The seed is retained in the output workflow only
            # for stable file provenance; it is not used when starts == 1.
            result = GS.fit(duration, response, valid, model=model, starts=1, seed=seed)
            fits.append({"sub": participant, **result.__dict__})
            print("fit", participant, model, flush=True)
    fits = pd.DataFrame(fits)
    fits.to_csv(output / "two_state_fits.csv", index=False)

    rows = []
    for model in MODELS:
        model_fits = fits[fits.model == model].set_index("sub")
        for target in ("randomized", "random walk"):
            observed = []
            predicted = []
            for participant in participants:
                duration, response, valid = units[f"{participant}|{target}"][0]
                fit = model_fits.loc[participant]
                mean = GS.observer(duration, fit.v_over_r, fit.q_over_r, fit["shift"])
                observed.append({"sub": participant, **GS.statistics(duration, response, valid)})
                predicted.append({"sub": participant, **GS.statistics(duration, mean, valid)})

            observed_frame = pd.DataFrame(observed).set_index("sub")
            predicted_frame = pd.DataFrame(predicted).set_index("sub")
            for statistic in observed_frame.columns:
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "stat": statistic,
                        "data": observed_frame[statistic].mean(),
                        "data_sem": observed_frame[statistic].sem(),
                        "mean": predicted_frame[statistic].mean(),
                        "model_sem": predicted_frame[statistic].sem(),
                        "parameters_from": "randomized",
                    }
                )
    pd.DataFrame(rows).to_csv(output / "two_state_cross_prediction.csv", index=False)
    print("wrote", output / "two_state_fits.csv")
    print("wrote", output / "two_state_cross_prediction.csv")


if __name__ == "__main__":
    main()
