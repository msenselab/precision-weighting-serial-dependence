"""Glasauer & Shi's two-state model fitted to the randomized sequences (Appendix C), as in the authors' code:
starting point p0 = (1, 1, 0) and one least-squares run. Writes results/model/glasauer_shi/two_state_fits.csv.

Usage: python scripts/model/glasauer_shi/fit_two_state.py
"""
import pandas as pd

from sdu import paths as SP
from sdu.model import fitting as FF
from sdu.model import glasauer_shi as GS


def main():
    units = GS.load()
    fits = []
    for participant in GS.participants(units):
        duration, response, valid = units[f"{participant}|randomized"]
        result = GS.fit(duration, response, valid, model="two-state", starts=1,
                        seed=FF.seed("gs-two-state-fit", participant, "two-state"))
        fits.append({"sub": participant, **result.__dict__})
    out = SP.GLASAUER_SHI / "two_state_fits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fits).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
