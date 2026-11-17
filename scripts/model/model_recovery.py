"""Model recovery of the switch rule, S0 versus S3 (Appendix B.6, Figure B5A).

Each generating model (C3-S0-B2 and C3-S3-B2, at every Experiment 2 participant's fitted parameters) produces
N_REP response data sets on the participant's own sequence and validity mask. Every data set is fitted with both
candidates (13 starts: the default point, 11 random points, the zero-retention point), and each fitted candidate is
simulated N_SIM times for the focal behavioural contrast of Experiment 2 (LL minus the other three transitions).
model_recovery_summary.py combines the 22 participants of each repetition into a cohort.

Usage: python scripts/model/model_recovery.py --task=<i> --ntasks=<n> [--reps=100] [--starts=12] [--nsim=10]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

from sdu import paths as SP
from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.model import stats as S

sys.path.insert(0, str(SP.ROOT / "scripts" / "model"))
from recovery_utils import SEED_CODE, fit_theta, fitted_theta, seed  # noqa: E402

N_REP = int(FF.option("reps", "100"))
N_STARTS = int(FF.option("starts", "12"))
N_SIM = int(FF.option("nsim", "10"))
EXP = 2
FOCAL = "LL_other_three"
CANDIDATES = {"S0": "C3-S0-B2", "S3": "C3-S3-B2"}
# text keys of the random seeds (see recovery_utils.SEED_CODE)
KEYS = {"simulation": "recv3-model|switch|{code}|{sub}|{generator}|{rep}",
        "fit": "recv3-fit|switch|{code}|{sub}|{generator}|{candidate}|{rep}",
        "check": "recv3-check|switch|{code}|{sub}|{generator}|{candidate}|{rep}"}


def rng(kind, **fields):
    return np.random.default_rng(seed(KEYS[kind].format(code=SEED_CODE[EXP], **fields)))


def run_unit(sub, functions, participant_data, fits):
    T, coherence, _, valid = participant_data
    rows = []
    for generator, label in CANDIDATES.items():
        generating_theta = fitted_theta(fits, EXP, sub, label)
        for rep in range(N_REP):
            generated = G.simulate(generating_theta, G.Model.parse(label), T, coherence, valid,
                                   rng("simulation", sub=sub, generator=generator, rep=rep), return_mean=False)[0]
            observed_focal = S.participant_stats_arrays(EXP, sub, generated)[FOCAL]
            arrays = (T, coherence, np.nan_to_num(generated), valid)
            for candidate, candidate_label in CANDIDATES.items():
                model = G.Model.parse(candidate_label)
                keys = dict(sub=sub, generator=generator, candidate=candidate, rep=rep)
                best, n_best, n_finite = fit_theta(functions, model, arrays, rng("fit", **keys), N_STARTS)
                check = rng("check", **keys)
                simulated_focal = [
                    S.participant_stats_arrays(EXP, sub, G.simulate(best.x, model, T, coherence, valid, check,
                                                                    return_mean=False)[0])[FOCAL]
                    for _ in range(N_SIM)]
                rows.append(dict(analysis="switch", exp=EXP, Sub=sub, generator=generator, candidate=candidate,
                                 rep=rep, focal=FOCAL, k=model.k(), LL=-best.fun, AIC=2 * model.k() + 2 * best.fun,
                                 n_at_best=n_best, n_starts=n_finite, n_sim=N_SIM, obs_focal=observed_focal,
                                 sim_focal=np.mean(simulated_focal), **dict(zip(G.NAMES, best.x)),
                                 **{f"true_{n}": v for n, v in zip(G.NAMES, generating_theta)}))
    return pd.DataFrame(rows)


def validate_existing(path, sub):
    frame = pd.read_csv(path)
    validate_frame(frame, sub, path)


def validate_frame(frame, sub, source):
    if len(frame) != 2 * 2 * N_REP or set(frame.generator) != set(CANDIDATES) \
            or set(frame.candidate) != set(CANDIDATES):
        raise ValueError(f"stale recovery data {source}")
    if frame.rep.nunique() != N_REP or not frame.n_sim.eq(N_SIM).all() or not frame.n_starts.eq(N_STARTS + 1).all():
        raise ValueError(f"stale recovery configuration in {source}")
    if not frame.exp.eq(EXP).all() or not frame.Sub.eq(sub).all():
        raise ValueError(f"wrong recovery unit in {source}")


def main():
    task = int(FF.option("task", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    tasks = int(FF.option("ntasks", "1"))
    directory = SP.MODEL_RECOVERY
    directory.mkdir(parents=True, exist_ok=True)
    functions = G.jax_funcs()
    data = FF.load(EXP)
    fits = FF.read_fits()
    consolidated_path = directory / "participant_fits.csv"
    consolidated = pd.read_csv(consolidated_path) if consolidated_path.exists() else pd.DataFrame()
    for sub in sorted(data)[task::tasks]:
        stored = consolidated[consolidated.Sub == sub] if not consolidated.empty else consolidated
        if not stored.empty:
            validate_frame(stored, sub, consolidated_path)
            continue
        path = directory / f"switch_exp{EXP}_{sub}.csv"
        if path.exists():
            validate_existing(path, sub)
            continue
        started = time.time()
        frame = run_unit(sub, functions, data[sub], fits)
        frame.to_csv(path.with_suffix(".tmp"), index=False)
        os.replace(path.with_suffix(".tmp"), path)
        print(f"done Sub={sub}: {len(frame)} fits, {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
