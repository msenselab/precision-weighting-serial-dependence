"""The SDU three-state Kalman observer on external magnitude-reproduction data, compared with the two-stage model of
Petzschner & Glasauer (2011) as implemented in kmodel1pv.m (Glasauer & Shi, 2021).

Datasets
  gs     Glasauer & Shi (2021) duration reproduction: one 400-trial sequence per participant x sequence type
         (RandLevel 1 random walk, 2 randomized), as in data_fit_paper.m. Units s, scale 1.
  pg_d   Petzschner & Glasauer (2011) distance estimation; pg_a angular estimation: per participant three
         180-trial sequences (ranges 1-3), the observer restarting at each, one parameter set shared by the three.
         Stimuli and responses divided by s = (overall stimulus mean) / 1.2, so the bounds set in seconds apply.
Models (engine Spec; coh = 1 on every trial, so the coherence terms ad0 and kappa are fixed; c = 0 only because
there is no coherence in these designs)
  kmodel  kmodel1pv: q/r >= 0 (least squares, as lsqnonlin from p0 = 1, plus a log grid of starts),
          Gaussian response noise sigma (MLE), k = 2; a = 10, off = 1 as in the published code.
Log-likelihoods are reported in the data's original units (SDU: LL_scaled - n log s).
  3SKF v3 (requires GKF_READOUT=1): the v3 space C1-C15 x S0-S3 x B0-B2 reduces on these data to R-B0/B1/B2. Every
  trial has coh = 1: the C ratios multiply by exp(alpha (1 - c)) = 1 and R_L is never used; there are no coherence
  switches, so S1-S3 equal S0; only phi^H acts. phi = 0 is the nested no-persistence case.
Usage: python fit_external.py <dataset> <out.csv>
"""
import multiprocessing as mp
import os
import sys
import zlib

import numpy as np
import pandas as pd

from sdu.model import engine as G

from sdu import paths as _SP
ROOT = str(_SP.DATA)                      # data/external/glasauer_shi_2021 (Petzschner & Glasauer: not in this repo)
assert G.READOUT, "3SKF v3 only: set GKF_READOUT=1"
# The v3 space on data without coherence: C and S have no effect, so it reduces to B0-B2 with one readout phi.
MODELS = {f"R-B{b}": G.Spec(0, "T0", 0, b + 1) for b in range(3)}
FIX = {"ad0": 0.0, "kappa": 1.0 if G.BOUNDS == "v3" else 0.0}
N_STARTS = int(os.environ.get("EXT_STARTS", "5"))


def load(ds):
    """-> scale s, list of (unit id, [(T, y, mask) per sequence]) in original units."""
    units = []
    if ds == "gs":
        d = pd.read_csv(f"{ROOT}/external/glasauer_shi_2021/Glasauer_Shi_2021b_duration_reproduction.csv").dropna(subset=["RandLevel"])
        # The CSV's column labels are swapped: "condition" (1-14) is the participant and "NSub" (1-20) the
        # 40-trial block; NT (1-800) is the trial order within a participant (checked 2026-11-09).
        for (sub, rl), g in d.groupby(["condition", "RandLevel"]):
            g = g.sort_values("NT")
            y = g.repDur.to_numpy(float)
            units.append((f"{int(sub)}|{'random walk' if rl == 1 else 'randomized'}",
                          [(g.dur.to_numpy(float), y, (g.valid.to_numpy() == 1) & np.isfinite(y))]))
        return 1.0, units
    kind, st, rs = {"pg_d": ("distance", "stimulus", "response"),
                    "pg_a": ("angular", "stimulus_deg", "response_deg")}[ds]
    d = pd.read_csv(f"{ROOT}/petzschner_glasauer_2020/Petzschner_Glasauer_2020_{kind}_estimation.csv")
    s = d[st].mean() / 1.2
    for sub, g in d.groupby("subject"):
        seqs = []
        for _, h in g.groupby("range"):
            h = h.sort_values("trial")
            y = h[rs].to_numpy(float)
            seqs.append((h[st].to_numpy(float), y, np.isfinite(y) & (y > 0)))
        units.append((str(int(sub)), seqs))
    return s, units


def kmodel(qr, T, a=10.0, off=1.0):
    """kmodel1pv.m: two-stage model (r = 1, q = qr r); returns the deterministic response."""
    r, q, p = 1.0, qr, 1.0
    resp = np.empty(len(T))
    x = 0.0
    for i, d in enumerate(T):
        z = np.log(a * d + off)
        if i == 0:
            x = z
        km = (p + q) / (p + q + r)
        p = km * r
        x = x + km * (z - x)
        w1 = 1 / r / (1 / p + 1 / r)
        resp[i] = (np.exp((1 - w1) * x + w1 * z) - off) / a
    return resp


def fit_kmodel(seqs):
    from scipy.optimize import least_squares

    def res(v):
        return np.concatenate([(y - kmodel(v[0], T))[m] for T, y, m in seqs])
    best = None
    for p0 in [1.0] + list(np.logspace(-3, 2, 6)):                 # lsqnonlin's p0 = 1, plus a grid
        r = least_squares(res, [p0], bounds=([0.0], [np.inf]))
        if best is None or r.cost < best.cost:
            best = r
    e = res(best.x)
    n = len(e)
    s2 = np.mean(e ** 2)
    return dict(model="kmodel", k=2, LL=-0.5 * n * (np.log(2 * np.pi * s2) + 1), n=n, qr=best.x[0], sigma=np.sqrt(s2))


def run(job):
    from scipy.optimize import minimize
    ds, uid, seqs, s = job
    J = G.jax_funcs()
    jnp = J["jnp"]
    arrs = []
    for T, y, m in seqs:
        tn = np.arange(len(T), dtype=float)
        arrs.append([jnp.asarray(v) for v in (np.log(T / s), np.ones(len(T)), 2 * tn / tn.max() - 1,
                                               np.where(m, y / s, 0.0), m)])
    n = int(sum(m.sum() for _, _, m in seqs))
    rng = np.random.default_rng(zlib.crc32(f"ext|{ds}|{uid}".encode()))
    rows = []
    for name, spec in MODELS.items():
        lo, hi = spec.bounds()
        for k_, v in FIX.items():
            lo[G.IX[k_]] = hi[G.IX[k_]] = v
        code = jnp.asarray(spec.vec())

        def f(th):
            tot, grad = 0.0, 0.0
            for a in arrs:
                v, g = J["vg"](jnp.asarray(th), code, *a)
                tot, grad = tot + float(v), grad + np.asarray(g, float)
            return tot, grad
        starts = [np.clip(G.X0, lo, hi)] + [np.where(lo == hi, lo, lo + (hi - lo) * rng.uniform(.15, .85, len(lo)))
                                            for _ in range(N_STARTS - 1)]
        if name.startswith("R-"):                     # v3 nests phi = 0: make sure that point is reached
            x00 = np.clip(G.X0, lo, hi)
            x00[G.IX["wuH"]] = 0.0
            starts.append(x00)
        best = None
        for x0 in starts:
            r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(maxiter=4000))
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        at_bound = [G.NAMES[i] for i in np.where((lo < hi) & ((np.abs(best.x - lo) < 1e-6) | (np.abs(best.x - hi) < 1e-6)))[0]]
        rows.append(dict(dataset=ds, unit=uid, model=name, k=int((lo < hi).sum()), LL=-best.fun - n * np.log(s), n=n,
                         at_bound=";".join(at_bound), **dict(zip(G.NAMES, best.x))))
    rows.append(dict(dataset=ds, unit=uid, **fit_kmodel(seqs)))
    return rows


def main():
    ds, out = sys.argv[1], sys.argv[2]
    s, units = load(ds)
    print(ds, "scale", round(s, 3), "units", len(units), flush=True)
    with mp.get_context("spawn").Pool(8) as pool:
        rows = sum(pool.map(run, [(ds, uid, seqs, s) for uid, seqs in units]), [])
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, len(rows))


if __name__ == "__main__":
    main()
