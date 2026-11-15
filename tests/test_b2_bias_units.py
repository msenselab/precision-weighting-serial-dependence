"""The reported B2 response-bias drift (sdu.reports.plots.b2_bias_sd) must be the variance the simulator uses.

Runs in a fresh process with the v3 readout configuration, because the engine reads its configuration at import.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import numpy as np
from sdu.model import engine as G
from sdu.reports import plots as P

spec = G.Spec(3, "T1s", 0, 3)                      # C3-S0-B2 of the v3 space
assert G.b_flags(spec.b)[4] and G.b_flags(spec.b)[5], "B2: latent bias in seconds in the response"
lo, hi = spec.bounds()
th = np.where(lo == hi, lo, (lo + hi) / 2)
th[G.IX["lam"]] = 0.0                              # independent bias draws
th[G.IX["lq3"]] = 0.08                             # bias SD 0.08 s at the reference coherence
th[G.IX["aq3"]] = 2.0                              # Low/High variance ratio 2
th[G.IX["lsde"]] = lo[G.IX["lsde"]]                # smallest response noise
th[G.IX["beta"]] = th[G.IX["wuH"]] = 0.0
p = G.unpack(th)
n = 40000
rng = np.random.default_rng(1)
T = rng.choice([0.8, 1.0, 1.2, 1.4, 1.6], n)
for c in (0.3, 0.7):
    coh = np.full(n, c)
    y = G.simulate(th, spec, T, coh, np.zeros(n), np.ones(n, bool), np.random.default_rng(2), return_mean=False)[0]
    mean = G.response_mean(th, spec, np.log(T), coh, np.zeros(n))
    noise = G.noise_sd(p, T, coh)
    simulated_bias_var = np.var(y - mean) - np.mean(noise ** 2)
    reported = P.b2_bias_sd(p, c) ** 2
    assert abs(simulated_bias_var / reported - 1) < 0.05, (c, simulated_bias_var, reported)
    assert abs(p["q3"] / reported - 1) > 0.5, "the observer-normalised q3 must differ from the B2 variance"
print("ok")
"""


def test_b2_bias_sd_matches_simulator():
    env = dict(os.environ, GKF_SCALE="lin", GKF_CORE="reduced", GKF_BOUNDS="v3", GKF_INIT="gs", GKF_NOISELESS="1",
               GKF_QDIAG="1", GKF_READOUT="1", PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
