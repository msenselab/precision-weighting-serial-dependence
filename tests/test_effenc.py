"""Fast efficient encoding (GKF_EFFENC): the measurement variance follows the width of the observer's predictive
prior, R_i = R_c(i) min(H P-_i H' sigma_s^2 / V0, 1)^gamma, and the inference stays the standard Kalman filter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_gkf import participant  # noqa: E402

from sdu.model import engine as G  # noqa: E402

# Efficient encoding scales the actual measurement noise; under noise-free encoding (GKF_NOISELESS, 3SKF v2) there is
# none, and the reference recursion below uses the engine's default (non-diagonal) Q.
pytestmark = pytest.mark.skipif(G.NOISELESS or G.QDIAG, reason="efficient encoding needs noisy measurements and the "
                                "default Q (not the 3SKF v2 configuration)")

def random_theta(spec, rng, sig=0.08):
    """Free parameters inside their bounds; sigma_s on the scale of the active bounds (v3: natural, v1/v2: log)."""
    lo, hi = spec.bounds()
    th = np.where(lo == hi, lo, lo + (hi - lo) * rng.uniform(0.25, 0.75, len(lo)))
    th[G.IX["lsig"]] = sig if G.BOUNDS == "v3" else np.log(sig)
    # process noise in the range of the v3 fits (predictive prior ~5-10% of the design variance, effenc_ratio check)
    q1, q2, q3 = 0.14, 0.016, 0.05                             # in units of sigma_s^2
    v3 = G.BOUNDS == "v3"
    for n, q in (("lq1", q1), ("lq2", q2), ("lq3", q3)):
        if lo[G.IX[n]] < hi[G.IX[n]]:
            th[G.IX[n]] = np.sqrt(q) * sig if v3 else (np.log(q) if n != "lq3" else np.log(q) / 2)
    for n in ("aq1", "aq2", "aq3"):                            # Low/High process-noise ratio 2
        if lo[G.IX[n]] < hi[G.IX[n]]:
            th[G.IX[n]] = 2.0 if v3 else np.log(2.0)
    return th

SPECS = [G.Spec(4, "T1n", 0, 3), G.Spec(4, "T1nS", 8, 3), G.Spec(5, "T1n", 8, 2), G.Spec(1, "T0", 0, 4)]


@pytest.fixture
def gamma(monkeypatch):
    monkeypatch.setattr(G, "EFF_GAMMA", 0.5)
    return 0.5


def textbook_r(th, spec, coh, gamma):
    """Covariance-only Kalman recursion on the absolute states s = [x, m, b] (written independently of engine.observer)
    with the efficient-encoding measurement variance. Returns R_i."""
    p = G.unpack(th)
    phL, phH = G.weights(th, G.T_CODES[spec.t])
    observes_bias, fb, hb = G.b_flags(spec.b)[:3]
    lam, q3 = (p["lam"], p["q3"]) if observes_bias else (0.0, 0.0)
    gam = p["sw"]
    H = np.array([1.0, 0.0, float(hb)])
    sw = G.switch_flags(coh)
    P = np.zeros((3, 3))
    if G.INIT == "gs":
        P[:2, :2] = G.MEM0
    else:
        v = G.V0 / p["sig"] ** 2
        P[:2, :2] = [[(1 + G.DEV0) * v, v], [v, v]]
    P[2, 2] = q3 / (1 - lam ** 2) if observes_bias else 0.0
    out = np.empty(len(coh))
    for i, c in enumerate(coh):
        q1, q2 = p["q1"] * np.exp(p["aq1"] * (1 - c)), p["q2"] * np.exp(p["aq2"] * (1 - c))
        Q = np.array([[q1 + q2, q2, 0], [q2, q2, 0], [0, 0, q3 * np.exp(p["aq3"] * (1 - c))]])
        if sw[i] and spec.s == 8:
            Q = Q / (1 - gam)
        phi = 0.0 if sw[i] and G.T_CODES[spec.t] in G.RESET_CODES else phL if c < 0.5 else phH
        F = np.array([[phi, 1 - phi, float(fb)], [0, 1, 0], [0, 0, lam]])
        P = F @ P @ F.T + Q
        R = (p["rlow"] if c < 0.5 else 1.0) * min(H @ P @ H * p["sig"] ** 2 / G.V0, 1.0) ** gamma
        K = P @ H / (H @ P @ H + R)
        P = P - np.outer(K, H @ P)
        out[i] = R
    return out


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_r_follows_the_predictive_prior(spec, gamma):
    _, a = participant(exp=1, k=2)
    th = random_theta(spec, np.random.default_rng(4))
    th[G.IX["lsw"]] = 0.3
    R = G.observer(th, spec, np.log(a["T"]), a["coh"])[5]
    assert np.allclose(R, textbook_r(th, spec, a["coh"], gamma), rtol=1e-9)
    Rc = np.where(a["coh"] < 0.5, G.unpack(th)["rlow"], 1.0)
    assert np.all(R <= Rc + 1e-12) and np.mean(R < Rc - 1e-9) > 0.5      # finer encoding on most trials


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_r_ignores_the_measurements(spec, gamma):
    """R_i depends on the cue sequence only, so the observer stays linear-Gaussian (standard Kalman inference)."""
    _, a = participant(exp=1, k=1)
    th = random_theta(spec, np.random.default_rng(5))
    z1 = G.measurements(th, a["T"], a["coh"], np.random.default_rng(1), spec)
    z2 = G.measurements(th, a["T"], a["coh"], np.random.default_rng(2), spec)
    o1, o2 = G.observer(th, spec, z1, a["coh"]), G.observer(th, spec, z2, a["coh"])
    for j in (1, 2, 3, 4, 5):                                             # c, g, A, K, R
        assert np.allclose(o1[j], o2[j])


def test_gamma_zero_is_the_fixed_encoding(monkeypatch):
    _, a = participant()
    spec = SPECS[0]
    th = random_theta(spec, np.random.default_rng(6))
    args = (th, spec, np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"])
    monkeypatch.setattr(G, "EFF_GAMMA", 0.0)
    l0 = G.loglik(*args).sum()
    monkeypatch.setattr(G, "EFF_GAMMA", 1e-12)
    assert np.isclose(l0, G.loglik(*args).sum(), atol=1e-6)
    monkeypatch.setattr(G, "EFF_GAMMA", 0.5)
    assert not np.isclose(l0, G.loglik(*args).sum())


def test_generated_noise_matches_r(gamma):
    spec = G.Spec(4, "T1n", 0, 1)
    th = random_theta(spec, np.random.default_rng(3), sig=0.15)
    n = 60
    coh = np.where(np.arange(n) % 3 == 0, 0.3, 0.7)
    R = G.observer(th, spec, np.zeros(n), coh)[5]
    z = np.stack([G.measurements(th, np.full(n, 1.2), coh, np.random.default_rng(s), spec) for s in range(4000)])
    var = (z - np.log(1.2)).var(axis=0)
    assert np.allclose(var / (G.unpack(th)["sig"] ** 2 * R), 1.0, atol=0.12)


def test_jax_matches_numpy(gamma):
    pytest.importorskip("jax")
    J = G.jax_funcs()
    _, a = participant(exp=1, k=3)
    rng = np.random.default_rng(7)
    for spec in SPECS:
        th = random_theta(spec, rng)
        ll_np = G.loglik(th, spec, np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"]).sum()
        ll_jx = -float(J["nll"](th, spec.vec(), np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"]))
        assert np.isclose(ll_np, ll_jx, atol=1e-7), spec.name
