"""Persistence at the readout (GKF_READOUT=1): the memory filter runs with phi = 0 and the T-level persistence acts
only on the report, u_i = x+_i + (1 - g_i) phi_i f+_{i-1}, without being written back to the memory."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_effenc import random_theta  # noqa: E402
from test_gkf import participant  # noqa: E402

from sdu.model import engine as G  # noqa: E402

# T1n (phi^L, phi^H = 0), T1s (phi^L, phi^H free), with S0, S1 (code 3), S2 (gain cut), the reset T, B codes 1-3
SPECS = [G.Spec(3, "T1n", 0, 3), G.Spec(7, "T1s", 0, 3), G.Spec(12, "T1nS", 0, 3), G.Spec(3, "T1n", 3, 3),
         G.Spec(1, "T1s", G.GAIN_CUT, 1), G.Spec(4, "T1n", 0, 2), G.Spec(15, "T1sS", 3, 2)]


@pytest.fixture
def readout(monkeypatch):
    monkeypatch.setattr(G, "READOUT", True)


def theta(spec, seed=3):
    th = random_theta(spec, np.random.default_rng(seed))
    for n, v in (("wuL", 0.6), ("beta", 0.6), ("wuH", 0.3)):
        lo, hi = spec.bounds()
        if lo[G.IX[n]] < hi[G.IX[n]]:
            th[G.IX[n]] = v if G.BOUNDS == "v3" else np.log(v / (1 - v))
    if G.BOUNDS == "v3" and "lsw" in spec.free():
        th[G.IX["lsw"]] = 0.0
    return th


def reference(th, spec, z, coh):
    """Two-state memory filter in (m, f) coordinates with phi = 0, written independently of the engine (bias not
    in the observer: B codes 1 and 3, S0), and the readout u_i = x+_i + (1 - g_i) phi(c_i) f+_{i-1}."""
    p = G.unpack(th)
    phL, phH = G.weights(th, G.T_CODES[spec.t])
    assert not G.b_flags(spec.b)[0] and spec.s == 0
    reset = spec.t.endswith("S")
    sw = G.switch_flags(coh)
    H = np.array([1.0, 1.0])
    x = np.array([G.MU0, 0.0])
    P = np.diag([G.MEM0 if G.INIT == "gs" else G.V0 / p["sig"] ** 2,
                 0.0 if G.INIT == "gs" else G.DEV0 * G.V0 / p["sig"] ** 2])
    F = np.diag([1.0, 0.0])
    u, gains = np.empty(len(z)), np.empty(len(z))
    for i, c in enumerate(coh):
        q2, q1 = p["q2"] * np.exp(p["aq2"] * (1 - c)), p["q1"] * np.exp(p["aq1"] * (1 - c))
        Q = np.array([[q2, -q2], [-q2, q1 + q2]]) if G.QDIAG else np.diag([q2, q1])
        R = p["rlow"] if c < 0.5 else 1.0
        f_prev = x[1]
        x, P = F @ x, F @ P @ F.T + Q
        K = P @ H / (H @ P @ H + R)
        x = x + K * (z[i] - H @ x)
        P = P - np.outer(K, H @ P)
        phi = 0.0 if (reset and sw[i]) else (phL if c < 0.5 else phH)
        gains[i] = H @ K
        u[i] = H @ x + (1 - gains[i]) * phi * f_prev
    return u, gains


def z_of(a):
    return a["T"] if G.LIN else np.log(a["T"])


@pytest.mark.parametrize("spec", [s for s in SPECS if s.s == 0 and s.b != 2], ids=lambda s: s.name)
def test_matches_the_reference(readout, spec):
    _, a = participant(exp=2, k=1)
    th = theta(spec)
    u, c, g, A, K, R = G.observer(th, spec, z_of(a), a["coh"])
    u0, g0 = reference(th, spec, z_of(a), a["coh"])
    assert np.allclose(g, g0, atol=1e-10) and np.allclose(u, u0, atol=1e-10)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_memory_does_not_depend_on_phi(readout, spec):
    """Gains, state maps and the carried state equal those of the same model with phi = 0 (T level Tn)."""
    _, a = participant(exp=1, k=0)
    th = theta(spec)
    spec0 = G.Spec(spec.c, "Tn", spec.s, spec.b)
    out, out0 = G.observer(th, spec, z_of(a), a["coh"]), G.observer(th, spec0, z_of(a), a["coh"])
    for j in (2, 3, 4, 5):                                       # g, A, K, R
        assert np.allclose(out[j], out0[j], atol=1e-12)
    assert not np.allclose(out[0], out0[0])                      # the readout differs


def test_zero_persistence_is_the_t0_model(monkeypatch):
    """With phi = 0 the readout version and the standard engine give identical reports and likelihoods."""
    _, a = participant(exp=2, k=2)
    spec = G.Spec(7, "Tn", 3, 3)
    th = theta(spec)
    args = (np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"])
    monkeypatch.setattr(G, "READOUT", False)
    ll0 = G.loglik(th, spec, *args)
    monkeypatch.setattr(G, "READOUT", True)
    assert np.allclose(G.loglik(th, spec, *args), ll0, atol=1e-12)


def test_differs_from_persistence_in_the_prediction(monkeypatch):
    _, a = participant(exp=2, k=2)
    spec = SPECS[0]
    th = theta(spec)
    monkeypatch.setattr(G, "READOUT", False)
    u_pred = G.observer(th, spec, z_of(a), a["coh"])[0]
    monkeypatch.setattr(G, "READOUT", True)
    u_read = G.observer(th, spec, z_of(a), a["coh"])[0]
    assert not np.allclose(u_pred, u_read)


def test_jax_matches_numpy(readout):
    pytest.importorskip("jax")
    J = G.jax_funcs()
    _, a = participant(exp=1, k=3)
    for spec in SPECS:
        th = theta(spec, seed=5)
        args = (np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"])
        ll_np = G.loglik(th, spec, *args).sum()
        ll_jx = -float(J["nll"](th, spec.vec(), *args))
        assert np.isclose(ll_np, ll_jx, atol=1e-7), spec.name
