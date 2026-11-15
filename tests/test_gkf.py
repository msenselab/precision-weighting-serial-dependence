"""Tests of the standard three-state observer (sdu.model.engine): the readout is the posterior of a
textbook Kalman filter (Appendix C.1, E3-E7), JAX = NumPy over the space, degeneracies, embeddings and the tilt."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdu.model import engine as G  # noqa: E402

from sdu import paths as SP
DATA = SP.MODEL_INPUT / "input_full_sequence.csv"
SPECS = [G.Spec(c, t, s, b) for c, t, s, b in [
    (0, "Tn", 0, 3), (4, "T0", 3, 1), (1, "T1n", 8, 2), (15, "T1s", 9, 3), (5, "T0S", 0, 2), (7, "T1nS", 9, 3),
    (12, "T1sS", 3, 2), (8, "T0", 8, 3), (2, "T1s", 0, 1), (13, "T1n", 3, 3), (6, "Tn", 9, 2), (0, "T1nS", 8, 1),
    (0, "T0W", 0, 3), (4, "T1nW", 0, 2), (1, "T1sW", 0, 1), (0, "TnW", 0, 3)]]


def participant(exp=2, k=0):
    d = pd.read_csv(DATA)
    d = d[d.exp == exp].sort_values(["Sub", "trial_num"])
    s = sorted(d.Sub.unique())[k]
    g = d[d.Sub == s]
    tn = g.trial_num.to_numpy(float)
    return s, dict(T=g.stimulus.to_numpy(float), coh=g.coherence.to_numpy(float),
                   tau=2 * (tn - tn.min()) / (tn.max() - tn.min()) - 1, y=g.response.to_numpy(float),
                   mask=g.response.notna().to_numpy(), trial=tn)


def random_theta(spec, rng):
    lo, hi = spec.bounds()
    th = np.where(lo == hi, lo, lo + (hi - lo) * rng.uniform(0.25, 0.75, len(lo)))
    th[G.IX["lsig"]] = np.log(0.08)
    return th


def textbook(th, spec, z, coh):
    """Kalman filter on the absolute states s = [x, m, b] of (E3), written from (E1)-(E7): one prediction, the
    optimal gain, the report H s with H = [1, 0, 0]."""
    p = G.unpack(th)
    phL, phH = G.weights(th, G.T_CODES[spec.t])
    observes_bias, fb, hb = G.b_flags(spec.b)[:3]
    lam, q3 = (p["lam"], p["q3"]) if observes_bias else (0.0, 0.0)
    gam = p["sw"]
    H = np.array([1.0, 0.0, float(hb)])
    Hx = np.array([1.0, 0.0, 0.0])
    sw = G.switch_flags(coh)
    phi1 = phL if coh[0] < 0.5 else phH
    s = np.array([G.MU0, G.MU0, 0.0])
    P = np.zeros((3, 3))
    if G.BOUNDS == "v2":
        v = G.V0 / p["sig"] ** 2
        P[:2, :2] = [[2 * v, v], [v, v]]
    else:
        P[:2, :2] = [[G.V0 + p["q1"] / (1 - phi1 ** 2), G.V0], [G.V0, G.V0]]
    P[2, 2] = q3 / (1 - lam ** 2) if observes_bias else 0.0
    u = np.empty(len(z))
    for i in range(len(z)):
        c = coh[i]
        q1, q2 = p["q1"] * np.exp(p["aq1"] * (1 - c)), p["q2"] * np.exp(p["aq2"] * (1 - c))
        Q = np.array([[q1 + q2, q2, 0], [q2, q2, 0], [0, 0, q3 * np.exp(p["aq3"] * (1 - c))]])
        R = p["rlow"] if c < 0.5 else 1.0
        if sw[i] and spec.s == 8:
            Q = Q / (1 - gam)
        if sw[i] and spec.s == 9:
            R = R * np.exp(G.ETA_R_MAX * gam)
        if sw[i] and spec.s == 3:                        # (E11): the slow state reverts, the fast state is kept
            A = np.diag([1.0, 1 - gam, 1.0])
            s, P = A @ s + np.array([0, gam * G.MU0, 0]), A @ P @ A.T
        phi = 0.0 if sw[i] and G.T_CODES[spec.t] in G.RESET_CODES else phL if c < 0.5 else phH
        if sw[i] and G.T_CODES[spec.t] in G.SWITCH_CODES:              # S5: free persistence on a switch trial
            phi = G.WMAX / (1 + np.exp(-th[G.IX["wsw"]]))
        F = np.array([[phi, 1 - phi, float(fb)], [0, 1, 0], [0, 0, lam]])
        s, P = F @ s, F @ P @ F.T + Q
        K = P @ H / (H @ P @ H + R)
        s, P = s + K * (z[i] - H @ s), P - np.outer(K, H @ P)
        u[i] = Hx @ s
    return u


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_report_is_the_textbook_kalman_posterior(spec):
    _, a = participant(exp=1, k=2)
    th = random_theta(spec, np.random.default_rng(4))
    th[G.IX["lsw"]] = 0.3
    z = G.measurements(th, a["T"], a["coh"], np.random.default_rng(1), spec)
    assert np.allclose(G.observer(th, spec, z, a["coh"])[0], textbook(th, spec, z, a["coh"]), atol=1e-10)


def test_jax_matches_numpy_across_the_model_space():
    pytest.importorskip("jax")
    J = G.jax_funcs()
    _, a = participant(exp=1, k=3)
    rng = np.random.default_rng(7)
    for spec in SPECS:
        th = random_theta(spec, rng)
        ll_np = G.loglik(th, spec, np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"]).sum()
        ll_jx = -float(J["nll"](th, spec.vec(), np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"]))
        assert np.isclose(ll_np, ll_jx, atol=1e-7), spec.name


def test_q3_has_no_effect_without_a_bias_state():
    _, a = participant()
    rng = np.random.default_rng(1)
    spec = G.Spec(3, "T1s", 0, 1)
    th = random_theta(spec, rng)
    l1 = G.loglik(th, spec, np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"]).sum()
    th[G.IX["aq3"]] = 2.5
    assert np.isclose(l1, G.loglik(th, spec, np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"]).sum())


@pytest.mark.parametrize("s", [0, 3, 8, 9])
def test_the_report_is_the_posterior(s):
    """g = H K: the weight of the measurement in the report is the gain of the update (one filter)."""
    _, a = participant()
    spec = G.Spec(5, "T1n", s, 2)
    th = random_theta(spec, np.random.default_rng(2))
    u, c, g, A, K, R = G.observer(th, spec, np.log(a["T"]), a["coh"])
    H = np.array([1.0, 1.0, 0.0])
    assert np.allclose(g, K @ H)


def test_r_scales_the_generated_measurement():
    """C4: the measurement noise at Low is sigma_s sqrt(r_low) (E2), and the simulation uses these measurements."""
    spec = G.Spec(4, "T0", 0, 1)
    th = random_theta(spec, np.random.default_rng(3))
    th[G.IX["lrlow"]], th[G.IX["lsig"]] = np.log(4.0), np.log(0.2)
    n = 20000
    coh = np.where(np.arange(n) % 2 == 0, 0.3, 0.7)
    z = G.measurements(th, np.full(n, 1.2), coh, np.random.default_rng(0)) - np.log(1.2)
    assert np.isclose(z[::2].std() / z[1::2].std(), 2.0, rtol=0.03)
    th[G.IX["lsde"]] = -6.0
    y1 = G.simulate(th, spec, np.full(50, 1.2), coh[:50], np.zeros(50), np.ones(50, bool), np.random.default_rng(5))[0]
    zz = G.measurements(th, np.full(50, 1.2), coh[:50], np.random.default_rng(5))
    p = G.unpack(th)
    expect = np.exp(G.observer(th, spec, zz, coh[:50])[0]) + p["d0"] + p["ad0"] * (1 - coh[:50]) - p["b2"] / 3
    assert np.allclose(y1, expect, atol=0.02)


def test_free_parameter_counts():
    base = {"full": 11, "reduced": 7, "reduced_ase": 8}[G.CORE] + int(G.NOISE_TREND)
    assert G.Spec(0, "T0", 0, 1).k() == base
    assert G.Spec(0, "Tn", 0, 1).k() == base - 1
    assert G.Spec(0, "T1n", 0, 1).k() == base
    assert G.Spec(0, "T1s", 0, 3).k() == base + 1 + 2
    assert G.Spec(15, "T1sS", 9, 3).k() == base + 1 + 2 + 4 + 1
    for name in ("C0-T2s-S0-B3", "C0-T0-S1-B3", "C0-T0-S0-B3-D1L"):     # outside the standard space (archived)
        with pytest.raises(AssertionError):
            G.Spec.parse(name)


@pytest.mark.parametrize("parent, child, exact", [
    ("C0-Tn-S0-B3", "C0-T0-S0-B3", True), ("C0-Tn-S0-B3", "C0-T1n-S0-B3", True), ("C0-T0-S0-B3", "C0-T1s-S0-B3", True),
    ("C1-T1s-S0-B2", "C5-T1s-S0-B2", True), ("C0-T1n-S0-B3", "C0-T1nS-S0-B3", False),
    ("C0-T0-S0-B3", "C0-T0-S9-B3", False), ("C0-T1s-S0-B1", "C0-T1s-S0-B3", False),
    ("C0-T0-S0-B3", "C0-T0W-S0-B3", True), ("C0-T0S-S0-B3", "C0-T0W-S0-B3", False),
])
def test_embedding_reproduces_the_parent(parent, child, exact):
    from sdu.model import fitting as FF
    _, a = participant()
    P, C = G.Spec.parse(parent), G.Spec.parse(child)
    th = random_theta(P, np.random.default_rng(11))
    args = (np.log(a["T"]), a["coh"], a["tau"], np.nan_to_num(a["y"]), a["mask"])
    if C.t == "T1nS":                           # a reset is not nested: equal only without switches
        args = (args[0], np.full(len(args[0]), 0.3)) + args[2:]
    embedded = FF.embed(th, P, C)
    lp, lc = G.loglik(th, P, *args).sum(), G.loglik(embedded, C, *args).sum()
    if P.b == 1 and C.b != 1:
        # Positive q3 bounds do not contain the zero-bias model. Check the mathematical
        # zero-variance limit separately from the admissible fitting start.
        lo, hi = C.bounds()
        assert np.all((embedded >= lo) & (embedded <= hi))
        assert embedded[G.IX["lq3"]] == lo[G.IX["lq3"]]
        limit = embedded.copy()
        limit[G.IX["lq3"]] = -20.0
        assert np.isclose(lp, G.loglik(limit, C, *args).sum(), rtol=0, atol=1e-6)
        return
    # not exact: the switch or bias parameter sits at its bound (sw ~ 5e-5, q3 = exp(-16)), a start, not a nesting
    assert np.isclose(lp, lc, atol=1e-6 if exact else 0.5), (lp, lc)


def test_tilt_form():
    """T1s tilts the persistence symmetrically on the log-odds scale around T0 (E9) and is T0 at beta = 0."""
    th = random_theta(G.Spec(0, "T1s", 0, 1), np.random.default_rng(5))
    th[G.IX["beta"]] = 0.8
    pL, pH = G.weights(th, G.T_CODES["T1s"])
    lg = lambda v: np.log(v / (1 - v))
    assert pL > pH and np.isclose(lg(pL) + lg(pH), 2 * lg(G.weights(th, G.T_CODES["T0"])[0]))
    th[G.IX["beta"]] = 0.0
    assert np.allclose(G.weights(th, G.T_CODES["T1s"]), G.weights(th, G.T_CODES["T0"]))


def test_bias_gain_follows_coupling():
    """K^(b) is non-zero exactly when the bias state enters the transition (F13) or the measurement (H13)."""
    rng = np.random.default_rng(3)
    n = 40
    logT = np.log(rng.choice([0.8, 1.0, 1.2, 1.4, 1.6], n))
    coh = rng.choice([0.3, 0.7], n)
    th = G.X0.copy()
    th[G.IX["lq3"]], th[G.IX["lam"]] = np.log(0.3), 1.0
    for b, (f, h, _, _) in G.B_COUPLE.items():
        K = G.observer(th, G.Spec(0, "T1n", 0, b), logT, coh)[4]
        assert (np.abs(K[:, 2]).max() > 1e-6) == bool(f or h), b
