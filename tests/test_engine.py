"""The three-state observer (sdu.model.engine): a textbook Kalman filter in the absolute states, the exact
likelihood, JAX = NumPy, the model space and the parameterization."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from sdu.model import engine as G
from sdu.model import fitting as FF
from sdu.reports import plots as P

MODELS = [G.Model.parse(n) for n in ("C1-S0-B0", "C3-S0-B2", "C3-S3-B2", "C4-S1-B1", "C7-S2-B2", "C9-S3-B1",
                                     "C12-S1-B2", "C15-S2-B1", "C6-S3-B0", "C0-S0-B2")]


def participant(exp=1, k=0):
    data = FF.load(exp)
    return data[sorted(data)[k]]


def theta(model, seed=3):
    """A parameter vector inside the bounds with appreciable retention and switch effects."""
    lo, hi = model.bounds()
    th = np.where(lo == hi, lo, lo + (hi - lo) * np.random.default_rng(seed).uniform(.25, .75, len(lo)))
    for name, value in (("phi_L", .6), ("phi_H", .3), ("switch", 0.0)):
        if lo[G.IX[name]] < hi[G.IX[name]]:
            th[G.IX[name]] = value
    return th


def textbook(th, model, s, coh):
    """Kalman filter on the absolute states (x, m, b) as in Appendix B.1: x- = m + F13 b, m- = m, b- = lambda b,
    Q = diag(q1, q2, q3), H = [1, 0, 0]; readout u = x+ + (1 - K^(x)) phi f+ of the previous trial."""
    p = G.unpack(th)
    tracked = model.b == 1
    lam, q3 = (p["lam"], p["q3"]) if tracked else (0.0, 0.0)
    gam, sw = p["sw"], G.switch_flags(coh)
    H = np.array([1.0, 0.0, 0.0])
    F = np.array([[0.0, 1.0, float(tracked)], [0.0, 1.0, 0.0], [0.0, 0.0, lam]])
    st = np.array([G.MU0, G.MU0, 0.0])
    P = np.diag([0.0, G.MEM0, q3 / (1 - lam ** 2) if tracked else 0.0])
    P[0, 0] = P[0, 1] = P[1, 0] = G.MEM0
    u = np.empty(len(s))
    for i, c in enumerate(coh):
        f_prev = st[0] - st[1]
        if sw[i] and model.s == 1:                                   # S1: the learned mean reverts toward mu0
            A = np.diag([1.0, 1 - gam, 1.0])
            st, P = A @ st + np.array([0.0, gam * G.MU0, 0.0]), A @ P @ A.T
        Q = np.diag([p["q1"] * np.exp(p["aq1"] * (1 - c)), p["q2"] * np.exp(p["aq2"] * (1 - c)),
                     q3 * np.exp(p["aq3"] * (1 - c))])
        R = p["rlow"] if c < .5 else 1.0
        st, P = F @ st, F @ P @ F.T + Q
        K = P @ H / (H @ P @ H + R)
        if sw[i] and model.s == 2:                                   # S2: reduced gain
            K = K * (1 - gam)
        st, P = st + K * (s[i] - H @ st), P - np.outer(K, H @ P)
        phi = 0.0 if (sw[i] and model.s == 3) else (p["phL"] if c < .5 else p["phH"])
        u[i] = st[0] + (1 - K[0]) * phi * f_prev
    return u


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_readout_is_the_textbook_filter(model):
    T, coh, _, _ = participant(exp=2, k=1)
    th = theta(model)
    assert np.allclose(G.observer(th, model, T, coh)[0], textbook(th, model, T, coh), atol=1e-10)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.name)
def test_likelihood_is_the_exact_gaussian(model):
    """Over a short sequence the log-likelihood equals the multivariate normal density of the responses: independent
    response noise, plus the AR(1) latent bias under B2."""
    T, coh, y, valid = (a[:40] for a in participant(exp=1, k=2))
    th = theta(model)
    p = G.unpack(th)
    mean = G.response_mean(th, model, T, coh)
    cov = np.eye(len(T)) * p["sde"] ** 2
    if model.b == 2:
        vd = p["q3s"] * np.exp(p["aq3"] * (1 - coh))
        b_cov = np.zeros((len(T), len(T)))
        var = p["q3s"] / (1 - p["lam"] ** 2)
        for i in range(len(T)):                                      # Var(b_i), Cov(b_i, b_j) of the AR(1) drift
            var = p["lam"] ** 2 * var + vd[i]
            b_cov[i, i] = var
            for j in range(i + 1, len(T)):
                b_cov[i, j] = b_cov[j, i] = p["lam"] ** (j - i) * var
        cov = cov + b_cov
    exact = multivariate_normal(mean[valid], cov[np.ix_(valid, valid)]).logpdf(y[valid])
    assert np.isclose(G.loglik(th, model, T, coh, y, valid).sum(), exact, atol=1e-8)


def test_jax_matches_numpy():
    pytest.importorskip("jax")
    J = G.jax_funcs()
    T, coh, y, valid = participant(exp=2, k=3)
    for model in MODELS:
        th = theta(model, seed=5)
        ll_np = G.loglik(th, model, T, coh, y, valid).sum()
        assert np.isclose(ll_np, -float(J["nll"](th, model.code(), T, coh, y, valid)), atol=1e-8), model.name


def test_model_space():
    space = G.space()
    assert len(space) == 148 and len({m.name for m in space}) == 148
    assert not any(m.b == 0 and "q3" in G.C_LOCI[m.c] for m in space)
    assert G.Model.parse("C3-S0-B2").k() == 10 and G.Model.parse("C3-S3-B2").k() == 10
    assert min(m.k() for m in space) == 8 and max(m.k() for m in space) == 14
    for name in ("C16-S0-B0", "C1-S4-B0", "C1-S0-B3", "C3-R-S0-B2"):
        with pytest.raises(AssertionError):
            G.Model.parse(name)


def test_memory_does_not_depend_on_retention():
    """The retention acts only at the readout: gains, state maps and the carried state are those of phi = 0."""
    T, coh, _, _ = participant(exp=1, k=0)
    for model in MODELS:
        th = theta(model)
        th0 = th.copy()
        th0[[G.IX["phi_L"], G.IX["phi_H"]]] = 0.0
        out, out0 = G.observer(th, model, T, coh), G.observer(th0, model, T, coh)
        for j in (2, 3, 4, 5):
            assert np.allclose(out[j], out0[j], atol=1e-12)
        assert not np.allclose(out[0], out0[0])


def test_q3_has_no_effect_without_a_bias_state():
    """Why the C levels that modulate q3 are not combined with B0."""
    T, coh, y, valid = participant()
    model = G.Model(9, 0, 0)
    th = theta(model)
    ll = G.loglik(th, model, T, coh, y, valid).sum()
    th[G.IX["ratio_q3"]] = 5.0
    assert np.isclose(ll, G.loglik(th, model, T, coh, y, valid).sum(), rtol=0, atol=1e-12)


def test_reported_bias_sd_is_the_simulated_bias():
    """plots.b2_bias_sd (Table B2) is the SD of the latent bias drift that the simulation produces."""
    model = G.Model.parse("C3-S0-B2")
    lo, hi = model.bounds()
    th = np.where(lo == hi, lo, (lo + hi) / 2)
    th[G.IX["lambda"]], th[G.IX["sd_q3"]], th[G.IX["ratio_q3"]] = 0.0, 0.08, 2.0
    th[G.IX["sigma_e"]] = lo[G.IX["sigma_e"]]
    th[[G.IX["phi_L"], G.IX["phi_H"]]] = 0.0
    p = G.unpack(th)
    n = 40000
    T = np.random.default_rng(1).choice(G.DURATIONS, n)
    for c in (0.3, 0.7):
        coh = np.full(n, c)
        y = G.simulate(th, model, T, coh, np.ones(n, bool), np.random.default_rng(2), return_mean=False)[0]
        bias_var = np.var(y - G.response_mean(th, model, T, coh)) - p["sde"] ** 2
        assert abs(bias_var / P.b2_bias_sd(p, c) ** 2 - 1) < 0.05


def test_ratio_parameters_are_low_high_ratios():
    th = G.X0.copy()
    th[G.IX["ratio_q1"]] = 4.0
    p = G.unpack(th)
    assert np.isclose(np.exp(p["aq1"] * (1 - .3)) / np.exp(p["aq1"] * (1 - .7)), 4.0)
