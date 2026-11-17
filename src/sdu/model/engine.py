"""The three-state Kalman observer (manuscript Section 5 and Appendix B.1-B.3) and its exact likelihood.

The observer
------------
The observer tracks a learned mean m (slow state), a fast deviation f = x - m (so the duration estimate is x = m + f)
and, under B1, a bias b. Each trial i is a prediction and a Kalman update with the presented duration s_i (in s) as
the measurement; the measurement is the stimulus itself, and its variance R_i only expresses the reliability the
observer assigns to it (R = 1 at high coherence, R_L at low coherence), so it sets the gain:

    prediction   m-_i = m+_{i-1},  f-_i = F13 b+_{i-1},  b-_i = lambda b+_{i-1},  P-_i = F P+_{i-1} F' + Q_i
    update       K_i = P-_i H' / (H P-_i H' + R_i),  H = [1, 1, 0];  state+ = state- + K_i (s_i - H state-)
    readout      u_i = x+_i + (1 - K^(x)_i) phi_i f+_{i-1}

The retention phi_i (phi_L at low, phi_H at high current coherence) acts only on the readout: the previous fast
deviation is carried into the estimate of the current trial but is not written back into the filter. The process
noise is diagonal in the states (x, m, b): Var(x innovation) = q1, Var(m innovation) = q2, Var(b innovation) = q3.
The learned mean starts at mu0 = 1.2 s with a large variance (MEM0), so that the first duration determines it.

Covariances are in units of the high-coherence measurement variance. The fitted noise parameters are the innovation
SDs in seconds, sd_qj = sqrt(q_j) * SIGMA_S, with SIGMA_S = 0.1 s the unit; q_j = (sd_qj / SIGMA_S)^2 is the value
reported in the manuscript (Table B2). Under B2 the bias is a latent drift of the response in seconds with SD sd_q3.

Response:  y_i = u_i + d0 + alpha_d0 (1 - coh_i) + b_i + e_i,  e_i ~ N(0, sigma_e^2), coh in {0.3, 0.7}.

The model space (Table B1): every model is one level of each dimension.
    C1-C15  coherence modulates a non-empty subset of {q1, q2, q3, R}: Low/High ratio of that variance
            (q_eff = q exp(alpha (1 - coh)), alpha = ln(ratio) / 0.4)
    S0      no switch operation
    S1      on a switch of coherence the learned mean reverts toward mu0: m <- (1 - gamma) m + gamma mu0
    S2      on a switch the gain is reduced: K <- (1 - kappa) K
    S3      on a switch the trace of the preceding trial is dropped: phi_i = 0
    B0      no bias state
    B1      the bias is tracked by the observer and enters the duration prediction (F13 = 1)
    B2      the bias is a latent AR(1) drift added to the response only (F13 = 0)
C levels that modulate q3 are not combined with B0 (q3 has no effect without a bias state): 148 models.
C0 (no coherence modulation) is used only for data without a coherence manipulation (Appendix C).

Likelihood
----------
The model is linear and Gaussian, so p(y_i | y_<i) is exact. Under B0 and B1 the responses are independent given the
durations; under B2 the latent bias is integrated out by a scalar Kalman filter over the response residuals.

`loglik`, `observer` and `simulate` are the NumPy reference; `jax_funcs` gives the same likelihood in JAX (with its
gradient) for fitting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- constants

DURATIONS = np.array([0.8, 1.0, 1.2, 1.4, 1.6])
MU0 = float(DURATIONS.mean())       # initial learned mean, 1.2 s
MEM0 = 1e6                          # initial variance of the learned mean (uninformative)
SIGMA_S = 0.1                       # unit of the observer's variances, in s: q_j = (sd_qj / SIGMA_S)^2
LOW_COHERENCE = 0.5                 # coherence below this is "Low" (30%), above it "High" (70%)

# ---------------------------------------------------------------- parameters

# The parameter vector (one per model; parameters a model does not use are fixed at FIXED):
#   sd_q1, sd_q2   innovation SD of the fast and slow state (s)
#   phi_L, phi_H   retention at low and high current coherence
#   lambda         persistence of the bias
#   sd_q3          innovation SD of the bias (s)
#   d0             response offset (s);  d0_LH  Low - High offset difference (s) = 0.4 alpha_d0
#   sigma_e        response-noise SD (s)
#   ratio_q1, ratio_q2, ratio_q3, ratio_R   Low/High ratio of q1, q2, q3, R
#   switch         logit of gamma (S1) or kappa (S2)
NAMES = ["sd_q1", "sd_q2", "phi_L", "phi_H", "lambda", "sd_q3", "d0", "d0_LH", "sigma_e",
         "ratio_q1", "ratio_q2", "ratio_q3", "ratio_R", "switch"]
IX = {n: i for i, n in enumerate(NAMES)}
BOUNDS = {"sd_q1": (0.0, 0.5), "sd_q2": (0.0, 0.5), "phi_L": (0.0, 1.0), "phi_H": (0.0, 1.0),
          "lambda": (0.0, 0.99), "sd_q3": (0.0, 0.5), "d0": (-0.5, 0.5), "d0_LH": (-0.5, 0.5),
          "sigma_e": (0.01, 0.5), "ratio_q1": (0.1, 10.0), "ratio_q2": (0.1, 10.0), "ratio_q3": (0.1, 10.0),
          "ratio_R": (0.1, 10.0), "switch": (-10.0, 10.0)}
LO = np.array([BOUNDS[n][0] for n in NAMES])
HI = np.array([BOUNDS[n][1] for n in NAMES])
X0 = np.array([dict(sd_q1=0.1, sd_q2=0.01, phi_L=0.2, phi_H=0.2, **{"lambda": 0.5}, sd_q3=0.02, d0=0.2,
                    d0_LH=-0.08, sigma_e=0.14, ratio_q1=1.0, ratio_q2=1.0, ratio_q3=1.0, ratio_R=1.0,
                    switch=-3.0)[n] for n in NAMES])           # default starting point
FIXED = {"phi_L": 0.0, "phi_H": 0.0, "lambda": 0.0, "sd_q3": 0.0, "ratio_q1": 1.0, "ratio_q2": 1.0,
         "ratio_q3": 1.0, "ratio_R": 1.0, "switch": 0.0}       # values of the parameters a model does not use

C_LOCI = {0: (), 1: ("q1",), 2: ("q2",), 3: ("q3",), 4: ("R",), 5: ("q1", "R"), 6: ("q2", "R"), 7: ("q3", "R"),
          8: ("q1", "q2"), 9: ("q1", "q3"), 10: ("q2", "q3"), 11: ("q1", "q2", "R"), 12: ("q1", "q3", "R"),
          13: ("q2", "q3", "R"), 14: ("q1", "q2", "q3"), 15: ("q1", "q2", "q3", "R")}
S_LEVELS = (0, 1, 2, 3)
B_LEVELS = (0, 1, 2)


@dataclass(frozen=True)
class Model:
    c: int      # 0-15 (C_LOCI)
    s: int      # 0-3
    b: int      # 0-2

    @property
    def name(self):
        return f"C{self.c}-S{self.s}-B{self.b}"

    @classmethod
    def parse(cls, name):
        m = re.fullmatch(r"C(\d+)-S(\d)-B(\d)", name)
        assert m, name
        model = cls(int(m[1]), int(m[2]), int(m[3]))
        assert model.c in C_LOCI and model.s in S_LEVELS and model.b in B_LEVELS, name
        return model

    def code(self):
        """Integer code of the switch and bias levels, passed to the JAX likelihood."""
        return np.array([self.s, self.b], np.int32)

    def free(self):
        """Names of the free parameters, in the order of NAMES."""
        f = {"sd_q1", "sd_q2", "phi_L", "phi_H", "d0", "d0_LH", "sigma_e"}
        if self.b:
            f |= {"lambda", "sd_q3"}
        f |= {f"ratio_{q}" for q in C_LOCI[self.c]}
        if self.s in (1, 2):
            f.add("switch")
        return [n for n in NAMES if n in f]

    def k(self):
        return len(self.free())

    def bounds(self):
        """Bounds of the parameter vector, with the unused entries fixed (lo = hi)."""
        lo, hi = LO.copy(), HI.copy()
        free = set(self.free())
        for n in NAMES:
            if n not in free:
                lo[IX[n]] = hi[IX[n]] = FIXED[n]
        return lo, hi


def space():
    """The 148 models of the manuscript: C1-C15 x S0-S3 x B0-B2, without the C levels that modulate q3 under B0."""
    return [Model(c, s, b) for c in range(1, 16) for s in S_LEVELS for b in B_LEVELS
            if not (b == 0 and "q3" in C_LOCI[c])]


def unpack(th, xp=np):
    """Parameter vector -> the quantities used by the observer (variances in units of the measurement variance)."""
    return dict(q1=(th[IX["sd_q1"]] / SIGMA_S) ** 2, q2=(th[IX["sd_q2"]] / SIGMA_S) ** 2,
                q3=(th[IX["sd_q3"]] / SIGMA_S) ** 2,                # bias tracked by the observer (B1)
                q3s=th[IX["sd_q3"]] ** 2,                           # latent response bias (B2), in s^2
                phL=th[IX["phi_L"]], phH=th[IX["phi_H"]], lam=th[IX["lambda"]],
                d0=th[IX["d0"]], ad0=th[IX["d0_LH"]] / 0.4, sde=th[IX["sigma_e"]],
                aq1=xp.log(th[IX["ratio_q1"]]) / 0.4, aq2=xp.log(th[IX["ratio_q2"]]) / 0.4,
                aq3=xp.log(th[IX["ratio_q3"]]) / 0.4, rlow=th[IX["ratio_R"]],
                sw=1 / (1 + xp.exp(-th[IX["switch"]])))


def switch_flags(coh):
    """True on trials whose coherence differs from the previous trial's."""
    c = np.asarray(coh, float)
    return np.r_[False, ~np.isclose(c[1:], c[:-1])]


def offsets(p, coh):
    return p["d0"] + p["ad0"] * (1 - coh)


# ---------------------------------------------------------------- NumPy reference


def observer(th, model: Model, s, coh):
    """Run the observer on the durations s (in s).

    Returns the readout u and, per trial, the linear maps of the filter, u_i = c_i state+_{i-1} + g_i s_i + const
    and state+_i = A_i state+_{i-1} + K_i s_i + const, in the coordinates (m, f, b): c (n, 3), g = K^(x) (n,),
    A (n, 3, 3) and K (n, 3) (K[:, 0] = K^(m), K[:, 1] = K^(x) - K^(m), K[:, 2] = K^(b)), and R (n,).
    """
    p = unpack(th)
    s, coh = np.asarray(s, float), np.asarray(coh, float)
    n = len(s)
    low = coh < LOW_COHERENCE
    sw = switch_flags(coh)
    tracked = model.b == 1                          # B1: the observer tracks the bias, which enters F
    lam_o = p["lam"] if tracked else 0.0
    q3_o = p["q3"] if tracked else 0.0
    H = np.array([1.0, 1.0, 0.0])
    I3 = np.eye(3)
    gam = p["sw"]
    Ar = I3.copy()                                  # S1: m <- (1 - gamma) m + gamma mu0 (x = m + f unchanged)
    Ar[0, 0], Ar[1, 0] = 1 - gam, gam
    o = np.array([gam * MU0, -gam * MU0, 0.0])
    F = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, float(tracked)], [0.0, 0.0, lam_o]])

    x = np.array([MU0, 0.0, 0.0])
    P = np.diag([MEM0, 0.0, q3_o / (1 - lam_o ** 2) if tracked else 0.0])
    u, g, R = np.empty(n), np.empty(n), np.empty(n)
    c_r, A, K = np.empty((n, 3)), np.empty((n, 3, 3)), np.empty((n, 3))
    for t in range(n):
        cc = coh[t]
        q2, q1 = p["q2"] * np.exp(p["aq2"] * (1 - cc)), p["q1"] * np.exp(p["aq1"] * (1 - cc))
        Q = np.diag([q2, q1, q3_o * np.exp(p["aq3"] * (1 - cc))])
        Q[0, 1] = Q[1, 0] = -q2                     # diagonal in (x, m): in (m, f) the drift of m leaves x
        Q[1, 1] = q1 + q2
        R[t] = p["rlow"] if low[t] else 1.0
        A3 = I3
        f_prev = x[1]                               # f+_{t-1}, the trace carried into the readout
        if sw[t] and model.s == 1:                  # S1: the learned mean reverts before the prediction
            x, P, A3 = Ar @ x + o, Ar @ P @ Ar.T, Ar
        phi = 0.0 if (sw[t] and model.s == 3) else (p["phL"] if low[t] else p["phH"])   # S3: trace dropped
        xp, Pp = F @ x, F @ P @ F.T + Q
        k = Pp @ H / (H @ Pp @ H + R[t])
        if sw[t] and model.s == 2:                  # S2: the gain is reduced
            k = k * (1 - gam)
        g[t] = H @ k
        P = Pp - np.outer(k, H @ Pp)
        x = xp + k * (s[t] - H @ xp)
        u[t] = H @ x + (1 - g[t]) * phi * f_prev
        c_r[t] = (H - g[t] * H) @ F @ A3 + (1 - g[t]) * phi * np.array([0.0, 1.0, 0.0])
        K[t] = k
        A[t] = (I3 - np.outer(k, H)) @ F @ A3
    return u, c_r, g, A, K, R


def response_mean(th, model, s, coh):
    """Mean response given the durations (no bias drift, no response noise)."""
    p = unpack(th)
    return observer(th, model, s, coh)[0] + offsets(p, np.asarray(coh, float))


def loglik(th, model: Model, s, coh, y, mask):
    """Per-trial log p(y_i | y_<i) (0 on trials with mask False)."""
    p = unpack(th)
    s, coh = np.asarray(s, float), np.asarray(coh, float)
    n = len(s)
    u = observer(th, model, s, coh)[0]
    resid = np.where(mask, y, 0.0) - (u + offsets(p, coh))
    se2 = p["sde"] ** 2
    ll = np.zeros(n)
    if model.b != 2:                                # independent Gaussian responses
        ll[mask] = -0.5 * (np.log(2 * np.pi * se2) + resid[mask] ** 2 / se2)
        return ll
    rho, q3 = p["lam"], p["q3s"]                    # B2: Kalman filter of the latent response bias
    vd = q3 * np.exp(p["aq3"] * (1 - coh))
    m, C = 0.0, q3 / (1 - rho ** 2)
    for t in range(n):
        m, C = rho * m, rho ** 2 * C + vd[t]
        if mask[t]:
            S = C + se2
            e = resid[t] - m
            ll[t] = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            kk = C / S
            m, C = m + kk * e, C - kk * C
    return ll


def simulate(th, model: Model, s, coh, valid, rng, return_mean=True):
    """Simulated responses (NaN where valid is False) and, optionally, the mean response."""
    p = unpack(th)
    s, coh = np.asarray(s, float), np.asarray(coh, float)
    n = len(s)
    b = np.zeros(n)
    if model.b == 2:                                # latent bias: stationary before the first trial
        vd = p["q3s"] * np.exp(p["aq3"] * (1 - coh))
        prev = rng.standard_normal() * np.sqrt(p["q3s"] / (1 - p["lam"] ** 2))
        for t in range(n):
            prev = b[t] = p["lam"] * prev + rng.standard_normal() * np.sqrt(vd[t])
    u = observer(th, model, s, coh)[0]
    y = u + offsets(p, coh) + b + rng.standard_normal(n) * p["sde"]
    mean = response_mean(th, model, s, coh) if return_mean else None
    return np.where(np.asarray(valid, bool), y, np.nan), mean


# ---------------------------------------------------------------- JAX


def jax_funcs():
    """JAX likelihood of every model: nll(theta, code, s, coh, y, mask), with code = Model.code(). One compilation
    serves all models: the C level enters through the fixed entries of theta, S and B through code."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def trial_ll(th, code, s, coh, y, mask):
        scode, bcode = code[0], code[1]
        p = unpack(th, jnp)
        low = coh < LOW_COHERENCE
        tracked = (bcode == 1).astype(float)
        latent = bcode == 2
        lam_o, q3_o = tracked * p["lam"], tracked * p["q3"]
        H = jnp.array([1.0, 1.0, 0.0])
        I3 = jnp.eye(3)
        gam = p["sw"]
        Ar = I3.at[0, 0].set(1 - gam).at[1, 0].set(gam)
        o = jnp.array([gam * MU0, -gam * MU0, 0.0])
        swf = jnp.concatenate([jnp.zeros(1, bool), jnp.abs(coh[1:] - coh[:-1]) > 1e-9])
        F = jnp.zeros((3, 3)).at[0, 0].set(1.0).at[1, 2].set(tracked).at[2, 2].set(lam_o)
        x0 = jnp.array([MU0, 0.0, 0.0])
        P0 = jnp.diag(jnp.array([MEM0, 0.0, q3_o / (1 - lam_o ** 2)]))

        def obs_step(carry, inp):
            x, P = carry
            st, cc, lo_, w = inp
            q2, q1 = p["q2"] * jnp.exp(p["aq2"] * (1 - cc)), p["q1"] * jnp.exp(p["aq1"] * (1 - cc))
            Q = jnp.diag(jnp.array([q2, q1, q3_o * jnp.exp(p["aq3"] * (1 - cc))]))
            Q = Q.at[0, 1].set(-q2).at[1, 0].set(-q2).at[1, 1].set(q1 + q2)
            R = jnp.where(lo_, p["rlow"], 1.0)
            f_prev = x[1]
            pre = w & (scode == 1)                                      # S1
            A3 = jnp.where(pre, Ar, I3)
            x = jnp.where(pre, Ar @ x + o, x)
            P = jnp.where(pre, Ar @ P @ Ar.T, P)
            phi = jnp.where(w & (scode == 3), 0.0, jnp.where(lo_, p["phL"], p["phH"]))   # S3
            xp, Pp = F @ x, F @ P @ F.T + Q
            k = Pp @ H / (H @ Pp @ H + R)
            k = jnp.where(w & (scode == 2), k * (1 - gam), k)            # S2
            g = H @ k
            xn, Pn = xp + k * (st - H @ xp), Pp - jnp.outer(k, H @ Pp)
            return (xn, Pn), H @ xn + (1 - g) * phi * f_prev

        _, u = jax.lax.scan(obs_step, (x0, P0), (s, coh, low, swf))
        resid = jnp.where(mask, y, 0.0) - (u + p["d0"] + p["ad0"] * (1 - coh))
        se2 = p["sde"] ** 2
        rho, q3 = latent * p["lam"], latent * p["q3s"]
        vd = q3 * jnp.exp(p["aq3"] * (1 - coh))

        def lik_step(carry, inp):                                       # latent bias (B2); zero otherwise
            m, C = carry
            rt, mt, vdt = inp
            m, C = rho * m, rho ** 2 * C + vdt
            S = C + se2
            e = rt - m
            ll = jnp.where(mt, -0.5 * (jnp.log(2 * jnp.pi * S) + e * e / S), 0.0)
            kk = jnp.where(mt, C / S, 0.0)
            return (m + kk * e, C - kk * C), ll

        C0 = jnp.where(latent, q3 / (1 - rho ** 2), 0.0)
        _, ll = jax.lax.scan(lik_step, (0.0, C0), (resid, mask, vd))
        return ll

    def nll(th, code, s, coh, y, mask):
        return -trial_ll(th, code, s, coh, y, mask).sum()

    return dict(jnp=jnp, nll=jax.jit(nll), vg=jax.jit(jax.value_and_grad(nll)), ll=jax.jit(trial_ll))
