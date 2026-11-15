"""Three-state Kalman observer of Appendix C: the standard Kalman model space (2026-10-31).

Every model is one generative model of the observer with standard Kalman inference: one prediction and one optimal
update per trial, and the readout (the estimate) is the posterior of the current duration (Appendix C.2). F, Q and R may depend on
the cues the observer sees on each trial (its coherence, and whether it switched). The levels (gkf names; public
names in kf_space.py):

  C1-C15  coherence modulates the observer's noise: non-empty subsets of {q1, q2, q3, R} (index 0, no modulation,
          exists only for the duplicate rule of space.py and is not a model)
  T       persistence phi of the fast state in F:
          Tn    phi = 0 (public T0)
          T0    phi constant (public T1)
          T1s   logit phi(coh) = logit phi +- beta (public T2)
          T1n   phi^L under Low coherence, 0 under High (public T3)
          T0S, T1sS, T1nS  as T0, T1s, T1n on repeat trials, phi = 0 on switch trials (public S4, the reset)
          TnW, T0W, T1sW, T1nW  as Tn, T0, T1s, T1n on repeat trials; free persistence phi_sw = WMAX sigma(wsw) on
                switch trials (public S5, 2026-11-01)
  S       switch of coherence: S0 none; S3 the slow state reverts toward the design mean in the transition of a
          switch trial (public S1); S8 process noise Q / (1 - sw) on a switch trial (public S2); S9 measurement noise
          R exp(eta_R), eta_R = ETA_R_MAX * sw, actual and believed (public S3)
  B1-B9   the third state b (bias) and where it is coupled (B_COUPLE; 2026-11-01). b is an AR(1) state (lambda, q3)
          in the three-state vector; it can enter the transition of the fast state (F13 = 1), the measurement
          (H13 = 1: z_i = x_i + b_i + v_i, actual and believed) and the report (y_i = exp(u_i + b_i)). The observer
          tracks b whenever it enters F or H; the gain on b, K^(b), is then non-zero, so whether b is updated follows
          from the coupling. B1 no b; B2 F; B4 H; B5 report; B6 F+H; B7 F+report; B8 H+report; B9 F+H+report;
          B3 (kept from the first space) b in seconds added to the report. b is in normalised log units (its
          innovation variance sigma_s^2 q3, like the observer's covariances), except under B3 (seconds, q3 in s^2).

Mechanisms outside the standard space (separate transitions for the report and the memory, gains scaled away from
their optimum, resets at one step, readouts that mix in another estimate, the adaptation rule) are archived with the
general module they were fitted with (in the project's working repository). For the models kept here this
module gives the same likelihood as that one (checked on 2026-10-31).

The observer works in the coordinates (m, f = x - m, b) with H = [1, 1, 0]; the absolute fast state is x = m + f.
All covariances of the observer are in units of the High-coherence measurement variance (R_High = 1); the actual
measurement is z_i = log T_i + sigma_s sqrt(R_i) eps_i, and the observer is calibrated (Appendix C.1, E2).

The report is linear in the measurements, u_i = c_i x+_{i-1} + g_i z_i + const, x+_i = A_i x+_{i-1} + K_i z_i + const;
the likelihood propagates the deviations caused by sensory noise through these maps and a response-level Kalman filter
over the B3 bias state. The response term exp(u) is handled by LIK (environment variable GKF_LIK):
  "lin"  linearised around the noise-free report (the fits up to 2026-10-28);
  "mm"   moment matching: the deviation of the log report is Gaussian given the past responses, so the mean, the
         variance of exp(u) and its covariance with the filter's state are exact (lognormal moments, Stein's lemma);
         only the predictive distribution of the response is approximated as Gaussian (checked in lin_check.py).
  "gh"   adaptive Gauss-Hermite: p(y_i | y_<i) is integrated over the deviation of the log report (Gaussian given the
         past responses), with the B3 bias Gaussian given it; the filter's posterior is moment-matched (assumed-density
         filter). Agrees with the particle filter of lin_check.py to within its Monte Carlo error (lin_check_gh.log);
         "lin" and "mm" do not when the sensory noise is large (up to 14 log-likelihood units per participant).

Scale (environment variable GKF_SCALE; 2026-10-31, robustness check of Appendix C.5): "log" (default) is the model
above. "lin" runs the same observer on durations instead of log durations, as the models of the first submission did:
z_i = T_i + sigma_s sqrt(R_i) eps_i (sigma_s in seconds), the design prior in seconds, and the response
y_i = u_i + offsets + b_i + e_i. The model is then linear-Gaussian in the deviations, so the likelihood is the exact
Kalman likelihood (LIK is not used).

Parameters: one unconstrained vector THETA (NAMES below) for every model. A model fixes the entries it does not use
(free_mask, FIXED); the optimiser keeps them at their bounds lo = hi. wrL, wrH and leta belonged to the archived
mechanisms; they stay in NAMES, always fixed, so that the stored fits keep their columns.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

SCALE = os.environ.get("GKF_SCALE", "log")                # "log" or "lin" (docstring)
LIN = SCALE == "lin"
LOG_DUR = np.log([0.8, 1.0, 1.2, 1.4, 1.6])
_DUR = np.exp(LOG_DUR) if LIN else LOG_DUR
MU0, V0 = float(_DUR.mean()), float(_DUR.var())           # design prior (log units, or s under "lin"); v0 = V_T (E1)


def to_scale(logT, xp=np):
    """The observer's input from log durations: log durations, or durations under "lin"."""
    return xp.exp(logT) if LIN else logT


def report(u, xp=np):
    """The response term of the observer's estimate: exp(u), or u under "lin"."""
    return u if LIN else xp.exp(u)


WMAX = 0.999                                               # transition weights in [0, .999]
LIK = os.environ.get("GKF_LIK", "lin")                     # exp(u) in the likelihood: "lin" or "mm" (docstring)
ETA_R_MAX = 5.0                                             # S9: eta_R = ETA_R_MAX * sw, in [0, 5]
S2_MAX = 20.0                                              # cap on the predictive variance of the log report (mm)

NAMES = ["lq1", "lq2", "wrL", "wrH", "wuL", "wuH", "beta", "lam", "lq3", "lsig",
         "d0", "ad0", "lsde", "h", "kappa", "b1", "b2", "aq1", "aq2", "aq3", "lrlow", "lsw", "leta", "wsw", "sdt"]
IX = {n: i for i, n in enumerate(NAMES)}
LO = np.array([-10, -15, -10, -10, -10, -10, -7.6, -6, -8, -8, -1, -2, -6, -3, -2, -1, -1, -5, -5, -5,
               np.log(0.1), -10, -1.3862943611198906, -10, -2], float)
HI = np.array([4, 3, 10, 10, 10, 10, 7.6, 8, 0, 0, 1, 2, 0, 3, 2, 1, 1, 5, 5, 5, np.log(10.0), 10, 8, 10, 2], float)
# Parameter bounds and initialisation (environment variable GKF_BOUNDS; 2026-11-02): "v1" (all fits up to
# 2026-11-02) or "v2": bounds set on the natural scale of each parameter (sensory noise sigma_s and response noise
# sigma_e in [.01, .5]; q1 in [.001, 20] and q2 in [1e-5, 1] in units of sigma_s^2; persistence phi in [0, 1];
# bias persistence lambda in [0, .999]; q3 in [1e-5, 1]; offset d0 in [-.5, .5] s and its Low - High difference
# (.4 alpha_d0) in [-.5, .5] s; Low/High ratios of the response noise in [.25, 4] and of the process noise in [.2, 5]),
# and the initial memory and deviation variances set to the design variance V0 (in log units; V0 / sigma_s^2 in the
# observer's normalised covariance) instead of V0 sigma_s^2 and the stationary variance q1 / (1 - phi^2).
BOUNDS = os.environ.get("GKF_BOUNDS", "v2")
if BOUNDS not in ("v1", "v2", "v3"):
    raise ValueError("GKF_BOUNDS must be v3, v2 or v1 (historical reproduction).")
if BOUNDS == "v2":
    WMAX = 1.0
    _V2 = {"lq1": (np.log(1e-3), np.log(20.0)), "lq2": (np.log(1e-5), 0.0), "wuL": (-12.0, 12.0),
           "wuH": (-12.0, 12.0), "wsw": (-12.0, 12.0), "lam": (-12.0, np.log(0.999 / 0.001)),
           "lq3": (np.log(1e-5) / 2, 0.0), "lsig": (np.log(0.01), np.log(0.5)), "d0": (-0.5, 0.5),
           "ad0": (-1.25, 1.25), "lsde": (np.log(0.01), np.log(0.5)), "kappa": (-np.log(4.0), np.log(4.0)),
           "aq1": (np.log(0.2) / 0.4, np.log(5.0) / 0.4), "aq2": (np.log(0.2) / 0.4, np.log(5.0) / 0.4),
           "aq3": (np.log(0.2) / 0.4, np.log(5.0) / 0.4),
           "sdt": (-np.log(4.0) / 2, np.log(4.0) / 2)}      # response-noise SD ratio, session end / start, in [.25, 4]
    for _n, (_lo, _hi) in _V2.items():
        LO[IX[_n]], HI[IX[_n]] = _lo, _hi
# "v3" (2026-11-04): every free parameter is the quantity itself, bounded on its natural scale by one rule per kind
# (no logit or log transform in the fitted vector): persistence phi^L, phi^H in [0, 1] (T2: "beta" is phi^L, "wuH"
# phi^H); bias persistence lambda in [0, .99]; sensory noise SD sigma_s in [.01, .5] log units; the process-noise SDs
# of deviation, memory and bias ("lq1", "lq2", "lq3": sqrt(q_j) sigma_s) in [0, .5] log units; response-noise SD
# sigma_e in [.01, .5] s; offset d0 in [-.5, .5] s; the Low - High offset difference ("ad0" = .4 alpha_d0) in
# [-.5, .5] s; every Low/High variance ratio (measurement "lrlow", process noise "aq1"-"aq3", response noise "kappa")
# in [.1, 10]. The bias SD is in log units, or in seconds for the seconds bias (B2, code B3), in [0, .5].
# Initialization GKF_INIT=gs.
if BOUNDS == "v3":
    WMAX = 1.0
    _V3 = {"lq1": (0.0, 0.5), "lq2": (0.0, 0.5), "lq3": (0.0, 0.5), "wuL": (0.0, 1.0), "wuH": (0.0, 1.0),
           "beta": (0.0, 1.0), "wsw": (0.0, 1.0), "lam": (0.0, 0.99), "lsig": (0.01, 0.5), "lsde": (0.01, 0.5),
           "d0": (-0.5, 0.5), "ad0": (-0.5, 0.5), "kappa": (0.1, 10.0), "aq1": (0.1, 10.0), "aq2": (0.1, 10.0),
           "aq3": (0.1, 10.0), "lrlow": (0.1, 10.0)}
    for _n, (_lo, _hi) in _V3.items():
        LO[IX[_n]], HI[IX[_n]] = _lo, _hi
X0 = np.array([np.log(1.5), np.log(0.002), -1.0, -1.0, -1.0, -1.0, 0.0, 0.5, np.log(0.05), np.log(0.1),
               0.2, -0.2, np.log(0.14), 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, -1.0, 0.0], float)
FIXED = {"wrL": -1.0, "wrH": -1.0, "wuL": -1.0, "beta": 0.0, "lam": 0.0, "lq3": -8.0,
         "aq1": 0.0, "aq2": 0.0, "aq3": 0.0, "lrlow": 0.0, "lsw": 0.0, "h": 0.0, "kappa": 0.0, "b1": 0.0, "b2": 0.0,
         "leta": 8.0, "wsw": -40.0, "sdt": 0.0}
if BOUNDS == "v3":                                        # fixed values on the natural scale (zero persistence, ratio 1)
    FIXED.update({"wrL": 0.0, "wrH": 0.0, "wuL": 0.0, "wuH": 0.0, "beta": 0.0, "wsw": 0.0, "lam": 0.0, "lq3": 0.0,
                  "aq1": 1.0, "aq2": 1.0, "aq3": 1.0, "lrlow": 1.0, "kappa": 1.0})
    X0 = X0.copy()
    for _n, _v in {"lq1": 0.1, "lq2": 0.01, "lq3": 0.02, "wuL": 0.2, "wuH": 0.2, "beta": 0.2, "wsw": 0.2,
                   "lam": 0.5, "lsig": 0.1, "d0": 0.2, "ad0": -0.08, "lsde": 0.14, "kappa": 1.0, "aq1": 1.0,
                   "aq2": 1.0, "aq3": 1.0, "lrlow": 1.0}.items():
        X0[IX[_n]] = _v
# Session trend of the response noise (environment variable GKF_NOISE_TREND=1; 2026-11-04): sigma_e is multiplied by
# exp(sdt * tau), tau the trial position in [-1, 1]; a log-linear change of the response noise over the session
# (e.g. practice). Off by default (sdt fixed at 0), so every earlier fit and check is unchanged.
NOISE_TREND = os.environ.get("GKF_NOISE_TREND", "0") == "1"
# Initial variance of the fast deviation under GKF_BOUNDS=v2, as a multiple of V0 (environment variable GKF_DEV0;
# 2026-11-04 sensitivity check): 1 (default, the fits of framework_v2: memory and deviation each V0, so the
# fast state starts with 2 V0) or 0 (no deviation before the first trial: the fast state starts with V0).
DEV0 = float(os.environ.get("GKF_DEV0", "1"))
# Initialization under GKF_BOUNDS=v2 (environment variable GKF_INIT; 2026-11-04): "design" (default; the fits of
# framework_v2: memory and deviation variance each V0) or "gs": the memory starts uninformative (variance MEM0 in units
# of sigma_s^2, so the first measurement sets it, as the first measurement sets the prior mean in the two-stage model
# of Petzschner & Glasauer, 2011, used by Glasauer & Shi, 2022), the deviation from the memory starts at zero
# (variance 0: there is no previous trial), and the bias at the stationary variance of its AR(1) process.
INIT = os.environ.get("GKF_INIT", "gs" if BOUNDS == "v3" else "design")
assert INIT in ("design", "gs"), INIT
MEM0 = 1e6
# Core of every model (environment variable GKF_CORE): "full" (the fits up to 2026-10-28) or "reduced" (final
# comparison, 2026-10-28): no session trend (b1, b2), no duration or coherence scaling of the response noise (h, kappa);
# the report is decoded from the internal estimate with an offset, its coherence modulation and constant noise.
# "reduced_ase" (check, 2026-10-29): the reduced core with the coherence scaling of the response noise (kappa,
# alpha_sigma_e in the appendix) restored.
# Fast efficient encoding (environment variable GKF_EFFENC = gamma; 2026-11-06, robustness check after Prat-Carrabin,
# Harl & Gershman, 2026, Nat Commun 17, 5554): the measurement variance follows the width of the observer's current
# predictive prior, R_i = R_c(i) * min(H P-_i H' / V_ref, 1)^gamma, V_ref = V0 / sigma_s^2 (the design variance in the
# observer's normalised units). gamma = 0 (default) is the fixed encoding of every other fit; gamma = 1/2 is their
# result (Fisher information ~ 1 / prior SD, so encoding variance ~ prior SD). P-_i does not depend on the
# measurements, so R_i is a deterministic function of the cue sequence (like C and S3), the observer is calibrated to
# it and the inference stays the standard Kalman filter. The cap at 1: the encoding is never coarser than for the
# stimulus range (the first trial under GKF_INIT=gs has an uninformative memory).
EFF_GAMMA = float(os.environ.get("GKF_EFFENC", "0"))
# Noise-free measurement (environment variable GKF_NOISELESS=1; 2026-11-09, the first submission's assumption): the
# measurement is the stimulus itself (z_i = log T_i, or T_i under "lin"); R enters only the observer's gain (what the
# observer believes), so no sensory noise is propagated into the responses. The gain then depends only on Q / R, so
# sigma_s (the unit of Q and R) is fixed at its default and drops out of the free parameters. Off by default.
NOISELESS = os.environ.get("GKF_NOISELESS", "0") == "1"
# Diagonal process noise in the states (x, m, b) (environment variable GKF_QDIAG=1; 2026-11-09, the first submission's
# Q = diag(q1, q2, q3)): the fast state's innovation is independent of the memory drift. In the coordinates
# (m, f = x - m) used here this is Q = [[q2, -q2], [-q2, q1 + q2]] (bias row unchanged). Off by default: then
# Q = diag(q2, q1) in (m, f), i.e. the memory drift also moves the fast state.
QDIAG = os.environ.get("GKF_QDIAG", "0") == "1"
# Persistence at the readout (environment variable GKF_READOUT=1; 2026-11-11): the memory filter has no persistence
# (F with phi = 0 on every trial, so x-_i = m+_{i-1}), and the persistence of the T level acts only on the report,
# after the update: u_i = x+_i + (1 - g_i) phi_i f+_{i-1}, f+_{i-1} = x+_{i-1} - m+_{i-1} (the previous fast deviation,
# before an S1 reversion). phi_i follows the T level as before (phi^L / phi^H by the current coherence; 0 on switch
# trials of the reset levels). It is the posterior mean under the readout prior x-_i + phi_i f+_{i-1} with the memory
# filter's gain g_i, and it is not written back: the state and covariance carried to the next trial do not depend on
# phi. Off by default.
READOUT = os.environ.get("GKF_READOUT", "0") == "1"
CORE = os.environ.get("GKF_CORE", "full")
CORE_DROP = {"reduced": ("h", "kappa", "b1", "b2"), "reduced_ase": ("h", "b1", "b2")}.get(CORE, ())

C_LOCI = {0: (), 1: ("q1",), 2: ("q2",), 3: ("q3",), 4: ("R",), 5: ("q1", "R"), 6: ("q2", "R"), 7: ("q3", "R"),
          8: ("q1", "q2"), 9: ("q1", "q3"), 10: ("q2", "q3"), 11: ("q1", "q2", "R"), 12: ("q1", "q3", "R"),
          13: ("q2", "q3", "R"), 14: ("q1", "q2", "q3"), 15: ("q1", "q2", "q3", "R")}
T_CODES = {"Tn": 0, "T0": 0, "T1n": 1, "T1s": 2, "T0S": 3, "T1nS": 4, "T1sS": 5, "TnW": 6, "T0W": 6, "T1nW": 7,
           "T1sW": 8}
T_FREE = {"Tn": (), "T0": (), "T1n": ("wuL",), "T1s": ("beta",), "T0S": (), "T1nS": ("wuL",), "T1sS": ("beta",),
          "TnW": ("wsw",), "T0W": ("wsw",), "T1nW": ("wuL", "wsw"), "T1sW": ("beta", "wsw")}
NO_PHI = ("Tn", "T1n", "T1nS", "TnW", "T1nW")  # phi^H = 0 (wuH fixed)
RESET_CODES = (3, 4, 5)                        # T0S, T1nS, T1sS: fast-state reset on switch trials
SWITCH_CODES = (6, 7, 8)                       # TnW/T0W, T1nW, T1sW: free persistence phi_sw on switch trials
S_LEVELS = (0, 2, 3, 8, 9)
# S2 (2026-11-09): the first submission's S2, kept as it was: on a switch trial the Kalman gain is scaled down,
# K <- K (1 - k_reset), k_reset = sw in [0, 1], and the covariance is updated with the scaled gain,
# P+ = (I - K H) P- (three_state_135_nolog/engine.py). Not an optimal gain; a comparison level of the minimal model.
GAIN_CUT = 2
# B level -> (F13, H13, report in log units, report in seconds)
B_COUPLE = {1: (0, 0, 0, 0), 2: (1, 0, 0, 0), 3: (0, 0, 0, 1), 4: (0, 1, 0, 0), 5: (0, 0, 1, 0), 6: (1, 1, 0, 0),
            7: (1, 0, 1, 0), 8: (0, 1, 1, 0), 9: (1, 1, 1, 0)}
B_ARR = np.array([B_COUPLE.get(b, (0, 0, 0, 0)) for b in range(10)], float)


def b_flags(b):
    """(observer tracks b, b in F, b in H, b in the log report, b in seconds in the report, latent b exists)."""
    f, h, r, s = B_COUPLE[b]
    return bool(f or h), f, h, r, s, bool(h or r or s)


@dataclass(frozen=True)
class Spec:
    c: int  # 0-15
    t: str  # key of T_CODES
    s: int  # S_LEVELS
    b: int  # 1-9 (B_COUPLE)

    @property
    def name(self):
        return f"C{self.c}-{self.t}-S{self.s}-B{self.b}"

    @classmethod
    def parse(cls, name):
        m = re.fullmatch(r"C(\d+)-(T\w+)-S(\d)-B(\d)", name)
        assert m, name
        spec = cls(int(m[1]), m[2], int(m[3]), int(m[4]))
        assert (
            spec.c in C_LOCI and spec.t in T_CODES and spec.s in S_LEVELS and spec.b in B_COUPLE
        ), name
        return spec

    def vec(self):
        """Integer code passed to the JAX likelihood: (T code, S level, B level)."""
        return np.array([T_CODES[self.t], self.s, self.b], np.int32)

    def free(self):
        """Names of the free parameters."""
        f = ["lq1", "lq2", "wuH", "lsig", "d0", "ad0", "lsde", "h", "kappa", "b1", "b2"]
        f += list(T_FREE[self.t])
        if self.b != 1:
            f += ["lam", "lq3"]
        loci = C_LOCI[self.c]
        f += [f"a{q}" for q in ("q1", "q2", "q3") if q in loci] + (["lrlow"] if "R" in loci else [])
        if self.s:
            f.append("lsw")
        if self.t in NO_PHI:
            f.remove("wuH")
        if NOISE_TREND:
            f.append("sdt")
        if NOISELESS:  # sigma_s only sets the unit of Q and R (gain ~ Q / R)
            f.remove("lsig")
        f = [n for n in f if n not in CORE_DROP]
        return [n for n in NAMES if n in f]

    def k(self):
        return len(self.free())

    def bounds(self):
        """Per-parameter bounds with the unused entries fixed (lo = hi)."""
        lo, hi = LO.copy(), HI.copy()
        free = set(self.free())
        for n in NAMES:
            if n not in free:
                lo[IX[n]] = hi[IX[n]] = FIXED.get(n, X0[IX[n]])
        if self.t in NO_PHI:
            lo[IX["wuH"]] = hi[IX["wuH"]] = (
                0.0 if BOUNDS == "v3" else -40.0
            )  # phi = 0 (v1/v2: machine precision)

        return lo, hi


# ---------------------------------------------------------------- parameters (numpy and JAX share the formulas)


def weights(th, tcode, xp=np):
    """(phi^L, phi^H), the persistence of the fast state under Low and High coherence (Appendix C.2, Dimension T);
    on switch trials of the reset levels (RESET_CODES) phi = 0."""
    if BOUNDS == "v3":                             # the persistences themselves; T2: beta = phi^L, wuH = phi^H
        phi, phL, beta = th[IX["wuH"]], th[IX["wuL"]], th[IX["beta"]]
        rows = [(phi, phi), (phL, phi), (beta, phi)] * 3
        if xp is np:
            return rows[int(tcode)]
        return tuple(xp.stack([r[j] for r in rows])[tcode] for j in range(2))
    sig = (lambda v: 1 / (1 + xp.exp(-v)))
    phi = WMAX * sig(th[IX["wuH"]])
    phL = WMAX * sig(th[IX["wuL"]])
    wc = min(WMAX, 1 - 1e-12)
    lp = xp.log(xp.clip(phi, 1e-12, wc) / (1 - xp.clip(phi, 1e-12, wc)))
    beta = th[IX["beta"]]
    tilt = (xp.minimum(sig(lp + beta), WMAX), xp.minimum(sig(lp - beta), WMAX))
    rows = [(phi, phi), (phL, phi), tilt] * 3      # Tn/T0, T1n, T1s; T0S, T1nS, T1sS; TnW/T0W, T1nW, T1sW
    if xp is np:
        return rows[int(tcode)]
    return tuple(xp.stack([r[j] for r in rows])[tcode] for j in range(2))


def phi_switch(th, xp=np):
    """phi_sw, the persistence on switch trials of the S5 levels (SWITCH_CODES)."""
    if BOUNDS == "v3":
        return th[IX["wsw"]]
    return WMAX / (1 + xp.exp(-th[IX["wsw"]]))


def unpack(th, xp=np):
    e = xp.exp
    if BOUNDS == "v3":                             # natural-scale parameters (see the v3 bounds above)
        sig = th[IX["lsig"]]
        return dict(q1=(th[IX["lq1"]] / sig) ** 2, q2=(th[IX["lq2"]] / sig) ** 2, lam=th[IX["lam"]],
                    q3=(th[IX["lq3"]] / sig) ** 2, sig=sig, d0=th[IX["d0"]], ad0=th[IX["ad0"]] / 0.4,
                    sde=th[IX["lsde"]], h=th[IX["h"]], kappa=xp.log(th[IX["kappa"]]) / 2, b1=th[IX["b1"]],
                    b2=th[IX["b2"]], aq1=xp.log(th[IX["aq1"]]) / 0.4, aq2=xp.log(th[IX["aq2"]]) / 0.4,
                    aq3=xp.log(th[IX["aq3"]]) / 0.4, rlow=th[IX["lrlow"]], sw=1 / (1 + e(-th[IX["lsw"]])),
                    sdt=th[IX["sdt"]] if len(th) > IX["sdt"] else 0.0,
                    q3s=th[IX["lq3"]] ** 2)              # seconds bias (B2): its SD is in seconds, variance s^2
    return dict(q1=e(th[IX["lq1"]]), q2=e(th[IX["lq2"]]), lam=1 / (1 + e(-th[IX["lam"]])), q3=e(2 * th[IX["lq3"]]),
                sig=e(th[IX["lsig"]]), d0=th[IX["d0"]], ad0=th[IX["ad0"]], sde=e(th[IX["lsde"]]), h=th[IX["h"]],
                kappa=th[IX["kappa"]], b1=th[IX["b1"]], b2=th[IX["b2"]], aq1=th[IX["aq1"]], aq2=th[IX["aq2"]],
                aq3=th[IX["aq3"]], rlow=e(th[IX["lrlow"]]), sw=1 / (1 + e(-th[IX["lsw"]])),
                sdt=th[IX["sdt"]] if len(th) > IX["sdt"] else 0.0,      # parameter vectors saved before sdt
                q3s=e(2 * th[IX["lq3"]]))                                # v1/v2: q3 itself (s^2 for the seconds bias)


def switch_flags(coh):
    c = np.asarray(coh, float)
    return np.r_[False, ~np.isclose(c[1:], c[:-1])]


# ---------------------------------------------------------------- NumPy reference


def observer(th, spec: Spec, z, coh):
    """Run the observer on measurements z (log scale). Returns the report u and the per-trial linear maps
    (c_i, g_i, A_i, K_i) of the deviations, plus R_i: u_i = c_i x+_{i-1} + g_i z_i + const. A measurement deviation
    dz_i (sensory noise, and b_i under H13 = 1) moves the posterior by K_i dz_i and the readout by g_i dz_i.
    """
    p = unpack(th)
    tcode = T_CODES[spec.t]
    phL, phH = weights(th, tcode)
    z, coh = np.asarray(z, float), np.asarray(coh, float)
    n = len(z)
    low = coh < 0.5
    obs_b, fb, hb = b_flags(spec.b)[:3]
    lam_o = p["lam"] if obs_b else 0.0
    q3_o = p["q3"] if obs_b else 0.0
    H = np.array([1.0, 1.0, float(hb)])  # measurement
    Hx = np.array([1.0, 1.0, 0.0])  # readout: the fast state x = m + f
    I3 = np.eye(3)
    gam = p["sw"]
    Ar = I3.copy()
    Ar[0, 0], Ar[1, 0] = 1 - gam, gam
    o = np.array([gam * MU0, -gam * MU0, 0.0])
    sw = switch_flags(coh)

    def F(phi):
        return np.array([[1.0, 0.0, 0.0], [0.0, phi, float(fb)], [0.0, 0.0, lam_o]])

    phi1 = phL if low[0] else phH
    x = np.array([MU0, 0.0, 0.0])
    if BOUNDS in ("v2", "v3"):  # design variance in log units (normalised)
        P = np.diag(
            [
                MEM0 if INIT == "gs" else V0 / p["sig"] ** 2,
                0.0 if INIT == "gs" else DEV0 * V0 / p["sig"] ** 2,
                q3_o / (1 - lam_o**2) if obs_b else 0.0,
            ]
        )
    else:
        P = np.diag([V0, p["q1"] / (1 - phi1**2), q3_o / (1 - lam_o**2) if obs_b else 0.0])
    u, g, R = np.empty(n), np.empty(n), np.empty(n)
    c_r, A, K = np.empty((n, 3)), np.empty((n, 3, 3)), np.empty((n, 3))
    for t in range(n):
        cc = coh[t]
        Q = np.diag(
            [
                p["q2"] * np.exp(p["aq2"] * (1 - cc)),
                p["q1"] * np.exp(p["aq1"] * (1 - cc)),
                q3_o * np.exp(p["aq3"] * (1 - cc)),
            ]
        )
        if QDIAG:  # diag(q1, q2) in (x, m): see GKF_QDIAG
            Q[0, 1] = Q[1, 0] = -Q[0, 0]
            Q[1, 1] = Q[1, 1] + Q[0, 0]
        if sw[t] and spec.s == 8:  # S8: more process noise on a switch trial
            Q = Q / (1 - gam)
        R[t] = p["rlow"] if low[t] else 1.0  # measurement variance, actual and believed
        if sw[t] and spec.s == 9:  # S9: noisier measurement on a switch trial
            R[t] = R[t] * np.exp(ETA_R_MAX * gam)
        A3 = I3
        f_prev = x[1]  # previous fast deviation f+_{t-1} (readout persistence, GKF_READOUT)
        if sw[t] and spec.s == 3:  # S3: the slow state reverts before the prediction
            x, P, A3 = Ar @ x + o, Ar @ P @ Ar.T, Ar
        if sw[t] and tcode in SWITCH_CODES:  # S5: free persistence on a switch trial
            Fi = F(phi_switch(th))
        else:
            Fi = F(0.0 if sw[t] and tcode in RESET_CODES else phL if low[t] else phH)  # reset (S4)
        phr = 0.0
        if READOUT:  # persistence only at the readout: the memory filter runs with phi = 0
            phr = Fi[1, 1]
            Fi = F(0.0)
        xp, Pp = Fi @ x, Fi @ P @ Fi.T + Q
        if EFF_GAMMA:  # fast efficient encoding (GKF_EFFENC)
            R[t] = R[t] * min((H @ Pp @ H) * p["sig"] ** 2 / V0, 1.0) ** EFF_GAMMA
        k = Pp @ H / (H @ Pp @ H + R[t])
        if sw[t] and spec.s == GAIN_CUT:  # first submission's S2: gain scaled down
            k = k * (1 - gam)
        g[t] = Hx @ k
        P = Pp - np.outer(k, H @ Pp)
        x = xp + k * (z[t] - H @ xp)
        u[t] = Hx @ x + (1 - g[t]) * phr * f_prev
        c_r[t] = (Hx - g[t] * H) @ Fi @ A3 + (1 - g[t]) * phr * np.array([0.0, 1.0, 0.0])
        K[t] = k
        A[t] = (I3 - np.outer(k, H)) @ Fi @ A3
    return u, c_r, g, A, K, R


def response_mean(th, spec, logT, coh, tau):
    p = unpack(th)
    u = observer(th, spec, to_scale(logT), coh)[0]
    return report(u) + p["d0"] + p["ad0"] * (1 - coh) + p["b1"] * tau + p["b2"] * (tau ** 2 - 1 / 3)


def noise_sd(p, T, coh, tau=0.0):
    return p["sde"] * (T / 1.2) ** p["h"] * np.exp(p["kappa"] * (0.5 - coh) / 0.4) * np.exp(p["sdt"] * tau)


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(24)
GH_LW = np.log(GH_W)
ADAPT_STEPS = 6


def gh_update(mx, Cx, a, E, r, se2, xp, dv=None):
    """One response of the "gh" likelihood. X = (dx, eps, b) ~ N(mx, Cx); the log report deviation is delta = a X and
    r = y - E - offsets = E expm1(delta) + d + e, with d = dv X the bias in seconds (B3; dv = 0 otherwise). p(r) is integrated over delta by Gauss-Hermite (exact given a
    Gaussian X; d is Gaussian given delta); the posterior of X, a mixture over the nodes, is moment-matched.
    Returns (posterior mean, posterior covariance, log p(r), PIT-free standardised residual)."""
    if dv is None:
        dv = xp.array([0.0, 0.0, 0.0, 0.0, 1.0])
    ca = Cx @ a
    s2 = a @ Cx @ a + 1e-14
    mu = a @ mx
    Cd = Cx - xp.outer(ca, ca) / s2                                       # Cov[X | delta]
    Cdv = Cd @ dv
    vy = dv @ Cdv + se2
    md = dv @ mx
    b4 = (dv @ ca) / s2                                                   # E[d | delta] = md + b4 (delta - mu)
    # adaptive quadrature: the integrand over delta can be much narrower than its prior (small response noise), so
    # the nodes are placed around its mode, found by Gauss-Newton from the linearised solution
    d0 = mu
    for _ in range(ADAPT_STEPS):
        mres = r - E * xp.expm1(d0) - md - b4 * (d0 - mu)
        slope = E * xp.exp(d0) + b4
        prec = 1 / s2 + slope ** 2 / vy
        d0 = d0 + (-(d0 - mu) / s2 + mres * slope / vy) / prec
    sd = 1 / xp.sqrt(1 / s2 + (E * xp.exp(d0) + b4) ** 2 / vy)
    dk = d0 + sd * GH_X
    mk = mx[None, :] + ((dk - mu) / s2)[:, None] * ca[None, :]            # E[X | delta_k]
    ek = r - E * xp.expm1(dk) - mk @ dv
    lw = (GH_LW + 0.5 * GH_X ** 2 + xp.log(sd) - 0.5 * (xp.log(2 * xp.pi * s2) + (dk - mu) ** 2 / s2)
          - 0.5 * (xp.log(2 * xp.pi * vy) + ek * ek / vy))
    lmax = lw.max()
    ll = lmax + xp.log(xp.exp(lw - lmax).sum())
    w = xp.exp(lw - ll)
    kk = Cdv / vy
    mk = mk + ek[:, None] * kk[None, :]
    Cp = Cd - xp.outer(kk, Cdv)
    mean = w @ mk
    dev = mk - mean[None, :]
    Cn = Cp + (dev * w[:, None]).T @ dev
    zres = (w @ ek) / xp.sqrt(vy)
    return mean, Cn, ll, zres


def loglik(
    th,
    spec: Spec,
    logT,
    coh,
    tau,
    y,
    mask,
    return_innovations=False,
    response_noise_scale=None,
):
    """Per-trial log p(y_i | y_<i) (trials with mask False contribute 0).

    ``response_noise_scale`` optionally multiplies the fitted readout-noise SD on
    each trial. Omitting it preserves the fitted model exactly.
    """
    p = unpack(th)
    logT, coh = np.asarray(logT, float), np.asarray(coh, float)
    n = len(logT)
    u, c_r, g, A, K, R = observer(th, spec, to_scale(logT), coh)
    E = report(u)
    resid = np.where(mask, y, 0.0) - (
        E + p["d0"] + p["ad0"] * (1 - coh) + p["b1"] * tau + p["b2"] * (tau**2 - 1 / 3)
    )
    Ed = np.ones(n) if LIN else E  # d report / d u
    sig = (
        p["sig"] * np.sqrt(R) * (0.0 if NOISELESS else 1.0)
    )  # actual sensory noise (none under GKF_NOISELESS)
    se = noise_sd(p, np.exp(logT), coh, tau)
    if response_noise_scale is not None:
        response_noise_scale = np.asarray(response_noise_scale, float)
        if response_noise_scale.shape != (n,):
            raise ValueError("response_noise_scale must have one value per trial")
        if not np.all(np.isfinite(response_noise_scale) & (response_noise_scale > 0)):
            raise ValueError("response_noise_scale must be finite and positive")
        se = se * response_noise_scale
    _, _, hb, rb, sb, lat = b_flags(spec.b)
    rho = p["lam"] if lat else 0.0
    q3l = p["q3s"] if sb else p["q3"]  # latent bias variance in its own units
    vd = q3l * np.exp(p["aq3"] * (1 - coh)) if lat else np.zeros(n)
    sb_ = 1.0 if sb else p["sig"]  # b in seconds (B3), or normalised log units
    m = np.zeros(4)
    C = np.diag([0.0, 0.0, 0.0, q3l / (1 - rho**2) if lat else 0.0])
    M = np.zeros((5, 4))
    M[:3, :3] = np.eye(3)
    M[4, 3] = rho
    ll, zs = np.zeros(n), np.full(n, np.nan)
    mm = LIK == "mm" and not LIN
    e4 = np.r_[0.0, 0.0, 0.0, 0.0, float(sb)]  # the seconds bias (B3) in the report
    for t in range(n):
        W = np.diag([0.0, 0.0, 0.0, 1.0, vd[t]])
        mx, Cx = M @ m, M @ C @ M.T + W  # X = (dx+_{t-1}, eps_t, b_t)
        a = np.r_[c_r[t], g[t] * sig[t], (g[t] * hb + rb) * sb_]  # log report deviation = a X
        if mm:
            s2 = min(a @ Cx @ a, S2_MAX)
            Mx = E[t] * np.exp(a @ mx + s2 / 2)  # E exp(u + a X)
            h = Mx * a + e4  # Cov(X, y) = Cx h
            mean, extra = Mx - E[t] + e4 @ mx, Mx**2 * (np.expm1(s2) - s2)
        else:
            h = Ed[t] * a + e4
            mean, extra = h @ mx, 0.0
        if mask[t] and LIK == "gh" and not LIN:
            mx, Cx, ll[t], zs[t] = gh_update(mx, Cx, a, E[t], resid[t], se[t] ** 2, np, e4)
        elif mask[t]:
            S = h @ Cx @ h + extra + se[t] ** 2
            e = resid[t] - mean
            ll[t] = -0.5 * (np.log(2 * np.pi * S) + e * e / S)
            zs[t] = e / np.sqrt(S)
            kk = Cx @ h / S
            mx, Cx = mx + kk * e, Cx - np.outer(kk, h @ Cx)
        N = np.zeros((4, 5))
        N[:3, :3], N[:3, 3], N[:3, 4], N[3, 4] = A[t], K[t] * sig[t], K[t] * hb * sb_, 1.0
        m, C = N @ mx, N @ Cx @ N.T
    return (ll, zs) if return_innovations else ll


def measurements(th, T, coh, rng, spec=None):
    """Internal measurements z_i = log T_i + sigma_s sqrt(R_i) eps_i (E2); T_i + ... under "lin"."""
    p = unpack(th)
    R = np.where(np.asarray(coh, float) < 0.5, p["rlow"], 1.0)
    if spec is not None and spec.s == 9:
        R = np.where(switch_flags(coh), R * np.exp(ETA_R_MAX * p["sw"]), R)
    if EFF_GAMMA:  # R_i follows the predictive covariance, which ignores z
        assert spec is not None, "GKF_EFFENC needs the model spec to generate measurements"
        R = observer(th, spec, np.zeros(len(R)), coh)[5]
    if NOISELESS:  # the measurement is the stimulus (R only in the gain)
        return to_scale(np.log(np.asarray(T, float)))
    return to_scale(np.log(np.asarray(T, float))) + p["sig"] * np.sqrt(R) * rng.standard_normal(
        len(R)
    )


def simulate(th, spec: Spec, T, coh, tau, valid, rng, return_mean=True):
    """Exact generative simulation; optionally skip the separate noise-free-mean pass for bulk predictive draws."""
    p = unpack(th)
    T, coh = np.asarray(T, float), np.asarray(coh, float)
    n = len(T)
    _, _, hb, rb, sb, lat = b_flags(spec.b)
    b = np.zeros(n)
    if lat:                                       # the latent bias: stationary pre-first state, then transitions
        q3l = p["q3s"] if sb else p["q3"]                       # latent bias variance in its own units
        vd = q3l * np.exp(p["aq3"] * (1 - coh))
        prev = rng.standard_normal() * np.sqrt(q3l / (1 - p["lam"] ** 2))
        for t in range(n):
            prev = b[t] = p["lam"] * prev + rng.standard_normal() * np.sqrt(vd[t])
        if not sb:
            b = b * p["sig"]                      # normalised -> log units
    z = measurements(th, T, coh, rng, spec) + hb * b
    u = observer(th, spec, z, coh)[0]
    offs = p["d0"] + p["ad0"] * (1 - coh) + p["b1"] * tau + p["b2"] * (tau ** 2 - 1 / 3)
    y = report(u + rb * b) + offs + sb * b + rng.standard_normal(n) * noise_sd(p, T, coh, tau)
    mean = response_mean(th, spec, np.log(T), coh, tau) if return_mean else None
    return np.where(np.asarray(valid, bool), y, np.nan), mean


# ---------------------------------------------------------------- JAX


def jax_funcs():
    """JAX likelihood for every model: nll(theta, code, logT, coh, tau, y, mask) with code = Spec.vec().
    One compilation serves all models (the model enters through code and through the fixed entries of theta).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    mm = LIK == "mm" and not LIN

    def trial_ll(th, code, logT, coh, tau, y, mask):
        tcode, scode, bcode = code[0], code[1], code[2]
        p = unpack(th, jnp)
        phL, phH = weights(th, tcode, jnp)
        low = coh < 0.5
        fl = jnp.asarray(B_ARR)[bcode]
        fb, hb, rb, sb = fl[0], fl[1], fl[2], fl[3]
        obs_b = jnp.maximum(fb, hb)
        lat = jnp.maximum(jnp.maximum(hb, rb), sb)
        lam_o, q3_o = obs_b * p["lam"], obs_b * p["q3"]
        H = jnp.array([1.0, 1.0, 0.0]).at[2].set(hb)
        Hx = jnp.array([1.0, 1.0, 0.0])
        I3 = jnp.eye(3)
        gam = p["sw"]
        Ar = I3.at[0, 0].set(1 - gam).at[1, 0].set(gam)
        o = jnp.array([gam * MU0, -gam * MU0, 0.0])
        swf = jnp.concatenate([jnp.zeros(1, bool), jnp.abs(coh[1:] - coh[:-1]) > 1e-9])
        reset = (tcode >= RESET_CODES[0]) & (tcode <= RESET_CODES[-1])
        swfree = tcode >= SWITCH_CODES[0]
        phsw = phi_switch(th, jnp)

        def F(phi):
            return (
                jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                .at[1, 1]
                .set(phi)
                .at[1, 2]
                .set(fb)
                .at[2, 2]
                .set(lam_o)
            )

        phi1 = jnp.where(low[0], phL, phH)
        x0 = jnp.array([MU0, 0.0, 0.0])
        if BOUNDS in ("v2", "v3"):
            P0 = jnp.diag(
                jnp.array(
                    [
                        MEM0 if INIT == "gs" else V0 / p["sig"] ** 2,
                        0.0 if INIT == "gs" else DEV0 * V0 / p["sig"] ** 2,
                        q3_o / (1 - lam_o**2),
                    ]
                )
            )
        else:
            P0 = jnp.diag(jnp.array([V0, p["q1"] / (1 - phi1**2), q3_o / (1 - lam_o**2)]))

        def obs_step(carry, inp):
            x, P = carry
            z, cc, lo_, w = inp
            Q = jnp.diag(
                jnp.array(
                    [
                        p["q2"] * jnp.exp(p["aq2"] * (1 - cc)),
                        p["q1"] * jnp.exp(p["aq1"] * (1 - cc)),
                        q3_o * jnp.exp(p["aq3"] * (1 - cc)),
                    ]
                )
            )
            if QDIAG:  # diag(q1, q2) in (x, m): see GKF_QDIAG
                Q = Q.at[0, 1].set(-Q[0, 0]).at[1, 0].set(-Q[0, 0]).at[1, 1].add(Q[0, 0])
            Q = jnp.where(
                w & (scode == 8), Q / (1 - gam), Q
            )  # S8: more process noise on a switch trial
            R = jnp.where(lo_, p["rlow"], 1.0)  # measurement variance, actual and believed
            R = R * jnp.where(
                w & (scode == 9), jnp.exp(ETA_R_MAX * gam), 1.0
            )  # S9: noisier switch measurement
            f_prev = x[1]  # previous fast deviation (GKF_READOUT)
            pre = w & (scode == 3)  # S3: reversion before the prediction
            A3 = jnp.where(pre, Ar, I3)
            x = jnp.where(pre, Ar @ x + o, x)
            P = jnp.where(pre, Ar @ P @ Ar.T, P)
            Fi = F(
                jnp.where(w & swfree, phsw, jnp.where(w & reset, 0.0, jnp.where(lo_, phL, phH)))
            )  # S5, S4
            phr = Fi[1, 1] if READOUT else 0.0
            if READOUT:  # persistence only at the readout
                Fi = F(0.0)
            xp, Pp = Fi @ x, Fi @ P @ Fi.T + Q
            if EFF_GAMMA:  # fast efficient encoding (GKF_EFFENC)
                R = R * jnp.minimum((H @ Pp @ H) * p["sig"] ** 2 / V0, 1.0) ** EFF_GAMMA
            k = Pp @ H / (H @ Pp @ H + R)
            k = jnp.where(
                w & (scode == GAIN_CUT), k * (1 - gam), k
            )  # first submission's S2: gain scaled down
            g = Hx @ k
            IKH = I3 - jnp.outer(k, H)
            xn, Pn = xp + k * (z - H @ xp), Pp - jnp.outer(k, H @ Pp)
            u = Hx @ xn + (1 - g) * phr * f_prev
            c = (Hx - g * H) @ Fi @ A3 + (1 - g) * phr * jnp.array([0.0, 1.0, 0.0])
            return (xn, Pn), (u, c, g, IKH @ Fi @ A3, k, R)

        _, (u, c_r, g, A, K, R) = jax.lax.scan(
            obs_step, (x0, P0), (to_scale(logT, jnp), coh, low, swf)
        )
        E = report(u, jnp)
        resid = jnp.where(mask, y, 0.0) - (
            E + p["d0"] + p["ad0"] * (1 - coh) + p["b1"] * tau + p["b2"] * (tau**2 - 1 / 3)
        )
        sig = (
            p["sig"] * jnp.sqrt(R) * (0.0 if NOISELESS else 1.0)
        )  # actual sensory noise (GKF_NOISELESS)
        se = (
            p["sde"]
            * (jnp.exp(logT) / 1.2) ** p["h"]
            * jnp.exp(p["kappa"] * (0.5 - coh) / 0.4)
            * jnp.exp(p["sdt"] * tau)
        )
        rho = lat * p["lam"]
        q3l = jnp.where(sb > 0, p["q3s"], p["q3"])  # latent bias variance in its own units
        vd = lat * q3l * jnp.exp(p["aq3"] * (1 - coh))
        sb_ = sb + (1 - sb) * p["sig"]  # b in seconds (B3), or normalised log units
        e4 = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0]) * sb
        M = jnp.zeros((5, 4)).at[:3, :3].set(jnp.eye(3)).at[4, 3].set(rho)

        def lik_step(carry, inp):
            m, C = carry
            rt, mt, st, Et, ct, gt, At, Kt, sgt, vdt = inp
            W = jnp.diag(jnp.array([0.0, 0.0, 0.0, 1.0, vdt]))
            mx, Cx = M @ m, M @ C @ M.T + W
            a = jnp.concatenate([ct, jnp.array([gt * sgt, (gt * hb + rb) * sb_])])
            if mm:
                s2 = jnp.minimum(a @ Cx @ a, S2_MAX)
                Mx = Et * jnp.exp(a @ mx + s2 / 2)
                h = Mx * a + e4
                mean, extra = Mx - Et + e4 @ mx, Mx**2 * (jnp.expm1(s2) - s2)
            else:
                h = (1.0 if LIN else Et) * a + e4
                mean, extra = h @ mx, 0.0
            S = h @ Cx @ h + extra + st**2
            e = rt - mean
            ll = jnp.where(mt, -0.5 * (jnp.log(2 * jnp.pi * S) + e * e / S), 0.0)
            kk = jnp.where(mt, Cx @ h / S, 0.0)
            if LIK == "gh" and not LIN:  # adaptive Gauss-Hermite (gkf.gh_update)
                mg, Cg, llg, _ = gh_update(mx, Cx, a, Et, rt, st**2, jnp, e4)
                ll = jnp.where(mt, llg, 0.0)
                mx, Cx = jnp.where(mt, mg, mx), jnp.where(mt, Cg, Cx)
            else:
                mx, Cx = mx + kk * e, Cx - jnp.outer(kk, h @ Cx)
            N = (
                jnp.zeros((4, 5))
                .at[:3, :3]
                .set(At)
                .at[:3, 3]
                .set(Kt * sgt)
                .at[:3, 4]
                .set(Kt * hb * sb_)
                .at[3, 4]
                .set(1.0)
            )
            return (N @ mx, N @ Cx @ N.T), ll

        C0 = jnp.diag(jnp.array([0.0, 0.0, 0.0, lat * q3l / (1 - rho**2)]))
        _, ll = jax.lax.scan(
            lik_step, (jnp.zeros(4), C0), (resid, mask, se, E, c_r, g, A, K, sig, vd)
        )
        return ll  # per-trial log p(y_i | y_<i); 0 where not scored

    def nll(th, code, logT, coh, tau, y, mask):
        return -trial_ll(th, code, logT, coh, tau, y, mask).sum()

    # ll: per-trial log-likelihoods (2026-11-03, for PSIS-LOO of the hierarchical fits)
    return dict(
        jnp=jnp, nll=jax.jit(nll), vg=jax.jit(jax.value_and_grad(nll)), ll=jax.jit(trial_ll)
    )
