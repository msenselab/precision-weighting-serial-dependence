"""The standard generative Kalman model space (Appendix C.2; 2026-10-29).

Every model is one generative model of the observer (F, Q, H, R, which may vary with the cues it observes on each
trial: the current coherence and whether it switched) with standard Kalman inference: one prediction per trial, the
optimal gain, and the report taken from that one inference. The levels and their implementation in gkf.py:

    T  transition of the fast state (persistence phi in F)
       T0  phi = 0, no persistence                         gkf Tn
       T1  phi constant                                    gkf T0
       T2  phi(coh_i): logit phi +- beta, one transition    gkf T1s
           for the report and the memory
       T3 phi under Low coherence only (phi^H = 0)         gkf T1n
    S  switch of coherence
       S0  none                                            gkf S0
       S1  mean reversion of the slow state in the         gkf S3 (reset before the prediction, covariance mapped)
           transition of a switch trial
       S2  more process noise on a switch trial,           gkf S8
           Q / (1 - sw)
       S3  noisier measurement on a switch trial,          gkf S9
           R exp(eta_R), actual and believed
       S4  the fast state does not persist across a switch  gkf S0 with the reset T level: T1 -> T0S, T2 -> T1sS,
           (phi_i = 0 on switch trials, report and memory;  T3 -> T1nS; under T0 (phi = 0) S4 is S0 (duplicate)
           contextual reset; 2026-10-30, moved from the T dimension, where it was T1S and T2LS)
       S5  free persistence phi_sw of the fast state on     gkf S0 with the switch T level: T0 -> TnW, T1 -> T0W,
           switch trials (2026-11-01; generalises S4, where  T2 -> T1sW, T3 -> T1nW. An extension fitted outside
           phi_sw = 0)                                       the 760-model grid (extension_names)
    C  coherence and the precision of a level: C1-C15 (non-empty subsets of q1, q2, q3, R)
    B  the third state b and where it is coupled (gkf B_COUPLE; 2026-11-01): B1 none; B2 transition of the fast
       state (F13); B3 report, in seconds (first space); B4 measurement (H13); B5 report (log); B6 F13 + H13;
       B7 F13 + report; B8 H13 + report; B9 F13 + H13 + report. The observer updates b (K^(b) != 0) exactly when b
       enters F or H, so whether the bias is updated is decided by the comparison, not by the formulation.
    The readout (the estimate) is always the posterior of the current duration (E7).

Public names (C*k*-T*k*-S*k*-B*k*) use these levels; internal() gives the gkf Spec. Mechanisms that are not one
generative model with standard inference (separate readout and storage transitions, gain scaling, resets at one step,
readouts that mix in another estimate) are archived in the project's working repository.
"""
import re

from sdu.model import engine as G

T_MAP = {"T0": "Tn", "T1": "T0", "T2": "T1s", "T3": "T1n"}
S_MAP = {0: 0, 1: 3, 2: 8, 3: 9, 4: 0}                                 # S4: gkf S0 with the reset T level
T_RESET = {"T0": "Tn", "T1": "T0S", "T2": "T1sS", "T3": "T1nS"}
T_SWITCH = {"T0": "TnW", "T1": "T0W", "T2": "T1sW", "T3": "T1nW"}
T_PUBLIC = {v: k for k, v in T_MAP.items()}
T_SWITCH_PUBLIC = {v: k for k, v in T_SWITCH.items()}
T_RESET_PUBLIC = {v: k for k, v in T_RESET.items() if k != "T0"}
S_PUBLIC = {v: k for k, v in S_MAP.items() if k != 4}
B_LEVELS = tuple(G.B_COUPLE)                                            # 1-9
Q3_FREE = {3: 0, 7: 4, 9: 1, 10: 2, 12: 5, 13: 6, 14: 8, 15: 11}       # B1: q3 inert -> the locus set without q3


def internal(name):
    """Public name -> gkf Spec."""
    m = re.fullmatch(r"C(\d+)-(T\d)-S(\d)-B(\d)", name)
    tm = T_RESET if m and m[3] == "4" else T_SWITCH if m and m[3] == "5" else T_MAP
    assert m and m[2] in tm and (int(m[3]) in S_MAP or m[3] == "5"), name
    return G.Spec(int(m[1]), tm[m[2]], S_MAP.get(int(m[3]), 0), int(m[4]))


def public(spec):
    """gkf Spec -> public name (only for specs of this space)."""
    if spec.t in T_SWITCH_PUBLIC:
        return f"C{spec.c}-{T_SWITCH_PUBLIC[spec.t]}-S5-B{spec.b}"
    if spec.t in T_RESET_PUBLIC:
        return f"C{spec.c}-{T_RESET_PUBLIC[spec.t]}-S4-B{spec.b}"
    return f"C{spec.c}-{T_PUBLIC[spec.t]}-S{S_PUBLIC[spec.s]}-B{spec.b}"


def all_names():
    """C1-C15 x T x S x B. There is no C0: coherence is the manipulated variable, so it modulates some noise."""
    return [f"C{c}-{t}-S{s}-B{b}" for c in range(1, 16) for t in T_MAP for s in S_MAP for b in B_LEVELS]


def duplicate_of(name):
    """The public name a model duplicates, or None: a B1 model with an inert q3 locus (the locus set without q3), and
    T0-S4 (no persistence to reset: T0-S0)."""
    sp = internal(name)
    if sp.b == 1 and sp.c in Q3_FREE:
        return public(G.Spec(Q3_FREE[sp.c], sp.t, sp.s, sp.b))
    if "-T0-S4-" in name:
        return name.replace("-S4-", "-S0-")
    return None


def distinct_names():
    """Distinct models. A B1 model whose only coherence locus is q3 (C3-B1) is the no-modulation model and is
    excluded with it (duplicate_of maps it to the C0 name, which is not in the space)."""
    return [n for n in all_names() if duplicate_of(n) is None]



# Reported model space (2026-11-02): all C levels and T levels; the switch rules of the first submission's design
# (S0 none; S3, the standard counterpart of the gain change; S4, the fast-state reset) and
# the bias levels B1 (none), B3 (report, first submission), B4 (measurement) and B9 (transition, measurement, report).
# The complete crossing (all_names) is reported as a robustness check.
CORE_S = (0, 3, 4)
CORE_B = (1, 3, 4, 9)


def core_names():
    """Reported space: C1-C15 x T0-T3 x {S0, S3, S4} x {B1, B3, B4, B9}."""
    return [n for n in distinct_names()
            if int(n.split("-")[2][1:]) in CORE_S and int(n.split("-")[3][1:]) in CORE_B]


# The bias coupled into the prediction but not the report (2026-11-05): Bt (prediction only, the B2 of the original
# 135-model space) and Btm (prediction + measurement), C1-C15 x T0-T3 x {S0, S3, S4}.
PREDICTION_B = (2, 6)


def prediction_bias_names():
    """The 330 models with the bias in the prediction and not in the report (labels Bt, Btm)."""
    return [n for n in distinct_names()
            if int(n.split("-")[2][1:]) in CORE_S and int(n.split("-")[3][1:]) in PREDICTION_B]


# Labels used in the manuscript (2026-11-02): consecutive codes in the reported space, letter codes for the levels of
# the complete crossing only. The code and the result files keep the public names above.
S_LABEL = {"S0": "S0", "S3": "S1", "S4": "S2", "S1": "Sm", "S2": "Sq", "S5": "Sw"}
B_LABEL = {"B1": "B1", "B3": "B2", "B4": "B3", "B9": "B4", "B2": "Bt", "B5": "Br", "B6": "Btm", "B7": "Btr", "B8": "Bmr"}


def label(name):
    """Public name -> manuscript label, e.g. C1-T3-S0-B9 -> C1-T3-S0-B4."""
    c, t, s, b = name.split("-")
    return f"{c}-{t}-{S_LABEL[s]}-{B_LABEL[b]}"


def extension_names():
    """S5 (free persistence on switch trials), fitted outside the grid: C x T x S5 x B, distinct models."""
    return [f"C{c}-{t}-S5-B{b}" for c in range(16) for t in T_MAP for b in (1, 2, 3)
            if not (b == 1 and c in Q3_FREE)]
