"""Current v3 representatives and the small comparison set used in figures."""

PRIMARY_MODELS = {2: "C3-T3-S0-B3", 1: "C1-T3-S2-B4"}

# Current representative, two nearby predictive alternatives, and the AIC-only
# winner.  The order is also the line-style order in the reproduction figure.
REPRODUCTION_MODELS = {
    2: ("C3-T3-S0-B3", "C7-T3-S0-B4", "C7-T3-S0-B3", "C3-T0-S0-B4"),
    1: ("C1-T3-S2-B4", "C3-T3-S2-B3", "C7-T3-S2-B4", "C3-T0-S0-B4"),
}
