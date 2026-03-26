"""
layer4/config.py
─────────────────
Shared constants for all Layer 4 analytics modules.

No file I/O, no path resolution — those live in run.py.
Paths that analytics modules need are injected by run.py at call time.

UI-editable fields (sent from the React frontend via the `scenario_configs`
form field and applied by backend/app.py before each run):

    SCENARIO_CONFIGS[key]["tool_complexity"]    — float 0–1
    SCENARIO_CONFIGS[key]["training_intensity"] — float 0–1
    SCENARIO_CONFIGS[key]["manager_signal"]     — float 0–1
    SCENARIO_CONFIGS[key]["support_model"]      — "chatbot" | "human" | "hybrid"

app.py monkey-patches this module's SCENARIO_CONFIGS dict for the duration
of each request, then restores the originals.  All other constants below
are backend-only and not exposed to the UI.
"""

# ── Scenario metadata ─────────────────────────────────────────────────────────

SCENARIO_LABELS: dict[str, str] = {
    "A": "Big-bang + Chatbot",
    "B": "Phased + Human + Training",
    "C": "Pilot + Strong Management",
}

# These are the DEFAULT values.
# At runtime, app.py overlays the values the user set via the UI sliders
# before passing control to Layer 3 and Layer 4.
SCENARIO_CONFIGS: dict[str, dict] = {
    "A": {
        # ── UI-editable ───────────────────────────────────────────────────────
        "tool_complexity":    0.65,   # cognitive load / UX difficulty (0 = trivial, 1 = very hard)
        "training_intensity": 0.10,   # fraction of budget / time spent on training
        "support_model":      "chatbot",  # first-line support channel
        "manager_signal":     0.40,   # strength of manager change-signal to reports
    },
    "B": {
        "tool_complexity":    0.65,
        "training_intensity": 0.70,
        "support_model":      "human",
        "manager_signal":     0.60,
    },
    "C": {
        "tool_complexity":    0.65,
        "training_intensity": 0.45,
        "support_model":      "hybrid",
        "manager_signal":     0.80,
    },
}

# ── Support model parameters ──────────────────────────────────────────────────
# Keyed by the support_model string used in SCENARIO_CONFIGS.
# These are NOT UI-editable — they are fixed calibrations derived from
# internal research / literature. The user controls *which* model is
# selected per scenario via the support_model field above.

SUPPORT_MODEL_PARAMS: dict[str, dict] = {
    "chatbot": {"support_drag": 0.05, "adoption_friction": 0.05, "p_fail": 0.38, "deflect": 0.45},
    "hybrid":  {"support_drag": 0.03, "adoption_friction": 0.06, "p_fail": 0.22, "deflect": 0.25},
    "human":   {"support_drag": 0.01, "adoption_friction": 0.02, "p_fail": 0.10, "deflect": 0.00},
}

# ── NPV constants ─────────────────────────────────────────────────────────────
# Not UI-editable. Changing these requires a new backend deployment.

WACC_ANNUAL    = 0.10
WEEKLY_RATE    = (1 + WACC_ANNUAL) ** (1 / 52) - 1
HEADCOUNT      = 1000
AVG_WEEKLY_SAL = 1200
P_CHURN        = 0.22
REPLACEMENT    = 45_000

INVESTMENT: dict[str, int] = {
    "A": -700_000,
    "B": -850_000,
    "C": -780_000,
}

COST_PER_TICKET: dict[str, float] = {
    "A": 26.04,
    "B": 45.0,
    "C": 45.0,
}