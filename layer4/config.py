"""
Layer 4 — Shared Configuration & Data Loading
Shared constants, paths, and the load_and_fix() foundation.
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR   = PROJECT_ROOT / 'layer3' / 'outputs'
L4_OUT       = SCRIPT_DIR / 'outputs'
L4_OUT.mkdir(parents=True, exist_ok=True)

# ── Scenario Labels ──────────────────────────────────────────────────────────
SCENARIO_LABELS = {
    'A': 'Big-bang + Chatbot',
    'B': 'Phased + Human + Training',
    'C': 'Pilot + Strong Management',
}

# ── Scenario Configs (mirrors Layer 3 run.py) ────────────────────────────────
SCENARIO_CONFIGS = {
    'A': {
        'tool_complexity'    : 0.65,
        'training_intensity' : 0.10,
        'support_model'      : 'chatbot',
        'manager_signal'     : 0.40,
    },
    'B': {
        'tool_complexity'    : 0.65,
        'training_intensity' : 0.70,
        'support_model'      : 'human',
        'manager_signal'     : 0.60,
    },
    'C': {
        'tool_complexity'    : 0.65,
        'training_intensity' : 0.45,
        'support_model'      : 'hybrid',
        'manager_signal'     : 0.80,
    },
}

# ── Support model parameters (from agent.py) ─────────────────────────────────
# These are read dynamically by hotspot_analysis to avoid hardcoding
SUPPORT_MODEL_PARAMS = {
    'chatbot': {'support_drag': 0.08, 'adoption_friction': 0.15, 'p_fail': 0.38, 'deflect': 0.45},
    'hybrid' : {'support_drag': 0.03, 'adoption_friction': 0.06, 'p_fail': 0.22, 'deflect': 0.25},
    'human'  : {'support_drag': 0.00, 'adoption_friction': 0.00, 'p_fail': 0.10, 'deflect': 0.00},
}

# ── NPV Constants ────────────────────────────────────────────────────────────
WACC_ANNUAL     = 0.10
WEEKLY_RATE     = (1 + WACC_ANNUAL) ** (1/52) - 1
HEADCOUNT       = 1000
AVG_WEEKLY_SAL  = 1200
P_CHURN         = 0.22
REPLACEMENT     = 45_000

INVESTMENT      = {'A': -700_000, 'B': -850_000, 'C': -780_000}
COST_PER_TICKET = {'A': 26.04, 'B': 45.0, 'C': 45.0}


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_and_fix():
    """Load scenario CSVs from Layer 3 outputs, compute adoption gain."""
    scenarios = {}

    if not OUTPUT_DIR.exists():
        print(f"Error: Could not find {OUTPUT_DIR}")
        print(f"Search path attempted: {OUTPUT_DIR.resolve()}")
        return {}

    for key, fname in [('A', 'output_scenario_a.csv'),
                       ('B', 'output_scenario_b.csv'),
                       ('C', 'output_scenario_c.csv')]:
        path = OUTPUT_DIR / fname
        if not path.exists():
            print(f"Warning: {fname} missing. Skipping.")
            continue

        df = pd.read_csv(str(path))

        # FIX 4: Frame adoption as GAIN from week-0 baseline
        baseline = df['adoption_mean'].iloc[0]
        df['adoption_gain'] = df['adoption_mean'] - baseline
        df['gain_p05']      = df['adoption_p05']  - baseline
        df['gain_p95']      = df['adoption_p95']  - baseline

        # FIX 1: Verify productivity delta
        delta_col = 'productivity_delta_true'
        if delta_col in df.columns:
            if df[delta_col].mean() > 0.5:
                raise ValueError(f"Scenario {key}: {delta_col} contains absolute values, not deltas.")
            df['productivity_delta'] = df[delta_col]

        scenarios[key] = df
        print(f"Scenario {key} loaded: Gain {df['adoption_gain'].iloc[-1]:.3f}")

    return scenarios


def load_parquet(scenario_key):
    """Load agent-level parquet, reset MultiIndex to columns."""
    pfile = {'A': 'agents_a.parquet', 'B': 'agents_b.parquet', 'C': 'agents_c.parquet'}
    path = OUTPUT_DIR / pfile[scenario_key]
    agents = pd.read_parquet(str(path))
    if isinstance(agents.index, pd.MultiIndex):
        agents = agents.reset_index()
    return agents
