"""
layer4/sensitivity_analysis.py
────────────────────────────────
Analytic 4: OAT (One-At-A-Time) Sensitivity Analysis.
Re-runs the Layer 3 model under parameter perturbations from the Scenario A baseline.

The enriched CSV path is injected by the caller — no hardcoded paths here.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

from layer4.config import SCENARIO_CONFIGS, SUPPORT_MODEL_PARAMS

# Scenario A is the OAT baseline
BASE_CONFIG = copy.deepcopy(SCENARIO_CONFIGS["A"])

PARAM_RANGES: dict[str, tuple] = {
    "training_intensity": (0.00, 1.00),
    "manager_signal":     (0.20, 0.80),
    "tool_complexity":    (0.80, 0.40),
}

PARAM_LABELS: dict[str, str] = {
    "training_intensity": "Training Intensity",
    "manager_signal":     "Manager Signal",
    "tool_complexity":    "Tool Complexity (lower = easier)",
}

SENSITIVITY_MC_RUNS = 5
SENSITIVITY_STEPS   = 52


def _run_single_config(
    config: dict,
    csv_path: str | Path,
    n_runs: int = 5,
    n_steps: int = 52,
) -> float:
    """Run the Layer 3 model with a given config, return mean Week-51 adoption."""
    from layer3.model import WorkforceModel

    adoptions = []
    for seed in range(n_runs):
        model  = WorkforceModel(config, rng=seed, csv_path=csv_path)
        result = model.run(n_steps)
        adoptions.append(result["adoption_rate"].iloc[-1])

    return float(np.mean(adoptions))


def run_oat_sensitivity(
    csv_path: str | Path,
    n_runs: int = SENSITIVITY_MC_RUNS,
) -> list[dict]:
    """OAT sensitivity: test each param at low & high, holding others at baseline."""
    results: list[dict] = []

    for param, (low_val, high_val) in PARAM_RANGES.items():
        print(f"    Testing {param}...")

        config_low          = BASE_CONFIG.copy()
        config_low[param]   = low_val
        adoption_low        = _run_single_config(config_low, csv_path, n_runs)

        config_high         = BASE_CONFIG.copy()
        config_high[param]  = high_val
        adoption_high       = _run_single_config(config_high, csv_path, n_runs)

        swing = abs(adoption_high - adoption_low)
        results.append({
            "param":        param,
            "label":        PARAM_LABELS.get(param, param),
            "low_val":      low_val,
            "high_val":     high_val,
            "adoption_low":  round(adoption_low, 4),
            "adoption_high": round(adoption_high, 4),
            "swing_pp":      round(swing * 100, 1),
        })

    # Categorical: chatbot vs human
    print("    Testing support_model...")
    config_cb                 = BASE_CONFIG.copy()
    config_cb["support_model"] = "chatbot"
    adoption_cb               = _run_single_config(config_cb, csv_path, n_runs)

    config_hu                 = BASE_CONFIG.copy()
    config_hu["support_model"] = "human"
    adoption_hu               = _run_single_config(config_hu, csv_path, n_runs)

    results.append({
        "param":        "support_model",
        "label":        "Support Model (chatbot→human)",
        "low_val":      "chatbot",
        "high_val":     "human",
        "adoption_low":  round(adoption_cb, 4),
        "adoption_high": round(adoption_hu, 4),
        "swing_pp":      round(abs(adoption_hu - adoption_cb) * 100, 1),
    })

    results.sort(key=lambda x: -x["swing_pp"])
    return results


def generate_sensitivity_narrative(sensitivity_results: list[dict]) -> str:
    lines = []
    lines.append(f"\n{'=' * 75}")
    lines.append("SENSITIVITY ANALYSIS — TORNADO CHART (Analytic 4)")
    lines.append(f"{'=' * 75}")
    lines.append("Method: One-At-A-Time (OAT) from Scenario A baseline.")
    lines.append(f"MC runs per config: {SENSITIVITY_MC_RUNS} | Steps: {SENSITIVITY_STEPS}")
    lines.append("")

    lines.append(
        f"{'Parameter':<35} {'Low→High':>15} {'Adopt Low':>11} {'Adopt High':>11} {'Swing':>10}"
    )
    lines.append("─" * 84)

    for r in sensitivity_results:
        low_s = str(r["low_val"]) if isinstance(r["low_val"], str) else f"{r['low_val']:.2f}"
        hi_s  = str(r["high_val"]) if isinstance(r["high_val"], str) else f"{r['high_val']:.2f}"
        lines.append(
            f"{r['label']:<35} "
            f"{low_s + '→' + hi_s:>15} "
            f"{r['adoption_low']:>10.1%} "
            f"{r['adoption_high']:>10.1%} "
            f"±{r['swing_pp']:>7.1f} pp"
        )

    lines.append("")
    lines.append(f"{'─' * 75}")
    lines.append("ANALYSIS")
    lines.append(f"{'─' * 75}")

    nonzero = [r for r in sensitivity_results if r["swing_pp"] > 0.0]
    zero    = [r for r in sensitivity_results if r["swing_pp"] == 0.0]

    if nonzero:
        top = nonzero[0]
        lines.append(f"\n🎯  BIGGEST LEVER: {top['label']} (±{top['swing_pp']:.1f} pp)")
        lines.append(f"   {top['param']}: {top['low_val']} → {top['high_val']}")
        lines.append(f"   Adoption: {top['adoption_low']:.1%} → {top['adoption_high']:.1%}")

        if top["param"] == "support_model":
            cp = SUPPORT_MODEL_PARAMS.get("chatbot", {})
            hp = SUPPORT_MODEL_PARAMS.get("human", {})
            lines.append("\n   ROOT CAUSE TRACE:")
            lines.append(
                f"   Chatbot: support_drag={cp.get('support_drag')}, "
                f"adoption_friction={cp.get('adoption_friction')}, p_fail={cp.get('p_fail')}"
            )
            lines.append(
                f"   Human:   support_drag={hp.get('support_drag')}, "
                f"adoption_friction={hp.get('adoption_friction')}, p_fail={hp.get('p_fail')}"
            )
            lines.append("   Switching removes all three suppression factors simultaneously.")

        for r in nonzero[1:]:
            lines.append(f"\n   SECONDARY LEVER: {r['label']} (±{r['swing_pp']:.1f} pp)")
    else:
        lines.append("\n⚠️  NO PARAMETERS PRODUCED MEASURABLE SWING")

    if zero:
        lines.append(f"\n⚠️  ZERO-IMPACT PARAMETERS ({len(zero)} found):")
        for r in zero:
            lines.append(f"   • {r['label']}: {r['low_val']} → {r['high_val']} = 0.0 pp")

    lines.append(f"\n{'─' * 75}")
    lines.append("STRATEGIC IMPLICATION")
    lines.append(f"{'─' * 75}")
    if nonzero:
        top = nonzero[0]
        lines.append(f"  The '{top['label']}' parameter is the GATEKEEPER.")
        lines.append("  Other levers only become effective AFTER this bottleneck is removed.")
    else:
        lines.append("  No single parameter produces measurable change. Consider multi-lever strategy.")

    lines.append(f"\n{'=' * 75}")
    return "\n".join(lines)