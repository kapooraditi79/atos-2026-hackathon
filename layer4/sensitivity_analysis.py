"""
Layer 4 — Analytic 4: OAT Sensitivity Analysis
One-At-A-Time parameter sweep from Scenario A baseline.
"""
import sys
import numpy as np
from config import PROJECT_ROOT, SCENARIO_CONFIGS, SUPPORT_MODEL_PARAMS


# Scenario A baseline (the "control" config)
BASE_CONFIG = SCENARIO_CONFIGS['A'].copy()

# Parameter ranges: (pessimistic, optimistic)
PARAM_RANGES = {
    'training_intensity' : (0.00, 1.00),
    'manager_signal'     : (0.20, 0.80),
    'tool_complexity'    : (0.80, 0.40),
}

PARAM_LABELS = {
    'training_intensity' : 'Training Intensity',
    'manager_signal'     : 'Manager Signal',
    'tool_complexity'    : 'Tool Complexity (lower = easier)',
}

SENSITIVITY_MC_RUNS = 5
SENSITIVITY_STEPS   = 52


def _run_single_config(config, n_runs=5, n_steps=52):
    """Run the Layer 3 model with a given config, return mean Week-51 adoption."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from layer3.model import WorkforceModel

    adoptions = []
    for seed in range(n_runs):
        model = WorkforceModel(config, rng=seed)
        result = model.run(n_steps)
        adoptions.append(result['adoption_rate'].iloc[-1])

    return float(np.mean(adoptions))


def run_oat_sensitivity(n_runs=5):
    """OAT sensitivity: test each param at low & high while holding others at baseline."""
    results = []

    for param, (low_val, high_val) in PARAM_RANGES.items():
        print(f"    Testing {param}...")
        config_low = BASE_CONFIG.copy()
        config_low[param] = low_val
        adoption_low = _run_single_config(config_low, n_runs=n_runs)

        config_high = BASE_CONFIG.copy()
        config_high[param] = high_val
        adoption_high = _run_single_config(config_high, n_runs=n_runs)

        swing = abs(adoption_high - adoption_low)
        results.append({
            'param': param,
            'label': PARAM_LABELS.get(param, param),
            'low_val': low_val, 'high_val': high_val,
            'adoption_low': round(adoption_low, 4),
            'adoption_high': round(adoption_high, 4),
            'swing_pp': round(swing * 100, 1),
        })

    # Categorical: chatbot vs human
    print(f"    Testing support_model...")
    config_cb = BASE_CONFIG.copy()
    config_cb['support_model'] = 'chatbot'
    adoption_cb = _run_single_config(config_cb, n_runs=n_runs)

    config_hu = BASE_CONFIG.copy()
    config_hu['support_model'] = 'human'
    adoption_hu = _run_single_config(config_hu, n_runs=n_runs)

    results.append({
        'param': 'support_model',
        'label': 'Support Model (chatbot→human)',
        'low_val': 'chatbot', 'high_val': 'human',
        'adoption_low': round(adoption_cb, 4),
        'adoption_high': round(adoption_hu, 4),
        'swing_pp': round(abs(adoption_hu - adoption_cb) * 100, 1),
    })

    results.sort(key=lambda x: -x['swing_pp'])
    return results


def generate_sensitivity_narrative(sensitivity_results):
    """
    Data-driven tornado chart narrative.
    Analysis section uses the actual results — no hardcoded agent.py constants.
    """
    lines = []
    lines.append(f"\n{'='*75}")
    lines.append("SENSITIVITY ANALYSIS — TORNADO CHART (Analytic 4)")
    lines.append(f"{'='*75}")
    lines.append("Method: One-At-A-Time (OAT) from Scenario A baseline.")
    lines.append(f"MC runs per config: {SENSITIVITY_MC_RUNS} | Steps: {SENSITIVITY_STEPS}")
    lines.append("")

    lines.append(f"{'Parameter':<35} {'Low→High':>15} {'Adopt Low':>11} {'Adopt High':>11} {'Swing':>10}")
    lines.append("─" * 84)

    for r in sensitivity_results:
        low_s = str(r['low_val']) if isinstance(r['low_val'], str) else f"{r['low_val']:.2f}"
        hi_s = str(r['high_val']) if isinstance(r['high_val'], str) else f"{r['high_val']:.2f}"
        lines.append(
            f"{r['label']:<35} "
            f"{low_s + '→' + hi_s:>15} "
            f"{r['adoption_low']:>10.1%} "
            f"{r['adoption_high']:>10.1%} "
            f"±{r['swing_pp']:>7.1f} pp"
        )

    lines.append("")

    # ── Data-driven analysis ──────────────────────────────────────────
    lines.append(f"{'─'*75}")
    lines.append("ANALYSIS")
    lines.append(f"{'─'*75}")

    nonzero = [r for r in sensitivity_results if r['swing_pp'] > 0.0]
    zero = [r for r in sensitivity_results if r['swing_pp'] == 0.0]

    if nonzero:
        top = nonzero[0]
        lines.append(f"\n🎯 BIGGEST LEVER: {top['label']} (±{top['swing_pp']:.1f} pp)")
        lines.append(f"   {top['param']}: {top['low_val']} → {top['high_val']}")
        lines.append(f"   Adoption: {top['adoption_low']:.1%} → {top['adoption_high']:.1%}")

        # If support_model is the top lever, explain using actual SUPPORT_MODEL_PARAMS
        if top['param'] == 'support_model':
            chatbot_p = SUPPORT_MODEL_PARAMS.get('chatbot', {})
            human_p = SUPPORT_MODEL_PARAMS.get('human', {})
            lines.append("")
            lines.append("   ROOT CAUSE TRACE (Layer 3 agent mechanics):")
            lines.append(f"   Chatbot: support_drag={chatbot_p.get('support_drag')}, "
                         f"adoption_friction={chatbot_p.get('adoption_friction')}, "
                         f"p_fail={chatbot_p.get('p_fail')}")
            lines.append(f"   Human:   support_drag={human_p.get('support_drag')}, "
                         f"adoption_friction={human_p.get('adoption_friction')}, "
                         f"p_fail={human_p.get('p_fail')}")
            lines.append(f"   Switching removes all three suppression factors simultaneously.")

        # Additional nonzero levers
        for r in nonzero[1:]:
            lines.append(f"\n   SECONDARY LEVER: {r['label']} (±{r['swing_pp']:.1f} pp)")
            lines.append(f"   {r['param']}: {r['low_val']} → {r['high_val']}")
    else:
        lines.append("\n⚠️ NO PARAMETERS PRODUCED MEASURABLE SWING")
        lines.append("   All parameters are below the model's adoption threshold.")

    if zero:
        lines.append(f"\n⚠️ ZERO-IMPACT PARAMETERS ({len(zero)} found):")
        for r in zero:
            lines.append(f"   • {r['label']}: {r['low_val']} → {r['high_val']} = 0.0 pp")

        lines.append("")
        lines.append("   These parameters cannot individually overcome the current")
        lines.append("   adoption bottleneck. They may become effective only after")
        lines.append("   the highest-impact lever is applied first.")

    # Strategic implication (data-driven: reference whichever param is on top)
    lines.append("")
    lines.append(f"{'─'*75}")
    lines.append("STRATEGIC IMPLICATION")
    lines.append(f"{'─'*75}")
    if nonzero:
        top = nonzero[0]
        lines.append(f"  The '{top['label']}' parameter is the GATEKEEPER.")
        lines.append(f"  Other levers only become effective AFTER this bottleneck is removed.")
        if zero:
            lines.append(f"  RECOMMENDATION: Address {top['label']} first, then re-test the")
            lines.append(f"  {len(zero)} zero-impact parameters to find secondary levers.")
    else:
        lines.append("  No single parameter produces measurable change from the current baseline.")
        lines.append("  Consider combined parameter changes (multi-lever strategy).")

    lines.append(f"\n{'='*75}")
    return "\n".join(lines)
