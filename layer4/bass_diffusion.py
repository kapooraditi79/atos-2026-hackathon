"""
Layer 4 — Analytic 1: Bass Diffusion Model
Fits Bass model (p, q, M) to adoption curves and generates CIO narrative.
"""
import numpy as np
from scipy.optimize import curve_fit
from config import SCENARIO_LABELS


def bass_cumulative(t, p, q, M, F0):
    """Standard Bass Model adjusted for pre-existing adoption (F0)."""
    p = max(p, 1e-7)
    exp_term = np.exp(-(p + q) * t)
    ratio = q / p
    f_t = (1.0 - exp_term) / (1.0 + ratio * exp_term)
    return F0 + (M - F0) * f_t


def fit_bass(df, scenario_key):
    """
    Fit Bass Diffusion with widened bounds (p_max=0.8, q_max=3.0)
    to capture rapid-onset adoption in high-intensity scenarios.
    """
    t_data = df['week'].values.astype(float)
    y_data = df['adoption_mean'].values.astype(float)
    F0 = float(y_data[0])

    result = {
        'scenario': scenario_key,
        'label': SCENARIO_LABELS.get(scenario_key, scenario_key),
        'F0': round(float(F0), 4),
        'hit_bounds': False
    }

    # Flat/stalled curve detection
    if (y_data.max() - y_data.min()) < 0.005:
        result.update({'p': 0.0, 'q': 0.0, 'M': F0, 'fit_success': False, 'r_squared': 0.0})
        return result

    try:
        p0 = [0.05, 0.4, 0.9, F0]
        bounds_lower = [0.0001, 0.001, F0, F0 - 0.01]
        bounds_upper = [0.80, 3.00, 1.0, F0 + 0.01]

        popt, _ = curve_fit(bass_cumulative, t_data, y_data, p0=p0,
                            bounds=(bounds_lower, bounds_upper))
        p_fit, q_fit, M_fit, F0_fit = popt

        if p_fit > 0.78 or q_fit > 2.94:
            result['hit_bounds'] = True

        y_pred = bass_cumulative(t_data, *popt)
        r2 = 1 - (np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2))

        t_star = np.log(q_fit / p_fit) / (p_fit + q_fit) if q_fit > p_fit else 0

        result.update({
            'p': round(p_fit, 5), 'q': round(q_fit, 5), 'M': round(M_fit, 4),
            't_star': round(max(0, t_star), 1), 'r_squared': round(r2, 4),
            'fit_success': True, 'fitted_curve': [round(v, 6) for v in y_pred]
        })
    except Exception as e:
        result.update({'fit_success': False, 'note': str(e)})

    return result


def generate_cio_narrative(bass_results):
    """
    Data-driven CIO narrative. No hardcoded scenario references —
    all text is derived from the fitted p, q, t*, M values.
    """
    lines = []
    for key, bp in bass_results.items():
        lines.append(f"\n{'='*70}")
        lines.append(f"SCENARIO {key}: {bp['label']}")
        lines.append(f"{'='*70}")

        if not bp['fit_success']:
            # Stalled adoption — derive text from actual data
            lines.append("STRATEGIC PERSONA: 'The Ghost Town'")
            lines.append(f"ANALYSIS: Adoption stalled at baseline ({bp['F0']:.1%}).")
            gain = bp['M'] - bp['F0']
            lines.append(f"INSIGHT: Zero adoption gain detected (Δ = {gain:.1%}). The rollout")
            lines.append(f"         configuration did not produce measurable behaviour change.")
            continue

        p, q, ts, r2 = bp['p'], bp['q'], bp['t_star'], bp['r_squared']
        qp_ratio = q / max(p, 1e-6)
        ceiling = bp['M']

        # Classify by peak velocity timing (data-driven thresholds)
        if ts < 3:
            lines.append("STRATEGIC PERSONA: 'The Drill Sergeant' (Mandatory/Forced Growth)")
            lines.append(f"ANALYSIS: Explosive onset (Peak week {ts}). High innovation coefficient (p={p}).")
            if bp['hit_bounds']:
                lines.append("⚠️ BOUNDARY WARNING: Growth is so aggressive it is exceeding standard")
                lines.append("   Bass model limits. This indicates a 'Step Function' adoption.")
            lines.append(f"INSIGHT: Immediate adoption spike suggests top-down mandate or intensive")
            lines.append(f"         training driving compliance rather than organic social proof.")
        elif qp_ratio > 50:
            lines.append("STRATEGIC PERSONA: 'The Social Butterfly' (Viral/Organic Growth)")
            lines.append(f"ANALYSIS: Classic S-Curve. Social contagion ratio (q/p) is {qp_ratio:.1f}.")
            lines.append(f"INSIGHT: A seed group successfully drove viral adoption across the org.")
            lines.append(f"         Peak momentum hits at Week {ts}. This is the most")
            lines.append(f"         sustainable long-term strategy if the timeline is acceptable.")
        else:
            lines.append("STRATEGIC PERSONA: 'The Balanced Rollout'")
            lines.append(f"ANALYSIS: Moderate growth with q/p ratio of {qp_ratio:.1f}.")
            lines.append(f"INSIGHT: Balanced mix of innovation (p={p:.4f}) and imitation (q={q:.4f}).")
            lines.append(f"         Peak adoption velocity at Week {ts}.")

        lines.append(f"\nSTATS: R²={r2:.4f} | M (Ceiling)={ceiling:.1%} | q/p={qp_ratio:.1f}")

    return "\n".join(lines)
