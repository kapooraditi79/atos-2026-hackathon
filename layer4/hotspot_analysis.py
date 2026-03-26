"""
Layer 4 — Analytic 3: Resistance Hotspot Detection
Identifies stuck+frustrated agents and traces to GMM clusters.
Interventions are DATA-DRIVEN based on actual scenario configs.
"""
import pandas as pd
from config import (
    SCENARIO_LABELS, SCENARIO_CONFIGS, SUPPORT_MODEL_PARAMS,
    OUTPUT_DIR, load_parquet
)


def find_hotspots(agents, scenario_key):
    """
    Identify resistance hotspot agents from weeks 30-51.
    Returns: (hotspot_agents, cluster_summary, total_agents)
    """
    step_col = next((c for c in ['Step', 'week'] if c in agents.columns), None)
    if not step_col:
        print(f"  ⚠️ No Step/week column in {scenario_key}")
        return pd.DataFrame(), pd.DataFrame(), 0

    late = agents[(agents[step_col] >= 30) & (agents[step_col] <= 51)]

    id_col = next((c for c in ['AgentID', 'agentid', 'agent_id'] if c in late.columns), None)
    if not id_col:
        print(f"  ⚠️ No AgentID column in {scenario_key}")
        return pd.DataFrame(), pd.DataFrame(), 0

    total_agents = agents[id_col].nunique()

    numeric_cols = ['adoption_stage', 'frustration']
    numeric_extra = [c for c in ['gmm_cluster', 'churn_risk', 'is_amplifier'] if c in late.columns]
    numeric_cols += numeric_extra

    by_agent = late.groupby(id_col)[numeric_cols].mean()

    if 'persona' in late.columns:
        persona_map = late.groupby(id_col)['persona'].first()
        by_agent['persona'] = persona_map

    # Hotspot filter: stuck AND frustrated
    mask = (by_agent['adoption_stage'] < 1.5) & (by_agent['frustration'] > 0.35)
    hotspot_agents = by_agent[mask].copy()

    # Cluster concentration analysis
    cluster_summary = pd.DataFrame()
    if 'gmm_cluster' in hotspot_agents.columns and len(hotspot_agents) > 0:
        hotspot_agents['cluster_id'] = hotspot_agents['gmm_cluster'].round().astype(int)
        cluster_summary = hotspot_agents.groupby('cluster_id').agg(
            count=('adoption_stage', 'count'),
            mean_stage=('adoption_stage', 'mean'),
            mean_frustration=('frustration', 'mean'),
            churn_risk_sum=('churn_risk', 'sum') if 'churn_risk' in hotspot_agents.columns else ('adoption_stage', 'count'),
        ).round(3)
        cluster_summary['pct_of_hotspots'] = (cluster_summary['count'] / len(hotspot_agents) * 100).round(1)

    return hotspot_agents, cluster_summary, total_agents


def generate_intervention_table(hotspot_agents, scenario_key):
    """
    Map hotspot patterns to recommended interventions.
    DATA-DRIVEN: reads the scenario's actual support_model config
    to generate correct p_fail and root cause text.
    """
    interventions = []
    if len(hotspot_agents) == 0:
        return interventions

    # Look up this scenario's ACTUAL support model params
    config = SCENARIO_CONFIGS.get(scenario_key, {})
    support_model = config.get('support_model', 'unknown')
    model_params = SUPPORT_MODEL_PARAMS.get(support_model, {})
    p_fail = model_params.get('p_fail', 'unknown')
    friction = model_params.get('adoption_friction', 'unknown')
    drag = model_params.get('support_drag', 'unknown')

    # Pattern 1: Frustrated + stuck + churn risk
    if 'churn_risk' in hotspot_agents.columns:
        churn_mask = hotspot_agents['churn_risk'] > 0.5
        n_churn = int(churn_mask.sum())
        if n_churn > 0:
            interventions.append({
                'pattern': f'High frustration + low stage + churn risk ({n_churn} agents)',
                'root_cause': (
                    f'{support_model.capitalize()} support p_fail={p_fail} generates frustration '
                    f'faster than 0.90/wk decay. support_drag={drag}, adoption_friction={friction}. '
                    f'Traced to: L1 support_dependency × L3 Poisson λ × p_fail.'
                ),
                'intervention': _get_support_intervention(support_model, p_fail),
            })

    # Pattern 2: Amplifiers broadcasting resistance
    if 'is_amplifier' in hotspot_agents.columns:
        amp_mask = hotspot_agents['is_amplifier'] > 0.5
        n_amp = int(amp_mask.sum())
        if n_amp > 0:
            interventions.append({
                'pattern': f'Hotspot agents with high is_amplifier ({n_amp} agents)',
                'root_cause': 'Bridge nodes (high betweenness from L2 network) are broadcasting '
                              'frustration to multiple clusters via negative SN signal.',
                'intervention': f'Manager 1-on-1 engagement with {n_amp} flagged amplifier agents. '
                                'Reduce their frustration → reduce negative SN broadcast.',
            })

    return interventions


def _get_support_intervention(support_model, p_fail):
    """Generate the correct intervention based on actual support model."""
    if support_model == 'chatbot':
        return (f'Improve chatbot KB for common ticket types (VPN, hardware). '
                f'Target: reduce p_fail from {p_fail:.0%} to 20%.')
    elif support_model == 'hybrid':
        return (f'Hybrid support p_fail={p_fail:.0%} is moderate. Consider adding '
                f'dedicated Tier-1 human support for Cluster 3 agents to reduce '
                f'unresolved ticket frustration.')
    elif support_model == 'human':
        return (f'Human support already has lowest p_fail ({p_fail:.0%}). '
                f'Focus on manager 1-on-1 engagement and targeted training for '
                f'remaining hotspot agents.')
    else:
        return f'Review support model configuration (current: {support_model}).'


def generate_hotspot_narrative(hotspot_results):
    """CIO-facing narrative for hotspot analysis."""
    lines = []
    lines.append(f"\n{'='*75}")
    lines.append("RESISTANCE HOTSPOT DETECTION — ANALYTIC 3")
    lines.append(f"{'='*75}")
    lines.append("Definition: Agents with mean(adoption_stage) < 1.5 AND ")
    lines.append("mean(frustration) > 0.35 across Weeks 30-51.")
    lines.append("")

    for key in ['A', 'B', 'C']:
        r = hotspot_results[key]
        n_hot = r['n_hotspots']
        total = r['total_agents']
        pct = (n_hot / total * 100) if total > 0 else 0

        lines.append(f"\n{'─'*75}")
        lines.append(f"SCENARIO {key}: {SCENARIO_LABELS[key]}")
        lines.append(f"{'─'*75}")
        lines.append(f"  Hotspot agents : {n_hot} / {total} ({pct:.1f}%)")

        # Cluster concentration
        if r['cluster_summary'] is not None and len(r['cluster_summary']) > 0:
            lines.append(f"  Cluster concentration:")
            for idx, row in r['cluster_summary'].iterrows():
                lines.append(
                    f"    Cluster {idx}: {int(row['count'])} agents "
                    f"({row['pct_of_hotspots']:.0f}% of hotspots) | "
                    f"avg stage={row['mean_stage']:.2f}, "
                    f"avg frustration={row['mean_frustration']:.2f}"
                )
            dom = r['cluster_summary']['count'].idxmax()
            dom_pct = r['cluster_summary'].loc[dom, 'pct_of_hotspots']
            if dom_pct > 50:
                lines.append(f"  ⚠️ Cluster {dom} dominates ({dom_pct:.0f}% of hotspots) — "
                             f"structural resistance pattern detected.")
        else:
            lines.append(f"  No cluster concentration data available.")

        # Interventions
        if r['interventions']:
            lines.append(f"\n  RECOMMENDED INTERVENTIONS:")
            for i, intv in enumerate(r['interventions'], 1):
                lines.append(f"  {i}. {intv['pattern']}")
                lines.append(f"     Root cause : {intv['root_cause']}")
                lines.append(f"     Action     : {intv['intervention']}")

        if n_hot == 0:
            lines.append(f"  ✅ No resistance hotspots detected — adoption is healthy.")

    lines.append(f"\n{'='*75}")
    return "\n".join(lines)
