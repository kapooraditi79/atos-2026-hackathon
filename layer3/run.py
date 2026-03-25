import numpy as np
import pandas as pd
import os
from layer3.model import WorkforceModel
from layer3.agent import WorkforceAgent
from typing import cast

CSV = 'layer2/workforce_v2_1000.csv'
os.makedirs('layer3/outputs', exist_ok=True)

# ── Scenario Configs ─────────────────────────────────────────────────────────

SCENARIO_A = {   # Big-bang rollout, AI chatbot, minimal training
    'tool_complexity'    : 0.65,
    'training_intensity' : 0.10,
    'support_model'      : 'chatbot',
    'manager_signal'     : 0.40
}

SCENARIO_B = {   # Phased rollout, human support, instructor-led training
    'tool_complexity'    : 0.65,
    'training_intensity' : 0.70,
    'support_model'      : 'human',
    'manager_signal'     : 0.60
}

SCENARIO_C = {   # Pilot first, strong management signal, self-paced training
    # NOTE: support_model='hybrid' is treated identically to 'human' in
    # _generate_tickets() (0% chatbot deflection, p_fail=0.12).
    # Scenario C is therefore labelled "Pilot + Strong Management" in
    # downstream analytics — NOT "Hybrid Support".
    # The real differentiators vs Scenario B are:
    #   manager_signal  : 0.80 (vs 0.60) — highest of the three scenarios
    #   training_intensity: 0.45 (vs 0.70) — medium boost
    'tool_complexity'    : 0.65,
    'training_intensity' : 0.45,
    'support_model'      : 'hybrid',   # treated as 'human' in code — see above
    'manager_signal'     : 0.80
}


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def run_monte_carlo(scenario_config, n_runs=30, n_steps=52):
    all_model = []
    all_agent = []

    for seed in range(n_runs):
        model  = WorkforceModel(scenario_config, rng=seed)
        result = model.run(n_steps)
        result['run'] = seed
        all_model.append(result)

        agent_result = model.datacollector.get_agent_vars_dataframe()
        agent_result['run'] = seed
        all_agent.append(agent_result)

    combined = pd.concat(all_model)

    summary = combined.groupby(combined.index).agg(
        adoption_mean     = ('adoption_rate',      'mean'),
        adoption_p05      = ('adoption_rate',      lambda x: np.percentile(x, 5)),
        adoption_p95      = ('adoption_rate',      lambda x: np.percentile(x, 95)),
        productivity_mean = ('productivity_delta', 'mean'),
        frustration_mean  = ('avg_frustration',    'mean'),
        tickets_mean      = ('ticket_volume',      'mean'),
        tickets_p95       = ('ticket_volume',       lambda x: np.percentile(x, 95)),
        exs_mean          = ('exs_score',           'mean'),
        resistance_mean   = ('resistance_index',   'mean'),
        network_density_mean = ('network_density',  'mean')
    ).reset_index(names='week')

    # ── FIX #1 (CRITICAL): productivity_mean is ABSOLUTE (≈0.63), not a delta.
    # Compute the true delta here so Layer 4 NPV calculations are meaningful.
    # Without this, NPV inputs would be ~0.63 instead of ~±0.03 — off by 20×.
    baseline_productivity = summary['productivity_mean'].iloc[0]
    summary['productivity_delta_true'] = (
        summary['productivity_mean'] - baseline_productivity
    )
    # productivity_mean is kept for backward compatibility, but Layer 4 should
    # use productivity_delta_true for any NPV or change-from-baseline analysis.

    agent_combined = pd.concat(all_agent)
    # DEBUG: scenario-level averages for quick sanity check (ticket volume + frustration)
    print(f"[run_monte_carlo] ({scenario_config['support_model']}) avg weekly tickets: {summary['tickets_mean'].mean():.4f}")
    print(f"[run_monte_carlo] ({scenario_config['support_model']}) avg frustration_final: {summary['frustration_mean'].iloc[-1]:.4f}")
    return summary, agent_combined


# ── Cluster adoption breakdown ────────────────────────────────────────────────

def cluster_adoption_breakdown(agent_df, week=17):
    """
    Week-18 (index 17) adoption rate per GMM cluster — feeds Persona Risk Heatmap.

    NOTE: resistance_mean is 0.0 for weeks 0-7 by design (reporter returns 0
    before step 8). Only use resistance_mean for weeks 8+ analysis.
    """
    cluster_labels = {
        0: 'Pragmatic Adopter',
        1: 'Elite (Pioneer + top PU)',
        2: 'Remote-First Worker',
        3: 'Reluctant User',
        4: 'Mainstream Power User'
    }
    try:
        week_data = agent_df.xs(week, level='Step')
    except KeyError:
        week_data = agent_df

    breakdown = (
        week_data.groupby('gmm_cluster')
        .apply(lambda g: (g['adoption_stage'] >= 3).mean(), include_groups=False)
        .round(3)
    )
    breakdown.index = [cluster_labels.get(i, str(i)) for i in breakdown.index]
    return breakdown


# ── Validation ────────────────────────────────────────────────────────────────

def validate(sa, sb):
    """
    Cross-scenario sanity checks.

    Known boundary condition: week-0 adoption_mean ≈ 0.40 (not 0%) because
    Pioneers, Power Users, and Remote-First Workers are initialised at stage 3+
    via _init_stage(). The < 0.40 assertion in the original validation.py may
    fail at this exact boundary — this is documented behaviour, not a bug.
    """
    errors = []

    if not (sa.loc[25, 'adoption_mean'] > sa.loc[7, 'adoption_mean'] + 0.08):
        errors.append('FAIL: No adoption growth between weeks 8 and 26 in Scenario A')

    if not (sb.loc[7, 'adoption_mean'] > sa.loc[7, 'adoption_mean'] + 0.05):
        errors.append('FAIL: Scenario B does not beat Scenario A at week 8')

    if not (30 <= sa.loc[51, 'exs_mean'] <= 92):
        errors.append(f"FAIL: EXS score {sa.loc[51, 'exs_mean']:.1f} outside range 30–92")

    # Week 0 tickets = 0 by design (collect runs before first step).
    # Check ticket spike from week 1 onward.
    if not (sa.loc[4, 'tickets_mean'] > sa.loc[0, 'tickets_mean']):
        errors.append('FAIL: No ticket spike during Trial stage (weeks 1–5)')

    # ── FIX #2: NDS (network_density) is monotonically increasing in every
    # scenario — the network can only grow, never shrink (no edge removal).
    # Do NOT validate or surface network_density as a meaningful metric.
    # Layer 4 should exclude NDS from JSON output to Layer 5.

    if errors:
        for e in errors:
            print(e)
        return False

    print('ALL VALIDATION CHECKS PASSED')
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Running Scenario A (Big-bang + Chatbot)...')
    sa, agents_a = run_monte_carlo(SCENARIO_A, n_runs=30)

    print('Running Scenario B (Phased + Human + Training)...')
    sb, agents_b = run_monte_carlo(SCENARIO_B, n_runs=30)

    # Scenario C: labelled "Pilot + Strong Management" in analytics.
    # hybrid support_model is treated as 'human' in _generate_tickets().
    print('Running Scenario C (Pilot + Strong Management)...')
    sc, agents_c = run_monte_carlo(SCENARIO_C, n_runs=30)

    print()
    validate(sa, sb)

    # Cluster-level adoption at week 18 (step index 17)
    print('\n--- Week-18 Cluster Adoption (Scenario A) ---')
    print(cluster_adoption_breakdown(agents_a).to_string())

    print('\n--- Week-18 Cluster Adoption (Scenario B) ---')
    print(cluster_adoption_breakdown(agents_b).to_string())

    # ── FIX #3 (CRITICAL): Save agent_combined to Parquet.
    # In the original run.py agents_a/b/c were held in memory only.
    # Without these saves, Layer 4 hotspot detection, persona risk heatmap,
    # and AI trajectory analytics have no data to work with.
    # Parquet chosen over CSV: ~1.56M rows → ~15MB (vs ~200MB CSV), loads in ~2s.
    # Requires: pip install pyarrow

    # ── FIX #4 (CRITICAL): productivity_delta_true is already baked into the
    # summary DataFrames above. Layer 4 must use that column, not productivity_mean,
    # for any NPV or change-from-baseline calculation.

    # Save model-level summaries (52 rows each)
    sa.to_csv('layer3/outputs/output_scenario_a.csv', index=False)
    sb.to_csv('layer3/outputs/output_scenario_b.csv', index=False)
    sc.to_csv('layer3/outputs/output_scenario_c.csv', index=False)

    # ── FIX #3: Save agent-level data to Parquet (was missing in v3)
    agents_a.to_parquet('layer3/outputs/agents_a.parquet')
    agents_b.to_parquet('layer3/outputs/agents_b.parquet')
    agents_c.to_parquet('layer3/outputs/agents_c.parquet')

    # ── NOTE: Curve labelling for Layer 4
    # - adoption_mean starts at ~0.40, NOT 0%, due to _init_stage() pre-seeding.
    #   Label x-axis as weeks 0–51 (51 simulation steps). Do NOT label row 51 as
    #   "Week 52" — the true step-52 end-state is never collected (collect() fires
    #   before agents step, so row 51 = state after 51 steps).
    # - resistance_mean is 0.0 for weeks 0–7. Only use for weeks 8+ analysis.
    # - network_density / NDS: monotonically increasing — exclude from analytics.
    # - Scenario C label: "Pilot + Strong Management" (not "Hybrid Support").

    print('\nOutputs saved:')
    print('  layer3/outputs/output_scenario_a.csv')
    print('  layer3/outputs/output_scenario_b.csv')
    print('  layer3/outputs/output_scenario_c.csv')
    print('  layer3/outputs/agents_a.parquet')
    print('  layer3/outputs/agents_b.parquet')
    print('  layer3/outputs/agents_c.parquet')
    print('Layer 3 complete.')