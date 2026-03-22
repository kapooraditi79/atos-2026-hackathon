import numpy as np
import pandas as pd
import os
from model import WorkforceModel

CSV = '../workforce_v2_1000.csv'
os.makedirs('outputs', exist_ok=True)

#Scenario Configs 
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

SCENARIO_C = {   # Pilot first, hybrid support, self-paced training
    'tool_complexity'    : 0.65,
    'training_intensity' : 0.45,
    'support_model'      : 'hybrid',
    'manager_signal'     : 0.80
}


# Monte Carlo
def run_monte_carlo(scenario_config, n_runs=30, n_steps=52):
    all_model = []
    all_agent = []

    for seed in range(n_runs):
        model  = WorkforceModel(CSV, scenario_config, rng=seed)
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
        resistance_mean   = ('resistance_index',   'mean')
    ).reset_index(names='week')

    agent_combined = pd.concat(all_agent)
    return summary, agent_combined


# Cluster adoption breakdown 
def cluster_adoption_breakdown(agent_df, week=17):
    """Week-18 (index 17) adoption rate per GMM cluster — feeds Persona Risk Heatmap."""
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
    # run.py — in cluster_adoption_breakdown(), change:
    breakdown = (
        week_data.groupby('gmm_cluster')
        .apply(lambda g: (g['adoption_stage'] >= 3).mean(), include_groups=False)
        .round(3)
    )
    breakdown.index = [cluster_labels.get(i, str(i)) for i in breakdown.index]
    return breakdown


#Validation
def validate(sa, sb):
    errors = []

    if not (sa.loc[25, 'adoption_mean'] > sa.loc[7, 'adoption_mean'] + 0.08):
        errors.append('FAIL: No adoption growth between weeks 8 and 26 in Scenario A')

    if not (sb.loc[17, 'adoption_mean'] > sa.loc[17, 'adoption_mean'] + 0.05):
        errors.append('FAIL: Scenario B does not beat Scenario A at week 18')

    if not (30 <= sa.loc[51, 'exs_mean'] <= 92):
        errors.append(f"FAIL: EXS score {sa.loc[51, 'exs_mean']:.1f} outside range 30–92")

    # Week 0 tickets = 0 by design (collect runs before step)
    # Check ticket spike from week 1 onward
    if not (sa.loc[4, 'tickets_mean'] > sa.loc[1, 'tickets_mean']):
        errors.append('FAIL: No ticket spike during Trial stage (weeks 1–5)')

    if errors:
        for e in errors:
            print(e)
        return False

    print('ALL VALIDATION CHECKS PASSED')
    return True


#Main 
if __name__ == '__main__':
    print('Running Scenario A (Big-bang + Chatbot)...')
    sa, agents_a = run_monte_carlo(SCENARIO_A, n_runs=30)

    print('Running Scenario B (Phased + Human + Training)...')
    sb, agents_b = run_monte_carlo(SCENARIO_B, n_runs=30)

    print('Running Scenario C (Pilot + Hybrid)...')
    sc, agents_c = run_monte_carlo(SCENARIO_C, n_runs=30)

    print()
    validate(sa, sb)

    # Cluster-level adoption at week 18
    print('\n--- Week-18 Cluster Adoption (Scenario A) ---')
    print(cluster_adoption_breakdown(agents_a).to_string())

    print('\n--- Week-18 Cluster Adoption (Scenario B) ---')
    print(cluster_adoption_breakdown(agents_b).to_string())

    # Save outputs for Layer 4 and React dashboard
    sa.to_csv('outputs/output_scenario_a.csv', index=False)
    sb.to_csv('outputs/output_scenario_b.csv', index=False)
    sc.to_csv('outputs/output_scenario_c.csv', index=False)

    print('\nOutputs saved to outputs/')
    print('Layer 3 complete.')