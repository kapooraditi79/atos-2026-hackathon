"""
Layer 4 — Run Orchestrator
Executes all 6 analytics steps in sequence.
Usage: python run.py (from d:\\atos\\layer4)
"""
import sys
import json
import pandas as pd
from pathlib import Path

# Ensure layer4 is on the path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    SCENARIO_LABELS, OUTPUT_DIR, L4_OUT,
    COST_PER_TICKET, load_and_fix, load_parquet
)
from bass_diffusion import fit_bass, generate_cio_narrative
from npv_analysis import compute_attrition_cost, compute_npv, generate_npv_narrative
from hotspot_analysis import (
    find_hotspots, generate_intervention_table,
    generate_hotspot_narrative
)
from sensitivity_analysis import (
    run_oat_sensitivity, generate_sensitivity_narrative,
    SENSITIVITY_MC_RUNS
)
from scenario_comparison import (
    build_comparison_table, generate_comparison_narrative
)
from package_output import package_all


def main():
    print("=" * 70)
    print("LAYER 4 — ANALYTICS ENGINE")
    print("Steps: Load | Bass | NPV | Hotspots | Sensitivity | Comparison | JSON Output")
    print("=" * 70)

    # ── Step 1: Load & Fix ────────────────────────────────────────────
    scenarios = load_and_fix()
    if not scenarios:
        print("ERROR: No scenario data loaded. Exiting.")
        return

    # ── Step 2: Bass Diffusion ────────────────────────────────────────
    print("\n[Step 2] Fitting Bass Diffusion models...")
    bass_params = {}
    for key in ['A', 'B', 'C']:
        if key in scenarios:
            bass_params[key] = fit_bass(scenarios[key], key)

    # Bass summary table
    print(f"\n{'='*85}")
    print(f"{'Scenario':<40} {'p':>8} {'q':>8} {'q/p':>8} {'M':>8} {'t*':>8} {'R2':>8}")
    print("-" * 85)
    for key in ['A', 'B', 'C']:
        b = bass_params.get(key, {})
        p_val, q_val = b.get('p', 0), b.get('q', 0)
        qp = f"{q_val/p_val:.1f}" if p_val > 0 else "N/A"
        print(f"{key + ': ' + SCENARIO_LABELS.get(key, key):<40} "
              f"{str(b.get('p', 'N/A')):>8} "
              f"{str(b.get('q', 'N/A')):>8} "
              f"{qp:>8} "
              f"{str(b.get('M', 'N/A')):>8} "
              f"{str(b.get('t_star', 'N/A')):>8} "
              f"{str(b.get('r_squared', 'N/A')):>8}")
    print("\n" + "=" * 85)
    print(generate_cio_narrative(bass_params))

    # ── Step 3: NPV ──────────────────────────────────────────────────
    print("\n[Step 3] Computing NPV...")
    attrition = {}
    parquet_map = {'A': 'agents_a.parquet', 'B': 'agents_b.parquet', 'C': 'agents_c.parquet'}
    for key, pfile in parquet_map.items():
        ppath = OUTPUT_DIR / pfile
        cost, n = compute_attrition_cost(ppath, key)
        attrition[key] = cost
        print(f"  Scenario {key}: {n} at-risk agents → £{cost:,.0f} attrition cost")

    npv_results = {}
    for key in ['A', 'B', 'C']:
        npv_results[key] = compute_npv(scenarios[key], key, attrition[key])

    print(generate_npv_narrative(npv_results))

    # Save NPV JSON
    npv_json_path = L4_OUT / 'npv_analysis.json'
    npv_compact = {}
    for key in ['A', 'B', 'C']:
        r = npv_results[key].copy()
        r.pop('weekly_cashflows', None)
        npv_compact[key] = r
    with open(str(npv_json_path), 'w') as f:
        json.dump(npv_compact, f, indent=2)
    print(f"\n  NPV JSON saved: {npv_json_path}")

    # ── Step 4: Resistance Hotspot Detection ─────────────────────────
    print("\n[Step 4] Detecting resistance hotspots...")
    hotspot_results = {}
    for key in ['A', 'B', 'C']:
        try:
            agents = load_parquet(key)
            hotspot_agents, cluster_summary, total = find_hotspots(agents, key)
            # DATA-DRIVEN: pass scenario_key so intervention uses correct p_fail
            interventions = generate_intervention_table(hotspot_agents, key)
            hotspot_results[key] = {
                'n_hotspots': len(hotspot_agents),
                'total_agents': total,
                'cluster_summary': cluster_summary,
                'interventions': interventions,
                'hotspot_agents': hotspot_agents,
            }
            print(f"  Scenario {key}: {len(hotspot_agents)} hotspot agents / {total} total")
        except Exception as e:
            print(f"  Scenario {key}: Hotspot detection failed — {e}")
            hotspot_results[key] = {
                'n_hotspots': 0, 'total_agents': 0,
                'cluster_summary': None, 'interventions': [],
                'hotspot_agents': pd.DataFrame(),
            }

    print(generate_hotspot_narrative(hotspot_results))

    # Save hotspot JSON
    hotspot_json = {}
    for key in ['A', 'B', 'C']:
        r = hotspot_results[key]
        entry = {
            'scenario': key,
            'label': SCENARIO_LABELS[key],
            'n_hotspots': r['n_hotspots'],
            'total_agents': r['total_agents'],
            'pct_hotspot': round(r['n_hotspots'] / r['total_agents'] * 100, 1) if r['total_agents'] > 0 else 0,
            'interventions': r['interventions'],
        }
        if r['cluster_summary'] is not None and len(r['cluster_summary']) > 0:
            entry['clusters'] = r['cluster_summary'].reset_index().to_dict('records')
        hotspot_json[key] = entry

    hs_path = L4_OUT / 'hotspot_analysis.json'
    with open(str(hs_path), 'w') as f:
        json.dump(hotspot_json, f, indent=2, default=str)
    print(f"\n  Hotspot JSON saved: {hs_path}")

    # ── Step 5: OAT Sensitivity Analysis ─────────────────────────────
    sensitivity_results = None
    print("\n[Step 5] Running OAT Sensitivity Analysis...")
    print("  (Re-running Layer 3 simulations — this may take a few minutes)")
    try:
        sensitivity_results = run_oat_sensitivity(n_runs=SENSITIVITY_MC_RUNS)
        print(generate_sensitivity_narrative(sensitivity_results))

        sens_path = L4_OUT / 'sensitivity_analysis.json'
        with open(str(sens_path), 'w') as f:
            json.dump(sensitivity_results, f, indent=2)
        print(f"\n  Sensitivity JSON saved: {sens_path}")
    except Exception as e:
        print(f"  ⚠️ Sensitivity analysis failed: {e}")
        print("  (Requires Layer 3 model to be importable. Check sys.path.)")

    # ── Step 6: Scenario Comparison Decision Table ───────────────────
    table_rows = None
    print("\n[Step 6] Building scenario comparison table...")
    try:
        table_rows = build_comparison_table(scenarios, npv_results, hotspot_results)
        print(generate_comparison_narrative(table_rows))

        comp_path = L4_OUT / 'comparison_table.json'
        with open(str(comp_path), 'w') as f:
            json.dump(table_rows, f, indent=2)
        print(f"\n  Comparison JSON saved: {comp_path}")
    except Exception as e:
        print(f"  ⚠️ Comparison table failed: {e}")
        import traceback; traceback.print_exc()

    # ── Step 7: Package JSON for Layer 5 ─────────────────────────────
    print("\n[Step 7] Packaging unified JSON for Layer 5...")
    try:
        final_output = package_all(
            scenarios=scenarios,
            bass_params=bass_params,
            npv_results=npv_results,
            hotspot_results=hotspot_results,
            sensitivity_results=sensitivity_results,
            comparison_table=table_rows,
        )

        output_path = L4_OUT / 'layer4_output.json'
        with open(str(output_path), 'w') as f:
            json.dump(final_output, f, indent=2, default=str)
        print(f"  ✅ Layer 4 unified output saved: {output_path}")
        print(f"  Scenarios packaged: {list(final_output['scenarios'].keys())}")
        if 'recommendation' in final_output:
            rec = final_output['recommendation']
            print(f"  Recommendation: Scenario {rec['best_scenario']} ({rec['label']}) — NPV = £{rec['npv']:+,}")
    except Exception as e:
        print(f"  ⚠️ JSON packaging failed: {e}")
        import traceback; traceback.print_exc()

    print("\n[OK] Layer 4 complete.")


if __name__ == '__main__':
    main()
