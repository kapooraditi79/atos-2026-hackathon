"""
layer4/run.py
──────────────
Analytics orchestrator — executes all 6 analytic steps in sequence.

Public entry point
------------------
    run_analytics(scenarios, output_dir, enriched_csv, skip_sensitivity=False) -> dict

Changes from original
---------------------
  FIX 1: Parquet path was `output_dir.parent / "layer3" / "outputs"` — should be
          `output_dir.parent / "layer3" / "outputs"` only when output_dir is the
          layer4 dir.  Now resolved correctly as a sibling of the layer4 dir.

  FIX 2: Added `skip_sensitivity` parameter so callers (main.py, Flask API)
          can skip the expensive OAT re-runs.

  FIX 3: Sensitivity module is imported inside the try-block so missing
          dependencies don't crash the whole analytics step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from layer4.config import SCENARIO_LABELS, COST_PER_TICKET
from layer4.bass_diffusion import fit_bass, generate_cio_narrative
from layer4.npv_analysis import compute_attrition_cost, compute_npv, generate_npv_narrative
from layer4.hotspot_analysis import (
    find_hotspots, generate_intervention_table, generate_hotspot_narrative,
)
from layer4.scenario_comparison import build_comparison_table, generate_comparison_narrative
from layer4.package_output import package_all


def _fix_scenario_df(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a Layer 3 summary DataFrame for Layer 4 consumption."""
    df = df.copy()
    baseline            = df["adoption_mean"].iloc[0]
    df["adoption_gain"] = df["adoption_mean"] - baseline
    df["gain_p05"]      = df["adoption_p05"]  - baseline
    df["gain_p95"]      = df["adoption_p95"]  - baseline

    if "productivity_delta_true" in df.columns:
        if df["productivity_delta_true"].mean() > 0.5:
            raise ValueError(
                f"Scenario {key}: productivity_delta_true contains absolute values, not deltas."
            )
        df["productivity_delta"] = df["productivity_delta_true"]
    elif "productivity_delta" not in df.columns:
        df["productivity_delta"] = 0.0

    return df


def run_analytics(
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: str | Path,
    enriched_csv: str | Path,
    skip_sensitivity: bool = False,
) -> dict:
    """
    Run all Layer 4 analytics steps and persist JSON artefacts.

    Parameters
    ----------
    scenarios        : {key: (summary_df, agent_df)} from layer3.run_scenarios()
    output_dir       : directory where Layer 4 JSON outputs are saved
    enriched_csv     : path to the workforce CSV from Layer 2
    skip_sensitivity : if True, skip OAT sensitivity re-runs (much faster)
    """
    output_dir   = Path(output_dir)
    enriched_csv = Path(enriched_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LAYER 4 — ANALYTICS ENGINE")
    print("Steps: Fix | Bass | NPV | Hotspots | Sensitivity | Comparison | JSON")
    print("=" * 70)

    # ── Step 1: Fix scenario DataFrames ───────────────────────────────────────
    scenario_dfs: dict[str, pd.DataFrame] = {}
    for key, (summary, _) in scenarios.items():
        scenario_dfs[key] = _fix_scenario_df(key, summary)
        print(f"  Scenario {key} loaded: Gain {scenario_dfs[key]['adoption_gain'].iloc[-1]:.3f}")

    # ── Step 2: Bass Diffusion ─────────────────────────────────────────────────
    print("\n[Step 2] Fitting Bass Diffusion models...")
    bass_params: dict[str, dict] = {}
    for key in ["A", "B", "C"]:
        if key in scenario_dfs:
            bass_params[key] = fit_bass(scenario_dfs[key], key)

    print(f"\n{'=' * 85}")
    print(f"{'Scenario':<40} {'p':>8} {'q':>8} {'q/p':>8} {'M':>8} {'t*':>8} {'R2':>8}")
    print("-" * 85)
    for key in ["A", "B", "C"]:
        b     = bass_params.get(key, {})
        p_val = b.get("p", 0)
        q_val = b.get("q", 0)
        qp    = f"{q_val / p_val:.1f}" if p_val > 0 else "N/A"
        print(
            f"{key + ': ' + SCENARIO_LABELS.get(key, key):<40} "
            f"{str(b.get('p', 'N/A')):>8} {str(b.get('q', 'N/A')):>8} "
            f"{qp:>8} {str(b.get('M', 'N/A')):>8} "
            f"{str(b.get('t_star', 'N/A')):>8} {str(b.get('r_squared', 'N/A')):>8}"
        )
    print("\n" + "=" * 85)
    print(generate_cio_narrative(bass_params))

    bass_path = output_dir / "bass_diffusion.json"
    with open(str(bass_path), "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk != "fitted_curve"}
             for k, v in bass_params.items()},
            f, indent=2,
        )
    print(f"\n  Bass JSON saved: {bass_path}")

    # ── Step 3: NPV ────────────────────────────────────────────────────────────
    print("\n[Step 3] Computing NPV...")
    attrition:   dict[str, float] = {}
    npv_results: dict[str, dict]  = {}

    # ── FIX 1: Parquet files are written by Layer 3 into output_dir/../layer3/outputs
    # output_dir here is <root>/layer4, so parquets are at <root>/layer3/outputs/
    layer3_outputs = output_dir.parent / "layer3" / "outputs"

    for key in ["A", "B", "C"]:
        parquet_path = layer3_outputs / f"agents_{key.lower()}.parquet"
        cost, n = compute_attrition_cost(parquet_path, key)
        attrition[key] = cost
        print(f"  Scenario {key}: {n} at-risk agents → £{cost:,.0f} attrition cost")

    for key in ["A", "B", "C"]:
        if key in scenario_dfs:
            npv_results[key] = compute_npv(scenario_dfs[key], key, attrition[key])

    print(generate_npv_narrative(npv_results))

    npv_compact = {
        k: {kk: vv for kk, vv in v.items() if kk != "weekly_cashflows"}
        for k, v in npv_results.items()
    }
    npv_path = output_dir / "npv_analysis.json"
    with open(str(npv_path), "w") as f:
        json.dump(npv_compact, f, indent=2)
    print(f"\n  NPV JSON saved: {npv_path}")

    # ── Step 4: Hotspot Detection ──────────────────────────────────────────────
    print("\n[Step 4] Detecting resistance hotspots...")
    hotspot_results: dict[str, dict] = {}

    for key in ["A", "B", "C"]:
        try:
            _, agent_df = scenarios[key]
            if isinstance(agent_df.index, pd.MultiIndex):
                agent_df = agent_df.reset_index()
            hotspot_agents, cluster_summary, total = find_hotspots(agent_df, key)
            interventions = generate_intervention_table(hotspot_agents, key)
            hotspot_results[key] = {
                "n_hotspots":      len(hotspot_agents),
                "total_agents":    total,
                "cluster_summary": cluster_summary,
                "interventions":   interventions,
                "hotspot_agents":  hotspot_agents,
            }
            print(f"  Scenario {key}: {len(hotspot_agents)} hotspot agents / {total} total")
        except Exception as e:
            print(f"  Scenario {key}: hotspot detection failed — {e}")
            hotspot_results[key] = {
                "n_hotspots": 0, "total_agents": 0,
                "cluster_summary": None, "interventions": [],
                "hotspot_agents": pd.DataFrame(),
            }

    print(generate_hotspot_narrative(hotspot_results))

    hotspot_json: dict = {}
    for key in ["A", "B", "C"]:
        r = hotspot_results[key]
        entry: dict = {
            "scenario":     key,
            "label":        SCENARIO_LABELS[key],
            "n_hotspots":   r["n_hotspots"],
            "total_agents": r["total_agents"],
            "pct_hotspot":  round(r["n_hotspots"] / r["total_agents"] * 100, 1)
                            if r["total_agents"] > 0 else 0,
            "interventions": r["interventions"],
        }
        cs = r.get("cluster_summary")
        if cs is not None and len(cs) > 0:
            entry["clusters"] = cs.reset_index().to_dict("records")
        hotspot_json[key] = entry

    hs_path = output_dir / "hotspot_analysis.json"
    with open(str(hs_path), "w") as f:
        json.dump(hotspot_json, f, indent=2, default=str)
    print(f"\n  Hotspot JSON saved: {hs_path}")

    # ── Step 5: OAT Sensitivity ────────────────────────────────────────────────
    sensitivity_results = None
    if skip_sensitivity:
        print("\n[Step 5] OAT Sensitivity skipped (--skip-sensitivity).")
    else:
        print("\n[Step 5] Running OAT Sensitivity Analysis...")
        print("  (Re-runs Layer 3 simulations — may take a few minutes)")
        try:
            from layer4.sensitivity_analysis import (
                run_oat_sensitivity, generate_sensitivity_narrative, SENSITIVITY_MC_RUNS,
            )
            sensitivity_results = run_oat_sensitivity(csv_path=enriched_csv, n_runs=SENSITIVITY_MC_RUNS)
            print(generate_sensitivity_narrative(sensitivity_results))
            sens_path = output_dir / "sensitivity_analysis.json"
            with open(str(sens_path), "w") as f:
                json.dump(sensitivity_results, f, indent=2)
            print(f"\n  Sensitivity JSON saved: {sens_path}")
        except Exception as e:
            print(f"  ⚠️  Sensitivity analysis failed: {e}")
            import traceback; traceback.print_exc()

    # ── Step 6: Comparison Table ───────────────────────────────────────────────
    table_rows = None
    print("\n[Step 6] Building scenario comparison table...")
    try:
        table_rows = build_comparison_table(scenario_dfs, npv_results, hotspot_results)
        print(generate_comparison_narrative(table_rows))
        comp_path = output_dir / "comparison_table.json"
        with open(str(comp_path), "w") as f:
            json.dump(table_rows, f, indent=2)
        print(f"\n  Comparison JSON saved: {comp_path}")
    except Exception as e:
        print(f"  ⚠️  Comparison table failed: {e}")
        import traceback; traceback.print_exc()

    # ── Step 7: Package for Layer 5 ───────────────────────────────────────────
    print("\n[Step 7] Packaging unified JSON for Layer 5...")
    final_output: dict = {}
    try:
        final_output = package_all(
            scenarios=scenario_dfs,
            bass_params=bass_params,
            npv_results=npv_results,
            hotspot_results=hotspot_results,
            sensitivity_results=sensitivity_results,
            comparison_table=table_rows,
        )
        output_path = output_dir / "layer4_output.json"
        with open(str(output_path), "w") as f:
            json.dump(final_output, f, indent=2, default=str)
        print(f"  ✓ Layer 4 unified output saved: {output_path}")
        if "recommendation" in final_output:
            rec = final_output["recommendation"]
            print(
                f"  Recommendation: Scenario {rec['best_scenario']} "
                f"({rec['label']}) — NPV = £{rec['npv']:+,}"
            )
    except Exception as e:
        print(f"  ⚠️  JSON packaging failed: {e}")
        import traceback; traceback.print_exc()

    print(f"\n{'═' * 70}")
    print("LAYER 4 COMPLETE ✓")
    print(f"{'═' * 70}")

    return final_output