"""
layer4/package_output.py
─────────────────────────
Step 7: Package all analytics into the unified Layer 5 JSON payload.

No file I/O — caller writes the resulting dict to disk.
"""

from __future__ import annotations

import pandas as pd

from layer4.config import SCENARIO_LABELS, SCENARIO_CONFIGS, SUPPORT_MODEL_PARAMS


def package_scenario(
    scenario_key: str,
    scenario_df: pd.DataFrame,
    bass: dict,
    npv: dict,
    hotspot: dict,
) -> dict:
    """Package all analytics for a single scenario into the Layer 5 JSON schema."""
    df           = scenario_df
    config       = SCENARIO_CONFIGS.get(scenario_key, {})
    support_model = config.get("support_model", "unknown")
    model_params  = SUPPORT_MODEL_PARAMS.get(support_model, {})

    # ── Adoption curve ────────────────────────────────────────────────────────
    w0  = float(df["adoption_mean"].iloc[0])
    w51 = float(df["adoption_mean"].iloc[-1])

    curve_points = []
    for _, row in df.iterrows():
        pt = {
            "week": int(row["week"]),
            "mean": round(float(row["adoption_mean"]), 4),
            "gain": round(float(row["adoption_gain"]), 4),
        }
        if "adoption_p05" in df.columns:
            pt["p05"] = round(float(row["adoption_p05"]) - w0, 4)
        if "adoption_p95" in df.columns:
            pt["p95"] = round(float(row["adoption_p95"]) - w0, 4)
        curve_points.append(pt)

    # ── Bass diffusion ────────────────────────────────────────────────────────
    bass_out = {
        "p":           bass.get("p", 0),
        "q":           bass.get("q", 0),
        "t_star":      bass.get("t_star"),
        "r_squared":   bass.get("r_squared", 0),
        "M":           bass.get("M", 0),
        "fit_success": bass.get("fit_success", False),
        "hit_bounds":  bass.get("hit_bounds", False),
    }

    # ── NPV ───────────────────────────────────────────────────────────────────
    ext = npv.get("extrapolation", {})
    npv_out = {
        "total":   npv["npv"],
        "roi_pct": round((npv["npv"] / abs(npv["investment"])) * 100, 1) if npv["investment"] != 0 else 0,
        "components": {
            "investment":          npv["investment"],
            "attrition_risk":      -npv.get("attrition_cost", 0),
            "productivity_impact": round(ext.get("late_prod_delta", 0) * 1000 * 1200 * 52, 0),
            "support_cost":        round(-ext.get("late_tickets_mean", 0) * npv.get("cost_per_ticket", 0) * 52, 0),
        },
        "steady_state_cf_weekly": round(
            (ext.get("late_prod_delta", 0) * 1000 * 1200)
            + (-ext.get("late_tickets_mean", 0) * npv.get("cost_per_ticket", 0))
        ),
    }

    # ── Hotspots ──────────────────────────────────────────────────────────────
    hotspot_list: list[dict] = []
    cs = hotspot.get("cluster_summary")
    if cs is not None and len(cs) > 0:
        for idx, row in cs.iterrows():
            entry = {
                "gmm_cluster":     int(idx),
                "agent_count":     int(row["count"]),
                "mean_stage":      round(float(row["mean_stage"]), 2),
                "mean_frustration": round(float(row["mean_frustration"]), 2),
                "pct_of_hotspots": round(float(row["pct_of_hotspots"]), 1),
            }
            if "churn_risk_sum" in row.index:
                entry["churn_risk_count"] = int(row["churn_risk_sum"])
            hotspot_list.append(entry)

    # ── Disruption signal ─────────────────────────────────────────────────────
    frust_col = "frustration_mean" if "frustration_mean" in df.columns else "avg_frustration"
    if frust_col in df.columns:
        fs              = df[frust_col]
        baseline_frust  = float(fs.iloc[0])
        peak_idx        = fs.idxmax()
        peak_week       = int(df.loc[peak_idx, "week"]) if "week" in df.columns else int(peak_idx)
        disruption = {
            "peak_frustration": round(float(fs.max()), 3),
            "peak_week":        peak_week,
            "week51_frustration": round(float(fs.iloc[-1]), 3),
            "disruption_area":  round(float((fs - baseline_frust).clip(lower=0).sum()), 2),
        }
    else:
        disruption = {"peak_frustration": 0, "peak_week": 0, "week51_frustration": 0, "disruption_area": 0}

    # ── Assemble ──────────────────────────────────────────────────────────────
    return {
        "scenario_id":      scenario_key,
        "label":            SCENARIO_LABELS[scenario_key],
        "config":           config,
        "support_params":   model_params,
        "bass":             bass_out,
        "adoption": {
            "week0_baseline": round(w0, 4),
            "week51_mean":    round(w51, 4),
            "adoption_gain":  round(w51 - w0, 4),
            "curve":          curve_points,
        },
        "npv": npv_out,
        "hotspots": {
            "total_agents": hotspot["total_agents"],
            "n_hotspots":   hotspot["n_hotspots"],
            "pct_hotspot":  round(hotspot["n_hotspots"] / hotspot["total_agents"] * 100, 1)
                            if hotspot["total_agents"] > 0 else 0,
            "clusters":     hotspot_list,
            "interventions": hotspot.get("interventions", []),
        },
        "disruption_signal": disruption,
    }


def package_all(
    scenarios: dict[str, pd.DataFrame],
    bass_params: dict,
    npv_results: dict,
    hotspot_results: dict,
    sensitivity_results: list[dict] | None = None,
    comparison_table: list[dict] | None = None,
) -> dict:
    """Package ALL scenarios + cross-scenario analytics into the final Layer 5 JSON."""
    output: dict = {
        "layer":       4,
        "description": "Analytics Engine — all scenario results and cross-scenario analysis",
        "scenarios":   {},
    }

    for key in ["A", "B", "C"]:
        if key in scenarios:
            output["scenarios"][key] = package_scenario(
                scenario_key=key,
                scenario_df=scenarios[key],
                bass=bass_params.get(key, {}),
                npv=npv_results.get(key, {}),
                hotspot=hotspot_results.get(key, {
                    "n_hotspots": 0, "total_agents": 0,
                    "cluster_summary": None, "interventions": [],
                }),
            )

    if sensitivity_results:
        output["tornado"] = [
            {
                "param":        r["param"],
                "label":        r["label"],
                "low_val":      r["low_val"],
                "high_val":     r["high_val"],
                "adoption_low": r["adoption_low"],
                "adoption_high": r["adoption_high"],
                "swing_pp":     r["swing_pp"],
            }
            for r in sensitivity_results
        ]

    if comparison_table:
        output["comparison"] = comparison_table

    if npv_results:
        best_key = max(npv_results, key=lambda k: npv_results[k]["npv"])
        output["recommendation"] = {
            "best_scenario": best_key,
            "label":         SCENARIO_LABELS[best_key],
            "npv":           npv_results[best_key]["npv"],
            "reasoning":     f"Scenario {best_key} has the highest 2-year NPV.",
        }

    return output