"""
layer4/scenario_comparison.py
───────────────────────────────
Analytic 5: Scenario Comparison Decision Table.
Side-by-side CIO decision table from all analytics results.

No file I/O.
"""

from __future__ import annotations

import pandas as pd

from layer4.config import SCENARIO_LABELS, HEADCOUNT, AVG_WEEKLY_SAL, COST_PER_TICKET


def build_comparison_table(
    scenarios: dict[str, pd.DataFrame],
    npv_results: dict,
    hotspot_results: dict,
) -> list[dict]:
    """Build the CIO decision table from all analytics results."""
    rows = []
    for key in ["A", "B", "C"]:
        df    = scenarios[key]
        npv   = npv_results[key]
        n_hot = hotspot_results[key]["n_hotspots"] if key in hotspot_results else 0

        w0    = df["adoption_mean"].iloc[0]
        w51   = df["adoption_mean"].iloc[-1]

        ci_width  = (df["adoption_p95"].iloc[-1] - df["adoption_p05"].iloc[-1]) * 100
        frust_col = "frustration_mean" if "frustration_mean" in df.columns else "avg_frustration"

        if frust_col in df.columns:
            peak_idx        = df[frust_col].idxmax()
            frust_peak_week = int(df.loc[peak_idx, "week"]) if "week" in df.columns else int(peak_idx)
            frust_peak_val  = df[frust_col].max()
            baseline_frust  = df[frust_col].iloc[0]
            disruption      = float((df[frust_col] - baseline_frust).clip(lower=0).sum())
        else:
            frust_peak_week, frust_peak_val, disruption = "N/A", 0, 0.0

        ext = npv.get("extrapolation", {})
        steady_cf = 0
        if ext:
            steady_cf = (
                ext.get("late_prod_delta", 0) * HEADCOUNT * AVG_WEEKLY_SAL
                + (-ext.get("late_tickets_mean", 0) * COST_PER_TICKET[key])
            )

        rows.append({
            "scenario":       key,
            "label":          SCENARIO_LABELS[key],
            "adoption_gain_pp": round((w51 - w0) * 100, 1),
            "ci_width_pp":    round(ci_width, 1),
            "npv":            npv["npv"],
            "investment":     npv["investment"],
            "roi_pct":        round((npv["npv"] / abs(npv["investment"])) * 100, 1)
                              if npv["investment"] != 0 else 0,
            "frust_peak_week": frust_peak_week,
            "frust_peak_val": round(float(frust_peak_val), 3),
            "disruption_area": round(disruption, 2),
            "hotspots":       n_hot,
            "steady_cf_weekly": round(steady_cf),
            "attrition_cost": npv.get("attrition_cost", 0),
        })

    return rows


def generate_comparison_narrative(table_rows: list[dict]) -> str:
    lines = []
    lines.append(f"\n{'=' * 90}")
    lines.append("SCENARIO COMPARISON — CIO DECISION TABLE (Analytic 5)")
    lines.append(f"{'=' * 90}")
    lines.append("All scenarios: same 1,000 agents, same network, same TAM.")
    lines.append("Only scenario_config (support model, training, manager signal) differs.")
    lines.append("")

    hdr = (
        f"{'Metric':<28} "
        f"{'A: Big-bang':>16} "
        f"{'B: Human+Train':>16} "
        f"{'C: Pilot+Mgmt':>16}"
    )
    lines.append(hdr)
    lines.append("─" * 78)

    a, b, c = table_rows[0], table_rows[1], table_rows[2]

    def row(label, vals, fmt="s"):
        if fmt == "pp":
            return f"{label:<28} {vals[0]:>+15.1f}pp {vals[1]:>+15.1f}pp {vals[2]:>+15.1f}pp"
        elif fmt == "ci":
            return f"{label:<28} {vals[0]:>15.1f}pp {vals[1]:>15.1f}pp {vals[2]:>15.1f}pp"
        elif fmt == "money":
            return (
                f"{label:<28} "
                f"{'£' + f'{vals[0]:+,}':>16} "
                f"{'£' + f'{vals[1]:+,}':>16} "
                f"{'£' + f'{vals[2]:+,}':>16}"
            )
        elif fmt == "pct":
            return f"{label:<28} {vals[0]:>+15.1f}% {vals[1]:>+15.1f}% {vals[2]:>+15.1f}%"
        elif fmt == "int":
            return f"{label:<28} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}"
        elif fmt == "float":
            return f"{label:<28} {vals[0]:>16.2f} {vals[1]:>16.2f} {vals[2]:>16.2f}"
        else:
            return f"{label:<28} {str(vals[0]):>16} {str(vals[1]):>16} {str(vals[2]):>16}"

    lines.append(row("Adoption Gain (W51)",   [a["adoption_gain_pp"], b["adoption_gain_pp"], c["adoption_gain_pp"]], "pp"))
    lines.append(row("90% CI Width (W51)",    [a["ci_width_pp"],      b["ci_width_pp"],      c["ci_width_pp"]],      "ci"))
    lines.append(row("24-Month NPV",          [a["npv"],              b["npv"],              c["npv"]],              "money"))
    lines.append(row("ROI",                   [a["roi_pct"],          b["roi_pct"],          c["roi_pct"]],          "pct"))
    lines.append(row("Investment",            [a["investment"],       b["investment"],       c["investment"]],       "money"))
    lines.append(row("Attrition Risk",        [-a["attrition_cost"],  -b["attrition_cost"],  -c["attrition_cost"]],  "money"))
    lines.append(row("Steady-State CF/wk",   [a["steady_cf_weekly"], b["steady_cf_weekly"], c["steady_cf_weekly"]], "money"))
    lines.append(row("Frustration Peak (wk)", [a["frust_peak_week"],  b["frust_peak_week"],  c["frust_peak_week"]],  "int"))
    lines.append(row("Disruption Area",       [a["disruption_area"],  b["disruption_area"],  c["disruption_area"]],  "float"))
    lines.append(row("Resistance Hotspots",   [a["hotspots"],         b["hotspots"],         c["hotspots"]],         "int"))

    lines.append(f"\n{'─' * 90}")
    lines.append("VERDICT")
    lines.append(f"{'─' * 90}")

    best          = max(table_rows, key=lambda r: r["npv"])
    worst         = min(table_rows, key=lambda r: r["npv"])
    lowest_disrupt = min(table_rows, key=lambda r: r["disruption_area"])

    lines.append(f"  BEST NPV         : Scenario {best['scenario']} ({best['label']}) — £{best['npv']:+,}")
    lines.append(f"  WORST NPV        : Scenario {worst['scenario']} ({worst['label']}) — £{worst['npv']:+,}")
    lines.append(
        f"  LOWEST DISRUPTION: Scenario {lowest_disrupt['scenario']} ({lowest_disrupt['label']}) — "
        f"area={lowest_disrupt['disruption_area']:.2f}"
    )
    lines.append("")

    for r in table_rows:
        risk = "LOW" if r["ci_width_pp"] < 5 else ("MEDIUM" if r["ci_width_pp"] < 15 else "HIGH")
        lines.append(f"  Scenario {r['scenario']}: CI width={r['ci_width_pp']:.1f}pp → Outcome certainty: {risk} RISK")

    lines.append("")
    lines.append(f"  RECOMMENDATION: Scenario {best['scenario']} is the dominant strategy.")
    lines.append(f"  Highest adoption gain (+{best['adoption_gain_pp']:.1f}pp), "
                 f"NPV £{best['npv']:+,}, {best['hotspots']} resistance hotspots.")
    if best["steady_cf_weekly"] > 0:
        lines.append(
            f"  Upfront investment (£{abs(best['investment']):,}) pays back at "
            f"£{best['steady_cf_weekly']:+,}/wk in steady state."
        )

    lines.append(f"\n{'=' * 90}")
    return "\n".join(lines)