"""
layer4/npv_analysis.py
───────────────────────
Analytic 2: Net Present Value (NPV) Analysis.
4-component DCF model: productivity, support costs, investment, attrition.

No hardcoded file paths — parquet path is passed in by the caller.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from layer4.config import (
    SCENARIO_LABELS, WACC_ANNUAL, WEEKLY_RATE, HEADCOUNT,
    AVG_WEEKLY_SAL, P_CHURN, REPLACEMENT, INVESTMENT, COST_PER_TICKET,
)


def compute_attrition_cost(parquet_path: str | Path, scenario_key: str) -> tuple[float, int]:
    """Identify at-risk agents by deduplicating on AgentID."""
    try:
        import pyarrow  # noqa: F401
        agents = pd.read_parquet(str(parquet_path))
    except ImportError:
        print("  ⚠️  pyarrow not found. Run 'pip install pyarrow'. Returning £0 attrition.")
        return 0.0, 0
    except Exception:
        return 0.0, 0

    if isinstance(agents.index, pd.MultiIndex):
        agents = agents.reset_index()

    step_col = next((c for c in ["Step", "week", "run"] if c in agents.columns), None)
    late      = agents[agents[step_col] >= 30] if step_col else agents

    id_col   = next((c for c in ["AgentID", "agentid", "agent_id"] if c in late.columns), None)

    at_risk = late[
        (late["adoption_stage"] < 2.0)
        & (late["frustration"]    > 0.25)
        & (late["churn_risk"]     > 0.5)
    ]

    if id_col:
        n_at_risk = at_risk[id_col].nunique()
    else:
        n_steps   = late[step_col].nunique() if step_col else 1
        n_at_risk = at_risk.shape[0] // max(n_steps, 1)

    n_at_risk = min(int(n_at_risk), HEADCOUNT)
    cost      = n_at_risk * P_CHURN * REPLACEMENT
    return cost, n_at_risk


def compute_npv(df: pd.DataFrame, scenario_key: str, attrition_cost: float) -> dict:
    """Compute 2-year NPV from the 4 cash-flow components."""
    ticket_cost = COST_PER_TICKET[scenario_key]
    invest      = INVESTMENT[scenario_key]
    npv         = float(invest)

    if "productivity_delta" not in df.columns:
        if "productivity_delta_true" in df.columns:
            df = df.copy()
            df["productivity_delta"] = df["productivity_delta_true"]
        else:
            df = df.copy()
            df["productivity_delta"] = 0.0

    weekly_cf: list[dict] = []
    for _, row in df.iterrows():
        t          = int(row["week"])
        cf_prod    = row["productivity_delta"] * HEADCOUNT * AVG_WEEKLY_SAL
        cf_support = -row["tickets_mean"] * ticket_cost
        cf_total   = cf_prod + cf_support
        pv         = cf_total / (1 + WEEKLY_RATE) ** (t + 1)
        npv       += pv
        weekly_cf.append({
            "week":       t,
            "cf_prod":    round(cf_prod, 2),
            "cf_support": round(cf_support, 2),
            "cf_net":     round(cf_total, 2),
            "pv":         round(pv, 2),
        })

    # Year 2 extrapolation
    late_mask   = (df["week"] >= 40) & (df["week"] <= 51)
    late_prod   = df.loc[late_mask, "productivity_delta"].mean()
    late_tickets = df.loc[late_mask, "tickets_mean"].mean()

    for t in range(52, 104):
        cf_total = (late_prod * HEADCOUNT * AVG_WEEKLY_SAL) + (-late_tickets * ticket_cost)
        npv += cf_total / (1 + WEEKLY_RATE) ** (t + 1)

    npv -= attrition_cost

    return {
        "scenario":       scenario_key,
        "label":          SCENARIO_LABELS[scenario_key],
        "npv":            round(npv),
        "investment":     invest,
        "attrition_cost": round(attrition_cost),
        "cost_per_ticket": ticket_cost,
        "extrapolation": {
            "late_prod_delta":   round(float(late_prod), 5),
            "late_tickets_mean": round(float(late_tickets), 2),
        },
        "weekly_cashflows": weekly_cf,
    }


def generate_npv_narrative(npv_results: dict) -> str:
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append("NET PRESENT VALUE (NPV) — 2-YEAR HORIZON")
    lines.append(f"{'=' * 70}")
    lines.append(
        f"Discount Rate: {WACC_ANNUAL:.0%} WACC | Headcount: {HEADCOUNT} | "
        f"Avg Salary: £{AVG_WEEKLY_SAL:,}/wk"
    )
    lines.append("")

    lines.append(f"{'Scenario':<32} {'Investment':>12} {'Attrition':>12} {'NPV':>14}")
    lines.append("-" * 72)

    best_key, best_npv = None, float("-inf")
    for key in ["A", "B", "C"]:
        r = npv_results[key]
        lines.append(
            f"{key + ': ' + r['label'][:25]:<32} "
            f"£{r['investment']:>+10,}  "
            f"£{-r['attrition_cost']:>+10,}  "
            f"£{r['npv']:>+12,}"
        )
        if r["npv"] > best_npv:
            best_npv = r["npv"]
            best_key = key

    for key in ["A", "B", "C"]:
        r   = npv_results[key]
        ext = r["extrapolation"]
        lines.append(f"\n{'─' * 70}")
        lines.append(f"SCENARIO {key}: {r['label']}")
        lines.append(f"{'─' * 70}")
        lines.append(f"  Investment     : £{r['investment']:>+,}")
        lines.append(f"  Attrition Risk : £{-r['attrition_cost']:>+,}")
        lines.append(f"  2-Year NPV     : £{r['npv']:>+,}")

        if r["npv"] > 0:
            roi = (r["npv"] / abs(r["investment"])) * 100
            lines.append(f"  ROI            : {roi:+.1f}%")
            lines.append("  ✅  POSITIVE NPV — Investment pays for itself.")
        else:
            lines.append("  ❌  NEGATIVE NPV — Investment destroys value.")

        steady_weekly = (
            ext["late_prod_delta"] * HEADCOUNT * AVG_WEEKLY_SAL
            + (-ext["late_tickets_mean"] * r["cost_per_ticket"])
        )
        lines.append(f"  Steady-state weekly CF (wk 40-51): £{steady_weekly:+,.0f}/wk")

    lines.append(f"\n{'=' * 70}")
    if best_key:
        lines.append(
            f"🏆  RECOMMENDATION: Scenario {best_key} "
            f"({npv_results[best_key]['label']}) — NPV = £{best_npv:+,}"
        )
    lines.append(f"{'=' * 70}")

    return "\n".join(lines)