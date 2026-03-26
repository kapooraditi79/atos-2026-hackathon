"""
layer3/run.py
──────────────
Monte Carlo orchestration for the three rollout scenarios.

Public entry point
------------------
    run_scenarios(enriched_csv, output_dir) -> dict[str, tuple[DataFrame, DataFrame]]

Returns a dict keyed by scenario letter ('A', 'B', 'C').
Each value is (summary_df, agent_df).

File outputs (CSVs + Parquets) are written to `output_dir` if provided.
Callers that only need the DataFrames can pass output_dir=None.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from layer3.model import WorkforceModel

# ── Scenario definitions ───────────────────────────────────────────────────────

SCENARIOS: dict[str, dict] = {
    "A": {   # Big-bang rollout, AI chatbot, minimal training
        "tool_complexity":    0.65,
        "training_intensity": 0.10,
        "support_model":      "chatbot",
        "manager_signal":     0.40,
    },
    "B": {   # Phased rollout, human support, instructor-led training
        "tool_complexity":    0.65,
        "training_intensity": 0.70,
        "support_model":      "human",
        "manager_signal":     0.60,
    },
    "C": {   # Pilot first, strong management signal, self-paced training
        "tool_complexity":    0.65,
        "training_intensity": 0.45,
        "support_model":      "hybrid",
        "manager_signal":     0.80,
    },
}

SCENARIO_LABELS: dict[str, str] = {
    "A": "Big-bang + Chatbot",
    "B": "Phased + Human + Training",
    "C": "Pilot + Strong Management",
}


# ── Monte Carlo runner ────────────────────────────────────────────────────────

def _run_monte_carlo(
    scenario_config: dict,
    csv_path: str | Path,
    n_runs: int = 30,
    n_steps: int = 52,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run n_runs seeds, return (summary_df, agent_df)."""
    all_model: list[pd.DataFrame] = []
    all_agent: list[pd.DataFrame] = []

    for seed in range(n_runs):
        model  = WorkforceModel(scenario_config, rng=seed, csv_path=csv_path)
        result = model.run(n_steps)
        result["run"] = seed
        all_model.append(result)

        agent_result = model.datacollector.get_agent_vars_dataframe()
        agent_result["run"] = seed
        all_agent.append(agent_result)

    combined = pd.concat(all_model)

    summary = (
        combined.groupby(combined.index)
        .agg(
            adoption_mean     = ("adoption_rate",      "mean"),
            adoption_p05      = ("adoption_rate",      lambda x: np.percentile(x, 5)),
            adoption_p95      = ("adoption_rate",      lambda x: np.percentile(x, 95)),
            productivity_mean = ("productivity_delta", "mean"),
            frustration_mean  = ("avg_frustration",    "mean"),
            tickets_mean      = ("ticket_volume",      "mean"),
            tickets_p95       = ("ticket_volume",      lambda x: np.percentile(x, 95)),
            exs_mean          = ("exs_score",          "mean"),
            resistance_mean   = ("resistance_index",   "mean"),
        )
        .reset_index(names="week")
    )

    # True productivity delta (change from week-0 baseline)
    baseline = summary["productivity_mean"].iloc[0]
    summary["productivity_delta_true"] = summary["productivity_mean"] - baseline

    support_model = scenario_config["support_model"]
    print(
        f"  [{support_model}] avg weekly tickets: {summary['tickets_mean'].mean():.4f} | "
        f"final frustration: {summary['frustration_mean'].iloc[-1]:.4f}"
    )

    agent_combined = pd.concat(all_agent)
    return summary, agent_combined


# ── Cluster adoption breakdown (utility used by Layer 4) ─────────────────────

def cluster_adoption_breakdown(agent_df: pd.DataFrame, week: int = 17) -> pd.Series:
    """Week-N adoption rate per GMM cluster."""
    cluster_labels = {
        0: "Pragmatic Adopter",
        1: "Elite (Pioneer + top PU)",
        2: "Remote-First Worker",
        3: "Reluctant User",
        4: "Mainstream Power User",
    }
    try:
        week_data = agent_df.xs(week, level="Step")
    except KeyError:
        week_data = agent_df

    breakdown = (
        week_data.groupby("gmm_cluster")
        .apply(lambda g: (g["adoption_stage"] >= 3).mean(), include_groups=False)
        .round(3)
    )
    breakdown.index = [cluster_labels.get(i, str(i)) for i in breakdown.index]
    return breakdown


# ── Cross-scenario validation ─────────────────────────────────────────────────

def validate(sa: pd.DataFrame, sb: pd.DataFrame) -> bool:
    errors = []

    if not (sa.loc[25, "adoption_mean"] > sa.loc[7, "adoption_mean"] + 0.08):
        errors.append("FAIL: No adoption growth between weeks 8 and 26 in Scenario A")

    if not (sb.loc[7, "adoption_mean"] > sa.loc[7, "adoption_mean"] + 0.05):
        errors.append("FAIL: Scenario B does not beat Scenario A at week 8")

    if not (30 <= sa.loc[51, "exs_mean"] <= 92):
        errors.append(f"FAIL: EXS score {sa.loc[51, 'exs_mean']:.1f} outside range 30–92")

    if not (sa.loc[4, "tickets_mean"] > sa.loc[0, "tickets_mean"]):
        errors.append("FAIL: No ticket spike during Trial stage (weeks 1–5)")

    if errors:
        for e in errors:
            print(f"  {e}")
        return False

    print("  ✓ ALL VALIDATION CHECKS PASSED")
    return True


# ── Public entry point ────────────────────────────────────────────────────────

def run_scenarios(
    enriched_csv: str | Path,
    output_dir: str | Path | None = None,
    n_runs: int = 30,
    n_steps: int = 52,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Run all requested scenarios under Monte Carlo sampling.

    Parameters
    ----------
    enriched_csv : path to the workforce CSV produced by Layer 2
    output_dir   : if provided, CSVs and Parquets are saved here
    n_runs       : Monte Carlo seeds per scenario
    n_steps      : simulation steps (weeks)

    Returns
    -------
    dict keyed by whichever scenario letters are in SCENARIOS (may be a
    subset of A/B/C when called from the API with a filtered SCENARIOS dict).
    """
    enriched_csv = Path(enriched_csv)
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    print("=" * 60)
    print("LAYER 3 — SIMULATION ENGINE")
    print("=" * 60)

    for key, config in SCENARIOS.items():
        label = SCENARIO_LABELS.get(key, key)
        print(f"\n  Running Scenario {key}: {label}  "
              f"({n_runs} seeds × {n_steps} steps)...")
        summary, agents = _run_monte_carlo(config, enriched_csv, n_runs, n_steps)
        results[key] = (summary, agents)

    # Cross-scenario validation — only meaningful when both A and B were run
    print()
    if "A" in results and "B" in results:
        validate(results["A"][0], results["B"][0])
    else:
        ran = sorted(results.keys())
        print(f"  ⚠ Cross-scenario validation skipped "
              f"(requires A + B; only ran: {', '.join(ran)})")

    # Cluster breakdowns (informational) — only for scenarios that were run
    for key in ("A", "B"):
        if key in results:
            print(f"\n  Week-18 cluster adoption — Scenario {key}:")
            print(cluster_adoption_breakdown(results[key][1]).to_string())

    # Persist outputs if requested
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for key, (summary, agents) in results.items():
            summary.to_csv(output_dir / f"output_scenario_{key.lower()}.csv", index=False)
            agents.to_parquet(output_dir / f"agents_{key.lower()}.parquet")

        print(f"\n  ✓ Layer 3 outputs saved → {output_dir}")

    print(f"\n{'═' * 60}")
    print("LAYER 3 COMPLETE ✓")
    print(f"{'═' * 60}")

    return results