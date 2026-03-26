"""
main.py
────────
Digital Twin of the Workforce — top-level orchestrator.

Usage
-----
    python main.py path/to/workforce.csv [--output-dir results] [--l3-runs 30] [--steps 52]

Arguments
---------
csv_path      Path to the raw workforce CSV (required)
--output-dir  Root directory for all pipeline outputs (default: ./dtw_outputs)
--l3-runs     Monte Carlo seeds for Layer 3 simulation (default: 30)
--steps       Simulation steps / weeks (default: 52)
--skip-l4     Stop after Layer 3; skip analytics
--skip-sensitivity
              Skip OAT sensitivity re-runs (fastest path to Layer 4 JSON)

Output layout
-------------
<output-dir>/
    layer2/
        covariance_by_persona.pkl
        collab_graph.pkl
        workforce_enriched.csv         ← enriched CSV consumed by Layers 3 & 4
    layer3/
        outputs/
            output_scenario_a.csv
            output_scenario_b.csv
            output_scenario_c.csv
            agents_a.parquet
            agents_b.parquet
            agents_c.parquet
    layer4/
        bass_diffusion.json
        npv_analysis.json
        hotspot_analysis.json
        sensitivity_analysis.json
        comparison_table.json
        layer4_output.json             ← unified payload for Layer 5 / React dashboard
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Digital Twin of the Workforce — full pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the raw workforce CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dtw_outputs"),
        metavar="DIR",
        help="Root directory for all pipeline outputs (default: ./dtw_outputs)",
    )
    parser.add_argument(
        "--l3-runs",
        type=int,
        default=30,
        metavar="N",
        help="Monte Carlo seeds for Layer 3 (default: 30)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=52,
        metavar="N",
        help="Simulation steps / weeks (default: 52)",
    )
    parser.add_argument(
        "--skip-l4",
        action="store_true",
        help="Stop after Layer 3; skip Layer 4 analytics",
    )
    return parser.parse_args(argv)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _banner(title: str, width: int = 70) -> None:
    print(f"\n{'#' * width}")
    print(f"#  {title}")
    print(f"{'#' * width}\n")


def _elapsed(start: float) -> str:
    secs = time.time() - start
    m, s = divmod(int(secs), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ── Pipeline stages ────────────────────────────────────────────────────────────

def run_layer2(df: pd.DataFrame, output_dir: Path):
    """
    Layer 2: validate clusters, build simulation inputs, return PersonaEngine
    and the path to the enriched CSV.
    """
    from layer2 import run_all_validations, build_all

    l2_dir = output_dir / "layer2"
    l2_dir.mkdir(parents=True, exist_ok=True)

    # Stage 2a — validate
    passed, df = run_all_validations(df)
    if not passed:
        print("\n[main] Layer 2 validation FAILED — aborting pipeline.")
        sys.exit(1)

    # Stage 2b — build artefacts + PersonaEngine
    engine = build_all(df, output_dir=l2_dir)

    enriched_csv = l2_dir / "workforce_enriched.csv"
    return engine, enriched_csv


def run_layer3(enriched_csv: Path, output_dir: Path, n_runs: int, n_steps: int):
    """
    Layer 3: run all three scenarios under Monte Carlo sampling.
    Returns dict of {key: (summary_df, agent_df)}.
    """
    from layer3 import run_scenarios

    l3_out = output_dir / "layer3" / "outputs"

    scenarios = run_scenarios(
        enriched_csv=enriched_csv,
        output_dir=l3_out,
        n_runs=n_runs,
        n_steps=n_steps,
    )
    return scenarios


def run_layer4(scenarios, enriched_csv: Path, output_dir: Path):
    """
    Layer 4: analytics, NPV, hotspot detection, sensitivity, comparison.
    Returns the final JSON payload dict.
    """
    from layer4 import run_analytics

    l4_out = output_dir / "layer4"

    final_output = run_analytics(
        scenarios=scenarios,
        output_dir=l4_out,
        enriched_csv=enriched_csv,
    )
    return final_output


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> dict:
    args = _parse_args(argv)

    # ── Validate input ─────────────────────────────────────────────────────────
    if not args.csv_path.exists():
        print(f"[main] ERROR: CSV not found: {args.csv_path}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 70}")
    print("  DIGITAL TWIN OF THE WORKFORCE — PIPELINE START")
    print(f"{'═' * 70}")
    print(f"  Input CSV  : {args.csv_path.resolve()}")
    print(f"  Output dir : {args.output_dir.resolve()}")
    print(f"  L3 MC runs : {args.l3_runs}")
    print(f"  Steps      : {args.steps}")
    print(f"  Skip L4    : {args.skip_l4}")
    print(f"{'═' * 70}\n")

    wall_start = time.time()

    # ── Load CSV once ──────────────────────────────────────────────────────────
    print("Loading input CSV...")
    df = pd.read_csv(str(args.csv_path))
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # ── Layer 2 ────────────────────────────────────────────────────────────────
    _banner("LAYER 2 — Workforce Model & Persona Engine")
    t = time.time()
    engine, enriched_csv = run_layer2(df, args.output_dir)
    print(f"\n  Layer 2 completed in {_elapsed(t)}")

    # ── Layer 3 ────────────────────────────────────────────────────────────────
    _banner("LAYER 3 — Agent-Based Simulation Engine")
    t = time.time()
    scenarios = run_layer3(enriched_csv, args.output_dir, args.l3_runs, args.steps)
    print(f"\n  Layer 3 completed in {_elapsed(t)}")

    final_output: dict = {}

    # ── Layer 4 (optional) ─────────────────────────────────────────────────────
    if not args.skip_l4:
        _banner("LAYER 4 — Analytics & Prediction Engine")
        t = time.time()
        final_output = run_layer4(scenarios, enriched_csv, args.output_dir)
        print(f"\n  Layer 4 completed in {_elapsed(t)}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  PIPELINE COMPLETE")
    print(f"  Total elapsed : {_elapsed(wall_start)}")
    print(f"  Outputs       : {args.output_dir.resolve()}")
    if not args.skip_l4 and "recommendation" in final_output:
        rec = final_output["recommendation"]
        print(f"  Recommendation: Scenario {rec['best_scenario']} ({rec['label']}) — NPV £{rec['npv']:+,}")
    print(f"{'═' * 70}\n")

    return final_output


if __name__ == "__main__":
    main()