"""
backend/app.py
───────────────
Flask API for the DTW dashboard.
Calls the actual layer1 → layer2 → layer3 → layer4 pipeline modules.

Place this file at:  <project_root>/backend/app.py
(i.e., DTW/backend/app.py, sibling of layer1/, layer2/, layer3/, layer4/)

Run:
    cd DTW
    python backend/app.py

Endpoints
---------
GET  /api/health                → {"status": "ok"}
POST /api/simulate              → multipart/form-data
       file=<csv>               IBM HR CSV or pre-synthesised workforce CSV
       scenarios=A,B,C          comma-separated subset
       n_agents=1000            (optional) agents to synthesise
       skip_sensitivity=false   (optional) skip OAT sensitivity
       scenario_configs=<json>  (optional) override per-scenario params from UI sliders
                                  e.g. {"A": {"tool_complexity": 0.5, ...}, ...}
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Add project root to sys.path so layer1/2/3/4 imports resolve ─────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)
CORS(app)

# ── IBM vs workforce CSV detection ────────────────────────────────────────────
_IBM_SENTINEL       = {"JobSatisfaction", "PerformanceRating", "JobInvolvement", "Attrition"}
_WORKFORCE_SENTINEL = {"persona", "satisfaction_score", "resistance_propensity", "digital_dexterity"}


def _detect_csv_type(df) -> str:
    cols = set(df.columns)
    if _IBM_SENTINEL.issubset(cols):
        return "ibm"
    if _WORKFORCE_SENTINEL.issubset(cols):
        return "workforce"
    return "unknown"


def _merge_scenario_configs(base: dict, overrides: dict) -> dict:
    """
    Deep-merge frontend slider overrides into the base SCENARIO_CONFIGS dict.

    base     — copy of config.SCENARIO_CONFIGS (keyed A/B/C)
    overrides — parsed JSON from the request field "scenario_configs"
                e.g. {"A": {"tool_complexity": 0.5, "training_intensity": 0.2,
                             "manager_signal": 0.6, "support_model": "hybrid"}}

    Only the four editable keys are accepted; anything else is ignored.
    """
    EDITABLE = {"tool_complexity", "training_intensity", "manager_signal", "support_model"}
    merged = {k: dict(v) for k, v in base.items()}   # shallow copy per scenario
    for scenario_key, params in overrides.items():
        key = scenario_key.strip().upper()
        if key not in merged:
            continue
        for param, val in params.items():
            if param in EDITABLE:
                merged[key][param] = val
    return merged


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "project_root": str(PROJECT_ROOT)})


# ── Main simulation endpoint ──────────────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def simulate():
    import pandas as pd

    try:
        # ── Parse request ─────────────────────────────────────────────────────
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        scenarios_raw    = request.form.get("scenarios", "A,B,C")
        scenarios_to_run = [s.strip().upper() for s in scenarios_raw.split(",") if s.strip()]
        n_agents         = int(request.form.get("n_agents", 1000))
        skip_sensitivity = request.form.get("skip_sensitivity", "false").lower() == "true"

        # ── Parse optional frontend slider overrides ──────────────────────────
        raw_overrides = request.form.get("scenario_configs", "")
        frontend_overrides: dict = {}
        if raw_overrides:
            try:
                frontend_overrides = json.loads(raw_overrides)
                print(f"[API] Received scenario_configs overrides for: {list(frontend_overrides.keys())}")
            except json.JSONDecodeError as exc:
                print(f"[API] Warning: could not parse scenario_configs JSON — {exc}. Using defaults.")

        # ── Read CSV ──────────────────────────────────────────────────────────
        content  = file.read().decode("utf-8")
        raw_df   = pd.read_csv(io.StringIO(content))
        csv_type = _detect_csv_type(raw_df)
        print(f"[API] CSV: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols  type={csv_type}")

        # ── Layer 1 — synthesise if needed ────────────────────────────────────
        if csv_type == "workforce":
            workforce_df = raw_df
            print("[API] Pre-synthesised workforce CSV — skipping Layer 1")
        else:
            print(f"[API] Running Layer 1 (n_agents={n_agents})...")
            from layer1 import generate_workforce
            workforce_df = generate_workforce.generate_workforce(raw_df, n_agents=n_agents)
            print(f"[API] Layer 1 done: {len(workforce_df)} agents")

        # ── Layer 2 ───────────────────────────────────────────────────────────
        print("[API] Running Layer 2 (GMM clustering + network)...")
        from layer2 import run_all_validations, build_all

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp   = Path(tmpdir)
            l2_dir = tmp / "layer2"
            l2_dir.mkdir()

            passed, workforce_df = run_all_validations(workforce_df)
            if not passed:
                return jsonify({"error": "Layer 2 validation failed — check workforce data quality"}), 422

            engine = build_all(workforce_df, output_dir=l2_dir)
            enriched_csv = l2_dir / "workforce_enriched.csv"
            print("[API] Layer 2 done")

            # ── Build effective scenario configs (defaults + UI overrides) ────
            import layer4.config as l4cfg

            effective_configs = _merge_scenario_configs(l4cfg.SCENARIO_CONFIGS, frontend_overrides)
            print(f"[API] Effective scenario configs: {effective_configs}")

            # ── Layer 3 ───────────────────────────────────────────────────────
            print(f"[API] Running Layer 3 scenarios={scenarios_to_run}...")
            from layer3 import run_scenarios
            import layer3.run as l3run

            # layer3 SCENARIOS dict maps key → config dict passed into each
            # ABM run.  We rebuild it from effective_configs so the slider
            # values (tool_complexity, training_intensity, etc.) are live.
            original_l3_scenarios = l3run.SCENARIOS

            # Build the patched dict — keep every existing key that layer3
            # needs, but overwrite the four editable fields from the UI.
            patched_l3_scenarios = {}
            for k in scenarios_to_run:
                if k not in original_l3_scenarios:
                    continue
                patched = dict(original_l3_scenarios[k])          # start from layer3 defaults
                if k in effective_configs:
                    for field in ("tool_complexity", "training_intensity",
                                  "manager_signal", "support_model"):
                        if field in effective_configs[k]:
                            patched[field] = effective_configs[k][field]
                patched_l3_scenarios[k] = patched

            if not patched_l3_scenarios:
                return jsonify({"error": f"No valid scenarios in: {scenarios_to_run}"}), 400

            l3run.SCENARIOS = patched_l3_scenarios
            try:
                l3_out = tmp / "layer3" / "outputs"
                scenarios_result = run_scenarios(
                    enriched_csv=enriched_csv,
                    output_dir=l3_out,
                    n_runs=15,
                    n_steps=52,
                )
            finally:
                l3run.SCENARIOS = original_l3_scenarios   # always restore

            print("[API] Layer 3 done")

            # ── Layer 4 ───────────────────────────────────────────────────────
            # Monkey-patch layer4.config with the effective configs so that
            # NPV calculations, support model params, and labels all reflect
            # the UI slider values for this request.
            print("[API] Running Layer 4 analytics...")
            from layer4.run import run_analytics

            original_l4_configs = l4cfg.SCENARIO_CONFIGS
            l4cfg.SCENARIO_CONFIGS = effective_configs
            try:
                l4_out = tmp / "layer4"
                final_output = run_analytics(
                    scenarios=scenarios_result,
                    output_dir=l4_out,
                    enriched_csv=enriched_csv,
                    skip_sensitivity=True,   # always skip in API — too slow
                )
            finally:
                l4cfg.SCENARIO_CONFIGS = original_l4_configs  # always restore

            print("[API] Layer 4 done")

            # ── Build API response ────────────────────────────────────────────
            response = _build_api_response(
                final_output=final_output,
                scenarios_result=scenarios_result,
                workforce_df=workforce_df,
                scenarios_run=scenarios_to_run,
                n_agents=len(workforce_df),
                effective_configs=effective_configs,
            )

            return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def _build_api_response(
    final_output: dict,
    scenarios_result: dict,
    workforce_df,
    scenarios_run: list[str],
    n_agents: int,
    effective_configs: dict,
) -> dict:
    """
    Extend the Layer 4 package_all() output with the extra fields
    the React frontend needs (weekly series, persona weekly series, etc.)
    Also echoes back the effective_configs used so the frontend can display
    what actually ran (useful if the user changed sliders).
    """
    # Persona distribution
    persona_dist = workforce_df["persona"].value_counts().to_dict()

    # Per-scenario weekly series
    for key, (summary_df, agent_df) in scenarios_result.items():
        if key not in final_output.get("scenarios", {}):
            continue

        s = final_output["scenarios"][key]

        # Weekly time series
        s["weekly"] = {
            "adoption_mean":     [round(v, 4) for v in summary_df["adoption_mean"].tolist()],
            "adoption_p05":      [round(v, 4) for v in summary_df["adoption_p05"].tolist()],
            "adoption_p95":      [round(v, 4) for v in summary_df["adoption_p95"].tolist()],
            "frustration_mean":  [round(v, 4) for v in summary_df["frustration_mean"].tolist()],
            "productivity_mean": [round(v, 4) for v in summary_df["productivity_mean"].tolist()],
        }

        # Final adoption rate (week 52) — used by frontend KPI tiles and banner
        s["final_adoption"] = round(float(summary_df["adoption_mean"].iloc[-1]), 4)

        # Week-52 persona adoption from agent data
        if isinstance(agent_df.index, object) and hasattr(agent_df.index, "names"):
            try:
                agent_df = agent_df.reset_index()
            except Exception:
                pass

        step_col = next((c for c in ["Step", "week"] if c in agent_df.columns), None)
        personas  = ["Tech Pioneer", "Power User", "Pragmatic Adopter", "Remote-First Worker", "Reluctant User"]

        persona_w52:    dict[str, float] = {}
        persona_weekly: dict[str, list]  = {p: [] for p in personas}

        if step_col and "persona" in agent_df.columns and "adoption_stage" in agent_df.columns:
            for week in range(52):
                week_data = agent_df[agent_df[step_col] == week]
                for p in personas:
                    p_data  = week_data[week_data["persona"] == p]
                    adopted = float((p_data["adoption_stage"] >= 3).mean()) if len(p_data) > 0 else 0.0
                    persona_weekly[p].append(round(adopted, 3))
            for p in personas:
                persona_w52[p] = persona_weekly[p][-1] if persona_weekly[p] else 0.0
        else:
            final_adop = float(summary_df["adoption_mean"].iloc[-1])
            for p in personas:
                persona_w52[p]    = final_adop
                persona_weekly[p] = [final_adop] * 52

        s["persona_w52"]    = persona_w52
        s["persona_weekly"] = persona_weekly

        # Productivity delta
        baseline_prod = float(summary_df["productivity_mean"].iloc[0])
        final_prod    = float(summary_df["productivity_mean"].iloc[-1])
        s["prod_delta_pct"] = round((final_prod - baseline_prod) / max(baseline_prod, 0.01) * 100, 2)

        s["hotspots"]    = s.get("hotspots", {}).get("n_hotspots", 0)
        npv_block       = s.get("npv", {})
        s["npv"]        = npv_block.get("total", 0)
        # Surface investment directly so the NPV waterfall chart can read s.investment
        s["investment"] = npv_block.get("components", {}).get("investment", 0)

        # Echo back the effective config that was actually used for this scenario
        s["effective_config"] = effective_configs.get(key, {})

    # Top-level metadata — include effective_configs so UI can confirm what ran
    final_output["metadata"] = {
        "n_employees":          n_agents,
        "scenarios_run":        scenarios_run,
        "persona_distribution": persona_dist,
        "effective_configs":    effective_configs,
    }

    return final_output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Starting DTW API on http://localhost:5000")
    app.run(debug=True, port=5000, threaded=False)