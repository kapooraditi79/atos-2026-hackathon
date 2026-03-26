# DTW Fixes — What Was Wrong & How to Apply

## Bugs Found

### Bug 1 — CRITICAL: Layer 1 missing from `main.py` (root crash cause)

`main.py` fed the raw IBM HR CSV directly to Layer 2, which expects the
**already-synthesised** `workforce_v2_1000.csv` schema.  Layer 2's
`validate_clusters.py` immediately aborts because columns like `persona`,
`satisfaction_score`, `digital_dexterity` are absent from the IBM file.

**Fix:** New `layer1/` module (`generate_workforce.py`) synthesises a
workforce DataFrame from any IBM HR CSV.  `main.py` now auto-detects the CSV
type and runs Layer 1 first when needed.

---

### Bug 2 — CRITICAL: Parquet path one level off in `layer4/run.py`

```python
# ORIGINAL (wrong)
parquet_path = output_dir.parent / "layer3" / "outputs" / f"agents_{key}.parquet"
```

`output_dir` here is `<root>/layer4`, so `.parent` is `<root>`.  That gives
`<root>/layer3/outputs/` — which appears correct but only by accident when
`output_dir` is two levels deep.  If the output root is at a different depth
the path is wrong.  The correct approach is to resolve relative to a known
anchor:

```python
# FIXED
layer3_outputs = output_dir.parent / "layer3" / "outputs"
parquet_path   = layer3_outputs / f"agents_{key.lower()}.parquet"
```

This is now consistent with how Layer 3 writes the files.

---

### Bug 3 — `--skip-sensitivity` flag listed in docstring but missing from argparse

`main.py` documented `--skip-sensitivity` but never added it to the
`ArgumentParser`, so passing it would raise an `unrecognized arguments` error.

**Fix:** Added `--skip-sensitivity` to argparse and threaded it through to
`run_layer4()` and `layer4/run.py`.

---

### Bug 4 — Flask `backend/app.py` reimplemented the pipeline inline

The previous Flask API had its own inline copies of L1–L4 logic, completely
separate from the refactored `layer1/`, `layer2/`, `layer3/`, `layer4/`
modules.  Changes to the actual simulation code would not affect the API.

**Fix:** New `backend/app.py` imports and calls your actual modules directly.

---

### Bug 5 — `layer4/run.py` had no `skip_sensitivity` parameter

The sensitivity step re-runs Layer 3 five times per parameter, making it
very slow in an API context.  There was no way to skip it programmatically.

**Fix:** Added `skip_sensitivity: bool = False` to `run_analytics()`.
The API always passes `skip_sensitivity=True`; the CLI respects `--skip-sensitivity`.

---

## File Structure After Applying Fixes

```
DTW/
├── layer1/                        ← NEW
│   ├── __init__.py
│   └── generate_workforce.py
├── layer2/
│   ├── __init__.py
│   ├── build_simulation_inputs.py
│   └── validate_clusters.py
├── layer3/
│   ├── __init__.py
│   ├── agent.py
│   ├── model.py
│   └── run.py
├── layer4/
│   ├── __init__.py
│   ├── bass_diffusion.py
│   ├── config.py
│   ├── hotspot_analysis.py
│   ├── npv_analysis.py
│   ├── package_output.py
│   ├── run.py                     ← REPLACED
│   ├── scenario_comparison.py
│   └── sensitivity_analysis.py
├── backend/
│   ├── app.py                     ← REPLACED
│   └── requirements.txt
└── main.py                        ← REPLACED
```

---

## How to Apply

1. **Copy `layer1/`** folder into your `DTW/` root (alongside `layer2/`, `layer3/`, etc.)

2. **Replace `main.py`** with the fixed version

3. **Replace `layer4/run.py`** with the fixed version

4. **Replace `backend/app.py`** with the fixed version (or create `backend/` if it doesn't exist)

---

## Running the CLI

```bash
cd DTW

# With IBM HR CSV (Layer 1 runs automatically)
python main.py WA_Fn-UseC_-HR-Employee-Attrition.csv --output-dir results

# With pre-synthesised workforce CSV (Layer 1 skipped automatically)
python main.py layer2/workforce_v2_1000.csv --output-dir results

# Fast run — skip the OAT sensitivity sweep
python main.py test.csv --output-dir results --skip-sensitivity

# Very fast run — skip sensitivity AND Layer 4
python main.py test.csv --output-dir results --skip-l4

# Custom agent count
python main.py test.csv --output-dir results --n-agents 500
```

---

## Running the API

```bash
cd DTW

# Install backend deps (only needed once)
pip install -r backend/requirements.txt

# Start the API
python backend/app.py
# → http://localhost:5000
```

The frontend uploads the CSV to `POST /api/simulate` with:
- `file` — the IBM HR CSV
- `scenarios` — e.g. `"A,B,C"` or `"B,C"`
- `n_agents` — optional, default 1000
- `skip_sensitivity` — optional, default false

---

## Notes on Simulation Speed (API)

The API uses `n_runs=15` instead of 30 for Layer 3 to keep response time
under ~90 seconds for 1000 agents.  The CLI uses `--l3-runs 30` by default.
OAT sensitivity is always skipped in the API.

For the CLI, `--skip-sensitivity` cuts total runtime roughly in half.