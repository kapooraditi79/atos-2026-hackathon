# Digital Twin Workforce (DTW)

The Digital Twin Workforce (DTW) is a strategic simulation engine designed to model organizational behavior during technological transformations. By creating high-fidelity digital twins of 1,000+ employees, DTW enables leaders to test different rollout strategies (Chatbot-led, Phased, Top-down) and predict their impact on adoption, attrition, and financial ROI.

## 🏗️ 4-Layer Architecture

DTW operates as a modular pipeline, transforming raw HR data into actionable strategic insights:

1. **Layer 1: Workforce Synthesis** — Transforms standard HR CSVs (e.g., IBM Attrition dataset) into a rich 'Digital Twin' workforce with personas, digital dexterity scores, and network influence.
2. **Layer 2: Cluster & Network Validation** — Maps employees into social clusters (Remote-First, Mainstream, Reluctant) and builds the organizational influence graph.
3. **Layer 3: Agentic Simulation** — A multi-agent simulation where individual agents decide to adopt or resist based on social signals, support friction, and management pressure.
4. **Layer 4: Strategic Analytics** — Applies Bass Diffusion modeling, NPV analysis, and resistance hotspot detection to derive the final verdict.

---

## 📈 Latest Strategic Analysis

Our latest 12-month simulation (1,000 agents × 52 weeks) compared three distinct rollout strategies:

### **Scenario B: Phased + Human + Training (The Winner)**

* **Adoption**: **99.9%** (Champion scenario)
* **24-Month NPV**: **£+18,507,837**
* **ROI**: **+2,177.4%**
* **Resistance Hotspots**: **0** (Healthy adoption across all persona clusters)
* **Verdict**: The **Dominant Strategy**. Upfront investment in human support (£850k) pays for itself via massive gains in steady-state productivity and near-zero attrition.

### **Scenario A: Big-bang + Chatbot**

* **Adoption**: 33.6%
* **NPV**: £-32,501,772 (Value-destroying)
* **Hotspots**: 56 (Critical resistance in 'Reluctant' and 'Remote' clusters)

### **Scenario C: Pilot + Strong Management**

* **Adoption**: 82.7%
* **NPV**: £-12,226,312 (Negative)
* **Attrition Risk**: High (113 at-risk agents due to 'Management Signal' friction)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Recommended: `uv` or `pip` for dependency management.
- Frontend intialization:

```bash
 npm create vite@latest frontend -- --template react
 ```

### CLI Usage

Run a full simulation from the command line:

```bash
# Full simulation with IBM HR data
python main.py HR_Data.csv --output-dir results

# Fast run (skip sensitivity analysis)
python main.py HR_Data.csv --output-dir results --skip-sensitivity
```

### API & Backend

The DTW engine can be exposed via a Flask API:

# Start the server
python backend/app.py
```

API endpoints include `POST /api/simulate` for custom workforce uploads and scenario testing.

---

## 🛠️ Development & Historical Fixes

* **Layer 1 Isolation**: Fixed `main.py` crash by ensuring Layer 1 runs automatically for raw CSV inputs.
* **Path Resolution**: Corrected Layer 4 parquet resolution to support arbitrary output directories.
* **Performance**: The API uses a calibrated `n_runs=15` to ensure sub-90s execution for 1000 agents.
* **Flexibility**: Added `--skip-sensitivity` to argparse for rapid iteration.

---

*© 2026 Digital Twin Workforce Project*
