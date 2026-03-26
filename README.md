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
# copy the contents of frontend.jsx into frontend/src/App.jsx
# for subsequent runs
cd frontend
npm run dev
 ```
- Backend Initialization
  ```bash
  # in root folder
  uv run python -m backend.app  
  ```


*© 2026 Digital Twin Workforce Project*
