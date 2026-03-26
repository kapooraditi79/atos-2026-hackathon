# Digital Twin Workforce (DTW)
---
<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white"/>
  <img alt="Framework" src="https://img.shields.io/badge/ABM-Mesa_3.x-green?logo=python"/>
  <img alt="Frontend" src="https://img.shields.io/badge/frontend-React_+_Plotly.js-purple?logo=react"/>
  <img alt="API" src="https://img.shields.io/badge/api-Flask-grey?logo=flask"/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-orange"/>
</p>

---
The Digital Twin Workforce (DTW) is a strategic simulation engine designed to model organizational behavior during technological transformations. By creating high-fidelity digital twins of 1,000+ employees, DTW enables leaders to test different rollout strategies (Chatbot-led, Phased, Top-down) and predict their impact on adoption, attrition, and financial ROI.

## 4-Layer Architecture

DTW operates as a modular pipeline, transforming raw HR data into actionable strategic insights:

1. **Layer 1: Workforce Synthesis** — Transforms standard HR CSVs (e.g., IBM Attrition dataset) into a rich 'Digital Twin' workforce with personas, digital dexterity scores, and network influence.
2. **Layer 2: Cluster & Network Validation** — Maps employees into social clusters (Remote-First, Mainstream, Reluctant) and builds the organizational influence graph.
3. **Layer 3: Agentic Simulation** — A multi-agent simulation where individual agents decide to adopt or resist based on social signals, support friction, and management pressure.
4. **Layer 4: Strategic Analytics** — Applies Bass Diffusion modeling, NPV analysis, and resistance hotspot detection to derive the final verdict.

---

## Latest Strategic Analysis

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


## Executive Summary

DTW creates **high-fidelity digital twins of 1,000+ employees** from standard HR data (e.g. IBM Employee Attrition CSV) and runs agent-based simulations to predict how different AI-tool rollout strategies impact **adoption, attrition, productivity, and financial ROI**.

Three competing rollout strategies are simulated head-to-head:

|  Scenario  | Strategy                     |    Adoption    |    24-Month NPV    |  Hotspots  |
| :---------: | ---------------------------- | :-------------: | :----------------: | :---------: |
| **A** | Big-bang + Chatbot           |      33.6%      |     £−32.5M     |     56     |
| **B** | Phased + Human + Training ✅ | **99.9%** | **£+18.5M** | **0** |
| **C** | Pilot + Strong Management    |      82.7%      |     £−12.2M     |    High    |

> **Verdict:** Scenario B — a phased rollout with human support and structured training — is the **dominant strategy**. The upfront £850k investment pays for itself via near-total adoption and massive steady-state productivity gains.

---

## 🏛️ Architecture Overview

DTW is a **modular, 4-layer pipeline** with a Flask API backend and a React + Plotly.js interactive dashboard.

```
main.py  ──▶  Layer 1  ──▶  Layer 2  ──▶  Layer 3  ──▶  Layer 4  ──▶  Dashboard
(Orchestrator)  (Synthesis)  (Validation)  (Simulation)  (Analytics)   (React+Plotly)
```

```
DTW/
├── main.py                      # Top-level CLI orchestrator for the full pipeline
├── pyproject.toml               # Project metadata & Python dependencies (uv/pip)
├── requirements.txt             # Alternative pip requirements
├── test.csv                     # IBM HR Employee Attrition dataset (input)
├── frontend.jsx                 # React + Plotly.js interactive dashboard (single-file)
│
├── layer1/                      # LAYER 1 — Workforce Synthesis
│   ├── __init.py                # Public API: generate_workforce()
│   └── generate_workforce.py    # MVN sampling → 1,000 synthetic agents from IBM HR data
│
├── layer2/                      # LAYER 2 — Cluster Validation & Network Building
│   ├── __init__.py              # Public API: run_all_validations(), build_all()
│   ├── validate_clusters.py     # GMM/K-Means ARI, monotonicity, correlation checks
│   ├── build_simulation_inputs.py # TAM scores, Rogers thresholds, collab graph, PersonaEngine
│   ├── workforce_v2_1000.csv    # Pre-generated 1,000-agent workforce (cached)
│   ├── covariance_by_persona.pkl # Pickled per-persona covariance matrices
│   ├── collab_graph.pkl         # Pickled NetworkX collaboration graph
│   └── cluster_validation_final.png # Validation visualisation
│
├── layer3/                      # LAYER 3 — Agent-Based Simulation Engine
│   ├── __init__.py              # Public API: run_scenarios()
│   ├── agent.py                 # WorkforceAgent: TAM adoption logic, ticket generation
│   ├── model.py                 # WorkforceModel: Mesa ABM model, network dynamics
│   └── run.py                   # Monte Carlo orchestrator for 3 scenarios (A/B/C)
│
├── layer4/                      # LAYER 4 — Strategic Analytics Engine
│   ├── __init__.py              # Public API: run_analytics()
│   ├── config.py                # Shared constants (NPV params, scenario configs, costs)
│   ├── bass_diffusion.py        # Analytic 1: Bass Diffusion model fitting (p, q, M, t*)
│   ├── npv_analysis.py          # Analytic 2: 4-component DCF / Net Present Value
│   ├── hotspot_analysis.py      # Analytic 3: Resistance hotspot detection & interventions
│   ├── sensitivity_analysis.py  # Analytic 4: OAT sensitivity (tornado chart)
│   ├── scenario_comparison.py   # Analytic 5: CIO decision table
│   ├── package_output.py        # Analytic 6: Unified JSON packaging for Layer 5 / dashboard
│   └── run.py                   # Analytics orchestrator (runs all 6 analyics steps)
│
├── backend/                     # Flask REST API
│   └── app.py                   # POST /api/simulate — end-to-end pipeline via HTTP
│
└── results/                     # Pipeline output directory
    └── layer2/ → layer3/ → layer4/ (generated at runtime)
```

---

## Layer-by-Layer Deep Dive

### Layer 1 — Workforce Synthesis (`layer1/generate_workforce.py`)

**Purpose:** Transform raw IBM HR data into 1,000 high-fidelity "digital twin" agents, each with 30+ behavioural attributes.

**How it works:**

1. **IBM HR Signal Mapping** — Reads standard columns (`JobSatisfaction`, `PerformanceRating`, `Attrition`, `OverTime`, etc.) and produces a persona distribution using a **Rogers diffusion curve**.
2. **Multivariate Normal (MVN) Sampling** — For each persona, 8 core dimensions are sampled from a parameterised MVN distribution with a **cross-persona correlation matrix** (manually calibrated `8×8` matrix):
   - `satisfaction_score`, `productivity_baseline`, `resistance_propensity`
   - `training_times_yr`, `digital_dexterity`, `collab_density`
   - `app_activation_rt`, `enps_score`
3. **Conditional Signal Generation** — Derived metrics (IT tickets, frustration, crash rate, session duration, support dependency, etc.) are sampled from per-persona Poisson / Beta / Normal distributions conditioned on the MVN core.
4. **Composite Metrics** — `digital_dexterity`, `friction_level`, `frustration_level`, `support_dependency`, and `churn_risk_flag` are computed as weighted composites.

**5 Personas (Rogers Diffusion):**

| Persona             | Typical Share |     Resistance     | Digital Dexterity | eNPS |
| ------------------- | :-----------: | :----------------: | :---------------: | :--: |
| Tech Pioneer        |      ~3%      |     Low (0.12)     |    High (8.8)    | +72 |
| Power User          |     ~13%     |     Low (0.22)     |    High (7.9)    | +60 |
| Pragmatic Adopter   |     ~30%     |   Medium (0.45)   |   Medium (5.5)   | +24 |
| Remote-First Worker |     ~16%     | Medium-High (0.55) |   Medium (5.1)   | +20 |
| Reluctant User      |     ~38%     |    High (0.80)    |     Low (3.1)     | −18 |

**Output:** `pandas.DataFrame` with 1,000 rows × 31 columns.

---

### Layer 2 — Cluster Validation & Simulation Inputs (`layer2/`)

**Purpose:** Validate the synthetic workforce, build the organisational influence network, and create the `PersonaEngine` that initialises every agent for the simulation.

**Pipeline (5 steps):**

| Step | Module                         | What it does                                                                                                                                                                                                                                                                                                  |
| ---- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `validate_clusters.py`       | **GMM & K-Means ARI** — Fits 5-component GMM and K-Means, compares to persona labels via Adjusted Rand Index. Validates **monotonicity** (persona means respect design-spec ordering) and **correlation structure** (dexterity ↔ activation positive, resistance ↔ activation negative). |
| 2    | `build_simulation_inputs.py` | **TAM Scores** — Computes `PU_INITIAL` (Perceived Usefulness) and `PEOU_INITIAL` (Perceived Ease of Use) as min-max-scaled composites from productivity, activation rate, dexterity, and friction.                                                                                                 |
| 3    | `build_simulation_inputs.py` | **Rogers Thresholds** — Assigns strictly-increasing thresholds for `Awareness → Interest → Trial → Adoption` using `10 − persona_dex_mean` × stage multipliers (0.3, 0.5, 0.9).                                                                                                               |
| 4    | `build_simulation_inputs.py` | **Collaboration Network** — Builds a probabilistic `NetworkX` graph (~12 avg degree) with **homophily bias** (1.5× same-persona bonus).                                                                                                                                                       |
| 5    | `build_simulation_inputs.py` | **Resistance Amplifiers** — Flags top-15% Reluctant Users by **betweenness centrality** as network amplifiers capable of broadcasting frustration.                                                                                                                                               |

**PersonaEngine class** — Wraps all artefacts (workforce DataFrame, covariance matrices, collaboration graph) and provides `get_agent_init(employee_id) → dict` for initialising each agent in Layer 3.

**Validation gates (hard-fail checks):**

- Monotonicity across all 5 personas
- Correlation sign constraints
- Soft warnings for GMM ARI, K-Means ARI, persona balance

**Output artefacts:**

- `workforce_enriched.csv` — Full DataFrame with TAM scores, thresholds, gmm_cluster
- `covariance_by_persona.pkl` — Pickled per-persona covariance matrices
- `collab_graph.pkl` — Pickled NetworkX graph (~6,000 edges)

---

### Layer 3 — Agent-Based Simulation Engine (`layer3/`)

**Purpose:** Run a full 52-week, 1,000-agent agent-based model (ABM) under 3 competing rollout strategies using Monte Carlo sampling.

#### `agent.py` — WorkforceAgent

Each agent is a **Mesa agent** with the following weekly step:

```
step() → _compute_tam() → _update_adoption_stage() → _generate_tickets()
       → _update_productivity() → _decay_frustration()
```

**TAM (Technology Acceptance Model) computation:**

```
PEOU = 0.45·(dexterity/10) + 0.25·(training) + 0.30·(LMS completion)
PU   = 0.50·(1−tool_complexity) + 0.15·(global_adoption) + 0.30·(satisfaction) − support_drag
SN   = local_adoption · (1−resistance) · eNPS · collab_weight
AI   = 0.50·PU + 0.30·PEOU + 0.20·SN + persona_boost + noise
```

**Adoption stages:** `Awareness → Interest → Trial → Adoption → Advocacy`

- Advance: agent must sustain AI > advance_threshold for **3 consecutive weeks**
- Revert: if AI < 0.15 and frustration > revert_threshold, drop one stage

**Support model effects:**

| Parameter         | Chatbot | Hybrid | Human |
| ----------------- | :-----: | :----: | :---: |
| Support Drag      |  0.05  |  0.03  | 0.01 |
| Adoption Friction |  0.05  |  0.06  | 0.02 |
| Ticket Fail Rate  |   38%   |  22%  |  10%  |
| Ticket Deflection |   45%   |  25%  |  0%  |

#### `model.py` — WorkforceModel

- Network seeded from GMM clusters (5 intra-cluster peers each)
- **Dynamic network evolution:** Advocates link to trialing agents; frustrated agents prune edges
- **Amplifier detection:** Top-10% by betweenness centrality → `is_amplifier=1`
- 6 model-level reporters: `adoption_rate`, `productivity_delta`, `avg_frustration`, `ticket_volume`, `resistance_index`, `exs_score`

#### `run.py` — Monte Carlo Orchestrator

- Runs each scenario with **30 Monte Carlo seeds** (15 for API, to keep sub-90s)
- Aggregates per-week statistics: mean, p05, p95 across runs
- Computes `productivity_delta_true` (change from week-0 baseline)
- Cross-scenario validation: ensures Scenario B dominates A by week 8

**Three Scenarios:**

| Scenario | Support | Training Intensity | Manager Signal |
| :------: | ------- | :----------------: | :------------: |
|    A    | Chatbot |        10%        |      0.40      |
|    B    | Human   |        70%        |      0.60      |
|    C    | Hybrid  |        45%        |      0.80      |

**Output:** `{key: (summary_df, agent_df)}`, plus CSVs and Parquet files.

---

### Layer 4 — Strategic Analytics Engine (`layer4/`)

**Purpose:** Transform raw simulation outputs into actionable C-suite insights via 6 analytics steps.

#### Analytic 1: Bass Diffusion (`bass_diffusion.py`)

Fits the **Bass Diffusion Model** `F(t) = F₀ + (M−F₀) · (1−e^(−(p+q)t)) / (1 + (q/p)·e^(−(p+q)t))` to each scenario's adoption curve using `scipy.optimize.curve_fit`.

**Key outputs:** Innovation coefficient `p`, imitation coefficient `q`, ceiling `M`, peak inflection point `t*`, goodness-of-fit `R²`.

**Persona archetypes:**

- `q/p > 50` → *"The Social Butterfly"* (viral, organic growth)
- `t* < 3` → *"The Drill Sergeant"* (mandated, forced adoption)
- Otherwise → *"The Balanced Rollout"*

#### Analytic 2: Net Present Value (`npv_analysis.py`)

A **4-component DCF model** over a 24-month (104-week) horizon:

```
NPV = Investment + Σ(PV of weekly cash flows) − Attrition Cost

Weekly CF = (productivity_delta × 1000 × £1,200/wk) − (tickets × cost_per_ticket)
```

| Constant          | Value             |
| ----------------- | ----------------- |
| WACC              | 10% annual        |
| Headcount         | 1,000             |
| Avg Weekly Salary | £1,200           |
| Replacement Cost  | £45,000/employee |
| P(Churn)          | 22%               |

Year 2 is extrapolated from the steady-state weeks 40–51.

#### Analytic 3: Resistance Hotspots (`hotspot_analysis.py`)

**Definition:** `mean(adoption_stage) < 1.5 AND mean(frustration) > 0.35` during weeks 30–51.

- Clusters hotspot agents by **GMM cluster ID**
- Traces root causes to support model parameters (`p_fail`, `support_drag`, `adoption_friction`)
- Generates **data-driven intervention recommendations** (e.g., "Improve chatbot KB to reduce p_fail from 38% to 20%")

#### Analytic 4: OAT Sensitivity (`sensitivity_analysis.py`)

**One-At-A-Time (OAT)** tornado analysis from Scenario A baseline:

| Parameter          | Range            | Method         |
| ------------------ | ---------------- | -------------- |
| Training Intensity | 0.00 → 1.00     | Re-run Layer 3 |
| Manager Signal     | 0.20 → 0.80     | Re-run Layer 3 |
| Tool Complexity    | 0.80 → 0.40     | Re-run Layer 3 |
| Support Model      | chatbot → human | Re-run Layer 3 |

Each perturbation runs 5 Monte Carlo seeds × 52 steps. Identifies the **single biggest lever** for adoption.

#### Analytic 5: Scenario Comparison (`scenario_comparison.py`)

Builds a **CIO decision table** comparing all scenarios across 10 metrics:
`Adoption Gain`, `90% CI Width`, `24-Month NPV`, `ROI%`, `Investment`, `Attrition Risk`, `Steady-State CF/wk`, `Frustration Peak`, `Disruption Area`, `Resistance Hotspots`.

#### Analytic 6: Unified JSON Packaging (`package_output.py`)

Packages all analytics into a single `layer4_output.json` payload consumed by the React dashboard, including:

- Per-scenario adoption curves, Bass params, NPV breakdown, hotspot clusters
- Cross-scenario comparison table and tornado chart data
- Final recommendation with best-NPV scenario

---

## Backend API (`backend/app.py`)

A **Flask REST API** that exposes the full pipeline via HTTP.

| Endpoint          | Method | Description                                  |
| ----------------- | ------ | -------------------------------------------- |
| `/api/health`   | GET    | Health check — returns `{"status": "ok"}` |
| `/api/simulate` | POST   | Run the full L1→L4 pipeline                 |

**POST `/api/simulate`** — multipart/form-data:

| Field                | Type    | Description                                     |
| -------------------- | ------- | ----------------------------------------------- |
| `file`             | File    | IBM HR CSV or pre-synthesised workforce CSV     |
| `scenarios`        | String  | Comma-separated scenarios (default:`"A,B,C"`) |
| `n_agents`         | Integer | Number of agents to synthesise (default: 1000)  |
| `skip_sensitivity` | Boolean | Skip OAT sensitivity (default: false)           |
| `scenario_configs` | JSON    | Override per-scenario slider values from the UI |

**Smart CSV detection:** Automatically detects whether the uploaded CSV is a raw IBM HR dataset (triggers Layer 1) or a pre-synthesised workforce CSV (skips Layer 1).

**Slider overrides:** The frontend sends UI slider values (`tool_complexity`, `training_intensity`, `manager_signal`, `support_model`) which are deep-merged into the default scenario configs before running Layers 3 & 4.

---

## Frontend Dashboard (`frontend.jsx`)

A **single-file React SPA** (~1,100 lines) with **Plotly.js** interactive charts.

**Key features:**

- **Drag-and-drop CSV upload** with file preview and IBM HR detection
- **Scenario config panel** — live sliders for tool complexity, training intensity, manager signal, and support model selector (per scenario A/B/C)
- **6 interactive Plotly.js charts:**
  - **Adoption S-Curve** — mean ± 90% CI band for all scenarios
  - **3D Persona Explorer** — adoption × training × manager signal scatter
  - **Frustration Heatmap** — persona × week friction grid
  - **NPV Waterfall** — 24-month NPV comparison bars
  - **Persona Radar** — 5-persona adoption polygon overlay
  - **Animated Adoption Race** — horizontal bar-chart race with play/pause/scrub controls
- **KPI tiles** — final adoption %, productivity Δ%, NPV, hotspot count per scenario
- **Winner banner** — automatically highlights the best-NPV scenario
- **Pipeline loading screen** — animated step-by-step progress indicator (L1 → L4)
- **IBM Plex Sans / Mono** typography, light enterprise design system

---

## Getting Started

### Prerequisites

* Python 3.10+
* Recommended: `uv` or `pip` for dependency management.
* Frontend intialization:

```shell
 npm create vite@latest frontend -- --template react
# copy the contents of frontend.jsx into frontend/src/App.jsx
# for subsequent runs
cd frontend
npm run dev
```

* Backend Initialization
  ```shell
  # in root folder
  uv run python -m backend.app  
  ```


### Output Layout

```
<output-dir>/
├── layer2/
│   ├── workforce_enriched.csv
│   ├── covariance_by_persona.pkl
│   └── collab_graph.pkl
├── layer3/outputs/
│   ├── output_scenario_a.csv      # Weekly summary (mean adoption, frustration, etc.)
│   ├── output_scenario_b.csv
│   ├── output_scenario_c.csv
│   ├── agents_a.parquet           # Per-agent-per-week state (for hotspot detection)
│   ├── agents_b.parquet
│   └── agents_c.parquet
└── layer4/
    ├── bass_diffusion.json        # Bass model params (p, q, M, t*, R²)
    ├── npv_analysis.json          # 24-month NPV per scenario
    ├── hotspot_analysis.json      # Resistance clusters + interventions
    ├── sensitivity_analysis.json  # OAT tornado chart data
    ├── comparison_table.json      # CIO decision table
    └── layer4_output.json         # Unified payload for the React dashboard
```

---

## Key Technical Decisions

| Decision                                              | Rationale                                                                                                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MVN sampling with shared correlation matrix** | Preserves realistic inter-feature relationships across personas (e.g., high dex ↔ low resistance), producing statistically valid synthetic data. |
| **Mesa framework for ABM**                      | Industry-standard Python ABM library with built-in data collection and agent scheduling.                                                          |
| **Per-cluster adoption thresholds**             | Prevents unrealistic uniform behaviour — Reluctant Users have lower advance thresholds (0.35) to allow non-zero but lagging adoption.            |
| **3-week persistence rule**                     | Agents must sustain AI > threshold for 3 weeks before advancing, dampening unrealistic cascade effects.                                           |
| **Monte Carlo with 30 seeds**                   | Captures stochastic variation; 90% CI band quantifies outcome uncertainty for risk-averse decision-makers.                                        |
| **Bass Diffusion fitting**                      | Compresses 52-week curves into 3 interpretable parameters (p, q, t*), enabling CIO-level strategic framing.                                       |
| **4-component NPV model**                       | Separates productivity uplift, support costs, investment, and attrition into auditable financial components.                                      |
| **Betweenness-centrality amplifiers**           | Identifies structurally influential resistors who can spread frustration across multiple network clusters.                                        |

---

*© 2026 Digital Twin Workforce Project*

