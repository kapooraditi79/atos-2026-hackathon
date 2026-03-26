"""
layer2/build_simulation_inputs.py
──────────────────────────────────
Transforms a validated workforce DataFrame into all artefacts the
agent-based simulation needs: TAM scores, Rogers thresholds, a
weighted collaboration network, amplifier flags, and the PersonaEngine
class that initialises each agent.

All file I/O is controlled by the caller via `output_dir`.
No CSV_PATH / hardcoded paths live here.
"""

from __future__ import annotations

import pickle
import warnings
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "satisfaction_score",
    "productivity_baseline",
    "resistance_propensity",
    "training_times_yr",
    "digital_dexterity",
    "collab_density",
    "app_activation_rt",
    "enps_score",
]

PERSONA_ORDER = [
    "Tech Pioneer",
    "Power User",
    "Pragmatic Adopter",
    "Remote-First Worker",
    "Reluctant User",
]

# Rogers diffusion stage multipliers — strictly increasing so thresholds are ordered.
STAGE_MULTIPLIERS = {
    "awareness_to_interest": 0.3,
    "interest_to_trial":     0.5,
    "trial_to_adoption":     0.9,
}

TARGET_DEGREE        = 12
SAME_PERSONA_BONUS   = 1.5
CROSS_PERSONA_BONUS  = 1.0
SAME_PERSONA_FRAC    = 0.25
AMPLIFIER_PERCENTILE = 85


# ── Step 1 — Per-persona covariance ───────────────────────────────────────────

def compute_and_save_covariance(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Compute per-persona covariance over the 8 MVN features and save to disk."""
    cov_df = df.groupby("persona")[FEATURE_NAMES].cov()
    path = output_dir / "covariance_by_persona.pkl"
    cov_df.to_pickle(str(path))
    print(f"  ✓ Covariance matrices saved → {path}")
    return cov_df


# ── Step 2 — TAM scores ────────────────────────────────────────────────────────

def compute_tam_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add PU_INITIAL and PEOU_INITIAL columns, both min-max scaled to [0, 10]."""
    df = df.copy()

    pu_raw = (
        df["productivity_baseline"] * 0.50
        + df["app_activation_rt"]   * 0.30
        + df["lms_completion"]      * 0.20
    )
    pu_min, pu_max = pu_raw.min(), pu_raw.max()
    df["PU_INITIAL"] = (pu_raw - pu_min) / (pu_max - pu_min) * 10

    peou_raw = df["digital_dexterity"] * 0.65 - df["friction_level"] * 0.35
    peou_min, peou_max = peou_raw.min(), peou_raw.max()
    df["PEOU_INITIAL"] = (peou_raw - peou_min) / (peou_max - peou_min) * 10

    print(
        f"  ✓ TAM scores — "
        f"PU_INITIAL: [{df['PU_INITIAL'].min():.2f}, {df['PU_INITIAL'].max():.2f}]  "
        f"PEOU_INITIAL: [{df['PEOU_INITIAL'].min():.2f}, {df['PEOU_INITIAL'].max():.2f}]"
    )
    return df


# ── Step 3 — Resistance weights + Rogers thresholds ───────────────────────────

def compute_weights_and_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Add resistance_weight, social_weight, and threshold_* columns."""
    df = df.copy()

    resistance_mean = df.groupby("persona")["resistance_propensity"].transform("mean")
    df["resistance_weight"] = resistance_mean * 0.85

    df["social_weight"] = df["collab_density"] * (1 - df["PU_INITIAL"] / 10)

    persona_dex_mean = df.groupby("persona")["digital_dexterity"].transform("mean")
    base_difficulty  = 10 - persona_dex_mean

    for stage, mult in STAGE_MULTIPLIERS.items():
        df[f"threshold_{stage}"] = base_difficulty * mult

    assert (df["threshold_awareness_to_interest"] < df["threshold_interest_to_trial"]).all()
    assert (df["threshold_interest_to_trial"]     < df["threshold_trial_to_adoption"]).all()
    print("  ✓ Rogers thresholds strictly increasing across all agents")

    return df


# ── Step 4 — Collaboration network ────────────────────────────────────────────

def _solve_alpha(df: pd.DataFrame) -> float:
    n = len(df)
    collab       = df["collab_density"].values
    avg_geo_mean = np.sqrt(collab.mean() ** 2)
    avg_bonus    = (
        (1 - SAME_PERSONA_FRAC) * CROSS_PERSONA_BONUS
        + SAME_PERSONA_FRAC * SAME_PERSONA_BONUS
    )
    alpha = TARGET_DEGREE / ((n - 1) * avg_geo_mean * avg_bonus)
    print(f"    Network alpha (density scaler): {alpha:.6f}")
    return alpha


def build_collab_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a probabilistic collaboration network with homophily bias."""
    alpha    = _solve_alpha(df)
    n        = len(df)
    G        = nx.Graph()
    G.add_nodes_from(range(n))

    personas = df["persona"].values
    collab   = df["collab_density"].values

    np.random.seed(42)
    for i in range(n):
        for j in range(i + 1, n):
            geo   = np.sqrt(collab[i] * collab[j])
            bonus = SAME_PERSONA_BONUS if personas[i] == personas[j] else CROSS_PERSONA_BONUS
            if np.random.random() < alpha * geo * bonus:
                G.add_edge(i, j)

    degrees = [d for _, d in G.degree()]
    print(
        f"  ✓ Collaboration graph — "
        f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}  "
        f"avg_degree={np.mean(degrees):.1f}"
    )
    return G


# ── Step 5 — Resistance amplifiers ────────────────────────────────────────────

def flag_amplifiers(df: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    """Flag top-AMPLIFIER_PERCENTILE% Reluctant Users by betweenness as amplifiers."""
    df = df.copy()

    betweenness       = nx.betweenness_centrality(G)
    reluctant_mask    = df["persona"] == "Reluctant User"
    reluctant_indices = df[reluctant_mask].index
    bt_scores         = {i: betweenness[i] for i in reluctant_indices}
    threshold         = np.percentile(list(bt_scores.values()), AMPLIFIER_PERCENTILE)

    df["is_amplifier_network"] = False
    amplifier_idx = [i for i in reluctant_indices if bt_scores[i] >= threshold]
    df.loc[amplifier_idx, "is_amplifier_network"] = True

    print(
        f"  ✓ Resistance amplifiers — "
        f"{df['is_amplifier_network'].sum()} / {reluctant_mask.sum()} Reluctant Users"
    )
    return df


# ── PersonaEngine ──────────────────────────────────────────────────────────────

class PersonaEngine:
    """
    Initialises agent state dictionaries for the agent-based simulation.

    Parameters
    ----------
    df             : enriched workforce DataFrame (employee_id as column)
    cov_by_persona : output of compute_and_save_covariance()
    graph          : output of build_collab_graph()
    """

    def __init__(self, df: pd.DataFrame, cov_by_persona: pd.DataFrame, graph: nx.Graph):
        self.df    = df.set_index("employee_id")
        self.cov   = cov_by_persona
        self.graph = graph
        self.id_to_node = {eid: idx for idx, eid in enumerate(df["employee_id"])}

    def get_agent_init(self, employee_id) -> dict:
        """Return a fully-initialised state dict for one agent."""
        row      = self.df.loc[employee_id]
        node_idx = self.id_to_node[employee_id]

        return {
            "persona":        row["persona"],
            "pu":             row["PU_INITIAL"],
            "peou":           row["PEOU_INITIAL"],
            "adoption_stage": "Awareness",
            "resistance":     row["resistance_propensity"],
            "resistance_w":   row["resistance_weight"],
            "social_w":       row["social_weight"],
            "thresholds": {
                "awareness_to_interest": row["threshold_awareness_to_interest"],
                "interest_to_trial":     row["threshold_interest_to_trial"],
                "trial_to_adoption":     row["threshold_trial_to_adoption"],
            },
            "ticket_lambda_wk": row["tickets_per_month"] / 4.33,
            "neighbours":     list(self.graph.neighbors(node_idx)),
            "is_amplifier":   row["is_amplifier_network"],
        }


# ── Validation ─────────────────────────────────────────────────────────────────

_EXPECTED_KEYS = [
    "persona", "pu", "peou", "adoption_stage", "resistance",
    "resistance_w", "social_w", "thresholds", "ticket_lambda_wk",
    "neighbours", "is_amplifier",
]


def validate_engine(engine: PersonaEngine, df: pd.DataFrame, spot_check_n: int = 100) -> None:
    all_ids = df["employee_id"].tolist()

    for eid in all_ids:
        init = engine.get_agent_init(eid)
        for key in _EXPECTED_KEYS:
            assert key in init,            f"Missing key: {key} for {eid}"
            assert init[key] is not None,  f"None value: {key} for {eid}"
    print(f"  ✓ Completeness: all {len(all_ids)} agents have {len(_EXPECTED_KEYS)} keys, no Nones")

    for eid in all_ids[:spot_check_n]:
        assert engine.get_agent_init(eid) == engine.get_agent_init(eid), \
            f"Inconsistent results for {eid}"
    print(f"  ✓ Consistency: spot-checked {spot_check_n} agents")

    personas = [engine.get_agent_init(eid)["persona"] for eid in all_ids]
    counts   = Counter(personas)
    print("  ✓ Coverage:")
    for p in PERSONA_ORDER:
        print(f"      {p}: {counts.get(p, 0)}")


# ── Public entry point ─────────────────────────────────────────────────────────

def build_all(df: pd.DataFrame, output_dir: Path | None = None) -> PersonaEngine:
    """
    Run the full build pipeline and return a ready PersonaEngine.

    Parameters
    ----------
    df         : validated workforce DataFrame
    output_dir : directory for artefact files (covariance pkl, graph pkl, enriched CSV).
                 Defaults to the layer2 package directory.
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAYER 2 — BUILD SIMULATION INPUTS")
    print("=" * 60)

    cov_df = compute_and_save_covariance(df, output_dir)
    df     = compute_tam_scores(df)
    df     = compute_weights_and_thresholds(df)
    G      = build_collab_graph(df)

    graph_path = output_dir / "collab_graph.pkl"
    with open(str(graph_path), "wb") as f:
        pickle.dump(G, f)
    print(f"  ✓ Collaboration graph saved → {graph_path}")

    df = flag_amplifiers(df, G)

    enriched_csv = output_dir / "workforce_enriched.csv"
    df.to_csv(str(enriched_csv), index=False)
    print(f"  ✓ Enriched dataset saved → {enriched_csv}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    engine = PersonaEngine(df, cov_df, G)

    print()
    print("=" * 60)
    print("LAYER 2 — PERSONA ENGINE VALIDATION")
    print("=" * 60)
    validate_engine(engine, df)

    print(f"\n{'═' * 60}")
    print("LAYER 2 COMPLETE ✓")
    print(f"{'═' * 60}")
    return engine