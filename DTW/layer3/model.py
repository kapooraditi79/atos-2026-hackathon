"""
layer3/model.py
────────────────
WorkforceModel: Mesa ABM model.

The path to the enriched CSV is passed in at construction time via
`csv_path` — no hardcoded relative paths.
"""

from __future__ import annotations

from pathlib import Path

import mesa
import networkx as nx
import numpy as np
import pandas as pd
from typing import cast

from layer3.agent import WorkforceAgent


class WorkforceModel(mesa.Model):

    def __init__(self, scenario_config: dict, rng, csv_path: str | Path):
        super().__init__(rng=rng)
        self.config = scenario_config
        self.graph  = nx.Graph()

        df = pd.read_csv(str(csv_path))
        for idx, row in df.iterrows():
            agent = WorkforceAgent(idx, self, row, scenario_config)
            self.graph.add_node(agent.unique_id)

        self._seed_network(df)
        self._compute_amplifiers()

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "adoption_rate":      lambda m: m.get_adoption_rate(),
                "productivity_delta": lambda m: m.get_productivity_delta(),
                "avg_frustration":    lambda m: m.get_avg_frustration(),
                "ticket_volume":      lambda m: m.get_weekly_tickets(),
                "resistance_index":   lambda m: m.get_resistance_index(),
                "exs_score":          lambda m: m.get_exs_score(),
            },
            agent_reporters={
                "persona":        "persona",
                "gmm_cluster":    "gmm_cluster",
                "adoption_stage": "adoption_stage",
                "frustration":    "frustration",
                "productivity":   "productivity",
                "churn_risk":     "churn_risk",
                "enps_norm":      "enps_norm",
                "training_norm":  "training_norm",
                "AI":             "AI",
                "is_amplifier":   "is_amplifier",
            },
        )

    # ── Mesa lifecycle ────────────────────────────────────────────────────────

    def step(self):
        self.datacollector.collect(self)
        self._cached_adoption_rate = self._compute_adoption_rate()
        self._agent_map = {a.unique_id: a for a in self.agents}
        self.agents.shuffle_do("step")
        self._update_network()
        self._cached_adoption_rate = None

    def run(self, n_steps: int = 52) -> pd.DataFrame:
        for _ in range(n_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()

    # ── Model-level reporters ─────────────────────────────────────────────────

    def _compute_adoption_rate(self) -> float:
        adopted = sum(1 for a in self.agents if cast(WorkforceAgent, a).adoption_stage >= 3)
        return adopted / len(self.agents)

    def get_adoption_rate(self) -> float:
        cached = getattr(self, "_cached_adoption_rate", None)
        return cached if cached is not None else self._compute_adoption_rate()

    def get_productivity_delta(self) -> float:
        return float(np.mean([cast(WorkforceAgent, a).productivity for a in self.agents]))

    def get_avg_frustration(self) -> float:
        return float(np.mean([cast(WorkforceAgent, a).frustration for a in self.agents]))

    def get_weekly_tickets(self) -> int:
        return int(sum(cast(WorkforceAgent, a).tickets_this_week for a in self.agents))

    def get_resistance_index(self) -> float:
        if self.steps < 8:
            return 0.0
        stuck = self.agents.select(lambda a: cast(WorkforceAgent, a).adoption_stage <= 1)
        return len(stuck) / len(self.agents)

    def get_exs_score(self) -> float:
        scores = [
            (1 - cast(WorkforceAgent, a).frustration)     * 0.35
            + (cast(WorkforceAgent, a).adoption_stage / 4) * 0.35
            + cast(WorkforceAgent, a).productivity         * 0.30
            for a in self.agents
        ]
        return float(np.mean(scores) * 100)

    # ── Network ───────────────────────────────────────────────────────────────

    def _seed_network(self, df: pd.DataFrame):
        for cluster_id in df["gmm_cluster"].unique():
            cluster_agents = df[df["gmm_cluster"] == cluster_id].index.tolist()
            for i in cluster_agents:
                peers = np.random.choice(
                    cluster_agents, size=min(5, len(cluster_agents)), replace=False
                )
                for peer in peers:
                    if i != peer:
                        self.graph.add_edge(i, peer)

    def _compute_amplifiers(self):
        bc = nx.betweenness_centrality(self.graph)
        threshold = np.percentile(list(bc.values()), 90)
        for a in self.agents:
            agent = cast(WorkforceAgent, a)
            agent.is_amplifier = 1 if bc.get(agent.unique_id, 0) >= threshold else 0

    def _update_network(self):
        advocates = list(self.agents.select(lambda a: cast(WorkforceAgent, a).adoption_stage == 4))
        trialing  = list(self.agents.select(lambda a: cast(WorkforceAgent, a).adoption_stage == 2))
        if trialing:
            for adv in advocates:
                peers = np.random.choice(trialing, size=min(2, len(trialing)), replace=False)
                for peer in peers:
                    if not self.graph.has_edge(adv.unique_id, peer.unique_id):
                        self.graph.add_edge(adv.unique_id, peer.unique_id)

        frustrated = list(
            self.agents.select(lambda a: cast(WorkforceAgent, a).frustration > 0.40)
        )
        for agent in frustrated:
            if np.random.random() < 0.35:
                edges = list(self.graph.edges(agent.unique_id))
                if edges:
                    idx = np.random.randint(len(edges))
                    self.graph.remove_edge(*edges[idx])