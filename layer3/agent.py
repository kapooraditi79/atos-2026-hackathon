"""
layer3/agent.py
────────────────
WorkforceAgent: one simulated employee in the ABM.

No file I/O, no CSV references.  All state is passed in via `row`
and `scenario_config` at construction time.
"""

from __future__ import annotations

import mesa
import numpy as np
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from layer3.model import WorkforceModel

STAGE_NAMES = ["Awareness", "Interest", "Trial", "Adoption", "Advocacy"]

# Per-cluster thresholds:
# Cluster 3 (Reluctant User) deliberately lower to allow non-zero adoption
# that still lags behind other clusters realistically.
ADVANCE_THRESHOLD = {0: 0.52, 1: 0.44, 2: 0.52, 3: 0.35, 4: 0.48}
REVERT_THRESHOLD  = {0: 0.18, 1: 0.10, 2: 0.20, 3: 0.25, 4: 0.15}


class WorkforceAgent(mesa.Agent):

    def __init__(self, unique_id, model, row, scenario_config: dict):
        super().__init__(model)  # type: ignore[arg-type]
        self.unique_id = unique_id

        # ── 8 MVN generative dimensions ──────────────────────────────────────
        self.digital_dexterity  = float(row["digital_dexterity"])
        self.training_norm      = float(row["training_times_yr"]) / 6.0
        self.lms_completion     = float(row["lms_completion"])
        self.satisfaction_norm  = float(row["satisfaction_score"]) / 10.0
        self.resistance         = float(row["resistance_propensity"])
        self.enps_norm          = (float(row["enps_score"]) + 100) / 200
        self.collab_density     = float(row["collab_density"])

        # ── Other state ───────────────────────────────────────────────────────
        self.persona             = row["persona"]
        self.support_dependency  = float(row["support_dependency"])
        self.frustration         = float(row["frustration_level"])
        self.productivity        = float(row["productivity_baseline"])
        self.adoption_stage: int = self._init_stage(float(row["app_activation_rt"]))
        self.tickets_this_week   = 0
        self.weeks_above_threshold = 0
        self.AI                  = 0.0

        # ── GMM cluster + churn risk ──────────────────────────────────────────
        self.gmm_cluster = int(row["gmm_cluster"])
        self.churn_risk  = int(row["churn_risk_flag"])
        self.is_amplifier = 0  # set by model._compute_amplifiers() after network seeding

        # ── Scenario config ───────────────────────────────────────────────────
        self.tool_complexity  = scenario_config.get("tool_complexity", 0.65)
        self.support_model    = scenario_config.get("support_model", "human")
        self.manager_signal   = scenario_config.get("manager_signal", 0.50)
        self.training_boost   = scenario_config.get("training_intensity", 0.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_stage(self, activation_rate: float) -> int:
        # Cap at stage 2 (Trial) so every persona still has visible adoption
        # growth during the simulation — prevents Pioneers/Power Users from
        # starting fully adopted (stage 4) and flatlines at 100% from week 0.
        if activation_rate > 0.65:   return 2
        elif activation_rate > 0.45: return 1
        else:                        return 0

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self):
        self._compute_tam()
        self._update_adoption_stage()
        self._generate_tickets()
        self._update_productivity()
        self._decay_frustration()

    # ── TAM ───────────────────────────────────────────────────────────────────

    def _compute_tam(self):
        from layer3.model import WorkforceModel  # deferred to avoid circular import
        m = cast("WorkforceModel", self.model)
        global_adoption = m.get_adoption_rate()

        nbrs = list(m.graph.neighbors(self.unique_id))
        if nbrs:
            nbr_lookup    = getattr(m, "_agent_map", {})
            local_adoption = sum(
                1 for uid in nbrs
                if uid in nbr_lookup
                and cast(WorkforceAgent, nbr_lookup[uid]).adoption_stage >= 3
            ) / len(nbrs)
        else:
            local_adoption = global_adoption

        effective_training = min(1.0, self.training_norm + self.training_boost * 0.5)
        PEOU = (
            (self.digital_dexterity / 10) * 0.45
            + effective_training          * 0.25
            + self.lms_completion         * 0.30
        )

        if self.support_model == "chatbot":
            support_drag = 0.05
        elif self.support_model == "hybrid":
            support_drag = 0.03
        else:
            support_drag = 0.01

        # Reduced from 0.30 → 0.15 to dampen the snowball cascade:
        # when early adopters reach stage-4, global_adoption rises and inflates
        # everyone else's AI, causing unrealistic 100% adoption in B & C.
        # The weight is redistributed to satisfaction_norm (personal attitude).
        PU = (
            (1 - self.tool_complexity)  * 0.50
            + global_adoption           * 0.15
            + self.satisfaction_norm    * 0.30
            - support_drag
        )

        collab_weight = 0.5 + self.collab_density * 0.5
        SN = local_adoption * (1 - self.resistance) * self.enps_norm * collab_weight

        # Persona-based resistance offset: makes Reluctant/Remote Users
        # accumulate enough AI to advance, preventing permanently-zero adoption.
        PERSONA_BOOST = {
            "Reluctant User":      0.08,
            "Remote-First Worker": 0.05,
            "Pragmatic Adopter":   0.02,
            "Power User":          0.00,
            "Tech Pioneer":        0.00,
        }
        persona_offset = PERSONA_BOOST.get(self.persona, 0.0)

        # Clipped Gaussian noise makes adoption stochastic — prevents agents
        # from deterministically marching to stage 4 the moment AI > threshold.
        raw_ai = 0.50 * PU + 0.30 * PEOU + 0.20 * SN + persona_offset
        self.AI = float(np.clip(raw_ai + np.random.normal(0, 0.04), 0.0, 1.0))

    # ── Adoption stage ────────────────────────────────────────────────────────

    def _update_adoption_stage(self):
        base_advance = ADVANCE_THRESHOLD[self.gmm_cluster]
        base_revert  = REVERT_THRESHOLD[self.gmm_cluster]

        # All clusters get the same manager bonus rate.
        # A previous value of 0.03 for cluster 3 was too low and prevented
        # Reluctant Users from benefiting from strong management (Scenario C).
        manager_bonus = self.manager_signal * 0.16

        if self.support_model == "chatbot":
            adoption_friction = 0.05
        elif self.support_model == "hybrid":
            adoption_friction = 0.06
        else:
            adoption_friction = 0.02

        advance_threshold        = base_advance - manager_bonus + adoption_friction
        revert_frustration_limit = base_revert - (0.05 if self.churn_risk else 0.0)

        if self.AI > advance_threshold and self.adoption_stage < 4:
            self.weeks_above_threshold += 1
            if self.weeks_above_threshold >= 3:   # raised from 2 → slows cascade
                self.adoption_stage += 1
                self.weeks_above_threshold = 0
        else:
            self.weeks_above_threshold = 0

        if self.AI < 0.15 and self.frustration > revert_frustration_limit:
            self.adoption_stage = max(0, self.adoption_stage - 1)
            self.weeks_above_threshold = 0

    # ── Tickets ───────────────────────────────────────────────────────────────

    def _generate_tickets(self):
        base_lambda      = self.support_dependency * 8
        stage_multiplier = [1.0, 1.1, 1.8, 0.7, 0.4][self.adoption_stage]

        if self.support_model == "chatbot":
            support_deflect = 0.45
        elif self.support_model == "hybrid":
            support_deflect = 0.25
        else:
            support_deflect = 0.0

        lam = (base_lambda / 4) * stage_multiplier * (1 - support_deflect)
        self.tickets_this_week = np.random.poisson(max(lam, 0))

        if self.tickets_this_week > 0:
            if self.support_model == "chatbot":
                p_fail = 0.38
            elif self.support_model == "hybrid":
                p_fail = 0.22
            else:
                p_fail = 0.10
            unresolved = np.random.binomial(self.tickets_this_week, p_fail)
            self.frustration = min(1.0, self.frustration + unresolved * 0.15)

    # ── Productivity ──────────────────────────────────────────────────────────

    def _update_productivity(self):
        stage_delta   = [-0.02, -0.01, -0.05, +0.03, +0.06][self.adoption_stage]
        friction_drag = self.frustration * 0.04
        self.productivity = float(
            np.clip(self.productivity + stage_delta - friction_drag, 0.0, 1.0)
        )
        dex_gain = 0.02 + self.training_boost * 0.03
        self.digital_dexterity = min(10.0, self.digital_dexterity + dex_gain)

    # ── Frustration decay ─────────────────────────────────────────────────────

    def _decay_frustration(self):
        self.frustration *= 0.90