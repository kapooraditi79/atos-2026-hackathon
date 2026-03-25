import mesa
import numpy as np
from typing import cast

STAGE_NAMES = ['Awareness','Interest','Trial','Adoption','Advocacy']
# agent.py — replace ADVANCE_THRESHOLD with these values
ADVANCE_THRESHOLD = {0: 0.52, 1: 0.44, 2: 0.52, 3: 0.60, 4: 0.48}  # raised +0.08 total to slow S-curve
REVERT_THRESHOLD  = {0: 0.18, 1: 0.10, 2: 0.20, 3: 0.25, 4: 0.15}

class WorkforceAgent(mesa.Agent):
    def __init__(self, unique_id, model, row, scenario_config):
        super().__init__(model) #type:ignore
        self.unique_id = unique_id
        #8 MVN generative dimensions
        self.digital_dexterity = float(row['digital_dexterity'])
        self.training_norm = float(row['training_times_yr']) / 6.0
        self.lms_completion = float(row['lms_completion'])
        self.satisfaction_norm = float(row['satisfaction_score']) / 10.0
        self.resistance = float(row['resistance_propensity'])
        self.enps_norm = (float(row['enps_score'])+100) / 200
        self.collab_density = float(row['collab_density'])
        
        #Other state variables 
        self.persona = row['persona']
        self.support_dependency = float(row['support_dependency'])
        self.frustration = float(row['frustration_level'])
        self.productivity = float(row['productivity_baseline'])
        self.adoption_stage: int = self._init_stage(float(row['app_activation_rt']))
        self.tickets_this_week = 0

        #GMM Cluster + churn risk
        self.gmm_cluster = int(row['gmm_cluster'])
        self.churn_risk = int(row['churn_risk_flag'])

        #Scenario configs
        self.tool_complexity = scenario_config.get('tool_complexity',0.65)
        self.support_model = scenario_config.get('support_model','human')
        self.manager_signal = scenario_config.get('manager_signal',0.50)
        self.training_boost = scenario_config.get('training_intensity',0.0)
        
    def _init_stage(self,activation_rate):
        if activation_rate > 0.85: return 4
        elif activation_rate > 0.65: return 3
        elif activation_rate > 0.45: return 2
        elif activation_rate > 0.25: return 1
        else: return 0

    def step(self):
        self._compute_tam()
        self._update_adoption_stage()
        self._generate_tickets()
        self._update_productivity()
        self._decay_frustration()
        
    #Compute TAM
    def _compute_tam(self):
        from layer3.model import WorkforceModel  # Import moved inside the method
        m = cast(WorkforceModel, self.model)
        global_adoption = m.get_adoption_rate()

        # Use local neighbour adoption when agent has graph edges (wires network into AI)
        nbrs = list(m.graph.neighbors(self.unique_id))
        if nbrs:
            nbr_lookup = getattr(m, '_agent_map', {})
            local_adoption = sum(
                1 for uid in nbrs
                if uid in nbr_lookup
                and cast(WorkforceAgent, nbr_lookup[uid]).adoption_stage >= 3
            ) / len(nbrs)
        else:
            local_adoption = global_adoption

    # PEOU — 3 per-agent inputs (dexterity, training, LMS readiness)
        effective_training = min(1.0, self.training_norm + self.training_boost * 0.5)  # 0.3→0.5
        PEOU = (self.digital_dexterity / 10) * 0.45 \
         + effective_training           * 0.25 \
         + self.lms_completion          * 0.30

    # support_drag: unresolved tickets (PU penalty)
        if self.support_model == 'chatbot':
            support_drag = 0.08
        elif self.support_model == 'hybrid':
            support_drag = 0.03 # Lowered 0.05→0.03 for mid-range C adoption
        else:
            support_drag = 0.0
            
        PU = (1 - self.tool_complexity)    * 0.50 \
       + global_adoption                * 0.30 \
       + self.satisfaction_norm         * 0.20 \
       - support_drag                             # ← Scenario-specific PU drag

    # SN — uses LOCAL neighbour adoption so graph structure matters
        collab_weight = 0.5 + self.collab_density * 0.5
        SN = local_adoption * (1 - self.resistance) * self.enps_norm * collab_weight

        self.AI = 0.50 * PU + 0.30 * PEOU + 0.20 * SN
        
    def _update_adoption_stage(self):
        base_advance = ADVANCE_THRESHOLD[self.gmm_cluster]
        base_revert  = REVERT_THRESHOLD[self.gmm_cluster]

        # Reluctant Users (cluster 3) are less responsive to manager signal
        # Cap the manager bonus for cluster 3 so they never fully overcome resistance
        if self.gmm_cluster == 3:
            manager_bonus = self.manager_signal * 0.03
        else:
            manager_bonus = self.manager_signal * 0.16  # doubled 0.08→0.16 to widen A/B/C spread

        # adoption_friction: makes it harder to cross the threshold in lower-support scenarios
        if self.support_model == 'chatbot':
            adoption_friction = 0.15
        elif self.support_model == 'hybrid':
            adoption_friction = 0.06 # Lowered 0.12→0.06 for mid-range C adoption
        else:
            adoption_friction = 0.0

        advance_threshold        = base_advance - manager_bonus + adoption_friction
        revert_frustration_limit = base_revert - (0.05 if self.churn_risk else 0.0)

        if self.AI > advance_threshold and self.adoption_stage < 4:
            self.adoption_stage += 1
        elif self.AI < 0.15 and self.frustration > revert_frustration_limit:
            self.adoption_stage = max(0, self.adoption_stage - 1)

    def _generate_tickets(self):
        base_lambda      = self.support_dependency * 8
        stage_multiplier = [1.0, 1.1, 1.8, 0.7, 0.4][self.adoption_stage]
        
        # Support model deflection
        if self.support_model == 'chatbot':
            support_deflect = 0.45
        elif self.support_model == 'hybrid':
            support_deflect = 0.25 # Scenario C intermediate deflection
        else:
            support_deflect = 0.0
            
        lam = (base_lambda / 4) * stage_multiplier * (1 - support_deflect)
        self.tickets_this_week = np.random.poisson(max(lam, 0))
        if self.tickets_this_week > 0:
            # Support model failure rate
            if self.support_model == 'chatbot':
                p_fail = 0.38
            elif self.support_model == 'hybrid':
                p_fail = 0.22 # Increased 0.18→0.22 for B/C separation
            else:
                p_fail = 0.10
                
            unresolved = np.random.binomial(self.tickets_this_week, p_fail)
            self.frustration = min(1.0, self.frustration + unresolved * 0.15)

    def _update_productivity(self):
        stage_delta   = [-0.02, -0.01, -0.05, +0.03, +0.06][self.adoption_stage]
        friction_drag = self.frustration * 0.04
        self.productivity = float(np.clip(self.productivity + stage_delta - friction_drag, 0.0, 1.0))
        # Training slowly builds dexterity — per-agent, scenario boosts speed
        dex_gain = 0.02 + self.training_boost * 0.03
        self.digital_dexterity = min(10.0, self.digital_dexterity + dex_gain)

    def _decay_frustration(self):
        # 10% weekly decay of frustration 
        # ensures half life of frustration ~6 weeks
        self.frustration *= 0.90
