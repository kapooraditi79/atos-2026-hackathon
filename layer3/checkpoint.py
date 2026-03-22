# checkpoint.py — put this in your layer3/ folder and run it
import pandas as pd
import numpy as np
from agent import WorkforceAgent

# Load one real row from the CSV
df = pd.read_csv('../workforce_v2_1000.csv')

# Pick a Reluctant User and a Tech Pioneer
reluctant_row = df[df['persona'] == 'Reluctant User'].iloc[0]
pioneer_row   = df[df['persona'] == 'Tech Pioneer'].iloc[0]

scenario = {
    'tool_complexity': 0.65,
    'support_model': 'human',
    'manager_signal': 0.50,
    'training_intensity': 0.10
}

# ── Test 1: Can we instantiate without crashing? ──────────────────────────────
# Use a mock model so self.model doesn't crash
class MockModel:
    def get_adoption_rate(self):
        return 0.15   # pretend 15% of workforce adopted
    
    def register_agent(self, agent):
        pass

reluctant = WorkforceAgent(0, MockModel(), reluctant_row, scenario)
pioneer   = WorkforceAgent(1, MockModel(), pioneer_row,   scenario)

print("=== TEST 1: Attributes loaded correctly ===")
print(f"Reluctant — dexterity: {reluctant.digital_dexterity:.2f}, resistance: {reluctant.resistance:.2f}, gmm_cluster: {reluctant.gmm_cluster}")
print(f"Pioneer   — dexterity: {pioneer.digital_dexterity:.2f},   resistance: {pioneer.resistance:.2f},   gmm_cluster: {pioneer.gmm_cluster}")
print(f"Reluctant adoption_stage: {reluctant.adoption_stage}  (expect 0 or 1)")
print(f"Pioneer   adoption_stage: {pioneer.adoption_stage}    (expect 3 or 4)")

# ── Test 2: TAM gives differentiated AI scores ────────────────────────────────
reluctant._compute_tam()
pioneer._compute_tam()

print("\n=== TEST 2: TAM Adoption Intention scores ===")
print(f"Reluctant AI: {reluctant.AI:.3f}  (expect ~0.24-0.28)")
print(f"Pioneer   AI: {pioneer.AI:.3f}    (expect ~0.38-0.44)")
print(f"Pioneer AI > Reluctant AI? {pioneer.AI > reluctant.AI}  ← must be True")

# ── Test 3: Stage transition logic ────────────────────────────────────────────
print("\n=== TEST 3: Advance threshold check ===")
reluctant_thresh = 0.52 - (0.50 * 0.08)
pioneer_thresh   = 0.36 - (0.50 * 0.08)
print(f"Reluctant threshold: {reluctant_thresh:.2f} | AI: {reluctant.AI:.3f} | Will advance? {reluctant.AI > reluctant_thresh}")
print(f"Pioneer   threshold: {pioneer_thresh:.2f}   | AI: {pioneer.AI:.3f}   | Will advance? {pioneer.AI > pioneer_thresh}")

# ── Test 4: Run one full step ─────────────────────────────────────────────────
stage_before = reluctant.adoption_stage
reluctant.step()   # calls all 5 sub-methods
print("\n=== TEST 4: Full step() runs without error ===")
print(f"Reluctant stage before: {stage_before}, after: {reluctant.adoption_stage}")
print(f"Reluctant tickets this week: {reluctant.tickets_this_week}")
print(f"Reluctant frustration after: {reluctant.frustration:.3f}")
print("PASSED — step() completed without crashing")