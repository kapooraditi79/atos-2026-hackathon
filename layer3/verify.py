# verify.py — put in layer3/ folder
import pandas as pd
import matplotlib.pyplot as plt

sa = pd.read_csv('layer3/outputs/output_scenario_a.csv')
sb = pd.read_csv('layer3/outputs/output_scenario_b.csv')
sc = pd.read_csv('layer3/outputs/output_scenario_c.csv')

# ── Check 1: Print key numbers ─────────────────────────────────────────────
print('=== SCENARIO A (Big-bang + Chatbot) ===')
print(f"Week 0  adoption: {sa.loc[0,  'adoption_mean']:.1%}")
print(f"Week 18 adoption: {sa.loc[17, 'adoption_mean']:.1%}")
print(f"Week 51 adoption: {sa.loc[51, 'adoption_mean']:.1%}")
print(f"Peak frustration week: {sa['frustration_mean'].idxmax()}")
print(f"Peak frustration val:  {sa['frustration_mean'].max():.3f}")
print(f"Peak ticket volume:    {sa['tickets_mean'].max():.0f}/week")
print()

print('=== SCENARIO B (Phased + Human + Training) ===')
print(f"Week 0  adoption: {sb.loc[0,  'adoption_mean']:.1%}")
print(f"Week 18 adoption: {sb.loc[17, 'adoption_mean']:.1%}")
print(f"Week 51 adoption: {sb.loc[51, 'adoption_mean']:.1%}")
print(f"Peak frustration week: {sb['frustration_mean'].idxmax()}")
print(f"Peak frustration val:  {sb['frustration_mean'].max():.3f}")
print()

print('=== CROSS-SCENARIO CHECKS ===')
print(f"B beats A at week 18? {sb.loc[17,'adoption_mean'] > sa.loc[17,'adoption_mean']}  "
      f"(B:{sb.loc[17,'adoption_mean']:.1%} vs A:{sa.loc[17,'adoption_mean']:.1%})")
print(f"A frustration peak > B frustration peak? "
      f"{sa['frustration_mean'].max() > sb['frustration_mean'].max()}")
print(f"A ticket peak > B ticket peak? "
      f"{sa['tickets_mean'].max() > sb['tickets_mean'].max()}")
print(f"Week-51 A adoption < B adoption? "
      f"{sa.loc[51,'adoption_mean'] < sb.loc[51,'adoption_mean']}")

# ── Check 2: Plot all three adoption curves ────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(sa['week'], sa['adoption_mean'],  label='A: Big-bang + Chatbot',  color='red')
plt.fill_between(sa['week'], sa['adoption_p05'], sa['adoption_p95'], alpha=0.15, color='red')
plt.plot(sb['week'], sb['adoption_mean'],  label='B: Phased + Human',      color='green')
plt.fill_between(sb['week'], sb['adoption_p05'], sb['adoption_p95'], alpha=0.15, color='green')
plt.plot(sc['week'], sc['adoption_mean'],  label='C: Pilot + Hybrid',      color='blue')
plt.fill_between(sc['week'], sc['adoption_p05'], sc['adoption_p95'], alpha=0.15, color='blue')
plt.xlabel('Week')
plt.ylabel('Adoption Rate')
plt.title('Adoption Curves — All Scenarios')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(sa['week'], sa['frustration_mean'], label='A: Chatbot',    color='red')
plt.plot(sb['week'], sb['frustration_mean'], label='B: Human',      color='green')
plt.plot(sc['week'], sc['frustration_mean'], label='C: Hybrid',     color='blue')
plt.xlabel('Week')
plt.ylabel('Avg Frustration')
plt.title('Frustration Over Time')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('layer3/outputs/verification_plot.png', dpi=150)
plt.show()
print('\nPlot saved to outputs/verification_plot.png')