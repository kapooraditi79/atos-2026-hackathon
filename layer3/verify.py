# verify.py — put in layer3/ folder
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_checks():
    # Load all scenarios
    sa = pd.read_csv('layer3/outputs/output_scenario_a.csv')
    sb = pd.read_csv('layer3/outputs/output_scenario_b.csv')
    sc = pd.read_csv('layer3/outputs/output_scenario_c.csv')

    print('\n=== SCENARIO A (Big-bang + Chatbot) ===')
    print(f"Week 0  adoption: {sa.loc[0,  'adoption_mean']:.1%}")
    print(f"Week 18 adoption: {sa.loc[17, 'adoption_mean']:.1%}")
    print(f"Week 51 adoption: {sa.loc[51, 'adoption_mean']:.1%}")
    print(f"Peak frustration:  {sa['frustration_mean'].max():.3f}")
    print(f"Peak tickets:      {sa['tickets_mean'].max():.0f}/week")

    print('\n=== SCENARIO B (Phased + Human + Training) ===')
    print(f"Week 0  adoption: {sb.loc[0,  'adoption_mean']:.1%}")
    print(f"Week 18 adoption: {sb.loc[17, 'adoption_mean']:.1%}")
    print(f"Week 51 adoption: {sb.loc[51, 'adoption_mean']:.1%}")
    print(f"Peak frustration:  {sb['frustration_mean'].max():.3f}")
    print(f"Peak tickets:      {sb['tickets_mean'].max():.0f}/week")

    print('\n=== SCENARIO C (Pilot + Strong Management + Hybrid) ===')
    print(f"Week 0  adoption: {sc.loc[0,  'adoption_mean']:.1%}")
    print(f"Week 18 adoption: {sc.loc[17, 'adoption_mean']:.1%}")
    print(f"Week 51 adoption: {sc.loc[51, 'adoption_mean']:.1%}")
    print(f"Peak frustration:  {sc['frustration_mean'].max():.3f}")
    print(f"Peak tickets:      {sc['tickets_mean'].max():.0f}/week")

    print('\n=== CROSS-SCENARIO CHECKS ===')
    print(f"B beats A at week 18? {sb.loc[17,'adoption_mean'] > sa.loc[17,'adoption_mean']} "
          f"(B:{sb.loc[17,'adoption_mean']:.1%} vs A:{sa.loc[17,'adoption_mean']:.1%})")
    print(f"B beats C at week 18? {sb.loc[17,'adoption_mean'] > sc.loc[17,'adoption_mean']} "
          f"(B:{sb.loc[17,'adoption_mean']:.1%} vs C:{sc.loc[17,'adoption_mean']:.1%})")
    print(f"A frustration peak > B frustration peak? {sa['frustration_mean'].max() > sb['frustration_mean'].max()}")
    print(f"C ticket volume < B ticket volume? {sc['tickets_mean'].mean() < sb['tickets_mean'].mean()}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 6))

    # Adoption subplot
    plt.subplot(1, 2, 1)
    plt.plot(sa['week'], sa['adoption_mean'], label='A: Chatbot', color='red', lw=2)
    plt.plot(sb['week'], sb['adoption_mean'], label='B: Human',   color='green', lw=2)
    plt.plot(sc['week'], sc['adoption_mean'], label='C: Hybrid',  color='blue', lw=2)
    plt.xlabel('Week')
    plt.ylabel('Adoption Rate')
    plt.title('Adoption Curves — Tri-Scenario Separation')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Frustration subplot
    plt.subplot(1, 2, 2)
    plt.plot(sa['week'], sa['frustration_mean'], label='A: Chatbot', color='red')
    plt.plot(sb['week'], sb['frustration_mean'], label='B: Human',   color='green')
    plt.plot(sc['week'], sc['frustration_mean'], label='C: Hybrid',  color='blue')
    plt.xlabel('Week')
    plt.ylabel('Avg Frustration')
    plt.title('Frustration Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('layer3/outputs/verification_plot.png', dpi=150)
    print('\nPlot saved to layer3/outputs/verification_plot.png')

if __name__ == "__main__":
    run_checks()