"""
layer2/validate_clusters.py
────────────────────────────
Validates synthetic workforce data quality:
  - GMM / K-Means clustering
  - Adjusted Rand Index (ARI)
  - Benchmark monotonicity
  - Correlation structure

Returns True/False; never writes files or reads CSVs directly.
All I/O is the caller's responsibility.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Feature set used for clustering ───────────────────────────────────────────
CLUSTER_FEATURES = [
    "satisfaction_score",
    "productivity_baseline",
    "resistance_propensity",
    "training_times_yr",
    "digital_dexterity",
    "collab_density",
    "app_activation_rt",
    "enps_score",
]

N_CLUSTERS = 5

# ── Expected persona-level monotonicity (from design spec) ────────────────────
# Tech Pioneer ≻ Power User ≻ Pragmatic Adopter ≻ Remote-First Worker ≻ Reluctant User
BENCHMARK_ORDER = [
    "Tech Pioneer",
    "Power User",
    "Pragmatic Adopter",
    "Remote-First Worker",
    "Reluctant User",
]

# Metrics that should decrease along BENCHMARK_ORDER
DECREASING_METRICS = ["digital_dexterity", "app_activation_rt", "productivity_baseline"]
# Metrics that should increase along BENCHMARK_ORDER
INCREASING_METRICS = ["resistance_propensity"]


# ── Individual checks ──────────────────────────────────────────────────────────

def _check_gmm_ari(df: pd.DataFrame) -> tuple[bool, float]:
    """Fit 5-component GMM and compare to persona labels via ARI."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=3)
    pred_labels = gmm.fit_predict(X)

    # Map persona strings → integers for ARI
    persona_codes = pd.Categorical(df["persona"]).codes
    ari = adjusted_rand_score(persona_codes, pred_labels)

    passed = ari > 0.30
    print(f"  {'✓' if passed else '✗'} GMM ARI = {ari:.4f}  (threshold > 0.30)")
    return passed, ari


def _check_kmeans_ari(df: pd.DataFrame) -> tuple[bool, float]:
    """Fit 5-centroid K-Means and compare to persona labels via ARI."""
    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    pred_labels = km.fit_predict(X)

    persona_codes = pd.Categorical(df["persona"]).codes
    ari = adjusted_rand_score(persona_codes, pred_labels)

    passed = ari > 0.25
    print(f"  {'✓' if passed else '✗'} K-Means ARI = {ari:.4f}  (threshold > 0.25)")
    return passed, ari


def _check_monotonicity(df: pd.DataFrame) -> bool:
    """Verify persona mean metrics respect the design-spec ordering."""
    means = df.groupby("persona")[DECREASING_METRICS + INCREASING_METRICS].mean()

    all_ok = True
    for metric in DECREASING_METRICS:
        vals = [means.loc[p, metric] for p in BENCHMARK_ORDER if p in means.index]
        ok = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        if not ok:
            print(f"  ✗ Monotonicity FAIL — {metric} should decrease along persona order")
            all_ok = False

    for metric in INCREASING_METRICS:
        vals = [means.loc[p, metric] for p in BENCHMARK_ORDER if p in means.index]
        ok = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        if not ok:
            print(f"  ✗ Monotonicity FAIL — {metric} should increase along persona order")
            all_ok = False

    if all_ok:
        print("  ✓ Monotonicity: all benchmark metrics respect persona ordering")
    return all_ok


def _check_correlations(df: pd.DataFrame) -> bool:
    """
    Verify expected sign relationships:
      digital_dexterity  ↑ → app_activation_rt ↑  (positive)
      resistance_propensity ↑ → app_activation_rt ↓  (negative)
    """
    ok = True

    corr_dex_act = df["digital_dexterity"].corr(df["app_activation_rt"])
    if corr_dex_act < 0.10:
        print(f"  ✗ Correlation FAIL — dexterity vs activation = {corr_dex_act:.3f} (expected > 0.10)")
        ok = False
    else:
        print(f"  ✓ dexterity vs activation = {corr_dex_act:.3f}")

    corr_res_act = df["resistance_propensity"].corr(df["app_activation_rt"])
    if corr_res_act > -0.05:
        print(f"  ✗ Correlation FAIL — resistance vs activation = {corr_res_act:.3f} (expected < -0.05)")
        ok = False
    else:
        print(f"  ✓ resistance vs activation = {corr_res_act:.3f}")

    return ok


def _check_persona_balance(df: pd.DataFrame) -> bool:
    """Warn if any persona is extremely over- or under-represented."""
    counts = df["persona"].value_counts(normalize=True)
    ok = True
    for persona, frac in counts.items():
        if frac < 0.05 or frac > 0.50:
            print(f"  ✗ Balance WARN — '{persona}' represents {frac:.1%} of workforce")
            ok = False
    if ok:
        print(f"  ✓ Persona balance: {counts.to_dict()}")
    return ok


def _assign_gmm_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit GMM and write gmm_cluster column back to the DataFrame.
    Returns updated copy.
    """
    df = df.copy()
    X  = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=3)
    df["gmm_cluster"] = gmm.fit_predict(X)
    print(f"  ✓ GMM clusters assigned  (n={N_CLUSTERS})")
    return df


# ── Public entry point ─────────────────────────────────────────────────────────

def run_all_validations(df: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    """
    Run all validation checks on the workforce DataFrame.

    Returns
    -------
    (passed: bool, enriched_df: pd.DataFrame)
        enriched_df has the gmm_cluster column attached.
        If passed is False, caller should abort the pipeline.
    """
    print("=" * 60)
    print("LAYER 2 — CLUSTER VALIDATION")
    print("=" * 60)

    hard_fail = False

    # Soft checks (warn but don't abort)
    _check_gmm_ari(df)
    _check_kmeans_ari(df)
    _check_persona_balance(df)

    # Hard checks (abort on failure)
    mono_ok  = _check_monotonicity(df)
    corr_ok  = _check_correlations(df)

    if not mono_ok or not corr_ok:
        hard_fail = True

    if hard_fail:
        print("\n  ✗ HARD CHECKS FAILED — fix data generation before proceeding.")
        return False, df

    # Assign GMM clusters to DataFrame
    df = _assign_gmm_clusters(df)

    print(f"\n{'═' * 60}")
    print("LAYER 2 VALIDATION COMPLETE ✓")
    print(f"{'═' * 60}")
    return True, df