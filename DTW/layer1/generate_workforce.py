"""
layer1/generate_workforce.py
─────────────────────────────
Generates a synthetic workforce DataFrame from a raw IBM HR CSV.

Maps IBM HR columns → persona distributions via scaled MVN sampling.
Returns a DataFrame with the exact schema expected by layer2.

Public entry point
──────────────────
    generate_workforce(ibm_df, n_agents=1000, seed=42) -> pd.DataFrame
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

warnings.filterwarnings("ignore")

# ── IBM HR column → internal signal map ──────────────────────────────────────
# These columns exist in WA_Fn-UseC_-HR-Employee-Attrition.csv
IBM_COLS = {
    "job_satisfaction":    "JobSatisfaction",        # 1-4
    "performance_rating":  "PerformanceRating",       # 1-4
    "job_involvement":     "JobInvolvement",           # 1-4
    "environment_sat":     "EnvironmentSatisfaction",  # 1-4
    "work_life_balance":   "WorkLifeBalance",          # 1-4
    "training_times":      "TrainingTimesLastYear",    # 0-6
    "years_at_company":    "YearsAtCompany",
    "total_working_years": "TotalWorkingYears",
    "attrition":           "Attrition",               # Yes/No
    "overtime":            "OverTime",                # Yes/No
    "job_level":           "JobLevel",                # 1-5
    "stock_option":        "StockOptionLevel",        # 0-3
}

# ── MVN parameters ────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "satisfaction_score", "productivity_baseline", "resistance_propensity",
    "training_times_yr",  "digital_dexterity",     "collab_density",
    "app_activation_rt",  "enps_score",
]

MEANS = {
    "Tech Pioneer":        [8.50, 0.88, 0.12, 5.10, 8.80, 0.80, 0.91,  72.0],
    "Power User":          [7.80, 0.82, 0.22, 4.40, 7.90, 0.86, 0.86,  60.0],
    "Pragmatic Adopter":   [5.80, 0.65, 0.45, 3.00, 5.50, 0.50, 0.60,  24.0],
    "Remote-First Worker": [6.50, 0.62, 0.55, 3.60, 5.10, 0.69, 0.54,  20.0],
    "Reluctant User":      [3.20, 0.48, 0.80, 1.40, 3.10, 0.26, 0.35, -18.0],
}

STDS = {
    "Tech Pioneer":        [0.65, 0.050, 0.055, 0.65, 0.60, 0.048, 0.048, 10.0],
    "Power User":          [0.70, 0.055, 0.060, 0.70, 0.65, 0.050, 0.050, 11.0],
    "Pragmatic Adopter":   [0.75, 0.060, 0.065, 0.75, 0.70, 0.048, 0.048, 11.0],
    "Remote-First Worker": [0.70, 0.058, 0.062, 0.70, 0.50, 0.050, 0.050,  9.0],
    "Reluctant User":      [0.72, 0.058, 0.060, 0.60, 0.62, 0.046, 0.046, 12.0],
}

# Correlation matrix (same structure as notebook)
_R = np.array([
    [1.00,  0.50, -0.52,  0.32,  0.44,  0.33,  0.36,  0.72],
    [0.50,  1.00, -0.40,  0.36,  0.55,  0.38,  0.50,  0.43],
    [-0.52,-0.40,  1.00, -0.46, -0.58, -0.36, -0.53, -0.60],
    [0.32,  0.36, -0.46,  1.00,  0.58,  0.28,  0.43,  0.33],
    [0.44,  0.55, -0.58,  0.58,  1.00,  0.40,  0.28,  0.53],
    [0.33,  0.38, -0.36,  0.28,  0.40,  1.00,  0.45,  0.38],
    [0.36,  0.50, -0.53,  0.43,  0.28,  0.45,  1.00,  0.55],
    [0.72,  0.43, -0.60,  0.33,  0.53,  0.38,  0.55,  1.00],
])

CLIPS = [
    (1.0, 10.0), (0.05, 1.0), (0.01, 0.99), (0.0, 6.0),
    (1.0, 10.0), (0.02, 1.0), (0.02, 1.0),  (-100.0, 100.0),
]

SIGNAL_PARAMS = {
    "ticket_base_lam": {
        "Tech Pioneer": 0.59, "Power User": 0.84, "Pragmatic Adopter": 1.68,
        "Remote-First Worker": 2.10, "Reluctant User": 4.20,
    },
    "crash_beta": {
        "Tech Pioneer": (1.0, 20.0), "Power User": (1.1, 16.0),
        "Pragmatic Adopter": (1.5, 11.0), "Remote-First Worker": (1.7, 9.5),
        "Reluctant User": (2.5, 7.5),
    },
    "load_time": {
        "Tech Pioneer": (1.8, 0.5), "Power User": (2.1, 0.6),
        "Pragmatic Adopter": (3.2, 0.9), "Remote-First Worker": (4.1, 1.2),
        "Reluctant User": (5.2, 1.5),
    },
    "session_min": {
        "Tech Pioneer": (55, 12), "Power User": (62, 14),
        "Pragmatic Adopter": (42, 16), "Remote-First Worker": (38, 14),
        "Reluctant User": (27, 10),
    },
    "login_lam": {
        "Tech Pioneer": 6.2, "Power User": 5.8, "Pragmatic Adopter": 4.1,
        "Remote-First Worker": 4.8, "Reluctant User": 2.8,
    },
    "failed_lam": {
        "Tech Pioneer": 0.4, "Power User": 0.6, "Pragmatic Adopter": 1.1,
        "Remote-First Worker": 1.5, "Reluctant User": 2.5,
    },
    "lms_comp": {
        "Tech Pioneer": (0.91, 0.05), "Power User": (0.84, 0.07),
        "Pragmatic Adopter": (0.64, 0.10), "Remote-First Worker": (0.71, 0.09),
        "Reluctant User": (0.36, 0.11),
    },
    "assess": {
        "Tech Pioneer": (86, 7), "Power User": (82, 9),
        "Pragmatic Adopter": (72, 11), "Remote-First Worker": (75, 10),
        "Reluctant User": (58, 13),
    },
    "complete_hrs": {
        "Tech Pioneer": (3.1, 0.9), "Power User": (3.8, 1.1),
        "Pragmatic Adopter": (5.8, 1.5), "Remote-First Worker": (5.2, 1.4),
        "Reluctant User": (8.2, 2.0),
    },
    "email": {
        "Tech Pioneer": (61, 8), "Power User": (71, 10),
        "Pragmatic Adopter": (42, 7), "Remote-First Worker": (36, 6),
        "Reluctant User": (22, 5),
    },
    "meetings": {
        "Tech Pioneer": (11, 2), "Power User": (13, 2.5),
        "Pragmatic Adopter": (8, 2), "Remote-First Worker": (7, 1.8),
        "Reluctant User": (5, 1.5),
    },
    "teams_msg": {
        "Tech Pioneer": (38, 7), "Power User": (45, 9),
        "Pragmatic Adopter": (22, 5), "Remote-First Worker": (36, 8),
        "Reluctant User": (11, 4),
    },
    "cat_probs": {
        "Tech Pioneer":        [0.10, 0.20, 0.15, 0.10, 0.10, 0.25, 0.10],
        "Power User":          [0.08, 0.25, 0.12, 0.15, 0.08, 0.22, 0.10],
        "Pragmatic Adopter":   [0.20, 0.18, 0.18, 0.14, 0.12, 0.12, 0.06],
        "Remote-First Worker": [0.15, 0.14, 0.28, 0.12, 0.18, 0.10, 0.03],
        "Reluctant User":      [0.28, 0.12, 0.16, 0.22, 0.14, 0.06, 0.02],
    },
    "churn_p": {
        "Tech Pioneer": 0.05, "Power User": 0.08, "Pragmatic Adopter": 0.16,
        "Remote-First Worker": 0.19, "Reluctant User": 0.32,
    },
}

CATEGORIES = [
    "Password Reset", "Software Install", "VPN / Network",
    "App Crash", "Hardware", "Access / Permissions", "Training Request",
]

# Rogers distribution (default, overridden when IBM data drives counts)
DEFAULT_COUNTS = {
    "Tech Pioneer":        50,
    "Power User":          150,
    "Pragmatic Adopter":   300,
    "Remote-First Worker": 200,
    "Reluctant User":      300,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nearest_psd(A: np.ndarray) -> np.ndarray:
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _build_cov(std_vec: np.ndarray, corr: np.ndarray) -> np.ndarray:
    D = np.diag(std_vec)
    cov = D @ corr @ D
    if np.linalg.eigvalsh(cov).min() < 0:
        cov = _nearest_psd(cov)
    return cov


def _fix_corr(R: np.ndarray) -> np.ndarray:
    eigs = np.linalg.eigvalsh(R)
    if eigs.min() < 0:
        R = _nearest_psd(R)
        D_inv = np.diag(1.0 / np.sqrt(np.diag(R)))
        R = D_inv @ R @ D_inv
    return R


def _ibm_to_counts(ibm_df: pd.DataFrame, n_agents: int) -> dict[str, int]:
    """
    Use IBM HR signals to scale the Rogers persona distribution.

    High JobSatisfaction + PerformanceRating → more Pioneers/Power Users.
    High Attrition/OverTime → more Reluctant Users.
    Falls back to DEFAULT_COUNTS if IBM columns are missing.
    """
    required = [IBM_COLS["job_satisfaction"], IBM_COLS["performance_rating"]]
    if not all(c in ibm_df.columns for c in required):
        # Fallback: scale default counts to n_agents
        total = sum(DEFAULT_COUNTS.values())
        counts = {p: max(5, int(round(c / total * n_agents))) for p, c in DEFAULT_COUNTS.items()}
    else:
        js  = ibm_df[IBM_COLS["job_satisfaction"]].mean()   # 1-4
        pr  = ibm_df[IBM_COLS["performance_rating"]].mean() # 1-4
        att = (ibm_df[IBM_COLS["attrition"]] == "Yes").mean() if IBM_COLS["attrition"] in ibm_df.columns else 0.16
        ot  = (ibm_df[IBM_COLS["overtime"]] == "Yes").mean()  if IBM_COLS["overtime"]  in ibm_df.columns else 0.28

        # Normalise js+pr composite (2-8 range) → 0-1
        composite = (js + pr - 2) / 6.0

        # Scale fractions based on IBM signals
        pioneer_frac   = 0.025 + composite * 0.015          # ~2.5-4%
        power_frac     = 0.100 + composite * 0.060          # ~10-16%
        reluctant_frac = 0.250 + att * 0.20 + ot * 0.10    # ~25-40%
        remote_frac    = 0.140 + ot * 0.05                  # ~14-18%
        pragmatic_frac = max(0.10, 1.0 - pioneer_frac - power_frac - reluctant_frac - remote_frac)

        fracs = {
            "Tech Pioneer":        pioneer_frac,
            "Power User":          power_frac,
            "Pragmatic Adopter":   pragmatic_frac,
            "Remote-First Worker": remote_frac,
            "Reluctant User":      reluctant_frac,
        }
        counts = {p: max(5, int(round(f * n_agents))) for p, f in fracs.items()}

    # Clamp total to n_agents
    total = sum(counts.values())
    diff  = n_agents - total
    counts["Pragmatic Adopter"] = max(5, counts["Pragmatic Adopter"] + diff)
    return counts


# ── Core generator ────────────────────────────────────────────────────────────

def generate_workforce(
    ibm_df: pd.DataFrame,
    n_agents: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic workforce DataFrame calibrated to the IBM HR CSV.

    Parameters
    ----------
    ibm_df   : raw IBM HR Employee Attrition DataFrame
    n_agents : total agents to generate (default 1000)
    seed     : numpy random seed

    Returns
    -------
    pd.DataFrame with the workforce_v2_1000.csv column schema.
    """
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    # Fix correlation matrix
    R = _fix_corr(_R.copy())

    # Build covariance per persona
    COV = {p: _build_cov(np.array(STDS[p]), R) for p in MEANS}

    counts = _ibm_to_counts(ibm_df, n_agents)

    print(f"  Layer 1 — persona distribution: {counts}")
    print(f"  Layer 1 — total agents: {sum(counts.values())}")

    rows = []
    for persona, n in counts.items():
        mu  = np.array(MEANS[persona])
        cov = COV[persona]
        sp  = SIGNAL_PARAMS

        # ── Sample 8 MVN features ──────────────────────────────────────────
        raw = np.random.multivariate_normal(mu, cov, size=n)
        for j, (lo, hi) in enumerate(CLIPS):
            raw[:, j] = np.clip(raw[:, j], lo, hi)

        sat, prod, resist = raw[:, 0], raw[:, 1], raw[:, 2]
        train   = np.round(raw[:, 3]).astype(int).clip(0, 6)
        dex_raw = raw[:, 4]
        collab  = raw[:, 5]
        activ   = raw[:, 6]
        enps    = raw[:, 7]

        # ── Conditional signals ────────────────────────────────────────────
        base_lam = sp["ticket_base_lam"][persona]
        p_mean_r = MEANS[persona][2]
        lam_ind  = np.clip(base_lam * (1.0 + 0.4 * (resist - p_mean_r) / STDS[persona][2]), 0.1, 12.0)
        tickets  = np.array([np.random.poisson(l) for l in lam_ind])

        res_hrs = np.random.exponential(
            scale=np.clip(2.0 + 3.5 * resist / (dex_raw / 10), 0.5, 48), size=n
        )
        reopen  = np.random.beta(1.0 + 2.0 * resist, 10.0 - 5.0 * resist + 1e-3, size=n).clip(0, 0.8)
        cat     = np.random.choice(CATEGORIES, size=n, p=sp["cat_probs"][persona])

        ca, cb  = sp["crash_beta"][persona]
        crash   = np.random.beta(ca, cb, n).clip(0, 0.5)
        load_t  = np.clip(np.random.normal(*sp["load_time"][persona],   n), 0.5, 20)
        session = np.clip(np.random.normal(*sp["session_min"][persona], n), 5, 150)
        dex_fb  = np.clip(enps / 100 * 9 + 1 + np.random.normal(0, 0.4, n), 1, 10)
        pulse   = np.clip(enps / 100 * 4 + 1 + np.random.normal(0, 0.2, n), 1, 5)
        logins  = np.array([np.random.poisson(sp["login_lam"][persona]) for _ in range(n)])
        failed  = np.array([np.random.poisson(sp["failed_lam"][persona]) for _ in range(n)])
        lms     = np.clip(np.random.normal(*sp["lms_comp"][persona],      n), 0, 1)
        assess  = np.clip(np.random.normal(*sp["assess"][persona],        n), 0, 100)
        comp_h  = np.clip(np.random.normal(*sp["complete_hrs"][persona],  n), 0.5, 25)
        email   = np.clip(np.random.normal(*sp["email"][persona],         n), 2, 150)
        meetings= np.clip(np.random.normal(*sp["meetings"][persona],      n), 0, 30)
        teams   = np.clip(np.random.normal(*sp["teams_msg"][persona],     n), 0, 120)

        collab_comp = np.clip(
            collab * 0.60 + email / 150 * 0.15 + meetings / 30 * 0.15 + teams / 120 * 0.10, 0, 1
        )
        supp_dep = np.clip(tickets / 10.0, 0, 1)
        frustrate = np.clip(res_hrs / 72 * 0.55 + reopen * 0.45, 0, 1)
        friction  = np.clip(frustrate * 0.50 + crash * 0.30 + load_t / 20 * 0.20, 0, 1)

        time_norm = np.clip((20.0 - comp_h) / (20.0 - 0.5), 0, 1)
        fail_norm = np.clip((8.0 - failed.astype(float)) / 8.0, 0, 1)
        meet_norm = np.clip(meetings / 25.0, 0, 1)

        dex_final = np.clip(
            dex_raw  * 0.52 + time_norm * 10 * 0.22
            + fail_norm * 10 * 0.14 + meet_norm * 10 * 0.08
            + (1 - crash) * 10 * 0.04,
            1.0, 10.0,
        )
        churn = (np.random.random(n) < sp["churn_p"][persona]).astype(int)

        for i in range(n):
            rows.append({
                "employee_id":           str(fake.uuid4()),
                "persona":               persona,
                "satisfaction_score":    round(float(sat[i]),     3),
                "productivity_baseline": round(float(prod[i]),    3),
                "resistance_propensity": round(float(resist[i]),  3),
                "training_times_yr":     int(train[i]),
                "digital_dexterity":     round(float(dex_final[i]), 3),
                "collab_density":        round(float(collab_comp[i]), 3),
                "app_activation_rt":     round(float(activ[i]),   3),
                "enps_score":            round(float(enps[i]),    1),
                "email_vol_daily":       round(float(email[i]),   2),
                "meetings_per_week":     round(float(meetings[i]),2),
                "teams_msg_daily":       round(float(teams[i]),   2),
                "tickets_per_month":     int(tickets[i]),
                "ticket_category":       cat[i],
                "resolution_hrs":        round(float(res_hrs[i]), 2),
                "reopened_rate":         round(float(reopen[i]),  3),
                "support_dependency":    round(float(supp_dep[i]),3),
                "frustration_level":     round(float(frustrate[i]),3),
                "app_crash_rate":        round(float(crash[i]),   3),
                "avg_load_time_sec":     round(float(load_t[i]),  2),
                "session_duration_min":  round(float(session[i]), 1),
                "friction_level":        round(float(friction[i]),3),
                "dex_feedback":          round(float(dex_fb[i]),  2),
                "pulse_sat":             round(float(pulse[i]),   2),
                "logins_per_day":        int(logins[i]),
                "failed_logins_wk":      int(failed[i]),
                "lms_completion":        round(float(lms[i]),     3),
                "assessment_score":      round(float(assess[i]),  1),
                "time_to_complete_hr":   round(float(comp_h[i]),  2),
                "churn_risk_flag":       int(churn[i]),
                "is_amplifier":          0,
            })

    df = pd.DataFrame(rows)
    print(f"  Layer 1 — generated: {df.shape[0]} rows × {df.shape[1]} cols")
    return df