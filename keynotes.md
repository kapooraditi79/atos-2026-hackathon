# Clustering Analysis: Causes & Solutions

## Cause 1 — Independent Feature Sampling Ignores Correlations

```python
df.loc[mask, 'digital_dexterity'] = np.random.normal(mean, std, n)
```

Each feature is sampled independently. This ignores the reality that in an employee population, features are correlated — a Reluctant User who has unusually high dexterity *for a Reluctant User* will also have unusually low ticket volume and unusually higher eNPS. The correlations between features are what make a cluster geometrically tight. Without them, your within-cluster scatter is spherically inflated — points spread equally in all directions rather than forming the tight ellipses that produce high silhouette scores.

The research-backed solution is multivariate generation with a covariance matrix:

```python
np.random.multivariate_normal(mean_vector, cov_matrix, n)
```

This is the GMM approach. One call generates all correlated features simultaneously and produces the tight, elliptical clusters that K-Means can cleanly separate.

---

## Cause 2 — Three Middle Personas Are Not Orthogonally Differentiated

**Power User**, **Pragmatic Adopter**, and **Remote-First Worker** all sit in the middle of your 5 feature dimensions. They are supposed to be different *types* of employees, not just different *levels* of the same attributes. Types require orthogonal dimensions — features where one persona is extreme while another is not.

Using only these 5 features:

```
satisfaction | productivity | resistance | training_times | digital_dexterity
```

...your three middle personas have no orthogonal distinguishing axis. They are all "moderately capable, moderately resistant." K-Means cannot separate them because they genuinely are not separated in these 5 dimensions.

Research on persona clustering consistently finds that **behavioral variables** — specifically the patterns of interaction, not just the levels — have far more discriminating power than attitudinal variables for separating user archetypes. You need features that differentiate *how* these personas work, not just *how well*.

### The Three Correct Orthogonal Differentiators

| Feature | Remote-First Worker | Power User | Pragmatic Adopter |
|---|---|---|---|
| `collab_density` | High async, low in-person | High everything | Genuinely mid |
| `app_activation_rt` | ~63% (different tool types) | 85%+ | ~60% |
| `enps_score` | +38 (autonomy-driven) | +58 | +28 (neutral) |

Adding these three to your clustering feature set gives each middle persona dimensions where they are genuinely distinct from one another.

---

## Cause 3 — K-Means Is the Wrong Algorithm for Non-Spherical Clusters

Research on behavioral user segmentation shows that density-based and model-based clustering methods consistently outperform K-Means for behavioral data, with K-Means typically achieving silhouette scores in the **0.25–0.36 range** even on well-designed datasets.

This is because K-Means assumes spherical, equally-sized clusters — which is never true for employee behavioral archetypes. **Gaussian Mixture Models (GMM)** handle non-spherical cluster shapes by modeling the covariance structure of each component, making them substantially more appropriate for data where clusters have different shapes and sizes.

> **Recommendation:** Switch from K-Means to GMM for both generation and validation.

---

## On Concordia — Where It Actually Fits

Concordia was designed to support a wide array of applications including evaluating performance of real digital services by simulating users and generating synthetic data. In digital environments it can generate synthetic user activity by constructing agent digital action logs along with agent reasoning for each action.

### What Concordia Is Not

Concordia is **not** the solution to your clustering problem. Here is why, precisely:

Concordia generates behavioral *traces* — sequences of actions, decisions, and reasoning in natural language. It does not generate tabular feature vectors. To use Concordia for your data problem, you would need to:

1. Define 1,000 agents
2. Run each one through a simulation scenario
3. Extract their behavior logs
4. Parse those logs into numerical features

That is weeks of engineering work and significant LLM API cost for what is ultimately a data generation preprocessing step.

### Where Concordia Belongs

Concordia belongs in your project — but in **Layer 3, not Layer 1**. It is the technology that gives your Mesa ABM agents genuine psychological depth: memory of past frustrations, natural language reasoning about whether to trust the chatbot, peer conversations about the rollout. That is its designed use case.

> Using Concordia to fix a clustering problem is using a piano to hammer a nail.

---

## The Actual Solution — Three Concrete Changes

### Change 1: Switch to Multivariate Normal Generation

Instead of sampling each feature independently, define a mean vector and covariance matrix per persona and sample all features simultaneously. The covariance matrix encodes the correlations between features that make clusters tight.

For each persona, define:

```python
mean_vector = [satisfaction, productivity, resistance, training, dexterity,
               collab_density, app_activation_rt, enps_score]

# A positive semi-definite matrix where:
#   diagonal entries    = variance of each feature
#   off-diagonal entries = covariance between feature pairs
cov_matrix = ...
```

#### Example Covariance Structure — Reluctant Users

| Feature Pair | Correlation |
|---|---|
| `resistance` ↔ `tickets` | +0.65 |
| `digital_dexterity` ↔ `app_activation_rt` | +0.70 |
| `dexterity` ↔ `tickets` | -0.60 |
| `satisfaction` ↔ `enps` | +0.75 |

#### Example Covariance Structure — Power Users

| Feature Pair | Correlation |
|---|---|
| `collab_density` ↔ `teams_msg` | +0.80 |
| `productivity` ↔ `app_activation_rt` | +0.65 |
| `resistance` ↔ `dexterity` | -0.30 |

The within-cluster correlation structure is what geometrically tightens each cluster into a coherent ellipsoid rather than a diffuse sphere. This is the mathematical mechanism behind high intra-cluster cohesion.

#### Building a Valid Covariance Matrix

```python
# 1. Define the correlation matrix R (values between -1 and +1)
R = ...

# 2. Define the standard deviation vector for each feature
sigma = np.array([...])

# 3. Compute the covariance matrix
cov_matrix = np.diag(sigma) @ R @ np.diag(sigma)

# 4. Verify positive semi-definiteness (all eigenvalues must be >= 0)
eigenvalues = np.linalg.eigvalsh(cov_matrix)
assert np.all(eigenvalues >= 0)
```

---

### Change 2: Add 3 Orthogonal Features to the Clustering

Expand your clustering feature set from **5 to 8**:

```
satisfaction, productivity, resistance, training_times, digital_dexterity,
collab_density, app_activation_rt, enps_score
```
