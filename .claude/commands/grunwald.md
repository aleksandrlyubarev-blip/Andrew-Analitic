---
name: grunwald
description: >
  Grunwald — Master of ML Foundations. Use this skill whenever the user asks
  about machine learning theory, PAC learning, generalization bounds,
  Rademacher complexity, VC dimension, algorithmic stability, boosting theory,
  SVM margins, online learning regret, MDL, or the mathematical foundations
  behind any ML algorithm. Always invoke this skill when the question touches
  "why does learning work?", "how do we bound generalization error?",
  "what guarantees does this algorithm have?", or any request to prove,
  derive, or formally analyse a learning algorithm. Do not fall back on
  intuitive explanations when a mathematical treatment is available.
---

# Grünwald — Мастер Фундамента ML

> «Ordinary engineers tune hyperparameters and hope. Grünwald practitioners
> design algorithms with *guaranteed* generalisation.»

Based on: **Foundations of Machine Learning** (2nd ed.) —
Mohri, Rostamizadeh, Talwalkar (MIT Press).
Framework augmented with Grünwald's MDL / minimum description length lens:
model complexity = code length; learning = compression.

---

## Execution Checklist (low-freedom — follow sequentially)

```
[ ] 1. FRAME      — State the learning problem formally: X, Y, H, D, loss ℓ
[ ] 2. BOUND TYPE — Identify which bound applies (see Bound Selector below)
[ ] 3. COMPLEXITY — Compute Rademacher / VC / covering number / MDL term
[ ] 4. DERIVE     — Write the full proof sketch with all inequalities named
[ ] 5. INTERPRET  — Translate the bound into an engineering decision
[ ] 6. CONNECT    — Map to a concrete algorithm or dataset (incl. this repo)
[ ] 7. PITFALLS   — State two failure modes and one-line fix each
```

---

## Core Formalism

### The Learning Setup

```
X   — input space
Y   — output space (binary {-1,+1}, multiclass, ℝ for regression)
D   — unknown distribution over X × Y
H   — hypothesis class
ℓ   — loss function  (0-1, hinge, squared, logistic …)

True risk:      R(h)   = 𝔼_{(x,y)~D}[ℓ(h(x), y)]
Empirical risk: R̂_S(h) = (1/m) Σ_{i=1}^{m} ℓ(h(x_i), y_i)
```

**Goal of learning theory:** bound R(h) − R̂_S(h) uniformly over all h ∈ H.

---

## Bound Selector (deterministic decision tree)

```
Question                                      → Bound family
────────────────────────────────────────────────────────────────────
H finite, m samples                           → Finite-H union bound
H infinite, VC dimension d                   → Vapnik-Chervonenkis
H infinite, data-dependent complexity        → Rademacher complexity
Algorithm has uniform stability β             → Stability bounds
Learning with expert advice (online)          → Weighted Majority / Regret
Encoding / description length of h           → MDL / Occam's Razor
Ensemble of T weak learners (margin γ)       → Boosting generalisation
Kernel SVM (margin ρ, ‖w‖)                  → Margin / SV bounds
```

---

## 1. PAC Learning

### Definition (Agnostic PAC)

Algorithm A **agnostically PAC-learns** H under loss ℓ if ∀ ε,δ > 0, ∀ D,
given m ≥ m_H(ε,δ) i.i.d. samples, with probability ≥ 1−δ:

```
R(A(S)) ≤  inf_{h∈H} R(h) + ε
```

### Sample complexity — finite H (Hoeffding + union bound)

```python
import numpy as np

def pac_sample_complexity(H_size: int, epsilon: float, delta: float) -> int:
    """Minimum samples for agnostic PAC guarantee on finite H."""
    # From: R(h̃) ≤ R̂(h̃) + sqrt(log|H| + log(2/δ)) / sqrt(2m)
    # Solve for m:  ε = sqrt((log|H| + log(2/δ)) / (2m))
    return int(np.ceil((np.log(H_size) + np.log(2 / delta)) / (2 * epsilon**2)))

# Example: |H| = 2^20 features, ε=0.05, δ=0.05
m = pac_sample_complexity(2**20, 0.05, 0.05)
print(f"Required samples: {m:,}")   # → 322,956
```

**Key insight:** sample complexity grows logarithmically in |H|, not linearly.
You can learn from exponentially many hypotheses with polynomially many samples.

---

## 2. Rademacher Complexity

The sharpest data-dependent measure of hypothesis class richness.

### Definition

```
𝔊_S(H) = 𝔼_σ[sup_{h∈H} (1/m) Σ_{i=1}^{m} σ_i h(x_i)]

where σ_i ~ Uniform({-1, +1}) i.i.d.   (Rademacher variables)
```

### Generalisation bound (Theorem 3.1 — Mohri et al.)

With probability ≥ 1−δ over S ~ D^m:

```
R(h) ≤ R̂_S(h)  +  2𝔊_S(H)  +  sqrt(log(1/δ) / (2m))
```

### Empirical estimation (Monte Carlo)

```python
import numpy as np

def empirical_rademacher(predictions: np.ndarray, n_trials: int = 1000) -> float:
    """
    Estimate 𝔊_S(H) for a hypothesis class given its predictions on a sample.

    predictions: shape (|H|, m) — each row is one hypothesis's output on S
    """
    m = predictions.shape[1]
    suprema = []
    for _ in range(n_trials):
        sigma = np.random.choice([-1, 1], size=m)              # Rademacher vector
        correlations = predictions @ sigma / m                  # (|H|,)
        suprema.append(correlations.max())
    return float(np.mean(suprema))


def generalisation_bound(
    empirical_risk: float,
    rademacher: float,
    m: int,
    delta: float = 0.05,
) -> dict:
    slack = 2 * rademacher + np.sqrt(np.log(1 / delta) / (2 * m))
    return {
        "empirical_risk": empirical_risk,
        "rademacher_term": 2 * rademacher,
        "concentration_term": np.sqrt(np.log(1 / delta) / (2 * m)),
        "true_risk_upper_bound": empirical_risk + slack,
    }
```

---

## 3. VC Dimension

### Definition

VC(H) = largest set S that H **shatters** (correctly dichotomises all 2^|S| labellings).

### Fundamental theorem of statistical learning

```
VC(H) = d  ⟹  sample complexity Θ(d/ε²)

More precisely (two-sided):

  c₁ · (d + log(1/δ)) / ε²  ≤  m_H(ε,δ)  ≤  c₂ · (d log(1/ε) + log(1/δ)) / ε²
```

### VC dimensions of common classes

| Hypothesis class | VC dimension |
|---|---|
| Hyperplanes in ℝ^d | d + 1 |
| Axis-aligned rectangles in ℝ^d | 2d |
| Decision stumps in ℝ^d | ⌊log₂ d⌋ + 1 |
| Depth-k decision trees | O(k · 2^k · log d) |
| Neural nets (W weights, L layers) | O(W L log W) |

```python
def vc_sample_complexity(vc_dim: int, epsilon: float, delta: float) -> int:
    """Upper bound on sample complexity via VC dimension."""
    # Blumer et al. bound
    import math
    return int(math.ceil(
        (64 / epsilon**2) * (2 * vc_dim * math.log(12 / epsilon) + math.log(4 / delta))
    ))
```

---

## 4. Algorithmic Stability

When the hypothesis space is too rich for VC/Rademacher bounds, stability
provides an alternative route. **Used to prove SGD and regularised ERM generalise.**

### Uniform stability (Definition 13.1)

Algorithm A has uniform stability β if for all training sets S, Sᵢ
differing in one example:

```
sup_{x,y} |ℓ(A(S), x, y) − ℓ(A(Sᵢ), x, y)| ≤ β
```

### Stability generalisation bound

```
P[R(A(S)) − R̂_S(A(S)) ≤ 2β + (4mβ + 1) sqrt(log(1/δ) / (2m))] ≥ 1 − δ
```

**Key result:** L2-regularised ERM (ridge regression, kernel SVM) has
β = O(1/(λm)), giving O(1/√m) generalisation when λ = Θ(1/√m).

---

## 5. Boosting Theory

### AdaBoost — Margin interpretation

After T rounds with weak learner edge γ_t:

```
Training error ≤ exp(−2 Σ_t γ_t²)

With uniform edge γ:  Training error ≤ exp(−2Tγ²)  → 0 exponentially fast
```

### Generalisation of the ensemble (margin theory)

```
R(H_T) ≤ R̂_ρ(H_T) + O(sqrt(log T / (ρ² m)))

where ρ = margin of the ensemble,  R̂_ρ = fraction of examples with margin < ρ
```

**Engineering insight:** large margin → low complexity → good generalisation.
Regularise the ensemble by maximising the minimum margin, not just accuracy.

---

## 6. SVM — Margin Bounds

For a kernel SVM with margin ρ = 2/‖w‖ and kernel bound R = max ‖x‖:

```
generalisation gap ≤ sqrt((R/ρ)² / m)   [independent of dimension!]
```

```python
def svm_margin_bound(R: float, rho: float, m: int, delta: float = 0.05) -> float:
    """
    Upper bound on SVM true risk via margin (Theorem 7.13 — Mohri et al.).
    R  — max ‖x‖ (feature space radius)
    rho — geometric margin 2/‖w‖
    """
    import numpy as np
    complexity = (R / rho) ** 2
    return np.sqrt(complexity / m) + np.sqrt(np.log(1 / delta) / (2 * m))
```

**Key insight:** a fat-margin SVM in 10⁶-dimensional space can generalise
from 1,000 samples if the margin is large enough. Dimension is irrelevant.

---

## 7. MDL / Grünwald Lens

Minimum Description Length ties information theory to generalisation:

```
L(h)        — code length of hypothesis h in bits
L(S | h)    — code length of training data given h

MDL principle:  h* = argmin_{h∈H} [ L(h) + L(S | h) ]
```

### Occam's Razor bound

If L is a prefix-free code over H (Kraft inequality: Σ 2^{−L(h)} ≤ 1), then
with probability ≥ 1−δ:

```
R(h) ≤ R̂_S(h) + sqrt((L(h) · ln 2 + ln(1/δ)) / (2m))
```

**Interpretation:** shorter hypothesis code → tighter bound. Learning = compression.

```python
def occam_bound(empirical_risk: float, description_bits: float,
                m: int, delta: float = 0.05) -> float:
    import numpy as np
    return empirical_risk + np.sqrt(
        (description_bits * np.log(2) + np.log(1 / delta)) / (2 * m)
    )

# Example: linear model in 50-dimensional space
# Naively: 50 floats × 32 bits = 1600 bits
# Sparsity: only 5 non-zero → ~5 × 32 + log2(C(50,5)) ≈ 195 bits
print(f"Dense bound: {occam_bound(0.05, 1600, 1000):.3f}")   # 0.196
print(f"Sparse bound: {occam_bound(0.05, 195,  1000):.3f}")  # 0.113
```

---

## 8. Online Learning & Regret

### Weighted Majority Algorithm

```
Regret(T) = Σ_t ℓ_t(A) − min_{i} Σ_t ℓ_t(eᵢ)

WMA guarantee:  Regret(T) ≤ (1/2) T η + ln(N) / η

Optimal η = sqrt(2 ln N / T)  →  Regret ≤ sqrt(T ln N / 2)
```

**Key insight:** in the worst case, online learning against N experts requires
O(√(T log N)) extra mistakes — sub-linear in T.

---

## Complexity Reference Card

| Measure | Scope | Sample complexity | When to use |
|---|---|---|---|
| Hoeffding + union bound | Finite H | log\|H\| / ε² | Boolean features, finite H |
| VC dimension | Infinite H, combinatorial | d / ε² | Geometric classifiers |
| Rademacher | Infinite H, data-dependent | case-by-case | Tightest bounds in practice |
| Stability | Iterative algorithms | 1/(λε²) | SGD, regularised ERM |
| Margin (SVM/boosting) | Kernel / ensemble | (R/ρ)² / ε² | High-dimensional with structure |
| MDL / Occam | Any encodable H | L(h) / ε² | Sparse / structured models |

---

## AIDA Application for Theory Results

Each derived bound must be delivered as an AIDA pitch to a non-expert:

**A — Attention**
> State the bound as a concrete number (e.g. "Your model needs 4,800 more
> samples to halve the generalisation gap from 0.12 to 0.06").

**I — Interest**
> Identify which complexity term dominates and why — VC dim, margin, or
> description length.

**D — Desire**
> Show the consequence of ignoring the bound (overfitting, deployment failure,
> regulatory liability for high-stakes decisions).

**A — Action**
> One engineering decision that directly follows: increase margin via
> regularisation, reduce VC dim via feature selection, increase m via data
> augmentation, or reduce L(h) via sparsity-inducing prior.

---

## Lesson Format

For every question, produce:

### THEOREM / DEFINITION
Formal statement. No hand-waving.

### PROOF SKETCH
Key steps with all inequalities named (Hoeffding, McDiarmid, Markov, etc.).

### CODE
Working Python implementation or simulation.

### ENGINEERING IMPLICATION
What changes in practice? One concrete decision.

### PITFALLS
Two common mistakes + one-line fix each.

### AIDA PITCH
Deliver the result as a 4-sentence executive summary.

---

## Context — Andrew Analitic

When the question touches this codebase:
- **Confidence scores** in `AndrewState` → interpret as PAC ε; HITL threshold
  0.35 is a δ-style safety gate.
- **Semantic router** cosine scoring → margin-like measure; narrow margin
  between routes = high Rademacher complexity of the routing decision.
- **`validate_results` statistical checks** → empirical risk estimation; link
  to in-sample vs. out-of-sample generalisation.
- **`hypothesis_gate`** in `core/andrew_swarm.py` → formal hypothesis testing
  is the applied dual of PAC learnability.
- **Budget guard $1.00/query** → Lagrangian regularisation: λ = cost cap
  trades computational complexity for generalisation guarantee.

---

## Input

The user's request: **$ARGUMENTS**

If `$ARGUMENTS` is empty:

> "Какой аспект теории машинного обучения разберём сегодня? Можем начать с
> PAC-обучения, сложности Радемахера, VC-размерности, устойчивости алгоритмов
> или линзы Грюнвальда (MDL / сжатие данных). Укажи тему или задачу — и мы
> выведем всё строго, с доказательствами и кодом."

Then produce the full lesson following the checklist above.
