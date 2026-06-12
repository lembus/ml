# Hyperparameter Tuning — A Comprehensive Learning & Interview Guide

---

## 1. Motivation & Intuition

### 1.1 What is the problem?

Every machine-learning model has two kinds of "knobs":

1. **Parameters**, which are *learned from data* by the optimizer (e.g., the weights of a neural network, the split thresholds of a tree).
2. **Hyperparameters**, which are *set before training* and govern the learning process itself (e.g., learning rate, batch size, regularization strength, number of layers, depth of trees, kernel choice in an SVM).

The values of these hyperparameters are **not** discovered by gradient descent because they often live outside the model's loss surface or because they control discrete structural choices. Yet they massively affect final performance. A learning rate of 1e-2 may converge beautifully while 1e-1 diverges and 1e-4 stalls; a depth-5 tree may overfit while a depth-3 tree generalizes; an L2 coefficient of 1e-4 may be optimal while 1e-2 underfits.

**Hyperparameter tuning** is the meta-problem of finding hyperparameter values that produce the best-generalizing model under some computational budget.

### 1.2 A concrete example

Suppose you are training a gradient-boosted tree on a tabular dataset using XGBoost. Some of the knobs include:

- `learning_rate` ∈ [1e-3, 1.0] (continuous, log-scale)
- `max_depth` ∈ {3, 4, …, 10} (integer)
- `subsample` ∈ [0.5, 1.0]
- `colsample_bytree` ∈ [0.5, 1.0]
- `min_child_weight` ∈ {1, 3, 5, 7, 9}
- `reg_lambda` ∈ [1e-4, 1e2] (log-scale)
- `n_estimators` ∈ {100, …, 5000}

Even with a modest 5–10 candidate values per knob, the joint space contains **millions** of configurations. Each evaluation requires fitting the full model and validating it (often with cross-validation), which can take minutes to hours. Brute-force enumeration is impossible. We need a smarter search strategy.

### 1.3 Why this matters in real ML systems

- **Production models**: A 1% absolute accuracy lift from better hyperparameters can be worth millions in revenue or thousands of GPU hours of avoided retraining.
- **Reproducibility & rigor**: Published results often need careful tuning of baselines so that comparisons are fair. Failure to tune a baseline well is a common source of irreproducible papers.
- **AutoML**: Hyperparameter optimization (HPO) is the engine inside AutoML systems like Auto-sklearn, Auto-PyTorch, and Vertex AI's HPO service.
- **Compute economics**: GPUs are expensive. The difference between random search and a good multi-fidelity method can mean 10–100× less compute for the same final score.
- **Neural architecture search (NAS)** is a generalization of HPO where the "hyperparameters" are architectural choices.

### 1.4 The intuition behind every method we will cover

Imagine you are tuning two hyperparameters and the validation loss surface looks like a smooth bowl with a narrow minimum. Different strategies attack this bowl differently:

- **Manual search**: A human eyeballs values based on intuition. Effective for experts but unscalable.
- **Grid search**: Place a regular lattice over the space and try every point. Wasteful if some dimensions don't matter.
- **Random search**: Sample uniformly at random. Surprisingly competitive because it avoids wasting samples on irrelevant dimensions.
- **Bayesian optimization**: Build a probabilistic model of the loss surface, then deliberately probe where the model thinks the optimum is or where uncertainty is highest.
- **Hyperband**: Run *many* configurations with tiny budgets, kill the bad ones early, and reinvest the budget in promising survivors.
- **BOHB**: Combine Bayesian optimization (smart sampling) with Hyperband (smart budget allocation).
- **Population-based methods**: Maintain a pool of models and let good ones "reproduce" (copy weights, perturb hyperparameters), like evolution.

The rest of this document fleshes out each of these methods, their math, when they shine, and how to use them in practice.

---

## 2. Conceptual Foundations

### 2.1 Key definitions

**Hyperparameter** (λ): a configuration variable that controls model architecture or the training procedure but is not learned by the optimizer.

**Configuration**: a complete assignment of values to all hyperparameters.

**Configuration space** (Λ): the set of all valid configurations. It can be:
- *Continuous*: e.g., learning rate ∈ [1e-5, 1].
- *Discrete (ordinal)*: e.g., number of hidden units ∈ {32, 64, 128, 256}.
- *Categorical*: e.g., optimizer ∈ {SGD, Adam, RMSProp}.
- *Conditional*: e.g., "momentum" only matters if optimizer = SGD; "β2" only matters if optimizer = Adam.

The conditional structure forms a tree-shaped or graph-shaped configuration space, which complicates many methods.

**Objective function** *f*(λ): a measure of model quality at configuration λ — typically validation loss, validation accuracy, or a domain-specific metric — averaged over a validation set or cross-validation folds. *f* is:
- *Black-box*: we have no analytic gradient with respect to λ.
- *Noisy*: stochastic mini-batches, random initializations, and CV variance produce noise.
- *Expensive*: each evaluation requires training a model.
- *Mixed-type*: λ has continuous, discrete, and categorical coordinates.

**Budget** (B): total amount of compute available, typically counted in:
- Number of full configurations evaluated.
- Total wall-clock time.
- Total epochs / training iterations across all configurations (relevant for multi-fidelity methods).

**Fidelity** (or *budget per configuration*): a parameter `r` that controls how thoroughly we evaluate a single configuration. Examples:
- Number of training epochs.
- Fraction of training data used.
- Resolution of input images.
- Number of MCMC samples (for probabilistic models).

**Multi-fidelity optimization**: methods that evaluate many configurations cheaply (low fidelity) to filter out bad ones, then evaluate few configurations expensively (high fidelity).

### 2.2 The optimization problem

Formally:

λ\* = argmin_{λ ∈ Λ} 𝔼[f(λ)]

where the expectation is over the noise sources (data shuffling, weight initialization, CV folds). We want to find λ\* using as few evaluations of *f* as possible.

### 2.3 Method-by-method conceptual overview

#### 2.3.1 Manual search

The user picks configurations by intuition, runs them, inspects results, and iterates. Strengths: leverages domain expertise, can find creative solutions, handles conditional structure naturally. Weaknesses: not reproducible, doesn't scale beyond a few dimensions, biased by human anchoring.

#### 2.3.2 Grid search

Discretize each dimension into a small set of values. Evaluate the Cartesian product.
- Two hyperparameters with 10 values each → 100 configurations.
- Five hyperparameters with 10 values each → 100 000 configurations.

Suffers from the **curse of dimensionality**. Worse, if only 2 of 5 hyperparameters truly matter, grid search wastes 80% of evaluations on irrelevant variation.

#### 2.3.3 Random search

Sample λ uniformly at random from Λ, evaluate, repeat. Bergstra and Bengio (2012) showed that for many practical problems the *effective dimensionality* (the number of hyperparameters that strongly affect performance) is small. Random search devotes an effective evaluation to every dimension regardless of which dimensions matter, while grid search collapses many evaluations onto useless slices.

Key result (informal): if only `d_eff` of `d` total hyperparameters matter, random search with `n` samples gives roughly `n` distinct values of each important dimension, whereas grid search gives only `n^{1/d}`.

#### 2.3.4 Bayesian optimization (BO)

Build a probabilistic *surrogate model* of f(λ) — typically a **Gaussian process (GP)** — using the configurations evaluated so far. The surrogate provides for any candidate λ:
- A predicted mean μ(λ).
- A predicted uncertainty σ(λ).

Define an **acquisition function** *a*(λ) that scores how desirable it is to evaluate λ next, balancing:
- *Exploitation*: low μ(λ) (a configuration the surrogate believes is good).
- *Exploration*: high σ(λ) (a configuration the surrogate is uncertain about, and could turn out to be excellent).

Pick λ_{next} = argmax *a*(λ), evaluate it, add it to the data set, refit the surrogate, repeat.

Common acquisition functions:
- **Expected Improvement (EI)**: E[max(f\* − f(λ), 0)] where f\* is the best value seen so far.
- **Upper / Lower Confidence Bound (UCB / LCB)**: μ(λ) ± κ σ(λ).
- **Probability of Improvement (PI)**: P(f(λ) < f\*).
- **Thompson Sampling**: draw a function from the posterior and minimize it.
- **Entropy Search / Predictive Entropy Search**: choose λ to maximize information gain about the location of the optimum.

BO is sample-efficient (often the best choice when each evaluation is very expensive) but scales poorly with dimensionality and number of evaluations (vanilla GP is O(n³)).

**Tree-structured Parzen Estimator (TPE)** is an alternative surrogate that handles conditional spaces better than vanilla GPs and is the default in Optuna and Hyperopt.

#### 2.3.5 Successive Halving (SH)

Idea: instead of training each candidate to completion, train them all briefly, keep the top fraction, train survivors longer, keep top fraction again, etc.

Algorithm with `n` configurations and budget per configuration `r`:
1. Evaluate all `n` configurations with budget `r`.
2. Keep the top 1/η (typically η = 3 or 4).
3. Multiply each survivor's budget by η.
4. Repeat until one configuration remains.

Risk: if performance at low budgets is a poor predictor of final performance (e.g., a configuration that needs warmup), SH kills the eventual winner early.

#### 2.3.6 Hyperband

Generalizes SH by hedging across different choices of `n` (and consequently the implicit "early-stopping aggressiveness"). It runs SH multiple times with different (n, r) pairs called *brackets*. The first bracket has many configurations and tiny budgets (very aggressive early stopping); the last bracket has few configurations and large budgets (no early stopping, equivalent to random search). This gives Hyperband robustness to the unknown best aggressiveness.

#### 2.3.7 BOHB

Bayesian Optimization HyperBand. Same outer loop as Hyperband, but instead of sampling configurations *uniformly at random* in each bracket, sample them from a TPE-style probabilistic model fit on previously evaluated configurations (at all fidelities). Combines:
- Hyperband's strong any-time performance and parallelism.
- BO's sample efficiency at long horizons.

#### 2.3.8 Population-based methods

Maintain a *population* of models that train concurrently. Periodically:
- *Selection*: identify under-performers.
- *Reproduction*: replace them with copies of better performers.
- *Variation*: perturb hyperparameters.

Variants:
- **Genetic algorithms (GAs)**: discrete encoding, crossover + mutation, fitness-based selection, generation-by-generation evolution.
- **Evolution strategies (ES)** (e.g., CMA-ES): continuous, model the distribution of good solutions as a multivariate Gaussian and update its mean and covariance.
- **Population-Based Training (PBT)** (DeepMind, 2017): population trains in parallel; periodically *exploit* (copy weights of better workers) and *explore* (perturb hyperparameters). Crucially, PBT yields a *schedule* of hyperparameters over training rather than a single static configuration — useful for non-stationary optimal hyperparameters (e.g., learning rate that should decay).

#### 2.3.9 Hyperparameter importance analysis

Once you have run an HPO procedure, you can analyze:
- *Which hyperparameters mattered most* (functional ANOVA).
- *Sensitivity* (how much performance changes when you perturb each one).
- *Ablations* (what happens if you set each hyperparameter to its default).

This produces actionable insight: shrink the search space for unimportant dimensions; expand or refine search for important ones; report which knobs really matter for a given algorithm-dataset pair.

### 2.4 Underlying assumptions and failure modes

| Method | Key assumptions | What breaks |
|---|---|---|
| Grid search | Discretization is fine-grained enough; all dimensions equally relevant | Curse of dimensionality; misses optima between grid points |
| Random search | Hyperparameters are roughly *separable* in importance | Pure exploration; no learning across evaluations |
| Bayesian opt. (GP) | f(λ) is smooth in λ under the chosen kernel; noise model correct | High dimensions, categorical/conditional spaces, very many evaluations |
| Successive Halving | Low-budget performance correlates with high-budget performance | Configurations needing long warmup are killed early |
| Hyperband | Same as SH but hedged across aggressiveness | Still wastes budget if all brackets are too aggressive |
| BOHB | TPE assumptions; SH assumptions | Same as both |
| GA / ES | Crossover / Gaussian mutations are useful in this space | Highly conditional spaces; very small populations get stuck |
| PBT | Hyperparameters can change mid-training without breaking learning | Some hyperparameters (e.g., model architecture) cannot be perturbed mid-training |
| fANOVA | Importance can be decomposed marginally; surrogate model trustworthy | Strong interactions; insufficient data |

---

## 3. Mathematical Formulation

### 3.1 Notation

- λ ∈ Λ — a configuration in the configuration space.
- *f*(λ) ∈ ℝ — true (noiseless) objective; lower is better unless stated.
- *y* = *f*(λ) + ε with ε ~ 𝒩(0, σ²_n) — noisy observation.
- 𝒟_t = {(λ_i, y_i)}_{i=1}^t — history of evaluations.
- *r* ∈ ℝ_+ — fidelity / budget per evaluation.
- f\*_t = min_{i ≤ t} y_i — best value seen so far at time t.

### 3.2 Random search vs. grid search: a quantitative comparison

Suppose only one of `d` hyperparameters truly affects performance, and the optimum lies in the top τ fraction of that dimension (e.g., τ = 0.05 means the optimum lies in the best 5% slice).

- **Grid search** with `n` points per dimension uses `n^d` evaluations total but only `n` distinct values along the relevant dimension. For grid search to find the optimum we need `1/n ≤ τ`, i.e., `n ≥ 1/τ`. Total budget: `n^d ≥ τ^{-d}` evaluations.
- **Random search** with `N` evaluations: each evaluation has probability τ of falling in the good region. Probability that *none* of N evaluations land in the good region is (1 − τ)^N. To achieve "find the good region with probability ≥ 1 − δ":
  
  (1 − τ)^N ≤ δ ⟹ N ≥ log δ / log(1 − τ) ≈ (1/τ) · log(1/δ).

So random search needs roughly **1/τ · log(1/δ)** evaluations versus grid search's **τ^{−d}** — exponentially fewer when d is moderate or large. This is the formal version of Bergstra & Bengio's argument.

### 3.3 Gaussian processes: the surrogate model behind BO

A Gaussian process over Λ is a stochastic process such that any finite set of evaluations {f(λ_1), …, f(λ_n)} is jointly Gaussian. It is fully specified by:

- A **mean function** *m*(λ) — often *m* ≡ 0 after standardizing data.
- A **kernel** (covariance function) *k*(λ, λ′) — encodes the assumption that nearby λ produce similar f.

**Common kernels:**
- Squared exponential (RBF):
  
  k(λ, λ′) = σ_f² exp(− ‖λ − λ′‖² / (2 ℓ²))
  
- Matérn (ν = 5/2 is popular for HPO; RBF can be too smooth):
  
  k(λ, λ′) = σ_f² (1 + √5 d/ℓ + 5 d²/(3 ℓ²)) exp(−√5 d/ℓ),  d = ‖λ − λ′‖.

The kernel hyperparameters (σ_f², ℓ, σ_n²) are called the GP's *own* hyperparameters and are typically learned by maximizing the marginal likelihood:

log p(y | X, θ) = −½ yᵀ (K + σ_n² I)⁻¹ y − ½ log |K + σ_n² I| − (n/2) log 2π.

#### 3.3.1 Posterior predictive

Given data 𝒟_t = (X, y) and a candidate λ, define:
- k_\* = [k(λ, λ_1), …, k(λ, λ_t)]ᵀ.
- K = [k(λ_i, λ_j)]_{i,j=1}^t.

Then:
- Posterior mean: μ(λ) = k_\*ᵀ (K + σ_n² I)⁻¹ y.
- Posterior variance: σ²(λ) = k(λ, λ) − k_\*ᵀ (K + σ_n² I)⁻¹ k_\*.

Cost: O(t³) for the matrix inverse, O(t²) per prediction.

### 3.4 Acquisition functions: derivations

Assume we are minimizing and let f\* = min y_i be the best (lowest) observation.

#### 3.4.1 Probability of Improvement (PI)

PI(λ) = P(f(λ) < f\* − ξ) = Φ((f\* − ξ − μ(λ)) / σ(λ))

where Φ is the standard normal CDF and ξ ≥ 0 is an exploration parameter. Pure PI is *too greedy* with ξ = 0; it picks configurations only marginally better than the current best.

#### 3.4.2 Expected Improvement (EI)

Define improvement I(λ) = max(f\* − f(λ), 0). Under the GP, f(λ) ~ 𝒩(μ(λ), σ²(λ)). Then:

EI(λ) = 𝔼[I(λ)]
      = ∫_{−∞}^{f\*} (f\* − f) · φ((f − μ)/σ)/σ df.

Let z = (f\* − μ(λ)) / σ(λ). After integration by parts:

**EI(λ) = (f\* − μ(λ)) Φ(z) + σ(λ) φ(z)** when σ(λ) > 0; EI = 0 when σ(λ) = 0.

**Derivation sketch.** Substitute u = (f − μ)/σ, du = df/σ:

EI = ∫_{−∞}^{z} (f\* − μ − σu) φ(u) du
   = (f\* − μ) Φ(z) − σ ∫_{−∞}^{z} u φ(u) du.

Use the identity ∫ u φ(u) du = −φ(u). Evaluating from −∞ to z:

∫_{−∞}^{z} u φ(u) du = −φ(z).

So EI = (f\* − μ) Φ(z) + σ φ(z). ∎

Two terms: (f\* − μ)Φ(z) is *exploitation* (large when μ ≪ f\*); σ φ(z) is *exploration* (large when σ is high). EI is the workhorse acquisition function in practice.

#### 3.4.3 Upper / Lower Confidence Bound (UCB / LCB)

For minimization, the **Lower Confidence Bound** is:

LCB(λ) = μ(λ) − κ σ(λ),  pick λ_{next} = argmin LCB(λ).

(For maximization, UCB(λ) = μ(λ) + κ σ(λ).)

κ controls exploration. Srinivas et al. (2010) proved a sublinear regret bound for GP-UCB if κ_t scales as O(√(log t)). In practice κ ∈ [1, 3] is common.

#### 3.4.4 Thompson Sampling

Draw f̃ ~ p(f | 𝒟_t), then minimize f̃. Naturally trades off exploration and exploitation, and parallelizes trivially (each parallel worker draws its own sample). For GPs, one samples function values at a finite candidate set from the joint posterior.

### 3.5 Tree-structured Parzen Estimator (TPE)

TPE models p(λ | y) instead of p(y | λ). Split history into:
- Good: y < y\* (e.g., bottom γ-quantile of observed y).
- Bad: y ≥ y\*.

Fit two density estimators (kernel density / Parzen):
- ℓ(λ) = p(λ | good), g(λ) = p(λ | bad).

TPE uses **Expected Improvement is proportional to ℓ(λ)/g(λ)** as the acquisition. Pick λ_{next} = argmax ℓ(λ)/g(λ). It is well-suited to tree-structured (conditional) configuration spaces because the densities are factorized along the tree.

### 3.6 Successive Halving

Inputs: total budget B, reduction factor η, initial number of configurations n, minimum resource r.

Pseudocode:
```
configs ← sample n configurations
resource ← r
while |configs| > 1:
    losses ← evaluate each config in configs with budget = resource
    keep the top (|configs| / η) configs by loss
    resource ← resource * η
return best config
```

Number of rounds: s_max + 1 ≈ log_η(R/r), where R is the maximum resource per configuration. After all rounds, total resource consumed:

B = n·r + (n/η)·rη + (n/η²)·rη² + … ≈ (s_max + 1) · n · r.

So Hyperband chooses n given the bracket budget B and resource caps:

n = ⌈ B / (R · (s+1)) · η^s ⌉, r = R · η^{−s},

where s is the bracket index from s_max down to 0.

### 3.7 Hyperband

Inputs: maximum resource R, reduction factor η.

Let s_max = ⌊log_η R⌋ and B_total = (s_max + 1) · R.

For s = s_max, s_max − 1, …, 0:
- n = ⌈ B_total / R · η^s / (s + 1) ⌉
- r = R · η^{−s}
- Run Successive Halving on n random configurations starting at resource r.

Bracket s = s_max: many configurations (n large), tiny per-config budget, very aggressive early stopping.
Bracket s = 0: one configuration, full budget, no early stopping (≈ random search of one config).

Hyperband's theoretical guarantee: under mild assumptions on the convergence of the loss curves, Hyperband achieves nearly the same rate as the best fixed (n, r) in hindsight, up to a log factor.

### 3.8 BOHB

Inner loop identical to Hyperband. The crucial change is in **how configurations are sampled at the start of each bracket**:

- For the first n_min evaluations, sample uniformly at random.
- After that, fit a TPE model on past evaluations (using their best-fidelity scores). With probability ρ (typically ~1/3), sample uniformly to maintain exploration; otherwise sample from the TPE model.

The TPE model uses *all* observations across all fidelities by training on the *highest-fidelity* observation available for each configuration.

### 3.9 Evolutionary methods

#### 3.9.1 Genetic algorithm (GA)

Encode each configuration λ as a *chromosome* (e.g., concatenated discrete strings). Maintain a population P of size N. Each generation:

1. **Evaluate** fitness f(λ) for every λ ∈ P.
2. **Select** parents (e.g., tournament: pick k random, keep best; or rank-proportional).
3. **Crossover**: combine two parents into a child (e.g., uniform crossover swaps each gene with prob ½).
4. **Mutate**: with small prob μ_m, randomly change each gene.
5. Replace worst members of P with offspring.

#### 3.9.2 Evolution strategies and CMA-ES

For continuous λ, model the search distribution as 𝒩(m, C). Each generation:

1. Sample λ_i ~ m + 𝒩(0, σ² C), i = 1..N.
2. Evaluate; rank.
3. Update m as a weighted mean of best samples.
4. Update C using rank-µ and rank-1 updates that adapt the covariance toward directions of past success.
5. Update step size σ using cumulative path-length adaptation.

CMA-ES is state-of-the-art among black-box optimizers for moderate-dimensional continuous spaces and is invariant to monotonic transformations of f.

#### 3.9.3 Population-Based Training (PBT)

Population of N models trains in parallel, each with its own hyperparameter schedule. Every k steps:

- **Exploit**: each worker compares to a random other worker; if its performance is in the bottom 20%, it copies the other's weights and hyperparameters.
- **Explore**: perturb hyperparameters, e.g., multiply by a factor sampled from {0.8, 1.25} or resample.

Output is not a single configuration but a *trajectory* of (weights, hyperparameters) over training. PBT was originally used for RL agents and large-scale neural networks where a static optimal hyperparameter doesn't exist (e.g., learning rate that should anneal).

### 3.10 Functional ANOVA (fANOVA)

Given a surrogate f̂ (e.g., a random forest fit on all (λ, y) pairs from the HPO run), decompose:

f̂(λ) = f_∅ + Σ_i f_i(λ_i) + Σ_{i<j} f_{i,j}(λ_i, λ_j) + …

where each f_S is chosen so that its integral over any of its variables vanishes. The *variance contribution* of subset S is:

V_S = ∫ f_S(λ_S)² dλ_S.

Total variance V = Σ_S V_S, and *fANOVA importance* of S is V_S / V. For S = {i} this is the *main effect* of hyperparameter i; for S = {i, j} it is the *interaction* between i and j. fANOVA is implemented in the SMAC and ConfigSpace ecosystem and is the standard tool for post-hoc importance analysis.

---

## 4. Worked Examples

### 4.1 Random vs. grid: a back-of-the-envelope demonstration

Suppose 4 hyperparameters; only `learning_rate` truly matters; the optimum lies in the top 10% of `learning_rate`'s range; we have a budget of 100 evaluations.

- **Grid search**: 100^{1/4} ≈ 3.16 → in practice we use 3 or 4 values per dim. With 3 values we test only `learning_rate` ∈ {low, mid, high}. Probability the "best" value falls in the top-10% slice of `learning_rate`: 0.10. So with grid we only have 3 chances along the relevant axis, and only one will land in the top-10% if at all.
- **Random search**: probability *no* sample falls in top-10%: 0.9^{100} ≈ 2.7e-5. Almost surely we get many top-10% samples.

This is why for n ≥ 20 or so, random search dominates grid search whenever effective dimensionality is small.

### 4.2 A complete Bayesian optimization step

We minimize a 1-D test function f(λ) = (λ − 2)² + sin(5λ) on Λ = [0, 4] with a GP using an RBF kernel (ℓ = 0.5, σ_f = 1, σ_n = 0.05).

**Initial design**: evaluate λ = 0.5, 2.5, 3.5. Suppose y = (3.10, 0.25, 1.32).

Currently, f\* = 0.25 at λ = 2.5.

**GP posterior at candidate λ = 1.5**: compute K_{star} = [k(1.5, 0.5), k(1.5, 2.5), k(1.5, 3.5)].

With ℓ = 0.5, the squared distances are 1, 1, 4, so kernel values are exp(−1/0.5) = 0.135, 0.135, exp(−8) ≈ 3.4e-4.

K (3×3) is built similarly. After the matrix inversion (mechanical), suppose μ(1.5) = 1.6 and σ(1.5) = 0.8.

**EI computation**: z = (f\* − μ)/σ = (0.25 − 1.6)/0.8 = −1.6875.

- Φ(−1.6875) ≈ 0.0458
- φ(−1.6875) ≈ 0.0967

EI = (0.25 − 1.6) · 0.0458 + 0.8 · 0.0967 = −0.0618 + 0.0774 = **0.0156**.

**At candidate λ = 2.0**: closer to f\*, suppose μ(2.0) = 0.4, σ(2.0) = 0.3.

z = (0.25 − 0.4)/0.3 = −0.5. Φ(−0.5) = 0.3085, φ(−0.5) = 0.3521.

EI = (0.25 − 0.4) · 0.3085 + 0.3 · 0.3521 = −0.0463 + 0.1056 = **0.0593**.

So λ = 2.0 has higher EI and would be picked next, even though its mean (μ = 0.4) is not as low as some other candidates — the moderate uncertainty plus closeness to the current best wins.

### 4.3 A Hyperband schedule with R = 81, η = 3

Compute s_max = ⌊log_3 81⌋ = 4. Total bracket budget B_total = (s_max + 1) · R = 5 · 81 = 405.

| s | n | r | Sequence (configs, resource per config) |
|---|---|---|---|
| 4 | 81 | 1 | (81,1) → (27,3) → (9,9) → (3,27) → (1,81) |
| 3 | 27 | 3 | (27,3) → (9,9) → (3,27) → (1,81) |
| 2 | 9 | 9 | (9,9) → (3,27) → (1,81) |
| 1 | 6 | 27 | (6,27) → (2,81) |
| 0 | 5 | 81 | (5,81) |

Total resource per bracket ≈ 5·81 = 405 (some rounding); total Hyperband budget ≈ 5·405 = 2025 resource units. Compare to pure random search with the same budget: 2025/81 = 25 full-resource configurations. Hyperband evaluates ~128 distinct configurations across the brackets, killing weak ones early.

If, say, "epoch" is the resource and R = 81 epochs, Hyperband touches 128 configurations using compute equivalent to 25 fully-trained models — a much wider exploration.

### 4.4 BOHB on a CNN

Imagine tuning learning rate (log-uniform 1e-5–1e-1), weight decay (log-uniform 1e-6–1e-2), batch size (categorical {32, 64, 128}), and dropout (uniform 0–0.5) for a CNN with R = 27 epochs, η = 3.

- Bracket s=3: 27 configs, 1 epoch each; survivors → 9 configs, 3 epochs; → 3 configs, 9 epochs; → 1 config, 27 epochs. The 27 initial configs are sampled uniformly at random for the very first bracket.
- After that bracket, BOHB has 27 + 9 + 3 + 1 = 40 datapoints (mostly at low fidelity). It fits a TPE model.
- Bracket s=2: now 9 configs at 3 epochs each, then survivors → 3 at 9, → 1 at 27. The 9 starting configs come from TPE with probability 1−ρ, uniform with probability ρ.
- Continue.

After many brackets, BOHB has explored hundreds of configurations cheaply and exploited the most promising regions intensively.

### 4.5 PBT on a reinforcement-learning agent

8 actors train on the same RL task in parallel; each starts with random (lr, entropy_coef, gamma) drawn from prior ranges. Every 1M environment steps:

1. Each actor reports mean episode reward.
2. Worst 25% (2 actors) are replaced: each copies a random *top-25%* actor's weights and hyperparameters.
3. The copied hyperparameters are then perturbed by ×0.8 or ×1.25.

After 50M steps, the schedule that emerges might look like: lr starts at 3e-4, drifts down to 5e-5; entropy_coef starts at 0.02, drops to 0.001 — a dynamic schedule no static configuration could provide. The final population's best agent is returned.

### 4.6 fANOVA on a tuning run

Suppose you ran 1000 BOHB evaluations on the CNN above. Fit a random forest surrogate on (λ, validation accuracy) pairs. Decompose variance:

| Hyperparameter / interaction | Variance fraction |
|---|---|
| learning_rate | 0.62 |
| weight_decay | 0.10 |
| dropout | 0.04 |
| batch_size | 0.02 |
| lr × weight_decay | 0.15 |
| lr × dropout | 0.03 |
| Other interactions | 0.04 |

Conclusion: ~60% of variance is explained by `learning_rate`'s main effect, ~15% by its interaction with weight decay. Future runs should focus search budget on these two; you can largely lock in default values for batch_size and dropout.

---

## 5. Relevance to Machine Learning Practice

### 5.1 Where HPO sits in the ML lifecycle

- **Training**: HPO drives the choice of optimizer, regularization, architecture knobs.
- **Inference**: post-training calibration knobs (temperature scaling, threshold) can also be tuned.
- **Evaluation**: HPO needs a careful evaluation protocol — usually a held-out validation set distinct from the test set, or nested cross-validation.
- **Monitoring**: in production, you may re-tune as data drifts; some systems (e.g., recommender systems) tune online via bandits or PBT-like methods.

### 5.2 How to choose a method

A practical decision tree:

1. **Tiny budget (≤ 20 evaluations) and few hyperparameters (≤ 3)**: grid or manual search. Bayesian optimization needs at least ~10 points of warmup; less than that, random search is more robust.
2. **Moderate budget (50–500 evaluations), continuous + a few categorical hyperparameters, expensive evaluations**: Bayesian optimization (GP-EI or TPE).
3. **Large budget, but each evaluation supports early stopping (epochs, dataset subsamples)**: Hyperband or BOHB.
4. **Massively parallel compute, neural networks, schedule-sensitive hyperparameters (lr, entropy)**: PBT.
5. **Continuous-only, smooth, ~10–50 dimensions, very expensive evaluations**: CMA-ES.
6. **Highly conditional / structured (e.g., AutoML pipelines)**: TPE, SMAC, BOHB.

### 5.3 Trade-offs

| Property | Grid | Random | BO (GP) | TPE | Hyperband | BOHB | CMA-ES | PBT |
|---|---|---|---|---|---|---|---|---|
| Sample efficiency | Low | Low | High | Med-High | Med | High | High (cont.) | High (any-time) |
| Parallelism | Trivial | Trivial | Hard (sequential acquisition) | Easier | High | High | Med (pop-based) | Native |
| Handles categorical/conditional | Yes | Yes | Awkward | Yes | Yes | Yes | No | Yes |
| Handles high-dim | No | OK | No (>~20) | OK | OK | OK | OK (~50) | OK |
| Provides schedule | No | No | No | No | No | No | No | Yes |
| Implementation complexity | Trivial | Trivial | High | Med | Med | Med-High | Med-High | High |

### 5.4 Common tools

- **Optuna** (TPE, Hyperband, NSGA-II, integrations everywhere). Default for many practitioners.
- **Ray Tune** (population-based training, ASHA, BOHB, PBT, distributed).
- **Hyperopt** (TPE; older but still used).
- **SMAC3** (random forests as surrogates; great for mixed/conditional spaces; produced fANOVA tools).
- **Ax / BoTorch** (Facebook; modern GP-BO with PyTorch backend; supports multi-objective, multi-fidelity).
- **scikit-optimize** (lightweight GP-BO).
- **Vertex AI HPO**, **Sagemaker HPO**, **W&B Sweeps** — managed services.
- **DEAP**, **pycma** — for evolutionary methods.

### 5.5 Computational considerations

- **Validation noise dominates** at small datasets: a single validation split can swing accuracy by a percent. Use k-fold CV or repeat evaluations.
- **Per-evaluation cost vs. number of configurations** is the central trade-off. A common waste is to use Bayesian optimization on a problem where each evaluation is fast — random search would be both simpler and competitive.
- **Multi-fidelity is the single largest win**: amortizing Hyperband over Bayesian optimization (i.e., BOHB) routinely yields 5–30× speedups vs. vanilla BO.
- **Parallelism** matters more than method theoretical advantages when you have a cluster: methods that parallelize well (Hyperband, PBT) often beat sequentially-superior ones (vanilla GP-BO).
- **Anytime performance**: in production, you usually need the best result within X hours. Hyperband's bracket structure gives strong any-time bounds: you have a passable result quickly and a great one if you wait.

### 5.6 Reporting and reproducibility

- Always **log the search space** in your paper or experiment notebook. "We tuned hyperparameters" without specifying ranges is meaningless.
- Always **report the budget**: number of trials, total wall-clock, number of CV folds.
- Run multiple seeds of the *HPO procedure itself*; report variance.
- For fairness: use the same HPO budget for baseline and proposed method.

---

## 6. Common Pitfalls & Misconceptions

### 6.1 Confusing parameters and hyperparameters

Beginners often confuse these. A useful test: if it changes during gradient descent on the training loss, it's a parameter. If you set it before training and a smaller / larger value would require retraining from scratch, it's a hyperparameter.

### 6.2 Tuning on the test set

The single biggest failure. Tuning hyperparameters using test-set performance leaks information and inflates reported numbers. Use:
- A **dedicated validation set**, or
- **Nested cross-validation**: outer CV for evaluation, inner CV for tuning.

### 6.3 Linear vs. log scales

Learning rates and regularization coefficients span orders of magnitude. Sampling them on a *linear* scale wastes most samples on large values. Always sample log-uniform for these.

Concretely: if the optimum is around 1e-3 but you sample uniformly on [0, 0.1], 90% of your samples are above 1e-2 and you'll find nothing useful.

### 6.4 Treating BO as a black box that magically works

Bayesian optimization has its own *meta-hyperparameters*: kernel choice, length scale priors, acquisition function, exploration parameter. Defaults work for low dimensions and smooth losses. They fail silently in:
- High dimensions (kernel is too smooth or too sharp; uncertainty estimates collapse).
- Categorical-heavy spaces (RBF kernel doesn't apply naturally).
- Noisy objectives (the GP overfits noise and proposes the same point repeatedly).

### 6.5 Killing configurations too aggressively

Successive Halving / Hyperband trust that early-budget rankings predict late-budget rankings. For configurations needing warmup (e.g., a low learning rate that needs many epochs to overtake a high-lr competitor), this fails. Mitigations:
- Use Hyperband's least-aggressive bracket as well.
- Use *learning curve extrapolation* (a small parametric model of the loss vs. epoch curve) rather than raw early-epoch performance.

### 6.6 Forgetting noise

Each evaluation of f is noisy (initialization seed, mini-batch order, CV fold). If σ_n is comparable to the gap between configurations, you'll think you've found a winner when you've just been lucky. Mitigations:
- Average over multiple seeds for top configurations.
- Use BO acquisition functions that account for noise (noisy EI, knowledge gradient).

### 6.7 Search space too narrow

If your search space doesn't contain the true optimum, no method can find it. Always start with a *wide* range and let the data tell you where to narrow. Hyperparameter importance analyses (fANOVA) help narrow intelligently after a first sweep.

### 6.8 Over-tuning

Running HPO with 10 000 trials on a small dataset is not science; it is overfitting. The HPO procedure itself can overfit the validation set. Symptoms: best HPO result far better than test result; best configurations on the boundary of the search space; tiny accuracy differences between runs.

### 6.9 Conflating "more trials" with "better"

Adding 50 more trials is rarely worth it once the marginal improvement per trial is below the noise floor of your evaluation. Stop when the rolling-best curve plateaus.

### 6.10 Misreading random search's strength

Random search is *not* magic. Bergstra & Bengio's argument requires the effective dimensionality to be small. In tightly coupled, high-dimensional spaces (e.g., pieces of an RL pipeline), random search can be the worst option.

### 6.11 Ignoring categorical structure

Treating "optimizer ∈ {SGD, Adam}" as a continuous variable in {0, 1} or {0, 0.5, 1} is meaningless. Use TPE or methods with proper categorical handling.

### 6.12 PBT pitfalls

PBT is powerful but tricky:
- Copying weights between models with very different recent learning rates can destabilize the receiver.
- Choosing the perturbation factor too aggressively leads to oscillations.
- Some hyperparameters (architectural ones) cannot be perturbed mid-training.

### 6.13 Misinterpreting fANOVA

fANOVA importance is *with respect to the surrogate*. If your surrogate is bad (too few datapoints, wrong model class), importance numbers are unreliable. Always check surrogate quality (e.g., out-of-bag R²) before trusting the decomposition.

---

# Interview Preparation

## A. Foundational Questions

**A1. What's the difference between a parameter and a hyperparameter?**

Parameters are values learned from data by the optimizer (weights, biases, split thresholds). They are updated via gradients (or other learning rules) on the training loss. Hyperparameters are configuration variables set *before* training begins and govern the learning process or model structure (learning rate, regularization strength, number of layers, kernel choice). They are not updated by the inner optimizer; they are chosen by the modeler or by an outer optimizer (HPO).

A useful operational test: if changing the value requires retraining from scratch (or at least restarting the optimizer state), it's a hyperparameter.

**A2. Why is grid search inefficient in high dimensions?**

Grid search evaluates every combination of pre-specified values. With `d` dimensions and `n` values per dimension, the cost is `n^d`. Even modest `n=10` and `d=6` requires a million evaluations.

Worse, grid search wastes budget on irrelevant dimensions. If only 2 of 6 hyperparameters truly matter, grid search still varies the other 4 — you spend most evaluations on redundant variation. Random search avoids this because each sample varies *all* dimensions independently.

**A3. Give the intuition for why random search beats grid search.**

Suppose only `d_eff` of `d` hyperparameters strongly affect performance. With `N` total evaluations:
- Grid search produces only `N^{1/d}` unique values along any single dimension.
- Random search produces ~`N` unique values along every dimension.

If the effective dimensionality is low, random search effectively conducts a 1-D scan along every important axis simultaneously, while grid search "wastes" evaluations replicating identical projections onto irrelevant axes. Bergstra and Bengio (2012) demonstrated this empirically and theoretically.

**A4. Explain Bayesian optimization in one paragraph.**

Bayesian optimization fits a probabilistic surrogate model — typically a Gaussian process — to the evaluations seen so far. The surrogate provides a posterior mean μ(λ) and uncertainty σ(λ) at every candidate configuration. An acquisition function (e.g., Expected Improvement) scores each candidate by trading off exploitation (low μ) against exploration (high σ). The next configuration to evaluate is the argmax of the acquisition function. After each evaluation, the surrogate is refit and the process repeats. BO is sample-efficient because every evaluation informs the model's belief about the entire search space.

**A5. What is the difference between Successive Halving and Hyperband?**

Successive Halving (SH) starts with `n` configurations, evaluates all with budget `r`, keeps the top `1/η`, multiplies their budget by `η`, and repeats. SH requires the user to choose `n` (or equivalently, the trade-off between exploration breadth and per-config training depth).

Hyperband removes this choice by running multiple SH instances ("brackets") with different `(n, r)` combinations: aggressive brackets explore many configurations briefly; conservative brackets behave like random search of fully-trained configurations. Hyperband hedges across the unknown best aggressiveness and provably matches the best fixed bracket up to log factors.

**A6. What does "multi-fidelity optimization" mean and why does it help?**

Multi-fidelity methods evaluate configurations at varying levels of cost ("fidelity"): a low-fidelity evaluation might be 1 epoch on 10% of the data; a high-fidelity evaluation might be 50 epochs on the full data. The intuition is that bad configurations can be eliminated cheaply, so we should use most of our compute on the configurations that look promising. Hyperband, BOHB, and learning-curve-extrapolation BO are multi-fidelity methods.

**A7. What is BOHB and how does it combine BO and Hyperband?**

BOHB uses the Hyperband outer loop (brackets of Successive Halving with varying aggressiveness) for budget allocation and early stopping. Its innovation is in *how* configurations are sampled at the start of each bracket. Instead of random sampling (vanilla Hyperband), BOHB fits a TPE model on past observations (using each configuration's highest-fidelity score) and samples new configurations from that model. It thus combines Hyperband's strong any-time performance with BO's improving sample efficiency over time.

**A8. How does PBT differ from other HPO methods?**

PBT (Population-Based Training) runs N models in parallel, sharing weights via copying when a worker performs poorly. Every checkpoint, underperformers copy a top performer's weights and hyperparameters and then perturb the hyperparameters. This produces a *schedule* of hyperparameters over the course of training, not a single static configuration. It is uniquely suited to settings where the optimal hyperparameter changes during training (learning rate annealing in RL or large-scale supervised learning).

**A9. What is functional ANOVA used for?**

fANOVA decomposes the variance of a surrogate model into contributions from individual hyperparameters and their interactions. After an HPO run, fANOVA tells you which hyperparameters mattered most and which interactions were significant. This lets you shrink the search space for unimportant dimensions and focus future search on the impactful ones. It's the standard tool for post-hoc HPO importance analysis.

---

## B. Mathematical Questions

**B1. Derive the formula for Expected Improvement under a Gaussian surrogate.**

Let f(λ) ~ 𝒩(μ(λ), σ²(λ)) under the GP, and let f\* = min y_i be the current best. Define improvement I = max(f\* − f(λ), 0) (we minimize). Then:

EI(λ) = 𝔼[I] = ∫_{−∞}^{f\*} (f\* − f) · (1/σ) φ((f − μ)/σ) df.

Substitute u = (f − μ)/σ, du = df/σ, with upper limit z = (f\* − μ)/σ:

EI = ∫_{−∞}^{z} (f\* − μ − σu) φ(u) du
   = (f\* − μ) ∫_{−∞}^{z} φ(u) du − σ ∫_{−∞}^{z} u φ(u) du
   = (f\* − μ) Φ(z) − σ · [−φ(u)]_{−∞}^{z}
   = (f\* − μ) Φ(z) + σ φ(z).

Hence **EI(λ) = (f\* − μ(λ)) Φ(z) + σ(λ) φ(z)** for σ > 0; EI = 0 when σ = 0. The first term is exploitation, the second exploration.

**B2. What's the computational complexity of Gaussian-process Bayesian optimization?**

Fitting the GP with `n` points requires inverting an `n × n` matrix → O(n³) time and O(n²) memory. Predictions cost O(n²) for variance. Optimizing the acquisition function over Λ adds further cost (typically thousands of acquisition evaluations per BO step). This is why vanilla GP-BO becomes impractical above ~1000–5000 evaluations and why approximations (sparse GPs, random forests as surrogates, deep kernel learning) or alternative surrogates (TPE, random forests) are used at scale.

**B3. Suppose you have a noisy objective. How does that change EI?**

The standard EI formula assumes f\* is the *true* best value, but with noise, f\* = min y_i is a *random* variable that under-estimates the true minimum. This makes EI overconfident: it thinks improvement is harder than it is and over-exploits. Mitigations:
- **Noisy EI**: replace f\* with the GP's predicted mean at the best observed input (or a quantile of it).
- **Knowledge Gradient (KG)**: directly models the expected one-step look-ahead reduction in posterior minimum.
- **Average y over multiple seeds** before fitting the GP.

**B4. Show that for separable objectives where only `d_eff` of `d` dims matter, random search needs O(τ⁻¹ log δ⁻¹) samples to find the optimum with probability 1 − δ, while grid search needs O(τ⁻ᵈ).**

Suppose the optimum lies in the top τ fraction of the relevant axis. For random search, each sample independently lands in the good region with probability τ. P(no good sample after N draws) = (1 − τ)^N. Demanding this be ≤ δ:

(1 − τ)^N ≤ δ ⟹ N ≥ log δ / log(1 − τ) ≈ τ⁻¹ log δ⁻¹ for small τ.

For grid search with `n` per dim, total cost `n^d`, but only `n` distinct values along the relevant axis. We need 1/n ≤ τ ⟹ n ≥ τ⁻¹ ⟹ total = n^d ≥ τ⁻ᵈ.

Random search beats grid search whenever d > 1.

**B5. Derive the GP posterior mean and variance.**

Joint distribution of training observations y (n×1) and test value f_\* at λ_\*:

[y; f_\*] ~ 𝒩(0, [K + σ_n²I, k_\*; k_\*ᵀ, k(λ_\*, λ_\*)]).

Conditioning the joint Gaussian on y:

f_\* | y ~ 𝒩(μ_\*, σ²_\*) with:

μ_\* = k_\*ᵀ (K + σ_n²I)⁻¹ y,
σ²_\* = k(λ_\*, λ_\*) − k_\*ᵀ (K + σ_n²I)⁻¹ k_\*.

This follows from the standard formula for conditioning a multivariate Gaussian: if [a; b] ~ 𝒩([μ_a; μ_b], [Σ_aa, Σ_ab; Σ_ba, Σ_bb]) then a | b ~ 𝒩(μ_a + Σ_ab Σ_bb⁻¹ (b − μ_b), Σ_aa − Σ_ab Σ_bb⁻¹ Σ_ba).

**B6. How do we choose κ in UCB? What's the regret bound?**

For GP-UCB on a compact domain with a stationary kernel, Srinivas et al. (2010) showed that choosing κ_t = O(√(γ_t log t)), where γ_t is the maximum information gain after t rounds, achieves cumulative regret O(√(t γ_t log t)) with high probability. For RBF kernels in d dimensions, γ_t = O((log t)^{d+1}), giving sublinear regret.

In practice, a constant κ ∈ [1.5, 3] works well. Smaller κ → exploitation; larger κ → exploration.

**B7. Show why TPE's acquisition is proportional to ℓ(λ)/g(λ).**

Define the threshold y\* such that p(y < y\*) = γ. TPE models p(λ | y < y\*) = ℓ(λ) and p(λ | y ≥ y\*) = g(λ). The marginal:

p(λ) = γ ℓ(λ) + (1 − γ) g(λ).

Expected improvement (in maximization form, over threshold y\*):

EI(λ) ∝ ∫ max(y\* − y, 0) p(y | λ) dy.

Bayes' rule: p(y | λ) = p(λ | y) p(y) / p(λ). Plugging in and simplifying (as shown in Bergstra et al., 2011), one obtains:

EI(λ) ∝ (γ + (1 − γ) g(λ)/ℓ(λ))⁻¹.

Maximizing EI is therefore equivalent to maximizing ℓ(λ) / g(λ): pick configurations the "good" model thinks are likely and the "bad" model thinks are unlikely.

**B8. In Hyperband, why is total resource per bracket roughly the same?**

Each bracket runs Successive Halving from `n` configurations at resource `r` per config. The geometric structure ensures that at every halving level k = 0, 1, …, s the number of configs times resource per config is the same:

(n / η^k) · (r · η^k) = n · r.

There are s+1 levels, so per-bracket budget ≈ (s+1) · n · r ≈ R (after substituting n and r per Hyperband's formulae). Total Hyperband budget ≈ (s_max + 1) · R, scaling linearly with R rather than exponentially.

**B9. Why does CMA-ES adapt a covariance matrix?**

If the objective has elongated, anisotropic level sets (e.g., a narrow valley aligned with no axis), an isotropic Gaussian search distribution is wasteful — most samples land orthogonal to the descent direction. CMA-ES learns the principal axes and scales of the local landscape from the past history of successful samples and reshapes its sampling Gaussian accordingly, allowing it to make large steps along correlated directions and small steps perpendicular to them. This makes it invariant to linear transformations of the search space.

---

## C. Applied Questions

**C1. You have 8 GPUs and 24 hours. You are tuning a transformer. Which HPO method?**

Probably **BOHB** or **ASHA** (Asynchronous SH). Reasons:
- Each evaluation is expensive (training a transformer for many epochs), so multi-fidelity is essential.
- 8 GPUs need parallelism; ASHA/BOHB are designed for asynchronous parallelism.
- Bayesian sampling improves over time, so BOHB beats vanilla Hyperband when budget is not tiny.

If the problem is highly schedule-sensitive (some RL settings, very long training), **PBT** might be a better choice — it adapts hyperparameters online and uses GPUs continuously.

**C2. A junior colleague says "I tuned the model with grid search across learning rate ∈ {0.001, 0.01, 0.1}". What problems do you see?**

- **Tiny coverage**: only 3 values across a range that should probably span 1e-5 to 1e-1.
- **Linear scale on a log-scale parameter**: 0.001 and 0.01 are very far apart relatively; you may have missed the optimum at, say, 3e-3.
- **No interaction with other hyperparameters**: changing only learning rate while keeping (e.g.) batch size and weight decay fixed may not reveal the best joint configuration.
- **No mention of validation protocol**: was this on a held-out set or test set?
- **No reproducibility info**: seed, search budget, time.

Better: random search or BO over (lr ∈ log-uniform[1e-5, 1e-1], wd ∈ log-uniform[1e-6, 1e-2], …) for some larger budget.

**C3. You ran BO and the search keeps proposing nearly the same point. What's wrong?**

Most likely the GP's noise model is mis-specified — σ_n² is too small, so the GP fits the noise and predicts a tight global minimum at the lucky-low point. The acquisition function then sticks to that point.

Diagnostics and fixes:
- Increase or learn σ_n² (re-fit the kernel hyperparameters by maximizing marginal likelihood).
- Use noisy EI / Knowledge Gradient acquisition.
- Average each evaluation over multiple seeds.
- Add a small exploration bonus or ensure ξ > 0 in PI/EI variants.
- Switch to UCB with reasonable κ.

**C4. You have a categorical hyperparameter (optimizer ∈ {SGD, Adam, RMSProp}) and many continuous ones. Which surrogate?**

A vanilla GP with RBF kernel does not handle categorical variables well. Options:
- **TPE** (Optuna, Hyperopt): natively supports mixed/conditional spaces.
- **Random forests** (SMAC): handle categorical splits natively.
- **GPs with mixed kernels**: e.g., RBF on continuous dims combined with a Hamming kernel on categoricals (more advanced; Ax/BoTorch supports this).
- For very small budgets, just run separate BO/Random searches per categorical level.

**C5. Your Hyperband run kills configurations with very low learning rates after 1 epoch, but you suspect those configurations would have won with more training. What do you do?**

Several options:
- Use Hyperband's **most conservative bracket** (s = 0) and ensure it gets budget — that bracket is essentially full random search with no early stopping.
- Use **learning curve extrapolation**: fit a parametric curve (e.g., LCNet or a sigmoid) to the loss-vs-epoch curve and project to full training; rank by the projected score, not the raw early-epoch score.
- Increase the minimum resource `r_min` so even the smallest budget covers any warmup phase.
- Use **PBT** so low-lr models are not killed but allowed to copy from higher-lr models if they fall too far behind, gaining a warmup head start.

**C6. After 200 BO evaluations, performance has plateaued. Should you keep running?**

Diagnose first:
- Plot the rolling-best metric vs. iterations. If flat for the last ~30% of trials, marginal returns are tiny.
- Check whether you're hitting the boundary of the search space — if yes, expand it.
- Check the surrogate uncertainty in unexplored regions — if low everywhere, the GP thinks it has converged.
- Consider running multiple seeds of the *whole HPO process* to assess the variance of "best" across runs.

If the plateau is genuine, stop. Money is better spent on architecture or data improvements.

**C7. You are tuning a model whose evaluation is very fast (1 second per fit). Which method?**

When evaluations are cheap, the overhead of fitting a GP and optimizing the acquisition function dominates. **Random search or even grid search** are usually best. If you have many cores, parallelize random search trivially. Bayesian optimization adds complexity without payoff at this regime.

**C8. Describe an end-to-end HPO pipeline for a production model.**

1. **Define the search space**: log-uniform for learning rates and regularization, uniform for dropout and momentum-like terms, categorical for optimizer and activations. Document ranges.
2. **Decide the budget**: e.g., 200 evaluations × 8 parallel workers × 1 hour each = 1600 GPU-hours.
3. **Select the validation protocol**: stratified 5-fold CV or held-out 80/10/10 split. Lock down the test set, never look at it during HPO.
4. **Pick the method**: BOHB on Ray Tune for parallel multi-fidelity HPO.
5. **Run with logging**: track every (λ, score, runtime, fidelity) tuple in MLflow / W&B.
6. **Post-hoc analysis**: fit fANOVA on the trajectory; identify which hyperparameters mattered.
7. **Refine and iterate**: shrink unimportant dims, expand important ones, rerun once more if budget allows.
8. **Final evaluation**: best configuration → retrain on train+val → evaluate on test set. Report.
9. **Re-tune cadence**: schedule re-tuning every N weeks or when data drift detection fires.

**C9. When would you choose CMA-ES?**

CMA-ES shines on:
- Continuous, moderate-dimensional (10–50) objectives.
- Smooth or weakly-noisy black-box functions.
- Problems with strong correlations between dimensions (anisotropic landscapes).

It's the default in some scientific computing and engineering optimization domains. For ML hyperparameters specifically, CMA-ES is less common because most ML configuration spaces have categorical/conditional structure CMA-ES doesn't natively handle. But for purely continuous, noisy black-boxes (e.g., training-free architecture metrics, neural network architecture parameters that are continuous), it is competitive.

**C10. How do you tune hyperparameters when each evaluation requires distributed training across 64 GPUs?**

When per-evaluation cost is enormous, you cannot afford many trials. Strategies:
- **Use multi-fidelity aggressively**: short proxy training runs (fewer epochs, smaller dataset, smaller model) to filter configs before full-scale evaluation.
- **Transfer from smaller models**: tune on a 100M-parameter model first; transfer the best configurations to the 10B-parameter model with mild adjustments (e.g., μP scaling laws for learning rate vs. width).
- **Use prior knowledge**: published rules of thumb, scaling laws, hyperparameter transfer techniques (e.g., μ-Transfer).
- **Tune very few knobs**: lock in defaults for everything but learning rate, batch size, and weight decay.
- **PBT** if you have spare cycles: lets the population co-adapt without restarting.

---

## D. Debugging & Failure-Mode Questions

**D1. After HPO, your model performs worse on the test set than the previous default. What might have happened?**

- **Test-set leakage during HPO**: rare but check.
- **Overfitting the validation set**: too many trials on a single small validation split. Best HPO score is an upward-biased estimator of true performance. Use nested CV or hold out an extra "HPO test" set.
- **Distribution shift** between validation and test.
- **Variance**: the "best" configuration's reported score had high noise; the second- or third-best configurations might generalize better. Average top-K configurations.
- **Search space too narrow** missed the previous defaults.

**D2. Your Bayesian optimization run never explores beyond a small region. Why?**

- Length scale ℓ in the kernel is too large → GP thinks the function is very smooth → uncertainty collapses early → acquisition function exploits a narrow region.
- Acquisition is too greedy (PI with ξ=0; EI with low noise estimate; UCB with too-small κ).
- Insufficient initial design — start with at least 10–20 random points before letting BO take over.

**D3. Hyperband's first bracket evaluates 81 configurations with 1 epoch each. They all look terrible. What's going on?**

Likely the model needs more than 1 epoch even to begin learning. The minimum resource `r_min` is too small. Increase `r_min` (e.g., to 5 epochs), reduce `R` proportionally, or skip the most aggressive bracket.

**D4. PBT's best worker keeps "winning" but has crazy hyperparameters that don't make sense. What should you check?**

PBT tends to find "lucky" trajectories — a worker may have copied a strong checkpoint and not actually have the best hyperparameters; its reported reward reflects the donor's training, not its own current configuration. Diagnostics:
- Check the worker's training curve before vs. after the most recent exploit/explore step.
- Re-train from scratch with the "winning" configuration to verify it's actually competitive.
- Reduce the perturbation factor; reduce the truncation fraction.

**D5. fANOVA tells you "weight decay" has 50% importance, but ablating weight decay barely changes performance. Reconcile.**

Possibilities:
- **Surrogate misfit**: the random forest fit on HPO history is wrong. Check OOB R² — if low, fANOVA is unreliable.
- **Interaction effect**: weight decay's main effect may be small but its interactions large. fANOVA's variance decomposition into mains can be misleading without inspecting interactions.
- **Search space issue**: if weight decay's range includes pathological values that crash training, fANOVA may attribute the variance to weight decay even though the practical default range is benign.
- **Confounding with another hyperparameter**: e.g., learning rate co-varies; the "importance" of weight decay reflects the indirect effect of overlapping configurations.

Always combine fANOVA with manual sensitivity sweeps for sanity checks.

**D6. You ran 1000 random search trials on a 30-dim space. The best is no better than the median. What went wrong?**

In high dimensions, random search's coverage degrades drastically. With 30 dimensions and 1000 trials you cover essentially nothing. Switch to:
- **Bayesian optimization** with a model that handles high dim (e.g., REMBO, ALEBO, or trust-region BO like TURBO).
- **Dimensionality reduction**: identify the few important dimensions by variance analysis on a small sweep, then BO on those.
- **Decompose** the problem: find natural subgroups (optimizer-related, regularization-related) and tune them in stages.

**D7. Your Bayesian optimization improves training loss but not validation loss. Why?**

You're tuning the wrong objective. The search has overfit the training metric. Always tune against validation loss (or a proxy of generalization), never training loss. Adding strong regularization ranges into the search space is also recommended.

**D8. You parallelize BO with 16 workers. Performance is much worse than sequential. Why?**

The naïve parallelization picks the same argmax of acquisition for every worker — you evaluate 16 nearly identical points. Solutions:
- **Local penalization**: after picking a point, modify the acquisition to penalize nearby candidates so the next worker's argmax is elsewhere.
- **Kriging Believer / Constant Liar**: pretend the in-flight evaluations returned predicted values, refit GP, then pick the next point.
- **Thompson sampling**: each worker draws an independent posterior sample and minimizes it — naturally parallelizable.
- **Multi-fidelity instead**: parallelism is much more natural in Hyperband / BOHB / PBT.

**D9. After moving to log-scale sampling, your tuning got worse. Why?**

Possible causes:
- The optimum was actually at the upper end of the linear range; log-scale concentrates samples near the lower end.
- The hyperparameter is bounded (e.g., dropout ∈ [0, 0.5]) and not multiplicative — log-scale doesn't help.
- The search space's bounds aren't aligned with the realistic operating range.

Use log-scale only for hyperparameters that span orders of magnitude (learning rates, regularization coefficients). Do not log-scale fractions, probabilities, integer counts (unless they span orders of magnitude), or angles.

**D10. Two HPO runs with the same method find different "best" configurations. Which to trust?**

Both, partially: this is a sign of multi-modal landscape, high noise, or insufficient budget. Mitigations:
- Run K seeds; report the distribution of best scores, not a single number.
- Take the top-N configurations from each seed and ensemble or compare them on a fresh evaluation.
- Increase budget if the gap between seeds is large relative to the gap between best and median configurations.

---

## E. Probing & Follow-up Questions

**E1. Suppose you replace the GP in BO with a neural network ensemble. What changes?**

You gain scalability (no O(n³) inversion) and possibly handle high dimensions better. You lose principled uncertainty estimates — neural network uncertainty is notoriously miscalibrated. Some methods (Deep Ensembles, MC Dropout, BNNs, Bayesian Last Layer) try to recover useful uncertainty. The acquisition function still works in form (EI, UCB), but exploration may be unreliable.

**E2. Why does BOHB outperform plain BO on most benchmarks?**

Two reasons:
1. BOHB's any-time performance: thanks to Hyperband's brackets, BOHB returns a passable configuration after a small budget, whereas vanilla BO needs warmup before its surrogate is informative.
2. Multi-fidelity: BOHB extracts signal from cheap evaluations to filter bad configurations before paying for full-fidelity training. Vanilla BO must always pay full-fidelity cost.

**E3. How would you design HPO for a setting where each evaluation is *interactive* (e.g., user A/B tests)?**

This shifts from black-box optimization to **bandit / online learning**. Use:
- **Multi-armed bandits** (Thompson Sampling, UCB1) over a discrete set of candidate configurations.
- **Contextual bandits** if user features are available.
- **Bayesian optimization with batch acquisition** if many users can be exposed simultaneously.
- Importantly, weigh **regret cost** (suboptimal configurations harm real users) explicitly. Not all HPO methods minimize cumulative regret; many minimize *simple regret* (just the final answer's quality).

**E4. What happens to BO if the objective is non-stationary in λ (e.g., the loss surface has both very smooth and very rugged regions)?**

A stationary kernel (constant length-scale across Λ) will be misspecified: it either over-smooths the rugged region or under-smooths the flat region. Solutions:
- **Non-stationary kernels**: input warping (Snoek et al. 2014) maps λ through a learnable monotonic transformation before applying the RBF kernel.
- **Tree-based surrogates** (random forest in SMAC): naturally handle non-stationarity through partitioning.
- **Local BO** (TURBO): maintain trust regions and refit local GPs.

**E5. Could you tune hyperparameters with reinforcement learning?**

Yes. Each HPO step becomes an action; reward is the resulting performance. Approaches:
- **Neural architecture search** (Zoph & Le, 2017): RNN controller proposes architectures; reward = validation accuracy.
- **REINFORCE-based controller** for hyperparameter sequences.
- **PBT** can be viewed as evolutionary RL.

In practice RL-based HPO is sample-inefficient compared to BO/Hyperband for fixed search spaces. It is more attractive when the action space is huge and structured (like NAS).

**E6. How does dropout interact with HPO of regularization?**

Dropout is itself a regularizer with hyperparameter `p`. If you also tune weight decay, batch normalization momentum, and label smoothing, you have multiple correlated regularizers. fANOVA often shows strong interactions among them. Practical advice:
- Don't tune all regularizers from scratch; use sensible defaults for some and tune one or two.
- If tuning all, expect strong interactions and use a method (TPE, RF surrogates) that captures them.

**E7. Explain "asynchronous" Successive Halving (ASHA).**

In synchronous SH, each rung waits for all configurations to finish before promoting survivors. With many parallel workers, the slowest config blocks the entire rung. ASHA promotes a configuration to the next rung *as soon as it ranks in the top 1/η among the configurations currently completed at its rung* — no synchronization needed. ASHA is the default in Ray Tune and modern parallel HPO.

**E8. Multi-objective HPO: how would you handle accuracy vs. latency?**

You no longer have a scalar objective; you have a Pareto front. Methods:
- **Scalarization**: combine objectives into one weighted sum (requires choosing weights).
- **Constrained optimization**: maximize accuracy subject to latency ≤ L.
- **Pareto-aware acquisition** (qEHVI in BoTorch, NSGA-II for evolutionary): explicitly explore the Pareto frontier.
- **Multi-objective TPE** (Optuna's NSGA-II / TPE-MO).

Report the Pareto front; let the deployment team pick the trade-off.

**E9. Compare fANOVA to ablation studies.**

fANOVA decomposes variance from a surrogate fit on existing HPO trajectories — it requires no extra evaluations and yields contributions from interactions. But it depends on surrogate quality and assumes the search space is the relevant universe.

Ablation studies hold all but one hyperparameter at the best configuration and sweep the remaining one. They give a clean local sensitivity but miss interactions and depend heavily on the chosen "best" configuration.

Use fANOVA for global insight; use ablations to verify specific claims.

**E10. Suppose you must defend the choice of HPO method in a paper. What are the three most important reporting items?**

1. **Search space**: ranges, scales (log/linear), categorical values, conditional structure. Without this, results are not reproducible.
2. **Budget**: number of trials, total compute, parallelism. Without this, comparisons are unfair.
3. **Variance**: re-run the entire HPO procedure with multiple seeds and report the distribution of final scores. A single best number is meaningless; the reproducibility of the *procedure* is what matters.

Bonus: also report the *fANOVA importance* if possible — it shows which hyperparameters mattered, helping readers understand what your method is sensitive to.

**E11. Why does noise hurt UCB more than EI?**

UCB explicitly subtracts κ·σ from μ. With high noise, σ stays large even after many evaluations of the same point, so UCB keeps recommending exploration of "uncertain" regions that are not actually uncertain — they're just noisy. EI also degrades with noise, but its formulation with f\* (current best) at least has a self-correcting term: as f\* drifts due to noise, the improvement target moves accordingly. Noisy variants of both are necessary for noisy objectives.

**E12. Final probing question — when does HPO not help at all?**

- When the model is fundamentally misspecified for the task (no hyperparameter setting will solve it).
- When the bottleneck is data quality, not the model.
- When evaluation noise exceeds plausible improvements.
- When the default hyperparameters are already near-optimal for your task (e.g., well-tuned defaults in modern libraries for standard datasets).
- When the cost of tuning exceeds the value of the marginal accuracy gained.

A good practitioner asks: "Is the wall I'm hitting actually a hyperparameter wall?" before spending GPU-weeks on HPO.

---

*End of guide.*
