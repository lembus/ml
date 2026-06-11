# Estimation Theory: A Comprehensive Learning Guide

> A standalone reference covering MLE, MAP, Bayesian Inference, and the Method of Moments, written for someone with no prior exposure. Each topic follows the same six-section structure, followed by an interview preparation section.

---

## Table of Contents

1. [Maximum Likelihood Estimation (MLE)](#1-maximum-likelihood-estimation-mle)
2. [Maximum A Posteriori (MAP)](#2-maximum-a-posteriori-map)
3. [Bayesian Inference](#3-bayesian-inference)
4. [Method of Moments](#4-method-of-moments)
5. [Interview Preparation](#5-interview-preparation)

---

# 1. Maximum Likelihood Estimation (MLE)

## 1.1 Motivation & Intuition

Imagine you flip a coin 10 times and see 7 heads. You suspect the coin may be biased. What is your best guess for the probability `p` that the coin lands heads? Intuitively, you'd say `7/10 = 0.7`. Why? Because among all possible values of `p`, the value `0.7` makes the data you actually observed *most plausible*. That is the entire idea of Maximum Likelihood Estimation: **choose the parameter values that make the observed data look as likely as possible.**

This is the workhorse of statistical estimation and the conceptual foundation of nearly every standard ML training procedure. When you train a logistic regression, a neural network with cross-entropy loss, a Gaussian mixture model via EM, or a linear regression with squared error — you are (often without realizing it) doing MLE.

**Real-world ML connections:**
- **Spam classifier**: Given labeled emails, find the parameters of a logistic regression that make the observed labels most probable.
- **Language models**: Find network weights that maximize the probability of the next token in a corpus.
- **Recommendation systems**: Estimate user/item embeddings that maximize the likelihood of observed ratings.

The deep insight is that "fitting a model to data" can be made mathematically precise by interpreting the model as a probability distribution and asking: *which setting of the parameters assigns the highest probability mass/density to what I actually saw?*

## 1.2 Conceptual Foundations

**Key terms:**
- **Statistical model**: A family of probability distributions `{p(x | θ) : θ ∈ Θ}` indexed by a parameter `θ`. The set `Θ` is the parameter space.
- **Sample / data**: Observations `x₁, …, xₙ`, typically assumed independent and identically distributed (i.i.d.) draws from some unknown member of the family.
- **Likelihood function**: Viewed as a function of `θ` (with the data held fixed), `L(θ) = p(x₁, …, xₙ | θ)`. Crucially, the likelihood is **not** a probability distribution over `θ`.
- **Log-likelihood**: `ℓ(θ) = log L(θ)`. We use logs because (a) products become sums, which are numerically stable and easier to differentiate, and (b) log is monotonic so the maximizer is unchanged.
- **MLE**: `θ̂_MLE = argmax_θ ℓ(θ)`.

**Step-by-step interaction:**
1. Choose a model family.
2. Write the joint probability of the observed data as a function of `θ`.
3. Take the log.
4. Differentiate with respect to `θ`, set to zero, solve (or use numerical optimization if no closed form).
5. Verify it is a maximum (second-order condition).

**Underlying assumptions:**
- The data are drawn from *some* member of the assumed model family (well-specified model).
- Observations are i.i.d. (or at least the joint distribution is correctly specified).
- The parameter space is identifiable: distinct `θ` produce distinct distributions.
- Sufficient regularity: differentiability, interior optima, finite Fisher information.

**What breaks when these assumptions fail:**
- **Misspecification**: MLE converges to the parameter that minimizes KL divergence to the true distribution (the "pseudo-true" parameter), which may not be meaningful.
- **Non-i.i.d. data** (e.g., time series, network data): the likelihood factorization is wrong; estimates can be biased and standard errors invalid.
- **Non-identifiability** (e.g., label-switching in mixture models, redundant parameters in over-parameterized neural nets): multiple distinct `θ` give the same likelihood; MLE is not unique.
- **Boundary or singular optima**: regularity conditions for asymptotic normality fail (e.g., variance parameter on the boundary of zero).

## 1.3 Mathematical Formulation

For i.i.d. data `x₁, …, xₙ ~ p(x | θ)`,

```
L(θ) = ∏ᵢ₌₁ⁿ p(xᵢ | θ),     ℓ(θ) = ∑ᵢ₌₁ⁿ log p(xᵢ | θ).
```

The MLE is `θ̂ = argmax_θ ℓ(θ)`. The first-order condition is the **score equation**:

```
∇_θ ℓ(θ) = ∑ᵢ ∇_θ log p(xᵢ | θ) = 0.
```

The score function `s(x; θ) = ∇_θ log p(x | θ)` has the fundamental property `E[s(X; θ)] = 0` under the true distribution. Its variance is the **Fisher Information**:

```
I(θ) = E[ s(X; θ) s(X; θ)ᵀ ] = -E[ ∇²_θ log p(X | θ) ].
```

The two equalities (information identity) hold under regularity.

### Worked derivation: Gaussian MLE

Let `x₁, …, xₙ ~ 𝒩(μ, σ²)`. Then

```
ℓ(μ, σ²) = -(n/2) log(2π) - (n/2) log σ² - (1/(2σ²)) ∑ᵢ (xᵢ - μ)².
```

Setting `∂ℓ/∂μ = 0`:  `(1/σ²) ∑ᵢ (xᵢ - μ) = 0  ⇒  μ̂ = (1/n) ∑ᵢ xᵢ` (sample mean).

Setting `∂ℓ/∂σ² = 0`:  `-(n/(2σ²)) + (1/(2σ⁴)) ∑ᵢ (xᵢ - μ̂)² = 0  ⇒  σ̂² = (1/n) ∑ᵢ (xᵢ - μ̂)²`.

Note: `σ̂²_MLE` divides by `n`, not `n-1`. It is **biased downward**. The unbiased estimator divides by `n-1` (Bessel's correction). MLE need not be unbiased.

### Asymptotic Properties (the three pillars)

Under regularity, as `n → ∞`:

1. **Consistency**: `θ̂ₙ →ᵖ θ*` (the true parameter).
2. **Asymptotic normality**: `√n (θ̂ₙ - θ*) →ᵈ 𝒩(0, I(θ*)⁻¹)`.
3. **Efficiency**: The asymptotic variance `I(θ*)⁻¹` matches the **Cramér–Rao lower bound** — no unbiased estimator can do better asymptotically.

**Sketch of asymptotic normality.** Taylor-expand the score around `θ*`:
```
0 = ∇ℓ(θ̂) ≈ ∇ℓ(θ*) + ∇²ℓ(θ*)(θ̂ - θ*).
```
Rearranging and dividing by `√n`:
```
√n(θ̂ - θ*) ≈ -[ (1/n) ∇²ℓ(θ*) ]⁻¹ · (1/√n) ∇ℓ(θ*).
```
By the Law of Large Numbers, `(1/n)∇²ℓ(θ*) → -I(θ*)`. By the Central Limit Theorem, `(1/√n)∇ℓ(θ*) → 𝒩(0, I(θ*))`. Combining via Slutsky: `√n(θ̂-θ*) → 𝒩(0, I(θ*)⁻¹)`.

### Connection to cross-entropy loss

For a classification model `p(y | x; θ)` with one-hot labels, the negative log-likelihood for a single example is

```
-log p(y | x; θ) = -∑_k 𝟙[y=k] log p(y=k | x; θ),
```

which is exactly the **cross-entropy** between the empirical label distribution and the model. Minimizing cross-entropy = maximizing likelihood. Similarly, squared loss for regression corresponds to MLE under a Gaussian noise model with fixed variance.

## 1.4 Worked Example: Bernoulli MLE

You observe `n = 10` coin flips: `x = (1,1,0,1,1,1,0,1,1,1)`, so `k = 8` heads. The model is `xᵢ ~ Bernoulli(p)`.

```
L(p) = pᵏ (1-p)^(n-k) = p⁸ (1-p)².
ℓ(p) = 8 log p + 2 log(1-p).
ℓ'(p) = 8/p - 2/(1-p) = 0   ⇒   8(1-p) = 2p   ⇒   p̂ = 8/10 = 0.8.
```

Fisher information: `I(p) = 1/(p(1-p))`. Asymptotic standard error: `√(p̂(1-p̂)/n) = √(0.16/10) ≈ 0.126`. So a 95% CI is roughly `0.8 ± 0.25`. With only 10 data points the uncertainty is large — a fact MLE makes explicit through `I⁻¹`.

## 1.5 Relevance to ML Practice

- **Training**: Cross-entropy and squared losses are MLE under categorical/Gaussian likelihoods. Gradient-based optimizers like SGD are doing approximate MLE.
- **Evaluation**: Held-out log-likelihood (perplexity in language models) is a direct estimator of generalization in the MLE framework.
- **When to use**: Large data, well-specified or near-correct models, when point estimates suffice.
- **When not to use**: Small data (high variance, overfitting), heavy misspecification, when uncertainty matters (use Bayesian instead), when there are identifiability issues.
- **Trade-offs**: MLE is asymptotically efficient but can severely overfit in small samples; offers no built-in regularization; produces point estimates only (no uncertainty without extra work like the Hessian or bootstrap).

## 1.6 Common Pitfalls

- **Confusing likelihood with probability over `θ`**: `L(θ)` does not integrate to 1 over `θ` and is not a posterior.
- **Forgetting that MLE can be biased** (e.g., Gaussian variance, AR(1) coefficient).
- **Treating asymptotic CIs as valid for small `n`** — they rely on the CLT.
- **Numerical underflow** from multiplying tiny probabilities — always use log-likelihood.
- **Local optima** in non-convex problems (neural nets, mixture models) — MLE optimization is not the same as the *global* MLE.
- **Ignoring identifiability** — symmetric mixture models, ICA without sign constraints, etc.

---

# 2. Maximum A Posteriori (MAP)

## 2.1 Motivation & Intuition

MLE has a serious weakness: with little data, it can produce wild estimates. If you flip a coin once and see heads, MLE says `p̂ = 1` — claiming the coin will *never* land tails. That's absurd because we have prior knowledge that coins are usually fair.

MAP estimation fixes this by injecting prior beliefs. We treat `θ` as a random variable with a **prior distribution** `p(θ)` encoding what we believe before seeing data, and after observing data we form the **posterior** `p(θ | x) ∝ p(x | θ) p(θ)`. The MAP estimate is the posterior mode:

```
θ̂_MAP = argmax_θ p(θ | x) = argmax_θ [log p(x | θ) + log p(θ)].
```

It is MLE plus a penalty term that pulls estimates toward the prior. This is the bridge between Bayesian thinking and **regularization** in ML: L2 regularization is MAP under a Gaussian prior, L1 regularization is MAP under a Laplace prior.

## 2.2 Conceptual Foundations

- **Prior `p(θ)`**: Your belief about `θ` before seeing data. Can encode domain knowledge, smoothness, sparsity, scale.
- **Likelihood `p(x | θ)`**: Same as in MLE.
- **Posterior `p(θ | x)`**: What you believe after seeing data. Bayes' rule: `p(θ | x) = p(x | θ) p(θ) / p(x)`. The denominator (the "evidence") is constant in `θ` and irrelevant for finding the mode.
- **MAP estimate**: The mode (argmax) of the posterior.

**Assumptions:**
- The prior is a meaningful representation of beliefs (or at least a reasonable inductive bias).
- The model is identifiable enough that the posterior is well-defined.
- For the regularization interpretation, the prior must be proper (integrable) — improper uniform priors recover MLE.

**What breaks:**
- A bad prior (overly tight, miscentered) can dominate the data and bias estimates indefinitely.
- MAP is **not invariant under reparameterization** — the mode of `p(θ | x)` is not the mode of `p(g(θ) | x)` for nonlinear `g`. The posterior mean is also not invariant, but the *full posterior* is. This is a genuine philosophical wart of MAP.
- In high dimensions, the mode can be unrepresentative of the bulk of probability mass (the "typical set" lies far from the mode).

## 2.3 Mathematical Formulation

```
θ̂_MAP = argmax_θ [ log p(x | θ) + log p(θ) ].
```

### L2 → Gaussian prior

Linear regression with `y = Xθ + ε`, `ε ~ 𝒩(0, σ²I)`, and prior `θ ~ 𝒩(0, τ²I)`:

```
log p(θ | y, X) = -(1/(2σ²)) ‖y - Xθ‖² - (1/(2τ²)) ‖θ‖² + const.
```

Maximizing equals minimizing

```
‖y - Xθ‖² + (σ²/τ²) ‖θ‖²,
```

which is **ridge regression** with `λ = σ²/τ²`. A tighter prior (smaller `τ²`) means stronger regularization. The closed-form solution is `θ̂ = (XᵀX + λI)⁻¹ Xᵀy`.

### L1 → Laplace prior

If instead `θⱼ ~ Laplace(0, b)` independently, then `log p(θ) = -‖θ‖₁ / b + const`. Maximizing the posterior gives

```
‖y - Xθ‖² + λ ‖θ‖₁,
```

which is **Lasso**. The Laplace prior has a sharp peak at zero, encouraging sparse solutions.

### General principle

Any regularizer `R(θ)` corresponds to an (improper or proper) prior `p(θ) ∝ exp(-R(θ))`. Conversely, every prior induces a regularizer. This is why MAP is sometimes called "penalized likelihood".

## 2.4 Worked Example: Beta-Bernoulli MAP

Same coin example: 1 flip, 1 head. With a `Beta(α, β)` prior on `p`,

```
p(p | x) ∝ p^k (1-p)^(n-k) · p^(α-1) (1-p)^(β-1) = p^(k+α-1) (1-p)^(n-k+β-1).
```

This is `Beta(k+α, n-k+β)`. The mode of `Beta(a, b)` (for `a, b > 1`) is `(a-1)/(a+b-2)`. So

```
p̂_MAP = (k + α - 1) / (n + α + β - 2).
```

With a `Beta(2, 2)` prior (mildly favoring fair coins), `k=1`, `n=1`:

```
p̂_MAP = (1 + 2 - 1)/(1 + 2 + 2 - 2) = 2/3 ≈ 0.667.
```

Compare to `p̂_MLE = 1`. The prior pulled us back from the absurd extreme. As `n → ∞`, the prior's influence fades and MAP → MLE.

## 2.5 Relevance to ML Practice

- **Regularization** in essentially every supervised learning algorithm is a MAP interpretation.
- **Weight decay** in neural networks ↔ Gaussian prior on weights.
- **Sparse models** (Lasso, sparse coding) ↔ Laplace or spike-and-slab priors.
- **Structured priors**: Total variation, group sparsity, graph Laplacians — all MAP regularizers.
- **When to use**: Small to moderate data, when domain knowledge can be encoded as a prior, when you want a single point estimate but with regularization.
- **When not to use**: When you need uncertainty quantification (use full Bayesian inference), when reparameterization invariance matters, in very high dimensions where the mode is misleading.

**Trade-offs**: MAP gives a point estimate (cheap, interpretable) but throws away uncertainty information. The choice of prior introduces bias but reduces variance — classic bias–variance trade-off.

## 2.6 Common Pitfalls

- **Forgetting MAP is not Bayesian inference** — it is just penalized MLE that ignores posterior uncertainty.
- **Reparameterization sensitivity**: MAP for `σ` differs from MAP for `log σ`.
- **Improper priors** can yield improper posteriors — always check.
- **Picking priors for computational convenience** (e.g., always Gaussian) when the actual beliefs are very different.
- **Confusing MAP with posterior mean**: For asymmetric posteriors they differ; the mean is often more informative.

---

# 3. Bayesian Inference

## 3.1 Motivation & Intuition

MLE and MAP both produce single numbers. But a single number hides what we don't know. Suppose two doctors estimate the probability that a patient has a disease at 0.6. One is 95% confident the truth is between 0.55 and 0.65; the other has no idea — it could be 0.1 or 0.9. The point estimate is the same; the implications are completely different.

Bayesian inference reports the full **posterior distribution** `p(θ | x)`, not just its peak. From the posterior we can compute:
- Point estimates (mean, median, mode).
- Credible intervals: "there is a 95% probability that `θ ∈ [a, b]` given the data".
- Predictive distributions for new data, integrating over parameter uncertainty.
- Decision-theoretic outputs (expected loss, expected utility).

Bayesian methods shine when data is scarce, stakes are high, and uncertainty matters: medical diagnosis, A/B testing, autonomous systems, scientific inference.

## 3.2 Conceptual Foundations

**Bayes' Rule:**

```
p(θ | x) = p(x | θ) p(θ) / p(x),     where p(x) = ∫ p(x | θ) p(θ) dθ.
```

- **Posterior**: distribution over `θ` after seeing data.
- **Prior**: distribution before.
- **Likelihood**: data-generating model.
- **Evidence / marginal likelihood `p(x)`**: a normalizing constant; also crucial for **model comparison**.

**Posterior predictive distribution** for new data `x*`:

```
p(x* | x) = ∫ p(x* | θ) p(θ | x) dθ.
```

This integrates over our remaining uncertainty about `θ` rather than plugging in a single value.

**Conjugate priors**: A prior is *conjugate* to a likelihood if the posterior is in the same family as the prior. This makes computation analytic.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli / Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Gaussian (mean, known var) | Gaussian | Gaussian |
| Gaussian (mean+var) | Normal-Inverse-Gamma | NIG |
| Multinomial | Dirichlet | Dirichlet |
| Exponential | Gamma | Gamma |

**Assumptions:**
- A meaningful prior is available.
- The likelihood is correctly specified (or robust to misspecification).
- Computations (integrals) are tractable, or approximated faithfully.

**What breaks:**
- Bad prior → biased posterior even with much data (slow mixing rates).
- Intractable integrals → must approximate via MCMC or variational methods, with their own failure modes.
- Model misspecification → posterior concentrates on the wrong place but reports false certainty.

## 3.3 Mathematical Formulation

### Conjugate example: Beta-Binomial

Prior: `p ~ Beta(α, β)`. Likelihood: `k ~ Binomial(n, p)`.

```
p(p | k) ∝ p^(α-1)(1-p)^(β-1) · p^k (1-p)^(n-k) = p^(α+k-1)(1-p)^(β+n-k-1),
```

so `p | k ~ Beta(α + k, β + n - k)`. The posterior mean is `(α + k)/(α + β + n)` — a weighted average of the prior mean `α/(α+β)` and the MLE `k/n`, with weight controlled by `α+β` (the "pseudo-count" of the prior).

### When conjugacy fails: approximate inference

For most modern ML models (deep nets, hierarchical models, complex likelihoods), the posterior is intractable. Two main families of approximations:

**1. Markov Chain Monte Carlo (MCMC).** Construct a Markov chain whose stationary distribution is `p(θ | x)`, then draw samples.

- **Metropolis–Hastings**: Propose `θ' ~ q(θ' | θ)`, accept with probability `min(1, [p(θ'|x)q(θ|θ')] / [p(θ|x)q(θ'|θ)])`. The unknown normalizer cancels.
- **Gibbs sampling**: Cycle through coordinates, sampling each from its full conditional. Useful when conditionals are tractable.
- **Hamiltonian Monte Carlo (HMC)**: Use gradient information and a physical analogy (Hamiltonian dynamics) to make large, informed moves. Used in Stan, PyMC.

MCMC is asymptotically exact but can be slow and hard to diagnose (mixing, burn-in, autocorrelation).

**2. Variational Inference (VI).** Pick a tractable family `q_φ(θ)` and optimize `φ` to minimize `KL(q_φ ‖ p(·|x))`. Equivalently, maximize the **Evidence Lower Bound (ELBO)**:

```
ELBO(φ) = E_{q_φ}[log p(x, θ)] - E_{q_φ}[log q_φ(θ)] = log p(x) - KL(q_φ ‖ p(·|x)).
```

Since `log p(x)` is constant in `φ`, maximizing ELBO minimizes the KL. VI converts inference into optimization, making it scalable (e.g., stochastic VI, amortized VI as in VAEs). The price: bias from the limited variational family, often underestimating posterior variance because reverse-KL is mode-seeking.

**3. Laplace approximation.** Fit a Gaussian centered at the MAP using the inverse Hessian as covariance. Cheap; accurate when posterior is approximately Gaussian.

### Bayesian Model Averaging vs. Selection

Given several candidate models `{M₁, …, M_K}`:

- **Selection**: Pick the single best model, e.g., the one with highest marginal likelihood `p(x | M_k)`, BIC, or cross-validation score. Throws away uncertainty about which model is correct.
- **Averaging (BMA)**: Form a mixture weighted by posterior model probabilities:

```
p(x* | x) = ∑_k p(x* | x, M_k) p(M_k | x),    p(M_k | x) ∝ p(x | M_k) p(M_k).
```

BMA accounts for model uncertainty and often improves predictive performance, but it requires computing `p(x | M_k)` (a hard integral) for each model. Marginal likelihood penalizes complexity automatically through the "Occam factor" — overly flexible models spread prior mass over many bad parameter settings, lowering `p(x | M_k)`.

## 3.4 Worked Example: Beta-Binomial Inference

Suppose you run an A/B test. Variant A: 30 conversions out of 200. Prior `Beta(1,1)` (uniform).

Posterior: `Beta(1 + 30, 1 + 170) = Beta(31, 171)`.

- Posterior mean: `31/202 ≈ 0.1535`.
- Posterior std: `√(ab/((a+b)²(a+b+1))) ≈ 0.0254`.
- 95% credible interval (from Beta quantiles): roughly `[0.107, 0.207]`.

Now Variant B: 45/200, posterior `Beta(46, 156)`, mean ≈ `0.228`. Probability that B is better than A:

```
P(p_B > p_A | data) = ∫∫ 𝟙[p_B > p_A] · Beta(p_A; 31,171) · Beta(p_B; 46,156) dp_A dp_B,
```

approximated by Monte Carlo: draw 100,000 samples from each posterior, count fraction where `p_B > p_A`. Likely ≈ 0.99. This directly answers the business question, in contrast to a frequentist p-value.

## 3.5 Relevance to ML Practice

- **Bayesian deep learning**: weight uncertainty (MC dropout, variational Bayes by backprop, deep ensembles as approximate Bayes).
- **Bayesian optimization**: surrogate posteriors over an expensive function (e.g., hyperparameter tuning).
- **Probabilistic programming**: Stan, PyMC, NumPyro, Pyro for general-purpose Bayesian modeling.
- **Calibration & uncertainty**: critical in safety-sensitive applications (medical, autonomous driving).
- **Online learning / bandits**: Thompson sampling = sample from posterior over arm rewards; provably near-optimal for many bandit settings.
- **A/B testing**: Bayesian decision-theoretic stopping rules avoid p-hacking issues.

**When not to use**: massive data with tractable likelihoods where MLE works fine; latency-critical inference where MCMC is too slow; situations where no meaningful prior exists and a flat prior conveys nothing.

**Trade-offs**: principled uncertainty and prior incorporation vs. computational cost and prior specification effort.

## 3.6 Common Pitfalls

- **Treating the posterior as exact when using approximations** — VI underestimates variance, MCMC may not have converged.
- **Improper posterior**: e.g., flat priors on scale parameters can give non-integrable posteriors. Always check.
- **Ignoring convergence diagnostics**: R-hat, effective sample size, trace plots.
- **Comparing models with different priors via marginal likelihood** — marginal likelihood is highly prior-sensitive.
- **Confusing credible and confidence intervals**: a 95% credible interval is `P(θ ∈ I | data) = 0.95`; a confidence interval is a frequentist coverage statement about the procedure.
- **Posterior collapse** in VAEs: the variational posterior matches the prior, ignoring the data.

---

# 4. Method of Moments

## 4.1 Motivation & Intuition

Long before MLE, statisticians had a simpler trick: if a distribution has parameters that determine its mean, variance, etc., then estimate those moments from the data and solve for the parameters. This is the **Method of Moments (MoM)**, introduced by Karl Pearson in 1894.

**Intuition**: A `Gamma(α, β)` distribution has mean `α/β` and variance `α/β²`. Compute the sample mean and variance, set them equal to these expressions, and solve for `α` and `β`. No likelihood maximization needed.

MoM is often simpler than MLE, requires no optimization, and serves as an excellent **initialization** for iterative methods like EM or gradient descent.

## 4.2 Conceptual Foundations

- **Population moment**: `μ_k(θ) = E_θ[X^k]`.
- **Sample moment**: `m_k = (1/n) ∑ᵢ xᵢ^k`.
- **MoM principle**: If there are `p` parameters, equate the first `p` population moments to the first `p` sample moments and solve.

**Assumptions:**
- The required population moments exist (finite).
- The system of equations has a unique solution.
- Sample moments are good estimates (LLN — generally true for i.i.d. data with finite moments).

**What breaks:**
- Heavy-tailed distributions (e.g., Cauchy) have no mean — MoM is undefined.
- Higher moments have very high variance, especially with small `n`, making the estimator noisy.
- The system may have no solution, multiple solutions, or solutions outside the valid parameter space.
- Generally **less efficient** than MLE: larger asymptotic variance.

## 4.3 Mathematical Formulation

For a `p`-parameter model, solve

```
m_k = μ_k(θ),    k = 1, …, p,
```

for `θ`. This is a system of `p` equations in `p` unknowns.

**Generalized Method of Moments (GMM)** allows more moment conditions than parameters, weighting them by an estimated covariance matrix; widely used in econometrics.

**Asymptotic properties**: Under regularity, MoM estimators are consistent and asymptotically normal, with asymptotic variance derivable via the delta method. They are *not* asymptotically efficient in general — MLE attains a smaller asymptotic variance.

## 4.4 Worked Example: Gamma Distribution

`X ~ Gamma(α, β)` with mean `μ = α/β` and variance `σ² = α/β²`. Given data with sample mean `x̄` and sample variance `s²`:

```
x̄ = α̂ / β̂      ⇒   α̂ = x̄ · β̂,
s² = α̂ / β̂²    ⇒   s² = x̄ / β̂   ⇒   β̂ = x̄ / s²,
α̂ = x̄² / s².
```

Concrete: `x̄ = 4`, `s² = 2` ⇒ `β̂ = 2`, `α̂ = 8`. Done — no optimization required. Compare to Gamma MLE which has no closed form and requires Newton's method on the digamma function.

## 4.5 Relevance to ML Practice

- **Initializer for EM** in mixture models: initial cluster means/variances from MoM-style heuristics.
- **Spectral methods / tensor decomposition**: Provably correct learning of latent variable models (HMMs, topic models, mixtures) by matching low-order moments — a modern use of MoM that bypasses local optima of MLE/EM.
- **GMM in econometrics**: standard tool for instrumental variable estimation.
- **Quick estimators** when you need a fast, closed-form answer.

**When to use**: simple distributions, when MLE is intractable, as initialization, when you need a baseline.

**When not to use**: heavy tails, very small samples (high-moment estimators are noisy), when efficiency matters and MLE is feasible.

**Trade-offs**: simplicity and robustness to optimization issues vs. statistical inefficiency.

## 4.6 Common Pitfalls

- **Using MoM on heavy-tailed data**: estimators of mean/variance are unreliable.
- **Negative variance estimates**: MoM can return values outside the parameter space (e.g., negative variance components in random-effects models).
- **Treating MoM as efficient**: it usually is not.
- **Forgetting that higher-order sample moments are very noisy** — using `m_4` for a 4-parameter model is statistically risky.

---

# 5. Interview Preparation

## 5.1 Foundational Questions

**Q1. What is the difference between likelihood and probability?**
A probability is a function of data with parameters fixed: `p(x | θ)` integrates to 1 over `x`. A likelihood is the same expression viewed as a function of `θ` with `x` fixed: `L(θ) = p(x | θ)`. The likelihood does *not* integrate to 1 over `θ` and is not a probability distribution over parameters.

**Q2. Why do we use the log-likelihood instead of the likelihood?**
(1) Numerical stability — products of small probabilities underflow; sums of logs do not. (2) Mathematical convenience — derivatives of sums are easier than derivatives of products. (3) Monotonicity — `log` is strictly increasing, so the maximizer is identical.

**Q3. How is MAP related to MLE?**
MAP adds a log-prior term: `θ̂_MAP = argmax [log p(x|θ) + log p(θ)]`. With a uniform prior, MAP = MLE. As `n → ∞`, the likelihood dominates and MAP → MLE under regularity.

**Q4. What is a conjugate prior and why do we care?**
A prior conjugate to a likelihood produces a posterior in the same family. This gives closed-form posteriors, enabling fast online updates and exact inference. Examples: Beta-Bernoulli, Gaussian-Gaussian, Dirichlet-Multinomial.

**Q5. Why does L2 regularization correspond to a Gaussian prior?**
A `𝒩(0, τ²I)` prior contributes `-‖θ‖²/(2τ²)` to the log-posterior, which is exactly an L2 penalty with strength `1/(2τ²)`. Maximizing the log-posterior = minimizing squared error + L2 penalty.

## 5.2 Mathematical Questions

**Q6. Derive the MLE for a Gaussian.**
(See section 1.3.) `μ̂ = x̄`, `σ̂² = (1/n) ∑(xᵢ - x̄)²`. Note `σ̂²` is biased; the unbiased version uses `n-1`.

**Q7. State and sketch the proof of asymptotic normality of MLE.**
`√n(θ̂ - θ*) →ᵈ 𝒩(0, I(θ*)⁻¹)`. Sketch: Taylor expand the score around `θ*`, use LLN on the Hessian, CLT on the score, and Slutsky to combine. (Section 1.3.)

**Q8. What is the Cramér–Rao lower bound?**
For an unbiased estimator `θ̂`, `Var(θ̂) ≥ I(θ)⁻¹`. MLE asymptotically attains this bound, so it is asymptotically efficient.

**Q9. Show that MAP is not invariant under reparameterization.**
If `φ = g(θ)` is a monotonic nonlinear transform, the change-of-variables formula gives `p(φ | x) = p(g⁻¹(φ) | x) |dg⁻¹/dφ|`. The Jacobian shifts the mode, so `argmax_φ p(φ|x) ≠ g(argmax_θ p(θ|x))`. The full posterior is invariant; the mode is not. The posterior mean is also non-invariant, but the median is invariant for 1-D monotonic transforms.

**Q10. Derive the Beta-Binomial posterior.**
(Section 3.3.) Posterior is `Beta(α + k, β + n - k)`. Posterior mean is a convex combination of prior mean and MLE.

**Q11. Derive the ELBO and explain why maximizing it equals minimizing KL to the posterior.**
`log p(x) = E_q[log p(x,θ) - log q(θ)] + KL(q ‖ p(·|x))`. The first term is the ELBO; since `log p(x)` is constant in `q`, maximizing ELBO minimizes KL.

## 5.3 Applied Questions

**Q12. You are training a neural net classifier. What estimation principle are you implicitly using?**
MLE under a categorical likelihood — equivalently, minimizing cross-entropy. Adding L2 weight decay turns it into MAP with a Gaussian prior.

**Q13. You have very little training data. MLE or MAP?**
MAP, or full Bayesian inference. With small `n`, MLE has high variance and overfits; a sensible prior (or regularizer) reduces variance at the cost of some bias. If uncertainty quantification matters, go full Bayesian.

**Q14. When would you choose MCMC over variational inference?**
MCMC when accuracy of the posterior shape is critical and you can afford the compute (small/medium models, scientific inference). VI when scalability matters (large datasets, deep models) and you can tolerate biased — typically variance-underestimating — approximations.

**Q15. How would you do A/B testing the Bayesian way?**
Place priors on conversion rates (e.g., `Beta(1,1)`), update with observed conversions, compute `P(p_B > p_A | data)` via posterior sampling, and stop when this exceeds a threshold or when expected loss is small enough. Avoids the multiple-testing pitfalls of repeated p-value checking.

**Q16. Explain Thompson sampling and why it works.**
For each arm, maintain a posterior over its reward. At each step, sample one parameter from each posterior and pull the arm with the highest sampled value. This naturally balances exploration (uncertain arms occasionally produce high samples) and exploitation (well-known good arms usually produce high samples). It achieves logarithmic regret in the standard multi-armed bandit setting.

## 5.4 Debugging & Failure-Mode Questions

**Q17. Your MLE for a Gaussian variance is essentially zero. What happened?**
Likely a singularity from a single-cluster fit collapsing onto a data point (common in GMM EM). Fix: regularize via a prior (Bayesian/MAP), enforce a minimum variance, or use a different initialization.

**Q18. Your MCMC sampler produces beautiful trace plots but the posterior is wrong. How is that possible?**
Possible causes: (a) bug in the log-posterior, (b) sampler is exploring a local mode and not the global posterior (multimodality), (c) improper posterior, (d) detailed-balance violation. Diagnostics: run multiple chains from different starts, check R-hat across chains, verify with synthetic data where the answer is known.

**Q19. Your variational posterior matches the prior — what is wrong?**
"Posterior collapse," common in VAEs. The KL term dominates and pulls `q` toward the prior, while the decoder learns to ignore the latent. Mitigations: KL annealing, free bits, more expressive decoders that depend on `z`, β-VAE with `β < 1`.

**Q20. MAP estimate differs wildly from posterior mean — what does that tell you?**
The posterior is asymmetric or multimodal. The mode may be in a narrow spike that contains little probability mass. Prefer the mean (or full posterior) for prediction; visualize the posterior to understand the shape.

**Q21. Your model's log-likelihood on training data is high but on test data is much lower. Diagnosis?**
Overfitting. MLE has chosen parameters that exploit noise. Remedies: regularization (= MAP with a prior), more data, simpler model, early stopping, cross-validation for hyperparameters.

## 5.5 Probing Follow-Ups

**Q22. "You said MLE is consistent. Under what assumptions exactly?"**
Identifiability, correct specification, compact parameter space (or interior optimum), continuity of the log-likelihood in `θ`, dominated likelihood (for uniform LLN), and a unique maximum of the expected log-likelihood at the true parameter. Without identifiability, MLE may converge to an equivalence class, not a point.

**Q23. "If MLE is asymptotically efficient, why do we ever use anything else?"**
Because asymptotic efficiency is a *limit* statement. In finite samples MLE can be biased, high-variance, computationally hard, or undefined. MAP/Bayesian methods can outperform in finite samples. Robust estimators trade efficiency for resistance to outliers. MoM is simpler and often used as initialization.

**Q24. "You used a flat prior to be 'objective'. What's wrong with that?"**
Flat priors are not invariant under reparameterization — a flat prior on `θ` is not flat on `log θ`. They can be improper (non-integrable) and lead to improper posteriors. They can also be informative on transformed scales. "Objective" priors like Jeffreys' prior `p(θ) ∝ √det I(θ)` are reparameterization-invariant but have their own pathologies in multivariate settings.

**Q25. "Why might marginal likelihood disagree with cross-validation for model selection?"**
Marginal likelihood penalizes prior-data mismatch and is sensitive to prior choice (the Occam factor depends on prior spread). Cross-validation evaluates predictive performance directly and is prior-robust. They can diverge when the prior is poor or when the model is misspecified. CV is generally preferred when prediction is the goal; marginal likelihood when scientific model comparison under a chosen prior is the goal.

**Q26. "Walk me through what changes when going from MLE to MAP to full Bayesian inference, in one sentence each."**
MLE: pick the `θ` making data most likely. MAP: pick the `θ` making the posterior largest (MLE plus prior penalty). Full Bayesian: keep the entire posterior over `θ` and integrate it into every downstream computation (predictions, decisions, intervals).

---

*End of guide.*
