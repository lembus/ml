# Probability & Distributions: A Complete Learning Guide

A standalone reference for ML practitioners. Each topic follows: Motivation → Concepts → Math → Worked Example → ML Relevance → Pitfalls. An interview section follows the main material.

---

## PART I — BASIC PROBABILITY

### 1. Motivation & Intuition

Probability exists because the world is uncertain and our information is incomplete. When you flip a coin, predict whether an email is spam, or estimate a click-through rate, you cannot give a single deterministic answer — you can only describe how plausible each outcome is. Probability is the calculus of plausibility.

Concrete example: imagine an inbox with 1,000 emails, 200 of them spam. If you pick one at random, "the probability it is spam" just means the long-run fraction: 200/1000 = 0.2. Now suppose you also know the email contains the word "lottery," which appears in 150 of the 200 spam messages but only 10 of the 800 non-spam ones. Intuitively, knowing "lottery is present" should sharply raise your belief that the email is spam. Quantifying that update is the job of **conditional probability** and **Bayes' theorem**.

In ML, every classifier outputs P(class | features); every generative model defines P(data); every Bayesian method updates P(parameters | data). Probability is not a side topic — it is the language ML is written in.

### 2. Conceptual Foundations

- **Sample space (Ω)**: the set of all possible outcomes of a random experiment. For a die: Ω = {1,2,3,4,5,6}.
- **Event (A)**: a subset of Ω. "Roll an even number" = {2,4,6}.
- **Probability measure P**: a function assigning each event a number in [0,1] satisfying the **Kolmogorov axioms**:
  1. P(A) ≥ 0 for every event A.
  2. P(Ω) = 1.
  3. Countable additivity: for disjoint A₁, A₂, …, P(⋃ Aᵢ) = Σ P(Aᵢ).
- **Random variable (X)**: a function from Ω to ℝ. It assigns a number to each outcome.
- **Conditional probability**: P(A | B) = P(A ∩ B) / P(B), defined when P(B) > 0. It rescales probabilities to the universe in which B is known to have occurred.
- **Independence**: A and B are independent iff P(A ∩ B) = P(A)P(B), equivalently P(A | B) = P(A). Knowing B tells you nothing about A.
- **Conditional independence**: A ⊥ B | C iff P(A ∩ B | C) = P(A | C)P(B | C). Crucial in graphical models and Naive Bayes.

Assumptions and what breaks:
- Kolmogorov's axioms assume a well-defined sample space. In practice, model misspecification (the "true" outcome wasn't in your Ω) silently invalidates everything downstream.
- Independence is almost always assumed for tractability (i.i.d. data). When violated (time series, network data, batch effects), variance estimates collapse and confidence intervals become dishonest.
- Conditioning on a zero-probability event is undefined; this matters for continuous variables, where P(X = x) = 0 and you must work with densities.

### 3. Mathematical Formulation

**Chain rule (multiplication rule).** For any events A₁,…,Aₙ:

P(A₁ ∩ A₂ ∩ … ∩ Aₙ) = P(A₁) · P(A₂ | A₁) · P(A₃ | A₁,A₂) · … · P(Aₙ | A₁,…,Aₙ₋₁).

Derivation: apply the definition of conditional probability iteratively. P(A₁ ∩ A₂) = P(A₁)P(A₂|A₁), then multiply by P(A₃|A₁,A₂), etc. The chain rule is what lets autoregressive models (language models!) factor a joint distribution over a sequence into a product of conditionals: P(x₁,…,x_T) = ∏ P(x_t | x_<t).

**Law of total probability.** If {B₁,…,B_k} partitions Ω:

P(A) = Σᵢ P(A | Bᵢ) P(Bᵢ).

**Bayes' theorem.** Combining the definition of conditional probability with the law of total probability:

P(B | A) = P(A | B) P(B) / P(A) = P(A | B) P(B) / Σⱼ P(A | Bⱼ) P(Bⱼ).

The terms have names that are worth memorizing:
- P(B) is the **prior** — what you believed before seeing evidence A.
- P(A | B) is the **likelihood** — how well B explains A.
- P(A) is the **marginal** or **evidence** — a normalizing constant.
- P(B | A) is the **posterior** — your updated belief.

### 4. Worked Example

Spam filter, numbers from the intuition section. Let S = "spam," L = "contains 'lottery'."

- P(S) = 0.2, P(¬S) = 0.8.
- P(L | S) = 150/200 = 0.75.
- P(L | ¬S) = 10/800 = 0.0125.

Compute P(S | L):

P(L) = P(L|S)P(S) + P(L|¬S)P(¬S) = 0.75·0.2 + 0.0125·0.8 = 0.15 + 0.01 = 0.16.

P(S | L) = (0.75 · 0.2) / 0.16 = 0.15 / 0.16 ≈ 0.9375.

Seeing "lottery" raises spam probability from 20% to ~94%. Notice: even with a strong likelihood, the posterior depends on the prior. If spam were only 1% of email, the posterior would be (0.75·0.01)/(0.75·0.01 + 0.0125·0.99) ≈ 0.378 — same evidence, very different conclusion.

### 5. Relevance to ML Practice

- **Naive Bayes** classifiers apply Bayes' theorem with a conditional independence assumption across features.
- **Bayesian inference** updates posteriors over model parameters as data arrives — used in Bayesian neural nets, Gaussian processes, A/B testing.
- **Probabilistic graphical models** encode conditional independence to make joint distributions tractable.
- **Autoregressive generative models** (GPT-style) use the chain rule to factor sequences.
- **Calibration**: a classifier's outputs should behave like real conditional probabilities; if they don't, downstream decision-making (thresholding, expected utility) breaks.

### 6. Common Pitfalls

- **Base-rate neglect**: ignoring the prior (the "1% spam" example above). Common in medical-test reasoning.
- **Confusing P(A|B) with P(B|A)** (the prosecutor's fallacy).
- **Assuming independence when correlation exists** — the #1 cause of underestimated variance in ML pipelines.
- **Conditioning on the future**: using information in training that wouldn't be available at inference (data leakage).

---

## PART II — KEY DISTRIBUTIONS

### Discrete Distributions

#### Bernoulli(p)
A single binary trial. X ∈ {0,1}, P(X=1) = p. PMF: P(X=x) = pˣ(1−p)¹⁻ˣ. Mean p, variance p(1−p) (maximized at p=0.5, encoding maximum uncertainty). The atom of all binary classification: every logistic-regression output is the parameter of a Bernoulli.

#### Binomial(n, p)
Sum of n independent Bernoulli(p) trials. P(X=k) = C(n,k) pᵏ (1−p)ⁿ⁻ᵏ. Mean np, variance np(1−p). Use case: count of successes in a fixed number of i.i.d. trials — A/B test conversion counts, number of correct predictions in a held-out set.

#### Geometric(p)
Number of Bernoulli trials until the first success. P(X=k) = (1−p)ᵏ⁻¹ p, k=1,2,… Mean 1/p, variance (1−p)/p². Memoryless: P(X > m+n | X > m) = P(X > n). The only discrete distribution with this property.

#### Poisson(λ)
Count of events in a fixed interval when events occur independently at constant rate λ. P(X=k) = e⁻λ λᵏ / k!. Mean = variance = λ. Derivation sketch: take Binomial(n, λ/n) and let n → ∞ — using (1 − λ/n)ⁿ → e⁻λ you recover the Poisson PMF. Use cases: click counts per second, rare-event modeling, number of tokens of a rare word.

#### Multinomial(n, p₁,…,p_k)
Generalizes Binomial to k outcomes. P(X₁=n₁,…,X_k=n_k) = n!/(n₁!…n_k!) ∏ pᵢⁿⁱ. The categorical (k-class softmax output) is its n=1 special case. Foundation of cross-entropy loss in multi-class classification.

### Continuous Distributions

#### Uniform(a, b)
Density f(x) = 1/(b−a) on [a,b], else 0. Mean (a+b)/2, variance (b−a)²/12. The "no information" distribution on a bounded interval; appears in random initialization and as a non-informative prior.

#### Gaussian/Normal N(μ, σ²)
Density:

f(x) = (1/√(2πσ²)) exp(−(x−μ)²/(2σ²)).

Mean μ, variance σ². Why ubiquitous: (a) the CLT makes it the limiting distribution of sums; (b) it has maximum entropy among distributions with fixed mean and variance; (c) closed under linear combinations and conditioning; (d) its log-likelihood is quadratic, leading to least-squares estimators.

#### Exponential(λ)
Density f(x) = λe⁻λx for x ≥ 0. Mean 1/λ, variance 1/λ². Models waiting times between Poisson events. Memoryless: P(X > s+t | X > s) = P(X > t) — the only continuous distribution with this property.

#### Gamma(α, β)
Density f(x) = (βᵅ/Γ(α)) x^(α−1) e^(−βx). Generalizes Exponential (α=1) and is the distribution of the sum of α i.i.d. Exponential(β) variables when α is integer. Mean α/β, variance α/β². Conjugate prior for the precision (1/σ²) of a Gaussian.

#### Beta(α, β)
Density f(x) ∝ x^(α−1)(1−x)^(β−1) on [0,1]. Mean α/(α+β). Conjugate prior for Bernoulli/Binomial: if prior is Beta(α,β) and you observe s successes in n trials, posterior is Beta(α+s, β+n−s). This is the cleanest illustration of Bayesian updating in all of statistics.

### Multivariate Distributions

#### Multivariate Gaussian N(μ, Σ)
Density in d dimensions:

f(x) = (2π)^(−d/2) |Σ|^(−1/2) exp(−½ (x−μ)ᵀ Σ⁻¹ (x−μ)).

μ ∈ ℝᵈ is the mean vector, Σ is the d×d positive-definite covariance matrix. Key properties:
- **Marginals are Gaussian.** Drop coordinates → still Gaussian.
- **Conditionals are Gaussian.** If you partition x = (x₁, x₂), then x₁ | x₂ is Gaussian with mean μ₁ + Σ₁₂Σ₂₂⁻¹(x₂ − μ₂) and covariance Σ₁₁ − Σ₁₂Σ₂₂⁻¹Σ₂₁ (the Schur complement). This is the foundation of Gaussian processes and Kalman filters.
- **Linear transforms stay Gaussian.** AX + b ~ N(Aμ+b, AΣAᵀ).

#### Dirichlet(α₁,…,α_k)
A distribution over the (k−1)-simplex (vectors of nonnegative numbers summing to 1). Density f(p) ∝ ∏ pᵢ^(αᵢ−1). Conjugate prior for the Multinomial. Used in topic models (LDA), Dirichlet-process clustering, and any setting requiring a prior over categorical distributions.

---

## PART III — MOMENTS

The k-th **raw moment** of X is E[Xᵏ]; the k-th **central moment** is E[(X−μ)ᵏ]. The first four characterize most of what people care about:

- **Mean** μ = E[X]: location.
- **Variance** σ² = E[(X−μ)²]: spread. Has units of X². Standard deviation σ has units of X.
- **Skewness** = E[(X−μ)³]/σ³: asymmetry. Positive → long right tail (income, file sizes); zero for symmetric distributions like Gaussian.
- **Kurtosis** = E[(X−μ)⁴]/σ⁴: tail heaviness. Gaussian has kurtosis 3; "excess kurtosis" subtracts 3. Heavy-tailed distributions (Student-t, Cauchy) have high kurtosis and produce frequent outliers — critical for risk modeling and robust ML.

**Moment generating function** M_X(t) = E[e^(tX)]. When it exists in a neighborhood of 0, it uniquely determines the distribution and its derivatives at 0 give the moments: M_X^(k)(0) = E[Xᵏ]. MGFs are the key tool for proving the CLT.

---

## PART IV — CENTRAL LIMIT THEOREM

### Motivation
Why does the Gaussian appear everywhere — heights, measurement errors, average gradients in SGD? Because whenever you average many small, independent contributions, the result tends to look Gaussian regardless of the contributions' shape. This is the CLT, and it explains why so many ML estimators have approximately normal sampling distributions, allowing confidence intervals and z-tests.

### Statement
Let X₁,…,Xₙ be i.i.d. with finite mean μ and finite variance σ². Define the standardized sum:

Z_n = (X̄_n − μ) / (σ/√n), where X̄_n = (1/n) Σ Xᵢ.

Then Z_n converges in distribution to N(0,1) as n → ∞.

### Sketch of proof via MGFs
Let Y_i = (X_i − μ)/σ, so E[Y_i]=0, Var(Y_i)=1. Then Z_n = (1/√n) Σ Y_i. Its MGF:

M_{Z_n}(t) = [M_Y(t/√n)]ⁿ.

Taylor-expand M_Y: M_Y(s) = 1 + 0·s + s²/2 + o(s²). So M_{Z_n}(t) = [1 + t²/(2n) + o(1/n)]ⁿ → exp(t²/2) as n → ∞, which is the MGF of N(0,1). By Lévy's continuity theorem, convergence of MGFs implies convergence in distribution.

### Convergence rate (Berry–Esseen)
If E[|X−μ|³] = ρ < ∞, then sup_x |F_{Z_n}(x) − Φ(x)| ≤ C ρ / (σ³ √n), with C ≈ 0.47. Convergence is O(1/√n), and skewed/heavy-tailed distributions need much larger n before the Gaussian approximation is good.

### Assumptions and failures
- **Finite variance is required.** Cauchy distribution has undefined mean and infinite variance — averages of n Cauchy variables are still Cauchy, never Gaussian. A famous CLT counterexample.
- **Independence.** Strong dependence (e.g., autocorrelated time series) slows or breaks convergence; "effective sample size" can be much smaller than n.
- **Identical distribution** can be relaxed (Lindeberg–Feller CLT) but each component must contribute negligibly to the total variance.

---

## PART V — LAW OF LARGE NUMBERS

### Motivation
The LLN says sample averages converge to expectations as n grows. It's why empirical risk minimization works: minimizing the average loss on training data is a sensible proxy for minimizing the (unobservable) expected loss on the data distribution.

### Two forms
- **Weak LLN (WLLN):** X̄_n →ᵖ μ (convergence in probability). For every ε > 0, P(|X̄_n − μ| > ε) → 0.
- **Strong LLN (SLLN):** X̄_n →ᵃˢ μ (almost sure convergence). The set of sample paths where X̄_n fails to converge to μ has probability zero.

SLLN implies WLLN; the difference is technical but matters in measure-theoretic proofs.

### Proof of WLLN via Chebyshev (assuming finite variance)
Chebyshev's inequality: P(|Y − E[Y]| > ε) ≤ Var(Y)/ε².

Apply to Y = X̄_n: E[X̄_n] = μ, Var(X̄_n) = σ²/n. Therefore P(|X̄_n − μ| > ε) ≤ σ²/(nε²) → 0. Done.

The SLLN (Kolmogorov) requires only finite mean, but the proof is harder.

### CLT vs LLN
LLN tells you X̄_n → μ. CLT tells you the *fluctuations* around μ scale like 1/√n and are Gaussian. CLT is a refinement of LLN.

---

## INTERVIEW PREPARATION

### Foundational

**Q1. What is the difference between independent and mutually exclusive events?**
Mutually exclusive means they cannot both happen: P(A∩B)=0. Independent means knowing one doesn't change the probability of the other: P(A∩B)=P(A)P(B). For events with positive probability, mutually exclusive implies *dependent* — if A happened, B definitely didn't.

**Q2. State Bayes' theorem and explain each term.**
P(H|D) = P(D|H)P(H)/P(D). H is the hypothesis, D the data. P(H) is the prior, P(D|H) the likelihood, P(D) = Σ P(D|Hᵢ)P(Hᵢ) the evidence (normalizer), P(H|D) the posterior.

**Q3. Why is the Gaussian so common in ML?**
Three reasons: (1) CLT makes it the limit of averages of independent contributions; (2) it is the maximum-entropy distribution for fixed mean and variance, so it encodes minimal additional assumptions; (3) it is analytically convenient — closed under linear maps, conditioning, and marginalization, with quadratic log-density giving rise to least-squares solutions.

**Q4. What does it mean for a distribution to be memoryless? Which ones are?**
P(X > s+t | X > s) = P(X > t). Only the geometric (discrete) and exponential (continuous) distributions have this property. It's why exponential models "time to next event" when events are Markovian.

### Mathematical

**Q5. Derive the mean and variance of a Bernoulli(p).**
E[X] = 1·p + 0·(1−p) = p. E[X²] = 1·p + 0·(1−p) = p. Var(X) = E[X²] − E[X]² = p − p² = p(1−p).

**Q6. Show that the variance of the sample mean of n i.i.d. variables with variance σ² is σ²/n.**
Var(X̄_n) = Var((1/n)Σ Xᵢ) = (1/n²) Σ Var(Xᵢ) = (1/n²)·nσ² = σ²/n. Independence is essential for the cross-covariances to vanish.

**Q7. Derive Bayes' theorem from the definition of conditional probability.**
P(A|B) = P(A∩B)/P(B) and P(B|A) = P(A∩B)/P(A). So P(A∩B) = P(B|A)P(A), substitute: P(A|B) = P(B|A)P(A)/P(B). □

**Q8. State the CLT precisely and explain what assumption fails for the Cauchy distribution.**
Stated above. Cauchy has no finite variance (its tails decay like 1/x²), so the CLT's hypothesis fails. The average of n i.i.d. Cauchy variables is itself Cauchy with the same scale — no concentration whatsoever.

**Q9. What is the conjugate posterior of a Beta(α,β) prior under Bernoulli observations?**
After observing s successes and f failures, the posterior is Beta(α+s, β+f). Derivation: posterior ∝ likelihood × prior = pˢ(1−p)^f · p^(α−1)(1−p)^(β−1) = p^(α+s−1)(1−p)^(β+f−1), which is Beta(α+s, β+f) up to normalization.

**Q10. Why does the Berry–Esseen rate matter in practice?**
It tells you the Gaussian approximation has error O(1/√n) with a constant depending on the third absolute moment. Heavy-tailed or highly skewed data require much larger n for normal-approximation-based confidence intervals to be trustworthy. For severely skewed loss distributions, n=30 (the folklore threshold) can be wildly insufficient.

**Q11. Derive the conditional distribution for a bivariate Gaussian.**
For (X₁, X₂) joint Gaussian with means μ₁, μ₂ and covariance blocks Σ₁₁, Σ₁₂, Σ₂₂, the conditional X₁ | X₂=x₂ is Gaussian with mean μ₁ + Σ₁₂Σ₂₂⁻¹(x₂−μ₂) and covariance Σ₁₁ − Σ₁₂Σ₂₂⁻¹Σ₂₁ (Schur complement). Proof: complete the square in the joint density's quadratic form, separating terms involving x₁ from those depending only on x₂.

### Applied

**Q12. You're A/B-testing a button. Conversions are 120/2000 vs 145/2000. How would you analyze this?**
Model each variant as Binomial(2000, p). For each, p̂ ≈ k/n with standard error √(p̂(1−p̂)/n). Compute z = (p̂_B − p̂_A)/√(SE_A² + SE_B²), or use a two-proportion z-test or a Bayesian approach with Beta(1,1) priors yielding Beta(k+1, n−k+1) posteriors and computing P(p_B > p_A) by Monte Carlo. Discuss multiple-testing if running many experiments.

**Q13. When would you choose Poisson regression over linear regression?**
When the response is a nonnegative count and variance grows with the mean. Linear regression assumes homoscedastic Gaussian noise, which is wrong for counts. Poisson GLM uses a log link and the Poisson likelihood. Watch for overdispersion (variance > mean) — switch to negative binomial if present.

**Q14. Your classifier outputs probabilities but they're poorly calibrated. What does that mean and how do you fix it?**
"Calibrated" means among predictions of probability p, roughly fraction p are positive. Modern deep nets are typically overconfident. Fix with post-hoc methods: Platt scaling (logistic regression on logits), isotonic regression, or temperature scaling (divide logits by T > 1, fit T on validation NLL). Calibration matters whenever downstream decisions use the probabilities, not just the argmax.

**Q15. Why might a Naive Bayes classifier work surprisingly well despite its "obviously wrong" independence assumption?**
Because classification only needs the argmax of posteriors, not accurate posterior values. Even badly miscalibrated estimates can rank classes correctly. Naive Bayes also has very low variance (few parameters), so on small datasets it often outperforms more flexible models that overfit.

### Debugging & failure-mode

**Q16. You bootstrap a metric and get a tiny confidence interval, but on new data the metric varies wildly. What went wrong?**
Likely violated independence — maybe rows from the same user, session, or time window are correlated, so the effective sample size is much smaller than n. Bootstrap at the cluster level (block bootstrap, user-level resampling) instead of row level.

**Q17. Your normal-approximation confidence interval covers negative values for a quantity that must be positive. What happened and what do you do?**
The Gaussian approximation is poor — your distribution is skewed or you don't have enough samples. Use a transformation (log), a distribution that respects the support (Gamma, log-normal), or a bootstrap percentile interval, or a Bayesian credible interval with an appropriate prior.

**Q18. A model trained with cross-entropy on imbalanced data predicts the majority class for everything. Why, in probabilistic terms?**
Cross-entropy is the negative log-likelihood of a Categorical/Bernoulli. With imbalance, the prior P(y=majority) is huge, so the unconstrained MLE for the marginal is to predict it always, which already drives loss low. Fix via reweighting (effectively reweighting the likelihood), resampling, or using losses like focal loss that down-weight easy examples.

### Probing follow-ups

**Q19. (Follow-up to Q3) "Maximum entropy under fixed mean and variance" — what does that buy us epistemologically?**
It means choosing Gaussian commits to the *least* additional information beyond the constraints. Any other distribution with the same mean and variance smuggles in extra assumptions about higher moments. This is Jaynes's principle of maximum entropy.

**Q20. (Follow-up to Q12) What if conversions are extremely rare, like 5/10000 vs 3/10000?**
The normal approximation becomes unreliable; np is small. Use exact tests (Fisher's exact, binomial test) or Bayesian methods with Beta priors. Discuss that you may need much larger samples or a different metric (e.g., revenue per user) to detect meaningful differences.

**Q21. (Follow-up to CLT) How would you empirically verify the CLT is "kicking in" for your data?**
Bootstrap the statistic of interest many times and inspect the resulting distribution: QQ-plot against a Gaussian, check skewness/kurtosis, compare bootstrap CI width to the normal-approximation CI. If they disagree noticeably, trust the bootstrap.

**Q22. (Follow-up to conjugacy) What do you do when no conjugate prior is available?**
Use approximate inference: MCMC (Metropolis-Hastings, HMC, NUTS) for asymptotically exact samples; variational inference for fast approximate posteriors via optimization; Laplace approximation (Gaussian centered at the MAP with covariance from the inverse Hessian) for cheap local approximations. The Bernstein–von Mises theorem says that under regularity conditions the posterior becomes approximately Gaussian as n → ∞ regardless of the prior, justifying Laplace asymptotically.

---

*End of guide.*
