# Feature Creation: A Comprehensive Learning and Interview Reference

This guide covers the full landscape of manually engineered features used in classical and modern machine learning systems. Each sub-topic is structured as a self-contained chapter with a matching interview question bank.

**Contents**

1. Interaction Features (polynomial, ratio, domain combinations)
2. Aggregation Features (group statistics, time-window aggregations)
3. Text Features (bag-of-words, TF-IDF, word and character n-grams)
4. Image Features (HOG, SIFT, color histograms, pretrained CNN features)
5. Time-Series Features (lags, rolling statistics, Fourier, wavelets)
6. Geospatial Features (distances, spatial aggregations, density measures)
7. Graph Features (centrality, clustering coefficients, PageRank)

---

# 1. Interaction Features

## 1.1 Motivation & Intuition

### Why do interactions exist?

A linear model like `price = β₀ + β₁·size + β₂·location_score` says: "every extra square foot adds the same amount of money, regardless of the neighborhood." That is obviously wrong. An extra square foot in Manhattan is worth far more than in rural Kansas. The *effect* of `size` depends on `location_score`. When the effect of one feature depends on another, we say the two features **interact**.

Concrete, everyday examples:

- **Medicine**: Aspirin and ibuprofen each reduce inflammation, but taken together the effect is not the simple sum — they interact in the bloodstream.
- **Credit scoring**: Having a high income is good. Having a long credit history is good. Having both is *more* than additively good because it signals stability.
- **E-commerce**: A discount of 20% matters a lot for a $500 item and barely at all for a $2 item. The effect of `discount_pct` depends on `price`.
- **Physics**: Kinetic energy is ½·m·v². Doubling mass and velocity does not double energy — it quadruples it. That is a classic multiplicative interaction.

### Why do we have to create them?

Some model families (random forests, gradient-boosted trees, neural networks) can learn interactions from raw features, because their non-linear splits or non-linear activations let them represent `f(x₁, x₂) ≠ g(x₁) + h(x₂)`. Other model families (linear regression, logistic regression, linear SVMs, most generalized linear models, most ranking models) can only represent additive combinations of their inputs. For these, if you want the model to *know* that `size × location` matters, you must build that column yourself.

Even for tree-based models, explicit interaction features can help when:
- The interaction is very strong but the raw signal in either feature alone is weak (trees may not find it from greedy splits).
- You have a strong domain prior (e.g. BMI = weight/height²) that would take the model many data points to discover.
- Training data is small, so you want to inject structure to reduce variance.

### Connection to real ML systems

- **Click-through-rate (CTR) prediction** in ads: logistic regression with massive cross-feature vectors (`user_country × ad_category`) was the industry workhorse for a decade and still appears in modern wide-and-deep models.
- **Recommendation**: matrix factorization is exactly `user_embedding · item_embedding`, a dot-product interaction.
- **Fraud detection**: `transaction_amount / avg_amount_last_30_days` is a ratio feature that almost any production fraud model uses.

## 1.2 Conceptual Foundations

### Key terms

- **Additive effect**: the contribution of a feature to the prediction does not depend on other features.
- **Interaction**: the contribution of one feature depends on the value of another.
- **Polynomial feature**: a feature formed by raising a single feature to a power, or multiplying several features together.
- **Ratio feature**: a feature formed by dividing two features (or a function of them).
- **Domain combination**: a feature built by combining raw features using subject-matter knowledge (BMI, price-per-room, miles-per-gallon).
- **Degree**: the total power of a polynomial feature. `x₁ · x₂²` has degree 3.
- **Main effect**: the coefficient on a single feature (no interaction).
- **Cross feature** (in categorical settings): the concatenation of two categorical values, treated as a new category. E.g., `country=US ∧ device=iOS` becomes the token `US__iOS`.

### How interactions interact with components of a model

1. **Feature construction step**: compute `x₁·x₂`, `x₁/x₂`, `x₁²`, etc., and append as new columns.
2. **Model fits coefficients** on the combined matrix. For a linear model:

   `ŷ = β₀ + β₁x₁ + β₂x₂ + β₁₂·(x₁·x₂)`

   β₁₂ is now the **interaction coefficient**: how much more (or less) the effect of x₁ is per unit of x₂.
3. **Regularization** (L1/L2) is usually applied to interaction terms as well, often with stronger regularization because there are many of them and most are noise.
4. **Evaluation** needs care: a strong interaction can mask a misspecified main effect or vice versa.

### Underlying assumptions

- The **functional form** you're imposing (polynomial, ratio, product) is a good local approximation. Any smooth function can be approximated by polynomials (Stone-Weierstrass theorem), but the degree required may be very high.
- For **ratio features**, the denominator is meaningfully non-zero across the data.
- For **domain combinations**, the combination is physically or behaviorally meaningful.
- Features should be **on comparable scales** when taking products; otherwise one feature dominates the interaction.

### What breaks when assumptions are violated

- **Division by near-zero** produces exploding values; even one such row can dominate the loss (for MSE) and wreck training.
- **Unscaled polynomial features** produce numerical ill-conditioning. If x is in thousands, x² is in millions, x³ in billions — the design matrix becomes nearly singular.
- **Spurious interactions** from too-high-degree polynomials cause overfitting; degree-10 polynomial regression on 30 points is the canonical runaway-oscillation example.
- **Multicollinearity** between x, x², x³ inflates coefficient variance; you can mitigate with orthogonal polynomials.

## 1.3 Mathematical Formulation

### Polynomial features

Let **x** ∈ ℝⁿ be a feature vector. The degree-d polynomial expansion is the set of all monomials:

**φ(x) = { x₁^{a₁} · x₂^{a₂} · ⋯ · xₙ^{aₙ} : aᵢ ∈ ℤ₊, ∑aᵢ ≤ d }**

The number of features in the expansion (including the constant 1 and raw features) is:

**|φ(x)| = C(n + d, d) = (n+d)! / (n! · d!)**

**Intuition**: each monomial picks a "multiset" of features of size ≤ d with replacement.

Quick sanity check: n=3 features, d=2 gives C(5,2)=10 features: {1, x₁, x₂, x₃, x₁², x₂², x₃², x₁x₂, x₁x₃, x₂x₃}. ✓

When we only want **interaction terms** (no pure powers like x₁²), the set becomes:

**φ_int(x) = { ∏_{i∈S} xᵢ : S ⊆ {1,…,n}, |S| ≤ d }**

and the count is **∑_{k=0}^{d} C(n, k)**.

### Ratio features

Ratio features model the case where it is the relative magnitude of two quantities that matters:

**r(x₁, x₂) = x₁ / x₂**

To avoid division by zero and heavy tails, practitioners use:

- **Laplace smoothing**: r = (x₁ + α) / (x₂ + α), α > 0.
- **Log-ratio**: log((x₁ + 1) / (x₂ + 1)), which symmetrizes and compresses.
- **Sigmoid of log-ratio**: equivalent to x₁/(x₁+x₂), bounded in [0,1].

### Why products implement interactions in a linear model

Consider a linear model with the product feature:

**ŷ = β₀ + β₁x₁ + β₂x₂ + β₁₂ x₁x₂**

The **partial derivative** with respect to x₁ is:

**∂ŷ/∂x₁ = β₁ + β₁₂ · x₂**

This explicitly says: the rate at which ŷ increases with x₁ depends on x₂. That is the algebraic definition of interaction.

### Deriving the "centering trick" for interpretability

Raw product features make β₁ and β₂ hard to interpret (they become conditional effects at x₂ = 0, which may not exist in the data). If we center features (x̃ᵢ = xᵢ − x̄ᵢ), then:

- β̃₁ is the effect of x₁ at the mean of x₂
- β̃₁₂ is unchanged (it is scale-free up to shifts)

This is why textbooks recommend centering *before* forming interactions.

### Orthogonal polynomials

To remove collinearity, use an orthogonal basis like Legendre or Chebyshev polynomials: P₀(x)=1, P₁(x)=x, P₂(x)=½(3x²−1), … They span the same polynomial space but satisfy

**∫₋₁¹ Pₘ(x) Pₙ(x) dx = 0 for m ≠ n**

so their columns in the design matrix are approximately uncorrelated, leading to stable, interpretable coefficients.

## 1.4 Worked Example

### Setting

Predict house price from `size` (sqft, range 500–5000) and `location_score` (0–10).

### Data (toy)

| size | loc | price |
|------|-----|-------|
| 1000 | 3 | 200 |
| 1000 | 7 | 400 |
| 3000 | 3 | 500 |
| 3000 | 7 | 1100 |

Prices in $1000s.

### Step 1 — Fit additive linear model

`price = β₀ + β₁·size + β₂·loc`

Solving by least squares gives approximately β₁ ≈ 0.175 per sqft, β₂ ≈ 100 per loc unit, β₀ ≈ −225. Predictions:

| size | loc | actual | predicted |
|------|-----|--------|-----------|
| 1000 | 3 | 200 | 250 |
| 1000 | 7 | 400 | 650 |
| 3000 | 3 | 500 | 600 |
| 3000 | 7 | 1100 | 1000 |

The model **systematically underestimates** the expensive (3000, 7) house and overestimates the (1000, 7) house. This is the signature of a missing interaction: size matters *more* in good locations.

### Step 2 — Add the interaction

New feature `size_x_loc = size · loc`. Fit:

`price = β₀ + β₁·size + β₂·loc + β₁₂·(size·loc)`

On these four points, the system has 4 equations and 4 unknowns, so we can solve exactly. Call the feature values (s, l, sl). The system is:

```
β₀ +  1000β₁ +  3β₂ +  3000β₁₂ = 200
β₀ +  1000β₁ +  7β₂ +  7000β₁₂ = 400
β₀ +  3000β₁ +  3β₂ +  9000β₁₂ = 500
β₀ +  3000β₁ +  7β₂ + 21000β₁₂ = 1100
```

Subtracting row 1 from row 2: 4β₂ + 4000β₁₂ = 200, so β₂ + 1000β₁₂ = 50.
Subtracting row 3 from row 4: 4β₂ + 12000β₁₂ = 600, so β₂ + 3000β₁₂ = 150.
These give 2000β₁₂ = 100 → **β₁₂ = 0.05**, then **β₂ = 0**.

Subtracting row 1 from row 3: 2000β₁ + 6000β₁₂ = 300, so 2000β₁ + 300 = 300 → **β₁ = 0**.

From row 1: β₀ + 0 + 0 + 3000·0.05 = 200 → **β₀ = 50**.

Final model: **price ≈ 50 + 0.05·(size · loc)**.

Predictions: 50 + 0.05·3000 = 200 ✓ (1000,3); 50 + 0.05·7000 = 400 ✓; 50 + 0.05·9000 = 500 ✓; 50 + 0.05·21000 = 1100 ✓. Exact fit.

### Interpretation

Price is almost entirely driven by the product `size × loc`. Neither feature alone predicts price well; their interaction is the whole story. An additive model can never recover this no matter how much data you throw at it.

### Ratio example

On the same data, compute `price_per_sqft = price / size`. You'd find it varies systematically with `loc`: 0.2, 0.4, 0.167, 0.367 → approximately a linear function of `loc`, which is another way to reveal the same interaction.

## 1.5 Relevance to ML Practice

### Where interactions appear in the stack

- **Training**: feature engineering pipelines (scikit-learn's `PolynomialFeatures`, Spark ML's `Interaction`, Feast, Tecton). GBMs like XGBoost/LightGBM also expose `max_depth` which controls the order of interactions trees can express (depth d → up to d-way interaction).
- **Inference**: interaction features must be computed identically online and offline — *training-serving skew* is a classic bug.
- **Monitoring**: track distribution of interaction features the same way you track raw features. A shift in `x₁·x₂` when x₁ and x₂ individually look fine is a red flag.

### When to use

- Linear/logistic models where interactions are believed or known.
- Small data regimes where a tree cannot discover the interaction reliably.
- When interpretability matters and you want a coefficient per effect.
- Hand-engineered domain features (BMI, debt-to-income, impressions-per-user).

### When not to use

- High-dimensional sparse features (e.g. one-hot of 10⁶ users × 10⁶ items): explicit interaction creates 10¹² features. Use embeddings and dot products instead.
- When you already have a deep network that can learn them.
- When the "interaction" is a numerical artifact (e.g. two features that are nearly collinear to begin with).

### Alternatives

- **Tree-based models**: represent interactions implicitly via splits.
- **Neural networks with ReLU/Tanh**: universal function approximators.
- **Factorization machines** and **field-aware factorization machines**: model pairwise interactions with low-rank parameterizations, avoiding the O(n²) parameter explosion.
- **Kernel methods**: polynomial kernel `(x·x' + c)^d` implicitly represents all polynomial interactions up to degree d without constructing them.

### Trade-offs

- **Bias-variance**: more interactions → lower bias, higher variance. Regularization and cross-validation are essential.
- **Interpretability**: a linear model with a few hand-picked interactions is far more interpretable than a deep net; polynomials of degree > 3 rarely interpret cleanly.
- **Computation**: dense d-degree polynomial expansion of n features is C(n+d,d) dimensional — prohibitive above, roughly, n=100 and d=3.
- **Robustness**: polynomial features extrapolate wildly outside training range; ratio features blow up at zero denominators.

## 1.6 Common Pitfalls & Misconceptions

- **"Interactions always help."** They help only when they exist in the true data-generating process. Adding all pairwise products to a linear model often reduces test performance via overfitting.
- **Not scaling before polynomial expansion.** `x` in [0, 10⁶] produces x² in [0, 10¹²]. Without standardization, the optimization landscape is catastrophic.
- **Division-by-zero in ratio features.** Always use a small ε or Laplace smoothing, and think about what zero denominators *mean*.
- **Confusing correlation with interaction.** Two features can be highly correlated but have no interaction effect, and vice versa.
- **Using centered features naively with categorical interactions.** Centering a one-hot feature doesn't give you what you think; use treatment coding or mean-coding carefully.
- **Forgetting that trees already capture interactions.** Adding `x₁·x₂` to an XGBoost model usually does not help and sometimes hurts by giving the greedy splitter a misleading surrogate.
- **Ignoring causal direction.** `price_per_sqft = price / size` includes the target on the left-hand side implicitly. Using it as a feature to predict price is target leakage.
- **Interpreting β₁ in the presence of `x₁·x₂`.** β₁ is no longer the main effect of x₁; it is the effect of x₁ when x₂ = 0, which may not exist in the data.

## 1.7 Interview Preparation — Interaction Features

### Foundational

**Q1. What is a feature interaction, and why might a model need explicit interaction features?**
An interaction is a setting where the effect of one feature on the target depends on the value of another. Linear and other additive models can only represent sums of individual feature effects, so they cannot capture interactions unless you construct the interacting feature explicitly — typically as a product or ratio. Tree models and neural nets can often learn interactions, but explicit interactions still help when the signal is weak, the data is small, or interpretability is required.

**Q2. How do polynomial features differ from interaction-only features?**
Polynomial features of degree d include all monomials up to degree d, which contains both pure powers (x₁², x₁³) and cross-products (x₁x₂). "Interaction-only" features drop the pure powers and keep just the cross-products.

**Q3. Give three examples of domain-knowledge features you've seen or would build.**
BMI = weight/height² (medicine); debt-to-income ratio (credit); price-per-impression (ads); distance between pickup and drop-off (ride-hailing); average-session-length × sessions-per-week (user engagement).

### Mathematical

**Q4. How many features does a degree-d polynomial expansion of n raw features produce?**
C(n+d, d). Derivation: each polynomial feature corresponds to choosing a multiset of size ≤ d from n options with repetition; by the stars-and-bars lemma this is C(n+d, d). For n=10, d=3 that is 286; for n=100, d=3 it is 176,851 — illustrating why high-dimensional polynomials are expensive.

**Q5. For a model ŷ = β₀ + β₁x₁ + β₂x₂ + β₁₂x₁x₂, derive ∂ŷ/∂x₁ and interpret β₁.**
∂ŷ/∂x₁ = β₁ + β₁₂·x₂. β₁ is the marginal effect of x₁ when x₂ = 0, not the "overall" effect. If x₂ = 0 is far outside the data, β₁ can be meaningless or misleading; centering x₂ gives β₁ the more useful interpretation of "effect of x₁ at the mean of x₂".

**Q6. Why can polynomial regression cause ill-conditioning and how is it mitigated?**
If raw features span orders of magnitude, powers span even more (e.g., x³ in [0, 10⁹]). The design matrix columns become nearly linearly dependent, raising the condition number and making the normal equations numerically unstable. Mitigations: standardize features before expansion, use orthogonal polynomials (Legendre/Chebyshev), or regularize (ridge).

**Q7. Why does the polynomial kernel (x·x' + c)^d implicitly capture polynomial interactions?**
Expanding the kernel gives a sum over all monomials of degree ≤ d with binomial weights. The corresponding feature map is the full polynomial expansion, so kernel methods use it without materializing it — dimension C(n+d, d) is replaced by computing an inner product in the original space, O(n) work per evaluation.

### Applied

**Q8. You're building a CTR model with 1M users and 1M ads. You want user×ad interactions. How?**
Explicit one-hot cross produces 10¹² features — infeasible. Practical options: (1) embeddings with a dot-product layer (matrix factorization / two-tower); (2) factorization machines; (3) hashing tricks to bucket user×ad pairs into a manageable dimension; (4) feature crosses only on low-cardinality slices like user_country × ad_category.

**Q9. In a linear regression of house prices, you add sqft² and get better training MSE but worse validation MSE. Diagnose.**
Overfitting. sqft² is usually highly correlated with sqft, and on a small or noisy validation set can latch onto sample-specific curvature. Fixes: regularize (ridge/lasso), cross-validate to select the polynomial degree, use orthogonal polynomials, or use a non-parametric model like GAM with smoothing splines.

**Q10. When would you prefer ratio features over product features?**
When the meaningful signal is relative magnitude, not joint level. Examples: price-per-sqft (relative), debt-to-income (relative), transaction_amount / avg_recent_amount (relative anomaly signal). Products are preferred when joint level matters (area = width × height, revenue = price × quantity).

### Debugging & failure modes

**Q11. After adding a ratio feature your logistic regression AUC drops from 0.82 to 0.55. What happened?**
Almost certainly division by zero or near-zero: the ratio blows up for a few rows and dominates the loss, pulling coefficients to absurd values. Also possible: the ratio introduced label leakage if the numerator or denominator encodes the target. Fix: add epsilon to the denominator, winsorize or log-transform, and re-examine construction.

**Q12. You add pairwise interactions for 200 features. Training takes 10× longer and validation accuracy falls. Why?**
C(200, 2) = 19,900 new features. Two things happen: (1) many are pure noise, and without regularization the model fits them (high variance); (2) colinearity inflates coefficient variance. Remedies: strong L1, use a model that handles interactions natively (GBM), or use factorization machines.

**Q13. The interaction coefficient β₁₂ is highly significant but βs₁ and β₂ are not. Is that okay?**
Yes — it is called "interaction without main effect." It is rare but legitimate. Be cautious: many practitioners enforce the *marginality principle* (keep main effects when including interactions) to ensure the model is parameterization-invariant to shifts in x. Reporting all three together is the defensible choice.

### Follow-up / probing

**Q14. Explain how XGBoost captures interactions without explicit features.**
Each split conditions on one feature; deeper trees chain splits on different features, so a leaf is reached by an AND of conditions across features. The prediction for that leaf therefore depends jointly on multiple features — an interaction. Tree depth d gives up to d-way interactions. Boosting multiple trees combines many such local interactions additively.

**Q15. What does the marginality principle say, and why do many modelers enforce it?**
The marginality principle: if an interaction is in the model, its constituent main effects should also be in the model. Reason: without the main effects, the interaction term's coefficient changes when you shift the features (x → x+c), which means the model is not invariant to arbitrary origin choices — bad for interpretability.

**Q16. Under what conditions would you expect a neural network to fail to learn a multiplicative interaction that is trivial to express as x₁·x₂?**
ReLU networks approximate products poorly at fine resolution — they must carve the (x₁, x₂) plane into many pieces to approximate the smooth product surface. With little data, high noise, or shallow networks, this approximation is weak. Explicitly engineering x₁·x₂ as an input feature or using multiplicative layers (gated units, bilinear layers) can sharply reduce sample complexity.

**Q17. Compare factorization machines to explicit polynomial expansion.**
Explicit pairwise expansion has O(n²) parameters, each fit independently; impossible for high-dimensional sparse features. Factorization machines represent each feature by a k-dim vector vᵢ and parameterize the interaction as ⟨vᵢ, vⱼ⟩. This has O(n·k) parameters, shares information across pairs, handles missing (feature,pair) combinations gracefully, and is evaluable in linear time via the FM trick: ½·∑_f ((∑ᵢ vᵢ,f xᵢ)² − ∑ᵢ vᵢ,f² xᵢ²).

---

# 2. Aggregation Features

## 2.1 Motivation & Intuition

### The core idea

An individual data row often does not carry enough context to be interpretable. A single credit-card transaction of $300 could be a normal grocery run or a rare red flag — you don't know without knowing *the context of this user*. **Aggregation features** summarize the history or neighborhood around the current row:

- "This user's average transaction last month was $120; today's $300 is 2.5× that."
- "The average price of items in this product category is $45."
- "In this user's session, there have been 17 page views before this one."

These are often the single most powerful features in production ML systems.

### Simple concrete examples

- **Fraud detection**: mean, std, max of transaction amount per user per rolling window (hour, day, week).
- **Recommendation**: average rating per item, number of ratings per item, mean rating given by each user.
- **Ads**: click-through rate per ad, per advertiser, per ad × device, computed on the last N days.
- **Retail**: average basket size per store per weekday; total units sold per SKU per month.

### Where aggregation features shine

They inject *population-level statistics* into a single row's prediction, giving the model access to information that would require a full join-and-groupby operation otherwise. They are particularly valuable when the target depends on how this row deviates from a baseline — which, in practice, is a huge fraction of commercial ML.

## 2.2 Conceptual Foundations

### Key terms

- **Group**: a set of rows sharing a key (user_id, session_id, item_id, geohash, hour-of-day).
- **Group statistic**: a scalar computed over the group (mean, std, min, max, count, quantile, skew, kurtosis, unique-count).
- **Time window**: a temporal slice (last 7 days, rolling 24 hours, today-so-far).
- **Rolling aggregation**: a statistic computed on a sliding window whose end moves with each row.
- **Expanding aggregation**: a statistic computed on all rows from the start of history up to (but not including) the current row.
- **Cardinality**: the number of unique values of a group key.
- **Target encoding** (a.k.a. mean encoding): aggregating the target variable by a categorical key (e.g. mean click rate per user-agent).
- **Leak**: computing the aggregation using data that would not be available at prediction time.

### Assumptions

1. **Stationarity within window**: the aggregation is informative because the within-group distribution is reasonably stable over the window.
2. **Sufficient count per group**: means and stds are unreliable when groups have only 1–2 observations.
3. **Temporal availability**: at prediction time you can actually compute the same aggregation with the same latency.
4. **Leakage-free construction**: for target encodings and time-based aggregations, use strictly past data.

### What breaks under violations

- **Non-stationarity**: "average CTR over the last 90 days" becomes stale during viral trends; the feature misleads the model.
- **Small groups**: means become dominated by one or two outliers. Smoothing (shrinkage to the global mean) is essential.
- **Offline vs online mismatch**: features computed in batch from historical logs can look different from features computed online due to data freshness, clock skew, or missing events — the canonical "training-serving skew" issue.
- **Leakage**: including today's target in an aggregation used as a feature inflates validation metrics catastrophically.

## 2.3 Mathematical Formulation

### Group statistics

Let {xᵢ}ᵢ=1,…,N be a dataset and g(i) the group key of row i. For group g, let Iᵍ = {i : g(i)=g}.

- **Count**: nᵍ = |Iᵍ|
- **Mean**: x̄ᵍ = (1/nᵍ) ∑ᵢ∈Iᵍ xᵢ
- **Variance**: σᵍ² = (1/(nᵍ−1)) ∑ᵢ∈Iᵍ (xᵢ − x̄ᵍ)²
- **Min / Max / Quantile q**: as usual, over Iᵍ

### Smoothed mean (empirical Bayes / James–Stein style)

When nᵍ is small, replace x̄ᵍ by a convex combination with the global mean μ:

**x̃ᵍ = (nᵍ · x̄ᵍ + m · μ) / (nᵍ + m)**

where m is a smoothing constant. Intuition: with nᵍ = 0 we get the global mean; as nᵍ → ∞ we get the raw group mean. m corresponds roughly to the "prior strength."

This is a **Bayesian posterior mean under a conjugate Gaussian prior**: if we assume x | μᵍ ~ N(μᵍ, σ²) and μᵍ ~ N(μ, σ²/m), then the posterior mean of μᵍ given data is exactly x̃ᵍ above.

### Time-window aggregations

For row at time t, a window W(t) = (t − Δ, t] selects the rows falling in that window. A rolling mean is:

**m(t) = (1/|W(t)|) ∑ᵢ:tᵢ∈W(t) xᵢ**

Note the half-open interval: **strictly before t**, to avoid using the current row in its own feature.

### Exponentially weighted moving average (EWMA)

Often more statistically efficient than fixed windows:

**m_t = α · xₜ + (1 − α) · m_{t−1}**, 0 < α ≤ 1

Unrolling: m_t = α ∑_{k=0}^{∞} (1−α)ᵏ · x_{t−k}, so recent observations are weighted exponentially more. The effective half-life is log(0.5) / log(1−α).

### Online variance (Welford's algorithm)

Essential for low-memory online feature computation. Maintain running (n, mean, M2):

```
n += 1
δ = x − mean
mean += δ / n
M2 += δ · (x − mean)
```

Then σ² = M2 / (n − 1). Numerically stable even for streaming data.

### Target encoding with leak-free cross-validation

For each row i with categorical key kᵢ and target yᵢ, compute the mean of y over all rows with the same key **excluding row i**:

**TE(i) = (∑_{j≠i, k_j=k_i} y_j) / (∑_{j≠i, k_j=k_i} 1)**

K-fold variant: for each fold, compute the target mean per key using only rows outside the fold. With smoothing:

**TE_smoothed(i) = (n_{k_i, −i} · TE(i) + m · μ) / (n_{k_i, −i} + m)**

## 2.4 Worked Example

### Setting

Fraud detection with the following toy log, chronologically ordered:

| tx_id | user | amount ($) | ts (h) | label |
|-------|------|------------|--------|-------|
| 1 | A | 30 | 1 | 0 |
| 2 | A | 45 | 3 | 0 |
| 3 | B | 200 | 4 | 0 |
| 4 | A | 500 | 5 | 1 |
| 5 | B | 220 | 6 | 0 |
| 6 | B | 50 | 8 | 1 |

Goal: for each tx, compute leak-free aggregation features:
- `user_avg_amount_prev`: average amount of user's previous transactions.
- `user_max_amount_prev`: max of user's previous amounts.
- `user_count_prev`: number of prior transactions by user.
- `ratio_vs_avg`: amount / user_avg_amount_prev.

### Construction

We iterate chronologically (essential) and use only **prior** information.

| tx | user | amount | prev_count | prev_mean | prev_max | ratio |
|----|------|--------|------------|-----------|----------|-------|
| 1 | A | 30 | 0 | NaN | NaN | NaN |
| 2 | A | 45 | 1 | 30 | 30 | 1.5 |
| 3 | B | 200 | 0 | NaN | NaN | NaN |
| 4 | A | 500 | 2 | 37.5 | 45 | 13.33 |
| 5 | B | 220 | 1 | 200 | 200 | 1.10 |
| 6 | B | 50 | 2 | 210 | 220 | 0.24 |

### Reading the features

- Tx 4 (fraud): ratio = **13.33**. User A's prior behavior averaged $37.50; a $500 charge is dramatically out of pattern. A tree model splits on `ratio > ~5` and nearly perfectly flags this event.
- Tx 6 (fraud): ratio = 0.24 — here the fraud is *small*. A pure amount-based rule misses it entirely. But user B's sudden transaction pattern change (from $200-range to $50) might still appear in another aggregation like `std` or `count`.

### Handling NaN for first-occurrence rows

For tx 1 and 3, there is no prior history. Options:
- Fill with global mean (naive; introduces leakage if computed on full data — compute it on training only).
- Fill with a flag value like −1 and add a binary indicator `is_first_tx`.
- Use Bayesian smoothing toward the global mean.

### Smoothing in action

Suppose the global mean amount is $150. With smoothing strength m = 3:

For tx 2: smoothed prev_mean = (1·30 + 3·150) / (1+3) = 480/4 = **120**. Ratio = 45/120 = 0.375 — much less alarming than 1.5 from the raw estimate (which itself wasn't alarming, but the smoothed estimate is better calibrated).

For tx 4: smoothed = (2·37.5 + 3·150) / (2+3) = (75+450)/5 = **105**. Ratio = 500/105 = 4.76 — still a large anomaly, but the estimate of "user's normal" is less over-influenced by just two prior small tx.

## 2.5 Relevance to ML Practice

### Where they are used

- **Fraud / abuse / risk**: per-user, per-device, per-IP rolling aggregates are the backbone.
- **Search and recommendation**: per-item click counts, per-query result quality.
- **Forecasting**: lagged group statistics (mean sales per SKU per store over the last 28 days).
- **Churn**: counts and aggregations over the user's activity timeline.
- **Ops / reliability ML**: per-service latency percentiles over rolling windows.

### Infrastructure reality

Large-scale aggregation features need **feature stores** with both batch (offline) and streaming (online) views. Tools: Feast, Tecton, SageMaker Feature Store, internal systems like Uber's Michelangelo or Twitter's feature platform. The batch view is used for training; the online view must match it exactly at inference time.

### When to use

- Whenever context from group membership is informative.
- When the target depends on deviations from history (anomaly-style problems).
- When you have many rows per group and windows are stationary.

### When not to use

- When groups are tiny (few rows) — aggregations are noisy.
- When stationarity is violated without a plan for re-computation.
- When online computation is infeasible and training-serving skew will be severe.

### Trade-offs

- **Information gain vs memory/latency cost**: maintaining rolling 24-hour features for 10⁸ users is a nontrivial engineering commitment.
- **Bias-variance**: narrow windows react fast but are high-variance; wide windows are smooth but lag.
- **Interpretability**: aggregation features are usually very interpretable ("user's 30-day average").
- **Leakage risk**: highest of any feature class; careless pipelines create phantom performance.

## 2.6 Common Pitfalls & Misconceptions

- **Leakage via same-row target**: when you aggregate target y grouped by key, do **not** include the current row. Use leave-one-out or K-fold.
- **Leakage across time**: sorting by group but not by time, or using pandas `.groupby().mean()` on the full dataset, lets future events influence past predictions. Always sort by time and compute strictly backward-looking aggregates.
- **Different aggregation definitions in train vs serve**: a classic "works great offline, tanks in production" pattern. Write the aggregation once and reuse.
- **Dropping zero-count groups**: sometimes the informative signal is that the user has *no* prior history. Encode count=0 explicitly rather than NaN-dropping.
- **Over-reliance on mean**: a single mean often isn't enough; means + stds + counts + extreme quantiles together carry far more signal.
- **Forgetting smoothing**: raw target encodings on small groups overfit viciously. Always smooth.
- **Ignoring seasonality**: a 24-hour rolling mean hides weekly patterns; a 7-day mean hides hourly patterns. Use both.

## 2.7 Interview Preparation — Aggregation Features

### Foundational

**Q1. Why are aggregation features typically among the strongest in fraud or recommendation systems?**
Because the target very often depends on *deviation from a baseline*, and the baseline can only be computed by aggregating historical rows in the same group (user, item, device, etc.). A single raw row contains very little of that context.

**Q2. What is the difference between rolling and expanding aggregations?**
A rolling window covers a fixed temporal extent ending at the current row; rows outside the window are excluded. An expanding aggregation includes everything from the start of history up to (but not including) the current row. Rolling responds quickly to shifts; expanding is smoother and can never drop signal but also never forgets stale patterns.

**Q3. List five group statistics that together give a fuller picture than mean alone.**
Count (sample size), standard deviation (spread), min/max (extremes), quantiles (distribution shape), skewness (asymmetry), nunique (diversity). Combining these often dominates single-statistic encodings.

### Mathematical

**Q4. Derive the Bayesian interpretation of smoothed mean encoding.**
Assume y_i | μ_k ~ N(μ_k, σ²) for each row in group k, with prior μ_k ~ N(μ, σ²/m). Posterior for μ_k given n_k observations with mean ȳ_k is also Gaussian:

posterior mean = (n_k · ȳ_k + m · μ) / (n_k + m)

This is the smoothed mean. m plays the role of the prior's effective sample size; with n_k = 0 the posterior equals the prior, matching the global mean.

**Q5. Welford's online variance — derive the update rule.**
Maintain (n, mean_n, M2_n = ∑(x_i − mean_n)²). When a new x arrives:

mean_{n+1} = mean_n + (x − mean_n)/(n+1)
M2_{n+1} = M2_n + (x − mean_n)(x − mean_{n+1})

Variance = M2_n / (n − 1). This is numerically stable vs the naive "sum of squares − square of sums" approach, which suffers catastrophic cancellation for large means.

**Q6. Given an EWMA with parameter α, what is the effective window length?**
The weights on past observations are α(1−α)ᵏ. Half-life (weight drops to 0.5) is k₁/₂ = log(0.5)/log(1−α). Effective window size (sum of weights proxy) is roughly 2/α − 1 or 1/α depending on definition.

**Q7. For K-fold target encoding, why do we use the global mean only from the training folds, not the full dataset?**
Because including the validation fold in the global mean leaks information about the held-out set into the feature's prior. The correct protocol: within each CV split, compute the target encoding (and its smoothing prior) using only the training data, then apply it to validation.

### Applied

**Q8. Design a feature pipeline that provides `user_rolling_30d_tx_count` in real time with < 100ms latency at a rate of 1M events/second.**
Approach: store per-user counters in a high-throughput key-value store (Redis, Cassandra). On each event, increment a sliding-window counter. Implement the window via either (a) discrete hourly buckets with TTLs (720 buckets per user, 30×24) and sum over the last 30 days at read time, or (b) a count-min sketch with time decay. Validate that the online value equals the offline replay from the event log — this training-serving parity check is essential.

**Q9. You have a categorical column with 500K unique values and want to mean-encode it. What do you do?**
Mean-encode with K-fold and strong Bayesian smoothing (m large). Consider adding the count feature alongside the encoding so the model knows the reliability. For values seen very few times, force back to the global mean. An alternative: use hashing and let the model learn the encoding, or use embedding layers if you are in a neural net.

**Q10. Name three aggregation features you would build for a ride-hailing ETA model.**
Rolling average speed on the route segment over the last 15 minutes; per-driver recent average deviation from predicted ETA; per pickup/dropoff polygon, hourly average trip duration over the past 4 weeks same-hour-same-weekday; surge level aggregate per zone; per-weather-condition average speed.

### Debugging & failure modes

**Q11. Your model's offline AUC is 0.98; in production it is 0.62. How do you diagnose?**
Number one suspicion: training-serving skew in aggregation features. Check: (a) are offline features computed using future data relative to the label timestamp? (b) are online features computed with the same SQL/code as offline? (c) are there latency differences causing online features to use stale data? Rebuild the offline features by replaying events through the *online* path ("backfill through production") and retrain. If AUC drops in offline to match production, the problem is confirmed.

**Q12. You switched from raw to mean-encoded user_id and training loss collapsed to zero. Why?**
Target leakage. You almost certainly included the current row in its own group when computing the encoding. `groupby('user_id')['y'].transform('mean')` on the whole dataframe computes each user's mean including the target you're trying to predict — so the feature trivially equals the label for one-transaction users and is near-perfect for small groups.

**Q13. Aggregations look fine in training but degrade steadily over weeks in production. What's happening?**
Possibilities: (1) the underlying distribution is drifting (covariate shift), making old aggregations stale; (2) model training is not refreshed frequently enough; (3) an upstream data pipeline introduced a silent schema change (e.g., new event types not in the training logs). Monitor feature distributions over time and compare to a rolling baseline; set alerts on PSI or KS statistics.

### Follow-up / probing

**Q14. Why is count often a useful feature alongside a mean encoding?**
It tells the model how confident the mean encoding is. A mean encoding computed from 2 rows is noisy; from 20,000 rows it's precise. Trees can use the count to choose how much to rely on the encoding for a given split.

**Q15. Explain how Welford's algorithm differs from the naive two-pass variance computation and why it matters.**
Two-pass: compute mean, then sum of (x − mean)² — requires storing or re-reading all data. One-pass naive: ∑x², ∑x, compute (∑x²)/n − (∑x/n)² — catastrophic cancellation when mean is large relative to the spread. Welford updates mean and M2 incrementally with O(1) memory and is numerically stable because it accumulates differences from the running mean rather than raw squares.

**Q16. What are the trade-offs between target encoding and embeddings for high-cardinality categoricals?**
Target encoding is a single scalar per category, very interpretable, trivially usable in any model including trees. It carries only one dimension of information and risks overfitting. Embeddings are k-dimensional, richer, learned jointly with the task, but require a differentiable model and significantly more engineering. For high-cardinality categoricals in GBMs, target encoding is still often the practical winner; in neural networks, embeddings.

**Q17. Can an aggregation feature be correlated with the target without being useful? Provide an example.**
Yes — if the aggregation includes the target itself (leakage), it will correlate almost perfectly but isn't useful because it won't exist at inference time. Another case: an aggregation that simply echoes another stronger feature (e.g. `per_country_avg_income` might correlate with the target but be dominated by `user_income` if present, adding nothing).

---

# 3. Text Features

## 3.1 Motivation & Intuition

### Why bother with bag-of-words in the transformer era?

Modern NLP is dominated by transformer embeddings. But bag-of-words, TF-IDF, and n-grams still matter:

- **Speed**: a linear SVM over TF-IDF classifies a million documents in seconds — orders of magnitude faster than BERT.
- **Interpretability**: you can literally inspect which words drove a prediction.
- **Strong baselines**: TF-IDF + linear model is famously hard to beat on many text classification problems with moderate data.
- **Low-resource languages and domains**: pretrained transformers may not exist; classical features always work.
- **Features for traditional ML**: GBMs over TF-IDF features are still used in production.

### Simple intuition

A document is just a sequence of words. The **bag-of-words** assumption says: ignore the order, just count the words. Surprisingly, that's enough for many tasks — sentiment classification, topic detection, spam filtering.

But raw counts have issues: the word "the" appears in every document and carries no information. **TF-IDF** solves this by downweighting words that are common across documents. **N-grams** put back a little of the order we threw away by using pairs or triples of consecutive words. **Character n-grams** are robust to typos and useful for morphology-rich languages.

### Real ML uses

- Spam classification in email.
- Topic tagging of news articles.
- Product review sentiment.
- Fast candidate retrieval in two-stage search systems (BM25 uses similar ideas).
- Clustering large document collections.

## 3.2 Conceptual Foundations

### Key terms

- **Token**: a unit of text, typically a word or a sub-word.
- **Tokenization**: process of splitting text into tokens (whitespace, regex, WordPiece, BPE, etc.).
- **Vocabulary**: the set of all tokens considered (size V).
- **Bag-of-Words (BoW)**: a document is represented as a length-V vector of token counts.
- **Term Frequency (TF)**: count (or relative frequency) of a term in a document.
- **Document Frequency (DF)**: number of documents that contain a term.
- **Inverse Document Frequency (IDF)**: a monotonically decreasing function of DF.
- **TF-IDF**: TF × IDF.
- **N-gram**: a contiguous sequence of n tokens. "bi-gram" (n=2), "tri-gram" (n=3).
- **Character n-gram**: an n-gram of characters instead of tokens. Useful for morphology, typo robustness.
- **Stopwords**: very common tokens ("the", "is", "and") often removed.
- **Stemming / Lemmatization**: reducing tokens to root form.
- **Sparse representation**: only non-zero entries are stored (CSR/COO matrices).
- **Hashing trick / feature hashing**: map tokens to a fixed-size bucket via a hash.

### How they compose into a pipeline

1. **Preprocess** — lowercase, strip punctuation, tokenize, optional stopword removal / stemming.
2. **Build vocabulary** — either full (all unique tokens above min_df) or via hashing (fixed size).
3. **Vectorize** — convert each document to a vector (counts, binary indicators, or TF-IDF weights).
4. **Feed to a model** — typically linear, though GBMs and naive Bayes are also used.

### Assumptions and what breaks

- **BoW ignores order**: "dog bites man" = "man bites dog". Fine for topic classification, disastrous for machine translation.
- **Independence of words**: naive Bayes assumes tokens are conditionally independent given the class — usually false but forgiving.
- **Fixed vocabulary**: out-of-vocabulary words at inference time are silently dropped (bad) or hashed (better).
- **Stationary language**: slang drifts; models trained on 2018 Twitter misread 2025 Twitter.

## 3.3 Mathematical Formulation

### Bag-of-Words

Let D be a corpus of N documents and V the vocabulary of size |V|. The BoW matrix X ∈ ℝᴺˣ|V| has:

**X_{ij} = count of term j in document i**

### Term Frequency variants

- **Raw count**: TF(t, d) = count(t, d).
- **Normalized**: TF(t, d) = count(t, d) / |d| (where |d| is document length). Makes documents of different lengths comparable.
- **Logarithmic**: TF(t, d) = 1 + log(count(t, d)) if count > 0, else 0. Dampens the effect of very frequent terms.
- **Binary**: TF(t, d) = 1 if t appears in d else 0.

### Inverse Document Frequency

Most common form:

**IDF(t) = log( N / DF(t) )**

Intuition: if a term appears in every document (DF=N), IDF = log(1) = 0. If it appears in only one document, IDF = log(N), the max.

Smoothed version (standard in scikit-learn):

**IDF(t) = log( (1 + N) / (1 + DF(t)) ) + 1**

This prevents zero IDF and zero division and ensures terms in every document still carry a tiny signal.

### TF-IDF

**TFIDF(t, d) = TF(t, d) · IDF(t)**

After this, each document vector is typically L2-normalized so that inner products correspond to cosine similarity.

### N-grams

For a tokenized document (w₁, w₂, ..., wₘ), n-grams are the sequences (w_i, w_{i+1}, ..., w_{i+n−1}) for i=1..m−n+1. Each unique n-gram becomes its own column in the vocabulary.

If base vocabulary size is V, the n-gram vocabulary can in principle be up to Vⁿ — almost always truncated by frequency cutoffs.

### Character n-grams

Same idea, but over characters. The tokenizer emits overlapping substrings of length n. For word "apple" with n=3 and boundary symbols (<,>): "<ap", "app", "ppl", "ple", "le>". Useful when:
- Tokens share morphology (running, runs, runner).
- Tokens may be misspelled.
- Language is agglutinative (Finnish, Turkish).

FastText famously averages character n-gram embeddings to get robust word vectors.

### Why log in IDF? (derivation via information theory)

Think of DF(t)/N as an estimate of P(t ∈ d), the probability a random document contains t. The *self-information* of that event is −log P = log(N/DF(t)) — exactly IDF. A term that is almost certain to appear carries almost no information; a rare term, a lot. TF-IDF is thus interpreted as "frequency-weighted self-information."

### Sparsity

Most documents use a tiny fraction of the vocabulary; X is extremely sparse (99%+ zero entries). All production toolkits represent X as a Compressed Sparse Row (CSR) matrix, reducing memory from O(N·V) to O(number of non-zeros) and enabling fast matrix-vector products.

### Hashing trick

For very large vocabularies, pre-assign each token to a bucket via a hash:

**index(t) = hash(t) mod B**

with B fixed (e.g. 2²⁰). Pros: no vocabulary to store, constant memory, handles OOV naturally. Cons: hash collisions mix distinct tokens; typically a small AUC hit in exchange for huge memory savings.

## 3.4 Worked Example

### Corpus

Document 1: "the cat sat on the mat"
Document 2: "the dog sat on the log"
Document 3: "the cat chased the dog"

### Step 1 — tokenize (whitespace + lowercase)

Vocabulary (sorted): {cat, chased, dog, log, mat, on, sat, the}. V = 8.

### Step 2 — BoW matrix

|  | cat | chased | dog | log | mat | on | sat | the |
|---|---|---|---|---|---|---|---|---|
| D1 | 1 | 0 | 0 | 0 | 1 | 1 | 1 | 2 |
| D2 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 2 |
| D3 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 2 |

### Step 3 — Document frequencies

| term | DF |
|---|---|
| cat | 2 |
| chased | 1 |
| dog | 2 |
| log | 1 |
| mat | 1 |
| on | 2 |
| sat | 2 |
| the | 3 |

### Step 4 — IDF (basic formula, log base e)

IDF(t) = log(N/DF) with N=3:

| term | IDF |
|---|---|
| cat | log(3/2) ≈ 0.405 |
| chased | log(3/1) ≈ 1.099 |
| dog | ≈ 0.405 |
| log | ≈ 1.099 |
| mat | ≈ 1.099 |
| on | ≈ 0.405 |
| sat | ≈ 0.405 |
| the | log(3/3) = 0 |

Note: "the" has IDF exactly 0 — it drops out entirely. That is the point.

### Step 5 — TF-IDF (using raw counts for TF)

| term | D1 TF-IDF | D2 TF-IDF | D3 TF-IDF |
|---|---|---|---|
| cat | 0.405 | 0 | 0.405 |
| chased | 0 | 0 | 1.099 |
| dog | 0 | 0.405 | 0.405 |
| log | 0 | 1.099 | 0 |
| mat | 1.099 | 0 | 0 |
| on | 0.405 | 0.405 | 0 |
| sat | 0.405 | 0.405 | 0 |
| the | 0 | 0 | 0 |

### Step 6 — Cosine similarity

D1 · D2 (dot product) = cat·cat + on·on + sat·sat + ... = 0 + 0 + 0 + 0 + 0 + 0.405² + 0.405² + 0 ≈ 0.328.
||D1||² = 0.405² + 1.099² + 0.405² + 0.405² ≈ 1.706. ||D1|| ≈ 1.306. Similarly ||D2|| ≈ 1.306.
cos(D1, D2) ≈ 0.328 / (1.306 · 1.306) ≈ **0.192**.

D1 · D3 = cat·cat = 0.405² ≈ 0.164. ||D3|| ≈ sqrt(0.405² + 1.099² + 0.405²) ≈ 1.234. 
cos(D1, D3) ≈ 0.164 / (1.306 · 1.234) ≈ **0.102**.

**Interpretation**: D1 and D2 are more similar to each other than D1 and D3 — because D1 and D2 share the "sat on" structure while D1 and D3 only share "cat." This matches intuition.

### Step 7 — Adding bigrams

Add bigrams {"the cat", "cat sat", "sat on", "on the", "the mat", ...}. Each document now also has columns for its bigrams; the representation becomes much richer and can distinguish between "the cat sat" and "sat the cat" if it ever occurred, though with far higher dimensionality. Typically, one truncates by `min_df` (keep only bigrams appearing in ≥ k documents) to control size.

## 3.5 Relevance to ML Practice

### Where used

- **Classification**: TF-IDF + linear model is the no-frills baseline for nearly every text classification task.
- **Retrieval**: BM25 (a TF-IDF variant) is still the dominant first-stage retriever in search engines, often paired with neural re-rankers.
- **Clustering**: TF-IDF with spherical k-means or HDBSCAN for topic discovery.
- **Feature extraction for downstream models**: e.g., pass TF-IDF vectors into a GBM alongside numeric features.
- **Spam/abuse detection**: character n-grams catch evasion patterns (v1@gra).

### When to use

- Strong baseline is needed quickly.
- Limited compute (no GPU).
- Interpretability is a hard requirement.
- Small to medium dataset; transformers underperform without fine-tuning on small labeled data.
- Data is domain-specific and pretrained models struggle.

### When not to use

- When order, negation, long-range context matter (sarcasm, natural-language inference).
- When you have lots of labels and compute and pretrained models.
- When out-of-vocabulary handling matters and you cannot retrain periodically.

### Alternatives

- **Word embeddings** (word2vec, GloVe, fastText) — dense, capture semantics.
- **Contextual embeddings** (BERT, GPT) — dense, context-aware.
- **Subword tokenizers** (BPE, WordPiece, SentencePiece) — vocabulary-agnostic.
- **Learned sparse representations** (SPLADE) — unify sparse TF-IDF-like interpretability with learned semantics.

### Trade-offs

- **Dimensionality**: V can be millions. Sparse storage mitigates but some models (SVMs) still pay.
- **OOV**: strict vocabularies miss unseen tokens; hashing collides them silently; character n-grams generalize better.
- **Robustness**: typos destroy word-level BoW; character-level is robust.
- **Semantics**: BoW treats "good" and "great" as unrelated. Embeddings know they're close.

## 3.6 Common Pitfalls & Misconceptions

- **Fitting TF-IDF on train + test**: computing IDF on the concatenated dataset uses test data to shape the training features — leakage. Always fit on train only and transform test.
- **Removing stopwords unconditionally**: for sentiment, "not" is a stopword in some lists but is semantically critical. Always inspect the stopword list.
- **Over-aggressive min_df**: excludes rare but informative terms.
- **Character n-grams on short tokens without word boundary markers**: loses crucial morphological structure. Use boundary symbols.
- **Assuming all text is English**: default tokenizers can do terrible things to Chinese, Japanese, Korean (no whitespace) or agglutinative languages.
- **Ignoring unicode normalization**: "café" can be two different byte strings (NFC vs NFD). Normalize consistently.
- **Not L2-normalizing**: cosine similarity assumes unit vectors; raw dot products on TF-IDF can be dominated by long documents.
- **Expecting TF-IDF to capture negation**: "not good" looks identical to "good" plus "not" separately.

## 3.7 Interview Preparation — Text Features

### Foundational

**Q1. Why does TF-IDF typically outperform raw bag-of-words?**
Raw BoW gives equal weight to every occurrence, so common uninformative terms ("the", "and") dominate. IDF explicitly downweights terms that appear in many documents, concentrating weight on terms that distinguish documents from each other. The result: TF-IDF vectors emphasize content-bearing words, giving linear classifiers a much cleaner signal.

**Q2. When would you use character n-grams instead of word n-grams?**
Morphology-rich languages (Finnish, Turkish); tasks involving typos, misspellings, or deliberate obfuscation (spam/abuse); short texts like usernames or queries; cross-script transliteration; when OOV rate is high. Character n-grams capture suffixes, prefixes, and subword patterns that word-level models miss.

**Q3. What problem does the hashing trick solve?**
It removes the need to store and maintain a vocabulary. Tokens are hashed to a fixed number of buckets, so you can build features from arbitrary text without first scanning the corpus, handle unseen tokens online, and control memory precisely. Price paid: hash collisions cause distinct tokens to share a column.

### Mathematical

**Q4. Derive the IDF formula from an information-theoretic perspective.**
Let p(t) = DF(t)/N be the empirical probability that a random document contains term t. The self-information of the event "document contains t" is −log p(t) = log(N / DF(t)), which is the IDF. This framing motivates TF-IDF as a frequency-weighted surprise score: common terms carry little surprise and are downweighted.

**Q5. Why is log used rather than 1/DF or N/DF linearly?**
Information-theoretically, log gives a natural scale where doubling the rarity adds a constant. Empirically, IDF growth should saturate as DF shrinks — a term in 1 of 10⁶ documents is not 10× more informative than one in 10 of 10⁶. The log damps extremes and produces better-behaved weights.

**Q6. Show that L2-normalized TF-IDF dot products equal cosine similarity.**
cos(a, b) = (a · b) / (||a|| · ||b||). If both a and b are L2-normalized (||a|| = ||b|| = 1), then cos(a, b) = a · b. So the dot product of normalized TF-IDF vectors is cosine similarity.

**Q7. Given vocabulary size V, how many bigrams can appear in theory? How many in practice?**
In theory, V². In practice, far fewer: natural language has strong locality (most word pairs never occur together), so bigrams follow a Zipf-like distribution. Typical corpora have perhaps 10-30× the unigram count in bigrams before filtering, and a factor of 10 less after min_df filtering.

### Applied

**Q8. You are asked to ship a sentiment classifier in 48 hours on a dataset of 100K reviews. Describe your choice.**
TF-IDF with unigrams+bigrams, min_df=5, L2-normalized, feeding a logistic regression with L2 regularization. Baseline in an hour. If needed, add character n-grams (3-5) and tune regularization with cross-validation. This routinely gets 90%+ of what fine-tuned BERT would deliver at a thousandth the cost and full interpretability.

**Q9. Your TF-IDF + LR model works well in English but poorly in Turkish. Why and how to fix?**
Turkish is heavily agglutinative — a single "word" encodes meanings that would be phrases in English. Word-level BoW fragments these forms (same root appearing in many surface forms) and explodes the vocabulary. Fixes: use character n-grams (3-6), use a subword tokenizer (BPE/SentencePiece), or lemmatize with a Turkish-specific tool.

**Q10. How would you add TF-IDF features to a gradient-boosted tree model when V is 500,000?**
GBMs don't handle ultra-high-dimensional sparse features well — they sample features per tree and memory becomes painful. Options: (a) reduce dimension with truncated SVD (LSA) to, say, 100 dense features; (b) keep only the top-k features by chi-squared or mutual information; (c) use a linear model as a "stacker" feeding predictions into the GBM; (d) use feature hashing to a smaller dimension.

### Debugging & failure modes

**Q11. Your TF-IDF + LR model has 99% training accuracy and 60% test accuracy. What's likely wrong?**
The vocabulary is almost certainly capturing tokens that memorize the training set — very rare words that appear in one class only. Fixes: set min_df higher (e.g., 5-10), enable max_features, strengthen regularization (increase L2 penalty), and check for tokens that are IDs/names leaking into features.

**Q12. You fit TF-IDF on the full train+val+test corpus for convenience, then run CV. Validation AUC looks suspiciously good. What happened?**
IDF was computed using the validation/test documents. The feature distribution is shaped by information from the held-out data — mild leakage that inflates validation scores. Strictly fit vectorizer on training only; use `.transform()` on val/test.

**Q13. After deploying, sliding window evaluation shows performance decaying slowly over months. What could explain it?**
Vocabulary drift — new tokens (product names, slang, events) appear in test traffic that weren't in training. The model has no features for them. Mitigations: periodic retraining, use subword/char n-grams for better OOV coverage, or move to a pretrained embedding model.

### Follow-up / probing

**Q14. Compare BM25 to TF-IDF.**
BM25 is an improvement on TF-IDF from information retrieval. Key differences: TF is saturated (TF(t, d) / (TF + k₁·(1 − b + b·|d|/avgdl))) so extra occurrences matter less after a point, and document length is normalized directly. Empirically BM25 beats plain TF-IDF on retrieval. It is still the production first-stage retriever in many search engines.

**Q15. Does scaling TF-IDF by L2 norm change the ranking of cosine similarities?**
No. Cosine similarity is invariant to scalar rescaling of vectors. L2 normalization only makes dot products numerically equal to cosine similarity for convenience.

**Q16. For a text classification task with 100 labels, would you build 100 per-class TF-IDF vocabularies?**
No. Use one global vocabulary; let the classifier weight terms per class. Per-class vocabularies introduce correlated biases, complicate inference, and don't help generalization. Class-balanced regularization is the right tool.

**Q17. Your team wants to replace TF-IDF with embedding averages. When is this a win and when is it not?**
Embeddings win when: semantics matter (synonyms, paraphrases); data is plentiful; a good pretrained model exists. Embeddings lose when: interpretability is required; the task is lexical (spam, phishing strings); training data is small and domain-specific so pretrained embeddings are off-domain; vocabulary is effectively infinite and transfer is weak. Usually the right answer is to run both and see.

---

# 4. Image Features

## 4.1 Motivation & Intuition

### The pre-deep-learning problem

Until 2012, hand-crafted image features *were* computer vision. How do you describe the content of an image in a way a classifier can use?

- **Color histograms** describe the "mood" of the image: a beach photo is blue-dominant, a forest photo green-dominant.
- **HOG (Histogram of Oriented Gradients)** captures shape: pedestrians have a characteristic pattern of edges.
- **SIFT (Scale-Invariant Feature Transform)** finds distinctive keypoints that are robust to rotation, scaling, and lighting — the basis of image stitching and object recognition for over a decade.

Deep learning changed the landscape, but these features still matter:

- On edge devices without GPUs.
- For very small datasets where a CNN overfits.
- In combination with deep features (pretrained CNN features are the modern go-to).
- For understanding *why* deep networks learn what they learn — the first CNN layers look remarkably like hand-crafted filter banks.

### Intuition

- **Color histogram**: chop each pixel into discrete color bins, count how many pixels fall into each bin, and use the counts as features.
- **HOG**: in small patches, compute the dominant gradient directions (edges). A bicycle has a signature pattern of edge angles that's invariant to lighting.
- **SIFT**: find "interesting" points in the image (corners, blobs), then describe each by the local gradient histogram. Match keypoints across images to recognize objects.
- **Pretrained CNN features**: take an off-the-shelf network (ResNet, ViT) trained on ImageNet, strip the final classifier, and use the penultimate layer as a feature vector. It encodes rich semantic information.

## 4.2 Conceptual Foundations

### Color histograms

A digital image is an H × W × 3 array of pixels (R, G, B). A histogram partitions each channel into B bins and counts pixels. Joint 3D histograms have B³ bins, often reduced to marginal per-channel histograms (3·B).

- **Assumption**: color distribution is informative for the task (scene classification, skin detection, logo detection).
- **Breaks when**: color is irrelevant or misleading (grayscale tasks, style-transfer invariant classes).

### HOG — Histogram of Oriented Gradients

Introduced by Dalal & Triggs (2005) for pedestrian detection.

**Pipeline**:
1. Compute image gradients Gx, Gy at each pixel.
2. Compute magnitude M and orientation θ per pixel:
   - M = √(Gx² + Gy²)
   - θ = arctan(Gy / Gx), usually in [0, π) for unsigned gradients.
3. Divide image into small **cells** (e.g. 8×8 pixels).
4. Each cell forms an orientation histogram over, say, 9 bins.
5. Group cells into overlapping **blocks** (e.g. 2×2 cells); normalize each block's concatenated histograms by its L2 norm — this adds robustness to lighting.
6. Concatenate all block descriptors for the final HOG feature vector.

**Invariances**: HOG is approximately invariant to small geometric transformations and local photometric changes. It is *not* invariant to large rotations or scale changes.

### SIFT — Scale-Invariant Feature Transform

A keypoint-based descriptor (Lowe, 1999/2004).

**Four stages**:

1. **Scale-space extrema detection**: build a Gaussian scale-space pyramid, compute differences of Gaussians (DoG) across scales, find local extrema in a 3D (x, y, scale) neighborhood.
2. **Keypoint localization**: reject low-contrast and edge-like points using Hessian analysis.
3. **Orientation assignment**: compute a dominant gradient orientation at each keypoint from a local orientation histogram; rotate the descriptor accordingly — giving rotation invariance.
4. **Descriptor computation**: around each keypoint, partition a 16×16 neighborhood into a 4×4 grid of cells; in each cell compute an 8-bin orientation histogram; concatenate for a 128-dim descriptor.

**Invariances**: scale (via scale-space), rotation (via orientation normalization), and to a lesser extent affine/photometric changes.

### Pretrained CNN features

Today's workhorse for "hand-crafted" image features:

1. Take a network trained on a large labeled dataset (ImageNet, LAION).
2. Remove the final classification head.
3. Forward-pass an image; the activations of the penultimate (or some chosen) layer form a dense vector (e.g. 2048-d for ResNet-50, 768-d for ViT-Base).
4. Use these as features in a downstream linear/MLP/tree model.

**Why it works**: convolutional layers learn progressively more abstract features — from edges (layer 1) to object parts (middle layers) to objects (late layers). The penultimate layer's representation is class-agnostic enough to transfer to many tasks.

**Assumptions**:
- The pretrained task covers sufficient visual variety.
- The new task's image domain is reasonably similar (or the features are fine-tuned).
- The model is appropriately normalized for the input data (same mean/std as pretraining).

### What breaks

- **HOG**: fails on smooth-textured objects with few edges; fails on highly variable rotations.
- **SIFT**: fails on textureless regions; patented for years (now public domain since 2020).
- **Color histograms**: fail when lighting varies substantially; color constancy is an active problem.
- **Pretrained CNN**: domain shift (medical, satellite, microscopy) can ruin features without fine-tuning.

## 4.3 Mathematical Formulation

### Color histograms

Given image I with pixels {p_k}, channel c, and B bins partitioning [0, 255]:

**h_c[b] = |{k : p_k,c ∈ bin_b}|**

Normalize: **h̃_c[b] = h_c[b] / ∑_b' h_c[b']**. Concatenate across channels for a feature vector of length 3B (marginal) or B³ (joint).

### Gradient computation for HOG

Gradients using simple masks [−1, 0, 1]:
- Gx(x, y) = I(x+1, y) − I(x−1, y)
- Gy(x, y) = I(x, y+1) − I(x, y−1)

Magnitude: M = √(Gx² + Gy²). Orientation: θ = atan2(Gy, Gx) mod π (unsigned) or mod 2π (signed).

### HOG cell histogram

For cell C with 9 unsigned orientation bins centered at θ_b = b · 20°, b = 0..8:

**H_C[b] = ∑_{(x,y) ∈ C} M(x, y) · w_b(θ(x, y))**

where w_b is a bilinear interpolation weight between the two nearest bins — avoiding quantization artifacts.

### Block normalization

For a block B = [H_{C1}, H_{C2}, H_{C3}, H_{C4}] (concatenation of four cells):

**v_B = B / √( ||B||² + ε² )**   (L2 norm)

or the "L2-Hys" variant: L2 normalize, clip values at 0.2, renormalize.

### DoG and scale-space for SIFT

The Gaussian scale space is:

**L(x, y, σ) = G(x, y, σ) * I(x, y)**

where G is the Gaussian kernel with standard deviation σ. Difference of Gaussians:

**D(x, y, σ) = L(x, y, kσ) − L(x, y, σ)**

approximates the scale-normalized Laplacian. Keypoints are local extrema of D over (x, y, σ).

### SIFT descriptor

For a keypoint at (x, y, σ) with dominant orientation θ, sample a 16×16 patch aligned to θ and scaled to σ. For each 4×4 sub-cell, compute an 8-bin gradient histogram (weighted by a Gaussian window centered at the keypoint). Concatenate → 4 × 4 × 8 = **128-d descriptor**.

Finally, L2-normalize, threshold values at 0.2 (to reduce influence of very large gradients), re-normalize.

### Pretrained CNN features

A CNN is a cascade of convolutions + non-linearities. For a network f = f_L ∘ ... ∘ f_1, the feature at layer ℓ is:

**φ_ℓ(x) = f_ℓ ∘ f_{ℓ−1} ∘ ... ∘ f_1(x)**

For a typical backbone like ResNet-50, φ_penultimate(x) ∈ ℝ²⁰⁴⁸. Linear probe: train a linear model on φ(x).

### Bag of Visual Words (BoVW) for SIFT

Classical image classification pipeline:
1. Extract SIFT descriptors from all training images.
2. Cluster them with k-means into K visual words (centroids).
3. Represent each image as a histogram over cluster assignments of its SIFT descriptors — a k-dim "bag of visual words."
4. Feed the BoVW vector to an SVM.

## 4.4 Worked Example

### Example 1: Color histogram for skin detection

You want to detect skin-colored regions. Convert RGB to HSV (color constancy). For each pixel, extract Hue h ∈ [0, 180). Build a 1D hue histogram with B = 18 bins.

Training: collect many skin images, pool all skin pixels, build the "skin hue histogram" h_skin. Similarly h_nonskin.

Classification at test time: for a pixel with hue h_p:

P(skin | h_p) ∝ h_skin[bin(h_p)] · P(skin) / (∑_c h_c[bin(h_p)] · P(c))

Threshold the posterior. This naive Bayes approach was the basis of early skin detection and still works surprisingly well.

### Example 2: HOG for a small image

Consider a 64 × 128 image of a pedestrian (standard Dalal–Triggs setup):
- Cells: 8 × 8 pixels → 8 columns × 16 rows = 128 cells.
- Blocks: 2 × 2 cells = 16 × 16 pixels, 50% overlap → 7 × 15 = 105 blocks.
- Each block has 2×2 cells × 9 bins = 36 values.
- Total HOG descriptor: 105 × 36 = **3,780-dim vector**.

The linear SVM trained on these 3,780-dim HOG vectors over thousands of labeled pedestrian images was state of the art for detection for nearly a decade.

### Example 3: Pretrained ResNet-50 feature extraction

A concrete linear-probe classifier for a small custom dataset of 5,000 labeled flower images:

```
1. Load ResNet-50 pretrained on ImageNet.
2. Remove final fc layer; set model.eval().
3. For each image: resize to 224×224, normalize with ImageNet mean/std, forward pass.
4. Collect 2048-dim features into matrix X ∈ ℝ^{5000×2048}.
5. Train logistic regression on X with labels.
6. Validate on held-out split.
```

Typical result on moderate-size datasets: 85-95% accuracy, comparable to fine-tuning a small CNN but orders of magnitude cheaper to iterate.

### Example 4 (numeric): tiny HOG cell

Consider one 2 × 2 patch with gradient magnitudes and orientations:

| (Gx, Gy) | M | θ (degrees) |
|---|---|---|
| (3, 0) | 3 | 0° |
| (0, 4) | 4 | 90° |
| (2, 2) | 2.83 | 45° |
| (−2, 2) | 2.83 | 135° |

Unsigned angles mapped to [0, 180): all four are already in range.

Bin centers (9 bins, width 20°): 0, 20, 40, 60, 80, 100, 120, 140, 160.

- θ=0°, M=3: bin 0 exactly → contributes 3 to bin 0.
- θ=90°, M=4: halfway between bin 80 and bin 100. Bilinear interp: 2 to bin 80, 2 to bin 100.
- θ=45°, M=2.83: between bin 40 and 60. Weight 0.75 to bin 40, 0.25 to bin 60: 2.12 to bin 40, 0.71 to bin 60.
- θ=135°, M=2.83: between bin 120 and 140. Weight 0.25 to bin 120, 0.75 to bin 140: 0.71 to bin 120, 2.12 to bin 140.

Final cell histogram: [3.00, 0, 2.12, 0.71, 2.00, 2.00, 0.71, 2.12, 0]. This is the feature that would be normalized within its block and concatenated with its neighbors to form the final HOG descriptor.

## 4.5 Relevance to ML Practice

### Where used

- **HOG**: still widely used in embedded systems, simple pedestrian detectors, industrial vision.
- **SIFT / SURF / ORB**: image matching, panoramic stitching, SLAM, AR, visual place recognition.
- **Color histograms**: retrieval from large image collections, thumbnail de-duplication, content-based filtering.
- **Pretrained CNN features**: the default starting point for any real-world image classification, retrieval, or clustering task with moderate data.

### When to use hand-crafted features

- Tiny datasets where deep models overfit.
- Compute-constrained devices (mobile, embedded).
- When interpretability is paramount.
- Low-level tasks (feature matching, stitching) where geometric invariances are well-characterized.

### When to use pretrained CNN / ViT features

- Medium-to-large datasets with labeled targets.
- When available pretraining domain matches task.
- When you can afford a GPU at training time.

### Trade-offs

- **Hand-crafted**: fast, tiny memory, interpretable, weak on semantics; require extensive tuning for each task.
- **Pretrained deep**: strong semantics, transfer well, but heavy compute; opaque; can be biased by pretraining data.

### Alternatives

- **End-to-end fine-tuned CNN/ViT**: best accuracy; most compute.
- **Self-supervised embeddings** (SimCLR, DINO, CLIP): state-of-the-art for many tasks with little labeled data.
- **Foundation model APIs**: zero-shot with CLIP or similar for classification/retrieval without any labels.

## 4.6 Common Pitfalls & Misconceptions

- **Using raw RGB histograms in varying lighting**: always convert to HSV or LAB first.
- **Ignoring normalization when using pretrained models**: forgetting to apply the same mean/std as pretraining can degrade features dramatically.
- **Assuming HOG is rotation invariant**: it is not. Only small orientations are tolerated; objects rotated 90° produce a very different descriptor.
- **Treating SIFT like a black-box embedding**: SIFT outputs many descriptors per image, not one. You need a pooling mechanism (BoVW, VLAD, Fisher Vectors) to get a fixed-size representation.
- **Forgetting to `.eval()` the pretrained model**: BatchNorm and Dropout in train mode alter features unpredictably.
- **Using penultimate features when earlier layers would be better**: mid-level layers retain spatial detail and may suit localization tasks better than the penultimate global vector.
- **Assuming pretrained means all-purpose**: a ResNet-50 pretrained on ImageNet may be a poor starting point for satellite, medical, or microscopy images; use a domain-appropriate pretraining dataset.

## 4.7 Interview Preparation — Image Features

### Foundational

**Q1. Walk me through the HOG pipeline.**
(1) Compute image gradients in x and y. (2) Compute per-pixel magnitude and orientation. (3) Split the image into small cells (typically 8×8 pixels). (4) Build an orientation histogram per cell, weighted by magnitude, using bilinear interpolation between adjacent bins. (5) Group cells into overlapping blocks and L2-normalize each block's concatenated histogram. (6) Concatenate all blocks to form the HOG feature vector.

**Q2. Why does SIFT use Difference of Gaussians?**
DoG approximates the scale-normalized Laplacian of Gaussian, which gives strong responses at blob-like structures (local extrema in scale space). Using DoG is computationally cheaper than computing the LoG directly and has almost the same detection quality.

**Q3. What does it mean that CNN features "transfer well"?**
A network trained on one large dataset (ImageNet) learns representations that are useful for other visual tasks with minimal or no re-training. The penultimate-layer features of such a network are broadly semantic, clustering images by object identity even if those objects weren't in the training set. Empirically, linear classifiers over these features outperform hand-crafted features on a wide range of downstream tasks.

### Mathematical

**Q4. Derive the SIFT descriptor dimensionality.**
Around each keypoint, a 16×16 pixel region is sampled, rotated to the dominant orientation, and split into a 4×4 grid of sub-cells (each 4×4 pixels). In each sub-cell, an 8-bin gradient orientation histogram is computed. Total dimensions: 4 × 4 × 8 = 128.

**Q5. Why does HOG apply block-level normalization rather than cell-level normalization?**
Lighting varies across the image, and normalizing across small cells would overfit to local noise. Blocks (typically 2×2 cells) are large enough to capture a stable reference for relative gradient magnitudes, giving robustness to illumination without destroying discriminative information. Overlapping blocks also mean each cell contributes to multiple normalized contexts, averaging out noise further.

**Q6. Given an H × W image, how does HOG descriptor length depend on cell and block size?**
With cell size c and block size b cells (typical b=2) with stride s cells, the number of cells is ⌊H/c⌋ · ⌊W/c⌋ and the number of blocks is approximately ⌊(⌊H/c⌋ − b)/s + 1⌋ · ⌊(⌊W/c⌋ − b)/s + 1⌋. Each block contributes b² × (number of orientation bins) values. Total descriptor = blocks × b² × bins.

**Q7. Why L2-normalize pretrained CNN features before using them as retrieval embeddings?**
Cosine similarity is the natural comparison for high-dimensional CNN features because they vary in magnitude across images in task-irrelevant ways. L2-normalizing makes dot products equal to cosine similarities and removes the magnitude confound, yielding more robust nearest-neighbor retrieval.

### Applied

**Q8. How would you build a product-image search system for an e-commerce catalog of 10M items?**
Use a pretrained or contrastively fine-tuned CNN (e.g., ResNet-50 or a CLIP-style model) to extract L2-normalized feature vectors. Store vectors in an approximate nearest neighbor index (FAISS, ScaNN). At query time, extract a feature from the query image and retrieve the top-K nearest items. For better domain adaptation, fine-tune on an in-domain contrastive loss with pairs of the same product.

**Q9. You have 500 labeled images of a rare medical condition. Which feature extraction approach do you choose?**
A pretrained CNN as a feature extractor with a simple classifier on top. With only 500 images, fine-tuning risks catastrophic overfitting. For medical images, prefer a model pretrained on medical data (MedCLIP, biomedical models) if available; otherwise use a standard ImageNet backbone plus aggressive regularization and augmentation.

**Q10. Pretrained features work poorly on satellite imagery. What to do?**
Retrain on a satellite-domain corpus (BigEarthNet, fMoW), use a self-supervised method like DINO, or use a domain-specific pretrained model. Alternative: gather more labels and fine-tune, using strong augmentations that preserve orientation and scale as appropriate for top-down imagery.

### Debugging & failure modes

**Q11. Your HOG-based pedestrian detector suddenly fails on images taken at dusk. Why?**
HOG is robust to linear illumination changes *within* a block but can fail when gradient magnitudes get very low (dim images): the block normalization can't amplify signal that isn't there, and noise dominates. Fixes: adjust exposure preprocessing, retrain with augmented dusk data, or switch to a deep model that learned invariance during training.

**Q12. Linear probe accuracy on pretrained ResNet features is 50% on your dataset, while others get 95%. What went wrong?**
Classic issues: (1) forgot to apply ImageNet mean/std normalization; (2) used BatchNorm in train mode (forgot `.eval()`); (3) extracted features from the wrong layer; (4) image preprocessing pipeline different from the one during pretraining; (5) domain mismatch between task and ImageNet.

**Q13. SIFT matches produce many false correspondences when matching street scene images. How to robustify?**
Apply Lowe's ratio test: accept a match only if the best match distance is less than 0.7× the second-best match distance, weeding out ambiguous keypoints. Then apply RANSAC on the geometric transform (fundamental or homography) to eliminate remaining outliers.

### Follow-up / probing

**Q14. Why did HOG and SIFT dominate pre-2012 vision but get displaced so quickly?**
They are hand-designed for specific invariances (illumination for HOG, scale/rotation for SIFT) and work well for tasks those invariances match. Deep CNNs learn features directly from data, including task-specific invariances, and they scale gracefully with more data and compute. Once AlexNet (2012) demonstrated a 10-point ImageNet accuracy jump, deep features rapidly became the default.

**Q15. How would you combine HOG and pretrained CNN features?**
Concatenate them into a single vector (after normalizing each) and feed to a model — this sometimes helps for small datasets where the explicit geometric invariances of HOG complement the semantic richness of CNN features. Alternatively, feed each through a separate branch of a model and fuse. In practice, deep features alone usually dominate once you have a few thousand labels.

**Q16. Why can't SIFT produce a single fixed-length image descriptor directly?**
SIFT outputs one descriptor per keypoint, and the number of keypoints varies per image. To get a fixed-length summary you need a pooling mechanism: histogram over k-means clusters (BoVW), VLAD (vector of locally aggregated descriptors), or Fisher vectors. Modern CNN global-pooling features replace this entire pipeline.

**Q17. When would you use a feature extractor rather than fine-tuning?**
(a) Very small dataset where fine-tuning would overfit; (b) constrained compute; (c) fast iteration over many downstream models (e.g., AutoML over extracted features); (d) domain exactly matches pretraining so features are already close to optimal; (e) privacy constraints that prevent gradient updates but allow inference.

---

# 5. Time-Series Features

## 5.1 Motivation & Intuition

### Why specialized time-series features exist

A time-series is a sequence of observations ordered in time: daily sales, hourly sensor readings, per-second server latency. Time introduces properties that cross-sectional data does not have:

- **Temporal ordering**: today depends on yesterday.
- **Autocorrelation**: recent values strongly predict upcoming values.
- **Seasonality**: weekly, monthly, yearly patterns.
- **Trend**: long-run increases or decreases.
- **Non-stationarity**: the distribution itself drifts.

Time-series features are engineered to expose these structural properties so a supervised model can exploit them.

### Concrete examples

- **Demand forecasting**: today's sales depend on last week's sales (lag 7), same day last year (lag 365), and rolling 28-day average.
- **Anomaly detection**: sudden deviation from the past 24-hour pattern signals an outage.
- **Financial prediction**: returns depend on volatility, which is measured as rolling standard deviation.
- **Power systems**: electricity demand has strong daily and weekly cycles best captured by Fourier features.
- **EEG and audio analysis**: wavelet features capture localized frequency patterns.

### Real ML context

Nearly every industrial forecasting system — retail, energy, logistics, cloud capacity planning — uses lagged features, rolling statistics, and calendar features to turn time-series forecasting into a standard supervised learning problem. On that framing, XGBoost and LightGBM with rich time-series features routinely match or beat dedicated statistical methods.

## 5.2 Conceptual Foundations

### Key terms

- **Lag**: value of a series at a previous time step. `x_{t-k}` is the k-step lag.
- **Rolling window**: a fixed-width window ending at time t (strictly before t in most supervised setups).
- **Seasonality**: periodic components (e.g. weekly).
- **Trend**: systematic change over time.
- **Stationarity**: statistical properties (mean, variance, autocorrelation) do not change over time.
- **Differencing**: y_t − y_{t−1}; used to remove a trend.
- **Fourier features**: sinusoidal basis functions with frequencies matching known seasonalities.
- **Wavelet features**: localized oscillatory basis functions capturing both time and frequency.
- **Stationarity tests**: ADF, KPSS.
- **Autocorrelation function (ACF)**: correlation of x_t with x_{t−k}.
- **Partial autocorrelation function (PACF)**: like ACF but conditioned on intermediate lags.

### Components and how they interact

1. **Feature matrix construction**: from a raw series, derive lags, rolling stats, calendar features, Fourier terms — *all leak-free*, i.e., using only values at or before t (or, for forecasting horizon h, at or before t−h).
2. **Model**: a standard regressor or classifier (GBM, MLP) fit on the engineered features.
3. **Prediction and iteration**: for multi-step forecasts, either use direct models per horizon or iterate recursively (feeding predictions back as lags).

### Assumptions

- Lag features assume short-term autocorrelation.
- Fourier features assume approximately stationary periodicities.
- Wavelet features assume localized transient patterns exist.
- Rolling stats assume local stationarity within window.

### What breaks

- **Non-stationarity**: features computed on windows that straddle a regime change will misinform the model.
- **Missing timestamps**: creates incorrect lags ("lag 1" is no longer yesterday if yesterday's data is missing).
- **Leakage**: careless rolling window construction includes future information.
- **Multiple related series**: ignoring cross-series signals (hierarchies, correlated products) loses information.

## 5.3 Mathematical Formulation

### Lag features

For horizon h and chosen lag set L = {1, 7, 14, 28, 365, ...}:

**x_t^(ℓ) = x_{t−ℓ}** for ℓ ∈ L

Use lags whose minimum is at least h (otherwise leakage for horizon h). For h = 7, lags must be ≥ 7.

### Rolling statistics

For a window of length W:

**mean_t = (1/W) ∑_{k=1}^{W} x_{t−k}**
**std_t = √((1/(W−1)) ∑_{k=1}^{W} (x_{t−k} − mean_t)²)**

Note: summation runs from k=1, excluding x_t itself.

Exponentially weighted: **EWMA_t = α·x_{t−1} + (1−α)·EWMA_{t−1}**.

### Seasonality — Fourier features

To encode a seasonal cycle of period P with K harmonics, add 2K features:

**s_k(t) = sin(2π k t / P)** for k=1,...,K
**c_k(t) = cos(2π k t / P)** for k=1,...,K

Any periodic function of period P can be written (approximately) as a finite Fourier series:

**f(t) ≈ a₀ + ∑_{k=1}^{K} (a_k cos(2πkt/P) + b_k sin(2πkt/P))**

So Fourier features give the model a linear basis rich enough to express any periodic pattern up to that harmonic order. Use K=3 for weekly (good enough for most shapes), K=5-10 for yearly when the shape is complex.

### Fourier transform (global frequency analysis)

Discrete Fourier transform of signal x of length N:

**X[k] = ∑_{n=0}^{N−1} x[n] · e^{−2πikn/N}**

Magnitude |X[k]| is the strength of frequency f_k = k/N cycles per sample. Use: compute FFT of rolling windows, feed top-frequency magnitudes as features.

### Wavelet transform (time-localized frequency analysis)

Fourier is global in time; a sine wave centered at frequency f exists forever. Wavelets are *localized* — they have support only in a finite time window.

A wavelet ψ(t) is a zero-mean function. The continuous wavelet transform of x(t) at scale a and shift b is:

**W(a, b) = (1/√a) ∫ x(t) · ψ((t − b)/a) dt**

Scale a controls the frequency (larger a → lower frequency); shift b controls position. The output is a 2D time-frequency map.

**Why this is useful**: for non-stationary signals with transient bursts (earthquakes, heartbeats, turbine vibrations), FFT smears the burst across all time while wavelets localize it. Classical wavelet families: Haar, Daubechies, Morlet.

Discrete wavelet transform (DWT) via filter banks decomposes a signal of length N into approximation and detail coefficients at multiple scales in O(N) time — comparable to FFT in practice and often more informative.

### Differencing

For trend removal: **d_t = x_t − x_{t−1}**. Seasonal differencing: **d_t = x_t − x_{t−P}** for period P.

Differenced series is often stationary, making downstream modeling simpler.

### Autocorrelation coefficient

For lag k:

**ρ(k) = Cov(x_t, x_{t−k}) / Var(x_t)**

Empirical ACF plot reveals characteristic scales (e.g., daily, weekly peaks). The first lag where ρ drops close to zero is a useful lag-depth choice for lag features.

## 5.4 Worked Example

### Series: daily retail sales

Synthetic data: trend + weekly seasonality + noise. Values for two weeks (day index t, sales x_t):

| t | x_t |
|---|-----|
| 1 | 100 |
| 2 | 110 |
| 3 | 115 |
| 4 | 120 |
| 5 | 130 |
| 6 | 150 |
| 7 | 160 |
| 8 | 105 |
| 9 | 118 |
| 10 | 122 |
| 11 | 130 |
| 12 | 135 |
| 13 | 155 |
| 14 | 170 |

### Task: forecast x_15 (horizon h = 1).

### Lag features (minimum lag = h = 1)

At t = 15: lag_1 = x_{14} = 170, lag_2 = x_{13} = 155, lag_7 = x_{8} = 105, lag_14 = x_1 = 100.

### Rolling statistics

Using a 7-day window ending at t=14 (strictly before 15? no — window is [t−7, t−1] = [8, 14] inclusive):
- mean = (105+118+122+130+135+155+170)/7 = 935/7 ≈ 133.57
- std ≈ 23.28
- max = 170, min = 105

### Fourier features for weekly seasonality (P = 7, K = 2)

For t = 15:
- sin(2π·1·15/7) = sin(2π·15/7) = sin(4π + 2π/7) = sin(2π/7) ≈ 0.782
- cos(2π·15/7) ≈ 0.623
- sin(2π·2·15/7) = sin(4π·15/7) = sin(60π/7) ≈ 0.975
- cos(2π·2·15/7) ≈ −0.223

### Calendar features (if we knew the anchor date)

Day-of-week (Mon=1, Sun=7), month, is-weekend, is-holiday, etc. These complement Fourier features — often both are used simultaneously; trees like sparse calendar flags, linear models like smooth Fourier.

### Building the feature row for t = 15

[lag_1=170, lag_2=155, lag_7=105, lag_14=100, roll7_mean=133.57, roll7_std=23.28, roll7_max=170, sin1=0.782, cos1=0.623, sin2=0.975, cos2=-0.223, day_of_week=..., ...]

### Training setup

Compute these features for every valid t (starting where lag_14 exists — so t ≥ 15 for full features). Target is x_t. Train a GBM. For forecasting multiple days ahead: either build a separate model per horizon or iterate.

### Wavelet feature for anomaly detection

Suppose we observe a sudden drop on day 10 that was not there before. A rolling mean smooths it out; FFT spreads it across frequencies. A wavelet transform at a short-scale wavelet (Haar at scale 2) would produce a large coefficient near t=10, providing a direct "transient" feature. Models then include `max_wavelet_coeff_in_last_7_days` to flag anomalies.

## 5.5 Relevance to ML Practice

### Where used

- **Retail/CPG forecasting**: LightGBM with lags, rolling, calendar + Fourier is the industry standard.
- **Energy load forecasting**: same plus weather features.
- **Anomaly detection**: rolling stats, wavelet, EWMA-based residuals.
- **IoT and predictive maintenance**: wavelet and FFT features of vibration data.
- **Financial returns**: volatility (rolling std), momentum (rolling returns), mean reversion proxies.

### When to use

- Structured tabular time-series problems with moderate to large history.
- When a GBM-based pipeline already exists; lag + rolling features plug in cleanly.
- When interpretability of which lags drive predictions is required.

### When not to use

- Extremely high-frequency data with complex temporal dependencies (often better handled by sequence models like RNN/Transformer/Temporal Convolutional Networks).
- Tasks requiring cross-series inductive biases (classic methods per series ignore global structure).
- Very short series (< lag depth) — no history to lag from.

### Alternatives

- **Classical methods**: ARIMA, ETS, Prophet.
- **Deep sequence models**: LSTM, Transformer, TFT (Temporal Fusion Transformer), N-BEATS, DeepAR.
- **State-space models**: Kalman filters, SSMs, structural time series.
- **Hybrid**: deep + classical features.

### Trade-offs

- **Bias-variance**: more lags + higher-order Fourier reduces bias but inflates variance; cross-validate carefully using time-series splits.
- **Leakage risk**: enormous in time-series; walk-forward CV is required.
- **Compute**: cheap; trees on time-series features run in seconds on millions of rows.
- **Interpretability**: very high — each lag, each window, each Fourier term has a clear meaning.

## 5.6 Common Pitfalls & Misconceptions

- **Random K-fold CV on time-series**: shuffles future into training folds — catastrophic leakage. Use time-series splits (expanding or rolling).
- **Computing rolling stats including t**: `rolling(window=7).mean()` in pandas includes the current observation by default. Shift by 1 to exclude it (`.shift(h)` for horizon h).
- **Forgetting to shift lags by the forecasting horizon**: if you forecast 7 days ahead, the smallest available lag is 7, not 1.
- **Using Fourier on non-periodic patterns**: adds noise features that overfit.
- **Ignoring irregular timestamps**: a "lag 1" on an irregular time grid is ambiguous; resample or interpolate first.
- **Static models on drifting series**: retrain regularly, use rolling windows for feature statistics.
- **Mixing multiple series without per-series context**: the model can't distinguish which series a feature row belongs to; include a series-ID feature or per-series target encoding.
- **Confusing differencing with detrending by regression**: differencing removes unit roots; regression detrending fits a functional form. They are not equivalent.

## 5.7 Interview Preparation — Time-Series Features

### Foundational

**Q1. What is the minimum lag you can use when forecasting 7 days ahead, and why?**
Lag 7. At prediction time for day t+7, the latest observation you have is day t. Lag k features are x_{t+7−k}. For these to be available, k ≥ 7. Using lag 1 would require the value of day t+6, which you don't have yet at t.

**Q2. Why do we add Fourier features alongside calendar features like day-of-week?**
Fourier features provide a smooth, continuous periodic basis — enabling linear models and smooth regularization. Calendar features like day-of-week are categorical and allow trees to make sharp, discontinuous splits. Both forms capture weekly seasonality but in complementary ways; in practice, including both often outperforms either alone.

**Q3. Explain the difference between Fourier and wavelet features.**
Fourier features decompose a signal into global sinusoids — they tell you *which* frequencies exist but not *when* they occurred. Wavelets use localized basis functions, producing a time-frequency map that reveals when each frequency is active. Fourier suits stationary periodicities; wavelets suit transient or non-stationary phenomena.

### Mathematical

**Q4. Derive the formula for a K-harmonic Fourier feature set for period P.**
We approximate a periodic function f(t) with period P by a truncated Fourier series f(t) ≈ a₀ + ∑_{k=1}^{K} (a_k cos(2πkt/P) + b_k sin(2πkt/P)). The features we add are {sin(2πkt/P), cos(2πkt/P) : k = 1..K}, giving 2K columns. Linear regression on these features recovers a_k and b_k.

**Q5. Why does differencing a series with a linear trend produce a stationary series?**
If x_t = μt + ε_t with ε_t stationary, then d_t = x_t − x_{t−1} = μ + (ε_t − ε_{t−1}). The differenced series has constant mean μ and a new noise process; the deterministic linear trend is removed. This is the elementary unit-root argument behind ARIMA.

**Q6. Compute the autocorrelation at lag 1 for a series generated by x_t = 0.7 x_{t−1} + ε_t with ε ~ N(0, 1).**
This is an AR(1) process with ϕ = 0.7. Its theoretical ACF is ρ(k) = ϕ^k. So ρ(1) = 0.7, ρ(2) = 0.49, and so on — decaying geometrically. The PACF would be 0.7 at lag 1 and zero thereafter, a characteristic AR(1) signature.

**Q7. Show why FFT cannot capture a transient burst as cleanly as a wavelet transform.**
A Dirac-like burst δ(t − t₀) has a Fourier transform with constant magnitude across all frequencies: FFT spreads the burst uniformly across the spectrum. A wavelet centered near t₀ produces a large coefficient at that location and small coefficients elsewhere — it localizes both in time and frequency, revealing exactly when the burst happened.

### Applied

**Q8. Design a feature set for forecasting hourly electricity demand 24 hours ahead.**
Lags: 24 (same hour yesterday), 24×7 (same hour last week), and 1-year lag (same hour last year). Rolling 24-hour, 7-day, 30-day means and stds (shifted by 24 hours). Calendar: hour-of-day, day-of-week, month, is-holiday. Fourier: K=5 for daily and K=3 for weekly. Weather features at matching hour (forecast values). Per-zone series ID (if multiple zones).

**Q9. Your forecasting model overfits to training data with near-perfect training but poor test performance. What's wrong?**
Time-series leakage is the prime suspect: rolling features that don't exclude the current value; shuffled CV splits; features built on full-dataset statistics. Rebuild with strict shifting and walk-forward CV. Also consider: too many lags (high variance), not enough regularization, or non-stationary regime change between train and test.

**Q10. How would you handle the case where your multivariate time-series has missing timestamps in some series?**
Resample everything to a common regular grid via interpolation (linear, forward-fill) or zero-fill with an explicit "was-missing" indicator feature. Compute lags on the regularized grid. If missing data is frequent or meaningful, treat "missing" as informative and add a binary missingness feature to preserve the signal.

### Debugging & failure modes

**Q11. Your model performs great in backtest but fails in live deployment. What's likely wrong?**
(1) Training-serving skew in feature pipelines (batch builder vs online builder diverge). (2) Backtest used information not available at prediction time (subtle leakage). (3) Concept drift since the backtest period. (4) Features include target-derived quantities (e.g., last-week's mean was computed using labels you wouldn't have had). Audit the online feature computation by replaying production logs through it.

**Q12. You get excellent AUC when predicting anomalies using Z-scores of the raw values vs. a rolling mean. But when deployed, the model misses most real anomalies. Why?**
During training, the rolling mean at each row probably included the current value (or future values if not shifted). At inference, only past values are available, so Z-scores look different. Correct definition: Z_t = (x_t − mean_{t−1, ..., t−W}) / std_{t−1, ..., t−W}.

**Q13. Your forecast is systematically biased low. What could cause this?**
Several possibilities: (1) trend continues beyond the training period and a purely lag-based model can't extrapolate it; (2) asymmetric loss or skewed target without proper handling; (3) missing holiday/promotion features causing the model to predict "average" behavior; (4) insufficient history for long-period lags like annual.

### Follow-up / probing

**Q14. When would you prefer direct multi-horizon forecasting over recursive forecasting?**
Direct: train a separate model (or multi-output model) for each horizon. Avoids error compounding but multiplies training cost. Recursive: predict one step, feed prediction in as input for next. Simpler but errors compound. Prefer direct for long horizons or when the target's characteristics change across horizons (e.g. daily vs. weekly dynamics).

**Q15. Explain partial autocorrelation and why it matters for lag selection.**
PACF at lag k is the correlation between x_t and x_{t−k} *after removing* the linear effect of all intermediate lags. For an AR(p) process, PACF is nonzero for the first p lags and approximately zero afterward. So PACF directly suggests how many lag features are "effective." In practice, ACF/PACF plots guide which lags to include before ML.

**Q16. How does LightGBM benefit from lag features compared to a classical ARIMA approach?**
LightGBM can handle multiple time series in one model (with series ID), accept exogenous features (weather, promotions, calendar) natively, model non-linear interactions, and does not require stationarity. It also scales to billions of rows. Classical ARIMA is per-series, assumes linearity and stationarity, and cannot leverage many features.

**Q17. What are the downsides of using wavelet features in production?**
Wavelet transforms require choosing a wavelet family and scale, which are problem-dependent hyperparameters. The resulting feature vector can be large (many scales × many times). Implementation has more moving parts than FFT. In non-research production pipelines, practitioners often stick with simpler engineered features unless wavelets clearly help in offline evaluation.

---

# 6. Geospatial Features

## 6.1 Motivation & Intuition

### Why location matters

Location is one of the most information-dense features in many problems. Where a user lives, where an accident happens, where an ad is served — all carry signal that cannot be captured by generic features:

- **Real estate**: an apartment's price depends enormously on its neighborhood, transit access, and distance to the city center.
- **Ride-hailing**: ETA predictions depend on the specific origin-destination geography, not generic "distance."
- **Fraud**: logins from a new country can be suspicious; the geographic pattern of the user's historical locations is itself a feature.
- **Public health**: disease spread tracks geography, population density, and mobility.
- **Supply chain**: shipping times depend on origin-destination routes.

### Intuition before math

- **Distance features**: how far two points are (straight line, along roads, along public transit).
- **Spatial aggregations**: summarize what's around a point — restaurants within 500m, average home price in the neighborhood.
- **Density measures**: how concentrated items are in an area — crime density, POI density, population density.

### Real ML context

Every mapping product, logistics platform, and location-based service uses a combination of these features. Modern systems complement them with learned representations (location embeddings from user mobility), but the hand-engineered geographic features remain core because they encode physics (distance, density) the model would otherwise have to rediscover from data.

## 6.2 Conceptual Foundations

### Key terms

- **Coordinate system**: latitude/longitude (WGS84), or projected coordinates (UTM, Web Mercator).
- **Haversine distance**: great-circle distance on a sphere.
- **Euclidean approximation**: flat-earth distance; accurate over small regions.
- **Geohash**: a string-based spatial index dividing the world into nested rectangular cells.
- **H3 / S2**: hexagonal and spherical grid systems used at Uber and Google.
- **Polygon**: a closed geographic region (neighborhood, ZIP code).
- **Point-in-polygon**: determining which polygon contains a point.
- **Spatial join**: joining point data to polygons by containment.
- **KDE (Kernel Density Estimation)**: smoothed density surface.
- **Isochrones**: regions reachable within a travel time budget.

### Components and interactions

- **Raw coordinates** (lat, lon) → too raw for most models; rarely used directly because they're non-linear in effect (the same coordinate delta means different things at different latitudes).
- **Discretize** by H3 / geohash / administrative boundaries → use as categorical features.
- **Pairwise distances** to anchor points (city center, nearest hospital, user's home).
- **Local aggregations** within radius or cell (count, mean).
- **Density features** from KDE or cell counts normalized by area.

### Assumptions

- **Distance matters monotonically** for the phenomenon (usually but not always true — e.g., price can increase with distance from a landfill).
- **Distances are Euclidean or great-circle** — ignoring road networks works for some problems (radio coverage) but not others (ETAs).
- **The coordinate system has been correctly handled** (projection, datum).

### What breaks

- **Euclidean on lat/lon at high latitudes**: grossly distorts real distances.
- **Ignoring barriers**: a park across a river is "close" by straight-line but far by road.
- **Stale polygons**: neighborhood boundaries change; using outdated GeoJSON files introduces silent errors.
- **Ignoring uncertainty**: GPS error is ~5m in the open, 50m+ in urban canyons; models that treat coordinates as exact can overfit.

## 6.3 Mathematical Formulation

### Haversine distance

For two points (lat₁, lon₁), (lat₂, lon₂) with radius of Earth R ≈ 6,371 km:

**a = sin²((lat₂ − lat₁)/2) + cos(lat₁)·cos(lat₂)·sin²((lon₂ − lon₁)/2)**
**c = 2 · arctan2(√a, √(1−a))**
**d = R · c**

This is the great-circle distance — the shortest distance on the surface of a sphere. Accurate to within 0.5% because Earth is an oblate spheroid, not a sphere. For higher accuracy use Vincenty's formula.

### Euclidean approximation (for small distances)

At mid-latitudes:
- **1° of latitude ≈ 111 km**
- **1° of longitude ≈ 111 km · cos(lat)**

For small distances (< 50 km away from equator), Euclidean in projected meters is fine:

**d ≈ √((Δlat · 111,000)² + (Δlon · 111,000 · cos(lat))²)**

### Manhattan / taxicab distance

Relevant for grid-based cities:

**d = |Δx| + |Δy|**

### Density estimation — KDE

Given points {x₁, ..., xₙ}, the KDE at a location x with bandwidth h is:

**f̂(x) = (1/(n·h²)) ∑ᵢ K((x − xᵢ)/h)**

where K is a kernel (Gaussian, Epanechnikov, uniform). The bandwidth h controls smoothness: too small → spiky, too large → flat.

### Spatial aggregation

For a point at x, within radius r:

**feature(x) = (1/|N_r(x)|) ∑_{y ∈ N_r(x)} value(y)**

where N_r(x) is the set of all points within distance r of x. For counts, replace value with 1.

### Geohash

A string encoding that alternates latitude and longitude bits. Prefix length controls precision:
- 4 chars → ~20 km cell
- 5 chars → ~5 km cell
- 7 chars → ~150 m cell

Cells are rectangular in lat/lon space; this distorts cell sizes across latitudes.

### H3 hexagons

H3 uses a hierarchy of hexagonal cells. Level 9 ≈ 0.1 km² per cell; level 12 ≈ 0.3 hectares. Hexagons have the advantage that all neighbors are equidistant (unlike squares where diagonal neighbors are farther), which matters for graph construction and KDE.

### Polygon containment

A point (p_x, p_y) is inside polygon P if a ray from the point crosses the polygon's edges an odd number of times (ray-casting algorithm), runs in O(|edges|). Spatial indexes (R-tree) speed this up for large polygon sets.

## 6.4 Worked Example

### Scenario

Predict daily demand for a bike-share station from station properties and local geography. Stations have (lat, lon). We also have:
- Locations of all subway stops.
- Administrative neighborhood polygons.
- Historical ride-origin points.

### Feature 1: distance to nearest subway

For station s with coordinates (lat_s, lon_s):

d_subway(s) = min over all subway stops i of Haversine((lat_s, lon_s), (lat_i, lon_i))

A station 100 m from a subway stop will likely have higher demand than one 1 km away. Tree models can learn breakpoints like "is within 250m of subway."

### Feature 2: distance to city center

d_center(s) = Haversine((lat_s, lon_s), (lat_center, lon_center))

Captures the gradient from downtown to suburbs.

### Feature 3: subway stop count within 500 m

Count the number of subway stops i where Haversine < 500 m. A station with 3 nearby subway stops has different dynamics than one with 0.

### Feature 4: neighborhood one-hot

Spatial-join the station's coordinate into a neighborhood polygon. Use the neighborhood as a categorical feature (or mean-encode by historical demand).

### Feature 5: ride-origin density (KDE)

Pre-compute a KDE surface from historical ride-origin points. Sample the KDE value at each station's location. A station in a hot zone (lots of nearby ride starts historically) will have different demand than one in a cold zone.

### Feature 6: H3 cell aggregations

Bucket all points into H3 cells at level 9. For each cell, compute historical origin count. Assign each station to its H3 cell's count. Provides a discrete complement to KDE.

### Numeric example — Haversine computation

Station A: (40.7580, −73.9855) — Times Square, NYC
Station B: (40.7061, −74.0087) — Financial District, NYC

Δlat = (40.7061 − 40.7580)·π/180 = −0.000906 rad
Δlon = (−74.0087 + 73.9855)·π/180 = −0.000405 rad
lat₁ = 40.7580·π/180 ≈ 0.7113 rad; lat₂ ≈ 0.7104 rad

a = sin²(−0.000906/2) + cos(0.7113)·cos(0.7104)·sin²(−0.000405/2)
  ≈ 2.05e−7 + 0.756·0.756·4.1e−8
  ≈ 2.05e−7 + 2.35e−8
  ≈ 2.29e−7
c ≈ 2·arctan2(√2.29e−7, √(1−2.29e−7)) ≈ 2·√2.29e−7 ≈ 9.57e−4 rad
d = 6371·9.57e−4 ≈ **6.10 km**

Manually: Times Square to Financial District is about 6 km — good.

### Building the full row

[distance_to_subway_m=120, distance_to_center_m=500, subways_within_500m=2, neighborhood=Midtown, kde_density=0.87, h3_cell_origin_count=1245, ...]

Feed to a GBM; inspect SHAP values to confirm distance and density features are highly predictive.

## 6.5 Relevance to ML Practice

### Use cases

- **Ride-hailing / delivery**: ETA, demand prediction, pricing, dispatch.
- **Real estate**: price prediction, investment screening.
- **Advertising**: geo-targeting, local ad relevance.
- **Public health**: disease surveillance, resource allocation.
- **Retail site selection**: where to put the next store.
- **Insurance**: risk assessment based on address and surroundings.
- **Fraud**: device or transaction geography as a signal.

### When to use

- When the target has a strong spatial component.
- When you have enough geographic context (POIs, polygons, historical events).
- When the model is interpretable and you need to explain predictions.

### When not to use

- When target is purely individual-level (no spatial dependence), simple raw coordinates may just be noise.
- When the data is not privacy-safe — precise coordinates can identify individuals.

### Alternatives / complements

- **Location embeddings**: train a neural net to produce embeddings of locations based on co-visitation patterns.
- **Graph-based**: treat map nodes as a graph; use GNNs (covered in graph features).
- **Pretrained map representations**: e.g., OpenStreetMap-based models.

### Trade-offs

- **Interpretability**: hand-engineered geo features are highly interpretable; embeddings are not.
- **Accuracy**: embeddings capture subtle co-occurrence patterns hand features miss.
- **Privacy**: raw lat/lon is identifiable; H3 cells at coarse resolution are safer.
- **Computation**: KDE over millions of points requires spatial indexes; H3 is O(1) lookups but coarse.

## 6.6 Common Pitfalls & Misconceptions

- **Euclidean distance on lat/lon**: distorts at high latitudes and globally. Use Haversine or project to UTM first.
- **Ignoring Earth curvature** for very long distances (flights, shipping routes). Use great-circle or geodesic.
- **Treating H3 cell index as a number**: the index is not ordinal; use it as a categorical, or use the cell's centroid coordinates.
- **Leakage via same-day nearby events**: a feature like "number of accidents within 500m today" leaks if the target itself contributes.
- **Stale POI databases**: if you use OpenStreetMap data from 2019 to train and 2025 POIs to serve, you have skew.
- **Ignoring map-matching error**: GPS traces off-road cause spurious distances; snap to road network first.
- **Assuming uniform population density**: KDE on raw event counts confounds density with exposure; normalize by population.
- **Overlooking projection choice**: different projections distort area and angles differently; choose based on the analysis (equal-area for densities, conformal for angles).

## 6.7 Interview Preparation — Geospatial Features

### Foundational

**Q1. Why is Haversine preferred over Euclidean for computing distances on Earth?**
Earth is approximately spherical, not flat. Euclidean distance computed directly on (lat, lon) is meaningless in units (degrees aren't distances) and distorts badly because 1° of longitude covers very different ground distances at different latitudes. Haversine computes the great-circle distance on a sphere, returning proper kilometer distances globally accurate to within 0.5%.

**Q2. What is a geohash and how is it used?**
A geohash is a string encoding that represents a bounding rectangle on the globe; longer strings mean smaller rectangles. Geohashes give a cheap, prefix-based hierarchy for spatial indexing, allowing fast lookups, joins, and aggregations by prefix. Common uses: bucket points to compute per-cell statistics, approximate nearest-neighbor candidate generation.

**Q3. What's the point of density features?**
They tell the model how concentrated phenomena are near a query point. A restaurant site's success depends on foot traffic density; a fraud signal depends on event density near a user. Density captures the "neighborhood context" that single coordinates cannot.

### Mathematical

**Q4. Derive a back-of-the-envelope flat-Earth distance formula valid near a reference latitude.**
At latitude ϕ, 1° of latitude ≈ 111 km (constant), 1° of longitude ≈ 111·cos(ϕ) km. So for two points with small separation (Δϕ, Δλ):

d ≈ √((Δϕ · 111)² + (Δλ · 111 · cos(ϕ))²) km

Valid for distances much smaller than Earth's radius; error grows with distance and with how much latitude changes.

**Q5. Explain bandwidth choice in KDE.**
Bandwidth h controls smoothness: small h produces spiky estimates that overfit; large h over-smooths and loses structure. Data-driven choices: Silverman's rule (h ≈ 1.06·σ·n^(−1/5) per dimension, rough but fast), cross-validated log-likelihood, or plug-in estimators. Often different bandwidths in x and y for anisotropic data.

**Q6. A KDE surface is smoothly continuous; H3 cell counts are piecewise constant. Which is appropriate for linear models and why?**
Both can work. H3 cell counts as categorical produce piecewise-constant features — easy for trees, but a linear model treats each cell as a separate indicator, risking overfitting with many cells and no spatial smoothing. KDE values give continuous signal that linear models can use with one coefficient. For linear models, KDE or Fourier-spatial features are typically preferred; for trees, cell IDs work well.

**Q7. Why are hexagons (H3) preferred over squares (geohash) for some applications?**
In a square grid, the nearest neighbors along the cardinal directions (edges) are closer than the diagonal neighbors — the "neighborhood" is anisotropic. In a hexagonal grid, all six neighbors are equidistant, giving isotropic behavior. This matters for KDE on cells, for graph algorithms that treat cell adjacency, and for visualizations where uniform neighborhoods are desirable.

### Applied

**Q8. Design geo features for an ETA prediction model in a ride-hailing app.**
Origin and destination H3 cells (categorical); Haversine OD distance; road-network distance and estimated travel time from a routing service; average recent speed in each cell; time-of-day interactions with cell; polygon memberships (airport, stadium) for OD; nearby traffic incident counts within r of OD line; historical residuals (what's the recent ETA error in this OD cell pair).

**Q9. You're predicting crime risk per city block. What geospatial features would you build?**
Per-block historical crime count by type (over leak-free windows); KDE of crime events at multiple bandwidths; distance to nearest police station, school, transit hub; POI density (bars, ATMs); population density per block; adjacency features (average crime in neighboring blocks); polygon memberships for neighborhoods and police precincts.

**Q10. How would you add privacy to a model using user home locations?**
Quantize to H3 level 7 or larger (≈ 1 km² cells) rather than raw coordinates, never persist raw lat/lon, add Laplace noise for differential privacy, or use location embeddings trained with privacy constraints (federated learning, per-user DP-SGD). If the ML task requires high precision, restrict that data to audited environments.

### Debugging & failure modes

**Q11. Your distance-to-nearest-hospital feature is surprisingly weak in a health-outcome model. Why?**
Several possibilities: the feature is measured as Euclidean when road access matters (travel time is the causal variable); hospital locations are stale or incomplete; the feature has a floor at zero but the real gradient happens at larger scales (km); the target is already conditioned on hospital access (patients already at hospitals). Re-compute with road-network travel time; audit the dataset's hospital coverage; consider interacting the feature with rurality.

**Q12. You used (lat, lon) directly as two features in a linear model. Performance is bad. What's wrong?**
Linear combinations of lat and lon can only produce linear "wave" patterns across geography — a very limited family. Better: discretize (H3/geohash), encode to distance-to-anchor features, or use Fourier features in 2D. Linear models cannot learn complex spatial patterns from raw coordinates.

**Q13. You see a sharp distribution shift in geospatial features after a calendar quarter. Why might that happen?**
Several causes: OSM or POI dataset was updated (new or removed points changed distances and counts); a city's GeoJSON boundaries were revised; an upstream service changed geocoding defaults; a new product launched in a specific region shifting event distributions. Version your geo reference data and rerun features against a fixed snapshot.

### Follow-up / probing

**Q14. How do you handle user locations that oscillate wildly (e.g., a mobile device reporting GPS at ±100m every second)?**
Aggregate over a time window: median or trimmed-mean of positions, Kalman smoothing, or map-matching to road / POI candidates. The result is a more stable location estimate and a confidence indicator that downstream features can consume.

**Q15. Why might you prefer travel-time contours (isochrones) to straight-line distances as features?**
Because real human movement follows networks — roads, transit, stairs. A beach 2 km across a river is effectively far. Isochrones represent "all locations reachable in ≤ t minutes" and capture the true accessibility relevant for most human-centric models.

**Q16. Compare H3 cell features with learned location embeddings.**
H3: interpretable, deterministic, no training needed, compositional (cell adjacency, parent-child); limited in semantic richness. Location embeddings: learned from data (co-visitation, movement patterns), capture latent concepts (coffee-shop districts, tourist zones), but opaque and require enough data to train. Best of both worlds: use H3 IDs as tokens, learn embeddings over them.

**Q17. What are the statistical pitfalls of using area-based aggregations (e.g., ZIP code means) as features?**
The **modifiable areal unit problem** (MAUP): results change when the unit of aggregation changes. Large ZIP codes average out local variation; small ZIPs have noisy counts. Always consider multiple scales, add count features alongside means (so the model knows reliability), and be cautious about inferring individual-level behavior from area-level features (ecological fallacy).

---

# 7. Graph Features

## 7.1 Motivation & Intuition

### Why graphs require their own feature engineering

Many data sources are naturally relational:

- **Social networks**: users connected by friendships, follows, mentions.
- **Knowledge graphs**: entities connected by relations ("Steve Jobs — founded — Apple").
- **Transaction networks**: accounts connected by money flows.
- **Web hyperlinks**: pages connected by links.
- **Biology**: proteins connected by interactions, genes by regulation.
- **Recommendation**: users and items connected by interactions.

A user is not just their attributes; their position in the network — who they know, how central they are, who their friends know — carries enormous signal. Graph features extract that positional information into features usable by any model.

### Intuition

- **Degree**: how many neighbors does this node have? High-degree nodes are influential (social networks) or hubs (networks).
- **Centrality**: different notions of "importance" — closeness to everyone (closeness), on many shortest paths (betweenness), connected to important others (eigenvector/PageRank).
- **Clustering coefficient**: how tightly knit is this node's neighborhood? Friends of friends that are also friends indicate a cohesive group.
- **PageRank**: a random-walk based centrality measure; the foundation of early Google search.

### Real ML context

- **Fraud rings**: detected by unusual connectivity patterns (dense subgraphs of new accounts).
- **Bot detection**: unusual centrality and neighbor features vs. human users.
- **Recommendation**: personalized PageRank scores users' affinity for items.
- **Network effects in growth models**: degree and triangle counts predict retention.
- **Drug discovery**: molecular graphs where node/edge features predict properties.

## 7.2 Conceptual Foundations

### Key terms

- **Graph**: G = (V, E) where V is nodes, E is edges.
- **Directed vs undirected**: edges have a direction or not.
- **Weighted**: edges carry numerical weights.
- **Adjacency matrix A**: n × n, A_{ij}=1 if (i,j) ∈ E else 0 (or weight).
- **Degree**: |neighbors(v)|. For directed graphs, in-degree and out-degree.
- **Neighborhood**: set of nodes connected to v.
- **Path**: sequence of adjacent edges.
- **Shortest path**: minimum-hop (or min-weight) path.
- **Triangle**: three nodes all mutually connected.
- **Clustering coefficient**: fraction of possible triangles at a node that actually exist.
- **Centrality**: generic term for node-importance measures.
- **PageRank**: stationary distribution of a random walk with teleportation.
- **Betweenness**: fraction of shortest paths between all node pairs that pass through v.
- **Closeness**: reciprocal of mean distance from v to all others.
- **Eigenvector centrality**: a node is important if connected to important nodes; leading eigenvector of A.
- **Community / cluster**: a subset of nodes more densely connected internally than externally.

### Components in a pipeline

1. **Graph construction**: often non-trivial — how to go from raw data (logs, co-purchases) to an edge list. Threshold, time-window, symmetrize.
2. **Feature computation**: for each node (or edge), compute scalar features. Some are O(|V| + |E|), some are O(|V|·|E|) or worse.
3. **Feature merging**: join to the node ID in the tabular dataset.
4. **Model**: standard tabular model (GBM) or graph neural network (separate topic).

### Assumptions

- The graph is **relevant** to the task.
- **Edges are meaningful** (same relation type, comparable weights).
- The graph is **representative** (no huge missing subgraph).
- For transductive features like PageRank, **node identities are stable** between training and inference.

### What breaks under violations

- **Noisy edges** mix in signal and hurt centrality.
- **Incomplete graph** biases all features downward.
- **New nodes at inference time** have no meaningful features (cold-start).
- **Time inconsistencies**: computing centrality on the *full* graph for training examples leaks future connections that didn't exist when the label was generated.

## 7.3 Mathematical Formulation

### Degree

**deg(v) = |{u : (u, v) ∈ E}|**. In matrix form, deg = A · 1.

For directed graphs:
- in-deg(v) = ∑ᵢ A_{iv}
- out-deg(v) = ∑ⱼ A_{vj}

### Clustering coefficient (local)

For node v with k = deg(v) neighbors:

**C(v) = (2 · T(v)) / (k · (k−1))**

where T(v) is the number of triangles at v (pairs of neighbors that are themselves connected). Interpretation: the fraction of the possible k(k−1)/2 pairs of neighbors that are actually connected.

Global (network-wide) clustering coefficient: three times the number of triangles divided by the number of connected triples.

### Eigenvector centrality

**Ax = λ x**, take the leading eigenvector. Intuition: a node's importance x_v is the sum of its neighbors' importances divided by λ. Valid for connected undirected graphs. Diverges or misbehaves for directed graphs with no incoming edges — motivates PageRank.

### PageRank

Define the random walk: from node v with probability (1 − α) follow an outgoing edge uniformly, with probability α teleport to a random node. Let p be the stationary distribution.

**p = (1 − α) · Pᵀ p + α · (1/n) · 1**

where P is the row-stochastic version of the adjacency matrix (P_{ij} = 1/out-deg(i) if (i,j) ∈ E). Solving for p:

**p = α (I − (1−α) Pᵀ)⁻¹ (1/n) · 1**

Usually α ≈ 0.15. Computed iteratively via power iteration:

**p_{k+1} = (1 − α) · Pᵀ p_k + α · (1/n) · 1**

Converges in 20–50 iterations for typical graphs.

### Personalized PageRank

Replace the uniform teleport distribution with a preference vector s supported on a query node (or set):

**p = (1 − α) · Pᵀ p + α · s**

Used for graph-based recommendation: personalized PageRank of items from the user's start vector equals their relative "relevance."

### Betweenness centrality

**B(v) = ∑_{s ≠ v ≠ t} σ_{st}(v) / σ_{st}**

where σ_{st} is the number of shortest paths from s to t and σ_{st}(v) the number passing through v. Computed in O(|V|·|E|) via Brandes' algorithm — expensive for large graphs; approximate via sampling.

### Closeness centrality

**Closeness(v) = 1 / ∑_{u ≠ v} d(v, u)**

where d(v, u) is the shortest-path distance. High closeness means quick reach to all others.

### Katz centrality

**x_v = α ∑_u A_{uv} x_u + β**

in matrix form: x = β (I − α Aᵀ)⁻¹ 1. It can be seen as counting weighted walks of all lengths to v; α < 1/λ_max ensures convergence.

## 7.4 Worked Example

### Small graph

Nodes: A, B, C, D, E
Edges (undirected): A–B, A–C, B–C, B–D, D–E

Draw mentally:
```
  A — B — D — E
   \ /
    C
```

### Degrees

deg(A) = 2 (B, C)
deg(B) = 3 (A, C, D)
deg(C) = 2 (A, B)
deg(D) = 2 (B, E)
deg(E) = 1 (D)

### Clustering coefficient

At A: neighbors {B, C}. Are B, C connected? Yes. Number of triangles at A = 1. C(A) = 2·1 / (2·1) = 1.

At B: neighbors {A, C, D}. Pairs: (A,C) yes; (A,D) no; (C,D) no. T(B) = 1. C(B) = 2·1 / (3·2) = 1/3 ≈ 0.33.

At C: neighbors {A, B}. Are A, B connected? Yes. T(C)=1. C(C)=1.

At D: neighbors {B, E}. Are B, E connected? No. T(D)=0. C(D)=0.

At E: deg=1, clustering coefficient undefined (convention: 0).

### PageRank (simplified computation, α=0.15)

Initialize p = [0.2, 0.2, 0.2, 0.2, 0.2].

Transition matrix P (row-stochastic, treating undirected edges as two directed edges):

Row A → B, C: P_A = [0, 0.5, 0.5, 0, 0]
Row B → A, C, D: P_B = [1/3, 0, 1/3, 1/3, 0]
Row C → A, B: P_C = [0.5, 0.5, 0, 0, 0]
Row D → B, E: P_D = [0, 0.5, 0, 0, 0.5]
Row E → D: P_E = [0, 0, 0, 1, 0]

First iteration: p' = 0.85 · Pᵀ p + 0.15 · (1/5).

Pᵀp computed column-by-column:
- (Pᵀp)_A = p_B · 1/3 + p_C · 0.5 = 0.2·(1/3) + 0.2·0.5 = 0.0667+0.10 = 0.1667
- (Pᵀp)_B = p_A · 0.5 + p_C · 0.5 + p_D · 0.5 = 0.10+0.10+0.10 = 0.30
- (Pᵀp)_C = p_A · 0.5 + p_B · 1/3 = 0.10+0.0667 = 0.1667
- (Pᵀp)_D = p_B · 1/3 + p_E · 1 = 0.0667+0.20 = 0.2667
- (Pᵀp)_E = p_D · 0.5 = 0.10

p' = 0.85·[0.1667, 0.30, 0.1667, 0.2667, 0.10] + 0.03·[1,1,1,1,1]
   = [0.1417+0.03, 0.255+0.03, 0.1417+0.03, 0.2267+0.03, 0.085+0.03]
   = [0.172, 0.285, 0.172, 0.257, 0.115]

Sum ≈ 1.00 ✓ (slight drift fixed at convergence).

After a few more iterations p stabilizes roughly to: p ≈ [0.185, 0.275, 0.185, 0.235, 0.120]. Note B has highest PageRank (high degree and bridges to the "D-E" part), E has lowest (leaf on the far side). This matches intuition.

### Betweenness

Shortest paths through each node:
- A–D must go A-B-D (length 2). Passes through B.
- A–E must go A-B-D-E. Passes through B and D.
- C–D must go C-B-D. Passes through B.
- C–E: C-B-D-E. Through B, D.
- A–C: direct edge. Through nobody.
- B–E: B-D-E. Through D.

Betweenness counts (unnormalized, each unordered pair counted once):
B is on {AD, AE, CD, CE} = 4 shortest paths
D is on {AE, CE, BE} = 3
A, C, E: 0

### Katz with α=0.1, β=1

Eigenvalues of A needed to set α. Skip computation, use approximation. For this small graph, Katz centralities follow roughly the same ordering as PageRank.

### Feature table for the whole graph

| node | deg | clust_coef | PR | betweenness |
|------|-----|------------|-----|-------------|
| A | 2 | 1.00 | 0.185 | 0 |
| B | 3 | 0.33 | 0.275 | 4 |
| C | 2 | 1.00 | 0.185 | 0 |
| D | 2 | 0 | 0.235 | 3 |
| E | 1 | 0 | 0.120 | 0 |

These five scalar features per node are ready to use in any supervised task: predicting node labels, edge probability, community membership, or downstream business targets.

## 7.5 Relevance to ML Practice

### Where used

- **Fraud rings**: feature vector per account includes degree-in, degree-out, PageRank, triangle count, size-of-connected-component. Abnormal rings show distinctive patterns.
- **Recommendation**: personalized PageRank from user node gives item relevance scores.
- **Influence prediction**: centrality as input to models predicting content spread.
- **Drug discovery**: atom-level graph features for molecular property prediction (though usually superseded by GNNs today).
- **Knowledge-graph reasoning**: path features for relation prediction.

### When to use

- Any problem where entity relationships exist and carry information.
- When you need interpretable features ("this user is unusually central").
- When you have a graph but not enough data/compute to train a graph neural network.

### When not to use

- When there's no meaningful graph structure (trying to create one from spurious correlations often makes things worse).
- For very large dynamic graphs where global features (betweenness, eigenvector centrality) are too expensive.
- When the task needs node attributes as well as structure — GNNs do both natively.

### Alternatives / complements

- **Graph neural networks** (GCN, GraphSAGE, GAT): learn representations from graph structure and node features jointly.
- **Node embeddings** (DeepWalk, node2vec): produce dense vectors via random walks; good initializer.
- **Graph kernels** (Weisfeiler-Lehman, shortest-path kernels) for graph-level classification.

### Trade-offs

- **Computation**: global features (betweenness, eigenvector centrality) are expensive — O(V·E) or matrix factorization scale. Local features (degree, clustering) are O(E).
- **Interpretability**: hand-engineered graph features are highly interpretable; GNNs are not.
- **Inductive bias**: features bake in assumptions (e.g., uniform teleport for PageRank). Learned embeddings adapt.
- **Cold-start**: new nodes with no edges yet have identical features (usually zero); GNNs still struggle but can leverage node attributes.

## 7.6 Common Pitfalls & Misconceptions

- **Mixing heterogeneous edge types**: computing PageRank on a graph that combines "follows" and "mentions" and "blocks" without thought produces meaningless scores. Aggregate carefully or compute per-edge-type features.
- **Using present-day graph for historical labels**: computing centrality today for an event labeled last year leaks future information. Use a snapshot at or before the label time.
- **Treating PageRank as probability of fraud**: it's a centrality measure; high centrality can correlate with many things, including being a celebrity account. Use with other features.
- **Ignoring disconnected components**: eigenvector centrality is defined per connected component; giant-component-only computation makes isolated subgraphs invisible. PageRank's teleportation sidesteps this.
- **Normalizing betweenness incorrectly across graphs of different sizes**: scale by 2/((n−1)(n−2)) for undirected, or comparable factor for directed.
- **Using degree alone for hubs**: a sybil network where every fake account has exactly 10 friends won't stand out by degree. Triangles and clustering reveal it.
- **Cold-start**: expect features to be noisy for new, low-degree nodes; models should either restrict to established nodes or include "is-new" flags.
- **Directed vs undirected confusion**: the same node often has very different in- and out-degrees and different in-closeness vs out-closeness. Compute both.

## 7.7 Interview Preparation — Graph Features

### Foundational

**Q1. What does PageRank measure and why did it replace raw in-degree as a web-importance score?**
PageRank measures the stationary distribution of a random walk with teleportation: a node is important if other important nodes link to it. Raw in-degree counts links uniformly, so a link farm with thousands of fake pages could artificially inflate a target's importance. PageRank downweights links from low-PageRank sources (since they have their own probability mass to distribute), resisting trivial manipulation.

**Q2. Distinguish between degree centrality, closeness centrality, and betweenness centrality.**
Degree: number of neighbors — local popularity. Closeness: reciprocal of the average shortest-path distance to all other nodes — global reachability. Betweenness: fraction of shortest paths between pairs of other nodes that go through this node — bottleneck / brokerage. They can disagree sharply: a popular node might have low betweenness if many alternative paths exist.

**Q3. What does a high clustering coefficient indicate about a node?**
Its neighbors are tightly connected to each other — the node sits in a cohesive community. Low clustering means its neighbors don't know each other, typical of brokers or hubs bridging different communities.

### Mathematical

**Q4. Derive the PageRank equation.**
Define a random walk: from node v with probability (1 − α), follow a random outgoing edge of v; with probability α, teleport to a random node. The stationary distribution p satisfies the balance equation: p_j = (1 − α) ∑_{i : i → j} p_i / out-deg(i) + α / n. In matrix form, p = (1 − α) Pᵀ p + (α/n) 1, where P is the row-stochastic transition matrix. Solving: p = α (I − (1 − α) Pᵀ)⁻¹ (1/n) · 1.

**Q5. Why does eigenvector centrality require a connected graph?**
The Perron-Frobenius theorem guarantees a unique positive leading eigenvector for non-negative irreducible matrices. An irreducible matrix corresponds to a strongly connected directed graph. Disconnected components mean the matrix is reducible; the eigenvector problem splits into separate problems per component and inter-component comparisons become undefined.

**Q6. Compute time complexity of exact betweenness centrality.**
Brandes' algorithm: O(|V| · |E|) for unweighted graphs (plus O(|V| · (|E| + |V| log |V|)) for weighted via Dijkstra). For graphs with millions of nodes, this is often infeasible; approximation via sampling a subset of source nodes gives O(k · |E|) for k samples with bounded error.

**Q7. Prove that PageRank's teleportation ensures convergence on any graph.**
The power iteration on a stochastic matrix converges when the matrix is ergodic (irreducible and aperiodic). The teleportation term adds a strictly positive ε to every transition: the modified transition matrix has entries bounded below by α/n > 0 everywhere. That makes it irreducible and aperiodic, so the power iteration converges to a unique stationary distribution regardless of the original graph's structure.

### Applied

**Q8. You're building a fraud detection model for payment accounts. How would graph features help?**
Build a graph with accounts as nodes and transactions as edges. Features: in-/out-degree (lots of small inbound + one large outbound is suspicious); triangle count (low in fake rings); local clustering coefficient; PageRank (high centrality among new accounts is weird); size of the connected component; ratio of new-to-old edges; community label (via Louvain); personalized PageRank from known-fraud seeds.

**Q9. Design graph features for a user-item recommendation problem.**
Construct bipartite graph: users and items connected by interactions. Features per user-item pair: common-neighbor count, Jaccard similarity of neighborhoods, Adamic-Adar score (weighted common neighbors), personalized PageRank from user to item, shortest-path distance. Aggregate candidate-generation scores from multiple graphs (co-view, co-purchase) and feed into a downstream GBM.

**Q10. How do you adapt graph features for a streaming/online system?**
Precompute static snapshots periodically (daily) and serve them for features that are expensive to update (betweenness, eigenvector centrality). Maintain incremental updates for cheap features (degree, triangle count) using sketches or streaming algorithms. For personalized PageRank, use random-walk sampling for fast approximate scores at query time.

### Debugging & failure modes

**Q11. Your PageRank scores are dominated by a few spammer accounts. Why?**
Spammer accounts may form tight clusters that accumulate PageRank mass and refuse to leak it. This is the "spider trap" problem; the α teleport term helps but may not be enough. Mitigations: personalize PageRank toward trusted seeds, use Trust/Distrust Rank variants, penalize self-loops and reciprocal-only links, or weight edges by trust / age / validation.

**Q12. You add graph features and model accuracy drops. Why might that happen?**
The graph is noisy or wrong — spurious edges inject noise. The features use time-inconsistent data (leakage). The features are too correlated with existing ones and overwhelm regularization. Implementation bug in cell-vs-node mapping. Re-examine the graph construction, the time alignment, and whether the added features have any marginal signal on a simple model.

**Q13. Graph feature computation crashes on your 1B-edge graph. How do you scale?**
Use distributed graph engines (Spark GraphFrames, Pregel/Giraph, Gnuplot). Approximate where possible: PageRank via power iteration with sparse matrices; betweenness via sampling; sketching for triangle counts (e.g., DOULION sampling). Consider shrinking via subsampling, or focus on local features (k-hop) that don't need global structure.

### Follow-up / probing

**Q14. How does personalized PageRank differ from random-walk-based node embeddings (DeepWalk)?**
Personalized PageRank gives a scalar score per target node from a source: how much of the source's "importance" flows to the target. DeepWalk generates vector embeddings by training a skip-gram over random-walk sequences; it captures co-occurrence in many walks as a low-dimensional similarity. PageRank is interpretable and exact; DeepWalk is dense and learned, better for similarity-based retrieval.

**Q15. When would you prefer hand-crafted graph features to a graph neural network?**
Small graphs or small budgets; strong interpretability requirements; when the graph structure is the main signal and node attributes are weak; when you're already running a GBM pipeline and want to augment it cheaply; when cold-start problems dominate (hand features on small neighborhoods plus smoothing often beat GNNs with unseen nodes).

**Q16. Explain the difference between eigenvector centrality and PageRank on a strongly connected graph.**
Both are spectral measures of importance. Eigenvector centrality is the leading eigenvector of A; PageRank is the leading eigenvector of the damped transition matrix (1 − α)Pᵀ + (α/n)11ᵀ. The teleportation makes PageRank well-defined even for graphs with sinks or disconnected components, and makes it robust to manipulation. On a strongly connected, undirected graph with α → 0, they become equivalent up to normalization.

**Q17. For a recommendation problem, which is better: to use graph features directly in a GBM or to train a GNN?**
Depends on data and scale. GBM with hand-crafted graph features is fast, interpretable, and often surprisingly competitive for moderate data; they can leverage the 20+ years of tabular ML tooling. GNNs shine when node/edge attributes are important, when the task benefits from multi-hop aggregation learned end-to-end, or when massive unlabeled graph data allows pretraining. In production, a two-stage system — GNN for embeddings + GBM for scoring — is a common strong baseline.

---

# End Matter

This guide covers the full spectrum of classical feature engineering — the craft that, despite deep learning's rise, continues to underpin production ML in retail, finance, ads, fraud, forecasting, search, recommendation, and healthcare. Mastery here separates applied scientists who ship effective systems from those who only build models.

Key themes across topics:

1. **Leakage control** is the universal top risk — temporal, group-wise, and target-wise.
2. **Every feature must be computable identically at training and inference time.**
3. **Feature engineering and model choice interact**: tree-based models want discrete or categorical features; linear models want scaled, interacted, and non-linearized continuous features.
4. **Hand-crafted features and learned representations are complements, not substitutes** — the strongest systems blend both.
5. **Interpretability is cheap with hand features and expensive with learned ones**; choose based on your deployment's requirements.

---

**[← Previous Chapter: Preprocessing](preprocessing_guide.md) | [Table of Contents](index.md) | [Next Chapter: Feature Selection →](feature_selection_guide.md)**
