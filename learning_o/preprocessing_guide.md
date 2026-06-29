# Data Preprocessing: A Comprehensive Reference Guide

This guide covers the foundational techniques used to transform raw data into a form suitable for machine learning. Preprocessing is often the stage where the largest gains in model quality are achieved, and also where the most subtle errors originate. The material is organized into five self-contained chapters:

1. Missing Data Handling
2. Categorical Encoding
3. Scaling & Normalization
4. Outlier Treatment
5. Datetime Processing

Each chapter follows a strict six-section structure (Motivation, Conceptual Foundations, Mathematical Formulation, Worked Examples, Relevance to ML Practice, Common Pitfalls), followed by a tiered interview question bank.

---

# Chapter 1: Missing Data Handling

## 1.1 Motivation & Intuition

Real datasets are almost never complete. A customer database might lack phone numbers for 30% of users. A medical record might be missing a blood test result because the test was never ordered. A sensor might fail for several hours, producing a gap in a time series. A survey respondent might refuse to answer a question about income.

Why does this matter for machine learning? Most ML algorithms require a fully populated feature matrix. Scikit-learn's `LogisticRegression`, `SVM`, and most neural network frameworks will raise an error if fed `NaN` values. Even algorithms that natively handle missingness (XGBoost, LightGBM, CatBoost) treat it with specific semantics that the practitioner must understand.

The choice of how to handle missing values has major consequences:

- Naive approaches (replacing all missing ages with the population mean) can introduce systematic bias and distort variance.
- Aggressive row-dropping can discard 80–90% of a dataset.
- Treating missingness itself as a signal (via indicator variables) can sometimes be *more* predictive than the underlying value.

**Concrete example.** In a loan default prediction model, applicants who do not report their income may be systematically different from those who do — perhaps they are self-employed, between jobs, or deliberately concealing low earnings. If you impute missing incomes with the population mean, you erase this signal. If you drop these rows, you may lose your most informative data. If you add a `income_is_missing` indicator, the model can learn that missingness itself is predictive of default.

The central insight: **missingness is rarely random. How you handle it should depend on *why* values are missing, not just *that* they are missing.**

## 1.2 Conceptual Foundations

### Rubin's Classification of Missingness

Donald Rubin (1976) introduced three categories that remain the foundation of missing-data theory.

**MCAR — Missing Completely At Random.** The probability of a value being missing is independent of all data, observed or unobserved. Example: a lab technician randomly drops 5% of samples. Under MCAR, the observed data is an unbiased random sample of the full data. This is the best-case scenario and is rare in practice.

**MAR — Missing At Random.** The probability of missingness depends on *observed* variables but not on the missing value itself. Example: elderly patients are less likely to report their weight, but within each age group, whether weight is reported does not depend on the actual weight. Conditional on age (observed), missingness is random. Most principled imputation methods assume MAR.

**MNAR — Missing Not At Random.** The probability of missingness depends on the unobserved value itself. Example: high earners are systematically less willing to disclose their income. This is the hardest case — you cannot fully correct for MNAR without external information or strong modeling assumptions.

### Why the Distinction Matters

- **Listwise deletion** is unbiased only under MCAR.
- **Mean imputation, k-NN, MICE, EM** are valid under MAR (and MCAR).
- **All standard methods are biased under MNAR.** You need explicit models of the missingness mechanism (pattern-mixture models, selection models) or domain knowledge.

### Key Terms

- **Response indicator** $R \in \{0,1\}$: $R_{ij} = 1$ if $X_{ij}$ is observed, 0 otherwise.
- **Complete case:** a row with no missing values.
- **Single imputation:** one value replaces each missing entry.
- **Multiple imputation:** several plausible imputations are drawn, analyses are run on each, and results are combined — propagating imputation uncertainty.
- **Ignorability:** the condition (MAR + distinct parameters for data and missingness) under which the missingness mechanism can be ignored when doing maximum-likelihood estimation.

### Assumptions and What Breaks

| Method | Assumes | Breaks when |
|---|---|---|
| Listwise deletion | MCAR | MAR or MNAR (biased estimates, lost power) |
| Mean/median imputation | MCAR; marginal distribution is enough | Variance is underestimated; correlations are distorted |
| k-NN imputation | MAR; local similarity captures structure | High dimensions; irrelevant features dominate distance |
| MICE | MAR; conditional models are correctly specified | Incompatible conditional models; MNAR |
| EM | MAR; parametric model is correct | Model misspecification; MNAR |

## 1.3 Mathematical Formulation

Let $X \in \mathbb{R}^{n \times d}$ be the complete data matrix (partially unobserved) and $R \in \{0,1\}^{n \times d}$ be the response indicator. Partition $X$ into observed $X_{\text{obs}}$ and missing $X_{\text{mis}}$. Let $\theta$ parameterize the data distribution and $\phi$ parameterize the missingness mechanism.

**Mechanisms in formal notation:**

- MCAR: $P(R \mid X, \phi) = P(R \mid \phi)$
- MAR: $P(R \mid X, \phi) = P(R \mid X_{\text{obs}}, \phi)$
- MNAR: $P(R \mid X, \phi)$ depends on $X_{\text{mis}}$

**Full likelihood:**

$$
L(\theta, \phi \mid X_{\text{obs}}, R) = \int P(X_{\text{obs}}, X_{\text{mis}} \mid \theta) \, P(R \mid X_{\text{obs}}, X_{\text{mis}}, \phi) \, dX_{\text{mis}}
$$

Under MAR and distinct parameters, the likelihood factorizes:

$$
L(\theta, \phi \mid X_{\text{obs}}, R) \propto P(X_{\text{obs}} \mid \theta) \cdot P(R \mid X_{\text{obs}}, \phi)
$$

So we can estimate $\theta$ using only $P(X_{\text{obs}} \mid \theta)$ — the missingness mechanism is *ignorable*. Under MNAR, the missing value is inside the integrand, and we cannot separate the two.

### Listwise Deletion

Estimate $\theta$ using only complete rows. Let $C = \{i : R_{ij} = 1 \ \forall j\}$. Then $\hat\theta_{\text{CC}} = \arg\max_\theta \prod_{i \in C} P(x_i \mid \theta)$. Unbiased under MCAR. Under MAR, the complete-case sample is not representative.

### Pairwise Deletion

For a statistic involving a subset of variables (e.g., covariance between $X_j$ and $X_k$), use all rows where *both* $X_j$ and $X_k$ are observed. Preserves more information than listwise, but the resulting covariance matrix may not be positive semi-definite (different pairs use different sample subsets).

### Mean Imputation

$$
\hat{X}_{ij} = \bar{X}_j = \frac{1}{|\mathcal{O}_j|} \sum_{k \in \mathcal{O}_j} X_{kj}, \quad \mathcal{O}_j = \{k : R_{kj} = 1\}
$$

After imputation, the marginal mean of $X_j$ is preserved, but the variance is *reduced* (imputed values have zero variance around the mean) and correlations between $X_j$ and other variables are *attenuated* (pulled toward zero).

**Proof sketch of variance attenuation.** Let $n$ rows, $m$ observed, $n-m$ imputed with $\bar{X}$. Sample variance after imputation:

$$
s^2_{\text{after}} = \frac{1}{n-1} \left[ \sum_{k \in \mathcal{O}} (X_k - \bar{X})^2 + (n-m) \cdot 0 \right] = \frac{m-1}{n-1} s^2_{\text{before}}
$$

So variance is shrunk by a factor of $(m-1)/(n-1) \approx m/n$.

### k-Nearest Neighbors Imputation

For a missing entry $X_{ij}$, find the $k$ rows closest to row $i$ based on observed features, and impute with a weighted average (or mode for categoricals):

$$
\hat{X}_{ij} = \frac{\sum_{l \in N_k(i)} w_{il} X_{lj}}{\sum_{l \in N_k(i)} w_{il}}
$$

where $w_{il}$ is often $1/d(x_i, x_l)$ or uniform. Distance $d$ is typically Euclidean over observed features, often requiring the data to be scaled first.

Assumption: similar rows (by observed features) have similar values for the missing feature. This is a local MAR assumption.

### Multiple Imputation by Chained Equations (MICE)

MICE imputes variables iteratively by regressing each incomplete variable on all others. For $m$ imputations:

1. Initialize missing entries with simple imputation (e.g., mean).
2. For each iteration $t = 1, 2, \ldots, T$:
   - For each variable $X_j$ with missing values:
     - Fit $f_j(X_j \mid X_{-j})$ on rows where $X_j$ is observed.
     - Replace missing $X_j$ entries by drawing from the predictive distribution of $f_j$.
3. After convergence, save the completed dataset. Repeat the full procedure $m$ times to get $m$ datasets.

The "chained equations" name comes from the sequential conditional regressions. Each $f_j$ can be a different model type: linear regression for continuous, logistic for binary, multinomial for categorical, predictive mean matching for robustness.

**Rubin's combining rules.** To combine estimates $\hat\theta_1, \ldots, \hat\theta_m$ from $m$ imputed datasets:

$$
\bar\theta = \frac{1}{m} \sum_{k=1}^m \hat\theta_k
$$

$$
\bar{U} = \frac{1}{m} \sum_{k=1}^m U_k \quad \text{(within-imputation variance)}
$$

$$
B = \frac{1}{m-1} \sum_{k=1}^m (\hat\theta_k - \bar\theta)^2 \quad \text{(between-imputation variance)}
$$

$$
T = \bar{U} + \left(1 + \frac{1}{m}\right) B \quad \text{(total variance)}
$$

Standard errors from $\sqrt{T}$ properly reflect imputation uncertainty.

### EM Algorithm for Missing Data

Suppose the complete-data log-likelihood is $\ell(\theta; X) = \log P(X \mid \theta)$. The EM algorithm alternates:

- **E-step:** Compute expected complete-data log-likelihood given observed data and current $\theta^{(t)}$:

$$
Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{X_{\text{mis}} \mid X_{\text{obs}}, \theta^{(t)}} [\ell(\theta; X_{\text{obs}}, X_{\text{mis}})]
$$

- **M-step:** Maximize $Q$:

$$
\theta^{(t+1)} = \arg\max_\theta Q(\theta \mid \theta^{(t)})
$$

For a multivariate Gaussian with missing entries, the E-step becomes imputation of missing $X_{\text{mis}}$ by their conditional expectation given $X_{\text{obs}}$ (using current $\mu, \Sigma$), plus a correction term accounting for conditional variance. The M-step updates $\mu, \Sigma$ using the filled-in data plus the correction. EM is guaranteed to monotonically increase the observed-data likelihood.

### Matrix Completion

For an $n \times d$ matrix $M$ with observed entries $\Omega$, find $\hat M$ of low rank that agrees with $M$ on $\Omega$. A common convex formulation:

$$
\min_{\hat M} \|\hat M\|_* \quad \text{s.t.} \quad \hat M_{ij} = M_{ij} \ \forall (i,j) \in \Omega
$$

where $\|\cdot\|_*$ is the nuclear norm (sum of singular values), a convex relaxation of rank. In practice, a soft version is solved:

$$
\min_{U, V} \sum_{(i,j) \in \Omega} (M_{ij} - U_i^\top V_j)^2 + \lambda (\|U\|_F^2 + \|V\|_F^2)
$$

This is the backbone of collaborative filtering recommender systems (e.g., Netflix prize). It assumes the true matrix is approximately low-rank — a strong global structure assumption.

### Deep Learning Approaches

- **Denoising autoencoders:** Corrupt observed data with random masking, train a network to reconstruct the full input. At inference, feed partial data and read off the reconstruction.
- **GAIN (Generative Adversarial Imputation Networks):** A generator imputes missing values; a discriminator tries to identify which entries were imputed vs. observed. Adversarial training improves realism.
- **VAE-based imputation (MIWAE, HI-VAE):** Use a variational autoencoder with a likelihood defined only over observed entries; sample from the posterior over latents to impute.
- **Transformer-based:** Treat each feature as a token; use attention with masking to fill in missing tokens (similar to BERT's masked language modeling).

## 1.4 Worked Examples

### Example 1: Attenuation from Mean Imputation

Toy dataset with ten observations of $(X, Y)$:

| i | X | Y |
|---|---|---|
| 1 | 1 | 2 |
| 2 | 2 | 4 |
| 3 | 3 | 5 |
| 4 | 4 | 8 |
| 5 | 5 | 9 |
| 6 | 6 | 12 |
| 7 | 7 | 13 |
| 8 | 8 | 16 |
| 9 | 9 | 17 |
| 10 | 10 | 20 |

True correlation $\rho(X, Y) \approx 0.999$. Suppose $Y$ is missing for rows 6–10 (the five largest $X$ values, so MAR conditional on $X$).

**Listwise deletion** uses rows 1–5: correlation computed on $X = \{1,\ldots,5\}$, $Y = \{2,4,5,8,9\}$ is still very high but the sample is biased toward low $X$ values; estimates of $\mathbb{E}[Y]$ will be low.

**Mean imputation.** Observed mean of $Y$ is $(2+4+5+8+9)/5 = 5.6$. Fill rows 6–10 with 5.6. Now compute correlation across all 10 rows:

- $\bar X = 5.5$, $\bar Y = (2+4+5+8+9+5.6\cdot 5)/10 = 5.6$
- The imputed block has zero variation in $Y$, so half the rows contribute nothing to $\text{Cov}(X,Y)$ while contributing fully to $\text{Var}(X)$.
- Resulting correlation: ≈ 0.60 — dramatically attenuated from the true 0.999.

Lesson: mean imputation destroys associations.

### Example 2: k-NN Imputation

Dataset:

| Row | Age | Income | Tenure |
|---|---|---|---|
| A | 25 | 40k | 1 |
| B | 30 | 55k | 3 |
| C | 28 | 50k | ? |
| D | 45 | 90k | 20 |
| E | 50 | 100k | 25 |

Impute `Tenure` for row C using k=2, Euclidean distance on standardized `Age` and `Income`.

Standardize: $\mu_{\text{Age}}=35.6$, $\sigma_{\text{Age}} \approx 10.8$; $\mu_{\text{Inc}}=67$, $\sigma_{\text{Inc}} \approx 26.8$.

- $z_C = ((28-35.6)/10.8, (50-67)/26.8) = (-0.70, -0.63)$
- $z_A = (-0.98, -1.01)$ → distance $\sqrt{0.079 + 0.144} = 0.47$
- $z_B = (-0.52, -0.45)$ → distance $\sqrt{0.033 + 0.032} = 0.25$
- $z_D = (0.87, 0.86)$ → distance $\sqrt{2.46 + 2.21} = 2.16$
- $z_E = (1.33, 1.23)$ → distance $\sqrt{4.12 + 3.46} = 2.75$

Nearest two to C: B (0.25) and A (0.47). Uniform average: $(3 + 1)/2 = 2$. Imputed tenure = 2. (A distance-weighted version would weight B more heavily.)

### Example 3: MICE Iteration

Three variables: `Age`, `Income`, `Spending`, each with some missing values. Initialize all missing cells with column means.

**Iteration 1:**
- Fit `Age` ~ `Income` + `Spending` on rows with Age observed. Predict missing Ages. Replace.
- Fit `Income` ~ `Age` + `Spending` on rows with Income observed. Predict missing Incomes (using just-updated Ages). Replace.
- Fit `Spending` ~ `Age` + `Income` on rows with Spending observed. Predict. Replace.

**Iteration 2:** repeat with updated values. After ~5–10 iterations the imputations typically stabilize (monitor trace plots of imputed values vs iteration).

For multiple imputation, repeat this whole procedure $m$ times (say $m=20$) with different random draws from the predictive distributions to get 20 completed datasets. Train your model on each; combine results with Rubin's rules.

## 1.5 Relevance to ML Practice

**When to use each method:**

- **Listwise deletion:** Only when missingness is <5% *and* plausibly MCAR *and* sample size is large. Acceptable for preliminary exploration.
- **Mean/median/mode:** Baseline. Acceptable when missingness is small (<5–10%) and you care more about predictive performance than inference. Always add a missingness indicator.
- **k-NN:** Good when you believe in local similarity. Expensive on large datasets ($O(n^2)$ naively). Sensitive to scaling and to irrelevant features.
- **MICE:** Gold standard for statistical inference (epidemiology, econometrics). Best when you need calibrated uncertainty. Expensive.
- **EM:** Useful when you have a clear parametric model (mixture models, factor analysis).
- **Matrix completion:** Best when data is inherently low-rank (user-item matrices, gene expression). Not typically used for tabular feature matrices with heterogeneous columns.
- **Deep learning:** Competitive on large, complex datasets with many features; overkill for small tabular data.

**Missingness as a feature.** In many ML systems — especially for credit, insurance, and medical models — adding a binary `is_missing_j` column for each feature $j$ with missing values is extremely effective. This is sometimes called the "missing indicator method." Some gradient-boosted tree libraries (XGBoost, LightGBM) do this implicitly: they learn which branch (left or right) missing values should go down at each split.

**Train–test leakage.** A common and serious error: computing imputation statistics (means, k-NN neighbors, MICE models) on the *full* dataset including the test set. The imputer must be fit on training data only and applied to test data. In scikit-learn, always wrap imputation inside a `Pipeline` so that cross-validation refits the imputer on each training fold.

**Production considerations.** Imputers are stateful. The production imputer must be the same object fit at training time (same means, same k-NN reference set, same MICE models), serialized and shipped with the model.

**Trade-offs:**

| Axis | Simple (mean) | k-NN | MICE | Deep |
|---|---|---|---|---|
| Bias | High | Lower | Low under MAR | Low |
| Variance | Low | Moderate | Properly reflected | Moderate |
| Compute | Tiny | $O(n^2 d)$ | Many regressions | Training cost |
| Interpretability | High | Moderate | Moderate | Low |
| Needs tuning | No | k, distance | Conditional models | Architecture |

## 1.6 Common Pitfalls & Misconceptions

1. **"I'll just drop missing rows."** Loses data; valid only under MCAR; often drops 50%+ of rows because missingness stacks across features.
2. **Imputing before train–test split.** Test statistics leak into training. Always fit imputers on training set only.
3. **Mean imputation hides information.** Even if you impute, add a missingness indicator column. The indicator is often more predictive than the imputed value.
4. **Not standardizing before k-NN.** Distance is dominated by large-scale features. Always scale first.
5. **Confusing single imputation with multiple imputation.** Single imputation understates uncertainty — downstream confidence intervals will be too narrow.
6. **Imputing the target.** Never impute the target variable. Drop rows where $y$ is missing (or model missingness as a separate outcome).
7. **Assuming MAR without checking.** MNAR is common in real data (income, sensitive attributes, survey non-response). Mean and MICE are both biased under MNAR.
8. **Ignoring categorical modes.** Don't use the mean for categorical features; use mode or a dedicated "Missing" category.
9. **Zero imputation by default.** Replacing missing with zero imposes an arbitrary numeric meaning (e.g., "zero income"), often worse than mean.
10. **Not monitoring drift in missingness rates.** If the rate of missing values changes between training and production, the imputer's assumptions break. Monitor missingness rates per feature.

## 1.7 Interview Questions

### Foundational

**Q1. Explain MCAR, MAR, and MNAR with an example of each.**

**A.** MCAR (Missing Completely At Random): missingness is independent of all data. Example — a random lab instrument failure. MAR (Missing At Random): missingness depends on observed variables but not on the missing value itself. Example — elderly patients (observed age) are less likely to report weight, but the missingness does not depend on the actual weight beyond what age implies. MNAR (Missing Not At Random): missingness depends on the unobserved value. Example — high-income individuals refuse to report income precisely because it's high. The classification matters because standard imputation methods assume MAR; they are biased under MNAR without additional modeling.

**Q2. Why is mean imputation often a bad idea even though it preserves the mean?**

**A.** Three reasons. (1) It reduces variance: imputed values have zero deviation from the mean, so sample variance shrinks by a factor of roughly $m/n$ where $m$ is the observed count. (2) It attenuates correlations toward zero, because the imputed block contributes to variance of $X$ but not to covariance with $Y$. (3) It throws away information about which values were missing — and missingness often carries signal.

**Q3. What's the difference between single and multiple imputation?**

**A.** Single imputation fills each missing cell with one value (deterministic or a single draw). Multiple imputation generates $m$ plausible completed datasets by drawing from the predictive distribution. Analyses are run on each and combined with Rubin's rules, which partition variance into within-imputation and between-imputation components. Single imputation underestimates uncertainty — downstream standard errors are too small, leading to overconfident inference.

**Q4. Why is missingness itself often a useful feature?**

**A.** Missingness is rarely random. It typically encodes latent information about the observation — e.g., "income not reported" might correlate with self-employment or financial stress. Adding a binary indicator captures this signal without requiring strong assumptions about the underlying value. In practice this often improves predictive performance more than any particular imputation method.

**Q5. How do XGBoost and LightGBM handle missing values natively?**

**A.** At each split, they learn a default direction for missing values by trying both branches during training and selecting the one that yields lower loss. This effectively treats missingness as a learned categorical branch. The approach is powerful but: (a) it assumes the "correct" direction is globally consistent per split, and (b) it doesn't explicitly encode missingness as a feature used by other splits. Adding an explicit missingness indicator can still help.

### Mathematical

**Q6. Derive the variance attenuation from mean imputation.**

**A.** Let $X_1, \ldots, X_m$ be observed, $X_{m+1}, \ldots, X_n$ imputed with $\bar X = \frac{1}{m}\sum_{k=1}^m X_k$. The sample variance after imputation:

$$
s^2_{\text{after}} = \frac{1}{n-1}\left[\sum_{k=1}^m (X_k - \bar X)^2 + \sum_{k=m+1}^n (\bar X - \bar X)^2\right] = \frac{1}{n-1}\sum_{k=1}^m (X_k - \bar X)^2
$$

Before imputation (observed only):

$$
s^2_{\text{obs}} = \frac{1}{m-1}\sum_{k=1}^m (X_k - \bar X)^2
$$

So $s^2_{\text{after}} = \frac{m-1}{n-1} s^2_{\text{obs}}$. For $m=50, n=100$, we have $s^2_{\text{after}} \approx 0.49 \, s^2_{\text{obs}}$ — the variance is roughly halved.

**Q7. State and explain Rubin's ignorability condition.**

**A.** Ignorability requires (a) MAR and (b) distinct parameters for the data distribution and the missingness mechanism. Under ignorability, the full-data likelihood factorizes into a term depending only on observed data and a separate term for the missingness mechanism. You can then estimate $\theta$ using $P(X_{\text{obs}} \mid \theta)$ alone, ignoring $P(R \mid \cdot)$. If MAR fails (MNAR), the integral over $X_{\text{mis}}$ couples $\theta$ and $\phi$, and you must jointly model both.

**Q8. Derive Rubin's combining rule for multiple imputation variance.**

**A.** With $m$ imputed datasets giving estimates $\hat\theta_k$ and within-imputation variances $U_k$:
- Point estimate: $\bar\theta = \frac{1}{m}\sum_k \hat\theta_k$
- Within-imputation variance: $\bar U = \frac{1}{m}\sum_k U_k$ — average sampling variance as if there were no missing data.
- Between-imputation variance: $B = \frac{1}{m-1}\sum_k (\hat\theta_k - \bar\theta)^2$ — variability across imputations reflecting imputation uncertainty.
- Total: $T = \bar U + (1 + 1/m) B$. The $(1 + 1/m)$ factor corrects for using a finite number of imputations rather than an infinite number. As $m \to \infty$, the correction vanishes.

**Q9. Why can pairwise deletion produce a non-PSD covariance matrix?**

**A.** Each pairwise covariance $\text{Cov}(X_j, X_k)$ is estimated from rows where both $X_j$ and $X_k$ are observed — a different subset for each pair. The resulting matrix combines estimates from different "samples" and can violate the positive semi-definiteness property that requires a single coherent sample. This breaks downstream methods like PCA, which require PSD input.

**Q10. Walk through one E-step and M-step for a bivariate Gaussian with missing $X_2$.**

**A.** Suppose $(X_1, X_2) \sim N(\mu, \Sigma)$ with $X_1$ observed and some $X_2$ missing. Conditional on $X_1 = x_1$:

$$
X_2 \mid X_1 = x_1 \sim N\left(\mu_2 + \frac{\sigma_{12}}{\sigma_{11}}(x_1 - \mu_1),\ \sigma_{22} - \frac{\sigma_{12}^2}{\sigma_{11}}\right)
$$

**E-step:** For each row $i$ with $X_2$ missing, compute $\hat x_{i2} = \mu_2^{(t)} + \frac{\sigma_{12}^{(t)}}{\sigma_{11}^{(t)}}(x_{i1} - \mu_1^{(t)})$ and the conditional variance $v = \sigma_{22}^{(t)} - (\sigma_{12}^{(t)})^2/\sigma_{11}^{(t)}$. Also compute $\mathbb{E}[X_{i2}^2 \mid X_{i1}] = \hat x_{i2}^2 + v$.

**M-step:** Update $\mu_2^{(t+1)} = \frac{1}{n}\sum_i \hat x_{i2}$, $\sigma_{22}^{(t+1)} = \frac{1}{n}\sum_i (\hat x_{i2}^2 + v_i) - (\mu_2^{(t+1)})^2$, and similarly $\sigma_{12}$. Crucially, the M-step uses the *expected* sufficient statistics, not just the point predictions — the added $v$ is what prevents variance collapse.

### Applied

**Q11. You have a tabular dataset with 50 features and 20% of values missing. How do you approach it?**

**A.** Step-by-step approach:

1. **Diagnostic.** Compute per-feature missingness rates. Visualize missingness patterns (e.g., `missingno` library). Check whether missingness in one feature correlates with values of another (hint of MAR) or of itself (hint of MNAR).
2. **Split first.** Hold out a test set before any imputation.
3. **Baseline.** Add per-feature missingness indicators. Use median imputation for numeric, mode or "Missing" category for categorical. Train a simple model.
4. **Upgrade selectively.** For features where the baseline imputation seems costly (based on feature importance or domain knowledge), try k-NN or MICE on training data inside a pipeline.
5. **Compare on validation.** Don't assume sophistication beats simplicity — mean imputation plus indicator often matches MICE on large datasets.
6. **Production.** Serialize the fitted imputer. Monitor missingness rates over time.

**Q12. When would you use MICE over k-NN imputation?**

**A.** MICE is preferable when (a) you need calibrated uncertainty for downstream inference (confidence intervals, p-values), (b) the data has many features and clear conditional relationships, or (c) missingness patterns are complex (different variables missing across rows). k-NN is preferable when (a) you're in a predictive pipeline and don't need inference, (b) the dataset has strong local structure (clusters or manifolds), and (c) features are homogeneous (all numeric, similarly scaled). k-NN is often faster to implement but slower to run at scale and more sensitive to irrelevant features.

**Q13. You're training a recommender system. How does matrix completion help with missing ratings?**

**A.** In collaborative filtering the $n \times d$ user-item rating matrix is ≥95% missing. Matrix completion factorizes $M \approx UV^\top$ with $U \in \mathbb{R}^{n \times r}, V \in \mathbb{R}^{d \times r}$ for small rank $r$, minimizing squared error over observed entries plus regularization. This is exactly Netflix-style collaborative filtering. The low-rank assumption captures the idea that users and items live in a low-dimensional latent preference space. Predictions for unobserved entries are $\hat M_{ij} = u_i^\top v_j$. Extensions include bias terms, non-negative factors, and neural architectures.

**Q14. In a production fraud model, how would you handle a feature (say `device_fingerprint_score`) that's missing at inference time for 5% of requests?**

**A.** First, understand why it's missing — is the scoring service timing out? Is it only available for certain device types? This is critical because the MAR/MNAR distinction determines whether you can cleanly impute.

In production I would: (1) always compute a missingness indicator and feed it to the model; (2) use a fast imputation (median or a conditional mean by segment) for the value itself; (3) monitor the missingness rate in production — a sudden increase signals an upstream issue; (4) evaluate model performance on the missing subgroup separately; (5) consider training a small fallback model that doesn't use the feature at all, for high-missingness segments.

**Q15. You notice missingness rate spiked from 2% to 15% overnight. What do you do?**

**A.** This is a production incident. (1) Do not retrain on the new data — the imputer was fit under old missingness assumptions. (2) Alert upstream teams; the root cause is usually an upstream data pipeline failure, not a sudden shift in real-world missingness. (3) Evaluate model metrics on the affected segment; if performance degrades, consider rolling back or switching to a more conservative scoring. (4) If the increase persists and is real, refit the imputer and retrain. (5) Investigate whether the newly-missing rows have shifted in other ways (covariate shift).

### Debugging & Failure Modes

**Q16. A colleague says their test RMSE improved after switching from mean to MICE imputation, but validation RMSE got worse. What's likely going on?**

**A.** Almost certainly data leakage. MICE fit on train+test combined uses test-set information to build conditional distributions, inflating test performance. A proper implementation fits MICE on training only (or refits within each CV fold using a pipeline). The fact that validation (which should be treated like test) got worse suggests the validation split wasn't included in the leaky fit — confirming the diagnosis.

**Q17. Your k-NN imputer takes 4 hours on a dataset that takes 5 minutes to train a model on. What's happening?**

**A.** k-NN imputation is $O(n^2 d)$ naive. For $n = 10^6$, that's $10^{12}$ distance computations. Possible fixes: (1) use approximate nearest neighbors (HNSW, FAISS); (2) restrict imputation to a subsample of candidates; (3) use simpler imputation and add indicators; (4) if features are sparse, use sparse-aware distances. Often the question is whether k-NN's marginal gain over median imputation is worth the compute — often it isn't.

**Q18. A model trained on MICE-imputed data performs well in evaluation but fails in production. What could be wrong?**

**A.** Several possibilities: (1) Different missingness patterns at inference time — MICE's conditional distributions were fit on training patterns, may be nonsense for new patterns. (2) The production imputation step is cheaper/different from training (e.g., uses mean instead of MICE for latency). If so, the model is being fed a different distribution than it was trained on. (3) Ranges of observed features shifted; MICE's conditional regressions extrapolate poorly. (4) Random draws from MICE's predictive distribution introduce variance — if production uses only point estimates, there's a silent mismatch.

**Q19. After adding missingness indicators, one of them has extremely high feature importance — higher than any real feature. Is this a red flag?**

**A.** Usually yes — it typically means missingness is a proxy for the target itself. For instance, in a churn model, `last_login_missing` might mean "never logged in" which is almost a deterministic predictor of churn. This is not necessarily wrong (missingness is genuinely informative), but you should verify: (a) that the relationship is expected and not due to a data pipeline artifact; (b) that the feature is available at inference time in the same way as at training time; (c) that this isn't label leakage (e.g., the feature is only missing for cases where the label was not yet recorded).

**Q20. An imputer works in training but crashes on new test data with "unseen category" errors. Why?**

**A.** The imputer was fit on training categoricals and encoded them with a fixed vocabulary. New data contains categories not seen at training time. Fixes: (1) configure the imputer to handle unseen categories (e.g., map to an "other" bucket); (2) collect representative categorical values during training; (3) use hashed encodings that don't require a fixed vocabulary.

### Probing Follow-ups

**Q21. Can you always detect MNAR from the data alone?**

**A.** No — not without additional assumptions or auxiliary information. MCAR vs MAR can be partially diagnosed (e.g., by testing whether missingness in $X_j$ depends on other observed variables). But MAR vs MNAR cannot be distinguished without knowing the missing values, which we don't have by definition. In practice we rely on domain knowledge (e.g., "income is typically MNAR"), sensitivity analyses (testing how estimates change under various MNAR assumptions), or collecting auxiliary data.

**Q22. If missingness indicator features are so useful, why don't we always rely on them alone and skip imputation?**

**A.** Two reasons. First, models like linear models and neural networks need a numeric value for the feature itself — they can't train with `NaN`. Imputation is needed mechanically. Second, the imputed value may still carry information (the indicator says "missing" but the imputed value says "imputed value consistent with the row's other features"). Tree models can sometimes use indicators alone if paired with a sentinel value like $-999$, but even there, a meaningful imputation plus indicator usually wins.

**Q23. How does MICE's assumption of compatible conditional distributions get violated?**

**A.** MICE fits one conditional model per variable (e.g., $X_1 \mid X_2, X_3, \ldots$). There's no guarantee that a joint distribution exists having all these specified conditionals. When conditionals are incompatible, the iterations may not converge to a proper joint. In practice this often shows up as unstable trace plots or dependence on the iteration order. Diagnostics: run MICE with different variable orderings; check convergence; compare marginal distributions of imputations to observed.

**Q24. In a deep learning model for tabular data, how would you handle missing values?**

**A.** Options include: (1) Learn a mask-aware input — concatenate a binary mask indicating missingness and fill missing entries with a learned embedding or a sentinel. (2) Use attention with masking so that missing positions don't contribute to the output (like BERT's masked attention). (3) Use a denoising autoencoder pretraining phase that teaches the model to reconstruct missing entries. (4) Use a specialized architecture like NeuMiss or GAIN. In practice, a simple median imputation plus mask concatenation works remarkably well.

**Q25. Suppose you discover that both your training set and your test set were imputed with the full dataset's means (leakage). Your held-out metrics look good. Should you care?**

**A.** Yes, seriously. The leaked means make the test set appear easier than it actually is — in production you won't have access to means computed on future data, and the distribution of "missing" may shift. Your reported test metrics are optimistic. The correct fix: refit the imputer on training data only, re-evaluate on test. Expect test metrics to drop; this drop is the real-world hit you were hiding from yourself.

---

**[← Previous Chapter: Hyperparameter Tuning](hyperparameter_tuning_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Feature Creation →](feature_creation_guide.md)**

---

# Chapter 2: Categorical Encoding

## 2.1 Motivation & Intuition

Most ML algorithms operate on numeric vectors. But many real features are categorical: `country = "Brazil"`, `device = "iPhone"`, `genre = "thriller"`, `merchant_id = 58234`. These must be converted to numbers — *encoded* — before modeling.

The naive approach — assigning each category an arbitrary integer — seems fine until you realize that most algorithms interpret numbers as having magnitude and order. Giving Brazil = 1 and Argentina = 2 tells a linear model that Argentina is "twice Brazil," which is nonsense.

Good encoding must preserve (or avoid falsely inducing) the structure of the categorical variable:
- Nominal categories (no order): `country`, `color`, `merchant_id`
- Ordinal categories (natural order): `education_level`, `grade = A/B/C`, `size = S/M/L`
- High-cardinality categories: `user_id`, `product_id`, `zip_code`

Different encoding choices have dramatically different consequences. One-hot encoding of a 10,000-value column creates 10,000 sparse columns. Label encoding of a nominal variable smuggles in false ordering. Target encoding can leak label information and overfit badly.

**Concrete example.** In a real-estate price model, `neighborhood` has 200 values. One-hot gives 200 features that a linear model must each fit separately — data-hungry. Target encoding replaces each neighborhood with its mean price — compact but leaks the target. The right choice depends on the model and the data size.

## 2.2 Conceptual Foundations

### Types of Categorical Variables

**Nominal:** no intrinsic order. "Red/Blue/Green". Encodings that impose order (label encoding, ordinal encoding with arbitrary integer assignment) are inappropriate for most linear models but can work for tree models.

**Ordinal:** intrinsic order. "Small < Medium < Large". Ordinal encoding (assigning consecutive integers in the natural order) is appropriate and information-preserving.

**High-cardinality nominal:** many values. `zip_code` (40,000 in the US), `user_id` (millions), `merchant_id`, `url`. One-hot explodes; specialized techniques are required.

### Core Encodings

**One-hot encoding.** Each category becomes a binary column. For a variable with $k$ categories, produce $k$ columns. Typically one column is dropped to avoid collinearity — the "dummy variable trap" — yielding $k-1$ columns. Drop is required for linear models with intercepts but not for tree models or regularized linear models.

**Label (ordinal integer) encoding.** Assign each category an integer 0 to $k-1$. Sensible for ordinal variables when integers reflect the natural order; problematic for nominal variables except with tree models (which can always split at any threshold).

**Target (mean) encoding.** Replace each category with the mean of the target over rows in that category. Compact — one column regardless of cardinality. Requires regularization and leakage prevention.

**Embedding encoding.** Learn a dense vector $v_c \in \mathbb{R}^k$ for each category $c$, jointly with the downstream model. Standard in deep learning for entity-type features.

### How They Interact

Consider a logistic regression $p(y=1 \mid x) = \sigma(w^\top x + b)$. With one-hot encoding of a $k$-category variable, we learn $k-1$ weights — one per category (dropped reference has weight 0). The model becomes a piecewise-constant function of the category. With target encoding, we get a single weight on a single column — much less flexible.

Tree models are more forgiving: a tree can recover arbitrary groupings of label-encoded integers by splitting, but it works best with low-cardinality or properly ordered labels.

### Assumptions and What Breaks

| Encoding | Assumes | Breaks when |
|---|---|---|
| One-hot | Manageable cardinality; each category has enough samples | High cardinality (sparsity, memory); rare categories (unstable coefficients) |
| Label (nominal) | Model can handle arbitrary integer structure (trees) | Linear models — injects false order |
| Label (ordinal) | The order is known and uniform | Order is noisy or non-uniform gaps |
| Target | Categories have enough samples; no leakage | Rare categories (overfitting); test-time unseen categories; leakage across folds |
| Embeddings | Lots of data; joint training | Small data; cold-start new categories |

## 2.3 Mathematical Formulation

### One-Hot Encoding

For a categorical variable $x \in \{c_1, \ldots, c_k\}$, define $\phi(x) \in \{0,1\}^k$ with $\phi_j(x) = \mathbb{1}[x = c_j]$.

**Dummy variable trap.** With an intercept $b$, and $\phi \in \{0,1\}^k$ always summing to 1, the columns of $\phi$ are linearly dependent with the intercept: $\sum_j \phi_j(x) = 1$ for all $x$. The design matrix has rank $k$, not $k+1$; one column must be dropped. Typically drop the "reference" category — its effect is absorbed into the intercept.

### Label Encoding

$\phi(x) = \text{index}(x) \in \{0, 1, \ldots, k-1\}$. Order depends on implementation (alphabetical in scikit-learn, first-appearance in some others). For ordinal variables, specify the order explicitly.

### Target Encoding (Mean Encoding)

Replace category $c$ with

$$
\hat\mu_c = \frac{1}{|I_c|} \sum_{i \in I_c} y_i
$$

where $I_c = \{i : x_i = c\}$.

**Smoothed target encoding.** To prevent overfitting on rare categories, shrink toward the global mean $\bar y$:

$$
\hat\mu_c^{\text{smooth}} = \frac{|I_c| \hat\mu_c + m \bar y}{|I_c| + m}
$$

where $m$ is a smoothing parameter. For large $|I_c|$, $\hat\mu_c^{\text{smooth}} \to \hat\mu_c$. For small $|I_c|$, it shrinks toward $\bar y$. This is a Bayesian-flavored estimator: $\bar y$ is the prior, category statistics update it.

**Out-of-fold (OOF) target encoding.** To prevent the target of row $i$ from being used to encode row $i$'s category (which is label leakage), split data into folds. To encode rows in fold $k$, compute $\hat\mu_c$ using only rows *not* in fold $k$. At test time, use the full training set.

**Leave-one-out target encoding.** Extreme form: encode each row using the mean of all other rows of the same category.

$$
\hat\mu_c^{(i)} = \frac{\sum_{j \in I_c, j \neq i} y_j}{|I_c| - 1} = \frac{|I_c| \hat\mu_c - y_i}{|I_c| - 1}
$$

### Embedding Encoding

Learn a matrix $E \in \mathbb{R}^{k \times d}$ (one row per category, $d$-dimensional embedding). The encoding is $\phi(x) = E_{\text{index}(x), :}$.

Training: if the embedding is a layer inside a neural network, $E$ is optimized along with all other parameters via gradient descent. For a categorical input $x = c_j$, the forward pass fetches $E_{j,:}$; the backward pass updates only $E_{j,:}$ for that row.

Initialization: random (Gaussian, uniform). Dimension $d$: rule of thumb $\min(50, (k+1)/2)$, or $d \sim \sqrt[4]{k}$. Not sensitive in practice; cross-validate.

### Frequency and Count Encoding

Encode each category with its frequency in training data: $\phi(x) = |I_x| / n$. Cheap, informative when frequency correlates with the target. Not order-preserving across categories with the same count (so ties must be handled).

### Binary / Base-N Encoding

Represent the label-encoded integer in base 2 (binary) or base $N$ as multiple columns. For 1000 categories, binary encoding needs only $\lceil \log_2 1000 \rceil = 10$ columns instead of 1000 one-hot columns. However, the bit columns have no real semantics for a linear model — typically only useful for tree models.

### Hashing Encoding (The Hashing Trick)

Apply a hash function $h: \text{Category} \to \{0, 1, \ldots, d-1\}$ and one-hot into $d$ dimensions:

$$
\phi_j(x) = \mathbb{1}[h(x) = j]
$$

Collisions occur but are typically benign in high $d$. Used heavily in online learning systems (Vowpal Wabbit, large-scale ad CTR models) where the vocabulary is unbounded and unknown in advance.

## 2.4 Worked Examples

### Example 1: One-Hot Encoding and the Dummy Trap

Dataset:

| Row | Color | Price |
|---|---|---|
| 1 | Red | 10 |
| 2 | Green | 12 |
| 3 | Blue | 15 |
| 4 | Red | 11 |

One-hot encoding with all 3 categories:

| Row | is_Red | is_Green | is_Blue | Price |
|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 10 |
| 2 | 0 | 1 | 0 | 12 |
| 3 | 0 | 0 | 1 | 15 |
| 4 | 1 | 0 | 0 | 11 |

With a linear model including intercept, the matrix $[\mathbf{1}, \text{is\_Red}, \text{is\_Green}, \text{is\_Blue}]$ has $\mathbf{1} = \text{is\_Red} + \text{is\_Green} + \text{is\_Blue}$ for every row — rank-deficient. Drop one column (say `is_Blue`) to get:

| Row | is_Red | is_Green | Price |
|---|---|---|---|
| 1 | 1 | 0 | 10 |
| 2 | 0 | 1 | 12 |
| 3 | 0 | 0 | 15 |
| 4 | 1 | 0 | 11 |

Now the intercept captures the mean price for Blue, `is_Red`'s coefficient captures the *difference* between Red and Blue, etc.

### Example 2: Target Encoding with Smoothing

Dataset (`Merchant` → `Default`):

| Merchant | Rows | Defaults | Rate |
|---|---|---|---|
| A | 1000 | 50 | 0.05 |
| B | 200 | 15 | 0.075 |
| C | 5 | 3 | 0.60 |
| D | 2 | 0 | 0.00 |

Global mean default rate: $\bar y = (50+15+3+0)/(1000+200+5+2) = 68/1207 \approx 0.0563$.

Raw encoding says merchant C has 60% default — but with only 5 rows this is statistically unreliable. Smoothed with $m = 30$:

- A: $(1000 \cdot 0.05 + 30 \cdot 0.0563)/(1030) = 51.69/1030 \approx 0.0502$
- B: $(200 \cdot 0.075 + 30 \cdot 0.0563)/(230) = 16.69/230 \approx 0.0726$
- C: $(5 \cdot 0.60 + 30 \cdot 0.0563)/(35) = 4.69/35 \approx 0.134$
- D: $(2 \cdot 0.0 + 30 \cdot 0.0563)/(32) = 1.689/32 \approx 0.0528$

The extreme values for C and D are pulled toward the global rate — far more reasonable.

### Example 3: OOF Target Encoding

Suppose we have 10 rows, two categories A (5 rows) and B (5 rows), binary labels.

With leave-one-out: to encode row 3 (category A, label 1), compute the mean of the other 4 A-rows' labels. Row 3's own label does not appear in its own encoding.

With 5-fold OOF: split into 5 folds of 2 rows each. To encode rows in fold 1, compute category means using folds 2-5 (8 rows). Encode rows in fold 2 using folds 1,3,4,5, etc. At test time, compute category means on all 10 training rows.

This prevents the model from seeing its own label through the encoding — a subtle but serious leakage otherwise.

### Example 4: Entity Embeddings

Imagine training a neural network on a retail sales dataset with `store_id` (50 stores) as a feature. Instead of one-hot encoding (50 sparse columns), allocate an embedding matrix $E \in \mathbb{R}^{50 \times 8}$ — each store gets an 8-dimensional learned vector.

Architecture sketch:
```
store_id (int) → Embedding(50, 8) → dense vector (8-d)
                                       ↓
other features → dense layer ←─────────┘
                           ↓
                       output
```

After training, the embeddings often reveal structure: nearby stores in embedding space tend to be geographically close, or share customer demographics, or have similar sales patterns. This is the key insight from Guo & Berkhahn (2016) — entity embeddings can be visualized with t-SNE and used as features in downstream models.

## 2.5 Relevance to ML Practice

**Model-specific recommendations:**

- **Linear models, logistic regression, SVM:** One-hot for low-cardinality (say <100), target encoding with OOF smoothing for high-cardinality. Never label-encode nominals.
- **Tree models (XGBoost, Random Forest):** Label encoding works for any cardinality; target encoding can help on very high cardinality. One-hot also fine but wasteful on high cardinality.
- **Neural networks:** Embeddings for high-cardinality (preferred). One-hot for low cardinality and small data.
- **CatBoost:** Native ordered target statistics — built-in OOF-style target encoding with permutation-based ordering. Often the best default for high-cardinality categorical data.

**When to use each:**

- Low cardinality (< 10): one-hot is often cleanest.
- Medium cardinality (10–1000): one-hot still feasible; target encoding if linear model; label for trees.
- High cardinality (> 1000): target encoding (with OOF + smoothing), embeddings, or hashing.
- Ordinal: ordinal encoding with explicit order.

**Trade-offs:**

- One-hot explodes in memory for high cardinality but is interpretable and faithful.
- Target encoding is compact but risks leakage and overfitting on rare categories.
- Embeddings are compact and powerful but require joint training and lots of data.
- Hashing is scalable but lossy (collisions).

**Production considerations:**

- Unseen categories at inference: one-hot → zero vector (usually fine). Label encoding → raise or map to "unknown" index. Target encoding → use global mean. Embeddings → typically have a reserved "unknown" token.
- Rare categories: bucket into "other" before encoding.
- Vocabulary drift: categories appear and disappear over time. Monitor and periodically refit.

## 2.6 Common Pitfalls & Misconceptions

1. **Label-encoding nominal variables for linear models.** This injects false order (Brazil=1 < Argentina=2) that the model treats literally. Use one-hot or target encoding instead.
2. **Forgetting to drop a column.** One-hot without dropping plus intercept produces perfect multicollinearity, destabilizing linear models.
3. **Target encoding without OOF / leave-one-out.** The model trains on encodings derived from its own labels — serious leakage, CV scores will be inflated.
4. **Applying target encoding to test data using only test labels.** The encoding must come from training. At test time, use training-set category statistics.
5. **Forgetting rare-category handling.** Rare categories in target encoding have high variance estimates — smoothing is essential.
6. **Unseen categories at inference.** If your production system can see categories not in training, ensure your encoder handles them (reserved index, global mean, hash).
7. **High-cardinality one-hot.** Creates 10,000 sparse features; linear models overfit; memory explodes. Use target/embeddings/hashing.
8. **Treating IDs as features.** Raw user_id or merchant_id values are usually nonsensical as features — use aggregated statistics (target encoding) or embeddings.
9. **Mixing encoding strategies inconsistently between train and production.** The encoder must be serialized and shipped alongside the model.
10. **Comparing encodings on a fixed random seed.** Target encoding is sensitive to fold split; evaluate with repeated CV to get stable estimates.

## 2.7 Interview Questions

### Foundational

**Q1. What's the dummy variable trap?**

**A.** When you one-hot encode a categorical variable with $k$ levels into $k$ columns and include an intercept, the columns of the design matrix satisfy $\sum_j \phi_j(x) = 1$ for every row. This means one of the one-hot columns is linearly dependent with the intercept — the matrix is rank-deficient and the coefficients are not uniquely identified. Standard solution: drop one reference category, leaving $k-1$ columns. The dropped category's effect is absorbed into the intercept. For tree models and regularized linear models (L2), dropping is not strictly required.

**Q2. When is label encoding appropriate?**

**A.** For ordinal variables (sizes, grades, education levels) where integers reflect the natural order and gaps are meaningful. For nominal variables used in *tree models*, label encoding is acceptable because trees split at thresholds and can recover arbitrary groupings. It is inappropriate for nominal variables in distance-based or linear models.

**Q3. Why is one-hot encoding bad for very high-cardinality features?**

**A.** (1) Memory: 10,000 categories create 10,000 sparse columns — quickly exceeding memory at scale. (2) Sparsity: each column is active only rarely, so per-column coefficients are estimated on few samples — high variance. (3) Cold start: categories not seen in training produce a zero vector. (4) Tree models: splitting on one-hot columns at each level is slow when there are many. Alternatives: target encoding, hashing, embeddings.

**Q4. What is target encoding and why does it need regularization?**

**A.** Target encoding replaces each category with the mean of the target over rows of that category. Regularization (typically smoothing toward the global mean, or OOF computation) is needed because (a) rare categories have few rows, so the mean is noisy and will overfit; (b) using the target directly creates label leakage if not done out-of-fold. Smoothing pulls small-sample categories toward the global mean; OOF prevents each row's own label from influencing its encoding.

**Q5. Explain entity embeddings.**

**A.** Entity embeddings represent each category as a dense learned vector $v_c \in \mathbb{R}^d$, stored in an embedding matrix. The vector is jointly trained with the rest of the neural network via backprop. Embeddings capture latent relationships between categories — semantically similar categories end up near each other in embedding space. For a high-cardinality variable like `product_id`, an 8-dimensional embedding is vastly more compact than one-hot (say 50,000 dimensions) and encodes learned similarity. Guo & Berkhahn showed this beats one-hot + any model on many tabular tasks.

### Mathematical

**Q6. Derive the smoothed target encoding formula as a Bayesian estimator.**

**A.** Model category $c$'s mean target as $\mu_c$, with prior $\mu_c \sim N(\bar y, \tau^2)$ and observations $y_i \mid \mu_c \sim N(\mu_c, \sigma^2)$ for $i \in I_c$. The posterior mean:

$$
\mathbb{E}[\mu_c \mid y] = \frac{\tau^2 |I_c|}{\tau^2 |I_c| + \sigma^2} \hat\mu_c + \frac{\sigma^2}{\tau^2 |I_c| + \sigma^2} \bar y
$$

Let $m = \sigma^2 / \tau^2$. Then:

$$
\mathbb{E}[\mu_c \mid y] = \frac{|I_c| \hat\mu_c + m \bar y}{|I_c| + m}
$$

which is the smoothed encoding formula. Interpretation: $m$ is the number of "pseudo-observations" of $\bar y$ you assume before seeing data; small $|I_c|$ relative to $m$ → estimate dominated by prior.

**Q7. For a $k$-category one-hot with an intercept, show the design matrix rank is $k$, not $k+1$.**

**A.** Let the design matrix include a column of 1s (intercept) and $k$ indicator columns. For every row, exactly one indicator equals 1 and the rest are 0, so the sum of indicator columns equals the column of 1s. The $k+1$ columns satisfy a linear relation $\mathbf{1} - \sum_j \phi_j = 0$, so the rank is at most $k$. It is exactly $k$ unless additional redundancy exists. The matrix is singular for linear-least-squares without regularization.

**Q8. How does leave-one-out target encoding's formula look for row $i$ in category $c$?**

**A.**

$$
\hat\mu_c^{(i)} = \frac{\sum_{j \in I_c, j \neq i} y_j}{|I_c| - 1} = \frac{|I_c| \hat\mu_c - y_i}{|I_c| - 1}
$$

This requires $|I_c| \geq 2$; singleton categories need a fallback (global mean or special "unknown" handling).

**Q9. What's the expected computational cost of one-hot encoding vs embedding for $k$ categories?**

**A.** One-hot: sparse matrix of $k$ columns; dense dot product with weight matrix of size $(k, d_{\text{out}})$ in $O(d_{\text{out}})$ per row (only one nonzero). Storage for one row: $O(1)$ (index). Embedding lookup: $O(d)$ to fetch and $O(d)$ to use; storage $O(kd)$ for the embedding matrix. For $k \gg d$, embeddings are far more memory-efficient.

**Q10. In the hashing trick, what is the probability of collision for $n$ categories hashed to $d$ bins?**

**A.** Under a uniform random hash, each category independently lands in a bin uniformly at random. The probability that a given category collides with at least one other is $1 - (1 - 1/d)^{n-1} \approx (n-1)/d$ for $d \gg n$. Expected number of collisions among $n$ categories: $\binom{n}{2}/d$. Practical rule: choose $d \gg n$ for few collisions; choose $d \sim n / \text{collision tolerance}$ otherwise.

### Applied

**Q11. You have a `zip_code` feature with 40,000 unique values. What do you do?**

**A.** Options in rough preference order for a tabular model: (1) Target encoding with OOF + smoothing — compact, captures predictive information. (2) Bucket into larger geographies — metropolitan area, state — then one-hot. (3) Replace with aggregated statistics (mean household income of the zip, from an external dataset). (4) Embeddings if using a neural model. (5) Hashing into a smaller space if streaming/online. One-hot encoding directly is almost never appropriate at this cardinality.

**Q12. How would you handle unseen categories at production inference?**

**A.** Depends on encoder: (a) One-hot — just output a zero vector; no category triggers. Works fine mostly. (b) Label encoding — reserve an "UNK" index at training; map unseen to UNK. (c) Target encoding — use global mean. (d) Embeddings — reserve a special UNK token during training, occasionally exposing it to the model (drop 5% of known categories to UNK during training) so the UNK embedding is meaningful. Always monitor unseen-category rate in production.

**Q13. Your boss wants to add a new feature `merchant_id` to your credit card fraud model. Walk through your approach.**

**A.** (1) Distribution: how many unique merchants? How skewed? Many merchants have <10 rows. (2) Signal: is per-merchant fraud rate informative? Compute per-merchant rates and variance across merchants. (3) Strategy: if cardinality is moderate (<5k) and signal is real, target encoding with OOF + smoothing works well for trees. For a neural model, an embedding layer. (4) Rare merchants: bucket everything below a threshold into "rare_merchant." (5) Unseen merchants in production: reserved bucket. (6) Drift: merchant populations change; plan for periodic re-encoding.

**Q14. Why does CatBoost have a specific advantage with categorical features?**

**A.** CatBoost implements *ordered target statistics*: it shuffles the data into a random permutation, then for each row encodes its category using only target statistics from rows earlier in the permutation (a form of online target encoding). This automatically prevents leakage without the analyst needing to set up OOF manually. It averages multiple permutations to reduce variance. Empirically this gives strong performance on tabular data with many categorical features, with much less feature engineering.

**Q15. You're training a linear model for ad CTR. The vocabulary of ads is unbounded and grows daily. How would you encode ad IDs?**

**A.** Hashing trick. Hash ad_id to a fixed-size feature space (say $2^{20}$ buckets). This allows online learning without needing to track a growing vocabulary. Collisions are usually benign at this scale (billions of rows across millions of ads). Pair with per-ad CTR features (target-encoded with smoothing using delayed target statistics to avoid leakage).

### Debugging & Failure Modes

**Q16. After switching from label encoding to one-hot, your gradient-boosted tree model got slightly worse. Why might that be?**

**A.** (1) With many categories, one-hot produces many sparse features; each split considers one at a time, and the tree can't "group" categories efficiently — it needs many splits to reach a configuration label encoding can express in one. (2) Sparse columns have rare positive values, leading to many near-useless splits. (3) Feature importance gets diluted across many one-hot columns. Tree models can natively handle label-encoded categoricals (LightGBM, CatBoost especially) without false-order issues.

**Q17. Your target-encoded features have suspiciously good CV scores, but the test set is worse. What happened?**

**A.** Almost certainly label leakage in the target encoding. Common causes: (a) target encoding computed on the full dataset before splitting (most common); (b) no OOF — row $i$'s category mean includes $y_i$ itself; (c) smoothing is too weak for rare categories, letting them memorize their row's labels. Fix: move the encoder inside your CV pipeline, use OOF or leave-one-out, increase smoothing.

**Q18. An embedding layer for `user_id` seems to not learn anything useful — all vectors look random. What's wrong?**

**A.** Likely causes: (1) Not enough data per user — each user's embedding is updated only by their own rows; users with 1–2 rows barely get optimized. (2) Learning rate too high for the embedding layer, dominating over the signal. (3) The downstream task doesn't actually require per-user differentiation. (4) Embedding dimension mismatch: too large for the data available. Diagnostics: visualize embeddings with t-SNE after training; compute variance of embeddings across users; restrict to users with ≥k rows.

**Q19. You switched from one-hot encoding for a 5-category feature to target encoding, and your linear model's performance dropped. Why?**

**A.** One-hot gave the model 4 free parameters for the feature (one per non-reference category), letting it learn independent effects. Target encoding collapses this to a *single* scalar — a single weight. If the category-target relationship is not monotone in the encoded value, the linear model can't capture it. Target encoding is a rank-1 approximation of what one-hot can express with $k-1$ weights.

**Q20. In production, you notice one-hot column `country_US` is always 0 for last week's data. What happened?**

**A.** Almost certainly the categorical value changed upstream — perhaps "US" became "USA" or "United States," and "US" now maps to no column (or to `other`). The encoder, fit on training, expected "US." Fixes: (1) monitor distribution of input categorical values; (2) ensure upstream data contracts; (3) add an "other" bucket at training; (4) retrain periodically.

### Probing Follow-ups

**Q21. Can target encoding be safer than one-hot in terms of overfitting?**

**A.** Yes and no. Target encoding uses one column regardless of cardinality, so it has lower parameter count — less overfitting from that angle. But the encoding itself is estimated from the target, so if not regularized, it encodes label information directly into features — potentially far more overfitting-prone. Properly regularized target encoding (OOF + smoothing) can outperform one-hot on high-cardinality; naive target encoding catastrophically overfits.

**Q22. Why don't we just use hashing for everything?**

**A.** Hashing is fast and scalable but: (a) collisions inject noise that tree models and interpretable linear models dislike; (b) you lose the ability to inspect per-category coefficients; (c) cold-start and drift are handled silently — you don't notice when the feature distribution changes. Hashing is ideal when vocabulary is unbounded and inference latency matters (online ad systems), not for production tabular analytics.

**Q23. How does frequency encoding compare to target encoding?**

**A.** Frequency encoding uses only input distribution — no target — so it's leakage-free and works at inference without labels. It captures "popularity" but not target relationship. Often used alongside target encoding (not instead of) because the two capture different information. Target encoding dominates when the category-target link is informative; frequency encoding wins when it's popularity that matters (e.g., rare events on rare categories).

**Q24. Two categorical features, `city` and `state`, are highly related. How should you encode them?**

**A.** Depending on the model: (1) For linear models, one-hot both and let the model handle correlation — use L2 regularization to stabilize. (2) Better: hierarchical encoding — target-encode `city` and also include target-encoded `state` as a fallback. Or bucket rare cities into their states. (3) For neural models, embedding both and letting the network learn. (4) Consider explicitly modeling the hierarchy with nested target encoding (shrink city encoding toward state encoding). The key is that the two features encode overlapping information; don't double-count and beware of collinearity.

**Q25. Can you combine encodings? When is this beneficial?**

**A.** Yes, and often it is. Examples: (1) One-hot + target encoding in parallel — lets the model use both the identity of the category and its target-related statistics. (2) Hashing + target encoding — hashing for the identity (fast lookup), target for signal. (3) Embeddings + target encoding as a precomputed feature fed to the embedding layer as a conditioning signal. Benefit: richer representation; cost: higher dimensionality and potential redundancy. Use sparingly and validate with CV.

---

# Chapter 3: Scaling & Normalization

## 3.1 Motivation & Intuition

Features in real datasets come in wildly different units and ranges. `age` might range 18–90; `income` ranges $0 to $1,000,000; `number_of_clicks` ranges 0 to 10⁶. If you feed these raw into a distance-based algorithm (k-NN, k-means), `income` will dominate the Euclidean distance — the other features effectively don't matter. If you feed them into gradient descent, parameters for small-scale features will require much larger learning rates than parameters for large-scale features, or vice versa — convergence will be slow and unstable.

**Concrete example.** A customer churn model with features `age` (18–90) and `monthly_spend` (0–10000). The Euclidean distance between customer A (age=25, spend=100) and customer B (age=26, spend=9000) is dominated by the $8900 spending difference; the age difference of 1 is completely drowned out. k-NN would say A is nearly identical to B — wrong for most business purposes.

Scaling transforms features to comparable ranges so that no feature automatically dominates. Normalization more broadly refers to transforming the distribution — centering, reducing variance, making Gaussian-like, making bounded.

Different scaling choices have different effects:
- Standardization preserves distribution shape, just centers and scales.
- Min-max forces everything into a bounded range.
- Robust scaling uses quantiles, resilient to outliers.
- Power transforms change the distribution shape (skew removal).
- Quantile transforms map to a uniform or Gaussian distribution regardless of original shape.

## 3.2 Conceptual Foundations

### Why Scaling Matters Differently for Different Algorithms

**Scale-sensitive:**
- **Distance-based:** k-NN, k-means, hierarchical clustering, SVM with RBF kernel — distance is the core operation, and it inherits feature-scale imbalances.
- **Gradient descent:** With feature scales differing by orders of magnitude, the loss landscape becomes a long narrow valley; gradient descent zigzags. Scaling → circular contours → fast convergence. Neural networks, logistic regression, SVM with gradient methods.
- **Regularized linear models:** L2 regularization applies the same $\lambda$ to all coefficients; if features are on different scales, regularization penalizes them unequally. Scale first.
- **PCA:** Variance is maximized along principal components; a high-variance feature (simply due to scale) dominates PCA direction, regardless of information content.

**Scale-invariant:**
- **Tree-based models:** Decision trees, random forests, gradient-boosted trees split at thresholds — any monotone transformation of a feature produces the same tree structure. Scaling is a no-op.
- **Naive Bayes (Gaussian):** Each feature is modeled independently; scaling doesn't help or hurt.
- **Rule-based systems.**

### Key Transformations

**Standardization (z-score):**

$$
z = \frac{x - \mu}{\sigma}
$$

Centers to mean 0, rescales to std 1. Preserves distribution shape.

**Min-max scaling:**

$$
x' = \frac{x - \min}{\max - \min}
$$

Scales to $[0,1]$. Sensitive to outliers (one extreme value compresses everything else).

**Robust scaling:**

$$
x' = \frac{x - \text{median}}{\text{IQR}}
$$

Uses median and IQR (interquartile range = Q75 − Q25). Resilient to outliers because quantiles are robust statistics.

**Max-abs scaling:**

$$
x' = x / \max |x|
$$

Scales to $[-1, 1]$ preserving sign and zero. Good for sparse matrices (preserves zeros).

**L2 (unit vector) normalization:** Scale each *row* so it has unit L2 norm. Used in text (TF-IDF) and some embedding pipelines. Not a scaling of columns but of rows.

**Power transforms:** Monotone transformations that stabilize variance and reduce skew. Box-Cox (positive-only), Yeo-Johnson (any real).

**Quantile transforms:** Map each value to its rank, then to a uniform $U[0,1]$ or a Gaussian. Force the distribution to have a specific shape.

### Assumptions and Consequences

| Method | Preserves | Destroys / Changes | Sensitive to |
|---|---|---|---|
| Standardization | Shape, Pearson correlations | Mean, scale | Outliers (affect $\mu, \sigma$) |
| Min-max | Order, proportions | Center, variance | Outliers (extreme values) |
| Robust | Order, shape roughly | Mean, variance | Not much |
| Max-abs | Sparsity, sign, order | Scale | Outliers |
| Power (Box-Cox/YJ) | Order (monotone) | Shape (toward Gaussian) | Requires parameter fit |
| Quantile (Gaussian) | Rank | Shape, variance relationship | Very small test sets |

## 3.3 Mathematical Formulation

### Standardization

With sample mean $\hat\mu = \frac{1}{n}\sum_i x_i$ and sample standard deviation $\hat\sigma = \sqrt{\frac{1}{n-1}\sum_i (x_i - \hat\mu)^2}$:

$$
z_i = \frac{x_i - \hat\mu}{\hat\sigma}
$$

After transformation, $\bar z = 0$, $s_z = 1$. Distribution shape is preserved: skewness and kurtosis are unchanged. If $x \sim N(\mu, \sigma^2)$, then $z \sim N(0, 1)$.

For vectors: $z = \Sigma^{-1/2}(x - \mu)$ is called *whitening* — removes both mean and covariance structure, yielding isotropic $z$.

### Min-Max Scaling

$$
x' = \frac{x - \min(x)}{\max(x) - \min(x)}
$$

For custom range $[a, b]$:

$$
x' = a + (b - a) \frac{x - \min(x)}{\max(x) - \min(x)}
$$

Sensitivity to outliers: one outlier 100× larger than the rest compresses the main distribution into a tiny subrange near 0.

### Robust Scaling

Let $Q_1, Q_3$ be the 25th and 75th percentiles and $\tilde x = \text{median}(x)$. Then $\text{IQR} = Q_3 - Q_1$.

$$
x' = \frac{x - \tilde x}{\text{IQR}}
$$

For normally distributed data, $\text{IQR} \approx 1.349 \sigma$, so robust scaling and z-score differ by a constant. But under outliers, robust scaling holds up while z-score explodes.

### Box-Cox Transformation

For $x > 0$:

$$
x^{(\lambda)} = \begin{cases} \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \log x & \lambda = 0 \end{cases}
$$

Parameter $\lambda$ is estimated by maximum likelihood assuming the transformed data is Gaussian:

$$
\hat\lambda = \arg\max_\lambda \left[ -\frac{n}{2} \log \hat\sigma^2_\lambda + (\lambda - 1) \sum_i \log x_i \right]
$$

where $\hat\sigma^2_\lambda$ is the sample variance of $\{x_i^{(\lambda)}\}$. The second term is the Jacobian of the transformation (log-likelihood adjustment for non-unit Jacobian).

Special cases: $\lambda=1$: identity (shifted). $\lambda=0$: log. $\lambda=0.5$: square root. $\lambda=-1$: reciprocal.

### Yeo-Johnson Transformation

Extends Box-Cox to handle non-positive values:

$$x^{(\lambda)} = \begin{cases}
\frac{(x + 1)^\lambda - 1}{\lambda} & x \geq 0, \lambda \neq 0 \\
\log(x + 1) & x \geq 0, \lambda = 0 \\
-\frac{(-x + 1)^{2-\lambda} - 1}{2 - \lambda} & x < 0, \lambda \neq 2 \\
-\log(-x + 1) & x < 0, \lambda = 2
\end{cases}$$

Continuous at 0 and for all $\lambda$. Parameter fit by MLE as in Box-Cox.

### Quantile Transformation

Step 1: compute empirical CDF $\hat F(x) = \frac{1}{n}\sum_i \mathbb{1}[x_i \leq x]$.
Step 2: map $x \mapsto \hat F(x)$, yielding a uniform $U[0,1]$ output.
Step 3 (optional): map to Gaussian via $\Phi^{-1}(\hat F(x))$, yielding a standard normal output.

This is a nonlinear, monotonically increasing transformation. It perfectly normalizes the marginal distribution regardless of original shape. Destroys relative magnitudes — only ranks survive.

At test time, the same training-fit quantile function is applied. New test values outside the training range are clipped to the min/max.

### Unit Vector (L2) Row Normalization

For row $x_i$:

$$
x_i' = x_i / \|x_i\|_2
$$

All rows then lie on the unit sphere. Commonly used in text (TF-IDF vectors) so that document length doesn't dominate dot-product similarity.

## 3.4 Worked Examples

### Example 1: Standardization vs Min-Max with an Outlier

Raw data: $x = [2, 3, 4, 5, 6, 100]$.

**Standardization:** $\mu = 20$, $\sigma \approx 38.9$. Standardized: $[-0.46, -0.44, -0.41, -0.39, -0.36, 2.06]$. The non-outliers are clustered around $-0.4$; the outlier sits at 2.06. Other features (if present) still have reasonable range.

**Min-max:** $\min = 2, \max = 100$, range 98. Scaled: $[0, 0.010, 0.020, 0.031, 0.041, 1.0]$. The non-outliers are crammed into $[0, 0.04]$ — essentially useless precision.

**Robust:** median = 4.5, IQR = $5 - 3 = 2$. Scaled: $[-1.25, -0.75, -0.25, 0.25, 0.75, 47.75]$. Non-outliers are spread $[-1.25, 0.75]$ — good dynamic range. The outlier stands out at 47.75 but doesn't distort the others.

**Takeaway.** Under outliers, robust scaling preserves non-outlier information. Standardization partially compresses; min-max catastrophically compresses.

### Example 2: Box-Cox Transformation Estimation

Data (right-skewed incomes): $x = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]$ (roughly Fibonacci). Log-likelihood is a function of $\lambda$:

$$
\ell(\lambda) = -\frac{n}{2} \log \hat\sigma^2_\lambda + (\lambda - 1) \sum_i \log x_i
$$

Evaluating at a few values:
- $\lambda=1$ (identity): variance $\approx 864$, $\ell \approx -\frac{10}{2}\log 864 + 0 \approx -33.9$.
- $\lambda=0$ (log): transformed data is $[0, 0.69, 1.10, 1.61, 2.08, 2.56, 3.04, 3.53, 4.01, 4.49]$, variance $\approx 2.23$, $\ell \approx -\frac{10}{2}\log 2.23 + (0-1)\sum \log x \approx -4.0 + (-1)(0 + 0.69 + 1.10 + \ldots) = -4.0 - 22.1 = -26.1$.
- $\lambda = 0.5$: values $[0, 0.83, 1.46, 2.47, 3.66, 5.21, 7.17, 9.66, 12.83, 16.89]$, variance $\approx 31.1$, $\ell \approx -\frac{10}{2}\log 31.1 + (-0.5)(22.1) \approx -17.2 - 11.05 = -28.25$.

MLE is around $\lambda \approx 0$ (log transform), consistent with the strong right skew.

### Example 3: Standardization and Gradient Descent

Consider a regression with two features: `rooms` (1–10) and `sqft` (500–5000). Loss contours before standardization are elongated along the `sqft` axis (high variance) and compressed along the `rooms` axis. Gradient descent takes many steps zigzagging.

After standardization, both axes have variance 1; the contours become nearly circular; gradient descent takes a nearly straight path to the optimum, converging in fewer iterations.

Quantitatively, the condition number of the Hessian (ratio of largest to smallest eigenvalue) governs convergence rate. For a feature matrix $X$ with column standard deviations $\sigma_1, \ldots, \sigma_d$, unscaled Hessian eigenvalues span the range of feature variances $\sigma_i^2$. After standardization all $\sigma_i = 1$; the Hessian condition number drops to the ratio of correlation eigenvalues — much smaller.

### Example 4: Quantile Transform

Original (exponential-distributed) data: $x = [0.1, 0.3, 0.8, 1.5, 2.2, 3.1, 4.5, 6.0, 8.2, 12.5]$.

Ranks (1–10). ECDF values: $\{0.1, 0.2, 0.3, \ldots, 1.0\}$ (or using $(r-0.5)/n$: $0.05, 0.15, 0.25, \ldots, 0.95$).

For Gaussian output, apply $\Phi^{-1}$:
$x' = \Phi^{-1}([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])$
$\approx [-1.64, -1.04, -0.67, -0.39, -0.13, 0.13, 0.39, 0.67, 1.04, 1.64]$

The output is nearly standard normal, regardless of the input's exponential shape.

## 3.5 Relevance to ML Practice

**Which algorithms need which scaling:**

| Algorithm | Standardization | Min-max | Notes |
|---|---|---|---|
| Linear/logistic regression | ✓ | (ok) | Especially with regularization |
| SVM | ✓✓ | ✓ | Critical for RBF kernel |
| k-NN, k-means | ✓✓ | ✓✓ | Distance-based — scaling is mandatory |
| Neural networks | ✓ | ✓ | Helps gradient descent |
| PCA | ✓ | — | Always center; often scale |
| Tree models | — | — | Scale-invariant |
| Naive Bayes | — | — | Doesn't help |

**When to use each scaling:**

- **Standardization:** Default choice for most scale-sensitive algorithms. Works well with Gaussian-like features.
- **Min-max:** When bounded input is needed (e.g., image pixel values to [0,1] for neural networks). Avoid with outliers.
- **Robust:** When data has outliers or long tails.
- **Power (Box-Cox/YJ):** When features are heavily skewed and you want to stabilize variance. Very useful before linear models that assume Gaussian residuals.
- **Quantile (Gaussian):** Nuclear option for producing Gaussian-shaped features. Useful for distance-based algorithms on heterogeneous data.

**Trade-offs:**

- **Standardization:** Simple, preserves shape. Sensitive to outliers.
- **Min-max:** Bounded output, sensitive to outliers, doesn't center.
- **Robust:** Resilient, but ignores useful distributional info.
- **Power/Quantile:** Very effective at normalizing shape, but complex and can hide structure.

**Train-test protocol.** Always fit the scaler on training data only and apply to validation/test. Leaking test statistics into the scaler inflates scores. In scikit-learn, wrap with a `Pipeline` so cross-validation handles this correctly.

**Production considerations.** Serialize the fitted scaler (means, stds, quantiles, or $\lambda$). At inference, apply the *same* transform. Monitor drift: if production feature means diverge from training means, the scaling becomes stale.

**Target scaling.** For regression, it is sometimes useful to also scale the target — especially for neural networks with bounded activation functions, or for numerical stability. Remember to invert the scaling at inference time.

**Sparse data.** Standardization destroys sparsity (subtracting a nonzero mean makes all zeros nonzero). For sparse data, use max-abs scaling (centered at zero, preserves zeros) or standardize without centering (`StandardScaler(with_mean=False)`).

## 3.6 Common Pitfalls & Misconceptions

1. **Fitting the scaler on the full dataset (including test).** Leakage. Always fit on training only, even inside cross-validation folds.
2. **Not scaling for scale-sensitive algorithms.** k-NN or SVM without scaling will be dominated by whichever feature has the largest raw magnitude.
3. **Scaling when using tree models.** Wasted compute, no benefit — and sometimes mild harm (loss of interpretability).
4. **Using min-max with outliers.** One extreme value crushes the rest into a tiny range.
5. **Applying Box-Cox to negative or zero values.** Box-Cox requires strict positivity. Use Yeo-Johnson or a shift.
6. **Forgetting to inverse-scale predictions.** If you scaled $y$, you must unscale model outputs before reporting.
7. **Scaling the test set with its own statistics.** Biases results, usually optimistically.
8. **Assuming standardization makes features Gaussian.** Standardization preserves distribution shape. If the feature was bimodal before, it's bimodal after. For Gaussian-like output, use power or quantile transforms.
9. **Dropping sparsity by accidentally centering sparse data.** Makes memory usage explode. Use max-abs or `with_mean=False`.
10. **Quantile transforming time series.** The quantile transform fit on the training window may not generalize if the distribution shifts over time — use rolling windows or explicit time-aware normalization.
11. **Not checking the effect of the scaler on specific features.** A Box-Cox fit on a bimodal feature may make things worse. Always inspect transformed distributions.
12. **Using different scalers for different features without justification.** Scaling is a modeling choice; document and CV-validate.

## 3.7 Interview Questions

### Foundational

**Q1. Why do we need feature scaling?**

**A.** Scale-sensitive algorithms (distance-based, gradient-based, regularized) weight features by their numeric magnitude. Without scaling, features with large raw ranges dominate distance computations, gradient magnitudes, and regularization penalties. Scaling puts features on comparable footing so the algorithm can learn the right relative importance from data rather than inheriting it from arbitrary units.

**Q2. Which algorithms are scale-invariant?**

**A.** Tree-based models (decision trees, random forests, gradient-boosted trees) — they split at thresholds, and any monotone transformation of a feature preserves the tree structure. Naive Bayes is also insensitive. All distance-based, gradient-based, and regularized linear models are scale-sensitive.

**Q3. What's the difference between standardization and min-max scaling?**

**A.** Standardization subtracts the mean and divides by the standard deviation, giving a distribution centered at 0 with std 1 but unbounded range. Min-max scales to a fixed range (typically [0,1]) but doesn't center or control variance. Standardization is more common for ML because it preserves shape and is robust-ish; min-max is useful when bounded inputs are required (image pixels, some neural-network inputs) but is very sensitive to outliers.

**Q4. When should you use robust scaling?**

**A.** When features have outliers or heavy tails. Robust scaling uses median and IQR — both resistant to outliers — so one extreme value doesn't distort the transformation. Useful on financial data, sensor data, web analytics (power-law distributions).

**Q5. What does Box-Cox do and when do we use it?**

**A.** Box-Cox is a family of power transformations parameterized by $\lambda$ that converts positive, skewed data toward Gaussianity. Common special cases: $\lambda=0$ is log, $\lambda=0.5$ is square root, $\lambda=1$ is identity. It requires strict positivity; Yeo-Johnson extends it to all reals. We use it to stabilize variance (make residuals more Gaussian), reduce skew (often helpful for linear models), and normalize distributions before distance-based operations.

### Mathematical

**Q6. Derive why standardization affects gradient descent convergence.**

**A.** Consider least-squares $L(w) = \frac{1}{2n}\|Xw - y\|^2$. The Hessian is $H = \frac{1}{n} X^\top X$. Convergence rate of gradient descent is governed by the condition number $\kappa(H) = \lambda_{\max}/\lambda_{\min}$. With unscaled features, $H$'s eigenvalues span the range of feature variances $\sigma_i^2$: a feature with std 1000 vs another with std 0.1 gives $\kappa \gtrsim 10^8$. After standardization, all $\sigma_i = 1$; $\kappa(H) = \kappa(\hat\Sigma)$ (correlation matrix), which is typically bounded. The number of gradient descent iterations to converge is $O(\kappa \log(1/\epsilon))$ (worst case).

**Q7. Derive the Box-Cox MLE objective.**

**A.** Assume $y = x^{(\lambda)} \sim N(\mu, \sigma^2)$. The likelihood in the *original* space requires a Jacobian correction:

$$
p(x \mid \lambda, \mu, \sigma^2) = p(y \mid \mu, \sigma^2) \left| \frac{dy}{dx} \right|
$$

With $y = (x^\lambda - 1)/\lambda$, $dy/dx = x^{\lambda - 1}$. Log-likelihood:

$$
\ell = -\frac{n}{2}\log(2\pi \sigma^2) - \frac{1}{2\sigma^2}\sum_i (y_i - \mu)^2 + (\lambda - 1)\sum_i \log x_i
$$

Profiling out $\mu$ and $\sigma^2$ (at their MLEs given $\lambda$): $\hat\mu = \bar y$, $\hat\sigma^2 = \frac{1}{n}\sum_i (y_i - \bar y)^2$. Substituting:

$$
\ell_{\text{profile}}(\lambda) = -\frac{n}{2}\log \hat\sigma^2_\lambda + (\lambda - 1)\sum_i \log x_i + \text{const}
$$

Maximize over $\lambda$.

**Q8. How does min-max scaling respond to outliers?**

**A.** $x' = (x - \min)/(\max - \min)$. If one data point is 100× larger than the rest, $\max$ jumps, and the denominator becomes dominated by the outlier. The rest of the data is crammed into a tiny range near 0. Quantitatively, if 99% of data is in $[0,10]$ and one point is 1000, after min-max scaling the 99% is in $[0, 0.01]$ — essentially all zeros.

**Q9. Show that for Gaussian data, IQR ≈ 1.349 σ.**

**A.** For $X \sim N(0,1)$, $Q_1 = \Phi^{-1}(0.25) \approx -0.6745$ and $Q_3 = \Phi^{-1}(0.75) \approx 0.6745$. So $\text{IQR} = Q_3 - Q_1 \approx 1.349$. For $X \sim N(\mu, \sigma^2)$, $\text{IQR} = 1.349 \sigma$. So robust scaling $(x - \tilde x)/\text{IQR}$ gives std of about $1/1.349 \approx 0.74$ for Gaussian data — close to but not exactly standardized.

**Q10. Prove that standardization preserves Pearson correlation between any two features.**

**A.** Pearson correlation: $\rho(X, Y) = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$. Let $X' = (X - \mu_X)/\sigma_X$, $Y' = (Y - \mu_Y)/\sigma_Y$. Both shifts and scales are affine: $X' = a X + b$ for constants $a, b$. Covariance satisfies $\text{Cov}(aX + b, cY + d) = ac \text{Cov}(X, Y)$. Thus $\text{Cov}(X', Y') = \frac{1}{\sigma_X \sigma_Y} \text{Cov}(X, Y)$. And $\sigma_{X'} = 1$, $\sigma_{Y'} = 1$. So $\rho(X', Y') = \text{Cov}(X', Y') = \rho(X, Y)$.

### Applied

**Q11. A k-NN model performs poorly. What's the first thing to check about preprocessing?**

**A.** Scaling. Without scaling, the largest-range feature dominates Euclidean distance. Apply standardization (or robust scaling if there are outliers) and re-evaluate. Also verify that categorical features are encoded appropriately (one-hot or embedding) — raw label-encoded categoricals impose false distances.

**Q12. Your neural network's loss plateaus after just a few steps. How could preprocessing help?**

**A.** Possible fixes: (1) Standardize inputs — unscaled features make gradients wildly disparate across weights. (2) Use min-max if inputs are pixels or bounded quantities. (3) For very skewed features, apply a log or Box-Cox transform before standardization. (4) Check for extreme outliers that produce huge initial gradients. (5) Ensure no constant features (zero variance) that mathematically can't produce gradient. (6) Consider batch normalization or layer normalization if the issue is in deeper layers.

**Q13. You're predicting income from features. Income is right-skewed. What do you do?**

**A.** (1) Consider log-transforming the target: $y' = \log(1 + y)$, train on $y'$, invert at inference with $\exp(\hat y') - 1$. This stabilizes variance and makes errors on high incomes less dominant. Equivalent to assuming multiplicative error in the original scale. (2) If features are similarly skewed, apply Box-Cox or Yeo-Johnson before standardization. (3) For tree models, target transform is less critical but can still help with loss design (use MAE or quantile loss on original scale, or log scale).

**Q14. You want to cluster documents with TF-IDF features. How do you scale them?**

**A.** TF-IDF vectors are sparse and non-negative. Apply L2 row normalization so each document has unit norm — then cosine similarity becomes dot product. *Do not* standardize with mean-centering because that destroys sparsity. Max-abs scaling is another option but usually L2 row-normalization is preferred for text.

**Q15. Your data is scaled at training, but at inference time you need to process one record at a time. How do you scale?**

**A.** The training-time scaler has stored the relevant statistics: means, stds, quantiles, $\lambda$, etc. Serialize the scaler object. At inference, load it and apply `transform(new_record)`. The statistics don't change based on the new data — the scaler is applied as a frozen function. This is why test-set leakage is so insidious: if the scaler was fit on test, production statistics differ and the model sees a different distribution.

### Debugging & Failure Modes

**Q16. You trained with standardization; in production, one feature has median 1000× larger than at training. What's happening?**

**A.** Distribution drift. Possible causes: (1) Upstream data change (unit change, currency change). (2) A bug — e.g., cents vs dollars. (3) New user population with different characteristics. The scaler is applying $x - \mu_{\text{train}}$ then $/\sigma_{\text{train}}$ — for a feature that's now 1000× larger, the standardized value is massive, driving predictions wildly. Fixes: (1) investigate and fix upstream; (2) if drift is real, refit scaler and retrain; (3) monitor feature statistics continuously with alerting.

**Q17. You applied Box-Cox and the model is now worse. Why?**

**A.** Possible reasons: (1) The feature has legitimate structure (bimodality, thresholds) that Box-Cox smoothed away. (2) Box-Cox chose a near-identity $\lambda$, so no real transformation happened and the added complexity hurts. (3) The estimated $\lambda$ is on a training fold that doesn't generalize. (4) You used Box-Cox on a feature where the model (tree) is scale-invariant. Diagnose by inspecting the estimated $\lambda$ and the transformed distribution.

**Q18. After scaling, the distribution still has outliers — but the scaler said it was "robust." What gives?**

**A.** Robust scaling uses median and IQR to *compute* the transformation, so the scaling coefficients are not distorted by outliers. But the transformation itself is a linear affine map — outliers remain outliers, just now at robust-scaled magnitudes (e.g., 10 IQRs away). If the goal is to also *reduce* outliers, you need additional steps: winsorization, quantile transformation, or log/power transforms.

**Q19. Your sparse matrix blew up in memory after scaling. Why?**

**A.** Standardization subtracts a mean from every value, including zeros — so a sparse matrix with 99% zeros becomes dense (99% of entries become $-\mu/\sigma \neq 0$). Memory explodes. Fixes: (1) Use `StandardScaler(with_mean=False)` which scales without centering. (2) Use `MaxAbsScaler` which preserves zeros. (3) Avoid standardization for sparse data entirely; just scale.

**Q20. You fit a quantile transformer on 1M training rows, but it's slow at inference. Why?**

**A.** The quantile transformer stores many quantile breakpoints (default 1000). At inference, it does binary search over these breakpoints per feature. For 1000 features × 1000 breakpoints = 1M lookups per inference — not negligible. Fixes: (1) Reduce number of quantiles (e.g., 100). (2) Use sparse quantile approximations. (3) Use a simpler transform (standardization or log) if acceptable. (4) Precompute a lookup table.

### Probing Follow-ups

**Q21. Why is scaling usually applied per-column, not per-row (for feature data)?**

**A.** Because each column represents a feature with its own scale and distribution — we want to normalize these disparate units. Per-row scaling (row normalization) is useful in different contexts: text (L2-normalize document vectors so length doesn't matter), embedding similarity (unit vectors for cosine). Mixing the two can destroy information: per-row standardization would subtract each row's mean, erasing shared signal.

**Q22. Does scaling help or hurt interpretability?**

**A.** Both. Scaled coefficients in a linear model can be directly compared across features (e.g., "a 1-std change in age has the same effect as a 0.5-std change in income"), which is interpretive gold. But the raw effect size in the original units is lost — you need to invert the scale to get back. Document-oriented interpretability often reports both scaled (for ranking) and unscaled (for units).

**Q23. Can scaling affect fairness?**

**A.** Yes, subtly. If a feature is used as a proxy for a protected attribute, the choice of scaling changes how strongly it contributes to distance or gradient. More concretely: standardization over the full training set uses overall statistics, which can disadvantage minority groups whose distribution differs. Robust scaling or group-specific scaling can mitigate this but may also introduce other fairness concerns. Fairness-aware preprocessing is an active research area.

**Q24. What's whitening and when is it useful?**

**A.** Whitening applies $z = \Sigma^{-1/2}(x - \mu)$, so that $z$ has identity covariance (zero mean, unit variance, zero pairwise correlation). Equivalent to PCA + standardization. Useful for: (1) models that benefit from uncorrelated inputs (shallow networks, logistic regression); (2) preprocessing in anomaly detection; (3) ICA (where whitening is a standard preprocessing step). Expensive for high dimensions; typically not used for modern deep networks (batch norm does something analogous adaptively).

**Q25. Can you design a scaler that is both robust to outliers and preserves sparsity?**

**A.** Yes. One option: max-abs scaling on non-zero entries, using a robust statistic for the max — e.g., the 99th percentile of absolute values. This yields a transform that scales to roughly $[-1,1]$, preserves zeros, and isn't distorted by single extreme outliers. Alternatively, apply log1p followed by max-abs: $x \mapsto \log(1 + |x|) \cdot \text{sign}(x) / \text{constant}$. Both work on sparse data; neither is in scikit-learn out of the box but can be implemented with `FunctionTransformer`.

---

# Chapter 4: Outlier Treatment

## 4.1 Motivation & Intuition

An outlier is an observation far from the bulk of the data. Examples: a customer with a $10 million purchase in a dataset of typical $100 purchases; a sensor reading of −200°C when the rest are 20°C; a 900-year-old age in a survey dataset.

Outliers can wreck ML models in several ways:
- **Statistical leverage:** a single outlier can pull a regression line far from the bulk of the data.
- **Scale distortion:** outliers blow up means and standard deviations, distorting scaling.
- **Gradient explosion:** in gradient-descent models, an outlier can produce enormous gradients that destabilize training.
- **k-means collapse:** one far-away outlier can become its own cluster, collapsing meaningful structure.

But outliers are not always noise — sometimes they are the signal. Fraud detection, intrusion detection, rare disease identification: outliers *are* the target. The challenge is distinguishing "error outliers" (bad data) from "phenomenon outliers" (real rare events).

**Concrete example.** A hospital dataset shows a patient with a pulse of 300 bpm. Is this a sensor error to be discarded, or a real tachycardia to be flagged? Domain knowledge is essential — you cannot decide from statistics alone.

Outlier treatment is about *choices*: detect, remove, cap, transform, or explicitly model. Each has different consequences.

## 4.2 Conceptual Foundations

### Types of Outliers

- **Univariate outliers:** extreme in one dimension.
- **Multivariate outliers:** not extreme in any single dimension but unusual in the joint distribution (e.g., a 7-year-old with a PhD).
- **Contextual outliers:** extreme only in a specific context (e.g., 80°F at the South Pole).
- **Collective outliers:** a group of points that collectively form an anomaly (e.g., a sudden cluster of log entries).

### Detection Methods

**Statistical methods:**
- **Z-score rule:** flag if $|z| > 3$ (or 2 or 4 depending on tolerance).
- **Modified Z-score (MAD-based):** robust to outliers themselves.
- **IQR rule:** flag if $x < Q_1 - 1.5 \cdot \text{IQR}$ or $x > Q_3 + 1.5 \cdot \text{IQR}$.
- **Tukey fences:** as above; 1.5 IQR for "outliers", 3 IQR for "far outliers."
- **Grubbs' test, Dixon's Q test:** formal statistical tests for single outliers under Gaussian assumption.

**Distance-based:**
- **k-NN distance:** flag points whose k-th nearest neighbor distance is large.
- **Local Outlier Factor (LOF):** measures local density deviation — flags points in sparser regions than their neighbors.

**Density-based:**
- **DBSCAN:** points not assigned to any cluster are outliers.
- **Isolation Forest:** recursively partitions feature space; outliers are isolated in few splits.

**Distribution-based:**
- **Gaussian fit + tail probability:** fit a Gaussian, flag points with low likelihood.
- **Mahalanobis distance:** multivariate generalization of z-score.

### Treatment Methods

**Deletion / trimming:**
- **Remove** the row entirely. Simple but loses data.
- **Trim top and bottom percentiles** (e.g., 1% each side).

**Winsorization:**
- Replace extreme values with the boundary value (e.g., clip to 1st/99th percentiles).
- Preserves sample size, reduces extreme influence.

**Transformation:**
- Log, Box-Cox, or quantile transform pulls outliers closer to the bulk.

**Robust modeling:**
- Use algorithms that are inherently robust (median regression, Huber loss, robust M-estimators).

**Flagging:**
- Add an `is_outlier` indicator and keep the original value.

### Assumptions and Trade-offs

Deletion assumes the outlier is bad data. Wrong if outliers are real phenomena (fraud, disease). Deletion also shrinks the dataset.

Winsorization assumes the boundary value is "close enough" — but for long-tailed real data, this can underestimate true variability.

Transformation assumes a simple functional form can normalize the distribution. Fails for genuinely bimodal or discrete-heavy-tailed data.

Robust methods cost some efficiency (larger confidence intervals under clean data) in exchange for stability under contamination.

## 4.3 Mathematical Formulation

### Z-Score

$$
z_i = \frac{x_i - \bar x}{s}, \quad \text{flag if } |z_i| > T
$$

Typical $T = 3$. For a Gaussian, $P(|Z| > 3) \approx 0.0027$, so you'd expect ~0.3% flagged even in clean data. Problem: $\bar x, s$ are themselves distorted by outliers — the method's own statistics are non-robust.

### Modified Z-Score (MAD-based)

Median Absolute Deviation: $\text{MAD} = \text{median}(|x_i - \tilde x|)$ where $\tilde x = \text{median}(x)$.

$$
z^{\text{mod}}_i = \frac{0.6745 (x_i - \tilde x)}{\text{MAD}}
$$

The constant 0.6745 makes MAD consistent with $\sigma$ for Gaussian data (since $\text{MAD} \approx 0.6745 \sigma$). Flag if $|z^{\text{mod}}| > 3.5$. Robust because median and MAD are not distorted by outliers.

### IQR Rule

$$
\text{Lower fence} = Q_1 - k \cdot \text{IQR}, \quad \text{Upper fence} = Q_3 + k \cdot \text{IQR}
$$

Flag values outside the fences. Typical $k = 1.5$ (mild) or $3$ (extreme). For Gaussian data, $k=1.5$ corresponds to about $\pm 2.7\sigma$ — comparable to $z > 3$.

### Winsorization

Define thresholds $\tau_\text{low}, \tau_\text{high}$ (often percentiles $P_1, P_{99}$). Replace:

$$
x_i^w = \begin{cases} \tau_\text{low} & x_i < \tau_\text{low} \\ x_i & \tau_\text{low} \leq x_i \leq \tau_\text{high} \\ \tau_\text{high} & x_i > \tau_\text{high} \end{cases}
$$

Preserves sample size, reduces influence of tails.

### Trimmed Mean

$$
\bar x_\alpha = \frac{1}{n - 2k} \sum_{i=k+1}^{n-k} x_{(i)}
$$

where $k = \lfloor \alpha n \rfloor$ and $x_{(1)} \leq x_{(2)} \leq \ldots \leq x_{(n)}$ are order statistics. $\alpha$-trimmed mean drops the top and bottom $\alpha$ fractions. $\alpha = 0$: ordinary mean. $\alpha = 0.5$: median.

### Huber Loss

$$
L_\delta(r) = \begin{cases} \frac{1}{2} r^2 & |r| \leq \delta \\ \delta (|r| - \frac{1}{2}\delta) & |r| > \delta \end{cases}
$$

Quadratic near 0 (efficient for small residuals), linear for large (bounded influence for outliers). Used in robust regression.

Gradient: $\frac{\partial L_\delta}{\partial r} = \begin{cases} r & |r| \leq \delta \\ \delta \cdot \text{sign}(r) & |r| > \delta \end{cases}$

So for outliers, the gradient is bounded at $\pm \delta$ — no single observation can dominate.

### Mahalanobis Distance

$$
d_M(x) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}
$$

For multivariate Gaussian, $d_M^2 \sim \chi^2_d$ where $d$ is the dimension. Flag if $d_M^2 > \chi^2_{d, 1-\alpha}$.

Problem: $\mu$ and $\Sigma$ are estimated from the same data; outliers inflate $\Sigma$, reducing their own Mahalanobis distance (masking). Fix: use robust estimators like Minimum Covariance Determinant (MCD).

### Isolation Forest

Train $T$ random trees. For each tree:
1. Select a random feature.
2. Select a random split value between min and max of that feature.
3. Recurse until each sample is isolated (leaf).

Outlier score for point $x$:

$$
s(x, n) = 2^{-\mathbb{E}[h(x)] / c(n)}
$$

where $h(x)$ is the path length from root to $x$'s leaf, and $c(n)$ is the expected path length in a random binary tree with $n$ samples. Outliers are isolated quickly (short path), giving $s$ close to 1. Normal points have $s$ closer to 0.5.

### Local Outlier Factor (LOF)

For point $x$ with neighborhood $N_k(x)$:

1. Reachability distance: $\text{rd}_k(x, y) = \max(d(x, y), d_k(y))$ where $d_k(y)$ is $y$'s k-th neighbor distance.
2. Local reachability density: $\text{lrd}_k(x) = 1 / \left[\frac{1}{k}\sum_{y \in N_k(x)} \text{rd}_k(x, y)\right]$.
3. LOF score: $\text{LOF}_k(x) = \frac{1}{k}\sum_{y \in N_k(x)} \frac{\text{lrd}_k(y)}{\text{lrd}_k(x)}$.

LOF > 1 indicates $x$ is in a sparser region than its neighbors — an outlier in the local context.

## 4.4 Worked Examples

### Example 1: Z-Score vs Modified Z-Score

Data: $x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]$.

Mean $\bar x = 14.5$, std $s \approx 30.1$. Z-scores:
- $z_1 = (1 - 14.5)/30.1 \approx -0.45$
- $z_{10} = (100 - 14.5)/30.1 \approx 2.84$

With threshold $T=3$, the outlier ($x = 100$) is *not* flagged! This is the "swamping" / "masking" problem: the outlier inflates $s$, protecting itself.

Modified z-scores: $\tilde x = 5.5$, $\text{MAD} = \text{median}(|x - 5.5|) = \text{median}(4.5, 3.5, 2.5, 1.5, 0.5, 0.5, 1.5, 2.5, 3.5, 94.5) = 2$.

- $z_1^{\text{mod}} = 0.6745 \cdot (1 - 5.5)/2 = -1.52$
- $z_{10}^{\text{mod}} = 0.6745 \cdot (100 - 5.5)/2 = 31.9$

Now the outlier is unambiguously flagged.

### Example 2: Winsorization

Data: $x = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 500, 600]$.

Percentiles: $P_5 \approx 5.6$, $P_{95} \approx 560$. Winsorize at $P_5$ and $P_{95}$:
- Values below 5.6: 5 → 5.6.
- Values above 560: 600 → 560.

Result: $[5.6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 500, 560]$. But the 500 is retained — one "outlier" survives because it's under $P_{95}$ (since $P_{95}$ is itself influenced by the outliers).

Better: winsorize at the 1st/99th percentile of a **robust** reference. Or use multiple rounds. Or use IQR-based clipping.

With IQR: $Q_1 = 7$, $Q_3 = 14$, $\text{IQR} = 7$. Upper fence = $14 + 1.5 \cdot 7 = 24.5$. Values $> 24.5$ (500, 600) get clipped to 24.5. Now truly outlier-bounded.

### Example 3: Huber Loss vs Squared Loss

Residuals: $r = [1, -1, 0.5, -0.5, 0.1, 100]$ (last one is an outlier).

Squared loss: $L_\text{sq} = 1 + 1 + 0.25 + 0.25 + 0.01 + 10000 = 10002.5$. The outlier contributes 99.97% of the loss; the gradient is dominated by that one observation.

Huber with $\delta = 1$: first five contribute $0.5 \cdot (1 + 1 + 0.25 + 0.25 + 0.01) = 1.255$. Outlier contributes $1 \cdot (100 - 0.5) = 99.5$. Still large, but the *gradient* for the outlier is bounded at $|\delta| = 1$ instead of 100 — a 100× reduction in per-sample influence.

### Example 4: Isolation Forest Intuition

Dataset: 1000 points clustered around (0,0), plus one point at (100, 100).

A random split on any feature with a random threshold will almost always separate the (100,100) point from the cluster in one or two splits — its average tree path length is ~2. Normal points need ~10 splits to be isolated from their neighbors (path ~10). The ratio gives scores near 1 for the outlier, ~0.5 for normal points.

This works without any distribution assumption — just the principle that outliers are easier to isolate.

## 4.5 Relevance to ML Practice

**When to treat outliers:**

- **Always investigate first.** Is it a data entry error, a sensor fault, or a real event? Raw removal without investigation loses information.
- **Remove / trim:** When outliers are clearly errors and you have lots of data.
- **Winsorize:** When outliers are real but you want bounded influence on scaling/gradient-based models.
- **Transform (log/Box-Cox):** When the distribution is heavy-tailed legitimately (incomes, network traffic).
- **Robust models (Huber, quantile regression):** When you suspect contamination but want to keep the data.
- **Flag as feature:** When outliers are the signal (fraud, anomalies).

**By model type:**
- Tree models: less sensitive to outliers (splits at quantiles); usually no treatment needed.
- Linear/logistic regression: highly sensitive; consider Huber or removal.
- k-means: highly sensitive; one outlier can become its own cluster. Consider k-medoids or outlier removal.
- Neural networks: sensitive in early training; gradient clipping, robust losses, or preprocessing help.

**Production considerations:**

- Production data will have outliers you didn't see at training. Be prepared:
  - Clip inputs at known bounds.
  - Monitor for values outside the training range.
  - Have a fallback (e.g., return a conservative prediction).
- Validate outlier treatment on held-out data, especially the tails.

**Trade-offs:**

- **Loss of data** (deletion) vs **bias from retained contamination** (keep all).
- **Hard thresholds** (IQR rule) vs **continuous robust losses** (Huber) — the latter are usually smoother.
- **Univariate** methods (fast, simple) vs **multivariate** (LOF, Isolation Forest — catch contextual outliers at the cost of complexity).

## 4.6 Common Pitfalls & Misconceptions

1. **Blanket outlier removal.** Deleting all points with $|z| > 3$ removes ~0.3% of clean Gaussian data *as if* they were outliers. In non-Gaussian data, even more.
2. **Using non-robust detection.** Z-score fails when outliers are present (masking/swamping). Use modified z-score or IQR.
3. **Removing outliers before splitting.** Same leakage problem as imputation — use only training to define thresholds.
4. **Confusing outliers with noise.** Outliers are specific rare events; noise is ubiquitous small variation. Treat them differently.
5. **Assuming outliers are errors.** In fraud detection, outliers *are* the target. Removing them destroys the signal.
6. **Applying univariate methods to multivariate outliers.** A point at (80, 80) in a dataset where features are typically around (50, 100) is not extreme in either dimension, but is anomalous jointly. Use Mahalanobis, LOF, or Isolation Forest.
7. **Winsorizing with fragile percentiles.** When many outliers exist, percentiles are themselves biased. Use IQR-based fences or robust reference distributions.
8. **Treating outliers before understanding them.** Always plot and investigate — a batch of outliers might point to a bug.
9. **Forgetting to treat outliers at inference.** The training pipeline clipped inputs; production must do the same.
10. **Using a single threshold for all features.** Different features have different scales and distributions; per-feature thresholds are typically needed.

## 4.7 Interview Questions

### Foundational

**Q1. What is an outlier and why do they matter?**

**A.** An outlier is an observation that lies far from the bulk of the data — it could be a data-entry error, a rare real event, or a measurement fault. They matter because many ML algorithms are sensitive to them: a single outlier can shift a regression line, blow up a mean/std, dominate a gradient step, or become its own cluster. Even scale-invariant algorithms (trees) can be hurt if the outlier is a label error. Proper handling requires understanding the cause.

**Q2. What's the difference between an outlier and noise?**

**A.** Noise is small, pervasive random variation affecting every observation (e.g., measurement error). Outliers are individual extreme observations, much farther from the bulk than noise would produce. Noise is typically handled with regularization, averaging, or robust loss functions; outliers may require explicit detection, removal, or transformation.

**Q3. Why is the z-score method unreliable for detecting outliers?**

**A.** Because z-score uses sample mean and std, both of which are themselves distorted by the outliers we're trying to detect. This causes "masking" (outliers inflate the std, hiding themselves) and "swamping" (normal points near the outlier also get flagged). Robust alternatives use median and MAD (modified z-score) or IQR, which are not affected by a few extreme points.

**Q4. When is winsorization preferable to outright removal?**

**A.** When (a) the sample size is small and losing data is costly, (b) the outliers are real but you want to bound their influence on mean/std-based statistics, or (c) you want a conservative approach that preserves row count. Winsorization is particularly common in finance (returns often contain extreme events that shouldn't dominate but also shouldn't be discarded).

**Q5. How does Isolation Forest detect outliers without distributional assumptions?**

**A.** It relies on the principle that outliers are "easy to isolate" — a random split along a random feature will separate an outlier from the bulk of the data quickly. Over many random trees, outliers have shorter average path lengths from root to leaf. Because it doesn't assume any distribution (Gaussian or otherwise), it works on arbitrary feature spaces including mixed types.

### Mathematical

**Q6. Derive the expected outlier rate under the IQR rule for Gaussian data.**

**A.** For $X \sim N(0,1)$: $Q_1 = -0.6745$, $Q_3 = 0.6745$, $\text{IQR} = 1.349$. Upper fence at $k = 1.5$: $Q_3 + 1.5 \cdot \text{IQR} = 0.6745 + 2.024 = 2.698$. Lower fence: $-2.698$. $P(|Z| > 2.698) = 2 \cdot (1 - \Phi(2.698)) \approx 2 \cdot 0.00348 \approx 0.0070$. So about 0.7% of clean Gaussian data will be flagged. For $k=3$: fences at $\pm 4.72$, flagging probability $\approx 2 \times 10^{-6}$.

**Q7. Show that Huber loss reduces to squared loss for small residuals and absolute loss for large.**

**A.** Huber: $L_\delta(r) = r^2/2$ for $|r| \leq \delta$, and $L_\delta(r) = \delta(|r| - \delta/2)$ for $|r| > \delta$.

For small $r$ ($|r| \ll \delta$): $L_\delta(r) = r^2/2$ — same as squared loss (scaled by 1/2).

For large $r$: $L_\delta(r) = \delta|r| - \delta^2/2 \sim \delta |r|$ linearly (as $r \to \infty$). The gradient is $\pm \delta$ — constant in magnitude, like MAE.

This gives quadratic sensitivity (good statistical efficiency) near zero and linear (bounded influence) in the tail. $\delta$ tunes the transition.

**Q8. Derive the Mahalanobis distance from the multivariate Gaussian density.**

**A.** Multivariate Gaussian:

$$
p(x) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^\top \Sigma^{-1} (x - \mu)\right)
$$

The exponent contains $(x - \mu)^\top \Sigma^{-1} (x - \mu)$, which is $d_M^2$. For fixed density contour, $d_M$ is constant. Under the Gaussian assumption, $d_M^2 \sim \chi^2_d$, which gives thresholds for flagging. The key insight: $d_M$ accounts for feature correlations, unlike Euclidean distance.

**Q9. Why does MAD have the factor 0.6745 for Gaussian consistency?**

**A.** For $X \sim N(0, \sigma^2)$, $P(|X - \mu| < \sigma \cdot 0.6745) = 0.5$ by definition of $0.6745 = \Phi^{-1}(0.75)$. So the median of $|X - \mu|$ is $0.6745\sigma$, i.e., $\text{MAD} = 0.6745 \sigma$. Thus $\sigma = \text{MAD}/0.6745 = 1.4826 \cdot \text{MAD}$. Conversely, to standardize, we compute $(x - \tilde x) / (\text{MAD}/0.6745) = 0.6745 (x - \tilde x) / \text{MAD}$, which is the modified z-score.

**Q10. In the Huber loss, what's the optimal $\delta$?**

**A.** Classic result: if $\delta = 1.345\sigma$ where $\sigma$ is the scale of the residuals, Huber achieves 95% asymptotic efficiency under Gaussian errors while remaining robust. This is the default in most software. More generally, $\delta$ is a hyperparameter balancing efficiency (higher $\delta$ → more like OLS) and robustness (lower $\delta$ → more like MAE). Tune via CV.

### Applied

**Q11. You're building a fraud detection model. How should you treat outliers?**

**A.** Emphatically: do not remove them — they're your positive class. Instead: (1) Keep outliers as first-class training data. (2) Use models suited to imbalance (tree ensembles, anomaly detection, one-class SVM, autoencoders). (3) Create features that characterize outliers (deviations from user history, velocity features). (4) Use stratified splits so outliers are represented in train and test. (5) Evaluate with precision-recall metrics (AUC-PR), not accuracy.

**Q12. You have a regression target (house prices) that has a few extreme values. What should you do?**

**A.** Options: (1) Log-transform the target: $\log(1 + y)$ compresses the tail and makes squared loss comparable across magnitudes. Invert at inference. (2) Use Huber or quantile loss directly on the original scale. (3) Use MAE (absolute error) loss — robust but less efficient. (4) Cap predictions at a reasonable maximum if the business doesn't care about extreme precision in the tail. (5) Consider two-stage models: classify high-price vs normal, then regress within each.

**Q13. A linear regression gives wildly different coefficients each run on different train subsets. Why might outliers be the issue?**

**A.** OLS has unbounded leverage — a single outlier can pull coefficients dramatically. If your random train splits include or exclude particular outliers, coefficients shift. Diagnose with leverage plots (Cook's distance). Fix: remove or Winsorize the outliers, use robust regression (Huber, RANSAC), or add L2 regularization to stabilize.

**Q14. In a time series (server CPU usage), you see a few huge spikes. Are they outliers?**

**A.** Depends on context: (a) If they represent real load events (deployment, traffic spike), they're signal — don't remove. Model them explicitly or use features that capture them. (b) If they're data glitches (sensor errors, restart artifacts), they're noise — consider removing or smoothing. (c) If you're forecasting, decide whether the forecast should predict spikes (include them and use quantile loss) or ignore them (use robust forecasting). Often you use a hybrid: detect spikes as anomalies, forecast the baseline separately.

**Q15. Your dataset is imbalanced AND has outliers. What's your approach?**

**A.** Distinguish first: are the minority class outliers, or separate issues? (1) If the minority class is defined by being outliers (fraud, anomaly), treat as supervised outlier detection; don't remove them. (2) If the minority class is a real small group and outliers are separate (data errors), handle separately: remove clear errors first, then address imbalance (oversample, class weights, focal loss). (3) Use metrics aware of both: precision-recall for imbalance, robust losses for outliers.

### Debugging & Failure Modes

**Q16. You removed outliers and the model got worse. Why?**

**A.** Possible reasons: (1) The outliers were real signal — genuinely informative rare events. Common in fraud, anomaly detection, rare classes. (2) Outlier removal reduced sample size too much. (3) The threshold was too aggressive and removed borderline useful points. (4) The test set still has the outliers, so training on a "cleaner" distribution doesn't generalize. (5) The outliers carried information through correlations with other features (removing them shifted multivariate structure).

**Q17. Your z-score-based outlier detector isn't catching obvious outliers. What's wrong?**

**A.** The outliers are probably "masking" — inflating std enough that they themselves don't exceed $3\sigma$. Classic fix: use robust detection (modified z-score with MAD, or IQR rule). Alternatively, iteratively remove detected outliers and recompute statistics (iterative z-score).

**Q18. Clip-based outlier handling in training but not at inference led to a big train-test gap. How?**

**A.** If you clipped training at the 1st/99th percentile but passed raw values at inference, the model has never seen values in the tails and will extrapolate poorly — or clip internally in unexpected ways. Fix: apply the same clipping in production. Serialize the bounds along with the model.

**Q19. Isolation Forest flags 20% of the data as outliers. Is something wrong?**

**A.** Likely yes. The default `contamination` parameter is around 0.1 (10%). A 20% flag rate suggests either (a) the contamination parameter was set too high, (b) the data has many distinct subpopulations that Isolation Forest is (incorrectly) treating as anomalies, or (c) the feature space is so high-dimensional that many points look "isolated." Solutions: lower contamination, investigate flagged clusters for substructure, reduce dimensionality or use more appropriate detector (LOF, DBSCAN).

**Q20. After adding an `is_outlier` flag, the model performs the same. What's happening?**

**A.** A few possibilities: (1) The outlier flag is redundant with existing features the model already uses. (2) The outliers are not actually related to the target. (3) The flag is sparse (few outliers) and adds little signal. (4) A tree model already handles outliers via its splits, so the explicit flag is uninformative. Diagnose: check feature importance, look at rows where flag=1 and see if their predictions differ from flag=0 predictions.

### Probing Follow-ups

**Q21. Is an outlier a property of a single data point or of the whole dataset?**

**A.** Of the dataset. "Outlier" is always defined relative to the rest of the data. A value of 100 is an outlier in a context where most values are 1–10, but normal in a context where values range 0–1000. This means outlier status can change with context (grouping, filtering, time), and outlier detection is inherently a data-dependent judgment.

**Q22. Why can removing outliers sometimes introduce bias?**

**A.** If the outliers are not errors but genuine tail observations from the same process, removing them biases estimates toward the center and understates variability. For example, in wealth data, removing the top 1% to "clean" the data makes the mean look smaller and variance look much smaller — a misleading summary. The underlying population has heavy tails, and pretending otherwise produces inaccurate models of reality.

**Q23. How does gradient clipping relate to outlier treatment?**

**A.** Gradient clipping caps the gradient norm at each training step. It doesn't remove outliers from the data, but it prevents any single step's gradient — often driven by outliers — from dominating training. It's a form of online robustness, similar in spirit to Huber's bounded gradient. Especially important for RNNs/transformers where gradient spikes cause instability.

**Q24. Can you have too many outliers for outlier-detection methods to work?**

**A.** Yes. Classical methods assume outliers are a small minority (~5–10% max). Beyond that, the "normal" distribution estimates are themselves contaminated and outliers become indistinguishable from the bulk. Methods with higher breakdown points (like MCD, with up to 50% breakdown) exist, but even they degrade eventually. When contamination is high, it's often better to reframe the problem: what looks like an outlier-heavy distribution may actually be a mixture of multiple populations that need separate modeling.

**Q25. How should outlier treatment interact with cross-validation?**

**A.** Outlier detection should be fit *inside* each CV fold, on training data only, and applied to validation data. Otherwise you leak information about which points are "unusual" in the full dataset. In practice: wrap outlier handling in a scikit-learn `Pipeline` or equivalent so CV refits on each fold. Be wary of methods (LOF, isolation forest) that depend on the global dataset structure — refit per fold.

---

# Chapter 5: Datetime Processing

## 5.1 Motivation & Intuition

Datetime features are everywhere: purchase timestamp, account creation date, event time, expiration date. Raw datetimes (like `2024-11-15 14:32:00`) cannot be fed directly into most ML models — they're not numbers in a meaningful sense. But they encode enormous signal: seasonality (holiday shopping spikes), day-of-week effects (weekend vs weekday), time-of-day patterns (lunch rush), trends (revenue growing year over year), lifecycle effects (months since account creation).

Badly processed datetimes can destroy models:
- Using raw Unix timestamps as a feature makes the model treat time linearly, missing cyclic patterns.
- Using day-of-week as integer 0–6 forces 0 (Monday) and 6 (Sunday) to appear maximally different when they're in fact adjacent days.
- Ignoring timezone makes global data incoherent.
- Ignoring holidays misses enormous demand spikes in retail, finance, travel.

**Concrete example.** A ride-share ETA prediction model needs to know:
- Time of day (rush hour vs 3 AM)
- Day of week (Monday commute vs Saturday evening)
- Is it a holiday?
- Weather-related seasonality (is it summer?)
- How recent is the data (trend effects)?

A single `timestamp` column contains all of this, but only if processed correctly.

## 5.2 Conceptual Foundations

### Components of a Datetime

- Year, month, day, hour, minute, second, microsecond
- Day of week, day of year, week of year
- Quarter, season
- Timezone offset, DST status
- Holiday / business day status
- Time since a reference event (account creation, subscription start)
- Time to a future event (subscription expiry)

Each has different semantics and ML implications.

### The Cyclic Problem

Many datetime features are naturally cyclic:
- Hour: 23 → 0 (midnight wraps around)
- Day of week: Sun (6) → Mon (0)
- Month: Dec (12) → Jan (1)
- Day of year: Dec 31 → Jan 1

Naive integer encoding breaks this — the model sees hour 23 and hour 0 as 23 units apart, but they're actually 1 hour apart. The fix is **cyclical encoding** using sine/cosine pairs.

### Time Since / Time Until

Relative temporal features are often more predictive than absolute dates:
- `days_since_signup`: captures customer maturity.
- `days_since_last_purchase`: captures engagement/churn.
- `days_until_contract_renewal`: captures pending decisions.

These features transform datetimes into continuous duration measures that carry ML-usable information.

### Holiday and Event Features

Holidays produce major demand shifts:
- `is_holiday` (binary)
- `days_to_next_holiday` (continuous)
- `is_black_friday_week` (binary)
- Event-type labels (religious, national, shopping)

For retail, finance, travel, ignoring holidays is a major modeling error.

### Trend and Seasonality Decomposition

Time series commonly have:
- **Trend:** long-run direction (growth, decay).
- **Seasonality:** repeating patterns (yearly, weekly, daily).
- **Cyclical:** non-seasonal oscillations (business cycles).
- **Residual / irregular.**

These can be extracted as features (using STL decomposition, Fourier terms) or handled by the model (e.g., tree model with year/month/week features).

### Time Zone and DST Issues

- Timestamps should be stored in UTC and converted to local time for local features (hour of day, is_business_hour).
- Daylight Savings Time causes 23- or 25-hour "days" twice a year — some regions only.
- Cross-timezone data (global e-commerce) needs careful alignment.

### Assumptions and What Breaks

- **Cyclical encoding assumes cyclic behavior.** If a feature has a clear non-cyclic anchor (e.g., the year is monotonic), don't cyclicalize.
- **Holiday features depend on the calendar.** Different countries / regions have different holidays. Stale holiday lists miss recent additions.
- **Time-since features accumulate.** A customer's `days_since_signup` grows monotonically; make sure this reflects real behavior (long-tenured vs new customers).
- **Train-test leakage with time-based splits.** Using random splits on time-series data leaks future information into training. Use time-ordered splits.

## 5.3 Mathematical Formulation

### Cyclical Encoding

For a cyclic feature $t$ with period $T$ (e.g., hour with $T=24$, day-of-week with $T=7$):

$$
\phi_{\sin}(t) = \sin\left(\frac{2\pi t}{T}\right), \quad \phi_{\cos}(t) = \cos\left(\frac{2\pi t}{T}\right)
$$

The pair $(\sin, \cos)$ maps the cyclic variable onto a circle in 2D. Any two times $t_1, t_2$ that are "close" on the cycle (1 hour apart at midnight) are close in 2D Euclidean distance.

Distance between $t_1$ and $t_2$ after encoding:

$$
d^2 = (\sin(2\pi t_1/T) - \sin(2\pi t_2/T))^2 + (\cos(2\pi t_1/T) - \cos(2\pi t_2/T))^2 = 2 - 2\cos(2\pi(t_1 - t_2)/T)
$$

Minimum at $t_1 - t_2 = 0$ or $T$; maximum at $t_1 - t_2 = T/2$.

### Fourier Features for Multiple Harmonics

A single $(\sin, \cos)$ pair captures the fundamental cycle. For richer patterns (bimodal daily traffic, morning and evening peaks), add higher harmonics:

$$
\phi_k(t) = \sin(2\pi k t / T), \cos(2\pi k t / T), \quad k = 1, 2, \ldots, K
$$

$K$ harmonics give $2K$ features capturing progressively finer cyclic detail. This is a Fourier series basis.

### Time-Since Features

Given a reference datetime $t_\text{ref}$ and observation time $t_i$:

$$
\Delta_i = (t_i - t_\text{ref}) / \text{unit}
$$

The unit can be seconds, hours, days, etc. Often useful to also apply log:

$$
\Delta_i^{\log} = \log(1 + \Delta_i)
$$

which compresses long durations (avoids linearity assumption).

### Holiday Indicators

Binary: $h_i = \mathbb{1}[t_i \in H]$ for holiday set $H$.
Distance: $d^{\text{hol}}_i = \min_{h \in H} |t_i - h|$ (days to nearest holiday).

### Week-of-Year with Leap-Year Handling

Most calendars have 52 or 53 weeks. Use ISO week number for consistency; watch out for the boundary (week 53 of year X vs week 1 of year X+1).

### Business Days and Trading Days

Define $B$ = set of business days (exclude weekends and holidays). Features:
- `days_to_next_business_day`
- `is_business_day`
- `business_day_of_month` (1 = first business day; last = last business day)

These are nonlinear functions of the raw date but well-defined.

### STL Decomposition

Seasonal-Trend decomposition using Loess:

$$
Y_t = T_t + S_t + R_t
$$

- $T_t$: trend (smooth via LOESS).
- $S_t$: seasonal (mean of deseasonalized at each period point).
- $R_t$: residual.

Iterative algorithm alternates detrending and deseasonalizing. Features $T_t$ and $S_t$ can be used as input to a downstream model.

## 5.4 Worked Examples

### Example 1: Cyclical Hour Encoding

Consider `hour = 23` and `hour = 0` (consecutive hours across midnight).

Integer encoding: difference = 23. Model treats these as maximally far apart.

Cyclical encoding:
- Hour 23: $\sin(2\pi \cdot 23/24) = \sin(5.76) \approx -0.259$, $\cos(2\pi \cdot 23/24) = \cos(5.76) \approx 0.966$. So $(−0.259, 0.966)$.
- Hour 0: $\sin(0) = 0$, $\cos(0) = 1$. So $(0, 1)$.
- Hour 12: $\sin(\pi) = 0$, $\cos(\pi) = -1$. So $(0, -1)$.

Distances:
- Hour 23 to Hour 0: $\sqrt{0.259^2 + 0.034^2} \approx 0.26$.
- Hour 12 to Hour 0: $\sqrt{0 + 4} = 2.0$.

Cyclical encoding correctly shows Hour 23 is close to Hour 0, and Hour 12 is maximally far.

### Example 2: Feature Engineering a Purchase Timestamp

Raw column: `purchase_ts` (datetime, UTC).

Engineered features (assuming US retail context):

| Feature | Formula | Rationale |
|---|---|---|
| `hour` | $t.hour$ | Time-of-day effects |
| `hour_sin`, `hour_cos` | cyclic encoding | Smooth model over hour boundary |
| `dayofweek` | $t.dayofweek$ | Weekend vs weekday |
| `dayofweek_sin`, `dayofweek_cos` | cyclic | Smooth weekly cycle |
| `month` | $t.month$ | Yearly seasonality |
| `month_sin`, `month_cos` | cyclic | Smooth yearly cycle |
| `quarter` | $t.quarter$ | Quarterly effects |
| `is_weekend` | $t.dayofweek \in \{5, 6\}$ | Bigger leisure shopping |
| `is_holiday` | look up in US calendar | Holiday spikes |
| `days_to_black_friday` | nearest-future Black Friday | Pre-BF buildup |
| `days_since_account_created` | $t - \text{account\_created}$ | Customer maturity |
| `days_since_last_purchase` | $t - \text{last\_purchase}$ | Engagement |
| `year` | $t.year$ | Trend |

Typical tabular model uses 10–15 features derived from a single timestamp. Each is engineered for a specific pattern.

### Example 3: Leakage via Time Splits

Dataset: 10,000 rows of daily sales from 2020–2024.

**Wrong:** Random 80/20 split. The model sees a mix of 2020–2024 in both sets; future data leaks into training. In production (on 2025 data), performance collapses.

**Right:** Time-ordered split. Train on 2020–2023; validate on first half of 2024; test on second half. Mirrors production — future data never seen at training.

If using cross-validation: `TimeSeriesSplit` (forward-rolling) rather than random CV.

### Example 4: Holiday Proximity Feature

A model predicting pizza delivery demand. Raw date alone shows baseline variation. Adding:
- `is_super_bowl_sunday`: +40% order volume captured.
- `days_to_super_bowl`: continuous feature, captures ramp-up.
- `is_friday_with_football`: context-specific spike.

Without these, the model just sees a random-seeming weekly spike on Super Bowl Sunday — or worse, treats it as an outlier and ignores it.

## 5.5 Relevance to ML Practice

**When to use each technique:**

- **Cyclical encoding:** Always, for hour, day-of-week, month, and day-of-year if using linear or neural models. Tree models can learn cyclic patterns but benefit too. For hour features especially, cyclical encoding is nearly free to add.
- **Time-since features:** Crucial for churn, retention, lifecycle models. Always consider these when modeling sessions or subscriptions.
- **Holiday features:** Crucial for retail, hospitality, travel. Less so for long-horizon macroeconomic forecasts.
- **Explicit year:** For capturing secular trends. In a model trained 2020–2024, `year` encodes the trend. In production 2025, it extrapolates — be cautious.

**Time splits:**

- Any time-dependent data should be split chronologically. Random splits for time series cause leakage.
- For cross-validation, use forward-rolling windows.
- For online learning, use strict temporal ordering.

**Holiday libraries:**

- Python `holidays` library covers 100+ countries.
- For finance, trading calendars (NYSE, LSE) have specific rules.
- For retail, custom event calendars (Black Friday, Prime Day) must be maintained.

**Time zones:**

- Store in UTC, convert for display/analysis.
- For location-dependent features (hour-of-day for local user), use timezone-aware datetimes and convert.
- DST transitions are a source of subtle bugs — test explicitly.

**Trade-offs:**

- More datetime features → richer signal but higher dimensionality and potential overfit. Start with basics (dayofweek, month, hour) and add as needed.
- Cyclical encoding doubles the feature count per datetime dimension but improves smoothness.
- Fourier features add more expressivity at linear cost.

**Production considerations:**

- Keep a clock service: all timestamps should align to a canonical source.
- Watch for clock drift, especially in distributed systems.
- Stale holiday lists are a silent source of error — refresh periodically.
- Feature store pipelines should handle timezone conversion consistently between training and production.

## 5.6 Common Pitfalls & Misconceptions

1. **Using raw timestamps as a feature.** Models don't natively understand "seconds since Unix epoch"; use extracted components.
2. **Integer encoding of cyclic features.** Makes hour 23 and hour 0 look maximally far apart.
3. **Ignoring time zones.** Especially in global products — "hour of day" is meaningless without timezone.
4. **Random train-test splits on time-series data.** Leaks future data; model appears great in CV, fails in production.
5. **Not updating holiday lists.** Countries add/change holidays; stale lists silently reduce performance over time.
6. **Double-counting** cyclical features. Don't include both raw `hour` and $(\sin, \cos)$ pair unless you specifically want both — increases dimensionality without benefit.
7. **Mixing UTC and local time.** Some features local, some UTC → inconsistent gradients.
8. **DST bugs.** A "day" in a DST-observing region is sometimes 23 or 25 hours. Aggregations can be off by one.
9. **Time-since features without bounds.** If `days_since_signup` can be 10,000 for some users, the model may not handle the extremes well — log-transform it.
10. **Forgetting leap years.** Feb 29 exists. Day-of-year ranges 1–366 in leap years, 1–365 otherwise.
11. **Using `is_holiday` without distance.** Holiday spikes often build over 1–3 days before; `days_to_holiday` captures this.
12. **Overfitting to specific dates.** A model that learns "sales spike on 2023-11-24" is overfitting; using `is_black_friday` lets it generalize.

## 5.7 Interview Questions

### Foundational

**Q1. Why can't you use a raw timestamp as a feature?**

**A.** Raw timestamps (e.g., Unix seconds since 1970) are interpreted as large linear numbers. A model sees "1700000000" vs "1700003600" as two values an hour apart on a linear scale — but this scale doesn't expose any of the structure that matters: day of week, time of day, seasonality, holiday proximity, or recency. Raw timestamps also drift monotonically, so using them as-is in a regression can overfit to "time has passed" without learning meaningful patterns.

**Q2. What's cyclical encoding and why do we need it?**

**A.** Cyclical encoding maps a cyclic variable $t$ with period $T$ to two features: $\sin(2\pi t/T)$ and $\cos(2\pi t/T)$. This places $t$ on a unit circle, so adjacent values on the cycle (hour 23 and hour 0) are close in the encoded 2D space. Without this, integer encoding tells the model "hour 23 is 23 units from hour 0" — the opposite of reality. Cyclical encoding lets linear models and neural networks learn smooth cyclic patterns.

**Q3. What's the difference between "time since" and "time to" features?**

**A.** "Time since" measures duration from a past event (e.g., `days_since_signup`, `days_since_last_login`). "Time to" measures duration until a future event (e.g., `days_until_subscription_ends`, `days_until_next_holiday`). Both are often predictive: "time since" captures engagement/maturity; "time to" captures anticipation/deadline effects. Many models need both. Beware of data leakage — don't use "time to" features that depend on future knowledge at training but aren't available at inference.

**Q4. Why shouldn't you use random train-test splits for time-series data?**

**A.** Random splits mix past and future in both sets. The model trains on some future data and tests on some past — meaningless for deployment, where you only have past data. Worse, it leaks future patterns into training (e.g., the model learns "holidays cause spikes" from future holidays, then tests on past holidays). Chronological splits — train on earliest, test on latest — mirror production. For CV, use `TimeSeriesSplit` (forward-rolling).

**Q5. How should holiday features be encoded?**

**A.** At minimum: `is_holiday` (binary). Better: multiple features — `is_holiday`, `days_to_next_holiday`, `days_since_last_holiday`, `holiday_type` (religious, national, shopping). For retail: specific events like `is_black_friday`, `is_christmas_week`. For finance: `is_trading_day`, `is_fomc_meeting`. Use a well-maintained library (Python `holidays`) for country calendars; supplement with custom events.

### Mathematical

**Q6. Prove that cyclical encoding $(\sin(2\pi t/T), \cos(2\pi t/T))$ preserves cyclic distance.**

**A.** Two points at $t_1, t_2$ encoded as $(s_1, c_1), (s_2, c_2)$. Squared Euclidean distance:

$d^2 = (s_1 - s_2)^2 + (c_1 - c_2)^2 = s_1^2 - 2s_1 s_2 + s_2^2 + c_1^2 - 2 c_1 c_2 + c_2^2 = 2 - 2(s_1 s_2 + c_1 c_2) = 2 - 2\cos(2\pi(t_1 - t_2)/T)$

(using trigonometric identity $\cos(A - B) = \cos A \cos B + \sin A \sin B$). This depends only on the cyclic difference $t_1 - t_2 \mod T$, not on absolute position. $d^2$ is minimum (0) when $t_1 = t_2$ (or differ by $T$) and maximum (4) when they differ by $T/2$.

**Q7. Why do we need both sin and cos, not just one?**

**A.** A single $\sin(2\pi t/T)$ gives the same value for $t$ and $T - t$ (mirror symmetry around $T/2$). So hour 3 and hour 21 would be indistinguishable. Adding $\cos$ breaks the symmetry: at $t=3$, $\cos(2\pi \cdot 3/24) \approx 0.707$; at $t=21$, $\cos(2\pi \cdot 21/24) \approx 0.707$ as well — wait, we still have issue. Actually, let me recheck. $\cos(\pi/4) = 0.707$, $\cos(7\pi/4) = 0.707$. So $\cos$ alone also has symmetry. Together: $(\sin, \cos)$ uniquely identifies $t \mod T$. At $t=3$: $(\sin(\pi/4), \cos(\pi/4)) = (0.707, 0.707)$. At $t=21$: $(\sin(7\pi/4), \cos(7\pi/4)) = (-0.707, 0.707)$. Different. The pair breaks the ambiguity.

**Q8. How do you extend cyclical encoding for multi-harmonic patterns?**

**A.** Include Fourier terms up to harmonic $K$:

$$
\{\sin(2\pi k t / T), \cos(2\pi k t / T) : k = 1, 2, \ldots, K\}
$$

giving $2K$ features. $K=1$ captures the fundamental cycle (one peak per period); $K=2$ adds a harmonic (can represent two peaks per period, e.g., morning and evening rush); higher $K$ captures finer structure. This is exactly the Fourier series basis. Overfitting risk increases with $K$.

**Q9. In an STL decomposition, what does each component represent?**

**A.** $Y_t = T_t + S_t + R_t$.
- $T_t$ (trend): smooth long-run movement, estimated by LOESS smoothing of detrended data.
- $S_t$ (seasonal): repeating pattern of fixed period (e.g., weekly); at each position in the cycle, $S$ is the average of residuals after detrending.
- $R_t$ (residual): what's left. Ideally a stationary, low-variance random process.
Decomposition is iterative: detrend → estimate seasonal → remove seasonal → re-estimate trend, until convergence.

**Q10. You have hourly data with strong bimodal daily patterns (morning and evening peaks). How would you model this without using many one-hot hour features?**

**A.** Use Fourier features with $K=2$ or $K=3$: $\{\sin(2\pi k t/24), \cos(2\pi k t/24) : k=1,2,3\}$. Six total features capture the fundamental cycle (single peak/trough) plus second and third harmonics (bimodal, trimodal patterns). A linear model over these 6 features can fit morning + evening peaks with correct shapes. Far fewer parameters than 24 hour dummies, and generalizes between hours.

### Applied

**Q11. You're building a churn model. What datetime features would you engineer?**

**A.** Core features:
- `days_since_signup`: tenure.
- `days_since_last_login`: engagement.
- `days_since_last_purchase`: transactional engagement.
- `login_frequency_last_30d`: recent activity.
- `average_session_gap_last_30d`: usage rhythm.
- `is_active_this_week`: binary engagement.
- `months_since_subscription_end` (if applicable): lapsed user.
- `dayofweek`, `hour` of recent activity (cyclical): user's typical pattern.
- `days_since_last_customer_service_contact`: indirect churn signal.

Also consider time-of-day of signup (possibly correlates with persona) and seasonality (churn rates may spike post-holiday).

**Q12. You're forecasting retail sales. Beyond obvious datetime features, what else matters?**

**A.** (1) Holidays and holiday proximity (days to Black Friday, Christmas). (2) Promotional calendar overlay (sales events, marketing campaigns). (3) Weather as a feature — not strictly datetime, but tied to date/location. (4) Prior year's same-week sales (year-over-year comparison). (5) Trailing moving averages of sales (7-day, 30-day). (6) Trend and seasonality decomposition features. (7) Event-driven features (new product launches, competitor events). The raw date captures only so much; domain events and external context carry most of the signal.

**Q13. You're deploying a model globally with timezone differences. How do you handle time features?**

**A.** (1) Store all timestamps in UTC. (2) For features tied to user local time (is_business_hour, hour_of_day), use the user's timezone (e.g., from account settings or IP geo) to convert UTC → local before feature extraction. (3) For features tied to server-side or global events, use UTC. (4) Test DST transitions in both training and production pipelines. (5) Document which features are UTC vs local. (6) For global aggregates (daily sales), decide on a canonical timezone (often UTC or business headquarters) and stick to it.

**Q14. In a credit card fraud detection model, what datetime features matter?**

**A.** (1) Hour of day (cyclical) — fraud often concentrates in overnight hours. (2) Day of week (cyclical). (3) `minutes_since_last_transaction` — velocity is a huge fraud signal. (4) `transactions_in_last_5_min`, `last_1_hour`, `last_24_hours` — rapid-fire transactions. (5) `is_new_merchant_for_user` (time-derived feature). (6) `time_of_day_deviation_from_user_norm`: does this transaction happen at an hour unusual for this user? (7) `days_since_last_address_change`: recent account changes flag fraud risk. (8) Holiday/weekend indicators (fraudsters may target when customer service is less responsive).

**Q15. Your time-series model is great on validation but bad in production on recent data. What are likely causes?**

**A.** (1) Distribution drift: recent data may have different patterns (new seasonality, changed user behavior). (2) Stale holiday list or calendar. (3) Retraining frequency too low — the model's temporal features are out of date. (4) Trend extrapolation: if the model uses `year` or `days_since_launch`, production values exceed anything seen in training. (5) Data lag: some features are backfilled with delays, causing train/production skew. Fixes: retrain frequently on rolling windows; monitor feature distributions; avoid features whose values grow unboundedly; use relative-time features where possible.

### Debugging & Failure Modes

**Q16. Your model seems to perform well overall but terribly on holidays. What's wrong?**

**A.** Likely missing holiday features. The model sees holidays as unusual versions of regular days (huge spike, atypical patterns) without any feature telling it "this is a holiday." Fix: add holiday indicators (binary), proximity features, and possibly holiday-type labels. Also consider that the training set may have too few holidays for the model to learn the pattern — in which case, larger training windows or cross-period aggregation helps.

**Q17. After adding hour-of-day as an integer feature, your linear model's RMSE got worse. What happened?**

**A.** Integer encoding imposes false linearity: the model assumes a monotonic relationship between hour and target (2am and 3am differ by 1 unit; 11pm and 1am differ by 22 units). For most real hour-of-day patterns, this is wrong. Fix: use cyclical encoding $(\sin, \cos)$ or one-hot encoding (24 columns). For tree models, integer encoding is fine — they can split at arbitrary thresholds.

**Q18. You trained on 2020–2023 data and deployed in 2024. Performance dropped 30%. What datetime issues could cause this?**

**A.** Multiple possibilities: (1) COVID-affected patterns in training data — models learned 2020-specific behaviors that don't generalize. (2) Secular trend: if `year` is a feature, 2024 is outside training range; the model extrapolates poorly. (3) Missing 2024-specific events (new holidays, schedule shifts). (4) Inflation, demographic shifts, market changes embedded in the data. Fixes: retrain on 2021–2023 (post-COVID), use relative-time features instead of absolute `year`, include event calendars, implement rolling retraining.

**Q19. A CV score of 0.95 AUC drops to 0.70 in production. You notice random CV was used on a time-ordered dataset. Explain.**

**A.** Classic time-series leakage. Random CV mixes future and past in train/validation, so the model "sees" future patterns during training. In production, the future hasn't happened yet, so the model faces a truly novel distribution. The 0.95 CV score is optimistic; 0.70 is closer to real performance. Fix: use `TimeSeriesSplit` or manual chronological splits. Retrain to get honest estimates of performance.

**Q20. Your `days_since_signup` feature has NaN values for old accounts. Why?**

**A.** Probably because `signup_date` is missing or was truncated for accounts older than a certain cutoff (common data retention). Treatments: (1) Impute with a large value (e.g., 10000 days) — represents "very old account." (2) Add an indicator `signup_date_missing`. (3) Log-transform `days_since_signup` so the imputation is less extreme. (4) Cap at a max (e.g., 5 years) so the model doesn't need to handle extreme values. Critically, fix the root cause upstream if possible.

### Probing Follow-ups

**Q21. Are tree models immune to the cyclic encoding problem?**

**A.** Mostly yes, but not entirely. Tree models can split at any threshold and can separately handle "hour ≤ 6" and "hour ≥ 22" to capture the nighttime group. But they need enough data at each leaf; with limited data, they may not learn the cycle structure efficiently. Cyclical encoding gives tree models a continuous representation that often speeds up learning and improves generalization in low-data regimes.

**Q22. How would you handle a feature like "days until product expiration" where the value is always positive and has a hard limit?**

**A.** Treat it as a right-censored duration feature. Options: (1) Raw feature with a log transform if the range is large. (2) Multiple features: linear `days_to_expiry`, `1/days_to_expiry` (urgency), binary `is_expiring_soon`. (3) Piecewise indicators (expiring in 0-7 days, 8-30, 31+). For tree models, raw or log-transformed is fine. For linear models, the piecewise or rate-based features often help.

**Q23. What's the difference between calendar time and event time for time-based features?**

**A.** Calendar time is absolute clock time (March 15, 2024). Event time is time measured in events or actions (the 50th transaction, the 3rd login). Event time features like "transactions since last churn signal" can be more predictive than calendar time for user behavior models — a week of inactivity from a power user vs a casual user means different things. Best practice: use both where appropriate, and distinguish them explicitly.

**Q24. Your model uses `year` as a feature. What's the risk, and how do you mitigate?**

**A.** Risk: `year` is monotonically increasing; production data will always have a higher `year` than training. Tree models can't predict outside the training range for a feature they learned on. Linear models will extrapolate, but extrapolation may be wildly wrong if the pattern changes. Mitigations: (1) Use `year` only as a trend component, fit separately and carefully. (2) Prefer relative features: `years_since_launch`, `days_since_product_release`. (3) Retrain frequently so the training data stays fresh. (4) If using trees, explicitly include recent years' data heavily.

**Q25. You want to encode the effect of paydays (typically end-of-month or mid-month). How?**

**A.** Options depend on granularity: (1) `day_of_month` as cyclical feature with period 30 (approximate). (2) Binary `is_near_month_end` (e.g., last 3 days of month). (3) `days_to_next_payday` (continuous), using either a fixed assumption (last day of month) or country-specific rules (US biweekly, UK end-of-month). (4) If applicable, combine with `days_to_next_15th` for mid-month pay. For retail/financial models, these often matter more than the raw date.

---

# Summary & Cross-Cutting Themes

Across all five preprocessing topics, a few principles recur:

**1. Leakage is the silent killer.** Fit all preprocessors on training data only; apply the frozen transform to validation and test. Wrap in a pipeline that respects CV folds.

**2. Missingness, rare categories, outliers, and holidays are all special values.** Consider whether they should be indicators (features), imputed away, or treated with specific logic.

**3. Assumptions matter.** MAR for imputation, MCAR for deletion, Gaussianity for z-scores, cyclic structure for sin/cos encoding, MAR for Box-Cox — violating assumptions causes silent bias.

**4. Model sensitivity varies.** Tree models are mostly scale- and shape-invariant; linear and distance-based models aren't. Preprocessing choices should be aligned with the downstream model.

**5. Train-production consistency is paramount.** Serialize the fitted preprocessors, monitor feature distributions in production, and ensure the same transform is applied in both environments. A mismatch here causes the vast majority of "why does my model degrade?" incidents.

**6. Interactions matter.** Preprocessing decisions compound: an outlier affects scaling, which affects distance-based imputation, which affects everything downstream. Think end-to-end, not feature by feature.

**7. Document and version.** Preprocessing code should be reproducible, versioned, and auditable — often more important than the model code itself for debugging production issues.

Mastery of preprocessing separates practitioners who spend months debugging production models from those who build stable, reliable ML systems. The techniques in this guide should function as your first line of defense.
