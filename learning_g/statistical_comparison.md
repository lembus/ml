# Statistical Comparison of Machine Learning Models

## 1. Motivation & Intuition

### Why This Topic Exists
In machine learning, we often ask: "Is Model A better than Model B?" 

Suppose you train a Neural Network and a Logistic Regression model on a dataset. The Neural Network achieves **94% accuracy**, and the Logistic Regression achieves **93% accuracy**. 

Is the Neural Network actually better? Or was it just lucky?

Machine learning models are subject to several sources of **randomness**:
1. **Data Selection:** The specific split of data into training and test sets is random.
2. **Initialization:** Many algorithms (like neural networks) start with random weight initializations.
3. **Optimization:** Stochastic optimization algorithms (like SGD) introduce trajectory randomness.

If you ran the experiment again with a different random seed, the Logistic Regression might score 94.5% and the Neural Network 93.5%. Statistical comparison provides a rigorous mathematical framework to determine if an observed difference in performance is "real" (systematic) or just noise (random chance).

### Real-World Analogy
Imagine two runners, Alice and Bob. They race once, and Alice wins by 0.1 seconds. Can you conclude Alice is faster? No. She might have had a slightly better starting block reaction, or a localized gust of wind might have favored her. 

To be confident, you make them race 10 times:
* If Alice wins 9 out of 10 times by a consistent margin, she is systematically faster.
* If they trade wins 50/50, their underlying speed is likely equivalent, and the 0.1-second lead in the first race was random noise.

### Connection to System Design
In production ML engineering, deploying a new model carries concrete operational costs: increased inference latency, higher compute requirements, complex pipeline integration, and engineering maintenance. You generally only want to replace an existing production model (the "champion") with a new candidate (the "challenger") if the challenger is **statistically significantly** better. Deploying a model that only *looks* better due to random dataset noise degrades system simplicity and inflates costs without delivering genuine user benefit.

---

## 2. Conceptual Foundations

### The Hypothesis Testing Framework
Statistical model comparison relies on classical **Hypothesis Testing**. This involves establishing two mutually exclusive claims:

1. **Null Hypothesis ($H_0$):** There is **no true difference** between the performance of the two models. Any observed difference in your sample scores is entirely due to random sampling noise.
2. **Alternative Hypothesis ($H_1$):** There is a **true, systematic difference** in performance between the models (e.g., Model A generalizes better than Model B).

### Key Components
* **Sample:** A finite collection of performance evaluations (e.g., accuracy or $F_1$ scores across $k$ distinct test folds).
* **Test Statistic:** A single value derived from the sample data that quantifies the magnitude of difference between the models relative to the variance in the data.
* **$p$-value:** The probability of observing a test statistic as extreme as (or more extreme than) the one calculated, **assuming the Null Hypothesis is strictly true**.
  * **Low $p$-value ($p < \alpha$, typically $0.05$):** The observed results are highly improbable under the Null Hypothesis. We reject $H_0$ and declare the difference **statistically significant**.
  * **High $p$-value ($p \ge \alpha$):** The observed variance easily accounts for the performance gap. We fail to reject $H_0$.

### Underlying Assumptions
Every statistical test rests on distributional and structural assumptions. Violating these assumptions invalidates the calculated $p$-values:

1. **Normality:** Parametric tests assume that the sample differences between model scores follow a Gaussian (Normal) distribution.
2. **Independence:** Tests assume individual evaluation scores are statistically independent. 
3. **Commensurability:** Performance metrics across different models must be evaluated on identical data partitions under identical scoring protocols.

### What Breaks When Assumptions Are Violated
* **Violating Independence (The ML Trap):** In standard $k$-fold Cross-Validation, training sets overlap significantly across folds (e.g., 90% overlap in 10-fold CV). Therefore, the test fold scores are correlated. Treating them as independent underestimates the standard error, artificially inflates the test statistic, and causes skyrocketing **Type I Error rates** (false positives).
* **Violating Normality:** If evaluation differences are heavily skewed or contain extreme outliers, parametric tests lose statistical power or yield misleading significance thresholds.

---

## 3. Mathematical Formulation

### Notation
* $k$: Number of evaluation iterations or cross-validation folds.
* $x_{i, A}$: Performance metric of Model A on fold $i$.
* $x_{i, B}$: Performance metric of Model B on fold $i$.
* $d_i = x_{i, A} - x_{i, B}$: The observed performance difference on fold $i$.
* $\bar{d}$: The sample mean of the differences across all $k$ folds.
* $S^2$: The sample variance of the differences.

---

### A. The Paired $t$-Test (Parametric)

Used when comparing two models evaluated on identical partitions of data.

#### Derivation & Formulation
1. Compute the **mean difference**:

$$
\bar{d} = \frac{1}{k} \sum_{i=1}^{k} d_i
$$

2. Compute the **sample variance of the differences**:

$$
S^2 = \frac{1}{k-1} \sum_{i=1}^{k} (d_i - \bar{d})^2
$$

3. Compute the **$t$-statistic**:

$$
t = \frac{\bar{d}}{\sqrt{\frac{S^2}{k}}}
$$

#### Intuition Behind the Terms
* **Numerator ($\bar{d}$):** The raw signal. How much better did Model A perform than Model B on average?
* **Denominator ($\sqrt{S^2/k}$):** The standard error of the mean difference (the noise). It scales variance down by the sample size $k$. If model performance fluctuates wildly across folds (high $S^2$), the denominator grows, shrinking the $t$-statistic toward zero.

#### Mapping Back to Concepts
The $t$-statistic is a direct signal-to-noise ratio. To reject the Null Hypothesis, the average performance gain ($\bar{d}$) must be substantially larger than the background evaluation noise ($\sqrt{S^2/k}$). The calculated $t$ is evaluated against Student's $t$-distribution with $v = k-1$ degrees of freedom.

---

### B. The $5 \times 2$ CV Paired $t$-Test (Cross-Validation Corrected)

To resolve the independence violation inherent in standard $k$-fold CV, Dietterich (1998) introduced the $5 \times 2$ CV test.

#### Formulation
We perform 5 separate replications of 2-fold cross-validation. In each replication $i \in \{1, \dots, 5\}$, the data is split randomly into two equal halves. Models A and B are trained on fold 1 and tested on fold 2, then reversed. 

Let $d_{i,1}$ and $d_{i,2}$ be the differences in performance on the two folds of replication $i$.
The mean difference for replication $i$ is:

$$
\bar{d}_i = \frac{d_{i,1} + d_{i,2}}{2}
$$

The variance for replication $i$ is:

$$
s_i^2 = (d_{i,1} - \bar{d}_i)^2 + (d_{i,2} - \bar{d}_i)^2
$$

The modified $t$-statistic uses the raw difference from the *very first fold* ($d_{1,1}$) in the numerator, normalized by the average variance across all 5 replications:

$$
t_{5\times2} = \frac{d_{1,1}}{\sqrt{\frac{1}{5} \sum_{i=1}^{5} s_i^2}}
$$

This statistic approximately follows a Student's $t$-distribution with $v = 5$ degrees of freedom.

---

### C. Wilcoxon Signed-Rank Test (Non-Parametric)

Used when the differences $d_i$ violate the normality assumption. It operates purely on the **rankings** of performance differences.

#### Step-by-Step Formulation
1. Calculate the raw differences $d_i = x_{i,A} - x_{i,B}$ for $i = 1, \dots, k$.
2. Remove any ties where $d_i = 0$, reducing the effective sample size to $N_r$.
3. Rank the absolute differences $|d_i|$ in ascending order. The smallest absolute difference receives rank 1, the largest receives rank $N_r$. (Ties in absolute values receive the average of their spanning ranks).
4. Let $R(\cdot)$ denote the assigned rank. Restore the original signs to compute the positive and negative rank sums:

$$
R^+ = \sum_{i: d_i > 0} R(|d_i|), \quad R^- = \sum_{i: d_i < 0} R(|d_i|)
$$

5. The test statistic $W$ is the smaller of the two rank sums:

$$
W = \min(R^+, R^-)
$$

For $N_r \ge 10$, the distribution of $W$ under $H_0$ converges to a Normal distribution with:

$$
\mu_W = \frac{N_r(N_r + 1)}{4}, \quad \sigma_W = \sqrt{\frac{N_r(N_r + 1)(2N_r + 1)}{24}}
$$

$$
Z = \frac{W - \mu_W}{\sigma_W}
$$

---

### D. Friedman Test with Nemenyi Post-Hoc (Multiple Model Comparison)

Used when simultaneously comparing $N$ distinct models across $k$ separate datasets or benchmark domains.

#### The Friedman Test
Let $r_i^j$ be the rank of model $j$ on dataset $i$ (rank 1 is best). 
The average rank of model $j$ across all datasets is:

$$
R_j = \frac{1}{k} \sum_{i=1}^{k} r_i^j
$$

Under the Null Hypothesis that all models perform equivalently, their average ranks should be equal: $R_1 = R_2 = \dots = R_N = \frac{N+1}{2}$.

The Friedman Chi-Square statistic quantifies divergence from this expectation:

$$
\chi_F^2 = \frac{12k}{N(N+1)} \sum_{j=1}^{N} \left( R_j - \frac{N+1}{2} \right)^2 = \frac{12}{k N(N+1)} \left[ \sum_{j=1}^{N} \left( \sum_{i=1}^{k} r_i^j \right)^2 \right] - 3k(N+1)
$$

Iman and Davenport (1980) showed that $\chi_F^2$ is undesirably conservative and proposed an $F$-distribution corrected statistic:

$$
F_F = \frac{(k-1)\chi_F^2}{k(N-1) - \chi_F^2}
$$

which follows an $F$-distribution with $(N-1)$ and $(k-1)(N-1)$ degrees of freedom.

#### Nemenyi Post-Hoc Test
If the Friedman test rejects $H_0$, we proceed to pairwise comparisons. Two models differ significantly if the distance between their average ranks exceeds the **Critical Difference (CD)**:

$$
CD = q_{\alpha} \sqrt{\frac{N(N+1)}{6k}}
$$

where $q_{\alpha}$ is the critical value derived from the Studentized range distribution divided by $\sqrt{2}$.

---

### E. Multiple Testing Corrections

When performing $m$ independent pairwise tests simultaneously at a significance level $\alpha$, the probability of at least one false positive (the **Family-Wise Error Rate**, FWER) inflates dramatically:

$$
\text{FWER} = 1 - (1 - \alpha)^m
$$

#### Bonferroni Correction (FWER Control)
To maintain a global family-wise error rate of $\alpha$, adjust the significance threshold for each individual test to:

$$
\alpha_{\text{per-test}} = \frac{\alpha}{m}
$$

*Trade-off:* Highly conservative; significantly increases Type II errors (false negatives).

#### Benjamini-Hochberg Procedure (False Discovery Rate Control)
Controls the expected proportion of false positives among all rejected hypotheses.
1. Sort the $m$ unadjusted $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. Find the largest rank $j$ such that:

$$
p_{(j)} \le \frac{j}{m} \alpha
$$

3. Reject all null hypotheses associated with $p_{(1)}, \dots, p_{(j)}$.

---

## 4. Worked Examples

### End-to-End Example: Paired $t$-Test vs. Wilcoxon Signed-Rank

We evaluate Support Vector Machine (SVM) and Random Forest (RF) classifiers across $k=5$ benchmark datasets using classification accuracy.

| Dataset ($i$) | SVM Accuracy ($x_{i,A}$) | RF Accuracy ($x_{i,B}$) | Difference ($d_i$) | Absolute Diff ($|d_i|$) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 0.81 | 0.82 | -0.01 | 0.01 |
| 2 | 0.79 | 0.83 | -0.04 | 0.04 |
| 3 | 0.88 | 0.86 | +0.02 | 0.02 |
| 4 | 0.74 | 0.82 | -0.08 | 0.08 |
| 5 | 0.85 | 0.86 | -0.01 | 0.01 |

We test at significance level $\alpha = 0.05$.

---

### Method 1: The Paired $t$-Test

#### Step 1: Compute Mean Difference ($\bar{d}$)

$$
\bar{d} = \frac{-0.01 - 0.04 + 0.02 - 0.08 - 0.01}{5} = \frac{-0.12}{5} = -0.024
$$

On average, SVM scores 2.4% lower than Random Forest.

#### Step 2: Compute Sample Variance ($S^2$)
Calculate squared deviations from the mean ($\bar{d} = -0.024$):
1. $(-0.01 - (-0.024))^2 = (0.014)^2 = 0.000196$
2. $(-0.04 - (-0.024))^2 = (-0.016)^2 = 0.000256$
3. $(0.02 - (-0.024))^2 = (0.044)^2 = 0.001936$
4. $(-0.08 - (-0.024))^2 = (-0.056)^2 = 0.003136$
5. $(-0.01 - (-0.024))^2 = (0.014)^2 = 0.000196$

Sum of squared deviations $= 0.00572$

$$
S^2 = \frac{0.00572}{5 - 1} = 0.00143
$$

#### Step 3: Compute $t$-Statistic

$$
t = \frac{-0.024}{\sqrt{\frac{0.00143}{5}}} = \frac{-0.024}{\sqrt{0.000286}} \approx \frac{-0.024}{0.01691} \approx -1.419
$$

#### Step 4: Evaluate Statistical Significance
* Degrees of freedom $v = 5 - 1 = 4$.
* For a two-tailed test at $\alpha = 0.05$, the critical values from the $t$-distribution table are $\pm 2.776$.

Since $-2.776 < -1.419 < 2.776$, we **fail to reject the Null Hypothesis**. 

*Teaching insight:* Even though Random Forest won on 4 out of 5 datasets, Dataset 4 introduced massive variance ($d_4 = -0.08$). The parametric test penalizes this high variance, concluding the sample size is too small to rule out random chance.

---

### Method 2: Wilcoxon Signed-Rank Test

#### Step 1: Rank the Absolute Differences
Our raw differences are: $-0.01, -0.04, +0.02, -0.08, -0.01$.
Absolute differences: $0.01, 0.04, 0.02, 0.08, 0.01$.

Sorting absolute differences in ascending order:
* $0.01$ (tied)
* $0.01$ (tied)
* $0.02$
* $0.04$
* $0.08$

Resolving tied ranks for the two $0.01$ values (occupying ranks 1 and 2):

$$
\text{Assigned Rank} = \frac{1 + 2}{2} = 1.5
$$

Table of ranked values:

| Dataset | Raw $d_i$ | $|d_i|$ | Rank | Sign |
| :--- | :--- | :--- | :--- | :--- |
| 1 | -0.01 | 0.01 | 1.5 | $-$ |
| 5 | -0.01 | 0.01 | 1.5 | $-$ |
| 3 | +0.02 | 0.02 | 3.0 | $+$ |
| 2 | -0.04 | 0.04 | 4.0 | $-$ |
| 4 | -0.08 | 0.08 | 5.0 | $-$ |

#### Step 2: Compute Signed Rank Sums

$$
R^+ = 3.0
$$

$$
R^- = 1.5 + 1.5 + 4.0 + 5.0 = 12.0
$$

#### Step 3: Compute Test Statistic $W$

$$
W = \min(R^+, R^-) = \min(3.0, 12.0) = 3.0
$$

#### Step 4: Evaluate Significance
For $N_r = 5$, the exact critical value for a two-tailed Wilcoxon test at $\alpha = 0.05$ is $W_{\text{crit}} = 0$. (To be significant, $W$ must be $\le 0$).

Since $3.0 > 0$, we **fail to reject the Null Hypothesis**. Both tests confirm that the performance gap is not statistically significant.

---

## 5. Relevance to Machine Learning Practice

### ML Lifecycle Integration
* **Model Training & Selection:** Used during hyperparameter optimization to verify if adding structural complexity (e.g., an extra attention layer) yields systematic gain over a simpler baseline.
* **Offline Evaluation:** Validating candidate models across out-of-time test sets before promoting them to staging registries.
* **Production Monitoring & A/B Testing:** Comparing live challenger models against production champions. In live traffic, statistical comparison uses two-sample tests (like the Welch's $t$-test or Mann-Whitney $U$ test) on streaming business metrics (conversion rates, click-through rates).

### Decision Matrix: Test Selection Guide

| Scenario | Recommended Protocol | Primary Reason |
| :--- | :--- | :--- |
| Comparing 2 models on 1 dataset | $5 \times 2$ CV Paired $t$-test | Controls Type I errors caused by overlapping CV training data. |
| Comparing 2 models on $N > 10$ datasets | Wilcoxon Signed-Rank Test | Robust against non-normal performance distributions across domains. |
| Comparing $M > 2$ models on $N > 10$ datasets | Friedman Test + Nemenyi Post-Hoc | Controls family-wise error rate across multiple comparisons. |
| Comparing models on live streaming A/B traffic | Two-Sample Welch's $t$-test / Mann-Whitney $U$ | Handles independent streams with unequal sample sizes and variances. |
| Business requires direct risk/probability metrics | Bayesian Correlated $t$-test | Provides exact posterior probabilities of superiority via ROPE. |

### Trade-Off Analysis
* **Statistical Power vs. Computational Cost:** The $5 \times 2$ CV test requires training 10 separate models. For massive LLMs, this compute cost is prohibitive. Practitioners often fall back to single train/test splits (e.g., McNemar's test for classification predictions), sacrificing statistical power to save compute.
* **Interpretability vs. Rigor:** Frequentist $p$-values are notoriously difficult for non-technical stakeholders to interpret. Bayesian model comparison solves this by mapping results directly to business risk: *"There is a 94.2% probability that Model B improves accuracy by at least 1%."*
* **Robustness vs. Sensitivity:** Non-parametric tests (Wilcoxon) discard raw metric magnitudes in favor of ranks. This makes them exceptionally robust to data pipeline anomalies and outliers, but slightly less sensitive (lower power) than parametric tests when the underlying performance differences are truly Gaussian.

---

## 6. Common Pitfalls & Misconceptions

### 1. The $k$-fold CV Independence Fallacy
* **The Mistake:** Running 10-fold cross-validation, taking the 10 accuracy scores for Model A and Model B, and running a standard paired $t$-test.
* **Why it Occurs:** Standard statistical software readily accepts two arrays of length 10 without checking data provenance.
* **How to Avoid:** Recognize that CV validation folds share training data. Use the $5 \times 2$ CV paired $t$-test or evaluate models across strictly independent benchmark datasets.

### 2. Confusing Statistical Significance with Practical Significance (Effect Size)
* **The Mistake:** Promoting a model to production because a test yielded $p = 0.0001$, even though the absolute accuracy gain is $0.005\%$.
* **Why it Occurs:** As sample size $N$ approaches infinity, the standard error $\frac{S}{\sqrt{N}}$ approaches zero. Consequently, even infinitesimal, practically meaningless differences generate astronomical $t$-statistics and vanishingly small $p$-values.
* **How to Avoid:** Always pair hypothesis tests with **Effect Size** estimation (e.g., Cohen's $d$). Define a **Region of Practical Equivalence (ROPE)** business threshold prior to experimentation.

### 3. $p$-Hacking via Multiple Comparisons
* **The Mistake:** Benchmarking a proposed architecture against 20 baseline models at $\alpha = 0.05$ and claiming superiority over the single baseline where $p < 0.05$.
* **Why it Occurs:** Ignoring the cumulative math of probability. With 20 independent comparisons, the probability of observing at least one false positive purely by chance is $1 - (0.95)^{20} \approx 64.15\%$.
* **How to Avoid:** Apply omnibus tests (Friedman) prior to pairwise investigations, or enforce strict false discovery rate controls (Benjamini-Hochberg).

---

## Interview Preparation Section

### Foundational Questions

#### Q1: Exactly what does a $p$-value of 0.03 mean when comparing the AUC-ROC of two classifiers?
**Answer:** A $p$-value of 0.03 means that **if the Null Hypothesis were strictly true** (meaning both classifiers possess identical underlying performance capabilities and any differences are random noise), there is a 3% probability of observing an AUC-ROC difference as large as, or larger than, the one calculated from your test sample. It does *not* mean there is a 97% chance Model A is better, nor does it mean there is a 3% chance the null hypothesis is true.

#### Q2: What are Type I and Type II errors in the context of automated model deployment pipelines?
**Answer:**
* **Type I Error (False Positive):** Rejecting $H_0$ when it is true. In production, this means deploying a challenger model that is actually no better than the champion model. This incurs migration engineering costs, risk of unseen bugs, and pipeline complexity for zero actual gain.
* **Type II Error (False Negative):** Failing to reject $H_0$ when $H_1$ is true. This means discarding a challenger model that genuinely possesses superior generalization capabilities, causing the business to miss out on performance improvements.

#### Q3: Why is statistical variance across evaluation splits critical? Why can't we just select the model with the highest average validation score?
**Answer:** Point estimates (averages) obscure stability. Suppose Model A scores $[0.89, 0.90, 0.91]$ across 3 folds ($\text{mean} = 0.90$). Model B scores $[0.70, 0.99, 1.00]$ ($\text{mean} = 0.896$). If you rely purely on averages or single splits, you might assume they are equivalent. However, Model B exhibits massive performance volatility ($S^2$). Statistical variance quantifies uncertainty; tests like the $t$-test penalize high variance to prevent deploying erratic models that might fail catastrophically on unseen production distributions.

---

### Mathematical Questions

#### Q4: Prove mathematically why overlapping training sets in $k$-fold cross-validation bias the sample variance estimate downward.
**Answer:**
Let $d_i$ be the performance difference on fold $i$. The standard variance estimator assumes $\text{Cov}(d_i, d_j) = 0$ for $i \neq j$. However, in $k$-fold CV, fold $i$ and fold $j$ share $\frac{k-2}{k-1}$ proportion of their training data. Therefore, the differences are positively correlated: $\text{Cov}(d_i, d_j) = \rho \sigma^2$, where $\rho > 0$.

The variance of the sample mean $\bar{d}$ is:

$$
\text{Var}(\bar{d}) = \text{Var}\left( \frac{1}{k} \sum_{i=1}^{k} d_i \right) = \frac{1}{k^2} \left[ \sum_{i=1}^{k} \text{Var}(d_i) + \sum_{i \neq j} \text{Cov}(d_i, d_j) \right]
$$

$$
\text{Var}(\bar{d}) = \frac{1}{k^2} \left[ k\sigma^2 + k(k-1)\rho\sigma^2 \right] = \frac{\sigma^2}{k} \left[ 1 + (k-1)\rho \right]
$$

The standard naive estimator calculates $\frac{S^2}{k}$, completely omitting the positive term $\frac{\sigma^2}{k}(k-1)\rho$. Because the true variance is substantially larger than the estimated variance, the denominator of the $t$-statistic is artificially shrunk, inflating the $t$-value and causing excessive false positive rejections of $H_0$.

#### Q5: Derive the statistical power relationship. How does an increase in variance ($\sigma^2$) impact the minimum required sample size ($N$) to detect an effect size $\delta$?
**Answer:**
Statistical power ($1 - \beta$) is the probability of correctly rejecting $H_0$ given a true difference $\delta$. For a two-sample one-tailed $t$-test, the test statistic under $H_1$ follows a non-central $t$-distribution centered at:

$$
t_{\text{expected}} = \frac{\delta}{\sqrt{\frac{2\sigma^2}{N}}}
$$

To achieve significance at threshold $\alpha$ with power $1-\beta$, this expected value must satisfy:

$$
\frac{\delta}{\sqrt{\frac{2\sigma^2}{N}}} \approx Z_{1-\alpha} + Z_{1-\beta}
$$

Solving explicitly for sample size $N$:

$$
\sqrt{N} = \frac{(Z_{1-\alpha} + Z_{1-\beta})\sqrt{2\sigma^2}}{\delta}
$$

$$
N = \frac{2\sigma^2 (Z_{1-\alpha} + Z_{1-\beta})^2}{\delta^2}
$$

This demonstrates that required sample size scales **linearly with variance ($\sigma^2$)** and **inversely with the square of the effect size ($\delta^2$)**. If variance doubles, you need twice as many benchmark runs to detect the same performance improvement.

#### Q6: Explain the operational mathematical difference between Family-Wise Error Rate (FWER) and False Discovery Rate (FDR).
**Answer:**
Let $V$ be the number of false positives (Type I errors) and $R$ be the total number of rejected null hypotheses.

* **FWER** controls the absolute probability of making *at least one* Type I error across all $m$ tests:

$$
\text{FWER} = P(V \ge 1)
$$

Controlling FWER (e.g., via Bonferroni $\frac{\alpha}{m}$) guards against any false discoveries, but aggressively saps statistical power as $m$ grows.

* **FDR** controls the *expected proportion* of false discoveries among all rejected hypotheses:

$$
\text{FDR} = E\left[ \frac{V}{R} \right] \quad (\text{defining } \frac{0}{0} = 0)
$$

Controlling FDR (via Benjamini-Hochberg) allows some false positives to slip through, provided the vast majority of your discoveries ($R$) are genuine. In large-scale ML experimentation (e.g., testing 1,000 feature configurations), FDR control is preferred to maintain discovery velocity.

---

### Applied & Design Questions

#### Q7: You are leading ML applied science for a high-frequency trading platform. A researcher proposes a new latency-heavy deep learning model that beats the linear regression baseline by 0.4% in offline accuracy ($p = 0.01$). How do you decide whether to ship it?
**Answer:**
I execute a strict multi-tiered design decision protocol:
1. **Validate Statistical Assumptions:** Check if offline validation splits suffered from temporal data leakage. Financial time series exhibit extreme autocorrelation; standard k-fold CV is invalid. I re-verify the $p$-value using **Purged and Embargoed Cross-Validation**.
2. **Evaluate Practical Effect Size:** Map the 0.4% accuracy boost directly to expected monetary value ($E[\text{Alpha}]$). 
3. **Analyze System Infrastructure Costs:** Calculate the cost of latency. If the deep learning model adds 15 milliseconds of inference latency, does execution slippage in live markets erase the 0.4% prediction edge? 
4. **Conclusion:** If financial modeling shows net expected profit is negative due to compute cost and latency slippage, I reject deployment despite $p = 0.01$. Statistical significance is a necessary condition for deployment, not a sufficient one.

#### Q8: You design a continuous training system that evaluates 50 newly tuned LLM prompt templates nightly against a production baseline. You flag templates as "superior" if $p < 0.05$. After one month, production quality noticeably degrades. What failed?
**Answer:**
The system succumbed to the **Multiple Comparisons Problem** compounded by automated selection bias.
* **Mechanism:** Evaluating 50 templates nightly at $\alpha = 0.05$ yields an expected $50 \times 0.05 = 2.5$ false positive promotions *every single night* purely by stochastic chance. Over 30 days, approximately 75 inferior or equivalent templates were falsely flagged as superior.
* **Remediation:** 1. Implement **Benjamini-Hochberg FDR control** on nightly evaluation batches.
  2. Require multi-stage promotion: candidates passing offline FDR control must automatically enter a **Shadow Mode** or **Canary Deployment** on live traffic.
  3. Switch to **Sequential Probability Ratio Tests (SPRT)** to continuously monitor streaming live metrics without inflating false positive rates across continuous time horizons.

---

### Debugging & Failure Modes

#### Q9: An engineer presents an experiment where Model A beat Model B on 10 out of 10 CV folds. However, a Paired $t$-test yields $p = 0.18$. How is it mathematically possible to win 100% of splits and fail to achieve significance?
**Answer:**
This occurs when the magnitude of the fold differences ($d_i$) exhibits extreme variance relative to the mean difference. 

* **Example Data:** Suppose 9 folds yield tiny positive improvements $d \in \{+0.001, \dots, +0.002\}$, but fold 10 yields a massive positive improvement $d_{10} = +0.500$.
* **Mathematical Failure:** The sample mean $\bar{d}$ is pulled up to $\approx 0.051$. However, the squared deviation of fold 10 from the mean ($(0.500 - 0.051)^2$) explodes the sample variance $S^2$. Because the $t$-statistic divides by $\sqrt{S^2/k}$, the gigantic standard error shrinks $t$ drastically.
* **Diagnostic Solution:** The $t$-test assumes normality. The presence of the massive fold 10 outlier violates Gaussian assumptions. I would immediately switch to the **Wilcoxon Signed-Rank Test**. Because Wilcoxon evaluates *ranks* rather than raw distances, winning 10 out of 10 positive ranks yields the maximum possible rank sum ($R^+ = 55, R^- = 0$), resulting in an exact non-parametric $p$-value of $\approx 0.002$, correctly identifying statistical superiority.

#### Q10: During an A/B test comparing recommendation algorithms, you compute $p$-values continuously every hour. At hour 6, $p$ drops to 0.04. You stop the test and declare victory. Why is this methodology scientifically invalid?
**Answer:**
This is a classic failure mode known as **Continuous Peeking (or $p$-value Peeking)**.
* **Why it Fails:** Standard $p$-values assume a fixed sample size $N$ determined *before* experimentation begins. The trajectories of test statistics under random sampling behave like random walks. If you monitor a random walk continuously, the probability that it stochastically crosses the $\alpha = 0.05$ boundary at *some point* during its trajectory approaches 100%, even if the true underlying difference is strictly zero.
* **Correction:** 1. Fix the sample size in advance using power calculations and forbid peeking.
  2. If continuous monitoring is an operational business necessity (e.g., to stop disastrous models quickly), use **Always-Valid $p$-values** or apply **Group Sequential Methods with Pocock / O'Brien-Fleming alpha-spending functions**, which dynamically adjust the required significance boundary higher at early peeking intervals to preserve the global Type I error rate.