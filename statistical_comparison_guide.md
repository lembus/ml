# Statistical Comparison for Machine Learning Models

## A Comprehensive Reference Guide

---

# 1. Motivation & Intuition

## Why Statistical Comparison Exists

Suppose you train two classifiers on the same dataset. Model A achieves 91.3% accuracy; Model B achieves 92.1% accuracy. Is Model B genuinely better, or did it simply get lucky on the particular data split you used? If you re-ran the experiment with a different random seed, different train/test partition, or different subset of the data, would Model B still win?

This is the core problem statistical comparison solves: **distinguishing real performance differences from noise**. Without formal statistical tests, practitioners routinely deploy models based on illusory improvements—differences that arose purely from randomness in data splitting, initialization, or sampling.

### A Concrete Example

Imagine you are a data scientist at an e-commerce company. Your current production model (Model A) for click-through-rate prediction uses logistic regression. A colleague proposes a gradient-boosted tree (Model B). You run both on a held-out test set:

- Model A: AUC = 0.823
- Model B: AUC = 0.831

Is the 0.008 difference meaningful? Consider what happens if you randomly reshuffle which examples land in train vs. test:

| Split | Model A AUC | Model B AUC | Difference |
|-------|------------|------------|------------|
| 1     | 0.823      | 0.831      | +0.008     |
| 2     | 0.829      | 0.825      | −0.004     |
| 3     | 0.818      | 0.834      | +0.016     |
| 4     | 0.826      | 0.827      | +0.001     |
| 5     | 0.821      | 0.830      | +0.009     |

The differences fluctuate. Sometimes Model A actually wins. A statistical test takes these fluctuations into account and tells you whether the average difference is large enough, relative to the variability, to be considered real.

### Why This Matters in Practice

**Deployment decisions**: Switching a production model carries engineering cost, potential regressions in edge cases, and operational risk. You need confidence that the new model is genuinely superior.

**Research claims**: When a paper claims "our method outperforms the baseline," reviewers and readers need to know whether the claim is statistically supported.

**Resource allocation**: Training and evaluating models consumes compute. If the difference between two architectures is not statistically significant, the engineering effort to productionize the more complex one may be wasted.

**Regulatory and business requirements**: In domains like healthcare and finance, model changes may require documented evidence that the new model is non-inferior or superior.

### The Fundamental Framework

All statistical comparison methods follow the same logical structure:

1. **Null hypothesis ($H_0$)**: The two models perform equally well (any observed difference is due to chance).
2. **Alternative hypothesis ($H_1$)**: The models have genuinely different performance levels.
3. **Test statistic**: A number computed from the data that measures how far the observed results deviate from what $H_0$ would predict.
4. **p-value**: The probability of observing a test statistic at least as extreme as the one computed, assuming $H_0$ is true. Small p-values suggest $H_0$ is unlikely.
5. **Decision**: If the p-value falls below a pre-specified threshold $\alpha$ (commonly 0.05), reject $H_0$ and conclude the difference is statistically significant.

The challenge in machine learning is that the "data points" used in these tests (performance scores from different folds or splits) are often **not independent** because the training sets overlap. This violates assumptions of classical tests and requires specialized corrections, which is the central technical theme of this guide.

---

# 2. Conceptual Foundations

## 2.1 Paired Tests

### Why Pairing Matters

When comparing two models, we can compare them on the same data or on different data. Pairing means evaluating both models on exactly the same test instances, then examining the **differences** in their predictions, instance by instance or fold by fold.

Why is this powerful? Consider two datasets with very different difficulty levels. If Model A gets dataset 1 (easy) and Model B gets dataset 2 (hard), you cannot fairly compare their raw scores. But if both models are evaluated on the same examples, the difficulty is "controlled for." Any systematic difference in performance is attributable to the models themselves, not to data difficulty.

Formally, pairing **reduces variance** in the comparison. The variance of the difference $d_i = \text{score}_A^{(i)} - \text{score}_B^{(i)}$ is:

$$\text{Var}(d) = \text{Var}(\text{score}_A) + \text{Var}(\text{score}_B) - 2 \cdot \text{Cov}(\text{score}_A, \text{score}_B)$$

When both models are evaluated on the same data, the covariance term is positive (both models tend to do well on easy examples and poorly on hard ones), which **reduces** the variance of $d$. Lower variance means the test has more power to detect real differences.

### Paired t-Test

**What it is**: A parametric test that checks whether the mean of paired differences is significantly different from zero.

**Setup**: You have $n$ paired performance measurements. For example, $n$ cross-validation folds, and for each fold $i$, you record the performance of Model A ($x_i^A$) and Model B ($x_i^B$). Define:

$$d_i = x_i^A - x_i^B$$

The test examines whether $\bar{d}$ (the sample mean of the differences) is significantly different from zero.

**Assumptions**:
1. The differences $d_i$ are drawn from a normal distribution (or $n$ is large enough for the Central Limit Theorem to apply).
2. The differences are independent of one another.
3. The differences are measured on a continuous scale.

**What breaks when assumptions are violated**:
- **Non-normality with small $n$**: The p-value becomes unreliable. With $n < 15$ and a skewed distribution of differences, the t-test may be anti-conservative (too many false positives) or conservative (too few true positives), depending on the direction of skew.
- **Non-independence**: This is the critical problem in ML. In $k$-fold cross-validation, the training sets of different folds overlap substantially (in 10-fold CV, any two folds share about 80% of their training data). This overlap means the performance estimates are positively correlated, which causes the paired t-test to **underestimate** the true variance of $\bar{d}$, producing p-values that are too small (inflated Type I error). This is why specialized corrections like the 5×2 cv test exist.

### Wilcoxon Signed-Rank Test

**What it is**: A non-parametric alternative to the paired t-test. Instead of assuming normally distributed differences, it only requires that the distribution of differences is symmetric around its median.

**How it works, step by step**:
1. Compute differences $d_i = x_i^A - x_i^B$.
2. Remove any $d_i = 0$ (ties with zero difference). Let $n_r$ be the number of remaining differences.
3. Rank the absolute values $|d_1|, |d_2|, \ldots, |d_{n_r}|$ from smallest to largest. Assign average ranks to ties.
4. Attach the original sign of $d_i$ to each rank: if $d_i > 0$, the rank is positive; if $d_i < 0$, the rank is negative.
5. Compute the test statistic: $W^+ = \sum_{\{i : d_i > 0\}} R_i$ (sum of positive ranks) or $W^- = \sum_{\{i : d_i < 0\}} R_i$ (sum of negative ranks). The test statistic is $W = \min(W^+, W^-)$.
6. Compare $W$ to critical values from the Wilcoxon distribution (or use a normal approximation for large $n_r$).

**Why ranks instead of raw values**: Ranks are robust to outliers. If one fold produces a wildly different performance score (perhaps due to an anomalous data split), the raw value of that outlier has a bounded effect on the rank-based test (at most it occupies the highest rank), whereas it could dominate the mean in a t-test.

**Assumptions**:
1. The differences are independent.
2. The distribution of differences is symmetric (not necessarily normal, but symmetric around the median).
3. The measurement scale is at least ordinal.

**What breaks when assumptions are violated**:
- **Asymmetry**: If the distribution of differences is highly skewed, the test may lose validity. However, it is still more robust than the t-test in this scenario.
- **Too few observations**: With $n_r < 6$, the test has very little power, and exact p-values become coarse.

---

## 2.2 Multiple Model Comparison

### The Problem with Pairwise Tests

If you have $k = 5$ models and compare all pairs using the paired t-test, you perform $\binom{5}{2} = 10$ tests. Even if all models are truly equivalent, each test has a 5% chance of a false positive. The probability of **at least one** false positive across all 10 tests is:

$$1 - (1 - 0.05)^{10} = 1 - 0.95^{10} \approx 0.40$$

A 40% chance of a spurious "significant" result is unacceptable. You need a procedure that first tests whether **any** difference exists among all models (an omnibus test), and only then performs controlled pairwise comparisons.

### Friedman Test

**What it is**: A non-parametric omnibus test for comparing $k$ models across $n$ datasets (or folds). It is the non-parametric equivalent of repeated-measures ANOVA.

**How it works**:
1. For each dataset (or fold) $j = 1, \ldots, n$, rank the $k$ models from best (rank 1) to worst (rank $k$). Ties receive average ranks.
2. For each model $i$, compute the average rank across all datasets: $\bar{R}_i = \frac{1}{n} \sum_{j=1}^{n} r_i^{(j)}$.
3. Compute the Friedman statistic:

$$\chi_F^2 = \frac{12n}{k(k+1)} \left[ \sum_{i=1}^{k} \bar{R}_i^2 - \frac{k(k+1)^2}{4} \right]$$

4. Under $H_0$ (all models are equivalent), $\chi_F^2$ follows approximately a $\chi^2$ distribution with $k - 1$ degrees of freedom (for large $n$).

**Intuition**: If all models are equally good, their average ranks should all be close to $(k+1)/2$ (the average of ranks 1 through $k$). The Friedman statistic measures how much the average ranks deviate from this expected value.

**Assumptions**:
1. The $n$ "blocks" (datasets or folds) are independent.
2. Within each block, the observations are on at least an ordinal scale.
3. No assumptions about the distribution of the underlying performance scores.

### Nemenyi Post-Hoc Test

**What it is**: After the Friedman test rejects $H_0$ (indicating that at least one model differs), the Nemenyi test identifies **which** pairs of models are significantly different.

**How it works**: Two models $i$ and $j$ are significantly different at significance level $\alpha$ if:

$$|\bar{R}_i - \bar{R}_j| > \text{CD}$$

where $\text{CD}$ (Critical Difference) is:

$$\text{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6n}}$$

Here, $q_\alpha$ is the critical value from the Studentized range distribution divided by $\sqrt{2}$, with parameters $k$ (number of models) and $\infty$ degrees of freedom.

**Visualization**: Results are often displayed using a **Critical Difference diagram**, where models are placed on a horizontal axis according to their average rank, and models connected by a horizontal bar are **not** significantly different from each other.

---

## 2.3 Cross-Validation Corrected Tests

### The Overlapping Training Sets Problem

Standard $k$-fold cross-validation creates $k$ pairs of (training set, test set). For 10-fold CV:
- Each training set contains 90% of the data.
- Any two training sets share approximately 80% of their data.

This overlap means that the performance estimates from different folds are **not independent**. A model that overfits to a particular subset of training examples will show correlated errors across folds that share those examples. The paired t-test, which assumes independence, will underestimate the variance and produce too many false positives.

### The 5×2 Cross-Validation Paired t-Test

**Proposed by Dietterich (1998)** as a solution to the inflated Type I error problem.

**Procedure**:
1. Repeat the following 5 times (iterations $i = 1, \ldots, 5$):
   a. Randomly partition the data into two equal halves: $S_1^{(i)}$ and $S_2^{(i)}$.
   b. Train Model A and Model B on $S_1^{(i)}$, test on $S_2^{(i)}$. Record performance differences $p_1^{(i)} = \text{perf}_A - \text{perf}_B$.
   c. Train Model A and Model B on $S_2^{(i)}$, test on $S_1^{(i)}$. Record performance differences $p_2^{(i)} = \text{perf}_A - \text{perf}_B$.
2. Compute the variance estimate for each of the 5 iterations:

$$s_i^2 = \left(p_1^{(i)} - \bar{p}^{(i)}\right)^2 + \left(p_2^{(i)} - \bar{p}^{(i)}\right)^2$$

where $\bar{p}^{(i)} = (p_1^{(i)} + p_2^{(i)}) / 2$.

3. Compute the test statistic:

$$t = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5} \sum_{i=1}^{5} s_i^2}}$$

4. Compare $t$ to a $t$-distribution with 5 degrees of freedom.

**Why this design**:
- Using only 2-fold splits means each training set is exactly half the data, so the two halves are **non-overlapping**, eliminating the training-set overlap problem within each iteration.
- Using 5 replications provides enough variance estimates while keeping degrees of freedom manageable.
- The test uses only $p_1^{(1)}$ in the numerator (not the average across all replications) because using the grand mean would introduce a correlation with the variance estimate, distorting the $t$-distribution approximation.

### Repeated Train-Test Splits (Random Subsampling)

An alternative to $k$-fold CV: randomly split the data into train/test $r$ times, each time with a fresh random partition.

**Advantages**: You can choose any train-test ratio (not constrained to $1/k$ for the test set). You can increase $r$ to get more precise estimates.

**Disadvantages**: The training sets across different splits overlap even more than in $k$-fold CV (since they are random), so naive paired tests are even more problematic. A correction factor (similar to the one introduced by Nadeau and Bengio) must be applied:

$$\hat{\text{Var}}_{\text{corrected}} = \left(\frac{1}{r} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \hat{\sigma}^2_d$$

where $\hat{\sigma}^2_d$ is the sample variance of the $r$ differences, $n_{\text{test}}$ is the test set size, and $n_{\text{train}}$ is the training set size. The term $n_{\text{test}} / n_{\text{train}}$ accounts for the correlation introduced by overlapping training sets.

---

## 2.4 Bayesian Comparison

### Why Go Bayesian?

Frequentist tests produce a p-value: the probability of observing data at least as extreme as what you got, **given that the null hypothesis is true**. This is not what most practitioners actually want. They want: "Given the data I observed, how likely is it that Model B is better than Model A?" This is a **posterior probability**, which requires Bayesian reasoning.

Additionally, frequentist tests force a binary reject/don't-reject decision. But in practice, you often want to say "the evidence mildly favors Model B" or "the evidence is inconclusive." Bayesian methods naturally express this uncertainty.

### Bayesian Correlated t-Test

**What it is**: A Bayesian reformulation of the corrected paired t-test that accounts for the correlation between cross-validation folds.

**Setup**: You have $n$ paired differences $d_1, \ldots, d_n$ from cross-validation. The model is:

$$d_i \sim \mathcal{N}(\mu, \sigma^2)$$

where $\mu$ represents the true performance difference and $\sigma^2$ represents the variability across folds.

**Key innovation**: The test models the correlation $\rho$ between folds due to overlapping training data. The effective sample size becomes:

$$n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}$$

A common choice (following Corani et al., 2017) is $\rho = 1/k$ for $k$-fold CV (each fold's test set is $1/k$ of the data).

**Posterior computation**: With a conjugate prior (typically a vague Normal-Inverse-Gamma prior on $(\mu, \sigma^2)$), the posterior of $\mu$ given the data is a Student-$t$ distribution. You can then compute:

$$P(\mu > 0 \mid \text{data}) \quad \text{(probability that Model A is better)}$$
$$P(\mu < 0 \mid \text{data}) \quad \text{(probability that Model B is better)}$$
$$P(|\mu| < \delta \mid \text{data}) \quad \text{(probability that models are practically equivalent)}$$

where $\delta$ is a **region of practical equivalence (ROPE)**—a threshold below which the performance difference is considered negligible (e.g., $\delta = 0.01$ for 1% accuracy difference).

### Bayes Factors

**What they are**: A ratio of how well two hypotheses explain the observed data, without committing to either one.

$$\text{BF}_{10} = \frac{P(\text{data} \mid H_1)}{P(\text{data} \mid H_0)}$$

- $\text{BF}_{10} > 1$: Data favors $H_1$ (models differ).
- $\text{BF}_{10} < 1$: Data favors $H_0$ (models are equivalent).
- $\text{BF}_{10} = 1$: Data is equally consistent with both hypotheses.

**Interpretation scale** (Jeffreys):

| Bayes Factor | Interpretation |
|-------------|---------------|
| 1–3         | Anecdotal evidence |
| 3–10        | Moderate evidence |
| 10–30       | Strong evidence |
| 30–100      | Very strong evidence |
| > 100       | Decisive evidence |

**Computation for model comparison**: For the simple case of testing $H_0: \mu = 0$ vs. $H_1: \mu \neq 0$:

$$\text{BF}_{10} = \frac{\int P(\text{data} \mid \mu, \sigma^2) \pi(\mu, \sigma^2) \, d\mu \, d\sigma^2}{P(\text{data} \mid \mu = 0, \hat{\sigma}^2)}$$

The numerator integrates the likelihood over the prior distribution of the parameters under $H_1$. The denominator evaluates the likelihood at $\mu = 0$. The ratio captures how much the data shifts the evidence toward or away from the null.

**Important subtlety**: The Bayes Factor is sensitive to the **prior** under $H_1$. A very vague prior (e.g., $\mu \sim \mathcal{N}(0, 1000)$) spreads the prior mass thinly over a vast range, most of which is far from the observed data, making $H_1$ look bad (the "Lindley paradox"). This is why the JZS (Jeffreys-Zellner-Siow) prior is commonly used: it places a Cauchy prior on the standardized effect size, which is broad but not absurdly so.

---

## 2.5 Statistical Power

### What Power Means

Statistical power is the probability that a test correctly rejects $H_0$ when $H_1$ is actually true—i.e., the probability of detecting a real difference when one exists.

$$\text{Power} = 1 - \beta$$

where $\beta$ is the probability of a Type II error (failing to detect a real difference).

**The four linked quantities**: For any hypothesis test, these four quantities are mathematically linked—specifying any three determines the fourth:

1. **Significance level** ($\alpha$): Probability of a false positive. Typically set at 0.05.
2. **Power** ($1 - \beta$): Probability of detecting a real effect. Commonly targeted at 0.80 or higher.
3. **Effect size** ($\delta$): The magnitude of the true difference you want to detect.
4. **Sample size** ($n$): The number of observations (folds, datasets, test examples).

### Effect Size Estimation

**Cohen's $d$** for paired comparisons:

$$d = \frac{\bar{d}}{s_d}$$

where $\bar{d}$ is the mean difference and $s_d$ is the standard deviation of the differences. Conventional benchmarks:

| Cohen's $d$ | Interpretation |
|------------|---------------|
| 0.2        | Small effect   |
| 0.5        | Medium effect  |
| 0.8        | Large effect   |

In ML, a "small" effect might be a 0.5% accuracy improvement with high variance across folds, while a "large" effect might be a 5% improvement with low variance.

### Sample Size Requirements

For a paired t-test with significance level $\alpha$ and power $1 - \beta$:

$$n \geq \left(\frac{z_{1-\alpha/2} + z_{1-\beta}}{d}\right)^2$$

where $z_q$ is the $q$-th quantile of the standard normal distribution.

**Example**: To detect a medium effect ($d = 0.5$) with $\alpha = 0.05$ and power = 0.80:

$$n \geq \left(\frac{1.96 + 0.84}{0.5}\right)^2 = \left(\frac{2.80}{0.5}\right)^2 = 5.6^2 = 31.36 \approx 32$$

You need at least 32 paired observations. In ML, this might mean evaluating on 32 independent datasets, which is often impractical. This is a fundamental challenge: many ML comparisons are underpowered.

### Power in Cross-Validation

In $k$-fold CV, you get $k$ paired observations, but they are not independent. The effective sample size is smaller than $k$, making the test less powerful. A 10-fold CV gives 10 observations, but their effective count might be closer to 5-7 due to correlation. This is one reason the 5×2 cv test uses only 5 degrees of freedom: it honestly reflects the limited independent information available.

---

## 2.6 Multiple Testing Correction

### The Multiple Comparisons Problem

When you perform multiple hypothesis tests simultaneously, the probability of at least one false positive (a "family-wise" error) increases rapidly. If you compare 10 models pairwise (45 tests), each at $\alpha = 0.05$:

$$P(\text{at least one false positive}) = 1 - (1 - 0.05)^{45} \approx 0.90$$

You have a 90% chance of declaring at least one spurious "significant" difference. Two fundamentally different strategies exist for controlling this inflation.

### Family-Wise Error Rate (FWER) Control

**FWER**: The probability of making **one or more** false positive conclusions among all tests.

**Bonferroni correction**: The simplest FWER method. Divide the significance level by the number of tests:

$$\alpha_{\text{adjusted}} = \frac{\alpha}{m}$$

where $m$ is the number of tests. A test is significant only if its p-value falls below $\alpha/m$.

**Example**: With $m = 10$ tests and $\alpha = 0.05$, each individual test must achieve $p < 0.005$.

**Holm-Bonferroni (step-down) procedure**: A less conservative improvement over Bonferroni:
1. Sort the $m$ p-values in ascending order: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. Find the smallest $j$ such that $p_{(j)} > \frac{\alpha}{m - j + 1}$.
3. Reject all hypotheses with indices $1, \ldots, j-1$; do not reject $j, j+1, \ldots, m$.

Holm-Bonferroni is uniformly more powerful than Bonferroni (it always rejects at least as many hypotheses) while still controlling the FWER at level $\alpha$.

### False Discovery Rate (FDR) Control

**FDR**: The expected **proportion** of false positives among all rejected hypotheses.

$$\text{FDR} = E\left[\frac{\text{false positives}}{\text{total rejections}}\right]$$

FDR is a less stringent criterion than FWER. FWER controls the probability of **any** false positive; FDR allows some false positives as long as they are a small fraction of all discoveries. This is appropriate when you are screening many comparisons and can tolerate a few errors.

**Benjamini-Hochberg procedure**:
1. Sort the $m$ p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. Find the largest $j$ such that $p_{(j)} \leq \frac{j}{m} \alpha$.
3. Reject all hypotheses with indices $1, 2, \ldots, j$.

**Example**: With $m = 5$ tests, $\alpha = 0.05$, and p-values 0.001, 0.013, 0.029, 0.052, 0.610:

| Rank $j$ | p-value | BH threshold $\frac{j}{5} \times 0.05$ | Reject? |
|-----------|---------|----------------------------------------|---------|
| 1         | 0.001   | 0.010                                  | Yes     |
| 2         | 0.013   | 0.020                                  | Yes     |
| 3         | 0.029   | 0.030                                  | Yes     |
| 4         | 0.052   | 0.040                                  | No      |
| 5         | 0.610   | 0.050                                  | No      |

The largest $j$ where $p_{(j)} \leq \frac{j}{m}\alpha$ is $j = 3$. So we reject hypotheses 1, 2, and 3.

**FWER vs. FDR: when to use which**:
- FWER (Bonferroni, Holm): Use when the cost of any single false positive is high (e.g., claiming a drug works when it doesn't, deciding to deploy a model based on a flawed comparison).
- FDR (Benjamini-Hochberg): Use when you are performing many exploratory comparisons and can tolerate a small fraction of false discoveries (e.g., screening many feature combinations, comparing many model variants in a hyperparameter search).

---

# 3. Mathematical Formulation

## 3.1 Paired t-Test: Full Derivation

**Data**: $n$ paired differences $d_1, d_2, \ldots, d_n$ where $d_i = x_i^A - x_i^B$.

**Sample statistics**:

$$\bar{d} = \frac{1}{n} \sum_{i=1}^{n} d_i \qquad s_d^2 = \frac{1}{n-1} \sum_{i=1}^{n} (d_i - \bar{d})^2$$

**Test statistic**:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}} = \frac{\bar{d} \sqrt{n}}{s_d}$$

**Derivation of why this is $t$-distributed**: Under $H_0$, $\mu_d = 0$. Assume $d_i \overset{\text{iid}}{\sim} \mathcal{N}(\mu_d, \sigma_d^2)$. Then:

$$\bar{d} \sim \mathcal{N}\left(\mu_d, \frac{\sigma_d^2}{n}\right)$$

Standardizing:

$$Z = \frac{\bar{d} - \mu_d}{\sigma_d / \sqrt{n}} \sim \mathcal{N}(0,1)$$

Since $\sigma_d$ is unknown, we replace it with $s_d$. The ratio:

$$t = \frac{\bar{d} - \mu_d}{s_d / \sqrt{n}}$$

follows a Student-$t$ distribution with $n-1$ degrees of freedom. Under $H_0$ ($\mu_d = 0$):

$$t = \frac{\bar{d}}{s_d / \sqrt{n}} \sim t_{n-1}$$

The $n-1$ degrees of freedom arise because $s_d^2$ uses $n-1$ in its denominator (one degree of freedom is consumed by estimating $\bar{d}$).

**Two-sided p-value**:

$$p = 2 \cdot P(T > |t_{\text{obs}}|) \quad \text{where } T \sim t_{n-1}$$

## 3.2 Wilcoxon Signed-Rank Test: Full Formulation

**Data**: $n$ paired differences $d_1, \ldots, d_n$ with $d_i \neq 0$.

**Step 1**: Remove zero differences. Let $n_r$ be the count of non-zero differences.

**Step 2**: Rank $|d_1|, |d_2|, \ldots, |d_{n_r}|$. Assign rank $R_i$ to $|d_i|$.

**Step 3**: Compute:

$$W^+ = \sum_{i: d_i > 0} R_i \qquad W^- = \sum_{i: d_i < 0} R_i$$

Note that $W^+ + W^- = \frac{n_r(n_r+1)}{2}$ (the sum of all ranks from 1 to $n_r$).

**Test statistic**: $W = \min(W^+, W^-)$.

**Under $H_0$**: Each rank is equally likely to be positive or negative (like flipping $n_r$ coins). The expected value and variance of $W^+$ are:

$$E[W^+] = \frac{n_r(n_r+1)}{4} \qquad \text{Var}(W^+) = \frac{n_r(n_r+1)(2n_r+1)}{24}$$

For large $n_r$ (typically $n_r \geq 20$), a normal approximation is used:

$$Z = \frac{W^+ - E[W^+]}{\sqrt{\text{Var}(W^+)}} \approx \mathcal{N}(0,1)$$

**Tie correction**: When ties exist in $|d_i|$, the variance is adjusted:

$$\text{Var}_{\text{corrected}}(W^+) = \frac{n_r(n_r+1)(2n_r+1)}{24} - \sum_{j=1}^{g} \frac{t_j^3 - t_j}{48}$$

where $g$ is the number of tied groups and $t_j$ is the size of the $j$-th tied group.

## 3.3 Friedman Test: Full Derivation

**Data**: $k$ models evaluated on $n$ datasets (or folds). Let $r_i^{(j)}$ be the rank of model $i$ on dataset $j$.

**Average rank** for model $i$:

$$\bar{R}_i = \frac{1}{n} \sum_{j=1}^{n} r_i^{(j)}$$

**Under $H_0$** (all models equivalent): Each permutation of ranks $(1, 2, \ldots, k)$ is equally likely for each dataset. The expected average rank for each model is:

$$E[\bar{R}_i] = \frac{k+1}{2}$$

**Friedman statistic**:

$$\chi_F^2 = \frac{12n}{k(k+1)} \sum_{i=1}^{k} \left(\bar{R}_i - \frac{k+1}{2}\right)^2$$

This simplifies to:

$$\chi_F^2 = \frac{12n}{k(k+1)} \left[\sum_{i=1}^{k} \bar{R}_i^2 - \frac{k(k+1)^2}{4}\right]$$

**Distribution**: Under $H_0$, $\chi_F^2 \sim \chi^2_{k-1}$ approximately (exact for large $n$).

**Iman-Davenport correction** (recommended for improved approximation):

$$F_F = \frac{(n-1) \chi_F^2}{n(k-1) - \chi_F^2}$$

which follows an $F$-distribution with $(k-1)$ and $(k-1)(n-1)$ degrees of freedom.

## 3.4 Nemenyi Critical Difference

$$\text{CD} = q_\alpha \sqrt{\frac{k(k+1)}{6n}}$$

where $q_\alpha$ is the critical value of the Studentized range statistic for $k$ groups and infinite degrees of freedom, divided by $\sqrt{2}$.

**Selected critical values** ($q_\alpha$ for $\alpha = 0.05$):

| $k$ | $q_{0.05}$ |
|-----|-----------|
| 2   | 1.960     |
| 3   | 2.343     |
| 4   | 2.569     |
| 5   | 2.728     |
| 6   | 2.850     |
| 7   | 2.949     |
| 8   | 3.031     |
| 9   | 3.102     |
| 10  | 3.164     |

## 3.5 5×2 CV Paired t-Test: Detailed Formulation

**Notation**: Let $p_j^{(i)}$ be the performance difference in the $j$-th fold ($j \in \{1,2\}$) of the $i$-th replication ($i \in \{1,\ldots,5\}$).

**Within-replication mean and variance**:

$$\bar{p}^{(i)} = \frac{p_1^{(i)} + p_2^{(i)}}{2}$$

$$s_i^2 = (p_1^{(i)} - \bar{p}^{(i)})^2 + (p_2^{(i)} - \bar{p}^{(i)})^2 = \frac{(p_1^{(i)} - p_2^{(i)})^2}{2}$$

**Test statistic**:

$$\tilde{t} = \frac{p_1^{(1)}}{\sqrt{\frac{1}{5}\sum_{i=1}^{5} s_i^2}}$$

**Distribution under $H_0$**: Approximately $t_5$ (5 degrees of freedom).

**Why $p_1^{(1)}$ in the numerator**: Using the grand average $\bar{p}$ of all 10 differences would produce a statistic whose numerator and denominator are correlated, violating the assumptions of the $t$-distribution. Dietterich showed that using any single $p_j^{(i)}$ (conventionally $p_1^{(1)}$) avoids this problem while still being an unbiased estimate of the true difference under $H_1$.

## 3.6 Nadeau-Bengio Corrected Variance

For $r$ repeated train-test splits with training set fraction $n_{\text{train}}/N$ and test set fraction $n_{\text{test}}/N$:

$$\hat{\text{Var}}_{\text{corrected}}(\bar{d}) = \left(\frac{1}{r} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \hat{\sigma}_d^2$$

where:

$$\hat{\sigma}_d^2 = \frac{1}{r-1} \sum_{i=1}^{r} (d_i - \bar{d})^2$$

**Corrected $t$-statistic**:

$$t_{\text{corrected}} = \frac{\bar{d}}{\sqrt{\left(\frac{1}{r} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \hat{\sigma}_d^2}}$$

This follows a $t$-distribution with $r-1$ degrees of freedom.

**Interpretation of the correction factor**: The $1/r$ term is the standard variance reduction from averaging $r$ observations. The $n_{\text{test}}/n_{\text{train}}$ term captures the "leakage" of correlation due to overlapping training sets. As the test set fraction decreases (and training set fraction increases), $n_{\text{test}}/n_{\text{train}}$ shrinks, but the overlap between training sets grows. This term represents the irreducible variance floor due to training-set dependence.

## 3.7 Bayesian Correlated t-Test: Formulation

**Model**: $d_i \sim \mathcal{N}(\mu, \sigma^2)$ for $i = 1, \ldots, n$ (but correlated due to CV overlap).

**Prior**: Conjugate Normal-Inverse-Gamma:

$$\mu \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2 / \kappa_0) \qquad \sigma^2 \sim \text{Inv-}\Gamma(\nu_0 / 2, \nu_0 \sigma_0^2 / 2)$$

With vague prior: $\mu_0 = 0$, $\kappa_0 \to 0$, $\nu_0 \to 0$.

**Posterior** (accounting for correlation $\rho$ between folds):

$$\mu \mid \mathbf{d} \sim t_{\nu_n}\left(\bar{d}, \frac{s_d^2}{n} \cdot \frac{1}{1 + (n-1)\rho} \cdot \frac{\nu_n}{\nu_n - 2}\right)$$

Wait—more precisely, with the vague prior, the posterior of $\mu$ is:

$$\mu \mid \mathbf{d} \sim t_{n-1}\left(\bar{d}, \frac{s_d^2}{n_{\text{eff}}}\right)$$

where $n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}$ and the scale parameter of the Student-$t$ is $s_d^2 / n_{\text{eff}}$.

**Practical computation**: Once you have this posterior, you compute:

$$P(\mu > 0 \mid \mathbf{d}) = P(T > -\bar{d} \cdot \sqrt{n_{\text{eff}}} / s_d)$$

where $T \sim t_{n-1}$.

For the ROPE:

$$P(|\mu| < \delta \mid \mathbf{d}) = P\left(-\delta < \mu < \delta \mid \mathbf{d}\right)$$

computed by evaluating the CDF of the posterior $t$-distribution at $\delta$ and $-\delta$.

## 3.8 Multiple Testing Corrections: Formal Definitions

### Bonferroni

Given $m$ hypotheses with p-values $p_1, \ldots, p_m$, reject $H_i$ if:

$$p_i \leq \frac{\alpha}{m}$$

**Proof of FWER control**: Let $I_0$ be the set of true null hypotheses ($|I_0| \leq m$).

$$\text{FWER} = P\left(\bigcup_{i \in I_0} \{p_i \leq \alpha/m\}\right) \leq \sum_{i \in I_0} P(p_i \leq \alpha/m) = |I_0| \cdot \frac{\alpha}{m} \leq m \cdot \frac{\alpha}{m} = \alpha$$

The first inequality is the union bound (Boole's inequality).

### Holm-Bonferroni

Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$ with corresponding hypotheses $H_{(1)}, \ldots, H_{(m)}$.

$$\text{Reject } H_{(j)} \text{ if } p_{(j)} \leq \frac{\alpha}{m - j + 1} \text{ for all } j' \leq j$$

In practice: start from $j = 1$; reject as long as $p_{(j)} \leq \frac{\alpha}{m - j + 1}$; stop at the first non-rejection and accept all remaining.

### Benjamini-Hochberg

Sort p-values: $p_{(1)} \leq \cdots \leq p_{(m)}$.

$$\text{Let } j^* = \max\left\{j : p_{(j)} \leq \frac{j}{m} \alpha\right\}$$

Reject $H_{(1)}, \ldots, H_{(j^*)}$.

**Proof sketch of FDR control** (under independence): Let $V$ = number of false rejections, $R$ = total rejections. Under the assumption that p-values for true null hypotheses are uniformly distributed on $[0,1]$ and independent of each other:

$$\text{FDR} = E\left[\frac{V}{R} \cdot \mathbb{1}(R > 0)\right] \leq \frac{m_0}{m} \alpha \leq \alpha$$

where $m_0$ is the number of true nulls. The proof uses a martingale argument (Benjamini and Hochberg, 1995).

---

# 4. Worked Examples

## 4.1 Paired t-Test Example

**Scenario**: You compare Model A (random forest) and Model B (gradient boosting) using 10-fold cross-validation. The accuracy on each fold is:

| Fold | Model A | Model B | Difference $d_i$ |
|------|---------|---------|------------------|
| 1    | 0.85    | 0.87    | −0.02            |
| 2    | 0.88    | 0.89    | −0.01            |
| 3    | 0.82    | 0.86    | −0.04            |
| 4    | 0.90    | 0.88    | +0.02            |
| 5    | 0.86    | 0.90    | −0.04            |
| 6    | 0.84    | 0.85    | −0.01            |
| 7    | 0.87    | 0.89    | −0.02            |
| 8    | 0.83    | 0.87    | −0.04            |
| 9    | 0.89    | 0.88    | +0.01            |
| 10   | 0.86    | 0.91    | −0.05            |

**Step 1: Compute sample statistics.**

$$\bar{d} = \frac{(-0.02) + (-0.01) + (-0.04) + 0.02 + (-0.04) + (-0.01) + (-0.02) + (-0.04) + 0.01 + (-0.05)}{10} = \frac{-0.20}{10} = -0.02$$

$$s_d^2 = \frac{1}{9} \sum_{i=1}^{10} (d_i - (-0.02))^2$$

Computing each $(d_i + 0.02)^2$:

$(0)^2 + (0.01)^2 + (-0.02)^2 + (0.04)^2 + (-0.02)^2 + (0.01)^2 + (0)^2 + (-0.02)^2 + (0.03)^2 + (-0.03)^2$

$= 0 + 0.0001 + 0.0004 + 0.0016 + 0.0004 + 0.0001 + 0 + 0.0004 + 0.0009 + 0.0009 = 0.0048$

$$s_d^2 = \frac{0.0048}{9} = 0.000533 \qquad s_d = 0.0231$$

**Step 2: Compute test statistic.**

$$t = \frac{-0.02}{0.0231 / \sqrt{10}} = \frac{-0.02}{0.00730} = -2.74$$

**Step 3: Compute p-value.** With $n - 1 = 9$ degrees of freedom, the two-sided p-value for $|t| = 2.74$:

Using a $t$-table or software: $p \approx 0.023$.

**Step 4: Decision.** Since $p = 0.023 < 0.05 = \alpha$, we reject $H_0$. There is statistically significant evidence that Model B outperforms Model A.

**Effect size**: $d = \frac{|\bar{d}|}{s_d} = \frac{0.02}{0.0231} = 0.87$, which is a large effect.

**Caveat**: This standard paired t-test ignores the correlation between CV folds. The true p-value may be higher.

## 4.2 Wilcoxon Signed-Rank Test Example

Using the same data as above:

**Step 1: Rank absolute differences.**

| $d_i$ | $|d_i|$ | Rank |
|--------|---------|------|
| −0.01  | 0.01    | 1.5  |
| −0.01  | 0.01    | 1.5  |
| +0.01  | 0.01    | 1.5  |
| −0.02  | 0.02    | 4.5  |
| −0.02  | 0.02    | 4.5  |
| −0.02  | 0.02    | 4.5  |
| −0.04  | 0.04    | 8    |
| −0.04  | 0.04    | 8    |
| −0.04  | 0.04    | 8    |
| −0.05  | 0.05    | 10   |

Wait—let me re-rank. The sorted absolute values are: 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.04, 0.04, 0.04, 0.05.

- 0.01 appears 3 times: ranks 1, 2, 3 → average rank = 2
- 0.02 appears 3 times: ranks 4, 5, 6 → average rank = 5
- 0.04 appears 3 times: ranks 7, 8, 9 → average rank = 8
- 0.05 appears 1 time: rank 10

**Step 2: Assign signs and sum.**

| $d_i$ | $|d_i|$ | Rank | Sign  |
|--------|---------|------|-------|
| −0.02  | 0.02    | 5    | −     |
| −0.01  | 0.01    | 2    | −     |
| −0.04  | 0.04    | 8    | −     |
| +0.02  | 0.02    | 5    | +     |
| −0.04  | 0.04    | 8    | −     |
| −0.01  | 0.01    | 2    | −     |
| −0.02  | 0.02    | 5    | −     |
| −0.04  | 0.04    | 8    | −     |
| +0.01  | 0.01    | 2    | +     |
| −0.05  | 0.05    | 10   | −     |

$W^+ = 5 + 2 = 7$

$W^- = 5 + 2 + 8 + 8 + 2 + 5 + 8 + 10 = 48$

$W = \min(7, 48) = 7$

**Step 3: Decision.** For $n = 10$ and $\alpha = 0.05$ (two-sided), the critical value from a Wilcoxon table is $W_{\text{crit}} = 8$. Since $W = 7 \leq 8$, we reject $H_0$.

Conclusion: Consistent with the paired t-test, the Wilcoxon test also finds a significant difference.

## 4.3 Friedman Test with Nemenyi Post-Hoc Example

**Scenario**: Five classifiers (A, B, C, D, E) evaluated on six datasets:

| Dataset | A    | B    | C    | D    | E    |
|---------|------|------|------|------|------|
| 1       | 0.85 | 0.89 | 0.82 | 0.91 | 0.87 |
| 2       | 0.78 | 0.83 | 0.76 | 0.85 | 0.80 |
| 3       | 0.92 | 0.90 | 0.88 | 0.94 | 0.91 |
| 4       | 0.80 | 0.84 | 0.79 | 0.86 | 0.82 |
| 5       | 0.88 | 0.87 | 0.85 | 0.90 | 0.89 |
| 6       | 0.83 | 0.86 | 0.81 | 0.88 | 0.84 |

**Step 1: Rank within each dataset** (1 = best, 5 = worst):

| Dataset | A | B | C | D | E |
|---------|---|---|---|---|---|
| 1       | 4 | 2 | 5 | 1 | 3 |
| 2       | 4 | 2 | 5 | 1 | 3 |
| 3       | 1 | 3 | 5 | 1 → wait |   |

Let me re-rank dataset 3: D=0.94 (rank 1), A=0.92 (rank 2), E=0.91 (rank 3), B=0.90 (rank 4), C=0.88 (rank 5).

| Dataset | A | B | C | D | E |
|---------|---|---|---|---|---|
| 1       | 4 | 2 | 5 | 1 | 3 |
| 2       | 4 | 2 | 5 | 1 | 3 |
| 3       | 2 | 4 | 5 | 1 | 3 |
| 4       | 4 | 2 | 5 | 1 | 3 |
| 5       | 3 | 4 | 5 | 1 | 2 |
| 6       | 4 | 2 | 5 | 1 | 3 |

**Step 2: Compute average ranks.**

$\bar{R}_A = (4+4+2+4+3+4)/6 = 21/6 = 3.50$

$\bar{R}_B = (2+2+4+2+4+2)/6 = 16/6 = 2.67$

$\bar{R}_C = (5+5+5+5+5+5)/6 = 30/6 = 5.00$

$\bar{R}_D = (1+1+1+1+1+1)/6 = 6/6 = 1.00$

$\bar{R}_E = (3+3+3+3+2+3)/6 = 17/6 = 2.83$

**Step 3: Compute Friedman statistic.**

$$\chi_F^2 = \frac{12 \times 6}{5 \times 6} \left[3.50^2 + 2.67^2 + 5.00^2 + 1.00^2 + 2.83^2 - \frac{5 \times 36}{4}\right]$$

$$= \frac{72}{30} \left[12.25 + 7.13 + 25.00 + 1.00 + 8.01 - 45.00\right]$$

$$= 2.4 \times [53.39 - 45.00] = 2.4 \times 8.39 = 20.14$$

**Step 4: p-value.** With $k - 1 = 4$ degrees of freedom, $\chi^2_{4} = 20.14$ gives $p < 0.001$. We reject $H_0$: at least one model differs significantly.

**Step 5: Nemenyi post-hoc.** With $k = 5$ models, $n = 6$ datasets, $\alpha = 0.05$:

$$\text{CD} = q_{0.05} \sqrt{\frac{k(k+1)}{6n}} = 2.728 \times \sqrt{\frac{5 \times 6}{6 \times 6}} = 2.728 \times \sqrt{\frac{30}{36}} = 2.728 \times 0.913 = 2.49$$

Two models are significantly different if $|\bar{R}_i - \bar{R}_j| > 2.49$.

Pairwise rank differences:

| Pair   | $|\bar{R}_i - \bar{R}_j|$ | Significant? |
|--------|---------------------------|-------------|
| D vs C | $|1.00 - 5.00| = 4.00$   | Yes         |
| D vs A | $|1.00 - 3.50| = 2.50$   | Yes (barely)|
| D vs E | $|1.00 - 2.83| = 1.83$   | No          |
| D vs B | $|1.00 - 2.67| = 1.67$   | No          |
| B vs C | $|2.67 - 5.00| = 2.33$   | No          |
| E vs C | $|2.83 - 5.00| = 2.17$   | No          |
| A vs C | $|3.50 - 5.00| = 1.50$   | No          |
| B vs A | $|2.67 - 3.50| = 0.83$   | No          |
| E vs A | $|2.83 - 3.50| = 0.67$   | No          |
| B vs E | $|2.67 - 2.83| = 0.16$   | No          |

**Conclusion**: Model D is significantly better than C, and marginally better than A. No other pairwise differences reach significance with $n = 6$ datasets (the test is quite conservative with few datasets).

## 4.4 5×2 CV Paired t-Test Example

**Scenario**: Comparing two models. For each of 5 replications, we randomly split data into two halves, train on each half, test on the other, and record the performance difference.

| Replication $i$ | $p_1^{(i)}$ | $p_2^{(i)}$ | $\bar{p}^{(i)}$ | $s_i^2$ |
|-----------------|-------------|-------------|-----------------|---------|
| 1               | 0.030       | 0.010       | 0.020           | $(0.030-0.020)^2 + (0.010-0.020)^2 = 0.0002$ |
| 2               | 0.025       | 0.015       | 0.020           | $(0.005)^2 + (-0.005)^2 = 0.00005$ |
| 3               | 0.040       | −0.005      | 0.0175          | $(0.0225)^2 + (-0.0225)^2 = 0.001013$ |
| 4               | 0.020       | 0.010       | 0.015           | $(0.005)^2 + (-0.005)^2 = 0.00005$ |
| 5               | 0.035       | 0.005       | 0.020           | $(0.015)^2 + (-0.015)^2 = 0.00045$ |

$$\frac{1}{5}\sum_{i=1}^{5} s_i^2 = \frac{0.0002 + 0.00005 + 0.001013 + 0.00005 + 0.00045}{5} = \frac{0.001763}{5} = 0.000353$$

$$\tilde{t} = \frac{p_1^{(1)}}{\sqrt{0.000353}} = \frac{0.030}{0.01878} = 1.597$$

**p-value** from $t_5$ distribution: $p \approx 0.17$ (two-sided).

**Conclusion**: At $\alpha = 0.05$, we do not reject $H_0$. Despite the positive mean difference across replications, the 5×2 CV test does not find sufficient evidence that the models differ. Note how this test is more conservative than a naive paired t-test—this is by design, to control Type I error.

## 4.5 Bayesian Correlated t-Test Example

**Scenario**: 10-fold CV differences: $\bar{d} = -0.02$, $s_d = 0.0231$, $n = 10$, $\rho = 1/k = 0.1$.

**Effective sample size**:

$$n_{\text{eff}} = \frac{10}{1 + 9 \times 0.1} = \frac{10}{1.9} = 5.26$$

**Posterior**: $\mu \mid \mathbf{d} \sim t_9\left(-0.02, \frac{0.0231^2}{5.26}\right) = t_9(-0.02, 0.0001015)$

The posterior standard deviation is $\sqrt{0.0001015} = 0.01007$.

**Compute probabilities** (with ROPE $\delta = 0.01$):

$P(\mu > 0 \mid \mathbf{d})$: Standardize: $z = (0 - (-0.02)) / 0.01007 = 1.986$. From the $t_9$ distribution: $P(T > 1.986) \approx 0.039$. So there is about a 3.9% posterior probability that Model A is better.

$P(\mu < -0.01 \mid \mathbf{d})$: Standardize: $z = (-0.01 - (-0.02))/0.01007 = 0.993$. $P(T < 0.993) \approx 0.82$. So there is about a 17.6% probability that the difference is within the ROPE.

$P(\mu < 0 \mid \mathbf{d}) \approx 0.961$. There is a 96.1% probability that Model B is genuinely better.

**Interpretation**: The Bayesian analysis tells us that Model B is very likely better, but the evidence for a practically significant difference (beyond the ROPE) is moderate, not overwhelming.

## 4.6 Benjamini-Hochberg Example

**Scenario**: You compare 6 pairs of models and obtain p-values: 0.003, 0.012, 0.035, 0.041, 0.150, 0.420.

| Rank $j$ | p-value | BH threshold $\frac{j}{6}\times 0.05$ | Reject? |
|-----------|---------|----------------------------------------|---------|
| 1         | 0.003   | 0.00833                                | Yes     |
| 2         | 0.012   | 0.01667                                | Yes     |
| 3         | 0.035   | 0.02500                                | No      |
| 4         | 0.041   | 0.03333                                | No      |
| 5         | 0.150   | 0.04167                                | No      |
| 6         | 0.420   | 0.05000                                | No      |

The largest $j$ where $p_{(j)} \leq \frac{j}{6} \times 0.05$ is $j = 2$. So we reject hypotheses 1 and 2 only.

Note that $p_{(3)} = 0.035 < 0.05$ would have been rejected without correction, but the BH procedure correctly identifies that this is not significant after accounting for multiple comparisons.

Compare with Bonferroni: $\alpha_{\text{adjusted}} = 0.05/6 = 0.00833$. Only hypothesis 1 ($p = 0.003$) would survive. BH is less conservative, retaining hypothesis 2.

---

# 5. Relevance to Machine Learning Practice

## 5.1 Where These Tests Are Used

**Model selection during development**: Before deploying a new model, compare it against the existing production model using a paired test on cross-validation results or multiple evaluation splits.

**Benchmark papers**: Research papers comparing multiple algorithms across standard datasets use the Friedman test with Nemenyi post-hoc to make rigorous claims about relative performance.

**AutoML and hyperparameter optimization**: When evaluating many configurations, FDR control helps identify which configurations are genuinely superior without being overwhelmed by false positives.

**A/B testing in production**: While not identical to the ML-specific tests discussed here, the statistical principles (paired comparisons, multiple testing correction) transfer directly to online experimentation. The key difference is that A/B tests typically have independent observations (different users), avoiding the CV correlation problem.

**Regulatory submissions**: In medical AI, statistical evidence of non-inferiority or superiority is required for regulatory approval. Power analysis determines the required sample size before the study begins.

## 5.2 Decision Guide: Which Test to Use

| Situation | Recommended Test | Why |
|-----------|-----------------|-----|
| Two models, enough data for repeated CV | 5×2 CV paired t-test | Controls Type I error properly |
| Two models, single train-test split | McNemar's test (per-example) | Proper for single split |
| Two models, limited data | Bayesian correlated t-test | Quantifies uncertainty, handles small samples |
| Two models, non-normal differences | Wilcoxon signed-rank | No normality assumption |
| Multiple models, multiple datasets | Friedman + Nemenyi | Proper omnibus + post-hoc |
| Many pairwise comparisons | Apply Holm-Bonferroni or BH | Controls FWER or FDR |
| Need probability of superiority | Bayesian correlated t-test or Bayes factors | Directly answers the question |

## 5.3 Trade-Offs

### Parametric vs. Non-Parametric

**Parametric** (t-tests): More powerful when normality holds; sensitive to outliers; produce interpretable effect sizes (Cohen's $d$). Preferred when $n$ is large enough for CLT or differences are approximately normal.

**Non-parametric** (Wilcoxon, Friedman): Robust to non-normality and outliers; less powerful when normality holds; work with ordinal data. Preferred when differences are skewed, when $n$ is small, or when you are ranking models across heterogeneous datasets.

### Frequentist vs. Bayesian

**Frequentist**: Binary decision (reject/don't reject); well-established and widely understood; does not require prior specification; can be gamed by optional stopping.

**Bayesian**: Provides full posterior; naturally handles small samples; ROPE allows "practically equivalent" conclusion; requires prior specification (though with enough data, results are robust to prior choice); immune to optional stopping.

### Power vs. Type I Error Control

More stringent multiple testing corrections (Bonferroni) reduce false positives but increase false negatives. Less stringent corrections (BH) increase power but tolerate more false discoveries. The choice depends on the cost of errors in your application.

## 5.4 Practical Considerations

**How many folds/splits?** More folds or replications provide lower-variance estimates but create more overlap (in CV) or reduce training set size (in random splits). The 5×2 CV test balances these concerns. For power analysis, 30+ independent paired observations are often needed for medium effects, which means evaluating on 30+ independent datasets—a luxury most practitioners do not have.

**What metric?** The choice of evaluation metric affects the distribution of differences. Accuracy differences on imbalanced datasets can have unusual distributions. AUC or F1 may produce more normally distributed differences.

**Computational cost**: The 5×2 CV test requires training each model 10 times (5 replications × 2 folds). For expensive models (large neural networks), this may be impractical. In such cases, a single well-chosen train-test split with McNemar's test, or a Bayesian analysis of a single k-fold CV, may be more feasible.

**Effect size matters more than p-values**: A statistically significant 0.1% accuracy improvement is unlikely to matter in practice. Always report effect sizes alongside p-values, and use ROPE-based Bayesian analysis when possible.

---

# 6. Common Pitfalls & Misconceptions

## Pitfall 1: Using the Standard Paired t-Test on CV Folds Without Correction

**The mistake**: Running a standard paired t-test on 10-fold CV results and treating the p-value at face value.

**Why it's wrong**: The 10 fold-level performance estimates are not independent because their training sets overlap heavily (80% overlap in 10-fold CV). This positive correlation inflates the denominator of the standard error estimate... actually, the correlation **reduces** the true variance of the mean difference, but the standard paired t-test computes the variance as if the observations were independent. The net effect is complex, but Dietterich (1998) showed empirically that the standard paired t-test on CV has an inflated Type I error rate (often 10-15% instead of the nominal 5%).

**Fix**: Use the 5×2 CV test, the Nadeau-Bengio corrected test, or a Bayesian correlated t-test.

## Pitfall 2: Confusing Statistical Significance with Practical Significance

**The mistake**: Concluding that "Model B is better" solely because $p < 0.05$, without examining the magnitude of the difference.

**Why it's wrong**: With enough data, even a 0.01% accuracy improvement can be statistically significant. But if deploying Model B costs $100K in engineering effort and the improvement translates to $500 in business value, it is not practically significant.

**Fix**: Always report and interpret effect sizes. Use ROPE-based Bayesian analysis to distinguish between "significantly different" and "meaningfully different."

## Pitfall 3: Performing Multiple Pairwise Comparisons Without Correction

**The mistake**: Comparing 5 models pairwise (10 tests), finding 2 "significant" results, and reporting them as genuine findings.

**Fix**: Use the Friedman test as an omnibus test first. If it rejects, apply Nemenyi or another post-hoc test with proper correction.

## Pitfall 4: Misinterpreting "Failure to Reject" as "No Difference"

**The mistake**: Concluding "the models are equivalent" when a test fails to reject $H_0$.

**Why it's wrong**: Failure to reject $H_0$ means the data is consistent with $H_0$, but it may also be consistent with a wide range of values under $H_1$. The test may simply lack power (too few datasets, too much variance).

**Fix**: Report confidence intervals for the difference. If the interval is narrow and contains zero, you have evidence for equivalence. If it is wide, you have an inconclusive result. Bayesian analysis naturally addresses this by computing $P(|\mu| < \delta)$.

## Pitfall 5: Cherry-Picking the Test

**The mistake**: Trying multiple statistical tests and reporting only the one that gives $p < 0.05$.

**Why it's wrong**: This is a form of p-hacking. Each test has different assumptions, power, and Type I error profiles. If you run three tests, you have roughly a $1 - 0.95^3 \approx 14\%$ chance of finding significance by chance.

**Fix**: Pre-specify the test before running the experiment. If you have a reason to run multiple tests (e.g., a parametric and non-parametric test for robustness), report all results and discuss discrepancies.

## Pitfall 6: Ignoring the Correlation Structure in Bayesian Tests

**The mistake**: Applying a Bayesian t-test to CV folds without accounting for the correlation $\rho$ between folds.

**Why it's wrong**: The posterior width will be too narrow, leading to overconfident conclusions. With $n = 10$ and $\rho = 0.1$, the effective sample size drops from 10 to about 5.3, which roughly doubles the posterior standard deviation.

**Fix**: Always use the Bayesian **correlated** t-test (Corani et al., 2017) with an appropriate correlation estimate.

## Pitfall 7: Using Bonferroni When BH Would Be Appropriate

**The mistake**: Applying Bonferroni correction in an exploratory analysis with 50 comparisons, then finding "nothing significant" because the threshold is $0.05/50 = 0.001$.

**Why it's wrong**: Bonferroni controls the probability of **any** false positive, which is unnecessarily stringent for exploratory analysis. You might miss many real findings.

**Fix**: Use FDR control (Benjamini-Hochberg) for exploratory analyses where you can tolerate a small fraction of false discoveries. Reserve Bonferroni/Holm for confirmatory analyses where each test represents a real-world decision.

## Pitfall 8: Using the Wrong $n$

**The mistake**: In the paired t-test, using the number of test examples as $n$ instead of the number of folds.

**Why it's wrong**: Each fold produces one (correlated) performance estimate. The unit of analysis is the fold, not the individual example. Using $n = 10000$ (test examples) instead of $n = 10$ (folds) gives impossibly small p-values.

**Fix**: $n$ is the number of paired performance measurements (folds, splits, or datasets), not the number of data points.

## Pitfall 9: Inconsistent Evaluation Protocol

**The mistake**: Training Model A with one data split and Model B with a different split, then comparing their test performances.

**Why it's wrong**: Differences in the train/test partition confound with differences in model performance. You cannot tell whether Model B is better or whether it simply got an easier test set.

**Fix**: Always use the same splits for both models. In CV, this means running both models on exactly the same fold assignments. In random subsampling, use the same random seeds.

## Pitfall 10: Overreliance on p-values for Reporting

**The mistake**: Reporting only "$p = 0.03$" without confidence intervals, effect sizes, or any measure of practical importance.

**Fix**: Report the mean difference, its confidence interval, the effect size (Cohen's $d$), and ideally the Bayesian posterior probabilities including the ROPE analysis.

---

# Interview Preparation Section

## Foundational Questions

### Q1: What is the difference between a paired and an unpaired test? Why do we prefer paired tests for model comparison?

**Answer**: An unpaired (independent two-sample) test compares two groups of observations that are not matched—each observation in group A has no specific correspondence to any observation in group B. A paired test matches observations: the $i$-th observation in group A is linked to the $i$-th observation in group B.

For model comparison, pairing is natural: both models are evaluated on the same data (same folds, same test examples). This controls for variation in data difficulty. The variance of the paired difference $d_i = x_i^A - x_i^B$ is $\text{Var}(x^A) + \text{Var}(x^B) - 2\text{Cov}(x^A, x^B)$. Since models evaluated on the same data are positively correlated (both do well on easy examples), the covariance term is positive, reducing the variance of the differences. Lower variance means higher statistical power—it is easier to detect a real difference.

### Q2: What does a p-value actually mean? What doesn't it mean?

**Answer**: A p-value is the probability of observing a test statistic at least as extreme as the one computed, **under the assumption that the null hypothesis is true**. If $p = 0.03$, there is a 3% chance of seeing data this extreme if the null (e.g., "both models are equally good") is true.

Common misconceptions: (1) The p-value is **not** the probability that $H_0$ is true. That would be $P(H_0 \mid \text{data})$, which requires Bayesian reasoning. (2) $1 - p$ is **not** the probability that $H_1$ is true. (3) The p-value does **not** measure the size of the effect. A small p-value can arise from a tiny effect with a large sample size. (4) $p > 0.05$ does **not** mean "no effect"—it means insufficient evidence to rule out $H_0$.

### Q3: Why is the standard paired t-test on cross-validation folds problematic?

**Answer**: In $k$-fold cross-validation, the training sets of different folds overlap substantially. In 10-fold CV, any two training sets share about 80% of their data. This means the performance estimates from different folds are positively correlated, violating the independence assumption of the paired t-test. The consequence is that the test underestimates the variance of the mean difference, producing inflated test statistics and p-values that are too small. Dietterich (1998) showed the Type I error can reach 10-15% instead of the nominal 5%. Solutions include the 5×2 CV test (which avoids overlapping training sets within each replication), the Nadeau-Bengio corrected variance (which adds a term $n_{\text{test}}/n_{\text{train}}$), and the Bayesian correlated t-test (which explicitly models the correlation $\rho$).

### Q4: What is the Friedman test, and when should you use it instead of multiple pairwise tests?

**Answer**: The Friedman test is a non-parametric omnibus test for comparing $k \geq 3$ treatments (models) across $n$ blocks (datasets or folds). It ranks the models within each block and tests whether the average ranks are significantly different from what would be expected under the null (all models equivalent).

You should use it instead of multiple pairwise tests because performing $\binom{k}{2}$ pairwise tests inflates the family-wise error rate. For $k = 5$, you perform 10 tests, giving roughly a 40% chance of at least one false positive at $\alpha = 0.05$. The Friedman test controls the overall Type I error at the specified $\alpha$, and if it rejects, you can follow up with post-hoc tests (like Nemenyi) that also control the family-wise error.

### Q5: Explain the Region of Practical Equivalence (ROPE) and why it matters.

**Answer**: The ROPE is a range $(-\delta, +\delta)$ around zero within which the performance difference between two models is considered too small to be practically meaningful. For example, $\delta = 0.01$ means a 1% accuracy difference is negligible.

It matters because statistical significance does not imply practical significance. A test might detect a 0.1% accuracy improvement with $p < 0.01$, but this difference is meaningless in most applications. Bayesian methods naturally incorporate the ROPE: you compute $P(|\mu| < \delta \mid \text{data})$, which gives the probability that the models are practically equivalent. This allows three conclusions—(1) Model A is practically better, (2) Model B is practically better, (3) the models are practically equivalent—rather than the binary reject/don't-reject of frequentist testing. The third conclusion is impossible in standard frequentist testing, where failing to reject only means "insufficient evidence."

---

## Mathematical Questions

### Q6: Derive the Nadeau-Bengio corrected variance for repeated train-test splits.

**Answer**: Consider $r$ repeated random splits. In each split $i$, you have a training set of size $n_{\text{train}}$ and a test set of size $n_{\text{test}}$. Let $d_i$ be the performance difference in split $i$.

In the standard setting with independent observations, $\text{Var}(\bar{d}) = \sigma_d^2 / r$. But here, $d_i$ and $d_j$ are correlated because their training sets overlap. Following Nadeau and Bengio (2003), assume a constant pairwise correlation $\rho$ between any two $d_i$ and $d_j$:

$$\text{Var}(\bar{d}) = \text{Var}\left(\frac{1}{r}\sum_{i=1}^{r} d_i\right) = \frac{1}{r^2}\left(\sum_{i} \text{Var}(d_i) + \sum_{i \neq j}\text{Cov}(d_i, d_j)\right)$$

$$= \frac{1}{r^2}\left(r\sigma_d^2 + r(r-1)\rho\sigma_d^2\right) = \frac{\sigma_d^2}{r}\left(1 + (r-1)\rho\right)$$

For large $r$, this approaches $\rho \sigma_d^2$, meaning adding more splits cannot reduce the variance below $\rho \sigma_d^2$.

Nadeau and Bengio showed that $\rho \approx n_{\text{test}}/n_{\text{train}}$ (a result derived from the analysis of linear estimators). Substituting:

$$\text{Var}(\bar{d}) \approx \frac{\sigma_d^2}{r} + \frac{n_{\text{test}}}{n_{\text{train}}} \sigma_d^2 = \left(\frac{1}{r} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)\sigma_d^2$$

The corrected $t$-statistic uses this in the denominator instead of $\hat{\sigma}_d^2 / r$.

### Q7: Why does the 5×2 CV test use $p_1^{(1)}$ in the numerator instead of the grand mean?

**Answer**: The test statistic is $\tilde{t} = p_1^{(1)} / \sqrt{\frac{1}{5}\sum s_i^2}$. Using the grand mean $\bar{p}$ (averaged over all 10 differences) would be problematic because $\bar{p}$ and the variance estimate $\sum s_i^2$ would be **correlated**. The $t$-distribution assumes the numerator and denominator are independent (more precisely, the numerator is normal and the denominator is an independent $\chi^2$-based estimate of the variance). When numerator and denominator are correlated, the resulting ratio no longer follows a $t$-distribution, and the p-values become unreliable.

By using a single observation $p_1^{(1)}$ in the numerator, which is independent of the variance estimate (since $s_1^2$ only uses the difference $p_1^{(1)} - p_2^{(1)}$, not the value of $p_1^{(1)}$ itself... actually, $s_1^2$ does depend on $p_1^{(1)}$). The deeper reason is that Dietterich empirically verified that this particular formulation controls Type I error at the nominal level, whereas using the grand mean does not. The test sacrifices some power (using a single observation instead of the average) to maintain the correct Type I error rate.

### Q8: Prove that the Bonferroni correction controls the family-wise error rate.

**Answer**: Let $H_1, \ldots, H_m$ be $m$ null hypotheses, with $I_0 \subseteq \{1, \ldots, m\}$ being the index set of truly null hypotheses ($|I_0| = m_0 \leq m$). Using the Bonferroni-adjusted threshold $\alpha/m$:

$$\text{FWER} = P\left(\bigcup_{i \in I_0} \{p_i \leq \alpha/m\}\right)$$

By the union bound (Boole's inequality):

$$\leq \sum_{i \in I_0} P(p_i \leq \alpha/m)$$

Under the null, each $p_i$ is uniformly distributed on $[0, 1]$ (or stochastically greater, for discrete tests), so $P(p_i \leq \alpha/m) \leq \alpha/m$:

$$\leq m_0 \cdot \frac{\alpha}{m} \leq m \cdot \frac{\alpha}{m} = \alpha$$

This proof makes no assumption about the dependence structure between tests—Bonferroni is valid regardless of whether the tests are independent, positively correlated, or negatively correlated. This generality is also its weakness: by using the union bound, it is conservative whenever the tests are positively correlated (which is the common case in model comparison).

### Q9: How does the Bayes Factor relate to the posterior odds?

**Answer**: By Bayes' theorem:

$$\frac{P(H_1 \mid \text{data})}{P(H_0 \mid \text{data})} = \frac{P(\text{data} \mid H_1)}{P(\text{data} \mid H_0)} \cdot \frac{P(H_1)}{P(H_0)}$$

That is:

$$\text{Posterior odds} = \text{Bayes Factor} \times \text{Prior odds}$$

If $P(H_0) = P(H_1) = 0.5$ (equal prior odds), then the posterior odds equal the Bayes Factor. A $\text{BF}_{10} = 5$ means the data has shifted the odds 5:1 in favor of $H_1$, regardless of the prior. If we started with prior odds 1:1, the posterior odds are 5:1, giving $P(H_1 \mid \text{data}) = 5/6 \approx 0.83$.

The Bayes Factor is attractive because it separates the evidence provided by the data from prior beliefs. However, it depends critically on the prior **within** each hypothesis—specifically, the prior on the parameters under $H_1$. This is the Lindley paradox: with a very diffuse prior under $H_1$, the Bayes Factor can favor $H_0$ even when the data strongly suggest $H_1$, because the prior mass is spread too thinly.

### Q10: Derive the effective sample size formula for correlated observations.

**Answer**: Given $n$ observations with pairwise correlation $\rho$ and common variance $\sigma^2$:

$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^{n} X_i\right) = \frac{1}{n^2}\left(\sum_i \text{Var}(X_i) + \sum_{i \neq j} \text{Cov}(X_i, X_j)\right)$$

$$= \frac{1}{n^2}\left(n\sigma^2 + n(n-1)\rho\sigma^2\right) = \frac{\sigma^2}{n}\left(1 + (n-1)\rho\right)$$

For independent observations, this would equal $\sigma^2 / n_{\text{eff}}$. Setting these equal:

$$\frac{\sigma^2}{n_{\text{eff}}} = \frac{\sigma^2}{n}(1 + (n-1)\rho)$$

$$n_{\text{eff}} = \frac{n}{1 + (n-1)\rho}$$

For $\rho = 0$: $n_{\text{eff}} = n$ (independent). For $\rho = 1$: $n_{\text{eff}} = 1$ (perfectly correlated, effectively one observation). For $\rho = 0.1$, $n = 10$: $n_{\text{eff}} = 10/1.9 \approx 5.3$.

---

## Applied Questions

### Q11: You are comparing a production logistic regression model against a proposed neural network. You have a dataset of 50,000 examples. Design a statistically rigorous evaluation protocol.

**Answer**: I would use the following protocol:

**Step 1: Choose the evaluation strategy.** Given 50K examples, I can afford repeated train-test splits. I would use the 5×2 CV paired t-test or, for richer information, the Bayesian correlated t-test with 10-fold CV.

**Step 2: Fix the splits before training.** Generate all random splits (or fold assignments) before training any model. Both models must use exactly the same splits.

**Step 3: Choose the metric.** Select the primary metric that aligns with the business objective (e.g., AUC for ranking, F1 for imbalanced classification, calibration metrics for probability estimation). Pre-register this choice.

**Step 4: Run the comparison.** Train both models on each training split, evaluate on the corresponding test split, and record the paired differences.

**Step 5: Statistical test.** Apply the 5×2 CV paired t-test or the Bayesian correlated t-test. Report the p-value or posterior probabilities, the mean difference, the confidence/credible interval, and Cohen's $d$.

**Step 6: Practical significance.** Define a ROPE (e.g., $\delta = 0.005$ for AUC). Check whether the difference exceeds the ROPE. If using Bayesian methods, report $P(\mu > \delta)$, $P(|\mu| < \delta)$, and $P(\mu < -\delta)$.

**Step 7: Secondary analyses.** Check performance on subgroups (e.g., different user segments) to ensure the neural network doesn't improve aggregate performance while degrading on important subpopulations.

### Q12: Your team has run a large experiment comparing 8 model architectures on 15 benchmark datasets. How do you analyze the results?

**Answer**:

**Phase 1: Omnibus test.** Apply the Friedman test with $k = 8$ models and $n = 15$ datasets. Compute the Friedman statistic and its p-value. If $p > \alpha$, there is no evidence that any model differs significantly—stop here.

**Phase 2: Post-hoc pairwise comparisons.** If the Friedman test rejects, apply the Nemenyi test. Compute the critical difference $\text{CD} = q_\alpha \sqrt{k(k+1)/(6n)}$ and identify all pairs of models whose average rank difference exceeds CD.

**Phase 3: Visualization.** Create a Critical Difference diagram showing all models on a rank axis, with bars connecting models that are not significantly different.

**Phase 4: Practical interpretation.** Beyond the ranks, examine the actual performance scores. A model with the best rank might win by tiny margins on most datasets but lose badly on a few. Report the win/tie/loss counts, median performance differences, and the specific datasets where models diverge.

**Alternative to Nemenyi**: If there is a designated control model (e.g., the current production model), use the Bonferroni-Dunn test instead of Nemenyi. This tests only $k - 1$ comparisons (each model against the control) instead of $\binom{k}{2}$, giving more power.

### Q13: You have a very expensive model that takes 12 hours to train. You cannot afford the 10 training runs required by the 5×2 CV test. What do you do?

**Answer**: Several options:

**Option 1: Single train-test split with McNemar's test.** Split data once (e.g., 80/20). Train both models. On the test set, construct a 2×2 contingency table counting examples that (a) both get right, (b) A right and B wrong, (c) A wrong and B right, (d) both get wrong. McNemar's test examines whether (b) and (c) differ significantly. This requires only 2 training runs (one per model) and tests at the per-example level, giving high statistical power for large test sets.

**Option 2: Bootstrap the test set predictions.** Train each model once. On the test set, generate $B = 1000$ bootstrap samples, compute the metric on each, and compare the distributions. This does not re-train the model, so the bootstrap only captures test-set variability, not training-set variability—but for a large test set, this may be sufficient.

**Option 3: Bayesian analysis of a single k-fold CV.** If you can afford $k$ training runs (e.g., $k = 5$ for a total of 10 training runs), use the Bayesian correlated t-test with $\rho = 1/k$. This gives you posterior probabilities despite the small number of folds.

**Option 4: Nested cross-validation with fewer folds.** Use 3-fold or 5-fold CV (3-6 training runs per model, so 6-12 total) with the Nadeau-Bengio corrected test. Accept that power will be limited.

The key trade-off is between computational cost and statistical rigor. With very few training runs, you are inevitably underpowered, and you should be transparent about this when reporting results.

### Q14: You are running a neural architecture search that evaluates 200 candidate architectures. How do you identify the best ones while controlling for multiple comparisons?

**Answer**: With 200 architectures and a single control (e.g., the simplest baseline), you have 199 pairwise comparisons. If comparing all pairs, you have $\binom{200}{2} = 19900$ comparisons.

**Recommended approach**: Use FDR control (Benjamini-Hochberg) rather than FWER control. This is an exploratory search, and you are willing to tolerate a small fraction of false discoveries among your "top architectures." At FDR = 0.05, on average 5% of the architectures you identify as "significantly better than the baseline" will actually not be.

**Procedure**:
1. For each of the 199 architectures, compute a p-value for the comparison against the control (using whatever paired test is appropriate for your evaluation protocol).
2. Apply the Benjamini-Hochberg procedure to the 199 p-values.
3. The architectures that survive the BH correction are your "shortlist."
4. From this shortlist, perform more rigorous evaluation (e.g., 5×2 CV test) on the top 5-10 candidates before selecting a final architecture.

**Why not Bonferroni?** With $m = 199$, Bonferroni requires $p < 0.05/199 = 0.000251$ for any individual architecture to be "significant." This is extremely conservative and will miss many genuinely good architectures.

---

## Debugging & Failure-Mode Questions

### Q15: You run a paired t-test comparing two models on 10-fold CV and get $p = 0.001$. But when you retrain with a different random seed, the "winning" model changes. What went wrong?

**Answer**: This is a classic symptom of the inflated Type I error of the uncorrected paired t-test on CV folds. The extremely low p-value ($p = 0.001$) is an artifact of the test treating the 10 fold estimates as independent, when in reality they share ~80% of their training data. The true variance of the mean difference is much larger than what the standard paired t-test computes.

When you retrain with a different seed, you get different fold assignments (and possibly different model initializations), revealing the true instability of the comparison. The "significant" result was driven by the particular fold assignment, not by a genuine model difference.

**Fix**: Use the 5×2 CV test (which would likely give $p \gg 0.05$ here) or the Nadeau-Bengio corrected test. Also, examine the effect size: if Cohen's $d < 0.2$, the difference was small to begin with.

### Q16: You apply the Friedman test to compare 4 models on 5 datasets and get $p = 0.60$. Does this mean all models are equivalent?

**Answer**: No. Failure to reject $H_0$ does not prove $H_0$. With only $n = 5$ datasets and $k = 4$ models, the Friedman test has very low power. Even if one model is substantially better, the test might not detect it.

To assess this, compute the required number of datasets for adequate power. For a large effect size ($W = 0.5$ in Kendall's $W$ formulation), $\alpha = 0.05$, power = 0.80, and $k = 4$, you need approximately $n \geq 15-20$ datasets. With only 5 datasets, the test is severely underpowered.

**Recommendation**: Report the average ranks and the effect size (Kendall's $W$), acknowledging the limited power. State that "no statistically significant difference was detected, but the study was underpowered to detect effects smaller than [threshold]." If possible, add more benchmark datasets.

### Q17: A colleague applies the Benjamini-Hochberg procedure and finds that 3 out of 20 comparisons are "significant." They claim the FDR is controlled at 5%. But in practice, 2 of those 3 turn out to be false positives (67% false discovery rate). Is the BH procedure broken?

**Answer**: No. The BH procedure controls the **expected** FDR at 5%, not the realized FDR in any single experiment. In expectation, across many experiments using the same procedure, the average fraction of false discoveries among rejections will be at most 5%. In any individual experiment, the realized FDR can be much higher or much lower.

Think of it like a coin with $P(\text{heads}) = 0.5$. You might flip 3 coins and get 3 heads (100% heads), but that doesn't mean the coin is biased. With many repetitions, the proportion of heads converges to 50%.

In this specific case, with only 3 rejections, the variance of the realized FDR is high. The BH procedure is working as designed—it provides a long-run guarantee, not a per-experiment guarantee.

### Q18: You are using the Bayesian correlated t-test and set $\rho = 1/k = 0.1$ for 10-fold CV. How sensitive are your conclusions to this choice?

**Answer**: This is an important sensitivity question. The value $\rho = 1/k$ is an approximation motivated by the fact that each fold's test set constitutes $1/k$ of the data. However, the true correlation depends on the learning algorithm, the dataset, and the metric.

**Sensitivity analysis**: Re-run the Bayesian test with several values of $\rho$: 0, 0.05, 0.1, 0.15, 0.2. If the conclusions (e.g., $P(\text{Model B better}) > 0.95$) are stable across this range, the choice of $\rho$ does not materially affect the result. If the conclusions change (e.g., from $P = 0.97$ at $\rho = 0.05$ to $P = 0.78$ at $\rho = 0.2$), report the range and acknowledge the sensitivity.

As $\rho$ increases, $n_{\text{eff}}$ decreases, the posterior becomes wider, and conclusions become less certain. Overestimating $\rho$ is conservative (wider posteriors), while underestimating $\rho$ is anti-conservative (overconfident).

### Q19: Your team compared 5 models, found a significant Friedman test, but the Nemenyi post-hoc test finds no significant pairwise differences. How is this possible?

**Answer**: This is a well-known phenomenon. The Friedman test is an omnibus test with the alternative hypothesis "not all models are equal." It can detect a **pattern** of rank differences across all models that, collectively, is unlikely under $H_0$. But the Nemenyi test is more conservative: it controls the family-wise error rate across all $\binom{k}{2}$ pairwise comparisons, which requires each individual comparison to clear a higher bar.

Analogy: ANOVA can detect that group means are not all equal, but Tukey's post-hoc might not identify which specific pairs differ.

**Practical solutions**:
1. Use a less conservative post-hoc test. If you have a designated control model, use the Bonferroni-Dunn test, which only makes $k - 1$ comparisons instead of $\binom{k}{2}$, giving more power.
2. Increase the number of datasets ($n$). The critical difference shrinks as $n$ grows.
3. Report the average ranks and effect sizes even without formal significance, noting that the omnibus test detected a difference but post-hoc power was insufficient.

### Q20: You use the 5×2 CV test and get $p = 0.09$. Your manager says "that's close to 0.05, so the new model is probably better—let's deploy it." How do you respond?

**Answer**: I would explain several issues with this reasoning:

**First**, the 5×2 CV test is already somewhat conservative (it has lower power than the uncorrected paired t-test), so $p = 0.09$ is genuinely non-significant. If we had used a test with inflated Type I error, we might have gotten $p < 0.05$, but that would have been misleading.

**Second**, treating $p = 0.09$ as "almost significant" is a form of wishful thinking. The p-value is a continuous measure, but the decision threshold ($\alpha = 0.05$) was set before the experiment. Moving the goalposts after seeing the result invalidates the frequentist framework.

**Third**, I would reframe the conversation around practical significance. What is the expected business impact of the performance difference? If we compute the mean difference and its confidence interval, does the interval contain values that would justify the engineering cost of deployment?

**Fourth**, I would suggest a Bayesian analysis. Compute $P(\text{new model better} \mid \text{data})$ and $P(\text{difference exceeds business-relevant threshold} \mid \text{data})$. This gives the manager the direct probability they are looking for, rather than a p-value.

**Fifth**, if the decision is high-stakes, I would propose collecting more data or running a proper online A/B test before committing to deployment.

---

## Follow-Up and Probing Questions

### Q21: "You mentioned the Nadeau-Bengio correction uses $\rho \approx n_{\text{test}}/n_{\text{train}}$. Where does this approximation come from?"

**Answer**: Nadeau and Bengio (2003) derived this result in the context of comparing estimators of prediction error. The key insight is that the correlation between two performance estimates from overlapping training sets depends on the ratio of test-set size to training-set size.

The formal derivation considers two random train-test splits that share a fraction of their training data. For linear models, the performance estimate on the test set can be decomposed into contributions from each training example. When two training sets overlap, the shared examples contribute to both estimates in the same way, creating correlation. The degree of overlap is controlled by the split ratio: with $n_{\text{test}}/n_{\text{train}} \to 0$ (very small test sets, very large training sets), there is maximal overlap and maximal correlation. With $n_{\text{test}}/n_{\text{train}} \to \infty$ (very large test sets), the training sets are tiny and barely overlap.

For non-linear models, this is an approximation, but empirical studies have shown it works reasonably well in practice. The approximation tends to be slightly conservative (overestimates correlation), which is safer than the alternative.

### Q22: "What happens to the Friedman test when two models are tied on a dataset?"

**Answer**: When models have identical performance on a dataset, they receive **average ranks**. For example, if models B and C both achieve 0.85 accuracy (tied for 2nd and 3rd place), each receives rank $(2 + 3)/2 = 2.5$.

Ties reduce the power of the Friedman test because they compress the rank differences. With many ties, the Friedman statistic should be corrected using:

$$\chi_F^2 = \frac{\frac{12}{nk(k+1)}\sum_{i=1}^{k}\bar{R}_i^2 - 3n(k+1)}{1 - \frac{1}{nk(k^2-1)}\sum_{j=1}^{n}\sum_{l=1}^{g_j}(t_{jl}^3 - t_{jl})}$$

where $g_j$ is the number of tied groups in block $j$ and $t_{jl}$ is the size of tied group $l$ in block $j$. The denominator shrinks the statistic when there are many ties, maintaining the correct Type I error rate.

### Q23: "Can you use parametric bootstrap instead of the tests we discussed?"

**Answer**: Yes, and parametric bootstrap is increasingly popular in ML. The idea: assume the performance differences follow some distribution (e.g., normal), estimate its parameters from the data, simulate many "virtual" experiments by drawing from the fitted distribution, and compute the fraction of virtual experiments where the test statistic exceeds the observed value.

**Advantages**: The bootstrap can capture complex dependency structures (e.g., between CV folds) by simulating correlated observations. It does not rely on small-sample distributional assumptions ($t$-distribution, $\chi^2$ approximation). It is flexible—you can construct bootstrap confidence intervals for any statistic (mean, median, quantile of differences).

**Disadvantages**: Computationally expensive (thousands of resamples, each potentially requiring retraining). The parametric assumption (e.g., normality) may be wrong. The non-parametric bootstrap (resampling with replacement) has its own issues: the bootstrap distribution of a $t$-statistic does not exactly match the $t$-distribution for small $n$, and resampling CV folds is tricky because the folds are not independent observations.

**Practical recommendation**: For large test sets where retraining is not needed (just resample predictions), the bootstrap is excellent. For comparing models where retraining is involved, the 5×2 CV test or Bayesian methods are more practical.

### Q24: "If Bayesian methods are so much better than frequentist tests, why do people still use frequentist tests?"

**Answer**: Bayesian methods are not strictly "better"—each framework has strengths.

**Reasons frequentist tests persist**: (1) Simplicity and tradition—p-values and t-tests are deeply ingrained in scientific training and publication standards. (2) Objectivity—frequentist methods do not require prior specification, which can be contentious in adversarial settings (e.g., regulatory submissions). (3) Computational convenience—a paired t-test is a one-line calculation; Bayesian computation may require MCMC or numerical integration. (4) Frequentist guarantees are well-defined—controlling the Type I error rate at exactly $\alpha$ over repeated use is a strong, simple-to-understand property.

**Where Bayesian methods genuinely shine**: (1) Small samples—when you have only 5-10 paired observations, the posterior naturally expresses the large remaining uncertainty. (2) The three-way conclusion—Bayesian methods can say "practically equivalent," which frequentist tests cannot. (3) Incorporating prior knowledge—if you know from previous benchmarks that the performance difference is likely small, you can encode this as a prior and get more precise estimates. (4) Decision-theoretic framing—Bayesian posteriors integrate naturally with cost-benefit analyses.

In practice, the best approach is often to report both: a frequentist test for comparability with prior work, and a Bayesian analysis for richer interpretation.

### Q25: "Walk me through what happens to Type I error and power as you change each of the four linked quantities."

**Answer**:

**Increasing $\alpha$ (significance level)**: Type I error increases (you accept more false positives). Power increases (you detect more real effects). Trade-off: liberal testing.

**Increasing $n$ (sample size)**: Type I error stays at $\alpha$ (controlled by design). Power increases (the test statistic's distribution under $H_1$ shifts further from the null distribution). No downside except computational cost.

**Increasing the effect size $\delta$**: Type I error stays at $\alpha$. Power increases (larger effects are easier to detect). This is not under the experimenter's control—it is a property of the models and data.

**Increasing power $1 - \beta$ (by adjusting $n$ or $\alpha$)**: If by increasing $n$: Type I error unchanged. If by increasing $\alpha$: Type I error increases. Power and Type I error are both decreasing functions of the critical value; making the test less conservative (lower critical value) increases both.

**The mathematical relationship** (for a one-sided $z$-test):

$$\text{Power} = \Phi\left(\frac{\delta\sqrt{n}}{\sigma} - z_{1-\alpha}\right)$$

where $\Phi$ is the standard normal CDF. This shows that power increases with $\delta$ (larger effect), $n$ (more data), and $\alpha$ (less stringent threshold), and decreases with $\sigma$ (more noise).

For a two-sided test, replace $z_{1-\alpha}$ with $z_{1-\alpha/2}$.

This formula is the basis for power analysis: given any three of $\{\alpha, 1-\beta, \delta/\sigma, n\}$, solve for the fourth.
