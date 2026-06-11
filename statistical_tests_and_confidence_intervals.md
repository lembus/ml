# Statistical Tests & Confidence Intervals — A Complete Learning Guide

---

## Part I: Hypothesis Testing

### 1. Motivation & Intuition

Imagine you ship a new ranking model. The old model has a click-through rate (CTR) of 4.0%. You run an A/B test for a week and the new model shows 4.2%. Is it actually better, or did you just get lucky with which users landed in which bucket?

This is the central question hypothesis testing answers: **given noisy data, can we distinguish a real effect from random fluctuation?** Without a principled framework, every team would invent its own threshold ("looks like a big jump to me"), and most "improvements" shipped to production would be noise.

Real ML applications:
- **A/B testing** model versions, UI changes, ranking algorithms.
- **Drift detection**: is today's input distribution different from training's?
- **Feature selection**: does this feature carry signal beyond chance?
- **Fairness audits**: do error rates differ across demographic groups by more than sampling variation?
- **Benchmark comparisons**: is model A *really* better than model B on this eval set?

The intuition: assume nothing is happening (the "null world"), compute how surprising the observed data would be in that world, and reject the null only if the data is sufficiently improbable under it.

### 2. Conceptual Foundations

**Null hypothesis ($H_0$)**: the default, "boring" claim — no effect, no difference, parameter equals some specific value. Example: $\mu_{\text{new}} = \mu_{\text{old}}$.

**Alternative hypothesis ($H_1$ or $H_a$)**: what we're trying to provide evidence for. Can be two-sided ($\mu_{\text{new}} \neq \mu_{\text{old}}$) or one-sided ($\mu_{\text{new}} > \mu_{\text{old}}$).

**Test statistic**: a function of the data whose distribution under $H_0$ is known (or approximable). Examples: t-statistic, z-statistic, chi-square statistic.

**p-value**: the probability, *assuming $H_0$ is true*, of observing a test statistic at least as extreme as the one observed. It is **not** the probability that $H_0$ is true. It is a statement about data given a hypothesis, not a hypothesis given data.

**Significance level ($\alpha$)**: a pre-chosen threshold (commonly 0.05) controlling the Type I error rate. If $p < \alpha$, reject $H_0$.

**Type I error (false positive)**: rejecting $H_0$ when it is true. Probability = $\alpha$.

**Type II error (false negative)**: failing to reject $H_0$ when $H_1$ is true. Probability = $\beta$.

**Power**: $1 - \beta$, the probability of correctly rejecting a false $H_0$. Power depends on effect size, sample size, $\alpha$, and noise.

**Effect size**: the magnitude of the difference (e.g., Cohen's $d = (\mu_1 - \mu_2)/\sigma$). Statistical significance ≠ practical significance: with enough data, trivially small effects become "significant."

**Underlying assumptions** (typical):
- Independent, identically distributed (i.i.d.) observations.
- For parametric tests: a specific distributional form (often normality, or normality of the sampling distribution via CLT).
- Known or correctly estimated variance structure.
- Correctly specified hypotheses *before* looking at the data.

**What breaks when assumptions fail:**
- Dependent samples (e.g., repeated measurements on the same user) → variance underestimated → false positives inflate.
- Heavy-tailed distributions with small $n$ → CLT hasn't kicked in → t-test p-values unreliable.
- Peeking at data and re-deciding hypotheses → "garden of forking paths," nominal $\alpha$ no longer holds.
- Optional stopping (checking p-values daily and stopping when significant) → Type I error can exceed 50%.

### 3. Mathematical Formulation

**General recipe.** Pick a test statistic $T(X)$ with a known distribution $F_0$ under $H_0$. Define a rejection region $R$ such that $\Pr_{H_0}(T \in R) = \alpha$. Reject $H_0$ iff observed $T \in R$.

The **p-value** for a two-sided test is $p = \Pr_{H_0}(|T| \geq |t_{\text{obs}}|)$.

#### One-sample z-test (variance known)

Data $X_1, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$, $\sigma$ known. Test $H_0: \mu = \mu_0$.

$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim \mathcal{N}(0,1) \text{ under } H_0.$$

Derivation: $\bar{X} \sim \mathcal{N}(\mu_0, \sigma^2/n)$ under $H_0$, standardize.

#### One-sample t-test (variance unknown)

Replace $\sigma$ with the sample standard deviation $s$:

$$t = \frac{\bar{X} - \mu_0}{s/\sqrt{n}} \sim t_{n-1} \text{ under } H_0.$$

The extra uncertainty from estimating $\sigma$ produces heavier tails — Student's t with $n-1$ degrees of freedom. As $n \to \infty$, $t_{n-1} \to \mathcal{N}(0,1)$.

#### Two-sample t-test (Welch's, unequal variances)

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}, \quad \text{df} \approx \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$

The Welch-Satterthwaite degrees of freedom approximate the variance of the denominator.

#### Chi-square test (goodness of fit / independence)

For observed counts $O_i$ and expected counts $E_i$ across $k$ cells:

$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k - 1 - p}$$

where $p$ is the number of estimated parameters. Intuition: each term measures squared deviation normalized by expected variance (under multinomial, $\text{Var}(O_i) \approx E_i$ when $E_i$ is small relative to $n$).

#### ANOVA / F-test

For $k$ groups, test $H_0: \mu_1 = \mu_2 = \dots = \mu_k$.

$$F = \frac{\text{MS}_{\text{between}}}{\text{MS}_{\text{within}}} = \frac{\frac{1}{k-1}\sum_j n_j (\bar{X}_j - \bar{X})^2}{\frac{1}{N-k}\sum_j \sum_i (X_{ij} - \bar{X}_j)^2}$$

Under $H_0$ with normality and equal variances, $F \sim F_{k-1, N-k}$. Intuition: ratio of "variation explained by group means" to "variation within groups." If groups have the same mean, both should estimate the same $\sigma^2$ and the ratio should hover around 1.

#### Power analysis

For a one-sample z-test with true mean $\mu_1$ and $H_0: \mu = \mu_0$:

$$\text{Power} = \Phi\left(\frac{|\mu_1 - \mu_0|\sqrt{n}}{\sigma} - z_{1-\alpha/2}\right).$$

To detect effect $\delta$ with power $1-\beta$ at level $\alpha$:

$$n \approx \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{\delta^2}.$$

Sample size scales as $1/\delta^2$ — halving the detectable effect quadruples required samples.

### 4. Worked Example

**A/B test for CTR.** Control: 50,000 users, 2,000 clicks (CTR = 4.00%). Treatment: 50,000 users, 2,150 clicks (CTR = 4.30%).

Model each click as Bernoulli. Two-proportion z-test:

$$\hat{p}_C = 0.0400, \quad \hat{p}_T = 0.0430, \quad \hat{p} = \frac{2000+2150}{100000} = 0.0415.$$

Pooled standard error:

$$\text{SE} = \sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_C}+\frac{1}{n_T}\right)} = \sqrt{0.0415 \cdot 0.9585 \cdot \frac{2}{50000}} \approx 0.001262.$$

Test statistic:

$$z = \frac{0.0430 - 0.0400}{0.001262} \approx 2.378.$$

Two-sided p-value: $2(1 - \Phi(2.378)) \approx 0.0174$. Since $0.0174 < 0.05$, reject $H_0$.

**Interpretation:** if there were truly no difference between control and treatment, we would see a CTR gap this large or larger only ~1.7% of the time. The observed lift is 0.30 percentage points — small but real, given the sample size. Whether to ship is a *product* decision (does 0.3pp justify launch costs?), not a statistical one.

### 5. Relevance to ML Practice

- **Online experimentation**: every reputable A/B testing platform (Optimizely, internal experimentation tools at Google/Meta/Netflix) is built on hypothesis testing, with corrections for sequential analysis (always-valid p-values, mSPRT) to handle peeking.
- **Model comparison on benchmarks**: McNemar's test for paired classifier predictions, paired t-test on cross-validation folds.
- **Drift detection**: KS test on feature distributions, chi-square on categorical features, between training and serving.
- **Feature significance**: classical regression t-tests on coefficients (with caveats about multiple comparisons and post-selection inference).

**When to use**: when you need to make a yes/no decision under uncertainty and the cost of false positives is meaningful.

**When not to use**: exploratory analysis (use estimation and visualization instead); huge-$n$ regimes where everything is "significant" (focus on effect sizes and confidence intervals); when assumptions are wildly violated (use non-parametric or bootstrap).

**Trade-offs**: parametric tests are powerful when assumptions hold but brittle when they don't. Non-parametric tests are robust but less powerful. Bayesian alternatives give probability statements about hypotheses but require priors and more compute.

### 6. Common Pitfalls & Misconceptions

- **"p-value is the probability $H_0$ is true."** No. It's $\Pr(\text{data} \mid H_0)$, not $\Pr(H_0 \mid \text{data})$.
- **"Failing to reject means $H_0$ is true."** Absence of evidence ≠ evidence of absence. You may simply have been underpowered.
- **Peeking and optional stopping** without correction inflates Type I error dramatically.
- **HARKing** (Hypothesizing After Results are Known): forming hypotheses after seeing data destroys the validity of p-values.
- **Multiple testing without correction**: testing 20 features at $\alpha=0.05$ produces about one false positive on average even when nothing is real.
- **Statistical vs. practical significance**: with $n=10^7$, a 0.001% lift can be "significant" yet useless.
- **Using a t-test on heavily skewed data with small $n$**: CLT hasn't applied; results are unreliable.
- **Ignoring dependence**: clustered data (multiple events per user) require cluster-robust SEs or randomization at the cluster level.

---

## Part II: Confidence Intervals

### 1. Motivation & Intuition

A point estimate ($\hat{\mu} = 4.30\%$) is a number. A confidence interval is a *range of plausible values* that honestly communicates uncertainty. Reporting only the point estimate to a stakeholder is misleading: it hides whether the answer is "definitely around 4.3%" or "could be anywhere from 2% to 6%."

In ML: every metric on a held-out test set (accuracy, AUC, calibration error) is a noisy estimate. CIs let you say "model A's accuracy is 91% ± 1.2%" rather than pretending you know it to four decimal places.

### 2. Conceptual Foundations

**Frequentist confidence interval**: a procedure that, *if repeated across many datasets*, contains the true parameter at least $(1-\alpha) \cdot 100\%$ of the time. This is **coverage probability**. The interval itself, once computed, either contains the true value or doesn't — there is no "95% chance" the true value is in *this specific* interval. The 95% refers to the procedure, not the realization.

**Bayesian credible interval**: a range with $(1-\alpha)$ posterior probability of containing the parameter, *given the observed data and prior*. This *is* a probability statement about the parameter. Different philosophy, often numerically similar with weak priors.

**Bootstrap CI**: instead of relying on theoretical sampling distributions, resample the data with replacement, recompute the statistic, and use the resulting empirical distribution.

**Assumptions**:
- Frequentist normal-theory CIs: i.i.d. data, correctly specified model, large enough $n$ (or true normality).
- Bootstrap: sample is representative; estimator is sufficiently smooth (fails for extremes like the maximum).
- Bayesian: prior is reasonable; posterior is computable.

### 3. Mathematical Formulation

**Wald (normal-approximation) CI for the mean**:

$$\bar{X} \pm z_{1-\alpha/2} \cdot \frac{s}{\sqrt{n}}.$$

Derivation: from $\Pr\!\left(-z_{1-\alpha/2} \leq \frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \leq z_{1-\alpha/2}\right) = 1 - \alpha$, solve for $\mu$.

**t-based CI** (small $n$, unknown variance):

$$\bar{X} \pm t_{n-1, 1-\alpha/2} \cdot \frac{s}{\sqrt{n}}.$$

**Wilson interval for a proportion** (more accurate than Wald near 0 or 1):

$$\frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}.$$

**Bayesian credible interval**: posterior $p(\theta \mid x) \propto p(x \mid \theta)\pi(\theta)$, then take quantiles or HPD region. For Beta-Binomial with prior $\text{Beta}(\alpha_0, \beta_0)$ and $k$ successes in $n$ trials, posterior is $\text{Beta}(\alpha_0 + k, \beta_0 + n - k)$ and the 95% credible interval is the 2.5%–97.5% quantiles of this Beta.

**Bootstrap percentile CI**: for $B$ resamples giving $\hat{\theta}^{*(1)}, \dots, \hat{\theta}^{*(B)}$, the interval is $[\hat{\theta}^{*}_{(\alpha/2)}, \hat{\theta}^{*}_{(1-\alpha/2)}]$. Bias-corrected and accelerated (BCa) versions adjust for bias and skewness in the bootstrap distribution.

### 4. Worked Example

Test set with 1,000 examples, 920 correct. Accuracy $\hat{p} = 0.92$.

**Wald 95% CI**: $0.92 \pm 1.96\sqrt{0.92 \cdot 0.08 / 1000} = 0.92 \pm 0.0168 = [0.903, 0.937]$.

**Wilson 95% CI**: numerator center $0.92 + 1.96^2/2000 = 0.9219$; denominator $1 + 1.96^2/1000 = 1.00384$; half-width term $\approx 0.01686$. Interval $\approx [0.902, 0.935]$. Very close to Wald here because $\hat{p}$ is not extreme.

**Bayesian (uniform Beta(1,1) prior)**: posterior is $\text{Beta}(921, 81)$. 2.5%–97.5% quantiles ≈ $[0.902, 0.936]$. Numerically nearly identical, but the *interpretation* is "given the data and prior, there's a 95% probability the true accuracy lies in this interval."

**Bootstrap**: resample the 1000 predictions with replacement 10,000 times, compute accuracy each time, take the 2.5th and 97.5th percentiles. Will land near $[0.903, 0.937]$.

### 5. Relevance to ML Practice

- **Reporting metrics**: never report accuracy/AUC/F1 without uncertainty. CIs (often bootstrap on the test set) prevent over-claiming.
- **Model comparison**: overlapping CIs for two models' accuracies are weak evidence against a difference, but non-overlap is strong evidence for one. The proper comparison is a CI on the *difference*.
- **Calibration**: estimating ECE with bootstrap CIs.
- **Data labeling**: agreement statistics with CIs.
- **Online experiments**: CIs on lift, often computed as the ship/no-ship summary.

**Bootstrap is preferred** when (a) the estimator's sampling distribution is unknown or asymmetric (AUC, median, quantile, gini), (b) the test set is moderate size, (c) you want a uniform recipe across metrics. **Don't bootstrap** when observations are heavily dependent (use block bootstrap), when extrapolating beyond observed data (extreme quantiles), or when the test set is tiny.

### 6. Common Pitfalls & Misconceptions

- **"95% probability the true value is in [a,b]"** — frequentist CIs do not say this. Switch to Bayesian if you want that statement.
- **Wald intervals for extreme proportions** can extend below 0 or above 1 and have poor coverage; use Wilson, Jeffreys, or Clopper-Pearson.
- **Bootstrapping dependent data** as if i.i.d. underestimates variance; use block bootstrap for time series.
- **CIs from selected models** (after feature selection) are too narrow — selection inflates apparent precision.
- **Confusing standard error with standard deviation**: SE is the SD of the sampling distribution of the estimator, not of the data.

---

## Part III: Multiple Testing Correction

### 1. Motivation & Intuition

If you test 20 hypotheses at $\alpha = 0.05$ and none are real, the expected number of false positives is $20 \times 0.05 = 1$. The probability of *at least one* false positive (the family-wise error rate, FWER) is $1 - 0.95^{20} \approx 0.64$. Run a feature significance test on 1,000 features and you'll find ~50 "significant" by chance alone.

Multiple testing correction adjusts thresholds so that the *family-wide* error rate is controlled, not just the per-test rate.

### 2 & 3. Concepts and Mathematical Formulation

**Family-Wise Error Rate (FWER)**: $\Pr(\text{at least one false positive})$.

**False Discovery Rate (FDR)**: expected proportion of false positives among rejections, $E[V/R \mid R > 0] \cdot \Pr(R>0)$ where $V$ = false rejections, $R$ = total rejections.

**Bonferroni**: reject $H_i$ iff $p_i < \alpha/m$ for $m$ tests. Controls FWER under any dependence. Conservative — power drops as $m$ grows.

**Holm-Bonferroni** (uniformly more powerful than Bonferroni, still controls FWER): sort p-values $p_{(1)} \leq \dots \leq p_{(m)}$. Find smallest $k$ such that $p_{(k)} > \alpha/(m - k + 1)$; reject $H_{(1)}, \dots, H_{(k-1)}$.

**Benjamini-Hochberg (BH)** for FDR: sort p-values, find largest $k$ such that $p_{(k)} \leq \frac{k}{m}\alpha$, reject the $k$ smallest. Controls FDR at level $\alpha$ under independence and certain positive dependence.

Bonferroni's intuition: union bound, $\Pr(\bigcup A_i) \leq \sum \Pr(A_i) = m \cdot \alpha/m = \alpha$.

### 4. Worked Example

Five p-values: 0.001, 0.008, 0.039, 0.041, 0.42. $\alpha = 0.05$, $m = 5$.

- **Bonferroni**: threshold $0.05/5 = 0.01$. Reject those with $p < 0.01$: tests 1 and 2.
- **Holm**: $p_{(1)} = 0.001 < 0.05/5 = 0.01$ → reject. $p_{(2)} = 0.008 < 0.05/4 = 0.0125$ → reject. $p_{(3)} = 0.039 < 0.05/3 = 0.0167$? No → stop. Reject 1, 2.
- **BH**: thresholds $\{0.01, 0.02, 0.03, 0.04, 0.05\}$. Largest $k$ with $p_{(k)} \leq k\alpha/m$: $p_{(4)} = 0.041 \leq 0.04$? No. $p_{(3)} = 0.039 \leq 0.03$? No. $p_{(2)} = 0.008 \leq 0.02$? Yes. Reject 1, 2. (Here BH happens to match Holm; with more tests in the borderline range BH typically rejects more.)

### 5. Relevance to ML Practice

Feature importance testing, multi-arm bandits / multi-variant experiments, fairness audits across many demographic slices, multiple hyperparameter comparisons, GWAS-style screening. Choose **FWER control** when any false positive is costly (e.g., regulatory claims). Choose **FDR control** when you can tolerate a controlled fraction of false discoveries among many (exploratory feature screening).

### 6. Pitfalls

- Forgetting that *every comparison made counts*, including ones not reported.
- Applying corrections only to "significant" results (post-hoc cherry-picking).
- Assuming BH controls FWER (it doesn't, and that's the point).

---

## Part IV: Non-parametric Tests

### 1. Motivation & Intuition

When you can't assume normality, equal variances, or any specific distribution, non-parametric tests use ranks or empirical distributions instead of raw values. They trade some statistical power for robustness.

### 2 & 3. Specific Tests

**Mann-Whitney U** (a.k.a. Wilcoxon rank-sum): tests whether two independent samples come from the same distribution (or more precisely, whether $\Pr(X_1 > X_2) = 1/2$). Pool both samples, rank them, sum ranks for one group. Statistic:

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2},$$

where $R_1$ is the rank sum of group 1. Under $H_0$, $U$ has known mean $n_1 n_2/2$ and variance $n_1 n_2 (n_1+n_2+1)/12$, approximately normal for moderate $n$.

**Wilcoxon signed-rank**: paired version. Compute differences $d_i$, rank $|d_i|$, sum signed ranks. Tests whether the median difference is zero. Replaces the paired t-test when normality is dubious.

**Kolmogorov-Smirnov (KS)**: compares two empirical CDFs $F_n$ and $G_m$ (or one empirical CDF to a theoretical one). Statistic:

$$D = \sup_x |F_n(x) - G_m(x)|.$$

Distribution-free under continuous $H_0$. Sensitive to differences anywhere in the distribution, but most powerful for differences near the median.

### 4. Worked Example

Group A: [3, 5, 8, 9, 11]; Group B: [1, 2, 4, 6, 7]. Pool and rank: 1→1, 2→2, 3→3, 4→4, 5→5, 6→6, 7→7, 8→8, 9→9, 11→10. Rank sum for A: 3+5+8+9+10 = 35. $U_A = 35 - 5\cdot6/2 = 20$. Mean under $H_0$: $25/2 = 12.5$. SD: $\sqrt{5 \cdot 5 \cdot 11 / 12} \approx 4.79$. $z \approx (20-12.5)/4.79 \approx 1.57$, two-sided $p \approx 0.12$. Not significant at 5% with this tiny sample.

### 5. Relevance to ML Practice

- **Latency comparisons**: response times are heavily right-skewed; use Mann-Whitney instead of t-test, or compare medians/quantiles.
- **Distribution drift**: KS test on a feature between training and serving.
- **Robust evaluation**: comparing model errors that are skewed or heavy-tailed.

**Don't use** when you have rich distributional knowledge and parametric assumptions hold (you'll lose power). **Avoid KS for discrete data** — its theoretical distribution assumes continuity; use chi-square instead.

### 6. Pitfalls

- Mann-Whitney does *not* test for equal means in general; it tests stochastic equality. With unequal variances or shapes, "significant" doesn't mean "different means."
- KS is insensitive to tail differences relative to its sensitivity at the median; for tail-focused drift, use Anderson-Darling.
- Many ties degrade rank-based tests; correct variance formulas exist but are often forgotten.

---

# Interview Preparation

## Foundational

**Q1. What is a p-value, in plain language?**
The probability of observing data at least as extreme as what was observed, assuming the null hypothesis is true. It is *not* the probability that the null is true, nor the probability that results are due to chance in any direct sense.

**Q2. Difference between Type I and Type II errors?**
Type I = rejecting a true null (false positive), rate $\alpha$. Type II = failing to reject a false null (false negative), rate $\beta$. Power = $1-\beta$. The two trade off: lowering $\alpha$ raises $\beta$ for fixed $n$ and effect size.

**Q3. What does "95% confidence" mean for a frequentist CI?**
If you repeated the experiment many times and constructed a CI each time using the same procedure, ~95% of those intervals would contain the true parameter. It says nothing probabilistic about the specific computed interval.

**Q4. When would you use a t-test vs z-test?**
z-test when population variance is known (rare) or $n$ is huge so $s \approx \sigma$. t-test when variance must be estimated and $n$ is small or moderate. For $n > 30$ they are nearly identical.

**Q5. Why do we need multiple testing correction?**
Per-test error rates compound. Testing $m$ independent hypotheses at $\alpha$ gives FWER $\approx 1-(1-\alpha)^m$, which approaches 1 quickly. Without correction, "discoveries" are dominated by noise.

## Mathematical

**Q6. Derive the standard error of a sample mean.**
$\text{Var}(\bar{X}) = \text{Var}(\frac{1}{n}\sum X_i) = \frac{1}{n^2}\sum \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \sigma^2/n$, by independence. SE = $\sigma/\sqrt{n}$.

**Q7. Why does the t-distribution have heavier tails than the normal?**
Because $s$ is itself a random variable. The denominator $s/\sqrt{n}$ can be unusually small, producing extreme t-values more often than a fixed $\sigma$ would. As $n \to \infty$, $s \to \sigma$ a.s., and $t_{n-1} \to \mathcal{N}(0,1)$.

**Q8. Derive the sample size for detecting effect $\delta$ with power $1-\beta$ at level $\alpha$ (two-sided).**
Under $H_0$, reject if $|Z| > z_{1-\alpha/2}$. Under $H_1$ with mean $\delta$, $Z \sim \mathcal{N}(\delta\sqrt{n}/\sigma, 1)$. Power = $\Pr(Z > z_{1-\alpha/2} \mid H_1) \approx \Phi(\delta\sqrt{n}/\sigma - z_{1-\alpha/2})$. Set this equal to $1-\beta$: $\delta\sqrt{n}/\sigma - z_{1-\alpha/2} = z_{1-\beta}$, so $n = (z_{1-\alpha/2}+z_{1-\beta})^2 \sigma^2 / \delta^2$.

**Q9. Why does Bonferroni control FWER under arbitrary dependence?**
Union bound: $\Pr(\bigcup_{i=1}^m \{p_i < \alpha/m\} \mid H_0^{\text{all}}) \leq \sum_i \Pr(p_i < \alpha/m) = m \cdot \alpha/m = \alpha$. No independence assumption needed.

**Q10. Show that the Wald CI for a proportion can fail.**
Wald: $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$. If $\hat{p}=0$, the interval collapses to $[0,0]$, giving 0% coverage of true positive $p$. Wilson and Jeffreys avoid this by using an effective count or a prior.

## Applied

**Q11. You ran an A/B test for a week, p=0.06. PM says "almost significant, can we ship?"**
No. The threshold was set in advance for a reason. Options: (a) report the lift with a CI and let the PM make a value-based call ("we estimate +0.2% ± 0.25%"), (b) collect more data with a pre-registered plan (and account for sequential testing), (c) examine practical significance and downside risk. Don't move the goalposts.

**Q12. You're comparing 50 features for predictive value. How do you handle multiple testing?**
Decide between FWER (Holm if any false positive is costly — e.g., feature will be deployed in a regulated setting) and FDR (BH if you're screening and willing to follow up the survivors). Report both adjusted and raw p-values. Watch for correlated features — raw FDR control assumes positive dependence, often okay but verify.

**Q13. Test set accuracy: model A 91.2%, model B 91.6%. Is B better?**
Need a CI on the *difference*, ideally on the same test set with paired comparison (McNemar's test for paired binary outcomes, or paired bootstrap on per-example correctness). Reporting marginal CIs and checking overlap is the wrong analysis and is too conservative.

**Q14. Latency dropped from 120ms to 110ms median in an experiment. t-test gives p=0.4. Why?**
Latency is heavy-tailed; means are dominated by tail outliers, and the variance is huge. A t-test on means is the wrong tool. Use Mann-Whitney on raw values, or a quantile regression / bootstrap on the median, or test specific quantiles (p50, p95) with appropriate methods.

**Q15. How would you build a CI for AUC?**
DeLong's method gives a closed-form CI accounting for the U-statistic structure. Bootstrap resampling (stratified by class, since class imbalance matters) is a flexible alternative. Avoid assuming normality of AUC directly.

## Debugging & Failure Modes

**Q16. An experiment shows a "highly significant" 0.01% lift at $n = 10^8$. Should you ship?**
Statistical significance is trivial at that scale. Compute the CI; if it's tight around an effect that doesn't matter for the business or for users, do not ship. Significance is necessary, not sufficient.

**Q17. Your daily-monitored A/B test "became significant" on day 4 and you stopped. What's wrong?**
Optional stopping inflates Type I error far above nominal $\alpha$ (often 2–5× or more). Use group-sequential designs (O'Brien-Fleming, Pocock), always-valid p-values (mSPRT), or commit to a fixed horizon.

**Q18. KS test says training and serving distributions differ (p < 0.001) but model performance is fine. What happened?**
With huge $n$, KS detects trivial differences. Statistical drift ≠ harmful drift. Look at effect size (the actual KS statistic), feature importance, and downstream metric impact instead of just the p-value.

**Q19. CI on accuracy is [0.88, 0.94], but on a slice of 50 examples it's [0.60, 0.95]. Why so wide?**
SE scales as $1/\sqrt{n}$. With $n=50$, the SE is roughly 4.5× larger than with $n=1000$. Slice-level CIs need either larger slices or hierarchical modeling that pools information across slices.

**Q20. Feature selection by p-value gave 30 features. Coefficient CIs from the final regression are very tight. Are they trustworthy?**
No. Post-selection inference is biased: selecting on significance produces over-confident intervals downstream. Use sample splitting, the lasso with selective inference, or stability selection.

## Probing Follow-ups

**Q21. "Why not just always use the bootstrap?"**
Bootstrap fails for: (a) extreme order statistics (max, min) — the bootstrap distribution is degenerate; (b) heavily dependent data unless block bootstrap; (c) very small samples where the empirical distribution misses tail behavior; (d) estimators with discontinuities. It's also computationally expensive at scale.

**Q22. "Frequentist vs Bayesian CIs — when does the difference matter?"**
With weak priors and lots of data, they coincide numerically. They diverge meaningfully when (a) priors carry real information (small $n$, regularization), (b) parameter is constrained (e.g., variance > 0; Bayesian intervals respect this naturally), (c) you need to make probability statements about the parameter for downstream decisions, (d) hierarchical structure where partial pooling helps.

**Q23. "What's the relationship between a CI and a hypothesis test?"**
Duality: a $(1-\alpha)$ CI is the set of null values that would *not* be rejected at level $\alpha$. Equivalently, $\mu_0$ lies outside the 95% CI iff a two-sided test of $H_0: \mu = \mu_0$ rejects at 5%.

**Q24. "Why is BH valid under positive dependence but not arbitrary dependence?"**
The original BH proof used independence; later work (Benjamini-Yekutieli) showed it holds under "positive regression dependence on a subset" (PRDS). Under arbitrary dependence, you need the BY correction with a $\log m$ factor, which is much more conservative.

**Q25. "If you could only report one number for an experiment result, what would it be?"**
None. Report at minimum a point estimate and an interval. If forced, the lower bound of the CI (or a posterior quantile) is more decision-relevant than a p-value, because it tells you the *worst-case plausible effect* and lets stakeholders weigh that against costs.

---

*End of guide.*
