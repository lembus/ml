# Statistical Tests & Confidence Intervals

## 1. Motivation & Intuition

### Why do we need this?
In the real world, we rarely have access to "perfect" information. If you want to know the average height of all adults in the world, you cannot physically measure everyone. Instead, you measure a small group (a **sample**) and try to infer the truth about the entire world (the **population**).

Because you are guessing based on incomplete data, your guess will always be slightly wrong. Statistical testing provides a framework to quantify **how wrong we might be** and **how confident we should be** in our results.

### Concrete Example: The "Lucky" Classifier
Imagine you train two Machine Learning models, Model A and Model B, to detect spam emails.
* On a test set of 100 emails, Model A gets 82 correct.
* On the same test set, Model B gets 84 correct.

Is Model B better? It *looks* better. But what if Model B just got lucky with a few easy emails? If we ran the test again on a different set of 100 emails, might Model A win?

Without statistics, you might deploy Model B, spending engineering resources to replace Model A, only to find out later it performs exactly the same. Statistical tests help you distinguish **signal** (Model B is genuinely better) from **noise** (random chance in the data selection).

### Connection to Machine Learning
In ML systems, this framework is critical for:
1. **A/B Testing:** Deciding if a new recommendation algorithm actually increases user clicks.
2. **Feature Selection:** Deciding if a specific feature (e.g., "user age") actually correlates with the target label or just appears to by random chance.
3. **Data Monitoring:** Detecting "Data Drift" (e.g., has the distribution of input images changed significantly since we trained the model?).

---

## 2. Conceptual Foundations

### Core Components of Hypothesis Testing

1. **The Null Hypothesis ($H_0$):** The skeptical default position. It usually states that there is "no effect," "no difference," or "no pattern."
   * *Example:* "Model A and Model B have the same accuracy."
2. **The Alternative Hypothesis ($H_1$ or $H_a$):** The claim you are trying to find evidence for.
   * *Example:* "Model B is different from Model A."
3. **The Test Statistic:** A single number calculated from your data that summarizes the difference between what you see and what you would expect if $H_0$ were true.
4. **The P-value:** The probability of observing a test statistic as extreme as (or more extreme than) the one you calculated, **assuming the Null Hypothesis is true**.
   * *Crucial nuance:* A low p-value (e.g., 0.01) means: "If Model A and B were actually identical, seeing this performance gap would happen by luck only 1% of the time." It does **not** mean "There is a 99% chance Model B is better."

### Errors and Power
When making a binary decision (Reject $H_0$ or Fail to Reject $H_0$), there are two ways to be right and two ways to be wrong:

| Decision | $H_0$ is True (No Effect) | $H_0$ is False (Real Effect) |
| :--- | :--- | :--- |
| **Reject $H_0$** | **Type I Error** ($\alpha$) <br> *False Positive* | **Correct** ($1 - \beta$) <br> *Power* |
| **Fail to Reject $H_0$** | **Correct** ($1 - \alpha$) | **Type II Error** ($\beta$) <br> *False Negative* |

* **Significance Level ($\alpha$):** The threshold for p-values (usually 0.05). If $p < \alpha$, we reject $H_0$. This is the probability of a False Positive.
* **Power ($1 - \beta$):** The probability of correctly rejecting the null when a real effect exists. High power means you are likely to detect real improvements.
* **Effect Size:** How big the difference actually is (e.g., 0.1% accuracy gain vs. 10% gain). Large effects are easier to detect (require smaller samples).

### Confidence Intervals (CI)
A point estimate (e.g., "Accuracy is 84%") is risky. A Confidence Interval provides a range (e.g., "Accuracy is 84% $\pm$ 2%").

* **Frequentist Interpretation:** If we repeated this experiment infinite times and calculated a 95% CI for each, 95% of those intervals would contain the true population parameter. (Note: It does *not* strictly mean "there is a 95% chance the true value is in *this specific* interval," though it is often loosely treated that way).

---

## 3. Mathematical Formulation

### The Z-Test (For large samples / known variance)
Used when comparing a sample mean $\bar{x}$ to a population mean $\mu$, given population variance $\sigma^2$ is known or $n$ is large ($>30$).

$$
Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}
$$

* **Numerator ($\bar{x} - \mu_0$):** The signal (observed difference).
* **Denominator ($\sigma / \sqrt{n}$):** The Standard Error (noise). As $n$ grows, noise shrinks, making Z larger for the same difference.

### The T-Test (For small samples / unknown variance)
When $n$ is small or $\sigma$ is unknown, we estimate $\sigma$ using the sample standard deviation $s$. This introduces extra uncertainty, so we use the Student's t-distribution (fatter tails than Normal).

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
$$

### Chi-Square Test ($\chi^2$)
Used for categorical data (e.g., counts of "Click" vs "No Click").

$$
\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}
$$

* $O_i$: Observed count in category $i$.
* $E_i$: Expected count in category $i$ (if $H_0$ were true).
* *Intuition:* If Observed is very different from Expected, $\chi^2$ is large $\rightarrow$ Reject $H_0$.

### ANOVA (Analysis of Variance) & F-Test
Used to compare means across 3+ groups (e.g., Model A vs Model B vs Model C).
Instead of doing many t-tests (which increases Type I error), ANOVA compares the variance **between** groups to the variance **within** groups.

$$
F = \frac{\text{Variance Between Groups}}{\text{Variance Within Groups}}
$$

* If $F$ is large, it means the groups are distinct compared to the noise inside them.

### Confidence Interval Formula
For a mean with unknown $\sigma$ (common case):

$$
CI = \bar{x} \pm t_{\alpha/2, df} \times \frac{s}{\sqrt{n}}
$$

* $\bar{x}$: Point estimate.
* $t_{\alpha/2, df}$: The critical value from t-table (determines width, e.g., 1.96 for Z at 95%).
* $\frac{s}{\sqrt{n}}$: Standard Error.

---

## 4. Worked Example: A/B Testing for Conversion

**Scenario:** You have a current checkout page (Control) and a new design (Treatment). You want to know if the new design increases the conversion rate.

**Data:**
* Control: $N_c = 1000$ users, $X_c = 100$ conversions. ($p_c = 0.10$)
* Treatment: $N_t = 1000$ users, $X_t = 120$ conversions. ($p_t = 0.12$)

**Step 1: Hypotheses**
* $H_0$: $p_t - p_c \le 0$ (New design is worse or equal)
* $H_1$: $p_t - p_c > 0$ (New design is better)

**Step 2: Calculate Pooled Proportion**
Assuming $H_0$ (proportions are same), the best estimate for the common proportion $\hat{p}$:

$$
\hat{p} = \frac{X_c + X_t}{N_c + N_t} = \frac{220}{2000} = 0.11
$$

**Step 3: Standard Error**

$$
SE = \sqrt{ \hat{p}(1-\hat{p}) \left(\frac{1}{N_c} + \frac{1}{N_t}\right) }
$$

$$
SE = \sqrt{ 0.11(0.89) (0.001 + 0.001) } = \sqrt{ 0.0979 \times 0.002 } \approx 0.014
$$

**Step 4: Z-Score**

$$
Z = \frac{p_t - p_c}{SE} = \frac{0.12 - 0.10}{0.014} = \frac{0.02}{0.014} \approx 1.43
$$

**Step 5: P-value**
Using a Z-table, the probability of $Z > 1.43$ is approximately **0.076**.

**Conclusion:**
If we set $\alpha = 0.05$, since $0.076 > 0.05$, we **Fail to Reject the Null Hypothesis**.
*Interpretation:* While 12% looks better than 10%, with sample sizes of 1000, this difference could plausibly happen by random chance 7.6% of the time. We need more data to be sure.

---

## 5. Relevance to Machine Learning Practice

### 1. Model Selection
When comparing two classifiers, you shouldn't just look at accuracy.
* **McNemar’s Test:** Used for paired nominal data (e.g., does Model A get specific examples right that Model B gets wrong?).
* **Paired t-test:** Used for cross-validation results (compare average accuracy of Model A vs B across 10 folds).

### 2. Feature Selection
* **Chi-Square:** Determine if a categorical feature (e.g., "Color") is independent of the target label. If independent ($p > 0.05$), drop the feature.
* **ANOVA:** Determine if a numerical feature (e.g., "Income") has different means across different target classes.

### 3. Drift Detection (ML Monitoring)
In production, we check if the incoming data distribution has changed compared to training data.
* **Kolmogorov-Smirnov (KS) Test:** Non-parametric test to compare two continuous distributions. If the KS statistic is high (low p-value), the data has drifted; retrain the model.

### 4. Multiple Testing Correction
If you test 100 different features to see which correlates with the target, setting $\alpha=0.05$ means you will find ~5 features that appear "significant" purely by chance (Type I error inflation).
* **Bonferroni Correction:** Divide $\alpha$ by the number of tests ($m$). New threshold $\alpha_{new} = \alpha / m$. Very conservative (hard to find real effects).
* **False Discovery Rate (FDR):** Controls the proportion of false positives among the rejected hypotheses. Better for feature selection in high-dimensional data.

---

## 6. Common Pitfalls & Misconceptions

1. **"P=0.05 means 95% chance $H_0$ is false."**
   * **Wrong.** P-value is $P(\text{Data} \mid H_0)$, not $P(H_0 \mid \text{Data})$.
2. **Peeking (P-hacking).**
   * Running an A/B test, checking the p-value every day, and stopping exactly when it dips below 0.05. This drastically inflates the false positive rate. You must fix the sample size *before* starting.
3. **Confusing Statistical Significance with Practical Significance.**
   * With huge datasets (common in ML), tiny differences (e.g., 0.0001% accuracy gain) can be statistically significant (low p-value) but useless for business. Always check Effect Size.
4. **Violating Assumptions.**
   * Using a t-test on highly skewed data (like salaries) or non-independent data (time-series) leads to invalid results. Use Non-parametric tests (Mann-Whitney, Wilcoxon) or Bootstrapping in these cases.

---

# Interview Preparation

## Foundational Questions

**Q1: Explain the difference between Type I and Type II errors in the context of a spam filter.**
* **Answer:**
  * $H_0$: Email is Not Spam. $H_1$: Email is Spam.
  * **Type I Error (False Positive):** We reject $H_0$ incorrectly. A legitimate email is marked as spam (sent to junk folder). This is usually high cost for user experience.
  * **Type II Error (False Negative):** We fail to reject $H_0$ when we should have. A spam email lands in the inbox. This is an annoyance but usually lower cost than missing an important email.
  * *Trade-off:* We often tune the model threshold to minimize Type I errors, potentially accepting more Type II errors.

**Q2: What is a p-value to a layperson?**
* **Answer:** "The p-value tells us how surprising the evidence is, assuming there is no real effect. Imagine we claim a coin is weighted. We flip it 10 times and get 10 heads. The p-value is the probability of getting 10 heads with a normal fair coin. Since that probability is tiny, we reject the idea that the coin is fair."

## Mathematical Questions

**Q3: How does the width of a Confidence Interval change if we increase sample size ($n$) or increase confidence level ($1-\alpha$)?**
* **Answer:**
  * Formula: $CI = \bar{x} \pm Z \frac{\sigma}{\sqrt{n}}$
  * **Increase $n$:** The denominator $\sqrt{n}$ increases, making the standard error smaller. The interval becomes **narrower** (more precise).
  * **Increase Confidence (e.g., 95% to 99%):** The Z-score increases (1.96 $\to$ 2.58). The interval becomes **wider**. To be more sure the true value is inside, we need to cast a wider net.

**Q4: Why do we use $N-1$ instead of $N$ when calculating Sample Variance?**
* **Answer:** This is Bessel's Correction.
  * The sample mean $\bar{x}$ is calculated from the data itself, which sits closer to the data points than the true population mean $\mu$ does.
  * If we used $N$, we would systematically underestimate the variance (Bias).
  * Using $N-1$ makes the sample variance an **unbiased estimator** of the population variance.

## Applied Questions

**Q5: You are comparing two Machine Learning models. Model A has 90% accuracy, Model B has 91% accuracy. The dataset size is 10,000. Is Model B better?**
* **Answer:**
  * We cannot say just by looking at the means. We need a statistical test, specifically a test for proportions (like McNemar's if paired, or Z-test if independent).
  * With $N=10,000$, the Standard Error is roughly $\sqrt{\frac{0.9(0.1)}{10000}} \approx 0.003$.
  * The difference is 0.01.
  * Z-score $\approx 0.01 / 0.003 \approx 3.33$.
  * Since $Z > 1.96$, the difference is statistically significant.
  * *However*, we must also ask: Is a 1% improvement worth the cost of deployment? (Practical significance).

**Q6: Your data violates the normality assumption. What do you do?**
* **Answer:**
  1. **Transform the data:** Apply Log or Box-Cox transformation to make it normal.
  2. **Non-parametric tests:** Use Mann-Whitney U test (instead of t-test) which compares ranks rather than raw values and doesn't require normality.
  3. **Bootstrapping:** Resample the data with replacement thousands of times to empirically build the sampling distribution and confidence intervals, bypassing theoretical assumptions entirely.

## Debugging & Failure Mode Questions

**Q7: You run an A/B test. The p-value is 0.04. You decide to launch. A week later, the metric crashes. What likely went wrong?**
* **Answer Candidates:**
  1. **Novelty Effect:** Users clicked the new feature just because it looked different, not because it was better. The effect wore off.
  2. **Peeking:** Did you stop the test *exactly* when p hit 0.04? If so, it was likely a False Positive.
  3. **Seasonality/Drift:** Was the test week special (e.g., a holiday)? The test conditions didn't match the general deployment conditions.
  4. **Simpson’s Paradox:** Did you aggregate data that shouldn't be aggregated? (e.g., The new model is worse on Android and worse on iOS, but due to uneven sampling, looks better overall).

## Follow-up / Probing

**Q8: Explain the relationship between Power, Sample Size, and Effect Size.**
* **Answer:** They are coupled. You generally fix $\alpha$ (0.05).
  * To detect a **smaller Effect Size**, you need **larger Sample Size**.
  * To increase **Power** (reduce false negatives), you need **larger Sample Size**.
  * If you have a fixed small sample size, you can only reliably detect **huge Effect Sizes** (otherwise your Power is low).
  * *Analogy:* Trying to see a small bird (small effect) requires a powerful telescope (large N). Seeing an elephant (large effect) can be done with the naked eye (small N).
