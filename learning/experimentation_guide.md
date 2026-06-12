# Experimentation in Machine Learning: A Comprehensive Reference Guide

---

## Table of Contents

1. [A/B Testing Design](#1-ab-testing-design)
2. [Multi-Armed Bandits](#2-multi-armed-bandits)
3. [Multi-Variable Testing](#3-multi-variable-testing)
4. [Causal Impact Analysis](#4-causal-impact-analysis)
5. [Experiment Platform Engineering](#5-experiment-platform-engineering)
6. [Interview Preparation](#6-interview-preparation)

---

# 1. A/B Testing Design

## 1.1 Motivation & Intuition

### Why Does A/B Testing Exist?

Suppose you run a streaming service and your team redesigns the homepage. Revenue rises 3% the week after launch. Was it the redesign, or was it a holiday weekend? You cannot know from observation alone because you changed the world and observed the result—there is no parallel universe where you kept the old homepage to compare against.

A/B testing manufactures that parallel universe. You randomly split users: half see the old page (control, **A**), half see the new page (treatment, **B**). Because assignment is random, the two groups are—on average—identical in every characteristic (age, spending habits, device, mood) except the treatment. Any difference in outcomes can therefore be attributed to the treatment, not to lurking confounders.

### Core Insight

The fundamental equation is:

$$
\text{Causal Effect} = E[Y \mid \text{Treatment}] - E[Y \mid \text{Control}]
$$

where $Y$ is the outcome metric (revenue, clicks, retention). Randomization guarantees that all confounders are balanced between groups, so this simple difference is an **unbiased estimate** of the true causal effect.

### Why This Matters for ML

In ML systems, A/B tests answer questions such as:

- Does the new ranking model increase engagement?
- Does a latency reduction from model compression increase conversion?
- Does a new recommendation algorithm reduce churn?
- Does a retrieval-augmented generation pipeline produce better answers than a vanilla LLM?

Without controlled experimentation, teams cannot distinguish genuine model improvements from noise, seasonality, or Simpson's-paradox-style confounds.

---

## 1.2 Conceptual Foundations

### 1.2.1 The Neyman–Rubin Potential Outcomes Framework

For each unit $i$ (user, session, request), define two **potential outcomes**:

| Symbol | Meaning |
|--------|---------|
| $Y_i(1)$ | Outcome if unit $i$ receives treatment |
| $Y_i(0)$ | Outcome if unit $i$ receives control |

The **individual treatment effect** is $\tau_i = Y_i(1) - Y_i(0)$.

The **fundamental problem of causal inference** is that we observe only one of $Y_i(1)$ or $Y_i(0)$ for each unit—never both. The unobserved one is the **counterfactual**.

The **Average Treatment Effect (ATE)** is:

$$
\tau = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]
$$

Randomization ensures:

$$
E[Y_i(1) \mid T_i = 1] = E[Y_i(1)] \quad \text{and} \quad E[Y_i(0) \mid T_i = 0] = E[Y_i(0)]
$$

so the naive difference in group means is unbiased for the ATE.

### 1.2.2 Randomization

**Why randomize?** Without randomization, treatment assignment may correlate with confounders. For example, if power users self-select into the new feature, you observe a positive correlation between the feature and engagement—but the cause is user type, not the feature.

#### Unit of Randomization

The **unit of randomization** is the entity that is independently assigned to a group. Common choices:

| Unit | When to Use | Risk if Wrong |
|------|-------------|---------------|
| **User** | Default for most product experiments | — |
| **Session** | Short-lived, no cross-session memory | Same user sees both variants → noisy/biased |
| **Page view** | Stateless changes (ad color) | Inconsistent experience within session |
| **Device** | Cross-platform products | Multiple users share device |
| **Cluster (geo, team)** | Network effects, marketplace experiments | Fewer independent units → low power |

**Rule of thumb:** randomize at the level where the treatment is experienced and outcomes are measured, and ensure that a single entity cannot be exposed to both treatment and control (the **Stable Unit Treatment Value Assumption**, SUTVA).

##### SUTVA

SUTVA has two parts:

1. **No interference:** one unit's outcome is not affected by another unit's assignment.
2. **No hidden variations of treatment:** treatment is the same for all treated units.

Violation example (interference): in a social network experiment, if User A (treated) shares content with User B (control), B's outcome is influenced by A's treatment. This is called **spillover** or **network interference**. Solutions include cluster randomization (randomize at the community level) or ego-network designs.

#### Stratified Randomization (Blocking)

Pure random assignment can produce imbalanced groups by chance, especially with small samples. **Stratification** partitions units into strata (e.g., mobile vs. desktop, new vs. returning users) and randomizes independently within each stratum.

Benefits:
- Guarantees balance on known important covariates.
- Reduces variance of the treatment effect estimator (more precise estimates).
- The variance reduction is proportional to how much the stratification variable explains outcome variance.

Mathematically, with $K$ strata each of size $N_k$, the stratified estimator is:

$$
\hat{\tau}_{\text{strat}} = \sum_{k=1}^{K} \frac{N_k}{N} \hat{\tau}_k
$$

where $\hat{\tau}_k = \bar{Y}_{k,T} - \bar{Y}_{k,C}$ is the within-stratum difference.

The variance of the stratified estimator is:

$$
\text{Var}(\hat{\tau}_{\text{strat}}) = \sum_{k=1}^{K} \left(\frac{N_k}{N}\right)^2 \text{Var}(\hat{\tau}_k)
$$

This is always $\leq$ the variance of the unstratified estimator when the strata differ in their mean outcomes.

#### Blocking

Blocking is conceptually similar to stratification but often refers to smaller, paired groups. In a **paired design**, you match two similar units and assign one to treatment and one to control. The treatment effect is estimated as the average of within-pair differences. This is the most aggressive variance reduction but requires good matching.

### 1.2.3 Sample Size Calculation

#### The Hypothesis Testing Framework

We frame the experiment as a statistical test:

- $H_0$: $\tau = 0$ (no treatment effect)
- $H_1$: $\tau = \delta$ (treatment effect equals some nonzero value $\delta$)

Two error types:

| | $H_0$ True | $H_0$ False |
|---|---|---|
| **Reject $H_0$** | Type I error ($\alpha$) | Correct (Power $= 1 - \beta$) |
| **Fail to reject** | Correct | Type II error ($\beta$) |

**Significance level** $\alpha$: probability of false positive (typically 0.05).

**Power** $1 - \beta$: probability of detecting a true effect of size $\delta$ (typically 0.80 or 0.90).

**Minimum Detectable Effect (MDE)**: the smallest effect size $\delta$ you want to be able to detect. This is a business decision: if a 1% lift in conversion is worth $10M/year, you want your experiment to detect 1% lifts.

#### Derivation of Sample Size Formula

Assume two equal-sized groups, each of size $n$, and the outcome has variance $\sigma^2$. Under $H_0$, the test statistic is:

$$
Z = \frac{\bar{Y}_T - \bar{Y}_C}{\sqrt{2\sigma^2 / n}} \sim N(0, 1)
$$

Under $H_1$ with true effect $\delta$:

$$
Z \sim N\left(\frac{\delta}{\sqrt{2\sigma^2 / n}},\; 1\right)
$$

We reject $H_0$ when $|Z| > z_{\alpha/2}$ (two-sided test). Power requires:

$$
P\left(Z > z_{\alpha/2} \mid \delta\right) = 1 - \beta
$$

Working through the algebra:

$$
\frac{\delta}{\sqrt{2\sigma^2 / n}} = z_{\alpha/2} + z_{\beta}
$$

Solving for $n$:

$$
\boxed{n = \frac{2\sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}}
$$

This is the sample size **per group**.

Key observations:
- $n \propto \sigma^2$: noisier metrics require larger samples.
- $n \propto 1/\delta^2$: detecting smaller effects requires quadratically more samples.
- Increasing power from 0.80 to 0.90 raises $z_\beta$ from 0.84 to 1.28, increasing $n$ by $\approx 50\%$.

#### Practical Adjustments

1. **Unequal group sizes:** If $n_T = r \cdot n_C$ with ratio $r$, replace $2$ with $(1 + 1/r)$.
2. **Proportions (binary outcomes):** $\sigma^2 = p(1-p)$ where $p$ is the base rate.
3. **Ratio metrics** (e.g., revenue per user): use the delta method for variance estimation.
4. **Clustered randomization:** inflate $n$ by the design effect $1 + (m-1)\rho$, where $m$ is the cluster size and $\rho$ is the intra-cluster correlation.
5. **Pre-experiment covariates (CUPED):** effective variance is $\sigma^2(1 - \rho_{XY}^2)$, reducing required $n$.

### 1.2.4 Duration Determination

#### Why Not Just Wait for the Required Sample Size?

Collecting $n$ observations as fast as possible introduces biases:

1. **Novelty effect:** Users initially engage more with anything new (Hawthorne-like effect). A short experiment overestimates the treatment effect. The effect decays over 1–4 weeks.

2. **Learning/habituation effect:** Users may initially under-engage because the new experience is unfamiliar, then gradually adapt. A short experiment underestimates the treatment effect.

3. **Day-of-week effects:** User behavior varies by weekday vs. weekend. Running only Monday–Friday misses weekend patterns. Always run for full weeks.

4. **Primacy bias:** Users who enter the experiment first are different from later arrivals (e.g., early adopters vs. organic users). If you stop early, your sample is biased.

5. **Peeking and optional stopping:** Checking results daily and stopping when $p < 0.05$ inflates the actual false positive rate to 20–30%. Either commit to a fixed duration or use sequential testing (see below).

**Rule of thumb:** Run experiments for **at least 2 full weeks**, preferably 4, to capture weekly cycles and allow novelty effects to dissipate. For longer-tail effects (subscription churn), run 4–8 weeks.

#### Sequential Testing

If business pressure demands early stopping, use methods with valid Type I error control:

- **Group sequential designs** (O'Brien–Fleming, Pocock boundaries): pre-specify a small number of interim analyses with adjusted significance thresholds.
- **Always-valid $p$-values** (mSPRT, mixture sequential probability ratio test): provide continuous monitoring with guaranteed $\alpha$ control.
- **Bayesian approaches:** Compute $P(\tau > 0 \mid \text{data})$ at any time; no $p$-value adjustment needed, but requires prior specification.

---

## 1.3 Mathematical Formulation

### 1.3.1 The Two-Sample Z-Test (Continuous Outcome)

Let $\bar{Y}_T, \bar{Y}_C$ be sample means, $s_T^2, s_C^2$ sample variances, $n_T, n_C$ sample sizes.

Test statistic:

$$
Z = \frac{\bar{Y}_T - \bar{Y}_C}{\sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}}
$$

Reject $H_0$ if $|Z| > z_{\alpha/2}$. Confidence interval for the treatment effect:

$$
\hat{\tau} \pm z_{\alpha/2} \sqrt{\frac{s_T^2}{n_T} + \frac{s_C^2}{n_C}}
$$

### 1.3.2 The Two-Sample Z-Test (Binary Outcome)

Let $\hat{p}_T, \hat{p}_C$ be observed proportions. Under $H_0$, pool:

$$
\hat{p} = \frac{n_T \hat{p}_T + n_C \hat{p}_C}{n_T + n_C}
$$

$$
Z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_T} + \frac{1}{n_C}\right)}}
$$

### 1.3.3 CUPED: Controlled-Experiment Using Pre-Experiment Data

**Problem:** High-variance metrics (e.g., revenue per user) require enormous sample sizes. Can we reduce variance without biasing the estimator?

**Idea:** Use a pre-experiment covariate $X$ (e.g., user's revenue in the week before the experiment) to "explain away" some variance.

Define the adjusted outcome:

$$
\tilde{Y}_i = Y_i - \theta (X_i - \bar{X})
$$

where $\theta$ is chosen to minimize $\text{Var}(\tilde{Y})$:

$$
\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}
$$

This is the OLS regression coefficient of $Y$ on $X$.

The variance of the adjusted outcome is:

$$
\text{Var}(\tilde{Y}) = \text{Var}(Y)(1 - \rho_{XY}^2)
$$

where $\rho_{XY}$ is the correlation between pre-experiment metric $X$ and outcome $Y$.

**Key insight:** If $\rho_{XY} = 0.7$, variance drops by $1 - 0.49 = 51\%$, effectively doubling the sample size. In practice, pre-experiment values of the same metric often have $\rho \geq 0.5$, making CUPED extremely useful.

**Unbiasedness:** Since $X$ is measured before the experiment, $E[X \mid T] = E[X \mid C]$ by randomization, so subtracting $\theta(X_i - \bar{X})$ does not bias the treatment effect.

### 1.3.4 The Delta Method for Ratio Metrics

Many ML metrics are ratios: CTR = clicks / impressions, revenue per user = total revenue / total users.

Let $R = \bar{Y} / \bar{X}$, where $\bar{Y}$ and $\bar{X}$ are sample means of numerator and denominator. By the delta method:

$$
\text{Var}(R) \approx \frac{1}{\mu_X^2}\left[\sigma_Y^2 - 2R\sigma_{XY} + R^2 \sigma_X^2\right] / n
$$

where $\mu_X = E[X]$, and $\sigma_Y^2, \sigma_X^2, \sigma_{XY}$ are variances and covariance per observation.

The treatment effect on the ratio metric is $\hat{R}_T - \hat{R}_C$, with variance given by summing the above for each group.

### 1.3.5 Multiple Testing Corrections

Running $m$ simultaneous tests (e.g., testing 20 metrics) inflates the **family-wise error rate** (FWER):

$$
\text{FWER} = 1 - (1-\alpha)^m \approx m\alpha \quad \text{for small } \alpha
$$

With $m = 20$ and $\alpha = 0.05$, FWER $\approx 0.64$—you have a 64% chance of at least one false positive!

**Corrections:**

| Method | Adjusted threshold | Controls |
|--------|-------------------|----------|
| **Bonferroni** | $\alpha / m$ | FWER (conservative) |
| **Šidák** | $1 - (1-\alpha)^{1/m}$ | FWER (slightly less conservative) |
| **Holm–Bonferroni** | Step-down procedure | FWER (uniformly more powerful than Bonferroni) |
| **Benjamini–Hochberg (BH)** | Step-up on sorted $p$-values | FDR (less conservative, controls expected false discovery proportion) |

In practice, most experiment platforms use BH to control the **False Discovery Rate (FDR)** at level $q$:

$$
\text{FDR} = E\left[\frac{\text{# false positives}}{\text{# total rejections}}\right] \leq q
$$

BH procedure: Sort $p$-values $p_{(1)} \leq \ldots \leq p_{(m)}$. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} q$. Reject all hypotheses with $p_{(i)} \leq p_{(k)}$.

---

## 1.4 Worked Examples

### Example 1: Sample Size for a Click-Through-Rate Experiment

**Setup:** You want to test a new recommendation model. Current CTR is $p_C = 0.05$ (5%). You want to detect a 10% relative improvement, i.e., $p_T = 0.055$ (5.5%). Use $\alpha = 0.05$, power $= 0.80$.

**Step 1: Compute parameters**

- $\delta = 0.055 - 0.05 = 0.005$
- $\sigma^2 \approx p(1-p)$. Under $H_0$, $p \approx 0.05$, so $\sigma^2 \approx 0.05 \times 0.95 = 0.0475$.
- $z_{\alpha/2} = z_{0.025} = 1.96$, $z_\beta = z_{0.20} = 0.84$.

**Step 2: Apply the formula**

$$
n = \frac{2 \times 0.0475 \times (1.96 + 0.84)^2}{0.005^2}
$$

$$
n = \frac{2 \times 0.0475 \times 7.84}{0.000025} = \frac{0.7448}{0.000025} = 29{,}792
$$

**Per group**, you need approximately **29,800 users**. Total: ~59,600 users.

**Step 3: Duration**

If the site gets 10,000 daily active users, and you allocate 50% to each group: $59{,}600 / 10{,}000 = 5.96$ days minimum. Round up to at least 7 days (one full week) to capture day-of-week effects.

### Example 2: CUPED Variance Reduction

**Setup:** Revenue per user has $\sigma^2 = 100$ (high variance). Pre-experiment revenue has correlation $\rho = 0.6$ with post-experiment revenue.

**Without CUPED:** $n = \frac{2 \times 100 \times (1.96 + 0.84)^2}{\delta^2}$.

**With CUPED:** Effective variance $= 100 \times (1 - 0.36) = 64$.

$$
n_{\text{CUPED}} = \frac{2 \times 64 \times (1.96 + 0.84)^2}{\delta^2}
$$

Ratio: $64/100 = 0.64$, so CUPED reduces the required sample size by 36%.

### Example 3: Stratified Analysis

**Setup:** 60% of users are mobile, 40% desktop. The treatment increases CTR by 0.01 on mobile but decreases by 0.005 on desktop.

**Unstratified (Simpson's paradox risk):** If mobile users have lower baseline CTR, the aggregate effect could mask the desktop degradation.

**Stratified estimator:**

$$
\hat{\tau}_{\text{strat}} = 0.6 \times 0.01 + 0.4 \times (-0.005) = 0.006 - 0.002 = 0.004
$$

The stratified analysis reveals heterogeneous effects and prevents masking.

---

## 1.5 Relevance to Machine Learning Practice

### Where A/B Tests Are Used

1. **Model launch decisions:** Compare new model vs. production model on online metrics (engagement, revenue, latency).
2. **Feature flagging:** Roll out a feature to X% of users and measure impact.
3. **Hyperparameter selection:** While offline evaluation uses validation sets, final decisions often require online A/B tests because offline metrics don't perfectly correlate with business metrics.
4. **Infra changes:** Verify that a serving optimization (quantization, caching, batching) doesn't degrade user-facing quality.

### When NOT to Use A/B Tests

- **Ethical constraints:** Cannot randomly withhold medical treatment.
- **Network effects:** Social features where SUTVA is violated. Use switchback or cluster randomization instead.
- **Long-term outcomes:** If the effect takes months to materialize (churn), A/B tests are expensive. Consider causal impact analysis or surrogate metrics.
- **Insufficient traffic:** If your product has 100 daily users, you cannot detect small effects. Consider Bayesian approaches or directional decision-making.

### Trade-offs

| Dimension | Consideration |
|-----------|--------------|
| **Bias–Variance** | Larger samples → lower variance; stratification/CUPED reduce variance without increasing bias |
| **Speed–Rigor** | Sequential testing offers faster decisions but slightly less power than fixed-horizon |
| **Sensitivity–False positives** | Multiple testing corrections reduce false positives but increase false negatives |
| **Granularity–Power** | Per-user randomization is most powerful; cluster randomization loses power but handles interference |

---

## 1.6 Common Pitfalls & Misconceptions

### Pitfall 1: Peeking at Results

Checking daily and stopping when $p < 0.05$ inflates the actual Type I error to 20–30%. This is because you are performing multiple comparisons over time. **Fix:** Use sequential testing or commit to a fixed end date.

### Pitfall 2: Underpowered Experiments

Running an experiment with insufficient sample size leads to:
- High probability of missing real effects (low power).
- Statistically significant results that overestimate the true effect (winner's curse / Type M error).
- Non-significant results being misinterpreted as "the treatment has no effect" (absence of evidence ≠ evidence of absence).

### Pitfall 3: Wrong Unit of Randomization

Randomizing at the page-view level when the treatment affects user behavior across pages. Users see inconsistent experiences, and observations are not independent. **Fix:** Randomize at the user level and analyze at the user level.

### Pitfall 4: Ignoring Network Effects

In marketplace or social products, treating one side (e.g., showing drivers a new UI) affects the other side (riders get faster pickups). SUTVA is violated. **Fix:** Use cluster randomization (by city, by time) or switchback designs.

### Pitfall 5: Survivorship Bias

If the treatment affects whether users remain in the sample (e.g., the new feature causes some users to leave), then the remaining treated users are a biased subset. **Fix:** Analyze using intent-to-treat (ITT) — analyze all users assigned, not just those who "completed" the treatment.

### Pitfall 6: Novelty and Primacy Effects

Short experiments overestimate effects because users are curious about changes. Long experiments may be needed to reach steady-state behavior. **Fix:** Segment by exposure time and check for effect decay.

### Pitfall 7: Multiple Metrics Without Correction

Testing 50 metrics at $\alpha = 0.05$ guarantees ~2.5 false positives. **Fix:** Designate a single primary metric before the experiment. Apply BH correction to secondary metrics.

### Pitfall 8: SRM (Sample Ratio Mismatch)

If the expected split is 50/50 but the observed split is 49/51, something is wrong with randomization. Possible causes: bot filtering that differentially affects groups, treatment causing more crashes (losing data), buggy assignment logic. **Fix:** Always check for SRM using a chi-squared test before interpreting results. An SRM invalidates all results.

---

# 2. Multi-Armed Bandits

## 2.1 Motivation & Intuition

### The Explore–Exploit Dilemma

Imagine a gambler facing three slot machines ("arms") with unknown payoff probabilities. Each pull yields a reward. The gambler wants to maximize total reward over $T$ pulls. There is a tension:

- **Explore:** Try arms you're uncertain about to learn which is best.
- **Exploit:** Pull the arm that currently appears best to accumulate reward.

Pure exploration wastes pulls on bad arms. Pure exploitation locks in on a possibly suboptimal arm because you never gathered enough information.

### Why Not Just A/B Test?

A classical A/B test explores for a fixed period, then exploits (launches the winner) forever. During the exploration phase, half of users receive the inferior treatment. The **regret** (reward lost relative to always choosing the best arm) accumulates linearly during the test.

Bandits reduce regret by **adapting allocation** during the experiment: as evidence accumulates that arm B is better, the algorithm shifts traffic toward B. This means fewer users experience inferior treatments.

### When Bandits Are Appropriate

- **Many variants to test** (e.g., 50 headlines for an ad): an A/B test allocating equal traffic to 50 variants is wasteful.
- **High opportunity cost** of exploration (e.g., medical trials, high-revenue pages).
- **Continuous optimization** rather than a one-time launch decision (e.g., ad creative rotation).

### When Bandits Are NOT Appropriate

- When you need a **statistically rigorous causal estimate** of the treatment effect (bandits bias the estimator).
- When there are **delayed rewards** (reward arrives days after the action).
- When you need to **measure secondary metrics** (bandits optimize one objective).
- When the effect is small and you need precise confidence intervals.

---

## 2.2 Conceptual Foundations

### Formal Setup

- $K$ arms, indexed $a \in \{1, \ldots, K\}$.
- Each arm $a$ has an unknown reward distribution with mean $\mu_a$.
- The best arm has mean $\mu^* = \max_a \mu_a$.
- At each round $t$, the agent selects arm $A_t$ and observes reward $R_t \sim P_{A_t}$.
- **Regret** after $T$ rounds:

$$
\text{Regret}(T) = T \mu^* - \sum_{t=1}^{T} \mu_{A_t} = \sum_{a:\mu_a < \mu^*} \Delta_a \cdot E[N_a(T)]
$$

where $\Delta_a = \mu^* - \mu_a$ is the **suboptimality gap** and $N_a(T)$ is the number of times arm $a$ was pulled.

**Goal:** Minimize regret over $T$ rounds.

**Theoretical lower bound** (Lai–Robbins, 1985): For any consistent algorithm,

$$
\text{Regret}(T) \geq \sum_{a: \Delta_a > 0} \frac{\Delta_a}{\text{KL}(P_a \| P^*)} \ln T
$$

This means regret must grow at least logarithmically in $T$. Good algorithms match this bound.

---

## 2.3 Algorithms

### 2.3.1 Epsilon-Greedy

**Algorithm:**
1. With probability $\epsilon$, choose an arm uniformly at random (explore).
2. With probability $1 - \epsilon$, choose the arm with the highest observed mean reward (exploit).

**Analysis:**

- Expected regret per round from exploration: $\epsilon \cdot \frac{1}{K} \sum_a \Delta_a$.
- Total regret: $O(\epsilon T)$ — linear in $T$, not optimal.
- **Decaying $\epsilon$:** Set $\epsilon_t = \min\left(1, \frac{cK}{d^2 t}\right)$ where $c > 0$ and $d = \min_{a: \Delta_a > 0} \Delta_a$. This achieves $O(\ln T)$ regret but requires knowledge of $d$.

**Pros:** Dead simple to implement. Easy to explain.
**Cons:** Explores uniformly (doesn't focus on promising arms). Linear regret with constant $\epsilon$.

### 2.3.2 Upper Confidence Bound (UCB1)

**Key idea:** Optimism in the face of uncertainty. For each arm, compute an upper confidence bound on its mean reward. Play the arm with the highest UCB — this automatically explores uncertain arms (wide confidence intervals) and exploits good arms (high estimated means).

**Algorithm (UCB1):**

At round $t$, select:

$$
A_t = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln t}{N_a(t-1)}}\right]
$$

where $\hat{\mu}_a$ is the sample mean of arm $a$ and $N_a(t-1)$ is the number of times $a$ has been pulled.

**Derivation of the confidence term:**

By Hoeffding's inequality, for bounded rewards $R \in [0,1]$:

$$
P\left(|\hat{\mu}_a - \mu_a| \geq \sqrt{\frac{\ln(1/\delta)}{2 N_a}}\right) \leq 2\delta
$$

Setting $\delta = 1/t^2$ (union bound over time) gives the exploration bonus $\sqrt{\frac{2 \ln t}{N_a}}$.

**Regret bound:**

$$
\text{Regret}(T) \leq \sum_{a: \Delta_a > 0} \frac{8 \ln T}{\Delta_a} + (1 + \frac{\pi^2}{3}) \sum_a \Delta_a
$$

This is $O(\sqrt{KT \ln T})$ in the worst case over gaps, matching the Lai–Robbins lower bound up to log factors.

**Intuition for the bound:** An arm with gap $\Delta_a$ will be pulled at most $O(\ln T / \Delta_a^2)$ times before the confidence interval shrinks below $\Delta_a$ and the algorithm stops selecting it. Each such pull costs $\Delta_a$, giving regret contribution $O(\ln T / \Delta_a)$.

### 2.3.3 Thompson Sampling

**Key idea:** Bayesian approach. Maintain a posterior distribution over each arm's mean. At each round, **sample** from each arm's posterior and play the arm with the highest sample.

**Algorithm (Bernoulli rewards):**

1. Initialize: $\alpha_a = 1, \beta_a = 1$ for all arms (uniform Beta prior).
2. At round $t$:
   a. For each arm $a$, sample $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$.
   b. Select $A_t = \arg\max_a \theta_a$.
   c. Observe reward $R_t \in \{0, 1\}$.
   d. Update: $\alpha_{A_t} \mathrel{+}= R_t$, $\beta_{A_t} \mathrel{+}= (1 - R_t)$.

**Why it works:** An arm with high uncertainty has a wide posterior — its sample sometimes exceeds the best arm's sample, triggering exploration. An arm with well-estimated low mean has a concentrated posterior — its sample rarely wins, so it's rarely explored.

**Regret:** Thompson Sampling achieves $O(\sqrt{KT \ln T})$ regret (Agrawal & Goyal, 2012), matching UCB and near-optimal.

**Posterior for Gaussian rewards:** If rewards are $N(\mu_a, \sigma^2)$ with known $\sigma^2$ and $\mu_a \sim N(\mu_0, \sigma_0^2)$:

After $n$ observations with sample mean $\bar{x}$:

$$
\mu_a \mid \text{data} \sim N\left(\frac{\sigma^2 \mu_0 + n \sigma_0^2 \bar{x}}{\sigma^2 + n \sigma_0^2}, \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n \sigma_0^2}\right)
$$

### 2.3.4 Comparison Table

| Algorithm | Regret | Requires Tuning | Bayesian | Handles Non-Stationary |
|-----------|--------|-----------------|----------|----------------------|
| $\epsilon$-greedy (constant) | $O(\epsilon T)$ | $\epsilon$ | No | Somewhat (constant exploration) |
| $\epsilon$-greedy (decaying) | $O(\ln T)$ | decay schedule | No | Poor |
| UCB1 | $O(\sqrt{KT \ln T})$ | No | No | Poor |
| Thompson Sampling | $O(\sqrt{KT \ln T})$ | Prior choice | Yes | Moderate (with discounting) |

---

## 2.4 Worked Example: Thompson Sampling with 3 Arms

**Setup:** $K = 3$ arms with true probabilities $\mu_1 = 0.3, \mu_2 = 0.5, \mu_3 = 0.7$. Start with Beta(1,1) priors.

**Round 1:**
- Sample: $\theta_1 \sim \text{Beta}(1,1) = 0.42$, $\theta_2 = 0.71$, $\theta_3 = 0.55$.
- Select arm 2 (highest sample). Observe $R = 1$.
- Update: $\alpha_2 = 2, \beta_2 = 1$. Posterior for arm 2: Beta(2,1).

**Round 2:**
- Sample: $\theta_1 \sim \text{Beta}(1,1) = 0.38$, $\theta_2 \sim \text{Beta}(2,1) = 0.82$, $\theta_3 \sim \text{Beta}(1,1) = 0.61$.
- Select arm 2. Observe $R = 0$.
- Update: $\alpha_2 = 2, \beta_2 = 2$. Posterior: Beta(2,2).

**Round 3:**
- Sample: $\theta_1 = 0.29$, $\theta_2 \sim \text{Beta}(2,2) = 0.45$, $\theta_3 = 0.88$.
- Select arm 3. Observe $R = 1$.
- Update: $\alpha_3 = 2, \beta_3 = 1$.

After many rounds, the posterior for arm 3 concentrates around 0.7, and Thompson Sampling increasingly selects arm 3. Poorly-performing arms are explored less as their posteriors concentrate below arm 3's posterior.

---

## 2.5 Relevance to ML Practice

- **Ad creative optimization:** Rotate multiple ad variants, automatically shifting budget to high-performers.
- **Recommendation ranking:** Explore new items vs. exploit known popular ones (contextual bandits with user features).
- **Hyperparameter tuning:** Successive halving and Hyperband are bandit-inspired methods.
- **Feature rollouts:** Gradually increase exposure to the winning variant.

**Contextual bandits** extend MABs by conditioning the policy on context $x_t$ (user features, query). The reward model is $E[R_t \mid A_t = a, X_t = x] = f_a(x)$. This connects to personalized recommendation and treatment effect heterogeneity.

---

## 2.6 Common Pitfalls

1. **Using bandits when you need causal inference:** Bandits maximize reward but produce biased treatment effect estimates because allocation is non-uniform. If the goal is "how much better is B than A?", use an A/B test.

2. **Ignoring delayed rewards:** If a click happens now but conversion happens in 3 days, the bandit learns from clicks and may favor clickbait. **Fix:** Use conversion as the reward with appropriate attribution windows, or use offline policy evaluation.

3. **Non-stationarity:** If reward distributions change over time (e.g., user preferences shift), vanilla UCB/TS converge to the initially best arm and fail to adapt. **Fix:** Use sliding-window or discounted variants (e.g., discounted Thompson Sampling).

4. **Cold-start problem:** With many arms (e.g., 10,000 items), pure exploration is infeasible. **Fix:** Use contextual bandits with feature-based generalization.

---

# 3. Multi-Variable Testing

## 3.1 Motivation & Intuition

An A/B test compares two variants of a **single factor** (e.g., button color: blue vs. green). But what if you want to simultaneously test button color, headline text, and image? You could run three sequential A/B tests, but:

1. **Interactions are missed:** The effect of color may depend on the headline. Sequential tests cannot detect interactions.
2. **Slow:** Three sequential tests take 3× as long.

**Multi-Variable Testing (MVT)** tests multiple factors simultaneously in a single experiment. Each user sees a specific **combination** of factor levels.

---

## 3.2 Conceptual Foundations

### Terminology

- **Factor:** A variable being tested (e.g., headline).
- **Level:** A specific value of a factor (e.g., "Save 20%" vs. "Limited Offer").
- **Treatment combination:** A specific assignment of levels to all factors (e.g., blue button + "Save 20%" headline + lifestyle image).
- **Main effect:** The average effect of changing one factor, averaged over all levels of other factors.
- **Interaction effect:** When the effect of one factor depends on the level of another factor.

### Full Factorial Design

A **full factorial design** tests every possible combination. With $F$ factors, each having $L_f$ levels:

$$
\text{Total combinations} = \prod_{f=1}^{F} L_f
$$

Example: 3 factors × 2 levels each = $2^3 = 8$ combinations. Each combination is a separate "arm" in the experiment.

**Advantages:** Estimates all main effects and all interaction effects. Maximally informative.

**Disadvantages:** The number of combinations grows exponentially. 5 factors × 4 levels = $4^5 = 1024$ combinations. With equal allocation, each combination gets $n/1024$ users — very few.

### Fractional Factorial Design

A **fractional factorial** tests only a subset of combinations, chosen so that main effects and low-order interactions can still be estimated, while higher-order interactions are **confounded** (aliased).

A **$2^{F-p}$ design** uses $2^{F-p}$ of the $2^F$ full-factorial combinations.

Example: A $2^{5-2}$ design tests 5 factors with only $2^3 = 8$ combinations (instead of 32). The price: some two-factor interactions are aliased with each other and cannot be separated.

**Resolution:** The resolution of a design indicates which effects are confounded.

| Resolution | Meaning |
|------------|---------|
| III | Main effects aliased with 2-factor interactions |
| IV | Main effects not aliased with 2-factor interactions, but 2-factor interactions aliased with each other |
| V | Main effects and 2-factor interactions all estimable; aliased only with 3-factor interactions |

**Choosing a design:** Use resolution V if two-factor interactions matter. Use resolution III if only main effects matter (screening).

### Analysis: ANOVA Framework

For a two-factor experiment with factors $A$ (levels $a = 1, \ldots, I$) and $B$ (levels $b = 1, \ldots, J$):

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
$$

where:
- $\mu$ = grand mean
- $\alpha_i$ = main effect of factor A at level $i$
- $\beta_j$ = main effect of factor B at level $j$
- $(\alpha\beta)_{ij}$ = interaction effect
- $\epsilon_{ijk} \sim N(0, \sigma^2)$ = residual noise

The total sum of squares decomposes:

$$
SS_{\text{total}} = SS_A + SS_B + SS_{AB} + SS_{\text{error}}
$$

Each $SS$ term has associated degrees of freedom. The F-statistic for testing factor $A$:

$$
F_A = \frac{SS_A / (I - 1)}{SS_{\text{error}} / (N - IJ)}
$$

---

## 3.3 Mathematical Formulation: $2^k$ Factorial

For $k$ factors each at 2 levels (coded $-1, +1$), the model is:

$$
Y = \beta_0 + \sum_{i} \beta_i x_i + \sum_{i<j} \beta_{ij} x_i x_j + \ldots + \epsilon
$$

The effect of factor $i$ is:

$$
\hat{\beta}_i = \frac{1}{2}\left[\bar{Y}(x_i = +1) - \bar{Y}(x_i = -1)\right]
$$

The interaction effect of factors $i$ and $j$:

$$
\hat{\beta}_{ij} = \frac{1}{2}\left[\bar{Y}(x_i x_j = +1) - \bar{Y}(x_i x_j = -1)\right]
$$

In orthogonal designs, all effect estimates are independent and have equal precision.

---

## 3.4 Worked Example: $2^3$ Full Factorial

**Factors:**
- $A$: Headline (short = $-1$, long = $+1$)
- $B$: Image (photo = $-1$, illustration = $+1$)
- $C$: Button color (blue = $-1$, green = $+1$)

**Results (mean CTR per combination):**

| $A$ | $B$ | $C$ | CTR |
|-----|-----|-----|-----|
| $-$ | $-$ | $-$ | 0.05 |
| $+$ | $-$ | $-$ | 0.06 |
| $-$ | $+$ | $-$ | 0.04 |
| $+$ | $+$ | $-$ | 0.07 |
| $-$ | $-$ | $+$ | 0.05 |
| $+$ | $-$ | $+$ | 0.07 |
| $-$ | $+$ | $+$ | 0.04 |
| $+$ | $+$ | $+$ | 0.08 |

**Main effect of A:**

$$
\hat{\beta}_A = \frac{1}{2}\left[\frac{0.06 + 0.07 + 0.07 + 0.08}{4} - \frac{0.05 + 0.04 + 0.05 + 0.04}{4}\right]
$$

$$
= \frac{1}{2}\left[0.07 - 0.045\right] = \frac{0.025}{2} = 0.0125
$$

Headline $A$ at long ($+1$) increases CTR by $2 \times 0.0125 = 0.025$ (2.5 percentage points) on average.

**Interaction $A \times B$:**

$$
\hat{\beta}_{AB} = \frac{1}{2}\left[\frac{0.05 + 0.07 + 0.05 + 0.08}{4} - \frac{0.06 + 0.04 + 0.07 + 0.04}{4}\right]
$$

$$
= \frac{1}{2}\left[0.0625 - 0.0525\right] = 0.005
$$

The interaction is positive: long headline + illustration together yield an extra CTR boost beyond their individual effects.

---

## 3.5 Relevance to ML Practice

- **UI optimization:** Test multiple UI elements simultaneously (layout, copy, color, CTA position).
- **Model serving configuration:** Simultaneously test model version, cache TTL, reranking strategy.
- **Email/notification campaigns:** Subject line × send time × body template.
- **Feature combinations:** Test whether features interact (e.g., autocomplete + spell-check).

### Trade-offs vs. Sequential A/B Tests

| | MVT | Sequential A/B |
|--|-----|----------------|
| **Speed** | Faster (one experiment) | Slower ($F$ experiments) |
| **Interactions** | Detected | Missed |
| **Complexity** | Higher (more variants) | Simpler |
| **Power per combination** | Lower (traffic split many ways) | Higher (traffic split two ways) |

---

## 3.6 Common Pitfalls

1. **Too many factors:** With 6 factors × 3 levels = 729 combinations. Traffic is too diluted. **Fix:** Use fractional factorial or prioritize factors with Pareto analysis.

2. **Ignoring interactions:** Analyzing only main effects when interactions are significant leads to suboptimal combinations. **Fix:** Always test for interactions before simplifying the model.

3. **Confounding in fractional designs:** A significant effect might be the main effect or its alias. **Fix:** Run follow-up "fold-over" experiments to de-alias.

4. **Combining with bandits:** Applying bandit allocation to factorial designs biases effect estimates. Use equal allocation for clean analysis.

---

# 4. Causal Impact Analysis

## 4.1 Motivation & Intuition

Sometimes you cannot run an A/B test:

- A policy change affects an entire country simultaneously (no control group).
- A product launch happened in one city, and you want to know its effect.
- A competitor entered a market, and you want to quantify the impact.

**Causal impact analysis** uses observational data plus structural assumptions to estimate treatment effects **without randomization**.

---

## 4.2 Conceptual Foundations

### 4.2.1 Difference-in-Differences (DiD)

**Setup:** You have two groups (treated and control) observed over two time periods (before and after treatment).

| | Before ($t=0$) | After ($t=1$) |
|--|----------------|---------------|
| **Treated** | $Y_{T,0}$ | $Y_{T,1}$ |
| **Control** | $Y_{C,0}$ | $Y_{C,1}$ |

**Naive approach 1:** Compare $Y_{T,1} - Y_{C,1}$. Problem: groups may have different baseline levels (selection bias).

**Naive approach 2:** Compare $Y_{T,1} - Y_{T,0}$. Problem: time trends affect both groups (maturation bias).

**DiD idea:** Subtract out both the group difference and the time trend:

$$
\hat{\tau}_{\text{DiD}} = (Y_{T,1} - Y_{T,0}) - (Y_{C,1} - Y_{C,0})
$$

This removes:
- Fixed group differences (first differencing).
- Common time trends (second differencing).

**The parallel trends assumption:** DiD assumes that absent treatment, the treated and control groups would have followed the same trend. Formally:

$$
E[Y_{T,1}(0) - Y_{T,0}(0)] = E[Y_{C,1}(0) - Y_{C,0}(0)]
$$

where $Y(0)$ denotes the potential outcome without treatment.

This is **untestable** (we don't observe $Y_{T,1}(0)$), but we can check plausibility by examining pre-treatment trends. If the two groups' outcomes moved in parallel before the intervention, it's more plausible they would have continued doing so.

**Regression formulation:**

$$
Y_{it} = \alpha + \beta \cdot \text{Treated}_i + \gamma \cdot \text{Post}_t + \delta \cdot (\text{Treated}_i \times \text{Post}_t) + \epsilon_{it}
$$

where $\delta = \hat{\tau}_{\text{DiD}}$ is the coefficient of interest.

#### What Breaks When Parallel Trends Fails

If the treated group was already trending differently before treatment (e.g., growing faster), DiD attributes the pre-existing trend divergence to the treatment, biasing $\hat{\tau}$ upward or downward. Common violations include anticipation effects (treated units change behavior before treatment), differential seasonality, and composition changes.

### 4.2.2 Synthetic Control Method

**Problem with DiD:** Sometimes there is no single good control group. The treated unit (e.g., California) may not have a parallel-trending counterpart.

**Idea:** Construct a **synthetic control** — a weighted combination of untreated units that best matches the treated unit's pre-treatment trajectory.

**Formally:** Let unit 1 be treated and units $2, \ldots, J+1$ be the "donor pool." Find weights $w_2, \ldots, w_{J+1}$ such that:

$$
\sum_{j=2}^{J+1} w_j Y_{j,t} \approx Y_{1,t} \quad \text{for all pre-treatment } t
$$

subject to $w_j \geq 0$ and $\sum w_j = 1$ (convex combination).

The treatment effect at time $t$ after intervention:

$$
\hat{\tau}_t = Y_{1,t} - \sum_{j=2}^{J+1} w_j Y_{j,t}
$$

The synthetic control is the counterfactual: what the treated unit would have looked like without treatment.

**Optimization:** Minimize the pre-treatment prediction error:

$$
\min_w \sum_{t=1}^{T_0} \left(Y_{1,t} - \sum_{j} w_j Y_{j,t}\right)^2
$$

subject to $w_j \geq 0, \sum w_j = 1$.

Often, covariates $X$ are also matched:

$$
\min_w \|X_1 - X_0 W\|_V^2
$$

where $V$ is a diagonal weighting matrix (importance of each covariate), selected by cross-validation over the pre-treatment period.

**Inference:** Since there's only one treated unit, standard errors are not available. Instead, use **placebo tests**: apply the synthetic control method to each donor unit (pretending it was treated) and compute the distribution of placebo effects. If the treated unit's effect is extreme relative to placebos, the effect is "significant."

### 4.2.3 Bayesian Structural Time Series (BSTS) / CausalImpact

Google's **CausalImpact** (Brodersen et al., 2015) extends synthetic control with a Bayesian approach:

1. Fit a structural time series model (local level + seasonality + regression on control series) to pre-treatment data.
2. After intervention, use the model to predict the counterfactual.
3. The treatment effect is the difference between observed and predicted counterfactual.
4. Bayesian inference provides posterior credible intervals for the effect.

**Advantages over classic synthetic control:**
- Provides uncertainty quantification (credible intervals, posterior probability of a causal effect).
- Handles seasonality and trend explicitly.
- Accommodates many control series via spike-and-slab priors (automatic variable selection).

---

## 4.3 Mathematical Formulation

### DiD with Multiple Periods (Event Study)

With multiple pre- and post-treatment periods, estimate:

$$
Y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \delta_k \cdot \mathbb{1}[\text{Treated}_i] \cdot \mathbb{1}[t = k] + \epsilon_{it}
$$

where $\alpha_i$ are unit fixed effects, $\gamma_t$ are time fixed effects, and the $\delta_k$ coefficients trace out the dynamic treatment effect. The period $k = -1$ (one period before treatment) is the reference.

**Pre-treatment coefficients** $\delta_{k}$ for $k < -1$ should be approximately zero if parallel trends holds. This is the standard "event study plot" used to validate the assumption.

### Two-Way Fixed Effects (TWFE) Caution

With staggered treatment adoption (different units treated at different times), the standard TWFE estimator:

$$
Y_{it} = \alpha_i + \gamma_t + \delta \cdot D_{it} + \epsilon_{it}
$$

can be **biased** because it uses already-treated units as controls for newly-treated units. Recent econometrics literature (Goodman-Bacon 2021, Callaway & Sant'Anna 2021, Sun & Abraham 2021) provides corrected estimators. The key insight is that $\hat{\delta}_{\text{TWFE}}$ is a weighted average of all possible 2×2 DiD comparisons, and some weights can be **negative**, leading to sign reversal.

---

## 4.4 Worked Example: DiD

**Setup:** A streaming service launches a new recommendation algorithm in the US on Jan 1. Canada (no change) serves as control.

| | Pre (Dec) Avg. Watch Time | Post (Jan) Avg. Watch Time |
|--|---------------------------|----------------------------|
| US (Treated) | 45 min | 52 min |
| Canada (Control) | 40 min | 44 min |

$$
\hat{\tau}_{\text{DiD}} = (52 - 45) - (44 - 40) = 7 - 4 = 3 \text{ minutes}
$$

Interpretation: Controlling for the common seasonal trend (+4 min in both countries), the new algorithm increased watch time by 3 minutes.

**Validity check:** Were US and Canada trending in parallel before Dec? Plot several prior months. If US was growing at 2 min/month and Canada at 1 min/month, parallel trends is violated, and the 3-minute estimate is biased upward.

---

## 4.5 Worked Example: Synthetic Control

**Setup:** A ride-sharing company launches a driver incentive program in San Francisco. Other cities (LA, NYC, Chicago, Seattle, Boston) are potential controls.

**Pre-intervention period:** 12 months of weekly ride counts. Find weights:

$$
\hat{Y}_{\text{SF, synth}} = 0.45 \cdot Y_{\text{LA}} + 0.30 \cdot Y_{\text{Seattle}} + 0.20 \cdot Y_{\text{NYC}} + 0.05 \cdot Y_{\text{Boston}}
$$

that best match SF's pre-intervention trajectory (Chicago gets zero weight because it doesn't help the fit).

**Post-intervention:** Observed SF rides = 15,000/week. Synthetic SF = 13,200/week.

$$
\hat{\tau} = 15{,}000 - 13{,}200 = 1{,}800 \text{ rides/week}
$$

**Placebo tests:** Apply the method to each donor city. If placebo effects are in the range $[-500, +600]$ and the SF effect is $+1,800$, this is highly unusual (p-value $\approx 0.03$), suggesting a real causal effect.

---

## 4.6 Relevance to ML Practice

- **Policy evaluation without experiments:** A government mandates algorithmic transparency. What happened to user engagement?
- **Market-level interventions:** Launching in a new market where randomization is infeasible.
- **Retrospective analysis:** The experiment wasn't planned; can we still estimate the effect?
- **Long-term effects:** Combine short-term A/B test data with long-term observational data using DiD on the post-experiment period.

### Trade-offs

| Method | Requires | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| DiD | Parallel trends, ≥1 control unit | Simple, intuitive | Biased if trends diverge |
| Synthetic Control | Many donor units, good pre-fit | Data-driven control, transparent weights | One treated unit, no standard errors (use placebos) |
| CausalImpact (BSTS) | Control time series, pre-intervention fit | Uncertainty quantification, handles seasonality | Assumes post-intervention control series unaffected |

---

## 4.6 Common Pitfalls

1. **Parallel trends failure:** The most common violation. Always plot pre-treatment trends and run an event study. If trends diverge, DiD estimates are unreliable.

2. **Spillovers to control group:** If the treatment in SF causes drivers to move to LA (control), LA's outcome changes and the control is contaminated.

3. **Overfitting in synthetic control:** With many donor units and few pre-treatment periods, the synthetic control may perfectly fit pre-treatment data by chance but diverge post-treatment for non-causal reasons. **Fix:** Keep the number of donors modest relative to pre-treatment periods.

4. **Compositional changes:** If the treatment changes who enters the sample (e.g., the new algorithm attracts different users), the treated group's composition shifts, confounding the effect.

5. **Staggered adoption in TWFE:** Using standard two-way fixed effects when units adopt treatment at different times can produce severely biased estimates. Use modern heterogeneity-robust estimators.

---

# 5. Experiment Platform Engineering

## 5.1 Motivation & Intuition

Running experiments at scale (thousands of concurrent experiments across millions of users) requires infrastructure that is reliable, fast, and scientifically sound. An experiment platform is the system that manages the lifecycle from experiment creation to analysis.

Without a proper platform:
- Engineers hardcode experiment logic, introducing bugs.
- Randomization is ad hoc and potentially biased.
- Metrics are computed inconsistently.
- Interactions between concurrent experiments are ignored.
- Results are stored in spreadsheets and not reproducible.

---

## 5.2 Conceptual Foundations

### 5.2.1 Architecture Overview

A modern experiment platform has three major components:

**1. Configuration Management**
- Stores experiment definitions: name, hypothesis, factors, levels, allocation percentage, start/end dates, targeting criteria, primary/secondary metrics.
- Version-controlled and auditable. Every change is logged.
- Supports experiment lifecycle states: Draft → Running → Paused → Completed → Archived.
- Implements review/approval workflows (a data scientist must sign off on sample size calculations before launch).

**2. Allocation Service (Randomization Engine)**
- At request time, deterministically maps a user to an experiment variant.
- Must be **deterministic** (same user always gets the same variant), **fast** (microsecond latency), and **statistically unbiased**.
- Common implementation: hash the concatenation of user ID and experiment ID, then map the hash to a bucket.

$$
\text{bucket} = \text{hash}(\text{user\_id} + \text{experiment\_id}) \mod 1000
$$

Variants are mapped to bucket ranges: control = [0, 499], treatment = [500, 999] for a 50/50 split.

- **Layer system** (Google's Overlapping Experiments Infrastructure): Experiments are organized into independent layers. Each layer partitions traffic. Experiments in different layers are orthogonal (user assignments are independent). Experiments in the same layer are mutually exclusive.

#### Why Hashing?

Hashing provides:
- **Determinism:** No need to store assignments; recompute on every request.
- **Scalability:** No database lookups.
- **Independence between experiments:** Different experiment IDs produce different hash sequences, so assignments are uncorrelated.

Properties of a good hash function: uniform distribution over buckets, independence between experiments (achieved by including experiment ID in the hash input), avalanche effect (small input changes produce large output changes).

**3. Analysis Pipeline**

- **Logging:** Every experiment exposure is logged (user ID, experiment ID, variant, timestamp). This is the source of truth.
- **Metric computation:** Joins exposure logs with outcome data (clicks, conversions, revenue) from the data warehouse.
- **Statistical analysis:** Computes point estimates, confidence intervals, p-values. Applies variance reduction (CUPED), multiple testing corrections, and heterogeneous treatment effect analysis.
- **Guardrail metrics:** Automatically flags if the experiment degrades critical metrics (latency, crash rate, error rate) regardless of the primary metric result.
- **Sample Ratio Mismatch (SRM) detection:** Compares expected vs. observed allocation ratios. An SRM indicates a data pipeline bug, and the experiment should be invalidated.
- **Reporting dashboard:** Interactive interface showing metric lifts, confidence intervals, time series of effects, subgroup analyses.

### 5.2.2 Handling Multiple Concurrent Experiments

At large companies, thousands of experiments run simultaneously. Key challenges:

1. **Interaction effects:** Experiment A (new model) and Experiment B (new UI) may interact — the combined effect differs from the sum of individual effects. The layer system addresses this by making cross-layer experiments orthogonal (independent assignment). Within a layer, only one experiment is active per user.

2. **Traffic starvation:** Too many experiments competing for traffic. Solution: prioritize experiments by business impact, use fractional allocation (5% per experiment), and ensure the control group is shared across experiments in the same layer.

3. **Holdout groups:** Maintain a persistent "holdback" group that never receives any experimental treatment. This allows measuring the cumulative long-term impact of all shipped experiments.

### 5.2.3 Trigger Analysis

Not all users in an experiment are actually **exposed** to the treatment. For example, a search ranking experiment only affects users who search. Analyzing all assigned users (intent-to-treat, ITT) is valid but dilutes the effect. **Trigger analysis** restricts to users who were actually triggered (exposed to the changed code path).

The trigger-dilute-bias trade-off:
- **ITT (all assigned):** Unbiased but diluted (effect estimate is smaller because non-triggered users contribute zero effect).
- **Triggered-only:** More precise but potentially biased if the treatment changes who triggers (e.g., a faster page load causes more users to reach the search box).

Best practice: Report both ITT and triggered analysis. If they disagree, investigate why.

### 5.2.4 Guardrail and Ecosystem Metrics

Beyond the primary metric, monitor:
- **Latency** (p50, p95, p99): New models may be slower.
- **Error rates:** Feature bugs caught by monitoring.
- **Revenue / engagement canaries:** Catch regressions early.
- **Long-term proxy metrics:** Metrics that predict long-term retention but are observable in the short term.

A common framework:
- **Primary metric:** The metric the experiment is designed to move. Must be specified before the experiment starts.
- **Secondary metrics:** Additional metrics of interest (may be adjusted for multiple comparisons).
- **Guardrail metrics:** Must not degrade. The experiment is automatically stopped if guardrails are violated (e.g., crash rate increases by > 0.1%).

---

## 5.3 Worked Example: Allocation Service Implementation

**Pseudocode:**

```
function get_variant(user_id, experiment_config):
    # Step 1: Check targeting criteria
    if not meets_targeting(user_id, experiment_config.targeting):
        return None  # User not eligible

    # Step 2: Compute deterministic hash
    hash_input = f"{user_id}_{experiment_config.layer_id}_{experiment_config.salt}"
    bucket = md5(hash_input) % 10000  # 10,000 buckets for 0.01% granularity

    # Step 3: Check if bucket falls in experiment's traffic allocation
    if bucket < experiment_config.traffic_start or bucket >= experiment_config.traffic_end:
        return None  # User not in this experiment's traffic

    # Step 4: Determine variant
    variant_bucket = md5(f"{user_id}_{experiment_config.id}") % 10000
    cumulative = 0
    for variant in experiment_config.variants:
        cumulative += variant.allocation * 10000  # e.g., 0.5 * 10000 = 5000
        if variant_bucket < cumulative:
            return variant.name

    return experiment_config.variants[-1].name  # fallback
```

Key design decisions:
- **Different hash for layer allocation vs. variant assignment:** Ensures that a user's position in the layer traffic (Step 3) is independent of their variant assignment (Step 4).
- **Salt per experiment:** Prevents correlation between experiments in the same layer.
- **10,000 buckets:** Provides 0.01% granularity for traffic allocation.

---

## 5.4 Common Pitfalls

1. **Non-deterministic assignment:** Using random number generators without seeding on user ID causes the same user to see different variants on different requests. This violates SUTVA and creates a terrible user experience.

2. **Logging exposure after the response:** If exposure is logged after the page renders, users who experience errors (and never render the page) are excluded from the treatment group — survivorship bias.

3. **Shared mutable state:** If the treatment modifies a shared resource (cache, model), control users are also affected (violation of SUTVA). Solution: separate code paths, separate model endpoints.

4. **Ignoring interactions between experiments:** If experiments interact, the layer system must enforce mutual exclusivity within layers. Without layers, experiments can produce contradictory results.

5. **Missing SRM checks:** A 51/49 split when 50/50 was expected indicates a bug. Shipping results from an experiment with SRM is worse than not experimenting at all, because the effect estimate is biased in an unknown direction.

6. **Inadequate pre-experiment planning:** No documented hypothesis, no pre-registered primary metric, no sample size calculation. This leads to HARKing (Hypothesizing After Results are Known) and inflated false positives.

---

# 6. Interview Preparation

## 6.1 Foundational Questions

### Q1: What is an A/B test and why do we use it?

**Answer:** An A/B test is a randomized controlled experiment that compares two variants (A = control, B = treatment) by randomly assigning users to each group and measuring the difference in a predefined outcome metric. We use it because randomization ensures that the two groups are statistically equivalent in all characteristics except the treatment, allowing us to attribute any observed difference in outcomes to the treatment itself. This establishes **causation**, not merely correlation. Without randomization, observed differences could be driven by confounders — characteristics that differ between groups and independently affect the outcome.

### Q2: Explain the difference between Type I error, Type II error, and power.

**Answer:**

- **Type I error ($\alpha$):** Rejecting the null hypothesis when it is true — a false positive. You conclude the treatment works when it doesn't. Typically controlled at 0.05.
- **Type II error ($\beta$):** Failing to reject the null when the alternative is true — a false negative. You miss a real effect.
- **Power ($1 - \beta$):** Probability of detecting a true effect of a given size. Typically targeted at 0.80 or higher.

The relationship: increasing power requires increasing sample size, increasing effect size, or increasing $\alpha$ (accepting more false positives). These trade off against each other.

### Q3: What is the Minimum Detectable Effect (MDE) and how do you choose it?

**Answer:** MDE is the smallest treatment effect that the experiment is designed to detect with a given power and significance level. It is a **business decision**, not a statistical one. You choose MDE based on: (1) the economic value of the improvement — if a 0.1% CTR lift is worth $1M/year, you set MDE at 0.1%; (2) practical feasibility — what effect size is realistic for the intervention; and (3) sample size constraints — smaller MDE requires quadratically more samples.

### Q4: What is SUTVA? Give an example of a violation.

**Answer:** The Stable Unit Treatment Value Assumption has two parts: (1) no interference — one unit's outcome depends only on its own treatment assignment, not others'; (2) no hidden treatment variations — the treatment is the same for all treated units.

Violation example: In a ride-sharing app, if treated drivers receive a navigation improvement, they complete rides faster, freeing up supply for control-group riders. Control riders experience shorter wait times, which dilutes the measured treatment effect. The control group is "contaminated" by the treatment.

### Q5: What is CUPED and why is it useful?

**Answer:** CUPED (Controlled-Experiment Using Pre-Experiment Data) is a variance-reduction technique. It adjusts the outcome variable using pre-experiment data that is correlated with the outcome. The adjusted outcome is $\tilde{Y}_i = Y_i - \theta(X_i - \bar{X})$, where $X_i$ is a pre-experiment covariate and $\theta$ is chosen to minimize variance (equal to the regression coefficient). This reduces variance by a factor of $(1 - \rho^2)$, where $\rho$ is the pre-post correlation. It does not introduce bias because $X$ was measured before randomization and is therefore balanced between groups. In practice, CUPED can reduce required sample sizes by 30–50%.

---

## 6.2 Mathematical Questions

### Q6: Derive the sample size formula for a two-sample Z-test.

**Answer:**

Under $H_0: \tau = 0$, the test statistic $Z = \frac{\bar{Y}_T - \bar{Y}_C}{\text{SE}}$ where $\text{SE} = \sqrt{2\sigma^2/n}$ (equal groups of size $n$). Under $H_1: \tau = \delta$, $Z \sim N(\delta/\text{SE}, 1)$.

For power $1 - \beta$: we need $P(Z > z_{\alpha/2} \mid H_1) = 1 - \beta$.

$$
P\left(\frac{\bar{Y}_T - \bar{Y}_C}{\text{SE}} > z_{\alpha/2}\right) = 1 - \beta
$$

Under $H_1$: $\frac{\bar{Y}_T - \bar{Y}_C}{\text{SE}} \sim N(\delta/\text{SE}, 1)$.

$$
P\left(N(\delta/\text{SE}, 1) > z_{\alpha/2}\right) = 1 - \beta
$$

$$
\Phi\left(\frac{\delta}{\text{SE}} - z_{\alpha/2}\right) = 1 - \beta
$$

$$
\frac{\delta}{\text{SE}} - z_{\alpha/2} = z_\beta
$$

$$
\frac{\delta}{\sqrt{2\sigma^2/n}} = z_{\alpha/2} + z_\beta
$$

$$
n = \frac{2\sigma^2(z_{\alpha/2} + z_\beta)^2}{\delta^2}
$$

### Q7: Why does detecting half the effect size require 4× the sample size?

**Answer:** From $n = 2\sigma^2(z_{\alpha/2} + z_\beta)^2 / \delta^2$, the sample size is proportional to $1/\delta^2$. If $\delta' = \delta/2$:

$$
n' = \frac{2\sigma^2(z_{\alpha/2} + z_\beta)^2}{(\delta/2)^2} = 4 \cdot \frac{2\sigma^2(z_{\alpha/2} + z_\beta)^2}{\delta^2} = 4n
$$

Intuitively, a smaller signal is harder to distinguish from noise. The signal-to-noise ratio is $\delta / (\sigma/\sqrt{n})$. To maintain the same SNR when $\delta$ halves, you must quadruple $n$ to halve $\sigma/\sqrt{n}$.

### Q8: Derive the CUPED variance reduction.

**Answer:**

Define $\tilde{Y} = Y - \theta(X - \bar{X})$.

$$
\text{Var}(\tilde{Y}) = \text{Var}(Y) + \theta^2 \text{Var}(X) - 2\theta \text{Cov}(Y, X)
$$

Minimize over $\theta$: $\frac{\partial}{\partial \theta} = 2\theta \text{Var}(X) - 2\text{Cov}(Y, X) = 0$

$$
\theta^* = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}
$$

Substituting back:

$$
\text{Var}(\tilde{Y}) = \text{Var}(Y) - \frac{\text{Cov}(Y,X)^2}{\text{Var}(X)} = \text{Var}(Y)\left(1 - \rho_{XY}^2\right)
$$

### Q9: How does the delta method work for ratio metrics?

**Answer:** Let $R = g(\bar{Y}, \bar{X}) = \bar{Y}/\bar{X}$. By the delta method, for a function $g$ of random variables:

$$
\text{Var}(g(\bar{Y}, \bar{X})) \approx \nabla g^\top \Sigma \nabla g
$$

where $\nabla g = (\partial g/\partial \bar{Y}, \partial g/\partial \bar{X})^\top = (1/\mu_X, -\mu_Y/\mu_X^2)^\top$ evaluated at population means, and $\Sigma$ is the covariance matrix of $(\bar{Y}, \bar{X})$.

$$
\text{Var}(R) \approx \frac{1}{\mu_X^2}\text{Var}(\bar{Y}) + \frac{\mu_Y^2}{\mu_X^4}\text{Var}(\bar{X}) - \frac{2\mu_Y}{\mu_X^3}\text{Cov}(\bar{Y}, \bar{X})
$$

$$
= \frac{1}{n\mu_X^2}\left[\sigma_Y^2 + R^2 \sigma_X^2 - 2R\sigma_{XY}\right]
$$

This allows constructing confidence intervals for ratio metric differences between treatment and control.

### Q10: Explain the Lai–Robbins lower bound for bandits.

**Answer:** For any algorithm that is **consistent** (i.e., for any suboptimal arm $a$, $E[N_a(T)] / T \to 0$ as $T \to \infty$), the expected number of times arm $a$ is pulled satisfies:

$$
\liminf_{T \to \infty} \frac{E[N_a(T)]}{\ln T} \geq \frac{1}{\text{KL}(P_a \| P^*)}
$$

where KL is the Kullback–Leibler divergence between arm $a$'s reward distribution and the optimal arm's distribution. The total regret therefore satisfies:

$$
\text{Regret}(T) \geq \sum_{a: \Delta_a > 0} \frac{\Delta_a}{\text{KL}(P_a \| P^*)} \ln T
$$

The intuition: to distinguish arm $a$ from the best arm, you need at least $O(1/\text{KL})$ samples. The $\ln T$ factor arises because as the horizon grows, the algorithm must maintain confidence that the best arm is truly best, requiring ongoing (logarithmic) exploration.

---

## 6.3 Applied Questions

### Q11: You're an ML engineer at a recommendation company. You've trained a new model that improves offline NDCG by 5%. How do you decide whether to launch it?

**Answer:**

1. **Offline validation:** Confirm the 5% NDCG improvement is robust (cross-validation, held-out test set, no data leakage, no evaluation bugs).

2. **Online experiment design:**
   - Define primary online metric (e.g., user engagement, click-through rate, or a composite metric aligned with business goals).
   - Compute required sample size based on the metric's variance, the expected effect size (5% NDCG improvement may translate to only 1% engagement lift), and desired power.
   - Set guardrail metrics: latency (new model may be slower), revenue, error rate.
   - Determine duration: at least 2 weeks to capture day-of-week effects and allow novelty effects to dissipate.

3. **Launch experiment:** Use the experiment platform to randomize at the user level. Monitor SRM daily.

4. **Analysis:** After the pre-determined duration, analyze the primary metric. Check guardrails. If the primary metric shows a statistically significant and practically meaningful improvement and guardrails are not violated, proceed to launch.

5. **Ramp:** Gradually increase allocation (10% → 25% → 50% → 100%) to catch long-tail issues.

6. **Long-term holdback:** Maintain a small holdback group (1–5%) to monitor for long-term effects (engagement decay, user churn).

### Q12: Your A/B test shows a +2% lift in clicks but a -1% lift in revenue. What do you do?

**Answer:**

1. **Check if both are statistically significant.** If clicks are significant but revenue is not, the revenue change may be noise.

2. **Understand the mechanism:** Does the treatment increase low-quality clicks (clickbait) that don't convert? Segment by click depth — do users who click more also convert less?

3. **Check for Simpson's paradox:** Does the effect differ by segment (mobile vs. desktop, new vs. returning)? The aggregate may mask opposing effects.

4. **Consider the metric hierarchy:** If revenue is the primary metric and it shows a negative trend, do not launch even if clicks are up. Clicks may be a vanity metric.

5. **Examine long-term metrics:** Does the treatment improve user retention, which would increase lifetime value? A short-term revenue dip may be acceptable if retention improves.

6. **Consider running longer:** If the experiment was short, the revenue signal may stabilize. Revenue often has higher variance than clicks and needs more data.

### Q13: You are designing an experiment for a two-sided marketplace (drivers and riders). How do you handle interference?

**Answer:**

Interference (SUTVA violation) is the core challenge. If treated drivers are faster, they increase supply for control riders, contaminating the control group.

**Options:**

1. **Cluster randomization by geo:** Randomize entire cities or regions. Treatment cities get the new feature; control cities don't. This eliminates within-city interference but reduces power (few independent clusters) and introduces between-city heterogeneity.

2. **Switchback design:** Within a city, alternate between treatment and control over time intervals (e.g., treatment on even hours, control on odd hours). Assumes interference doesn't persist across time intervals. Analyze with appropriate time-series methods.

3. **Two-sided randomization:** Randomize independently on both sides. Analyze using the subset of interactions where both sides have the same assignment (e.g., treated driver picks up treated rider). This reduces sample size but eliminates cross-treatment contamination.

4. **Synthetic control / geo experiments:** Use untreated markets as synthetic controls for treated markets (see Causal Impact Analysis section).

### Q14: You've been running an experiment for 3 days and the CEO asks for preliminary results. The p-value is 0.03. Should you stop the experiment?

**Answer:**

**No.** Stopping at $p < 0.05$ after repeated checks dramatically inflates the false positive rate due to the **multiple comparisons over time** problem. If you check daily for 30 days, the probability of observing $p < 0.05$ at least once—even under the null—can exceed 25%.

**What to do:**

1. If you pre-committed to a fixed duration, explain that stopping early invalidates the statistical guarantee and the result could be a false positive.

2. If speed is critical, use a **sequential testing** framework (e.g., always-valid p-values via mSPRT or group sequential boundaries). These methods control the Type I error even under continuous monitoring.

3. Show the CEO the p-value trajectory over time. If it's monotonically decreasing and stable around 0.03, that's more convincing than a brief dip below 0.05 that might bounce back.

4. Present the confidence interval, not just the p-value. A $p = 0.03$ with a CI of $[0.001, 0.05]$ is barely significant and may not be practically meaningful.

---

## 6.4 Debugging & Failure-Mode Questions

### Q15: Your experiment shows a statistically significant result, but the sample ratio is 52% treatment / 48% control instead of the expected 50/50. What happened?

**Answer:** This is a **Sample Ratio Mismatch (SRM)**, which indicates a systematic problem with the experiment. Possible causes:

1. **Treatment causes crashes or redirects:** If the treatment variant has a bug that crashes the app, affected users' data is lost, reducing the treatment count. But if logging happens before the crash, the treatment count is inflated.

2. **Bot filtering:** If bot detection is applied post-assignment and bots have a non-uniform distribution across variants (e.g., bots always hash into certain buckets), one group loses more users.

3. **Caching or CDN issues:** Cached responses may not respect experiment assignment, causing some users to receive the wrong variant.

4. **Initialization bias:** If the experiment is ramped up gradually and the ramp logic has a bug, early users may be disproportionately assigned to one group.

**Impact:** SRM invalidates all results because the imbalance could be correlated with the outcome. The effect estimate is biased in an unknown direction.

**Action:** Do not interpret the experiment results. Debug the root cause. If it cannot be identified and fixed, discard the experiment and rerun.

### Q16: Your experiment shows no statistically significant effect ($p = 0.4$). Does this mean the treatment has no effect?

**Answer:** **No.** Absence of evidence is not evidence of absence. Possible explanations:

1. **Underpowered experiment:** If the sample size was too small to detect the true effect, the test fails to reject even when there is a real effect (Type II error). Check the power: what MDE was the experiment designed to detect? If MDE = 5% but the true effect is 1%, you're underpowered.

2. **High variance metric:** Revenue per user has enormous variance. Even a large effect may be buried in noise. Consider CUPED or switching to a lower-variance proxy metric.

3. **Dilution:** Not all assigned users were exposed to the treatment. If only 10% triggered, the ITT effect is 10× smaller than the triggered effect.

4. **True null:** The treatment genuinely has no effect. This is a valid conclusion only if the experiment was adequately powered.

**Report the confidence interval:** If the CI is $[-0.5\%, +0.8\%]$, you can conclude that the true effect is likely small. If the CI is $[-5\%, +8\%]$, you simply don't know.

### Q17: Your experiment shows a significant positive effect in the first week but the effect disappears by week 3. What happened?

**Answer:** This is likely a **novelty effect.** Users initially engage more with anything new (curiosity, exploration), inflating short-term metrics. As the novelty wears off, behavior returns to baseline.

**Diagnosis:**
- Plot the treatment effect over time (cumulative and per-day). If it decays monotonically, novelty effect is the likely cause.
- Segment by user tenure: new users (no novelty, since everything is new) vs. existing users (who notice the change).

**Mitigation:**
- Run the experiment longer (4+ weeks) to reach steady state.
- Use only new users as a cleaner test (they have no prior expectation).
- Model the decay curve and extrapolate the asymptotic effect.

The opposite pattern (initially no effect, then growing) suggests a **learning effect** — users need time to discover and adopt the feature.

### Q18: Two experiments running simultaneously show contradictory results. Experiment A improves metric X, and Experiment B degrades metric X. Both are statistically significant. What's going on?

**Answer:**

1. **Interaction between experiments:** If A and B are in the same layer, users should only be in one of them (mutual exclusivity), so this shouldn't happen. If they're in different layers, a user could be in both. The combined effect of A+B may differ from A alone or B alone. Check whether users in both experiments show a different pattern.

2. **Different user populations:** If A and B target different user segments (e.g., mobile vs. desktop), the contradictory results may be real for their respective populations.

3. **Temporal confound:** If A started two weeks before B, and an external event occurred between, the "contradiction" may be a time effect.

4. **Multiple testing:** With many experiments, some will show significant effects by chance. Check effect sizes, not just p-values.

**Resolution:** Look at the 2×2 matrix of (A assignment × B assignment) if they're in different layers. This is essentially a factorial analysis.

---

## 6.5 Follow-Up and Probing Questions

### Q19: "You mentioned CUPED reduces variance. Does it introduce bias?"

**Answer:** No. CUPED adjusts by a pre-experiment covariate $X$. Since $X$ is measured before randomization, it is independent of the treatment assignment: $E[X \mid T] = E[X \mid C]$. Therefore, subtracting $\theta(X_i - \bar{X})$ shifts both groups' means by the same amount, and the treatment effect $\hat{\tau} = \bar{\tilde{Y}}_T - \bar{\tilde{Y}}_C$ remains unbiased.

However, if $X$ is measured during the experiment (e.g., using day-1 behavior to adjust day-2 outcome), $X$ may be affected by treatment, and the adjustment becomes a "bad control" that biases the estimate.

### Q20: "What happens to power if you use a one-sided test instead of two-sided?"

**Answer:** Power increases because the critical value drops from $z_{\alpha/2}$ to $z_{\alpha}$. For $\alpha = 0.05$: two-sided threshold is 1.96, one-sided is 1.645. The sample size formula becomes:

$$
n = \frac{2\sigma^2(z_\alpha + z_\beta)^2}{\delta^2}
$$

This is smaller than the two-sided formula. However, one-sided tests cannot detect effects in the opposite direction, which is dangerous — a harmful treatment would not be flagged. Most experiment platforms use two-sided tests for safety.

### Q21: "How would you handle an experiment where the treatment affects the denominator of your ratio metric?"

**Answer:** If the treatment changes the number of sessions per user (denominator of sessions-per-user), a naive ratio comparison is biased. Solutions:

1. **Use the user as the unit of analysis:** Compute the metric at the user level (e.g., total clicks / total sessions per user), then compare user-level means.

2. **Delta method:** Properly model the variance of the ratio $R = \bar{Y} / \bar{X}$ using the delta method, which accounts for the covariance between numerator and denominator.

3. **Linearization:** Use the linearized metric $\tilde{Y}_i = Y_i - \hat{R} \cdot X_i$, which converts the ratio to a per-user metric that can be analyzed with standard methods.

### Q22: "You're running a bandit and after 10,000 pulls, Thompson Sampling has allocated 90% of traffic to arm B. Can you report that arm B is better with 95% confidence?"

**Answer:** **Not directly.** Bandit allocations do not produce valid frequentist confidence intervals because the allocation is adaptive — it depends on observed data. The probability of selecting arm B is not $P(\text{data} \mid H_0)$; it's $P(\text{data} \mid \text{posterior})$.

However, you can:
1. Report the Bayesian posterior probability $P(\mu_B > \mu_A \mid \text{data})$, which is well-calibrated for Thompson Sampling.
2. Use inverse propensity weighting (IPW) to construct unbiased effect estimates from adaptive data: weight each observation by $1/P(\text{assigned to this arm})$.
3. If you need frequentist guarantees, reserve a small fraction of traffic for uniform randomization (hybrid approach).

### Q23: "How does the parallel trends assumption differ from the assumption that treated and control groups have the same mean outcome?"

**Answer:** DiD does **not** require that the groups have the same mean outcome. It only requires that they would have **changed by the same amount** absent treatment. The groups can have different levels (e.g., US watches more TV than Canada), as long as their trends are parallel. This is weaker than requiring identical levels and is why DiD is useful — it handles time-invariant confounders (different baseline levels) automatically through the differencing.

### Q24: "Your synthetic control perfectly matches the treated unit in the pre-period. Is that always good?"

**Answer:** Not necessarily. **Overfitting** is a risk. If the donor pool has many units and the pre-treatment period is short, the optimization can find weights that perfectly match by fitting noise. The synthetic control will then diverge from the true counterfactual post-treatment, producing spurious treatment effects.

**Diagnostics:**
- Check that the weights are reasonable (not all weight on one obscure unit).
- Examine the number of donors vs. pre-treatment periods.
- Use cross-validation: fit on part of the pre-period, validate on the rest.
- Perform placebo tests in time: apply the method at a fake intervention date in the pre-period.

### Q25: "How would you decide between running a traditional A/B test vs. using a multi-armed bandit?"

**Answer:** The choice depends on the goal:

- **A/B test** if you need a precise, unbiased estimate of the treatment effect (for scientific understanding, institutional learning, building long-term knowledge), or if you need to measure multiple metrics (secondary metrics, guardrails), or if rewards are delayed.

- **Bandit** if you primarily want to maximize cumulative reward (minimize regret), if you have many variants to test (e.g., 50 headlines), if the cost of exploration is high (medical, high-revenue), or if you're doing continuous optimization without a fixed "launch" decision.

In practice, most product experimentation uses A/B tests because organizations need institutional knowledge about what works and why. Bandits are used for specific applications like ad creative rotation, recommendation exploration, or personalization.

### Q26: "An experiment has been running for 4 weeks. You discover that a bug caused 5% of users to not be logged. Is the experiment still valid?"

**Answer:** It depends on whether the logging failure is **correlated with treatment assignment or the outcome**.

- If the logging bug affected treatment and control equally (e.g., a random infrastructure failure), the experiment is valid but the effective sample size is 5% smaller. Power is marginally reduced but bias is not introduced.

- If the bug is correlated with treatment (e.g., the treatment causes a specific error that drops logs), the experiment has survivorship bias. The missing 5% may have had worse outcomes, making the treatment look better than it is.

**Diagnostics:** Check SRM after accounting for the missing data. Check whether the logging failure rate differs between treatment and control. If it does, the experiment is compromised.

### Q27: "You want to measure the long-term impact of a feature that was launched 6 months ago without a holdback group. What can you do?"

**Answer:** Without a holdback, you have no concurrent control group. Options:

1. **DiD or Synthetic Control:** If the feature was launched in some markets but not others, use unlaunched markets as controls (with DiD or synthetic control). If it was launched everywhere simultaneously, this doesn't work.

2. **Interrupted Time Series (ITS):** Model the pre-launch trend and extrapolate. The treatment effect is the deviation from the extrapolated trend. This is weak because any coincident event is confounded with the launch.

3. **Create a holdback now:** Remove the feature from a small random group. This is an "inverse A/B test." Compare the holdback group (now experiencing removal) to the rest. The effect is the negative of the feature's impact. Caution: removal effects may differ from launch effects (loss aversion, habit disruption).

4. **Instrumental variables:** If there's a natural source of variation in feature exposure (e.g., some users discovered it organically), use this as an instrument. Requires strong assumptions.

**Best practice going forward:** Always maintain a small persistent holdback group for important features.

---

## Summary Table: Methods Comparison

| Method | When to Use | Key Assumption | Main Strength | Main Weakness |
|--------|-------------|----------------|---------------|---------------|
| A/B Test | Randomization feasible, need causal estimate | SUTVA, random assignment | Unbiased causal effect | Exploration waste, fixed duration |
| Multi-Armed Bandit | Many variants, maximize reward | Stationarity, immediate rewards | Minimizes regret | Biased effect estimates |
| Full Factorial MVT | Few factors, interactions matter | Sufficient traffic for all combos | Detects interactions | Combinatorial explosion |
| Fractional Factorial | Many factors, screening | Higher-order interactions negligible | Efficient screening | Confounded effects |
| Difference-in-Differences | No randomization, parallel trends plausible | Parallel trends | Handles time-invariant confounders | Fails if trends diverge |
| Synthetic Control | One treated unit, many donors | Good pre-fit, no spillover | Data-driven counterfactual | One treated unit, overfitting risk |
| CausalImpact (BSTS) | Time series data, need uncertainty | Control series unaffected by treatment | Bayesian credible intervals | Model misspecification |
