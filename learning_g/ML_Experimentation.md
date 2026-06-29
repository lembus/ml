# Experimentation: A/B Testing, Bandits, and Causal Inference

## 1. Motivation & Intuition

### Why Do We Experiment?

In software and machine learning, we often have ideas we *think* will improve a product: changing a button color, deploying a new recommendation algorithm, or altering a pricing tier. However, intuition is often wrong.

**Experimentation** (or Controlled Testing) is the scientific method applied to product development. It allows us to move from "I think X is better" to "I have statistical evidence that X causes an improvement in Y."

### The Core Problem: Correlation vs. Causation

Imagine you launch a new "Dark Mode" on your app on Friday. Traffic spikes on Saturday.

- **Did Dark Mode cause the spike?**
- **Or is it just because it's the weekend?**

If you simply observe data after the launch, you cannot separate external factors (seasonality, marketing campaigns) from the impact of your change. Experimentation solves this by creating parallel universes:

1. **Group A (Control):** The world *without* the change.
2. **Group B (Treatment):** The world *with* the change.

By keeping everything else identical (time, user demographics, external events) and randomizing who goes into which group, any difference in outcome is mathematically attributable to your change.

---

## 2. A/B Testing Design

### 2.1 Randomization

Randomization is the heart of experimentation. It ensures that the Control and Treatment groups are statistically comparable before the intervention.

#### Unit of Analysis (Randomization Unit)

This is the "atom" you are splitting.

- **User-level:** Most common. User ID 123 sees A; User ID 124 sees B. Good for consistent user experience.
- **Session-level:** Every time you open the app, you might see A or B. Good for anonymous traffic, but bad if the change requires learning (e.g., a UI change).
- **Request-level:** Good for backend changes (e.g., latency testing of a new DB query).
- **Geo-level (Switchback):** Everyone in New York sees A; everyone in Boston sees B. Used when network effects exist (e.g., Uber surge pricing).

#### Stratification & Blocking

Simple randomization (flipping a coin for every user) works well at scale ($N > 10{,}000$). However, for small samples, you might accidentally get all iPhone users in Group A.

- **Stratification:** You divide the population into subgroups (strata) based on key traits (e.g., Country, Device Type) and *then* randomize within those strata. This guarantees balanced representation.

### 2.2 Sample Size & Duration

We cannot run a test forever. We need to know *when* we have enough data to make a decision.

#### Key Concepts

1. **Minimum Detectable Effect (MDE):** The smallest improvement that matters to the business. If the conversion rate increases by 0.0001%, do we care? Probably not. We might set MDE at 1%.
2. **Statistical Power ($1 - \beta$):** The probability of detecting an effect *if it actually exists*. Usually set to 80%.
3. **Significance Level ($\alpha$):** The probability of saying there is an effect when *there isn't one* (False Positive). Usually set to 5%.

#### Duration Hazards

- **Seasonality:** A test running only on Monday/Tuesday might not apply to weekend users.
- **Primacy/Novelty Effects:** Users might click a button just because it's new (Novelty), or hate it because it's different (Primacy). Tests often need to "burn in" for a week to let these effects stabilize.

---

## 3. Mathematical Formulation

### 3.1 Hypothesis Testing

We formally state two hypotheses:

- **Null Hypothesis ($H_0$):** The treatment has no effect ($p_{treatment} - p_{control} = 0$).
- **Alternative Hypothesis ($H_1$):** The treatment has an effect ($p_{treatment} - p_{control} \neq 0$).

### 3.2 The Z-Test (for Proportions)

If we are measuring a Conversion Rate (Click-through rate, Purchase rate), we use a Z-test for the difference of two proportions.

Let:

- $N_c, N_t$ = Sample size of Control and Treatment.
- $X_c, X_t$ = Number of conversions in Control and Treatment.
- $\hat{p}_c = \frac{X_c}{N_c}, \quad \hat{p}_t = \frac{X_t}{N_t}$ = Observed conversion rates.

**The Pooled Probability ($\hat{p}$):**

$$
\hat{p} = \frac{X_c + X_t}{N_c + N_t}
$$

**Standard Error (SE):**

$$
SE = \sqrt{ \hat{p}(1-\hat{p}) \left( \frac{1}{N_c} + \frac{1}{N_t} \right) }
$$

**Z-Score:**

$$
Z = \frac{\hat{p}_t - \hat{p}_c}{SE}
$$

**Intuition:** The Z-score tells us how many standard deviations the difference is away from zero. If $|Z| > 1.96$, we reject the Null Hypothesis (at 95% confidence).

### 3.3 Sample Size Calculation

To find the required sample size per group ($n$), typically utilizing a formula derived from the power function:

$$
n \approx \frac{2 \sigma^2 (Z_{\alpha/2} + Z_{\beta})^2}{\delta^2}
$$

Where:

- $\sigma^2$: Variance of the metric (e.g., $p(1-p)$ for proportions).
- $\delta$: The Minimum Detectable Effect (MDE).
- $Z_{\alpha/2} \approx 1.96$ (for $\alpha=0.05$).
- $Z_{\beta} \approx 0.84$ (for 80% power).

**Key Takeaway:** Sample size grows quadratically as MDE ($\delta$) shrinks. Detecting a tiny change requires massive data.

---

## 4. Multi-Armed Bandits (MAB)

### Motivation

A/B testing is "Explore then Exploit." You spend weeks testing (Exploration), losing money on the bad variant, before switching 100% to the winner (Exploitation). **Multi-Armed Bandits** optimize *while* testing. They dynamically shift traffic to the winning variation.

### 4.1 Epsilon-Greedy ($\epsilon$-greedy)

- Flip a coin with probability $\epsilon$ (e.g., 10%).
- If heads: Choose a random arm (Explore).
- If tails: Choose the arm with the highest current average reward (Exploit).

**Pros:** Simple to implement.
**Cons:** Continues to explore bad arms forever unless you decay $\epsilon$.

### 4.2 Thompson Sampling (Bayesian)

Treat the reward probability of each arm as a probability distribution (usually Beta).

1. Sample a value from each arm's distribution.
2. Pick the arm with the highest sampled value.
3. Update that arm's distribution with the result (Success/Failure).

**Intuition:** If we are unsure about an arm, its distribution is wide (high variance). We might sample a high number by chance, leading us to try it. If we are sure an arm is bad, its distribution is narrow and low, so we rarely pick it.

### 4.3 Upper Confidence Bound (UCB)

Pick the arm that has the highest potential upside.

$$
Score_j = \mu_j + \sqrt{\frac{2 \ln t}{n_j}}
$$

- $\mu_j$: Average reward of arm $j$.
- $\sqrt{\dots}$: The "Optimism" term. It grows if we haven't picked arm $j$ in a long time ($n_j$ is small), forcing us to explore it.

---

## 5. Causal Impact Analysis

Sometimes you **cannot** randomize (e.g., a Super Bowl TV ad, or a law change in one state).

### 5.1 Difference-in-Differences (DiD)

Compares the change in the treatment group over time to the change in a control group over time.

- **Assumption (Parallel Trends):** Without the treatment, the two groups would have moved in parallel.
- **Math:**

$$
\text{Impact} = (Y_{Treat, Post} - Y_{Treat, Pre}) - (Y_{Control, Post} - Y_{Control, Pre})
$$

### 5.2 Synthetic Control

If no single "Control" group exists (e.g., California passes a law, no other state is exactly like CA), we create a **Synthetic California**.

- We use a weighted average of other states ($0.3 \times \text{NY} + 0.2 \times \text{TX} + \dots$) to match California's pre-intervention data perfectly.
- The divergence after the intervention is the causal impact.

---

## 6. Worked Example

**Scenario:** We want to test if changing a "Buy Now" button from Blue to Green increases Click-Through Rate (CTR). Current CTR is 10%. We want to detect a lift to 10.5% (MDE relative = 5%, absolute = 0.5%).

**Step 1: Design**

- **Unit:** User_ID.
- **Power:** 80%, **Alpha:** 5%.
- Using a calculator, we need $N \approx 100{,}000$ users per variant.

**Step 2: Execution**

- We run the test for 2 weeks.
- **Control (Blue):** 100,000 users, 10,100 clicks ($\hat{p}_c = 0.101$).
- **Treatment (Green):** 100,000 users, 10,400 clicks ($\hat{p}_t = 0.104$).

**Step 3: Analysis**

- Pooled $\hat{p} = \frac{10100+10400}{200000} = 0.1025$.
- $SE = \sqrt{0.1025(1-0.1025)\left(\frac{1}{100000} + \frac{1}{100000}\right)} \approx 0.00135$.
- Difference $d = 0.104 - 0.101 = 0.003$.
- $Z = \frac{0.003}{0.00135} \approx 2.22$.

**Conclusion:** Since $2.22 > 1.96$, the result is statistically significant. The Green button works.

---

## 7. Interview Questions

### Foundational Questions

**Q: Explain the difference between A/B Testing and Multi-Armed Bandits. When would you use one over the other?**

A/B testing aims for *statistical significance* and learning. It creates a fixed period of regret (loss) to gain certainty. Use it for major product decisions (UI overhaul, pricing) where long-term certainty is required. MAB aims to *minimize regret*. It dynamically adjusts traffic to the winner. Use it for short-term optimization (news headlines, ad selection) or continuous learning where the "best" answer changes over time.

**Q: What is the "Peeking Problem" in A/B testing?**

Checking the results of a test continuously (e.g., every hour) and stopping as soon as it looks significant. This inflates the False Positive rate (Alpha) drastically. If you peek often enough, you *will* eventually find a "significant" result by random chance. The fix is to fix the sample size in advance and do not stop early, or use Sequential Testing methods (like SPRT).

### Mathematical Questions

**Q: Derive the relationship between Sample Size and Minimum Detectable Effect (MDE).**

Start with the Z-statistic formula: $Z = \frac{\delta}{SE}$. To reject $H_0$, we need the effect $\delta$ to be larger than the noise defined by $\alpha$ and $\beta$:

$$
\delta \approx (Z_{\alpha/2} + Z_{\beta}) \cdot \sqrt{\frac{2\sigma^2}{n}}
$$

Squaring both sides:

$$
\delta^2 \propto \frac{1}{n} \implies n \propto \frac{1}{\delta^2}
$$

**Implication:** If you want to detect an effect that is half as small, you need 4 times the data.

**Q: How do you calculate the confidence interval for a lift?**

$$
CI = (\hat{p}_t - \hat{p}_c) \pm Z_{\alpha/2} \cdot SE
$$

If the interval includes 0 (e.g., [-0.1%, +0.5%]), the result is not statistically significant.

### Applied & System Design Questions

**Q: You are testing a new ranking algorithm for Netflix. What metrics do you choose?**

This tests your ability to distinguish between proxy metrics and guardrail metrics.

- **Primary Metric (Goal):** Retention Rate (hard to measure quickly) or Total Watch Time (good proxy).
- **Secondary Metrics:** Click-through rate (CTR), Play start delay.
- **Guardrail Metrics (Do not harm):** Latency (did the model slow down the app?), Cancellation rate, Support tickets.

**Q: We ran an A/A test (Control vs Control) and the p-value was 0.01. What does this mean?**

An A/A test should theoretically show no difference. A p-value of 0.01 implies a "significant difference."

- **Possibility 1 (Bad Luck):** 1% of the time, this happens by chance. Run it again.
- **Possibility 2 (System Bug):** The randomization logic is broken (e.g., hashing function bias).
- **Possibility 3 (Data Pipeline):** Events from one group are being logged differently or dropped.
- **Action:** Do not proceed to A/B testing until the A/A test passes (uniform p-value distribution).

### Debugging & Failure Modes

**Q: An experiment showed a 5% lift in clicks, but overall revenue dropped. What happened?**

This is "Cannibalization" or a "Quality vs. Quantity" trade-off. The change might have made it easier to click (e.g., clickbait), attracting low-intent users who don't buy anything. Or, it cannibalized clicks from a higher-value area (e.g., users clicked "Free Trial" instead of "Buy Now"). The lesson: always optimize for the metric closest to business value (Revenue), not just user behavior (Clicks).

**Q: Network Effects (Interference). You are testing a driver-side change for Uber. Why might standard randomization fail?**

If you treat Driver A (give them a bonus) but not Driver B, Driver A works more. This steals rides from Driver B, making Driver B look artificially worse. This violates the SUTVA assumption (Stable Unit Treatment Value Assumption). Solutions include Switchback Testing (time-based split: 9–10 AM everyone is A, 10–11 AM everyone is B) or Cluster Randomization (city-based split).

**Q: How do you handle outliers in a metric like "Revenue per User"?**

Revenue is often heavy-tailed (whales spend \$10k, most spend \$0). One whale in the Treatment group can generate a false positive. Fixes include:

1. **Capping:** Cap values at the 99th percentile.
2. **Log-transform:** Analyze $\log(\text{Revenue})$ to reduce variance.
3. **Non-parametric tests:** Use Mann-Whitney U test instead of T-test (rank-based, robust to outliers).

### Follow-up Probes

- "How does your answer change if the metric is rare (e.g., 'Click on Delete Account button')?" — Use Poisson distribution or oversampling techniques.
- "What if we can't wait 2 weeks for the sample size?" — Increase MDE (accept we can only find big wins), or use variance reduction techniques like CUPED.
