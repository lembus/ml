# Machine Learning Monitoring

## 1. Motivation & Intuition

### Why This Topic Exists

In traditional software engineering, code is deterministic. If you write a function to add two numbers, it will work today, tomorrow, and ten years from now, assuming the underlying hardware or compiler doesn't change.

Machine Learning systems are different. They depend on **data**, which is non-deterministic and constantly evolving. A model trained to predict housing prices in 2020 will fail miserably in 2025 not because the code "broke," but because the world changed.

The core problem monitoring solves is **"Silent Failure."** When a web server crashes, you get a 500 error — it is loud and obvious. When an ML model fails, it often continues to return predictions, but they are increasingly wrong. Without monitoring, you might lose revenue or degrade user experience for weeks before noticing.

### Real-World Example

Imagine a credit card fraud detection system.

- **The Model:** Trained on transaction data from 2023. It knows that high-value transactions at 3 AM from a new IP address are suspicious.
- **The Change:** In 2024, a new iPhone releases. Thousands of legitimate users suddenly buy expensive phones at midnight (pre-order launch).
- **The Failure:** The model flags all these legitimate purchases as fraud (False Positives). The code didn't crash, but the system is failing.
- **The Monitor:** A monitoring system tracking "positive prediction rate" would see a spike from 1% to 15% and alert the team immediately.

---

## 2. Conceptual Foundations

### The Three Pillars of ML Monitoring

To prevent silent failures, we monitor three distinct layers.

#### 1. Data Quality (Input Layer)

This ensures the data entering the model matches the data the model was trained on.

- **Schema Validation:** Are we receiving a string where we expect an integer?
- **Missing Values:** Did a sensor break, causing 50% of our input feature `temperature` to become `NaN`?
- **Feature Drift:** Has the statistical distribution of the input changed? (e.g., User ages shifted from 20–30 to 40–50).

#### 2. Model Health (Internal Layer)

This checks the model's behavior, independent of the ground truth (which is often delayed).

- **Prediction Confidence:** Is the model becoming "unsure"? (e.g., A classifier usually outputs probabilities like 0.9 or 0.1, but now outputs 0.51 everywhere).
- **Calibration:** Does a predicted probability of 80% actually correspond to an 80% success rate?

#### 3. Performance (Output Layer)

This measures the actual business value and system efficiency.

- **Ground Truth Evaluation:** Accuracy, Precision, Recall (requires labels, often delayed).
- **System Metrics:** Latency (how fast is inference?) and Throughput (requests per second).

### Key Terms

- **Training-Serving Skew:** When the data seen during inference differs from the data used during training.
- **Concept Drift:** The relationship between input features ($X$) and the target variable ($Y$) changes (e.g., inflation changes what a "high price" means).
- **Data Drift (Covariate Shift):** The distribution of input features ($P(X)$) changes, but the relationship to the target might remain the same.

### Assumptions & Violations

- **Assumption:** The future resembles the past.
  - *Violation:* COVID-19 hits, changing consumer behavior instantly. Models trained on pre-COVID data fail.

- **Assumption:** Labels are available for evaluation.
  - *Violation:* In loan default prediction, you only know if a prediction was wrong months or years later. You must rely on proxy metrics (like prediction distribution) in the interim.

---

## 3. Mathematical Formulation

We rely on statistical distances to quantify "how different" today's data is from the training data.

### 3.1 Kullback-Leibler (KL) Divergence

Used to measure how one probability distribution $P$ diverges from a second expected probability distribution $Q$.

$$
D_{KL}(P \| Q) = \sum_{i} P(i) \ln\left(\frac{P(i)}{Q(i)}\right)
$$

- **$P(i)$:** The actual distribution (current production data).
- **$Q(i)$:** The reference distribution (training data).
- **Intuition:** If $P(i) \approx Q(i)$, the ratio is 1, $\ln(1) = 0$, so divergence is 0. If $P(i)$ is high where $Q(i)$ is low, the term explodes, signaling a drift.
- **Note:** KL Divergence is not symmetric ($D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$).

### 3.2 Population Stability Index (PSI)

PSI is the industry standard for drift because it is symmetric and scale-invariant. It is essentially a symmetrized version of KL Divergence.

$$
PSI = \sum_{i} (Actual\%_i - Expected\%_i) \times \ln\left(\frac{Actual\%_i}{Expected\%_i}\right)
$$

**Interpretation:**

- **PSI < 0.1:** No significant drift.
- **0.1 ≤ PSI < 0.2:** Moderate drift; investigate.
- **PSI ≥ 0.2:** Significant drift; retraining likely required.

### 3.3 Latency Percentiles

Averages hide outliers. In ML systems, the "tail" latency matters. We denote $P_{99}$ latency as the value below which 99% of requests fall.

$$
P_{99} = \inf \{ x \in \mathbb{R} : P(T \le x) \ge 0.99 \}
$$

Where $T$ is the random variable representing response time.

---

## 4. Worked Example: Calculating Feature Drift (PSI)

**Scenario:** You have a model that predicts car insurance risk. One feature is `Driver_Age`. You want to check if the age demographic of your users has drifted since training.

**Step 1: Bin the Reference Data (Training)**

Total Reference samples: 1000.

- Bin 1 (18–25): 200 users (20%)
- Bin 2 (26–50): 500 users (50%)
- Bin 3 (50+): 300 users (30%)

**Step 2: Bin the Current Data (Production)**

Total Production samples: 2000.

- Bin 1 (18–25): 200 users (10%) → *Fewer young drivers.*
- Bin 2 (26–50): 1200 users (60%) → *More middle-aged drivers.*
- Bin 3 (50+): 600 users (30%) → *Same proportion.*

**Step 3: Calculate PSI for each bin**

**Bin 1 (18–25):**

- $Actual\% = 0.10$, $Expected\% = 0.20$
- $Diff = 0.10 - 0.20 = -0.10$
- $\ln(0.10 / 0.20) = \ln(0.5) \approx -0.693$
- $Contribution = (-0.10) \times (-0.693) = \mathbf{0.0693}$

**Bin 2 (26–50):**

- $Actual\% = 0.60$, $Expected\% = 0.50$
- $Diff = 0.60 - 0.50 = 0.10$
- $\ln(0.60 / 0.50) = \ln(1.2) \approx 0.182$
- $Contribution = 0.10 \times 0.182 = \mathbf{0.0182}$

**Bin 3 (50+):**

- $Actual\% = 0.30$, $Expected\% = 0.30$
- $Diff = 0$
- $Contribution = \mathbf{0}$

**Step 4: Sum contributions**

$$
PSI = 0.0693 + 0.0182 + 0 = \mathbf{0.0875}
$$

**Conclusion:** The PSI is 0.0875. This is $< 0.1$, so the drift is considered insignificant. No alert is triggered.

---

## 5. Relevance to Machine Learning Practice

### When to Use

- **High-Stakes Environments:** Financial, medical, or security applications where errors are costly.
- **Dynamic Environments:** E-commerce, news recommendations, or social media where trends shift daily.

### System Design: The Feedback Loop

In a mature MLOps system, monitoring is the trigger for the lifecycle:

1. **Monitor** detects Drift (PSI > 0.2).
2. **Alert** notifies the Data Science team.
3. **Analysis** confirms the drift is real (not a data pipeline bug).
4. **Retraining** is triggered on newer data.
5. **Evaluation** confirms the new model performs better on current data.
6. **Deployment** replaces the old model.

### Trade-offs

- **Computational Cost:** Calculating PSI/KL for every feature on every request is expensive. *Solution:* Sampling. Compute drift on a random 10% of requests or batch process once a day.
- **False Alerts:** If thresholds are too tight, you get "alert fatigue." If too loose, you miss failures.
- **Proxy vs. Reality:** Monitoring prediction distribution is a *proxy* for accuracy. The model could be confidently wrong (distribution looks stable, but accuracy dropped).

---

## 6. Common Pitfalls & Misconceptions

### 1. The "Set It and Forget It" Fallacy

**Mistake:** Deploying a model and assuming it works until a user complains.
**Why:** Models degrade naturally. Without active monitoring, performance *will* drop.
**Fix:** Automated dashboards and alerting rules.

### 2. Confusing Data Issues with Model Issues

**Mistake:** Seeing performance drop and immediately retraining the model.
**Why:** The issue might be an upstream data pipeline bug (e.g., `Age` is now being sent in months instead of years). Retraining won't fix a broken pipeline.
**Fix:** Monitor **Data Quality** (schema/types) before monitoring **Model Performance**.

### 3. Monitoring Averages Only

**Mistake:** "Average latency is 200ms, we are fine."
**Why:** If 1% of requests take 10 seconds, you are losing users.
**Fix:** Monitor $P_{95}$ and $P_{99}$ latency.

---

## 7. Interview Questions

### Foundational Questions

**Q1: What is the difference between Concept Drift and Data Drift?**

- **Data Drift (Covariate Shift):** The distribution of input features $P(X)$ changes. Example: Users get younger. The mapping $X \to Y$ is still valid, but we are seeing inputs from a region of the feature space we might not have trained well on.
- **Concept Drift:** The relationship between inputs and outputs $P(Y|X)$ changes. Example: Before 2020, "buying a mask" was rare/suspicious ($X$). After 2020, it became normal behavior. The same input $X$ now implies a different label $Y$.

**Q2: Why can't we just monitor Accuracy?**

Accuracy requires "Ground Truth" labels. In production, labels are often:

1. **Delayed:** (e.g., waiting 30 days to see if a user churns).
2. **Expensive:** (e.g., requiring human review).
3. **Impossible:** (e.g., recommending a video the user didn't click — we don't know if they *would* have liked it).

Therefore, we must monitor proxy metrics like feature drift and prediction distribution.

### Mathematical Questions

**Q3: Explain KL Divergence. What happens if the production bin has probability 0 but the training bin has probability > 0?**

- Formula: $D_{KL} = \sum P(i) \ln(P(i)/Q(i))$.
- If production $P(i) > 0$ and training $Q(i) = 0$, we divide by zero, and KL divergence goes to infinity. This indicates the model is seeing a value in production it *never* saw in training (a new category).
- If production $P(i) = 0$ and training $Q(i) > 0$, the contribution is 0 (by convention $0 \ln 0 = 0$). This means a category disappeared, which increases distance but doesn't explode the metric.
- To handle $Q(i)=0$, we often apply smoothing (adding a small $\epsilon$ to all bins).

**Q4: Why do we use PSI instead of a simple T-test or KS-test?**

- **Scale Invariance:** Statistical tests like T-tests are sensitive to sample size. With "Big Data" (millions of rows), even a tiny, practically irrelevant difference will be "Statistically Significant" (p-value < 0.05).
- **Stability:** PSI quantifies the *magnitude* of the shift in a way that correlates well with model degradation, regardless of whether the sample size is 10k or 10M.

### Applied & Scenario Questions

**Q5: You are monitoring a recommender system. The "Average Predicted Probability" of a user clicking an ad drops from 2% to 1% overnight. What are your hypotheses?**

Investigation order:

1. **Data Pipeline Break:** Is a key feature (like `User_History`) missing or null? (Check Missing Value rates).
2. **Technical Deployment:** Was a new model version deployed yesterday? (Check version logs).
3. **Traffic Shift:** Did a marketing campaign bring in low-intent users (bots/crawlers)? (Check Traffic Source distribution).
4. **Seasonality:** Is it a holiday or weekend?
5. **External Change:** Did the UI change, making ads less visible?

**Q6: How do you monitor a model that predicts values in the range [0, 1] but isn't a probability (e.g., a regression model predicting a score)?**

- **Outlier Detection:** Track the min/max and standard deviation. If the model starts predicting 1.5 or -0.5, something is wrong.
- **Distribution Matching:** Use PSI on the output histograms.
- **Smoothness:** If inputs change slightly, outputs should change slightly (Lipschitz continuity). Large jumps suggest instability.

### Debugging & Failure Modes

**Q7: Your drift monitor alerts that feature `Income` has high drift (PSI = 0.5). You retrain the model on the new data, but the new model performs *worse*. Why?**

- **Bad Labels:** The new data might have incorrect labels (e.g., a bug in the logging of user clicks).
- **Non-Stationary Data:** The data might be drifting too fast (oscillating). By the time you retrain and deploy, the world has shifted back.
- **Feedback Loop:** The model itself might be influencing the data generation (e.g., the model stops showing loans to high-income people, so the new training set has no high-income people).
- **Data Quality Issue:** The "Drift" was actually a data bug (e.g., currency changed from USD to JPY). Retraining the model to learn "JPY is normal" is wrong; you should have fixed the currency conversion bug.

**Q8: How do you fix the "Feedback Loop" issue mentioned above?**

- **Exploration/Exploitation:** Reserve a small percentage of traffic (epsilon-greedy) to receive random recommendations or predictions from a baseline model. This ensures you continue to capture ground truth labels for regions of the feature space the current model dislikes.
- **Positional Bias Correction:** In ranking, log the position where the item was shown and use it as a feature or weight during training (Inverse Propensity Weighting).
