# Robust Evaluation Under Shift

## 1. Motivation & Intuition

### Why This Matters

In traditional statistics and introductory machine learning, we make a massive, comforting assumption: the future will look exactly like the past. We assume that the data used to train a model comes from the exact same "world" as the data the model will encounter in the real world.

In reality, the world changes.

Imagine you build a system to recognize cars for a self-driving vehicle. You train it using millions of images taken in California (sunny, dry, wide roads). If you deploy this car in Norway during winter (snowy, dark, narrow roads), the model will likely fail catastrophically. The underlying concept of a "car" hasn't changed, but the **distribution** of the input data has shifted.

**Robust Evaluation** is the discipline of stressing a model to ensure it works not just on data it has seen before, but on data that is different, difficult, or adversarial. It is the bridge between a model that works in a Jupyter notebook and a model that works in production.

### Real-World Connection

- **Healthcare:** A pneumonia detection model trained on X-rays from Hospital A might fail at Hospital B because Hospital B uses a different machine brand that adds a subtle visual tint (Covariate Shift).
- **Finance:** A credit risk model trained on 2019 data (economic boom) will fail to predict defaults in 2020 (pandemic/recession) because the relationship between income and default risk has changed (Concept Drift).

---

## 2. Conceptual Foundations

To understand shift, we must first understand the statistical components of supervised learning.

### The Components

A machine learning problem consists of two random variables:

1. **$X$ (Features):** The input data (e.g., the pixels of an image).
2. **$Y$ (Labels):** The target output (e.g., "Cat" or "Dog").

Their relationship is governed by the **Joint Distribution**, $P(X, Y)$. Using probability rules, this can be broken down in two ways:

1. $P(X, Y) = P(Y|X) \cdot P(X)$ (Causal direction: Features generate labels).
2. $P(X, Y) = P(X|Y) \cdot P(Y)$ (Anti-causal: Labels generate features).

### The I.I.D. Assumption

Standard ML assumes data is **Independent and Identically Distributed (I.I.D.)**. This means the training set ($P_{train}$) and the test/production set ($P_{test}$) are identical. When $P_{train} \neq P_{test}$, we have **Distribution Shift**.

### Types of Distribution Shift

#### 1. Covariate Shift (Input Shift)

- **Definition:** The distribution of inputs $P(X)$ changes, but the relationship between inputs and labels $P(Y|X)$ stays the same.
- **Example:** Your training data has users aged 20–30. Your production data has users aged 50–60. The rule "higher income → higher spend" (the $P(Y|X)$) might still hold, but the inputs are different.

#### 2. Label Shift (Prior Probability Shift)

- **Definition:** The distribution of labels $P(Y)$ changes, but the conditional distribution $P(X|Y)$ stays the same.
- **Example:** A medical diagnostic tool. In training (normal times), 1% of patients have the flu. In production (flu season), 20% have the flu. The symptoms of the flu ($X|Y$) haven't changed, but the prevalence ($Y$) has.

#### 3. Concept Drift

- **Definition:** The relationship between inputs and outputs $P(Y|X)$ changes. This is the hardest shift to handle.
- **Example:** In 2000, a house with 3 bedrooms cost \$100k. In 2025, the same house ($X$) costs \$500k ($Y$). The "concept" of pricing has changed due to inflation/market dynamics.

---

## 3. Mathematical Formulation

We denote the **Source Domain** (Training) as $S$ and the **Target Domain** (Test/Production) as $T$.

### 3.1 Covariate Shift

$$P_S(Y|X) = P_T(Y|X) \quad \text{but} \quad P_S(X) \neq P_T(X)$$

To correct for this during training, we often use **Importance Weighting**. We weight the training loss for sample $x$ by the ratio of densities:

$$\beta(x) = \frac{P_T(x)}{P_S(x)}$$

If a data point is rare in training but common in production, $\beta(x)$ is high, telling the model: "Pay extra attention to this example!"

### 3.2 Label Shift

$$P_S(X|Y) = P_T(X|Y) \quad \text{but} \quad P_S(Y) \neq P_T(Y)$$

### 3.3 Detection Methods

#### A. The Two-Sample Test (Statistical Approach)

We treat the training data $X_S$ and test data $X_T$ as two samples. We want to accept or reject the null hypothesis $H_0: P_S = P_T$.

**Kolmogorov-Smirnov (KS) Test:**

Used for 1-dimensional features. It measures the maximum distance between the Cumulative Distribution Functions (CDFs) of the two distributions.

$$D_{KS} = \sup_x | F_S(x) - F_T(x) |$$

If $D_{KS}$ is large (p-value < 0.05), the features have drifted.

#### B. Maximum Mean Discrepancy (MMD)

Used for high-dimensional data. Instead of comparing densities directly (which is hard in high dimensions), we map data into a "Kernel feature space" (a higher-dimensional representation) and compare their means (averages).

$$\text{MMD}^2(S, T) = \left\| \frac{1}{n_S} \sum_{i=1}^{n_S} \phi(x_S^{(i)}) - \frac{1}{n_T} \sum_{j=1}^{n_T} \phi(x_T^{(j)}) \right\|^2$$

If the centers (means) of the two datasets in this abstract space are far apart, the distributions are different.

#### C. The Domain Classifier (Model-Based Approach)

This is the most practical method for complex data (like images).

1. Take training data and label it $0$.
2. Take production data and label it $1$.
3. Train a binary classifier to distinguish between them.
4. **Result:** If the classifier gets accuracy $\approx 0.5$ (random guessing), the distributions are indistinguishable (No shift). If accuracy is high (e.g., $0.9$), the distributions are distinct (Shift detected).

---

## 4. Worked Example: Credit Card Fraud Detection

A scenario involving **Covariate Shift**.

**Scenario:** You build a fraud detection model for a bank.

- **Training Data (Source):** Transactions from 2020. Mostly in-person swipes and standard e-commerce.
- **New Data (Target):** Transactions from 2024. A massive surge in "Tap-to-Pay" and mobile wallet transactions.

### Step 1: Diagnosis

You notice the model's performance is degrading. You suspect the inputs have changed.

### Step 2: Domain Classifier Test

You construct a dataset:

- Class 0: 10,000 transaction records from 2020.
- Class 1: 10,000 transaction records from 2024.
- Features: `transaction_amount`, `merchant_category`, `time_of_day`, `is_mobile`.

You train a Random Forest.

**Result:** The ROC-AUC is **0.85**. This is high. The model can easily tell 2020 data from 2024 data.

**Feature Importance:** The model indicates `is_mobile` and `transaction_amount` are the top features driving this distinction. This confirms **Covariate Shift**.

### Step 3: Evaluation Strategy

You cannot trust standard accuracy. You must perform **Subpopulation Analysis**. You split your test set into groups:

- Group A: Standard Card Swipes.
- Group B: Mobile Wallet.

**Results:**

- Group A Accuracy: 98%
- Group B Accuracy: 60%

The global accuracy might still look okay (e.g., 90%) if Group B is small, but the model is failing on the new/growing segment.

### Step 4: Mitigation

You re-train the model. Since you don't have many *labeled* frauds for 2024 yet, you use **Importance Weighting**. You weigh the few mobile transactions you have in the training set higher, so the model prioritizes learning that pattern.

---

## 5. Relevance to Machine Learning Practice

### When to Use Robust Evaluation

- **High-Risk Deployments:** Medical diagnosis, autonomous driving, loan approval.
- **Non-Stationary Environments:** Stock trading, fraud detection, social media recommendation (trends change daily).

### Evaluation Strategies

1. **Temporal Validation (Time-Split):** Never do a random 80/20 split for time-series data. Train on Jan–Sept, Validate on Oct, Test on Nov. This mimics the production constraint of predicting the future.
2. **Worst-Case Performance:** Do not optimize for average accuracy. Optimize for the accuracy of the worst-performing subgroup (e.g., performance on minority demographics).
3. **Stress Testing (OOD):** Intentionally feed the model noise, rotated images, or data from a different country to see how brittle it is.

### Trade-offs

- **Robustness vs. Accuracy:** A highly robust model (conservative) might have lower accuracy on the "easy" data than a specialized (overfitted) model.
- **Cost:** Continuous monitoring and retraining require significant compute and engineering infrastructure.

---

## 6. Common Pitfalls & Misconceptions

**"My model has 99% accuracy, so it's fine."**
If the test set has the same biases as the training set, the metric is an illusion. You must test on Out-of-Distribution (OOD) data.

**Confusing Shift Types:**
Treating Concept Drift like Covariate Shift. If $P(Y|X)$ changes (Concept Drift), re-weighting input samples won't help. You *must* get new labels and retrain.

**Monitoring Features Individually:**
Checking if the mean of Feature A changed, then Feature B. In reality, correlations break. Feature A and B might have the same means, but their correlation might have flipped. Multivariate tests (like the Domain Classifier) catch this.

---

## 7. Interview Questions

### Foundational Questions

**Q1: Explain the difference between Covariate Shift and Concept Drift. Give a concrete example of each.**

- **Covariate Shift** is when the distribution of inputs $P(X)$ changes, but the logic mapping inputs to labels $P(Y|X)$ remains constant. *Example:* A face recognition model trained on adults is tested on children. The "face" concept is the same, but the input features (size, proportions) differ.
- **Concept Drift** is when the mapping $P(Y|X)$ changes. *Example:* A housing price model. A house with features $X$ (3 bed, 2 bath) costs \$200k in 2010 but \$400k in 2020. The input $X$ is identical, but the label $Y$ has changed.

**Q2: What is the I.I.D. assumption, and why is it dangerous in production systems?**

I.I.D. stands for Independent and Identically Distributed. It assumes training and test data are sampled from the same probability distribution. It is dangerous because production data almost always drifts over time (non-stationary), leading to silent performance degradation where the model is confident but wrong.

### Mathematical Questions

**Q3: How does Importance Weighting correct for Covariate Shift? Write down the weight formula.**

Importance weighting corrects the bias in the training set to make it look like the test set. We minimize the weighted loss: $\sum \beta(x_i) L(f(x_i), y_i)$.

The weight is the density ratio:

$$\beta(x) = \frac{P_{target}(x)}{P_{source}(x)}$$

Intuitively, if a sample is twice as likely to appear in the target domain than the source, we count its loss twice as much during training.

**Q4: Explain how a Domain Classifier is used to detect shift. What is the objective function being optimized?**

We train a binary classifier $D(x)$ to distinguish Source ($y_D=0$) from Target ($y_D=1$). The objective is standard binary cross-entropy:

$$\mathcal{L} = - \mathbb{E}_{x \sim P_T} [\log D(x)] - \mathbb{E}_{x \sim P_S} [\log (1 - D(x))]$$

If the trained classifier achieves an AUC significantly $> 0.5$, the distributions are separable, indicating shift.

### Applied Questions

**Q5: You are building a news classification system. How would you design the evaluation to ensure robustness?**

1. **Temporal Split:** Train on news from Jan–June, test on July–Aug. This tests robustness to new events/topics.
2. **Subgroup Analysis:** Evaluate performance per topic (Sports vs. Politics) and per source (formal outlets vs. tabloids). Ensure the model isn't failing on specific subgroups.
3. **Adversarial/Stress Test:** Test on text with typos or slang to check robustness to noisy inputs.

**Q6: We trained a churn prediction model, and the accuracy dropped from 90% to 70% in two months. How do you debug this?**

1. **Check Data Quality:** Was there an upstream engineering change? (e.g., `NaN` values, unit changes).
2. **Detect Shift Type:**
   - Compare feature statistics (Mean, Variance) and run a Domain Classifier to check for Covariate Shift.
   - Check the label distribution ($P(Y)$). Did churn rates spike globally? (Label Shift).
   - Check the error analysis. Are users with the *same* features now churning when they didn't before? (Concept Drift).
3. **Action:** If Covariate Shift, retrain or reweight. If Concept Drift, you need new labels immediately for retraining.

### Debugging & Failure Modes

**Q7: You applied Importance Weighting, but your model performance got *worse*. Why?**

1. **Extreme Weights:** If $P_S(x)$ is very small (near zero) for some points where $P_T(x)$ is high, the weights $\beta(x)$ explode (variance explosion). The model focuses entirely on a few outliers. **Fix:** Clip the weights.
2. **Estimation Error:** The density ratio $\beta(x)$ must be estimated (usually by a model). If the density estimation is poor, the weights are garbage.
3. **Concept Drift:** Weighting only fixes Covariate Shift. If Concept Drift is present, weighting the "right" inputs to the "wrong" old labels hurts performance.

**Q8: Your Domain Classifier has an AUC of 0.5 (random), but your main model performance is still dropping on the test set. What happened?**

This suggests **Concept Drift** ($P(Y|X)$ changed). The Domain Classifier only looks at $X$. It sees that the input distribution hasn't changed (AUC 0.5). However, the relationship to the label *has* changed. Since the Domain Classifier ignores $Y$, it cannot detect this. You need to monitor the conditional distribution or the final model's loss.
