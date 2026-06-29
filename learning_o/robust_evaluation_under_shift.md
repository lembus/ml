# Robust Evaluation Under Shift

---

## 1. Motivation & Intuition

### Why Does This Topic Exist?

Every machine learning model is trained on a specific dataset drawn from a specific distribution at a specific point in time. The implicit contract of standard evaluation—split your data into train/test, measure accuracy on the test set, and declare victory—rests on a fragile assumption: **the data the model encounters in production will look like the data it was evaluated on.** In practice, this assumption is almost always violated.

Consider a concrete scenario. You train a credit-default model on loan applications from 2018–2020. The model achieves 95% AUC on a held-out test set from the same period. You deploy it in March 2020. Within weeks, a global pandemic restructures the economy. The income distributions of applicants change (covariate shift). The base rate of defaults spikes (label shift). The very relationship between income and default probability changes because government stimulus alters repayment behavior (concept drift). Your 95% AUC is now meaningless—not because the model was poorly built, but because the world moved out from under it.

**Robust evaluation under shift** is the discipline of designing evaluation protocols that anticipate, detect, and quantify how model performance degrades when the data distribution changes. It answers the question: *"How will this model behave when the world is different from training time?"*

### A Simple Analogy

Imagine you are testing a bridge. Standard evaluation is like testing the bridge under normal traffic on a sunny day. Robust evaluation is asking: What happens during a hurricane? What if a convoy of heavy trucks crosses simultaneously? What if the ground beneath one pylon erodes? You are not just measuring performance—you are **stress-testing** the system's behavior under conditions it was not explicitly designed for.

### Why Standard Evaluation Fails

Standard i.i.d. (independent and identically distributed) evaluation measures average-case performance on a distribution that matches training. This fails in several ways:

1. **It hides subpopulation failures.** A model with 92% overall accuracy might have 60% accuracy on a minority demographic. Averaging masks this.
2. **It cannot predict future performance.** If the deployment distribution differs from the evaluation distribution, test-set metrics are not predictive of production metrics.
3. **It gives false confidence.** High test-set accuracy creates an illusion of reliability that collapses upon distribution shift.
4. **It ignores worst-case behavior.** Many safety-critical applications (medical diagnosis, autonomous driving, criminal justice) require guarantees about worst-case, not average-case, performance.

### The Core Questions

Robust evaluation under shift asks:

- **What kinds of distribution shift can occur?** (Taxonomy)
- **How do we detect that shift has occurred?** (Detection)
- **How do we evaluate a model's resilience to shift before deployment?** (Proactive evaluation)
- **How do we measure worst-case and subgroup performance?** (Robustness metrics)

---

## 2. Conceptual Foundations

### 2.1 Distribution Shift: A Taxonomy

Let $P_{\text{train}}(X, Y)$ denote the joint distribution of features $X$ and labels $Y$ during training, and $P_{\text{test}}(X, Y)$ denote the joint distribution at test/deployment time. Distribution shift means $P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)$.

Using the factorization $P(X, Y) = P(Y \mid X) \, P(X) = P(X \mid Y) \, P(Y)$, we can decompose shift into distinct types based on which factor changes.

---

#### 2.1.1 Covariate Shift

**Definition:** The marginal distribution of inputs changes, but the labeling function remains the same:

$$
P_{\text{train}}(X) \neq P_{\text{test}}(X), \quad P_{\text{train}}(Y \mid X) = P_{\text{test}}(Y \mid X)
$$

**Intuition:** The *questions* the model is asked change, but the *correct answer* for any given question stays the same.

**Concrete example:** A medical imaging model is trained on chest X-rays from Hospital A (modern digital equipment, mostly adult patients). It is deployed at Hospital B (older analog equipment, pediatric population). The images look different (different pixel intensity distributions, different body sizes), but the ground-truth relationship between pathology and image features is unchanged—a pneumonia looks like a pneumonia. The model fails not because the concept changed, but because it encounters inputs in regions of feature space it never trained on.

**Why it matters:** Covariate shift is the most commonly studied form of shift. It causes performance degradation because the model was optimized for one input distribution and may have poor predictions in low-density regions of the training distribution that become high-density regions at test time.

**What breaks:** Models that rely heavily on features correlated with the input distribution (rather than causally related to the label) degrade rapidly. For example, if the model learned that "high image brightness → healthy" because Hospital A's healthy patients happened to have brighter images, that spurious correlation fails at Hospital B.

**Key assumption for correction:** If you know the density ratio $w(x) = P_{\text{test}}(x) / P_{\text{train}}(x)$, you can correct for covariate shift by importance-weighting the training loss. This assumes the support of $P_{\text{test}}$ is contained within the support of $P_{\text{train}}$ (i.e., no truly novel inputs).

---

#### 2.1.2 Label Shift (Prior Probability Shift)

**Definition:** The marginal distribution of labels changes, but the class-conditional feature distribution remains the same:

$$
P_{\text{train}}(Y) \neq P_{\text{test}}(Y), \quad P_{\text{train}}(X \mid Y) = P_{\text{test}}(X \mid Y)
$$

**Intuition:** The *prevalence* of each class changes, but what each class "looks like" does not.

**Concrete example:** A disease classifier is trained on a dataset where 10% of samples are positive (reflecting a balanced clinical study design). At deployment in a general population, the true disease prevalence is 0.1%. The features of sick patients look the same; there are just far fewer of them. A model calibrated to the 10% prior will dramatically overestimate the probability of disease.

**Why it matters:** Label shift directly impacts calibration, precision, recall, and any metric that depends on the class distribution. A model trained with balanced classes will produce miscalibrated probabilities when deployed in a heavily imbalanced setting.

**What breaks:** Predicted class probabilities become unreliable. Decision thresholds optimized on the training class balance become suboptimal. Precision drops because the model's false positive rate, even if low, overwhelms the true positives when the positive class is rare.

**Relationship to covariate shift:** Label shift and covariate shift are formally dual. Under label shift, $P(X \mid Y)$ is fixed and $P(Y)$ changes; under covariate shift, $P(Y \mid X)$ is fixed and $P(X)$ changes. In general, both can occur simultaneously.

---

#### 2.1.3 Concept Drift

**Definition:** The conditional distribution $P(Y \mid X)$ changes:

$$
P_{\text{train}}(Y \mid X) \neq P_{\text{test}}(Y \mid X)
$$

**Intuition:** The *meaning* of the features changes. The same input should now receive a different label.

**Concrete example:** A spam classifier learns that emails containing "bitcoin" are spam (because in 2015, most such emails were scams). By 2021, many legitimate financial newsletters discuss bitcoin. The relationship between the feature "contains bitcoin" and the label "spam" has changed. The model is not seeing different inputs—it is seeing the *same* inputs with *different* correct labels.

**Why it matters:** Concept drift is the most dangerous form of shift because it invalidates the model's learned decision boundary. No amount of importance weighting or threshold adjustment can fix a model whose fundamental input-output relationship is wrong. The only remedy is retraining on data from the new distribution.

**Subtypes of concept drift:**

| Subtype | Description | Example |
|---|---|---|
| **Sudden drift** | Abrupt change at a single point in time | Regulatory change redefining fraud |
| **Gradual drift** | Slow transition from old concept to new | Evolving user preferences |
| **Incremental drift** | Continuous, steady change | Inflation adjusting price thresholds |
| **Recurring drift** | Periodic oscillation between concepts | Seasonal purchasing patterns |

**What breaks:** Everything. The model's learned function $f(x) \approx \mathbb{E}[Y \mid X = x]$ under $P_{\text{train}}$ is no longer an approximation to $\mathbb{E}[Y \mid X = x]$ under $P_{\text{test}}$. Calibration, discrimination, and ranking all degrade.

---

#### 2.1.4 Summary Comparison

| Property | Covariate Shift | Label Shift | Concept Drift |
|---|---|---|---|
| What changes | $P(X)$ | $P(Y)$ | $P(Y \mid X)$ |
| What stays fixed | $P(Y \mid X)$ | $P(X \mid Y)$ | Neither (typically) |
| Model's learned function still valid? | Yes, but evaluated on new inputs | Yes, but calibration breaks | No |
| Correctable without retraining? | Partially (importance weighting) | Partially (threshold/calibration adjustment) | Generally no |
| Detection difficulty | Moderate | Moderate | Hard |

---

### 2.2 Detection Methods

Detecting distribution shift is a prerequisite for responding to it. There are three families of detection methods: statistical tests on raw data, model-based methods, and performance-monitoring approaches.

---

#### 2.2.1 Statistical Tests

##### Kolmogorov-Smirnov (KS) Test

**What it does:** Tests whether two one-dimensional samples come from the same continuous distribution.

**How it works:** Given two samples $\{x_1, \ldots, x_m\}$ from distribution $P$ and $\{y_1, \ldots, y_n\}$ from distribution $Q$, compute the empirical CDFs $\hat{F}_P$ and $\hat{F}_Q$. The KS statistic is:

$$
D_{KS} = \sup_x |\hat{F}_P(x) - \hat{F}_Q(x)|
$$

This measures the maximum vertical distance between the two empirical CDFs. Under the null hypothesis that $P = Q$, the distribution of $D_{KS}$ is known (Kolmogorov distribution, scaled by sample size), yielding a $p$-value.

**Strengths:** Non-parametric (no distributional assumptions), well-understood asymptotics, easy to implement.

**Limitations:**
- **Univariate only.** It tests one feature at a time. For high-dimensional data, you must run it per-feature, which introduces multiple-testing problems and misses multivariate shifts (e.g., a change in the correlation between two features that individually look unchanged).
- **Sensitivity to sample size.** With large samples, the KS test detects tiny, practically meaningless shifts. With small samples, it misses large shifts.
- **Insensitive to tail changes.** The supremum focuses on the region of maximum difference, which may not capture important distributional changes in tails.

##### Maximum Mean Discrepancy (MMD)

**What it does:** Tests whether two samples come from the same distribution using kernel methods. Unlike KS, MMD naturally handles multivariate data.

**How it works:** MMD measures the distance between the mean embeddings of two distributions in a reproducing kernel Hilbert space (RKHS). Given a kernel $k(\cdot, \cdot)$ (e.g., Gaussian RBF), the squared MMD between distributions $P$ and $Q$ is:

$$
\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x, x')] - 2\,\mathbb{E}_{x \sim P, y \sim Q}[k(x, y)] + \mathbb{E}_{y,y' \sim Q}[k(y, y')]
$$

Intuitively, MMD compares the average pairwise similarity *within* each distribution to the average pairwise similarity *between* distributions. If $P = Q$, these are equal and $\text{MMD}^2 = 0$. If $P \neq Q$, the within-distribution similarities exceed the between-distribution similarities.

The empirical estimate from samples $\{x_i\}_{i=1}^m$ and $\{y_j\}_{j=1}^n$:

$$
\widehat{\text{MMD}}^2 = \frac{1}{m(m-1)}\sum_{i \neq i'} k(x_i, x_{i'}) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j) + \frac{1}{n(n-1)}\sum_{j \neq j'} k(y_j, y_{j'})
$$

A permutation test is typically used to obtain a $p$-value: pool the samples, randomly re-split many times, compute MMD for each split, and check where the observed MMD falls in this null distribution.

**Strengths:** Handles multivariate data natively. With a characteristic kernel (like Gaussian RBF), MMD = 0 if and only if $P = Q$, so it can detect any distributional difference.

**Limitations:**
- **Kernel choice sensitivity.** The bandwidth of the RBF kernel critically affects what scale of distributional difference is detected. Too small a bandwidth detects only local differences; too large a bandwidth smooths everything out. Common practice: use the median heuristic (set bandwidth to the median pairwise distance).
- **Computational cost.** Naïve computation is $O((m+n)^2)$ due to all pairwise kernel evaluations. For large datasets, linear-time approximations (random Fourier features) are used.
- **Interpretability.** MMD tells you *that* distributions differ, not *how* or *where*.

---

#### 2.2.2 Model-Based Methods

##### Classifier Two-Sample Test (C2ST)

**What it does:** Trains a binary classifier to distinguish between samples from $P$ and samples from $Q$. If the classifier achieves accuracy significantly above 50%, the distributions are different.

**How it works:**
1. Label all samples from $P$ as class 0 and all samples from $Q$ as class 1.
2. Train a binary classifier (e.g., logistic regression, random forest, neural network) on this combined dataset.
3. Evaluate the classifier's accuracy via cross-validation.
4. If accuracy $\gg 50\%$, reject the null hypothesis $P = Q$.

**Strengths:**
- **Flexible and powerful.** The test is as powerful as the classifier. A neural network C2ST can detect complex, nonlinear distributional differences that kernel methods with fixed kernels might miss.
- **Interpretable.** Feature importances of the trained classifier tell you *which features* are shifting. Predicted probabilities tell you *which samples* are most shifted.
- **Scales to high dimensions.** Unlike per-feature KS tests, C2ST considers the joint distribution.

**Limitations:**
- **Requires careful regularization.** An overfitting classifier will always achieve high accuracy, producing false alarms. Cross-validation is essential.
- **Training cost.** You must train a model, which is more expensive than computing a test statistic.
- **Sensitivity depends on model choice.** A linear classifier will not detect nonlinear shifts; an overly complex model may overfit.

##### Domain Classifier

**What it does:** A specific instantiation of C2ST used in domain adaptation literature. A neural network is trained to classify whether an input came from the source (training) domain or the target (deployment) domain.

In domain-adversarial training (e.g., DANN—Domain-Adversarial Neural Network), the domain classifier is integrated into the training pipeline: the feature extractor is trained to *fool* the domain classifier (via a gradient reversal layer), encouraging domain-invariant representations. For evaluation purposes, the domain classifier's accuracy serves as a proxy for the magnitude of covariate shift.

---

#### 2.2.3 Performance Monitoring

When ground-truth labels are available (possibly with delay), you can directly monitor model performance over time. When labels are not available, proxy signals are used.

##### Accuracy Drift

Track the model's accuracy (or AUC, F1, etc.) over time on labeled data. A statistically significant decline indicates that either the input distribution or the concept has shifted.

**Implementation:** Compute the metric in rolling windows (e.g., weekly) and apply a change-point detection algorithm (CUSUM, ADWIN, Page-Hinkley) or simply monitor for sustained deviation from a baseline.

**Key challenge:** In many production systems, ground-truth labels arrive with delay (e.g., loan defaults are observed 6–12 months after prediction). This makes direct accuracy monitoring sluggish.

##### Confidence Distribution Changes

Even without labels, you can monitor the model's *output distribution*:

- **Prediction confidence histogram:** Track the distribution of $\max_y P(Y=y \mid X)$ over time. A well-calibrated model on in-distribution data will show a characteristic confidence profile. Shift in this distribution suggests the model is encountering unfamiliar inputs.
- **Prediction entropy:** Track $H(Y \mid X) = -\sum_y P(y \mid x) \log P(y \mid x)$. Rising average entropy suggests the model is less certain, which may indicate shift.
- **Class distribution of predictions:** If the predicted class proportions change, this may indicate label shift or covariate shift that pushes inputs across decision boundaries.

**Strengths:** No labels required, can be computed in real time.

**Limitations:** Confidence-based monitoring detects *any* change in model behavior but cannot distinguish between benign shift (new inputs the model handles correctly) and harmful shift (degraded performance). An uncalibrated model's confidence distribution may be misleading.

---

### 2.3 Evaluation Strategies

Detection tells you *that* shift is occurring. Evaluation strategies proactively test a model's resilience to shift *before* deployment.

---

#### 2.3.1 Out-of-Distribution (OOD) Datasets / Stress Testing

**Concept:** Evaluate the model on datasets that are intentionally different from the training distribution to assess how gracefully performance degrades.

**Approaches:**

- **Natural OOD benchmarks.** Use datasets collected from different sources, time periods, geographies, or demographics. Example: train an image classifier on ImageNet and evaluate on ImageNet-V2 (a later re-collection with the same classes but different images), ImageNet-Sketch, ImageNet-R (renditions), or ObjectNet (objects in unusual poses/contexts).
- **Synthetic perturbations.** Apply controlled corruptions (Gaussian noise, blur, JPEG compression, contrast changes) at varying severity levels. ImageNet-C is a benchmark of 15 corruption types at 5 severity levels.
- **Domain-specific stress tests.** For NLP: paraphrase inputs, introduce typos, change dialects. For tabular data: simulate missing features, introduce outliers. For medical imaging: simulate scanner artifacts.

**Design principles:**
- **Cover a range of shift magnitudes.** Test at mild, moderate, and severe shift to understand the degradation curve, not just a single point.
- **Match plausible deployment shifts.** The perturbations should reflect realistic deployment conditions, not arbitrary corruptions.

---

#### 2.3.2 Temporal Validation

**Concept:** Respect the temporal structure of data. Instead of random train/test splits, split data chronologically: train on the past, evaluate on the future.

**Implementation:**

- **Fixed temporal split.** Train on data before time $t$, evaluate on data after $t$. This directly measures how well the model generalizes forward in time.
- **Expanding window.** Sequentially retrain on all data up to time $t$ and evaluate on $t+1, t+2, \ldots$. Measures the effect of increasing training data and of temporal distance.
- **Sliding window.** Train on the most recent $w$ time units of data and evaluate on the next period. Captures the trade-off between recency and data volume.
- **Walk-forward validation.** The time-series equivalent of cross-validation. For each fold, the training set is all data before a cutoff, and the test set is the period immediately following. Folds advance through time.

**Why this matters:** Random splits in temporal data are a cardinal sin because they allow information leakage from the future into the past. A model evaluated via random split on time-series data will appear far better than it actually is. Temporal validation directly measures the model's ability to generalize to future data—which is what deployment actually requires.

---

#### 2.3.3 Subpopulation Analysis

**Concept:** Disaggregate overall performance metrics into performance on meaningful subgroups defined by protected attributes (race, gender, age), data characteristics (image quality, text length), or domain-specific slices (disease type, customer segment).

**Why overall metrics hide problems:** A model with 90% accuracy overall might have:
- 95% accuracy on the majority subgroup (80% of data)
- 70% accuracy on a minority subgroup (20% of data)

The overall metric is dominated by the majority group. Subpopulation analysis reveals the 70% failure that would be invisible in the aggregate.

**Methodology:**

1. **Define slices.** Use domain knowledge, protected attributes, or data-driven methods (e.g., clustering in embedding space, Domino/SliceFinder to automatically discover underperforming subgroups).
2. **Compute per-slice metrics.** Report accuracy, calibration, precision, recall for each slice.
3. **Measure performance gaps.** The difference in metric values between the best- and worst-performing slices is a key robustness indicator.
4. **Track slice stability.** A slice that performs well on i.i.d. test data but poorly on shifted data is a robustness vulnerability.

---

#### 2.3.4 Adversarial Evaluation

**Concept:** Actively search for inputs or subgroups on which the model fails.

**Types:**

- **Input-level adversarial perturbation.** Apply small, often imperceptible perturbations to inputs that cause misclassification. For images, methods include FGSM (Fast Gradient Sign Method), PGD (Projected Gradient Descent), and AutoAttack. For text, methods include character swaps, synonym substitution, paraphrase attacks.
- **Worst-case subgroup analysis.** Instead of looking at pre-defined subgroups, algorithmically search for the subgroup (defined by feature predicates) on which the model performs worst. This is the idea behind George (Group DRO) and tools like the What-If Tool or FairLearn.
- **Stress testing with adversarial data augmentation.** Generate training-time augmentations that simulate adversarial conditions and evaluate model behavior.

**Key distinction:** Adversarial evaluation differs from OOD evaluation in that adversarial evaluation specifically targets the model's weaknesses, while OOD evaluation assesses general robustness to distribution change. A model can be robust to OOD data but brittle to adversarial perturbations (and vice versa).

---

### 2.4 Robustness Metrics

Standard metrics (accuracy, F1, AUC) measure average performance. Robustness metrics specifically capture worst-case behavior and fairness.

---

#### 2.4.1 Group Fairness Metrics

These metrics compare model behavior across predefined groups (e.g., demographic groups defined by a sensitive attribute $A$).

| Metric | Definition | Intuition |
|---|---|---|
| **Demographic Parity** | $P(\hat{Y}=1 \mid A=a) = P(\hat{Y}=1 \mid A=b)$ | Positive prediction rate is equal across groups |
| **Equalized Odds** | $P(\hat{Y}=1 \mid Y=y, A=a) = P(\hat{Y}=1 \mid Y=y, A=b)$ for $y \in \{0,1\}$ | TPR and FPR are equal across groups |
| **Equal Opportunity** | $P(\hat{Y}=1 \mid Y=1, A=a) = P(\hat{Y}=1 \mid Y=1, A=b)$ | TPR is equal across groups (relaxation of equalized odds) |
| **Predictive Parity** | $P(Y=1 \mid \hat{Y}=1, A=a) = P(Y=1 \mid \hat{Y}=1, A=b)$ | Precision is equal across groups |
| **Calibration** | $P(Y=1 \mid \hat{P}=p, A=a) = p$ for all $p$, all $a$ | Predicted probabilities are accurate within each group |

**Impossibility result (Chouldechova, 2017; Kleinberg et al., 2016):** Except in trivial cases (equal base rates across groups or perfect prediction), it is mathematically impossible to simultaneously satisfy calibration, equal FPR, and equal FNR across groups. This means fairness metric choice involves value judgments, not just technical decisions.

---

#### 2.4.2 Worst-Case Performance Metrics

- **Minimax performance:** $\min_{g \in \mathcal{G}} \text{metric}(g)$, where $\mathcal{G}$ is the set of subgroups. This is the performance on the worst-performing subgroup.
- **Performance gap:** $\max_{g \in \mathcal{G}} \text{metric}(g) - \min_{g \in \mathcal{G}} \text{metric}(g)$. Measures disparity.
- **Conditional Value at Risk (CVaR):** Instead of looking at the single worst subgroup, CVaR considers the average performance over the worst $\alpha$ fraction of samples: $\text{CVaR}_\alpha = \mathbb{E}[\ell \mid \ell \geq F_\ell^{-1}(1-\alpha)]$, where $\ell$ is the per-sample loss and $F_\ell^{-1}$ is the quantile function of the loss distribution. This is smoother than minimax and less sensitive to outliers.
- **Effective Robustness:** Measures performance on OOD data relative to a baseline model, controlling for in-distribution accuracy. A model is "effectively robust" if its OOD performance exceeds what would be predicted from its in-distribution performance alone.

---

## 3. Mathematical Formulation

### 3.1 Covariate Shift and Importance Weighting

**Setup:** We want to estimate the expected risk under the test distribution:

$$
R_{\text{test}}(f) = \mathbb{E}_{(x,y) \sim P_{\text{test}}}[\ell(f(x), y)]
$$

We only have labeled samples from $P_{\text{train}}$. Under the covariate shift assumption ($P_{\text{test}}(Y \mid X) = P_{\text{train}}(Y \mid X)$):

$$
R_{\text{test}}(f) = \mathbb{E}_{(x,y) \sim P_{\text{test}}}[\ell(f(x), y)] = \int \ell(f(x), y) \, P_{\text{test}}(y \mid x) \, P_{\text{test}}(x) \, dx \, dy
$$

Since $P_{\text{test}}(y \mid x) = P_{\text{train}}(y \mid x)$:

$$
= \int \ell(f(x), y) \, P_{\text{train}}(y \mid x) \, \frac{P_{\text{test}}(x)}{P_{\text{train}}(x)} \, P_{\text{train}}(x) \, dx \, dy
$$

$$
= \mathbb{E}_{(x,y) \sim P_{\text{train}}} \left[ \frac{P_{\text{test}}(x)}{P_{\text{train}}(x)} \, \ell(f(x), y) \right]
$$

Define the importance weight $w(x) = \frac{P_{\text{test}}(x)}{P_{\text{train}}(x)}$. Then:

$$
R_{\text{test}}(f) = \mathbb{E}_{(x,y) \sim P_{\text{train}}} \left[ w(x) \, \ell(f(x), y) \right]
$$

**Empirical estimator:**

$$
\hat{R}_{\text{test}}(f) = \frac{1}{n} \sum_{i=1}^n w(x_i) \, \ell(f(x_i), y_i)
$$

**Estimating $w(x)$:** Since we typically do not know $P_{\text{train}}(x)$ or $P_{\text{test}}(x)$ in closed form, $w(x)$ is estimated. Common approaches:

1. **Density ratio estimation (e.g., KLIEP, KMM, uLSIF):** Directly estimate the ratio $P_{\text{test}} / P_{\text{train}}$ without estimating either density separately.
2. **Classifier-based:** Train a classifier to distinguish training from test data. If $D(x)$ is the classifier's probability that $x$ is from the test set, then $w(x) \approx D(x) / (1 - D(x))$ (by Bayes' rule, assuming balanced classes).

**Variance problem:** Importance weighting can produce estimates with very high variance when $w(x)$ takes extreme values (i.e., when there are test inputs in regions with very low training density). This is mitigated by clipping weights: $\tilde{w}(x) = \min(w(x), C)$ for some cap $C$, at the cost of introducing bias.

---

### 3.2 Label Shift Correction

**Setup:** Under label shift, $P_{\text{train}}(X \mid Y) = P_{\text{test}}(X \mid Y)$ but $P_{\text{train}}(Y) \neq P_{\text{test}}(Y)$.

The test-time posterior is:

$$
P_{\text{test}}(Y \mid X) = \frac{P_{\text{test}}(X \mid Y) \, P_{\text{test}}(Y)}{P_{\text{test}}(X)}
$$

$$
= \frac{P_{\text{train}}(X \mid Y) \, P_{\text{test}}(Y)}{P_{\text{test}}(X)}
$$

$$
= \frac{P_{\text{train}}(Y \mid X) \, P_{\text{train}}(X) \, P_{\text{test}}(Y)}{P_{\text{train}}(Y) \, P_{\text{test}}(X)}
$$

The ratio $P_{\text{test}}(Y) / P_{\text{train}}(Y)$ is a per-class importance weight. In practice:

1. Estimate $P_{\text{test}}(Y)$ from unlabeled test data using the model's predictions and the confusion matrix (Black Box Shift Estimation—BBSE; Lipton et al., 2018).
2. Adjust model predictions by the ratio $P_{\text{test}}(Y=c) / P_{\text{train}}(Y=c)$ for each class $c$.

**BBSE procedure:**

Let $C$ be the $K \times K$ confusion matrix of the classifier on training data: $C_{yc} = P_{\text{train}}(\hat{Y} = c \mid Y = y)$.

On unlabeled test data, observe the predicted label frequencies: $\hat{\mu}_c = P_{\text{test}}(\hat{Y} = c)$.

Under label shift:

$$
\hat{\mu}_c = \sum_y C_{yc} \, P_{\text{test}}(Y = y)
$$

In matrix form: $\hat{\boldsymbol{\mu}} = C^\top \mathbf{p}_{\text{test}}$, so:

$$
\mathbf{p}_{\text{test}} = (C^\top)^{-1} \hat{\boldsymbol{\mu}}
$$

This gives an estimate of the test-time class prior, which can be used to re-weight or re-calibrate predictions.

---

### 3.3 Maximum Mean Discrepancy — Detailed Derivation

Let $\mathcal{F}$ be a function class. The MMD between $P$ and $Q$ is:

$$
\text{MMD}(\mathcal{F}, P, Q) = \sup_{f \in \mathcal{F}} \left( \mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{y \sim Q}[f(y)] \right)
$$

When $\mathcal{F}$ is the unit ball of an RKHS $\mathcal{H}$ with kernel $k$, this becomes:

$$
\text{MMD}^2(\mathcal{H}, P, Q) = \|\mu_P - \mu_Q\|_{\mathcal{H}}^2
$$

where $\mu_P = \mathbb{E}_{x \sim P}[\phi(x)]$ is the kernel mean embedding and $\phi$ is the feature map such that $k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$.

Expanding:

$$
\|\mu_P - \mu_Q\|^2 = \langle \mu_P, \mu_P \rangle - 2\langle \mu_P, \mu_Q \rangle + \langle \mu_Q, \mu_Q \rangle
$$

$$
= \mathbb{E}_{x,x' \sim P}[k(x,x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)] + \mathbb{E}_{y,y' \sim Q}[k(y,y')]
$$

**Interpretation of each term:**

- $\mathbb{E}[k(x,x')]$ for $x, x' \sim P$: Average similarity within $P$.
- $\mathbb{E}[k(y,y')]$ for $y, y' \sim Q$: Average similarity within $Q$.
- $\mathbb{E}[k(x,y)]$ for $x \sim P, y \sim Q$: Average similarity between $P$ and $Q$.

If $P = Q$, all three are equal, so $\text{MMD}^2 = 0$. If $P \neq Q$, within-distribution similarities exceed between-distribution similarity, making $\text{MMD}^2 > 0$.

---

### 3.4 Group Distributionally Robust Optimization (Group DRO)

Standard ERM minimizes average risk:

$$
\min_\theta \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i)
$$

Group DRO instead minimizes worst-case group risk:

$$
\min_\theta \max_{g \in \mathcal{G}} \mathbb{E}_{(x,y) \sim P_g}[\ell(f_\theta(x), y)]
$$

where $\mathcal{G} = \{P_1, \ldots, P_G\}$ is a set of subgroup distributions.

**Why this helps:** ERM allows the model to sacrifice performance on small subgroups to improve average performance. Group DRO forces the model to perform well on *every* subgroup, including minorities.

**Practical algorithm (Sagawa et al., 2020):** Maintain per-group weights $\{q_g\}$ that are increased when a group has high loss and decreased otherwise. At each step:

1. Compute per-group average loss $\hat{R}_g(\theta)$.
2. Update group weights: $q_g \propto q_g \cdot \exp(\eta \, \hat{R}_g(\theta))$ (exponentiated gradient ascent).
3. Update model parameters to minimize the weighted sum $\sum_g q_g \, \hat{R}_g(\theta)$ (gradient descent).

This is a saddle-point optimization: minimize over model parameters, maximize over group weights.

---

### 3.5 Conditional Value at Risk (CVaR)

For a random loss $L$, the CVaR at level $\alpha$ is:

$$
\text{CVaR}_\alpha(L) = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha(L)]
$$

where $\text{VaR}_\alpha(L) = \inf\{l : P(L \leq l) \geq 1-\alpha\}$ is the $(1-\alpha)$-quantile of the loss.

**Equivalent optimization form (Rockafellar & Uryasev, 2000):**

$$
\text{CVaR}_\alpha(L) = \min_{\nu \in \mathbb{R}} \left\{ \nu + \frac{1}{\alpha} \mathbb{E}[\max(L - \nu, 0)] \right\}
$$

**Intuition:** $\text{CVaR}_{0.1}$ averages the loss over the worst 10% of samples. It is a smooth relaxation of the minimax objective: instead of caring only about the single worst sample, it considers the worst tail of the loss distribution.

**Connection to DRO:** Minimizing CVaR is equivalent to DRO over a specific uncertainty set. Specifically, CVaR-based DRO minimizes the worst-case expected loss over all distributions within a chi-squared or KL divergence ball around the empirical distribution. This provides a principled way to robustify a model without explicitly defining subgroups.

---

## 4. Worked Examples

### 4.1 Detecting Covariate Shift with a Classifier Two-Sample Test

**Setting:** You trained a churn prediction model on customer data from Q1–Q3 2024. You have new, unlabeled customer data from Q1 2025. Before deploying predictions, you want to check if the input distribution has shifted.

**Step 1: Construct the combined dataset.**

| Feature | Age | Monthly Spend ($) | Tenure (months) | Label |
|---|---|---|---|---|
| Training sample 1 | 34 | 120 | 24 | 0 (train) |
| Training sample 2 | 45 | 85 | 36 | 0 (train) |
| ... | ... | ... | ... | 0 (train) |
| New sample 1 | 28 | 45 | 6 | 1 (new) |
| New sample 2 | 52 | 200 | 3 | 1 (new) |
| ... | ... | ... | ... | 1 (new) |

The binary label here is *not* the churn label—it indicates whether the sample is from the training set (0) or the new data (1).

**Step 2: Train a classifier.**

Train a logistic regression (or gradient boosted tree) on this dataset with 5-fold cross-validation. Record the average cross-validation accuracy.

**Step 3: Interpret the result.**

- **Cross-val accuracy ≈ 50%.** The classifier cannot distinguish training from new data. No detectable covariate shift. Proceed with deployment.
- **Cross-val accuracy = 68%.** The classifier can reliably distinguish the datasets. Significant covariate shift detected. Investigate further.

**Step 4: Diagnose the shift.**

Inspect the classifier's feature importances. Suppose "Tenure" has the highest importance with coefficient strongly negative (new data has shorter tenure). This tells you that the new customer cohort has systematically shorter tenure than the training cohort—perhaps due to a recent marketing campaign that acquired many new customers.

**Step 5: Decide.**

You might: (a) retrain the churn model including recent data, (b) apply importance weighting to account for the tenure distribution shift, or (c) evaluate the churn model specifically on the short-tenure subgroup to assess whether it is reliable for this cohort.

---

### 4.2 Temporal Validation Example

**Setting:** You are building a stock price direction predictor (up/down) using daily features. You have data from Jan 2020 to Dec 2024.

**Bad approach (random split):**

Randomly select 80% of days for training, 20% for testing. This allows the model to use Feb 2023 data to predict Jan 2023—temporal leakage.

**Correct approach (walk-forward):**

| Fold | Training period | Test period |
|---|---|---|
| 1 | Jan 2020 – Dec 2021 | Jan – Jun 2022 |
| 2 | Jan 2020 – Jun 2022 | Jul – Dec 2022 |
| 3 | Jan 2020 – Dec 2022 | Jan – Jun 2023 |
| 4 | Jan 2020 – Jun 2023 | Jul – Dec 2023 |
| 5 | Jan 2020 – Dec 2023 | Jan – Jun 2024 |
| 6 | Jan 2020 – Jun 2024 | Jul – Dec 2024 |

**Analysis:** Suppose the model achieves:

| Fold | Accuracy |
|---|---|
| 1 | 58% |
| 2 | 55% |
| 3 | 53% |
| 4 | 51% |
| 5 | 49% |
| 6 | 47% |

A monotonic decline in accuracy with temporal distance from training data strongly suggests concept drift: the patterns the model learned are becoming less predictive over time. This would be invisible with a random split (which might show a stable 54%).

---

### 4.3 Subpopulation Analysis with Fairness Metrics

**Setting:** A hiring classifier predicts whether a candidate will succeed in a job. The model is evaluated on a test set of 1,000 candidates.

| Group | $n$ | Accuracy | TPR (Recall) | FPR | Precision |
|---|---|---|---|---|---|
| Overall | 1000 | 85% | 80% | 10% | 89% |
| Male | 700 | 87% | 84% | 8% | 91% |
| Female | 300 | 80% | 68% | 15% | 82% |

**Analysis:**

- **Overall accuracy is 85%.** This looks acceptable.
- **Accuracy gap:** $87\% - 80\% = 7\%$ between groups.
- **Equal opportunity violation:** TPR gap is $84\% - 68\% = 16\%$. The model is substantially less likely to correctly identify qualified female candidates.
- **Equalized odds violation:** Both TPR and FPR differ. The model not only misses more qualified females but also incorrectly rejects fewer unqualified males.
- **Worst-case subgroup performance:** 68% TPR for females is the minimax metric. Depending on the application's requirements, this may be unacceptably low.

**Actions:**
- Investigate *why* the model underperforms on females (less training data? different feature distributions? proxying on correlated features?).
- Consider Group DRO training to improve worst-group performance.
- Report minimax TPR alongside overall TPR to stakeholders.

---

### 4.4 Importance Weighting Numeric Example

**Setting:** A binary classifier is trained on data where $P_{\text{train}}(X = \text{young}) = 0.8$ and $P_{\text{train}}(X = \text{old}) = 0.2$. At deployment, $P_{\text{test}}(X = \text{young}) = 0.5$ and $P_{\text{test}}(X = \text{old}) = 0.5$.

**Step 1: Compute importance weights.**

$$
w(\text{young}) = \frac{P_{\text{test}}(\text{young})}{P_{\text{train}}(\text{young})} = \frac{0.5}{0.8} = 0.625
$$

$$
w(\text{old}) = \frac{P_{\text{test}}(\text{old})}{P_{\text{train}}(\text{old})} = \frac{0.5}{0.2} = 2.5
$$

**Step 2: Compute unweighted vs. weighted risk.**

Suppose the model has loss 0.1 on young samples and loss 0.4 on old samples.

Unweighted training risk: $0.8 \times 0.1 + 0.2 \times 0.4 = 0.08 + 0.08 = 0.16$

Importance-weighted (estimating test risk): $0.8 \times 0.625 \times 0.1 + 0.2 \times 2.5 \times 0.4 = 0.05 + 0.2 = 0.25$

The importance-weighted estimate reveals that the model's risk on the deployment distribution is substantially higher (0.25 vs. 0.16) because the deployment distribution upweights the subgroup (old) on which the model performs poorly.

---

## 5. Relevance to Machine Learning Practice

### 5.1 Where Robust Evaluation Appears in ML Systems

**Training:**
- Group DRO and CVaR-based training modify the training objective to improve worst-case performance. This is relevant for fairness-constrained models and safety-critical applications.
- Data augmentation as a form of robustness training: augmenting with shifted/corrupted inputs during training improves robustness at evaluation time.

**Evaluation (pre-deployment):**
- Temporal validation is standard for any time-dependent application (finance, recommendation systems, demand forecasting).
- Subpopulation analysis is required by regulation in some domains (e.g., healthcare AI must report performance across demographic subgroups per FDA guidance).
- OOD benchmarking is increasingly standard in vision and NLP (WILDS benchmark suite).

**Monitoring (post-deployment):**
- Production ML systems at major tech companies run continuous distribution shift detection (feature drift monitoring) as part of MLOps pipelines. Tools: Evidently AI, WhyLabs, Arize, NannyML, AWS SageMaker Model Monitor.
- Confidence monitoring and prediction drift are typical first-line defenses when labels are delayed.
- Retraining triggers are often based on detected shift or performance degradation.

**Inference:**
- Importance weighting at inference time can correct for known covariate shift.
- Selective prediction (abstaining on low-confidence inputs) is a robustness strategy at inference time.

### 5.2 When to Use Each Approach

| Situation | Recommended approach |
|---|---|
| Deploying to a known-different population | OOD evaluation + importance weighting |
| Time-series or temporal data | Temporal validation (mandatory) |
| Fairness-sensitive application | Subpopulation analysis + fairness metrics |
| Safety-critical (medical, autonomous) | Adversarial evaluation + worst-case metrics |
| Long-running production model | Continuous drift monitoring |
| Labels available with delay | Confidence/prediction drift monitoring |
| Class imbalance likely to change | Label shift correction |
| Unknown shift, want general robustness | Group DRO training + CVaR metric |

### 5.3 Trade-Offs

**Robustness vs. average-case performance:** Optimizing for worst-case subgroup performance (Group DRO, minimax) typically reduces average-case performance. This is because the model allocates capacity to hard subgroups at the expense of easy ones. The trade-off is real and must be decided by stakeholders.

**Detection sensitivity vs. false alarm rate:** Aggressive drift detection (low thresholds, many statistical tests) catches real shift early but produces false alarms that trigger unnecessary retraining. Conservative detection misses shift until performance degrades visibly.

**Evaluation cost vs. coverage:** Comprehensive robustness evaluation (many OOD datasets, many subgroups, adversarial attacks, temporal folds) is expensive in compute and human effort. Prioritization based on deployment risk is essential.

**Importance weighting bias vs. variance:** Clipping importance weights reduces variance but introduces bias. Un-clipped weights are unbiased but can have variance so high that the estimate is practically useless. Self-normalized importance weights (dividing by the sum of weights) are a standard compromise.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "High test accuracy means the model is production-ready."

**Why it is wrong:** Test accuracy on an i.i.d. split tells you about average-case performance on a distribution that matches training. It says nothing about performance under shift, on subpopulations, or on adversarial inputs.

**How to avoid:** Always complement i.i.d. evaluation with at least one form of shift-aware evaluation (temporal validation, OOD testing, or subpopulation analysis).

### Pitfall 2: "We can correct for any distribution shift with importance weighting."

**Why it is wrong:** Importance weighting only works under covariate shift (or label shift), where $P(Y \mid X)$ (or $P(X \mid Y)$) is unchanged. It cannot correct for concept drift. Even under covariate shift, importance weighting fails when the test distribution has support outside the training distribution (division by zero in the density ratio) or when the density ratio is extreme (high-variance estimates).

### Pitfall 3: "Random train/test splits are always fine."

**Why it is wrong:** For temporal data, random splits cause information leakage from the future. For grouped data (e.g., multiple measurements per patient), random splits cause information leakage between groups. Both lead to overly optimistic evaluation.

**How to avoid:** Use temporal splits for time data. Use group-aware splits (group-k-fold) for grouped data.

### Pitfall 4: "The KS test on each feature is sufficient to detect multivariate shift."

**Why it is wrong:** Two distributions can have identical marginals but different joint distributions (e.g., identical feature means and variances but different correlations). Per-feature KS tests will detect no shift.

**How to avoid:** Use multivariate methods (MMD, C2ST) or at minimum test on derived features (principal components, model embeddings).

### Pitfall 5: "A model that is fair on i.i.d. test data will remain fair under shift."

**Why it is wrong:** Fairness metrics are properties of the joint distribution of predictions, labels, and group membership. Under shift, this joint distribution changes, and fairness guarantees can break. A model that satisfies equalized odds on the training distribution may violate it severely on a shifted distribution.

### Pitfall 6: "Concept drift requires explicit detection before action."

**Why it is wrong in practice:** By the time concept drift is statistically detected, the model may have been making poor predictions for a long time (especially if labels are delayed). Proactive strategies—periodic retraining, online learning, sliding-window models—can mitigate drift without waiting for detection.

### Pitfall 7: "Confidence calibration implies robustness."

**Why it is wrong:** A well-calibrated model on in-distribution data can become severely miscalibrated under shift. Calibration is a property of the data distribution, not an intrinsic property of the model. Temperature scaling calibrated on the training distribution does not transfer to shifted distributions.

### Pitfall 8: "Adversarial robustness and distributional robustness are the same."

**Why it is wrong:** Adversarial robustness concerns worst-case performance over small perturbations of individual inputs (an $\ell_p$-ball around each input). Distributional robustness concerns performance under shifts of the entire data distribution. A model can be adversarially robust but fail under covariate shift (and vice versa). Methods that improve one do not necessarily improve the other.

### Pitfall 9: "More data always fixes distribution shift."

**Why it is wrong:** If the additional data comes from the same (outdated) distribution, it does not help with shift to a new distribution. In fact, more data from the old distribution can make the model *more* confident in its (now-incorrect) predictions. What matters is data from the *target* distribution.

### Pitfall 10: "Monitoring model confidence is a reliable proxy for performance."

**Why it is wrong:** Confidence monitoring catches shift only when the model "knows it doesn't know." Under concept drift, the model can be *confidently wrong*—the inputs look familiar, but the correct label has changed. Confidence monitoring will show no alarm while performance degrades.

---

## Interview Preparation Section

---

### Foundational Questions

---

**Q1: What is distribution shift? Why does it matter for ML systems in production?**

**A:** Distribution shift is any change between the data distribution seen during training/evaluation and the distribution encountered during deployment. It matters because all standard evaluation metrics assume the test data is drawn from the same distribution as the training data. When this assumption breaks, metrics computed during development are not predictive of real-world performance. In production, shift is the norm, not the exception: user populations change, the world evolves, and data collection pipelines introduce artifacts. A model that is not evaluated for robustness under shift may fail silently—performing well by metrics that are no longer valid while making harmful decisions.

---

**Q2: Explain the three main types of distribution shift. Give a concrete example of each.**

**A:**

*Covariate shift ($P(X)$ changes, $P(Y|X)$ constant):* A sentiment classifier trained on product reviews from Amazon is deployed on Twitter. The writing style, vocabulary, and sentence length distributions are different (shift in $P(X)$), but the underlying relationship between language and sentiment remains the same. The model struggles because it encounters input patterns it rarely saw during training.

*Label shift ($P(Y)$ changes, $P(X|Y)$ constant):* A medical test for a rare disease is validated in a clinical trial where 50% of patients are positive (oversampled by design). At deployment in a general hospital, only 1% of patients are positive. What a positive case "looks like" ($P(X|Y=1)$) is unchanged, but the prevalence is 50× lower. The model's predicted probabilities, calibrated to the trial's 50% rate, are now miscalibrated.

*Concept drift ($P(Y|X)$ changes):* A loan default model learns that "has a mortgage" correlates with lower default risk (homeowners are stable). Then a housing crisis occurs: having a mortgage now correlates with *higher* default risk because home values collapse. The same feature now has the opposite meaning. No correction short of retraining can fix this.

---

**Q3: What is the difference between covariate shift and concept drift? Why does this distinction matter practically?**

**A:** Covariate shift means the inputs change but the labeling function stays the same. The model's learned decision boundary is still correct—it is just being evaluated on inputs it has not seen. Concept drift means the labeling function itself changes—the model's learned boundary is now wrong regardless of the inputs.

This distinction matters practically because the remediation strategies are fundamentally different. For covariate shift, you can apply importance weighting, collect more data from the new input region, or fine-tune on representative data, without changing the model's core learned function. For concept drift, the learned function is invalid and the model must be retrained (or adapted online) on data reflecting the new $P(Y|X)$.

---

**Q4: What does "robust evaluation" mean and how does it differ from standard evaluation?**

**A:** Standard evaluation computes a performance metric on a test set drawn from the same distribution as the training data. It answers: "How well does the model do on average on data similar to what it trained on?"

Robust evaluation extends this in four directions:

1. Evaluating on intentionally shifted distributions (OOD testing).
2. Evaluating performance on subgroups rather than aggregates (subpopulation analysis).
3. Evaluating worst-case performance rather than average-case (minimax metrics, adversarial evaluation).
4. Evaluating temporal generalization (temporal validation).

It answers: "How will this model perform when conditions change?"

---

**Q5: Why can't you use a random train/test split for time-series data?**

**A:** A random split allows future observations into the training set and past observations into the test set. The model then has access to patterns that only become visible after the test point—a form of information leakage. For example, if the model learns from a data point in December 2024 and is tested on a point in October 2024, it is implicitly using future information. This produces inflated evaluation metrics that do not reflect real deployment performance, where the model can only use information available at prediction time. Temporal validation (train on the past, test on the future) eliminates this leakage.

---

### Mathematical Questions

---

**Q6: Derive the importance-weighted risk estimator for covariate shift. What assumptions are required?**

**A:** Starting from the test risk:

$$
R_{\text{test}}(f) = \mathbb{E}_{P_{\text{test}}}[\ell(f(x), y)] = \int \ell(f(x), y) \, p_{\text{test}}(x, y) \, dx \, dy
$$

Factor the joint: $p_{\text{test}}(x, y) = p_{\text{test}}(y|x) \, p_{\text{test}}(x)$. Under the covariate shift assumption, $p_{\text{test}}(y|x) = p_{\text{train}}(y|x)$, so:

$$
= \int \ell(f(x), y) \, p_{\text{train}}(y|x) \, p_{\text{test}}(x) \, dx \, dy
$$

Multiply and divide by $p_{\text{train}}(x)$:

$$
= \int \ell(f(x), y) \, p_{\text{train}}(y|x) \, \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)} \, p_{\text{train}}(x) \, dx \, dy = \mathbb{E}_{P_{\text{train}}} \left[ w(x) \, \ell(f(x), y) \right]
$$

where $w(x) = p_{\text{test}}(x) / p_{\text{train}}(x)$.

**Assumptions:**
1. **Covariate shift:** $P(Y|X)$ is identical across domains. If this fails (concept drift), the derivation is invalid.
2. **Absolute continuity / support overlap:** $p_{\text{train}}(x) > 0$ wherever $p_{\text{test}}(x) > 0$. If there are test inputs in regions with zero training density, $w(x)$ is undefined.
3. **Finite variance of $w(x) \cdot \ell$:** If $w(x)$ takes extreme values, the empirical estimate has high variance and is practically useless.

---

**Q7: Explain the MMD test statistic. What does each term represent? When does MMD = 0?**

**A:**

$$
\text{MMD}^2(P, Q) = \mathbb{E}_{x,x' \sim P}[k(x,x')] - 2\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)] + \mathbb{E}_{y,y' \sim Q}[k(y,y')]
$$

Term 1 ($\mathbb{E}[k(x,x')]$ for $x, x' \sim P$): Average kernel similarity between two independent draws from $P$. Measures internal cohesion of $P$.

Term 2 ($-2\mathbb{E}[k(x,y)]$ for $x \sim P, y \sim Q$): Negative of twice the average cross-distribution similarity. Measures how similar $P$ and $Q$ are to each other.

Term 3 ($\mathbb{E}[k(y,y')]$ for $y, y' \sim Q$): Internal cohesion of $Q$.

$\text{MMD}^2 = 0$ if and only if $P = Q$ (when the kernel is characteristic, e.g., Gaussian RBF). Intuitively, if $P = Q$, the within-distribution and between-distribution similarities are all equal, so the expression cancels to zero.

$\text{MMD}^2 > 0$ when $P \neq Q$ because points within the same distribution are, on average, more similar to each other than to points from the other distribution.

---

**Q8: Explain why Chouldechova's impossibility result matters for robust evaluation under shift.**

**A:** Chouldechova (2017) proved that when base rates differ across groups ($P(Y=1 \mid A=a) \neq P(Y=1 \mid A=b)$) and the classifier is imperfect, you cannot simultaneously achieve: (i) calibration within each group, (ii) equal false positive rates, and (iii) equal false negative rates.

For robust evaluation under shift, this has two consequences:

First, you must *choose* which fairness metric to prioritize, and this is a value judgment that cannot be resolved technically. Different metrics lead to different evaluative conclusions about the same model.

Second, under distribution shift, base rates often change (label shift). Even a model that approximately satisfies multiple fairness constraints on the training distribution will violate some of them on the shifted distribution, simply because the base rates have changed. This means fairness evaluation must be repeated under each anticipated shift scenario, not just on the i.i.d. test set.

---

**Q9: Write out the Group DRO objective. How does it differ from ERM? What is the optimization procedure?**

**A:**

ERM: $\min_\theta \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i)$

Group DRO: $\min_\theta \max_{g \in \{1,\ldots,G\}} \hat{R}_g(\theta)$, where $\hat{R}_g(\theta) = \frac{1}{n_g}\sum_{i \in \text{group } g} \ell(f_\theta(x_i), y_i)$

ERM treats all samples equally and minimizes the population average. Group DRO minimizes the loss of the worst-performing group. This forces the model to allocate capacity to hard/minority groups rather than achieving marginal gains on the majority.

The optimization is a saddle-point problem: minimize over $\theta$, maximize over group weights $q \in \Delta^G$ (the simplex). Practically, it is solved by alternating:

- **Inner maximization (over $q$):** Increase the weight on groups with high current loss. Implemented via exponentiated gradient: $q_g^{(t+1)} \propto q_g^{(t)} \exp(\eta_q \hat{R}_g(\theta^{(t)}))$.
- **Outer minimization (over $\theta$):** Standard gradient descent on the weighted loss $\sum_g q_g \hat{R}_g(\theta)$.

This converges to a solution where the model's worst-group loss is minimized.

---

**Q10: Derive how label shift can be corrected using the confusion matrix approach (BBSE).**

**A:** Let $\hat{Y} = h(X)$ be the classifier's prediction. Define the confusion matrix $C \in \mathbb{R}^{K \times K}$ with $C_{yc} = P_{\text{train}}(\hat{Y} = c \mid Y = y)$.

On unlabeled test data, we observe the marginal predicted label frequencies:

$$
\mu_c = P_{\text{test}}(\hat{Y} = c)
$$

Under label shift, $P(X|Y)$ is constant across domains, so the classifier's confusion behavior for each class is constant: $P_{\text{test}}(\hat{Y} = c | Y = y) = P_{\text{train}}(\hat{Y} = c | Y = y) = C_{yc}$.

By the law of total probability:

$$
\mu_c = \sum_{y=1}^K P_{\text{test}}(\hat{Y}=c \mid Y=y) P_{\text{test}}(Y=y) = \sum_{y=1}^K C_{yc} \, p_y^{\text{test}}
$$

In matrix form: $\boldsymbol{\mu} = C^\top \mathbf{p}^{\text{test}}$

Solving: $\mathbf{p}^{\text{test}} = (C^\top)^{-1} \boldsymbol{\mu}$

Requirements: $C$ must be invertible (the classifier must have non-degenerate confusion patterns), and $\boldsymbol{\mu}$ must be estimated from a sufficiently large unlabeled test sample.

Once $\mathbf{p}^{\text{test}}$ is estimated, predictions are corrected by adjusting the prior: multiply the predicted class probability by $p_y^{\text{test}} / p_y^{\text{train}}$ and renormalize.

---

### Applied Questions

---

**Q11: You deploy a fraud detection model. Over three months, the flagged fraud rate drops from 5% to 1%, but you have no labeled ground truth yet. How do you determine whether this is a real decrease in fraud or a model degradation?**

**A:** Without ground truth, I cannot directly measure accuracy. My approach would be:

1. **Check for covariate shift.** Run a C2ST or MMD test comparing the input features from the training period to the most recent three months. If the test detects shift, the model may be receiving inputs it is not trained for.

2. **Analyze the confidence distribution.** If the model's average confidence on positive predictions has decreased, this suggests the model is less certain about its fraud calls, consistent with encountering unfamiliar patterns (degradation). If confidence remains stable but the number of high-confidence positives has simply decreased, this is more consistent with a genuine decline in fraud.

3. **Examine the prediction distribution over subpopulations.** Has the flag rate dropped uniformly, or only for certain transaction types, geographies, or customer segments? A uniform drop is more consistent with a real prevalence change. A subgroup-specific drop suggests the model may have lost sensitivity for that subgroup.

4. **Apply BBSE.** If I trust the confusion matrix from the training period, I can estimate the true fraud rate from the model's predictions on the unlabeled data. If the estimated true fraud rate is much higher than 1%, the model is likely losing recall.

5. **Prioritize rapid labeling.** Sample recent predictions (stratified by predicted probability) for manual review to get partial ground truth. This is the definitive test.

---

**Q12: Your image classification model achieves 93% accuracy on the standard test set but only 71% on a new dataset from a different hospital. What systematic evaluation would you perform?**

**A:**

1. **Characterize the shift.** Run C2ST on raw pixel features (or, better, on intermediate embeddings from the model's feature extractor) between the two datasets. Identify which features or embedding dimensions differ most. Examine metadata: are there differences in scanner manufacturer, imaging protocol, patient demographics?

2. **Subpopulation breakdown.** Compute accuracy stratified by available metadata (age, sex, disease subtype, image acquisition parameters) on both datasets. Identify whether the 71% is uniformly lower or driven by specific subgroups. If the model is 90% accurate on one subgroup at the new hospital but 40% on another, the problem is localized.

3. **Error analysis.** Examine the confusion matrix on the new dataset. Are certain classes disproportionately confused? If the model consistently confuses class A with class B at the new hospital, investigate whether the imaging protocol makes these classes less distinguishable.

4. **Calibration analysis.** Plot reliability diagrams for both datasets. If the model is overconfident on the new dataset, it suggests the model's uncertainty is not generalizing.

5. **Corruption robustness.** Test the model on synthetic corruptions of the original test set (blur, noise, contrast changes) at the severity levels that match the visual differences observed in the new hospital's images. This tests whether the degradation is explainable by image-quality differences.

6. **Remediation.** Fine-tune on a small labeled sample from the new hospital (domain adaptation). Alternatively, train with augmentations matching the new hospital's characteristics. Re-evaluate to confirm improvement.

---

**Q13: You are building an ML pipeline for a financial institution. How would you design the evaluation framework to be robust to distribution shift?**

**A:**

**During development:**
- Temporal validation: train on historical data, test on the most recent periods. Report performance per time period to visualize temporal degradation.
- Subpopulation analysis: report metrics stratified by customer segment, geography, product type, and demographic groups. Set minimum acceptable performance for each subgroup.
- Stress testing: evaluate on historical crisis periods (2008, COVID-2020) as natural OOD datasets. If the model will not encounter such events in training data, synthetically simulate extreme scenarios (e.g., 50% increase in unemployment, sudden interest rate change).

**Pre-deployment:**
- Canary evaluation: deploy to a small, randomly sampled fraction of traffic. Compare model predictions with the existing system. Flag any systematic disagreements for human review.
- Shadow mode: run the model in parallel with the production system for a trial period, collecting predictions without acting on them.

**Post-deployment monitoring:**
- Feature drift: nightly batch comparisons (MMD or C2ST) between daily input features and a reference window.
- Prediction drift: monitor the distribution of predicted probabilities, predicted class proportions, and prediction entropy.
- Performance drift: as ground truth arrives (with delay), compute rolling accuracy/AUC with change-point detection. Set automated alerts for statistically significant degradation.
- Fairness monitoring: track per-group metrics continuously. Alert on widening performance gaps.

**Retraining triggers:**
- Significant drift detected (p < 0.01 on MMD/C2ST sustained over multiple days).
- Rolling performance metric drops below a predefined threshold.
- New labeled data becomes available in sufficient quantity.

---

**Q14: A colleague argues that since your model is well-calibrated on the validation set, it should be reliable in production. How do you respond?**

**A:** Calibration is a distributional property—it measures the alignment between predicted probabilities and observed frequencies on a *specific* dataset drawn from a *specific* distribution. A model calibrated on the validation set is calibrated for the validation distribution. Under distribution shift, calibration can degrade arbitrarily.

Concretely: if the model predicts $P(\text{positive}) = 0.3$ for a certain input profile, and 30% of such inputs in the validation set are actually positive, the model is calibrated there. But if the base rate of the positive class increases at deployment (label shift), the true frequency at that predicted probability may now be 0.5. The model is confidently wrong.

Temperature scaling and Platt scaling, which are common calibration methods, are fit on the validation distribution and do not transfer to shifted distributions without re-fitting.

What I would recommend instead: (a) evaluate calibration on multiple distributions (temporal folds, OOD datasets); (b) use calibration metrics as one of several evaluation criteria, not a guarantee; (c) monitor calibration in production via rolling reliability diagrams.

---

### Debugging & Failure-Mode Questions

---

**Q15: Your production model's accuracy drops by 8% over six months, but a KS test on each feature shows no significant shift. What could explain this?**

**A:** Several possibilities:

1. **Multivariate shift that marginals miss.** The individual feature distributions may be unchanged, but the *correlations* between features have shifted. For example, income and age may each have the same marginal distribution, but the relationship between them changed (younger people now earn more). Per-feature KS tests cannot detect this. An MMD or C2ST on the full feature vector would.

2. **Concept drift.** $P(X)$ has not changed, but $P(Y|X)$ has. The same inputs now warrant different labels. No input distribution test (KS, MMD, C2ST) will detect concept drift because the inputs themselves are unchanged.

3. **Label shift.** The class proportions have changed, which affects metrics like accuracy even if the model's discrimination ability is unchanged. Check whether the class distribution of predictions or labels has shifted.

4. **Gradual shift.** The KS test compares two time windows. If the shift is gradual and each consecutive comparison window is similar to its predecessor, individual KS tests may fail to reach significance even though the cumulative drift is large. Comparing the current window to the original training data (not just the previous window) would detect this.

5. **Data pipeline issue.** A data engineering change (new encoding, missing value imputation change, feature computation bug) may alter the effective feature representation without changing the raw feature distributions that the KS test examines.

---

**Q16: You apply importance weighting to correct for covariate shift, but the model's performance on the target domain gets worse, not better. What went wrong?**

**A:**

1. **Extreme importance weights / high variance.** Some target samples fall in regions with very low training density, producing enormous importance weights. A single outlier weight dominates the loss, destabilizing optimization. Check the distribution of weights: if $\max(w) / \text{mean}(w) > 100$, the weights need clipping.

2. **Violated covariate shift assumption.** If concept drift is also present ($P(Y|X)$ has changed), importance weighting is correcting for the wrong thing—it adjusts the input distribution but uses the (now-incorrect) training labels.

3. **Poor density ratio estimation.** If the estimated $w(x)$ is inaccurate (e.g., because the density ratio model is misspecified), the correction is wrong. Validate the density ratio by checking that the reweighted training distribution matches the target distribution (e.g., reweighted feature means/variances should match target feature means/variances).

4. **Support violation.** There are target inputs with zero or near-zero training density. Importance weighting cannot extrapolate—it can only reweight existing training data. If the target has inputs that the model has literally never seen, reweighting does not help.

---

**Q17: Your team runs a C2ST and finds 78% accuracy distinguishing training from deployment data. A manager asks: "Is this a problem?" How do you respond?**

**A:** 78% accuracy means the classifier can distinguish the two datasets substantially better than chance (50%). This confirms there is a real, detectable distributional shift. However, this alone doesn't tell us whether the shift is *harmful*.

To determine whether it is a problem, I would investigate:

1. **Which features drive the shift?** Inspect feature importances from the C2ST classifier. If the shifting features are ones the production model relies on heavily, the risk is higher. If the shift is in features the model ignores, the impact may be negligible.

2. **How does the production model actually perform on the shifted data?** If labels are available, compute metrics directly. If not, use proxy signals (confidence distribution, prediction entropy).

3. **Is the shift benign or pathological?** Some shift is expected and harmless—e.g., a natural seasonal change that the model handles well. Other shift undermines core model assumptions.

My recommendation to the manager: "78% C2ST accuracy confirms significant distributional shift. The urgency depends on whether this shift affects the features the model relies on. I recommend we run a focused performance audit on the shifted data before deciding on retraining."

---

**Q18: Your model satisfies equalized odds on the test set. After deployment, a fairness audit reveals significant equalized-odds violations. No one changed the model. What happened?**

**A:** Several mechanisms can cause fairness constraints to break under shift even when the model is unchanged:

1. **Label shift (base rate change).** If the base rate $P(Y=1)$ changed differently across groups at deployment, the relationship between TPR/FPR and group membership is altered. By Chouldechova's theorem, when base rates differ and the classifier is imperfect, satisfying equalized odds becomes impossible. The test-set base rates may have been similar across groups, but deployment-time base rates may differ.

2. **Covariate shift within groups.** The input distribution within each group may have shifted differently. If Group A's inputs shifted toward harder-to-classify regions while Group B's did not, TPR for Group A will drop while Group B's remains stable.

3. **Differential concept drift.** The relationship $P(Y|X)$ may have changed differently for different groups (e.g., an economic policy affects one demographic more than another).

4. **Threshold sensitivity.** If the model uses a decision threshold calibrated on the test set, and the deployment-time score distribution has shifted, the same threshold produces different TPR/FPR trade-offs.

**Remedy:** Re-audit fairness on deployment data. If shift is confirmed, recalibrate thresholds per-group on deployment data, or retrain with Group DRO.

---

### Follow-Up and Probing Questions

---

**Q19: "You mentioned the median heuristic for MMD kernel bandwidth. Can you explain what it is and why it might fail?"**

**A:** The median heuristic sets the RBF kernel bandwidth $\sigma$ to the median of all pairwise Euclidean distances in the combined sample: $\sigma = \text{median}(\{||x_i - x_j||_2 : i < j\})$.

**Why it is used:** It provides a data-dependent, scale-adaptive bandwidth without hyperparameter tuning. At this scale, the kernel is sensitive to structure at the "typical" inter-point distance.

**When it fails:**
- **Multi-scale structure.** If the distribution has both fine-grained and coarse-grained structure, no single bandwidth captures both. The median heuristic picks one scale and misses the other. Multi-kernel MMD (averaging over several bandwidths) is more robust.
- **High dimensions.** In high dimensions, pairwise distances concentrate (all distances become similar), making the median less informative and the kernel less discriminative.
- **Vastly different sample sizes.** If one sample is much larger, the median distance is dominated by the larger sample's internal structure.

---

**Q20: "You said confidence monitoring can fail under concept drift because the model is 'confidently wrong.' Can you quantify this? How would you detect it?"**

**A:** Under concept drift, inputs remain in-distribution (they are familiar to the model), so the model's internal representations are in well-explored regions. Confidence remains high because the model's learned features still activate normally. But the mapping from features to labels has changed, so high confidence is assigned to the wrong class.

**Quantifying the problem:** The model's expected calibration error (ECE) will spike. In the bin where the model predicts $P(\text{positive}) \in [0.8, 0.9]$, the actual fraction of positives may now be 0.3 instead of 0.85. But you need labels to compute ECE, which may be delayed.

**Detection strategies:**
- **Agreement with simple baselines.** If a simple, recently updated model (e.g., logistic regression retrained weekly on recent data) disagrees systematically with the production model, this suggests concept drift.
- **Label arrival monitoring.** As labels trickle in, compute metrics immediately on the earliest available labels, even if the sample is small. A significant drop signals drift before full label coverage is available.
- **Monitoring stability of feature-label correlations.** Track mutual information or correlation between key features and labels on recent labeled data. A change signals concept drift.
- **Detect upstream structural changes.** Monitor for known external events (regulation changes, market shifts, policy updates) that are likely to alter $P(Y|X)$.

---

**Q21: "Walk me through how you would implement a complete drift detection and response system for a production recommendation engine."**

**A:**

**Layer 1: Input drift detection (real-time).**
- Compute feature statistics (means, variances, quantiles) on sliding windows (1-hour, 1-day). Compare to a reference window (the training data or a recent stable period). Use Page-Hinkley or CUSUM for change-point detection.
- Run C2ST weekly between the current week's data and the reference data. Alert if accuracy exceeds a calibrated threshold (e.g., 55%).

**Layer 2: Output drift detection (real-time).**
- Monitor the distribution of predicted item categories, recommendation scores, and diversity metrics. A sudden change in what the model recommends indicates either input shift or behavioral change.
- Track the click-through rate (CTR) on recommendations in real time. CTR is a near-immediate signal: if it drops, the recommendations are less relevant.

**Layer 3: Performance monitoring (delayed but definitive).**
- Track engagement metrics (CTR, conversion, session time) with appropriate delay. Use A/B test infrastructure if available to compare the model against a baseline.
- Run temporal evaluation monthly: retrain on data from the past 6 months, evaluate on the most recent month, compare to the production model.

**Response protocol:**
- **Minor drift, stable CTR:** Log and continue monitoring. Refresh the reference window.
- **Moderate drift, declining CTR:** Trigger accelerated retraining. Deploy a candidate model to shadow traffic.
- **Severe drift or CTR collapse:** Fall back to a simpler model (popularity-based) while retraining. Investigate root cause (data pipeline issue? external event?).
- **Subgroup-specific degradation:** Retrain with Group DRO or add targeted data collection for the affected subgroup.

---

**Q22: "Suppose you are told you can either (a) improve your model's average accuracy by 2% or (b) improve worst-subgroup accuracy by 8% while average accuracy drops by 1%. Which do you choose and why?"**

**A:** The answer depends on the application context, and saying so is the right starting point.

For a **safety-critical or fairness-regulated** application (medical diagnosis, criminal justice, lending), I choose (b) without hesitation. An 8% improvement for the worst-performing subgroup likely means significantly better outcomes for a vulnerable population. The 1% average accuracy drop is acceptable because the average was inflated by ignoring subgroup disparities. Regulators and ethical review boards evaluate worst-case, not average-case.

For a **low-stakes, high-volume** application (ad targeting, content recommendation), the calculus is different. A 2% average improvement might translate to millions of dollars in revenue, and the subgroup failure may have limited downstream harm. Even here, though, I would investigate: if the worst subgroup represents a demographic group, the reputational and legal risks of poor performance may outweigh the revenue from better average accuracy.

My default recommendation: choose (b) unless there is a strong, documented reason why average-case performance is strictly more important. The trend in ML practice and regulation is toward accountability for worst-case outcomes, and models that sacrifice minority subgroups for average gains are increasingly unacceptable.

---

**Q23: "How does Group DRO relate to CVaR optimization? Are they solving the same problem?"**

**A:** They are related but not identical. Both aim to improve worst-case performance, but they define "worst case" differently.

**Group DRO** assumes a *known partition* of the data into groups $\{g_1, \ldots, g_G\}$ (e.g., demographic groups) and minimizes the maximum group loss. It requires group labels.

**CVaR optimization** does not require group labels. Instead, it focuses on the worst-$\alpha$ fraction of *individual samples* by loss. It minimizes the average loss over the hardest $\alpha$-quantile of the loss distribution.

**Connection:** Group DRO can be viewed as a special case of distributional robust optimization (DRO) where the uncertainty set is $\{P : P = \sum_g q_g P_g, q \in \Delta^G\}$. CVaR-based DRO uses an uncertainty set defined by a divergence ball around the empirical distribution (e.g., chi-squared ball). When the worst-case distribution in this ball happens to concentrate mass on a single group, CVaR DRO recovers something similar to Group DRO.

**Practical difference:** If you know the relevant subgroups, Group DRO is more targeted—it directly addresses group-level disparities. If you do not know the relevant subgroups (or if failure modes cut across groups), CVaR is more flexible because it discovers the worst-case subpopulation implicitly through the loss distribution.

**Trade-off:** Group DRO is easier to interpret and monitor (you track per-group metrics). CVaR is more general but the "worst-case subpopulation" it optimizes for may not correspond to any meaningful real-world group.

---

**Q24: "You have a model in production. You detect significant covariate shift via MMD, but performance metrics (computed on delayed labels) show no degradation. Should you still act?"**

**A:** Not immediately, but I would not ignore it either. Here is my reasoning:

**Why no action yet:** The purpose of drift detection is to *predict* performance degradation. If direct performance measurement shows no degradation, the drift may be benign—the model generalizes well to the shifted inputs. Acting on every detected shift leads to unnecessary retraining costs and instability.

**Why I would not ignore it:**
- Labels are delayed. Current performance metrics reflect *past* performance, not *current* performance. The degradation from current shift may not appear in metrics for weeks or months.
- The shift may be progressing. Today's benign shift could accelerate. Continued monitoring is essential.
- The shift may affect subpopulations not yet visible in aggregate metrics.

**My response:**
1. Log the detection and the magnitude of shift.
2. Increase monitoring frequency for the affected features and for subgroup-level metrics.
3. Preemptively prepare a retraining pipeline with recent data so that if degradation does appear, response is fast.
4. Investigate the cause of the shift. If it is a known, expected change (seasonality, planned policy change), monitor but do not react. If it is unexpected, escalate attention.

This balances the cost of premature action against the risk of delayed response.

---

**[← Previous Chapter: Statistical Comparison](statistical_comparison_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Deployment →](deployment_guide.md)**
