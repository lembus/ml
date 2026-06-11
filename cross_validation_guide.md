# Cross-Validation: A Complete Reference Guide

## Preface

This guide is a standalone reference on cross-validation (CV) for machine learning. It assumes no prior familiarity with CV and builds every concept from first principles. The document is organized into eight major topics, each following a strict six-section structure (Motivation & Intuition → Conceptual Foundations → Mathematical Formulation → Worked Examples → Relevance to ML Practice → Common Pitfalls), followed by a comprehensive interview-preparation section.

The eight topics are:

1. k-Fold Cross-Validation (and stratified / repeated variants)
2. Leave-One-Out Cross-Validation (LOOCV)
3. Hold-Out Validation (simple train/test and time-based)
4. Time-Series Cross-Validation (rolling, expanding, blocked)
5. Grouped Cross-Validation (leave-one-group-out, group k-fold)
6. Spatial Cross-Validation (spatial blocking, location-based)
7. Nested Cross-Validation
8. Pitfalls & Best Practices (data leakage, independence assumptions)

Before diving in, it is useful to understand the overarching problem that all CV methods attempt to solve.

---

## Global Motivation: Why Cross-Validation Exists At All

Machine learning is fundamentally a game of generalization. You fit a model on some finite data, and then you deploy it on data you have never seen. The central question is: **how well will it perform on that unseen data?**

Training error (performance on the data used to fit the model) is a notoriously bad answer to this question. A flexible model can memorize its training set and achieve near-zero training error while performing disastrously on new data. A decision tree grown to purity has zero training error; a k-nearest-neighbor classifier with k=1 has zero training error. Neither tells you anything useful about generalization.

What we actually want is the **true risk** — the expected loss on a fresh sample drawn from the same distribution as the training data:

$$R(f) = \mathbb{E}_{(x,y)\sim P}[L(y, f(x))]$$

We cannot compute R(f) directly because we do not know P. We must **estimate** it from the finite data we have. Cross-validation is a family of techniques for building unbiased (or nearly unbiased) estimates of R(f) using only the training data.

The naive approach is to set aside a single hold-out set and evaluate on it. That works, but a single hold-out gives a noisy estimate (it depends on which points you happened to hold out) and wastes data (the model is trained on less than the full set). Cross-validation generalizes this idea: instead of one split, perform many splits and average. The average is a more stable, lower-variance estimate of R(f), and every data point gets to serve as a validation point exactly once.

The story gets more complicated when the iid assumption is violated. If your data has temporal ordering, groups, or spatial correlation, a random split leaks information from the test set into training, and the CV estimate becomes optimistic — sometimes spectacularly so. The specialized CV methods covered in this guide exist precisely to handle these non-iid regimes.

With that context set, we can dive into the individual methods.

---
---

# Topic 1: k-Fold Cross-Validation

## 1.1 Motivation & Intuition

Imagine you have 1,000 emails labeled as spam or not-spam, and you want to know how well a logistic regression classifier will perform on new emails you have not yet seen. You train the model on all 1,000 emails. Its training accuracy is 98%. Is that the number you would report to your boss?

Obviously not — the model has seen every one of those emails during training, so achieving high accuracy on them is not informative. What you really want is accuracy on emails the model has not seen.

A natural first idea is to hold out 200 of the emails randomly, train on the remaining 800, and test on the 200. Suppose you get 89%. That is more credible. But now suppose you did the same experiment again with a different random 200. You might get 91%, or 86%, or 93%. Which is right? All of them are valid estimates of the same underlying quantity (true accuracy), but each is noisy because it is based on only 200 test examples and one particular training-test split.

**k-fold cross-validation is the natural fix.** Instead of picking one hold-out set, partition the data into k equal pieces (say k=5, giving pieces of size 200). Train on 4 of them and test on the 5th. Then train on a different 4 and test on the remaining one. Repeat until every piece has served as the test set exactly once. Average the 5 accuracy numbers. You now have:

- An accuracy estimate based on effectively all 1,000 test predictions (each example was tested once, on a model that did not see it)
- Five fold-level accuracies that let you estimate the variability of your estimate
- A training procedure that used 800 of the 1,000 examples each time (only 20% reduction from the full dataset)

This is the core intuition. k-fold CV gives you a better-than-hold-out estimate of generalization performance for a small extra computational cost (k model fits instead of 1).

### Why this matters in real ML systems

- **Model selection**: Before shipping a model, you want to choose between candidates (logistic regression vs. random forest vs. gradient boosting). CV gives you a more reliable ranking than a single hold-out.
- **Hyperparameter tuning**: Choosing tree depth, regularization strength, or learning rate requires comparing many model configurations. CV reduces the noise in those comparisons.
- **Confidence intervals**: The spread of fold-level scores gives you a rough estimate of how precise your performance number is.
- **Debugging**: If one fold has wildly different performance than the others, you have learned something about your data — maybe there is a subgroup the model fails on.

## 1.2 Conceptual Foundations

### Key definitions

- **Dataset** $D = \{(x_i, y_i)\}_{i=1}^n$: the full set of labeled examples.
- **Learning algorithm** $\mathcal{A}$: a function that takes a dataset and returns a predictor. For example, "logistic regression with $L_2$ penalty $\lambda = 0.1$" is a learning algorithm.
- **Predictor** $f = \mathcal{A}(D)$: the trained model.
- **Loss function** $L(y, \hat{y})$: measures how bad prediction $\hat{y}$ is when the truth is $y$. Examples: 0-1 loss for classification, squared error for regression, log-loss for probabilistic prediction.
- **Fold**: one of the k disjoint subsets into which the data is partitioned.
- **Training fold(s)**: the k-1 folds used to fit the model in a given iteration.
- **Validation fold** (also "test fold" in CV context): the single fold held out for evaluation in a given iteration.

### The procedure step by step

1. Shuffle the data (random permutation). This is critical if the data was stored in any non-random order.
2. Partition into k disjoint folds $D_1, D_2, \ldots, D_k$ of roughly equal size. If n is not divisible by k, fold sizes differ by at most 1.
3. For $i = 1, 2, \ldots, k$:
   a. Train $f_{-i} = \mathcal{A}(D \setminus D_i)$ on all folds except the i-th.
   b. Compute the loss on the held-out fold: $e_i = \frac{1}{|D_i|} \sum_{(x,y) \in D_i} L(y, f_{-i}(x))$
4. The CV estimate is the average of fold losses: $\hat{R}_{CV} = \frac{1}{k} \sum_{i=1}^k e_i$.

An equivalent and slightly more common formulation averages over individual examples rather than folds:

$$\hat{R}_{CV} = \frac{1}{n} \sum_{i=1}^n L(y_i, f_{-\kappa(i)}(x_i))$$

where $\kappa(i)$ is the fold containing example $i$. These two formulations are identical when all folds have the same size and differ by a negligible reweighting when they do not.

### Stratified k-Fold

In classification, the class distribution matters. If your data is 95% class 0 and 5% class 1, a random partition into 5 folds might produce a fold with only 2% positives and another with 9% positives. The resulting per-fold accuracies are not comparable and the CV estimate has extra variance that has nothing to do with the model.

**Stratified k-fold CV** partitions the data so that each fold has approximately the same class distribution as the full dataset. For binary classification with 5% positives and k=5, each fold will have ~5% positives, give or take rounding. For multiclass, each class is stratified independently. For regression, "stratified" is less standard but analogous approaches bin the continuous target and stratify on the bins.

Stratification is almost always the correct default for classification. It costs nothing, it reduces variance of the CV estimate, and it prevents degenerate folds (e.g., a fold with zero positives, which would make AUC undefined).

### Repeated k-Fold

A single k-fold CV gives one estimate of performance based on one particular random partition. If you rerun with a different random seed, you get a slightly different number. Repeated k-fold runs the whole k-fold procedure R times with different random partitions and averages the R*k fold losses.

This trades computation (R*k model fits instead of k) for a lower-variance estimate of generalization error. It is particularly useful when:
- The dataset is small and fold-to-fold variability is high
- You are comparing models whose CV estimates are close and you need more precision to distinguish them
- You want tighter confidence intervals

For moderate-to-large datasets, repeated k-fold is usually overkill; a single 5- or 10-fold is sufficient.

### Assumptions

k-fold CV as described above assumes:

1. **IID data**: examples are independent and identically distributed. Specifically, examples are drawn independently from a common distribution $P$, and the test-time distribution is also $P$.
2. **Exchangeability of folds**: any permutation of the data would give essentially the same partition quality.
3. **No grouping structure**: examples are independent units, not nested within higher-level groups.
4. **No temporal structure**: the order of examples does not matter.
5. **No spatial structure**: examples are not correlated based on physical or network proximity.
6. **Test distribution equals train distribution**: there is no covariate shift or label shift at deployment.

### What breaks when assumptions are violated

- **Temporal dependence**: if today's stock price depends on yesterday's, randomly mixing training and test examples lets the model "see the future" during training. CV estimates become wildly optimistic.
- **Grouped data**: if multiple rows come from the same patient, user, or document, a random split puts some rows in train and others in test. The model can exploit person-specific signal and the CV estimate is overoptimistic.
- **Spatial correlation**: neighboring weather stations have correlated readings. A random split lets the model benefit from knowing a nearby station's reading, which does not generalize to locations far from any training station.
- **Class imbalance**: random folds may have wildly different positive rates, inflating variance. Stratification fixes this.
- **Covariate shift**: CV estimates risk under the training distribution, not the deployment distribution. If deployment data differs, CV underestimates deployment error even when every assumption above is satisfied.

## 1.3 Mathematical Formulation

### Notation setup

Let $D = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ be the dataset with $n$ examples, drawn iid from distribution $P$ over $\mathcal{X} \times \mathcal{Y}$. Let $\mathcal{A}$ be a learning algorithm and let $L: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}$ be a loss function. The **true risk** of the predictor $\mathcal{A}(D)$ is

$$R(\mathcal{A}, D) = \mathbb{E}_{(x,y) \sim P}[L(y, \mathcal{A}(D)(x))]$$

which is a random variable because $D$ is random. We often want to know the **expected risk** (averaged over training sets of size $n$):

$$\bar{R}_n = \mathbb{E}_D[R(\mathcal{A}, D)]$$

### The k-fold CV estimator

Partition $\{1, 2, \ldots, n\}$ into $k$ disjoint index sets $I_1, \ldots, I_k$. Let $D_{-j} = \{(x_i, y_i) : i \notin I_j\}$ denote the training set for fold $j$, and let $f_{-j} = \mathcal{A}(D_{-j})$. The k-fold CV estimator is

$$\hat{R}_{CV}(k) = \frac{1}{n} \sum_{j=1}^k \sum_{i \in I_j} L(y_i, f_{-j}(x_i))$$

Assuming fold sizes are equal at $n/k$, this can also be written as an average of fold-level mean losses:

$$\hat{R}_{CV}(k) = \frac{1}{k} \sum_{j=1}^k \underbrace{\frac{k}{n} \sum_{i \in I_j} L(y_i, f_{-j}(x_i))}_{\text{mean loss on fold } j \equiv e_j}$$

### What does CV actually estimate?

This is a subtle point that interviewers love. The k-fold estimator does **not** estimate $\bar{R}_n$ (expected risk of a model trained on $n$ examples). It estimates $\bar{R}_{n(1-1/k)}$ (expected risk of a model trained on $n(1-1/k)$ examples), because each $f_{-j}$ is trained on $n - n/k$ examples.

For large $n$ and moderate $k$, the difference is small. For small $n$ or small $k$, there is a pessimistic bias: the CV estimate is systematically higher (worse) than the performance of a model trained on all $n$ examples. This is why the final deployed model is typically retrained on the **full** dataset after CV has been used to select it — you use CV to estimate generalization of the modeling procedure, then ship a model trained on everything.

### Bias of the CV estimator

$$\text{Bias}[\hat{R}_{CV}(k)] = \mathbb{E}[\hat{R}_{CV}(k)] - \bar{R}_n = \bar{R}_{n(1-1/k)} - \bar{R}_n$$

For well-behaved learning curves (risk decreasing in training size), this bias is positive (pessimistic). It decreases as $k$ increases:

- $k = n$ (LOOCV): trained on $n-1$ examples — almost no bias.
- $k = 10$: trained on $0.9n$ — small bias.
- $k = 2$ (half-split): trained on $n/2$ — potentially substantial bias.

### Variance of the CV estimator

This is harder to pin down because the fold-level errors $e_1, \ldots, e_k$ are **not independent** — they use overlapping training sets. A classic decomposition (Bengio & Grandvalet, 2004) shows that there is no unbiased estimator of $\text{Var}[\hat{R}_{CV}]$ from a single run of CV. Practitioners often use the naive $\frac{1}{k} \text{Var}(e_1, \ldots, e_k)$ as a rough indicator, understanding that it ignores the correlation between folds.

Intuitively:
- **Smaller $k$** → training sets have less overlap between folds → lower correlation between $e_j$'s → closer to the "ideal" low-variance average, BUT each training set is smaller, so each $f_{-j}$ is a worse estimate of the full model's behavior, adding variance through the model instability.
- **Larger $k$** → training sets overlap more → higher correlation between $e_j$'s → the average is less effective at variance reduction, BUT each $f_{-j}$ is closer to the full model.

The classic rule of thumb: $k = 5$ or $k = 10$ balances these effects well in practice.

### Formal bias-variance of 5-fold vs 10-fold vs LOOCV

Let $\sigma^2_{pt}$ denote the variance of loss at a single point, and let $\rho$ denote the average correlation between $e_j$ and $e_{j'}$ for $j \ne j'$. Then roughly

$$\text{Var}[\hat{R}_{CV}(k)] \approx \frac{1}{k} \sigma^2_{fold} \cdot [1 + (k-1)\rho]$$

For $\rho = 0$: variance decreases linearly in $k$. For $\rho = 1$: variance is constant in $k$. Reality is in between — correlation grows with $k$, so the variance-reduction benefit of larger $k$ plateaus or reverses past a point. That point is typically around $k=10$ in practice.

## 1.4 Worked Examples

### Example A: 5-fold CV by hand on a small dataset

Suppose we have $n = 10$ examples, and we want to evaluate a classifier using 5-fold CV.

Data (features omitted, just IDs and labels):
```
ID:    1  2  3  4  5  6  7  8  9  10
y:     0  1  0  1  1  0  1  0  1  0
```

**Step 1**: Shuffle (suppose new order is: 3, 7, 1, 9, 5, 2, 10, 4, 8, 6).

**Step 2**: Split into 5 folds of size 2:
- Fold 1: {3, 7}
- Fold 2: {1, 9}
- Fold 3: {5, 2}
- Fold 4: {10, 4}
- Fold 5: {8, 6}

**Step 3**: For each fold, train on the other 8, predict the held-out 2. Suppose predictions and true labels are:

| Fold | Held-out | True y | Predicted | Correct? |
|------|----------|--------|-----------|----------|
| 1    | 3        | 0      | 0         | ✓        |
| 1    | 7        | 1      | 1         | ✓        |
| 2    | 1        | 0      | 1         | ✗        |
| 2    | 9        | 1      | 1         | ✓        |
| 3    | 5        | 1      | 1         | ✓        |
| 3    | 2        | 1      | 0         | ✗        |
| 4    | 10       | 0      | 0         | ✓        |
| 4    | 4        | 1      | 1         | ✓        |
| 5    | 8        | 0      | 0         | ✓        |
| 5    | 6        | 0      | 1         | ✗        |

**Step 4**: Fold accuracies: $e_1 = 1.0, e_2 = 0.5, e_3 = 0.5, e_4 = 1.0, e_5 = 0.5$.

**Step 5**: CV estimate of accuracy: $\hat{R}_{CV} = (1.0 + 0.5 + 0.5 + 1.0 + 0.5) / 5 = 0.7$.

(Equivalently: 7 correct out of 10 held-out predictions = 0.7.)

Standard deviation across folds: $\sqrt{(0.3^2 + 0.2^2 + 0.2^2 + 0.3^2 + 0.2^2)/5} \approx 0.245$. This is a rough uncertainty estimate, though with only 10 examples it is not to be trusted too far.

### Example B: Stratified 5-fold on imbalanced data

Dataset: 1,000 transactions, 20 are fraud (class 1), 980 are legit (class 0). Base rate = 2%.

**Unstratified 5-fold** randomly partitions into 5 folds of 200. The expected number of positives per fold is 4, but the actual count follows a hypergeometric distribution. The probability that at least one fold has zero positives is non-trivial (roughly 10-15%). If it happens, you cannot compute AUC for that fold, and fold-level precision/recall are undefined.

**Stratified 5-fold** partitions positives and negatives separately:
- Positives (20): split into folds of size 4 each → each fold has exactly 4 positives.
- Negatives (980): split into folds of size 196 each → each fold has exactly 196 negatives.
- Combine: each fold has 200 examples, 4 positives, 196 negatives. Exactly 2% positive in every fold.

Now every fold can compute AUC, precision, recall, F1 reliably, and fold-to-fold variance is reduced.

### Example C: Repeated 5-fold to tighten a confidence interval

You are comparing two models on a medical dataset of 300 patients. Single 5-fold CV:
- Model A accuracy: 0.83, fold SD: 0.04
- Model B accuracy: 0.85, fold SD: 0.05

The difference is 0.02, smaller than the SDs. Is model B really better? Hard to tell from one run.

You run repeated 5-fold with $R = 20$ repeats. Now you have 100 fold-level accuracies per model. You can do a proper paired test (each repeat produces a matched pair) and get tighter estimates:
- Model A accuracy: 0.832 ± 0.012
- Model B accuracy: 0.851 ± 0.013
- Difference: 0.019 ± 0.008 (paired)

The paired SE of the difference is smaller than individual SEs because the same folds are used for both models in each repeat, canceling out fold-specific difficulty. Now you can more confidently say B is better.

### Example D: Why k=2 can be badly biased

You have 100 training examples and train a deep learning model that has a steep learning curve — going from 50 to 100 training examples reduces error from 20% to 12%. 

- **k=2 CV**: each fold model is trained on 50 examples, has ~20% error → CV estimate ≈ 20%.
- **k=10 CV**: each fold model is trained on 90 examples, has ~13% error → CV estimate ≈ 13%.
- **Model retrained on all 100**: error ~12%.

The k=2 estimate is 8 percentage points too pessimistic. The k=10 estimate is only 1 point off. When learning curves are steep (small data, high-capacity models), large k matters.

## 1.5 Relevance to ML Practice

### Where k-fold CV is standard

- **Tabular ML with small-to-medium data (say, <100k rows)**: 5-fold or 10-fold is the default for model selection, feature selection, and hyperparameter tuning.
- **Kaggle competitions**: a "trust your CV" culture exists because public leaderboard scores are noisy and overfittable. Teams rely on internal CV.
- **Biostatistics and clinical ML**: small datasets make CV essential for credible performance reporting.
- **Traditional ML pipelines**: scikit-learn's `cross_val_score` and `GridSearchCV` build in k-fold as the default.

### Where k-fold CV is NOT the right tool

- **Very large datasets (>1M rows)**: a single hold-out is usually fine. The variance of the hold-out estimate is already low because the test set is large, and k model fits on a huge dataset is expensive.
- **Deep learning**: training a single model is expensive (hours to days). Doing it k times for every experiment is often infeasible. Most deep learning workflows use a single fixed validation set, sometimes combined with multiple random seeds for variance estimation.
- **Time-series data**: random partitions violate temporal ordering. Use time-series CV (Topic 4).
- **Grouped data**: random partitions leak group-level information. Use group CV (Topic 5).
- **Spatial data with autocorrelation**: use spatial CV (Topic 6).

### Trade-offs

| Choice | Pros | Cons |
|---|---|---|
| Small k (e.g., k=3) | Cheap; folds more independent | Pessimistic bias; noisier estimate |
| Large k (e.g., k=20) | Small bias | Expensive; folds highly correlated, variance reduction plateaus |
| k=5 | Good default balance | — |
| k=10 | Slightly smaller bias than k=5 | 2x more expensive |
| Stratified | Lower variance for classification | Slightly more complex |
| Repeated | Tighter estimates | R-fold more expensive |

### Computational cost

Training k models is k-times the cost of training one. For a model that takes T seconds and memory M:

- 5-fold: 5T time, M memory (sequential), 5M memory (parallel).
- 10-fold: 10T, up to 10M parallel.
- Repeated 10-fold with 5 repeats: 50T.

With modern parallelism, k-fold is embarrassingly parallel (each fold is independent). On a machine with k cores, wall-clock time is roughly T (one model fit) plus overhead.

## 1.6 Common Pitfalls & Misconceptions

### Pitfall 1: Preprocessing before splitting ("leakage through scaling")

Bad:
```python
X_scaled = StandardScaler().fit_transform(X)  # uses all data
cross_val_score(model, X_scaled, y, cv=5)
```

The scaler has used the means and standard deviations of the entire dataset, including the validation folds. Each fold's scaling incorporates information from the "test" portion. This biases the CV estimate optimistically.

Good:
```python
pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
cross_val_score(pipeline, X, y, cv=5)
```
Now the scaler is refit on the training portion of each fold.

The same issue arises with feature selection (e.g., "keep the top 50 features by correlation with y" applied to all data before CV), target encoding, SMOTE/class balancing, imputation with the global mean, PCA, and any transformation that uses target information or aggregate statistics. **Rule**: any transformation that uses data-dependent parameters must be fit inside each CV fold.

### Pitfall 2: Using CV score as the final reported performance when you also tuned hyperparameters

If you do hyperparameter search using CV and report the best CV score, that number is optimistically biased because you selected the max over a noisy set of candidates (a form of multiple testing / selection bias). The correct approach is nested CV (Topic 7) or a separate held-out test set that is only touched once.

### Pitfall 3: Thinking stratification helps regression

Stratification is a classification-centric concept. For regression, the analog is binning the target into quantiles and stratifying on the bins, or using continuous stratification strategies. Blindly applying stratified CV to a regression task (most libraries will refuse) does nothing useful.

### Pitfall 4: Using a single fold's score as "the" performance

The fold-to-fold variance is informative. Reporting only the mean hides it. Always report mean and either SD or SE across folds.

### Pitfall 5: Different k for different experiments

If you use 5-fold for model A and 10-fold for model B, the comparison is apples-to-oranges because the training set sizes differ. Always use the same CV protocol for all models you are comparing.

### Pitfall 6: Random seeds that make results look better

Running many random seeds and reporting the best is cherry-picking. Fix seeds in advance and report all runs.

### Pitfall 7: Forgetting to shuffle

If data is stored in a non-random order (e.g., all class 0 then all class 1, or ordered by time), non-shuffled k-fold creates degenerate folds. Always shuffle explicitly (or ensure the CV utility shuffles).

### Pitfall 8: Overestimating the precision of CV scores

The standard error of a 10-fold CV mean is not simply $SD/\sqrt{10}$ because folds are correlated. Hypothesis tests that ignore this (e.g., a naive t-test on the 10 fold scores) have inflated Type I error. For statistically rigorous model comparison, use corrected tests (e.g., Nadeau-Bengio corrected t-test) or bootstrap.

---
---

# Topic 2: Leave-One-Out Cross-Validation (LOOCV)

## 2.1 Motivation & Intuition

LOOCV is the extreme case of k-fold CV where $k = n$, the number of examples. Each fold contains exactly one example. You train $n$ models, each trained on $n-1$ examples, and test each on the single left-out point. Then you average the $n$ per-point losses.

**Why would you do this?** Two reasons:

1. **Maximum training data per fold**: each model sees $n-1$ examples, which is as close to the full dataset as possible. So the bias of LOOCV (as an estimator of the full-data model's risk) is minimal.
2. **Deterministic**: unlike 5-fold, LOOCV does not depend on a random partition. Given the data, LOOCV produces the same number every time.

**Why would you NOT do this?** Two reasons:

1. **Computational cost**: $n$ model fits instead of 5 or 10. For $n = 10,000$, that is 1,000x more expensive than 10-fold. For most models, this is a deal-breaker.
2. **High variance**: this is counterintuitive but important. The $n$ leave-one-out models are nearly identical to each other (differing by one training point), so their errors are highly correlated. The average of highly correlated random variables has higher variance than the average of independent ones — LOOCV's variance-reduction benefit from $n$ folds is offset by this correlation.

So LOOCV is a tool for a specific niche: **small datasets where the bias of k-fold is unacceptable, and models that are cheap to refit (especially those with leave-one-out shortcuts).**

Concrete example: you have 30 patients in a rare-disease study. 5-fold CV would train each fold model on 24 patients — a 20% reduction that might matter a lot for model stability. LOOCV trains each fold model on 29 patients, barely different from the full 30. For such a small dataset, LOOCV's near-zero bias is worth its variance penalty.

## 2.2 Conceptual Foundations

### Procedure

1. For $i = 1, 2, \ldots, n$:
   a. Leave out example $(x_i, y_i)$.
   b. Train $f_{-i} = \mathcal{A}(D \setminus \{(x_i, y_i)\})$ on the remaining $n - 1$ examples.
   c. Compute the loss for the held-out point: $e_i = L(y_i, f_{-i}(x_i))$.
2. Report $\hat{R}_{LOO} = \frac{1}{n} \sum_{i=1}^n e_i$.

### Why LOOCV has high variance (a deeper look)

A common misunderstanding: "LOOCV uses $n$ folds, so its estimate should be very precise, with variance $\sigma^2 / n$." This is wrong because of correlation.

Consider two leave-one-out fits $f_{-i}$ and $f_{-j}$. Both are trained on $n-1$ points, and their training sets overlap in $n-2$ points. The two models are extremely similar. Their errors on random test points would be similar as well. This correlation does not just persist between pairs — it is pervasive across all LOOCV fits.

Write $\hat{R}_{LOO} = \frac{1}{n} \sum_i e_i$. Then

$$\text{Var}[\hat{R}_{LOO}] = \frac{1}{n^2} \left[ \sum_i \text{Var}(e_i) + \sum_{i \ne j} \text{Cov}(e_i, e_j) \right]$$

If the $e_i$'s are independent, the cross-covariance is zero and the variance scales as $1/n$. If they are perfectly correlated, the variance is $\text{Var}(e_1)$ — independent of $n$. Reality is in between, but closer to correlated than independent for stable learners.

A classical result (Hastie, Tibshirani, Friedman, Elements of Statistical Learning, ch. 7): LOOCV has higher variance than 10-fold CV for most practical learning algorithms and realistic sample sizes.

### The stability connection

A **stable** learning algorithm is one whose output does not change much when a single training point is added or removed. LOOCV's variance depends critically on stability:
- Stable algorithms (linear regression, regularized methods with large $\lambda$): LOOCV works well.
- Unstable algorithms (deep decision trees, kNN with small k, neural networks): LOOCV can be badly noisy.

### Assumptions

All standard CV assumptions (iid, no groups, no temporal/spatial structure) plus:
- **Model can tolerate $n-1$ training points**: some algorithms (factorization methods, certain iterative estimators) have convergence issues on tiny changes to training set.
- **Cheap refit or closed-form LOO**: without this, LOOCV is computationally hopeless.

## 2.3 Mathematical Formulation

### Closed-form LOOCV for linear regression

For ordinary least squares with design matrix $X \in \mathbb{R}^{n \times p}$ and targets $y \in \mathbb{R}^n$, the fitted values are $\hat{y} = X\hat{\beta} = Hy$ where $H = X(X^T X)^{-1} X^T$ is the hat matrix.

The LOOCV squared-error loss has a remarkable closed form — you do **not** need to refit the model $n$ times:

$$\hat{R}_{LOO} = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat{y}_i}{1 - h_{ii}} \right)^2$$

where $h_{ii}$ is the $i$-th diagonal element of $H$ (the "leverage" of point $i$).

This is called the **PRESS statistic** (Predicted REsidual Sum of Squares, or sometimes divided by $n$).

**Derivation sketch**: Using the Sherman-Morrison formula for a rank-one update of the inverse $(X^T X)^{-1}$, one can show that the LOO prediction at point $i$ is

$$\hat{y}_i^{(-i)} = \hat{y}_i - \frac{h_{ii}(y_i - \hat{y}_i)}{1 - h_{ii}}$$

so the LOO residual is

$$y_i - \hat{y}_i^{(-i)} = \frac{y_i - \hat{y}_i}{1 - h_{ii}}$$

Squaring and averaging gives the PRESS formula. The beautiful consequence: to compute LOOCV error for linear regression, you only need to fit the model **once** and inspect the residuals and leverages. The $n$ refits collapse into a single fit plus $n$ divisions.

### Generalized cross-validation (GCV)

For very large $n$, even computing all the $h_{ii}$ values is expensive. **GCV** replaces each $h_{ii}$ with the average $\bar{h} = \text{tr}(H)/n = p/n$ (for full-rank OLS, $\text{tr}(H) = p$):

$$\hat{R}_{GCV} = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat{y}_i}{1 - p/n} \right)^2 = \frac{RSS/n}{(1 - p/n)^2}$$

GCV approximates LOOCV and is particularly useful for selecting smoothing parameters in ridge regression, splines, and kernel methods.

### LOOCV for ridge regression

For ridge regression with penalty $\lambda$, replace $H$ with $H_\lambda = X(X^T X + \lambda I)^{-1} X^T$. The PRESS formula applies with $h_{ii}$ being the diagonal of $H_\lambda$:

$$\hat{R}_{LOO}(\lambda) = \frac{1}{n} \sum_{i=1}^n \left( \frac{y_i - \hat{y}_i(\lambda)}{1 - h_{ii}(\lambda)} \right)^2$$

This gives an efficient way to tune $\lambda$ by evaluating LOOCV at many values of $\lambda$ from a single eigendecomposition of $X^T X$.

### Bias of LOOCV

LOOCV estimates the expected risk of a model trained on $n-1$ examples. For large $n$, this is essentially identical to the expected risk of a model trained on $n$ examples. Specifically, the bias is approximately

$$\bar{R}_{n-1} - \bar{R}_n \approx -\frac{\partial \bar{R}_m}{\partial m}\bigg|_{m=n}$$

which is small when the learning curve has flattened (large $n$, or a low-capacity model).

### Variance of LOOCV

There is no closed form for the variance of LOOCV in general, but bounds exist in terms of algorithm stability. If the algorithm has **uniform stability** $\beta$ (meaning changing one training point changes any prediction by at most $\beta$), then

$$\text{Var}[\hat{R}_{LOO}] \leq O(\beta + 1/n)$$

For ridge regression with penalty $\lambda$, $\beta \propto 1/\lambda$, so heavier regularization means more stable LOOCV.

## 2.4 Worked Examples

### Example A: LOOCV for OLS on a tiny dataset

Fit $y = \beta_0 + \beta_1 x$ to 5 points:
```
x: 1  2  3  4  5
y: 2  3  5  4  6
```

OLS fit: $\hat{\beta}_0 = 1.3, \hat{\beta}_1 = 0.9$. Fitted values $\hat{y}$:
```
x:      1    2    3    4    5
y:      2    3    5    4    6
ŷ:     2.2  3.1  4.0  4.9  5.8
resid: -0.2 -0.1  1.0 -0.9  0.2
```

The hat matrix $H = X(X^T X)^{-1} X^T$ with $X$ being the design matrix with intercept column. For this simple case, the leverage formula gives:

$$h_{ii} = \frac{1}{n} + \frac{(x_i - \bar{x})^2}{\sum_j (x_j - \bar{x})^2}$$

With $\bar{x} = 3$, $\sum(x_i - 3)^2 = 4 + 1 + 0 + 1 + 4 = 10$:
```
x:    1    2    3    4    5
h_ii: 0.6  0.3  0.2  0.3  0.6
```

(Note: the endpoints have high leverage; the center has low leverage.)

LOO residuals $= \text{resid}_i / (1 - h_{ii})$:
```
x:         1        2        3         4         5
resid:    -0.2     -0.1      1.0      -0.9      0.2
1 - h:     0.4     0.7      0.8        0.7      0.4
LOO res:  -0.5    -0.143    1.25     -1.286    0.5
```

LOO squared errors: $0.25, 0.0204, 1.5625, 1.654, 0.25$.

LOOCV MSE: $(0.25 + 0.0204 + 1.5625 + 1.654 + 0.25)/5 = 0.7474$.

In-sample MSE (for comparison): $(0.04 + 0.01 + 1.0 + 0.81 + 0.04)/5 = 0.38$.

The LOOCV estimate is almost double the in-sample MSE, revealing the optimism of training error. Note especially the outlier-like points $x=3$ and $x=4$: their LOO residuals are much larger than their training residuals because removing them and predicting them from the rest of the data is hard.

### Example B: LOOCV for kNN

Dataset: 50 points in 2D, binary labels. Model: kNN with $k=1$.

LOOCV for 1-NN has a neat property: for each point $x_i$, the LOO prediction is the label of its **nearest neighbor other than itself**. You can compute all LOOCV predictions in one pass by finding the 2-nearest neighbors of each point (itself and its nearest other point) and using the second one's label.

If 40 of 50 points get this right, LOOCV accuracy is 80%.

Compare to 1-NN's training accuracy: 100% (each point is its own nearest neighbor). LOOCV reveals the real-world performance that training error hides.

### Example C: Illustrating LOOCV's variance

Consider two datasets, both with $n = 100$, drawn from the same distribution. For both, compute:
- Hold-out (80-20): one number.
- 5-fold CV: average of 5.
- LOOCV: average of 100.

Hypothetical results:
```
                  Dataset 1   Dataset 2
Hold-out:         0.82         0.78
5-fold CV:        0.81         0.82
LOOCV:            0.80         0.84
```

Here LOOCV gave different numbers on the two datasets (0.80 vs 0.84, range of 0.04), while 5-fold gave similar numbers (0.81 vs 0.82, range of 0.01). This is counterintuitive — why does the estimator with "more folds" have higher run-to-run variance?

The answer is the correlation of LOOCV errors within a dataset: removing one of 100 points barely changes the model, so $f_{-1}$ and $f_{-2}$ are nearly identical. If they both systematically fail on the same test structure (e.g., a hard subregion of the data), all 100 LOOCV errors reflect this shared failure. The "100 independent evaluations" is an illusion; there is really just one evaluation averaged over 100 test points.

### Example D: PRESS in action

For ridge regression choosing $\lambda$, compute LOOCV MSE at $\lambda \in \{0.01, 0.1, 1, 10, 100\}$ using PRESS. On a dataset of $n=200$, $p=50$:

```
λ       Training MSE   LOOCV MSE (PRESS)
0.01    0.42           0.95
0.1     0.48           0.74
1       0.55           0.68      ← minimum
10      0.71           0.73
100     1.05           1.08
```

Training MSE is monotonically increasing in $\lambda$ (regularization biases the fit). LOOCV is U-shaped with a minimum at $\lambda = 1$. You pick $\lambda = 1$ and refit on all data.

This entire process required one eigendecomposition of $X^T X$ and element-wise computations for each $\lambda$ — no $n$-fold refitting needed.

## 2.5 Relevance to ML Practice

### When LOOCV is the right choice

- **Small datasets** (say, $n \lesssim 50$): the bias of 5- or 10-fold is substantial, and LOOCV is computationally feasible.
- **Linear models with closed-form LOO** (OLS, ridge, Gaussian processes): free LOOCV is a strong reason to use it.
- **Smoothing parameter selection** in splines, kernel methods: GCV approximation of LOOCV is standard.
- **Almost-unbiased estimation of generalization error** for publishing small-sample studies.

### When LOOCV is wrong

- **Moderate or large datasets**: 5- or 10-fold is cheaper and has lower variance.
- **Unstable learners** (deep NNs, decision trees without pruning): variance is too high.
- **Expensive models** (anything requiring GPU training): $n$ refits is infeasible.
- **Classification metrics requiring multiple examples per fold** (e.g., AUC): LOOCV gives one prediction per fold, so AUC is computed over all $n$ predictions pooled — which is fine, but if you want per-fold metrics, LOOCV does not produce them.

### Common alternatives

- **10-fold CV**: the standard workhorse, almost always better than LOOCV for practical purposes.
- **Bootstrap .632+**: attempts to combine the low bias of LOOCV with lower variance.
- **Repeated k-fold**: higher confidence at moderate computational cost.

## 2.6 Common Pitfalls & Misconceptions

### Pitfall 1: "LOOCV is always the best because it uses the most data"

Wrong. LOOCV's high variance often makes it less reliable than 10-fold for model selection, even though it has lower bias.

### Pitfall 2: Using LOOCV with unstable learners

LOOCV + a full-depth decision tree is a recipe for noise. The tree structure can flip wildly with one point removed, making fold-level errors nearly meaningless individually.

### Pitfall 3: Computing LOOCV naïvely for linear models

If you have OLS or ridge regression and you are not using PRESS/GCV, you are doing orders of magnitude more work than necessary. Always use the closed form when available.

### Pitfall 4: Conflating LOOCV with jackknife

The jackknife is a related resampling scheme for bias and variance estimation of statistics, not for model generalization. They share the "leave one out" mechanic but answer different questions.

### Pitfall 5: Forgetting that closed-form LOO assumes the model is the same

The PRESS formula for OLS assumes the same linear model (same features, same regularization) is fit on the leave-one-out data. If your "model" includes a step like "select the best 5 features from 50 using LOO-correlation", each LOO fit may select different features, and the closed form is invalid — you would need to redo feature selection in each fold.

---
---

# Topic 3: Hold-Out Validation (Simple Train-Test Split and Time-Based Splitting)

## 3.1 Motivation & Intuition

The hold-out method is the simplest possible generalization-estimation scheme: split your data once into a training set and a test set, fit the model on the training set, and evaluate on the test set. Done.

**Why bother with anything else?** Two reasons:

1. **Noise**: a single split gives a single noisy estimate. You might get lucky or unlucky with the split.
2. **Data waste**: if you hold out 30% of your data for testing, you only train on 70%.

For small datasets, these downsides are severe, which is why k-fold CV was invented. But for **large datasets**, the hold-out method is often preferred for three pragmatic reasons:

- With 10 million examples, even a 10% held-out test set has 1 million examples — more than enough for a precise estimate.
- Training on 90% vs 100% of a huge dataset makes negligible difference in model quality (learning curves have plateaued).
- Training one model vs. k models can be the difference between a 4-hour experiment and a 40-hour experiment.

The other major use of hold-out is **time-based splitting**: when your data has a temporal structure, the correct way to split is chronologically, not randomly. This is a form of hold-out specialized for non-iid temporal data.

### Concrete example: a recommendation system at scale

Netflix has billions of user-item ratings. They want to evaluate a new recommender. 5-fold CV would require training the model 5 times, each on 4/5 of the data. With a training time of 6 hours per model, that is 30 hours for one experiment.

Alternative: split chronologically. Train on all ratings before 2023-01-01, test on ratings after. Single 6-hour training run. The test set (all ratings in Q1-Q4 2023) is easily millions of examples — the hold-out estimate is already very precise. Plus, the chronological split mimics production: predict future ratings from past.

## 3.2 Conceptual Foundations

### Simple random hold-out

**Procedure:**
1. Shuffle the data.
2. Split into two parts: training (typically 70-80%) and test (20-30%).
3. Train on the training portion.
4. Evaluate on the test portion.
5. Report the test performance.

Variants:
- **70/30 split**: common default for small-to-medium datasets.
- **80/20**: slight preference for more training data.
- **90/10**: for very large datasets where even 10% is plenty of test data.
- **Three-way split (train/val/test)**: train on 60%, validate on 20% (for hyperparameter tuning), test on 20% (for final reporting). The test set is used exactly once.

### Time-based (chronological) split

For data with a time index, the only valid way to split is chronologically. All training points have timestamps earlier than all test points.

**Procedure:**
1. Sort data by time.
2. Pick a cutoff time $t^*$.
3. Training set: all examples with timestamp $< t^*$.
4. Test set: all examples with timestamp $\geq t^*$.

The cutoff is often chosen so that the test set has enough data (e.g., 20% of examples, or the last 3 months, or the most recent full year).

### Key assumptions

- **For random hold-out**: iid data — the standard CV assumptions.
- **For time-based hold-out**: test distribution is "future" of train distribution; you expect some drift but want to evaluate under it. Explicitly, no iid assumption — instead, you assume the temporal ordering is meaningful and training should only see the past.

### What breaks

- **Random hold-out on time series**: the model learns from future data points to predict past ones, which is not what happens in deployment. Optimistic results.
- **Random hold-out on grouped data**: leaks group information (same patient in train and test).
- **Time-based hold-out when data patterns are stationary but non-iid (e.g., clustered)**: can underestimate generalization to new clusters.
- **Small datasets**: hold-out has too much variance regardless of the split method.

### Three-way vs. two-way splits

In hyperparameter tuning or model selection, you are implicitly "cheating" on your test set if you use its performance to pick between models. You have optimized a choice using the test data, so the chosen model's test performance is optimistically biased.

The three-way split solves this:
- **Train**: fit models.
- **Validation**: compare models, tune hyperparameters.
- **Test**: after everything is decided, evaluate once to report the number.

Only looking at the test set once is the only way to get an honest generalization estimate.

## 3.3 Mathematical Formulation

Let $D = \{(x_i, y_i)\}_{i=1}^n$. Partition into disjoint sets $D_{\text{train}}$ of size $m$ and $D_{\text{test}}$ of size $n - m$. Define $f = \mathcal{A}(D_{\text{train}})$.

The hold-out estimate is

$$\hat{R}_{\text{HO}} = \frac{1}{n-m} \sum_{(x,y) \in D_{\text{test}}} L(y, f(x))$$

### Bias and variance

**Bias**: $\hat{R}_{\text{HO}}$ estimates $\bar{R}_m$ (risk of a model trained on $m$ examples), not $\bar{R}_n$. Pessimistic bias by the same argument as k-fold (the learning curve).

**Variance**: conditional on $D_{\text{train}}$, the test set is an iid sample, so

$$\text{Var}[\hat{R}_{\text{HO}} | D_{\text{train}}] = \frac{\sigma^2_L(f)}{n - m}$$

where $\sigma^2_L(f)$ is the per-example loss variance under distribution $P$ for the specific predictor $f$. Marginally (over training set draws), there is additional variance from training set randomness:

$$\text{Var}[\hat{R}_{\text{HO}}] = \underbrace{\mathbb{E}_{D_{\text{train}}}\left[\frac{\sigma^2_L}{n-m}\right]}_{\text{test-set noise}} + \underbrace{\text{Var}_{D_{\text{train}}}[R(f)]}_{\text{training instability}}$$

### Sizing the test set

Given a confidence level $1 - \alpha$ and a desired half-width $\epsilon$, the test set size needed for a classification accuracy estimate (normal approximation) is approximately

$$n_{\text{test}} \approx \frac{z_{\alpha/2}^2 \cdot \hat{p}(1-\hat{p})}{\epsilon^2}$$

For $\alpha = 0.05$, $z = 1.96$, $\hat{p} = 0.8$ (assumed accuracy), and $\epsilon = 0.01$ (want estimate within 1%):

$$n_{\text{test}} \approx \frac{(1.96)^2 \cdot 0.16}{0.0001} \approx 6147$$

If your dataset has 60,000 examples, a 10% hold-out (6,000) is right at the boundary. A 20% hold-out (12,000) comfortably beats it.

For smaller datasets, this formula shows why single hold-outs are inadequate — you cannot get tight intervals without burning through your data.

### Time-based split: what it estimates

$\hat{R}_{\text{HO, time}}$ does **not** estimate risk under a fixed distribution $P$. It estimates risk under the deployment distribution $P_{\text{deploy}}$ approximated by the test period. If $P_{\text{deploy}}$ differs from $P_{\text{train}}$ (covariate shift), this is exactly what you want — you are measuring how the model performs under realistic shift.

## 3.4 Worked Examples

### Example A: Simple random hold-out

Dataset: 1,000 customer records, binary churn label.
- Shuffle.
- Train: 700 records.
- Test: 300 records.
- Fit logistic regression.
- Test accuracy: 0.82.

Standard error for accuracy estimate:
$$SE = \sqrt{\frac{0.82 \cdot 0.18}{300}} \approx 0.022$$

95% CI: $0.82 \pm 1.96 \cdot 0.022 = [0.776, 0.864]$. A fairly wide interval despite 300 test examples.

### Example B: Three-way split for hyperparameter tuning

Dataset: 10,000 examples.
- Train: 6,000.
- Validation: 2,000.
- Test: 2,000.

Procedure:
1. Fit gradient boosting with 20 hyperparameter configurations on the 6,000 training examples.
2. Evaluate each on the 2,000 validation examples. Pick the best.
3. Optionally, retrain the best config on train+validation combined (8,000 examples).
4. Evaluate once on the 2,000 test examples. Report that number.

Step 4's result is the honest estimate. The best-validation number is optimistic because it was selected.

### Example C: Time-based split for a forecasting model

Dataset: daily sales for 1,000 stores, over 4 years (~1.46M rows).

Time-based split:
- Train: years 1-3 (~1.1M rows).
- Test: year 4 (~365k rows).

Fit a regression model for daily sales given features. Evaluate on year 4. The test performance includes any seasonality shift, changes in consumer behavior, and new stores opened in year 4.

Compare to random 80/20 hold-out: you would be training with examples from year 4 and testing on examples from year 1. The model could learn "consumer behavior pattern X emerged in year 4" from training examples and use it to predict year 1 — which, in real deployment, is impossible.

### Example D: Out-of-time validation for fraud detection

Dataset: 1 year of transactions, ~10M labeled for fraud.

Bad split: random 80/20 → test accuracy 99.5%.
Good split: train on months 1-10, validate on month 11, test on month 12 → test accuracy 97.8%.

The random split overestimates performance by 1.7 percentage points because fraud patterns evolve monthly — in deployment, you are always predicting next month's fraud based on past months. The time-based split reveals true deployment performance.

### Example E: When hold-out beats k-fold

Dataset: 5 million emails, spam classification.
- Single 80/20 split. Test set = 1 million examples.
- Test accuracy: 0.9912 ± 0.0001.

A 5-fold CV would train 5 models (very expensive) and produce an estimate with essentially the same precision but at 5x the compute. The hold-out is strictly better here.

## 3.5 Relevance to ML Practice

### Where hold-out dominates

- **Industrial ML with large data**: time and compute matter; a single split is enough.
- **Deep learning**: standard practice is train/val/test, with the validation set used throughout training for early stopping.
- **Any time you need to deploy a model in the loop with an ongoing data collection system**: the deployment boundary IS the split.
- **Competitions with a "held-out leaderboard"**: the organizer's hold-out is the test.

### Where time-based hold-out is essential

- **Forecasting** (demand, weather, financial).
- **Recommender systems**: user preferences drift.
- **Fraud detection**: adversaries adapt.
- **Anything with concept drift**.

### Trade-offs

| Aspect | Random hold-out | Time-based hold-out |
|---|---|---|
| Assumes iid | Yes | No |
| Respects deployment | Sometimes | Yes, if deployment is "future" |
| Data efficiency | Modest | Modest |
| Variance | Medium (shrinks with test size) | Medium (shrinks with test size) |
| Simplicity | Very simple | Simple |
| Valid for time series | No | Yes |

### When NOT to use a single hold-out

- Small data ($n < 1000$): the hold-out is too noisy.
- Imbalanced data: the test set might have too few minorities.
- Grouped data without explicit group-aware splitting.
- Any setting where you want a confidence interval on performance: use CV or bootstrap.

## 3.6 Common Pitfalls & Misconceptions

### Pitfall 1: Using the test set more than once

The cardinal sin. Every time you peek at test performance, you incorporate that information into your decision process and bias the next estimate. If you tune hyperparameters on the test set, the test set becomes a validation set, and you need a new test set.

### Pitfall 2: Random splitting time-series data

Random splits violate temporal causality. The fix is a chronological split. This is among the most common and highest-impact mistakes in applied ML.

### Pitfall 3: Random splitting grouped data

If the same entity (patient, user) appears multiple times, random splits place some of their records in train and others in test. The model learns entity-specific signal and the test score is inflated.

### Pitfall 4: Choosing the split to make the results look good

Some practitioners run many random splits and pick the one that gives the best test score. This is blatant cherry-picking. The mitigation: fix the split before any modeling, or use deterministic splits based on hashing IDs.

### Pitfall 5: A 50/50 split "to be safe"

50/50 halves your training data for no benefit. The variance of the hold-out estimate goes as $1/n_{\text{test}}$; there is little gain in test precision past a certain point, and training loss of 50% hurts more than the small gain in test precision helps.

### Pitfall 6: Ignoring class imbalance in the split

A random 80/20 of highly imbalanced data can produce a test set with zero minorities. Use stratified sampling for the split in this case.

### Pitfall 7: Assuming "train-test split" implies "unseen test data"

If you preprocess before splitting, or if the same records appear in train and test due to duplicates, your test set is contaminated. De-duplicate and split first, preprocess second (inside a pipeline).

### Pitfall 8: Interpreting a single hold-out score as precise

One number is one number. Hold-out gives no within-split variance estimate. If you need uncertainty, use CV, bootstrap, or compute the analytical SE (for simple metrics).

---
---

# Topic 4: Time-Series Cross-Validation (Rolling, Expanding, Blocked)

## 4.1 Motivation & Intuition

Imagine you are building a model to predict tomorrow's electricity demand based on weather, day-of-week, and recent demand history. You have 5 years of hourly data. If you use standard 5-fold CV, you randomly shuffle all 43,800 hours and split. A "training" hour might be in January 2024 and a "test" hour might be in March 2022. Your model is learning patterns from the future to predict the past. In deployment, this is impossible — you can only use data up to now.

This is not just an academic concern. Consider:

- **Autocorrelation**: today's demand is highly correlated with yesterday's. If yesterday is in the training set and today is in the test, the model effectively has access to a near-copy of the test feature.
- **Distribution shift**: energy consumption patterns change over years (more electric cars, efficiency improvements, climate change). A model trained on shuffled data sees a mix; a model trained on 2020-2024 must generalize to 2025.
- **Deterministic future leakage**: features engineered across the whole series (normalized against the global mean, or PCA over all hours) carry future information into the past. Even if you split randomly, the features themselves leak.

Time-series CV methods all share one fundamental rule: **the training set must consist entirely of data that occurs before the test set.** Three main variants exist:

1. **Expanding window** (anchored): training sets grow over time. Fold 1 trains on [0, T1), tests on [T1, T2). Fold 2 trains on [0, T2), tests on [T2, T3). And so on. Mimics production: you use all available past data.

2. **Rolling window** (sliding): training sets have a fixed size that slides forward. Fold 1 trains on [0, T1), tests on [T1, T2). Fold 2 trains on [T1 - W, T2), tests on [T2, T3), where W is the window width. Useful when older data is less relevant (regime changes).

3. **Blocked CV**: divide time into contiguous blocks, and within each block use some data for training and some for testing, with a **gap** between them to avoid leakage from serial correlation.

### Concrete example: stock price prediction

You have 10 years of daily S&P 500 data. You want to evaluate a model.

- **Standard 5-fold (wrong)**: Train on random 80% of days, test on other 20%. Test R² is 0.4. Deployed model performs at R² = -0.1. Why? The model learned trajectory patterns across the whole history, and "nearby" days (which are highly autocorrelated) were in both train and test. In deployment, it has to predict truly future days without this leakage.

- **Expanding-window CV (right)**: 
  - Fold 1: train on years 1-5, test on year 6. R² = 0.05.
  - Fold 2: train on years 1-6, test on year 7. R² = 0.02.
  - Fold 3: train on years 1-7, test on year 8. R² = -0.03.
  - Fold 4: train on years 1-8, test on year 9. R² = 0.01.
  - Fold 5: train on years 1-9, test on year 10. R² = 0.00.
  - Average: R² ≈ 0.01.

The honest estimate is barely above zero — reflecting reality for daily stock prediction (markets are approximately efficient). The naïve CV was grossly misleading.

## 4.2 Conceptual Foundations

### The autocorrelation problem

A time series $y_1, y_2, \ldots, y_T$ has **autocorrelation** $\rho(h) = \text{Corr}(y_t, y_{t+h})$. For most real series, $\rho(1), \rho(2), \ldots$ are significantly positive (nearby values are similar).

When training and test sets are interleaved, the model can exploit this. An extreme example: the "predict $y_t$" task becomes trivial if $y_{t-1}$ is known (and highly correlated with $y_t$) and appears as a feature or even just in the form of neighboring training examples that let the model interpolate.

### Non-stationarity

A time series is **stationary** if its joint distribution is invariant under time shifts. Real series are rarely stationary:
- **Trend**: mean changes over time.
- **Seasonality**: periodic variation.
- **Regime changes**: underlying dynamics shift at certain points.

Time-series CV evaluates the model under realistic non-stationarity: train on past, test on future under whatever the future looks like. This is the whole point.

### Expanding window

Also called "anchored CV" or "walk-forward validation". The training set starts from the beginning of the series and grows with each fold. The test window is a fixed chunk of "next" data.

**Pros**:
- Uses maximum available training data.
- Most closely mimics the deployment scenario where you train on everything up to now.
- Robust when the data-generating process is slowly drifting (not abruptly changing).

**Cons**:
- Later folds see much more training data than earlier folds → fold-level performances are not comparable (you are evaluating different-sized models).
- If there's a regime change in the middle of the series, the training set for later folds includes data from both regimes, which can confuse the model.

### Rolling window

Also called "sliding window". The training window has a fixed size $W$ that moves forward with each fold.

**Pros**:
- All folds train on the same amount of data → results are directly comparable.
- Naturally adapts to regime changes (old data is forgotten).
- Useful when you believe only the recent past is relevant for the near future.

**Cons**:
- Discards old data that might still contain useful signal.
- Requires choosing $W$, which adds a hyperparameter.
- May have high variance when $W$ is small.

### Blocked CV

Divide the series into blocks. Within each block, split into an earlier (train) part and a later (test) part, with a **gap** between them. The gap's purpose is to prevent residual autocorrelation from leaking training info into test. Folds are defined by which block is the "test block" for that fold.

A common variant: hv-block CV. Use $h$ blocks as train, $v$ blocks as validation, with a gap.

**When to use**: when you have multiple relatively-independent time chunks (e.g., data from multiple subjects, each with their own time course) or when you want balanced training sizes across folds.

### Purged and embargoed CV (Lopez de Prado)

For financial time series with overlapping labels (e.g., predicting 5-day forward return: the label for day $t$ depends on prices through day $t+5$), even a chronological split has leakage because the training label horizon can extend into the test period.

**Purging**: remove training examples whose label horizon overlaps the test period.
**Embargo**: after the test period, skip an embargo period before starting the next training period to avoid serial correlation leakage backwards in time.

These are niche but critical in quantitative finance.

### Assumptions

- **Temporal ordering is meaningful**: true by construction for time series.
- **Near-future distribution is similar to near-past**: otherwise no learning is possible at all. Time-series CV does not fix this; it just honestly measures the performance under this scenario.
- **Labels are contemporaneous with features**, or the label horizon is accounted for.
- **No future-looking features**: features at time $t$ should only use information up to time $t$.

### What breaks when assumptions are violated

- **Look-ahead bias in features**: if your feature at time $t$ uses a global mean or a quantile computed from the full dataset, that feature encodes future info. The CV is still invalid.
- **Abrupt regime change**: if the test period is in a new regime (pandemic, financial crisis, product launch), even time-series CV will underestimate error in the specific sense that the most recent training data does not represent the test period. The CV is "valid" in reflecting the actual deployment scenario but predictions are unreliable.
- **Overlapping labels not purged**: if labels span multiple time steps, chronological splits can still leak.

## 4.3 Mathematical Formulation

### Notation

Let $\{(x_t, y_t)\}_{t=1}^T$ be a time series. Define cutoff points $T_0 < T_1 < \ldots < T_K = T$.

### Expanding window CV

For fold $i = 1, \ldots, K$:
- Training indices: $\{1, 2, \ldots, T_{i-1}\}$
- Test indices: $\{T_{i-1} + 1, \ldots, T_i\}$

Estimator:
$$\hat{R}_{\text{expand}} = \frac{1}{K} \sum_{i=1}^K \frac{1}{T_i - T_{i-1}} \sum_{t=T_{i-1}+1}^{T_i} L(y_t, f_{-i}(x_t))$$

where $f_{-i} = \mathcal{A}(\{(x_s, y_s) : s \leq T_{i-1}\})$.

### Rolling window CV

Window width $W$. For fold $i = 1, \ldots, K$:
- Training indices: $\{T_{i-1} - W + 1, \ldots, T_{i-1}\}$ (truncated at 1)
- Test indices: $\{T_{i-1} + 1, \ldots, T_i\}$

Same estimator form, but $f_{-i}$ is trained on a window of fixed size $W$.

### Blocked CV with gap $g$

Divide time into $K$ blocks. For fold $i$:
- Test block: block $i$.
- Training blocks: blocks $j \ne i$, but exclude any training points within $g$ time steps of a test point.

Estimator is the standard fold-average form.

### Why gaps matter mathematically

Suppose $y_t = \phi y_{t-1} + \epsilon_t$ (AR(1) with parameter $\phi$). Then $\text{Corr}(y_t, y_{t+h}) = \phi^h$. If you split immediately (no gap) at time $t^*$, the last training point is $y_{t^*}$ and the first test point is $y_{t^* + 1}$, with correlation $\phi$.

The model, having fit to $y_1, \ldots, y_{t^*}$, knows $y_{t^*}$ perfectly and can predict $y_{t^* + 1}$ with accuracy close to $\phi \cdot y_{t^*}$ — trivially easy. To break this, you need a gap of size at least several autocorrelation time-scales.

### Choosing the gap size

Fit an AR or VAR model to estimate the autocorrelation structure. Pick a gap $g$ such that $|\rho(g)|$ is sufficiently small (say, < 0.05). For many financial series, this is on the order of 1-5 days; for slowly-varying series, much longer.

## 4.4 Worked Examples

### Example A: Expanding window CV on daily sales

Data: 5 years of daily sales for a retailer, $T = 1825$ days.

CV setup: 5 folds, each testing on 1 year.

| Fold | Train range          | Test range           | Train size | Test size |
|------|----------------------|----------------------|------------|-----------|
| 1    | days 1–365           | days 366–730         | 365        | 365       |
| 2    | days 1–730           | days 731–1095        | 730        | 365       |
| 3    | days 1–1095          | days 1096–1460       | 1095       | 365       |
| 4    | days 1–1460          | days 1461–1825       | 1460       | 365       |

4 folds (you cannot predict the first year since there is no training data before it).

Suppose MAPE per fold: 8.2%, 7.9%, 7.5%, 7.3%.

- Trend: MAPE improves as training set grows (more data → better model). 
- Report average: 7.73%.
- Note: if the trend persists, a single very long time series would give better performance than the average. The CV is slightly pessimistic for the final deployed model.

### Example B: Rolling window CV on the same data

Window size: 730 days (2 years). Test window: 90 days.

| Fold | Train range         | Test range          |
|------|---------------------|---------------------|
| 1    | days 1–730          | days 731–820        |
| 2    | days 91–820         | days 821–910        |
| 3    | days 181–910        | days 911–1000       |
| ...  | ...                 | ...                 |
| 13   | days 1081–1810      | days 1811–1825 (partial) |

~12 folds, each training on 730 days, testing on 90 days.

If the retailer opened a new flagship store at day 1500, rolling window CV will show deteriorating performance for folds spanning the transition (old-window models miss the regime change) and improved performance once the training window moves fully past day 1500.

Expanding window, by contrast, carries old data forever — the new-store signal is "diluted" in the long history, and the model may under-weight it.

### Example C: Blocked CV with gap

Data: continuous physiological recording from 20 subjects, 1 hour each. You want to detect arrhythmias from ECG.

Approach:
- Within each subject, split into 10-minute blocks.
- Subject-wise blocked CV: each fold tests on one subject (this is actually group CV).
- Within-subject blocked CV (for a different purpose, e.g., adaptation): use 4 blocks for training, 1 block for test, 1 block gap.

With 1-second autocorrelation being significant, the gap of (say) 30 seconds ensures training and test samples are not trivially linked.

### Example D: Illustrating gap importance

Synthetic AR(1) with $\phi = 0.9$, $T = 1000$. Predict $y_t$ from $y_{t-1}, y_{t-2}, \ldots, y_{t-5}$.

- **No gap, chronological split 800/200**: test MSE = 0.21.
- **Gap of 5**: test MSE = 0.41.
- **Gap of 20**: test MSE = 0.44.
- **True MSE of the optimal AR model**: 0.45 (asymptotically).

The no-gap result is overly optimistic because the training set ends with points highly correlated with the start of the test set. With a gap of 20 or more, the estimate converges to the true MSE.

### Example E: Purged CV in finance

Task: predict 5-day-forward return. Feature at time $t$: momentum indicator. Label at time $t$: $r_{t,t+5} = (P_{t+5} - P_t)/P_t$.

If you split at time $t^*$ with training in $[1, t^*]$, a training example at time $t^* - 3$ has label $r_{t^*-3, t^*+2}$ — which spans into the test region. The label directly encodes future test-period information.

**Purged CV**: remove training examples whose label end-point exceeds $t^*$. Also, embargo for 5 days after $t^*$ before the test starts to guard against the reverse leak.

Result: honest performance estimate for a 5-day-return model.

## 4.5 Relevance to ML Practice

### Where time-series CV is standard

- **Demand forecasting** (retail, electricity, ride-hailing).
- **Financial modeling** (equity returns, credit risk with temporal features).
- **Recommender systems** with temporal drift.
- **IoT / sensor data** with trends.
- **A/B testing analysis with temporal effects**.
- **Adtech bidding models**, which must handle rapidly changing marketplaces.

### When expanding vs rolling

- **Expanding** when: you believe the full history is relevant, the process is slowly changing, and more data clearly helps.
- **Rolling** when: regime changes occur, old data is misleading, you want comparable folds, or you want to stress-test under concept drift.

### Trade-offs

| Choice | Pros | Cons |
|---|---|---|
| Expanding window | Uses all data; mimics production | Non-comparable folds; carries obsolete data |
| Rolling window | Comparable folds; handles drift | Discards data; adds window-size hyperparameter |
| Blocked with gap | Simple; respects autocorrelation | Less data efficient; gap is a hyperparameter |
| Purged CV | Handles overlapping labels | Complex; requires knowing label horizon |

### Reporting time-series CV results

- Show all fold-level scores (do not just average), because fold-to-fold differences can indicate concept drift.
- Consider breaking out performance by time period (monthly/quarterly).
- If seasonality matters, make sure the test windows span complete seasons (e.g., test on full years, not partial).

### Alternatives

- **Single time-based hold-out** for large data.
- **Prequential evaluation** (streaming): evaluate each prediction before including the truth into training.
- **Backtesting simulations** for trading strategies, which are closely related to rolling-window CV but include transaction costs and slippage.

## 4.6 Common Pitfalls & Misconceptions

### Pitfall 1: Using random CV on time-series

The #1 error in this domain. Symptoms: optimistic offline metrics, disappointing live performance.

### Pitfall 2: Look-ahead bias in features

Even with correct temporal splits, features that use the entire dataset (global z-scoring, global PCA, features computed from the full history) leak future info. Features must be computed using only data available at the prediction time.

### Pitfall 3: Forgetting to respect label horizons

If the label at time $t$ depends on data through $t + h$, you need purging and embargoing.

### Pitfall 4: Evaluating on a test period that is not representative

If you test on a single quarter and that quarter had an unusual event (holiday season, pandemic), your estimate reflects that event, not general performance. Use multi-fold CV across many test periods.

### Pitfall 5: Shuffling inside the training fold, then splitting further

Sometimes people do time-series CV at the outer level, then random CV on each training fold for hyperparameter tuning. This still leaks within the training set (e.g., your hyperparameter tuning sees future info). Use nested time-series CV.

### Pitfall 6: Assuming model trained on the full history is optimal

If there are regime changes, training on the full history might be worse than training on just the recent window. Evaluate rolling vs. expanding and pick what performs best.

### Pitfall 7: Ignoring seasonality

For daily data with weekly and yearly cycles, test windows shorter than one week or one year can give biased estimates (e.g., testing only on weekdays). Use test windows that are multiples of the seasonal period.

### Pitfall 8: Data that is time-indexed but not actually time-dependent

Some datasets have timestamps but the modeling task does not depend on them (e.g., image classification with capture dates). Time-series CV is overkill; use standard CV.

---
---

# Topic 5: Grouped Cross-Validation (Leave-One-Group-Out, Group k-Fold)

## 5.1 Motivation & Intuition

Your data is 10,000 chest X-ray images labeled for pneumonia. You train a deep learning model with 5-fold random CV and get 95% AUC. You deploy the model at a new hospital and see 78% AUC. What went wrong?

**The issue**: each patient in your dataset typically has 3–10 X-rays taken over different visits. When you randomly split by image, some of patient A's images go into training and some into testing. The model learns patient-specific features (positioning quirks, chest anatomy, incidental markers) and uses them to "predict" test images of the same patient — which is easy.

In deployment, the model encounters images of entirely new patients it has never seen. The patient-specific shortcut is gone, and performance collapses.

**The fix**: split by patient, not by image. Each patient's images go entirely into training OR entirely into testing, never split. Then CV measures what it is supposed to measure: generalization to new patients.

This is the core idea of **grouped cross-validation**: when your data has hierarchical structure (examples nested within groups), you must split at the group level to get honest generalization estimates.

### Where this matters: real-world examples

- **Medical imaging**: patients → images (or patients → slices → pixels for 3D scans).
- **Speech recognition**: speakers → utterances.
- **NLP**: documents → sentences (if you want document-level generalization).
- **Recommender systems**: users → interactions (if predicting user preferences).
- **Educational**: schools → students → questions.
- **Genomics**: patients → cells → reads.
- **Federated learning**: each client's data is a group.
- **Computer vision**: scenes → images (same scene from different angles).

The common thread: each group shares properties that the model can exploit as a shortcut. These properties are not present for new groups at test time.

## 5.2 Conceptual Foundations

### Core definitions

- **Group**: a set of examples that share some hidden correlated property (e.g., all records from the same patient). Formally, each example has a group ID $g_i \in \{1, \ldots, G\}$.
- **Grouped CV**: any CV scheme that ensures no group has examples split between train and test in the same fold.

### Leave-One-Group-Out (LOGO)

For $j = 1, 2, \ldots, G$:
- Training set: all examples from groups $\ne j$.
- Test set: all examples from group $j$.

Produces $G$ folds. Useful when $G$ is small and manageable (say $G \leq 50$), or when you specifically want to evaluate generalization to each individual new group.

### Group k-Fold

For $G > k$, LOGO is infeasibly expensive. Group k-fold partitions the **groups** into $k$ subsets of groups. Each fold uses one subset as test groups.

The guarantee: any example in a test fold belongs to a group whose other examples are also in that test fold (not in training).

Group sizes may vary, so fold sizes (in terms of number of examples) are approximate; this is a feature of group CV, not a bug.

### When group structure is subtle

Sometimes the group structure is not obvious:

- **Entity resolution**: two records might "look different" but be duplicates of the same underlying entity. If you split them across train/test, you leak.
- **Temporal groups**: events happening in the same window might cluster (e.g., all fraud attempts from the same campaign). Even if you have per-event records, the campaign is the real group.
- **Feature-based groups**: sometimes a feature reveals a group (e.g., ZIP code identifies a household). If you don't group by household, your split may be random within households.

Finding group structure is a domain knowledge problem, not an algorithmic one.

### Stratified group CV

When the groups themselves have imbalanced labels (some groups are all positive, some all negative), you want stratification at the group level. E.g., in medical imaging, some patients have many positive scans and some have none. A good split balances the label distribution across folds while keeping groups intact.

Scikit-learn provides `StratifiedGroupKFold` for this.

### Assumptions

- **Group IDs are correctly assigned**: data engineering must get this right.
- **Deployment involves new groups**: the point of grouped CV is to simulate new-group generalization. If deployment only involves existing groups (e.g., a per-user model where we always fine-tune for each user), non-grouped CV might be more appropriate.
- **Number of groups is sufficient**: with only 5 groups, LOGO gives 5 folds with high variance. Group CV needs many groups.

### What breaks

- **Too few groups**: group CV with $G = 10$ is like 10-fold, but fold quality depends on the group variance. High-variance folds.
- **Imbalanced groups**: one group has 10,000 examples, others have 10. Single-group-test folds are very noisy when the group is small, and very expensive when large.
- **Overlapping groups**: one example belongs to multiple groups (e.g., a child is in both "patients with condition X" and "patients with condition Y"). Hard edge case; usually requires stricter splits.

## 5.3 Mathematical Formulation

### Setup

Let $\{(x_i, y_i, g_i)\}_{i=1}^n$ where $g_i$ is the group index. Let $G_j = \{i : g_i = j\}$ be the indices in group $j$, with $|G_j| = n_j$.

### LOGO estimator

For $j = 1, \ldots, G$:
$$e_j = \frac{1}{n_j} \sum_{i \in G_j} L(y_i, f_{-G_j}(x_i))$$
where $f_{-G_j}$ is trained on $\{(x_i, y_i) : g_i \ne j\}$.

$$\hat{R}_{\text{LOGO}} = \frac{1}{G} \sum_{j=1}^G e_j$$

Alternatively, an example-weighted version:
$$\hat{R}_{\text{LOGO}}^{\text{wt}} = \frac{1}{n} \sum_{j=1}^G \sum_{i \in G_j} L(y_i, f_{-G_j}(x_i))$$

These differ when groups are of different sizes. The group-average form gives equal weight to each group (fair for evaluating per-group behavior). The example-weighted form treats each example equally (matches standard loss averaging).

### Group k-fold estimator

Partition $\{1, \ldots, G\}$ into $k$ subsets $S_1, \ldots, S_k$. For fold $\ell$:
- Test examples: $\bigcup_{j \in S_\ell} G_j$.
- Training: the complement.

Fold loss: average loss over test examples. Estimator: average over folds (or example-weighted).

### Variance considerations

A key distinction: in standard k-fold CV, the variance of the estimator depends on the loss variance per example and correlations between folds. In grouped CV, an additional source of variance enters: **between-group variance**. If groups differ a lot in difficulty, fold-level errors will vary based on which groups happen to be in the test fold.

If we write $\mathbb{E}[L | g_i = j] = \mu_j$ as the expected loss conditional on group $j$, then

$$\text{Var}[e_j] = \text{Var}[\mu_j] + \mathbb{E}[\text{Var}[L | g_i = j]]/n_j$$

With a small number of groups, the between-group variance $\text{Var}[\mu_j]$ dominates. Rule of thumb: you need at least 20-50 groups for a reliable grouped CV estimate, and many more if group-level variance is high.

### Group CV as a bound on iid CV

Key theoretical fact: grouped CV generally gives **worse** performance estimates than random CV (higher error). This is the correct behavior — random CV overestimates by leveraging within-group correlations. Grouped CV reveals the truth.

## 5.4 Worked Examples

### Example A: LOGO on a medical dataset

Dataset: 50 patients, each with 20 chest X-rays. Total $n = 1000$, $G = 50$. Binary label: has pneumonia or not.

LOGO:
- 50 folds.
- Each fold: train on 49 patients (980 images), test on 1 patient (20 images).

Result: fold-level accuracies from 0.45 to 0.95. Average: 0.78.

Contrast with random 5-fold:
- Each fold: train on 800 images (from ~50 patients), test on 200 images (from ~50 patients, ~4 images per patient in test).
- Average accuracy: 0.95.

The 17-point gap is the "leakage premium": random 5-fold captures within-patient memorization that does not generalize.

The LOGO fold variance is also informative: accuracy of 0.45 on one patient suggests the model fails systematically on some patient phenotypes. Worth investigating.

### Example B: Group 5-fold on a large dataset

Dataset: 100,000 customer service call recordings from 5,000 agents. Goal: predict call disposition.

LOGO would be 5,000 folds — too expensive. Use group 5-fold:
- Partition 5,000 agents into 5 groups of 1,000.
- Each fold: test on 1 group of agents (call volumes ~20,000).
- Balances computational efficiency with group awareness.

### Example C: When groups leak via non-obvious features

Dataset: house price prediction, one row per sale. Some houses have been sold multiple times.

Naïve: random 5-fold over sales. But multiple sales of the same house share features (neighborhood, zip, square footage — even non-sale-specific features like the school district).

Group by address: group 5-fold with address as the group ID. This ensures the model is evaluated on unseen houses.

But some neighborhoods are over-represented. Possibly group by neighborhood instead (even more conservative). The "right" answer depends on deployment: if the model will be used on truly new houses in existing neighborhoods, group-by-address is appropriate. If it will be used on new neighborhoods, group-by-neighborhood.

### Example D: Stratified group CV

Dataset: 200 patients, 1000 CT scans. 50 patients have cancer, 150 do not. Per-scan label: malignant or benign. But cancer patients only have ~30% of scans labeled malignant; non-cancer patients have 0 malignant scans.

Simple group 5-fold could produce a fold where all 10 cancer patients (of 50) happen to be in training, leaving zero malignant scans in the test fold.

Stratified group 5-fold: partition cancer patients and non-cancer patients into groups separately, then combine. Ensures each fold has ~10 cancer patients and ~30 non-cancer patients, so all folds have a reasonable malignant rate.

### Example E: Number-of-groups effect on variance

Two experiments, same data, different grouping:

**Experiment 1**: 10 groups.
- Group 5-fold: each test fold has 2 groups. Fold accuracies: 0.72, 0.81, 0.65, 0.88, 0.75. SD = 0.087. Average = 0.762.

**Experiment 2**: 100 sub-groups (same data, finer grouping).
- Group 5-fold: each test fold has 20 sub-groups. Fold accuracies: 0.76, 0.78, 0.74, 0.77, 0.77. SD = 0.015. Average = 0.764.

Same underlying truth, but far tighter estimate with more groups (because fold averages are over more group draws).

## 5.5 Relevance to ML Practice

### Where grouped CV is standard

- **Medical ML**: essentially always.
- **Speech and audio**: group by speaker.
- **NLP with user-level tasks**: group by user.
- **Large document processing**: group by document when evaluating document-level tasks.
- **Any dataset with repeated measurements on the same unit**.

### Deployment-driven decision

The critical question: "Who/what will be new at deployment?" If new patients/users/entities will appear, use group CV. If the model will only ever see existing entities, standard CV might be fine (though you should think carefully — even then, there's usually some concept drift at the entity level).

### Trade-offs

| Aspect | Grouped CV | Random CV |
|---|---|---|
| Honesty for new groups | ✓ | ✗ |
| Fold size | Varies | Equal |
| Variance | Higher (fewer effective "units") | Lower |
| Computational cost | Same as k-fold | Baseline |
| Requires group IDs | Yes | No |

### Alternatives and complements

- **Domain-specific CV schemes**: for some domains, the group structure is complex enough that custom splits are needed.
- **Meta-learning CV**: if you want to evaluate "how well does this model fine-tune to a new group", use a more specialized protocol.
- **Multi-task CV**: when groups have different tasks.

## 5.6 Common Pitfalls & Misconceptions

### Pitfall 1: Using random CV when group structure exists

The main pitfall. The red flag: you know your data has repeated measurements (users, patients, devices) and you are using scikit-learn's default `KFold`. Stop and switch to `GroupKFold`.

### Pitfall 2: Using too few groups

Group CV with 5 groups has high variance. If you have fewer than ~20 groups, consider LOGO with multiple repetitions or a more conservative single hold-out.

### Pitfall 3: Grouping at the wrong level

If you group by "patient visit" when the relevant unit is "patient", you still leak (same patient's other visits are in training). Always group at the highest level that corresponds to "new entity at deployment".

### Pitfall 4: Ignoring group imbalance

Groups of size 1,000 and groups of size 10 produce wildly different fold compositions. For algorithms sensitive to sample weighting, this can matter. Consider weighting or sub-sampling.

### Pitfall 5: Thinking group CV solves all leakage

Group CV solves within-group leakage. It does not solve temporal leakage (use time-series CV too), spatial leakage (use spatial CV), or feature-engineering leakage (fit features inside each fold).

### Pitfall 6: Hyperparameter tuning with the same groups as evaluation

If you use group CV for hyperparameter tuning and then report the best performance, you have selection bias. Use nested group CV: outer folds for performance, inner folds for tuning, groups preserved throughout.

### Pitfall 7: Misapplying LOGO when G is huge

Leaving out one user at a time when you have 1,000,000 users is not "group CV" in any useful sense — it's essentially LOOCV at the group level, which has the same variance and cost issues. Use group k-fold.

### Pitfall 8: Not reporting per-group (or per-fold) performance

Grouped CV gives you a natural way to see performance per group. Exploit it: worst-case group is often as important as the average.

---
---

# Topic 6: Spatial Cross-Validation (Spatial Blocking, Location-Based Splits)

## 6.1 Motivation & Intuition

Suppose you are building a model to predict soil contamination levels from remotely sensed features (satellite imagery, elevation, land use). You have measurements at 5,000 locations across a region. You use standard 5-fold CV and get R² = 0.85 — excellent. You deploy in an adjacent region and R² = 0.3. What happened?

**The issue**: your data has **spatial autocorrelation**. Locations that are close together tend to have similar contamination levels (same soil type, same land use history, same drainage patterns). In a random 5-fold split, a test point almost always has a training point nearby. The model effectively does spatial interpolation — easy when nearby training is available, hard when extrapolating to a new region.

This is the spatial analog of temporal autocorrelation. It is ubiquitous in:
- **Environmental science**: soil, air quality, biodiversity.
- **Geology**: mineral exploration, seismic hazard.
- **Epidemiology**: disease mapping, vector distributions.
- **Precision agriculture**: crop yield, nutrient levels.
- **Urban planning and real estate**: property values.
- **Wildlife ecology**: species distribution modeling.

The fix: **spatial cross-validation**. Split the data so that test points are geographically separated from training points, forcing the model to extrapolate rather than interpolate.

### The "true generalization" question

Spatial CV answers a specific, deployment-relevant question: "How well will this model perform on geographically new areas?" 

Standard random CV answers a different question: "How well will this model fill in missing values among sampled locations?" If your deployment is interpolation within sampled areas, standard CV is fine. If it is extrapolation to new areas, spatial CV is mandatory.

### Concrete example: species distribution modeling

Ornithologists train a model to predict the presence of a rare bird based on elevation, temperature, vegetation density, etc. Data: 500 observation points in the Pacific Northwest.

- **Standard 5-fold**: AUC = 0.92.
- **Spatial 5-fold with 50 km buffer**: AUC = 0.68.

The first estimate is valid for predicting presence at new points within the Pacific Northwest network of observed locations. The second is valid for predicting presence in new areas. If the goal is to extend predictions to the Rocky Mountains, the second is closer to reality (and even that is optimistic, because it does not account for range shifts).

## 6.2 Conceptual Foundations

### Spatial autocorrelation

Tobler's First Law of Geography: "Everything is related to everything else, but near things are more related than distant things." Quantified:

- Positive autocorrelation: nearby values are similar (the usual case).
- Negative autocorrelation: nearby values differ systematically (rare).
- No autocorrelation: spatially random.

Measured by:
- **Moran's I**: correlation-like statistic.
- **Variogram**: $\gamma(h) = \frac{1}{2}\text{Var}(Z(s) - Z(s+h))$. Shows how dissimilarity grows with distance.

The **range** of the variogram is the distance beyond which autocorrelation is negligible. Spatial CV splits must respect this range.

### Spatial blocking

Divide the study area into blocks (squares, hexagons, or environmental zones). Assign each block entirely to train or test.

**Procedure**:
1. Overlay a grid of size $L \times L$ on the map.
2. Each point belongs to exactly one grid cell.
3. Partition grid cells into $k$ folds (random, or stratified by environmental variables).
4. For each fold, use all points in its cells as the test set; all other points for training.

**Variants**:
- **Regular grid**: simple squares.
- **Spatially balanced blocks**: ensure each fold has coverage across the region.
- **Environmental blocks**: group blocks with similar environmental profiles.

### Buffer zones

Even block CV has leakage at the boundaries: points near the edge of a test block have training points just across the border. A **buffer zone** removes training points within a certain distance of the test block.

- Buffer size = spatial range of autocorrelation.
- Downside: wastes data near every boundary.

### Leave-One-Location-Out (LOLO)

For each observation, hold it out and train on the rest (analogous to LOOCV). Sometimes combined with a buffer: hold out a location and exclude all training points within $r$ of it. This is called "spatial LOOCV" or "buffered LOOCV".

Useful for small datasets; expensive for large ones.

### Environmental CV / domain CV

Rather than splitting geographically, split based on environmental characteristics (e.g., by biome, by soil type, by climate zone). Tests extrapolation to environmentally new areas. Stricter than spatial blocking because nearby points can belong to different environments.

### Assumptions

- **Spatial autocorrelation is the main source of structural correlation** (not, say, a hidden temporal structure).
- **Deployment involves extrapolating to new areas**: otherwise, spatial CV is too conservative.
- **Block / buffer sizes are larger than the autocorrelation range**: otherwise leakage persists.

### What breaks

- **Too small blocks**: still correlated across folds.
- **Too large blocks**: too few effective folds, low-variance estimator becomes high-variance.
- **Highly clustered sampling**: if your training data is all in one region, spatial CV with blocks in other regions has very small test sets (or empty ones).
- **Anisotropic autocorrelation**: if correlations depend on direction (common for environmental phenomena with prevailing winds or drainage), square blocks may not cover all leakage directions. Need orientation-aware buffers.

## 6.3 Mathematical Formulation

### Setup

$\{(x_i, y_i, s_i)\}_{i=1}^n$ where $s_i \in \mathbb{R}^2$ is the spatial location (latitude/longitude or projected coordinates).

### Variogram

$$\gamma(h) = \frac{1}{2|N(h)|} \sum_{(i,j) \in N(h)} (z_i - z_j)^2$$

where $N(h)$ is pairs of points with distance $\approx h$. The **range** $a$ is where $\gamma$ plateaus at the sill (total variance). Beyond $a$, points are approximately uncorrelated.

### Spatial block CV with blocks of size $L$

Let $B(s) = (\lfloor s_x / L \rfloor, \lfloor s_y / L \rfloor)$ be the block index of location $s$. Partition block IDs into $k$ folds.

For fold $\ell$:
- Test: $\{i : B(s_i) \in \text{fold}_\ell\}$.
- Train: the rest.

### With buffer $r$

For fold $\ell$:
- Test: $\{i : B(s_i) \in \text{fold}_\ell\}$.
- Train: $\{i : B(s_i) \notin \text{fold}_\ell \text{ and } d(s_i, \text{nearest test point}) > r\}$.

Buffer $r$ should be at least the variogram range $a$.

### Effective sample size

Due to spatial autocorrelation, your $n$ points do not act as $n$ independent observations. The **effective sample size** (ESS) is approximately

$$n_{\text{eff}} \approx \frac{n}{1 + \rho \cdot (\bar{k} - 1)}$$

where $\rho$ is average spatial correlation within a neighborhood of average size $\bar{k}$. For highly autocorrelated data, $n_{\text{eff}} \ll n$. Spatial CV with proper blocking approximates what you can legitimately infer from the data.

## 6.4 Worked Examples

### Example A: Predicting soil organic carbon

Data: 2,000 soil samples across a 500 × 500 km region. Task: predict soil organic carbon from remotely sensed features.

**Variogram analysis**: range = 40 km. Autocorrelation negligible past 40 km.

**Standard 5-fold CV**: RMSE = 0.48 (normalized). Model looks great.

**Spatial 5-fold CV, blocks of 100 × 100 km (25 blocks, 5 per fold)**: RMSE = 0.71. More honest.

**Spatial 5-fold CV with 40 km buffer**: RMSE = 0.79. More honest still.

**Insight**: the model heavily exploits spatial proximity. When forced to extrapolate, it is substantially worse. The true out-of-region deployment would likely get 0.75-0.85 RMSE.

### Example B: Species distribution modeling with Maxent

Data: 300 presence-only records of a bird species across Europe. 10,000 background points.

**Block selection**: 200 × 200 km grid. 100 blocks, 5 folds.

**Procedure**:
1. Randomly assign blocks to folds.
2. For each fold, train Maxent on points in other folds' blocks.
3. Evaluate on test fold.
4. Check that each test fold has a reasonable number of presence points (otherwise the AUC estimate is noisy).

**Result**: AUC varies from 0.71 to 0.84 across folds, depending on which part of Europe is the test region. Average 0.78.

**Interpretation**: model generalizes moderately well across regions, but some regions (notably the one with high fold-specific AUC of 0.84) are well-represented in training. The average 0.78 is the best estimate of performance on novel regions.

### Example C: Buffer size sensitivity

Same soil carbon dataset. Vary buffer size $r$:

| Buffer $r$ (km) | RMSE |
|---|---|
| 0 (no buffer) | 0.71 |
| 10 | 0.74 |
| 20 | 0.76 |
| 40 (≈ range) | 0.79 |
| 80 | 0.80 |
| 160 | 0.80 |

RMSE plateaus around $r = a$ (the variogram range). Increasing the buffer past that wastes data without improving honesty.

### Example D: Environmental vs. spatial CV

Soil sample dataset again. Now cluster the samples by environmental features (climate, land use, soil type) into 10 clusters using k-means on environmental variables.

**Environmental 5-fold**: each fold is a disjoint set of environmental clusters (2 per fold).
RMSE: 0.85.

This is higher than geographic block CV because environmentally similar points can be far apart (a temperate forest patch in Portugal and one in Germany might be in the same cluster), and environmentally different points can be nearby (a forest patch adjacent to farmland). Environmental CV tests generalization to new environments, which is stricter.

### Example E: Why LOLO with buffer catches residual leakage

$n = 100$ points on a grid. Spatial LOOCV without buffer:
- For each test point, the nearest training point is usually adjacent → very easy prediction.
- "CV" error is essentially just the nugget effect of the variogram.

Spatial LOOCV with buffer $r = $ grid spacing:
- For each test point, exclude the 8 neighbors from training.
- Now the model must actually extrapolate.
- CV error reflects this.

## 6.5 Relevance to ML Practice

### Where spatial CV is standard

- **Environmental and ecological modeling**.
- **Remote sensing** (when extrapolating to new regions).
- **Geochemistry and mining**.
- **Epidemiology** with spatial components.
- **Urban analytics** with cross-city transfer.

### Trade-offs

| Method | When to use |
|---|---|
| Random CV | Predictive interpolation within sampled region |
| Spatial block CV | Generalizing to new parts of the same region |
| Spatial block + buffer | Stricter; use when autocorrelation range is known |
| Environmental CV | Generalizing to new environments |
| LOLO | Small samples; fine-grained spatial evaluation |

### Common pitfalls specific to spatial ML

Spatial data often arrives with coordinate systems, projections, and distance metrics that must be handled consistently. Block sizes in kilometers vs. degrees can differ by 10x at different latitudes. Always project to a planar coordinate system before blocking.

### Alternatives

- **Transfer learning evaluation protocols**: explicitly split train and test regions.
- **Conformal prediction with spatial calibration**: uncertainty-aware prediction that accounts for spatial structure.
- **Spatial Gaussian processes**: model the autocorrelation explicitly.

## 6.6 Common Pitfalls & Misconceptions

### Pitfall 1: Using random CV for spatial data

The main issue. Symptom: huge gap between CV performance and performance in new regions.

### Pitfall 2: Using spatial CV when unnecessary

If the deployment is truly "fill in missing values in the sampled region", spatial CV is overly pessimistic. Evaluate what you want to measure.

### Pitfall 3: Blocks too small

Blocks smaller than the autocorrelation range leak. Compute or estimate the range first.

### Pitfall 4: Ignoring edge effects

Without buffering, the points at block boundaries have their "nearest neighbor" just across the boundary, still in the training set.

### Pitfall 5: Imbalanced block composition

A block might have zero test points (if no data was collected there). Random block assignment can produce folds with highly variable sizes. Stratify or rebalance.

### Pitfall 6: Confusing geographic and environmental proximity

Two points can be spatially close but environmentally different (e.g., a beach and an adjacent rainforest). Two points can be environmentally similar but spatially distant. Choose the CV axis that matches your deployment question.

### Pitfall 7: Ignoring temporal dynamics in spatiotemporal data

Some datasets have both spatial and temporal structure (e.g., monthly satellite observations over time). CV must respect both axes. Usually: split temporally at the outer level, spatially at the inner level.

### Pitfall 8: Treating spatial CV as a panacea

Spatial CV reveals spatial generalization but does not solve covariate shift, label shift, or changes in underlying processes. It is one tool among many.

---
---

# Topic 7: Nested Cross-Validation

## 7.1 Motivation & Intuition

You have a dataset and a learning algorithm with hyperparameters (say, gradient boosting with tree depth, learning rate, and regularization to choose). You want to (a) pick the best hyperparameters and (b) estimate how well the resulting model will generalize.

Natural first idea: use CV for both.

```
for each hyperparameter config:
    compute 5-fold CV score
pick config with best score
report that score as the performance
```

This is wrong. The reported number is **biased optimistically**. Here is why: you tested many configurations and selected the best. The best score is inflated by "winner's curse" — even if all configurations had identical true performance, the max of noisy estimates exceeds their true mean.

A concrete example: run 20 configurations, all truly 0.80 accuracy. CV gives noisy estimates; one happens to come in at 0.84 and you pick it. The 0.84 is not the true performance of that config — it is the config's true performance plus a favorable noise draw. If you reported 0.84, you would be too optimistic.

**Nested CV solves this by using two levels of CV:**

- **Inner loop**: hyperparameter tuning. For each outer training fold, find the best hyperparameters using an inner CV on that training fold.
- **Outer loop**: performance estimation. Each outer test fold is evaluated with a model whose hyperparameters were selected only using the outer training fold.

The outer test fold was never seen during hyperparameter selection. Its performance is an unbiased estimate of the "tune-then-fit" procedure's performance.

### Concrete example

- **Single-loop CV (wrong)**: 5-fold CV, 20 configurations. Best CV score = 0.86. Reported performance: 0.86.
- **Nested CV (right)**: 
  - Outer: 5 folds. 
  - For each outer fold: inner 5-fold CV over 20 configs, pick best hyperparameters, refit on outer training fold, evaluate on outer test fold.
  - Across outer folds, outer test scores: 0.82, 0.84, 0.81, 0.83, 0.82. Average = 0.824.
- **True deployment performance**: ≈ 0.82.

The single-loop overestimated by 0.04; nested is accurate.

### When the inflation is large

Bias from single-loop CV grows with:
- Number of hyperparameter configurations tried.
- Per-configuration CV noise (small datasets, high variance models).
- Extreme configurations that can overfit to CV noise.

On small datasets with extensive hyperparameter search, the inflation can exceed 5 percentage points.

## 7.2 Conceptual Foundations

### The two loops

**Outer loop** (performance estimation):
- $K$ outer folds.
- For each outer fold, hold it out as the outer test set. The rest is the outer training set.
- At the end, average the outer test scores to get the performance estimate.

**Inner loop** (model selection / hyperparameter tuning):
- For each outer training set, run an inner CV.
- The inner CV compares hyperparameter configurations.
- The best configuration is then refit on the full outer training set (no inner hold-out).
- That refit model is evaluated on the outer test set.

### What nested CV estimates

The performance estimate from nested CV is the performance of the **entire procedure** — the procedure being "train on $n(K-1)/K$ examples, do hyperparameter tuning via inner CV, fit final model, predict."

It is **not** the performance of any single specific hyperparameter configuration.

This is an important subtlety. Different outer folds may select different hyperparameters (indeed, they often do). Nested CV does not tell you which hyperparameters to use at deployment. It tells you how good the "tuning + fitting" pipeline is, on average.

**For deployment**: once you trust the pipeline (based on nested CV), retrain the full pipeline (tuning + fitting) on all the data. The final model's hyperparameters may not match any of the ones chosen in the outer folds, and that is fine.

### Parameter stability

One diagnostic: check how stable the chosen hyperparameters are across outer folds.
- If all folds pick the same config, you have high confidence in that config.
- If each fold picks wildly different configs, your search space may be too close to degenerate (many configs perform similarly within CV noise). Consider narrowing.

### Assumptions

- All standard CV assumptions apply to both loops.
- **No leakage between loops**: the inner CV must only see data from the outer training set.
- **Stable model selection**: if the hyperparameter landscape is very flat, small data changes (different outer folds) lead to different selections, inflating variance.

### What breaks

- **Doing hyperparameter tuning once and evaluating with a separate CV**: this is better than single-loop, but inner-CV training samples overlap with outer-test samples (depending on how it is structured) unless you are careful.
- **Reusing the same folds inside and outside**: if inner CV uses the same fold structure, the inner "validation" set partially overlaps with the outer training set in ways that can leak.
- **Touching the outer test set for any preprocessing**: the outer test set must be truly untouched until final evaluation.

## 7.3 Mathematical Formulation

### Setup

Learning algorithm family $\{\mathcal{A}_\theta : \theta \in \Theta\}$ indexed by hyperparameters $\theta$. Dataset $D$.

### Single-loop CV with model selection (biased)

For each $\theta \in \Theta$, compute $\hat{R}_{CV}(\theta; D)$. Select $\theta^* = \arg\min_\theta \hat{R}_{CV}(\theta; D)$. Report $\hat{R}_{CV}(\theta^*; D)$.

The bias:
$$\mathbb{E}[\hat{R}_{CV}(\theta^*; D)] \leq \min_\theta \mathbb{E}[\hat{R}_{CV}(\theta; D)] - \text{selection bias}$$

The more $\theta$'s you compare, the larger the selection bias (related to the maximum of correlated random variables).

### Nested CV

Outer folds: partition $D$ into $D_1, \ldots, D_K$.

For each outer fold $i$:
1. Inner fold structure: partition $D \setminus D_i$ into $K'$ inner folds.
2. Compute $\hat{R}_{\text{inner}}(\theta; D \setminus D_i)$ for each $\theta$.
3. $\theta_i^* = \arg\min_\theta \hat{R}_{\text{inner}}(\theta; D \setminus D_i)$.
4. Fit $f_i = \mathcal{A}_{\theta_i^*}(D \setminus D_i)$ on all of $D \setminus D_i$.
5. Evaluate: $e_i = \frac{1}{|D_i|} \sum_{(x,y) \in D_i} L(y, f_i(x))$.

Nested CV estimate:
$$\hat{R}_{\text{nested}} = \frac{1}{K} \sum_{i=1}^K e_i$$

### Why this is unbiased

Outer fold $i$ uses $\theta_i^*$ chosen without seeing $D_i$. So $f_i$ was trained via a well-defined procedure (A = tune + fit on $D \setminus D_i$) and is tested on independent data $D_i$. The nested CV estimator is the CV estimator for the procedure A applied to $D$:

$$\hat{R}_{\text{nested}} \approx \mathbb{E}_D[\text{Risk of procedure } A(D)]$$

with the usual slight bias from training on $(K-1)/K$ of the data.

### Computational cost

If outer folds = $K$, inner folds = $K'$, hyperparameter grid size = $M$:

- Single-loop CV with tuning: $K \cdot M$ model fits.
- Nested CV: $K \cdot K' \cdot M$ model fits.

For $K = 5, K' = 5, M = 20$: 100 fits (single) vs 500 fits (nested). This is the price of unbiased estimation.

### Choosing $K$ and $K'$

- $K$ (outer): 5 or 10. Same considerations as standard CV.
- $K'$ (inner): 3 or 5. Less critical because inner CV is only for selection, not for the final reported estimate.
- If compute is a constraint, smaller $K'$ is fine: inner CV noise does not affect the bias of the nested estimate, only the quality of hyperparameter selection.

### Variants

- **Nested CV with repeated outer folds**: for tighter estimates, repeat the outer loop with different random splits.
- **Nested CV with early stopping**: instead of evaluating all $M$ configurations, use Bayesian optimization or other smarter search on the inner loop. Bias is the same (still honest), and compute is reduced.
- **Mostly inner, briefly outer**: for expensive models, some argue for using only the outer loop for a few final-model candidates, and relying on a single hold-out at the outermost level.

## 7.4 Worked Examples

### Example A: Nested 5-fold CV with 20 hyperparameter configs

Dataset: $n = 1000$. Algorithm: random forest. Grid: 20 configurations (4 depths × 5 max-features values).

Outer 5-fold split: 5 outer folds of 200 each.

For each outer fold (say fold 1):
- Outer training: 800 examples.
- Inner 5-fold on those 800: 5 inner folds of 160 each.
- For each of 20 configs: compute 5-fold CV score on the 800.
- Pick the best (say: depth=6, max_features=sqrt).
- Refit RF with those hyperparams on all 800.
- Evaluate on outer test fold (200 examples).

Suppose the 5 outer test scores are 0.83, 0.82, 0.84, 0.81, 0.83. Nested CV estimate: 0.826.

For deployment: retrain the whole procedure (inner CV tuning + fit) on all 1000 examples. The final model's hyperparameters may differ from any chosen in the outer folds.

Total model fits: 5 outer × 5 inner × 20 configs = 500 fits, plus 5 outer refits on full outer train sets, plus 1 final fit on all data. ≈506 fits total.

### Example B: Single-loop vs. nested on a small dataset

Dataset: $n = 100$, 50 features, binary classification.

**Single-loop 5-fold CV**, 50 configurations: best CV accuracy = 0.78.

**Nested 5x5 CV**, same 50 configurations: outer test accuracies 0.70, 0.72, 0.74, 0.71, 0.73. Nested estimate = 0.72.

Gap: 6 percentage points. The single-loop overestimated substantially because with $n = 100$ and 50 configs to choose from, there is plenty of room for selection bias.

Deploying the model based on the single-loop 0.78 would lead to disappointment. The nested 0.72 is the honest expectation.

### Example C: Hyperparameter stability diagnosis

Run nested 5-fold CV for lasso regression, tuning $\lambda$ from {0.01, 0.1, 1, 10, 100}.

Outer fold selections:
- Fold 1: $\lambda = 0.1$
- Fold 2: $\lambda = 0.1$
- Fold 3: $\lambda = 1$
- Fold 4: $\lambda = 0.1$
- Fold 5: $\lambda = 0.1$

Most folds pick $\lambda = 0.1$. High confidence that this (or near it) is the right choice. At deployment, refit with $\lambda$ chosen by one more CV on full data — probably $\lambda = 0.1$.

Contrast:
- Fold 1: $\lambda = 0.01$
- Fold 2: $\lambda = 100$
- Fold 3: $\lambda = 1$
- Fold 4: $\lambda = 10$
- Fold 5: $\lambda = 0.1$

Wild instability. Several possibilities: (1) the hyperparameter landscape is flat (all values perform similarly), (2) the data is too noisy for any stable choice, (3) your search grid is too coarse or extreme. Investigate before deploying.

### Example D: Partial nested CV for very expensive models

Deep learning, $n = 10,000$, each training takes 2 hours.

Full nested 5x5 with 10 configs: 250 model fits × 2 hours = 500 hours = 3 weeks on one GPU. Infeasible.

Pragmatic alternative:
- Reserve a 20% test set up front (never touched).
- Do single-loop CV on the remaining 80% with all 10 configs, pick best.
- Refit best config on the 80%, evaluate once on the 20% test.

This is a "hold-out test" alternative to nested CV. Fewer fits (50 vs 250) and still unbiased for the final test score, at the cost of using less data.

### Example E: When nested CV doesn't help

If you have only one hyperparameter configuration (no tuning), single-loop CV is unbiased. Nested CV reduces to standard CV with extra useless computation.

If hyperparameter search is so small that selection bias is tiny (say, comparing 2 configs), nested CV's benefit is marginal, and the extra cost may not be worth it.

## 7.5 Relevance to ML Practice

### Where nested CV is standard

- **Rigorous performance reporting** in scientific publications.
- **Small-to-medium datasets** with hyperparameter tuning.
- **Regulated settings** (medical devices, financial products) where honest performance claims matter legally.
- **Benchmarking algorithms** against each other.

### When to skip nested CV

- **Large datasets**: use train/val/test split. The val set is large enough for unbiased model selection.
- **When compute is limiting**: a single hold-out test set achieves the same unbiased-ness with far less compute.
- **Simple models with no hyperparameters**: nothing to select.

### Trade-offs

| Approach | Pros | Cons |
|---|---|---|
| Single-loop CV | Cheap | Biased |
| Nested CV | Unbiased | K× more expensive |
| Train/val/test hold-out | Unbiased, cheap | Needs more data |
| Bootstrap-based approaches | Good for small $n$ | Complex to implement well |

### Practical tips

- For a quick sanity check, compare single-loop CV to a final hold-out estimate. If they match, selection bias is small and single-loop is fine. If they diverge, you need nested or larger hold-out.
- Report hyperparameter stability alongside nested CV performance.
- Use the same CV folds across algorithms for fair comparison (paired analysis).

## 7.6 Common Pitfalls & Misconceptions

### Pitfall 1: Reporting single-loop CV as "performance" after tuning

The #1 error. Any time you select among configurations, the selected score is optimistically biased.

### Pitfall 2: Using nested CV but reusing the same fold structure at both levels

If inner and outer folds are related (e.g., inner CV shares points with outer test fold), leakage can occur. Make them independent.

### Pitfall 3: Touching the outer test set in any preprocessing step

Classic mistake: scaler fit on full data (including outer test). Even if the model itself only trains on outer training, the scaler leaked.

### Pitfall 4: Not retraining the procedure for deployment

Nested CV estimates the procedure's performance, not a specific model's. The deployed model should be a fresh run of the procedure on all data, not copies of the outer-fold models.

### Pitfall 5: Being surprised that outer folds pick different hyperparameters

This is expected. Hyperparameters are a random variable when data is random. Nested CV handles this by estimating the performance of the procedure as a whole.

### Pitfall 6: Nested CV on very small data

With $n = 50$ and nested 5x5 CV, each inner training fold has 32 examples. Hyperparameter tuning on 32 examples is extremely noisy. Consider simpler model classes or more data.

### Pitfall 7: Using nested CV to pick hyperparameters

Nested CV is a performance estimator, not a hyperparameter selector. After nested CV, if you want to deploy, do one more round of tuning CV on the full dataset to pick final hyperparameters.

### Pitfall 8: Ignoring inner CV stability

If inner CV gives highly variable scores, your hyperparameter selections are noisy, and the overall estimator's variance can be large. Worth checking.

---
---

# Topic 8: Pitfalls & Best Practices (Data Leakage Prevention, Independence Assumptions)

## 8.1 Motivation & Intuition

Every previous topic has hinted at leakage and assumption violations. This topic compiles the most important ones into a unified treatment, because **the single largest cause of CV giving wrong answers is leakage**, not algorithm choice.

A leaked model can look near-perfect in CV and fail catastrophically in deployment. The pattern is so common that detecting it is a core ML skill.

### What is leakage?

**Leakage is any situation where information about the test set (or future data) has influenced the model-building process.** The model's CV performance benefits from information that will not be available at deployment, so the CV estimate is inflated.

Leakage is not always obvious. It can hide in:
- Preprocessing steps fit on the wrong data.
- Features that encode the target directly or indirectly.
- Group structure that random CV ignores.
- Temporal structure that random CV ignores.
- Spatial structure that random CV ignores.
- Duplicate records across train and test.
- Oversampling methods applied before splitting.

### Concrete story: a real-world leakage example

A team builds a model to predict hospital readmission. They include "discharge summary length" as a feature. CV accuracy: 92%. Deployment: 61%.

**Why**: discharge summary length is correlated with readmission — but the direction is backward. Patients who will be readmitted tend to have more detailed discharge summaries (the clinician documents more carefully for complex cases). The feature is partially a product of future events, not a predictor of them. In training, the model used this backward correlation. In deployment, for truly new admissions, the feature is not as predictive.

This is a subtle form of leakage: not time-based in the obvious sense (the discharge summary is available at admission time for a new patient), but the summary-length pattern in historical data reflects outcomes that are not yet known for current patients.

## 8.2 Conceptual Foundations: Categories of Leakage

### 1. Train-test contamination

The most blatant form: test data literally appearing in the training set.
- Duplicate records (same row, different IDs).
- Near-duplicates (same patient, slightly different record).
- Data that was appended to the training set after the split was defined.

### 2. Preprocessing leakage

Fitting preprocessing on the entire dataset (including test) before splitting.
- Standardization (mean and std computed on all data).
- Normalization to a global range.
- PCA fit on all data.
- Target encoding (replacing categorical values with target means computed on all data).
- Feature selection (picking features by correlation with target on all data).
- Imputation (filling missing values with global statistics).

### 3. Target leakage (post-hoc features)

Features that encode future information about the target.
- "Customer received discount" as a feature for churn prediction, when discounts are offered only to customers who signaled intent to leave.
- "Number of support tickets" for fraud detection, when tickets are filed after fraud detection.
- "Transaction was approved" when approval itself depends on the fraud label.

### 4. Temporal leakage

Using data from the future to predict the past.
- Random CV on time-series.
- Features computed with future observations (moving averages centered on each point, future-looking features).
- Labels that span future periods without purging.

### 5. Grouped leakage

Examples from the same group in both train and test.
- Multiple records per user / patient / device.
- Articles by the same author.
- Images from the same scene/session.

### 6. Spatial leakage

Nearby locations in both train and test, benefiting from autocorrelation.

### 7. Selection bias / sampling leakage

The training data is not representative of the deployment data in a structured way.
- Sampled only positive examples (case-control studies used without adjustment).
- Sampled only from certain groups (gender, age, geography).
- Survivorship bias.

### 8. Label leakage during annotation

The annotator had access to information that won't be available at prediction.
- Radiologist labeling an X-ray while seeing the patient's final diagnosis.
- Human-in-the-loop labelers who saw the model's prediction before labeling.

## 8.3 Mathematical Formulation: Independence and Its Violations

### The iid assumption in CV

Formally, CV assumes training and test examples are iid draws from a common distribution $P$. Mathematically, this appears in two places:

1. **Within the dataset**: all examples are exchangeable. Any permutation gives an equivalent training set.
2. **Between training and deployment**: test distribution equals training distribution.

If either is violated, CV estimates are biased.

### Quantifying leakage bias

Let $R_{\text{true}}$ be true deployment risk. Let $R_{\text{CV}}$ be CV estimate. Define leakage bias:

$$\Delta_{\text{leak}} = R_{\text{true}} - R_{\text{CV}}$$

Positive $\Delta_{\text{leak}}$ means CV was optimistic (deployment is worse). In all leakage types listed, $\Delta_{\text{leak}} \geq 0$ (usually strictly).

### The independence model for grouped data

Consider a mixed-effects decomposition:

$$y_{i,g} = f(x_{i,g}) + u_g + \epsilon_{i,g}$$

where $u_g$ is a group-level random effect and $\epsilon_{i,g}$ is noise. Random CV can exploit $u_g$ (since some $u_g$ are in both train and test folds), but group CV forces the model to predict $u_g$ for new groups (which it cannot, except via the mean $\mathbb{E}[u_g]$).

The leakage bias for random CV in this setting is approximately

$$\Delta_{\text{leak}} \approx \text{Var}(u_g) / \text{Var}(y)$$

which is the fraction of variance explained by group effects. In many medical datasets, this is 30-50%.

### Temporal leakage: AR model example

For $y_t = \phi y_{t-1} + \epsilon_t$, the best predictor of $y_t$ given past is $\hat{y}_t = \phi y_{t-1}$, with MSE $\sigma_\epsilon^2$. If CV lets you also see $y_{t-1}$ directly (adjacent in time), MSE can drop close to zero. The bias scales with $\phi^2 \cdot \text{Var}(y)$.

## 8.4 Worked Examples

### Example A: Preprocessing leakage demonstration

Dataset: $n = 1000$, 50 features, binary label.

**Wrong (leaked)**:
```python
X_scaled = StandardScaler().fit_transform(X)  # uses all data
selector = SelectKBest(k=10).fit(X_scaled, y)  # uses all data, including label
X_selected = selector.transform(X_scaled)
scores = cross_val_score(LogisticRegression(), X_selected, y, cv=5)
print(scores.mean())  # 0.91
```

**Correct (pipeline)**:
```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('model', LogisticRegression())
])
scores = cross_val_score(pipeline, X, y, cv=5)
print(scores.mean())  # 0.82
```

The wrong version overestimates by 9 points. The feature selection was leaking target information into "preselected" features whose goodness was validated on the same test data.

### Example B: Target leakage detection

Feature: "was_contacted_by_support_in_last_30_days" for a churn-prediction model.

Workflow to detect:
1. Check timing: support contact is often triggered by cancellation attempts. The feature may be contemporaneous with churn, not antecedent.
2. Compute correlation: if correlation with target is suspiciously high (say > 0.5), investigate.
3. Ablation: remove the feature and see if CV performance drops. A drop from 0.92 → 0.78 suggests the feature was doing the heavy lifting in a suspicious way.
4. Train on data from time $T$ and evaluate on data from $T+\tau$ with $\tau$ large enough that contacts after churn events are excluded.

Resolution: either remove the feature, or properly define it using only data available at prediction time.

### Example C: Duplicate detection

Large medical dataset. Workflow:
1. Compute fingerprint of each record (hash of key features).
2. Find duplicate fingerprints.
3. Examine: are they different examples that coincidentally match, or are they copies of the same underlying record?
4. For true duplicates, remove one copy. For legitimate similar records, keep both but consider group CV.

### Example D: Cross-validation on a mixed iid / time-series dataset

Dataset: sensor readings from 100 devices over 3 years.

Questions to ask:
- Will deployment involve new devices? → Group CV (by device).
- Will deployment involve future time periods on existing devices? → Time-series CV.
- Both? → Combined scheme.

**Combined scheme**: split outer folds by device (each fold tests on 20 devices). Within each outer training set, split inner folds by time for hyperparameter tuning. This evaluates generalization to new devices in future time periods.

### Example E: Sanity check with a dummy feature

One way to detect leakage: add a feature that is just the target (with noise). If CV performance improves a lot (from 0.80 to 0.95), the pipeline is exploiting target information. Then try adding a feature that is the target shifted by some transformation. Find where performance stops improving — that is your leakage threshold.

This sounds silly but it catches real issues.

## 8.5 Relevance to ML Practice

### Best practices checklist

Before running CV, verify:

1. **Preprocessing inside a pipeline**: all data-dependent transformations happen inside the CV loop.
2. **Splitting strategy matches deployment**: time-ordered if deployment is future; group-aware if deployment is new entities; spatial if deployment is new regions.
3. **Labels do not leak through features**: each feature must be computable from data available at prediction time.
4. **No duplicate or near-duplicate records across train and test**.
5. **Any oversampling (SMOTE, etc.) happens inside the CV fold, not before**.
6. **Imputation strategies use training-fold statistics only**.
7. **Feature engineering that uses target (target encoding, weight-of-evidence) happens inside the fold**.
8. **Random seeds are fixed for reproducibility** but results are reported across multiple seeds when possible.
9. **You report both mean and spread of CV scores, not just mean**.
10. **Final test set (for generalization estimation) is touched exactly once**.

### Detection tools

- **Permutation test**: shuffle labels and rerun CV. If scores stay high, something is wrong.
- **Time-reversal test**: if there's a time axis, try training on future and predicting past. If performance is about the same, you have time-invariant leakage.
- **Feature importance audit**: inspect top features for anything that seems future-looking.
- **Train-test distribution comparison**: plot feature distributions in train vs test. Big differences indicate leakage or distribution shift.

### When to suspect leakage

Red flags:
- CV accuracy is "too good" for the problem.
- Top features are surprisingly "easy" ones (dates, IDs, metadata).
- Performance doesn't degrade when you remove seemingly important features.
- CV doesn't match final hold-out test set.
- CV doesn't match production performance after deployment.

## 8.6 Common Pitfalls & Misconceptions

This whole topic is essentially a pitfalls guide. Here are additional ones not explicitly covered.

### Pitfall 1: "I only used CV to tune hyperparameters, so it can't be biased"

Wrong. Tuning on CV and reporting that same CV score is biased. The CV score was used to select; the reported value is the max of a noisy process.

### Pitfall 2: "Leakage only affects small datasets"

Big datasets can still leak. Preprocessing leakage scales with the strength of the leaked information, not with dataset size. A target encoding leak can inflate performance on a 10M-row dataset just as much as on a 100-row dataset.

### Pitfall 3: "As long as I use a pipeline, I'm safe"

Pipelines handle standard preprocessing correctly. They do not handle:
- Feature engineering outside the pipeline.
- Group/time/spatial structure.
- Duplicates.
- Target-encoded features not placed inside the pipeline.

### Pitfall 4: "I can trust the CV as long as I use a different test set for final evaluation"

Close, but if you use the CV extensively (many iterations of feature engineering, tuning, comparison), the CV can still be leaking even if the final test is honest. The final test gives you an unbiased estimate of the chosen model, but it doesn't reveal how much you were fooling yourself along the way.

### Pitfall 5: "Leakage only inflates performance; it's not a safety issue"

Inflated performance leads to deploying underperforming models, which can be a safety issue in medical or high-stakes domains. It also masks bugs. A model that "performs better than it should" may have a subtle bug you should debug.

### Pitfall 6: "The test set is fine as long as it was split randomly"

Random splits are correct for iid data. They are wrong for time-series, grouped, spatial, or otherwise structured data.

### Pitfall 7: "I used scikit-learn, so it's correct"

scikit-learn provides tools for correct CV, but default `KFold` is random and does not enforce group/time/spatial awareness. You have to choose the right splitter.

### Pitfall 8: "My model reports feature importance. I'll trust that to spot leakage."

Feature importance shows which features the model uses. It does not show whether those features encode leaked information. Manual inspection is essential.

---
---

# Interview Preparation Section

The following questions are organized by difficulty. Answers are technically rigorous and at the level expected in an applied scientist / ML scientist interview.

## Foundational Questions

### Q1: What is cross-validation and why do we need it?

**Answer**: Cross-validation is a resampling method for estimating the generalization error of a learning procedure. The core idea is to partition the available data into subsets; train on some and evaluate on the rest; and average the evaluation scores over multiple configurations of training/test splits.

We need it because training error is a biased (optimistic) estimate of true generalization error — the model has seen the training data. Setting aside a single hold-out set gives an unbiased estimate but (a) is noisy (depends on the specific split) and (b) wastes training data. CV addresses both: it averages across multiple splits to reduce noise, and each example gets to be a test point exactly once, so information is not wasted.

More formally, CV estimates $\mathbb{E}_{D'}[R(\mathcal{A}(D'))]$ where $D'$ is a dataset of size $n(k-1)/k$ drawn from the same distribution as $D$. This is slightly pessimistic for the deployment model (trained on all $n$) but much more accurate than training error.

### Q2: What is the difference between training, validation, and test sets?

**Answer**: 
- **Training set**: used to fit model parameters (weights, tree splits, etc.).
- **Validation set**: used to select among models or tune hyperparameters. The validation set is visible during model development but is not used to fit parameters directly.
- **Test set**: used exactly once, after all modeling decisions are finalized, to obtain an honest generalization estimate.

Using validation and test sets interchangeably is a classic error. The validation set is "seen" multiple times during tuning, so its reported score is biased. The test set must be touched once to remain unbiased. With CV, the training/validation split is replaced by CV inside the training data; the outer test set remains separate (this is nested CV or a final hold-out).

### Q3: Why is training error not a good estimate of generalization error?

**Answer**: The model is optimized to minimize loss on the training data, so the training error is the minimum over the model class for that specific data. Generalization error is the expected loss on a fresh sample. The difference is called **optimism** and it grows with model flexibility.

Extreme example: 1-nearest-neighbor has zero training error on any dataset (every point is its own nearest neighbor). Its generalization error is far from zero. Any flexible model — decision trees, deep networks, kernel methods — has training error that can be driven arbitrarily low, while generalization depends on matching the true data-generating process.

Statistical learning theory formalizes this via concentration bounds: with high probability, generalization error ≤ training error + complexity term, where the complexity term depends on model capacity (VC dimension, Rademacher complexity, etc.). The gap can be large for flexible models on small datasets — exactly where CV helps most.

### Q4: Why k=5 or k=10 is the standard for k-fold CV?

**Answer**: It's a bias-variance tradeoff of the CV estimator.

- **Small k**: lower variance because folds are nearly independent (small training set overlap). But high bias because each model is trained on a small subset (1/2 for k=2), far from the full dataset size.
- **Large k**: low bias because each model is trained on almost all the data. But high variance because the fold models are very similar (overlapping training sets), so their errors are correlated; averaging over many correlated estimators doesn't reduce variance as much as averaging over independent ones.

Empirically and theoretically (Hastie, Tibshirani, Friedman), k=5 or k=10 hits the sweet spot for typical ML tasks. k=5 is slightly cheaper; k=10 is slightly less biased. Either is defensible.

### Q5: What is stratified k-fold and when should you use it?

**Answer**: Stratified k-fold partitions data so that each fold has approximately the same class distribution as the full dataset. Use it for classification, especially:
- Imbalanced data (minority class might be missing or underrepresented in random folds).
- Small datasets where random fluctuations in class distribution add noise to fold-level scores.
- Any multiclass problem where some classes have few examples.

For regression, stratification isn't standard but can be applied by binning the target and stratifying on the bins. This is useful when the target distribution is long-tailed or multimodal.

Stratification almost always reduces variance of the CV estimator without changing bias, so it is a free win when applicable. The exception is when groups or other structure preempt the splitting strategy.

### Q6: Explain the intuition behind nested CV without math.

**Answer**: Nested CV answers two questions at once without letting them interfere with each other: "which model configuration is best?" and "how well will the chosen configuration perform on new data?"

If you use the same CV for both, you contaminate the performance estimate. You would be saying "this is the score of the best configuration," but "best" is selected based on that very score — a form of cherry-picking. The selected score is artificially high.

Nested CV puts the "pick the best" operation inside an inner loop, using only the training portion of each outer fold. The outer fold's test portion never influenced the selection, so its score is honest.

The cost is computational: you do CV inside CV. The benefit is an unbiased estimate.

### Q7: What does "data leakage" mean in the context of CV?

**Answer**: Data leakage is any situation where information about the test fold influences the training process, causing CV to overestimate generalization performance.

Common sources:
- Preprocessing (scaling, PCA, imputation, feature selection) fit on all data before splitting.
- Target encoding categorical variables using target statistics from all data.
- Features that inadvertently encode the label (e.g., features computed after the outcome is known).
- Same entity (patient, user) appearing in both train and test sets (without group CV).
- Time-series data split randomly instead of chronologically.
- Duplicate or near-duplicate records.

Detection: check if CV score drops substantially when preprocessing moves inside a pipeline, when you switch to group/time-based CV, or when you deduplicate.

## Mathematical Questions

### Q8: Derive the PRESS statistic for leave-one-out CV of linear regression.

**Answer**: For OLS, $\hat{\beta} = (X^T X)^{-1} X^T y$ and fitted values $\hat{y} = Hy$ where $H = X(X^T X)^{-1} X^T$.

When we leave out point $i$, the new design matrix is $X_{(-i)}$ (minus row $i$). We need the fit $\hat{\beta}^{(-i)}$ and the LOO prediction at $x_i$: $\hat{y}_i^{(-i)} = x_i^T \hat{\beta}^{(-i)}$.

Key identity (Sherman-Morrison): if we write $X^T X = X_{(-i)}^T X_{(-i)} + x_i x_i^T$, then

$$(X_{(-i)}^T X_{(-i)})^{-1} = (X^T X - x_i x_i^T)^{-1} = (X^T X)^{-1} + \frac{(X^T X)^{-1} x_i x_i^T (X^T X)^{-1}}{1 - h_{ii}}$$

where $h_{ii} = x_i^T (X^T X)^{-1} x_i$.

Using this, we can show (after algebraic manipulation):
$$\hat{y}_i^{(-i)} = \hat{y}_i - \frac{h_{ii}}{1 - h_{ii}}(y_i - \hat{y}_i)$$

Therefore:
$$y_i - \hat{y}_i^{(-i)} = \frac{y_i - \hat{y}_i}{1 - h_{ii}}$$

And the PRESS statistic (LOOCV sum of squared errors):
$$\text{PRESS} = \sum_i \left( \frac{y_i - \hat{y}_i}{1 - h_{ii}} \right)^2$$

$$\hat{R}_{\text{LOOCV}} = \frac{\text{PRESS}}{n}$$

The elegance: one fit, no leave-one-out refitting, just reweight the residuals by $1/(1 - h_{ii})$.

### Q9: Why does LOOCV have higher variance than 10-fold CV?

**Answer**: Counterintuitively, averaging over more folds (n for LOOCV vs 10) does not always reduce variance because the averaging is over **correlated** random variables.

Formally:
$$\text{Var}[\hat{R}_{CV}] = \frac{1}{k}\sigma^2 \cdot [1 + (k-1)\bar{\rho}]$$

where $\bar{\rho}$ is the average pairwise correlation between fold losses and $\sigma^2$ is the per-fold loss variance.

For LOOCV, the $n$ leave-one-out fits differ by only 2 points (one removed, one added), so the predictors are nearly identical, leading to high $\bar{\rho}$ close to 1. The bracket $[1 + (n-1)\bar{\rho}]$ scales nearly linearly in $n$, so $\text{Var} \approx \sigma^2 \bar{\rho}$ — independent of $n$.

For 10-fold, each training set differs from another by $\sim 20\%$ of points, so fold models are less correlated. $\bar{\rho}$ is smaller. The bracket is smaller relative to $k$, and variance reduction from averaging is effective.

The classical result (Kohavi, 1995; Hastie et al.): 10-fold CV has lower MSE (bias² + variance) than LOOCV for most practical settings, despite LOOCV's lower bias.

### Q10: What does k-fold CV actually estimate, mathematically?

**Answer**: Let $\bar{R}_m = \mathbb{E}_{D \sim P^m}[R(\mathcal{A}(D))]$ be the expected risk of a model trained on $m$ iid examples. Each fold of k-fold CV trains on $n(1-1/k)$ examples, so

$$\mathbb{E}[\hat{R}_{CV}(k)] = \bar{R}_{n(1-1/k)}$$

NOT $\bar{R}_n$. The difference $\bar{R}_{n(1-1/k)} - \bar{R}_n$ is the pessimistic bias of k-fold CV.

For $k = n$ (LOOCV): bias $\to 0$ as $n \to \infty$.

For $k = 2$: bias can be large if the learning curve is steep between $n/2$ and $n$.

Asymptotically, both bias and variance go to zero as $n \to \infty$, so the CV estimate is consistent.

The practical implication: when you use CV to select a model, then retrain on all $n$ examples, the deployed model's true risk is *better* than the CV estimate (because it was trained on more data), all else equal.

### Q11: Derive the bias-variance decomposition and explain how CV fits.

**Answer**: For squared error loss at a point $x$:
$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{bias}^2} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{variance}} + \underbrace{\sigma^2_\epsilon}_{\text{irreducible}}$$

where the expectation is over training sets.

CV aims to estimate the left-hand side (expected squared error) by empirical averaging. The estimate itself has bias and variance:
- **Bias of CV**: $\mathbb{E}[\hat{R}_{CV}] - \bar{R}_n$, reflecting that CV trains on less than the full data.
- **Variance of CV**: $\text{Var}[\hat{R}_{CV}]$, reflecting fold-to-fold noise.

So there is a meta-bias-variance: the quantity being estimated has bias and variance, and the estimator of that quantity also has bias and variance. Nested complexity!

Practical rule: more data $\to$ model bias and variance both decrease, CV bias decreases, CV variance decreases. Bigger k $\to$ CV bias decreases, CV variance potentially increases. Choose k around 10 to balance.

### Q12: How does the choice of loss function interact with CV?

**Answer**: CV is loss-agnostic in principle — you can use any loss function. But some practical considerations:

1. **Metric vs. loss**: you might train with cross-entropy but evaluate with AUC. CV should use the evaluation metric for model selection (since that's what matters for deployment).

2. **Aggregation**: some metrics aggregate naturally across folds (MSE, accuracy — just weighted average). Others don't: AUC computed per-fold and averaged differs from pooled AUC (computed over all predictions at once). For AUC-type metrics, the pooled approach is preferred if feasible.

3. **Variance**: metrics based on ranks (AUC, Spearman) can have different variance characteristics than pointwise metrics. CV variance estimates may need to be adjusted.

4. **Thresholds**: metrics like precision and recall depend on a decision threshold. If the threshold is tuned per fold, the metric values are noisier than if a fixed threshold is used. This is another axis of CV design.

5. **Small test folds**: for imbalanced data and per-fold AUC, test folds must be large enough to have enough positives. Stratified CV helps.

### Q13: Prove that the LOOCV estimator of the mean squared error is (nearly) unbiased for the expected prediction error of a model trained on $n-1$ samples.

**Answer**: Let $\mathcal{D}_n = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ be $n$ iid samples from $P$. Let $\mathcal{A}$ be a deterministic learning algorithm. The LOOCV estimator:

$$\hat{R}_{\text{LOO}} = \frac{1}{n} \sum_{i=1}^n L(y_i, \mathcal{A}(\mathcal{D}_n \setminus \{(x_i, y_i)\})(x_i))$$

Take expectation over $\mathcal{D}_n$:

$$\mathbb{E}[\hat{R}_{\text{LOO}}] = \frac{1}{n} \sum_{i=1}^n \mathbb{E}[L(y_i, \mathcal{A}(\mathcal{D}_n \setminus \{(x_i, y_i)\})(x_i))]$$

By exchangeability of iid samples, each term in the sum has the same expectation:

$$= \mathbb{E}[L(y_1, \mathcal{A}(\mathcal{D}_n \setminus \{(x_1, y_1)\})(x_1))]$$

Here $\mathcal{D}_n \setminus \{(x_1, y_1)\}$ is a dataset of $n-1$ iid samples, call it $\mathcal{D}_{n-1}$, and $(x_1, y_1)$ is independent of $\mathcal{D}_{n-1}$ (since $x_1$ is independent of $x_2, \ldots, x_n$). So:

$$\mathbb{E}[\hat{R}_{\text{LOO}}] = \mathbb{E}_{\mathcal{D}_{n-1}, (x, y)}[L(y, \mathcal{A}(\mathcal{D}_{n-1})(x))] = \bar{R}_{n-1}$$

which is the expected risk of a model trained on $n-1$ iid samples. The LOOCV estimator is thus exactly unbiased for $\bar{R}_{n-1}$, and nearly unbiased for $\bar{R}_n$ (the target of interest for deployment) when the learning curve is flat near $n$.

### Q14: Give a mathematical explanation for why CV with overlapping training sets has higher estimator variance than CV with disjoint training sets.

**Answer**: Consider two estimators of the same quantity, $A = \frac{1}{k}\sum e_i$. 

$$\text{Var}(A) = \frac{1}{k^2}\left[\sum_i \text{Var}(e_i) + \sum_{i \ne j} \text{Cov}(e_i, e_j)\right]$$

If $e_i$'s are independent, the covariance term is zero, and $\text{Var}(A) = \sigma^2/k$.

If $e_i$'s have positive pairwise covariance $c > 0$ (as they do in CV: overlapping training sets produce similar models → similar errors), then:

$$\text{Var}(A) = \frac{\sigma^2}{k} + \frac{k(k-1)c}{k^2} = \frac{\sigma^2}{k} + \frac{(k-1)c}{k}$$

As $k \to \infty$, the first term $\to 0$ but the second $\to c$. So variance is lower-bounded by the covariance, not by zero.

For LOOCV, $c$ is close to $\sigma^2$ (highly correlated fold errors), so variance $\approx \sigma^2$ regardless of $n$. For 10-fold, $c$ is smaller, so the averaging helps more.

## Applied Questions

### Q15: Design a cross-validation scheme for a fraud detection system with severe class imbalance and temporal drift.

**Answer**: Key considerations:
- **Temporal structure**: fraud patterns evolve; deployment is always "predict fraud in the next period from past data."
- **Class imbalance**: fraud is rare (<1% typically). Random splits may have few positives per fold.
- **Grouped structure**: same merchant / same card across many transactions — group-level leakage possible.
- **Label delay**: real fraud labels often arrive weeks after the transaction, creating a gap between feature time and label time.

Proposed scheme:
1. **Outer: time-based expanding window**, e.g., 6 monthly folds. Each fold tests on the next month using all prior months as training.
2. **Within training period**: group-aware (e.g., by card or merchant) to ensure no card appears in both train and validation in the same outer fold.
3. **Stratified on fraud label**: within the group-and-time constraints, balance fraud rate across sub-splits.
4. **Purge and embargo**: exclude training data from the final days before the test period (to handle label delay). Embargo a short period after the test fold before the next training start.
5. **Report per-fold**: AUC, precision@k (where k is the review budget), and calibration. Trends across folds indicate drift.
6. **Use inner CV for hyperparameter tuning**: nested within the outer time structure.

Trade-off: gives an honest estimate of deployment performance at the cost of more complex tooling.

### Q16: You're evaluating a recommendation system. What CV scheme?

**Answer**: Depends on the deployment question.

**Scenario A**: Predict user $u$'s rating on a new item.
- Group by user, split items across folds (or: for each user, split their rated items into train and test).
- Matches deployment: the user will rate items they haven't seen.
- Can be done per-user (leave-one-item-out) or globally (split user-item pairs).

**Scenario B**: Predict ratings for new users.
- Group by user. New users go entirely into test.
- Tests cold-start performance.

**Scenario C**: Predict future interactions.
- Time-based split. Train on interactions before $T$, test on interactions after $T$.
- Most realistic for production deployment.

**Scenario D**: A combination.
- Outer time-based split. Within each training period, group-based CV for cold-start evaluation.

Additional considerations:
- Popularity bias: random test sampling favors popular items. For realistic evaluation, consider per-user evaluation with appropriate weighting.
- Implicit vs. explicit feedback: implicit feedback (clicks, views) has no "negative" labels, so evaluation uses ranking metrics (NDCG, MRR).
- Long-tail handling: evaluate separately on head and tail items.

### Q17: You have 1 million rows of time-series sensor data from 1,000 sensors. You want to predict a fault condition on future data, on potentially new sensors. Design CV.

**Answer**: Two axes: temporal and grouped.

Choices:
1. **Pure time-based CV**, ignoring sensor ID: evaluates generalization to future time, same sensors.
2. **Pure group CV**, ignoring time: evaluates generalization to new sensors, any time.
3. **Combined**: outer time-split, inner sensor-split (or vice versa) — evaluates both.

Deployment dictates: if the model will be deployed on new sensors in the field, option 3 is needed. If only on existing sensors predicting future faults, option 1 suffices.

Recommended design:
- **Outer folds**: time-based, say 4 folds of 3 months each. Train on earlier, test on later.
- **Inner validation (for hyperparameter tuning)**: within each outer training set, split sensors into 5 groups. Train on 4 groups, validate on 1.
- **Final evaluation**: on each outer test fold, evaluate by sensor group. Also report aggregate.
- **Purge**: gap between training end and test start, long enough to handle label delay in fault detection.

Metrics: ROC-AUC, precision@k for fault flags, and cost-weighted loss if fault detection has asymmetric costs.

### Q18: A colleague runs CV on a dataset with many duplicate records. How does this affect the CV estimate and what should be done?

**Answer**: Duplicates cause leakage. The exact duplicate of a test record is in the training set, so the model can simply "look up" the answer during training and reproduce it at test time. The CV estimate is inflated relative to what the model does for genuinely new records.

Magnitude depends on the fraction of duplicates. 10% duplication rate can inflate accuracy by several percentage points.

Actions:
1. **Deduplicate**: compute record-level hashes (or fuzzy matching for near-duplicates), keep one copy of each.
2. **If duplicates are legitimate (e.g., repeated measurements of the same entity)**: group CV by the entity ID.
3. **Report before/after**: show CV scores pre- and post-deduplication to quantify the leakage.
4. **Investigate root cause**: are duplicates from a data pipeline bug, or from legitimate domain-specific repetition? The answer informs downstream decisions.

### Q19: A simple CV gives 0.90 accuracy, but nested CV gives 0.80. What's happening?

**Answer**: The 10-point gap is selection bias from hyperparameter tuning.

Single-loop CV: for each hyperparameter configuration, compute CV score. Pick the configuration with the highest score. Report that score. Because you picked the maximum over many noisy estimates, the reported score is biased upward.

Nested CV: for each outer fold, tune hyperparameters on that fold's training data only (via inner CV). Refit and evaluate on the outer test fold. Each outer test score is computed with hyperparameters chosen without seeing that test fold, so the nested estimate is unbiased.

The 10-point gap indicates substantial over-fitting to the CV scores during hyperparameter search. This is likely because:
- Many configurations were tried.
- The dataset is small (high CV noise).
- Some configurations can exploit random patterns in folds (flexible models).

The nested 0.80 is the honest estimate of what the "tune + fit" pipeline will achieve in deployment.

For deployment, retrain the pipeline (tuning + fit) on all data. The deployed model's true performance will be near 0.80 (likely slightly better since it uses slightly more data).

### Q20: You are building a model to predict 30-day mortality for ICU patients. Design the CV protocol.

**Answer**: Multiple considerations:
- **Grouped**: patients are the relevant unit. One patient can have multiple ICU admissions or multiple records within one stay. Group by patient (or by patient-admission, depending on the prediction task).
- **Temporal**: if data spans years, treatments and patient populations change. Temporal split ensures evaluation reflects future deployment.
- **Class imbalance**: mortality rate is often 5-15%, so stratification by outcome matters.
- **Site differences**: if data is from multiple hospitals, test generalization to a held-out hospital.

Proposed design:
1. **Outer folds**: hold out one hospital at a time (to test site generalization). Or, if only one hospital's data is available: hold out the most recent time period.
2. **Middle split**: within each outer training set, temporal split (train on years 1-3, validate on year 4).
3. **Innermost CV**: for hyperparameter tuning, group k-fold by patient within the training period.
4. **Stratification**: stratify on the mortality label in all splits.
5. **Metrics**: AUC, calibration (e.g., Hosmer-Lemeshow or calibration curves), decision-curve analysis. Report by subgroup (age, sex, admission source) for fairness audit.
6. **Label leakage check**: ensure features do not include any post-outcome information (e.g., discharge notes, death certificates, autopsy reports).

This is intentionally layered because clinical models face real threats of cross-site, cross-time, and within-patient leakage. The gold-standard protocol is multi-site external validation.

### Q21: You have 50 labeled images and want to evaluate a model. What do you do?

**Answer**: 50 images is tiny. Options:

1. **LOOCV**: 50 folds, each holding out one image. Low bias, but high variance on the estimator. Use if the model has low variance (e.g., simple linear model).
2. **5-fold stratified CV**: 10 images per fold. Higher bias but lower variance. Report fold-level variability.
3. **Repeated 5-fold**: 10 repeats with different random seeds, stratified, gives 50 fold-level scores. Good compromise.
4. **Leave-one-group-out** if there's a group structure (e.g., each image is of a patient).

With such small data, I would:
- Prefer LOOCV or repeated stratified CV over single 5-fold (variance too high).
- Report confidence intervals via bootstrap on CV predictions.
- Be honest about the uncertainty: the "accuracy" is really a range.
- Consider data augmentation or transfer learning to compensate for the small dataset.
- For deep learning, fine-tuning a pretrained model is much more data-efficient than training from scratch.
- Avoid extensive hyperparameter tuning (selection bias will be severe).

### Q22: Your training data was collected from hospital A. You want to estimate how the model will perform at hospital B. What can CV do?

**Answer**: Standard CV on hospital A's data tells you nothing about hospital B — it assumes the test distribution equals the training distribution.

Options to get closer to the hospital-B deployment question:

1. **Collect labeled data from hospital B**: most direct. Even a small validation set at B gives an unbiased estimate.
2. **Simulated covariate shift**: if you have at least some unlabeled B data, reweight the A data to look more like B using importance weighting or domain adaptation, then evaluate.
3. **Subgroup CV on A**: if A has subgroups (different wards, specialties) that vary in ways similar to the A-B difference, leave-one-subgroup-out CV approximates cross-site generalization.
4. **Domain generalization methods**: train on A with explicit invariance objectives, evaluate CV-style on hold-out B data.

Standard CV on A gives a best-case estimate — an upper bound on what you'll see at B, unless B is actually more similar to A than A's internal variation.

## Debugging & Failure-Mode Questions

### Q23: Your CV accuracy is 95%, but the model performs at 75% on hold-out test data. What are the likely causes?

**Answer**: Some form of leakage or distribution mismatch. Diagnostic steps:

1. **Preprocessing leakage**: is any preprocessing fit outside the CV loop? Scaling, PCA, feature selection, imputation, target encoding. Move all of them inside the pipeline.

2. **Duplicates between train and test**: check for identical or near-identical records crossing the train/test boundary.

3. **Temporal structure**: is this time-series data? Random CV leaks future to past. Use time-based split.

4. **Grouped structure**: are there repeated entities (users, devices, patients)? Random CV leaks within-group info. Use group CV.

5. **Target leakage in features**: are any features created using the target directly (or post-outcome information)? Examples: summary statistics that include the label, features computed from future events.

6. **Distribution shift**: is the hold-out test from a different time, location, or population? This isn't leakage but a real deployment challenge — the CV was honest for the training distribution, but the test is from a different distribution.

7. **Selection bias in hold-out**: is the hold-out constructed differently (e.g., manually curated, biased sampling)? If so, it reflects that bias.

8. **Class imbalance handling**: if you applied SMOTE or oversampling before splitting, the CV sees synthesized examples similar to test examples.

A systematic debugging approach: take the CV pipeline and run it on the hold-out test data as a single fold. If the score is still 95%, the issue is external (distribution shift). If it drops, the issue is inside the CV pipeline.

### Q24: Your CV scores have high variance across folds (e.g., 0.75, 0.92, 0.68, 0.95, 0.80). What does this mean and what do you do?

**Answer**: High fold-to-fold variance can indicate several things:

1. **Small test folds**: with few examples per fold, fold-level estimates are noisy. Increase $n$ (more data) or increase fold size (fewer folds).

2. **Fold composition variance**: some folds may have harder examples or different class distributions by chance. Use stratified CV (for classification) to reduce.

3. **Grouped structure**: if some groups are harder than others, and different folds contain different groups, fold-level variance is expected. Use group-stratified CV or report per-group performance.

4. **Model instability**: high-variance learners (deep trees, high-dimensional models) give different fits on different training folds, producing variable test scores. Use more data, more regularization, or ensembling.

5. **Drift-like structure**: in a dataset that is time-indexed but used with random CV, some folds happen to have more "old" data than others, producing variable difficulty. Move to time-series CV.

6. **Outliers**: a few hard examples dominate some folds' errors. Check if specific examples drive the differences.

Actions:
- Inspect each fold's examples to find common themes in the bad folds.
- Use repeated CV to check if the pattern is stable across seeds.
- Report median and IQR in addition to mean.
- Consider more folds (higher k) to get finer-grained estimates, if computationally feasible.

### Q25: After deploying a model that had CV accuracy of 85%, production performance is only 65%. Triage.

**Answer**: The offline-online gap can come from leakage (CV was overoptimistic) or distribution shift (deployment is different).

1. **Compare deployment distribution to training**:
   - Plot feature distributions in recent production data vs. training.
   - Compute PSI (population stability index) or KS test for each feature.
   - Check for new categories, missing values, or scale changes.

2. **Check engineering pipeline consistency**:
   - Is feature engineering identical in production vs. training?
   - Bug in production preprocessing is extremely common.
   - Missing values handled differently? Default values different?

3. **Check label distribution**:
   - Base rate shift (e.g., fraud rate changed)?
   - Label delay: are recent "labels" actually incomplete?

4. **Re-audit for leakage**:
   - Was there target leakage that worked in historical data but not in production?
   - Were features computed with future information that isn't available live?

5. **Check model input**:
   - Are all features actually being populated? Missing features can silently fail.
   - Are categorical encodings the same?

6. **Rerun CV with stricter splits**:
   - If random CV was used, try time-based. If a group structure exists, try group CV. See if the CV score drops to match production.

Typical findings: in my experience (and in published literature), most offline-online gaps come from:
- Leakage (training data contains future information).
- Distribution shift (training was on old data, production is on new regime).
- Engineering bugs (feature computed differently in production).

A good deployment strategy includes ongoing monitoring comparing online predictions to offline CV performance, with alerts when they diverge.

### Q26: You use nested CV and the inner loop picks wildly different hyperparameters across outer folds. What does this mean?

**Answer**: Hyperparameter instability. Several interpretations:

1. **Flat hyperparameter landscape**: many configurations perform similarly, so CV noise dominates the selection. In this case, the nested CV estimate is still valid (all configs behave similarly, so the procedure's performance is well-defined), but your inner CV is noisy and wasteful.
   - *Action*: smaller hyperparameter grid, or decide based on a stability / simplicity criterion rather than pure CV score.

2. **Data-dependent optimal hyperparameters**: different subsets of data genuinely prefer different hyperparameters. This suggests the problem has multiple regimes or the model isn't quite right.
   - *Action*: investigate whether there are subgroups that need different models.

3. **Small data / high noise**: CV noise is larger than hyperparameter sensitivity.
   - *Action*: more data, or use more robust selection methods (e.g., one-SE rule).

4. **Degenerate search grid**: grid is too coarse, or includes extreme values that get picked by chance in some folds.
   - *Action*: refine the grid around the region where most folds converge.

The 1-SE rule: instead of picking the hyperparameter with the best CV score, pick the simplest one within 1 standard error of the best. This increases stability at a small cost in accuracy.

### Q27: You're getting negative R² values in time-series CV. How can that be?

**Answer**: R² is defined as $1 - \text{SS}_{\text{res}} / \text{SS}_{\text{tot}}$ where $\text{SS}_{\text{tot}}$ is computed from the test set's mean. A negative R² means the model performs worse than predicting the test set's mean — which is entirely possible, especially in time-series CV.

Causes:
1. **Concept drift**: the relationship between features and target in the test period differs from training. The model's predictions are systematically off.
2. **Baseline artifact**: the test set's mean differs from the training set's mean. The model predicts near the training mean and does worse than "predict the test mean."
3. **Bad features**: in a non-stationary world, features predictive in the past may have spurious relationships with the test-period target.
4. **Overfitting to a training-specific regime**.

Diagnosis:
- Check the test period for unusual events (pandemic, market crash, launch).
- Compare predictions vs. actuals over time — does the model systematically under or over predict?
- Compare to naive baselines (predict previous value, predict training mean, predict seasonal).

Often in time-series, a naive baseline (last observed value) is hard to beat. A CV score of "negative R²" might still be useful if you're beating it, but it indicates your model is fundamentally struggling.

### Q28: You've been asked to debug a CV setup that gives suspiciously high performance. Walk through your audit.

**Answer**: Systematic audit checklist:

1. **Target relationship check**: look at feature-target correlations. Any feature with correlation > 0.9 is suspicious.
2. **Shuffle-labels test**: permute the target labels, rerun CV. If CV score stays high, there's leakage via something the shuffling didn't affect (e.g., identity features).
3. **Pipeline check**: is all preprocessing inside `Pipeline`? Including feature selection, imputation, scaling, target encoding, oversampling.
4. **Split-before-everything check**: literally split first, then inspect — do train and test sets share any records? Fuzzy-match IDs?
5. **Temporal check**: are there date/time columns? If yes, ensure no future-leaking features. Try a time-based split, see if performance drops.
6. **Group check**: are there entity IDs (user, patient, device)? If yes, try group CV, see if performance drops.
7. **Feature audit**: one by one, remove each feature and rerun CV. Features that cause large drops warrant inspection: what do they represent? When are they computed? Are they truly available at prediction time?
8. **External test**: if you have a holdout you've never touched, run the final model on it. A large gap signals leakage.
9. **Sanity metric**: add a pure-noise feature. If the model relies on it in some folds but not others, the CV is noisy but not leaking. If performance inflates, there's something rotten.
10. **Compare to published baselines**: if a similar task in the literature achieves 75% and yours gets 95%, be skeptical.

The most common causes, in my experience: preprocessing leakage (40%), temporal leakage (20%), grouped leakage (15%), target-encoded features outside pipeline (10%), duplicates (10%), other (5%).

## Follow-Up Probing Questions

### Q29: What's the relationship between CV and AIC/BIC?

**Answer**: AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) are penalized likelihood criteria for model selection:

- AIC $= -2 \log \mathcal{L} + 2p$
- BIC $= -2 \log \mathcal{L} + p \log n$

where $\mathcal{L}$ is the maximized likelihood and $p$ is the number of parameters.

Connections to CV:
1. **AIC asymptotically equivalent to LOOCV** for parametric models under regularity conditions (Stone, 1977). For large $n$, picking the model with lowest LOOCV equals picking the lowest AIC.

2. **BIC asymptotically equivalent to a form of CV at a different information rate**. BIC's penalty grows with $\log n$, so it prefers simpler models than AIC/LOOCV.

3. **AIC is "prediction optimal"**: minimizes expected prediction error.
4. **BIC is "consistent"**: picks the true model (if it's in the candidate set) with probability $\to 1$ as $n \to \infty$.

Practical differences:
- CV is more flexible — works for any model, any loss — but requires refitting.
- AIC/BIC are one-shot — compute likelihood and parameter count, done — but assume a particular likelihood structure.
- For models like decision trees, lasso, or neural networks where "number of parameters" is not well-defined, AIC/BIC are less applicable.

### Q30: Can you use CV for unsupervised learning?

**Answer**: Yes, but it requires defining a "loss" that can be evaluated on held-out data.

Examples:
- **Clustering**: use predictive log-likelihood for a generative cluster model (GMM). Train GMM on training fold, evaluate held-out log-likelihood on test fold.
- **Density estimation**: same — evaluate held-out log-likelihood.
- **Dimensionality reduction**: if you have a downstream supervised task, CV with that task. Otherwise, reconstruction error (train encoder on 80%, evaluate reconstruction on 20%).
- **Topic modeling (LDA)**: held-out perplexity.
- **Anomaly detection**: if you have (even partial) anomaly labels, treat as supervised. Otherwise, you need domain-specific metrics.

Challenges:
- Many unsupervised metrics (silhouette score, Davies-Bouldin) do not have a natural "train-test" interpretation — they are computed on the clustering result, not on generalization.
- For algorithms like k-means with random initialization, CV must also handle initialization variance.

### Q31: What is the .632 bootstrap and how does it compare to CV?

**Answer**: The .632 bootstrap is an alternative to CV for estimating generalization error, based on bootstrap resampling.

**Procedure**:
1. For each of $B$ bootstrap iterations, sample a bootstrap training set of size $n$ (with replacement) from the original data.
2. "Out-of-bag" (OOB) examples are those not selected — approximately 36.8% of the data.
3. Train the model on the bootstrap sample, evaluate on OOB.
4. The naive bootstrap estimate is the average OOB error.
5. The .632 estimate combines this with training error: $\hat{R}_{.632} = 0.368 \cdot \hat{R}_{\text{train}} + 0.632 \cdot \hat{R}_{\text{OOB}}$.

**Why .632**: the probability that any given example is NOT in a bootstrap sample is $(1 - 1/n)^n \to 1/e \approx 0.368$, so 0.632 is in the bootstrap. The weighting balances the optimistic bias of training error and the pessimistic bias of OOB error.

**.632+**: an improvement that adjusts for overfitting more carefully.

**Comparison to CV**:
- Bootstrap is computationally similar to CV (B fits vs k fits).
- For small datasets, .632+ often has lower MSE of the estimator than k-fold or LOOCV.
- For larger datasets, CV is typically preferred — simpler, well-understood.
- Both are consistent estimators under standard assumptions.

Random forests use OOB error (essentially a bootstrap CV estimate) as a free generalization estimate.

### Q32: How does CV interact with regularization path algorithms like lasso?

**Answer**: Lasso has a regularization parameter $\lambda$. The standard approach is to compute the full regularization path (solutions for a grid of $\lambda$ values) and use CV to pick the best $\lambda$.

Mechanics:
1. For each CV fold, compute the full lasso path on the training fold (efficient via LARS or coordinate descent).
2. For each $\lambda$ in the grid, compute the fold's validation error.
3. After all folds, average validation errors across folds at each $\lambda$.
4. Pick $\lambda^*$ = argmin of the averaged error (or 1-SE rule).
5. Refit lasso on the full data at $\lambda^*$.

Subtleties:
- The path algorithms are fast enough that you can compute the whole path quickly, making this very tractable.
- Different folds may prefer different $\lambda$'s; picking one global $\lambda$ from averaged CV error is the standard compromise.
- The 1-SE rule: pick the largest $\lambda$ within 1 SE of the minimum CV error. Encourages sparsity / simpler models with only marginal loss.

Similar reasoning applies to ridge (using GCV for efficiency), elastic net (2D grid), and other penalty-path models.

### Q33: Does CV give you uncertainty estimates for model predictions?

**Answer**: Not directly. CV gives you an estimate of mean performance across (random training set, test point). It tells you how well the procedure performs on average, but not how uncertain a specific test prediction is.

For prediction uncertainty, you need additional methods:
- **Bootstrap**: fit $B$ models on bootstrap samples, look at prediction distribution for each test point.
- **Bayesian methods**: posterior predictive distribution directly gives uncertainty.
- **Conformal prediction**: uses CV residuals to construct prediction intervals with finite-sample coverage guarantees, without distributional assumptions.
- **Ensemble variance**: in random forests or MC dropout, variance across ensemble members.

Conformal prediction is particularly nice: it uses held-out residuals (from CV or a calibration set) to construct intervals $[\hat{y} - q_\alpha, \hat{y} + q_\alpha]$ with coverage $\geq 1 - \alpha$ on exchangeable data. This connects CV infrastructure directly to calibrated uncertainty.

### Q34: Why not always use nested CV?

**Answer**: Nested CV is unbiased but expensive. Reasons not to use it:

1. **Large datasets**: a single train/val/test split is equally unbiased and much cheaper. With millions of examples, the val and test sets are large enough for precise estimates.

2. **Expensive models**: deep learning fits can take days. Nested CV can multiply this by 25x or more — infeasible.

3. **No hyperparameter tuning**: if your procedure has no tuning, standard CV is unbiased.

4. **Simple tuning with external validation**: if your tuning process is tightly controlled (e.g., select between 2 obvious candidates), the selection bias is small.

5. **Prior experience**: when you've already validated the hyperparameter choices in prior experiments, you don't need to retune + evaluate from scratch.

Trade-off: nested CV is the rigorous choice for reporting. For iterative development, cheaper proxies (single-loop CV) are acceptable, with a final nested run or hold-out before publication / deployment.

### Q35: How do you handle CV with imputation of missing values?

**Answer**: Imputation must happen inside each CV fold.

**Wrong**:
```python
X_imputed = SimpleImputer(strategy='mean').fit_transform(X)  # uses all data
cross_val_score(model, X_imputed, y, cv=5)
```

The imputer's "mean" uses data from all folds, including test folds. Minor leakage, but real.

**Right**:
```python
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LogisticRegression())
])
cross_val_score(pipeline, X, y, cv=5)
```

Same logic for more complex imputation (KNN imputation, iterative imputation, model-based imputation) — all fit inside the fold.

Subtleties:
- If missingness is informative (missing-at-random vs. missing-not-at-random), the missing pattern itself is a feature. Include a binary indicator.
- Test-time imputation may encounter new missing patterns. Make sure the imputer gracefully handles this.
- For time-series, imputation should use only past observations, not future.

### Q36: How would you use CV to estimate a confidence interval for a metric like AUC?

**Answer**: Several approaches:

1. **Fold-level variance** (simple but flawed): compute AUC per fold, compute SD across folds, form normal-interval CI as mean ± $z_{\alpha/2} \cdot$ SD. Flaw: fold AUCs are correlated (overlapping training sets), so SD underestimates the true SE.

2. **Nadeau-Bengio corrected t-test**: accounts for correlation in fold-level statistics. Replace the naive variance with $(1/k + \frac{k-1}{k} \cdot \frac{n_{\text{test}}}{n_{\text{train}}}) \cdot s^2$ where $s^2$ is the sample variance of fold scores. This inflates the SE appropriately.

3. **Bootstrap-of-CV**: for each bootstrap sample of the data, run CV and compute the mean AUC. The distribution of these bootstrap-CV estimates approximates the sampling distribution of the CV estimator. Expensive but correct.

4. **DeLong's method**: specifically for AUC, gives an analytical variance formula based on the pooled predictions from all folds. Use pooled AUC and DeLong CI.

5. **Conformal / jackknife-style**: use held-out residuals to construct intervals with finite-sample coverage.

For practical reporting: fold-level mean plus SD is the most common, with the understanding that it's optimistic. For rigorous CIs, DeLong (for AUC) or bootstrap.

### Q37: If you have severe class imbalance, should you balance the classes before or after CV?

**Answer**: After splitting — always inside the CV fold, on the training portion only.

**Wrong**:
```python
X_balanced, y_balanced = SMOTE().fit_resample(X, y)
cross_val_score(model, X_balanced, y_balanced, cv=5)
```
Test folds now contain synthetic examples similar to training ones → leakage.

**Right** (using imbalanced-learn pipeline):
```python
from imblearn.pipeline import Pipeline as ImbPipeline
pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('model', LogisticRegression())
])
cross_val_score(pipeline, X, y, cv=5)
```
Now SMOTE is applied only to the training portion of each fold. Test sets remain natural imbalanced samples.

Additional considerations:
- The stratified k-fold should split the *original* imbalanced labels.
- If you oversample the training set, evaluation metrics should still reflect the original class distribution — use metrics like AUC, AUPRC, F1, not raw accuracy.
- Some evaluations (especially precision-recall curves) should be computed on the test set with its natural imbalance.

### Q38: What are some alternatives to CV that you might consider?

**Answer**: 

1. **Hold-out test set**: simplest, works for large datasets.
2. **Bootstrap (.632, .632+)**: resampling with replacement. Alternative to k-fold for small data.
3. **Cross-validated prediction** (for estimating Y given X): use each fold's predictions to form a full dataset of CV-predictions, then compute aggregate metrics.
4. **Bayesian model selection**: Bayes factors, marginal likelihood — principled but computationally hard for many models.
5. **Information criteria**: AIC, BIC, DIC, WAIC — cheap, but require likelihood structure.
6. **Cross-validated likelihood for hierarchical models**: specialized extensions for random effects.
7. **Out-of-bag (OOB) error**: natural for bagging algorithms like random forests.
8. **Pseudo-test**: build a synthetic "test" period from the end of your data (time series) or random rows (iid).
9. **Theoretical bounds**: VC-dimension bounds, Rademacher complexity — rigorous but usually loose.

CV remains the workhorse because it's model-agnostic, conceptually simple, and widely supported. The alternatives shine in specific niches.

### Q39: A senior colleague argues that CV is unnecessary for deep learning because "you just need a big enough validation set." How do you respond?

**Answer**: They have a point but it's incomplete.

**Where they're right**:
- Training a deep learning model once takes hours to days. Running k-fold with k=5 multiplies that by 5, which is often impractical.
- For very large datasets (ImageNet-scale), a single validation set of 5-10% is already large enough for precise estimates.
- Standard practice in DL is single train/val/test split, and this works for most benchmarks.

**Where I'd push back**:
- For smaller datasets (say, $<$ 100k examples), a single validation set has high variance, and model selection decisions are noisy. Even DL can benefit from CV or at least multi-seed training.
- For scientific reporting or high-stakes applications (medical, legal), a single hold-out is insufficient; some form of CV is needed for confidence intervals.
- For DL on tabular or small medical datasets, k-fold is absolutely applicable and the variance reduction is valuable.
- For hyperparameter tuning, selection bias is real in DL — performance on the validation set is biased by the number of configurations tried.

The compromise: **train with multiple random seeds** (3-5) to estimate the variance due to initialization. This is cheaper than full CV and addresses one important source of variance (non-determinism of training). Combine with a single validation set for most decisions and a held-out test set touched once for final reporting.

### Q40: Summarize when you would use each type of CV.

**Answer**: A decision tree of sorts:

1. **Is the data time-ordered, and does deployment require predicting the future?**
   - Yes → **Time-series CV** (expanding or rolling window).
   - Do labels span future horizons? → Add purging/embargoing.

2. **Do examples cluster into groups, and will deployment see new groups?**
   - Yes → **Group k-fold** or **LOGO**.
   - With classification imbalance? → **Stratified group k-fold**.
   - Combine with time-series if both matter.

3. **Is there spatial structure, and will deployment be in new regions?**
   - Yes → **Spatial block CV** with buffer.
   - Combine with environmental CV for even stricter tests.

4. **Is the data iid (no time, groups, space), and moderate-sized?**
   - Classification → **Stratified k-fold** (k = 5 or 10).
   - Regression → **k-fold** (k = 5 or 10).
   - Small data with cheap model → **LOOCV** or **repeated k-fold**.

5. **Is the data iid, large (>1M rows), and compute-limited?**
   - Train/val/test **hold-out**.

6. **Am I tuning hyperparameters and need to report honest performance?**
   - Yes → **Nested CV** (any of the above, nested).
   - Or: hold out a final test set.

7. **Do I need prediction intervals / uncertainty?**
   - CV gives expected performance, not prediction uncertainty. Use bootstrap, conformal, or Bayesian methods.

The meta-rule: **the CV scheme must match the deployment scenario**. Think carefully about what "new" means at deployment (new data point, new entity, new time, new region) and design CV to mirror that.

---

## Final Remarks

Cross-validation is one of those topics that seems simple on the surface but hides enormous subtlety. The basic mechanics (k-fold, LOOCV) can be taught in ten minutes, but the correct application across domains — time series, groups, spatial data, nested tuning — requires understanding the assumptions each method makes and where they break.

The unifying principle across all of this: **cross-validation estimates the performance of a procedure on a distribution that resembles the training data**. Everything else — stratification, grouping, time-awareness, spatial blocking — is about ensuring that the "distribution" on which you evaluate actually matches the deployment distribution you care about.

If you internalize that principle, you will design CV schemes that give honest answers. If you forget it, you will deploy models that look great offline and disappoint in production. The gap between a 95% CV score and a 75% production score is almost always explainable by a mismatch between what CV measured and what production requires.

The interview question "design a CV scheme for X" is really asking: "do you understand what CV is measuring, and can you adapt it to the real-world structure of the problem?" Every example in this guide — medical imaging, fraud, forecasting, ecology, recommendations — is an instance of that same exercise.

---

**[← Previous Chapter: Metrics](ml_metrics_guide.md) | [Table of Contents](index.md) | [Next Chapter: Statistical Comparison →](statistical_comparison_guide.md)**
