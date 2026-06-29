# k-Nearest Neighbors: A Comprehensive Learning Guide

> A standalone reference covering k-NN, distance metrics, algorithmic variants, efficient data structures, the curse of dimensionality, and parameter selection — with mathematical rigor and machine-learning-focused interview preparation.

---

## Table of Contents

1. [k-Nearest Neighbors (Core Algorithm)](#topic-1-k-nearest-neighbors)
2. [Distance Metrics](#topic-2-distance-metrics)
3. [Algorithm Variations (Weighted kNN, Radius-Based Neighbors)](#topic-3-algorithm-variations)
4. [Efficiency Structures (kd-trees, Ball Trees, LSH)](#topic-4-efficiency-structures)
5. [Curse of Dimensionality](#topic-5-curse-of-dimensionality)
6. [Parameter Selection](#topic-6-parameter-selection)
7. [Interview Preparation — Master Set](#interview-preparation)

---

# Topic 1: k-Nearest Neighbors

## 1.1 Motivation & Intuition

### Why does kNN exist?

Imagine you are handed a basket of fruits and asked to classify a new piece of fruit as an apple or an orange. You have no formula for "appleness"; you have only examples. The most natural human strategy is: *look at the fruits most similar to this one and vote*. If four of the five most similar fruits are apples, you call it an apple.

This is exactly k-Nearest Neighbors (kNN). It is the machine-learning formalization of the ancient heuristic *"judge something by the company it keeps."*

kNN exists because many problems resist tidy parametric modeling:

- **Decision boundaries are irregular.** A linear model carves the input space with a hyperplane; logistic regression assumes log-odds are linear in the features. But real boundaries — between handwritten 4s and 9s, between benign and malignant cells in feature space — can be wiggly, disconnected, or even fractal. kNN lets the *data itself* shape the boundary.
- **We have no model, only examples.** In many early-stage ML problems, you have a dataset long before you have a hypothesis about the generative process. kNN gives a strong baseline with essentially zero modeling commitment.
- **The function we want to learn is arbitrarily complex but locally smooth.** The assumption "nearby inputs have similar outputs" is extremely weak and holds remarkably often.

### A concrete example before the math

Suppose we want to predict whether a customer will churn. We have 10,000 past customers described by *(monthly_spend, tenure_months, support_tickets)* and a label *churned / stayed*. A new customer arrives with features $(45, 3, 7)$. The kNN procedure:

1. Compute the distance from $(45, 3, 7)$ to each of the 10,000 past customers.
2. Pick the $k = 5$ closest.
3. If 4 of them churned, predict *churn*.

No training. No coefficients. No assumptions about linearity. Just memory and a distance function.

### Connection to real ML systems

kNN or its close cousins appear in many modern systems, often in disguise:

- **Recommendation systems** (Spotify, Netflix, Amazon): given a user's taste embedding, retrieve the $k$ most similar items. This is kNN in a learned embedding space.
- **Retrieval-Augmented Generation (RAG):** given a query embedding, retrieve the top-$k$ most relevant documents. This is kNN over a vector database.
- **Anomaly detection:** points whose nearest neighbors are unusually far away are candidates for anomalies (LOF, kNN distance-based methods).
- **Semantic search, face recognition, plagiarism detection, deduplication:** all use nearest-neighbor lookups over learned or hand-crafted feature vectors.
- **Baselines for benchmarking:** before committing to a deep model, practitioners often check what a 1-NN or k-NN baseline achieves on raw or pretrained embeddings. If kNN beats your fancy model, something is wrong with your fancy model.

### The underlying intuition in one sentence

> *The label of a point is well-approximated by the labels of its neighbors, provided the feature space is arranged so that closeness corresponds to similarity in the target.*

Everything else — distance metrics, $k$ selection, indexing, the curse of dimensionality — is about making this sentence true and computationally tractable.

---

## 1.2 Conceptual Foundations

### Key terms

- **Instance / sample / example:** a labeled data point $(\mathbf{x}_i, y_i)$ with features $\mathbf{x}_i \in \mathbb{R}^d$ and label $y_i$.
- **Training set:** the reference set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$. In kNN the "training set" is literally stored — there are no parameters to fit.
- **Query point:** a new point $\mathbf{x}_q$ whose label we want to predict.
- **Distance (or dissimilarity) function:** a function $d : \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}_{\ge 0}$ that measures how far apart two points are.
- **Neighborhood $N_k(\mathbf{x}_q)$:** the set of $k$ points in $\mathcal{D}$ with the smallest $d(\mathbf{x}_i, \mathbf{x}_q)$.
- **Hyperparameter $k$:** the number of neighbors considered. Not learned from data; chosen via cross-validation.
- **Lazy learner / instance-based / non-parametric:** kNN does no training work; all computation is deferred to query time. It stores instances rather than learning parameters.
- **Voting rule (classification):** majority vote among neighbors, possibly weighted.
- **Averaging rule (regression):** mean (or weighted mean) of neighbors' targets.

### How the components interact, step by step

At query time, for a new point $\mathbf{x}_q$:

1. **Compute distances.** For every $\mathbf{x}_i$ in the training set, compute $d(\mathbf{x}_q, \mathbf{x}_i)$.
2. **Sort or partially sort.** Identify the indices of the $k$ smallest distances.
3. **Aggregate labels.**
   - *Classification:* $\hat{y}_q = \arg\max_c \sum_{i \in N_k(\mathbf{x}_q)} \mathbb{1}[y_i = c]$ (plurality vote).
   - *Regression:* $\hat{y}_q = \frac{1}{k} \sum_{i \in N_k(\mathbf{x}_q)} y_i$ (arithmetic mean).
4. **Return prediction.**

That is the entire algorithm.

### Underlying assumptions (and what breaks when they fail)

kNN looks assumption-free but quietly depends on several things:

**A1. Local smoothness / continuity of the target function.**
Points close in feature space have similar labels. If the target is highly non-smooth (e.g., a parity function $y = \oplus_i x_i$ on binary features), kNN fails catastrophically because nearby points in Hamming distance can have opposite labels.

**A2. The distance metric reflects the notion of similarity relevant to the task.**
If we measure distance in raw units, a feature measured in millimeters dominates one measured in kilometers purely by unit choice. The metric is an *implicit model*; a bad metric is indistinguishable from a bad hypothesis.

**A3. Features are on comparable scales.**
Closely tied to A2, but worth stating separately because it is the single most common mistake practitioners make. *Always* standardize or normalize before using Euclidean kNN.

**A4. The training distribution covers the query region.**
kNN interpolates within the convex hull of the training data; it does not extrapolate gracefully. Queries in regions with no nearby training points return predictions driven by arbitrary far-away neighbors.

**A5. Moderate dimensionality.**
In very high dimensions, distances concentrate (all points become nearly equidistant), making "nearest" meaningless. This is the curse of dimensionality (Topic 5).

**A6. Classes are roughly balanced (for majority vote).**
If class $A$ outnumbers class $B$ by 1000:1, random regions of space will be dominated by $A$ even where $B$ truly rules. Fixes: class-weighting, resampling, distance weighting.

**A7. No substantial label noise in local neighborhoods.**
1-NN is especially fragile: a single mislabeled neighbor flips the prediction. Larger $k$ provides noise averaging but trades away resolution.

### The bias–variance picture

- **Small $k$ (e.g., $k=1$):** low bias, high variance. The decision boundary can be arbitrarily complex; it tracks individual points, including noise.
- **Large $k$:** high bias, low variance. The boundary smooths out; in the limit $k = n$, every query returns the global majority class.
- **Sweet spot:** chosen via cross-validation (see Topic 6).

---

## 1.3 Mathematical Formulation

### Notation

Let:
- $\mathcal{X} \subseteq \mathbb{R}^d$ be the feature space.
- $\mathcal{Y}$ be the label space ($\{1, \dots, C\}$ for classification, $\mathbb{R}$ for regression).
- $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$ be the training set.
- $d(\cdot, \cdot) : \mathcal{X} \times \mathcal{X} \to \mathbb{R}_{\ge 0}$ be a distance.
- $\pi : \{1, \dots, n\} \to \{1, \dots, n\}$ be a permutation such that

$$
d(\mathbf{x}_q, \mathbf{x}_{\pi(1)}) \le d(\mathbf{x}_q, \mathbf{x}_{\pi(2)}) \le \cdots \le d(\mathbf{x}_q, \mathbf{x}_{\pi(n)}).
$$

Then $N_k(\mathbf{x}_q) = \{\pi(1), \dots, \pi(k)\}$.

### Classification

The plurality-vote prediction is:

$$
\hat{y}_q \;=\; \arg\max_{c \in \{1,\dots,C\}} \; \sum_{i \in N_k(\mathbf{x}_q)} \mathbb{1}[y_i = c].
$$

Equivalently, the posterior estimate is

$$
\hat{p}(y = c \mid \mathbf{x}_q) \;=\; \frac{1}{k}\sum_{i \in N_k(\mathbf{x}_q)} \mathbb{1}[y_i = c],
$$

and $\hat{y}_q = \arg\max_c \hat{p}(y = c \mid \mathbf{x}_q)$.

### Regression

$$
\hat{y}_q \;=\; \frac{1}{k}\sum_{i \in N_k(\mathbf{x}_q)} y_i.
$$

For weighted variants (Topic 3), each neighbor carries weight $w_i = f(d(\mathbf{x}_q, \mathbf{x}_i))$, and

$$
\hat{y}_q \;=\; \frac{\sum_{i \in N_k} w_i\, y_i}{\sum_{i \in N_k} w_i}.
$$

### Why kNN estimates the true conditional distribution (Cover & Hart, 1967)

A famous result motivates kNN theoretically. Let $\eta(\mathbf{x}) = P(Y = 1 \mid X = \mathbf{x})$ for binary classification, and let the Bayes error rate be $R^* = \mathbb{E}[\min(\eta(X), 1-\eta(X))]$.

**Cover–Hart theorem (1-NN asymptotic risk).** As $n \to \infty$, the 1-NN classifier's expected error $R_{\text{1NN}}$ satisfies

$$
R^* \;\le\; R_{\text{1NN}} \;\le\; 2 R^* (1 - R^*).
$$

The interpretation is striking: in the infinite-data limit, **1-NN is at most twice as bad as the optimal Bayes classifier** — despite having no model at all. The intuition: as $n \to \infty$, the nearest neighbor's feature vector converges to $\mathbf{x}_q$, and its label is drawn from $\eta(\mathbf{x}_q)$. Asymptotically, 1-NN produces a sample from the true conditional, which is within a factor of 2 of the Bayes-optimal rule (which always picks the mode).

**For general $k$,** as $n \to \infty$ and $k \to \infty$ with $k/n \to 0$, the $k$-NN error converges to $R^*$ — kNN is **consistent** (Stone, 1977). The constraints $k \to \infty$ (enough neighbors to average noise) and $k/n \to 0$ (neighbors remain local) are both necessary.

### Derivation: Why standardization matters

Consider Euclidean distance on two features with very different scales:

$$
d(\mathbf{x}, \mathbf{x}')^2 \;=\; (x_1 - x_1')^2 + (x_2 - x_2')^2.
$$

Suppose $x_1 \in [0, 1000]$ (income in dollars) and $x_2 \in [0, 1]$ (a probability). Differences in $x_1$ are typically $O(100)$, while differences in $x_2$ are $O(0.1)$. Therefore

$$
(x_1 - x_1')^2 \gg (x_2 - x_2')^2
$$

almost always, and the distance is effectively $|x_1 - x_1'|$. The probability feature is ignored.

**Fix:** standardize each feature to zero mean and unit variance:

$$
\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}.
$$

Now each feature contributes on comparable scales. More principled still: use the Mahalanobis metric (Topic 2), which learns the right scaling from data.

### Computational complexity

- **Training:** $O(1)$ in time, $O(nd)$ in memory (store the training set).
- **Query (naive):** $O(nd)$ per query — compute all $n$ distances, then $O(n)$ or $O(n \log k)$ to find the top-$k$.
- **Query (with indexing):** $O(\log n)$ amortized in low dimensions using kd-trees or ball trees; degrades to $O(n)$ in high dimensions.
- **Total query time for $m$ test points:** $O(mn)$ naive — often the bottleneck.

---

## 1.4 Worked Examples

### Example 1: Classification by hand

Suppose we have 6 labeled 2D points:

| i | $x_1$ | $x_2$ | $y$ |
|---|------|------|---|
| 1 | 1.0  | 1.0  | A |
| 2 | 1.5  | 2.0  | A |
| 3 | 3.0  | 4.0  | B |
| 4 | 5.0  | 7.0  | B |
| 5 | 3.5  | 5.0  | B |
| 6 | 2.0  | 2.5  | A |

Query: $\mathbf{x}_q = (2.5, 3.0)$. Use $k=3$ with Euclidean distance.

**Step 1. Compute distances.**

$$
\begin{aligned}
d_1 &= \sqrt{(1.0-2.5)^2 + (1.0-3.0)^2} = \sqrt{2.25 + 4.00} = \sqrt{6.25} = 2.500 \\
d_2 &= \sqrt{(1.5-2.5)^2 + (2.0-3.0)^2} = \sqrt{1.00 + 1.00} = \sqrt{2.00} \approx 1.414 \\
d_3 &= \sqrt{(3.0-2.5)^2 + (4.0-3.0)^2} = \sqrt{0.25 + 1.00} = \sqrt{1.25} \approx 1.118 \\
d_4 &= \sqrt{(5.0-2.5)^2 + (7.0-3.0)^2} = \sqrt{6.25 + 16.00} = \sqrt{22.25} \approx 4.717 \\
d_5 &= \sqrt{(3.5-2.5)^2 + (5.0-3.0)^2} = \sqrt{1.00 + 4.00} = \sqrt{5.00} \approx 2.236 \\
d_6 &= \sqrt{(2.0-2.5)^2 + (2.5-3.0)^2} = \sqrt{0.25 + 0.25} = \sqrt{0.50} \approx 0.707
\end{aligned}
$$

**Step 2. Sort ascending.**
$d_6 \approx 0.707$ (A), $d_3 \approx 1.118$ (B), $d_2 \approx 1.414$ (A), $d_5 \approx 2.236$ (B), $d_1 = 2.500$ (A), $d_4 \approx 4.717$ (B).

**Step 3. Take top $k=3$.** Neighbors: $\{6, 3, 2\}$ with labels $\{A, B, A\}$.

**Step 4. Vote.** A: 2 votes, B: 1 vote. **Predict A.**

Note how sensitive this is. With $k=1$, the sole neighbor is point 6 (A) — same answer. With $k=5$, neighbors are $\{6, 3, 2, 5, 1\} = \{A, B, A, B, A\}$ — A wins 3–2. With $k=4$, it is 2–2, a tie that must be broken (e.g., pick the class of the closest tied neighbor, which is A).

### Example 2: Regression with distance weighting

Same 6 points, but now $y$ is a continuous target:

| i | $y$ |
|---|-----|
| 1 | 10  |
| 2 | 12  |
| 3 | 30  |
| 4 | 70  |
| 5 | 40  |
| 6 | 11  |

Using $k=3$ unweighted:

$$
\hat{y}_q = \frac{y_6 + y_3 + y_2}{3} = \frac{11 + 30 + 12}{3} = \frac{53}{3} \approx 17.67.
$$

Using inverse-distance weighting, $w_i = 1/d_i$:

$$
\begin{aligned}
w_6 &= 1/0.707 \approx 1.414 \\
w_3 &= 1/1.118 \approx 0.894 \\
w_2 &= 1/1.414 \approx 0.707
\end{aligned}
$$

$$
\hat{y}_q = \frac{1.414 \cdot 11 + 0.894 \cdot 30 + 0.707 \cdot 12}{1.414 + 0.894 + 0.707} = \frac{15.55 + 26.82 + 8.48}{3.015} = \frac{50.85}{3.015} \approx 16.87.
$$

The weighted estimate pulls toward the nearest neighbor (point 6, $y=11$), as we would expect.

### Example 3: Why scaling matters (numeric demonstration)

Suppose:
- Point A: (income = 50000, credit_score = 700)
- Point B: (income = 51000, credit_score = 300)
- Query:    (income = 50500, credit_score = 700)

Without scaling, $d(q, A) = \sqrt{500^2 + 0^2} = 500$ and $d(q, B) = \sqrt{500^2 + 400^2} \approx 640.3$. A is closer. Fine.

But if income were scaled up by accidentally recording in cents rather than dollars, A's income coordinate is 5,000,000 and the income axis dwarfs credit_score entirely — the query's credit-score identity with A becomes invisible next to any income fluctuation. Conversely, if we standardize so both features have zero mean and unit variance, the two features contribute on equal footing and the credit-score match between A and the query is properly rewarded.

---

## 1.5 Relevance to Machine Learning Practice

### When to use kNN

- **Small-to-medium datasets** (say, $n < 10^6$) with moderate dimensionality ($d < 50$ raw; $d$ up to a few hundred with good embeddings).
- **Strong baseline** before committing to a complex model. If kNN with good features is competitive, it is a warning that either (a) the signal is simple, or (b) your complex model is under-trained.
- **Interpretability of a specific prediction:** "We flagged this customer because she looks like these 5 similar ones who churned." Nearest neighbors are inherently explanatory.
- **Multi-class problems with many classes** (e.g., face recognition with thousands of identities) where parametric models struggle but distance-based lookups excel.
- **Streaming / online updates:** adding new data is $O(1)$ — just append.
- **Hybrid systems:** kNN over learned embeddings is state-of-the-art in many retrieval problems.

### When NOT to use kNN

- **Very high raw dimensionality** (e.g., 10,000-dim bag-of-words). Distances lose meaning; use dimensionality reduction or learned embeddings first.
- **Very large datasets with strict latency requirements** unless you have strong indexing (FAISS, ScaNN, HNSW).
- **Sparse or heterogeneous features** where Euclidean distance does not reflect similarity.
- **Extrapolation outside the training distribution:** kNN cannot sensibly predict for regions with no training data.
- **Highly imbalanced classes** without careful weighting.
- **Noisy labels** with small $k$.

### Common alternatives

| Need | Alternative |
|---|---|
| Smooth decision boundary, interpretable coefficients | Logistic regression, linear SVM |
| Non-linear boundary, medium data | Kernel SVM, random forests, gradient boosting |
| Huge data, structured features | Neural networks |
| Density estimation, probabilistic predictions | Kernel density estimation (KDE) |
| Very fast queries | Parametric models (amortize cost at training) |
| Anomaly detection | Isolation Forest, LOF, One-class SVM |

### Trade-offs

- **Bias–variance.** Controlled by $k$. Small $k$: low bias, high variance. Large $k$: high bias, low variance.
- **Interpretability.** Very interpretable *per prediction* (which neighbors drove it) but offers no global "model" — no coefficients, no feature importance in the usual sense.
- **Robustness.** Poor to label noise at small $k$; improved by larger $k$ or weighted voting.
- **Computational cost.** Training: trivial. Inference: expensive without indexing. Memory: stores the entire training set — unacceptable for truly large datasets.
- **Feature engineering burden.** kNN is exquisitely sensitive to feature choice and scaling. Getting the metric right is most of the work.

### Where kNN hides in modern systems

- **FAISS / ScaNN / HNSW / Annoy / Milvus / Pinecone / Weaviate:** all are approximate nearest neighbor (ANN) indexes — the engines behind vector search.
- **Contrastive learning and self-supervised representation learning:** evaluation is often done via a kNN classifier on top of frozen features (the "linear probe" of the non-parametric world).
- **RAG pipelines:** retrieve top-$k$ chunks by embedding similarity.
- **Duplicate detection, entity resolution, record linking:** all kNN over hand-crafted or learned similarity.

---

## 1.6 Common Pitfalls & Misconceptions

1. **Not scaling features.** The single most common mistake. Always standardize (z-score) or min-max scale numeric features before Euclidean kNN.

2. **Using raw categorical features with Euclidean distance.** The distance between "red" and "blue" encoded as 2 and 3 is not meaningfully 1. Use one-hot + appropriate metric (Hamming / cosine) or a learned embedding.

3. **Ignoring the curse of dimensionality.** Running kNN on raw 1024-dim TF-IDF vectors and wondering why every point is equidistant from every other point.

4. **Choosing $k$ by gut.** $k$ is a hyperparameter; use cross-validation (Topic 6).

5. **Confusing "lazy" with "fast."** kNN has no training cost but O(n) query cost per point — often far slower than a fitted parametric model at inference.

6. **Treating kNN as assumption-free.** It has strong implicit assumptions: scale, metric, class balance, local smoothness.

7. **Using 1-NN in noisy settings.** One mislabeled neighbor causes misclassification. Prefer $k \ge 5$ unless the data is unusually clean.

8. **Ignoring ties.** With an even $k$ in binary classification, ties are possible. Deterministic tie-breaking (e.g., by class of closest tied neighbor, or by overall class frequency) matters for reproducibility.

9. **Leaking the query into training.** When evaluating on the training set directly, 1-NN returns 0 error (every point is its own neighbor). Always use leave-one-out or a proper held-out split.

10. **Forgetting to match distance and task.** Cosine distance is standard for text embeddings; Euclidean is not equivalent unless vectors are L2-normalized (in which case they agree up to a monotone transform — a subtlety discussed in Topic 2).

---

# Topic 2: Distance Metrics

## 2.1 Motivation & Intuition

### Why do distance metrics matter?

kNN is only as good as its notion of "close." The distance function is the **hypothesis class** of kNN — it is what implicitly encodes which features matter, how they combine, and what counts as similarity. Two identical datasets with two different distance functions yield two completely different classifiers.

Consider a trivial example. You want to cluster cities by similarity for a logistics problem.
- Measured by **Euclidean distance** (straight-line), New York and Los Angeles are far.
- Measured by **graph distance on the road network**, they are farther, but now the geography of interstates matters.
- Measured by **flight time**, they are *closer* than New York and a small town 300 miles away without an airport.

Each metric is correct for a different question. Distance is not a universal property of data; it is a modeling choice.

### Real ML motivations

- **Tabular data with mixed units (income, age, probability):** Euclidean distance without scaling is meaningless; Mahalanobis or standardized Euclidean is appropriate.
- **Categorical / binary data (DNA sequences, user clicks):** Hamming distance counts mismatches directly.
- **Text embeddings / document similarity:** cosine similarity handles magnitude invariance (long documents vs short ones).
- **Correlated features (spectroscopy, financial time series):** Mahalanobis accounts for correlation.
- **Manhattan (L1):** robust to outliers in individual coordinates; natural for grid-like data.

---

## 2.2 Conceptual Foundations

### What makes a "metric"?

A function $d : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a **metric** if, for all $\mathbf{x}, \mathbf{y}, \mathbf{z} \in \mathcal{X}$:

1. **Non-negativity:** $d(\mathbf{x}, \mathbf{y}) \ge 0$.
2. **Identity of indiscernibles:** $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$.
3. **Symmetry:** $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$.
4. **Triangle inequality:** $d(\mathbf{x}, \mathbf{z}) \le d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$.

If (2) is relaxed to $d(\mathbf{x}, \mathbf{x}) = 0$ (without the converse), $d$ is a **pseudometric**. If (4) is dropped, it is a **semi-metric**. *Cosine distance* $1 - \cos(\theta)$ is a semi-metric but not a metric (fails triangle inequality). *Squared Euclidean* is not a metric for the same reason.

Why does this matter? Many efficient data structures (kd-trees, ball trees, VP-trees) rely on the triangle inequality to prune searches. If your distance is not a metric, those structures may return incorrect nearest neighbors.

### The core family

We will cover five widely used distances:

1. **Euclidean (L2).** Standard geometric distance.
2. **Manhattan (L1, taxicab).** Sum of absolute differences.
3. **Minkowski (Lp).** Generalizes both.
4. **Mahalanobis.** Scale- and correlation-aware Euclidean.
5. **Hamming.** Number of positions where two vectors differ.

We will also touch on **cosine** and **Chebyshev (L∞)** where relevant.

### Assumptions and failure modes

Each metric encodes assumptions:
- **Euclidean** assumes isotropic, uncorrelated, equally-scaled features.
- **Manhattan** assumes features contribute additively and linearly, and is less sensitive to large single-coordinate differences.
- **Mahalanobis** assumes a (learned or given) covariance structure that captures the relevant scale and correlations.
- **Hamming** assumes features are discrete / binary and positionally independent.
- **Cosine** assumes magnitude is irrelevant and direction carries all the information (natural for normalized embeddings).

When assumptions are violated:
- Using Euclidean on features of wildly different scales → dominant feature swallows everything.
- Using Hamming on continuous features → nearly all pairs differ in every coordinate; the metric is useless.
- Using cosine on non-normalized, sign-meaningful features (e.g., signed returns) → direction cannot capture magnitude information.

---

## 2.3 Mathematical Formulation

### Euclidean (L2)

$$
d_2(\mathbf{x}, \mathbf{y}) \;=\; \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2} \;=\; \|\mathbf{x} - \mathbf{y}\|_2.
$$

Equivalent to the $\ell^2$ norm of the difference. Isotropic: rotating the coordinate system leaves distances unchanged.

*Squared Euclidean* $d_2^2$ is often used for efficiency (avoids the square root) and gives identical neighbor orderings, but is not a true metric.

### Manhattan (L1)

$$
d_1(\mathbf{x}, \mathbf{y}) \;=\; \sum_{j=1}^{d} |x_j - y_j| \;=\; \|\mathbf{x} - \mathbf{y}\|_1.
$$

Named after the grid-like street layout of Manhattan — to go from one intersection to another, you traverse city blocks north-south and east-west, never diagonally.

Level sets of L1 around a point are diamonds (rotated squares); level sets of L2 are circles. A unit L1 ball is properly *contained* in the unit L2 ball — which implies $d_1 \ge d_2$ always, with equality iff the vectors differ in only one coordinate.

### Minkowski (Lp)

$$
d_p(\mathbf{x}, \mathbf{y}) \;=\; \left(\sum_{j=1}^{d} |x_j - y_j|^p\right)^{1/p}, \quad p \ge 1.
$$

- $p = 1$: Manhattan.
- $p = 2$: Euclidean.
- $p \to \infty$: **Chebyshev** distance, $d_\infty(\mathbf{x}, \mathbf{y}) = \max_j |x_j - y_j|$.

**Proof sketch for $p \to \infty$.** Let $M = \max_j |x_j - y_j|$. Then

$$
d_p = \left(\sum_j |x_j - y_j|^p\right)^{1/p} = M \left(\sum_j \left(\frac{|x_j - y_j|}{M}\right)^p\right)^{1/p}.
$$

Each ratio is in $[0,1]$ and at least one equals 1. As $p \to \infty$, terms with ratio $< 1$ vanish, so the sum approaches the number of coordinates achieving the max (finite), and the $p$-th root of that approaches 1. Hence $d_p \to M$.

**When is Minkowski a metric?** Only for $p \ge 1$. For $0 < p < 1$, $d_p$ fails the triangle inequality. These *fractional Lp* "distances" are sometimes used experimentally in high dimensions (see Aggarwal, Hinneburg, Keim, 2001) because they can preserve neighbor rank better than L2 when $d$ is huge — but they are not true metrics and not supported by standard tree-based indexes.

### Mahalanobis

Let $\Sigma$ be a symmetric positive-definite $d \times d$ matrix (typically the sample covariance of the data, or a learned matrix). The Mahalanobis distance is

$$
d_M(\mathbf{x}, \mathbf{y}) \;=\; \sqrt{(\mathbf{x} - \mathbf{y})^\top \Sigma^{-1} (\mathbf{x} - \mathbf{y})}.
$$

**Intuition.** Consider the whitening transform $\mathbf{z} = \Sigma^{-1/2} \mathbf{x}$. Then

$$
d_M(\mathbf{x}, \mathbf{y}) = \|\Sigma^{-1/2}(\mathbf{x} - \mathbf{y})\|_2 = \|\mathbf{z} - \mathbf{z}'\|_2.
$$

Mahalanobis is **Euclidean distance after whitening**. It rescales each principal direction by the inverse standard deviation and de-correlates features.

**Special cases:**
- $\Sigma = I$: Mahalanobis = Euclidean.
- $\Sigma = \text{diag}(\sigma_1^2, \dots, \sigma_d^2)$: Mahalanobis = standardized Euclidean with per-feature standardization.
- General $\Sigma$: captures feature correlations.

**Why it's useful.** Imagine two features that are highly positively correlated (e.g., height in cm and height in inches measured with slight noise). Naive Euclidean distance treats them as independent and double-counts variation along their common axis. Mahalanobis with the empirical $\Sigma$ accounts for this correlation, reducing the effective weight along redundant directions.

**Learned Mahalanobis.** Metric learning methods (LMNN, NCA, ITML) learn $\Sigma^{-1}$ (equivalently, a linear projection $L$ such that $\Sigma^{-1} = L^\top L$) to maximize discriminative power for a given labeled task.

### Hamming distance

For vectors $\mathbf{x}, \mathbf{y}$ over a finite alphabet (binary, categorical, or string):

$$
d_H(\mathbf{x}, \mathbf{y}) \;=\; \sum_{j=1}^{d} \mathbb{1}[x_j \neq y_j].
$$

Counts the number of positions that differ. Widely used in:
- Error-correcting codes.
- DNA / protein sequences.
- Binary hashing (LSH, see Topic 4).
- Comparing categorical records.

On binary vectors, Hamming equals the squared Euclidean distance, and equals $L_1$ distance. (Check: differing bits contribute 1 to each.)

**Normalized Hamming:** $d_H / d$, giving a number in $[0,1]$.

### Cosine similarity and distance

For nonzero vectors $\mathbf{x}, \mathbf{y}$:

$$
\cos(\theta) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}, \qquad d_{\cos}(\mathbf{x}, \mathbf{y}) = 1 - \cos(\theta).
$$

Cosine similarity is invariant to scaling of either vector — only direction matters.

**Relation to Euclidean.** On L2-normalized vectors ($\|\mathbf{x}\| = 1$):

$$
\|\mathbf{x} - \mathbf{y}\|_2^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^\top \mathbf{y} = 2(1 - \cos\theta).
$$

So on unit vectors, Euclidean and cosine give identical neighbor rankings. This is why normalizing embeddings and using Euclidean is often practically equivalent to cosine.

### Derivation: Triangle inequality for L2

We prove $\|\mathbf{x} + \mathbf{y}\|_2 \le \|\mathbf{x}\|_2 + \|\mathbf{y}\|_2$. By the Cauchy–Schwarz inequality, $\mathbf{x}^\top \mathbf{y} \le \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$. Then

$$
\|\mathbf{x} + \mathbf{y}\|_2^2 = \|\mathbf{x}\|^2 + 2\mathbf{x}^\top\mathbf{y} + \|\mathbf{y}\|^2 \le \|\mathbf{x}\|^2 + 2\|\mathbf{x}\|\|\mathbf{y}\| + \|\mathbf{y}\|^2 = (\|\mathbf{x}\| + \|\mathbf{y}\|)^2.
$$

Taking square roots gives the triangle inequality. Apply with $\mathbf{x} \to \mathbf{a} - \mathbf{b}$ and $\mathbf{y} \to \mathbf{b} - \mathbf{c}$ to conclude $d(\mathbf{a}, \mathbf{c}) \le d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c})$.

### Derivation: Mahalanobis as whitened Euclidean

Given $\Sigma = U \Lambda U^\top$ (eigendecomposition), define $\Sigma^{-1/2} = U \Lambda^{-1/2} U^\top$. Then

$$
\begin{aligned}
d_M(\mathbf{x}, \mathbf{y})^2 &= (\mathbf{x} - \mathbf{y})^\top \Sigma^{-1} (\mathbf{x} - \mathbf{y}) \\
&= (\mathbf{x} - \mathbf{y})^\top \Sigma^{-1/2} \Sigma^{-1/2} (\mathbf{x} - \mathbf{y}) \\
&= \|\Sigma^{-1/2}(\mathbf{x} - \mathbf{y})\|_2^2.
\end{aligned}
$$

The transformation $\mathbf{x} \mapsto \Sigma^{-1/2}\mathbf{x}$ is called **whitening** because it maps a zero-mean random vector with covariance $\Sigma$ to one with covariance $I$ — the "white noise" spectrum.

---

## 2.4 Worked Examples

### Example 1: Comparing Euclidean and Manhattan

$\mathbf{x} = (0, 0)$, $\mathbf{y} = (3, 4)$.
- $d_2 = \sqrt{9 + 16} = 5$.
- $d_1 = 3 + 4 = 7$.
- $d_\infty = \max(3, 4) = 4$.

For $\mathbf{y}' = (5, 0)$:
- $d_2 = 5$.
- $d_1 = 5$.
- $d_\infty = 5$.

Euclidean sees $\mathbf{y}$ and $\mathbf{y}'$ as equidistant from the origin; Manhattan sees $\mathbf{y}'$ as closer (5 < 7) because $\mathbf{y}'$ is "further along one axis" while $\mathbf{y}$ spreads its offset over two axes. Which is "correct" depends on your task.

### Example 2: Mahalanobis with correlated features

Suppose

$$
\Sigma = \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1 \end{pmatrix}.
$$

The eigenvalues are $1.9$ (along direction $(1,1)/\sqrt{2}$) and $0.1$ (along $(1,-1)/\sqrt{2}$).

$$
\Sigma^{-1} = \frac{1}{1 - 0.81}\begin{pmatrix} 1 & -0.9 \\ -0.9 & 1 \end{pmatrix} = \frac{1}{0.19}\begin{pmatrix} 1 & -0.9 \\ -0.9 & 1 \end{pmatrix} \approx \begin{pmatrix} 5.26 & -4.74 \\ -4.74 & 5.26 \end{pmatrix}.
$$

Consider two candidate points relative to the origin:
- $\mathbf{a} = (1, 1)$ (along the high-variance direction).
- $\mathbf{b} = (1, -1)$ (along the low-variance direction).

Euclidean: $\|\mathbf{a}\| = \|\mathbf{b}\| = \sqrt{2}$, equidistant.

Mahalanobis:

$$
d_M(\mathbf{0}, \mathbf{a})^2 = \mathbf{a}^\top \Sigma^{-1} \mathbf{a} \approx 5.26 - 4.74 - 4.74 + 5.26 = 1.04, \quad d_M \approx 1.02.
$$

$$
d_M(\mathbf{0}, \mathbf{b})^2 = \mathbf{b}^\top \Sigma^{-1} \mathbf{b} \approx 5.26 + 4.74 + 4.74 + 5.26 = 20.00, \quad d_M \approx 4.47.
$$

Mahalanobis says $\mathbf{b}$ is much farther — because variations in the $(1,-1)$ direction are rare (low variance), so a small deviation there is "surprising" and counts as more distant. $\mathbf{a}$ lies along a high-variance ridge and is therefore unsurprising.

This is the classic application to anomaly detection: a point's Mahalanobis distance from the mean is its number of "standard deviations" in a statistical sense, accounting for correlation.

### Example 3: Hamming on strings

$\mathbf{x} = \texttt{"KITTEN"}$, $\mathbf{y} = \texttt{"SITTEN"}$. One differing position: $d_H = 1$.

$\mathbf{x} = \texttt{"KITTEN"}$, $\mathbf{y} = \texttt{"KITCAR"}$. Positions 4, 5, 6 differ: $d_H = 3$.

Hamming requires equal-length strings. For variable-length strings, use **edit distance (Levenshtein)** — which is a metric but considerably more expensive to compute ($O(mn)$ dynamic programming).

### Example 4: Cosine vs Euclidean on text

Two documents represented by word-count vectors:
- $\mathbf{x} = (10, 0, 5)$ (short doc).
- $\mathbf{y} = (100, 0, 50)$ (long doc, same topic — just longer).

$d_2(\mathbf{x}, \mathbf{y}) = \sqrt{90^2 + 45^2} \approx 100.6$. They look very far.

But $\mathbf{y} = 10 \mathbf{x}$, so $\cos(\theta) = 1$ and $d_{\cos} = 0$. Cosine says they are identical topically.

For topic similarity where magnitude reflects document length (not topical dissimilarity), cosine is the natural choice.

---

## 2.5 Relevance to Machine Learning Practice

### Matching metric to data

| Data type | Typical metric |
|---|---|
| Continuous, standardized | Euclidean |
| Continuous with outliers | Manhattan |
| Continuous, correlated | Mahalanobis |
| Binary / categorical | Hamming (or Jaccard for sets) |
| Text / normalized embeddings | Cosine or Euclidean on unit vectors |
| Sparse high-dim | Cosine, Jaccard, or learned metrics |
| Mixed (numeric + categorical) | Gower's distance, or feature-specific kernels |

### Trade-offs

- **Euclidean** is isotropic, convenient, and supported by every index, but sensitive to scaling.
- **Manhattan** is more robust to outliers along individual axes; some evidence (Aggarwal et al. 2001) that $L_1$ preserves neighbor rank better than $L_2$ in high dimensions.
- **Mahalanobis** adds computational cost ($O(d^2)$ per distance vs $O(d)$) and requires a stable estimate of $\Sigma$. With $n < d$, $\Sigma$ is singular and must be regularized (Ledoit-Wolf shrinkage).
- **Hamming** is cheap (xor + popcount on bits) but loses any notion of ordinal or gradient structure.

### Where each shows up in real systems

- Image embeddings (ResNet, CLIP) + cosine: image search.
- Hashed features + Hamming: deduplication, near-duplicate detection (e.g., perceptual hashes).
- Financial time series + Mahalanobis: outlier/anomaly detection accounting for asset correlations.
- Gene expression + Euclidean (post log-transform + standardization): clustering samples.
- Location data + Haversine (great-circle): spatial nearest neighbors on the globe.

---

## 2.6 Common Pitfalls & Misconceptions

1. **Thinking cosine distance is a metric.** It is not — it violates the triangle inequality. Use *angular distance* $\theta$ (i.e., $\arccos$ of cosine similarity) if a metric is needed.

2. **Forgetting $\Sigma$ must be invertible.** In high-dimensional or low-sample regimes, the sample covariance is singular. Use shrinkage or pseudoinverse.

3. **Applying Euclidean to binary features naively.** Technically legal but inefficient and often inappropriate; Hamming or Jaccard is usually better.

4. **Using squared Euclidean in algorithms that assume a metric.** kd-trees can still work because squared distances are monotonic in Euclidean distance, but do not use squared-Euclidean in structures (e.g., VP-trees) that rely on the triangle inequality.

5. **Mixing metrics across features without thinking.** With mixed types, a principled composite (e.g., Gower's) is better than ad-hoc stitching.

6. **Assuming metric choice is minor.** In high dimensions or with heterogeneous features, metric choice dominates $k$ selection. Spend time here.

7. **Ignoring that normalization changes neighbors.** L2-normalizing embeddings changes the nearest-neighbor ordering under Euclidean (making it equivalent to cosine). This is almost always what you want for embeddings, but be deliberate.

8. **Confusing similarity and distance.** Cosine *similarity* is in $[-1, 1]$ (or $[0,1]$ for nonnegative vectors); cosine *distance* is $1 - \text{similarity}$. Many library APIs differ in sign convention.

---

# Topic 3: Algorithm Variations

## 3.1 Motivation & Intuition

### Why not just vanilla kNN?

Vanilla (uniform, $k$-fixed) kNN treats all $k$ neighbors equally and uses a fixed count regardless of how far away they are. This causes problems:

**Problem 1: The 10th neighbor is very far from the query.** In a sparse region, your "neighbors" may be hundreds of units away. They barely say anything about the query, but they get equal vote with the one genuinely close neighbor.

**Problem 2: Fixed $k$ is wrong in density-varying data.** In a dense cluster, $k=5$ captures a tiny local region; in a sparse region, it captures a huge one. The "resolution" of kNN drifts with local density.

**Problem 3: Ties and ambiguity.** With even $k$ in binary classification, ties happen. With small $k$, a single noisy point dominates.

Two main variations address these:

- **Weighted kNN:** give closer neighbors more influence.
- **Radius-based (fixed-radius) neighbors:** use all neighbors within distance $r$, letting $k$ vary with density.

### Concrete example

Imagine predicting house prices. A new listing's 5 nearest neighbors might be: the house next door (same block, very relevant), and four homes half a mile away in a different neighborhood. Uniform average gives the next-door house only 20% weight. Inverse-distance weighting might give it 80%. The weighted prediction is more faithful to local structure.

Radius-based neighbors: "find all listings within 0.1 miles." In dense urban areas, this returns many comparables; in rural areas, it returns few (or none, signaling low confidence).

---

## 3.2 Conceptual Foundations

### Weighted kNN

Each of the $k$ neighbors carries a weight $w_i \ge 0$ that depends on its distance to the query. Common weight schemes:

- **Uniform:** $w_i = 1$ (standard kNN).
- **Inverse distance:** $w_i = 1/d_i$ (with $\epsilon$ to avoid division by zero).
- **Inverse squared distance:** $w_i = 1/d_i^2$.
- **Gaussian (RBF) kernel:** $w_i = \exp(-d_i^2 / (2 h^2))$ for bandwidth $h$.
- **Triangular / Epanechnikov / tricube kernels:** compactly supported alternatives from kernel density estimation.
- **Rank-based:** $w_i = (k + 1 - \text{rank}_i)$, independent of the numerical distance — often surprisingly robust.

**Prediction rules:**
- Regression: $\hat{y}_q = \sum w_i y_i / \sum w_i$.
- Classification: $\hat{p}(y = c \mid \mathbf{x}_q) = \sum_i w_i \mathbb{1}[y_i = c] / \sum_i w_i$; predict $\arg\max_c$.

### Radius-based neighbors

Given a radius $r > 0$, the neighbor set is

$$
N_r(\mathbf{x}_q) = \{i : d(\mathbf{x}_i, \mathbf{x}_q) \le r\}.
$$

Predictions are computed from $N_r$ (typically via the same weighting rules). $|N_r|$ varies across queries.

**Edge case:** if $N_r = \emptyset$, the prediction is undefined — a feature, not a bug. The model can decline to predict, flagging out-of-distribution queries. This is a powerful property for safety-critical applications.

### Assumptions

- **Weighted kNN** assumes closer neighbors are more informative — true when the target is locally smooth.
- **Radius-based** assumes you can choose a meaningful $r$, which requires knowing the characteristic length-scale of your data (often set via cross-validation or as a quantile of pairwise distances).

### What can go wrong

- **Inverse distance weighting with $d_i \to 0$:** the weight blows up. In practice, queries that coincide with training points dominate — fine if they are correct, catastrophic if noisy. Add a small $\epsilon$: $w_i = 1/(d_i + \epsilon)$.
- **Radius-based in sparse regions:** empty neighborhoods, undefined predictions. Either widen $r$, default to a fallback, or flag as rejection.
- **Radius-based in dense regions:** huge neighborhoods, slow queries, and over-averaging.

### Connection to kernel density estimation and Nadaraya–Watson

Weighted kNN with a kernel is a close cousin of the **Nadaraya–Watson estimator** — a nonparametric regression method:

$$
\hat{f}(\mathbf{x}) = \frac{\sum_{i=1}^n K_h(\mathbf{x} - \mathbf{x}_i)\, y_i}{\sum_{i=1}^n K_h(\mathbf{x} - \mathbf{x}_i)}.
$$

When $K_h$ has compact support (e.g., uniform kernel on $[-h, h]$), this is exactly radius-based weighted kNN with radius $h$.

---

## 3.3 Mathematical Formulation

### General weighted prediction

$$
\hat{y}_q = \frac{\sum_{i \in S} w_i\, y_i}{\sum_{i \in S} w_i}, \qquad S \in \{N_k(\mathbf{x}_q), N_r(\mathbf{x}_q)\}.
$$

### Gaussian kernel weights and bandwidth

With $w_i = \exp(-d_i^2 / (2h^2))$:
- As $h \to 0$: only the nearest neighbor matters ($\to$ 1-NN).
- As $h \to \infty$: all neighbors contribute equally ($\to$ global average).

The bandwidth $h$ plays a role analogous to $k$: it controls the bias–variance trade-off.

### Derivation: Optimal weighting under known noise model

Suppose for each training point $y_i = f(\mathbf{x}_i) + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$ independently, and we want to estimate $f(\mathbf{x}_q) \approx f(\mathbf{x}_i)$ for $\mathbf{x}_i$ near $\mathbf{x}_q$. The minimum-variance unbiased linear estimator is the inverse-variance weighted mean:

$$
\hat{y}_q = \frac{\sum_i (1/\sigma_i^2) y_i}{\sum_i (1/\sigma_i^2)}.
$$

If we model "effective noise" as increasing with distance (because $f(\mathbf{x}_i) \neq f(\mathbf{x}_q)$ when $\mathbf{x}_i \neq \mathbf{x}_q$), and in particular $\sigma_i^2 \propto d_i^2$, this recovers inverse-squared-distance weighting. This provides a loose theoretical justification — though it glosses over the bias from replacing $f(\mathbf{x}_i)$ with $f(\mathbf{x}_q)$.

### Derivation: Bias–variance of kernel-weighted regression (1D sketch)

Assume $y = f(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$, with a smooth $f$. For the Nadaraya–Watson estimator with bandwidth $h$ in 1D under mild conditions:

$$
\text{Bias}[\hat f(x)] \approx \tfrac{1}{2} h^2 \mu_2(K) f''(x), \qquad \text{Var}[\hat f(x)] \approx \frac{\sigma^2 R(K)}{n h\, p(x)},
$$

where $\mu_2(K) = \int u^2 K(u)\, du$, $R(K) = \int K(u)^2\, du$, and $p(x)$ is the density. The MSE-optimal bandwidth scales as $h^* \propto n^{-1/5}$ (1D) and more generally $h^* \propto n^{-1/(d+4)}$ — a rate that degrades with dimensionality (a theme we revisit in Topic 5).

### Radius-based classification probability

With uniform weights in $N_r(\mathbf{x}_q)$:

$$
\hat{p}(y = c \mid \mathbf{x}_q) = \frac{|\{i \in N_r : y_i = c\}|}{|N_r|}.
$$

With $|N_r| = 0$, the estimator is undefined; the model may abstain or fall back.

---

## 3.4 Worked Examples

### Example 1: Weighted classification

Five 2D points with labels and distances from the query:

| i | $d_i$ | $y_i$ |
|---|-------|-----|
| 1 | 0.5   | A   |
| 2 | 0.6   | A   |
| 3 | 1.0   | B   |
| 4 | 1.1   | B   |
| 5 | 1.2   | B   |

$k = 5$. Uniform vote: 2 (A) vs 3 (B) → predict **B**.

Inverse-distance weights $w_i = 1/d_i$:
- $w_1 = 2.000$, $w_2 \approx 1.667$, $w_3 = 1.000$, $w_4 \approx 0.909$, $w_5 \approx 0.833$.
- A: $2.000 + 1.667 = 3.667$.
- B: $1.000 + 0.909 + 0.833 = 2.742$.

Weighted vote: A wins → predict **A**. The two close A neighbors outweigh the three more distant B neighbors.

### Example 2: Radius-based regression

Training data: 1D, $x_i \in \{1, 2, 3, 4, 5\}$, $y_i \in \{2.1, 3.9, 6.0, 8.1, 10.0\}$ (roughly $y = 2x$).

Query $x_q = 2.5$, radius $r = 1.2$.

Neighbors: all $x_i$ with $|x_i - 2.5| \le 1.2$: $x = 2$ ($y = 3.9$) and $x = 3$ ($y = 6.0$) qualify ($|2 - 2.5| = 0.5$, $|3 - 2.5| = 0.5$); also $x = 1.5$ would, but no such point exists.

Uniform average: $(3.9 + 6.0)/2 = 4.95$. True value $2 \cdot 2.5 = 5.0$ — close.

If we had used $r = 0.4$, neither $x = 2$ nor $x = 3$ would qualify. Empty neighborhood → abstain.

### Example 3: Gaussian-weighted kNN

Same data as Example 1, with Gaussian kernel $w_i = \exp(-d_i^2 / 2)$ (bandwidth $h = 1$):
- $w_1 = \exp(-0.125) \approx 0.8825$
- $w_2 = \exp(-0.18) \approx 0.8353$
- $w_3 = \exp(-0.5) \approx 0.6065$
- $w_4 = \exp(-0.605) \approx 0.5461$
- $w_5 = \exp(-0.72) \approx 0.4868$

A-class weight: $0.8825 + 0.8353 = 1.7178$.
B-class weight: $0.6065 + 0.5461 + 0.4868 = 1.6394$.

Gaussian weighting predicts **A**, but by a much narrower margin than inverse-distance. A smaller bandwidth would amplify the A preference; a larger bandwidth would recover the uniform-vote outcome.

---

## 3.5 Relevance to Machine Learning Practice

### When to use weighted kNN

- Tabular regression where neighbors at various distances contribute meaningfully.
- Any setting where uniform kNN underperforms but you believe the right $k$ is roughly correct.
- In classification, when classes are imbalanced or boundaries are noisy: smoother than unweighted.

### When to use radius-based

- When you can define a natural distance threshold ("within 5 km", "similarity > 0.8").
- When you want the model to **abstain** on out-of-distribution inputs — sparse-region queries return empty neighborhoods, a built-in rejection mechanism.
- In DBSCAN and density-based clustering, radius-based neighborhood queries are fundamental.

### Trade-offs

- **Weighted kNN** reduces variance from arbitrary neighbor cutoffs but adds a weight-function hyperparameter (or bandwidth). More to tune.
- **Radius-based** gives density-adaptive behavior but can return wildly varying neighbor counts, causing unstable predictions in sparse regions.
- **Computational cost** is largely the same: both require finding all points within some distance criterion. Well-indexed structures (ball trees, R-trees) support range queries natively.

### Combinations

- **Weighted radius-based kNN** is common: all points within $r$, weighted by a kernel. This is Nadaraya–Watson regression.
- **Capped radius-based:** return at most $k$ points within radius $r$; gives a bounded compute cost.

---

## 3.6 Common Pitfalls & Misconceptions

1. **Blowing up with inverse distance at coincident points.** Always regularize: $w_i = 1/(d_i + \epsilon)$ or cap weights.

2. **Choosing weighting scheme without CV.** Inverse-distance is not universally better than uniform. Evaluate empirically.

3. **Setting radius in raw units.** In high dimensions, all typical pairwise distances concentrate in a narrow range; picking a radius by eyeballing raw scales doesn't work. Use quantiles of pairwise distances.

4. **Confusing bandwidth with $k$.** They are both smoothing parameters, but the mapping between them depends on local density.

5. **Overconfident predictions in near-empty radius neighborhoods.** If only one point is within $r$, reporting it as the prediction with full confidence is misleading. Report $|N_r|$ alongside the prediction.

6. **Double-smoothing.** Using both a large $k$ and heavy distance weighting can over-smooth — the combined effective bandwidth is huge. Pick one smoothing mechanism and tune it.

---

# Topic 4: Efficiency Structures

## 4.1 Motivation & Intuition

### The problem

Naive kNN does $n$ distance computations per query, each $O(d)$. At $n = 10^7$ and $d = 128$, that is over a billion operations per query — seconds or more on modern CPUs. Sub-linear query time is essential.

We need **spatial indexes**: data structures that let us prune the search, visiting only a small fraction of $n$.

### Three families

- **kd-trees:** recursive axis-aligned splits. Best for low-dim ($d \lesssim 20$) Euclidean/Minkowski data.
- **Ball trees:** recursive partition into nested balls. Handles higher dimensions and arbitrary metrics better than kd-trees.
- **Locality-sensitive hashing (LSH):** hashes points so that similar points collide with high probability. Approximate, sublinear, scales to very high dimensions.

Modern ANN libraries (FAISS, HNSW, ScaNN, Annoy) typically layer on additional ideas: product quantization, graph-based navigation, inverted file indexes. We focus on the three classical structures named in this guide.

---

## 4.2 Conceptual Foundations

### kd-tree

A binary tree that recursively partitions $\mathbb{R}^d$ with axis-aligned hyperplanes.

**Construction.** At depth $\ell$, split on axis $j = \ell \mod d$ (or on the axis with maximum variance). Choose the split value as the median along that axis. Recurse on each half.

- Result: $O(n \log n)$ build time, $O(n)$ space.
- Each node stores a splitting axis and value; each leaf stores points.

**Querying for nearest neighbor.**
1. Descend the tree to the leaf containing the query (as if inserting it).
2. That leaf's points form an initial candidate set.
3. Backtrack: at each internal node, if the hyperplane is closer to the query than the current best distance, the other branch might contain a closer point — explore it. Otherwise, prune.

**Key property.** In low dimensions, the expected query time is $O(\log n)$. In high dimensions, most branches must be explored (the pruning fails), and performance degrades to $O(n)$.

**Why it fails in high dim.** In dimension $d$, a hypersphere of radius $r$ around the query must be compared to $d$ hyperplanes in any path. The probability that the sphere intersects the "other side" of a hyperplane grows quickly with $d$; pruning effectively stops working around $d = 20$ (rule of thumb: kd-trees help when $n \gg 2^d$).

### Ball tree

A binary tree that recursively partitions points into **balls** (metric balls, not axis-aligned boxes).

**Construction.** At each node, find two pivot points (e.g., far apart points obtained via a 2-step heuristic), partition the remaining points by proximity to each pivot, and recurse. Each node stores a center $\mathbf{c}$ and a radius $R$ such that every point in the subtree is within $R$ of $\mathbf{c}$.

- $O(n \log n)$ construction.
- $O(n)$ space.

**Querying.** At each node, a triangle-inequality bound tells us the minimum possible distance from the query to any point in the ball:

$$
\min_{\mathbf{x} \in \text{ball}(c, R)} d(\mathbf{x}_q, \mathbf{x}) \ge d(\mathbf{x}_q, \mathbf{c}) - R.
$$

If this lower bound exceeds the current best-known distance, prune the entire subtree. Otherwise, descend.

**Advantages over kd-trees.**
- Works with any metric (not just axis-aligned geometry).
- Handles high dimensions better because balls adapt to data shape.
- Better on non-uniform data densities.

**Disadvantages.**
- More expensive per node (computing a metric distance to the center).
- Still degrades in very high dimensions — no spatial structure beats the curse.

### Locality-sensitive hashing (LSH)

A fundamentally different approach: approximate nearest neighbor via hashing.

**Core idea.** Design a hash family $\mathcal{H}$ such that for any two points $\mathbf{x}, \mathbf{y}$ and a randomly drawn $h \in \mathcal{H}$:
- If $d(\mathbf{x}, \mathbf{y}) \le R$, then $P[h(\mathbf{x}) = h(\mathbf{y})] \ge p_1$.
- If $d(\mathbf{x}, \mathbf{y}) \ge cR$ (for $c > 1$), then $P[h(\mathbf{x}) = h(\mathbf{y})] \le p_2$.

With $p_1 > p_2$, *close points collide more often than far points.*

**Amplification.** Concatenate $L$ independent hashes into a key of length $L$: now $p_1^L$ and $p_2^L$ diverge more. Use $T$ independent hash tables to boost recall.

**Query.** Hash the query into each of the $T$ tables; the candidate set is the union of buckets it falls into. Compute exact distances on candidates only.

### LSH families for common metrics

- **Hamming distance:** $h(\mathbf{x})$ = a random bit position of $\mathbf{x}$. Two points collide iff they agree on that bit. Collision probability $= 1 - d_H/d$.
- **Euclidean (via random projection):** $h(\mathbf{x}) = \lfloor (\mathbf{a}^\top \mathbf{x} + b) / w \rfloor$ for random $\mathbf{a} \sim \mathcal{N}(0, I)$, random $b \sim \text{Uniform}[0, w]$. Based on $p$-stable distributions (Datar, Immorlica, Indyk, Mirrokni, 2004).
- **Cosine / angular:** $h(\mathbf{x}) = \text{sign}(\mathbf{a}^\top \mathbf{x})$ for random Gaussian $\mathbf{a}$. Collision probability $= 1 - \theta / \pi$ (Charikar, 2002).
- **Jaccard (sets):** MinHash. Pick a random permutation $\pi$ of the universe; $h(S) = \min_{i \in S} \pi(i)$. Collision probability equals Jaccard similarity.

### Assumptions and failure modes

- **kd-trees** assume axis-aligned features with meaningful scales and low dimensionality.
- **Ball trees** assume a true metric (triangle inequality).
- **LSH** provides approximate results; guarantees are probabilistic. Getting good recall requires careful tuning of $L$ and $T$.

---

## 4.3 Mathematical Formulation

### kd-tree pruning rule

At an internal node splitting axis $j$ at value $s$:
- The query has coordinate $q_j$ along axis $j$.
- Distance from $\mathbf{x}_q$ to the splitting hyperplane along axis $j$ is $|q_j - s|$.
- If the current best-known nearest distance $d^*$ satisfies $d^* < |q_j - s|$, **prune** the farther subtree.

Formally, the farther subtree contains only points $\mathbf{x}$ with $x_j$ on the far side of $s$, so

$$
d(\mathbf{x}_q, \mathbf{x})^2 \ge (q_j - s)^2 = |q_j - s|^2.
$$

If this already exceeds $(d^*)^2$, no point in the subtree can beat the current best.

### Ball-tree pruning rule

For a subtree rooted at a ball with center $\mathbf{c}$ and radius $R$:

$$
\min_{\mathbf{x} \in \text{subtree}} d(\mathbf{x}_q, \mathbf{x}) \ge \max(0, d(\mathbf{x}_q, \mathbf{c}) - R).
$$

(By triangle inequality: $d(\mathbf{x}_q, \mathbf{c}) \le d(\mathbf{x}_q, \mathbf{x}) + d(\mathbf{x}, \mathbf{c}) \le d(\mathbf{x}_q, \mathbf{x}) + R$, so $d(\mathbf{x}_q, \mathbf{x}) \ge d(\mathbf{x}_q, \mathbf{c}) - R$.)

Prune if $d(\mathbf{x}_q, \mathbf{c}) - R \ge d^*$.

### LSH query complexity

Let $\rho = \log(1/p_1) / \log(1/p_2)$. With parameters $L = O(\log_{1/p_2} n)$ and $T = O(n^\rho)$ hash tables, LSH achieves:
- **Query time:** $O(d n^\rho)$.
- **Space:** $O(n^{1+\rho} + nd)$.

For Euclidean LSH, typically $\rho \approx 1/c^2$ (where $c$ is the approximation ratio). For $c = 2$, $\rho = 1/4$, so queries are $O(n^{1/4})$ — *sublinear* in $n$, a dramatic improvement for huge datasets.

### Derivation: Hamming LSH collision probability

Let $h$ pick a random index $i \in \{1, \dots, d\}$ and return $x_i$. For two vectors $\mathbf{x}, \mathbf{y}$ with Hamming distance $d_H$:
- The $i$-th coordinate agrees on $d - d_H$ out of $d$ positions.
- $P[h(\mathbf{x}) = h(\mathbf{y})] = (d - d_H)/d = 1 - d_H/d$.

If we set the distance threshold at $R = r_1$, then points within $r_1$ collide with probability $\ge 1 - r_1/d$; points beyond $r_2 = cr_1$ collide with probability $\le 1 - r_2/d = 1 - cr_1/d$. The gap allows amplification.

---

## 4.4 Worked Examples

### Example 1: kd-tree on 2D

Dataset: $(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)$. Build a 2D kd-tree.

- **Depth 0 (split on $x$).** Sort by $x$: $2, 4, 5, 7, 8, 9$. Median (of 6) — take 7 (the 4th). Root: $(7, 2)$, split $x = 7$.
  - **Left subtree** (points with $x < 7$): $(2,3), (5,4), (4,7)$.
    - **Depth 1 (split on $y$).** Sort by $y$: $3, 4, 7$. Median = 4. Node: $(5, 4)$.
      - Left ($y < 4$): $(2, 3)$. Leaf.
      - Right ($y \ge 4$): $(4, 7)$. Leaf.
  - **Right subtree** (points with $x \ge 7$, excluding root): $(9, 6), (8, 1)$.
    - **Depth 1 (split on $y$).** Median = either; pick $(9, 6)$.
      - Left ($y < 6$): $(8, 1)$. Leaf.
      - Right ($y \ge 6$): empty.

Query: $(6, 5)$. NN search:

1. At root $(7, 2)$: distance $\sqrt{1 + 9} = \sqrt{10} \approx 3.16$. Current best = 3.16.
2. Query $x = 6 < 7$: descend left.
3. At $(5, 4)$: distance $\sqrt{1 + 1} = \sqrt{2} \approx 1.41$. Update best = 1.41.
4. Query $y = 5 \ge 4$: descend right.
5. At $(4, 7)$: distance $\sqrt{4 + 4} = \sqrt{8} \approx 2.83$. No improvement.
6. Backtrack to $(5, 4)$. Check the other branch: hyperplane at $y = 4$, $|5 - 4| = 1 < 1.41$, so the left side *could* contain a closer point. Descend.
7. At $(2, 3)$: distance $\sqrt{16 + 4} = \sqrt{20} \approx 4.47$. No improvement.
8. Backtrack to root. Check the right branch: hyperplane at $x = 7$, $|6 - 7| = 1 < 1.41$, so the right side could have closer points. Descend.
9. At $(9, 6)$: distance $\sqrt{9 + 1} = \sqrt{10} \approx 3.16$. No improvement.
10. Check left ($y < 6$): hyperplane at $y = 6$, $|5 - 6| = 1 < 1.41$. Descend.
11. At $(8, 1)$: distance $\sqrt{4 + 16} = \sqrt{20}$. No improvement.
12. Done. Nearest neighbor: $(5, 4)$ at distance $\sqrt{2}$.

Despite a small tree, we visited every node — in 2D with 6 points, pruning is limited. Benefits kick in with larger $n$.

### Example 2: LSH for near-duplicate documents with MinHash

Documents are sets of shingles (n-grams). Jaccard similarity between sets $A, B$ is $|A \cap B| / |A \cup B|$.

**MinHash.** Pick random permutation $\pi$ of the universe of shingles. Define $h_\pi(A) = \min_{s \in A} \pi(s)$. A classic result:

$$
P[h_\pi(A) = h_\pi(B)] = \frac{|A \cap B|}{|A \cup B|} = J(A, B).
$$

Use $L$ independent hashes to form a signature vector of length $L$. Estimate $J(A, B) \approx$ fraction of agreeing positions.

For near-duplicate detection: treat signatures as bands of $r$ rows each, hash each band to a bucket. Two documents are *candidates* if they agree on any band. With $L = br$ hashes, $b$ bands of $r$ each, two documents with Jaccard $s$ become candidates with probability

$$
1 - (1 - s^r)^b.
$$

This is an S-shaped curve in $s$: nearly 0 for low similarity, nearly 1 for high similarity, with a steep transition at $s^* \approx (1/b)^{1/r}$. Tuning $b$ and $r$ sets the threshold.

### Example 3: Ball-tree pruning arithmetic

Query at origin, current best distance $d^* = 2.0$.
- Node A: center $(5, 0)$, radius $2.5$. Distance query→center = 5. Lower bound = $5 - 2.5 = 2.5 > 2.0$ → **prune**.
- Node B: center $(3, 0)$, radius $1.5$. Distance query→center = 3. Lower bound = $3 - 1.5 = 1.5 < 2.0$ → **explore**.

---

## 4.5 Relevance to Machine Learning Practice

### Choosing an index

| Situation | Recommended index |
|---|---|
| $n < 10^4$ | Naive brute force; indexing overhead wastes time. |
| $n$ moderate, $d < 20$, Euclidean | kd-tree. |
| $n$ moderate, $d$ up to ~50, any metric | Ball tree. |
| $n$ large, $d$ high (embeddings, images) | ANN: HNSW, IVF+PQ (FAISS), ScaNN, Annoy. |
| Near-duplicate detection, Jaccard on sets | MinHash + banded LSH. |
| Binary hashes | Multi-index Hamming, bit-sampling LSH. |
| Cosine on dense vectors | Angular LSH, HNSW on L2-normalized vectors. |

### Production considerations

- **Build vs query trade-off.** HNSW has slow construction but very fast queries. FAISS IVF is tunable. Annoy builds multiple trees and queries them in parallel.
- **Recall vs speed trade-off.** All approximate methods expose a knob (e.g., `efSearch` in HNSW, `nprobe` in FAISS). Measure recall@k empirically.
- **Memory.** Product quantization (PQ) compresses vectors 10–100×, essential for billion-scale indexes.
- **Updates.** Many ANN indexes do not support fast insert/delete; rebuild periodically, or use graph-based methods that support incremental updates.

### When exact matters

- Legal / medical applications where recall must be guaranteed.
- Small datasets where brute force is fast enough.
- Verification step after approximate retrieval (re-ranking).

### When approximate suffices

- Retrieval for downstream systems (RAG, recommendations) where 95% recall is plenty.
- Real-time systems where latency budget is hard (e.g., <50 ms).

---

## 4.6 Common Pitfalls & Misconceptions

1. **Using kd-trees in 100 dimensions.** They degrade to brute force. Use an ANN method.

2. **Ignoring metric compatibility.** Ball trees with non-metric "distances" (e.g., cosine distance without normalization) can return wrong results because the triangle inequality fails.

3. **Assuming LSH is exact.** It is an approximation with a tunable probability. Always measure recall.

4. **Building an index on data that changes frequently.** Stale indexes return stale results. Plan for incremental updates or rebuilds.

5. **Conflating index speed with model quality.** A faster index returning wrong neighbors is not better — it is worse.

6. **Leaving kd-tree splits at naïve middle-axis rule.** Smarter splits (max-variance axis) can dramatically improve pruning.

7. **Forgetting that high recall at top-1 is different from high recall at top-100.** Pick the operating point that matches your downstream use.

8. **LSH with too few hash tables.** Low recall on near-duplicates. Tune $L$ and $T$ (or equivalently $b$ and $r$) against your specific data.

---

# Topic 5: Curse of Dimensionality

## 5.1 Motivation & Intuition

### The phenomenon

In high-dimensional spaces, geometric and statistical intuitions built for 2D or 3D break in surprising ways. For kNN, the key consequence is:

> **As dimension $d$ grows, pairwise distances concentrate: all points become approximately equidistant from any query, and "nearest neighbor" loses meaning.**

This is the **curse of dimensionality**, a term coined by Richard Bellman in the context of dynamic programming.

### A concrete illustration

Place $n = 100$ points uniformly in the unit hypercube $[0, 1]^d$. For a random query:
- In $d = 2$: the nearest neighbor is typically within distance $\sim 0.1$, the farthest at most $\sqrt{2} \approx 1.41$. A factor of ~14 between min and max.
- In $d = 100$: both min and max cluster near $\sqrt{100/6} \approx 4.08$ (the "typical" distance between uniform cube points), differing by only a few percent. The nearest neighbor is barely closer than the farthest.

The very notion of "neighborhood" becomes fuzzy.

### Another illustration: volume concentration

In a $d$-dimensional unit ball, the fraction of volume within the outermost $\epsilon$-shell is

$$
1 - (1 - \epsilon)^d.
$$

For $\epsilon = 0.01$ and $d = 500$: $1 - 0.99^{500} \approx 1 - 0.0066 \approx 0.993$. Over 99% of the volume is in the outermost 1% of the radius. Most points are "near the surface." There is no "middle" of a high-dimensional ball in any meaningful volumetric sense.

### Why this hurts kNN

kNN assumes close-by points are special. When distance loses contrast, the $k$ nearest are nearly indistinguishable from the $k$ farthest, and predictions become indistinguishable from using the whole dataset.

---

## 5.2 Conceptual Foundations

### Three manifestations

1. **Distance concentration.** Pairwise distances cluster around a typical value; relative contrast vanishes.

2. **Sparsity.** Data density per unit volume drops exponentially in $d$. To maintain a given density, you need exponentially more data: $n \propto r^d$ for fixed local count $r$.

3. **Sample-size requirements for local methods.** For kernel regression with a given bias-variance trade-off, the optimal MSE scales as $n^{-4/(d+4)}$. To halve the MSE, you need $2^{(d+4)/4}$ times more data — in $d = 100$, that is $\sim 10^8$× more data.

### Relative contrast

Define the **relative contrast** at a query $\mathbf{x}_q$:

$$
C_n(\mathbf{x}_q) = \frac{\max_i d(\mathbf{x}_q, \mathbf{x}_i) - \min_i d(\mathbf{x}_q, \mathbf{x}_i)}{\min_i d(\mathbf{x}_q, \mathbf{x}_i)}.
$$

Beyer et al. (1999) showed that, under general conditions on the data distribution, $C_n(\mathbf{x}_q) \to 0$ in probability as $d \to \infty$ — **for independent, identically distributed features**.

### What is the "intrinsic dimensionality"?

Here is the crucial caveat. The "$d$" in the curse is the **intrinsic dimension of the data**, not the ambient dimension. A 1024-dim image embedding may live on a manifold of dimension $\sim 10$. Natural data is usually highly structured, and intrinsic dimension is far below ambient. This is why kNN on deep embeddings works at all.

Mitigations:
- **Dimensionality reduction** (PCA, autoencoders, UMAP) to unveil the intrinsic dimension.
- **Learned metrics** (metric learning) that focus on relevant directions.
- **Feature selection** to drop irrelevant axes.

### Assumptions under which the curse bites hardest

- **Independent features:** the worst case. Each adds a fresh fluctuation to distances.
- **Uniform distributions:** no structure for methods to exploit.
- **Unstructured metric:** treating all axes equally when some are noise.

### When the curse is benign

- Features lie on a low-dim manifold (natural images, language embeddings, gene-expression profiles).
- A subset of features dominates (sparse-effect settings).
- The metric is learned or otherwise adapted to the data.

---

## 5.3 Mathematical Formulation

### Concentration of Euclidean distance between two iid points

Let $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(\mathbf{0}, I_d)$ independently. Then $\mathbf{x} - \mathbf{y} \sim \mathcal{N}(\mathbf{0}, 2 I_d)$ and

$$
\|\mathbf{x} - \mathbf{y}\|_2^2 \sim 2 \chi^2_d.
$$

$\chi^2_d$ has mean $d$ and variance $2d$, so

$$
\mathbb{E}[\|\mathbf{x}-\mathbf{y}\|_2^2] = 2d, \qquad \text{Var}[\|\mathbf{x}-\mathbf{y}\|_2^2] = 8d.
$$

The coefficient of variation of $\|\mathbf{x}-\mathbf{y}\|_2^2$ is $\sqrt{8d}/(2d) = 2/\sqrt{d} \to 0$.

For the distance itself (not squared), a finer analysis using the delta method gives

$$
\|\mathbf{x} - \mathbf{y}\|_2 \approx \sqrt{2d}, \qquad \text{std dev} \approx 1/\sqrt{2} \cdot \sqrt{2} = 1.
$$

So distances have **absolute** spread $O(1)$ but **mean** $O(\sqrt{d})$. Relative spread vanishes.

### Beyer–Goldstein–Ramakrishnan–Shaft (1999) theorem

Let $\mathbf{x}_1, \dots, \mathbf{x}_n$ be iid from an arbitrary distribution in $\mathbb{R}^d$, and $\mathbf{x}_q$ independent. Define

$$
D_{\max}^{(d)} = \max_i d(\mathbf{x}_q, \mathbf{x}_i), \quad D_{\min}^{(d)} = \min_i d(\mathbf{x}_q, \mathbf{x}_i).
$$

Under mild conditions (essentially, that the variance of the pairwise-distance-distribution-component is bounded relative to the mean),

$$
\lim_{d \to \infty} \mathbb{E}\!\left[\frac{D_{\max}^{(d)}}{D_{\min}^{(d)}}\right] = 1.
$$

The farthest and nearest neighbors converge in ratio — the nearest neighbor is no longer meaningfully close.

### Aggarwal et al. (2001) on $L_p$ norms

For iid features, the relative contrast of $L_p$ distances satisfies

$$
\frac{D_{\max} - D_{\min}}{D_{\min}} \sim d^{(1/p) - (1/2)}
$$

as $d \to \infty$. For $p = 2$, this is $d^0 = O(1)$ — contrast is constant. For $p > 2$, it decays; for $p < 2$ (especially fractional Lp), it *grows*, suggesting $L_1$ and fractional Lp are preferable in high dimensions. Interesting in theory, but fractional Lp is not a metric.

### Derivation: Volume of a thin shell

The volume of a $d$-ball of radius $r$ is $V_d r^d$ where $V_d = \pi^{d/2}/\Gamma(d/2 + 1)$. The ratio of a thin shell of thickness $\epsilon$ to total volume is

$$
\frac{V_d - V_d(1-\epsilon)^d \cdot (\text{unit ball vol})}{V_d \cdot (\text{unit ball vol})} = 1 - (1-\epsilon)^d.
$$

Taylor expansion: $1 - (1-\epsilon)^d \approx 1 - e^{-\epsilon d}$. For $\epsilon d \gg 1$, this is nearly 1.

**Takeaway:** in high dim, a uniform distribution on a ball is practically a uniform distribution on its surface.

### Required sample size for density estimation

For kernel density estimation in $d$ dimensions, the MSE-optimal bandwidth is $h^* \propto n^{-1/(d+4)}$, yielding $\text{MSE}^* \propto n^{-4/(d+4)}$. For fixed target MSE, the required $n$ scales as $\text{MSE}^{-(d+4)/4}$ — exponential in $d$.

For 1D to reach MSE $10^{-4}$: $n \propto 10^{5}$. For $d = 20$: $n \propto 10^{24}$. Beyond reach.

---

## 5.4 Worked Examples

### Example 1: Distance contrast via simulation (explained numerically)

Sample $n = 1000$ points uniformly in $[0,1]^d$. Pick a random query. Compute the ratio $\max d / \min d$.

Typical empirical values:
- $d = 2$: ratio $\approx 10$–20.
- $d = 10$: ratio $\approx 2$–3.
- $d = 100$: ratio $\approx 1.1$–1.2.
- $d = 1000$: ratio $\approx 1.02$.

At $d = 1000$, the nearest neighbor is only 2% closer than the farthest. "Nearest" is essentially meaningless.

### Example 2: Needed points for coverage

We want a 1% radius around every query (i.e., at least one training point within 1% of the range along each axis). In $d$ dims, that corresponds to training points covering $(0.01)^d$ fraction of the volume. To have one point in expectation per such "cell," we need

$$
n \ge 100^d.
$$

$d = 2$: $n = 10^4$. Feasible. $d = 10$: $n = 10^{20}$. Infeasible.

### Example 3: Gaussian pairwise distance spread

Simulating (conceptually): $n = 1000$ points from $\mathcal{N}(\mathbf{0}, I_d)$.
- $d = 5$: pairwise distances range roughly $1$ to $7$, mean $\approx 3$.
- $d = 50$: pairwise distances range $8$ to $13$, mean $\approx 10$.
- $d = 500$: range $30.5$ to $32$, mean $\approx 31.6$.

The spread (in absolute units) is $O(1)$ while the mean grows as $\sqrt{d}$.

### Example 4: kNN misbehavior in a 1000-dim synthetic

Task: binary classification with 5 informative features and 995 noise features drawn from $\mathcal{N}(0, 1)$.

- kNN on all 1000 features: ~50% accuracy (random guess). The 995 noise features overwhelm the signal — pairwise distances are dominated by noise.
- kNN on the 5 informative features: ~95% accuracy.
- Feature selection / dimensionality reduction to recover the 5 informative features: ~95% accuracy.

This is the canonical demonstration: kNN's performance is effectively a function of the signal-to-noise ratio in the distance metric.

---

## 5.5 Relevance to Machine Learning Practice

### Implications for kNN

- Raw, high-dim feature spaces are hostile to kNN.
- Embed or project first. Learned embeddings (BERT, CLIP, SimCLR) map naturally high-dim objects (text, images) into ~128–1024 dim spaces where distances are meaningful because the embedding is trained to make them so.
- Always evaluate kNN on processed features, not raw.

### Mitigation strategies

- **Feature selection** (L1, mutual information, tree-based importance): drop irrelevant features before kNN.
- **PCA / truncated SVD:** linear dimensionality reduction, cheap and fast, preserves variance directions.
- **Manifold learning** (UMAP, t-SNE, Isomap): nonlinear, preserves local structure; useful for visualization and sometimes for downstream kNN (especially UMAP).
- **Autoencoders:** learn a low-dim bottleneck preserving reconstruction fidelity.
- **Metric learning** (LMNN, NCA, triplet / contrastive loss): learn a distance that focuses on label-relevant variation.
- **Random projections** (Johnson–Lindenstrauss): project to $O(\log n / \epsilon^2)$ dims while preserving pairwise distances up to $(1 \pm \epsilon)$. Cheap preprocessing.
- **Deep embeddings:** pretrained or task-fine-tuned representations.

### Trade-offs

- Dimensionality reduction injects bias (information loss). Tune reduction rank via CV.
- Learned metrics need labeled data; unsupervised reduction does not.
- Approximate nearest-neighbor indexes (LSH, HNSW) shift the problem: they accept distance contrast loss by accepting recall loss.

---

## 5.6 Common Pitfalls & Misconceptions

1. **Blaming "high dimension" without checking intrinsic dimension.** 1024-dim embeddings from a good encoder may have intrinsic dim 10–20, where kNN works fine.

2. **Trusting PCA blindly in supervised tasks.** PCA preserves variance, not label information. LDA or supervised embedding methods may be better.

3. **Normalizing after dimensionality reduction without re-thinking.** The scales of PCA components differ from raw features; re-standardize if needed.

4. **Using Johnson-Lindenstrauss with too few target dims.** The bound requires $k \ge O(\log n / \epsilon^2)$; undershooting causes distance distortion.

5. **Confusing dimensionality with dataset size.** Adding more data does not overcome the curse if features are genuinely $d$-dimensional noise.

6. **Believing fractional Lp is a magic fix.** It may mildly help distance contrast but loses the triangle inequality and metric-index support.

7. **Ignoring locality.** High-dim data often has local manifold structure. Methods like UMAP and locality-preserving projections leverage this.

---

# Topic 6: Parameter Selection

## 6.1 Motivation & Intuition

### Why $k$ matters

The choice of $k$ (and of weighting scheme, metric, etc.) determines everything about kNN's behavior.

- **$k = 1$:** ultra-high resolution, zero regularization. Every noisy point has its own neighborhood.
- **$k = n$:** every prediction is the global majority / mean. Zero resolution.

There is no universal best $k$. The right choice depends on:
- Noise level in labels.
- Local data density.
- Intrinsic dimensionality.
- The task (classification vs regression; 0/1 loss vs squared loss vs rankings).

Therefore: **cross-validation.**

### Why weighting matters

For a given $k$, different weighting schemes yield different predictions. Weighting interpolates between kNN and broader smoothing. Choosing between weighting schemes is itself a hyperparameter decision.

### Real-world example

A churn model might use $k = 50$ in regions dense with paying customers (lots of data; can afford many neighbors for stability) but that same $k$ undersmoothes predictions in rare segments (e.g., enterprise customers with <100 examples). Adaptive $k$ or radius-based neighbors address this.

---

## 6.2 Conceptual Foundations

### Cross-validation for $k$

**k-fold CV** (here "k-fold" overloads "k"; call folds $K$ to avoid confusion):
1. Partition data into $K$ folds.
2. For each candidate $k \in \{1, 3, 5, \dots, k_{\max}\}$:
   - For each fold $i$: train on the other $K-1$ folds, evaluate on fold $i$.
   - Average the error across folds.
3. Pick the $k$ with lowest CV error.

**Leave-one-out CV (LOOCV).** A special case with $K = n$. For kNN specifically, LOOCV is very cheap: predicting $\mathbf{x}_i$ with LOO is just the $(k+1)$-NN of $\mathbf{x}_i$ *excluding* $\mathbf{x}_i$ itself. You can compute LOO errors for many $k$ values by sorting each row of the pairwise distance matrix once.

**Stratified CV.** For classification with imbalance, preserve class proportions in each fold.

**Nested CV.** When you also tune distance metric, weighting, preprocessing: inner CV picks hyperparameters, outer CV estimates generalization. Prevents test-set leakage.

### Heuristics and rules of thumb

- **Odd $k$ for binary classification** — prevents ties.
- **$k \approx \sqrt{n}$** — a common starting point; balances $k \to \infty$ (consistency) and $k/n \to 0$ (locality).
- **Start with $k = 5$ or $k = 10$** — often close to optimal for medium data.
- **Multiclass:** make $k$ a multiple of the number of classes plus one, to reduce ties.

### Distance weighting schemes

Common schemes to evaluate:
1. Uniform.
2. Inverse-distance.
3. Inverse-squared distance.
4. Gaussian (with bandwidth $h$).
5. Tricube kernel (popular in LOESS):

$$
K(u) = (1 - |u|^3)^3, \quad |u| \le 1.
$$

6. Rank-based.

Treat the choice as a hyperparameter and CV alongside $k$.

### Assumptions

- CV assumes the held-out folds are representative of the query distribution. Violated if:
  - Temporal data (use time-series CV, not random splits).
  - Grouped data (use group-aware CV).
  - Covariate shift (CV underestimates production error).

### Alternatives to CV for $k$

- **Penalized CV / one-standard-error rule:** pick the largest $k$ whose CV error is within one SE of the best — favors simpler (more regularized) models.
- **Bayesian optimization / grid search over joint (k, metric, weighting).**
- **Meta-learning** (cross-dataset transfer) for initial $k$ estimates.

---

## 6.3 Mathematical Formulation

### CV risk estimator

For loss $\ell$ and candidate $k$:

$$
\widehat{R}_{\text{CV}}(k) = \frac{1}{K}\sum_{i=1}^{K} \frac{1}{|\mathcal{F}_i|}\sum_{j \in \mathcal{F}_i} \ell(y_j, \hat{y}_j^{(-i)}(k)),
$$

where $\hat{y}_j^{(-i)}(k)$ is the kNN prediction for $\mathbf{x}_j$ using all data except fold $i$.

Select $\hat{k} = \arg\min_k \widehat{R}_{\text{CV}}(k)$.

### One-standard-error rule

Let $\widehat{\text{SE}}(k)$ be the standard error of the CV estimate across folds. Pick

$$
\hat{k}_{1SE} = \max\{k : \widehat{R}_{\text{CV}}(k) \le \widehat{R}_{\text{CV}}(\hat{k}) + \widehat{\text{SE}}(\hat{k})\}.
$$

In kNN, larger $k$ is simpler (more regularized), so this favors more stable models.

### Efficient LOOCV for kNN

Precompute the full pairwise distance matrix (or use an index). For each $i$, the $k$-NN of $\mathbf{x}_i$ excluding itself is the $(k+1)$ closest points in the row. You can compute LOO predictions for all candidate $k$ values simultaneously by keeping the top-$k_{\max}+1$ per row. Total cost: $O(n^2 d)$ for the distance matrix, $O(n \cdot k_{\max})$ for aggregation.

### Derivation: Bias–variance of kNN regression

Assume $y = f(\mathbf{x}) + \epsilon$ with $\mathbb{E}[\epsilon] = 0$, $\text{Var}[\epsilon] = \sigma^2$. For a query $\mathbf{x}_q$ and its $k$ nearest neighbors $\mathbf{x}_{(1)}, \dots, \mathbf{x}_{(k)}$ (treating as fixed):

$$
\hat{y}_q = \frac{1}{k}\sum_{j=1}^k (f(\mathbf{x}_{(j)}) + \epsilon_{(j)}).
$$

$$
\mathbb{E}[\hat{y}_q] - f(\mathbf{x}_q) = \frac{1}{k}\sum_{j=1}^k [f(\mathbf{x}_{(j)}) - f(\mathbf{x}_q)] \quad (\text{bias}),
$$

$$
\text{Var}[\hat{y}_q] = \frac{\sigma^2}{k}.
$$

**Interpretation:**
- **Variance** decreases as $\sigma^2 / k$: larger $k$ averages out noise.
- **Bias** grows with $k$ because distant neighbors have $f$ values farther from $f(\mathbf{x}_q)$ (assuming $f$ is smooth — the bias is small for small $k$ if $f$ varies smoothly).

MSE is minimized at an intermediate $k$, the empirical CV-optimal point.

### Effective number of parameters

For kNN regression, the "degrees of freedom" behaves like $n/k$ — large $k$ is roughly like a model with few parameters. This gives an AIC-like intuition for regularization.

---

## 6.4 Worked Examples

### Example 1: CV curve for $k$ on a toy classification problem

Suppose we run 5-fold CV for $k \in \{1, 3, 5, 7, 9, 11, 15, 21, 31\}$ and observe these error rates:

| $k$ | CV error | SE |
|-----|----------|----|
| 1   | 0.24     | 0.03 |
| 3   | 0.19     | 0.03 |
| 5   | 0.17     | 0.02 |
| 7   | 0.16     | 0.02 |
| 9   | 0.16     | 0.02 |
| 11  | 0.17     | 0.02 |
| 15  | 0.19     | 0.03 |
| 21  | 0.22     | 0.03 |
| 31  | 0.26     | 0.03 |

- Minimum-CV rule: $\hat k = 7$ or $9$ (tied at 0.16).
- 1-SE rule: the best error is 0.16 with SE 0.02; any $k$ with error $\le 0.18$ qualifies. $k = 11$ (err 0.17) qualifies. $k = 15$ (err 0.19) does not. Pick $\hat{k}_{1SE} = 11$.

The 1-SE rule prefers a slightly simpler model that is statistically indistinguishable from the best.

### Example 2: Weighting vs uniform

Same problem; compare uniform weights to inverse-distance at $k = 9$:
- Uniform: CV error 0.16.
- Inverse-distance: CV error 0.14.

Choose inverse-distance weighting. Then re-CV over $k$ with inverse-distance to find its own optimal $k$ — hyperparameters interact and should be optimized jointly.

### Example 3: $k = \sqrt n$ rule of thumb

Dataset with $n = 10{,}000$. Heuristic: $k \approx 100$. This is a good anchor for a preliminary model; confirm via CV. On real problems, the CV-optimal $k$ often lands somewhere between 5 and $\sqrt n$, but not always — noisy problems push it higher; low-noise high-resolution tasks push it lower.

### Example 4: Metric + $k$ joint grid

Run a 3×5 grid:

| Metric ↓ / $k$ → | 5 | 7 | 9 | 11 | 15 |
|---|---|---|---|---|---|
| Euclidean      | 0.21 | 0.19 | 0.18 | 0.19 | 0.22 |
| Manhattan      | 0.20 | 0.18 | 0.17 | 0.18 | 0.21 |
| Mahalanobis    | 0.17 | 0.15 | **0.14** | 0.15 | 0.18 |

Best configuration: Mahalanobis + $k = 9$. The joint grid reveals that the metric matters more than $k$ in this case.

---

## 6.5 Relevance to Machine Learning Practice

### Production workflow

1. **Preprocess:** standardize features; handle categoricals; optionally dimensionality-reduce.
2. **Pick metric:** start with Euclidean on standardized features, or cosine on embeddings.
3. **CV over $k$ and weighting:** use 5- or 10-fold, stratified for classification.
4. **Apply the 1-SE rule** (optional) for a more conservative, stable choice.
5. **Validate** on a held-out test set.
6. **Monitor production:** drift detection on the neighbor-distance distribution (if queries' neighbor distances start drifting large, your reference set is stale).

### When CV is misleading

- **Temporal data:** random-split CV leaks future info. Use forward-chaining (time-series CV).
- **Grouped data** (e.g., multiple rows per user): group-aware CV prevents the same user appearing in train and validation.
- **Covariate shift between CV and production:** CV over-estimates production performance.
- **Small datasets:** CV estimates are noisy; consider repeated CV (run multiple random partitions and average).

### Trade-offs

- **More folds → less biased estimate, higher variance, more compute.** 5- or 10-fold is the conventional sweet spot.
- **LOOCV is nearly unbiased but has higher variance** (each LOO fit is highly correlated with the others).
- **Grid vs random search:** random search is often more efficient when only a few hyperparameters matter.
- **Bayesian optimization:** overkill for just tuning $k$; useful for joint (k, weighting, metric, preprocessing) searches.

### Distance-weighting selection in practice

- Start with uniform. If CV shows a dropoff for larger $k$, try inverse-distance — it softens the effect of distant neighbors.
- If predictions are noisy at small $k$, try Gaussian weighting with bandwidth tuned.
- Rank-based weighting is a defensive choice in high dimensions (because raw distances concentrate).

---

## 6.6 Common Pitfalls & Misconceptions

1. **Using test set for $k$ selection.** Catastrophic but common. Use CV on a train/validation partition, hold the test set for final evaluation.

2. **Not stratifying for imbalanced classes.** Minority class may be entirely absent from some folds.

3. **Ignoring temporal structure.** Training on future and testing on past inflates CV accuracy. Always reason about whether your CV respects the data's generative process.

4. **Cross-validating $k$ without also validating preprocessing.** If standardization is fit on all data including validation folds, you leak. Apply preprocessing inside the CV loop.

5. **Picking $k$ on CV error, then reporting CV error as performance.** That is optimistic. Use nested CV or a held-out test.

6. **Only trying even $k$.** Ties in binary classification. Prefer odd $k$.

7. **Picking the absolute minimum-error $k$.** A tiny improvement may not be significant. Apply the 1-SE rule or check confidence intervals.

8. **Ignoring that $k$ should scale with $n$ as data grows.** In production, as data accumulates, re-CV $k$ periodically.

9. **Cross-validating with very small fold sizes in high-dim.** The curse of dimensionality makes small-fold CV especially noisy. Use more folds or repeated CV.

10. **Conflating "no training cost" with "no tuning cost."** kNN is lazy at training but requires substantial tuning at design time: $k$, metric, weighting, preprocessing.

---

# Interview Preparation

Questions below are organized by difficulty: **Foundational**, **Mathematical**, **Applied**, and **Debugging & Failure-Mode**, with a section of **Common Follow-ups & Probing**.

## Foundational (definitions, intuition, high-level)

**Q1. What is kNN and how does it work?**

**Answer.** kNN is a non-parametric, instance-based (lazy) learning algorithm. Training stores the dataset; at inference, for a query $\mathbf{x}_q$, we (1) compute distances to every training point, (2) select the $k$ smallest, (3) for classification take the plurality vote of their labels, for regression take the mean. It makes no parametric assumption about the underlying function; instead, it assumes local smoothness — that similar inputs have similar labels.

---

**Q2. Why is kNN called a "lazy learner"?**

**Answer.** Because it does no work at training time. A parametric model like logistic regression computes parameters during training and applies them cheaply at inference. kNN stores examples verbatim and defers all computation to inference. The "laziness" trades training speed for inference speed — a critical distinction when designing latency-sensitive systems.

---

**Q3. What are the main hyperparameters of kNN, and what do they control?**

**Answer.**
- **$k$:** number of neighbors. Controls bias–variance: small $k$ = low bias, high variance; large $k$ = opposite.
- **Distance metric:** Euclidean, Manhattan, Mahalanobis, cosine, Hamming, etc. Encodes the notion of similarity.
- **Weighting:** uniform, inverse-distance, kernel-based. Modulates how much close vs far neighbors contribute.
- **Preprocessing:** standardization, dim reduction. Not strictly a kNN hyperparameter but inseparable from it.

---

**Q4. Is kNN parametric or non-parametric? Why?**

**Answer.** Non-parametric. It does not commit to a fixed-size parameter vector that summarizes the training data. Instead, the "model" grows with $n$ — kNN stores all $n$ points, and its effective complexity scales with data size. The decision boundary can be arbitrarily complex given enough data.

---

**Q5. Why do we standardize features before kNN?**

**Answer.** Because Euclidean (and most) distance metrics give each feature weight proportional to its variance. A feature with variance $100^2$ contributes $10^4$× more to squared distance than a feature with variance 1. Without standardization, distances are dominated by whichever feature has the largest scale, making the algorithm effectively ignore other features. Standardizing (zero mean, unit variance) equalizes contributions.

---

**Q6. What does "curse of dimensionality" mean for kNN?**

**Answer.** In high dimensions, pairwise distances concentrate: the ratio of farthest to nearest neighbor tends to 1. "Nearest" loses meaning; kNN's core assumption (that close points are special) fails. Concretely, in $\mathbb{R}^d$ with iid coordinates, $\text{Var}[\|\mathbf{X} - \mathbf{Y}\|] / \mathbb{E}[\|\mathbf{X} - \mathbf{Y}\|] \to 0$. Mitigations: dim reduction, learned embeddings, metric learning, feature selection.

---

**Q7. What is the difference between kNN for classification and kNN for regression?**

**Answer.** Mechanically nearly identical — find the $k$ nearest neighbors — but the aggregation rule differs. Classification: majority vote (or weighted) of neighbor labels, estimating $p(y=c \mid \mathbf{x})$. Regression: mean (or weighted mean) of neighbor targets, estimating $\mathbb{E}[y \mid \mathbf{x}]$. Both are conditional-expectation estimators, one discrete, one continuous.

---

**Q8. What is the Cover–Hart theorem?**

**Answer.** Cover & Hart (1967) showed that the 1-NN classifier's asymptotic error rate $R_\infty$ satisfies $R^* \le R_\infty \le 2 R^* (1 - R^*)$, where $R^*$ is the Bayes-optimal error. So even with zero training (just memorization), 1-NN is within a factor of 2 of the best possible classifier — remarkable given it has no model. For general $k$ with $k \to \infty$, $k/n \to 0$, kNN is consistent ($R \to R^*$).

---

## Mathematical (derivations, assumptions, edge cases)

**Q9. Derive the bias–variance decomposition of kNN regression.**

**Answer.** Assume $y = f(\mathbf{x}) + \epsilon$ with $\mathbb{E}[\epsilon] = 0$, $\text{Var}[\epsilon] = \sigma^2$, independent noise. Condition on the nearest-neighbor set $\{(\mathbf{x}_{(j)}, y_{(j)})\}_{j=1}^k$. Then

$$
\hat{y}_q = \tfrac{1}{k}\sum_{j=1}^k (f(\mathbf{x}_{(j)}) + \epsilon_{(j)}).
$$

- $\mathbb{E}[\hat{y}_q] = \tfrac{1}{k}\sum_j f(\mathbf{x}_{(j)})$.
- $\text{Bias} = \tfrac{1}{k}\sum_j [f(\mathbf{x}_{(j)}) - f(\mathbf{x}_q)]$ — grows with $k$ because farther neighbors differ more from $f(\mathbf{x}_q)$.
- $\text{Var}[\hat{y}_q] = \sigma^2 / k$ — shrinks with $k$.
- $\text{MSE} = \text{Bias}^2 + \text{Var}$; optimal $k$ balances both.

---

**Q10. Prove that Euclidean and cosine distance give the same neighbor ordering for L2-normalized vectors.**

**Answer.** For $\|\mathbf{x}\| = \|\mathbf{y}\| = 1$:

$$
\|\mathbf{x} - \mathbf{y}\|_2^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}^\top\mathbf{y} = 2 - 2\cos\theta = 2(1 - \cos\theta) = 2 d_{\cos}(\mathbf{x}, \mathbf{y}).
$$

So $\|\mathbf{x} - \mathbf{y}\|_2^2$ is a monotonically increasing function of $d_{\cos}$. Thus the ordering of neighbors by Euclidean distance equals the ordering by cosine distance. Without normalization, the relation fails: large-norm vectors are penalized more by Euclidean than by cosine.

---

**Q11. Show that Mahalanobis distance equals Euclidean distance after whitening.**

**Answer.** Let $\Sigma$ be symmetric positive-definite with $\Sigma^{-1/2} = U\Lambda^{-1/2}U^\top$ (from eigendecomposition $\Sigma = U\Lambda U^\top$). Then

$$
d_M(\mathbf{x}, \mathbf{y})^2 = (\mathbf{x}-\mathbf{y})^\top \Sigma^{-1}(\mathbf{x}-\mathbf{y}) = (\Sigma^{-1/2}(\mathbf{x}-\mathbf{y}))^\top (\Sigma^{-1/2}(\mathbf{x}-\mathbf{y})) = \|\Sigma^{-1/2}\mathbf{x} - \Sigma^{-1/2}\mathbf{y}\|_2^2.
$$

So the linear transform $\mathbf{x} \mapsto \Sigma^{-1/2}\mathbf{x}$ (whitening) converts Mahalanobis into Euclidean.

---

**Q12. What goes wrong if $\Sigma$ in Mahalanobis is near-singular? How would you fix it?**

**Answer.** A near-singular $\Sigma$ has eigenvalues near zero; $\Sigma^{-1}$ has huge eigenvalues that amplify tiny variations along those directions. The Mahalanobis distance becomes unstable — a small feature perturbation produces an enormous distance, driven by noise in the low-variance direction.

**Fixes:**
- **Ridge / shrinkage:** replace $\Sigma$ with $\Sigma + \lambda I$, or use Ledoit-Wolf shrinkage $\Sigma \leftarrow (1-\alpha)\hat\Sigma + \alpha \bar\sigma^2 I$ with data-driven $\alpha$.
- **Pseudoinverse:** use $\Sigma^+$ (Moore-Penrose), effectively projecting onto the non-null-space.
- **Dim reduction first:** PCA to retain top-$k$ components, then fit $\Sigma$ in the reduced space.

---

**Q13. Derive the query time complexity of kNN with a kd-tree in low dimensions.**

**Answer.** In $d$ dimensions with balanced kd-tree construction ($O(n \log n)$ build), nearest-neighbor queries have expected time $O(\log n)$ *when $d$ is small*, but rigorous analyses (Friedman, Bentley, Finkel, 1977) show worst-case $O(n^{1-1/d} + k)$. For $d = 2$: $O(\sqrt n)$. For $d = 10$: $O(n^{0.9})$ — barely better than brute force. The pruning relies on axis-aligned hyperplanes bounding the candidate region; in high $d$, most hyperplanes fail to prune, and both subtrees must be explored at nearly every internal node.

---

**Q14. Explain why the Minkowski distance with $p < 1$ is not a metric.**

**Answer.** The triangle inequality requires $d(\mathbf{a}, \mathbf{c}) \le d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c})$. For $L_p$ with $p < 1$, consider $\mathbf{a} = (0,0)$, $\mathbf{b} = (1, 0)$, $\mathbf{c} = (1, 1)$.
- $d(\mathbf{a}, \mathbf{c}) = (1^p + 1^p)^{1/p} = 2^{1/p}$.
- $d(\mathbf{a}, \mathbf{b}) + d(\mathbf{b}, \mathbf{c}) = 1 + 1 = 2$.

For $p = 0.5$: $d(\mathbf{a}, \mathbf{c}) = 2^2 = 4 > 2$. Triangle inequality fails. (Equivalently: the unit ball under $L_p$ with $p < 1$ is non-convex, a necessary failure for a norm.)

---

**Q15. What does LSH guarantee, and how do you tune the number of hash tables?**

**Answer.** A $(R, cR, p_1, p_2)$-sensitive hash family collides "close" pairs (distance $\le R$) with prob $\ge p_1$ and "far" pairs (distance $\ge cR$) with prob $\le p_2$, with $p_1 > p_2$. To achieve high recall of true neighbors:

- Concatenate $L$ hash functions per table → collision probs become $p_1^L, p_2^L$ (stronger separation but lower recall).
- Use $T$ independent tables → recall = $1 - (1 - p_1^L)^T$ (for near pairs), false-positive expected cost $\propto n p_2^L T$.

Choose $L$ large enough to reject far points; choose $T$ large enough that $(1 - p_1^L)^T$ is small (recall is high). Optimal $T = O(n^\rho)$ where $\rho = \log(1/p_1)/\log(1/p_2)$. Query time $O(d n^\rho)$ — sublinear.

---

## Applied (real-world ML scenarios, design decisions)

**Q16. You are building a semantic-search system over 50M documents. Should you use kNN? How?**

**Answer.** Yes, but with care:
1. **Embed documents** via a pretrained encoder (BGE, E5, OpenAI text-embedding-3, etc.) into ~1024-dim vectors.
2. **L2-normalize** embeddings — makes cosine and Euclidean equivalent.
3. **Use an ANN index** (HNSW via FAISS or native; or IVF+PQ for memory-constrained settings). Exact kNN over 50M × 1024 is ~50 GFLOPs per query — too slow.
4. **Evaluate recall@k** on a held-out labeled set; tune `efSearch` (HNSW) or `nprobe` (IVF) to hit target latency.
5. **Re-rank** the top few hundred ANN results with a cross-encoder for precision.
6. **Monitor** drift in query embeddings and periodically rebuild the index.

---

**Q17. A team uses kNN (k=5) for fraud detection with 99% non-fraud. Accuracy is 99% but almost no fraud is caught. What's happening and how to fix?**

**Answer.** The class imbalance dominates. With 99% non-fraud, even if the 5 nearest neighbors contain 1 fraud and 4 non-fraud, the vote is non-fraud; a "perfect" kNN predicts non-fraud everywhere, scoring 99% accuracy by trivially refusing to predict the minority.

**Fixes:**
- **Class-weighted voting:** weight each neighbor by its class's inverse frequency.
- **Distance weighting:** closer fraud neighbors outweigh farther non-fraud.
- **Threshold adjustment:** use $\hat p(\text{fraud} \mid \mathbf{x})$ and pick a threshold below 0.5 to maximize the chosen metric (F1, PR-AUC).
- **Resampling:** oversample fraud (SMOTE) or undersample non-fraud for the reference set.
- **Switch metrics:** track F1, precision at high recall, PR-AUC — not accuracy.
- **Learned metric / embedding:** train a triplet-loss encoder that pulls fraud cases together.

---

**Q18. How would you choose $k$ in production?**

**Answer.**
1. **Cross-validation** on held-out training data, evaluating the business metric (F1, NDCG, MSE) over $k \in \{1, 3, 5, ..., \sqrt n\}$.
2. Apply the **1-SE rule** to prefer larger (simpler) $k$ when differences are not statistically significant.
3. **Stratify** folds for imbalance.
4. For **time-series**, use forward-chaining CV.
5. **Re-CV periodically** as data grows.
6. **Nested CV** if also tuning metric/weighting.
7. Sanity-check $k$ against $\sqrt n$ as a baseline.

---

**Q19. Your kNN model is slow at query time. What do you do?**

**Answer.** Work through this decision tree:
1. **Profile.** Is the cost in distance computation or top-$k$ selection?
2. **Dimensionality:** reduce if $d$ is large (PCA or learned embedding).
3. **Index:** if $d < 20$, kd-tree; if $d < 100$, ball tree; if higher, ANN (HNSW, FAISS IVF+PQ, ScaNN).
4. **Data:** subsample or prototype-reduce (CNN — Condensed NN, editing algorithms) to shrink the reference set without losing decision-boundary info.
5. **Approximation:** accept, say, 95% recall for 100× speedup via LSH or HNSW.
6. **Hardware:** batch queries, GPU FAISS, half-precision.
7. **Caching:** for repeated queries, memoize.

---

**Q20. When would you prefer radius-based neighbors over $k$-NN?**

**Answer.**
- When the notion of "close" has a natural threshold (e.g., "within 5 km", "similarity > 0.8").
- When you want the model to **abstain** on out-of-distribution inputs — a query with an empty $r$-neighborhood can be flagged rather than predicted.
- In density-varying data, where a fixed $k$ misrepresents local structure (e.g., DBSCAN clustering depends on radius-based neighborhoods).

---

**Q21. How do you combine numeric and categorical features for kNN?**

**Answer.** Options:
1. **One-hot encode categoricals**, then Euclidean on the joint vector — crude but often acceptable if one-hots are scaled to match numeric variances.
2. **Gower's distance:** a weighted average of feature-specific distances (e.g., normalized absolute diff for numeric, Hamming-like for categorical). Principled but computationally heavier.
3. **Embedding:** map categoricals to learned dense vectors (e.g., via a target-encoder or a neural net's embedding layer), concatenate with standardized numerics, use Euclidean.
4. **Kernel-combined:** $k(\mathbf{x}, \mathbf{y}) = k_{\text{num}} \cdot k_{\text{cat}}$; convert to distance via $d^2 = 2(1-k)$ for normalized kernels.

---

## Debugging & Failure-Mode

**Q22. Your kNN achieves 100% accuracy on the training set but 60% on the test set. What's happening?**

**Answer.** If $k = 1$, each training point is its own nearest neighbor — trivially predicting itself correctly. Training-set "error" with 1-NN is 0 by construction. This is severe overfitting to individual training points.

**Diagnosis:** check $k$. **Fix:** increase $k$, use CV to pick it. Also, evaluate on held-out data or via LOOCV; never trust in-sample 1-NN error.

---

**Q23. A kNN model on 10-dim data suddenly degrades after a feature engineering change that added 200 noise features. Why?**

**Answer.** Curse of dimensionality. The 200 noise features add $\sim \sqrt{200}$ worth of random distance to every pair. Signal-to-noise in distances crashes: informative distances get swamped. Contrast between nearest and farthest neighbors vanishes. The added features are individually tiny but collectively dominate.

**Fix:** drop them, or run feature selection (L1, mutual information), or re-embed/project to remove them.

---

**Q24. CV picks $k = 1$. Should you use $k = 1$ in production?**

**Answer.** Probably not. $k = 1$ is extremely sensitive to label noise and outliers — a single mislabel flips the prediction. CV preferring $k = 1$ often indicates:
- **Very clean, low-noise data** — then $k = 1$ may genuinely be optimal.
- **Training–test leak** in the CV setup (e.g., duplicate points).
- **Small data where $k = 1$'s high variance happens to help on this specific fold.**

Investigate. Apply the 1-SE rule: pick the largest $k$ within one SE of the optimum; usually $k \ge 3$. Also, for probabilistic predictions, $k = 1$ gives only $\{0, 1\}$ estimates — useless for calibrated probabilities.

---

**Q25. Your kNN on product recommendations returns the same items for very different users. What went wrong?**

**Answer.** Common causes:
- **Global popular items are at the "center" of the embedding space.** Nearly every user's nearest items are popular ones by default.
- **Embeddings are poorly separated** — they collapse to a low-rank manifold where most users are near the same region.
- **No standardization / scaling across embedding dimensions.**
- **User embedding averaging** over a long history washes out preferences.

**Fixes:**
- Penalize popularity (e.g., subtract item popularity from similarity, or divide by log-popularity).
- Inspect embedding variance; if too low-rank, retrain with a richer loss (contrastive, triplet).
- Use recency-weighted or session-based user embeddings.
- Add diversity re-ranking (MMR).

---

**Q26. A kNN system's latency doubles after migrating from a 32-core CPU to a 64-core CPU. Why might that be?**

**Answer.** Several possibilities:
- **False sharing / cache contention:** more cores competing for the same cache lines in the shared index, causing slowdowns.
- **NUMA effects:** index memory is on one socket; cores on the other socket incur cross-socket traffic.
- **Index not thread-safe:** coarse locks serialize queries; more threads = more contention.
- **Batching regressions:** the previous system amortized per-query overhead better.

**Diagnosis:** profile with `perf` or a VTune-like tool; check `numactl` bindings; verify thread-safety of the ANN library; test with varying thread counts.

---

**Q27. You compare kNN with Euclidean vs Mahalanobis on a dataset. Mahalanobis is worse. Why?**

**Answer.** Likely causes:
- **Sample covariance is poorly estimated.** With $n$ small relative to $d$, $\hat\Sigma$ is noisy; $\hat\Sigma^{-1}$ even more so. Mahalanobis amplifies this noise.
- **Heavy tails / outliers** skew $\hat\Sigma$. Robust covariance (MCD) can help.
- **Classes have different covariances**, so the pooled $\hat\Sigma$ is not the right whitening for either.
- **Data is already standardized** and uncorrelated; Mahalanobis adds no information, only estimation noise.

**Fixes:** shrinkage (Ledoit-Wolf), class-conditional covariances (QDA-style), robust covariance.

---

**Q28. kNN predictions are much worse for queries at the "edge" of the feature space. Why?**

**Answer.** Edge queries' neighborhoods are **asymmetric** — all their neighbors lie to one side. In regression, this biases estimates toward the interior (neighbors have systematically different feature values than the query). In classification, classes that lie near the edge are underrepresented in the neighborhood.

**Fixes:**
- **Flag boundary queries** (large mean distance to neighbors) for abstention or manual review.
- **Local linear regression** (a step beyond kNN) corrects the bias by fitting a linear model to the $k$ neighbors; widely used in LOESS.
- **Augment training data** near the boundary (active learning).

---

**Q29. Your LSH-based retrieval has low recall on some queries but high on others. Why?**

**Answer.** LSH guarantees are probabilistic and depend on the query's *true distance* to its nearest neighbor. For queries whose true nearest neighbor is at distance $\le R$, recall is high (tuned with $p_1^L T$). For queries with no neighbor within $R$ but with a "somewhat close" neighbor in the $(R, cR]$ range, LSH may miss. Also, heavy-tailed data distributions yield many outlier queries with atypical neighbor structure.

**Fixes:**
- Increase $T$ (more tables).
- Widen $R$ in the LSH design (or use multi-probe LSH).
- Use data-adaptive partitioning (e.g., learned hashes, IVF with k-means centroids).
- Re-rank candidates with exact distance computation.

---

## Common Follow-ups & Probing

**P1. "You said Cover–Hart gives 1-NN within 2× the Bayes rate. Why is that bound tight, and what does it imply?"**

**Answer.** Tightness: consider a problem where the Bayes rate $R^* = 1/2 - \epsilon$ for small $\epsilon$. Then $2R^*(1 - R^*) \approx 1/2 - 2\epsilon^2$, close to $R^*$. For easy problems ($R^* \to 0$), the bound is $2R^* \to 0$; for hard problems (near random), the bound approaches $1/2$ — no worse than random, unsurprising. The implication: **1-NN's excess risk is bounded by a multiplicative factor of the Bayes rate, uniformly**. This is astonishing given no modeling assumptions. It is why 1-NN on good features is a formidable baseline.

---

**P2. "Why might you use squared Euclidean instead of Euclidean?"**

**Answer.** Squared Euclidean avoids the square root: cheaper. Since $f(x) = x^2$ is monotonic on $[0, \infty)$, neighbor rankings are identical — only the absolute values differ. For finding top-$k$, use squared Euclidean. Caveat: squared Euclidean is *not* a metric (fails triangle inequality: $d(\mathbf{a}, \mathbf{c})^2 \not\le d(\mathbf{a}, \mathbf{b})^2 + d(\mathbf{b}, \mathbf{c})^2$ in general), so metric-tree pruning must use the true Euclidean bound, or pre-square the bound carefully.

---

**P3. "How does distance weighting interact with $k$?"**

**Answer.** With strong distance weighting (e.g., Gaussian with small bandwidth, or inverse-squared-distance), increasing $k$ has diminishing effect because distant neighbors are down-weighted to near zero. In the limit, weighting makes $k$ almost irrelevant beyond a threshold. This is both a feature (robust to $k$) and a risk (bandwidth/weight scheme becomes the dominant hyperparameter). Best practice: CV jointly over $(k, \text{weight scheme})$.

---

**P4. "If I give you 1 billion 128-dim vectors and a 10ms latency budget, how do you build a kNN system?"**

**Answer.**
1. **Storage:** 1B × 128 × 4 bytes = 512 GB. Too big for RAM on commodity machines. Use **product quantization (PQ)**: compress each vector to ~16 bytes → 16 GB total, fits in RAM.
2. **Partitioning:** **IVF** (inverted file) with $\sim \sqrt{n}$ coarse centroids (~30k). Assign each vector to its nearest centroid.
3. **Index:** FAISS `IVF30000,PQ16` or similar. Build on a GPU box (hours).
4. **Query:** Compute query to each of 30k centroids ($O(kd)$); pick top `nprobe` (say 32); compute asymmetric distance with PQ codes within those cells. Total: ~10 ms on CPU, faster on GPU.
5. **Re-rank:** For top-100 candidates, compute exact distance on the full vectors (if available) to refine.
6. **Memory-tiered storage:** full vectors on SSD, PQ codes in RAM.
7. **Monitoring:** recall@10 on a labeled set, p99 latency, index freshness.

---

**P5. "What's the intuition behind the $\rho$ parameter in LSH?"**

**Answer.** $\rho = \log(1/p_1) / \log(1/p_2)$ measures how much better close points collide than far points. Good LSH families have small $\rho$: $p_1$ close to 1 (close points collide often), $p_2$ close to 0 (far points rarely). Query time and space scale as $n^\rho$. For optimal hash families:
- Euclidean / $L_2$: $\rho \to 1/c^2$ asymptotically.
- Hamming: $\rho = 1/c$.
- Angular / cosine: $\rho \to 1/c^2$ for small angles.

Smaller $\rho$ = more sub-linear query time. **The art of LSH is designing hash families with small $\rho$ for your metric.**

---

**P6. "Explain intuitively why Mahalanobis is 'Euclidean with learned scaling.'"**

**Answer.** Mahalanobis treats directions of high data variance as "cheap" to traverse — moving along the main spread of the data is unsurprising, so it contributes less to distance. Directions of low variance are "expensive" — deviating there means you are outside the typical data manifold. It is as if each direction has its own unit of measurement, calibrated to one standard deviation of data along that direction. This makes Mahalanobis scale- and rotation-aware in a way Euclidean is not.

---

**P7. "What's the difference between kNN and kernel density estimation?"**

**Answer.** Both are non-parametric methods based on neighbor information, but:
- **kNN density estimate:** $\hat p(\mathbf{x}) \propto k / (n V_k(\mathbf{x}))$, where $V_k(\mathbf{x})$ is the volume of the ball containing the $k$ nearest neighbors. Adapts locally: small volume in dense regions, large in sparse regions.
- **KDE:** $\hat p(\mathbf{x}) = \tfrac{1}{n h^d}\sum K((\mathbf{x} - \mathbf{x}_i)/h)$. Fixed bandwidth $h$ across the space; averages kernel contributions from all points.

kNN density is adaptive; KDE is uniform-bandwidth. For kNN classification, we can view it as ratio of class-conditional kNN densities — explicitly, $p(c \mid \mathbf{x}) \approx k_c / k$ where $k_c$ is the count of class-$c$ neighbors.

---

**P8. "How would you explain kNN's bias–variance trade-off to a non-technical stakeholder?"**

**Answer.** "Think of $k$ as how many opinions you consult before deciding. With $k = 1$, you trust the single most similar example — great if it's reliable, terrible if it's wrong. With $k = 100$, you aggregate many opinions, so one bad example can't mislead you, but you also lose sensitivity to local detail — you're averaging over a wide variety of cases. The right $k$ is the one that best balances trust in local detail (small $k$) against robustness to noise (large $k$). We tune this by simulating predictions on held-out data."

---

**P9. "Why is kNN bad at extrapolation?"**

**Answer.** Extrapolation requires predicting in regions with no training data. kNN's prediction is a combination of training labels; outside the training distribution, the "neighbors" are themselves far away and uninformative. A linear regression extrapolates (possibly badly, but via a formula); kNN cannot — it has no functional form to extend. In practice, this is sometimes a feature: a radius-based kNN with empty neighborhoods signals "I don't know," which parametric models rarely do honestly.

---

**P10. "What does it mean that kNN is 'universally consistent'?"**

**Answer.** As $n \to \infty$ with $k \to \infty$ and $k/n \to 0$, kNN's error converges to the Bayes error — in any measurable classification problem, under mild regularity. This is the **Stone (1977) theorem**. It says kNN is asymptotically optimal for any distribution. But "asymptotic" can mean "needs far more data than you have," and consistency does not imply efficiency — kNN's *rate* of convergence degrades with dimension.

---

**P11. "When does inverse-distance weighting become problematic?"**

**Answer.** Three main cases:
1. **Coincident points:** $d_i = 0$ → infinite weight. Mitigate with $w_i = 1/(d_i + \epsilon)$.
2. **Very high dimensions:** distances concentrate around a mean; $1/d_i$ is nearly constant, so weighting collapses to uniform. The scheme loses discriminative power.
3. **Noisy nearest neighbor:** if the closest point is a mislabeled outlier, inverse-distance amplifies its influence. Rank-based weighting (rank, not raw distance) is more robust.

---

**P12. "How would you use kNN for a system that continually learns?"**

**Answer.** kNN is naturally online — append new $(\mathbf{x}, y)$ to the reference set. Considerations:
- **Index updates:** HNSW supports incremental insertion; kd-trees do not (rebuild periodically).
- **Concept drift:** if the distribution shifts, old points become misleading. Use a sliding window, decaying weights (older points contribute less), or drift detection.
- **Storage growth:** $O(n)$ memory; eventually need condensation (Hart's CNN, Wilson's edited NN) or sampling.
- **Hyperparameter drift:** re-CV $k$ periodically.
- **Quality of new data:** online kNN is vulnerable to adversarial injection — any new point that is "near" queries of interest can poison results. Validate before admitting.

---

**P13. "What are 'prototype' or 'condensed' nearest neighbor methods?"**

**Answer.** Techniques to shrink the reference set while preserving kNN behavior.
- **CNN (Condensed NN, Hart 1968):** iteratively add points only if the current reference set mis-classifies them. Keeps decision-boundary-defining points; discards redundant interior points.
- **ENN (Edited NN, Wilson 1972):** remove points whose labels disagree with their kNN vote — cleans up noisy labels.
- **Tomek links:** pairs of nearest neighbors of different classes — useful for boundary smoothing.
- **Learning Vector Quantization (LVQ):** learn a small number of prototypes that reproduce kNN decisions.

Applications: reducing inference cost, denoising labels, visualizing decision boundaries.

---

**P14. "Why might kNN be competitive with deep learning in some settings?"**

**Answer.** When:
- **Features are already excellent** (pretrained embeddings). The "model" is encoded in the feature space; kNN needs no further parameters.
- **Sample size is small.** Deep networks overfit; kNN's non-parametric inductive bias (local smoothness) is a good match.
- **Interpretability is required.** Nearest neighbors provide instance-level explanations.
- **The task is retrieval, not classification.** kNN *is* retrieval.
- **Classes are extremely numerous or changing** (thousands of faces, evolving product catalogs). kNN handles open-ended class sets naturally.

In fact, kNN on frozen features is the standard evaluation protocol for self-supervised representation learning (e.g., SimCLR, DINO) — if the features are good, kNN is enough.

---

**P15. "What's the difference between supervised and semi-supervised kNN?"**

**Answer.** Standard kNN is supervised — all reference points have labels. Semi-supervised extensions:
- **Label propagation / spreading:** build a graph of nearest neighbors, propagate labels from labeled to unlabeled nodes via diffusion.
- **Self-training:** use confident kNN predictions on unlabeled data as pseudo-labels; retrain.
- **Transductive kNN:** treat the test set's features as also informing the neighborhood structure (e.g., for manifold-based classification).

Useful when labels are expensive but unlabeled data is plentiful.

---

# Closing Notes

## A minimal mental model of kNN

1. **Everything rides on the metric.** The distance function is the model; the rest is bookkeeping.
2. **$k$ is the regularizer.** Bias–variance in one dial.
3. **High dimensions kill naive kNN.** Embed, project, select — make the *effective* dimension small.
4. **Indexes trade exactness for speed.** Pick the exchange rate you can tolerate.
5. **kNN is a baseline that refuses to be beaten on good features.** Take it seriously.

## Recommended further reading

- Cover & Hart (1967), *Nearest neighbor pattern classification.*
- Stone (1977), *Consistent nonparametric regression.*
- Friedman, Bentley, Finkel (1977), *An algorithm for finding best matches in logarithmic expected time.*
- Omohundro (1989), *Five balltree construction algorithms.*
- Beyer, Goldstein, Ramakrishnan, Shaft (1999), *When is "nearest neighbor" meaningful?*
- Aggarwal, Hinneburg, Keim (2001), *On the surprising behavior of distance metrics in high dimensional space.*
- Datar, Immorlica, Indyk, Mirrokni (2004), *Locality-sensitive hashing scheme based on p-stable distributions.*
- Weinberger & Saul (2009), *Distance metric learning for large margin nearest neighbor classification (LMNN).*
- Malkov & Yashunin (2016), *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (HNSW).*
- Johnson, Douze, Jégou (2017), *Billion-scale similarity search with GPUs (FAISS).*

---

**[← Previous Chapter: Support Vector Machines](svm_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Naive Bayes →](naive_bayes_reference.md)**
