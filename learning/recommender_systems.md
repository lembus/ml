# Recommender Systems: A Comprehensive Learning Guide

---

## Table of Contents

1. [Collaborative Filtering](#1-collaborative-filtering)
2. [Content-Based Filtering](#2-content-based-filtering)
3. [Hybrid Methods](#3-hybrid-methods)
4. [Ranking Optimization](#4-ranking-optimization)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Scalability Challenges](#6-scalability-challenges)
7. [Cold Start Problems](#7-cold-start-problems)
8. [Interview Questions & Answers](#8-interview-questions--answers)

---

# 1. Collaborative Filtering

## 1.1 Motivation & Intuition

Suppose you run a bookstore with 10 million books and 50 million customers. Each customer has read only a tiny fraction of the catalog—perhaps 50 books out of 10 million. Your task: predict which unread books each customer would enjoy, so you can surface them prominently on the homepage.

One natural approach is to ask: "Which other customers have tastes similar to mine, and what did they enjoy that I haven't tried yet?" This is the core intuition behind **collaborative filtering** (CF). The word "collaborative" captures the idea that many users collectively contribute their preference signals (purchases, ratings, clicks) and the system exploits patterns across the entire user base to generate recommendations for each individual.

A concrete example clarifies this. Imagine three users and four movies:

| | Movie A | Movie B | Movie C | Movie D |
|---|---|---|---|---|
| Alice | 5 | 4 | ? | 1 |
| Bob | 5 | ? | 4 | 2 |
| Carol | ? | 3 | 5 | 5 |

Alice and Bob both loved Movie A and disliked Movie D. This pattern suggests they share similar taste. Bob rated Movie C a 4, and Alice hasn't seen it. CF predicts Alice would likely also enjoy Movie C (roughly a 4). Conversely, Carol's preferences diverge from Alice's—Carol liked Movie D, which Alice disliked—so Carol's love of Movie C is a weaker signal for Alice.

**Why CF matters in practice.** CF requires no knowledge about what the movies are about (genre, cast, plot). It works purely from observed interaction data. This is enormously powerful because:

- Content metadata is expensive to curate and often incomplete.
- User behavior reveals preferences that users themselves may not articulate.
- CF can surface serendipitous recommendations—items a user would never have searched for.

**Where CF is used.** Netflix, Spotify, Amazon, YouTube, and virtually every large-scale recommendation platform relies on some form of collaborative filtering as a foundational component.

**The fundamental data structure** in CF is the **user-item interaction matrix** $R \in \mathbb{R}^{m \times n}$, where $m$ is the number of users, $n$ is the number of items, and $R_{ui}$ records the interaction between user $u$ and item $i$. This matrix is almost always **extremely sparse**—typically more than 99% of entries are missing.

---

## 1.2 Conceptual Foundations

### 1.2.1 Memory-Based vs. Model-Based Methods

Collaborative filtering methods split into two broad families:

**Memory-based methods** (also called neighborhood methods) directly use the raw interaction matrix at prediction time. They find similar users (user-based CF) or similar items (item-based CF) and aggregate known ratings from the neighborhood.

**Model-based methods** learn a compact parametric model from the interaction data—for example, matrix factorization learns latent factor vectors for each user and item. At prediction time, the model generates predictions without needing the full raw matrix.

| Property | Memory-Based | Model-Based |
|---|---|---|
| Interpretability | High (you can point to specific neighbors) | Lower (latent factors are abstract) |
| Scalability | $O(m^2)$ or $O(n^2)$ for similarity computation | Typically $O(k(m+n))$ after training |
| Handling sparsity | Struggles with very sparse data | Better; latent factors regularize | 
| Cold start | Fails entirely for new users/items | Can partially address with side info |
| Training phase | None (lazy learner) | Required |

### 1.2.2 User-Based Collaborative Filtering

**Idea:** To predict how user $u$ would rate item $i$, find users most similar to $u$ who have rated $i$, and take a weighted average of their ratings.

Step-by-step:

1. **Compute pairwise user similarity.** For every pair of users $(u, v)$, compute a similarity score $\text{sim}(u, v)$ based on items they have both rated.
2. **Select a neighborhood.** For user $u$ and target item $i$, select the top-$k$ users most similar to $u$ who have rated $i$. Call this set $\mathcal{N}_k(u; i)$.
3. **Predict the rating.** Combine the neighbors' ratings using a weighted average, adjusting for each user's mean rating:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_k(u; i)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in \mathcal{N}_k(u; i)} |\text{sim}(u, v)|}
$$

The subtraction of $\bar{r}_u$ and $\bar{r}_v$ corrects for the fact that some users are systematically generous raters (average rating of 4.5) while others are harsh (average rating of 2.0). The prediction adds back $\bar{r}_u$ at the end, ensuring the prediction is on user $u$'s personal scale.

**Common similarity measures:**

- **Pearson correlation:** Measures linear correlation between users' ratings on co-rated items. Values range from $-1$ to $+1$.

$$
\text{sim}_{\text{Pearson}}(u, v) = \frac{\sum_{i \in I_{uv}}(r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}}(r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{i \in I_{uv}}(r_{vi} - \bar{r}_v)^2}}
$$

where $I_{uv}$ is the set of items rated by both $u$ and $v$.

- **Cosine similarity:** Treats each user's rating vector as a direction in item-space.

$$
\text{sim}_{\text{cosine}}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\| \cdot \|\mathbf{r}_v\|}
$$

Note: raw cosine does not correct for user mean; **adjusted cosine** subtracts the user mean first and is generally preferred.

**Practical issue: insufficient co-ratings.** If two users have rated only one or two items in common, the similarity estimate is unreliable. A common fix is to require a minimum number of co-rated items (e.g., at least 5) or to shrink the similarity toward zero when co-ratings are few:

$$
\text{sim}_{\text{shrunk}}(u, v) = \frac{|I_{uv}|}{|I_{uv}| + \lambda} \cdot \text{sim}(u, v)
$$

### 1.2.3 Item-Based Collaborative Filtering

**Idea:** Instead of finding similar users, find items similar to item $i$ that user $u$ has already rated, and predict based on those.

$$
\hat{r}_{ui} = \frac{\sum_{j \in \mathcal{N}_k(i; u)} \text{sim}(i, j) \cdot r_{uj}}{\sum_{j \in \mathcal{N}_k(i; u)} |\text{sim}(i, j)|}
$$

Here $\mathcal{N}_k(i; u)$ is the set of $k$ items most similar to $i$ that user $u$ has rated.

**Why item-based CF is often preferred in production:**

1. **Stability.** Item-item similarities change slowly over time (a movie is similar to the same set of movies regardless of new users joining). User-user similarities shift as users rate new items. This means item-item similarities can be precomputed and cached.
2. **Interpretability.** "We recommend Movie X because you liked Movie Y" is a natural explanation.
3. **Computational advantage.** Typically $n \ll m$ (fewer items than users), so computing item-item similarities is cheaper.

Amazon's original recommendation system was built on item-based collaborative filtering for precisely these reasons.

**Adjusted cosine for item similarity.** When computing item-item similarity, we must normalize for user effects. User $u$ might rate everything high. Adjusted cosine subtracts each user's mean:

$$
\text{sim}_{\text{adj-cos}}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2} \cdot \sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}
$$

where $U_{ij}$ is the set of users who rated both $i$ and $j$.

---

## 1.3 Matrix Factorization

### 1.3.1 The Core Idea

The interaction matrix $R \in \mathbb{R}^{m \times n}$ has $m \times n$ entries, but most are missing. Matrix factorization (MF) assumes that $R$ has low intrinsic rank—meaning user preferences are governed by a relatively small number of latent factors.

Concretely, we approximate:

$$
R \approx P Q^\top
$$

where $P \in \mathbb{R}^{m \times k}$ and $Q \in \mathbb{R}^{n \times k}$. Each user $u$ is described by a $k$-dimensional vector $\mathbf{p}_u \in \mathbb{R}^k$, and each item $i$ by $\mathbf{q}_i \in \mathbb{R}^k$. The predicted rating is:

$$
\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i = \sum_{f=1}^{k} p_{uf} \cdot q_{if}
$$

**Intuition for the latent factors.** In a movie recommender, one latent dimension might capture "action vs. drama," another might capture "mainstream vs. indie," another "humor level." A user who likes action movies would have a large value for the action dimension in $\mathbf{p}_u$, and a high-action movie would have a large value for the same dimension in $\mathbf{q}_i$. The dot product aggregates alignment across all dimensions.

But crucially, the factors are learned from data, not predefined. They emerge automatically during optimization and may not correspond to any human-interpretable concept.

### 1.3.2 SVD for Recommender Systems

Classical **Singular Value Decomposition** (SVD) decomposes any matrix $R \in \mathbb{R}^{m \times n}$ as:

$$
R = U \Sigma V^\top
$$

where $U \in \mathbb{R}^{m \times r}$ has orthonormal columns (left singular vectors), $\Sigma \in \mathbb{R}^{r \times r}$ is diagonal with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$, and $V \in \mathbb{R}^{n \times r}$ has orthonormal columns (right singular vectors). Here $r = \text{rank}(R)$.

**The Eckart-Young theorem** states that the best rank-$k$ approximation of $R$ (in Frobenius norm) is obtained by retaining only the $k$ largest singular values:

$$
R_k = U_k \Sigma_k V_k^\top
$$

This gives the optimal low-rank approximation in the least-squares sense.

**Problem with direct SVD for recommendations:** Classical SVD requires a *fully observed* matrix. Our interaction matrix is overwhelmingly sparse—most entries are unknown (not zero!). We cannot simply fill missing entries with zeros, because a zero might mean "user hates this item" or "user hasn't seen this item," and the distinction is critical.

**Solution: optimize only over observed entries.** Instead of decomposing the full matrix, we minimize reconstruction error on observed ratings only:

$$
\min_{P, Q} \sum_{(u,i) \in \mathcal{O}} \left(r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\right)^2 + \lambda \left(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2\right)
$$

where $\mathcal{O}$ is the set of observed user-item pairs and $\lambda$ controls L2 regularization to prevent overfitting. This is the formulation commonly called "SVD" in the recommender systems literature (e.g., the Netflix Prize), though it is technically not the same as the classical matrix SVD—it is an incomplete-data low-rank factorization.

### 1.3.3 Adding Biases

Raw dot products miss a crucial phenomenon: some users rate everything high, some items are universally popular. A more expressive model adds bias terms:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i
$$

where $\mu$ is the global mean rating, $b_u$ captures user $u$'s tendency to rate above or below the mean, and $b_i$ captures item $i$'s tendency to receive above- or below-average ratings.

The objective becomes:

$$
\min_{P, Q, b} \sum_{(u,i) \in \mathcal{O}} \left(r_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\right)^2 + \lambda\left(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2\right)
$$

In practice, biases alone explain a significant fraction of the variance. The latent factors capture the residual—the part of user preference that is genuinely about taste alignment.

### 1.3.4 SVD++ (Incorporating Implicit Feedback)

The key insight of **SVD++** (Koren, 2008) is that the mere act of a user rating an item—regardless of the rating value—carries information. A user who has rated 500 action movies is probably an action fan, even before looking at the specific ratings.

SVD++ augments the user representation with an implicit feedback term:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \left(\mathbf{p}_u + |\mathcal{I}_u|^{-1/2} \sum_{j \in \mathcal{I}_u} \mathbf{y}_j\right)^\top \mathbf{q}_i
$$

where $\mathcal{I}_u$ is the set of all items user $u$ has interacted with (regardless of rating value) and $\mathbf{y}_j \in \mathbb{R}^k$ is a learned vector for each item that captures the implicit signal from interacting with item $j$.

**Dissecting the formula:**

- $\mathbf{p}_u$: the explicit user factor, learned from ratings.
- $\sum_{j \in \mathcal{I}_u} \mathbf{y}_j$: the implicit user factor, accumulated from all items the user has interacted with.
- $|\mathcal{I}_u|^{-1/2}$: normalization factor. Without this, users with more interactions would have disproportionately large implicit factors. The square-root normalization (rather than $|\mathcal{I}_u|^{-1}$) was found empirically to work better than simple averaging.
- The combined vector $\mathbf{p}_u + |\mathcal{I}_u|^{-1/2} \sum_{j} \mathbf{y}_j$ serves as the enriched user representation.

SVD++ consistently outperformed basic MF in the Netflix Prize competition.

### 1.3.5 Alternating Least Squares (ALS)

ALS is an optimization algorithm for matrix factorization that exploits a key structural property: while the joint optimization over $P$ and $Q$ simultaneously is non-convex, fixing one and optimizing the other is a convex least-squares problem with a closed-form solution.

**Algorithm:**

1. Initialize $P$ and $Q$ randomly (or with small random values).
2. **Fix $Q$, solve for $P$:** For each user $u$, the optimal $\mathbf{p}_u$ satisfies:

$$
\mathbf{p}_u = \left(Q_u^\top Q_u + \lambda I\right)^{-1} Q_u^\top \mathbf{r}_u
$$

where $Q_u$ is the submatrix of $Q$ containing only rows for items user $u$ has rated, and $\mathbf{r}_u$ is the vector of those ratings. This is simply ridge regression.

3. **Fix $P$, solve for $Q$:** Symmetrically, for each item $i$:

$$
\mathbf{q}_i = \left(P_i^\top P_i + \lambda I\right)^{-1} P_i^\top \mathbf{r}_i
$$

4. Repeat steps 2–3 until convergence.

**Why ALS is popular:**

- **Parallelism.** When $Q$ is fixed, each user's $\mathbf{p}_u$ can be computed independently. This makes ALS embarrassingly parallel—ideal for distributed systems like Spark.
- **Implicit feedback.** ALS extends naturally to implicit feedback settings (Hu, Koren, Volinsky 2008), where the objective becomes a weighted regression over all user-item pairs (not just observed ones), with confidence weights $c_{ui} = 1 + \alpha \cdot r_{ui}$ for observed interactions.
- **No learning rate.** Unlike SGD, ALS has no learning rate to tune.

**Comparison: SGD vs. ALS for MF:**

| Property | SGD | ALS |
|---|---|---|
| Update granularity | Per-observation | Per-user or per-item |
| Parallelism | Harder (updates conflict) | Embarrassingly parallel |
| Implicit feedback | Tricky (need negative sampling) | Natural formulation |
| Convergence speed | Fast per iteration, many iterations | Slower per iteration, fewer iterations |
| Hyperparameters | Learning rate, schedule, momentum | Regularization weight only |

### 1.3.6 Derivation: SGD Updates for Biased MF

The objective is:

$$
L = \sum_{(u,i) \in \mathcal{O}} \left(r_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i\right)^2 + \lambda\left(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2\right)
$$

Define the prediction error for observation $(u, i)$:

$$
e_{ui} = r_{ui} - \hat{r}_{ui} = r_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i
$$

Take partial derivatives:

$$
\frac{\partial L}{\partial b_u} = -2e_{ui} + 2\lambda b_u \quad \Rightarrow \quad b_u \leftarrow b_u + \eta(e_{ui} - \lambda b_u)
$$

$$
\frac{\partial L}{\partial b_i} = -2e_{ui} + 2\lambda b_i \quad \Rightarrow \quad b_i \leftarrow b_i + \eta(e_{ui} - \lambda b_i)
$$

$$
\frac{\partial L}{\partial \mathbf{p}_u} = -2e_{ui}\mathbf{q}_i + 2\lambda \mathbf{p}_u \quad \Rightarrow \quad \mathbf{p}_u \leftarrow \mathbf{p}_u + \eta(e_{ui}\mathbf{q}_i - \lambda \mathbf{p}_u)
$$

$$
\frac{\partial L}{\partial \mathbf{q}_i} = -2e_{ui}\mathbf{p}_u + 2\lambda \mathbf{q}_i \quad \Rightarrow \quad \mathbf{q}_i \leftarrow \mathbf{q}_i + \eta(e_{ui}\mathbf{p}_u - \lambda \mathbf{q}_i)
$$

where $\eta$ is the learning rate. Each observed rating produces an update to all four parameter groups.

---

## 1.4 Neural Collaborative Filtering

### 1.4.1 Why Go Beyond Linear Dot Products?

Standard matrix factorization models the user-item interaction as $\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$—a linear function of the latent factors. This assumes that the interaction between any two latent dimensions is purely multiplicative and that the combination across dimensions is purely additive. But user preferences may involve more complex patterns:

- A user might like action movies AND comedy, but specifically hate action-comedies (a nonlinear interaction).
- Threshold effects: a user might love sci-fi but only if the production quality (another latent dimension) exceeds some threshold.

**Neural Collaborative Filtering (NCF)** (He et al., 2017) replaces the fixed dot product with a learned neural network:

$$
\hat{r}_{ui} = f_\theta(\mathbf{p}_u, \mathbf{q}_i)
$$

where $f_\theta$ is a neural network parameterized by $\theta$.

### 1.4.2 Architecture: Generalized Matrix Factorization (GMF)

The simplest NCF variant, GMF, generalizes the dot product with element-wise product and a learned output layer:

$$
\hat{y}_{ui} = \sigma\left(\mathbf{h}^\top (\mathbf{p}_u \odot \mathbf{q}_i)\right)
$$

where $\odot$ is element-wise (Hadamard) product, $\mathbf{h} \in \mathbb{R}^k$ is a learned weight vector, and $\sigma$ is the sigmoid function (for implicit feedback, producing a probability of interaction).

When $\mathbf{h} = \mathbf{1}$ (the all-ones vector) and $\sigma$ is the identity, this reduces to the standard dot product $\mathbf{p}_u^\top \mathbf{q}_i$.

### 1.4.3 Architecture: Deep Matrix Factorization (MLP)

A multi-layer perceptron (MLP) operates on the concatenation of user and item embeddings:

$$
\mathbf{z}_0 = \begin{bmatrix} \mathbf{p}_u \\ \mathbf{q}_i \end{bmatrix}
$$

$$
\mathbf{z}_1 = \text{ReLU}(W_1 \mathbf{z}_0 + \mathbf{b}_1)
$$

$$
\mathbf{z}_2 = \text{ReLU}(W_2 \mathbf{z}_1 + \mathbf{b}_2)
$$

$$
\vdots
$$

$$
\hat{y}_{ui} = \sigma(\mathbf{h}^\top \mathbf{z}_L)
$$

The MLP can learn arbitrary nonlinear interactions between user and item features. The typical architecture uses a "tower" pattern: decreasing layer widths (e.g., 128 → 64 → 32 → 16) to compress the combined representation into a prediction.

### 1.4.4 Architecture: NeuMF (Fusing GMF and MLP)

The full NCF model, called NeuMF, fuses GMF and MLP with separate embedding spaces:

$$
\hat{y}_{ui} = \sigma\left(\mathbf{h}^\top \begin{bmatrix} \mathbf{p}_u^{GMF} \odot \mathbf{q}_i^{GMF} \\ \text{MLP}(\mathbf{p}_u^{MLP}, \mathbf{q}_i^{MLP}) \end{bmatrix}\right)
$$

Having separate embeddings for GMF and MLP allows each pathway to learn independently. GMF captures the linear interaction signal; MLP captures nonlinear patterns. The fusion layer learns how to combine them.

**Training details for implicit feedback:**

The model is trained as a binary classification problem. Observed interactions $(u, i) \in \mathcal{O}$ are positive examples ($y_{ui} = 1$). Negative examples are sampled from unobserved interactions (negative sampling). The loss is binary cross-entropy:

$$
L = -\sum_{(u,i) \in \mathcal{O}} \log \hat{y}_{ui} - \sum_{(u,j) \notin \mathcal{O}, \text{sampled}} \log(1 - \hat{y}_{uj})
$$

A typical negative sampling ratio is 4:1 (four negatives per positive).

### 1.4.5 Pre-training Strategy

He et al. recommend pre-training: first train GMF and MLP separately, then initialize the fused NeuMF model with pretrained GMF and MLP weights, and fine-tune with a smaller learning rate. This avoids the difficulty of training the full model from random initialization.

---

## 1.5 Sequence-Aware and Session-Based Recommendations

### 1.5.1 Motivation

Standard CF methods treat each user's history as an unordered set of interactions. But the order matters:

- A user who watched *The Matrix*, then *Inception*, then *Interstellar* is on a sci-fi trajectory. Recommending *Arrival* makes more sense than recommending *The Notebook*.
- In e-commerce, within a single browsing session, a user might click on running shoes, then socks, then a water bottle—revealing a purchasing intent for running gear.

**Sequence-aware recommendations** model the temporal order of interactions to predict what a user will want next.

### 1.5.2 Session-Based Recommendations with RNNs (GRU4Rec)

**GRU4Rec** (Hidasi et al., 2016) was among the first to apply recurrent neural networks to session-based recommendation. The setting: anonymous browsing sessions (no persistent user IDs), where the model must predict the next item click from the sequence of items viewed so far in the session.

**Architecture:**

1. Each item $i$ has a learned embedding $\mathbf{e}_i \in \mathbb{R}^d$.
2. The sequence of clicked items $(i_1, i_2, \ldots, i_t)$ in a session is fed through a GRU:

$$
\mathbf{h}_t = \text{GRU}(\mathbf{e}_{i_t}, \mathbf{h}_{t-1})
$$

3. The hidden state $\mathbf{h}_t$ serves as the session representation at time $t$.
4. A score for each candidate item $j$ is computed as:

$$
s_j = \mathbf{h}_t^\top \mathbf{e}_j
$$

5. The scores are converted to a ranking over all items.

**Training with ranking loss.** GRU4Rec uses a pairwise ranking loss (BPR-like or TOP1 loss) rather than cross-entropy, because the task is ranking, not classification:

$$
L_{\text{BPR}} = -\frac{1}{N_s} \sum_{j=1}^{N_s} \log \sigma(s_{i_{t+1}} - s_j)
$$

where $s_{i_{t+1}}$ is the score of the true next item and $s_j$ are scores of sampled negative items.

**Session-parallel mini-batches.** Since sessions vary in length, GRU4Rec introduces a clever training scheme: it processes multiple sessions in parallel, advancing one step per session per batch. When a session ends, it is replaced by the next session in the dataset, and the corresponding hidden state is reset.

### 1.5.3 Self-Attention for Sequential Recommendation (SASRec)

**SASRec** (Kang and McAuley, 2018) replaces RNNs with the Transformer self-attention mechanism:

1. The user's interaction sequence $(i_1, \ldots, i_t)$ is embedded: $\mathbf{E} = [\mathbf{e}_{i_1}, \ldots, \mathbf{e}_{i_t}]$.
2. Learnable positional embeddings $\mathbf{P}$ are added: $\hat{\mathbf{E}} = \mathbf{E} + \mathbf{P}$.
3. Self-attention with causal masking is applied (each position can only attend to previous positions):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
$$

where $M$ is a mask with $M_{ij} = -\infty$ for $j > i$.

4. The output at position $t$ is the representation for predicting item $i_{t+1}$.
5. Prediction: $s_j = \mathbf{f}_t^\top \mathbf{e}_j$ where $\mathbf{f}_t$ is the final representation at position $t$.

**Advantages over GRU4Rec:**

- **Parallelism.** Self-attention processes all positions simultaneously during training (no sequential dependency).
- **Long-range dependencies.** Attention can directly connect any two positions, whereas RNNs must propagate information step by step.
- **Adaptive weighting.** The model learns which past interactions are most relevant for the current prediction, rather than relying on the GRU's fixed gating mechanism.

### 1.5.4 BERT4Rec

**BERT4Rec** (Sun et al., 2019) adapts the BERT masked language model approach. Instead of only predicting the next item (left-to-right), it randomly masks items in the sequence and trains the model to predict masked items from surrounding context (bidirectional). This provides richer training signal because every position can serve as a training target.

At inference time, BERT4Rec appends a [MASK] token to the end of the sequence and predicts it—effectively predicting the next item.

---

## 1.6 Worked Example: Biased Matrix Factorization with SGD

**Setup:** 3 users, 4 items, rank $k = 2$, $\lambda = 0.01$, $\eta = 0.01$.

Observed ratings:

| | Item 1 | Item 2 | Item 3 | Item 4 |
|---|---|---|---|---|
| User 1 | 5 | 3 | ? | 1 |
| User 2 | 4 | ? | ? | 1 |
| User 3 | ? | 1 | ? | 5 |

**Step 1: Compute global mean.**

$\mu = \frac{5 + 3 + 1 + 4 + 1 + 1 + 5}{7} = \frac{20}{7} \approx 2.857$

**Step 2: Initialize parameters.**

$$
b_u = [0, 0, 0], \quad b_i = [0, 0, 0, 0]
$$

$$
P = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.1 \\ 0.2 & 0.4 \end{bmatrix}, \quad Q = \begin{bmatrix} 0.4 & 0.1 \\ 0.2 & 0.3 \\ 0.1 & 0.2 \\ 0.3 & 0.5 \end{bmatrix}
$$

**Step 3: Process observation $(u=1, i=1, r_{11} = 5)$.**

Predict: $\hat{r}_{11} = \mu + b_1 + b_1 + \mathbf{p}_1^\top \mathbf{q}_1 = 2.857 + 0 + 0 + (0.1)(0.4) + (0.2)(0.1) = 2.857 + 0.06 = 2.917$

Error: $e_{11} = 5 - 2.917 = 2.083$

Update biases:
- $b_{u_1} \leftarrow 0 + 0.01(2.083 - 0.01 \cdot 0) = 0.02083$
- $b_{i_1} \leftarrow 0 + 0.01(2.083 - 0.01 \cdot 0) = 0.02083$

Update factors:
- $\mathbf{p}_1 \leftarrow [0.1, 0.2] + 0.01([2.083 \cdot 0.4, 2.083 \cdot 0.1] - 0.01 \cdot [0.1, 0.2])$
  $= [0.1, 0.2] + 0.01([0.833, 0.208] - [0.001, 0.002])$
  $= [0.1, 0.2] + [0.00832, 0.00206]$
  $= [0.10832, 0.20206]$
- $\mathbf{q}_1 \leftarrow [0.4, 0.1] + 0.01([2.083 \cdot 0.1, 2.083 \cdot 0.2] - 0.01 \cdot [0.4, 0.1])$
  $= [0.4, 0.1] + 0.01([0.2083, 0.4166] - [0.004, 0.001])$
  $= [0.4, 0.1] + [0.00204, 0.00416]$
  $= [0.40204, 0.10416]$

This process repeats for every observed rating, cycling through the data for multiple epochs until the loss converges. Over many epochs, the biases absorb the user/item-level effects, and the latent factors converge to capture the taste-alignment signal.

After training, to predict the missing $\hat{r}_{13}$ (User 1, Item 3):

$$
\hat{r}_{13} = \mu + b_{u_1} + b_{i_3} + \mathbf{p}_1^\top \mathbf{q}_3
$$

---

## 1.7 Relevance to ML Practice

**Where CF is used:**

- **Rating prediction.** Predicting how a user would rate an unrated item (Netflix star ratings, product reviews). MF methods dominate here.
- **Top-N recommendation.** Ranking all items and surfacing the top $N$ to the user. This is the most common production use case. Implicit feedback models (ALS for implicit data, BPR, NCF) are preferred.
- **Candidate generation.** In two-stage recommendation architectures (e.g., YouTube), the first stage uses a lightweight CF model to retrieve a few hundred candidate items from millions; the second stage re-ranks them with a more complex model.

**When to use CF:**

- Sufficient user interaction data exists.
- Users have overlapping tastes (the matrix has enough structure to learn from).
- Content metadata is unavailable or unreliable.

**When NOT to use CF:**

- New platform with very few users/interactions (cold start).
- Highly specialized domain where item content is critical (e.g., legal document recommendations).
- When explainability from item features is required.

**Trade-offs:**

- **Bias-variance:** Higher $k$ (more latent factors) increases model capacity but risks overfitting. Regularization $\lambda$ controls this.
- **Computational cost:** MF training is $O(|\mathcal{O}| \cdot k)$ per epoch for SGD; ALS is more expensive per step but converges faster. Neural methods are costlier but more expressive.
- **Implicit vs. explicit feedback:** Explicit ratings are higher quality but rare. Implicit signals (clicks, purchases) are abundant but noisy (a click doesn't mean the user liked the item).

---

## 1.8 Common Pitfalls & Misconceptions

**Pitfall 1: Treating missing entries as zeros.** This is the most common beginner mistake. A missing rating is not a zero—it means the user hasn't interacted with the item. Treating missing as zero in MF will push the model to predict zeros for unobserved pairs, which is the opposite of what we want (we want to predict positive ratings for items users would like).

**Pitfall 2: Confusing recommender "SVD" with linear algebra SVD.** The recommender systems community uses "SVD" loosely to refer to any low-rank matrix factorization optimized on observed entries. Classical SVD requires a complete matrix. The optimization problem and algorithm are different.

**Pitfall 3: Ignoring popularity bias.** CF methods tend to recommend popular items because they appear in many users' histories and thus accumulate strong factor representations. This creates a feedback loop: popular items are recommended more, get more interactions, become even more popular. Corrective measures include popularity-based negative sampling and re-ranking for diversity.

**Pitfall 4: Evaluating on random splits.** In temporal settings, splitting data randomly for train/test ignores the fact that we can only predict the future from the past. Using chronological splits is essential for realistic evaluation.

**Pitfall 5: Overfitting with too many latent factors.** A 500-dimensional MF model on a dataset with 1,000 users will memorize the training data. The number of latent factors $k$ should be much smaller than the number of users and items.

**Pitfall 6: Assuming NCF always beats MF.** Rendle et al. (2020) showed that with proper tuning, dot-product MF can match or outperform MLP-based NCF. The neural architecture adds expressiveness but also optimization difficulty. The claim that neural always wins is not well-supported empirically.

**Pitfall 7: Session-based models without enough session data.** GRU4Rec and SASRec need many sessions with at least a few interactions each. If most sessions contain only one click, these models have insufficient sequential signal and degrade to popularity-based recommendations.

---

# 2. Content-Based Filtering

## 2.1 Motivation & Intuition

Collaborative filtering relies entirely on user-item interaction patterns. But what if a new item enters the catalog with zero interactions? CF cannot recommend it. What if a user has rated only three items—there isn't enough interaction history for CF to find similar users.

**Content-based filtering (CBF)** takes a different approach: it recommends items whose *attributes* (content features) match the user's demonstrated preferences. If you've read three mystery novels and rated them highly, CBF will recommend other mystery novels—based on the books' genre, author, writing style, and other content features—regardless of what other users have done.

**Concrete example.** Suppose you're building a news article recommender. Each article has features: topic (politics, sports, tech), entities mentioned (specific people, companies), writing style (opinion, investigative, brief). A user who consistently reads long-form investigative tech articles should see more of those. CBF can surface a brand-new article the moment it's published, because the recommendation depends on the article's content, not on other users' reactions to it.

**Key advantage over CF:** CBF does not suffer from the new-item cold start problem. As long as we have the item's features, we can recommend it.

**Key limitation:** CBF cannot recommend items outside a user's established preference profile. If a user has only read sci-fi, CBF will never suggest a cookbook, even if the user would enjoy it. This lack of *serendipity* is a fundamental weakness.

---

## 2.2 Conceptual Foundations

### 2.2.1 Item Representation Learning

The quality of content-based filtering is determined entirely by the quality of item representations. The system needs a feature vector $\mathbf{x}_i \in \mathbb{R}^d$ for each item $i$ that captures the item's salient attributes.

**Traditional approaches:**

- **Bag of words / TF-IDF:** For text-based items (articles, product descriptions), represent each item as a vector of term frequencies weighted by inverse document frequency. Each dimension corresponds to a word in the vocabulary.

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log \frac{N}{\text{DF}(t)}
$$

where $\text{TF}(t, d)$ is the frequency of term $t$ in document $d$, $N$ is the total number of documents, and $\text{DF}(t)$ is the number of documents containing term $t$.

- **Categorical features:** For structured data (genre, color, brand), use one-hot or multi-hot encodings.
- **Engineered features:** Domain-specific attributes extracted manually (e.g., beats per minute for music, calorie count for recipes).

**Modern approaches (learned representations):**

- **Word2Vec / Doc2Vec:** Learn dense vector representations of items from text.
- **Pretrained language models (BERT, Sentence-BERT):** Encode item text descriptions into dense vectors that capture semantic meaning.
- **Image embeddings (ResNet, CLIP):** For visual items, extract feature vectors from pretrained vision models. A fashion recommender might use the penultimate layer of a ResNet trained on fashion images.
- **Multi-modal fusion:** Combine text, image, and structured features into a single representation.

### 2.2.2 User Profiling

The user profile $\mathbf{u} \in \mathbb{R}^d$ summarizes the user's preferences in the same feature space as items.

**Simple averaging.** Given items $\mathcal{I}_u^+$ that user $u$ liked:

$$
\mathbf{u} = \frac{1}{|\mathcal{I}_u^+|} \sum_{i \in \mathcal{I}_u^+} \mathbf{x}_i
$$

**Weighted averaging.** Weight each item by the user's rating (or recency):

$$
\mathbf{u} = \frac{\sum_{i \in \mathcal{I}_u} r_{ui} \cdot \mathbf{x}_i}{\sum_{i \in \mathcal{I}_u} r_{ui}}
$$

**Learned user profiles.** Train a classifier or regression model per user: given an item's feature vector $\mathbf{x}_i$, predict the user's rating. Each user's model parameters serve as the user profile.

### 2.2.3 Prediction

The predicted relevance of item $i$ for user $u$ is computed via a similarity function:

$$
\text{score}(u, i) = \text{sim}(\mathbf{u}, \mathbf{x}_i)
$$

---

## 2.3 Similarity Measures

### 2.3.1 Cosine Similarity

$$
\text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|} = \frac{\sum_{j=1}^d a_j b_j}{\sqrt{\sum_{j=1}^d a_j^2} \cdot \sqrt{\sum_{j=1}^d b_j^2}}
$$

**Range:** $[-1, 1]$ for general vectors, $[0, 1]$ for non-negative vectors (TF-IDF, counts).

**When to use:** When magnitude doesn't matter—only the direction (relative proportions of features). A long article and a short article about the same topic should be equally similar to a user who likes that topic.

**Key property:** Invariant to scaling. $\text{cos}(c \cdot \mathbf{a}, \mathbf{b}) = \text{cos}(\mathbf{a}, \mathbf{b})$ for any $c > 0$.

### 2.3.2 Jaccard Similarity

For binary or set-valued features:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Range:** $[0, 1]$.

**When to use:** When items are described by sets of tags, categories, or keywords. For example, two movies share genres {Action, Sci-Fi} and {Sci-Fi, Thriller}: intersection = {Sci-Fi}, union = {Action, Sci-Fi, Thriller}, $J = 1/3$.

**Limitation:** Ignores feature weights—all set elements are equally important.

### 2.3.3 Euclidean Distance (and its use as similarity)

$$
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{j=1}^d (a_j - b_j)^2}
$$

To convert distance to similarity:

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{1}{1 + d(\mathbf{a}, \mathbf{b})}
$$

**When to use:** When magnitude matters. If one user profile has high values for action features and another has moderate values, Euclidean distance captures this magnitude difference, while cosine similarity might miss it.

**Limitation:** Sensitive to feature scaling. If one feature ranges in $[0, 1]$ and another in $[0, 1000]$, the second feature dominates the distance. Always normalize features before using Euclidean distance.

### 2.3.4 Comparison Summary

| Measure | Best For | Invariant to Scale? | Handles Sets? |
|---|---|---|---|
| Cosine | Text vectors (TF-IDF, embeddings) | Yes | No |
| Jaccard | Tags, categories, binary features | N/A | Yes |
| Euclidean | Numeric features where magnitude matters | No | No |

---

## 2.4 Worked Example: TF-IDF Content-Based Filtering

**Setup:** 4 articles, vocabulary of 5 terms. User has read and liked articles 1 and 2.

| Term | Art.1 | Art.2 | Art.3 | Art.4 | DF | IDF = log(4/DF) |
|---|---|---|---|---|---|---|
| neural | 3 | 0 | 2 | 0 | 2 | 0.693 |
| network | 2 | 0 | 1 | 0 | 2 | 0.693 |
| climate | 0 | 4 | 0 | 3 | 2 | 0.693 |
| model | 1 | 1 | 1 | 1 | 4 | 0.000 |
| policy | 0 | 2 | 0 | 2 | 2 | 0.693 |

**Step 1: Compute TF-IDF vectors.**

Article 1: $[3 \times 0.693,\; 2 \times 0.693,\; 0,\; 1 \times 0,\; 0] = [2.079, 1.386, 0, 0, 0]$

Article 2: $[0,\; 0,\; 4 \times 0.693,\; 0,\; 2 \times 0.693] = [0, 0, 2.772, 0, 1.386]$

Article 3: $[2 \times 0.693,\; 1 \times 0.693,\; 0,\; 0,\; 0] = [1.386, 0.693, 0, 0, 0]$

Article 4: $[0,\; 0,\; 3 \times 0.693,\; 0,\; 2 \times 0.693] = [0, 0, 2.079, 0, 1.386]$

**Step 2: Build user profile** (average of liked articles):

$\mathbf{u} = \frac{1}{2}([2.079, 1.386, 0, 0, 0] + [0, 0, 2.772, 0, 1.386]) = [1.040, 0.693, 1.386, 0, 0.693]$

**Step 3: Compute cosine similarity with unseen articles.**

For Article 3: $\mathbf{x}_3 = [1.386, 0.693, 0, 0, 0]$

$\text{cos}(\mathbf{u}, \mathbf{x}_3) = \frac{1.040 \times 1.386 + 0.693 \times 0.693}{\sqrt{1.040^2 + 0.693^2 + 1.386^2 + 0.693^2} \times \sqrt{1.386^2 + 0.693^2}}$

Numerator: $1.441 + 0.480 = 1.921$

$\|\mathbf{u}\| = \sqrt{1.082 + 0.480 + 1.921 + 0.480} = \sqrt{3.963} = 1.991$

$\|\mathbf{x}_3\| = \sqrt{1.921 + 0.480} = \sqrt{2.401} = 1.550$

$\text{cos}(\mathbf{u}, \mathbf{x}_3) = \frac{1.921}{1.991 \times 1.550} = \frac{1.921}{3.086} = 0.623$

For Article 4: $\mathbf{x}_4 = [0, 0, 2.079, 0, 1.386]$

$\text{cos}(\mathbf{u}, \mathbf{x}_4) = \frac{1.386 \times 2.079 + 0.693 \times 1.386}{1.991 \times \sqrt{2.079^2 + 1.386^2}}$

Numerator: $2.882 + 0.960 = 3.842$

$\|\mathbf{x}_4\| = \sqrt{4.322 + 1.921} = \sqrt{6.243} = 2.499$

$\text{cos}(\mathbf{u}, \mathbf{x}_4) = \frac{3.842}{1.991 \times 2.499} = \frac{3.842}{4.975} = 0.772$

**Result:** Article 4 (similarity 0.772) is ranked above Article 3 (0.623). Since the user liked both a "neural networks" article and a "climate policy" article, Article 4 (climate + policy) is more aligned with the blended profile. Article 3 (neural networks only) matches only half of the user's interests.

---

## 2.5 Relevance to ML Practice

**Where CBF is used:**

- **New item recommendations.** When items enter the catalog continuously (news articles, new products), CBF can recommend them immediately.
- **Niche domains.** In domains with few users (specialized professional tools, academic papers), CF cannot build reliable co-occurrence patterns.
- **Explainability.** "We recommended this because you like sci-fi movies starring Tom Hanks" is directly derived from content features.

**When NOT to use CBF:**

- When content features are poor or unavailable (e.g., music—audio features alone poorly capture taste).
- When serendipity is critical—CBF creates "filter bubbles."
- When user preferences are complex and poorly captured by item features.

**Trade-offs:**

- **No cold-start for items** (major advantage over CF) vs. **no serendipity** (major limitation).
- **Feature engineering cost:** The quality of CBF is bounded by the quality of item representations.
- **Over-specialization:** Users get stuck in narrow preference niches.

---

## 2.6 Common Pitfalls & Misconceptions

**Pitfall 1: Confusing "content-based" with "using features."** Many modern systems use content features as side information in a CF framework (feature-augmented MF). This is a hybrid approach, not pure CBF. Pure CBF does not use any inter-user information.

**Pitfall 2: Averaging diverse preferences destroys signal.** If a user likes both horror and romantic comedy, the averaged user profile will be somewhere between the two genres—and will match neither well. Clustering the user's interests into multiple profiles or using attention mechanisms helps.

**Pitfall 3: Ignoring feature quality.** Garbage in, garbage out. If TF-IDF features are built on poorly tokenized text, or if item metadata has errors, CBF recommendations will be poor.

**Pitfall 4: Not updating user profiles.** As a user's tastes evolve, the profile must be updated. A static profile based on early interactions becomes stale. Recency weighting or decay factors help.

---

# 3. Hybrid Methods

## 3.1 Motivation & Intuition

We've seen that collaborative filtering excels when interaction data is plentiful but struggles with cold start, while content-based filtering handles new items well but misses collaborative patterns and serendipity. The natural question: can we get the best of both?

**Hybrid recommender systems** combine collaborative and content-based signals (and potentially other sources like contextual, social, or knowledge-graph information) to overcome the individual limitations of each approach.

Nearly all production recommender systems are hybrids. Netflix, YouTube, Spotify, and Amazon all combine multiple signals in their recommendation pipelines.

---

## 3.2 Conceptual Foundations

Burke (2002) proposed a taxonomy of hybrid methods:

### 3.2.1 Weighted Hybridization

Run CF and CBF independently, produce separate recommendation scores, and combine them:

$$
\text{score}_{\text{hybrid}}(u, i) = \alpha \cdot \text{score}_{\text{CF}}(u, i) + (1 - \alpha) \cdot \text{score}_{\text{CBF}}(u, i)
$$

where $\alpha \in [0, 1]$ is a mixing weight.

**Adaptive weighting.** The optimal $\alpha$ may depend on the user. For a new user with few interactions, CBF is more reliable, so $\alpha$ should be lower. For a veteran user, CF is more powerful, so $\alpha$ should be higher. This can be learned:

$$
\alpha(u) = \sigma\left(\beta_0 + \beta_1 \cdot \log|\mathcal{I}_u|\right)
$$

where $|\mathcal{I}_u|$ is the number of user $u$'s interactions and $\sigma$ is the sigmoid function.

**Advantages:** Simple; each component can be developed, tuned, and debugged independently.

**Limitations:** The two systems don't share information during training—they can't learn from each other's strengths.

### 3.2.2 Switching Hybridization

Use different recommenders for different contexts:

$$
\text{score}_{\text{hybrid}}(u, i) = \begin{cases} \text{score}_{\text{CBF}}(u, i) & \text{if } |\mathcal{I}_u| < \tau \\ \text{score}_{\text{CF}}(u, i) & \text{otherwise} \end{cases}
$$

A threshold $\tau$ on user activity determines which system to trust. Alternatively, the switching criterion might be based on the confidence of each system's predictions.

**Advantages:** Explicitly addresses cold start by routing new users to CBF.

**Limitations:** Abrupt transitions can cause jarring changes in recommendation style. The threshold $\tau$ requires tuning.

### 3.2.3 Feature Combination (Feature Augmentation)

Treat collaborative signals as additional features for a content-based model, or vice versa.

**Example: Content features in MF.** Standard MF learns $\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i$. Feature-augmented MF adds content features:

$$
\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i + \mathbf{p}_u^\top W \mathbf{x}_i + \mathbf{v}^\top \mathbf{x}_i
$$

where $\mathbf{x}_i$ is the item's content feature vector, $W$ is a learned projection, and $\mathbf{v}$ captures a direct content effect.

**Example: Factorization Machines (FM).** FMs generalize MF to incorporate arbitrary features:

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{j=1}^{p} w_j x_j + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_j x_{j'}
$$

where $\mathbf{x}$ is a combined feature vector encoding user ID (one-hot), item ID (one-hot), and any content features. The pairwise interaction term $\langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle$ captures the interaction between any two features using factorized parameters.

FMs are extremely powerful because:
- Setting $\mathbf{x} = [\text{user one-hot}, \text{item one-hot}]$ recovers standard MF.
- Adding content features seamlessly extends the model.
- The factorized interaction makes it feasible even with millions of features.

**Deep hybrid models.** Modern architectures like **Wide & Deep** (Cheng et al., 2016) and **DeepFM** (Guo et al., 2017) combine memorization (wide linear model or FM capturing explicit feature interactions) with generalization (deep network learning implicit feature interactions). The wide component memorizes specific co-occurrences (e.g., "users who installed app X also installed app Y"); the deep component generalizes to unseen combinations through learned embeddings.

### 3.2.4 Meta-Level Hybridization

One system's output serves as the input to another. For example, CBF builds a user profile from content features; this profile is then used as features for a CF model.

### 3.2.5 Cascade Hybridization

A coarse-grained recommender generates a shortlist; a fine-grained recommender re-ranks it. This is extremely common in production:

1. **Candidate generation:** Fast, simple models (e.g., ALS matrix factorization, approximate nearest neighbors) retrieve ~1000 candidates from millions of items.
2. **Ranking:** A complex model (e.g., deep neural network combining collaborative, content, and contextual features) scores and ranks the ~1000 candidates.
3. **Re-ranking:** Business rules, diversity constraints, and freshness adjustments produce the final displayed list.

---

## 3.3 Mathematical Formulation of Factorization Machines

Consider a feature vector $\mathbf{x} \in \mathbb{R}^p$ where $p$ may be very large (millions of features for one-hot encoded user + item + side features).

**Linear model:** $\hat{y} = w_0 + \sum_j w_j x_j$

This misses all feature interactions.

**Full pairwise interaction model:** $\hat{y} = w_0 + \sum_j w_j x_j + \sum_{j<j'} w_{jj'} x_j x_{j'}$

This has $O(p^2)$ interaction parameters—infeasible for $p$ in the millions.

**FM insight: factorize the interaction matrix.** Instead of learning a separate weight $w_{jj'}$ for each pair, factorize the interaction weight matrix $W \in \mathbb{R}^{p \times p}$ as $W \approx V V^\top$ where $V \in \mathbb{R}^{p \times k}$:

$$
\hat{y} = w_0 + \sum_{j=1}^{p} w_j x_j + \sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_j x_{j'}
$$

The interaction term can be computed in $O(pk)$ instead of $O(p^2)$ using the identity:

$$
\sum_{j<j'} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_j x_{j'} = \frac{1}{2}\left[\left(\sum_{j=1}^p \mathbf{v}_j x_j\right)^2 - \sum_{j=1}^p (\mathbf{v}_j x_j)^2\right]
$$

where the squaring and subtraction are element-wise over the $k$ latent dimensions.

**Derivation of the efficient computation:**

$$
\sum_{j=1}^{p}\sum_{j'=1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'}\rangle x_j x_{j'} = \left\langle \sum_j \mathbf{v}_j x_j, \sum_{j'} \mathbf{v}_{j'} x_{j'}\right\rangle = \left\|\sum_j \mathbf{v}_j x_j\right\|^2
$$

This double sum includes $j = j'$ terms (diagonal), which contribute $\sum_j \langle \mathbf{v}_j, \mathbf{v}_j \rangle x_j^2 = \sum_j \|\mathbf{v}_j\|^2 x_j^2$.

The off-diagonal sum (including both $(j, j')$ and $(j', j)$) equals the full sum minus the diagonal, and the upper-triangular sum is half of that:

$$
\sum_{j<j'} \langle \mathbf{v}_j, \mathbf{v}_{j'}\rangle x_j x_{j'} = \frac{1}{2}\left(\left\|\sum_j \mathbf{v}_j x_j\right\|^2 - \sum_j \|\mathbf{v}_j\|^2 x_j^2\right)
$$

---

## 3.4 Common Pitfalls & Misconceptions

**Pitfall 1: Thinking hybrid = better automatically.** A poorly tuned hybrid can underperform a well-tuned single approach. The combination introduces additional hyperparameters and potential error modes.

**Pitfall 2: Double-counting signals.** If both CF and CBF components use item popularity as a signal, the hybrid amplifies the popularity bias.

**Pitfall 3: Ignoring calibration.** CF scores and CBF scores may be on completely different scales. Before weighted combination, they must be normalized (e.g., min-max scaling, z-score normalization, or conversion to ranks).

**Pitfall 4: Overcomplicating the cascade.** Adding more stages doesn't always help—each stage introduces latency, and errors in early stages propagate.

---

# 4. Ranking Optimization

## 4.1 Motivation & Intuition

When a user opens Netflix, they see a ranked list of shows. The user interacts primarily with the top few items—scrolling past the first 10 results is rare. This means the **order** of items matters far more than the predicted rating scores themselves.

Consider two systems:
- System A predicts: [Item1: 4.2, Item2: 4.0, Item3: 3.9, Item4: 3.8] — correctly ranks a good item first.
- System B predicts: [Item1: 3.1, Item3: 3.0, Item2: 2.9, Item4: 2.8] — mis-ranks items 2 and 3.

Even though System B's predictions are "closer" to the true ratings in absolute terms, System A is better because it gets the top-of-list right. **Ranking optimization** directly optimizes the order of items, rather than minimizing prediction error.

---

## 4.2 Conceptual Foundations

### 4.2.1 Three Paradigms

Ranking methods are categorized by how they frame the learning problem:

**Pointwise:** Treat each item independently. Predict a relevance score for each item, then sort by predicted score. This reduces ranking to regression or classification.

**Pairwise:** Consider pairs of items. For each pair $(i, j)$ where item $i$ should rank above item $j$, learn a model that scores $i$ higher than $j$. The model optimizes the number of correctly ordered pairs.

**Listwise:** Consider the entire ranked list as the unit of optimization. Directly optimize a list-level metric (e.g., NDCG) or optimize a loss defined over the permutation.

---

## 4.3 Pointwise Approach

**Formulation:** Given user $u$ and item $i$ with relevance label $y_{ui}$ (e.g., rating, click indicator), minimize a standard prediction loss:

$$
L_{\text{pointwise}} = \sum_{u} \sum_{i \in \mathcal{I}_u} \ell\left(y_{ui}, f(u, i; \theta)\right)
$$

where $\ell$ is squared error (regression), cross-entropy (classification), or any standard loss.

**Why this is suboptimal for ranking:** Pointwise losses weight all prediction errors equally. But an error at the top of the ranking (confusing the 1st and 2nd item) is much worse than an error at position 50 vs. 51 for user experience. Pointwise methods don't capture this position-dependent importance.

---

## 4.4 Pairwise Approach: BPR

### 4.4.1 Bayesian Personalized Ranking (BPR)

**BPR** (Rendle et al., 2009) is the dominant pairwise ranking framework for implicit feedback.

**Setting:** We observe only positive interactions (clicks, purchases). We have no explicit negative feedback. BPR makes the assumption: items a user has interacted with should be ranked above items the user has not interacted with.

**Notation:** For user $u$, let $\mathcal{I}_u^+$ be items the user has interacted with and $\mathcal{I}_u^-$ be items the user has not interacted with. We want:

$$
\hat{r}_{ui} > \hat{r}_{uj} \quad \text{for all } i \in \mathcal{I}_u^+, \; j \in \mathcal{I}_u^-
$$

**Optimization criterion:** BPR maximizes the posterior probability of the model parameters given the observed pairwise preferences:

$$
\text{BPR-OPT} = \sum_{(u, i, j) \in D_S} \ln \sigma\left(\hat{r}_{ui} - \hat{r}_{uj}\right) - \lambda \|\Theta\|^2
$$

where $D_S = \{(u, i, j) \mid i \in \mathcal{I}_u^+, j \in \mathcal{I}_u^-\}$ is the set of training triples, $\sigma$ is the sigmoid function, and $\Theta$ are the model parameters.

**Derivation from Bayesian principles:**

1. Assume each user $u$ has a total order $>_u$ over items.
2. We want to maximize $p(\Theta \mid >_u)$ for all users, which by Bayes' rule is proportional to $p(>_u \mid \Theta) \cdot p(\Theta)$.
3. Assume the pairwise preference $i >_u j$ has probability $p(i >_u j \mid \Theta) = \sigma(\hat{r}_{ui} - \hat{r}_{uj})$.
4. Assume independence of pairwise comparisons (a simplification).
5. The likelihood is: $\prod_{(u,i,j) \in D_S} \sigma(\hat{r}_{ui} - \hat{r}_{uj})$.
6. With a Gaussian prior on $\Theta$ (i.e., L2 regularization), taking the log gives the BPR-OPT objective.

**SGD updates for BPR with matrix factorization:**

Let $\hat{r}_{uij} = \hat{r}_{ui} - \hat{r}_{uj} = \mathbf{p}_u^\top(\mathbf{q}_i - \mathbf{q}_j) + b_i - b_j$.

The gradient of the loss with respect to a parameter $\theta$ is:

$$
\frac{\partial \text{BPR-OPT}}{\partial \theta} = \sum_{(u,i,j)} \frac{-e^{-\hat{r}_{uij}}}{1 + e^{-\hat{r}_{uij}}} \cdot \frac{\partial \hat{r}_{uij}}{\partial \theta} - \lambda\theta
$$

For $\theta = \mathbf{p}_u$: $\frac{\partial \hat{r}_{uij}}{\partial \mathbf{p}_u} = \mathbf{q}_i - \mathbf{q}_j$

For $\theta = \mathbf{q}_i$: $\frac{\partial \hat{r}_{uij}}{\partial \mathbf{q}_i} = \mathbf{p}_u$

For $\theta = \mathbf{q}_j$: $\frac{\partial \hat{r}_{uij}}{\partial \mathbf{q}_j} = -\mathbf{p}_u$

**Negative sampling.** Enumerating all $(u, i, j)$ triples is infeasible. BPR samples: for each observed interaction $(u, i)$, sample a random item $j$ not in $\mathcal{I}_u^+$. Uniform sampling is simple but biased toward easy negatives (obscure items the user would obviously not interact with). **Hard negative sampling** (sampling from popular items) provides more informative gradients but can destabilize training.

---

## 4.5 Listwise Approaches

### 4.5.1 ListNet

**ListNet** (Cao et al., 2007) defines a probability distribution over permutations and minimizes the cross-entropy between the true and predicted permutation distributions.

For a list of $n$ items with scores $s = (s_1, \ldots, s_n)$, the **Plackett-Luce model** defines the probability of a permutation $\pi$:

$$
P(\pi \mid s) = \prod_{k=1}^{n} \frac{\exp(s_{\pi(k)})}{\sum_{j=k}^{n} \exp(s_{\pi(j)})}
$$

This is computationally expensive ($n!$ permutations). ListNet uses the **top-one probability**: the probability that each item $i$ is ranked first:

$$
P_s(i) = \frac{\exp(s_i)}{\sum_{j=1}^n \exp(s_j)}
$$

The loss is the cross-entropy between the top-one probabilities from true relevance labels $y$ and predicted scores $\hat{s}$:

$$
L_{\text{ListNet}} = -\sum_{i=1}^{n} P_y(i) \log P_{\hat{s}}(i)
$$

This is essentially a softmax cross-entropy loss over items in the list, treating the true relevance scores as the "label" distribution.

### 4.5.2 LambdaMART

**LambdaMART** (Burges, 2010) is the most successful learning-to-rank algorithm for information retrieval. It combines two ideas:

1. **LambdaRank:** Instead of defining a loss function and deriving gradients, directly define the gradients (called "lambdas") that encode how much each pair of items should be swapped. The key insight: weight each pairwise gradient by the change in NDCG that would result from swapping those two items.

$$
\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(\hat{s}_i - \hat{s}_j)}} \cdot |\Delta \text{NDCG}_{ij}|
$$

where $|\Delta \text{NDCG}_{ij}|$ is the absolute change in NDCG if items $i$ and $j$ were swapped. This focuses training on pairs where a swap would most improve the metric.

2. **MART (Multiple Additive Regression Trees):** Gradient boosted decision trees (GBDT) serve as the scoring function. GBDT is ideal for learning-to-rank because it handles heterogeneous features naturally, is fast at inference, and produces strong baselines.

LambdaMART was the winning algorithm in several Yahoo! and Microsoft learning-to-rank competitions.

---

## 4.6 Multi-Objective Optimization

### 4.6.1 The Problem

Real recommender systems optimize multiple objectives simultaneously:

- **Engagement:** Click-through rate, watch time, listen time. High engagement keeps users on the platform.
- **Satisfaction:** User ratings, thumbs up/down, survey responses. Users might click on clickbait (high engagement) but be dissatisfied afterward.
- **Diversity:** Avoid showing 10 similar items. Diverse recommendations expose users to new topics and prevent filter bubbles.
- **Freshness/novelty:** Surfacing new items, not just safe popular choices.
- **Revenue:** For marketplace platforms, the value of a recommendation may depend on the item's profit margin.

These objectives often conflict. Clickbait maximizes short-term engagement but hurts satisfaction. Popular items maximize engagement but hurt diversity.

### 4.6.2 Approaches

**Scalarization.** Combine objectives into a single score:

$$
\text{score}(u, i) = w_1 \cdot \hat{P}(\text{click}) + w_2 \cdot \hat{P}(\text{satisfy}) + w_3 \cdot \text{diversity\_bonus}(i)
$$

The weights $w_k$ encode business priorities and are often tuned via online experiments (A/B tests).

**Multi-task learning.** Train a single model with a shared representation and multiple output heads:

$$
L = \alpha L_{\text{click}} + \beta L_{\text{satisfaction}} + \gamma L_{\text{completion}}
$$

Shared lower layers learn general user-item representations; task-specific upper layers specialize.

**Pareto optimization.** Find the Pareto frontier—the set of solutions where improving one objective necessarily worsens another. Present the frontier to decision-makers who choose a point. In practice, this is done by varying the weights $w_k$ and running multiple models.

**Constrained optimization.** Maximize the primary objective subject to constraints on secondary objectives:

$$
\max \text{Engagement} \quad \text{s.t.} \quad \text{Diversity} \geq \tau_1, \quad \text{Satisfaction} \geq \tau_2
$$

This is often implemented via re-ranking: the ranking model produces an engagement-optimal list, and a post-processing step re-orders to satisfy diversity and fairness constraints (e.g., Maximal Marginal Relevance).

---

## 4.7 Common Pitfalls & Misconceptions

**Pitfall 1: Optimizing MSE when the task is ranking.** MSE penalizes errors at position 500 the same as errors at position 1. Use a ranking loss.

**Pitfall 2: BPR with biased negative sampling.** Uniform negative sampling over-represents easy negatives. The model wastes capacity distinguishing obviously irrelevant items. Popularity-weighted or dynamic hard negative sampling improves training.

**Pitfall 3: Ignoring position bias in logged data.** Users click items at position 1 more than position 5, regardless of relevance. If you train on click data without correcting for position bias, the model learns to replicate the existing ranking. **Inverse propensity weighting** or **position-aware models** correct for this.

**Pitfall 4: Short-term metric optimization.** Maximizing click-through rate without considering satisfaction leads to clickbait. Long-term metrics (return visits, subscription retention) should be factored in.

**Pitfall 5: Assuming LambdaMART always beats neural models.** LambdaMART dominates on tabular feature sets. But when the input is raw text, images, or sequences, neural models that learn representations end-to-end can outperform GBDT-based rankers.

---

# 5. Evaluation Metrics

## 5.1 Motivation & Intuition

You've built a recommender system. How do you know it's good? You need metrics that quantify the quality of recommendations. But "quality" is multidimensional: accuracy (do users like what you recommend?), ranking quality (are the best items near the top?), diversity (are recommendations varied?), novelty (are recommendations surprising?), and coverage (does the system serve all items?).

Different metrics capture different aspects, and optimizing for one can hurt another.

---

## 5.2 Accuracy Metrics

### 5.2.1 Precision@k

Among the top $k$ recommended items, what fraction is relevant?

$$
\text{Precision@}k = \frac{|\{\text{recommended items in top } k\} \cap \{\text{relevant items}\}|}{k}
$$

**Example:** Top-5 recommendations: [A, B, C, D, E]. User actually liked: {A, C, F, G}.

$\text{Precision@5} = \frac{|\{A, C\}|}{5} = 0.4$

**Interpretation:** 40% of the recommendations were relevant.

### 5.2.2 Recall@k

Among all relevant items, what fraction appears in the top $k$?

$$
\text{Recall@}k = \frac{|\{\text{recommended items in top } k\} \cap \{\text{relevant items}\}|}{|\{\text{relevant items}\}|}
$$

Using the same example: $\text{Recall@5} = \frac{2}{4} = 0.5$

**Interpretation:** 50% of the items the user would like were found in the top 5.

### 5.2.3 Mean Average Precision (MAP)

**Average Precision (AP)** for a single user computes Precision@k at every position where a relevant item appears, then averages:

$$
\text{AP} = \frac{1}{|\text{relevant items}|} \sum_{k=1}^{N} \text{Precision@}k \cdot \mathbb{1}[\text{item at position } k \text{ is relevant}]
$$

**Example:** Ranked list [Rel, Irrel, Rel, Irrel, Rel]. Relevant items = 3.

- Position 1 (Rel): $\text{Prec@1} = 1/1 = 1.0$
- Position 3 (Rel): $\text{Prec@3} = 2/3 = 0.667$
- Position 5 (Rel): $\text{Prec@5} = 3/5 = 0.6$

$\text{AP} = \frac{1}{3}(1.0 + 0.667 + 0.6) = 0.756$

**MAP** averages AP over all users:

$$
\text{MAP} = \frac{1}{|U|}\sum_{u \in U} \text{AP}(u)
$$

AP rewards relevant items being ranked higher. If the same three relevant items appeared at positions 1, 2, 3 instead of 1, 3, 5, AP would be $(1 + 1 + 1)/3 = 1.0$.

### 5.2.4 Normalized Discounted Cumulative Gain (NDCG)

NDCG handles graded relevance (not just binary). Each item has a relevance score $\text{rel}_i$ (e.g., 0, 1, 2, 3).

**Discounted Cumulative Gain at position $k$:**

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}
$$

The numerator $2^{\text{rel}_i} - 1$ exponentially rewards higher relevance. The denominator $\log_2(i+1)$ provides a logarithmic discount—items at lower positions contribute less.

**Ideal DCG (IDCG@k):** The DCG achieved by the ideal ranking (items sorted by decreasing relevance).

$$
\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}
$$

**Worked example:**

Predicted ranking with relevance: [3, 2, 0, 1, 2]

$\text{DCG@5} = \frac{2^3-1}{\log_2 2} + \frac{2^2-1}{\log_2 3} + \frac{2^0-1}{\log_2 4} + \frac{2^1-1}{\log_2 5} + \frac{2^2-1}{\log_2 6}$

$= \frac{7}{1} + \frac{3}{1.585} + \frac{0}{2} + \frac{1}{2.322} + \frac{3}{2.585}$

$= 7 + 1.893 + 0 + 0.431 + 1.161 = 10.484$

Ideal ranking with relevance: [3, 2, 2, 1, 0]

$\text{IDCG@5} = \frac{7}{1} + \frac{3}{1.585} + \frac{3}{2} + \frac{1}{2.322} + \frac{0}{2.585}$

$= 7 + 1.893 + 1.5 + 0.431 + 0 = 10.824$

$\text{NDCG@5} = \frac{10.484}{10.824} = 0.969$

---

## 5.3 Beyond-Accuracy Metrics

### 5.3.1 Novelty

How "surprising" are the recommendations? Items that are obscure or new to the user are more novel.

**Self-information based novelty:**

$$
\text{Novelty@}k = \frac{1}{k} \sum_{i \in L_k} -\log_2 p(i)
$$

where $p(i)$ is the popularity of item $i$ (fraction of users who interacted with it). Popular items have high $p(i)$ and low novelty; obscure items have low $p(i)$ and high novelty.

### 5.3.2 Diversity

How different are the recommended items from each other?

**Intra-List Diversity (ILD):**

$$
\text{ILD@}k = \frac{1}{k(k-1)} \sum_{i \in L_k} \sum_{j \in L_k, j \neq i} (1 - \text{sim}(i, j))
$$

This averages the dissimilarity between all pairs in the recommendation list. Higher ILD means more diverse recommendations.

### 5.3.3 Serendipity

Recommendations that are both relevant AND unexpected. An item that the user likes but would not have discovered on their own.

$$
\text{Serendipity@}k = \frac{1}{k} \sum_{i \in L_k} \max(0, \text{rel}(i) - \text{expected}(i))
$$

where $\text{expected}(i)$ is the prediction from a simple baseline (e.g., popularity-based). An item that is relevant but would have been recommended by popularity anyway has low serendipity.

### 5.3.4 Coverage

**Catalog coverage:** What fraction of items ever appears in any user's recommendation list?

$$
\text{Coverage} = \frac{|\bigcup_{u} L_k(u)|}{|I|}
$$

Low coverage means the system concentrates on a small subset of popular items.

**User coverage:** What fraction of users receives "good" recommendations (e.g., at least one relevant item in top-$k$)?

---

## 5.4 Online Evaluation Metrics

Offline metrics are computed on held-out data. Online metrics are measured from live user behavior:

- **Click-through rate (CTR):** Fraction of displayed recommendations that users click.
- **Conversion rate:** Fraction of clicks that lead to a purchase, signup, or other target action.
- **Dwell time:** How long users spend on a recommended item (especially for content like articles or videos).
- **Session length / return rate:** Do users come back more often?

**A/B testing** is the standard for online evaluation: randomly assign users to the current system (control) vs. the new system (treatment) and compare metrics.

**Pitfall: surrogate mismatch.** A system that improves NDCG offline may not improve CTR online, because offline metrics assume missing interactions are negative (they might just be undiscovered). Online evaluation is the ground truth.

---

## 5.5 Common Pitfalls & Misconceptions

**Pitfall 1: Using Precision@k alone.** Precision@k ignores the ranking order within the top $k$. If both systems have Precision@5 = 0.4, but one puts relevant items at positions 1 and 2 while the other puts them at positions 4 and 5, the first is clearly better. NDCG or MAP capture this.

**Pitfall 2: Confusing binary and graded metrics.** Precision, Recall, and MAP assume binary relevance (relevant/not). NDCG handles graded relevance. Using binary metrics when you have graded labels wastes information.

**Pitfall 3: Evaluating on all items vs. sampled items.** Some papers evaluate metrics over the full item catalog; others sample a subset of negative items. These produce very different absolute numbers and are not comparable.

**Pitfall 4: Ignoring popularity bias in evaluation.** If the test set is skewed toward popular items, a popularity-based baseline can score high on Precision@k while being useless for personalization.

**Pitfall 5: Not reporting confidence intervals.** Recommendation metrics can be noisy across users. Report standard errors or bootstrap confidence intervals.

---

# 6. Scalability Challenges

## 6.1 Motivation & Intuition

A production recommender system at scale (e.g., YouTube with billions of videos and billions of users) must answer the question: "What should we show this user?" in under 100 milliseconds. Scoring every item in the catalog per request is computationally infeasible.

The fundamental bottleneck is **retrieval:** given a user representation (embedding), find the most similar items among millions or billions of candidates.

**Brute-force nearest neighbor search** computes the distance between the query vector and every candidate vector—$O(n)$ for each query. With $n = 10^9$ items and $d = 256$ dimensions, this requires $2.56 \times 10^{11}$ floating-point operations per query. At 100 queries per second, that's $2.56 \times 10^{13}$ FLOPS—utterly infeasible.

**Approximate nearest neighbor (ANN)** methods trade a small amount of accuracy for massive speedups, typically finding the true nearest neighbors 95%+ of the time while being 100–1000x faster.

---

## 6.2 Approximate Nearest Neighbors

### 6.2.1 Tree-Based Methods (KD-Trees, Ball Trees)

**KD-trees** recursively partition the space by splitting along one coordinate at a time. For low-dimensional data ($d < 20$), they provide $O(\log n)$ query time. But in high dimensions, the curse of dimensionality renders them ineffective—practically every branch must be explored, degrading to brute-force.

**Annoy (Approximate Nearest Neighbors Oh Yeah)** builds a forest of random projection trees. Each tree splits the space using random hyperplanes. At query time, it searches multiple trees and takes the union of candidates, then re-ranks them by exact distance. Used at Spotify for music recommendations.

### 6.2.2 Locality-Sensitive Hashing (LSH)

**Core idea:** Design hash functions such that similar items hash to the same bucket with high probability, while dissimilar items hash to different buckets.

**For cosine similarity:** Random hyperplane LSH. Choose a random vector $\mathbf{r}$. Define $h(\mathbf{x}) = \text{sign}(\mathbf{r}^\top \mathbf{x})$. The probability that two points $\mathbf{a}$ and $\mathbf{b}$ have the same hash is:

$$
P[h(\mathbf{a}) = h(\mathbf{b})] = 1 - \frac{\theta(\mathbf{a}, \mathbf{b})}{\pi}
$$

where $\theta(\mathbf{a}, \mathbf{b})$ is the angle between $\mathbf{a}$ and $\mathbf{b}$. Similar vectors (small angle) have high probability of the same hash.

Using $L$ hash functions with $B$ bits each creates $L$ hash tables. At query time, retrieve candidates from the same buckets as the query across all tables, then re-rank by exact distance.

**Trade-offs:** More hash tables increase recall but also increase storage and query time. More bits per hash increase precision (fewer false positives) but decrease recall (more false negatives).

### 6.2.3 Graph-Based Methods (HNSW)

**Hierarchical Navigable Small World (HNSW)** graphs build a multi-layer graph where each layer is a navigable small-world graph with decreasing density. The top layer has the fewest nodes (long-range connections), and the bottom layer contains all nodes (short-range connections).

**Query algorithm:**
1. Start at a random entry point in the top layer.
2. Greedily navigate toward the query by moving to the neighbor closest to the query.
3. When no closer neighbor exists in the current layer, drop to the next lower layer.
4. At the bottom layer, perform a more thorough local search.

HNSW provides state-of-the-art recall-speed tradeoffs and is used in Facebook's FAISS library (which also implements IVF and PQ—see below).

### 6.2.4 Inverted File Index with Product Quantization (IVF-PQ)

**IVF (Inverted File Index):** Partition the vector space into $C$ clusters using k-means. At query time, find the $p$ nearest cluster centroids and search only those clusters ($p \ll C$, typically $p = 1$–$10$).

**PQ (Product Quantization):** Compress each $d$-dimensional vector into a compact code. Split the vector into $M$ subvectors of dimension $d/M$. For each subspace, learn a codebook of $K$ centroids (typically $K = 256$). Represent each subvector by its nearest centroid index (1 byte for $K = 256$).

A 256-dimensional float32 vector (1024 bytes) can be compressed to $M = 32$ bytes with PQ, a 32x compression. Distances in PQ space approximate the original distances.

**IVF-PQ combined:** Partition using IVF, store PQ codes in each partition, compute approximate distances using PQ tables. This is the backbone of Facebook's FAISS and is used for billion-scale nearest neighbor search.

---

## 6.3 Common Pitfalls & Misconceptions

**Pitfall 1: Using exact nearest neighbors in production.** Unless the catalog is small (<100k items), exact search is too slow. Always use ANN.

**Pitfall 2: Choosing ANN parameters without benchmarking recall.** ANN methods have a recall-speed tradeoff. Always measure recall@10 (or recall@100) against the ground-truth brute-force result at the target latency. A system that's 10x faster but has 50% recall is useless.

**Pitfall 3: Ignoring index update costs.** New items must be added to the ANN index. Some methods (LSH) support streaming inserts; others (IVF-PQ with centroids) require periodic retraining of the cluster structure.

**Pitfall 4: Assuming ANN = recommendation.** ANN is just the retrieval step. The retrieved candidates still need to be scored and re-ranked by a more sophisticated model.

---

# 7. Cold Start Problems

## 7.1 Motivation & Intuition

A new user signs up with zero interaction history. A new product is added to the catalog with zero purchases. Collaborative filtering, which relies on interaction patterns, has nothing to work with. This is the **cold start problem**, and it is one of the most practically important challenges in recommendation systems.

There are three variants:

1. **New user cold start:** A user with no interaction history joins the system. We cannot compute CF similarity to existing users and have no latent factor vector for them.
2. **New item cold start:** A new item is added with no interactions. No user has rated it, so it has no latent factor vector.
3. **System cold start:** A brand-new platform with very few users and items—insufficient interaction data for CF to work at all.

---

## 7.2 Strategies for New Users

**Demographic-based recommendations.** Collect basic demographic information (age, gender, location) at signup and use these features to match the new user to a demographic cluster. Recommend items popular among users with similar demographics.

$$
\text{score}(u_{\text{new}}, i) = \frac{1}{|C(u_{\text{new}})|} \sum_{v \in C(u_{\text{new}})} r_{vi}
$$

where $C(u_{\text{new}})$ is the cluster of users with similar demographics.

**Onboarding / preference elicitation.** Ask the new user to rate a small set of items during signup. Choosing which items to present is itself an optimization problem—select items that maximally discriminate between user types (items with high variance in ratings across demographic clusters).

**Bandits for exploration.** Use a multi-armed bandit (e.g., Thompson sampling, Upper Confidence Bound) to trade off exploring the new user's preferences and exploiting early signals. Each arm corresponds to an item category or recommendation strategy. Initially, the system explores widely; as it learns the user's preferences, it exploits.

**Transfer learning from side information.** If the user authenticates via a social login (e.g., Google, Facebook), their public profile data (interests, group memberships) can be used to initialize a user representation.

**Meta-learning approaches (MELU, MeLU).** Train a meta-learner that can quickly adapt to a new user from a few interactions. The meta-learner is trained across many users, learning an initialization of user parameters that can be fine-tuned with just 3–5 ratings.

---

## 7.3 Strategies for New Items

**Content-based features.** The most direct solution: represent items by their content features and use CBF to recommend new items. In matrix factorization, the item's latent vector can be initialized by regressing from content features:

$$
\mathbf{q}_i = W \mathbf{x}_i + \mathbf{b}
$$

where $W$ and $\mathbf{b}$ are learned from items with sufficient interactions.

**Warm-start in two-tower models.** In a two-tower architecture (user tower + item tower), the item tower maps item features to an embedding. A new item's embedding is computed by the item tower directly—no interaction data needed. The quality depends on the item tower's generalization to unseen items.

**Exploration bonuses.** Artificially boost the score of new items to ensure they receive some initial exposure. This is a form of exploration-exploitation tradeoff.

**Hybrid approach: popularity fallback.** For items too new to have any signal, recommend based on overall popularity within the item's category.

---

## 7.4 Common Pitfalls & Misconceptions

**Pitfall 1: Ignoring cold start entirely.** Many papers report results on warm users/items only. In production, a significant fraction of traffic may involve cold entities. A system must degrade gracefully.

**Pitfall 2: Showing the same popular items to all new users.** This solves cold start but provides zero personalization and makes a poor first impression. Even minimal demographic-based personalization is better.

**Pitfall 3: Collecting too much information at onboarding.** Asking users to rate 50 items before they can use the platform causes signup abandonment. Keep onboarding friction minimal (3–5 items).

**Pitfall 4: Not distinguishing "missing" from "disliked."** A new item with zero interactions might be excellent—it just hasn't been exposed. Treating it as unpopular in training data is incorrect.

---

# 8. Interview Questions & Answers

## 8.1 Foundational Questions

### Q1: Explain the difference between collaborative filtering and content-based filtering. When would you prefer one over the other?

**Answer:** Collaborative filtering (CF) recommends items based on patterns in user-item interaction data—users who agreed in the past are likely to agree in the future. It requires no knowledge of item content. Content-based filtering (CBF) recommends items whose features match the user's demonstrated preferences—it analyzes item attributes like genre, text, or images.

Choose CF when interaction data is abundant and when serendipitous discovery is valued (CF can recommend items outside a user's known interest areas by leveraging similar users' behavior). Choose CBF when interaction data is sparse, when new items must be recommended immediately (no cold-start problem for items), or when explainability is needed (CBF explanations directly reference item attributes). In practice, nearly all production systems use hybrids.

### Q2: What is the user-item interaction matrix? Why is it sparse, and why does sparsity matter?

**Answer:** The user-item interaction matrix $R \in \mathbb{R}^{m \times n}$ has one row per user and one column per item. Entry $R_{ui}$ records the interaction (rating, click, purchase) between user $u$ and item $i$, or is missing if no interaction occurred.

Sparsity arises because each user interacts with a tiny fraction of items. Netflix has ~15,000 titles and ~230 million subscribers; the average subscriber watches ~100 titles, meaning >99.3% of entries are missing. Amazon has hundreds of millions of products; any individual purchases perhaps thousands—sparsity exceeding 99.99%.

Sparsity matters because: (1) memory-based CF struggles to find reliable neighbors when co-rated items are few; (2) matrix factorization methods must regularize heavily to avoid overfitting; (3) evaluation is difficult because most true preferences are unobserved.

### Q3: What are the key differences between explicit and implicit feedback? How does this affect model design?

**Answer:** Explicit feedback comprises direct user opinions (star ratings, thumbs up/down, reviews). It is high-quality but rare—most users don't rate most items. Implicit feedback comprises observed user behavior (clicks, purchases, watch time, scrolls). It is abundant but noisy—a click doesn't mean the user liked the item, and absence of a click doesn't mean dislike (the user may not have seen the item).

Model design implications: for explicit feedback, regression losses (MSE) are natural, and the task is rating prediction. For implicit feedback, the task is typically ranking (which items should appear at the top?). Binary classification (clicked/not clicked) or pairwise ranking losses (BPR) are used. The key challenge with implicit feedback is negative sampling—we must assume that unobserved interactions are "less preferred" (not necessarily negative), and the choice of negative sampling strategy significantly impacts performance.

### Q4: Explain NDCG. Why is it preferred over Precision@k in many ranking scenarios?

**Answer:** NDCG measures the quality of a ranked list by giving more credit to relevant items placed at higher positions. It uses a logarithmic discount: relevance at position 1 counts much more than relevance at position 10.

Two key advantages over Precision@k: (1) NDCG handles graded relevance (e.g., relevance scores of 0, 1, 2, 3), while Precision treats relevance as binary. A highly relevant item at position 1 is valued more than a marginally relevant item. (2) NDCG is position-aware: two lists with the same set of relevant items in the top-$k$ can have different NDCG scores depending on their ordering. Precision@k cannot distinguish between them. NDCG is normalized to $[0, 1]$ by dividing by the ideal DCG, making it comparable across queries with different numbers of relevant items.

### Q5: Describe the cold start problem. List three concrete strategies for handling a new user who just signed up with no interaction history.

**Answer:** The cold start problem occurs when a recommender lacks sufficient interaction data for a user (new user) or item (new item). For new users: (1) **Demographic-based initialization**—collect age, location, and other signals at signup to match the user to a demographic cluster and recommend cluster-popular items. (2) **Preference elicitation at onboarding**—present 3–5 carefully selected items (chosen for high discriminative power across user segments) and ask for ratings. Even a few explicit preferences dramatically narrow the user's taste profile. (3) **Bandit-based exploration**—use Thompson sampling or UCB to serve diverse recommendations in early sessions, using each interaction to rapidly learn preferences while still providing reasonable recommendations.

---

## 8.2 Mathematical Questions

### Q6: Derive the SGD update rules for biased matrix factorization. What is the role of each term in the update?

**Answer:** The objective minimizes regularized squared error over observed ratings:

$$
L = \sum_{(u,i) \in \mathcal{O}} (r_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2 + b_u^2 + b_i^2)
$$

Define the prediction error: $e_{ui} = r_{ui} - \mu - b_u - b_i - \mathbf{p}_u^\top \mathbf{q}_i$.

Take the gradient with respect to $\mathbf{p}_u$:

$$
\frac{\partial L}{\partial \mathbf{p}_u} = -2e_{ui}\mathbf{q}_i + 2\lambda \mathbf{p}_u
$$

The SGD update (stepping opposite to the gradient): $\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta(e_{ui}\mathbf{q}_i - \lambda \mathbf{p}_u)$.

The first term $\eta \cdot e_{ui}\mathbf{q}_i$ moves the user embedding toward the item embedding scaled by the error—if the prediction was too low ($e_{ui} > 0$), the user vector is adjusted to increase alignment with the item vector. The second term $-\eta\lambda \mathbf{p}_u$ is weight decay, shrinking the user vector toward zero to prevent overfitting.

Analogous updates hold for $\mathbf{q}_i$, $b_u$, and $b_i$.

### Q7: In SVD++, explain the $|I_u|^{-1/2}$ normalization term. Why not use $|I_u|^{-1}$?

**Answer:** The implicit feedback component in SVD++ aggregates item factors $\mathbf{y}_j$ over all items user $u$ has interacted with: $\sum_{j \in I_u} \mathbf{y}_j$. Without normalization, users with more interactions would have much larger implicit factor sums, dominating the prediction. The normalization brings different users to comparable scales.

Using $|I_u|^{-1}$ (simple averaging) would mean each item's implicit signal is weighted inversely proportional to the user's activity level. This over-penalizes active users—their per-item implicit weight becomes negligible. The $|I_u|^{-1/2}$ normalization is a middle ground: it reduces the magnitude of the implicit factor for active users but doesn't shrink it as aggressively. Empirically, Koren found that this provides the best bias-variance tradeoff. The square root scaling is related to the intuition from ensemble methods: aggregating $n$ independent signals should reduce variance by $\sqrt{n}$, not $n$.

### Q8: Derive the efficient computation of the Factorization Machine interaction term. Show it achieves $O(pk)$ instead of $O(p^2k)$.

**Answer:** The pairwise interaction term is:

$$
\sum_{j=1}^{p} \sum_{j'=j+1}^{p} \langle \mathbf{v}_j, \mathbf{v}_{j'} \rangle x_j x_{j'}
$$

Direct computation iterates over all $O(p^2)$ pairs, each requiring $O(k)$ for the dot product: $O(p^2 k)$.

**Trick:** Expand the square of the sum:

$$
\left(\sum_{j=1}^{p} \mathbf{v}_j x_j\right)^2 = \sum_{j}\sum_{j'} \langle \mathbf{v}_j, \mathbf{v}_{j'}\rangle x_j x_{j'}
$$

This includes diagonal terms ($j = j'$) which equal $\sum_j \|\mathbf{v}_j\|^2 x_j^2$. The upper-triangular terms (what we want) are half the off-diagonal:

$$
\sum_{j<j'} \langle \mathbf{v}_j, \mathbf{v}_{j'}\rangle x_j x_{j'} = \frac{1}{2}\left[\left\|\sum_j \mathbf{v}_j x_j\right\|^2 - \sum_j \|\mathbf{v}_j\|^2 x_j^2\right]
$$

Computing $\sum_j \mathbf{v}_j x_j$ is $O(pk)$ (sum $p$ vectors of dimension $k$). Squaring it is $O(k)$. Computing $\sum_j \|\mathbf{v}_j\|^2 x_j^2$ is also $O(pk)$. Total: $O(pk)$, a massive improvement when $p$ is large.

### Q9: Explain the mathematical relationship between BPR and binary cross-entropy loss. When are they equivalent?

**Answer:** BPR loss for a triple $(u, i, j)$ is:

$$
L_{\text{BPR}} = -\log \sigma(\hat{r}_{ui} - \hat{r}_{uj})
$$

Binary cross-entropy (BCE) for a positive item $i$ and a negative item $j$:

$$
L_{\text{BCE}} = -\log \sigma(\hat{r}_{ui}) - \log(1 - \sigma(\hat{r}_{uj})) = -\log \sigma(\hat{r}_{ui}) - \log \sigma(-\hat{r}_{uj})
$$

These are generally different. BPR depends only on the *difference* $\hat{r}_{ui} - \hat{r}_{uj}$ (it is translation-invariant: adding a constant to all scores doesn't change the loss). BCE depends on the absolute scores—it pushes positive scores toward $+\infty$ and negative scores toward $-\infty$.

They are equivalent when $\hat{r}_{uj} = 0$ for all negative items (i.e., the model has a fixed zero baseline for negatives). Then $L_{\text{BCE}} = -\log \sigma(\hat{r}_{ui})$, which is the one-sided version. The pairwise formulation of BPR is preferable when we care only about relative order, not absolute score calibration.

### Q10: Show that the Plackett-Luce top-one probability reduces to softmax. What are the implications for ListNet?

**Answer:** The Plackett-Luce model defines the probability of a full permutation $\pi$ as:

$$
P(\pi \mid s) = \prod_{k=1}^n \frac{\exp(s_{\pi(k)})}{\sum_{l=k}^n \exp(s_{\pi(l)})}
$$

The marginal probability that item $i$ is ranked first (top-one probability) requires summing over all permutations where $\pi(1) = i$:

$$
P(i \text{ is first} \mid s) = \sum_{\pi:\pi(1)=i} P(\pi \mid s)
$$

The first factor in every such permutation is $\frac{\exp(s_i)}{\sum_{l=1}^n \exp(s_l)}$. The remaining factors form a sub-permutation over $n-1$ items, and the sum of all sub-permutation probabilities is 1 (they form a valid probability distribution). Therefore:

$$
P(i \text{ is first} \mid s) = \frac{\exp(s_i)}{\sum_{l=1}^n \exp(s_l)}
$$

This is exactly the softmax function. The implication for ListNet is that minimizing cross-entropy between the ground-truth top-one probabilities and predicted top-one probabilities is equivalent to a standard softmax cross-entropy loss over the candidate items, making it computationally tractable and differentiable.

---

## 8.3 Applied Questions

### Q11: You're building a recommendation system for a new e-commerce platform with 10,000 items and 100,000 users. Most users have fewer than 5 interactions. Describe your approach.

**Answer:** With extreme sparsity (average user has <5 interactions out of 10,000 items, sparsity >99.95%), pure collaborative filtering will struggle. I'd pursue a multi-stage approach:

**Phase 1 (Day 1): Content-based + popularity.** Extract item features (product category, brand, description embeddings from a pretrained language model, price range, images through a pretrained CNN). Build a CBF system that matches user profiles to item features. Blend with popularity-based recommendations as a fallback.

**Phase 2 (Weeks 1–4): Hybrid warm-start.** As interactions accumulate, introduce a factorization machine or two-tower model that combines user ID embeddings, item ID embeddings, and content features. The content features prevent cold start for new items; the ID embeddings capture collaborative signal as it emerges. Use BPR or sampled softmax loss for implicit feedback.

**Phase 3 (Month 2+): Full hybrid pipeline.** Implement a cascade architecture: an ANN-based retrieval stage (two-tower model) generates ~200 candidates, followed by a feature-rich ranking model (gradient-boosted trees or deep model) that incorporates collaborative, content, contextual (time of day, device type), and real-time features (items viewed in current session).

Throughout, use online A/B testing to validate improvements. Carefully monitor for popularity bias and implement diversity re-ranking.

### Q12: Your recommendation model has high offline NDCG but low click-through rate in production. Diagnose potential causes.

**Answer:** Several causes can explain offline-online discrepancy:

1. **Train-test distribution mismatch.** Offline evaluation uses historical data, which is biased by the previous recommendation system. Items that were never shown can't be in the test set, creating a systematic blind spot. Solution: use unbiased evaluation methods (e.g., inverse propensity scoring) or deploy via A/B test.

2. **Position bias in logged data.** The training data is generated by a previous system that assigned items to specific positions. Items at position 1 receive more clicks regardless of relevance. If the model learns to replicate this positional effect, it has high apparent accuracy but doesn't actually improve relevance. Solution: model position as a separate feature, or use propensity-weighted training.

3. **Temporal distribution shift.** Offline evaluation uses a fixed historical snapshot; online behavior evolves. User interests shift seasonally or in response to current events. Solution: retrain frequently; use recent data for evaluation.

4. **Serving latency or bugs.** The model might be too slow, causing timeouts and fallback to a simpler model. Or a serialization bug might corrupt features. Solution: monitoring and observability.

5. **Surrogate metric mismatch.** NDCG measures ranking quality of rated items, but CTR measures whether users click at all. The model might rank relevant items well within the rated set but miss entirely the items users would actually click on (items the user hasn't rated in the historical data).

### Q13: You observe that your recommender system predominantly recommends popular items, creating a "rich get richer" feedback loop. How do you address this?

**Answer:** Popularity bias is one of the most pernicious issues in production recommenders. Multi-pronged approach:

**Training-time interventions:** (1) Use popularity-weighted negative sampling in BPR—sample negative items proportionally to popularity so the model must learn to distinguish truly relevant items from merely popular ones. (2) Add inverse popularity as a feature or regularization term: penalize the model for relying on popularity as the sole signal.

**Ranking-time interventions:** (1) **Maximal Marginal Relevance (MMR):** Re-rank by iteratively selecting the item that maximizes a linear combination of relevance and diversity from already-selected items. (2) Apply a popularity discount: $\text{score}_{\text{adjusted}} = \text{score}_{\text{model}} - \beta \cdot \log(\text{popularity})$.

**Exploration mechanisms:** (1) Reserve a fraction of recommendation slots (e.g., 10%) for exploration—showing items selected via bandits rather than the ranking model. (2) Use $\epsilon$-greedy or Thompson sampling to balance exploitation (showing what the model thinks is best) with exploration (giving less popular items a chance).

**Metric monitoring:** Track catalog coverage, Gini coefficient of item exposure (lower is more equitable), and long-tail exposure rate. Set business targets for these metrics.

### Q14: Design a two-tower model for candidate retrieval at YouTube scale (billions of videos).

**Answer:** The two-tower architecture has a user tower and an item tower that independently produce embeddings, enabling efficient ANN retrieval.

**User tower inputs:** User ID embedding, watch history (average or attention-weighted embedding of recently watched video embeddings), search history, demographic features (age bucket, gender, country), device type, time features (hour of day, day of week).

**Item tower inputs:** Video ID embedding, title/description text embedding (from a frozen pretrained language model), category, upload time, channel embedding, video duration, thumbnail embedding.

**Architecture:** Each tower is a multi-layer feedforward network (e.g., 3 layers of 512 → 256 → 128) producing a 128-dimensional normalized embedding. The scoring function is the dot product (or cosine similarity) of the two tower outputs.

**Training:** Use sampled softmax: for each positive (user, video) pair, sample ~1000 negative videos (using log-uniform sampling to correct for popularity bias). The loss is cross-entropy over the softmax of dot-product scores.

**Serving:** Pre-compute all video embeddings. Build an HNSW or IVF-PQ index. At query time, compute the user embedding from real-time features, query the ANN index for the top 1000 candidates in ~10ms, pass to the ranking stage.

**Critical design decisions:** (1) The towers must be independent at inference (no cross-attention) to allow pre-computation. (2) Fresh features (e.g., "video trending in the last hour") must be handled by the ranking stage, not the retrieval stage. (3) The negative sampling distribution must correct for the "implicit in batch" negatives—naively, popular videos appear as in-batch negatives more often.

### Q15: How would you incorporate real-time user behavior (e.g., items clicked in the current session) into recommendations?

**Answer:** This requires a **session-aware** or **real-time feature** architecture:

**Option 1: Session embedding in the user tower.** Encode the current session's item clicks as a sequence, compute a session embedding (via average pooling, LSTM, or self-attention over item embeddings), and concatenate with the long-term user embedding before the user tower's MLP layers. The challenge: the user tower output changes within a session, so we can't pre-compute it—must compute it live.

**Option 2: Re-ranking with session context.** The retrieval stage uses the pre-computed long-term user embedding. The ranking stage incorporates real-time session features as additional input. This is more practical because only ~1000 candidates need rescoring.

**Option 3: Sequential model (SASRec-style).** Maintain a running session representation. After each click, update the session state and regenerate recommendations. This provides the most responsive experience but requires fast inference.

**Production considerations:** Feature freshness—how quickly does a click propagate to the recommendation? Feature stores like Feast or Tecton with sub-second write-to-read latency are essential. Also, be careful of echo chambers: if the user misclicks on an item, the system might immediately pivot toward that item's category. Require multiple consistent signals before adapting.

---

## 8.4 Debugging & Failure-Mode Questions

### Q16: Your matrix factorization model's training loss is decreasing, but the test RMSE is increasing after epoch 5. What's happening and how do you fix it?

**Answer:** Classic overfitting. The model memorizes training ratings but fails to generalize.

**Diagnosis:** Plot training loss and validation RMSE together. The gap widens after epoch 5. Check: (1) Is the number of latent factors $k$ too large relative to the data size? (2) Is regularization $\lambda$ too small? (3) Is the training data too sparse for the model's capacity?

**Fixes:** (1) Increase $\lambda$. Plot validation RMSE as a function of $\lambda$ on a log scale ($10^{-4}$ to $10^1$). (2) Reduce $k$. Try $k = 10, 20, 50, 100$—find the sweet spot. (3) Use early stopping: monitor validation RMSE and stop training when it begins increasing. (4) Add implicit feedback (SVD++) to provide additional regularization through the constraint of modeling implicit behavior. (5) Data augmentation: if interaction data is truly too sparse, consider adding side information.

### Q17: Your ANN index returns candidates that are mostly irrelevant (low precision). The embedding model has good offline recall@100 when evaluated with brute-force search. What's wrong?

**Answer:** If brute-force search gives good recall but the ANN index doesn't, the issue is in the ANN index configuration, not the embeddings.

**Possible causes:** (1) **Insufficient probing.** For IVF-PQ, the nprobe parameter (number of clusters to search) may be too low. Increase nprobe from 1 to 10 or 20 and measure recall. (2) **Too aggressive quantization.** PQ compression with too few subspaces ($M$ too small) or codebook size ($K$ too small) loses too much information. Increase $M$ or use OPQ (Optimized PQ) for better reconstruction. (3) **Index trained on stale data.** If the embeddings were updated but the IVF centroids were not retrained, the cluster assignments are suboptimal. Retrain the index. (4) **Dimensionality too high for the ANN method.** Some methods degrade in high dimensions. Try dimensionality reduction (PCA to reduce from 256 to 64 before indexing). (5) **Data distribution mismatch.** The index was built on training data, but test queries come from a different distribution (e.g., new users with different feature patterns).

### Q18: After deploying your recommender, you notice that item A (which was removed from the catalog) is still being recommended. Trace the bug.

**Answer:** This is a serving infrastructure issue, not a modeling issue. The recommendation pipeline has stale state.

**Diagnosis trace:** (1) Check if item A is still in the candidate retrieval index. ANN indices (e.g., FAISS) are periodic snapshots—if the index wasn't rebuilt after item A's removal, it's still there. (2) Check the item metadata store. If the ranking model looks up item features from a database, the entry might still exist. (3) Check the serving cache. Recommendations are often cached per user to avoid recomputation. If user $u$'s cached recommendations include item A and the cache TTL hasn't expired, the stale recommendation persists.

**Fixes:** (1) Implement a real-time item blacklist that filters candidate lists before serving, regardless of index state. (2) Reduce cache TTL or invalidate cache entries when catalog changes occur. (3) Add monitoring: alert when a recommended item ID doesn't exist in the current catalog.

### Q19: Your BPR model achieves good NDCG on popular users but very poor NDCG on users with fewer than 10 interactions. Diagnose and address this.

**Answer:** This is the "long tail" user problem. Users with few interactions have poorly learned latent factors because: (1) The gradient updates for infrequent users are few and noisy. (2) Regularization pushes their factors toward zero (the prior), producing bland, unpersonalized recommendations.

**Diagnosis:** Segment evaluation by user activity level. Plot NDCG vs. number of interactions. If performance drops sharply below ~20 interactions, the model lacks sufficient signal for these users.

**Fixes:** (1) **SVD++ or side information:** Incorporate implicit signals and content features so that even users with few explicit ratings have informative representations. (2) **Regularization per user group:** Use stronger regularization for active users (who have enough data) and weaker for sparse users (to let the few signals matter more). Alternatively, use adaptive regularization (e.g., frequency-adaptive $\lambda$). (3) **Meta-learning:** Train a meta-learner that can adapt quickly from few interactions. (4) **Hybrid fallback:** For users below a threshold of interactions, switch to content-based or popularity-based recommendations.

### Q20: You run an A/B test comparing your new model (treatment) to the existing model (control). Treatment shows +2% CTR but -5% session length. Should you launch?

**Answer:** This requires careful analysis because the metrics conflict:

**Possible explanation:** The new model is better at generating clicks (possibly clickbait-like, or items at the top are more attention-grabbing) but worse at sustaining engagement (users click, find the content unsatisfying, and leave). Alternatively, the model might surface items that are so relevant that users find what they want faster and leave satisfied (shorter sessions but higher satisfaction).

**Investigation steps:** (1) Measure downstream metrics: conversion rate (if e-commerce), completion rate (if video), user return rate (next-day retention). (2) Segment by user type: do power users behave differently from casual users? (3) Check for novelty effects: is the CTR boost driven by users exploring the new interface and fading over time? (4) Survey a sample of users in each group about satisfaction.

**Decision framework:** If retention and satisfaction metrics are neutral or positive alongside the CTR gain, the shorter sessions may simply mean users are finding value faster—this is good. If retention drops, the model is producing clickbait and should not launch. If metrics are mixed, consider a longer test period to separate novelty effects from sustained impact.

---

## 8.5 Follow-Up and Probing Questions

### Q21: "You mentioned matrix factorization learns latent factors. How do you choose the number of factors $k$? What happens if $k$ is too large or too small?"

**Answer:** $k$ controls the model's capacity. Too small: the model cannot capture the diversity of user preferences (underfitting). If user tastes vary along 20 independent dimensions but $k = 5$, the model conflates distinct preference axes. Too large: the model memorizes training data (overfitting), especially problematic with sparse data.

**Choosing $k$:** Use cross-validation or a held-out validation set. Plot validation RMSE (or NDCG) as a function of $k$. Typically there's a sweet spot—say $k = 50$–$200$ for a medium-scale dataset. Beyond this point, validation performance plateaus or degrades. Also consider compute constraints: training cost and serving latency scale with $k$. For retrieval models, higher $k$ means larger ANN indices.

### Q22: "How would you handle the case where a user's preferences change over time?"

**Answer:** This is the **temporal dynamics** problem. Approaches:

1. **Recency weighting:** Weight recent interactions more heavily when computing user profiles or training MF. Use exponential decay: $w(t) = e^{-\gamma(t_{\text{now}} - t)}$.
2. **Time-aware MF (TimeSVD++):** Koren's extension models user biases and latent factors as functions of time: $b_u(t) = b_u + \alpha_u \cdot \text{sign}(t - t_u) \cdot |t - t_u|^\beta$. Latent factors are split into a stable component and a time-varying component.
3. **Sequential models (SASRec, BERT4Rec):** Naturally handle temporal dynamics because they model the sequence of interactions with causal or masked attention. Recent interactions receive implicit emphasis.
4. **Sliding window training:** Retrain the model periodically on a rolling window of recent data (e.g., last 6 months). Old interactions that no longer reflect current preferences are excluded.

### Q23: "In your two-tower model, why must the towers be independent? What if you wanted cross-attention between user and item?"

**Answer:** The towers must be independent at inference for **computational efficiency**. The item tower's output (item embedding) is pre-computed offline for all items and stored in an ANN index. The user tower's output (user embedding) is computed at query time. The retrieval step is a single ANN lookup—$O(\log n)$ or sub-linear. If there were cross-attention between user and item, the score for each (user, item) pair would depend on both, meaning we'd need to compute the score for every candidate item at query time—back to $O(n)$.

Cross-attention is used in the **ranking stage**, which only processes ~1000 candidates retrieved by the two-tower model. Architectures like DCN (Deep & Cross Network) or DIN (Deep Interest Network) use attention between the user's historical item embeddings and the candidate item embedding to capture fine-grained relevance.

### Q24: "What is the difference between ALS for explicit ratings vs. ALS for implicit feedback? Explain the confidence weighting."

**Answer:** For explicit ratings, ALS minimizes the squared error over observed entries only—unobserved entries are ignored. For implicit feedback (Hu, Koren, Volinsky 2008), there are no missing entries: every (user, item) pair has an observed value (e.g., number of views, $r_{ui}$), and we define:

- **Preference:** $p_{ui} = 1$ if $r_{ui} > 0$, else $p_{ui} = 0$ (binary: did the user interact?).
- **Confidence:** $c_{ui} = 1 + \alpha \cdot r_{ui}$ (higher interaction count means higher confidence in the preference).

The objective becomes:

$$
\min_{P,Q} \sum_u \sum_i c_{ui}(p_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)
$$

This sums over *all* user-item pairs (not just observed ones), which is $O(mn)$—much larger. The ALS update for user $u$ is:

$$
\mathbf{p}_u = (Q^\top C^u Q + \lambda I)^{-1} Q^\top C^u \mathbf{p}_u^{\text{pref}}
$$

where $C^u$ is a diagonal matrix of confidence values for user $u$. The trick: $C^u = I + (C^u - I)$, where $(C^u - I)$ is sparse (nonzero only for observed items). This allows efficient computation in $O(k^2 n + k^2 |\mathcal{I}_u| + k^3)$ instead of naive $O(k^2 n + k n + k^3)$.

### Q25: "You mentioned LambdaMART uses gradients weighted by $|\Delta\text{NDCG}|$. Why not just directly optimize NDCG?"

**Answer:** NDCG depends on the **ranking** (the permutation of items), which is a discrete structure. The sort operation is not differentiable: swapping two items changes the metric discontinuously, so there is no gradient to backpropagate through. Standard gradient-based optimization requires a smooth, differentiable loss function.

LambdaMART's insight is to bypass the non-differentiable loss entirely and instead *define the gradients directly*. The "lambdas" are designed so that the resulting updates would approximate gradient descent on an NDCG-like objective if such a smooth objective existed. The $|\Delta\text{NDCG}|$ weighting ensures that pairs where a swap would significantly improve NDCG receive larger gradient magnitudes, focusing optimization effort where it matters most.

Recent work on differentiable sorting (e.g., NeuralNDCG, SoftRank) approximates the sort operation with continuous relaxations (e.g., using softmax-based permutation matrices), enabling direct gradient-based optimization of NDCG approximations. These approaches are promising but LambdaMART remains the dominant method for tabular features.

### Q26: "How do you evaluate a recommender system when you only have implicit feedback and no explicit relevance labels?"

**Answer:** This is a central practical challenge. Approaches:

**Leave-one-out protocol:** For each user, hold out the last interaction as the test item. Train on all other interactions. Evaluate whether the held-out item ranks in the top $k$ among all items the user hasn't interacted with (Hit Rate@k, NDCG@k). This uses temporal ordering to simulate a realistic prediction scenario.

**Temporal split:** Use all interactions before time $T$ for training and interactions after $T$ for testing. This avoids data leakage from future information.

**Negative sampling for evaluation:** Since computing metrics over the full item set is expensive, sample 100–999 random negatives per test item and compute metrics on the resulting candidate set. This is a common approximation but can bias metric estimates (popular items are more likely to be sampled as negatives, inflating metrics for models that avoid popular items). Full-catalog evaluation is preferable when computationally feasible.

**Counterfactual evaluation:** Use logged interaction data with propensity scores. If item $i$ was shown at position $p$ with probability $\pi(i, p)$, weight each observation by $1/\pi(i, p)$ to correct for position and selection bias. This provides unbiased offline estimates of online metrics.

**Online A/B testing:** The gold standard. Deploy the model to a random fraction of users and measure real engagement metrics. All offline methods are proxies; A/B tests measure what ultimately matters.

### Q27: "Explain how negative sampling affects BPR training. What happens if you always sample easy negatives?"

**Answer:** In BPR, each training triple $(u, i, j)$ contains a positive item $i$ and a negative item $j$. The gradient magnitude depends on the sigmoid: $\sigma(\hat{r}_{ui} - \hat{r}_{uj})$. When $j$ is an "easy" negative (clearly irrelevant, e.g., a children's toy for an adult user interested in electronics), the model's score difference $\hat{r}_{ui} - \hat{r}_{uj}$ is already large and $\sigma$ is close to 1, producing near-zero gradients. The model learns nothing from these examples.

**Consequence of always-easy negatives:** Training converges to a solution that can distinguish broadly different items but cannot make fine-grained distinctions within a user's interest area. For example, it might learn "this user likes electronics" but not "this user prefers Sony headphones over Bose headphones."

**Hard negative sampling** selects negatives that are close to the positive in embedding space or that are popular (likely to be genuinely confusing). This produces informative gradients and forces the model to learn fine-grained distinctions. However, too-hard negatives (which might actually be false negatives—items the user would like but hasn't interacted with) can introduce label noise. A balanced approach: mix uniform and popularity-weighted sampling, or use dynamic hard negative mining (sample from items the current model ranks highly for the user but that aren't in the positive set).

### Q28: "How would you implement diversity-aware re-ranking without significantly degrading relevance?"

**Answer:** The standard approach is **Maximal Marginal Relevance (MMR):**

Starting with a ranked list $R$ from the relevance model and an initially empty selected list $S$:

$$
\text{next item} = \arg\max_{i \in R \setminus S} \left[\lambda \cdot \text{rel}(i) - (1-\lambda) \cdot \max_{j \in S} \text{sim}(i, j)\right]
$$

where $\text{rel}(i)$ is the relevance score from the ranking model, $\text{sim}(i, j)$ is the content similarity between items, and $\lambda$ controls the relevance-diversity tradeoff. The algorithm greedily selects items that are both relevant and dissimilar to already-selected items.

**Minimizing relevance degradation:** (1) Set $\lambda$ high (0.7–0.9) so relevance dominates but diversity still gets a voice. (2) Apply MMR only within a "swap window" of top-$2k$ candidates rather than the full list—this ensures all selected items are at least reasonably relevant. (3) Evaluate the NDCG cost of diversity re-ranking on the validation set and set a maximum acceptable NDCG drop (e.g., <2%). (4) Use category-level diversity (ensure coverage of multiple genres) rather than item-level diversity, which is less disruptive to per-item relevance ordering.

---

**[← Previous Chapter: Time Series](time_series_guide.md) | [Table of Contents](../README.md) | [Next Chapter: RL (Reinforcement Learning) →](rl_guide.md)**
