# Trees & Ensembles: A Comprehensive Learning Guide

---

# Part I — Decision Trees

## 1. Motivation & Intuition

Imagine you're a loan officer deciding whether to approve an applicant. You probably don't run a logistic regression in your head. Instead, you ask a sequence of questions: "Is income above $50k? If yes, is credit score above 700? If yes, approve." This sequential, rule-based reasoning is exactly what a **decision tree** formalizes.

Decision trees exist because:

- **Tabular data is messy.** It mixes numeric features (age, income), categorical features (state, occupation), missing values, and nonlinear interactions. Linear models struggle here without heavy feature engineering.
- **Humans want interpretability.** A tree produces a flowchart you can read. Regulators, doctors, and business stakeholders can follow the logic.
- **Nonlinearity and interactions come free.** A tree can learn "approve if (income > 50k AND credit > 700) OR (income > 200k)" without you specifying the interaction.

**Real ML connection:** Trees are the building blocks of the most successful tabular-data models in production — Random Forests, XGBoost, LightGBM, CatBoost — which still dominate Kaggle tabular competitions and power fraud detection, credit scoring, recommendation ranking, and click-through prediction at scale.

**Toy example.** Predict whether someone will play tennis given weather:

```
           [Outlook?]
          /    |    \
       Sunny  Overcast  Rain
        /        |        \
  [Humidity?]   YES    [Wind?]
    /   \              /    \
  High  Norm         Strong Weak
   NO   YES           NO     YES
```

Each internal node is a test on one feature; each leaf is a prediction.

## 2. Conceptual Foundations

**Key terms.**

- **Node:** a point in the tree. **Root:** topmost. **Internal node:** has a test. **Leaf:** terminal, outputs a prediction.
- **Split:** a test of the form `x_j ≤ t` (numeric) or `x_j ∈ S` (categorical) that routes samples left or right.
- **Impurity:** a measure of how mixed the labels are in a node. Pure = all same class.
- **Recursive partitioning:** the algorithm that builds the tree by repeatedly choosing the best split and recursing into each child.

**How it works, step by step.**

1. Start with all training data at the root.
2. For every feature `j` and every candidate threshold `t`, compute how much a split on `(j, t)` would reduce impurity.
3. Pick the best `(j, t)`. Send samples with `x_j ≤ t` left, others right.
4. Recurse on each child.
5. Stop when a stopping criterion is met: max depth reached, node size too small, impurity below threshold, or no split improves the objective.
6. Each leaf stores a prediction — the majority class (classification) or mean target (regression).

**Assumptions.**

- Axis-aligned splits are adequate. (A tree cannot natively learn "x₁ + x₂ > 5"; it approximates diagonal boundaries with staircases.)
- Features are informative individually or through low-order interactions at each depth.
- Training data represents the deployment distribution.

**What breaks when assumptions are violated.**

- **Highly rotated/linear boundaries:** trees need many splits to approximate them, causing overfitting and poor generalization.
- **High-cardinality categoricals:** naive handling explodes the split search and biases splitting criteria toward such features.
- **Distribution shift:** leaves built on stale data give confidently wrong predictions.
- **Small, noisy data:** trees greedily memorize noise — a single tree has very high variance.

## 3. Mathematical Formulation

Let the training set be $\{(x_i, y_i)\}_{i=1}^n$ with $x_i \in \mathbb{R}^d$. At a node with samples $S$, let $p_k = \frac{1}{|S|}\sum_{i \in S} \mathbb{1}[y_i = k]$ be the fraction of class $k$.

### 3.1 Splitting criteria

**Gini impurity** (used in CART):
$$G(S) = \sum_{k=1}^K p_k(1 - p_k) = 1 - \sum_k p_k^2.$$

Intuition: probability that two random samples drawn from $S$ have different labels. Minimized (=0) when one $p_k = 1$.

**Entropy** (used in ID3/C4.5):
$$H(S) = -\sum_{k=1}^K p_k \log_2 p_k.$$

Intuition: expected bits to encode the class label. Zero when pure.

**Information gain** for a split that partitions $S$ into $S_L, S_R$:
$$IG(S, \text{split}) = H(S) - \frac{|S_L|}{|S|}H(S_L) - \frac{|S_R|}{|S|}H(S_R).$$

Choose the split maximizing $IG$ (or equivalently minimizing weighted child impurity).

**Variance reduction** (regression):
$$\Delta = \text{Var}(S) - \frac{|S_L|}{|S|}\text{Var}(S_L) - \frac{|S_R|}{|S|}\text{Var}(S_R),$$
where $\text{Var}(S) = \frac{1}{|S|}\sum_{i \in S}(y_i - \bar{y}_S)^2$. Equivalent to minimizing squared-error loss of the leaf mean prediction.

**Why Gini vs. entropy?** Both are concave; both reach 0 at purity and max at uniform. Empirically nearly indistinguishable. Gini is cheaper (no log). Entropy has a stronger information-theoretic interpretation.

### 3.2 Finding the best numeric split

For feature $j$, sort values $x_{(1)j} \le \dots \le x_{(n)j}$. Candidate thresholds are midpoints between consecutive distinct values. Scan left-to-right maintaining running class counts so each threshold evaluation is $O(1)$ after the initial $O(n \log n)$ sort. Total per node: $O(nd \log n)$.

### 3.3 Pruning

A fully grown tree overfits. Pruning reduces complexity.

**Cost-complexity pruning (CCP, aka weakest-link).** Define
$$R_\alpha(T) = R(T) + \alpha |T|,$$
where $R(T)$ is the training error (or misclassification rate) of tree $T$, $|T|$ is the number of leaves, and $\alpha \ge 0$ is a complexity penalty. For each internal node $t$ with subtree $T_t$, the **effective alpha** is
$$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1},$$
the per-leaf cost of keeping $T_t$ instead of collapsing it to a leaf. Prune the node with smallest $\alpha_{\text{eff}}$, repeat, producing a nested sequence $T_0 \supset T_1 \supset \dots$. Choose $\alpha$ (equivalently the tree in the sequence) by cross-validation.

**Reduced error pruning.** Simpler: hold out a validation set. For each internal node (bottom-up), replace its subtree with a leaf if validation accuracy does not decrease. Greedy and effective when data is plentiful.

### 3.4 Handling categorical variables

- **Binary target, K-category feature:** an optimal partition can be found in $O(K \log K)$ by sorting categories by their mean target value and scanning (Breiman's theorem). For multiclass or regression with non-convex objectives, heuristics apply.
- **One-hot encoding:** creates K binary features. Simple but hurts — trees can only isolate one category at a time, leading to unbalanced, deep splits.
- **Target/ordinal encoding:** replace category with its mean target (with smoothing/holdout to prevent leakage — see CatBoost below).

### 3.5 Missing values

- **Surrogate splits (CART):** at each split, store backup features that best mimic the primary split's routing. When the primary value is missing, use the surrogate.
- **Default direction (XGBoost):** during training, try routing missing values both ways and keep the direction that reduces loss more. Sparsity-aware.
- **Imputation before training:** possible but usually inferior.

### 3.6 Monotonic constraints

Sometimes domain knowledge requires $\hat{f}$ to be monotonically non-decreasing in a feature (e.g., predicted default probability should not decrease as debt increases). Enforce by: at every split on that feature, constrain the left child's leaf values to be $\le$ the right child's via propagated bounds during tree construction.

## 4. Worked Example

Dataset (predict PlayTennis):

| Outlook | Temp | Humidity | Wind | Play |
|---|---|---|---|---|
| Sunny | Hot | High | Weak | No |
| Sunny | Hot | High | Strong | No |
| Overcast | Hot | High | Weak | Yes |
| Rain | Mild | High | Weak | Yes |
| Rain | Cool | Normal | Weak | Yes |
| Rain | Cool | Normal | Strong | No |
| Overcast | Cool | Normal | Strong | Yes |
| Sunny | Mild | High | Weak | No |
| Sunny | Cool | Normal | Weak | Yes |
| Rain | Mild | Normal | Weak | Yes |
| Sunny | Mild | Normal | Strong | Yes |
| Overcast | Mild | High | Strong | Yes |
| Overcast | Hot | Normal | Weak | Yes |
| Rain | Mild | High | Strong | No |

14 samples: 9 Yes, 5 No. Root entropy:
$$H = -\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14} \approx 0.940.$$

Try splitting on **Outlook**. Sunny: 2Y/3N ($H=0.971$). Overcast: 4Y/0N ($H=0$). Rain: 3Y/2N ($H=0.971$).
$$H_{\text{split}} = \tfrac{5}{14}(0.971) + \tfrac{4}{14}(0) + \tfrac{5}{14}(0.971) = 0.694.$$
$$IG(\text{Outlook}) = 0.940 - 0.694 = 0.246.$$

Similar computations: IG(Humidity)=0.151, IG(Wind)=0.048, IG(Temp)=0.029. **Outlook wins** at the root.

Recurse. The Overcast branch is pure → leaf "Yes". For Sunny (2Y/3N): IG(Humidity on this subset) = 0.971 (perfectly separates) — splits cleanly. For Rain: Wind perfectly separates. The resulting tree is the one sketched in §1 and is consistent with the data.

## 5. Relevance to ML Practice

**Where used.**
- As a **standalone model** when interpretability is paramount (medical triage, regulatory models). Rare alone at scale because single trees are high-variance.
- As the **base learner** of every major tabular ensemble (RF, XGBoost, LightGBM, CatBoost).
- For **feature importance** and EDA — quick view of which features matter and how they interact.

**When to use.** Tabular data, mixed feature types, nonlinearities, modest interpretability needs. Small-to-medium datasets where deep learning underperforms.

**When not to use.** Unstructured data (images, text, audio) — CNNs/transformers dominate. Problems with strong linear structure (use linear models). Situations needing smooth probability estimates (trees produce piecewise-constant, poorly calibrated probabilities).

**Alternatives.** Linear/logistic models, kNN, SVMs, neural nets, and — more often — tree ensembles rather than a single tree.

**Trade-offs.**
- Bias–variance: shallow tree = high bias, low variance; deep tree = low bias, high variance.
- Interpretable up to a depth (~5–6); beyond that, humans lose track.
- Fast to train and predict; no feature scaling required.
- Unstable: small data perturbations yield different trees (motivates bagging).

## 6. Common Pitfalls & Misconceptions

1. **"Trees don't overfit."** Fully grown trees overfit catastrophically. Always prune or regularize (max_depth, min_samples_leaf, CCP).
2. **Confusing Gini with Gini coefficient** (inequality measure). Different quantities.
3. **One-hot encoding high-cardinality categoricals.** Harms tree splits. Prefer target encoding or native categorical support.
4. **Using training accuracy to pick the tree.** Always cross-validate the complexity parameter.
5. **Believing feature importance = causal importance.** Importance is split-based and biased toward high-cardinality and continuous features.
6. **Assuming trees handle extrapolation.** A regression tree predicts the mean of its training leaf — it cannot extrapolate beyond the training range.
7. **Ignoring class imbalance.** Impurity measures are biased when class priors are skewed; use class weights or rebalancing.

---

# Part II — Random Forests

## 1. Motivation & Intuition

A single decision tree is unstable: resample the data and you get a very different tree. This high variance is the single biggest weakness of trees. The cure is **ensembling** — train many trees and average them. If their errors are sufficiently decorrelated, averaging cancels them out while preserving signal.

**Wisdom of crowds.** If 100 diverse, weakly-correlated experts each get 60% accuracy, a majority vote is far above 60%. Random forests engineer that diversity.

**Real ML systems.** Random forests are a default first model for tabular data: they train in parallel, need little tuning, handle mixed types, give OOB error for free, and are robust baselines at companies like Netflix, banks, and pharma.

## 2. Conceptual Foundations

**Bagging (Bootstrap Aggregating).** Draw $B$ bootstrap samples (size-$n$ samples with replacement) from the training set. Train a tree on each. Average (regression) or majority-vote (classification) their predictions. Bagging reduces variance without increasing bias — provided base learners are high-variance and roughly unbiased (deep trees are perfect).

**Feature bagging (random subspaces).** At each split, consider only a random subset of $m$ features (typical: $m = \sqrt{d}$ for classification, $m = d/3$ for regression). This forces trees to be diverse: they can't all latch onto the same one or two dominant features.

**Out-of-bag (OOB) error.** Each bootstrap sample omits about $1/e \approx 36.8\%$ of the training data. For each sample $i$, aggregate predictions only from the trees that did NOT see $i$ in their bootstrap. This gives a cross-validation-like estimate for free.

**Feature importance.**
- **Mean Decrease Impurity (MDI):** sum of impurity reductions across all splits using feature $j$, averaged over trees. Fast but biased toward high-cardinality and continuous features.
- **Permutation importance:** permute feature $j$'s values in OOB data and measure accuracy drop. Less biased, more expensive, model-agnostic.

**ExtraTrees (Extremely Randomized Trees).** Two additional randomizations vs. RF: (i) splits use the full training set (no bootstrap by default), and (ii) thresholds are drawn **randomly** for each candidate feature rather than optimized. Faster, often comparable accuracy, slightly higher bias / lower variance.

**Assumptions.** Base trees have low bias (grown deep). Errors of different trees are weakly correlated. Training data is i.i.d.

**What breaks.**
- If all strong features dominate, trees become correlated — feature bagging mitigates this.
- If the signal is weak and the noise is huge, bagging cannot manufacture it.
- Under distribution shift, OOB error overstates deployment performance.

## 3. Mathematical Formulation

Let $\hat{f}_b(x)$ be the $b$-th tree. The forest predicts
$$\hat{f}_{\text{RF}}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(x) \quad (\text{regression}),$$
or a majority vote of $\{\hat{f}_b(x)\}$ (classification), or the average of class-probability vectors.

**Variance reduction, explained.** If $\hat{f}_b(x)$ are identically distributed (not independent) with variance $\sigma^2$ and pairwise correlation $\rho$, then
$$\text{Var}(\hat{f}_{\text{RF}}(x)) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2.$$
As $B \to \infty$ the second term vanishes — **the floor is $\rho \sigma^2$.** This is the key identity of bagging/RF: *reducing correlation $\rho$ is as important as adding more trees.* Feature bagging lowers $\rho$.

**Bootstrap math.** Prob a specific sample is not drawn in $n$ picks with replacement: $(1 - 1/n)^n \to e^{-1} \approx 0.368$.

**Bias.** $\mathbb{E}[\hat{f}_{\text{RF}}] = \mathbb{E}[\hat{f}_b]$, so bias is unchanged by bagging. All improvement comes from variance reduction.

## 4. Worked Example

Suppose each tree has test MSE = $\sigma^2 = 1.0$, and trees are uncorrelated ($\rho=0$). With $B=100$: forest variance = $0.01$ — a 100× reduction. With $\rho=0.3$ (typical without feature bagging): variance = $0.3 + 0.007 \approx 0.307$, only 3× reduction. Add feature bagging, drive $\rho$ to $0.1$: variance = $0.109$, about 9× reduction. This is why `max_features` is the most important RF hyperparameter after the number of trees.

**OOB example.** Train 500 trees on 10,000 samples. Each tree's bootstrap misses ~3,680 samples. Across 500 trees, each sample is OOB in ~$500/e \approx 184$ trees — plenty to aggregate a stable prediction. Average accuracy of these OOB aggregated predictions = OOB accuracy.

## 5. Relevance to ML Practice

**Where used.** Strong default for tabular classification/regression. Fraud, credit, churn, clinical risk. Often used for quick feature-importance screening before training a boosted model. Robust in low-data regimes where boosting overfits more easily.

**When to use.** You want a strong baseline with minimal tuning. You value OOB error, parallelizable training, and robustness.

**When not to use.** When you can afford boosting — XGBoost/LightGBM usually outperform RF by 1–5 percentage points on tabular benchmarks. When you need well-calibrated probabilities (RF probabilities are mid-quality; apply isotonic regression or Platt scaling). Very large datasets where memory of many deep trees is problematic.

**Alternatives.** Gradient boosting, single pruned trees, linear models, neural nets.

**Trade-offs.**
- Low variance, slightly higher bias than fully grown single tree — net win.
- Parallel training; serial prediction across trees.
- Memory: $B$ trees can be large.
- Less interpretable than a single tree but permutation importance and partial-dependence plots recover much of it.

## 6. Common Pitfalls & Misconceptions

1. **"More trees overfit."** They don't. Past a point, they just add cost. $B$ is a compute/accuracy trade-off, not a bias/variance one.
2. **Using default `max_features=d`.** Removes the feature-bagging benefit — you've just built a bagged ensemble with high correlation.
3. **Trusting MDI importance uncritically.** High-cardinality numeric features look artificially important.
4. **Comparing OOB to test accuracy and assuming equivalence under shift.** OOB is an internal estimate and inherits any train/test distribution mismatch.
5. **Thinking ExtraTrees are always faster or always better.** They skip threshold optimization (faster per split) but often need more trees.
6. **Bagging unstable base learners only.** Bagging stable models (linear regression) barely helps.

---

# Part III — Gradient Boosting

## 1. Motivation & Intuition

Bagging reduces variance by averaging **independently trained** models. Boosting takes a different angle: train models **sequentially**, each one fixing the residual errors of the current ensemble. This reduces bias (not just variance) and, empirically, wins.

**Intuition.** You predict house prices. Your first tree predicts $\hat{y}^{(1)}$. It's wrong by $r_i = y_i - \hat{y}_i^{(1)}$. Train a second tree to predict those residuals. Add it (with a small learning rate) to the first. Repeat. You're doing gradient descent in function space — each tree is a step toward the loss-minimizing function.

**Why it dominates tabular ML.** XGBoost (2016), LightGBM (2017), and CatBoost (2018) repeatedly win Kaggle competitions and power production systems at Uber, Microsoft, Yandex, and countless ad-tech companies. They combine tree flexibility with gradient-descent optimization and serious regularization.

## 2. Conceptual Foundations

**Additive model.** The ensemble is
$$F_M(x) = \sum_{m=0}^M \nu \cdot h_m(x),$$
where $h_m$ is the $m$-th weak learner (tree) and $\nu \in (0,1]$ is the **learning rate** (shrinkage).

**Pseudo-residuals.** For loss $L(y, F)$, define
$$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}.$$
For squared loss, $r_{im} = y_i - F_{m-1}(x_i)$ — literal residuals. For log-loss, $r_{im} = y_i - p_i$ where $p_i = \sigma(F_{m-1}(x_i))$. Each new tree $h_m$ is fit to these pseudo-residuals.

**Line search / leaf values.** After fitting $h_m$'s structure, choose the leaf values to minimize the actual loss (not just squared error on residuals) — yielding "Newton-like" steps in XGBoost.

**Assumptions.** Loss is differentiable. Learning rate is small enough. Regularization (tree depth, leaf counts, L2 on leaf values) controls overfitting.

**What breaks.** Large learning rates → divergence or wild overfitting. Too many trees without early stopping → overfit. Noisy labels → boosting aggressively memorizes them.

## 3. Mathematical Formulation

### 3.1 Generic gradient boosting (Friedman, 2001)

1. Init $F_0(x) = \arg\min_c \sum_i L(y_i, c)$ (e.g., mean for MSE, log-odds for log-loss).
2. For $m = 1, \dots, M$:
   - Compute pseudo-residuals $r_{im}$.
   - Fit regression tree $h_m$ to $\{(x_i, r_{im})\}$, producing leaves $R_{jm}$.
   - For each leaf, compute optimal value $\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$.
   - Update $F_m(x) = F_{m-1}(x) + \nu \sum_j \gamma_{jm} \mathbb{1}[x \in R_{jm}]$.

### 3.2 XGBoost — second-order objective

XGBoost minimizes, at iteration $t$,
$$\mathcal{L}^{(t)} = \sum_i L(y_i, F_{t-1}(x_i) + h_t(x_i)) + \Omega(h_t),$$
with regularizer
$$\Omega(h) = \gamma T + \tfrac{1}{2}\lambda \sum_{j=1}^T w_j^2,$$
where $T$ is #leaves, $w_j$ is the value of leaf $j$, $\gamma$ penalizes leaf count, $\lambda$ is L2 on leaf values. A second-order Taylor expansion around $F_{t-1}$ gives
$$\mathcal{L}^{(t)} \approx \sum_i \left[g_i h_t(x_i) + \tfrac{1}{2} h_i h_t(x_i)^2\right] + \Omega(h_t),$$
with $g_i = \partial_F L$, $h_i = \partial_F^2 L$. For a fixed tree structure with leaves having sample sets $I_j$, the optimal leaf weight is
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda},$$
and the optimal loss is
$$\mathcal{L}^* = -\tfrac{1}{2} \sum_j \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T.$$
The **split gain** for splitting a leaf into $L, R$ is
$$\text{Gain} = \tfrac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma.$$
Splits with $\text{Gain} < 0$ are rejected — this is implicit pre-pruning. Explicit post-pruning also collapses negative-gain branches.

**Sparsity awareness.** XGBoost learns a "default direction" per split for missing/zero values by trying both directions during split search.

**Approximate splitting.** On huge data, exact threshold search is costly. XGBoost uses a weighted quantile sketch: propose candidate split points at quantiles of the feature distribution weighted by $h_i$ (the second-order info). Typically 32–256 bins give near-exact accuracy at a fraction of the cost.

### 3.3 LightGBM — speed innovations

- **Histogram binning.** Discretize each feature into 255 bins up front; split search is $O(\#\text{bins})$ per feature per node, vs. $O(n)$ for exact.
- **Leaf-wise growth.** Instead of level-wise (grow all leaves at depth $d$ before $d+1$), grow the single leaf with highest loss reduction. Faster to low loss, more prone to overfitting — control with `max_depth` or `num_leaves`.
- **GOSS (Gradient-based One-Side Sampling).** Keep all samples with large gradients (hard examples) and randomly subsample those with small gradients, reweighting to keep the gradient sum unbiased. Typical: retain top $a\%$ by $|g|$, sample $b\%$ of the rest, reweight sampled ones by $(1-a)/b$.
- **EFB (Exclusive Feature Bundling).** In high-dimensional sparse data, many features are mutually exclusive (rarely nonzero together — e.g., one-hot groups). Bundle them into a single feature with offset encoding, cutting feature count dramatically.

### 3.4 CatBoost — categorical and bias innovations

- **Ordered Target Statistics.** Replace category $c$ with the mean of $y$ over **prior** examples sharing that category, according to a random permutation of the data:
$$\hat{x}_{\sigma(i), c} = \frac{\sum_{j: \sigma(j) < \sigma(i),\, x_{jc}=c} y_j + a \cdot P}{\sum_{j: \sigma(j)<\sigma(i),\, x_{jc}=c} 1 + a}.$$
Prior $P$ and smoothing $a$ prevent overfitting when categories are rare. Multiple permutations reduce variance.
- **Ordered Boosting.** Standard boosting computes gradients using residuals from a model trained on the **same** data — **target leakage** ("prediction shift"). CatBoost trains a separate model $M_i$ that doesn't include sample $i$ and uses $M_i$ to compute $i$'s gradient. Implemented efficiently via permutations.
- **Oblivious (symmetric) trees.** At each depth all nodes use the same $(j, t)$. This makes the tree a balanced binary decision table with $2^d$ leaves indexed by binary features. Pros: very fast inference (vectorizable), strong regularization (less expressive per tree, so boosting has to do real work). Cons: slightly higher bias per tree.

## 4. Worked Example

Regression on 5 points with squared loss, learning rate $\nu=0.5$, stumps (depth-1 trees).

Data: $x = [1,2,3,4,5]$, $y = [1,2,4,4,6]$.

**Init:** $F_0 = \bar{y} = 3.4$. Residuals: $[-2.4, -1.4, 0.6, 0.6, 2.6]$.

**Iter 1.** Fit a stump to residuals. Best split at $x \le 2.5$: left mean $= -1.9$, right mean $= 1.267$. Update:
$F_1(x) = 3.4 + 0.5 \cdot h_1(x)$. So $F_1 = [2.45, 2.45, 4.033, 4.033, 4.033]$.

New residuals: $[-1.45, -0.45, -0.033, -0.033, 1.967]$.

**Iter 2.** Best stump split at $x \le 4.5$: left mean $\approx -0.491$, right mean $=1.967$. Update:
$F_2 = F_1 + 0.5 \cdot h_2$. At $x=5$: $F_2 = 4.033 + 0.5(1.967) = 5.017$. At $x=1$: $F_2 = 2.45 + 0.5(-0.491) = 2.205$.

After 2 iterations predictions $[2.20, 2.20, 3.79, 3.79, 5.02]$ vs. truth $[1,2,4,4,6]$ — already much better than $F_0=3.4$ for all. Continue for many iterations and watch the training residuals shrink (and test residuals eventually stop shrinking — use early stopping).

## 5. Relevance to ML Practice

**Where used.** Click-through rate prediction, ad ranking, fraud detection, credit scoring, insurance pricing, demand forecasting, medical risk, competition-winning tabular ML. Often deployed as part of a two-stage pipeline (retrieval + ranking).

**When to use.** Tabular data of any size; missing values / mixed types; interactions; you care about predictive accuracy and can afford hyperparameter tuning.

**When not to use.** Images/text/audio (use deep nets). When you need a fully interpretable single model (use a single pruned tree or GAM). When training-serving latency budgets are extremely tight and you can't afford a 100–1000-tree ensemble (though all three libraries have fast inference modes and can export to ONNX/C).

**Choosing among XGBoost / LightGBM / CatBoost.**
- **XGBoost:** most mature, excellent defaults, great docs, robust on medium data. Slightly slower on huge data than LightGBM.
- **LightGBM:** fastest on large data, great for high-dimensional sparse features, leaf-wise growth needs regularization attention.
- **CatBoost:** best out-of-the-box on data with many categorical variables, least tuning, strong defaults; slightly slower training but very fast inference from oblivious trees.

**Trade-offs.**
- Lower bias than RF, potentially higher variance → requires early stopping.
- Sequential training (harder to parallelize across trees; parallelism is within a tree).
- More hyperparameters than RF: learning rate, depth, #trees, subsample, colsample, L1/L2, min_child_weight, etc.
- Tune typically via early stopping + learning rate (small $\nu$ with many trees beats large $\nu$ with few).

## 6. Common Pitfalls & Misconceptions

1. **No early stopping.** Training for a fixed, large `n_estimators` overfits. Always use a validation set with `early_stopping_rounds`.
2. **Huge learning rate.** $\nu = 0.3$ is fast but unstable. Prefer $\nu \in [0.01, 0.1]$ with more trees.
3. **Tuning too many knobs blindly.** Tune depth / num_leaves, learning rate, min_child_weight, and subsampling first. L1/L2 second.
4. **Treating feature importance as causal.** Same caveat as RF, more acute because boosting aggressively exploits leakage.
5. **Target leakage via naive target encoding.** CatBoost handles this; elsewhere use K-fold target encoding with out-of-fold statistics.
6. **Ignoring class imbalance.** Use `scale_pos_weight` or resampling; don't just threshold at 0.5.
7. **Assuming calibration.** Boosted probabilities can be miscalibrated (especially with small leaves and strong regularization). Apply Platt / isotonic calibration if you use the probabilities directly.
8. **Forgetting that LightGBM's leaf-wise growth overfits more.** Cap `num_leaves` and `max_depth`.

---

# Part IV — Stacking & Voting

## 1. Motivation & Intuition

Different models make different errors. A linear model captures global linear trends; a tree captures local interactions; a neural net captures smooth nonlinearities. If we could **combine** them intelligently, we might exceed any individual model.

**Voting** does this by simple averaging/majority. **Stacking** goes further: it learns **how** to combine them.

**Real ML.** Every Kaggle-winning solution stacks. Production systems often use a single model for latency, but A/B test stacked ensembles at the top of the funnel or in offline evaluation.

## 2. Conceptual Foundations

**Voting ensembles.**
- **Hard voting (classification):** each model votes for a class; majority wins.
- **Soft voting:** average predicted probabilities; pick argmax. Usually better because it uses confidence.
- **Weighted voting:** weights reflect each model's validation performance. Weights can be tuned (e.g., by grid search or constrained optimization on a validation set).

**Stacking.**
1. Split training data into K folds.
2. For each **base model** $f_m$, produce **out-of-fold (OOF) predictions** $\hat{y}_i^{(m)}$ on every training sample — using only folds that excluded sample $i$.
3. Construct a meta-feature matrix $Z \in \mathbb{R}^{n \times M}$ whose rows are $(\hat{y}_i^{(1)}, \dots, \hat{y}_i^{(M)})$ and train a **meta-learner** (usually simple — logistic/linear regression, or a shallow GBDT) on $(Z, y)$.
4. At inference, each base model (retrained on all training data) predicts; meta-learner combines.

**Blending.** A lighter variant: hold out a single validation set (say 20%). Train base models on 80%, predict on validation, train meta-learner on those 20% predictions. Simpler, less data-efficient, no fold plumbing; susceptible to overfitting the holdout if you have many base models.

**Assumptions.** Base model errors are decorrelated. OOF predictions are produced correctly (no leakage). Meta-learner doesn't overfit (choose it simple relative to the number of training rows and number of base models).

**What breaks.**
- **Leakage in OOF:** if any preprocessing (target encoding, scaling with train+val stats, feature selection) leaks across folds, the meta-learner overfits and deployment performance collapses.
- **Correlated base models:** if all base models are GBDTs with slightly different seeds, the meta-learner has nothing to gain from combination.

## 3. Mathematical Formulation

**Soft voting.** With $M$ classifiers producing $p_m(y=k|x)$, ensemble is
$$\hat{p}(y=k|x) = \sum_m w_m p_m(y=k|x), \quad \sum_m w_m = 1.$$
If $p_m$ are unbiased with variance $\sigma_m^2$ and pairwise correlation $\rho_{mm'}$, weighted average variance is
$$\text{Var} = \sum_m w_m^2 \sigma_m^2 + \sum_{m \ne m'} w_m w_{m'} \rho_{mm'} \sigma_m \sigma_{m'},$$
minimized (for equal $\sigma$) when $w_m$ tilt toward models with low correlation to others.

**Stacking regression.** Meta-learner is
$$\hat{y}(x) = g\bigl(f_1(x), \dots, f_M(x); \theta\bigr).$$
Often $g$ is linear: $\hat{y} = \sum_m \beta_m f_m(x) + \beta_0$, with $\beta$ trained by OLS (or constrained to $\beta_m \ge 0$, $\sum \beta_m = 1$ for interpretability).

## 4. Worked Example

Binary classification, three base models: logistic regression (LR), random forest (RF), gradient boosting (GB). Meta: logistic regression.

1. 5-fold split. For fold $k$, train LR/RF/GB on remaining 4 folds, predict on fold $k$. After 5 folds, each training sample has OOF probabilities $(p_{LR}, p_{RF}, p_{GB})$.
2. Train meta-LR on these triples → learns, say, $\hat{p} = \sigma(-2 + 0.8 p_{LR} + 2.1 p_{RF} + 3.3 p_{GB})$. GB dominates but RF and LR add information.
3. Retrain LR/RF/GB on full training data. At inference: feed $x$ to all three, get $(p_{LR}, p_{RF}, p_{GB})$, pass through meta-LR, output $\hat{p}$.

Expected gain: typically 0.1–1 AUC point over the best single model when base models are diverse.

## 5. Relevance to ML Practice

**Where used.** Kaggle competitions (nearly universal). Offline top-of-stack models. Robust production ranking systems where diversity hedges against drift of any single model.

**When to use.** You've exhausted single-model tuning and need more accuracy; you have diverse base models; you have enough data for trustworthy OOF.

**When not to use.** Latency-critical systems (stacking multiplies inference cost and deployment complexity). Small datasets where OOF is noisy. When a single well-tuned GBDT is within the needed tolerance — simplicity wins.

**Trade-offs.** Accuracy vs. complexity/latency/maintainability. Debuggability worsens. Retraining cadence becomes a coordination problem. Interpretability suffers.

## 6. Common Pitfalls & Misconceptions

1. **Using in-fold predictions for meta-training.** Base models have seen those labels → meta-learner learns to trust them, catastrophic test failure. Always OOF.
2. **Overly complex meta-learner.** A deep net meta over 5 base models on 10k rows overfits. Keep meta simple (linear, ridge, shallow GBDT).
3. **Identical base models.** No diversity = no gain. Mix model families.
4. **Leakage via preprocessing.** Fit target encoders / scalers inside each fold, not globally.
5. **Confusing soft and hard voting.** Hard voting can discard strong confidence signals and often underperforms soft voting when probabilities are well-calibrated.
6. **Blending-as-stacking confusion.** Blending uses a single holdout; stacking uses K-fold OOF. Blending is simpler and riskier on small data.

---

# Interview Preparation

## Foundational

**Q1. Why does a random forest reduce variance but not bias compared to a single deep tree?**
A deep tree is approximately unbiased but high-variance. Averaging many such trees keeps $\mathbb{E}[\hat{f}_{\text{RF}}] = \mathbb{E}[\hat{f}_b]$ (bias unchanged) but, for correlation $\rho$ and tree variance $\sigma^2$, $\text{Var}(\hat{f}_{\text{RF}}) = \rho\sigma^2 + (1-\rho)\sigma^2/B$. Bagging drives down the second term; feature bagging drives down $\rho$. Net: variance reduction with unchanged bias.

**Q2. Gini vs. entropy — which to use?**
Empirically equivalent. Gini is $\sum p_k(1-p_k)$, entropy is $-\sum p_k \log p_k$. Both concave, zero at purity. Gini is cheaper (no log). Choose based on framework defaults.

**Q3. What is out-of-bag error and when does it fail?**
For each training sample, aggregate predictions from trees whose bootstrap didn't include it; compute accuracy/MSE. It's a near-free CV-like estimate. Fails under distribution shift, time-series dependence (bootstrap breaks temporal order), or when $B$ is so small that some samples have few OOB trees.

**Q4. Why can gradient boosting overfit while random forests don't (much)?**
Each boosting iteration fits residuals, so enough iterations can memorize training labels. RF trees are trained independently on resampled data; adding more trees only averages more noise, not memorizing it further. Hence boosting requires early stopping.

## Mathematical

**Q5. Derive the optimal leaf weight in XGBoost.**
Second-order approximation: $\mathcal{L}^{(t)} \approx \sum_j \bigl[ (\sum_{i \in I_j} g_i) w_j + \tfrac{1}{2}(\sum_{i \in I_j} h_i + \lambda) w_j^2\bigr] + \gamma T$. Differentiate w.r.t. $w_j$: $\sum g_i + (\sum h_i + \lambda) w_j = 0 \Rightarrow w_j^* = -\frac{\sum g_i}{\sum h_i + \lambda}$. Substitute back to get $\mathcal{L}^* = -\tfrac{1}{2}\sum_j \frac{(\sum g_i)^2}{\sum h_i + \lambda} + \gamma T$. The split-gain formula follows by taking the change in $\mathcal{L}^*$ when splitting a leaf.

**Q6. Why is $1/e$ the fraction of OOB samples?**
Each of $n$ draws with replacement misses a specific sample with probability $1 - 1/n$. Over $n$ draws: $(1 - 1/n)^n \to 1/e \approx 0.368$ as $n \to \infty$.

**Q7. Show that fitting a tree to negative gradients of squared loss equals fitting residuals.**
$L = \tfrac{1}{2}(y - F)^2 \Rightarrow -\partial L/\partial F = y - F$. So pseudo-residuals = residuals.

**Q8. Why does CatBoost's ordered boosting reduce prediction shift?**
Standard boosting computes sample $i$'s gradient using a model trained on $i$ itself — leakage causes systematic bias because leaf values overfit. Ordered boosting computes $i$'s gradient from a model $M_i$ trained without $i$ (implemented via permutation-based incremental models), so the gradient is unbiased w.r.t. the target and the ensemble generalizes better, particularly with heavy categorical encoding.

**Q9. When does bagging fail to reduce variance?**
When base learners are stable (low variance): e.g., bagging a linear regression with enough data barely changes variance. Bagging pays off with unstable (high-variance) learners like deep trees or kNN with small k.

## Applied

**Q10. Your RF has 99% training accuracy, 70% test accuracy. What do you do?**
Classic overfit. (i) Reduce `max_depth`, increase `min_samples_leaf`, increase `max_features` diversity by lowering it. (ii) Check for label noise / leakage — if a feature perfectly predicts the target in train only, you have leakage. (iii) Verify train/test distributions match (covariate shift). (iv) Consider fewer, more diverse features or feature selection. (v) Cross-validate to confirm the gap is real.

**Q11. You have 1000 categorical features with high cardinality. Which GBDT and why?**
CatBoost first — native ordered target statistics handle high cardinality without leakage. LightGBM is second — its native categorical handling is fast but less principled on target-leakage. XGBoost would require careful target encoding with K-fold OOF.

**Q12. Design a ranking system using GBDT for web search.**
Use LightGBM/XGBoost with a ranking objective (LambdaRank / pairwise). Features: query–doc relevance (BM25, embeddings), query features (length, intent), document features (PageRank, freshness), user features (CTR history). Train on clicks with relevance labels; use NDCG@k for evaluation. Early-stop on held-out queries (group-level, not row-level, split). Serve with feature pipelines and model in ONNX for latency; monitor NDCG drift.

**Q13. When would you choose RF over XGBoost in production?**
Small data (<10k rows) where boosting overfits even with early stopping; when retraining cadence is frequent and you want robust, low-tuning defaults; when you need OOB validation; when parallel training is needed and you have many cores but few tuning cycles.

## Debugging & Failure-mode

**Q14. Training loss decreases but validation loss increases after iteration 200. What's happening?**
Overfitting. Use `early_stopping_rounds` (e.g., stop if val doesn't improve for 50 rounds). Reduce learning rate and increase tree count; increase min_child_weight; add L2 regularization; reduce depth.

**Q15. Your model's MDI feature importance ranks `user_id_hash` highest. Why?**
MDI is biased toward high-cardinality features — they offer more candidate splits and spuriously reduce impurity on noise. Remove ID-like features. Use permutation importance or SHAP instead for less biased attribution.

**Q16. Probabilities from your gradient-boosted classifier look extreme (many near 0 and 1), and Brier score is bad. Fix?**
Boosted trees can be miscalibrated due to aggressive fitting. Apply post-hoc calibration (Platt/logistic or isotonic) on a held-out set. Alternatively increase regularization (shallower trees, higher min_child_weight) which tempers confidence.

**Q17. Stacked model improves CV AUC by 0.02 but production AUC is worse than the best base model. Likely cause?**
Leakage in OOF — e.g., target encoding computed on the full training set before folding, so meta-features are over-optimistic. Or distribution shift affected one base model more than others; meta-learner over-relied on it. Rebuild OOF with strict per-fold preprocessing; monitor base-model drift.

## Follow-up / Probing

**Q18. You said "reduce correlation among trees." What actually controls that correlation in practice?**
(i) `max_features`: smaller → more diverse trees. (ii) Bootstrap sampling. (iii) Extremely random splits (ExtraTrees). (iv) Row subsampling per tree. (v) Different random seeds alone are insufficient because the greedy split algorithm is largely deterministic given the same data and features.

**Q19. Why does learning rate shrinkage help boosting?**
Each tree makes a small step in function space. Small $\nu$ means many small, averaged steps — like SGD with small step size — which smooths the approximation path and prevents any single tree from overcommitting to noisy residuals. Empirically, smaller $\nu$ + more trees + early stopping consistently outperforms large $\nu$.

**Q20. Explain the bias–variance trade-off in boosting depth.**
Deeper trees reduce bias per iteration (capture more interactions) but increase variance and correlation among successive trees, making the ensemble overfit faster. Typical sweet spot: depth 4–8 for XGBoost, num_leaves 31–127 for LightGBM. Task-dependent: deeper for high-interaction problems (recsys), shallower for smooth signals.

**Q21. In what sense is gradient boosting "gradient descent in function space"?**
Consider loss functional $L[F] = \sum_i L(y_i, F(x_i))$. Its functional gradient at $F_{m-1}$, evaluated at $x_i$, is $\partial L(y_i, F_{m-1}(x_i))/\partial F(x_i) = -r_{im}$. Moving $F$ in the direction of the negative gradient means adding $h_m \approx r_{im}$. Since the step must be a representable function, we project the gradient onto the base-learner class (regression trees) by fitting $h_m$ to residuals. The learning rate $\nu$ is the step size.

**Q22. Oblivious trees in CatBoost — why do they regularize?**
Requiring all nodes at a depth to share the same split drastically reduces the model class per tree: a depth-$d$ oblivious tree has only $d$ distinct splits (vs. up to $2^d - 1$ in a general binary tree). This higher bias per tree forces boosting to use more trees and combine information more broadly, reducing variance. It also makes inference a simple binary indexing — very fast at serving time.

**Q23. How would you detect and prevent target leakage with target encoding?**
Detect: suspiciously high training performance with low test performance; a feature's importance spikes after adding it. Prevent: compute target statistics within K-fold OOF — for each fold, encode using only other folds' statistics. Add smoothing $\hat{x} = \frac{n_c \bar{y}_c + a \bar{y}}{n_c + a}$ to stabilize rare categories. CatBoost does this automatically via permutation-based ordered TS.

**Q24. What's the computational complexity of exact vs. histogram GBDT split finding?**
Exact: $O(n d)$ per node after pre-sorting (with $O(n d \log n)$ preprocessing). Histogram: $O(n d)$ once to build histograms per tree, then $O(d \cdot B)$ per split where $B \ll n$ is #bins (e.g., 255). Histogram wins dramatically on large $n$ and is the default in LightGBM and modern XGBoost (`tree_method=hist`).

**Q25. When would voting outperform stacking?**
When the meta-learner overfits due to few rows or noisy OOF predictions; when base models are similarly accurate and errors are roughly symmetric; when simplicity/latency matter. A well-weighted soft vote is often within noise of stacking and far simpler to deploy and debug.
