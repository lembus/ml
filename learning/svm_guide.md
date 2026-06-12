# Support Vector Machines: A Comprehensive Learning Guide

---

## Table of Contents

1. Linear SVMs: Hard Margin and Soft Margin
2. Dual Formulation: Lagrange Multipliers and KKT Conditions
3. Kernel Methods: The Kernel Trick, Mercer's Theorem, and Common Kernels
4. Support Vectors: Sparsity and Geometric Interpretation
5. Extensions: ν-SVM, One-Class SVM, Structured SVM
6. Multiclass Strategies: One-vs-One, One-vs-All, DAGs
7. Interview Preparation

---

# 1. Linear SVMs: Hard Margin and Soft Margin

## 1.1 Motivation & Intuition

Imagine you run a small email service, and you want to automatically classify incoming messages as **spam** or **not spam**. Each email can be represented by a vector of numerical features — perhaps word counts, the presence or absence of certain phrases, the sender's reputation score, and so on. Now suppose you plot a handful of emails in a 2D space where the x-axis is "number of times the word *FREE* appears" and the y-axis is "number of exclamation marks." You notice that spam tends to cluster in the upper right, and legitimate mail clusters in the lower left.

Your goal is to draw a line separating these two clusters so that, when a new email arrives, you can check which side of the line it falls on and classify it accordingly. This is called a **linear classifier**. But here's the thing: if the clusters are well separated, there are *infinitely many* lines that could do the job. Which one should you choose?

The Support Vector Machine (SVM) answers this question with a simple, powerful principle: **choose the line that maximizes the distance to the closest point of either class**. This distance is called the **margin**. Intuitively, a line that sits far from all data points is more robust — small perturbations of the data (a bit of noise, a slightly weird new email) are less likely to push a point onto the wrong side.

This "maximum margin" principle is not just aesthetically pleasing. It has deep theoretical justification: classifiers with larger margins tend to generalize better to unseen data, a fact that can be formalized using statistical learning theory (Rademacher complexity bounds, VC dimension arguments, and PAC-Bayes bounds all give supporting reasons).

### Real-world ML connections

- **Text classification**: SVMs were the dominant method for document classification for over a decade before deep learning took over. Their ability to handle very high-dimensional, sparse feature vectors made them a natural fit.
- **Bioinformatics**: Gene expression data often has thousands of features and relatively few samples. SVMs handle this "$p \gg n$" regime gracefully.
- **Image classification** (pre-deep-learning): Combined with hand-crafted features (SIFT, HOG), SVMs were state of the art on many benchmarks.
- **Novelty detection**: The one-class SVM variant is still used for anomaly detection in industrial monitoring.

### Hard margin vs. soft margin: the core distinction

A **hard margin** SVM insists on perfect separation: no training point is allowed to be misclassified or even inside the margin. This only works when the data is **linearly separable** — which is rarely the case in practice. Real data has noise, overlapping classes, mislabeled examples.

A **soft margin** SVM allows some points to violate the margin (even to be misclassified), but penalizes these violations. This makes the method practical on real data and introduces a regularization hyperparameter — conventionally called $C$ — that controls the trade-off between margin size and violation penalty.

## 1.2 Conceptual Foundations

### Key terms

- **Hyperplane**: In $d$-dimensional space, a hyperplane is a $(d-1)$-dimensional flat surface that splits the space into two halves. In 2D it's a line; in 3D a plane; in higher dimensions we call it a hyperplane. Every hyperplane can be written as $\{x : w^\top x + b = 0\}$, where $w \in \mathbb{R}^d$ is a normal vector perpendicular to the hyperplane, and $b \in \mathbb{R}$ is a scalar offset (bias).
- **Decision function**: Given $w$ and $b$, we classify a point $x$ based on the sign of $f(x) = w^\top x + b$. If $f(x) > 0$ we predict class $+1$; if $f(x) < 0$ we predict class $-1$. The labels $y \in \{-1, +1\}$ are the conventional encoding for SVMs (as opposed to $\{0, 1\}$).
- **Functional margin** of a point $(x_i, y_i)$: $\hat{\gamma}_i = y_i (w^\top x_i + b)$. Positive means correctly classified; negative means wrong side.
- **Geometric margin**: The actual Euclidean distance from $x_i$ to the hyperplane, which equals $\hat{\gamma}_i / \|w\|$. The geometric margin is invariant to rescaling $(w, b)$; the functional margin is not.
- **Margin of the classifier**: The minimum geometric margin over the training set, $\gamma = \min_i \hat{\gamma}_i / \|w\|$.
- **Support vectors**: Training points that lie exactly on the margin boundary (and in the soft-margin case, points inside the margin or on the wrong side). These are the only points that influence the final classifier.

### How the pieces interact

The SVM optimization looks for the hyperplane $(w, b)$ that maximizes the geometric margin, subject to all points being correctly classified (hard margin) or being "mostly" correctly classified (soft margin). The support vectors are the points that "define" the solution — remove any non-support-vector and the solution is unchanged.

### Assumptions

- **Hard margin SVM** assumes the data is *linearly separable*. If this fails, no solution exists.
- Both variants assume **features are meaningful in a Euclidean sense** — distances and inner products matter. This makes feature scaling important.
- Classical linear SVM assumes **i.i.d. data from some fixed distribution** (as with most ERM-based learners).
- SVMs are **binary by default**; multiclass requires additional machinery (Section 6).

### What breaks when assumptions fail

- Non-separable data: hard-margin SVM has no feasible solution. The optimizer will either fail or (in naive implementations) blow up.
- Badly scaled features: the margin geometry is dominated by whichever feature has the largest scale, distorting the solution. This is why standardization (zero mean, unit variance) is standard practice.
- Non-i.i.d. or heavily imbalanced data: standard SVM can over-fit to the majority class. Class weighting or resampling helps.
- Presence of outliers: hard-margin SVM is extremely sensitive — a single outlier near the opposing class can ruin the solution. Soft margin was invented largely to address this.

## 1.3 Mathematical Formulation

### Notation setup

Let the training data be $\{(x_i, y_i)\}_{i=1}^n$ with $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$. We seek a linear decision function $f(x) = w^\top x + b$.

### The functional margin and the rescaling trick

For any point $x_i$, its signed distance to the hyperplane $\{x : w^\top x + b = 0\}$ is

$$
\text{dist}(x_i) = \frac{w^\top x_i + b}{\|w\|}.
$$

The *signed* distance becomes positive when $x_i$ is on the $+1$ side and negative on the $-1$ side. Multiplying by $y_i$ gives the geometric margin:

$$
\gamma_i = \frac{y_i (w^\top x_i + b)}{\|w\|}.
$$

Notice that $(w, b)$ and $(cw, cb)$ for any $c > 0$ define the *same hyperplane*. The functional margin $y_i(w^\top x_i + b)$ scales with $c$, but the geometric margin does not. This gives us a useful normalization: we are free to choose $(w, b)$ so that the smallest functional margin equals $1$:

$$
\min_i y_i(w^\top x_i + b) = 1.
$$

Under this normalization, the geometric margin equals $1/\|w\|$. So **maximizing the margin is equivalent to minimizing $\|w\|$** (or equivalently $\frac{1}{2}\|w\|^2$, a convenient convex differentiable objective).

### Hard-margin primal

The hard-margin SVM is the convex quadratic program:

$$
\boxed{\min_{w, b} \; \tfrac{1}{2}\|w\|^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \geq 1, \; i = 1, \ldots, n.}
$$

The objective is strictly convex in $w$; the constraints are linear. If the data is linearly separable, this problem has a unique minimizer in $w$ (the bias $b$ is also unique once $w$ is fixed).

### Soft-margin primal

When the data is not separable, we introduce **slack variables** $\xi_i \geq 0$, one per training point. The slack $\xi_i$ measures how much the constraint $y_i(w^\top x_i + b) \geq 1$ is violated:

- $\xi_i = 0$: point is on or beyond the correct margin boundary.
- $0 < \xi_i < 1$: point is inside the margin but still correctly classified.
- $\xi_i = 1$: point lies exactly on the separating hyperplane.
- $\xi_i > 1$: point is misclassified.

The soft-margin SVM is:

$$
\boxed{\min_{w, b, \xi} \; \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0.}
$$

The hyperparameter $C > 0$ controls the trade-off:

- Large $C$: heavy penalty on slack, so the model tries hard to classify every training point correctly. This approaches hard-margin behavior and can overfit.
- Small $C$: more tolerance for violations, wider margin, more regularization. Can underfit if too small.

### Hinge-loss reformulation

We can eliminate the slack variables explicitly. The constraint $\xi_i \geq \max(0, 1 - y_i(w^\top x_i + b))$ is tight at the optimum (we always push $\xi_i$ down as far as allowed). Substituting gives the **unconstrained hinge-loss form**:

$$
\min_{w, b} \; \tfrac{1}{2}\|w\|^2 + C\sum_{i=1}^n \max\bigl(0, \; 1 - y_i(w^\top x_i + b)\bigr).
$$

This is the form most often seen in modern ML texts. The hinge loss $\max(0, 1 - y f(x))$ is convex, piecewise linear, and zero whenever $y f(x) \geq 1$ (i.e., the point is correctly classified with margin at least 1). It is *not* differentiable at $y f(x) = 1$, but subgradient methods or smoothed variants work fine.

Dividing through by $C$ and letting $\lambda = 1/(nC)$ gives an equivalent form

$$
\min_{w, b} \; \frac{1}{n}\sum_{i=1}^n \max\bigl(0, 1 - y_i(w^\top x_i + b)\bigr) + \lambda \|w\|^2,
$$

making the connection to standard regularized risk minimization explicit: $\ell_2$ regularization plus the hinge loss.

### Connecting math to intuition

- $\tfrac{1}{2}\|w\|^2$: we want the normal vector short, because margin $= 1/\|w\|$.
- Constraint $y_i(w^\top x_i + b) \geq 1$: every point should be *at least one functional unit* on the correct side.
- Slack $\xi_i$: how much point $i$ is willing to violate this requirement.
- $C \sum \xi_i$: total penalty for violations; $C$ is the exchange rate between "margin width" currency and "violation" currency.

## 1.4 Worked Example

Let's work through a concrete 2D hard-margin problem.

### Setup

Four training points in $\mathbb{R}^2$:

- $x_1 = (1, 1)$, $y_1 = +1$
- $x_2 = (2, 2)$, $y_2 = +1$
- $x_3 = (-1, -1)$, $y_3 = -1$
- $x_4 = (-2, -2)$, $y_4 = -1$

These lie on the line $x^{(2)} = x^{(1)}$, with positives in the upper-right and negatives in the lower-left. By symmetry, the optimal separator should pass through the origin with normal vector pointing in the $(1, 1)/\sqrt{2}$ direction.

### Guessing the solution

Try $w = (a, a)$ and $b = 0$ for some $a > 0$. The functional margin at $x_1 = (1,1)$ is $y_1 (w^\top x_1 + b) = 1 \cdot (a + a) = 2a$. For this to equal 1 (the normalization), we need $a = 1/2$.

So $w^\star = (1/2, 1/2)$, $b^\star = 0$, and $\|w^\star\|^2 = 1/2$. Geometric margin $= 1/\|w^\star\| = \sqrt{2}$.

Let's verify: the distance from $(1, 1)$ to the line $x^{(1)} + x^{(2)} = 0$ is $|1 + 1|/\sqrt{1^2 + 1^2} = 2/\sqrt{2} = \sqrt{2}$. ✓

The **support vectors** are $x_1$ and $x_3$, the points achieving the minimum margin. Points $x_2$ and $x_4$ are further from the hyperplane and don't affect the solution — we could delete them and recover the same $(w, b)$.

### Adding noise: a case for soft margin

Now add a fifth point $x_5 = (0.1, -0.1)$ with $y_5 = +1$. This point is on the "wrong" side of our previous separator. Hard-margin SVM is now infeasible: no line can have all $+1$ on one side.

With soft margin, we need $\xi_5 > 0$ to accommodate this point. Suppose $C = 1$ and we keep $w = (1/2, 1/2), b = 0$. Then $y_5(w^\top x_5) = 1 \cdot (0.05 - 0.05) = 0$, so $\xi_5 = 1$ (the constraint $\geq 1 - \xi_5$ becomes $0 \geq 1 - \xi_5 \Rightarrow \xi_5 \geq 1$). Total penalty: $\tfrac{1}{2}(1/2) + 1 \cdot 1 = 1.25$.

Alternative: rotate the hyperplane slightly to improve $x_5$'s margin but worsen others, potentially lowering the total objective. The exact optimum requires solving the QP numerically, but the principle is clear: the solver searches the trade-off space between margin width and aggregate slack.

## 1.5 Relevance to Machine Learning Practice

### Where SVMs shine

- **Small-to-medium datasets** with clean feature spaces. Training is $O(n^2)$–$O(n^3)$ depending on the solver, so scales poorly beyond hundreds of thousands of examples.
- **High-dimensional sparse data** (text, genomics). Linear SVM with $\ell_2$ regularization is competitive with logistic regression; sometimes slightly better due to the margin principle.
- **Non-linear boundaries** via kernels (Section 3). For many structured-input problems (strings, graphs), kernels give a principled way to apply SVMs without explicit feature engineering.
- **Novelty / anomaly detection** via one-class SVM (Section 5).

### When not to use SVMs

- **Very large datasets** ($n \gtrsim 10^6$). Kernel SVM becomes prohibitive. Linear SVMs (via SGD) scale, but so do logistic regression and neural networks, which often transfer-learn more easily.
- **When you need calibrated probabilities out of the box**. SVMs output margins, not probabilities. You can post-hoc calibrate with Platt scaling or isotonic regression, but logistic regression gives probabilities natively.
- **Online / streaming settings**: re-training a batch SVM is expensive. Online variants exist (Pegasos, NORMA) but are less standard.

### Common alternatives

- **Logistic regression**: similar linear model, log-loss instead of hinge. Gives probabilities; hinge is slightly more robust because it truly ignores correctly-classified confident points.
- **Random forests / gradient boosting**: handle heterogeneous features, missing values, and non-linearities with less tuning.
- **Neural networks**: dominant for large-scale perception tasks; SVMs can still be competitive as classifier heads on top of learned representations.

### Trade-offs

- **Bias-variance**: small $C$ → higher bias, lower variance; large $C$ → opposite. Classical U-shape.
- **Interpretability**: linear SVM gives a weight vector $w$ with per-feature contributions. Kernel SVMs are much harder to interpret.
- **Robustness**: hinge loss is somewhat robust to outliers because the loss grows linearly, not quadratically. But extreme outliers can still dominate via the slack term.
- **Computational cost**: training $O(n^2 d)$–$O(n^3)$ for kernel SVM; linear SVM via specialized solvers (LIBLINEAR) is close to linear in $n$. Prediction cost is $O(|SV| \cdot d)$ for kernel, $O(d)$ for linear.

## 1.6 Common Pitfalls & Misconceptions

1. **"SVMs don't need regularization."** Wrong. The $\tfrac{1}{2}\|w\|^2$ term *is* $\ell_2$ regularization. The $C$ hyperparameter must be tuned.
2. **Forgetting to standardize features.** SVM distances are Euclidean; features with large scales dominate. Always standardize (mean 0, std 1) or normalize.
3. **Thinking more support vectors = better.** More support vectors usually means the problem is noisier or $C$ is small. Sparsity (few SVs) is generally good for generalization and prediction speed.
4. **Using hard margin on real data.** Almost always wrong. Use soft margin with a tuned $C$.
5. **Treating $C$ as if it has a natural scale.** $C$ depends on data size and feature scaling. Always tune via cross-validation; typical grid $C \in \{10^{-3}, 10^{-2}, \ldots, 10^3\}$.
6. **Confusing hinge loss with 0-1 loss.** Hinge upper-bounds 0-1, but minimizing hinge is not the same as minimizing training error. Hinge penalizes being "not confident enough" even when correct.
7. **Using SVM for probability estimation directly.** The raw score $w^\top x + b$ is a margin, not a probability. Calibrate if you need probabilities.
8. **Ignoring class imbalance.** Standard SVM weights all slack equally; the majority class dominates. Use class-weighted $C$ ($C_+$ and $C_-$) or resample.

---

# 2. Dual Formulation: Lagrange Multipliers and KKT Conditions

## 2.1 Motivation & Intuition

The SVM primal problem is a convex quadratic program in $(w, b)$ with $n$ inequality constraints. We could hand this to a generic QP solver and be done. But we won't — there's a better way.

The **dual formulation** rewrites the problem in terms of one variable per training example ($\alpha_i$) instead of per feature ($w_j$). This has two huge benefits:

1. When the feature space is very high-dimensional (or infinite-dimensional after kernelization), the primal has too many variables. The dual always has exactly $n$ variables, where $n$ is the number of training points.
2. The dual only involves the data through inner products $x_i^\top x_j$. This is what enables the **kernel trick** (Section 3): we can replace inner products with any valid kernel function, allowing us to work in feature spaces we never explicitly construct.

Lagrange multipliers and the Karush-Kuhn-Tucker (KKT) conditions are the mathematical machinery that turn a constrained optimization problem into its dual form. Understanding these is essential to understanding *why* SVMs have the properties they do — especially sparsity.

### Intuitive picture

Think of the constraints $y_i(w^\top x_i + b) \geq 1$ as walls confining the optimization. For each constraint, imagine a "pressure" $\alpha_i \geq 0$ that the constraint exerts on the solution. If a constraint is inactive (the point is strictly inside its correct half-space), the pressure is zero. If a constraint is active (the point lies exactly on the margin boundary), the pressure can be positive. These pressures are exactly the Lagrange multipliers, and — as we'll see — they are the dual variables.

**Complementary slackness** (a KKT condition) says: either the constraint is tight, or the multiplier is zero. This is why only support vectors ($\alpha_i > 0$) affect the solution.

## 2.2 Conceptual Foundations

### Key terms

- **Lagrangian**: A single scalar function that combines the objective and constraints, weighted by multipliers. Turns a constrained problem into an unconstrained one (at the cost of introducing new variables).
- **Primal problem**: The original optimization (min over $w, b, \xi$).
- **Dual problem**: A different optimization (max over $\alpha$), derived by minimizing the Lagrangian over primal variables first.
- **Weak duality**: Dual optimal value $\leq$ primal optimal value, always.
- **Strong duality**: Dual optimal value $=$ primal optimal value. Holds for SVM because the problem is convex and satisfies Slater's condition (strict feasibility).
- **KKT conditions**: Necessary and (under convexity + constraint qualification) sufficient optimality conditions for constrained problems. They characterize the optimal solution.

### How the pieces interact

1. Write the Lagrangian $\mathcal{L}(w, b, \xi, \alpha, \mu)$ with multipliers $\alpha_i$ for margin constraints and $\mu_i$ for slack non-negativity.
2. Minimize $\mathcal{L}$ over $(w, b, \xi)$ for fixed multipliers. Setting gradients to zero gives us expressions for $w$ in terms of the $\alpha_i$.
3. Substitute back to get the **dual objective**, a function of $\alpha$ alone.
4. The dual problem is: maximize this dual objective over $\alpha \geq 0$ (plus one linear equality constraint).

The solution to the dual, combined with the stationarity equations, gives us the primal solution.

### Assumptions

- **Convexity**: needed for strong duality. SVM is a convex QP, so this holds.
- **Slater's condition**: there exists a strictly feasible point. In soft-margin SVM this is trivially satisfied ($\xi_i$ can always be large enough). In hard-margin SVM this requires linear separability.
- **Smoothness**: Lagrangian must be differentiable where needed. Holds here.

### What breaks when assumptions fail

- Without convexity, the dual is still a lower bound but strong duality can fail (duality gap).
- Without Slater (hard-margin SVM with non-separable data), primal is infeasible and the whole formulation breaks.
- Numerical issues: extremely large $C$ can make the QP ill-conditioned.

## 2.3 Mathematical Formulation

### Deriving the dual (soft margin)

Start with the primal:

$$
\min_{w, b, \xi} \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i, \quad \text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0.
$$

Introduce Lagrange multipliers $\alpha_i \geq 0$ for the margin constraints and $\mu_i \geq 0$ for the slack non-negativity:

$$
\mathcal{L}(w, b, \xi, \alpha, \mu) = \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i - \sum_i \alpha_i\bigl[y_i(w^\top x_i + b) - 1 + \xi_i\bigr] - \sum_i \mu_i \xi_i.
$$

The dual function is $g(\alpha, \mu) = \min_{w, b, \xi} \mathcal{L}$. Take gradients:

**$\partial \mathcal{L}/\partial w = 0$**:

$$
w - \sum_i \alpha_i y_i x_i = 0 \quad\Longrightarrow\quad w = \sum_i \alpha_i y_i x_i. \tag{★}
$$

**$\partial \mathcal{L}/\partial b = 0$**:

$$
-\sum_i \alpha_i y_i = 0 \quad\Longrightarrow\quad \sum_i \alpha_i y_i = 0.
$$

**$\partial \mathcal{L}/\partial \xi_i = 0$**:

$$
C - \alpha_i - \mu_i = 0 \quad\Longrightarrow\quad \mu_i = C - \alpha_i.
$$

Combined with $\mu_i \geq 0$, this gives $0 \leq \alpha_i \leq C$ (called the **box constraint**).

Substitute (★) back into the Lagrangian. The $\|w\|^2$ term becomes

$$
\tfrac{1}{2}\|w\|^2 = \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j.
$$

The term $\sum_i \alpha_i y_i w^\top x_i$ becomes $\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^\top x_j$. The bias terms cancel (because $\sum_i \alpha_i y_i = 0$). The slack-related terms $C\sum_i \xi_i - \sum_i\alpha_i\xi_i - \sum_i\mu_i\xi_i$ collapse to 0 using $\mu_i = C - \alpha_i$. After simplification:

$$
\boxed{\max_{\alpha} \; \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_i y_j x_i^\top x_j \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \; \sum_i \alpha_i y_i = 0.}
$$

This is the **dual quadratic program**. It has $n$ variables, one box constraint per variable, and one equality constraint.

For the **hard-margin** case, omit the slack variables: $\mu_i$ disappears and we get $\alpha_i \geq 0$ without the upper bound. Everything else is the same.

### The KKT conditions

At optimality, the following must hold simultaneously:

1. **Stationarity**: $w = \sum_i \alpha_i y_i x_i$, $\sum_i \alpha_i y_i = 0$, $\alpha_i + \mu_i = C$.
2. **Primal feasibility**: $y_i(w^\top x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$.
3. **Dual feasibility**: $\alpha_i \geq 0$, $\mu_i \geq 0$.
4. **Complementary slackness**:
   - $\alpha_i [y_i(w^\top x_i + b) - 1 + \xi_i] = 0$
   - $\mu_i \xi_i = 0$, i.e. $(C - \alpha_i)\xi_i = 0$.

From complementary slackness we read off the **structure of support vectors**:

- $\alpha_i = 0$: point is not a support vector. It's outside the margin, correctly classified. The margin constraint is slack.
- $0 < \alpha_i < C$: **margin support vector**. Then $\mu_i > 0$, so $\xi_i = 0$, and the point lies exactly on the margin boundary: $y_i(w^\top x_i + b) = 1$.
- $\alpha_i = C$: **bound (non-margin) support vector**. Slack $\xi_i$ can be positive; the point is inside the margin or misclassified.

### Recovering $b$

Once we have $\alpha$, $w$ is given by (★). For $b$, pick any margin SV (any $i$ with $0 < \alpha_i < C$). Then $y_i(w^\top x_i + b) = 1$, so

$$
b = y_i - w^\top x_i = y_i - \sum_j \alpha_j y_j x_j^\top x_i.
$$

For numerical stability, average over all margin SVs.

### Prediction in dual form

The decision function is

$$
f(x) = w^\top x + b = \sum_i \alpha_i y_i x_i^\top x + b.
$$

Only the support vectors (nonzero $\alpha_i$) contribute. This is the **sparsity property**, and it's what makes prediction efficient.

### Mapping math to intuition

- $\alpha_i$ quantifies how "important" point $i$ is to the final classifier. Zero $\alpha$ = irrelevant.
- The constraint $\sum_i \alpha_i y_i = 0$ comes from the bias term; intuitively, the "positive-class pressure" must balance the "negative-class pressure."
- The box $\alpha_i \leq C$ caps how much influence any single noisy point can have. This is the formal source of SVM's robustness: in hard margin ($C = \infty$), a single badly-placed point can dominate.

## 2.4 Worked Example

Let's dualize a tiny 1D problem.

Three points: $x_1 = -1, y_1 = -1$; $x_2 = +1, y_2 = +1$; $x_3 = +2, y_3 = +1$. Hard margin.

The inner product matrix:

$$
K = \begin{pmatrix} 1 & -1 & -2 \\ -1 & 1 & 2 \\ -2 & 2 & 4 \end{pmatrix}, \quad y_i y_j K_{ij} = \begin{pmatrix} 1 & 1 & 2 \\ 1 & 1 & 2 \\ 2 & 2 & 4 \end{pmatrix}.
$$

Dual:

$$
\max_\alpha \; (\alpha_1 + \alpha_2 + \alpha_3) - \tfrac{1}{2}\bigl[\alpha_1^2 + \alpha_2^2 + 4\alpha_3^2 + 2\alpha_1\alpha_2 + 4\alpha_1\alpha_3 + 4\alpha_2\alpha_3\bigr]
$$

subject to $\alpha_i \geq 0$ and $-\alpha_1 + \alpha_2 + \alpha_3 = 0$.

**Intuition**: $x_1$ and $x_2$ are the closest opposing points; we expect $x_3$ (further from the boundary) to have $\alpha_3 = 0$. Let's check by assuming $\alpha_3 = 0$ and solving.

Then $\alpha_1 = \alpha_2 \equiv \alpha$. Objective becomes $2\alpha - \tfrac{1}{2}(2\alpha^2 + 2\alpha^2) = 2\alpha - 2\alpha^2$. Maximize: derivative $2 - 4\alpha = 0 \Rightarrow \alpha = 1/2$.

So $\alpha_1^\star = \alpha_2^\star = 1/2$, $\alpha_3^\star = 0$.

Recover $w = \sum_i \alpha_i y_i x_i = (1/2)(-1)(-1) + (1/2)(+1)(+1) + 0 = 1$.

Recover $b$ using $x_2$ (a margin SV): $y_2(w \cdot x_2 + b) = 1 \Rightarrow 1 \cdot (1 + b) = 1 \Rightarrow b = 0$.

Decision function: $f(x) = x$. The hyperplane is $x = 0$, exactly midway between $x_1$ and $x_2$. $x_3$ is indeed irrelevant.

Check margin: $|f(x_1)|/\|w\| = 1/1 = 1$, and $|f(x_2)|/\|w\| = 1$. Both equal, as expected at the optimum.

## 2.5 Relevance to Machine Learning Practice

### Why the dual matters

- **Kernelization**: the dual only needs inner products $x_i^\top x_j$, which can be replaced with kernel evaluations $k(x_i, x_j)$. Without the dual, kernels would be unusable.
- **High-dimensional features**: if $d \gg n$, the dual ($n$ variables) is much smaller than the primal ($d$ variables).
- **Interpretability of support vectors**: the dual makes sparsity explicit and gives a principled way to read off which points matter.
- **Specialized solvers**: SMO (Sequential Minimal Optimization) exploits the dual structure to solve large SVMs efficiently by updating two $\alpha_i$'s at a time.

### Primal vs. dual: when to use which

- **Linear SVM, $d \ll n$**: solve primal. LIBLINEAR uses coordinate descent or trust-region methods on the primal, scaling near-linearly in $n$.
- **Kernel SVM, any size**: must use dual (or kernel matrix approximations).
- **Small-to-medium kernel SVMs**: use SMO (LIBSVM). $O(n^2)$ to $O(n^3)$ in practice.
- **Very large kernel SVMs**: random features or Nyström approximations to approximate the kernel, then solve as linear SVM.

## 2.6 Common Pitfalls & Misconceptions

1. **Confusing $\alpha_i$ with probabilities.** Lagrange multipliers are not probabilities. They're non-negative but unbounded (up to $C$), and have no probabilistic interpretation.
2. **Thinking more $\alpha_i > 0$ means better fit.** More support vectors typically means more noise or smaller $C$, often worse generalization.
3. **Ignoring the equality constraint $\sum \alpha_i y_i = 0$.** Many tutorials gloss over this, but it's essential to recovering $b$ correctly.
4. **Computing $b$ from a bound SV.** Bound SVs (with $\alpha_i = C$) satisfy $y_i f(x_i) \leq 1$, not $= 1$, so using them to compute $b$ gives wrong answers. Always use margin SVs ($0 < \alpha_i < C$).
5. **Solving the dual when the primal is simpler.** For large $n$, small $d$, linear SVM, the primal is *much* better. Don't reflexively go to the dual.
6. **Assuming strong duality always holds.** It holds for SVM because of convexity and Slater, but in non-convex problems (e.g., some structured prediction with non-convex losses), duality gaps can be significant.

---

# 3. Kernel Methods: The Kernel Trick, Mercer's Theorem, and Common Kernels

## 3.1 Motivation & Intuition

Linear classifiers are elegant, but real-world decision boundaries are often not linear. Consider classifying points as inside vs. outside the unit circle. No line in 2D can separate these — but if we lift the points into 3D by adding a third feature $z = x_1^2 + x_2^2$, the classes become linearly separable (a horizontal plane at $z = 1$ splits them).

This "feature lifting" trick generalizes: pick a map $\phi : \mathbb{R}^d \to \mathbb{R}^D$ where $D$ may be huge, train a linear model in the new space, and you effectively have a non-linear model in the original space.

The problem: if $D$ is huge (e.g., all degree-$p$ monomials in $d$ variables is $\binom{d+p}{p}$ dimensions — blows up fast) or infinite, explicitly computing $\phi(x)$ is expensive or impossible.

The **kernel trick** is the following observation: many useful learning algorithms (SVM, kernel ridge regression, kernel PCA) only depend on $\phi(x)$ through inner products $\phi(x_i)^\top \phi(x_j)$. So we don't need $\phi$ at all — just a function $k(x_i, x_j)$ that *equals* this inner product, which we can often compute cheaply without ever materializing $\phi$.

For the polynomial case: the kernel $k(x, x') = (x^\top x' + 1)^p$ implicitly corresponds to all monomials up to degree $p$, and computes in $O(d)$ regardless of how big $D$ is.

### Real-world ML connections

- **Strings and graphs**: Kernels exist for structured objects where a feature-vector representation isn't obvious — string kernels (substring match counts), graph kernels (shortest-path / Weisfeiler-Lehman), tree kernels (for parsing).
- **Gaussian processes**: The kernel becomes a covariance function; the whole machinery of Bayesian regression over function spaces rests on kernels.
- **Kernel ridge regression, kernel PCA, spectral clustering**: All use the kernel trick.
- **Deep learning connections**: Neural tangent kernel (NTK) shows that infinite-width networks trained by gradient descent behave like kernel methods, with a specific "NTK" kernel. This provides theoretical insight into deep learning.

### The intuition in one sentence

A kernel is a similarity measure: $k(x, x')$ is large when $x$ and $x'$ are "similar" (in the sense of some implicit feature space). The SVM uses this similarity to classify new points by comparing them against training support vectors.

## 3.2 Conceptual Foundations

### Key terms

- **Feature map** $\phi : \mathcal{X} \to \mathcal{H}$: a function from input space to a (possibly infinite-dimensional) Hilbert space.
- **Kernel** $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$: a function that corresponds to an inner product in some feature space, $k(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}$.
- **Gram matrix** (or **kernel matrix**) $K$: the $n \times n$ matrix with entries $K_{ij} = k(x_i, x_j)$.
- **Positive semi-definite (PSD) kernel**: a kernel whose Gram matrix is PSD for *every* finite set of points. Equivalently, $\sum_{i,j} c_i c_j k(x_i, x_j) \geq 0$ for all real $c_i$.
- **Reproducing Kernel Hilbert Space (RKHS)**: the Hilbert space $\mathcal{H}_k$ uniquely associated with a PSD kernel $k$, in which functions can be evaluated via inner products with $k(\cdot, x)$.
- **Mercer's theorem**: gives conditions under which a function $k$ is a valid kernel (i.e., arises from some inner product).

### Common kernels

- **Linear**: $k(x, x') = x^\top x'$. Corresponds to identity feature map; gives back the standard linear SVM.
- **Polynomial**: $k(x, x') = (x^\top x' + c)^p$. Corresponds to monomials up to degree $p$. $c \geq 0$ is a free parameter; $p$ is the degree.
- **Radial basis function (RBF) / Gaussian**: $k(x, x') = \exp(-\gamma \|x - x'\|^2)$. Infinite-dimensional feature space. $\gamma > 0$ is bandwidth (high $\gamma$ = narrow = more local).
- **Sigmoid / tanh**: $k(x, x') = \tanh(\kappa x^\top x' + \theta)$. Historically popular but *not* PSD for all parameter values.
- **Laplacian**: $k(x, x') = \exp(-\gamma\|x - x'\|_1)$.
- **String, graph, tree kernels**: domain-specific.

### How the pieces interact

1. Choose a kernel $k$ (this encodes your prior assumption about similarity).
2. Compute the Gram matrix $K$ on training data.
3. Solve the dual SVM QP using $K$ in place of inner products.
4. Predict on a new point $x$ using $f(x) = \sum_{i \in SV} \alpha_i y_i k(x_i, x) + b$.

### Assumptions

- The kernel must be **symmetric and PSD** for the underlying optimization to be convex and for a feature space interpretation to exist.
- Kernels implicitly assume the similarity notion they encode is meaningful for the problem. An RBF kernel assumes local similarity; a polynomial kernel assumes multiplicative feature interactions matter.

### What breaks when assumptions fail

- **Non-PSD kernel**: the dual QP may be non-convex and ill-posed; solvers may return garbage or fail.
- **Wrong kernel choice**: a kernel that doesn't match problem structure gives poor generalization. E.g., RBF with huge $\gamma$ memorizes the training set.
- **Poorly scaled features with RBF**: the kernel becomes effectively constant (all pairs equally dissimilar) or effectively an identity (all pairs equally similar).

## 3.3 Mathematical Formulation

### Mercer's theorem (classical form)

Let $\mathcal{X}$ be a compact subset of $\mathbb{R}^d$, and let $k : \mathcal{X}\times\mathcal{X}\to\mathbb{R}$ be continuous and symmetric. Define the integral operator $T_k : L^2(\mathcal{X}) \to L^2(\mathcal{X})$ by

$$
(T_k f)(x) = \int_\mathcal{X} k(x, x') f(x') \, dx'.
$$

If $T_k$ is positive semi-definite — i.e., $\int\int k(x, x') f(x) f(x') dx dx' \geq 0$ for all $f \in L^2(\mathcal{X})$ — then there exist an orthonormal sequence of eigenfunctions $\{\psi_i\}$ of $T_k$ and corresponding non-negative eigenvalues $\{\lambda_i\}$ such that

$$
k(x, x') = \sum_{i=1}^\infty \lambda_i \psi_i(x) \psi_i(x'),
$$

uniformly convergent on $\mathcal{X}\times\mathcal{X}$.

**Implication**: define $\phi(x) = (\sqrt{\lambda_1}\psi_1(x), \sqrt{\lambda_2}\psi_2(x), \ldots)$. Then $k(x, x') = \langle \phi(x), \phi(x')\rangle_{\ell^2}$. So every PSD kernel *is* an inner product in some feature space.

### Modern view: PSD $\Leftrightarrow$ valid kernel

A function $k$ is a valid kernel if and only if for every finite set of points $\{x_1, \ldots, x_n\}$, the Gram matrix $K$ is symmetric PSD. This is equivalent to Mercer's condition on compact domains but is also meaningful on discrete domains where string/graph kernels live.

### Rules for constructing kernels

If $k_1, k_2$ are kernels, $c \geq 0$, $f$ is any real function, $\phi$ any map:

- $c \cdot k_1$ is a kernel.
- $k_1 + k_2$ is a kernel.
- $k_1 \cdot k_2$ (pointwise product) is a kernel.
- $f(x) k_1(x, x') f(x')$ is a kernel.
- $k_1(\phi(x), \phi(x'))$ is a kernel.
- Limits of kernels (under uniform convergence) are kernels.

These closure properties let us build complex kernels from simple ones.

### Deriving the polynomial kernel's feature map (degree 2, $d = 2$, $c = 1$)

$k(x, x') = (x^\top x' + 1)^2 = (x_1 x'_1 + x_2 x'_2 + 1)^2$.

Expanding:

$$
= x_1^2 x_1'^2 + x_2^2 x_2'^2 + 2 x_1 x_2 x_1' x_2' + 2 x_1 x_1' + 2 x_2 x_2' + 1.
$$

This equals $\phi(x)^\top \phi(x')$ with

$$
\phi(x) = (x_1^2,\; x_2^2,\; \sqrt{2} x_1 x_2,\; \sqrt{2} x_1,\; \sqrt{2} x_2,\; 1)^\top.
$$

A 6-dimensional feature space, computed by a single $O(d)$ inner product.

### The RBF kernel and its infinite feature space

$k(x, x') = e^{-\gamma\|x - x'\|^2}$. Using $\|x - x'\|^2 = \|x\|^2 + \|x'\|^2 - 2x^\top x'$:

$$
k(x, x') = e^{-\gamma\|x\|^2} e^{-\gamma\|x'\|^2} e^{2\gamma x^\top x'}.
$$

Expanding the Taylor series of $e^{2\gamma x^\top x'} = \sum_{k=0}^\infty \frac{(2\gamma)^k (x^\top x')^k}{k!}$, we see the RBF kernel contains polynomial kernels of all degrees, weighted by factorials. Its feature space is infinite-dimensional.

### Kernelized SVM

Replace every occurrence of $x_i^\top x_j$ in the dual with $k(x_i, x_j)$:

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j k(x_i, x_j), \quad 0 \leq \alpha_i \leq C, \; \sum_i \alpha_i y_i = 0.
$$

Prediction:

$$
f(x) = \sum_{i\in SV} \alpha_i y_i k(x_i, x) + b.
$$

$w$ is not explicitly computed — it lives in the (possibly infinite-dimensional) feature space. We only ever manipulate kernel evaluations.

### Representer theorem

For a broad class of regularized loss minimization problems over RKHS,

$$
\min_{f \in \mathcal{H}_k} \frac{1}{n}\sum_i L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}_k}^2,
$$

the optimal $f^\star$ lies in the $n$-dimensional subspace spanned by $\{k(\cdot, x_i)\}_{i=1}^n$:

$$
f^\star(x) = \sum_{i=1}^n \beta_i k(x_i, x).
$$

This is why kernel methods reduce to finite-dimensional problems despite living in infinite-dimensional spaces. The SVM is a special case with $L$ = hinge loss.

## 3.4 Worked Example

### RBF kernel on 1D data

Training points: $x_1 = 0, y_1 = +1$; $x_2 = 2, y_2 = -1$; $x_3 = 4, y_3 = +1$.

These are not linearly separable in 1D. Let's use RBF with $\gamma = 1$.

Kernel matrix:

$$
K = \begin{pmatrix} 1 & e^{-4} & e^{-16} \\ e^{-4} & 1 & e^{-4} \\ e^{-16} & e^{-4} & 1 \end{pmatrix} \approx \begin{pmatrix} 1 & 0.0183 & 10^{-7} \\ 0.0183 & 1 & 0.0183 \\ 10^{-7} & 0.0183 & 1 \end{pmatrix}.
$$

The key point: $K_{12}$ and $K_{23}$ (opposite-class pairs) are small; $K_{13}$ (same-class but far apart) is also small. With such sharp kernel (high $\gamma$), each point is nearly "isolated," and the model essentially memorizes them.

If we set $\gamma = 0.1$, the kernel matrix is smoother:

$$
K \approx \begin{pmatrix} 1 & 0.67 & 0.20 \\ 0.67 & 1 & 0.67 \\ 0.20 & 0.67 & 1 \end{pmatrix},
$$

and now the geometry is more interesting. At a new point $x = 1$:

$$
k(x, x_1) = e^{-0.1} \approx 0.905, \quad k(x, x_2) = e^{-0.1} \approx 0.905, \quad k(x, x_3) = e^{-0.9} \approx 0.407.
$$

So $x = 1$ is similar to both $x_1$ (positive) and $x_2$ (negative), suggesting it's near the boundary. The classification depends on the $\alpha$ values (and signs), which the solver determines.

### Qualitative effect of $\gamma$

- Small $\gamma$: kernel is wide, similarity decays slowly, model is smoother, higher bias, lower variance.
- Large $\gamma$: kernel is narrow, each training point only "votes" for its immediate neighborhood, model becomes spiky, lower bias, higher variance. In the limit, RBF SVM approaches nearest-neighbor behavior.

## 3.5 Relevance to Machine Learning Practice

### Kernel selection heuristics

- **Default starting point**: RBF. Flexible, works well with standardized features. Tune $\gamma$ and $C$ jointly via grid search.
- **Text classification**: linear kernel. High-dimensional sparse TF-IDF features are already rich; non-linearity often overfits.
- **Small, structured data**: polynomial of degree 2 or 3, or domain-specific kernels.
- **Structured data (strings, graphs, trees)**: specialized kernels; often state-of-the-art in bioinformatics before deep learning.

### Hyperparameter tuning

For RBF: a common heuristic for $\gamma$ is $\gamma = 1/(d \cdot \text{Var}(X))$ (the `scale` setting in scikit-learn) or the *median heuristic* $\gamma = 1/\text{median}\|x_i - x_j\|^2$. Combined with log-scale grid search on $C$, these usually land in a reasonable neighborhood.

### Computational considerations

- Gram matrix is $n \times n$: memory $O(n^2)$, computation $O(n^2 d)$. Infeasible for $n \gg 10^5$.
- Training: $O(n^2)$ to $O(n^3)$ depending on solver and problem.
- Prediction: $O(|SV|)$ kernel evaluations per test point. If $|SV|$ is small, fast; if many, slow.
- **Approximations**: Nyström method approximates $K$ using $m \ll n$ landmark points; random Fourier features approximate RBF explicitly by a finite-dimensional map. Both turn kernel SVM into (approximate) linear SVM, trading accuracy for speed.

### When not to use kernel SVMs

- Very large datasets (millions of points) — training is prohibitive; use linear SVM on learned features (from a neural network) instead.
- When interpretability is critical — kernelized models are black boxes.
- Online / streaming settings — retraining is expensive.
- When the relevant non-linearity is very problem-specific and poorly captured by generic kernels.

## 3.6 Common Pitfalls & Misconceptions

1. **"The kernel trick makes everything tractable."** No. It makes the *inner product* efficient, but training still costs $O(n^2)$ memory for the Gram matrix. This is the main bottleneck at scale.
2. **Using non-PSD "kernels."** The sigmoid kernel is a classic example: it can give a non-PSD Gram matrix, making the dual non-convex. Sometimes still works, but the theoretical guarantees fail.
3. **Not scaling features before using RBF.** Features on very different scales distort distances. Standardize first.
4. **Interpreting $w$ after kernelization.** There is no explicit $w$ in a useful sense; it lives in the RKHS. You can compute feature importance for polynomial kernels by expanding, but not generally.
5. **Using RBF with $\gamma$ too large.** The model memorizes the training set (every point is its own cluster), test performance collapses. Always validate.
6. **Thinking kernels are magic.** They encode assumptions. A poorly-matched kernel gives poor results. For structured data, designing the right kernel is as much work as feature engineering.
7. **Forgetting the bias term.** Many implementations include an offset; others don't. In principle, one can include a constant "1" feature and absorb $b$ into $w$, but double-check your library.

---

# 4. Support Vectors: Sparsity and Geometric Interpretation

## 4.1 Motivation & Intuition

When you train an SVM, most of the training data turns out to be *irrelevant* — only a subset of points, the **support vectors**, define the decision boundary. This is unlike logistic regression, where every training point contributes to the final weights.

Why does this happen? Because the hinge loss is zero for points that are confidently and correctly classified. Once a point is comfortably on the right side of the margin, the optimizer stops caring about it. It's only the points near the boundary — the hard cases — that matter.

This property, called **sparsity**, has two important consequences:

1. **Compact models**: prediction time depends only on the number of support vectors, not the total training size. If your model has 500 SVs out of 100,000 training points, prediction is 200× faster than a "dense" model.
2. **Interpretability**: you can inspect the support vectors to understand which training examples are "informative" or "borderline" — useful for debugging, active learning, and data quality assessment.

### The geometric picture

Imagine the decision hyperplane as a fence, with parallel "margin fences" on either side, a distance $1/\|w\|$ away. The SVM's optimization tries to:

- Place the fence so that the margin fences are as far apart as possible (maximize margin).
- Keep all $+1$ points on or beyond the positive margin fence, and all $-1$ points on or beyond the negative one (hard margin).
- In soft margin, allow some points to slip inside or across the fence, but at a cost.

The support vectors are the points "touching" or "violating" the fences. Everyone else is firmly in their territory and doesn't participate.

## 4.2 Conceptual Foundations

### Three types of points in a soft-margin SVM

Recall from Section 2, complementary slackness gives:

1. **Non-SVs** ($\alpha_i = 0$): $y_i(w^\top x_i + b) > 1$. Point is strictly outside the margin, on the correct side. Removing this point doesn't change the solution.
2. **Free / margin support vectors** ($0 < \alpha_i < C$): $y_i(w^\top x_i + b) = 1$, $\xi_i = 0$. Point lies exactly on the margin boundary. These define the hyperplane's position.
3. **Bound support vectors** ($\alpha_i = C$): $y_i(w^\top x_i + b) \leq 1$ (possibly $\leq 0$, i.e., misclassified), $\xi_i \geq 0$. Point is inside the margin or on the wrong side. These are the "hard cases" and carry slack.

### Sparsity property

The decision function $f(x) = \sum_i \alpha_i y_i k(x_i, x) + b$ only sums over $i$ with $\alpha_i > 0$ — the support vectors. If only $s$ of $n$ training points are SVs, the model is $n/s$ times faster to evaluate.

### Geometric margin interpretation

At optimum:
- Margin SVs sit exactly on the margin boundaries (functional margin = 1).
- The geometric margin is $1/\|w^\star\|$.
- The width of the "no-man's-land" between positive and negative margin fences is $2/\|w^\star\|$.

### Assumptions

- Sparsity depends on the hinge loss's "flat zero" for correctly-classified confident points. Other losses (log-loss, squared loss) don't produce this.
- Sparsity is *not* guaranteed to be strong. With noisy data and small $C$, many points become bound SVs.

### What breaks it

- Very noisy data, tiny $C$: nearly every point is inside the margin, so nearly every point is a bound SV. Sparsity lost.
- Large class imbalance without reweighting: the minority class can end up with many SVs, the majority with few.
- Kernel too narrow (RBF with high $\gamma$): the model memorizes individual points, each becoming its own SV. In the extreme, all points become SVs.

## 4.3 Mathematical Formulation

### Fraction of SVs as a lower bound on error

There is a classical result: for any trained SVM, the leave-one-out cross-validation error is upper-bounded by the fraction of support vectors, $|SV|/n$. Intuitively, only SVs affect the classifier; removing a non-SV leaves the decision function unchanged, so non-SVs are correctly classified under leave-one-out. Therefore:

$$
\text{LOO error} \leq \frac{|SV|}{n}.
$$

This gives a cheap generalization estimate — though usually loose, since not all SVs flip under leave-one-out.

### Geometric derivation of margin = $1/\|w\|$

Take any two points on opposite margin fences: $w^\top x_+ + b = 1$ and $w^\top x_- + b = -1$. Subtract: $w^\top(x_+ - x_-) = 2$. The vector from one fence to the other, projected onto the unit normal $w/\|w\|$, has length

$$
\frac{w^\top(x_+ - x_-)}{\|w\|} = \frac{2}{\|w\|}.
$$

So the *total* margin width is $2/\|w\|$, and the half-margin (from hyperplane to each fence) is $1/\|w\|$. ✓

### Structure of $w$ in kernel form

$w = \sum_{i\in SV} \alpha_i y_i \phi(x_i)$. The direction of $w$ is a weighted combination of support vector feature maps, with weights $\alpha_i y_i$. Positive-class SVs push $w$ one way, negative-class SVs the other. Balance is enforced by $\sum_i \alpha_i y_i = 0$.

## 4.4 Worked Example

### Illustrating sparsity

Return to the 1D example from Section 2: $x_1=-1, x_2=+1, x_3=+2$ with labels $-, +, +$. We found $\alpha_1 = \alpha_2 = 1/2$, $\alpha_3 = 0$.

- $x_1$: margin SV (on boundary).
- $x_2$: margin SV (on boundary).
- $x_3$: non-SV (confidently correct, $\alpha = 0$).

Remove $x_3$: solution unchanged. Remove $x_1$ or $x_2$: solution changes dramatically.

### Noisy example

Suppose we add $x_4 = 0.5$ with $y_4 = -1$ (a negative point sitting on the positive side). With soft-margin and $C = 1$:

- $x_4$ is misclassified by the old solution, so it can't have $\alpha_4 = 0$.
- Likely $\alpha_4 = C = 1$ (a bound SV), with slack $\xi_4 > 0$.
- The hyperplane shifts a bit to accommodate $x_4$, possibly making $x_2$ a non-SV or widening the bound support to include $x_3$.

Key takeaway: noisy points tend to become bound SVs, absorbing the "pressure" of their misclassification through slack.

## 4.5 Relevance to Machine Learning Practice

### Uses of SV inspection

- **Active learning**: SVs are the most informative points. In an active learning loop, querying near the margin boundary gives the biggest reduction in uncertainty.
- **Data quality diagnosis**: points that are bound SVs with very high slack ($\xi_i \gg 1$) are candidates for mislabeled data — worth manual inspection.
- **Model compression**: for prediction latency, keeping only high-$\alpha$ SVs can give a smaller approximate model.
- **Budget SVMs**: variants that limit the number of SVs explicitly, trading accuracy for prediction speed.

### Trade-offs

- Smaller $C$: more SVs (wider margin, more tolerance), slower prediction, but usually better generalization.
- Larger $C$: fewer SVs (tighter margin), faster prediction, higher overfitting risk.
- High-variance kernel (e.g., RBF with large $\gamma$): more SVs, nearly nearest-neighbor behavior.

### Practical sanity check

Always report $|SV|/n$ after training. If it's $>$ 50%, something's probably off:
- $C$ too small, or
- Kernel poorly tuned, or
- Data very noisy / overlapping, or
- Classes heavily imbalanced.

## 4.6 Common Pitfalls & Misconceptions

1. **Equating "support vector" with "outlier."** Not the same. Margin SVs are just boundary points — often *typical* examples of their class. Bound SVs with large slack may be outliers, but not all SVs are outliers.
2. **Thinking non-SVs are useless.** They're useless *for the final decision function*, but they contributed to establishing the boundary during training (implicitly, by being satisfied constraints).
3. **Assuming sparsity holds regardless of settings.** Sparsity is data- and hyperparameter-dependent. It's an empirical property of hinge-loss minimization, not a guaranteed one.
4. **Sparsity $\neq$ feature sparsity.** SVM sparsity is in *training examples*; $\ell_1$-SVM (using $\|w\|_1$ instead of $\|w\|_2^2$) gives *feature* sparsity — a totally different thing.
5. **Interpreting $\alpha_i$ as "difficulty."** Partially true: $\alpha_i = C$ signals a hard-to-classify point. But margin SVs also have positive $\alpha$, and they're not particularly hard.

---

# 5. Extensions: ν-SVM, One-Class SVM, Structured SVM

## 5.1 Motivation & Intuition

The standard soft-margin SVM has one hyperparameter $C$, whose scale is opaque — $C = 100$ might be tiny or huge depending on the dataset. People wanted a more interpretable version. Enter **ν-SVM**, where the hyperparameter $\nu \in (0, 1]$ directly bounds the fraction of margin violations and lower-bounds the fraction of SVs.

A different generalization: what if we only have data from one class (e.g., "normal" transactions) and want to detect anomalies? The **one-class SVM** learns a tight boundary around the data in feature space; points outside are anomalies.

A third generalization: what if the output is not a single label but a structured object — a parse tree, a sequence of labels, a graph? **Structured SVM** generalizes the margin principle to arbitrary output spaces with an associated loss function.

These extensions share the DNA of the original SVM — margin maximization, hinge loss, kernel compatibility — but target different problem types.

## 5.2 ν-SVM

### Formulation

Instead of $C$, introduce $\nu \in (0, 1]$ and a margin variable $\rho$ to be optimized:

$$
\min_{w, b, \xi, \rho} \; \tfrac{1}{2}\|w\|^2 - \nu\rho + \tfrac{1}{n}\sum_i \xi_i
$$

$$
\text{s.t.} \quad y_i(w^\top x_i + b) \geq \rho - \xi_i, \; \xi_i \geq 0, \; \rho \geq 0.
$$

Note the functional margin target is now $\rho$ (optimized), not fixed at 1. We penalize the *negative* of $\rho$ (because we want $\rho$ large).

### The ν property

At optimum, $\nu$ has two simultaneous interpretations:

- **Upper bound on the fraction of margin errors** (points with $\xi_i > 0$).
- **Lower bound on the fraction of support vectors**.

So if you set $\nu = 0.1$, you're saying "at most 10% of points can be margin violations, and I expect at least 10% SVs." This is much more interpretable than $C$.

Relationship to $C$-SVM: there's a bijection between $(\nu)$ and (some value of $C$ depending on data), so they span the same solution space — ν just parametrizes it differently.

### Dual

$$
\max_\alpha \; -\tfrac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_i y_j k(x_i, x_j), \quad 0 \leq \alpha_i \leq 1/n, \; \sum_i \alpha_i y_i = 0, \; \sum_i \alpha_i \geq \nu.
$$

### When to use

- Exploratory work where you don't know what $C$ should be.
- Situations where controlling the error rate is the goal.
- One-class problems (next subsection) use the same ν idea.

## 5.3 One-Class SVM

### Problem setup

Given only positive examples (or unlabeled "normal" data), learn a function that is high on training data and low elsewhere — a model of the support of the data distribution. New points with low scores are anomalies.

### Schölkopf's formulation

Separate the data from the origin in feature space with maximum margin:

$$
\min_{w, \xi, \rho} \; \tfrac{1}{2}\|w\|^2 + \tfrac{1}{\nu n}\sum_i \xi_i - \rho
$$

$$
\text{s.t.} \quad w^\top \phi(x_i) \geq \rho - \xi_i, \; \xi_i \geq 0.
$$

Decision function: $f(x) = \mathrm{sign}(w^\top \phi(x) - \rho)$. Predict +1 (normal) if positive, -1 (anomaly) otherwise.

The $\nu$ parameter has the same dual interpretation: upper-bounds outliers, lower-bounds SVs.

### Geometric picture with RBF kernel

In the feature space of an RBF kernel, all points lie on the unit hypersphere (since $k(x, x) = 1$). The one-class SVM hyperplane separating these from the origin corresponds, in input space, to a **contour of the kernel density estimate**. Effectively, one-class SVM with RBF kernel is a principled way to do density-level-set estimation.

### Alternative: SVDD (Support Vector Data Description)

A cousin formulation: find the smallest hypersphere enclosing the data, allowing some outliers. For RBF kernel, SVDD and one-class SVM are equivalent.

### When to use

- Anomaly / novelty detection with only "normal" data.
- Fraud detection, intrusion detection, equipment health monitoring.
- Modern alternatives: Isolation Forest, autoencoder reconstruction error, deep SVDD. One-class SVM is still relevant for small datasets and when kernels encode good similarity.

## 5.4 Structured SVM

### The problem

Predict not a class label but a structured object $y \in \mathcal{Y}$, where $\mathcal{Y}$ is huge or combinatorial:

- **Sequence labeling**: $y$ is a sequence of tags (named entity recognition, POS tagging).
- **Parsing**: $y$ is a parse tree.
- **Object detection**: $y$ is a set of bounding boxes.

Standard multiclass doesn't scale ($|\mathcal{Y}|$ exponential in sequence length). We need structure-aware learning.

### Joint feature map

Define $\Psi(x, y)$ jointly over input and output. The prediction rule is

$$
\hat{y}(x) = \arg\max_{y \in \mathcal{Y}} w^\top \Psi(x, y).
$$

This generalizes the binary SVM's $\mathrm{sign}(w^\top x + b)$.

### Margin rescaling formulation (Tsochantaridis et al., 2005)

Given a **loss function** $\Delta(y, y')$ quantifying how bad predicting $y'$ is when true label is $y$ (e.g., Hamming loss for sequences):

$$
\min_{w, \xi} \tfrac{1}{2}\|w\|^2 + \frac{C}{n}\sum_i \xi_i
$$

$$
\text{s.t.} \quad \forall i, \forall y' \neq y_i: \; w^\top [\Psi(x_i, y_i) - \Psi(x_i, y')] \geq \Delta(y_i, y') - \xi_i.
$$

Intuition: for every wrong output $y'$, the score of the true $y_i$ should exceed the score of $y'$ by at least $\Delta(y_i, y')$ — the **margin is proportional to how wrong $y'$ is**.

### Computational challenge

The constraint set has $n \cdot |\mathcal{Y}|$ elements — exponential. Solved via:

- **Cutting plane methods**: iteratively find the most violated constraint for each $i$ (using a "loss-augmented inference" oracle $\arg\max_y \Delta(y_i, y) + w^\top \Psi(x_i, y)$) and add it.
- **Subgradient methods** on the equivalent hinge-like loss.

The loss-augmented inference oracle is problem-specific (Viterbi for sequences, CKY for parsing) — as long as you have an efficient inference algorithm, structured SVM is tractable.

### Applications

- NER, POS tagging (classical approaches before neural seq2seq).
- Constituency and dependency parsing.
- Image segmentation (pixel-wise structured prediction).
- Ranking / learning to rank (RankSVM is a special case).

### When to use

Before neural networks, structured SVM was state-of-the-art for many NLP tasks. Today, deep sequence models (LSTMs, transformers) with CRF layers or direct likelihood training have largely replaced it. Structured SVM remains relevant:

- With very limited data.
- When you have strong domain structure and want margin guarantees.
- In hybrid pipelines where a deep model produces features and a structured SVM does final prediction.

## 5.5 Relevance to Machine Learning Practice

- **ν-SVM**: use when you want interpretable control over error rate.
- **One-class SVM**: still a solid baseline for anomaly detection; understand its assumptions (RBF = density-level-set). Modern deep alternatives often do better on high-dimensional data (images, audio).
- **Structured SVM**: mostly historical in mainstream ML, but the concepts (loss-augmented inference, joint feature maps) still appear in structured prediction research.

## 5.6 Common Pitfalls & Misconceptions

1. **Thinking ν-SVM is a different model.** It's the same hypothesis class, just re-parameterized.
2. **Applying one-class SVM to high-dimensional raw data.** It works best in moderate-dimensional, well-normalized feature spaces. For images/audio, first learn a representation (e.g., autoencoder) then apply one-class SVM on that.
3. **Confusing one-class SVM with binary SVM on imbalanced data.** Very different. Binary needs both classes; one-class only needs one. For imbalanced binary problems, use class-weighted binary SVM, not one-class.
4. **Ignoring the loss function in structured SVM.** $\Delta$ drives the margin scaling and the loss-augmented inference. Mis-specifying $\Delta$ can severely hurt structured predictions.
5. **Expecting structured SVM to scale like binary SVM.** The inference oracle is the bottleneck; if it's $O(|\mathcal{Y}|)$, you're stuck.

---

# 6. Multiclass Strategies: One-vs-One, One-vs-All, DAGs

## 6.1 Motivation & Intuition

Binary SVMs output a single scalar, $\mathrm{sign}(w^\top x + b)$. For $K > 2$ classes, we need more structure. The straightforward approach is to train many binary classifiers and combine them.

There are three classical strategies:

1. **One-vs-All (OvA / OvR)**: train $K$ binary SVMs, each class vs. the rest. Predict the class whose SVM gives the highest score.
2. **One-vs-One (OvO)**: train $\binom{K}{2}$ binary SVMs, one per pair of classes. Predict by voting.
3. **Directed Acyclic Graph (DAGSVM)**: train $\binom{K}{2}$ binary SVMs as in OvO, but at prediction time use a DAG structure to evaluate only $K - 1$ of them.

There are also "true multiclass" SVMs (Crammer-Singer, Weston-Watkins) that solve a single joint optimization — less common in practice due to computational cost and comparable or worse accuracy.

## 6.2 One-vs-All (OvA)

### Method

For each class $c \in \{1, \ldots, K\}$, train a binary SVM $f_c(x) = w_c^\top x + b_c$ with $+1$ labels for class $c$ and $-1$ labels for all other classes. Predict

$$
\hat{y}(x) = \arg\max_c f_c(x).
$$

### Pros

- Only $K$ classifiers.
- Each classifier uses all training data.
- Simple to implement, parallelizable.

### Cons

- Each binary problem is imbalanced (one class vs. $K-1$ others). With many classes, the positive class is a small minority.
- Scores $f_c$ are not comparable across classifiers (different SVMs, different scales). Argmaxing is a heuristic.
- Tie-breaking is ambiguous when multiple $f_c(x)$ are equal.

### When to use

Default for many implementations, works reasonably well when $K$ is small to moderate. Often competitive with OvO in practice.

## 6.3 One-vs-One (OvO)

### Method

For each pair $(c, c')$ with $c < c'$, train a binary SVM using only data from classes $c$ and $c'$. Total: $\binom{K}{2} = K(K-1)/2$ classifiers.

At prediction, each classifier votes for one of its two classes. Predict the class with the most votes. (Ties broken by summed margin scores or some other rule.)

### Pros

- Each binary problem uses less data (only two classes' worth), so training each is fast.
- Binary problems are balanced by construction.

### Cons

- $O(K^2)$ classifiers — memory and prediction cost scales quadratically in $K$.
- Large $K$ (100+ classes) makes this impractical.
- Each classifier only sees a fraction of data, so individual classifiers may be weaker.

### When to use

Preferred when $K$ is modest (≤ 20 or so) and training data per class is reasonable. LIBSVM uses OvO by default for multiclass SVM.

## 6.4 DAGSVM

### Method

Train $\binom{K}{2}$ pairwise SVMs as in OvO. At prediction, arrange classifiers in a DAG: start at a root node, which runs one pairwise classifier (class $a$ vs. class $b$). The loser is eliminated from the candidate set. Move to the next node, which picks a new pair from the surviving candidates. Continue until one class remains.

Each prediction uses exactly $K - 1$ classifier evaluations (versus $\binom{K}{2}$ for OvO).

### Pros

- Same training as OvO but faster prediction.
- Empirically comparable accuracy to OvO.

### Cons

- Prediction accuracy can depend on the DAG's ordering (which pair to evaluate first). Random orderings typically work fine.
- More complex to implement and reason about than OvA or OvO.

### When to use

Large $K$ where OvO's prediction cost is prohibitive, but OvA accuracy is insufficient. Historically relevant; less common today.

## 6.5 Crammer-Singer Multiclass SVM

### Idea

Define $K$ weight vectors $w_1, \ldots, w_K$ and the decision rule $\hat{y}(x) = \arg\max_c w_c^\top x$. The training objective demands that for each training point $(x_i, y_i)$:

$$
w_{y_i}^\top x_i \geq w_c^\top x_i + 1 - \xi_i \quad \forall c \neq y_i.
$$

The loss is the multiclass hinge:

$$
L(w, x_i, y_i) = \max_{c \neq y_i} \max(0, 1 + w_c^\top x_i - w_{y_i}^\top x_i).
$$

Single joint QP, not a collection of binary problems.

### Pros

- Principled multiclass formulation.
- No ambiguity about how to combine binary outputs.

### Cons

- Single large QP, harder to parallelize.
- In practice, comparable accuracy to OvA/OvO — the extra theoretical elegance doesn't always translate to empirical wins.

### Weston-Watkins variant

Similar idea, different loss structure (sums over $c \neq y_i$ instead of max). Both are sometimes referred to as "all-in-one" multiclass SVMs.

## 6.6 Relevance to Machine Learning Practice

### Current landscape

- **Small $K$ (≤ 10)**: any method works; OvO is LIBSVM default and is fine.
- **Medium $K$ (10–100)**: OvA is usually best due to scaling.
- **Large $K$ (hundreds)**: SVM becomes inefficient regardless of strategy. Use softmax classifier on top of learned features (neural networks), or hierarchical approaches, or approximate methods like negative sampling.
- **Extreme classification** ($K$ in thousands-millions): specialized algorithms (FastXML, PecosXL) — SVMs are not competitive.

### Probabilistic outputs

None of these strategies give probabilities directly. For calibrated multiclass probabilities:

- Platt scaling per binary classifier + pairwise coupling (for OvO) or softmax normalization (for OvA).
- Alternative: use multiclass logistic regression from the start.

## 6.7 Common Pitfalls & Misconceptions

1. **Assuming $\arg\max f_c$ is well-calibrated across OvA classifiers.** Scores aren't probabilities and aren't on comparable scales. The argmax is a heuristic that usually works but can fail.
2. **Ignoring class imbalance in OvA.** The positive class is a minority in each binary problem. Use per-class $C$ weights or resample.
3. **Scaling OvO to huge $K$.** Memory and prediction time explode.
4. **Evaluating DAGSVM without understanding ordering effects.** Different orderings can give different accuracy; average over random orderings if you care.
5. **Using multiclass SVM when softmax + neural network would be vastly better.** SVM multiclass strategies are legacy; for large modern problems, use NNs.

---

# Interview Preparation

The following questions span foundational, mathematical, applied, debugging, and probing categories. Answers aim for the rigor expected in an ML/applied-scientist interview.

## Foundational Questions

### Q1. Intuitively, what does a Support Vector Machine do, and why is the margin principle useful?

An SVM finds a separating hyperplane between two classes that maximizes the margin — the distance to the closest training point from either class. The intuition is twofold:

1. **Robustness**: a wider margin means the classifier is less sensitive to small perturbations of the data. If we jitter a point, it's still classified correctly.
2. **Generalization**: statistical learning theory (VC bounds, Rademacher complexity) tells us that classifiers with large geometric margins have generalization error bounds that don't depend on the ambient dimension — they depend instead on $R^2/\gamma^2$, where $R$ is the radius of the data and $\gamma$ the margin. This gives a principled reason why margin maximization helps, especially in high dimensions.

### Q2. What's the difference between the functional margin and the geometric margin?

Functional margin of $(x_i, y_i)$ is $y_i(w^\top x_i + b)$ — positive if correct, zero on the hyperplane, scales linearly with $\|w\|$. Geometric margin is $y_i(w^\top x_i + b)/\|w\|$ — invariant to rescaling $(w, b)$ and equal to the actual Euclidean distance to the hyperplane. The geometric margin is what we actually care about; the functional margin is a useful intermediate quantity.

### Q3. Why does the SVM use hinge loss instead of log-loss?

Hinge loss: $\max(0, 1 - yf(x))$. Log-loss: $\log(1 + e^{-yf(x)})$. Both are convex upper bounds on 0-1 loss, but hinge is flat (exactly zero) for correctly-classified confident points, while log-loss is always positive. This is why SVMs have sparse solutions (only points near the margin contribute), while logistic regression uses all points. Hinge is also slightly more robust to outliers on the correct side (they incur zero loss). Log-loss gives probabilistic outputs; hinge does not.

### Q4. What are support vectors and why are they called that?

Support vectors are training points that "support" the hyperplane — they determine its position. Formally, they have non-zero dual coefficients $\alpha_i$, corresponding to active margin constraints. Remove a non-support vector: solution unchanged. Remove a support vector: solution generally changes. In the soft-margin case, there are two kinds:
- **Margin SVs** ($0 < \alpha_i < C$): on the margin boundary.
- **Bound SVs** ($\alpha_i = C$): inside margin or misclassified.

### Q5. What does the hyperparameter $C$ control?

$C$ balances margin width against classification violations. Small $C$: heavy regularization, wide margin, more tolerance for errors, higher bias. Large $C$: tight margin, tries to classify every training point correctly, higher variance. $C \to \infty$ recovers hard-margin SVM. $C$ is usually tuned via cross-validation on a logarithmic grid.

## Mathematical Questions

### Q6. Derive the dual of the soft-margin SVM.

Primal:

$$
\min_{w, b, \xi} \tfrac{1}{2}\|w\|^2 + C\sum_i \xi_i, \; \text{s.t. } y_i(w^\top x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0.
$$

Lagrangian with $\alpha_i \geq 0, \mu_i \geq 0$:

$$
\mathcal{L} = \tfrac{1}{2}\|w\|^2 + C\sum \xi_i - \sum \alpha_i[y_i(w^\top x_i + b) - 1 + \xi_i] - \sum \mu_i \xi_i.
$$

Set $\nabla_w \mathcal{L} = 0$: $w = \sum \alpha_i y_i x_i$. Set $\partial_b \mathcal{L} = 0$: $\sum \alpha_i y_i = 0$. Set $\partial_{\xi_i} \mathcal{L} = 0$: $\alpha_i + \mu_i = C$, so $0 \leq \alpha_i \leq C$.

Substitute back. The $\|w\|^2$ becomes $\sum_{i,j}\alpha_i\alpha_j y_iy_j x_i^\top x_j$; combined with the $-\sum\alpha_i y_i w^\top x_i$ term and simplification:

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_iy_j x_i^\top x_j, \; \text{s.t. } 0 \leq \alpha_i \leq C, \; \sum\alpha_i y_i = 0.
$$

### Q7. State the KKT conditions and explain what they tell us about support vectors.

KKT for soft-margin SVM:
- Stationarity: $w = \sum\alpha_i y_i x_i$, $\sum\alpha_i y_i = 0$, $\alpha_i + \mu_i = C$.
- Primal feasibility: $y_i(w^\top x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$.
- Dual feasibility: $\alpha_i \geq 0$, $\mu_i \geq 0$.
- Complementary slackness: $\alpha_i[y_i(w^\top x_i + b) - 1 + \xi_i] = 0$ and $\mu_i \xi_i = 0$.

From these we read off:
- $\alpha_i = 0 \Rightarrow$ point strictly outside margin, non-SV.
- $0 < \alpha_i < C \Rightarrow \mu_i > 0 \Rightarrow \xi_i = 0$, point on margin boundary, margin SV.
- $\alpha_i = C \Rightarrow \xi_i \geq 0$, point inside margin or misclassified, bound SV.

### Q8. Prove that the geometric margin of an SVM is $1/\|w\|$ at optimum.

Under the normalization $\min_i y_i(w^\top x_i + b) = 1$, for margin SVs we have $y_i(w^\top x_i + b) = 1$, so

$$
\text{distance to hyperplane} = \frac{|w^\top x_i + b|}{\|w\|} = \frac{1}{\|w\|}.
$$

Since margin SVs achieve the minimum distance, this is the geometric margin.

### Q9. Explain Mercer's theorem and why it matters for SVMs.

Mercer's theorem: A symmetric continuous function $k : \mathcal{X}\times\mathcal{X} \to \mathbb{R}$ is a kernel (i.e., $k(x, x') = \langle\phi(x), \phi(x')\rangle$ for some feature map $\phi$) if and only if it is positive semi-definite — equivalently, the integral operator $T_k$ is PSD on $L^2(\mathcal{X})$, equivalently, every finite Gram matrix is PSD.

This matters because the kernelized SVM dual contains $k(x_i, x_j)$ terms. If $k$ is PSD, there's an underlying feature space, the dual QP is convex, and strong duality holds. If $k$ isn't PSD, the dual is non-convex and the whole framework breaks.

### Q10. Why is the SVM dual useful even when the primal could be solved directly?

Three reasons:
1. **Kernelization**: the dual depends on data only through inner products, which can be replaced by any valid kernel — enabling non-linear models without explicit feature maps.
2. **Dimensionality**: the dual has $n$ variables regardless of the feature dimension $d$. When $d \gg n$ (or $d = \infty$ via kernels), the dual is the only option.
3. **Sparsity structure**: the dual makes SV sparsity explicit and enables efficient solvers like SMO.

For linear SVMs with $n \gg d$, the primal is actually preferred — dual advantages vanish.

### Q11. Show that the RBF kernel corresponds to an infinite-dimensional feature space.

$k(x, x') = e^{-\gamma\|x-x'\|^2} = e^{-\gamma\|x\|^2}e^{-\gamma\|x'\|^2}e^{2\gamma x^\top x'}$. Taylor-expand the last factor:

$$
e^{2\gamma x^\top x'} = \sum_{k=0}^\infty \frac{(2\gamma)^k (x^\top x')^k}{k!}.
$$

Each $(x^\top x')^k$ is a polynomial kernel of degree $k$, with feature space $\mathbb{R}^{\binom{d+k-1}{k}}$. Summing over all $k$ gives an infinite direct sum, so the RBF feature space is infinite-dimensional.

### Q12. Derive how to recover the bias $b$ from the dual solution.

For any margin SV (one with $0 < \alpha_i < C$), complementary slackness gives $\xi_i = 0$ and $y_i(w^\top x_i + b) = 1$. Solving for $b$:

$$
b = y_i - w^\top x_i = y_i - \sum_j \alpha_j y_j x_j^\top x_i \quad (\text{or } \sum_j \alpha_j y_j k(x_j, x_i) \text{ kernelized}).
$$

In practice, average over all margin SVs for numerical stability:

$$
b = \frac{1}{|M|}\sum_{i \in M}\bigl(y_i - \sum_j \alpha_j y_j k(x_j, x_i)\bigr),
$$

where $M = \{i : 0 < \alpha_i < C\}$.

## Applied Questions

### Q13. Given a text classification problem with 10M features (word n-grams) and 100K documents, how would you train an SVM?

Linear kernel, primal solver. Specifically:
- Use LIBLINEAR (or scikit-learn's `LinearSVC` / `SGDClassifier` with hinge loss).
- Features are sparse (most are zero per document); exploit sparsity.
- Normalize TF-IDF vectors to unit length.
- Cross-validate $C$ on a log grid.
- Expect training in minutes, prediction in microseconds.

Don't use kernel SVM — the $n \times n$ Gram matrix (100K × 100K = 10B entries) is infeasible, and in high-dimensional sparse text data, non-linear kernels typically don't help anyway.

### Q14. How would you choose between RBF and linear kernel?

Heuristics:
- **Linear kernel** if $d \geq n$ (high-dim sparse features like text, genomics). Non-linearity rarely helps; overfitting risk is high.
- **RBF** if $d$ is modest (say $\leq$ a few hundred), $n$ is in the thousands, and you suspect non-linear structure.
- Always try linear first — it's faster to train and tune and establishes a baseline.
- If linear accuracy is unsatisfactory and data size allows, try RBF with grid search on $(C, \gamma)$.

### Q15. Your SVM has 95% of training points as support vectors. What does this suggest?

Likely one or more of:
- **$C$ too small**: margin is too wide, too many points inside it. Try increasing $C$.
- **RBF $\gamma$ too large**: model is memorizing each training point.
- **Data extremely noisy or overlapping**: the classes simply aren't separable even with kernels.
- **Heavy class imbalance**.

Actions: tune $C$ and kernel parameters via cross-validation; inspect mislabeled or borderline examples; consider feature engineering or a different model family.

### Q16. How do you handle class imbalance in SVM?

Several options:
1. **Class-weighted $C$**: use $C_+ = C \cdot n/n_+$ and $C_- = C \cdot n/n_-$ so that total slack penalty is balanced across classes. Sklearn's `class_weight='balanced'` does this.
2. **Resample**: under-sample majority or over-sample minority (e.g., SMOTE).
3. **Threshold tuning**: after training, move the decision threshold away from zero to optimize recall or precision for the minority class.
4. **Different loss** (e.g., cost-sensitive hinge).

Class-weighted $C$ is usually the first thing to try.

### Q17. How do you get calibrated probabilities out of an SVM?

SVMs output margins, not probabilities. To calibrate:
1. **Platt scaling**: fit a logistic regression $P(y=1|x) = \sigma(Af(x) + B)$ on a held-out set. Simple, works reasonably for smooth distributions.
2. **Isotonic regression**: fits a non-parametric monotone function from scores to probabilities. More flexible, needs more calibration data.

Always calibrate on held-out data, not the training set — otherwise you're just fitting noise. Scikit-learn's `CalibratedClassifierCV` wraps this.

### Q18. For a problem with 50 classes, which multiclass strategy would you pick?

OvA is typical here. OvO would need $\binom{50}{2} = 1225$ classifiers — training cost and prediction cost both heavy. DAGSVM reduces prediction cost but still has 1225 classifiers in memory. OvA trains 50 classifiers, each on the full dataset, and predicts in $O(50)$ time.

At this scale, also consider whether SVM is even the right choice — logistic regression with softmax (multinomial LR) gives comparable accuracy with calibrated probabilities and scales better. If features are learned (e.g., from a neural net), a linear softmax head is usually best.

## Debugging & Failure Modes

### Q19. You trained an SVM, achieved 99% training accuracy but only 60% test accuracy. What happened and what do you do?

Classic overfitting. For SVM specifically:
- $C$ too large: the model is fitting every training point, including noise.
- RBF $\gamma$ too large: model is hyper-local, memorizing.
- Too little training data relative to feature dimension.

Debug steps:
1. Cross-validate $C$ and $\gamma$ on a log grid.
2. Check the number of SVs — if nearly all, reduce $C$ or $\gamma$.
3. Plot learning curves (train/val accuracy vs. data size) to distinguish overfitting from data shortage.
4. Try a simpler kernel (linear) as a baseline.
5. Add more training data or regularize more aggressively.

### Q20. Your SVM is very slow at inference. Why, and how do you fix it?

Prediction cost is $O(|SV| \cdot d)$ for linear and $O(|SV|)$ kernel evaluations for kernelized. Root causes:
- Many support vectors (usually means $C$ too small or $\gamma$ too large).
- Large feature dimension.

Fixes:
1. Reduce $|SV|$: increase $C$, retune kernel. Trade off with accuracy.
2. Use linear SVM if possible — prediction is $O(d)$, independent of SVs, because $w$ can be materialized.
3. Approximate the kernel with Nyström or random Fourier features, then use linear SVM.
4. Budget SVMs: limit $|SV|$ explicitly during training.
5. If accuracy matters more than latency, consider a different model (gradient boosting, neural network).

### Q21. Training your SVM on a dataset seems to fail (solver doesn't converge). What are possible causes?

- **Features on wildly different scales**: the QP is ill-conditioned. Standardize.
- **Duplicate or near-duplicate points with conflicting labels**: makes the problem hard. Deduplicate / investigate label noise.
- **$C$ extremely large**: problem approaches hard-margin, may be infeasible or numerically unstable. Try smaller $C$.
- **Non-PSD "kernel"**: (e.g., sigmoid kernel with wrong params). Check Gram matrix eigenvalues.
- **Too large dataset for the solver**: switch to a scalable solver (LIBLINEAR for linear, SGD-based, or Nyström).

### Q22. You change $\gamma$ in your RBF SVM from 0.1 to 10 and accuracy drops dramatically. Why?

Higher $\gamma$ means a narrower RBF — each training point only influences its immediate neighborhood. With $\gamma = 10$, the kernel decays very fast; the model essentially becomes a very local classifier, close to 1-NN on kernel-weighted distances. Training accuracy might stay high (memorization) but test accuracy collapses due to high variance / overfitting.

Fix: tune $\gamma$ via cross-validation. A good heuristic starting point is $\gamma = 1/(d \cdot \text{Var}(X))$ or the median heuristic.

### Q23. After deployment, your SVM's accuracy degrades over time. Why might this happen?

**Distribution shift**: the data-generating process has changed. SVMs, like most supervised models, assume i.i.d. from the training distribution. Common shifts:
- **Covariate shift**: $p(x)$ changes while $p(y|x)$ is stable. Importance weighting can help.
- **Concept drift**: $p(y|x)$ changes. Need retraining or online adaptation.
- **Label shift**: $p(y)$ changes. Reweight predictions.

Monitor for drift via: tracking input feature distributions (e.g., PSI, KL divergence), tracking confidence distributions, tracking accuracy on held-out labeled monitoring sets. Periodic retraining is the typical remedy.

## Probing / Depth Questions

### Q24. Why does the SVM's margin appear in the generalization bound, and what does that imply?

Generalization bounds based on **margin theory** (e.g., Bartlett, Shawe-Taylor) show that the test error of a margin-maximizing classifier is bounded by something like

$$
\hat{R}_\gamma(h) + O\Bigl(\sqrt{\tfrac{R^2/\gamma^2}{n}}\Bigr),
$$

where $\hat{R}_\gamma$ is the empirical margin loss, $R$ is the data radius, $\gamma$ the margin. Crucially, this bound doesn't directly depend on the ambient dimension $d$ — only on the ratio $R/\gamma$.

Implication: even in very high-dimensional feature spaces (e.g., RBF's infinite-dim space), if we maintain a large margin on training data, generalization can remain controlled. This is a theoretical justification for kernel SVMs despite their huge implicit feature spaces.

### Q25. Explain the representer theorem and why SVMs benefit from it.

Representer theorem: for a loss minimization problem of the form

$$
\min_{f \in \mathcal{H}_k} \frac{1}{n}\sum_i L(y_i, f(x_i)) + \lambda \Omega(\|f\|_{\mathcal{H}_k}),
$$

with $\Omega$ monotonically increasing, the optimal $f^\star$ lies in the $n$-dimensional span of $\{k(\cdot, x_i)\}$, i.e., $f^\star(x) = \sum_i \beta_i k(x_i, x)$.

Why SVMs benefit: SVM objective is regularized hinge loss in RKHS. Without the representer theorem, we'd need to optimize over an infinite-dimensional space. With it, we optimize over $n$ coefficients — a finite, tractable problem. The $\beta_i$ here correspond (up to signs) to $\alpha_i y_i$ in the SVM dual.

### Q26. Compare SVM and logistic regression. When is one preferred?

Both are linear classifiers with convex loss + $\ell_2$ regularization. Differences:

| Property | SVM (hinge) | LR (log loss) |
|----------|-------------|---------------|
| Loss on correct confident pts | Zero | Positive |
| Solution sparsity (in SVs) | Yes | No |
| Probabilistic output | No (need calibration) | Yes |
| Robustness to outliers | Slightly better (linear growth) | Comparable |
| Scalability | Similar (both linear in n via SGD) | Similar |
| Multi-class | Needs strategy (OvA, etc.) | Native (softmax) |

Prefer SVM when you want margin-based robustness, don't need probabilities, or want sparse solutions. Prefer LR when probabilities are needed, when you want a single native multiclass formulation, or when integrating into probabilistic pipelines.

### Q27. Can SVMs be combined with deep learning? How?

Yes, several ways:
1. **Feature learning + SVM head**: train a deep network, extract features from a late layer, train an SVM on those features. Pre-ResNet/BERT era, this was common — e.g., SVMs on CNN features.
2. **Large-margin softmax**: replace the softmax's log-loss with a hinge-like loss in neural networks. "DeepSVM" / "L-Softmax" / face recognition losses (ArcFace, CosFace) draw from this lineage.
3. **Neural tangent kernel**: infinite-width neural networks trained by gradient descent are equivalent to kernel ridge regression with the NTK. So kernel methods and deep learning are deeply connected theoretically.

In production today, end-to-end neural networks usually beat SVM-on-features, but SVMs remain relevant as simple, effective linear heads on top of frozen foundation-model embeddings.

### Q28. The sigmoid "kernel" $\tanh(\kappa x^\top x' + \theta)$ is sometimes used. Is it a valid kernel?

Not always. The sigmoid kernel is PSD only for specific parameter values; for many choices, the Gram matrix has negative eigenvalues. Historically it was used because it made SVMs look like neural networks (a two-layer perceptron with hidden units corresponding to SVs). In practice it can work empirically but lacks theoretical guarantees — the dual becomes non-convex, strong duality may fail, and numerical issues arise. Better alternatives (RBF, polynomial) exist.

### Q29. Given infinite data and compute, would you always prefer kernel SVM over linear SVM?

No. Kernel SVM has higher capacity (more flexible), but:
- Training is $O(n^2)$ or $O(n^3)$ even with the best solvers — infinite data doesn't help if you can't fit the kernel matrix in memory.
- For many real problems (especially high-dimensional sparse data), linear SVM is already near the Bayes optimum.
- Kernel SVM's benefits appear mainly on moderate-dimensional, medium-sized problems where the right kernel captures non-linear structure.

With infinite data and compute, a flexible neural model often outperforms kernel SVM because it can *learn* the relevant feature representation, whereas the kernel is fixed upfront.

### Q30. If you had to explain to a non-technical stakeholder why SVM generalizes well, what would you say?

"An SVM doesn't just find a decision line — it finds the decision line that is *most conservative*, maximizing the buffer zone between the two classes. Imagine sorting fruit: you could draw the line right at the edge of the apple cluster, but then a slightly-different-looking apple might get called an orange. Instead, the SVM draws the line in the middle of the empty zone between apples and oranges, so small variations in fruit appearance don't flip the decision. This conservativeness is why SVMs tend to do well on new, unseen data — they don't chase individual training examples."

---

## Closing Note

SVMs are a mature, elegant, and theoretically rich family of methods. While deep learning has taken over many domains where SVMs once reigned, the conceptual machinery — margin maximization, duality, the kernel trick, sparsity via complementary slackness — appears throughout modern ML. Understanding SVMs deeply pays dividends when reading papers on kernel methods, Gaussian processes, structured prediction, and the theoretical analysis of neural networks.

Study tip: implement a soft-margin SVM from scratch using SMO at least once. The interplay between primal, dual, and KKT conditions becomes concrete when you have to code it.
