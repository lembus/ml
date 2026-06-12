# Dimensionality Reduction: A Comprehensive Learning Guide

## Table of Contents

**Preface: Why Dimensionality Reduction?**

**Part I — Linear Methods**
1. Principal Component Analysis (PCA)
2. Linear Discriminant Analysis (LDA)
3. Factor Analysis (FA)
4. Independent Component Analysis (ICA)

**Part II — Non-linear Methods**
5. t-SNE
6. UMAP
7. Autoencoders
8. Manifold Learning (ISOMAP, LLE, Laplacian Eigenmaps)

**Part III — Feature Selection**
9. Filter Methods
10. Wrapper Methods
11. Embedded Methods

Each topic follows a six-section structure (Motivation, Conceptual Foundations, Mathematical Formulation, Worked Example, ML Practice, Pitfalls) and concludes with tiered interview questions.

---

## Preface: Why Dimensionality Reduction?

Modern ML datasets routinely have thousands or millions of features: pixels in an image, tokens in a document, gene expression levels, sensor readings. Working in such high-dimensional spaces causes three concrete problems:

**1. The curse of dimensionality.** In high dimensions, distances lose discriminative power. If you sample points uniformly in a $d$-dimensional unit hypercube, the ratio of the distance to the nearest neighbor and the farthest neighbor approaches 1 as $d \to \infty$. This breaks methods that rely on distance (k-NN, clustering, kernel methods). Relatedly, the volume of a unit ball vanishes relative to its bounding cube, so "nearby" becomes vacuous.

**2. Statistical inefficiency.** Estimating a joint density or a decision boundary requires exponentially more samples as dimensions grow. With 10 features and 10 bins each, you have $10^{10}$ cells; you will not fill them.

**3. Computational and storage cost.** Many algorithms scale superlinearly in $d$. Covariance matrices alone are $O(d^2)$ in memory.

**Two orthogonal strategies exist:**

- **Feature extraction** (a.k.a. projection / representation learning): construct new features as functions of the original ones. PCA, autoencoders, t-SNE, etc. live here.
- **Feature selection**: keep a subset of the original features. Lasso, mutual-information filtering, recursive elimination, etc. live here.

Feature extraction usually gives better compression; feature selection preserves interpretability. Both are used in practice, often together.

A second conceptual axis is **linear vs. non-linear**. Linear methods assume the important structure lies on an affine subspace — a plane through the origin (possibly after centering). Non-linear methods assume it lies on a curved manifold. Real data (images of faces under pose/lighting, gene expression under disease state) usually lives on a low-dimensional manifold embedded in a high-dimensional ambient space, but a linear approximation is often good enough and is much cheaper, more stable, and more interpretable.

A third axis is **supervised vs. unsupervised**. PCA is unsupervised (ignores labels); LDA is supervised (uses labels to maximize class separability). Choose based on whether you have labels and whether downstream use is classification.

---

# Part I — Linear Methods

---

## 1. Principal Component Analysis (PCA)

### 1.1 Motivation & Intuition

Imagine you measure the height and weight of 1000 people. These are strongly correlated: tall people tend to be heavier. If you plot the data, you see an elongated cloud along a diagonal line. Almost all the information is captured by a single number: position along that diagonal. The perpendicular direction (how "stocky" vs. "lean" someone is given their height) contains much less variation.

PCA formalizes this. It finds the directions in which data varies the most, ranks them, and lets you keep only the top few. The intuition is that **directions of high variance carry most of the information**, and directions of low variance are either noise or redundancy.

Concrete ML motivations:
- **Compression**: A $1024 \times 1024$ grayscale image is a point in $\mathbb{R}^{1{,}048{,}576}$, but natural images occupy a much lower-dimensional subspace — you can often reconstruct them from the top 50–200 principal components with little visible loss.
- **Noise reduction**: If noise is isotropic (equal variance in all directions) and signal is structured (high variance in a few directions), keeping top components preserves signal while discarding noise.
- **Visualization**: Projecting to 2 or 3 dimensions lets you eyeball structure.
- **Decorrelating features**: Downstream models (e.g., linear regression with collinearity, Gaussian Naive Bayes) benefit from uncorrelated inputs.
- **Preconditioning**: Whitening via PCA can speed up gradient descent.

### 1.2 Conceptual Foundations

**Setup.** You have $n$ data points $x_1, \dots, x_n \in \mathbb{R}^d$, stacked into a matrix $X \in \mathbb{R}^{n \times d}$ (one row per observation). Center the data: replace $x_i$ with $x_i - \bar{x}$, where $\bar{x} = \frac{1}{n}\sum_i x_i$. This is non-optional — PCA is about variance, and variance is measured around the mean.

**Goal.** Find an orthonormal set of directions $w_1, w_2, \dots, w_k$ (each a unit vector in $\mathbb{R}^d$) such that:
1. $w_1$ maximizes the variance of the projected data $\{w_1^\top x_i\}$.
2. $w_2$ maximizes variance subject to being orthogonal to $w_1$.
3. And so on.

These $w_j$ are the **principal components** (PCs), also called the **loadings** or **principal axes**. The projected coordinates $w_j^\top x_i$ are called **scores**.

**Equivalent formulations.** PCA admits three equivalent characterizations:
- **Max-variance**: the criterion above.
- **Min-reconstruction-error**: find a $k$-dimensional subspace such that projecting data onto it and back minimizes the average squared distance to the original.
- **Eigen-decomposition of the covariance**: PCs are eigenvectors of the sample covariance matrix $\Sigma = \frac{1}{n-1} X^\top X$ (for centered $X$), ranked by eigenvalue.

These all give the same answer. The first is a max problem, the second is a min problem, the third is an algebraic characterization.

**Assumptions and what breaks.**
1. **Linearity.** PCA finds linear subspaces. If data lies on a curved manifold (e.g., the Swiss roll), PCA finds a poor approximation.
2. **Variance = information.** A direction with low variance might still carry crucial discriminative information. This is why PCA is *not* a feature selection method for classification: a feature orthogonal to the class boundary might have tiny variance and be discarded, yet be the most predictive feature. (LDA fixes this.)
3. **Large variance = high signal.** Only true if features are on comparable scales. Measuring one feature in meters and another in millimeters inflates the latter by $10^6$ in variance, dominating the first PC. **Always standardize** unless features are genuinely on the same scale and you want that scale to matter.
4. **Gaussian-like data.** PCA is optimal for Gaussian data (it recovers the directions of the Gaussian's covariance ellipsoid). For heavy-tailed or multimodal data, outliers can pull PCs around dramatically.
5. **Mean-centered.** If you forget to center, the first "PC" points roughly toward the data centroid — informative about location, not shape.

### 1.3 Mathematical Formulation

Let $X \in \mathbb{R}^{n \times d}$ be centered ($\sum_i x_i = 0$). The sample covariance is
$$
\Sigma = \frac{1}{n-1} X^\top X \in \mathbb{R}^{d \times d}.
$$
This is symmetric positive semi-definite, so it has $d$ real non-negative eigenvalues $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \geq 0$ with corresponding orthonormal eigenvectors $v_1, \dots, v_d$.

**Derivation (max-variance view).** The variance of the projection $w^\top x$ (for a unit vector $w$) is
$$
\mathrm{Var}(w^\top x) = w^\top \Sigma w.
$$
We want to maximize this subject to $\|w\|=1$. Use a Lagrange multiplier:
$$
\mathcal{L}(w, \lambda) = w^\top \Sigma w - \lambda(w^\top w - 1).
$$
Setting $\nabla_w \mathcal{L} = 0$:
$$
2\Sigma w - 2\lambda w = 0 \implies \Sigma w = \lambda w.
$$
So $w$ must be an eigenvector of $\Sigma$, and the variance is $w^\top \Sigma w = \lambda$. The maximum is achieved by the eigenvector with the largest eigenvalue: $w_1 = v_1$, variance $\lambda_1$. Subsequent PCs come from imposing orthogonality to previous ones; the same Lagrangian machinery gives $w_j = v_j$.

**Derivation (min-reconstruction view).** Let $W_k = [w_1, \dots, w_k] \in \mathbb{R}^{d \times k}$ have orthonormal columns. The projection of $x$ onto the subspace spanned by $W_k$ is $W_k W_k^\top x$. The reconstruction error is
$$
J(W_k) = \frac{1}{n}\sum_i \|x_i - W_k W_k^\top x_i\|^2.
$$
Using $\|x\|^2 = \|W_k W_k^\top x\|^2 + \|x - W_k W_k^\top x\|^2$ (Pythagoras, since the two components are orthogonal),
$$
J(W_k) = \frac{1}{n}\sum_i \|x_i\|^2 - \frac{1}{n}\sum_i \|W_k^\top x_i\|^2 = \mathrm{const} - \mathrm{tr}(W_k^\top \Sigma W_k).
$$
Minimizing $J$ is equivalent to maximizing $\mathrm{tr}(W_k^\top \Sigma W_k)$, which is again maximized by choosing the top $k$ eigenvectors.

**SVD approach.** Any real matrix $X$ has a singular value decomposition
$$
X = U S V^\top,
$$
where $U \in \mathbb{R}^{n \times n}$ and $V \in \mathbb{R}^{d \times d}$ are orthogonal, and $S$ is rectangular-diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$ (the singular values). Then
$$
X^\top X = V S^\top S V^\top,
$$
so columns of $V$ are eigenvectors of $X^\top X$ (and thus of $\Sigma$, up to the $n-1$ scaling), and eigenvalues are $\lambda_j = \sigma_j^2 / (n-1)$.

The scores (projected data) are
$$
Z = X V_k = U_k S_k,
$$
where $V_k$, $U_k$, $S_k$ are the leading $k$ columns / block.

**Why SVD is preferred in practice:**
- Numerically stable (no need to form $X^\top X$, which can be ill-conditioned).
- Works when $d > n$ (e.g., 1000 images of $10^6$ pixels each): then $X^\top X$ is $10^6 \times 10^6$ but you only need the rank $\leq n$ non-trivial part. SVD handles this; or use the "economy" SVD, or the dual trick (eigendecompose $X X^\top \in \mathbb{R}^{n \times n}$ instead).

**Explained variance.** The total variance of centered data is $\mathrm{tr}(\Sigma) = \sum_j \lambda_j$. The fraction explained by the first $k$ PCs is
$$
\mathrm{EV}(k) = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}.
$$
A plot of $\mathrm{EV}(k)$ vs. $k$ (or of individual $\lambda_j$) is called a **scree plot**. The classic heuristic: choose $k$ at the "elbow," or to capture e.g. 95% of variance.

**Whitening.** After projection, dividing each component by $\sqrt{\lambda_j}$ gives a representation with unit covariance. This is called PCA whitening:
$$
z_{\text{white}} = S_k^{-1} U_k^\top x \cdot \sqrt{n-1}.
$$

### 1.4 Worked Example

Take four 2D points: $(2, 1), (3, 2), (4, 3), (5, 4)$.

**Step 1: Center.** Mean is $(3.5, 2.5)$. Centered:
$(-1.5, -1.5), (-0.5, -0.5), (0.5, 0.5), (1.5, 1.5)$.

All points lie exactly on the line $y = x$ (after centering). We expect PC1 to be $\frac{1}{\sqrt{2}}(1, 1)$ and all variance in that direction.

**Step 2: Covariance.**
$$
\Sigma = \frac{1}{3}\begin{pmatrix} \sum x_i^2 & \sum x_i y_i \\ \sum x_i y_i & \sum y_i^2 \end{pmatrix} = \frac{1}{3}\begin{pmatrix} 5 & 5 \\ 5 & 5 \end{pmatrix} = \begin{pmatrix} 5/3 & 5/3 \\ 5/3 & 5/3 \end{pmatrix}.
$$

**Step 3: Eigendecomposition.** Solve $\det(\Sigma - \lambda I) = 0$:
$(5/3 - \lambda)^2 - (5/3)^2 = 0 \implies \lambda(\lambda - 10/3) = 0$.
So $\lambda_1 = 10/3$, $\lambda_2 = 0$.

Eigenvector for $\lambda_1$: $(\Sigma - \frac{10}{3}I)v = 0 \implies -\frac{5}{3}v_1 + \frac{5}{3}v_2 = 0$, giving $v_1 = v_2$. Normalized: $v^{(1)} = \frac{1}{\sqrt{2}}(1,1)^\top$.

Eigenvector for $\lambda_2 = 0$: orthogonal, so $v^{(2)} = \frac{1}{\sqrt{2}}(1,-1)^\top$.

**Step 4: Explained variance.** $\lambda_1 / (\lambda_1 + \lambda_2) = 100\%$. PC1 captures everything, matching our geometric intuition.

**Step 5: Scores.** Project the centered data onto $v^{(1)}$:
$z_i = v^{(1)\top}(x_i - \bar{x})$.
$z_1 = \frac{1}{\sqrt{2}}(-1.5 - 1.5) = -3/\sqrt{2} \approx -2.12$.
Similarly: $-1/\sqrt{2}, 1/\sqrt{2}, 3/\sqrt{2}$.

Reconstruction from PC1 alone: $\hat{x}_i = \bar{x} + z_i \cdot v^{(1)}$. Zero reconstruction error since $\lambda_2 = 0$.

**A more realistic scenario (sketch).** For MNIST ($28 \times 28 = 784$ features), the top ~50 PCs explain ~80% of variance; top ~150 explain ~95%. Reconstructions from the top 50 PCs are recognizable but slightly blurred.

### 1.5 Relevance to ML Practice

**Where it's used:**
- **Preprocessing** before classifiers/regressors that suffer from multicollinearity or curse of dimensionality.
- **Visualization** (top 2–3 PCs).
- **Compression / denoising** (keep top $k$, discard rest, reconstruct).
- **Whitening** before ICA or before image-processing pipelines.
- **Feature engineering** in classical tabular ML.
- **Eigenfaces / eigendigits** — historical but pedagogically important.
- **Speeding up k-NN or kernel methods** by reducing $d$.
- **Initialization** for non-linear methods (e.g., initializing t-SNE with PCA coordinates stabilizes it).

**When not to use it:**
- When class labels matter and discriminative directions might have low variance — use LDA or supervised methods.
- When structure is genuinely non-linear — use autoencoders, UMAP, t-SNE, kernel PCA.
- When interpretability of the original features is required — use feature selection instead.
- When features are categorical or mixed-type — PCA assumes Euclidean geometry.
- On sparse high-dimensional data (e.g., TF-IDF), PCA densifies the representation, which can be a memory disaster. Use truncated SVD without centering, or use sparse methods.

**Alternatives:**
- Kernel PCA for non-linear structure via the kernel trick.
- Sparse PCA (enforce sparsity on PC loadings for interpretability).
- Random projections (Johnson–Lindenstrauss): surprisingly competitive with PCA for approximate distance preservation, and much cheaper.
- Probabilistic PCA (a generative model giving PCA as the MAP under a specific Gaussian latent model — useful when you need likelihoods or want to handle missing data).
- ICA when you want statistical independence rather than decorrelation.

**Trade-offs:**
- **Interpretability**: PCs are linear combinations of all original features. A PC that loads on 500 features is uninterpretable.
- **Computational cost**: Full SVD is $O(\min(nd^2, n^2d))$. Randomized SVD is $O(nd k)$ for top-$k$.
- **Bias–variance**: PCA introduces bias (discards directions) to reduce variance (fewer parameters downstream). For small $n$, large $d$, this is usually a good trade.
- **Robustness**: PCA is non-robust to outliers. Robust PCA variants (e.g., RPCA via Principal Component Pursuit) exist.

### 1.6 Common Pitfalls & Misconceptions

1. **Not centering the data.** Leads to a first PC pointing toward the origin rather than capturing variance structure. Scikit-learn's `PCA` centers automatically; `TruncatedSVD` does not — pick deliberately.
2. **Not standardizing when scales differ.** A feature measured in nanometers will dominate one measured in kilometers purely by scale. Use z-score standardization unless there's a scientific reason not to.
3. **Using PCA for classification feature selection.** PCs are unsupervised. The class-discriminative direction may be a low-variance PC and get thrown out.
4. **Interpreting PCs as causal factors.** A PC is the direction of max variance, nothing more. Two totally different data-generating processes can yield the same PCs.
5. **Sign ambiguity.** PCs are defined up to sign. Flipping $v_j \to -v_j$ gives a valid eigenvector. Don't interpret "positive loading on feature X" without pinning down the sign convention.
6. **Applying PCA and then re-using eigenvectors computed on the training set.** Correct: fit PCA on train, apply the learned $V_k$ to test. Wrong: refit on test. This is a common leakage bug.
7. **Treating PCA as dimensionality reduction when $n < d$.** At most $n-1$ non-trivial PCs exist. Requesting more than that gives zero-variance directions.
8. **Assuming PCs are uncorrelated in the population.** They are uncorrelated *on the sample*. On a new sample, projections onto the learned axes may have non-zero sample correlation.
9. **Confusing PCA with Factor Analysis.** PCA decomposes variance; FA is a generative model with an explicit noise term. Different assumptions, different outputs (see §3).
10. **Scree plot over-reading.** The "elbow" is often ambiguous. Use explained variance thresholds (95%) or downstream cross-validation to choose $k$.

### 1.7 Interview Questions — PCA

#### Foundational

**Q1. What is PCA doing, in one sentence?**
Finding an orthonormal basis of directions in which the centered data has maximal variance, then projecting onto the top-$k$ such directions to get a lower-dimensional representation.

**Q2. Why do we center the data before PCA?**
Because variance is defined around the mean. Without centering, the first "PC" points toward the data centroid, not toward the direction of maximum spread. The reconstruction-error formulation also assumes the subspace passes through the origin.

**Q3. Why do we often standardize features before PCA?**
PCA maximizes variance in the original units. Features with larger numerical scales (e.g., income in dollars vs. age in years) contribute more variance purely because of units, which biases PC directions toward those features. Standardizing (dividing by std) makes variance scale-free.

**Q4. What does "explained variance ratio" mean?**
The fraction of total variance captured by a principal component: $\lambda_j / \sum_i \lambda_i$. Cumulative version gives the fraction captured by the top $k$ PCs.

**Q5. PCA vs. LDA — what's the conceptual difference?**
PCA is unsupervised and maximizes total variance; LDA is supervised and maximizes between-class variance relative to within-class variance. PCA finds "informative" directions (in the variance sense); LDA finds "discriminative" directions.

#### Mathematical

**Q6. Derive PCA from the max-variance criterion.**
Centered data, $\Sigma = \frac{1}{n-1}X^\top X$. Maximize $w^\top \Sigma w$ subject to $\|w\|=1$. Lagrangian: $\mathcal{L} = w^\top \Sigma w - \lambda(w^\top w - 1)$. $\nabla_w \mathcal{L} = 2\Sigma w - 2\lambda w = 0 \implies \Sigma w = \lambda w$. So $w$ is an eigenvector of $\Sigma$, with objective value $\lambda$. Top eigenvalue yields top PC.

**Q7. Show PCA also minimizes reconstruction error.**
For $W_k$ with orthonormal columns, project $\hat{x}_i = W_k W_k^\top x_i$. Error $\sum_i \|x_i - \hat{x}_i\|^2 = \sum_i \|x_i\|^2 - \mathrm{tr}(W_k^\top X^\top X W_k)$. Minimizing is equivalent to maximizing the trace, which by the Courant–Fischer / Ky Fan theorem is maximized by the top $k$ eigenvectors of $X^\top X$.

**Q8. How does SVD relate to PCA?**
$X = USV^\top$. Then $X^\top X = V S^2 V^\top$, so columns of $V$ are PCs, singular values squared give eigenvalues ($\lambda_j = \sigma_j^2/(n-1)$), and the scores are $XV = US$.

**Q9. If $d > n$, how many non-zero PCs can you have?**
At most $n - 1$ (one degree of freedom lost to centering). The rank of the centered $X$ is at most $n-1$, so $\Sigma$ has at most $n-1$ non-zero eigenvalues.

**Q10. Show that the PCs are uncorrelated (on the sample).**
The score matrix is $Z = XV$. Then $Z^\top Z = V^\top X^\top X V = V^\top V S^2 V^\top V = S^2$, which is diagonal. So different score columns have zero sample covariance.

**Q11. What is the relationship between PCA and the eigenvectors of the Gram matrix $XX^\top$?**
$XX^\top = U S^2 U^\top$. Columns of $U$ are eigenvectors of the $n \times n$ Gram matrix. This is the dual view: useful when $n \ll d$. The scores can be computed from $U$ and $S$ directly. This also underlies kernel PCA: replace $X X^\top$ with a kernel matrix $K$.

**Q12. When do we prefer the eigendecomposition of $\Sigma$ vs. the SVD of $X$?**
SVD is numerically more stable (forming $X^\top X$ squares the condition number) and handles $n \ll d$ and $n \gg d$ gracefully. Eigendecomp of $\Sigma$ is cheaper if $d \ll n$ and $\Sigma$ is explicitly available.

#### Applied

**Q13. Your dataset has 10k features, 500 samples. How do you do PCA efficiently?**
$d \gg n$. Don't form the $d \times d$ covariance. Either (a) SVD of the centered $n \times d$ matrix directly (economy SVD, $O(n^2 d)$), or (b) eigendecompose the $n \times n$ Gram matrix, recover PCs via $V = X^\top U S^{-1}$.

**Q14. You apply PCA and your downstream classifier gets worse. What went wrong?**
Likely the discriminative direction was in a low-variance PC that got discarded. PCA ignores labels. Try LDA, supervised feature selection, or keep more components. Also check whether standardization was correct, whether test data was projected using the training $V$, and whether classes are well-separated only in the original space.

**Q15. You want to visualize a dataset in 2D but your data is highly non-linear. Why might PCA fail?**
PCA only finds linear subspaces. A manifold like the Swiss roll projects to an uninterpretable blob. Use t-SNE, UMAP, or kernel PCA.

**Q16. How do you pick the number of components?**
Options: (a) elbow in the scree plot, (b) cumulative explained variance threshold (commonly 90–95%), (c) cross-validation on the downstream task, (d) likelihood-based criteria under probabilistic PCA, (e) parallel analysis (compare eigenvalues to those from random permuted data).

**Q17. Is PCA affected by outliers?**
Yes, heavily. A single outlier can pull a PC toward itself. Robust alternatives: robust PCA (decomposes $X$ into low-rank + sparse), weighted PCA, or pre-cleaning with robust covariance estimators (MCD).

**Q18. You have $10^6$ samples, $10^4$ features, need top 50 PCs. What algorithm?**
Randomized SVD. $O(n d k)$ vs. $O(n d^2)$ for full SVD. Scikit-learn's `PCA` with `svd_solver='randomized'` or libraries like scikit-learn's `IncrementalPCA` for streaming.

#### Debugging & Failure Modes

**Q19. You ran PCA, kept 2 components, plotted, and see a giant blob with one outlier far away. What's happening?**
An extreme outlier is inflating the variance along its direction, making that direction PC1. Most of your data is projected to a small range near the origin. Remove or winsorize outliers; consider robust PCA.

**Q20. Your PCA output is wildly different when you re-run it on a slightly perturbed dataset. Why?**
Probably two eigenvalues are nearly equal, so the subspace they span is well-defined but the individual eigenvectors rotate freely within it. This is a near-degeneracy. Check the eigenvalue gaps; interpret the subspace, not individual PCs.

**Q21. After PCA + linear regression, your test error exploded. What's a likely cause?**
Data leakage: you fit PCA on train+test combined. Or, you fit PCA on test. Always fit on train only, transform test with the learned projection.

**Q22. A colleague says, "I standardized, ran PCA, kept 10 components explaining 99% variance, and my classifier is at chance." Diagnose.**
Standardization gives every feature equal variance. If most features are pure noise and a few carry the signal, standardization *amplifies* noise. 99% of variance is then 99% of noise. Try no standardization, or use supervised methods that can down-weight noise.

#### Probing / Depth

**Q23. Why is whitening sometimes desirable, and why is it sometimes harmful?**
Whitening makes the projected data isotropic (identity covariance). Good for: ICA (which requires whitened input), downstream methods that assume isotropy. Harmful when: low-variance directions are noise-dominated, because whitening blows up the noise by the factor $1/\sqrt{\lambda_j}$.

**Q24. What's the probabilistic interpretation of PCA?**
Probabilistic PCA (Tipping & Bishop, 1999): $x = W z + \mu + \epsilon$ where $z \sim \mathcal{N}(0, I_k)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I_d)$. Marginalizing $z$: $x \sim \mathcal{N}(\mu, WW^\top + \sigma^2 I)$. The MLE of $W$ (up to orthogonal rotation) is the top-$k$ PCA loadings scaled by $\sqrt{\lambda_j - \sigma^2}$. In the limit $\sigma^2 \to 0$, PPCA recovers PCA exactly. This framework gives you: likelihoods, missing data handling via EM, Bayesian extensions.

**Q25. Can PCA be kernelized?**
Yes — kernel PCA. Replace inner products $x_i^\top x_j$ with kernel evaluations $k(x_i, x_j)$. Eigendecompose the centered kernel matrix $K$. Result: non-linear dimensionality reduction. Downside: no closed-form out-of-sample mapping (need the Nyström extension or similar), and cost is $O(n^3)$.

**Q26. Does PCA find "the" best low-dimensional representation?**
Only under specific criteria (variance maximization / squared reconstruction error). It is optimal among *linear* methods. For non-linear structure, it's provably suboptimal. Also, it assumes squared error is the right loss; for binary or count data, other methods (logistic PCA, exponential family PCA) do better.

**Q27. What does "rotation" mean in PCA and FA, and why is PCA "unrotatable"?**
In Factor Analysis, rotations (varimax, promax) are applied to loadings for interpretability — they are identifiability artifacts. For PCA, rotating PCs destroys the max-variance ordering, so it's rarely done. Technically, any rotation of the top $k$ PCs still spans the optimal subspace, but individual PCs lose their variance-ranking meaning.

**Q28. A new datapoint arrives. How do you update PCA without recomputing everything?**
Incremental PCA / online PCA. Techniques: Oja's rule (gradient ascent on variance), randomized streaming SVD, incremental algorithms that update $U$, $S$, $V$ rank-1. Scikit-learn exposes `IncrementalPCA`.

---

## 2. Linear Discriminant Analysis (LDA)

### 2.1 Motivation & Intuition

Consider a two-class classification problem where the classes form two elongated, parallel clouds — say, one cloud running along the direction $(1, 0)$ at $y = 1$, and another at $y = -1$. The total variance is dominated by the $x$-axis (elongation), but the discriminative direction is the $y$-axis. If you run PCA and keep one component, you get $x$ — useless for classification. The *right* answer is the direction that separates the class means while keeping each class tight.

LDA formalizes this: find a projection that **maximizes between-class separation relative to within-class spread**. It is the supervised cousin of PCA.

Concrete ML motivations:
- **Supervised preprocessing**: reduce dimensionality while preserving discriminative information.
- **A classifier in its own right**: Gaussian class-conditionals with shared covariance yields LDA as the Bayes-optimal classifier.
- **Face recognition**: "Fisherfaces" (LDA applied to face images) outperformed "Eigenfaces" (PCA) when labels were available.
- **Multi-class visualization**: project to $c-1$ dimensions where $c$ is the number of classes.

### 2.2 Conceptual Foundations

**Setup.** Data $\{(x_i, y_i)\}_{i=1}^n$ with $x_i \in \mathbb{R}^d$, $y_i \in \{1, \dots, c\}$. Let $n_k$ be the count of class $k$, $\mu_k$ its mean, $\mu$ the overall mean.

**Scatter matrices.**
- **Within-class scatter**: $S_W = \sum_{k=1}^c \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^\top$. Measures spread inside each class.
- **Between-class scatter**: $S_B = \sum_{k=1}^c n_k (\mu_k - \mu)(\mu_k - \mu)^\top$. Measures spread of class means around the overall mean.
- **Total scatter**: $S_T = \sum_i (x_i - \mu)(x_i - \mu)^\top = S_W + S_B$ (a standard decomposition).

**Goal.** Find a projection matrix $W \in \mathbb{R}^{d \times k}$ that maximizes
$$
J(W) = \frac{|W^\top S_B W|}{|W^\top S_W W|}.
$$
Maximize between-class spread; minimize within-class spread.

**Assumptions.**
1. **Classes are Gaussian with shared covariance** ($p(x|y=k) = \mathcal{N}(\mu_k, \Sigma)$). If covariances differ substantially, LDA is misspecified — use QDA (quadratic discriminant analysis), which drops the shared-covariance assumption but loses the linearity.
2. **Enough samples to estimate $S_W$ reliably.** $S_W$ is a $d \times d$ matrix needing $\gg d$ samples to be invertible and well-conditioned. When $n < d$, $S_W$ is singular (see "small-sample problem" below).
3. **Linear separability is meaningful.** If the optimal decision surface is highly non-linear, LDA is limited.
4. **Class priors reflect reality.** Sample class frequencies estimate priors; imbalance can distort $S_B$.

**What breaks:**
- **Unequal covariances**: one class may be projected onto a direction where it overlaps another class badly.
- **Non-Gaussian classes**: a long thin curved class can't be collapsed to a point along a line.
- **$n < d$**: $S_W$ singular; standard LDA undefined. Fix with PCA pre-projection, regularization (shrinkage LDA), or pseudo-inverse.
- **More than $c-1$ components requested**: $S_B$ has rank at most $c - 1$ (it's a sum of $c$ rank-1 matrices minus one constraint from $\mu$), so you can't extract more than $c-1$ non-trivial discriminant directions.

### 2.3 Mathematical Formulation

**Two-class case (Fisher's original formulation).** For a single direction $w$:
$$
J(w) = \frac{w^\top S_B w}{w^\top S_W w}.
$$
Set gradient to zero: $2 S_B w (w^\top S_W w) - 2 S_W w (w^\top S_B w) = 0$, equivalently $S_B w = \lambda S_W w$ where $\lambda = J(w)$. This is a **generalized eigenvalue problem**. If $S_W$ is invertible, it becomes $S_W^{-1} S_B w = \lambda w$ — a standard eigenvalue problem.

In the two-class case, $S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^\top$ is rank 1, so there's only one non-trivial solution:
$$
w^* \propto S_W^{-1}(\mu_1 - \mu_2).
$$

**Multi-class case.** The top $k \leq c-1$ generalized eigenvectors of $S_B v = \lambda S_W v$ give the discriminant directions.

**Connection to Gaussian classification.** Assume $p(x|y=k) = \mathcal{N}(\mu_k, \Sigma)$ with shared $\Sigma$, priors $\pi_k$. The log-posterior is
$$
\log p(y=k|x) \propto -\frac{1}{2}(x-\mu_k)^\top \Sigma^{-1}(x - \mu_k) + \log \pi_k.
$$
Expanding and dropping terms not depending on $k$:
$$
\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \frac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log \pi_k.
$$
This is linear in $x$ — hence "Linear" Discriminant Analysis. The decision boundary between classes $k$ and $l$ is $\delta_k(x) = \delta_l(x)$, a hyperplane. With shared $\Sigma$ replaced by $S_W / (n - c)$, this recovers the LDA classifier.

**Why is this the same thing as Fisher's criterion?** Both reduce to finding directions $\Sigma^{-1}(\mu_k - \mu_l)$, which are exactly the generalized eigenvectors of $S_W^{-1} S_B$. Fisher derived it geometrically; Gaussian Bayes derives it probabilistically; they coincide.

**Regularized LDA.** Replace $S_W$ with $(1-\alpha)S_W + \alpha I$ (or $+ \alpha \cdot \mathrm{diag}(S_W)$) for some $\alpha \in (0,1)$. Fixes singularity when $n < d$ and improves stability. Ledoit–Wolf shrinkage gives a data-driven $\alpha$.

### 2.4 Worked Example

Two classes in 2D:
- Class 1: $(1,2), (2,3), (3,3)$.
- Class 2: $(4,1), (5,2), (6,1)$.

**Step 1: Class means.**
$\mu_1 = (2, 8/3) \approx (2, 2.67)$, $\mu_2 = (5, 4/3) \approx (5, 1.33)$.

**Step 2: Within-class scatter.**
Class 1 deviations: $(-1, -0.67), (0, 0.33), (1, 0.33)$. Contribution:
$$
\begin{pmatrix} 1 & 0.67 \\ 0.67 & 0.447 \\ \end{pmatrix} + \begin{pmatrix}0 & 0 \\ 0 & 0.11\end{pmatrix} + \begin{pmatrix}1 & 0.33 \\ 0.33 & 0.11\end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 0.67 \end{pmatrix}.
$$

Class 2 deviations: $(-1, -0.33), (0, 0.67), (1, -0.33)$. Contribution:
$$
\begin{pmatrix}1 & 0.33 \\ 0.33 & 0.11\end{pmatrix} + \begin{pmatrix}0 & 0 \\ 0 & 0.447\end{pmatrix} + \begin{pmatrix}1 & -0.33 \\ -0.33 & 0.11\end{pmatrix} = \begin{pmatrix}2 & 0 \\ 0 & 0.67\end{pmatrix}.
$$

$$
S_W = \begin{pmatrix}4 & 1 \\ 1 & 1.33\end{pmatrix}.
$$

**Step 3: Fisher's direction.**
$\mu_1 - \mu_2 = (-3, 1.33)$.
$\det(S_W) = 4(1.33) - 1 = 4.33$.
$S_W^{-1} = \frac{1}{4.33}\begin{pmatrix}1.33 & -1 \\ -1 & 4\end{pmatrix}$.

$S_W^{-1}(\mu_1 - \mu_2) = \frac{1}{4.33}\begin{pmatrix}1.33(-3) - 1(1.33) \\ -1(-3) + 4(1.33)\end{pmatrix} = \frac{1}{4.33}\begin{pmatrix}-5.33 \\ 8.33\end{pmatrix} \approx \begin{pmatrix}-1.23 \\ 1.92\end{pmatrix}.$

Normalize: $w \approx (-0.54, 0.84)$.

**Interpretation.** The discriminant direction points "up and to the left," capturing that class 1 is upper-left and class 2 is lower-right. Compare with the naive "mean-difference" direction $\mu_1 - \mu_2 \propto (-3, 1.33)$, which normalized is $(-0.91, 0.40)$. LDA tilts upward more, because $S_W$ shows that the $y$-axis has lower within-class variance, making it more discriminative.

**Projected scores** (class 1 followed by class 2):
$w^\top x$: class 1 gives roughly $\{1.14, 1.44, 0.90\}$, class 2 gives $\{-1.32, -0.98, -2.40\}$. Cleanly separated.

### 2.5 Relevance to ML Practice

**Where it's used:**
- Classical preprocessing for classification when classes are roughly Gaussian.
- As a classifier itself — often a strong baseline, especially in low-data regimes.
- Bioinformatics (gene expression → disease class).
- Face recognition (Fisherfaces).
- Multi-class visualization: $c=3$ gives 2D projection naturally.

**When not to use it:**
- Non-Gaussian classes (use QDA, nonparametric classifiers, or non-linear embeddings).
- Severe class imbalance (handle priors carefully or use weighted variants).
- High $d$, low $n$ without regularization.
- Strongly non-linear decision boundaries (use kernel LDA or neural classifiers).

**Alternatives:**
- **Logistic regression** — also linear, less restrictive assumptions (no Gaussianity), often preferred unless the Gaussian assumption is genuinely appropriate.
- **QDA** — per-class covariances; more parameters, more flexible, but overfits easily.
- **Regularized / Shrinkage LDA** — for $n \lesssim d$.
- **Kernel Fisher Discriminant** — non-linear generalization.
- **Neural classifiers** — when data is abundant and non-linear.

**Trade-offs:**
- **Bias–variance**: LDA's Gaussian assumption is biased but gives low-variance estimates (few parameters).
- **Interpretability**: directions are linear combinations of features, like PCA. Somewhat interpretable with few features.
- **Computation**: Dominated by eigendecomposition of $S_W^{-1} S_B$, $O(d^3)$. For large $d$, use PCA first.

### 2.6 Common Pitfalls & Misconceptions

1. **Using LDA when classes aren't (approximately) Gaussian with shared covariance.** LDA's performance degrades silently; no built-in warning.
2. **Requesting more than $c-1$ components.** $S_B$ has rank $\leq c-1$; remaining directions are arbitrary.
3. **$n < d$ without regularization.** $S_W$ singular. Symptoms: numerical explosions, bizarre directions. Fix: reduce $d$ via PCA first, or add ridge.
4. **Forgetting to center class-wise.** $S_W$ uses class-specific means, not the overall mean. Confusing the two is a common bug.
5. **Severe class imbalance.** Tiny classes contribute little to $S_B$, so the discriminant direction is dominated by the majority pair. Can reweight or use balanced sampling.
6. **Interpreting LDA as "the" Bayes classifier for any data.** Only under the Gaussian + shared-covariance assumption.
7. **Using LDA when you really want PCA** or vice versa. LDA requires labels; if labels unreliable or irrelevant downstream, PCA might be preferable.
8. **PCA→LDA pipeline without careful component count choice.** Discarding too many PCs can remove discriminative directions. Use CV to choose.
9. **Assuming LDA is robust to outliers.** Like PCA, sensitive to extreme points.
10. **Conflating LDA (dimensionality reduction) with LDA (Latent Dirichlet Allocation, for topic modeling).** Totally unrelated — same acronym, different technique.

### 2.7 Interview Questions — LDA

#### Foundational

**Q1. What does LDA try to maximize?**
The ratio of between-class scatter to within-class scatter, so projected classes are far apart and internally tight.

**Q2. Why is it called "linear" discriminant analysis?**
The decision boundary (under shared Gaussian covariance) is linear in $x$; equivalently, the projection onto discriminant directions is linear.

**Q3. How does LDA differ from PCA?**
PCA is unsupervised, maximizing total variance; LDA is supervised, maximizing class separability. PCA might discard discriminative directions; LDA explicitly preserves them.

**Q4. What's the maximum number of LDA components you can extract for $c$ classes?**
$c - 1$ (rank of $S_B$).

**Q5. What assumptions does LDA make?**
Class-conditional Gaussianity with a shared covariance matrix; linear class boundaries; reliable class priors.

#### Mathematical

**Q6. Derive Fisher's linear discriminant for two classes.**
Objective $J(w) = \frac{w^\top S_B w}{w^\top S_W w}$ with $S_B = (\mu_1-\mu_2)(\mu_1-\mu_2)^\top$. Take gradient:

$$
\nabla J = \frac{2 S_B w (w^\top S_W w) - 2 S_W w (w^\top S_B w)}{(w^\top S_W w)^2} = 0,
$$

so $S_B w = \lambda S_W w$ with $\lambda = J$. Since $S_B w = (\mu_1-\mu_2) \cdot [(\mu_1-\mu_2)^\top w]$ is a scalar times $(\mu_1-\mu_2)$, we get $w \propto S_W^{-1}(\mu_1 - \mu_2)$.

**Q7. Show LDA arises from Bayes-optimal classification under Gaussian class-conditionals with shared covariance.**
$\log p(y=k|x) = -\tfrac{1}{2}(x-\mu_k)^\top \Sigma^{-1}(x-\mu_k) + \log \pi_k + \mathrm{const}$. Expand the quadratic: $-\tfrac{1}{2}x^\top \Sigma^{-1}x + x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log \pi_k$. The first term doesn't depend on $k$, so the discriminant function is $\delta_k(x) = x^\top \Sigma^{-1}\mu_k - \tfrac{1}{2}\mu_k^\top \Sigma^{-1}\mu_k + \log \pi_k$ — linear in $x$.

**Q8. What happens when $S_W$ is singular?**
Standard LDA undefined. Solutions: (a) PCA first to reduce $d$ below $n$; (b) regularize: $S_W \leftarrow S_W + \alpha I$; (c) use pseudo-inverse; (d) null-space LDA (project into null space of $S_W$, do LDA there).

**Q9. What is $S_T = S_W + S_B$? Prove.**
Total scatter decomposes: $\sum_i (x_i - \mu)(x_i - \mu)^\top$. Split by class. For fixed class $k$ with $n_k$ points and mean $\mu_k$:
$\sum_{i \in k} (x_i - \mu)(x_i - \mu)^\top = \sum_{i \in k}[(x_i - \mu_k) + (\mu_k - \mu)][(x_i - \mu_k) + (\mu_k - \mu)]^\top$. Cross terms sum to zero (since $\sum_{i \in k} (x_i - \mu_k) = 0$), yielding $\sum_{i \in k}(x_i-\mu_k)(x_i-\mu_k)^\top + n_k (\mu_k - \mu)(\mu_k - \mu)^\top$. Summing over $k$: $S_W + S_B$.

**Q10. What's the difference between LDA and QDA?**
QDA allows per-class covariances $\Sigma_k$. The quadratic term $x^\top \Sigma_k^{-1} x$ no longer cancels across classes, giving quadratic decision boundaries. More parameters ($O(c d^2)$ vs. $O(d^2)$ for LDA), higher variance, more data needed.

#### Applied

**Q11. You have 3 classes in 10-D space. How many LDA dimensions can you extract?**
Up to $c - 1 = 2$. Convenient for visualization.

**Q12. LDA is giving you poor results. What do you check?**
Gaussian assumption (per-class normality / similar covariances — plot, Box's M test), sample sizes (is $S_W$ singular?), class imbalance, outliers, whether the problem is linearly separable at all.

**Q13. When would you choose logistic regression over LDA?**
When class-conditionals aren't Gaussian, or when you don't want to estimate $\Sigma$ (logistic only models $p(y|x)$ directly). Logistic is less restrictive and often equally accurate.

**Q14. Face recognition with 100 training images and $10^4$ pixels — LDA directly?**
No. $S_W$ singular. Run PCA first (keep, say, 50 components), then LDA on those. Classic "Fisherfaces" pipeline.

**Q15. Your dataset has 95% class A, 5% class B. How does this affect LDA?**
$S_B$ weights by class size, so class B contributes little to between-class scatter. The direction is dominated by within-A variation. Options: reweight classes equally, use balanced sampling, include class priors carefully in the classifier.

#### Debugging & Failure Modes

**Q16. LDA gives near-zero eigenvalues for all but one direction with 4 classes. Why?**
Classes' means may be roughly collinear, so $S_B$ has effective rank 1 instead of 3. Check class means; may indicate redundancy or that 3 of 4 classes are nearly identical.

**Q17. Adding features makes LDA worse. How?**
More features → noisier $S_W$ estimate; if added features are noise, they contribute more variance to $S_W$ than to $S_B$, shrinking the criterion. Also, $S_W$ may approach singularity. Regularize or do supervised feature selection first.

**Q18. Your LDA classifier has high training accuracy but poor test accuracy. Diagnose.**
Overfitting. Likely small $n$ vs. $d$; $S_W$ estimate is noisy and inverting it amplifies noise. Apply shrinkage LDA (Ledoit-Wolf), PCA pre-reduction, or cross-validate on component count.

#### Probing / Depth

**Q19. Connection between LDA and a least-squares problem?**
For two classes, encoding $y \in \{n/n_1, -n/n_2\}$ (or any two distinct values) and doing linear regression $\min_w \|Xw - y\|^2$ yields $w \propto S_W^{-1}(\mu_1 - \mu_2)$ — the same as Fisher's direction. So two-class LDA = least-squares with particular encoding.

**Q20. Why does the shared-covariance assumption matter so much?**
It's what makes the discriminant linear. With different $\Sigma_k$, the quadratic terms don't cancel and you get QDA (quadratic boundary). If you apply LDA when classes have different shapes/orientations, the linear approximation can be badly biased.

**Q21. Can LDA be kernelized?**
Yes — Kernel Fisher Discriminant (KFD). Express $w = \sum_i \alpha_i \phi(x_i)$ and reformulate in terms of kernel matrices. Gives non-linear discriminants at $O(n^3)$ cost.

**Q22. What happens if you apply LDA but ignore priors (treat all classes equally in $S_B$)?**
You get a direction that maximizes separation of means treated uniformly, which may differ from the Bayes-optimal direction under the true priors. For heavily imbalanced cases, the uniform version can be more useful if you care about minority class.

**Q23. How does LDA relate to Canonical Correlation Analysis (CCA)?**
LDA with one-hot class encoding is equivalent to CCA between features and class indicators. This generalizes LDA to predicting continuous targets (regression) and to multi-output problems.

---

## 3. Factor Analysis (FA)

### 3.1 Motivation & Intuition

Suppose you administer 20 psychological test items (arithmetic, vocabulary, pattern completion, etc.) to 1000 people. Scores are correlated — people good at one thing tend to be good at related things. You suspect a small number of **latent factors** ("verbal ability," "quantitative ability") drive most of the correlations, but each item also has its own idiosyncratic noise (a bad day, a confusing question).

Factor Analysis is the generative model that formalizes this:
$$
x = \Lambda z + \mu + \epsilon,
$$
where $z \in \mathbb{R}^k$ are latent factors (few), $\Lambda \in \mathbb{R}^{d \times k}$ is the loadings matrix, $\epsilon$ is per-feature noise with diagonal covariance $\Psi = \mathrm{diag}(\psi_1, \dots, \psi_d)$.

Crucial difference from PCA: FA has an **explicit noise model**, and the noise is per-feature rather than isotropic. This matters because features in many datasets have wildly different noise levels (one sensor is accurate, another is flaky), and FA accounts for this while PCA conflates noise and signal.

Concrete ML/data-science motivations:
- **Psychometrics, marketing surveys**: isolate underlying constructs from noisy items.
- **Dense feature synthesis**: summarize many correlated measurements into few factors.
- **Missing data**: FA's probabilistic formulation naturally handles missing values via EM.
- **Denoising**: the model explicitly separates shared signal ($\Lambda z$) from per-feature noise.

### 3.2 Conceptual Foundations

**Key terms.**
- **Common factors** $z$: unobserved latent variables, usually $z \sim \mathcal{N}(0, I_k)$.
- **Loadings** $\Lambda$: how each factor expresses in each observed feature.
- **Unique factors / specific factors** $\epsilon$: idiosyncratic per-feature variation, $\epsilon \sim \mathcal{N}(0, \Psi)$ with $\Psi$ diagonal.
- **Communality** of feature $j$: $\sum_{l=1}^k \Lambda_{jl}^2$, the variance of feature $j$ explained by factors.
- **Uniqueness** of feature $j$: $\psi_j$, variance *not* explained by factors.

**Implied covariance.** Under the model, $\mathrm{Cov}(x) = \Lambda \Lambda^\top + \Psi$. FA is fit by finding $\Lambda, \Psi$ that best reproduce the observed covariance — typically via MLE under Gaussianity.

**Assumptions.**
1. Factors are Gaussian with identity covariance (WLOG — any other Gaussian can be absorbed into $\Lambda$).
2. Noise is diagonal — features are conditionally independent given factors. This is FA's defining assumption.
3. Linearity — $x$ is a linear function of $z$ plus noise.
4. $k < d$, typically $k \ll d$.

**What breaks.**
- If noise is correlated across features (e.g., two sensors share a cable picking up interference), the diagonality assumption fails; factors absorb this correlation and get misinterpreted.
- Non-Gaussian factors can't be identified up to more than an orthogonal rotation — and even Gaussian factors are identifiable only up to rotation. This is why ICA, which drops Gaussianity, can identify independent components uniquely.
- Non-linear relationships: FA finds a linear factor structure and misses curvature.

**Rotation.** Because $\Lambda z$ and $(\Lambda Q)(Q^\top z)$ give the same $x$ for any orthogonal $Q$, factor loadings are identified only up to rotation. This means there's no "canonical" $\Lambda$; interpreters choose rotations that make loadings easier to read.

- **Varimax** (orthogonal rotation): maximizes variance of squared loadings within each factor. Pushes loadings toward 0 or $\pm 1$, yielding factors with few "high" loadings each — interpretable.
- **Promax / Oblimin** (oblique rotations): allow factors to be correlated. More realistic when underlying constructs aren't truly orthogonal (e.g., verbal and quantitative abilities are correlated).
- **Quartimax**: maximizes variance of squared loadings within each *variable* — a variable loads on few factors.

### 3.3 Mathematical Formulation

**Model.** $z \sim \mathcal{N}(0, I_k)$, $\epsilon \sim \mathcal{N}(0, \Psi)$ independent of $z$, $\Psi$ diagonal PSD. Then $x | z \sim \mathcal{N}(\Lambda z + \mu, \Psi)$, and marginally
$$
x \sim \mathcal{N}(\mu, \Lambda \Lambda^\top + \Psi).
$$

**Posterior over factors.** By Gaussian conjugacy,
$$
z | x \sim \mathcal{N}\left( (I + \Lambda^\top \Psi^{-1}\Lambda)^{-1}\Lambda^\top \Psi^{-1}(x - \mu), \; (I + \Lambda^\top \Psi^{-1}\Lambda)^{-1} \right).
$$
The posterior mean is used as the "factor score" for a given $x$.

**MLE via EM.** Log-likelihood:
$$
\ell(\Lambda, \Psi) = -\frac{n}{2}\log|2\pi(\Lambda\Lambda^\top + \Psi)| - \frac{1}{2}\sum_i (x_i - \mu)^\top(\Lambda\Lambda^\top + \Psi)^{-1}(x_i - \mu).
$$
No closed form in general. EM algorithm:

- **E-step.** For each $x_i$, compute $\mathbb{E}[z_i | x_i]$ and $\mathbb{E}[z_i z_i^\top | x_i]$ using the posterior above.
- **M-step.** Update
$$
\Lambda^{new} = \left(\sum_i (x_i - \mu)\mathbb{E}[z_i|x_i]^\top\right)\left(\sum_i \mathbb{E}[z_i z_i^\top | x_i]\right)^{-1},
$$
$$
\Psi^{new} = \mathrm{diag}\left(\frac{1}{n}\sum_i (x_i - \mu)(x_i - \mu)^\top - \Lambda^{new}\mathbb{E}[z_i|x_i](x_i - \mu)^\top\right).
$$
Iterate. Converges to a local maximum of the likelihood.

**Relationship to PCA.** Let $\Psi = \sigma^2 I$ (isotropic noise instead of diagonal). Then FA reduces to **Probabilistic PCA** (PPCA). Taking $\sigma^2 \to 0$ recovers classical PCA. So PCA is a *constrained* FA: noise forced to be isotropic, not per-feature. This is why FA gives different answers when features have wildly different noise levels: it *models* that mismatch explicitly.

**Rotation formally.** $\Lambda$ and $\Lambda Q$ (for orthogonal $Q$) induce the same $\mathrm{Cov}(x)$. Define a rotation criterion $V(\Lambda Q)$ (e.g., varimax) and optimize over $Q$.

Varimax: maximize
$$
V(\Lambda) = \sum_{l=1}^k \left[\frac{1}{d}\sum_j \lambda_{jl}^4 - \left(\frac{1}{d}\sum_j \lambda_{jl}^2\right)^2\right].
$$
This is the variance of squared loadings per factor; pushing it up encourages each factor's loadings to be either clearly large or clearly near zero.

### 3.4 Worked Example

**Sketch.** Suppose 6 test items measure two underlying abilities:
- Items 1–3: math (arithmetic, algebra, geometry).
- Items 4–6: verbal (synonyms, reading, grammar).

Ideal loadings (after rotation):
$$
\Lambda = \begin{pmatrix} 0.9 & 0.1 \\ 0.8 & 0.1 \\ 0.85 & 0.0 \\ 0.1 & 0.9 \\ 0.0 & 0.85 \\ 0.05 & 0.8 \end{pmatrix}.
$$

Uniqueness $\Psi \approx \mathrm{diag}(0.2, 0.3, 0.2, 0.2, 0.3, 0.3)$ — each item has its own noise.

Implied covariance: $\Lambda\Lambda^\top + \Psi$. Feature 1 and feature 2: $\mathrm{Cov}(x_1, x_2) = (0.9)(0.8) + (0.1)(0.1) = 0.73$. Features 1 and 4: $\mathrm{Cov}(x_1, x_4) = (0.9)(0.1) + (0.1)(0.9) = 0.18$. Two clear clusters.

Before rotation (raw MLE), loadings might be mixed: $\Lambda^{raw}$ could place both math and verbal signal in PC-like directions, mixing abilities. Varimax rotates into the interpretable block form above.

**Numeric factor score (posterior).** Given $x = (1, 0.8, 0.9, 0.2, 0.1, 0.0)^\top$ (a math-heavy student), posterior mean for $z$ is approximately $(1, 0)^\top$: factor 1 (math) is high, factor 2 (verbal) is low.

**Full 4-step toy computation.** Two features, one factor. $x_1 = \lambda_1 z + \epsilon_1$, $x_2 = \lambda_2 z + \epsilon_2$, $z \sim \mathcal{N}(0,1)$, $\epsilon_j \sim \mathcal{N}(0, \psi_j)$.

Sample covariance (say): $\mathrm{Var}(x_1) = 1$, $\mathrm{Var}(x_2) = 1$, $\mathrm{Cov}(x_1, x_2) = 0.5$.

Model says: $1 = \lambda_1^2 + \psi_1$, $1 = \lambda_2^2 + \psi_2$, $0.5 = \lambda_1 \lambda_2$.

With $k=1, d=2$: $d(d+1)/2 = 3$ equations, $d+k = 3$ unknowns — just identified. Take $\lambda_1 = \lambda_2 = \sqrt{0.5}$, $\psi_1 = \psi_2 = 0.5$. Each feature has 50% of its variance from the common factor, 50% unique.

### 3.5 Relevance to ML Practice

**Where it's used:**
- Psychometric / behavioral / marketing data with hypothesized latent traits.
- Finance (the "factor models" behind APT, Fama–French, risk factor decomposition).
- Dense feature compression with interpretable latent structure.
- Missing-data imputation via EM.
- Mixture of FA for density estimation / clustering.

**When not to use it:**
- Pure compression / reconstruction without concern for interpretability — use PCA.
- Non-Gaussian / independent sources — use ICA.
- Non-linear manifold structure — use autoencoders or UMAP.
- When you need source-separation — ICA gives identifiability that FA can't.

**Alternatives:**
- PCA for variance-focused dimensionality reduction.
- PPCA if you want PCA with a likelihood.
- ICA for independent non-Gaussian sources.
- Mixture models for clustering with latent structure.
- VAEs for non-linear latent variable modeling.

**Trade-offs:**
- **Interpretability**: FA is more interpretable than PCA *with* a good rotation; useless without.
- **Computational cost**: EM is iterative and $O(nkd)$ per iteration; scales well.
- **Identifiability**: only up to orthogonal rotation; so $\Lambda$ is not uniquely defined, and claims about "factor 1 vs. factor 2" must be consumed carefully.
- **Robustness**: Gaussian assumption means outliers distort heavily; robust FA variants exist.

### 3.6 Common Pitfalls & Misconceptions

1. **Confusing FA with PCA.** They differ in noise model (FA: diagonal per-feature; PCA: no explicit noise / isotropic implicit noise) and objective (FA: likelihood; PCA: variance). When uniquenesses $\psi_j$ are nearly equal, results look similar; when they aren't, they diverge.
2. **Ignoring rotation.** Raw MLE loadings are often uninterpretable. Always apply varimax or an oblique rotation for human reading.
3. **Over-interpreting factors causally.** A factor is just a latent variable in a specific linear Gaussian model. Naming it "intelligence" is a leap. Factor reality is a century-old controversy in psychometrics.
4. **Heywood cases.** MLE can produce $\psi_j \leq 0$ (non-identifiable; unique variance negative). Fix: constrain $\psi_j \geq 0$, or regularize.
5. **Local maxima.** EM gets stuck in local optima; random restarts or good initialization (e.g., from PCA) help.
6. **Choosing $k$ arbitrarily.** Common heuristics: Kaiser (keep eigenvalues $> 1$), scree plot, parallel analysis, likelihood-ratio tests, BIC. Cross-validate where possible.
7. **Non-Gaussian data.** FA's MLE assumes Gaussian features. Heavy-tailed or skewed data distorts loadings.
8. **Using orthogonal rotations when factors should be correlated.** Varimax assumes factor independence; real constructs often aren't independent.
9. **Running FA on standardized data without noting it.** Results depend on whether you fit to correlation or covariance. Pre-register which.
10. **Believing you can tell FA and PCA apart by looking at the first component.** Often very similar; differences emerge in later components and in the treatment of high-noise features.

### 3.7 Interview Questions — Factor Analysis

#### Foundational

**Q1. What's the generative model of FA?**
$x = \Lambda z + \mu + \epsilon$, $z \sim \mathcal{N}(0, I_k)$, $\epsilon \sim \mathcal{N}(0, \Psi)$ with $\Psi$ diagonal.

**Q2. Core difference between FA and PCA?**
FA has a per-feature noise term ($\Psi$ diagonal); PCA has either no explicit noise or isotropic ($\sigma^2 I$). FA is a likelihood-based generative model; PCA is a variance-maximizing projection.

**Q3. What is a "communality"?**
For feature $j$: $\sum_l \Lambda_{jl}^2$ — variance of feature $j$ explained by common factors. Uniqueness is $1 - $ communality (after normalization).

**Q4. What does rotation do?**
Changes $\Lambda$ without changing the implied covariance. Used for interpretability.

**Q5. What's the difference between orthogonal and oblique rotation?**
Orthogonal keeps factors uncorrelated (e.g., varimax). Oblique allows factor correlations (e.g., promax, oblimin). Oblique is often more realistic but complicates interpretation.

#### Mathematical

**Q6. Derive the marginal covariance of $x$ under the FA model.**
$\mathrm{Cov}(x) = \mathbb{E}[(x-\mu)(x-\mu)^\top] = \mathbb{E}[(\Lambda z + \epsilon)(\Lambda z + \epsilon)^\top] = \Lambda \mathbb{E}[zz^\top]\Lambda^\top + \mathbb{E}[\epsilon \epsilon^\top] = \Lambda \Lambda^\top + \Psi$ (cross terms vanish by independence).

**Q7. What's the posterior distribution of $z$ given $x$?**
Gaussian, via Bayes. Mean: $(I + \Lambda^\top \Psi^{-1}\Lambda)^{-1}\Lambda^\top \Psi^{-1}(x - \mu)$. Covariance: $(I + \Lambda^\top \Psi^{-1}\Lambda)^{-1}$. Derivation: complete the square in the joint Gaussian density.

**Q8. Why is FA only identifiable up to rotation?**
$\Lambda \Lambda^\top = (\Lambda Q)(\Lambda Q)^\top$ for orthogonal $Q$. The model's sufficient statistic (covariance) is invariant to rotations of $\Lambda$.

**Q9. Show PCA is the limit of PPCA as $\sigma^2 \to 0$.**
Under PPCA, the MLE for $\Lambda$ is $U_k(\Lambda_k - \sigma^2 I)^{1/2}R$ for any orthogonal $R$ (Tipping-Bishop). As $\sigma^2 \to 0$, $\Lambda \to U_k \Lambda_k^{1/2}$, and the projected coordinates become the PCA scores (up to rotation).

**Q10. What's the EM algorithm for FA?**
E-step: compute $\mathbb{E}[z|x]$ and $\mathbb{E}[zz^\top|x]$ from posterior. M-step: update $\Lambda$ by regressing $x - \mu$ on $\mathbb{E}[z|x]$; update $\Psi$ as the diagonal of the residual covariance.

#### Applied

**Q11. You have 50 survey questions and suspect 5 latent traits. How do you proceed?**
Fit FA with $k=5$; use parallel analysis or BIC to confirm $k$. Apply varimax (if traits are independent) or promax (if they correlate). Inspect loadings per factor to name each trait. Compute factor scores per respondent for downstream use.

**Q12. Features have wildly different variances: FA or PCA?**
Both depend on scales. For FA, standardize first or fit to correlation matrix. For PCA, same. Factor models are particularly sensitive because uniqueness interpretation depends on absolute noise levels.

**Q13. Factor loadings look messy. What do you do?**
Apply varimax. If factors are truly correlated, try promax. Re-examine $k$.

**Q14. You want to impute missing values in a correlated feature set. Why is FA a natural tool?**
FA's generative form lets you compute $p(x_{\mathrm{missing}} | x_{\mathrm{observed}})$ analytically (Gaussian conditional) once $\Lambda, \Psi$ are fit. EM handles missing-at-random elegantly.

**Q15. Should you use FA or ICA for blind source separation (cocktail party problem)?**
ICA. FA only identifies subspace, not individual sources (rotation ambiguity under Gaussianity). ICA exploits non-Gaussianity to pin down specific sources.

#### Debugging & Failure Modes

**Q16. FA produces negative uniquenesses. Explanation and fix?**
Heywood case. The model is stretched beyond its valid parameter space, often due to small $n$, model misspecification, or $k$ too large. Constrain $\psi_j > 0$ (use software that supports this), reduce $k$, or regularize.

**Q17. Your factors change drastically when you add a few new samples. Why?**
Likely small $n$ or weakly identified factors (small eigenvalue gaps). Also, rotation is unstable across fits. Use bootstrapping to assess stability; increase $n$.

**Q18. After FA, two factors look like mirror images (opposite signs). What happened?**
Sign ambiguity, same as in PCA. Fix a convention (e.g., the highest-loading feature in each factor has positive loading).

#### Probing / Depth

**Q19. Why is the Gaussian assumption in FA more than a convenience?**
Gaussianity makes rotation ambiguity irremediable: any rotation preserves the joint distribution. Without Gaussianity (ICA), you can recover sources uniquely (up to sign and order).

**Q20. Connection between FA and linear latent-variable models (e.g., VAEs)?**
FA is the simplest linear-Gaussian latent model. A VAE extends this to non-linear decoders and amortized inference (neural encoder for $q(z|x)$). In the limit of linear encoder/decoder and Gaussian everything, a VAE reduces to PPCA / FA.

**Q21. How do you choose $k$ in FA?**
Parallel analysis, Kaiser criterion, BIC/AIC, likelihood-ratio tests (only if nested models), cross-validated held-out likelihood. Parallel analysis (comparing to eigenvalues from permuted data) is empirically one of the most reliable.

**Q22. Connection between varimax and sparse PCA?**
Both seek interpretable sparse loadings. Varimax achieves this post-hoc on FA loadings; sparse PCA enforces sparsity in the objective (e.g., $\ell_1$ penalty on loadings). Sparse PCA is more principled if sparsity is the goal; varimax is a cheap post-processing step.

**Q23. How would you do FA on non-Gaussian discrete data (e.g., 5-point Likert scale items)?**
Use polychoric correlations (assume a latent Gaussian underlies each item) to build the input covariance, then apply FA to that. Or use item-response-theory models, which are the principled generalization.

---

## 4. Independent Component Analysis (ICA)

### 4.1 Motivation & Intuition

The **cocktail party problem**: at a party, two people talk simultaneously. Two microphones at different positions each pick up a different linear mixture of the two voices. From the two recorded signals alone — no reference recordings, no position info — can you recover the two original voices?

This is **blind source separation**. "Blind" means no prior information about the sources; "separation" means recovering them from mixtures.

The key insight: real-world sources (speech, EEG sources, images) tend to be **non-Gaussian** and **independent** of each other. Mixing independent sources produces signals that are "more Gaussian" (by the Central Limit Theorem — sums of independent RVs trend Gaussian). So to unmix, **look for directions that make projected data as non-Gaussian as possible**.

Concrete ML motivations:
- **Signal separation**: audio, EEG, fMRI (separating noise from brain sources).
- **Feature extraction** for natural images: ICA filters resemble Gabor wavelets — localized, oriented edge detectors similar to early visual cortex.
- **Preprocessing for non-Gaussian data**: decorrelation (PCA) is insufficient when you need *independence*.
- **Financial time series**: identify independent "shocks" driving correlated returns.

### 4.2 Conceptual Foundations

**Generative model.** $x = A s$, where $x \in \mathbb{R}^d$ is observed, $s \in \mathbb{R}^d$ is a vector of independent latent sources, $A$ is an unknown (but square and invertible) **mixing matrix**. Goal: recover $s$ (and $A$) from a sample $\{x_i\}$, up to inherent ambiguities.

**Ambiguities.** From $x = As$:
- **Scale**: if you double $s_j$ and halve column $j$ of $A$, you get the same $x$.
- **Permutation**: reordering sources is equivalent to permuting columns of $A$.
- **Sign**: flipping the sign of $s_j$ and column $j$ of $A$ is unobservable.

So ICA recovers sources up to scale, sign, and permutation. These are *inherent* ambiguities of blind separation.

**Why Gaussian sources are a non-starter.** If all $s_j$ are Gaussian, the joint $s$ is rotationally symmetric; mixing by $A$ just gives another Gaussian, and there's no "special" basis to recover. You can rotate $s$ by any orthogonal matrix without changing its distribution — *perfect* non-identifiability. ICA requires **at most one** source to be Gaussian.

**Assumptions.**
1. Sources are **statistically independent** ($p(s) = \prod_j p(s_j)$) — strictly stronger than uncorrelated.
2. **At most one source is Gaussian.**
3. **Square mixing** ($A$ invertible; same number of sources as sensors). Overcomplete / undercomplete cases need extensions.
4. **Instantaneous, linear mixing** (no convolution / delays). Convolutive ICA handles delays.
5. **Stationarity** across samples — mixing $A$ doesn't change.

**What breaks when assumptions fail.**
- Two Gaussian sources → one rotational degree of freedom not pinned down.
- Non-stationary mixing → $A$ drifts, separation degrades.
- Overcomplete ($d_s > d_x$) → need sparsity priors; hard.
- Dependent sources → recovered components are mixtures of "true" sources.

**Preprocessing: whitening.** Almost every ICA algorithm first whitens $x$ (so $\mathrm{Cov}(x) = I$). This reduces the problem from finding a general $A$ to finding a **rotation** — because for whitened $x$, the unmixing matrix must be orthogonal (since unmixed sources should have identity covariance, and rotating whitened data preserves that). Whitening turns ICA into a search over the smaller space of rotations.

### 4.3 Mathematical Formulation

**Goal.** Find $W = A^{-1}$ (the **unmixing matrix**) such that $y = Wx$ has independent components. We measure "how independent" in various ways:

#### 4.3.1 Non-Gaussianity measures

**Kurtosis** (4th cumulant):
$$
\kappa(y) = \mathbb{E}[y^4] - 3(\mathbb{E}[y^2])^2.
$$
Zero for Gaussian. Positive for heavy-tailed ("super-Gaussian," like Laplace, speech). Negative for light-tailed ("sub-Gaussian," like uniform).

Kurtosis is cheap to compute but highly sensitive to outliers (fourth moments). For a whitened unit-variance projection $y = w^\top x$, maximize $|\kappa(y)|$ or $\kappa(y)^2$ over directions $w$.

**Negentropy**:
$$
J(y) = H(y_{\mathrm{gauss}}) - H(y),
$$
where $y_{\mathrm{gauss}}$ is a Gaussian with the same variance as $y$, and $H$ is differential entropy. Gaussian maximizes entropy for fixed variance, so $J(y) \geq 0$ with equality iff Gaussian. Negentropy is a theoretically principled but computationally expensive measure. Approximations (Hyvärinen's G-functions) are used in practice:
$$
J(y) \approx (\mathbb{E}[G(y)] - \mathbb{E}[G(\nu)])^2,
$$
with $\nu \sim \mathcal{N}(0,1)$ and $G$ a non-quadratic, e.g., $G(u) = \log\cosh(u)$ or $G(u) = -\exp(-u^2/2)$.

#### 4.3.2 Mutual information minimization

For $y = Wx$,
$$
I(y_1, \dots, y_d) = \sum_j H(y_j) - H(y).
$$
Minimizing $I$ → sources independent. Since $H(y) = H(x) + \log|\det W|$, minimizing MI reduces to choosing $W$ to make each $y_j$ as non-Gaussian as possible (after whitening) — linking MI and negentropy.

#### 4.3.3 FastICA (Hyvärinen, 1999)

For a single component, find $w$ maximizing non-Gaussianity via fixed-point iteration:
$$
w \leftarrow \mathbb{E}[x \, g(w^\top x)] - \mathbb{E}[g'(w^\top x)]\, w,
$$
$$
w \leftarrow w / \|w\|,
$$
where $g$ is derivative of $G$ (e.g., $g(u) = \tanh(u)$). Convergence is cubic near optimum. For multiple components, Gram–Schmidt-like deflation keeps $w_j \perp w_{1:j-1}$ (valid since data is whitened).

**Derivation sketch.** Maximize $\mathbb{E}[G(w^\top x)]$ subject to $\|w\|=1$. KKT: $\mathbb{E}[x g(w^\top x)] - \beta w = 0$. Newton's method on this yields the FastICA update.

#### 4.3.4 Infomax (Bell–Sejnowski)

Maximize the entropy of a non-linear transformation of $y = Wx$: pass each $y_j$ through a sigmoid, then maximize output entropy. Gradient:
$$
\Delta W \propto (W^{-\top} + (1 - 2\sigma(Wx))x^\top),
$$
where $\sigma$ is the logistic. Equivalent to maximum likelihood under a specific non-Gaussian source prior.

### 4.4 Worked Example

Two independent sources, uniform on $[-1,1]$ (sub-Gaussian):
$$
s = \begin{pmatrix} s_1 \\ s_2 \end{pmatrix}, \quad \mathbb{E}[s] = 0, \; \mathrm{Cov}(s) = \tfrac{1}{3}I.
$$

Mix: $A = \begin{pmatrix} 1 & 2 \\ 3 & 1 \end{pmatrix}$. Observations $x = As$.

**Scatter plot intuition.** $s$ is uniform on a square; $x$ is uniform on a parallelogram (the square stretched and sheared by $A$). The sides of the parallelogram point along $Ae_1 = (1,3)$ and $Ae_2 = (2,1)$ — the source directions.

**Step 1: Whiten.** Compute $\mathrm{Cov}(x) = A \mathrm{Cov}(s) A^\top = \tfrac{1}{3} A A^\top$.
$A A^\top = \begin{pmatrix}5 & 5 \\ 5 & 10\end{pmatrix}$, so $\mathrm{Cov}(x) = \begin{pmatrix}5/3 & 5/3 \\ 5/3 & 10/3\end{pmatrix}$.

Whitening: $\tilde{x} = \mathrm{Cov}(x)^{-1/2}x$. Now $\tilde{x}$ has $\mathrm{Cov} = I$, and the parallelogram becomes a (rotated) square.

**Step 2: Find the rotation.** Parameterize $W = R(\theta)$ (rotation). Maximize $\sum_j |\kappa(w_j^\top \tilde{x})|$ over $\theta$. Uniform sources have $\kappa = -6/5 < 0$ (sub-Gaussian), distinct from the Gaussian-like $\kappa = 0$ of arbitrary linear combinations.

**Step 3: Recover.** Optimal $\theta$ aligns $w_1, w_2$ with the sides of the unit square in whitened space. Then $y = W\tilde{x}$ matches $s$ up to sign/permutation/scale.

**Concrete test.** In PCA of $x$, the top direction is the long axis of the parallelogram — not a source direction. ICA correctly finds the sides.

**EEG-style example (sketch).** 64 scalp EEG channels record a linear mixture of ~64 cortical sources plus artifacts (eye blinks, heartbeat). ICA unmixes: eye blinks end up as a single component with a stereotyped frontal topography and characteristic waveform. Analysts remove that component and project back for denoised EEG.

### 4.5 Relevance to ML Practice

**Where it's used:**
- **Biomedical signal processing**: EEG/MEG artifact removal, fMRI source separation.
- **Audio**: speech separation, music source separation (historically; supplanted by deep learning).
- **Image analysis**: ICA of natural image patches yields Gabor-like filters, connecting to V1 simple cells.
- **Feature extraction** when decorrelation (PCA) isn't enough.
- **Financial time series**: independent latent shocks.

**When not to use it:**
- Data is (approximately) Gaussian — ICA has nothing to lock onto.
- You need only decorrelation (PCA).
- Sources are not independent (e.g., videos of a scene with correlated objects).
- Convolutive / non-linear mixing — use convolutive ICA or deep separation models.
- Large-scale modern audio separation — supervised deep models (Conv-TasNet, Demucs) dominate.

**Alternatives:**
- PCA (if decorrelation suffices).
- Sparse coding / dictionary learning (similar goal, different formulation).
- Non-negative matrix factorization (for non-negative sources).
- Deep neural separation models (for supervised, large-data settings).

**Trade-offs:**
- **Interpretability**: ICA components are often interpretable (a specific source).
- **Computational cost**: FastICA is fast; Infomax uses gradient descent; both $O(nd^2)$ per iteration.
- **Identifiability**: up to scale/sign/permutation, unique if assumptions hold.
- **Robustness**: kurtosis-based ICA is outlier-sensitive; negentropy-based (log-cosh) more robust.

### 4.6 Common Pitfalls & Misconceptions

1. **Treating ICA as PCA.** They solve different problems. PCA decorrelates; ICA makes independent. Independence is stronger.
2. **Skipping whitening.** Most ICA algorithms assume whitened input; skipping → wrong results.
3. **Assuming sources can be Gaussian.** ICA fundamentally requires non-Gaussianity.
4. **Misinterpreting sign and scale.** The output $y_j$'s magnitude is arbitrary; only shape and timing of the source are recoverable.
5. **Overfitting when $d_s > d_x$ (overcomplete ICA).** Needs sparsity priors; plain ICA doesn't apply.
6. **Using kurtosis on heavy-tailed noisy data.** Tail events dominate. Switch to negentropy or robust measures.
7. **Running ICA on raw (non-centered, non-whitened) data.** Results are garbage. Always center and whiten.
8. **Trying to interpret "which component is which" across runs.** Permutation ambiguity means "component 3" in run A might be "component 7" in run B. Match components by correlation or template similarity.
9. **Applying ICA to time series without considering temporal structure.** Temporal ICA (using autocovariance) can exploit source time structure for identifiability, unlike spatial-only ICA.
10. **Too many components.** If $d$ is large and true source count is small, ICA finds spurious "sources" in noise.

### 4.7 Interview Questions — ICA

#### Foundational

**Q1. What's the core assumption of ICA?**
Sources are statistically independent and non-Gaussian (at most one can be Gaussian). Mixing is linear.

**Q2. PCA vs. ICA — one-liner.**
PCA decorrelates (uncorrelated); ICA separates (statistically independent). Independence implies uncorrelated but not vice versa for non-Gaussians.

**Q3. Why does non-Gaussianity matter?**
Because Gaussian RVs are rotationally symmetric — mixtures of independent Gaussians are indistinguishable from rotations of other mixtures. Non-Gaussianity breaks the symmetry and makes sources identifiable.

**Q4. What are the ambiguities of ICA?**
Scale, sign, and permutation of recovered sources.

**Q5. What's the "cocktail party problem"?**
Two speakers, two microphones; recover each voice from the mixtures alone, exploiting source independence.

#### Mathematical

**Q6. Why does ICA whiten first?**
Whitening standardizes $\mathrm{Cov}(x) = I$. The unmixing matrix then must be orthogonal (since $\mathrm{Cov}(y) = W \mathrm{Cov}(x) W^\top = I$ requires $WW^\top = I$). This reduces the search from general $d^2$-parameter $W$ to $d(d-1)/2$-parameter rotations.

**Q7. Derive why two Gaussian sources can't be separated.**
If $s_1, s_2 \sim \mathcal{N}(0, 1)$ independent, joint $\mathcal{N}(0, I)$. Joint density depends on $\|s\|^2$ only — rotationally symmetric. For any orthogonal $R$, $Rs$ has the same distribution. So observing $x = As$ is consistent with any $A' = AR$ — $A$ not identifiable.

**Q8. Define kurtosis and state its sign for sub/super-Gaussian.**
$\kappa = \mathbb{E}[y^4] - 3\mathrm{Var}(y)^2$ (for zero-mean). Zero → Gaussian. Positive → super-Gaussian (heavy tails, peaky, e.g. Laplace). Negative → sub-Gaussian (light tails, flat, e.g. uniform).

**Q9. What is negentropy and why is it a good contrast function?**
$J(y) = H(y_{\mathrm{gauss}}) - H(y) \geq 0$, zero iff Gaussian. Robust (unlike kurtosis) because it's an integrated measure; hard to estimate directly (entropy estimation is noisy), so approximations via $G$-functions are used.

**Q10. State the FastICA fixed-point update and why it converges cubically.**
$w \leftarrow \mathbb{E}[xg(w^\top x)] - \mathbb{E}[g'(w^\top x)]w$, then normalize. This is a Newton-method step on the constrained objective; for whitened data near the optimum, convergence is cubic.

#### Applied

**Q11. EEG data with eye-blink artifacts. How does ICA help?**
Decompose signal into components. Eye blinks have a stereotyped spatial map and waveform (spiky, periodic). Identify and zero out blink components, then recompose the signal without them.

**Q12. When would you prefer PCA over ICA?**
When you just need decorrelated features, reconstruction, or visualization; when data is approximately Gaussian (ICA adds no information); when compute budget is tight; when sources aren't truly independent.

**Q13. You have 10 sensors and suspect 3 sources. What do you do?**
Pre-reduce dimensionality with PCA to 3 components (discarding noise), then run ICA in the 3D subspace. This is standard; otherwise ICA fits spurious components to noise.

**Q14. Can ICA handle convolutive mixtures (e.g., room reverberation)?**
Plain ICA no; convolutive ICA yes — formulated in the frequency domain (where convolution becomes multiplication) but introduces permutation ambiguity across frequency bins, which must be resolved.

**Q15. Why does ICA of natural image patches yield Gabor-like filters?**
Natural images have sparse edge structure. Independent edge-like sources mixed linearly yield patches, so ICA "inverts" this and finds edge-like components — localized, oriented, bandpass — resembling Gabor wavelets and V1 simple cell receptive fields.

#### Debugging & Failure Modes

**Q16. ICA returns components that look like rotations of PCA axes. What's wrong?**
Data is approximately Gaussian, so ICA has no non-Gaussian structure to exploit; it ends up essentially equivalent to PCA (up to a rotation). Check marginal distributions — if roughly Gaussian, ICA isn't the right tool.

**Q17. Your ICA components differ wildly across random restarts. Why?**
Local optima. Different random initializations find different stationary points. Solutions: multiple restarts with consistency check, deflationary (one-at-a-time) with whitening, or use the Infomax objective with careful initialization.

**Q18. You apply ICA to fMRI; components look like brain maps but one is clearly motion artifact. What do you do?**
Remove the artifact component, inverse-transform. This is a standard ICA-based cleaning pipeline for fMRI.

#### Probing / Depth

**Q19. How is ICA related to maximum likelihood?**
ICA can be derived as ML estimation under independent non-Gaussian source priors. Specifically, Infomax is ML with a logistic prior. The log-likelihood reduces to $\sum_j \mathbb{E}[\log p_j(y_j)] + \log|\det W|$, and maximizing it → independence.

**Q20. How does ICA connect to information theory?**
Minimizing mutual information among $y_j = (Wx)_j$ maximizes independence. Since $I = \sum H(y_j) - H(y) = \sum H(y_j) - H(x) - \log|\det W|$, minimizing $I$ on whitened data equals minimizing $\sum H(y_j)$ — i.e., maximizing non-Gaussianity (since Gaussian maximizes entropy for fixed variance).

**Q21. What's the difference between spatial ICA and temporal ICA for time series?**
Spatial ICA treats each time sample as an observation, finding spatially independent components. Temporal ICA transposes: finds temporally independent components. For fMRI: spatial ICA gives brain maps; temporal ICA gives time courses. They're duals.

**Q22. Relationship between ICA and sparse coding?**
Very close. Sparse coding looks for a dictionary where each datapoint has a sparse representation. Sparsity → super-Gaussian distribution on coefficients → ICA with super-Gaussian prior. When the dictionary is complete and square, they coincide.

**Q23. How would you modify ICA for non-stationary mixing?**
Adaptive ICA or block-wise ICA (fit separate $A$'s in blocks). Better: state-space models or deep separation networks that explicitly model non-stationarity.

---

# Part II — Non-linear Methods

---

## 5. t-SNE (t-Distributed Stochastic Neighbor Embedding)

### 5.1 Motivation & Intuition

PCA finds global axes of variance but flattens curved structure. For instance, the MNIST digit manifold is highly non-linear — each digit class is a curved, low-dimensional surface in pixel space. PCA projection to 2D gives a mush.

t-SNE (van der Maaten & Hinton, 2008) takes a different approach: **preserve local neighborhood structure**. If two points are close in high-D, they should be close in 2D. Farness can be distorted. The result is visualizations with tight, well-separated clusters, where within-cluster similarity is preserved and cross-cluster distances are essentially meaningless (only ordinal "these are two different groups" is reliable).

The "t" in t-SNE refers to the Student's $t$-distribution with 1 degree of freedom used in the low-dim embedding — a key choice that addresses the "crowding problem" (see below).

Concrete ML motivations:
- **Data exploration and diagnostics**: visualizing embeddings (word embeddings, image features, cluster structure).
- **Sanity checks on representations**: learned features should cluster semantically.
- **Communicating findings**: a clear 2D plot is worth many numbers.

### 5.2 Conceptual Foundations

**Core idea.** Define two probability distributions over pairs of points:
- $p_{ij}$ in high-D: probability that point $i$ would pick point $j$ as its neighbor.
- $q_{ij}$ in low-D: same for the embedding.
Minimize the KL divergence $\mathrm{KL}(P \| Q)$ — forcing high-D neighbors to stay neighbors in low-D.

**The perplexity parameter.** For each point $i$, $p_{j|i}$ is a Gaussian centered on $i$ with bandwidth $\sigma_i$:
$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-\|x_i - x_k\|^2/2\sigma_i^2)}.
$$
The bandwidth $\sigma_i$ is chosen so that the **perplexity** — effective number of neighbors — matches a user-specified value (typically 5–50):
$$
\mathrm{Perp}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_j p_{j|i}\log_2 p_{j|i}.
$$
Small perplexity → tight, local neighborhoods → many small clusters shown. Large perplexity → broader neighborhoods → coarser global structure. Different perplexities reveal different scales; no single "right" value.

After computing $p_{j|i}$ per point, symmetrize: $p_{ij} = (p_{j|i} + p_{i|j})/(2n)$. This ensures every point has non-negligible contribution.

**Low-dim distribution.** In the embedding $\{y_i\} \subset \mathbb{R}^2$:
$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}}.
$$
The numerator is the kernel of a Student's $t$ with $\nu = 1$ (heavy-tailed). This is the signature t-SNE move.

**The crowding problem.** In 2D, a point has only $O(r^2)$ area within distance $r$, while in 100D it has $O(r^{100})$ volume. When projecting many moderately-separated points from high-D to 2D, there isn't enough "room" to place them all at correct distances; intermediate-distance points get crushed together. The heavy-tailed $t$ distribution lets low-D pairs sit at much larger distances than a Gaussian would allow, alleviating this crushing. Specifically, $q_{ij} \sim r^{-2}$ for large $r$, so moderate distances in high-D map to substantial distances in low-D without paying a large KL penalty.

**Assumptions.**
1. Local Euclidean neighborhoods are meaningful in high-D.
2. Perplexity is well-chosen for the scale of interest.
3. There's enough variation in low-D to place $n$ points.

**What breaks.**
- **Global structure is unreliable.** Cluster *sizes*, *inter-cluster distances*, and *density differences* in the t-SNE plot don't correspond to anything in the original data.
- **Stochastic**: different runs give different plots (random init, stochastic optimization).
- **Sensitive to perplexity**: different choices → different plots. Report multiple.
- **Scales poorly**: naive implementation $O(n^2)$; Barnes-Hut or FIt-SNE bring this to $O(n \log n)$ or $O(n)$.

### 5.3 Mathematical Formulation

**Objective:**
$$
C = \mathrm{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}.
$$

**Gradient:**
$$
\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}.
$$

**Interpretation as forces.** Pairs with $p_{ij} > q_{ij}$ (closer in high-D than low-D) attract. Pairs with $p_{ij} < q_{ij}$ repel. The $(1 + \|y_i - y_j\|^2)^{-1}$ factor weights short-range interactions more than long-range.

**Derivation of gradient.** Let $Z = \sum_{k \neq l}(1 + \|y_k - y_l\|^2)^{-1}$ and $w_{ij} = (1 + \|y_i - y_j\|^2)^{-1}$, so $q_{ij} = w_{ij}/Z$. Then
$$
C = -\sum p_{ij}\log q_{ij} + \mathrm{const} = -\sum p_{ij}\log w_{ij} + \log Z.
$$
$\partial_{y_i}\log w_{ij} = -2(y_i - y_j) w_{ij}$.
$\partial_{y_i}\log Z = (1/Z)\sum_j -2(y_i - y_j)w_{ij}^2 \cdot 2 = -4\sum_j (y_i - y_j)q_{ij}w_{ij}$ (factor of 2 for symmetry).
Combining (and using $\sum_j p_{ij}$ contributions) gives the quoted gradient.

**Early exaggeration.** Multiply $p_{ij}$ by a constant (e.g., 4 or 12) during the first few hundred iterations. This inflates attractive forces, letting points group into clusters before settling into local arrangements. Standard practice.

**Complexity.** Exact: $O(n^2)$ per iteration — prohibitive for $n > 10^4$. Barnes-Hut: approximates repulsive forces using a quadtree, $O(n \log n)$. FIt-SNE: interpolation-based, $O(n)$. These are the default modern implementations.

### 5.4 Worked Example

**Toy.** 6 points in 3D forming two clusters: $\{(0,0,0), (0.1, 0.05, -0.1), (-0.05, 0.1, 0.05)\}$ and $\{(5,5,5), (5.1, 4.9, 5.05), (4.95, 5.1, 4.98)\}$.

**Step 1: Compute high-D distances.** Within-cluster ~0.1–0.2; between-cluster ~8.6.

**Step 2: Set perplexity.** With 6 points, perplexity = 2 is reasonable. Binary-search $\sigma_i$ for each point.

**Step 3: Compute $p_{j|i}$.** For point 1, tight Gaussian around it gives high $p_{2|1}, p_{3|1}$ (within-cluster) and essentially zero for 4,5,6.

**Step 4: Initialize $y_i$ randomly in 2D.** (Better: initialize from PCA coordinates.)

**Step 5: Gradient descent.** Early exaggeration phase (say 250 steps) pushes cluster 1 points together, cluster 2 points together. Then normal phase fine-tunes.

**Step 6: Final embedding.** Two well-separated 2D blobs. Within-cluster distances are proportionally similar; the *gap* between clusters in 2D is much larger relative to within-cluster distance than in 3D — an artifact of the crowding-avoidance from the $t$ distribution. Don't interpret the gap as "8× farther" because it's not.

**Realistic example (MNIST).** 70k digits in $\mathbb{R}^{784}$. Apply PCA to 50D first (standard preprocessing). Run Barnes-Hut t-SNE with perplexity 30. Result: 10 well-separated clusters in 2D, each corresponding to a digit class, with some between-class confusion (e.g., 4s and 9s sometimes neighbor each other — reflecting their visual similarity).

### 5.5 Relevance to ML Practice

**Where it's used:**
- **Visualizing embeddings**: from models (CNN, Transformer feature spaces), from autoencoders, from text.
- **Cluster discovery and validation**: if clusters separate in t-SNE, likely separable in high-D.
- **Anomaly inspection**: outliers often show up as singletons or in odd positions.
- **Educational and exploratory tool.**

**When not to use it:**
- Downstream ML input — not suitable, t-SNE distances aren't meaningful globally.
- Clustering via t-SNE + k-means — biased, don't do it. Cluster in high-D.
- Comparing datasets or runs — t-SNE is stochastic and scale-dependent.
- Very large datasets ($>10^6$) without FIt-SNE or GPU — slow.
- When global geometry matters — use UMAP or PCA.
- When you need an inductive (out-of-sample) map — t-SNE doesn't give a function; you'd have to re-run with new points included. There are "parametric t-SNE" variants that use NNs, but not standard.

**Alternatives:**
- UMAP: similar, often faster, claims to preserve global structure better.
- PCA: linear, fast, preserves global variance.
- LargeVis: a close cousin to UMAP.
- PHATE, TriMap: other modern alternatives.

**Trade-offs:**
- **Interpretability**: clusters yes, distances/shapes no.
- **Computational cost**: $O(n \log n)$ modern; memory $O(n)$.
- **Stability**: low (stochastic; run multiple times).
- **Global vs. local**: heavily local.

### 5.6 Common Pitfalls & Misconceptions

1. **Reading cluster sizes or distances literally.** They aren't meaningful. Only topology (who's near whom) is.
2. **Not running multiple seeds.** Different initializations give different plots; make sure conclusions are robust.
3. **Using a single perplexity.** Try multiple (5, 30, 100) to see structure at different scales.
4. **Feeding raw high-D data.** Pre-reduce with PCA (to 30–50D) to denoise and speed things up.
5. **Using t-SNE output as ML features.** Don't. The embedding is for visualization.
6. **Clustering on t-SNE coordinates.** Biased by the embedding's artifacts. Cluster in original / PCA space.
7. **Naïve implementations on big data.** $O(n^2)$ is death; use Barnes-Hut or FIt-SNE.
8. **Expecting reproducibility across libraries.** Implementations differ in defaults.
9. **Misinterpreting early-stopping shapes.** During optimization, t-SNE goes through stages (pre-clustering); only the final settled plot is meaningful.
10. **Expecting out-of-sample generalization.** New points require re-fitting (or use parametric variants / UMAP).

### 5.7 Interview Questions — t-SNE

#### Foundational

**Q1. What does t-SNE preserve?**
Local neighborhood structure — points that are close in high-D are close in low-D. Global distances are not preserved.

**Q2. What is perplexity?**
Effective number of neighbors per point, controlling the local Gaussian bandwidth. Typical range 5–50.

**Q3. Why is the Student's $t$ distribution used in low-D?**
To address the crowding problem — its heavy tails let moderately distant high-D pairs sit at substantially larger low-D distances without paying a big KL penalty, preventing the embedding from collapsing.

**Q4. Is t-SNE supervised?**
No — it ignores labels. You can color points by label post-hoc.

**Q5. Can I use t-SNE output as features for a classifier?**
No. The embedding is for visualization; it's stochastic, non-invertible, and doesn't preserve global structure.

#### Mathematical

**Q6. Write the t-SNE objective.**
$\mathrm{KL}(P\|Q) = \sum_{i\neq j}p_{ij}\log(p_{ij}/q_{ij})$, where $p_{ij}$ is Gaussian-based high-D neighborhood probability (with perplexity-tuned bandwidth) and $q_{ij}$ is $t$-distribution-based in low-D.

**Q7. Derive the t-SNE gradient and interpret as physical forces.**
$\partial C/\partial y_i = 4\sum_j (p_{ij} - q_{ij})(y_i - y_j)(1+\|y_i - y_j\|^2)^{-1}$. Attractive if high-D neighbors aren't close enough; repulsive if low-D pairs too close; weighted by a short-range kernel.

**Q8. Why is t-SNE $O(n^2)$ naively? How to speed up?**
Each pair $(i,j)$ contributes a repulsive term requiring a sum over all pairs. Barnes-Hut trees approximate far-away contributions in $O(\log n)$ per point; FIt-SNE interpolates on a grid.

**Q9. Why symmetric $p_{ij}$?**
Ensures every point contributes to the cost meaningfully — conditional $p_{j|i}$ can be very small for outliers, making $i$'s contribution vanish. Symmetrizing avoids that.

**Q10. What does "early exaggeration" do mathematically?**
Multiplies $p_{ij}$ by a constant $\alpha > 1$ (say 12). Amplifies attractive forces relative to repulsive ones, letting points form tight clusters first.

#### Applied

**Q11. Your t-SNE shows two clusters very far apart. Is that meaningful?**
Their separation is meaningful; the *distance* between them isn't. t-SNE tends to exaggerate inter-cluster gaps.

**Q12. MNIST with 50k samples — default perplexity?**
30 is standard. Pre-reduce with PCA to ~50D first. Use Barnes-Hut or FIt-SNE.

**Q13. How do you evaluate t-SNE quality?**
Neighborhood preservation metrics: trustworthiness, continuity (measure how many high-D neighbors stay neighbors in low-D). Visual inspection vs. known labels. Robustness across perplexities and seeds.

**Q14. Client asks: "Run t-SNE monthly and track how clusters move." What's wrong?**
Stochastic initialization + different data → different embeddings. Cluster "movement" is often just re-randomization. Use the same seed, same data (with new points added), and compare via Procrustes — or use UMAP, which is more stable.

**Q15. Use t-SNE for downstream clustering? What to do instead?**
Cluster in the original or PCA space. Use t-SNE only for visualization of clusters found elsewhere.

#### Debugging & Failure Modes

**Q16. Your t-SNE plot looks like a uniform "ring" or "starfish." Why?**
Likely converged too early, or perplexity too low, or data is nearly isotropic. Try more iterations, larger perplexity, or check the data for meaningful structure.

**Q17. Clusters in t-SNE don't correspond to class labels. Diagnosis?**
Maybe labels aren't recoverable from features (features uninformative); or t-SNE is emphasizing a different structure (e.g., acquisition site rather than class); or perplexity is mis-tuned. Try various perplexities and color by multiple metadata.

**Q18. t-SNE takes days to run on 1M points. Fix?**
Use FIt-SNE (GPU or CPU) — $O(n)$ scaling. Or subsample. Or switch to UMAP.

#### Probing / Depth

**Q19. Deep reason t-SNE doesn't preserve global structure?**
The KL is asymmetric: it heavily penalizes $p_{ij} > 0$ paired with small $q_{ij}$ (missed neighbors) but lightly penalizes $p_{ij} \approx 0$ paired with moderate $q_{ij}$ (false neighbors). So distant pairs are free to be placed almost anywhere — only neighbors are tightly constrained.

**Q20. How would you make t-SNE parametric (for out-of-sample)?**
Parametric t-SNE (van der Maaten 2009): fit a neural network $y = f_\theta(x)$ minimizing the same KL. Trainable end-to-end; new points embed via a forward pass.

**Q21. t-SNE vs. UMAP — what differs?**
Objectives differ: UMAP uses a cross-entropy over fuzzy-set memberships rather than KL. UMAP claims better global structure via its construction and is often faster. In practice, results are often similar.

**Q22. Can t-SNE be used for supervised dim reduction?**
Not natively, but you can modify distances to incorporate labels (supervised t-SNE variants) or use UMAP's supervised mode. Straight t-SNE ignores labels.

**Q23. Why is t-SNE's cost non-convex and does it matter?**
The $q_{ij}$ depends on low-D positions, making the landscape non-convex; different seeds → different local minima. Matters when comparing runs; use Procrustes or consistent init (e.g., PCA seed).

---

## 6. UMAP (Uniform Manifold Approximation and Projection)

### 6.1 Motivation & Intuition

t-SNE is slow at scale and arguably distorts global structure. UMAP (McInnes, Healy, Melville, 2018) is a newer alternative with three practical wins:

1. **Faster**: scales to millions of points on a laptop (≈$O(n^{1.14})$ empirically with the default approximate-NN backbone).
2. **Claims better global structure preservation** — inter-cluster distances are more meaningful than in t-SNE.
3. **Supports an inductive map**: can embed new points into an existing embedding.

Theoretically, UMAP is derived from ideas in algebraic topology (Riemannian geometry + simplicial sets), but operationally it behaves like "a faster, slightly different t-SNE." For a practitioner, the main differences are speed, slightly different hyperparameters, and somewhat different aesthetic (often cleaner separations).

Concrete ML motivations: Same as t-SNE — exploring embeddings, clustering visualization — but now at the scale of modern datasets (1M–10M points tractable).

### 6.2 Conceptual Foundations

UMAP's construction has two phases:
1. **Build a fuzzy topological representation in high-D.**
2. **Optimize a low-D embedding whose fuzzy topology matches.**

**Phase 1 — High-D graph.**
- For each point $x_i$, find its $k$ nearest neighbors.
- Each neighbor $x_j$ is connected by an edge with a weight representing membership in $x_i$'s local neighborhood:
$$
w_{j|i} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right),
$$
where $\rho_i$ is the distance to the nearest neighbor (making the nearest always have weight 1) and $\sigma_i$ is chosen so that $\sum_j w_{j|i} = \log_2 k$ — analogous to perplexity.
- Symmetrize via fuzzy-set union: $w_{ij} = w_{j|i} + w_{i|j} - w_{j|i} w_{i|j}$ (the "probabilistic t-conorm").

This produces a weighted directed graph on the data, treated as a fuzzy simplicial set.

**Phase 2 — Low-D optimization.**
- Place points $y_i$ in $\mathbb{R}^2$ (or $\mathbb{R}^k$). The low-D neighborhood membership between $y_i, y_j$ is
$$
v_{ij} = (1 + a\|y_i - y_j\|^{2b})^{-1},
$$
where $a, b$ are fit from user-specified `min_dist` and `spread` parameters. This plays the role of t-SNE's $t$-kernel but with tunable tails.
- Minimize cross-entropy:
$$
C = \sum_{i\neq j}\left[w_{ij}\log \frac{w_{ij}}{v_{ij}} + (1 - w_{ij})\log \frac{1 - w_{ij}}{1 - v_{ij}}\right].
$$
This is the fuzzy-set cross-entropy — a key difference from t-SNE, which uses KL.

**Optimization is stochastic gradient descent** over edges (attractive forces) and negative samples (repulsive forces), à la word2vec. This is why UMAP is so fast: it never sums over all $O(n^2)$ pairs.

**Hyperparameters.**
- `n_neighbors` ($k$, default 15): controls local-vs-global scale. Small → fine local structure; large → more global view.
- `min_dist` (default 0.1): how tight clusters are allowed to get.
- `metric`: distance to use (Euclidean, cosine, etc.).
- `n_components`: output dimensionality (usually 2 or 3; can go higher).

**Assumptions (the topological ones).**
1. Data lies on a manifold that is locally Euclidean.
2. The manifold is locally connected (no singularities).
3. A common Riemannian metric exists on this manifold. UMAP circumvents this by using per-point local metrics (via $\rho_i, \sigma_i$).

**What breaks.**
- For very few points per neighborhood, local metric estimation is noisy.
- If the true manifold has boundaries / holes / very different scales, fixed `n_neighbors` may misrepresent them.
- Very high-D inputs without pre-reduction suffer from concentration of distances (all distances ~equal).

### 6.3 Mathematical Formulation

**High-D weights.** $\rho_i = \min_{j \neq i} d(x_i, x_j)$ (distance to nearest neighbor). $\sigma_i$ solves $\sum_{j=1}^k \exp(-\max(0, d(x_i, x_j) - \rho_i)/\sigma_i) = \log_2 k$ by binary search.

Directed weight: $w_{j|i} = \exp(-\max(0, d_{ij} - \rho_i)/\sigma_i)$.
Symmetric: $w_{ij} = w_{j|i} + w_{i|j} - w_{j|i} w_{i|j}$.

**Low-D kernel.** $v_{ij} = (1 + a \|y_i - y_j\|^{2b})^{-1}$. Curve $a, b$ fitted so that the kernel equals 1 for distances below `min_dist` and decays smoothly beyond.

**Cross-entropy loss:**
$$
C = \sum_{i<j} w_{ij}\log \frac{w_{ij}}{v_{ij}} + (1-w_{ij})\log\frac{1-w_{ij}}{1-v_{ij}}.
$$
The first term (attractive) says: when $w_{ij}$ large, make $v_{ij}$ large (bring points together). The second term (repulsive) says: when $w_{ij}$ small, make $v_{ij}$ small (push apart). Unlike t-SNE, UMAP has explicit repulsive terms, which improves global structure preservation.

**Gradient (simplified).** For an edge $(i,j)$ with weight $w_{ij}$, gradient on $y_i$:
- Attractive: $-2ab\|y_i - y_j\|^{2b-2} / (1 + a\|y_i - y_j\|^{2b}) \cdot w_{ij}(y_i - y_j)$.
- Repulsive (from negative samples, per unconnected pair): $+ 2b/((\epsilon + \|y_i - y_k\|^2)(1 + a\|y_i - y_k\|^{2b})) \cdot (y_i - y_k)$.

SGD over edges + $n_{\mathrm{neg}}$ negative samples per update makes total cost $\propto n \cdot \mathrm{epochs} \cdot (1 + n_{\mathrm{neg}})$.

### 6.4 Worked Example

**Toy.** 500 samples from two interleaved half-circles (noisy moons), in 2D but embedded in 50D via random padding (add 48 noise dimensions).

**Step 1: Preprocessing.** PCA to 10D (denoise) or feed directly (UMAP handles high-D).

**Step 2: KNN graph.** For each point, find 15 nearest neighbors.

**Step 3: Local $\sigma_i$.** Binary-search so sum of edge weights ≈ $\log_2 15 \approx 3.9$.

**Step 4: Symmetrize.** Fuzzy union of directed edges.

**Step 5: Init in 2D.** Spectral embedding of the graph (Laplacian eigenmaps) — a good init that captures global structure.

**Step 6: SGD.** Sample positive edges (pull together) and negative samples (push apart). Run ~200 epochs.

**Result.** Two well-separated moons, with the curvature of each preserved — much closer to the true 2D structure than t-SNE would give on the same data. The noise dimensions are effectively ignored.

**Realistic example (scRNA-seq).** 50k cells × 20k genes. Pre-reduce to top 50 PCs. UMAP with `n_neighbors=30`, `min_dist=0.3`. Result: clusters corresponding to cell types, with continuous trajectories where cell types are developmentally related — UMAP often preserves these continua where t-SNE fragments them.

### 6.5 Relevance to ML Practice

**Where it's used:**
- **Single-cell genomics** (UMAP is now the default; replaced t-SNE in most pipelines).
- **Embedding exploration** at scale.
- **Preprocessing** for downstream clustering (more defensible than t-SNE + clustering, but still controversial).
- **Label propagation**: supervised UMAP uses labels to shape the embedding, useful for semi-supervised visualization.
- **Inductive transformation**: new points can be embedded into an existing UMAP without re-fitting.

**When not to use it:**
- When you need interpretable linear features — use PCA.
- When truly local details are critical and you can afford it — t-SNE sometimes gives tighter local clusters.
- When reproducibility across runs is paramount — UMAP is stochastic too (though more stable than t-SNE).

**Alternatives:**
- t-SNE, PCA, Autoencoders, Isomap, LLE, PHATE, TriMap.

**Trade-offs:**
- **Speed**: big win over t-SNE.
- **Global structure**: better than t-SNE but still not trustworthy for precise distances.
- **Hyperparameter sensitivity**: moderate; defaults often OK but explore `n_neighbors`.
- **Inductive**: yes, small advantage for production pipelines.

### 6.6 Common Pitfalls & Misconceptions

1. **Believing UMAP preserves global distances.** It preserves global *topology* better than t-SNE, but absolute distances still aren't meaningful.
2. **Running on raw very-high-D.** Pre-reduce with PCA when $d > 100$ for speed and denoising.
3. **Using UMAP output for clustering uncritically.** Same caveat as t-SNE; the embedding can introduce artifacts.
4. **Not exploring `n_neighbors`.** Just like perplexity for t-SNE, this parameter determines scale.
5. **Treating supervised UMAP as a predictive model.** It shapes the embedding to align with labels but doesn't predict labels on new points reliably.
6. **Expecting deterministic results.** Set random seed, understand it's SGD — same seed gives reproducibility.
7. **Assuming UMAP "just works" on non-Euclidean data.** You often need custom metrics.
8. **Ignoring the `min_dist` parameter.** Small values give very tight clusters that can look misleadingly clean.

### 6.7 Interview Questions — UMAP

#### Foundational

**Q1. How does UMAP differ from t-SNE at a high level?**
UMAP uses a fuzzy simplicial-set construction and cross-entropy loss over edge memberships; t-SNE uses Gaussian high-D + $t$ low-D probability distributions with KL divergence. UMAP is faster (via SGD with negative sampling) and typically preserves global structure better.

**Q2. What does `n_neighbors` control?**
The size of the local neighborhood used to build the topological representation. Small → local detail; large → global context.

**Q3. What does `min_dist` control?**
How tightly points can cluster in the low-D embedding. Smaller → denser, tighter clusters.

**Q4. Is UMAP an inductive or transductive algorithm?**
Inductive — the fit produces a model that can embed new points. (Under the hood, via interpolation in the neighbor graph and low-D coordinates.)

**Q5. Is UMAP deterministic?**
No — it uses stochastic gradient descent. But with a fixed seed, reproducible.

#### Mathematical

**Q6. Write UMAP's loss function.**
Fuzzy cross-entropy:
$C = \sum_{i<j}[w_{ij}\log(w_{ij}/v_{ij}) + (1-w_{ij})\log((1-w_{ij})/(1-v_{ij}))]$,
where $w_{ij}$ is the symmetrized high-D edge weight and $v_{ij}$ the low-D similarity.

**Q7. Why fuzzy union for symmetrization rather than averaging?**
Fuzzy union $w_{ij} = w_{j|i} + w_{i|j} - w_{j|i}w_{i|j}$ treats directed weights as set memberships — "either is close to the other" is stronger than average, and preserves the topological interpretation (fuzzy simplicial set structure).

**Q8. How does UMAP scale with $n$?**
Approximately $O(n \log n)$ for the KNN step (e.g., NN-Descent, HNSW), plus $O(n \cdot \mathrm{epochs} \cdot (1 + n_{\mathrm{neg}}))$ for SGD. Empirically near-linear on real datasets.

**Q9. How are $a, b$ in the low-D kernel determined?**
Fit via non-linear least squares so the curve $(1 + a r^{2b})^{-1}$ behaves like a smoothed step: ≈1 for $r < \mathrm{min\_dist}$, decaying beyond.

**Q10. Why does UMAP use negative sampling?**
Sums over all pairs are $O(n^2)$. Randomly sampling a handful of negatives per positive edge gives unbiased gradients in expectation at $O(n \cdot \mathrm{epochs})$ cost.

#### Applied

**Q11. 1M points, 100 features. UMAP or t-SNE?**
UMAP. t-SNE at 1M is a pain even with FIt-SNE; UMAP is comfortable on a laptop.

**Q12. Single-cell RNA-seq analysis — typical pipeline?**
QC → log-transform → PCA (30–50 components) → UMAP with `n_neighbors=15–30`, `min_dist=0.1–0.5`. Color by cluster labels or gene expression.

**Q13. Using UMAP for out-of-sample embedding — any caveats?**
The new point is placed based on its KNN among training points. If the new point is far from all training data (novelty), its placement is unreliable.

**Q14. Supervised UMAP — use case?**
When labels exist and you want the embedding to respect them for visualization. Don't use it as a classifier.

**Q15. Should downstream clustering be on UMAP output?**
Same caveat as t-SNE: embedding artifacts can affect cluster boundaries. Safer to cluster in PCA / original space and color UMAP by cluster.

#### Debugging & Failure Modes

**Q16. UMAP result changes drastically with small `n_neighbors` change. What's happening?**
Data has fine local structure vs. coarser global structure; switching scale reveals different facets. Not a bug. Report multiple settings.

**Q17. Clusters in UMAP that don't exist in the data. How to verify?**
Check if clusters are stable across seeds and hyperparameters. Test via supervised metrics (if labels available). Consider: are you over-clustering because `min_dist` is too small?

**Q18. UMAP is slow on my 10M point dataset. What do you do?**
Reduce to PCA-50D first. Reduce `n_epochs` (default 200 often too many). Use GPU UMAP (cuML). Downsample if feasible.

#### Probing / Depth

**Q19. Does UMAP preserve density?**
Not by default. `densMAP` is an extension that preserves local density information.

**Q20. UMAP vs. Isomap — conceptual relationship?**
Isomap uses global geodesic distances (shortest paths in the KNN graph) and MDS. UMAP uses only local neighborhoods + SGD on a fuzzy-set cross-entropy. UMAP is faster and more flexible; Isomap is more interpretable / classical.

**Q21. Is the "preserves global structure" claim fully justified?**
Empirically UMAP often does better than t-SNE at it, but it's not a rigorous guarantee. UMAP's explicit repulsive terms and spectral initialization help.

**Q22. Can UMAP output be trusted for cluster counting?**
Not purely. Combine with domain knowledge and a clustering algorithm in the original space.

**Q23. How does initialization affect UMAP?**
Significantly. Default is spectral embedding of the graph (captures global structure). Random init → more local-quality focus, worse global.

---

## 7. Autoencoders

### 7.1 Motivation & Intuition

Autoencoders (AEs) are neural networks trained to reconstruct their input. An **encoder** $f: \mathbb{R}^d \to \mathbb{R}^k$ compresses $x$ to a latent code $z$; a **decoder** $g: \mathbb{R}^k \to \mathbb{R}^d$ reconstructs $\hat{x} = g(z)$. Training minimizes reconstruction error $\|x - g(f(x))\|^2$.

If $k < d$ and both $f, g$ are linear, the optimum recovers PCA — the encoder projects onto the top $k$ PCs. Making $f, g$ non-linear (MLPs, CNNs, Transformers) gives a non-linear generalization of PCA, capable of learning any non-linear manifold representation.

Concrete ML motivations:
- **Learned compression** (images, sensor data) going beyond PCA.
- **Denoising**: train to reconstruct clean input from noisy input — learns a manifold projection.
- **Pre-training**: unsupervised feature learning before a supervised task (classical before self-supervised).
- **Anomaly detection**: large reconstruction error flags outliers.
- **Generative modeling**: VAEs, which are autoencoders with a probabilistic twist.

### 7.2 Conceptual Foundations

**Key terms.**
- **Encoder**: $z = f_\phi(x)$. Typically MLP or CNN.
- **Decoder**: $\hat{x} = g_\theta(z)$. Mirror of encoder in simple AEs.
- **Bottleneck**: the latent $z$, deliberately lower-dimensional than $x$ (if undercomplete) or regularized otherwise.
- **Reconstruction loss**: $\|x - \hat{x}\|^2$ for continuous data; cross-entropy for binary/categorical; mixed losses for mixed data.

**Architectural variants.**

- **Undercomplete AE.** $\dim(z) < \dim(x)$. The bottleneck forces compression. Similar in spirit to PCA; non-linear activations give non-linear PCA.

- **Overcomplete AE.** $\dim(z) \geq \dim(x)$. Without regularization, can learn identity (trivial). Must be regularized.

- **Sparse AE.** Penalty encourages most latent units to be near zero per input (e.g., $\ell_1$ penalty on $z$, or KL to a low Bernoulli mean). Gives interpretable, part-based features.

- **Denoising AE (DAE).** Input is corrupted ($\tilde{x} = x + \mathrm{noise}$); target is clean $x$. Forces the model to learn the data manifold — reconstructing requires projecting noisy points back onto the manifold.

- **Contractive AE (CAE).** Penalty $\|\partial f/\partial x\|_F^2$ — makes the latent code locally insensitive to input perturbations. Related to DAE; explicitly encourages robustness.

- **Variational AE (VAE).** Probabilistic: encoder outputs $q(z|x) = \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma^2_\phi(x)))$, latent prior $p(z) = \mathcal{N}(0, I)$, decoder $p(x|z)$. Trained to maximize ELBO. Gives a generative model.

- **Convolutional AE.** For images: encoder = conv + pool layers; decoder = transposed conv / upsample + conv.

- **Sequence AE.** RNN / Transformer encoder-decoder for text/time series.

**Assumptions.**
1. Reconstruction loss captures what's important — e.g., MSE assumes Gaussian noise; cross-entropy assumes Bernoulli.
2. Bottleneck / regularization is sufficient to prevent trivial identity.
3. Data lies on a manifold learnable by the chosen architecture.
4. Training data is representative.

**What breaks.**
- Too-large bottleneck without regularization → identity mapping, no learning.
- Mode collapse: AE learns to reconstruct only the "average" pattern, blurring details.
- Out-of-distribution: reconstruction of OOD points may succeed (undesired for anomaly detection) if decoder generalizes too well.
- Latent space structure: plain AE latents aren't structured — $z$'s for similar $x$'s may not be close; interpolations can be nonsense. VAEs address this.

### 7.3 Mathematical Formulation

**Basic AE.** $\min_{\phi, \theta} \sum_i \|x_i - g_\theta(f_\phi(x_i))\|^2$.

**Linear AE = PCA.** If $f(x) = Wx$ and $g(z) = W'z$ with MSE loss, the optimum at rank $k$ satisfies: $W'W$ projects onto the top-$k$ PC subspace. The AE doesn't uniquely produce the PCs themselves (any basis of the PC subspace works — orthogonal $U$ is ambiguous).

**Denoising AE.** $\min \mathbb{E}_{x, \tilde{x}}[\|x - g(f(\tilde{x}))\|^2]$. Score-matching connection: Vincent (2011) showed DAEs with small Gaussian noise learn $\nabla_x \log p(x)$ — the score function. This is the foundation of modern diffusion models.

**Sparse AE.** $\min \|x - \hat{x}\|^2 + \lambda \|z\|_1$ (or KL-to-Bernoulli penalty).

**Contractive AE.** $\min \|x - \hat{x}\|^2 + \lambda \|\partial f/\partial x\|_F^2$. Encourages invariance to small input changes.

**Variational AE.**
$$
\mathcal{L}(\phi, \theta; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x)\|p(z)).
$$
First term: reconstruction (under Gaussian $p(x|z)$ this reduces to squared error). Second term: regularizer toward prior. Optimized via the reparameterization trick: sample $\epsilon \sim \mathcal{N}(0, I)$, set $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ — enabling gradient flow through stochastic sampling. The KL for Gaussians is closed form:
$$
\mathrm{KL} = \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1).
$$

**ELBO derivation:** Start from $\log p(x) = \log \int p(x,z)\,dz$. Multiply and divide by $q(z|x)$:
$$
\log p(x) = \log \mathbb{E}_q\left[\frac{p(x,z)}{q(z|x)}\right] \geq \mathbb{E}_q[\log p(x,z) - \log q(z|x)] = \mathbb{E}_q[\log p(x|z)] - \mathrm{KL}(q\|p).
$$
By Jensen. The gap to $\log p(x)$ is $\mathrm{KL}(q_\phi(z|x)\|p(z|x))$ — closed when $q$ matches the true posterior.

### 7.4 Worked Example

**MNIST autoencoder.**

Architecture: Encoder $784 \to 256 \to 64 \to 2$ (with ReLU); decoder mirror.

Training: Adam, MSE loss, batch 128, 20 epochs.

After training:
- **Reconstructions**: recognizable digits, slightly blurred. MSE loss tends to blur; cross-entropy (treating pixels as Bernoulli) often sharper.
- **Latent space**: plot 2D latent codes for the training set colored by digit class. Classes largely separate; 4/9 overlap (similar shapes).
- **Interpolation**: pick two points $z_1, z_2$, decode $g(tz_1 + (1-t)z_2)$ for $t \in [0, 1]$. Smooth morph between digits — a sign the decoder has learned a reasonable manifold.

**VAE variant.** Same architecture with two output heads in encoder ($\mu, \log \sigma^2$) and KL regularization. The latent becomes smoother and more "filled in" — random samples from the prior decode to plausible digits. Plain AE samples from random $z$ often don't decode meaningfully (gaps in the latent space).

**Toy numeric.** 1D AE on a sine-curve manifold in 2D. Inputs: $(t, \sin t)$ for $t$ uniform. An AE with a 1-dim bottleneck and enough capacity learns $z \approx t$ (or a monotone transform thereof) and reconstructs accurately. PCA on this would fail: a line through the data captures little of the curve, and reconstructions lie on a straight line rather than the sine.

### 7.5 Relevance to ML Practice

**Where they're used:**
- **Denoising** and inpainting (older, pre-diffusion).
- **Anomaly detection**: high reconstruction error on OOD inputs.
- **Pre-training** for downstream classification — though supplanted by masked-language-model-style self-supervision (BERT, MAE).
- **Generative models**: VAEs directly; AEs as components in GAN/diffusion pipelines.
- **Representation learning** when labels are scarce.
- **Image/audio/video compression** (learned codecs: compressive AEs, Ballé et al.).
- **Sequence modeling** (seq2seq language models are encoder-decoder AEs).

**When not to use:**
- When PCA does the job — don't over-engineer.
- Tiny datasets — AEs overfit; use PCA or classical methods.
- When you need a probabilistic model — use VAE (or diffusion).
- When you need exact invertibility — use flow-based models.
- When the task is supervised and data is plentiful — end-to-end supervised learning outperforms.

**Alternatives:**
- PCA / kernel PCA (classical, linear/kernelized).
- t-SNE / UMAP (for visualization, not reconstruction).
- VAEs (probabilistic generative).
- GANs, diffusion models (generative, typically better samples).
- Flow-based models (invertible, exact likelihood).
- Self-supervised learning (masked AE, contrastive methods).

**Trade-offs.**
- **Flexibility**: very high — handles non-linearity, multi-modality, structured input.
- **Interpretability**: low — latent dimensions don't correspond to semantic concepts without extra tricks.
- **Computational cost**: high — neural training.
- **Need for data**: large.
- **Robustness**: training can be unstable, sensitive to hyperparameters.

### 7.6 Common Pitfalls & Misconceptions

1. **Overcomplete AE without regularization.** Learns identity; no useful representation.
2. **Using MSE on images uncritically.** Produces blurry reconstructions; perceptual or adversarial losses often better.
3. **Confusing AE and VAE latent spaces.** VAE's is structured (prior regularized); plain AE's isn't — random latents may decode to garbage.
4. **Anomaly detection assuming OOD → high reconstruction error.** Can fail if the decoder generalizes too well; sometimes OOD inputs reconstruct *better* than in-distribution (pathological but documented).
5. **Treating bottleneck size as "number of latent concepts."** It's just capacity; real concepts may span many or few dimensions.
6. **No validation.** Training loss can drop while generalization fails; use held-out reconstruction or downstream task metrics.
7. **Using AE output for distance comparisons.** AEs don't guarantee preserved distances unless trained to (via metric learning).
8. **Forgetting to normalize inputs.** NNs are scale-sensitive; normalization matters.
9. **Assuming VAE samples are high-quality.** VAEs often blur; GANs/diffusion generate sharper images.
10. **Confusing "reconstruction" with "understanding."** An AE can reconstruct inputs without learning anything semantically meaningful — this is well-known with overcomplete, unregularized models.

### 7.7 Interview Questions — Autoencoders

#### Foundational

**Q1. What does an autoencoder learn?**
A compressed representation of inputs through an encoder-decoder bottleneck that minimizes reconstruction error.

**Q2. What's an undercomplete AE?**
One where the latent dimensionality is less than the input — forces compression.

**Q3. What's a denoising AE?**
Trained to reconstruct clean inputs from corrupted versions; learns the data manifold.

**Q4. What's the relationship between a linear AE and PCA?**
A linear undercomplete AE with MSE loss recovers the PCA subspace (possibly rotated; PCs themselves aren't unique).

**Q5. Why would you regularize an autoencoder?**
To prevent trivial identity mapping (especially when overcomplete), encourage useful latent structure (sparsity, contractivity), or enforce a prior (VAE).

#### Mathematical

**Q6. Why is a linear AE equivalent to PCA?**
Minimizing $\|X - XW^\top W\|^2$ (with $W \in \mathbb{R}^{k \times d}$ orthonormal rows) is minimized when rows of $W$ span the top-$k$ PC subspace. Proof: Eckart-Young theorem — best rank-$k$ approximation is the truncated SVD.

**Q7. Derive the VAE ELBO.**
$\log p(x) = \log \int p(x,z)dz = \log \mathbb{E}_{q(z|x)}[p(x,z)/q(z|x)] \geq \mathbb{E}_q[\log p(x|z)] + \mathbb{E}_q[\log p(z)] - \mathbb{E}_q[\log q(z|x)] = \mathbb{E}_q[\log p(x|z)] - \mathrm{KL}(q\|p)$ by Jensen. Gap is $\mathrm{KL}(q(z|x)\|p(z|x))$.

**Q8. What's the reparameterization trick and why is it needed?**
To backpropagate through a sampling step. Instead of $z \sim q_\phi(z|x)$, sample $\epsilon \sim p(\epsilon)$ (fixed, parameter-free) and compute $z = g_\phi(x, \epsilon)$. Gradients flow through $g$. For Gaussian $q$: $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

**Q9. Write the VAE loss in closed form for Gaussian prior and Gaussian posterior.**
$\mathcal{L} = \|x - \hat{x}\|^2 + \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$. First term under Gaussian $p(x|z)$ with fixed variance; second is $\mathrm{KL}(\mathcal{N}(\mu, \mathrm{diag}\sigma^2)\|\mathcal{N}(0, I))$.

**Q10. What does a contractive AE penalize?**
Jacobian Frobenius norm $\|\partial f(x)/\partial x\|_F^2$ — encourages invariance of latent to small input perturbations.

#### Applied

**Q11. Anomaly detection with AEs — how?**
Train on normal data. At test, large reconstruction error signals anomaly. Threshold on reconstruction error or score function.

**Q12. Your AE produces blurry images. Why and fix?**
MSE penalizes large pixel-wise errors uniformly, favoring blurred averages over sharp but slightly misaligned details. Use perceptual loss (VGG features), adversarial loss (GAN), or multi-scale loss.

**Q13. Your VAE latents aren't disentangled. What do you do?**
Try $\beta$-VAE (scale up KL weight to encourage disentanglement), FactorVAE, or structured priors. Warning: disentanglement without inductive bias is known to be unidentifiable (Locatello et al. 2019).

**Q14. When to use AE vs. PCA for dimensionality reduction?**
AE when non-linear structure is important and data is abundant. PCA when data is approximately linear, interpretability matters, or compute is limited. Start with PCA; move to AE if necessary.

**Q15. You want to embed images and text into a shared space. How do AEs help?**
Dual encoders, shared latent, reconstruct both modalities — a cross-modal AE. Or use contrastive learning (CLIP-style), which is often stronger.

#### Debugging & Failure Modes

**Q16. AE training loss plateaus high. What do you check?**
Architecture capacity (too small), learning rate, activation functions, loss function choice, data normalization, initialization. Try a simpler task first (overfit a small subset).

**Q17. VAE collapse to posterior = prior (ignoring input). Why?**
"Posterior collapse" — decoder is too powerful and can reconstruct without using $z$. Solutions: weaker decoder, KL annealing (start with low KL weight, ramp up), free bits (ensure KL per dim is at least $\lambda$).

**Q18. Your AE-based anomaly detector has many false negatives. What's happening?**
The AE generalizes to OOD — it can reconstruct anomalies too. Fix: constrain capacity, use contractive/denoising regularization, use reconstruction *probability* not just error (probabilistic AE), or switch to explicit density models.

#### Probing / Depth

**Q19. Why does denoising AE connect to score-matching?**
Vincent (2011): for a DAE with small Gaussian noise, the optimal denoiser $f^*(\tilde{x}) = \mathbb{E}[x|\tilde{x}]$. Under small noise, $f^*(\tilde{x}) - \tilde{x} \approx \sigma^2 \nabla \log p(\tilde{x})$, so the DAE learns an estimate of the score. This underlies diffusion models.

**Q20. What's the difference between a VAE and an AE + random latent sampling?**
VAE regularizes the encoder output toward a prior, making the latent space "fillable" — sampling the prior gives valid data. A plain AE has no such regularization; sampling near but off the training $z$'s decodes to garbage.

**Q21. Why do VAEs produce blurry images when GANs can produce sharp ones?**
VAE optimizes pixel-wise likelihood under a diagonal Gaussian decoder — the optimal decoder outputs the posterior mean, which averages over plausible pixels. GANs match distributions, rewarding sharp samples. Fix VAE sharpness: autoregressive decoders, more expressive likelihoods (PixelCNN decoder), or adversarial augmentation (VAE-GAN).

**Q22. Relationship between autoencoders and manifold learning?**
An undercomplete AE with enough capacity learns a parametric representation of the data manifold. The encoder is a chart; the decoder is an embedding. Compared to classical manifold learning (Isomap, LLE), AEs give an out-of-sample extension for free.

**Q23. Can an AE be used for missing-data imputation?**
Yes — train with masking (random dropouts on input) and reconstruct the full input. Related: Masked Autoencoders (MAE, He et al. 2021), used for pretraining vision transformers.

**Q24. Discuss the difference between $\beta$-VAE and standard VAE.**
$\beta$-VAE scales the KL term by $\beta > 1$, trading reconstruction quality for latent disentanglement. At $\beta = 1$ → standard VAE. Higher $\beta$ → more structured latents, worse reconstructions.

---

## 8. Manifold Learning: ISOMAP, LLE, Laplacian Eigenmaps

### 8.1 Motivation & Intuition

PCA assumes data lies near a linear subspace; but consider the **Swiss roll**: a 2D rectangular sheet rolled into a 3D spiral. Points on opposite edges of the unrolled sheet are close in 3D (across the spiral) but far along the manifold. PCA, using Euclidean 3D distances, conflates them. The *intrinsic* geometry requires walking along the surface.

Manifold learning methods assume data lies on a low-dimensional manifold smoothly embedded in high-D space and try to recover the intrinsic coordinates.

Three classic methods:
- **Isomap**: preserves *geodesic* (on-manifold) distances.
- **Locally Linear Embedding (LLE)**: preserves *local linear reconstructions*.
- **Laplacian Eigenmaps**: preserves *local neighborhoods* via graph spectral embedding.

All rely on building a neighborhood graph first and differ in how they use it.

Concrete ML motivations:
- Modeling data where the *intrinsic* dimension is small but the embedding space is large (images of one object under rotation/lighting — intrinsic dim = pose parameters).
- Visualization of curved structure.
- Preprocessing for downstream tasks when linear methods fail.

### 8.2 Conceptual Foundations (Common Setup)

**Step 0: Nearest neighbor graph.** For each $x_i$, find its $k$-nearest neighbors (or all within radius $\epsilon$). Draw edges.

Assumptions for all three methods:
1. **Manifold hypothesis**: data lies on or near a $d'$-dim manifold, $d' \ll d$.
2. **Local linearity** (LLE) or **local metric accuracy** (Isomap, LE): nearby points in Euclidean distance are also nearby on the manifold.
3. **Connected graph**: if the neighbor graph has disconnected components, methods break (Isomap can't compute cross-component distances; LLE/LE produce block-diagonal solutions).
4. **Dense-enough sampling**: manifold approximated by its samples; sparse sampling distorts.

**What breaks:**
- **Shortcuts**: if a point has neighbors on a different fold of the manifold (common in high-D with noise), Euclidean distance fails as a proxy for geodesic, producing shortcuts.
- **Holes or topology changes**: manifolds with non-trivial topology (e.g., torus) are not embeddable in $\mathbb{R}^{d'}$ — you get distortions.
- **Uneven density**: sparse regions get stretched, dense regions compressed.
- **Noise in ambient space** inflates local distances non-uniformly.

### 8.3 Isomap

#### 8.3.1 Idea

Replace Euclidean distances with **geodesic distances** estimated by shortest paths in the neighbor graph. Then apply classical **Multi-Dimensional Scaling (MDS)** to embed in low-D while preserving those distances.

#### 8.3.2 Algorithm

1. Build the KNN or epsilon-neighbor graph.
2. Edge weight = Euclidean distance.
3. Compute all-pairs shortest paths $D_{ij}$ (Dijkstra or Floyd-Warshall). This approximates geodesic distance.
4. Apply classical MDS: center the squared-distance matrix via $B = -\frac{1}{2}H D^{(2)} H$, where $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ is the centering matrix. Eigendecompose $B = V \Lambda V^\top$. Take top $d'$ eigenvectors: $Y = V_{d'} \Lambda_{d'}^{1/2}$.

#### 8.3.3 Math

Why does MDS on geodesic distances give the intrinsic coordinates? Because $B = -\frac{1}{2}HDH$, if distances *are* Euclidean, $B = XX^\top$ for centered $X$, whose eigendecomposition gives the coordinates. For a manifold, if geodesics approximate Euclidean distances in some $\mathbb{R}^{d'}$ (i.e., if the manifold is isometric to a subset of Euclidean space), the same logic yields a faithful low-D embedding.

**Complexity.** Shortest paths: $O(n^2 \log n)$ (Dijkstra $n$ times). Eigendecomp: $O(n^3)$ naive. Memory: $O(n^2)$.

**When Isomap works.** Manifold isometrically embeddable in $\mathbb{R}^{d'}$, graph connected, geodesic estimates accurate.

#### 8.3.4 Worked Example

**Swiss roll.** Generate 1000 points: $(t \cos t, h, t \sin t)$ for $t \in [3, 9]$, $h \in [0, 10]$. Neighbor graph on 10-NN. Shortest paths approximate the true "unrolled" distances. MDS gives a 2D embedding that recovers the flat rectangle.

### 8.4 Locally Linear Embedding (LLE)

#### 8.4.1 Idea

Each point is a linear combination of its neighbors. The same combination weights should work in the low-D embedding, because local linearity is invariant to isometric transformations.

#### 8.4.2 Algorithm

1. Find $k$-nearest neighbors.
2. **Compute reconstruction weights.** For each $i$, solve $\min_{W_i} \|x_i - \sum_{j \in N(i)} W_{ij} x_j\|^2$ subject to $\sum_j W_{ij} = 1$ (and $W_{ij} = 0$ for non-neighbors). Closed form via local Gram matrix inversion.
3. **Embed.** Find low-D $Y \in \mathbb{R}^{n \times d'}$ minimizing $\sum_i \|y_i - \sum_j W_{ij} y_j\|^2$ subject to centering and unit-covariance constraints.

#### 8.4.3 Math

Step 2: For neighbors of $x_i$, let $C$ be the local Gram: $C_{jk} = (x_i - x_j)^\top (x_i - x_k)$. Solve $C w = \mathbf{1}$; normalize $w$ to sum to 1.

Step 3: The cost is $\mathrm{tr}(Y^\top M Y)$ with $M = (I - W)^\top(I - W)$. Minimizing under orthonormality gives the bottom $d' + 1$ eigenvectors of $M$, discarding the trivial constant one.

**Complexity.** Weights $O(n k^3)$. Eigen-solve on sparse $M$: $O(n \cdot d' \cdot k)$ with iterative methods.

**Strengths.** Only local structure used; no global shortest paths. Robust to non-isometric manifolds.

**Weaknesses.** Collapses regions where weights are ambiguous; sensitive to $k$; handles only locally-linear manifolds.

### 8.5 Laplacian Eigenmaps (LE)

#### 8.5.1 Idea

Build a graph with weighted edges (close = high weight). Embed points so that connected pairs are close in low-D. This reduces to eigendecomposition of the graph Laplacian.

#### 8.5.2 Algorithm

1. KNN graph with edge weights $W_{ij} = \exp(-\|x_i - x_j\|^2/t)$ for $(i,j) \in E$, else 0.
2. Diagonal degree matrix $D_{ii} = \sum_j W_{ij}$.
3. Graph Laplacian $L = D - W$.
4. Solve generalized eigenvalue problem $L y = \lambda D y$. Take the bottom $d' + 1$ eigenvectors (drop the all-ones one with eigenvalue 0) as embedding coordinates.

#### 8.5.3 Math

Minimize $\sum_{i,j}W_{ij}\|y_i - y_j\|^2 = 2 \mathrm{tr}(Y^\top L Y)$ subject to $Y^\top D Y = I$. Lagrangian gives generalized eigenproblem $Ly = \lambda Dy$.

**Connection to spectral clustering.** Spectral clustering uses the same eigenvectors, then k-means. LE = embedding step of spectral clustering.

**Connection to heat kernel / diffusion maps.** LE is a linearization of the heat equation on the graph. Diffusion maps extend LE by exponentiating the transition matrix.

**Complexity.** Sparse eigendecomp $O(n \cdot d' \cdot k)$.

### 8.6 Relevance to ML Practice

**Historical vs. modern.** Isomap, LLE, and LE were developed 2000-2003 and were state-of-the-art before t-SNE/UMAP. They remain useful for:
- Interpretable, deterministic embeddings (closed-form solutions).
- Spectral theory underpinning graph neural networks (message-passing relates to Laplacian operators).
- Small datasets where $O(n^2)$ cost is tolerable.

**When not to use:** Large $n$ ($>10^5$) without approximations; heavy noise; visualization as primary goal (t-SNE/UMAP better aesthetics).

**Trade-offs:**
- **Isomap**: good global structure; fails on disconnected / non-isometric manifolds; $O(n^2)$ memory.
- **LLE**: purely local; fast; sensitive to $k$.
- **Laplacian Eigenmaps**: theoretically clean; deep links to spectral theory; may collapse sparse regions.

### 8.7 Common Pitfalls & Misconceptions

1. **Using Euclidean distances in high-D without checking concentration.** All distances look similar; KNN is noisy.
2. **Choosing $k$ or $\epsilon$ poorly.** Too small: disconnected graph. Too large: shortcuts.
3. **Expecting Isomap to work on closed manifolds (sphere, torus).** Can't be isometrically mapped to flat space.
4. **Ignoring disconnected components.** Methods break silently or produce block-diagonal.
5. **Treating LLE's weights as probabilities.** They sum to 1 but can be negative.
6. **Not standardizing features.** Distance-based methods are scale-sensitive.
7. **Assuming intrinsic dimensionality is automatically recovered.** You set $d'$; use intrinsic-dim estimators to inform the choice.
8. **No out-of-sample extension natively.** Use Nystrom approximation or parametric variants.
9. **Misreading LE eigenvector 1 (constant).** It's trivial; always discard.
10. **Thinking "non-linear is always better."** For near-linear data, PCA outperforms and is cheaper.

### 8.8 Interview Questions — Manifold Learning

#### Foundational

**Q1. What is the manifold hypothesis?**
High-D data often lies near a low-D manifold; intrinsic dimension is small even if the ambient space is large.

**Q2. Core idea of Isomap?**
Replace Euclidean distances with graph-shortest-path (geodesic) distances; apply MDS to embed.

**Q3. Core idea of LLE?**
Each point is a local linear combo of neighbors; preserve those combinations in low-D.

**Q4. Core idea of Laplacian Eigenmaps?**
Embed points via bottom eigenvectors of the graph Laplacian; connected points stay close.

**Q5. What's a geodesic distance?**
Shortest path along the manifold surface, not the straight-line Euclidean distance.

#### Mathematical

**Q6. Derive the classical MDS embedding.**
Given pairwise squared distances $D^{(2)}$, form $B = -\frac{1}{2}H D^{(2)} H$. If $B \succeq 0$ with rank $d'$, $B = YY^\top$ where $Y$ are coordinates. Eigendecomp: $Y = V_{d'}\Lambda_{d'}^{1/2}$.

**Q7. Derive the LLE embedding step.**
Minimize $\|(I-W)Y\|_F^2$ subject to $Y^\top Y = nI$, $\mathbf{1}^\top Y = 0$. Cost $= \mathrm{tr}(Y^\top M Y)$, $M = (I-W)^\top(I-W)$. Optimal $Y$ = bottom non-trivial eigenvectors of $M$.

**Q8. Define the graph Laplacian and its properties.**
$L = D - W$. Symmetric, PSD. $L \mathbf{1} = 0$. Multiplicity of eigenvalue 0 = number of connected components. Fiedler value controls graph connectivity.

**Q9. Why does LE use bottom eigenvectors, not top?**
Minimizing $\mathrm{tr}(Y^\top L Y)$ — we want small eigenvalues to place connected points close.

**Q10. Relationship between LE and spectral clustering?**
Spectral clustering = LE embedding + k-means. The LE step provides a representation where clusters become Euclidean-separable.

#### Applied

**Q11. Swiss roll — which method do you use?**
Isomap (classic success case), LLE, or LE all work in principle. UMAP dominates in practice.

**Q12. Isomap on 100k points — feasible?**
Marginal. $O(n^2 \log n)$ shortest paths; $O(n^2)$ memory. Use Landmark Isomap or switch to UMAP.

**Q13. Your Isomap embedding has a "shortcut." Why?**
A neighbor edge between points on different folds introduced a shortest-path shortcut. Reduce $k$ or $\epsilon$.

**Q14. You need out-of-sample embeddings. Workaround?**
Nystrom extension for LE; reconstruct new weights on training neighbors for LLE; or use UMAP (native inductive).

#### Debugging & Failure Modes

**Q15. LLE collapses most points to the origin. Why?**
The trivial solution $y_i = \text{const}$ is not properly excluded. Check solver and constraints.

**Q16. Isomap gives NaN/errors. Probable cause?**
Disconnected KNN graph — infinite shortest paths. Increase $k$ or $\epsilon$.

#### Probing / Depth

**Q17. Relationship between Isomap and PCA?**
Both eigendecompose a matrix derived from pairwise distances. Isomap replaces Euclidean with geodesics. For flat manifolds, they coincide.

**Q18. Relationship between LE and heat diffusion?**
LE eigenvectors approximate eigenfunctions of the Laplace-Beltrami operator, governing heat diffusion on the manifold. Small-eigenvalue eigenfunctions describe slow global modes.

**Q19. How do these connect to modern graph neural networks?**
GNN message-passing is closely related to iterative Laplacian smoothing $(I - \alpha L)$, linking GNNs to spectral graph theory and LE.

**Q20. Why has UMAP largely supplanted these methods?**
Scalability (linear vs. quadratic), inductive embeddings, better global structure preservation, more visually separable output.

**Q21. Compare Isomap, LLE, LE on a sphere.**
Sphere is non-Euclidean; Isomap distorts (geodesic-to-Euclidean mapping to disk). LLE produces fish-eye distortion. LE gives spherical harmonics as embeddings — elegant but still distorted.

---

# Part III — Feature Selection

---

Feature selection differs fundamentally from feature extraction: the output is a *subset* of original features, preserving their names and meanings. This is critical for interpretability, domain communication, and cases where collecting features has cost (medical tests, sensors).

Three broad families:
- **Filter methods**: score each feature against the target independent of a model.
- **Wrapper methods**: treat feature subsets as hyperparameters, search using a model's validation performance.
- **Embedded methods**: let the model do selection as part of training (Lasso, tree importance).

## 9. Filter Methods

### 9.1 Motivation & Intuition

You have 10,000 genes and 100 samples; training any model on all features is disastrous (overfitting, compute). Before training, cheaply score each feature by "is this feature associated with the target?" and keep the top $k$. This is **filter selection**: fast, model-agnostic, embarrassingly parallel.

Intuition: if feature $j$ has no statistical relationship to $y$, it's probably noise; drop it. Keep features with strong marginal association.

Concrete ML motivations:
- Ultra-high-D bioinformatics, text classification, sensor data.
- Pre-filtering before expensive wrappers.
- Fast baseline before anything fancy.

### 9.2 Conceptual Foundations

**Steps:**
1. Compute a score $s_j = \mathrm{score}(x_{\cdot j}, y)$ for each feature $j$ independently.
2. Rank features by $|s_j|$.
3. Keep top $k$ (or threshold by p-value).

**Scores for numeric $x$, numeric $y$:**
- **Pearson correlation** $\rho$: linear association. Range $[-1, 1]$. Captures only linear relationships.
- **Spearman correlation**: Pearson on ranks; captures monotone relationships.
- **Distance correlation** (Szekely): zero iff independent; captures any dependence.

**Scores for numeric $x$, categorical $y$:**
- **ANOVA F-statistic**: ratio of between-class to within-class variance. Equivalent to t-test for 2 classes.
- **Mutual information** $I(X; Y)$: zero iff independent; captures arbitrary dependence. Estimated via kNN (Kraskov) or binning.

**Scores for categorical $x$, categorical $y$:**
- **Chi-square** on contingency table: tests independence.
- **Mutual information**: directly applicable.
- **Cramer's V**: normalized chi-square.

**Assumptions.**
1. Marginal association approximates joint importance. Violated when features are only useful in combination (XOR).
2. Features are somewhat independent of each other, so redundancy isn't heavily double-counted.
3. The chosen statistic captures the relevant kind of association.

**What breaks.**
- **XOR features**: two features together perfectly predict $y$, each alone uncorrelated — filters miss both.
- **Redundant correlated features**: many pass filter, inflating top-$k$ uselessly.
- **Non-linear relationships** invisible to Pearson.
- **Small samples**: mutual info estimates are noisy.

### 9.3 Mathematical Formulation

**Pearson.** $\rho_{XY} = \mathrm{Cov}(X,Y) / (\sigma_X \sigma_Y)$. Sample: $r = \sum(x_i - \bar{x})(y_i - \bar{y}) / \sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}$.

**Mutual information.** $I(X;Y) = H(X) + H(Y) - H(X,Y) = \mathbb{E}_{p(x,y)}[\log (p(x,y)/(p(x)p(y)))]$. For continuous variables, practical estimators (Kraskov) use kNN distances.

**ANOVA F.** $F = \frac{\sum_k n_k (\bar{x}_k - \bar{x})^2 / (c-1)}{\sum_k \sum_{i \in k}(x_i - \bar{x}_k)^2 / (n - c)}$. Large $F$ indicates class means differ.

**Chi-square.** $\chi^2 = \sum_{i,j}(O_{ij} - E_{ij})^2 / E_{ij}$, where $O$ is observed contingency, $E$ is expected under independence. Null distribution: $\chi^2$ with $(r-1)(c-1)$ DOF.

### 9.4 Worked Example

Dataset: 5 features, binary target, 100 samples. Compute Pearson correlation per feature with $y$:

$|\rho_1| = 0.8, |\rho_2| = 0.7, |\rho_3| = 0.05, |\rho_4| = 0.1, |\rho_5| = 0.9$.

Rank: 5, 1, 2, 4, 3. Keep top 3: {5, 1, 2}.

Note: $x_3, x_4$ might jointly predict $y$ via interaction, which filters miss.

**MI example.** Binary $x$, binary $y$: $p(x=1,y=1) = 0.4$, $p(x=1,y=0) = 0.1$, $p(x=0,y=1) = 0.1$, $p(x=0,y=0) = 0.4$. Marginals: $p(x=1) = 0.5$, $p(y=1) = 0.5$.
$I = 0.4\log\frac{0.4}{0.25} + 0.1\log\frac{0.1}{0.25} + 0.1\log\frac{0.1}{0.25} + 0.4\log\frac{0.4}{0.25} \approx 0.278$ nats. Substantial dependence.

### 9.5 Relevance to ML Practice

**When to use:** Initial screening in high-D; fast exploration; prefiltering before wrapper; interpretable pipelines.

**When not to use:** Strong interaction effects; when a small precise subset matters (wrapper better); when redundancy must be managed.

**Extensions:** mRMR (max-relevance, min-redundancy), ReliefF (considers nearest-neighbor context, captures some interactions).

**Trade-offs:** Very fast, easy to understand. Low power for interactions. Redundancy not handled.

### 9.6 Common Pitfalls

1. **Using Pearson for non-linear relationships.** Use MI or Spearman.
2. **Chi-square with low cell counts.** Violates asymptotic assumptions; use Fisher's exact test.
3. **Selecting on full data, then evaluating.** Data leakage. Always CV with selection inside each fold.
4. **Redundant features dominating top-$k$.** Add redundancy filtering (mRMR, correlation cap).
5. **Ignoring multiple testing.** With 10k features, control FDR (Benjamini-Hochberg).
6. **Using MI with tiny data.** Estimator bias is severe; results unreliable.

### 9.7 Interview Questions — Filter Methods

#### Foundational

**Q1. What's a filter method?**
Feature scoring independent of the learning algorithm, based on statistical association with target.

**Q2. Pearson vs. Spearman vs. MI — when each?**
Pearson: linear, numeric-numeric, fast. Spearman: monotone, robust to outliers. MI: any dependence, more costly, needs good estimator.

**Q3. Main limitation of filter methods?**
Marginal view — miss feature interactions (e.g., XOR).

#### Mathematical

**Q4. Derive the connection between Pearson and least-squares slope.**
Simple regression: $\hat{\beta} = \rho \sigma_Y / \sigma_X$. So $\rho$ is the slope in standardized units.

**Q5. Explain ANOVA F in terms of variance decomposition.**
Total variance = between-class + within-class. F is the ratio (normalized by DOF). Under $H_0$: $F \sim F_{c-1, n-c}$.

**Q6. Why is MI preferred over Pearson for categorical targets?**
Pearson not meaningfully defined for categorical data. MI captures any dependence.

#### Applied

**Q7. 20k genes, 200 samples, binary label. Strategy?**
Per-gene t-test or MI. Correct for multiple testing (BH, FDR 5%). Keep top 100-500. Then wrapper / model.

**Q8. Filter picks 2 near-identical features. Problem?**
Redundant. Use mRMR or correlation cap (drop one from each correlated pair).

#### Debugging

**Q9. High-MI features give low downstream performance. Why?**
MI captures dependence but maybe the downstream model can't exploit it (e.g., non-linear MI but linear model); or noise correlation; or MI estimator was biased.

#### Probing

**Q10. How would you detect interaction-only features that a marginal filter misses?**
ReliefF (considers feature context via neighbors), or forward selection wrapper that evaluates feature pairs. Alternatively, fit a model with all pairwise interactions and check feature importances.

**Q11. How does mRMR work?**
mRMR = max relevance, min redundancy. Greedily select the feature that maximizes $I(x_j; y) - \frac{1}{|S|}\sum_{s \in S}I(x_j; x_s)$, where $S$ is the already-selected set. Balances target association against redundancy with selected features.

**Q12. How do you set the number of features to keep?**
Cross-validation on downstream task. Or use stability selection (run filter on bootstrap samples, keep features selected in $> 50\%$ of runs).

---

## 10. Wrapper Methods

### 10.1 Motivation & Intuition

Filters score features independently; but what if two features are individually useless but jointly perfect (XOR)? Or what if two individually excellent features carry the same information? A filter can't handle either case.

**Wrapper methods** solve this by evaluating *subsets* of features using the actual model's cross-validated performance as the score. Think of it as treating the feature set as a hyperparameter and searching over it.

Analogy: instead of testing each ingredient separately to predict a cake's quality, bake cakes with different ingredient combinations and taste-test them. More expensive, but captures synergies.

Concrete ML motivations:
- When interactions matter (image patches, gene pathways, feature combinations in tabular data).
- When you need a small, precise feature set (e.g., building a cheap sensor array — measure only the 5 most useful signals).
- When model-specific effects matter (Lasso might select different features than random forest; a wrapper optimizes for your chosen model).

### 10.2 Conceptual Foundations

**Framework:** Define a search space (all $2^d$ subsets), a search strategy (because $2^d$ is astronomical), and an evaluation criterion (CV score of a chosen model).

**Search strategies:**

**Forward Selection (FS).**
1. Start with empty set $S = \{\}$.
2. For each remaining feature $j$, evaluate model on $S \cup \{j\}$.
3. Add the $j^*$ that gives the best score.
4. Repeat until $|S| = k$ or score stops improving.

Greedy; $O(k \cdot d)$ model fits. Captures synergies only insofar as greedy adds can find them.

**Backward Elimination (BE).**
1. Start with all features $S = \{1, \dots, d\}$.
2. For each feature $j \in S$, evaluate model on $S \setminus \{j\}$.
3. Remove the $j^*$ whose absence hurts least (or helps most).
4. Repeat until $|S| = k$ or score drops.

$O((d - k) \cdot d)$ model fits. More expensive per step but starts from the full model, which can be better for finding interaction-based features.

**Recursive Feature Elimination (RFE).**
1. Train model on all features.
2. Rank features by model-derived importance (e.g., absolute coefficient in SVM/logistic, or impurity decrease in tree).
3. Remove bottom $m$ features.
4. Retrain; repeat until $k$ remain.

Hybrid between wrapper and embedded: uses model internals but retrains iteratively. Standard with SVMs (SVM-RFE) and linear models.

**Stepwise (bidirectional).**
Each iteration, consider both adding and removing a feature; proceed with whichever improves score most. Combines FS and BE; can recover from early mistakes.

**Assumptions.**
1. CV score is a reliable estimate of generalization.
2. Greedy search is a good-enough proxy for exhaustive search.
3. The model used for evaluation is the model you'll deploy (otherwise you've optimized for the wrong objective).

**What breaks.**
- **Computational cost**: $d$ in the thousands makes wrappers very expensive with complex models.
- **Overfitting the selection.** Many evaluations of CV score → risk of overfitting to the validation splits. Use nested CV (outer loop for performance estimate, inner for selection).
- **Greedy traps**: forward selection may miss XOR features unless both are added in the same step (unlikely). Backward is better for this but more expensive.
- **Non-stationarity across folds**: feature importance can vary across CV folds, giving unstable selections.

### 10.3 Mathematical Formulation

Let $S \subseteq \{1, \dots, d\}$ be a feature subset, $\mathcal{A}$ a learning algorithm, $L$ a loss function, and $\mathrm{CV}(S)$ the cross-validated score of $\mathcal{A}$ trained on features $S$.

**Forward selection:** $S_0 = \emptyset$. At step $t$:
$$
j^* = \arg\max_{j \notin S_{t-1}} \mathrm{CV}(S_{t-1} \cup \{j\}), \quad S_t = S_{t-1} \cup \{j^*\}.
$$
Stop when $\mathrm{CV}(S_t) \leq \mathrm{CV}(S_{t-1}) + \delta$ (no improvement beyond tolerance).

**Backward elimination:** $S_0 = \{1, \dots, d\}$. At step $t$:
$$
j^* = \arg\max_{j \in S_{t-1}} \mathrm{CV}(S_{t-1} \setminus \{j\}), \quad S_t = S_{t-1} \setminus \{j^*\}.
$$

**RFE.** Train model $f$ on $S$. Importance $I_j = |w_j|$ (or similar). Remove $\arg\min_j I_j$. Retrain.

**Complexity.** Forward: $O(k \cdot (d - k/2) \cdot C_{\mathrm{train}})$. Backward: $O((d-k) \cdot (d+k)/2 \cdot C_{\mathrm{train}})$. RFE: $O(d/m \cdot C_{\mathrm{train}})$ where $m$ features are removed per round.

### 10.4 Worked Example

**Forward selection.** 6 features, logistic regression, 5-fold CV.

- Round 1: test each feature alone. CV accuracies: $\{0.65, 0.72, 0.58, 0.60, 0.70, 0.55\}$. Best: feature 2 (0.72). $S = \{2\}$.
- Round 2: test $\{2,j\}$ for $j \in \{1,3,4,5,6\}$. Scores: $\{0.78, 0.73, 0.82, 0.76, 0.71\}$. Best: $\{2,4\}$ (0.82). $S = \{2,4\}$.
- Round 3: test $\{2,4,j\}$. Scores: $\{0.83, 0.81, 0.80, 0.79\}$. Best: $\{2,4,1\}$ (0.83). Marginal gain 0.01 — decide to stop.

Final: $S = \{2, 4\}$ (or $\{2, 4, 1\}$ if threshold is met).

Note: feature 4 alone scored only 0.60 but combined with feature 2 gave a big boost — interaction effect captured by the wrapper, which a filter would miss.

### 10.5 Relevance to ML Practice

**Where it's used:**
- Small-to-medium $d$ (up to ~100) with computationally cheap models.
- When feature interactions are suspected.
- Building minimal-cost feature sets (sensor selection, test panel design).
- SVM-RFE in bioinformatics (classic).

**When not to use:**
- $d$ in thousands+ — too slow. Use embedded methods or filter first.
- When using a very expensive model (deep NN) — each CV is prohibitive.
- When you need stability — wrappers are notoriously variable across data perturbations.

**Alternatives:**
- Embedded methods (Lasso, tree importance) — comparable quality, much faster.
- Filters + wrappers (filter to top 100, then wrapper).
- Bayesian optimization over feature subsets.

**Trade-offs:**
- **Quality**: generally finds better subsets than filters (accounts for interactions and model).
- **Cost**: much more expensive.
- **Stability**: lower (path-dependent greedy search).
- **Generality**: tied to a specific model — optimal subset for logistic regression may differ from that for random forest.

### 10.6 Common Pitfalls

1. **Using a single train-test split for scoring.** Noisy. Use $k$-fold CV.
2. **Not using nested CV.** If you select features via CV and report that same CV score as your estimated performance, you're overfit. Use outer CV for performance, inner CV for selection.
3. **Greedy path dependency.** Forward selection may find a suboptimal set. Try bidirectional; consider random restarts.
4. **Stopping too early.** Threshold matters — a useful feature may give small individual improvement.
5. **Wrapper for model A, deploy model B.** The optimal subset depends on the model. Match them.
6. **RFE removing important features early.** Multicollinearity can cause a useful feature's coefficient to be small; RFE removes it. Stabilize with group-level RFE or drop-out-based importance.
7. **Running wrappers on $d = 10{,}000$.** Computationally intractable for most models; pre-filter.
8. **Ignoring feature interdependence in BE.** Removing one of two correlated features can drastically change the other's importance in retrained model.

### 10.7 Interview Questions — Wrapper Methods

#### Foundational

**Q1. What's a wrapper method?**
Evaluates feature subsets by training a model on each subset and using cross-validated performance as the score.

**Q2. Forward selection vs. backward elimination — when to prefer each?**
FS is cheaper when you need few features from many ($k \ll d$). BE is better when you suspect features are important mainly via interactions (starts with full context). BE is prohibitive if $d$ is large.

**Q3. What is RFE?**
Train model, rank features by model-internal importance, remove least important, retrain. Repeat.

**Q4. Why is a wrapper better than a filter for finding interacting features?**
Because it evaluates subsets, not individual features — the combined effect of features is measured directly.

#### Mathematical

**Q5. State the computational complexity of forward selection.**
$O(k \cdot d \cdot C_{\mathrm{CV}})$ where $C_{\mathrm{CV}}$ is the cost of one cross-validated model fit.

**Q6. Formalize the nested CV setup for wrapper + performance estimation.**
Outer loop: $K_{\mathrm{out}}$-fold split. For each outer fold, run feature selection (using inner $K_{\mathrm{in}}$-fold CV) on the outer training set. Train final model on selected features (outer train), evaluate on outer test. Report mean outer test score.

#### Applied

**Q7. 500 features, need top 10 for a logistic regression. Approach?**
Filter to top 50 (MI or ANOVA). Then forward selection with 5-fold CV on those 50. Report via nested CV.

**Q8. SVM-RFE — why is it effective?**
SVM gives a clear importance ranking via $|w_j|$; iteratively removing least-important features and retraining lets the model adapt weights, often finding a compact, high-accuracy subset.

**Q9. Your wrapper-selected features differ across CV folds. Is this a problem?**
Indicates instability. Report feature selection frequency across folds; use stability metrics. Consider embedded methods for more stable selection or use stability selection.

#### Debugging

**Q10. Forward selection scores plateau after 3 features, but you know 5 are useful. Why?**
Possible: remaining useful features are correlated with the selected 3, so marginal gain is near zero. Or the model has saturated. Try backward elimination, or relax the stopping criterion.

**Q11. RFE removes a feature early that you know is important. Why?**
Multicollinearity: correlated partner absorbs the signal, giving the important feature a near-zero coefficient. Pre-cluster correlated features and remove clusters as units.

#### Probing

**Q12. Is forward selection guaranteed to find the optimal subset?**
No — it's greedy and can get stuck in suboptimal paths. Only exhaustive search ($2^d$ subsets) guarantees optimality, which is NP-hard. Branch-and-bound can help for small $d$.

**Q13. How does Bayesian optimization help with wrapper methods?**
Model the CV score as a function of feature-inclusion binary vector using a Gaussian process (or similar); use acquisition function to choose which subset to evaluate next. More sample-efficient than greedy search.

**Q14. How do you handle categorical features in a wrapper?**
Same framework — include/exclude each category group as a unit. The model handles encoding (one-hot, etc.). The wrapper evaluates subsets of original features, not dummies.

---

## 11. Embedded Methods

### 11.1 Motivation & Intuition

Filters ignore the model; wrappers optimize over subsets but are expensive. **Embedded methods** build feature selection *into* the model's training objective — the model itself decides which features matter as it learns.

The prototypical example is **Lasso** (L1-regularized regression): the L1 penalty drives coefficients of unimportant features exactly to zero. At convergence, only features with non-zero coefficients are "selected." No separate search — it happens during optimization.

Tree-based models provide another form: at each split, the tree selects the feature that best reduces impurity. Features never or rarely chosen for splits are unimportant.

Concrete ML motivations:
- Feature selection and model fitting in a single step — efficient.
- Principled regularization (bias-variance trade-off).
- Interpretable sparse models (Lasso yields a compact linear model).
- Scalable to high-D (Lasso with coordinate descent handles $d = 10^6$).

### 11.2 Conceptual Foundations

**Lasso (L1 regularization).**

Recall ridge regression: $\min \|y - X\beta\|^2 + \lambda \|\beta\|_2^2$. The L2 penalty shrinks coefficients but never zeros them.

Lasso: $\min \|y - X\beta\|^2 + \lambda \|\beta\|_1$. The L1 penalty creates a diamond-shaped constraint region. At corners of the diamond, some coefficients are exactly zero — feature selection.

**Why does L1 give exact zeros but L2 doesn't?** Geometrically: the L1 ball has corners aligned with axes; the contours of the loss (ellipses) are most likely to first touch the L1 ball at a corner, where one or more coordinates are zero. The L2 ball is round; touching points generically have all coordinates non-zero.

Analytically: the L1 penalty's subgradient at $\beta_j = 0$ is the interval $[-\lambda, \lambda]$; if the gradient of the data-fit term at $\beta_j = 0$ lies in this interval, the optimum stays at zero. The L2 penalty has gradient $2\lambda \beta_j$ at $\beta_j = 0$, which is zero — no force to keep $\beta_j$ at zero.

**Elastic net.** $\min \|y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$. Combines L1 (sparsity) and L2 (stability). Useful when features are correlated: Lasso picks one arbitrarily from a correlated group; elastic net picks the group together.

**Tree-based importance.**

Decision trees naturally rank features by how useful they are for splitting. Two flavors:
- **Impurity-based importance** (MDI, mean decrease in impurity): for each feature, sum the total impurity decrease across all splits in the forest that use that feature. Built-in to tree libraries. Fast, but biased toward high-cardinality / continuous features.
- **Permutation importance**: for each feature, randomly shuffle its values and measure the decrease in model performance. Model-agnostic. Unbiased, but costly ($d \times n_{\mathrm{permutations}}$ evaluations) and can be misleading when features are correlated (permuting one of two correlated features doesn't hurt much because the other compensates).

**Feature importance from gradients / attention.** In neural networks: saliency maps, attention weights, integrated gradients, SHAP values. These are feature *attribution* methods, not selection per se, but thresholds on them yield selection.

**Assumptions.**
1. **Lasso**: True model is sparse (few features matter). Linear relationship. If violated, Lasso still gives sparse solutions but they may be misleading.
2. **Tree importance**: model captures the relevant associations. Correlated features split importance in unpredictable ways.
3. **General**: the model is well-suited to the data; embedded selection inherits all model assumptions.

**What breaks.**
- **Lasso with correlated features**: selects arbitrarily from a group. Elastic net or group Lasso needed.
- **Lasso with many more features than samples ($d \gg n$)**: selects at most $n$ features (hard constraint from the optimization). Elastic net relaxes this.
- **Impurity importance with categorical features**: high-cardinality features (many unique values) get inflated importance because they offer more potential splits. Use permutation importance instead.
- **Permutation importance with correlated features**: permuting one leaves the correlated partner, so the model barely notices — underestimates the feature's true importance.

### 11.3 Mathematical Formulation

**Lasso.** $\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1$.

Subgradient optimality: for each $j$,
$$
-\frac{1}{n} x_j^\top(y - X\beta) + \lambda \partial|\beta_j| \ni 0,
$$
where $\partial|\beta_j| = \{\mathrm{sign}(\beta_j)\}$ if $\beta_j \neq 0$ and $[-1, 1]$ if $\beta_j = 0$.

At $\beta_j = 0$: feature $j$ is selected out iff $|\frac{1}{n} x_j^\top r| \leq \lambda$ (where $r$ is the residual from other features). The larger $\lambda$, the more features zeroed.

**Coordinate descent** (main solver): cycle through $j = 1, \dots, d$, solving the univariate problem
$$
\beta_j \leftarrow S_\lambda\left(\frac{1}{n}x_j^\top r_{-j}\right),
$$
where $S_\lambda(z) = \mathrm{sign}(z)\max(|z| - \lambda, 0)$ is the soft-thresholding operator and $r_{-j} = y - X_{-j}\beta_{-j}$ is the partial residual.

**Regularization path.** As $\lambda$ decreases from $\lambda_{\max} = \max_j |\frac{1}{n}x_j^\top y|$ (all-zero solution) to 0 (OLS solution), features enter one by one. The solution path is piecewise linear (the "LARS" property). Plot coefficients vs. $\lambda$ — this is the **Lasso path**.

**Elastic net.** $\min \frac{1}{2n}\|y - X\beta\|^2 + \lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$. Equivalently reparametrize with $\alpha = \lambda_1/(\lambda_1 + 2\lambda_2)$ (mixing parameter, $\alpha=1$ is Lasso, $\alpha=0$ is ridge). Soft-threshold update becomes $\beta_j \leftarrow S_{\lambda\alpha}(z_j) / (1 + \lambda(1-\alpha))$.

**Tree impurity importance.** For a tree $T$ with splits $\{s_1, \dots, s_m\}$, impurity decrease at split $s$ on feature $j$ with left/right children:
$$
\Delta I(s) = I(\mathrm{parent}) - \frac{n_L}{n}I(\mathrm{left}) - \frac{n_R}{n}I(\mathrm{right}).
$$
Feature $j$'s importance: $\mathrm{Imp}(j) = \sum_{s: \mathrm{feature}(s) = j} n_s \Delta I(s)$. For a forest: average over trees.

**Permutation importance.** For feature $j$:
$$
\mathrm{PI}(j) = \mathrm{Score}(X, y) - \mathrm{Score}(X_{\pi_j}, y),
$$
where $X_{\pi_j}$ is $X$ with column $j$ randomly shuffled. Average over multiple shuffles.

### 11.4 Worked Example

**Lasso.** 3 features, 5 observations, regression.
$X = \begin{pmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \\ 0 & 0 & 0 \\ 1 & 0 & 0.5 \end{pmatrix}$, $y = (2, 1, 3, 0, 2)^\top$.

True model: $y = 2x_1 + x_2$ (feature 3 = average of 1 and 2, redundant).

**At large $\lambda$.** All coefficients zero. $\lambda_{\max} = \max_j |x_j^\top y / n|$. Compute: $x_1^\top y = 2 + 0 + 3 + 0 + 2 = 7$, $x_2^\top y = 0 + 1 + 3 + 0 + 0 = 4$, $x_3^\top y = 1 + 0.5 + 3 + 0 + 1 = 5.5$. So $\lambda_{\max} = 7/5 = 1.4$.

**Decreasing $\lambda$.** Feature 1 enters first (largest correlation with residual). Then feature 2. Feature 3 may never enter (its signal is explained by 1 and 2).

**At $\lambda = 0$.** OLS: infinitely many solutions (collinearity). Lasso path prefers the sparse one.

**Tree importance.** Fit a random forest. Feature 1 used most for splits with large impurity reduction. Feature 3 gets some importance (it's correlated with 1 and 2) but less than 1 or 2. Feature 2 gets moderate importance. Result: 1 > 2 > 3.

**Permutation importance.** Shuffling feature 1: large accuracy drop (contains unique signal). Shuffling feature 3: small drop (1 and 2 compensate). More accurate ranking.

### 11.5 Relevance to ML Practice

**Where it's used:**
- **Lasso/elastic net**: standard for sparse linear models in genomics, econometrics, NLP.
- **Tree importance**: default feature ranking in XGBoost/LightGBM/RF pipelines.
- **Permutation importance**: model-agnostic post-hoc importance.
- **Group Lasso**: selects groups of related features (e.g., all dummies from one categorical variable).
- **SHAP values**: modern feature attribution; theoretically grounded importance.

**When to use:**
- When interpretability and sparsity are goals.
- When $d$ is large (Lasso scales to millions).
- When the model already provides importance signals.

**When not to use:**
- When non-sparse models are needed (ridge is better than Lasso if all features contribute).
- When impurity importance is biased (high-cardinality categoricals — use permutation).
- When strict subset selection is required and the embedded method gives weights, not binary inclusion.

**Alternatives:**
- Filters + wrappers (for model-agnostic selection).
- Stability selection (Lasso on bootstrap samples; keeps only features selected in most samples).
- SHAP-based selection (fit model, compute SHAP, keep top-SHAP features).
- Knockoff filter (controls false discovery rate rigorously).

**Trade-offs.**
- **Lasso**: computationally cheap ($O(nd \cdot \mathrm{iterations})$); unstable with correlations.
- **Elastic net**: stable grouping; extra hyperparameter.
- **Tree importance**: fast; biased by cardinality.
- **Permutation importance**: unbiased; $O(d \cdot n)$ evaluations; misleading under correlation.
- **SHAP**: theoretically principled; expensive for large models.

### 11.6 Common Pitfalls

1. **Interpreting Lasso's selected features as "the true features."** Lasso picks arbitrarily from correlated groups. Use stability selection or elastic net for reliability.
2. **Using impurity importance for categorical features.** High-cardinality features get inflated scores. Use permutation importance.
3. **Permutation importance with correlated features.** Underestimates importance of each member of a correlated pair. Use conditional permutation or group-level permutation.
4. **Not tuning $\lambda$ in Lasso.** Too large: underfitting (too few features). Too small: overfitting. Use CV (LassoCV).
5. **Confusing feature importance with causal effect.** Importance says "useful for prediction," not "causes the outcome."
6. **Using Lasso when $d \gg n$ and expecting more than $n$ features selected.** Lasso selects at most $n$ features. Use elastic net.
7. **Reading SHAP magnitudes without checking directionality.** A feature can be "important" but its effect might be surprising or unintuitive.
8. **Comparing importances across different models.** Importances are model-specific; a feature important for RF might not be for logistic regression.
9. **Not standardizing features before Lasso.** L1 penalty treats all coefficients equally; if features are on different scales, larger-scale features are penalized more in absolute terms.
10. **Computing permutation importance on training data.** Gives inflated importance for overfit models. Use a held-out set.

### 11.7 Interview Questions — Embedded Methods

#### Foundational

**Q1. What is an embedded method?**
Feature selection built into model training — the model itself determines which features to use as part of its optimization.

**Q2. Why does L1 give sparse solutions but L2 doesn't?**
L1 ball has corners on axes; loss contours touch corners first, zeroing some coefficients. L2 ball is round; touches generically at non-zero coordinates. Subgradient: L1 penalty at zero has interval $[-\lambda, \lambda]$ which can absorb small gradients; L2 has gradient zero at zero, no zeroing force.

**Q3. What's the difference between impurity importance and permutation importance?**
Impurity: sum of impurity decreases for a feature across splits (biased toward high cardinality). Permutation: model performance drop when feature is shuffled (unbiased but costly and affected by correlations).

**Q4. What is elastic net?**
Combines L1 and L2 penalties: $\lambda_1\|\beta\|_1 + \lambda_2\|\beta\|_2^2$. Sparsity from L1, stability from L2. Selects groups of correlated features.

**Q5. What's the soft-thresholding operator?**
$S_\lambda(z) = \mathrm{sign}(z)\max(|z| - \lambda, 0)$. Core update in Lasso coordinate descent. Shrinks toward zero; sets to exactly zero if $|z| \leq \lambda$.

#### Mathematical

**Q6. Derive the Lasso subgradient optimality condition.**
$\frac{\partial}{\partial \beta_j}[\frac{1}{2n}\|y - X\beta\|^2 + \lambda|\beta_j|] \ni 0$. The data term gives $-\frac{1}{n}x_j^\top(y - X\beta)$. The L1 term's subgradient is $\lambda \cdot \mathrm{sign}(\beta_j)$ if $\neq 0$, or $\lambda \cdot [-1,1]$ if $= 0$. At $\beta_j = 0$: $|x_j^\top r/n| \leq \lambda$ (feature not selected).

**Q7. Show that Lasso selects at most $n$ features when $d > n$.**
The Lasso solution satisfies KKT: the active set $A = \{j : \beta_j \neq 0\}$ defines a system $X_A^\top(y - X_A \beta_A) = n\lambda \cdot \mathrm{sign}(\beta_A)$ with $|A|$ unknowns. $X_A$ has rank $\leq n$, so $|A| \leq n$.

**Q8. Why does the Lasso path (coefficients vs. $\lambda$) start from zero at $\lambda_{\max}$?**
At $\lambda_{\max} = \max_j |x_j^\top y/n|$, the KKT condition $|x_j^\top y/n| \leq \lambda$ holds for all $j$ with equality for at most one — so $\beta = 0$ is optimal. Below $\lambda_{\max}$, features enter one at a time.

**Q9. Why is Lasso unstable with correlated features?**
If $x_j \approx x_k$, either can explain the same signal. Small data perturbations can flip which one gets selected. The Lasso solution path can show erratic switching. Elastic net stabilizes by selecting both.

**Q10. What is group Lasso?**
Penalty $\lambda \sum_g \|\beta_g\|_2$ where $g$ indexes groups. Zeros out entire groups (all dummies of a categorical, or all features in a pathway) rather than individual features.

#### Applied

**Q11. You have 50k features, 10k samples, binary classification. Pipeline?**
LassoCV or elastic net with logistic loss. Use coordinate descent solver (glmnet or sklearn). Plot CV curve for $\lambda$. Examine selected features. Validate on held-out set. Stability selection for confidence.

**Q12. Random forest says feature X is important but permutation importance says it's not. Why?**
X is correlated with another feature Y. Impurity importance splits credit between them; permutation of X alone doesn't hurt because Y compensates. Conclusion: X and Y together are important; individual attribution is ambiguous.

**Q13. You need to select genes for a diagnostic panel. What embedded method?**
Elastic net (handles correlated genes, gives a sparse set). Supplement with stability selection for robustness.

**Q14. How do you choose Lasso's $\lambda$?**
Cross-validation: fit Lasso for a grid of $\lambda$ values, pick the $\lambda$ giving best CV score. Common choices: $\lambda_{\min}$ (best score) or $\lambda_{\mathrm{1se}}$ (most regularized within 1 SE of best — sparser, more conservative).

**Q15. A colleague uses feature importance from a single decision tree. What's wrong?**
Single trees are high-variance. Importances change drastically across data subsets. Use a forest (averages over many trees) and prefer permutation importance.

#### Debugging & Failure Modes

**Q16. Lasso selects 5 features. Adding 3 more via domain knowledge improves test performance. What happened?**
Lasso may have missed interacting features (L1 penalizes individual coefficients) or excluded features correlated with selected ones. Or $\lambda$ was too large. Try elastic net with smaller $\lambda$; check if the 3 features form a group with selected ones.

**Q17. Permutation importance gives negative values for some features. Interpretation?**
Feature adds noise; removing it *helps*. Model uses it to memorize training data (overfit). These features should be dropped. Or the signal in the feature is opposite to a correlated feature, and permuting breaks a cancellation.

**Q18. LassoCV selects different features on different random seeds. Why?**
Randomness in CV splits + correlated features = unstable selection. Use fixed seeds for reproducibility; more importantly, use stability selection to identify reliably selected features.

#### Probing / Depth

**Q19. What is stability selection (Meinshausen-Buhlmann)?**
Run Lasso on many bootstrap/subsamples of the data. A feature is declared "stable" if selected in a high fraction (e.g., >60%) of subsamples. Controls false discovery rate while giving robust selection.

**Q20. How does the knockoff filter control FDR for feature selection?**
Construct "knockoff" features $\tilde{X}$ that match the original features' correlation structure but are independent of the response. Run Lasso on $[X, \tilde{X}]$. Features are selected only if their coefficient exceeds their knockoff's — provably controls FDR.

**Q21. Connection between Lasso and Bayesian variable selection?**
Lasso = MAP estimate under a Laplace prior on $\beta$. Bayesian spike-and-slab prior gives a more principled selection (posterior inclusion probability per feature) but is computationally harder.

**Q22. How does SHAP relate to embedded importance?**
SHAP provides a theoretically grounded decomposition of each prediction into feature contributions (Shapley values from game theory). It's model-agnostic and satisfies properties (efficiency, symmetry, dummy) that impurity/permutation importance don't. SHAP can be used post-hoc with any embedded model.

**Q23. Can you do Lasso with non-linear models?**
Not directly, but: (a) L1 penalty on neural network weights (doesn't zero cleanly due to non-convexity); (b) use structured sparsity (group Lasso on input layer weights); (c) use proxy: train NN, compute SHAP or gradient-based importance, threshold.

**Q24. What's the advantage of the Lasso regularization path over a single $\lambda$?**
The path shows how features enter/exit as $\lambda$ changes, revealing relative importance ordering, feature stability (features entering/leaving erratically are unstable), and helping choose $\lambda$ by examining the path shape.

---

# Appendix: Cross-Cutting Interview Questions

These questions test understanding across multiple topics and are typical of senior-level interviews.

**Q1. You have 10k features, 500 samples, classification task. Walk through your full dimensionality reduction / feature selection pipeline.**

Step 1: Exploratory analysis — check for missing values, distributions, correlations. Step 2: Filter (MI or ANOVA-F) to remove clearly irrelevant features (keep top 500-1000). Step 3: PCA or elastic net. If interpretability matters, elastic net gives a sparse feature set. If compression matters, PCA to 50-100 components. Step 4: If PCA, consider LDA (supervised, adds class information). Step 5: Train model on reduced features; cross-validate. Step 6: For final feature count, use CV on downstream task, not just explained variance. Step 7: Validate on held-out test set.

**Q2. PCA says 50 components explain 95% variance, but your classifier is at chance. Diagnose.**

PCA is unsupervised — 95% of variance might be noise or structure irrelevant to the target. The discriminative direction might be in the 5% you discarded. Try: LDA (supervised); use all PCs plus regularization; or use embedded feature selection that considers the target.

**Q3. Compare t-SNE, UMAP, and PCA for a biologist who wants to "see clusters."**

PCA: fast, deterministic, preserves global variance, but may not show clusters if they're non-linearly separable. UMAP: default for most bioinformatics (single-cell RNA-seq); good cluster separation, scales well, somewhat preserves global topology. t-SNE: great cluster separation but slower, global structure unreliable. Recommendation: Start with PCA for a global overview, then UMAP for detailed cluster visualization.

**Q4. When is feature selection better than feature extraction?**

When interpretability is required (stakeholders need to know *which* original features matter). When feature collection has cost (choose the cheapest informative features). When the model needs to be auditable (regulatory, medical). When original features have domain meaning that projections destroy.

**Q5. You're building an anomaly detection system for network traffic. Which dimensionality reduction method do you use?**

Depends on the anomaly type. Autoencoder: train on normal traffic; anomalies have high reconstruction error. PCA: monitor deviation from the top-$k$ subspace (reconstruction residual). One-class SVM in PCA-reduced space. For visualization: UMAP to spot clusters of attack types. Avoid t-SNE for production monitoring (stochastic, no inductive map without parametric variant).

**Q6. A colleague says "I ran PCA, then ICA, then t-SNE." Is this sequence sensible?**

PCA first is reasonable (denoise, reduce dimensionality). ICA after PCA can recover independent sources if the data is a linear mixture — fine if that's the hypothesis. t-SNE after ICA for visualization — possible, but unusual; typically you'd t-SNE the PCA output or the raw data. The ICA step should have a specific purpose (source separation), not just be thrown in.

**Q7. Your Lasso model selects 20 features. Your random forest says a different 20 are important. Why and what do you do?**

Different models capture different structures. Lasso finds linear predictors; RF finds features useful for splits (possibly non-linear, interaction-based). Both are "correct" under their model. Resolution: (a) check overlap — features in both are strong candidates; (b) use SHAP on the RF for interpretable feature attribution; (c) domain knowledge to adjudicate; (d) elastic net for grouped selection if features are correlated.

**Q8. How do you estimate the intrinsic dimensionality of a dataset?**

Methods: (a) Scree plot / explained variance from PCA. (b) Maximum likelihood estimator (Levina-Bickel): estimate local intrinsic dimensionality via nearest-neighbor distances. (c) Correlation dimension. (d) Topological methods (persistent homology). (e) Reconstruction error vs. dimension curve for autoencoders. In practice, (a) and (b) are most common.

**Q9. For a production ML system, would you use t-SNE, UMAP, or PCA for feature engineering?**

PCA. It's deterministic, has an inductive map (learned projection matrix), is computationally cheap, and gives interpretable variance-based diagnostics. t-SNE and UMAP are for visualization only in production contexts (no meaningful feature engineering value, non-deterministic, unstable).

**Q10. Explain how manifold learning, autoencoders, and kernel PCA all relate to each other.**

All learn non-linear low-dimensional representations. Kernel PCA: PCA in a reproducing kernel Hilbert space (implicitly non-linear). Isomap/LLE/LE: explicit graph-based manifold geometry. Autoencoders: parametric non-linear encoder-decoder. Kernel PCA and LE both eigendecompose kernel/Laplacian matrices. AEs generalize all of them with enough capacity but need more data and compute. LE's graph structure connects to GNNs; kernel PCA connects to Gaussian processes; AEs connect to VAEs and modern deep generative models.

---

*End of Dimensionality Reduction Guide*

---

**[← Previous Chapter: Clustering](clustering_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Anomaly Detection →](anomaly_detection_guide.md)**
