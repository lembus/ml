# Dimensionality Reduction & Feature Selection: A Comprehensive Reference

---

## Module 1: Linear Dimensionality Reduction

### 1. Motivation & Intuition

Imagine you are trying to describe a car to a friend. You could list thousands of details: the exact air pressure in the front left tire, the number of threads on the steering wheel, the molecular composition of the paint, etc. This is **high-dimensional data**.

However, to distinguish a Ferrari from a Jeep, you only need a few key "features": size, shape, engine power, and ground clearance. This simplification is **dimensionality reduction**.

In Machine Learning, high-dimensional data (e.g., images with millions of pixels, genomic data with thousands of genes) creates the **Curse of Dimensionality**:

1. **Data Sparsity:** As dimensions increase, data points become exponentially sparse, making it difficult for algorithms to find statistically significant patterns.
2. **Meaningless Distances:** In high dimensions, the Euclidean distance between *any* given pair of random points tends to converge to the exact same value, causing distance-based algorithms (like $k$-Nearest Neighbors or $k$-Means) to fail.
3. **Overfitting:** More features require exponentially more parameters to learn, increasing the risk of modeling sample noise rather than the underlying signal.

**Linear methods** operate on the fundamental assumption that the "useful" data lies on a lower-dimensional flat geometric surface (a line, a plane, or a hyperplane) embedded within the high-dimensional space.

> **Visual Concept:** *Think of a 3D cloud of data points shaped like a flat pancake tilted at an angle. PCA rotates the 3D coordinate system so that two axes lie flat along the pancake, allowing us to drop the third (thickness) axis with almost zero loss of information.*

---

### 2. Conceptual Foundations

#### Principal Component Analysis (PCA)
PCA finds a new set of coordinate axes (called *principal components*) that are orthogonal (perpendicular) to each other and strictly aligned with the directions of maximum variance in the dataset.
* **PC1 (First Principal Component):** The spatial direction along which the data varies the most.
* **PC2:** The direction that captures the second-most variance, strictly constrained to be orthogonal to PC1.

#### Linear Discriminant Analysis (LDA)
While PCA is an *unsupervised* technique that looks for pure variance, LDA is a *supervised* technique that searches for **class separability**. It finds a linear projection that maximizes the distance between the means of different classes while simultaneously minimizing the spread (variance) within each individual class.

#### Factor Analysis (FA) vs. Independent Component Analysis (ICA)
* **Factor Analysis (FA):** Assumes the observed features are linear combinations of unobserved, strictly Gaussian "latent variables" plus Gaussian noise. It focuses entirely on modeling the **covariance** between features.
* **Independent Component Analysis (ICA):** Assumes the observed data is a linear mixture of statistically independent, **non-Gaussian** source signals. It focuses on "unmixing" these signals (solving the classic *Cocktail Party Problem*—e.g., isolating two individual speaking voices recorded on a single microphone).

---

### 3. Mathematical Formulation

#### PCA: The Variance Maximization Derivation

Let $X$ be an $n \times d$ dataset ($n$ samples, $d$ features), centered such that the column means are zero ($\mu = 0$). We seek a unit vector $w$ (where $w^T w = 1$) such that the variance of the data projected onto $w$ is maximized.

The projection of $X$ onto vector $w$ is given by the vector $z = Xw$. 

The empirical variance of this projected vector $z$ is:

$$
\text{Var}(z) = \frac{1}{n} z^T z = \frac{1}{n} (Xw)^T (Xw) = w^T \left( \frac{1}{n} X^T X \right) w
$$

The inner term $\frac{1}{n} X^T X$ is the sample **Covariance Matrix**, denoted as $\Sigma$. Therefore, our objective is:

$$
\max_{w} \ w^T \Sigma w \quad \text{subject to} \quad w^T w = 1
$$

To solve this constrained optimization problem, we introduce a **Lagrange multiplier** $\lambda$:

$$
\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda (w^T w - 1)
$$

Taking the partial derivative with respect to the vector $w$ and setting it to zero:

$$
\frac{\partial \mathcal{L}}{\partial w} = 2 \Sigma w - 2 \lambda w = 0 \implies \Sigma w = \lambda w
$$

**Conclusion:** The projection vector $w$ must be an **eigenvector** of the covariance matrix $\Sigma$, and $\lambda$ is its corresponding **eigenvalue**. Because our original goal was to maximize variance ($w^T \Sigma w$), and substituting $\Sigma w = \lambda w$ yields $w^T(\lambda w) = \lambda(w^T w) = \lambda$, **the variance captured along $w$ is exactly equal to the eigenvalue $\lambda$**. Therefore, to maximize variance, PC1 must be the eigenvector associated with the largest eigenvalue $\lambda_1$.

#### The SVD Approach
In production ML systems, calculating the $d \times d$ covariance matrix explicitly is computationally unstable for high $d$. Instead, we apply Singular Value Decomposition directly to the centered matrix $X$:

$$
X = U S V^T
$$

Where the right-singular vectors (the columns of $V$) are strictly identical to the eigenvectors $w$ of $X^TX$. The eigenvalues are related to the singular values $s_i$ (diagonal entries of $S$) via:

$$
\lambda_i = \frac{s_i^2}{n - 1}
$$

#### LDA: Fisher’s Criterion
LDA seeks a projection vector $w$ that maximizes the ratio of *between-class scatter* ($S_B$) to *within-class scatter* ($S_W$):

$$
J(w) = \frac{w^T S_B w}{w^T S_W w}
$$

Differentiating $J(w)$ with respect to $w$ yields the generalized eigenvalue problem: $S_B w = \lambda S_W w \implies (S_W^{-1} S_B) w = \lambda w$.

---

### 4. Worked Example: PCA on a 2D Dataset

**Given Dataset:** Three 2D points: $A(1, 2), B(3, 4), C(5, 6)$

**Step 1: Mean Centering**
* Feature 1 mean: $\mu_x = (1 + 3 + 5) / 3 = 3$
* Feature 2 mean: $\mu_y = (2 + 4 + 6) / 3 = 4$

Subtracting the means yields our centered matrix $X_{\text{centered}}$:
* $A' = (-2, -2)$
* $B' = (0, 0)$
* $C' = (2, 2)$

**Step 2: Compute Covariance Matrix ($\Sigma$)**
Using $n-1$ degrees of freedom ($n=3$):

$$
\text{Var}(x) = \frac{(-2)^2 + 0^2 + 2^2}{3-1} = \frac{8}{2} = 4
$$

$$
\text{Cov}(x,y) = \frac{(-2)(-2) + 0(0) + (2)(2)}{3-1} = \frac{8}{2} = 4
$$

$$
\Sigma = \begin{bmatrix} 4 & 4 \\ 4 & 4 \end{bmatrix}
$$

**Step 3: Eigen Decomposition**
Solve the characteristic equation $\det(\Sigma - \lambda I) = 0$:

$$
\det \begin{bmatrix} 4-\lambda & 4 \\ 4 & 4-\lambda \end{bmatrix} = (4-\lambda)^2 - 16 = 0
$$

$$
16 - 8\lambda + \lambda^2 - 16 = 0 \implies \lambda(\lambda - 8) = 0
$$

* **Eigenvalues:** $\lambda_1 = 8, \quad \lambda_2 = 0$
* **Explained Variance Ratio of PC1:** $\frac{8}{8 + 0} = 100\%$

**Takeaway:** The raw data points fall perfectly along the line $y = x + 1$. Because 100% of the dataset's variance lies along a single 1D vector, PCA discovers that the 2D dataset actually possesses an intrinsic dimensionality of 1.

---

### 5. Relevance to Machine Learning Practice

* **Preprocessing Pipeline:** Applied immediately prior to linear regression, logistic regression, or SVMs to break multicollinearity and shrink model memory footprint.
* **Exploratory Data Analysis (EDA):** Projecting 1,000-feature customer data down to 2 or 3 principal components to visually check for natural market segments.
* **Signal Denoising:** Transforming data into $k$ components, zeroing out trailing components, and reconstructing the matrix projects the data back onto its cleanest underlying subspace.

---

### 6. Common Pitfalls & Misconceptions

* **Forgetting to Standardize:** If Feature A is measured in kilometers (range 0–1,000) and Feature B in millimeters (range 0–1,000,000), Feature B's numerical variance will dwarf Feature A. **You must run `StandardScaler` ($\mu=0, \sigma=1$) prior to PCA.**
* **Assuming Linearity Fits All:** If data lies on a hollow sphere or a coiled spiral, linear transformations will smash overlapping layers together, destroying the topology.
* **Loss of Feature Identity:** Principal components are linear composites (e.g., $\text{PC1} = 0.42\cdot\text{Age} - 0.11\cdot\text{Income}$). You can no longer tell a stakeholder "Age drove this prediction."

---
---

## Module 2: Non-linear Dimensionality Reduction

### 1. Motivation & Intuition

The physical world is rarely flat. Data frequently resides on a **manifold**—a curved, flexible, low-dimensional surface tangled inside a high-dimensional space. 

Imagine a world map. The Earth is a 3D sphere, but atlases flatten it into a 2D page. If you use a linear projection (like taking a 2D photograph of a globe), Greenland appears massively distorted and the opposite sides of the Pacific Ocean look infinitely far apart. To map it correctly, you must *unroll* the geometry.

> **Visual Concept:** *The classic "Swiss Roll" dataset looks like a rolled-up spiral cake. Two points on adjacent layers of the roll might be 2 inches apart via a straight 3D Euclidean line (jumping across the empty air gap), but 12 inches apart if you are forced to walk along the baked surface of the cake. Linear PCA jumps the gap; Manifold Learning walks the cake.*

---

### 2. Conceptual Foundations

#### t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE converts Euclidean distances between data points into conditional probabilities that represent similarities:
1. **High-Dimensional Space:** Models the probability that point $x_i$ picks $x_j$ as its neighbor using a **Gaussian (Normal) distribution**.
2. **Low-Dimensional Space:** Models the equivalent neighborhood probability $y_i \to y_j$ using a **Student's t-distribution** (specifically a Cauchy distribution with 1 degree of freedom).
3. **The Crowding Problem:** In 100-dimensional space, a single point can have dozens of mutually equidistant nearest neighbors. In 2D space, there is physically not enough geometric room to place all those neighbors at the exact same radius without them overlapping. The Student's t-distribution has "heavier tails" than a Gaussian, meaning it mathematically permits moderately distant points in high-dimensions to be pushed much further away in 2D space, preventing visual crowding.

#### UMAP (Uniform Manifold Approximation and Projection)
Built on Riemannian geometry and algebraic topology. UMAP constructs a high-dimensional fuzzy simplicial set (a weighted graph of local neighborhoods) and optimizes a low-dimensional graph to structurally match it. Unlike t-SNE, UMAP preserves **global cluster relationships** (the distance between cluster A and cluster B actually means something) and runs exponentially faster on massive datasets.

#### Autoencoders
Unsupervised neural networks explicitly trained to reconstruct their own input data:

$$
\text{Input } (x) \longrightarrow \begin{bmatrix} \text{Encoder} \\ f_\theta(x) \end{bmatrix} \longrightarrow \text{Bottleneck } (z) \longrightarrow \begin{bmatrix} \text{Decoder} \\ g_\phi(z) \end{bmatrix} \longrightarrow \text{Reconstruction } (\hat{x})
$$

By forcing a 784-dimensional image through a 32-dimensional bottleneck layer ($z$), the network is forced to learn a compressed, non-linear representation of the data distribution.

---

### 3. Mathematical Formulation

#### t-SNE Optimization Objective
Let $P_{ij}$ be the symmetric joint probability of points $i$ and $j$ in the high-dimensional space, and $Q_{ij}$ be the equivalent probability in the low-dimensional embedding. t-SNE minimizes the **Kullback-Leibler (KL) Divergence** over all pairs via Gradient Descent:

$$
\mathcal{L}_{\text{t-SNE}} = D_{\text{KL}}(P \parallel Q) = \sum_{i \neq j} P_{ij} \log \left( \frac{P_{ij}}{Q_{ij}} \right)
$$

Because KL divergence is asymmetric:
* If $P_{ij}$ is high (points are close in raw data) and $Q_{ij}$ is low (placed far apart in 2D), the cost is **massive**. The algorithm aggressively pulls true neighbors together.
* If $P_{ij}$ is low (points are far apart in raw data) and $Q_{ij}$ is high (placed close in 2D), the penalty is minuscule. Thus, t-SNE prioritizes local neighborhoods over global distances.

#### Autoencoder Objective Function
Optimized via standard backpropagation using Mean Squared Error (MSE) or Binary Cross-Entropy:

$$
\mathcal{L}_{\text{AE}}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^{n} \left\| x^{(i)} - g_\phi \left( f_\theta(x^{(i)}) \right) \right\|^2
$$

---

### 4. Worked Example: Manifold Logic via ISOMAP

Imagine 3 data points sitting on a horseshoe curve: $A \to B \to C$. 
* Straight-line Euclidean distance: $d_{\text{euc}}(A, B) = 2$, $d_{\text{euc}}(B, C) = 2$, but the gap jump $d_{\text{euc}}(A, C) = 1.5$. 

**ISOMAP Execution Steps:**
1. **Neighborhood Graph:** Connect all points within a threshold $k=1$. $A$ connects to $B$ (weight 2), $B$ connects to $C$ (weight 2). $A$ does *not* connect directly to $C$ because the straight line cuts through empty manifold space.
2. **Shortest Path Matrix:** Compute the *geodesic* distance between all pairs using Dijkstra's Algorithm:
   * $D_{\text{geo}}(A, B) = 2$
   * $D_{\text{geo}}(B, C) = 2$
   * $D_{\text{geo}}(A, C) = 2 + 2 = 4$
3. **MDS Embedding:** Pass this new corrected distance matrix $D_{\text{geo}}$ into Classical Multidimensional Scaling (MDS). The algorithm unbends the horseshoe into a straight 1D line of length 4.

---

### 5. Relevance to Machine Learning Practice

* **Embeddings Inspection:** Visualizing latent spaces of LLMs (e.g., mapping 1,536-dimensional OpenAI vector embeddings down to 2D UMAP plots to verify if "Financial Documents" cluster away from "Medical Transcripts").
* **Outlier Detection:** Training an Autoencoder purely on normal operational server logs. When a cyberattack occurs, the anomalous log fails to compress through the bottleneck cleanly, spiking the reconstruction error $\|x - \hat{x}\|^2$.

---

### 6. Common Pitfalls & Misconceptions

* **Misinterpreting t-SNE Cluster Sizes:** A dense, tight cluster in t-SNE does *not* mean the underlying data has low variance; t-SNE adapts its local search radius dynamically, artificially expanding dense clusters and compressing sparse ones.
* **Using t-SNE/UMAP directly as ML Model Inputs:** Non-linear graph methods do not learn an explicit parametric function $f(x) = y$. You cannot easily project a real-time incoming streaming inference point into an existing fixed t-SNE map without re-running optimization.
* **The Perplexity Trap:** Setting t-SNE perplexity to 5 on a 10,000-point dataset will break the manifold into hundreds of artificial, disconnected micro-clusters.

---
---

## Module 3: Feature Selection

### 1. Motivation & Intuition

Dimensionality reduction creates *synthetic* features out of combinations of raw inputs. **Feature Selection** discards columns entirely, retaining a pure subset of the original variables.

**Why favor selection over transformation?**
1. **Regulatory Interpretability:** In credit risk modeling, if you reject a loan application, legal statutes (like the Fair Credit Reporting Act) require you to state the exact reason (e.g., "Debt-to-Income ratio > 45%"). You cannot legally cite "Principal Component 4."
2. **Inference Latency & Cost:** If a predictive medical model relies on 50 routine blood markers and 1 expensive DNA sequencing metric, discovering that the model retains 98% accuracy without the DNA metric saves thousands of dollars per patient.

---

### 2. Conceptual Foundations