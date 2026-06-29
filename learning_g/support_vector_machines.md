# Support Vector Machines (SVM)

## 1. Motivation & Intuition

### The Problem of Separation
Imagine you have a set of red and blue balls lying on a table. Your goal is to separate them using a straight stick.

* **Scenario A:** You place the stick such that it just barely touches the red balls on one side.
* **Scenario B:** You place the stick such that it sits comfortably in the middle of the gap between the red and blue balls.

If you add a new red ball to the table, the stick in **Scenario A** is likely to misclassify it because it was "hugging" the original data too closely. The stick in **Scenario B**, however, has a "buffer zone" (margin) that allows for some variation in where new balls fall.

### The "Widest Street" Analogy

The core intuition of SVM is **Margin Maximization**.

1. Think of the separating line not as a thin thread, but as a **street**.
2. The "width" of the street is determined by the distance to the closest data points on either side.
3. SVM tries to find the widest possible street that separates the two classes.
4. The data points that lie strictly on the "gutters" (edges) of the street are the only ones that matter. These are called **Support Vectors**.

### Connection to Machine Learning
In ML terms, "widest street" translates to **robustness**. A classifier with a large margin is less likely to overfit. It generalizes better because it is not overly sensitive to the precise location of training points, provided they are not the critical support vectors.

---

## 2. Conceptual Foundations

### Key Components

1. **Hyperplane:** In 2D, this is a line. In 3D, a flat sheet. In $n$-dimensions, a hyperplane is a subspace of dimension $n-1$ that divides the space into two halves. This is our decision boundary.
2. **Margin:** The perpendicular distance between the decision boundary and the closest data point(s).
3. **Support Vectors:** The specific data points that lie closest to the decision boundary. They "support" or define the margin. If you remove other points, the boundary doesn't change. If you move a support vector, the boundary moves.
4. **Hard Margin vs. Soft Margin:**
   * **Hard Margin:** Assumes data is perfectly separable (no errors allowed). The "street" must be completely clean.
   * **Soft Margin:** Acknowledges that real data is noisy. It allows some points to violate the margin (be inside the street or on the wrong side) to achieve a better overall fit.

### Underlying Assumptions
* **Linearity (initially):** Standard SVM assumes classes can be separated by a linear boundary. 
* **Binary Classification:** SVM is natively a binary (two-class) classifier.
* **Feature Scaling:** SVM is distance-based. If one feature ranges from 0–1 and another from 0–1000, the larger feature will dominate the distance calculation. **Scaling is mandatory.**

---

## 3. Mathematical Formulation

### 3.1 The Hyperplane
We define a hyperplane using a weight vector $\mathbf{w}$ and a bias $b$. For a new input vector $\mathbf{x}$, the decision function is:

$$
f(\mathbf{x})=\mathbf{w}^T\mathbf{x}+b
$$

* If $f(\mathbf{x}) > 0$, predict Class +1.
* If $f(\mathbf{x}) < 0$, predict Class -1.
* If $f(\mathbf{x}) = 0$, the point lies exactly on the hyperplane.

### 3.2 The Geometric Margin
The distance from any point $\mathbf{x}_i$ to the hyperplane defined by $(\mathbf{w}, b)$ is given by geometry as:

$$
\text{Distance}=\frac{|\mathbf{w}^T\mathbf{x}_i+b|}{||\mathbf{w}||}
$$

We want to enforce that for all sample points, the sign of the result matches the label $y_i$ (where $y_i\in\{-1, +1\}$). To fix the scale of $\mathbf{w}$ and $b$, we enforce that for the closest points (support vectors), $|\mathbf{w}^T\mathbf{x}+b|=1$.

Thus, for all data points:

$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1
$$

With this constraint, the distance of the support vectors to the hyperplane becomes:

$$
\text{Margin}=\frac{1}{||\mathbf{w}||}
$$

### 3.3 The Primal Optimization Problem (Hard Margin)
To maximize the margin ($\frac{1}{||\mathbf{w}||}$), we can equivalently minimize $||\mathbf{w}||$, or mathematically more convenient, minimize $\frac{1}{2}||\mathbf{w}||^2$.

**Objective:**

$$
\min_{\mathbf{w}, b}\frac{1}{2}||\mathbf{w}||^2
$$

**Subject to constraints:**

$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1\quad\forall i
$$

This is a **Quadratic Programming (QP)** problem with linear constraints. It is convex, meaning any local minimum is the global minimum.

### 3.4 Soft Margin (The $C$ Parameter)
Real-world data is rarely perfectly separable. We introduce **slack variables** $\xi_i$ which measure how much a point violates the margin.

* If $\xi_i=0$, the point is correctly classified and outside the margin.
* If $0<\xi_i<1$, the point is correctly classified but inside the margin.
* If $\xi_i>1$, the point is misclassified.

**New Objective:**

$$
\min_{\mathbf{w}, b, \xi}\frac{1}{2}||\mathbf{w}||^2+C\sum_{i=1}^N\xi_i
$$

**Subject to:**

$$
y_i(\mathbf{w}^T\mathbf{x}_i+b)\ge 1-\xi_i
$$

$$
\xi_i\ge 0
$$

* **$C$ (Cost Hyperparameter):** * **Large $C$:** High penalty for errors. Tries hard not to misclassify. Low bias, High variance (risk of overfitting).
  * **Small $C$:** Low penalty for errors. Accepts misclassifications for a wider margin. High bias, Low variance (smoother boundary).

### 3.5 The Dual Formulation & Lagrange Multipliers
To solve this efficiently and enable the "Kernel Trick," we convert the Primal problem to the **Dual problem** using Lagrange Multipliers ($\alpha_i$).

We form the Lagrangian $L$:

$$
L(\mathbf{w}, b, \alpha)=\frac{1}{2}||\mathbf{w}||^2-\sum_{i=1}^N\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1]
$$

We minimize $L$ with respect to $\mathbf{w}$ and $b$, and maximize with respect to $\alpha$. Taking partial derivatives and setting them to zero yields:

1. $\mathbf{w}=\sum_{i=1}^N\alpha_i y_i\mathbf{x}_i$ *(The weight vector is a linear combination of support vectors)*.
2. $\sum_{i=1}^N\alpha_i y_i=0$.

Substituting these back into the Lagrangian gives the **Dual Objective**:

$$
\max_{\alpha}\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_j y_i y_j(\mathbf{x}_i^T\mathbf{x}_j)
$$

Subject to: $0\le\alpha_i\le C$ and $\sum\alpha_i y_i=0$.

**Crucial Insight:** The optimization depends *only* on the dot product between data points, $\mathbf{x}_i^T\mathbf{x}_j$.

### 3.6 Kernel Methods
When data is not linearly separable in its native space, we map the data $\mathbf{x}$ into a higher-dimensional space $\phi(\mathbf{x})$ where it becomes linearly separable. Calculating $\phi(\mathbf{x})$ explicitly is computationally expensive.

Since the Dual form only uses dot products, we replace $\mathbf{x}_i^T\mathbf{x}_j$ with a **Kernel Function**:

$$
K(\mathbf{x}_i, \mathbf{x}_j)=\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j)
$$

**Common Kernels:**

1. **Linear:** $K(x, z)=x^T z$
2. **Polynomial:** $K(x, z)=(x^T z+c)^d$
3. **Radial Basis Function (RBF) / Gaussian:** $K(x, z)=\exp(-\gamma||x-z||^2)$
   * Maps data to an infinite-dimensional space.
   * $\gamma$ controls the spread. High $\gamma$ = complex model (overfitting risk). Low $\gamma$ = simple model.

---

## 4. Worked Example (1D Toy Problem)

**Problem:** We have 3 points:
* $x_1=1$, Label $y_1=-1$
* $x_2=3$, Label $y_2=+1$
* $x_3=4$, Label $y_3=+1$

We want to find the Hard Margin hyperplane (a single point in 1D) defined by $w$ and bias $b$.

### Step 1: Inspect Visually
The decision boundary should sit halfway between the closest opposing points ($x_1=1$ and $x_2=3$). Therefore, the boundary should be at $x=2$.

### Step 2: Dual Formulation Calculation
We maximize:

$$
L_D(\alpha)=\alpha_1+\alpha_2+\alpha_3-\frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i x_j
$$

Subject to the constraint: 

$$
\sum\alpha_i y_i=-\alpha_1+\alpha_2+\alpha_3=0\implies\alpha_1=\alpha_2+\alpha_3
$$

**Observation:** $x_3$ (at position 4) lies safely behind $x_2$ (at position 3). It is not a support vector. By the sparsity property, $\alpha_3=0$. 

This simplifies our constraint to $\alpha_1=\alpha_2$. Let us call this shared value $\alpha$.

Substitute into the objective function:

$$
L_D=2\alpha-\frac{1}{2}[\alpha^2 y_1^2 x_1^2+\alpha^2 y_2^2 x_2^2+2\alpha^2 y_1 y_2 x_1 x_2]
$$

Using $x_1=1, x_2=3, y_1=-1, y_2=1$:

$$
L_D=2\alpha-\frac{1}{2}[\alpha^2(1)+\alpha^2(9)+2\alpha^2(-1)(1)(3)]
$$

$$
L_D=2\alpha-\frac{1}{2}[10\alpha^2-6\alpha^2]=2\alpha-2\alpha^2
$$

To maximize, take the derivative with respect to $\alpha$ and set it to zero:

$$
2-4\alpha=0\implies\alpha=0.5
$$

Thus: $\alpha_1=0.5,\ \alpha_2=0.5,\ \alpha_3=0$.

### Step 3: Recover $w$ and $b$

$$
w=\sum\alpha_i y_i x_i=(0.5)(-1)(1)+(0.5)(1)(3)+0
$$

$$
w=-0.5+1.5=1
$$

To find $b$, use any active support vector (e.g., $x_2$):

$$
y_2(w x_2+b)=1
$$

$$
1(1\cdot 3+b)=1\implies b=-2
$$

### Step 4: The Decision Function

$$
f(x)=1\cdot x-2
$$

The boundary is solved where $f(x)=0 \implies x=2$.
* Test $x_1=1 \implies 1-2 = -1$ (Class -1) $\to$ **Correct**
* Test $x_2=3 \implies 3-2 = +1$ (Class +1) $\to$ **Correct**

---

## 5. Relevance to Machine Learning Practice

### When to use SVMs
1. **Small-to-Medium Tabular Datasets:** SVMs (especially kernelized) struggle when sample sizes grow ($N>100,000$) because the kernel matrix scales quadratically at $O(N^2)$.
2. **High-Dimensional Spaces:** SVMs excel when the number of features exceeds the number of samples (e.g., text classification, genomics) because margin maximization inherently guards against overfitting.
3. **Anomaly Detection:** One-Class SVMs serve as a standard baseline for unsupervised outlier detection.

### Trade-offs
| Dimension | Advantage | Disadvantage |
| :--- | :--- | :--- |
| **Optimization** | Convex objective guarantees finding the global minimum. | Slow training convergence on large sample sets. |
| **Robustness** | Decision boundary ignores outliers far from the margin. | Highly sensitive to unscaled features and bad $C$ values. |
| **Output** | Memory efficient at inference (stores only support vectors). | Outputs raw geometric distances, not true probabilities. |

### Multiclass Strategies
Because SVMs are natively binary, handling $K$ classes requires decomposition:
* **One-vs-Rest (OvR):** Trains $K$ separate classifiers. Highest score wins. *(Industry standard)*.
* **One-vs-One (OvO):** Trains $K(K-1)/2$ classifiers for every possible class pair. Majority vote wins.
* **Directed Acyclic Graph (DAG SVM):** Organizes pairwise OvO classifiers into a routing tree to speed up inference time.

---

## 6. Common Pitfalls & Misconceptions

1. **Forgetting to Scale Features:** This is the single most common failure mode. If Feature A ranges from [0, 1] and Feature B ranges from [0, 10000], the Euclidean distance calculation will be completely hijacked by Feature B. **Always standardize inputs (`mean=0, var=1`) prior to fitting.**
2. **Treating Decision Values as Probabilities:** An SVM decision value of `+2.4` means the point lies 2.4 margin units away from the boundary; it does *not* mean there is a 99% probability of belonging to Class +1. Converting distances to probabilities requires post-hoc fitting (e.g., Platt Scaling).
3. **Overusing Non-Linear Kernels:** Applying an RBF kernel to sparse, ultra-high-dimensional data (like TF-IDF text vectors) adds massive computational overhead for zero gain. High-dimensional data is almost always linearly separable already; use a Linear Kernel.

---

## Interview Preparation Section

### Foundational Questions

#### Q1: Explain the concept of the "Margin" in SVM. Why do we want to maximize it?
**Answer:** The margin is the perpendicular distance between the decision boundary and the closest training samples. We maximize it because statistical learning theory (specifically VC dimension bounds) demonstrates that maximizing the geometric margin minimizes the upper bound of the model's generalization error. It builds an explicit buffer against input noise.

#### Q2: What is the "Sparsity Property" of Support Vector Machines?
**Answer:** Sparsity refers to the mathematical phenomenon where the vast majority of training points are assigned a Lagrange multiplier of $\alpha_i=0$. The final decision boundary relies *exclusively* on the tiny subset of points where $\alpha_i>0$ (the support vectors). This makes inference exceptionally lightweight, as non-support vectors can be purged from memory after training.

---

### Mathematical Questions

#### Q3: Why do we optimize the Dual problem rather than the Primal problem?
**Answer:** 1. **Computational Complexity:** The optimization complexity of the Primal form scales with the feature dimension $D$, whereas the Dual form scales with the number of samples $N$. When $D \gg N$, the Dual is vastly faster to compute.
2. **The Kernel Trick:** The Dual formulation groups all feature vectors into dot products ($\mathbf{x}_i^T\mathbf{x}_j$). This allows us to substitute an arbitrary kernel function $K(\mathbf{x}_i, \mathbf{x}_j)$ directly into the objective, projecting data into infinite-dimensional spaces without ever explicitly computing the mapped coordinates $\phi(\mathbf{x})$.

#### Q4: State the Karush-Kuhn-Tucker (KKT) Complementary Slackness condition for a Soft Margin SVM and explain its physical meaning.
**Answer:** The complementary slackness condition is formulated as:

$$
\alpha_i[y_i(\mathbf{w}^T\mathbf{x}_i+b)-1+\xi_i]=0
$$

For this product to equal zero, at least one term must be zero:
* If $\alpha_i=0$, the sample point sits safely outside the margin.
* If $\alpha_i>0$, then $y_i(\mathbf{w}^T\mathbf{x}_i+b)=1-\xi_i$, forcing the sample to sit either strictly on the margin boundary ($\xi_i=0$) or inside the margin violation zone ($\xi_i>0$). This mathematically proves that only support vectors dictate model updates.

#### Q5: Mathematically prove why the RBF Kernel projects data into an infinite-dimensional feature space.
**Answer:** The standard 1D RBF kernel is expressed as $K(x, z)=\exp(-\gamma(x-z)^2)$. Expanding the squared distance gives:

$$
K(x, z)=\exp(-\gamma x^2)\exp(-\gamma z^2)\exp(2\gamma xz)
$$

Using the Taylor Series expansion for the exponential term $\exp(2\gamma xz) = \sum_{k=0}^{\infty} \frac{(2\gamma xz)^k}{k!}$, we can rewrite the kernel as an infinite sum of polynomial inner products:

$$
K(x, z)=\sum_{k=0}^{\infty} \left[ \left(\sqrt{\frac{(2\gamma)^k}{k!}}\exp(-\gamma x^2)x^k\right) \left(\sqrt{\frac{(2\gamma)^k}{k!}}\exp(-\gamma z^2)z^k\right) \right]
$$

This matches the dot product definition $\phi(x)^T\phi(z)$, where the feature mapping $\phi(x)$ contains an infinite number of terms corresponding to every polynomial degree $k \in [0, \infty)$.

---

### Applied & Scenario Questions

#### Q6: You train an RBF kernel SVM. Your training accuracy is 99.8%, but your cross-validation accuracy is 54.1%. Diagnose the problem and prescribe solutions.
**Answer:** The model is suffering from severe variance (overfitting). The decision boundary has warped into tight, isolated hyper-spheres around individual training examples.

**Prescriptions:**
1. **Lower $\gamma$ (Gamma):** Reduce the kernel coefficient. A high gamma shrinks the radius of influence for support vectors; lowering it forces the kernel to consider broader neighborhoods, smoothing the decision surface.
2. **Lower $C$ (Cost):** Decrease the penalty for misclassifications. This allows more training points to violate the margin, trading a slight increase in training error for a smoother, more generalizable boundary.

#### Q7: How do you handle severe class imbalance (e.g., 99% Negative, 1% Positive) inside a standard SVM architecture?
**Answer:** Standard SVM minimizes global classification error, meaning it will happily predict the majority class 100% of the time to achieve 99% accuracy.
* **Approach A (Cost-Sensitive Learning):** Pass asymmetrical weights to the objective function (`class_weight='balanced'`). This scales the slack penalty $C$ inversely proportional to class frequencies ($C_{pos} \gg C_{neg}$).
* **Approach B (One-Class SVM):** If the minority class represents anomalies rather than a coherent structural cluster, discard the minority labels entirely. Train an unsupervised One-Class SVM strictly on the 99% baseline data to map a tight geometric boundary around "normal" operations.

---

### Debugging & Failure Modes

#### Q8: You deploy a trained SVM model to production. During inference, latency spikes to 450ms per request. What property of the model caused this, and how do you resolve it?
**Answer:** The model contains an excessively large number of Support Vectors. Because SVM inference requires computing the dot product of the live input against *every stored support vector*, inference complexity is $O(M \cdot D)$ where $M$ is the number of support vectors. 

**Resolutions:**
1. Retrain the model with a smaller $C$ parameter to force a wider margin, which naturally purges borderline support vectors.
2. If using an RBF kernel, attempt to replace it with a Linear Kernel or an approximated feature map (like **Fourier / Nyström approximations**), which allows collapsing the weights back into a static $1 \times D$ vector $\mathbf{w}$.

#### Q9: You run a kernelized SVM on a tabular dataset containing $N=600,000$ rows. The training script runs out of RAM and crashes after 20 minutes. Why?
**Answer:** To solve the exact Dual optimization, the solver must instantiate the Gram Matrix (Kernel Matrix) in memory. This requires calculating pairwise dot products for every combination of training instances. 

A $600,000 \times 600,000$ matrix of 64-bit floats requires roughly **2.88 Terabytes of RAM** ($600000^2 \times 8\text{ bytes}$). Standard kernel SVM cannot scale to large sample sizes; you must pivot to Stochastic Gradient Descent (e.g., `SGDClassifier(loss='hinge')`) or mini-batch neural networks.