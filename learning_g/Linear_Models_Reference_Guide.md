# Linear Models: Reference Guide

---

# Linear Regression (Ordinary Least Squares)

## 1. Motivation & Intuition
Imagine you are a real estate agent trying to price a house. You know that larger houses generally sell for more money. If you plotted the size of 100 recently sold houses (x-axis) against their selling price (y-axis), you would see a cloud of points generally trending upward.

**Linear Regression** is the mathematical method of drawing a single straight line through this cloud that best represents the relationship.
* **The Goal:** Find the line that is "closest" to all points simultaneously.
* **The Prediction:** Once you have this line, if a new house comes on the market with a specific size, you look at the line to predict its price.
* **Real-World Connection:** In ML systems, this is used for "Continuous Value Prediction." Examples include predicting server latency based on load, forecasting sales revenue based on ad spend, or estimating the remaining battery life of a device.

## 2. Conceptual Foundations

### Key Components
* **Dependent Variable ($y$):** The target you want to predict (e.g., Price).
* **Independent Variables / Features ($X$):** The inputs used to make the prediction (e.g., Size, Number of Rooms).
* **Parameters / Coefficients ($\beta$):** The weights the model learns. A higher weight means that feature has a stronger impact on the output.
* **Residual ($e$):** The error. The vertical distance between the actual data point and the line.

### How it Works
The model assumes the world works like this:

$$
\text{Price} = \text{Base Price} + (\text{Weight} \times \text{Size}) + \text{Noise}
$$

The "Noise" accounts for randomness we cannot explain (e.g., a buyer really liked the garden). We try to find the Base Price (intercept) and Weight (slope) that minimize the total squared noise.

### Underlying Assumptions (The OLS Checklist)
For standard Ordinary Least Squares (OLS) to be statistically valid—and to satisfy the Gauss-Markov theorem as the Best Linear Unbiased Estimator (BLUE)—these assumptions must hold:
1. **Linearity:** The relationship between $X$ and $y$ is linear.
2. **Independence:** Observations are independent of each other.
3. **Homoscedasticity:** The variance of the errors is constant across all levels of the features. 
4. **Exogeneity:** The features are not correlated with the error term ($\text{Cov}(X, \epsilon) = 0$).
5. **Normality:** The error terms follow a Normal (Gaussian) distribution (required primarily for hypothesis testing and confidence intervals).

## 3. Mathematical Formulation

### Notation
* $y$: Vector of targets ($n \times 1$)
* $X$: Matrix of features ($n \times p$), where $n$ is samples and $p$ is features. A column of $1$s is appended for the intercept.
* $\beta$: Vector of coefficients ($p \times 1$)
* $\epsilon$: Vector of errors ($n \times 1$)

The model is expressed as:

$$
y = X\beta + \epsilon
$$

### The Cost Function
We minimize the **Sum of Squared Residuals (SSR)**:

$$
J(\beta) = \sum_{i=1}^{n} (y_i - x_i^T \beta)^2 = ||y - X\beta||^2_2
$$

### Deriving the Normal Equation
Expanding the matrix form of the cost function:

$$
J(\beta) = (y - X\beta)^T (y - X\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta
$$

Taking the gradient with respect to $\beta$:

$$
\nabla_{\beta} J(\beta) = -2X^T y + 2X^T X \beta
$$

Setting the gradient to zero to find the global minimum:

$$
2X^T X \beta = 2X^T y \implies X^T X \beta = X^T y
$$

**The Normal Equation solution:**

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

### Geometric Interpretation
The vector $y$ lives in an $n$-dimensional Euclidean space. The columns of $X$ span a $p$-dimensional subspace (a hyperplane). Because of noise, $y$ generally does not lie perfectly on this hyperplane. OLS finds the orthogonal projection of $y$ onto the subspace spanned by $X$. The residual vector $e = y - \hat{y}$ is strictly perpendicular to this feature plane.

## 4. Worked Example
**Scenario:** Predict Exam Score ($y$) based on Hours Studied ($x$).
* Student A: 1 hour $\rightarrow$ Score 2
* Student B: 2 hours $\rightarrow$ Score 3
* Student C: 3 hours $\rightarrow$ Score 5

**Step 1: Setup Matrices**

$$
X = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad y = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}
$$

**Step 2: Compute $X^T X$**

$$
X^T X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}
$$

**Step 3: Compute $X^T y$**

$$
X^T y = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix} = \begin{bmatrix} 10 \\ 23 \end{bmatrix}
$$

**Step 4: Solve for $\beta$**
First, invert $(X^T X)$:

$$
(X^T X)^{-1} = \frac{1}{(3)(14) - (6)(6)} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix}
$$

Multiply by $X^T y$:

$$
\hat{\beta} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} \begin{bmatrix} 10 \\ 23 \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 2 \\ 9 \end{bmatrix} \approx \begin{bmatrix} 0.33 \\ 1.50 \end{bmatrix}
$$

**Result:** $\text{Score} = 0.33 + 1.50 \times (\text{Hours})$.

## 5. Relevance to Machine Learning Practice
* **Usage:** Default baseline for tabular continuous regression tasks, econometric modeling, and A/B test inference.
* **Trade-offs:** Highly interpretable and computationally trivial ($O(p^2n + p^3)$), but incapable of modeling non-linear interactions natively without manual basis expansions (e.g., polynomial features).

## 6. Common Pitfalls
* **Multicollinearity:** When features are highly correlated, $X^T X$ approaches singularity (determinant $\approx 0$), causing matrix inversion to fail or resulting in wildly unstable coefficient variance.
* **Outlier Sensitivity:** Because errors are squared, a single extreme outlier exerts massive leverage over the slope of the line.

---

# Logistic Regression

## 1. Motivation & Intuition
Linear regression predicts unbounded continuous numbers. When predicting binary classifications (e.g., Will a user click an ad? Yes/No), a standard linear line might output values like $1.4$ or $-0.2$, which violate the laws of probability.

**Logistic Regression** wraps the linear model in a squashing function that guarantees the output remains strictly between $0$ and $1$, interpreting the result as a class probability.

## 2. Conceptual Foundations
* **Log-Odds (Logit):** Instead of mapping features directly to a probability $p$, the linear combination maps to the natural logarithm of the odds ratio.
* **Sigmoid Function:** The non-linear activation curve that maps $(-\infty, +\infty) \rightarrow (0, 1)$.
* **Decision Boundary:** The hyperplane separating the classes. Though the output probability curve is S-shaped, the underlying decision boundary in feature space remains strictly linear.

## 3. Mathematical Formulation

### The Sigmoid Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

### The Predictive Model

$$
P(y=1 | x) = \hat{y} = \sigma(X\beta) = \frac{1}{1 + e^{-X\beta}}
$$

### Log-Odds Interpretation
Rearranging the probability function demonstrates the linear backbone:

$$
\ln \left( \frac{p}{1 - p} \right) = X\beta
$$

### Cost Function (Binary Cross-Entropy / Log-Loss)
Applying standard Least Squares to a sigmoid function results in a non-convex loss landscape riddled with local minima. Instead, we maximize the log-likelihood of the Bernoulli distribution, which yields the convex Binary Cross-Entropy loss:

$$
J(\beta) = - \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1 - \hat{y}_i) \right]
$$

Because $\nabla_{\beta} J(\beta) = X^T(\hat{y} - y)$ lacks a closed-form algebraic solution, parameters must be learned iteratively via Gradient Descent or Newton-Raphson optimization (Iteratively Reweighted Least Squares).

## 4. Worked Example
**Scenario:** Model predicting if a student passes ($1$) or fails ($0$) an exam based on hours studied ($x$).
Given learned weights: $\beta_0 = -3.0$ (Intercept), $\beta_1 = 1.0$ (Slope).

**Inference:** A student studies for 4 hours.

$$
z = -3.0 + (1.0 \times 4) = 1.0
$$

$$
P(\text{Pass}) = \frac{1}{1 + e^{-1.0}} = \frac{1}{1 + 0.3679} \approx 0.731
$$

With a classification threshold of $0.5$, the model predicts **Class 1 (Pass)**.

## 5. Relevance to ML Practice
* **Usage:** Industry standard baseline for CTR prediction, fraud detection, and credit scoring pipelines.
* **Multiclass Extensions:**
  * **One-vs-Rest (OvR):** Trains $K$ separate binary classifiers.
  * **Multinomial (Softmax):** Generalizes the sigmoid to a probability distribution across $K$ mutually exclusive classes:
  
  $$P(y=k | x) = \frac{e^{x^T \beta_k}}{\sum_{j=1}^K e^{x^T \beta_j}}$$

## 6. Common Pitfalls
* **Complete Separation:** If a feature perfectly separates the classes (e.g., all instances where $x > 5$ are Class 1), the maximum likelihood optimizer will push the weights $\beta \rightarrow \infty$ to drive probabilities to absolute $1$ or $0$, causing numerical overflow.
* **Class Imbalance:** If positive samples represent 0.1% of the data, the model can achieve 99.9% accuracy by collapsing its intercept to predict zero for all inputs.

---

# Regularization Techniques

## 1. Motivation & Intuition
When a linear model contains too many features relative to the number of observations ($p \approx n$), it memorizes sampling noise rather than underlying signal. This represents **High Variance (Overfitting)**.

Regularization adds a penalty term to the cost function that charges a "budget" for large weights, forcing the model to favor simpler, smoother functions that generalize better to unseen data.

## 2. Mathematical Formulation

$$
J_{\text{regularized}}(\beta) = J_{\text{data}}(\beta) + \lambda \cdot R(\beta)
$$

Where $\lambda \ge 0$ is a hyperparameter dictating penalty strength.

### Ridge Regression ($L_2$)
* **Penalty:** Squared Euclidean norm of the weights.

$$
R(\beta) = ||\beta||_2^2 = \sum_{j=1}^{p} \beta_j^2
$$

* **Closed-Form Solution:** $$\hat{\beta}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T y$$

Adding $\lambda$ to the diagonal of $X^T X$ guarantees the matrix is strictly positive-definite and invertible, resolving multicollinearity.
* **Bayesian Interpretation:** Equivalent to Maximum A Posteriori (MAP) estimation assuming a **Gaussian Prior** ($\beta \sim \mathcal{N}(0, \tau^2)$) centered at zero.

### Lasso Regression ($L_1$)
* **Penalty:** Absolute sum of the weights.

$$
R(\beta) = ||\beta||_1 = \sum_{j=1}^{p} |\beta_j|
$$

* **Properties:** Forces irrelevant feature weights to collapse to **exactly zero**, acting as an automated feature selection mechanism. Cannot be solved analytically; requires Coordinate Descent.
* **Bayesian Interpretation:** Equivalent to MAP estimation assuming a **Laplace Prior**, which features a sharp, non-differentiable spike directly at zero.

### Elastic Net
Combines both penalties to stabilize Lasso when features are highly correlated:

$$
R(\beta) = \lambda_1 ||\beta||_1 + \lambda_2 ||\beta||_2^2
$$

## 3. Geometric Intuition
The unpenalized OLS loss surface forms elliptical contours centered at the unconstrained minimum. Regularization defines a constraint region centered at the origin $(0,0)$:
* **Ridge ($L_2$):** The constraint boundary is a smooth **Circle**. The expanding OLS ellipses touch the circle at an arbitrary point on its curve, shrinking weights toward zero without touching the axes.
* **Lasso ($L_1$):** The constraint boundary is a **Diamond** with sharp vertices lying directly on the coordinate axes. When the expanding OLS ellipse hits the diamond, it intersects a sharp vertex with high probability. A vertex on an axis means one or more coordinates are exactly $0$.

## 4. Relevance to ML Practice
* **Data Scaling Requirement:** Features **must** be standardized ($\mu=0, \sigma=1$) prior to regularization. If Feature A ranges from $[1, 10]$ and Feature B ranges from $[1000, 100000]$, the raw penalty will unfairly suppress Feature B simply due to unit scale.

---

# Interview Preparation Section

## Foundational Questions

### Q1: Compare the assumptions required for standard OLS inference versus pure ML prediction.
**Answer:** For statistical inference (p-values, confidence intervals), OLS requires linearity, strict independence, homoscedasticity, exogeneity, and normally distributed errors. For pure machine learning prediction on test data, the Gauss-Markov properties are secondary; we care only that the data distribution remains stationary ($P_{\text{train}}(X,y) \approx P_{\text{test}}(X,y)$) and that the linear approximation captures enough signal to minimize generalization error. Violating normality or homoscedasticity degrades confidence intervals but often leaves predictive accuracy unaffected.

### Q2: Why do we exclude the intercept term $\beta_0$ from regularization penalties?
**Answer:** The intercept shifts the regression hyperplane up or down to match the mean of the target distribution $\bar{y}$. Penalizing $\beta_0$ forces the model surface toward the origin $(0,0)$ along the y-axis. This introduces systematic bias into predictions for datasets where the target variable has a large baseline mean, without providing any reduction in variance or feature complexity.

## Mathematical Questions

### Q3: Derive the Ridge Regression closed-form solution from its objective function.
**Answer:** The Ridge objective function is:

$$
J(\beta) = (y - X\beta)^T (y - X\beta) + \lambda \beta^T \beta
$$

Expanding terms:

$$
J(\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X \beta + \lambda \beta^T \beta
$$

Taking the gradient with respect to $\beta$:

$$
\nabla_{\beta} J(\beta) = -2X^T y + 2X^T X \beta + 2\lambda \beta
$$

Setting the gradient to zero:

$$
2(X^T X + \lambda I)\beta = 2X^T y \implies \hat{\beta}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T y
$$

### Q4: Prove that $L_2$ regularization is mathematically equivalent to placing a zero-mean Gaussian prior over the parameters.
**Answer:**
By Bayes' Theorem, the posterior probability of weights given data is:

$$
P(\beta | X, y) \propto P(y | X, \beta) \cdot P(\beta)
$$

Assume the likelihood is Gaussian with variance $\sigma^2$:

$$
P(y | X, \beta) \propto \exp \left( -\frac{1}{2\sigma^2} ||y - X\beta||_2^2 \right)
$$

Assume the prior over $\beta$ is independent zero-mean Gaussian with variance $\tau^2$:

$$
P(\beta) \propto \exp \left( -\frac{1}{2\tau^2} ||\beta||_2^2 \right)
$$

Multiplying these expressions combines their exponents. To maximize the posterior (MAP), we minimize the negative log-posterior:

$$
-\ln P(\beta | X, y) = \frac{1}{2\sigma^2} ||y - X\beta||_2^2 + \frac{1}{2\tau^2} ||\beta||_2^2 + C
$$

Multiplying through by $2\sigma^2$ yields the exact Ridge formulation where $\lambda = \frac{\sigma^2}{\tau^2}$:

$$
\text{Loss} = ||y - X\beta||_2^2 + \lambda ||\beta||_2^2
$$

## Applied Questions

### Q5: You are deploying a production CTR model with 10 million sparse features. Memory constraints allow deploying only 50,000 non-zero weights. How do you design the training pipeline?
**Answer:**
1. **Model Selection:** Use Logistic Regression optimized with $L_1$ (Lasso) penalty to enforce strict weight sparsity.
2. **Optimizer:** Standard gradient descent cannot handle $L_1$'s non-differentiability at zero well at scale. Use **Follow-The-Regularized-Leader Proximal (FTRL-Proximal)**, an online learning algorithm specifically engineered to induce true zero weights on sparse streaming datasets.
3. **Hyperparameter Tuning:** Run a binary search sweep over the $L_1$ penalty parameter $\lambda$. Increase $\lambda$ until the number of non-zero parameters drops safely below the 50,000 memory threshold.
4. **Fallback:** If performance degrades too heavily at 50k features, apply Feature Hashing (the Hashing Trick) to compress the input dimensionality prior to training.

### Q6: You train a linear regression model and notice that the training error is high, and the validation error is virtually identical to the training error. Diagnose the system and propose remediations.
**Answer:**
This is a classic signature of **High Bias (Underfitting)**. The linear model lacks the capacity to capture the underlying deterministic structure of the data. 
* **Remedy 1 (Feature Engineering):** Introduce non-linear domain features, interaction terms ($x_1 \cdot x_2$), or polynomial basis expansions ($x^2, x^3$).
* **Remedy 2 (Relax Constraints):** If regularization was applied, drop $\lambda \rightarrow 0$.
* **Remedy 3 (Model Complexity):** Upgrade to a non-linear estimator capable of partitioning complex boundaries (e.g., Gradient Boosted Decision Trees or Neural Networks).

## Debugging & Failure Modes

### Q7: A colleague trains a Lasso model on a dataset containing 10,000 identical duplicate copies of a single useful feature $x_1$. What happens to the learned coefficients?
**Answer:**
Lasso ($L_1$) suffers from extreme instability in the presence of severe multicollinearity. Because all 10,000 features provide identical loss reductions, the subgradient optimization algorithm will arbitrarily select *one* instance of the feature, assign it the full necessary coefficient weight, and zero out the remaining 9,999 identical copies. Which specific duplicate gets selected is entirely non-deterministic and depends on floating-point rounding errors or feature column ordering. 

### Q8: During production monitoring, your linear regression pricing model suddenly starts outputting negative price predictions for certain valid user inputs. What broke?
**Answer:**
1. **Domain Covariate Shift:** The live data stream has drifted outside the spatial distribution of the training data (extrapolation). If the model learned a negative slope for a particular feature (e.g., building age), extreme real-world values for that feature can drag the entire linear summation below zero.
2. **Target Formulation Error:** The model was trained on raw linear target values rather than a mathematically constrained space. To prevent negative outputs in continuous regression natively, the system should be retrained to predict log-transformed targets ($\ln(y)$), mapping predictions back to real space via exponentiation ($\hat{y} = e^{X\beta}$).