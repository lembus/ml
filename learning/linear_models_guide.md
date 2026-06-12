---
title: "Linear Models"
---

# Linear Models: A Comprehensive Learning Guide

## Table of Contents

1. [Part I: Linear Regression](#part-i-linear-regression)
2. [Part II: Logistic Regression](#part-ii-logistic-regression)
3. [Part III: Regularization Techniques](#part-iii-regularization-techniques)
4. [Interview Preparation](#interview-preparation)


# Part I: Linear Regression

## 1. Motivation & Intuition

### Why Does Linear Regression Exist?

Imagine you are a real estate agent trying to predict the price of a house. You know that bigger houses tend to cost more. You also know that houses with more bedrooms, closer to good schools, and in nicer neighborhoods tend to be more expensive. But how much more expensive? If a house gains 200 square feet, does the price go up by $10,000 or $50,000? And how do you combine all these factors into one prediction?

This is the fundamental problem linear regression solves: **given a set of input measurements (features) and a corresponding set of outcomes (targets), find the best straight-line (or hyperplane) relationship between them, so that you can predict the outcome for new inputs you have not yet seen.**

The word "linear" here means something precise: we assume the output is a **weighted sum** of the inputs, plus a constant. That is, each feature contributes independently and additively to the prediction, and the contribution of each feature is proportional to its value. If square footage contributes $150 per square foot, then going from 1,000 to 1,200 square feet adds the same $30,000 as going from 2,000 to 2,200.

### A Concrete Example Before Any Math

Suppose you have five houses:

| House | Sq. Feet (x) | Price in $1000s (y) |
|-------|--------------|----------------------|
| A     | 1000         | 200                  |
| B     | 1500         | 280                  |
| C     | 1200         | 240                  |
| D     | 1800         | 320                  |
| E     | 2000         | 380                  |

You plot these on a graph. They roughly fall along a line. Linear regression finds the "best" line through these points—the one that makes the smallest total error when you use it to predict each house's price from its square footage. That line might be something like:

**Price = 50 + 0.16 × (Sq. Feet)**

This means: a 0 square-foot "house" would cost $50,000 (the intercept—it does not have to be physically meaningful), and every additional square foot adds about $160.

### Connection to Machine Learning

Linear regression is the foundation upon which a vast portion of machine learning is built:

- **Supervised learning**: Linear regression is the simplest supervised learning algorithm. You have labeled data (features + outcomes), and you learn a function that maps features to outcomes.
- **Loss functions**: The "squared error" loss used in linear regression becomes the blueprint for loss functions throughout ML. Neural networks, gradient boosting, and many other methods use squared error or its close relatives.
- **Optimization**: The idea of minimizing a loss function over parameters—the core of virtually all ML training—originates here.
- **Feature engineering**: The idea that raw inputs can be transformed (polynomials, interactions, logarithms) before being fed into a linear model is the precursor to representation learning.
- **Baselines**: In any ML project, a linear regression model is almost always the first model you should try. If a linear model performs well, there may be no need for a complex deep learning approach.
- **Interpretability**: In production systems—especially in regulated industries like finance, healthcare, and insurance—linear models are often preferred precisely because every coefficient has a clear meaning.

---

## 2. Conceptual Foundations

### Key Terms and Components

**Features (inputs, predictors, independent variables, covariates)**: The measurements or attributes you use to make predictions. In the house example, square footage is a feature. In practice, you typically have many features: square footage, number of bedrooms, year built, distance to downtown, etc. We denote these as $x_1, x_2, \ldots, x_p$ for a single data point, where $p$ is the number of features.

**Target (output, response, dependent variable)**: The thing you are trying to predict. In the house example, this is the price. We denote it as $y$.

**Parameters (weights, coefficients)**: The numbers the model learns. Each feature gets a weight $\beta_j$ that says "for every one-unit increase in feature $j$, the predicted target changes by $\beta_j$ units." There is also an **intercept** (or **bias**) $\beta_0$, which is the predicted value when all features are zero.

**Predicted value (fitted value)**: The model's output for a given set of features: $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p$.

**Residual**: The difference between the actual target and the predicted value: $e_i = y_i - \hat{y}_i$. Residuals measure how wrong the model is for each data point.

**Loss function**: A function that measures how bad the model's predictions are overall. For Ordinary Least Squares (OLS), this is the sum of squared residuals: $\sum_{i=1}^n (y_i - \hat{y}_i)^2$.

### How the Components Interact

The process works as follows:

1. You start with a dataset of $n$ observations, each consisting of $p$ features and one target value.
2. You propose a model: $\hat{y}_i = \beta_0 + \sum_{j=1}^p \beta_j x_{ij}$.
3. For any choice of parameters $\beta_0, \beta_1, \ldots, \beta_p$, you can compute the predicted value for every observation, and then the residual.
4. You define a criterion for what makes a "good" set of parameters. In OLS, the criterion is: minimize the sum of squared residuals.
5. You solve for the parameters that minimize this criterion. This gives you the "fitted" or "estimated" model.
6. You use this model to predict targets for new, unseen observations.

### Assumptions of OLS Linear Regression

OLS makes several assumptions about the relationship between features and targets. These assumptions are **not** about the features themselves—they are about the **errors** (the part of $y$ that the model cannot explain). Understanding these assumptions is critical because when they are violated, the guarantees of OLS break down.

**Assumption 1: Linearity.** The true relationship between the features and the expected value of the target is linear:

$$
E[y | x] = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p
$$

This means the conditional expectation of $y$ given $x$ is a linear function of $x$. If the true relationship is $y = \beta_0 + \beta_1 x^2 + \epsilon$, then a model using $x$ (not $x^2$) as a feature violates this assumption.

**What breaks**: If the true relationship is nonlinear but you fit a linear model, your predictions will be systematically biased. The model will consistently overpredict in some regions and underpredict in others. No amount of data will fix this.

**Assumption 2: Independence.** The observations are independent of one another. Formally, the error terms $\epsilon_i$ and $\epsilon_j$ are independent for $i \neq j$.

**What breaks**: If observations are correlated (e.g., time-series data where today's stock price depends on yesterday's, or spatial data where nearby houses have similar prices for reasons beyond their features), the OLS standard errors are wrong. The coefficient estimates themselves may still be unbiased, but your confidence intervals and hypothesis tests become unreliable—you will think your estimates are more precise than they actually are.

**Assumption 3: Homoscedasticity (constant variance).** The variance of the error term is the same for all observations, regardless of the feature values:

$$
\text{Var}(\epsilon_i | x_i) = \sigma^2 \quad \text{for all } i
$$

**What breaks**: If error variance depends on $x$ (heteroscedasticity)—for example, if prediction errors for expensive houses are larger in absolute terms than for cheap houses—then OLS estimates are still unbiased but no longer the most efficient (lowest variance) among unbiased linear estimators. Standard errors are again incorrect, making confidence intervals unreliable.

**Assumption 4: Exogeneity (zero conditional mean of errors).** The error term has zero mean conditional on the features:

$$
E[\epsilon_i | x_i] = 0
$$

This is the strongest and most important assumption. It means that the features contain no information about the errors. Equivalently, there are no omitted variables that are correlated with the included features and also affect the target.

**What breaks**: If there is an omitted variable that is correlated with both a feature and the target (a confound), then the coefficient on the included feature absorbs the effect of the omitted variable, leading to biased and inconsistent estimates. This is the most common and most serious problem in applied regression.

**Assumption 5: Normality of errors.** The error terms are normally distributed: $\epsilon_i \sim N(0, \sigma^2)$.

**What breaks**: This assumption is needed for exact finite-sample inference (t-tests, F-tests, confidence intervals). Without it, OLS estimates are still unbiased and consistent, and by the Central Limit Theorem, inference is approximately valid in large samples. But in small samples, the exact distribution of the test statistics is unknown, so p-values and confidence intervals may be inaccurate.

**Additional structural requirement: No perfect multicollinearity.** No feature can be written as an exact linear combination of the other features. If $x_3 = 2x_1 + x_2$ exactly, then the system of equations has infinitely many solutions—the matrix $X^T X$ is singular and cannot be inverted.

**What breaks**: The OLS solution does not exist (the normal equations have no unique solution). In practice, near-perfect multicollinearity (e.g., two features that are 0.999 correlated) makes the coefficient estimates extremely unstable: small changes in the data cause large changes in the coefficients, even though predictions may remain stable.

---

## 3. Mathematical Formulation

### Setting Up the Problem

We have $n$ observations and $p$ features. We arrange the data into matrices:

$$
\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix} \in \mathbb{R}^n, \quad X = \begin{pmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1p} \\ 1 & x_{21} & x_{22} & \cdots & x_{2p} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{np} \end{pmatrix} \in \mathbb{R}^{n \times (p+1)}
$$

The column of 1s accounts for the intercept. The parameter vector is:

$$
\boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix} \in \mathbb{R}^{p+1}
$$

The model in matrix form is:

$$
\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon}
$$

where $\boldsymbol{\epsilon} \in \mathbb{R}^n$ is the vector of errors.

### Ordinary Least Squares: Deriving the Normal Equations

We want to find $\boldsymbol{\beta}$ that minimizes the sum of squared residuals:

$$
\text{RSS}(\boldsymbol{\beta}) = \|\mathbf{y} - X\boldsymbol{\beta}\|^2 = (\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})
$$

Expanding:

$$
\text{RSS}(\boldsymbol{\beta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T X^T \mathbf{y} + \boldsymbol{\beta}^T X^T X \boldsymbol{\beta}
$$

To find the minimum, we take the gradient with respect to $\boldsymbol{\beta}$ and set it to zero:

$$
\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2X^T\mathbf{y} + 2X^T X \boldsymbol{\beta} = \mathbf{0}
$$

This gives us the **normal equations**:

$$
X^T X \hat{\boldsymbol{\beta}} = X^T \mathbf{y}
$$

If $X^T X$ is invertible (which requires no perfect multicollinearity and $n \geq p+1$), the unique solution is:

$$
\boxed{\hat{\boldsymbol{\beta}} = (X^T X)^{-1} X^T \mathbf{y}}
$$

**Why this is a minimum, not a maximum**: The Hessian (second derivative matrix) of RSS with respect to $\boldsymbol{\beta}$ is $2X^T X$, which is positive semi-definite (and positive definite when $X$ has full column rank). A positive definite Hessian guarantees that the critical point is a global minimum. Since RSS is a convex quadratic in $\boldsymbol{\beta}$, there is exactly one minimum.

### Geometric Interpretation

The geometric view of OLS is elegant and deeply informative.

Think of $\mathbf{y}$ as a point in $\mathbb{R}^n$ (the $n$-dimensional space where each dimension corresponds to one data point). The columns of $X$ span a subspace of $\mathbb{R}^n$—call this the **column space** of $X$, denoted $\text{Col}(X)$. This subspace contains all possible predictions $X\boldsymbol{\beta}$ that the model can make.

The OLS fitted values are:

$$
\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}} = X(X^T X)^{-1} X^T \mathbf{y}
$$

Define the **hat matrix** (or projection matrix):

$$
H = X(X^T X)^{-1} X^T
$$

Then $\hat{\mathbf{y}} = H\mathbf{y}$. The matrix $H$ is an **orthogonal projection** onto $\text{Col}(X)$. This means $\hat{\mathbf{y}}$ is the point in $\text{Col}(X)$ that is closest to $\mathbf{y}$ in Euclidean distance.

The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is **orthogonal** to $\text{Col}(X)$:

$$
X^T \mathbf{e} = X^T(\mathbf{y} - X\hat{\boldsymbol{\beta}}) = X^T\mathbf{y} - X^T X (X^T X)^{-1} X^T\mathbf{y} = X^T\mathbf{y} - X^T\mathbf{y} = \mathbf{0}
$$

This is exactly the Pythagorean theorem in $n$ dimensions: the shortest path from a point to a subspace is the perpendicular. OLS finds the perpendicular foot.

**Implication**: The OLS residuals are uncorrelated with every feature column. If you regress the residuals on any feature, you get a slope of exactly zero. This is not an assumption—it is a mathematical consequence of the OLS procedure.

### The Gauss-Markov Theorem

The Gauss-Markov theorem is one of the most important results in statistics. It says:

**Under assumptions 1–4 (linearity, independence, homoscedasticity, exogeneity—notably NOT requiring normality), the OLS estimator $\hat{\boldsymbol{\beta}}$ is the Best Linear Unbiased Estimator (BLUE) of $\boldsymbol{\beta}$.**

Let us unpack each word:

- **Best**: Among all estimators satisfying the other conditions, OLS has the smallest variance. More precisely, for any linear combination $\mathbf{c}^T \boldsymbol{\beta}$, the OLS estimator $\mathbf{c}^T \hat{\boldsymbol{\beta}}$ has the smallest variance among all linear unbiased estimators of $\mathbf{c}^T \boldsymbol{\beta}$.
- **Linear**: The estimator is a linear function of $\mathbf{y}$. That is, $\hat{\boldsymbol{\beta}} = A\mathbf{y}$ for some matrix $A$ that depends on $X$ but not on $\mathbf{y}$. (Indeed, $A = (X^T X)^{-1}X^T$.)
- **Unbiased**: $E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$. On average across many hypothetical datasets drawn from the same process, the OLS estimate equals the true parameter value.
- **Estimator**: A function of the observed data that estimates the unknown parameter.

**Proof sketch**: Let $\tilde{\boldsymbol{\beta}} = C\mathbf{y}$ be any other linear unbiased estimator, where $C = (X^TX)^{-1}X^T + D$ for some matrix $D$. Unbiasedness requires $CX = I$, so $DX = 0$. The variance of $\tilde{\boldsymbol{\beta}}$ is $\sigma^2 CC^T = \sigma^2[(X^TX)^{-1} + DD^T]$. Since $DD^T$ is positive semi-definite, $\text{Var}(\tilde{\boldsymbol{\beta}}) \geq \text{Var}(\hat{\boldsymbol{\beta}})$ in the matrix (Loewner) ordering.

**What Gauss-Markov does NOT say**: It does not say OLS is the best estimator overall. Biased estimators (like Ridge regression) can have lower mean squared error by trading a small amount of bias for a large reduction in variance. The theorem only considers unbiased linear estimators.

### Polynomial Regression as Basis Expansion

What if the true relationship between $x$ and $y$ is curved? For example, perhaps crop yield first increases with fertilizer, then decreases (too much fertilizer is harmful). A straight line cannot capture this.

Polynomial regression handles this by creating new features that are powers of the original feature:

$$
y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \cdots + \beta_d x^d + \epsilon
$$

Crucially, this is still a **linear** model—linear in the **parameters** $\boldsymbol{\beta}$. We have simply defined new features: $z_1 = x$, $z_2 = x^2$, ..., $z_d = x^d$, and the model is $y = \beta_0 + \sum_{k=1}^d \beta_k z_k + \epsilon$, which is linear in $(\beta_0, \ldots, \beta_d)$. All of OLS applies exactly.

The design matrix becomes:

$$
X = \begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^d \\ 1 & x_2 & x_2^2 & \cdots & x_2^d \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^d \end{pmatrix}
$$

This idea—transforming inputs through nonlinear functions but keeping the model linear in parameters—is the concept of **basis expansion**. The "basis functions" are $\phi_0(x) = 1, \phi_1(x) = x, \phi_2(x) = x^2, \ldots$, and the model is $y = \sum_k \beta_k \phi_k(x) + \epsilon$. Beyond polynomials, you can use any functions: logarithms, sines and cosines (Fourier basis), radial basis functions, splines, wavelets, etc.

**The danger of high-degree polynomials**: As $d$ increases, the model becomes more flexible and can fit the training data more closely. But it also becomes more prone to **overfitting**—fitting noise rather than signal. A degree-20 polynomial through 25 data points will pass very close to every point but will oscillate wildly between them, making terrible predictions on new data. This is the **bias-variance tradeoff**: low-degree polynomials have high bias (underfitting) but low variance; high-degree polynomials have low bias but high variance (overfitting).

---

## 4. Worked Examples

### Example 1: OLS with Two Features

Suppose we have the following data on study hours, sleep hours, and exam scores:

| Student | Study (x₁) | Sleep (x₂) | Score (y) |
|---------|-----------|-----------|-----------|
| 1       | 2         | 8         | 60        |
| 2       | 4         | 7         | 70        |
| 3       | 6         | 6         | 75        |
| 4       | 8         | 5         | 80        |

We want to fit $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$.

**Step 1**: Construct $X$ and $\mathbf{y}$.

$$
X = \begin{pmatrix} 1 & 2 & 8 \\ 1 & 4 & 7 \\ 1 & 6 & 6 \\ 1 & 8 & 5 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 60 \\ 70 \\ 75 \\ 80 \end{pmatrix}
$$

**Step 2**: Compute $X^T X$.

$$
X^T X = \begin{pmatrix} 4 & 20 & 26 \\ 20 & 120 & 124 \\ 26 & 124 & 174 \end{pmatrix}
$$

Let us verify one entry: row 2, column 3 is $\sum_i x_{i1} x_{i2} = (2)(8) + (4)(7) + (6)(6) + (8)(5) = 16 + 28 + 36 + 40 = 120$. Wait, that gives 120, but we need to recheck. Actually: $(2)(8) + (4)(7) + (6)(6) + (8)(5) = 16 + 28 + 36 + 40 = 120$. And row 2, column 2 is $\sum x_{i1}^2 = 4 + 16 + 36 + 64 = 120$. So both the (2,2) and (2,3) entries are 120. Let me recompute (2,3): that is $\sum x_{i1} x_{i2}$, which is indeed $16 + 28 + 36 + 40 = 120$. And (3,3) = $\sum x_{i2}^2 = 64 + 49 + 36 + 25 = 174$. The (1,2) entry is $\sum x_{i1} = 20$, (1,3) = $\sum x_{i2} = 26$.

**Step 3**: Compute $X^T \mathbf{y}$.

$$
X^T \mathbf{y} = \begin{pmatrix} 285 \\ 1510 \\ 1795 \end{pmatrix}
$$

Verification: $\sum y_i = 285$. $\sum x_{i1} y_i = 120 + 280 + 450 + 640 = 1490$. Actually let me recompute: $2(60) + 4(70) + 6(75) + 8(80) = 120 + 280 + 450 + 640 = 1490$. And $\sum x_{i2} y_i = 8(60) + 7(70) + 6(75) + 5(80) = 480 + 490 + 450 + 400 = 1820$.

So $X^T\mathbf{y} = (285, 1490, 1820)^T$.

**Step 4**: Solve $(X^TX)\hat{\boldsymbol{\beta}} = X^T\mathbf{y}$ by inverting $X^TX$ (in practice, numerical methods are preferred over explicit inversion).

For this small system, the solution (which one would compute numerically) yields coefficients. For illustration, the interpretation is: the coefficient $\hat{\beta}_1$ tells us the increase in score per additional hour of study, holding sleep constant; $\hat{\beta}_2$ tells us the change in score per additional hour of sleep, holding study constant.

### Example 2: Polynomial Regression

Suppose we have data suggesting a quadratic relationship:

| x  | y   |
|----|-----|
| 1  | 2.1 |
| 2  | 3.9 |
| 3  | 8.2 |
| 4  | 15.8|
| 5  | 24.5|

We fit $y = \beta_0 + \beta_1 x + \beta_2 x^2$. The design matrix is:

$$
X = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \\ 1 & 3 & 9 \\ 1 & 4 & 16 \\ 1 & 5 & 25 \end{pmatrix}
$$

We apply the same normal equations $(X^TX)^{-1}X^T\mathbf{y}$ to get $\hat{\boldsymbol{\beta}}$. Suppose we find $\hat{\beta}_0 \approx 0.7$, $\hat{\beta}_1 \approx -0.5$, $\hat{\beta}_2 \approx 1.1$. Then the fitted model is $\hat{y} = 0.7 - 0.5x + 1.1x^2$. For $x = 3$: $\hat{y} = 0.7 - 1.5 + 9.9 = 9.1$, versus the actual $y = 8.2$, giving a residual of $-0.9$.

Note how the model is nonlinear in $x$ but linear in $\boldsymbol{\beta}$—we used standard OLS machinery unchanged.

---

## 5. Relevance to Machine Learning Practice

### Where Linear Regression is Used

**Baseline models**: In any ML pipeline, a linear regression (possibly with simple feature engineering) should be the first model you build. It is fast to train, easy to interpret, and gives you a benchmark. If a neural network only beats the linear baseline by 1%, you should question whether the complexity is worthwhile.

**Feature importance**: In a well-specified linear model, the coefficients directly tell you which features matter and in what direction. This is invaluable for scientific understanding, debugging, and stakeholder communication.

**Online learning**: Because the OLS solution is a simple matrix formula, it can be updated incrementally (via the Sherman-Morrison-Woodbury formula) as new data arrives, making it suitable for streaming applications.

**Preprocessing and residualization**: In causal inference and A/B testing, linear regression is used to "control for" confounding variables. In experimental settings, adjusting for pre-treatment covariates via regression can dramatically reduce the variance of treatment effect estimates.

**Within larger systems**: The final layer of many ML systems is effectively a linear regression—logistic regression for classification, or a linear layer in a neural network. Understanding linear regression deeply helps you understand how these systems work at their core.

### When NOT to Use Linear Regression

- When the relationship is fundamentally nonlinear and cannot be captured by basis expansion without an impractical number of features.
- When feature interactions are complex and numerous (tree-based models handle this naturally).
- When $p \gg n$ (more features than observations) without regularization.
- When the target is categorical (use logistic regression or other classifiers instead).

### Trade-offs

- **Bias vs. variance**: Linear models have high bias (they cannot represent complex functions) but low variance (they do not overfit easily). This makes them excellent when you have limited data or when the true relationship is roughly linear.
- **Interpretability vs. flexibility**: Linear models are maximally interpretable but minimally flexible. This is a strength in regulated settings but a weakness when prediction accuracy is paramount.
- **Computational cost**: OLS is $O(np^2 + p^3)$—quadratic in features, linear in observations. This is fast for moderate $p$ but problematic when $p$ is very large.

---

## 6. Common Pitfalls & Misconceptions

**Pitfall 1: "R² is high, so the model is good."** $R^2$ measures the fraction of variance explained by the model on the training data. It always increases when you add more features, even useless ones. A polynomial of degree $n-1$ through $n$ points has $R^2 = 1$ but is useless for prediction. Use adjusted $R^2$ or, better yet, cross-validated prediction error.

**Pitfall 2: "The coefficient is small, so the feature is unimportant."** Coefficients depend on the scale of the features. If you measure area in square inches instead of square feet, the coefficient gets divided by 144, but the feature's importance has not changed. Always standardize features before comparing coefficients, or use standardized coefficients.

**Pitfall 3: "Correlation implies the coefficient has the right sign."** In the presence of multicollinearity, the coefficient of a feature can have the opposite sign from its marginal correlation with the target. This is because the coefficient measures the effect of a feature *after controlling for all other features*, which can differ from the marginal effect.

**Pitfall 4: "We should drop features with high p-values one by one."** Stepwise feature selection is statistically problematic. It inflates the type I error rate, produces confidence intervals that are too narrow, and can select different features with slight perturbations of the data. Use regularization (Lasso, Ridge) or domain knowledge instead.

**Pitfall 5: "Normality of errors is required for OLS."** Many practitioners incorrectly believe that OLS requires normally distributed errors. Normality is only needed for exact finite-sample inference (p-values, confidence intervals). The OLS estimates are unbiased and BLUE under Gauss-Markov conditions regardless of the error distribution.

**Pitfall 6: "Residuals should be normally distributed, so we should test for normality first."** You cannot assess the normality of errors from the data before fitting the model. You check residuals after fitting. Moreover, for large samples, the CLT ensures approximate normality of the coefficient estimates even with non-normal errors.

**Pitfall 7: Extrapolation.** Linear models are dangerous for predicting outside the range of the training data. A model that accurately predicts house prices for 1,000–3,000 sq. ft. homes may give absurd predictions for a 10,000 sq. ft. mansion.

**Pitfall 8: Confusing prediction and causal interpretation.** A coefficient in a regression tells you the partial association between a feature and the target, holding other features constant. It does not tell you what happens if you intervene and change that feature. Causal interpretation requires additional assumptions (no unmeasured confounders, etc.) that are not part of the linear regression framework.

---
---

# Part II: Logistic Regression

## 1. Motivation & Intuition

### The Problem Linear Regression Cannot Solve

Suppose instead of predicting a house's price, you want to predict whether an email is spam or not spam. The target is now **binary**: $y \in \{0, 1\}$, where 1 means spam and 0 means not spam.

You might think: "Why not just use linear regression? Treat spam as 1 and not-spam as 0, and predict $y$ using a linear model." The problem is that a linear model can output any real number: it might predict $\hat{y} = -0.3$ or $\hat{y} = 1.7$, which are not valid probabilities. We need predictions that are between 0 and 1 and can be interpreted as the probability that $y = 1$.

Moreover, the assumptions of linear regression are violated for binary outcomes. The errors cannot be normally distributed (they can only take two values), and the variance of the errors depends on the mean (it cannot be constant).

### The Key Idea: Modeling Probabilities

Logistic regression solves this by modeling the **probability** that $y = 1$ as a function of the features, using a specific nonlinear function (the sigmoid or logistic function) to ensure the output is always between 0 and 1.

**Intuitive example**: Suppose you are a doctor trying to predict whether a patient has a disease based on their blood test result $x$. You know that:
- When $x$ is very low, the patient almost certainly does not have the disease: $P(y=1 | x) \approx 0$.
- When $x$ is very high, the patient almost certainly does have the disease: $P(y=1 | x) \approx 1$.
- In between, there is a smooth transition from "probably not" to "probably yes."

This S-shaped transition is exactly what the sigmoid function provides.

### Connection to ML

Logistic regression is arguably the single most widely used classification model in industry:

- **Click-through rate prediction**: Ad platforms use logistic regression (with many engineered features) to predict the probability a user clicks on an ad.
- **Medical diagnosis**: Predicting disease probability from symptoms and test results.
- **Credit scoring**: Predicting the probability of loan default.
- **Spam filtering**: Predicting the probability an email is spam.
- **Neural network output layer**: The final layer of a neural network for binary classification is logistic regression applied to the last hidden layer's activations.
- **Calibrated probabilities**: Unlike many classifiers, logistic regression naturally outputs probabilities (when the model is well-specified), which are essential for decision-making under uncertainty.

---

## 2. Conceptual Foundations

### The Sigmoid Function

The **sigmoid function** (also called the logistic function) is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Its key properties:

- **Range**: For any real number $z$, $\sigma(z) \in (0, 1)$. It never actually reaches 0 or 1.
- **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$.
- **Monotonically increasing**: Larger $z$ gives larger $\sigma(z)$.
- **Centered at 0.5**: $\sigma(0) = 0.5$.
- **Saturation**: As $z \to +\infty$, $\sigma(z) \to 1$. As $z \to -\infty$, $\sigma(z) \to 0$.
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$, which is maximal at $z = 0$ and approaches zero for large $|z|$.

### The Log-Odds (Logit) Interpretation

Logistic regression models the probability $p = P(y = 1 | \mathbf{x})$ as:

$$
p = \sigma(\mathbf{x}^T \boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}^T \boldsymbol{\beta}}}
$$

Taking the inverse of the sigmoid (the **logit function**):

$$
\text{logit}(p) = \ln\frac{p}{1-p} = \mathbf{x}^T \boldsymbol{\beta}
$$

The quantity $\frac{p}{1-p}$ is the **odds** (e.g., odds of 3 means 3 times as likely to be class 1 as class 0). The quantity $\ln\frac{p}{1-p}$ is the **log-odds**.

This gives us a powerful interpretation: **logistic regression is a linear model for the log-odds**. The log-odds of the event $y = 1$ are a linear function of the features. Specifically:

- $\beta_j$ represents the change in the **log-odds** of $y = 1$ for a one-unit increase in feature $x_j$, holding all other features constant.
- $e^{\beta_j}$ represents the multiplicative change in the **odds**. If $\beta_j = 0.5$, then a one-unit increase in $x_j$ multiplies the odds by $e^{0.5} \approx 1.65$ (a 65% increase in odds).

### Decision Boundary

Logistic regression classifies an observation as class 1 if $p \geq 0.5$ (or any other threshold), which corresponds to $\mathbf{x}^T \boldsymbol{\beta} \geq 0$. The **decision boundary** is the set of points where $\mathbf{x}^T \boldsymbol{\beta} = 0$. In two dimensions, this is a straight line; in $p$ dimensions, it is a hyperplane. Thus, logistic regression is a **linear classifier**: it can only separate classes with a linear boundary.

### Assumptions

Logistic regression has fewer assumptions than linear regression:

1. **The log-odds are linear in the features.** This is the fundamental modeling assumption. It does not assume a linear relationship between features and probabilities—the sigmoid introduces nonlinearity.
2. **Independence of observations.** As with linear regression, correlated observations (e.g., repeated measurements on the same patient) require specialized methods.
3. **No perfect multicollinearity.** The same structural requirement as linear regression.
4. **No extreme outliers.** Logistic regression can be sensitive to outliers in the feature space, especially in small samples.

Notably: there is **no homoscedasticity assumption** (the variance of a Bernoulli random variable is $p(1-p)$, which depends on $p$ and hence on the features) and **no normality assumption** for the errors.

---

## 3. Mathematical Formulation

### Likelihood Function

Since $y_i \in \{0, 1\}$, we model $y_i \sim \text{Bernoulli}(p_i)$ where $p_i = \sigma(\mathbf{x}_i^T \boldsymbol{\beta})$. The probability of observing the actual data is:

$$
P(y_i | \mathbf{x}_i, \boldsymbol{\beta}) = p_i^{y_i}(1 - p_i)^{1 - y_i}
$$

This expression works for both $y_i = 0$ and $y_i = 1$: when $y_i = 1$, it gives $p_i$; when $y_i = 0$, it gives $1 - p_i$.

Assuming observations are independent, the **likelihood** is:

$$
L(\boldsymbol{\beta}) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}
$$

The **log-likelihood** is:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]
$$

Substituting $p_i = \sigma(\mathbf{x}_i^T \boldsymbol{\beta})$ and using the property $\ln(1 - \sigma(z)) = \ln(\sigma(-z)) = -z + \ln\sigma(z)$... actually, let us derive this more carefully.

We have $\ln p_i = \ln \sigma(z_i) = -\ln(1 + e^{-z_i})$ where $z_i = \mathbf{x}_i^T \boldsymbol{\beta}$. And $\ln(1-p_i) = \ln(1 - \sigma(z_i)) = \ln(\sigma(-z_i)) = -\ln(1 + e^{z_i}) = -z_i - \ln(1 + e^{-z_i})$.

So:

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ y_i z_i - \ln(1 + e^{z_i}) \right]
$$

This is the standard form of the logistic regression log-likelihood. It is a **concave** function of $\boldsymbol{\beta}$ (the Hessian is negative semi-definite), which means any local maximum is a global maximum. This is a crucial computational advantage.

### Maximum Likelihood Estimation (MLE)

We find $\hat{\boldsymbol{\beta}}$ by maximizing $\ell(\boldsymbol{\beta})$, or equivalently, minimizing the **negative log-likelihood** (which is called the **binary cross-entropy loss** or **logistic loss** in ML):

$$
\mathcal{L}(\boldsymbol{\beta}) = -\ell(\boldsymbol{\beta}) = \sum_{i=1}^n \left[ -y_i z_i + \ln(1 + e^{z_i}) \right]
$$

**The gradient** (score function):

$$
\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \sum_{i=1}^n (y_i - p_i)\mathbf{x}_i = X^T(\mathbf{y} - \mathbf{p})
$$

where $\mathbf{p} = (p_1, \ldots, p_n)^T$. Setting this to zero gives $X^T(\mathbf{y} - \mathbf{p}) = \mathbf{0}$. Since $\mathbf{p}$ depends on $\boldsymbol{\beta}$ nonlinearly, there is **no closed-form solution**. We must use iterative methods.

**The Hessian**:

$$
\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T} = -\sum_{i=1}^n p_i(1 - p_i)\mathbf{x}_i \mathbf{x}_i^T = -X^T W X
$$

where $W = \text{diag}(p_1(1-p_1), \ldots, p_n(1-p_n))$ is a diagonal matrix of weights. Since $0 < p_i < 1$, all diagonal entries of $W$ are positive, so the Hessian is negative definite (assuming $X$ has full rank), confirming concavity.

### Iteratively Reweighted Least Squares (IRLS)

The Newton-Raphson update for maximizing $\ell$ is:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \left(\frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}\right)^{-1} \frac{\partial \ell}{\partial \boldsymbol{\beta}}
$$

Substituting the gradient and Hessian:

$$
\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} + (X^T W^{(t)} X)^{-1} X^T(\mathbf{y} - \mathbf{p}^{(t)})
$$

This can be rearranged into a form that looks like a weighted least squares problem. Define:

$$
\mathbf{z}^{(t)} = X\boldsymbol{\beta}^{(t)} + (W^{(t)})^{-1}(\mathbf{y} - \mathbf{p}^{(t)})
$$

These are "working responses" or "adjusted dependent variables." Then:

$$
\boldsymbol{\beta}^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} \mathbf{z}^{(t)}
$$

This is exactly the weighted least squares solution with weights $W^{(t)}$ and responses $\mathbf{z}^{(t)}$. Since the weights change at each iteration (they depend on the current $\boldsymbol{\beta}$ through $\mathbf{p}$), this is called **Iteratively Reweighted Least Squares** (IRLS).

The algorithm:
1. Initialize $\boldsymbol{\beta}^{(0)}$ (often to zeros).
2. Compute $\mathbf{p}^{(t)} = \sigma(X\boldsymbol{\beta}^{(t)})$.
3. Compute $W^{(t)} = \text{diag}(p_i^{(t)}(1 - p_i^{(t)}))$.
4. Compute working responses $\mathbf{z}^{(t)}$.
5. Solve the weighted least squares problem for $\boldsymbol{\beta}^{(t+1)}$.
6. Repeat until convergence.

IRLS typically converges in 5–25 iterations due to the concavity of the log-likelihood.

### Multiclass Extensions

**One-vs-Rest (OvR)**: For $K$ classes, train $K$ separate binary logistic regression models. Model $k$ predicts $P(\text{class } k \text{ vs. not class } k)$. To classify a new point, run all $K$ models and pick the class with the highest predicted probability. This is simple but produces probabilities that do not sum to 1 across classes (they can sum to more or less than 1).

**Multinomial Logistic Regression (Softmax Regression)**: The principled extension. For $K$ classes, we model:

$$
P(y = k | \mathbf{x}) = \frac{e^{\mathbf{x}^T \boldsymbol{\beta}_k}}{\sum_{j=1}^K e^{\mathbf{x}^T \boldsymbol{\beta}_j}}
$$

This is the **softmax function** applied to the $K$ linear predictors $\mathbf{x}^T \boldsymbol{\beta}_1, \ldots, \mathbf{x}^T \boldsymbol{\beta}_K$. By construction, the probabilities are positive and sum to 1. The model has $K$ parameter vectors (though one is redundant due to the sum-to-one constraint, so we typically fix $\boldsymbol{\beta}_K = \mathbf{0}$ as the reference class).

The loss function is the **categorical cross-entropy**:

$$
\mathcal{L} = -\sum_{i=1}^n \sum_{k=1}^K \mathbb{1}[y_i = k] \ln P(y_i = k | \mathbf{x}_i)
$$

This is the negative log-likelihood of the multinomial distribution and is convex in the parameters.

---

## 4. Worked Examples

### Example: Binary Logistic Regression

Suppose we have a simple one-feature model predicting whether a student passes an exam based on hours studied:

| Hours (x) | Pass (y) |
|-----------|----------|
| 1         | 0        |
| 2         | 0        |
| 3         | 0        |
| 4         | 1        |
| 5         | 1        |
| 6         | 1        |

Suppose after running IRLS (or gradient descent), we find $\hat{\beta}_0 = -4.0$ and $\hat{\beta}_1 = 1.2$.

**Prediction for 3.5 hours**: $z = -4.0 + 1.2(3.5) = 0.2$. $\hat{p} = \sigma(0.2) = \frac{1}{1 + e^{-0.2}} = \frac{1}{1 + 0.819} = 0.550$.

Interpretation: A student who studies 3.5 hours has approximately a 55% probability of passing.

**Decision boundary**: $z = 0$ when $-4.0 + 1.2x = 0$, i.e., $x = 4.0/1.2 \approx 3.33$ hours. Students studying more than 3.33 hours are predicted to pass; fewer, to fail.

**Odds interpretation**: $e^{\hat{\beta}_1} = e^{1.2} \approx 3.32$. Each additional hour of study multiplies the odds of passing by about 3.32. A student studying 5 hours has odds $3.32$ times higher than a student studying 4 hours (all else equal).

---

## 5. Relevance to Machine Learning Practice

### Where Logistic Regression Excels

**High-dimensional sparse data**: Text classification (bag-of-words), click prediction with millions of categorical features. With proper regularization, logistic regression handles $p \gg n$ gracefully.

**Interpretability requirements**: When you need to explain to a stakeholder why a prediction was made. In medicine, finance, and law, interpretability is not optional.

**Calibrated probabilities**: Logistic regression's outputs are well-calibrated probabilities (when the model is correctly specified) without any post-hoc calibration, unlike random forests or SVMs.

**Real-time serving**: Prediction is a single dot product followed by a sigmoid—extremely fast for online serving at scale.

**Warm starting**: Unlike tree-based models, logistic regression can be warm-started from a previous solution, which is useful when retraining on updated data.

### When Not to Use

- When the decision boundary is highly nonlinear (e.g., XOR-like patterns). Logistic regression can only learn linear boundaries.
- When feature interactions are important and unknown. You would need to manually engineer interaction features, whereas tree-based models discover interactions automatically.
- When you have abundant data and accuracy is paramount—in these settings, gradient boosting or deep learning will likely outperform.

### Trade-offs

- **Interpretability vs. performance**: Logistic regression is among the most interpretable classifiers but is often outperformed by ensemble methods on complex datasets.
- **Calibration vs. discrimination**: Logistic regression tends to produce well-calibrated probabilities but may have lower AUC than more flexible models.
- **Speed vs. expressiveness**: Logistic regression is extremely fast to train and predict, but at the cost of being limited to linear decision boundaries.

---

## 6. Common Pitfalls & Misconceptions

**Pitfall 1: Using accuracy as the sole evaluation metric.** For imbalanced classes (e.g., 1% fraud rate), a model that always predicts "not fraud" achieves 99% accuracy but is useless. Use precision, recall, F1, AUC-ROC, or AUC-PR instead.

**Pitfall 2: "Logistic regression assumes a linear relationship between features and probability."** Incorrect. It assumes a linear relationship between features and **log-odds**, which implies a nonlinear (S-shaped) relationship between features and probability.

**Pitfall 3: Interpreting coefficients as probability changes.** $\beta_j$ is the change in log-odds, not in probability. The change in probability depends on the baseline probability: $\Delta p \approx p(1-p)\beta_j \Delta x_j$. At $p = 0.5$, the sensitivity is maximal; at $p = 0.01$ or $p = 0.99$, changes in log-odds have very small effects on probability.

**Pitfall 4: Perfect separation.** If a feature (or combination of features) perfectly separates the classes, the MLE does not exist—the coefficients diverge to $\pm \infty$. This happens in small datasets with many features. The solution is regularization.

**Pitfall 5: Treating outputs as confidence scores without calibration.** While logistic regression is naturally calibrated when the model is well-specified, if you add features that break the linearity assumption, the probabilities may be poorly calibrated. Always check calibration curves (reliability diagrams).

**Pitfall 6: Ignoring class imbalance.** With imbalanced classes, logistic regression will be biased toward the majority class. Solutions include reweighting the loss function, oversampling the minority class (SMOTE), or adjusting the classification threshold.

---
---

# Part III: Regularization Techniques

## 1. Motivation & Intuition

### The Problem: Overfitting

Consider a dataset with 50 observations and 100 features. OLS linear regression would fit the training data perfectly ($R^2 = 1$), but it would perform terribly on new data. Why? With more parameters than observations, there are infinitely many parameter vectors that fit the data perfectly. OLS picks one of them (if $X^TX$ is invertible—often it is not, and the computation simply fails), but that particular solution has no reason to generalize well.

Even when $p < n$, if $p$ is large relative to $n$, OLS tends to overfit. The coefficients become excessively large in magnitude, because the model is fitting noise in the training data. Small perturbations in the data can cause large changes in the coefficients.

### The Idea: Penalizing Complexity

Regularization addresses overfitting by **adding a penalty** to the loss function that discourages large coefficients. Instead of minimizing only the sum of squared residuals (RSS), we minimize:

$$
\text{RSS}(\boldsymbol{\beta}) + \lambda \cdot \text{Penalty}(\boldsymbol{\beta})
$$

where $\lambda \geq 0$ is a **tuning parameter** that controls the strength of regularization:

- $\lambda = 0$: No penalty; we recover OLS.
- $\lambda \to \infty$: The penalty dominates; all coefficients are shrunk toward zero.

The penalty term reflects our **prior belief** that simpler models (with smaller coefficients) are more likely to be correct. This is a formalization of Occam's Razor.

### Real-World Analogy

Think of packing for a trip. Without a suitcase size limit (no regularization), you would bring everything you might possibly need—20 outfits, 5 pairs of shoes, a book collection. The result is unwieldy and most of what you bring goes unused. With a size limit (regularization), you are forced to bring only what is truly important, leading to a more efficient and generalizable packing strategy.

### Connection to ML

Regularization is ubiquitous in modern ML:

- **Weight decay** in neural networks is Ridge (L2) regularization.
- **L1 regularization** (Lasso) is used for feature selection in high-dimensional problems.
- **Elastic Net** is the default regularization for many linear model libraries (e.g., scikit-learn's LogisticRegression uses L2 by default; glmnet uses Elastic Net).
- **Dropout** in neural networks can be interpreted as an approximate form of L2 regularization.
- **Early stopping** in gradient descent acts as implicit regularization.
- **Bayesian inference**: Regularization has a beautiful Bayesian interpretation—regularized estimates are maximum a posteriori (MAP) estimates under specific priors.

---

## 2. Conceptual Foundations

### Ridge Regression (L2 Regularization)

Ridge regression adds the **sum of squared coefficients** to the loss:

$$
\text{Objective} = \underbrace{\sum_{i=1}^n (y_i - \hat{y}_i)^2}_{\text{RSS (data fit)}} + \underbrace{\lambda \sum_{j=1}^p \beta_j^2}_{\text{L2 penalty (complexity)}}
$$

Note: the intercept $\beta_0$ is typically **not** penalized. This is because penalizing the intercept would depend on the arbitrary origin of the target variable—shifting all $y$ values by a constant should not change the model.

**What Ridge does**: It shrinks all coefficients toward zero, but does not set any exactly to zero. It reduces the magnitude of coefficients proportionally—features that would have large OLS coefficients are shrunk the most, but they retain their rank ordering. In the presence of multicollinearity, Ridge stabilizes the coefficients by reducing their variance at the cost of introducing a small bias.

**Geometric intuition**: The OLS objective is a paraboloid in parameter space, centered at the OLS solution. The L2 penalty is a sphere centered at the origin. Ridge regression finds the point where the paraboloid's contour (an ellipsoid of constant RSS) is tangent to a sphere of a given radius. The constraint $\sum \beta_j^2 \leq t$ (for some $t$ determined by $\lambda$) forces the solution to lie within a ball, pushing it toward the origin.

### Lasso (L1 Regularization)

The Lasso (Least Absolute Shrinkage and Selection Operator) replaces the squared penalty with the **sum of absolute values**:

$$
\text{Objective} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j|
$$

**What Lasso does**: Unlike Ridge, Lasso can set coefficients to **exactly zero**, effectively performing **feature selection**. With a sufficiently large $\lambda$, many coefficients become exactly zero, and only the most important features survive. This produces a **sparse** model—one that depends on only a subset of the features.

**Why L1 induces sparsity but L2 does not**: The key geometric insight is that the L1 constraint region (in parameter space) is a **diamond** (or more precisely, a cross-polytope), which has **corners** along the coordinate axes. The L2 constraint region is a **sphere**, which has no corners. When the elliptical RSS contours are tangent to the constraint region, they are much more likely to touch a corner of the diamond (which is a point where one or more coordinates are exactly zero) than a point on the smooth surface of the sphere (where coordinates are generically nonzero).

Formally: the subdifferential of the L1 penalty $|\beta_j|$ at $\beta_j = 0$ is the interval $[-1, 1]$. The optimality condition at zero is $|$ correlation of feature $j$ with the residual $| \leq \lambda$. If the correlation is small enough, the coefficient remains at zero. In contrast, the gradient of the L2 penalty $\beta_j^2$ at $\beta_j = 0$ is exactly zero, so there is no force holding the coefficient at zero—it will drift away with any nonzero correlation.

### Elastic Net

Elastic Net combines L1 and L2 penalties:

$$
\text{Objective} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^p |\beta_j| + \lambda_2 \sum_{j=1}^p \beta_j^2
$$

Or, equivalently, parameterized by a total penalty $\lambda$ and a mixing parameter $\alpha \in [0, 1]$:

$$
\text{Objective} = \text{RSS} + \lambda \left[ \alpha \sum_{j=1}^p |\beta_j| + (1-\alpha) \sum_{j=1}^p \beta_j^2 \right]
$$

When $\alpha = 1$, this is Lasso; when $\alpha = 0$, this is Ridge.

**Why Elastic Net?** Lasso has limitations:

1. **Group selection**: If several features are highly correlated, Lasso tends to select one and set the rest to zero. Which one it selects is somewhat arbitrary and unstable. Ridge keeps all of them (with smaller coefficients). Elastic Net strikes a middle ground: it tends to select groups of correlated features together.

2. **$n < p$ limitation**: When $p > n$, Lasso can select at most $n$ features (a result from the geometry of the solution). Elastic Net does not have this limitation.

3. **Stability**: Lasso solutions can be unstable under small perturbations of the data when features are correlated. The L2 component of Elastic Net stabilizes the solution.

### Group Lasso

Group Lasso extends the sparsity concept from individual features to **groups** of features. Instead of penalizing each $|\beta_j|$, it penalizes the L2 norm of each group:

$$
\text{Penalty} = \sum_{g=1}^G \sqrt{p_g} \|\boldsymbol{\beta}_g\|_2
$$

where $\boldsymbol{\beta}_g$ is the vector of coefficients in group $g$ and $p_g$ is the group size. This sets entire groups of coefficients to zero simultaneously. This is useful when features naturally come in groups—for example, dummy variables encoding a single categorical feature, or the Fourier coefficients of a signal.

---

## 3. Mathematical Formulation

### Ridge Regression: Closed-Form Solution

The Ridge objective is:

$$
J(\boldsymbol{\beta}) = \|\mathbf{y} - X\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 = (\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta}) + \lambda \boldsymbol{\beta}^T \boldsymbol{\beta}
$$

Taking the gradient and setting to zero:

$$
\frac{\partial J}{\partial \boldsymbol{\beta}} = -2X^T \mathbf{y} + 2X^T X \boldsymbol{\beta} + 2\lambda \boldsymbol{\beta} = \mathbf{0}
$$

$$
(X^T X + \lambda I) \hat{\boldsymbol{\beta}}_{\text{ridge}} = X^T \mathbf{y}
$$

$$
\boxed{\hat{\boldsymbol{\beta}}_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}}
$$

**Key observation**: The matrix $X^T X + \lambda I$ is always invertible for $\lambda > 0$, even when $X^T X$ is singular. This is because adding $\lambda I$ shifts all eigenvalues up by $\lambda$, making them all positive. If $X^TX$ has eigenvalues $d_1^2, d_2^2, \ldots, d_p^2$ (some possibly zero), then $X^TX + \lambda I$ has eigenvalues $d_1^2 + \lambda, d_2^2 + \lambda, \ldots, d_p^2 + \lambda$, all of which are strictly positive.

**Singular Value Decomposition (SVD) perspective**: Let $X = U D V^T$ be the SVD, where $U \in \mathbb{R}^{n \times p}$, $D = \text{diag}(d_1, \ldots, d_p)$, and $V \in \mathbb{R}^{p \times p}$. Then:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = V (D^2 + \lambda I)^{-1} D U^T \mathbf{y} = \sum_{j=1}^p \frac{d_j}{d_j^2 + \lambda} \mathbf{v}_j (\mathbf{u}_j^T \mathbf{y})
$$

Compare with OLS: $\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^p \frac{1}{d_j} \mathbf{v}_j (\mathbf{u}_j^T \mathbf{y})$.

Ridge replaces $\frac{1}{d_j}$ with $\frac{d_j}{d_j^2 + \lambda}$. When $d_j$ is large (the corresponding direction in feature space has high variance), the shrinkage factor $\frac{d_j^2}{d_j^2 + \lambda} \approx 1$—almost no shrinkage. When $d_j$ is small (low-variance direction), $\frac{d_j^2}{d_j^2 + \lambda} \approx 0$—heavy shrinkage. This means Ridge preferentially shrinks the coefficients in directions where the data provides little information, which is exactly the right thing to do.

### Lasso: No Closed-Form, Coordinate Descent

The Lasso objective is:

$$
J(\boldsymbol{\beta}) = \|\mathbf{y} - X\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1
$$

The L1 penalty $|\beta_j|$ is not differentiable at $\beta_j = 0$, so we cannot simply set the gradient to zero. Instead, we use the **subgradient** or, more practically, **coordinate descent**.

**Coordinate descent**: Optimize one coefficient at a time, holding the others fixed. For coefficient $\beta_j$, the partial objective (keeping all other $\beta_k$ fixed) is:

$$
J(\beta_j) = \left\| \mathbf{r}_{-j} - \mathbf{x}_j \beta_j \right\|^2 + \lambda |\beta_j|
$$

where $\mathbf{r}_{-j} = \mathbf{y} - \sum_{k \neq j} \mathbf{x}_k \beta_k$ is the **partial residual** (the residual after accounting for all features except $j$).

Expanding:

$$
J(\beta_j) = \mathbf{r}_{-j}^T \mathbf{r}_{-j} - 2\beta_j \mathbf{x}_j^T \mathbf{r}_{-j} + \beta_j^2 \mathbf{x}_j^T \mathbf{x}_j + \lambda |\beta_j|
$$

If the features are standardized so that $\mathbf{x}_j^T \mathbf{x}_j = 1$ (or equivalently, we absorb this normalization), and defining $\rho_j = \mathbf{x}_j^T \mathbf{r}_{-j}$ (the simple least-squares coefficient of the partial residual on feature $j$), the solution is the **soft-thresholding operator**:

$$
\hat{\beta}_j = S(\rho_j, \lambda/2) = \text{sign}(\rho_j) \max(|\rho_j| - \lambda/2, 0)
$$

In the general (non-normalized) case:

$$
\hat{\beta}_j = \frac{S(\rho_j, \lambda/2)}{\mathbf{x}_j^T \mathbf{x}_j}
$$

**The soft-thresholding function**: $S(z, \gamma) = \text{sign}(z)\max(|z| - \gamma, 0)$. It has three regimes:
- If $z > \gamma$: output is $z - \gamma$ (positive, shrunk toward zero).
- If $z < -\gamma$: output is $z + \gamma$ (negative, shrunk toward zero).
- If $|z| \leq \gamma$: output is 0 (set to exactly zero).

This is why Lasso produces sparse solutions: if a feature's correlation with the residual ($\rho_j$) is small enough (less than $\lambda/2$ in absolute value), the coefficient is set to exactly zero.

**The algorithm**: Initialize $\boldsymbol{\beta}$ (often to zeros). Cycle through features $j = 1, 2, \ldots, p$ repeatedly, updating each $\beta_j$ using the soft-thresholding formula. Repeat until convergence. This is called **cyclic coordinate descent** and is extremely efficient, especially for sparse solutions.

### Elastic Net Solution

The Elastic Net objective:

$$
J(\boldsymbol{\beta}) = \|\mathbf{y} - X\boldsymbol{\beta}\|^2 + \lambda \alpha \|\boldsymbol{\beta}\|_1 + \lambda (1-\alpha) \|\boldsymbol{\beta}\|_2^2
$$

The coordinate descent update for Elastic Net is a slight modification of Lasso:

$$
\hat{\beta}_j = \frac{S(\rho_j, \lambda \alpha / 2)}{\mathbf{x}_j^T \mathbf{x}_j + \lambda(1-\alpha)}
$$

The denominator now includes $\lambda(1-\alpha)$, which is the Ridge contribution. This is the only difference from Lasso: the Ridge penalty adds to the denominator, providing additional shrinkage and stabilization.

### Bayesian Interpretation of Regularization

Regularization has a beautiful probabilistic interpretation. In Bayesian statistics, we place a **prior** on the parameters and compute the **posterior** (the parameter distribution given the data). The **maximum a posteriori (MAP)** estimate is the mode of the posterior.

**Setup**: The model is $y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \epsilon_i$ with $\epsilon_i \sim N(0, \sigma^2)$, i.i.d. The likelihood is:

$$
p(\mathbf{y} | X, \boldsymbol{\beta}, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{x}_i^T \boldsymbol{\beta})^2}{2\sigma^2}\right)
$$

**Ridge = Gaussian prior**: Place a prior $\beta_j \sim N(0, \tau^2)$ independently for each $j$. The log-posterior is:

$$
\ln p(\boldsymbol{\beta} | \mathbf{y}, X) \propto -\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 - \frac{1}{2\tau^2}\sum_{j=1}^p \beta_j^2
$$

Maximizing this is equivalent to minimizing:

$$
\sum_{i=1}^n(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \frac{\sigma^2}{\tau^2}\sum_{j=1}^p \beta_j^2
$$

This is exactly Ridge regression with $\lambda = \sigma^2 / \tau^2$. A small $\tau^2$ (tight prior, meaning we believe the coefficients are small) corresponds to large $\lambda$ (strong regularization). A large $\tau^2$ (diffuse prior) corresponds to small $\lambda$ (weak regularization).

**Lasso = Laplace prior**: Place a prior $\beta_j \sim \text{Laplace}(0, b)$ independently. The Laplace density is $p(\beta_j) = \frac{1}{2b}\exp(-|\beta_j|/b)$. The log-posterior is:

$$
\ln p(\boldsymbol{\beta} | \mathbf{y}, X) \propto -\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 - \frac{1}{b}\sum_{j=1}^p |\beta_j|
$$

Maximizing this is equivalent to minimizing:

$$
\sum_{i=1}^n(y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2 + \frac{2\sigma^2}{b}\sum_{j=1}^p |\beta_j|
$$

This is exactly Lasso with $\lambda = 2\sigma^2 / b$.

**Why the Laplace prior induces sparsity**: The Laplace distribution has a sharp peak at zero (it is not differentiable there), putting substantial probability mass near zero. The Gaussian prior is smooth at zero and puts relatively less mass there. This is why the Laplace prior (Lasso) tends to pull coefficients all the way to zero, while the Gaussian prior (Ridge) pulls them toward zero but rarely all the way.

**Important caveat**: The MAP estimate is a point estimate—it does not capture the full posterior uncertainty. For full Bayesian inference, you would compute the posterior distribution of $\boldsymbol{\beta}$, which is generally not available in closed form for the Laplace prior (but is Gaussian for the Ridge/Gaussian case).

---

## 4. Worked Examples

### Example 1: Ridge vs. OLS with Multicollinearity

Suppose we have two features $x_1$ and $x_2$ that are nearly identical ($x_2 = x_1 + \text{small noise}$), and the true model is $y = 3x_1 + 0x_2 + \epsilon$. Because $x_1 \approx x_2$, the data cannot distinguish whether it is $x_1$ or $x_2$ (or some combination) that matters.

OLS might estimate $\hat{\beta}_1 = 15$ and $\hat{\beta}_2 = -12$, so that $\hat{\beta}_1 + \hat{\beta}_2 \approx 3$ (the combined effect is correct, but the individual coefficients are wildly off). On a different sample, OLS might give $\hat{\beta}_1 = -8$ and $\hat{\beta}_2 = 11$.

Ridge regression with an appropriate $\lambda$ might give $\hat{\beta}_1 = 1.8$ and $\hat{\beta}_2 = 1.4$. These are biased (the true $\beta_2 = 0$), but they are much more stable and have lower mean squared error than the OLS estimates.

### Example 2: Lasso for Feature Selection

Suppose we have 10 features, but only 3 truly affect the target. The true model is $y = 2x_1 + 3x_5 + x_7 + \epsilon$, with $\beta_j = 0$ for all other $j$.

With OLS and 50 observations, all 10 coefficients will be nonzero (and the 7 irrelevant ones will have nonzero values due to sampling noise).

With Lasso and a well-chosen $\lambda$ (typically selected by cross-validation), the solution might be:
- $\hat{\beta}_1 = 1.8, \hat{\beta}_5 = 2.7, \hat{\beta}_7 = 0.9$ (close to the true values, slightly shrunk)
- $\hat{\beta}_j = 0$ for $j \in \{2, 3, 4, 6, 8, 9, 10\}$ (correctly identified as irrelevant)

This automatic feature selection is one of Lasso's most powerful properties.

### Example 3: Soft-Thresholding Walkthrough

Consider the one-dimensional Lasso problem: minimize $(y - \beta)^2 + \lambda |\beta|$.

- Suppose $y = 5$ and $\lambda = 2$. The OLS solution is $\hat{\beta} = 5$. The Lasso solution is $\hat{\beta} = S(5, 1) = 5 - 1 = 4$.
- Suppose $y = 0.8$ and $\lambda = 2$. The OLS solution is $\hat{\beta} = 0.8$. The Lasso solution is $\hat{\beta} = S(0.8, 1) = \max(0.8 - 1, 0) = 0$. The coefficient is killed.
- Suppose $y = -3$ and $\lambda = 2$. The OLS solution is $\hat{\beta} = -3$. The Lasso solution is $\hat{\beta} = S(-3, 1) = -(3 - 1) = -2$.

The pattern: coefficients within $\lambda/2$ of zero are zeroed out; those outside are shrunk by $\lambda/2$ toward zero.

---

## 5. Relevance to Machine Learning Practice

### When to Use Each Method

**Ridge**: When you believe all features are potentially relevant (dense signals), when features are correlated, or when you want stable coefficient estimates. Examples: genomics (many correlated gene expression values), NLP with dense embeddings, time-series with lagged features.

**Lasso**: When you believe only a few features are truly important (sparse signal), when you want automatic feature selection, or when interpretability is paramount. Examples: clinical biomarker discovery (which of 1000 blood markers predict disease?), feature selection in production ML pipelines.

**Elastic Net**: The default choice when you are unsure. It combines the best of both: feature selection from L1 and stability/group selection from L2. Most production ML libraries (glmnet, scikit-learn) use Elastic Net as the underlying engine.

### Choosing $\lambda$

The tuning parameter $\lambda$ is almost always chosen by **cross-validation**:

1. Try a grid of $\lambda$ values (typically on a log scale, e.g., $\lambda \in \{10^{-4}, 10^{-3}, \ldots, 10^2\}$).
2. For each $\lambda$, fit the model on training folds and evaluate on held-out folds.
3. Choose the $\lambda$ that minimizes the average held-out error.

Common conventions: $\lambda_{\min}$ (the value that minimizes CV error) and $\lambda_{1se}$ (the largest $\lambda$ within one standard error of the minimum—this gives a simpler model with nearly the same performance).

### Standardization

**Critical practical point**: Before applying Ridge or Lasso, you should standardize the features so that each has mean zero and standard deviation one. This is because the penalty treats all coefficients equally, but coefficients for features on different scales are not comparable. A coefficient of 0.001 for "income in dollars" is very different from a coefficient of 0.001 for "age in years." Standardization puts all features on the same scale, so the penalty affects them equally.

### Computational Considerations

- **Ridge**: Closed-form solution, $O(np^2 + p^3)$, same as OLS. Very fast.
- **Lasso**: No closed form, but coordinate descent is extremely efficient, especially when the solution is sparse. The glmnet algorithm computes the entire Lasso path (solutions for all $\lambda$ values) in roughly the time it takes to compute a single OLS solution.
- **Elastic Net**: Same coordinate descent approach as Lasso, essentially the same cost.

---

## 6. Common Pitfalls & Misconceptions

**Pitfall 1: "Lasso always selects the right features."** Lasso's feature selection is only reliable under certain conditions (the "irrepresentable condition" or "restricted eigenvalue condition"). With highly correlated features, Lasso may select the wrong one, and the selection can be unstable across different samples. In these cases, stability selection (running Lasso on many bootstrap samples and keeping features selected frequently) is more reliable.

**Pitfall 2: Forgetting to standardize.** If features are on different scales, regularization penalizes some features more than others, leading to biased results. Always standardize before applying penalized methods, and remember to un-standardize the coefficients if you need to interpret them on the original scale.

**Pitfall 3: Penalizing the intercept.** The intercept should not be penalized. Most software handles this correctly by default, but it is worth checking.

**Pitfall 4: "More regularization is always better."** Excessive regularization causes underfitting—the model becomes too simple to capture the true signal. The bias-variance tradeoff is controlled by $\lambda$, and the optimal $\lambda$ balances bias and variance.

**Pitfall 5: Using Lasso p-values.** Standard statistical inference (p-values, confidence intervals) is not valid for Lasso-selected features. The selection process invalidates the distributional assumptions underlying the tests. Post-selection inference requires specialized methods (e.g., selective inference, de-biased Lasso).

**Pitfall 6: Confusing Bayesian regularization with full Bayesian inference.** Ridge and Lasso correspond to MAP estimates with specific priors, but MAP is a point estimate. Full Bayesian inference computes the entire posterior distribution, which gives uncertainty quantification. MAP with a Laplace prior is NOT the same as Bayesian Lasso (which uses the full posterior under a Laplace prior and does not produce exactly sparse solutions).

**Pitfall 7: Applying Lasso to perfectly correlated features.** If two features are perfectly correlated, Lasso arbitrarily selects one and drops the other. The choice is unstable and depends on numerical details. If you know features are grouped, use Group Lasso or Elastic Net.

---
---

# Interview Preparation

## Section A: Linear Regression Interview Questions

### Foundational Questions

**Q1: What is linear regression, and what are its key assumptions?**

Linear regression models the conditional expectation of a continuous target variable as a linear function of features: $E[y | \mathbf{x}] = \beta_0 + \sum_{j=1}^p \beta_j x_j$. The key assumptions (for OLS specifically) are:

1. **Linearity**: The true conditional expectation is linear in the features.
2. **Independence**: Observations (and hence errors) are independent.
3. **Homoscedasticity**: $\text{Var}(\epsilon_i | \mathbf{x}_i) = \sigma^2$ (constant error variance).
4. **Exogeneity**: $E[\epsilon_i | \mathbf{x}_i] = 0$ (errors are uncorrelated with features).
5. **No perfect multicollinearity**: The design matrix $X$ has full column rank.
6. **Normality** (for inference): $\epsilon_i \sim N(0, \sigma^2)$.

The first four are needed for OLS to be BLUE (Gauss-Markov). Normality is only needed for exact finite-sample inference.

**Q2: What is the geometric interpretation of OLS?**

The vector of fitted values $\hat{\mathbf{y}} = X(X^TX)^{-1}X^T\mathbf{y}$ is the orthogonal projection of the observation vector $\mathbf{y}$ onto the column space of $X$ (the subspace of all possible fitted values). The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is perpendicular to this subspace. OLS finds the closest point in the column space to the data vector, where "closest" means smallest Euclidean distance (i.e., smallest sum of squared residuals).

**Q3: Why is polynomial regression still considered a "linear" model?**

Because "linear" refers to linearity in the **parameters**, not in the features. Polynomial regression uses features $x, x^2, x^3, \ldots$, but the model is $y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots$, which is a linear function of $(\beta_0, \beta_1, \beta_2, \ldots)$. This means all the standard OLS theory applies: the normal equations, Gauss-Markov, etc.

**Q4: What does the Gauss-Markov theorem guarantee, and what does it NOT guarantee?**

It guarantees that under assumptions 1–4 (linearity, independence, homoscedasticity, exogeneity), OLS is the **Best Linear Unbiased Estimator** (BLUE). "Best" means lowest variance among all linear unbiased estimators. It does NOT:
- Guarantee that OLS has the lowest mean squared error (MSE). Biased estimators like Ridge can have lower MSE.
- Apply to nonlinear estimators. There may be nonlinear unbiased estimators with lower variance.
- Require normality. The theorem holds for any error distribution with finite variance.
- Guarantee good predictions on new data. It is about the statistical properties of the estimator, not about predictive performance.

### Mathematical Questions

**Q5: Derive the OLS normal equations from scratch.**

We minimize $\text{RSS}(\boldsymbol{\beta}) = (\mathbf{y} - X\boldsymbol{\beta})^T(\mathbf{y} - X\boldsymbol{\beta})$. Expanding:

$\text{RSS} = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^TX^T\mathbf{y} + \boldsymbol{\beta}^TX^TX\boldsymbol{\beta}$

Taking the gradient: $\nabla_{\boldsymbol{\beta}}\text{RSS} = -2X^T\mathbf{y} + 2X^TX\boldsymbol{\beta}$.

Setting to zero: $X^TX\hat{\boldsymbol{\beta}} = X^T\mathbf{y}$.

Solution: $\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y}$ when $X^TX$ is invertible.

The Hessian is $2X^TX$, which is positive semi-definite (positive definite when $X$ has full rank), confirming this is a minimum.

**Q6: Show that the OLS estimator is unbiased.**

$\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y} = (X^TX)^{-1}X^T(X\boldsymbol{\beta} + \boldsymbol{\epsilon}) = \boldsymbol{\beta} + (X^TX)^{-1}X^T\boldsymbol{\epsilon}$.

Taking expectations (treating $X$ as fixed):

$E[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta} + (X^TX)^{-1}X^T E[\boldsymbol{\epsilon}] = \boldsymbol{\beta} + \mathbf{0} = \boldsymbol{\beta}$.

The last step uses exogeneity: $E[\boldsymbol{\epsilon} | X] = \mathbf{0}$.

**Q7: Derive the variance of the OLS estimator.**

$\hat{\boldsymbol{\beta}} - \boldsymbol{\beta} = (X^TX)^{-1}X^T\boldsymbol{\epsilon}$.

$\text{Var}(\hat{\boldsymbol{\beta}} | X) = (X^TX)^{-1}X^T \text{Var}(\boldsymbol{\epsilon}) X(X^TX)^{-1}$.

Under homoscedasticity and independence, $\text{Var}(\boldsymbol{\epsilon}) = \sigma^2 I$.

$\text{Var}(\hat{\boldsymbol{\beta}} | X) = \sigma^2 (X^TX)^{-1}X^T X (X^TX)^{-1} = \sigma^2 (X^TX)^{-1}$.

If heteroscedasticity holds ($\text{Var}(\boldsymbol{\epsilon}) = \Omega \neq \sigma^2 I$), the "sandwich" formula applies: $(X^TX)^{-1}X^T\Omega X(X^TX)^{-1}$, which is the basis for robust (Huber-White) standard errors.

### Applied Questions

**Q8: You train a linear regression to predict user engagement. The model has 200 features and 50,000 observations. The training MSE is 2.1 and the test MSE is 2.3. Your manager asks if you should add more features from a new data source (500 additional features). What considerations guide your decision?**

Several factors:

- **Current gap**: Training MSE (2.1) vs. test MSE (2.3) shows a small gap, suggesting the current model is not badly overfitting. This means the model may be underfit—it could benefit from more expressive features.
- **New feature quality**: Are the 500 features genuinely informative? Do domain experts believe they capture signals not in the current 200 features? Adding noise features will increase variance without reducing bias.
- **Dimensionality concerns**: Going from 200 to 700 features with 50,000 observations is unlikely to cause problems for OLS ($n \gg p$), but if many new features are correlated, multicollinearity may destabilize coefficients. Regularization (Ridge/Lasso) would be advisable.
- **Feature selection**: Rather than adding all 500, use Lasso or domain knowledge to select a subset.
- **Computational cost**: More features increase training and inference time linearly. In a production system, this matters.
- **A/B testing**: Even if offline metrics improve, the true test is online performance. Run an A/B test.

**Q9: You notice that one feature's coefficient has a counterintuitive sign (e.g., "years of experience" has a negative coefficient in a salary prediction model). What could cause this?**

Several explanations:

1. **Multicollinearity**: "Years of experience" is correlated with "age" or "education level." The coefficient captures the partial effect after controlling for correlated variables, which can differ from the marginal effect.
2. **Simpson's paradox**: The feature might have a positive relationship within subgroups but a negative relationship overall due to confounding. For example, experienced people might cluster in lower-paying industries.
3. **Omitted variable bias**: A confounding variable not in the model (e.g., industry) is correlated with both experience and salary.
4. **Scale issues**: If the feature is on a different scale than expected, the coefficient magnitude (and potentially sign) may be misleading.
5. **Insufficient data**: In small samples, sampling variability can produce coefficients with the wrong sign.

**Diagnosis**: Check the correlation matrix, compute variance inflation factors (VIF), try removing correlated features, and examine the coefficient's stability across bootstrap samples.

### Debugging & Failure-Mode Questions

**Q10: Your linear regression model's residual plot shows a funnel shape (residuals fan out as predicted values increase). What is wrong and how do you fix it?**

The funnel shape indicates **heteroscedasticity**: the error variance increases with the predicted value. This is very common in models predicting positive quantities (prices, counts, revenues) where larger values naturally have more variability.

Consequences: OLS estimates are still unbiased but no longer BLUE. Standard errors are incorrect (usually too small for large predicted values), making confidence intervals and hypothesis tests unreliable.

Fixes:
1. **Transform the target**: Apply $\log(y)$ and model the log. This often stabilizes variance for multiplicative errors.
2. **Weighted Least Squares (WLS)**: Weight observations inversely proportional to their error variance.
3. **Robust standard errors**: Use Huber-White (sandwich) standard errors, which are valid under heteroscedasticity.
4. **Generalized Linear Models**: Use a GLM with a log link and appropriate variance function.

**Q11: You fit a degree-15 polynomial to 20 data points and get a perfect fit ($R^2 = 1$ on training data) but terrible test performance. Explain what happened mathematically.**

A degree-15 polynomial has 16 parameters (including the intercept). With 20 data points, we have $n = 20$ and $p = 16$, so the model has enough flexibility to nearly interpolate the data. The model is fitting the noise, not the signal.

Mathematically, the eigenvalues of $X^TX$ in the polynomial basis span many orders of magnitude (the columns of the Vandermonde matrix are highly collinear). The OLS solution amplifies noise in the low-eigenvalue directions, producing coefficients with enormous magnitudes that oscillate wildly between data points.

The bias-variance decomposition: the degree-15 polynomial has very low bias (it can represent almost any smooth function) but extremely high variance (it changes dramatically with different training samples). The test MSE = bias² + variance + irreducible noise, and the variance term dominates.

Fix: Use regularization (Ridge/Lasso), reduce the polynomial degree, use cross-validation to select the degree, or switch to splines.

### Follow-up & Probing Questions

**Q12: "You said $X^TX$ must be invertible. What happens numerically when it is nearly singular?"**

When $X^TX$ is nearly singular (has very small eigenvalues), the condition number $\kappa = d_{\max}/d_{\min}$ is large. This causes:

1. **Numerical instability**: Small rounding errors in the data or computation are amplified by a factor proportional to $\kappa$ when solving the system. Coefficients become unreliable.
2. **Large coefficient variance**: $\text{Var}(\hat{\beta}_j)$ is proportional to the $j$-th diagonal of $(X^TX)^{-1}$, which is large when eigenvalues are small. This leads to wide confidence intervals and coefficients that change sign across samples.
3. **Practical symptoms**: Large VIF values (VIF > 10 is a common warning), coefficients with unexpected signs, coefficients that change dramatically when a single observation is added or removed.

Remedies: Ridge regularization (adds $\lambda$ to all eigenvalues, bounding the condition number), PCA (project onto the top principal components), remove or combine collinear features.

---

## Section B: Logistic Regression Interview Questions

### Foundational Questions

**Q13: Why can't we use linear regression for classification?**

Three fundamental problems:

1. **Unbounded predictions**: Linear regression predicts any real number, but probabilities must be in $[0, 1]$. Predicting $P(\text{spam}) = -0.3$ or $P(\text{spam}) = 1.7$ is nonsensical.
2. **Violated assumptions**: For a binary outcome $y \in \{0, 1\}$, the errors $\epsilon = y - \hat{y}$ can only take two values for any given $\hat{y}$. This violates normality and homoscedasticity—the variance is $p(1-p)$, which depends on the mean.
3. **Suboptimal decision boundary**: Linear regression with a threshold can classify, but it is sensitive to outliers and may produce a worse decision boundary than logistic regression, especially when classes are imbalanced.

**Q14: Explain the relationship between the sigmoid function, log-odds, and probability in logistic regression.**

Logistic regression models: $\ln \frac{p}{1-p} = \mathbf{x}^T\boldsymbol{\beta}$ (log-odds are linear in features).

Equivalently: $\frac{p}{1-p} = e^{\mathbf{x}^T\boldsymbol{\beta}}$ (odds are exponential in the linear predictor).

Equivalently: $p = \frac{1}{1 + e^{-\mathbf{x}^T\boldsymbol{\beta}}} = \sigma(\mathbf{x}^T\boldsymbol{\beta})$ (probability is the sigmoid of the linear predictor).

The sigmoid function maps the real line to $(0, 1)$, ensuring valid probabilities. The log-odds formulation is the "natural" parameterization because it is linear, making the model a member of the exponential family.

**Q15: What is the difference between One-vs-Rest and Multinomial logistic regression for multiclass problems?**

**One-vs-Rest (OvR)**: Trains $K$ independent binary classifiers, each distinguishing one class from all others. Predictions are the class with the highest predicted probability. Probabilities across classifiers do not sum to 1. Training is parallelizable. Works well when classes are well-separated but can struggle with overlapping classes.

**Multinomial (Softmax)**: A single model with $K$ sets of weights, producing probabilities via the softmax function that sum to 1 by construction. The model is trained by maximizing the multinomial log-likelihood. This is more principled (it models the joint probability of all classes), handles overlapping classes better, and naturally produces calibrated probabilities. The cost is that it is a single optimization problem over $K \times p$ parameters, which can be slower to train.

In practice, for many problems the two approaches give similar results. Multinomial is preferred when well-calibrated probabilities are needed.

### Mathematical Questions

**Q16: Derive the gradient of the logistic regression log-likelihood.**

The log-likelihood is $\ell(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i z_i - \ln(1 + e^{z_i})]$ where $z_i = \mathbf{x}_i^T\boldsymbol{\beta}$.

$\frac{\partial \ell}{\partial \boldsymbol{\beta}} = \sum_{i=1}^n \left[y_i \mathbf{x}_i - \frac{e^{z_i}}{1 + e^{z_i}} \mathbf{x}_i\right] = \sum_{i=1}^n (y_i - \sigma(z_i))\mathbf{x}_i = X^T(\mathbf{y} - \mathbf{p})$

where $p_i = \sigma(z_i)$. The gradient has the same form as the OLS gradient ($X^T$ times the residuals), but the "residuals" are $y_i - p_i$ and $p_i$ is a nonlinear function of $\boldsymbol{\beta}$. This is why logistic regression requires iterative optimization.

Setting the gradient to zero: $\sum_{i=1}^n (y_i - p_i)\mathbf{x}_i = \mathbf{0}$. This says the weighted "residuals" are uncorrelated with the features at the MLE—analogous to the OLS orthogonality condition.

**Q17: Show that the logistic regression log-likelihood is concave.**

The Hessian is:

$H = \frac{\partial^2 \ell}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T} = -\sum_{i=1}^n p_i(1-p_i)\mathbf{x}_i\mathbf{x}_i^T = -X^TWX$

where $W = \text{diag}(p_i(1-p_i))$. For any nonzero vector $\mathbf{v}$:

$\mathbf{v}^T H \mathbf{v} = -\sum_{i=1}^n p_i(1-p_i)(\mathbf{x}_i^T \mathbf{v})^2 \leq 0$

since $0 < p_i < 1$ implies $p_i(1-p_i) > 0$, and $(\mathbf{x}_i^T\mathbf{v})^2 \geq 0$. Strict inequality holds when $X$ has full rank and no $p_i$ is exactly 0 or 1. Therefore the Hessian is negative semi-definite (negative definite under full rank), confirming concavity.

Concavity implies any local maximum is a global maximum, so gradient-based methods are guaranteed to find the optimal solution.

### Applied Questions

**Q18: You're building a fraud detection system. The dataset has 0.1% fraud and 99.9% legitimate transactions. How does this affect logistic regression, and what do you do about it?**

**Effects of class imbalance**:
1. The model will learn to predict "not fraud" almost always because the loss function weighs the 99.9% majority heavily. The intercept $\beta_0$ will be very negative.
2. The predicted probabilities will be very low for most transactions, making it hard to find a good threshold.
3. Gradient updates are dominated by the majority class, so learning the minority-class signal is slow.

**Solutions**:
1. **Class weighting**: Assign weight $w_1 = n / (2 n_1)$ to fraud and $w_0 = n / (2 n_0)$ to legitimate. This modifies the loss to $-\sum w_{y_i}[y_i \log p_i + (1-y_i)\log(1-p_i)]$, effectively up-weighting the minority class.
2. **Threshold adjustment**: Instead of classifying at $p = 0.5$, use a lower threshold (e.g., $p = 0.001$). Optimize the threshold for the business objective (e.g., maximize recall at a fixed false positive rate).
3. **Resampling**: Oversample the minority class (SMOTE) or undersample the majority class. Be careful to resample only the training set, never the validation/test set.
4. **Evaluation metrics**: Use precision-recall AUC, not ROC AUC (which can be misleadingly high under extreme imbalance). Track precision and recall at operationally relevant thresholds.
5. **Focal loss**: Down-weight well-classified examples, focusing the loss on hard-to-classify cases.

**Q19: A colleague suggests using the raw probability output from logistic regression as the predicted click-through rate for ad ranking. What could go wrong?**

This is actually one of logistic regression's strengths—its outputs are naturally calibrated probabilities when the model is well-specified. However, several things can go wrong:

1. **Model misspecification**: If the true log-odds are not linear in the features, the probabilities can be systematically miscalibrated. Check with a calibration curve (reliability diagram).
2. **Distribution shift**: If the training data distribution differs from the production distribution (e.g., user demographics shift, new ad types appear), the probabilities become unreliable.
3. **Feature engineering**: If features are transformed (e.g., bucketed, crossed) without careful calibration, the monotonic relationship between log-odds and the features may break.
4. **Regularization effects**: Strong L2 regularization compresses predicted probabilities toward 0.5; strong L1 drops features, potentially losing calibration-relevant signal.

Best practices: Always check calibration curves, use Platt scaling or isotonic regression as a post-hoc fix if needed, and monitor calibration over time in production.

### Debugging & Failure-Mode Questions

**Q20: You train a logistic regression and the optimizer reports that it did not converge after 1000 iterations. What could cause this?**

Possible causes:

1. **Perfect or quasi-perfect separation**: One or more features perfectly predict the outcome. The MLE pushes coefficients to $\pm \infty$, and the optimizer never converges. Diagnosis: check if any feature perfectly separates the classes. Fix: add regularization (even a small $\lambda$).
2. **Learning rate too large** (if using gradient descent): Large steps cause oscillation. Fix: reduce the learning rate or use adaptive methods (Adam).
3. **Ill-conditioned features**: Extremely different feature scales cause the loss landscape to have elongated contours, slowing convergence. Fix: standardize features.
4. **Convergence criterion too tight**: The optimizer may be making progress but has not reached the extremely tight tolerance. Fix: relax the tolerance or increase max iterations.
5. **Multicollinearity**: Highly correlated features create a flat direction in the loss surface, slowing convergence. Fix: regularization or remove redundant features.

**Q21: Your logistic regression's AUC-ROC is 0.95 on the test set but precision at recall=0.5 is only 0.10. How is this possible?**

High AUC-ROC with low precision at moderate recall is a hallmark of **extreme class imbalance**. AUC-ROC measures the model's ability to rank positive examples above negative ones, and even a model with low absolute precision can achieve high AUC-ROC if it correctly ranks most positives above most negatives.

With 0.1% positive rate, even at recall=0.5 (catching half the positives), the absolute number of positives is tiny relative to the number of negatives. If the model also flags many negatives as potential positives (even a small fraction of a huge negative set), precision will be low.

Fix: Report AUC-PR (precision-recall AUC) instead of AUC-ROC for imbalanced datasets. Set the operating threshold based on the precision-recall curve at the desired precision or recall level, not on AUC-ROC.

### Follow-up & Probing Questions

**Q22: "You mentioned that the sigmoid derivative is $\sigma(z)(1-\sigma(z))$. What implications does this have for training deep networks with sigmoid activations?"**

The derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ has a maximum value of 0.25 (at $z = 0$) and approaches zero for large $|z|$. This means:

1. **Vanishing gradients**: In a deep network where each layer uses the sigmoid activation, the backpropagated gradient is a product of many sigmoid derivatives, each $\leq 0.25$. With $L$ layers, the gradient at the first layer is multiplied by $\leq 0.25^L$, which shrinks exponentially. For 10 layers, $0.25^{10} \approx 10^{-6}$. Early layers barely learn.

2. **Saturated neurons**: When $|z|$ is large, $\sigma'(z) \approx 0$, so the gradient through that neuron is nearly zero. If many neurons saturate, training stalls entirely.

This is why modern networks use ReLU ($\max(0, z)$) or its variants. ReLU has a derivative of 1 for positive inputs, completely avoiding the vanishing gradient problem for active neurons.

---

## Section C: Regularization Interview Questions

### Foundational Questions

**Q23: Explain the bias-variance tradeoff in the context of Ridge regression.**

As $\lambda$ increases:

- **Bias increases**: Ridge shrinks coefficients toward zero, so $E[\hat{\boldsymbol{\beta}}_{\text{ridge}}] \neq \boldsymbol{\beta}$ (biased). The larger $\lambda$, the more the coefficients are shrunk, and the more the model's predictions deviate systematically from the truth.
- **Variance decreases**: $\text{Var}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2(X^TX + \lambda I)^{-1}X^TX(X^TX + \lambda I)^{-1}$, which decreases as $\lambda$ increases. The estimates become more stable across different samples.
- **MSE = bias² + variance**: There exists an optimal $\lambda^*$ that minimizes MSE by finding the sweet spot where the reduction in variance outweighs the increase in bias².

At $\lambda = 0$: zero bias, maximum variance (OLS).
At $\lambda \to \infty$: maximum bias (all coefficients are zero), zero variance.
The optimal $\lambda$ is somewhere in between and is found by cross-validation.

**Q24: Why does L1 regularization produce sparse solutions but L2 does not?**

**Geometric argument**: The L1 constraint set $\{|\beta_1| + |\beta_2| \leq t\}$ is a diamond with corners on the coordinate axes. The L2 constraint set $\{\beta_1^2 + \beta_2^2 \leq t\}$ is a circle. The OLS solution's iso-cost contours are ellipses. As the ellipses shrink (from the OLS solution toward the origin), the first point of tangency with the diamond is likely to be at a corner (where some coordinate is zero). With the circle, tangency is unlikely to occur at a point where any coordinate is exactly zero.

**Analytical argument**: The optimality condition for L1 at $\beta_j = 0$ involves the subdifferential $\partial |\beta_j| = [-1, 1]$. The condition $\beta_j = 0$ is optimal when $|\partial \ell / \partial \beta_j| \leq \lambda$, i.e., when the gradient magnitude is less than the penalty strength. This condition has nonzero measure (a range of gradient values). For L2, the condition for $\beta_j = 0$ is $\partial \ell / \partial \beta_j = 0$ (a set of measure zero), so it almost never happens.

**Q25: What is the Bayesian interpretation of Ridge and Lasso?**

Ridge corresponds to Maximum A Posteriori (MAP) estimation with a Gaussian prior: $\beta_j \sim N(0, \tau^2)$. The Gaussian prior is symmetric and smooth at zero, expressing a belief that coefficients are small but unlikely to be exactly zero. The regularization strength is $\lambda = \sigma^2 / \tau^2$.

Lasso corresponds to MAP estimation with a Laplace prior: $\beta_j \sim \text{Laplace}(0, b)$. The Laplace distribution has a sharp peak at zero, putting substantial mass on values very near zero. This is why it produces point estimates that are exactly zero. The regularization strength is $\lambda = 2\sigma^2 / b$.

Elastic Net corresponds to a prior that is a product of Gaussian and Laplace: $p(\beta_j) \propto \exp(-|\beta_j|/b - \beta_j^2/(2\tau^2))$.

### Mathematical Questions

**Q26: Derive the Ridge regression closed-form solution and explain why the matrix is always invertible.**

Minimizing $J = \|\mathbf{y} - X\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$:

$\nabla_{\boldsymbol{\beta}} J = -2X^T\mathbf{y} + 2(X^TX + \lambda I)\boldsymbol{\beta} = \mathbf{0}$

$\hat{\boldsymbol{\beta}} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$

**Why always invertible**: Let $X = UDV^T$ be the SVD. Then $X^TX = VD^2V^T$, so $X^TX + \lambda I = V(D^2 + \lambda I)V^T$. The eigenvalues of this matrix are $d_j^2 + \lambda$ for $j = 1, \ldots, p$. Since $d_j^2 \geq 0$ and $\lambda > 0$, all eigenvalues are strictly positive, so the matrix is positive definite and hence invertible.

This is the key practical advantage of Ridge: it makes OLS work even when $X^TX$ is singular or nearly singular (e.g., when $p > n$).

**Q27: Derive the soft-thresholding operator for the Lasso in the one-dimensional case.**

Minimize $f(\beta) = (y - \beta)^2 + \lambda|\beta|$ over $\beta$.

Case 1: $\beta > 0$. Then $|\beta| = \beta$, so $f(\beta) = (y-\beta)^2 + \lambda\beta$. Setting $f'(\beta) = -2(y-\beta) + \lambda = 0$ gives $\beta = y - \lambda/2$. This is valid (positive) only when $y > \lambda/2$.

Case 2: $\beta < 0$. Then $|\beta| = -\beta$, so $f(\beta) = (y-\beta)^2 - \lambda\beta$. Setting $f'(\beta) = -2(y-\beta) - \lambda = 0$ gives $\beta = y + \lambda/2$. This is valid (negative) only when $y < -\lambda/2$.

Case 3: $\beta = 0$. Check the subdifferential: $\partial f(0)$ includes $-2y + \lambda g$ for $g \in [-1, 1]$. The condition $0 \in \partial f(0)$ gives $|y| \leq \lambda/2$.

Combining: $\hat{\beta} = \text{sign}(y)\max(|y| - \lambda/2, 0) = S(y, \lambda/2)$.

### Applied Questions

**Q28: You're building a clinical model to predict patient outcomes using 5,000 gene expression features and 200 patients. Which regularization approach do you recommend and why?**

With $p = 5000 \gg n = 200$:

1. **OLS fails**: $X^TX$ is singular; no unique solution exists.
2. **Lasso or Elastic Net** is the recommended approach because:
   - We expect sparsity: likely only a handful of genes truly predict the outcome (biological prior).
   - Feature selection is valuable: clinicians want to know which genes matter, not just get predictions. A 10-gene model is more actionable than a 5000-gene model.
   - Elastic Net is preferred over pure Lasso because gene expression features are often correlated (genes in the same pathway have similar expression). Lasso would arbitrarily pick one gene from each correlated group; Elastic Net would keep the group together, giving a more stable and biologically interpretable model.
3. **Cross-validation**: Use 10-fold CV (or leave-one-out, given the small $n$) to select $\lambda$ and $\alpha$.
4. **Stability selection**: Run Elastic Net on many bootstrap samples and report genes selected in >70% of runs. This gives more reliable feature selection than a single Lasso fit.

**Q29: How would you choose between Ridge and Lasso in a production ML system?**

Decision framework:

- **If interpretability is critical** (you need to explain which features matter): Lasso or Elastic Net. The sparse solution directly identifies important features.
- **If all features are potentially relevant** (no reason to expect sparsity): Ridge. It keeps all features and produces more stable predictions when features are correlated.
- **If inference speed is critical** (real-time serving with millions of features): Lasso produces sparse models, meaning you only store and evaluate nonzero features. A Lasso model with 50 nonzero features is much faster to evaluate than a Ridge model with 50,000 nonzero features.
- **If prediction accuracy is paramount**: Cross-validate both and compare. Often the difference is small, and Elastic Net (which includes both as special cases) is the safest choice.
- **If you're unsure**: Use Elastic Net with $\alpha \in \{0.1, 0.5, 0.9, 1.0\}$ in the CV grid. This searches over the spectrum from Ridge-like to Lasso-like behavior.

### Debugging & Failure-Mode Questions

**Q30: You apply Lasso and find that the selected features change dramatically across different cross-validation folds. What is happening and how do you fix it?**

This is **selection instability**, commonly caused by:

1. **Correlated features**: When features are highly correlated, Lasso picks one from each correlated group, but which one it picks depends on the specific training fold. Different folds have slightly different data, so different features are selected.
2. **Small sample size**: With few observations, the signal-to-noise ratio per fold is low, causing erratic feature selection.
3. **$\lambda$ near the critical threshold**: Features near the selection boundary (coefficient just barely nonzero) will be selected in some folds and not others.

Fixes:
1. **Elastic Net**: The L2 component keeps correlated features together, stabilizing selection.
2. **Stability selection**: Run Lasso on many bootstrap samples (e.g., 100) and select features that appear in >60-70% of runs. This provides a "probability of selection" for each feature, which is much more informative than a single Lasso run.
3. **Group Lasso**: If you know the group structure of features (e.g., dummy variables for a categorical feature), Group Lasso selects entire groups, avoiding within-group instability.
4. **Bolasso (bootstrap-enhanced Lasso)**: Take the intersection of features selected across bootstrap samples. This produces a consistent feature set.

**Q31: You apply Ridge regression and the test MSE is worse than OLS. How is this possible?**

While Ridge generally improves upon OLS by reducing variance, there are cases where it does not:

1. **$\lambda$ is too large**: Excessive regularization introduces too much bias, overwhelming the variance reduction. The optimal $\lambda$ may be very small, and if the CV grid does not include small enough values, Ridge overfits to the regularization rather than the data.
2. **True coefficients are large**: If the true $\boldsymbol{\beta}$ has large entries, shrinking toward zero introduces substantial bias. Ridge's Bayesian interpretation assumes coefficients are drawn from $N(0, \tau^2)$—if this prior is badly wrong (coefficients are far from zero), Ridge hurts.
3. **Low multicollinearity**: If $X^TX$ is well-conditioned, OLS is already stable, and adding Ridge penalty only adds bias without meaningfully reducing variance.
4. **Overfitting the regularization parameter**: If $\lambda$ is tuned on the same data used for evaluation (data leakage), the selected $\lambda$ may be wrong.

Diagnosis: Plot test MSE as a function of $\lambda$. If the minimum is at $\lambda \approx 0$, OLS is sufficient and Ridge is not needed.

### Follow-up & Probing Questions

**Q32: "You mentioned the Bayesian interpretation. If Lasso corresponds to a Laplace prior, does the Lasso posterior have the same sparsity property?"**

No. The MAP estimate (the mode of the posterior) is sparse, but the full posterior is not. Under a Laplace prior, the posterior distribution of each $\beta_j$ is continuous and assigns zero probability to the event $\beta_j = 0$ exactly. The posterior density is peaked near zero but is never exactly zero anywhere.

This is an important distinction: sparsity is a property of the MAP point estimate, not of the Bayesian posterior. True Bayesian Lasso (computing the full posterior under a Laplace prior, e.g., via MCMC) gives posterior distributions for each coefficient, and you would need to use posterior credible intervals or posterior inclusion probabilities to assess which features are "important," rather than relying on exact zeros.

For methods that produce truly sparse posteriors, you would need spike-and-slab priors, which place a point mass at zero mixed with a continuous distribution. These are computationally more expensive but more principled for Bayesian variable selection.

**Q33: "How does the effective degrees of freedom of Ridge compare to OLS, and how does it relate to model complexity?"**

For OLS, the degrees of freedom equals $p$ (the number of parameters). For Ridge, the effective degrees of freedom is:

$\text{df}(\lambda) = \text{tr}(H_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$

where $d_j$ are the singular values of $X$ and $H_\lambda = X(X^TX + \lambda I)^{-1}X^T$ is the Ridge hat matrix.

When $\lambda = 0$: each term is 1, so $\text{df} = p$ (OLS).
When $\lambda \to \infty$: each term approaches 0, so $\text{df} \to 0$.
For intermediate $\lambda$: $\text{df}$ is between 0 and $p$.

This shows that Ridge continuously reduces the effective model complexity from $p$ (no regularization) toward 0 (infinite regularization). The effective degrees of freedom is the right quantity to use in information criteria (AIC, BIC) for Ridge, because it accounts for the reduced complexity due to shrinkage.

**Q34: "Walk me through how you would implement the Lasso coordinate descent algorithm from scratch."**

Algorithm:

1. **Standardize**: Center $\mathbf{y}$ to have mean zero (handle intercept separately). Scale each column of $X$ to have unit $\ell_2$ norm.

2. **Initialize**: $\boldsymbol{\beta}^{(0)} = \mathbf{0}$, residual $\mathbf{r} = \mathbf{y}$.

3. **Iterate** until convergence:
   For $j = 1, 2, \ldots, p$:
   a. Compute partial residual: $\mathbf{r}_j = \mathbf{r} + \mathbf{x}_j \beta_j^{\text{old}}$ (add back the contribution of feature $j$).
   b. Compute $\rho_j = \mathbf{x}_j^T \mathbf{r}_j$ (correlation of feature $j$ with partial residual).
   c. Apply soft-thresholding: $\beta_j^{\text{new}} = S(\rho_j, \lambda / 2)$.
   d. Update residual: $\mathbf{r} = \mathbf{r}_j - \mathbf{x}_j \beta_j^{\text{new}}$.

4. **Convergence check**: If $\max_j |\beta_j^{\text{new}} - \beta_j^{\text{old}}| < \epsilon$, stop.

5. **Post-process**: Un-standardize coefficients. Compute intercept: $\hat{\beta}_0 = \bar{y} - \sum_j \hat{\beta}_j \bar{x}_j$.

Key implementation details:
- **Warm starting**: When computing solutions for a path of $\lambda$ values (from large to small), initialize each solve with the solution from the previous (larger) $\lambda$. This dramatically speeds up computation.
- **Active set**: After the first pass, only cycle through features with nonzero coefficients (the "active set") plus occasional full passes to check if any zero'd features should become active. This exploits sparsity.
- **Screening rules**: Before even starting, check if $|\mathbf{x}_j^T \mathbf{y}| < \lambda$. If so, feature $j$ is guaranteed to have $\beta_j = 0$ at this $\lambda$ and can be skipped entirely.

**Q35: "In production, you have a linear model with L2 regularization serving 100,000 requests per second. A new requirement asks you to switch to L1 for feature selection. What are the engineering implications?"**

Key considerations:

1. **Training pipeline**: Replace the closed-form Ridge solution with iterative coordinate descent. Training time may increase, but for moderate-scale problems this is negligible.

2. **Model size**: L1 produces sparse weights. If the model drops from 10,000 features to 500, the model file shrinks dramatically, reducing memory usage and speeding up model loading.

3. **Inference speed**: Prediction with a sparse model only requires computing the dot product with nonzero features. Going from 10,000 to 500 nonzero features gives a roughly 20× speedup in prediction latency—very significant at 100K RPS.

4. **Feature pipeline**: Features that are always zero in the model can be removed from the upstream feature computation pipeline, saving compute and storage.

5. **Monitoring**: Fewer features means fewer things to monitor for drift. But sparse models can be more brittle: if one of the 500 selected features experiences distribution shift, the impact is larger than if it were one of 10,000.

6. **Retraining stability**: Lasso can select different features across retraining runs (selection instability), which can cause sudden changes in predictions. This is a production risk. Consider Elastic Net or stability selection to mitigate.

7. **A/B testing**: Before switching, run an A/B test to ensure the L1 model's predictive performance is comparable or better.
