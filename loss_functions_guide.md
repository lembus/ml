# Loss Functions: A Comprehensive Learning Guide

> A standalone reference for machine learning interviews and applied science work. No prior knowledge assumed. Each topic follows a six-section structure (Motivation & Intuition, Conceptual Foundations, Mathematical Formulation, Worked Examples, Relevance to Practice, Common Pitfalls) followed by tiered interview questions.

---

## Table of Contents

0. [Preface: What is a Loss Function?](#preface)
1. [Regression Losses](#regression-losses)
   - MSE, MAE, Huber, Log-Cosh, Quantile
2. [Classification Losses](#classification-losses)
   - Binary cross-entropy, hinge, squared hinge
   - Categorical cross-entropy / softmax loss
   - Focal loss, class-weighted cross-entropy
3. [Probabilistic Losses](#probabilistic-losses)
   - NLL, KL divergence, Wasserstein distance
4. [Custom Loss Functions](#custom-losses)
   - Domain-specific formulations, multi-task learning
5. [Loss Surface Analysis](#loss-surface)
   - Flat vs. sharp minima and generalization
6. [Cross-Topic Interview Question Bank](#cross-topic-qa)

---

<a name="preface"></a>
## 0. Preface: What is a Loss Function?

Before diving into specific families of loss functions, we need a precise grasp of what a loss function *is* and what role it plays in a machine learning system. This section is short but foundational; everything that follows builds on it.

### 0.1 The role of a loss function

Machine learning is, at its heart, the act of choosing a function $f_\theta$ from some family (parameterized by $\theta$) such that $f_\theta(x)$ is "close to" the true label $y$ for inputs $x$ drawn from some distribution. To turn this vague goal into something we can optimize, we need a **scalar score** that measures *how bad* a prediction is. That scalar score is the **loss**.

Formally, a loss function is a map

$$
\ell : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}
$$

where $\ell(\hat{y}, y)$ is the penalty incurred when the model predicts $\hat{y}$ but the truth is $y$. Lower is better; zero usually means a perfect prediction.

Three key roles the loss plays:

- **Optimization target.** Training is the process of minimizing the *expected* loss (or its empirical surrogate, the average loss on the training set). Gradient descent and its variants compute $\nabla_\theta \ell$ and step against it.
- **Specification of "good".** The choice of loss *defines* what we mean by "a good model". Two losses can produce two completely different models from the same data. Squared error makes the model regress toward the *mean*; absolute error makes it regress toward the *median*. There is no model-agnostic notion of "best" — only "best with respect to this loss".
- **Connection to probability.** Many losses are equivalent to negative log-likelihood under some probabilistic model. Choosing MSE is equivalent to assuming Gaussian residuals; choosing cross-entropy is equivalent to assuming a Bernoulli or categorical likelihood. Understanding this correspondence demystifies a great deal of ML.

### 0.2 Risk, empirical risk, and surrogates

We almost always care about the **expected risk** under the true data distribution $\mathcal{D}$:

$$
R(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}}\big[\ell(f_\theta(x), y)\big]
$$

We cannot compute this — we don't know $\mathcal{D}$ — so we minimize the **empirical risk** on $n$ training samples:

$$
\hat{R}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i)
$$

This is the **Empirical Risk Minimization (ERM)** principle. The gap between $R(\theta)$ and $\hat{R}(\theta)$ is the **generalization gap** and is the central object of study in learning theory.

A subtlety: the loss we *want* to minimize (e.g., classification error / 0-1 loss) is often non-differentiable or even discontinuous, so we minimize a **surrogate loss** that is differentiable and behaves as a proxy. Cross-entropy, hinge, and logistic loss are all surrogates for the 0-1 loss. A surrogate is **classification-calibrated** (or **consistent**) if minimizing the expected surrogate loss also minimizes the expected 0-1 loss in the limit of infinite data.

### 0.3 What makes a loss function "good"?

When designing or choosing a loss, look for:

- **Aligned with the downstream objective.** If you are predicting a price and care equally about over- and under-prediction in dollars, MAE is a better match than MSE. If you are training a search ranker, you probably want a ranking loss, not pointwise regression.
- **Differentiable (or at least sub-differentiable) almost everywhere.** Gradient methods need this.
- **Convex if possible at the model output level.** Even though deep networks make the *whole* problem non-convex, having a convex loss with respect to the model's *output* (logits or scores) is a property of cross-entropy and squared error and aids analysis.
- **Numerically stable.** Many losses (cross-entropy especially) need to be implemented in log-space to avoid overflow/underflow.
- **Robust to label noise / outliers** when appropriate. MAE and Huber tolerate outliers; MSE does not.
- **Calibrated.** Probabilistic predictions should match observed frequencies; some losses (cross-entropy) yield calibrated outputs by construction, while others (hinge) do not.

With this groundwork in place, we now turn to specific families.

---

<a name="regression-losses"></a>
## 1. Regression Losses

### 1.1 Motivation & Intuition

Regression problems ask the model to predict a *real-valued* target: house price, tomorrow's temperature, time-to-failure, click-through rate, etc. The loss must score how far a real-valued prediction $\hat{y}$ is from a real-valued truth $y$.

The natural starting point is the **residual** $r = y - \hat{y}$. Almost every regression loss is some scalar function of the residual: $\ell(\hat{y}, y) = \rho(r)$.

The key design questions are:

1. **How should we penalize large errors?** If a single bad prediction is catastrophic (e.g., medical dosing), we want to penalize large errors *very* heavily — quadratic or worse. If outliers are common and meaningless (e.g., sensor glitches), we want to be *robust*, capping or linearly penalizing large errors.
2. **Symmetric or asymmetric?** Sometimes overestimating costs more than underestimating (or vice versa). Inventory: overestimating demand costs storage; underestimating costs lost sales. These costs are rarely equal.
3. **Point estimate or interval?** Sometimes we want a single best guess (mean, median); sometimes we want a range (a 90% prediction interval). The loss controls which.

A concrete intuitive example: imagine you are predicting house prices. Most houses sell for $100k–$1M, but a rare mansion sells for $50M. If your model is trained with MSE, that one mansion will dominate the gradient updates — the model will be pulled enormously toward predicting higher values to reduce that one squared error. With MAE, the mansion contributes the same per-unit-error as a normal house, so the model focuses on getting the bulk of the data right.

That single intuition — **how aggressively does the loss respond to large residuals?** — explains nearly every difference among MSE, MAE, Huber, and Log-Cosh.

### 1.2 Conceptual Foundations

The five regression losses we cover are best understood on a single spectrum:

| Loss        | Behavior near zero | Behavior on large \|r\| | What it predicts                  |
|-------------|--------------------|-------------------------|-----------------------------------|
| MSE         | Quadratic          | Quadratic (unbounded)   | Conditional **mean** $E[y\|x]$    |
| MAE         | Linear             | Linear (unbounded)      | Conditional **median**            |
| Huber       | Quadratic          | Linear                  | Robust mean                       |
| Log-Cosh    | Quadratic          | Linear (smooth Huber)   | Robust mean (smooth)              |
| Quantile    | Asymmetric linear  | Asymmetric linear       | Conditional **quantile** $\tau$   |

**Key terms:**

- **Outlier:** a data point whose target is far from the bulk of the distribution, often due to measurement error or genuinely rare events.
- **Robust loss:** a loss whose influence function (gradient with respect to a residual) is bounded — meaning a single arbitrarily large residual cannot dominate the gradient.
- **Convexity:** all five losses are convex in the residual, so for linear models they have a unique global minimizer. For deep networks the loss surface is non-convex regardless.
- **Sub-differentiable:** MAE, Huber (at the elbow), and quantile are not differentiable at isolated points but have well-defined sub-gradients there. Optimizers handle this fine in practice.

**Underlying assumptions:**

- MSE assumes residuals are zero-mean Gaussian (homoscedastic in its simplest form). Deriving from MLE, $y \mid x \sim \mathcal{N}(f(x), \sigma^2)$ gives MSE.
- MAE assumes residuals are zero-median Laplacian: $y \mid x \sim \text{Laplace}(f(x), b)$.
- Huber and Log-Cosh make no clean parametric assumption; they are pragmatic compromises.
- Quantile loss is non-parametric for the conditional quantile; it makes no distributional assumption.

**What breaks under violations:**

- If you use MSE on heavy-tailed data, the *empirical* mean is a poor estimator (high variance), and your model will be pulled toward outliers.
- If you use MAE on data where the conditional distribution is genuinely Gaussian, you waste statistical efficiency: MAE has higher variance than MSE under Gaussian noise (it ignores magnitude information).
- If you use Huber with $\delta$ much larger than typical residuals, it degenerates into MSE; if much smaller, it degenerates into MAE. Choosing $\delta$ requires knowing the residual scale.

### 1.3 Mathematical Formulation

Let $r_i = y_i - \hat{y}_i$ denote the residual on example $i$.

#### 1.3.1 Mean Squared Error (MSE / L2)

$$
\ell_{\text{MSE}}(\hat{y}, y) = (y - \hat{y})^2, \qquad \mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

Sometimes written with a $\tfrac{1}{2}$ for clean derivatives.

**Derivation from Gaussian MLE.** Assume $y_i = f_\theta(x_i) + \epsilon_i$ with $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ independently. The log-likelihood is

$$
\log p(y_1, \dots, y_n \mid x, \theta) = \sum_{i=1}^n \left[ -\tfrac{1}{2} \log(2\pi\sigma^2) - \frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} \right]
$$

Maximizing this w.r.t. $\theta$ is equivalent to minimizing $\sum_i (y_i - f_\theta(x_i))^2$, which is MSE up to a constant. So **MSE = MLE under Gaussian noise**.

**Gradient:** $\frac{\partial \ell}{\partial \hat{y}} = -2(y - \hat{y}) = 2r$ (or $r$ with the $\tfrac{1}{2}$ convention). Linear in the residual — large errors produce large gradients.

**Optimal point estimate.** Minimizing $E[(Y - c)^2]$ over $c$ gives $c^* = E[Y]$. Hence the regressor learned by MSE estimates the conditional mean $E[y \mid x]$.

#### 1.3.2 Mean Absolute Error (MAE / L1)

$$
\ell_{\text{MAE}}(\hat{y}, y) = |y - \hat{y}|, \qquad \mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
$$

**Derivation from Laplace MLE.** Assume $\epsilon_i \sim \text{Laplace}(0, b)$, with density $\frac{1}{2b} \exp(-|r|/b)$. The log-likelihood is

$$
\sum_i \left[ -\log(2b) - \frac{|y_i - f_\theta(x_i)|}{b} \right]
$$

Maximizing gives MAE.

**Gradient:** $\frac{\partial \ell}{\partial \hat{y}} = -\text{sign}(y - \hat{y}) = -\text{sign}(r)$ for $r \neq 0$. The gradient is **constant in magnitude** — a residual of 1000 produces the same magnitude gradient as a residual of 1.

**Sub-gradient at zero:** any value in $[-1, 1]$ is a valid sub-gradient. In practice optimizers use $0$.

**Optimal point estimate.** Minimizing $E[|Y - c|]$ over $c$ gives $c^* = \text{median}(Y)$. So MAE estimates the conditional median.

#### 1.3.3 Huber Loss

Huber smoothly interpolates between MSE (small residuals) and MAE (large residuals), with a threshold $\delta > 0$:

$$
\ell_{\text{Huber}}(r) =
\begin{cases}
\tfrac{1}{2} r^2 & |r| \leq \delta \\
\delta\big(|r| - \tfrac{1}{2}\delta\big) & |r| > \delta
\end{cases}
$$

The piecewise definition is constructed so the function and its first derivative are continuous at $r = \pm\delta$:

- At $|r| = \delta$: quadratic branch gives $\tfrac{1}{2}\delta^2$; linear branch gives $\delta(\delta - \tfrac{1}{2}\delta) = \tfrac{1}{2}\delta^2$. ✓
- Derivative at $r = \delta^-$: $r = \delta$. Derivative at $r = \delta^+$: $\delta$. ✓ (Second derivative jumps from 1 to 0; not $C^2$.)

**Gradient:**

$$
\frac{\partial \ell_{\text{Huber}}}{\partial r} =
\begin{cases}
r & |r| \leq \delta \\
\delta \cdot \text{sign}(r) & |r| > \delta
\end{cases}
$$

This is **bounded by $\delta$** in magnitude — that is what makes Huber robust. Outliers cannot produce arbitrarily large gradient updates.

**Choosing $\delta$.** Common heuristics: a multiple of the MAD (median absolute deviation) of residuals, e.g. $\delta = 1.345 \cdot \text{MAD}$ for 95% efficiency under Gaussian noise (a classic robust statistics result).

#### 1.3.4 Log-Cosh Loss

A smooth, $C^\infty$ approximation to Huber:

$$
\ell_{\text{log-cosh}}(r) = \log(\cosh(r))
$$

Series expansion: for small $r$, $\cosh(r) \approx 1 + r^2/2$ so $\log(\cosh(r)) \approx r^2/2$. For large $|r|$, $\cosh(r) \approx \tfrac{1}{2} e^{|r|}$, so $\log(\cosh(r)) \approx |r| - \log 2$. Thus log-cosh is quadratic near zero and linear in the tails — exactly Huber-like behavior, but with no parameter to tune and infinitely differentiable.

**Gradient:** $\frac{\partial \ell}{\partial r} = \tanh(r)$, which is bounded in $(-1, 1)$. So like Huber, gradients are bounded; like Huber, large residuals don't dominate.

**Numerical caution:** $\log(\cosh(r))$ overflows for large $r$ if computed naively because $\cosh(r) = (e^r + e^{-r})/2 \to \infty$. The stable form is

$$
\log(\cosh(r)) = |r| + \log(1 + e^{-2|r|}) - \log 2
$$

which is well-behaved for any real $r$.

#### 1.3.5 Quantile Loss (Pinball Loss)

For a chosen quantile level $\tau \in (0, 1)$:

$$
\ell_\tau(r) =
\begin{cases}
\tau \cdot r & r \geq 0 \\
(\tau - 1) \cdot r & r < 0
\end{cases}
= \max\big(\tau r, (\tau - 1) r\big)
$$

Equivalently, $\ell_\tau(y, \hat{y}) = (y - \hat{y})\big(\tau - \mathbb{1}[y < \hat{y}]\big)$.

**Intuition:** Underestimating ($r > 0$) costs $\tau \cdot r$; overestimating ($r < 0$) costs $(1 - \tau) \cdot |r|$. When $\tau = 0.5$, both costs are $0.5 |r|$ — proportional to MAE, so the median is recovered. When $\tau = 0.9$, underestimating is 9× more costly than overestimating, pulling the optimum toward the 90th percentile.

**Optimal estimate.** The minimizer of $E[\ell_\tau(Y - c)]$ over $c$ is the $\tau$-th quantile of $Y$. (Proof: differentiate w.r.t. $c$, set to zero, get $\tau = P(Y < c^*)$.)

**Use case:** Train *multiple* models — say with $\tau = 0.05, 0.5, 0.95$ — and you have a 90% prediction interval for free. This is a major practical use in demand forecasting, energy load prediction, and uncertainty quantification.

### 1.4 Worked Examples

#### 1.4.1 Why MSE chases the mean and MAE chases the median

Consider a single-feature toy dataset where for one specific input $x_0$ we have observed five targets:

$$
y \in \{1, 2, 3, 4, 100\}
$$

(One real-world value of 100 — perhaps a typo or a one-off luxury sale.)

We want a constant predictor $c$ for this input. Compute the optimum under each loss:

**Under MSE:** $\arg\min_c \frac{1}{5}\sum (y_i - c)^2 = \bar{y} = 22$.

**Under MAE:** $\arg\min_c \frac{1}{5}\sum |y_i - c| = \text{median}(y) = 3$.

The MSE estimate of 22 is *not close to any actual data point*. The single outlier of 100 dragged the mean far above the bulk. The MAE estimate of 3 sits squarely in the middle of the bulk.

**Under Huber with $\delta = 5$:** The outlier residual $|100 - c|$ uses the linear branch; the rest use the quadratic branch. Setting the derivative to zero,

$$
\sum_{|r_i| \leq 5} -(y_i - c) + \sum_{|r_i| > 5} -\delta \cdot \text{sign}(y_i - c) = 0
$$

For $c$ near the bulk, all residuals except 100 use the quadratic branch:

$$
-(1 - c) - (2 - c) - (3 - c) - (4 - c) - 5 \cdot \text{sign}(100 - c) = 0
$$

$$
-10 + 4c - 5 = 0 \implies c = 3.75
$$

Note this stays close to the bulk — the outlier contributes a *bounded* pull of magnitude $\delta = 5$, not the full $100 - c$.

**Under quantile $\tau = 0.9$:** Optimum is the 90th percentile, which for this 5-point distribution is somewhere in [4, 100]. Concretely, any $c$ such that 90% of points are at most $c$ — so $c = 100$ for the empirical distribution.

This single example crystallizes the design philosophy of each loss.

#### 1.4.2 Numerical example: gradient on an outlier

Suppose model output is $\hat{y} = 10$ and true value is $y = 1010$, so $r = 1000$.

| Loss      | Value at $r=1000$  | Gradient w.r.t. $\hat{y}$ |
|-----------|--------------------|---------------------------|
| MSE       | $1{,}000{,}000$    | $-2000$                   |
| MAE       | $1000$             | $-1$                      |
| Huber ($\delta=1$) | $999.5$   | $-1$                      |
| Log-Cosh  | $\approx 999.31$   | $-\tanh(1000) \approx -1$ |
| Quantile $\tau=0.5$ | $500$    | $-0.5$                    |

The MSE gradient is **2000× larger** than MAE/Huber/Log-Cosh on this single example. With even one such outlier in a minibatch, MSE training updates will be dominated by it. The robust losses limit the per-example gradient magnitude regardless of how big the residual gets.

#### 1.4.3 Building prediction intervals with quantile loss

Suppose we want to predict daily electricity demand and provide a 90% prediction interval. Approach: train three independent models on the same features, with different quantile loss levels:

- Model A: $\tau = 0.05$ → predicts the lower 5% boundary.
- Model B: $\tau = 0.5$  → predicts the median (point forecast).
- Model C: $\tau = 0.95$ → predicts the upper 95% boundary.

At inference, output $[A(x), B(x), C(x)]$. The interval $[A(x), C(x)]$ should contain 90% of true outcomes (if calibrated).

**Caveat — quantile crossing.** Independently trained quantile models can occasionally produce $A(x) > C(x)$ for some $x$, which is incoherent. Solutions: train jointly with a monotonicity constraint, or post-hoc sort.

### 1.5 Relevance to ML Practice

**MSE** is the default for continuous regression with low-noise, well-behaved targets. It is the loss implicitly assumed in linear regression, ridge regression, neural-network regression heads, and many forecasting methods. It is also the natural choice for *image reconstruction* in autoencoders (per-pixel MSE), variational autoencoders (Gaussian decoder log-likelihood), and many physics-based simulators.

**MAE** is preferred when the target distribution is heavy-tailed or contaminated with outliers, when you care about the *typical* error rather than the average squared error, and when the median is a more meaningful summary than the mean. Common in financial forecasting (where the occasional huge market move shouldn't dominate training) and demand forecasting at the product level.

**Huber and Log-Cosh** are go-to choices when you want robustness without giving up the smoother gradients near the optimum that quadratic losses provide. They train more stably than MAE because the gradient near zero is proportional to the residual (gentle steps when close to optimum) rather than a step function. This matters for SGD convergence — pure MAE can oscillate around the optimum because the gradient never shrinks. Huber is widely used in object detection (`smooth L1` loss in Faster R-CNN, SSD, etc., is exactly Huber with $\delta = 1$ for bounding-box regression).

**Quantile loss** is essential for any application that needs *uncertainty estimates* without a full Bayesian treatment: probabilistic forecasting (energy load, retail demand, ride-share ETAs at Uber/Lyft), risk modeling (VaR in finance), and decision systems where worst-case bounds matter. Gradient boosting libraries (XGBoost, LightGBM) all support quantile loss out of the box.

**Trade-offs.**

- *MSE vs MAE — bias–variance.* Under truly Gaussian noise, MSE is the maximum-likelihood estimator and is statistically efficient. MAE under Gaussian noise has higher variance. But under any non-Gaussian noise, MSE pays a price for its narrow assumption.
- *Huber vs MAE — convergence vs robustness.* Huber's smooth elbow gives better SGD convergence than MAE while preserving robustness. Cost: a hyperparameter $\delta$ to tune.
- *Log-Cosh vs Huber — smoothness vs flexibility.* Log-cosh has no hyperparameter and is $C^\infty$, but you can't directly tune the elbow location.
- *Quantile vs point estimates — information vs cost.* Quantile loss gives a richer output (full predictive distribution if you train enough quantiles) at the cost of training multiple heads/models.

**Computational cost.** All five are $O(1)$ per example to compute and differentiate — negligible compared to the model forward pass. The cost difference between losses is essentially zero; the choice is purely statistical.

### 1.6 Common Pitfalls & Misconceptions

1. **"Lower MSE always means a better model."** False — MSE on a different test set, or compared across datasets with different label scales, is uninterpretable. Always normalize (e.g., RMSE scaled by target standard deviation, or use $R^2$).

2. **Forgetting that MSE scales targets quadratically.** If your target is in units like "dollars" but ranges from 1 to $10^6$, the MSE values will be dominated by the high-magnitude examples even without true outliers. Many practitioners log-transform the target ($y \to \log(1+y)$) before applying MSE for this reason.

3. **Confusing MAE with "median of errors".** MAE is the *mean* of the absolute errors. The *median* absolute error is a separate metric (MedAE) and is even more robust.

4. **Using Huber without tuning $\delta$.** Default $\delta = 1$ is appropriate only if your residuals are already on a roughly unit scale. On predictions of, say, house prices in dollars with residuals in the thousands, $\delta = 1$ makes Huber behave essentially like MAE everywhere.

5. **Believing log-cosh is "free Huber".** It's elegant, but it has fewer knobs. If you genuinely need to tune robustness, use Huber.

6. **Quantile crossing.** As noted above — independently trained quantile heads can produce inconsistent intervals.

7. **Symmetric loss when costs are asymmetric.** Many real applications have asymmetric costs (over-predicting inventory ≠ under-predicting). Using MSE/MAE in such cases silently optimizes for the wrong thing. Use quantile loss with $\tau$ chosen to match the cost ratio: if over-predicting costs $c_o$ per unit and under-predicting costs $c_u$, then $\tau = c_u / (c_u + c_o)$.

8. **Interpreting MAE optimum as "robust mean".** It's the median, not a robust mean. The two coincide only for symmetric distributions.

9. **Ignoring that MSE is implied by your model architecture.** A linear regression head with no output activation, trained with cross-entropy, is mis-specified; conversely, a softmax head trained with MSE is a known anti-pattern (slow convergence, poor probabilistic calibration).

10. **Reporting RMSE on the training set as evidence of "good fit".** Training error is monotonically decreasing in capacity. Always evaluate on held-out data.

### 1.7 Interview Questions: Regression Losses

#### Foundational

**Q1. What is the difference between MSE and MAE, and when would you prefer each?**

MSE squares residuals, putting more weight on large errors; MAE takes the absolute value, weighting errors linearly. As a result, MSE estimates the conditional mean and is sensitive to outliers, while MAE estimates the conditional median and is robust. Prefer MSE when residuals are approximately Gaussian, the loss matches your business cost (squared dollar errors), or you want a smooth loss with gradients proportional to residual magnitude (helpful for optimization). Prefer MAE when the target distribution has heavy tails, outliers are common, or the median is the more meaningful summary statistic.

**Q2. Why is MSE the natural loss under Gaussian noise?**

Assuming $y = f(x) + \epsilon$ with $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the negative log-likelihood is $\frac{1}{2\sigma^2}\sum(y_i - f(x_i))^2$ plus a constant. So maximum-likelihood estimation under Gaussian noise is exactly MSE minimization. In other words, choosing MSE is equivalent to assuming Gaussian residuals. This connection is why MSE is the "right" loss for problems where the central limit theorem suggests Gaussian errors and "wrong" when residuals are heavy-tailed.

**Q3. What loss does Huber loss interpolate between, and why is this useful?**

Huber is quadratic for $|r| \leq \delta$ (like MSE) and linear for $|r| > \delta$ (like MAE). This gives you MSE's smooth gradient near zero — crucial for stable convergence — while bounding the gradient magnitude at $\delta$, giving MAE's robustness to outliers. Pure MAE has a constant-magnitude gradient that can cause optimizers to oscillate around the optimum; pure MSE blows up on outliers. Huber gets the best of both at the cost of one hyperparameter $\delta$.

**Q4. What does quantile loss optimize, and why is it useful?**

Quantile loss with parameter $\tau$ optimizes the model to predict the $\tau$-th conditional quantile of $y \mid x$. It penalizes underestimates by $\tau$ per unit and overestimates by $(1-\tau)$ per unit, an asymmetric absolute-error loss. This is useful for two reasons: (1) you can construct prediction intervals by training models at, e.g., $\tau = 0.05$ and $\tau = 0.95$; and (2) when over- and under-prediction have different real costs, the $\tau$-quantile is the cost-optimal point estimate, with $\tau$ matching the cost ratio.

**Q5. What is "log-cosh" loss and why does anyone use it?**

Log-cosh, $\log(\cosh(r))$, is a smooth approximation to Huber. Near zero it behaves like $r^2/2$; far from zero it behaves like $|r| - \log 2$. Its gradient is $\tanh(r)$, bounded in $(-1, 1)$, so it's robust like Huber. The advantages over Huber: it has no hyperparameter, and it is $C^\infty$ smooth, useful for second-order methods that need a continuous Hessian. The disadvantage: you can't tune the robustness threshold to match your data scale.

#### Mathematical

**Q6. Derive the optimal constant predictor under MAE.**

We want $\arg\min_c E[|Y - c|]$. Let $F$ be the CDF of $Y$. Then $E[|Y-c|] = \int_{-\infty}^c (c-y) dF(y) + \int_c^\infty (y-c) dF(y)$. Differentiating w.r.t. $c$:

$$
\frac{d}{dc} E[|Y-c|] = F(c) - (1 - F(c)) = 2F(c) - 1
$$

Setting to zero: $F(c^*) = 1/2$, i.e. $c^*$ is the median of $Y$. (At points where $F$ has a plateau, any value in the plateau is optimal.)

**Q7. Derive the gradient of Huber loss with respect to $\hat{y}$.**

$\ell(r) = \tfrac{1}{2}r^2$ for $|r| \leq \delta$, gradient $\partial\ell/\partial r = r$. For $|r| > \delta$, $\ell(r) = \delta(|r| - \tfrac{1}{2}\delta)$, gradient $\partial\ell/\partial r = \delta\,\text{sign}(r)$. Then $\partial \ell/\partial \hat{y} = (\partial\ell/\partial r)(\partial r/\partial \hat{y}) = -(\partial\ell/\partial r)$ since $r = y - \hat{y}$. So overall:

$$
\frac{\partial \ell}{\partial \hat{y}} =
\begin{cases}
\hat{y} - y & |y - \hat{y}| \leq \delta \\
\delta \cdot \text{sign}(\hat{y} - y) & \text{otherwise}
\end{cases}
$$

The gradient magnitude is bounded by $\delta$ — the source of the loss's robustness.

**Q8. Show that quantile loss has minimizer equal to the $\tau$-quantile.**

Consider $\rho_\tau(r) = (\tau - \mathbb{1}[r<0])\, r$. Compute $\partial E[\rho_\tau(Y-c)]/\partial c$. Splitting on whether $Y \geq c$ or $Y < c$:

$$
E[\rho_\tau(Y-c)] = \tau \int_c^\infty (y-c) dF(y) + (1-\tau) \int_{-\infty}^c (c-y) dF(y)
$$

Differentiating w.r.t. $c$:

$$
\frac{d}{dc} = -\tau (1 - F(c)) + (1-\tau) F(c) = F(c) - \tau
$$

Setting to zero: $F(c^*) = \tau$, i.e. $c^*$ is the $\tau$-quantile. ∎

**Q9. Why is the gradient of MAE at $r = 0$ "ill-defined" and why doesn't this break SGD?**

MAE is non-differentiable at $r = 0$ — its left derivative is $-1$ and right derivative is $+1$. Formally, the sub-differential at $r=0$ is the entire interval $[-1, 1]$. SGD doesn't break because (a) for almost all examples the residual is non-zero, so the gradient is well-defined; and (b) in the rare case $r = 0$ exactly, optimizers conventionally use $0$ as the sub-gradient, which is a valid choice. Theoretically, sub-gradient methods are guaranteed to converge for convex problems even at non-smooth points.

**Q10. Compare the asymptotic statistical efficiency of MAE vs MSE under Gaussian noise.**

Under Gaussian noise with variance $\sigma^2$, the variance of the sample mean (MSE estimator) is $\sigma^2/n$. The variance of the sample median (MAE estimator) is $\pi\sigma^2/(2n) \approx 1.57\sigma^2/n$ for large $n$. So MAE has about 64% relative efficiency vs MSE under Gaussian noise — you need ~57% more data to achieve the same estimator variance. Under Laplace noise the ranking flips: MAE is the MLE and is more efficient.

#### Applied

**Q11. You're predicting bounding box coordinates in object detection. Which regression loss should you use, and why?**

Use Smooth L1 (Huber with $\delta=1$). Reason: bounding box regression often has occasional very large errors during training (especially at the start when proposals are far from ground truth). Pure MSE on these large errors produces enormous gradients that destabilize training; pure MAE has a constant gradient that gives equal-magnitude updates near and far from optimum, slowing fine refinement. Huber gives stable, large updates when far away and gentle, proportional updates when close. This is exactly why the original Faster R-CNN and SSD papers use Smooth L1 for box regression but cross-entropy for classification.

**Q12. You're forecasting demand for thousands of products. Many products have demand spikes from rare promotions. Which loss?**

Two reasonable choices: (a) MAE if you mostly care about typical days; (b) quantile loss with multiple $\tau$ levels if you want to size safety stock (an inventory decision needs an upper bound on demand, not a point estimate). Avoid pure MSE, which would over-fit the spikes and inflate forecasts. A practical pattern: train a quantile model with $\tau = 0.5$ for the point forecast and $\tau = 0.95$ for the inventory decision threshold. Pure MSE would also be problematic if the spike is the *training-time* outlier you want to ignore but the *inference-time* event you want to forecast — a sign you may need a separate model for promotional periods.

**Q13. You replaced MSE with MAE and your model's predictions all collapsed to a single value. What might be happening?**

Several possibilities: (a) The model's expressive capacity for a particular subgroup is limited and MAE optimum is the median, which can be a single value if the conditional distribution is concentrated. (b) MAE has flat regions where many predictions yield identical loss (e.g., for an even-sized minibatch with grouped residuals), which can stall optimization. (c) Optimizer issues — MAE's constant-magnitude gradient may interact poorly with adaptive optimizers like Adam if learning rate is too large. (d) The data has a strong mode and MAE concentrates predictions there; MSE smooths over it. Diagnostic: plot prediction histograms — if all predictions fall on one or a few values, you have collapse; consider Huber instead, or a smaller learning rate.

**Q14. When predicting house prices that range from $50k to $50M, what loss and target transformation would you use?**

Apply $\log(1 + y)$ to the target, then train with MSE on log-prices. Three reasons: (1) Most percentage-error-style business decisions correspond to multiplicative, not additive, error — being off by 10% on a $100k house is comparable to being off by 10% on a $10M house, but in raw dollars these differ by 100×. Log-MSE captures this. (2) Heteroscedasticity: variance of price typically scales with price level; log-transform stabilizes variance. (3) The right-skewed price distribution becomes more symmetric in log-space, better matching MSE's Gaussian assumption. At inference, exponentiate the prediction back to dollar space, or report it directly in log space depending on downstream consumers.

#### Debugging & Failure Modes

**Q15. Your validation MSE keeps fluctuating wildly between epochs while your training loss decreases smoothly. What's going on?**

Almost certainly a few outliers in the validation set whose residuals dominate the squared sum. Each epoch, depending on which side of those outliers your model lands, validation MSE swings. Diagnosis: plot the per-example loss histogram on validation — you'll likely see a long tail. Fixes: (a) report robust validation metrics (MAE, median absolute error) alongside MSE; (b) clip or down-weight known-bad labels; (c) consider Huber loss in training to be less seduced by those points.

**Q16. You trained a model with Huber loss and got worse outlier-resistance than expected. Why?**

Likely $\delta$ is set too large relative to your residual scale. If most residuals are smaller than $\delta$, Huber operates in its quadratic branch almost always — i.e., it's MSE in disguise. Set $\delta$ to roughly the 50–80th percentile of training residuals (or use $1.345 \times \text{MAD}$ for a classical robust-stats default). Conversely, $\delta$ too small turns Huber into MAE everywhere, costing convergence near the optimum.

**Q17. Your quantile model with $\tau=0.05$ gives predictions consistently higher than your $\tau=0.95$ model. What happened?**

Quantile crossing — the two models were trained independently and there's no monotonicity constraint between them. This is a well-known failure mode. Fixes: (a) train a single multi-output model with a monotonicity penalty on the quantile dimension; (b) constrained optimization via parameterizing the upper quantile as $q_{0.05} + \text{softplus}(\Delta)$; (c) post-hoc isotonic regression to enforce monotonicity in $\tau$; (d) use a single model that outputs the parameters of a parametric distribution, then read quantiles off that. Pragmatic shortcut: at inference, sort the predictions if you only need 2–3 quantiles.

#### Probing / Depth-of-Understanding

**Q18. If you rescale all targets by a factor of 10, how do MSE, MAE, Huber, and quantile losses change?**

MSE scales by $100\times$ (quadratic in residual). MAE scales by $10\times$. Quantile scales by $10\times$. Huber: in the quadratic branch, scales by $100\times$; in the linear branch, scales by $10\times$. Because of this, **Huber's behavior depends on $\delta$ relative to the residual scale**, so rescaling targets without rescaling $\delta$ changes the effective elbow location. This is why $\delta$ needs to be set relative to your residual distribution, not as a fixed constant across problems.

**Q19. Can you motivate Huber loss from a maximum-likelihood perspective?**

Yes — Huber is the MLE under a "Gaussian-with-Laplace-tails" noise distribution. Specifically, the density $p(r) \propto \exp(-\ell_{\text{Huber}}(r))$ is a smooth bell with Gaussian-like center and Laplace-like tails. Equivalently, it's a Gaussian with infrequent contamination by a heavy-tailed distribution — a model of clean-data-plus-occasional-outliers. This provides a principled derivation, though Huber is more commonly justified pragmatically (interpolation between L1 and L2) than from this MLE viewpoint.

**Q20. How does the choice of regression loss affect bias and variance of the estimator?**

Bias-variance is loss-relative. Under the squared-error loss, the bias-variance decomposition $E[(\hat{y} - y)^2] = \text{Bias}^2 + \text{Variance} + \text{IrreducibleNoise}$ holds, and choosing models to minimize MSE balances these. Under MAE, no such clean additive decomposition exists; the appropriate decomposition is for the absolute error and involves the median rather than the mean. So the very *meaning* of bias-variance trade-off depends on the loss. Practically, robust losses tend to produce *biased* estimates (toward the median, away from the mean) but *lower variance* in the presence of outliers, which is a favorable trade in heavy-tailed settings.

**Q21. Suppose a colleague proposes using $\ell(r) = r^4$ as a regression loss for "even more outlier sensitivity." What do you say?**

It optimizes for a moment higher than the variance — minimizing $E[(Y-c)^4]$ doesn't yield the mean exactly but yields an even higher-influence estimator. The gradient grows as $r^3$, so a single outlier with residual 1000 produces a gradient of $10^9$ — completely catastrophic numerically. SGD will diverge unless extreme gradient clipping is used. There are theoretical situations where higher-order moment minimization makes sense (e.g., specific signal processing tasks targeting kurtosis), but for general regression it's a bad idea. The trend goes the *other* way: people move from MSE to MAE/Huber for robustness, not from MSE to higher orders for sensitivity.

**Q22. Is MSE always convex?**

MSE is convex *as a function of $\hat{y}$* for any fixed $y$ — it's a quadratic. It's convex in the model parameters $\theta$ if the model is linear in $\theta$ (linear regression). For a deep neural network, MSE composed with the network is generally non-convex in $\theta$ because the network is a non-convex function of its parameters. This distinction — convexity in the *output* vs in the *parameters* — is important: many "convex losses" (cross-entropy, hinge, MSE) refer to convexity in the output space.

---

<a name="classification-losses"></a>
## 2. Classification Losses

### 2.1 Motivation & Intuition

In classification, the target $y$ is *categorical* — one of $K$ classes. We can't simply take "class A minus predicted class A" because there's no meaningful arithmetic on category labels. The model output is typically a vector of $K$ scores or probabilities, one per class, and the loss must measure how good that score vector is given that exactly one class is correct.

The fundamental design choices:

1. **Probabilistic vs margin-based.** Probabilistic losses (cross-entropy) ask the model to output a *probability distribution* and reward placing high probability on the true class. Margin-based losses (hinge, squared hinge) ask the model to output a *score* and reward placing the true class's score above the others by at least some margin.
2. **Calibration.** Cross-entropy produces calibrated probabilities (under correct model specification) — a "0.7" prediction is correct ~70% of the time. Hinge does not.
3. **Robustness to misclassification confidence.** Cross-entropy is unbounded — a wildly confident wrong prediction (predicting probability 0.001 for the true class) incurs huge loss. Hinge is bounded once the margin is satisfied.
4. **Class imbalance.** When 99% of examples belong to one class, naive losses bias the model toward predicting only that class. Specialized losses (focal, weighted) re-balance the gradient to focus on the rare or hard-to-classify examples.

A concrete example: suppose you train a fraud detector where 1 in 1000 transactions is fraud. With standard cross-entropy, predicting "non-fraud" with probability 0.999 on every example gives a tiny average loss because 999 out of 1000 examples are non-fraud — the model has almost no incentive to ever predict "fraud". You need to either reweight examples (class-weighted cross-entropy) or down-weight easy examples (focal loss) to recover useful gradients on the rare class.

The same intuition extends across all classification losses: **the loss determines what the model spends its capacity on**.

### 2.2 Conceptual Foundations

**Key terms.**

- **Logits ($z$):** unnormalized model outputs, real-valued. For $K$ classes, the model produces $z \in \mathbb{R}^K$.
- **Probability vector ($p$):** $z$ passed through softmax (or sigmoid for binary), giving a distribution over classes.
- **One-hot label ($y$):** target encoded as a $K$-dimensional vector with a 1 in the true class position and 0s elsewhere. Sometimes a soft distribution is used instead (label smoothing, knowledge distillation).
- **Margin:** in margin-based losses, the gap between the score of the true class and the highest competing class. We want this positive and "comfortably" so.
- **Calibration:** a model is calibrated if predictions of probability $p$ are correct with frequency $p$.
- **Class-conditional:** distribution of features given a class; relevant when reasoning about prior shifts.

**Components and their interactions.**

1. The model produces logits $z = f_\theta(x)$.
2. A *link function* (softmax, sigmoid, identity) maps logits to scores or probabilities.
3. The loss compares the scored output to the label.
4. Backpropagation pushes logits in directions that reduce the loss.

For cross-entropy with softmax, the gradient with respect to logits has a beautifully clean form: $\partial \mathcal{L} / \partial z_k = p_k - y_k$. This is one reason this loss-link combination is so dominant.

**Underlying assumptions.**

- Cross-entropy assumes the model is producing class probabilities and the label distribution is the truth (one-hot or soft). It implicitly minimizes the KL divergence from predicted to true distribution.
- Hinge loss assumes a discriminative, margin-based view — we don't care about probabilities, only that the right class beats the wrong ones.
- Softmax assumes classes are *mutually exclusive*. For multi-label problems (an image can be both "cat" and "outdoor"), use independent sigmoids per label and binary cross-entropy.
- Focal loss assumes that the easy examples ($p_t \to 1$) provide diminishing learning signal and should be down-weighted.
- Class-weighted cross-entropy assumes the prior shift between training and deployment is known (or that class importance differs).

**What breaks.**

- Cross-entropy on imbalanced data without weighting: model collapses to predicting the majority class.
- Softmax on multi-label: can't represent more than one positive label (probabilities sum to 1).
- Hinge loss when calibrated probabilities are needed (e.g., for thresholding): outputs aren't probabilities.
- Focal loss on already-balanced data with no hard-example issue: can hurt performance by under-weighting useful gradient signal.

### 2.3 Mathematical Formulation

#### 2.3.1 Binary Cross-Entropy (BCE) / Logistic Loss

For binary classification, $y \in \{0, 1\}$ and the model outputs a single logit $z$. The probability of class 1 is $p = \sigma(z) = 1/(1 + e^{-z})$.

$$
\ell_{\text{BCE}}(z, y) = -\big[y \log p + (1-y) \log(1-p)\big]
$$

In terms of the logit:

$$
\ell_{\text{BCE}}(z, y) = \log(1 + e^{-z(2y-1)}) = \text{softplus}(-z(2y-1))
$$

Letting $\tilde{y} = 2y - 1 \in \{-1, +1\}$, the form becomes $\log(1 + e^{-\tilde{y}z})$, often called **logistic loss**.

**Derivation from Bernoulli MLE.** $y \sim \text{Bernoulli}(p)$ has likelihood $p^y (1-p)^{1-y}$. Negative log-likelihood is exactly BCE.

**Gradient w.r.t. logit:**

$$
\frac{\partial \ell}{\partial z} = p - y
$$

A startlingly clean expression: the gradient is just the prediction error. Three-line derivation: $\partial p/\partial z = p(1-p)$; $\partial \ell/\partial p = -y/p + (1-y)/(1-p) = (p-y)/(p(1-p))$; multiply.

**Optimal output.** For a fixed input $x$, minimizing $E[\ell_{\text{BCE}}(z, Y) \mid x]$ gives $p^* = P(Y=1 \mid x)$ — the true conditional probability. This is the **calibration property** of BCE.

**Numerical stability.** Computing $\sigma(z)$ then $\log p$ separately overflows/underflows for large $|z|$. The stable form is `softplus(-z*(2y-1))` directly, or in TF/PyTorch use the `with_logits` versions of the loss that fuse sigmoid and log.

#### 2.3.2 Hinge Loss

For binary classification with $\tilde{y} \in \{-1, +1\}$ and score $z$:

$$
\ell_{\text{hinge}}(z, \tilde{y}) = \max(0, 1 - \tilde{y} z)
$$

**Intuition.** If $\tilde{y} z \geq 1$ — the score has the right sign and magnitude $\geq 1$ — loss is zero. The model is "comfortably correct". If $\tilde{y} z < 1$ — either wrong sign or too small a margin — loss grows linearly. This is the loss optimized by a soft-margin SVM.

**Sub-gradient w.r.t. $z$:**

$$
\frac{\partial \ell}{\partial z} =
\begin{cases}
-\tilde{y} & \tilde{y} z < 1 \\
0 & \tilde{y} z \geq 1
\end{cases}
$$

The "support vectors" in SVM theory are exactly the examples with $\tilde{y} z < 1$ — the only ones contributing to the gradient.

**Key property: no calibrated probabilities.** Once $\tilde{y} z \geq 1$, the loss is zero and no further optimization happens for that example. The score $z$ doesn't represent a probability; you can't read $\sigma(z)$ as "P(class 1)" with any calibration guarantee.

#### 2.3.3 Squared Hinge

$$
\ell_{\text{sq-hinge}}(z, \tilde{y}) = \max(0, 1 - \tilde{y} z)^2
$$

Same hinge shape, but squared. Gradient: $-2\tilde{y}\max(0, 1-\tilde{y}z)$, which is *smooth* (continuously differentiable) at the kink $\tilde{y} z = 1$. Penalizes margin violations more aggressively than linear hinge.

**Trade-off vs hinge.** Squared hinge penalizes large margin violations quadratically, so it's *less* robust to mislabeled examples than linear hinge. But its differentiable gradient makes it nicer for second-order optimization. Used in some L2-SVM formulations and in some neural network classifiers.

#### 2.3.4 Categorical Cross-Entropy (CCE) / Softmax Loss

For $K$-way classification, model outputs logits $z \in \mathbb{R}^K$, mapped through softmax:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

Label is one-hot $y$ with $y_c = 1$ for the true class $c$. The loss is

$$
\ell_{\text{CCE}}(z, y) = -\sum_{k=1}^K y_k \log p_k = -\log p_c
$$

Often called "softmax loss" since softmax + cross-entropy are typically fused for numerical stability. The right formulation uses the **log-sum-exp trick**:

$$
\log p_k = z_k - \log\sum_j e^{z_j} = z_k - \text{LSE}(z)
$$

where LSE is computed stably as $\text{LSE}(z) = \max_j z_j + \log\sum_j e^{z_j - \max_j z_j}$.

**Gradient w.r.t. logits (the famous result):**

$$
\frac{\partial \ell_{\text{CCE}}}{\partial z_k} = p_k - y_k
$$

Derivation: $\partial p_k/\partial z_j = p_k(\delta_{jk} - p_j)$ (Jacobian of softmax). Then

$$
\frac{\partial \ell}{\partial z_j} = -\sum_k \frac{y_k}{p_k} \cdot p_k(\delta_{jk} - p_j) = -\sum_k y_k (\delta_{jk} - p_j) = p_j - y_j
$$

(using $\sum_k y_k = 1$). The same clean form as binary cross-entropy.

**Derivation from categorical MLE.** $y \sim \text{Categorical}(p)$ has likelihood $\prod_k p_k^{y_k}$. Negative log-likelihood is exactly CCE.

**Equivalence to KL divergence.** Since the label is fixed (one-hot or soft),

$$
\ell_{\text{CCE}} = H(y, p) = H(y) + D_{\text{KL}}(y \,\|\, p)
$$

The first term $H(y)$ is constant in $p$, so minimizing CCE is exactly minimizing $D_{\text{KL}}(y \,\|\, p)$.

**Multiclass softmax loss vs multilabel sigmoid loss.** For multiclass (one true class), use softmax + CCE. For multilabel (any subset of classes can be true), use $K$ independent sigmoids and sum BCEs over the $K$ outputs. They look similar but the assumptions differ — softmax enforces $\sum_k p_k = 1$, sigmoid does not.

#### 2.3.5 Class-Weighted Cross-Entropy

Add a per-class weight $w_c$ to address imbalance:

$$
\ell_{\text{wCCE}}(z, y) = -w_c \log p_c \quad \text{(with $c$ the true class)}
$$

Common weighting schemes:

- **Inverse frequency:** $w_c \propto 1/n_c$ where $n_c$ is the number of training examples with class $c$. Effectively gives every class equal total contribution to the loss.
- **Effective number of samples:** $w_c \propto (1 - \beta) / (1 - \beta^{n_c})$ for some $\beta \in [0, 1)$, reflecting that additional samples have diminishing returns due to overlap (Cui et al. 2019).
- **Cost-based:** $w_c$ derived from the business cost of misclassifying class $c$ (false-negative cost in fraud, false-alarm cost in medical screening, etc.).

**Effect on gradient.** Multiplies the per-example gradient by $w_c$, equivalent to over-sampling minority class examples.

#### 2.3.6 Focal Loss

Designed for extreme class imbalance (e.g., dense object detection where most anchors are background). Down-weights *easy* examples (those the model already gets right with high confidence) so gradient comes mostly from *hard* examples.

For binary classification, define $p_t$:

$$
p_t = \begin{cases} p & y = 1 \\ 1 - p & y = 0 \end{cases}
$$

so $p_t$ is the probability the model assigned to the true class. Focal loss is

$$
\ell_{\text{focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log p_t
$$

- $\gamma \geq 0$ is the **focusing parameter**; $\gamma = 0$ recovers BCE.
- $\alpha_t \in [0, 1]$ is an optional per-class weight (similar to class-weighted CE).
- $(1 - p_t)^\gamma$ is the **modulating factor**: when $p_t$ is high (easy example), it's near 0, suppressing the loss; when $p_t$ is low (hard example), it's near 1, leaving the loss unchanged.

**Why it works.** Consider $p_t = 0.9$ (easy) vs $p_t = 0.1$ (hard) under $\gamma = 2$:

- Easy: $(1 - 0.9)^2 = 0.01$ — loss is reduced 100×.
- Hard: $(1 - 0.1)^2 = 0.81$ — loss is barely reduced.

So gradient mass shifts toward hard examples. In the original RetinaNet paper, this enabled training a single-shot detector competitive with two-stage methods (which previously needed hard negative mining to handle the background/foreground imbalance).

**Multiclass focal loss.** Same idea on softmax outputs:

$$
\ell_{\text{focal-mc}}(z, y) = -\sum_k y_k (1 - p_k)^\gamma \log p_k
$$

Reduces to single-term form with the true class.

### 2.4 Worked Examples

#### 2.4.1 Binary cross-entropy with a numeric example

Logit $z = 2.0$, true label $y = 1$.

- $p = \sigma(2.0) = 1 / (1 + e^{-2}) \approx 0.881$
- $\ell = -\log(0.881) \approx 0.127$
- Gradient w.r.t. logit: $p - y = 0.881 - 1 = -0.119$. (Negative — push logit *up*.)

Now flip the label: $y = 0$.

- $\ell = -\log(1 - 0.881) = -\log(0.119) \approx 2.13$ — about 17× larger because the model was very wrong.
- Gradient: $p - y = 0.881 - 0 = 0.881$. (Positive — push logit *down* hard.)

This shows the asymmetry: the loss for a confidently-wrong prediction is much larger than for a confidently-right one, and the gradient magnitude is correspondingly larger.

#### 2.4.2 Categorical cross-entropy: 3-class example

Logits $z = [2.0, 1.0, 0.1]$, true class $c = 0$ (one-hot $y = [1, 0, 0]$).

Softmax:
- $\sum e^{z_j} = e^{2.0} + e^{1.0} + e^{0.1} \approx 7.389 + 2.718 + 1.105 = 11.21$
- $p = [0.659, 0.242, 0.0986]$

Loss: $-\log(0.659) \approx 0.417$.

Gradient w.r.t. logits: $p - y = [-0.341, 0.242, 0.0986]$.

The gradient subtracts 1 from the true class's softmax probability (since it should ideally be 1.0) and leaves others alone. Backprop propagates this through the network.

What if logits were $z = [0.1, 1.0, 2.0]$? Then $p = [0.0986, 0.242, 0.659]$, true class probability is only 0.0986, loss is $-\log(0.0986) \approx 2.32$. Gradient: $[-0.901, 0.242, 0.659]$ — large positive gradient on the (wrongly) confident class to push it down, large negative on the true class to push it up.

#### 2.4.3 Hinge loss vs cross-entropy: when do they differ in practice?

Take a two-class problem with one example, true label $\tilde{y} = +1$, model outputs score $z$.

- Hinge loss: $\max(0, 1 - z)$. Zero when $z \geq 1$.
- Logistic loss: $\log(1 + e^{-z})$. Approaches 0 only as $z \to \infty$.

Plot mentally:

| $z$  | Hinge | Logistic |
|------|-------|----------|
| -2   | 3.0   | 2.13     |
| 0    | 1.0   | 0.69     |
| 1    | 0.0   | 0.31     |
| 2    | 0.0   | 0.13     |
| 5    | 0.0   | 0.0067   |

For confidently-correct examples ($z \gg 1$), hinge is *exactly* 0 and contributes no gradient — the model can ignore it. Logistic still produces a (small) gradient, continuing to push the score higher. This means a logistic-trained model keeps refining confidence forever (which can lead to overconfidence), while a hinge-trained model stops once the margin is met (which can lead to under-confidence but better generalization on some problems).

#### 2.4.4 Why focal loss matters: imbalance in object detection

Consider 100,000 anchors per image in a dense object detector, of which only 100 are positives (containing an object) — a 1:1000 imbalance. Suppose the model after some training assigns:

- 99,000 of the 99,900 negatives correctly with $p_t = 0.99$ (very easy)
- 900 of the 99,900 negatives at $p_t = 0.7$ (somewhat hard)
- 100 positives at $p_t = 0.5$ (hard)

**Standard BCE total loss** (per image):

- Easy negatives: $99{,}000 \times -\log(0.99) \approx 99{,}000 \times 0.01005 \approx 995$
- Hard negatives: $900 \times -\log(0.7) \approx 900 \times 0.357 \approx 321$
- Positives: $100 \times -\log(0.5) \approx 100 \times 0.693 \approx 69$
- **Total ≈ 1385.** Easy negatives contribute 72% of the loss!

**Focal loss with $\gamma = 2$:**

- Easy negatives: $99{,}000 \times (0.01)^2 \times 0.01005 \approx 0.0995$
- Hard negatives: $900 \times (0.3)^2 \times 0.357 \approx 28.9$
- Positives: $100 \times (0.5)^2 \times 0.693 \approx 17.3$
- **Total ≈ 46.3.** Easy negatives contribute 0.2%; positives contribute 37%.

This dramatic reweighting — without explicit hard-example mining — is what made single-stage detectors viable.

### 2.5 Relevance to ML Practice

**Cross-entropy** is the workhorse classification loss — used in essentially every modern classifier from logistic regression to ImageNet-scale CNNs to transformer language models. The combination softmax+cross-entropy (often computed as a fused operation for numerical stability) underlies next-token prediction in LLMs, where the loss at each position is the categorical cross-entropy of the predicted distribution over the vocabulary.

**Hinge loss** dominated classification before deep learning's rise (linear SVMs, latent SVMs in computer vision). It still appears in margin-based methods (triplet loss in metric learning is hinge-like), in some representation learning contexts, and in problems where calibrated probabilities aren't needed and robustness to over-confident wrong examples is wanted.

**Squared hinge** is rarely used in modern practice; it appears in some classical formulations and certain solvers that benefit from continuous gradients.

**Class-weighted cross-entropy** is the simplest and most common fix for class imbalance — used routinely in medical imaging (rare disease detection), fraud detection, churn prediction, and any tabular ML problem with non-uniform classes. Most deep learning frameworks have a `class_weight` or `weight` parameter built in.

**Focal loss** is the standard choice for *severe* imbalance with many easy negatives — dense object detection (RetinaNet and successors), segmentation problems with large background areas, and click prediction in advertising (where the vast majority of impressions are non-clicks).

**Trade-offs.**

- *Cross-entropy vs hinge.* CE gives calibrated probabilities and unbounded gradients; hinge gives bounded gradients past the margin and no calibration. For modern deep models with softmax output heads, CE is almost always the default.
- *Cross-entropy vs MSE for classification.* MSE is a poor classification loss because (a) gradient w.r.t. logits is $(p - y) \cdot p (1-p)$ (sigmoid case), which vanishes when $p$ is near 0 or 1, slowing learning even when wrong — the *gradient saturation* problem; (b) it doesn't yield calibrated probabilities; (c) the optimal output is again the conditional mean, but for one-hot labels that's just $P(y=1|x)$, so MSE is *consistent* but inefficient. Use cross-entropy.
- *Class-weighted CE vs focal.* Class weights assume the imbalance is the only problem (the rare class examples are themselves uniformly hard). Focal loss assumes some examples are intrinsically easier than others regardless of class. Often you combine both ($\alpha$ for class weight, $\gamma$ for focusing).
- *Computational cost.* All these losses are O(K) per example for K classes; for very large $K$ (e.g., language model vocabularies of 100k+) the softmax+CE is the dominant cost in the head, and approximations like sampled softmax or hierarchical softmax are sometimes used during training.

**Where each is currently used.**

- LLMs: softmax + CE on vocabulary at each position.
- ImageNet classifiers: CE with optional label smoothing.
- Object detection: CE for classification of region proposals (Faster R-CNN), focal loss for dense detectors (RetinaNet, FCOS).
- Semantic segmentation: pixel-wise CE, often with class weights or focal variants for rare classes (medical, satellite).
- Recommendation: BCE on click probability, often with negative sampling.
- Few-shot / metric learning: triplet loss (a hinge variant), contrastive loss.

### 2.6 Common Pitfalls & Misconceptions

1. **Computing softmax then log separately.** Causes overflow for large logits and underflow for log of tiny probabilities. Always use the fused `cross_entropy_with_logits` / `log_softmax` + `nll_loss` pattern in PyTorch/TF.

2. **Using softmax for multi-label problems.** Softmax forces $\sum_k p_k = 1$, so it can't represent multiple positive labels. Use independent sigmoids and sum BCEs.

3. **Applying class weights to validation loss.** Class weights are a *training* device. If you weight the validation loss too, you can't compare across runs that use different weights. Report unweighted validation loss (and metrics like balanced accuracy or per-class F1) for comparability.

4. **Setting focal loss $\gamma$ too aggressively.** Very large $\gamma$ (say, $\gamma = 5$) suppresses gradient on all but the very hardest examples, which can hurt convergence. Empirically $\gamma \in [1, 2]$ is most common.

5. **Forgetting that focal loss already includes class re-weighting via $\alpha$.** People sometimes apply both class weights *and* $\alpha$ and get unexpectedly extreme weighting.

6. **Believing hinge loss outputs are probabilities.** They are uncalibrated scores. To get probabilities from a hinge-trained model you need post-hoc calibration (Platt scaling, isotonic regression).

7. **Using CE without label smoothing on noisy labels.** CE with one-hot targets pushes the true class probability all the way to 1.0, which on noisy labels means the model is encouraged to be confident in the wrong label. Label smoothing (replace one-hot $y$ with $(1-\epsilon)y + \epsilon/K$) regularizes this.

8. **Ignoring numerical issues in the focal loss modulating factor.** $(1-p_t)^\gamma$ can underflow when $p_t$ is very close to 1 or be unstable in float16. Implementations should use `log1p` or compute in log-space.

9. **Using class-weighted CE when the deployment prior matches training.** If your training data is naturally imbalanced and so is your deployment data, you may not want to upweight the rare class — the model will then over-predict it at deployment. Class weights make most sense when (a) you care about the rare class specifically, (b) the cost of misclassification is asymmetric, or (c) deployment and training priors differ.

10. **Comparing losses between models with different output dimensions.** A 1000-class CE has a different scale than a 10-class CE (max possible loss is $\log 1000$ vs $\log 10$ for a uniform prediction). Always look at per-class metrics or normalized losses when comparing.

### 2.7 Interview Questions: Classification Losses

#### Foundational

**Q1. Why is cross-entropy preferred over MSE for classification?**

Three reasons. (1) **Gradient behavior**: For sigmoid output and MSE, the gradient with respect to logits is $(p-y) \cdot p(1-p)$, which vanishes when $p$ is near 0 or 1 even if the prediction is wrong (the "vanishing gradient" / saturation problem). For sigmoid + BCE, the gradient is just $p - y$, which is large precisely when the prediction is wrong. (2) **Probabilistic interpretation**: CE corresponds to maximum-likelihood under a Bernoulli/categorical model and yields calibrated probabilities. MSE has no such interpretation in classification. (3) **Convexity in the output**: BCE is convex in the logit, MSE composed with sigmoid is non-convex even before considering the network.

**Q2. What's the difference between softmax cross-entropy and binary cross-entropy with sigmoid?**

Softmax cross-entropy is for *multiclass* (one of $K$ mutually exclusive classes); softmax forces the predicted probabilities to sum to 1. Binary cross-entropy with sigmoid is for *binary* (or independent multi-label) problems; each output is an independent probability. For multi-label classification (an image can have multiple objects), use sigmoid + BCE on each label, *not* softmax — because softmax can't represent multiple positives.

**Q3. What problem does focal loss solve?**

Focal loss addresses extreme class imbalance combined with many easy examples. In dense object detection, ~99.9% of anchor boxes are background (negative class), and most of those are *easy* — the model learns to classify them confidently early. Under standard cross-entropy these easy examples still contribute to the loss because their loss isn't zero, just small, and there are so many of them they dominate the gradient. Focal loss multiplies the per-example CE by $(1-p_t)^\gamma$, which goes to zero quickly for easy examples ($p_t$ near 1) but stays near 1 for hard examples ($p_t$ small). This lets the optimizer focus on hard examples without needing explicit hard-negative mining.

**Q4. What is hinge loss and when would you use it?**

Hinge loss for binary classification with $\tilde{y} \in \{-1, +1\}$ and score $z$ is $\max(0, 1 - \tilde{y}z)$. It's zero when the example is correctly classified with margin $\geq 1$ and grows linearly otherwise. It's the loss optimized by SVMs. Use it when you don't need calibrated probabilities, want robustness to extremely confident wrong predictions (bounded gradient), and want a loss with a "cutoff" where well-classified examples no longer contribute. In modern deep learning it's relatively rare; cross-entropy dominates because of its calibration and differentiability.

**Q5. How does class-weighted cross-entropy work, and what are common ways to set the weights?**

It multiplies the per-example loss by a weight $w_c$ depending on the true class, effectively up-weighting minority classes. Common choices: (a) inverse-frequency weights $w_c \propto 1/n_c$; (b) "balanced" weights $w_c = N / (K \cdot n_c)$; (c) effective-number weights based on $1/(1-\beta^{n_c})$ that account for diminishing returns from very large classes; (d) cost-based weights derived from business or clinical asymmetries in misclassification cost.

#### Mathematical

**Q6. Derive the gradient of cross-entropy with softmax with respect to the logits.**

Logits $z \in \mathbb{R}^K$, softmax $p_k = e^{z_k} / \sum_j e^{z_j}$, one-hot label $y$ with $y_c = 1$. Loss $\ell = -\sum_k y_k \log p_k$.

The Jacobian of softmax: $\partial p_k / \partial z_j = p_k (\delta_{jk} - p_j)$. (Verify for $j=k$: $p_k(1 - p_k)$; for $j \neq k$: $-p_k p_j$.)

Then $\partial \ell / \partial z_j = -\sum_k (y_k / p_k) \cdot p_k (\delta_{jk} - p_j) = -\sum_k y_k (\delta_{jk} - p_j) = p_j - y_j$ (using $\sum_k y_k = 1$).

So $\nabla_z \ell = p - y$. Linear in the prediction error. ∎

**Q7. Show that minimizing cross-entropy is equivalent to minimizing KL divergence from the true to the predicted distribution.**

For label distribution $y$ (could be one-hot or soft) and prediction $p$:

$$
H(y, p) = -\sum_k y_k \log p_k
$$

$$
D_{\text{KL}}(y \,\|\, p) = \sum_k y_k \log(y_k / p_k) = -\sum_k y_k \log p_k + \sum_k y_k \log y_k = H(y, p) - H(y)
$$

So $H(y, p) = H(y) + D_{\text{KL}}(y \,\|\, p)$. Since $H(y)$ doesn't depend on the model, minimizing CE w.r.t. model parameters is exactly minimizing KL divergence. ∎

**Q8. What's the optimal predicted probability under CE? Show why CE produces calibrated outputs.**

Consider a fixed input $x$. The expected loss is $E_{Y|x}[-\log p(Y|x)] = -\sum_k P(Y=k | x) \log p_k$. Minimizing over $p$ subject to $\sum p_k = 1$ via Lagrange multipliers:

$$
\frac{\partial}{\partial p_k}\left[-\sum_j P(Y=j|x) \log p_j + \lambda(\sum_j p_j - 1)\right] = -P(Y=k|x)/p_k + \lambda = 0
$$

So $p_k^* \propto P(Y=k | x)$. Combined with normalization, $p_k^* = P(Y=k | x)$. The minimizer is the *true* conditional probability — that's calibration.

**Q9. Derive the gradient of BCE with sigmoid w.r.t. the logit.**

$p = \sigma(z) = 1/(1+e^{-z})$, $\ell = -[y\log p + (1-y)\log(1-p)]$. Compute:

$\partial p / \partial z = p(1-p)$ (standard sigmoid derivative).

$\partial \ell / \partial p = -y/p + (1-y)/(1-p) = [-y(1-p) + (1-y)p] / [p(1-p)] = (p-y) / [p(1-p)]$.

So $\partial \ell / \partial z = (p-y) / [p(1-p)] \cdot p(1-p) = p - y$. ∎ The $p(1-p)$ terms cancel — this is what avoids the saturation problem MSE+sigmoid suffers from.

**Q10. Show that focal loss reduces to BCE when $\gamma = 0$.**

$\ell_{\text{focal}} = -\alpha_t (1-p_t)^\gamma \log p_t$. With $\gamma = 0$: $(1-p_t)^0 = 1$, so $\ell = -\alpha_t \log p_t$. Setting $\alpha_t = 1$ recovers BCE exactly. So focal loss is a strict generalization. ∎

**Q11. Derive the optimal soft prediction under quadratic-hinge loss and show why it doesn't yield calibrated probabilities.**

Consider $E[\max(0, 1 - \tilde{Y} z)^2 \mid x]$ where $\tilde{Y} \in \{-1, +1\}$ with $P(\tilde{Y} = +1 \mid x) = \eta(x)$. The expectation is

$$
\eta \cdot \max(0, 1-z)^2 + (1-\eta) \cdot \max(0, 1+z)^2
$$

Minimize over $z$. Within the region $-1 < z < 1$, both terms are active: $\eta(1-z)^2 + (1-\eta)(1+z)^2$. Setting derivative to zero: $-2\eta(1-z) + 2(1-\eta)(1+z) = 0 \Rightarrow z^* = 2\eta - 1$. So $z^* \in (-1, 1)$ is a linear function of $\eta$, not an inverse-sigmoid. The score doesn't represent log-odds, so $\sigma(z^*) \neq \eta$ — uncalibrated. To recover $\eta$ from $z^*$ you'd use the linear map $\eta = (z^* + 1)/2$, but only for examples with $-1 < z^* < 1$; outside that range hinge gives no probabilistic information at all.

#### Applied

**Q12. You're training a classifier on a dataset with 99% class A and 1% class B, and want high recall on B. What loss strategies should you try?**

Multiple options, often combined. (1) **Class-weighted CE** with $w_B \approx 99 \times w_A$. (2) **Oversampling** of class B in minibatches (equivalent to class weighting in expectation, sometimes more stable). (3) **Focal loss** with $\gamma$ around 2 — useful if many of the class A examples are easy. (4) **Threshold tuning at inference** — train with standard CE but lower the decision threshold to favor class B. (5) **Different decision metrics** — optimize directly for ROC AUC, PR AUC, or F-beta in evaluation; the loss is a surrogate. Often the best approach: train with weighted CE *and* tune the decision threshold separately. Always validate on a held-out set with the actual deployment imbalance.

**Q13. You're building a multi-label image classifier where each image can have any subset of 100 tags. Which loss?**

Use 100 independent sigmoid outputs and sum binary cross-entropy across them. Don't use softmax — softmax makes the labels mutually exclusive, which is wrong here. Optionally add per-tag class weights or focal weighting if some tags are rare.

**Q14. You're training an LLM and your loss curve plateaus at log(K) where K is your vocab size. What's wrong?**

$-\log(1/K)$ is the loss of uniform-distribution predictions — the model isn't learning anything. Check (a) bug in label/input alignment (next-token prediction shifted incorrectly); (b) attention masking errors that prevent any information flow; (c) initialization that produced flat softmax outputs at every position; (d) the data is genuinely random / unlearnable in your formulation; (e) gradient clipping too aggressive. Begin with a sanity check: train on a tiny memorizable subset (10 sequences) and confirm loss falls to near zero.

**Q15. Your model outputs need to be calibrated probabilities for downstream decision-making (e.g., medical risk score). You're considering CE vs hinge. Which and why? What if your CE-trained model is still uncalibrated?**

Use cross-entropy; it's calibrated by design at the optimum. Hinge produces uncalibrated scores. However, a CE-trained deep network is often *over-confident* in practice — predicting probabilities that are too high (or low) relative to actual frequencies. Modern deep nets with high capacity tend to interpolate the training data, pushing logits to extremes. Remedy: post-hoc calibration with temperature scaling (divide logits by a scalar $T > 1$ before softmax, choose $T$ on a held-out validation set to minimize NLL). Other options: Platt scaling (logistic regression on scores), isotonic regression, label smoothing during training, or Bayesian/ensemble methods.

#### Debugging & Failure Modes

**Q16. Your training loss is dropping but accuracy is stuck at the majority-class baseline. What's going on and how do you fix it?**

Almost certainly class imbalance with no rebalancing. The model is learning to confidently predict the majority class on every input, which monotonically reduces CE (because the average label is heavily biased toward majority) but doesn't help minority-class accuracy. Diagnose by looking at per-class precision/recall — you'll see ~100% on majority, ~0% on minority. Fix with class weights, oversampling, focal loss, or threshold tuning. Also consider whether the features actually carry signal for the minority class — sometimes the issue is that the rare class is genuinely hard to separate.

**Q17. After adding label smoothing, your model's confidence dropped substantially but accuracy improved slightly. Is this expected?**

Yes — exactly the intended behavior. Label smoothing prevents the model from driving logits to ±infinity in pursuit of one-hot targets, which has two effects: (1) lower max softmax probabilities (less confidence) — appearing as better calibration; (2) smoother decision boundaries that often generalize slightly better. The accuracy improvement is typically small (0.5–1%) but the calibration improvement is large. Common in image classification (Inception, ResNet) and machine translation.

**Q18. You added focal loss and your model is now under-fitting on the majority class. What went wrong?**

You may have set $\gamma$ too high (e.g., $\gamma > 3$), which is suppressing gradient on too many easy *correct* examples — including some you actually need to keep getting right. Or you stacked focal loss's $\alpha$ with separate class weights, doubly down-weighting the majority. Try $\gamma = 1$ first, validate, then go up only if needed. Also, focal loss is designed for *severe* imbalance ($\geq$ 1:100); if your imbalance is mild, plain CE (or weighted CE) is usually better.

**Q19. Your validation cross-entropy is going *up* while validation accuracy continues to go up. What does this mean?**

Classic signature of **overconfident wrong predictions**. Accuracy is unchanged when the model gets the same example right — but if it gets it right with higher probability over time, accuracy is unaffected; if it gets *some* examples wrong with growing confidence, those wrong examples contribute large CE values that dominate the average even as other examples improve. This is one of the standard motivations for early stopping based on a held-out probabilistic metric, label smoothing, or post-hoc calibration. Often appears in long training runs of deep classifiers without smoothing.

#### Probing / Depth-of-Understanding

**Q20. Why does the gradient of softmax+CE w.r.t. logits have such a clean form — and what's the practical significance?**

The "magical" cancellation $\partial \ell/\partial z = p - y$ comes from the chain-rule interaction: the softmax Jacobian factor $p_k(\delta_{jk} - p_j)$ exactly cancels the $1/p_k$ from $\partial \log p_k / \partial p_k$. The practical significance is enormous: (1) numerically stable — no division by tiny probabilities; (2) gradient never saturates (unlike sigmoid+MSE); (3) the magnitude is bounded by 2 per logit (since $p, y \in [0,1]$), aiding gradient-clipping and learning-rate scheduling; (4) implementations can fuse softmax and CE into a single op for speed and stability (`logsoftmax + nll`).

**Q21. Both BCE and squared error are "consistent" (in the limit yield Bayes-optimal classifier). Why isn't squared error used?**

Statistical consistency is necessary but not sufficient. Squared error has (a) saturation — gradient vanishes for confident wrong predictions, slowing learning; (b) implicit Gaussian-noise model on labels, which is wrong for binary labels; (c) optimal score under MSE is again $P(Y=1|x)$, so BCE and MSE *agree* on the optimum, but BCE gets there with a much friendlier optimization landscape. Empirically, BCE-trained neural classifiers converge in fewer epochs and to slightly better solutions. (Recent work like "Mean Squared Error vs Cross-Entropy in Classification" by Hui & Belkin 2021 has shown MSE can be competitive in some regimes — but this is an active research result, not the default practice.)

**Q22. What is "label smoothing", what loss does it correspond to, and what's its effect?**

Label smoothing replaces the one-hot label $y$ with $(1-\epsilon)y + \epsilon \cdot u$ where $u$ is a uniform distribution over classes. CE with the smoothed label is $-(1-\epsilon)\log p_c - (\epsilon/K)\sum_k \log p_k$. Equivalently, it's standard CE plus a uniform-prior KL term. Effects: (1) Discourages logit magnitudes from growing without bound (acts as regularization); (2) Improves calibration; (3) Can be viewed as a form of confidence penalty. Used in many production-quality models (Transformer for translation: $\epsilon = 0.1$). Trade-off: marginally lower top-1 accuracy in some settings, better calibration in nearly all.

**Q23. How does focal loss interact with the choice of optimizer and learning rate?**

Focal loss reduces the gradient magnitude on easy examples by a factor of $(1-p_t)^\gamma$, which can drop *several orders of magnitude*. The total gradient magnitude is much smaller than equivalent CE. Two practical implications: (1) you may need a *higher* learning rate than for plain CE to compensate; (2) gradient clipping thresholds should be reconsidered; (3) Adam's adaptive scaling helps absorb the change in gradient magnitude, so it's relatively forgiving — SGD with a fixed learning rate may need re-tuning. Check both losses with your specific setup; don't transfer hyperparameters blindly.

**Q24. If your dataset has noisy labels, which classification losses are more robust and why?**

Robust to label noise: losses with bounded gradients on individual examples — generalized cross-entropy ($L_q$), symmetric cross-entropy, and bounded losses like MAE-on-softmax. Standard CE is *unbounded*: a noisy label with $p_t \to 0$ produces $-\log(0) \to \infty$, and the gradient $p - y$ has magnitude $\leq 1$ but the model is pulled hard toward fitting the wrong label. Hinge loss is *somewhat* robust because once a wrong-but-confident prediction is in the wrong region, its gradient is bounded ($\leq 1$). Focal loss is *not* robust — it actually amplifies focus on hard (often noisy) examples. For noisy label problems, consider losses specifically designed for it (Generalized Cross Entropy by Zhang & Sabuncu 2018, Symmetric Cross Entropy by Wang et al. 2019), or co-teaching/sample-selection methods that are orthogonal to the loss choice.

**Q25. You've heard "softmax loss" and "cross-entropy loss" used interchangeably. Are they the same thing?**

In practice yes — "softmax loss" is shorthand for "softmax + cross-entropy loss", the fused operation. Strictly, cross-entropy is a general concept ($H(y, p) = -\sum y_k \log p_k$) that takes any probability distribution as input; the source of $p$ doesn't matter. "Softmax loss" specifies that $p$ comes from softmax over logits. The naming is sloppy but universal in the literature.

---

<a name="probabilistic-losses"></a>
## 3. Probabilistic Losses

### 3.1 Motivation & Intuition

Many ML problems aren't about predicting a *single value* — they're about predicting a *distribution*. Examples:

- A weather model that outputs not just "70°F tomorrow" but a probability distribution over temperatures.
- A language model that outputs a probability distribution over the next token.
- A generative model (VAE, diffusion, GAN) whose entire goal is to produce samples from a target distribution.
- A Bayesian neural network whose output is a posterior distribution over predictions.

For such problems, the loss must measure *distance between distributions*, not distance between scalars. This section covers the three most important distributional losses:

- **Negative log-likelihood (NLL):** the foundational loss; everything else can be viewed through this lens.
- **KL divergence:** measures how one distribution diverges from another.
- **Wasserstein distance:** geometric distance between distributions, sensitive to *where* mass is, not just whether it overlaps.

A concrete intuition for why these matter beyond classification: imagine training a generative model of handwriting. Two model distributions might both assign zero probability to most pixel arrangements but support different ones — KL divergence is then infinite or undefined (depending on direction), giving the optimizer *no signal*. Wasserstein distance still measures "how far the two distributions are from each other" in pixel space, providing a useful gradient. This is the core motivation for Wasserstein GANs and a beautiful example of how loss choice can determine whether training is feasible at all.

The same intuition that drove our regression-loss discussion applies here, just in distribution space: **what does the loss penalize, and what gradient does it produce when the prediction is far from the truth?**

### 3.2 Conceptual Foundations

**Key terms.**

- **Density:** $p(x)$ for continuous distributions, $P(X=x)$ for discrete. Throughout, we'll use $p$ to mean the model's distribution and $q$ (sometimes $p_{\text{data}}$) for the data distribution.
- **Likelihood:** the probability the model assigns to observed data, viewed as a function of model parameters.
- **Cross-entropy:** $H(q, p) = -E_{x \sim q}[\log p(x)]$. The expected NLL of model $p$ on data drawn from $q$.
- **Entropy:** $H(p) = -E_{x \sim p}[\log p(x)]$. Cross-entropy with itself.
- **KL divergence:** $D_{\text{KL}}(q \,\|\, p) = E_{x\sim q}[\log(q(x)/p(x))]$. Asymmetric "distance" from $q$ to $p$.
- **Wasserstein-$k$ distance:** $W_k(p, q) = (\inf_\gamma E_{(x,y)\sim\gamma}[\|x-y\|^k])^{1/k}$ where $\gamma$ ranges over couplings (joint distributions with $p, q$ as marginals). Most often $k=1$.
- **Coupling:** a joint distribution over pairs $(x, y)$ whose marginals are $p$ and $q$.
- **Earth-Mover (EM) distance:** intuitive name for $W_1$ — minimum "work" to transform pile $p$ into pile $q$ where work = mass × distance moved.

**Component interaction.**

NLL is a *function of model parameters*: given a parametric distribution $p_\theta$ and data points, NLL = $-\sum_i \log p_\theta(x_i)$. Minimizing it is MLE.

KL divergence is a *function of two distributions*: $D_{\text{KL}}(q \,\|\, p_\theta)$. When $q$ is the empirical data distribution (delta functions on training points), KL becomes essentially equivalent to NLL (up to a constant — see derivation below).

Wasserstein distance is also a *function of two distributions*, but uses geometry of the underlying space, not pointwise probability ratios. This is why it stays informative when the distributions don't overlap.

**Underlying assumptions.**

- NLL: model assigns positive density everywhere data could appear, model is well-specified or close to it, data is iid.
- KL: same as NLL, plus the second distribution must be absolutely continuous w.r.t. the first (no division by zero).
- Wasserstein: a metric is defined on the underlying space (e.g., Euclidean distance in $\mathbb{R}^d$).

**What breaks under violations.**

- NLL with model that assigns *zero* probability to an observed point: loss is $-\log 0 = +\infty$. Catastrophic in evaluation; in training, gradient may be undefined or huge.
- KL divergence between two distributions with disjoint support: $D_{\text{KL}}(q \,\|\, p) = +\infty$. No useful gradient.
- KL is asymmetric: $D_{\text{KL}}(q \,\|\, p) \neq D_{\text{KL}}(p \,\|\, q)$ in general; using the wrong direction can cause "mode-seeking" vs "mode-covering" behavior with very different consequences.
- Wasserstein on raw distributions: computationally expensive in high dimensions; the dual formulation needed for GAN training requires Lipschitz constraints (gradient penalty, weight clipping).

### 3.3 Mathematical Formulation

#### 3.3.1 Negative Log-Likelihood (NLL)

Given $n$ iid observations and a parametric model $p_\theta$:

$$
\mathcal{L}_{\text{NLL}}(\theta) = -\sum_{i=1}^n \log p_\theta(x_i) \quad \text{or} \quad -\frac{1}{n}\sum_i \log p_\theta(x_i)
$$

This is the *negative* log-likelihood; minimizing it = maximizing the likelihood = MLE.

**Examples.**

- **Bernoulli (binary):** $p_\theta(y) = \theta^y (1-\theta)^{1-y}$. NLL = $-[y\log\theta + (1-y)\log(1-\theta)]$. **This is exactly BCE.**
- **Categorical (multiclass):** $p_\theta(y) = \prod_k \theta_k^{y_k}$. NLL = $-\sum_k y_k \log \theta_k$. **This is exactly CE.**
- **Gaussian:** $p_\theta(y \mid x) = \mathcal{N}(\mu(x), \sigma^2)$. NLL = $\tfrac{1}{2}\log(2\pi\sigma^2) + (y-\mu)^2/(2\sigma^2)$. With fixed $\sigma^2$, **this is MSE up to constants.**
- **Laplace:** as derived earlier, **this is MAE up to constants.**
- **Poisson:** $p_\theta(y) = \lambda^y e^{-\lambda}/y!$. NLL = $-y\log\lambda + \lambda + \log(y!)$. Used for count regression.

So NLL is a *unifying framework*: most "named" losses you'll meet are NLL under specific distributional assumptions.

**Properties of MLE.** Under regularity conditions (identifiable model, smooth log-likelihood, etc.), the MLE is:

1. **Consistent**: $\hat{\theta}_n \to \theta^*$ (true parameter) as $n \to \infty$.
2. **Asymptotically normal**: $\sqrt{n}(\hat{\theta}_n - \theta^*) \to \mathcal{N}(0, I(\theta^*)^{-1})$ where $I$ is the Fisher information.
3. **Asymptotically efficient**: variance achieves the Cramér-Rao lower bound.

**Failure mode: misspecification.** If the true distribution $q$ is not in the model family $\{p_\theta\}$, MLE converges to the $\theta^*$ that minimizes $D_{\text{KL}}(q \,\|\, p_\theta)$ — the closest point in the model family to truth, in KL sense. This is a *projection*, not the truth itself.

#### 3.3.2 KL Divergence

Discrete:

$$
D_{\text{KL}}(q \,\|\, p) = \sum_x q(x) \log \frac{q(x)}{p(x)}
$$

Continuous:

$$
D_{\text{KL}}(q \,\|\, p) = \int q(x) \log \frac{q(x)}{p(x)} dx = E_{x\sim q}\left[\log \frac{q(x)}{p(x)}\right]
$$

**Properties.**

- **Non-negative:** $D_{\text{KL}}(q \,\|\, p) \geq 0$, with equality iff $p = q$ almost everywhere. (Proof: Jensen's inequality on the convex function $-\log$.)
- **Asymmetric:** generally $D_{\text{KL}}(q \,\|\, p) \neq D_{\text{KL}}(p \,\|\, q)$.
- **Not a metric:** fails symmetry and triangle inequality.
- **Decomposition:** $D_{\text{KL}}(q \,\|\, p) = H(q, p) - H(q)$. So minimizing KL over $p$ (with $q$ fixed) is equivalent to minimizing cross-entropy.

**The two directions matter.**

- **Forward KL: $D_{\text{KL}}(q \,\|\, p)$ ("data given model").** Penalizes $p$ being small where $q$ is large. Forces $p$ to *cover all the modes* of $q$ ("mode-covering" / "zero-avoiding"). When $q$ is multimodal and $p$ is unimodal (e.g., a single Gaussian), $p$ ends up spreading mass across all modes — possibly with most density between them.
- **Reverse KL: $D_{\text{KL}}(p \,\|\, q)$ ("model given data").** Penalizes $p$ being large where $q$ is small. Forces $p$ to *concentrate on a single mode* ("mode-seeking" / "zero-forcing"). Variational inference typically uses reverse KL (because computing it requires expectations under $p$, which we can sample from).

This distinction explains a lot of phenomena:
- VAEs (reverse KL in their ELBO) often produce blurry samples because they don't capture all modes sharply.
- Maximum likelihood (forward KL minimization, since data → empirical $q$) produces models that don't ignore modes but may smear over them.
- Mode collapse in GANs is a related phenomenon: the discriminator's signal effectively pushes the generator toward reverse KL behavior.

**Connection to NLL.**

Let $q$ be the empirical data distribution: $q(x) = \frac{1}{n}\sum_i \delta_{x_i}(x)$. Then

$$
D_{\text{KL}}(q \,\|\, p_\theta) = \frac{1}{n}\sum_i \log \frac{1/n}{p_\theta(x_i)} = -\frac{1}{n}\sum_i \log p_\theta(x_i) - \log n
$$

So minimizing $D_{\text{KL}}(q_{\text{empirical}} \,\|\, p_\theta)$ over $\theta$ is exactly minimizing average NLL (the $-\log n$ is constant). This is the formal sense in which **MLE = forward KL minimization**.

#### 3.3.3 Cross-Entropy as a Loss

We've already met cross-entropy as a classification loss. In the broader probabilistic setting:

$$
H(q, p) = -E_{x\sim q}[\log p(x)]
$$

Minimizing $H(q, p)$ over $p$ (with $q$ fixed) gives $p = q$ — same minimum as KL. The relationship $H(q,p) = H(q) + D_{\text{KL}}(q \| p)$ shows they differ only by the constant $H(q)$.

In practice, the line between "cross-entropy", "negative log-likelihood", and "forward KL" is mostly cosmetic — they're all the same objective from different vantage points.

#### 3.3.4 Wasserstein Distance

For two probability measures $p, q$ on a metric space $(X, d)$:

$$
W_k(p, q) = \left( \inf_{\gamma \in \Pi(p, q)} \int d(x, y)^k \, d\gamma(x, y) \right)^{1/k}
$$

where $\Pi(p, q)$ is the set of all couplings (joint distributions with marginals $p$ and $q$). Most common: $k = 1$ (Earth-Mover distance).

**Intuition.** Imagine $p$ and $q$ as piles of dirt. $W_1(p, q)$ is the minimum total "work" (mass × distance) to reshape pile $p$ into pile $q$. The infimum over couplings is the optimal transport plan — for each unit of mass in $p$, where do you ship it in $q$?

**Why it's better than KL for some problems.**

Consider two delta distributions $p = \delta_0$, $q = \delta_t$ on the real line.

- $D_{\text{KL}}(p \| q) = +\infty$ (assuming densities; for measures it's $+\infty$ when supports differ).
- $W_1(p, q) = |t|$.

KL gives no useful gradient as $t$ varies; Wasserstein gives a smooth, distance-proportional one. This is the crucial property exploited by Wasserstein GANs (WGAN, Arjovsky et al. 2017) — when generator and data distributions don't overlap (which they don't, in early training), KL/JS-divergence-based losses (like vanilla GAN) provide vanishing gradients, while Wasserstein doesn't.

**Kantorovich-Rubinstein duality.** Computing the infimum over couplings is intractable for high-dimensional distributions. But $W_1$ has a beautiful dual:

$$
W_1(p, q) = \sup_{\|f\|_L \leq 1} E_{x \sim p}[f(x)] - E_{x \sim q}[f(x)]
$$

where the supremum is over 1-Lipschitz functions $f$. This is what WGAN exploits: replace the discriminator with a "critic" $f$ that aims to maximize $E_p[f] - E_q[f]$, with $f$ constrained to be 1-Lipschitz (via weight clipping in original WGAN, or gradient penalty in WGAN-GP).

The generator is then trained to *minimize* the critic's value on its outputs — effectively minimizing an estimate of $W_1$ between data and generated distributions.

**Computational considerations.**

- Exact $W_p$ between two empirical distributions of $n$ samples is the optimal transport problem — solvable in $O(n^3 \log n)$ by linear programming.
- **Sinkhorn divergence** (Cuturi 2013): an entropy-regularized version, computable in $O(n^2)$ via the Sinkhorn algorithm. Extremely popular for differentiable transport in deep learning.
- **Sliced Wasserstein:** project to 1D and use the closed-form 1D Wasserstein, average over many random projections. Fast and gives a metric, though weaker than full Wasserstein.

### 3.4 Worked Examples

#### 3.4.1 NLL on a 1D Gaussian regression

Suppose model outputs both $\mu(x)$ and $\log\sigma^2(x)$ for each input. The NLL of one example $(x, y)$ is

$$
\ell = \tfrac{1}{2}\log(2\pi) + \tfrac{1}{2}\log\sigma^2 + \frac{(y-\mu)^2}{2\sigma^2}
$$

Two interesting limits:

- **Fixed $\sigma^2$:** the $\log\sigma^2$ term is constant, only $(y-\mu)^2$ matters → MSE.
- **Predicted $\sigma^2$:** if the model is unconfident at noisy input $x$, it can *raise* $\sigma^2$ to reduce the squared-error term, paying only $\tfrac{1}{2}\log\sigma^2$. This is **heteroscedastic regression** — the model learns aleatoric uncertainty.

Numeric example: $y = 5.0$, $\mu = 4.5$, $\sigma = 0.5$.
- $(y-\mu)^2/(2\sigma^2) = 0.25 / 0.5 = 0.5$.
- $\tfrac{1}{2}\log(2\pi \cdot 0.25) = \tfrac{1}{2}\log(1.571) \approx 0.226$.
- $\ell \approx 0.726$.

Now $\sigma = 2.0$: $(y-\mu)^2/(2\sigma^2) = 0.25 / 8 = 0.031$, $\tfrac{1}{2}\log(2\pi \cdot 4) \approx 1.612$, $\ell \approx 1.643$. The model is penalized for being unnecessarily uncertain.

With $\sigma = 0.1$ (very confident, but wrong): $(y-\mu)^2/(2\sigma^2) = 0.25/0.02 = 12.5$, $\tfrac{1}{2}\log(2\pi \cdot 0.01) \approx -2.07$, $\ell \approx 10.43$. Heavy penalty for confident wrong predictions.

This trade-off is the essence of probabilistic regression: predicting $\sigma$ correctly matters as much as predicting $\mu$ correctly.

#### 3.4.2 KL divergence between two Gaussians

For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$
D_{\text{KL}}(p \,\|\, q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \tfrac{1}{2}
$$

Special case: $D_{\text{KL}}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)) = \tfrac{1}{2}(\sigma^2 + \mu^2 - 1 - \log\sigma^2)$. **This is the closed-form KL term in the VAE ELBO** — the regularizer pulling the encoder posterior toward the standard normal prior.

Numeric: $\mu = 0.5$, $\sigma^2 = 0.4$. KL = $\tfrac{1}{2}(0.4 + 0.25 - 1 - \log 0.4) = \tfrac{1}{2}(-0.35 - (-0.916)) = \tfrac{1}{2}(0.566) \approx 0.283$.

The optimal latent encodings in a VAE are pulled toward $\mu = 0, \sigma = 1$, regularizing the latent space.

#### 3.4.3 Wasserstein vs KL on shifted Gaussians

Two 1D Gaussians, $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(t, 1)$. As $t$ varies:

- $D_{\text{KL}}(p \| q) = \tfrac{1}{2}t^2$ (using the formula above, $\sigma_1=\sigma_2=1$, $\mu_1-\mu_2 = -t$).
- $W_1(p, q) = |t|$. (For 1D Gaussians, Wasserstein has a closed form via inverse CDFs.)

For $t = 5$ (well-separated but overlapping):

- KL = 12.5.
- $W_1 = 5$.

Both grow with $t$ — fine. Now consider deltas $p = \delta_0$, $q = \delta_t$ (no overlap):

- KL = $+\infty$ for any $t \neq 0$.
- $W_1 = |t|$.

This is the canonical example. KL "saturates" to infinity when distributions don't overlap, providing no gradient. Wasserstein gives a smooth, useful signal.

#### 3.4.4 Cross-entropy vs MSE in soft-label distillation

In knowledge distillation, the student model is trained to match a "soft" label distribution $q$ produced by a teacher (typically softmax of teacher logits divided by temperature $T$). Two common losses:

- **MSE between probability vectors:** $\sum_k (p_k - q_k)^2$. Gradient: $2(p_k - q_k) \cdot \partial p_k/\partial z_j$ — involves the softmax Jacobian.
- **Cross-entropy:** $-\sum_k q_k \log p_k$. Gradient w.r.t. logits: $p_k - q_k$ (same beautiful form as one-hot CE).

CE is preferred — same numerical and optimization advantages as in standard classification, plus it's the proper scoring rule for matching a distribution. The temperature $T$ is included in both teacher and student softmax, and the loss is typically multiplied by $T^2$ to compensate for gradient scale.

### 3.5 Relevance to ML Practice

**NLL** is the workhorse of probabilistic ML:
- Generative modeling: language models, autoregressive image models (PixelCNN), normalizing flows are all trained by maximizing log-likelihood (= minimizing NLL).
- Probabilistic regression: heteroscedastic regression, Bayesian neural networks.
- Survival analysis: Weibull/Cox NLL.
- Density estimation: Gaussian mixture models, kernel density (in evaluation), normalizing flows.

**KL divergence** is fundamental to:
- Variational inference (ELBO = $-\text{NLL} + D_{\text{KL}}(q_\phi(z|x) \,\|\, p(z))$).
- VAEs, where the KL term regularizes the encoder.
- Knowledge distillation (matching student output to teacher distribution).
- Reinforcement learning policy regularization (TRPO, PPO have KL constraints to limit policy updates).
- Information bottleneck methods, mutual information estimation.

**Wasserstein distance** is used in:
- WGAN and variants (WGAN-GP, SN-GAN with Wasserstein loss).
- Optimal transport for domain adaptation, label transfer.
- Sinkhorn divergences for differentiable transport (cross-modal alignment, generative modeling).
- Distributional reinforcement learning (some variants).
- Evaluation: FID (Fréchet Inception Distance) is essentially $W_2$ between two Gaussian fits.

**Trade-offs.**

- *NLL vs proper scoring rules.* NLL is one example of a proper scoring rule (rewards calibrated probabilistic forecasts). Other proper scoring rules: Brier score (= MSE on probabilities), CRPS (continuous ranked probability score) for continuous predictions. NLL is most common but isn't always the best fit; CRPS, for instance, doesn't blow up on probability-zero observations.
- *KL forward vs reverse.* Discussed at length above. The asymmetry has real consequences for the kind of solution you get; choose based on whether you need mode-covering or mode-seeking behavior.
- *KL vs Wasserstein.* KL is computationally trivial when distributions are parametric (closed forms abound) but gives no gradient on disjoint supports. Wasserstein needs more machinery (dual formulation, Sinkhorn, etc.) but stays informative everywhere.
- *Computational cost.* NLL is essentially free to compute when the model is parametric. KL is closed form for many distribution pairs. Wasserstein requires solving (or approximating) optimal transport — $O(n^3)$ exact, $O(n^2)$ for Sinkhorn, $O(n \log n)$ for sliced Wasserstein.

### 3.6 Common Pitfalls & Misconceptions

1. **"KL divergence is a distance metric."** No — it's asymmetric and violates the triangle inequality. Symmetric variants exist: Jensen-Shannon divergence $JSD(p, q) = \tfrac{1}{2}D_{\text{KL}}(p \| m) + \tfrac{1}{2}D_{\text{KL}}(q \| m)$ where $m = \tfrac{1}{2}(p+q)$. Even JSD isn't a metric, but $\sqrt{\text{JSD}}$ is.

2. **Using forward vs reverse KL interchangeably.** They produce qualitatively different solutions. Be explicit about which direction you mean.

3. **Reporting NLL without specifying the base.** $\log_2$ vs $\log_e$ vs $\log_{10}$ change values by a constant factor. In language modeling, "perplexity" $= \exp(\text{NLL with natural log})$ standardizes this.

4. **Computing log-likelihood naively for tiny probabilities.** Always work in log-space throughout. `log(prob_a * prob_b * prob_c)` should be `log(prob_a) + log(prob_b) + log(prob_c)`. Use `logsumexp` for marginalization.

5. **Forgetting that KL diverges for distributions with disjoint supports.** This is precisely why Wasserstein is preferred for many generative modeling settings.

6. **Confusing cross-entropy and KL divergence.** They differ by a constant ($H(q)$). For optimization with respect to $p$, this constant doesn't matter — they're equivalent objectives. But the absolute values differ; don't compare CE values to KL values directly.

7. **Treating Wasserstein as universally better.** It has higher variance under sample-based estimation in high dimensions (the curse of dimensionality affects $W_p$ severely), and it doesn't have closed forms for most distributions. KL is often more practical when applicable.

8. **Implementing KL in VAEs without temperature/$\beta$ tuning.** The unweighted ELBO often produces "posterior collapse" — the model ignores the latent variable entirely because the KL term punishes any departure from the prior. $\beta$-VAE introduces a tunable coefficient; warm-up schedules anneal it from 0 to 1.

9. **Computing Wasserstein in high dimensions naively.** Sample complexity grows exponentially with dimension. In high-D applications (images), use Sinkhorn or sliced approximations, or work in a learned feature space (FID does this).

10. **Believing "WGAN solves all GAN problems."** WGAN's theoretical advantages (continuity of the loss in distribution space) don't trivially transfer to better samples. Many empirical comparisons show modern non-Wasserstein GANs (StyleGAN with non-saturating loss + R1 regularization) outperform WGAN-GP. The Wasserstein critic is also tricky to satisfy the Lipschitz constraint correctly.

### 3.7 Interview Questions: Probabilistic Losses

#### Foundational

**Q1. What's the relationship between NLL, cross-entropy, and KL divergence?**

For an empirical data distribution $\hat{q}$ and a model $p_\theta$:
- **NLL**: $-\frac{1}{n}\sum_i \log p_\theta(x_i)$
- **Cross-entropy** $H(\hat{q}, p_\theta)$: $-E_{x \sim \hat{q}}[\log p_\theta(x)] = $ NLL exactly (for empirical $\hat{q}$).
- **KL divergence** $D_{\text{KL}}(\hat{q} \| p_\theta)$: $H(\hat{q}, p_\theta) - H(\hat{q})$ = NLL minus a constant.

So minimizing NLL, cross-entropy, or forward KL (against the empirical distribution) are all the same optimization problem.

**Q2. What's the difference between forward KL and reverse KL, and when do you use each?**

Forward KL $D_{\text{KL}}(q \| p)$ penalizes $p$ being small where $q$ has mass — drives $p$ to "cover" all modes of $q$ (mode-covering). Reverse KL $D_{\text{KL}}(p \| q)$ penalizes $p$ being large where $q$ is small — drives $p$ to concentrate where $q$ is largest (mode-seeking). Forward KL is the natural objective when fitting a model to data (MLE). Reverse KL is natural when *summarizing* a complex true distribution with a simpler approximation $p$ that we can sample from (variational inference). VAEs use reverse KL on the latent posterior.

**Q3. Why does the Wasserstein distance often work better than KL/JS divergence for training generative models?**

Because Wasserstein is *continuous and informative even when the supports of the model and data distributions don't overlap*. KL and JS go to infinity (or $\log 2$) when supports are disjoint, providing zero useful gradient. In early GAN training the generator's output distribution typically doesn't overlap with the real data's, so vanilla GAN loss provides poor gradients. Wasserstein measures the *geometric* distance between distributions and gives a smooth, finite signal everywhere.

**Q4. What does "MLE = forward KL minimization" mean precisely?**

Let $\hat{q}$ be the empirical distribution of training data. Then $D_{\text{KL}}(\hat{q} \| p_\theta) = \sum_i \tfrac{1}{n}\log(1/n) - \tfrac{1}{n}\sum_i \log p_\theta(x_i) = -\log n - \tfrac{1}{n}\text{NLL}$. The first term doesn't depend on $\theta$, so $\arg\min_\theta D_{\text{KL}}(\hat{q} \| p_\theta) = \arg\min_\theta \text{NLL}(\theta) = $ MLE. So MLE is *exactly* KL minimization between the empirical data distribution and the model.

**Q5. What is a "proper scoring rule" and is NLL one?**

A scoring rule $S(p, y)$ is a loss assessing a probabilistic prediction $p$ given observed outcome $y$. It is **proper** if $E_{y \sim q}[S(q, y)] \leq E_{y \sim q}[S(p, y)]$ for all $p \neq q$ — i.e., reporting your true belief is optimal. Equivalently, the model is incentivized to be calibrated. NLL is proper (a foundational result; Savage 1971). Other proper scoring rules: Brier score (squared error on probabilities), CRPS (for continuous outcomes). Improper rules incentivize hedging or extreme reporting.

#### Mathematical

**Q6. Derive the closed-form KL between two univariate Gaussians.**

For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$
D_{\text{KL}}(p\|q) = E_p[\log p - \log q] = E_p\left[-\tfrac{1}{2}\log(2\pi\sigma_1^2) - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \tfrac{1}{2}\log(2\pi\sigma_2^2) + \frac{(x-\mu_2)^2}{2\sigma_2^2}\right]
$$

Use $E_p[(X-\mu_1)^2] = \sigma_1^2$ and $E_p[(X-\mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$:

$$
= \tfrac{1}{2}\log\frac{\sigma_2^2}{\sigma_1^2} - \tfrac{1}{2} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2}
$$

$$
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \tfrac{1}{2}
$$

**Q7. Show that KL divergence is non-negative.**

By Jensen's inequality applied to the convex function $-\log$:

$$
D_{\text{KL}}(q \| p) = E_{x\sim q}[-\log(p(x)/q(x))] \geq -\log E_{x\sim q}[p(x)/q(x)] = -\log \int p(x) dx = -\log 1 = 0
$$

with equality iff $p(x)/q(x)$ is constant a.s., which combined with normalization means $p = q$ a.e. ∎

**Q8. State the Kantorovich-Rubinstein duality for $W_1$ and explain why WGAN uses it.**

$W_1(p, q) = \sup_{f: \|f\|_L \leq 1} E_{x \sim p}[f(x)] - E_{x \sim q}[f(x)]$, where the supremum ranges over 1-Lipschitz functions. WGAN parameterizes a "critic" network $f_\phi$ to approximate this supremum and uses $E_{x\sim \text{data}}[f_\phi(x)] - E_{z\sim \text{noise}}[f_\phi(G(z))]$ as an estimate of $W_1(p_{\text{data}}, p_G)$. The generator then minimizes this estimate. The critical practical issue is enforcing the Lipschitz constraint: original WGAN clipped weights, WGAN-GP added a gradient penalty $\lambda E[(\|\nabla_x f(x)\| - 1)^2]$ to encourage $\|\nabla_x f\| \approx 1$.

**Q9. What's the gradient of the NLL of a Gaussian model with respect to mean and (log) variance?**

For one example $y$, model $\mathcal{N}(\mu, \sigma^2)$:

$\ell = \tfrac{1}{2}\log(2\pi\sigma^2) + (y-\mu)^2/(2\sigma^2)$.

$\partial \ell / \partial \mu = -(y-\mu)/\sigma^2$.

$\partial \ell / \partial \sigma^2 = 1/(2\sigma^2) - (y-\mu)^2/(2\sigma^4)$.

Often we parameterize $\log\sigma^2$ (call it $s$) for stability:
$\partial \ell / \partial s = (\partial \ell/\partial \sigma^2)(d\sigma^2/ds) = \sigma^2 \cdot [1/(2\sigma^2) - (y-\mu)^2/(2\sigma^4)] = \tfrac{1}{2}[1 - (y-\mu)^2/\sigma^2]$.

Note the gradient on $\mu$ is the standardized residual; the gradient on $s$ is positive when the residual is small (overpredicting confidence) and negative when residual is large (underpredicting variance).

**Q10. Show that the KL between two distributions with disjoint support is infinite.**

Let $\text{supp}(q) \cap \text{supp}(p) = \emptyset$. Then for $x \in \text{supp}(q)$, $p(x) = 0$, so $\log(q(x)/p(x)) = +\infty$, and $D_{\text{KL}}(q \| p) = E_q[+\infty] = +\infty$. (For measures rather than densities, the KL is undefined / $+\infty$ when $q$ is not absolutely continuous w.r.t. $p$.)

#### Applied

**Q11. You're training a VAE and notice "posterior collapse" — the model ignores the latent variable. What happened and how would you fix it?**

The encoder is producing $q_\phi(z|x) \approx p(z) = \mathcal{N}(0, I)$ for all $x$, so the decoder receives essentially noise and learns to ignore $z$, behaving like an unconditional generator. This minimizes the KL term to zero at the cost of high reconstruction loss — but if the decoder is powerful enough to model the data marginal anyway (as with autoregressive decoders), this is locally optimal. Fixes: (1) **KL annealing** — start with $\beta = 0$ on the KL term, gradually increase to 1; (2) **$\beta$-VAE with $\beta < 1$** to permanently downweight KL; (3) **Free bits** — treat the KL as zero whenever it falls below a threshold per latent dimension; (4) **Less powerful decoder** so it actually needs $z$; (5) **Architectural changes** — e.g., conditional priors, hierarchical latents.

**Q12. You're predicting conditional probabilities and want to use Wasserstein distance instead of KL. Where would Wasserstein make sense, where wouldn't it?**

Wasserstein is appropriate when the *output space* has a meaningful metric and small geometric perturbations are conceptually small. Examples where it makes sense: (a) predicting a distribution over geographical locations (latitude/longitude); (b) age estimation as a soft distribution over years (so predicting "30" when truth is "31" is closer than predicting "70"); (c) ordinal regression with many levels. Where it doesn't make sense: (a) standard categorical classification where labels have no geometric structure (cat vs dog vs car has no notion of "distance between classes"); (b) one-hot binary tasks. Even when conceptually appropriate, you must pay the computational cost.

**Q13. You're training a language model and want to quantify how surprising new test data is to your model. How do you do this and why do we report perplexity?**

Compute the average per-token NLL on test data: $\bar{\ell} = -\frac{1}{N}\sum_t \log p(w_t \mid w_{<t})$ (using natural log). Report **perplexity** $= \exp(\bar{\ell})$. Perplexity has the interpretation of "effective branching factor" — equivalent to an N-way uniform distribution. A model with perplexity 50 is "as uncertain as choosing uniformly among 50 words", which is more interpretable than a raw NLL value. Perplexity is also unit-invariant w.r.t. the log base used (since $\exp(\log_e) = e^{\log_e}$, etc.).

**Q14. In knowledge distillation, why use temperature-scaled softmax and what loss?**

Soften the teacher's output distribution by dividing logits by temperature $T > 1$ before softmax: $p^T_k = \exp(z_k/T) / \sum_j \exp(z_j/T)$. Higher $T$ produces softer, more spread-out distributions that reveal the teacher's "dark knowledge" — relative similarities between non-target classes that get squashed at $T = 1$. The student is trained to match $p^T$ via cross-entropy or KL divergence. Cross-entropy is preferred for the same reasons it beats MSE in standard classification (clean gradient, calibrated). The total loss usually combines a soft-target CE (with $T > 1$, scaled by $T^2$) and a hard-label CE (with $T = 1$).

#### Debugging & Failure Modes

**Q15. Your variational autoencoder reports very low KL but reconstructions look like noise. What's happening?**

Posterior collapse — the encoder is matching the prior, contributing zero KL, and the decoder has had to learn to reconstruct from essentially uninformative latent codes. Reconstructions therefore resemble the data marginal but lack input-specific detail. Apply the fixes from Q11.

**Q16. Your WGAN's critic loss is decreasing toward $-\infty$ instead of stabilizing. What's the problem?**

The Lipschitz constraint isn't being enforced. Without it, the critic can output arbitrarily large values and the supremum in the K-R duality is unbounded. Check: (a) weight clipping bounds — too lax allows critic to exceed 1-Lipschitz; (b) gradient penalty — verify it's actually computed and contributes meaningfully to loss; (c) gradient penalty $\lambda$ — too small lets critic slip; (d) batch norm in critic — known to interact badly with gradient penalty (use layer norm instead). Also: WGAN-GP penalizes $(\|\nabla_x f\| - 1)^2$, not $\max(0, \|\nabla_x f\| - 1)^2$; the soft "encouragement to be 1-Lipschitz everywhere" is the design.

**Q17. Your model's NLL on validation is much worse than training, but accuracy is similar. What does this tell you?**

The model is overconfident on validation. Confidence asymmetry: getting an example right with $p = 0.99$ vs $p = 0.6$ doesn't change accuracy, but contributes very different NLL. On training the model can be overconfident because it has memorized; on validation that overconfidence becomes calibration error and inflates NLL. This is a strong case for label smoothing during training and post-hoc temperature scaling for deployment.

**Q18. You have a model that produces a probability distribution over class labels. Should you evaluate it with accuracy, NLL, or Brier score?**

Depends on the use case. **Accuracy** matters if you'll act on the argmax and don't care about confidence. **NLL** is appropriate if you need calibrated probabilities for downstream Bayesian decision-making and care about *avoiding very confident wrong predictions*. **Brier score** is similar to NLL (both proper scoring rules) but is *bounded* and behaves more gracefully on probability-zero events; some practitioners prefer it for that reason. Best practice: report all three and probably ECE (expected calibration error) too.

#### Probing / Depth-of-Understanding

**Q19. Why is KL divergence asymmetric, and what's the deeper meaning?**

The asymmetry stems from the expectation being taken under one specific distribution. $D_{\text{KL}}(q \| p) = E_{q}[\log q/p]$ averages over events drawn from $q$. So the "weight" on each event reflects $q$'s view of likelihood. This means $D_{\text{KL}}(q\|p)$ measures the "extra bits" needed to encode samples from $q$ using a code optimized for $p$ — a coding-theoretic interpretation that is inherently directional. Symmetric alternatives (JSD, Hellinger) average over both views or compare in symmetric ways.

**Q20. Explain the bias-variance tradeoff in MLE.**

Under regularity conditions, MLE is asymptotically *unbiased* (zero bias as $n \to \infty$) and asymptotically *efficient* (smallest variance among unbiased estimators). For finite $n$, MLE often has small bias (e.g., for Gaussian variance, $\hat\sigma^2_{\text{MLE}} = (1/n)\sum(x_i - \bar x)^2$ underestimates $\sigma^2$ by a factor of $(n-1)/n$). MLE under model misspecification is *biased* in a different sense — it converges to the projection of the true distribution onto the model family, not to truth. Adding regularization (Bayesian priors → MAP estimation; or explicit penalties → penalized MLE) introduces bias to reduce variance and improve generalization, just like in regression.

**Q21. What's the difference between Wasserstein distance, KL divergence, and JSD geometrically?**

- KL is a "log-ratio" measure — looks at pointwise ratios of densities.
- JSD averages over both distributions and is symmetric, but still ignores geometry — only cares whether mass is shared at the *same point*.
- Wasserstein is a "transport" measure — cares about *where* mass is and the cost of moving it. Two distributions on disjoint sets can be very close in $W$ if the sets are nearby and very far if the sets are distant. KL/JSD don't see any difference.

In a sentence: KL/JSD are "vertical" (compare densities at the same location), Wasserstein is "horizontal" (account for geometric distance between locations).

**Q22. Sinkhorn divergence vs Wasserstein: trade-offs?**

Sinkhorn introduces an entropic regularization $\lambda H(\gamma)$ on the transport plan, which (a) makes the optimization strictly convex and solvable via Sinkhorn iteration in $O(n^2)$ rather than $O(n^3)$; (b) gives smoother (differentiable) transport plans; (c) has a *bias* — Sinkhorn divergence $S_\lambda(p, q) = T_\lambda(p,q) - \tfrac{1}{2}T_\lambda(p,p) - \tfrac{1}{2}T_\lambda(q,q)$ corrects this so $S_\lambda(p, p) = 0$. Use Sinkhorn when computational cost matters or you want smoother gradients; use exact Wasserstein when you have small samples and want exact transport for theoretical clarity.

**Q23. When you say a loss is "calibrated" or "consistent", what's the technical meaning?**

A loss $\ell$ is **classification-calibrated** if minimizing $E_x[\ell(f(x), Y)]$ (over $f$ with no constraints) yields a function $f^*$ such that $\text{sign}(f^*(x))$ agrees with the Bayes-optimal classifier $\text{sign}(2P(Y=1|x) - 1)$. Equivalent: minimizing the surrogate loss leads to minimizing the 0-1 loss in the limit of infinite data. **Consistent** is similar but a stronger statistical-learning-theory notion: an algorithm is consistent if its risk converges to the Bayes risk as $n \to \infty$. Calibration of the loss is necessary for consistency (with appropriate function class assumptions).

A separate notion of "calibrated probabilities" applies to model outputs: a model's predicted probabilities are calibrated if examples with predicted probability $p$ are correct with frequency $p$. Cross-entropy minimizes a loss whose Bayes optimum is the true probability — so a CE-trained model is calibrated *at the Bayes optimum*, but in finite samples or with limited capacity can be miscalibrated.

---

<a name="custom-losses"></a>
## 4. Custom Loss Functions

### 4.1 Motivation & Intuition

The losses we've covered so far are *general-purpose*: MSE, cross-entropy, etc. But many real ML problems have structure that these don't capture:

- **Asymmetric costs.** Predicting a benign tumor as malignant has a different cost than the reverse.
- **Ranking and metric learning.** "Make this recommended item appear higher than that one" isn't naturally a regression or classification.
- **Multi-task objectives.** A self-driving car perception model might predict bounding boxes, classes, depth, and segmentation simultaneously — combining four different losses.
- **Domain-specific evaluation.** SSIM for image similarity (matches human perception better than MSE), IoU for object detection bounding boxes, edit distance for sequence prediction, BLEU/ROUGE for machine translation/summarization.
- **Constraints and regularization.** Loss terms that enforce structural properties (sparsity, smoothness, monotonicity).

A custom loss is the right hammer when:

1. The downstream objective doesn't decompose into a standard loss family.
2. The standard loss optimizes for the wrong thing (high MSE on perception tasks where humans don't care about pixel-level accuracy).
3. You have multiple objectives that need to be combined.
4. There are constraints or priors you want to encode directly.

A concrete intuition: imagine training an image-super-resolution model. With MSE, the model produces blurry outputs because the optimal pixel-wise estimate under MSE is the *mean* of all plausible high-resolution images consistent with the low-resolution input — and that mean is blurry. The fix isn't a different general-purpose loss but a perceptual/adversarial loss that scores high-resolution outputs by how realistic they look in feature space, not pixel space. The loss design *itself* solves the modeling problem.

### 4.2 Conceptual Foundations

Custom losses come in several styles. Understanding the styles helps you design your own.

**1. Reweighting standard losses by example or feature.**

The simplest customization: take an existing loss and apply a per-example weight that captures problem-specific importance.

$$
\mathcal{L}_{\text{custom}} = \frac{1}{n}\sum_i w(x_i, y_i) \cdot \ell(\hat{y}_i, y_i)
$$

Examples: class-weighted CE, importance-sampled NLL, sample-weighted MSE for noisy labels, focal loss (which weights by hardness).

**2. Composite losses (weighted sum of multiple losses).**

When you have multiple objectives, combine them linearly:

$$
\mathcal{L}_{\text{multi}} = \sum_t \lambda_t \mathcal{L}_t
$$

Used everywhere in modern ML — detection (classification + box regression), VAE (reconstruction + KL), super-resolution (pixel + perceptual + adversarial), and most multi-task setups. The art is choosing the $\lambda_t$ weights, which we'll discuss.

**3. Custom probabilistic losses.**

If your output has a non-standard distribution, derive NLL for it. Examples:
- **Tweedie distribution** for insurance claims (mass at zero + continuous positive part).
- **Negative binomial** for over-dispersed count data.
- **Mixture density network** outputs: $p_\theta(y|x) = \sum_k \pi_k \mathcal{N}(\mu_k, \sigma_k^2)$ with NLL loss.
- **Censored or truncated likelihoods** in survival analysis.

**4. Pairwise / listwise losses.**

When the unit of supervision is comparisons or rankings rather than absolute values:

- **Triplet loss:** $\ell = \max(0, d(a, p) - d(a, n) + m)$ where $a$ = anchor, $p$ = positive, $n$ = negative, $d$ = embedding distance, $m$ = margin. Used in face recognition, metric learning.
- **Contrastive loss (NT-Xent in SimCLR):** generalize triplet to one positive vs many negatives via softmax over similarities.
- **Pairwise ranking (RankNet):** $\ell = -\log \sigma(s_i - s_j)$ for pair $(i, j)$ where $i$ should rank above $j$.
- **ListNet, ListMLE, LambdaRank:** listwise ranking losses based on relative orderings of an entire list.

**5. Surrogate losses for non-differentiable metrics.**

When the actual metric you care about is non-differentiable (IoU, BLEU, F1, ranking metrics like NDCG), you need a smooth surrogate.

- **Soft IoU:** approximate intersection-over-union with smooth set operations on probability maps.
- **Lovász hinge / Lovász softmax:** convex surrogate for IoU.
- **Soft DTW** for time series alignment.
- **Differentiable BLEU approximations** (REINFORCE-based or via Gumbel-softmax tricks for sampling).

**6. Constrained or regularized losses.**

Add penalty terms that encode prior knowledge or constraints:

- **L1 / L2 regularization** on weights (Lasso, ridge).
- **Total variation** for smooth image outputs.
- **Entropy regularization** for diverse policies in RL.
- **Lagrangian losses** for constrained optimization (with a learned dual variable).

### 4.3 Mathematical Formulation

#### 4.3.1 Asymmetric / cost-sensitive losses

Suppose false positives cost $c_{FP}$ and false negatives cost $c_{FN}$. The expected cost-weighted loss for binary classification is

$$
\mathcal{L}_{\text{cost}} = \frac{1}{n}\sum_i \big[ c_{FN} \cdot y_i \cdot \ell_+(\hat{y}_i) + c_{FP} \cdot (1-y_i) \cdot \ell_-(\hat{y}_i) \big]
$$

where $\ell_+$ and $\ell_-$ are the per-class losses (e.g., $-\log p$ and $-\log(1-p)$ for cross-entropy). The Bayes-optimal threshold shifts away from 0.5 toward favoring the class with lower per-error cost.

#### 4.3.2 Multi-task loss combinations

Linear combination:

$$
\mathcal{L}_{\text{total}} = \sum_t \lambda_t \mathcal{L}_t(\theta)
$$

Choosing $\lambda_t$ is critical. Three approaches:

**(a) Manual tuning / grid search.** Treat $\lambda_t$ as hyperparameters. Simple, often what people do, but expensive when there are many tasks.

**(b) Uncertainty-based weighting (Kendall et al. 2018).** Treat each task's loss as a Gaussian (or Laplacian) NLL with a learned variance $\sigma_t^2$:

$$
\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2} \mathcal{L}_t + \log \sigma_t
$$

The model jointly learns task losses and per-task uncertainties; high-uncertainty tasks get downweighted. Widely used for multi-task perception models. In practice, parameterize $s_t = \log \sigma_t^2$ to keep it stable: $\mathcal{L}_{\text{total}} = \sum_t e^{-s_t} \mathcal{L}_t + \tfrac{1}{2} s_t$.

**(c) Gradient-based balancing (GradNorm, PCGrad, MGDA).** Adjust weights based on gradient magnitudes/directions to prevent any single task from dominating or to handle conflicting gradients. More complex but can outperform uncertainty-based weighting.

#### 4.3.3 Triplet loss

For an anchor $a$, positive $p$ (same class as $a$), negative $n$ (different class):

$$
\ell_{\text{triplet}}(a, p, n) = \max(0, d(f(a), f(p)) - d(f(a), f(n)) + m)
$$

where $f$ is the embedding network, $d$ is typically Euclidean distance (squared Euclidean on normalized embeddings is also common), and $m > 0$ is the margin.

**Intuition:** the embedding of the anchor should be at least $m$ closer to its positive than to any negative. Triplets that already satisfy this give zero loss; "hard" or "semi-hard" triplets drive learning.

**Hard negative mining.** Random triplets are mostly easy (gradient zero). Online mining strategies pick hard triplets from each batch:
- Hardest negative: $\arg\max_n d(a, n)$ — most violating, but unstable.
- Semi-hard negative: nearest negative that's still farther than the positive — Schroff et al.'s FaceNet recipe.

#### 4.3.4 InfoNCE / NT-Xent (contrastive learning)

Generalizes triplet to one positive vs many negatives:

$$
\ell_{\text{InfoNCE}}(a, p, \{n_j\}) = -\log \frac{\exp(\text{sim}(a, p)/\tau)}{\exp(\text{sim}(a, p)/\tau) + \sum_j \exp(\text{sim}(a, n_j)/\tau)}
$$

where $\text{sim}$ is cosine similarity (post-projection, in SimCLR) and $\tau$ is temperature.

This is exactly the form of categorical cross-entropy where the "classes" are the candidate items and the correct "class" is the true positive. It's also an estimator of mutual information between anchor and positive (the "InfoNCE bound"; van den Oord et al. 2018).

#### 4.3.5 Soft IoU / Dice loss for segmentation

For binary segmentation with predicted probability map $p$ and ground-truth mask $y$:

$$
\text{IoU}(p, y) \approx \frac{\sum_i p_i y_i}{\sum_i p_i + \sum_i y_i - \sum_i p_i y_i}
$$

$$
\ell_{\text{soft-IoU}} = 1 - \text{IoU}(p, y)
$$

**Dice loss** is closely related:

$$
\ell_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}
$$

where $\epsilon$ avoids division by zero on empty masks. Dice and IoU are both "set-based" — they care about overlap, not pixel-by-pixel accuracy. They naturally handle severe class imbalance (background vs lesion), which is why they're standard in medical image segmentation.

A common choice in practice: combine Dice with cross-entropy, $\lambda_1 \mathcal{L}_{\text{Dice}} + \lambda_2 \mathcal{L}_{\text{CE}}$, to get the best of both — Dice's class-balance robustness and CE's pixel-wise calibration.

#### 4.3.6 Perceptual loss

Instead of comparing pixel values directly, compare feature representations from a pre-trained network:

$$
\ell_{\text{perceptual}}(\hat{y}, y) = \sum_l \|\phi_l(\hat{y}) - \phi_l(y)\|_2^2
$$

where $\phi_l$ is the $l$-th layer activation of (typically) a frozen VGG network. The intuition: features encode semantic content; matching them gives perceptually meaningful similarity (a slightly blurred image has small pixel MSE but very small perceptual difference too; an image with the same content but different style has large perceptual difference).

Used in super-resolution, style transfer, image translation. Often combined with adversarial loss for sharp results.

#### 4.3.7 Mixture density network NLL

A model with output $\{(\pi_k, \mu_k, \sigma_k)\}_{k=1}^K$ defines $p(y|x) = \sum_k \pi_k(x) \mathcal{N}(y; \mu_k(x), \sigma_k^2(x))$. The NLL is

$$
\ell = -\log \sum_k \pi_k \cdot \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left(-\frac{(y-\mu_k)^2}{2\sigma_k^2}\right)
$$

Computed stably via log-sum-exp. Handles multi-modal predictions where simple regression fails (e.g., trajectory prediction where multiple futures are plausible).

### 4.4 Worked Examples

#### 4.4.1 Multi-task loss for object detection

A detector predicts, per anchor, (a) class probabilities and (b) bounding-box offsets. The total loss is

$$
\mathcal{L}_{\text{det}} = \frac{1}{N_{\text{cls}}} \sum_i \ell_{\text{cls}}(p_i, y_i^{\text{cls}}) + \lambda \cdot \frac{1}{N_{\text{box}}} \sum_i \mathbb{1}[y_i^{\text{cls}} > 0] \cdot \ell_{\text{box}}(t_i, t_i^*)
$$

- $\ell_{\text{cls}}$: cross-entropy or focal loss over $K+1$ classes (including background).
- $\ell_{\text{box}}$: smooth-L1 (Huber) on regressed box offsets.
- $\mathbb{1}[y_i^{\text{cls}} > 0]$: only regress boxes for foreground anchors.
- $\lambda$: balance between classification and regression. Faster R-CNN uses $\lambda = 1$ after careful normalization.

Numerical example: for a single image with 1000 anchors, 5 foreground:
- $\ell_{\text{cls}}$: average cross-entropy over all 1000.
- $\ell_{\text{box}}$: average Huber over the 5 foreground.
- Total loss ≈ 0.7 (classification) + 0.4 (box) = 1.1.

If you forget the indicator, you'd be regressing background anchors to (often arbitrary) zero offsets, polluting gradients massively.

#### 4.4.2 Uncertainty-weighted multi-task loss

Suppose two tasks: depth regression ($\mathcal{L}_1 = $ MSE on depth) and semantic segmentation ($\mathcal{L}_2 = $ CE per pixel). Use Kendall et al.'s formulation with learned $s_1, s_2$ ($s = \log \sigma^2$):

$$
\mathcal{L}_{\text{total}} = e^{-s_1} \mathcal{L}_1 + e^{-s_2} \mathcal{L}_2 + \tfrac{1}{2}(s_1 + s_2)
$$

Initialize $s_1 = s_2 = 0$. During training, $s_t$ adjusts: if task $t$ has high noise / uncertainty (gradient bouncing around), the optimizer learns to raise $s_t$, which downweights $\mathcal{L}_t$ at the cost of paying $\tfrac{1}{2}s_t$. Equilibrium balances gradient contribution across tasks.

If after training you observe $s_1 = -2$ (i.e., $\sigma_1^2 = 0.135$), $s_2 = 1$ (i.e., $\sigma_2^2 = 2.72$), then segmentation is being treated as 20× higher-noise than depth, and its loss is being downweighted accordingly. You can use this to diagnose mis-balanced tasks.

#### 4.4.3 Triplet loss with concrete numbers

Embeddings are in $\mathbb{R}^2$ for clarity. Margin $m = 0.5$.

- Anchor $a = (1, 0)$, positive $p = (1.2, 0.1)$ (close), negative $n = (2, 0)$ (further).
- $d(a, p) = \sqrt{0.04 + 0.01} = \sqrt{0.05} \approx 0.224$.
- $d(a, n) = 1.0$.
- $\ell = \max(0, 0.224 - 1.0 + 0.5) = \max(0, -0.276) = 0$. Easy triplet, no gradient.

Now consider $n' = (1.3, 0)$ (closer):
- $d(a, n') = 0.3$.
- $\ell = \max(0, 0.224 - 0.3 + 0.5) = \max(0, 0.424) = 0.424$. Hard triplet: gradient pushes $a$ and $p$ closer, $a$ and $n'$ further apart.

If we sampled only random triplets, most would look like the first (zero loss, no gradient). Hard mining concentrates training on the second type.

#### 4.4.4 Soft Dice loss on a tiny segmentation example

3×3 binary mask. Ground truth $y = $ all zeros except $(1,1) = 1$ — a single positive pixel. Prediction $p$:

```
[[0.1, 0.1, 0.0],
 [0.1, 0.8, 0.1],
 [0.0, 0.1, 0.0]]
```

- $\sum p_i y_i = p_{1,1} \cdot 1 = 0.8$ (only one positive ground-truth pixel).
- $\sum p_i = 1.3$.
- $\sum y_i = 1$.
- Dice $= 2 \cdot 0.8 / (1.3 + 1) = 1.6 / 2.3 \approx 0.696$.
- $\ell_{\text{Dice}} = 1 - 0.696 \approx 0.304$.

Compare to BCE: $-\log(0.8)$ at the one positive plus log terms for each background pixel. The Dice loss emphasizes the *single missed positive*, while BCE averages it down across many easy negatives. This is precisely why Dice is preferred for severely imbalanced segmentation.

### 4.5 Relevance to ML Practice

**When to design a custom loss.**

- The standard loss optimizes for the wrong metric (e.g., MSE-trained super-resolution producing blurry images).
- You have multiple objectives needing combined optimization.
- Your evaluation metric is non-standard (IoU, ranking, BLEU).
- Your output has a non-standard distribution.
- You have asymmetric costs.

**When NOT to.**

- "I want a more sophisticated loss because it sounds more sophisticated." Standard losses work for >90% of problems; custom loss design adds engineering burden and failure modes.
- The problem can be reformulated to use a standard loss with reweighting (most class-imbalance situations).
- The custom loss has many hyperparameters and you don't have the budget to tune them.

**Multi-task learning specifics.**

Multi-task learning adds tasks for *better representations* (the hope is shared features generalize) and *deployment efficiency* (one model for many predictions). But it has known failure modes:

- **Task interference:** gradient from one task can be in tension with another, slowing convergence on both.
- **Negative transfer:** combined performance worse than single-task.
- **Loss scale dominance:** a task with naturally large loss values dominates gradients.

Mitigations: gradient surgery (PCGrad), uncertainty weighting, task-specific learning rates, task-specific subnetworks (mixture of experts, multi-gate MoE).

**Trade-offs.**

- *Standard vs custom.* Standard losses are well-understood, well-optimized, well-supported. Custom losses can encode problem structure precisely but require validation that they actually optimize what you want and are stable.
- *Single composite vs multiple stages.* Sometimes a multi-stage training (pretrain on one loss, fine-tune on another) is more stable than combined optimization.
- *Differentiable surrogates vs RL/policy-gradient.* Some non-differentiable metrics (BLEU, ROUGE, ranking metrics) can be optimized with REINFORCE or other policy-gradient methods, sidestepping the need for a smooth surrogate. Trade-off: high-variance gradients vs designing a possibly-misaligned surrogate.

**Computational considerations.**

- Per-pixel losses (Dice, soft IoU) over high-resolution images can be expensive; usually OK for typical resolutions.
- Pairwise losses (triplet) are $O(n^2)$ in batch size for full enumeration; usually mined to a tractable subset.
- Listwise ranking losses can be $O(n^2)$ or $O(n \log n)$ per query.
- Mixture density networks need stable log-sum-exp computation.
- Wasserstein/Sinkhorn surrogates have nontrivial cost as discussed earlier.

### 4.6 Common Pitfalls & Misconceptions

1. **Naive linear combination of losses without normalization.** If $\mathcal{L}_1$ has typical values around 100 and $\mathcal{L}_2$ around 0.1, even with $\lambda_1 = \lambda_2 = 1$ the first totally dominates. Always normalize losses to a comparable scale before combining, or use uncertainty weighting.

2. **Cross-entropy + Dice where the Dice term is unstable on empty masks.** Without an $\epsilon$ smoothing constant, Dice can explode or be exactly zero (degenerate gradients) when there are no positive pixels in a sample. Always add smoothing.

3. **Triplet loss with random negatives.** Most random triplets are trivially satisfied → zero gradient → no learning. Always do hard or semi-hard negative mining.

4. **Using cosine similarity in a contrastive loss without normalizing embeddings.** Cosine similarity assumes unit-norm vectors; without L2-normalization at the end of the encoder, the loss interacts with norm in unintended ways. Normalize.

5. **Forgetting that focal loss is itself a custom loss.** People treat it as standard, but it has hyperparameters and assumptions; it can hurt on balanced datasets.

6. **Soft-IoU loss with logits instead of probabilities.** IoU operates on $\{0, 1\}$ masks; soft IoU operates on $[0, 1]$ probabilities. Pass softmax/sigmoid output, not raw logits.

7. **Multi-task uncertainty weighting forgetting the $\log\sigma$ regularizer.** Without the $\frac{1}{2}\log\sigma^2$ term, the model trivially makes all $\sigma_t \to \infty$ to minimize loss to zero. The log term penalizes infinite uncertainty.

8. **Treating the perceptual loss as differentiable through the VGG.** It is — but make sure the VGG is frozen (no gradient updates to its parameters) and in eval mode (batchnorm fixed). Common bug.

9. **Custom loss without validating against held-out evaluation metric.** Your custom loss should produce models that score better on the *true* business or research metric. Always evaluate on the actual target metric, not just on the training loss.

10. **Reinventing existing losses badly.** Many "novel" losses are minor reparameterizations of focal, hinge, or weighted CE. Before designing from scratch, search the literature for similar problems.

11. **Asymmetric losses without recalibrating thresholds.** If you train with cost-weighted loss, the model's output probability distribution shifts. The decision threshold may need to move. Either recalibrate or compute the cost-optimal threshold post-hoc.

12. **Using Dice / IoU loss with logits-style outputs in multi-class.** Multi-class extensions need to be done per-class then averaged (macro Dice). Easy to get the indexing wrong.

### 4.7 Interview Questions: Custom Loss Functions

#### Foundational

**Q1. When should you write a custom loss instead of using a standard one?**

When (a) the standard loss optimizes a different objective than your true business/research goal (MSE on perceptual tasks, CE on imbalanced data without weights); (b) your output has a non-standard distribution (counts, censored, mixture); (c) you have multiple objectives to combine; (d) your evaluation metric is non-standard (IoU, BLEU, ranking metrics); or (e) costs are asymmetric and you want the loss to encode that. Don't write a custom loss out of aesthetic preference — standard losses solve most problems and are battle-tested.

**Q2. Explain triplet loss and why we need hard negative mining.**

Triplet loss is $\max(0, d(a, p) - d(a, n) + m)$ over an (anchor, positive, negative) triple. It pushes anchor and positive closer than anchor and negative by at least margin $m$. For a random triplet on a sufficiently trained model, $d(a, n)$ is much larger than $d(a, p) + m$, so the loss is zero and there is no gradient. Most random triplets are like this — useless for training. Hard (or semi-hard) negative mining selects triplets where the margin condition is *almost* violated, providing meaningful gradient. This is essential for triplet-loss-trained models like FaceNet to learn anything.

**Q3. What's the purpose of the perceptual loss and where is it used?**

Perceptual loss compares activations of a pre-trained network (typically VGG) at intermediate layers, rather than comparing raw pixel values. The intuition: features at intermediate VGG layers encode semantic/textural content that aligns with human perception. Pixel MSE produces blurry outputs because the optimal per-pixel mean over plausible high-res images is itself blurry; perceptual loss penalizes feature-space differences instead, recovering sharpness. Used in super-resolution, style transfer, image translation, and as a component (alongside adversarial loss) in many image-generation models.

**Q4. In multi-task learning, what are the main challenges of combining losses?**

(1) **Scale mismatch:** different tasks produce naturally different loss magnitudes; without normalization, larger-magnitude tasks dominate gradients. (2) **Conflicting gradients:** updates that help task A may hurt task B (negative transfer). (3) **Hyperparameter explosion:** weights $\lambda_t$ for each task need tuning. (4) **Convergence rate differences:** some tasks converge faster, leaving others under-trained when training stops. Mitigations: uncertainty weighting (Kendall et al.), gradient-based balancing (GradNorm, PCGrad, MGDA), task-specific learning rates, careful loss normalization.

**Q5. What is the Dice loss and why is it preferred for medical image segmentation?**

Dice loss is $1 - \frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$. It is a soft, differentiable approximation of the Dice similarity coefficient (a set-overlap metric). It's preferred for medical segmentation because (a) it's *set-based* — penalizes failure-to-overlap directly rather than per-pixel error; (b) it handles severe class imbalance (most pixels are background) naturally — adding more background pixels with small predictions barely changes Dice, while it dominates pixel-wise CE; (c) it's directly aligned with the standard evaluation metric (Dice/IoU). Often combined with CE for stability.

#### Mathematical

**Q6. Derive the gradient of the Kendall et al. uncertainty-weighted multi-task loss with respect to the log-variances.**

Loss: $\mathcal{L}_{\text{total}} = \sum_t e^{-s_t} \mathcal{L}_t + \tfrac{1}{2}s_t$.

$\frac{\partial \mathcal{L}_{\text{total}}}{\partial s_t} = -e^{-s_t} \mathcal{L}_t + \tfrac{1}{2}$.

Setting to zero (instantaneous optimum): $s_t^* = \log(2 \mathcal{L}_t)$. So the optimal $\sigma_t^2 = 2 \mathcal{L}_t$ — the variance equals twice the (current value of) the task loss. If a task has high loss, the model raises its uncertainty, which downweights its loss in the total. The $\tfrac{1}{2}$ regularizer prevents the trivial solution $\sigma_t \to \infty$.

**Q7. Show that NT-Xent / InfoNCE has the form of categorical cross-entropy.**

$\ell_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(a, p)/\tau)}{\sum_{j \in \{p\} \cup \mathcal{N}} \exp(\text{sim}(a, j)/\tau)}$. Define logits $z_j = \text{sim}(a, j)/\tau$ over candidates $j \in \{p\} \cup \mathcal{N}$. Then $p_j = \exp(z_j) / \sum \exp(z_k) = $ softmax. With one-hot label putting all mass on the true positive $p$, the cross-entropy is $-\log p_p$ — exactly the InfoNCE expression.

So contrastive learning is "classification" where the classes are dynamically defined by the batch. The "vocabulary size" equals the number of negatives plus one.

**Q8. Why does soft Dice loss have gradient issues on empty masks, and how does the smoothing constant help?**

If $\sum y_i = 0$ (no positives in the ground truth) and $\sum p_i$ is small, both numerator and denominator of Dice are near zero. Without smoothing, we get $0/0$ — undefined and producing NaN gradients. Adding $\epsilon$ to both gives $\frac{\epsilon}{\sum p_i + \epsilon}$, well-defined. Practically, $\epsilon$ acts as a Bayesian prior: the loss for "predict zero" on an empty mask is roughly $1 - \epsilon/\epsilon \to 0$ when $p_i \to 0$. Common values: $\epsilon = 1$ (additive) or $\epsilon = 10^{-7}$ (numeric).

**Q9. Show that the soft-IoU and Dice losses can disagree on which prediction is better.**

For predictions $p_1, p_2$ on a fixed mask, the orderings $\text{IoU}(p_1) > \text{IoU}(p_2)$ and $\text{Dice}(p_1) > \text{Dice}(p_2)$ aren't always the same in the soft case. They are monotone transforms in the *binary* case ($\text{Dice} = 2 \text{IoU} / (\text{IoU} + 1)$), but for soft predictions two predictions can have the same hard-IoU and different soft-IoU and soft-Dice.

**Q10. Derive the gradient of the contrastive loss w.r.t. the anchor's embedding.**

$\ell = -\log \frac{\exp(a^\top p/\tau)}{Z}$, $Z = \sum_j \exp(a^\top j/\tau)$.

$\ell = -a^\top p / \tau + \log Z$.

$\partial \ell / \partial a = -p/\tau + (1/Z) \sum_j \exp(a^\top j/\tau)\,(j/\tau) = -p/\tau + \sum_j q_j (j/\tau)$

where $q_j = \exp(a^\top j/\tau)/Z$ is the softmax weight of candidate $j$.

So $\nabla_a \ell = (1/\tau)(\bar{j}_q - p)$ where $\bar{j}_q$ is the softmax-weighted mean of candidates.

Direction: if the (weighted average of) negatives is closer than the positive, gradient pushes anchor away from this average and toward the positive. ∎

#### Applied

**Q11. You're training a model that should predict bounding boxes and class labels. Walk me through designing the loss.**

(1) **Per-anchor classification loss**: cross-entropy over $K+1$ classes (including background). Use focal loss if dense, with severe imbalance. (2) **Per-anchor box regression**: smooth-L1 (Huber with $\delta = 1$) on parameterized box deltas $(t_x, t_y, t_w, t_h)$, where $t_w = \log(w/w_a)$ — log space to handle scale invariance. (3) **Mask the regression**: only compute on positive anchors (where ground-truth assigns a real box); $\mathbb{1}[y > 0]$ or equivalent. (4) **Combine**: $\mathcal{L}_{\text{cls}} + \lambda \mathcal{L}_{\text{box}}$, with $\lambda$ chosen so contributions are comparable in magnitude. Faster R-CNN uses $\lambda = 1$ with appropriate normalization (per anchor for class, per positive for box). (5) **Evaluation**: report mAP at standard IoU thresholds; that's what matters, not training loss.

**Q12. You're designing a recommender system that needs to rank items by relevance for each user. What loss?**

Several options depending on your data:

- **Pointwise**: BCE on click/no-click labels — simple but doesn't directly optimize ranking.
- **Pairwise**: BPR (Bayesian Personalized Ranking), $-\log \sigma(s_+ - s_-)$ on (positive, negative) item pairs — directly optimizes pairwise correctness.
- **Listwise**: ListMLE, ListNet, or LambdaRank — directly optimize listwise metrics like NDCG.
- **Sampled softmax**: when item catalog is huge (millions), use cross-entropy over the positive plus a sample of negatives, equivalent to InfoNCE.

In production: usually a combination — pairwise or sampled-softmax for training plus reranking for diversity/business rules. The choice depends on data structure (binary clicks vs explicit ratings) and computational constraints.

**Q13. Your multi-task model performs worse on each task than the single-task baselines. What might be wrong and what would you try?**

Negative transfer. Possible causes: (1) task gradients are conflicting; (2) loss scales are mismatched (one task dominates); (3) task data distributions differ in ways the shared representation can't reconcile; (4) capacity is insufficient and tasks are stealing from each other. Try: (a) **Loss balancing**: uncertainty weighting or GradNorm; (b) **Gradient surgery**: PCGrad to project conflicting gradients; (c) **Architecture changes**: task-specific heads with more capacity, or hard parameter sharing only at lower layers; (d) **Mixture of Experts**: each task uses a different combination of expert subnetworks; (e) **Train tasks separately first, then jointly** (curriculum); (f) **If nothing helps, train separate models** — sometimes that's the right answer.

**Q14. You're training a generative model and want sharper outputs than MSE produces. Walk me through the loss design.**

The standard recipe combines three losses:

1. **Reconstruction loss** ($\ell_1$ or MSE): pixel-level fidelity, ensures basic content correctness.
2. **Perceptual loss** ($\ell_2$ on VGG features): semantic / textural fidelity, prevents the "blurry mean" failure of pure pixel loss.
3. **Adversarial loss** (GAN discriminator): pushes outputs onto the data manifold, adds high-frequency realism.

Total: $\mathcal{L} = \lambda_1 \mathcal{L}_{\ell_1} + \lambda_2 \mathcal{L}_{\text{perc}} + \lambda_3 \mathcal{L}_{\text{adv}}$. Tune the weights such that all three contribute meaningfully — typically perceptual ~10× pixel and adversarial smaller and ramped up after a warm-up. This is roughly the Pix2Pix / SRGAN / ESRGAN recipe.

#### Debugging & Failure Modes

**Q15. You added a new term to your composite loss, but training collapsed. What would you check?**

(1) **Loss scale**: is the new term orders of magnitude bigger than existing terms, dominating gradients? Add a small weight initially. (2) **Sign**: did you accidentally flip the direction (maximizing instead of minimizing something)? (3) **Numerical stability**: log of zero, division by tiny number, NaN propagation. (4) **Gradient flow**: does the new term have meaningful gradients w.r.t. parameters, or is it constant / disconnected from the graph? (5) **Conflicts**: does the new term oppose existing terms in a way that destabilizes both? Add it in incrementally, monitor each term's value separately. Always log per-term losses, not just the total.

**Q16. Your contrastive loss is producing collapsed representations (all embeddings cluster at one point). What happened?**

The model found a trivial solution where all inputs map to the same point — making all positives close (good) but also making all negatives close (apparently bad, but the loss doesn't penalize it enough). This happens when (a) the temperature $\tau$ is too high, smoothing the softmax so all candidates look equal; (b) the batch is too small, providing too few negatives; (c) the model has stop-gradient asymmetry that allows collapse (can happen in BYOL/SimSiam-style methods if the predictor isn't asymmetric enough). Solutions: lower $\tau$, larger batches, normalization tricks (SimCLR uses L2 norm + projection head), or use methods designed to prevent collapse (Barlow Twins, VICReg).

**Q17. After hard negative mining, your triplet-loss model's loss spikes erratically. Why?**

Hardest-negative mining picks the *most* violating triplets — but these are often outliers, mislabeled examples, or genuinely ambiguous cases. Training on these aggressively destabilizes optimization. Use **semi-hard mining** instead: pick negatives that violate the margin but aren't the absolute worst. Or use a **soft variant** like Multi-Similarity Loss or the proxy-NCA family, which weights all negatives smoothly rather than picking just one.

**Q18. Your Dice loss won't drop below ~0.3 even with a high-capacity model. The CE loss is small. What gives?**

Two possible diagnoses: (1) Labels are inconsistent with the kind of segmentation the metric defines — e.g., your ground-truth boundaries are off by a few pixels relative to where the visual edges actually are, so even a perfect prediction gets penalized by Dice; (2) Dice is bounded below by structural mismatch — if the model is great at the bulk of the mask but mismatches at boundaries, Dice may stall at a moderate value. Also consider class imbalance within the foreground (multi-class Dice averaged across very rare classes can stall at high values). Inspect per-class Dice and the predicted vs ground-truth masks visually — often the issue is data-level, not loss-level.

#### Probing / Depth-of-Understanding

**Q19. The Lovász hinge / Lovász softmax are convex surrogates for IoU. Why prefer them over soft IoU / Dice?**

Soft IoU and Dice are *non-convex* in the model output; their gradients are generally not well-behaved theoretically. Lovász hinge is the *Lovász extension* of the discrete IoU loss — provably the tightest convex extension, with theoretical guarantees about being aligned with the actual IoU metric. In practice, on segmentation problems, Lovász softmax has been shown to give better IoU than soft Dice or CE in some benchmarks (Berman et al. 2018), particularly for small-object or unbalanced classes. Trade-off: more complex to implement, more expensive to compute.

**Q20. When your loss is non-differentiable (e.g., BLEU, F1 at a threshold, NDCG), what are your options?**

(1) **Surrogate loss**: design a smooth, differentiable approximation. Examples: soft-IoU for IoU, Lovász for IoU, sigmoid-based F1, NDCG-style listwise losses. (2) **REINFORCE / policy gradient**: sample predictions, compute the actual non-differentiable metric as reward, and use the score-function estimator $\nabla \log p \cdot R$. High variance, often needs baselines. (3) **Direct loss minimization** / **STE (straight-through estimator)**: pretend the non-differentiable function is the identity in backward pass. Crude but sometimes effective (used in quantization). (4) **Task-specific tricks**: Gumbel-softmax for discrete sampling, soft-DTW for sequence alignment. (5) **Two-phase training**: pretrain on a proxy loss, fine-tune with RL on the actual metric (common in NLG: pretrain LM with CE, fine-tune with REINFORCE on BLEU/ROUGE).

**Q21. Why is L1 regularization on weights a "loss" term, and how does it differ from L2?**

L1 ($\sum_j |w_j|$) and L2 ($\sum_j w_j^2$) are loss terms added to the data loss for **regularization** — encoding prior preferences for parameter values. They're "loss" in the sense they add to the objective being minimized. Differences: (1) **Sparsity**: L1 produces sparse solutions (many weights exactly zero) because its gradient $\text{sign}(w)$ doesn't shrink near zero; L2 shrinks proportionally, never fully zeroing. (2) **Bayesian interpretation**: L2 = Gaussian prior on weights; L1 = Laplacian prior. (3) **Computational**: L1 is non-differentiable at zero, requiring sub-gradient methods or proximal updates (ISTA, FISTA). L2 is smooth and adds a clean term to the Hessian.

**Q22. How would you design a custom loss for an ordinal regression problem (e.g., predicting age groups: child, teen, adult, senior)?**

Treating it as multi-class CE loses the ordering — predicting "senior" when truth is "child" is penalized the same as predicting "teen". Three reasonable approaches: (a) **Ordinal cross-entropy / cumulative link models**: predict $K-1$ binary "is-greater-than-class-$k$" probabilities with shared structure; (b) **Distance-aware loss**: weight CE by distance between predicted and true ordinal, e.g., $\ell = \sum_k |k - c| \cdot p_k$; (c) **Earth Mover / Wasserstein loss**: use 1D Wasserstein between predicted soft distribution and one-hot true label — this naturally encodes the ordering. (d) **Treat as regression** on the integer label with MSE/MAE — simple but loses class structure if class boundaries are non-uniform. Choose based on whether classes are evenly spaced, calibrated probabilities are needed, etc.

**Q23. In multi-task learning, when is it better to share parameters and when to keep them separate?**

Share when tasks have related underlying structure (similar features useful across tasks); typically share lower layers. Keep separate when tasks compete for representation capacity or have unrelated structure; typically use task-specific heads. The empirical rule: share early, branch late. If even shared lower layers hurt one task, the tasks may simply not be related enough — consider training separate models. Mixture-of-Experts architectures (MoE, Multi-gate MoE in production recommender systems at YouTube/Pinterest) provide a middle ground: shared expert pool, task-specific gating.

---

<a name="loss-surface"></a>
## 5. Loss Surface Analysis: Flat vs Sharp Minima and Generalization

### 5.1 Motivation & Intuition

So far we've discussed *what* loss to optimize. But once you've chosen a loss, you have a *function* $\mathcal{L}(\theta)$ over parameter space — a surface (typically very high-dimensional) that the optimizer traverses looking for a low point. The geometry of this surface profoundly affects:

- **Whether training succeeds** (existence of paths down to low loss).
- **How fast it succeeds** (curvature, conditioning).
- **Whether the trained model generalizes** (the central topic of this section).

The deepest empirical observation from a decade of deep learning research: **two models with the same training loss can have wildly different generalization performance, and the geometry of the loss surface around them is a key explanatory factor.**

The leading hypothesis: **flat minima generalize better than sharp minima**. A "flat" minimum is a wide, shallow basin where loss changes slowly as parameters move; a "sharp" minimum is a narrow trough where small parameter perturbations spike the loss.

The intuitive reason: training and test loss surfaces are similar but not identical (they differ by the generalization gap, which depends on data sampling). At a flat minimum on training, you remain near low loss on test even as the surface shifts slightly. At a sharp minimum, the same shift moves you up the cliff — high test loss.

A concrete picture: imagine two valleys side by side, one wide and gentle, one narrow and deep. Both have the same minimum height. Now shift the entire landscape by a small amount horizontally (representing the difference between training and test distributions). The wide valley's bottom is still at low altitude near where you were; the narrow valley's "bottom" now has you on a steep wall. Same training loss, very different test loss.

This intuition has led to:
- New optimizers designed to find flat minima (SAM, ASAM, GSAM).
- Theory connecting batch size, learning rate, and minimum sharpness.
- An understanding of why SGD generalizes better than full-batch gradient descent (it finds flatter minima, on average).
- Generalization bounds based on geometric properties of the minimum.

The flat-vs-sharp story is not a complete explanation — it has known counterexamples and ongoing controversies — but it is the most useful framework practitioners have for reasoning about loss-surface geometry and generalization.

### 5.2 Conceptual Foundations

**Key terms.**

- **Loss surface (loss landscape):** the function $\mathcal{L}(\theta)$ visualized over parameter space. For deep networks, this is high-dimensional (millions to billions of parameters), so we work with low-dimensional projections, summaries, and metrics.
- **Critical point:** a point where $\nabla \mathcal{L}(\theta) = 0$.
- **Minimum (local/global):** a critical point where the Hessian $H = \nabla^2 \mathcal{L}$ is positive semi-definite (local) or where the loss is the global infimum (global).
- **Saddle point:** a critical point where $H$ has both positive and negative eigenvalues. In high dimensions, saddles vastly outnumber local minima.
- **Sharpness:** measures how rapidly loss increases when parameters move away from the minimum. Quantified by Hessian eigenvalues, weight perturbation tests, or other proxies.
- **Flatness:** the opposite — small loss change for parameter perturbations. A flat minimum is a wide basin.
- **Mode connectivity:** the empirical finding that distinct minima of deep networks are often connected by paths of low loss in parameter space (Garipov et al., Draxler et al. 2018). Suggests the "many isolated minima" mental picture is wrong for deep networks.
- **Loss landscape "linear interpolation":** plot $\mathcal{L}(\alpha\theta_1 + (1-\alpha)\theta_2)$ for $\alpha \in [0, 1]$. Reveals barriers/connectivity between two solutions.

**Component interaction.**

The optimizer (SGD, Adam) does a random walk on $\mathcal{L}$, biased toward decreasing values by the gradient. Several factors interact:

1. **Gradient noise (from stochastic minibatches)** acts like temperature in physics — it lets the optimizer escape sharp minima (small basins) but stay in flat minima (large basins).
2. **Learning rate** scales the step size; larger learning rate effectively increases noise and escape velocity.
3. **Batch size** controls noise magnitude: smaller batch → more noise per step.
4. **Curvature** of the loss surface determines effective step size: in directions of high curvature, large steps overshoot; in flat directions, the optimizer drifts.
5. **Architecture** (depth, width, normalization, skip connections) shapes the loss surface itself. ResNets famously have much smoother loss surfaces than plain feedforward nets (Li et al. 2018).

**Underlying assumptions.**

- The flat-minima-generalize-better hypothesis assumes some smooth generalization gap connecting train and test losses, and assumes sharpness is a meaningful proxy for that gap. Both are empirically supported but not airtight.
- Hessian-based sharpness measures assume $\mathcal{L}$ is twice-differentiable at the minimum.
- "Local minimum" terminology assumes the optimum is isolated; for over-parameterized networks, minima often form connected manifolds (linear subspaces of zero-loss solutions).

**What breaks under violations.**

- **Reparameterization invariance:** Hessian eigenvalues change under simple parameter rescalings (e.g., scaling weights of one layer up and the next layer down to preserve outputs). So naive sharpness can be made arbitrarily large or small without changing the model's function. Dinh et al. 2017 showed this with explicit constructions, prompting the development of *invariant* sharpness measures (adaptive sharpness, ASAM).
- **Over-parameterized regime:** classical generalization theory (VC bounds, Rademacher complexity) breaks because models have far more parameters than data points yet generalize well. Loss-surface geometry is one of several proposed explanations.
- **Adam vs SGD:** Adam adaptively rescales gradients, effectively computing in a different geometry, which complicates direct application of SGD-based loss-surface analyses.

### 5.3 Mathematical Formulation

#### 5.3.1 Hessian-based sharpness

The Hessian at $\theta^*$ is $H = \nabla^2 \mathcal{L}(\theta^*) \in \mathbb{R}^{d \times d}$. Its eigenvalues $\{\lambda_i\}$ describe local curvature: large positive $\lambda_i$ = sharp in that direction, small $\lambda_i$ = flat.

Common scalar summaries of sharpness:

- **Largest eigenvalue $\lambda_{\max}(H)$:** dominant curvature; classical sharpness.
- **Trace $\text{tr}(H) = \sum_i \lambda_i$:** total curvature; relates to expected loss change under random Gaussian perturbation.
- **Spectral norm:** $\lambda_{\max}(H)$.
- **Effective rank / participation ratio:** how many directions have meaningful curvature; deep nets typically have a few large eigenvalues and a vast majority near zero.

For a quadratic local model $\mathcal{L}(\theta^* + \delta) \approx \mathcal{L}(\theta^*) + \tfrac{1}{2}\delta^\top H \delta$, the loss change under perturbation $\delta$ is $\tfrac{1}{2}\delta^\top H \delta$. For $\|\delta\|$ fixed, the worst-case direction maximizes $\lambda$, giving loss change $\tfrac{1}{2}\lambda_{\max} \|\delta\|^2$.

#### 5.3.2 Worst-case sharpness: Keskar et al. 2017

Keskar et al. introduced a sharpness metric that doesn't require computing the Hessian:

$$
\text{Sharpness}_\epsilon(\theta) = \frac{\max_{\|\delta\|_\infty \leq \epsilon (1+\|\theta\|_\infty)} \mathcal{L}(\theta + \delta) - \mathcal{L}(\theta)}{1 + \mathcal{L}(\theta)}
$$

This is the largest loss increase achievable by perturbing $\theta$ within a small box. The normalization tries to (partially) handle scale invariance issues. Computed via a few iterations of inner-loop projected gradient ascent.

#### 5.3.3 PAC-Bayes connection

PAC-Bayes generalization bounds give a theoretical foundation for "flat = generalizes" intuition. For a posterior $Q$ over parameters and prior $P$, with high probability over data:

$$
\mathbb{E}_{\theta \sim Q}[R(\theta)] \leq \mathbb{E}_{\theta \sim Q}[\hat{R}(\theta)] + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \log(1/\delta) + \log n}{2(n-1)}}
$$

If you set $Q = \mathcal{N}(\theta^*, \sigma^2 I)$ — a Gaussian centered at the trained model — then:
- The expected empirical risk $\mathbb{E}_Q[\hat{R}(\theta)]$ is small only if the loss is flat near $\theta^*$ (otherwise random perturbations spike it).
- The KL term $D_{\text{KL}}(Q \| P)$ is small when $\sigma$ is large (close to the prior).

So **flatness allows large $\sigma$ with small empirical risk, giving tighter generalization bounds**. This is the formal mechanism behind the flat-minima story.

#### 5.3.4 SGD as implicit regularizer toward flat minima

Continuous-time SGD can be modeled by a stochastic differential equation:

$$
d\theta = -\nabla \mathcal{L}(\theta) dt + \sqrt{\frac{2 \eta}{B} C(\theta)} dW
$$

where $\eta$ is learning rate, $B$ is batch size, $C(\theta)$ is the gradient covariance, and $W$ is Brownian motion. The effective "temperature" is $\eta / B$ — confirming the practical observation that **higher learning rate or smaller batch size = more noise = preference for flatter minima**.

The stationary distribution of this SDE concentrates on flat minima (lower expected loss under noise), providing a principled reason why SGD's noise *itself* selects flat solutions, not just convergence speed.

#### 5.3.5 Sharpness-Aware Minimization (SAM)

SAM (Foret et al. 2021) explicitly searches for flat minima by minimizing the worst-case loss in a neighborhood:

$$
\min_\theta \max_{\|\epsilon\| \leq \rho} \mathcal{L}(\theta + \epsilon)
$$

The inner max is approximated by a single ascent step:

$$
\hat{\epsilon}(\theta) = \rho \frac{\nabla \mathcal{L}(\theta)}{\|\nabla \mathcal{L}(\theta)\|}
$$

Then SAM updates $\theta$ via the gradient at the perturbed point: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta + \hat{\epsilon}(\theta))$.

The intuition: at sharp minima, the perturbation finds a high-loss point and the resulting update pushes you away from sharp regions. At flat minima, the perturbation barely changes the loss, so the update is effectively normal SGD.

Cost: ~2× the per-step cost of SGD (two forward-backward passes). Empirically improves test accuracy on many benchmarks (CIFAR, ImageNet, LM fine-tuning) by 0.5–2%. Variants: ASAM (adaptive, fixes scale invariance), GSAM, ESAM.

#### 5.3.6 Mode connectivity

Garipov et al. 2018 showed empirically that for any two minima $\theta_1, \theta_2$ found by independent SGD runs, there exists a *low-loss path* connecting them in parameter space — typically a quadratic Bezier curve. The straight-line interpolation $\alpha\theta_1 + (1-\alpha)\theta_2$ usually has a loss barrier, but a slightly curved path doesn't. Practical implication: the loss landscape isn't a collection of isolated craters but a connected low-loss manifold with bumps. Used in practice for efficient ensembling (snapshot ensembles, SWA — see below).

#### 5.3.7 Stochastic Weight Averaging (SWA)

Izmailov et al. 2018 noticed that averaging multiple weights along the SGD trajectory (after a long warm-up) gives better test accuracy than the final weight. Specifically:

$$
\theta_{\text{SWA}} = \frac{1}{K} \sum_{k=1}^K \theta_{(k)}
$$

where $\theta_{(k)}$ are weights at $K$ checkpoints during late training (often with a cyclic learning rate).

Why it works: the averaged point lies in a flatter region of the loss surface than any individual checkpoint. This is consistent with the flat-minima generalization story. Cheap, often improves accuracy 0.5–1%.

### 5.4 Worked Examples

#### 5.4.1 Sharp vs flat 1D illustration

Consider two loss functions of a single parameter $\theta$:

- Sharp: $\mathcal{L}_S(\theta) = 100 \theta^2$. At $\theta^* = 0$, second derivative is 200.
- Flat: $\mathcal{L}_F(\theta) = 1 \cdot \theta^2$. At $\theta^* = 0$, second derivative is 2.

Both have the *same* training loss at the minimum (zero). Now suppose the *test* loss is shifted: $\mathcal{L}_S^{\text{test}}(\theta) = 100(\theta - 0.1)^2$, similarly for flat.

At $\theta^* = 0$ (the training optimum):
- Sharp test loss: $100 \cdot 0.01 = 1$.
- Flat test loss: $1 \cdot 0.01 = 0.01$.

Same training loss, **100× difference** in test loss because the sharp minimum is much more sensitive to the train-test shift.

This 1D toy captures the intuition. In high dimensions the same effect operates in many directions simultaneously, with the *worst* (largest curvature) direction often dominating.

#### 5.4.2 Empirical sharpness experiment

Train two models on the same task with different batch sizes:

- Model A: batch size 128, lr 0.1.
- Model B: batch size 4096, lr 0.1.

Both achieve similar training loss. Measure the largest Hessian eigenvalue at the minimum (via power iteration).

Empirical findings (Keskar et al. 2017): Model A (small batch) has $\lambda_{\max} \approx 50$, Model B (large batch) has $\lambda_{\max} \approx 800$. Test accuracy: Model A 91%, Model B 88%.

The narrative: small batch = noisy gradients = "kicks" model out of sharp minima = flatter solution = better generalization. Large batch = smooth gradients = optimizer settles in the first sharp local minimum it finds.

(Caveat: with proper learning-rate tuning — specifically the linear-scaling rule $\eta \propto B$ — the gap can be largely closed. The pure batch-size effect isn't as clean as initially reported. But the relationship between noise and minimum flatness is real.)

#### 5.4.3 SAM walkthrough

A single SAM update at parameters $\theta$:

1. Compute gradient $g = \nabla \mathcal{L}(\theta)$.
2. Compute perturbation $\hat{\epsilon} = \rho \cdot g / \|g\|$ (unit-norm rescaled to radius $\rho$).
3. Compute "perturbed" gradient $g' = \nabla \mathcal{L}(\theta + \hat{\epsilon})$.
4. Update $\theta \leftarrow \theta - \eta g'$.

Numeric example with 1D quadratic $\mathcal{L}(\theta) = a\theta^2/2$:

- $g = a\theta$.
- $\hat{\epsilon} = \rho \cdot \text{sign}(\theta)$.
- $g' = a(\theta + \rho \cdot \text{sign}(\theta))$.
- Update: $\theta \leftarrow \theta - \eta a (\theta + \rho \cdot \text{sign}(\theta)) = (1 - \eta a)\theta - \eta a \rho \cdot \text{sign}(\theta)$.

Compared to SGD's $\theta \leftarrow (1 - \eta a)\theta$, SAM has an extra term that pushes $\theta$ slightly past zero (toward the negative side if $\theta > 0$), potentially overshooting. In the quadratic case, this isn't useful (single sharp minimum); in non-convex settings, it helps escape sharp minima toward flatter ones.

### 5.5 Relevance to ML Practice

**Where loss-surface analysis matters in practice:**

- **Optimizer choice and tuning.** SGD with momentum tends to find flatter minima than Adam, partly explaining why SGD-trained models often generalize better on image classification (despite Adam's faster training-loss reduction). When deploying models, "fastest training loss" rarely correlates with "best test performance"; the geometry matters.
- **Learning rate schedules.** Warm-up + cosine decay isn't just an optimization trick; it relates to noise levels and which minima you settle into. Large initial learning rate explores broadly; small final learning rate refines within a basin.
- **Batch size selection.** Batch sizes interact with noise levels, which affect minimum flatness. The "linear scaling rule" ($\eta \propto B$) and "square root rule" ($\eta \propto \sqrt{B}$) come from analysis of the SDE limit of SGD.
- **Model architecture.** Skip connections (ResNets, Transformers), normalization (batch/layer norm), and width affect the loss surface geometry. Li et al. 2018's loss-landscape visualizations are striking: ResNets have much smoother surfaces than plain feedforward nets of similar depth, which is one reason they train more easily.
- **Regularization design.** Weight decay, dropout, label smoothing all interact with minimum flatness. Weight decay in particular has been shown to bias toward flatter minima.
- **Sharpness-aware methods.** SAM and variants are now production-grade for challenging fine-tuning tasks (LLM fine-tuning, robust ImageNet training).
- **Ensembling.** Mode connectivity informs efficient ensembling: snapshot ensembles, SWA, fast geometric ensembling. Cheaper than independent training, often better.

**When the flat-minima framework helps and when it doesn't.**

Helps with: explaining batch size effects, motivating SAM-like methods, designing ensembling schemes, understanding why some optimizers generalize better.

Doesn't help with: precise quantitative predictions of test loss (sharpness measures are noisy proxies); explaining all generalization phenomena (some networks generalize despite sharp minima, e.g., recent MLP-Mixer style models); guiding design of new losses (loss-surface analysis is about a fixed loss).

**Trade-offs.**

- *Flat minima vs training speed.* Achieving flat minima sometimes requires more training (longer SGD trajectories, SAM's 2× cost). On a fixed compute budget, the tradeoff isn't always favorable.
- *Sharpness measures vs scale invariance.* Naive sharpness can be gamed by reparameterization (Dinh et al. 2017). Adaptive measures (ASAM) help but add complexity.
- *Empirical vs theoretical understanding.* Many flat-minima results are empirical correlations; the theory (PAC-Bayes, etc.) is suggestive but not airtight.

### 5.6 Common Pitfalls & Misconceptions

1. **"Lower training loss = better model."** Famously false — overfitting is exactly the opposite. Two models with the same training loss can have very different test loss; the geometry of the minimum is part of why.

2. **"Flat means small Hessian eigenvalues — period."** Hessian eigenvalues are scale-dependent. Reparameterizing the network (e.g., scaling layer 1 weights by 10 and layer 2 weights by 0.1) can change Hessian magnitudes without changing the function. Use scale-invariant measures (ASAM, normalized sharpness) for serious analysis.

3. **"SGD finds flat minima because of its gradient noise."** Partly true, but it's also the *direction* of gradient noise (along principal data directions) and the *magnitude* (set by lr/batch ratio) that matter. Just adding random noise to a different optimizer doesn't reproduce SGD's effect.

4. **"Adam doesn't generalize as well because it overfits training loss faster."** Empirically Adam often has slightly worse test accuracy than SGD with momentum on image classification — but on language modeling and many other tasks Adam dominates. The flat-minima argument isn't a universal answer.

5. **"Mode connectivity means there's only one minimum."** No — there are many local minima, but they're connected by low-loss paths. The picture isn't "one big basin"; it's "a network of low-loss valleys with passes between them, not isolated craters."

6. **"SAM always helps."** SAM helps on many benchmarks but is not a free lunch — it doubles training time per step, adds a hyperparameter $\rho$, and can hurt when $\rho$ is mis-tuned. Standard practice: try SAM after a working SGD baseline; treat $\rho$ as a hyperparameter.

7. **"My visualization of the loss landscape (2D plot along random directions) shows a smooth bowl — so my optimum is flat."** Random 2D projections of high-dimensional surfaces can be misleading. The directions you didn't project onto may be sharp. Use Hessian-based analysis or many random projections, or directions chosen along principal Hessian eigenvectors.

8. **"More parameters = more overfitting."** Modern over-parameterized networks routinely have $10^9$ parameters and $10^7$ training examples and generalize fine. The classical bias-variance tradeoff doesn't directly apply; the loss surface geometry, implicit regularization of SGD, and architectural inductive biases together explain the modern phenomenon. (See "double descent" literature.)

9. **"Larger batch always trains faster."** True for time-per-step (more parallelism), but per-step progress can be worse if gradient noise was helping you avoid bad minima. Linear scaling of learning rate partially compensates.

10. **"My loss is monotonically decreasing on training, so optimization is fine."** It might be — but you might also be moving toward a sharp minimum that won't generalize. Always monitor validation loss and (when you can) sharpness or related diagnostics.

### 5.7 Interview Questions: Loss Surface Analysis

#### Foundational

**Q1. What's the difference between a sharp and a flat minimum, and why does it matter?**

A sharp minimum is a narrow region where small parameter perturbations cause large loss increases (high local curvature). A flat minimum is a wide region where loss changes slowly (low curvature). It matters because the test loss surface is similar to but not identical to the training loss surface — the difference creates an effective shift. At a sharp minimum on training, that shift moves you up the steep walls, giving high test loss; at a flat minimum, you stay near low loss. Empirically, flatter minima generalize better, and many modern training techniques (SAM, large learning rates, small batch sizes, SWA) implicitly or explicitly bias the optimizer toward flat minima.

**Q2. Why does SGD generalize better than full-batch gradient descent on many tasks?**

The leading explanation: SGD's gradient noise (from minibatch sampling) acts as a temperature, allowing the optimizer to escape sharp minima while remaining trapped in flat ones. Full-batch GD has no such noise, so it converges to the first minimum it reaches — often a sharp one. The effective temperature scales as $\eta/B$ (learning rate over batch size), so larger batches (with proportionally adjusted learning rate per the linear scaling rule) can partially recover the noise effect. This isn't the only explanation — implicit regularization in the optimization trajectory, mode connectivity, and architectural effects all play roles.

**Q3. What is Sharpness-Aware Minimization (SAM)?**

SAM minimizes the worst-case loss in a small neighborhood around the parameters: $\min_\theta \max_{\|\epsilon\|\leq\rho} \mathcal{L}(\theta + \epsilon)$. Approximated by a single gradient ascent step inside the ball to find $\hat\epsilon$, then taking the gradient at the perturbed point and updating $\theta$ accordingly. This biases optimization toward flat minima — at flat points, the worst-case loss is close to the loss itself; at sharp points, it's much higher. Empirically improves generalization across many tasks at the cost of ~2× training time per step.

**Q4. What's "mode connectivity" and why is it surprising?**

Garipov et al. and Draxler et al. (2018) showed that distinct minima of deep networks (found by independent SGD runs) can be connected by *low-loss paths* in parameter space — typically simple curves like quadratic Beziers. Naive linear interpolation between two minima usually crosses a high-loss barrier, but slightly curved paths don't. This is surprising because the classical mental picture of a non-convex loss surface is "many isolated minima separated by ridges"; mode connectivity suggests deep network loss surfaces are better described as "a connected low-loss manifold with some bumpy regions." It has practical use: efficient ensembling along the connecting paths.

**Q5. What is Stochastic Weight Averaging (SWA) and why does it work?**

SWA averages model weights from multiple checkpoints in the late stage of training (often with a cyclic learning rate). The averaged weights typically lie in a flatter region of the loss landscape than any single checkpoint — partly because averaging cancels out the random fluctuations of SGD around its mean trajectory. Test accuracy improves by 0.5–1% on many benchmarks at essentially no cost (just save and average checkpoints). It's consistent with the flat-minima generalization story and is one of the cheapest "free improvements" available in modern training.

#### Mathematical

**Q6. Show that the expected loss under random Gaussian parameter perturbation depends on the trace of the Hessian.**

Let $\theta^*$ be a local minimum, $H$ the Hessian. For small $\delta$, $\mathcal{L}(\theta^* + \delta) \approx \mathcal{L}(\theta^*) + \tfrac{1}{2}\delta^\top H \delta$. With $\delta \sim \mathcal{N}(0, \sigma^2 I)$:

$$
\mathbb{E}[\mathcal{L}(\theta^* + \delta)] \approx \mathcal{L}(\theta^*) + \tfrac{1}{2}\sigma^2 \mathbb{E}[\delta^\top H \delta / \sigma^2]\sigma^2 = \mathcal{L}(\theta^*) + \tfrac{1}{2}\sigma^2 \text{tr}(H)
$$

(Using $\mathbb{E}[\delta^\top H \delta] = \text{tr}(H \cdot \text{Cov}(\delta)) = \sigma^2 \text{tr}(H)$.) So expected loss under Gaussian perturbation grows linearly in $\sigma^2$ with slope $\tfrac{1}{2}\text{tr}(H)$ — flatter minima (lower trace) are more robust to random perturbations.

**Q7. Sketch the PAC-Bayes argument connecting flatness to generalization.**

PAC-Bayes bound: for any prior $P$ and any data-dependent posterior $Q$, with high probability:

$$
\mathbb{E}_{\theta \sim Q}[R(\theta)] \leq \mathbb{E}_{\theta \sim Q}[\hat{R}(\theta)] + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \text{const}}{n}}
$$

Choose $Q = \mathcal{N}(\theta^*, \sigma^2 I)$. The RHS is small if both terms are small:

- $\mathbb{E}_Q[\hat R(\theta)]$ small: Gaussian perturbations of $\theta^*$ keep training risk small, i.e., $\theta^*$ is in a flat region.
- $D_{\text{KL}}(Q \| P)$ small: $\sigma$ is large enough to keep $Q$ close to a broad prior.

Both can be satisfied simultaneously only if the loss is flat near $\theta^*$. So flatter minima yield tighter generalization bounds. ∎ (Schematic — actual PAC-Bayes results require careful prior choice and fine print.)

**Q8. Explain the Dinh et al. 2017 result: sharpness can be made arbitrarily large by reparameterization.**

For a ReLU network, multiplying layer $\ell$'s weights by $c$ and layer $\ell+1$'s weights by $1/c$ leaves the function $f_\theta$ unchanged (and thus the loss). But the Hessian eigenvalues w.r.t. the parameters change because the parameter scale changes. Concretely, gradients in layer $\ell$ scale by $1/c$ and in layer $\ell+1$ by $c$, so Hessian elements rescale accordingly. By choosing $c$ large or small, you can make any Hessian eigenvalue arbitrarily large or small — without changing the model's predictions or generalization. This shows raw Hessian-based sharpness is not a *function-level* property but a *parameterization-level* one, motivating scale-invariant alternatives like ASAM (which uses adaptive perturbation radii proportional to $|w|$).

**Q9. Derive the optimal SAM perturbation $\hat\epsilon$ from the inner maximization.**

Inner problem: $\max_{\|\epsilon\|_2 \leq \rho} \mathcal{L}(\theta + \epsilon)$. First-order Taylor: $\mathcal{L}(\theta + \epsilon) \approx \mathcal{L}(\theta) + \epsilon^\top g$ where $g = \nabla \mathcal{L}(\theta)$.

So we maximize $\epsilon^\top g$ subject to $\|\epsilon\| \leq \rho$. Cauchy-Schwarz gives $\epsilon^\top g \leq \|\epsilon\| \|g\| \leq \rho \|g\|$, achieved at $\epsilon^* = \rho g / \|g\|$.

Hence $\hat\epsilon = \rho g / \|g\|$ — the SAM perturbation is in the direction of the gradient, scaled to the constraint boundary.

**Q10. What does the SDE limit of SGD tell you about implicit regularization?**

Under continuous-time approximation, SGD becomes:

$$
d\theta = -\nabla \mathcal{L}(\theta)\,dt + \sqrt{\frac{2 \eta}{B} C(\theta)}\,dW
$$

where $C(\theta)$ is the gradient covariance. The stationary distribution is approximately $p(\theta) \propto \exp(-\mathcal{L}(\theta)/(\eta/B))$ (with corrections for non-isotropic $C$). So SGD samples from a Gibbs-like distribution at "temperature" $\eta/B$, concentrated on low-loss regions but penalizing sharp ones (because high-curvature areas have small basin volume). This mathematically reproduces the flat-minima preference.

#### Applied

**Q11. You're training an LLM and notice your validation loss is much higher than training loss after 1 epoch. Could this be related to loss-surface geometry?**

Possibly, but more likely it's an overfitting indicator regardless of geometry. With LLM pre-training on huge corpora, after 1 epoch you usually haven't overfit to specific examples but might have memorized corpus-specific statistics that don't transfer. Possible interventions: (a) more data; (b) stronger regularization (weight decay, dropout); (c) smaller learning rate or longer warm-up — both bias toward flatter minima; (d) smaller batch size; (e) try SAM if compute budget permits. But first verify it's not a data leakage issue (validation set contamination), which is a more common cause of large train-val gaps.

**Q12. You want to use SAM for fine-tuning a model. How do you choose $\rho$?**

$\rho$ controls the perturbation radius; too small and SAM behaves like vanilla SGD, too large and training destabilizes. Typical starting values: $\rho = 0.05$ for image classification (CNNs), $\rho = 0.005$ to $0.01$ for fine-tuning (where models start from a flatter region). Sweep $\rho \in \{0.001, 0.01, 0.05, 0.1\}$ on a small validation set. ASAM (adaptive SAM) is less sensitive because it scales $\rho$ per parameter by $|w|$. Worth comparing: SAM, ASAM, and plain SGD with similar compute budget (since SAM costs 2× per step).

**Q13. Your model achieves great training loss but poor test loss. You suspect a sharp minimum. What can you try?**

Several techniques bias toward flatter minima: (a) **Smaller batch size** (more gradient noise); (b) **Higher learning rate** with cosine decay; (c) **SAM or ASAM**; (d) **SWA** (stochastic weight averaging) — collect checkpoints late in training, average; (e) **Stronger L2 / weight decay**; (f) **Label smoothing** (softens targets, reduces logit magnitudes); (g) **Mixup / CutMix** data augmentation (effectively averages targets, smooths loss surface); (h) **Longer training with lower learning rate** — refines within a basin rather than locking into the first one. Try them in order of cheap-to-expensive, monitor validation loss after each.

**Q14. You're choosing between batch size 64 and batch size 4096 for a fixed compute budget. What do you consider?**

Tradeoffs: (a) **Per-step time**: batch 4096 is much faster per step (more parallelism); fewer steps in your budget. (b) **Generalization**: batch 64 typically generalizes better due to flatter minima — but with proper learning-rate scaling (linear or square-root) and warmup, the gap shrinks. (c) **Statistical efficiency**: very small batches have high gradient variance, slowing convergence. (d) **Hardware utilization**: very small batches under-utilize GPUs. Practical: pick the largest batch that fits + still uses small enough effective learning rate to maintain noise. For most deep learning tasks today, batch sizes in the 256–8192 range with proper LR scaling work well; below 64 is usually wasteful, above 65k can hit "too smooth" generalization issues.

#### Debugging & Failure Modes

**Q15. After switching from SGD to AdamW, your model's training loss drops faster but validation loss is worse. Why?**

Two leading explanations: (1) Adam-family optimizers tend to find sharper minima than SGD with momentum because the per-coordinate learning rates effectively flatten the loss surface in the optimizer's view — masking sharpness that hurts generalization. (2) The default learning rates for AdamW vs SGD are very different, and re-tuning is essential. Mitigations: (a) try SGD with momentum + cosine LR + warm-up as baseline; (b) tune AdamW with smaller learning rate and weight decay; (c) try SAM on top of AdamW; (d) consider using Adam's training speed for early epochs and switching to SGD for later refinement. Also worth verifying: is the gap on a held-out test set or just validation? Could be data leak.

**Q16. You measured the largest Hessian eigenvalue of two trained models and got 50 vs 5000. The "flatter" model (50) has worse test accuracy. What's going on?**

Several possibilities: (a) **Reparameterization artifact** — Hessian eigenvalues aren't function-invariant; the "flatter" model might just have differently scaled weights. Check with adaptive sharpness (ASAM-style). (b) **Sample size of Hessian estimate** — Hessian eigenvalues of large neural networks are often estimated with stochastic approximations (Lanczos with subsampling); estimates can be noisy. Re-estimate with larger samples. (c) **Sharp minima can still generalize** — the flat-vs-sharp story is statistical, not deterministic. Some sharp minima generalize fine; some flat minima don't. The correlation is meaningful but not airtight. (d) **Model architectures differ** — comparing across architectures is fraught; sharpness baselines differ.

**Q17. You're training with SAM and the loss is oscillating wildly. What might be wrong?**

Likely $\rho$ is too large — the perturbation is so big that the perturbed gradient has essentially nothing to do with the unperturbed one, causing erratic updates. Try halving $\rho$. Other causes: (a) the per-batch perturbation is being computed with a different normalization than the per-batch gradient; (b) numerical instability in the perturbation step (especially in fp16); (c) SAM interacting badly with batch normalization (the perturbation changes BN statistics, leading to mismatched train/eval behavior). Switching to ASAM often helps with (b) and (c).

#### Probing / Depth-of-Understanding

**Q18. The flat-minima hypothesis says flat minima generalize better. What are the main critiques and what does the literature say?**

Main critiques: (1) **Reparameterization invariance** (Dinh et al. 2017): naive sharpness measures are not function-level invariants. Counter: use adaptive/invariant sharpness measures. (2) **Counterexamples**: some sharp minima generalize well, some flat ones don't (Andriushchenko et al. and others have constructed examples). (3) **Sharpness is correlated with but doesn't cause better generalization** — both might be effects of better optimization choices. (4) **Doesn't explain everything** — generalization in deep learning involves implicit regularization, data structure, architecture, etc.; sharpness is one factor among many. Despite critiques, the hypothesis has empirical traction: SAM and related methods consistently improve test accuracy across many domains, suggesting the flat-minima intuition captures something real even if not the full story.

**Q19. How do skip connections (ResNet) affect the loss landscape?**

Li et al. 2018 visualized the loss surface of plain feedforward networks vs ResNets and found that ResNets have *much smoother* loss landscapes — fewer barriers, fewer local minima, broader basins. The mechanism: skip connections preserve identity at initialization, so even very deep networks compute approximately reasonable functions early in training; gradients flow through skip connections without vanishing. This makes optimization easier (smoother landscape) and indirectly promotes finding flatter minima (broader basins available). Without skip connections, very deep networks have a "shattered gradient" problem and pathological loss surfaces. Skip connections are a major reason why ResNets enable training networks with hundreds of layers.

**Q20. Why does weight decay help with generalization, and how does it relate to loss-surface geometry?**

Weight decay $\lambda \|\theta\|^2$ added to the loss has multiple effects: (1) **Classical regularization**: shrinks weights toward zero, reducing model capacity; (2) **Bayesian interpretation**: Gaussian prior on weights. (3) **Implicit flatness bias**: by penalizing large parameter magnitudes, weight decay biases the optimizer toward minima where the loss can be reached with smaller weights — these tend to be flatter in many architectures. The interplay with normalization is subtle: in networks with batch norm, weight decay's effect on the *function* is somewhat decoupled from its effect on parameter magnitudes (because BN rescales activations). For BN-equipped networks, weight decay primarily acts as an effective learning-rate regularizer rather than as a function-level prior.

**Q21. What is "double descent" and how does it relate to loss-surface geometry?**

Double descent (Belkin et al. 2019, Nakkiran et al. 2020): as model capacity increases, test error first goes down (classical regime), then up (overfitting at the interpolation threshold), then down again (modern over-parameterized regime). The second descent is surprising classically. One explanation involves loss-surface geometry: at the interpolation threshold, the model just barely fits the training data, requiring a "uniquely tight" solution that's likely sharp and brittle. With even more capacity, many interpolating solutions exist, and SGD's implicit bias picks flatter ones. So the second descent is partly explained by the abundance of flat minima in over-parameterized regimes. (Other explanations involve the bias-variance tradeoff in random feature models.)

**Q22. If you could measure only one quantity to predict generalization, what would you measure and why?**

There's no perfect answer — but candidates include: (a) **Validation loss** (the gold standard, requires held-out data); (b) **Adaptive sharpness** (ASAM-style); (c) **Margin** (for classification, distance from decision boundary); (d) **Norm-based bounds** (Frobenius norm of weights times margin terms, as in Bartlett-style bounds); (e) **Spectral norm of the network Jacobian**. None is reliable across all settings — Jiang et al. 2020 (Fantastic Generalization Measures) studied dozens of measures across many models and found weak universal predictors. The most robust answer for a practitioner is "use a held-out validation set" — geometric measures are useful diagnostics but not replacements.

**Q23. What does mode connectivity imply for ensembling and Bayesian methods?**

Mode connectivity opens cheap ensembling: instead of training $K$ models from scratch, train one to convergence, then sample additional models along the connecting low-loss curve to other found minima. This gives a diverse ensemble at a fraction of the cost. Practical methods: SWA-Gaussian (Gaussian approximation along the trajectory), Fast Geometric Ensembling (FGE), Snapshot Ensembling. For Bayesian methods, mode connectivity suggests posterior distributions over neural network parameters are concentrated on the connected low-loss manifold rather than on isolated points — informing the design of approximate posterior families (e.g., subspace inference, where the posterior is constrained to a learned low-dimensional subspace).

**Q24. How do learning rate schedules interact with loss-surface geometry?**

A high initial learning rate explores broadly — large effective noise lets SGD escape sharp minima. As LR decays (cosine, step, exponential), the optimizer settles into a basin and refines. Warmup further helps: starting too aggressively can knock the model into a bad region or cause instability; gradual ramp-up provides smooth entry. Late-training low LR with checkpoint averaging (SWA) leverages this — the late trajectory wanders within a flat basin, and averaging cancels the wander to give the basin's center. The "1cycle" policy (Smith) uses a triangular LR schedule explicitly designed to find good basins quickly. So LR schedules aren't just optimization tricks — they're loss-surface navigation strategies.

---

<a name="cross-topic-qa"></a>
## 6. Cross-Topic Interview Question Bank

This section contains questions that span multiple loss-function topics — the kind of integrative reasoning interviewers use to test whether your understanding goes beyond memorized definitions. Answer each by drawing connections across the families covered.

#### Foundational synthesis

**X1. Walk me through how you'd choose a loss function for a new ML problem.**

Step 1: **What are you predicting?**
- Real number(s) → regression family (MSE, MAE, Huber, Log-Cosh, quantile).
- Category from a fixed set → classification (CE, hinge).
- Probability distribution → NLL of a chosen family.
- Set / mask → IoU/Dice (or pixel-wise CE).
- Ranking → pairwise/listwise ranking loss.
- Embedding for retrieval → contrastive or triplet.

Step 2: **What are the data properties?**
- Outliers? → robust losses (MAE, Huber, log-cosh; Wasserstein over KL).
- Class imbalance? → weighted CE, focal loss, Dice for segmentation.
- Asymmetric costs? → cost-weighted CE, quantile loss.
- Noisy labels? → bounded losses (MAE on softmax, generalized CE).

Step 3: **What downstream metric do you actually care about?**
- Calibrated probabilities? → CE, NLL (avoid hinge, MSE-on-classification).
- Accuracy? → CE.
- IoU/Dice? → consider Lovász or soft-Dice surrogate.
- BLEU/ROUGE/business metric? → may need RL on top of pretraining loss.

Step 4: **Combine if needed.** Multi-task → weighted sum (uncertainty-weighted ideal). Generative → reconstruction + perceptual + adversarial.

Step 5: **Validate the choice empirically** — train, evaluate on the actual target metric, and adjust. Loss design is iterative.

**X2. Which standard losses are NLL of a specific distribution? Tabulate.**

| Loss | Distribution | Domain |
|------|--------------|--------|
| MSE | Gaussian (fixed variance) | Real-valued regression |
| MAE | Laplace | Real-valued regression |
| Heteroscedastic Gaussian NLL | Gaussian (predicted variance) | Probabilistic regression |
| BCE | Bernoulli | Binary classification |
| Categorical CE | Categorical | Multiclass classification |
| Poisson NLL | Poisson | Count data |
| Negative-binomial NLL | Negative binomial | Over-dispersed counts |
| MDN NLL | Mixture of Gaussians | Multi-modal regression |
| Tweedie deviance | Tweedie | Insurance / mixed zero-and-positive |
| Cox partial likelihood | Cox model | Survival / time-to-event |

This is the unifying view: most losses are MLE for some distributional model. Choosing the loss is choosing the model.

**X3. Compare the trade-offs of cross-entropy, hinge loss, and MSE for binary classification.**

- **Cross-entropy**: probabilistic, calibrated outputs, gradient $p - y$ is clean and never saturates, unbounded penalty for confident wrong predictions (sensitive to label noise). Default choice for deep classifiers.
- **Hinge**: margin-based, no probabilistic interpretation, gradient is bounded ($\pm 1$ or $0$), zero loss for confidently correct examples (sparse support). Better for label-noise robustness, worse for calibration. Used in SVMs and some metric-learning settings.
- **MSE**: quadratic in residual, gradient with sigmoid output saturates ($p(1-p) \to 0$ for confident predictions), no probabilistic interpretation. Generally a poor choice for classification because of the saturation problem during early training.

For modern deep classification, CE is the default; hinge appears in margin-based / metric-learning settings; MSE is essentially never used for classification in production.

**X4. KL divergence, cross-entropy, and NLL are often conflated. When does it actually matter to distinguish them?**

For *optimization* over a model $p_\theta$ given fixed data, all three are equivalent (they differ only by constants in $\theta$). It matters when:
- **Comparing values across runs/models with different label distributions** — CE values are sensitive to the entropy of the label distribution; KL isn't (subtracts that off).
- **Theoretical analysis** — KL is the "right" distance-like quantity in information theory; CE is the right loss when you have data.
- **Reverse vs forward KL** — when you choose to minimize $D_{\text{KL}}(p_\theta \| q)$ vs $D_{\text{KL}}(q \| p_\theta)$, you get different solutions. The standard MLE setting always uses forward KL = NLL.
- **Constants in your loss** — if combining with other losses (e.g., $\beta$-VAE balances reconstruction NLL and a KL term), the absolute scale of the KL matters.

#### Integrative reasoning

**X5. You're training a model that predicts both a mean and variance for a target (heteroscedastic regression). What loss do you use, and how does this compare to MSE plus a separate uncertainty estimator?**

Use Gaussian NLL: $\ell = \tfrac{1}{2}\log\sigma^2(x) + (y - \mu(x))^2 / (2\sigma^2(x))$. This jointly fits mean and variance via maximum likelihood. The training will (a) learn $\mu(x)$ to match $y$ on average, and (b) learn $\sigma^2(x)$ to match the *residual variance* — high in noisy regions of input space, low where the model can be confident.

Compared to MSE + separate uncertainty estimator: (a) the joint NLL is *learned end-to-end*, automatically calibrating; (b) separate uncertainty estimators (e.g., temperature scaling, MC dropout) come after training and may not align with actual residual structure. Trade-off: NLL training can be unstable if $\sigma$ is allowed to go very small (overconfident) — common fix is to clip $\log\sigma$ to a reasonable range, or use a Beta or Student's-t likelihood that's more robust.

**X6. Compare focal loss with class-weighted cross-entropy. When does one beat the other?**

Class-weighted CE: $-w_c \log p_c$ multiplies each example's loss by a class-specific constant. Effectively over-samples minority classes.

Focal loss: $-(1 - p_t)^\gamma \log p_t$ multiplies by an *example-specific*, *prediction-dependent* factor that down-weights examples the model already gets right.

When **class-weighted CE is better**: imbalance is the only issue; all examples in each class are roughly equally hard; you have enough examples per class that the rare class isn't intrinsically harder.

When **focal loss is better**: severe imbalance with many easy negatives (e.g., dense object detection, where 99.9% of anchors are background and most are easy); the *hardness* of examples varies dramatically within each class; class re-weighting alone isn't sufficient because even minority class has easy examples consuming gradient signal.

Often combined (focal loss has an $\alpha$ parameter for class re-weighting).

**X7. You're designing a generative model. Walk me through the loss design tradeoffs of (a) MSE, (b) NLL of a parametric distribution, (c) GAN loss (JS-divergence), (d) Wasserstein loss.**

(a) **MSE**: easy to optimize, but produces blurry outputs (the optimal pixel-wise prediction is the *mean* of plausible images, which is blurry). Models pixel distribution as Gaussian — wrong for images.

(b) **NLL of explicit distribution** (PixelCNN, VAEs, normalizing flows): proper probabilistic model; can compute likelihoods of new data; samples can lack sharpness because the model averages over uncertain modes. Good for density estimation; less optimal for sample quality alone.

(c) **GAN (JS divergence)**: doesn't define an explicit likelihood, samples can be very sharp, but training is unstable (mode collapse, vanishing gradients when distributions don't overlap). Mode collapse is partly explained by JS divergence behavior.

(d) **Wasserstein**: provides smooth gradients even when generator and data distributions don't overlap (the canonical advantage); requires Lipschitz constraint enforcement (weight clipping or gradient penalty); training more stable than vanilla GAN; sample quality often comparable but not always better than modern non-Wasserstein GANs.

In practice, modern image generation models often combine multiple losses: reconstruction (pixel + perceptual) for fidelity, adversarial for sharpness, and sometimes regularization (R1, R2, gradient penalty) for stability.

**X8. Connect: choice of loss → optimal output of model → calibration → downstream decision-making.**

Loss determines what the optimizer pushes the model toward. Specifically:

- MSE → conditional mean. For classification with one-hot, this is $P(Y=1|x)$.
- MAE → conditional median. For binary classification, this snaps to 0 or 1 unless $P=0.5$.
- Cross-entropy → calibrated $P(Y|x)$.
- Hinge → uncalibrated score; sign matches Bayes-optimal but magnitude is uninformative.
- Quantile $\tau$ → conditional $\tau$-quantile.

For downstream decision-making (deciding to act or not based on predicted probability), you need the model's output to match the cost-relevant aspect of the true distribution. If your decision threshold is "act if probability > 0.7", you need calibrated probabilities, which essentially restricts you to CE-style losses (or post-hoc calibration).

If your decision is asymmetric (cost of false positive ≠ cost of false negative), the cost-optimal threshold isn't 0.5 but $c_{FP}/(c_{FP} + c_{FN})$. You can either (a) train with standard CE and tune the threshold, or (b) train with cost-weighted CE / quantile loss with $\tau$ matching the cost ratio. Both are valid; threshold tuning is easier to iterate.

**X9. How do regularization terms (L1, L2) interact with the choice of data loss?**

L1 / L2 regularization adds $\lambda \|w\|_p$ to the data loss, biasing the model toward small weights. The interaction:

- **MSE + L2 = ridge regression**: closed-form solution, nicely conditioned.
- **MSE + L1 = LASSO**: produces sparse solutions, requires proximal methods.
- **CE + L2 = standard "regularized logistic regression"**: standard practice, doesn't change the loss's qualitative behavior.
- **CE + L1**: encourages sparse weights — used for feature selection.

The regularization doesn't change the loss's *probabilistic interpretation* directly, but the combined objective is the *MAP estimate* under a Gaussian (L2) or Laplacian (L1) prior on weights. So regularization is a Bayesian prior in disguise.

In deep learning, weight decay (L2) is also tightly coupled with optimizer behavior (decoupled in AdamW, coupled in vanilla Adam — a major Loshchilov & Hutter result). And as discussed in §5, weight decay biases toward flatter minima, providing additional regularization beyond the explicit norm penalty.

**X10. Connect loss-surface analysis with loss-function choice. Does the choice of loss affect minimum sharpness?**

Yes, in several ways:

- **Convexity in output**: CE and MSE are both convex in the model output, but composed with a non-linear network they create different surfaces. Empirically, CE-trained networks often have *sharper* minima than MSE-trained ones — but better generalization, possibly because CE creates a more useful inductive bias.
- **Outlier-robust losses (Huber, MAE)**: produce smoother loss surfaces because they cap gradient magnitudes. This can give flatter minima but also less informative gradient direction — trade-off.
- **Composite losses**: each component term contributes its own curvature; the total surface's flatness depends on alignment of components. Conflicting components → complex landscape with many sharp minima.
- **Label smoothing**: smooths the surface near the optimum (because logits don't grow without bound), tending to give flatter minima. Empirically improves generalization, consistent with the flat-minima story.

So loss choice doesn't just specify the optimum; it shapes the entire landscape and influences which optimum the optimizer finds.

#### Practical & debugging

**X11. Suppose you're seeing diverging training (loss → infinity or NaN). Walk through the diagnosis with respect to your loss choice.**

(1) **Numerical issues in the loss itself**:
- Cross-entropy: log of zero (probability vanishingly small), happens if logits are extreme. Use `cross_entropy_with_logits` to fuse softmax+log stably.
- NLL of Gaussian: tiny variance prediction → log term blows down, $(y-\mu)^2/\sigma^2$ blows up. Clip $\log\sigma^2$.
- Wasserstein critic: if Lipschitz constraint isn't enforced, critic can blow up.
- Soft Dice / IoU: division by sum of probabilities; if all predictions are zero, NaN. Add smoothing $\epsilon$.

(2) **Outlier examples producing huge gradients**:
- MSE with a massive outlier residual produces gradient proportional to residual.
- Switch to robust loss (MAE, Huber, log-cosh) or apply gradient clipping.

(3) **Learning rate too high** for the loss's curvature:
- Loss with large output gradients (e.g., MSE with un-normalized targets) needs smaller LR.
- Try a 10× smaller LR, see if it stabilizes.

(4) **Composite loss with exploding term**:
- Print per-term loss values. Likely one term is much larger than others.

(5) **Label issues**:
- Labels outside valid range (e.g., -1 instead of 0/1 in BCE).
- All-zero or all-one batches with weighted losses.

Always log per-batch losses, not just per-epoch; NaN spikes are easy to miss in epoch summaries.

**X12. You've moved your model from research to production. The training loss is still your old CE, but the business metric is "fraction of cases correctly handled, weighted by transaction value". How do you adjust?**

Several options to align loss with metric:

1. **Sample-weighted training**: apply per-example weights proportional to transaction value during training. This is the cheapest fix and works as long as transaction values aren't extreme outliers (if they are, you may need to log-transform or cap them).

2. **Cost-weighted CE with per-class weights**: if the metric breaks down to "false positives cost X, false negatives cost Y", use those costs as per-class weights.

3. **Threshold tuning**: even keeping standard CE, choose decision threshold to maximize the value-weighted accuracy on a held-out set.

4. **Quantile loss**: if business metric prefers conservative predictions in some regime, train at the appropriate quantile.

5. **Direct optimization** (REINFORCE on the actual metric): if the metric is non-decomposable across examples (e.g., F1, AUC), simple loss reweighting won't suffice.

Always validate with the *actual* business metric on held-out data — it's easy to over-fit to a surrogate.

**X13. Your team trained two LLMs, A and B. A has lower training cross-entropy. B has lower validation CE. Reviewers ask "which is better?" What's your answer?**

"It depends" — but flesh it out:

- If the difference in training CE is small (~10-20% relative) and validation CE clearly favors B, B generalizes better. Use B.
- If the difference is large in training CE but B is only marginally better in validation, A may have higher capacity used productively. Test on additional held-out distributions.
- For LLMs specifically, downstream task performance (MMLU, HumanEval, etc.) often matters more than raw validation perplexity. Best practice: report both and let downstream metrics decide.

Also worth examining: are the losses comparable? Same vocabulary, same tokenizer? Same evaluation set? Discrepancies here invalidate direct comparison.

For deployment: B (lower validation CE) is the safer default. For further research/scaling: investigate why A overfits more — different optimizer, batch size, schedule, regularization?

**X14. Why might your validation MSE be 1.5× your training MSE while validation accuracy is essentially the same as training? What does this say about your model's failure mode?**

The model's *predictions* are similar enough in argmax/threshold sense to give similar accuracy, but its *residuals* are larger on validation — meaning predictions are correct in direction but consistently miss the true value by more on validation. Two possible explanations:

(1) **Distribution shift in target scale**: the validation set has targets with larger magnitudes, so absolute residuals are larger even with proportional accuracy. Verify by computing relative error or normalized RMSE.

(2) **Overfitting to specific values, generalizing only on coarse class**: model has learned which "category" of value to predict but not the precise value within it. For continuous targets in classification-like problems (e.g., predicting price in dollars), this is common.

Mitigations: report multiple metrics (RMSE, MAE, R², MAPE) on both splits; investigate per-segment residuals to find where degradation happens; consider whether MSE is actually the right loss for the business question.

**X15. You're picking a loss for a regression task where the targets span 6 orders of magnitude (1 to 1,000,000). How would you proceed?**

(1) **Log-transform the target**: $y' = \log(1 + y)$. Then use MSE on log-targets — corresponds to multiplicative error in original space. This is usually the right choice for prices, populations, response times, etc.

(2) **Quantile loss** if you care about percentile predictions.

(3) **Heteroscedastic Gaussian NLL** if you want to model the variance scaling with magnitude (since variance is typically $\propto y^2$ for multiplicative errors, this often coincides with log-MSE).

(4) **Tweedie deviance loss** if your data has zero-inflation (insurance claims pattern).

Avoid: raw MSE, which will be dominated by the largest examples; raw MAE, which weights all errors equally regardless of magnitude (probably also wrong).

#### Probing & meta-questions

**X16. If you had to teach someone the single most important concept about loss functions, what would it be and why?**

"The loss defines what 'good' means — choose it carefully because the optimizer will give you exactly what you ask for, no more and no less." Concrete examples to illustrate: MSE vs MAE giving different optima (mean vs median); CE vs hinge giving calibrated vs uncalibrated outputs; pixel loss vs perceptual loss giving blurry vs sharp generations. The deeper point: ML practitioners spend most of their time on architecture and data, but loss design is often the highest-leverage choice — it specifies the objective, and everything else is just trying to optimize it.

**X17. What's the relationship between proper scoring rules, calibration, and consistency?**

A **proper scoring rule** is a loss for probabilistic predictions where reporting your true belief is optimal in expectation: $E_{Y \sim q}[\ell(q, Y)] \leq E_{Y \sim q}[\ell(p, Y)]$ for all $p$. NLL, Brier score, and CRPS are proper.

**Calibration** of model outputs: predictions of probability $p$ are correct with frequency $p$.

**Consistency** of an estimator/algorithm: as $n \to \infty$, the algorithm's risk (expected loss) approaches the Bayes risk.

Connection: minimizing a proper scoring rule on infinite data yields the true distribution as the optimum (calibration at the Bayes optimum). With finite data and limited model capacity, you can be miscalibrated even when minimizing a proper rule. Consistency is about asymptotic behavior; calibration is about finite-sample behavior of a specific model. Choosing a proper scoring rule is necessary but not sufficient for getting calibrated outputs in practice — temperature scaling and other post-hoc methods are still useful.

**X18. If you could only use one loss for all problems, which would you pick and why is that a stupid question?**

Negative log-likelihood is the most universal — most "named" losses are NLL of some distribution, and the framework adapts to any output type with the right likelihood. So defensibly: NLL.

Why the question is stupid: choosing the loss is choosing the *model* (the distributional assumption). "One loss for all problems" is "one distribution for all problems" — Gaussian for everything (MSE), Bernoulli for everything (BCE), etc. Each works for the things it fits and fails for the rest. The art is matching the loss to the problem.

It's also stupid because composite losses (multi-task, regularized, perceptual + adversarial + reconstruction) are how real systems work. No single loss handles object detection, super-resolution, ranking, and survival analysis equally well.

**X19. How do you think about loss design as your deployment context evolves (research → prototype → production)?**

In **research**, prioritize losses that give the cleanest gradient signal and clearest experimental conclusions — usually standard CE / MSE / NLL. Comparability across baselines matters.

In **prototype**, start with the simplest sensible loss for the problem; iterate quickly. Resist the urge to design a custom loss until you've established a baseline with a standard one.

In **production**, the loss must align with business metrics (which may not be standard ML metrics), be numerically stable across edge cases, integrate with monitoring (per-example losses for analysis), and support iteration without retraining everything from scratch. Composite losses are common; uncertainty-weighted multi-task losses are common; carefully tuned class weights are nearly always part of the picture for imbalanced problems.

**X20. What's an under-appreciated insight about losses that experienced practitioners learn the hard way?**

A few candidates:

- **Loss curves lie.** A monotonically decreasing training loss tells you almost nothing about model quality. You need validation loss, distributional metrics (calibration, per-class), and downstream task performance.
- **The "right" loss often produces worse training loss.** Adding label smoothing, weight decay, or SAM increases training loss while improving generalization. Don't optimize for training loss.
- **Composite losses need per-term monitoring.** A single "total loss" number hides which term is doing the work and which is being ignored. Always log each term separately.
- **Rare-class examples often need different treatment than just reweighting.** Sometimes the rare class is fundamentally harder (lower SNR, more diverse), requiring more capacity / data augmentation / specialized architecture, not just loss reweighting.
- **Out-of-distribution test loss can disagree wildly with in-distribution.** The loss surface near your minimum has different "shape" in different directions of distribution shift. Robustness requires dedicated testing on shifted distributions, not just in-distribution validation.
- **Calibration matters in production but is rarely measured during research.** A model with 90% accuracy and bad calibration is hard to integrate with downstream decision systems; the decision threshold is unstable across deployment regimes. Proper scoring rules + post-hoc calibration are basic production hygiene.

---

## Closing Notes

Loss functions are the most under-discussed and high-leverage component of an ML system. They specify the *objective*, while everything else (architecture, optimizer, data) is the *means*. A subtle change in loss can completely change what the model learns; a thoughtful loss can solve problems that better architectures cannot.

The unifying themes across this guide:

- **Most "named" losses are NLL of an implicit distributional assumption.** Choosing the loss is choosing the model. (MSE = Gaussian, MAE = Laplace, CE = Bernoulli/Categorical, etc.)
- **The optimal output under a loss is a specific summary of the conditional distribution.** Mean (MSE), median (MAE), full distribution (NLL), $\tau$-quantile (quantile loss). Pick the loss to match the summary you actually want.
- **Robustness, calibration, and computational stability are all loss properties.** Robust losses (MAE, Huber, log-cosh, Wasserstein) bound the per-example influence. Calibrated losses (CE, NLL) yield outputs you can interpret as probabilities. Numerically stable implementations (log-sum-exp tricks, fused softmax+CE, Sinkhorn iterations) matter as much as the math.
- **Composite losses are the norm in production**, not the exception. Object detection: classification + regression + (sometimes) auxiliary losses. Generative models: pixel + perceptual + adversarial. Multi-task: weighted sum across tasks. The art is balance.
- **The geometry of the loss surface determines whether your trained model generalizes.** Flat minima generalize better than sharp ones; SGD's noise, learning rate schedules, batch size, weight decay, and architectural choices like skip connections all influence which minima you find. SAM, SWA, and related methods explicitly seek flatness.

For interview prep, internalize the *mappings*: Gaussian → MSE, Bernoulli → BCE, Laplace → MAE; mean / median / quantile / mode optimums; gradient forms (especially $p - y$ for softmax+CE); when each loss family applies and why; common failure modes and their fixes.

For practice, always: verify your loss decreases as expected on a tiny sanity-check dataset; log per-term losses for composite objectives; evaluate on the actual target metric, not just the training loss; check calibration in production-bound systems; and remember that the loss is part of the model specification, not just an optimization detail.

---

**[← Previous Chapter: Optimization Theory](optimization_theory.md) | [Table of Contents](index.md) | [Next Chapter: Regularization →](regularization_guide.md)**
