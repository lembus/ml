# Optimization Theory — A Deep Reference for ML Interviews

> **Scope.** This guide is a standalone, textbook-style treatment of optimization as it appears in modern machine learning. It covers convex/non-convex landscapes, first-order methods (GD, SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW, AdaDelta), second-order methods (Newton, BFGS, L-BFGS), convergence analysis, and stochastic optimization (expected risk minimization, variance reduction). It assumes no prior exposure but builds up to the level expected of an ML / applied scientist interview.

---

## 1. Motivation & Intuition

### 1.1 Why does optimization exist?

Machine learning, at its computational core, is optimization. Suppose you want a model that predicts house prices. You choose a function family $f_\theta$ parameterized by $\theta$ (weights of a neural net, coefficients of a linear model, etc.), and you measure how wrong your predictions are with a **loss function** $L(\theta)$. Training the model means **finding parameters that make this loss small**:

$$
\theta^\star = \arg\min_\theta L(\theta).
$$

If the loss is small, your model predicts well. That is literally all "training" is. Everything else — architectures, data pipelines, evaluation — exists to set up, constrain, or measure this minimization.

The hard part is that $L$ is usually:

- **High-dimensional.** A modern neural net can have hundreds of billions of parameters; $\theta \in \mathbb{R}^d$ with $d$ astronomical.
- **Non-convex.** The surface has lots of valleys, ridges, and flat plateaus, not a single neat bowl.
- **Expensive to evaluate.** Computing $L(\theta)$ requires a forward pass over data; computing $\nabla L(\theta)$ requires backpropagation.
- **Stochastic.** We cannot afford to evaluate the true loss exactly, so we estimate it on mini-batches, introducing noise.

Optimization theory is the body of mathematics that tells us *which algorithms can find good $\theta$, how fast, and under what conditions*.

### 1.2 The loss landscape picture

Imagine standing on a mountain in a thick fog. You can feel the slope under your feet (the gradient) but cannot see the global terrain. You want to find the lowest point in the whole range. A reasonable plan is to step in the downhill direction, over and over. This is **gradient descent**, the simplest optimizer and the intellectual root of nearly every method used in practice.

- If the terrain is a single smooth bowl (convex), downhill always leads to the bottom. Done.
- If the terrain has multiple valleys (non-convex), downhill only promises you reach *some* valley, not necessarily the deepest.
- If the terrain has horse-saddle shapes (saddle points), downhill can even get you stuck temporarily, because the gradient goes to zero there without you being at a minimum.

Real neural network landscapes look more like the second and third: highly non-convex, riddled with saddle points, but — surprisingly — many of the local minima are "good enough" that we don't need the global one. A large part of modern optimization research is *not* about finding global minima but about navigating these landscapes quickly and cheaply.

### 1.3 Concrete example before any math

Suppose you have three data points $(x, y) = (1, 2), (2, 4), (3, 7)$ and want to fit $y = wx$. The squared-error loss is

$$
L(w) = \tfrac{1}{2}\left[(w \cdot 1 - 2)^2 + (w \cdot 2 - 4)^2 + (w \cdot 3 - 7)^2\right].
$$

Expand this and you get a parabola in $w$. You could solve it in closed form, but pretend you can't. You pick a starting guess $w_0 = 0$, compute the slope $L'(0)$, see it's negative (loss decreases as $w$ increases), so you step right. Compute slope again, still negative but smaller, step right again. Eventually you overshoot, slope turns positive, you step left, and you oscillate near the minimum. With a small enough step size, you converge.

Every optimizer we will discuss is a variation on this idea: different ways of *choosing the step direction and step size*, sometimes using curvature, sometimes using history, sometimes using noise — but always following the same "evaluate, update, repeat" loop.

### 1.4 Why the algorithm matters

You might think "just use the best optimizer." But the best optimizer depends on everything:

- **SGD with momentum** tends to generalize better on vision models with clean labels.
- **AdamW** is the near-universal default for transformers.
- **L-BFGS** dominates on small, deterministic, well-conditioned problems (classical ML, some fine-tuning, physics-informed networks).
- **Second-order methods** become attractive again at massive scale because they converge in far fewer iterations, even though each iteration is expensive.

Understanding *why* each algorithm was invented — what failure mode it fixes — is the difference between a practitioner who tunes optimizers by trial and error and one who reasons about them.

---

## 2. Conceptual Foundations

### 2.1 The anatomy of an optimization problem

An **unconstrained minimization problem** has the form

$$
\min_{\theta \in \mathbb{R}^d} \; f(\theta),
$$

where $f : \mathbb{R}^d \to \mathbb{R}$ is the **objective function** (also called loss, cost, or energy). In ML, $\theta$ is the vector of model parameters and $f$ is the training loss.

Key objects associated with $f$:

- **Gradient** $\nabla f(\theta) \in \mathbb{R}^d$ — the vector of partial derivatives $\partial f / \partial \theta_i$. Points in the direction of steepest *ascent*; its magnitude measures the local rate of change.
- **Hessian** $\nabla^2 f(\theta) \in \mathbb{R}^{d \times d}$ — the matrix of second partial derivatives $\partial^2 f / (\partial \theta_i \partial \theta_j)$. Describes local curvature: how the gradient changes as you move.
- **Directional derivative** $\nabla f(\theta)^\top v$ — rate of change of $f$ along direction $v$. Zero when $v$ is orthogonal to the gradient.

### 2.2 Critical points: minima, maxima, saddles

A **critical point** (also called stationary point) is any $\theta^\star$ where $\nabla f(\theta^\star) = 0$. This is necessary but not sufficient for being a minimum. To classify it, we look at the Hessian $H = \nabla^2 f(\theta^\star)$:

- **Local minimum:** $H$ is positive semidefinite, and strictly positive definite in a neighborhood. All directions curve upward.
- **Local maximum:** $H$ is negative semidefinite. All directions curve downward.
- **Saddle point:** $H$ has both positive and negative eigenvalues. Some directions curve up, others curve down.
- **Degenerate critical point:** $H$ has a zero eigenvalue. Higher-order information is needed (it's a "flat" direction). Common in overparameterized neural nets.

A **global minimum** is a point with the smallest $f$ value *anywhere*. A **local minimum** is one with the smallest value in some *neighborhood*. In convex problems every local minimum is global; in non-convex ones, they need not coincide.

**Why saddle points matter in deep learning.** It was long believed local minima were the main obstacle in training neural nets. Work around 2014 (Dauphin et al.) showed that in high dimensions, critical points are overwhelmingly *saddle points*, not local minima. Intuitively: for a point to be a local minimum, *every* one of $d$ eigenvalues of $H$ must be non-negative. The chance of that dwindles as $d$ grows. Plain gradient descent can slow down dramatically near saddles because $\|\nabla f\|$ is small there. Momentum-based and adaptive methods are much more effective at escaping them.

### 2.3 Convex sets, convex functions, and why convexity is special

A **convex set** $C \subseteq \mathbb{R}^d$ is one where, for any two points $x, y \in C$ and $\lambda \in [0,1]$, the segment $\lambda x + (1-\lambda) y \in C$. Lines, balls, and half-spaces are convex; donuts and most sublevel sets of neural-net losses are not.

A function $f$ is **convex** if its domain is a convex set and for any $x, y$ and $\lambda \in [0,1]$,

$$
f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y).
$$

Geometrically: the chord between any two points on the graph lies on or above the graph. A function is **strictly convex** if the inequality is strict whenever $x \neq y$ and $\lambda \in (0,1)$. It is **strongly convex with parameter $\mu > 0$** if

$$
f(y) \ge f(x) + \nabla f(x)^\top (y - x) + \tfrac{\mu}{2} \|y - x\|^2,
$$

i.e., $f$ lies above a quadratic lower bound at every point. Equivalently, $\nabla^2 f \succeq \mu I$ when the Hessian exists.

**Why convexity is the holy grail for optimization:**

1. **Every local minimum is global.** If you find *any* stationary point, you've found the answer.
2. **Gradient descent converges to the global optimum** (under mild smoothness).
3. **Convergence rates are provable and often fast** (linear rate for strongly convex smooth functions).
4. **Duality theory** gives us principled bounds and efficient algorithms (e.g., SVMs, LASSO).

**Why neural networks violate convexity.** Neural nets are compositions of linear maps and non-linear activations. Even a single hidden layer breaks convexity. The loss surface of a deep network has combinatorially many symmetric minima (you can permute neurons in a hidden layer without changing the function), each giving the same loss but at different $\theta$. Convex analysis still *guides* our methods — many algorithms are designed assuming convexity and then applied as heuristics — but the guarantees weaken or disappear.

### 2.4 Smoothness and Lipschitz conditions

Two regularity assumptions dominate optimization theory:

- **$L$-smoothness:** $\nabla f$ is Lipschitz continuous with constant $L$:
  $$
  \|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|.
  $$
  Equivalently, when twice differentiable, $\|\nabla^2 f\| \le L$. This bounds curvature from above and is what tells us how big a step we can take without overshooting.

- **$\mu$-strong convexity:** as above. Bounds curvature from below and guarantees the function has a unique minimum.

The ratio $\kappa = L / \mu$ is the **condition number**. Small $\kappa$ (close to 1) means the loss surface is "round"; large $\kappa$ means it's "stretched." Ill-conditioned problems are slow for gradient descent because the optimal step in one direction is too large in another.

**What breaks when assumptions fail:**

- If the gradient is not Lipschitz (e.g., $f(x) = x^4$ far from zero, or ReLU networks at kinks), theoretical step-size bounds no longer apply; you need adaptive or line-search methods.
- If $\mu = 0$ (convex but not strongly convex), you lose linear convergence and drop to sublinear rates.
- If the function is non-convex, you lose global convergence — you can only hope to converge to a *stationary* point, which might be a saddle.

### 2.5 Deterministic vs stochastic optimization

In ML, the true loss is an *expectation* over data distribution:

$$
F(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f_\theta(x), y)].
$$

We do not see $\mathcal{D}$; we only see a finite sample. Two related problems arise:

- **Expected Risk Minimization (ERM, the "true" problem):** minimize $F$.
- **Empirical Risk Minimization (empirical proxy):** minimize $\hat{F}(\theta) = \tfrac{1}{n}\sum_{i=1}^n \ell_i(\theta)$ on the training set.

Deterministic optimization operates on $\hat F$. But with $n$ in the millions or billions, even computing $\hat F$ once is costly. **Stochastic optimization** replaces $\nabla \hat F$ with a cheap unbiased estimate (e.g., the gradient on a mini-batch). The price is noise; the benefit is throughput. The entire modern training stack — SGD, Adam, variance reduction methods — exists to handle this trade-off.

### 2.6 Step size, descent directions, and the update template

Almost every optimizer in this document has the form

$$
\theta_{t+1} = \theta_t + \eta_t \, d_t,
$$

where $d_t$ is a **descent direction** (satisfies $\nabla f(\theta_t)^\top d_t < 0$) and $\eta_t > 0$ is a **step size** (learning rate). Different optimizers differ in:

- How $d_t$ is chosen (raw gradient, momentum-adjusted, preconditioned, Newton step, quasi-Newton approximation).
- How $\eta_t$ is chosen (fixed, decaying schedule, line search, adaptive per-coordinate).

Keeping this template in mind makes the zoo of optimizers much less intimidating.

---

## 3. Mathematical Formulation

### 3.1 The Taylor expansion: the shared ancestor

All of optimization grows out of Taylor expansion. Around a current iterate $\theta_t$,

$$
f(\theta_t + p) \approx f(\theta_t) + \nabla f(\theta_t)^\top p + \tfrac{1}{2} p^\top \nabla^2 f(\theta_t) \, p + O(\|p\|^3).
$$

Different optimizers truncate and modify this expansion differently:

- **First-order methods** use only the linear term and replace the Hessian with a simple scalar $1/\eta$ or a heuristic diagonal.
- **Second-order methods** use the quadratic approximation in full.
- **Quasi-Newton methods** approximate the Hessian using past gradients.

### 3.2 Gradient Descent (GD)

**Derivation.** Minimize the quadratic surrogate $m_t(p) = f(\theta_t) + \nabla f(\theta_t)^\top p + \tfrac{1}{2\eta}\|p\|^2$. Setting $\nabla m_t(p) = 0$ gives $p = -\eta \nabla f(\theta_t)$, so

$$
\boxed{\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)}.
$$

We can equivalently derive it from the smoothness inequality: if $f$ is $L$-smooth,

$$
f(y) \le f(x) + \nabla f(x)^\top (y - x) + \tfrac{L}{2}\|y - x\|^2.
$$

Plugging $y = x - \eta \nabla f(x)$ gives

$$
f(x - \eta \nabla f(x)) \le f(x) - \eta\|\nabla f(x)\|^2 + \tfrac{L\eta^2}{2}\|\nabla f(x)\|^2 = f(x) - \eta(1 - \tfrac{L\eta}{2})\|\nabla f(x)\|^2.
$$

Any $\eta < 2/L$ guarantees a decrease. The optimal choice for this bound is $\eta = 1/L$, yielding

$$
f(x_{t+1}) \le f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2.
$$

This is the fundamental **descent lemma** and the starting point for most convergence proofs.

### 3.3 Stochastic Gradient Descent (SGD) and mini-batching

When $\hat{F}(\theta) = \tfrac{1}{n}\sum_i \ell_i(\theta)$, the full gradient costs $O(n)$. **Stochastic gradient descent** replaces it with a single-sample estimate: draw $i_t$ uniformly and use $g_t = \nabla \ell_{i_t}(\theta_t)$. It is **unbiased**: $\mathbb{E}[g_t] = \nabla \hat F(\theta_t)$.

$$
\theta_{t+1} = \theta_t - \eta_t g_t.
$$

**Mini-batch SGD** averages over a mini-batch $B_t$ of size $b$:

$$
g_t = \tfrac{1}{b}\sum_{i \in B_t} \nabla \ell_i(\theta_t), \qquad \text{Var}(g_t) = \tfrac{\sigma^2}{b}.
$$

The variance scales as $1/b$, so larger batches give smoother updates but linearly more compute per step. The trade-off is central to modern training.

**Why stochastic noise is OK.** It even *helps* in non-convex landscapes by perturbing the iterate out of shallow local minima and saddle points. The implicit regularization effect of small-batch SGD is a well-documented reason vision models trained with SGD often generalize better than those trained with Adam.

### 3.4 Momentum (classical / heavy-ball)

GD in long narrow valleys oscillates: it corrects aggressively across the valley but barely moves along it. **Momentum** smooths the trajectory by averaging past gradients.

Polyak's **heavy-ball** update introduces a velocity vector $v_t$:

$$
\begin{aligned}
v_{t+1} &= \beta v_t - \eta \nabla f(\theta_t), \\
\theta_{t+1} &= \theta_t + v_{t+1}.
\end{aligned}
$$

Typical values: $\beta \in [0.9, 0.99]$. Unrolling, the update is a geometrically weighted sum of past gradients:

$$
v_{t+1} = -\eta \sum_{k=0}^{t} \beta^{t-k} \nabla f(\theta_k).
$$

In a narrow valley, along the steep axis gradients alternate sign and average toward zero; along the flat axis they have the same sign and accumulate. Effective step sizes differ by axis — exactly what we wanted.

**Physical analogy.** $\theta$ is a ball rolling on the loss surface with momentum (mass), friction $1 - \beta$, and a gravitational force $-\nabla f$.

### 3.5 Nesterov Accelerated Gradient (NAG)

Nesterov's 1983 variant improves the worst-case convergence rate from $O(1/k)$ to $O(1/k^2)$ for smooth convex problems — a genuinely surprising result that for a long time was thought to be the end of the story for first-order methods.

Nesterov's trick is to compute the gradient at a **lookahead** point, not at $\theta_t$:

$$
\begin{aligned}
\tilde{\theta}_t &= \theta_t + \beta v_t, \\
v_{t+1} &= \beta v_t - \eta \nabla f(\tilde{\theta}_t), \\
\theta_{t+1} &= \theta_t + v_{t+1}.
\end{aligned}
$$

Intuition: momentum will carry you to roughly $\theta_t + \beta v_t$ anyway; evaluate the gradient *there* so you can correct before you overshoot.

An equivalent reformulation, convenient for deep learning frameworks, is

$$
\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t) + \beta (\theta_t - \theta_{t-1}) - \beta \eta (\nabla f(\theta_t) - \nabla f(\theta_{t-1})).
$$

The "$-\beta \eta (\nabla f(\theta_t) - \nabla f(\theta_{t-1}))$" term is the correction that classical momentum lacks — a first-order approximation to curvature without computing the Hessian.

### 3.6 Adaptive methods: why they exist

On deep networks some coordinates of $\theta$ (e.g., in embedding layers for rare tokens) see gradient signal rarely; others (for frequent tokens) see it constantly. A single scalar $\eta$ is either too big for the frequent coordinates or too small for the rare ones. **Adaptive methods** keep a per-coordinate running statistic of gradient magnitudes and scale the step accordingly.

#### 3.6.1 AdaGrad (Duchi, Hazan, Singer, 2011)

Accumulate squared gradients element-wise and divide the step by the square root:

$$
\begin{aligned}
G_t &= G_{t-1} + g_t \odot g_t, \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t.
\end{aligned}
$$

Here $\odot$ is element-wise product and division is element-wise. Effective per-coordinate step $\eta / \sqrt{G_t^{(i)}}$ shrinks as that coordinate accumulates history — great for sparse features, but the denominator grows monotonically and learning effectively halts on long training runs.

#### 3.6.2 RMSProp (Hinton, 2012 lecture notes)

Fix AdaGrad's shrinking-step problem by using an **exponential moving average** instead of a sum:

$$
\begin{aligned}
v_t &= \rho \, v_{t-1} + (1 - \rho) \, g_t \odot g_t, \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t.
\end{aligned}
$$

Typical $\rho = 0.9$ or $0.99$. $v_t$ estimates the recent average squared gradient, so the scale adjusts to the current regime rather than the entire history.

#### 3.6.3 Adam (Kingma & Ba, 2014)

Adam = Momentum + RMSProp + bias correction. Maintain first and second moment estimates:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t.
\end{aligned}
$$

At step $t$, $m_t$ is biased toward zero (since $m_0 = 0$ and EMA warms up slowly). Correct with

$$
\hat m_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat v_t = \frac{v_t}{1 - \beta_2^t}.
$$

Update:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat v_t} + \epsilon} \hat m_t.
$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The bias correction is critical in the first ~1000 steps; without it, early updates are dramatically under-scaled.

#### 3.6.4 AdamW (Loshchilov & Hutter, 2017)

In vanilla Adam, **L2 regularization** is implemented by adding $\lambda \theta$ to the gradient. But Adam then divides this term by $\sqrt{\hat v_t}$, so parameters with large historical gradients get *less* weight decay — the opposite of what L2 is supposed to do. AdamW **decouples** weight decay from the gradient-based update:

$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda \theta_t \right).
$$

The $\lambda \theta_t$ term is applied *after* the adaptive rescaling, so every parameter is decayed uniformly. This single change turns out to meaningfully improve generalization, and AdamW is now the default for training transformers.

#### 3.6.5 AdaDelta (Zeiler, 2012)

Like RMSProp but eliminates the need for a learning rate $\eta$ by tracking an EMA of squared *updates* too:

$$
\begin{aligned}
v_t &= \rho v_{t-1} + (1-\rho) g_t \odot g_t, \\
\Delta \theta_t &= -\frac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}} \odot g_t, \\
u_t &= \rho u_{t-1} + (1-\rho) \Delta \theta_t \odot \Delta \theta_t, \\
\theta_{t+1} &= \theta_t + \Delta \theta_t.
\end{aligned}
$$

The ratio $\sqrt{u}/\sqrt{v}$ makes the update dimensionally consistent (a pseudo-Newton step) and eliminates $\eta$. In practice it's rarely used now; AdamW dominates, but AdaDelta is a useful conceptual stop on the road.

### 3.7 Newton's method

Return to the quadratic Taylor expansion and *don't* truncate the Hessian:

$$
m_t(p) = f(\theta_t) + \nabla f(\theta_t)^\top p + \tfrac{1}{2} p^\top H_t \, p, \qquad H_t = \nabla^2 f(\theta_t).
$$

If $H_t \succ 0$, minimize over $p$: $\nabla_p m_t = \nabla f(\theta_t) + H_t p = 0$, giving

$$
\boxed{p_t = -H_t^{-1} \nabla f(\theta_t), \quad \theta_{t+1} = \theta_t + p_t}.
$$

**Quadratic convergence.** For sufficiently smooth $f$ with invertible Hessian at $\theta^\star$, once you're close enough the error roughly *squares* at every step: $\|\theta_{t+1} - \theta^\star\| \le C\|\theta_t - \theta^\star\|^2$. An error of $10^{-3}$ becomes $10^{-6}$, then $10^{-12}$ — so three Newton steps can do what gradient descent takes thousands of iterations to achieve.

**On a pure quadratic $f(x) = \tfrac{1}{2}x^\top A x - b^\top x$:** $\nabla f = Ax - b$, $H = A$, so $p = -A^{-1}(Ax - b) = -x + A^{-1}b$, yielding $x_{t+1} = A^{-1}b$ — the exact minimum, in **one step**.

**Practical obstacles:**

- **Hessian storage:** $d^2$ entries. At $d = 10^9$ this is $10^{18}$ floats — impossible.
- **Hessian inversion:** $O(d^3)$ per step.
- **Indefiniteness:** Far from the optimum, $H$ may have negative eigenvalues; $-H^{-1}\nabla f$ may not be a descent direction. Common fixes: **damping** (solve $(H + \lambda I) p = -\nabla f$, a trust-region / Levenberg–Marquardt regularization), **modified Cholesky**, or **Gauss–Newton** (approximate $H \approx J^\top J$, guaranteed PSD).

### 3.8 Quasi-Newton methods: BFGS and L-BFGS

Quasi-Newton methods avoid computing $H$ directly by maintaining a matrix $B_t \approx H_t$ (or $H_t \approx B_t^{-1}$) that is updated using gradient differences.

Define

$$
s_t = \theta_{t+1} - \theta_t, \qquad y_t = \nabla f(\theta_{t+1}) - \nabla f(\theta_t).
$$

The **secant equation** $B_{t+1} s_t = y_t$ demands that our approximation correctly captures the observed change in gradient along $s_t$. (Heuristically: $y_t \approx H s_t$ by the mean value theorem.) This alone does not determine $B_{t+1}$; we pick it to be as close as possible to $B_t$ in some sense, subject to symmetry and positive definiteness.

**BFGS update** (for the inverse Hessian approximation $H_t \approx \nabla^2 f^{-1}$):

$$
H_{t+1} = \left(I - \rho_t s_t y_t^\top\right) H_t \left(I - \rho_t y_t s_t^\top\right) + \rho_t s_t s_t^\top, \qquad \rho_t = \frac{1}{y_t^\top s_t}.
$$

Update rule: $\theta_{t+1} = \theta_t - \eta_t H_t \nabla f(\theta_t)$, with $\eta_t$ chosen by line search.

**Properties:**

- Preserves positive definiteness when $y_t^\top s_t > 0$ (the **curvature condition**), which holds automatically with exact or appropriate inexact line search.
- **Superlinear convergence** under standard assumptions — between linear (first-order) and quadratic (Newton).
- Stores a dense $d \times d$ matrix — $O(d^2)$ memory, $O(d^2)$ per update.

**L-BFGS (Limited-memory BFGS)** never materializes $H_t$. It stores only the last $m$ pairs $(s_i, y_i)$ (typically $m = 5$–$20$) and computes $H_t \nabla f(\theta_t)$ via a two-loop recursion in $O(md)$ time and $O(md)$ memory. This is what makes it usable for problems with millions of parameters. The price: $H_t$ is now a "limited memory" approximation; convergence is slightly worse than full BFGS but still typically linear/superlinear.

**L-BFGS two-loop recursion** (sketch). Let $q = \nabla f(\theta_t)$. First loop (backward over $i = t-1, \dots, t-m$):

$$
\alpha_i = \rho_i s_i^\top q, \quad q \leftarrow q - \alpha_i y_i.
$$

Initialize $r = H_0 q$ (often $H_0 = \gamma I$ with $\gamma = (s_{t-1}^\top y_{t-1})/(y_{t-1}^\top y_{t-1})$). Second loop (forward):

$$
\beta_i = \rho_i y_i^\top r, \quad r \leftarrow r + (\alpha_i - \beta_i) s_i.
$$

Return $r$ as $H_t \nabla f(\theta_t)$.

**L-BFGS is most useful when:** the problem is deterministic (full-batch) or low-noise, curvature is meaningful (no batch norm or other stochastic layers confusing the secant condition), and the landscape is well-behaved. It is the default in scikit-learn's logistic regression, many SciPy optimizers, and physics-informed neural networks.

### 3.9 Convergence rates: the language

A sequence $\theta_t \to \theta^\star$ converges:

- **Linearly** (geometric, exponential): $\|\theta_t - \theta^\star\| \le C \rho^t$ for some $\rho \in (0,1)$. Error drops by a constant factor each step. On a log plot, the error is a straight line.
- **Sublinearly**: e.g., $O(1/t)$ or $O(1/\sqrt{t})$. Error decays but slower than exponential.
- **Superlinearly**: $\|\theta_{t+1} - \theta^\star\| / \|\theta_t - \theta^\star\| \to 0$. Faster than linear but slower than quadratic.
- **Quadratically**: $\|\theta_{t+1} - \theta^\star\| \le C\|\theta_t - \theta^\star\|^2$. The gold standard; doubling the number of correct digits each step.

When we talk about "rate $O(1/k)$" we usually mean the *optimality gap* $f(\theta_k) - f(\theta^\star) = O(1/k)$.

**Canonical rates for minimizing smooth ($L$-Lipschitz gradient) $f$:**

| Setting | Gradient descent | Nesterov | SGD | Newton |
|---|---|---|---|---|
| Smooth convex | $O(1/k)$ | $O(1/k^2)$ | $O(1/\sqrt{k})$ | — |
| Smooth strongly convex | $O((1 - \mu/L)^k)$ (linear) | $O((1 - \sqrt{\mu/L})^k)$ | $O(1/k)$ | Quadratic (local) |
| Smooth non-convex | $\min_t \|\nabla f\|^2 = O(1/k)$ | $O(1/k)$ | $O(1/\sqrt{k})$ | local quadratic |

For strongly convex problems, Nesterov's rate improves the exponent of the condition number from $\kappa$ to $\sqrt{\kappa}$ — a massive improvement when $\kappa = 10^6$.

**Derivation sketch of GD's $O(1/k)$ rate for smooth convex $f$** (convexity + descent lemma). The descent lemma gives

$$
f(\theta_{t+1}) \le f(\theta_t) - \tfrac{1}{2L}\|\nabla f(\theta_t)\|^2.
$$

Convexity gives $f(\theta_t) - f(\theta^\star) \le \nabla f(\theta_t)^\top (\theta_t - \theta^\star) \le \|\nabla f(\theta_t)\|\|\theta_t - \theta^\star\|$. Standard algebra, summing over $t$, yields

$$
f(\theta_k) - f(\theta^\star) \le \frac{L\|\theta_0 - \theta^\star\|^2}{2k}.
$$

**Derivation sketch of linear rate under strong convexity.** Strong convexity implies $\|\nabla f(\theta)\|^2 \ge 2\mu (f(\theta) - f(\theta^\star))$ (Polyak–Łojasiewicz inequality). Combined with the descent lemma, this gives

$$
f(\theta_{t+1}) - f(\theta^\star) \le (1 - \mu/L)(f(\theta_t) - f(\theta^\star)).
$$

Iterating: $f(\theta_k) - f(\theta^\star) \le (1 - \mu/L)^k (f(\theta_0) - f(\theta^\star))$ — geometric decay.

**Why SGD is slower.** The gradient estimate has bounded variance $\sigma^2$, so the descent lemma gets a $+\tfrac{L\eta^2 \sigma^2}{2}$ noise term. To keep the noise term from dominating, you need $\eta_t \to 0$. Standard schedule $\eta_t = c/\sqrt{t}$ gives $O(1/\sqrt{k})$ in the convex case. The intuition is fundamental: you cannot converge faster than the noise allows unless you *reduce* the noise.

### 3.10 Stochastic optimization: ERM and noise structure

The ML training problem is

$$
\min_\theta F(\theta) = \mathbb{E}_\xi[f(\theta; \xi)],
$$

where $\xi$ is the randomness (which example, which mini-batch, which data augmentation). In ERM, $\xi$ is uniform over training indices. Stochastic gradients

$$
g_t = \nabla f(\theta_t; \xi_t), \qquad \mathbb{E}[g_t \mid \theta_t] = \nabla F(\theta_t)
$$

are unbiased but have variance $\sigma^2 = \mathbb{E}\|g_t - \nabla F(\theta_t)\|^2$.

**Two noise regimes matter in ML:**

1. **Strong noise:** $\sigma^2$ is large compared to $\|\nabla F\|$ near the optimum. SGD requires decaying step sizes and plateaus at error $O(\sigma^2 \eta)$ with fixed $\eta$.
2. **Interpolation / overparameterization:** when a neural net can fit the training set exactly, $\nabla f_i(\theta^\star) = 0$ for every $i$, so $\sigma^2 \to 0$ as $\theta \to \theta^\star$. SGD then recovers GD-like linear rates with a fixed step size. This is why overparameterized deep nets can be trained with fixed or slowly-decaying learning rates.

### 3.11 Variance reduction techniques

A class of algorithms that achieve full-batch convergence rates with per-iteration cost closer to SGD, by using clever control variates.

#### 3.11.1 SVRG (Stochastic Variance Reduced Gradient, Johnson & Zhang, 2013)

Periodically (every $m$ steps) compute the full gradient $\tilde \mu = \nabla \hat F(\tilde \theta)$ at an anchor point $\tilde \theta$. Use it as a control variate:

$$
g_t = \nabla f_{i_t}(\theta_t) - \nabla f_{i_t}(\tilde \theta) + \tilde \mu.
$$

This is still an unbiased estimator of $\nabla \hat F(\theta_t)$, but with much lower variance — $\nabla f_{i_t}(\theta_t) - \nabla f_{i_t}(\tilde \theta)$ captures the *change*, which is small when $\theta_t$ is near $\tilde \theta$. SVRG achieves **linear convergence** for smooth strongly convex ERM without decaying step sizes.

#### 3.11.2 SAG / SAGA (Le Roux et al., 2012; Defazio et al., 2014)

Keep a table of the most recent $\nabla f_i$ for every $i$ and update only one entry per step. The average of the table is an unbiased estimator of $\nabla \hat F$ with reducing variance. SAGA is an unbiased variant of SAG that inherits SVRG-like guarantees with smaller memory for certain problems.

**Why variance reduction is less common in deep learning.**

- Requires $O(n)$ extra memory (SAG/SAGA) or periodic full passes (SVRG).
- The theoretical gains assume smoothness and strong convexity — neural nets have neither.
- Practical gains are small or negative with random data augmentation and dropout.

Variance reduction methods are widely used in convex ML (logistic regression, SVMs at scale) and in certain corners of reinforcement learning. For deep nets, SGD-with-momentum and Adam remain dominant.

### 3.12 Summary of update rules (reference card)

| Method | Update | Extra memory |
|---|---|---|
| GD | $\theta_{t+1} = \theta_t - \eta \nabla f$ | None |
| SGD | $\theta_{t+1} = \theta_t - \eta g_t$ | None |
| Momentum | $v \leftarrow \beta v - \eta g;\; \theta \leftarrow \theta + v$ | $O(d)$ |
| NAG | gradient at lookahead $\tilde\theta = \theta + \beta v$ | $O(d)$ |
| AdaGrad | $\theta \leftarrow \theta - \eta g / \sqrt{G};\; G \mathrel{+}= g^2$ | $O(d)$ |
| RMSProp | $v \leftarrow \rho v + (1-\rho)g^2;\; \theta \leftarrow \theta - \eta g/\sqrt v$ | $O(d)$ |
| Adam | momentum + RMSProp + bias correction | $O(d)$ |
| AdamW | Adam with decoupled weight decay | $O(d)$ |
| Newton | $\theta \leftarrow \theta - H^{-1} g$ | $O(d^2)$ + $O(d^3)$ compute |
| BFGS | secant-based $H$ update | $O(d^2)$ |
| L-BFGS | stored history of $m$ pairs | $O(md)$ |
| SVRG | $g = \nabla f_i(\theta) - \nabla f_i(\tilde\theta) + \tilde\mu$ | $O(d)$ + periodic full pass |

---

## 4. Worked Examples

### 4.1 A fully worked 2D quadratic

Let $f(x, y) = \tfrac{1}{2}(x^2 + 10 y^2)$. The Hessian is $H = \begin{pmatrix} 1 & 0 \\ 0 & 10 \end{pmatrix}$. Eigenvalues $\mu = 1$, $L = 10$, so $\kappa = 10$. Minimum is $(0, 0)$ with $f = 0$. Start at $(x_0, y_0) = (10, 1)$ so initial loss is $\tfrac{1}{2}(100 + 10) = 55$.

**Gradient:** $\nabla f(x, y) = (x, 10y)$.

#### Gradient descent, $\eta = 1/L = 0.1$

Step 1: $\nabla f(10, 1) = (10, 10)$. Update: $(10 - 0.1 \cdot 10, 1 - 0.1 \cdot 10) = (9, 0)$. Loss: $\tfrac{1}{2}(81 + 0) = 40.5$.

Step 2: $\nabla f(9, 0) = (9, 0)$. Update: $(8.1, 0)$. Loss: $32.805$.

Notice GD killed the stiff $y$ direction in one step (because $\eta = 1/L = 1/(10) = 0.1$ and $\eta \cdot 10 = 1$ exactly cancels the update), but it creeps along $x$ at rate $1 - 0.1 = 0.9$ per step. After 50 steps, $x \approx 10 \cdot 0.9^{50} \approx 0.052$, loss $\approx 0.0014$. It would take ~130 steps to reach $f < 10^{-6}$.

This is the quintessential picture of ill-conditioning: one direction converges immediately, the other creeps. The ratio of timescales is exactly the condition number $\kappa = 10$.

#### Momentum, $\eta = 0.1$, $\beta = 0.9$

Step 0: $v_0 = 0$, $\theta_0 = (10, 1)$.

Step 1: $g_1 = (10, 10)$. $v_1 = 0.9 \cdot 0 - 0.1 \cdot (10, 10) = (-1, -1)$. $\theta_1 = (9, 0)$. Loss = 40.5.

Step 2: $g_2 = (9, 0)$. $v_2 = 0.9 \cdot (-1, -1) - 0.1 \cdot (9, 0) = (-1.8, -0.9)$. $\theta_2 = (7.2, -0.9)$. Loss = $\tfrac{1}{2}(51.84 + 8.1) = 29.97$.

Step 3: $g_3 = (7.2, -9)$. $v_3 = 0.9 \cdot (-1.8, -0.9) - 0.1 \cdot (7.2, -9) = (-2.34, -0.81) + (0, 0.9) \cdot \text{wait let me recompute}$. 

Redo cleanly: $v_3 = 0.9 \cdot (-1.8, -0.9) - 0.1 \cdot (7.2, -9) = (-1.62, -0.81) + (-0.72, 0.9) = (-2.34, 0.09)$. $\theta_3 = (7.2, -0.9) + (-2.34, 0.09) = (4.86, -0.81)$.

Observe: in the $y$ direction momentum swings across the valley, *overshooting* the minimum. Classical momentum can undershoot or overshoot depending on $\beta$ and $\eta$. Over many iterations it averages out and converges faster than GD in the slow $x$ direction. Eventually momentum outperforms GD by a factor $\sqrt{\kappa} \approx 3$ in convergence rate.

#### Newton's method

$p = -H^{-1} \nabla f$. Since $f$ is quadratic, $\nabla f = H \theta$, so $p = -H^{-1} H \theta = -\theta$. Update: $\theta_1 = \theta_0 - \theta_0 = (0, 0)$. **One step to the exact minimum**, regardless of the starting point or condition number. This is the best-case scenario that makes Newton's method so appealing in principle, though we usually cannot afford $H^{-1}$.

### 4.2 Linear regression by SGD: concrete arithmetic

Take $n = 4$ points: $(x, y) = (1, 2), (2, 3.8), (3, 5.9), (4, 8.1)$. Model: $\hat y = w x$. Per-sample loss $\ell_i(w) = \tfrac{1}{2}(w x_i - y_i)^2$. Gradient $\ell_i'(w) = x_i(w x_i - y_i)$.

Start $w_0 = 0$, $\eta = 0.03$.

- **SGD step 1**: pick $i=1$. $g = 1 \cdot (0 - 2) = -2$. $w_1 = 0 - 0.03 \cdot (-2) = 0.06$.
- **SGD step 2**: pick $i=3$. $g = 3 \cdot (0.06 \cdot 3 - 5.9) = 3 \cdot (0.18 - 5.9) = 3 \cdot (-5.72) = -17.16$. $w_2 = 0.06 - 0.03 \cdot (-17.16) = 0.06 + 0.515 = 0.575$.
- **SGD step 3**: pick $i=2$. $g = 2 \cdot (0.575 \cdot 2 - 3.8) = 2 \cdot (1.15 - 3.8) = 2 \cdot (-2.65) = -5.3$. $w_3 = 0.575 + 0.03 \cdot 5.3 = 0.734$.
- **SGD step 4**: pick $i=4$. $g = 4 \cdot (0.734 \cdot 4 - 8.1) = 4 \cdot (2.936 - 8.1) = 4 \cdot (-5.164) = -20.656$. $w_4 = 0.734 + 0.03 \cdot 20.656 = 1.353$.

Notice the erratic step sizes — step 2 and step 4 were enormous because they used the points with large $x$, which have large gradients. Full-batch GD would average these out; SGD bounces. The closed-form optimum is $w^\star \approx 2.01$, and continued SGD will fluctuate toward it.

**Mini-batch of size 2** smooths this out. Batch $\{1, 2\}$ at $w = 0$: $g = \tfrac{1}{2}[(0 - 2) + 2(0 - 3.8)] = \tfrac{1}{2}[-2 - 7.6] = -4.8$. Step: $w_1 = 0.144$. The variance is visibly lower.

### 4.3 A saddle point that traps plain GD

Consider $f(x, y) = x^2 - y^2$ at $(0, 0)$. Gradient $(2x, -2y)$, Hessian $\operatorname{diag}(2, -2)$ — a saddle. If you start exactly on the $y = 0$ line, GD has gradient $(2x, 0)$ and moves you straight to $(0, 0)$ and stays there; you are stuck at a saddle forever.

Start instead at $(10, 0.01)$ (a tiny $y$ perturbation). Step $\eta = 0.1$: gradient $(20, -0.02)$, next point $(8, 0.012)$. The $y$ coordinate *grows* exponentially: $y_{t+1} = y_t + 0.1 \cdot 2 y_t = 1.2 y_t$. Eventually $y$ escapes to $\pm \infty$ and $x \to 0$. So GD does escape — but the escape time depends on how quickly the unstable eigenvector amplifies noise. SGD escapes faster because its noise pushes off the saddle at each step, and momentum-based methods are even better because they accumulate velocity along unstable directions.

### 4.4 Adam on a logistic regression toy problem

Binary classification: $x \in \mathbb{R}^2$, label $y \in \{0, 1\}$. Model $\sigma(w^\top x + b)$ where $\sigma$ is sigmoid. Cross-entropy loss. Two points: $(x_1, y_1) = ((1, 0), 0)$, $(x_2, y_2) = ((0, 1), 1)$. Initialize $w = (0, 0)$, $b = 0$, Adam defaults, $\eta = 0.1$.

Step 1 gradient on $(x_1, y_1)$: prediction $\sigma(0) = 0.5$. Error $0.5 - 0 = 0.5$. $\nabla_w \ell = 0.5 \cdot (1, 0) = (0.5, 0)$. $\nabla_b \ell = 0.5$.

Adam updates for $w$ coordinate 1: $m_1 = (1 - 0.9) \cdot 0.5 = 0.05$. $\hat m = 0.05 / (1 - 0.9) = 0.5$. $v_1 = (1 - 0.999) \cdot 0.25 = 2.5 \times 10^{-4}$. $\hat v = 2.5 \times 10^{-4} / (1 - 0.999) = 0.25$. Step: $-0.1 \cdot 0.5 / \sqrt{0.25} = -0.1$. So $w_1 \leftarrow -0.1$.

Observe the bias correction's role: without it, $m$ would have been 0.05 and $v$ would have been $2.5 \times 10^{-4}$, yielding a step of $-0.1 \cdot 0.05 / \sqrt{2.5\times 10^{-4}} = -0.316$, which is much too large on the $v$ side (the small $v$ comes from just starting up, not from the true gradient magnitude). Bias correction normalizes both by their expected asymptotic scale.

### 4.5 L-BFGS on logistic regression in a few lines

Logistic regression with $n = 100$, $d = 10$. Full gradient costs $O(nd) = 10^3$ ops. L-BFGS with $m = 10$:

- Compute $g_0 = \nabla \hat F(\theta_0)$.
- Approximate Newton direction $p_0 = -g_0$ (no history yet).
- Line search $\eta$ satisfying Wolfe conditions: picks say $\eta = 1$.
- Update $\theta_1 = \theta_0 + p_0$, compute $g_1$, store $s_0 = \theta_1 - \theta_0$ and $y_0 = g_1 - g_0$.
- Iteration 1: two-loop recursion on history of length 1 gives $p_1 \approx -\gamma_1 g_1$ where $\gamma_1$ is derived from $s_0, y_0$ — already better than vanilla GD.
- By iteration 5 or so, the approximation captures the curvature well and convergence is superlinear.

On convex problems with $d \sim 10^4$ and $n$ fitting in memory, L-BFGS typically converges in tens of iterations and dominates SGD. On non-convex problems with batches and augmentations, its advantage shrinks or reverses.

---

## 5. Relevance to Machine Learning Practice

### 5.1 What optimizer do I actually use?

**Short answer depends on problem type:**

- **Transformers / LLMs / most NLP:** AdamW is the overwhelming default. Typical hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.95$–$0.999$, $\eta$ peaking at $10^{-4}$ to $3 \times 10^{-4}$ with linear warmup + cosine decay, weight decay $0.1$.
- **Convolutional vision networks (ResNet-style, CIFAR/ImageNet):** SGD with momentum ($\beta = 0.9$) and a decaying step-size schedule is the classical choice and generalizes slightly better than Adam on clean-labeled, well-behaved data.
- **Diffusion models / generative models:** AdamW or Adam, often with $\beta_2 = 0.999$ and careful warmup.
- **Reinforcement learning:** Adam for neural policies; RMSProp historically used.
- **Small/deterministic problems (logistic regression, small GPs):** L-BFGS.
- **Recommenders with sparse features:** AdaGrad historically; now mostly Adam or Adam variants like LazyAdam.

### 5.2 Learning rate scheduling

A single step size rarely works. Common schedules:

- **Linear warmup** over ~1–10% of training: start near 0, ramp to peak. Critical for transformers; without it, the adaptive variance $v_t$ is initialized poorly and early updates destabilize training.
- **Cosine decay** from peak to 0: smooth decay that tends to land well without tuning endpoints. Default for LLMs.
- **Step decay** (e.g., divide by 10 every 30 epochs): classical CNN training.
- **One-cycle / super-convergence** (Smith): triangle up and down over training; can enable much larger peak learning rates.
- **Plateau / reduce-on-plateau**: halve $\eta$ when validation loss stops improving. Heuristic but effective.

### 5.3 Batch size, learning rate, and the linear scaling rule

Doubling the batch size halves the gradient variance. Under the approximation that optimization progress depends mainly on *signal-to-noise ratio*, this suggests you can double the learning rate when you double the batch size — the **linear scaling rule**. Works well up to a critical batch size (problem-dependent, often in the thousands for LLMs) beyond which gains saturate and you need to tune separately. The critical batch size is roughly $\sigma^2 / \|\nabla F\|^2$ — it's a function of how noisy your gradient is relative to its magnitude.

### 5.4 When adaptive methods fail (and what to do)

Adam and friends can underperform SGD on some vision benchmarks and, more dramatically, fail to generalize on noisy-label datasets. Hypotheses:

- Adam's effective step size is tied to the *direction* of the gradient, not magnitude; this may interact badly with label noise.
- Sharp minima found by Adam may generalize worse than flat minima found by SGD.
- Variance estimates $v_t$ in early training are unstable; bias correction is essential but can still drive oversized steps.

Common remedies: **warmup**, **gradient clipping**, switching to **SGD mid-training** (Adam-to-SGD papers), or using **AdamW + decoupled decay**.

### 5.5 Gradient clipping

When gradients blow up (common in RNNs, transformers at init, RL), clip the global norm:

$$
g \leftarrow g \cdot \min\!\left(1, \frac{c}{\|g\|}\right).
$$

Typical $c \in [0.1, 1.0]$. Stabilizes training at the cost of slight bias. Modern LLM training almost always uses clipping at $c = 1.0$.

### 5.6 Second-order methods in modern ML

Pure Newton is intractable at deep-learning scale, but Hessian-aware ideas return in several forms:

- **K-FAC** approximates the Fisher information matrix with a block-diagonal structure; competitive with Adam on some problems.
- **Shampoo** (distributed-Adagrad-with-structure) and **Sophia** (Hessian-diagonal estimation via a Hutchinson trick) are actively researched second-order methods for LLM training.
- **Natural gradient** methods replace the Euclidean metric with a Fisher metric.
- **Gauss–Newton** and **Levenberg–Marquardt** remain standard for non-linear least squares (classical ML, computer vision, physics-informed networks).

At inference / post-training time, **L-BFGS** is still used for fine-tuning small models, meta-learning outer loops, and convex components of ML pipelines.

### 5.7 Interaction with regularization

- **L2 regularization in SGD:** $\nabla(f + \tfrac{\lambda}{2}\|\theta\|^2) = \nabla f + \lambda \theta$. Gives update $\theta \leftarrow (1 - \eta \lambda) \theta - \eta \nabla f$. L2 = weight decay, for SGD.
- **L2 in Adam:** adding $\lambda \theta$ to the gradient gets divided by $\sqrt{\hat v}$ — effectively decays small-gradient parameters *less*. Not what you want. Use AdamW.
- **Dropout / data augmentation:** add noise to the gradient, effectively increasing $\sigma^2$. Adapt $\eta$ or batch size accordingly.

### 5.8 Evaluation and monitoring

Good optimizers surface problems quickly. Watch:

- **Train loss curve**: should decrease monotonically (on average). Sudden spikes suggest step size too large, or dying-ReLU / vanishing gradients.
- **Gradient norms**: should settle into a steady regime. Blow-ups mean clip or reduce $\eta$; collapses mean vanishing gradients or overfit.
- **Learning rate schedule vs. loss**: after a warmup phase, loss should track $\eta$ decreases.
- **Update norms vs. weight norms**: the ratio $\|\Delta \theta\| / \|\theta\|$ should be in a healthy range (often around $10^{-3}$); too small means learning has stalled, too large means instability.

### 5.9 Trade-offs at a glance

| Concern | GD | SGD / mini-batch | Momentum / NAG | Adam(W) | Newton | L-BFGS |
|---|---|---|---|---|---|---|
| Compute per step | $O(n)$ | $O(b)$ | $O(b)$ | $O(b)$ | $O(d^3)$ | $O(md)$ |
| Memory | $O(d)$ | $O(d)$ | $O(d)$ | $O(d)$ | $O(d^2)$ | $O(md)$ |
| Convergence, smooth convex | $O(1/k)$ | $O(1/\sqrt{k})$ | $O(1/k^2)$ | $O(1/\sqrt{k})$ | local quadratic | super-linear |
| Robust to ill-conditioning? | No | No | Somewhat | Yes (coord-wise) | Yes | Yes |
| Robust to noise? | N/A | Yes | Yes | Yes | No | No |
| Memory of past gradients? | No | No | Yes | Yes (squared) | No | Yes |
| Works out-of-box on deep nets? | No | Yes (w/ tuning) | Yes | Yes | No | Rarely |

---

## 6. Common Pitfalls & Misconceptions

### 6.1 "Adam is always better than SGD."

Adam often converges faster in training loss, but on many supervised problems SGD + momentum generalizes better. The right choice is problem-dependent. Default to SGD+momentum for CNN image classification, AdamW for transformers, and test both when uncertain.

### 6.2 "Gradient descent finds the global minimum."

Only for convex problems. For non-convex losses, GD finds *a* stationary point — at best a local minimum, and often a saddle (though usually SGD's noise escapes saddles). Neural networks happen to have many "good" local minima; this is why training works, not a guarantee of global optimality.

### 6.3 "Saddle points are local minima."

No. A saddle has $\nabla f = 0$ but indefinite Hessian. Plain GD can slow dramatically near saddles because gradients are tiny; momentum and stochastic noise help escape. In high dimensions, critical points are overwhelmingly saddles, not minima.

### 6.4 "I should use the largest batch size I can."

Larger batches reduce variance, but beyond the **critical batch size** you get diminishing returns per gradient step, and often worse generalization. A mini-batch size of 32–256 is typical for vision; much larger (millions of tokens) for LLMs, but with proportionally adjusted learning rates.

### 6.5 "Weight decay is the same as L2."

Only for SGD. For Adam, L2 adds $\lambda \theta$ to the gradient *before* the adaptive rescale, which couples decay to gradient magnitude and weakens it for parameters that already moved a lot. AdamW fixes this by applying decay after rescaling.

### 6.6 "Newton's method is always faster than gradient descent."

Locally, yes — quadratic convergence beats linear. Globally, Newton can fail: (1) $H$ may be indefinite, giving an ascent direction; (2) Newton step may be huge and overshoot; (3) $H^{-1}$ costs $O(d^3)$; (4) in deep learning, $d^2$ memory is prohibitive. Always apply damping or trust-region control.

### 6.7 "Setting the learning rate as high as possible trains fastest."

Up to a critical threshold, yes. Beyond it, the loss diverges or oscillates. There's a narrow band where training is stable and fast. Cyclic / one-cycle schedules try to exploit the upper end of this band temporarily.

### 6.8 "Loss stopped decreasing — I should stop training."

Might be a plateau, especially near saddles. Common responses: reduce learning rate, wait longer, add warm restart, switch optimizer. Blindly early-stopping on training loss is usually fine; on validation loss with high variance (small val set), you may stop too early.

### 6.9 "Momentum just adds past gradients."

Momentum is not just a smoothing filter; it creates a velocity variable that accumulates. In narrow valleys it accelerates; in chaotic regions it can overshoot. Nesterov's lookahead is the principled way to correct for this overshoot.

### 6.10 "Bias correction in Adam is a minor detail."

Without bias correction, the first several hundred updates have an effective step size far smaller (or larger, for $v$) than intended, distorting early training and sometimes causing divergence. It is not optional.

### 6.11 "Gradient clipping doesn't change the optimum."

It *can* — clipping is a biased operation. In practice the bias is small because clipping fires rarely at steady state. But if you clip aggressively, you are effectively constraining the implicit step direction and may converge to a different point.

### 6.12 "L-BFGS works for any problem."

L-BFGS needs fairly clean deterministic gradients (or very large consistent batches). With mini-batches, data augmentation, dropout, or batch norm, the secant equation is violated at every step and L-BFGS's approximate Hessian becomes garbage. Use SGD-family methods in those regimes.

### 6.13 "If validation is bad, I need a better optimizer."

Often not. If training loss is good but validation loss is bad, the problem is generalization, not optimization — more regularization, more data, or architectural changes. Better optimizers shine when training loss is stuck.

### 6.14 "Convex optimization is a solved problem."

It's solved in theory; practice is full of conditioning issues, constrained problems, non-smoothness, and scale. Convex optimization is still a rich field especially at the intersection of structured sparsity, online learning, and distributed systems.

### 6.15 "Convergence rate tells you runtime."

The $O(1/k)$ vs $O(1/k^2)$ distinction is about iteration count, not wall-clock time. Newton's "one step" is $O(d^3)$ of compute. A modern GPU does many SGD steps in the time Newton does one. The rate is a *per-iteration* statement and must be multiplied by per-iteration cost to get a meaningful comparison.

---

## 7. Interview Preparation

Questions are organized by difficulty: **Foundational** → **Mathematical** → **Applied** → **Debugging / failure mode** → **Probing / follow-ups**. Answers are written at the level expected of a strong ML / applied scientist candidate.

### 7.1 Foundational questions

**Q1. What is the gradient and why do we use the negative gradient direction in gradient descent?**

The gradient $\nabla f(\theta)$ is the vector of partial derivatives. It points in the direction of steepest *ascent* of $f$. Its negative is therefore the direction of steepest *descent*: locally, moving by $-\eta \nabla f$ decreases $f$ fastest per unit displacement for small $\eta$. This follows from the first-order Taylor expansion $f(\theta + p) \approx f(\theta) + \nabla f(\theta)^\top p$; among unit vectors $p$, $p = -\nabla f / \|\nabla f\|$ minimizes $\nabla f^\top p$.

**Q2. Define a convex function. Why is convexity such a useful property in optimization?**

A function is convex if its epigraph (set above the graph) is convex — equivalently, $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ for all $\lambda \in [0,1]$. Usefulness: (1) every local minimum is global, (2) stationary points are global minima, (3) standard algorithms have provable polynomial convergence rates, (4) duality theory provides efficient algorithms and tight bounds. For *strongly* convex and smooth functions, linear convergence is achievable.

**Q3. Difference between batch, mini-batch, and stochastic gradient descent?**

Batch GD uses the full dataset gradient per step — expensive but noise-free. SGD uses one sample — cheap but noisy. Mini-batch uses $b$ samples, trading off: variance scales as $\sigma^2 / b$, compute scales as $b$. Mini-batch is the practical default because it exploits GPU parallelism, gives reasonable variance, and empirically tends to generalize at least as well as batch.

**Q4. What is a saddle point? Why do they matter for neural-network training?**

A critical point where the Hessian has both positive and negative eigenvalues — some directions curve up, others down. In high dimensions, critical points of random smooth functions are overwhelmingly saddles (all $d$ eigenvalues being non-negative becomes exponentially unlikely). Near saddles, gradients are small, so plain GD stalls. Noise (SGD) and momentum-based methods help escape along unstable eigen-directions.

**Q5. Why do we use momentum? What problem does it solve?**

Momentum accumulates past gradients with exponential decay, smoothing the update. It helps in two ways: (1) In narrow valleys, oscillating gradients along the steep axis cancel while aligned gradients along the flat axis accumulate, producing an effective per-direction step. (2) Near saddles or flat regions, velocity carries the iterate across short barriers. Nesterov's variant further improves convergence by evaluating the gradient at a lookahead point.

**Q6. What's the difference between AdaGrad, RMSProp, and Adam?**

- **AdaGrad** divides the step by $\sqrt{\sum_t g_t^2}$ — monotonically decreasing step sizes. Works on sparse problems but halts on long runs.
- **RMSProp** replaces the sum with an exponential moving average, so step sizes adapt to recent gradient scale without vanishing.
- **Adam** = RMSProp + momentum + bias correction. Tracks both first and second moments with EMAs; bias-corrects for the $m_0 = v_0 = 0$ initialization.

**Q7. Why does Adam use bias correction?**

At step $t$, $m_t = (1 - \beta_1)\sum_{k \le t} \beta_1^{t-k} g_k$. Under a stationary gradient distribution with mean $\mu$, $\mathbb{E}[m_t] = (1 - \beta_1^t) \mu$. Early on ($t$ small), $m_t$ underestimates $\mu$ by factor $(1 - \beta_1^t)$. Dividing by this factor gives an unbiased estimate $\hat m_t$. Without correction, the first few hundred steps have artificially small updates (for $m$) or inflated step sizes (for $v$), which destabilizes training.

**Q8. What's the difference between Adam and AdamW?**

In Adam with L2 regularization, $\lambda \theta$ is added to the gradient *before* dividing by $\sqrt{\hat v_t}$ — so parameters with large $v$ get *less* decay, which is the opposite of what regularization should do. AdamW applies weight decay *after* the adaptive step: $\theta \leftarrow \theta - \eta (\hat m / \sqrt{\hat v} + \lambda \theta)$. This decouples decay from gradient-based updates and empirically improves generalization. AdamW is the current default for transformers.

**Q9. Why is Newton's method rarely used in deep learning?**

Three reasons: (1) Storing the Hessian needs $O(d^2)$ memory, which is $10^{18}$ for modern LLMs. (2) Solving $H p = -\nabla f$ is $O(d^3)$ compute per step. (3) In non-convex regions $H$ may be indefinite, so $-H^{-1}\nabla f$ can point uphill. Practical large-scale optimization relies on first-order methods or structured approximations (K-FAC, Shampoo, Sophia).

**Q10. What does "$O(1/k)$ convergence rate" actually mean?**

After $k$ iterations, the optimality gap $f(\theta_k) - f(\theta^\star)$ (or distance to the optimum) is bounded by a constant over $k$. So to reach $\epsilon$ error you need $k = O(1/\epsilon)$ iterations. Compare: $O(1/\sqrt k)$ needs $O(1/\epsilon^2)$, and linear $O(\rho^k)$ needs only $O(\log(1/\epsilon))$.

**Q11. What does "Lipschitz gradient" mean and why do we need it?**

$\nabla f$ is $L$-Lipschitz if $\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$. Equivalently, $\|\nabla^2 f\| \le L$ when $f$ is twice differentiable. This bounds curvature from above and ensures that $f$ doesn't wiggle too fast, which is what lets us pick a safe step size $\eta \le 1/L$.

**Q12. What is the condition number and why does it matter?**

For a strongly convex smooth function with parameters $\mu$ (strong convexity) and $L$ (smoothness), $\kappa = L/\mu \ge 1$. Large $\kappa$ means the loss surface is stretched — some directions curve sharply, others are flat. GD's convergence rate is $(1 - 1/\kappa)^k$, so high $\kappa$ means slow convergence. Nesterov improves this to $(1 - 1/\sqrt{\kappa})^k$. Adaptive methods try to automatically precondition and effectively reduce $\kappa$ coordinate-wise.

### 7.2 Mathematical questions

**Q13. Derive the gradient descent step size that maximizes decrease per step on a smooth convex function.**

From the smoothness inequality (descent lemma):

$$
f(x - \eta \nabla f(x)) \le f(x) - \eta \|\nabla f\|^2 + \tfrac{L\eta^2}{2}\|\nabla f\|^2.
$$

Minimize RHS over $\eta$: derivative w.r.t. $\eta$ is $-\|\nabla f\|^2 + L\eta\|\nabla f\|^2 = 0$, giving $\eta^\star = 1/L$. Substituting,

$$
f(x_{t+1}) \le f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2.
$$

This is the tightest worst-case guarantee with constant step size.

**Q14. Derive the $O(1/k)$ rate for gradient descent on smooth convex functions.**

By the descent lemma with $\eta = 1/L$:

$$
f(x_{t+1}) \le f(x_t) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2. \tag{1}
$$

Convexity gives $f(x^\star) \ge f(x_t) + \nabla f(x_t)^\top (x^\star - x_t)$, so

$$
f(x_t) - f(x^\star) \le \nabla f(x_t)^\top (x_t - x^\star) \le \|\nabla f(x_t)\|\|x_t - x^\star\|. \tag{2}
$$

A cleaner path: by the three-point / co-coercivity lemma,

$$
\|x_{t+1} - x^\star\|^2 \le \|x_t - x^\star\|^2 - \tfrac{2}{L}(f(x_t) - f(x^\star)).
$$

Summing $t = 0, \dots, k-1$ and using $f(x_{t+1}) \le f(x_t)$:

$$
\tfrac{2k}{L}(f(x_k) - f(x^\star)) \le \|x_0 - x^\star\|^2,
$$

giving $f(x_k) - f(x^\star) \le \tfrac{L \|x_0 - x^\star\|^2}{2k} = O(1/k)$.

**Q15. Derive the linear rate under strong convexity + smoothness.**

Strong convexity gives the Polyak–Łojasiewicz inequality $\|\nabla f(x)\|^2 \ge 2\mu(f(x) - f(x^\star))$. Combined with (1):

$$
f(x_{t+1}) - f(x^\star) \le f(x_t) - f(x^\star) - \tfrac{1}{2L}\|\nabla f(x_t)\|^2 \le (1 - \tfrac{\mu}{L})(f(x_t) - f(x^\star)).
$$

Iterating: $f(x_k) - f(x^\star) \le (1 - \mu/L)^k (f(x_0) - f(x^\star))$ — linear / geometric.

**Q16. Derive Newton's method from the quadratic Taylor expansion.**

At iterate $x_t$, the second-order Taylor approximation is

$$
m_t(p) = f(x_t) + \nabla f(x_t)^\top p + \tfrac{1}{2} p^\top H_t p.
$$

Minimizing: $\nabla_p m_t = \nabla f(x_t) + H_t p = 0 \Rightarrow p = -H_t^{-1} \nabla f(x_t)$. Update: $x_{t+1} = x_t - H_t^{-1} \nabla f(x_t)$. For $H_t \succ 0$, $p$ is a descent direction and minimizes the local quadratic model exactly.

**Q17. Show Newton's method is quadratically convergent (local).**

Assume $f$ is $C^2$, $H(x^\star) \succ 0$, and $H$ is Lipschitz with constant $M$. Near $x^\star$:

$$
\nabla f(x_t) = \nabla f(x^\star) + H(x^\star)(x_t - x^\star) + O(\|x_t - x^\star\|^2),
$$

and $\nabla f(x^\star) = 0$. Then

$$
x_{t+1} - x^\star = x_t - x^\star - H(x_t)^{-1} \nabla f(x_t).
$$

With $H(x_t)^{-1} = H(x^\star)^{-1} + O(\|x_t - x^\star\|)$, the linear term cancels and the remainder is $O(\|x_t - x^\star\|^2)$. Hence $\|x_{t+1} - x^\star\| \le C\|x_t - x^\star\|^2$ for some constant $C$ — quadratic convergence.

**Q18. Show that on a quadratic $f(x) = \tfrac12 x^\top A x - b^\top x$, Newton converges in one step.**

$\nabla f = Ax - b$, $H = A$. $p = -A^{-1}(Ax - b) = -x + A^{-1}b$. So $x_1 = x_0 + p = A^{-1}b = x^\star$. This is the exact minimizer, regardless of $x_0$.

**Q19. Derive the Nesterov acceleration rate intuition.**

Nesterov's method can be analyzed via a clever Lyapunov / estimate-sequence argument. One way to see why $O(1/k^2)$ is achievable: construct a sequence $\phi_k(x) \ge f(x)$ that is tight at the optimum and converges to $f(x^\star)$ at rate $O(1/k^2)$. The key is the momentum coefficient $\beta_k = (k-1)/(k+2)$ (or similar) which creates a *telescoping* contribution — each step's progress is the *difference* of a decreasing quantity, not a constant fraction. In simpler terms: Nesterov mixes iterates across multiple time scales in a way that cancels accumulated error.

Matching Nemirovski's lower bound shows this rate is *optimal* among gradient methods for smooth convex problems — you cannot do better with only gradient access.

**Q20. Derive the BFGS update from the secant equation.**

We want $B_{t+1} s_t = y_t$ where $s_t = x_{t+1} - x_t$, $y_t = \nabla f(x_{t+1}) - \nabla f(x_t)$. Among all symmetric PD matrices satisfying this, BFGS picks the one minimizing a specific weighted Frobenius distance from $B_t$:

$$
B_{t+1} = \arg\min_{B : B = B^\top,\, Bs = y} \|B - B_t\|_W,
$$

where $W$ is the inverse-Hessian-weighted norm. Solving gives the rank-two update

$$
B_{t+1} = B_t - \frac{B_t s_t s_t^\top B_t}{s_t^\top B_t s_t} + \frac{y_t y_t^\top}{y_t^\top s_t}.
$$

Applying Sherman–Morrison twice yields the inverse-Hessian form (what you'd actually implement):

$$
H_{t+1} = (I - \rho_t s_t y_t^\top) H_t (I - \rho_t y_t s_t^\top) + \rho_t s_t s_t^\top, \; \rho_t = 1/(y_t^\top s_t).
$$

**Q21. Why does L-BFGS need only $O(md)$ memory?**

L-BFGS never forms $H_t$ explicitly. It stores the last $m$ pairs $(s_i, y_i)$, each $d$-dimensional. The product $H_t g$ for any vector $g$ can be computed by the two-loop recursion using only these pairs plus an initial diagonal scaling — never materializing $H_t$. Total memory: $2md + O(d)$.

**Q22. Why does SGD require decaying step sizes?**

The stochastic descent inequality has a noise term $\tfrac{L \eta^2}{2} \sigma^2$. With constant $\eta$, the error plateaus at $O(\eta \sigma^2)$ — the iterate bounces around the optimum. To drive error to zero, $\eta_t \to 0$ is necessary, but $\sum_t \eta_t = \infty$ must hold to keep making progress (the Robbins–Monro conditions). $\eta_t = c/\sqrt{t}$ is the classical optimal trade-off for convex non-strongly-convex problems, yielding $O(1/\sqrt k)$.

**Q23. Derive variance of a mini-batch gradient.**

$g_B = \tfrac{1}{b}\sum_{i \in B} \nabla \ell_i$. If $B$ is a uniform sample without replacement from $n$ points and per-sample gradient variance is $\sigma^2$:

$$
\text{Var}(g_B) = \tfrac{\sigma^2}{b} \cdot \tfrac{n - b}{n - 1}.
$$

The second factor (finite-population correction) is $\approx 1$ when $b \ll n$. With replacement (i.i.d.), $\text{Var}(g_B) = \sigma^2 / b$. Hence the linear scaling rule: doubling $b$ lets you roughly double $\eta$ while keeping variance-to-signal ratio constant.

**Q24. Show that SVRG has unbiased gradient estimates with reduced variance.**

$g_t = \nabla f_i(x_t) - \nabla f_i(\tilde x) + \tilde \mu$ where $\tilde \mu = \nabla \hat F(\tilde x)$. Unbiasedness: $\mathbb{E}_i[g_t] = \nabla \hat F(x_t) - \nabla \hat F(\tilde x) + \tilde \mu = \nabla \hat F(x_t)$. Variance reduction: as $x_t \to \tilde x$, the pair $\nabla f_i(x_t) - \nabla f_i(\tilde x) \to 0$, so $g_t \to \tilde \mu = \nabla \hat F(\tilde x) \approx \nabla \hat F(x_t)$ with near-zero variance. Under smoothness, $\text{Var}(g_t) \le L^2 \|x_t - \tilde x\|^2 \to 0$ as the algorithm converges.

**Q25. When is $-H^{-1} \nabla f$ *not* a descent direction?**

When $H$ is not positive definite. If $H$ has a negative eigenvalue $\lambda < 0$ with eigenvector $v$, then $v^\top H^{-1} v = 1/\lambda < 0$, so the "Newton step" can align with the gradient rather than against it. Standard remedies: damping ($H \to H + \tau I$), trust region, modified Cholesky, or Gauss–Newton (which replaces $H$ by $J^\top J \succeq 0$).

### 7.3 Applied questions

**Q26. You're training a 7B-parameter LLM. Which optimizer do you choose and why? What are the hyperparameters?**

AdamW is the default. Reasons: (1) adaptive per-coordinate scaling handles the wide variety of gradient magnitudes across layer norms, embedding, and attention parameters; (2) decoupled weight decay improves generalization and regularizes the embedding matrix sensibly; (3) momentum smooths noisy gradients from small per-GPU batches. Typical hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, peak LR $\sim 3 \times 10^{-4}$, weight decay $0.1$, linear warmup for 2000 steps followed by cosine decay to 10% of peak, gradient clip at $1.0$. Mixed-precision (bf16) for the forward/backward, but optimizer states in fp32 — critical because the second moment $v_t$ is often tiny.

**Q27. You're training a ResNet-50 on ImageNet. Why might SGD with momentum outperform Adam?**

Empirically, SGD+momentum generalizes better on clean image classification benchmarks. Hypotheses: (1) Adam's adaptive step sizes bias toward sharp minima that generalize worse; (2) SGD's isotropic noise matches the structure of CNN loss surfaces better than Adam's anisotropic noise; (3) the implicit regularization of constant-rate SGD is well-tuned for the task. Typical schedule: LR $0.1$, momentum $0.9$, weight decay $10^{-4}$, step-decay (×0.1 at epochs 30, 60, 80), batch size 256.

**Q28. You halve your batch size. How do you adjust the learning rate?**

Under the linear scaling rule, halve the LR. Rationale: variance of the mini-batch gradient roughly doubles, so to keep the signal-to-noise ratio constant you reduce step size. Caveat: at very small batches, warmup becomes more important; at very large batches, linear scaling breaks (you've hit the critical batch size) and you may need separate tuning.

**Q29. Your loss diverges after a few hundred steps with Adam. What do you check?**

Ordered list of likely causes: (1) LR too high — try halving. (2) No warmup — $v_t$ is unstable early; add linear warmup. (3) Gradients blowing up — add gradient clipping at norm 1. (4) Mixed-precision underflow in $v_t$ — keep optimizer states in fp32. (5) Bad initialization producing huge initial gradients. (6) Data bug — NaN or extreme values in inputs. (7) Loss formulation error (e.g., sigmoid on a logit rather than BCE-with-logits producing numerical instability).

**Q30. You have a well-behaved convex problem with 10k parameters and 1M data points. Which optimizer?**

L-BFGS. Reasons: (1) Convex and smooth → superlinear convergence is achievable. (2) $d = 10^4$, $m = 10$ → $O(md) = 10^5$ memory — trivial. (3) Deterministic full-batch gradient cost is manageable. (4) Very few iterations needed. scikit-learn's `LogisticRegression` uses L-BFGS by default for exactly this reason.

**Q31. When would you use a second-order method in deep learning?**

Rarely, but: (1) small networks where $d^2$ is affordable (physics-informed NNs, meta-learning inner loops); (2) fine-tuning the last few layers of a large model; (3) MAML and variants where the outer loop is second-order by design; (4) approximate second-order methods like K-FAC, Shampoo, or Sophia that exploit structure to be feasible at scale. For full pretraining of LLMs, first-order AdamW dominates.

**Q32. You're training on embeddings with hundreds of thousands of rare tokens. Which optimizer?**

AdaGrad or Adam with care. Rare-token embeddings receive sparse gradients; AdaGrad's per-coordinate scaling is ideal because it increases effective step size for rarely-updated coordinates. In PyTorch, use `SparseAdam` or `Adagrad` for embedding layers and AdamW for dense layers. Or use a single optimizer but enable sparse gradient support.

**Q33. How do you choose an initial learning rate?**

Two common approaches: (1) **LR range test** (Smith): linearly increase LR from tiny ($10^{-8}$) to huge ($1$), plot loss, pick slightly below where loss starts diverging. (2) **Orders of magnitude scan**: try $10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}$ for a few epochs, pick whichever converges fastest without diverging. For well-known architectures, start from the paper's recipe.

**Q34. You're fine-tuning a pre-trained model. Should the learning rate be higher or lower than pre-training?**

Lower, typically 10x–100x smaller. Pre-trained weights are already near a good minimum; large steps risk destroying that structure. Common recipe: small LR for the backbone, 10x larger LR for the newly-added head. Even smaller for parameter-efficient fine-tuning (LoRA, etc.). For RLHF or instruction tuning, LR on the order of $10^{-6}$–$10^{-5}$ is common.

**Q35. Why do we warmup the learning rate for transformers?**

At initialization, adaptive second-moment estimates $v_t$ are essentially zero, so the effective step size $\eta / \sqrt{v_t}$ can be huge — leading to divergent first updates. Warmup ramps $\eta$ from near-zero to its peak over a few thousand steps, giving $v_t$ time to stabilize. Without warmup, transformers frequently diverge in the first few hundred steps with default AdamW settings.

**Q36. What does gradient clipping do, and when do you use it?**

Rescales the gradient to have norm at most $c$: $g \leftarrow g \cdot \min(1, c/\|g\|)$. Use when gradients occasionally explode — RNNs, transformers at init, RL, ill-conditioned losses. It is a biased operation but, in practice, clipping fires only on occasional outliers, so the bias is small. Typical $c = 1.0$ for LLMs.

**Q37. When is full-batch gradient descent the right choice?**

When the dataset is small enough to fit in memory (say $n < 10^5$) and the loss is convex or very benign. Examples: logistic regression on tabular data, simple least squares, physics / inverse problems, small Gaussian processes. In these cases full GD (or L-BFGS) converges cleanly without the noise of SGD, and the per-iteration cost is tolerable.

**Q38. Your model trains well on one GPU but diverges when you scale to 64 GPUs. Why?**

Almost certainly a learning-rate / batch-size issue. Data-parallel training with 64 GPUs has a 64x larger effective batch. If you didn't adjust the LR, you're now under-stepping. Under the linear scaling rule, scale LR by 64 (or more carefully). But large-batch regimes beyond the critical batch size also need stronger warmup, sometimes larger weight decay, and layer-wise adaptive rate scaling (LARS/LAMB) for very large batches.

### 7.4 Debugging & failure-mode questions

**Q39. Training loss oscillates wildly without converging. What's happening?**

Classic "LR too high" signature. Each step overshoots the optimum, reverses direction, overshoots again. Verify by halving the LR — if the oscillation shrinks, that was the problem. Secondary suspects: batch size too small (noise dominates signal), or adaptive optimizer with unstable $v_t$ (add warmup / clipping).

**Q40. Loss decreases for 10 epochs then suddenly jumps. Why?**

Several possibilities: (1) NaN gradient propagated somewhere (unstable sigmoid/exp/log). (2) A rare example in the batch with enormous gradient (outlier). (3) Learning-rate warmup wrapping around or a schedule bug. (4) Numerical instability in mixed precision (bf16 is usually fine, fp16 can overflow in $v_t$). Check for NaN/Inf, gradient norm spikes, and data preprocessing anomalies.

**Q41. Your network's training loss plateaus at a high value from the start. Diagnosis?**

Common causes in priority order: (1) Dead activations (all-zero ReLUs) — check activation statistics. (2) Learning rate too small — gradient vanishing. (3) Vanishing gradients from initialization (pre-residual deep nets without proper init). (4) Data issue — labels misaligned, features normalized wrong. (5) Loss function bug (e.g., cross-entropy with already-softmaxed inputs). Inspect gradient norms and per-layer activations.

**Q42. You used Adam and got worse test accuracy than SGD+momentum with the same architecture. Why?**

Several well-known hypotheses: (1) Adam tends to find sharper minima that don't generalize as well; SGD's noise finds flatter ones. (2) L2 regularization is weaker in Adam than SGD; switch to AdamW. (3) Adam's effective step size is larger along directions with small gradient history, which may hurt implicit regularization. Try AdamW with decoupled weight decay, or a two-phase schedule (Adam early, SGD late).

**Q43. Your L-BFGS optimizer stops making progress on a deep-learning problem. Why?**

L-BFGS requires consistent gradients for its secant equation $y_t \approx H s_t$. In DL, stochastic mini-batches, dropout, batch norm running statistics, data augmentation — all violate this. The quasi-Hessian approximation becomes noise. Fix: use very large batches (full-batch if feasible) and disable dropout / use eval-mode batch-norm during L-BFGS. Or just switch to SGD/AdamW.

**Q44. You increased the batch size 10x to speed up training but now your model generalizes worse. Why?**

Large batches produce low-noise gradients that converge to sharper minima. Small-batch SGD's noise acts as an implicit regularizer, pushing into flatter minima that generalize better. Fix: increase LR proportionally (but you may already be past the critical batch size), add more explicit regularization (weight decay, dropout, label smoothing, longer training).

**Q45. Gradients are fine but weights barely change. What's going on?**

Likely an effective-step-size issue. In AdamW, the actual step is $-\eta \hat m / \sqrt{\hat v}$. If $\hat v$ is very large (recent gradients were big but now small), the step is tiny even if $\hat m$ is OK. Check $\|\Delta \theta\| / \|\theta\|$ per layer — should be around $10^{-3}$ in healthy training. Specific fixes: raise LR, reduce $\beta_2$ so $v_t$ forgets old large gradients faster.

**Q46. Validation loss starts increasing while training loss continues to drop. What optimizer change might help?**

This is overfitting, not an optimizer problem. But: (1) add/increase weight decay; (2) use AdamW instead of Adam if you weren't; (3) reduce LR to slow training; (4) early stop on validation. Note: switching optimizers alone usually doesn't fix overfitting; regularization and data are the levers.

**Q47. When fine-tuning a transformer, loss spikes at the start of training. Why?**

Adam's $v_t$ at the start of fine-tuning is either zero (fresh optimizer state) or carries over from pre-training (if you kept it). Either way, combined with a new task distribution, the first few updates can be huge. Fixes: linear warmup (even short), gradient clipping, or freezing the backbone for initial steps.

**Q48. Your gradient norm is 100× smaller than expected. Debug.**

Possibilities: (1) Gradient vanishing through many layers (check per-layer gradient norms; bad activations, residual path broken). (2) Weights too small (under-initialized); inputs / activations too small. (3) Wrong loss scale (e.g., averaging over batch when you should sum or vice versa). (4) BatchNorm / LayerNorm squishing activations too aggressively. (5) Mixed precision losing gradient magnitude — use loss scaling or bf16.

**Q49. You switched from SGD to Adam and training became unstable. Why?**

Adam's effective per-coordinate step size is tuned by $v_t$. At the very start, $v_t$ is tiny, so the step is huge — even with bias correction, the first few steps can overshoot if the initialization is near-boundary. Also, if you kept the SGD learning rate, it's wildly too large for Adam (Adam's effective step is gradient-normalized; SGD's is not). Reduce LR by 10×–100× and add warmup.

**Q50. Training loss is stuck at exactly the value of a random-guess baseline. What's happening?**

Almost always a model / data bug rather than an optimizer issue. Check: (1) Is the loss connected to the parameters (a detached tensor somewhere)? (2) Is the prediction exactly the same for every input (dead model)? (3) Are labels correctly aligned with inputs? (4) Is the last layer's bias causing all predictions to equal the majority class? (5) Are your logits passing through sigmoid twice?

### 7.5 Probing & follow-up questions

**Q51. "Why exactly does Nesterov get $O(1/k^2)$ while momentum only gets $O(1/k)$?"**

The deep answer uses estimate sequences. Nesterov constructs iterates that minimize a sequence of upper bounds $\phi_k(x) \ge f(x)$ whose minima converge to $f(x^\star)$ at rate $O(1/k^2)$. Classical momentum lacks this guarantee because it doesn't use the lookahead. Intuitively: Nesterov's lookahead at $\tilde \theta = \theta + \beta v$ corrects for the overshoot that momentum would otherwise incur, giving a first-order approximation of curvature for free. This matches the theoretical lower bound for first-order methods on smooth convex problems.

**Q52. "Under what assumptions can SGD converge to a global minimum of a non-convex function?"**

A few strong settings: (1) **Over-parameterized deep networks under NTK regime**: the loss behaves effectively convex in function space. (2) **PL inequality** — $\|\nabla f\|^2 \ge 2\mu(f - f^\star)$ holds globally; implies any first-order method converges linearly even without convexity. (3) **Saddle-escape results** (Jin et al.) — perturbed SGD escapes saddles in polynomial time and converges to an approximate second-order stationary point. (4) **Langevin dynamics / SGLD** — with appropriate noise schedule, converges to the global optimum in the limit via an MCMC-like argument, though very slowly.

**Q53. "Why does momentum's coefficient $\beta$ typically equal $0.9$ but not, say, $0.5$?"**

The effective averaging horizon is $1/(1-\beta)$ steps. $\beta = 0.9$ gives ~10 steps of memory; $\beta = 0.99$ gives 100. Too small and you lose momentum benefits; too large and you're too sluggish to respond to changes in gradient direction. The "sweet spot" of 0.9 is empirical but corresponds to horizons that match typical loss-curvature timescales. Nesterov analyses give optimal $\beta = (\sqrt{L} - \sqrt{\mu}) / (\sqrt{L} + \sqrt{\mu})$ for strongly convex smooth problems, which approaches 1 as $\kappa \to \infty$.

**Q54. "If Adam is so good, why does the original paper's convergence proof have a known bug?"**

Reddi et al. (2018, "On the Convergence of Adam and Beyond") showed that Adam's convergence proof is flawed: there exist convex problems where Adam does not converge. The issue: $v_t$ can *decrease* when a large-but-rare gradient is forgotten, so the step size can grow arbitrarily and diverge. AMSGrad fixes this by using $\hat v_t = \max(\hat v_{t-1}, v_t)$, guaranteeing monotone non-decreasing denominators. In practice, Adam works fine for DL, but this shows theory and practice can diverge.

**Q55. "What is Polyak averaging and why does it help?"**

Return an average $\bar \theta = \tfrac{1}{T}\sum_t \theta_t$ (or an EMA) instead of the last iterate $\theta_T$. For stochastic convex optimization, $\bar \theta$ achieves better convergence guarantees — the variance of the average is smaller than that of any single iterate. In deep learning, a related idea is **exponential moving average** (EMA) of weights, common in generative models and modern LLM training. The EMA weights often generalize better because they smooth out late-stage SGD noise.

**Q56. "What's the difference between 'stationary point' and 'critical point'? What about 'second-order stationary'?"**

These are often used interchangeably. **Critical / first-order stationary**: $\nabla f = 0$. **Second-order stationary**: $\nabla f = 0$ AND $\nabla^2 f \succeq 0$ (no negative eigenvalues). The second-order condition rules out saddles. Modern non-convex optimization theory focuses on algorithms that provably converge to approximate second-order stationary points, because that's the best you can hope for without special structure.

**Q57. "If I could compute the exact Hessian cheaply, would Newton's method always beat Adam?"**

In terms of iterations, yes — quadratic convergence dominates. But (1) near saddle points, Newton marches toward them, not away; you need cubic regularization or trust-region methods. (2) In stochastic settings, even a cheap Hessian still suffers from variance; you'd need variance reduction on the Hessian too. (3) The Hessian changes during training (non-stationary loss surface); naive Newton can jump too far. (4) In deep networks with symmetries / degeneracies, $H$ is near-singular everywhere, so $H^{-1}$ is numerically problematic. Some form of damping / regularization is essential.

**Q58. "Why doesn't random restart work in deep learning?"**

Because deep nets have so many parameters that random restarts don't sample meaningfully different regions of the landscape. Two training runs from different inits typically reach different-but-equally-good minima (related by symmetries of the model). Random restart is useful for small, highly non-convex problems (e.g., Gaussian mixture fitting); for neural nets, focus goes into better inits (Kaiming, Xavier) rather than multi-start.

**Q59. "Explain the implicit bias of gradient descent."**

For separable logistic regression, GD's iterates converge to the max-margin classifier *even though the loss has no finite minimum* (it decreases indefinitely toward zero). The direction of $\theta_t$ converges to the L2 max-margin solution. This is called implicit regularization — the algorithm, not the loss, picks a particular minimum. Related results hold for deep linear networks, where GD prefers low-rank solutions. This is part of why deep nets generalize well despite being wildly over-parameterized.

**Q60. "When would you prefer a trust-region method over a line-search method?"**

Trust-region methods fix a *step size* (the trust radius) and find the best direction within that ball; line-search methods fix a *direction* (e.g., Newton) and find the best step. Trust-region is preferred when (1) the Hessian is indefinite — you don't need to modify it, just restrict the step to where the quadratic model is reliable. (2) The quadratic model is a poor fit globally but good locally. (3) You want principled handling of degenerate cases. Levenberg–Marquardt is essentially a trust-region method for non-linear least squares.

**Q61. "How do distributed data-parallel training dynamics differ from single-GPU training?"**

Data parallel = average gradients across workers = effectively larger batch. Everything about the linear scaling rule applies. Additional wrinkles: (1) Communication cost — AllReduce of gradients is a bottleneck; optimizations include gradient compression, bucketing, overlap with compute. (2) Synchronous vs asynchronous — sync is standard; async introduces gradient staleness that acts like additional noise. (3) Optimizer states — in ZeRO / FSDP, optimizer states are sharded across GPUs to save memory; this changes the communication pattern but not the math.

**Q62. "What is a sharpness-aware minimization (SAM) optimizer and why does it work?"**

SAM minimizes a worst-case perturbed loss: $\min_\theta \max_{\|\delta\| \le \rho} f(\theta + \delta)$. Practically, it computes $\delta^\star = \rho \nabla f / \|\nabla f\|$, then takes a gradient step at $\theta + \delta^\star$. This biases toward *flat* minima (where perturbations don't increase loss much). Flat minima are believed to generalize better. SAM and its variants (ASAM, ESAM) often beat AdamW on generalization at the cost of 2× gradient compute.

**Q63. "How does the learning rate schedule interact with weight decay?"**

In AdamW, the effective regularization per step is $\eta \lambda \theta$. If $\eta$ decays, weight decay also effectively decays. Some recipes compensate by increasing $\lambda$ as $\eta$ decreases. In some implementations the decay is $\lambda \theta$ (LR-independent); in others, it's folded into the gradient (LR-dependent). Always check which your framework uses. The choice can change effective regularization by orders of magnitude over a schedule.

**Q64. "Is the bias-correction in Adam still needed if training is very long?"**

Mathematically, the correction factor $1 - \beta_1^t \to 1$ exponentially, so after a few thousand steps it's essentially 1 and the correction is trivially-satisfied. But it's essential in the first ~1000 steps for $\beta_1 = 0.9$ (and $\sim 10000$ for $\beta_2 = 0.999$). For warmup-stable training, bias correction is harmless even at steady state.

**Q65. "What breaks in optimizer theory when batch normalization is present?"**

Batch norm makes the loss scale-invariant in the weights of the preceding linear layer (scaling weights by $c$ doesn't change the output). This means: (1) the loss surface has a zero-eigenvalue direction (the scale), violating strong convexity assumptions; (2) the effective learning rate is weight-norm-dependent, making LR schedules behave non-linearly; (3) weight decay has a more subtle effect — it implicitly controls effective LR by shrinking weights. These dynamics are why BN-networks are so forgiving of LR choice but also why they have subtle pathological failure modes.

**Q66. "In a mixture of optimizers (different LR for different layers), what's the theoretical justification?"**

No unified theory, but several partial justifications: (1) Each layer has different gradient statistics; uniform LR is unavoidably suboptimal across them. (2) In transfer learning, pre-trained layers need smaller steps than new layers. (3) **LARS/LAMB** normalize by layer weight norm, providing scale-invariance that helps with very large batches. (4) **Layer-wise adaptive learning rates** can be seen as a block-diagonal approximation to natural gradient / K-FAC. Practical recipes for fine-tuning large models often use 3–10 LR groups.

---

## 8. Closing notes

This guide covered the backbone of optimization used in modern ML: convex and non-convex landscape structure; first-order methods from plain GD through the full adaptive family; second-order methods from Newton through L-BFGS; convergence-rate theory connecting all of them; and the stochastic / variance-reduction layer that makes it all work on massive data.

A few meta-points worth carrying into an interview:

1. **The algorithm is always solving the Taylor expansion.** First-order methods truncate at linear; second-order at quadratic; quasi-Newton reconstruct curvature from history. Everything else is choice of step size and how to handle noise.

2. **Convergence rates are about iterations, not time.** A "faster-converging" algorithm that costs 1000x more per step may be slower in wall-clock time. Always reason jointly about rate × per-iteration cost × parallelizability.

3. **In practice, the optimizer is never the main bottleneck to generalization** — data, architecture, and regularization are. The optimizer matters for *trainability* and *speed*, and occasionally for generalization at the margin (AdamW vs Adam, SGD vs Adam on CNNs).

4. **Theory and practice often diverge** — Adam's proof is broken yet Adam works. L-BFGS is superlinear in theory yet unreliable on stochastic deep networks. Use theory to build intuition and default recipes, but verify empirically.

5. **The "right" optimizer depends on the problem.** AdamW for transformers; SGD+momentum for classical vision; L-BFGS for small convex problems; specialized second-order methods for specific domains. When in doubt, start with AdamW defaults, add warmup and clipping, tune the peak learning rate, and iterate.

Good luck.
