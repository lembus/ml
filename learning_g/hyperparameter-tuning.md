# Hyperparameter Tuning

## 1. Motivation & Intuition

### Why This Topic Exists
In Machine Learning (ML), models are rarely "ready out of the box." They come with configuration settings—called **hyperparameters**—that govern the learning process itself. Unlike the internal weights of a neural network (which are learned from data), hyperparameters must be set *before* training begins.

Imagine you are trying to tune a complex radio to catch a clear signal from a distant station:
* **Parameters:** The internal circuitry adjusting to the frequency (automatic).
* **Hyperparameters:** The knobs you turn manually, like "Gain," "Bass," or "Treble."

If the "Gain" is too high, the audio is loud but distorted (overfitting). If it's too low, you hear nothing (underfitting). Finding the perfect combination of knobs is difficult because:
1. **Unknown Interactions:** Turning up the Bass might require turning down the Gain to avoid distortion.
2. **Expensive Evaluation:** Every time you move a knob, you have to listen (train the model) for a while to judge the quality.

### Real-World Connection
In system design, poor hyperparameter choices lead to suboptimal models. A state-of-the-art Transformer architecture can perform worse than a simple logistic regression if its learning rate is set incorrectly. Hyperparameter Tuning is the process of automating the search for the "golden" configuration that maximizes model performance (e.g., accuracy, F1-score) on a validation set.

---

## 2. Conceptual Foundations

### Key Definitions

* **Parameter ($\theta$):** Internal variables learned by the model (e.g., weights in a Neural Network, coefficients in Regression).
* **Hyperparameter ($\lambda$):** Configuration variables set before training (e.g., Learning Rate $\alpha$, Tree Depth, Regularization strength $\lambda$).
* **Objective Function ($f(\lambda)$):** The metric we want to maximize/minimize (e.g., Validation Accuracy). Evaluating $f(\lambda)$ is expensive because it requires training a model from scratch.
* **Search Space ($\Lambda$):** The valid ranges for all hyperparameters (e.g., Learning Rate $\in [10^{-5}, 10^{-1}]$).

### The Search Components
1. **Search Strategy:** How do we choose the next set of hyperparameters to test? (Grid, Random, Bayesian).
2. **Evaluation Strategy:** How do we assess performance? (Cross-validation, Hold-out set).
3. **Resource Management:** How much budget (time/compute) do we allocate to each configuration? (Full training vs. Early stopping).

### The Methods Explained

#### A. Manual & Grid Search
* **Manual:** The "Graduate Student Descent." You guess values based on intuition. It is unscientific and hard to reproduce.
* **Grid Search:** You define a fixed set of values for each hyperparameter and try every possible combination (Cartesian product). It guarantees finding the best combination within the grid but suffers from the **Curse of Dimensionality** (adding one parameter multiplies the search cost).

#### B. Random Search
You sample hyperparameters randomly from defined distributions.
* **Intuition:** In high-dimensional spaces, usually only a few hyperparameters matter (Low Effective Dimensionality). Grid search wastes time checking irrelevant parameters. Random search explores the "important" dimensions more densely.

#### C. Bayesian Optimization (BayesOpt)
Instead of shooting in the dark (Random), BayesOpt builds a probabilistic model (surrogate) of the objective function.
* **Surrogate Model:** "I think the function looks like this based on what I've seen." (Usually a Gaussian Process).
* **Acquisition Function:** "Where should I look next?" It balances **Exploration** (looking in uncertain areas) and **Exploitation** (refining known good areas).

#### D. Multi-fidelity & Hyperband
Training a model to completion to test a bad hyperparameter is wasteful.
* **Multi-fidelity:** Use cheap approximations (train on 10% of data, or for few epochs) to estimate performance.
* **Successive Halving:** Start with $N$ random configurations. Train them for a small time. Keep the top 50%, kill the rest. Double the training time for the survivors. Repeat.
* **Hyperband:** An algorithm that wraps Successive Halving to automatically determine how many configurations to start with and how aggressively to cut them.

#### E. BOHB (Bayesian Optimization + HyperBand)
Hyperband is efficient at pruning bad models but selects configurations randomly. BOHB replaces the random selection in Hyperband with Bayesian Optimization to intelligently guide the search for new configurations.

---

## 3. Mathematical Formulation

### The Optimization Problem
We seek the hyperparameter configuration $\lambda^*$ from the space $\Lambda$ that minimizes the validation loss $\mathcal{L}_{val}$:

$$
\lambda^* = \underset{\lambda \in \Lambda}{\arg\min} \ \mathbb{E}_{D_{val}} \left[ \mathcal{L}(\mathcal{A}_{\lambda}(D_{train}), D_{val}) \right]
$$

Where $\mathcal{A}_{\lambda}$ is the learning algorithm configured by $\lambda$.

### 1. Bayesian Optimization (Gaussian Processes)
We model the objective function $f(\lambda)$ as a random function using a **Gaussian Process (GP)**.

**GP Definition:** A GP is defined by a mean function $\mu(\lambda)$ and a covariance kernel $k(\lambda, \lambda')$.

$$
f(\lambda) \sim \mathcal{GP}(\mu(\lambda), k(\lambda, \lambda'))
$$

**The Update (Posterior):**
Given observed data $D_t = \{(\lambda_1, y_1), ..., (\lambda_t, y_t)\}$, the posterior distribution for a new point $\lambda_{new}$ is Gaussian with:
* **Posterior Mean:** $\mu_t(\lambda_{new})$ (Prediction of performance)
* **Posterior Variance:** $\sigma_t^2(\lambda_{new})$ (Uncertainty/Confidence)

**Acquisition Function: Expected Improvement (EI)**
We choose the next $\lambda_{t+1}$ by maximizing EI. Let $f^*$ be the best value observed so far (minimization task).

$$
EI(\lambda) = \mathbb{E} [\max(f^* - f(\lambda), 0)]
$$

Using the GP posterior, this has a closed form:

$$
EI(\lambda) = (f^* - \mu_t(\lambda))\Phi(Z) + \sigma_t(\lambda)\phi(Z)
$$

$$
\text{where } Z = \frac{f^* - \mu_t(\lambda)}{\sigma_t(\lambda)}
$$

* $\Phi$: CDF of standard normal.
* $\phi$: PDF of standard normal.
* **Intuition:** The first term rewards high predicted performance (Exploitation). The second term rewards high uncertainty (Exploration).

### 2. Hyperband (Successive Halving)
Let $B$ be the total budget (e.g., total epochs). Let $n$ be the number of configurations and $r$ be the resources per config.
In each round $i$:
1. Run $n_i$ configurations for $r_i$ resources.
2. Rank them by validation loss.
3. Keep the top $1/\eta$ fraction (usually $\eta=3$).
4. Increase resources $r_{i+1} = \eta \cdot r_i$.

Mathematical constraint: $n_i \cdot r_i \approx \text{Constant Budget}$.

---

## 4. Worked Examples

### Example 1: Grid vs. Random Search (The "Low Effective Dimensionality" Proof)

**Scenario:** We have a function $f(x, y) = g(x) + h(y)$.
* $g(x) = -x^2$ (Important: changing $x$ changes output significantly).
* $h(y) = 0.001y$ (Unimportant: changing $y$ barely matters).
* Goal: Maximize $f(x, y)$.

**Grid Search Approach:**
* We lay a $3 \times 3$ grid.
* Values tested for $x$: $\{-1, 0, 1\}$. Values for $y$: $\{-1, 0, 1\}$.
* Total evaluations: 9.
* **Result:** We only test 3 unique values of the important parameter $x$. If the optimum is at $x=0.5$, we miss it entirely.

**Random Search Approach:**
* We sample 9 points randomly.
* Values tested for $x$: $\{-0.8, 0.2, 0.55, -0.1, ...\}$.
* **Result:** We test 9 unique values of $x$. We have a much higher probability of hitting close to $x=0.5$ because we didn't waste budget repeating $x$ values just to vary the unimportant $y$.

### Example 2: Bayesian Optimization Step-by-Step

**Objective:** Minimize $f(x)$ (unknown expensive function).  
**Current Data:** We have tried 2 points:
1. $x_1=2, y_1=10$
2. $x_2=5, y_2=12$

**Step 1: Fit GP.** The GP fits a curve that goes through $(2,10)$ and $(5,12)$.
* At $x=3.5$ (midpoint), the GP predicts a mean close to 11, but the variance (uncertainty) is high because it's far from observed points.
* At $x=2.1$, variance is low (we know the value at 2 is 10).

**Step 2: Calculate Acquisition (EI).**
* **Region A ($x \approx 2$):** Mean is good (low), but variance is low. Low potential for *improvement*.
* **Region B ($x \approx 8$):** Mean might be high (based on prior), but variance is massive (we've never looked there). High exploration value.
* **Region C ($x \approx 3.5$):** Moderate mean, moderate variance.

**Decision:** The EI function peaks at $x=8$ (pure exploration).  
**Action:** We evaluate $f(8)$. Result: $f(8) = 6$ (New best!).  
**Update:** Add $(8,6)$ to data, update GP. Now the GP knows the function dips at 8. Next step will likely exploit around 8 to find the precise minimum.

---

## 5. Relevance to Machine Learning Practice

### When to use what?

| Method | Best For | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Grid Search** | Low dimensions (1-2 params), Benchmarking | Simple, parallelizable | Extremely inefficient in high dimensions |
| **Random Search** | General baseline, initial exploration | Beats Grid, easy to implement | Uninformed (doesn't learn from history) |
| **Bayesian Opt** | Expensive models, few parameters (<20) | Very sample efficient (needs fewer runs) | Hard to parallelize (sequential), doesn't scale to high dims |
| **Hyperband** | Deep Learning (Iterative training) | Aggressive resource saving | Can throw away good models that learn slowly |
| **BOHB** | State-of-the-Art Deep Learning | Combines efficiency of Hyperband + smarts of Bayes | Complex to implement |

### Trade-offs
1. **Computational Cost:** BayesOpt takes time to compute the "next step" (optimizing the acquisition function). If training the model is instant (e.g., Linear Regression), BayesOpt is overkill; use Random. If training takes 3 days, use BayesOpt.
2. **Bias-Variance in Tuning:**
   * **Over-tuning:** If you aggressively optimize hyperparameters on a small validation set, you will overfit the hyperparameters to that specific split. The model will fail on the Test set.
   * **Solution:** Use Cross-Validation for the objective function (more expensive but robust).

### Hyperparameter Importance (fANOVA)
After running a search, it is critical to analyze **functional ANOVA (fANOVA)**. This technique quantifies how much of the variance in performance is caused by specific hyperparameters (main effects) or their interactions.
* *Practical Insight:* If fANOVA shows "Batch Size" accounts for 1% of variance, stop tuning it. Fix it and focus resources on "Learning Rate."

---

## 6. Common Pitfalls & Misconceptions

### 1. The "Linear Scale" Mistake
* **Pitfall:** Searching for Learning Rate uniformly between 0.0001 and 1.
* **Why:** 90% of your samples will be between 0.1 and 1. But the difference between 0.0001 and 0.01 is huge (100x), while 0.8 and 0.9 is negligible.
* **Fix:** Always search Learning Rates and Regularization terms on a **Log Scale** ($10^{-4}, 10^{-3}, ...$).

### 2. Information Leakage
* **Pitfall:** Tuning hyperparameters using the Test set.
* **Why:** You are optimizing your model to look good on the test set. You have no unseen data left to evaluate true generalization.
* **Fix:** Train / Validation / Test split. Tune on Validation. Report on Test.

### 3. Comparing Apples to Oranges
* **Pitfall:** Comparing Model A (tuned for 100 hours) vs. Model B (tuned for 1 hour) and claiming A is architecturally better.
* **Fix:** In research, "tuning budget" must be equalized for fair comparison.

### 4. Ignoring Random Seeds
* **Pitfall:** Thinking a hyperparameter change improved the model when it was actually just a lucky random initialization.
* **Fix:** For rigorous tuning, evaluate each configuration multiple times with different seeds (expensive but necessary).

---

# Interview Preparation

## Foundational Questions

**Q1: Explain the difference between Model Parameters and Hyperparameters.**
* **Answer:** Parameters are internal variables learned from the training data (e.g., weights $w$, biases $b$). Hyperparameters are external configurations set before training starts (e.g., learning rate $\eta$, batch size). Parameters define the transformation of input to output; hyperparameters define the strategy for finding the parameters.

**Q2: Why is Random Search generally preferred over Grid Search?**
* **Answer:** Random Search is preferred due to the concept of **Low Effective Dimensionality**. In high-dimensional spaces, usually only a few hyperparameters significantly affect performance. Grid search wastes computations exploring the unimportant dimensions while discretizing the important ones too sparsely. Random search projects unique values onto every dimension, exploring the important parameters much more densely for the same budget.

## Mathematical Questions

**Q3: In Bayesian Optimization, what is the role of the Acquisition Function? Describe Expected Improvement (EI).**
* **Answer:** The Acquisition Function guides the search by determining which point to evaluate next. It takes the posterior distribution (from the Gaussian Process) and calculates a score representing the value of sampling a point. It balances **Exploration** (sampling high-variance/uncertain regions) and **Exploitation** (sampling high-mean/promising regions). 
  
  **EI** is defined as $EI(x) = \mathbb{E}[\max(f_{best} - f(x), 0)]$. It calculates the expected magnitude of improvement over the current best result.

**Q4: How does Hyperband allocate resources?**
* **Answer:** Hyperband uses an algorithm called **Successive Halving**. It starts with a large population of random configurations ($n$) and allocates a small resource budget ($r$) to each. It evaluates them, discards the worst fraction (usually bottom 75%), and promotes the survivors to the next round with double the budget. This allows it to quickly identify poor performers without wasting full training resources, allocating the majority of compute to the most promising candidates.

## Applied Questions

**Q5: You are training a massive Transformer model (GPT-style) that takes 1 week to train. You have limited compute. How do you tune the Learning Rate?**
* **Answer:**
  1. **Do not use Grid/Random Search:** Too expensive.
  2. **Learning Rate Sweep (Linear warmup, decay):** Perform a short run (e.g., 1 epoch or fewer steps) increasing LR exponentially to find the point of divergence (Loss explosion). Pick a value slightly below that peak.
  3. **Manual "Babysitting":** Start 3–4 distinct values (log scale). Monitor the loss curve for the first few hours. Kill runs that plateau early or diverge. This is essentially manual Successive Halving.
  4. **Gradient Noise Scale:** *(Advanced)* Use metrics derived from gradient statistics to estimate optimal batch size/LR ratio without full training.

**Q6: How do you handle hyperparameters that are conditional? (e.g., Parameter 'A' only exists if Parameter 'B' is set to 'True').**
* **Answer:** This requires a **Structured Search Space** (or Conditional Hyperparameter Space).
  * Simple Grid/Random search struggles here because sampling 'A' when 'B=False' is wasted computation.
  * **Tree-structured Parzen Estimators (TPE)** or hierarchical Bayesian Optimization methods handle this natively by modeling the dependency graph. The search algorithm only proposes values for child hyperparameters if the parent is active.

## Debugging & Failure Modes

**Q7: You implemented Bayesian Optimization, but it is performing worse than Random Search. What could be happening?**
* **Failure Modes:**
  1. **Boundary Issues:** The optimal value lies at the very edge of your defined search space, and the GP Prior pushes the search toward the center (mean reversion).
  2. **Hyperparameter Scale:** You didn't normalize your hyperparameters. If one input is $[0, 0.001]$ and another is $[0, 1000]$, the isotropic kernel (RBF) in the GP will fail to model the distance correctly.
  3. **Over-Exploration:** The acquisition function parameter (e.g., $\kappa$ in UCB or $\xi$ in EI) is set too high, causing the model to explore noise rather than exploit the minimum.

**Q8: Your hyperparameter search found a configuration that gave 99% accuracy. When you deployed the model, it dropped to 85%. Why?**
* **Answer:** This is **Hyperparameter Overfitting**.
  * You effectively "trained" the hyperparameters on the validation set. The validation set is no longer an unbiased estimator of generalization error.
  * **Fix:** You need a "Test Set" that was kept completely locked away during the hyperparameter tuning phase. Only evaluate on it once the final configuration is chosen. Or, use Nested Cross-Validation.

## Follow-up / Probing Questions

* **Interviewer:** *"You mentioned Gaussian Processes scale poorly. Why?"*
  * **Response:** GPs require inverting a kernel covariance matrix of size $N \times N$, where $N$ is the number of observed points. Matrix inversion is $\mathcal{O}(N^3)$. If you run optimization for thousands of steps, the computational overhead of the GP calculation eventually exceeds the time it takes to train the model itself.

* **Interviewer:** *"What if my hyperparameter space is discrete/categorical (e.g., Optimizer: Adam vs. SGD)?"*
  * **Response:** Standard GPs assume continuous spaces. For categorical data, you must use kernels that handle discrete distances (like Hamming distance) or switch to tree-based surrogate models like **Random Forests** (used in SMAC) or **TPE** (used in Hyperopt), which handle categorical splits naturally rather than relying on smooth Euclidean distance metrics.