# Causal Inference

## 1. Motivation & Intuition

### Why This Topic Exists
In standard machine learning (supervised learning), the focus is purely on **prediction**. The core question asked is: *"Given that we observe $X$, what is the most likely value of $Y$?"* This relies entirely on correlation.

However, in business, science, and engineering, the objective is often **intervention**. The core question shifts to: *"If I take action $T$, how will it change $Y$?"* This relies on causation.

**The core problem:** Correlation does not imply causation. A machine learning model might correctly predict that people who take sleeping pills sleep longer on average; that does not mean prescribing a random person a sleeping pill will increase their sleep (they may suffer from insomnia, which is the underlying cause of both the pill-taking and the erratic sleep schedule).

### Intuition: The Counterfactual
Causal inference is the formal study of **"What if?"**

Imagine a patient has a headache, takes a pill, and an hour later the headache is gone.
* **Factual:** The patient took the pill; the headache resolved.
* **Counterfactual:** *If the patient had NOT taken the pill, would the headache still have resolved?*

If the headache would have resolved on its own, the pill had zero causal effect. The fundamental difficulty of causal inference is that **we can never observe the counterfactual.** We cannot simultaneously administer and withhold a pill to the exact same person at the exact same moment in time.

### Real-World Machine Learning Connection
1. **Churn Prediction vs. Churn Prevention:** A standard XGBoost model identifies *who* is likely to churn. A causal model identifies *who will stay ONLY IF they are given an incentive* (a paradigm known as Uplift Modeling).
2. **Recommender Systems:** Did showing an advertisement *cause* a user to purchase an item, or was the user going to purchase the item anyway?
3. **Algorithmic Fairness:** Did a credit-scoring model reject an applicant *because* of their protected attribute (e.g., gender), or due to historical income variables correlated with that attribute?

---

## 2. Conceptual Foundations

### A. The Two Languages of Causality
Causal inference is traditionally unified through two primary frameworks. 

#### 1. The Potential Outcomes Framework (Rubin Causal Model)
This framework treats causal inference strictly as a missing-data problem.
* **Unit ($i$):** An individual atomic entity (e.g., a user, a patient, a transaction).
* **Treatment ($T_i$):** The binary intervention applied ($1 = \text{Treated}$, $0 = \text{Control}$).
* **Potential Outcomes:**
  * $Y_{1i}$: The outcome for unit $i$ if they receive the treatment.
  * $Y_{0i}$: The outcome for unit $i$ if they do *not* receive the treatment.
* **Observed Outcome ($Y_i$):** We observe $Y_{1i}$ if $T_i=1$, and $Y_{0i}$ if $T_i=0$. 

#### 2. Structural Causal Models (Pearl's Framework / DAGs)
This framework uses directed graphs to map the data-generating process.
* **DAG (Directed Acyclic Graph):** Nodes represent random variables; directed edges (arrows) represent direct causal influence.
* **SCM (Structural Causal Model):** A set of deterministic assignments $Y = f(X, U)$ describing how variables generate one another, where $U$ represents unobserved background noise.

### B. Core Assumptions
To compute causal effects from observational data without randomized experimentation, three strict assumptions must hold:

1. **SUTVA (Stable Unit Treatment Value Assumption):**
   * *No Interference:* The treatment assigned to unit $i$ does not affect the potential outcomes of unit $j$. (Violated heavily in marketplace networks or vaccine rollouts).
   * *Consistency:* The treatment is unambiguously defined. If $T_i=1$, the observed outcome $Y_i$ is strictly equal to the potential outcome $Y_{1i}$.
2. **Conditional Ignorability (Unconfoundedness):**
   * Given a set of observable covariates $X$, the treatment assignment is conditionally independent of the potential outcomes: $(Y_1, Y_0) \perp T \mid X$. In plain terms, there are no unobserved confounders.
3. **Positivity (Overlap):**
   * For every valid combination of features $x$, the probability of being assigned to either treatment or control is strictly bounded away from $0$ and $1$: $0 < P(T=1 \mid X=x) < 1$.

### C. Graph Structures & Information Flow
Understanding directional relationships in DAGs determines what variables must be controlled for:

1. **Chain ($A \rightarrow B \rightarrow C$):** $A$ causes $B$, which causes $C$. $A$ and $C$ are mutually dependent, but become **independent** if we condition on the mediator $B$.
2. **Fork ($A \leftarrow B \rightarrow C$):** $B$ is a **Confounder** causing both $A$ and $C$. $A$ and $C$ exhibit a spurious statistical correlation. Conditioning on $B$ blocks this path and eliminates the correlation.
3. **Collider ($A \rightarrow B \leftarrow C$):** Both $A$ and $C$ cause $B$. $A$ and $C$ are naturally independent. **Crucial pitfall:** If you condition on a collider $B$, you *open* the path, manufacturing a false, spurious correlation between $A$ and $C$ (known as Berkson's Paradox or Selection Bias).

---

## 3. Mathematical Formulation

### A. The Average Treatment Effect (ATE)
The individual treatment effect is defined as $\tau_i = Y_{1i} - Y_{0i}$. Because individual effects cannot be measured directly, we estimate the population expectation:

$$
ATE = E[Y_1 - Y_0] = E[Y_1] - E[Y_0]
$$

### B. Selection Bias Decomposition
If we naively subtract the mean of the observed control group from the mean of the observed treatment group, we get:

$$
\underbrace{E[Y \mid T=1] - E[Y \mid T=0]}_{\text{Observed Difference in Means}} = \underbrace{E[Y_1 - Y_0]}_{\text{True ATE}} + \underbrace{\{E[Y_0 \mid T=1] - E[Y_0 \mid T=0]\}}_{\text{Selection Bias}}
$$

* **Intuition of Selection Bias:** The difference in baseline (untreated) potential outcomes between the group that opted into treatment versus the group that did not. If sicker patients elect to take a drug ($T=1$), their baseline health $E[Y_0 \mid T=1]$ is naturally lower than healthy non-takers $E[Y_0 \mid T=0]$, making a genuinely helpful drug look statistically harmful.

### C. The Do-Operator & Backdoor Adjustment
In Judea Pearl's notation, $P(Y \mid T=t)$ represents passive observation, whereas $P(Y \mid do(T=t))$ represents active physical intervention. 

**The Backdoor Adjustment Formula:**
If a set of observed variables $X$ satisfies the backdoor criterion (it blocks every spurious path between $T$ and $Y$ without containing any descendants of $T$), the interventional distribution can be identified from purely observational data:

$$
P(Y \mid do(T=t)) = \sum_{x} P(Y \mid T=t, X=x) P(X=x)
$$

### D. Primary Estimation Methods

#### 1. Inverse Probability Weighting (IPW)
We model the propensity score—the probability of receiving treatment given observed features:

$$
e(x) = P(T=1 \mid X=x)
$$

We then re-weight the observed population to create a synthetic pseudo-population where treatment assignment is statistically uncorrelated with $X$:

$$
\hat{\mu}_{IPW} = \frac{1}{N} \sum_{i=1}^N \left( \frac{Y_i T_i}{e(X_i)} - \frac{Y_i (1-T_i)}{1-e(X_i)} \right)
$$

#### 2. Doubly Robust Estimators
Combines an outcome regression model $\hat{\mu}(t, x) = E[Y \mid T=t, X=x]$ with a propensity score model $\hat{e}(x)$. The estimator remains asymptotically unbiased if **either** the propensity model OR the outcome model is correctly specified.

---

## 4. Worked Example: Simpson's Paradox

**Scenario:** We evaluate whether a new therapeutic drug ($T$) improves patient recovery ($Y$).  
**Confounder:** Patient Gender ($X$). Men naturally recover at higher rates than women, and men are also prescribed the experimental drug more frequently.

### The Observational Dataset

| Stratum | Treated ($T=1$) Recovered / Total | Control ($T=0$) Recovered / Total | Recovery Rate ($T=1$) | Recovery Rate ($T=0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Men** | 81 / 87 | 234 / 270 | **93%** | 87% |
| **Women** | 192 / 263 | 55 / 80 | **73%** | 69% |
| **Combined** | 273 / 350 | 289 / 350 | **78%** | **83%** |

### Step 1: Naive Observational Calculation
Evaluating the raw aggregate data ("Combined" row):
* $P(Y=1 \mid T=1) = 78\%$
* $P(Y=1 \mid T=0) = 83\%$
* **Naive Conclusion:** The drug decreases recovery rates by $5\%$. It appears harmful.

### Step 2: Causal Adjustment Calculation
We apply the Backdoor Adjustment formula, conditioning on the confounder $X \in \{\text{Men}, \text{Women}\}$. 

Assume the true population baseline distribution is equal: $P(X=\text{Men}) = 0.5$ and $P(X=\text{Women}) = 0.5$.

1. **Calculate Causal Effect within Strata:**
   * Men: $E[Y \mid T=1, X=\text{M}] - E[Y \mid T=0, X=\text{M}] = 93\% - 87\% = +6\%$
   * Women: $E[Y \mid T=1, X=\text{W}] - E[Y \mid T=0, X=\text{W}] = 73\% - 69\% = +4\%$

2. **Marginalize over the Confounder:**
   $$ATE = \sum_{x \in \{M, W\}} \left( E[Y \mid T=1, X=x] - E[Y \mid T=0, X=x] \right) P(X=x)$$
   $$ATE = (0.06 \times 0.5) + (0.04 \times 0.5) = +5\%$$

**True Conclusion:** The drug genuinely increases recovery rates by $5\%$. The aggregate data was completely inverted due to severe selection bias: the treatment group was heavily weighted toward high-recovering men, while the control group was weighted toward lower-recovering women.

---

## 5. Relevance to Machine Learning Practice

### 1. Experimental vs. Observational Data
* **A/B Testing (RCTs):** The industry gold standard. Randomization physically forces $P(T=1 \mid X) = P(T=1)$, guaranteeing that unobserved confounders are balanced across groups.
* **Observational Causal Inference:** Deployed when running an A/B test is technically impossible, economically prohibitive, or ethically unacceptable (e.g., intentionally degrading system latency for critical users).

### 2. Instrumental Variables (IV)
Used when unobserved confounding violates conditional ignorability, provided an "Instrument" ($Z$) exists. $Z$ must affect the treatment $T$, but have zero direct causal path to $Y$ except through $T$.
* *System Design Example:* Estimating price elasticity of demand. Unobserved local economic shocks confound price ($T$) and sales ($Y$). An algorithm change randomly altering server tax calculations ($Z$) acts as a valid instrument.

### 3. Regression Discontinuity Design (RDD)
Applied when treatment assignment is deterministically triggered by a running variable crossing a strict programmatic threshold.
* *System Design Example:* A user account is automatically flagged for premium support ($T$) if their rolling 30-day spend exceeds $\$1,000$. By comparing users who spent $\$999.50$ against those who spent $\$1,000.50$, we simulate a localized randomized trial around the cutoff.

### 4. Causal Discovery (PC Algorithm)
Used to computationally infer DAG structures directly from observational data rather than relying on domain assumptions.
* **Constraint-Based (PC Algorithm):** Executes recursive conditional independence tests ($\chi^2$ or Fisher-z) to systematically prune edges from a fully connected undirected graph, orienting colliders afterward.
* **Limitation:** Cannot differentiate between structures within the same **Markov Equivalence Class** (e.g., observational data alone cannot distinguish whether $A \rightarrow B$ or $B \rightarrow A$).

---

## 6. Common Pitfalls & Misconceptions

1. **Conditioning on a Collider (Selection Bias):**
   * *Mistake:* Filtering a training dataset down to only "successful" outcomes to evaluate what features drive success.
   * *Why it fails:* If $A$ and $B$ both independently contribute to entering the dataset (Collider $C$), analyzing only the selected subset forces an artificial negative correlation between $A$ and $B$.
2. **Controlling for Mediators (Post-Treatment Bias):**
   * *Mistake:* Including intermediate variables generated *after* the treatment assignment as features in an adjustment regression.
   * *Why it fails:* If $T \rightarrow M \rightarrow Y$, controlling for $M$ absorbs the exact mechanism by which the treatment operates, biasing the estimated causal effect toward zero.
3. **Violating Positivity via Tree-Based Models:**
   * *Mistake:* Using deep decision trees to calculate propensity scores $e(x)$.
   * *Why it fails:* Unregularized decision trees easily output deterministic leaf probabilities of exactly $1.0$ or $0.0$. Inserting these into an IPW estimator results in division by zero or infinite variance.

---

## Interview Preparation Guide

### Foundational Questions

**Q1: Explain the conceptual distinction between $P(Y \mid T=1)$ and $P(Y \mid do(T=1))$.** **Answer:** $P(Y \mid T=1)$ represents the conditional probability of observing outcome $Y$ given that we passively detect treatment $T=1$. This quantity captures both the true causal mechanism and any spurious associations driven by confounders. $P(Y \mid do(T=1))$ represents the interventional distribution: the probability of $Y$ if we physically manipulate the system to force $T=1$. Graphically, the $do$-operator deletes all inbound edges pointing into node $T$, isolating the true causal effect.

**Q2: What is SUTVA, and how does two-sided marketplace dynamics violate it?** **Answer:** SUTVA stands for the Stable Unit Treatment Value Assumption. It requires that the potential outcome of unit $i$ depends strictly on the treatment assigned to unit $i$, with zero spillover effects from other units. In a ride-sharing marketplace, if we randomize a rider discount ($T=1$) to 50% of users, those treated users will book more rides, depleting the finite supply of available drivers. Consequently, users in the control group ($T=0$) experience higher wait times and lower booking rates *because* of the treatment applied to the other group, invalidating standard A/B test inference.

---

### Mathematical Questions

**Q3: Prove that Inverse Probability Weighting recovers the true population expectation of the potential outcome $E[Y_1]$.** **Answer:** We wish to evaluate the expected value of the weighted observed outcome for the treated group:

$$
E\left[ \frac{Y \cdot T}{e(X)} \right]
$$

Substitute the observed outcome identity $Y = Y_1 T + Y_0(1-T)$. Since $T \cdot (1-T) = 0$ and $T^2 = T$:

$$
E\left[ \frac{Y_1 \cdot T}{e(X)} \right]
$$

Apply the Law of Total Expectation by conditioning on covariates $X$:

$$
E_X \left[ E \left[ \frac{Y_1 \cdot T}{e(X)} \mid X \right] \right] = E_X \left[ \frac{1}{e(X)} E[Y_1 \cdot T \mid X] \right]
$$

By the Conditional Ignorability assumption, $Y_1 \perp T \mid X$. We can factor the inner expectation:

$$
E_X \left[ \frac{1}{e(X)} E[Y_1 \mid X] \cdot E[T \mid X] \right]
$$

By definition, the propensity score is $e(X) = E[T \mid X] = P(T=1 \mid X)$. The terms cancel:

$$
E_X \left[ \frac{1}{e(X)} E[Y_1 \mid X] \cdot e(X) \right] = E_X [ E[Y_1 \mid X] ] = E[Y_1]
$$

---

### Applied Machine Learning Scenarios

**Q4: An online retailer wants to measure the causal impact of launching a subscription loyalty program on 12-month customer lifetime value (LTV). Random assignment was denied by product leadership. How do you design this estimation system?** **Answer:** 1. **DAG Formulation:** Map known confounders driving both subscription opt-in ($T$) and LTV ($Y$). Major confounders include historical spend, account age, user engagement frequency, and traffic source.
2. **Model Selection:** Implement a Doubly Robust Estimator or a Causal Forest (Generalized Random Forest). Estimate propensity scores $e(x)$ using regularized logistic regression to ensure support overlap ($0 < \hat{e}(x) < 1$).
3. **Falsification Testing (Placebo Outcome):** To prove the validity of the unconfoundedness assumption to leadership, execute a **Negative Outcome Test**. Run the exact causal pipeline estimating the "effect" of the loyalty subscription on customer spend *6 months prior to the program's existence*. If the estimator outputs a statistically significant non-zero effect, unobserved confounders remain in the system, and observational claims must be rejected.

---

### Debugging & Failure Modes

**Q5: You deployed an Uplift Model using a Two-Model approach (T-Learner): Base Model A trained exclusively on treated users, Base Model B trained exclusively on control users. Estimated individual treatment effects ($\hat{\tau}_i = \hat{Y}_{1i} - \hat{Y}_{0i}$) exhibit extreme, erratic variance. What caused this failure?** **Answer:** The T-Learner calculates causal lift by subtracting two independent machine learning models. If the true underlying treatment effect $\tau$ is relatively small compared to the baseline outcome variance, any small estimation errors or distinct regularization penalties between Model A and Model B are amplified upon subtraction. 

**Architectural Fix:** Replace the T-Learner with an **X-Learner** or an **R-Learner** (Robinson's transformation). The R-Learner uses cross-fitting to isolate baseline effects, directly optimizing a single loss function for the heterogeneous treatment effect $\tau(x)$:

$$
\hat{\tau} = \arg\min_{\tau} \sum \left( (Y_i - \hat{m}(X_i)) - \tau(X_i)(T_i - \hat{e}(X_i)) \right)^2
$$

---

## 7. Additional Estimation Methods & Quasi-Experimental Designs

### A. Difference-in-Differences (DiD)

**When to Use:** Treatment is applied at a specific point in time to a specific group. Pre-treatment outcome data exist for both treated and untreated groups.

**Core Assumption (Parallel Trends):** In the absence of treatment, the treated and control groups would have followed identical outcome trajectories over time. Formally:

$$E[Y_{0,\text{post}} - Y_{0,\text{pre}} \mid D=1] = E[Y_{0,\text{post}} - Y_{0,\text{pre}} \mid D=0]$$

This is untestable for the post-treatment period (we cannot observe the counterfactual for the treated group), but falsifiable pre-treatment via placebo tests.

**The DiD Estimator:**

Define $D_i \in \{0, 1\}$ as the treatment group indicator and $T_t \in \{0, 1\}$ as the post-treatment period indicator. The DiD estimate removes time-invariant confounding and common time trends by differencing twice:

$$\hat{\tau}_{\text{DiD}} = \underbrace{\left(E[Y \mid D=1, T=1] - E[Y \mid D=1, T=0]\right)}_{\text{Change in treated group}} - \underbrace{\left(E[Y \mid D=0, T=1] - E[Y \mid D=0, T=0]\right)}_{\text{Change in control group}}$$

Equivalently estimated via OLS regression with an interaction term:

$$Y_{it} = \alpha + \beta_1 D_i + \beta_2 T_t + \tau (D_i \times T_t) + \epsilon_{it}$$

The coefficient $\tau$ on the interaction term directly identifies the average treatment effect on the treated (ATT).

**Event Study Design (Parallel Pre-Trends Test):** Run DiD with multiple pre- and post-treatment period indicators, estimating a coefficient for each period relative to the period immediately before treatment. Under valid parallel trends, all pre-treatment period coefficients should be statistically indistinguishable from zero. A significant pre-trend coefficient indicates that treated and control groups were already diverging before treatment, invalidating the parallel trends assumption.

**System Design Example:** A tech platform deploys a new recommendation algorithm ($D=1$) to Germany on January 1, while maintaining the old algorithm for UK users ($D=0$). DiD compares the change in session duration in Germany (pre vs. post January 1) against the same change in the UK. The UK serves as the counterfactual time trend, absorbing common seasonal effects (e.g., January post-holiday traffic drops).

**Staggered DiD:** When treatment rolls out to different units at different points in time (common in product launches), naive two-period DiD produces biased estimates. The Callaway-Sant'Anna estimator handles heterogeneous treatment timing by computing cohort-specific ATTs (groups that received treatment at the same time) and aggregating them.

---

### B. Synthetic Control Method

**When to Use:** When only a single unit (or small number of units) receives treatment, making traditional DiD unreliable due to insufficient control units. Most applicable to policy-level interventions affecting entire regions, platforms, or countries.

**Mechanism:** Given a treated unit $i=1$ and $J$ potential donor (control) units, construct a synthetic control as a weighted combination of donors that best matches the treated unit's pre-treatment outcome trajectory:

$$W^* = \arg\min_{W \in \Delta^J} \sum_{t \in \mathcal{T}_{\text{pre}}} \left(Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt}\right)^2$$

subject to $w_j \geq 0$ and $\sum_j w_j = 1$ (convex hull constraint prevents extrapolation beyond the observed donor distribution).

**Effect Estimation:** The post-treatment causal effect at time $t$ is estimated as:

$$\hat{\tau}_t = Y_{1t} - \sum_{j=2}^{J+1} w_j^* Y_{jt}$$

**Inference via Permutation Tests:** Apply the same synthetic control procedure iteratively to each donor unit in turn, treating each as a placebo-treated unit. If the treated unit's post-treatment gap is extreme relative to the distribution of placebo gaps, the effect is statistically significant — without requiring distributional assumptions about the error term.

**System Design Example:** California passes unique privacy legislation restricting ad targeting in 2020. No other state enacted equivalent legislation, making a traditional DiD control group impossible. The synthetic control constructs "Synthetic California" as a convex combination of Texas (28%), New York (22%), and Florida (17%) that best replicates California's 2015–2019 advertising revenue trajectory. The post-2020 divergence between California and its synthetic counterpart estimates the privacy law's revenue impact.

---

### C. Power Analysis & Sample Size Determination

Running an experiment without power analysis risks two symmetric failure modes:
* **Underpowering ($n$ too small):** The experiment fails to detect a real effect that exists (Type II Error / False Negative). The product team concludes the intervention has no effect and does not ship a change that would have improved the metric.
* **Overpowering ($n$ too large):** The experiment runs longer than necessary, delaying decisions and consuming traffic that could be allocated to other experiments.

**Core Parameters:**
* $\alpha$: Significance level (Type I error rate). Typically $0.05$.
* $1 - \beta$: Statistical power. Typically $0.80$ or $0.90$.
* $\delta$: Minimum Detectable Effect (MDE) — the smallest effect size that is practically meaningful.
* $\sigma^2$: Variance of the outcome metric.

**Required Sample Size per Arm (Two-Sample $z$-test):**

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}$$

where $z_{\alpha/2} = 1.96$ (for $\alpha = 0.05$, two-tailed) and $z_\beta = 0.84$ (for $80\%$ power) or $z_\beta = 1.28$ (for $90\%$ power).

**Intuition:** The required $n$ scales with variance (noisier metrics require more data) and inversely with $\delta^2$ (detecting smaller effects requires quadratically more data). Halving the MDE requires 4× the sample size.

**MDE as a Business Decision:** The MDE should be anchored to practical significance, not statistical convenience. Setting MDE = "what the current sample size can detect" is circular reasoning. The correct process: define the minimum lift that would change the product decision (e.g., $\delta = 0.5\%$ conversion lift on a high-margin product), then solve for $n$.

**Variance Reduction (CUPED — Controlled-experiment Using Pre-Experiment Data):** For high-variance outcome metrics, use pre-experiment covariate adjustment to reduce $\sigma^2$:

$$Y_{\text{CUPED}} = Y - \hat{\theta} \cdot (X - E[X]), \quad \hat{\theta} = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

where $X$ is a pre-experiment measurement of the same metric (e.g., last week's conversion rate for users). CUPED reduces the required sample size by the factor $(1 - \rho_{XY}^2)$ where $\rho_{XY} = \text{Corr}(Y, X)$. A pre-experiment correlation of $\rho = 0.7$ reduces required sample size by $49\%$.

**Multiple Testing Correction:** When running simultaneous A/B tests on overlapping user populations (common in large platforms), without correction each experiment has an inflated false positive rate. Apply Benjamini-Hochberg FDR control (less conservative than Bonferroni, controls the expected fraction of false discoveries rather than the probability of any false discovery) when running many parallel experiments.

---

### D. Heterogeneous Treatment Effects (CATE) — Full Framework

The Average Treatment Effect (ATE) tells us the mean effect across the population. The **Conditional Average Treatment Effect (CATE)** estimates how effects vary across individuals:

$$\tau(x) = E[Y_1 - Y_0 \mid X = x]$$

Different subpopulations respond differently to the same intervention. CATE estimation is the technical foundation of **Uplift Modeling** — targeting interventions only at individuals for whom they are effective.

#### S-Learner (Single Model)
Train one model $\hat{\mu}(T, X)$ on the full dataset with treatment $T$ included as a feature:

$$\hat{\tau}(x) = \hat{\mu}(1, x) - \hat{\mu}(0, x)$$

* **Strength:** Simple; leverages all data in a single model.
* **Limitation:** Regularized models (LASSO, trees with min-leaf-size constraints) may shrink the treatment feature's contribution toward zero, attenuating CATE estimates — especially when the treatment effect is small relative to outcome variance. The model has no incentive to prioritize treatment effect estimation over overall prediction accuracy.

#### T-Learner (Two Independent Models)
Train separate outcome models on the treated and control subsets independently:

$$\hat{\mu}_1(x) = E[Y \mid X=x, T=1], \quad \hat{\mu}_0(x) = E[Y \mid X=x, T=0]$$

$$\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

* **Strength:** Each model can be independently regularized and optimized.
* **Limitation:** Subtraction amplifies independent estimation errors. If each model has variance $\sigma^2$, the CATE variance is $2\sigma^2$ (errors are uncorrelated). Particularly unstable in imbalanced treatment/control splits where one model has far less data.

#### X-Learner (Cross-Imputation)
Designed specifically for imbalanced treatment/control splits (e.g., $10\%$ treated, $90\%$ control):

1. **Step 1:** Fit $\hat{\mu}_0$ on control units, $\hat{\mu}_1$ on treated units (identical to T-learner).
2. **Step 2:** Impute individual treatment effects using cross-predictions:
   * For treated unit $i$: $\tilde{\tau}_i^{(1)} = Y_i - \hat{\mu}_0(X_i)$ (actual treated outcome minus predicted control outcome).
   * For control unit $j$: $\tilde{\tau}_j^{(0)} = \hat{\mu}_1(X_j) - Y_j$ (predicted treated outcome minus actual control outcome).
3. **Step 3:** Fit $\hat{\tau}_1(x)$ by regressing $\tilde{\tau}_i^{(1)}$ on $X_i$ for treated units; fit $\hat{\tau}_0(x)$ by regressing $\tilde{\tau}_j^{(0)}$ on $X_j$ for control units.
4. **Step 4:** Combine using the propensity score as weight:
   $$\hat{\tau}(x) = \hat{e}(x) \cdot \hat{\tau}_1(x) + (1 - \hat{e}(x)) \cdot \hat{\tau}_0(x)$$

The X-learner borrows strength across arms via cross-imputation. With 90% control units, $\hat{\mu}_0$ is fitted on abundant data, enabling accurate imputation of the control potential outcome for treated units — effectively augmenting the treated training signal using the rich control data.

#### Causal Forests (Generalized Random Forests)
Extends random forests to CATE estimation by splitting on features that maximize heterogeneity in treatment response:

* **Splitting criterion:** At each node, select the feature split that maximizes variance in estimated treatment effects across child nodes (rather than variance in outcome directly).
* **Local CATE:** Each leaf provides a localized CATE estimate via a weighted local regression within the leaf over training samples whose forest-induced distance to the query point is small.
* **Honest Estimation:** Uses sample splitting (half the data to grow the tree structure; the other half to compute leaf-level estimates). This prevents overfitting-driven bias — a split selected to minimize within-leaf outcome variance will not spuriously inflate CATE heterogeneity when estimates are computed on held-out samples.
* **Asymptotically Valid CIs:** Provides confidence intervals for $\hat{\tau}(x)$ via infinitesimal jackknife variance estimation across the forest ensemble.

#### Uplift Segmentation in Practice
After estimating $\hat{\tau}(x)$, rank users by predicted uplift and segment into actionable behavioral groups:
* **Persuadables:** High positive $\hat{\tau}(x)$ — will convert only if treated. The primary targeting population.
* **Sure Things:** High $\hat{\mu}_0(x)$ but low $\hat{\tau}(x)$ — convert regardless of treatment; spending intervention budget here is wasteful.
* **Lost Causes:** Low $\hat{\mu}_0(x)$ and low $\hat{\tau}(x)$ — treatment has no effect and the baseline outcome is poor.
* **Sleeping Dogs:** Negative $\hat{\tau}(x)$ — treatment actively suppresses conversion (e.g., users who resent promotional emails and churn in response).

---

### E. Mediation Analysis

Mediation analysis decomposes the total causal effect of treatment $T$ on outcome $Y$ into:
* **Natural Direct Effect (NDE):** The effect of $T$ on $Y$ holding the mediator $M$ constant at its value under no treatment.
* **Natural Indirect Effect (NIE):** The component of $T$'s effect on $Y$ transmitted through the mediator $M$.

**Causal Graph:** $T \rightarrow Y$ (direct path) and $T \rightarrow M \rightarrow Y$ (indirect path through mediator).

**Total Effect Decomposition:**

$$\text{Total Effect} = \text{NDE} + \text{NIE}$$

**Estimation (Baron-Kenny, Linear Setting):**

Fit three regression equations:
1. $Y = c \cdot T + \epsilon_1$ → Estimate of total effect $c$.
2. $M = a \cdot T + \epsilon_2$ → Effect of treatment on mediator.
3. $Y = c' \cdot T + b \cdot M + \epsilon_3$ → Direct effect $c'$ (treatment effect after holding $M$ fixed).

* Indirect (mediated) effect: $\widehat{NIE} = \hat{a} \times \hat{b}$.
* Total effect: $\hat{c} = \hat{c}' + \hat{a}\hat{b}$.
* **Proportion mediated:** $\frac{\hat{a}\hat{b}}{\hat{c}}$.

**System Design Example:** An e-commerce platform sends a promotional email ($T$) and observes increased purchase conversion ($Y$). Mediation analysis decomposes whether the email's effect operates primarily through increased site visits ($M$ = click-through traffic) or through direct persuasion (users who visit anyway but convert at higher rates). If 80% of the effect is mediated through traffic, investment in email open-rate optimization is more impactful than landing page copy changes; if 80% is direct, the reverse holds.

**Identification Caveat:** Baron-Kenny estimation requires the no-unmeasured-mediator-confounding assumption: there must be no unobserved variable $U$ that affects both $M$ and $Y$. This is violated whenever mediator assignment is not randomized (e.g., site visits are self-selected) — the standard fix is to randomize the mediator itself in a second-stage experiment, or use sensitivity analysis to bound the estimated mediation effect under plausible levels of mediator confounding.

---

## Extended Interview Preparation

### Foundational Questions

**Q6: What is the Parallel Trends Assumption in DiD, and how do you test it empirically?**
**Answer:** Parallel Trends assumes that absent treatment, the treated and control groups would have followed identical outcome trajectories over time. It is untestable for the post-treatment period (the counterfactual trajectory for the treated group is unobserved), but testable pre-treatment via an **Event Study Design**: estimate separate DiD coefficients for each pre-treatment time period relative to the period immediately before treatment. If treated and control groups were truly on parallel pre-trends, all pre-treatment coefficients should be statistically indistinguishable from zero. A significant pre-treatment trend coefficient indicates that the groups were diverging before the intervention — either due to anticipation effects (treated units change behavior before treatment) or selection bias (treated and control units have structurally different growth dynamics), both of which invalidate the DiD causal interpretation.

**Q7: What is the Minimum Detectable Effect (MDE) in power analysis, and why is it a business decision rather than a purely statistical one?**
**Answer:** The MDE is the smallest treatment effect the experiment is designed to reliably detect at a specified power level $(1 - \beta)$. It is a business decision because it requires answering: "What is the smallest improvement that would change our product decision?" A 0.1% lift in conversion rate may be commercially meaningless for a low-volume product but worth $5M annually for a high-traffic e-commerce platform. Anchoring MDE to statistical convenience (e.g., setting it to whatever the current traffic supports) creates two failure modes: designing experiments that can only detect unrealistically large effects (missing real improvements) or detecting effects too small to justify the engineering cost of deployment. The correct process is to define practical significance first, then solve for the required sample size — not vice versa.

### Mathematical Questions

**Q8: A CATE model using the T-Learner produces highly variable uplift predictions even when the true ATE is stable and precisely estimated. Prove mathematically why this instability arises.**
**Answer:** The T-Learner estimates $\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$. Let the individual estimation errors be $\epsilon_1(x) = \hat{\mu}_1(x) - \mu_1(x)$ and $\epsilon_0(x) = \hat{\mu}_0(x) - \mu_0(x)$. The CATE estimation error is:

$$\hat{\tau}(x) - \tau(x) = \epsilon_1(x) - \epsilon_0(x)$$

The variance of this error:

$$\text{Var}(\hat{\tau}(x)) = \text{Var}(\epsilon_1(x)) + \text{Var}(\epsilon_0(x)) - 2\,\text{Cov}(\epsilon_1(x), \epsilon_0(x))$$

Because $\hat{\mu}_0$ and $\hat{\mu}_1$ are fitted on disjoint subsets of data (control and treated), their errors are statistically independent: $\text{Cov}(\epsilon_1, \epsilon_0) = 0$. Therefore $\text{Var}(\hat{\tau}(x)) = \text{Var}(\epsilon_1(x)) + \text{Var}(\epsilon_0(x))$. Even if each model achieves individually low variance on its training population, the subtracted quantity inherits the sum of both model variances — producing amplified instability in the estimated uplift, especially in data-sparse subpopulations where at least one model extrapolates.

### Applied Questions

**Q9: Design a causal inference pipeline to measure the revenue impact of a globally deployed personalization engine when A/B testing is not possible because the system must be deployed uniformly.**
**Answer:**
1. **Exploit Rollout Timing (Staggered DiD):** If the personalization engine rolls out region-by-region over several weeks, use a staggered rollout DiD design. Earlier-receiving regions are the treatment group; later-receiving regions serve as the contemporaneous control. Use the Callaway-Sant'Anna heterogeneous-timing DiD estimator to avoid the "negative weighting" bias that arises from two-way fixed effects estimators in staggered designs.
2. **Synthetic Control Validation:** For the first region to receive treatment (no other treated region exists yet as a comparison), construct a synthetic control from unaffected regions that best matches pre-rollout revenue trajectory, seasonality, and traffic composition.
3. **Falsification via Pre-Trend Test:** Run an event study plotting DiD coefficients for each week prior to any rollout. All pre-treatment coefficients should be statistically indistinguishable from zero. Any significant pre-trend signals non-parallel trajectories that would invalidate the causal interpretation.
4. **Mediation Analysis:** Decompose the estimated revenue effect into direct (users shown personalized content convert at higher rates conditional on visiting) and indirect (personalized content increases return visit frequency, generating more conversion opportunities) channels to prioritize future product investment.
