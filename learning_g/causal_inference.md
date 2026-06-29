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
