# Causal Inference: A Comprehensive Learning Guide

## Table of Contents

1. [Foundational Concepts: The Potential Outcomes Framework](#1-foundational-concepts-the-potential-outcomes-framework)
2. [Causal Graphs and Directed Acyclic Graphs (DAGs)](#2-causal-graphs-and-directed-acyclic-graphs)
3. [Identification and Estimation](#3-identification-and-estimation)
4. [Experimental vs. Observational Data](#4-experimental-vs-observational-data)
5. [Causal Discovery](#5-causal-discovery)
6. [Interview Preparation](#6-interview-preparation)

---

# 1. Foundational Concepts: The Potential Outcomes Framework

## 1.1 Motivation & Intuition

### Why Does Causal Inference Exist?

Consider a deceptively simple question: *Does this new recommendation algorithm increase user engagement?* You deploy the algorithm, and engagement goes up 5%. Can you conclude the algorithm caused the improvement? Perhaps a holiday season drove the increase. Perhaps a competitor went down. Perhaps the users who received the new algorithm were already more engaged.

The core problem is this: **correlation is not causation**, and yet almost every decision we care about — in medicine, policy, business, and machine learning — is fundamentally causal. We do not merely want to know that drug-takers recover more often; we want to know whether *taking the drug* causes recovery. We do not merely want to know that users who see a banner ad buy more; we want to know whether *showing the ad* causes them to buy more.

Standard predictive machine learning answers questions of the form "What is E[Y | X]?" — given that we observe features X, what do we expect Y to be? But predictive models are silent on interventional questions: "What would happen to Y if we *set* X to a particular value?" A model might learn that carrying a lighter is associated with lung cancer (because smokers carry lighters), but intervening to remove lighters from the population would not reduce cancer rates. The distinction between *observing* and *intervening* is exactly what causal inference formalizes.

### A Concrete Example

Suppose a hospital wants to know whether a new surgery (Treatment = 1) is better than medication alone (Treatment = 0) for heart disease patients. They have records of 10,000 patients:

- Among 6,000 patients who received surgery, 4,200 survived (70% survival).
- Among 4,000 patients who received medication, 3,200 survived (80% survival).

Naively, medication looks better. But suppose sicker patients are more likely to be sent to surgery. The patients who received surgery were fundamentally different from those who received medication — they were sicker to begin with. The observed difference in outcomes conflates the *effect of the treatment* with the *effect of being sicker*. This is **confounding**, and it is the central obstacle causal inference was designed to overcome.

### The Fundamental Problem

To truly measure the effect of surgery on a specific patient, we would need to observe *two parallel universes*: one where the patient receives surgery, and one where the same patient receives medication at the same time, under identical conditions. We could then compare the outcomes. Obviously, we can only observe one of these universes — the one that actually happened. The outcome in the universe that did not happen is called a **counterfactual**, and the impossibility of observing both potential realities simultaneously is called the **Fundamental Problem of Causal Inference** (Holland, 1986).

All of causal inference is, in essence, a collection of strategies for making credible inferences about counterfactuals despite never observing them directly.

### Connection to Machine Learning

Why should an ML practitioner care? Because ML systems increasingly make or inform *decisions*:

- **A/B testing in production**: When you run an A/B test on a new ranking model, you are conducting a randomized experiment. Understanding causal inference tells you *why* randomization works, when it can fail (network effects, interference), and how to analyze the results correctly.
- **Offline policy evaluation**: You want to estimate how a new recommendation policy would perform *without deploying it*. This is a counterfactual question: "What would the click-through rate have been if we had used policy π instead of the logging policy π₀?"
- **Fairness and bias**: Asking "Does this model discriminate based on race?" is a causal question. Adjusting for confounders vs. mediators changes the answer completely.
- **Feature selection and model debugging**: Understanding which features *cause* the outcome (vs. are merely correlated) determines which features are robust to distribution shift and which will break under intervention.

---

## 1.2 Conceptual Foundations

### The Rubin Causal Model (Potential Outcomes Framework)

The potential outcomes framework, developed primarily by Donald Rubin (building on earlier work by Jerzy Neyman), formalizes causation by imagining *potential outcomes* — the outcomes that *would* occur under each possible treatment assignment.

**Key components:**

**Units**: The entities we study. Each unit is indexed by i ∈ {1, ..., N}. A unit could be a patient, a user, a web session, a county, etc.

**Treatment**: A variable Wᵢ indicating which intervention unit i receives. In the simplest (binary) case, Wᵢ ∈ {0, 1}, where 1 denotes "treated" and 0 denotes "control." The framework extends naturally to multi-valued and continuous treatments.

**Potential Outcomes**: For each unit i, we define two potential outcomes:
- Yᵢ(1): the outcome that *would* be realized if unit i receives treatment (Wᵢ = 1).
- Yᵢ(0): the outcome that *would* be realized if unit i receives control (Wᵢ = 0).

These are *fixed attributes* of the unit — they exist conceptually regardless of what actually happens. They are not random variables in the traditional sense; the randomness comes from the treatment assignment mechanism, not from the outcomes themselves (in the Neyman tradition).

**Observed Outcome**: What we actually see is:

Yᵢ = Wᵢ · Yᵢ(1) + (1 − Wᵢ) · Yᵢ(0)

This equation makes the Fundamental Problem of Causal Inference precise: for each unit, we observe *exactly one* of the two potential outcomes. The other is missing — it is a counterfactual.

**Individual Treatment Effect (ITE)**: The causal effect of treatment for unit i is:

τᵢ = Yᵢ(1) − Yᵢ(0)

This quantity is *never directly observable* for any single unit, because we never see both Yᵢ(1) and Yᵢ(0) simultaneously.

### Neyman-Rubin Counterfactuals

The term "counterfactual" refers to the potential outcome that was *not* realized. If patient i received surgery (Wᵢ = 1), then Yᵢ(0) — what would have happened under medication — is the counterfactual. Counterfactuals are not mere speculation; they are precisely defined mathematical objects within the framework. The entire enterprise of causal inference is about estimating functions of these unobserved quantities.

A critical philosophical point: counterfactuals must be *well-defined*. The treatment must be something that could, in principle, be applied or withheld. This is Rubin's dictum: **"No causation without manipulation."** Asking "What is the causal effect of being female on salary?" is ill-defined under this framework because one cannot, in principle, manipulate a person's sex while holding everything else constant. (Pearl's framework handles such questions differently via structural equations, as we will see.)

### Stable Unit Treatment Value Assumption (SUTVA)

SUTVA is the foundational assumption that makes the potential outcomes framework coherent. It has two components:

**Component 1: No Interference (No Spillover)**

The potential outcomes for unit i depend *only* on the treatment assigned to unit i, and not on the treatments assigned to any other units.

Formally: Yᵢ(W₁, W₂, ..., Wₙ) = Yᵢ(Wᵢ) for all possible treatment assignment vectors.

This means that whether your neighbor receives the vaccine does not affect *your* health outcome. Whether user j sees the new UI does not affect *user k's* click-through rate.

**Component 2: No Hidden Versions of Treatment**

There is only one version of each treatment level. Receiving "treatment" means the same thing for every unit.

This means that every patient assigned to "surgery" receives the same type of surgery — not different procedures lumped together under one label.

**What breaks when SUTVA is violated?**

*Interference*: If you vaccinate many people in a community, the unvaccinated also benefit from herd immunity. Their Yᵢ(0) now depends on how many others were vaccinated, so we cannot define a single Yᵢ(0). Similarly, in a social network A/B test, if treated users share content with control users, the control users' outcomes are contaminated. This is a major challenge in tech platforms where users interact.

*Hidden treatment variations*: If "treatment" means different things for different units — perhaps some patients get a skilled surgeon and others get an inexperienced one — then the treatment effect is an ill-defined average over heterogeneous interventions.

### Assumptions and What Breaks

Beyond SUTVA, causal inference from observational data typically requires:

**Ignorability (Unconfoundedness / Conditional Independence)**: 

(Yᵢ(0), Yᵢ(1)) ⊥ Wᵢ | Xᵢ

Given observed covariates X, the treatment assignment is independent of the potential outcomes. In other words, after conditioning on X, there are no remaining unmeasured confounders. This is sometimes called "selection on observables" — all the factors that jointly influence both treatment and outcome are captured in the observed covariates.

When this fails: if there is an unmeasured confounder (e.g., "patient motivation" affects both whether they elect surgery and whether they recover), conditioning on observed covariates is insufficient, and estimated effects will be biased.

**Positivity (Overlap)**:

0 < P(Wᵢ = 1 | Xᵢ = x) < 1 for all x with P(Xᵢ = x) > 0

Every unit must have a nonzero probability of receiving either treatment or control, conditional on its covariates. If certain subgroups *always* receive treatment (e.g., all critically ill patients get surgery), we cannot estimate what would have happened to them without surgery — there is no comparable control group.

When this fails: estimation becomes impossible or extremely unstable in regions of the covariate space where one treatment arm is empty or nearly so. Propensity score methods amplify this problem through extreme weights.

**Consistency**:

If Wᵢ = w, then Yᵢ = Yᵢ(w)

The observed outcome under treatment w equals the potential outcome under treatment w. This connects the theoretical potential outcomes to the data we actually observe. It is essentially a formalization of "no hidden treatment versions" from SUTVA.

---

## 1.3 Mathematical Formulation

### Defining Causal Estimands

The **Average Treatment Effect (ATE)** is the most common target:

τ_ATE = E[Yᵢ(1) − Yᵢ(0)] = E[Yᵢ(1)] − E[Yᵢ(0)]

This is the average causal effect across the entire population. Note that this is *not* the same as E[Y | W = 1] − E[Y | W = 0], which is the observed difference in means. The two coincide only under specific conditions (randomization or unconfoundedness).

The **Average Treatment Effect on the Treated (ATT)**:

τ_ATT = E[Yᵢ(1) − Yᵢ(0) | Wᵢ = 1]

This answers: "Among those who actually received treatment, what was the average effect?" The counterfactual here is: what would have happened to the treated units had they *not* been treated?

The **Conditional Average Treatment Effect (CATE)**:

τ(x) = E[Yᵢ(1) − Yᵢ(0) | Xᵢ = x]

This is the treatment effect for units with covariates X = x. It captures treatment effect *heterogeneity* — the effect may differ across subgroups.

### Identification Under Unconfoundedness

Under the assumptions of ignorability, positivity, and consistency, we can identify the ATE from observed data:

τ_ATE = E[Yᵢ(1)] − E[Yᵢ(0)]

Consider E[Yᵢ(1)]:

E[Yᵢ(1)] = Eₓ[E[Yᵢ(1) | Xᵢ]]                    (law of iterated expectations)
           = Eₓ[E[Yᵢ(1) | Wᵢ = 1, Xᵢ]]              (by ignorability: Y(1) ⊥ W | X)
           = Eₓ[E[Yᵢ | Wᵢ = 1, Xᵢ]]                  (by consistency: Y = Y(1) when W = 1)

Similarly:

E[Yᵢ(0)] = Eₓ[E[Yᵢ | Wᵢ = 0, Xᵢ]]

Therefore:

τ_ATE = Eₓ[E[Y | W = 1, X] − E[Y | W = 0, X]]

This is the **adjustment formula** (or g-formula / standardization). It says: compute the treatment-control difference within each stratum defined by X, then average over the distribution of X. Each quantity on the right-hand side is estimable from observed data.

### The Role of the Propensity Score

The **propensity score** is defined as:

e(x) = P(W = 1 | X = x)

Rosenbaum and Rubin (1983) proved a fundamental dimensionality reduction result: if (Y(0), Y(1)) ⊥ W | X, then (Y(0), Y(1)) ⊥ W | e(X). That is, conditioning on the scalar propensity score achieves the same deconfounding as conditioning on the full covariate vector X. This is enormously useful when X is high-dimensional.

**Derivation sketch**: For any set A, P(W = 1 | Y(0) ∈ A, e(X)) can be shown to equal e(X) by iterated expectations and the unconfoundedness assumption. Since this does not depend on Y(0), we have Y(0) ⊥ W | e(X), and analogously for Y(1).

---

## 1.4 Worked Example

### Setup

Suppose we have 8 patients. We know each patient's age (X), whether they received surgery (W), and whether they survived (Y). We want to estimate the ATE of surgery on survival.

| Patient | Age (X) | Surgery (W) | Survived (Y) |
|---------|---------|-------------|---------------|
| 1       | Young   | 1           | 1             |
| 2       | Young   | 1           | 1             |
| 3       | Young   | 0           | 1             |
| 4       | Young   | 0           | 0             |
| 5       | Old     | 1           | 0             |
| 6       | Old     | 1           | 0             |
| 7       | Old     | 0           | 1             |
| 8       | Old     | 0           | 0             |

**Naive comparison (ignoring confounding):**

E[Y | W = 1] = (1 + 1 + 0 + 0) / 4 = 0.50
E[Y | W = 0] = (1 + 0 + 1 + 0) / 4 = 0.50

Naive estimate: 0.50 − 0.50 = 0. No apparent effect.

**Stratified analysis (adjusting for Age):**

Young patients:
- E[Y | W = 1, Young] = (1 + 1) / 2 = 1.0
- E[Y | W = 0, Young] = (1 + 0) / 2 = 0.5
- Effect among young: 1.0 − 0.5 = +0.5

Old patients:
- E[Y | W = 1, Old] = (0 + 0) / 2 = 0.0
- E[Y | W = 0, Old] = (1 + 0) / 2 = 0.5
- Effect among old: 0.0 − 0.5 = −0.5

Averaging over the population (50% young, 50% old):

τ_ATE = 0.5 × (+0.5) + 0.5 × (−0.5) = 0.0

In this toy example, the stratified estimate also gives 0, but the *reason* is different: surgery helps young patients and hurts old patients, averaging to zero. The naive comparison happened to give the right answer here, but for the wrong reason — and in general, confounding will bias the naive comparison away from the truth.

**Propensity score calculation:**

e(Young) = P(W = 1 | Young) = 2/4 = 0.5
e(Old) = P(W = 1 | Old) = 2/4 = 0.5

In this example, treatment assignment is balanced within strata, so propensity scores are equal. In real data with confounding, these would differ across groups.

---

## 1.5 Relevance to Machine Learning Practice

### Where Causal Inference Appears in ML

**Online experimentation (A/B testing)**: Every A/B test is a randomized experiment. The Rubin framework tells us exactly what we are estimating (ATE), under what assumptions (SUTVA), and how randomization eliminates confounding. Violations of SUTVA — network effects in social products, shared resources in marketplace experiments — are among the hardest challenges in industry experimentation.

**Uplift modeling / Heterogeneous treatment effects**: Rather than estimating a single ATE, we want to estimate τ(x) = E[Y(1) − Y(0) | X = x] for targeting — showing different treatments to different users based on who benefits most. Methods like causal forests (Wager and Athey, 2018) and meta-learners (S-learner, T-learner, X-learner) are built directly on the potential outcomes framework.

**Counterfactual evaluation (off-policy evaluation)**: In reinforcement learning and recommendation systems, we have logged data from a behavior policy and want to evaluate a new target policy. Inverse propensity scoring (IPS) directly applies the propensity score from the Rubin framework, reweighting logged rewards by the ratio of target to behavior policy probabilities.

**Fairness**: Causal definitions of fairness (e.g., counterfactual fairness) ask: "Would the model's prediction change if the individual's protected attribute had been different, holding everything else that is not causally downstream of the attribute constant?" This is a counterfactual question that requires a causal model.

### When to Use vs. Not Use

Use the potential outcomes framework when you have a well-defined treatment whose effect you want to estimate, when "manipulation" of the treatment is conceptually meaningful, and when the primary question is about the *effect* of an intervention.

Do not force this framework when the question is purely predictive (no intervention), when the "treatment" is not well-defined or manipulable (e.g., "effect of race"), or when the fundamental assumptions (especially unconfoundedness) are clearly violated and no remedies (IV, RDD, etc.) are available.

### Trade-offs

*Interpretability vs. flexibility*: The potential outcomes framework gives clear, interpretable estimands (ATE, ATT, CATE) at the cost of requiring strong assumptions. More flexible approaches (e.g., deep learning for causal effect estimation) may capture complex relationships but make it harder to verify assumptions.

*Bias vs. variance*: Methods that adjust more aggressively for confounders (e.g., using many covariates) reduce bias from confounding but may increase variance, especially with limited data or near-violations of positivity.

*Internal vs. external validity*: Randomized experiments (high internal validity) may not generalize to the target population (limited external validity). Observational studies may have broader coverage but weaker causal identification.

---

## 1.6 Common Pitfalls & Misconceptions

**Pitfall 1: Confusing prediction with causation.** A model that predicts Y well from X is not identifying causal effects. The highest-weighted features in a predictive model may be confounders or colliders, not causes. For example, "hospital admission" predicts pneumonia mortality, but admitting patients does not cause death — sicker patients are both more likely to be admitted and more likely to die.

**Pitfall 2: Assuming unconfoundedness without justification.** Unconfoundedness is untestable from data alone (you cannot verify that you have measured all confounders). Analysts often assume it because it is convenient, not because it is plausible. Always ask: "What unmeasured factor could jointly affect treatment and outcome?"

**Pitfall 3: Ignoring SUTVA violations.** In networked settings (social media, marketplaces), interference between units is the norm, not the exception. Standard ATE estimators are biased when SUTVA is violated. Graph-cluster randomization and other designs are needed.

**Pitfall 4: Conditioning on post-treatment variables.** Adjusting for variables that are *affected by* the treatment (mediators or colliders) introduces bias rather than removing it. For example, if you are estimating the effect of a job training program on earnings, you should not condition on "currently employed" — employment is itself affected by the training.

**Pitfall 5: Conflating ATT with ATE.** The average effect among the treated (ATT) can differ substantially from the population ATE. If treatment is targeted at those who benefit most, the ATT will overestimate the ATE. Policy conclusions depend on which estimand is relevant.

**Pitfall 6: Applying the Rubin framework to non-manipulable attributes.** Asking "What is the causal effect of gender on salary?" requires additional structure (e.g., a structural causal model specifying what it means to "intervene" on gender). The potential outcomes framework alone struggles here because the counterfactual "the same person but different gender" is not well-defined.

---

# 2. Causal Graphs and Directed Acyclic Graphs (DAGs)

## 2.1 Motivation & Intuition

### Why Do We Need Graphs?

The potential outcomes framework gives us precise *estimands* (what we want to estimate) and clear *assumptions* (unconfoundedness, SUTVA). But it does not tell us *which variables to adjust for*. When you have ten covariates, should you condition on all of them? Some of them? Which ones?

Getting this wrong is not merely suboptimal — it can make things *worse*. Conditioning on the wrong variable (a collider) can *introduce* bias where none existed. Failing to condition on a confounder leaves bias in place. The adjustment set is not "all available variables" and is not determined by statistical significance.

Causal graphs, and specifically Directed Acyclic Graphs (DAGs), provide a transparent language for encoding *qualitative causal assumptions* about the data-generating process. From a DAG, we can mechanically determine which variables to adjust for, which variables to leave alone, and whether the causal effect is even identifiable from the available data. This is the central contribution of Judea Pearl's framework.

### A Simple Example

Consider three variables: Ice Cream Sales (I), Drowning Deaths (D), and Temperature (T).

Observation: Ice cream sales and drowning deaths are correlated. Does ice cream cause drowning?

Obviously not. Temperature is a common cause: hot weather increases both ice cream sales and swimming (hence drowning). The causal structure is:

```
T → I
T → D
```

Temperature is a **confounder**. If we condition on temperature, the spurious association between I and D disappears. The DAG makes this reasoning precise and mechanical.

Now consider a subtler example: Talent (T), Movie Funding (F), and Celebrity Status (C).

```
T → C
F → C
```

Here, Talent and Funding both contribute to Celebrity Status. T and F are independent causes. But if we condition on C (e.g., look only at celebrities), we create a spurious association between T and F: among celebrities, those who are less talented must have had more funding, and vice versa. Conditioning on the **collider** C induces a false association. The DAG warns us against this.

### Connection to ML

In ML, DAGs help determine which features to include in a model when the goal is causal effect estimation (vs. prediction). They guide variable selection for debiasing, help identify when standard regression will fail, and underpin the structural equations behind methods like do-calculus. In fairness, DAGs explicitly encode the causal pathways through which a protected attribute affects the outcome, distinguishing direct discrimination from indirect effects.

---

## 2.2 Conceptual Foundations

### Structural Causal Models (SCMs)

A Structural Causal Model is a triple M = (U, V, F) where:

- **U** = {U₁, ..., Uₘ}: a set of *exogenous* (external, unobserved) variables. These represent all factors outside the model.
- **V** = {V₁, ..., Vₙ}: a set of *endogenous* (internal, modeled) variables. These are the variables we care about.
- **F** = {f₁, ..., fₙ}: a set of *structural equations*, one for each endogenous variable:

  Vᵢ = fᵢ(PAᵢ, Uᵢ)

  where PAᵢ ⊆ V \ {Vᵢ} are the endogenous parents of Vᵢ, and Uᵢ ⊆ U are the exogenous inputs.

Each structural equation is a *deterministic functional relationship*, not a statistical regression. It encodes the *mechanism* by which Vᵢ is generated from its direct causes. The randomness in the model comes entirely from the exogenous variables U.

**Example**: Consider a model of drug effect.

- U_Z: genetic predisposition (unobserved)
- Z: health status (affected by genetics)
- W: drug taken (may depend on health status)
- Y: recovery (depends on drug and health)

Structural equations:
- Z = f_Z(U_Z)                  [health status determined by genetics]
- W = f_W(Z, U_W)               [treatment decision based on health]
- Y = f_Y(W, Z, U_Y)            [outcome depends on treatment and health]

The associated DAG has edges: Z → W, Z → Y, W → Y.

### DAGs: Definition and Terminology

A Directed Acyclic Graph (DAG) G = (V, E) associated with an SCM has:
- Nodes V corresponding to the endogenous variables.
- Directed edges E encoding the direct causal relationships (Vⱼ → Vᵢ if Vⱼ ∈ PAᵢ).
- **Acyclicity**: there are no directed cycles. A variable cannot be its own ancestor. This rules out feedback loops (though extensions like cyclic SCMs exist for special cases).

**Terminology:**
- **Parent**: Vⱼ is a parent of Vᵢ if there is a direct edge Vⱼ → Vᵢ.
- **Child**: Vᵢ is a child of Vⱼ if Vⱼ → Vᵢ.
- **Ancestor**: Vⱼ is an ancestor of Vᵢ if there is a directed path from Vⱼ to Vᵢ.
- **Descendant**: Vᵢ is a descendant of Vⱼ if Vⱼ is an ancestor of Vᵢ.
- **Path**: A sequence of adjacent nodes connected by edges (which may point in any direction along the path).

### Three Fundamental Graph Structures

Every path in a DAG is composed of three elementary building blocks. Understanding these is essential for d-separation.

**1. Chain (Mediation): A → B → C**

A causes B, which causes C. B is a *mediator*. Information flows from A to C through B. If we condition on B (hold B fixed), the flow from A to C is *blocked*. Example: Smoking → Tar in lungs → Cancer. Conditioning on tar blocks the path from smoking to cancer.

**2. Fork (Common Cause / Confounding): A ← B → C**

B is a *common cause* of A and C. Even though A does not cause C (or vice versa), they are associated because they share a common cause. Information flows between A and C through B. If we condition on B, the flow is *blocked* — the spurious association disappears. Example: Temperature → Ice Cream Sales, Temperature → Drowning. Conditioning on temperature removes the spurious association.

**3. Collider (Common Effect): A → B ← C**

A and B both cause C. C is a *collider* on the path A → C ← B. Unlike chains and forks, this path is *blocked by default*. A and C are marginally independent. However, if we condition on B (or any descendant of B), the path is *opened*, creating a spurious association between A and C. This is called **collider bias** or **selection bias** (or "explaining away" / Berkson's paradox). Example: Talent → Celebrity ← Funding. Among celebrities, talent and funding become negatively correlated.

### d-Separation

**d-separation** (directional separation) is the graphical criterion for determining whether two sets of variables are conditionally independent given a third set, according to the DAG.

**Definition**: A path p between nodes X and Y is **blocked** by a set of nodes Z if:
1. p contains a chain A → B → C or a fork A ← B → C, and the middle node B is in Z (conditioning blocks the path), OR
2. p contains a collider A → B ← C, and neither B nor any descendant of B is in Z (not conditioning keeps it blocked).

Two sets of nodes X and Y are **d-separated** by Z if *every* path between any node in X and any node in Y is blocked by Z.

**d-separation implies conditional independence**: If X and Y are d-separated by Z in the DAG, then X ⊥ Y | Z in every distribution compatible with the DAG. (This is the *global Markov property*.)

The converse — "d-connected implies conditional dependence" — holds generically (for "almost all" parameterizations) but not universally. There can be pathological parameter choices that create unexpected independencies not implied by the graph (called "unfaithful" distributions).

### The Backdoor Criterion

The **backdoor criterion** gives sufficient conditions for identifying the causal effect of X on Y by adjustment.

**Definition**: A set of variables Z satisfies the backdoor criterion relative to (X, Y) if:
1. No node in Z is a descendant of X.
2. Z blocks every path between X and Y that contains an arrow *into* X (i.e., every "backdoor path").

**Backdoor Adjustment Formula**: If Z satisfies the backdoor criterion, then:

P(Y | do(X = x)) = Σ_z P(Y | X = x, Z = z) P(Z = z)

This connects the interventional distribution (the effect of *setting* X) to observable conditional distributions.

**Intuition**: Backdoor paths are paths from X to Y that start with an arrow pointing *into* X. These paths represent confounding — they create spurious associations between X and Y that are not due to the causal effect of X on Y. Blocking these paths (by conditioning on appropriate variables) eliminates the confounding.

### The Frontdoor Criterion

When the backdoor criterion cannot be satisfied (because confounders are unmeasured), the **frontdoor criterion** sometimes provides an alternative identification strategy.

**Setup**: Consider X → M → Y with an unmeasured confounder U affecting both X and Y (U → X and U → Y). We cannot adjust for U (it is unmeasured), so the backdoor criterion fails. But if M is a mediator that fully mediates the effect of X on Y, and there is no direct edge from X to Y or from U to M, the frontdoor criterion applies.

**Conditions**: A set M satisfies the frontdoor criterion relative to (X, Y) if:
1. X blocks all directed paths from M to Y... [correction: M intercepts all directed paths from X to Y]
2. There is no unblocked backdoor path from X to M.
3. All backdoor paths from M to Y are blocked by X.

**Frontdoor Adjustment Formula**:

P(Y | do(X = x)) = Σ_m P(M = m | X = x) Σ_{x'} P(Y | X = x', M = m) P(X = x')

**Intuition**: We decompose the problem into two steps. First, we get the effect of X on M (which is unconfounded). Second, we get the effect of M on Y (adjusting for X as a confounder of M → Y, which we *can* observe). We then combine these two steps.

**Classic example**: Smoking → Tar → Cancer, with a genetic confounder U → Smoking and U → Cancer. We cannot measure U, but we can measure tar deposits. The effect of smoking on cancer can be identified through tar.

### Do-Calculus (Pearl's Framework)

The **do-operator** formalizes intervention. P(Y | do(X = x)) is the distribution of Y when X is *set to* (intervened upon to be) x, as opposed to merely *observed* to be x. The do-operator is defined via the SCM: intervening on X means replacing the structural equation for X with X = x and propagating forward.

**Graphically**: The intervention do(X = x) corresponds to *removing all incoming edges to X* in the DAG (since X is no longer determined by its parents — it is set externally) and setting X = x. The resulting graph is called the *mutilated graph* G̅ₓ.

**Do-calculus** consists of three rules that allow us to manipulate expressions involving do-operators, potentially reducing them to expressions involving only observed quantities. Let G̅ₓ denote the graph with incoming edges to X removed, and G_X̲ denote the graph with outgoing edges from X removed:

**Rule 1 (Insertion/deletion of observations)**:
P(Y | do(X), Z, W) = P(Y | do(X), W) if Y ⊥ Z | X, W in G̅ₓ (the graph with incoming edges to X removed, keeping the intervened version).

**Rule 2 (Action/observation exchange)**:
P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if Y ⊥ Z | X, W in G̅ₓ with outgoing edges from Z removed (G̅ₓ,Z̲).

**Rule 3 (Insertion/deletion of actions)**:
P(Y | do(X), do(Z), W) = P(Y | do(X), W) if Y ⊥ Z | X, W in G̅ₓ with all edges into Z removed *except* edges into Z from nodes that are not ancestors of any W-node in G̅ₓ.

These three rules, together with standard probability axioms, are **complete** for identification: if a causal effect is identifiable from the DAG, it can be derived using do-calculus. If do-calculus cannot derive it, the effect is not identifiable from the given assumptions.

The backdoor and frontdoor criteria are both derivable as special cases of do-calculus.

---

## 2.3 Mathematical Formulation

### Formal Definition of d-Separation

Let G = (V, E) be a DAG. Let X, Y, and Z be disjoint subsets of V.

A path p = (v₁, v₂, ..., vₖ) (undirected, meaning edges can point in any direction) is **active** given Z if:
- For every non-collider on p (i.e., every node vᵢ such that the path does not converge at vᵢ with both edges pointing into vᵢ), vᵢ ∉ Z.
- For every collider on p (i.e., every node vᵢ such that both adjacent edges on p point into vᵢ: vᵢ₋₁ → vᵢ ← vᵢ₊₁), either vᵢ ∈ Z or some descendant of vᵢ is in Z.

X and Y are d-separated by Z if there is no active path between any node in X and any node in Y.

### The Truncated Factorization Formula

An SCM induces a joint distribution over V that factorizes according to the DAG:

P(v₁, v₂, ..., vₙ) = ∏ᵢ P(vᵢ | paᵢ)

where paᵢ denotes the values of the parents of Vᵢ.

Under the intervention do(X = x), we *remove* the factor for X from the product (because X is no longer determined by its parents) and set X = x:

P(v₁, ..., vₙ | do(X = x)) = ∏_{i: Vᵢ ≠ X} P(vᵢ | paᵢ) |_{X=x}    if consistent with X = x
                             = 0                                        otherwise

This is the **truncated factorization formula** (also called the g-formula in the causal DAG context). The backdoor adjustment formula can be derived from this.

### Deriving the Backdoor Adjustment from the Truncated Factorization

Consider the DAG: Z → X → Y and Z → Y (Z is a confounder). The joint distribution factorizes as:

P(Z, X, Y) = P(Z) P(X | Z) P(Y | X, Z)

Under do(X = x), we remove P(X | Z):

P(Z, Y | do(X = x)) = P(Z) P(Y | X = x, Z)

Marginalizing over Z:

P(Y | do(X = x)) = Σ_z P(Y | X = x, Z = z) P(Z = z)

This is the backdoor adjustment formula, derived from first principles.

---

## 2.4 Worked Example: d-Separation

Consider the DAG:

```
A → B → D
A → C → D
B → E ← C
```

Question: Is A d-separated from D given {B}?

**Enumerate all paths from A to D:**

Path 1: A → B → D
- B is a non-collider (chain). B ∈ {B}, so this path is **blocked**.

Path 2: A → C → D
- C is a non-collider (chain). C ∉ {B}, so this path is **active**.

Since Path 2 is active, A and D are **not** d-separated by {B}. A and D are dependent given B.

Question: Is A d-separated from D given {B, C}?

Path 1: A → B → D — blocked (B ∈ {B, C}).
Path 2: A → C → D — blocked (C ∈ {B, C}).

Both paths are blocked, so A ⊥ D | {B, C}. ✓

Question: Is B independent of C given nothing?

Path: B ← A → C — A is a non-collider (fork). A ∉ {}, so this path is active. B and C are dependent (marginally). ✓

Path: B → E ← C — E is a collider. E ∉ {} and no descendant of E is in {}, so this path is blocked. ✓

Overall: Only the first path matters. B and C are dependent.

Question: Is B independent of C given {E}?

Path: B ← A → C — active (A not conditioned on).
Path: B → E ← C — E is a collider, and E ∈ {E}, so this path is now **active**.

Both paths are active. B and C are dependent given E (the collider path opens up).

---

## 2.5 Relevance to Machine Learning Practice

**Feature selection for causal estimation**: The DAG tells you exactly which covariates to include in your adjustment set. Including a collider introduces bias; excluding a confounder leaves bias. Standard feature selection methods (e.g., based on predictive accuracy or mutual information) do not distinguish these cases.

**Detecting and handling confounding in observational ML**: When you train a model on observational data and want to make causal claims (e.g., "this feature *causes* churn"), you need to know the causal structure. A DAG encodes your assumptions transparently and lets others critique them.

**Transfer learning and distribution shift**: Structural causal models formalize why certain relationships are stable across domains (causal mechanisms) while others are not (spurious correlations). If the mechanism P(Y | direct causes of Y) is invariant across environments, models that learn causal relationships generalize better.

**Algorithmic fairness**: DAGs distinguish *direct* effects (X → Y), *indirect* effects (X → M → Y), and *spurious* associations (X ← U → Y). Different fairness criteria correspond to blocking different paths in the DAG.

---

## 2.6 Common Pitfalls & Misconceptions

**Pitfall 1: "Adjust for everything."** This is perhaps the most common mistake. Adjusting for a collider introduces bias. Adjusting for a mediator estimates something different from the total causal effect. Only variables satisfying the backdoor criterion should be conditioned on.

**Pitfall 2: Confusing DAGs with statistical models.** A DAG is a *qualitative* causal assumption, not a fitted model. The edges represent *direct causal mechanisms*, not statistical associations. Two variables can be statistically associated without a direct edge (via a common cause or a chain).

**Pitfall 3: Treating DAGs as discovered from data alone.** While causal discovery algorithms exist, DAGs fundamentally encode *domain knowledge* and *assumptions*. No amount of observational data can, in general, distinguish between X → Y and Y → X without additional assumptions (like temporal ordering). DAGs from domain knowledge should be the starting point; data-driven discovery is a supplement.

**Pitfall 4: Forgetting about unmeasured confounders.** A DAG can include unmeasured variables (typically shown as bidirected edges). If you omit a confounder from the DAG because you did not measure it, your analysis implicitly assumes it does not exist. Always ask: "Are there plausible unmeasured common causes?"

**Pitfall 5: Conflating d-separation with independence.** d-separation implies conditional independence in all distributions compatible with the DAG. But the converse fails for "unfaithful" distributions — rare parameterizations where independencies arise by coincidence. In practice, faithfulness is usually reasonable, but it is good to be aware of the gap.

**Pitfall 6: Misapplying the frontdoor criterion.** The frontdoor criterion requires very specific conditions — no direct effect of X on Y bypassing M, no confounding of X → M, and M fully mediating. These conditions rarely hold exactly in practice, limiting the applicability of the frontdoor criterion.

---

# 3. Identification and Estimation

## 3.1 Motivation & Intuition

### Why Separate "Identification" from "Estimation"?

Causal inference has two distinct steps that beginners often conflate:

1. **Identification**: Can the causal effect be expressed as a function of observed data? This is a *logical/mathematical* question that depends on the causal structure (encoded in the DAG or assumed via unconfoundedness). No amount of data helps here — identification is about whether the effect is *in principle* recoverable.

2. **Estimation**: Given that the effect is identified, how do we *compute* it from finite data? This is a *statistical* question involving estimators, bias, variance, and convergence rates.

If a causal effect is not identified, no estimator — no matter how sophisticated — can consistently recover it. You must either change your assumptions, find additional data, or accept that the question cannot be answered from the available evidence.

### Concrete Analogy

Imagine you want to weigh an object, but your scale is missing. Identification asks: "Is there any procedure using available instruments that yields the weight?" If you have a balance and reference weights, the answer is yes — you have identified a measurement procedure. Estimation asks: "How precisely can we execute this procedure?" Even with the right procedure, measurement error exists — that is the estimation problem.

In causal inference: the adjustment formula P(Y | do(X)) = Σ_z P(Y | X, Z) P(Z) is an *identification result*. The specific method used to compute this from finite data — regression, matching, weighting — is an *estimation choice*.

---

## 3.2 Conceptual Foundations

### Average Treatment Effect (ATE)

The ATE was defined earlier:

τ_ATE = E[Y(1)] − E[Y(0)]

This is the primary estimand in most causal analyses. Under unconfoundedness and positivity, it is identified by the adjustment formula.

### Conditional Average Treatment Effect (CATE)

τ(x) = E[Y(1) − Y(0) | X = x]

CATE captures treatment effect heterogeneity. It is the building block for *personalized* treatment decisions. In ML, estimating CATE is the goal of methods like causal forests, Bayesian additive regression trees (BART), and various meta-learners.

The ATE is the expectation of the CATE:

τ_ATE = E_X[τ(X)]

### Propensity Score Methods

**Propensity Score Matching (PSM):**

The idea: for each treated unit, find one or more control units with similar propensity scores, and compare their outcomes. By matching on the propensity score, the matched groups should be approximately balanced on all observed covariates (by the balancing property of the propensity score).

Procedure:
1. Estimate the propensity score e(x) = P(W = 1 | X = x), typically via logistic regression or a flexible model (gradient boosting, etc.).
2. For each treated unit i, find the nearest control unit j such that |e(Xᵢ) − e(Xⱼ)| is minimized.
3. Estimate the ATT as the average difference in outcomes between matched pairs:
   τ̂_ATT = (1/N₁) Σ_{i: Wᵢ=1} [Yᵢ − Y_{j(i)}]
   where j(i) is the matched control for treated unit i.

Matching can be 1:1 (each treated unit gets one match), 1:k (multiple matches), or with replacement (a control unit can be used multiple times). Caliper matching restricts matches to be within a maximum distance.

**Limitations of matching:**
- Discards unmatched units, reducing sample size.
- The order of matching matters (with replacement mitigates this).
- Matching on the estimated propensity score (rather than the true score) introduces additional variability.
- Only addresses measured confounders; does not help with unmeasured confounding.
- Sensitive to the propensity score model specification.

### Inverse Probability Weighting (IPW)

Instead of matching, we can reweight the observed data so that the weighted sample mimics a randomized experiment.

**Derivation:**

Under unconfoundedness and positivity:

E[Y(1)] = E[E[Y(1) | X]] = E[E[Y | W = 1, X]]

Using the identity E[Y · 1(W=1) / e(X)] = E[Y(1)] (by iterated expectations):

E[Y(1)] = E[Y · W / e(X)]

Similarly: E[Y(0)] = E[Y · (1 − W) / (1 − e(X))]

The **IPW estimator** (Horvitz-Thompson estimator) for the ATE is:

τ̂_IPW = (1/N) Σᵢ [Wᵢ Yᵢ / ê(Xᵢ) − (1 − Wᵢ) Yᵢ / (1 − ê(Xᵢ))]

where ê is the estimated propensity score.

**Intuition**: Treated units with low propensity scores (who "shouldn't have been treated" based on their covariates) get high weights — they are informative about what treatment does to units that usually do not receive it. Control units with high propensity scores (who "should have been treated") also get high weights.

**The Hájek (normalized) estimator**:

τ̂_Hájek = [Σᵢ Wᵢ Yᵢ / ê(Xᵢ)] / [Σᵢ Wᵢ / ê(Xᵢ)] − [Σᵢ (1−Wᵢ) Yᵢ / (1−ê(Xᵢ))] / [Σᵢ (1−Wᵢ) / (1−ê(Xᵢ))]

This normalizes the weights to sum to 1 within each group, which often reduces variance at the cost of a small bias.

**Problems with IPW:**
- **Extreme weights**: When ê(x) is close to 0 or 1 (positivity violation), weights explode, leading to enormous variance. A single unit can dominate the entire estimate.
- **Model dependence**: If the propensity score model is misspecified, the weights are wrong, and the estimate is biased.
- **Finite-sample instability**: Even when positivity holds in theory, practical violations (near-zero propensity scores) are common.

**Mitigations**: Weight trimming (capping extreme weights), stabilized weights (multiplying by marginal treatment probability), or overlap weighting (weighting by the probability of being assigned to the *opposite* group).

### Doubly Robust Estimators

The **augmented inverse probability weighting (AIPW)** estimator combines outcome modeling and propensity weighting. It is "doubly robust" because it is consistent if *either* the propensity score model *or* the outcome model is correctly specified (though not necessarily both).

**Formula:**

τ̂_AIPW = (1/N) Σᵢ { [Wᵢ Yᵢ / ê(Xᵢ) − (Wᵢ − ê(Xᵢ)) / ê(Xᵢ) · μ̂₁(Xᵢ)]
                     − [(1−Wᵢ) Yᵢ / (1−ê(Xᵢ)) + (Wᵢ − ê(Xᵢ)) / (1−ê(Xᵢ)) · μ̂₀(Xᵢ)] }

where μ̂₁(x) = Ê[Y | W = 1, X = x] and μ̂₀(x) = Ê[Y | W = 0, X = x] are estimated outcome regression functions.

**Simplified form:** Let μ̂(w, x) denote the estimated outcome model. The AIPW estimator can be written as:

τ̂_AIPW = (1/N) Σᵢ { [μ̂₁(Xᵢ) − μ̂₀(Xᵢ)]
                     + Wᵢ / ê(Xᵢ) · [Yᵢ − μ̂₁(Xᵢ)]
                     − (1−Wᵢ) / (1−ê(Xᵢ)) · [Yᵢ − μ̂₀(Xᵢ)] }

**Intuition**: The first term is the regression-based estimate (plug-in). The second and third terms are *bias corrections* using the propensity score. If the outcome model is perfect (Yᵢ − μ̂(Wᵢ, Xᵢ) = 0 for all i), the correction terms vanish and the estimator reduces to the regression imputation estimator. If the propensity model is perfect, the IPW component corrects for any misspecification in the outcome model.

**Properties:**
- Doubly robust: consistent if either model is correct.
- Semiparametrically efficient: achieves the lowest possible asymptotic variance among regular estimators when both models are correctly specified.
- Can be combined with machine learning (cross-fitting / sample splitting) to use flexible models while maintaining valid inference — this is the **Debiased Machine Learning (DML)** framework of Chernozhukov et al.

---

## 3.3 Mathematical Formulation

### IPW Derivation in Detail

We want to show: E[Y · W / e(X)] = E[Y(1)].

E[Y · W / e(X)] = E[E[Y · W / e(X) | X]]                   (iterated expectations)
                 = E[(1/e(X)) · E[Y · W | X]]                (e(X) is a function of X)
                 = E[(1/e(X)) · E[Y(1) · W | X]]             (consistency: Y = Y(1) when W = 1)
                 = E[(1/e(X)) · E[Y(1) | X] · E[W | X]]      (ignorability: Y(1) ⊥ W | X)
                 = E[(1/e(X)) · E[Y(1) | X] · e(X)]
                 = E[E[Y(1) | X]]
                 = E[Y(1)]                                     ∎

### Doubly Robust Property: Proof Sketch

Let μ₁(x) = E[Y | W = 1, X = x] (true outcome model) and e(x) = P(W = 1 | X = x) (true propensity).

Define the AIPW influence function:

φ(Y, W, X) = μ₁(X) + W/e(X) · [Y − μ₁(X)]

Consider E[φ]:

E[φ] = E[μ₁(X)] + E[W/e(X) · (Y − μ₁(X))]
     = E[μ₁(X)] + E[E[W/e(X) · (Y − μ₁(X)) | X]]
     = E[μ₁(X)] + E[(1/e(X)) · E[W(Y − μ₁(X)) | X]]
     = E[μ₁(X)] + E[(1/e(X)) · e(X) · E[Y − μ₁(X) | W=1, X]]
     = E[μ₁(X)] + E[E[Y | W=1, X] − μ₁(X)]
     = E[μ₁(X)] + 0
     = E[Y(1)]

Now suppose we use estimated models μ̂₁ and ê. The estimator is still consistent if *either*:
- μ̂₁ → μ₁ (outcome model is consistent): the correction term vanishes.
- ê → e (propensity model is consistent): the IPW part correctly reweights any outcome model errors.

If both converge, we additionally get semiparametric efficiency. This double robustness is the key advantage.

---

## 3.4 Worked Example: IPW Estimation

Consider 6 observations:

| Unit | X    | W | Y | e(X) = P(W=1|X) |
|------|------|---|---|------------------|
| 1    | 0.2  | 1 | 3 | 0.3              |
| 2    | 0.5  | 1 | 5 | 0.5              |
| 3    | 0.8  | 1 | 7 | 0.7              |
| 4    | 0.2  | 0 | 2 | 0.3              |
| 5    | 0.5  | 0 | 4 | 0.5              |
| 6    | 0.8  | 0 | 6 | 0.7              |

**Naive difference in means:**

E[Y | W=1] = (3+5+7)/3 = 5.0
E[Y | W=0] = (2+4+6)/3 = 4.0
Naive estimate: 1.0

**IPW estimate (Horvitz-Thompson):**

Σ Wᵢ Yᵢ / e(Xᵢ) = 3/0.3 + 5/0.5 + 7/0.7 = 10 + 10 + 10 = 30

Σ (1−Wᵢ) Yᵢ / (1−e(Xᵢ)) = 2/0.7 + 4/0.5 + 6/0.3 = 2.857 + 8 + 20 = 30.857

τ̂_HT = (1/6)(30 − 30.857) = −0.143

**IPW estimate (Hájek / normalized):**

Treated weights: 1/0.3 + 1/0.5 + 1/0.7 = 3.333 + 2 + 1.429 = 6.762
Weighted treated mean: 30 / 6.762 = 4.436

Control weights: 1/0.7 + 1/0.5 + 1/0.3 = 1.429 + 2 + 3.333 = 6.762
Weighted control mean: 30.857 / 6.762 = 4.563

τ̂_Hájek = 4.436 − 4.563 = −0.127

Note how the IPW estimate differs from the naive estimate because it upweights units whose treatment assignment was "surprising" given their covariates. Unit 1 (low propensity, got treated) and Unit 6 (low control propensity, got control) receive high weights.

---

## 3.5 Relevance to Machine Learning Practice

**Off-policy evaluation in recommendation and RL**: IPW is the foundation of off-policy evaluation. If a logging policy chose actions with probabilities π₀(a|x), the reward under a target policy π can be estimated as:

V̂(π) = (1/N) Σᵢ [π(Aᵢ|Xᵢ) / π₀(Aᵢ|Xᵢ)] · Rᵢ

This is exactly IPW. The doubly robust version (DR-OPE) adds an outcome model for stability. Extreme importance weights (large π/π₀ ratios) are a constant challenge, addressed by weight clipping, self-normalized estimators, or MAGIC estimators.

**Causal ML for uplift modeling**: Companies like Uber, Microsoft, and booking platforms use AIPW-based methods to estimate heterogeneous treatment effects at scale. The DML framework allows using gradient-boosted trees or neural networks for the nuisance models (propensity score and outcome) while maintaining valid confidence intervals for the treatment effect.

**Treatment effect estimation with high-dimensional data**: When X is high-dimensional (hundreds of features), traditional propensity score methods struggle. Modern approaches combine penalized regression (LASSO), random forests, or deep learning with cross-fitting to estimate nuisance functions, then plug these into AIPW.

### When to Use Each Method

*Propensity score matching*: Simple, interpretable, good for balance diagnostics. Use when you want transparent covariate balance checks and the treatment/control groups have substantial overlap.

*IPW*: Preserves the full sample (no discarding), easy to implement. Use when overlap is good and propensity scores are not extreme.

*Doubly robust / AIPW*: The modern default for causal effect estimation. Use whenever possible — it provides protection against misspecification and achieves efficiency.

*Outcome regression (g-computation)*: Simple to implement, does not require propensity score estimation. But relies entirely on correct outcome model specification and can extrapolate dangerously in regions with poor overlap.

---

## 3.6 Common Pitfalls & Misconceptions

**Pitfall 1: Extreme propensity scores.** When e(x) ≈ 0 or e(x) ≈ 1, IPW weights explode. This is a practical positivity violation. A single unit can dominate the estimate. Always check the distribution of propensity scores for extreme values and consider trimming, truncation, or overlap weights.

**Pitfall 2: Using propensity scores for prediction.** The propensity score is a tool for causal estimation, not prediction. A high propensity score does not mean a unit *should* be treated — it means the unit is *likely to have been* treated.

**Pitfall 3: Thinking "doubly robust" means "always correct."** If *both* the propensity score model and the outcome model are wrong, the AIPW estimator is biased. "Doubly robust" means you need to get *at least one* right. With flexible ML models and cross-fitting, the bar is "consistent at suitable rates," not "exactly correct."

**Pitfall 4: Forgetting model diagnostics.** After propensity score estimation, always check covariate balance (standardized mean differences, overlap of propensity score distributions). If the score does not balance covariates, the model is misspecified.

**Pitfall 5: Confusing ATT and ATE estimands.** Matching on the treated (finding controls for each treated unit) estimates ATT by default, not ATE. To estimate ATE, you need to also match in the reverse direction or use methods that target ATE explicitly.

**Pitfall 6: Ignoring variance estimation.** Because propensity scores are estimated (not known), naive standard errors understate uncertainty. Bootstrap or analytical variance formulas that account for the estimation step are needed. For AIPW, the influence function provides a convenient route to standard errors.

---

# 4. Experimental vs. Observational Data

## 4.1 Motivation & Intuition

### Why Distinguish Experiments from Observations?

The entire difficulty of causal inference arises from **confounding** — systematic differences between treated and untreated groups that affect the outcome. The simplest and most powerful way to eliminate confounding is **randomization**: assign treatment randomly, so that treatment status is independent of all potential confounders, measured or unmeasured.

A **Randomized Controlled Trial (RCT)** achieves this. By randomly assigning units to treatment or control, we guarantee (in expectation) that the two groups are identical in every respect except the treatment. Any difference in outcomes can be attributed to the treatment itself.

But RCTs are not always feasible. Ethical constraints prevent randomizing harmful exposures (we cannot randomly assign people to smoke). Practical constraints make large-scale experiments expensive or slow. Business constraints may prevent withholding a feature from users. And some questions are about naturally occurring phenomena where experiments are impossible (the effect of a natural disaster on economic growth).

When experiments are impossible, we must rely on **observational data** — data generated by the real world, where treatment assignment is not controlled by the researcher. In observational studies, confounding is the default. The methods of Section 3 (matching, IPW, AIPW) are designed specifically for this setting, but they require *assumptions* (unconfoundedness) that cannot be verified from data alone.

Between the gold standard of RCTs and the worst case of purely observational data, there is a middle ground: **quasi-experimental designs** that exploit natural variations in treatment assignment to approximate experimental conditions. These include natural experiments, instrumental variables, and regression discontinuity designs.

### Connection to ML

In industry ML, experimentation (A/B testing) is the primary tool for evaluating product changes. Understanding when A/B tests are valid, when they can mislead, and how to supplement them with observational methods is a core competence of ML scientists. Moreover, many important ML decisions — "Did the new model improve long-term retention?" — involve outcomes that are hard to measure experimentally (long time horizons, network effects), pushing practitioners toward quasi-experimental or observational methods.

---

## 4.2 Conceptual Foundations

### Randomized Controlled Trials (RCTs)

In an RCT, the treatment assignment W is determined by a random mechanism controlled by the experimenter:

P(W = 1 | X, Y(0), Y(1)) = P(W = 1) = p   (under complete randomization)

or more generally, in a stratified/blocked design:

P(W = 1 | X) = p(X)   with known p(X)

This *by design* ensures:
- **Unconfoundedness**: (Y(0), Y(1)) ⊥ W | X holds because treatment is assigned independently of potential outcomes.
- **Known propensity score**: e(x) = P(W = 1 | X = x) is set by the experimenter, not estimated.
- **No unmeasured confounders**: Randomization eliminates *all* confounding, including from variables never measured.

**The simple difference-in-means estimator:**

τ̂ = (1/N₁) Σ_{i:Wᵢ=1} Yᵢ − (1/N₀) Σ_{i:Wᵢ=0} Yᵢ

Under complete randomization, this is unbiased for the ATE:

E[τ̂] = E[Y(1)] − E[Y(0)] = τ_ATE

The variance is:

Var(τ̂) = S₁² / N₁ + S₀² / N₀ − S_τ² / N

where S₁², S₀² are the sample variances of Y(1) and Y(0) respectively, and S_τ² is the variance of the individual treatment effects. Since S_τ² is unobservable, the conservative (Neyman) variance estimator drops this term, giving slightly wider confidence intervals.

**Common issues with RCTs in tech:**

*Network interference*: On platforms where users interact (social networks, marketplaces), treating user A can affect user B's outcomes, violating SUTVA. Solutions include graph-cluster randomization (randomize clusters of connected users) and ego-network randomization.

*Compliance / non-compliance*: Users may not comply with their assignment. In the intent-to-treat (ITT) analysis, we analyze by assigned group regardless of compliance. To estimate the effect of actual treatment, we need instrumental variable methods (the assignment is an instrument for actual treatment).

*Multiple testing*: Running many simultaneous experiments inflates false positive rates. Corrections (Bonferroni, Benjamini-Hochberg) are needed.

*Peeking*: Looking at results repeatedly during the experiment and stopping when significance is reached inflates Type I error. Sequential testing methods (group sequential designs, always-valid p-values) address this.

*Novelty and primacy effects*: Short-term effects may not reflect long-term behavior. Users might engage more with a new feature simply because it is novel.

### Natural Experiments

A **natural experiment** occurs when some natural or institutional process creates variation in treatment assignment that is as-if random, without deliberate experimental manipulation.

**Examples:**
- A policy change affects some geographic regions but not others (difference-in-differences).
- An eligibility cutoff based on a continuous variable (regression discontinuity).
- A random lottery determines who receives a scholarship (natural randomization).
- Vietnam War draft lottery: draft eligibility was determined by birthday, creating quasi-random variation in military service.

Natural experiments are valuable because they approximate randomization without the ethical or practical costs of actual experiments. But they rely on the *as-if random* assumption being credible — this is a judgment call that depends on domain knowledge.

### Instrumental Variables (IV)

An **instrument** Z is a variable that:
1. **Relevance**: Z is correlated with the treatment W (Z affects W).
2. **Exclusion restriction**: Z affects the outcome Y *only through* W, not directly.
3. **Independence**: Z is independent of unmeasured confounders U.

Graphically: Z → W → Y, with U → W and U → Y, but no edge Z → Y and no association between Z and U.

**Why IV is needed**: When there are unmeasured confounders U between W and Y, we cannot identify the causal effect of W on Y by adjusting for observed covariates alone (because U is unobserved). An instrument provides a "back door" around the confounding.

**IV estimation (linear case):**

If Y = α + τW + ε and W = γ₀ + γ₁Z + ν, with E[Zε] = 0 (exclusion) and γ₁ ≠ 0 (relevance):

The **Wald estimator** (for binary Z) is:

τ̂_IV = [E[Y | Z=1] − E[Y | Z=0]] / [E[W | Z=1] − E[W | Z=0]]

Numerator = the effect of Z on Y (the "reduced form").
Denominator = the effect of Z on W (the "first stage").
Ratio = the effect of W on Y, purged of confounding.

**Intuition**: The instrument creates variation in W that is unrelated to the confounders. We isolate this "clean" variation and see how it affects Y. The denominator rescales the reduced form effect by the strength of the instrument's influence on treatment.

**Two-Stage Least Squares (2SLS)**:
1. First stage: Regress W on Z (and covariates) to get predicted values Ŵ.
2. Second stage: Regress Y on Ŵ (and covariates) to estimate τ.

The predicted values Ŵ contain only the variation in W that comes from Z — the "clean" variation. Using Ŵ instead of W removes the confounded component.

**LATE: Local Average Treatment Effect**

With heterogeneous treatment effects and imperfect compliance, IV identifies the **Local Average Treatment Effect (LATE)** — the average effect for *compliers* only (units whose treatment status is influenced by the instrument). This is a narrower estimand than ATE. It is still useful, but one should be clear about what population the estimate applies to.

**Complier types (with binary Z, binary W):**
- *Compliers*: W = 1 when Z = 1, W = 0 when Z = 0 (follow the instrument).
- *Always-takers*: W = 1 regardless of Z.
- *Never-takers*: W = 0 regardless of Z.
- *Defiers*: W = 0 when Z = 1, W = 1 when Z = 0 (do the opposite).

Under the **monotonicity** assumption (no defiers), IV identifies the LATE for compliers.

**Weak instruments**: When Z has a small effect on W (the first stage F-statistic is low), the IV estimate is biased toward the OLS estimate, and confidence intervals are misleading. The rule of thumb is F > 10 in the first stage, though this is conservative.

### Regression Discontinuity Design (RDD)

**Setup**: Treatment assignment is determined by whether a *running variable* X crosses a cutoff c:

W = 1 if X ≥ c
W = 0 if X < c

**Examples**: A scholarship is awarded to students scoring above 80 on an exam. A drug is approved if the efficacy measure exceeds a threshold.

**Key insight**: Units just above and just below the cutoff are almost identical in all respects except treatment. By comparing outcomes for units near the cutoff, we approximate a local randomized experiment.

**Formally**: The RDD estimates:

τ_RDD = lim_{x↓c} E[Y | X = x] − lim_{x↑c} E[Y | X = x]

This is the treatment effect *at the cutoff* — a local effect for units with X ≈ c. It is not the ATE for the full population.

**Estimation**: Fit separate regressions (often local polynomials) on each side of the cutoff, using data within a bandwidth h of c:

- Ê[Y | X = x, W = 1] for x ∈ [c, c + h]
- Ê[Y | X = x, W = 0] for x ∈ [c − h, c)

The treatment effect is the difference in fitted values at x = c.

**Sharp vs. Fuzzy RDD:**

*Sharp RDD*: Treatment is deterministically assigned by the cutoff. P(W = 1 | X = c + ε) = 1, P(W = 1 | X = c − ε) = 0 for any ε > 0.

*Fuzzy RDD*: The cutoff creates a *jump* in the probability of treatment, but not from 0 to 1. Some units above the cutoff do not receive treatment, and some below do. The cutoff acts as an instrument for treatment, and the effect is estimated by IV methods:

τ_fuzzy = [lim_{x↓c} E[Y|X=x] − lim_{x↑c} E[Y|X=x]] / [lim_{x↓c} E[W|X=x] − lim_{x↑c} E[W|X=x]]

This is a LATE for compliers at the cutoff.

**Assumptions:**

- *Continuity*: The potential outcome functions E[Y(0) | X = x] and E[Y(1) | X = x] are continuous at x = c. If there is a discontinuity in the outcome that is *not* due to treatment, the design is invalid.
- *No manipulation*: Units cannot precisely manipulate their running variable to cross the cutoff. If students can manipulate their exam score to just exceed 80, the "near-the-cutoff" groups are no longer comparable. McCrary density tests check for bunching at the cutoff.
- *No other discontinuities*: No other relevant variable changes discontinuously at the cutoff.

---

## 4.3 Mathematical Formulation

### RDD: Formal Identification

Under the continuity assumption:

τ_RDD = E[Y(1) − Y(0) | X = c]

Proof:

lim_{x↓c} E[Y | X = x] = lim_{x↓c} E[Y(1) | X = x]    (since W = 1 for X > c)
                         = E[Y(1) | X = c]                 (by continuity)

lim_{x↑c} E[Y | X = x] = lim_{x↑c} E[Y(0) | X = x]    (since W = 0 for X < c)
                         = E[Y(0) | X = c]                 (by continuity)

Therefore:

lim_{x↓c} E[Y | X = x] − lim_{x↑c} E[Y | X = x] = E[Y(1) − Y(0) | X = c] ∎

### IV: Formal Identification of LATE

Under relevance, exclusion, independence, and monotonicity:

τ_LATE = E[Y(1) − Y(0) | Complier]

= {E[Y | Z = 1] − E[Y | Z = 0]} / {E[W | Z = 1] − E[W | Z = 0]}

Proof sketch:

E[Y | Z = 1] − E[Y | Z = 0]
= E[Y(1) · W(1) + Y(0) · (1 − W(1))] − E[Y(1) · W(0) + Y(0) · (1 − W(0))]
= E[(Y(1) − Y(0)) · (W(1) − W(0))]

Under monotonicity, W(1) − W(0) ∈ {0, 1} (no defiers). It equals 1 for compliers and 0 otherwise.

= E[Y(1) − Y(0) | Complier] · P(Complier)

Similarly: E[W | Z = 1] − E[W | Z = 0] = E[W(1) − W(0)] = P(Complier)

Dividing: {E[Y(1) − Y(0) | Complier] · P(Complier)} / P(Complier) = E[Y(1) − Y(0) | Complier] ∎

---

## 4.4 Worked Example: Regression Discontinuity

A company offers a loyalty bonus to customers whose annual spending exceeds $1,000. We want to estimate the effect of the bonus on next-year retention.

Running variable X = annual spending. Cutoff c = $1,000. Treatment W = receives bonus. Outcome Y = retained next year (0/1).

Suppose we have data:

| Spending ($) | Bonus (W) | Retained (Y) |
|-------------|-----------|--------------|
| 950         | 0         | 0            |
| 970         | 0         | 1            |
| 985         | 0         | 0            |
| 990         | 0         | 1            |
| 995         | 0         | 1            |
| 1005        | 1         | 1            |
| 1010        | 1         | 1            |
| 1020        | 1         | 1            |
| 1030        | 1         | 0            |
| 1050        | 1         | 1            |

Within a narrow bandwidth around $1,000:
- Just below (950–995): 3/5 = 60% retention
- Just above (1005–1050): 4/5 = 80% retention

RDD estimate: 80% − 60% = 20 percentage points increase in retention from the bonus.

This is a local effect *at the cutoff*. Customers spending $500 or $5,000 might respond differently to the bonus.

**Validity checks:**
1. Is there bunching at $1,000? If customers manipulate spending to just exceed the threshold, the estimate is biased. Check the density of spending around the cutoff.
2. Do other covariates change discontinuously at the cutoff? If customer demographics jump at $1,000, something other than the bonus may be driving the difference.
3. Sensitivity to bandwidth: Does the estimate change dramatically if we use $950–$1,050 vs. $980–$1,020? Stability across bandwidths increases credibility.

---

## 4.5 Relevance to Machine Learning Practice

### A/B Testing as the Industry Gold Standard

In tech companies, A/B testing is the primary method for evaluating product changes. The ML practitioner needs to understand:

- **Power analysis**: Before the experiment, determine the sample size needed to detect a meaningful effect. This depends on the baseline metric, the minimum detectable effect (MDE), the significance level (α), and power (1 − β). Underpowered experiments waste resources; overpowered experiments delay launches.
- **Metric selection**: Primary metrics (the one you care about), guardrail metrics (the ones that must not deteriorate), and proxy metrics (faster-moving signals). Choosing the right metrics is a design decision with causal implications.
- **CUPED / variance reduction**: Pre-experiment covariates can be used to reduce the variance of the treatment effect estimator (analogous to ANCOVA). This uses the observational relationship between pre-experiment behavior and the outcome to tighten confidence intervals without introducing bias.

### When RCTs Are Not Enough

*Long-term effects*: An experiment might run for 2 weeks, but you care about 12-month retention. Surrogacy methods (using short-term outcomes to predict long-term effects) and observational follow-up studies are needed.

*Network effects*: In two-sided marketplaces, treating sellers affects buyers. Graph-cluster randomization, switchback experiments (time-based randomization), and synthetic control methods are alternatives.

*Ethical constraints*: You cannot randomly degrade some users' experience. Quasi-experimental methods (difference-in-differences from natural rollouts, RDD from eligibility thresholds) provide alternatives.

### When to Use Which Design

*RCT*: The default when feasible, ethical, and the SUTVA assumption is reasonable.

*IV*: When you have a plausible instrument and unconfoundedness is not credible. Common in economics (distance to college as an instrument for education, draft lottery for military service).

*RDD*: When treatment assignment is based on a continuous score with a cutoff. Common in education (test score thresholds), marketing (spending thresholds), and regulatory settings.

*Diff-in-Diff*: When treatment is rolled out to some groups/regions at a specific time, and parallel trends are plausible.

*Observational methods (matching, IPW, AIPW)*: When no quasi-experimental design is available and you must rely on unconfoundedness.

---

## 4.6 Common Pitfalls & Misconceptions

**Pitfall 1: Trusting A/B tests blindly.** Even randomized experiments can give wrong answers: SUTVA violations (interference), improper randomization unit (randomizing at user level when the feature affects sessions), SRM (Sample Ratio Mismatch indicating a bug in the randomization), instrumentation effects (the experiment infrastructure itself affects behavior).

**Pitfall 2: Generalizing RDD estimates.** RDD gives a *local* estimate at the cutoff. Extrapolating to the full population is not justified. The effect of a bonus on borderline customers ($1,000 spend) may differ from its effect on high-spend or low-spend customers.

**Pitfall 3: Invalid instruments.** The exclusion restriction (Z affects Y only through W) is untestable. Many proposed instruments fail this assumption. For example, "distance to college" as an instrument for education might also affect earnings through neighborhood quality. Always argue for instrument validity on substantive, not statistical, grounds.

**Pitfall 4: Confusing ITT with TOT/LATE.** The intent-to-treat (ITT) effect measures the effect of *assignment* to treatment. The treatment-on-treated (TOT) effect measures the effect of *actually receiving* treatment. With non-compliance, ITT < TOT (in absolute value), and they answer different questions.

**Pitfall 5: P-hacking and multiple comparisons.** Running many experiments, checking many metrics, or using many subgroup analyses inflates false positive rates. Pre-registration of hypotheses and appropriate multiple testing corrections are essential.

**Pitfall 6: Ignoring external validity.** An RCT on one user population at one time may not generalize. Treatment effects can vary across populations, contexts, and time periods. Internal validity (is the estimate correct for this sample?) and external validity (does it apply elsewhere?) are separate considerations.

---

# 5. Causal Discovery

## 5.1 Motivation & Intuition

### Why Discover Causal Structure?

All the methods discussed so far — adjustment, matching, IPW, IV, RDD — require *knowing* (or assuming) the causal structure. The DAG is an *input*, not an *output*. But where does the DAG come from?

In many domains, domain experts can propose a plausible DAG. A doctor knows that smoking causes tar buildup, which causes cancer. An economist knows that education affects earnings. But in high-dimensional settings with dozens or hundreds of variables, constructing a DAG from domain knowledge alone is infeasible. And in novel domains (genomics, complex systems, neural circuits), the causal structure may be genuinely unknown.

**Causal discovery** is the task of inferring causal structure from data. Given observations of many variables, can we determine which variables cause which? This is an ambitious goal — and, as we will see, one that faces fundamental limitations.

### A Simple Intuition

Consider three variables X, Y, Z. Suppose we observe:
- X and Y are correlated.
- Y and Z are correlated.
- X and Z are conditionally independent given Y.

This pattern is consistent with the chain X → Y → Z (or X ← Y ← Z), or the fork X ← Y → Z. These are **Markov equivalent** — they imply the same set of conditional independencies. We cannot distinguish between them from observational data alone.

But if we observed:
- X and Z are marginally independent.
- X and Z are dependent *given* Y.

This pattern uniquely identifies the collider X → Y ← Z. No other structure produces this pattern. This is a case where the causal direction *is* identifiable from data.

The fundamental insight: observational data can narrow down the set of possible DAGs (the "Markov equivalence class"), but usually cannot identify a single unique DAG. Additional assumptions (temporal ordering, functional form restrictions, interventional data) are needed to further pin down the causal structure.

---

## 5.2 Conceptual Foundations

### Markov Equivalence Classes

Two DAGs are **Markov equivalent** if they imply the same set of conditional independencies. A **Markov equivalence class** is a set of all DAGs that are Markov equivalent to each other.

Markov equivalent DAGs share the same *skeleton* (the undirected graph) and the same *v-structures* (colliders A → C ← B where A and B are not adjacent). The class can be represented by a **Completed Partially Directed Acyclic Graph (CPDAG)**: a graph with some directed edges (where all DAGs in the class agree on the direction) and some undirected edges (where the direction varies across the class).

**Example**: X → Y → Z and X ← Y ← Z and X ← Y → Z all have the same skeleton (X — Y — Z) and no v-structures. They form a Markov equivalence class represented by the CPDAG X — Y — Z.

But X → Y ← Z has a v-structure at Y and is in its own equivalence class. Its CPDAG is X → Y ← Z (fully directed).

### Constraint-Based Methods: The PC Algorithm

The **PC algorithm** (named after Peter Spirtes and Clark Glymour, or alternatively "PC" for its computational nature) is the foundational constraint-based causal discovery method.

**Assumptions:**
1. **Causal Markov Condition**: Every variable is independent of its non-descendants given its parents.
2. **Faithfulness**: Every conditional independence in the data is entailed by the DAG (there are no "accidental" independencies due to parameter cancellations).
3. **Causal sufficiency**: There are no unmeasured common causes.

**Algorithm:**

*Phase 1: Skeleton discovery*
1. Start with a complete undirected graph (every pair of variables connected).
2. For each pair (X, Y), test if X ⊥ Y. If so, remove the edge.
3. For each remaining pair (X, Y), test if X ⊥ Y | Z for each single variable Z. If so, remove the edge and record Z as the separating set.
4. For each remaining pair (X, Y), test if X ⊥ Y | {Z₁, Z₂} for each pair of variables. If so, remove the edge.
5. Continue increasing the size of the conditioning set until no more edges can be removed.

The key efficiency gain: we only test conditioning sets up to the current degree of each node, so as edges are removed, the conditioning sets get smaller.

*Phase 2: Orient v-structures*
For each unshielded triple X — Z — Y (X and Y not adjacent, both adjacent to Z): if Z was *not* in the separating set of X and Y (the set that made them independent), then orient the triple as X → Z ← Y (it is a collider / v-structure).

Intuition: If Z were a non-collider on the path X — Z — Y (i.e., X ← Z → Y or X → Z → Y), conditioning on Z would *block* the path, making X and Y conditionally independent given Z. So Z would appear in the separating set. If Z is *not* in the separating set, it must be a collider.

*Phase 3: Orient remaining edges by propagation*
Apply orientation rules to avoid creating new v-structures or cycles:
- If X → Z — Y, and X and Y are not adjacent, orient as Z → Y (to avoid a new v-structure X → Z ← Y, which would be inconsistent if not detected in Phase 2).
- If there is a directed path X → ... → Y, and an undirected edge X — Y, orient as X → Y (to avoid a cycle).

The result is a CPDAG representing the Markov equivalence class.

**Conditional independence testing**: The practical workhorse of the PC algorithm. For continuous data, Fisher's z-test for partial correlations (testing if the partial correlation is zero) is standard. For discrete data, chi-squared or G-tests are used. For nonlinear relationships, kernel-based tests (HSIC) or conditional mutual information estimators are available.

### Score-Based Methods

Instead of testing conditional independencies, score-based methods search over the space of DAGs to find the one that best fits the data according to a scoring criterion.

**Scoring criteria:**

*Bayesian Information Criterion (BIC)*:

BIC(G) = −2 log L(θ̂_G; D) + |θ_G| log N

where L is the maximized likelihood, |θ_G| is the number of free parameters (penalizing complexity), and N is the sample size. Lower BIC is better.

*Bayesian scores*: The marginal likelihood P(D | G) = ∫ P(D | θ, G) P(θ | G) dθ. This naturally incorporates Occam's razor through the prior and integration.

**Search strategies:**

*Greedy Equivalence Search (GES)*: A two-phase algorithm.
- Forward phase: Start with the empty graph. Greedily add edges that most improve the score, respecting acyclicity.
- Backward phase: Greedily remove edges that improve the score.

GES is consistent (recovers the true CPDAG in the large-sample limit) under the same assumptions as the PC algorithm (faithfulness, causal sufficiency).

*Other approaches*: Hill climbing, tabu search, integer linear programming formulations, and hybrid methods that combine constraint-based skeleton discovery with score-based orientation.

### Functional Causal Models and Identifiability Beyond Markov Equivalence

Under additional assumptions about the *functional form* of structural equations, causal direction can sometimes be identified even within a Markov equivalence class.

**Linear Non-Gaussian Acyclic Model (LiNGAM)**: If the structural equations are *linear* and the exogenous noise terms are *non-Gaussian*, the full DAG is identifiable from observational data (not just the equivalence class). This is because the asymmetry of non-Gaussian distributions breaks the symmetry between X → Y and Y → X.

**Additive Noise Models (ANMs)**: Y = f(X) + ε, where ε ⊥ X. If f is nonlinear, the causal direction X → Y is identifiable: one can test whether the residuals of regressing Y on X are independent of X, and whether the residuals of regressing X on Y are independent of Y. The "correct" direction will show independent residuals; the wrong direction will not (generically).

**Causal discovery from interventional data**: If we have data from both observational settings and experimental interventions on specific variables, the Markov equivalence class can often be narrowed or fully resolved. Each intervention provides additional constraints.

---

## 5.3 Mathematical Formulation

### Formal Statement of the PC Algorithm

**Input**: Data D over variables V = {V₁, ..., Vₚ}, significance level α for independence tests.

**Output**: A CPDAG G.

**Procedure:**

Step 0: Form the complete undirected graph C on V.

Step 1: (Skeleton)
```
n = 0
repeat:
  for each adjacent pair (Vᵢ, Vⱼ) in C:
    for each subset S ⊆ adj(Vᵢ) \ {Vⱼ} with |S| = n:
      if Vᵢ ⊥ Vⱼ | S (at level α):
        remove edge Vᵢ — Vⱼ from C
        record Sepset(Vᵢ, Vⱼ) = S
        break
  n = n + 1
until n > max degree in C
```

Step 2: (V-structure orientation)
```
for each unshielded triple (Vᵢ, Vₖ, Vⱼ) (Vᵢ—Vₖ—Vⱼ, Vᵢ not adjacent to Vⱼ):
  if Vₖ ∉ Sepset(Vᵢ, Vⱼ):
    orient as Vᵢ → Vₖ ← Vⱼ
```

Step 3: (Orientation propagation)
Apply Meek's rules R1–R3 exhaustively to orient remaining edges without creating new v-structures or cycles.

**Complexity**: In the worst case (dense graph), the number of independence tests is exponential in the number of variables. For sparse graphs (bounded degree), the complexity is polynomial.

### BIC Score Decomposition

For DAGs with local structure, the BIC score decomposes:

BIC(G) = Σᵢ BIC_local(Vᵢ, PA_G(Vᵢ))

where BIC_local(Vᵢ, PA) = −2 log L(θ̂; Vᵢ | PA) + dim(PA) · log N

This decomposition is crucial for efficient search: adding or removing a single edge only affects the local scores of the child node, not the entire graph.

---

## 5.4 Worked Example: PC Algorithm

Consider four variables: X₁, X₂, X₃, X₄ with the true DAG:

X₁ → X₂ → X₃, X₁ → X₃, X₄ → X₃

(X₃ has three parents: X₂, X₁, and X₄.)

True conditional independencies (assuming faithfulness):
- X₁ ⊥ X₄ (no path connecting them that is active marginally? Actually X₁ → X₃ ← X₄ makes X₃ a collider, so the path X₁ → X₃ ← X₄ is blocked marginally. Yes, X₁ ⊥ X₄.)
- X₂ ⊥ X₄ | X₁ (path X₂ ← X₁ → X₃ ← X₄: X₃ is a collider, not conditioned on → blocked. Path X₂ → X₃ ← X₄: X₃ is a collider, not conditioned on → blocked. So X₂ ⊥ X₄ | X₁.)

**Phase 1: Skeleton**

Start with complete graph: X₁—X₂, X₁—X₃, X₁—X₄, X₂—X₃, X₂—X₄, X₃—X₄.

n = 0 (test marginal independencies):
- X₁ ⊥ X₄? YES → remove X₁—X₄. Sepset(X₁, X₄) = ∅.
- All other pairs are marginally dependent (they share active paths).

n = 1 (test conditional on single variables):
- X₂ ⊥ X₄ | X₁? Test: YES → remove X₂—X₄. Sepset(X₂, X₄) = {X₁}.
- (Other tests find no further independencies.)

Skeleton: X₁—X₂, X₁—X₃, X₂—X₃, X₃—X₄.

**Phase 2: V-structure orientation**

Unshielded triples:
- X₂—X₃—X₄ (X₂ and X₄ not adjacent): Check Sepset(X₂, X₄) = {X₁}. Is X₃ in {X₁}? NO → orient as X₂ → X₃ ← X₄. ✓
- X₁—X₃—X₄ (X₁ and X₄ not adjacent): Check Sepset(X₁, X₄) = ∅. Is X₃ in ∅? NO → orient as X₁ → X₃ ← X₄. ✓ (consistent with above)
- X₁—X₂—X₃: X₁ and X₃ are adjacent, so this is *shielded*. Skip.

**Phase 3: Orientation propagation**

After Phase 2: X₁ → X₃ ← X₄, X₂ → X₃ ← X₄ (confirmed), X₁—X₂ still undirected.

Meek's rules: X₁—X₂ and X₂ → X₃. If we orient X₂ → X₁, there is no issue. If we orient X₁ → X₂, that creates a directed path X₁ → X₂ → X₃ alongside the direct edge X₁ → X₃, which is fine (no new v-structure or cycle). Neither direction creates a problem, so X₁—X₂ remains undirected in the CPDAG.

Final CPDAG: X₁ — X₂ → X₃ ← X₄, with X₁ → X₃.

This represents a Markov equivalence class of two DAGs:
1. X₁ → X₂ → X₃ ← X₄, X₁ → X₃
2. X₁ ← X₂ → X₃ ← X₄, X₁ → X₃

The algorithm correctly identifies the colliders but cannot determine the direction of the X₁—X₂ edge from observational data alone.

---

## 5.5 Relevance to Machine Learning Practice

**Feature engineering and selection**: Knowing the causal graph can inform which features to include in a model. Causal features (direct causes of the target) are more robust to distribution shift than spurious correlates.

**Root cause analysis**: In production ML systems, when a metric degrades, causal discovery can help identify which upstream variable changed. For example, if model accuracy dropped, was it due to a data pipeline change, a feature distribution shift, or a label quality issue?

**Scientific discovery**: In genomics, neuroscience, and climate science, causal discovery from large observational datasets can generate hypotheses for experimental validation. The PC algorithm and its variants have been applied to gene regulatory network inference.

**Limitations in practice:**
- Faithfulness violations are more common than often assumed, especially in small samples.
- Conditional independence tests have limited power with finite data, especially with many variables.
- Causal sufficiency (no unmeasured confounders) is a strong assumption. Extensions like the FCI (Fast Causal Inference) algorithm relax this but produce less informative output (Partial Ancestral Graphs instead of CPDAGs).
- Computational cost grows rapidly with the number of variables.

---

## 5.6 Common Pitfalls & Misconceptions

**Pitfall 1: Treating discovered graphs as ground truth.** Causal discovery algorithms produce *hypotheses*, not proven causal relationships. The output depends heavily on the sample size, the independence test, the significance level, and the validity of assumptions (faithfulness, causal sufficiency). Domain knowledge should always be used to validate and supplement algorithmic output.

**Pitfall 2: Ignoring Markov equivalence.** From observational data alone, we can only identify the Markov equivalence class, not a unique DAG. Many analyses implicitly choose one DAG from the class without acknowledging the ambiguity. Interventional data or domain knowledge is needed to resolve the remaining ambiguity.

**Pitfall 3: Conflating correlation patterns with causal structure.** The PC algorithm uses conditional independencies, not correlations. Two highly correlated variables may not be adjacent in the DAG (if they share a common cause). Two weakly correlated variables may be directly causally related.

**Pitfall 4: Applying causal discovery to insufficient data.** Conditional independence tests require substantial sample sizes, especially when conditioning on many variables. With small samples and many variables, the algorithms produce unreliable graphs with many false edges.

**Pitfall 5: Assuming causal sufficiency.** If there are unmeasured confounders, the PC algorithm can produce incorrect results (false edges, wrong orientations). The FCI algorithm handles this by allowing for latent variables, but it produces weaker conclusions.

**Pitfall 6: Ignoring temporal information.** When temporal ordering is available (X measured before Y), it provides powerful constraints (X cannot be caused by Y). Many practitioners ignore this "free" information and rely solely on statistical tests.

---

# 6. Interview Preparation

## Foundational Questions

### Q1. What is the fundamental problem of causal inference?

**Answer:** The fundamental problem of causal inference (Holland, 1986) is that for any individual unit, we can only observe *one* potential outcome — the one corresponding to the treatment actually received. We can never observe both Y(1) and Y(0) for the same unit simultaneously. Therefore, the individual treatment effect τᵢ = Yᵢ(1) − Yᵢ(0) is never directly observable.

This is not a problem of measurement precision or data collection — it is a *logical* impossibility. Even with infinite resources and perfect measurement, you cannot simultaneously observe what happens to the same person when they both take and do not take a drug.

The consequence: all of causal inference is about making *population-level* or *subgroup-level* inferences (ATE, ATT, CATE) by combining observations across units, under assumptions that link the observed data to the counterfactual quantities.

### Q2. Explain SUTVA in the context of an A/B test on a social media platform.

**Answer:** SUTVA requires (1) no interference: each user's outcome depends only on their own treatment assignment, not on others', and (2) no hidden treatment versions: the treatment is well-defined and identical for all treated users.

On a social media platform, both components are often violated:

*Interference*: If user A (in the treatment group, seeing a new feed algorithm) shares content with user B (in the control group), user B's engagement is affected by A's treatment assignment. The content B sees is different because A's behavior changed. B's potential outcome Y_B(0) now depends on A's treatment, violating the "no interference" assumption.

*Hidden treatment versions*: If the new algorithm has different effects depending on network structure (dense vs. sparse friend graph), the "treatment" is effectively different for different users. The average effect becomes an ill-defined average over heterogeneous treatments.

**Consequences**: Standard ATE estimators are biased. The direction of bias depends on the sign and magnitude of spillover effects. If treatment has positive spillovers (treated users improve control users' experience), the experiment *understates* the true effect.

**Solutions**: Graph-cluster randomization (randomize connected clusters to reduce cross-group interaction), ego-network randomization, or switchback designs (time-based randomization where all users see the same treatment at the same time).

### Q3. What is the difference between observational data and experimental data? Why does it matter for causal inference?

**Answer:** In experimental data (RCTs), treatment assignment is controlled by the researcher via randomization. This guarantees that treatment is independent of all confounders, observed and unobserved. In observational data, treatment assignment is determined by the real world — patients choose to take drugs, users self-select into programs — and is typically correlated with confounders.

This distinction matters because in experiments, the simple difference in means is an unbiased estimator of the ATE — no additional assumptions or adjustments are needed. In observational data, the same estimator is biased by confounding. We must rely on assumptions (unconfoundedness, positivity) and adjustment methods (matching, IPW, AIPW) to recover causal effects, and these assumptions are untestable.

The practical consequence: experimental evidence is stronger for causal claims, but observational data is often all we have. The key skill is understanding *which assumptions are needed* and *how credible they are* in a given setting.

### Q4. What is a propensity score, and why is it useful?

**Answer:** The propensity score e(x) = P(W = 1 | X = x) is the conditional probability of receiving treatment given observed covariates. Rosenbaum and Rubin (1983) proved that if unconfoundedness holds given X, it also holds given the scalar e(X). This means we can reduce a high-dimensional covariate adjustment problem to a one-dimensional problem.

This is useful because (1) it avoids the curse of dimensionality in stratification/matching, (2) it provides a single summary of how "likely" each unit was to receive treatment, enabling balance diagnostics, and (3) it enables IPW estimators that reweight the observed data to mimic a randomized experiment.

The propensity score is *not* useful for prediction — it does not predict outcomes. It models treatment *assignment*, not treatment *effect*.

---

## Mathematical Questions

### Q5. Derive the IPW estimator for E[Y(1)] and show it is unbiased.

**Answer:**

We want to show: E[YW/e(X)] = E[Y(1)].

Start from the left side:

E[YW/e(X)]
= E[E[YW/e(X) | X]]                          (iterated expectations)
= E[(1/e(X)) · E[YW | X]]                     (e(X) is determined by X)

Now, E[YW | X] = E[Y · 1(W=1) | X]:

By consistency, when W = 1, Y = Y(1). So:

E[YW | X] = E[Y(1) · W | X]
           = E[Y(1) | X] · P(W = 1 | X)        (by unconfoundedness: Y(1) ⊥ W | X)
           = E[Y(1) | X] · e(X)

Substituting back:

E[(1/e(X)) · E[Y(1) | X] · e(X)]
= E[E[Y(1) | X]]
= E[Y(1)]                                      ∎

The sample analog is τ̂ = (1/N) Σᵢ Wᵢ Yᵢ / ê(Xᵢ), which is consistent for E[Y(1)] when ê is consistent for e.

### Q6. Explain the doubly robust property of the AIPW estimator mathematically. Under what conditions does it fail?

**Answer:**

The AIPW estimator for E[Y(1)] uses both a propensity model ê(x) and an outcome model μ̂₁(x):

φ̂ᵢ = μ̂₁(Xᵢ) + Wᵢ/ê(Xᵢ) · [Yᵢ − μ̂₁(Xᵢ)]

Let the true models be e(x) and μ₁(x). Consider what happens when we use potentially incorrect ẽ and μ̃₁:

E[μ̃₁(X) + W/ẽ(X) · (Y − μ̃₁(X))]

*Case 1: ẽ = e (propensity correct, outcome may be wrong)*

E[μ̃₁(X) + W/e(X) · (Y − μ̃₁(X))]
= E[μ̃₁(X)] + E[E[W/e(X) · (Y − μ̃₁(X)) | X]]
= E[μ̃₁(X)] + E[(1/e(X)) · e(X) · E[Y − μ̃₁(X) | W=1, X]]
= E[μ̃₁(X)] + E[μ₁(X) − μ̃₁(X)]
= E[μ₁(X)] = E[Y(1)] ✓

*Case 2: μ̃₁ = μ₁ (outcome correct, propensity may be wrong)*

E[μ₁(X) + W/ẽ(X) · (Y − μ₁(X))]

Since E[Y − μ₁(X) | W=1, X] = 0 when the outcome model is correct:

= E[μ₁(X)] + E[W/ẽ(X) · (Y − μ₁(X))]
= E[μ₁(X)] + E[E[W · (Y − μ₁(X)) | X] / ẽ(X)]
= E[μ₁(X)] + E[e(X) · 0 / ẽ(X)]
= E[μ₁(X)] = E[Y(1)] ✓

*Failure*: If *both* models are wrong, the estimator is generally biased. The bias depends on the product of the two errors (a cross-term), so if both models are "somewhat right," the bias can be small. This motivates using flexible ML models with cross-fitting.

### Q7. In a regression discontinuity design, what does the continuity assumption formally require, and how would you check it?

**Answer:**

The continuity assumption requires that E[Y(0) | X = x] and E[Y(1) | X = x] are continuous functions of the running variable x at the cutoff c. This means that in the absence of treatment, units just above and just below the cutoff would have similar expected outcomes.

Formally: lim_{x↓c} E[Y(w) | X = x] = E[Y(w) | X = c] for w ∈ {0, 1}.

This assumption would be violated if some *other* discontinuity (a different program, a behavioral change) occurs at the same cutoff.

**Checking:**

1. **McCrary density test / manipulation test**: Check whether the density of the running variable is continuous at the cutoff. A discontinuity (bunching) suggests manipulation — units adjusting their X to be just above or below c.

2. **Covariate balance at the cutoff**: Check whether pre-treatment covariates (age, income, prior behavior) are continuous at the cutoff. If demographics jump discontinuously at c, the assumption is suspect.

3. **Placebo cutoffs**: Estimate the "treatment effect" at fake cutoffs away from c. If significant effects appear at placebo cutoffs, the methodology or data may have issues.

4. **Bandwidth sensitivity**: Check whether the estimate changes substantially with different bandwidths. Extreme sensitivity suggests the local polynomial fit is unreliable.

### Q8. Explain why conditioning on a collider introduces bias. Provide a mathematical example.

**Answer:**

Consider the DAG: X → Z ← Y, where X and Y are independent causes of Z.

Structural equations:
X ~ N(0, 1)
Y ~ N(0, 1)
Z = X + Y + ε, where ε ~ N(0, σ²)

Marginally, X ⊥ Y (they are independent by assumption).

Now condition on Z = z. Given Z = z, we know that X + Y ≈ z (minus noise). So if we observe that X is large, Y must be small (to maintain the sum), and vice versa:

E[Y | X = x, Z = z] ≈ z − x

X and Y become negatively correlated given Z. This is the *explaining away* effect: knowing one cause's contribution "explains" part of the observed effect, making the other cause less likely.

In causal inference, this is problematic because conditioning on Z creates a spurious association between X and Y where none existed. If Z is a descendant of both the treatment and an unrelated variable, adjusting for Z (e.g., including it in a regression) biases the treatment effect estimate.

**Concrete ML example**: Suppose "Hired" (Z) depends on both "Skill" (X) and "Networking" (Y). Among hired employees, skill and networking appear negatively correlated (skilled people who were hired didn't need as much networking). If you adjust for "Hired" when estimating the effect of Skill on job performance, you introduce a spurious association through Networking.

---

## Applied Questions

### Q9. You are an ML scientist at a ride-sharing company. The product team wants to know: "Does offering a 20% discount increase ride frequency?" How would you design and analyze this study?

**Answer:**

**Design (RCT approach):**

1. **Randomization unit**: Randomize at the user level (not ride level) to avoid within-user contamination.

2. **Sample size / power analysis**: Determine the MDE. If baseline ride frequency is 4 rides/month with standard deviation 3, and we want to detect a 10% increase (0.4 rides), at α = 0.05 and power = 0.80: N ≈ 2 × (1.96 + 0.84)² × 3² / 0.4² ≈ 2 × 7.84 × 9 / 0.16 ≈ 883 per group. Round up and account for dropout.

3. **Duration**: Run for at least 2–4 weeks to capture weekly ride patterns. Consider novelty effects — initial spike in usage may not persist.

4. **SUTVA considerations**: If treated users take rides that would otherwise go to control users (supply-side competition), there is interference. Use geographic or temporal clustering if this is a concern.

5. **Metrics**: Primary: rides per user per week. Guardrail: revenue per ride (do discounted rides cannibalize full-price rides?). Secondary: retention at 30/60 days.

**Analysis:**

- Intent-to-treat: Compare average rides between treatment and control using a t-test or regression with pre-treatment covariates (CUPED for variance reduction).
- Check for SRM (sample ratio mismatch) to detect randomization bugs.
- Segment analysis: CATE estimation (do price-sensitive users respond more?) using causal forests or stratified analysis.
- Long-term: If the experiment is short, consider surrogate outcomes or extrapolation methods.

**If RCT is infeasible (already rolled out the discount):**

- Use a natural experiment: if the discount was rolled out in some cities first, use difference-in-differences with parallel trends assumption.
- If there was a spending threshold for eligibility, use RDD.
- As a last resort, propensity score matching on user characteristics, but acknowledge the unconfoundedness assumption is strong.

### Q10. Your model predicts customer churn well, but the business wants to know which factors *cause* churn so they can intervene. How do you approach this?

**Answer:**

First, clarify the distinction: a predictive model identifies *correlates* of churn, not *causes*. A feature like "number of support tickets" might predict churn strongly, but reducing support tickets by making it harder to submit them would likely *increase* churn, not decrease it. The tickets are a symptom of dissatisfaction, not a cause of churn.

**Approach:**

1. **Causal graph construction**: Work with domain experts (product managers, support teams) to draw a DAG. Identify plausible causal drivers (price, product quality, competitor actions, customer service quality) and distinguish them from mediators (support tickets, engagement metrics) and consequences (churn itself).

2. **Identify actionable interventions**: For each proposed cause, define a concrete intervention. "Improve product quality" is vague; "reduce app crash rate" is actionable. Causal inference requires well-defined treatments.

3. **Estimate causal effects**: For each candidate cause:
   - If past A/B test data exists (e.g., a price change experiment), analyze it for the causal effect on churn.
   - If observational data only, use appropriate methods: if the causal graph suggests unconfoundedness is plausible given available covariates, use AIPW; if there are natural experiments (a feature outage in some regions), use quasi-experimental methods.
   - Consider running targeted experiments on the most promising interventions.

4. **Heterogeneous effects**: Estimate CATE to understand which customer segments benefit most from each intervention. Use causal forests or meta-learners.

5. **Validate**: Use hold-out experiments (A/B tests) to confirm causal estimates from observational analysis before scaling.

### Q11. Explain how you would use instrumental variables in an ML context. Give a realistic example and discuss the assumptions.

**Answer:**

**Scenario**: A music streaming service wants to know whether playlist personalization (Treatment W) increases listening time (Outcome Y). However, users who opt into personalization are systematically different (more engaged, more tech-savvy) — there are unmeasured confounders U.

**Instrument**: The service ran a randomized encouragement design: some users received an in-app prompt encouraging them to try personalized playlists (Z = 1), while others did not (Z = 0). Not all prompted users tried personalization (non-compliance), and some non-prompted users found it on their own.

Z = encouragement (randomized)
W = actual use of personalization (endogenous)
Y = monthly listening hours

**Assumption verification:**

1. *Relevance*: The prompt increases personalization usage. Check: E[W | Z=1] − E[W | Z=0] > 0 and first-stage F-statistic > 10.

2. *Exclusion restriction*: The prompt affects listening time *only through* its effect on personalization usage, not directly. This requires that the prompt itself (the notification) does not change listening behavior except by encouraging personalization. A short, neutral prompt is better than an elaborate one that might independently affect engagement.

3. *Independence*: Randomization of Z ensures Z ⊥ U.

4. *Monotonicity*: No users are "defiers" (would use personalization only when *not* prompted). Reasonable here.

**Estimation:**

First stage: Ŵ = γ₀ + γ₁Z (regress personalization on encouragement)
Second stage: Y = α + τŴ (regress listening on predicted personalization)

The coefficient τ from 2SLS estimates the LATE: the effect of personalization for *compliers* — users whose personalization use was influenced by the prompt. This is the relevant estimand for a voluntary feature.

---

## Debugging & Failure-Mode Questions

### Q12. You ran an A/B test and found a statistically significant result, but the effect reverses in a follow-up experiment. What went wrong?

**Answer:** Several possibilities:

1. **Multiple testing / p-hacking**: If many metrics or subgroups were tested in the first experiment and the "significant" one was cherry-picked, the result may be a false positive. The follow-up is likely showing the true (null) effect.

2. **Novelty effects**: Users in the first experiment responded to the novelty of the change, not the change itself. Once the novelty wears off, the effect disappears or reverses.

3. **Sample Ratio Mismatch (SRM)**: A bug in the randomization of the first experiment created unequal groups, biasing the result. Always check that treatment and control sample sizes match the expected split.

4. **Simpson's Paradox / confounding in the experiment**: If the randomization was compromised (e.g., a feature flag was correlated with a product launch), the first result was confounded.

5. **Different populations**: If the follow-up ran at a different time, in a different market, or after a product change, the treatment effect may genuinely differ (external validity failure).

6. **Interference / SUTVA violation**: If the first experiment had spillover effects (which happen to help one group more), and the follow-up experiment isolated units better, the effects could differ.

**Diagnostic steps**: Check SRM in both experiments. Compare baseline metrics between treatment and control. Examine the metric distribution (not just the mean). Look for interaction effects with time, user segment, or concurrent experiments.

### Q13. You are using propensity score matching to estimate the effect of a marketing campaign on purchases. Your propensity score model has an AUC of 0.99. Is this good or bad?

**Answer:** This is likely **bad** for causal inference, even though it sounds great from a prediction perspective.

An AUC of 0.99 means the propensity score model can almost perfectly separate treated from untreated units. This implies near-violation of **positivity**: there exist regions of the covariate space where treatment is almost certain or almost impossible. In these regions, the propensity scores are near 0 or 1, causing:

1. **Extreme IPW weights**: Units with e(x) ≈ 0 who received treatment get enormous weights, making the estimator unstable.
2. **Poor overlap**: The treated and control groups occupy different regions of the covariate space, making comparison difficult.
3. **Extrapolation**: Any matching or regression must extrapolate across groups with little common support.

**What to do:**
- Inspect the propensity score distribution: plot the density of e(x) separately for treated and control. If distributions barely overlap, the common support region is too small.
- Restrict analysis to the overlap region (trim extreme propensity scores). This changes the estimand to the ATE for the overlapping subpopulation.
- Use overlap weights: weight each unit by e(x)(1 − e(x)), which naturally downweights units with extreme propensity scores.
- Consider whether the groups are fundamentally too different for a meaningful comparison. If treated users are so distinct from control users that they cannot be compared, observational causal inference may not be credible.

### Q14. An analyst adjusts for all available covariates in a regression to estimate a causal effect, arguing "more controls is always better." What is wrong with this reasoning?

**Answer:** This reasoning is incorrect for several reasons:

1. **Collider bias**: If any of the covariates is a collider (a common effect of treatment and outcome, or of their causes), conditioning on it introduces bias where none existed. Including all variables guarantees that any colliders in the data are conditioned on.

2. **Post-treatment variables (mediators)**: If a covariate is *caused by* the treatment (e.g., treatment → mediator → outcome), conditioning on it blocks the causal path and estimates only the *direct* effect, not the *total* effect. If the goal is the total effect, mediators should not be conditioned on.

3. **Amplification bias**: Conditioning on a variable that is a *cause of treatment only* (not the outcome) can amplify bias from unmeasured confounding, rather than reducing it. This is called "bias amplification" (Pearl, 2010).

4. **Precision loss**: Including irrelevant covariates increases variance without reducing bias.

**The correct approach**: Use the DAG to determine the appropriate adjustment set. Include confounders (common causes of treatment and outcome), exclude colliders, exclude mediators (unless estimating direct effects), and exclude instruments.

The backdoor criterion provides the precise answer: adjust for a set Z that blocks all backdoor paths from treatment to outcome without opening new spurious paths.

### Q15. You apply the PC algorithm to a dataset with 50 variables and 200 observations. The resulting graph has very few edges. Should you trust it?

**Answer:** Probably not. With only 200 observations and 50 variables, conditional independence tests have very low power, especially when conditioning on multiple variables. Low power means the algorithm will fail to reject independence too often (Type II errors), leading to:

1. **Missing edges**: True causal relationships are not detected because the test lacks power to identify the conditional dependence.
2. **Incorrect orientations**: With many edges incorrectly removed, the remaining graph's structure may be misleading, and v-structure detection becomes unreliable.

The sparse graph is likely an artifact of *underpowered tests*, not a true reflection of a sparse causal structure.

**Remedies:**
- Increase sample size substantially. For reliable conditional independence tests with conditioning sets of size k, you generally need N >> p^k observations (rough scaling).
- Use a more liberal significance level (larger α) to increase power, at the cost of more false positives (extra edges). Sensitivity analysis across α values is informative.
- Use score-based methods (like GES with BIC), which may be more robust in small samples.
- Incorporate domain knowledge to constrain the search space (fix certain edges or orientations).
- Use regularized methods designed for high-dimensional settings.
- Use bootstrap stability analysis: run the algorithm many times on bootstrap samples and keep only edges that appear frequently.

---

## Follow-Up and Probing Questions

### Q16. (Follow-up to Q1) If the individual treatment effect is never observable, how can we make *any* causal claims?

**Answer:** We make causal claims at the *population level*, not the individual level. While τᵢ = Yᵢ(1) − Yᵢ(0) is never observed for any single unit, the *average* treatment effect τ = E[Y(1) − Y(0)] can be identified and estimated under appropriate assumptions.

The key insight: we replace the individual comparison (same unit under two treatments) with a *group comparison* (similar units under different treatments). Under randomization or unconfoundedness, the groups are comparable, so the group-level comparison identifies the average individual effect.

Mathematically: E[Y(1)] = E[Y | W=1] under randomization (because W ⊥ Y(1)), and similarly for Y(0). The group means are unbiased for the potential outcome means.

This is a *statistical* solution to a *logical* problem: we cannot observe individual counterfactuals, but we can make valid *average* statements by aggregating across exchangeable units.

### Q17. (Follow-up to Q5) What happens to the IPW estimator when the propensity score is estimated rather than known?

**Answer:** When e(x) is replaced by an estimate ê(x):

1. **Variance**: The variance of the IPW estimator *decreases* (counterintuitively) compared to using the true propensity score, in finite samples. This is because the estimated propensity score is a function of the data that happens to reduce variance (it "adapts" to the sample). Theoretically, in the well-specified parametric case, the variance of the estimator with estimated propensity scores is never larger than with true propensity scores.

2. **Bias**: If the propensity model is misspecified, ê(x) does not converge to e(x), and the IPW estimator is biased. The bias is generally first-order (not negligible).

3. **Inference**: Standard errors must account for the estimation of the propensity score. Naive standard errors (treating ê as known) are typically too small. The bootstrap or analytical variance formulas (using the influence function of the two-step estimator) provide correct coverage.

4. **With machine learning estimators**: When using flexible models (random forests, neural networks) for ê, the convergence rate may be slower than √N, requiring sample splitting (cross-fitting) to avoid regularization bias. The DML framework of Chernozhukov et al. provides the theoretical foundation.

### Q18. (Follow-up to Q8) In an ML pipeline, how would you detect if you are accidentally conditioning on a collider?

**Answer:**

1. **Draw the DAG**: The most reliable approach. Before fitting any model, sketch the causal relationships between all variables. Identify which variables are colliders (effects of two or more causes). Any collider that appears as a feature in your model is a potential source of collider bias.

2. **Check conditional vs. marginal associations**: If two features become correlated *after* including a third feature in the model but were uncorrelated before, the third feature may be a collider. Formally, if Corr(X, Y) ≈ 0 but Corr(X, Y | Z) ≠ 0, Z may be a collider on the path between X and Y.

3. **Sensitivity analysis**: Remove the suspected collider from the model and see if the estimated treatment effect changes. If removing a variable *reduces* the estimate toward zero or eliminates significance, and there is no reason to believe that variable is a confounder, collider bias is a plausible explanation.

4. **Domain knowledge red flags**: Variables that are *consequences* of both the treatment and the outcome (or their causes) are colliders. Examples: "patient hospitalized" when studying drug effects (drug → hospitalization ← disease severity), "model score" when studying the effect of a feature on the label.

### Q19. (Follow-up to Q9) The ride-sharing experiment shows a 15% increase in rides but a 5% decrease in revenue per ride. How do you advise the product team?

**Answer:** This scenario involves *competing metrics* and requires careful reasoning about the total effect and business implications.

**Decompose the overall revenue effect:**

Total revenue = Rides × Revenue per ride

If rides increase by 15% and revenue per ride decreases by 5%:
Approximate total revenue change ≈ 15% − 5% = +10% (more precisely: 1.15 × 0.95 = 1.0925, so +9.25%)

So total revenue increases. But this calculation assumes the effects are independent, which they may not be.

**Consider mechanisms:**

- The discount mechanically reduces revenue per ride (by 20% for discounted rides). The 5% *average* decrease is smaller because not all rides are discounted — the treatment adds new rides (at the discount) while existing rides remain full-price.
- The new rides may be in different segments (shorter rides, off-peak) with different unit economics.

**Long-term considerations:**

- Does the discount create a *reference price effect*? Will users expect discounts going forward and refuse to pay full price? Measure whether full-price rides decrease after the discount period ends.
- Is the 15% increase sustainable, or is it a novelty/stockpiling effect?
- What is the cost of the discount program (including opportunity cost of foregone revenue on rides that would have happened anyway)?

**Recommendation structure:**

- Present the total revenue impact (+9.25%) as the primary finding.
- Flag the revenue-per-ride decrease and its mechanism.
- Recommend a longer experiment to assess long-term effects and habituation.
- Suggest testing different discount levels (10%, 15%) to find the optimal point.
- Estimate the CATE: which users drive the ride increase? Target the discount to price-sensitive users for better ROI.

### Q20. (Follow-up to Q11) Can you use the prediction probability of a machine learning model as an instrument?

**Answer:** No, model predictions are almost never valid instruments. An instrument must satisfy:

1. **Relevance**: The prediction should be correlated with the treatment. If the model predicts treatment, this may hold, but only mechanically (the model was trained on the treatment variable).

2. **Exclusion restriction**: The prediction must affect the outcome *only through* the treatment. This almost always fails because any variable that predicts treatment well likely also predicts the outcome directly (through shared confounders). The model prediction is a function of observed features, many of which directly affect the outcome.

3. **Independence**: The prediction must be independent of unmeasured confounders. Since the prediction is a function of observed covariates (which are generally correlated with unmeasured confounders), this fails.

A model prediction is essentially a summary of observed covariates — it inherits all the confounding present in those covariates. Using it as an instrument does not create any "exogenous" variation; it merely repackages endogenous variation.

Valid instruments come from *external sources of variation* — natural randomization, policy changes, geographic variation — not from the data's own features.

### Q21. Explain the difference between the backdoor and frontdoor criteria. When would you use one over the other?

**Answer:**

**Backdoor criterion**: Identifies the causal effect of X on Y by finding a set Z that blocks all *backdoor paths* (non-causal paths from X to Y) without blocking the causal path or opening new spurious paths. Requires that Z contains no descendants of X and blocks all confounding paths.

Use when: you have measured all relevant confounders (or a sufficient adjustment set).

**Frontdoor criterion**: Identifies the causal effect of X on Y through a mediator M, even when there are *unmeasured confounders* between X and Y. Requires that (1) M intercepts all directed paths from X to Y, (2) there is no confounding of X → M, and (3) X blocks all backdoor paths from M to Y.

Use when: there are unmeasured confounders between X and Y, but you can observe a mediator M with the right properties.

**Key difference**: The backdoor criterion requires measuring confounders directly. The frontdoor criterion circumvents unmeasured confounders by routing identification through a mediator. The frontdoor criterion is applicable in fewer situations (the conditions are restrictive), but it handles unmeasured confounding that the backdoor criterion cannot.

**Example**: Smoking (X) → Tar (M) → Cancer (Y), with genetic confounder U → X, U → Y. Backdoor criterion fails (U unmeasured). Frontdoor criterion works (M = tar satisfies all conditions).

### Q22. You are building a fairness-aware ML model. How does causal inference inform your approach compared to purely statistical fairness criteria?

**Answer:**

**Statistical fairness** (demographic parity, equalized odds, calibration) is defined in terms of conditional probabilities of predictions and outcomes given protected attributes. These criteria are purely *observational* — they do not distinguish between causal and spurious associations.

**Causal fairness** asks: "Would the model's prediction change if the individual's protected attribute had been different?" This is a *counterfactual* question.

The distinction matters because statistical and causal criteria can conflict:

1. **Legitimate mediators**: Suppose gender (A) affects field of study (M), which affects salary (Y). A model that uses field of study is statistically discriminatory (predictions differ by gender), but the causal pathway A → M → Y may be considered *legitimate* (it reflects choices, not discrimination). Causal fairness can distinguish *direct* discrimination (A → Y) from *indirect* effects through legitimate mediators.

2. **Resolving variables**: Some variables may be both caused by the protected attribute and directly relevant to the task. Whether to condition on them depends on the causal structure and the normative definition of fairness.

3. **Interventional fairness**: "If we changed the model's use of feature X, what would happen to disparities?" requires a causal model. Statistical criteria cannot answer interventional questions.

**Practical approach**: Draw the causal DAG including the protected attribute, its causes and effects, mediators, and the outcome. Decide which causal pathways represent unfair discrimination vs. legitimate variation. Implement a model that blocks the unfair pathways while allowing the legitimate ones. This requires explicit normative choices informed by the causal structure.

### Q23. Explain Difference-in-Differences (DiD) and its key assumption. How does it relate to the potential outcomes framework?

**Answer:**

DiD estimates causal effects by comparing the *change* in outcomes over time between a group that receives treatment and a group that does not.

**Setup**: Two groups (Treatment, Control) observed at two time points (Before, After). Treatment is applied to the Treatment group between the two time periods.

**Estimator:**

τ̂_DiD = (Ȳ_T,After − Ȳ_T,Before) − (Ȳ_C,After − Ȳ_C,Before)

The first difference (within-group change) removes time-invariant group-level confounders. The second difference (between-group comparison of changes) removes time trends common to both groups.

**Key assumption — Parallel trends**: In the absence of treatment, the treatment group's outcome would have changed by the same amount as the control group's. Formally:

E[Y(0)_T,After − Y(0)_T,Before] = E[Y(0)_C,After − Y(0)_C,Before]

This is a *counterfactual* assumption: it concerns Y(0) for the treatment group in the post period, which is never observed.

**In the potential outcomes framework**: The parallel trends assumption allows us to use the control group's change as a proxy for the counterfactual change the treatment group would have experienced without treatment.

τ_DiD = E[Y(1)_T,After − Y(0)_T,After] = ATT

The DiD estimator identifies the ATT (effect on the treated), not the ATE.

**When parallel trends fail**: If the groups were already on different trajectories before treatment (e.g., the treatment group was growing faster), DiD overestimates or underestimates the effect. Pre-treatment trends should be checked (do the groups move in parallel before the intervention?), though parallel pre-trends do not guarantee parallel post-treatment counterfactual trends.

### Q24. What is the Stable Unit Treatment Value Assumption (SUTVA), and how would you design an experiment when you expect it to be violated?

**Answer:**

SUTVA requires (1) no interference between units and (2) no hidden treatment versions. It ensures that each unit's potential outcomes are well-defined and depend only on its own treatment.

**When SUTVA is expected to be violated (e.g., marketplace experiment):**

1. **Graph-cluster randomization**: Instead of randomizing individual users, randomize clusters of connected users (defined by interaction graphs). Users within a cluster all receive the same treatment, reducing cross-cluster interference. The trade-off: fewer independent clusters means lower statistical power, but reduced bias from interference.

2. **Switchback designs**: Randomize treatment at the *time* level (e.g., alternating treatment and control periods for all users). This is useful in two-sided marketplaces where supply and demand interact: during a "treatment on" period, all participants experience the treatment. The causal effect is identified by comparing treatment-on and treatment-off periods. Challenges: carryover effects between periods, and the need for rapid switching to avoid confounding with time trends.

3. **Two-sided randomization**: In marketplaces, randomize on both sides (e.g., randomize sellers into treatment/control and buyers into treatment/control) and analyze the factorial design.

4. **Ego-network randomization**: Assign treatment at the user level but measure outcomes for the user's "ego network" (the user and their immediate contacts). This allows modeling spillover effects explicitly.

5. **Modeling interference**: Instead of assuming SUTVA, explicitly model the interference structure. For example, assume each unit's outcome depends on the *fraction* of its neighbors who are treated. Estimate both the direct effect and the spillover effect.

### Q25. What is the relationship between do-calculus and the potential outcomes framework? Are they equivalent?

**Answer:**

The two frameworks are largely equivalent in terms of what causal effects they can define and identify, but they emphasize different aspects and have different strengths:

**Potential outcomes (Rubin)**: Defines causal effects as comparisons of potential outcomes (Yᵢ(1) vs. Yᵢ(0)). Focuses on *estimands* and *estimation strategies*. Assumptions are stated in terms of potential outcomes (unconfoundedness, consistency). Well-suited for experimental design, randomization inference, and clinical trials.

**Structural causal models / do-calculus (Pearl)**: Defines causal effects via interventions (do-operator) on structural equations. Focuses on *identification* from DAGs. Assumptions are encoded in the graph structure. Well-suited for reasoning about complex causal structures, determining adjustment sets, and handling non-manipulable treatments.

**Equivalences:**
- E[Y(1)] in potential outcomes = E[Y | do(X = 1)] in do-calculus.
- Unconfoundedness (Y(w) ⊥ W | X) corresponds to the backdoor criterion being satisfied by X.
- Both frameworks agree on when a causal effect is identifiable and what the resulting estimand is.

**Differences in emphasis:**
- Pearl's framework provides *algorithmic tools* (d-separation, do-calculus, backdoor/frontdoor criteria) for determining what to adjust for — the potential outcomes framework lacks these graphical tools.
- The potential outcomes framework provides *estimation theory* (IPW, AIPW, matching) and integrates naturally with statistical inference — Pearl's framework is weaker on estimation and inference.
- The potential outcomes framework requires treatments to be manipulable (no causation without manipulation). Pearl's framework handles questions about non-manipulable attributes (e.g., the effect of race) via structural equations.

In practice, modern causal inference uses *both* frameworks: the DAG/do-calculus for identification (what to adjust for) and the potential outcomes framework for estimation (how to compute the answer from data).

---

**[← Previous Chapter: Learning Theory](learning_theory_guide.md) | [Table of Contents](index.md) | [Next Chapter: Linear Models →](linear_models_guide.md)**
