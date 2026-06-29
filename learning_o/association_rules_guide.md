# Association Rule Mining — A Textbook-Quality Learning Guide

> A comprehensive, self-contained reference covering the theory, algorithms, evaluation, and practice of association rule mining. Intended to be readable end-to-end by someone with no prior exposure, and dense enough to function as a standalone reference for interview preparation.

---

## Table of Contents

0. [Preliminaries: The Setting of Association Rule Mining](#0-preliminaries-the-setting-of-association-rule-mining)
1. [Topic 1 — The Apriori Algorithm](#topic-1--the-apriori-algorithm)
2. [Topic 2 — FP-Growth](#topic-2--fp-growth)
3. [Topic 3 — Evaluation Metrics Beyond Support/Confidence/Lift](#topic-3--evaluation-metrics-beyond-supportconfidencelift)
4. [Topic 4 — Applications: Market Basket Analysis & Cross-Selling](#topic-4--applications-market-basket-analysis--cross-selling)
5. [Interview Preparation](#interview-preparation)

---

## 0. Preliminaries: The Setting of Association Rule Mining

Before diving into any single algorithm, it is essential to fix the vocabulary and the problem being solved. Every section below will repeatedly reference these objects.

### 0.1 The universe of items and transactions

Let

$$
\mathcal{I} = \{i_1, i_2, \dots, i_m\}
$$

be a finite **universe of items**. In a supermarket, items are products like *milk, bread, diapers, beer*. In a streaming service, items are movies. In a clickstream dataset, items are URLs.

A **transaction** $T$ is a subset of $\mathcal{I}$:

$$
T \subseteq \mathcal{I}.
$$

A **database** $\mathcal{D} = \{T_1, T_2, \dots, T_N\}$ is a multiset of $N$ transactions. Each transaction is typically identified by a transaction ID (TID).

**Important**: in the classical formulation, transactions are *sets*, not multisets. Buying two cartons of milk counts the same as buying one. Quantities and prices are ignored. This is a deliberate simplification; weighted and quantitative variants exist but are extensions.

An **itemset** $X$ is any subset of $\mathcal{I}$. A $k$-itemset is an itemset of cardinality $k$. We say a transaction $T$ **contains** $X$ if $X \subseteq T$.

### 0.2 Association rules

An **association rule** is an implication of the form

$$
X \Rightarrow Y, \quad X, Y \subseteq \mathcal{I}, \quad X \cap Y = \varnothing, \quad X \neq \varnothing, \quad Y \neq \varnothing.
$$

- $X$ is called the **antecedent** (LHS, body, premise).
- $Y$ is called the **consequent** (RHS, head).

**Critical caveat**: the symbol $\Rightarrow$ is *not* logical implication, and it does *not* denote causation. It means a statistical co-occurrence pattern: "transactions containing $X$ tend to also contain $Y$." Confusing this with causality is the single most common mistake practitioners make.

### 0.3 The two canonical measures (full treatment in Topic 1)

- **Support** of an itemset $Z$:
  $$\text{supp}(Z) = \frac{|\{T \in \mathcal{D} : Z \subseteq T\}|}{N}.$$
  The fraction of transactions containing $Z$. Sometimes reported as an absolute count rather than a fraction.
- **Confidence** of a rule $X \Rightarrow Y$:
  $$\text{conf}(X \Rightarrow Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}.$$
  An empirical estimate of $P(Y \subseteq T \mid X \subseteq T)$.

### 0.4 The standard two-step framework

The classical pipeline, due to Agrawal, Imieliński, and Swami (1993), splits rule mining into two clean subproblems:

1. **Find all frequent itemsets**: itemsets $Z$ with $\text{supp}(Z) \geq \sigma$ for a user-chosen minimum support threshold $\sigma$. *This step is computationally hard.*
2. **Generate rules from frequent itemsets**: for each frequent $Z$ and each non-empty proper subset $X \subsetneq Z$, form the candidate rule $X \Rightarrow Z \setminus X$ and keep it if $\text{conf}(X \Rightarrow Z \setminus X) \geq \gamma$ for a minimum confidence threshold $\gamma$. *This step is cheap given step 1.*

Virtually all the algorithmic sophistication — Apriori, FP-Growth, Eclat, LCM, and so on — concerns step 1.

### 0.5 Why this is hard

There are $2^m - 1$ non-empty itemsets over $m$ items. In a supermarket with $m = 10{,}000$ products, this is astronomical. Naïvely scanning the database to count every possible itemset is infeasible. The entire field exists to answer: *how do we find frequent itemsets without enumerating the exponential search space?*

With the stage set, we can now study each topic in depth.

---

---

## Topic 1 — The Apriori Algorithm

### 1.1 Motivation & Intuition

#### The real problem

Imagine you run a supermarket with 10,000 products and millions of shopping transactions. Your intuition says "people who buy diapers tend to buy beer on Friday nights" (the famous, likely apocryphal example). You'd like to discover *all* such patterns automatically, without a human analyst proposing them one at a time.

A naïve approach — counting every possible combination of products — is impossible. Even if you restricted yourself to combinations of size 3, you'd have $\binom{10000}{3} \approx 1.67 \times 10^{11}$ triples to count. And you'd need a separate scan of the database for every single one.

Apriori was the first widely adopted algorithm that made this tractable. It exploits a beautifully simple observation:

> If an itemset is rare, every itemset that contains it is at least as rare.

Or equivalently, stated positively:

> If an itemset is frequent, every one of its subsets is also frequent.

This is the **Apriori property** (also called **anti-monotonicity** or **downward closure**). It lets us eliminate enormous swaths of the search space without ever counting them.

#### Concrete intuition before math

Say "milk" appears in only 0.5% of transactions and your minimum support threshold is 2%. Then *any* itemset containing "milk" — {milk, bread}, {milk, eggs, cheese}, and so on — is also below 2%. You do not need to count any of them. You just crossed millions of itemsets off your list by running one support check.

Apriori operationalizes this: it builds up frequent itemsets *level by level*, from size 1 to size $k$, and at each level uses the frequent $(k-1)$-itemsets from the previous level as building blocks, refusing to even consider a $k$-itemset unless all its $(k-1)$-subsets are already known to be frequent.

#### Connection to ML systems

Association rule mining is a form of **unsupervised pattern discovery**. It pre-dates modern recommender systems but remains useful because:
- It scales to very sparse, very high-cardinality categorical data where most ML models struggle.
- Its output (rules) is human-interpretable — unlike matrix factorizations or neural embeddings.
- It is often used in exploratory analysis, feature engineering (e.g. generating candidate categorical interaction features), and for building initial baselines for recommender systems.

---

### 1.2 Conceptual Foundations

#### Key definitions

- **Item, itemset, transaction, database** — as in §0.1.
- **Minimum support $\sigma$** — a user-chosen threshold in $[0, 1]$ (or an absolute count). An itemset is **frequent** if its support is $\geq \sigma$.
- **Minimum confidence $\gamma$** — a user-chosen threshold in $[0, 1]$. A rule is **strong** if its confidence is $\geq \gamma$ (in addition to the antecedent+consequent itemset being frequent).
- **Candidate itemset** — an itemset we are considering as possibly frequent but have not yet verified with a database scan.
- **$L_k$** — the set of frequent $k$-itemsets discovered.
- **$C_k$** — the set of candidate $k$-itemsets we intend to count.

#### The Apriori property (anti-monotonicity)

**Claim**: If $Z$ is frequent, then every $Z' \subseteq Z$ is frequent.

**Proof**: If $Z' \subseteq Z$, then any transaction $T$ with $Z \subseteq T$ also satisfies $Z' \subseteq T$. Hence

$$
|\{T : Z \subseteq T\}| \leq |\{T : Z' \subseteq T\}| \Longrightarrow \text{supp}(Z) \leq \text{supp}(Z').
$$

So $\text{supp}(Z) \geq \sigma \Rightarrow \text{supp}(Z') \geq \sigma$. $\blacksquare$

The contrapositive is what we actually use algorithmically:

> **If any $(k-1)$-subset of a $k$-itemset is infrequent, the $k$-itemset is infrequent.**

#### The algorithm at a conceptual level

Apriori alternates between two steps repeatedly:

1. **Candidate generation** (no database access): build $C_k$ from $L_{k-1}$.
2. **Support counting** (one database scan): count how many transactions contain each candidate, keep only those meeting $\sigma$, producing $L_k$.

It terminates when some level yields no frequent itemsets.

#### Underlying assumptions

1. **Binary occurrence semantics.** We only record whether an item appeared in a transaction, not how many or at what price.
2. **I.i.d. transactions (implicit).** Statistical interpretation of support as $P(X \subseteq T)$ assumes transactions are independent draws from some underlying distribution. Violated, for instance, when transactions are strongly time-correlated (a sale, seasonal buying).
3. **Static database.** Apriori is a batch algorithm. Incremental variants exist but vanilla Apriori assumes the database does not change during mining.
4. **A fixed, hard support threshold $\sigma$.** The algorithm has no notion of "almost frequent." Small perturbations in the threshold can cause discontinuous changes in output.
5. **The user is willing to enumerate all frequent itemsets.** For low $\sigma$ or dense data, the number of frequent itemsets can be astronomical.

#### What breaks when assumptions fail

- **Dense data** (e.g., dense gene expression matrices): the number of frequent itemsets explodes combinatorially. Apriori may never terminate in practice.
- **Very low $\sigma$**: same problem. You end up enumerating nearly every itemset.
- **Skewed item frequencies**: a few very common items (like "shopping bag") will appear in almost every frequent itemset, drowning out genuinely interesting patterns.
- **Highly correlated transactions**: support no longer approximates a meaningful probability.
- **Dynamic data**: results become stale the moment mining finishes.

---

### 1.3 Mathematical Formulation

#### Notation

- $\mathcal{D}$: database of $N$ transactions.
- $\text{supp}(X) = \frac{1}{N}|\{T \in \mathcal{D} : X \subseteq T\}|$: relative support.
- $\text{count}(X) = N \cdot \text{supp}(X)$: absolute support (integer count).
- $\sigma$: minimum relative support; $\sigma_{\text{abs}} = \lceil \sigma N \rceil$ minimum absolute support.
- $L_k$: frequent $k$-itemsets.
- $C_k$: candidate $k$-itemsets.

#### Derivation of the core algorithm

We want $L = \bigcup_{k \geq 1} L_k$.

**Base case**: $L_1$. Scan the database once, count the support of every 1-itemset (every item), keep those meeting $\sigma$.

**Inductive step**: assuming we have $L_{k-1}$, construct $L_k$.

The key lemma is the contrapositive of the Apriori property: every frequent $k$-itemset has all $(k-1)$-subsets in $L_{k-1}$. So we only need to consider $k$-itemsets all of whose $(k-1)$-subsets are in $L_{k-1}$. Call this set $C_k$.

**Candidate generation: $C_k = \text{apriori\_gen}(L_{k-1})$.** It has two sub-steps:

*Step 1 — Join.* Assume items within each itemset are stored in a fixed total order (e.g., lexicographic). For $p, q \in L_{k-1}$ with

$$
p = \{p_1, \dots, p_{k-2}, p_{k-1}\}, \quad q = \{p_1, \dots, p_{k-2}, q_{k-1}\}, \quad p_{k-1} < q_{k-1},
$$

i.e., $p$ and $q$ share their first $k-2$ items and differ only in the last, form the candidate $\{p_1, \dots, p_{k-2}, p_{k-1}, q_{k-1}\}$.

Why only join pairs sharing a $(k-2)$-prefix? Because any frequent $k$-itemset $Z$ has at least two $(k-1)$-subsets obtained by removing its last or second-to-last element, and these two subsets differ only in their last element. Joining any other pair of frequent $(k-1)$-itemsets either produces a duplicate candidate or a candidate missing the required subset structure. This restriction avoids generating the same candidate multiple times.

*Step 2 — Prune.* For each candidate $c \in C_k$, check all $k$ of its $(k-1)$-subsets: if any is not in $L_{k-1}$, remove $c$ from $C_k$. This is the direct application of the Apriori property.

**Support counting.** Scan the database once. For each transaction $T$, for each candidate $c \in C_k$, test if $c \subseteq T$ and increment $\text{count}(c)$. Naïvely this is $O(|\mathcal{D}| \cdot |C_k|)$; in practice, Apriori organizes $C_k$ in a **hash tree** (a trie-like structure indexed by item positions) so that we visit only a small subset of candidates for each transaction.

**Produce $L_k$.** $L_k = \{c \in C_k : \text{count}(c) \geq \sigma_{\text{abs}}\}$.

**Terminate** when $L_k = \varnothing$.

#### Pseudocode

```
Input:  database D, min support σ
Output: all frequent itemsets L

L_1 ← {i ∈ I : supp({i}) ≥ σ}          # single database scan
k ← 2
while L_{k-1} ≠ ∅:
    C_k ← apriori_gen(L_{k-1})          # join + prune; no DB access
    for each transaction T in D:        # single database scan
        for each c ∈ C_k with c ⊆ T:
            count[c] += 1
    L_k ← {c ∈ C_k : count[c] ≥ σ·N}
    k ← k + 1
return L_1 ∪ L_2 ∪ … ∪ L_{k-1}
```

#### Rule generation

Given a frequent itemset $Z$ with $|Z| \geq 2$, for every non-empty proper subset $X \subsetneq Z$, consider the rule $X \Rightarrow Z \setminus X$ and compute

$$
\text{conf}(X \Rightarrow Z \setminus X) = \frac{\text{supp}(Z)}{\text{supp}(X)}.
$$

Both $\text{supp}(Z)$ and $\text{supp}(X)$ are already known (they are frequent itemsets found by the mining step — note $\text{supp}(X) \geq \text{supp}(Z) \geq \sigma$).

**Confidence is anti-monotonic on the consequent side.** If $X \Rightarrow Y$ has confidence below $\gamma$, then for any $X' \subsetneq X$, the rule $X' \Rightarrow Y \cup (X \setminus X')$ (moving items from antecedent to consequent) has confidence no larger. Proof sketch: shrinking the antecedent can only increase its support, so

$$
\text{conf}(X' \Rightarrow Z \setminus X') = \frac{\text{supp}(Z)}{\text{supp}(X')} \leq \frac{\text{supp}(Z)}{\text{supp}(X)} = \text{conf}(X \Rightarrow Z \setminus X)
$$

(the last inequality because $X' \subseteq X \Rightarrow \text{supp}(X') \geq \text{supp}(X)$, and we're dividing a fixed numerator by a larger denominator). This lets us prune rule generation similarly to itemset mining: once a rule with small consequent fails, rules with larger consequents can be skipped.

#### Lift and why we need it

Confidence alone is misleading. If $Y$ is very common (say $\text{supp}(Y) = 0.9$), then *any* rule $X \Rightarrow Y$ has confidence at least roughly $0.9$ by chance. The rule tells us nothing.

**Lift** corrects for the baseline frequency of $Y$:

$$
\text{lift}(X \Rightarrow Y) = \frac{\text{conf}(X \Rightarrow Y)}{\text{supp}(Y)} = \frac{\text{supp}(X \cup Y)}{\text{supp}(X) \cdot \text{supp}(Y)}.
$$

Interpretations:
- $\text{lift} = 1$: $X$ and $Y$ are statistically independent (under the empirical distribution).
- $\text{lift} > 1$: positive association — $X$ makes $Y$ *more* likely than baseline.
- $\text{lift} < 1$: negative association.
- Note lift is **symmetric**: $\text{lift}(X \Rightarrow Y) = \text{lift}(Y \Rightarrow X)$. This is both a feature (it captures mutual association) and a limitation (it cannot distinguish direction).

Lift is the ratio of joint probability to the product of marginals — the same quantity that appears in pointwise mutual information, $\text{PMI} = \log(\text{lift})$.

---

### 1.4 Worked Example

Let's run Apriori end-to-end on a toy database.

**Database** (TID → items), with $\mathcal{I} = \{A, B, C, D, E\}$:

| TID | Items         |
|-----|---------------|
| 1   | A, B, D       |
| 2   | B, C, E       |
| 3   | A, B, C, E    |
| 4   | B, E          |
| 5   | A, C, E       |

$N = 5$. Use $\sigma = 0.4$, so $\sigma_{\text{abs}} = \lceil 0.4 \cdot 5 \rceil = 2$.

**Pass 1: find $L_1$.**

Count each single item:
- A: appears in TIDs 1, 3, 5 → count 3, supp 0.6
- B: 1, 2, 3, 4 → 4, supp 0.8
- C: 2, 3, 5 → 3, supp 0.6
- D: 1 → 1, supp 0.2   ← below threshold
- E: 2, 3, 4, 5 → 4, supp 0.8

$$L_1 = \{\{A\}, \{B\}, \{C\}, \{E\}\}.$$ Item D is pruned permanently — by anti-monotonicity no superset of {D} can be frequent either.

**Pass 2: generate $C_2$ from $L_1$, count, obtain $L_2$.**

All 2-subsets of $\{A,B,C,E\}$: $\{A,B\}, \{A,C\}, \{A,E\}, \{B,C\}, \{B,E\}, \{C,E\}$. (All $(k-1) = 1$-subsets of each are trivially in $L_1$, so pruning changes nothing at $k=2$.)

Scan DB and count:
- $\{A,B\}$: TIDs 1, 3 → 2  ✓
- $\{A,C\}$: TIDs 3, 5 → 2  ✓
- $\{A,E\}$: TIDs 3, 5 → 2  ✓
- $\{B,C\}$: TIDs 2, 3 → 2  ✓
- $\{B,E\}$: TIDs 2, 3, 4 → 3  ✓
- $\{C,E\}$: TIDs 2, 3, 5 → 3  ✓

$$
L_2 = \{\{A,B\}, \{A,C\}, \{A,E\}, \{B,C\}, \{B,E\}, \{C,E\}\}.
$$

**Pass 3: generate $C_3$ from $L_2$, count, obtain $L_3$.**

*Join step.* Using lexicographic order $A < B < C < E$, find pairs in $L_2$ sharing their first $k-2 = 1$ item:

- Prefix $A$: $\{A,B\}, \{A,C\}, \{A,E\}$ → candidates $\{A,B,C\}, \{A,B,E\}, \{A,C,E\}$.
- Prefix $B$: $\{B,C\}, \{B,E\}$ → candidate $\{B,C,E\}$.
- Prefix $C$: only $\{C,E\}$ → nothing.

$$
C_3^{\text{join}} = \{\{A,B,C\}, \{A,B,E\}, \{A,C,E\}, \{B,C,E\}\}.
$$

*Prune step.* For each candidate, check all three 2-subsets are in $L_2$:
- $\{A,B,C\}$: subsets $\{A,B\}, \{A,C\}, \{B,C\}$ — all in $L_2$ ✓
- $\{A,B,E\}$: $\{A,B\}, \{A,E\}, \{B,E\}$ — all in $L_2$ ✓
- $\{A,C,E\}$: $\{A,C\}, \{A,E\}, \{C,E\}$ — all in $L_2$ ✓
- $\{B,C,E\}$: $\{B,C\}, \{B,E\}, \{C,E\}$ — all in $L_2$ ✓

$C_3 = C_3^{\text{join}}$. (In general the prune step does remove candidates; here no subset is missing because $L_2$ is rich.)

*Scan DB and count:*
- $\{A,B,C\}$: need A,B,C all present. TID 3 only → 1  ✗
- $\{A,B,E\}$: TID 3 only → 1  ✗
- $\{A,C,E\}$: TIDs 3, 5 → 2  ✓
- $\{B,C,E\}$: TIDs 2, 3 → 2  ✓

$$
L_3 = \{\{A,C,E\}, \{B,C,E\}\}.
$$

**Pass 4: generate $C_4$.**

Join pairs in $L_3$ sharing first 2 items: $\{A,C,E\}$ and $\{B,C,E\}$ share no 2-prefix (their first items differ: A vs B). No candidates.

$C_4 = \varnothing$, $L_4 = \varnothing$, algorithm halts.

**Final frequent itemsets**:

$$
L = L_1 \cup L_2 \cup L_3 = \{\{A\},\{B\},\{C\},\{E\},\{A,B\},\{A,C\},\{A,E\},\{B,C\},\{B,E\},\{C,E\},\{A,C,E\},\{B,C,E\}\}.
$$

**Rule generation** with $\gamma = 0.7$.

Take $\{A,C,E\}$ (support 2/5 = 0.4) and enumerate all rules:

| Rule              | Numer (supp Z) | Denom (supp ant.) | Confidence | Passes? |
|-------------------|----------------|--------------------|------------|---------|
| A ⇒ C,E           | 0.4            | 0.6                | 0.667      | no      |
| C ⇒ A,E           | 0.4            | 0.6                | 0.667      | no      |
| E ⇒ A,C           | 0.4            | 0.8                | 0.5        | no      |
| A,C ⇒ E           | 0.4            | 0.4                | 1.0        | ✓       |
| A,E ⇒ C           | 0.4            | 0.4                | 1.0        | ✓       |
| C,E ⇒ A           | 0.4            | 0.6                | 0.667      | no      |

Compute lift for the surviving rules:
- $\text{lift}(A,C \Rightarrow E) = 1.0 / 0.8 = 1.25$.
- $\text{lift}(A,E \Rightarrow C) = 1.0 / 0.6 \approx 1.67$.

Both exceed 1 → positive association. The rule $A,E \Rightarrow C$ has the stronger lift: observing $\{A,E\}$ raises the probability of $C$ from 0.6 (baseline) to 1.0, a 1.67× multiplier.

Repeat for $\{B,C,E\}$ and the 2-itemsets analogously. This exhausts the rule-mining step.

---

### 1.5 Relevance to Machine Learning Practice

#### Where Apriori is used

- **Market basket analysis** (the canonical use case, Topic 4).
- **Recommender system baselines.** "Customers who bought X also bought Y" widgets are literally rules with high lift.
- **Categorical feature interaction discovery.** In tabular ML with high-cardinality categorical columns, frequent itemsets over (column=value) pairs can reveal useful interaction features for downstream models.
- **Clickstream and web usage mining.** Sequences of URLs visited in a session, treated as itemsets, yield navigational patterns.
- **Fraud and anomaly pattern mining.** Rare but strong co-occurrences (low support, very high lift) can surface suspicious combinations.
- **Bioinformatics.** Frequent co-occurring gene mutations or protein interactions.
- **Intrusion detection.** Patterns of system calls or network events.

#### When *not* to use Apriori

- **Dense data**: if most items are frequent, the search space explodes. FP-Growth or specialized methods are preferable, and often even those are inadequate.
- **Very low support thresholds**: Apriori performs poorly; each pass produces huge $C_k$.
- **Continuous-valued features**: you must discretize, which is lossy.
- **Data with strong temporal structure**: sequence mining (SPADE, PrefixSpan) or sequential rule mining is more appropriate.
- **When the goal is prediction, not description**: a supervised model (logistic regression, gradient boosting, neural nets) usually outperforms rule-based prediction in accuracy, though at the cost of interpretability.

#### Alternatives

- **FP-Growth** (Topic 2): often 10–100× faster on typical datasets.
- **Eclat**: depth-first enumeration with vertical TID-list data layout; fast intersection-based counting.
- **LCM** (Linear time Closed itemset Miner): state of the art for closed/maximal itemsets.
- **SON, PCY, DIC**: classical Apriori speedups (partitioning, hashing).
- **Matrix factorization / collaborative filtering**: for recommender-style tasks, typically more accurate but less interpretable.

#### Trade-offs

| Concern             | Apriori behavior                                                               |
|---------------------|---------------------------------------------------------------------------------|
| Interpretability    | Excellent — rules are human-readable.                                          |
| Bias/variance       | No statistical model, so "bias–variance" doesn't apply in the usual sense; but rules mined on a small database have high sampling variance. |
| Robustness          | Very sensitive to $\sigma$; hard threshold creates discontinuities.             |
| Scalability         | Poor in dense/low-σ regimes due to exponential candidate generation.            |
| Computational cost  | $O(\text{passes} \times |\mathcal{D}| \times |C_k|)$; passes = one per level.   |
| Memory              | $C_k$ can be gigantic; bottleneck on large problems.                            |

---

### 1.6 Common Pitfalls & Misconceptions

1. **"A rule with high confidence is a strong rule."** Often wrong: high confidence can merely reflect that the consequent is universally common. Always check lift (or some other lift-like measure).
2. **"$X \Rightarrow Y$ means $X$ causes $Y$."** No. Association $\neq$ causation. The canonical diaper→beer pattern, even if real, doesn't mean buying diapers *causes* buying beer. Both might be driven by a common cause (a father sent shopping Friday evening).
3. **"Higher support = more useful."** Extremely common itemsets are often trivially useful ({plastic bag, receipt}) and drown out the interesting findings. The support threshold balances this: too high → only obvious patterns; too low → combinatorial explosion.
4. **"Apriori needs only a few database scans."** It needs one per level $k$. If the longest frequent itemset has length 10, that is 10 full passes over the database — a major I/O cost.
5. **Forgetting the fixed-order assumption in candidate generation.** Without a total order on items, the join step produces duplicate candidates and misses others.
6. **Counting duplicates in the "prune" step.** Students often confuse the join step (which produces *possible* candidates) with the prune step (which removes those with any infrequent $(k-1)$-subset). Both are needed.
7. **Using relative and absolute support inconsistently.** Fix one convention.
8. **Assuming lift > 1 means "statistically significant."** It does not. Lift is a point estimate with its own sampling error. For formal significance you need a statistical test (e.g. a $\chi^2$ test or a Fisher exact test on the 2×2 contingency table).
9. **Mining on a biased sample.** If transactions are sampled non-uniformly (e.g., only high-value customers), the rules reflect that sample, not the population.
10. **Combining very different shopper types.** Mining across segments with different preferences produces weak average rules; segmentation first, then rule mining, often works better.

---

---

## Topic 2 — FP-Growth

### 2.1 Motivation & Intuition

#### Apriori's weakness

Apriori makes two assumptions that hurt it:
1. It generates candidates explicitly (the $C_k$ sets).
2. It makes one database pass per level $k$.

On a dataset with 100,000 transactions, if frequent itemsets reach length 10, that's 10 passes. Worse, when many items are frequent, $|C_k|$ explodes — for example $|C_2| = \binom{|L_1|}{2}$, which can be millions of candidates.

#### The FP-Growth idea

In 2000, Han, Pei, and Yin proposed **FP-Growth** (Frequent Pattern Growth), an algorithm that:
1. Compresses the database into a compact in-memory **prefix tree** (the FP-tree) using only **two** database scans.
2. Mines frequent itemsets by recursively analyzing subtrees, with **no candidate generation at all**.

The key insight: if many transactions share common prefixes (when items are sorted by frequency), those prefixes only need to be stored *once* in a tree. In real data, transactions overlap heavily, so the tree is much smaller than the database itself.

#### Concrete intuition before math

Imagine five transactions:

```
T1: f, c, a, m, p
T2: f, c, a, b, m
T3: f, b
T4: c, b, p
T5: f, c, a, m, p
```

If we sort items within each transaction by overall frequency (descending) — say $f > c > a > b > m > p$ — many transactions share the prefix $f \to c \to a$. If we build a tree whose root is empty, and add each transaction as a path, common prefixes merge.

The tree for T1, T2, T5 shares $f \to c \to a$, with a count of 3 at each of those nodes. After all five transactions, the tree has dramatically fewer nodes than the original 5 × 5 = 25 item occurrences. Scanning this tree is much faster than scanning the database.

Then, to find frequent itemsets containing (say) item $p$, we need only look at the paths in the tree that end at a node labeled $p$. We extract those path prefixes (with their counts), forming a **conditional pattern base** for $p$, and build a smaller FP-tree from them (the **conditional FP-tree for $p$**), on which we recurse.

#### Connection to ML systems

FP-Growth is the workhorse behind most production frequent-itemset miners. Its parallel variant (PFP, implemented in Spark MLlib as `FPGrowth`) scales to very large transaction databases. In practice, "frequent pattern mining at scale" almost always means FP-Growth or a close relative.

---

### 2.2 Conceptual Foundations

#### Key definitions

- **FP-tree (Frequent-Pattern tree)**: a prefix tree built from transactions, where:
  - Each node stores an item label, a count, and a parent pointer.
  - A node-link pointer threads together all nodes with the same item label.
  - The root is a special empty node.
- **Header table**: a table listing each frequent 1-item, its total support count, and a pointer to the first node in the FP-tree with that label. Used to traverse all occurrences of an item.
- **Conditional pattern base (CPB) of item $i$**: the multiset of prefix paths of $i$ in the FP-tree. Each path carries the count of the $i$-node at its end. Intuitively, the CPB is a "subdatabase" containing exactly the transactions that include $i$, with $i$ itself removed.
- **Conditional FP-tree of item $i$**: an FP-tree built from the CPB of $i$, using a local minimum-support pruning step.
- **Suffix**: as we recurse, we track which items are *already* in the itemset being built. For the conditional tree of $i$, the suffix grows by $i$.

#### Algorithm at a conceptual level

1. **Scan 1**: count each item's support, build $L_1$ (frequent items), sort by descending frequency.
2. **Scan 2**: build the FP-tree. For each transaction, drop infrequent items, sort remaining items by $L_1$ order, insert the resulting sorted list as a path in the tree (merging with existing prefixes and incrementing counts).
3. **Mine recursively**: for each item $i$ in $L_1$, bottom-up (least frequent first):
   a. Traverse node-links from the header table to collect all prefix paths → CPB.
   b. Build the conditional FP-tree from the CPB, pruning items below support.
   c. If the conditional tree is non-empty, emit the itemsets it encodes (with $i$ as suffix), and recurse on it.

#### Why this works

The conditional FP-tree of $i$ represents exactly the transactions in which $i$ occurs, restricted to other frequent items. So mining itemsets from this subtree is equivalent to mining itemsets that co-occur with $i$. By recursing, we build up all frequent itemsets containing $i$ in a divide-and-conquer fashion. Crucially, this requires *no candidate generation* — we only materialize itemsets that are actually supported by real transactions.

#### Underlying assumptions

1. **The FP-tree fits in memory.** If it does not, partitioned/parallel variants are needed.
2. **Transactions share many common prefixes after frequency sorting.** This is empirically true in market-basket-style data; not always true in, say, uniformly random data where FP-Growth offers little advantage.
3. **Same binary-occurrence, static-database, i.i.d. assumptions as Apriori.**

#### What breaks when assumptions fail

- **Sparse, low-overlap data**: the FP-tree barely compresses the database. You pay the construction cost for little gain.
- **Dense data with huge frequent item sets**: conditional FP-trees themselves blow up. FP-Growth still helps relative to Apriori but may still be slow.
- **Out-of-memory trees**: you need partitioned variants (PFP, Spark's MLlib implementation).

---

### 2.3 Mathematical Formulation

Unlike Apriori, FP-Growth is better described as a structural/recursive algorithm than via a single compact equation. We formalize the key objects.

#### The FP-tree

An FP-tree is a rooted tree $\mathcal{T} = (V, E, \text{root})$ where each node $v \in V \setminus \{\text{root}\}$ carries:
- $\text{item}(v) \in \mathcal{I}$
- $\text{count}(v) \in \mathbb{Z}_{>0}$
- A parent pointer to its parent in the tree.
- A node-link pointer to the next node with the same item label (forming a linked list across the tree).

**Path-count semantics.** The count on node $v$ equals the number of transactions whose prefix (after filtering and sorting) ends at or passes through $v$. Equivalently, if $v$ has children $c_1, \dots, c_r$ all with item labels different from $\text{item}(v)$,

$$
\text{count}(v) \geq \sum_{j=1}^r \text{count}(c_j)
$$

with equality iff no transaction ended exactly at $v$.

**Header table.** A map $H : L_1 \to V^* $ from each frequent item to the list of nodes with that label, plus each item's total support.

#### Construction

Given $\mathcal{D}$, sorted-by-frequency order $\prec$ on $L_1$:

```
create root of T (null)
for each transaction T in D:
    remove items in T not in L_1
    sort remaining items by ≺
    insert_tree(sorted_items, root)

insert_tree([p | P], node):
    if node has child c with item(c) = p:
        count(c) += 1
    else:
        create new child c with item p, count 1
        link c into header table for item p
    if P not empty:
        insert_tree(P, c)
```

#### Conditional pattern base

For item $i$, traverse the node-link list in the header table. For each node $v_j$ with $\text{item}(v_j) = i$ and count $c_j$:
- Walk up from $v_j$'s parent to root, collecting item labels: this is the **prefix path** $\pi_j$.
- Emit the entry $(\pi_j, c_j)$.

The **conditional pattern base** is $\text{CPB}(i) = \{(\pi_j, c_j)\}_j$, a multiset of weighted transactions.

**Intuition**: each prefix path $\pi_j$ is a "compressed transaction" (the set of items co-occurring with $i$ along that branch), and $c_j$ is how many original transactions it stands for.

#### Conditional FP-tree

Construct an FP-tree from $\text{CPB}(i)$ exactly as above: compute each item's total weighted count across the CPB, keep items with total count $\geq \sigma_{\text{abs}}$, re-sort, and insert each weighted prefix path. Call this tree $\mathcal{T}_i$, the **conditional FP-tree of $i$** (or of suffix $\{i\}$).

#### Mining

```
FPgrowth(tree T, suffix α):
    if T has a single path P:
        for every non-empty subset β of items in P:
            emit itemset β ∪ α with support = min count along β
    else:
        for each item a in T's header table, in ascending support:
            emit itemset β = {a} ∪ α with support = a's count in T
            construct CPB(a) and conditional FP-tree T_a
            if T_a is non-empty:
                FPgrowth(T_a, β)
```

**Single-path optimization.** If a conditional FP-tree has only one path $p_1 \to p_2 \to \dots \to p_r$ (after pruning), all $2^r - 1$ non-empty subsets of its items are frequent itemsets (combined with the current suffix), with support equal to the minimum count along the subset. This avoids further recursion.

#### Complexity

No clean closed-form. Empirically:
- Construction: $O(|\mathcal{D}| \cdot \ell)$ where $\ell$ is average transaction length — two passes.
- Mining: depends on tree size and frequent-itemset count. On typical data FP-Growth is significantly faster than Apriori, often by an order of magnitude.

Theoretical worst-case is no better than Apriori — in pathological data FP-trees can be as large as the database — but practice overwhelmingly favors FP-Growth.

---

### 2.4 Worked Example

Reuse the toy database from Topic 1:

| TID | Items       |
|-----|-------------|
| 1   | A, B, D     |
| 2   | B, C, E     |
| 3   | A, B, C, E  |
| 4   | B, E        |
| 5   | A, C, E     |

$N = 5$, $\sigma_{\text{abs}} = 2$.

#### Scan 1: item counts and ordering

Counts: A=3, B=4, C=3, D=1, E=4.

Prune D (count 1 < 2). Sort remaining by descending count, break ties lexicographically:
- B (4), E (4), A (3), C (3).

Use frequency order $\prec$: $B \prec E \prec A \prec C$.

#### Scan 2: build FP-tree

Re-sort each transaction by $\prec$ and insert:

- T1: {A, B, D} → drop D → {A, B} → sorted: B, A.
- T2: {B, C, E} → sorted: B, E, C.
- T3: {A, B, C, E} → sorted: B, E, A, C.
- T4: {B, E} → sorted: B, E.
- T5: {A, C, E} → sorted: E, A, C.  *(no B)*

Insert sequentially:

**After T1 (B, A):**
```
root
└── B:1
    └── A:1
```

**After T2 (B, E, C):** path starts at B (merge), then E (new child of B), then C.
```
root
└── B:2
    ├── A:1
    └── E:1
        └── C:1
```

**After T3 (B, E, A, C):** B (merge, count 3), E (merge under B, count 2), A (new under E), C (new under A).
```
root
└── B:3
    ├── A:1
    └── E:2
        ├── C:1
        └── A:1
            └── C:1
```

**After T4 (B, E):** B (merge, count 4), E (merge, count 3). No deeper insertion.
```
root
└── B:4
    ├── A:1
    └── E:3
        ├── C:1
        └── A:1
            └── C:1
```

**After T5 (E, A, C):** starts at root with E — a new branch.
```
root
├── B:4
│   ├── A:1
│   └── E:3
│       ├── C:1
│       └── A:1
│           └── C:1
└── E:1
    └── A:1
        └── C:1
```

**Header table** (with node-link pointers linking all occurrences):

| Item | Total count | Node-links                                        |
|------|-------------|---------------------------------------------------|
| B    | 4           | → B:4                                             |
| E    | 4           | → E:3 (under B) → E:1 (root)                      |
| A    | 3           | → A:1 (under B) → A:1 (under E-under-B) → A:1 (under E-root) |
| C    | 3           | → C:1 (under E:3) → C:1 (under A-under-E:3) → C:1 (under A-under-E:1) |

#### Mining (process items in ascending support, so C, A, E, B)

**Item C (suffix {C}, support 3).** Emit {C}:3. Traverse node-links for C, find three nodes. Prefix paths:
- C:1 under (B:4 → E:3) → prefix path B, E with count 1
- C:1 under (B:4 → E:3 → A:1) → prefix B, E, A, count 1
- C:1 under (root → E:1 → A:1) → prefix E, A, count 1

**CPB(C)** = { (B,E):1, (B,E,A):1, (E,A):1 }.

Totals across the CPB: B=2, E=3, A=2. All ≥ 2 → all kept.

Sort by CPB-frequency (E=3, B=2, A=2; order $E \prec B \prec A$ breaking ties lexicographically) and reinsert:

- (B,E):1 → sorted: E, B → insert
- (B,E,A):1 → sorted: E, B, A → insert
- (E,A):1 → sorted: E, A → insert

Conditional FP-tree $\mathcal{T}_C$:
```
root
└── E:3
    ├── B:2
    │   └── A:1
    └── A:1
```

Mine $\mathcal{T}_C$ with suffix {C}. Not single-path (E has two children). Process items in ascending support: A (2), B (2), E (3).

- *Sub-item A in $\mathcal{T}_C$* (suffix {A,C}). Emit {A,C}:2. Prefix paths of A: under (E:3 → B:2) path E,B count 1; directly under E:3 path E count 1. CPB={(E,B):1, (E):1}. Totals: E=2, B=1. Prune B. Conditional tree has single node E:2.
  - Emit {E,A,C}:2.
  - Recurse on single-path tree (just E:2) with suffix {A,C}: nothing more to emit beyond what's already done.
- *Sub-item B in $\mathcal{T}_C$* (suffix {B,C}). Emit {B,C}:2. Prefix paths of B: under E:3 path E count 2. CPB={(E):2}. Totals E=2 ≥ 2. Conditional tree: E:2, single path.
  - Emit {E,B,C}:2.
- *Sub-item E in $\mathcal{T}_C$* (suffix {E,C}). Emit {E,C}:3. Prefix paths of E are empty (E is at root level in $\mathcal{T}_C$). No further recursion.

So from the C-branch: {C}:3, {A,C}:2, {E,A,C}:2, {B,C}:2, {E,B,C}:2, {E,C}:3.

**Item A (suffix {A}, support 3).** Emit {A}:3. Prefix paths of A:
- A:1 under B:4 → prefix B, count 1
- A:1 under (B:4 → E:3) → prefix B,E, count 1
- A:1 under (root → E:1) → prefix E, count 1

CPB(A)={(B):1, (B,E):1, (E):1}. Totals: B=2, E=2. Both kept. Order (tie, alphabetical) $B \prec E$.

Reinsert:
- (B):1 → B
- (B,E):1 → B, E
- (E):1 → E

Conditional tree $\mathcal{T}_A$:
```
root
├── B:2
│   └── E:1
└── E:1
```

Mine with suffix {A}. Not single-path.
- *E in $\mathcal{T}_A$* (suffix {E,A}). Emit {E,A}:2. Prefix paths of E: under B:2 path B, count 1; root path count 1. CPB={(B):1, ():1}. Totals: B=1 < 2, prune. Conditional tree empty. Stop.
- *B in $\mathcal{T}_A$* (suffix {B,A}). Emit {B,A}:2. Prefix paths of B: empty. Stop.

From A-branch: {A}:3, {E,A}:2, {B,A}:2.

**Item E (suffix {E}, support 4).** Emit {E}:4. Prefix paths of E:
- E:3 under B:4 → prefix B, count 3
- E:1 under root → prefix ∅, count 1

CPB(E)={(B):3, ():1}. Total B=3 ≥ 2. Conditional tree: B:3 (single path).
- Emit {B,E}:3.

**Item B (suffix {B}, support 4).** Emit {B}:4. Prefix paths of B: only path, empty. Nothing to recurse on.

#### Collected frequent itemsets

{A}:3, {B}:4, {C}:3, {E}:4, {A,B}:2, {A,C}:2, {A,E}:2, {B,C}:2, {B,E}:3, {C,E}:3, {A,C,E}:2, {B,C,E}:2.

**Sanity check**: identical to Apriori's output on the same database. ✓

The mechanism was entirely different — no candidate generation, two DB scans total, all subsequent work in memory on the FP-tree.

---

### 2.5 Relevance to Machine Learning Practice

#### Where FP-Growth is used

- The default choice whenever you want frequent itemsets at scale. Spark MLlib, Mahout, and most modern rule miners use FP-Growth or a close relative.
- **PFP (Parallel FP-Growth)**: partition items among workers so each mines a subset of conditional FP-trees, then merge results. Powers web-scale frequent-pattern discovery.
- **Streaming / online variants**: such as FP-Stream for sliding windows over transaction streams.

#### When to use FP-Growth vs Apriori

- **Use FP-Growth** for most real problems: it is near-always faster.
- **Use Apriori** when:
  - The implementation is already available and the dataset is small.
  - Pedagogy / interpretability of the mining process matters.
  - You need the intermediate candidate sets $C_k$ for analysis.
- **Use neither** (go to Eclat, LCM, or approximate methods) when:
  - Data is dense and support is low — FP-tree itself becomes unwieldy.
  - You only need closed or maximal itemsets (LCM excels).
  - You need probabilistic guarantees or sampling-based approximations.

#### Trade-offs

| Concern             | FP-Growth                                                                |
|---------------------|---------------------------------------------------------------------------|
| Memory              | Requires FP-tree in memory. Can be large for dense data.                 |
| I/O                 | Exactly two database scans. Huge advantage over Apriori for disk-bound data. |
| Speed               | Typically 5–100× faster than Apriori on market-basket-style data.        |
| Parallelism         | Naturally parallelizable by partitioning the header table — hence PFP.   |
| Implementation complexity | Substantially more complex than Apriori (trees, node-links, recursion). |
| Robustness to $\sigma$ | Similar hard-threshold issue; very low $\sigma$ still explodes.        |

---

### 2.6 Common Pitfalls & Misconceptions

1. **"FP-Growth avoids candidate generation entirely."** Technically true in that it never materializes $C_k$, but the recursion implicitly enumerates the same frequent-itemset space. The *savings* come from sharing prefixes and avoiding database re-scans — not from searching less of the conceptual space.
2. **"FP-Growth always fits in memory."** Not guaranteed. For dense data or tiny $\sigma$, the FP-tree can rival the database in size. Production systems use partitioned variants.
3. **Forgetting to sort items by frequency before inserting.** Different orders produce different trees with very different sizes. Frequency-descending maximizes prefix sharing in typical data.
4. **Sorting by frequency ascending.** This gives correct results but produces much less compact trees (opposite of the usual rule).
5. **Confusing conditional pattern base with conditional FP-tree.** The CPB is the raw list of weighted prefix paths; the conditional FP-tree is the FP-tree you build from the CPB, after re-pruning items that are locally infrequent.
6. **Skipping the local re-prune step** when building a conditional FP-tree. You must recompute support within the CPB; an item that was globally frequent may be locally infrequent in the sub-database.
7. **Assuming the single-path optimization is always applicable.** Only conditional trees that really are single paths admit the $2^r - 1$ shortcut. If there is branching, you must recurse.
8. **Using FP-Growth on very small datasets.** The overhead of tree construction is wasted; Apriori is just as fast.
9. **Believing FP-Growth handles the same interpretability downsides as Apriori.** It does: the output is still a pile of rules, and pruning/ranking them (next topic's metrics) is still required.

---

---

## Topic 3 — Evaluation Metrics Beyond Support/Confidence/Lift

### 3.1 Motivation & Intuition

#### Why support and confidence alone are insufficient

Consider two rules:

- $R_1$: bread $\Rightarrow$ milk, with support 0.4 and confidence 0.8.
- $R_2$: caviar $\Rightarrow$ champagne, with support 0.001 and confidence 0.95.

Which is more interesting? Using only a support threshold, $R_2$ might be filtered out. Using only a confidence threshold, both pass. Yet they tell us very different things. Support measures *how often* the pattern occurs; confidence measures *how reliable* the implication is given the antecedent.

Even together, support and confidence miss many important aspects:

- **Independence check.** Consider cereal $\Rightarrow$ milk in a store where 95% of shoppers buy milk anyway. A rule with 90% confidence looks great, but it's worse than the baseline. **Lift** fixes this — but lift itself is symmetric and doesn't distinguish rule direction.
- **Strength of implication failure.** Lift tells us the multiplicative effect on the consequent's probability. It does not tell us how much a rule's failure would surprise us — which is what **conviction** captures.
- **Departure from independence in absolute terms.** Lift is a ratio and can be numerically huge for low-support items simply because the denominator is small. **Leverage** gives an absolute-scale measure.
- **Group-level consistency.** When we care about how internally-coherent a set of items is (rather than an ordered rule), measures like **collective strength** and **all-confidence** come into play.

We need a family of measures, each answering a different question.

#### Connection to ML systems

Ranking rules is exactly analogous to ranking candidate features, candidate ad creatives, or candidate recommendations. Different metrics optimize different downstream outcomes. In practice, ML engineers typically:
- Filter by support (for statistical reliability),
- Filter by confidence (for rule-quality baseline),
- Rank by lift or a lift-variant (for surprise / informativeness),
- Sanity-check with a significance test or an independence-aware metric (to weed out spurious patterns).

---

### 3.2 Conceptual Foundations

#### Key terms

- **$P(X)$, $P(Y)$, $P(X,Y)$**: shorthand for empirical probabilities estimated from the database. $P(X) = \text{supp}(X)$, etc. We switch freely between these notations.
- **Contingency table (2×2)** for a rule $X \Rightarrow Y$:

  |          | $Y$             | $\neg Y$        | total    |
  |----------|-----------------|-----------------|----------|
  | $X$      | $P(X,Y)$        | $P(X, \neg Y)$  | $P(X)$   |
  | $\neg X$ | $P(\neg X, Y)$  | $P(\neg X, \neg Y)$ | $P(\neg X)$ |
  | total    | $P(Y)$          | $P(\neg Y)$     | $1$      |

  Every pairwise association metric is a function of this 2×2 table.

- **Null hypothesis of independence**: $P(X,Y) = P(X) P(Y)$. Many metrics measure departure from this.

#### The measures, at a high level

- **Support**: frequency.
- **Confidence**: $P(Y \mid X)$.
- **Lift**: $P(X,Y) / (P(X) P(Y))$ — multiplicative departure from independence.
- **Leverage (Piatetsky-Shapiro's measure)**: $P(X,Y) - P(X)P(Y)$ — additive departure.
- **Conviction**: $P(X) P(\neg Y) / P(X, \neg Y)$ — degree to which the rule's failure exceeds what independence would predict.
- **Collective strength**: a symmetric measure comparing actual co-occurrence+co-absence against expected.
- **Jaccard, cosine, all-confidence, max-confidence, Kulczynski, IR, φ-coefficient** — various other scalar summaries; we include a brief catalog.

#### Assumptions

All these metrics share:
1. The contingency table is a faithful estimate of the joint distribution. Fails for tiny support.
2. Interactions are pairwise (between $X$ and $Y$ as groups). Fails for high-order interactions unless you treat an itemset as a whole.
3. The null of independence is meaningful. In causal or confounded data, "independence" may not be the right baseline.

---

### 3.3 Mathematical Formulation

We build up each metric, show its derivation, and place it in context.

#### 3.3.1 Leverage

**Definition**:

$$
\text{lev}(X \Rightarrow Y) = P(X,Y) - P(X) P(Y).
$$

**Derivation intuition**: under independence, $P(X,Y) = P(X)P(Y)$. Leverage is the *raw excess* co-occurrence over the independence baseline.

**Range**: $[-0.25, 0.25]$.

*Proof of bound.* Let $a = P(X,Y)$, $x = P(X)$, $y = P(Y)$. The constraint $0 \leq a \leq \min(x,y)$ and the Fréchet bound $a \geq \max(0, x + y - 1)$ give the feasible region. The quantity $a - xy$ is maximized when $a = \min(x,y) = x = y = 0.5$, giving $0.5 - 0.25 = 0.25$; minimized when $a = \max(0, x+y-1) = 0$ with $x = y = 0.5$, giving $-0.25$. $\blacksquare$

**Key properties**:
- $\text{lev} = 0$ iff independent.
- $\text{lev} > 0$: positive association.
- Symmetric: $\text{lev}(X \Rightarrow Y) = \text{lev}(Y \Rightarrow X)$.
- On an absolute scale (not a ratio), so easier to compare across rules with different supports.
- Downward closed in a useful sense: if $\text{lev}(X,Y)$ is small, supersets cannot have arbitrarily high leverage (see the *Magnum Opus* system by Webb for uses in rule discovery).

**When to prefer leverage**: when you want a single scale-consistent number that is near zero for weak effects and monotone in the strength of the effect, without the ratio-blowup of lift.

#### 3.3.2 Conviction

**Definition**:

$$
\text{conv}(X \Rightarrow Y) = \frac{P(X) P(\neg Y)}{P(X, \neg Y)} = \frac{1 - P(Y)}{1 - \text{conf}(X \Rightarrow Y)}.
$$

**Derivation**. Consider the "counter-rule" $X \Rightarrow \neg Y$. Its confidence is $P(\neg Y \mid X) = 1 - \text{conf}(X \Rightarrow Y)$. Under independence, this would equal $P(\neg Y) = 1 - P(Y)$. Conviction is the ratio:

$$
\text{conv}(X \Rightarrow Y) = \frac{P(\neg Y)}{P(\neg Y \mid X)} = \frac{1 - P(Y)}{1 - \text{conf}(X \Rightarrow Y)}.
$$

Equivalently, using $P(X, \neg Y) = P(X) - P(X,Y) = P(X)(1 - \text{conf})$:

$$
\text{conv}(X \Rightarrow Y) = \frac{P(X)(1 - P(Y))}{P(X)(1 - \text{conf}(X \Rightarrow Y))} \cdot \frac{1}{1} \cdot \frac{P(X)}{P(X)} = \frac{P(X) P(\neg Y)}{P(X, \neg Y)}.
$$

**Range**: $[0.5, \infty]$ in typical cases.
- $\text{conv} = 1$: independence.
- $\text{conv} = \infty$: rule holds always (i.e., $\text{conf} = 1$, so the rule is a logical implication in the data).
- $\text{conv} < 1$: negative association (antecedent makes consequent *less* likely).

**Key properties**:
- **Asymmetric**: $\text{conv}(X \Rightarrow Y) \neq \text{conv}(Y \Rightarrow X)$ in general. Good: captures the directionality that lift misses.
- **Infinite for logical rules**: unlike confidence (which tops out at 1) and lift (which saturates at $1/P(Y)$ when confidence is 1), conviction genuinely distinguishes perfect rules from merely strong ones.
- **Undefined when $\text{conf} = 1$**: handle via $+\infty$ convention.

**When to prefer conviction**: when you care about directed rules and want a measure that rewards perfect (or near-perfect) implications strongly. A rule with confidence 0.99 and baseline 0.90 has lift only $\approx 1.1$ but can have very high conviction.

#### 3.3.3 Collective strength

The **collective strength** of an itemset $Z$ (not a rule — it is a *group* measure) is designed to quantify the internal coherence of an itemset relative to what independence would predict.

Define:
- **Violation rate** $v(Z)$: the empirical probability that a transaction contains *some but not all* items of $Z$, i.e., a "violation" of the ideal co-occurrence. Formally,
  $$v(Z) = P(\text{some } i \in Z \text{ in } T, \text{ some } j \in Z \text{ not in } T) = 1 - P(Z \subseteq T) - P(Z \cap T = \varnothing).$$
- **Expected violation rate under independence**:
  $$E[v(Z)] = 1 - \prod_{i \in Z} P(i) - \prod_{i \in Z} (1 - P(i)).$$

**Definition**:

$$
\text{CS}(Z) = \frac{1 - v(Z)}{1 - E[v(Z)]} \cdot \frac{E[v(Z)]}{v(Z)}.
$$

**Interpretation**:
- $\text{CS} = 1$: observed violation rate exactly matches independence.
- $\text{CS} = 0$: every transaction violates — items are perfectly negatively correlated at the group level.
- $\text{CS} = \infty$: no transaction violates — items always appear together or never appear together.

**Range**: $[0, \infty]$.

**Properties**: collective strength is *symmetric* across all items of $Z$; it treats them as an unordered group. It penalizes both rare co-occurrence and excessive co-absence deviating from independence.

**When to prefer collective strength**: when you want a single score for how "tight" an itemset is as a group, especially for $|Z| \geq 3$ where pairwise measures aggregate awkwardly. Introduced by Aggarwal and Yu, 1998.

#### 3.3.4 A catalog of other useful measures

Given a rule $X \Rightarrow Y$ (or, for symmetric measures, an unordered pair $\{X, Y\}$):

- **Jaccard coefficient** (symmetric):
  $$J(X,Y) = \frac{P(X,Y)}{P(X) + P(Y) - P(X,Y)}.$$
  The fraction of transactions containing either $X$ or $Y$ that contain both. Insensitive to items outside $X \cup Y$.

- **Cosine similarity** (symmetric):
  $$\cos(X,Y) = \frac{P(X,Y)}{\sqrt{P(X) P(Y)}}.$$
  Null-invariant: does not change when the number of transactions containing neither $X$ nor $Y$ changes. Useful in sparse data where absence is the norm.

- **All-confidence**:
  $$\text{all-conf}(X,Y) = \min\{\text{conf}(X \Rightarrow Y), \text{conf}(Y \Rightarrow X)\} = \frac{P(X,Y)}{\max\{P(X), P(Y)\}}.$$
  A conservative measure: both directions must be strong.

- **Max-confidence**:
  $$\text{max-conf}(X,Y) = \max\{\text{conf}(X \Rightarrow Y), \text{conf}(Y \Rightarrow X)\}.$$
  The optimistic counterpart.

- **Kulczynski measure** (symmetric):
  $$K(X,Y) = \frac{1}{2}\left(\text{conf}(X \Rightarrow Y) + \text{conf}(Y \Rightarrow X)\right).$$
  Average of the two directional confidences.

- **Imbalance ratio (IR)**:
  $$\text{IR}(X,Y) = \frac{|P(X) - P(Y)|}{P(X) + P(Y) - P(X,Y)}.$$
  Measures asymmetry of supports. Paired with Kulczynski, separates "balanced strong" rules from "lopsided coincidence."

- **φ-coefficient (Pearson correlation on 0/1 indicators)**:
  $$\phi(X,Y) = \frac{P(X,Y) - P(X)P(Y)}{\sqrt{P(X)P(\neg X)P(Y)P(\neg Y)}}.$$
  Equivalent to Pearson correlation for binary variables; related to $\chi^2$ by $\chi^2 = N \phi^2$.

- **Pointwise mutual information (PMI)**:
  $$\text{PMI}(X,Y) = \log \frac{P(X,Y)}{P(X)P(Y)} = \log \text{lift}(X,Y).$$

**Null-invariance** deserves special note. A measure $M(X,Y)$ is **null-invariant** if it is unchanged when the count of transactions containing *neither* $X$ nor $Y$ changes. Lift, leverage, conviction, and φ are *not* null-invariant: adding a pile of empty transactions dilutes them. Jaccard, cosine, all-confidence, max-confidence, and Kulczynski *are* null-invariant. For very sparse data (most transactions contain none of the items of interest), null-invariance is often desirable — otherwise you get misleadingly strong-looking associations driven entirely by how you filtered your data.

---

### 3.4 Worked Examples

Use the toy database from Topic 1/2. Focus on the rule $A \Rightarrow C$ and the itemset $\{A, C, E\}$.

Supports from earlier: $P(A) = 0.6$, $P(C) = 0.6$, $P(E) = 0.8$, $P(A,C) = 0.4$, $P(A,C,E) = 0.4$.

#### 3.4.1 Basic metrics for $A \Rightarrow C$

- Support: $P(A,C) = 0.4$.
- Confidence: $P(A,C)/P(A) = 0.4/0.6 \approx 0.667$.
- Lift: $0.667 / 0.6 \approx 1.111$. Slight positive association.
- Leverage: $P(A,C) - P(A)P(C) = 0.4 - 0.36 = 0.04$. Small but positive.
- Conviction: $(1 - 0.6)/(1 - 0.667) = 0.4/0.333 = 1.2$.

Cross-checks:
- All numbers exceed the "independence" values (lift 1, leverage 0, conviction 1), consistently indicating positive association. ✓
- Conviction (1.2) indicates the rule's failure is 20% less frequent than independence would predict.

#### 3.4.2 Collective strength of $\{A, C, E\}$

- Joint occurrence: $P(\text{all 3 in } T) = P(A,C,E) = 0.4$ (TIDs 3, 5).
- Joint absence: $P(\text{none of 3 in } T)$. Look at each TID:
  - T1 (A,B,D): contains A. Not "none."
  - T2 (B,C,E): contains C,E. Not "none."
  - T3 (A,B,C,E): all three. Not "none."
  - T4 (B,E): contains E. Not "none."
  - T5 (A,C,E): all three. Not "none."
  Every transaction contains at least one of $A,C,E$. So $P(\text{none}) = 0$.
- Violation rate: $v = 1 - 0.4 - 0 = 0.6$.
- Under independence: $\prod P(i) = 0.6 \cdot 0.6 \cdot 0.8 = 0.288$; $\prod (1-P(i)) = 0.4 \cdot 0.4 \cdot 0.2 = 0.032$. So $E[v] = 1 - 0.288 - 0.032 = 0.68$.
- Collective strength:
  $$\text{CS} = \frac{1 - 0.6}{1 - 0.68} \cdot \frac{0.68}{0.6} = \frac{0.4}{0.32} \cdot \frac{0.68}{0.6} = 1.25 \cdot 1.1333 \approx 1.417.$$

Interpretation: the itemset $\{A,C,E\}$ is about 1.4× more cohesive than independence would predict. Mildly positive group structure.

#### 3.4.3 Null-invariance illustration

Imagine we add 95 new transactions, each empty. The supports of all itemsets drop by a factor of $5/100 = 0.05$ of their originals. Specifically:
- $P(A) = 0.03, P(C) = 0.03, P(A,C) = 0.02$ (just 2 of 100).
- Lift: $0.02 / (0.03 \cdot 0.03) \approx 22.2$.  Explodes from 1.11 to 22.2!
- Leverage: $0.02 - 0.0009 = 0.0191$. Falls from 0.04 to 0.019.
- Jaccard: $P(A,C)/(P(A)+P(C)-P(A,C)) = 0.02/(0.03 + 0.03 - 0.02) = 0.02/0.04 = 0.5$. Unchanged from before: $0.4/(0.6+0.6-0.4) = 0.4/0.8 = 0.5$. ✓ null-invariant.
- Cosine: $0.02/\sqrt{0.03 \cdot 0.03} = 0.02/0.03 \approx 0.667$. Same as before: $0.4/\sqrt{0.36} = 0.667$. ✓.

**Takeaway**: in sparse data, lift-like measures can give wildly different answers depending on how you defined the "universe" of transactions. If you're unsure whether to include those 95 empty transactions, use a null-invariant measure.

---

### 3.5 Relevance to Machine Learning Practice

#### When to use which metric

- **Initial filtering**: support (statistical reliability) and confidence (baseline quality).
- **Ranking interestingness**: lift or conviction for directed rules; Kulczynski + IR for balanced symmetric rules.
- **Sparse data where most transactions are empty**: use null-invariant measures (Jaccard, cosine, all-confidence, Kulczynski).
- **Three-or-more-item groupings**: collective strength or all-confidence across items.
- **When formal statistical significance is required**: φ-coefficient + $\chi^2$, Fisher exact test, or a permutation test. Metrics like lift alone are *not* significance tests.

#### In production ML systems

- **Recommender systems**: "customers who bought X also bought Y" is a lift-ranked association rule. Modern systems use lift as a candidate generator and a learned model for final ranking.
- **Market basket monitoring**: drift in top-lift or top-leverage rules can indicate changing customer behavior (covered in Topic 4).
- **Feature engineering**: mine high-leverage rules and turn the antecedent itemsets into binary features for downstream supervised models.
- **Causal discovery baselines**: rules with extremely high lift or conviction are candidate causal hypotheses (to be confirmed with intervention or careful confounder analysis).
- **Monitoring and alerting**: when a rule's lift sharply departs from a moving baseline, that may indicate a pricing change, supply issue, or promotion interaction.

#### Trade-offs

| Metric              | Symmetric? | Null-invariant? | Rewards logical rules? | Scale        |
|---------------------|------------|------------------|------------------------|--------------|
| Support             | —          | no               | no                     | [0,1]        |
| Confidence          | no         | no               | yes (→1)              | [0,1]        |
| Lift                | yes        | no               | caps at 1/P(Y)         | [0,∞)        |
| Leverage            | yes        | no               | no                     | [−0.25,0.25] |
| Conviction          | no         | no               | **yes (→∞)**          | [0,∞]        |
| Jaccard             | yes        | yes              | no                     | [0,1]        |
| Cosine              | yes        | yes              | no                     | [0,1]        |
| All-confidence      | yes        | yes              | no                     | [0,1]        |
| Kulczynski          | yes        | yes              | no                     | [0,1]        |
| Collective strength | yes        | no               | yes                    | [0,∞]        |
| φ-coefficient       | yes        | no               | no                     | [−1,1]       |

Rule of thumb: report at least one null-invariant and one non-null-invariant metric.

---

### 3.6 Common Pitfalls & Misconceptions

1. **Using lift as a significance measure.** Lift has no built-in notion of sample size or statistical significance. A lift of 10 from a single transaction is meaningless. Pair lift with a $\chi^2$ or Fisher exact p-value.
2. **Ignoring the asymmetry of directed rules.** Using a symmetric measure (lift, leverage) to rank directed rules throws away directional information that may matter.
3. **Applying non-null-invariant measures to heterogeneous transaction sets.** If you mine over "all site visits" vs "only logged-in-user visits," lift will change dramatically for the same pair of items.
4. **Reporting conviction for a perfectly confident rule without handling infinity.** Some libraries return `inf`, others `NaN`, others very large numbers. Decide explicitly.
5. **Thinking collective strength is a rule metric.** It's an itemset metric — it does not care about antecedent/consequent split.
6. **Chasing high-lift-low-support rules as "gold."** Very high lift at tiny support is often noise. Leverage controls for this by folding in the support magnitude.
7. **Using a single metric.** The "rule interestingness" literature converges on "use multiple metrics; be suspicious of anything that's great on only one." Tan, Kumar, and Srivastava's 2002 paper tabulates properties of 21+ measures.
8. **Forgetting that all these metrics estimate population-level probabilities from a sample.** Sampling error propagates; confidence intervals are informative (via bootstrap or asymptotic approximations).

---

---

## Topic 4 — Applications: Market Basket Analysis & Cross-Selling

### 4.1 Motivation & Intuition

#### The business problem

A retailer wants to answer questions like:
- *Which products are bought together?* (assortment planning)
- *Which products, if placed near each other, increase combined sales?* (store layout)
- *Which product can we recommend to a customer already holding basket $B$?* (real-time recommendation)
- *Which combination of products should we discount to maximize margin?* (promotion design)
- *When customers buy X, are they more or less likely to also buy Y this time vs. six months ago?* (trend/drift monitoring)
- *Are there segments of customers with systematically different basket compositions?* (segmentation)

**Market basket analysis (MBA)** is the name for applying association rule mining (and related techniques) to answer these. **Cross-selling** is the specific application of using such patterns to recommend additional items during a customer's purchase journey.

#### Why it's hard

- Retailers often carry tens of thousands of SKUs. Pairwise combinations alone are in the hundreds of millions.
- Most products are bought rarely; most pairs never co-occur. The data is extremely sparse.
- Shopping behavior is strongly non-stationary (seasonal, promotional, weather-driven).
- "Interesting" rules are usually not the most frequent ones — finding them requires careful metric choice (Topic 3).
- Causation vs. association: just because people *do* buy X and Y together does not mean moving them next to each other will *increase* combined sales. Classical rule mining gives correlational insight only.

#### Connection to ML systems

Modern retailers combine rule-based and learned approaches:
- Frequent itemsets / high-lift pairs → candidate set for a recommender.
- A neural or gradient-boosted ranker → final ranking of candidates.
- Rule-based overrides for business logic ("never recommend alcohol to minors," "out-of-stock exclusions," etc.).
- Rules as interpretable monitors: if the top-lift rules change week-over-week, something is happening.

---

### 4.2 Conceptual Foundations

#### Data pipeline

1. **Transaction extraction.** A transaction is typically an order, a basket, or a session. The definition matters: basket ≠ session, and online ≠ in-store.
2. **Item normalization.** Products roll up into SKUs, families, categories, or brands. A rule at SKU level is actionable (place this specific item); a rule at category level is insightful (bakery → dairy).
3. **Filtering.** Remove ubiquitous items ("shopping bag," "loyalty coupon") that act as noise.
4. **Mining.** Apriori or FP-Growth produces frequent itemsets and rules.
5. **Ranking and pruning.** Use multiple metrics (Topic 3) and domain filters.
6. **Deployment.** Serve rules as a candidate generator for a recommender, store-layout planner, or promotion engine.

#### Key concepts unique to MBA

- **Redundant rules.** If $\{A\} \Rightarrow \{B\}$ has confidence 0.8 and $\{A,C\} \Rightarrow \{B\}$ has confidence 0.81, the second is essentially redundant. Formal redundancy notions (closed rules, essential rules, improvement threshold) remove them.
- **Closed itemset**: a frequent itemset $Z$ is *closed* if no superset of $Z$ has the same support. Reporting only closed frequent itemsets (and rules derived from them) eliminates a large class of redundancies.
- **Maximal itemset**: frequent itemset with no frequent superset. Sparser still, but loses support values for non-maximal frequent subsets.
- **Rule improvement**: $\text{improvement}(X \Rightarrow Y) = \text{conf}(X \Rightarrow Y) - \max_{X' \subsetneq X} \text{conf}(X' \Rightarrow Y)$. Keep only rules whose improvement exceeds a threshold.
- **Cross-selling vs up-selling.** Cross-selling suggests *additional* items (X ⇒ Y where Y complements X). Up-selling suggests a *higher-tier* version of X itself. Rule mining typically addresses cross-selling directly; up-selling needs tiered product hierarchies.
- **Hold-out evaluation.** Because rule mining is descriptive, evaluation requires a clear task definition. Typical setups: split transactions chronologically; mine rules on the first block; evaluate recommendation hit-rate on the second block.

#### Assumptions in MBA

1. **Transaction definition is correct and consistent.** If some baskets are split across orders and others are not, patterns are distorted.
2. **Items are well-deduplicated.** Two SKUs for "the same product in a different pack size" should often roll up.
3. **Transactions are exchangeable within the mining window.** Strong temporal drift violates this.
4. **Recommendations derived from co-occurrence actually change customer behavior.** This is an intervention claim that rules cannot verify — needs an A/B test.

---

### 4.3 Mathematical Formulation

#### Rule-based recommendation scoring

Given a customer's in-progress basket $B$ and a mined rule set $\mathcal{R}$, a score for recommending item $y \notin B$ is often:

$$
\text{score}(y \mid B) = \max_{X \subseteq B,\ (X \Rightarrow Y) \in \mathcal{R},\ y \in Y} f(\text{conf}(X \Rightarrow Y), \text{lift}(X \Rightarrow Y), \dots)
$$

where $f$ is a ranking combiner (e.g., weighted sum, product, or max over lift). Simpler variants use only the strongest matching rule.

**Top-$k$ recommendation**: rank all candidate $y$ by score and return the top $k$.

#### Lift-based co-purchase association

For each ordered pair $(i,j)$ with $i \in B$, compute $\text{lift}(i \Rightarrow j)$ and sum (or take the max) over $i \in B$ to score $j$. This degenerates rule mining to pairwise associations but is extremely fast and often competitive as a baseline.

#### Closed itemsets — formal definition

Define the **closure operator**:

$$
\text{cl}(Z) = \bigcap_{T \in \mathcal{D},\ Z \subseteq T} T.
$$

I.e., the intersection of all transactions containing $Z$. $Z$ is **closed** if $\text{cl}(Z) = Z$.

**Theorem** (completeness). The set of closed frequent itemsets, with their supports, is sufficient to reconstruct the support of every frequent itemset.

*Proof sketch.* For any frequent $W$, let $Z = \text{cl}(W)$. Then $Z \supseteq W$, and by construction every transaction containing $W$ also contains $Z$, so $\text{supp}(W) = \text{supp}(Z)$. Hence $Z$ is closed, frequent, and has the same support as $W$. Thus the closed frequent itemsets act as "representatives" for all frequent itemsets. $\blacksquare$

Closed itemsets can be dramatically fewer than all frequent itemsets, which is the basis of algorithms like CLOSET, CHARM, and LCM.

#### Rule improvement

$$
\text{improvement}(X \Rightarrow Y) = \text{conf}(X \Rightarrow Y) - \max_{X' \subsetneq X,\ X' \neq \varnothing} \text{conf}(X' \Rightarrow Y).
$$

A rule is **productive** (in Webb's terminology) if $\text{improvement} > 0$; strictly, one requires improvement to exceed a small threshold or pass a statistical significance test against each subset rule.

---

### 4.4 Worked Example

#### Setting

A grocery retailer has 500,000 baskets over a month, 8,000 SKUs, rolled up to 600 product categories. They want to (i) discover cross-sell opportunities and (ii) deploy a real-time "recommended next item" widget.

#### Step 1: define the transaction

A transaction = one checkout basket. Loyalty-card linked baskets are optionally aggregated per customer-day for another view.

#### Step 2: item normalization and filtering

- Roll SKU to category: 8,000 → 600.
- Drop universal items: "plastic bag" (98% of baskets), "loyalty discount marker" (78%). These add no signal and bloat the rule set.
- Drop very rare items (< 10 baskets). Not statistically reliable.

#### Step 3: mining

Run FP-Growth with $\sigma = 0.005$ (i.e., present in at least 2,500 of 500,000 baskets) and no upper length limit.

Result: ~45,000 frequent itemsets, of which ~3,100 are closed.

#### Step 4: rule generation

Generate rules from closed frequent itemsets with $\gamma = 0.20$. Roughly 12,000 rules emerge. Apply an improvement threshold of 0.02: drops to ~1,800. Filter further by lift ≥ 1.3: ~950 rules.

#### Step 5: illustrative rules (hypothetical but realistic)

| Rule                              | Support | Confidence | Lift | Conviction |
|-----------------------------------|---------|------------|------|------------|
| {pasta, pasta sauce} ⇒ {parmesan} | 0.018   | 0.41       | 3.2  | 1.5        |
| {diapers} ⇒ {baby wipes}          | 0.032   | 0.65       | 5.8  | 2.4        |
| {chips, salsa} ⇒ {beer}           | 0.011   | 0.48       | 2.9  | 1.6        |
| {ground beef} ⇒ {hamburger buns}  | 0.027   | 0.36       | 4.1  | 1.4        |
| {salmon, lemon, capers} ⇒ {dill}  | 0.0008  | 0.58       | 21   | 2.3        |

The salmon-capers-dill rule has extraordinary lift but tiny support: actionable only in niche gourmet segments.

#### Step 6: deployment

**Cross-sell widget** in the online cart:
- User has {pasta, pasta sauce} in cart.
- Lookup rules whose antecedent is a subset of the cart: {pasta} ⇒ ..., {pasta sauce} ⇒ ..., {pasta, pasta sauce} ⇒ ...
- Score candidate consequents by max-lift.
- Deduplicate against items already in cart; filter by stock availability.
- Show top 3: parmesan, garlic bread, red wine.

**Store layout**: place parmesan at the end of the pasta aisle.

**Promotion design**: run an A/B test discounting beer 10% when chips + salsa are in cart. Compare total margin and attachment rate to a holdout group. *This step is essential*: the rule only asserts correlation; the A/B test measures causal effect.

#### Step 7: monitoring

Weekly, compute lift of the top-100 rules and track each against a baseline. Alert on any rule whose lift falls below a threshold or changes by more than a configured percentage — often indicative of stock issues, pricing errors, or a shift in customer behavior.

---

### 4.5 Relevance to Machine Learning Practice

#### Integration with modern recommender systems

Association rules are rarely the *sole* recommender in a large retailer, but they serve several roles:
- **Candidate generation**: rules produce a manageable set of candidate items, which a neural or gradient-boosted ranker then scores.
- **Cold-start recommendations**: for new users or new items where collaborative filtering has no signal, category-level rules still fire.
- **Explanation layer**: rules give human-readable rationales ("users who bought pasta often buy parmesan"), useful for transparency and compliance.
- **Fallback**: when the learned model is unavailable (latency budget, outage), rules provide a safe default.

#### Trade-offs compared with collaborative filtering / matrix factorization / neural recommenders

| Concern                  | Rule-based MBA            | Learned recommender         |
|--------------------------|---------------------------|------------------------------|
| Interpretability         | Excellent                 | Poor (without post-hoc work) |
| Cold-start handling      | Decent (category rules)   | Poor without side features   |
| Peak accuracy            | Modest                    | High                         |
| Engineering cost         | Low                       | High                         |
| Non-stationarity         | Manual refresh required   | Continuous retraining        |
| Long-tail coverage       | Good (if $\sigma$ low)    | Often poor                   |
| Personalization          | Weak (basket-level only)  | Strong (user-level)          |

In practice: rules as a candidate generator or explanation layer; learned model as ranker. This hybrid is the industry default.

#### When to prefer pure rule-based

- Small retailers without ML infrastructure.
- Regulatory or audit environments demanding fully interpretable recommendations.
- Physical store layout, shelf placement, catalog design — decisions that are not real-time and benefit from human review of rules.
- Early-stage analytics where understanding *what* happens matters more than optimizing *predictions*.

---

### 4.6 Common Pitfalls & Misconceptions

1. **Assuming co-occurrence implies a good cross-sell.** People might buy X and Y together without the recommendation changing anything. Use A/B tests to measure lift in *sales* due to the recommendation, not lift in the statistical sense.
2. **Keeping universal items in the mining.** Rules involving "shopping bag" or "tax" dominate the output and are worthless.
3. **Using only support and confidence.** Leads to trivial rules (common items ⇒ common items). Use lift, leverage, and conviction.
4. **Ignoring the time dimension.** A rule mined over last year's data may be stale today. Mine over rolling windows and refresh frequently.
5. **Mining at the wrong granularity.** SKU-level mining is too fine for broad layout decisions; category-level is too coarse for item-level recommendations. Often run both.
6. **Not removing within-basket duplicates.** A basket containing two cartons of milk should count milk once.
7. **Cross-segment aggregation.** Mining across all customers can hide segment-specific patterns (e.g., patterns unique to parents with young children). Segment first, then mine — especially when segments have very different basket distributions.
8. **Recommending items the customer already has.** Trivial but common bug: filter against the current basket.
9. **Conflating session and basket.** An online session might contain browse events, cart additions, and removals; only purchases define the basket.
10. **Chasing astronomical lift at very low support.** Lift of 20 on 40 baskets is probably coincidence. Always apply a minimum support or a significance filter.
11. **Assuming rules generalize across geographies.** Regional, cultural, and seasonal shifts make rules location-specific.
12. **Forgetting inventory and margin.** A recommendation that promotes an out-of-stock or unprofitable item is actively harmful. Rule mining is step 1; business-layer filtering is step 2.

---

---

## Interview Preparation

A comprehensive question bank organized by difficulty tier. Answers are written to match the depth expected in a senior ML / applied scientist interview loop.

### Foundational Questions

**F1. What is an association rule, and what do support and confidence measure?**

*Answer.* An association rule is an expression $X \Rightarrow Y$ where $X, Y$ are disjoint non-empty itemsets drawn from a set of transactions. It is a statistical pattern, not a logical implication or a causal claim.
- **Support** is the fraction of transactions containing $X \cup Y$; it measures how *frequent* the joint pattern is: $\text{supp}(X \cup Y) = |\{T : X \cup Y \subseteq T\}| / N$.
- **Confidence** is an empirical conditional probability estimate: $\text{conf}(X \Rightarrow Y) = \text{supp}(X \cup Y)/\text{supp}(X) = \hat{P}(Y \subseteq T \mid X \subseteq T)$. It measures how *reliable* the rule is given the antecedent.

Both are necessary but neither is sufficient: high confidence on a common consequent may just reflect its marginal frequency.

**F2. State the Apriori property in your own words and explain why it matters.**

*Answer.* Apriori property (anti-monotonicity of support): all subsets of a frequent itemset are themselves frequent; equivalently, if any subset is infrequent, the superset is infrequent.

Why it matters: it lets us prune the search space without counting. At level $k$, we consider only $k$-itemsets all of whose $(k-1)$-subsets were frequent at level $k-1$. Without this, we would need to scan the database for every possible itemset, which is exponential in the number of items.

**F3. What's the difference between confidence and lift? When is one misleading?**

*Answer.* Confidence is $\hat{P}(Y \mid X)$; lift is $\hat{P}(Y \mid X) / \hat{P}(Y)$, the ratio of confidence to the baseline frequency of $Y$.

Confidence alone is misleading when the consequent is common: any rule with a highly prevalent $Y$ will have high confidence by default. For instance, if 90% of baskets contain a shopping bag, "anything ⇒ shopping bag" has 90% confidence but tells us nothing. Lift normalizes for this: $\text{lift} = 1$ means independent, $>1$ positive association, $<1$ negative.

**F4. What is the difference between a frequent itemset and an association rule?**

*Answer.* A frequent itemset is just a set of items with support $\geq \sigma$; it has no antecedent/consequent split. Rule generation is a separate step: for each frequent itemset $Z$ with $|Z| \geq 2$, partition it into non-empty disjoint $X$ and $Z \setminus X$, form the rule $X \Rightarrow Z \setminus X$, and keep it if its confidence meets the threshold. Multiple rules can come from one frequent itemset.

**F5. What are closed and maximal frequent itemsets?**

*Answer.*
- **Closed frequent itemset**: a frequent itemset $Z$ with no superset of the same support. Equivalent to $Z$ being a fixed point of the closure operator.
- **Maximal frequent itemset**: a frequent itemset with no frequent superset.

Every maximal itemset is closed, but not vice versa. Reporting only closed itemsets is *lossless* (supports of all frequent itemsets are recoverable). Reporting only maximal itemsets is *lossy for supports* but more compact. Closed itemsets eliminate redundancy without information loss.

**F6. Intuitively, why does FP-Growth avoid the level-wise passes that Apriori requires?**

*Answer.* Apriori needs one pass per length because its candidate generation depends on $L_{k-1}$, which in turn required counting in the previous pass. FP-Growth instead compresses the entire database into an FP-tree in two initial passes, and then does all subsequent work in-memory on the tree (and on recursively-derived conditional trees). The tree contains enough information to answer all support queries needed for mining, without re-reading the database.

---

### Mathematical Questions

**M1. Derive the Apriori join rule: why do we join only pairs of $L_{k-1}$ itemsets that share their first $k-2$ items?**

*Answer.* Assume items are totally ordered. Any frequent $k$-itemset $Z = \{z_1 < z_2 < \dots < z_k\}$ has $k$ different $(k-1)$-subsets obtained by removing one element. Two of these share $z_1, \dots, z_{k-2}$: namely $Z \setminus \{z_k\}$ and $Z \setminus \{z_{k-1}\}$. These two $(k-1)$-subsets differ only in their last element ($z_{k-1}$ vs $z_k$).

By the Apriori property, both are in $L_{k-1}$. So to obtain $Z$ as a candidate, we join these two specific $(k-1)$-subsets. Restricting the join to only pairs sharing a $(k-2)$-prefix has two consequences:
1. It suffices (every frequent $k$-itemset is produced exactly once).
2. It's non-redundant (pairs not sharing a $(k-2)$-prefix either produce invalid candidates or duplicate already-generated candidates).

Any other pair of frequent $(k-1)$-itemsets either produces a set of cardinality other than $k$ (impossible, since their union must be size $k$ to be a candidate, and two sets of size $k-1$ with union size $k$ must share exactly $k-2$ elements) or duplicates a candidate already generated from the $(k-2)$-prefix pair. Thus joining on the $(k-2)$-prefix is both complete and minimal. $\blacksquare$

**M2. Show that confidence is anti-monotonic in the antecedent direction.**

*Answer.* Claim: for a fixed $Z$, if $X' \subsetneq X \subseteq Z$ with both partitions yielding valid rules, then $\text{conf}(X' \Rightarrow Z \setminus X') \leq \text{conf}(X \Rightarrow Z \setminus X)$.

Proof: both rules have the same numerator $\text{supp}(Z)$. Denominators are $\text{supp}(X')$ and $\text{supp}(X)$ respectively. Since $X' \subseteq X$, every transaction containing $X$ also contains $X'$, so $\text{supp}(X') \geq \text{supp}(X)$. Thus

$$
\text{conf}(X' \Rightarrow Z \setminus X') = \frac{\text{supp}(Z)}{\text{supp}(X')} \leq \frac{\text{supp}(Z)}{\text{supp}(X)} = \text{conf}(X \Rightarrow Z \setminus X). \quad \blacksquare
$$

This lets us prune rule generation: if a rule fails the confidence threshold, all rules obtained by further shrinking the antecedent (equivalently, further growing the consequent) also fail.

**M3. Prove the range of leverage: $\text{lev} \in [-0.25, 0.25]$.**

*Answer.* Let $a = P(X,Y), x = P(X), y = P(Y)$, with $a, x, y \in [0,1]$. By the Fréchet–Hoeffding bounds, $\max(0, x+y-1) \leq a \leq \min(x, y)$. Leverage $= a - xy$.

*Upper bound.* Maximize $a - xy$ over feasible $a, x, y$. Given $x, y$, maximum $a = \min(x, y)$. WLOG $x \leq y$: upper value is $x - xy = x(1 - y)$. Maximize $x(1 - y)$ subject to $x \leq y$: set $y = x$ (pushing $y$ down to its lower bound $x$ increases $1-y$). Then maximize $x(1-x)$ over $x \in [0,1]$: maximum at $x = 0.5$, giving $0.25$.

*Lower bound.* Minimize $a - xy$. Given $x, y$, minimum $a = \max(0, x+y-1)$.
- If $x + y \leq 1$: $a = 0$, giving $-xy$. Minimum over $x + y \leq 1$ is at $x = y = 0.5$: $-0.25$.
- If $x + y > 1$: $a = x + y - 1$, giving $x + y - 1 - xy = -(1-x)(1-y)$. Maximum magnitude at $x = y = 0.5$... which violates $x+y > 1$. Boundary: $x + y = 1$, giving 0. Interior values give magnitudes less than 0.25.

Thus leverage $\in [-0.25, 0.25]$. $\blacksquare$

**M4. Derive conviction from first principles and explain the formula $\text{conv} = P(\neg Y) / P(\neg Y \mid X)$.**

*Answer.* We want a measure that is $\infty$ when the rule is logically exact (never violated) and $1$ under independence, and that captures *rule failures* rather than rule successes.

Rule $X \Rightarrow Y$ fails on a transaction iff $X \subseteq T$ and $Y \not\subseteq T$. The probability of failure is $P(X, \neg Y)$. Under independence, this equals $P(X) P(\neg Y)$. The ratio of expected failures under independence to actual failures is

$$
\text{conv}(X \Rightarrow Y) = \frac{P(X) P(\neg Y)}{P(X, \neg Y)}.
$$

Factor $P(X)$ out:

$$
= \frac{P(\neg Y)}{P(X, \neg Y)/P(X)} = \frac{P(\neg Y)}{P(\neg Y \mid X)} = \frac{1 - P(Y)}{1 - P(Y \mid X)} = \frac{1 - P(Y)}{1 - \text{conf}(X \Rightarrow Y)}.
$$

Interpretations:
- Independence: $P(\neg Y \mid X) = P(\neg Y)$, so conv = 1.
- Perfect rule ($\text{conf} = 1$): denominator → 0, conv → $\infty$.
- Negative association ($P(\neg Y \mid X) > P(\neg Y)$): conv < 1.

This is asymmetric, bounded below by 0, and unbounded above — properties that make it well-suited to ranking directed, high-confidence rules. $\blacksquare$

**M5. What is the computational complexity of Apriori? What dominates?**

*Answer.* Let $k_{\max}$ be the size of the longest frequent itemset. Apriori makes $k_{\max} + 1$ passes over the database (including the final pass that finds $L_{k_{\max}+1} = \varnothing$). Per pass $k$:
- Candidate generation: naïvely $O(|L_{k-1}|^2)$ for the join, $O(|C_k| \cdot k)$ for pruning.
- Support counting: each transaction of length $\ell$ and each candidate is checked; organized by a hash tree, the effective cost is $O(|\mathcal{D}| \cdot |C_k| \cdot \text{avg hit rate})$ — but the hash tree prunes most candidate-transaction pairs.

Overall: $O\!\left(\sum_{k=1}^{k_{\max}} |C_k| \cdot |\mathcal{D}| \right)$.

The cost is dominated by:
1. $|C_2|$ — often the largest candidate set, since 2-itemsets are quadratic in $|L_1|$.
2. Database scans — each pass costs at least $O(|\mathcal{D}|)$ I/O.

Consequently, Apriori is I/O-bound on large, disk-resident datasets and CPU-bound on dense, in-memory ones.

**M6. Give the support value of an itemset found from a conditional FP-tree. Justify.**

*Answer.* When mining conditional FP-tree $\mathcal{T}_\alpha$ (with suffix $\alpha$), each node $v$ carries a count $c_v$ that represents the number of original transactions containing both $\alpha$ and the path from root to $v$. For an itemset $\beta$ formed by selecting items along a path in $\mathcal{T}_\alpha$, its support as an itemset in the original database is

$$
\text{supp}(\beta \cup \alpha) = \min_{v \in \text{path}(\beta)} c_v
$$

— the minimum count along that path, because the count of a deeper node is always ≤ counts of its ancestors. For itemsets formed from multiple branches (non-single-path cases), we recurse into further conditional trees; each recursion's conditional pattern base correctly aggregates the counts. $\blacksquare$

**M7. Prove that lift is symmetric while conviction is not.**

*Answer.*
- Lift: $\text{lift}(X \Rightarrow Y) = P(X,Y)/(P(X)P(Y))$. Swapping $X$ and $Y$ gives the same expression (joint and marginals are symmetric in their arguments). Hence symmetric.
- Conviction: $\text{conv}(X \Rightarrow Y) = P(X)P(\neg Y) / P(X, \neg Y)$. Swapping: $\text{conv}(Y \Rightarrow X) = P(Y)P(\neg X) / P(Y, \neg X)$. These are not equal in general, because the numerator depends on $P(\neg Y)$ vs $P(\neg X)$ and the "failure" event changes.

Concrete counterexample: $P(X) = 0.5, P(Y) = 0.9, P(X,Y) = 0.45$. Then $P(X, \neg Y) = 0.05$, $\text{conv}(X\Rightarrow Y) = 0.5 \cdot 0.1 / 0.05 = 1.0$ (independence). And $P(\neg X, Y) = 0.45$, so $P(Y, \neg X) = 0.45$, giving $\text{conv}(Y \Rightarrow X) = 0.9 \cdot 0.5 / 0.45 = 1.0$ — equal here, but choose $P(X,Y) = 0.48$: $\text{conv}(X\Rightarrow Y) = 0.5 \cdot 0.1 / 0.02 = 2.5$; $P(Y, \neg X) = 0.42$, $\text{conv}(Y\Rightarrow X) = 0.9 \cdot 0.5 / 0.02 = 22.5$. Different. $\blacksquare$

---

### Applied Questions

**A1. You're building a cross-sell recommender for an online grocery retailer. Would you use Apriori, FP-Growth, or something else? Walk through your decision.**

*Answer.* Decision factors:
- **Data size**: millions of transactions, tens of thousands of SKUs. Apriori would make many passes and produce huge $C_2$. FP-Growth needs two DB passes and handles large data better. Pick FP-Growth (Spark MLlib / equivalent).
- **Pattern length**: grocery baskets are often moderate length (5-30 items). Both algorithms can handle, but dense baskets favor FP-Growth.
- **Support threshold**: for a recommender we want long-tail patterns, so $\sigma$ will be low (perhaps 0.001–0.005). Low support penalizes Apriori much more than FP-Growth.
- **Refresh cadence**: daily/weekly refresh is typical. FP-Growth's two-pass design makes this cheap; Apriori's multi-pass design may not fit a nightly job.
- **Infrastructure**: if the team already uses Spark, `spark.ml.fpm.FPGrowth` is off-the-shelf.

For the final system I'd probably layer:
1. FP-Growth to mine frequent itemsets and rules.
2. A post-processing step: keep only closed itemsets, apply a lift and improvement threshold.
3. Candidate generation: for a given basket $B$, retrieve rules whose antecedent is a subset of $B$.
4. A learned ranker (gradient boosting on features like basket contents, user history, price, margin, stock) that re-scores candidates.
5. Business-rule filters (in-stock, age-restricted, already-in-cart).
6. An A/B test framework to measure actual uplift in sales, not just statistical lift.

Pure association rules would be a baseline or candidate generator, not the final ranker.

**A2. You mined 100,000 rules with support ≥ 1% and confidence ≥ 70%. The business asks "which are interesting?" How do you answer?**

*Answer.* "Interesting" has no single definition. I'd present a multi-metric view:

1. **Start with lift** to remove rules that are high-confidence only because the consequent is common. Keep lift ≥ some threshold (1.3 is typical).
2. **Apply a conviction filter** (say ≥ 1.5) to highlight directional, high-information rules.
3. **Apply an improvement threshold** (0.02) to drop rules that are just subsets of stronger rules.
4. **Restrict to closed itemsets** to eliminate redundancy.
5. **Apply domain filters**: drop rules involving plastic bags, loyalty markers, universal items.
6. **Significance test**: for each remaining rule, compute a $\chi^2$ statistic on the 2×2 contingency table; drop rules with p > 0.01 after Benjamini-Hochberg correction.
7. **Group by category**: present the top 10 rules per product category, rather than a global top-100 dominated by one or two popular categories.
8. **Deliver alongside actionability metadata**: for each rule, show support, lift, conviction, a narrative interpretation, and a recommended action.

Then discuss with stakeholders what "interesting" means to them — e.g., for promotion design they want high leverage (absolute effect size); for cross-selling they want high lift on high-margin items.

**A3. Your rule "diapers ⇒ beer" has lift 3.0, confidence 0.4, support 0.05. The product team deploys a promotion moving beer next to diapers. Sales don't change. What went wrong?**

*Answer.* Most likely explanations, ranked by plausibility:

1. **Confounding by a third variable.** Customers who buy diapers+beer are a specific segment (parents, typically). Moving products doesn't create new parents or change when they shop. The correlation was real but caused by a joint lifestyle factor, not by any proximity-induced purchase behavior.
2. **Baseline effect already saturated.** Customers who want both already know where to find them. The change doesn't make new customers buy beer.
3. **Substitution.** Customers who would have bought beer in another aisle simply buy it in the new location. Same total sales, different shelf attribution.
4. **Measurement error.** The evaluation window was too short, or seasonality masked the effect.
5. **Misinterpretation of lift.** Lift 3.0 means co-occurrence is 3× independence, but the absolute incremental value is $\text{leverage} = \text{supp}(X,Y) - \text{supp}(X)\text{supp}(Y) = 0.05 - (0.05/0.4)(\text{supp}(Y))$. If beer's marginal support is high, leverage is small. Intervention ROI scales with leverage, not lift.
6. **The rule was overfit to noise.** Confidence 0.4 with support 0.05 in a ~100,000-transaction study: the 2×2 table has ~5,000 co-occurrences. OK sample size, but lift 3 is not a huge effect. Statistical significance is borderline.

Takeaway: association is not causation, and ranking rules by lift does not rank them by business impact.

**A4. How would you handle the cold-start problem (new items) when using rule-based recommendations?**

*Answer.* Pure rule mining can't help new items by definition — they have no transaction history. Strategies:

1. **Category-level backoff**: mine rules at the product category level. A new brand of coffee inherits the association "coffee ⇒ creamer" from its category.
2. **Attribute-based rules**: instead of items, mine rules over (attribute, value) pairs: brand, size, flavor, price range. New items get rule coverage via matching attributes.
3. **Content-based bootstrap**: recommend the new item to customers who bought items similar in content (text embedding, image similarity).
4. **Exploration**: deliberately surface new items to a small fraction of traffic (ε-greedy or Thompson sampling) to gather data quickly; migrate to rule-based once support is sufficient.
5. **Hybrid**: combine rule-mined candidates (existing items) with a content-based candidate set for new items, merged by a learned ranker.
6. **Lookalike items**: identify an existing item most similar to the new one (by attributes or early browsing behavior) and inherit its rules temporarily.

**A5. You need to run association rule mining daily on a growing transaction database. How do you architect it?**

*Answer.*
- **Store raw transactions in a columnar store** (Parquet on S3 / HDFS), partitioned by date.
- **Feature extraction layer**: each night, normalize items (SKU → category), apply filters (universal items, banned lists).
- **Mining job**: Spark FP-Growth on a rolling window (e.g., last 30 days). Configure $\sigma$ and $\gamma$ based on business needs.
- **Post-processing**: keep only closed frequent itemsets, compute all metrics (lift, leverage, conviction, Jaccard), apply improvement threshold.
- **Significance filter**: compute $\chi^2$ or Fisher p-values with a multiple-testing correction.
- **Output store**: persist rules in a low-latency key-value store (Redis, DynamoDB), indexed by antecedent prefix for real-time lookup.
- **Serving**: a microservice takes a basket, finds all rules whose antecedent $\subseteq$ basket, returns top-$k$ consequents by lift.
- **Monitoring**:
  - Metric: number of rules, lift distribution, daily rule churn.
  - Alert: drastic changes in top-100 rule lifts.
  - Business metric: cross-sell attach rate, uplift from A/B tests.
- **Retraining cadence**: daily for high-velocity retail; weekly or monthly for slower domains.
- **Incremental mining option**: if full re-mining is too expensive, use a partitioned/incremental algorithm (e.g., SPO, Borgelt's apriori with delta).

**A6. Compare rule mining to modern recommender systems. Does rule mining still have a role?**

*Answer.* Yes, in several capacities:
- **Candidate generator** for a learned ranker: rules produce a few dozen candidates from the itemset space much faster than a full scoring pass.
- **Cold-start**: category-level rules handle new users/items where collaborative filtering fails.
- **Interpretable explanation layer**: "customers who bought X also bought Y" is a comprehensible rationale, useful for customer trust, regulatory compliance (e.g., in finance or healthcare recommendations), and debugging.
- **Fallback path**: when the learned system is unavailable, rules provide a safe default.
- **Business rule encoding**: some rules are just business constraints ("promote our own brand whenever possible") that a learned model shouldn't have to rediscover.
- **Non-personalized contexts**: aggregate "what goes with what" for store layouts, catalog organization, printed coupon booklets.

In absolute accuracy, a well-tuned neural recommender or matrix factorization will usually beat rules for personalized ranking. But accuracy isn't the only axis: transparency, latency, and engineering simplicity often favor rules.

---

### Debugging & Failure-Mode Questions

**D1. Your Apriori implementation returns far fewer rules than expected. What do you check?**

*Answer.*
- **Support threshold too high**: halve $\sigma$ and see if the count grows.
- **Confidence threshold too high**: same.
- **Item encoding issue**: are "Milk 2L" and "Milk 2 L" treated as distinct? Check item normalization.
- **Filtering too aggressive**: did you drop items that shouldn't have been removed?
- **Wrong data split**: are you mining a tiny subset by accident?
- **Off-by-one in the halt condition**: verify the algorithm doesn't stop at $L_1$.
- **Join step producing fewer candidates than it should**: check the ordering convention; different implementations use different orderings.
- **Pruning logic too aggressive**: verify that the $k$ subsets checked are all $(k-1)$-subsets and that the lookup in $L_{k-1}$ is correct (e.g., consistent hashing, consistent item order).
- **Memory truncation**: some libraries silently cap output size.

**D2. FP-Growth runs for hours and never finishes. What's wrong?**

*Answer.*
- **Support threshold too low**: the FP-tree is huge or conditional trees recurse deeply. Try a higher $\sigma$ first.
- **Data is dense**: many items per transaction means little prefix sharing. FP-Growth degenerates. Try Eclat or LCM, or increase $\sigma$.
- **Item ordering not by frequency**: if the implementation lets you control it, verify frequency-descending order.
- **Memory pressure causing swap**: the FP-tree and conditional trees may have fit in RAM in testing but not in production. Use a partitioned/parallel variant (Spark's PFP).
- **A few ultra-frequent items bloating conditional trees**: drop or split them.
- **Recursion depth on single-path optimization**: some implementations recurse instead of using the closed-form; patch or use a library that does.
- **Unbounded rule length**: cap max itemset length to a reasonable number (say 10).

**D3. You trained a rule-based recommender, and it recommends the same popular items to everyone. Why?**

*Answer.*
- **Confidence-only ranking**: popular items have high confidence by default. Switch to lift or an improvement-filtered metric.
- **Support threshold too high**: only the most popular items survive mining, so only popular-item rules exist.
- **No personalization layer**: rules are basket-level, not user-level. Combine with user embeddings or collaborative filtering for personalization.
- **Lack of item diversity constraints**: add a diversity penalty (e.g., MMR) when selecting top-$k$.
- **No segmentation**: mine rules per customer segment; popular items may dominate only in aggregate.

**D4. Your mined rules change dramatically from week to week. Is that a bug or a feature?**

*Answer.* It could be either; diagnose:
- **Real non-stationarity**: seasonal events (holidays, weather), promotions, stock-outs cause genuine shifts in basket composition. Rules should reflect this.
- **Sampling noise at low support**: if $\sigma$ is near the statistical-reliability threshold, small changes in transaction counts flip rules in and out. Fix: raise $\sigma$ or use a longer mining window.
- **Discontinuity at threshold**: rules with supports near the cutoff oscillate. Fix: use a soft threshold or report a margin band.
- **Data pipeline bugs**: intermittent ingestion failures, duplicated orders, changed SKU mappings. Check pipeline health.
- **Currency/promotional effects**: a week of heavy promotion on one product skews all rules involving it.

Mitigations:
- Exponentially weighted rule statistics (EWMA) over daily minings.
- Tracking rule churn as a KPI; alert when churn exceeds expected range.
- Version rule outputs so you can always diff.

**D5. Lift is infinite for a rule. How?**

*Answer.* Lift $= P(X,Y)/(P(X)P(Y))$. For lift to be infinite with $X, Y$ both observed, we'd need $P(X)P(Y) = 0$ while $P(X,Y) > 0$ — impossible.

More likely:
- Implementation bug: dividing a positive numerator by zero, possibly because filtering removed transactions used in marginal computation but not in joint computation.
- Smoothing issue: Laplace smoothing was applied inconsistently between numerator and denominator.
- Confidence is 1 and someone mis-coded lift as confidence / something-that-can-be-zero.

For *conviction* returning infinity, the situation is normal: $\text{conf} = 1$ means the rule never fails, which is legitimately captured as $\infty$. Handle it explicitly in the output.

**D6. You observe a rule $A \Rightarrow B$ with lift 15, but $B \Rightarrow A$ has lift... also 15. Why?**

*Answer.* Because lift is symmetric. $\text{lift}(A \Rightarrow B) = P(A,B)/(P(A)P(B)) = \text{lift}(B \Rightarrow A)$. This is a feature of lift, not a bug. If you want directional behavior, use confidence, conviction, or a metric like Laplace-corrected confidence.

---

### Follow-up & Probing Questions

**P1. Interviewer: "Why doesn't Apriori just enumerate all $2^m$ itemsets?"**

*Answer.* It effectively doesn't even come close in practice because the Apriori property prunes the search tree aggressively. But formally, worst case, if *every* itemset is frequent, Apriori does enumerate the full lattice and is no better than brute force. The point is that on real data with skewed distributions, the pruning eliminates the vast majority of itemsets without counting them.

A deeper follow-up: *when* is the worst case hit? When data is very dense (every transaction has most items). In such regimes, both Apriori and FP-Growth struggle; specialized miners (closed/maximal only, sampling-based approximations, or dimensionality reduction) are needed.

**P2. Interviewer: "You say rule mining is 'unsupervised.' How would you evaluate its output?"**

*Answer.* Several levels:

1. **Statistical quality**: internal metrics (support, lift, conviction, $\chi^2$). Purely descriptive.
2. **Coverage**: fraction of transactions explained by the top-$k$ rules; distribution of rule supports.
3. **Predictive test**: hold out the last time slice, mine on the earlier slice, evaluate whether mined rules "predict" held-out co-occurrences (hit rate, precision@k).
4. **Downstream task**: use rules as features in a supervised model; compare AUC/MAP with and without.
5. **Business A/B test**: deploy recommendations driven by rules; measure click-through, attach rate, revenue uplift, margin. This is the gold standard.
6. **Human review**: domain experts assess whether the top rules are plausible and actionable.

The internal metrics are cheap and fast; business A/B tests are expensive but the only way to measure causal impact.

**P3. Interviewer: "Can you give me a reason a rule with extremely high lift might still be useless?"**

*Answer.* Many reasons:

- **Tiny support**: lift 100 from 3 co-occurrences in a million transactions is statistical noise.
- **Trivial rule**: {digital camera} ⇒ {memory card} has huge lift but provides no business insight — it's already known.
- **Tautological rule**: {pregnancy test, positive} ⇒ {prenatal vitamin interest}. Strong but derivable from domain logic.
- **Non-actionable rule**: {Monday morning coffee} ⇒ {afternoon coffee}. The antecedent and consequent are in different transactions or time-shifted; rule mining treats them as co-occurring but that's not actionable in a single-basket setting.
- **Spurious pattern from a data artifact**: rules involving coupons, markers, or internal codes that shouldn't have been exposed.
- **Leakage**: the antecedent and consequent are really different encodings of the same event.
- **Segment-specific pattern treated as universal**: rule may hold for one customer segment and reverse for another; the aggregate doesn't describe any individual.

**P4. Interviewer: "Scale estimate: one million transactions, 10,000 items, average basket size 20. How many candidate 2-itemsets does Apriori generate at support 1%?"**

*Answer.* At support 1%, item $i$ is frequent if it appears in ≥ 10,000 transactions. Say roughly 500 items are frequent (a common order of magnitude for retail).

Then $|C_2| = \binom{500}{2} = 124{,}750$. Apriori will count each against each of the million transactions' 2-subsets ($\binom{20}{2} = 190$ pairs per transaction, on average). Using a hash tree, only a few thousand of the 124,750 candidates are checked per transaction. Total cost: roughly $10^6 \cdot \text{few thousand} = $ a few billion operations — manageable on modern hardware but not trivial.

At support 0.1%: ~2,000 frequent items, $|C_2| = 2M$. At this point Apriori begins to struggle; FP-Growth's memory-resident tree is a much better fit.

**P5. Interviewer: "Your boss says the rule set 'feels boring' — mostly pairs they already knew. How do you surface more interesting rules?"**

*Answer.*
- **Switch metric**: rank by conviction or leverage, which reward surprising, high-absolute-effect rules, rather than by confidence or pure lift.
- **Apply an improvement threshold**: rules that add nothing over subset rules go away.
- **Subtract a baseline**: compare current-period rules to a previous-period baseline; present rules whose lift changed most (emerging or fading associations).
- **Segment and compare**: a rule that's 2× stronger in segment A than segment B is more interesting than either in isolation.
- **Long-tail mode**: lower $\sigma$, then filter by significance; surface rules involving rarer items.
- **Anti-rules**: rules with lift < 1 tell you what *doesn't* go together — often surprising.
- **Drop trivial categories**: pre-filter rules where antecedent and consequent are in the same obvious family.

**P6. Interviewer: "If I handed you a database with 10 billion transactions and you had to mine rules in 1 hour, what would you do?"**

*Answer.*
- **Sample**: random 1% of transactions is 100M, still big but plausible. Lift and leverage estimates from a large uniform sample are accurate.
- **Distributed FP-Growth (PFP)**: partition items, parallelize mining across many workers. Spark MLlib `FPGrowth` is designed for this.
- **Approximate counting**: Count-Min sketch for item and pair supports; accept some error for huge speedup.
- **Tier-based mining**: first pass at high $\sigma$ (very cheap); second pass on interesting subcategories at lower $\sigma$.
- **Limit rule length**: cap at 3 or 4. Most business-useful rules are short.
- **Pre-filter ruthlessly**: remove universal items, ultra-rare items, banned categories *before* mining.
- **Sketch-based lift estimation**: compute pair-level lift directly via Count-Min or MinHash rather than through full itemset mining, if only pairwise rules are needed.

Trade-off: speed vs. exhaustiveness. For most business applications, an approximate but fast result beats an exact but slow one.

**P7. Interviewer: "Explain the connection between association rule mining and mutual information."**

*Answer.* Pointwise mutual information of two events:

$$
\text{PMI}(X; Y) = \log \frac{P(X,Y)}{P(X) P(Y)} = \log \text{lift}(X,Y).
$$

So lift is the exponentiated PMI. Ranking rules by lift is equivalent to ranking by PMI. Mutual information as a full scalar (integrating PMI over the joint distribution) isn't directly used in basic rule mining, but:
- Feature-selection methods that pick top-$k$ features by MI are closely related.
- Conditional mutual information captures three-way interactions; corresponds to evaluating whether $P(Y \mid X, Z)$ differs from $P(Y \mid X)$, which is the principle behind "improvement" in rule mining.
- The information-theoretic interpretation explains why lift is scale-free and symmetric: it's a probability ratio, not an additive quantity.

**P8. Interviewer: "Are there Bayesian approaches to association rule mining?"**

*Answer.* Yes. Classical rule mining uses empirical point estimates with hard thresholds; Bayesian approaches replace these with posterior distributions:

- **Shrinkage estimates for support/confidence**: put a Beta prior on $P(Y \mid X)$ and use the posterior mean. Very helpful for low-support rules.
- **Bayesian sets** (Ghahramani & Heller): rank items by their posterior probability of belonging to a "cluster" defined by a query itemset — a principled alternative to lift.
- **Hierarchical Bayesian models** over item categories: borrow strength across related items to estimate supports more robustly in sparse data.
- **Bayesian networks**: go beyond pairwise rules to model full joint distributions; rule mining can be viewed as an extremely restricted special case.
- **Interestingness as posterior surprise**: a rule is "interesting" if observed joint frequency differs from prior expectation by many standard deviations of the posterior.

The main benefit: calibrated uncertainty. The main drawback: computational cost and model-specification effort. In practice most teams use frequentist rule mining with significance tests as a pragmatic compromise.

**P9. Interviewer: "What would you use to estimate the statistical significance of a mined rule?"**

*Answer.* Build the 2×2 contingency table for $X$ and $Y$:
- $a = \text{count}(X,Y)$
- $b = \text{count}(X, \neg Y)$
- $c = \text{count}(\neg X, Y)$
- $d = \text{count}(\neg X, \neg Y)$

Options:
1. **Pearson's $\chi^2$ test of independence**: for each cell, compute expected count under $H_0$, sum $(O-E)^2/E$. Valid when expected counts ≥ 5. p-value from $\chi^2_1$.
2. **Fisher's exact test**: for small samples; computes exact p-value from the hypergeometric distribution.
3. **G-test** (likelihood ratio): asymptotically equivalent to $\chi^2$ but based on KL divergence.
4. **Bootstrap / permutation**: resample transactions and recompute lift; report percentile rank of observed lift.

**Multiple testing correction is essential**: if you mined 10,000 rules, naive p < 0.01 thresholds produce ~100 false positives. Use Benjamini-Hochberg (FDR control) or Bonferroni. Webb's "statistically sound rule discovery" framework is the standard reference.

**P10. Interviewer: "Do you see any connection between association rules and modern deep-learning-based recommenders?"**

*Answer.* Deep recommenders (e.g., two-tower models, transformer-based sequence recommenders) learn dense representations where co-occurrence is implicit rather than explicit. Still, multiple bridges exist:

- **Word2vec ≈ PMI factorization** (Levy & Goldberg, 2014). The skip-gram with negative sampling objective implicitly factorizes a shifted PMI matrix. Applied to items in baskets ("item2vec"), this means deep representations are learning lift-like information in disguise.
- **Pretrained item embeddings** often look like dimensionality-reduced lift matrices.
- **Attention in transformers** over basket sequences resembles rule firing: the model "attends" to prior items that best predict the next — a learned, contextual version of $X \Rightarrow Y$.
- **Explanations for neural recommenders** often back-solve for the nearest rule that would explain the model's output ("we recommended Y because you bought X"). This is association-rule reasoning as a post-hoc interpretability layer.
- **Cold-start and explainability** remain strengths of explicit rules; neural systems struggle on both.

In practice, modern systems are hybrids: neural models for ranking, rule-like structures for candidate generation, explanations, and fallbacks.

---

## Appendix A: Quick-Reference Formula Card

| Quantity | Formula | Range |
|---|---|---|
| Support | $\text{supp}(Z) = P(Z \subseteq T)$ | [0, 1] |
| Confidence | $\text{conf}(X \Rightarrow Y) = P(Y\|X)$ | [0, 1] |
| Lift | $\frac{P(X,Y)}{P(X)P(Y)}$ | [0, ∞) |
| Leverage | $P(X,Y) - P(X)P(Y)$ | [−0.25, 0.25] |
| Conviction | $\frac{P(X)P(\neg Y)}{P(X,\neg Y)} = \frac{1-P(Y)}{1-\text{conf}}$ | [0, ∞] |
| Jaccard | $\frac{P(X,Y)}{P(X)+P(Y)-P(X,Y)}$ | [0, 1] |
| Cosine | $\frac{P(X,Y)}{\sqrt{P(X)P(Y)}}$ | [0, 1] |
| All-confidence | $\frac{P(X,Y)}{\max(P(X), P(Y))}$ | [0, 1] |
| Kulczynski | $\frac{1}{2}(\text{conf}(X\Rightarrow Y) + \text{conf}(Y\Rightarrow X))$ | [0, 1] |
| Imbalance ratio | $\frac{\|P(X)-P(Y)\|}{P(X)+P(Y)-P(X,Y)}$ | [0, 1] |
| φ-coefficient | $\frac{P(X,Y)-P(X)P(Y)}{\sqrt{P(X)P(\neg X)P(Y)P(\neg Y)}}$ | [−1, 1] |
| Collective strength | $\frac{1-v}{1-E[v]}\cdot\frac{E[v]}{v}$ | [0, ∞] |
| PMI | $\log\text{lift}$ | (−∞, ∞) |

## Appendix B: Key References

- Agrawal, Imieliński, Swami (1993). *Mining association rules between sets of items in large databases*. SIGMOD. — Original formulation.
- Agrawal, Srikant (1994). *Fast algorithms for mining association rules*. VLDB. — Apriori.
- Han, Pei, Yin (2000). *Mining frequent patterns without candidate generation*. SIGMOD. — FP-Growth.
- Piatetsky-Shapiro (1991). *Discovery, analysis, and presentation of strong rules*. — Leverage.
- Brin, Motwani, Ullman, Tsur (1997). *Dynamic itemset counting and implication rules for market basket data*. SIGMOD. — Conviction, DIC.
- Aggarwal, Yu (1998). *A new framework for itemset generation*. PODS. — Collective strength.
- Tan, Kumar, Srivastava (2002). *Selecting the right interestingness measure for association patterns*. KDD. — Comparative study of 21+ metrics.
- Webb (2007). *Discovering significant patterns*. Machine Learning. — Statistically sound rule discovery; improvement.
- Pasquier, Bastide, Taouil, Lakhal (1999). *Discovering frequent closed itemsets for association rules*. ICDT. — Closed itemsets.
