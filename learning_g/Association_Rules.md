# Association Rules & Frequent Pattern Mining

## 1. Motivation & Intuition

### Why does this topic exist?
Imagine you manage a large supermarket. You have millions of receipts (transactions) from customers, but you don't know *how* they shop. Do people who buy cereal also buy milk? Do people who buy spicy salsa also buy tortilla chips?

Association Rule Mining exists to uncover these hidden relationships within large datasets. It is a rule-based machine learning method for discovering interesting relations between variables in large databases.

### Intuitive Example: "The Beer and Diapers" Legend
A classic example in data mining is the correlation between beer and diapers. A retailer analyzing checkout data noticed that on Friday afternoons, young men who bought diapers were also likely to buy beer.

* **The Rule:** $\{Diapers\} \to \{Beer\}$
* **The Action:** The store placed beer next to diapers.
* **The Result:** Sales for both increased.

### Connection to Real-World Systems
While simple, this logic underpins massive modern machine learning architectures:
1. **Recommender Systems:** "Customers who bought this item also bought..." engines.
2. **Cross-Selling Strategies:** Automated banking triggers offering a credit card to a user who just opened a high-yield savings account.
3. **Medical Diagnosis:** Probabilistic mapping of co-occurring symptoms to underlying pathogens.
4. **Web Usage Mining:** Predicting a user's $t+1$ clickstream state based on their current session trajectory.

---

## 2. Conceptual Foundations

### Key Terminology

1. **Item ($I$):** A single object or attribute (e.g., Milk, Bread).
2. **Itemset ($X$):** A collection of one or more items. 
   * **$k$-itemset:** An itemset containing strictly $k$ items.
3. **Transaction ($T$):** A discrete record containing a set of items (e.g., a single customer receipt).
4. **Database ($D$):** A collection of all recorded transactions.

### The Association Rule
An association rule is an implication expression of the form:

$$
X \to Y
$$

Where:
* $X$ is the **Antecedent** (the "if" condition).
* $Y$ is the **Consequent** (the "then" result).
* $X$ and $Y$ are mutually disjoint itemsets ($X \cap Y = \emptyset$).

### The Three Pillars of Measurement
To separate signal from noise, we cannot exhaustively evaluate every possible permutation. Instead, candidate rules are filtered through three primary statistical lenses:

1. **Support:** How empirically popular is an itemset across the entire database?
2. **Confidence:** How reliably does the consequent $Y$ appear when the antecedent $X$ is present?
3. **Lift:** Does the occurrence of $X$ actively change the probability of observing $Y$ compared to pure random chance?

### Underlying Assumptions
* **Binary Presence:** Standard formulations assume items exist strictly in a boolean state $\{0, 1\}$. Purchase volume (e.g., buying 12 cartons of milk vs. 1 carton) is flattened to $1$.
* **Static Database:** Traditional frequent pattern algorithms assume the underlying database $D$ remains frozen during the mining pass.
* **Homogeneous Value:** Standard algorithms treat all items as having equal utility; a $\$0.10$ plastic bag is weighted identically to a $\$2,000$ television.

### What breaks when assumptions are violated?
* **High-Cardinality Skew:** If continuous transaction data is forced into categorical bins without proper normalization, "rare but high-value" item combinations fall below global minimum support thresholds and are erased.
* **Temporal Drift:** In real-time streaming environments, an itemset that is mathematically infrequent over a 12-month window may represent an emergent, hyper-frequent viral trend over a 4-hour window.

---

## 3. Mathematical Formulation

Let $N$ be the total number of discrete transactions in database $D$.  
Let $\sigma(X)$ represent the absolute frequency count (number of transactions) containing itemset $X$.

### 1. Support
Support measures the prior probability of an itemset occurring within the database.

$$
\text{Support}(X) = P(X) = \frac{\sigma(X)}{N}
$$

For a directional rule $X \to Y$, the support is the joint probability of both itemsets appearing in the exact same transaction:

$$
\text{Support}(X \to Y) = P(X \cup Y) = \frac{\sigma(X \cup Y)}{N}
$$

* **Intuition:** Filters out statistical flukes. If a rule happens twice in 10,000,000 transactions, its operational utility is near zero regardless of how high its confidence is.

### 2. Confidence
Confidence measures the conditional probability of finding the consequent $Y$ given that the transaction already contains the antecedent $X$.

$$
\text{Confidence}(X \to Y) = P(Y | X) = \frac{P(X \cup Y)}{P(X)} = \frac{\sigma(X \cup Y)}{\sigma(X)}
$$

* **Intuition:** "Out of all the receipts containing item $X$, what percentage of them also listed item $Y$?"

### 3. Lift
Lift measures the ratio of the observed joint probability of $X$ and $Y$ to the expected joint probability if $X$ and $Y$ were completely independent occurrences.

$$
\text{Lift}(X \to Y) = \frac{P(Y | X)}{P(Y)} = \frac{P(X \cup Y)}{P(X)P(Y)}
$$

**Interpretation of Lift values:**
* **$\text{Lift} = 1$:** $X$ and $Y$ are independent. The occurrence of $X$ provides no information about $Y$.
* **$\text{Lift} > 1$:** $X$ and $Y$ are positively correlated. The presence of $X$ increases the likelihood of observing $Y$.
* **$\text{Lift} < 1$:** $X$ and $Y$ are negatively correlated (substitutes). The presence of $X$ actively suppresses the likelihood of observing $Y$.

---

## 4. Algorithms: Apriori & FP-Growth

Mining association rules is fundamentally divided into two phases:
1. **Frequent Itemset Generation:** Locate all subsets $X \subset I$ where $\text{Support}(X) \ge \text{min\_support}$.
2. **Rule Generation:** Extract all valid permutations $X \to Y$ from those frequent itemsets where $\text{Confidence}(X \to Y) \ge \text{min\_confidence}$.

Phase 1 represents an exponential search space of size $2^d - 1$. 

### A. The Apriori Algorithm

**Core Principle:** The *Downward Closure Property* (Apriori Principle).
> If an itemset is frequent, then all of its non-empty subsets must also be frequent.

Conversely, if an itemset is deemed infrequent, any superset generated by adding items to it is guaranteed to be mathematically infrequent. This allows the algorithm to aggressively prune the combinatorial tree.

**Algorithm Execution Steps:**
1. Scan $D$ to derive frequency counts for all single items ($1$-itemsets). Discard those below `min_support`.
2. **Join Step:** Combine surviving $k$-itemsets with each other to generate candidate $(k+1)$-itemsets.
3. **Prune Step:** Inspect every generated candidate $(k+1)$-itemset; if any of its $k$-subsets were previously discarded, delete the candidate immediately.
4. Scan $D$ again to compute the actual empirical support for the remaining candidates.
5. Increment $k$ and repeat steps 2–4 until no new frequent itemsets can be generated.

### B. FP-Growth (Frequent Pattern Growth)

**Core Principle:** Apriori suffers from severe I/O bottlenecks because it requires $k$ full database scans. FP-Growth collapses this to strictly **two scans** by compressing the transaction history into an in-memory prefix tree called a **Frequent Pattern Tree (FP-Tree)**.

**Algorithm Execution Steps:**
1. **First Scan:** Calculate single-item support counts. Discard infrequent items. Sort surviving items in descending order of frequency ($L$-order).
2. **Second Scan (Tree Construction):** Read transactions one-by-one. Filter and sort each transaction according to $L$-order. Insert the sequence into the FP-Tree; shared prefixes share memory nodes, while divergence creates new branches. Increment node hit-counters.
3. **Mining the Tree:** For each item in the tree (starting from the least frequent):
   * Construct its **Conditional Pattern Base** (the set of prefix paths leading directly to that item's nodes).
   * Construct a **Conditional FP-Tree** out of those prefix paths.
   * Recursively mine the conditional tree to extract frequent patterns without candidate generation.

---

## 5. Worked Example

**Database ($D$) containing $N=5$ transactions:**
* $T_1$: {Milk, Bread, Eggs}
* $T_2$: {Milk, Bread}
* $T_3$: {Milk, Diapers, Beer, Eggs}
* $T_4$: {Bread, Diapers, Beer}
* $T_5$: {Bread, Milk, Diapers, Beer}

**System Thresholds:** * $\text{min\_support} = 0.60$ (Must appear in $\ge 3$ transactions)
* $\text{min\_confidence} = 0.75$

### Step 1: Frequent Itemset Generation (Apriori approach)

**Scan 1: Evaluate $1$-itemsets**
* Milk: $4/5 = 0.80$ *(Pass)*
* Bread: $4/5 = 0.80$ *(Pass)*
* Diapers: $3/5 = 0.60$ *(Pass)*
* Beer: $3/5 = 0.60$ *(Pass)*
* Eggs: $2/5 = 0.40$ *(Fail — Prune Eggs from all future steps)*

**Scan 2: Generate candidate $2$-itemsets from surviving $1$-itemsets**
* {Milk, Bread}: Present in $T_1, T_2, T_5 \to \sigma = 3 \to \text{Support} = 0.60$ *(Pass)*
* {Milk, Diapers}: Present in $T_3, T_5 \to \sigma = 2 \to \text{Support} = 0.40$ *(Fail)*
* {Milk, Beer}: Present in $T_3, T_5 \to \sigma = 2 \to \text{Support} = 0.40$ *(Fail)*
* {Bread, Diapers}: Present in $T_4, T_5 \to \sigma = 2 \to \text{Support} = 0.40$ *(Fail)*
* {Bread, Beer}: Present in $T_4, T_5 \to \sigma = 2 \to \text{Support} = 0.40$ *(Fail)*
* {Diapers, Beer}: Present in $T_3, T_4, T_5 \to \sigma = 3 \to \text{Support} = 0.60$ *(Pass)*

*Surviving Frequent Itemsets:* `{Milk, Bread}` and `{Diapers, Beer}`.

### Step 2: Rule Generation

Evaluating candidate directional rules generated from the frequent itemset **`{Diapers, Beer}`**:

* **Candidate A:** $\text{Diapers} \to \text{Beer}$
  $$\text{Confidence} = \frac{\sigma(\text{Diapers} \cup \text{Beer})}{\sigma(\text{Diapers})} = \frac{3}{3} = 1.00 \quad (100\%)$$
  $1.00 \ge 0.75 \implies \mathbf{Valid\ Rule.}$

* **Candidate B:** $\text{Beer} \to \text{Diapers}$
  $$\text{Confidence} = \frac{\sigma(\text{Diapers} \cup \text{Beer})}{\sigma(\text{Beer})} = \frac{3}{3} = 1.00 \quad (100\%)$$
  $1.00 \ge 0.75 \implies \mathbf{Valid\ Rule.}$

**Evaluating the statistical significance (Lift) of Candidate A:**

$$
\text{Lift}(\text{Diapers} \to \text{Beer}) = \frac{\text{Confidence}(\text{Diapers} \to \text{Beer})}{P(\text{Beer})} = \frac{1.00}{(3/5)} = \frac{1.00}{0.60} \approx \mathbf{1.67}
$$

Because $1.67 > 1.0$, the relationship represents a true positive association rather than random co-occurrence.

---

## 6. Advanced Evaluation Metrics

Relying strictly on Confidence creates systemic blind spots when analyzing heavily skewed datasets. If a consequent $Y$ is overwhelmingly frequent across the entire database, $P(Y|X)$ will yield high confidence scores even if $X$ has zero actual bearing on $Y$.

### 1. Conviction
Measures the directional implication strength of a rule by comparing the expected frequency of $X$ appearing without $Y$ (assuming independence) against the actual observed frequency of incorrect predictions.

$$
\text{Conviction}(X \to Y) = \frac{1 - \text{Support}(Y)}{1 - \text{Confidence}(X \to Y)}
$$

* **Properties:** Unlike Lift, Conviction is directional ($\text{Conviction}(X \to Y) \neq \text{Conviction}(Y \to X)$).
* If $\text{Confidence} = 1.0$, $\text{Conviction} \to \infty$, signifying absolute logical implication.

### 2. Leverage
Measures the absolute difference between the observed joint frequency of $X$ and $Y$ and the theoretical joint frequency expected under pure independence.

$$
\text{Leverage}(X \to Y) = \text{Support}(X \to Y) - \big(\text{Support}(X) \times \text{Support}(Y)\big)
$$

* **Properties:** Bounded within the range $[-0.25, 0.25]$. 
* A Leverage of $0$ indicates perfect independence. Favors itemsets with high baseline support over rare itemsets with high relative lift.

### 3. Collective Strength
Evaluates joint itemsets rather than directional rules. It quantifies the ratio of the probability of joint fulfillment (all items present or all absent) against expected independence fulfillment.

$$
\text{CS}(X) = \frac{1 - v(X)}{1 - E[v(X)]} \times \frac{E[v(X)]}{v(X)}
$$

*(Where $v(X)$ is the violation rate—the proportion of transactions containing some, but not all, items in $X$).*

---

## 7. Relevance to Machine Learning Practice

### Applied Systems Integration
* **Cold-Start Fallback Engines:** When collaborative filtering models encounter a brand-new user with zero historical vector embeddings, rules mined via FP-Growth act as deterministic fallback heuristics.
* **Sparse Feature Engineering:** Frequent itemsets can be extracted from raw interaction logs and converted into explicit boolean features ($x \in \{0,1\}$) fed into downstream Gradient Boosted Decision Trees (e.g., XGBoost) to model interaction terms explicitly.
* **LLM Prompt Optimization:** Mining co-occurrence patterns in high-performing system prompts to build deterministic prompt-templating pipelines.

### Algorithmic Trade-offs

| Dimension | Characteristics |
| :--- | :--- |
| **Interpretability** | **Extreme.** Outputs map to deterministic Boolean logic (`IF A THEN B`), allowing direct auditing by legal, medical, or business compliance teams. |
| **Computational Cost** | **High.** NP-Hard search space complexity. Apriori scales poorly with database length ($N$); FP-Growth scales poorly with unique item vocabulary ($d$). |
| **Sparsity Sensitivity** | **Optimal for Sparse Data.** Thrives on retail checkout vectors (where transaction density $\ll 1\%$). Fails completely on dense matrix data (e.g., continuous image pixels or sensor arrays) due to combinatorial explosion. |

---

## 8. Common Pitfalls & Misconceptions

1. **Equating Lift with Causality:** Association mining captures observational co-occurrence. Discovering $\text{Umbrellas} \to \text{Wet Floor Signs}$ does not mean selling umbrellas causes floors to become wet.
2. **The "Rare Item" Trap:** Setting global minimum support thresholds too high automatically filters out long-tail, high-margin items (e.g., luxury goods). Setting them too low crashes the memory heap during the candidate join phase.
3. **The "Null-Transaction" Skew:** Lift is not null-invariant. Flooding a database with transactions that contain *neither* item $X$ nor item $Y$ artificially inflates the global denominator $N$, depressing $P(X)$ and $P(Y)$ and causing Lift scores to spike wildly.

---
---

# Machine Learning Interview Preparation Section

## Part 1: Foundational Questions

### Q1: Define Support, Confidence, and Lift. What distinct analytical role does each play?
**Answer:**
* **Support** ($P(X \cap Y)$) establishes **statistical significance**. It acts as a high-pass filter to eliminate noise, ensuring computational resources are not wasted analyzing unrepeatable anomalies.
* **Confidence** ($P(Y|X)$) establishes **directional reliability**. It measures the predictive accuracy of the rule's inference statement.
* **Lift** ($\frac{P(X \cap Y)}{P(X)P(Y)}$) establishes **correlation strength**. It normalizes the confidence metric against the baseline popularity of the consequent, preventing false positives triggered by universally popular items.

### Q2: Explain the Downward Closure Property. How does it optimize Apriori execution?
**Answer:**
The Downward Closure Property dictates that any subset of a frequent itemset must also be frequent. Mathematically, if $P(X) \ge \theta$, and $Y \subset X$, then $P(Y) \ge P(X) \ge \theta$.

**Optimization:** It provides a deterministic pruning heuristic. If a $k$-itemset is evaluated and fails the `min_support` threshold, the algorithm can immediately prune all $(k+1)$-itemsets derived from it. This prevents the algorithm from traversing an intractable $2^d$ search tree.

### Q3: Under what system memory and I/O conditions would you choose FP-Growth over Apriori?
**Answer:**
* **Apriori** requires $k$ sequential read passes over the entire database stored on disk (where $k$ is the size of the largest frequent itemset). If the database $D$ is multi-terabyte and disk I/O is the primary latency bottleneck, Apriori degrades severely.
* **FP-Growth** requires strictly two disk reads: one for vocabulary counting, and one to build the FP-Tree. However, the entire FP-Tree and its recursive conditional prefix trees *must reside in RAM*. Therefore, FP-Growth is selected when **RAM capacity exceeds the compressed transaction volume**, while Apriori (or partitioned distributed variants) is preferred in memory-constrained, disk-heavy environments.

---

## Part 2: Mathematical Questions

### Q4: Prove mathematically that $\text{Lift}(X \to Y)$ is symmetric, whereas $\text{Confidence}(X \to Y)$ is asymmetric.
**Answer:**
By definition:

$$
\text{Lift}(X \to Y) = \frac{P(X \cap Y)}{P(X)P(Y)}
$$

$$
\text{Lift}(Y \to X) = \frac{P(Y \cap X)}{P(Y)P(X)}
$$

Because set intersection is commutative ($X \cap Y = Y \cap X$) and scalar multiplication is commutative ($P(X)P(Y) = P(Y)P(X)$), the two expressions are algebraically identical: $\text{Lift}(X \to Y) = \text{Lift}(Y \to X)$.

Conversely, for Confidence:

$$
\text{Confidence}(X \to Y) = \frac{P(X \cap Y)}{P(X)} \quad\text{vs.}\quad \text{Confidence}(Y \to X) = \frac{P(X \cap Y)}{P(Y)}
$$

Because marginal probabilities of distinct itemsets are generally unequal ($P(X) \neq P(Y)$), the denominators diverge, making Confidence asymmetric.

### Q5: What is the exact theoretical worst-case upper bound of candidate itemsets generated by Apriori for a database with unique item vocabulary $d$?
**Answer:**
In the absolute worst-case scenario (where $\text{min\_support} = 0$ or the database is perfectly dense such that every item co-occurs with every other item), the algorithm explores the complete Boolean power set $\mathcal{P}(I)$. 

Excluding the empty set $\emptyset$, the total number of evaluated candidates is:

$$
\sum_{k=1}^{d} \binom{d}{k} = 2^d - 1
$$

### Q6: If $\text{Confidence}(A \to B) = 1.0$, does this guarantee that $A$ and $B$ are positively correlated? Provide a mathematical counter-example.
**Answer:**
**No.** Absolute confidence does not guarantee positive correlation.

**Counter-example:** Suppose an e-commerce database contains $100$ transactions ($N=100$).
* Item $A$ (Specialty Cable) appears in $5$ transactions: $\sigma(A) = 5 \implies P(A) = 0.05$.
* Item $B$ (Free Shipping Promo) is applied to all $100$ transactions: $\sigma(B) = 100 \implies P(B) = 1.0$.
* The joint occurrence $\sigma(A \cup B) = 5$.

$$
\text{Confidence}(A \to B) = \frac{5}{5} = 1.0 \quad (100\%)
$$

$$
\text{Lift}(A \to B) = \frac{P(A \cap B)}{P(A)P(B)} = \frac{0.05}{(0.05 \times 1.0)} = \mathbf{1.0}
$$

A Lift of $1.0$ proves that $A$ and $B$ are completely independent. The high confidence is an artifact of $B$'s universal saturation.

---

## Part 3: Applied & Scenario Questions

### Q7: You run an association mining job on an enterprise retail database and extract 45,000 rules meeting your thresholds. How do you architect an automated curation pipeline to serve only the top 5 rules to end-users?
**Answer:**
1. **Hard Metric Filtering:** Strip all rules where $\text{Lift} \le 1.1$ to eliminate weak correlations.
2. **Consequent Deduplication:** Group rules by their consequent ($Y$). If rules $\{A\} \to \{C\}$, $\{B\} \to \{C\}$, and $\{A, B\} \to \{C\}$ exist, calculate the marginal gain of the compound antecedent. If $\text{Conf}(A \cup B \to C) \approx \text{Conf}(A \to C)$, discard the compound rule via Occam's Razor.
3. **Actionability & Business Value Weighting:** Multiply the statistical score by a domain metric (e.g., Gross Margin). 
   $$\text{Utility Score} = \text{Lift}(X \to Y) \times \text{Margin}(Y)$$
4. **Ranking via Conviction/Leverage:** Sort surviving ties using **Conviction** (to prioritize rules with the lowest historical failure rate) or **Leverage** (to prioritize rules that drive the absolute highest transaction volume).

### Q8: You are tasked with running FP-Growth on a dataset containing 10 billion transactions, but your production spark cluster node has a strict RAM limit of 32GB. How do you resolve this?
**Answer:**
Implement **Parallel FP-Growth (PFP)** via distributed map-reduce partitioning:
1. **Sharding by Vocabulary:** Perform a distributed scan to get $1$-itemset frequencies. Group items into $G$ independent computational groups.
2. **Transaction Projection:** Map each transaction across the cluster. If a transaction contains frequent items belonging to Group 3 and Group 7, project (copy) the relevant subset of that transaction to the specific worker nodes responsible for Group 3 and Group 7.
3. **Independent Local Mining:** Each worker node builds a localized, highly compressed FP-Tree strictly for its assigned vocabulary slice. Because the global vocabulary is partitioned across nodes, no single local FP-Tree breaches the 32GB RAM ceiling.

### Q9: How would you adapt Association Rule Mining algorithms to process continuous numerical customer attributes (e.g., Age, Annual Income, Session Duration)?
**Answer:**
Traditional ARM requires discrete categorical states. Continuous data must be transformed via:
1. **Quantile Discretization:** Bin continuous features into statistically uniform buckets (e.g., `Income: [0-25th percentile] $\to$ Income_Low`). Avoid equal-width binning, which creates artificially sparse outlier buckets.
2. **Fuzzy Association Rules:** Instead of binary mapping $x \in \{0, 1\}$, assign membership degrees $x \in [0, 1]$ using fuzzy set functions (e.g., an individual age 33 might map to `Age_Young: 0.7` and `Age_Middle: 0.3`). Update the classical Support count formula to sum fuzzy membership intersections:
   $$\sigma(X) = \sum_{i=1}^{N} \prod_{x \in X} \mu_x(T_i)$$

---

## Part 4: Debugging & Failure Modes

### Q10: A junior data scientist runs Apriori on a grocery dataset with `min_support=0.001` (0.1%). The job runs out of memory (OOM) and crashes during candidate generation. Explain the exact mechanical failure point.
**Answer:**
The OOM exception occurs during the transition from **$1$-itemsets to $2$-itemsets** (or $2 \to 3$). 

If a supermarket carries $50,000$ unique SKUs, setting `min_support=0.001` allows virtually every item to survive Scan 1. In Step 2 (Join Phase), Apriori attempts to instantiate candidate $2$-itemsets in memory:

$$
\binom{50,000}{2} = \frac{50,000 \times 49,999}{2} \approx \mathbf{1.25\ billion\ candidate\ pairs}
$$

Allocating object references, array wrappers, and hash-map keys for $1.25 \times 10^9$ discrete pairs exhausts the JVM/Python memory heap before the algorithm can even scan the database to prune them.

### Q11: An automated cross-selling pipeline deploys the rule $\{\text{Phone Case}\} \to \{\text{Smartphone}\}$ based on a valid Lift of 4.2. Conversion rates drop to zero. What went wrong structurally?
**Answer:**
The model captured an **asymmetric chronological dependency** while operating on a **flattened basket representation**.

While Phone Cases and Smartphones co-occur frequently on receipts (driving Lift high), the purchasing intent is strictly unidirectional in time: *users buy cases because they bought a phone*. A user browsing phone cases has almost certainly *already* purchased their smartphone. Recommending a $\$1,000$ smartphone to a user buying a $\$15$ accessory violates the chronological sequence of consumer intent.

**Remediation:** Transition from static Association Rule Mining to **Sequential Pattern Mining** (e.g., PrefixSpan algorithm) where transactions are timestamp-ordered chains $T_{t1} \to T_{t2}$.

---

## Part 5: Follow-up Probing

### Q12: Interviewer Probing Question: *"You mentioned earlier that Lift is heavily skewed by null transactions. Explain the concept of 'Null-Invariance' and identify two alternative metrics that solve this."*
**Answer:**
A metric is **Null-Invariant** if adding or removing transactions that contain *neither* antecedent $X$ nor consequent $Y$ leaves the computed score unchanged. 

In massive recommendation systems (e.g., Netflix), $99.99\%$ of the user-item interaction matrix consists of zeros (null transactions). Because $\text{Lift}(X \to Y) = \frac{N \cdot \sigma(X \cup Y)}{\sigma(X)\sigma(Y)}$, scaling $N$ upward directly scales Lift upward, creating phantom correlations.

**Null-Invariant Alternatives:**

1. **Cosine Similarity (or Ochiai Coefficient):**
   $$\text{Cosine}(X, Y) = \frac{P(X \cap Y)}{\sqrt{P(X)P(Y)}} = \frac{\sigma(X \cup Y)}{\sqrt{\sigma(X)\sigma(Y)}}$$
   Notice that the global database size $N$ cancels out completely.

2. **Kulczynski Measure (Kulc):**
   $$\text{Kulc}(X, Y) = \frac{1}{2} \left( P(Y|X) + P(X|Y) \right) = \frac{1}{2} \left( \frac{\sigma(X \cup Y)}{\sigma(X)} + \frac{\sigma(X \cup Y)}{\sigma(Y)} \right)$$
   Evaluates strictly the arithmetic mean of the two directional confidence scores, isolating the evaluation entirely to transactions where at least one of the items actively exists.