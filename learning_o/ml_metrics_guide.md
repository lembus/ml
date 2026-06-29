# Machine Learning Metrics: A Comprehensive Reference Guide

*A standalone reference covering classification, regression, clustering, and statistical significance metrics for ML interview preparation.*

---

## How to Use This Guide

This guide assumes no prior exposure to evaluation metrics and builds from first principles. Each metric family follows the same six-section format:

1. **Motivation & Intuition**
2. **Conceptual Foundations**
3. **Mathematical Formulation**
4. **Worked Examples**
5. **Relevance to ML Practice**
6. **Common Pitfalls & Misconceptions**

A consolidated **Interview Preparation** section follows the reference material, organized by topic and difficulty.

---

# PART I — CLASSIFICATION METRICS

## Chapter 1. Basic Classification Metrics (Accuracy, Precision, Recall, F1, Fβ)

### 1.1 Motivation & Intuition

Imagine you build a spam email classifier. How do you know if it's any good? You might say *"it's correct 95% of the time"* — that's **accuracy**. But suppose only 2% of emails are actually spam. A classifier that labels everything "not spam" would be 98% accurate yet completely useless.

This thought experiment reveals why we need metrics beyond accuracy:

- Some predictions matter more than others (missing spam is bad; flagging a legitimate email as spam may be worse, or vice versa).
- Classes are often imbalanced, making accuracy misleading.
- Different applications have different cost structures.

**Real-world example — cancer screening.** A test that flags 100% of healthy patients as healthy has 99%+ accuracy (since cancer is rare), but misses every actual cancer. We need metrics that separately measure:

- Of the patients I flagged as having cancer, how many actually do? → **precision**
- Of the patients who actually have cancer, how many did I catch? → **recall**

### 1.2 Conceptual Foundations

For a binary classifier, any prediction falls into one of four categories relative to ground truth:

|                      | Actual Positive     | Actual Negative     |
| -------------------- | ------------------- | ------------------- |
| **Predicted Positive** | True Positive (TP)  | False Positive (FP) |
| **Predicted Negative** | False Negative (FN) | True Negative (TN)  |

This is the **confusion matrix**. Every classification metric derives from these four counts.

**Key terms.**

- **Positive class**: the class of interest (spam, disease present, click).
- **Negative class**: the other class.
- **True/False**: was the prediction correct?
- **Positive/Negative**: what did we predict?

So "False Positive" = "we predicted positive, but it was wrong" = a legitimate email flagged as spam.

**Derived metrics.**

- **Accuracy** — fraction of all predictions that are correct. *"Overall, how often am I right?"*
- **Precision** (Positive Predictive Value) — of items labeled positive, how many actually are? *"When I raise an alarm, how often is it real?"*
- **Recall** (Sensitivity, True Positive Rate) — of actual positives, how many did I catch?
- **Specificity** (True Negative Rate) — of actual negatives, how many did I correctly label negative?
- **F1** — harmonic mean of precision and recall; a single number balancing both.
- **Fβ** — weighted harmonic mean, tunable via β.

**Assumptions.**

- Labels are binary and reliable.
- The "positive" class is clearly defined.
- The relative cost of FP vs FN is relevant to which metric you emphasize.

**What breaks when assumptions fail.**

- Noisy labels → all metrics drift toward the noise floor.
- Class imbalance → accuracy becomes misleading.
- For score-based classifiers, precision/recall depend on the threshold chosen.

### 1.3 Mathematical Formulation

Let $n$ be the total number of predictions with TP, FP, TN, FN counted as above.

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN} = \frac{TP+TN}{n}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad \text{Specificity} = \frac{TN}{TN + FP}
$$

**F1 derivation.** We want a single metric that is high only when *both* precision and recall are high. The arithmetic mean fails — it gives 0.5 when precision=1 and recall=0. The **harmonic mean** of $a$ and $b$ is

$$
H(a,b) = \frac{2ab}{a+b}.
$$

The harmonic mean is small whenever either value is small, giving the desired "punishing" property:

$$
F_1 = \frac{2 \, P \, R}{P + R}.
$$

**Fβ derivation.** To weight recall β times as important as precision, use the weighted harmonic mean. Starting from

$$
F_\beta = \frac{1 + \beta^2}{\frac{1}{P} + \frac{\beta^2}{R}},
$$

algebraic manipulation gives

$$
\boxed{F_\beta = \frac{(1+\beta^2) \, P \, R}{\beta^2 \, P + R}.}
$$

- $\beta = 1$: F1, equal weight.
- $\beta = 2$: F2, recall twice as important (medical screening).
- $\beta = 0.5$: F0.5, precision twice as important (precise retrieval).

The $\beta^2$ (not $\beta$) reflects Van Rijsbergen's effectiveness-theory derivation: at $F_\beta = P = R$, the marginal rate of substitution between $P$ and $R$ is exactly $\beta$.

### 1.4 Worked Example

**Spam classifier on 1,000 emails.** Ground truth: 100 spam, 900 legitimate. The classifier predicts 120 as spam; of those 120, 80 are actually spam.

Build the confusion matrix:

- TP = 80 (correctly flagged spam)
- FP = 120 − 80 = 40 (legitimate emails flagged as spam)
- FN = 100 − 80 = 20 (spam missed)
- TN = 900 − 40 = 860 (legitimate correctly passed)

Metrics:

- Accuracy = (80 + 860) / 1000 = **0.94**
- Precision = 80 / 120 ≈ **0.667**
- Recall = 80 / 100 = **0.80**
- Specificity = 860 / 900 ≈ 0.956
- F1 = 2(0.667)(0.80) / (0.667 + 0.80) ≈ **0.727**
- F2 = (1+4)(0.667)(0.80) / (4·0.667 + 0.80) ≈ **0.770**

F2 > F1 > precision because we weight recall more heavily and recall (0.80) exceeds precision (0.667).

### 1.5 Relevance to ML Practice

- **Accuracy** — use when classes are balanced and errors are symmetric.
- **Precision** — use when FP is expensive (ad recommendations, treatment decisions, paging on-call).
- **Recall** — use when FN is expensive (cancer screening, fraud detection).
- **F1** — single metric when you want balanced treatment.
- **Fβ** — when cost asymmetry is known and you still want a single number.

**Choosing β.** If a false negative is $k \times$ as costly as a false positive, $\beta \approx \sqrt{k}$. In practice $\beta \in \{0.5, 1, 2\}$ covers most applications.

**Alternatives.** Log-loss grades probabilistic predictions; cost-weighted accuracy uses explicit dollar costs; Matthews Correlation Coefficient (MCC) handles severe imbalance.

### 1.6 Common Pitfalls

1. **Reporting accuracy on imbalanced data.** 99% accuracy on 99%-negative data may literally mean "always negative."
2. **Optimizing F1 without knowing the cost structure.** F1 treats precision and recall symmetrically even if business costs don't.
3. **Confusing precision with specificity.** Specificity's denominator is $TN+FP$; precision's is $TP+FP$. These diverge under class imbalance.
4. **Ignoring the decision threshold.** "Precision = 0.8" requires specifying the classification threshold.
5. **Comparing F1 across datasets with different class priors.** F1 is prior-dependent and not universally comparable.

---

## Chapter 2. Threshold-Dependent Metrics (ROC-AUC, PR-AUC, Threshold Selection)

### 2.1 Motivation & Intuition

Modern classifiers output a probability or score, not a hard label. To get a label we apply a **threshold** $\tau$: predict positive if score > $\tau$. Different thresholds produce different confusion matrices — hence different precision and recall.

Rather than evaluate at one threshold, we often want a **threshold-free** metric: *"how well does this model rank positives above negatives regardless of where we cut?"*

**Example.** Two spam filters both achieve F1 = 0.7 at threshold 0.5. But one's scores cluster around 0.4 and 0.6 (poor separation) and the other around 0.1 and 0.9 (excellent separation). The second is more robust — shifting the threshold degrades performance gracefully. Threshold-free metrics capture this.

### 2.2 Conceptual Foundations

**ROC curve** (Receiver Operating Characteristic): plot **TPR** (recall) vs **FPR** (1 − specificity) as $\tau$ varies from $+\infty$ (everything negative) down to $-\infty$ (everything positive).

- Random classifier: diagonal from (0,0) to (1,1).
- Perfect classifier: corner at (0,1).
- AUC = area under this curve.

**ROC-AUC** has an elegant probabilistic interpretation: the probability a random positive scores higher than a random negative.

- AUC = 0.5: random.
- AUC = 1.0: perfect.
- AUC = 0.0: perfectly wrong (flip predictions).

**Precision-Recall (PR) curve**: plot precision vs recall as $\tau$ varies.

- Chance baseline: horizontal at $y = $ prevalence of positive class.
- **More informative than ROC under heavy imbalance** because it ignores the (often huge) TN count.

**PR-AUC** or equivalently **Average Precision (AP)**: area under the PR curve.

**Assumptions.**

- Scores are ordinally comparable across examples.
- Labels are reliable.
- For PR curves, the "positive" class is the class of interest.

**What breaks.**

- Non-comparable scores across examples (e.g., ensembled models on different scales) → AUC meaningless.
- **ROC-AUC is insensitive to class imbalance** because FPR normalizes by TN, which is huge when negatives dominate. This makes ROC look optimistically high on rare-positive problems.

### 2.3 Mathematical Formulation

Let $s(x)$ denote the score. At threshold $\tau$:

$$
\text{TPR}(\tau) = P(s(X) \geq \tau \mid Y=1), \quad \text{FPR}(\tau) = P(s(X) \geq \tau \mid Y=0).
$$

**ROC-AUC**:

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(u)) \, du = P\bigl(s(X^+) > s(X^-)\bigr) + \tfrac{1}{2}P\bigl(s(X^+) = s(X^-)\bigr).
$$

This is the **Mann–Whitney U statistic** normalized. With $m$ positives and $n$ negatives:

$$
\widehat{\text{AUC}} = \frac{1}{mn} \sum_{i=1}^m \sum_{j=1}^n \Bigl[\mathbb{1}(s_i^+ > s_j^-) + \tfrac{1}{2}\mathbb{1}(s_i^+ = s_j^-)\Bigr].
$$

**Average Precision (PR-AUC):**

$$
\text{AP} = \sum_{k=1}^{n} (R_k - R_{k-1}) \cdot P_k,
$$

where $(P_k, R_k)$ is the precision and recall at the $k$-th threshold (score-sorted). This is a step (right-rectangle) approximation; linear interpolation on PR curves can be misleading because they need not be monotone.

**Optimal threshold selection.**

1. **Youden's J**: $\tau^* = \arg\max (\text{TPR} - \text{FPR})$. Geometrically, the point on the ROC curve farthest from the diagonal.
2. **F1-max**: $\tau^* = \arg\max F_1(\tau)$.
3. **Cost-sensitive**: for a *calibrated* classifier with costs $c_{FP}, c_{FN}$ and priors $\pi_0, \pi_1$,

$$
\tau^* = \frac{c_{FP} \pi_0}{c_{FP} \pi_0 + c_{FN} \pi_1}.
$$

4. **Target precision/recall**: smallest $\tau$ such that $P(\tau) \geq p_{\text{target}}$ (or $R(\tau) \geq r_{\text{target}}$).

### 2.4 Worked Example

Five positives (P) and five negatives (N) with scores:

- P: 0.9, 0.8, 0.6, 0.5, 0.3
- N: 0.7, 0.4, 0.35, 0.2, 0.1

**ROC-AUC** via Mann-Whitney: 25 total pairs. For each positive, count negatives it beats:

- 0.9 beats all 5 → 5
- 0.8 beats all 5 → 5
- 0.6 beats 0.4, 0.35, 0.2, 0.1 → 4
- 0.5 beats 0.4, 0.35, 0.2, 0.1 → 4
- 0.3 beats 0.2, 0.1 → 2

Total = 20. AUC = 20 / 25 = **0.80**.

**Threshold $\tau = 0.4$:**

- Positives flagged: {0.9, 0.8, 0.6, 0.5} → 4 TP, 1 FN.
- Negatives flagged: {0.7} → 1 FP, 4 TN.
- Precision = 0.80, Recall = 0.80, F1 = 0.80.

Only when $\tau$ crosses an actual score does the confusion matrix change — which is why ROC/PR curves are step functions on finite data.

### 2.5 Relevance to ML Practice

**ROC-AUC** — model comparison when downstream threshold isn't fixed; balanced problems; measuring inherent discrimination.

**PR-AUC** — highly imbalanced problems (fraud, rare disease). A 100,000:1 negative-to-positive ratio makes ROC-AUC deceptively high while PR-AUC exposes poor performance.

**Threshold selection in production:**

- Offline: optimize on validation per business objective.
- Online: rebalance with A/B test feedback, monitor for drift.
- Adaptive: update thresholds as class prior shifts.

**Trade-offs.**

- AUC ignores calibration: a model with AUC = 0.95 might have all probabilities near 0.5 — fine for ranking, poor for decisions.
- AUC is insensitive to ordering at early vs late regions. Two models with equal AUC can have very different top-k precision.

### 2.6 Common Pitfalls

1. **Using ROC-AUC for highly imbalanced problems.** TNs dominate FPR; AUC looks great despite poor absolute performance. Use PR-AUC.
2. **Treating AUC as a deployment metric.** Deployment uses a threshold; same AUC can yield very different F1.
3. **Linearly interpolating PR curves.** Between points on a PR curve, intermediate (P, R) are not valid. Use step interpolation or compute AP directly.
4. **Forgetting confidence intervals.** With few positives, AUC is noisy. Use bootstrap or the Hanley-McNeil variance formula.
5. **Choosing threshold on test set.** Threshold must be picked on validation, then evaluated on test.

---

## Chapter 3. Multiclass Metrics (Averaging Strategies, Confusion Matrix Analysis)

### 3.1 Motivation & Intuition

Binary metrics don't extend directly to classification with $K > 2$ classes. With classes {cat, dog, bird}, *"precision of what?"* We need principled ways to (1) define per-class metrics and (2) aggregate them into a single number. The aggregation choice is a *modeling decision*, not a mathematical one: do we care equally about every class, or proportionally to class frequency?

### 3.2 Conceptual Foundations

$K \times K$ confusion matrix: rows = actual classes, columns = predicted. The diagonal holds correct predictions.

**One-vs-rest view.** For class $k$, treat it as the positive class and all others as negative. Compute $TP_k, FP_k, FN_k, TN_k$ and per-class $P_k, R_k, F1_k$.

**Averaging strategies.**

- **Macro-average**: arithmetic mean across classes. Equal weight per class.
  $$\text{Macro-F1} = \frac{1}{K}\sum_k F1_k$$
- **Weighted average**: weighted by class support.
  $$\text{Weighted-F1} = \sum_k \frac{|Y=k|}{n} F1_k$$
- **Micro-average**: aggregate TP/FP/FN *across* classes, then compute.
  $$\text{Micro-P} = \frac{\sum_k TP_k}{\sum_k (TP_k+FP_k)}, \quad \text{Micro-R} = \frac{\sum_k TP_k}{\sum_k (TP_k+FN_k)}$$

For single-label multiclass, micro-P = micro-R = accuracy. Every mistake is one class's FP and another's FN, so $\sum FP_k = \sum FN_k$, making both denominators equal to $n$.

**Interpretation.**

- Macro: every class matters equally. Use when minorities matter.
- Weighted: class-frequency-weighted. Reflects performance on "typical" samples.
- Micro: aggregate across classes. Equals accuracy for single-label problems.

**What breaks.**

- Macro on imbalanced data with a rare tricky class: one bad class drags macro-F1 down. (Sometimes this is the point.)
- Micro on imbalanced data: dominated by majority class (= accuracy).

### 3.3 Mathematical Formulation

With confusion matrix $C$, $C_{ij}$ = examples of true class $i$ predicted as $j$:

- $TP_k = C_{kk}$
- $FP_k = \sum_{i \neq k} C_{ik}$ (column sum minus diagonal)
- $FN_k = \sum_{j \neq k} C_{kj}$ (row sum minus diagonal)

For **multi-label** classification, macro and micro genuinely differ because a single example may contribute multiple TPs.

**Cohen's κ** corrects for chance agreement:

$$
\kappa = \frac{p_o - p_e}{1 - p_e},
$$

where $p_o$ = observed agreement (= accuracy) and $p_e$ = expected agreement if predictions were independent of labels with the same marginals.

### 3.4 Worked Example

Three-class classifier (A, B, C), 100 examples. Rows = true, columns = predicted:

|        | Pred A | Pred B | Pred C |
| ------ | ------ | ------ | ------ |
| True A | 40     | 5      | 5      |
| True B | 5      | 25     | 0      |
| True C | 10     | 5      | 5      |

Class supports: A = 50, B = 30, C = 20.

Per class:

- A: TP=40, FP=15, FN=10 → P=0.727, R=0.800, F1=0.762
- B: TP=25, FP=10, FN=5 → P=0.714, R=0.833, F1=0.769
- C: TP=5, FP=5, FN=15 → P=0.500, R=0.250, F1=0.333

Aggregated:

- Macro-F1 = (0.762 + 0.769 + 0.333) / 3 = **0.621**
- Weighted-F1 = 0.5(0.762) + 0.3(0.769) + 0.2(0.333) = **0.679**
- Micro-F1 = (40+25+5) / 100 = 0.70 = accuracy

Macro-F1 is lowest because class C performs poorly and macro weights all classes equally. Weighted-F1 downweights C (only 20% of samples) so its poor performance matters less.

### 3.5 Relevance to ML Practice

- **Class-balanced problems**: any averaging works; report accuracy and macro-F1.
- **Imbalanced problems, minority classes important**: macro-F1 is standard.
- **Multi-label**: micro vs macro differ meaningfully; choose based on per-class vs aggregate focus.
- **Hierarchical classes**: hierarchical F1 or top-$k$ accuracy.
- **Competition conventions**: ImageNet uses top-1/top-5 accuracy; COCO uses mAP.

**Confusion-matrix inspection is essential for debugging.** Aggregate metrics hide structure — off-diagonal entries reveal systematic errors. E.g., ImageNet models often confuse dog breeds but rarely dogs with cars.

**Normalization.** Normalize rows to get recall per class, columns for precision per class — choose based on the question.

### 3.6 Common Pitfalls

1. **Accuracy on imbalanced multiclass.** Hides minority-class failure.
2. **Confusing micro-F1 with accuracy for multi-label.** They differ for multi-label, coincide for single-label.
3. **Reporting only one averaging.** Present both macro and weighted; justify the choice.
4. **Skipping confusion matrices.** Aggregate metrics obscure *which* classes fail.
5. **Comparing micro-F1 across datasets.** Dominated by class prior — apples to oranges.

---

## Chapter 4. Ranking Metrics (Average Precision, NDCG, MRR, MAP)

### 4.1 Motivation & Intuition

In search, recommendation, and retrieval, the output is an *ordered list* rather than a single label. What matters:

- Are the relevant items near the top?
- How relevant are they (not just binary)?
- Does the user find something useful quickly?

Returning the correct document at position 1 is better than returning it at position 100, even if both technically "retrieve" it. Ranking metrics quantify this *up-weight-the-top* intuition.

### 4.2 Conceptual Foundations

For a query $q$, rank a set of items by relevance. Ground truth is binary (relevant/not) or graded (0–4 scale).

**Reciprocal Rank (RR).** If the first relevant item is at rank $k$, $RR = 1/k$. If no relevant item, $RR = 0$.

**Mean Reciprocal Rank (MRR).** Average RR across queries. High MRR means the *first* relevant result tends to appear near the top. MRR ignores everything after the first hit.

**Average Precision (AP).** For binary relevance on one query,

$$
\text{AP} = \frac{1}{|\text{rel}|} \sum_{k=1}^{n} P@k \cdot \mathbb{1}[\text{item } k \text{ is relevant}].
$$

$P@k$ = fraction of top-$k$ that are relevant. AP averages $P@k$ over ranks where relevant items appear — implicitly weighting higher ranks more.

**Mean Average Precision (MAP).** Mean of AP over queries. IR standard.

**Normalized Discounted Cumulative Gain (NDCG).** For graded relevance:

- **Gain** at rank $k$: $\text{rel}_k$ or $2^{\text{rel}_k} - 1$ (exponential gain emphasizes high grades).
- **Discount**: $\log_2(k+1)$.
- **DCG@K** = $\sum_{k=1}^{K} \text{gain}_k / \log_2(k+1)$.
- **IDCG@K** = DCG of the *ideal* ordering (items sorted by true relevance).
- **NDCG@K** = DCG@K / IDCG@K ∈ [0, 1].

Normalization makes NDCG comparable across queries with different relevance profiles.

**Assumptions.**

- Relevance annotations are correct.
- Position bias follows the log discount (a reasonable but imperfect proxy for attention).
- Queries are independent.

**What breaks.**

- Incomplete labels: MAP penalizes ranking unlabeled-but-actually-relevant items highly.
- Non-standard discount: if users scan differently (mobile vs desktop), log discount may not fit.
- Sparse grades → NDCG noisy.

### 4.3 Mathematical Formulation

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

$$
\text{AP}(q) = \frac{\sum_{k=1}^n P@k(q) \cdot r_k(q)}{\sum_k r_k(q)}, \quad \text{MAP} = \frac{1}{|Q|} \sum_{q} \text{AP}(q)
$$

$$
\text{DCG@K} = \sum_{k=1}^K \frac{2^{\text{rel}_k}-1}{\log_2(k+1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

### 4.4 Worked Example

Ten documents for one query, graded 0–3:

| Rank  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| ----- | - | - | - | - | - | - | - | - | - | -- |
| Grade | 3 | 2 | 3 | 0 | 1 | 2 | 3 | 2 | 0 | 0  |

**MRR** (relevant ≡ grade ≥ 1): first relevant at rank 1 → MRR contribution = 1.0.

**AP@10** (binary relevance). Relevant positions: {1, 2, 3, 5, 6, 7, 8}.

| Relevant rank $k$ | $P@k$ |
| ----------------- | ----- |
| 1                 | 1/1   |
| 2                 | 2/2   |
| 3                 | 3/3   |
| 5                 | 4/5   |
| 6                 | 5/6   |
| 7                 | 6/7   |
| 8                 | 7/8   |

$\text{AP} = (1 + 1 + 1 + 0.800 + 0.833 + 0.857 + 0.875) / 7 \approx \mathbf{0.909}$.

**NDCG@5** using exponential gain $2^{\text{rel}}-1$:

| $k$ | rel | gain | $\log_2(k+1)$ | discounted |
| --- | --- | ---- | ------------- | ---------- |
| 1   | 3   | 7    | 1.000         | 7.000      |
| 2   | 2   | 3    | 1.585         | 1.893      |
| 3   | 3   | 7    | 2.000         | 3.500      |
| 4   | 0   | 0    | 2.322         | 0.000      |
| 5   | 1   | 1    | 2.585         | 0.387      |

DCG@5 ≈ **12.780**. Ideal top-5 ordering (sorting all grades): 3, 3, 3, 2, 2 →

$$
\text{IDCG@5} = \frac{7}{1.000} + \frac{7}{1.585} + \frac{7}{2.000} + \frac{3}{2.322} + \frac{3}{2.585} \approx 17.369
$$

NDCG@5 = 12.780 / 17.369 ≈ **0.736**.

### 4.5 Relevance to ML Practice

- **Search engines**: MAP, NDCG.
- **Recommenders**: NDCG, MAP, HR@k.
- **Q&A**: MRR.
- **Object detection**: mAP (COCO, Pascal VOC), but with an IoU threshold sweep — a different construction from IR's MAP.

**When to use which.**

- Binary relevance, single relevant answer: MRR.
- Binary relevance, multiple relevant: MAP.
- Graded relevance: NDCG.
- Cut off at $k$ (ignore deep results): @K variants.

**Trade-offs.**

- MRR is simple and interpretable but throws away info beyond the first hit.
- NDCG depends on the chosen gain formula (linear vs exponential).
- MAP assumes label completeness, rarely true. Pooling-based evaluation (TREC) mitigates this.

### 4.6 Common Pitfalls

1. **Computing MAP with incomplete labels.** Unlabeled items default to "not relevant," penalizing legitimate retrievals.
2. **Mixing NDCG formulas.** Linear vs exponential gain yield different numbers. State your convention.
3. **Omitting the cutoff.** NDCG@10 ≠ NDCG@5.
4. **Conflating COCO mAP and IR MAP.** Different constructions (IoU sweep in detection).
5. **Ignoring position bias in training data.** Click data is biased toward items shown higher — use inverse propensity weighting or randomized exploration.

---

## Chapter 5. Calibration Metrics (ECE, Reliability Diagrams)

### 5.1 Motivation & Intuition

A classifier outputs probability 0.8. What does it mean? Ideally: among examples with output 0.8, about 80% are positive. A **calibrated** model's confidence matches empirical frequency.

Calibration matters when probabilities drive decisions:

- Medical: "30% probability of cancer" differs meaningfully from 90%.
- Weather: "70% rain" must match historical frequency.
- Active learning: sample where the model is genuinely uncertain.
- Model combination: averaging requires comparable probability scales.

Modern deep nets are typically *overconfident* — scores cluster near 0 and 1 — despite good accuracy (Guo et al., 2017).

### 5.2 Conceptual Foundations

**Reliability diagram.** Bin predictions by confidence (e.g., [0, 0.1), [0.1, 0.2), …). For each bin plot (mean confidence in bin) vs (empirical accuracy in bin). A perfectly calibrated model traces $y = x$.

**Expected Calibration Error (ECE).** Weighted gap between confidence and accuracy:

$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|.
$$

**Maximum Calibration Error (MCE).** Worst bin gap: $\max_m |\text{acc}(B_m) - \text{conf}(B_m)|$.

**Brier Score.** Squared error between predicted probability and binary outcome:

$$
\text{Brier} = \frac{1}{n} \sum_i (p_i - y_i)^2.
$$

Brier decomposes into reliability + resolution − uncertainty, capturing both calibration and sharpness.

**Log-loss / NLL** is also calibration-sensitive: punishes confident wrong predictions harshly.

**Assumptions.**

- Predictions in [0, 1].
- Each bin contains enough samples for reliable accuracy estimates.
- Binning is meaningful.

**What breaks.**

- Too few samples per bin → noisy estimates.
- Uniform binning when most predictions cluster → sparse bins; use adaptive (equal-frequency) binning.
- Aggregate ECE can hide subgroup miscalibration.

### 5.3 Mathematical Formulation

For bin $B_m$:

$$
\text{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} p_i, \quad \text{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} y_i
$$

**Calibration correction methods.**

- **Platt scaling**: fit $\sigma(a \cdot s + b)$ on held-out data.
- **Isotonic regression**: non-parametric monotone map from score to probability.
- **Temperature scaling**: divide logits by $T$: $p = \text{softmax}(z/T)$. Fit $T$ to minimize NLL on validation. Preserves argmax (so top-1 accuracy unchanged) while flattening overconfident distributions.

### 5.4 Worked Example

100 predictions in 10 bins of width 0.1:

| Bin          | Count | Mean conf | Accuracy |
| ------------ | ----- | --------- | -------- |
| [0.0, 0.1)   | 5     | 0.05      | 0.00     |
| [0.1, 0.2)   | 10    | 0.15      | 0.10     |
| [0.2, 0.3)   | 15    | 0.25      | 0.20     |
| [0.3, 0.4)   | 10    | 0.35      | 0.30     |
| [0.4, 0.5)   | 10    | 0.45      | 0.50     |
| [0.5, 0.6)   | 10    | 0.55      | 0.40     |
| [0.6, 0.7)   | 10    | 0.65      | 0.70     |
| [0.7, 0.8)   | 10    | 0.75      | 0.70     |
| [0.8, 0.9)   | 10    | 0.85      | 0.80     |
| [0.9, 1.0]   | 10    | 0.95      | 0.80     |

Contributions to ECE:

- (5/100)(0.05) = 0.0025
- (10/100)(0.05) = 0.005
- (15/100)(0.05) = 0.0075
- (10/100)(0.05) = 0.005
- (10/100)(0.05) = 0.005
- (10/100)(0.15) = 0.015
- (10/100)(0.05) = 0.005
- (10/100)(0.05) = 0.005
- (10/100)(0.05) = 0.005
- (10/100)(0.15) = 0.015

ECE ≈ **0.070** (about 7 percentage points miscalibrated on average).

MCE = max(0.15, 0.15, …) = **0.15**. The two worst bins are [0.5, 0.6) (under-predicting) and [0.9, 1.0] (over-predicting).

### 5.5 Relevance to ML Practice

**Modern deep nets are overconfident**: Guo et al. showed ResNets have ECE > 5% after standard training; temperature scaling reduces ECE below 1% with a single parameter.

**Calibration-critical applications**: medical triage, loan/legal decisions, risk scoring, cost-sensitive decisions, ensembling, active learning, Bayesian optimization, RL.

**Calibration-non-critical**: ranking (AUC is calibration-invariant), pure argmax classification.

**Trade-offs.**

- Calibration can slightly change accuracy; usually negligible.
- Temperature scaling: simple, effective for neural nets, single parameter.
- Isotonic regression: more flexible but data-hungry.

### 5.6 Common Pitfalls

1. **Only reporting overall ECE.** Subgroups may be badly miscalibrated while aggregate looks fine — check subgroup ECE.
2. **Too few or too many bins.** Prefer adaptive (equal-frequency) binning.
3. **Measuring calibration on training data.** Must be held-out.
4. **Confusing calibration with accuracy.** A random 50/50 predictor is perfectly calibrated yet useless. Calibration is *necessary but not sufficient*.
5. **Applying temperature scaling across distributions.** Calibration doesn't transfer across domain shifts.

---


# PART II — REGRESSION METRICS

## Chapter 6. Scale-Dependent Regression Metrics (MSE, RMSE, MAE, MAPE, sMAPE)

### 6.1 Motivation & Intuition

Regression predicts continuous values. The "error" is the gap between prediction and ground truth, but different aggregations have different properties:

- Penalize large errors disproportionately? (MSE)
- Treat errors linearly? (MAE)
- Measure relative error? (MAPE, sMAPE)
- Keep units interpretable? (RMSE rather than MSE)

The choice depends on outlier treatment, whether absolute or relative errors matter, and whether targets can hit zero (breaks MAPE).

### 6.2 Conceptual Foundations

With true $y_i$, predicted $\hat{y}_i$, residual $e_i = y_i - \hat{y}_i$:

- **MSE**: $\frac{1}{n}\sum e_i^2$ — quadratic penalty, units of target squared.
- **RMSE**: $\sqrt{\text{MSE}}$ — same units as target, interpretable.
- **MAE**: $\frac{1}{n}\sum |e_i|$ — linear penalty, robust to outliers.
- **MAPE**: $\frac{100}{n}\sum |e_i/y_i|$ — scale-invariant, breaks at $y_i = 0$, asymmetric.
- **sMAPE**: $\frac{100}{n}\sum \frac{|e_i|}{(|y_i|+|\hat{y}_i|)/2}$ — partially addresses MAPE asymmetry.

**Implicit distributional assumptions.**

- MSE: Gaussian errors (MSE minimization ≡ MLE under Gaussian noise).
- MAE: Laplace errors.
- MAPE: multiplicative errors; target positive.

**What breaks.**

- MSE is outlier-dominated; a single huge error can dominate.
- MAE is non-differentiable at 0 (handle with subgradients or Huber).
- MAPE undefined at $y_i = 0$, explodes as $y_i \to 0$.
- sMAPE has edge cases when both $y$ and $\hat{y} = 0$.

### 6.3 Mathematical Formulation

$$
\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2, \quad \text{RMSE} = \sqrt{\text{MSE}}
$$

$$
\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|
$$

$$
\text{MAPE} = \frac{100}{n}\sum_i \frac{|y_i - \hat{y}_i|}{|y_i|}
$$

$$
\text{sMAPE} = \frac{100}{n}\sum_i \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
$$

**Optimal-prediction connection.**

- Minimizer of $E[(Y - \hat{y})^2]$: conditional mean $E[Y|X]$.
- Minimizer of $E[|Y - \hat{y}|]$: conditional median.

This is why MAE-trained models predict medians and MSE-trained models predict means — important when the conditional distribution is skewed.

**Bias-variance decomposition** of MSE: $E[(Y - \hat{Y})^2] = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{noise}}$.

### 6.4 Worked Example

Five houses:

| $i$ | $y$ | $\hat{y}$ | $e$ | $|e|$ | $e^2$ | $|e|/|y|$ |
| --- | --- | --------- | --- | ----- | ----- | --------- |
| 1   | 100 | 110       | −10 | 10    | 100   | 0.10      |
| 2   | 200 | 180       | 20  | 20    | 400   | 0.10      |
| 3   | 300 | 330       | −30 | 30    | 900   | 0.10      |
| 4   | 400 | 400       | 0   | 0     | 0     | 0.00      |
| 5   | 500 | 600       | −100| 100   | 10000 | 0.20      |

- MSE = (100 + 400 + 900 + 0 + 10000) / 5 = **2280**
- RMSE = √2280 ≈ **47.75**
- MAE = (10+20+30+0+100) / 5 = **32**
- MAPE = 100 × (0.1+0.1+0.1+0+0.2) / 5 = **10%**
- sMAPE: terms = 10/105, 20/190, 30/315, 0/400, 100/550 = (0.0952, 0.1053, 0.0952, 0, 0.1818). Mean × 100 ≈ **9.55%**

RMSE ≫ MAE is a diagnostic for outlier influence — house 5's $|e|=100$ contributes 10,000 to MSE but only 100 to MAE.

### 6.5 Relevance to ML Practice

**Training loss vs evaluation metric mismatch is a common mistake.** If you report RMSE but train with MAE, you're producing medians to be scored on a mean metric.

- **MSE/RMSE**: default when Gaussian noise is plausible. Smooth, stable gradients, closed-form solutions for linear regression.
- **MAE**: outliers common; robustness > small-RMSE gains.
- **MAPE**: retail/finance/demand forecasting, target bounded away from zero.
- **sMAPE**: when MAPE is almost right but asymmetry bites.

**Alternative**: log-transform a positive target then use MSE on logs — gives multiplicative-error interpretation and avoids division by zero.

**Trade-offs.** MSE: outlier-sensitive but mathematically convenient. MAE: robust but non-smooth; Huber is a compromise. MAPE: scale-invariant but unstable near zero.

### 6.6 Common Pitfalls

1. **Reporting MSE instead of RMSE.** RMSE is in original units; preferred for reporting.
2. **MAPE on targets that hit zero.** Division by zero.
3. **Conflating MSE-minimizer with "best forecast."** Best depends on the loss you actually care about.
4. **Comparing RMSE across datasets.** RMSE = 10 on prices-in-dollars ≠ RMSE = 10 on prices-in-millions.
5. **Forgetting sMAPE's residual asymmetry.** "Symmetric" only in a limited sense.

---

## Chapter 7. Scale-Independent Regression Metrics (R², Adjusted R²)

### 7.1 Motivation & Intuition

RMSE = 10 — good or bad? If target ranges 1–10, terrible. If 1–1000, excellent. Scale-dependent metrics need context.

**R²** provides a relative measure: *"How much better is this model than predicting the mean?"* Scale-free, bounded (typically) in $(-\infty, 1]$, with interpretable anchors.

### 7.2 Conceptual Foundations

Compare to a null baseline (predict $\bar{y}$ always). Its error = total variance of $y$. Our model's error = residual variance. $R^2 = 1 -$ (our error / null error).

**Properties.**

- $R^2 = 1$: perfect.
- $R^2 = 0$: equivalent to predicting $\bar{y}$.
- $R^2 < 0$: worse than predicting the mean — happens on held-out data.
- For OLS linear regression with intercept on *training* data, $R^2 \in [0, 1]$.

**Adjusted R².** Plain $R^2$ never decreases when you add a feature (it can at best stay the same). Adjusted $R^2$ penalizes complexity:

$$
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n-1)}{n - k - 1}
$$

**Assumptions.**

- $\bar{y}$ computed on the same split as $R^2$.
- Data i.i.d.
- For linear regression theoretical results: homoscedastic Gaussian errors.

**What breaks.**

- On test set, $R^2 < 0$ is possible — not a bug; the model is worse than the mean.
- Tiny datasets → high-variance $R^2$.
- Time series: $\bar{y}$ baseline is naïve; a naïve-forecast baseline is often more meaningful (giving "forecast skill").

### 7.3 Mathematical Formulation

$SS_{\text{tot}} = \sum_i (y_i - \bar{y})^2$, $SS_{\text{res}} = \sum_i (y_i - \hat{y}_i)^2$.

$$
R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

**Connection to correlation.** For simple linear regression of $y$ on $x$, $R^2 = r_{xy}^2$. For general OLS, $R^2 = \text{Corr}(y, \hat{y})^2$.

**Derivation of adjusted form.** Use unbiased variance estimates with degrees of freedom:

$$
R^2_{\text{adj}} = 1 - \frac{SS_{\text{res}}/(n-k-1)}{SS_{\text{tot}}/(n-1)} = 1 - (1-R^2)\frac{n-1}{n-k-1}
$$

### 7.4 Worked Example

Same 5 houses. $\bar{y} = 300$.

$SS_{\text{tot}} = 200^2 + 100^2 + 0 + 100^2 + 200^2 = 100{,}000$
$SS_{\text{res}} = 100 + 400 + 900 + 0 + 10000 = 11{,}400$

$R^2 = 1 - 11{,}400 / 100{,}000 = \mathbf{0.886}$

With $k = 3$ features on $n = 5$ samples:

$$
R^2_{\text{adj}} = 1 - (1 - 0.886)\frac{4}{1} = 1 - 0.456 = \mathbf{0.544}
$$

The adjusted value drops dramatically — a deserved penalty for using 3 features on only 5 points.

### 7.5 Relevance to ML Practice

- **Regression on tabular data**: $R^2$ is the standard summary.
- **Feature engineering**: adjusted $R^2$ for comparing models of different dimensionality.
- **Time series**: use naïve-forecast R² (replace $\bar{y}$ with last observation or seasonal average).
- **Non-linear models**: $R^2$ still works; "explained variance" interpretation is looser.

**Alternatives**: AIC, BIC (information criteria penalizing complexity), cross-validated R² ($Q^2$).

### 7.6 Common Pitfalls

1. **Assuming $R^2 \geq 0$ always.** Negative on test data with bad models.
2. **Using training $R^2$ for feature selection.** Monotonically non-decreasing with features — use adjusted $R^2$, AIC, or CV.
3. **Treating $R^2$ as comparable across datasets.** It's unitless but still distribution-dependent; a high $R^2$ on easy data doesn't mean a better model than a lower $R^2$ on hard data.
4. **Interpreting $R^2$ as accuracy.** It's explained variance, not correctness rate.
5. **Computing $R^2$ on transformed scale and reporting on original.** Back-transforming flips interpretations.

---

## Chapter 8. Robust Regression Metrics (Huber Loss, Quantile Loss)

### 8.1 Motivation & Intuition

MSE is fragile to outliers; MAE is robust but non-smooth. **Huber loss** combines them: quadratic near zero (smooth, efficient), linear far from zero (robust).

**Quantile loss** optimizes for a specific conditional quantile, not the mean. Useful for:

- Prediction intervals (P5, P95).
- Asymmetric business costs (stockout > overstock).

### 8.2 Conceptual Foundations

**Huber loss** (parameter $\delta > 0$):

$$
L_\delta(e) = \begin{cases} \tfrac{1}{2} e^2 & |e| \leq \delta \\ \delta(|e| - \tfrac{1}{2}\delta) & |e| > \delta \end{cases}
$$

- Small $|e|$: quadratic, like MSE.
- Large $|e|$: linear with slope $\delta$, like MAE scaled.
- Continuous and differentiable at the transition.

**Quantile (pinball) loss** at quantile $\tau \in (0, 1)$:

$$
L_\tau(e) = \begin{cases} \tau \cdot e & e \geq 0 \\ (\tau - 1) \cdot e & e < 0 \end{cases}
$$

- $\tau = 0.5$: equivalent (up to scale) to MAE → predicts median.
- $\tau < 0.5$: penalizes over-prediction more.
- $\tau > 0.5$: penalizes under-prediction more.

**Assumptions.**

- Huber: $\delta$ tuned appropriately. Too small → effectively MAE. Too large → effectively MSE.
- Quantile: adequate data per quantile; extreme quantiles unstable with little data.

### 8.3 Mathematical Formulation

Huber derivative (bounded gradient for outliers):

$$
\frac{\partial L_\delta}{\partial e} = \begin{cases} e & |e| \leq \delta \\ \delta \cdot \text{sign}(e) & \text{otherwise} \end{cases}
$$

**Quantile loss minimizer derivation.** Let $f$ be the density of $Y$.

$$
E[L_\tau(Y - \hat{y})] = \tau \int_{\hat{y}}^{\infty}(y-\hat{y}) f(y) dy + (\tau-1)\int_{-\infty}^{\hat{y}}(y-\hat{y})f(y)dy.
$$

Differentiate w.r.t. $\hat{y}$:

$$
-\tau P(Y \geq \hat{y}) + (1-\tau) P(Y < \hat{y}) = 0
$$

Solving gives $P(Y < \hat{y}) = \tau$ — $\hat{y}$ is the $\tau$-quantile of $Y$.

### 8.4 Worked Example

Residuals $e = [-2, 1, 0.5, 10, -1]$, $\delta = 1$.

Huber values:

- $e = -2$, $|e|>1$: $1 \cdot (2 - 0.5) = 1.5$
- $e = 1$: $0.5 \cdot 1 = 0.5$
- $e = 0.5$: $0.5 \cdot 0.25 = 0.125$
- $e = 10$: $1 \cdot (10 - 0.5) = 9.5$
- $e = -1$: $0.5 \cdot 1 = 0.5$

Sum = 12.125, mean ≈ **2.425**.

Compare plain squared loss: (4+1+0.25+100+1)/5 = 21.25. The outlier contributes 100 to MSE but only 9.5 to Huber — about 5× less dominated.

**Quantile loss at $\tau = 0.9$** on the same residuals. Convention: $e = y - \hat{y}$. If $e > 0$ (under-predicted), penalty $= 0.9 |e|$; if $e < 0$ (over-predicted), penalty $= 0.1 |e|$.

- $e=-2$: 0.2
- $e=1$: 0.9
- $e=0.5$: 0.45
- $e=10$: 9.0
- $e=-1$: 0.1

Sum = 10.65, mean ≈ **2.13**. A predictor minimizing this aims higher (closer to the 90th percentile) to avoid under-prediction.

### 8.5 Relevance to ML Practice

**Huber use cases.** Robust regression with outliers; deep learning where rare large errors occur (RL rewards, reward modeling); default option in GBDT frameworks.

**Quantile use cases.** Probabilistic forecasting (P10/P50/P90); inventory planning; asymmetric loss; available in LightGBM, XGBoost, quantile forests.

**Conformal prediction** builds on quantile-loss ideas for calibrated prediction intervals.

**Trade-offs.** Huber needs $\delta$ tuning. Quantile regression with multiple quantiles needs multiple models (or multi-output). Neither gives a clean single-summary number.

### 8.6 Common Pitfalls

1. **Arbitrary $\delta$.** Heuristic: $\delta \approx 1.5 \cdot \text{MAD}$ of pilot residuals, or Huber's 1.345 for 95% Gaussian efficiency.
2. **Confusing quantile regression with residual quantile.** Quantile regression predicts the conditional quantile of $Y|X$; it's not summarizing residuals.
3. **Reporting Huber loss without $\delta$.** Results are incomparable.
4. **Quantile crossing.** Independent P10 and P90 models can produce P10 > P90 for some $x$. Use monotone constraints, quantile forests, or post-hoc sorting.
5. **Using quantile loss for point prediction.** Its output is quantile-specific — not a "best point" unless $\tau = 0.5$.

---


# PART III — CLUSTERING METRICS

## Chapter 9. Internal Clustering Metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)

### 9.1 Motivation & Intuition

Clustering produces groupings without ground truth. Evaluating *without labels* requires **internal** metrics — measures of geometric structure:

- Are points close to their cluster? (cohesion)
- Are clusters far from each other? (separation)

"Good" clustering = tight clusters well separated.

### 9.2 Conceptual Foundations

Data $X = \{x_1, \ldots, x_n\}$, clusters $C_1, \ldots, C_K$ with centroids $\mu_k$.

**Silhouette** for point $i$ in cluster $C_k$:

- $a(i)$ = mean distance from $i$ to other points in $C_k$ (cohesion).
- $b(i) = \min_{k' \neq k}$ of mean distance from $i$ to $C_{k'}$ (separation).
- $s(i) = (b(i) - a(i)) / \max(a(i), b(i)) \in [-1, 1]$.

$s(i) \approx 1$: well-clustered. $\approx 0$: on boundary. $< 0$: likely misclustered.

**Davies-Bouldin (DB) index.** Mean worst-case cluster-pair ratio:

- $R_{ij} = (\sigma_i + \sigma_j) / d(\mu_i, \mu_j)$ ($\sigma$ = cluster spread).
- $D_i = \max_{j \neq i} R_{ij}$.
- $DB = \text{mean}_i D_i$. Lower is better.

**Calinski-Harabasz (CH) index.**

$$
CH = \frac{\text{BCSS} / (K-1)}{\text{WCSS} / (n-K)}
$$

where BCSS = between-cluster sum of squares, WCSS = within. Higher is better.

**Assumptions.**

- Distances meaningful (Euclidean by default).
- Clusters roughly convex/blob-shaped.
- Features on comparable scales (standardize first).

**What breaks.** Non-convex clusters (spirals, moons) — centroid-based metrics fail. High-dimensional data — distance concentration makes all metrics degenerate. Heavy-tailed cluster sizes distort averages.

### 9.3 Mathematical Formulation

**Silhouette:**

$$
a(i) = \frac{1}{|C_{k(i)}|-1}\sum_{j \in C_{k(i)}, j \neq i} d(x_i, x_j)
$$

$$
b(i) = \min_{k' \neq k(i)} \frac{1}{|C_{k'}|}\sum_{j \in C_{k'}} d(x_i, x_j)
$$

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

**Davies-Bouldin:**

$$
\sigma_k = \frac{1}{|C_k|}\sum_{i \in C_k} d(x_i, \mu_k)
$$

$$
R_{ij} = \frac{\sigma_i + \sigma_j}{d(\mu_i, \mu_j)}, \quad D_i = \max_{j \neq i} R_{ij}, \quad DB = \frac{1}{K}\sum_k D_k
$$

**Calinski-Harabasz:**

$$
BCSS = \sum_k |C_k| \|\mu_k - \bar{\mu}\|^2, \quad WCSS = \sum_k \sum_{i \in C_k} \|x_i - \mu_k\|^2
$$

$$
CH = \frac{BCSS/(K-1)}{WCSS/(n-K)}
$$

### 9.4 Worked Example

Six points in 1D at [1, 2, 3, 10, 11, 12] clustered as {1, 2, 3} and {10, 11, 12}. Centroids 2 and 11. Grand mean 6.5.

**Silhouette for $x=1$:**

- $a(1) = (|1-2| + |1-3|)/2 = 1.5$
- $b(1) = (|1-10|+|1-11|+|1-12|)/3 = 10$
- $s(1) = (10 - 1.5) / 10 = 0.85$

By symmetry, silhouette ≈ **0.85** for all points — excellent.

**Davies-Bouldin:**

- $\sigma_A = (1+0+1)/3 \approx 0.667$, $\sigma_B \approx 0.667$
- $d(\mu_A, \mu_B) = 9$
- $R_{AB} = (0.667 + 0.667)/9 \approx 0.148$
- $DB \approx \mathbf{0.148}$ — very low → excellent

**Calinski-Harabasz:**

- $BCSS = 3(2-6.5)^2 + 3(11-6.5)^2 = 121.5$
- $WCSS = 2 + 2 = 4$
- $CH = (121.5/1)/(4/4) = \mathbf{121.5}$ — very high → excellent

All three metrics agree; clustering is strongly supported.

### 9.5 Relevance to ML Practice

**Choosing $K$:**

- Elbow method on WCSS (heuristic).
- Silhouette curve over $K$; pick argmax.
- CH / DB curves similarly.

**Limitations.** Internal metrics favor spherical clusters → biased toward K-means-compatible structures. DBSCAN and density-based clustering often score poorly on these metrics even when the clustering is correct. Use as sanity checks, not absolute truth.

### 9.6 Common Pitfalls

1. **Not standardizing features.** Scale imbalance dominates distances.
2. **Silhouette-maximizing K without domain input.** Often $K=2$ wins trivially.
3. **Euclidean distance on categorical/mixed data.** Use Gower or appropriate metric.
4. **Applying to non-convex clusters.** Valid clusterings can score poorly.
5. **Single-metric decisions.** Report multiple; disagreement is informative.

---

## Chapter 10. External Clustering Metrics (ARI, NMI, Homogeneity, Completeness)

### 10.1 Motivation & Intuition

When ground-truth labels exist (for evaluation only; clustering is still unsupervised in training), **external metrics** measure cluster-label alignment, accounting for:

- Arbitrary cluster numbering.
- Different cluster and class counts.
- Chance agreement.

### 10.2 Conceptual Foundations

Let $U$ = ground-truth labels, $V$ = cluster assignments. Contingency table $n_{ij}$ = # points with label $i$, cluster $j$.

**Rand Index (RI)**: probability two random points are "agreed upon" (same label and same cluster, or different in both).

**Adjusted Rand Index (ARI)**: RI corrected for chance. 0 = random; 1 = perfect; can be slightly negative.

**Mutual Information (MI)**: $I(U; V)$ — shared info.

**Normalized Mutual Information (NMI)**: MI normalized to [0, 1]. Common normalizations: arithmetic mean, geometric mean, max of entropies.

**Homogeneity**: each cluster contains only members of a single class. Homogeneity = 1 ⇒ pure clusters.

**Completeness**: all members of a class are in the same cluster. Completeness = 1 ⇒ classes not split.

**V-measure**: harmonic mean of homogeneity and completeness.

### 10.3 Mathematical Formulation

$$
RI = \frac{a + b}{\binom{n}{2}}
$$

($a$ = pairs same-class AND same-cluster; $b$ = pairs different-class AND different-cluster).

$$
ARI = \frac{\sum_{ij}\binom{n_{ij}}{2} - \frac{\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}}{\binom{n}{2}}}{\tfrac{1}{2}\left[\sum_i \binom{a_i}{2}+\sum_j \binom{b_j}{2}\right] - \frac{\sum_i \binom{a_i}{2}\sum_j \binom{b_j}{2}}{\binom{n}{2}}}
$$

where $a_i = \sum_j n_{ij}$ (row sum), $b_j = \sum_i n_{ij}$ (column sum).

$$
I(U;V) = \sum_{i,j} \frac{n_{ij}}{n}\log\frac{n \cdot n_{ij}}{a_i b_j}
$$

$$
\text{NMI}_{\text{arith}} = \frac{I(U;V)}{(H(U)+H(V))/2}
$$

$$
h = 1 - \frac{H(U|V)}{H(U)}, \quad c = 1 - \frac{H(V|U)}{H(V)}, \quad v = \frac{2hc}{h+c}
$$

### 10.4 Worked Example

Six points, true labels [A, A, A, B, B, B], clustered as [1, 1, 2, 2, 3, 3].

Contingency:

|         | C1 | C2 | C3 |
| ------- | -- | -- | -- |
| Class A | 2  | 1  | 0  |
| Class B | 0  | 1  | 2  |

Row sums: 3, 3. Col sums: 2, 2, 2.

$\binom{6}{2} = 15$. Same-class pairs: $\binom{3}{2} + \binom{3}{2} = 6$. Different-class: 9. Same-cluster pairs: $\binom{2}{2} \cdot 3 = 3$. Different-cluster: 12.

Pairs same-class AND same-cluster:

- C1: 2 A's → 1 pair.
- C2: 1A + 1B → 0.
- C3: 2 B's → 1 pair.

$a = 2$.

Pairs different-class AND different-cluster. Enumerate A–B cross-pairs:

- (A in C1) × (B in C2): 2×1 = 2 (different cluster ✓)
- (A in C1) × (B in C3): 2×2 = 4 ✓
- (A in C2) × (B in C2): 1×1 = 1 (same cluster ✗)
- (A in C2) × (B in C3): 1×2 = 2 ✓

Different-cluster different-class = 2+4+2 = 8.

$RI = (2+8)/15 \approx \mathbf{0.667}$.

**ARI.**

$\sum_{ij} \binom{n_{ij}}{2} = \binom{2}{2}+\binom{1}{2}+\binom{0}{2}+\binom{0}{2}+\binom{1}{2}+\binom{2}{2} = 1+0+0+0+0+1 = 2$
$\sum_i \binom{a_i}{2} = 3+3 = 6$, $\sum_j \binom{b_j}{2} = 1\cdot 3 = 3$.

Expected index $= 6 \cdot 3 / 15 = 1.2$.

ARI $= (2 - 1.2) / (0.5(6+3) - 1.2) = 0.8 / 3.3 \approx \mathbf{0.242}$.

Modest positive — better than random but far from perfect, reflecting the mixed C2.

**Homogeneity and completeness.** C2 is impure (1A, 1B), so homogeneity < 1. Class A is split (across C1 and C2) and class B split (across C2 and C3), so completeness < 1. Both penalties flag the split at C2.

### 10.5 Relevance to ML Practice

- Benchmarking unsupervised methods (topic models, community detection) when labels are available.
- Evaluating embeddings by clustering them and comparing to known labels.
- Assessing label discovery.

**Choosing metrics.** ARI: balanced, chance-corrected. NMI: information-theoretic; biased toward more clusters — use AMI to correct. Homogeneity / completeness / V-measure: useful for diagnosis.

### 10.6 Common Pitfalls

1. **Using plain Rand Index.** Always prefer ARI or AMI.
2. **Interpreting NMI as fair across different $K$.** NMI is biased toward higher $K$; use AMI for fair comparison.
3. **Focusing on one of homogeneity/completeness.** They're in tension — reporting one hides failure on the other.
4. **Using external metrics as training signal.** They require labels; if you have labels, use supervised learning.
5. **Ignoring cluster-size imbalance.** External metrics can mask systematic errors in minority classes.

---


# PART IV — STATISTICAL SIGNIFICANCE TESTS

## Chapter 11. Statistical Significance for Model Comparison (McNemar, Paired t-test, Wilcoxon Signed-Rank)

### 11.1 Motivation & Intuition

Model A scores F1 = 0.82. Model B scores 0.84. Is B genuinely better, or is this noise?

**Statistical significance tests** quantify the probability that an observed difference arose by chance if the two models were truly equivalent. Without them, every incremental improvement risks being noise, and research literature accumulates false positives.

**Why *paired* tests?** Both models see the *same* test examples. Example-level difficulty is a shared nuisance: subtracting within-example cancels out "this example is easy" variance and leaves only model-difference variance. Paired tests are far more powerful than unpaired ones here.

**Real scenario.** You train a new recommender. You run A/B tests or holdout comparisons. Before declaring victory, test: is the improvement statistically significant and practically meaningful?

### 11.2 Conceptual Foundations

Three commonly used paired tests:

1. **McNemar's test** — compares two classifiers on the *same* dataset using their binary correct/incorrect per example. Focuses on the contingency of *disagreements*.
2. **Paired t-test** — for paired continuous observations (e.g., per-fold accuracies in cross-validation). Assumes the paired differences are approximately normal.
3. **Wilcoxon signed-rank test** — non-parametric version of the paired t-test; uses ranks of absolute differences, drops the normality assumption.

**Null hypothesis conventions.**

- McNemar: $H_0$: the two models have equal error rate, i.e., $P(\text{only A wrong}) = P(\text{only B wrong})$.
- Paired t: $H_0$: mean of paired differences is zero.
- Wilcoxon: $H_0$: distribution of paired differences is symmetric about zero.

**p-value.** Probability of observing a test statistic at least as extreme as the one computed, under $H_0$. A small p-value (typically < 0.05) rejects $H_0$.

**Effect size matters.** "Significant" ≠ "important." On 10⁸ examples any trivial difference is significant. Always report effect size (odds ratio, mean difference, Cohen's $d$, etc.).

**Multiple comparisons.** Running $m$ tests at $\alpha = 0.05$ gives expected false positives $\approx 0.05m$. Correct with Bonferroni ($\alpha/m$), Holm, or Benjamini-Hochberg (for FDR control).

**Assumptions at a glance.**

| Test         | Data type                | Key assumptions                          |
| ------------ | ------------------------ | ---------------------------------------- |
| McNemar      | Paired binary            | Exchangeability of discordant pairs      |
| Paired t     | Paired continuous        | Differences roughly normal (or large $n$) |
| Wilcoxon SR  | Paired continuous/ordinal| Differences are symmetric about median   |

**What breaks.**

- McNemar with few discordant pairs ($b + c < 25$): normal approximation fails — use exact binomial form.
- Paired t with heavy-tailed differences and small $n$: Type I error inflated.
- Wilcoxon assumes symmetry; if violated, consider sign test.
- All tests assume independence across pairs. Overlapping CV folds violate this — inflating Type I error (Dietterich 1998; Nadeau & Bengio 2003).

### 11.3 Mathematical Formulation

#### McNemar's test

For two classifiers on same test set, build the 2×2 disagreement table:

|                      | B correct | B wrong |
| -------------------- | --------- | ------- |
| **A correct**        | $a$       | $b$     |
| **A wrong**          | $c$       | $d$     |

Only $b$ and $c$ (discordant pairs) matter — the cells $a$ and $d$ are where both agree, carrying no information about which model is better.

**Intuition.** Under $H_0$, a discordant pair is equally likely to be (A correct, B wrong) or (A wrong, B correct). So $b$ follows $\text{Binomial}(b+c, 0.5)$.

**Test statistic** (continuity-corrected normal approximation, valid when $b+c \geq 25$):

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1
$$

For small $b + c$, use the **exact** two-tailed binomial:

$$
p = 2 \cdot P\bigl(X \leq \min(b, c) \mid X \sim \text{Bin}(b+c, 0.5)\bigr)
$$

#### Paired t-test

$n$ pairs $(x_i, y_i)$, differences $d_i = x_i - y_i$, with mean $\bar{d}$ and sample SD $s_d$.

$$
t = \frac{\bar{d}}{s_d / \sqrt{n}} \sim t_{n-1} \text{ under } H_0
$$

Two-sided p-value: $p = 2(1 - F_{t_{n-1}}(|t|))$.

Cohen's $d$ for paired data: $d = \bar{d}/s_d$.

#### Wilcoxon signed-rank

1. Compute $d_i$; discard zeros.
2. Rank $|d_i|$ (average ranks for ties).
3. $W^+ = \sum_{d_i > 0} \text{rank}(|d_i|)$, $W^- = \sum_{d_i < 0} \text{rank}(|d_i|)$.
4. Test statistic $W = \min(W^+, W^-)$.
5. For $n > 20$, approximate:

$$
z = \frac{W^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}
$$

For small $n$, use exact tables.

### 11.4 Worked Examples

#### Example A — McNemar

Models A and B on 500 test examples:

- Both correct: 400
- Only A correct: 50 ($b$)
- Only B correct: 30 ($c$)
- Both wrong: 20

Discordant = 80 (≥ 25), continuity-corrected χ²:

$$
\chi^2 = \frac{(|50-30|-1)^2}{80} = \frac{19^2}{80} = \frac{361}{80} \approx 4.51
$$

$p \approx 0.034 < 0.05$ → reject $H_0$; A significantly outperforms B.

Effect size (odds ratio): $b/c = 50/30 \approx 1.67$. "When they disagree, A is right 1.67× more often than B."

#### Example B — Paired t-test on CV folds

10-fold CV accuracies:

| Fold | A    | B    | d = A − B |
| ---- | ---- | ---- | --------- |
| 1    | 0.82 | 0.80 | 0.02      |
| 2    | 0.85 | 0.84 | 0.01      |
| 3    | 0.78 | 0.77 | 0.01      |
| 4    | 0.83 | 0.79 | 0.04      |
| 5    | 0.81 | 0.80 | 0.01      |
| 6    | 0.86 | 0.82 | 0.04      |
| 7    | 0.79 | 0.78 | 0.01      |
| 8    | 0.84 | 0.81 | 0.03      |
| 9    | 0.82 | 0.80 | 0.02      |
| 10   | 0.85 | 0.83 | 0.02      |

$\bar{d} = 0.021$. Computing $s_d$: deviations from mean squared sum ≈ 0.001290; $s_d \approx \sqrt{0.001290/9} \approx 0.01197$.

$$
t = \frac{0.021}{0.01197/\sqrt{10}} = \frac{0.021}{0.003787} \approx 5.55
$$

With 9 df, $p \approx 0.0004$ → reject $H_0$; A significantly better.

Cohen's $d = 0.021 / 0.01197 \approx 1.75$ — very large effect.

**Caveat.** Standard CV paired t-test has inflated Type I error because CV folds share training data. Nadeau & Bengio's corrected t-test adjusts variance:

$$
\sigma^2_{\text{corrected}} = \sigma^2 \left(\frac{1}{n} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)
$$

With this correction, $t$ shrinks; $p$ may no longer pass 0.05.

#### Example C — Wilcoxon signed-rank

Same 10 differences: {0.02, 0.01, 0.01, 0.04, 0.01, 0.04, 0.01, 0.03, 0.02, 0.02}. No zeros; all positive.

Sorted absolute values with tied-rank averaging:

- Four 0.01's at positions 1–4 → avg rank = 2.5
- Three 0.02's at positions 5–7 → avg rank = 6
- One 0.03 at position 8 → rank 8
- Two 0.04's at positions 9–10 → avg rank = 9.5

$W^+ = 4(2.5) + 3(6) + 1(8) + 2(9.5) = 10 + 18 + 8 + 19 = \mathbf{55}$

$W^- = 0$. For $n = 10$, the critical value at $\alpha = 0.05$ two-sided is $W \leq 8$. Since $\min(W^+, W^-) = 0 \leq 8$, reject $H_0$; A significantly better.

All three tests agree. But the honest reading is: with 10 shared-data folds and identical-sign differences, we have strong *within-this-experiment* evidence. Generalization to a new dataset is a stronger claim requiring replication.

### 11.5 Relevance to ML Practice

- **Model selection & paper benchmarks**: don't declare a method better than baselines on a single run; use paired tests across seeds or folds plus CIs.
- **A/B testing**: use appropriate tests (often two-proportion z-test for conversion, Mann–Whitney for revenue).
- **Regulatory / medical deployments**: significance is typically required.
- **Leaderboards**: tiny F1 differences rarely significant.

**Choosing the right test.**

- Binary per-example correctness → **McNemar**.
- Continuous per-fold scores, approximately normal differences → **paired t-test** (use the Nadeau–Bengio variance correction for CV).
- Continuous per-fold scores, non-normal or ordinal differences → **Wilcoxon signed-rank**.

**Trade-offs.**

- Paired t is most powerful when assumptions hold. Wilcoxon is ~95% as powerful under normality and much better under heavy tails. McNemar doesn't generalize beyond paired binary, but is exact and assumption-light within that setting.
- Bootstrap confidence intervals for the metric difference are often more informative than p-values.

### 11.6 Common Pitfalls

1. **Using unpaired tests on paired data.** Loses massive power; prefer paired tests whenever examples are shared.
2. **Ignoring CV dependence.** Standard paired t-test on CV fold accuracies has inflated Type I error — use corrected variance.
3. **Confusing significance with importance.** Huge $n$ makes trivial differences significant; report effect size and practical impact.
4. **Not correcting for multiple comparisons.** Bonferroni, Holm, or BH for many tests.
5. **Reporting only one point estimate.** Always report CI (bootstrap is easy and assumption-light) to convey uncertainty.
6. **Applying the normal-approximation form of McNemar with few discordants.** Use exact binomial when $b + c < 25$.
7. **Double dipping.** Selecting the best of many variants *and* testing on the same data inflates Type I error — use a fresh holdout.

---

# PART V — INTERVIEW PREPARATION

This section contains an extensive question bank organized by topic and difficulty. Answers are written at the depth expected for ML / applied-scientist interviews.

## 12. Foundational Questions

### Classification basics

**Q1. Define precision, recall, F1, Fβ. When would you use each?**

**A.** Precision $= TP/(TP+FP)$ answers "of predicted positives, how many are truly positive?" Recall $= TP/(TP+FN)$ answers "of actual positives, how many did we catch?" F1 $= 2PR/(P+R)$ is the harmonic mean, a single balanced metric. $F_\beta = (1+\beta^2)PR/(\beta^2 P + R)$ weights recall $\beta^2$ times as much as precision.

Use precision when false positives are costly (ad spend, drug approval). Use recall when false negatives are costly (disease screening, fraud). F1 when both matter equally. $F_\beta$ when you know the cost ratio. The harmonic mean is chosen because it's small whenever either operand is small — punishing imbalance.

**Q2. Why is accuracy misleading for imbalanced classes?**

**A.** A trivial classifier that always predicts the majority class achieves high accuracy when the majority dominates. With 99% negatives, accuracy ≥ 0.99 tells us almost nothing about a model's skill; the model may never correctly identify any positive. Metrics that directly expose minority-class behavior (precision, recall, F1, PR-AUC) are needed.

**Q3. What is the ROC curve, and why is AUC threshold-free?**

**A.** The ROC plots TPR vs FPR as the decision threshold sweeps from $+\infty$ to $-\infty$. Each threshold produces one (FPR, TPR) point. AUC is the area under this curve — the *probability* that a random positive outranks a random negative. Since it's computed over all thresholds, it measures inherent ranking ability rather than a single-threshold performance.

**Q4. What is calibration and why might a model be accurate but poorly calibrated?**

**A.** Calibration means predicted probabilities match empirical frequencies: among examples predicted 0.8, about 80% are positive. Accuracy (under argmax) and calibration are separate: a ResNet might score 95% accuracy but output 0.99 for nearly every correct prediction — overconfident and so miscalibrated, even though top-1 is right. Calibration matters when probabilities drive downstream decisions (expected cost, risk thresholds, ensembling).

### Regression basics

**Q5. When would you use MAE over MSE?**

**A.** Use MAE when (a) outliers are present and you don't want them to dominate the fit, (b) you care about median rather than mean predictions, (c) you want errors reported in original units without the squaring-scale inflation. MSE inflates large errors quadratically; a few severe outliers can dominate the total. MAE's flat gradient away from zero makes it more stable under heavy-tailed noise but harder to optimize (non-smooth at zero).

**Q6. What does R² measure? Can it be negative?**

**A.** R² = 1 − SS_res/SS_tot. It measures the proportion of variance in $y$ explained by the model relative to the mean baseline. R² = 1 is perfect; 0 means equivalent to predicting $\bar{y}$. On a held-out set a model *can* be worse than the mean baseline, giving **negative R²**. It is bounded in [0, 1] only for OLS with intercept evaluated on training data.

### Clustering basics

**Q7. How do internal and external clustering metrics differ?**

**A.** Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) use only the data and cluster assignments to measure cohesion and separation. External metrics (ARI, NMI, homogeneity, completeness) require ground-truth labels to quantify alignment. Internal metrics are usable during unsupervised model selection; external metrics are for benchmark evaluation when labels happen to be available for diagnosis — not for training.

**Q8. Why would silhouette be misleading for DBSCAN?**

**A.** Silhouette assumes roughly convex, well-separated clusters evaluated via mean distance. DBSCAN returns arbitrarily shaped density clusters and a "noise" label. Silhouette for a banana-shaped cluster surrounded by another banana-shaped cluster can be low even when the density clustering is correct. Moreover, silhouette for noise points isn't well-defined. Use density-based validity indices (DBCV) instead.

### Significance

**Q9. What is McNemar's test, and when do you use it?**

**A.** McNemar's test compares two classifiers on the *same* test set with binary (correct/incorrect) outcomes per example. It focuses only on discordant pairs — examples where the two models disagree — because concordant pairs are uninformative about which model is better. Under $H_0$, the split of discordants is Binomial(0.5). Use it when you have one test set and two classifiers and want to know if accuracy difference is significant. It's preferred over a two-proportion test because it uses pairing to cancel per-example difficulty.

**Q10. When would you use Wilcoxon signed-rank vs paired t-test?**

**A.** Wilcoxon is a non-parametric alternative. Use it when the paired differences are non-normal, ordinal, or contain outliers. The paired t-test assumes differences are approximately normal (less critical with large $n$ due to CLT) and is most powerful under that assumption. For CV-based comparisons with small $k$ (say 10 folds), Wilcoxon is often the safer choice.

---

## 13. Mathematical Questions

**Q11. Derive the F1 score from the harmonic mean and explain why the harmonic mean is appropriate.**

**A.** The harmonic mean of $P$ and $R$ is $H = 2 / (1/P + 1/R) = 2PR/(P+R)$. Three properties:

1. **Bounded by the smaller operand.** $H \leq \min(P, R)$. So F1 can be high only if both are high.
2. **Punishes imbalance.** Arithmetic mean gives 0.5 when $(P, R) = (1, 0)$ — clearly wrong. Harmonic mean is 0.
3. **Rate interpretation.** Precision and recall are rates (successes per attempt); harmonic mean is the natural aggregator of rates.

**Q12. Derive the probabilistic interpretation of AUC.**

**A.** Let $S_+ = s(X^+)$ and $S_- = s(X^-)$ with $X^+$, $X^-$ drawn independently from positive/negative distributions. We show $\text{AUC} = P(S_+ > S_-) + \tfrac{1}{2}P(S_+ = S_-)$.

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(u))\,du
$$

Changing variables to $\tau$ (the threshold) with $u = \text{FPR}(\tau)$:

$$
\text{AUC} = -\int_{-\infty}^{\infty} \text{TPR}(\tau) \, d\text{FPR}(\tau) = \int_{-\infty}^{\infty} P(S_+ \geq \tau) f_-(\tau) d\tau
$$

where $f_-$ is the density of $S_-$. This is $E_{S_-}[P(S_+ \geq S_-)] = P(S_+ \geq S_-)$, which splits into the strict-inequality probability plus half the equality probability (by convention).

**Q13. Derive the optimal Bayes threshold for cost-sensitive classification.**

**A.** Let $c_{FP}$ be the cost of a false positive, $c_{FN}$ of a false negative. Expected cost of predicting positive for $x$: $c_{FP} \cdot P(Y=0|x)$. Expected cost of predicting negative: $c_{FN} \cdot P(Y=1|x)$.

Predict positive iff $c_{FP} P(Y=0|x) < c_{FN} P(Y=1|x)$. Setting $p = P(Y=1|x)$:

$$
c_{FP}(1-p) < c_{FN} p \Rightarrow p > \frac{c_{FP}}{c_{FP}+c_{FN}} =: \tau^*.
$$

So optimal threshold $\tau^* = c_{FP}/(c_{FP}+c_{FN})$. Under unequal priors $(\pi_0, \pi_1)$ on a balanced-trained model, the effective threshold is $\tau^* = c_{FP}\pi_0/(c_{FP}\pi_0 + c_{FN}\pi_1)$. **Only valid if the classifier is calibrated** — otherwise probabilities don't correspond to true conditional probabilities and the derivation breaks.

**Q14. Why does MSE minimization yield the conditional mean, and MAE the conditional median?**

**A.** For MSE, differentiate $E[(Y - \hat{y})^2]$ with respect to $\hat{y}$:

$$
\frac{d}{d\hat{y}} E[(Y-\hat{y})^2] = -2(E[Y] - \hat{y}) = 0 \Rightarrow \hat{y} = E[Y].
$$

Conditioning on $X$: minimizer of $E[(Y-\hat{y})^2 | X]$ is $E[Y|X]$.

For MAE, $E[|Y - \hat{y}|] = \int_{-\infty}^{\hat{y}}(\hat{y}-y)f(y)dy + \int_{\hat{y}}^\infty (y-\hat{y})f(y)dy$. Differentiate:

$$
F(\hat{y}) - (1 - F(\hat{y})) = 0 \Rightarrow F(\hat{y}) = 1/2,
$$

i.e., $\hat{y}$ is the median. Extends conditionally.

**Q15. Derive $R^2 = \text{Corr}(y, \hat{y})^2$ for OLS.**

**A.** For OLS with intercept, $\hat{y} - \bar{y}$ is orthogonal to $y - \hat{y}$, and $\overline{\hat{y}} = \bar{y}$. Then

$$
SS_{\text{tot}} = \sum (y_i - \bar{y})^2 = \sum (y_i - \hat{y}_i + \hat{y}_i - \bar{y})^2 = SS_{\text{res}} + SS_{\text{reg}}
$$

where $SS_{\text{reg}} = \sum (\hat{y}_i - \bar{y})^2$. By orthogonality,

$$
\text{Cov}(y, \hat{y}) = \text{Cov}(\hat{y}, \hat{y}) = \text{Var}(\hat{y})
$$

So

$$
\text{Corr}(y, \hat{y})^2 = \frac{\text{Cov}(y, \hat{y})^2}{\text{Var}(y)\text{Var}(\hat{y})} = \frac{\text{Var}(\hat{y})}{\text{Var}(y)} = \frac{SS_{\text{reg}}}{SS_{\text{tot}}} = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = R^2.
$$

For general non-linear $\hat{y}$, this identity fails because orthogonality may not hold.

**Q16. Derive the McNemar test statistic from a binomial argument.**

**A.** Let $b, c$ denote the counts where exactly one model is correct. Under $H_0$ (equal error rates), given $b + c$ discordants, each is independently assigned to A-wins or B-wins with probability 0.5. So $b \sim \text{Binomial}(b+c, 0.5)$.

Expected $b$ = $(b+c)/2$, variance = $(b+c)/4$.

Normal approximation: $z = (b - (b+c)/2) / \sqrt{(b+c)/4} = (b - c)/\sqrt{b+c}$.

Squaring: $z^2 = (b-c)^2/(b+c) \sim \chi^2_1$. Continuity correction gives $(|b-c|-1)^2/(b+c)$.

For small $b+c$, normal approximation is poor — use exact binomial two-tailed p-value: $p = 2 \cdot P(X \leq \min(b,c) \mid X \sim \text{Bin}(b+c, 0.5))$.

**Q17. Show that log-loss punishes overconfidence more than Brier.**

**A.** Log-loss for a single example: $\ell = -[y \log p + (1-y)\log(1-p)]$. As $p \to 0$ for $y=1$, $\ell \to \infty$. Brier: $(p-y)^2$, bounded by 1.

The gradient with respect to $p$ for a confidently wrong example: log-loss $\partial \ell/\partial p = -(y-p)/(p(1-p)) \to \infty$ as $p \to 0$ (for $y=1$). Brier $\partial/\partial p = 2(p - y)$, bounded.

Consequence: log-loss training creates strong pressure away from high-confidence errors. Brier is gentler. This is why log-loss-trained nets often exhibit overconfidence (they pushed correct labels to extremes, but failed on some rare wrong ones), while Brier-scored models are gentler.

**Q18. Derive the Wilcoxon signed-rank mean and variance under H₀.**

**A.** Under $H_0$ (differences symmetric about zero), each rank $1, 2, \ldots, n$ is assigned to a positive difference with probability 0.5 independently. Let $W^+ = \sum_i R_i$ where $R_i$ is the rank if the $i$-th ordered |difference| has positive sign, 0 otherwise.

$$
E[W^+] = \sum_{i=1}^n i \cdot 0.5 = \frac{n(n+1)}{4}
$$

$$
\text{Var}(W^+) = \sum_{i=1}^n i^2 \cdot 0.25 = \frac{n(n+1)(2n+1)}{24}
$$

Normal approximation: $z = (W^+ - n(n+1)/4)/\sqrt{n(n+1)(2n+1)/24}$. Ties require variance correction.

---

## 14. Applied Questions

**Q19. You're designing a fraud detection system. Which metric do you optimize and how do you choose the threshold?**

**A.** Fraud is heavily imbalanced (say 0.1% positive), making accuracy and ROC-AUC misleading. Start with **PR-AUC** for overall model assessment — it focuses on the minority class and is sensitive to ranking of positives against negatives.

For deployment threshold, model the business cost: each missed fraud costs the expected loss (say $500 avg), each false alarm costs review time (say $5) plus customer friction. With calibrated probabilities, the optimal threshold is $\tau^* = c_{FP}/(c_{FP}+c_{FN}) = 5/505 ≈ 0.01$. Validate by sweeping thresholds on held-out data and picking the one that minimizes total expected cost subject to operational constraints (analyst team capacity caps max FPs/day).

Also monitor: precision at the chosen threshold (so analysts don't drown in false positives), recall (coverage of actual fraud), and drift (PR-AUC over rolling windows).

**Q20. Your recommender's NDCG improved by 0.01 offline but CTR dropped in A/B test. What happened and what do you do?**

**A.** Several plausible causes:

1. **Offline-online gap**: NDCG was computed on biased logged data where positions were determined by old policy. The new model optimizes for *graded relevance* but CTR depends on *display order under exposure bias*. Without counterfactual or IPS-weighted evaluation, offline gains can be illusory.
2. **Metric mismatch**: NDCG uses stored relevance grades; CTR depends on live user attention, novelty, diversity — often anti-correlated with relevance.
3. **Diversity/freshness**: the new model may be less diverse, surfacing items similar to users' past behavior, decreasing click rate.
4. **Presentation bias & position bias**: models can game NDCG by favoring items with historically higher positions.

**Actions.** Instrument with IPS-corrected offline metrics. Add diversity (intra-list diversity), freshness, and serendipity metrics. Run interleaved comparisons (TDI, team-draft interleaving) which are more sensitive than A/B. Confirm with longer-horizon metrics (session length, retention) since CTR ≠ user value.

**Q21. You deploy a classifier with 95% validation AUC; after 3 months AUC drops to 0.80. Debug.**

**A.** Systematic checklist:

1. **Label drift / concept drift.** The mapping $P(Y|X)$ changed. Compare old vs new label distributions; if possible, re-label recent data.
2. **Covariate shift.** $P(X)$ changed; some input features have different distributions. Use population stability index (PSI), KL divergence, or domain classifier (a model trained to distinguish old vs new data — high accuracy implies shift).
3. **Data pipeline bugs.** Feature engineering breaks, unit changes, missing-value imputation differences. Inspect feature distributions and null rates.
4. **Feedback loop.** Model decisions affect future data (e.g., fraud model blocks transactions we never see), biasing the incoming distribution.
5. **Leakage that degraded.** A feature that was a proxy in training no longer proxies at serving.
6. **Seasonality.** Some classes appear seasonally.

**Fix-in-place mitigations.** Retrain on recent data; use adaptive thresholds based on recent base rates; use online learning if signal allows; add drift alarms. Consider calibration correction if ranking is preserved but probabilities shifted.

**Q22. Your regression model has RMSE = 10, MAE = 4. What does this tell you?**

**A.** If errors were Gaussian, we'd expect $\text{RMSE}/\text{MAE} \approx 1.25$. Here the ratio is 2.5, much higher. This signals **heavy-tailed errors / outliers** — a small set of very large residuals inflates MSE but contributes linearly to MAE.

Next steps:

1. Plot residuals vs $\hat{y}$ and vs features to find structure.
2. Look at the tail of the residual distribution; identify worst-predicted examples.
3. Check for data issues (bad labels, corrupted features).
4. If outliers are legitimate but rare, consider Huber loss in training, robust models (quantile forests), or a two-stage system (separate "unusual case" classifier and regressor).
5. If business penalizes large errors nonlinearly, keep MSE; otherwise prefer MAE-based evaluation.

**Q23. When would you choose macro-F1 over weighted-F1?**

**A.** Choose macro-F1 when you care about performance on *each class equally*, regardless of frequency. E.g., medical classification where rare diseases matter as much as common ones. Weighted-F1 mirrors the class distribution — good when "average test example" is what you care about, which is common in typical product metrics but often hides minority-class failure.

Report **both**: discrepancies reveal imbalance-sensitive behavior. If macro-F1 ≪ weighted-F1, some classes are failing; investigate per-class metrics.

**Q24. How do you evaluate a clustering model in production when ground truth is unavailable?**

**A.** Combine multiple signals:

1. **Internal metrics**: silhouette, Davies-Bouldin, Calinski-Harabasz — limited but cheap.
2. **Stability**: run on multiple bootstrapped subsamples or with perturbed hyperparameters; measure via ARI across runs. Stable clusterings are more trustworthy.
3. **Downstream utility**: if clusters feed a recommender or segmentation, measure downstream task lift.
4. **Human evaluation**: sample points per cluster; have annotators rate purity or coherence.
5. **Business KPIs**: does the clustering segment correlate with meaningful outcomes (LTV, retention)?

Avoid comparing internal metrics across algorithms with different geometric assumptions (e.g., silhouette against DBSCAN). Track cluster stability over time to catch drift.

**Q25. You have 10,000 training and 100 test examples. How confident can you be in your test AUC?**

**A.** Very limited. With 100 examples and, say, 10 positives, AUC's standard error is roughly $\sqrt{\text{AUC}(1-\text{AUC})/\min(n_+, n_-)}$ (Hanley–McNeil approximation), yielding SE ~0.1–0.15. A bootstrap 95% CI on AUC might span ±0.10.

Remedies: bootstrap CIs on test AUC; cross-validated AUC (pool out-of-fold predictions); move examples from training to test if the training set is abundant and overparameterized. Alternatively, use paired tests across repeated CV to compare models instead of single-split evaluations.

**Q26. How do you test whether model A is significantly better than B using 5×2 CV?**

**A.** Dietterich's 5×2 CV paired t-test. Run 2-fold CV five times with different random splits. For each replication, compute differences in fold errors $d_1^{(r)}, d_2^{(r)}$ and variance $s_r^2 = (d_1^{(r)} - \bar{d}^{(r)})^2 + (d_2^{(r)} - \bar{d}^{(r)})^2$.

$$
t = \frac{d_1^{(1)}}{\sqrt{\frac{1}{5}\sum_{r=1}^5 s_r^2}} \sim t_5
$$

This controls Type I error better than 10-fold CV with naive paired t because 5×2 CV has less data overlap between folds. For binary correctness at the example level, prefer McNemar on a fixed test set.

---

## 15. Debugging & Failure-Mode Questions

**Q27. A classifier achieves ROC-AUC 0.99 but F1 of 0.2 at default threshold. Explain.**

**A.** Extreme class imbalance. With, say, 99.9% negative, the model ranks well (high AUC) but the default threshold of 0.5 may be far above optimal because the typical positive-class probability is low. Solutions: lower the threshold (target precision/recall), recalibrate (Platt/isotonic), or train with class weights / focal loss. Also: look at PR-AUC, which would expose the issue directly (much lower than ROC-AUC).

**Q28. Validation accuracy 0.95, test accuracy 0.70. What went wrong?**

**A.** Most likely cause is **data leakage** or **train-val similarity not matching train-test similarity**. Possibilities:

1. Val was drawn from same data source as train with overlap (duplicates, near-duplicates, temporal leakage).
2. Temporal drift: val is from same period as train, test from later period.
3. Group leakage: same user/patient/device appeared in both train and val.
4. Hyperparameters tuned on val overfit to val's quirks.
5. Target leakage: a feature includes future information.

Fixes: rebuild splits with strict grouping/time-based validation; audit features for leakage; use nested CV if tuning with small data. Do *not* look at test again until fixes are validated on a fresh validation set.

**Q29. Silhouette score says K=2 is best; but domain knowledge says 5 clusters. What to do?**

**A.** Trust domain knowledge over internal metrics. Silhouette often prefers K=2 because it rewards the cleanest split, which is usually a coarse global division (e.g., male/female, urban/rural). Look at:

- Domain-meaningful within-cluster analysis at K=5 (do clusters correspond to interpretable personas?).
- Hierarchical clustering: K=2 might be the top split; K=5 a refinement. Report a dendrogram.
- Stability across subsamples at K=5.
- External tasks that the clusters feed — does K=5 improve downstream performance?

Internal metrics are heuristics, not truth.

**Q30. Why did my PR-AUC drop when I added a "correct" new feature?**

**A.** Candidates:

1. **Feature is leaky at train time but not test time.** Training-only shortcut raises CV-PR-AUC but collapses on held-out.
2. **Feature has high variance and you didn't regularize.** New dimension is noisy.
3. **Feature correlates with an existing label-like shortcut you were unknowingly using; adding the new feature actually breaks the shortcut.**
4. **Imbalanced update of calibration.** If PR-AUC was computed on changed predictions with different score distribution, differences can appear even though ranking is similar. Check rank correlations between old and new predictions.
5. **Evaluation noise.** With few positives, PR-AUC has wide CIs; a small drop may be within noise. Bootstrap.

**Q31. Your calibration is bad on subgroups but fine overall. How did this happen, how fix?**

**A.** Averaging across subgroups can cancel out opposing miscalibrations. Subgroup A over-predicts, subgroup B under-predicts, and overall gap averages to near zero.

Fixes:

- Compute per-subgroup ECE and reliability diagrams.
- Fit group-conditional recalibrators (e.g., per-subgroup Platt or isotonic). Beware small subgroups — use shrinkage toward overall calibrator.
- Training-time adjustments: include group as a feature; re-weight under-represented groups.

Fairness interaction: poor subgroup calibration can violate group fairness metrics even if overall metrics are fine. For regulated contexts, report both aggregate and per-subgroup.

**Q32. Your A/B test is "significant" (p < 0.001) but business impact is negligible. What went wrong?**

**A.** **Significance ≠ importance.** With millions of observations, tiny effects become detectable but may be practically meaningless. Report effect size (absolute and relative), confidence interval, and compare to minimum meaningful effect defined *pre-experiment*. Also watch for SRM (sample ratio mismatch), novelty effects, and Simpson's paradox across segments.

Solution: define a minimum detectable effect (MDE) and a minimum meaningful effect (MME) in advance. Power the experiment for MME; don't celebrate effects below it.

**Q33. Your model's log-loss decreased but Brier score didn't. Explain.**

**A.** Log-loss is dominated by *confidently wrong* predictions (its tails are unbounded). If your update reduced a few extreme errors (e.g., $p = 0.999$ for a true 0 example → $p = 0.99$), log-loss drops dramatically. Brier is bounded and more uniform, so the same change causes only modest improvement. This can indicate that you fixed extreme overconfidence but didn't improve overall calibration or accuracy broadly.

---

## 16. Probing Follow-Up Questions

**Q34. What's the difference between micro-F1 and accuracy in single-label multiclass?**

**A.** They are identical. Proof: For each mistake, the predicted class's FP count goes up by 1 and the true class's FN goes up by 1. So $\sum_k FP_k = \sum_k FN_k = n - \text{correct}$. Micro-precision = $TP / (TP + FP) = \text{correct}/n$ = accuracy, and similarly micro-recall. They differ only in multi-label settings.

**Q35. Why is ROC-AUC insensitive to class imbalance while PR-AUC isn't?**

**A.** ROC uses FPR = FP/(FP+TN). When $|\text{neg}|$ is huge, TN dominates and small changes in FP barely budge FPR, so AUC looks high. PR uses Precision = TP/(TP+FP), which responds directly to FP without smoothing by TN. So PR-AUC surfaces the "you have to trade off many false alarms for each true positive" reality.

Put formally: ROC doesn't involve the class prior explicitly, so it's invariant to subsampling negatives; PR depends on prior and reflects the actual working-point trade-off at deployment.

**Q36. What's the difference between bagging-based AUC uncertainty and DeLong's method?**

**A.** DeLong (1988) is a closed-form method exploiting AUC's U-statistic structure to compute variance and covariance analytically for comparing AUCs. It's exact under asymptotic conditions and handles paired classifiers elegantly. Bootstrap is simulation-based, assumption-light, and works for any metric but is computationally heavier. For AUC comparisons specifically, DeLong is standard; for arbitrary metrics (F1, PR-AUC), bootstrap is the go-to.

**Q37. Why does temperature scaling preserve accuracy while improving calibration?**

**A.** Softmax with temperature: $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$. Temperature $T$ divides all logits by a constant — monotonic transformation per-example. The argmax over logits is unchanged for any $T > 0$, so top-1 accuracy is preserved. But the relative differences between logits are rescaled: $T > 1$ flattens the distribution (less confident), $T < 1$ sharpens. Fit $T$ on validation to minimize NLL → better calibration without changing the predicted class.

**Q38. In what sense is NDCG gain-formula dependent?**

**A.** Two common gain functions: linear ($\text{rel}_k$) and exponential ($2^{\text{rel}_k}-1$). Exponential gain amplifies the importance of high grades — a grade-3 item contributes $2^3-1=7$ vs grade-2's $2^2-1=3$ (ratio 2.33), compared with linear's 3 vs 2 (ratio 1.5). Different gain choices produce different NDCG values and sometimes different rankings of methods. Always state which you use. Linear NDCG is older (Järvelin & Kekäläinen); exponential is more common today (used by most major search engines).

**Q39. Why can adjusted R² decrease when you add a feature while plain R² cannot?**

**A.** Plain R² = $1 - SS_{\text{res}}/SS_{\text{tot}}$. Adding a feature can only reduce $SS_{\text{res}}$ (or keep it equal in the degenerate case) because optimizing over more parameters can't make the fit worse. So $R^2$ monotonically non-decreases.

Adjusted R² $= 1 - (1-R^2)\frac{n-1}{n-k-1}$. As $k$ grows, denominator $n-k-1$ shrinks, inflating $(1-R^2)$'s weight. If adding a feature barely reduces $R^2$ error, the degrees-of-freedom penalty dominates, so adjusted R² drops.

**Q40. Why do CV folds inflate Type I error in paired t-tests and how is Nadeau-Bengio corrected?**

**A.** Standard paired t assumes independent pairs. In $k$-fold CV, each pair of fold accuracies shares training data (folds $i$ and $j$ share roughly $(k-2)/k$ of training examples). This positive correlation inflates the numerator's variability relative to the denominator's naive estimate — the statistic's tails are heavier than $t_{k-1}$ suggests.

Nadeau & Bengio (2003) proposed the corrected variance:

$$
\sigma^2_{\text{corr}} = \left(\frac{1}{k} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \sigma^2_{\text{naive}}
$$

This inflates the denominator to account for shared training data, shrinking the t statistic. Alternative: Dietterich's 5×2 CV test.

**Q41. What's the Bonferroni correction and when is it too conservative?**

**A.** With $m$ tests and family-wise error rate target $\alpha$, Bonferroni tests each at level $\alpha/m$. Guarantees FWER ≤ $\alpha$. Too conservative when tests are correlated (many tests on related hypotheses) — you lose power.

Alternatives: Holm (step-down, uniformly more powerful than Bonferroni, still controls FWER); Benjamini-Hochberg (controls false discovery rate — fraction of rejections that are false positives — much more powerful, appropriate when you tolerate some false discoveries).

Use FDR when doing many exploratory tests (feature screening, A/B portfolio). Use FWER for high-stakes claims (clinical, regulatory).

**Q42. Bootstrap CI vs analytical SE for AUC — what are the pros and cons?**

**A.** Bootstrap: resamples with replacement; percentile or BCa CIs. Pros: assumption-light, handles small samples and arbitrary metrics, captures non-normality. Cons: computationally expensive, can fail on metrics that aren't smooth functionals (e.g., rank-based with heavy ties), gives only marginal CIs (not pairwise).

Analytical (Hanley-McNeil / DeLong for AUC): fast, gives pairwise covariances for model comparison. Cons: assumes U-statistic asymptotics, relies on normality, can misbehave for very small or extreme AUCs.

In practice: DeLong for AUC model comparison; bootstrap for arbitrary metrics; both for belt-and-braces reporting.

**Q43. How would you design a metric for a retrieval system that balances relevance and diversity?**

**A.** Start from NDCG (or MAP) for relevance. Add diversity via:

1. **Intra-list diversity**: mean pairwise item distance in the top-K.
2. **α-nDCG**: discounts gain for documents covering topics already covered earlier in the ranking — explicitly penalizes redundancy.
3. **Subtopic recall**: what fraction of query intents (subtopics) are represented in top-K.
4. **ERR-IA**: intent-aware expected reciprocal rank.

Combine via scalarization: $\lambda \cdot \text{NDCG} + (1-\lambda) \cdot \text{diversity}$, or Pareto frontier. Validate with user studies since the "right" trade-off is application-specific.

**Q44. Explain the variance-bias decomposition of MSE and its interpretation.**

**A.** For a model $\hat{f}(x)$ trained on random data $D$, at test point $x_0$:

$$
E_D[(Y_0 - \hat{f}(x_0))^2] = \underbrace{(E_D[\hat{f}(x_0)] - f(x_0))^2}_{\text{Bias}^2} + \underbrace{\text{Var}_D(\hat{f}(x_0))}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Irreducible noise}}
$$

Bias: systematic error of an "average" model of this family. Variance: sensitivity to training data. Noise: residual unexplained variance. Interpretation: total MSE splits into what family can't represent (bias), what the fitting process randomly adds (variance), and what can't be predicted at all (noise). Ensembles reduce variance without much change to bias; model capacity increases reduce bias but raise variance (classically — the double descent phenomenon adds nuance).

**Q45. When would you prefer Huber over quantile regression?**

**A.** Huber when you want a **single point prediction** that is robust to outliers — it still targets a conditional-mean-like statistic (more precisely, a trimmed mean, depending on $\delta$). Huber is smooth and differentiable, works well with gradient-based training, and returns one model.

Quantile regression when you want **specific distributional statistics** (P10, P50, P90) — for prediction intervals, asymmetric loss, or probabilistic forecasting. Requires separate models per quantile (or a joint multi-quantile model) and produces distributional summaries, not a single best estimate.

If all you want is a robust central estimate, use Huber. If you need intervals or asymmetric losses, use quantile regression.

---

## Final Cheat-Sheet Summary

**Classification**

- Balanced data, equal costs → accuracy, F1.
- Imbalanced → PR-AUC, F1, MCC.
- Threshold-free comparison → ROC-AUC (balanced) or PR-AUC (imbalanced).
- Probabilistic decisions → ECE + Brier / log-loss.
- Multiclass → macro-F1 (equal class weight), weighted-F1 (frequency weight); inspect confusion matrix.

**Regression**

- Gaussian-ish errors, mean-targeting → MSE/RMSE.
- Outliers → MAE or Huber.
- Scale-independent → MAPE (if $y > 0$), R².
- Intervals / asymmetric loss → quantile loss.

**Clustering**

- No labels → silhouette, DB, CH (but only for convex clusters); stability analysis.
- Labels available → ARI, AMI, V-measure.
- Density / non-convex → DBCV, downstream task metrics.

**Significance**

- Paired binary → McNemar.
- Paired continuous, ~normal → paired t (corrected for CV dependence).
- Paired continuous, non-normal → Wilcoxon signed-rank.
- Multiple tests → Bonferroni / Holm / BH depending on FWER vs FDR goal.
- Always report effect size and CI, not just p-value.

**Golden rules**

1. Match the metric to the business objective, not vice versa.
2. Report confidence intervals, not just point estimates.
3. Inspect subgroups and confusion matrices — aggregates hide failures.
4. Never tune the threshold or hyperparameters on the final test set.
5. Statistical significance is necessary but not sufficient — practical significance matters.

*End of guide.*

---

**[← Previous Chapter: Data Quality & Labeling](data_quality_labeling_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Cross-Validation →](cross_validation_guide.md)**
