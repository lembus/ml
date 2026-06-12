# Data Quality & Labeling — A Comprehensive ML Reference Guide

*A textbook-quality learning guide for ML practitioners and applied scientists. No prior exposure assumed. Structured for both foundational learning and interview preparation.*

---

## Table of Contents

**Part I — Data Quality Dimensions**
1. Motivation & Intuition
2. Conceptual Foundations
3. Mathematical Formulation
4. Worked Examples
5. Relevance to ML Practice
6. Common Pitfalls & Misconceptions

**Part II — Labeling Strategies**
- Manual Labeling
- Weak Supervision (Snorkel)
- Active Learning
- Semi-Supervised Learning

**Part III — Label Quality Assessment**
- Inter-annotator Agreement (Cohen's κ, Fleiss' κ)
- Gold Standard Evaluation & Adjudication
- Confident Learning

**Part IV — Interview Preparation**
- Foundational, Mathematical, Applied, Debugging, Probing follow-ups

---

# PART I — DATA QUALITY DIMENSIONS

## 1. Motivation & Intuition

### Why data quality is the ML problem

Machine learning models are functions fit to data. Everything the model "knows" it learned from training examples, and everything it reports about the world at inference time it inherits from its input features. The oldest aphorism in computing — *garbage in, garbage out* — is not rhetoric in ML; it is a mathematical certainty. A model cannot distinguish between a signal and an artifact if the artifact looks, at the level of features and labels, identical to real signal.

Consider three concrete failures that are routinely used to teach this lesson:

- **Amazon's resume-screening model (2015–2018)** was trained on ten years of hiring decisions. Because historical hiring data reflected past bias, the model learned to penalize resumes containing the word "women's" (as in "women's chess club captain"). The model was technically correct with respect to its training labels — the labels themselves were the problem.
- **A widely cited chest X-ray pneumonia detector** achieved strong test performance, but was later shown to be detecting the hospital of origin rather than disease. Scanners at specific hospitals left radiographic signatures; hospitals with higher pneumonia prevalence were identifiable from the scanner artifact. Accuracy was high, but the feature being used was a data-collection artifact, not a clinical finding.
- **Early COVID-19 detection models** published in 2020 were shown in a *Nature Machine Intelligence* review to have systematic quality problems: pediatric negatives paired with adult positives, duplicated images across train and test, and labels derived from inconsistent sources. None of the 62 reviewed models was clinically usable.

In every case, the failure was not model architecture. It was data.

### The five classical dimensions

The academic framework for thinking about data quality was crystallized by Wang & Strong (1996) and has since been adopted (with variations) by essentially every data-quality standard (ISO/IEC 25012, DAMA-DMBOK). We focus on the five dimensions most directly relevant to ML:

| Dimension | Informal question it answers |
|---|---|
| Accuracy | "Do my values correctly describe reality?" |
| Completeness | "Is anything missing?" |
| Consistency | "Do different sources agree with each other?" |
| Timeliness | "Is the data fresh enough to be useful?" |
| Validity | "Does the data satisfy its declared rules and formats?" |

A data point can be accurate but stale (timeliness violation), valid in format but wrong in value (accuracy violation), complete in every record but missing whole segments of the population (completeness violation). Each dimension is logically independent and each must be measured separately.

### Intuition from everyday examples

- **Accuracy**: A thermometer reads 20°C when the true temperature is 22°C. The value is wrong.
- **Completeness**: A hospital exports patient vitals, but the export silently drops records where a specific field is null. The dataset looks fine in isolation but under-samples an important subpopulation.
- **Consistency**: One database stores country as "United States", another as "USA", a third as "US". A join on country fails silently.
- **Timeliness**: A fraud model trained on card transactions from six months ago misses a new attack pattern that emerged last week.
- **Validity**: A "price" field contains the string "N/A" in 3% of rows; downstream code casts to float and crashes.

### Why an ML system needs all five, not just accuracy

A naïve view treats "data quality" as synonymous with "label accuracy." This is wrong for three reasons:

1. **Features matter too.** Most real ML failures are feature-side, not label-side. A feature that is 99% valid but has a format change on day 100 produces a silent regression.
2. **Completeness bias is the hardest to detect.** If a segment is missing, the model simply does not learn about it. The training metrics look fine because the missing segment is not in the test set either.
3. **Timeliness governs drift.** Even a perfect model on a perfect historical dataset degrades as the world changes. Treating data as static is a common source of post-deployment failure.

Good ML systems monitor all five dimensions continuously, and the monitoring itself is a first-class engineering artifact — often encoded as validation rules in tools like TensorFlow Data Validation, Great Expectations, or Deequ.

---

## 2. Conceptual Foundations

### Accuracy

**Definition.** Accuracy is the degree to which a recorded value corresponds to the true value of the attribute it represents. It has two components:

- **Syntactic accuracy**: The value is in the set of admissible values (e.g., "blue" is in the color dictionary).
- **Semantic accuracy**: The value correctly describes reality (e.g., the car is actually blue).

Syntactic accuracy is cheap to check (lookup table, regex). Semantic accuracy requires a source of truth. In ML, that source of truth is sometimes a gold label, sometimes a downstream event (did the user actually click?), sometimes a physical measurement (did the sensor agree with a reference?).

**Measurement error** is the difference between recorded and true value. We typically decompose measurement error into:
- **Systematic error (bias)**: A repeatable offset (e.g., thermometer calibrated 2°C high).
- **Random error (variance)**: Zero-mean noise (e.g., sensor jitter).

Systematic error is far more dangerous in ML because it does not average out with more data.

**Assumptions.** Accuracy assessment presumes: (a) the existence of a ground truth, (b) the ability to sample from it, and (c) that the sample is representative. All three can fail: ground truth may be undefined (subjective labels), unreachable (historical data where reality is unknowable), or unrepresentative (gold samples drawn from a privileged slice).

**Failure modes.** When ground truth is itself derived from an imperfect process (e.g., "ground truth" cancer diagnoses from radiologist labels), measured accuracy is bounded by the quality of that process. This is sometimes called the *ceiling* problem: you cannot evaluate a model above the quality of your evaluation data.

### Completeness

**Definition.** Completeness is the degree to which all required data is present. There are three distinct flavors:

- **Column completeness**: For a given record, are all expected fields populated?
- **Row completeness (coverage)**: Are all expected records present?
- **Population completeness**: Is every relevant subpopulation represented proportionally?

The third is the most insidious. A dataset can have 100% column completeness and 100% row completeness against a schema and still grossly under-represent certain groups.

**Missing data mechanisms** (Rubin, 1976) — the canonical taxonomy:

- **MCAR (Missing Completely At Random)**: Missingness is independent of all data, observed or unobserved. A sensor that drops packets uniformly at random. Safe to drop rows.
- **MAR (Missing At Random)**: Missingness depends only on observed variables. Older users skip the income field at a higher rate, but once we condition on age, missingness is random. Imputation is valid.
- **MNAR (Missing Not At Random)**: Missingness depends on the unobserved value itself. High earners skip the income field at a higher rate. Standard imputation is biased; you need a model of the missingness mechanism.

**Assumptions.** Completeness assessment presumes you know what "complete" looks like. This is often wrong — teams frequently discover six months into production that they never had the data they thought they had.

**Failure modes.** Silent completeness failures (a field going from 99% populated to 90% populated without alarm) are the source of a large fraction of ML production incidents. Monitoring null rates per feature is non-negotiable.

### Consistency

**Definition.** Consistency has two distinct meanings:

- **Internal consistency**: A single record does not contradict itself (birth date and age agree).
- **Cross-source consistency**: Multiple representations of the same entity agree (the CRM and the billing system report the same email address).

**Schema evolution** is the dynamic version of consistency. Schemas change — fields are added, renamed, split, merged — and consistency over time requires either versioned schemas (Avro, Protocol Buffers) or explicit reconciliation logic.

**Entity resolution** is the problem of recognizing that two records refer to the same entity. "John Smith, 123 Main St" and "J. Smith, 123 Main Street" probably denote the same person, but the decision requires fuzzy matching and is fundamentally probabilistic.

**Assumptions.** Consistency assumes a canonical representation exists. When two systems disagree, which one is right? Often the answer is "neither, they diverged three years ago and nobody noticed." Consistency checks without a master data management strategy devolve into arbitrary judgment calls.

**Failure modes.** The worst consistency failures are those where both sides *look* self-consistent but silently encode different semantics (e.g., `revenue` is net in one table and gross in another). No format check catches this.

### Timeliness

**Definition.** Timeliness is the degree to which data is current for the task. It has two sub-dimensions:

- **Currency (freshness)**: Age of the data relative to the real-world state it represents.
- **Latency**: Delay between a real-world event and its availability in the system.

A fraud model may require sub-second timeliness. A quarterly financial forecasting model tolerates weeks.

**Assumptions.** Timeliness presumes a decision was made about the required freshness, usually derived from the rate at which the underlying distribution changes. This assumption is rarely made explicit, which is why it is so often violated.

**Failure modes.** The most common timeliness failure in ML is **training–serving skew** caused by stale features at training time versus fresh features at inference time, or vice versa. If the training pipeline uses yesterday's batch aggregates but the serving pipeline uses real-time aggregates, the model sees different distributions in the two settings.

### Validity

**Definition.** Validity is the degree to which data conforms to defined syntactic rules and domain constraints. Examples:

- Types: `age` is an integer.
- Ranges: `age ∈ [0, 120]`.
- Formats: email matches RFC 5322.
- Referential integrity: every `order.customer_id` exists in the `customers` table.
- Business rules: if `status == 'refunded'`, then `refund_date` is not null.

**Assumptions.** Validity rules are a formal codification of domain knowledge. They assume that (a) the rules are complete (every invalid value is caught by some rule) and (b) the rules are correct (no valid value fails a rule). Both fail in practice.

**Failure modes.** Over-strict validity rules reject real data and create completeness problems. Under-strict rules let invalid data through and create accuracy problems. The trade-off is fundamental and cannot be eliminated, only tuned.

### How the dimensions interact

The five dimensions are independent axes, but they interact operationally:

- Remediating a completeness issue by imputation changes accuracy (imputed values are inferences, not measurements).
- Enforcing strict validity rejects records and reduces completeness.
- Improving timeliness usually increases cost and may reduce accuracy (less time to verify each record).
- Consistency repairs (choosing one value when two disagree) introduce accuracy risk if the wrong side is chosen.

These trade-offs are the subject of the *data quality triangle* folklore — you can hit any two of {high accuracy, high completeness, low latency} but rarely all three.

---

## 3. Mathematical Formulation

### 3.1 Accuracy and measurement error

Let $x_i$ be the true value of attribute $X$ for record $i$, and $\hat{x}_i$ be the recorded value. The measurement error is
$$
\varepsilon_i = \hat{x}_i - x_i.
$$

The classical decomposition is
$$
\varepsilon_i = b + \eta_i, \quad \mathbb{E}[\eta_i] = 0, \quad \text{Var}(\eta_i) = \sigma^2,
$$
where $b$ is the systematic bias and $\eta_i$ is zero-mean random noise.

**Mean Squared Error (MSE)** of the measurement process is
$$
\text{MSE} = \mathbb{E}[(\hat{x} - x)^2] = b^2 + \sigma^2.
$$

This is the familiar bias–variance decomposition applied to the measurement device itself. It tells you two different remediations:

- If bias dominates ($b^2 \gg \sigma^2$): recalibrate. More data will not help.
- If variance dominates ($\sigma^2 \gg b^2$): average over more measurements. Bias will not shrink.

For categorical attributes, accuracy is measured as an **error rate**:
$$
\text{Acc} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[\hat{x}_i = x_i].
$$

This is identical to classification accuracy, which is why label-noise analysis and data-accuracy analysis use the same toolkit.

### 3.2 Completeness and missingness

Let $M_i \in \{0, 1\}$ indicate whether record $i$ is missing a particular attribute ($M_i = 1$ for missing). Let $X_i$ be the value that *would have been* observed.

**Missingness mechanisms** formalize the conditional dependence structure. Let $X = (X_{obs}, X_{mis})$ partition variables into observed and missing pieces. Let $\phi$ parameterize the missingness process.

- **MCAR**: $P(M \mid X, \phi) = P(M \mid \phi)$. Missingness is independent of data.
- **MAR**: $P(M \mid X, \phi) = P(M \mid X_{obs}, \phi)$. Missingness depends only on observed variables.
- **MNAR**: $P(M \mid X, \phi)$ depends on $X_{mis}$.

Under MCAR and MAR, the missingness mechanism is **ignorable** for likelihood-based inference: maximum likelihood over observed data is still consistent. Under MNAR, the mechanism must be jointly modeled. This is why MNAR is called *non-ignorable*.

**Little's MCAR test** provides a χ² test statistic for the null hypothesis that the data is MCAR. Rejecting MCAR does not tell you whether you are in MAR or MNAR, but failing to reject is weak evidence for MCAR.

**Completeness rate** for attribute $j$:
$$
C_j = 1 - \frac{1}{n}\sum_{i=1}^n M_{ij}.
$$

**Coverage** against an expected population $\mathcal{P}$ with known segment proportions $\pi_k$:
$$
\text{Coverage}_k = \frac{n_k / n}{\pi_k},
$$
which equals 1 under perfect proportional representation, < 1 under under-sampling, > 1 under over-sampling.

### 3.3 Consistency metrics

Let $v^{(A)}$ and $v^{(B)}$ be the values of the same attribute for the same entity in sources $A$ and $B$. The **pairwise consistency rate** is
$$
\text{Consist}(A, B) = \frac{1}{n} \sum_{i=1}^n \mathbb{1}[v_i^{(A)} = v_i^{(B)}].
$$

For continuous attributes, equality is replaced with tolerance:
$$
\mathbb{1}[|v_i^{(A)} - v_i^{(B)}| \le \tau].
$$

For fuzzy matching (entity resolution), one uses a similarity measure $s(v^{(A)}, v^{(B)}) \in [0, 1]$ (Jaccard, Levenshtein ratio, cosine over embeddings) with a decision threshold.

### 3.4 Timeliness metrics

Let $t_\text{event}$ be the real-world event timestamp and $t_\text{avail}$ the time the data becomes available in the system. Then:

**Latency**: $L_i = t_\text{avail}^{(i)} - t_\text{event}^{(i)}$.

**Freshness at query time** $t_q$: $F_i = t_q - t_\text{event}^{(i)}$.

A common SLA formulation is a quantile constraint: $P(L_i > L_\text{max}) < \alpha$ (e.g., 99% of records within 5 minutes).

The **effective freshness decay** of a feature can be modeled as an exponential relevance function:
$$
r(F) = \exp(-F / \tau),
$$
where $\tau$ is a half-life governed by the rate of change of the underlying process.

### 3.5 Validity rates

For a set of $K$ validity rules $R_1, \ldots, R_K$, the per-rule violation rate is
$$
V_k = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[\text{record } i \text{ violates } R_k].
$$

The **record-level validity rate** is
$$
V = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\left[\bigwedge_k R_k(i)\right].
$$

Note $V \le \min_k(1 - V_k)$; strict rules compose multiplicatively, which is why adding more rules tends to sharply reduce apparent validity.

---

## 4. Worked Examples

### 4.1 A credit-risk dataset diagnosis

Imagine a dataset of 100,000 loan applications with features: `age`, `income`, `credit_score`, `employment_length`, `zip_code`, `applied_at` (timestamp), `default` (label).

Before training anything, we compute the five dimensions.

**Accuracy.**
- `credit_score` is sourced from a bureau. We sample 500 rows and re-pull the bureau values. 492 match exactly, 8 are off by more than 10 points. Accuracy ≈ 98.4%.
- The 8 mismatches are all from bureau vendor B (used for 15% of records). Mismatch rate for vendor B is 3.8% vs. 0.2% for vendor A. This is systematic error: vendor B has a stale pipeline.

**Completeness.**
- `income` is null for 7,200 records (7.2%).
- Breaking down by `age` bucket: null rate is 4% for ages 25–44, 15% for ages 65+. This is strong evidence against MCAR.
- We run a regression: `P(income null)` vs `age`, `employment_length`. The model explains most variance. Conditional on age and employment length, missingness looks random → MAR is plausible.
- Decision: use multiple imputation conditioning on age and employment length, not listwise deletion (which would drop 65+ disproportionately).

**Consistency.**
- `zip_code` comes from two sources: the application form and an address-verification API.
- They agree in 91% of records. 9% disagreement — but 7% of those disagreements are because the API returns ZIP+4 while the form returns ZIP-5.
- After canonicalizing to ZIP-5, agreement is 98%.
- Remaining 2% disagreement is mostly typos on the form. We treat the API as authoritative.

**Timeliness.**
- `credit_score` at training time is from the day of application. At serving time, we plan to use a real-time pull.
- This is a **training–serving skew** risk: the real-time value may reflect recent delinquencies not present in the historical snapshot.
- Mitigation: retrieve historical credit scores from the bureau with a lookback that matches the real-time latency (e.g., use the score as-of the timestamp, not as-of today).

**Validity.**
- 42 records have `age > 120`.
- 310 records have `income < 0`.
- 88 records have `employment_length > age - 14`.
- Total rule-violating records: 431 (0.43%). Investigation shows most are data-entry errors; 12 are actual fraud (the loan forms are adversarial inputs). For the ML task we filter these, but we flag them as a separate fraud-detection signal.

**End-to-end summary.** Before modeling, we identified (a) vendor-specific systematic accuracy error, (b) age-correlated missingness requiring MAR-compatible imputation, (c) a format discrepancy in ZIP that would have corrupted a join, (d) a training–serving skew risk on credit score, and (e) adversarial invalid records that are themselves informative. None of these would have been caught by running a model and looking at accuracy.

### 4.2 Computing coverage for a fraud model

A payments company trains a fraud model on transactions from Jan–June. The expected traffic mix by country is approximately:

| Country | Expected share $\pi_k$ | Observed in training $n_k/n$ | Coverage $n_k/n / \pi_k$ |
|---|---|---|---|
| US | 0.60 | 0.72 | 1.20 |
| UK | 0.15 | 0.15 | 1.00 |
| DE | 0.10 | 0.08 | 0.80 |
| BR | 0.10 | 0.03 | 0.30 |
| Other | 0.05 | 0.02 | 0.40 |

Brazil is severely under-sampled (coverage 0.30). The team investigates: the fraud-review system for BR was down for 3 weeks in March, and during that time transactions were auto-approved (no label). The training data has fewer BR examples *and* the ones it has are biased toward a period of abnormal fraud patterns. The model will under-perform on BR.

Remediation options: (1) collect more BR data; (2) reweight the training loss by inverse coverage; (3) train a country-specific head; (4) document the known blind spot and route BR to a more conservative rule-based system. The right choice depends on how much signal is salvageable.

### 4.3 Validity rule set for a user table

| Rule | Violations | Notes |
|---|---|---|
| `email` matches RFC 5322 | 2.1% | Most are trailing spaces, fixable. |
| `age ∈ [13, 110]` | 0.8% | Minimum 13 per platform ToS. |
| `signup_date ≤ now` | 0.03% | Clock skew on one shard. |
| `country_code` ∈ ISO-3166 | 0.4% | Legacy values like "UK" instead of "GB". |
| Joint: `verified == true ⇒ phone is not null` | 0.1% | Logic bug; breaks downstream SMS flow. |

Each rule encodes a piece of domain knowledge. The record-level validity (all rules pass simultaneously) is about 96.6%. The most important number is not the 96.6% — it is the *trend*. If it drops to 94% overnight, something changed.

---

## 5. Relevance to Machine Learning Practice

### Where data quality enters the ML lifecycle

**Training.** The training set defines the function class the model can represent accurately. Label noise caps achievable accuracy (we'll see this formally in Part III via confident learning). Feature noise reduces signal-to-noise ratio and can require architectural changes (e.g., more regularization, data augmentation). Coverage gaps produce systematic errors that no amount of training time repairs.

**Validation/evaluation.** Perhaps the highest-leverage application: your validation metric is only as meaningful as your validation labels. A 95% accuracy model evaluated against 90%-accurate labels tells you essentially nothing about the 85–95% range of true performance. Gold standard subsets with adjudicated labels are essential for trustworthy evaluation.

**Inference.** Serving-time data quality issues manifest as silent degradation. Common incidents: an upstream ETL job drops records with a specific feature value, the model's predictions shift, downstream metrics move, and root cause takes days. Monitoring feature distributions (KS tests, PSI, null rates) against a training baseline catches these.

**Monitoring.** Data quality monitors are typically organized into tiers:
- **Tier 1 (blocking)**: schema violations, null rates > threshold, row count drops. Fail the pipeline.
- **Tier 2 (alerting)**: distribution shifts, missingness spikes, staleness. Page on-call.
- **Tier 3 (observability)**: trends, quantile tracking, feature-level metrics. Dashboards.

### When to invest where

- **High-variance, low-bias measurement** (noisy but unbiased sensors) → collect more data, average, add regularization.
- **High-bias measurement** → fix the data source before modeling. More model complexity cannot fix systematic error.
- **MCAR missingness** → mean imputation, listwise deletion, or dedicated ML imputation; all are reasonable.
- **MAR missingness** → multiple imputation using observed covariates, or models that handle missingness natively (e.g., XGBoost's default-direction splits).
- **MNAR missingness** → this is a research problem. Either model the missingness mechanism explicitly (selection models, pattern-mixture models) or acknowledge the limitation in your evaluation.
- **Cross-source inconsistency** → designate a system of record. Do not average or pick randomly.
- **Timeliness-constrained features** → compute features from a consistent temporal context ("feature store" pattern: point-in-time correct joins).

### Trade-offs

- **Quality vs. cost**: Manually auditing every record is infeasible. Stratified sampling and auto-generated quality rules are the scalable alternative.
- **Quality vs. latency**: Stronger validation takes time. Real-time systems must tier their checks.
- **Interpretability vs. robustness**: A model that handles missing features natively (e.g., XGBoost) is robust but can mask upstream issues. Explicit imputation creates a visible record of what was imputed.
- **Bias vs. variance**: Dropping records with any quality issue reduces variance (cleaner signal) but introduces selection bias (systematic exclusion of certain populations).

### Tooling landscape

| Tool | Role |
|---|---|
| Great Expectations (Python) | Declarative data expectations; pipeline integration. |
| TensorFlow Data Validation (TFDV) | Schema inference, distribution skew detection, training–serving skew. |
| Deequ (Spark) | Constraint-based validation on large data. |
| Whylogs / WhyLabs | Lightweight profiling and drift monitoring. |
| Monte Carlo, Bigeye, Anomalo | Commercial data observability; anomaly detection across warehouses. |
| Snorkel, Cleanlab | Label quality specifically (we meet these in Parts II & III). |

---

## 6. Common Pitfalls & Misconceptions

**1. "Accuracy is the only dimension that matters for ML."** Wrong — completeness bias and timeliness (training–serving skew) cause more production incidents than label accuracy.

**2. "Missing-at-random means missing at random."** No. MAR is a technical term meaning missingness depends only on *observed* variables. It is weaker than MCAR but much stronger than MNAR. The colloquial use of "random" is MCAR, not MAR.

**3. "More data fixes data quality problems."** False for systematic error. If your measurement device is biased, more data converges to the biased value. Fix the source first.

**4. "We don't have a data quality problem — our metrics look fine."** Metrics on the training set do not reveal coverage gaps. A missing subpopulation is not in the test set either.

**5. "Validation rules will catch everything."** Validation rules encode known failures. New failure modes are not covered. Drift detection complements (does not replace) rule-based validation.

**6. "Imputation is free."** Imputation introduces model-dependent fiction into your dataset. Downstream models learn the imputer's artifacts. At minimum, retain a missingness indicator column.

**7. "Outliers are noise."** Sometimes. Often they are the most informative records (fraud, failures, edge cases). Blind outlier removal is a form of silent relabeling.

**8. "We fixed the data quality bug in the pipeline, so historical data is now correct."** Historical data remains broken. Either re-process historical data through the fix or explicitly partition before/after.

**9. "Gold labels are perfect labels."** Gold labels are the labels you chose to trust. They are often 95–98% accurate, not 100%. Evaluate the evaluator.

**10. "Consistency across sources means the data is correct."** Two databases can agree and both be wrong if they share a common upstream feed.

---

# PART II — LABELING STRATEGIES

## Manual Labeling

### 1. Motivation & Intuition

Supervised learning requires labels, and labels must come from somewhere. The most direct answer is to have humans look at the data and record the correct output. Manual labeling is simultaneously the oldest, the most expensive, and often the most accurate labeling strategy. Every other strategy in this section exists because manual labeling does not scale.

A concrete example: ImageNet, the dataset that ignited modern deep learning, contains ~14 million images labeled by ~49,000 crowd workers across ~167 countries over two years. The cost of producing it (in wages, infrastructure, and quality control) ran into the millions of dollars. Even at that scale, label noise in ImageNet has been estimated at 3–6% for the test set (Northcutt et al., 2021).

Manual labeling is attractive when:
- The task is genuinely hard to automate (medical image interpretation, nuanced text classification, physical-world annotations).
- Labels are critical to business decisions (compliance, safety, legal).
- The dataset is small enough to afford.

It is infeasible when:
- The volume is enormous (billions of records).
- Expertise is narrow and expensive (radiologists, lawyers).
- The distribution drifts faster than the annotation pipeline.

### 2. Conceptual Foundations

**Annotator types.**
- **Expert annotators**: Domain specialists (e.g., pathologists for histopathology). High accuracy, low throughput, high cost.
- **Trained annotators**: Non-experts who have been trained on specific guidelines. Medium cost, medium throughput.
- **Crowd workers** (MTurk, Prolific, Appen, Scale AI): Low cost per label, high throughput, variable quality. Used for simple tasks with clear guidelines.

**The annotation task design problem.** What you ask annotators to do determines what you get. Key design choices:

- **Granularity**: Classify into 3 classes or 30? Finer classes have lower agreement but more information.
- **Format**: Binary, multi-class, multi-label, free text, bounding boxes, keypoints, relationship graphs. Each has its own noise profile.
- **Decomposition**: Break complex judgments into simple binary questions (e.g., instead of "Is this comment toxic?" ask "Does this contain profanity?", "Does this target a group?", "Does this call for violence?"). Agreement rises, depth increases.
- **Instruction specificity**: Too vague → high disagreement. Too narrow → annotators game the rules, missing the intent.

**Quality control mechanisms.**
- **Qualification tasks / screening**: Filter annotators up front with a test.
- **Gold questions**: Inject pre-labeled items to measure per-annotator accuracy in real time.
- **Redundancy**: Have $k$ annotators label each item; aggregate (majority vote, Dawid–Skene).
- **Consensus with adjudication**: If the $k$ annotators disagree, escalate to an expert.
- **Calibration sessions**: Periodic meetings to align on edge cases.
- **Double-pass labeling**: One annotator labels, another reviews — used for high-stakes data.

**Aggregation algorithms.** When multiple annotators label the same item, you have a noisy consensus problem. Common aggregators:
- **Majority vote**: Simple, assumes equal annotator quality.
- **Weighted vote**: Weight each annotator by their gold-question accuracy.
- **Dawid–Skene (1979)**: EM algorithm that jointly infers annotator skill and true label, assuming independent errors.

**Underlying assumptions.**
- Annotators are conditionally independent given the true label (broken if they share guidelines that are themselves biased).
- True labels exist (broken for genuinely subjective tasks: is this text funny?).
- Annotators attend to the task (broken by spammers, satisficers, automation by LLMs).

**What breaks when assumptions violate.** The biggest modern failure mode: annotators silently delegate to LLMs. Veselovsky et al. (2023) estimated 33–46% of summaries on MTurk were LLM-assisted. Your "human labels" are actually model outputs in disguise, destroying the independence of human and model signal.

### 3. Mathematical Formulation

**The observation model.** For item $i$ with true label $y_i$, annotator $a$ reports label $\hat{y}_{i,a}$. Model the annotator as a confusion matrix $\theta^{(a)}$ with
$$
\theta^{(a)}_{j,k} = P(\hat{y}_{i,a} = k \mid y_i = j).
$$
A perfect annotator has $\theta^{(a)} = I$.

**Majority vote** estimator:
$$
\hat{y}_i^{\text{MV}} = \arg\max_k \sum_{a=1}^A \mathbb{1}[\hat{y}_{i,a} = k].
$$

**Dawid–Skene EM algorithm.** Treat $y_i$ as latent, $\theta^{(a)}$ and class priors $\pi_k = P(y_i = k)$ as parameters.

- **E-step**: Compute soft labels given current parameter estimates.
$$
q_{i,k} = P(y_i = k \mid \hat{y}_{i,1}, \ldots, \hat{y}_{i,A}) \propto \pi_k \prod_{a=1}^A \theta^{(a)}_{k, \hat{y}_{i,a}}.
$$
- **M-step**: Update parameters.
$$
\pi_k = \frac{1}{n}\sum_i q_{i,k}, \quad \theta^{(a)}_{j,k} = \frac{\sum_i q_{i,j} \mathbb{1}[\hat{y}_{i,a} = k]}{\sum_i q_{i,j}}.
$$

Iterate until convergence. Initialize with majority vote.

**Cost model.** Suppose each label costs $c$ per annotator and we use $k$ annotators per item. Labeling $n$ items costs $nkc$. If individual annotator accuracy is $p$, the majority-vote accuracy with $k$ annotators is (by the binomial tail)
$$
\text{Acc}_k = \sum_{j=\lceil k/2 \rceil}^k \binom{k}{j} p^j (1-p)^{k-j}.
$$
For $p = 0.8$: $k=1 \to 0.80$, $k=3 \to 0.896$, $k=5 \to 0.942$. Each redundancy doubling yields diminishing returns.

### 4. Worked Examples

**A content moderation project.** A team needs 50,000 labeled user reviews (positive / negative / spam). Budget: $10,000.

- Per-label price on a crowd platform: $0.05.
- With $k=1$ annotator: 200,000 labels possible. Covers the dataset but accuracy ≈ 0.80.
- With $k=3$ annotators: 66,000 labels possible, accuracy ≈ 0.90.
- With $k=5$: 40,000 labels possible, accuracy ≈ 0.94.

The $k=3$ configuration covers roughly the whole dataset with high-90s accuracy after EM aggregation. The team picks $k=3$ plus 5% gold questions (so effective throughput is $10,000 / (0.05 \times 3.15) = 63,500$ items; still enough to cover the 50k).

After a pilot batch: per-annotator accuracy on gold questions ranges from 0.58 to 0.96. Annotators below 0.75 are removed. Dawid–Skene is run on the remaining data; the inferred posterior is used to flag items with max-posterior < 0.8 for expert adjudication.

**Using Dawid–Skene numerically.** Three annotators label item $i$ as (1, 1, 0). Suppose prior $\pi_0 = \pi_1 = 0.5$, and the annotators have estimated confusion matrices:

- $\theta^{(1)} = \theta^{(2)} = \theta^{(3)} = \begin{pmatrix} 0.9 & 0.1 \\ 0.2 & 0.8 \end{pmatrix}$

(rows = true label, columns = reported label). Compute:
$$
P(y=1 \mid \hat{y}) \propto 0.5 \cdot 0.8 \cdot 0.8 \cdot 0.2 = 0.064,
$$
$$
P(y=0 \mid \hat{y}) \propto 0.5 \cdot 0.1 \cdot 0.1 \cdot 0.9 = 0.0045.
$$
Normalizing: $P(y=1 \mid \hat{y}) \approx 0.934$. Majority vote said 1; Dawid–Skene agrees but with calibrated confidence. The soft label 0.934 is usable for loss weighting.

### 5. Relevance to ML Practice

Manual labeling underpins almost every supervised dataset used in production ML. It is the reference against which every cheaper labeling strategy is evaluated. Specifically:

- **Test set creation**: Even if training labels are noisy (from weak supervision or models), test sets almost always require careful human labeling with adjudication.
- **Seed sets**: Active learning, weak supervision, and semi-supervised learning all require a small high-quality labeled set to bootstrap.
- **Evaluation of labeling strategies**: You cannot tell if your Snorkel pipeline is correct unless you compare its outputs to trusted human labels.

**When to use it.**
- Small-to-medium datasets where quality dominates cost.
- Tasks requiring domain expertise.
- Creating benchmark test sets.

**When not to use it.**
- Billion-scale datasets — shift to weak supervision or self-supervision.
- Rapidly changing distributions — annotation cycle cannot keep up.
- Highly subjective tasks without a defensible ground truth.

**Trade-offs.**
- Accuracy vs. cost: annotator redundancy trades dollars for accuracy.
- Cost vs. time: more annotators in parallel reduces wall-clock time but not cost.
- Specificity vs. generalizability: narrow guidelines produce consistent but brittle labels.

### 6. Common Pitfalls & Misconceptions

**1. "More annotators per item is always better."** Diminishing returns after 3–5. Beyond that, money is better spent covering more items.

**2. "Annotators are interchangeable."** Individual accuracy varies by a factor of 2. Gold questions are essential to identify and weight annotators.

**3. "The guidelines are clear."** Guidelines that seem clear to the team routinely produce <60% agreement on a pilot batch. Always pilot before scaling.

**4. "We can trust the platform's quality controls."** Every platform has a spam problem. Ship your own gold questions and measure.

**5. "Human labels are ground truth."** Human inter-annotator agreement often caps at 85–92% on hard tasks. Your "ground truth" is bounded above by this.

**6. "Crowd workers are not using LLMs."** Many are. Build anti-LLM checks (tasks that are hard for current LLMs, paraphrase detection, etc.) or acknowledge the risk.

---

## Weak Supervision

### 1. Motivation & Intuition

Manual labeling is expensive. But in most domains, experts have a lot of knowledge that *can't* easily become labels but *can* become rules. A radiologist might say "if this text report contains 'opacity' and 'bilateral', it's almost always pneumonia." A lawyer might say "if a contract clause contains 'change of control', it's almost always a material provision." A spam-detection team might have dozens of regex patterns, blocklists, and model outputs that each correlate weakly with the true label.

Weak supervision is the idea of combining many **noisy, heuristic, possibly-conflicting** labeling sources into a single probabilistic label per example, without ever needing a human to look at that specific example. The flagship framework is **Snorkel** (Ratner et al., 2016–2020), developed at Stanford and widely adopted in industry (Google, Apple, Intel, large banks).

The intuition: if you have 20 noisy rules and they mostly agree with each other, their consensus is much more reliable than any single rule. And crucially, **you can estimate how good each rule is without knowing the true labels** — by looking at how often rules agree and disagree with each other.

This is the same idea behind ensemble methods, but applied to the labeling process itself.

### 2. Conceptual Foundations

**Labeling function (LF).** A Python function (or rule) that takes an example and returns either a label or *abstain*. Examples for a spam classifier:

```python
def lf_contains_viagra(x): return SPAM if "viagra" in x.text else ABSTAIN
def lf_has_many_caps(x): return SPAM if caps_ratio(x) > 0.5 else ABSTAIN
def lf_short_message(x): return HAM if len(x.text) < 10 else ABSTAIN
def lf_from_known_sender(x): return HAM if x.sender in trusted_list else ABSTAIN
```

LFs may abstain (return no label) on examples they don't apply to. This is crucial — LFs are usually high-precision on a subset and silent elsewhere.

**Coverage, overlap, conflict.** For an LF $\lambda$:
- **Coverage**: fraction of examples where $\lambda$ does not abstain.
- **Overlap**: fraction of examples where $\lambda$ and another LF both label.
- **Conflict**: fraction of examples where they label differently.

**Label model.** Given the matrix of LF outputs (size $n \times m$ for $n$ examples and $m$ LFs), learn a generative model of the joint distribution of LF outputs and true labels. Infer a probability for each possible true label of each example.

**End-to-end Snorkel pipeline.**
1. Write LFs using domain knowledge.
2. Apply LFs to unlabeled data → get the $n \times m$ label matrix $\Lambda$.
3. Fit a label model on $\Lambda$ (without ground truth!) to estimate LF accuracies and correlations.
4. Use the label model to produce probabilistic training labels $\tilde{y}_i \in [0,1]^K$.
5. Train a discriminative model (e.g., a neural net) on $(x_i, \tilde{y}_i)$ using a noise-aware loss.

The discriminative model in step 5 is the actual deployed classifier. The weak-supervision pipeline is used only to produce labels for it.

**Why does step 3 work without ground truth?** Key insight: if you know any two LFs are conditionally independent given the true label, then their agreement rate tells you their individual accuracies *up to a sign symmetry*. With three or more LFs, agreement patterns uniquely identify accuracies. This is the **conditional independence trick** and it's the conceptual core of Snorkel's label model.

**Underlying assumptions.**
- LFs are conditionally independent given the true label (or their dependencies are explicitly modeled).
- LFs are better than random.
- The discriminative model trained on noisy labels will generalize (empirically: yes, if you have many more unlabeled points than LFs).

**What breaks when assumptions violate.**
- **Correlated LFs** (two LFs both triggered by the same underlying feature): the label model over-weights their joint signal. Remedy: learn or specify a dependency structure (Snorkel MeTaL).
- **Systematically biased LF set**: if all LFs miss a class or subpopulation, no aggregation saves you. Coverage analysis is mandatory.
- **Abstain-heavy LFs**: an LF that covers 0.1% of data contributes almost nothing. Enforce minimum coverage.

### 3. Mathematical Formulation

**Setup.** $n$ unlabeled examples, $m$ LFs, $K$ classes. The label matrix $\Lambda \in \{-1, 0, 1, \ldots, K\}^{n \times m}$ where $\Lambda_{i,j} = 0$ means abstain (or equivalently $\Lambda_{i,j} \in \{ABSTAIN\} \cup \{1, \ldots, K\}$).

**Generative model (binary case).** Model the true label $Y \in \{-1, +1\}$ with prior $\pi$, and each LF's output $\lambda_j \in \{-1, 0, +1\}$ with conditional distribution
$$
P(\lambda_j \mid Y) \text{ parameterized by } \alpha_j \text{ (accuracy) and } \beta_j \text{ (coverage)}.
$$

A common parameterization:
$$
P(\lambda_j = 0 \mid Y) = 1 - \beta_j, \quad P(\lambda_j = Y \mid Y, \lambda_j \ne 0) = \alpha_j.
$$

The joint probability, assuming conditional independence:
$$
P(Y, \lambda_1, \ldots, \lambda_m) = P(Y) \prod_{j=1}^m P(\lambda_j \mid Y).
$$

**Learning without labels.** Given $n$ samples of $\Lambda$, maximize the marginal log-likelihood:
$$
\mathcal{L}(\theta) = \sum_{i=1}^n \log \sum_{y \in \{-1, +1\}} P(Y = y) \prod_{j=1}^m P(\lambda_{i,j} \mid Y = y; \theta).
$$

Identifiability: with $m \ge 3$ LFs that are pairwise conditionally independent given $Y$, the parameters $\alpha_j$ are identifiable from the observable agreement rates between LFs (Ratner et al., "triplet method"). The key identity: for LFs $j, k$ that are conditionally independent given $Y$,
$$
\mathbb{E}[\lambda_j \lambda_k \mid |\lambda_j|=|\lambda_k|=1] = \mathbb{E}[\lambda_j Y] \cdot \mathbb{E}[\lambda_k Y].
$$
Three such equations over three pairs $(j,k), (j,l), (k,l)$ let you solve for each individual $\mathbb{E}[\lambda_j Y]$ — i.e., each LF's accuracy — up to a global sign.

**Soft label output.** Posterior probability per example:
$$
\tilde{y}_i = P(Y = 1 \mid \Lambda_{i, \cdot}) = \frac{P(Y=1) \prod_j P(\lambda_{ij} \mid Y=1)}{\sum_y P(Y=y) \prod_j P(\lambda_{ij} \mid Y=y)}.
$$

**Noise-aware discriminative loss.** When training the downstream model $f_\phi$ with soft labels $\tilde{y}_i \in [0,1]$ (for binary), use expected cross-entropy:
$$
\mathcal{L}(\phi) = -\sum_{i=1}^n \left[ \tilde{y}_i \log f_\phi(x_i) + (1 - \tilde{y}_i) \log(1 - f_\phi(x_i)) \right].
$$

This is not the same as training on hard labels; it propagates label uncertainty into the gradient, reducing the update on ambiguous examples.

### 4. Worked Examples

**Toy spam classifier.** Five LFs, five examples, binary labels (+1 = spam, −1 = ham, 0 = abstain):

| Example | LF1 (viagra) | LF2 (caps) | LF3 (short_trust) | LF4 (trusted_sender) | LF5 (known_spam_ip) |
|---|---|---|---|---|---|
| x₁ | +1 | +1 | 0 | 0 | +1 |
| x₂ | 0 | 0 | 0 | −1 | 0 |
| x₃ | 0 | +1 | 0 | 0 | 0 |
| x₄ | −1* | 0 | 0 | −1 | 0 |
| x₅ | +1 | 0 | 0 | 0 | +1 |

\*LF1 never returns −1 in practice; I'm writing it for illustration. Assume in this toy it does.

Pairwise observed agreement (ignoring abstains):
- LF1 & LF2: agree on x₁, both label; on x₃ LF1 abstains. Agreement = 1/1.
- LF1 & LF5: agree on x₁, x₅. Agreement = 2/2.
- LF2 & LF5: agree on x₁. Agreement = 1/1.

With just 5 examples, estimates are noisy, but the method scales: on 10,000 unlabeled examples, Snorkel's triplet method gives stable accuracy estimates.

Assume the label model estimates: $\alpha_1 = 0.9$, $\alpha_2 = 0.7$, $\alpha_5 = 0.95$, LF3 accuracy $0.8$, LF4 accuracy $0.85$, prior $\pi = 0.3$ (spam).

For x₁: LF1, LF2, LF5 all say +1.
$$
P(Y=+1 \mid \Lambda) \propto 0.3 \cdot 0.9 \cdot 0.7 \cdot 0.95 = 0.18
$$
$$
P(Y=-1 \mid \Lambda) \propto 0.7 \cdot (1-0.9) \cdot (1-0.7) \cdot (1-0.95) = 0.00105
$$
Posterior ≈ 0.994 spam. Very confident.

For x₃: only LF2 fires with +1.
$$
P(Y=+1 \mid \Lambda) \propto 0.3 \cdot 0.7 = 0.21
$$
$$
P(Y=-1 \mid \Lambda) \propto 0.7 \cdot 0.3 = 0.21
$$
Posterior ≈ 0.5. LF2 alone, with accuracy 0.7 and low prior, is not informative enough to override the prior. A soft label near 0.5 tells the downstream model to essentially ignore this example.

This illustrates the signature behavior: high-confidence soft labels when many good LFs agree, noncommittal labels when evidence is weak.

**Full pipeline on a real dataset (schematic).**
1. Start with ~10,000 unlabeled medical transcripts.
2. Write 15 LFs capturing regex patterns, keyword lists, cross-references to a clinical ontology, and outputs from a pretrained classifier.
3. Apply LFs, compute coverage per LF (6–40%) and pairwise conflicts (2–15%).
4. Fit Snorkel's label model → estimated LF accuracies ranging 0.65 to 0.92.
5. Generate soft labels for all 10,000 transcripts; abstain rates drop to 2%.
6. Train a BERT classifier on the soft-labeled data using cross-entropy with probabilistic targets.
7. Evaluate on a 500-example manually-labeled test set: F1 = 0.82. (For comparison: 500-example fully-supervised baseline: F1 = 0.74. Snorkel + 10k unlabeled beats the small fully-supervised set.)

### 5. Relevance to ML Practice

**Where weak supervision shines.**
- Domains with lots of unlabeled data and scarce gold labels.
- Domains with dense programmatic knowledge (regex, ontologies, pretrained models, KB lookups).
- Tasks that need frequent re-labeling as the spec changes — you modify LFs, not relabel.

**Common deployment pattern.** A team writes initial LFs, fits a label model, trains a downstream classifier, evaluates on a small gold set, and iterates: identifies errors, writes or refines LFs, re-fits. The LFs are version-controlled code, not static labels.

**When not to use it.**
- No programmatic priors (pure vision tasks with no metadata, novel domains without any rules).
- Tasks where a single LF's bias is shared by all plausible LFs (no diversity means no gain from aggregation).
- When you genuinely have abundant labels — a high-quality labeled dataset beats weak supervision.

**Alternatives.**
- **Semi-supervised learning**: uses a small labeled set plus unlabeled data but no external rules.
- **Transfer learning**: uses a pretrained model as the single "labeler."
- **Data programming with LLMs**: use LLM prompts as LFs. This is increasingly common and is called *prompted weak supervision*; it can be very effective but introduces correlated errors across LFs that share a model.

**Trade-offs.**
- **Bias vs. variance**: many diverse LFs reduce variance but increase risk of correlated bias.
- **Interpretability**: the LF code is interpretable; the final classifier is not.
- **Maintenance**: LFs are living code. Changing data requires updating LFs, not relabeling.

### 6. Common Pitfalls & Misconceptions

**1. "I need hundreds of LFs."** Usually 10–30 diverse LFs are sufficient. Quality matters more than quantity.

**2. "LFs with low coverage are useless."** A 5%-coverage, 95%-precision LF can be very valuable; it identifies a high-confidence subset.

**3. "The label model fixes biased LFs."** It does not. If all LFs systematically miss a class, the label model has no way to recover that class.

**4. "Correlated LFs are fine because the label model will figure it out."** Only if you model the correlation. Vanilla Snorkel assumes conditional independence; correlated LFs bias the estimated accuracies.

**5. "I don't need a gold test set if I'm using Snorkel."** You absolutely do. Without one, you cannot tell if your pipeline is working.

**6. "Snorkel replaces the model — the noisy labels are the output."** No. The noisy labels train a downstream model that generalizes beyond the LFs.

---

## Active Learning

### 1. Motivation & Intuition

If you have a fixed labeling budget, not every unlabeled example is equally useful. Labeling a clear-cut example the model already gets right contributes almost nothing. Labeling an example the model is confused about forces the model to refine its boundary.

Active learning formalizes this: use the model itself to choose which examples to label next. By selecting the examples that will most reduce uncertainty or error, active learning typically reaches the same accuracy as random labeling with 2–10× fewer labels.

A concrete example: for a medical image diagnosis task with 100,000 unlabeled scans and a budget for 1,000 annotations, random labeling yields a baseline. Active learning — repeatedly labeling the 100 scans the current model is least sure about, then retraining — can match that baseline with 300 labels and substantially exceed it with 1,000.

This is the right tool when:
- Labels are expensive (medical, legal, physical experiments).
- Unlabeled data is abundant and cheap.
- You can retrain (or fine-tune) the model iteratively.

### 2. Conceptual Foundations

**The active learning loop.**
1. Start with a small labeled seed set and a large unlabeled pool.
2. Train a model on the labeled set.
3. Use the model to score all unlabeled examples by an **acquisition function**.
4. Select the top-$k$ scoring examples (the *batch*).
5. Label them (by human annotation).
6. Add them to the labeled set. Go to step 2.

**Acquisition functions — what to query.**

- **Uncertainty sampling**: pick examples where the model is most uncertain.
  - **Least confidence**: $\arg\min_x \max_y P(y \mid x)$. Pick examples whose top-1 probability is lowest.
  - **Margin**: $\arg\min_x [P(y_1 \mid x) - P(y_2 \mid x)]$. Pick examples where the top two classes are close.
  - **Entropy**: $\arg\max_x H(P(y \mid x)) = -\sum_y P(y\mid x) \log P(y \mid x)$. Pick examples with highest predictive entropy.

- **Query by Committee (QBC)**: train an ensemble; pick examples where the ensemble members disagree most. Disagreement measures include vote entropy and KL divergence to the consensus.

- **Expected Model Change**: pick examples that would cause the largest gradient update if labeled. Approximates "the example that will most change my model."

- **Expected Error Reduction**: pick examples that, after labeling, would most reduce the expected error on the remaining pool. Theoretically optimal, computationally expensive (requires training a model for every candidate × every possible label).

- **BALD (Bayesian Active Learning by Disagreement)**: information-theoretic, picks examples where the label is maximally informative about the model parameters.

- **Diversity / Core-set methods**: rather than pure uncertainty, pick batches that cover the input space. Avoids redundant queries on clustered hard examples.

**Batch active learning.** In practice you label in batches, not one at a time, because retraining after every label is infeasible. The naive approach (top-$k$ by uncertainty) selects $k$ very similar examples. Good batch AL balances informativeness and diversity:
- **BatchBALD**: extends BALD with combinatorial selection accounting for joint mutual information.
- **Coreset**: selects a batch that is a good k-center cover of the pool in feature space.
- **BADGE**: uses gradient embeddings and k-means++ seeding to pick diverse, high-magnitude gradients.

**Underlying assumptions.**
- The model is a reasonable proxy for "hard examples."
- You can retrain/fine-tune between rounds.
- Labels are iid given features (no feedback loops to the world).

**What breaks when assumptions violate.**
- **Cold start**: the initial model is bad, so its uncertainty estimates are unreliable. Start with random labeling, switch to AL after ~1 round.
- **Outliers**: uncertainty sampling aggressively queries outliers (which are "hard"). These may be label noise, not informative signal. Filter or use density-weighted queries.
- **Distribution shift**: if the label distribution on queried examples differs from deployment, the resulting model may over-fit to hard cases and under-fit typical cases.

### 3. Mathematical Formulation

**Least confidence acquisition.**
$$
a_{\text{LC}}(x) = 1 - \max_y P(y \mid x; \theta).
$$
Select $\arg\max_x a_{\text{LC}}(x)$.

**Entropy.**
$$
a_H(x) = -\sum_y P(y \mid x; \theta) \log P(y \mid x; \theta).
$$

**Query by Committee — vote entropy.** Ensemble of $C$ classifiers. Let $V(y, x) = \sum_{c=1}^C \mathbb{1}[h_c(x) = y]$.
$$
a_{\text{VE}}(x) = -\sum_y \frac{V(y, x)}{C} \log \frac{V(y, x)}{C}.
$$

**BALD** (mutual information between label and parameters):
$$
a_{\text{BALD}}(x) = H\left[\mathbb{E}_\theta[P(y \mid x, \theta)]\right] - \mathbb{E}_\theta\left[H[P(y \mid x, \theta)]\right].
$$
This is **(predictive entropy)** − **(expected entropy under each parameter draw)**. High when the model is uncertain overall (first term high) but each individual parameter draw is confident (second term low) — i.e., when the ensemble disagrees confidently. Sometimes called "epistemic uncertainty" disentangled from aleatoric.

**Expected Error Reduction** (binary, for concreteness). For candidate $x^*$, assume labeling it $y$ changes the model to $\theta^{+(x^*, y)}$. Then the expected error reduction is
$$
a_{\text{EER}}(x^*) = \mathbb{E}_{y^* \sim P(y \mid x^*; \theta)} \left[ \sum_{x \in \mathcal{U}} \left(1 - \max_y P(y \mid x; \theta^{+(x^*, y^*)})\right) \right].
$$
We want to minimize post-labeling expected error on the unlabeled pool $\mathcal{U}$. Computational cost: $O(|U|^2 \times K)$ per iteration — tractable only for small pools.

**BatchBALD** generalizes BALD to sets:
$$
a_{\text{BatchBALD}}(\{x_1, \ldots, x_b\}) = I(Y_1, \ldots, Y_b ; \theta \mid x_1, \ldots, x_b).
$$
The joint mutual information naturally penalizes redundant points: two near-identical points yield almost the same information as one, so the set score does not double.

### 4. Worked Examples

**Small concrete pool.** Suppose a binary text classifier with a logistic regression on TF-IDF has produced predicted probabilities on a pool of 10 unlabeled examples:

| Example | P(y=1) | Least conf | Entropy |
|---|---|---|---|
| x₁ | 0.99 | 0.01 | 0.08 |
| x₂ | 0.51 | 0.49 | 0.999 |
| x₃ | 0.95 | 0.05 | 0.29 |
| x₄ | 0.60 | 0.40 | 0.97 |
| x₅ | 0.05 | 0.05 | 0.29 |
| x₆ | 0.88 | 0.12 | 0.53 |
| x₇ | 0.49 | 0.49 | 0.9997 |
| x₈ | 0.30 | 0.30 | 0.88 |
| x₉ | 0.70 | 0.30 | 0.88 |
| x₁₀ | 0.02 | 0.02 | 0.14 |

Entropy ranks: x₇, x₂, x₄, x₈/x₉ (tie), … Top-1 query is x₇. Least confidence also picks x₂/x₇. Pure margin: x₇ (p closest to 0.5).

Batch top-3: {x₇, x₂, x₄}. If x₂ and x₇ are near-duplicates in feature space, BatchBALD or a coreset selector would pick {x₇, x₄, x₈} instead to maximize diversity.

**Full simulated run.** Baseline: random labeling of 100 examples yields validation accuracy 0.72. Uncertainty sampling over 10 rounds of 10 labels each: 0.76 after 50 labels, 0.79 after 100, 0.82 after 150. Random labeling needs ~300 labels to match 0.82.

### 5. Relevance to ML Practice

**When active learning is worth it.**
- Labels are expensive relative to compute (medical, legal, bespoke domains).
- The unlabeled pool is much larger than the labeling budget.
- Retraining is feasible (minutes to hours, not days).

**When not to use it.**
- Labels are cheap — simpler to just label everything relevant.
- The pool is small — the overhead of acquisition dominates.
- The model is a frozen foundation model you cannot retrain — AL loses its feedback loop. (Though AL can still drive annotation budgets for classifier heads and evaluation sets.)

**Combining with other strategies.**
- **AL + weak supervision**: use weak labels to bootstrap, then use AL to target the areas where the label model is uncertain.
- **AL + semi-supervised**: use consistency regularization on unlabeled points and AL to pick the most informative ones to label.
- **AL for evaluation**: instead of randomly sampling a test set, actively sample the subset that most informatively estimates model performance (stratified sampling based on predicted probability).

**Trade-offs.**
- **Sample efficiency vs. label distribution shift**: AL produces a biased labeled set skewed toward hard cases. Do not use this set as your evaluation distribution.
- **Acquisition cost vs. quality**: EER is theoretically optimal but expensive. Uncertainty sampling is nearly as effective in practice with much less compute.
- **Batch size vs. iteration efficiency**: small batches need frequent retraining; large batches need diversification. Typical industrial setup: batches of 100–1000.

### 6. Common Pitfalls & Misconceptions

**1. "Active learning always beats random."** Not always. In the early rounds (cold start) the model is too weak to identify useful examples. Random baselines are important.

**2. "Uncertainty sampling alone is enough."** In batch settings it picks redundant, similar-looking hard examples. Always diversify.

**3. "The labeled set is a good test set."** It is not. It is selected, not iid. Maintain a separate randomly-sampled test set.

**4. "Uncertainty catches outliers and label noise; we should fix that."** Sometimes the "uncertain" examples are mislabeled in your seed set or are genuine outliers. Review the queried examples before labeling.

**5. "Label one example at a time."** You almost never do this in practice. Batch AL is the production reality.

**6. "The budget determines the model accuracy."** Often the bottleneck is annotator throughput or distributional coverage, not the budget.

---

## Semi-Supervised Learning

### 1. Motivation & Intuition

Suppose you have 1,000 labeled examples and 100,000 unlabeled examples. Active learning asks: which examples should I label next? Semi-supervised learning (SSL) asks a different question: given this asymmetry, can I extract signal from the unlabeled data *without* labeling any more of it?

The intuition is that the unlabeled data tells you about the input distribution $P(X)$, and assumptions about how $P(Y \mid X)$ relates to $P(X)$ let you use that. If similar inputs have similar labels (smoothness assumption), then the structure of $P(X)$ constrains the possible decision boundaries, and the model is rewarded for respecting that structure even where no labels are available.

Concrete example: in a MNIST digit classification problem with 100 labeled and 60,000 unlabeled, methods like FixMatch (2020) reach ~99% accuracy — within 0.5% of fully supervised. This was a decisive demonstration that on tasks with strong inductive biases (images, speech), SSL substantially closes the gap between a small labeled set and a full one.

### 2. Conceptual Foundations

**Core SSL assumptions (typically one or more must hold).**
- **Smoothness**: if two inputs are close in input space, their labels are likely the same.
- **Cluster**: data tends to form clusters, and points in the same cluster share a label.
- **Manifold**: high-dimensional data lies on a low-dimensional manifold, and labels vary smoothly along it.
- **Low-density separation**: the decision boundary passes through low-density regions of $P(X)$.

**Families of methods.**

**(A) Self-training / Pseudo-labeling.**
1. Train a model on labeled data.
2. Predict labels on unlabeled data.
3. Add high-confidence predictions to the labeled set as "pseudo-labels."
4. Retrain.
5. Repeat.

Variants: Noisy Student (iterative self-training with stronger student architectures and input noise), MixMatch, FixMatch.

**(B) Consistency regularization.** Key idea: for any unlabeled example, two slightly different views of it (augmentations, dropout perturbations) should yield similar predictions. Enforce this via a consistency loss:
$$
\mathcal{L}_{\text{cons}} = \|f(x) - f(x')\|^2
$$
where $x'$ is a perturbed version of $x$. This is a regularizer that encodes the smoothness assumption directly.

**(C) Entropy minimization.** Train the model to produce low-entropy (confident) predictions on unlabeled data. Drives the decision boundary into low-density regions.

**(D) Hybrid methods** (the current state of the art):
- **FixMatch** (Sohn et al., 2020): for each unlabeled example, produce a weakly-augmented view and a strongly-augmented view. Use the model's prediction on the weakly-augmented view as a pseudo-label if its max probability exceeds a threshold. Train the strongly-augmented view to match that pseudo-label.
- **MixMatch**: averages predictions over multiple augmentations and applies sharpening.
- **UDA (Unsupervised Data Augmentation)**: consistency between augmented and unaugmented views.

**(E) Graph-based SSL.** Construct a graph over labeled and unlabeled nodes, propagate labels along edges (label propagation, label spreading). Strongest when the data has natural graph structure.

**Underlying assumptions.** Each method inherits the smoothness/cluster/manifold assumption. If those fail, SSL can actually hurt. A specific failure: under class imbalance in unlabeled data, pseudo-labeling amplifies the majority class — the model gets more confident on common classes and labels all ambiguous points as majority.

**What breaks when assumptions violate.**
- **Distribution shift between labeled and unlabeled**: the unlabeled data is no longer informative about the same $P(Y\mid X)$.
- **Noisy pseudo-labels with confirmation bias**: self-training amplifies its own mistakes.
- **Class imbalance**: threshold-based pseudo-labeling ignores minority classes.

### 3. Mathematical Formulation

**Notation.** Labeled set $\mathcal{D}_L = \{(x_i, y_i)\}_{i=1}^{n_L}$, unlabeled set $\mathcal{D}_U = \{x_j\}_{j=1}^{n_U}$, model $f_\theta(x) \in \Delta^{K-1}$ (predicted distribution over $K$ classes).

**Self-training / pseudo-labeling.** Standard form:
$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{n_L} \sum_{(x,y)\in\mathcal{D}_L} \ell(f_\theta(x), y)}_{\text{supervised loss}} + \lambda_u \underbrace{\frac{1}{n_U} \sum_{x\in\mathcal{D}_U} \mathbb{1}[\max_k f_\theta(x)_k > \tau] \cdot \ell(f_\theta(x), \hat{y})}_{\text{pseudo-label loss}}
$$
where $\hat{y} = \arg\max_k f_\theta(x)_k$ and $\tau$ is a confidence threshold (commonly 0.95).

**Consistency regularization.** Let $T$ be an augmentation distribution. Loss:
$$
\mathcal{L}_{\text{cons}}(\theta) = \mathbb{E}_{x \in \mathcal{D}_U, t_1, t_2 \sim T}\left[ d(f_\theta(t_1(x)), f_\theta(t_2(x))) \right].
$$
$d$ is a divergence: squared $L_2$, cross-entropy, KL. Often one view is passed through a "teacher" with stop-gradient.

**FixMatch loss.**
$$
\mathcal{L}(\theta) = \mathcal{L}_s + \lambda_u \mathcal{L}_u,
$$
$$
\mathcal{L}_u = \frac{1}{\mu B} \sum_{b=1}^{\mu B} \mathbb{1}\left[\max_k p_m(y | \alpha(u_b)) > \tau\right] \cdot H(\hat{y}_b, p_m(y | \mathcal{A}(u_b))),
$$
where $\alpha$ is weak augmentation, $\mathcal{A}$ is strong augmentation, $\hat{y}_b = \arg\max p_m(y|\alpha(u_b))$, $\tau = 0.95$ typically. The key structural feature: pseudo-labels come from *weak* augmentation, gradients flow through *strong* augmentation. This implicitly encodes the smoothness assumption (weak aug ≈ original; strong aug is a perturbed view that should still map to the same label).

**Entropy minimization.** Add a regularizer:
$$
\mathcal{L}_{\text{ent}}(\theta) = \frac{1}{n_U} \sum_{x \in \mathcal{D}_U} H(f_\theta(x)) = -\frac{1}{n_U} \sum_x \sum_k f_\theta(x)_k \log f_\theta(x)_k.
$$

This pushes predictions away from the uniform distribution and towards one-hot — which is exactly what you want if the smoothness assumption holds, but is pathological if $P(Y\mid X)$ is genuinely high-entropy.

**Label propagation.** Build an affinity matrix $W$ over all points (labeled + unlabeled). Define the graph Laplacian $L = D - W$. Label-propagation minimizes
$$
\sum_{i \in L} (f(x_i) - y_i)^2 + \mu f^\top L f,
$$
which solves to $f = (I - (1-\alpha) S)^{-1} Y$ for normalized propagation matrix $S$.

### 4. Worked Examples

**Pseudo-labeling on a binary text task.** 100 labeled tweets (50 positive, 50 negative), 50,000 unlabeled.

Round 0: Train a logistic regression → 0.72 accuracy on held-out test.
Round 1: Predict on 50,000 unlabeled. 18,000 have max probability > 0.9. Take those as pseudo-labels. Retrain on 100 + 18,000. Accuracy → 0.76.
Round 2: Predict again. Now 28,000 pass threshold. Retrain. Accuracy → 0.78.
Round 3: 31,000 pass. Accuracy → 0.78 (plateau).

Notes:
- Performance gained 6 absolute points from free (unlabeled) data.
- Plateau reflects the model running out of genuinely confident new examples it had been getting wrong.
- Risk: if the 100 labeled examples were biased (e.g., all from one topic), the pseudo-labels extend that bias. Always sanity-check by evaluating on a diverse held-out test set.

**FixMatch on a toy image task.** 250 labeled images from CIFAR-10, 49,750 unlabeled. Training with FixMatch's weak/strong augmentation protocol:
- Supervised-only baseline (250 labels): ~45% accuracy.
- Fully supervised (all 50,000 labels): ~95%.
- FixMatch (250 labels + 49,750 unlabeled): ~86%.

Nearly a 40-point gain from SSL. This kind of gap is why SSL matters — it makes small labeled sets behave more like medium ones.

### 5. Relevance to ML Practice

**When SSL works well.**
- Abundant unlabeled data from the same distribution as labeled.
- Strong inductive biases are available (CNNs for images, pretrained encoders for text).
- Reasonable class balance.
- Augmentation strategies that preserve class identity (this is why SSL is dominant in vision).

**When SSL struggles.**
- Tabular data without natural augmentations.
- Extreme class imbalance (threshold-based pseudo-labeling ignores minorities).
- Unlabeled data drawn from a different distribution (SSL can hurt; this is sometimes called "negative transfer").
- Very small labeled sets where the initial model is badly miscalibrated.

**Relationship to other paradigms.**
- **Self-supervised pretraining** (SimCLR, MoCo, BYOL, BERT MLM) is a precursor to SSL. Pretrain on unlabeled data, then fine-tune with labels. Often combined: self-supervised pretraining + SSL fine-tuning.
- **SSL vs. active learning**: complementary. AL picks what to label; SSL uses what you didn't.
- **SSL vs. weak supervision**: SSL uses the model as its own labeler. WS uses external rules. Hybrid systems (e.g., Snorkel + consistency regularization) exist.

**Trade-offs.**
- **Sample efficiency vs. complexity**: SSL algorithms are more complex to tune (confidence thresholds, augmentation strength, loss weights).
- **Bias amplification vs. data leverage**: pseudo-labels can amplify training-set bias. Monitor class balance of pseudo-labels.
- **Compute cost**: consistency regularization doubles the forward passes per batch.

### 6. Common Pitfalls & Misconceptions

**1. "More unlabeled data is always better."** Only if it is from the same distribution. Off-distribution unlabeled data can hurt.

**2. "SSL is a replacement for labeled data."** No — it is a multiplier. Without some labeled data and a decent initial model, SSL has nothing to bootstrap from.

**3. "Confidence thresholding solves noisy pseudo-labels."** It reduces but does not eliminate them. Miscalibrated models produce confidently wrong predictions; the threshold filters some but not all.

**4. "FixMatch works on any task."** Its magic is augmentation-dependent. On tabular data without meaningful weak/strong augmentation, it degenerates to vanilla pseudo-labeling.

**5. "We should use SSL because it's cheap."** SSL introduces complexity and failure modes. Run a labeled baseline first; if it's good enough, skip SSL.

**6. "Entropy minimization is always beneficial."** It's an assumption of the cluster/low-density structure. On high-entropy true distributions (e.g., ambiguous multi-label text), it causes over-confidence.

---

# PART III — LABEL QUALITY ASSESSMENT

## Inter-Annotator Agreement

### 1. Motivation & Intuition

Suppose two annotators independently label 1,000 documents as spam or ham. They agree on 900. Is that good? Bad? Compared to what?

The naive answer — 90% agreement — ignores the fact that some agreement happens purely by chance. If both annotators labeled every document as ham (maybe because most documents actually are ham), they'd "agree" 95% of the time without looking at the data at all. So 90% agreement is actually worse than random guessing in a skewed distribution.

Inter-annotator agreement (IAA) metrics correct for chance. They answer the question "how much do the annotators agree *beyond what we'd expect by random chance*?"

These metrics serve three practical purposes:
1. **Validate task design.** Low IAA usually means your labeling guidelines are ambiguous, not that your annotators are bad.
2. **Bound model performance.** No model can exceed human agreement on its own task. If IAA is 0.85, reporting 0.95 model accuracy on labels from those annotators is suspect.
3. **Identify which annotators to trust.** Per-annotator agreement flags unreliable workers.

The most famous metric is **Cohen's kappa** (κ), introduced in 1960. Its multi-annotator generalization is **Fleiss' kappa**.

### 2. Conceptual Foundations

**Observed agreement ($p_o$).** The fraction of items on which annotators give the same label.

**Expected (chance) agreement ($p_e$).** The fraction of items on which we'd expect agreement if each annotator independently drew labels from their own marginal label distribution.

**Kappa.** The chance-corrected agreement:
$$
\kappa = \frac{p_o - p_e}{1 - p_e}.
$$
Interpretation:
- $\kappa = 1$: perfect agreement.
- $\kappa = 0$: agreement no better than chance.
- $\kappa < 0$: worse than chance (systematic disagreement).

**Landis–Koch interpretation benchmarks** (common in the literature; these are conventions, not universal truths):
- 0.00–0.20: slight.
- 0.21–0.40: fair.
- 0.41–0.60: moderate.
- 0.61–0.80: substantial.
- 0.81–1.00: almost perfect.

For ML tasks, IAA < 0.6 usually means the task is ill-defined; 0.7–0.8 is typical for non-trivial tasks; 0.85+ is excellent.

**Variants of kappa.**
- **Cohen's κ**: two annotators, categorical labels.
- **Fleiss' κ**: $n$ annotators, each item labeled by the same $n$ annotators (or a fixed $n$ from a larger pool). Not the direct generalization of Cohen's κ — it uses a different chance model.
- **Krippendorff's α**: most general — handles any number of annotators, missing data, and multiple data types (nominal, ordinal, interval, ratio). Widely used in content analysis.
- **Weighted κ (Cohen's)**: for ordinal labels, penalizes near-misses less than far-off disagreements.
- **Gwet's AC1**: an alternative to κ that is more robust to the "kappa paradox" in skewed distributions.

**The kappa paradox.** When marginals are highly skewed (e.g., 95% of documents are ham), κ can be very low even when $p_o$ is very high. This is because $p_e$ is also very high, so the ratio $(p_o - p_e)/(1 - p_e)$ is dominated by the small denominator. Two annotators agreeing 95% of the time with 95%–5% class prevalence may have κ ≈ 0. This is a real phenomenon that confuses practitioners; Gwet's AC1 addresses it by using a different chance-correction formula.

**Underlying assumptions.**
- Categorical data (for basic κ).
- Annotators are exchangeable (for Fleiss').
- Items are independent.
- The chance model (independent random draws from marginals) is reasonable.

**What breaks when assumptions violate.**
- **Skewed marginals**: κ paradox (discussed above).
- **Correlated annotators**: if two annotators confer before labeling, their agreement is not independent — κ over-estimates agreement quality.
- **Ordinal labels treated as nominal**: off-by-one disagreements count as full disagreement. Use weighted κ.

### 3. Mathematical Formulation

**Cohen's kappa.** Two annotators label $n$ items into $K$ categories. Let $p_o$ be observed agreement rate:
$$
p_o = \frac{1}{n} \sum_{i=1}^n \mathbb{1}[\text{annotator } 1 \text{ and } 2 \text{ agree on item } i] = \sum_{k=1}^K P_{kk},
$$
where $P_{jk}$ is the fraction of items annotator 1 labeled $j$ and annotator 2 labeled $k$.

Let $p_{1,k}$ = fraction of items annotator 1 labels $k$, and $p_{2,k}$ the same for annotator 2. Expected agreement under independence:
$$
p_e = \sum_{k=1}^K p_{1,k} \cdot p_{2,k}.
$$

Then
$$
\kappa = \frac{p_o - p_e}{1 - p_e}.
$$

**Fleiss' kappa.** $n$ items, $N$ total annotators, each item labeled by $m$ of them (same $m$ per item). Let $n_{ik}$ = number of annotators who assigned category $k$ to item $i$.

Per-item agreement:
$$
P_i = \frac{1}{m(m-1)} \left(\sum_{k=1}^K n_{ik}^2 - m\right).
$$

This is the probability that two randomly chosen annotators agree on item $i$.

Average observed agreement:
$$
\bar{P} = \frac{1}{n} \sum_{i=1}^n P_i.
$$

Marginal frequency of category $k$:
$$
p_k = \frac{1}{nm} \sum_{i=1}^n n_{ik}.
$$

Expected agreement:
$$
\bar{P}_e = \sum_{k=1}^K p_k^2.
$$

Fleiss' kappa:
$$
\kappa_F = \frac{\bar{P} - \bar{P}_e}{1 - \bar{P}_e}.
$$

**Weighted kappa (Cohen's).** For ordinal labels. Let $w_{jk}$ be a weight matrix ($w_{jj} = 0$ for identical labels, larger values for more distant disagreements). Linear weights: $w_{jk} = |j - k|/(K-1)$. Quadratic: $w_{jk} = (j-k)^2/(K-1)^2$.

$$
\kappa_w = 1 - \frac{\sum_{jk} w_{jk} P_{jk}}{\sum_{jk} w_{jk} p_{1,j} p_{2,k}}.
$$

Quadratic weighted kappa is what Kaggle uses in many classification competitions for ordinal targets.

**Krippendorff's alpha.** Define a distance metric $\delta$ appropriate to the data type. For nominal: $\delta(a, b) = \mathbb{1}[a \ne b]$. For interval: $\delta(a, b) = (a - b)^2$.

Observed disagreement:
$$
D_o = \frac{1}{\sum_i m_i (m_i - 1)} \sum_i \sum_{j \ne k} n_{ij} n_{ik} \delta(j, k) / (m_i - 1),
$$
(with care over the exact formula; several equivalent forms exist).

Expected disagreement:
$$
D_e = \frac{1}{n(n-1)} \sum_{j, k} n_j n_k \delta(j, k).
$$

$$
\alpha = 1 - \frac{D_o}{D_e}.
$$

The machinery is more involved but the pattern is identical: $1 - (\text{observed}/\text{expected})$.

### 4. Worked Examples

**Cohen's kappa, numerical.** Two annotators label 100 documents as positive (P) or negative (N). Confusion matrix:

|  | Ann 2: P | Ann 2: N | Row total (Ann 1) |
|---|---|---|---|
| Ann 1: P | 45 | 5 | 50 |
| Ann 1: N | 10 | 40 | 50 |
| Col total (Ann 2) | 55 | 45 | 100 |

Observed agreement: $p_o = (45 + 40)/100 = 0.85$.

Marginals: $p_{1,P} = 0.50$, $p_{2,P} = 0.55$, $p_{1,N} = 0.50$, $p_{2,N} = 0.45$.

Expected agreement: $p_e = 0.50 \cdot 0.55 + 0.50 \cdot 0.45 = 0.275 + 0.225 = 0.50$.

$$
\kappa = \frac{0.85 - 0.50}{1 - 0.50} = \frac{0.35}{0.50} = 0.70.
$$

Substantial agreement.

**The kappa paradox.** Two annotators label 100 documents as spam (S) or ham (H), where H is dominant.

|  | Ann 2: S | Ann 2: H |
|---|---|---|
| Ann 1: S | 1 | 4 |
| Ann 1: H | 3 | 92 |

$p_o = (1 + 92)/100 = 0.93$. Looks great.

Marginals: $p_{1,S} = 0.05$, $p_{2,S} = 0.04$, $p_{1,H} = 0.95$, $p_{2,H} = 0.96$.

$p_e = 0.05 \cdot 0.04 + 0.95 \cdot 0.96 = 0.002 + 0.912 = 0.914$.

$$
\kappa = \frac{0.93 - 0.914}{1 - 0.914} = \frac{0.016}{0.086} \approx 0.186.
$$

κ = 0.19 despite 93% agreement. The annotators are not really agreeing beyond chance; they're just both correctly noting that most documents are ham.

Remedies: report $p_o$ alongside κ; consider Gwet's AC1; stratify the analysis by class.

**Fleiss' kappa numerical.** 5 annotators label 4 items into 3 categories (A, B, C).

| Item | # labeling A | # labeling B | # labeling C |
|---|---|---|---|
| 1 | 5 | 0 | 0 |
| 2 | 1 | 4 | 0 |
| 3 | 0 | 2 | 3 |
| 4 | 3 | 2 | 0 |

$m = 5$, $n = 4$, $K = 3$.

$P_1 = (25 + 0 + 0 - 5)/(5 \cdot 4) = 20/20 = 1$.
$P_2 = (1 + 16 + 0 - 5)/20 = 12/20 = 0.6$.
$P_3 = (0 + 4 + 9 - 5)/20 = 8/20 = 0.4$.
$P_4 = (9 + 4 + 0 - 5)/20 = 8/20 = 0.4$.

$\bar{P} = (1 + 0.6 + 0.4 + 0.4)/4 = 0.6$.

Marginals: total labels per category = A: 9, B: 8, C: 3. Total assignments = $n m = 20$.
$p_A = 9/20 = 0.45$, $p_B = 0.40$, $p_C = 0.15$.

$\bar{P}_e = 0.45^2 + 0.40^2 + 0.15^2 = 0.2025 + 0.16 + 0.0225 = 0.385$.

$$
\kappa_F = \frac{0.6 - 0.385}{1 - 0.385} = \frac{0.215}{0.615} \approx 0.350.
$$

Fair agreement. Item 3 (where annotators split 2-3) pulled the agreement down; item 1 (unanimous) was strong.

### 5. Relevance to ML Practice

**Label quality pipelines.** Every serious labeling project tracks per-annotator κ against a gold standard (per-annotator accuracy) and inter-annotator κ per batch (cohort agreement). Both should be reported in model documentation.

**Task calibration.** In early stages of an annotation project, run a pilot batch of 200–500 items with 3+ annotators. If κ < 0.6, do not scale — refine guidelines, add examples, decompose the task.

**Model ceiling analysis.** Report IAA on the same data the model is evaluated on. If IAA = 0.85 and model κ-with-gold = 0.82, the model is effectively at human-level. Further modeling effort may yield diminishing returns.

**Ordinal tasks.** Use weighted κ or Krippendorff's α. Unweighted κ on ordinal data (e.g., Likert ratings) severely underrates agreement.

**Trade-offs.**
- **κ vs. raw agreement**: κ corrects for chance but can be misleading under skew. Always report both.
- **Cohen vs. Fleiss vs. Krippendorff**: Cohen for two annotators, Fleiss for fixed panel, Krippendorff for flexibility (missing data, different data types). Krippendorff is the most general but less intuitive.
- **Simple vs. weighted**: if your label space has a natural ordering or semantic distance, use weighted.

### 6. Common Pitfalls & Misconceptions

**1. "High κ means the labels are correct."** No. κ measures reliability (annotators agree), not validity (annotators are right). Two annotators can share a misconception and agree on a wrong label.

**2. "κ is interpretable on an absolute scale."** Interpretation depends on the task difficulty and label distribution. κ = 0.7 on medical imaging is excellent; κ = 0.7 on coin flips is suspicious.

**3. "If κ is low, the annotators are bad."** Usually it's the *guidelines* that are bad. Improve task design before blaming people.

**4. "Fleiss' κ is Cohen's κ with more annotators."** They use different chance models. Fleiss' assumes all annotators share the same marginal; Cohen's does not.

**5. "Weighted κ is always better."** Only if the label space is ordinal. On nominal labels, weights don't make sense.

**6. "We don't need IAA — we only have one annotator per item."** You absolutely need at least a sample of multi-labeled items to calibrate. Otherwise you have no measure of label quality.

---

## Gold Standard Evaluation & Adjudication

### 1. Motivation & Intuition

When every item is labeled once by one annotator, you don't know if any given label is correct. When every item is labeled by five annotators, you're paying a lot. The gold standard workflow finds a middle ground: a small, carefully-curated subset of items is labeled exhaustively and adjudicated by experts to produce high-confidence "ground truth," and this gold set is used to evaluate both the bulk annotation process and the model.

Beyond cost savings, gold sets serve distinct roles:
- **Quality control** for annotators (the gold items are randomly inserted into the workflow; performance on them measures the annotator).
- **Task calibration** (the gold set is the canonical reference for what a "correct" label looks like).
- **Model evaluation** (models are tested on gold because it is the most trusted subset).
- **Drift detection** (if annotators or models begin to drift, gold agreement drops).

### 2. Conceptual Foundations

**Gold standard construction.**
1. Select candidate items (typically stratified: balanced across classes, balanced across difficulty strata if known).
2. Have multiple expert annotators label each item independently.
3. Adjudicate disagreements:
   - **Consensus discussion**: annotators meet to resolve.
   - **Adjudicator tiebreaker**: a senior expert resolves.
   - **Majority vote with discussion only at ties**.
4. Document the rationale for each gold label, especially for borderline cases.

**Gold set size.** Trade-offs:
- Too small: high variance in reported metrics.
- Too large: costs money, reduces coverage of bulk labeling.
- Rule of thumb for ML evaluation: at least 500–1000 items per class for a stable F1 estimate. For annotator QC: 5–10% of the bulk stream should be gold.

**Adjudication protocols.**

- **Sequential**: annotator A labels, annotator B labels, if they disagree annotator C decides.
- **Hierarchical**: multiple rounds with progressively more senior reviewers.
- **Committee**: all annotators discuss; consensus required.
- **Expert re-review**: flagged items (by disagreement, by low model confidence, by active-learning uncertainty) are routed to experts.

**Calibration.** Regular calibration sessions where annotators label the same items and then discuss disagreements are crucial for maintaining guideline alignment. Without them, annotator drift accumulates.

**Underlying assumptions.**
- Experts can produce a more reliable label than non-experts.
- Adjudication reduces rather than introduces bias.
- The gold set is representative of the evaluation distribution.

**What breaks when assumptions violate.**
- **Expert bias**: if experts share a training background, their agreement may reflect shared preconceptions rather than ground truth.
- **Gold set drift from production**: if you freeze a gold set and the world changes, the gold becomes an increasingly poor stand-in.
- **Gold set leakage**: if gold items appear in training, the model memorizes them and evaluation becomes meaningless. Guard against this aggressively.

### 3. Mathematical Formulation

**Gold label as consensus estimator.** For $m$ expert annotators labeling item $i$ with labels $\hat{y}_{i,1}, \ldots, \hat{y}_{i,m}$, the adjudicated label $y_i^*$ can be modeled as:
$$
y_i^* = \arg\max_k P(y = k \mid \hat{y}_{i,1}, \ldots, \hat{y}_{i,m}).
$$
This is exactly the same form as Dawid–Skene's soft label; the difference is that for gold the procedure typically forces a hard decision through human adjudication rather than EM.

**Gold set as evaluation sample.** Let $y^*$ be gold labels, $\hat{y}$ be model predictions. Reported metric (e.g., accuracy):
$$
\hat{A} = \frac{1}{n_G} \sum_{i=1}^{n_G} \mathbb{1}[\hat{y}_i = y_i^*].
$$

Standard error (for accuracy): $\text{SE} \approx \sqrt{\hat{A}(1-\hat{A})/n_G}$. For $n_G = 500$ and $A = 0.9$: SE ≈ 0.013, 95% CI ±0.026. For $n_G = 100$: SE ≈ 0.03, CI ±0.06. Small gold sets produce wide confidence intervals, which is often underappreciated.

**Gold set bias on sub-metrics.** If gold set has $n_{G,k}$ items of class $k$, the per-class recall has SE $\approx \sqrt{R_k(1-R_k)/n_{G,k}}$. Rare classes in a small gold set produce very unstable per-class metrics.

**Bounded evaluation.** If gold labels are themselves noisy with error rate $\eta$, the reported accuracy is bounded:
$$
A_{\text{reported}} \le 1 - \eta + \text{error correlations}
$$
(roughly). Quantitatively: for model accuracy $a$ and label error $\eta$ (independent), the observed match rate is $a(1-\eta) + (1-a)\eta/(K-1)$, which for binary is $a(1-\eta) + (1-a)\eta$. If $\eta = 0.05$, an actually-perfect model is measured at only $1 - \eta = 0.95$.

### 4. Worked Examples

**A text classification gold set.** Team labels 50,000 reviews with 3 annotators per item, majority vote. For 1,000 randomly selected items:
- Add 2 additional expert annotators (so 5 total per gold item).
- When the 5 annotators disagree, escalate to an adjudicator who reviews annotator rationales and decides.
- 78% of items have unanimous agreement; 17% have 4-of-5; 5% are adjudicated.

Of the 5% adjudicated (50 items), 30 get the majority label, 20 get the minority label. Post-adjudication, gold labels are used to:
- Compute per-annotator agreement with gold: Ann A: 0.94, Ann B: 0.88, Ann C: 0.75. Ann C is flagged for retraining.
- Evaluate the model: 85% accuracy on gold.
- Sanity-check the crowd labels: crowd majority agrees with gold on 91% of gold items.

The 9% gap between crowd and gold (on gold items) is a direct estimate of the crowd's error rate. The bulk labeled set is estimated at 91% accuracy overall.

**A subtle failure: gold set leaks.** Team finds that a data-loading script accidentally includes some gold items in training. After removing the leakage, test accuracy drops from 94% to 87%. The original "94%" was not evaluation — it was memorization. This is a class of failure that routinely plagues academic benchmarks (Northcutt et al. 2021 catalogued similar issues in ImageNet, CIFAR, and others).

### 5. Relevance to ML Practice

**When gold sets are essential.**
- Regulated domains (medical, financial) where evaluation must be defensible.
- Any benchmark used to pick between models — if the benchmark has 5% noise, model improvements below 5% are indistinguishable from noise.
- Annotator QC programs.

**When gold sets are insufficient.**
- When the evaluation question requires production feedback (online A/B, user behavior).
- When the task distribution shifts faster than you can re-adjudicate.

**Best practices.**
- Freeze the gold set version; track model performance across versions.
- Maintain a separate *challenge* gold set of hard or adversarial cases.
- Regularly re-adjudicate a fraction of the gold set to detect annotator drift.
- Track per-class SE alongside point estimates.

**Trade-offs.**
- **Gold size vs. label cost**: more gold means more cost but tighter CIs.
- **Adjudication rigor vs. throughput**: full consensus discussion is expensive; escalation-only is faster but less thorough.
- **Freezing gold vs. updating**: frozen gold enables time comparisons but becomes stale.

### 6. Common Pitfalls & Misconceptions

**1. "Gold is ground truth."** Gold is your best estimate of ground truth. It is not perfect.

**2. "The gold set is small, so let's not worry about its statistics."** Small gold sets produce wide CIs. Report them.

**3. "Adjudication always improves labels."** It can also introduce adjudicator bias. Rotate adjudicators or use committee decisions.

**4. "We evaluated on gold, so we're good."** Only if gold is distributionally representative of deployment. A stratified gold set that over-samples rare classes will give a distorted picture of operational accuracy.

**5. "Gold labels don't change."** If the task evolves, gold labels should too. Track gold versions like code.

**6. "Training on gold is fine as long as we leave some out for testing."** If you do hyperparameter tuning on a test set, it becomes a validation set. Gold test sets must be held out from all model development decisions.

---

## Confident Learning

### 1. Motivation & Intuition

Every other technique in this section assumes you can either trust your labels (gold) or explicitly model their noise (Dawid–Skene, Snorkel). Confident learning (CL) flips the script: given a trained classifier and observed labels, **find the examples where the label is probably wrong** and either remove or relabel them.

The key insight is that a well-calibrated model makes mistakes in a structured way. If the model confidently predicts class A for many examples whose label is B, some of those "B" labels are probably mislabeled A's. By examining the pattern of model probabilities vs. observed labels, you can estimate the joint distribution of (true label, observed label) — and thus identify which specific examples are likely label errors.

Confident learning was introduced by Northcutt et al. (2021) in "Confident Learning: Estimating Uncertainty in Dataset Labels" and implemented in the open-source library **cleanlab**. Its most eye-catching result: the commonly-used test sets of ImageNet, MNIST, CIFAR, Amazon Reviews, IMDb, and more contain 3.3% average labeling errors — and correcting them changes which models win on benchmarks.

Real use cases:
- Auditing a large labeled dataset for errors before training.
- Relabeling the most suspicious examples (instead of random rechecking).
- Correcting training labels to improve model accuracy.
- Producing cleaner evaluation sets.

### 2. Conceptual Foundations

**Setup.** Observed (noisy) labels $\tilde{y}_i \in \{1, \ldots, K\}$ for each of $n$ examples. True (unknown) labels $y_i^*$. Predicted class probabilities $\hat{p}_{i,k} = P(y_i^* = k \mid x_i)$ from a model that has been trained on $(\tilde{y}, x)$ using cross-validation (to ensure $\hat{p}$ is out-of-fold).

**The confident joint.** Construct a matrix $C_{y_i^*, \tilde{y}_i}$ estimating the joint count of (true label, observed label). For each observed label class $k$, compute a class-specific confidence threshold $t_k$ = average predicted probability of class $k$ on examples that are observed to be class $k$. Then count an example $i$ as "confidently of class $j$" if $\hat{p}_{i,j} \ge t_j$.

Define:
$$
C_{j, k} = \left| \left\{ i : \hat{p}_{i,j} \ge t_j, \; \tilde{y}_i = k, \; j = \arg\max_{l : \hat{p}_{i,l} \ge t_l} \hat{p}_{i,l} \right\} \right|.
$$

This reads as: the number of examples whose observed label is $k$ but which the model confidently places in class $j$. Off-diagonal entries represent likely label errors.

**Normalization.** Adjust each row so it sums to the observed per-class count:
$$
\tilde{C}_{j, k} = C_{j, k} \cdot \frac{|\{i : \tilde{y}_i = k\}|}{\sum_{j'} C_{j', k}}.
$$

This corrects for the fact that confidence thresholds under-count or over-count each class asymmetrically.

**Joint and marginal estimation.** Divide by $n$ to get an estimate of $P(y^*, \tilde{y})$. Diagonal: probability of correctly-labeled examples of each class. Off-diagonal: probability of (true, observed) mislabel pairs.

**Error identification methods.**
1. **Prune by Class (PBC)**: remove, for each class $k$, the $n \cdot \sum_{j \ne k} P(y^* = j, \tilde{y} = k)$ examples with lowest $\hat{p}_{i,k}$.
2. **Prune by Noise Rate (PBNR)**: for each off-diagonal $(j, k)$, prune the $n \cdot P(y^* = j, \tilde{y} = k)$ examples with highest margin $\hat{p}_{i,j} - \hat{p}_{i,k}$.
3. **Both (PBC ∩ PBNR)**: intersection of the two sets — highest precision.
4. **Confusion**: simple method: flag $i$ if $\arg\max_k \hat{p}_{i,k} \ne \tilde{y}_i$. Doesn't account for noise rate per class; often over-flags.

**Underlying assumptions.**
- **Class-conditional noise**: $P(\tilde{y} \mid y^*, x) = P(\tilde{y} \mid y^*)$. Label noise depends only on the true class, not on the features. Strong assumption; often violated in practice but CL is fairly robust.
- **Calibrated-enough predictions**: the model probabilities don't need to be perfectly calibrated, but should rank-order correctly. Cross-validation helps because the model hasn't seen any specific example during training.
- **Sufficient correct labels per class**: the thresholds $t_k$ are averages over class $k$ labels; with few examples, these are unstable.

**What breaks when assumptions violate.**
- **Feature-dependent noise** (systematic mislabeling of certain regions of feature space): CL can still detect errors, but the joint estimate is biased.
- **Severe class imbalance**: the minority-class threshold is estimated from few points; high variance, more false error flags.
- **Poor classifier**: if the model is badly calibrated or under-trained, the flagged errors are largely noise themselves.

### 3. Mathematical Formulation

**Notation.** $n$ examples, $K$ classes, observed labels $\tilde{y}$, out-of-fold predictions $\hat{p}_{i,k}$.

**Per-class threshold.**
$$
t_k = \frac{1}{|\{i : \tilde{y}_i = k\}|} \sum_{i : \tilde{y}_i = k} \hat{p}_{i, k}.
$$

An example is "confidently of class $j$" if $\hat{p}_{i,j} \ge t_j$. For examples confident in multiple classes, assign to the class with highest probability (tiebreaker).

**Confident joint.**
$
C_{j, k} = |\{ i : \hat{p}_{i, j} \ge t_j \text{ and } j = \arg\max_{l : \hat{p}_{i,l} \ge t_l} \hat{p}_{i,l} \text{ and } \tilde{y}_i = k \}|.
$

**Calibration step.** For each $k$, scale so that the row sums match the observed class counts:
$$
\hat{Q}_{j, k} = \frac{C_{j, k}}{\sum_{j'} C_{j', k}} \cdot \frac{|\{i : \tilde{y}_i = k\}|}{n}.
$$

$\hat{Q}$ is an estimate of the joint distribution $P(y^*, \tilde{y})$.

**Noise and inverse noise matrices.** The noise transition matrix $T_{j, k} = P(\tilde{y} = k \mid y^* = j)$ satisfies
$$
P(\tilde{y} = k) = \sum_j T_{j,k} P(y^* = j).
$$
Once $\hat{Q}$ is estimated, $\hat{T}$ follows from row-normalizing.

**Loss correction.** Given $\hat{T}$, the noise-corrected loss for cross-entropy is:
$$
\mathcal{L}_{\text{corr}}(f(x), \tilde{y}) = -\log \left( \sum_{j} T_{j, \tilde{y}} f_j(x) \right),
$$
where $f_j(x)$ is the model's predicted probability of class $j$. Training with $\mathcal{L}_{\text{corr}}$ on the noisy data is (asymptotically) equivalent to training on clean data. This is the **forward correction** method (Patrini et al., 2017).

### 4. Worked Examples

**Small numerical example.** $K = 3$ classes (A, B, C), $n = 30$ examples. Observed labels: 10 of each class. Out-of-fold predicted probabilities (schematic, only summaries shown):

- Mean $\hat{p}_A$ on examples with $\tilde{y} = A$: $t_A = 0.75$.
- Mean $\hat{p}_B$ on examples with $\tilde{y} = B$: $t_B = 0.70$.
- Mean $\hat{p}_C$ on examples with $\tilde{y} = C$: $t_C = 0.80$.

Suppose:
- Of 10 examples with $\tilde{y} = A$: 8 have $\hat{p}_A \ge 0.75$ (confidently A). 2 have $\hat{p}_B \ge 0.70$ and $\hat{p}_B > \hat{p}_A$ (confidently B).
- Of 10 examples with $\tilde{y} = B$: 7 confidently B, 1 confidently A, 2 neither.
- Of 10 examples with $\tilde{y} = C$: 9 confidently C, 1 confidently A.

Confident joint $C$ (rows = true-according-to-model, cols = observed):
$$
\begin{pmatrix}
8 & 1 & 1 \\
2 & 7 & 0 \\
0 & 0 & 9 \\
\end{pmatrix}
$$

Off-diagonal entries: 4 suspected label errors. For example, the (1, 2) entry = 1 means "1 example is confidently class A but labeled B". That specific example is a candidate for relabeling.

**Realistic scenario — auditing a benchmark.** A team has a 50,000-example sentiment dataset with 3 classes (positive, negative, neutral). They train a BERT model with 5-fold cross-validation and obtain out-of-fold probabilities. CL identifies ~1,500 (3%) of examples as likely mislabeled. Manual review of 200 randomly-sampled flagged examples:
- 140 are actual label errors (70% precision).
- 40 are genuinely ambiguous ("it's alright I guess" ← positive, negative, or neutral?).
- 20 are correctly labeled; the model was wrong.

The team relabels the 140 clear errors. They also add an "ambiguous" category to the annotation guidelines, flagging the 40 genuinely ambiguous cases for adjudication. The cleaned dataset produces a +1.2% improvement on the held-out test set.

### 5. Relevance to ML Practice

**Use cases.**
- Pre-training data audits: run CL, manually review flagged examples, fix errors.
- Continuous label monitoring: flag examples during inference where the model disagrees confidently with stored labels; feed back to annotators.
- Cleaning evaluation sets (highest priority — test set errors are most damaging).
- Combining with active learning: CL-flagged examples make excellent candidates for expert review.

**When to use it.**
- Large dataset where manual audit is infeasible.
- A reasonably good model is available or trainable.
- Labels were produced by a process known to be noisy (crowd, weak supervision, automated extraction).

**When not to use it.**
- Tiny datasets — model too weak to detect errors.
- Extreme class imbalance — threshold estimation unstable.
- Features that are themselves corrupted — CL detects label noise, not feature noise.

**Relationship to other techniques.**
- **CL vs. Dawid–Skene**: DS models annotator noise explicitly from multiple annotations per item; CL works from just one observed label per item plus a model. CL is applicable in more settings.
- **CL vs. loss-correction methods**: CL identifies specific errors; loss correction absorbs noise during training without identifying errors. CL is more interpretable and actionable.
- **CL + Snorkel**: the output of a Snorkel label model is noisy; CL can then identify the worst errors.

**Trade-offs.**
- **Precision vs. recall of flagged errors**: tuning thresholds trades off false positives and false negatives among flagged errors.
- **Compute cost**: requires cross-validated model training (k models), nontrivial for large data.
- **Assumption risk**: class-conditional noise assumption can fail; CL degrades gracefully but estimates become biased.

### 6. Common Pitfalls & Misconceptions

**1. "CL finds all label errors."** No. It finds errors that are inconsistent with the model's predictions. Systematic errors that also fool the model slip through.

**2. "Use the training predictions."** Use out-of-fold predictions. In-sample predictions are biased by the (possibly noisy) training label.

**3. "Remove every flagged example."** Human-review a sample first. Models can flag genuinely hard examples as errors.

**4. "CL assumes class-conditional noise — my data has feature-dependent noise so I can't use it."** CL has been shown empirically robust to violations of this assumption. It's a recommended assumption, not a hard requirement.

**5. "Error correction always improves accuracy."** If the test set has the same errors as the training set, "correcting" training labels can actually hurt test accuracy. Always evaluate on a cleaned test set.

**6. "I need perfect calibration."** CL is more robust than its derivation suggests. Rank-calibrated predictions are usually enough.

---

# PART IV — INTERVIEW PREPARATION

This section is organized by topic cluster and by difficulty. Each question is followed by a detailed answer at the level expected for an ML / applied scientist interview.

---

## A. Data Quality Dimensions — Interview Questions

### A.1 Foundational

**Q: What are the five classical dimensions of data quality, and why isn't accuracy sufficient on its own?**

A: Accuracy, completeness, consistency, timeliness, and validity. Accuracy alone is insufficient because a dataset can have individually accurate records but gross coverage gaps (e.g., no representation of a subpopulation), consistency problems across sources (two systems encoding the same attribute differently), stale data (accurate yesterday, wrong today), or format violations that fail downstream processing. A focus only on accuracy leads to the "clean labels, biased data" failure mode that dominates real-world ML incidents — particularly silent coverage loss and training–serving skew, neither of which show up in accuracy metrics.

**Q: Explain the difference between systematic error and random error in a measurement process. Why does this distinction matter for ML?**

A: Random error is zero-mean noise that averages out with more observations. Systematic error (bias) is a repeatable offset that does not average out. For ML, this distinction determines the remediation strategy. If a sensor has high variance but is unbiased, collecting more data and applying regularization suffices. If a sensor is biased — say, consistently over-reporting by 2°C — more data converges to the biased value. You must fix the measurement process. The formal decomposition $\text{MSE} = \text{bias}^2 + \text{variance}$ mirrors the bias–variance decomposition for models, applied to the data source.

**Q: What is training–serving skew, and where in the data quality framework does it fit?**

A: Training–serving skew is any systematic difference between the data distribution used at training time and the distribution seen at inference time, whether in feature values, feature freshness, or schema. It is primarily a timeliness and consistency failure. A classic manifestation: training uses a batch-computed feature (e.g., 7-day moving average computed nightly), serving uses a real-time feature (the latest value). Even if both are correctly computed, they represent different quantities. Detection requires comparing feature distributions at training and serving, ideally with point-in-time correct joins and feature stores to eliminate the skew at the source.

### A.2 Mathematical

**Q: Define MCAR, MAR, and MNAR formally. Give an example of each.**

A: Let $X = (X_{obs}, X_{mis})$ and let $M$ be the missingness indicator.
- **MCAR**: $P(M \mid X) = P(M)$ — missingness independent of all data. Example: a sensor that drops 1% of readings uniformly at random.
- **MAR**: $P(M \mid X) = P(M \mid X_{obs})$ — missingness depends only on observed variables. Example: older users skip the income field, but once you condition on age (observed), missingness is random.
- **MNAR**: $P(M \mid X)$ depends on $X_{mis}$ itself. Example: high-income individuals are more likely to skip the income field; missingness depends on the missing value.

MCAR and MAR are "ignorable" in the technical sense: maximum-likelihood estimation over observed data remains consistent. MNAR requires explicit modeling of the missingness mechanism (selection models, pattern-mixture models).

**Q: Given 95% observed agreement on a binary task with 95% class imbalance, compute κ and interpret.**

A: Consider the 2×2 contingency with heavy skew — e.g., both annotators label 95% as the majority class. Observed $p_o = 0.95$ is driven almost entirely by chance; both marginals are (0.95, 0.05), so $p_e = 0.95^2 + 0.05^2 = 0.905$. Then $\kappa = (0.95 - 0.905)/(1 - 0.905) = 0.47$. Modest, despite the high raw agreement. This is the kappa paradox. Remedies: report $p_o$ alongside κ, use Gwet's AC1 (which uses a different chance correction), or stratify the analysis by class.

**Q: A feature is 99% populated in training, 90% populated in serving. Estimate the impact on model accuracy, assuming the feature has moderate predictive power.**

A: The impact depends on (a) how the model handles missingness and (b) the informativeness of the feature. If the model is trained on nearly complete data and the missing mechanism at serving is unrelated to training's imputation assumption, the serving distribution of the feature — effectively a mixture with 10% missing — is no longer identically distributed with training. If the model uses a default value for missing (e.g., 0), and 0 is far from the typical value, predictions on missing-feature examples drift toward whatever the model learned for that region. Quantitatively, if the feature contributes an effect of magnitude $\delta$ to the prediction, and serving missingness increases from 1% to 10%, the accuracy can degrade by up to ~$9\% \times \text{feature's share of prediction}$. This is a silent regression — always monitor feature completeness in serving.

### A.3 Applied

**Q: You discover that a critical feature in your production model went from 99% populated to 85% populated over the past month. Walk me through your investigation.**

A: First, classify the missingness pattern. Is it uniform across segments (suggests upstream data outage) or concentrated in specific segments (suggests a source-specific bug)? Query the missingness rate stratified by date, user cohort, product, region, and upstream source. Second, determine whether this is a feature-generation issue (the feature was present in raw data but dropped in transformation) or a source-data issue (raw data itself is incomplete). Third, quantify impact: compute model accuracy on examples with and without the feature, and compare on recent data; also check whether model outputs have shifted (predicted distribution drift). Fourth, fix the source if possible; otherwise, add the missing indicator as a feature and retrain, or route affected records to a fallback. Finally, instrument monitoring so this pattern would be caught within days, not a month — both at the data layer (null rate SLOs) and at the model layer (serving distribution vs. training baseline).

**Q: You're building a fraud detection system. What data quality controls would you put in place from day one?**

A: A minimum viable set:
1. Schema validation at every ingestion point (Protocol Buffers, Avro, or equivalent).
2. Per-feature null rate and range monitors with alerting at ±2σ from baseline.
3. Per-segment coverage monitoring (country, device type, user segment) to catch silent under-sampling.
4. Training–serving skew monitoring: a daily job that compares feature distributions between training data and a sample of serving data.
5. Label latency and completeness tracking: fraud labels arrive weeks after the transaction; track the labeling delay distribution and the fraction of transactions eventually labeled.
6. Adversarial-data guards: invalid records (negative amounts, etc.) are frequently fraud signals themselves; capture them separately, not dropped.
7. A weekly audit of a random sample against source-of-truth data where available.
8. Feature store with point-in-time correctness to eliminate temporal leakage and skew.

**Q: How do you decide between listwise deletion, mean/median imputation, model-based imputation, and multiple imputation for missing data?**

A: The choice depends on the missingness mechanism and downstream use.
- **Listwise deletion**: safe only under MCAR; under MAR or MNAR it introduces selection bias. Use when missingness rate is low (<5%) and MCAR is plausible.
- **Simple imputation** (mean/median): fast and often adequate for non-predictive features or when the downstream model is robust (tree-based). Distorts variance and correlations; use with care for linear models.
- **Model-based imputation**: fit a model for each partially-observed variable given the others (e.g., MICE). Respects conditional relationships; valid under MAR. Good default for tabular data with structured missingness.
- **Multiple imputation**: produce several imputed datasets, run analysis on each, combine results. Correctly propagates imputation uncertainty into downstream inference. Preferred for statistical analysis; less common in ML because combining multiple training runs is awkward.
- **Native handling** (e.g., XGBoost's default directions, neural networks with mask embeddings): often the best option for ML. The model learns the optimal split direction for missing values.

Always retain a missingness indicator column — it is frequently predictive.

### A.4 Debugging & failure-mode

**Q: A model has 95% offline accuracy but 70% online accuracy. What are the most likely data quality causes?**

A: The prime suspects are distribution shift and training–serving skew. Specific diagnoses:
1. **Temporal leakage**: training set includes information that would not be available at serving time (features computed with future data). Offline evaluation is artificially inflated.
2. **Training–serving feature computation mismatch**: the feature code for training batch and serving real-time differs subtly (different null handling, different aggregation window, different timezone).
3. **Label leakage**: training labels are correlated with features that in production are produced after labeling (e.g., using the "account_closed" status as a feature when predicting churn).
4. **Selection bias in training**: training data is a filtered subset (e.g., only reviewed transactions) that is not representative of production.
5. **Stale feature store**: offline evaluation uses fresh feature values; production uses stale ones.
6. **Offline test not held out**: test set was used in hyperparameter tuning — the 95% is overfit.

Remediation steps: perform point-in-time correct feature reconstruction, compare feature distributions offline vs. online, check for label leakage by training with feature ablation, and evaluate on a chronologically held-out set.

**Q: Your fraud model performs well overall but has 10× worse precision on transactions from Brazil. What data quality failures could cause this?**

A: Possibilities span all five dimensions.
- **Coverage**: Brazilian transactions are under-represented in training; the model hasn't learned Brazil-specific patterns.
- **Label quality**: the Brazilian fraud review team is smaller or uses different criteria; labels may be systematically noisier.
- **Feature availability**: certain features (e.g., device fingerprinting) may be less available for Brazilian users due to regulatory or infrastructure differences.
- **Timeliness**: the Brazilian market may have shifted fraud patterns more recently than the training window.
- **Consistency**: transaction amounts in BRL vs. USD may be inconsistently normalized.

Investigate by computing per-feature null rates, label density, and performance metrics stratified by country. Likely the fix is a combination of more Brazilian labels, feature engineering specific to the market, and possibly a country-specific model head.

### A.5 Probing follow-ups

**Q: You said you'd use multiple imputation for MAR data. But what if you can't confirm MAR vs MNAR?**

A: In practice, you rarely can confirm MAR definitively — the MNAR scenario by definition depends on the unobserved value, which you can't check directly. Pragmatic approach: (a) do a sensitivity analysis, imputing under MAR and also under MNAR with plausible bias assumptions, and see how much the conclusions change. If they don't change much, MAR is safe enough. If they do, you need to reason about the mechanism explicitly. (b) Where possible, reduce the MNAR risk at the source — e.g., redesign the form so skipping is impossible, or add partial progress tracking.

**Q: Why is MSE decomposed as $b^2 + \sigma^2$ and not $b + \sigma^2$?**

A: MSE is the expected *squared* error: $\mathbb{E}[(\hat{x} - x)^2]$. Writing $\hat{x} = x + b + \eta$ with $\eta$ zero-mean:
$$
\mathbb{E}[(\hat{x} - x)^2] = \mathbb{E}[(b + \eta)^2] = b^2 + 2b\mathbb{E}[\eta] + \mathbb{E}[\eta^2] = b^2 + 0 + \sigma^2.
$$
The bias enters squared because MSE penalizes direction equally — a bias of +2 and −2 contribute the same MSE. Adding bias linearly would be MAE-like and would allow positive and negative biases to cancel, which is not what the expected squared error captures.

---

## B. Labeling Strategies — Interview Questions

### B.1 Foundational

**Q: What is weak supervision? How does it differ from semi-supervised learning?**

A: Weak supervision uses multiple **noisy, possibly-conflicting labeling sources** (rules, heuristics, pretrained models, crowd labels) to produce probabilistic labels for unlabeled data, without requiring gold labels for each example. The canonical framework is Snorkel: labeling functions → label model → noisy probabilistic labels → discriminative model. Semi-supervised learning, by contrast, uses a small amount of gold-labeled data plus a large amount of unlabeled data. It does not use external rules; it uses the model's own predictions (pseudo-labeling) or distributional/smoothness assumptions (consistency regularization). Both can be combined: use weak supervision to generate initial labels, then apply SSL to improve the model using the remaining unlabeled data.

**Q: Explain active learning. When does it fail to outperform random sampling?**

A: Active learning iteratively selects examples to label based on an acquisition function (uncertainty, query-by-committee, expected error reduction). The goal is to reach target accuracy with fewer labels. Failure modes:
1. **Cold start**: when the initial model is very weak, its uncertainty estimates are unreliable; random may be safer for the first batch.
2. **Outlier hunting**: uncertainty sampling aggressively queries outliers, which may be label noise rather than informative boundary points.
3. **Redundant batches**: naive top-k by uncertainty selects very similar examples; diversity-aware methods (coreset, BADGE, BatchBALD) are needed.
4. **Distribution mismatch**: active learning produces a biased labeled set skewed toward hard cases; this set is not a valid evaluation distribution.
5. **Ceiling tasks**: when labels are already cheap or the pool is small, the overhead of AL outweighs its benefits.

**Q: What is a labeling function in Snorkel, and what makes a good set of them?**

A: A labeling function (LF) is a Python function that takes an example and returns a label or ABSTAIN. Good LFs (a) have high precision where they fire (usually >70%), (b) cover distinct subsets (diverse triggers), (c) don't overlap too redundantly, and (d) collectively cover most of the data. You do not need to hit 100% coverage with any single LF — a 10%-coverage, 90%-precision LF contributes valuable signal. The typical set size is 10–30 LFs, balancing quantity and quality.

### B.2 Mathematical

**Q: Derive the majority-vote expected accuracy for $k$ independent annotators with individual accuracy $p$.**

A: Majority vote is correct when more than half the annotators are correct. Under independence and equal accuracy $p$:
$$
P(\text{correct}) = \sum_{j=\lceil k/2 \rceil}^k \binom{k}{j} p^j (1-p)^{k-j}.
$$
For odd $k$ the result is monotonic in $k$ when $p > 0.5$ and decreasing when $p < 0.5$. Concretely for $p = 0.7$: $k=1$: 0.70; $k=3$: 0.784; $k=5$: 0.837; $k=7$: 0.874. Each redundancy doubling shrinks the error rate but with diminishing returns. The asymptotic result (Condorcet's jury theorem) is that as $k \to \infty$, the probability of correct majority vote tends to 1 if $p > 0.5$ and to 0 if $p < 0.5$.

**Q: Sketch the Dawid–Skene EM algorithm.**

A: Let $y_i$ be the latent true label, $\hat{y}_{i,a}$ observed label from annotator $a$, $\theta^{(a)}_{j,k} = P(\hat{y}_{i,a} = k \mid y_i = j)$, and $\pi_j = P(y_i = j)$.

- **E-step**: compute posterior
$$
q_{i, j} = \frac{\pi_j \prod_a \theta^{(a)}_{j, \hat{y}_{i,a}}}{\sum_{j'} \pi_{j'} \prod_a \theta^{(a)}_{j', \hat{y}_{i,a}}}.
$$
- **M-step**:
$$
\pi_j = \frac{1}{n}\sum_i q_{i,j}, \quad \theta^{(a)}_{j,k} = \frac{\sum_i q_{i,j} \mathbb{1}[\hat{y}_{i,a}=k]}{\sum_i q_{i,j}}.
$$

Iterate to convergence; initialize with majority vote. Output: soft labels $q_{i,j}$ and per-annotator confusion matrices.

**Q: In Snorkel, why does conditional independence between labeling functions allow accuracy estimation without ground truth?**

A: For binary LFs $\lambda_j, \lambda_k$ that are conditionally independent given the true label $Y$ (with $Y \in \{-1, +1\}$),
$$
\mathbb{E}[\lambda_j \lambda_k \mid |\lambda_j|, |\lambda_k| = 1] = \mathbb{E}[\lambda_j \mid Y] \cdot \mathbb{E}[\lambda_k \mid Y] \cdot \mathbb{E}[Y^2] = \mathbb{E}[\lambda_j Y] \mathbb{E}[\lambda_k Y].
$$
The LHS is observable from the label matrix. If we have three LFs $j, k, l$, we get three equations for three products of LF accuracies, from which each LF's accuracy is identifiable (up to a global sign, resolved by assuming LFs are better than random on average). This is Snorkel's "triplet method" and is what enables unsupervised accuracy estimation.

### B.3 Applied

**Q: You have 500 labeled examples and 500,000 unlabeled. You need to build a production text classifier. Walk me through your options.**

A: Candidate strategies, roughly in order of typical effort:
1. **Start with a fully-supervised baseline** using a pretrained encoder (BERT, sentence-transformers) fine-tuned on the 500 labels. Pretrained features dramatically boost small-data performance; often this alone gets surprisingly far.
2. **Active learning**: use the baseline to select 500 more labels (total 1,000). Typically beats random labeling for the same budget.
3. **Weak supervision (Snorkel)**: write 15–30 LFs from domain knowledge, apply to the 500k, train a downstream model on weak labels. Especially powerful if programmatic priors exist.
4. **Semi-supervised learning**: combine the 500 labels with consistency regularization (UDA) or pseudo-labeling on the 500k unlabeled. Works well for text with pretrained encoders.
5. **Hybrid**: start with Snorkel to generate initial labels, then use active learning to identify the items where the label model is uncertain for expert review.

Decision driver: if programmatic rules are available, weak supervision is highly leveraged; if not, go pretrained encoder + active learning + SSL. Maintain a separate gold test set throughout.

**Q: When would you choose crowd-sourced labeling over expert labeling?**

A: Crowd-sourcing is preferable when (a) the task is understandable by non-experts with clear guidelines (e.g., "is this image a cat?"), (b) volume is high (thousands to millions), (c) the cost per label must be low, and (d) quality control mechanisms (gold questions, redundancy, Dawid–Skene) can maintain acceptable accuracy. Expert labeling is required when the task requires specialized training (medical, legal, scientific), where a wrong label has high downstream cost, or where the domain vocabulary is opaque to lay workers. Hybrid designs are common: crowd labels for bulk data, expert adjudication for ambiguous cases and gold sets.

**Q: How would you design a labeling pipeline that handles concept drift?**

A: Assume the label distribution or feature-label relationship will shift over time. Key elements:
1. **Continuous labeling stream**: don't label in one big batch and stop. Sample and label a fraction of production data continuously.
2. **Monitoring for drift**: track feature distributions, model confidence distributions, and label distributions over time. Alert when any diverges meaningfully from training baseline.
3. **Re-labeling triggers**: when drift is detected in a segment, prioritize that segment for relabeling and retraining.
4. **Gold set refresh**: rotate out old gold items and adjudicate new ones regularly.
5. **Versioned data**: keep labeled data tagged by time so old and new labels can be distinguished during training (e.g., decay weights for old data).
6. **Recency-aware evaluation**: evaluate the model on the most recent time window, not on a mix.
7. **Human-in-the-loop**: flag low-confidence predictions in production for human review; those reviews become new labels.

### B.4 Debugging & failure-mode

**Q: You deploy a Snorkel-trained model and it performs much worse in production than in offline evaluation. What are the likely causes?**

A: Ranked by frequency:
1. **Correlated LFs**: multiple LFs trigger on the same underlying feature. The label model over-weights the signal; offline evaluation on gold (small) doesn't expose it. Fix: decorrelate LFs or use a label model that models dependencies (Snorkel MeTaL).
2. **Biased LF coverage**: all LFs miss a class or segment; the downstream model is untrained on that slice. Fix: coverage audit, add LFs targeting under-covered slices.
3. **Gold set not representative**: the gold set used for offline evaluation was curated for class balance, but production is skewed. Fix: match gold distribution to production.
4. **Distribution shift**: production data differs from the unlabeled set Snorkel was trained on. Fix: retrain Snorkel periodically on fresh unlabeled data.
5. **LF staleness**: regex and keyword LFs decay as language evolves (particularly for social media or user-generated content). Fix: ongoing LF maintenance.

**Q: Pseudo-labeling is amplifying errors in a minority class. What's happening and how do you fix it?**

A: The model starts biased toward the majority class. Predictions on minority examples tend to be low-confidence or majority-predicting. Under threshold-based pseudo-labeling (e.g., FixMatch), only high-confidence predictions become pseudo-labels — and these are disproportionately majority class. Retraining on this biased pseudo-label set makes the next iteration more majority-biased. Remedies:
1. Class-balanced sampling in both labeled and pseudo-labeled data.
2. Per-class confidence thresholds (lower for minority classes).
3. Reweight the pseudo-label loss by inverse class frequency.
4. Upsample minority in the gold seed set.
5. Use consistency regularization without pseudo-labeling for minority classes; it doesn't require confidence thresholding.
6. Algorithms specifically designed for class-imbalanced SSL (CReST, DASO).

### B.5 Probing follow-ups

**Q: You recommended active learning and said it can 10× label efficiency. But you evaluate on a random test set — doesn't AL's selected labels bias the model?**

A: Yes, the *labeled training set* is biased by AL (skewed toward hard examples). But the test set is held out and iid. Under most circumstances, AL produces a better model on the iid test distribution than random-labeling the same budget, because the informative boundary examples improve the decision surface in general, not just near the hard points. The risk is when AL's bias exceeds the informational benefit — which can happen for extremely imbalanced classes, adversarial outliers, or when the model is miscalibrated. The mitigation is a mix of random + AL sampling (e.g., 80% AL, 20% random), which preserves most of the efficiency gain while maintaining some distributional coverage.

**Q: Confidence thresholding in FixMatch: why weak augmentation for the pseudo-label and strong for the gradient?**

A: Three reasons together. (1) Weak augmentation preserves the semantic content of the input, so if the model is confident on it, the prediction is more trustworthy as a pseudo-label. (2) Strong augmentation produces a perturbed view that is harder for the model, forcing a gradient update if the prediction diverges — this is where learning happens. (3) The combination operationalizes the smoothness assumption: weakly augmented ≈ original, so its label should equal strongly augmented's label. Learning pushes the model to be consistent across this augmentation gap. If both views were strongly augmented, both predictions would be unreliable and no meaningful signal would flow. If both were weakly augmented, there would be nothing to learn — the views would be too similar.

---

## C. Label Quality Assessment — Interview Questions

### C.1 Foundational

**Q: What is inter-annotator agreement, and why do we correct for chance?**

A: IAA quantifies how much annotators agree on labeling. Raw agreement rates over-state alignment when label distributions are skewed — two annotators who always label "common" agree a lot without looking at the data. Chance correction (κ, α) measures agreement beyond random coincidence. Concretely, $\kappa = (p_o - p_e)/(1 - p_e)$ where $p_e$ is the agreement expected if annotators drew labels randomly from their marginals. This gives a more meaningful signal of whether the annotators are reading the same task the same way.

**Q: Explain the difference between Cohen's κ and Fleiss' κ.**

A: Cohen's κ is for two annotators, categorical labels; it uses their individual marginals to compute expected agreement. Fleiss' κ is for a fixed number of annotators per item (not necessarily the same annotators across items) and uses a shared marginal across all annotators. Fleiss' κ is not the direct multi-annotator generalization of Cohen's — it assumes all annotators share the same overall label distribution. For varying numbers of annotators per item or missing data, Krippendorff's α is preferred.

**Q: What does confident learning do, and when should you use it?**

A: Confident learning (Northcutt et al.) identifies likely label errors in a dataset by comparing model predictions (out-of-fold) against observed labels. It estimates a joint distribution of (true label, observed label) via class-specific confidence thresholds, flagging off-diagonal entries as likely errors. Use it when (a) you have a reasonably-sized dataset, (b) labels may be noisy (crowd, weak supervision), and (c) a reasonable classifier is trainable. Output: a ranked list of candidate errors for human review or automated removal. Best practice is to relabel flagged examples rather than drop them, since flagged items are often genuinely informative boundary cases.

### C.2 Mathematical

**Q: Derive Cohen's κ for the 2×2 case and demonstrate the kappa paradox numerically.**

A: Confusion matrix:
$$
\begin{array}{c|cc|c}
 & B_1=Y & B_1=N & \text{Row} \\
\hline
A_1=Y & a & b & a+b \\
A_1=N & c & d & c+d \\
\hline
\text{Col} & a+c & b+d & n
\end{array}
$$
$p_o = (a+d)/n$. Marginals: $p_{A,Y}=(a+b)/n$, $p_{A,N}=(c+d)/n$, $p_{B,Y}=(a+c)/n$, $p_{B,N}=(b+d)/n$. Chance agreement: $p_e = p_{A,Y} p_{B,Y} + p_{A,N} p_{B,N}$. κ = $(p_o - p_e)/(1 - p_e)$.

Paradox example: $a=1, b=4, c=3, d=92$ (skewed to N). $p_o = 93/100 = 0.93$. Marginals: (0.05, 0.95) and (0.04, 0.96). $p_e = 0.002 + 0.912 = 0.914$. $\kappa = (0.93 - 0.914)/(1 - 0.914) = 0.019 / 0.086 \approx 0.19$.

High observed agreement, low κ. Both annotators are mostly saying "N" because "N" is common; they're not really agreeing on the difficult cases.

**Q: Why is weighted kappa used for ordinal labels? Derive its form.**

A: For ordinal labels, the distance between labels is meaningful: disagreeing by 1 is not the same as disagreeing by 4. Standard κ treats all disagreements equally, which under-rates ordinal agreement. Weighted κ introduces a weight matrix $w_{jk}$ ($w_{jj} = 0$, larger for farther labels):
$$
\kappa_w = 1 - \frac{\sum_{jk} w_{jk} P_{jk}}{\sum_{jk} w_{jk} p_{A,j} p_{B,k}}.
$$
Linear weights $w_{jk} = |j-k|/(K-1)$ penalize proportionally; quadratic weights $w_{jk} = (j-k)^2/(K-1)^2$ penalize larger disagreements more heavily. Quadratic weighted kappa is the standard for ordinal tasks and matches the structure of MSE. Setting $w_{jk} = \mathbb{1}[j \ne k]$ recovers Cohen's κ.

**Q: In confident learning, why is out-of-fold prediction required?**

A: In-sample predictions are biased: the model has seen the example's (possibly noisy) label during training, so its prediction is tilted toward that label. For a mislabeled example, the model's in-sample prediction may agree with the wrong label, missing the error. Out-of-fold predictions (from a model trained on different folds) are unbiased estimates of what the model would say about a fresh example, preserving the ability to detect disagreement between true class and observed label. Formally, CL's estimator of the joint $P(y^*, \tilde{y})$ relies on treating $\hat{p}_{i,k}$ as an unbiased estimate of $P(y^* = k \mid x_i)$; in-sample predictions violate this.

### C.3 Applied

**Q: You've crowd-labeled 100,000 items with 3 annotators each. κ on a 1,000-item pilot was 0.78. Describe your quality control pipeline.**

A: Multi-layered QC:
1. **Per-annotator gold scoring**: 5–10% of items are pre-labeled gold. Per-annotator accuracy is computed continuously. Annotators below a threshold (say, 0.8) are filtered out; their labels are optionally re-done.
2. **Dawid–Skene aggregation**: instead of majority vote, use DS to incorporate per-annotator reliability. Soft labels are produced.
3. **Flagging for adjudication**: items where DS soft label is below 0.8 are escalated to a senior annotator.
4. **Drift detection**: weekly per-annotator and overall κ against rolling gold baseline. Sudden drops flag guideline misunderstanding or annotator fatigue.
5. **Calibration sessions**: periodic discussion of contentious items to align.
6. **Confident learning audit**: quarterly, train a model on the aggregated labels and use CL to flag likely errors. Expert re-review of flagged items.
7. **Version control**: gold set, guidelines, and aggregated labels are versioned so that changes can be audited.

**Q: You need to build an evaluation set for a clinical NLP task. Describe the process.**

A: Goals: high-quality labels, representativeness, documented ambiguity.
1. Stratified sampling: sample across disease categories, text sources (discharge summaries vs. progress notes vs. radiology reports), and any known difficulty axes (e.g., negation, uncertainty, family history).
2. Multi-expert labeling: each item labeled by at least 3 clinicians with documented specialty and experience level.
3. Adjudication: any disagreement escalates to a committee or a senior adjudicator. Rationale is documented.
4. Ambiguous case capture: items where even after adjudication there's substantive disagreement are labeled as "ambiguous" and either excluded from primary evaluation or used in a separate "hard" evaluation set.
5. Gold set size: for per-class F1 estimates with 95% CI width of 0.05, aim for 400–600 examples per class; fewer for very rare classes with larger CIs accepted.
6. Protocol for updates: lock the gold set version; if the task evolves, create v2 and track performance across versions.
7. Hold-out protocol: the gold test set is never used for model training or hyperparameter tuning. A separate gold validation set is used for tuning.

**Q: A team reports 98% model accuracy. Inter-annotator κ on their test set is 0.82. Do you believe the model accuracy?**

A: Skeptically. κ of 0.82 is substantial but not near-perfect; it implies roughly 85–92% raw annotator agreement, depending on distribution. If human annotators disagree on ~10% of items, the "ground truth" label is not reliable enough to distinguish 98% from 95% — both are inside the noise floor of evaluation. Probable possibilities: (a) the test set was curated/filtered to remove ambiguous items (inflating both κ and model accuracy); (b) the test set is skewed class-wise (majority class accuracy dominates); (c) label leakage into training; or (d) the model is genuinely excellent, in which case the evaluation set is the bottleneck — you cannot distinguish between very good and superhuman with noisy labels. Ask to see per-class metrics, the distribution of label confidence, and whether confident learning has been run on the test set to find remaining errors.

### C.4 Debugging & failure-mode

**Q: You run Cohen's κ on a binary classification task and get κ = 0.1 but $p_o = 0.95$. What's going on and how do you respond?**

A: This is the kappa paradox: heavy class skew inflates $p_e$ to near $p_o$, crushing the κ. It does not necessarily mean annotators are bad — both may be correctly identifying the common class. Actions: (1) report $p_o$ alongside κ — the full picture requires both; (2) compute Gwet's AC1, which is more robust to skew; (3) stratify the analysis by class — compute agreement on just the rare-class items, which reveals whether annotators agree on the hard cases; (4) consider whether the label distribution is reasonable — skew could reflect a population prior or could be a data-quality issue. If rare-class agreement is also high, the annotators are genuinely aligned; κ is just a poor metric for this distribution.

**Q: Confident learning flags 15% of your training data as likely errors. Should you remove all of them?**

A: Almost certainly not. 15% is a large fraction; CL precision is typically 60–80%, so many flagged examples are genuinely correct but hard cases. Removing all would reduce data volume substantially and bias training toward easy examples. Recommended workflow:
1. Sample 100–300 flagged examples and have a human review them.
2. Categorize: clear label errors, ambiguous cases, correctly-labeled-but-hard.
3. If precision is high (>80%), consider programmatic removal for clear-error categories; relabel ambiguous ones.
4. If precision is moderate (60–80%), prioritize human review over programmatic removal.
5. Retain the flagged-hard-but-correct examples; they are often the most informative.
6. Evaluate the resulting cleaned dataset on a held-out (and ideally separately cleaned) test set. If test accuracy improves, the cleanup is working; if it degrades, you've removed informative examples.

### C.5 Probing follow-ups

**Q: You use Dawid–Skene to aggregate annotations. What if the annotators are not conditionally independent — e.g., they're all using the same guidelines and share the same misconceptions?**

A: Conditional independence is a key DS assumption; violation leads to under-estimation of error correlations. If annotators share biases, DS converges to a consensus that agrees with the shared bias, not the true label. Practical consequences: DS will not catch systematic errors in guidelines, and per-annotator confusion matrices will be underestimates of their true error rates on the biased items. Mitigations: (a) incorporate a gold-standard set to anchor accuracy estimation; (b) add annotators from different training backgrounds when possible; (c) use DS's variant that models pairwise annotator dependencies; (d) periodically audit with an external expert who was not part of guideline drafting.

**Q: Confident learning is a way to clean training labels. But how do you clean *test* labels without circular reasoning?**

A: Several approaches, none perfect:
1. **Use a held-out cross-validated model to predict on the test set**: essentially CL applied with a model never trained on the test data. The predictions are out-of-sample and can flag disagreements without circularity.
2. **Multi-annotator adjudication on flagged items**: CL identifies suspicious test items; these go to multiple expert annotators for relabeling. The final test labels are adjudicated, not model-derived.
3. **Cross-dataset validation**: if the same task exists in multiple datasets, agreement across datasets signals true labels; disagreement flags noise.
4. **Sanity checks by domain experts**: a random sample of the flagged test items is manually reviewed and either fixed or retained.

The principle: CL flags candidates; humans resolve them. Never silently replace test labels with model predictions, because that tautologically boosts evaluation.

**Q: You're asked to add a new class to an existing labeled dataset. The old labels don't distinguish this class. How do you proceed?**

A: This is a label-schema evolution problem, which is a consistency-over-time issue.
1. Define the new class rigorously with examples.
2. Don't retrofit the old labels automatically — instead, treat old labels as "legacy" and relabel a representative sample under the new schema.
3. If relabeling all of the old data is infeasible: (a) train a model to map old-schema labels to new-schema labels where possible; (b) hold out a subset for manual relabeling; (c) evaluate whether the old data is useful under the new schema or should be excluded.
4. Maintain dual-labeled sets: old schema for comparison with historical model versions, new schema for current model training.
5. Update guidelines, calibration items, and gold sets before scaling.
6. Track which records are under which schema version explicitly.

---

## D. Cross-cutting scenario-based questions

These are the kinds of broad questions that test synthesis across topics.

**Q: You join a team with a working production ML system. Where do you start auditing its data quality and labeling practices?**

A: My first pass covers:
1. **Data lineage**: what is the source of each feature and label? Which fields are computed, which are pulled from upstream systems? Are there any undocumented transformations?
2. **Schema and validity monitoring**: is there any validation (format, range, referential)? Are null rates and schema breaks monitored?
3. **Training–serving parity**: compute feature distributions on training data vs. a recent production sample. Any drift?
4. **Label provenance**: who labels, how, with what guidelines? Is there a gold standard? Is IAA measured?
5. **Segmented performance**: how does the model perform across segments (time, geography, user cohort)? Are there blind spots indicated by coverage gaps?
6. **Test set integrity**: is the test set frozen, held out from training, and representative of production?
7. **Pipeline reproducibility**: can I rerun training from scratch on a specific data snapshot and get the same result?

The first two usually reveal the fastest wins.

**Q: Your team has a choice: $50k for more labels, or $50k for a data quality engineer. Which do you choose, and why?**

A: Almost always the engineer, with exceptions. Reasoning: labeled data is a one-time asset that decays with drift; an engineer builds durable infrastructure (monitoring, validation, versioning, gold set management) that multiplies the value of all future labels. The exceptions: (a) you have a large unmet need for labels on a stable task and the pipeline is already solid; (b) you're in a research prototype stage where building infrastructure is premature. In both cases, labels can unblock progress. Usually, though, early-stage ML teams over-invest in labels relative to infrastructure, and this asymmetry is worth correcting.

**Q: Describe a real or hypothetical end-to-end ML system with clear data quality and labeling design choices at each stage.**

A: A fraud detection system for card transactions:

- **Ingestion**: every transaction enters with a schema (Protobuf). Validation checks reject malformed records immediately. Validity rate is tracked per source.
- **Feature engineering**: features are computed via a feature store with point-in-time correctness to avoid temporal leakage. A training–serving skew monitor compares distributions daily.
- **Label generation**: labels come from customer disputes, manual fraud review, and chargebacks, arriving with a 30–90 day lag. Each label source has a confidence weight. Snorkel-style weak-supervision LFs (rule-based flags from existing systems) generate additional weak labels for real-time feedback.
- **Label aggregation**: Dawid–Skene over multiple sources for each transaction; final label is a probability.
- **Gold set**: manually adjudicated 2,000 transactions with clinical-level review; refreshed quarterly. Used as the primary evaluation set.
- **Active learning**: weekly, the 500 most uncertain predictions are sent to human review. These labels flow back into training.
- **Confident learning**: monthly, an out-of-fold model flags likely errors in training data. Suspected errors are re-reviewed.
- **Evaluation**: on the gold set, segmented by geography, product, and time. Per-segment metrics are tracked with confidence intervals.
- **Monitoring**: feature distribution drift, label latency distribution, per-segment performance decay. Alerts trigger investigation.
- **Retraining**: monthly cadence plus drift-triggered retraining. Each model version is evaluated against the current gold set and compared to the prior version.

At every stage, data quality considerations — completeness, accuracy, timeliness, consistency, validity — have concrete mechanisms.

---

## Closing note

Data quality and labeling are often treated as janitorial in ML curricula, but in applied-scientist practice they are where most of the accuracy, most of the incidents, and most of the business risk live. A model is the crystallization of a labeling process; understanding the labeling process is understanding what the model really learned. The mathematical framework in this guide — missingness mechanisms, chance-corrected agreement, conditional-independence identification in Snorkel, EM in Dawid–Skene, confident-joint estimation in CL — is the foundation for working fluently in this domain.

For interviews, expect questions that test (a) recognition of failure modes (kappa paradox, MNAR, confirmation bias in self-training), (b) design choices under constraints (budget, domain, drift), and (c) numerical fluency (compute κ on a 2×2, derive majority-vote accuracy, sketch EM). Practice doing these by hand on small examples; the mechanical fluency pays off.

---

**[← Previous Chapter: Feature Stores](feature_stores_guide.md) | [Table of Contents](../README.md) | [Next Chapter: Metrics →](ml_metrics_guide.md)**
