# Topic: Data Quality & Labeling

## 1. Motivation & Intuition

### Why This Matters
In Machine Learning, the maxim "Garbage In, Garbage Out" is absolute. A state-of-the-art Transformer model trained on mislabeled, inconsistent, or biased data will yield unreliable predictions. Data Quality and Labeling are the upstream processes that determine the ceiling of your model's performance.

While model architecture receives much academic attention, industry practitioners often spend 80% of their time on data engineering. Data quality ensures the signal is distinguishable from noise, and labeling strategies ensure the model has a reliable teacher.

### Intuitive Example
Imagine teaching a child to identify fruits.
* **Accuracy:** If you hand them an apple but call it a "banana," they will learn the wrong association.
* **Consistency:** If Mom calls it a "tomato" (botanically correct) but Dad calls it a "vegetable" (culinary context), the child becomes confused.
* **Completeness:** If you only show them red apples, they will not recognize a green apple.
* **Timeliness:** If you teach them using photos of fruits from 50 years ago that have since been genetically modified to look different, they may struggle in a modern grocery store.

### Connection to ML Systems
In a fraud detection system:
* **Poor Labeling:** If analysts randomly label legitimate transactions as fraud, the model learns false patterns (noise).
* **Active Learning:** Since fraud is rare, you cannot review every transaction. You must intelligently select only the most suspicious or ambiguous cases for human review to maximize learning efficiency.

---

## 2. Conceptual Foundations

### Part A: Data Quality Dimensions

1. **Accuracy:** The degree to which data correctly describes the "real world" object or event.
   * *Ground Truth Alignment:* Does the database record match the physical reality?
   * *Measurement Error:* Statistical noise introduced by the sensor or collection method (e.g., a thermometer consistently reading $0.5^\circ\text{C}$ high).

2. **Completeness:** The extent to which expected data is present.
   * *Missing Completely at Random (MCAR):* Missingness has no pattern (e.g., a server crash dropped random packets).
   * *Missing Not at Random (MNAR):* Missingness correlates with the value itself (e.g., high-income earners refusing to disclose salary). MNAR is dangerous because it introduces bias.

3. **Consistency:** The absence of contradiction within the data.
   * *Cross-source Discrepancies:* User A has a "Active" status in the Billing DB but "Churned" in the CRM.
   * *Schema Evolution:* A column `phone_number` changes format from integer to string over time, breaking downstream parsers.

4. **Timeliness:** The delay between the event occurrence and the data availability.
   * *Freshness:* Is the data current? (e.g., using last month's stock price to predict today's trade).
   * *Latency:* The time it takes for a data point to move from ingestion to the feature store.

5. **Validity:** The data conforms to defined business rules or constraints.
   * *Domain Constraints:* Age cannot be negative; Probability must be $[0, 1]$.
   * *Format Compliance:* Dates must be ISO 8601 (YYYY-MM-DD).

### Part B: Labeling Strategies

1. **Manual Labeling:** Humans review data and assign classes.
   * *Gold Standard:* A small, high-quality dataset labeled by domain experts, used to evaluate models and other annotators.
   * *Crowdsourcing:* Using platforms (e.g., Amazon MTurk) for cheap, high-volume labels, often requiring redundancy to ensure quality.

2. **Weak Supervision:** Programmatically generating noisy labels using heuristics or external knowledge bases. Instead of hand-labeling 10k points, you write a function: `if "pneumonia" in text: return POSITIVE`.
   * *Snorkel Framework:* A popular system that treats these heuristics as "Labeling Functions," models their correlations and accuracies, and combines them into probabilistic labels.

3. **Active Learning:** An iterative process where the model queries an oracle (human) to label *only* the data points that will improve the model the most (e.g., points where the model is uncertain).

4. **Semi-Supervised Learning:** Leveraging a small labeled dataset and a large unlabeled dataset.
   * *Pseudo-labeling:* Train a model on labeled data $\rightarrow$ predict on unlabeled data $\rightarrow$ add confident predictions to the training set $\rightarrow$ retrain.

### Part C: Label Quality Assessment

1. **Inter-Annotator Agreement (IAA):** A metric measuring how often multiple human annotators agree. If humans cannot agree on the label, the task may be ill-defined or the data ambiguous.
2. **Confident Learning:** A technique (e.g., Cleanlab) to identify label errors by finding instances where the model's predicted probability consistently diverges from the given noisy label (e.g., Model is 99% sure it's a Dog, but the label says Cat).

---

## 3. Mathematical Formulation

### Inter-Annotator Agreement: Cohen's Kappa ($\kappa$)
Accuracy is a poor metric for agreement because annotators can agree purely by chance (e.g., if 90% of items are Class A, two lazy annotators guessing "Class A" every time will have 100% agreement but 0% intelligence).

Cohen's Kappa corrects for chance agreement.

**The Equation:**

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Where:
* $p_o$: **Observed Agreement**. The proportion of items where annotators actually agreed.
* $p_e$: **Expected Agreement**. The proportion of agreement expected by chance, based on the marginal distributions of each annotator's choices.

**Interpretation:**
* $\kappa = 1$: Perfect agreement.
* $\kappa = 0$: Agreement is exactly what you'd expect by random chance.
* $\kappa < 0$: Disagreement is worse than random chance (systematic disagreement).

### Active Learning: Uncertainty Sampling
To select the "most informative" sample $x$ for a human to label, we often pick the instance where the model is least certain.

Let $\hat{y}$ be the predicted class label and $P(\hat{y}|x)$ be the probability distribution over classes.

1. **Least Confidence:**
   Select $x$ that minimizes the probability of the most likely class.
   
   $$x^*_{LC} = \arg\min_x \left( \max_{\hat{y}} P(\hat{y}|x) \right)$$

2. **Margin Sampling:**
   Select $x$ with the smallest difference between the top two most probable classes ($\hat{y}_1$ and $\hat{y}_2$).
   
   $$x^*_{M} = \arg\min_x \left( P(\hat{y}_1|x) - P(\hat{y}_2|x) \right)$$
   
   *Intuition:* If the margin is small, the model is "confused" between two choices.

3. **Entropy Sampling:**
   Select $x$ with the highest predictive entropy (maximum confusion across all classes).
   
   $$x^*_{E} = \arg\max_x \left( - \sum_{i} P(y_i|x) \log P(y_i|x) \right)$$

---

## 4. Worked Example: Calculating Cohen's Kappa

**Scenario:** Two doctors (Rater A and Rater B) diagnose 100 patients as either "Sick" or "Healthy". We want to know how reliable their diagnoses are.

**Confusion Matrix:**

| | Rater B: Sick | Rater B: Healthy | **Total (Rater A)** |
| :--- | :---: | :---: | :---: |
| **Rater A: Sick** | 45 | 15 | **60** |
| **Rater A: Healthy** | 25 | 15 | **40** |
| **Total (Rater B)** | **70** | **30** | **100** |

**Step 1: Calculate Observed Agreement ($p_o$)** These are the cases where they agreed (the diagonal).

$$
p_o = \frac{\text{Agree Sick} + \text{Agree Healthy}}{\text{Total}} = \frac{45 + 15}{100} = \frac{60}{100} = 0.60
$$

**Step 2: Calculate Expected Agreement ($p_e$)** We calculate the probability that they agree by random chance based on their individual biases (Marginals).

* **Chance of "Sick":**
  * Rater A says Sick 60% of the time ($0.6$).
  * Rater B says Sick 70% of the time ($0.7$).
  * Probability both say Sick by chance: $0.6 \times 0.7 = 0.42$.

* **Chance of "Healthy":**
  * Rater A says Healthy 40% of the time ($0.4$).
  * Rater B says Healthy 30% of the time ($0.3$).
  * Probability both say Healthy by chance: $0.4 \times 0.3 = 0.12$.

* **Total Expected Agreement:**
  
  $$p_e = 0.42 + 0.12 = 0.54$$

**Step 3: Calculate Kappa ($\kappa$)**

$$
\kappa = \frac{0.60 - 0.54}{1 - 0.54} = \frac{0.06}{0.46} \approx 0.13
$$

**Conclusion:** A Kappa of $0.13$ is **very slight agreement**. Even though they agreed 60% of the time ($p_o$), most of that was due to the fact that both doctors have a bias toward diagnosing "Sick". If you trained an ML model on this data, it would be highly unreliable because the ground truth is ambiguous.

---

## 5. Relevance to Machine Learning Practice

### When to Use Which Strategy
* **Manual Labeling:** Use for "Gold Standard" evaluation sets (Test Sets). You typically cannot trust weak supervision for the final evaluation of your model.
* **Weak Supervision:** Use when you have massive unlabeled data and subject matter experts (SMEs) are too expensive or slow. Ideal for "Cold Start" problems.
* **Active Learning:** Use when data is abundant but labeling is expensive (e.g., radiologists reading MRI scans).
* **Semi-Supervised:** Use when the data manifold is smooth (similar inputs likely have similar labels) and labeled data is scarce.

### System Design Trade-offs
1. **Cost vs. Quality:** Crowdsourcing is cheap but noisy. Experts are expensive but accurate. A common design is a **tiered approach**: use weak supervision for training data, crowdsourcing for easy validation, and experts for hard edge cases and final test sets.
2. **Latency vs. Completeness:** In streaming systems (e.g., recommendation engines), waiting for complete data (e.g., did the user finish the movie?) might take too long. You may train on "clicked" (timely proxy) vs "watched" (complete ground truth).

---

## 6. Common Pitfalls & Misconceptions

1. **The "Golden Truth" Fallacy:**
   * *Misconception:* Believing the label provided by a human is the absolute truth.
   * *Reality:* Humans make mistakes, get tired, and have biases. Always measure IAA. If IAA is low, refine the labeling instructions, not the model.

2. **Selection Bias in Active Learning:**
   * *Pitfall:* If you only label the "uncertain" points, your test set becomes unrepresentative of the real-world distribution.
   * *Fix:* Always maintain a randomly sampled hold-out set that is *not* subject to active learning selection logic to evaluate true performance.

3. **Data Leakage via Timeliness:**
   * *Pitfall:* Using features in training that wouldn't be available at inference time.
   * *Example:* Predicting "Loan Default" using a feature "Number of Missed Payments." If "Missed Payments" is updated only *after* the default event is recorded in the DB, the model will look perfect in training but fail in production.

4. **Feedback Loops:**
   * *Pitfall:* A model predicts "Spam," the email is moved to the spam folder, and the user never sees it to correct it if it's wrong. The model believes its prediction was correct because it received no complaint. This reinforces the error.

---

## 7. Interview Preparation Questions

### Foundational Questions

**Q1: What is the difference between "Missing at Random" (MAR) and "Missing Not at Random" (MNAR)? Why does it matter?** * **Answer:** MAR means the probability of data being missing depends on *observed* data but not the missing value itself (e.g., men are less likely to answer a depression survey, but within the "men" group, missingness is random). MNAR means the missingness depends on the value itself (e.g., very depressed people are too lethargic to answer the survey).
* **Significance:** MAR can be handled by imputation or weighting. MNAR introduces systematic bias that imputation cannot fix; it requires explicit modeling of the missingness mechanism or domain knowledge to correct.

**Q2: Explain the concept of "Gold Standard" data. Is it always 100% accurate?** * **Answer:** Gold Standard refers to the highest quality reference data available, usually created by domain experts under strict guidelines. It is used as the ground truth for evaluation. However, it is *not* guaranteed to be 100% accurate because experts are human and can err or disagree. This is why measuring Inter-Annotator Agreement is crucial even for gold sets.

### Mathematical Questions

**Q3: Why is Accuracy a bad metric for measuring Inter-Annotator Agreement? Derive the intuition behind Cohen's Kappa.** * **Answer:** Accuracy fails because it does not account for random chance. If two annotators diagnose a rare disease (1% prevalence), and both simply guess "Healthy" for every patient, they achieve 99% accuracy but 0% actual agreement on the pathology.
* **Derivation:** We need a metric relative to a baseline. The baseline is $p_e$ (expected agreement by chance). The improvement over chance is $p_o - p_e$. The maximum possible improvement is $1 - p_e$. Kappa is the ratio: $\frac{\text{Actual Improvement}}{\text{Max Possible Improvement}}$.

**Q4: In Active Learning, explain how Entropy Sampling differs from Least Confidence Sampling. When would they yield different results?** * **Answer:**
  * **Least Confidence** considers only the probability of the *single* most likely class ($1 - P(\hat{y}_{max})$).
  * **Entropy** considers the distribution across *all* classes ($-\sum p \log p$).
  * **Difference:** In a binary classification, they are identical. In multi-class, if a model predicts probabilities $[0.4, 0.4, 0.1, 0.1]$, Least Confidence sees $0.6$ uncertainty. If it predicts $[0.4, 0.2, 0.2, 0.2]$, LC still sees $0.6$. However, Entropy will be higher in the second case because the remaining probability mass is more spread out (higher chaos/uncertainty). Entropy is generally more robust for multi-class problems.

### Applied Questions

**Q5: You are building a Named Entity Recognition (NER) system for legal contracts. You have no labeled data but access to 5 lawyers who charge \$500/hour. Design a labeling strategy.** * **Answer:**
  1. **Cold Start with Weak Supervision:** Write heuristic rules (regex for dates, "Section X" patterns) using Snorkel to create a noisy training set. Train a baseline model.
  2. **Active Learning:** Use the baseline model to infer on unlabeled contracts. Select the most uncertain sentences or those containing potential entities.
  3. **Labeling Interface:** Present *only* these high-value samples to the lawyers. Use a "pre-fill" approach (model highlights potential entities, lawyer accepts/rejects) to speed up their workflow.
  4. **Evaluation:** Set aside a small, random sample for lawyers to label from scratch (blind) to create a rigorous Gold Test Set. Do not use Active Learning on the Test Set.

**Q6: How do you handle schema evolution in a production ML pipeline?** * **Answer:**
  1. **Data Contracts:** Enforce schema validation at the ingestion layer (e.g., Protobuf, Avro). If data violates the schema, reject it and alert producers immediately.
  2. **Feature Store Versioning:** Version features (e.g., `user_clicks_v1`, `user_clicks_v2`). The model is pinned to `v1`.
  3. **Backfilling:** When the schema changes, write a translation layer to backfill historical data into the new format so the model can be retrained on a consistent history.

### Debugging & Failure Modes

**Q7: Your model has 95% accuracy on the test set but performs poorly in production. Upon investigation, you find the input data quality is perfect. What labeling issue might cause this?** * **Answer:** This suggests **Label Leakage** or **dataset mismatch**.
  * *Leakage:* The labelers might have had access to information that the model won't have in production (e.g., the "filename" contained the class label, and the model learned to read the filename).
  * *Sampling Bias:* The test set was not a random sample of the production distribution (e.g., test set had only short documents, production has long ones).

**Q8: You notice that over time, your model's performance on the "Other" category is degrading, while specific categories remain accurate. Why?** * **Answer:** This is a classic **Class Discovery / Concept Drift** issue. The "Other" bucket is a catch-all. As the world changes, new specific classes emerge that fall into "Other" (e.g., a new type of fraud). The "Other" class becomes multimodal and complex, making it harder to learn. The fix is to cluster the "Other" samples, identify new distinct classes, and update the labeling schema to include them.

### Follow-up / Probing

* *Interviewer:* "You mentioned crowdsourcing. How do you prevent spammers on MTurk from ruining your dataset?"
* *Response:* Use **Golden Questions** (Honey Pots). Insert known correct answers randomly into the task stream. If a worker gets the known answers wrong, discard their work. Also, check for "robot behavior" (answering too quickly) and require a minimum reputation score.