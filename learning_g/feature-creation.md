# Feature Creation & Engineering

## 1. Motivation & Intuition

### Why do we need Feature Creation?
Machine learning models are mathematical functions that map inputs to outputs. However, "raw" data—strings of text, timestamps, pixel grids, or GPS coordinates—is rarely in a format that these functions can ingest directly or learn from effectively.

Imagine you are trying to predict if a house will sell for a high price. You have a database with a column `Date_Sold`.

* **Raw Data:** `2023-12-25`
* **The Problem:** A standard regression model (like Linear Regression) treats numbers as magnitudes. It doesn't inherently understand that "December" implies winter or that "2023" is later than "2022" in a way that affects inflation.
* **The Solution (Feature Creation):** You transform `2023-12-25` into new features:
  * `Is_Holiday` (1 if yes, 0 if no)
  * `Days_Since_Listing` (Integer)
  * `Season` (One-hot encoded: Winter)

**The Intuition:** Feature creation is the process of injecting **domain knowledge** into the data. You are acting as a translator, converting raw observations into a representation that makes the underlying patterns obvious to the algorithm.

### Connection to Real-World Systems
In production systems, the "intelligence" often lies more in the features than the model architecture.

* **E-commerce:** A raw User ID is useless. A feature like *"Average purchase value in the last 30 days"* (Aggregation) is predictive.
* **Fraud Detection:** A raw transaction amount of $500 is ambiguous. A feature like *"Ratio of transaction amount to median daily spend"* (Interaction) immediately signals an anomaly.

---

## 2. Conceptual Foundations

### A. Interaction Features
These capture relationships between two or more existing variables.
* **Polynomial Features:** Raising features to a power ($x^2, x^3$) or multiplying them ($x_1 \cdot x_2$). Useful when the relationship between the feature and target is non-linear.
* **Ratio Features:** Dividing one feature by another (e.g., $\frac{\text{Price}}{\text{SquareFootage}}$). This creates a "normalized" view that allows comparison across different scales.
* **Domain Combinations:** Logical combinations, such as `Is_Raining AND Is_Weekend`.

### B. Aggregation Features
Used primarily when dealing with one-to-many relationships (e.g., one user has many clicks). We must summarize the "many" into a fixed set of numbers for the "one."
* **Statistical Aggregates:** Mean, Median, Min, Max, Standard Deviation.
* **Count Features:** Number of occurrences (e.g., "Number of logins").
* **Time-Window Aggregates:** Statistics calculated over a specific recent period (e.g., "Clicks in the last hour" vs. "Clicks in the last 24 hours").

### C. Text Features
Converting unstructured text into vectors.
* **Bag-of-Words (BoW):** Counts word occurrences, ignoring order.
* **TF-IDF:** Weighs words by how unique they are to a specific document compared to the whole corpus.
* **N-grams:** Capturing sequences of $N$ words (e.g., "not happy" is a 2-gram). This captures local context that BoW misses.
* **Character N-grams:** Sequences of characters. Useful for typos or sub-word patterns (e.g., "ing" suffix).

### D. Image Features
Traditional computer vision techniques before Deep Learning often relied on hand-crafted features.
* **Color Histograms:** The distribution of pixel intensities (e.g., how much "red" is in the image?).
* **HOG (Histogram of Oriented Gradients):** Captures the direction of edges and corners. Great for object detection. 
* **SIFT (Scale-Invariant Feature Transform):** Detects "keypoints" that are recognizable even if the image is rotated or zoomed.
* **Pretrained CNN Features:** Using the output of a hidden layer from a deep neural network (like ResNet) trained on ImageNet as a dense feature vector.

### E. Time-Series Features
Transforming a sequence of data points indexed in time order.
* **Lag Features:** The value of the variable at previous time steps ($t-1, t-7$).
* **Rolling Statistics:** Moving average or variance over a sliding window.
* **Fourier/Wavelet Transforms:** Decomposing a signal into its constituent frequencies. Useful for detecting seasonality (e.g., daily cycles).

### F. Geospatial Features
* **Distance Calculations:** Euclidean or Haversine (Great-circle) distance to key points (e.g., "Distance to nearest subway").
* **Spatial Aggregations:** Grouping by zip code or H3 hex to calculate statistics (e.g., "Average income in this grid cell").
* **Density Measures:** Count of points of interest within a radius.

### G. Graph Features
Used when data represents connections (nodes and edges).
* **Node Centrality:** How "important" a node is (e.g., Degree Centrality = number of connections).
* **PageRank:** A recursive measure of influence (you are important if important nodes point to you).
* **Clustering Coefficient:** The probability that two of your friends are also friends with each other.

---

## 3. Mathematical Formulation

### A. Interaction: Polynomials
If we have a feature vector $x = [x_1, x_2]$, a degree-2 polynomial expansion generates:

$$
\phi(x) = [1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]
$$

* $x_1 x_2$ captures the **interaction effect**: does the impact of $x_1$ depend on the value of $x_2$?

### B. Text: TF-IDF
Term Frequency-Inverse Document Frequency reflects how important a word $t$ is to a document $d$ in a corpus $D$.

1. **Term Frequency ($TF$):**
   $$TF(t, d) = \frac{\text{count}(t, d)}{\text{total words in } d}$$

2. **Inverse Document Frequency ($IDF$):**
   $$IDF(t, D) = \log \left( \frac{N}{|\{d \in D : t \in d\}|} \right)$$
   Where $N$ is total documents. If a word appears in *every* document (like "the"), the ratio is 1, and $\log(1) = 0$, so the weight is zero.

3. **TF-IDF:**
   $$TF\text{-}IDF = TF \cdot IDF$$

### C. Time-Series: Fourier Transform
To find cyclical features, we decompose a time series $x_n$ into frequencies $k$:

$$
X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i 2\pi k n / N}
$$

The magnitude $|X_k|$ tells us the strength of the cycle at frequency $k$. If you see a spike at $k$ corresponding to 24 hours, you create a feature $\sin(\frac{2\pi t}{24})$.

### D. Graph: PageRank
The rank $PR(u)$ of node $u$ is given by:

$$
PR(u) = \frac{1-d}{N} + d \sum_{v \in B(u)} \frac{PR(v)}{L(v)}
$$

* $d$: Damping factor (usually 0.85, probability of continuing to click).
* $B(u)$: Set of nodes linking *to* $u$.
* $L(v)$: Number of outbound links from node $v$.

**Intuition:** Your score is a sum of your neighbors' scores, divided by how generous they are with their links.

---

## 4. Worked Example: Ride-Sharing Demand Prediction

**Scenario:** We want to predict the number of ride requests in a specific city zone for the next hour.

**Raw Data:**
* `Timestamp`: 2023-10-31 18:00:00
* `Zone_ID`: 45 (Midtown)
* `Weather`: "Heavy Rain"
* `Zone_Graph`: Adjacency list of city zones.

**Step-by-Step Feature Creation:**

1. **Time Features (Temporal):**
   * Extract `Hour`: 18.
   * Extract `DayOfWeek`: Tuesday.
   * **Cyclical Encoding:** Converting hour 18 into coordinates so 23 and 0 are close:
     $$x_{\text{hour}} = \sin\left(\frac{2\pi \times 18}{24}\right), \quad y_{\text{hour}} = \cos\left(\frac{2\pi \times 18}{24}\right)$$

2. **Weather Features (Text/Categorical):**
   * One-hot encoding `Weather`: `[0, 1, 0]` (assuming Rain is index 1).
   * **Interaction:** `Is_Rush_Hour` (Hour $\in [17, 19]$) $\times$ `Is_Raining`. This captures the specific chaos of wet rush hours.

3. **Lag Features (Time-Series):**
   * `Demand_T-1`: Demand at 17:00.
   * `Demand_T-24`: Demand at 18:00 yesterday (captures daily seasonality).

4. **Graph Features (Spatial):**
   * Use `Zone_Graph` to find neighbors of Zone 45.
   * **Spatial Aggregation:** Calculate average demand in *neighboring* zones at $T-1$. If demand is high next door, it might spill over.

5. **Resulting Feature Vector:**

   $$X = [0.86, -0.5, \dots, 1, 1, 150, 140, 110]$$

   *(Cyclical Time, One-Hot Weather, Interaction, Lags, Neighbor Aggregates)*

---

## 5. Relevance to Machine Learning Practice

### When to use what?
* **Tree-based models (XGBoost, Random Forest):**
  * Handle raw interactions poorly; explicitly adding **Interaction Features** often boosts performance.
  * Do not require scaling (e.g., Ratio features don't strictly need normalization).
* **Linear Models / Neural Networks:**
  * Struggle with non-linearities; **Polynomial Features** are critical for linear models.
  * **Standardization** is mandatory.
* **Deep Learning (Transformers/CNNs):**
  * Often automate feature extraction for Text and Images. You rarely do manual HOG or TF-IDF for state-of-the-art NLP/Vision anymore, but you *do* still need aggregations and time-features for tabular inputs.

### Trade-offs
* **Dimensionality:** Adding n-grams or polynomial features can explode the feature space ($N^2$ or more), leading to the **Curse of Dimensionality** (model overfits, sparse data).
* **Interpretability:** `Age` is interpretable. `PCA_Component_4` or `Age^2 * Income` is harder to explain to stakeholders.
* **Computational Cost:** Calculating Graph Centrality or complex Window Aggregations can be slow during inference (latency).

---

## 6. Common Pitfalls & Misconceptions

1. **Data Leakage (The Cardinal Sin):**
   * *Mistake:* Calculating "Mean Target Encoding" or "Global Average" using the *entire* dataset (including the test set or future data).
   * *Correction:* Always calculate aggregations using only *past* data (for time-series) or strictly on the *training* fold (for cross-validation).

2. **Look-Ahead Bias:**
   * *Mistake:* Using a lag feature like `Demand_Next_Hour` or aggregating over a window that includes the current timestamp in a way that isn't available at inference time.
   * *Correction:* Ensure strictly causal timestamping ($t-1$ onwards).

3. **Ignoring Feature Scaling:**
   * *Mistake:* Using Euclidean distance features (Geospatial) mixed with Price (thousands) and Age (double digits) in a K-Means or KNN algorithm.
   * *Correction:* Normalize all features to similar scales (e.g., Z-score).

4. **Spurious Correlations:**
   * *Mistake:* Generating 10,000 interaction features and selecting the best ones based on training score.
   * *Consequence:* You will find patterns that are pure noise (statistical flukes) and fail in production.

---

## 7. Applied Scientist Interview Guide

### Foundational Questions

**Q1. What is the "Curse of Dimensionality" and how does feature engineering contribute to it?** **Answer:** As the number of features increases, the volume of the feature space increases exponentially, making the data sparse. This makes it harder for models to generalize (overfitting). Generating n-grams (Text) or high-degree polynomials (Interaction) are common causes. Solutions include Regularization (L1/L2), PCA, or Feature Selection.

**Q2. Explain the difference between Label Encoding and One-Hot Encoding. When would you use one over the other?** **Answer:** **Label Encoding** assigns an integer (1, 2, 3) to categories. It introduces an ordinal relationship ($3 > 1$) which may not exist (e.g., Red vs Blue). Use it for ordinal data (Low, Med, High) or Tree models that can handle it natively. **One-Hot Encoding** creates binary columns for each category. It is safer for linear models and neural networks, but increases dimensionality significantly.

### Mathematical Questions

**Q3. Derive the dimension of the feature space if we apply a Degree-2 Polynomial expansion to a dataset with $d$ features.** **Answer:** A degree-2 expansion includes:
1. The bias term ($1$ feature: $1$)
2. Original features ($d$ features: $x_i$)
3. Squared terms ($d$ features: $x_i^2$)
4. Interaction terms (Combinations of 2 from $d$: $\binom{d}{2} = \frac{d(d-1)}{2}$)

Summing these parts gives the total dimension:

$$
\text{Total} = 1 + 2d + \frac{d^2 - d}{2} = \frac{d^2 + 3d + 2}{2} = \binom{d+2}{2}
$$

*(Verification check: If $d=2$, the longhand expansion yields $1 + 2 + 2 + 1 = 6$ terms. Plugging into the formula yields $\frac{4 + 6 + 2}{2} = 6$. The formula holds).*

**Q4. Why do we take the Logarithm of Inverse Document Frequency in TF-IDF?** **Answer:** * **Smoothing:** It dampens the scaling effect of the IDF. Without the log, a very rare word would scale linearly to an astronomical weight, dominating the vector space.
* **Information Theory:** $\log(1/p)$ defines the amount of "self-information" or "surprise" contained in an event with probability $p$. Taking the log maps the raw document occurrence rate directly onto a standard Shannon Information scale.

### Applied Questions

**Q5. You are building a model to predict credit card fraud. You have transaction data (User, Amount, Merchant, Time). What features do you build?** **Answer:**
* **Aggregations (User History):** *"Average spend in last 30 days"*, *"Standard deviation of user spend"*.
* **Interactions (Anomaly Detection):** `Current_Amount` / `User_30d_Avg_Spend`. If this ratio $> 10$, it flags high risk.
* **Velocity (Time-Series):** *"Time elapsed since last transaction"*. Rapid-fire bursts of transactions are strongly correlated with stolen cards.
* **Graph:** *"PageRank of Merchant"*. Allows the model to evaluate whether the target merchant sits inside a cluster of known fraudulent nodes.

**Q6. How do you handle "New Users" (Cold Start) when relying heavily on User-Aggregation features?** **Answer:**
* **Imputation:** Fill the missing historical aggregations with the global median of all active users.
* **Look-alike mapping:** Group the new user by known onboarding demographics (e.g., Country + Signup Device) and impute the subgroup's average.
* **Explicit Signaling:** Introduce a binary indicator column `Is_New_User`. This informs the tree splits or network weights that the accompanying aggregated values are synthetic baseline defaults rather than observed behavior.

### Debugging & Failure Modes

**Q7. Your NLP classifier achieves 96% training accuracy, but degrades month-over-month in production. You engineered Bag-of-Words and TF-IDF features on customer support tickets. What happened?** **Answer:** **Vocabulary Drift.** The TF-IDF vectorizer locked its vocabulary matrix to the exact terms present in the static training set. As new product releases occur, customers introduce new jargon, error codes, and slang. The production vectorizer silently drops these out-of-vocabulary (OOV) words, passing sparse zero-vectors to the classifier. *Fix:* Transition to a dynamic Hashing Vectorizer, or schedule an automated monthly vocabulary re-fit pipeline.

**Q8. You created a "Distance to City Center" geospatial feature for a real estate pricing model. It achieved state-of-the-art results in Chicago, but failed completely when deployed to Los Angeles. Why?** **Answer:** **Domain Assumption Violation.** The feature implicitly assumed a *monocentric* urban topology (a single central business district dictating land value, like Chicago the Loop). Los Angeles is *polycentric*—it possesses multiple decentralized economic hubs (Santa Monica, Downtown, Century City, Pasadena). *Fix:* Replace the single-point distance measure with a spatial density feature (e.g., *"Number of commercial POIs within a 3-mile radius"* or *"Travel time to nearest major employment cluster"*).