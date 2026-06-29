# k-Nearest Neighbors (kNN) & Distance Learning

## 1. Motivation & Intuition

### The "Birds of a Feather" Principle
Imagine you move to a new neighborhood and want to guess the political affiliation of your new next-door neighbor. You have no information about them yet. However, you know that the five neighbors living closest to them are all registered as Party A. A reasonable guess would be that your new neighbor also belongs to Party A.

This is the intuition behind **k-Nearest Neighbors (kNN)**. It relies on the assumption of **locality**: similar data points tend to exist in close proximity to one another.

### Real-World Connection
In Machine Learning, we do not guess political parties based on physical addresses, but we do classify data based on "feature space" proximity.
* **Recommendation Systems:** If User A likes movies X, Y, and Z, and User B is "close" to User A (meaning they also liked X and Y), we recommend movie Z to User B.
* **Image Recognition:** To determine if an image contains a "cat," we look for images in our database that are most similar (closest in pixel values) to the new image. If the majority of similar images are cats, we classify the new one as a cat.

---

## 2. Conceptual Foundations

### Core Components
kNN is a **non-parametric**, **lazy learning** algorithm used for both **classification** and **regression**.

1. **The Instance ($x$):** The data point we want to predict (the query).
2. **The Training Set ($D$):** The database of labeled examples $\{x_i, y_i\}$.
3. **Distance Metric ($d$):** A ruler to measure how "close" two points are.
4. **Hyperparameter $k$:** The number of neighbors to consider.

### Step-by-Step Interaction
1. **Store Data:** Unlike other algorithms (like Linear Regression) that learn a formula during training, kNN simply stores the entire training dataset in memory. This is why it is "lazy"—no computation happens until a query is made.
2. **Calculate Distances:** When a new query point arrives, the algorithm calculates the distance between that point and *every* point in the training data.
3. **Find Neighbors:** It sorts these distances and selects the top $k$ points with the smallest distances.
4. **Vote (Classification) or Average (Regression):**
   * **Classification:** The algorithm takes a "majority vote" of the labels of the $k$ neighbors.
   * **Regression:** The algorithm averages the target values of the $k$ neighbors.

### Underlying Assumptions
* **Locality:** Points close to each other have similar target labels.
* **Smoothness:** The decision boundary is not infinitely jagged; changes happen gradually (though small $k$ can create jagged boundaries).
* **Feature Relevance:** All features contribute meaningfully to the distance. If you have 1 meaningful feature and 99 noise features, the distance calculation will be dominated by noise.

**What breaks?** If the "Locality" assumption is violated (e.g., a "checkerboard" pattern where classes flip constantly), kNN fails to generalize.

---

## 3. Mathematical Formulation

### Notation
* Let $x = (x^{(1)}, x^{(2)}, \dots, x^{(d)})$ be a data point in $d$-dimensional space.
* Let $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$ be the training set.
* Let $x_q$ be the query point we wish to predict.

### Distance Metrics
The choice of "ruler" determines which points are considered neighbors.

#### 1. Euclidean Distance ($L_2$ Norm)
The straight-line distance between two points. Standard for continuous, physical data.

$$
d(x_i, x_q) = \sqrt{\sum_{j=1}^{d} (x_i^{(j)} - x_q^{(j)})^2}
$$

#### 2. Manhattan Distance ($L_1$ Norm)
The distance traveling along grid lines (like a taxi in a city block). Preferred in high-dimensional spaces or when features are strictly independent.

$$
d(x_i, x_q) = \sum_{j=1}^{d} |x_i^{(j)} - x_q^{(j)}|
$$

#### 3. Minkowski Distance ($L_p$ Norm)
A generalized form where $p$ is a parameter.

$$
d(x_i, x_q) = \left( \sum_{j=1}^{d} |x_i^{(j)} - x_q^{(j)}|^p \right)^{1/p}
$$

* If $p=1$, it is Manhattan distance.
* If $p=2$, it is Euclidean distance.
* If $p \to \infty$, it is Chebyshev distance (maximum coordinate difference).

#### 4. Mahalanobis Distance
Accounts for the **correlation** between variables. If one feature has a huge variance (spread) and another has small variance, Euclidean distance is biased toward the large variance feature. Mahalanobis normalizes this using the covariance matrix $\Sigma$.

$$
d(x_i, x_q) = \sqrt{(x_i - x_q)^T \Sigma^{-1} (x_i - x_q)}
$$

* If $\Sigma$ is the Identity matrix (uncorrelated, unit variance), this reduces to Euclidean distance.

#### 5. Hamming Distance
Used for categorical or binary strings. It counts the number of positions at which the corresponding symbols are different.

$$
d(x_i, x_q) = \sum_{j=1}^{d} \mathbb{I}(x_i^{(j)} \neq x_q^{(j)})
$$

* Where $\mathbb{I}$ is the indicator function ($1$ if different, $0$ if same).

### Prediction Formulas

**Classification (Majority Vote):**

$$
\hat{y}_q = \text{mode} \{ y_i : x_i \in N_k(x_q) \}
$$

Where $N_k(x_q)$ is the set of the $k$ nearest neighbors.

**Regression (Simple Average):**

$$
\hat{y}_q = \frac{1}{k} \sum_{x_i \in N_k(x_q)} y_i
$$

**Weighted kNN:**
To penalize far-away neighbors, we weight their vote by the inverse of their distance ($w_i = 1/d(x_i, x_q)$).

$$
\hat{y}_q = \frac{\sum_{x_i \in N_k} w_i y_i}{\sum_{x_i \in N_k} w_i}
$$

---

## 4. Worked Example

**Scenario:** We want to predict if a new fruit is an **Apple** or an **Orange** based on two features: **Weight in grams** and **Redness on a 0–10 scale**.

**Training Data:**
1. Fruit A: (150g, 8) $\rightarrow$ Apple
2. Fruit B: (130g, 9) $\rightarrow$ Apple
3. Fruit C: (180g, 2) $\rightarrow$ Orange
4. Fruit D: (170g, 3) $\rightarrow$ Orange

**Query Point:** New Fruit $X$ (160g, 7)  
**Hyperparameter:** $k=3$  
**Metric:** Euclidean  

**Step 1: Calculate Distances**
* $d(A, X) = \sqrt{(150-160)^2 + (8-7)^2} = \sqrt{(-10)^2 + (1)^2} = \sqrt{101} \approx 10.05$
* $d(B, X) = \sqrt{(130-160)^2 + (9-7)^2} = \sqrt{(-30)^2 + (2)^2} = \sqrt{904} \approx 30.07$
* $d(C, X) = \sqrt{(180-160)^2 + (2-7)^2} = \sqrt{(20)^2 + (-5)^2} = \sqrt{425} \approx 20.62$
* $d(D, X) = \sqrt{(170-160)^2 + (3-7)^2} = \sqrt{(10)^2 + (-4)^2} = \sqrt{116} \approx 10.77$

**Step 2: Sort and Select Top $k=3$**
1. Fruit A (10.05) $\rightarrow$ Apple
2. Fruit D (10.77) $\rightarrow$ Orange
3. Fruit C (20.62) $\rightarrow$ Orange  
*(Fruit B is excluded as it is the 4th closest).*

**Step 3: Majority Vote**
* Neighbors: `{Apple, Orange, Orange}`
* Vote Count: Apple = 1, Orange = 2

**Prediction:** The New Fruit $X$ is an **Orange**.

*Note: Notice how "Weight" dominated the calculation (differences of 10–30) versus Redness (differences of 1–5). This highlights the absolute necessity of **Feature Scaling** before running kNN.*

---

## 5. Efficiency Structures & The Curse of Dimensionality

### Naive Approach ($O(N)$)
Calculating distance to every point is computationally expensive. If you have 10 million rows, every single prediction requires 10 million calculations. This yields an inference complexity of $O(N \cdot d)$.

### Tree-Based Indexing ($O(\log N)$)
To accelerate inference, space-partitioning data structures allow the algorithm to prune points that are definitively outside the search radius.

**1. KD-Trees (k-Dimensional Trees):**
* **Concept:** A binary search tree that iteratively bisects the search space into halves using axis-aligned hyperplanes.
* **Algorithm:** At level 0, split data by the $x$-coordinate median. At level 1, split by the $y$-coordinate median. Repeat cyclically across dimensions.
* **Pros:** Extremely fast query times for low-dimensional data ($d < 20$).
* **Cons:** Rapidly degrades to brute-force $O(N)$ search time in high-dimensional spaces.

**2. Ball Trees:**
* **Concept:** Partitions data into nested hyperspheres (balls) rather than rigid Cartesian boxes.
* **Algorithm:** Nodes are defined by a centroid point and a radius that fully encompasses a subset of child points.
* **Pros:** Outperforms KD-Trees in higher dimensions because spheres wrap irregularly distributed data more tightly than rectangular bounding boxes.

**3. Locality-Sensitive Hashing (LSH):**
* **Concept:** An **approximate** nearest neighbor (ANN) method. It uses specialized hash functions designed to collide similar items into the same buckets with high probability.
* **Use Case:** Massive-scale systems (e.g., deduplicating billions of web documents) where trading a small amount of accuracy for orders-of-magnitude speedups is required.

### The Curse of Dimensionality
As the dimensionality ($d$) of the feature space increases, the geometric volume grows exponentially.
1. **Sparsity:** Data points separate drastically. To maintain continuous spatial density across 10 dimensions compared to 1 dimension, an exponentially larger dataset is required.
2. **Distance Convergence:** In high-dimensional space, the ratio of the distance to the nearest point versus the farthest point approaches $1$. Mathematically:

$$
\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0
$$

3. **Result:** The concept of spatial "nearest" loses all predictive meaning.

**Solution:** Apply Dimensionality Reduction techniques (PCA, t-SNE, UMAP) prior to executing kNN.

---

## 6. Relevance to Machine Learning Practice

### System Applications
* **Baseline Modeling:** kNN serves as a standard non-parametric sanity check. If a complex Deep Neural Network cannot decisively outperform a properly tuned kNN, the complex model's architecture or feature set is likely flawed.
* **Missing Value Imputation:** `KNNImputer` estimates missing feature entries by computing the weighted average of the $k$ closest complete samples.
* **Outlier & Anomaly Detection:** Data points exhibiting an unusually large distance to their $k$-th nearest neighbor are flagged as spatial anomalies.

### Architectural Trade-offs

| Dimension | Characteristic | Impact |
| :--- | :--- | :--- |
| **Training Time** | Zero / $O(1)$ | No model parameters are computed; data is strictly indexed. |
| **Inference Time** | Expensive / $O(N \cdot d)$ | Requires spatial search across memory at prediction time. |
| **Memory Footprint**| Heavy | Requires keeping the entire historical training dataset in RAM. |
| **Interpretability** | High | Predictions can be audited directly by inspecting the retrieved neighbors. |
| **Scale Sensitivity**| Extreme | Unscaled features with large numeric magnitudes completely dominate distance metrics. |

---

## 7. Common Pitfalls & Misconceptions

1. **Forgetting to Scale Features**
   * *Mistake:* Feeding raw features like `Annual Salary ($100,000)` alongside `Age (30)`.
   * *Result:* The distance metric is 100% dictated by Salary due to magnitude. Age is functionally ignored.
   * *Fix:* Standardize features using `StandardScaler` ($\mu=0, \sigma=1$) or normalize via `MinMaxScaler` ($[0,1]$).

2. **Arbitrary Hyperparameter Selection**
   * *Mistake:* Blindly guessing $k$ (e.g., defaulting to $k=1$).
   * *Result:* Setting $k=1$ causes severe **Overfitting** (high variance, sensitive to noise). Setting $k$ too high causes **Underfitting** (high bias, over-smoothed boundaries).
   * *Fix:* Execute $k$-fold cross-validation to plot validation error curves. A common starting heuristic is $k \approx \sqrt{N}$.

3. **Deploying Euclidean Metrics on Raw High-Dimensional Data**
   * *Mistake:* Running standard $L_2$ kNN directly on raw text vectors or uncompressed image pixel arrays.
   * *Result:* Near-random predictions driven by distance convergence (Curse of Dimensionality).
   * *Fix:* Project data into lower-dimensional manifolds or swap to angular/ranking metrics (Cosine Similarity).

---

## Applied Scientist Interview Preparation

### Foundational Questions

#### Q1: Explain the Bias-Variance trade-off in the context of tuning the hyperparameter $k$.
**Answer:**
* **Small $k$ (e.g., $k=1$):** Results in **high variance and low bias**. The model memorizes local noise and outliers. The decision boundaries are heavily jagged and isolated islands form around anomalous training points. It overfits.
* **Large $k$ (e.g., $k=N$):** Results in **low variance and high bias**. The model averages out spatial nuances. In the extreme case where $k=N$, the algorithm completely ignores the input query $x_q$ and statically outputs the global majority class of the dataset. It underfits.

#### Q2: Why is kNN classified as a "Lazy Learner"? How does this architectural distinction impact production system deployment?
**Answer:** kNN is lazy because it constructs no generalized discriminative function (like weight matrices or splitting trees) during the training phase. Training is computationally trivial ($O(1)$), consisting merely of storing data in memory. 

In production, this shifts the computational burden entirely to **inference time**. While eager models (e.g., Logistic Regression) execute an $O(d)$ dot-product for real-time predictions, kNN must search memory structures, incurring high latency ($O(\log N)$ to $O(N)$) and requiring substantial server RAM to hold the live dataset.

---

### Mathematical Questions

#### Q3: Under what exact geometric or statistical conditions should a practitioner select Manhattan Distance ($L_1$) over Euclidean Distance ($L_2$)?
**Answer:** Manhattan distance should be selected over Euclidean in two primary scenarios:
1. **High-Dimensional Spaces:** As dimensionality increases, the squaring operation inside the Euclidean metric disproportionately amplifies the largest feature differences. $L_1$ dampens this effect, preserving better relative separation between nearest and farthest neighbors.
2. **Grid-Constrained Features:** When continuous movement along diagonal vectors is physically or logically invalid (e.g., routing algorithms on city blocks, or sparse discrete count data), $L_1$ reflects the true path cost.

#### Q4: State the asymptotic error bound of the 1-Nearest Neighbor classifier relative to the optimal Bayes Error Rate.
**Answer:** Let $E^*$ represent the optimal Bayes Error Rate (the theoretical absolute minimum error achievable by any classifier due to inherent data noise). Cover and Hart (1967) proved that as the dataset size $N \to \infty$, the expected error rate of the 1-NN classifier, $E_{1NN}$, is bounded by:

$$
E^* \leq E_{1NN} \leq 2E^*(1 - E^*)
$$

This demonstrates that strictly memorizing the single closest historical example yields an error rate that is, at worst, less than twice the theoretical perfection limit.

---

### Applied & System Design Questions

#### Q5: You are designing a production recommendation engine over a catalog of 10 million items. Real-time API latency requirements mandate sub-10ms response times. How do you architect the kNN subsystem?
**Answer:**
1. **Bypass Exact Search:** Exact brute-force $O(N \cdot d)$ computation will breach the 10ms SLA.
2. **Reject Exact Trees:** While KD-Trees/Ball Trees offer $O(\log N)$ lookups in theory, modern item embedding vectors typically contain 128 to 1536 dimensions. At this dimensionality, tree pruning fails, reverting search latency back to $O(N)$.
3. **Implement Approximate Nearest Neighbors (ANN):** Deploy vector database indexes utilizing **HNSW (Hierarchical Navigable Small World)** graphs or **IVF-PQ (Inverted File with Product Quantization)** via frameworks like Faiss, Milvus, or ScaNN. These trade ~1–2% recall accuracy for logarithmic lookup speeds (~1–3ms).
4. **Pre-compute & Cache:** For static user profiles, run batch inference offline, writing the computed top-$k$ item IDs directly into a low-latency key-value store (e.g., Redis).

#### Q6: You are modeling a dataset containing mixed modalities: continuous metrics (Age, Income) and high-cardinality categorical attributes (Zip Code, Occupation). How do you formulate a valid distance metric?
**Answer:** Standard Euclidean distance cannot natively evaluate categorical strings. 
1. **Avoid Naive One-Hot Encoding:** Expanding high-cardinality categories into binary vectors creates massive dimensionality expansion, triggering distance sparsity.
2. **Deploy Gower's Distance:** Gower's coefficient computes a normalized composite dissimilarity score $S_{ij} \in [0,1]$ across mixed feature types:

$$
S_{ij} = \frac{\sum_{m=1}^{d} w_m \cdot s_{ij}^{(m)}}{\sum_{m=1}^{d} w_m}
$$

* For **continuous** features: $s_{ij}^{(m)} = 1 - \frac{|x_i^{(m)} - x_j^{(m)}|}{\text{range}(m)}$
* For **categorical** features: $s_{ij}^{(m)} = \mathbb{I}(x_i^{(m)} = x_j^{(m)})$ *(Hamming similarity)*

---

### Debugging & Failure Modes

#### Q7: You train a classification kNN model. It achieves 100% accuracy on the training set, but exactly 50% (random chance) accuracy on a balanced holdout test set. Diagnose the root cause.
**Answer:**
The model is suffering from catastrophic overfitting caused by setting **$k=1$**. 

During training evaluation, when querying a point $x_i$ that exists inside the training set, the algorithm calculates $d(x_i, x_i) = 0$. It retrieves itself as its own sole nearest neighbor, guaranteeing 100% training accuracy. However, because $k=1$ constructs hyper-fragmented Voronoi cells around every individual noise point, the decision boundary fails to generalize to unseen test coordinates.

#### Q8: Upon auditing a deployed kNN model, you discover it outputs the exact same majority class label for 98% of incoming production queries. What went wrong during model pipeline construction?
**Answer:**
This failure mode is triggered by two primary defects:
1. **Unscaled Feature Dominance:** One specific continuous feature (e.g., `Lifetime Platform Spend` ranging from \$0 to \$5,000,000) was left unscaled alongside smaller features (e.g., `Session Count` ranging from 1 to 50). The mathematical distance is entirely dictated by the variance of the massive feature. If that feature correlates weakly with the majority class at higher magnitudes, it pulls all query points toward that class's spatial cluster.
2. **Hyperparameter Saturation:** The value of $k$ was tuned excessively high relative to the size of the dataset ($k \to N$), forcing the model to act as a static global voting tally rather than a local spatial classifier.