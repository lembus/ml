# Versioning in Machine Learning Systems

---

## Table of Contents

1. [Model Versioning](#1-model-versioning)
2. [Data Versioning](#2-data-versioning)
3. [Code Versioning](#3-code-versioning)
4. [Experiment Tracking](#4-experiment-tracking)
5. [Reproducibility](#5-reproducibility)
6. [Interview Preparation](#6-interview-preparation)

---

# 1. Model Versioning

## 1.1 Motivation & Intuition

### Why Model Versioning Exists

Imagine you train a fraud-detection model on Monday. It works well, so your team deploys it to production on Tuesday. On Wednesday, a colleague retrains the model with a new feature. On Thursday, you discover that Tuesday's model was actually better for a certain class of transactions, but nobody saved it — the weights were overwritten, the hyperparameters weren't recorded, and the preprocessing pipeline that produced the training data was modified in place. You have no way to recover what was deployed 48 hours ago.

This scenario is not hypothetical. It is the default outcome when machine learning teams treat models the way early software teams treated code — as ephemeral files on someone's laptop. Model versioning exists to solve a cluster of problems that all stem from the same root cause: **a trained ML model is a complex artifact with many invisible dependencies, and without systematic tracking, it becomes impossible to reproduce, compare, audit, or roll back.**

### The Problem in Concrete Terms

A "model" is not just a file of weights. It is the product of:

- A specific dataset (which rows, which features, which preprocessing)
- A specific codebase (which architecture, which loss function, which data loader)
- A specific set of hyperparameters (learning rate, batch size, regularization)
- A specific software environment (library versions, CUDA version, hardware)
- A specific training run (random seed, order of data, number of epochs)

Change any one of these and you get a different model. Model versioning is the practice of recording all of these dependencies alongside the model artifact itself, so that any model can be uniquely identified, retrieved, compared against alternatives, and reproduced.

### Real-World Problems Model Versioning Solves

**Rollback.** Production model performance degrades after an update. You need to instantly revert to the previous version. Without versioning, "the previous version" does not exist in any retrievable form.

**A/B testing.** You want to compare model v3 against model v4 in production. This requires both models to be stored, retrievable, and associated with metadata that describes how they differ.

**Audit and compliance.** In regulated industries (finance, healthcare), you must be able to answer: "Which model made this decision on this date, and what data was it trained on?" This is a legal requirement under regulations like GDPR's right to explanation and the EU AI Act.

**Debugging.** A model produces an unexpected prediction. Was it trained on a dataset that contained a labeling error? Was a feature engineered differently? You need to trace backward from the deployed artifact to every input that produced it.

**Collaboration.** Multiple data scientists experiment with different architectures simultaneously. Without a registry, they overwrite each other's work, lose track of which experiment produced which result, and cannot systematically compare outcomes.

---

## 1.2 Conceptual Foundations

### Key Components

**Model Registry.** A centralized catalog that stores model artifacts and their metadata. Think of it as a database where each entry is a trained model, and each entry has rich metadata: who trained it, when, on what data, with what hyperparameters, and what its evaluation metrics are. Examples include MLflow Model Registry, AWS SageMaker Model Registry, Weights & Biases Model Registry, and Vertex AI Model Registry.

A model registry typically organizes models hierarchically:

```
Registry
  └── Registered Model (e.g., "fraud-detector")
        ├── Version 1
        │     ├── Artifact (weights file)
        │     ├── Metadata (hyperparams, metrics, author, date)
        │     ├── Stage (e.g., "Production")
        │     └── Lineage (dataset hash, code commit, environment)
        ├── Version 2
        │     └── ...
        └── Version 3
              └── ...
```

The **Registered Model** is a logical name for a model that serves a particular purpose (e.g., "churn-predictor"). Each **Version** under that name is a specific trained instance. Versions can be assigned **Stages** (or **Aliases** in newer systems) such as "Staging," "Production," or "Archived" to indicate their lifecycle status.

**Lineage Tracking.** The record of how a model came into existence. Lineage answers the question: "Given this model artifact, what exact sequence of data, code, and computation produced it?" Lineage is a directed acyclic graph (DAG) where nodes are artifacts (datasets, code snapshots, model files) and edges represent transformations (training, preprocessing, evaluation).

A complete lineage record for a model includes:

- The training dataset version (a hash or pointer to the exact data)
- The code commit (the Git SHA of the training script)
- The environment specification (Docker image digest or conda lockfile)
- The hyperparameters used
- The parent model, if any (for fine-tuned or incrementally trained models)
- The evaluation dataset and metrics

**Artifact Storage.** The physical storage of model files — weight tensors, configuration files, tokenizer vocabularies, preprocessing objects (scalers, encoders), and any other file required to reconstruct the model for inference. Artifact storage must handle:

- **Large files.** A single model checkpoint can be tens of gigabytes (GPT-3 scale models are hundreds of gigabytes).
- **Immutability.** Once stored, an artifact should never be modified in place. If you retrain, you create a new version.
- **Content addressing.** Artifacts are identified by content hashes (e.g., SHA-256), so two identical artifacts are recognized as the same regardless of when or where they were produced.
- **Access control.** Not everyone should be able to promote a model to production or delete an old version.

Common backends for artifact storage include cloud object stores (S3, GCS, Azure Blob Storage), artifact-specific stores (MLflow's artifact store, Weights & Biases artifact storage), and container registries (for models packaged as Docker images).

### How Components Interact

The typical workflow proceeds as follows:

1. **Train.** A data scientist runs a training script. The script logs hyperparameters and metrics to an experiment tracker (covered in Section 4).
2. **Register.** When a training run produces a satisfactory model, the artifact is registered in the model registry. This creates a new version under the appropriate registered model name. The registry records the artifact's storage location, its metadata, and its lineage (links to the dataset version, code commit, and environment).
3. **Stage.** The new version is assigned a stage. It might start as "None" or "Staging." After validation, it can be promoted to "Production."
4. **Serve.** The serving infrastructure reads from the registry to determine which version is currently in "Production" and loads the corresponding artifact from storage.
5. **Monitor.** Production metrics are fed back and associated with the model version, enabling comparison across versions over time.
6. **Rollback.** If the new production model underperforms, the previous version's stage is restored to "Production," and the serving infrastructure loads the older artifact.

### Assumptions and What Breaks When They Are Violated

**Assumption: Artifacts are immutable.** The registry assumes that a version, once registered, never changes. If someone modifies a model file in place (e.g., overwrites the weights at the same storage path), the registry's metadata becomes a lie — it points to an artifact that no longer matches the recorded hash. This breaks reproducibility and makes rollback unreliable.

**Assumption: Metadata is complete.** If a model is registered without its hyperparameters, or without a link to the training data version, the lineage is broken. You can retrieve the artifact, but you cannot reproduce or audit it. This commonly happens when teams adopt a registry after-the-fact and backfill metadata inconsistently.

**Assumption: The serving layer reads from the registry.** If deployments are done by manually copying files rather than by querying the registry for the current production version, the registry becomes a passive record rather than the source of truth. Drift between the registry and actual production state is a common failure mode.

**Assumption: Storage is durable.** If artifacts are stored on local disk or an ephemeral filesystem, they can be lost when machines are decommissioned. Cloud object stores with replication and versioning are the standard solution.

---

## 1.3 Mathematical Formulation

Model versioning is primarily an engineering discipline rather than a mathematical one, but there are formal concepts that underpin it.

### Content Addressing

A model artifact $A$ is identified by its content hash:

$$
\text{id}(A) = H(A)
$$

where $H$ is a cryptographic hash function (e.g., SHA-256). Two artifacts $A_1$ and $A_2$ are considered identical if and only if $H(A_1) = H(A_2)$. The probability of a collision (two different artifacts producing the same hash) is approximately $2^{-128}$ for SHA-256, which is negligible for all practical purposes.

Content addressing provides:
- **Deduplication:** If the same model is registered twice, the storage layer can recognize the duplicate and avoid storing it twice.
- **Integrity verification:** When an artifact is retrieved, its hash can be recomputed and compared to the stored hash. A mismatch indicates corruption or tampering.

### Lineage as a Directed Acyclic Graph

Model lineage can be formalized as a DAG $G = (V, E)$ where:

- $V$ is a set of versioned artifacts: datasets $D_i$, code snapshots $C_j$, environment specifications $E_k$, and model artifacts $M_l$.
- $E$ is a set of directed edges representing transformations. An edge $(D_i, M_l)$ means dataset $D_i$ was used to produce model $M_l$. An edge $(C_j, M_l)$ means code snapshot $C_j$ was used. An edge $(M_l, M_m)$ means model $M_l$ was fine-tuned to produce $M_m$.

The graph is acyclic because a model cannot be an input to the process that produced it.

Given a model $M_l$, its complete lineage is the set of all ancestors in the DAG:

$$
\text{Lineage}(M_l) = \{ v \in V \mid \text{there exists a directed path from } v \text{ to } M_l \}
$$

This set contains every artifact and transformation that contributed to $M_l$'s existence.

### Version Ordering

Versions within a registered model are typically ordered by creation timestamp $t$:

$$
M^{(1)}, M^{(2)}, \ldots, M^{(n)} \quad \text{where } t(M^{(i)}) < t(M^{(i+1)})
$$

A stage assignment is a function $S: \{M^{(1)}, \ldots, M^{(n)}\} \to \{\text{None}, \text{Staging}, \text{Production}, \text{Archived}\}$ with the constraint that at most one version is in "Production" at any time:

$$
|\{ M^{(i)} : S(M^{(i)}) = \text{Production} \}| \leq 1
$$

Some modern registries relax this constraint using **aliases** instead of stages, allowing multiple production models (e.g., one per region or one per use case) under the same registered model name.

---

## 1.4 Worked Examples

### Example: End-to-End Model Versioning with MLflow

Suppose you are building a loan default prediction model. Here is the complete workflow.

**Step 1: Train and log.** Your training script trains a gradient-boosted tree and logs everything to MLflow:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

mlflow.set_experiment("loan-default")

with mlflow.start_run() as run:
    # Log parameters
    params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}
    mlflow.log_params(params)

    # Train
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    mlflow.log_metric("test_auc", auc)  # e.g., 0.847

    # Log the dataset hash for lineage
    import hashlib
    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(X_train).values.tobytes()
    ).hexdigest()[:12]
    mlflow.log_param("train_data_hash", data_hash)

    # Log the model artifact
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {run.info.run_id}")
    # Output: Run ID: a1b2c3d4e5f6...
```

At this point, the model is stored as an artifact inside the MLflow tracking server. It is associated with the run ID, which links it to the parameters, metrics, and data hash.

**Step 2: Register the model.** After reviewing the metrics, you register the model:

```python
model_uri = f"runs:/{run.info.run_id}/model"
model_version = mlflow.register_model(model_uri, "loan-default-predictor")
# Creates version 1 under registered model "loan-default-predictor"
```

The registry now contains:
| Field | Value |
|-------|-------|
| Registered Model | loan-default-predictor |
| Version | 1 |
| Run ID | a1b2c3d4e5f6 |
| Test AUC | 0.847 |
| n_estimators | 200 |
| max_depth | 5 |
| learning_rate | 0.1 |
| Train data hash | 7f3a2b... |
| Created | 2025-03-15 14:22:00 |

**Step 3: Promote to production.** After staging tests pass:

```python
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("loan-default-predictor", "production", "1")
```

**Step 4: Serve.** The serving layer queries the registry:

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/loan-default-predictor@production")
prediction = model.predict(new_application_data)
```

**Step 5: Rollback scenario.** Version 2 is trained and promoted, but production monitoring shows increased false negatives. Rollback is a single metadata change:

```python
client.set_registered_model_alias("loan-default-predictor", "production", "1")
# Version 1 is instantly restored as the production model
# No retraining, no file copying — just a pointer change
```

The serving layer, on its next refresh, loads version 1. Total rollback time: seconds.

---

## 1.5 Relevance to Machine Learning Practice

### Where Model Versioning Is Used

**Training → Registry → Deployment pipeline.** In production ML systems (Uber Michelangelo, Google TFX, Meta's FBLearner), the model registry is the bridge between training and serving. No model reaches production without passing through the registry.

**A/B testing.** Two registry versions are deployed simultaneously to different traffic segments. Metrics are tracked per version. The winner is promoted; the loser is archived.

**Multi-model systems.** A recommendation system might use separate models for candidate retrieval, ranking, and re-ranking. Each has its own registered model with independent versioning. The system's overall version is the tuple of individual model versions: (retriever-v3, ranker-v7, reranker-v2).

**Regulatory compliance.** Financial institutions must maintain model inventories. A model registry serves as this inventory, with lineage providing the audit trail required by regulators.

### When to Use It

Use model versioning from the very first model you intend to deploy, not "later when things get complex." The cost of setting up a registry is low. The cost of reconstructing a lost model from scratch — retraining, re-evaluating, re-validating — is high.

### When Not to Use It

For throwaway exploratory notebooks where you are trying ideas and will never deploy the result, full registry overhead is unnecessary. However, even in exploration, logging runs to an experiment tracker (Section 4) is still good practice.

### Trade-offs

| Dimension | Model Versioning | No Versioning |
|-----------|-----------------|---------------|
| Reproducibility | Full: any version can be reconstructed | None: lost when overwritten |
| Rollback speed | Seconds (pointer change) | Hours to days (retrain) |
| Storage cost | Higher (every version is stored) | Lower (only latest exists) |
| Complexity | Requires infrastructure setup | No setup |
| Collaboration | Multiple scientists work in parallel | Serial, conflict-prone |
| Auditability | Complete lineage trail | No trail |

The storage cost trade-off is manageable: model compression, deduplication (via content hashing), and tiered storage (hot/warm/cold) keep costs reasonable even for large model catalogs.

---

## 1.6 Common Pitfalls & Misconceptions

**Pitfall: Versioning only the weights file.** A model artifact without its preprocessing pipeline, feature schema, and tokenizer vocabulary is incomplete. If the scaler that normalized features is not stored alongside the model, the model will produce garbage predictions when loaded in a new environment. Always package the complete inference pipeline.

**Pitfall: Using file naming as versioning.** Naming files `model_v1.pkl`, `model_v2_final.pkl`, `model_v2_final_ACTUALLY_final.pkl` is not versioning. It provides no metadata, no lineage, no atomic promotion, and no rollback mechanism.

**Pitfall: Mutable "latest" pointers without history.** Some teams maintain a single "latest" pointer that always points to the newest model. If the pointer is updated before the new model is validated, and the old model's location is not recorded, rollback is impossible.

**Pitfall: Ignoring the training-serving skew in versioning.** The model was trained with scikit-learn 1.2 but served with scikit-learn 1.3. A version number in the registry does not protect you from this. The environment specification must be part of the versioned artifact.

**Misconception: "We use Git for model versioning."** Git is designed for text files. Storing multi-gigabyte binary model files in Git is impractical — it bloats the repository, slows clones, and provides no model-specific metadata (metrics, stage, lineage). Git LFS partially addresses file size but not metadata. Use Git for code, a model registry for models.

**Misconception: "Model versioning is only for production."** Even in research, systematic versioning prevents the common scenario of "I got a great result last week but I can't reproduce it." The cost of logging is trivial compared to the cost of lost work.

**Pitfall: Not versioning the feature transform.** You register model v5 with its weights, but the feature engineering pipeline that creates the input features is not versioned. When the pipeline is updated for model v6, model v5 can no longer be served correctly because its expected input schema no longer matches the pipeline output.

---

# 2. Data Versioning

## 2.1 Motivation & Intuition

### Why Data Versioning Exists

Consider a team training a sentiment analysis model. On Monday, the training dataset has 100,000 labeled reviews. On Wednesday, a data engineer fixes a labeling bug that affected 2,000 reviews, changing their labels from positive to negative. On Friday, another team member adds 15,000 new reviews. A model trained on Friday's data will behave differently from one trained on Monday's data, but if nobody recorded what the dataset looked like on Monday, the difference is invisible.

Data versioning exists because **data is the most impactful and least controlled input to ML systems.** A one-line code change is visible in a Git diff. A 2% label change in a million-row dataset is invisible without data versioning.

### The Scope of the Problem

Data in ML systems comes in many forms, each with versioning challenges:

- **Raw data.** Log files, images, sensor readings. Often append-only and growing continuously.
- **Processed data.** The output of ETL pipelines — cleaned, filtered, transformed. Changes when the pipeline code changes or when raw data changes.
- **Feature data.** The actual input to models — numeric vectors, embeddings, tokenized sequences. Changes when feature engineering code changes.
- **Labels.** Human annotations, ground-truth targets. Change when annotators correct mistakes or when labeling guidelines are updated.

Each of these can change independently. A version of the model's training data is the intersection of a specific raw data snapshot, a specific processing pipeline, and a specific set of labels. Data versioning tracks all of these.

### Concrete Problems Data Versioning Solves

**Reproducing past results.** "What data did the model that was in production last month train on?" Without data versioning, this question is unanswerable. With it, the model's lineage includes a pointer to the exact dataset version.

**Debugging data quality.** A model's accuracy drops. Is it a model problem or a data problem? If you can compare the current training data against the previous version, you can identify changes — new mislabeled examples, a broken feature, a distribution shift in a source — and isolate the cause.

**Compliance and audit.** "Was this user's data used to train the model that denied their loan application?" Answering this requires knowing the exact contents of the training set at the time the model was trained.

**Time travel.** You want to evaluate a new model architecture against the dataset as it existed three months ago, to compare fairly with a model that was trained at that time. Without data versioning, you train on today's data, introducing temporal leakage into the comparison.

---

## 2.2 Conceptual Foundations

### DVC (Data Version Control)

DVC is a tool that brings Git-like version control to data files. The core idea: **Git tracks small metadata files (pointers); DVC tracks the large data files those pointers reference.**

Here is how DVC works, step by step:

1. You have a large file, say `train.csv` (5 GB). You run `dvc add train.csv`.
2. DVC computes the MD5 hash of `train.csv`, say `a1b2c3d4...`.
3. DVC moves `train.csv` to a cache directory (`.dvc/cache/a1/b2c3d4...`).
4. DVC creates a small file `train.csv.dvc` containing the hash and file metadata:
   ```yaml
   outs:
     - md5: a1b2c3d4e5f6...
       size: 5368709120
       path: train.csv
   ```
5. You commit `train.csv.dvc` to Git. The actual data file is in `.gitignore`.
6. When someone else clones the repo, they run `dvc pull`. DVC reads `train.csv.dvc`, finds the hash, and retrieves the file from the configured remote storage (S3, GCS, etc.).

The result: **Git history tracks the sequence of data versions (via the `.dvc` files), and each version points to the actual data in a content-addressable store.** You can check out any past Git commit and run `dvc checkout` to get the exact data that existed at that point.

**DVC Pipelines.** DVC also tracks data transformations as pipelines defined in a `dvc.yaml` file:

```yaml
stages:
  preprocess:
    cmd: python preprocess.py
    deps:
      - raw_data.csv
      - preprocess.py
    outs:
      - processed_data.csv

  train:
    cmd: python train.py
    deps:
      - processed_data.csv
      - train.py
    outs:
      - model.pkl
    metrics:
      - metrics.json:
          cache: false
```

When you run `dvc repro`, DVC checks which stages have changed inputs (by comparing hashes) and only re-runs those stages. This provides both versioning and reproducibility of the entire data pipeline.

### Delta Lake

Delta Lake is a storage layer that adds versioning, ACID transactions, and schema enforcement to data stored in cloud object stores (typically as Parquet files). It is designed for large-scale data (terabytes to petabytes) in data lake architectures.

The key concept is the **transaction log** (also called the Delta log). Every change to a Delta table — inserts, updates, deletes, schema changes — is recorded as a JSON entry in a `_delta_log/` directory:

```
my_table/
  _delta_log/
    00000000000000000000.json   # Table creation
    00000000000000000001.json   # First batch of inserts
    00000000000000000002.json   # An update operation
    00000000000000000003.json   # A delete operation
  part-00000-...snappy.parquet
  part-00001-...snappy.parquet
  ...
```

Each log entry records which Parquet files were added and which were removed by that transaction. To reconstruct the table at any version, Delta Lake replays the log up to that version and reads the corresponding Parquet files. This is called **time travel**:

```sql
-- Read the table as it was at version 5
SELECT * FROM my_table VERSION AS OF 5;

-- Read the table as it was at a specific timestamp
SELECT * FROM my_table TIMESTAMP AS OF '2025-01-15 10:00:00';
```

Delta Lake provides several properties critical for ML data versioning:

- **ACID transactions.** Concurrent writes do not corrupt the table. A half-written batch is never visible to readers.
- **Schema enforcement.** You cannot accidentally insert a row with a missing column or a wrong data type.
- **Schema evolution.** You can add new columns without breaking existing readers.
- **Audit history.** The transaction log is a complete record of who changed what, when.
- **Efficient versioning.** Because Delta Lake stores the log of changes (not full copies), versioning a 1 TB table that receives 1 GB of daily updates costs only ~1 GB per version, not 1 TB.

### Feature Store Versioning

A feature store is a centralized system for managing features — the numeric/categorical values that serve as input to ML models. Examples include Feast, Tecton, Hopsworks, and Vertex AI Feature Store. Feature stores serve two purposes:

1. **Consistency.** The features used in training match the features used in inference. This eliminates **training-serving skew**, where the model was trained on features computed one way but served features computed a different way.
2. **Reuse.** Features engineered by one team can be discovered and reused by other teams, avoiding redundant computation.

Feature store versioning operates at two levels:

**Feature definition versioning.** The code that computes a feature (e.g., "average transaction amount over the last 30 days") is versioned. When the computation logic changes (e.g., switching from mean to median), a new version of the feature definition is created. Old models can continue to use the old definition; new models use the new one.

**Feature value versioning.** The actual computed values are versioned by time. A feature store maintains a time-indexed record of feature values, so you can query: "What was user X's average_transaction_amount as of January 15, 2025 at 3:00 PM?" This is essential for:

- **Point-in-time correctness.** When constructing training data, you must use feature values as they were known at the time of each training example, not the current values. Otherwise, you introduce **label leakage** (using future information to predict the past).
- **Reproducibility.** To retrain a model identically, you need the exact feature values that were used, not today's values.

The architecture typically looks like:

```
Feature Definition (versioned code)
        │
        ▼
Feature Pipeline (batch or streaming computation)
        │
        ▼
Feature Store
  ├── Online Store (low-latency, current values for serving)
  └── Offline Store (historical values for training)
        │
        ▼
Training Data (point-in-time join of features and labels)
```

### Assumptions and Violations

**Assumption: Data is immutable once versioned.** DVC and Delta Lake both assume that a version, once created, is never modified. If someone manually edits a Parquet file in a Delta table without going through the transaction log, the version history becomes inconsistent.

**Assumption: The storage backend is reliable.** DVC pointers are useless if the remote storage (S3 bucket) is deleted. Delta Lake's transaction log is useless if the underlying Parquet files are corrupted. Both require durable, replicated storage.

**Assumption: Feature pipelines are deterministic.** If a feature computation involves randomness (e.g., sampling) or external API calls (e.g., geocoding), the same input data will produce different feature values on different runs. Feature store versioning captures the output, but reproducing the computation requires controlling all sources of non-determinism.

**Assumption: Schema is stable or evolves in compatible ways.** If a feature is renamed, deleted, or changes type without proper versioning, models that depend on the old schema will break. Feature stores mitigate this by enforcing schema contracts, but enforcement requires discipline.

---

## 2.3 Mathematical Formulation

### Content-Based Versioning

A dataset $D$ consisting of files $\{f_1, f_2, \ldots, f_k\}$ is versioned by computing a Merkle hash:

$$
H(D) = H\bigl(H(f_1) \| H(f_2) \| \cdots \| H(f_k)\bigr)
$$

where $\|$ denotes concatenation. This means:

- If any single file changes, $H(D)$ changes.
- If the set of files changes (a file is added or removed), $H(D)$ changes.
- If nothing changes, $H(D)$ remains the same, regardless of when or where you compute it.

DVC uses this approach. The `.dvc` file stores the hash of the output, and the pipeline's `dvc.lock` file stores the hashes of all inputs and outputs for each stage.

### Delta Lake's Transaction Log Formalism

A Delta table at version $v$ is defined by the set of active files:

$$
T(v) = \bigcup_{i=0}^{v} \text{add}(i) \setminus \bigcup_{i=0}^{v} \text{remove}(i)
$$

where $\text{add}(i)$ is the set of Parquet files added in transaction $i$ and $\text{remove}(i)$ is the set removed. Each transaction is atomic: either all adds and removes in transaction $i$ take effect, or none do.

The difference between two versions is:

$$
\Delta(v_1, v_2) = T(v_2) \setminus T(v_1) \cup T(v_1) \setminus T(v_2)
$$

This symmetric difference tells you which files were added or removed between versions, enabling efficient comparison.

### Point-in-Time Feature Retrieval

Given an entity $e$ (e.g., a user), a feature $f$, and a query timestamp $t_q$, the point-in-time correct value is:

$$
\text{value}(e, f, t_q) = f_e(t^*) \quad \text{where } t^* = \max\{ t : t \leq t_q \text{ and } f_e(t) \text{ exists} \}
$$

In words: retrieve the most recent feature value that was known at or before the query time. This prevents future information from leaking into historical queries.

For a training example with event timestamp $t_\text{event}$, the correct feature vector is:

$$
\mathbf{x} = \bigl[\text{value}(e, f_1, t_\text{event}), \text{value}(e, f_2, t_\text{event}), \ldots, \text{value}(e, f_d, t_\text{event})\bigr]
$$

If you instead use $\text{value}(e, f_j, t_\text{now})$ — the current value — you leak future information into the training data, inflating apparent model performance and degrading real-world performance.

---

## 2.4 Worked Examples

### Example: DVC Workflow for an Image Classification Dataset

You are building an image classifier. Your dataset is 10,000 images in a folder `data/images/` with labels in `data/labels.csv`.

**Step 1: Initialize DVC.**

```bash
cd my_project
git init
dvc init
```

This creates a `.dvc/` directory (DVC's internal state) and a `.dvcignore` file.

**Step 2: Track the data.**

```bash
dvc add data/images
dvc add data/labels.csv
git add data/images.dvc data/labels.csv.dvc data/.gitignore
git commit -m "v1: initial dataset, 10k images"
git tag data-v1
```

DVC creates `data/images.dvc` (with the hash of the entire `images/` directory) and `data/labels.csv.dvc` (with the hash of the labels file). The actual files are moved to `.dvc/cache/`.

**Step 3: Configure remote storage.**

```bash
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push
```

The cached data files are uploaded to S3.

**Step 4: Data changes — a colleague fixes labels.**

Your colleague corrects 200 mislabeled images in `labels.csv` and adds 500 new images:

```bash
# After modifying labels.csv and adding images to data/images/
dvc add data/images
dvc add data/labels.csv
git add data/images.dvc data/labels.csv.dvc
git commit -m "v2: fix 200 labels, add 500 images"
git tag data-v2
dvc push
```

Now Git history has two commits with two different `.dvc` files. Each `.dvc` file points to a different content hash.

**Step 5: Retrieve a past data version.**

You want to retrain with the original dataset to investigate a regression:

```bash
git checkout data-v1
dvc checkout
# data/images/ now contains the original 10,000 images
# data/labels.csv now contains the original labels
```

After retraining:

```bash
git checkout main
dvc checkout
# Back to the current dataset
```

### Example: Point-in-Time Feature Join

You have a feature store with user transaction features:

| user_id | feature | value | event_time |
|---------|---------|-------|------------|
| u1 | avg_txn_30d | 45.20 | 2025-01-01 |
| u1 | avg_txn_30d | 52.10 | 2025-01-15 |
| u1 | avg_txn_30d | 48.75 | 2025-02-01 |

You have training examples (labels) with timestamps:

| user_id | label | label_time |
|---------|-------|------------|
| u1 | 0 | 2025-01-10 |
| u1 | 1 | 2025-01-20 |

**Point-in-time correct join:**

For the first example (label_time = 2025-01-10):
- The most recent feature value at or before 2025-01-10 is 45.20 (from 2025-01-01).
- Feature vector: [45.20].

For the second example (label_time = 2025-01-20):
- The most recent feature value at or before 2025-01-20 is 52.10 (from 2025-01-15).
- Feature vector: [52.10].

**Incorrect (leaky) join:**

If you naively join on user_id without time filtering, both examples might get the latest value (48.75 from 2025-02-01). This means the first training example, which occurred on January 10, uses a feature value that did not exist until February 1. The model learns from future information.

---

## 2.5 Relevance to Machine Learning Practice

### Where Data Versioning Is Used

**Training pipelines.** Every training run should record which data version it used. This is the single most important use of data versioning.

**Feature stores.** Companies like Uber, Airbnb, and Spotify operate large feature stores that version both feature definitions and feature values.

**Data quality monitoring.** By comparing data versions over time, you can detect distribution shifts, missing values, or schema changes before they affect model performance.

**Regulatory compliance.** GDPR's right to erasure requires knowing which datasets contain a user's data. Data versioning with lineage tracking makes this possible.

### Trade-offs

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| DVC | Git-native, works with any file type, pipeline support | Requires separate storage backend, not designed for tabular data updates |
| Delta Lake | ACID transactions, time travel, schema enforcement, efficient for large tables | Tied to Spark/Databricks ecosystem, Parquet-only |
| Feature Store | Point-in-time correctness, training-serving consistency | Infrastructure overhead, requires organizational adoption |
| Manual snapshots | Simple, no tooling needed | No deduplication, expensive storage, no change tracking |

### When to Use What

- **Small datasets (< 10 GB), file-based (images, audio, text files):** DVC.
- **Large tabular datasets (> 10 GB) with frequent updates:** Delta Lake (or Apache Iceberg, Apache Hudi — competing formats with similar capabilities).
- **Features shared across teams and models:** Feature store.
- **Prototyping:** Manual snapshots are acceptable temporarily, but migrate to a proper system before production.

---

## 2.6 Common Pitfalls & Misconceptions

**Pitfall: Versioning data without versioning the schema.** You track that the dataset changed, but not how. Was a column added? Was a column's type changed from float to int? Schema changes can silently break models. Delta Lake enforces schemas; DVC does not — you must add schema validation yourself.

**Pitfall: Confusing data versioning with data backup.** Backups protect against data loss. Versioning tracks the history of intentional changes. A backup strategy does not tell you what the dataset looked like two weeks ago if it has been modified since then.

**Pitfall: Not versioning labels separately from features.** Labels often change independently of features (due to re-annotation, labeling error correction, or guideline updates). If labels and features are versioned as a single monolithic dataset, you cannot isolate the impact of a label change.

**Pitfall: Using timestamps as version identifiers.** "The dataset from January 15" is ambiguous if the dataset was modified three times on January 15. Content hashes are unambiguous.

**Misconception: "We copy the dataset to a new folder for each version."** This works for tiny datasets but is prohibitively expensive for large ones (100 GB × 100 versions = 10 TB). Content-addressable storage (DVC, Delta Lake) stores only the changed portions, reducing storage by orders of magnitude.

**Misconception: "Delta Lake is just Parquet with extra files."** The transaction log is what makes Delta Lake fundamentally different from raw Parquet. Without it, you have no atomicity, no versioning, no time travel, and no concurrent-write safety.

**Pitfall: Ignoring data pipeline versioning.** You version the output dataset but not the code that produced it. When you need to regenerate the dataset (e.g., for a new data source), you cannot reproduce the exact same output because the pipeline has changed. DVC pipelines address this by versioning the pipeline definition alongside the data.

---

# 3. Code Versioning

## 3.1 Motivation & Intuition

### Why Code Versioning for ML Is Different

Software engineers have used Git for over a decade. ML engineers use Git too, but ML code has properties that make versioning more complex than traditional software:

1. **Code produces artifacts, not just behavior.** In traditional software, the code *is* the product. In ML, the code is a recipe that produces a model, which is the product. The same code with different data or different random seeds produces different models.

2. **Experiments branch more aggressively than features.** A software team might have 5-10 active feature branches. An ML team might have 50 active experiments, each a slight variation of the training script (different architecture, different hyperparameter, different data augmentation). These experiments are not "features" and do not cleanly merge.

3. **The runtime environment matters more.** A Python web application runs the same on Python 3.10 and 3.11 in most cases. An ML training script can produce different results (or crash) depending on the CUDA version, PyTorch version, or even the GPU hardware. The environment is part of the code.

4. **Configuration is as important as code.** The difference between two experiments is often a YAML configuration file, not the training script itself. Versioning the script without the config is incomplete.

### The Core Problem

Without disciplined code versioning, ML teams encounter:

- **"It worked on my machine."** A colleague cannot reproduce your result because they have a different library version.
- **Lost experiments.** You tried 20 hyperparameter combinations last week but did not commit each one. Now you cannot determine which combination produced the best result.
- **Merge conflicts in notebooks.** Jupyter notebooks are JSON files. Git diffs of JSON are unreadable. Two people editing the same notebook produce merge conflicts that are nearly impossible to resolve.
- **Configuration drift.** The training script reads from a config file. Someone changes the config but does not commit the change. The committed code no longer reproduces the result.

---

## 3.2 Conceptual Foundations

### Git Workflows for ML

**Trunk-Based Development with Experiment Branches.** The most common approach for ML teams:

```
main (stable, deployable training code)
  ├── experiment/new-architecture
  ├── experiment/larger-batch-size
  ├── experiment/data-augmentation-v2
  ├── feature/add-monitoring
  └── bugfix/data-loader-memory-leak
```

The `main` branch contains code that is known to work — it can train a model, evaluate it, and deploy it. Experiment branches diverge from `main` to try variations. When an experiment succeeds, its changes are merged back into `main`. When it fails, the branch is preserved (not deleted) as a record of what was tried.

Key practices:

- **Every experiment is a commit.** Before running an experiment, commit the code and config. After the run, the experiment tracker (Section 4) links the results to this commit SHA.
- **Config files are committed.** Hyperparameters, data paths, architecture specifications — everything that affects the experiment outcome is in version control.
- **Notebooks are for exploration, scripts are for experiments.** Notebooks are hard to version, hard to review, and hard to run programmatically. Convert notebooks to scripts before running experiments.

**GitFlow for ML.** Some teams use a more structured workflow:

```
main ← only tagged, released model-training code
  └── develop ← integration branch for validated changes
        ├── experiment/* ← individual experiments
        └── feature/* ← infrastructure changes
```

This adds overhead but provides clearer separation between exploratory work and production-ready code.

### Container Image Tagging

A trained model is the product of code + data + environment. The environment includes:

- Operating system and version
- Python version
- Library versions (PyTorch, TensorFlow, scikit-learn, etc.)
- System libraries (CUDA, cuDNN, NCCL)
- Hardware drivers

Docker containers encapsulate all of this into a single, reproducible artifact. Container image tagging is the practice of versioning these containers so that any training run can be executed in exactly the same environment.

**Image Naming Convention:**

```
registry.example.com/ml-team/training-env:py3.10-cuda11.8-torch2.1-v3
                │            │                        │
                │            │                        └── Tag (version identifier)
                │            └── Repository (logical grouping)
                └── Registry (where images are stored)
```

**Tagging Strategies:**

1. **Semantic versioning.** `training-env:1.2.3` — major.minor.patch. Increment major for breaking changes (e.g., new CUDA version), minor for new library additions, patch for bug fixes. Clear but requires manual management.

2. **Content-based tags.** `training-env:sha-a1b2c3d4` — the tag is derived from the image's content hash or the Git commit that produced the Dockerfile. Unambiguous but not human-readable.

3. **Date-based tags.** `training-env:2025-03-15` — the tag is the build date. Simple but does not convey what changed.

4. **Combined tags.** `training-env:1.2.3-sha-a1b2c3d4` — both human-readable version and content hash for unambiguity.

**Image Digest vs. Tag.** A Docker tag is a mutable pointer — `training-env:latest` can point to a different image tomorrow. An image digest (`sha256:a1b2c3d4...`) is immutable — it always refers to the same image. For reproducibility, always record the digest, not just the tag.

```bash
# Pull by tag (mutable — might change):
docker pull registry.example.com/training-env:v3

# Pull by digest (immutable — always the same):
docker pull registry.example.com/training-env@sha256:a1b2c3d4e5f6...
```

### What Belongs in Version Control

| Artifact | Version Control? | Where? |
|----------|-----------------|--------|
| Training scripts | Yes | Git |
| Configuration files (YAML, JSON) | Yes | Git |
| Dockerfiles | Yes | Git |
| Requirements files (requirements.txt, conda.yml) | Yes | Git |
| Jupyter notebooks | Cautiously | Git (with nbstripout to remove outputs) |
| Model weights | No | Model registry / artifact store |
| Training data | No | DVC / Delta Lake / feature store |
| Docker images | No | Container registry |
| Secrets (API keys, credentials) | Never | Secret manager (Vault, AWS Secrets Manager) |

---

## 3.3 Mathematical Formulation

Code versioning is primarily an engineering practice, but the formal underpinning of Git is a **content-addressable Merkle DAG.**

### Git's Object Model

Every object in Git is identified by its SHA-1 hash:

- A **blob** is a file's contents: $\text{id}(\text{blob}) = \text{SHA1}(\text{"blob " + size + "\0" + content})$
- A **tree** is a directory listing (pointers to blobs and other trees): $\text{id}(\text{tree}) = \text{SHA1}(\text{"tree " + size + "\0" + entries})$
- A **commit** points to a tree and its parent commit(s): $\text{id}(\text{commit}) = \text{SHA1}(\text{"commit " + size + "\0" + tree + parents + metadata})$

Because each commit's hash depends on its tree (which depends on all file contents) and its parent commit (which depends on all prior history), **any change to any file at any point in history produces a different commit hash for all subsequent commits.** This is what makes Git's history tamper-evident.

### Reproducibility Tuple

A complete experiment is identified by the tuple:

$$
\text{Experiment} = (C, D, E, H, S)
$$

where:
- $C$ = code commit hash (Git SHA)
- $D$ = data version hash (DVC hash or Delta Lake version)
- $E$ = environment hash (Docker image digest)
- $H$ = hyperparameter configuration (serialized and hashed)
- $S$ = random seed

Two experiments with the same tuple should (in theory) produce identical results. In practice, non-determinism in GPU operations, floating-point order of operations, and multi-threaded data loading can introduce small variations even with the same seed and environment.

---

## 3.4 Worked Examples

### Example: Complete Git Workflow for an ML Experiment

**Starting state:** You have a working training pipeline on `main`.

**Step 1: Create an experiment branch.**

```bash
git checkout -b experiment/resnet50-lr-sweep
```

**Step 2: Modify the configuration.**

```yaml
# config/experiment.yaml
model:
  architecture: resnet50
  pretrained: true
training:
  learning_rate: 0.001  # Changed from 0.01
  batch_size: 64
  epochs: 50
  optimizer: adam
data:
  version: "data-v2"  # DVC tag
  augmentation: "standard"
```

**Step 3: Commit.**

```bash
git add config/experiment.yaml
git commit -m "experiment: ResNet50 with lr=0.001"
```

**Step 4: Record the full reproducibility tuple.** Your training script logs:

```python
import subprocess, hashlib, json

git_sha = subprocess.check_output(
    ["git", "rev-parse", "HEAD"]
).decode().strip()

docker_digest = subprocess.check_output(
    ["docker", "inspect", "--format", "{{.Id}}", "training-env:v3"]
).decode().strip()

config_hash = hashlib.sha256(
    json.dumps(config, sort_keys=True).encode()
).hexdigest()[:12]

experiment_tracker.log_params({
    "git_sha": git_sha,
    "docker_digest": docker_digest,
    "config_hash": config_hash,
    "data_version": "data-v2",
    "seed": 42
})
```

**Step 5: Run the experiment.**

```bash
docker run --gpus all -v $(pwd):/workspace training-env:v3 \
    python train.py --config config/experiment.yaml --seed 42
```

**Step 6: Review results and merge (if successful).**

```bash
git checkout main
git merge experiment/resnet50-lr-sweep
git tag model-v4  # Tag the code version that produced the new model
```

**Step 7: Tag the Docker image for the release.**

```bash
docker tag training-env:v3 registry.example.com/training-env:model-v4
docker push registry.example.com/training-env:model-v4
```

Now the complete lineage is recorded: code (git tag `model-v4`), data (DVC tag `data-v2`), environment (Docker image `training-env:model-v4`), config (committed in the repo), seed (logged in experiment tracker).

---

## 3.5 Relevance to Machine Learning Practice

### Code Versioning in Production ML Systems

**CI/CD for ML.** Continuous integration for ML means that every commit to `main` triggers automated tests: data validation, model training (on a small subset), evaluation, and comparison against a baseline. If the new model meets performance thresholds, it is automatically registered and deployed. The Git commit is the trigger for this entire pipeline.

**Infrastructure as Code.** The deployment infrastructure (Kubernetes manifests, Terraform configs, Airflow DAGs) is versioned alongside the ML code. When a model version is deployed, the infrastructure version is recorded, enabling rollback of both the model and its serving infrastructure.

**Monorepo vs. Polyrepo.** Some ML teams keep all code (training, serving, data pipelines, infrastructure) in a single repository (monorepo). Others separate them. Monorepos simplify cross-cutting changes (e.g., renaming a feature requires changes in data pipeline, training, and serving) but become unwieldy at scale. Polyrepos provide independence but require coordination.

### Common Alternatives

| Approach | Use Case | Limitation |
|----------|----------|------------|
| Git | All code and config | Not designed for large binary files |
| Git LFS | Large files in Git | Storage costs, limited metadata |
| Notebooks in Git | Exploration | Merge conflicts, non-reproducible outputs |
| nbstripout + Git | Clean notebooks | Loses output history |
| Papermill | Parameterized notebooks | Adds complexity |

---

## 3.6 Common Pitfalls & Misconceptions

**Pitfall: Not committing before running experiments.** If you run an experiment from uncommitted code, the Git SHA does not represent the actual code that ran. The experiment is unreproducible. Always commit (even to a throwaway branch) before running.

**Pitfall: Hardcoding paths and credentials.** Config files committed to Git should not contain absolute paths (`/home/alice/data/`) or credentials. Use environment variables or a secrets manager.

**Pitfall: Using mutable Docker tags for training.** If your training script runs in `training-env:latest`, and someone pushes a new image to that tag, your next training run uses a different environment without any explicit change. Always pin to a specific tag or digest.

**Pitfall: Divergent experiment branches that are never cleaned up.** After 6 months, you have 200 stale experiment branches. Nobody knows which ones were successful. Practice: delete branches after merging or after deciding the experiment failed; preserve the record in the experiment tracker, not in Git branches.

**Misconception: "Git tracks everything we need for reproducibility."** Git tracks code and config. It does not track data, environment, or random state. You need additional systems (DVC, Docker, experiment trackers) to achieve full reproducibility.

**Misconception: "We can just re-run the code to get the same model."** Even with the same code, different data versions, library versions, hardware, or random seeds produce different models. "Re-runnable" is not "reproducible" without controlling all inputs.

**Pitfall: Ignoring notebook hygiene.** Committing notebooks with large outputs (plots, dataframes, model summaries) bloats the repository and creates massive diffs. Use `nbstripout` as a Git filter to automatically remove outputs before committing.

---

# 4. Experiment Tracking

## 4.1 Motivation & Intuition

### Why Experiment Tracking Exists

Training a machine learning model is inherently an experimental process. You try different architectures, different hyperparameters, different data preprocessing strategies, and different training regimes. Each combination produces a different result. Experiment tracking is the systematic recording of what you tried, what happened, and what you learned.

Without experiment tracking, the following scenario is universal: you train 30 models over two weeks, each with different settings. You find the best one based on validation metrics. Three days later, a colleague asks: "What learning rate did the best model use? What batch size? Which dataset version?" You check your notes. You find cryptic entries like "lr=0.01, good results" and "tried larger model, OOM." You cannot reconstruct which run produced which result.

Experiment tracking exists to replace informal notes with a structured, queryable, automated system.

### What Gets Tracked

An experiment tracking system records four categories of information for each training run:

1. **Parameters.** All inputs that define the experiment: hyperparameters (learning rate, batch size, dropout rate), architecture choices (number of layers, hidden size), data settings (dataset version, augmentation strategy), and any other configurable value. These are set before the run begins and do not change during the run.

2. **Metrics.** All outputs that measure the experiment's outcome: training loss, validation loss, accuracy, AUC, F1 score, latency, throughput. Metrics are typically logged at multiple points during training (e.g., per epoch or per step) to provide a time series of how the model learned.

3. **Artifacts.** Any file produced during the experiment: the trained model weights, plots (learning curves, confusion matrices), evaluation reports, sample predictions, error analyses. Artifacts are stored by the tracking system and linked to the run that produced them.

4. **Metadata.** Contextual information about the run: who launched it, when it started and ended, how long it took, what hardware it ran on, what Git commit and Docker image were used, whether it succeeded or failed, and any tags or notes the experimenter added.

### Concrete Example

Here is what a single experiment record looks like:

```
Run ID: run_2025_03_15_142200
Status: Completed
Duration: 2h 14m
User: alice

Parameters:
  model.architecture: transformer
  model.num_layers: 6
  model.hidden_size: 512
  model.num_heads: 8
  training.learning_rate: 3e-4
  training.batch_size: 32
  training.epochs: 50
  training.optimizer: adamw
  training.weight_decay: 0.01
  data.version: data-v3
  data.augmentation: random_crop+horizontal_flip
  environment.git_sha: a1b2c3d
  environment.docker_digest: sha256:e5f6g7h8...
  seed: 42

Metrics (final):
  train_loss: 0.234
  val_loss: 0.312
  val_accuracy: 0.891
  val_f1: 0.876
  test_accuracy: 0.883
  test_f1: 0.869
  training_time_hours: 2.23
  gpu_memory_peak_gb: 11.4

Metrics (time series):
  val_loss: [0.85, 0.62, 0.51, 0.43, 0.38, ..., 0.312]
  val_accuracy: [0.55, 0.68, 0.75, 0.80, 0.83, ..., 0.891]

Artifacts:
  model_checkpoint: s3://artifacts/run_2025.../model.pt
  learning_curve_plot: s3://artifacts/run_2025.../learning_curve.png
  confusion_matrix: s3://artifacts/run_2025.../confusion_matrix.png
  predictions_sample: s3://artifacts/run_2025.../sample_predictions.csv

Tags: ["transformer", "baseline", "v3-data"]
Notes: "First transformer baseline on v3 data. Comparable to CNN-v2 but slower to converge."
```

This record is self-contained. Months later, anyone can look at this record and understand exactly what was done and what happened.

---

## 4.2 Conceptual Foundations

### Architecture of an Experiment Tracking System

The core components are:

**Tracking Server.** A backend service that receives and stores experiment data. It exposes an API that training scripts call to log parameters, metrics, and artifacts. Examples: MLflow Tracking Server, Weights & Biases backend, Neptune.ai, Comet ML, ClearML.

**Client Library.** A Python (or other language) library that training scripts import to log data. The client sends data to the tracking server via HTTP or gRPC.

**Storage Backend.** Where the actual data lives. Parameters and metrics are typically stored in a database (PostgreSQL, MySQL, SQLite). Artifacts are stored in object storage (S3, GCS, Azure Blob). The tracking server mediates access to both.

**UI.** A web interface for browsing, comparing, and visualizing experiments. Key features include:
- **Table view:** Sort and filter runs by any parameter or metric.
- **Comparison view:** Side-by-side comparison of selected runs.
- **Plot view:** Overlaid learning curves, parallel coordinates plots, scatter plots of hyperparameters vs. metrics.
- **Artifact viewer:** Display images, tables, and text artifacts inline.

**Experiment / Run Hierarchy.** Most systems organize runs into experiments (logical groups). An experiment might be "fraud-detector-architecture-search" containing 50 runs, each a different architecture. This prevents a single flat list of thousands of runs.

### The Experiment Lifecycle

```
1. CONFIGURE
   Define parameters → Record in config file → Commit to Git

2. LAUNCH
   Start run in tracker → Log parameters → Log environment metadata

3. TRAIN
   Per step/epoch: log metrics (loss, accuracy, learning rate)
   Periodically: save checkpoints as artifacts

4. EVALUATE
   Log final metrics → Log evaluation artifacts (confusion matrix, etc.)

5. ANALYZE
   Compare runs in UI → Identify best performers → Record observations

6. DECIDE
   Promote best model to registry → Archive failed runs → Plan next experiments
```

### Parameters vs. Metrics vs. Artifacts vs. Metadata

Understanding the distinction is important:

**Parameters** are independent variables — things you choose. They define the experiment.
- Examples: learning rate, architecture name, number of layers, dropout rate, dataset version.
- Logged once at the start of the run.
- Used for filtering and grouping: "show me all runs with learning_rate < 0.001."

**Metrics** are dependent variables — things you measure. They describe the outcome.
- Examples: loss, accuracy, F1, AUC, latency, memory usage.
- Logged repeatedly during training (per step or per epoch) and once at the end (final values).
- Used for ranking: "sort runs by val_accuracy descending."

**Artifacts** are files associated with the run. They provide detailed evidence.
- Examples: model weights, plots, predictions, error analysis reports.
- Logged at specific points (end of training, end of evaluation).
- Used for inspection: "show me the confusion matrix for the best run."

**Metadata** is context about the run itself, typically recorded automatically.
- Examples: start time, end time, duration, user, status (running/completed/failed), hardware, Git SHA.
- Used for filtering: "show me all runs by alice in the last week."

### Organizing Experiments

As the number of runs grows from tens to thousands, organization becomes critical:

**Experiments (Projects).** Group runs by objective. "churn-prediction-v2" contains all runs aimed at improving the churn model.

**Tags.** Label runs with free-form tags for ad-hoc grouping. Tags like "baseline," "ablation," "final-candidate" help filter runs across experiments.

**Run Groups.** Some tools support grouping runs within an experiment. A hyperparameter sweep might produce 100 runs that are logically a single group.

**Notes and Descriptions.** Free-text annotations on individual runs record qualitative observations that metrics do not capture: "model oscillated after epoch 30, suggesting learning rate too high" or "suspiciously high accuracy — possible data leak."

---

## 4.3 Mathematical Formulation

### Experiment as a Function

An experiment can be modeled as a function:

$$
f: \Theta \times \mathcal{D} \times \mathcal{E} \times \mathcal{S} \to \mathcal{M}
$$

where:
- $\Theta$ is the hyperparameter space (learning rate, batch size, architecture, etc.)
- $\mathcal{D}$ is the space of dataset versions
- $\mathcal{E}$ is the space of environments (library versions, hardware)
- $\mathcal{S}$ is the space of random seeds
- $\mathcal{M}$ is the metric space (loss, accuracy, etc.)

A single experiment run is an evaluation of $f$ at a specific point:

$$
m = f(\theta, d, e, s)
$$

Experiment tracking is the systematic recording of the inputs $(\theta, d, e, s)$ and the output $m$ for each evaluation.

### Hyperparameter Optimization as Search

Hyperparameter optimization is a search over $\Theta$ to find:

$$
\theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{s \sim \mathcal{S}}[L(\theta, d, e, s)]
$$

where $L$ is the validation loss. The expectation over seeds accounts for stochastic variation. In practice, we evaluate a finite number of seeds and report the mean (or sometimes a single seed for computational reasons).

The experiment tracker records each evaluation, enabling:
- **Visualization** of the search trajectory.
- **Early stopping** of unpromising runs by monitoring learning curves.
- **Transfer** of knowledge between experiments (e.g., "learning rates above 0.01 consistently diverge for this architecture").

### Metric Time Series

For a run with $T$ training steps, the metric time series is:

$$
\{(t_1, l_1), (t_2, l_2), \ldots, (t_T, l_T)\}
$$

where $t_i$ is the step number and $l_i$ is the metric value at step $i$.

The tracker stores this time series and enables:
- **Learning curve visualization:** Plot $l_i$ vs. $t_i$ for one or more runs.
- **Early stopping criterion:** Stop training when $l_i - l_{i-k} < \epsilon$ for patience $k$ and threshold $\epsilon$.
- **Convergence comparison:** Two runs can be compared not just by their final metric but by their convergence behavior (e.g., one converges faster but to a worse optimum).

---

## 4.4 Worked Examples

### Example: MLflow Experiment Tracking

**Step 1: Set up MLflow.**

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("text-classification-bert")
```

**Step 2: Train with tracking.**

```python
with mlflow.start_run(run_name="bert-base-lr3e5") as run:
    # Log parameters
    mlflow.log_params({
        "model": "bert-base-uncased",
        "learning_rate": 3e-5,
        "batch_size": 16,
        "epochs": 5,
        "max_seq_length": 128,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "dataset_version": "v2.1",
        "git_sha": "a1b2c3d"
    })

    for epoch in range(5):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        # Log metrics per epoch
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        }, step=epoch)

        print(f"Epoch {epoch}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
        # Epoch 0: val_acc=0.8234, val_f1=0.8102
        # Epoch 1: val_acc=0.8712, val_f1=0.8645
        # Epoch 2: val_acc=0.8891, val_f1=0.8823
        # Epoch 3: val_acc=0.8945, val_f1=0.8901
        # Epoch 4: val_acc=0.8967, val_f1=0.8934

    # Log final test metrics
    test_loss, test_acc, test_f1 = evaluate(model, test_loader)
    mlflow.log_metrics({
        "test_accuracy": test_acc,   # 0.8912
        "test_f1": test_f1           # 0.8867
    })

    # Log artifacts
    save_confusion_matrix(model, test_loader, "confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    save_model(model, "model/")
    mlflow.log_artifacts("model/", artifact_path="model")

    # Tag the run
    mlflow.set_tags({
        "model_family": "bert",
        "experiment_type": "baseline"
    })
```

**Step 3: Compare runs.**

After running multiple experiments (different learning rates, batch sizes, etc.), use the MLflow UI or programmatic API:

```python
from mlflow import MlflowClient
client = MlflowClient()

# Get all runs in the experiment, sorted by validation F1
experiment = client.get_experiment_by_name("text-classification-bert")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_f1 DESC"],
    max_results=5
)

for run in runs:
    lr = run.data.params["learning_rate"]
    bs = run.data.params["batch_size"]
    f1 = run.data.metrics["val_f1"]
    print(f"lr={lr}, bs={bs} → val_f1={f1:.4f}")

# Output:
# lr=3e-5, bs=16 → val_f1=0.8934
# lr=5e-5, bs=16 → val_f1=0.8901
# lr=3e-5, bs=32 → val_f1=0.8867
# lr=1e-4, bs=16 → val_f1=0.8743
# lr=1e-5, bs=16 → val_f1=0.8621
```

**Step 4: Promote the best model.**

```python
best_run = runs[0]
model_uri = f"runs:/{best_run.info.run_id}/model"
mlflow.register_model(model_uri, "text-classifier")
```

This connects experiment tracking (Section 4) to model versioning (Section 1): the registered model's lineage includes the run ID, which links to all parameters, metrics, and artifacts.

---

## 4.5 Relevance to Machine Learning Practice

### Where Experiment Tracking Is Used

**Hyperparameter optimization.** Every HPO method (grid search, random search, Bayesian optimization, Hyperband) produces many runs. The tracker records all of them, enabling analysis of which hyperparameters matter most.

**Architecture search.** Neural architecture search (NAS) can produce thousands of runs. The tracker is the only way to manage this scale.

**Ablation studies.** Systematically removing components to measure their contribution requires careful tracking. "What is the accuracy without data augmentation, without dropout, without the attention mechanism?"

**Model comparison.** Before deploying a new model, compare it against the current production model. The tracker provides the metrics for both.

**Debugging.** "Why did this run fail?" The tracker shows the parameters that were used, the metric trajectory before failure, and any logged artifacts (e.g., gradient norms that reveal exploding gradients).

**Knowledge management.** A team's experiment tracker is an institutional memory. New team members can browse past experiments to understand what has been tried, what worked, and what failed.

### Tool Comparison

| Feature | MLflow | W&B | Neptune | ClearML |
|---------|--------|-----|---------|---------|
| Open source | Yes | No (free tier) | No (free tier) | Yes |
| Self-hosted | Yes | Enterprise | Enterprise | Yes |
| Model registry | Yes | Yes | Yes | Yes |
| Hyperparameter viz | Basic | Excellent | Good | Good |
| Collaboration | Basic | Excellent | Good | Good |
| Artifact storage | Any backend | W&B servers | Neptune servers | Any backend |
| System metrics (GPU, CPU) | No (needs plugin) | Automatic | Automatic | Automatic |

### Trade-offs

**Tracking granularity.** Logging every training step produces detailed learning curves but generates large amounts of data. Logging only per epoch is lighter but may miss short-lived phenomena (e.g., a loss spike). A common middle ground: log every $N$ steps, where $N$ is chosen so that a typical run produces 500-1000 data points.

**Tracking overhead.** The tracking client adds latency to each logging call. For fast training loops (thousands of steps per second), synchronous logging can become a bottleneck. Solutions include asynchronous logging (buffering data and sending in batches) and local logging with post-hoc upload.

**What to track.** The temptation is to track everything. In practice, tracking too many irrelevant metrics creates noise that obscures signal. Track metrics that you will actually use for decisions: validation metrics that determine model selection, system metrics that determine infrastructure requirements, and diagnostic metrics that aid debugging.

---

## 4.6 Common Pitfalls & Misconceptions

**Pitfall: Logging parameters after the run starts.** If you log hyperparameters midway through training, they may not be recorded if the run crashes. Log all parameters at the beginning.

**Pitfall: Not logging the environment.** You track hyperparameters and metrics but not the PyTorch version, CUDA version, or GPU type. When you try to reproduce a result on different hardware, you get different metrics and cannot determine whether the difference is due to the hardware or a bug.

**Pitfall: Inconsistent parameter names.** One run logs `lr=0.001`, another logs `learning_rate=0.001`, a third logs `optim.lr=0.001`. These are the same parameter with different names, making comparison impossible. Establish naming conventions early.

**Pitfall: Not logging failed runs.** Failed runs (out of memory, diverging loss, bug in data loader) contain valuable information: they define the boundary of what does not work. Deleting failed runs loses this information.

**Misconception: "Experiment tracking replaces a lab notebook."** Tracking systems record quantitative data. They do not record qualitative reasoning: why you chose a particular architecture, what hypothesis you were testing, what you learned from the results. Complement automated tracking with written notes.

**Misconception: "More metrics is always better."** Logging 200 metrics per step when you only ever look at 5 creates storage costs and UI clutter. Be intentional about what you track.

**Pitfall: Using the experiment tracker as the only model store.** Experiment trackers store model artifacts, but they are not model registries. A model in the tracker is "an artifact produced by run X." A model in a registry is "the production fraud detector, version 3." The registry adds lifecycle management (stages, aliases, access control) that the tracker does not provide.

---

# 5. Reproducibility

## 5.1 Motivation & Intuition

### Why Reproducibility Matters

Reproducibility is the ability to obtain the same results when repeating an experiment under the same conditions. In ML, this means: given the same data, code, environment, and random seed, you should get the same (or very nearly the same) trained model and evaluation metrics.

Reproducibility matters for several reasons, each with concrete consequences:

**Scientific validity.** If a result cannot be reproduced, it might be an artifact of a bug, a lucky random seed, or data leakage. A non-reproducible result is untrustworthy. In research, irreproducible results waste the time of everyone who tries to build on them.

**Debugging.** When a model behaves unexpectedly in production, you need to reproduce the problem to diagnose it. If you cannot recreate the exact model, you cannot systematically isolate the cause.

**Collaboration.** When a colleague says "I got 93% accuracy," you need to reproduce their result before you can meaningfully compare your approach.

**Regulatory compliance.** In regulated industries, auditors may require you to demonstrate that a model can be retrained from its recorded inputs and produce the same output. Non-reproducibility is a compliance failure.

**Iteration.** Making progress requires controlled experiments: change one thing, measure the effect. If the baseline is not reproducible (you get different results each time you retrain without changing anything), you cannot distinguish the effect of your change from random variation.

### Why ML Reproducibility Is Hard

ML reproducibility is harder than traditional software reproducibility because of the number of hidden variables:

1. **Non-determinism in GPU computation.** GPU operations like `cudnn.benchmark`, atomicAdd in backward passes, and parallel reduction operations do not guarantee deterministic results. The order of floating-point additions can change between runs, producing slightly different gradients.

2. **Data ordering.** If the data loader shuffles data differently between runs, the model sees training examples in a different order, producing different gradient updates and a different final model.

3. **Library versions.** A minor version change in PyTorch (e.g., 1.13 to 2.0) can change default behaviors, numerical implementations, or API semantics.

4. **Hardware differences.** Different GPU architectures (V100 vs. A100) use different CUDA kernels with different numerical properties. A model trained on a V100 may not exactly reproduce on an A100 even with the same code and data.

5. **Distributed training.** Multi-GPU and multi-node training introduces additional non-determinism from communication ordering, gradient synchronization timing, and per-device random states.

6. **Third-party dependencies.** Your code might depend on a web API, a database query, or a random data augmentation library, each introducing uncontrolled variation.

---

## 5.2 Conceptual Foundations

### Levels of Reproducibility

It is useful to distinguish three levels:

**Exact reproducibility (bit-for-bit).** The same inputs produce the same outputs down to the last decimal place. This is achievable for CPU-only computation with deterministic libraries and fixed seeds. It is extremely difficult (and sometimes impossible) for GPU computation.

**Statistical reproducibility.** The same inputs produce results within a tight confidence interval. Running the experiment 5 times with different seeds yields metrics that vary by less than 0.5%. This is the practical standard for most ML work.

**Methodological reproducibility.** A different team, given the paper/documentation, can implement the method from scratch and achieve results within the reported range. This is the weakest form but the one that matters most for scientific claims.

### Environment Specification

The environment is the complete software and hardware context in which code runs. Two tools dominate environment specification in ML:

**Docker.** A Docker image packages the entire software environment — OS, system libraries, Python version, pip packages, CUDA toolkit — into a single artifact. A Dockerfile specifies how to build this image:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Python dependencies (pinned versions)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ /app/src/
WORKDIR /app

ENTRYPOINT ["python3", "src/train.py"]
```

The resulting image has a digest (SHA-256 hash) that uniquely identifies it. Two machines running the same image digest are guaranteed to have the same software environment.

**Advantages of Docker:**
- Complete isolation from the host system.
- Immutable once built (images are layered and content-addressed).
- Portable across machines and cloud providers.
- The image digest provides an unambiguous version identifier.

**Limitations of Docker:**
- Does not capture hardware (GPU model, number of GPUs, CPU architecture).
- Does not capture kernel version or driver version (these are provided by the host).
- Images can be large (5-20 GB for ML environments with CUDA).
- Building images is slow if not cached properly.

**Conda.** An environment manager that specifies Python version, pip packages, conda packages, and some system libraries in a YAML file:

```yaml
name: ml-training
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - pytorch=2.1.0
  - torchvision=0.16.0
  - cudatoolkit=11.8
  - numpy=1.24.3
  - pandas=2.0.3
  - scikit-learn=1.3.0
  - pip:
    - mlflow==2.7.1
    - transformers==4.33.2
```

**Advantages of Conda:**
- Lighter than Docker (no full OS image).
- Good for development workflows (activate/deactivate environments quickly).
- Handles non-Python dependencies (e.g., CUDA toolkit) via channels.

**Limitations of Conda:**
- Does not capture OS, system libraries, or kernel.
- "Works on my machine" can still happen if two machines have different OS versions.
- Dependency resolution can be slow and occasionally produces inconsistent environments.
- `conda env export` captures the full environment, but the output is platform-specific (packages available on Linux may not be available on macOS).

**Best practice:** Use Conda for development, Docker for training and deployment. The Docker image can include a Conda environment, combining the benefits of both.

### Seed Management

Random seeds control all sources of randomness in an ML pipeline:

```python
import random
import numpy as np
import torch

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic GPU operations (may slow down training):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**What seeds control:**
- Weight initialization (random values for model parameters).
- Data shuffling (order of training examples per epoch).
- Data augmentation (random crops, flips, rotations).
- Dropout (which neurons are dropped during training).
- Stochastic components of optimization (e.g., stochastic weight averaging).

**What seeds do not control:**
- Non-deterministic CUDA operations (atomicAdd, cuDNN autotuning).
- Multi-threaded data loading order (depends on OS thread scheduling).
- Distributed training communication order.

**PyTorch deterministic mode:**

```python
torch.use_deterministic_algorithms(True)
```

This forces PyTorch to use deterministic implementations for all operations. Some operations do not have deterministic implementations and will raise an error. This mode may slow down training significantly (10-30% in some cases) because deterministic implementations are often less optimized than their non-deterministic counterparts.

### The Reproducibility Checklist

A comprehensive reproducibility specification includes:

| Component | What to Record | Tool |
|-----------|---------------|------|
| Code | Git commit SHA | Git |
| Data | Content hash or version ID | DVC, Delta Lake |
| Environment (software) | Docker image digest or conda lockfile | Docker, Conda |
| Environment (hardware) | GPU model, count, driver version | Logged in experiment tracker |
| Hyperparameters | All values, including defaults | Experiment tracker |
| Random seeds | All seed values | Experiment tracker |
| Library determinism settings | cudnn.deterministic, benchmark, use_deterministic_algorithms | Logged in code |
| Training procedure | Number of epochs, early stopping criteria, checkpoint strategy | Experiment tracker |
| Data ordering | Sampler seed, worker seeds | Logged in code |

---

## 5.3 Mathematical Formulation

### Reproducibility as a Distance Metric

Given two runs of the same experiment with the same specification but different random states, let $m_1$ and $m_2$ be the resulting metric vectors. We define the reproducibility gap:

$$
\Delta(m_1, m_2) = \|m_1 - m_2\|
$$

For a single scalar metric (e.g., accuracy), this is simply:

$$
\Delta = |m_1 - m_2|
$$

Over $N$ repeated runs with the same configuration, the reproducibility variance is:

$$
\sigma^2_{\text{repro}} = \frac{1}{N-1} \sum_{i=1}^{N} (m_i - \bar{m})^2 \quad \text{where } \bar{m} = \frac{1}{N}\sum_{i=1}^{N} m_i
$$

This variance quantifies how much a result varies due to uncontrolled randomness. A key principle: **you should not claim that method A outperforms method B unless the performance difference exceeds the reproducibility variance:**

$$
|\bar{m}_A - \bar{m}_B| \gg \sigma_{\text{repro}}
$$

More precisely, use a statistical test (e.g., paired t-test, Wilcoxon signed-rank) to determine whether the difference is statistically significant given the observed variance.

### Deterministic Computation

A computation $f$ is deterministic if:

$$
\forall \text{ inputs } x: \quad f(x) = f(x)
$$

This is trivially true for pure mathematical functions. In practice, a computation is non-deterministic when:

1. **Floating-point non-associativity.** $(a + b) + c \neq a + (b + c)$ in floating-point arithmetic. When operations are parallelized, the order of additions can vary between runs.

   Example: summing $[1.0, 10^{-16}, -1.0]$ in order gives $(1.0 + 10^{-16}) - 1.0 = 0$ (because $1.0 + 10^{-16}$ rounds to $1.0$ in float64). Summing in a different order: $(10^{-16} + (-1.0)) + 1.0 = 10^{-16}$.

2. **Thread scheduling.** Multi-threaded data loaders, GPU kernel scheduling, and distributed communication are controlled by the OS and hardware, not by the random seed.

3. **External state.** Reading from a database, calling an API, or using the current time introduces state that is not controlled by the experiment specification.

### The Cost of Determinism

Enforcing determinism often comes at a performance cost. Let $T_{\text{det}}$ be the training time with deterministic mode and $T_{\text{nondet}}$ be the training time without. The relative overhead is:

$$
\text{overhead} = \frac{T_{\text{det}} - T_{\text{nondet}}}{T_{\text{nondet}}} \times 100\%
$$

In practice, this overhead ranges from 5% to 50% depending on the model architecture and operations used. For example, scatter/gather operations in GNNs have particularly expensive deterministic implementations.

The decision is a trade-off: **is the reproducibility gain worth the training time cost?** For debugging, yes — you need deterministic behavior to isolate the effect of a change. For large-scale production training, statistical reproducibility (multiple seeds, confidence intervals) is often more practical.

---

## 5.4 Worked Examples

### Example: Making a PyTorch Training Pipeline Reproducible

**Step 1: Set all random seeds.**

```python
import os
import random
import numpy as np
import torch

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
```

**Step 2: Configure PyTorch for determinism.**

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Some operations need this environment variable for deterministic behavior:
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

**Step 3: Make data loading deterministic.**

```python
def seed_worker(worker_id):
    """Ensure each DataLoader worker has a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=generator
)
```

**Step 4: Pin the environment.**

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# requirements.txt:
# torch==2.1.0+cu118
# torchvision==0.16.0+cu118
# numpy==1.24.3
# scikit-learn==1.3.0
# mlflow==2.7.1
```

Build and tag with a digest:

```bash
docker build -t training-env:v1 .
# Record the digest:
docker inspect --format='{{.Id}}' training-env:v1
# sha256:a1b2c3d4e5f6g7h8...
```

**Step 5: Run and verify reproducibility.**

```bash
# Run 1
docker run --gpus all training-env:v1 python train.py --seed 42
# Output: test_accuracy=0.8912, test_f1=0.8867

# Run 2 (exact same command)
docker run --gpus all training-env:v1 python train.py --seed 42
# Output: test_accuracy=0.8912, test_f1=0.8867
```

If the outputs match, the pipeline is deterministically reproducible. If they do not match (common when using non-deterministic CUDA operations or specific layer types), enable PyTorch's deterministic mode and check for operations that lack deterministic implementations.

**Step 6: Measure reproducibility variance across seeds.**

```bash
for seed in 42 43 44 45 46; do
    docker run --gpus all training-env:v1 python train.py --seed $seed
done
```

Results:
| Seed | Test Accuracy | Test F1 |
|------|--------------|---------|
| 42 | 0.8912 | 0.8867 |
| 43 | 0.8934 | 0.8891 |
| 44 | 0.8889 | 0.8845 |
| 45 | 0.8921 | 0.8878 |
| 46 | 0.8907 | 0.8860 |

Mean accuracy: $0.8913 \pm 0.0016$ (std). This tells you that any claimed improvement over this baseline must exceed ~0.5% (roughly $2\sigma$) to be convincingly better than noise.

---

## 5.5 Relevance to Machine Learning Practice

### Where Reproducibility Matters in Practice

**Model validation.** Before deploying a model, regulators or internal governance may require that the model can be retrained from its specification and produce consistent results.

**Debugging production issues.** A model in production produces unexpected predictions. To debug, you recreate the exact model. Without reproducibility, "recreate the exact model" is impossible.

**Benchmarking.** Comparing your method to a baseline requires that the baseline result is reproducible. If the baseline varies by 2% between runs and your improvement is 1%, the comparison is meaningless.

**Continuous training.** Some systems retrain models on a schedule (daily, weekly). Reproducibility ensures that performance changes are due to new data, not random variation.

### Practical Strategies

**Strategy 1: Deterministic mode for debugging, statistical mode for evaluation.**

During development and debugging, enable full deterministic mode so that changes produce predictable effects. For final evaluation and comparison, run multiple seeds and report confidence intervals.

**Strategy 2: Docker for everything, Conda for nothing.**

Using Docker for both development and production eliminates "works on my machine" issues entirely. The overhead of Docker is small compared to the time wasted debugging environment differences.

**Strategy 3: Pin everything, update deliberately.**

Pin all library versions in requirements files. Update them explicitly, not implicitly (e.g., `pip install torch` installs the latest, which might break things). When updating, record the change and test for reproducibility impact.

**Strategy 4: Reproducibility as a CI check.**

Add a CI test that trains a small model with a fixed seed and checks that the output metrics match a recorded baseline. If they do not, the commit is flagged. This catches environment changes, code bugs, and non-determinism early.

### Trade-offs

| Dimension | Full Determinism | Statistical Reproducibility |
|-----------|-----------------|---------------------------|
| Training speed | Slower (5-50% overhead) | Full speed |
| Result consistency | Bit-for-bit identical | Within confidence interval |
| Debugging ease | High (effects are deterministic) | Lower (need to average over seeds) |
| Hardware flexibility | Low (different GPUs may not match) | High (variance is measured, not eliminated) |
| Practical achievability | Difficult for large models | Always achievable |

---

## 5.6 Common Pitfalls & Misconceptions

**Pitfall: Setting torch.manual_seed but not np.random.seed or random.seed.** Data augmentation libraries often use NumPy or Python's random module. If only PyTorch's seed is set, augmentation is non-deterministic.

**Pitfall: Forgetting data loader worker seeds.** Each DataLoader worker has its own random state. Without `worker_init_fn`, workers use uncontrolled seeds, making data loading non-deterministic even with `shuffle=True` and a fixed generator.

**Pitfall: Using `cudnn.benchmark = True` and expecting reproducibility.** When `benchmark` is enabled, cuDNN benchmarks multiple implementations of each operation and selects the fastest. The selection may differ between runs, producing different results. Disable benchmarking for reproducibility.

**Pitfall: Pinning requirements.txt but not system libraries.** Your Python packages are pinned, but the Docker base image uses `apt-get install` without version pinning. A base image update pulls a new version of a system library, changing behavior.

**Pitfall: Ignoring the difference between pip freeze and pip install.** `pip freeze` records what is installed. It may include packages that were installed as dependencies of dependencies, with loose version constraints. Using `pip freeze` output as a requirements file is fragile. Instead, maintain a curated requirements file with your direct dependencies pinned, and use `pip freeze` only for auditing.

**Misconception: "Same code + same data = same model."** This ignores the environment (library versions, hardware, CUDA version) and randomness (seeds, non-deterministic operations). Same code + same data + same environment + same seed + deterministic mode = same model (usually).

**Misconception: "Reproducibility is only about random seeds."** Seeds are one of many factors. Environment, data ordering, non-deterministic operations, and hardware all affect reproducibility. Seeds are necessary but not sufficient.

**Misconception: "If I can't get bit-for-bit reproducibility, there's no point."** Statistical reproducibility (reporting mean ± std over multiple seeds) is scientifically rigorous and practically achievable. It is the standard in ML research and is sufficient for most industrial applications.

**Pitfall: Not recording negative results.** A reproducible pipeline that produces bad results is still valuable — it tells you what does not work. Delete the model weights to save storage, but keep the experiment record.

---

# 6. Interview Preparation

## Foundational Questions

### Q1: What is a model registry, and how does it differ from simply saving model files to disk?

**Answer:** A model registry is a centralized catalog that stores model artifacts (weights, configs, preprocessing objects) along with rich metadata: who trained the model, when, on what data, with what hyperparameters, and what its evaluation metrics are. It also manages model lifecycle through stages or aliases (staging, production, archived).

Simply saving files to disk provides no metadata, no lineage, no lifecycle management, and no access control. If you save `model.pkl` to disk, you have no record of how it was produced, no mechanism to compare it to alternatives, and no way to atomically promote or roll back. A registry adds structure, discoverability, and governance around model artifacts. It transforms a model from an anonymous file into a traceable, governable asset.

---

### Q2: Explain the concept of lineage tracking. Why is it necessary?

**Answer:** Lineage tracking records the complete provenance of a model: which dataset it was trained on, which code version produced it, which environment it was trained in, and which hyperparameters were used. It is modeled as a directed acyclic graph where nodes are versioned artifacts and edges represent transformations.

Lineage is necessary for four reasons. First, reproducibility — without knowing all inputs, you cannot recreate the model. Second, debugging — if a model misbehaves, lineage lets you trace back to a potential root cause (e.g., a corrupted data version). Third, compliance — regulators require audit trails that connect predictions to the data and code that produced them. Fourth, impact analysis — when a dataset is found to contain an error, lineage lets you identify every model that was trained on that dataset.

---

### Q3: What is DVC and how does it work?

**Answer:** DVC (Data Version Control) extends Git to handle large data files. Git tracks small pointer files (`.dvc` files), each containing a content hash of the corresponding data file. The actual data is stored in a content-addressable cache and can be pushed to remote storage (S3, GCS, etc.).

When you run `dvc add data.csv`, DVC computes the MD5 hash of the file, moves it to the cache, and creates `data.csv.dvc` containing the hash. You commit `data.csv.dvc` to Git. To retrieve a past data version, check out the historical Git commit (which restores the old `.dvc` file) and run `dvc checkout`, which restores the corresponding data from the cache.

DVC also supports pipelines (`dvc.yaml`), which define multi-stage data processing workflows with dependency tracking. `dvc repro` only re-runs stages whose inputs have changed, providing both versioning and incremental computation.

---

### Q4: What is the difference between parameters, metrics, artifacts, and metadata in experiment tracking?

**Answer:** Parameters are inputs you choose before the run starts — hyperparameters like learning rate and batch size, architecture choices, data configuration. They define the experiment.

Metrics are outputs you measure — loss, accuracy, F1, latency. They describe the outcome and are typically logged as time series during training and as final values after evaluation.

Artifacts are files produced during the run — model checkpoints, evaluation plots, prediction samples. They provide detailed evidence beyond scalar metrics.

Metadata is contextual information about the run itself — who launched it, when, how long it took, what hardware it used, what Git commit was active. It is typically recorded automatically.

The distinction matters for how each is used: parameters are used for filtering and grouping runs, metrics for ranking and comparison, artifacts for inspection and debugging, and metadata for audit and infrastructure planning.

---

### Q5: What are the different levels of reproducibility, and which is typically achievable in practice?

**Answer:** Three levels exist. Exact (bit-for-bit) reproducibility means identical inputs always produce identical outputs, down to the last floating-point digit. This is achievable on CPU with deterministic libraries and fixed seeds, but extremely difficult on GPU due to non-deterministic CUDA operations.

Statistical reproducibility means repeated runs with different seeds produce results within a tight confidence interval (e.g., accuracy varies by less than 0.5%). This is the practical standard for most ML work. You report mean ± standard deviation over multiple seeds.

Methodological reproducibility means a different team, given the documentation, can implement the method independently and achieve results in the reported range. This is the weakest form but the one that matters most for scientific claims.

In practice, statistical reproducibility is the target. It requires fixed environments (Docker), controlled randomness (seeds), and multiple-seed evaluation. Exact reproducibility requires deterministic mode, which incurs a 5-50% training time overhead and restricts which operations can be used.

---

### Q6: Explain the concept of point-in-time correctness in feature stores.

**Answer:** Point-in-time correctness means that when constructing training data, each example uses only feature values that were available at the time that example was generated — not future values.

Consider a fraud detection model. For a transaction at time $t$, the feature "average transaction amount over the last 30 days" should be computed using only transactions before time $t$. If you instead compute this feature using data up to the current date (which is later than $t$), you introduce label leakage — the model sees information that would not be available at prediction time.

Feature stores enforce point-in-time correctness by maintaining time-indexed feature values and performing "as-of" joins: for each training example with timestamp $t$, the join retrieves the most recent feature value known at or before $t$. Without this, models appear to perform much better during training (because they have access to future information) but perform worse in production.

---

## Mathematical Questions

### Q7: Explain content-addressable storage mathematically. What properties does the hash function need?

**Answer:** Content-addressable storage identifies objects by a cryptographic hash of their content: $\text{id}(A) = H(A)$, where $H$ is a hash function like SHA-256.

The hash function needs three properties. First, determinism: the same input always produces the same hash. Second, collision resistance: it is computationally infeasible to find two different inputs $A_1 \neq A_2$ such that $H(A_1) = H(A_2)$. For SHA-256, a collision requires approximately $2^{128}$ operations (birthday bound), which is beyond the capability of any known computing system. Third, preimage resistance: given $h$, it is infeasible to find $A$ such that $H(A) = h$. This prevents forging an artifact that matches a known hash.

Content addressing enables three things: deduplication (identical artifacts are stored once), integrity verification (recompute the hash and compare), and immutable versioning (the hash changes if the content changes, so the same hash always refers to the same content).

For datasets, DVC uses a Merkle hash — the hash of a directory is the hash of the hashes of its files: $H(D) = H(H(f_1) \| H(f_2) \| \ldots \| H(f_k))$. This means any change to any file changes the directory hash, providing change detection at any granularity.

---

### Q8: Formalize Delta Lake's time travel mechanism. How is a table at version $v$ reconstructed?

**Answer:** A Delta table's state at version $v$ is defined by the set of active Parquet files:

$$
T(v) = \left(\bigcup_{i=0}^{v} \text{add}(i)\right) \setminus \left(\bigcup_{i=0}^{v} \text{remove}(i)\right)
$$

Each transaction $i$ has a JSON log entry specifying which files were added ($\text{add}(i)$) and which were removed ($\text{remove}(i)$). To reconstruct version $v$, the engine replays transactions 0 through $v$, accumulating the set of active files, then reads those Parquet files.

For efficiency, Delta Lake periodically creates **checkpoint files** that snapshot the accumulated state at a version. To read version $v$, the engine finds the most recent checkpoint at or before $v$, loads it, then replays only the subsequent transactions. This reduces reconstruction time from $O(v)$ to $O(v - v_{\text{checkpoint}})$.

The difference between two versions can be computed as the symmetric difference:

$$
\Delta(v_1, v_2) = T(v_2) \setminus T(v_1) \cup T(v_1) \setminus T(v_2)
$$

This tells you exactly which data files were added or removed between versions, enabling efficient change detection without scanning the full dataset.

ACID semantics are enforced via optimistic concurrency control: each transaction attempts to write a new log entry. If another transaction wrote a conflicting entry in the meantime, the write fails and is retried.

---

### Q9: Why does floating-point non-associativity break bit-for-bit reproducibility in GPU training? Give a concrete example.

**Answer:** IEEE 754 floating-point arithmetic is not associative: $(a + b) + c \neq a + (b + c)$ in general, because each addition rounds its result to the nearest representable value.

Concrete example with float32 (approximately 7 decimal digits of precision):

Let $a = 1.0$, $b = 10^{-8}$, $c = -1.0$.

Computation 1 (left-to-right): $(1.0 + 10^{-8}) + (-1.0)$. The first addition: $1.0 + 10^{-8} = 1.0$ because $10^{-8}$ is below the precision of float32 relative to 1.0. Then $1.0 + (-1.0) = 0.0$. Result: $0.0$.

Computation 2 (right-to-left): $1.0 + (10^{-8} + (-1.0))$. The first addition: $10^{-8} + (-1.0) = -0.99999999$, which is representable. Then $1.0 + (-0.99999999) = 10^{-8}$. Result: $10^{-8}$.

In GPU computation, a reduction operation (e.g., summing a vector for a loss function or gradient) is parallelized across threads. Each run may partition the work differently, changing the order of additions, producing different rounding behavior, and thus different results. Over thousands of gradient updates, these differences accumulate, producing models with measurably different weights.

Setting `torch.use_deterministic_algorithms(True)` forces fixed-order reductions, eliminating this source of non-determinism at the cost of performance.

---

### Q10: How do you formally determine whether two models (trained with different seeds) have statistically different performance?

**Answer:** Use paired evaluation across multiple seeds. Train model A and model B each with $N$ seeds $s_1, \ldots, s_N$ (ideally $N \geq 5$, preferably $\geq 10$). For each seed, compute the metric difference:

$$
d_i = m_A(s_i) - m_B(s_i)
$$

Test the null hypothesis $H_0: \mathbb{E}[d] = 0$ using a paired t-test:

$$
t = \frac{\bar{d}}{s_d / \sqrt{N}}
$$

where $\bar{d} = \frac{1}{N}\sum d_i$ and $s_d = \sqrt{\frac{1}{N-1}\sum(d_i - \bar{d})^2}$.

Compare $t$ to the $t$-distribution with $N-1$ degrees of freedom. If $|t| > t_{\alpha/2, N-1}$ (typically $\alpha = 0.05$), reject $H_0$ and conclude the performance difference is statistically significant.

The paired design is important because it accounts for seed-to-seed correlation: if seed 42 happens to produce an easy train/val split, both models benefit, and the pairing cancels this out.

For non-parametric analysis (e.g., when $N$ is very small or distributions are non-Gaussian), use the Wilcoxon signed-rank test instead.

Practical note: many ML papers compare single-seed results. This is scientifically weak — a 0.5% accuracy difference could easily be within the reproducibility variance.

---

## Applied Questions

### Q11: You are designing an ML platform for a team of 20 data scientists who train and deploy models daily. How would you design the versioning infrastructure?

**Answer:** The system should integrate four versioning layers.

For code, I would use a Git monorepo (or a small number of repos) with trunk-based development. Each experiment is a branch. CI/CD runs on every merge to main, executing linting, unit tests, and a small-scale training smoke test.

For data, the choice depends on data type. For tabular data in a data warehouse, Delta Lake (or Iceberg) provides versioned, ACID-compliant tables with time travel. For unstructured data (images, text), DVC with S3-backed storage. A feature store (Feast or Tecton) manages shared features with point-in-time correctness.

For environments, I would standardize on Docker images built from pinned Dockerfiles stored in the code repo. A CI pipeline builds and tags images on Dockerfile changes. Images are stored in a container registry (ECR, GCR) with content-based digests.

For models, MLflow or a comparable registry serves as the central catalog. Every training run logs to the experiment tracker (MLflow Tracking, W&B), recording parameters, metrics, artifacts, and the reproducibility tuple (git SHA, data version, Docker digest, seed). Successful models are registered in the model registry with lineage links. Deployment reads from the registry, ensuring a single source of truth.

The key integration points: the experiment tracker links to the code version (Git SHA), the data version (DVC hash or Delta Lake version), and the environment version (Docker digest). The model registry links to the experiment run. This creates a complete lineage DAG from raw data through to deployed model.

For a team of 20, I would also add access control (who can promote to production), automated validation (canary deployment before full rollout), and a monitoring system that associates production metrics with model versions for continuous performance tracking.

---

### Q12: Your team discovers that a model deployed 3 months ago was trained on data containing PII that should have been excluded. Walk through how a well-versioned system would help you respond.

**Answer:** Start with the model registry. Identify the model version that was in production 3 months ago. The registry gives you the exact model artifact and the run ID that produced it.

From the run ID, query the experiment tracker. It records the data version used for training (e.g., DVC hash `abc123` or Delta Lake version 47).

Retrieve that exact data version. With DVC: `git checkout <commit>; dvc checkout`. With Delta Lake: `SELECT * FROM training_data VERSION AS OF 47`. Examine the data to identify which rows contain PII and which model inputs were affected.

Assess the impact. Was the PII used as a feature (meaning the model learned from it), or was it in a field that was not used as a feature (meaning it was present in the dataset but did not influence the model)? If it was used as a feature, the model may need to be retrained without that feature.

For remediation: create a new data version with PII removed or masked. Retrain the model on the clean data (using the same code commit and environment, from the reproducibility tuple). Register the retrained model as a new version and promote it to production. Archive the contaminated version.

For audit: the lineage trail provides evidence of what happened, when it was discovered, and what corrective action was taken. Document the incident, linking to specific version IDs at each step.

Without versioning, this scenario is a nightmare: you cannot identify which data was used, you cannot reproduce the model, and you cannot demonstrate to regulators that the issue has been resolved.

---

### Q13: How would you implement a canary deployment with model versioning?

**Answer:** A canary deployment routes a small fraction of production traffic to the new model while the majority continues to be served by the current model.

Step 1: The model registry has two relevant versions — the current production model (v5, alias "production") and the candidate (v6, alias "canary"). Both are retrievable from the registry.

Step 2: The serving infrastructure loads both models and routes traffic according to a configurable split (e.g., 5% to v6, 95% to v5). Each prediction is tagged with the model version that produced it.

Step 3: Production monitoring computes metrics (accuracy, latency, error rate) per model version. The experiment tracker or monitoring system stores these metrics associated with the version.

Step 4: After a sufficient observation period (hours to days depending on traffic volume), compare metrics. If v6 meets or exceeds v5 on all key metrics, gradually increase its traffic share (5% → 25% → 50% → 100%). If v6 underperforms, remove it from the canary and keep v5 at 100%.

Step 5: Once v6 reaches 100% traffic, update the registry: assign "production" alias to v6. Archive v5 (but do not delete it — keep it available for rapid rollback).

The model registry is critical here because it provides the mechanism for the serving layer to know which models to load, and it provides the aliasing system that makes promotion and rollback a metadata change rather than a deployment.

---

### Q14: Your feature store has a feature "user_avg_purchase_last_30d." The computation logic is changed from mean to median. How do you handle this versioning-wise?

**Answer:** This is a feature definition version change. The correct approach is:

Create a new version of the feature definition, not overwrite the old one. The old definition (mean) is now `user_avg_purchase_last_30d_v1` (or version 1 in the feature store's version system). The new definition (median) is `user_avg_purchase_last_30d_v2`.

Backfill the new feature. Compute `v2` values for historical timestamps so that training data can be constructed using either version.

Update downstream consumers explicitly. Models that were trained on `v1` continue to use `v1` for serving. New models are trained on `v2`. This prevents a silent change from breaking existing models.

If the feature store does not support definition versioning natively, the practical approach is to create a new feature with a new name (e.g., `user_median_purchase_last_30d`) and treat the old feature as deprecated.

The critical mistake to avoid: changing the computation in place and recomputing historical values. This rewrites history — models that were trained on the old values cannot be reproduced, and any analysis that compared old vs. new data is invalidated.

---

## Debugging & Failure-Mode Questions

### Q15: A model was retrained on identical data, code, and hyperparameters, but metrics differ by 2%. What are the possible causes?

**Answer:** Investigate in order of likelihood:

1. **Random seed mismatch.** Was the same seed used? Check all seed sources: PyTorch, NumPy, Python `random`, CUDA, data loader workers. A common failure: the training script sets `torch.manual_seed` but not `np.random.seed`, and data augmentation uses NumPy.

2. **Data loader non-determinism.** If `num_workers > 0`, each worker has its own random state. Without a `worker_init_fn` that seeds each worker deterministically, the order in which batches are constructed varies between runs.

3. **cuDNN non-determinism.** If `cudnn.benchmark = True`, cuDNN selects potentially different algorithms between runs. If `cudnn.deterministic = False`, some cuDNN operations use non-deterministic implementations.

4. **CUDA non-deterministic operations.** Some PyTorch operations (e.g., `scatter_add`, `index_add`, certain loss functions) use `atomicAdd` on GPU, which is non-deterministic. Enable `torch.use_deterministic_algorithms(True)` to detect these.

5. **Environment differences.** Even with "identical" code, a different library version (e.g., PyTorch patch release) can change numerical behavior. Verify the Docker image digest is identical, not just the tag.

6. **Hardware differences.** Different GPU models use different CUDA kernels. A V100 and an A100 will produce different results for the same code and seed.

7. **Floating-point non-associativity.** Even with deterministic mode, different memory layouts (due to different batch sizes or input shapes) can change the order of floating-point operations within a single operation.

A 2% difference is large — it almost certainly indicates one of the first four causes rather than the last three (which typically produce < 0.1% differences).

---

### Q16: A colleague says "I can't reproduce my result from last week. I'm using the same code and the same data." What questions would you ask?

**Answer:** Walk through the reproducibility tuple systematically:

"What do you mean by 'same code'? Is it the same Git commit, or did you make changes and revert them? Check `git log` and `git diff`."

"What do you mean by 'same data'? Is it the same file (same content hash), or the same filename? Did the upstream data source change? Check the DVC hash or verify the file checksum."

"What environment are you running in? Same machine? Same Docker image? Did you update any packages since last week? Run `pip freeze` and compare."

"What seed did you use last week? Did you record it? Was it set for all random sources (torch, numpy, random, CUDA)?"

"Were any configuration values hardcoded in the script that might have been changed and reverted? Check the config file hash."

"Was the data loaded in the same order? Did the data loader use the same number of workers?"

"Are you comparing the same metric at the same point in training? Perhaps last week you evaluated at epoch 50 and this week at epoch 48 due to early stopping with a slightly different patience threshold."

"Did you run on the same GPU? Check `nvidia-smi` for the GPU model and driver version."

Nine times out of ten, the problem is one of: uncommitted code changes, an environment difference, or a missing seed.

---

### Q17: Your Delta Lake table has accumulated 500 versions over 6 months, and query performance is degrading. What is happening and how do you fix it?

**Answer:** Delta Lake query performance degrades when there are too many small Parquet files (the "small file problem") and when the transaction log becomes very long.

Each transaction can add new files. Frequent small writes (e.g., streaming inserts of individual records) produce thousands of tiny Parquet files. Reading the table requires opening each file, which has per-file overhead from metadata parsing and I/O setup. This makes queries slow.

The transaction log itself also grows. Without checkpoints, reading the table at version 500 requires replaying all 500 log entries. Delta Lake creates checkpoints periodically (by default every 10 transactions), but with 500 versions, there are still many entries to process.

Solutions:

Run `OPTIMIZE` (also called compaction or bin-packing). This rewrites small Parquet files into larger ones, typically targeting 1 GB per file. This is a new transaction that removes the small files and adds large ones.

Run `VACUUM` to delete old Parquet files that are no longer referenced by any recent version. By default, `VACUUM` keeps files needed for versions within the last 7 days. After vacuuming, time travel to versions older than the retention period is no longer possible.

Configure more frequent checkpoints if needed.

Consider Z-ordering (`OPTIMIZE ... ZORDER BY (column)`) if queries filter on specific columns, which co-locates related data for better predicate pushdown.

Ongoing prevention: batch writes (insert data in larger batches rather than individual records) and schedule regular `OPTIMIZE` and `VACUUM` jobs.

---

### Q18: An experiment tracker shows that the best run had a validation accuracy of 99.8% on a binary classification task where the baseline is 85%. What should you suspect?

**Answer:** Suspect data leakage. A jump from 85% to 99.8% is almost never legitimate in a real binary classification problem.

Common causes:

**Label leakage.** A feature directly encodes the label. For example, a feature "account_status" that was set to "flagged" after the fraud was detected — this feature is only set because the label is positive, so the model trivially learns to predict from it.

**Temporal leakage.** Features computed using future data relative to the label timestamp. The point-in-time correctness violation described in Section 2. For example, using a user's total transaction count over the next 30 days as a feature to predict whether a transaction on day 0 is fraudulent.

**Train/test contamination.** Duplicate rows across train and test sets. If the same example (or nearly identical examples, e.g., from the same user within minutes) appears in both, the model memorizes rather than generalizes.

**Target encoding leakage.** A feature that was computed from the training labels (e.g., the mean label value for each categorical level) was applied to the test set using the training labels, effectively leaking the target.

**Data preprocessing leakage.** Fitting a scaler or imputer on the combined train+test data before splitting, allowing test set statistics to influence training set normalization.

Diagnostic steps: (1) Examine feature importances — if one feature dominates, inspect it for leakage. (2) Check for train/test overlap — compute the intersection of train and test identifiers. (3) Train a model with only the top feature and see if it achieves near-perfect accuracy. (4) Remove the suspect feature and retrain — if accuracy drops to baseline, the feature was leaking.

---

## Follow-Up & Probing Questions

### Q19: "You mentioned content hashing for artifact identification. What happens if two different models have the same hash?"

**Answer:** A hash collision means $H(A_1) = H(A_2)$ for $A_1 \neq A_2$. For SHA-256, the probability of a collision among $n$ artifacts is approximately $\frac{n^2}{2^{257}}$ (birthday bound). With $10^{18}$ artifacts (far more than any realistic system), the probability is approximately $10^{-41}$ — effectively impossible.

In practice, hash collisions are never the problem. When two artifacts produce the same hash, they are the same artifact. This is a feature, not a bug — it enables deduplication.

If you are concerned about adversarial collision attacks (someone deliberately crafting a model file to match a known hash), SHA-256 is resistant to this. For SHA-1, collision attacks exist (SHAttered, 2017), which is one reason newer systems prefer SHA-256.

The practical failure mode is not collision but truncation: some systems use truncated hashes (e.g., first 12 hex characters of MD5) for readability. Truncation dramatically reduces collision resistance. Use full hashes internally and truncated hashes only for display.

---

### Q20: "You described Docker for environment reproducibility. What are its limitations for GPU-intensive ML workloads?"

**Answer:** Docker provides excellent software environment isolation but has important limitations for ML:

First, the GPU driver and CUDA runtime are partially outside the container. The container uses the host's GPU driver (mounted into the container by the NVIDIA Container Toolkit). If the host driver version differs between two machines, the container behaves differently despite having the same image. The CUDA toolkit inside the container must be compatible with the host's driver version.

Second, different GPU hardware produces different results. An A100 and a V100 use different CUDA cores, different tensor cores, and different numerical precision characteristics. The same code with the same seed can produce different results on different GPU models. Docker does not abstract the hardware.

Third, multi-GPU configurations vary. A machine with 4 GPUs connected via NVLink behaves differently from one with 4 GPUs connected via PCIe. Communication patterns, gradient synchronization, and memory layout all depend on the GPU topology.

Fourth, image size is a practical issue. ML Docker images with CUDA, cuDNN, PyTorch, and common libraries typically weigh 5-15 GB. Pulling and building these images is slow, especially on CI systems.

Mitigation: record the GPU model, driver version, and topology alongside the Docker image digest. Use multi-stage builds to reduce image size. Pin the NVIDIA base image to a specific digest. For cross-hardware reproducibility, accept statistical reproducibility over exact reproducibility.

---

### Q21: "How does feature store versioning interact with model versioning?"

**Answer:** The interaction is critical for end-to-end lineage. When a model is trained, its lineage must include not just the dataset version but the feature definition versions used.

Specifically: a model trained at time $t$ uses feature definition $f_v$ (version $v$ of feature $f$). If the feature definition is later updated to version $v+1$ (e.g., the computation logic changes), the model still expects inputs computed with version $v$.

The serving infrastructure must ensure that the model receives features computed with the same version it was trained on. This means either:

1. The feature store serves features using the definition version specified in the model's metadata. When you query a feature for inference, you specify the definition version.

2. The model's serving wrapper includes the feature computation logic from training time, bypassing the feature store's current definition. This is common when the feature store does not support definition versioning.

If this alignment is broken — the model expects $v$ but receives $v+1$ — you get training-serving skew. The symptoms are subtle: the model does not crash, it just makes slightly worse predictions. This skew is one of the most insidious bugs in production ML systems because it is invisible without monitoring and lineage tracking.

---

### Q22: "Your team is debating whether to use DVC or Delta Lake for data versioning. What factors determine the right choice?"

**Answer:** The choice depends on four factors:

**Data type.** DVC handles any file type (images, audio, text, binary). Delta Lake handles tabular data stored as Parquet. If your data is primarily images or unstructured files, DVC is the clear choice. If it is primarily tabular data in a data warehouse, Delta Lake is the clear choice. For mixed workloads, you may need both.

**Scale.** DVC versions entire files. If you change one row in a 100 GB CSV, DVC stores the entire 100 GB again. Delta Lake versions at the file level within a table — changing one row replaces only the Parquet file containing that row (typically 128 MB). For large tables with frequent small updates, Delta Lake is far more storage-efficient.

**Query patterns.** Delta Lake supports SQL queries over historical versions: `SELECT * FROM table VERSION AS OF 5`. DVC restores the entire file to disk, requiring you to load and query it yourself. If you need ad-hoc SQL queries over past data versions, Delta Lake is superior.

**Ecosystem.** DVC is ecosystem-agnostic — it works with any programming language and any storage backend. Delta Lake is tied to Spark and Parquet (though Rust-based readers are expanding compatibility). If your stack is already Spark-based, Delta Lake fits naturally. If not, the migration cost is significant.

**Pipeline integration.** DVC has built-in pipeline support (define stages, track dependencies, reproduce selectively). Delta Lake is a storage layer and does not manage computational pipelines — you need a separate orchestrator (Airflow, Dagster, Prefect).

A common setup: DVC for unstructured data and pipeline definition, Delta Lake (or Iceberg) for structured tabular data, and a feature store for shared features.

---

### Q23: "How would you test that your ML pipeline is reproducible?"

**Answer:** Implement a three-level testing strategy:

**Level 1: Determinism test.** Run the pipeline twice with the same seed on the same machine and verify bit-for-bit identical outputs:

```python
def test_determinism():
    result_1 = train(seed=42, data="test_data_v1", epochs=2)
    result_2 = train(seed=42, data="test_data_v1", epochs=2)
    assert result_1["test_accuracy"] == result_2["test_accuracy"]
    assert torch.allclose(result_1["model_weights"], result_2["model_weights"])
```

If this fails, there is a non-determinism source. Use `torch.use_deterministic_algorithms(True)` to identify the operation.

**Level 2: Statistical reproducibility test.** Run with multiple seeds and verify that variance is within acceptable bounds:

```python
def test_statistical_reproducibility():
    accuracies = [train(seed=s, data="test_data_v1", epochs=5)["test_accuracy"]
                  for s in range(5)]
    assert np.std(accuracies) < 0.01  # Variance below 1%
```

**Level 3: Cross-environment test.** Run on two different machines (or Docker images) and verify results are within statistical bounds:

```python
def test_cross_environment():
    result_machine_a = run_on_machine("a", seed=42, data="test_data_v1")
    result_machine_b = run_on_machine("b", seed=42, data="test_data_v1")
    assert abs(result_machine_a - result_machine_b) < 0.02  # Within 2%
```

Integrate the determinism test into CI (run on every commit). Run the statistical test weekly. Run the cross-environment test when upgrading infrastructure.

---

### Q24: "A production model's performance degrades gradually over 3 months. Versioning is in place but nobody noticed until now. How do you diagnose?"

**Answer:** This is likely data drift or concept drift. The model has not changed (same version in the registry), but the data it receives has shifted.

Step 1: Retrieve the model version from the registry. Confirm it has not been changed (same artifact hash).

Step 2: Compare the distribution of incoming production data (the last 3 months) to the training data distribution. The data version in the model's lineage identifies the training data. Compute distribution statistics — means, variances, histograms — for each feature. Look for features whose distributions have shifted significantly (KL divergence, KS test).

Step 3: Check label distribution. If the class balance in production has shifted (e.g., fraud rate increased from 1% to 3%), the model's calibration may be off.

Step 4: Retrain with recent data. Use the same code version and environment (from the reproducibility tuple) but a more recent data version. Compare metrics on a held-out set drawn from recent data.

Step 5: Implement monitoring going forward. Set up alerts for distribution drift (feature and prediction distributions), performance degradation (monitor accuracy, AUC on ground-truth labels as they become available), and data quality issues (null rates, out-of-range values).

The versioning system's role here: it confirms the model itself did not change, focusing the investigation on data and environment factors. Without versioning, you would first need to determine whether someone accidentally replaced the model, which would waste time on a red herring.

---

### Q25: "What is the difference between model versioning and model checkpointing?"

**Answer:** Checkpointing is saving the model's state at regular intervals during training (e.g., every epoch, every 1000 steps). Its purpose is fault tolerance — if training crashes at epoch 40, you can resume from the epoch 39 checkpoint instead of starting over.

Model versioning is registering a finished model (or a selected checkpoint) in a model registry with metadata, lineage, and lifecycle management. Its purpose is governance — tracking which models exist, which is in production, and how each was produced.

Key differences:

Checkpoints are numerous and ephemeral. A single training run might produce 100 checkpoints, most of which are discarded after training completes. Only the best checkpoint (by validation metric) or the final checkpoint is typically retained.

Versions are curated and permanent. A registered model version represents a deliberate decision that this artifact is worth tracking. It has been evaluated and is a candidate for deployment.

Checkpoints have limited metadata (epoch number, step number, loss value). Versions have rich metadata (all hyperparameters, full evaluation metrics, lineage, stage, author).

You might use checkpointing within training but model versioning for the final product. The transition point is: "I've evaluated this checkpoint, I'm satisfied, and I'm registering it in the registry."

A common mistake: treating the checkpoint directory as a model registry. This provides no discoverability (how do you find the best model across all training runs?), no comparison (how do you rank models?), and no lifecycle management (which one is in production?).
