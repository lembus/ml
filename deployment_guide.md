# ML Model Deployment: A Comprehensive Reference Guide

---

## Part I — Serving Patterns

---

### 1. Motivation & Intuition

Training a model is only half the journey. A model sitting on a researcher's laptop produces zero business value. **Deployment** is the engineering discipline of making a trained model available to users, applications, or downstream systems so it can actually produce predictions on new data.

Consider a concrete scenario: you have trained a fraud-detection model that, given a credit-card transaction, outputs a probability of fraud. Three very different business contexts demand three very different serving architectures:

- **Batch serving.** Every night, the bank wants to scan the previous day's transactions and flag suspicious ones for human review the next morning. Latency does not matter — what matters is throughput and cost. You run a scheduled job that reads millions of rows from a data warehouse, scores each one, and writes the results back.

- **Real-time serving.** The bank wants to block fraudulent transactions *at the moment of purchase*. The model must respond within 50 ms. You wrap it in an API, deploy it behind a load balancer, and ensure it can handle thousands of concurrent requests.

- **Edge serving.** The bank deploys a lightweight version of the model directly onto the point-of-sale terminal hardware. There is no network call — the model runs on the device itself, even when connectivity is poor.

Each pattern optimizes for different constraints: latency, throughput, cost, connectivity, and hardware. Choosing the wrong serving pattern can make a perfectly accurate model useless in production.

**Why this matters for ML engineers specifically:**

1. Model accuracy on a test set is necessary but not sufficient. A model that cannot be served reliably, at the required latency, at the required scale, fails in practice.
2. The choice of serving pattern affects *which models* you can even consider. A 10-billion-parameter transformer is easy to serve in batch but may be impossible to serve at the edge.
3. Deployment choices create feedback loops — how you serve determines what telemetry you collect, which determines how you retrain, which determines what you serve next.

---

### 2. Conceptual Foundations

#### 2.1 Batch Serving

**Definition.** Batch serving computes predictions for a large collection of inputs at once, typically on a fixed schedule (hourly, daily, weekly) or triggered by an event (new data lands in the warehouse).

**Key components:**

- **Scheduler.** An orchestrator (Airflow, Cron, Prefect, Dagster) triggers the job at the right time.
- **Data source.** A data warehouse (BigQuery, Redshift, Snowflake), a data lake (S3 + Parquet/Delta Lake), or a database.
- **Compute engine.** A distributed processing framework (Spark, Dask, Ray) or a simple Python script for smaller datasets.
- **Model artifact.** The serialized model loaded into the compute engine.
- **Output sink.** Where predictions are written — a database table, a feature store, a file in object storage.

**How it works, step by step:**

1. The scheduler fires at the configured time.
2. The job reads input data (e.g., all transactions from the last 24 hours).
3. The model artifact is loaded into memory.
4. Inputs are transformed into the feature format the model expects.
5. The model scores each input (often in vectorized mini-batches for efficiency).
6. Predictions are written to the output sink.
7. Downstream consumers (dashboards, alerting systems, human reviewers) read the predictions.

**Map-Reduce pattern.** For very large datasets, the scoring job is distributed. In the *map* phase, data is partitioned across workers, each of which loads a copy of the model and scores its partition independently. In the *reduce* phase (if needed), results are aggregated, deduplicated, or joined with metadata. Spark's `mapPartitions` is the canonical implementation: the model is loaded once per partition (not once per row), amortizing model loading cost.

**Assumptions:**

- Predictions do not need to be available immediately — freshness requirements are measured in minutes to hours.
- The full input dataset is available before scoring begins.
- The compute budget is flexible (you can use spot instances, scale horizontally).

**What breaks when assumptions are violated:**

- If freshness requirements tighten (e.g., predictions needed within seconds), batch serving introduces unacceptable staleness.
- If input data arrives continuously rather than in discrete dumps, you either micro-batch (increasing orchestration complexity) or switch to streaming.
- If the dataset is too large to fit in a single job's memory/time window, you need to partition carefully or switch to a streaming architecture.

#### 2.2 Real-Time Serving

**Definition.** Real-time serving (also called online serving) computes predictions on demand, one request at a time (or in small batches), with latency targets typically between 10 ms and 500 ms.

**Key components:**

- **API server.** A web server (Flask, FastAPI, TorchServe, TensorFlow Serving, Triton Inference Server) that exposes an endpoint accepting input features and returning predictions.
- **Protocol.** REST (HTTP + JSON), gRPC (HTTP/2 + Protocol Buffers), or a streaming protocol.
- **Load balancer.** Distributes incoming requests across multiple server replicas (NGINX, AWS ALB, Envoy).
- **Model runtime.** The framework that executes inference (PyTorch, TensorFlow, ONNX Runtime).
- **Feature store (optional).** A low-latency key-value store (Redis, DynamoDB, Feast) that serves precomputed features so the API does not have to compute them on the fly.

**REST vs. gRPC vs. Streaming:**

| Property | REST (HTTP/JSON) | gRPC (HTTP/2 + Protobuf) | Streaming (Kafka/Kinesis) |
|---|---|---|---|
| Serialization | JSON (text) | Protocol Buffers (binary) | Avro, Protobuf, JSON |
| Latency overhead | Higher (text parsing) | Lower (binary, multiplexed) | Variable (consumer lag) |
| Ease of debugging | High (human-readable) | Medium (binary payloads) | Lower (async, distributed) |
| Bidirectional | No (request-response) | Yes (streaming RPCs) | Yes (pub-sub) |
| Browser support | Native | Requires grpc-web proxy | Requires WebSocket bridge |
| Use case | General-purpose APIs | Low-latency, internal services | Event-driven architectures |

**REST** is the default choice for external-facing APIs due to ubiquitous tooling and human-readable payloads. **gRPC** is preferred for internal microservice communication where latency matters and both client and server are controlled. **Streaming** (Kafka, Kinesis) is used when the prediction is part of an event-driven pipeline — e.g., every time a user clicks on a product, an event is published to a Kafka topic, a consumer reads it, scores it through the model, and publishes the prediction to another topic for downstream consumers.

**How a real-time request flows:**

1. A client sends a request containing raw input (or a key to look up precomputed features).
2. The load balancer routes the request to one of N replicas.
3. The server deserializes the input.
4. If needed, features are retrieved from the feature store and combined with the input.
5. The input tensor is passed to the model runtime for inference.
6. The prediction is serialized and returned to the client.

**Assumptions:**

- Each request is independent (no cross-request state needed for inference).
- Latency is a hard constraint — there is an SLA (e.g., P99 < 100 ms).
- The model is small enough to fit in a single server's memory (or is partitioned across servers — see model partitioning below).

**What breaks when assumptions are violated:**

- If requests are *not* independent (e.g., you need the previous 100 predictions for context), you need a stateful serving layer, which is significantly more complex.
- If latency requirements cannot be met, you must optimize the model (quantization, distillation, pruning), upgrade hardware (GPUs, specialized accelerators), or relax the SLA.
- If the model does not fit in memory, you must shard it across multiple servers, introducing network latency for inter-shard communication.

#### 2.3 Edge Serving

**Definition.** Edge serving runs the model directly on an end-user device (phone, IoT sensor, browser, embedded system) rather than on a centralized server.

**Why edge:**

- **Latency.** No network round-trip. Inference is limited only by on-device compute.
- **Privacy.** Data never leaves the device. Critical for healthcare (on-device diagnostics), finance (on-device fraud checks), and jurisdictions with strict data-residency laws.
- **Connectivity.** Works offline — essential for agriculture drones, industrial robots, autonomous vehicles.
- **Cost.** No server infrastructure to maintain per-request.

**Key techniques to make models fit on-device:**

1. **Quantization.** Reducing the numerical precision of model weights and activations.
   - *Post-training quantization (PTQ):* Convert a trained FP32 model to INT8 after training. Fast, no retraining needed, but some accuracy loss.
   - *Quantization-aware training (QAT):* Simulate quantization during training so the model learns to be robust to reduced precision. Better accuracy, more effort.
   - *Precision levels:* FP32 → FP16 → INT8 → INT4. Each halving of bit-width roughly halves model size and increases throughput, but accuracy degrades progressively.

2. **Pruning.** Removing weights (or entire neurons/channels) that contribute little to the output.
   - *Unstructured pruning:* Zeroing out individual weights. High compression ratios but requires sparse-matrix hardware to realize speed gains.
   - *Structured pruning:* Removing entire filters or attention heads. Lower compression ratios but yields immediate speed gains on standard hardware because the resulting model is a smaller dense model.

3. **Knowledge distillation.** Training a smaller "student" model to mimic the outputs of a larger "teacher" model. The student learns from the teacher's soft probabilities, which contain more information than hard labels.

4. **Architecture design.** Using architectures purpose-built for efficiency: MobileNet (depthwise separable convolutions), EfficientNet (compound scaling), TinyBERT, DistilBERT.

**Assumptions:**

- The target device has sufficient compute (CPU, NPU, DSP) and memory.
- Accuracy loss from compression is acceptable.
- The model does not need to be updated frequently (OTA updates are costly and unreliable).

**What breaks when assumptions are violated:**

- If the model is too large even after compression, it simply will not run on the device.
- If accuracy loss is unacceptable, edge serving may not be viable; a hybrid approach (edge for fast approximate answers, cloud for high-accuracy fallback) is needed.
- If the model needs frequent updates, the OTA update pipeline becomes a significant engineering burden.

---

### 3. Mathematical Formulation

#### 3.1 Throughput and Latency Analysis

For a real-time serving system, the key performance metrics are:

**Throughput** $\lambda$ (requests per second the system can handle):

$$\lambda = \frac{N \cdot C}{T_{\text{inference}}}$$

where $N$ is the number of server replicas, $C$ is the concurrency per replica (number of simultaneous requests), and $T_{\text{inference}}$ is the average inference time per request.

**Latency** (time from request arrival to response departure):

$$T_{\text{total}} = T_{\text{network}} + T_{\text{queue}} + T_{\text{preprocess}} + T_{\text{inference}} + T_{\text{postprocess}} + T_{\text{serialize}}$$

In most systems, $T_{\text{inference}}$ dominates. $T_{\text{queue}}$ increases under high load following queuing theory — for an M/M/1 queue (Poisson arrivals, exponential service times, one server):

$$T_{\text{queue}} = \frac{\rho}{(1 - \rho) \cdot \mu}$$

where $\rho = \lambda / \mu$ is the utilization (arrival rate divided by service rate). As $\rho \to 1$ (the server approaches full utilization), $T_{\text{queue}} \to \infty$. This is why you never run serving infrastructure at 100% utilization — a common rule of thumb is to keep $\rho < 0.7$.

#### 3.2 Quantization Error

When quantizing from FP32 to INT8, a floating-point value $x \in [x_{\min}, x_{\max}]$ is mapped to an integer $x_q \in [0, 255]$:

$$x_q = \text{round}\left(\frac{x - x_{\min}}{s}\right), \quad s = \frac{x_{\max} - x_{\min}}{255}$$

The *dequantized* value is:

$$\hat{x} = x_q \cdot s + x_{\min}$$

The quantization error for a single value is:

$$\epsilon = x - \hat{x}$$

The maximum absolute error is $|s / 2|$ (half the step size). For a layer with weight matrix $W$ and input $x$, the quantized output differs from the full-precision output by:

$$\Delta y = Wx - \hat{W}\hat{x} \approx \Delta W \cdot x + W \cdot \Delta x$$

where $\Delta W = W - \hat{W}$ and $\Delta x = x - \hat{x}$. This first-order approximation shows that quantization error accumulates through layers — motivating calibration (choosing $x_{\min}, x_{\max}$ carefully) and QAT.

#### 3.3 Batch Serving Throughput

For a batch job with $D$ data points, $P$ partitions, model load time $T_{\text{load}}$, and per-item inference time $t$:

$$T_{\text{total}} = T_{\text{load}} + \frac{D}{P} \cdot t \quad \text{(per partition, run in parallel)}$$

Total wall-clock time with perfect parallelism across $P$ workers is $T_{\text{load}} + (D/P) \cdot t$. In practice, overhead from data shuffling, skew (uneven partition sizes), and I/O adds a factor:

$$T_{\text{wall}} = T_{\text{load}} + \frac{D}{P} \cdot t + T_{\text{overhead}}(P)$$

$T_{\text{overhead}}$ grows with $P$ (communication cost), so there is a sweet spot — adding more partitions beyond this point *increases* total time due to coordination overhead (Amdahl's law applied to data-parallel inference).

---

### 4. Worked Examples

#### Example 1: Batch Scoring Pipeline

**Scenario.** You have 10 million user profiles to score with a churn-prediction model every night.

**Step 1 — Data ingestion.** An Airflow DAG triggers at 2 AM. The first task reads user profiles from a BigQuery table into a Spark DataFrame partitioned into 100 partitions.

**Step 2 — Model loading.** Using `mapPartitions`, each Spark executor loads the model once:

```python
def score_partition(iterator):
    import joblib
    model = joblib.load("/mnt/models/churn_v3.pkl")  # loaded once per partition
    for batch in iterator:
        features = preprocess(batch)
        predictions = model.predict_proba(features)[:, 1]
        yield batch.assign(churn_prob=predictions)

scored_df = spark_df.mapPartitions(score_partition)
```

**Step 3 — Write results.** The scored DataFrame is written back to BigQuery. Each of the 100 partitions writes independently.

**Step 4 — Downstream.** A BI dashboard reads the `churn_prob` column the next morning. Users with `churn_prob > 0.8` are flagged for retention campaigns.

**Throughput calculation.** Model inference takes 0.1 ms per row. With 100 partitions and 100,000 rows per partition:
- Per-partition time: $100{,}000 \times 0.0001\text{s} = 10\text{s}$
- Wall-clock time (parallel): ~10 s + overhead ≈ 30 s total
- This comfortably fits in a nightly window.

#### Example 2: Real-Time REST API

**Scenario.** A recommendation model must serve suggestions when a user opens the app, with P99 latency < 200 ms.

**Step 1 — Server setup.** The model is wrapped in a FastAPI endpoint:

```python
from fastapi import FastAPI
import onnxruntime as ort

app = FastAPI()
session = ort.InferenceSession("recommendation_v2.onnx")

@app.post("/recommend")
async def recommend(user_id: str):
    features = feature_store.get(user_id)  # ~5 ms from Redis
    input_tensor = preprocess(features)     # ~1 ms
    output = session.run(None, {"input": input_tensor})  # ~20 ms
    items = postprocess(output)             # ~1 ms
    return {"recommendations": items}
```

**Step 2 — Deployment.** Three replicas behind an AWS ALB. Each replica runs on a `c5.xlarge` (4 vCPUs, 8 GB RAM). The ALB distributes requests round-robin.

**Step 3 — Latency budget.** Feature lookup: 5 ms. Preprocessing: 1 ms. Inference: 20 ms. Postprocessing: 1 ms. Network: 10 ms. Total: 37 ms (well under 200 ms P99).

**Step 4 — Capacity planning.** Each replica handles ~50 concurrent requests at 20 ms inference ≈ 2,500 RPS per replica. Three replicas ≈ 7,500 RPS total. Peak traffic is 5,000 RPS → comfortable headroom.

#### Example 3: Edge Deployment with Quantization

**Scenario.** Deploy an image-classification model on a mobile phone.

**Original model:** ResNet-50, FP32, 97.8 MB, 4.1 billion FLOPs, 76.1% top-1 accuracy on ImageNet.

**Step 1 — Post-training quantization (INT8):**
- Size: 97.8 MB → 24.5 MB (4× reduction)
- Accuracy: 76.1% → 75.4% (0.7 pp drop)
- Latency on Snapdragon 888: 45 ms → 18 ms

**Step 2 — Structured pruning (remove 30% of filters):**
- Size: 24.5 MB → 17.1 MB
- Accuracy: 75.4% → 74.8% (additional 0.6 pp drop)
- Latency: 18 ms → 12 ms

**Step 3 — Export.** Convert to TensorFlow Lite (`.tflite`) for Android or Core ML (`.mlmodel`) for iOS. The model runs entirely on the device's NPU.

**Result.** The model went from 97.8 MB / 76.1% / ~200 ms (server-side FP32) to 17.1 MB / 74.8% / 12 ms (on-device INT8 + pruned). The 1.3 pp accuracy trade-off is acceptable for the use case.

---

### 5. Relevance to Machine Learning Practice

**Where each pattern is used:**

| Pattern | Typical use cases | Latency | Freshness |
|---|---|---|---|
| Batch | Recommendation precomputation, churn scoring, reporting, periodic retraining data generation | Minutes–hours | Stale by design |
| Real-time | Fraud detection, ad bidding, search ranking, chatbots, content moderation | Milliseconds | Near-real-time |
| Edge | On-device keyboard prediction, camera filters, voice assistants, autonomous driving | Sub-millisecond to milliseconds | Real-time, local |

**When to use batch:** Predictions are consumed asynchronously; freshness requirements are lax; the input dataset is large and well-defined; cost efficiency is paramount (spot instances, off-peak scheduling).

**When to use real-time:** Predictions must reflect the user's current state; latency is an SLA; the input arrives unpredictably; the system must respond per-request.

**When to use edge:** Privacy is critical; connectivity is unreliable; ultra-low latency is required; per-request cloud cost is prohibitive at scale (billions of on-device inferences per day).

**Trade-offs:**

- **Batch** is cheapest per prediction but introduces staleness. You cannot react to real-time signals.
- **Real-time** offers freshness but requires always-on infrastructure, load balancing, and careful latency engineering.
- **Edge** eliminates network latency and server costs but constrains model size, makes updates hard, and splits your model fleet across heterogeneous devices.

**Hybrid architectures** are common: edge models handle latency-critical initial responses, while cloud models provide richer follow-up analysis. Similarly, batch precomputation + real-time adjustment (e.g., precompute base recommendations in batch, adjust ranking in real-time based on current session context) is a standard pattern.

---

### 6. Common Pitfalls & Misconceptions

1. **"We can just wrap the training code in a Flask app."** Training code often has unnecessary dependencies (data loaders, augmentation, logging to experiment trackers) that bloat the serving container, increase cold-start time, and introduce security vulnerabilities. Always create a minimal serving artifact.

2. **Ignoring cold-start time.** The first request after a server boots is slow because the model must be loaded into memory (and potentially onto the GPU). For large models, cold start can be 30+ seconds. Mitigations: warm-up requests at startup, keeping minimum replicas always running, model preloading.

3. **Optimizing only inference time.** In many systems, feature computation and network I/O dominate latency, not model inference itself. Profile the full request path before optimizing the model.

4. **Batch serving with stale features.** If your batch job scores users with features computed yesterday, but the user's behavior changed today, the predictions are stale. Combine batch predictions with real-time feature adjustments when freshness matters.

5. **Over-quantizing.** Dropping from FP32 to INT4 may seem appealing for size reduction, but accuracy can collapse for models not designed for extreme quantization. Always measure accuracy on a held-out calibration set *at each precision level*.

6. **"Edge = mobile."** Edge also includes browsers (TensorFlow.js, ONNX.js), embedded devices (Raspberry Pi, Jetson Nano), and automotive ECUs. Each has different constraints.

7. **Ignoring model versioning.** When you update the model on the server, the change is immediate. When you update the model on edge devices, you have a fragmented fleet running different versions for weeks. Plan for version coexistence.

---

## Part II — Model Formats

---

### 1. Motivation & Intuition

A trained model exists as a set of learned parameters (weights, biases, embeddings) plus a computational graph (the sequence of operations that transforms input into output). To move a model from the training environment to production, you need to **serialize** it — convert the in-memory objects into a file that can be stored, versioned, transferred, and loaded by a serving system.

The choice of serialization format has downstream consequences:

- **Portability.** Can the model be loaded by a different framework or language? A model trained in PyTorch that must be served in a C++ inference engine needs a portable format.
- **Performance.** Some formats enable graph-level optimizations (operator fusion, constant folding) that make inference faster.
- **Security.** Some formats (Pickle) can execute arbitrary code on load, creating security vulnerabilities.
- **Completeness.** Some formats save only weights; others save the full computational graph, preprocessing, and postprocessing logic.

#### The four main formats:

| Format | Framework | Saves Graph? | Portable? | Optimized? | Secure? |
|---|---|---|---|---|---|
| Pickle (.pkl) | Python (sklearn, general) | No (saves Python objects) | No (Python only) | No | **No** (arbitrary code execution) |
| ONNX (.onnx) | Cross-framework | Yes (standardized ops) | Yes (C++, Java, JS, ...) | Yes (via ONNX Runtime) | Yes |
| TF SavedModel | TensorFlow | Yes (TF graph + assets) | Partial (TF ecosystem) | Yes (XLA, TF-TRT) | Mostly |
| TorchScript (.pt) | PyTorch | Yes (traced/scripted graph) | Partial (libtorch C++) | Yes (torch.jit optimizations) | Mostly |

### 2. Conceptual Foundations

#### 2.1 Pickle

Pickle is Python's native serialization protocol. It converts any Python object — a scikit-learn pipeline, a dictionary, a custom class — into a byte stream and back. Libraries like scikit-learn use `joblib` (a Pickle variant optimized for large NumPy arrays).

**What Pickle saves:** The Python object's `__dict__`, its class hierarchy, and enough metadata to reconstruct it. It does *not* save the source code of the class — it saves a reference to the class by module path and name.

**Critical limitation — security.** Pickle can execute arbitrary Python code during deserialization. A malicious Pickle file can run `os.system("rm -rf /")` when you call `pickle.load()`. *Never load a Pickle file from an untrusted source.* In production, this means Pickle files must come from a trusted, access-controlled model registry.

**Critical limitation — portability.** Pickle files can only be loaded by Python, and they require the exact same class definitions to be available in the loading environment. If the class moved to a different module, or a dependency version changed, loading fails.

#### 2.2 ONNX (Open Neural Network Exchange)

ONNX is a vendor-neutral, open format for representing machine learning models. It defines a standard set of operators (MatMul, Conv, Relu, Softmax, etc.) and a protobuf-based file format.

**How it works:**

1. You train a model in PyTorch, TensorFlow, or scikit-learn.
2. You export the model to ONNX using a framework-specific exporter (`torch.onnx.export`, `tf2onnx`, `sklearn-onnx`).
3. The exporter traces the computation graph and maps each operation to an ONNX operator.
4. The resulting `.onnx` file contains the graph definition and the learned weights.
5. At serving time, ONNX Runtime (or another ONNX-compatible runtime) loads the file and runs inference.

**Why ONNX matters:**

- **Portability.** Train in PyTorch, serve in C++. Train in TensorFlow, serve in a browser (via ONNX.js).
- **Optimization.** ONNX Runtime applies graph optimizations (constant folding, operator fusion, memory planning) automatically.
- **Hardware acceleration.** ONNX Runtime supports execution providers for CUDA, TensorRT, OpenVINO, DirectML, and more.

**Limitations:**

- Not all operations are supported. Custom layers, complex control flow (if/else, loops), and dynamic shapes can cause export failures.
- The ONNX opset (version of the operator specification) must be compatible between exporter and runtime.

#### 2.3 TensorFlow SavedModel

The SavedModel is TensorFlow's standard serialization format. It stores:

- The computational graph (as a protobuf `GraphDef`).
- The trained variables (weights, as checkpoint files).
- Signatures (named entry points, like "predict" or "classify", that define input/output tensor specifications).
- Assets (vocabulary files, lookup tables).

**The directory structure:**

```
saved_model/
├── saved_model.pb          # Graph definition + signatures
├── variables/
│   ├── variables.data-00000-of-00001  # Weight values
│   └── variables.index     # Weight metadata
└── assets/                 # Additional files (vocab, etc.)
```

SavedModel integrates with the entire TensorFlow ecosystem: TF Serving, TF Lite (for mobile/edge), TF.js (for browser), and TFX (for production pipelines). XLA compilation and TensorRT integration provide hardware-level optimizations.

#### 2.4 TorchScript

TorchScript is PyTorch's mechanism for converting eager-mode Python code into a serializable, optimizable intermediate representation.

**Two modes:**

- **Tracing (`torch.jit.trace`).** You provide a sample input; PyTorch records every operation that executes. The resulting graph captures the exact sequence of operations for that input. Limitation: data-dependent control flow (if/else branches, variable-length loops) is *not* captured — the trace follows only the code path exercised by the sample input.

- **Scripting (`torch.jit.script`).** PyTorch's compiler parses a subset of Python and converts it to TorchScript IR. This handles control flow correctly but restricts you to the supported Python subset (no arbitrary Python libraries, limited data structures).

**The resulting `.pt` file** can be loaded in C++ via `libtorch`, enabling deployment without a Python runtime. This is critical for production environments where Python's GIL, startup time, or dependency management is unacceptable.

### 3. Mathematical Formulation

Model serialization is primarily an engineering concern, but one quantitative aspect is **model size estimation**.

For a neural network with $L$ layers, where layer $l$ has a weight matrix $W_l \in \mathbb{R}^{m_l \times n_l}$ and bias $b_l \in \mathbb{R}^{m_l}$:

$$\text{Total parameters} = \sum_{l=1}^{L} (m_l \cdot n_l + m_l)$$

At FP32 (4 bytes per parameter):

$$\text{Size (bytes)} = 4 \times \sum_{l=1}^{L} (m_l \cdot n_l + m_l)$$

For ONNX with INT8 quantization, the size is approximately $\frac{1}{4}$ of the FP32 size (plus a small overhead for scale/zero-point parameters per tensor).

**Example:** BERT-base has ~110M parameters. FP32 size: $110 \times 10^6 \times 4 = 440$ MB. INT8 ONNX: ~110 MB. FP16: ~220 MB.

### 4. Worked Examples

#### Example: Exporting a PyTorch Model to ONNX

```python
import torch
import torch.nn as nn

# 1. Define and train a simple model
class FraudDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

model = FraudDetector()
model.eval()  # Critical: set to eval mode (disables dropout, batchnorm running stats)

# 2. Create a dummy input with the correct shape
dummy_input = torch.randn(1, 50)

# 3. Export to ONNX
torch.onnx.export(
    model, dummy_input, "fraud_detector.onnx",
    input_names=["transaction_features"],
    output_names=["fraud_probability"],
    dynamic_axes={"transaction_features": {0: "batch_size"}},  # allow variable batch
    opset_version=17
)

# 4. Verify: load with ONNX Runtime and run inference
import onnxruntime as ort
session = ort.InferenceSession("fraud_detector.onnx")
test_input = {"transaction_features": torch.randn(5, 50).numpy()}
output = session.run(None, test_input)
print(output[0].shape)  # (5, 1) — five predictions
```

**Key details:**
- `model.eval()` is critical — without it, dropout and batch normalization behave as during training, producing incorrect predictions.
- `dynamic_axes` allows the batch dimension to vary at inference time. Without this, the ONNX model would only accept batch size 1.
- `opset_version` controls which ONNX operators are available. Higher opsets support more operations but require a newer runtime.

### 5. Relevance to Machine Learning Practice

- **Pickle** is fine for prototyping and scikit-learn models in trusted environments. Never use it for models received from external parties.
- **ONNX** is the go-to for cross-framework deployment and when you want hardware-specific optimizations without framework lock-in.
- **TF SavedModel** is the natural choice if your entire stack is TensorFlow, especially with TF Serving.
- **TorchScript** is the natural choice for PyTorch-native deployments, especially when you need C++ inference.

**Practical recommendation:** For any model that will be served in production, avoid Pickle. Export to ONNX or the framework's native optimized format (SavedModel, TorchScript). This gives you graph-level optimizations, hardware acceleration, and eliminates the Python runtime dependency.

### 6. Common Pitfalls & Misconceptions

1. **Forgetting `model.eval()` before export.** This is the single most common deployment bug. Dropout and batch norm behave differently in training vs. inference mode. Exporting in training mode produces silently wrong predictions.

2. **Assuming ONNX export is always lossless.** Some operations (custom CUDA kernels, Python-side control flow) cannot be represented in ONNX. Always verify numerical equivalence between the original model and the ONNX model on a validation set.

3. **"Pickle is fine for production."** Pickle is *convenient*, not *safe* or *portable*. Every security audit flags Pickle deserialization as a vulnerability.

4. **Ignoring opset compatibility.** An ONNX model exported with opset 17 may not load on a runtime that only supports opset 13. Pin your opset version to match your deployment runtime.

5. **Tracing dynamic models.** Using `torch.jit.trace` on a model with data-dependent control flow silently produces a model that only follows one code path. Use `torch.jit.script` for models with if/else or variable-length loops.

---

## Part III — Scaling Strategies

---

### 1. Motivation & Intuition

A single server running a model can handle a limited number of requests per second. When traffic exceeds this capacity, requests queue up, latency increases, and eventually the server crashes or drops requests. **Scaling** is the set of strategies for increasing a system's capacity to handle more traffic.

Consider a concrete progression:

- **Day 1.** Your recommendation API gets 10 requests per second. A single server handles it easily.
- **Month 3.** A marketing campaign drives traffic to 1,000 RPS. One server is overwhelmed.
- **Year 1.** The product is international. Traffic varies from 500 RPS at 3 AM to 10,000 RPS at noon. You need elasticity.

There are three fundamental approaches:

- **Vertical scaling.** Get a bigger server (more CPU, more RAM, faster GPU).
- **Horizontal scaling.** Get more servers and distribute traffic across them.
- **Model partitioning.** Split the model itself across multiple servers.

### 2. Conceptual Foundations

#### 2.1 Vertical Scaling

**Definition.** Replacing a server with a more powerful one — more CPU cores, more RAM, a faster GPU, faster SSDs.

**Advantages:** Simple. No code changes. No distributed systems complexity. One server, one model copy, no inter-node communication.

**Limitations:**
- **Hard ceiling.** There is a largest available machine. You cannot vertically scale beyond the biggest instance your cloud provider offers.
- **Cost curve.** Cost scales super-linearly. A machine with 2× the CPU costs more than 2× the price.
- **Single point of failure.** One server = one failure domain. If it crashes, all inference stops.

**When to use:** Early-stage products with moderate traffic; models that benefit from large memory (e.g., large embedding tables that must fit in RAM); GPU inference where a single larger GPU (A100 80GB vs. A10 24GB) eliminates the need for model partitioning.

#### 2.2 Horizontal Scaling

**Definition.** Running multiple copies (replicas) of the model server behind a load balancer. Each replica independently handles requests.

**Key components:**

- **Load balancer.** Routes incoming requests to replicas. Strategies:
  - *Round-robin:* Requests are distributed sequentially. Simple but ignores server load.
  - *Least connections:* Sends to the replica with the fewest active connections. Better under variable request latency.
  - *Weighted:* Assigns more traffic to more powerful replicas.
  - *Consistent hashing:* Routes requests with the same key (e.g., user ID) to the same replica. Useful for caching.

- **Auto-scaling.** Automatically adjusts the number of replicas based on a metric:
  - *Target CPU utilization:* Scale up when average CPU > 70%, scale down when < 30%.
  - *Request queue depth:* Scale up when the queue grows beyond a threshold.
  - *Custom metrics:* Scale on P99 latency, GPU utilization, or requests per second.

**How auto-scaling works:**

1. A metrics collector (CloudWatch, Prometheus) monitors the target metric.
2. When the metric crosses a threshold, the auto-scaler sends a signal to the orchestrator (Kubernetes HPA, AWS Auto Scaling).
3. The orchestrator provisions new replicas (launches pods/instances, pulls the container image, loads the model).
4. The load balancer detects the new replicas and begins routing traffic to them.
5. When traffic subsides, the process reverses — replicas are drained (existing requests complete) and terminated.

**Scaling parameters:**

- *Minimum replicas:* Always keep at least this many running (avoids cold-start latency).
- *Maximum replicas:* Cap to control cost.
- *Cooldown period:* After a scaling event, wait this long before scaling again (prevents oscillation).
- *Scale-up speed:* How many replicas to add per step.

**Assumptions:**
- Each replica is stateless — it can handle any request without relying on local state from previous requests.
- The model fits in a single replica's memory.
- Requests are independent and can be distributed arbitrarily.

**What breaks when assumptions are violated:**
- Stateful models (e.g., conversational agents maintaining session context) require sticky sessions or external state stores, complicating load balancing.
- If the model does not fit in a single replica's memory, you need model partitioning (see below).
- If requests have data dependencies (e.g., request B depends on the result of request A), simple load balancing cannot parallelize them.

#### 2.3 Model Partitioning

**Definition.** Splitting a single model across multiple servers because it is too large to fit in one server's memory or because parallel execution across servers is faster.

**Two main strategies:**

- **Pipeline parallelism (layer sharding).** Different layers of the model run on different servers. Server 1 runs layers 1–12, server 2 runs layers 13–24. A request flows through server 1 first, then server 2. Latency is the sum of per-shard latencies plus inter-shard communication. Throughput can be increased by pipelining: while server 2 processes request A's second half, server 1 starts processing request B's first half.

- **Tensor parallelism.** A single layer's computation is split across multiple servers. For a matrix multiplication $Y = XW$ where $W \in \mathbb{R}^{d \times d}$, you split $W$ column-wise across $K$ GPUs:

$$W = [W_1 | W_2 | \ldots | W_K], \quad Y_k = X \cdot W_k$$

Each GPU computes a partial result $Y_k$. The full result $Y$ is assembled via an all-gather operation. This reduces per-GPU memory by $K$ but introduces communication overhead proportional to the tensor size and the number of GPUs.

**Ensemble distribution.** A different form of model partitioning: instead of splitting one model, you distribute an ensemble of models across servers. Each server runs a different model, and predictions are aggregated (averaged, voted on) by a coordinator.

### 3. Mathematical Formulation

#### 3.1 Horizontal Scaling Capacity

With $N$ replicas, each handling $\mu$ requests per second, the system capacity is:

$$\lambda_{\max} = N \cdot \mu$$

If arrival rate $\lambda > \lambda_{\max}$, the system is overloaded. Auto-scaling reacts with a delay $\delta$:

$$N(t) = \max\left(N_{\min}, \min\left(N_{\max}, \left\lceil \frac{\lambda(t - \delta)}{\mu \cdot \rho_{\text{target}}} \right\rceil\right)\right)$$

where $\rho_{\text{target}}$ is the target utilization (e.g., 0.7). The delay $\delta$ includes metric collection latency + orchestrator reaction time + instance boot time + model loading time. For a Kubernetes pod with a pre-pulled container image, $\delta$ might be 30–60 seconds. For a new VM with a large model, $\delta$ might be 5–10 minutes. This delay is why you maintain minimum replicas and why predictive auto-scaling (scaling based on forecasted traffic, not current traffic) is valuable.

#### 3.2 Pipeline Parallelism Throughput

With $K$ pipeline stages, each taking time $T_k$, and a micro-batch size of $m$:

- Latency for a single request: $\sum_{k=1}^{K} T_k$
- Throughput with pipelining (steady state): $\frac{1}{\max_k T_k}$ (limited by the slowest stage)
- Pipeline bubble: the first $K-1$ stages are idle while the pipeline fills and drains. With $M$ micro-batches, the bubble fraction is:

$$\text{Bubble fraction} = \frac{K - 1}{M + K - 1}$$

For $K = 4$ stages and $M = 16$ micro-batches: bubble = $3/19 \approx 15.8\%$. Increasing $M$ amortizes the bubble but increases memory usage (more in-flight micro-batches).

#### 3.3 Tensor Parallelism Communication Cost

For an all-reduce on a tensor of size $S$ bytes across $K$ GPUs connected by links with bandwidth $B$:

$$T_{\text{comm}} = \frac{2(K-1)}{K} \cdot \frac{S}{B}$$

(using the ring all-reduce algorithm). For $S = 50$ MB, $K = 8$ GPUs, $B = 50$ GB/s (NVLink):

$$T_{\text{comm}} = \frac{14}{8} \cdot \frac{50 \times 10^6}{50 \times 10^9} = 1.75 \times 10^{-3}\text{s} = 1.75\text{ ms}$$

If inference takes 20 ms per layer and there are 4 all-reduce operations per layer, communication adds $4 \times 1.75 = 7$ ms, increasing per-layer time by 35%. This communication overhead is why tensor parallelism only helps when the computation time per GPU is significantly reduced by the split.

### 4. Worked Examples

#### Example: Auto-Scaling Configuration

**Scenario.** You deploy a search-ranking model on Kubernetes. Baseline traffic: 2,000 RPS. Peak traffic (Black Friday): 20,000 RPS. Each pod handles 500 RPS at 70% CPU utilization.

**Configuration:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ranking-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ranking-model
  minReplicas: 6      # handles 3,000 RPS baseline with headroom
  maxReplicas: 50      # handles 25,000 RPS peak with headroom
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70   # scale when CPU > 70%
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30   # wait 30s before acting on scale-up signal
      policies:
      - type: Pods
        value: 5          # add up to 5 pods at a time
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # wait 5 min before scaling down (avoid flapping)
      policies:
      - type: Pods
        value: 2          # remove at most 2 pods at a time
        periodSeconds: 60
```

**Capacity analysis:**

- Minimum replicas (6) × 500 RPS = 3,000 RPS (handles baseline + 50% headroom).
- Maximum replicas (50) × 500 RPS = 25,000 RPS (handles Black Friday + 25% headroom).
- Scale-up from 6 to 40 pods at 5 pods/minute takes 7 minutes — traffic ramp must be slower, or set minimum replicas higher before the event.

### 5. Relevance to Machine Learning Practice

**Decision framework:**

- **Start vertical.** Use the biggest single machine that fits your budget. Simpler to debug, no distributed systems headaches.
- **Go horizontal when** vertical is too expensive, you need fault tolerance, or traffic is variable enough to justify auto-scaling.
- **Use model partitioning when** the model does not fit on a single device or when parallelism is needed to meet latency SLAs for very large models (e.g., LLMs with hundreds of billions of parameters).

---

**[← Previous Chapter: Robust Evaluation Under Shift](robust_evaluation_under_shift.md) | [Table of Contents](index.md) | [Next Chapter: Monitoring →](monitoring_guide.md)**

**Common anti-patterns:**

- Horizontal scaling without removing statefulness first (sessions, in-memory caches) — leads to inconsistent behavior across replicas.
- Setting auto-scaling min replicas to 0 (scale-to-zero) — cold starts are brutal for models with large load times.
- Ignoring load balancer health checks — a replica that has loaded the model but is still warming up gets traffic and times out.

### 6. Common Pitfalls & Misconceptions

1. **"Just add more servers" without profiling.** If the bottleneck is a database call, not model inference, adding more model servers does nothing.

2. **Symmetric scaling assumptions.** Scaling up is usually faster than scaling down. Aggressive scale-down can cause thrashing (repeated scale-up/down cycles) if traffic is bursty.

3. **Ignoring GPU memory fragmentation.** On GPUs, even if total memory is sufficient, fragmentation can prevent loading a model. Restart the process or use memory pools.

4. **Model partitioning without fast interconnect.** Tensor parallelism across machines connected by 10 Gbps Ethernet (instead of NVLink/InfiniBand) can be *slower* than running on a single GPU due to communication overhead.

5. **Auto-scaling on the wrong metric.** Scaling on CPU utilization for a GPU-bound model means the auto-scaler never triggers. Scale on GPU utilization or request latency instead.

---

## Part IV — A/B Testing Framework

---

### 1. Motivation & Intuition

You have deployed Model A (the current production model) and have trained Model B (a challenger). Model B has better offline metrics (higher AUC, lower RMSE). Should you replace Model A with Model B?

**Not necessarily.** Offline metrics measure performance on historical data. Production involves different distributions, user behavior, and business dynamics. The only way to know if Model B is truly better *in production* is to test it on live traffic.

**A/B testing** is the statistical framework for comparing two (or more) variants in production by randomly assigning users to variants and measuring the impact on a business metric (revenue, click-through rate, engagement, etc.).

**Why randomization?** Without randomization, you might deploy Model B to power users (who are inherently more engaged) and conclude it is better, when in reality the difference is due to user selection, not the model. Randomization ensures that, in expectation, the two groups are identical in all respects except the treatment.

### 2. Conceptual Foundations

#### 2.1 Core Components

- **Control (A).** The current production system. The baseline against which the challenger is measured.
- **Treatment (B).** The new variant being tested.
- **Randomization unit.** The entity that is randomly assigned to A or B. Common choices:
  - *User ID:* Each user sees one variant throughout the experiment. Ensures consistent experience.
  - *Session ID:* Each session is independently randomized. Allows within-user comparison but can confuse users if variants differ visibly.
  - *Request ID:* Each request is independently randomized. Highest statistical power but no consistency — a user may see different results on consecutive requests.
  - *Device ID / Cookie:* Useful when user identity is not available.

  **Choosing the randomization unit is one of the most consequential design decisions.** Using request-level randomization for a visual UI change is disastrous (the user sees the interface flickering between variants). Using user-level randomization for a rare event (like purchase) reduces statistical power because each user contributes at most one observation.

- **Metric.** The quantity you are trying to improve. Primary metric (the one you make the ship/no-ship decision on) and guardrail metrics (metrics that must not degrade, like latency, error rate, or revenue).

- **Traffic allocation.** What fraction of traffic goes to each variant. Typically 50/50 for maximum statistical power, but asymmetric splits (90/10, 95/5) are common when the risk of the treatment is high.

#### 2.2 Statistical Framework

**Null hypothesis ($H_0$):** There is no difference between A and B. $\mu_A = \mu_B$.

**Alternative hypothesis ($H_1$):** There is a difference. $\mu_A \neq \mu_B$ (two-sided) or $\mu_B > \mu_A$ (one-sided).

**Test statistic.** For comparing means with large samples, the two-sample z-test:

$$Z = \frac{\bar{X}_B - \bar{X}_A}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$

where $\bar{X}_A, \bar{X}_B$ are sample means, $s_A^2, s_B^2$ are sample variances, and $n_A, n_B$ are sample sizes.

**p-value.** The probability of observing a test statistic at least as extreme as the one observed, assuming $H_0$ is true. If $p < \alpha$ (typically $\alpha = 0.05$), reject $H_0$.

**Statistical power ($1 - \beta$).** The probability of correctly rejecting $H_0$ when it is false — i.e., the probability of detecting a real effect. Typical target: 80%.

**Type I error ($\alpha$).** Concluding there is a difference when there is none (false positive).
**Type II error ($\beta$).** Concluding there is no difference when there is one (false negative).

#### 2.3 Statistical Power and Sample Size

The minimum sample size per group to detect an effect of size $\delta$ with significance level $\alpha$ and power $1 - \beta$:

$$n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2\sigma^2}{\delta^2}$$

where $z_{1-\alpha/2} \approx 1.96$ for $\alpha = 0.05$ (two-sided) and $z_{1-\beta} \approx 0.84$ for 80% power.

**Intuition behind each term:**

- **$\sigma^2$ (variance):** Higher variance in the metric → harder to detect a signal in the noise → need more data.
- **$\delta$ (effect size):** Smaller effects are harder to detect → need more data.
- **$\alpha$ (significance level):** Stricter significance → need more data.
- **$1 - \beta$ (power):** Higher power → need more data.

#### 2.4 Sequential Testing

Classical hypothesis testing assumes a fixed sample size: you decide $n$ in advance, run the experiment, collect exactly $n$ observations, and then analyze. **Sequential testing** allows you to analyze results continuously as data accumulates and stop the experiment early if the evidence is conclusive.

**Why this matters:** In practice, you want to stop losing experiments early (limit user harm) and ship winning experiments early (capture value sooner). But naively peeking at p-values repeatedly inflates the Type I error rate — if you check daily and stop when $p < 0.05$, the true false positive rate is much higher than 5%.

**Solutions:**

- **Alpha spending functions (O'Brien-Fleming, Pocock).** Allocate the total $\alpha$ budget across interim analyses. O'Brien-Fleming uses a very strict threshold early and relaxes it later; Pocock uses a constant threshold throughout.

- **Always-valid confidence sequences.** Construct confidence intervals that are valid at *every* time point, not just at a prespecified sample size. If the interval excludes zero, the effect is significant regardless of when you look. This is the modern approach used by companies like Netflix and Spotify.

- **Bayesian sequential testing.** Instead of p-values, compute the posterior probability that B is better than A. Stop when the posterior exceeds a threshold (e.g., P(B > A) > 0.95 or P(B > A) < 0.05).

### 3. Mathematical Formulation

#### 3.1 Sample Size Derivation

For a two-sample z-test comparing means, the minimum detectable effect (MDE) $\delta$ with significance $\alpha$ and power $1 - \beta$:

Under $H_0$: $\bar{X}_B - \bar{X}_A \sim \mathcal{N}\left(0, \frac{2\sigma^2}{n}\right)$

Under $H_1$: $\bar{X}_B - \bar{X}_A \sim \mathcal{N}\left(\delta, \frac{2\sigma^2}{n}\right)$

We reject $H_0$ when $|Z| > z_{1-\alpha/2}$. Power is:

$$1 - \beta = P\left(|Z| > z_{1-\alpha/2} \mid H_1\right) = P\left(Z > z_{1-\alpha/2} \mid \mu = \delta\right)$$

Substituting and solving for $n$:

$$P\left(\frac{\bar{X}_B - \bar{X}_A}{\sqrt{2\sigma^2/n}} > z_{1-\alpha/2} \mid \mu = \delta\right) = 1 - \beta$$

$$P\left(\frac{(\bar{X}_B - \bar{X}_A) - \delta}{\sqrt{2\sigma^2/n}} > z_{1-\alpha/2} - \frac{\delta}{\sqrt{2\sigma^2/n}}\right) = 1 - \beta$$

The left side is a standard normal under $H_1$, so:

$$z_{1-\alpha/2} - \frac{\delta}{\sqrt{2\sigma^2/n}} = -z_{1-\beta}$$

$$\frac{\delta}{\sqrt{2\sigma^2/n}} = z_{1-\alpha/2} + z_{1-\beta}$$

$$n = \frac{2\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$$

#### 3.2 Multiple Comparisons

When testing $m$ variants simultaneously (A/B/C/D), the probability of at least one false positive is:

$$P(\text{at least one false positive}) = 1 - (1 - \alpha)^m \approx m\alpha \text{ for small } \alpha$$

For $m = 20$ variants at $\alpha = 0.05$: $P \approx 0.64$. You expect roughly one false positive by chance.

**Corrections:**
- **Bonferroni:** Use $\alpha / m$ for each test. Conservative (low power).
- **Holm-Bonferroni:** Stepwise procedure. Less conservative.
- **Benjamini-Hochberg (FDR control):** Controls the expected fraction of false discoveries, not the probability of any false discovery. More powerful when $m$ is large.

### 4. Worked Examples

#### Example: CTR A/B Test

**Scenario.** You are testing a new recommendation model. The primary metric is click-through rate (CTR). Baseline CTR is 5% ($p_A = 0.05$). You want to detect a 5% *relative* improvement (i.e., $p_B = 0.0525$, an absolute lift of 0.25 percentage points). Significance: $\alpha = 0.05$, power: 80%.

**Step 1 — Compute variance.** For a proportion, $\sigma^2 = p(1-p)$.

$$\sigma^2 \approx 0.05 \times 0.95 = 0.0475$$

**Step 2 — Compute $\delta$.** The absolute effect size: $\delta = 0.0525 - 0.05 = 0.0025$.

**Step 3 — Compute sample size per group.**

$$n = \frac{2 \times 0.0475 \times (1.96 + 0.84)^2}{0.0025^2} = \frac{2 \times 0.0475 \times 7.84}{0.00000625} = \frac{0.7448}{0.00000625} \approx 119{,}168$$

**Step 4 — Interpret.** You need ~119,168 users per group, or ~238,336 total. If your site gets 50,000 users per day and you allocate 100% to the experiment (50/50 split), you need about 5 days to reach significance.

**Step 5 — Run the test.** After 5 days:
- Control: 119,500 users, 5,975 clicks, CTR = 5.00%
- Treatment: 119,200 users, 6,283 clicks, CTR = 5.27%

$$Z = \frac{0.0527 - 0.0500}{\sqrt{\frac{0.05 \times 0.95}{119500} + \frac{0.0527 \times 0.9473}{119200}}} = \frac{0.0027}{\sqrt{0.000000398 + 0.000000418}} = \frac{0.0027}{0.000904} \approx 2.99$$

$p$-value $\approx 0.003 < 0.05$. Reject $H_0$. The new model's improvement is statistically significant.

### 5. Relevance to Machine Learning Practice

**A/B testing is the gold standard for evaluating ML models in production.** Offline metrics (AUC, RMSE, accuracy) measure predictive performance on historical data. A/B tests measure *business impact* on live traffic. These can diverge:

- A model with higher AUC may produce predictions that are better calibrated for the wrong objective (e.g., optimizing clicks when the business cares about purchases).
- A model that is 2% more accurate but 3× slower may degrade the user experience enough to reduce engagement.
- A model may perform well on the historical distribution but poorly on the current distribution (distribution shift).

**Interaction with other deployment patterns:**

- A/B tests can compare batch vs. real-time serving of the same model.
- A/B tests can compare different model architectures, feature sets, or hyperparameters.
- A/B tests can evaluate infrastructure changes (e.g., switching from TensorFlow Serving to Triton).

**Guardrail metrics** prevent shipping a model that improves the primary metric at the expense of something critical. Example guardrails: P99 latency must not increase by more than 10%; error rate must not exceed 0.1%; revenue must not decrease.

### 6. Common Pitfalls & Misconceptions

1. **Peeking at results.** Checking p-values daily and stopping when $p < 0.05$ inflates the false positive rate well beyond 5%. Use sequential testing or commit to a fixed sample size.

2. **Wrong randomization unit.** Using request-level randomization for user-facing changes causes flickering. Using user-level randomization for rare events (conversions) drastically reduces power.

3. **Network effects / interference (SUTVA violation).** The Stable Unit Treatment Value Assumption (SUTVA) assumes that one user's treatment does not affect another user's outcome. In social networks or marketplaces, this is violated — treating sellers differently affects buyers. Mitigation: cluster randomization (randomize by geographic region or social cluster).

4. **Novelty / primacy effects.** Users may engage more with a new feature simply because it is new (novelty) or less because they dislike change (primacy). Wait for the effect to stabilize (typically 1–2 weeks) before drawing conclusions.

5. **Survivorship bias.** If the treatment causes some users to churn (leave the platform), the remaining treatment users may look artificially good. Always measure the metric on the *assigned* population (intent-to-treat), not just the users who remained.

6. **Underpowered tests.** Running a test for "a few days" without a power analysis often means the test is too small to detect realistic effect sizes. Non-significant results are then misinterpreted as "no effect" when in reality the test simply lacked power.

7. **Multiple testing without correction.** Testing 20 metrics and celebrating the one with $p < 0.05$ is p-hacking. Pre-register the primary metric and apply corrections for secondary metrics.

---

## Part V — Canary Deployments

---

### 1. Motivation & Intuition

Imagine deploying a new model to 100% of traffic simultaneously. If the model has a bug — a preprocessing error that produces NaN predictions for certain input types, a version mismatch in the feature pipeline, or a latency regression — every user is affected at once. Rollback takes minutes to hours. During that time, revenue drops, users churn, and trust erodes.

**Canary deployment** mitigates this risk by rolling out the new model to a small fraction of traffic first (the "canary"), monitoring for problems, and gradually increasing traffic if everything looks healthy — or automatically rolling back if something goes wrong.

The name comes from the coal-mining practice of bringing a canary into the mine: if the canary died, miners knew the air was toxic before they were affected.

### 2. Conceptual Foundations

#### 2.1 How Canary Deployment Works

**Phase 1 — Canary launch.** Deploy the new model alongside the existing one. Route 1–5% of traffic to the new model. The other 95–99% continues to use the old model.

**Phase 2 — Monitoring.** Compare the canary's metrics against the baseline:
- Latency (P50, P95, P99)
- Error rate (5xx responses, exceptions, NaN predictions)
- Business metrics (CTR, conversion rate, revenue)
- Infrastructure metrics (CPU, memory, GPU utilization)

**Phase 3 — Gradual promotion.** If canary metrics are healthy after a burn-in period (e.g., 30 minutes), increase traffic: 1% → 5% → 25% → 50% → 100%. Each step has its own monitoring period.

**Phase 4 — Automatic rollback.** If any metric crosses a predefined threshold at any step, traffic is automatically reverted to the old model. Examples:
- Error rate exceeds 1% (absolute) or 2× baseline (relative).
- P99 latency exceeds 500 ms or 1.5× baseline.
- Revenue per request drops by more than 3%.

#### 2.2 Canary vs. A/B Testing

These are complementary, not interchangeable:

| Aspect | Canary deployment | A/B testing |
|---|---|---|
| Purpose | Risk mitigation (is it broken?) | Impact measurement (is it better?) |
| Traffic split | Small (1–5%) initially | Equal (50/50) for power |
| Duration | Hours to days | Days to weeks |
| Decision criterion | Absence of degradation | Statistical significance of improvement |
| Metrics focus | Error rates, latency, crashes | Business KPIs |
| Rollback | Automatic | Manual decision after analysis |

**Typical workflow:** Canary → if healthy, promote to A/B test → if significantly better, promote to 100%.

#### 2.3 Automatic Rollback Triggers

Rollback triggers must be carefully designed. Too sensitive → false alarms and unnecessary rollbacks. Too lenient → real problems go undetected.

**Statistical approach to rollback triggers:**

For each metric $m$, compute a rolling baseline from the control population:

$$\bar{m}_{\text{control}}(t) = \frac{1}{w} \sum_{i=t-w}^{t} m_{\text{control}}(i)$$

where $w$ is the rolling window (e.g., 10 minutes). The canary metric is compared:

$$\text{deviation}(t) = \frac{m_{\text{canary}}(t) - \bar{m}_{\text{control}}(t)}{\hat{\sigma}_{\text{control}}(t)}$$

Rollback if $|\text{deviation}(t)| > k$ for a threshold $k$ (e.g., $k = 3$, corresponding to a 3-sigma event). Using the control population's concurrent metric (rather than a historical baseline) automatically accounts for time-of-day effects, seasonal patterns, and external events.

**Multi-metric rollback logic:**

- Rollback if *any* critical metric (error rate, latency) crosses its threshold (AND logic for safety metrics).
- Rollback if a *combination* of degraded metrics exceeds a composite score (useful for noisy individual metrics).

### 3. Mathematical Formulation

#### 3.1 Expected Loss from Canary

Let $L$ be the per-request loss (in dollars) from a broken model, $R$ be the request rate (RPS), $p$ be the canary traffic fraction, and $T_d$ be the time to detect and rollback:

$$\text{Expected loss} = L \cdot R \cdot p \cdot T_d$$

Compare this to a full deployment:

$$\text{Expected loss (full)} = L \cdot R \cdot 1.0 \cdot T_d'$$

where $T_d'$ is the detection time without a canary (typically longer because there is no explicit comparison). The canary reduces expected loss by a factor of $p \cdot (T_d / T_d')$. For $p = 0.01$ and comparable detection times, loss is reduced by 100×.

#### 3.2 Detection Sensitivity

For a canary receiving $p$ fraction of traffic at rate $R$, the number of canary observations in time $t$ is $n = p \cdot R \cdot t$.

To detect a regression of size $\delta$ in a metric with variance $\sigma^2$:

$$t_{\text{detect}} = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot \sigma^2}{\delta^2 \cdot p \cdot R}$$

**Implication:** Smaller canary fractions require longer detection times. A 1% canary takes 10× longer to detect the same effect as a 10% canary. This creates a trade-off: smaller canaries limit blast radius but delay detection of subtle regressions.

### 4. Worked Examples

#### Example: Canary Rollout of a New Ranking Model

**Scenario.** Deploying a new search ranking model. Traffic: 10,000 RPS. Revenue per request: $0.02.

**Step 1 — Launch canary at 1%.** 100 RPS goes to the new model. Monitoring window: 30 minutes.

**Step 2 — Monitor.** After 30 minutes, 180,000 canary requests have been processed.
- Error rate: 0.01% (baseline: 0.01%) — healthy.
- P99 latency: 85 ms (baseline: 80 ms) — within tolerance (< 1.5× baseline).
- Revenue per request: $0.0198 (baseline: $0.0200) — within noise.

**Step 3 — Promote to 5%.** 500 RPS to canary. Monitor for 30 minutes (900,000 requests).
- All metrics healthy. Revenue per request: $0.0201 — slightly above baseline.

**Step 4 — Promote to 25%.** After another 30 minutes, no issues.

**Step 5 — Promote to 50%.** This effectively becomes an A/B test. Run for 2 days to measure statistical significance.

**Step 6 — Promote to 100%.** The new model is now fully deployed.

**Rollback scenario.** Suppose at 5%, the error rate spikes to 2.5%:
- The rollback trigger fires (threshold: error rate > 5× baseline).
- The orchestrator routes 100% of traffic back to the old model within 30 seconds.
- Total exposure: 2.5% of 500 RPS × 5 minutes (time to detect and act) = ~3,750 affected requests out of the ~1.5 million served during that window.

### 5. Relevance to Machine Learning Practice

**Canary deployments are essential for ML systems because ML models fail in unique ways:**

- **Silent failures.** The model returns a valid HTTP 200 response with a plausible-looking prediction that is completely wrong. Traditional software crashes; ML models fail silently. This is why business metrics (not just error rates) must be monitored during canary.

- **Distribution shift.** The model was trained on historical data but production traffic has shifted. A canary catches this if the shift is large enough to affect business metrics.

- **Feature pipeline bugs.** A feature used by the new model is computed incorrectly in production (but was correct in the offline evaluation). This is the most common source of train/serve skew.

**Integration with CI/CD:**

Modern ML deployment pipelines automate the full flow:

1. Model training completes → model artifact is stored in a model registry (MLflow, Weights & Biases, SageMaker Model Registry).
2. A CI/CD pipeline (GitHub Actions, Jenkins, Argo CD) deploys the model as a canary.
3. Automated monitoring checks metrics for a configured burn-in period.
4. If healthy, the pipeline automatically promotes traffic percentages.
5. If unhealthy, the pipeline automatically rolls back and alerts the on-call engineer.

Tools like Argo Rollouts, Flagger (for Kubernetes), and AWS CodeDeploy natively support canary strategies with custom metric checks.

### 6. Common Pitfalls & Misconceptions

1. **Monitoring only infrastructure metrics.** A canary that only checks latency and error rates will miss silent model quality regressions. Always include business metrics.

2. **Canary too small for too long.** A 0.1% canary needs an enormous amount of time to detect anything other than crashes. Use at least 1% and increase quickly if health checks pass.

3. **Ignoring canary population bias.** If the load balancer sends certain types of traffic to the canary (e.g., geographically clustered requests due to hashing), the canary population may not be representative. Use random assignment, not deterministic routing.

4. **No automated rollback.** A canary without automated rollback is just a deployment with fewer servers. The value of canary is in the automatic safety net.

5. **Conflating canary with A/B testing.** Canary answers "is it broken?" A/B testing answers "is it better?" A healthy canary does not prove the model is superior — it proves it is not catastrophically worse.

6. **Skipping canary for "minor changes."** Feature pipeline changes, library version bumps, and configuration changes frequently cause production incidents. Canary every change, no matter how small.

---

## Part VI — Interview Preparation

---

### Foundational Questions

**Q1: What is the difference between batch and real-time serving? When would you choose one over the other?**

Batch serving computes predictions for an entire dataset on a schedule (e.g., nightly). Real-time serving computes predictions per-request with low latency. Choose batch when freshness requirements are lax and throughput is the priority (e.g., daily recommendation emails). Choose real-time when predictions must reflect the user's current state (e.g., fraud detection at the moment of a transaction). Hybrid approaches are common: precompute base predictions in batch, then adjust in real-time based on session context.

**Q2: Explain the concept of a randomization unit in A/B testing. Why does it matter?**

The randomization unit is the entity (user, session, request, device) that is randomly assigned to control or treatment. It matters because it determines (a) consistency of user experience — if you randomize at the request level, a single user may see different variants on consecutive requests, and (b) statistical power — user-level randomization reduces the number of independent observations if each user generates many requests. The choice must match the metric: for per-user metrics (retention), randomize by user; for per-request metrics (latency), request-level randomization gives more power.

**Q3: What is model quantization? What are the trade-offs?**

Quantization reduces the numerical precision of a model's weights and activations (e.g., FP32 → INT8). Benefits: 2–4× smaller model, 2–4× faster inference, lower power consumption. Trade-offs: accuracy degradation (larger for aggressive quantization like INT4), potential changes in model behavior for edge cases, and the need for a calibration dataset to determine quantization parameters. Post-training quantization is fast but less accurate; quantization-aware training recovers accuracy but requires retraining.

**Q4: What is a canary deployment and how does it differ from a blue-green deployment?**

A canary deployment gradually shifts traffic from the old model to the new one (1% → 5% → 25% → 100%), monitoring at each step. A blue-green deployment maintains two complete environments (blue = current, green = new) and switches *all* traffic at once via a load balancer change. Canary is lower risk (small initial blast radius), while blue-green is simpler (no traffic splitting) but higher risk (100% exposure immediately). Canary is preferred for ML models because they fail silently and need gradual validation.

**Q5: What is the difference between structured and unstructured pruning?**

Unstructured pruning zeroes out individual weights based on magnitude (small weights → zero). It achieves high sparsity (e.g., 90% of weights removed) but the resulting sparse matrices only yield speedups on hardware that supports sparse operations. Structured pruning removes entire channels, filters, or attention heads, producing a smaller *dense* model that runs faster on any hardware. The trade-off: unstructured achieves higher compression ratios but requires specialized hardware; structured achieves lower compression ratios but yields immediate, hardware-universal speedups.

---

### Mathematical Questions

**Q6: Derive the minimum sample size for a two-sample proportion test in A/B testing.**

For proportions $p_A$ and $p_B$ with absolute difference $\delta = p_B - p_A$:

The pooled variance under $H_1$ is approximately $\sigma^2 \approx p_A(1 - p_A) + p_B(1 - p_B) \approx 2p(1-p)$ when $p_A \approx p_B \approx p$.

From the z-test framework, the test statistic under $H_1$ must exceed $z_{1-\alpha/2}$ with probability $1-\beta$:

$$z_{1-\alpha/2} - \frac{\delta}{\sqrt{2p(1-p)/n}} = -z_{1-\beta}$$

Solving:

$$n = \frac{2p(1-p)(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$$

For $p = 0.10$, $\delta = 0.01$, $\alpha = 0.05$, $\beta = 0.20$:

$$n = \frac{2 \times 0.10 \times 0.90 \times (1.96 + 0.84)^2}{0.01^2} = \frac{0.18 \times 7.84}{0.0001} = 14{,}112 \text{ per group}$$

**Q7: In pipeline parallelism, derive the bubble fraction. How can it be reduced?**

With $K$ stages and $M$ micro-batches, the pipeline takes $(M + K - 1)$ time steps total (it takes $K-1$ steps for the pipeline to fill and $K-1$ to drain). Total computation work is $M \times K$ (each of $M$ micro-batches passes through $K$ stages). But total available time slots are $K \times (M + K - 1)$ (each stage has $M + K - 1$ slots). Idle slots (bubble) are:

$$\text{Bubble} = K(M + K - 1) - MK = K(K-1)$$

$$\text{Bubble fraction} = \frac{K(K-1)}{K(M + K - 1)} = \frac{K-1}{M + K - 1}$$

Reduce by: (1) increasing $M$ (more micro-batches), (2) reducing $K$ (fewer stages), (3) interleaved scheduling (1F1B — one forward, one backward — to reduce memory while filling the pipeline faster).

**Q8: Explain the ring all-reduce communication cost formula and when it becomes a bottleneck.**

In ring all-reduce, $K$ nodes form a ring. Each node sends $(K-1)$ chunks of size $S/K$ in the reduce-scatter phase and $(K-1)$ chunks in the all-gather phase. Total data transferred per node: $2 \cdot \frac{K-1}{K} \cdot S$.

At bandwidth $B$: $T_{\text{comm}} = \frac{2(K-1)}{K} \cdot \frac{S}{B}$.

This becomes a bottleneck when $T_{\text{comm}}$ is comparable to or exceeds the computation time it enables. For tensor parallelism, each layer requires all-reduce on the activation tensor. If computation per layer (after splitting across $K$ GPUs) is $T_{\text{comp}} / K$, the communication overhead fraction is:

$$\text{Overhead} = \frac{T_{\text{comm}}}{T_{\text{comp}}/K} = \frac{2(K-1)S}{K \cdot T_{\text{comp}} \cdot B}$$

This is significant when $S$ is large (big activations), $B$ is small (slow interconnect), or $K$ is large (many GPUs). NVLink ($\sim$600 GB/s) keeps this manageable; Ethernet ($\sim$12.5 GB/s for 100 GbE) makes it prohibitive for all but the largest computations.

---

### Applied Questions

**Q9: You are deploying a real-time fraud detection model. Walk through your serving architecture design.**

I would design the following system:

**Model:** A gradient-boosted tree (XGBoost/LightGBM) for its low latency and interpretability, exported to ONNX for optimized inference.

**Feature serving:** A feature store (Feast + Redis) with two tiers: (a) precomputed user-level features (historical transaction count, average spend) updated hourly via batch pipeline, and (b) real-time features (transactions in the last 5 minutes) updated via a streaming pipeline (Kafka → Flink → Redis).

**Serving infrastructure:** FastAPI or gRPC server running ONNX Runtime. Deployed as a Kubernetes Deployment with HPA. Load balanced via Envoy. Minimum 5 replicas for redundancy.

**Latency budget:** Feature lookup (Redis): 2 ms. Preprocessing: 1 ms. Inference (XGBoost/ONNX): 3 ms. Network: 5 ms. Total: ~11 ms. SLA: P99 < 50 ms (comfortable headroom).

**Monitoring:** Canary deployment for model updates. Metrics: prediction distribution (drift detection), latency, error rate, fraud-flagged rate. Alert if fraud-flagged rate deviates by >2σ from historical baseline.

**Fallback:** If the model server is unavailable, fall back to a rule-based system (flag transactions >$5,000 or from known risky regions) to maintain fraud coverage during outages.

**Q10: Your team trained a transformer-based model that has better offline metrics than the current production model. How do you deploy it?**

1. **Export and optimize.** Export to ONNX or TorchScript. Apply FP16 quantization. Benchmark latency on the target hardware. If latency exceeds the SLA, consider distillation to a smaller model or dynamic batching to improve throughput.

2. **Canary deployment.** Deploy alongside the production model. Route 2% of traffic. Monitor for 24 hours: error rate, latency (P50, P95, P99), prediction distribution (JS divergence from production model), and business metrics.

3. **A/B test.** If canary is healthy, expand to 50/50 A/B test. Run for the duration required by the power analysis (typically 1–2 weeks for business metrics with reasonable effect sizes). Track primary metric (e.g., revenue) and guardrail metrics (latency, error rate).

4. **Decision.** If the primary metric is statistically significantly better and no guardrail metrics are violated, promote to 100%. If the primary metric is neutral but latency is worse, the higher infrastructure cost may not be justified — keep the existing model.

5. **Post-launch monitoring.** Continue monitoring for 2 weeks after full deployment. Watch for distribution shift (the model may perform well initially but degrade as traffic patterns change).

**Q11: A product team wants to A/B test 8 different recommendation algorithms simultaneously. What are the statistical concerns?**

Multiple testing is the primary concern. With 8 treatments and 1 control (8 comparisons), the familywise error rate under Bonferroni correction requires $\alpha_{\text{per-test}} = 0.05/8 = 0.00625$. This dramatically increases the required sample size — roughly 2× more per group compared to a single comparison.

Alternatives: (1) Use Benjamini-Hochberg FDR control (less conservative, controls the *rate* of false discoveries rather than *any* false discovery). (2) Use a multi-armed bandit approach (Thompson Sampling, UCB) that adaptively allocates more traffic to better-performing arms, converging faster than fixed-allocation A/B testing but providing weaker statistical guarantees. (3) Run in two phases: first, a screening phase with small traffic to eliminate clearly inferior variants; then, a powered A/B test comparing the top 2–3 candidates.

Also consider power: with 9 groups, each group gets 1/9 of traffic. If total traffic supports 50,000 users per group per week, that might be sufficient for large effects but underpowered for detecting small (2–3%) improvements. Recommend pre-computing the power for the expected MDE and adjusting the experiment duration accordingly.

---

### Debugging & Failure-Mode Questions

**Q12: You deployed a new model via canary. After 2 hours, the canary error rate is 0.5% vs. baseline 0.1%. But the model passed all offline tests. What do you investigate?**

Step 1: **Examine the errors.** Are they model errors (bad predictions) or infrastructure errors (timeouts, OOM)? If infrastructure → check container resources, model memory footprint. If model → proceed to step 2.

Step 2: **Check input distribution.** Are the canary errors concentrated on specific input types? Compare the distribution of inputs to the canary vs. control. The canary may be receiving a non-representative traffic slice (e.g., due to load balancer hashing).

Step 3: **Feature pipeline.** The most common culprit: a feature used by the new model is computed differently in production than in training. Check for feature skew by comparing production feature values against training feature distributions. Look for NaN, null, or out-of-range values.

Step 4: **Preprocessing version mismatch.** The offline evaluation used preprocessing code from the training pipeline. The serving pipeline may use a different version (e.g., a different tokenizer version, different normalization constants).

Step 5: **Model format conversion errors.** If the model was converted (e.g., PyTorch → ONNX), numerical differences from conversion can cause divergent predictions. Verify numerical equivalence on a batch of inputs: are the serving model's outputs close to the training model's outputs?

Step 6: **Data type mismatch.** The model expects FP32 inputs but the serving pipeline sends FP64 or INT32. Or categorical features are encoded with a different mapping.

**Q13: Your batch scoring pipeline, which ran fine for months, suddenly takes 3× longer. What do you investigate?**

Step 1: **Data volume.** Did the input dataset grow? A 3× increase in rows trivially explains a 3× increase in runtime.

Step 2: **Data skew.** Are partitions balanced? If one partition has 80% of the data (skew), the job's wall-clock time is dominated by that one partition, even if total data hasn't changed.

Step 3: **Resource contention.** Are other jobs competing for cluster resources? Check if the executors were given fewer cores/memory than usual due to resource contention.

Step 4: **Model change.** Was the model updated? A larger model (e.g., from a gradient-boosted tree to a neural network) takes longer per prediction.

Step 5: **Feature computation.** Has a feature computation become more expensive? A new feature that requires a database join or API call per row can dominate runtime.

Step 6: **Infrastructure.** Check for hardware degradation (slow disks, network throttling), spot instance preemption (causing restarts), or cloud provider issues.

**Q14: An A/B test shows the new model has +2% revenue but -5% user engagement. How do you decide whether to ship?**

This is a case of *metric trade-offs* and requires business judgment, not just statistics.

Step 1: **Verify both results are statistically significant.** If the engagement drop is not significant, it may be noise.

Step 2: **Understand the mechanism.** Why does revenue increase while engagement decreases? Possible explanation: the model shows higher-priced items, increasing revenue per transaction but reducing browsing and return visits. This is a short-term revenue gain at the expense of long-term retention.

Step 3: **Segment analysis.** Is the engagement drop concentrated in specific user segments (new users, power users)? A 5% engagement drop in new users is more alarming than in power users.

Step 4: **Long-term impact.** Run a holdback (keep 5% of users on the old model indefinitely) to measure long-term retention effects. Short A/B tests often miss delayed impacts.

Step 5: **Decision framework.** If the revenue gain is sustainable and the engagement drop does not predict churn (verify with historical data), shipping may be justified. If the engagement drop predicts long-term revenue loss that exceeds the short-term gain, do not ship. Present both the quantitative analysis and the risk assessment to leadership.

---

### Follow-Up and Probing Questions

**Q15: "You mentioned ONNX. How does ONNX Runtime optimize inference compared to running the model in PyTorch directly?"**

ONNX Runtime applies graph-level optimizations that PyTorch's eager mode cannot:

1. **Constant folding.** Pre-computes operations whose inputs are all constants (e.g., batch norm folding — merging batch normalization into the preceding convolution layer).
2. **Operator fusion.** Combines sequential operations into a single kernel (e.g., Conv + BatchNorm + ReLU → a single fused kernel, eliminating intermediate memory allocations).
3. **Memory planning.** Analyzes the graph to reuse memory buffers when tensors' lifetimes do not overlap, reducing peak memory usage.
4. **Execution providers.** Dispatches operations to the optimal hardware backend (CUDA, TensorRT, OpenVINO) based on the available hardware. TensorRT integration, for example, further optimizes convolutions and attention operations for NVIDIA GPUs.
5. **Thread and parallelism optimization.** Uses inter-op and intra-op parallelism tuned to the hardware.

PyTorch's eager mode executes operations one at a time with Python overhead for each. TorchScript and `torch.compile` (PyTorch 2.0+) narrow this gap by capturing and optimizing the graph, but ONNX Runtime's maturity and hardware-specific optimizations often retain a 20–50% latency advantage.

**Q16: "You set auto-scaling to target 70% CPU utilization. Why not 90%?"**

At 90% utilization, the system operates near its capacity limit. From queuing theory (M/M/1 model), average wait time grows non-linearly as utilization ($\rho$) approaches 1:

$$T_{\text{wait}} = \frac{\rho}{(1-\rho) \cdot \mu}$$

At $\rho = 0.7$: $T_{\text{wait}} = 0.7 / (0.3 \cdot \mu) = 2.33/\mu$.
At $\rho = 0.9$: $T_{\text{wait}} = 0.9 / (0.1 \cdot \mu) = 9.0/\mu$.

Going from 70% to 90% utilization increases average wait time by ~4×. Additionally, 70% leaves headroom for (a) traffic spikes that arrive faster than auto-scaling can react, (b) garbage collection pauses, and (c) variation in per-request inference time (some inputs are more expensive). The 70% target is a balance between cost efficiency (not wasting compute) and latency stability.

**Q17: "In your canary deployment, how do you handle features that are shared between the old and new model?"**

This is a critical detail. If both models share the same feature pipeline, the canary tests only the model itself. If the new model requires new features, the canary also tests the feature pipeline — which is where most production bugs live.

Recommended approach: deploy the new feature pipeline alongside the old one. The canary model reads from the new pipeline; the production model reads from the old one. Monitor feature values for consistency (compare shared features between pipelines — they should match). After the new model is fully deployed, decommission the old feature pipeline.

If running parallel feature pipelines is too expensive, an alternative is to shadow the new pipeline: compute features from both pipelines for every request, but only use the old pipeline's features for scoring. Log and compare the outputs. Once validated, switch to the new pipeline and deploy the new model.

**Q18: "You suggested sequential testing to avoid the peeking problem. Walk me through how an always-valid confidence sequence works."**

An always-valid confidence sequence (CS) provides a confidence interval for the treatment effect that is valid at every sample size simultaneously, not just at a pre-specified $n$.

The key idea is a *mixture* over all possible fixed-sample confidence intervals. For a running mean $\bar{X}_t$ based on $t$ observations with variance $\sigma^2$, the confidence sequence at level $1 - \alpha$ is:

$$\bar{X}_t \pm \sigma \sqrt{\frac{2(\log \log (2t) + 0.72 \log(5.2/\alpha))}{t}}$$

(This is one common form; exact formulations vary.) Notice that the width shrinks as $t$ grows (like $\sqrt{\log \log t / t}$) but more slowly than the fixed-sample width ($\sqrt{1/t}$). This extra width is the price for validity at every time point.

In practice: at each time step $t$, compute the CS for the treatment effect $\delta = \mu_B - \mu_A$. If the entire interval is above zero, conclude B > A and stop. If it is entirely below zero, conclude A > B and stop. If it contains zero, continue the experiment. Because the CS is valid at every $t$, the Type I error rate is controlled at $\alpha$ regardless of when you stop.

This is more powerful than Bonferroni-corrected repeated testing (which is extremely conservative) and more principled than Bayesian stopping rules (which depend on prior choices).

**Q19: "How do you monitor for silent model failures in production — cases where the model returns valid outputs that are wrong?"**

Silent failures are the hardest to detect because they produce no errors or exceptions. Strategies:

1. **Prediction distribution monitoring.** Track the distribution of model outputs over time (mean, variance, percentiles). If the fraud model suddenly predicts 0.01% fraud rate when the historical rate is 1%, something is wrong — even though each individual prediction is a valid probability.

2. **Feature distribution monitoring.** Track input feature distributions. If a feature's mean or variance shifts dramatically, the model may be receiving corrupted inputs. Compare against training-time distributions using KL divergence, Population Stability Index (PSI), or Kolmogorov-Smirnov tests.

3. **Ground truth delay monitoring.** For metrics with delayed ground truth (e.g., did the user actually churn?), set up a pipeline that joins predictions with eventual outcomes. Alert if the gap between predicted and actual exceeds a threshold.

4. **Ensemble disagreement.** Run a lightweight shadow model alongside the primary model. If the two models disagree on a large fraction of predictions, investigate. This is expensive but catches model-specific failures.

5. **Adversarial / edge-case probes.** Periodically send known-answer inputs (synthetic test cases with known correct outputs) through the production pipeline. If the model's output on these probes drifts, something in the pipeline has changed.

6. **Business metric correlation.** Correlate model predictions with downstream outcomes. If the recommendation model's predicted engagement score no longer correlates with actual clicks (measured with a lag), the model has degraded.

**Q20: "Your model must serve 50,000 RPS with P99 < 100 ms. Walk me through the capacity planning."**

Step 1: **Benchmark a single replica.** Deploy the model on the target hardware (e.g., c5.2xlarge, 8 vCPUs). Load test with realistic inputs. Measure: at what concurrency does P99 cross 100 ms? Suppose a single replica handles 800 RPS at P99 = 95 ms with 4 concurrent workers.

Step 2: **Compute replica count.** At 70% target utilization: effective capacity per replica = $800 \times 0.7 = 560$ RPS. Replicas needed: $\lceil 50{,}000 / 560 \rceil = 90$ replicas.

Step 3: **Add headroom.** For fault tolerance (2 availability zones, survive losing one): $90 \times 2 = 180$ replicas total, 90 per AZ. For traffic spikes (20% above peak): $90 \times 1.2 = 108$ per AZ.

Step 4: **Auto-scaling range.** Minimum replicas: 90 (handles baseline at 70% utilization). Maximum: 150 (handles 1.67× peak). Scale on P95 latency > 80 ms (proactive, before P99 breach).

Step 5: **Load balancer.** Use least-connections routing. Configure health checks with warm-up period (first 30 seconds after pod start, health check returns unhealthy to prevent routing before model is loaded).

Step 6: **Cost estimate.** 108 c5.2xlarge instances × $0.34/hr = $36.72/hr ≈ $26,438/month for one AZ. Consider spot instances for non-peak capacity (save 60–70%) with on-demand fallback.

Step 7: **Validate.** Run a shadow load test at 50,000 RPS against the staging environment. Verify P99 < 100 ms. Measure GPU/CPU utilization, memory, and network. Iterate if needed (upgrade instance type, optimize model, enable batching).
