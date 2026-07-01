# Machine Learning Deployment & Production Systems

## 1. Motivation & Intuition

### Why This Topic Matters

In machine learning, a model that lives only in a Jupyter Notebook is effectively useless. **Deployment** is the process of integrating a machine learning model into an existing production environment so it can make decisions based on new data.

Imagine you have built a model that can detect early signs of heart disease from X-rays with 99% accuracy.

- **Without Deployment:** The model sits on your laptop. A doctor has to email you an X-ray, you run the code, and email the result back. This is unscalable and slow.
- **With Deployment:** You embed the model into the hospital's database system. When an X-ray is taken, the model automatically scans it and alerts the doctor immediately if risks are found.

### The Core Problem: The "Lab vs. World" Gap

The environment where models are trained (research/lab) is fundamentally different from where they operate (production/world).

- **Lab:** Static data, infinite time to debug, single user, focus on accuracy.
- **Production:** Dynamic/streaming data, millisecond latency requirements, millions of users, focus on reliability and cost.

Deployment is the bridge that crosses this gap.

---

## 2. Conceptual Foundations

Deployment breaks down into three pillars: **Serving Patterns** (how predictions are delivered), **Infrastructure** (formats and scaling), and **Evaluation** (testing in production).

### A. Serving Patterns

#### 1. Batch Inference (Offline)

- **Definition:** The model makes predictions on a large set of data all at once, usually on a schedule (e.g., every night at 2 AM).
- **Analogy:** The postal service. You drop letters in a box all day; they are collected and processed in one big batch at night, then delivered the next day.
- **Mechanism:** A "Cron Job" triggers a script → Reads data from a Data Warehouse → Model predicts → Results saved back to a Database.
- **Use Case:** Churn prediction (identifying customers likely to quit this month), generating weekly music recommendations.

#### 2. Real-time Inference (Online)

- **Definition:** The model makes a prediction immediately upon request.
- **Analogy:** A phone call. You speak, and you expect an immediate response.
- **Mechanism:**
  - **REST API / gRPC:** The model is wrapped in a web server (like Flask or FastAPI). An app sends a JSON "request" (data), and the server sends back a JSON "response" (prediction).
  - **Streaming (Kafka/Kinesis):** The model "listens" to a continuous stream of data events, processes them as they arrive, and pushes results to another stream.
- **Use Case:** Fraud detection (blocking a credit card transaction instantly), Uber pricing (calculating fare before you book).

#### 3. Edge Deployment

- **Definition:** The model runs directly on the user's device (phone, IoT sensor) rather than on a central server.
- **Analogy:** A reflex. If you touch a hot stove, your spinal cord pulls your hand away instantly; it doesn't wait for your brain to process it.
- **Optimization:** Devices have limited battery and memory. We use:
  - **Quantization:** Reducing precision of numbers (e.g., from 32-bit floats to 8-bit integers) to save space with minimal accuracy loss.
  - **Pruning:** Removing "weak" connections in a neural network to make it smaller.
- **Use Case:** FaceID (unlocking phone), "Hey Siri" detection (privacy and speed).

### B. Model Formats

To move a model from Python (training) to C++ or Java (production), we need a standard file format.

- **Pickle:** Python's native format. *Pitfall:* Not secure (can execute malicious code) and Python-specific.
- **ONNX (Open Neural Network Exchange):** A universal format that allows you to train in PyTorch and run in C++, Java, or JavaScript.
- **TensorFlow SavedModel / TorchScript:** Framework-specific formats that package the model structure and weights for production engines.

### C. Scaling Strategies

When traffic increases (1 user → 1 million users), the system must scale.

- **Vertical Scaling (Scale Up):** Buying a bigger machine (more RAM, faster CPU). *Limit:* Even the biggest machine has a limit.
- **Horizontal Scaling (Scale Out):** Adding *more* machines (instances). We use a **Load Balancer** to distribute incoming requests evenly across these machines.
- **Model Partitioning (Sharding):** If the model itself is too big for one machine (e.g., GPT-4), we split the model across multiple GPUs (Layers 1-10 on GPU A, Layers 11-20 on GPU B).

### D. Safe Deployment & Testing

#### 1. A/B Testing

We define two groups: Control (A) sees the old model, Treatment (B) sees the new model. We compare metrics (e.g., Click-Through Rate) to see if B is statistically better.

#### 2. Canary Deployment

Instead of switching everyone to the new model at once, we roll it out to a small percentage (the "canary in the coal mine").

- Step 1: 1% of users see Model B.
- Step 2: If no errors, increase to 10%.
- Step 3: If safe, 100%.
- *Benefit:* If the model crashes, only 1% of users are affected.

---

## 3. Mathematical Formulation

While deployment is architectural, **Queueing Theory** and **Statistics** govern system performance and testing.

### A. Little's Law (System Throughput)

This law relates the number of requests in the system to latency and throughput.

$$
L = \lambda \times W
$$

Where:

- $L$ = Average number of items in the system (Queue Length + Processing).
- $\lambda$ (Lambda) = Average arrival rate (Throughput/Requests per second).
- $W$ = Average time an item spends in the system (Latency).

**Intuition:** If you want to handle more requests ($\lambda$) without increasing the wait time ($W$), you must increase the capacity of the system to hold more parallel requests ($L$) via scaling.

### B. A/B Testing Statistics

To decide if Model B is better than Model A, we use Hypothesis Testing.

#### 1. The Hypothesis

- Null Hypothesis ($H_0$): $p_A = p_B$ (The models perform the same).
- Alternative Hypothesis ($H_1$): $p_B > p_A$ (Model B is better).

#### 2. The Z-Statistic (for proportions like Click Rate)

$$
Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}
$$

Where:

- $\hat{p}_A, \hat{p}_B$: Conversion rates of groups A and B.
- $n_A, n_B$: Number of users in groups A and B.
- $\hat{p}$: Pooled conversion rate $= \frac{X_A + X_B}{n_A + n_B}$.

#### 3. Power Analysis (Sample Size)

Before starting, we calculate how many users ($n$) we need to detect a specific improvement (Effect Size $\delta$) with a certain confidence.

$$
n \propto \frac{\sigma^2}{\delta^2}
$$

**Intuition:** To detect a tiny improvement ($\delta$ is small), you need a massive amount of data ($n$ must be huge).

---

## 4. Worked Examples

### Example 1: End-to-End Deployment of a Credit Risk Model

**Scenario:** A bank wants to predict if a transaction is fraudulent.

**Phase 1: Batch (MVP)**

- **Workflow:** At 11:59 PM, the team dumps all daily transactions into a CSV.
- **Process:** A Python script loads the CSV, runs `model.predict()`, and flags suspicious accounts.
- **Outcome:** Analysts review the flags the next morning.
- **Limitation:** Fraud is detected 24 hours late. The money is already gone.

**Phase 2: Real-time (REST API)**

- **Workflow:** The model is saved as a `SavedModel` and wrapped in a FastAPI app.
- **Process:** When a user swipes a card, the card terminal sends a `POST` request to `https://api.bank.com/predict` with JSON: `{"amount": 500, "merchant": "Apple", "time": "12:00"}`.
- **Response:** The API returns `{"fraud_score": 0.95, "action": "decline"}` in 200ms.
- **Outcome:** Fraud prevented instantly.

**Phase 3: Canary Rollout**

- The team trains a new XGBoost model. They don't want to block legitimate cards by mistake.
- **Router Config:** 95% of traffic goes to `Old_Model`. 5% goes to `New_Model`.
- **Monitoring:** The team watches the "False Positive Rate" on the 5% traffic. It looks good.
- **Expansion:** Shift to 50/50, then 100% `New_Model`.

---

## 5. Relevance to Machine Learning Practice

### Where It Fits in the Lifecycle

Deployment is the "Day 2" of MLOps.

1. Data Prep
2. Training
3. **Deployment (Serving/Scaling)**
4. Monitoring (Drift detection)
5. Retraining

### Trade-offs

- **Latency vs. Accuracy:** A massive Transformer model might be 1% more accurate but take 2 seconds to infer. In high-frequency trading, a simpler Linear Regression that takes 1ms is preferred.
- **Batch vs. Real-time:** Real-time is expensive (servers running 24/7). Batch is cheap (servers run for 1 hour). Use Batch unless you absolutely need instant answers.
- **Edge vs. Cloud:** Edge offers privacy (data stays on device) and zero latency, but you cannot update the model easily (user must update the app).

---

## 6. Common Pitfalls & Misconceptions

**Training-Serving Skew:**
The data the model sees in production is processed differently than in training. For example, in training you normalized "Age" by dividing by 100. In production, the engineer forgot to divide by 100. The model sees "Age: 35" as a massive outlier (because it expects 0.35) and fails.

**The "Works on My Machine" Syndrome:**
Libraries (numpy, pandas) have different versions on your laptop vs. the server. The fix is **Docker Containers**. Containerization packages the code *and* the environment (OS, libraries) together so it runs exactly the same everywhere.

**Feedback Loops:**
The model changes the user behavior, which generates new training data, which reinforces the model's bias. For example, a recommendation system recommends "Clickbait". Users click it. The model learns "Users love clickbait" and recommends more. The quality of content degrades over time.

---

## 7. Interview Questions

### Foundational Questions

**Q: Explain the difference between Model Training and Model Inference.**

**Training** is the process of learning patterns from historical data (calculating weights). It is computationally intensive, runs offline, and takes hours/days. **Inference** is using the trained model to make predictions on new, unseen data. It must be fast (milliseconds), runs online or batch, and uses the fixed weights from training.

**Q: When would you choose Batch Serving over Real-time Serving?**

Choose **Batch** when:

1. Instant results are not required (e.g., email marketing lists).
2. You need to process massive volumes of data efficiently (high throughput is more important than low latency).
3. Compute cost is a concern (spot instances can be used).

Choose **Real-time** only when the prediction is needed to block or enable an immediate user action (e.g., fraud blocking, search autocomplete).

**Q: What is Serialization?**

Serialization is converting an object in memory (like a trained model class) into a byte stream (a file) that can be stored or transmitted. Deserialization is the reverse. Common formats are Pickle (Python specific) and ONNX (Interoperable).

### Mathematical & Technical Questions

**Q: In an A/B test, how does the sample size correlate with the Minimum Detectable Effect (MDE)?**

The sample size $n$ is inversely proportional to the square of the MDE ($\delta$).

$$
n \propto \frac{1}{\delta^2}
$$

This means if you want to detect an improvement that is **half** as big (e.g., 1% lift vs 2% lift), you need **four times** as much data. This is crucial for trade-offs: detecting tiny improvements requires expensive, long-running tests.

**Q: You are scaling a service. Your average latency is 100ms and you have 10 concurrent threads. According to Little's Law, what is your maximum theoretical throughput?**

Little's Law states $L = \lambda W$, so $\lambda = L / W$.

- $L$ (Capacity) = 10 requests.
- $W$ (Latency) = 0.1 seconds.
- $\lambda = 10 / 0.1 = 100$ requests per second (RPS).

To handle 200 RPS, you either need to halve the latency (50ms) or double the concurrency (20 threads).

### Applied Scenarios (System Design)

**Q: Design a deployment strategy for a "Smart Keyboard" on a mobile phone that predicts the next word.**

- **Constraint:** Zero latency required. Typing cannot lag. Internet connection is unreliable. Privacy is key (keystrokes shouldn't leave the phone).
- **Serving Pattern:** **Edge Deployment**. The model must sit on the device.
- **Model Format:** Quantized (int8) TensorFlow Lite or CoreML model to save memory/battery.
- **Update Strategy:** Federated Learning or periodic downloads of model weights when on Wi-Fi/Charging.
- **Key Metric:** Keystroke saving rate, Latency per inference (<10ms).

**Q: You deploy a model and the accuracy drops drastically compared to training. What do you check?**

This is **Training-Serving Skew**.

1. **Feature Consistency:** Are features computed exactly the same way? (e.g., is "time of day" UTC in training but local time in prod?)
2. **Data Drift:** Has the input distribution changed? (e.g., a new viral trend the model never saw).
3. **Missing Data:** Are fields null in production that were populated in training?
4. **Pipeline Bugs:** Check the preprocessing code path in the inference service.

### Debugging & Failure Modes

**Q: We deployed a new model using a Canary release (1% traffic). The server memory usage instantly spiked to 100% and crashed. Why?**

- **Memory Leak:** The model code might be creating new objects/tensors for every request without releasing them (e.g., appending to a global list).
- **Batch Size Mismatch:** The production load balancer might be sending requests faster than the model clears them, causing the internal queue to explode.
- **Model Size:** The new model might simply be too large for the RAM available on the instance (e.g., loading a 10GB model on an 8GB RAM server).
- **Concurrency:** Loading multiple copies of the model (one per worker) might exhaust shared memory.

**Q: What is the "Thundering Herd" problem in scaling?**

If a large number of scaling instances (autoscaling) all start up simultaneously and try to connect to the database or cache to load the model/data, they can crash the backend dependencies. The fix is exponential backoff (random delays before connecting) and slow-start strategies.

---

## Inference Optimization

Serving a model is rarely about running `model.predict()` unchanged. The same network can be made 2–10× cheaper and faster *before* you ever buy a bigger GPU. These optimizations trade a small, measurable accuracy loss for large latency/cost wins.

### Quantization
Store and compute weights/activations in lower precision than FP32.
* **FP16 / BF16:** Halve memory and roughly double throughput on modern accelerators with negligible accuracy loss. BF16 keeps FP32's exponent range (fewer overflow issues) at the cost of mantissa precision — the default for inference today.
* **INT8 (Post-Training Quantization, PTQ):** Map FP32 ranges to 8-bit integers using a scale factor: $x_{\text{int}} = \text{round}(x / s)$, where $s = \frac{\max|x|}{127}$. Requires a small **calibration set** to estimate activation ranges. Typically <1% accuracy drop, ~4× memory reduction.
* **Quantization-Aware Training (QAT):** Simulate quantization during training (fake-quant nodes) so the model *learns* to be robust to it. Recovers most of the accuracy PTQ loses, at the cost of a retraining cycle. Use when PTQ degrades a sensitive model below SLA.
* **Pitfall:** Quantizing **outlier-heavy** layers (e.g., attention logits, LayerNorm) can blow up error. Modern schemes keep these in higher precision (mixed precision) or use per-channel rather than per-tensor scales.

### Pruning & Distillation
* **Pruning:** Zero out weights with small magnitude. *Unstructured* pruning yields sparse matrices that need special kernels to actually speed up; *structured* pruning removes whole channels/heads and gives real wall-clock gains on commodity hardware.
* **Knowledge Distillation:** Train a small **student** to mimic a large **teacher's** soft output distribution (logits), not just hard labels. The soft targets carry "dark knowledge" (relative class similarities) that lets a 6-layer model recover most of a 24-layer model's quality. DistilBERT (~40% smaller, ~60% faster, ~97% of BERT's GLUE score) is the canonical example.

### Compilation & Kernel Fusion
Frameworks execute ops one-at-a-time with Python overhead. **Graph compilers** (TensorRT, ONNX Runtime, TVM, `torch.compile`) fuse adjacent ops (e.g., Conv→BatchNorm→ReLU into a single kernel), pick hardware-optimal kernels, and eliminate launch overhead. TensorRT additionally does precision calibration and layer auto-tuning for NVIDIA GPUs. This is usually the **highest ROI** optimization for GPU serving and requires no accuracy trade-off.

---

## Inference Serving Frameworks

A model file is not a service. Dedicated inference servers add the production machinery you'd otherwise rebuild by hand — request scheduling, batching, multi-model hosting, versioning, health/metrics endpoints, and GPU memory management.

### NVIDIA Triton Inference Server
The general-purpose standard for high-throughput serving. Its capabilities are worth understanding concretely:
* **Model repository & versioning:** models live in a versioned directory; Triton hot-loads/unloads versions and can serve several versions of one model simultaneously (enabling in-place canary/A-B at the server layer).
* **Concurrent model execution (instance groups):** you configure $N$ **instances** of a model per GPU. Triton runs them in parallel CUDA streams, so a small model that under-utilizes a GPU can host several copies to saturate it, while requests queue per-instance.
* **Backends:** one server runs TensorRT, ONNX Runtime, PyTorch, TensorFlow, and Python backends side by side, so a mixed fleet is one deployment.
* **Model ensembles / business logic scripting:** wire preprocess → model → postprocess (or a DAG of models) *inside* the server, avoiding network hops between steps.
* **Scheduler:** a per-model queue feeds the **dynamic batcher** (below); the **sequence batcher** additionally routes stateful requests from the same session to the same instance.

### LLM-Specific Servers (vLLM / TGI) and PagedAttention
Autoregressive LLM serving is dominated by one resource: the **KV-cache**. Its two pathologies are what these servers fix:
1. **Unknown output length** → you don't know how much cache a request needs, so classic serving *pre-allocates* for the max sequence length, wasting most of it.
2. **Fragmentation** → variable-length caches leave unusable gaps in GPU memory (internal + external fragmentation), so a naïve allocator wastes 60–80% of KV memory.

**PagedAttention** (the core of vLLM) borrows the operating-system **virtual-memory** idea. The KV-cache is split into fixed-size **blocks** (e.g., 16 tokens each). A per-sequence **block table** maps logical token positions to physically non-contiguous blocks, exactly like OS page tables:
* Memory is allocated **on demand**, one block at a time as generation proceeds — no pre-allocation waste; fragmentation drops to under one block per sequence.
* Blocks can be **shared** across sequences via copy-on-write — so parallel samples, beam search, or a shared system prompt reuse the same physical KV blocks until they diverge, cutting memory for these cases dramatically.

The freed memory translates directly into a **larger batch** (more sequences resident at once), and larger batches are what turn the memory-bandwidth-bound decode into higher throughput — the source of vLLM's oft-cited multi-× gains.

### Dynamic Batching
GPUs are throughput devices: processing 32 requests together costs barely more than processing 1, because weights are streamed from VRAM once and reused across the batch. A dynamic batcher holds incoming requests for a tiny window (e.g., 5 ms) to assemble a batch, trading a small latency increase for a large throughput gain. **The core tension:** larger batches → higher throughput but higher tail latency (both from the queue wait and the longer compute). The **max batch size** (cap compute so worst case fits the SLA) and **queue-wait timeout** (short enough that a lone request isn't starved under low load) are tuned directly against your p99.

### Continuous Batching (for autoregressive LLMs)
Static batching wastes the GPU when sequences in a batch finish at different times: a batch of 8 where 7 prompts want 10 tokens and 1 wants 500 leaves the GPU running a batch-of-1 for 490 steps. **Continuous (in-flight) batching** operates at the **iteration** level — after every single decode step it evicts finished sequences and admits waiting ones, keeping the running batch full. Combined with PagedAttention (which makes admitting a new sequence cheap), it is the main reason modern LLM servers reach 2–4× the throughput of static batching. The scheduler also separates the compute-bound **prefill** (processing the prompt) from the memory-bound **decode** to avoid a long prompt stalling everyone's token generation.

---

## Advanced Deployment Strategies

Beyond canary, three patterns govern how new model versions reach traffic safely:

### Shadow Deployment (Dark Launch)
The new model receives a **copy** of live production traffic but its predictions are **logged, never served**. You compare shadow vs. live offline.
* **Use when:** You need real-traffic validation with **zero user risk** — there is no way for a shadow mistake to reach a user.
* **Cost/limitation:** You pay for double inference compute, and you cannot measure **business metrics** that depend on the prediction actually being acted on (e.g., click-through), only prediction-level agreement and latency.

### Blue-Green Deployment
Maintain two complete, identical environments: **Blue** (current live) and **Green** (new version). Deploy and smoke-test on Green, then **flip the router** to send 100% of traffic to Green atomically.
* **Advantage:** Instant rollback — flip back to Blue if anything breaks. No partial-state weirdness.
* **Cost:** You run double the infrastructure during the cutover. Contrast with **canary**, which shifts traffic *gradually* (1% → 5% → 25% → 100%) on shared infra and catches issues at low blast radius but exposes some users to the new version immediately.

| Strategy | User risk | Infra cost | Rollback | Measures business metrics? |
| :--- | :--- | :--- | :--- | :--- |
| **Shadow** | None | 2× compute | N/A (not serving) | No (logging only) |
| **Canary** | Low (small %) | ~1× (shared) | Fast (stop ramp) | Yes |
| **Blue-Green** | All-or-nothing | 2× during cutover | Instant (flip) | Yes |

---

## Model Monitoring & Observability

A model that passed offline eval can silently rot in production because the *world* changes. Monitoring spans three layers:

### 1. Operational (System) Health
Standard service SRE: request rate (throughput), error rate, and **latency percentiles** (track p50/p95/**p99**, never just the mean — tail latency drives user experience and timeout cascades). Plus resource saturation: GPU utilization, memory, queue depth.

### 2. Data & Feature Drift
The input distribution at serving time diverges from training. Detect *without labels* by comparing distributions:
* **Population Stability Index (PSI):** bin a feature, compare serving vs. reference proportions.
$$\text{PSI} = \sum_{i=1}^{B} (p_i^{\text{serve}} - p_i^{\text{train}}) \cdot \ln\!\left(\frac{p_i^{\text{serve}}}{p_i^{\text{train}}}\right)$$
Rule of thumb: PSI < 0.1 stable, 0.1–0.25 moderate drift, > 0.25 significant — investigate. Also used: KL divergence, KS test, and embedding-distribution distances for unstructured inputs.

### 3. Concept Drift & Model Quality
The relationship $P(y \mid x)$ itself changes (e.g., fraud patterns evolve, fashion preferences shift). This is the dangerous one because inputs can look perfectly in-distribution while accuracy collapses.
* **The label-delay problem:** Ground truth often arrives hours-to-months later (did the user repay the loan? was the transaction actually fraud?). You usually *cannot* compute accuracy in real time.
* **Proxies while waiting for labels:** monitor **prediction drift** (distribution of model outputs over time), confidence/score distributions, and downstream business KPIs (conversion, charge-back rate). A sudden shift in the *output* distribution is an early alarm even before labels land.

### Alerting & Response
Wire drift and quality breaches to alerts that trigger a defined response: auto-rollback to the previous version, route to a fallback heuristic, or kick off a retraining pipeline. **Closing this loop** (monitor → detect → retrain → redeploy) is what makes a deployment a *continuously trained* system rather than a static artifact.

---

## Continuous Delivery for ML (CD4ML)

ML deployment has **three** versioned artifacts where software has one: **code**, **data**, and **model**. Reproducibility requires versioning all three together (e.g., Git for code, DVC/lakeFS for data, a **model registry** like MLflow for models with lineage linking each model back to the exact data + code that produced it).

A mature pipeline automates: data validation → training → offline evaluation gate → registration → staged rollout (shadow/canary) → monitoring → (drift-triggered) retraining. The **evaluation gate** is critical: a model is only promoted if it beats the incumbent on held-out metrics *and* passes fairness/slice checks — preventing a numerically-better-but-broken model from auto-deploying.

---

## Additional Interview Questions

**Q: A model passes offline evaluation but its INT8-quantized production version shows a 4% accuracy drop. What's likely happening and how do you fix it without abandoning quantization?**

Post-training quantization clips activation outliers; layers with heavy-tailed activations (attention, certain final layers) lose the most. Fixes, in order of effort: (1) use **per-channel** instead of per-tensor scales; (2) keep the sensitive layers in **FP16 (mixed precision)** while quantizing the rest; (3) improve the **calibration set** to better represent production ranges; (4) if still short, switch to **quantization-aware training** so the model learns robustness to the rounding.

**Q: When would you choose shadow deployment over a canary?**

Shadow when the risk of *any* user seeing a bad prediction is unacceptable (e.g., medical, credit decisions) and you only need to validate prediction quality/latency against real traffic. Its limitation is that it can't measure metrics requiring the prediction to be *acted on* (CTR, conversion) — for those you need canary or blue-green, which actually serve the new version to some users.

**Q: Your input feature distributions look stable (low PSI everywhere) but model accuracy is dropping. What kind of drift is this and why is it hard?**

This is **concept drift**: $P(y\mid x)$ changed while $P(x)$ stayed the same, so distribution-based input monitors stay silent. It's hard because you can only catch it via **labels**, which are typically delayed. Mitigate with proxy signals (prediction-distribution drift, business KPIs) for early warning and a scheduled/triggered retraining cadence to keep the model current.

**Q: Why does dynamic batching improve throughput but risk violating a latency SLA, and how do you tune it?**

GPUs amortize fixed per-call overhead across a batch, so throughput rises with batch size — but assembling a batch means **waiting** for requests, and larger batches take longer to compute, both inflating tail latency. Tune the **max batch size** and **queue-wait timeout** against your p99 budget: cap batch size so worst-case compute fits the SLA, and set the timeout short enough that under low load a request isn't starved waiting for batch-mates that never arrive.
