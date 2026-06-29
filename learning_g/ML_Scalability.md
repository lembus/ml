# Scalability & Systems in Machine Learning

---

## Module 1: Distributed Training

### Data Parallelism, Model Parallelism, and Mixed Precision

### 1. Motivation & Intuition

**Why do we need this?**

Imagine you are trying to read every book in a massive library to prepare for an exam.

1. **Single Worker:** If you do it alone, it might take 100 years. This is training a massive model on a single GPU — it takes too long.
2. **Data Parallelism (The Team Approach):** You hire 10 friends. You photocopy the library. Each friend reads a different set of 100 books simultaneously. At the end of the day, you all meet and share what you learned. This speeds up the process by roughly 10x.
3. **Model Parallelism (The Specialist Approach):** Suppose one book is actually a scroll 10 miles long — too heavy for one person to hold. You need three friends just to hold and read *one* book together (one holds the start, one the middle, one the end). This is necessary when a model is too large to fit into the memory of a single GPU.

**Real-World Context:**
Modern Large Language Models (LLMs) like GPT-4 have trillions of parameters. A single H100 GPU has 80GB of memory. A trillion-parameter model requires terabytes just to store the weights, let alone the gradients and optimizer states. We *must* distribute the workload to train these models at all.

### 2. Conceptual Foundations

#### A. Data Parallelism (DP)

- **Replica:** The model is copied (replicated) onto every GPU.
- **Partitioning:** The dataset is split into $N$ chunks (where $N$ is the number of GPUs).
- **Forward/Backward Pass:** Each GPU processes its chunk of data independently and calculates gradients.
- **Synchronization:** Before updating the model weights, GPUs must communicate to average their gradients. This ensures every model replica stays identical.
  - **Synchronous (All-Reduce):** Everyone waits until *all* GPUs finish their batch. Gradients are averaged, weights updated, and then the next step begins. Stable but limited by the slowest GPU (straggler).
  - **Asynchronous (Parameter Server):** GPUs push gradients to a central server and pull updated weights whenever they are ready, without waiting for others. Faster, but mathematically unstable (stale gradients).

#### B. Model Parallelism (MP)

Used when the model fits on no single GPU.

- **Pipeline Parallelism:** Splits the model by layers (e.g., GPU 1 holds layers 1–10, GPU 2 holds 11–20). Data flows through GPU 1, then GPU 2.
  - *Issue:* "Bubbles." GPU 2 sits idle while GPU 1 works on the first batch.
- **Tensor Parallelism:** Splits the individual matrices within a layer. If you have a massive matrix multiplication $A \times B$, you split $A$ and $B$ across GPUs, compute parts of the product, and combine the results.

#### C. Mixed Precision

Deep learning usually uses 32-bit floating point (FP32). Mixed precision uses 16-bit formats (FP16 or BF16) for matrix multiplications and FP32 for weight updates.

- **Benefit:** Reduces memory usage by ~50% and speeds up math on Tensor Cores.
- **Risk:** Underflow (numbers become too small to represent in 16 bits).

### 3. Mathematical Formulation

#### Synchronous Data Parallelism Update

Let $\theta_t$ be the model weights at step $t$. We have $K$ GPUs. Each GPU $k$ computes the gradient $g_k$ on a mini-batch $B_k$:

$$
g_k(\theta_t) = \frac{1}{|B_k|} \sum_{(x,y) \in B_k} \nabla_\theta \mathcal{L}(f(x; \theta_t), y)
$$

The global gradient is the average of all local gradients:

$$
\Delta \theta_{global} = \frac{1}{K} \sum_{k=1}^K g_k(\theta_t)
$$

The update rule (SGD) is:

$$
\theta_{t+1} = \theta_t - \eta \cdot \Delta \theta_{global}
$$

*Note:* The communication overhead comes from computing the sum $\sum g_k$.

#### Mixed Precision & Loss Scaling

In FP16, the smallest representable positive number is $\approx 6 \times 10^{-8}$. Gradients in deep networks often fall below this (e.g., $10^{-9}$), becoming zero (underflow).

**Solution:** Multiply the loss by a scale factor $S$ (e.g., 1024) before backpropagation.

1. Forward pass: Compute Loss $L$.
2. Scale: $L_{scaled} = L \times S$.
3. Backward: Compute gradients $\nabla L_{scaled} = S \cdot \nabla L$. Since gradients are now larger, they stay within FP16 range.
4. Unscale: Divide gradients by $S$ (in FP32) before the optimizer update to correct the magnitude.

### 4. Worked Example: Distributed Training Setup

**Scenario:** Training a BERT-Large model (340M params) on 4 GPUs using Synchronous Data Parallelism.

1. **Initialization:** The master process broadcasts the initial weights $\theta_0$ to GPU 0, 1, 2, 3.
2. **Step 1 (Forward):** Batch size = 32. Each GPU takes 8 samples. GPU 0 calculates loss for samples 0–7. GPU 1 for 8–15, etc.
3. **Step 2 (Backward):** Each GPU computes gradients for its 8 samples.
4. **Step 3 (All-Reduce):** System uses **Ring All-Reduce**. GPU 0 sends its gradients to GPU 1, GPU 1 to 2, etc., in a ring. After the ring cycle, every GPU has the sum of all gradients.
5. **Step 4 (Update):** Each GPU divides the summed gradients by 4 (average). Each GPU updates its local copy of weights. They are now identical again.

### 5. Relevance to ML Practice

- **When to use DP:** Almost always. Even for small models, it speeds up training linearly with GPU count (up to a point).
- **When to use MP:** Only when the model (weights + optimizer states + gradients) exceeds a single GPU's VRAM.
- **Standard Stack:** PyTorch Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP).

### 6. Common Pitfalls

**The Straggler Problem:** In synchronous DP, if GPU 3 is slow (overheating, network lag), GPUs 0, 1, and 2 sit idle waiting for it. Fix: Check hardware health; use asynchronous methods (rare nowadays) or outlier mitigation.

**Silent Accuracy Drop in Mixed Precision:** If you forget **Loss Scaling**, gradients underflow to zero. The model "trains" (runs without error), but the loss never decreases because the updates are effectively zero.

---

## Module 2: Distributed Inference & Optimization

### Serving, Caching, and Resource Management

### 1. Motivation & Intuition

Training is about throughput (images per second). Inference is about **latency** (how fast does the user get a reply?).

If you put a raw PyTorch model behind a web server (Flask), it will be slow. It processes one request at a time, re-interpreting Python code for every call.

**Analogy:** A raw model server is like a chef cooking one burger at a time from scratch.

**Optimized Serving:**

1. **Batching:** The chef waits 50ms to see if 3 more orders come in, then cooks 4 patties on the grill at once.
2. **Compilation (TensorRT):** The chef pre-chops all ingredients and organizes the kitchen so no movement is wasted (optimizing the computation graph).
3. **Quantization:** Using smaller plates (int8) to move food faster (lower memory bandwidth).

### 2. Conceptual Foundations

#### A. Model Serving Optimizations

- **Dynamic Batching:** The server waits a short window (e.g., 2ms) to group incoming requests into a single tensor batch. GPU efficiency increases significantly with batch size.
- **Model Compilation:** Tools like TVM or TensorRT analyze the model graph. They fuse layers (e.g., Conv2D + ReLU becomes one operation) to reduce memory access overhead.

#### B. Caching

- **Feature Caching:** Storing computed features for frequently accessed entities (e.g., user embedding for a recommendation system) in Redis/Memcached so you don't re-compute them.
- **Prediction Caching:** If the input is identical to a previous request (common in search queries), serve the cached result.

#### C. Resource Management (Quantization)

- **Quantization:** Converting weights from FP32 (32 bits) to INT8 (8 bits).
  - *Post-Training Quantization (PTQ):* Train in FP32, convert to INT8 later.
  - *Quantization Aware Training (QAT):* Simulate INT8 noise during training so the network adapts to the precision loss.

### 3. Mathematical Formulation: Quantization

Mapping a real value $x$ (FP32) to an integer $q$ (INT8):

$$
q = \text{round}\left( \frac{x}{S} + Z \right)
$$

Where:

- $S$ (Scale) $= \frac{x_{max} - x_{min}}{2^b - 1}$
- $Z$ (Zero-point) maps the real value 0 to an integer (ensuring zero padding works).
- $b$ is the number of bits (e.g., 8).

**Re-quantization:** When multiplying two INT8 matrices, the result is INT32. We must rescale it back to INT8 for the next layer.

### 4. Worked Example: Optimizing a ResNet-50

1. **Baseline:** PyTorch model in Flask. Latency = 50ms. Throughput = 20 req/s.
2. **Step 1 (ONNX/TensorRT):** Export model to ONNX. Compile with TensorRT. Layer fusion reduces kernel launches. Latency drops to 15ms.
3. **Step 2 (Quantization):** Calibrate with a small dataset to find min/max ranges for activations. Convert to INT8. Model size drops 4x. Latency drops to 8ms.
4. **Step 3 (Dynamic Batching):** Use Triton Inference Server. Configure `max_batch_size=16` and `max_queue_delay_microseconds=2000`. Throughput increases to 400 req/s because the GPU is now fully utilized, though single-request latency rises slightly to 10ms (due to the wait).

### 5. Relevance to ML Practice

- **Cost vs. Latency:** Real-time apps (self-driving cars) prioritize latency (batch size 1). Offline jobs (tagging photos) prioritize throughput (large batches).
- **Spot Instances:** Use cheap, interruptible AWS Spot instances for offline batch inference. Implement checkpointing to resume if the instance is killed.

### 6. Common Pitfalls

**Cache Thrashing:** If you cache too much, the cache fills up, and you spend more time evicting items than serving them.

**Distribution Shift in Quantization:** If you calibrate your INT8 model on "daytime" images but serve "nighttime" images, the fixed range $x_{min}, x_{max}$ might clip the new data, destroying accuracy.

---

## Module 3: Privacy & Security in ML

### 1. Motivation & Intuition

Models memorize training data. If you train a text predictor on private emails, a user might type "My social security number is..." and the model might autocomplete a real person's SSN from the training set.

- **Membership Inference:** An attacker can check if a specific person was in the training set (e.g., checking if a patient was in a "cancer positive" dataset).

### 2. Conceptual Foundations

#### A. Differential Privacy (DP)

A mathematical guarantee that the output of an algorithm is essentially the same whether any *single* individual's data is included or not.

- **Mechanism:** Add random noise (Laplace or Gaussian) to the gradients during training (DP-SGD).
- **Privacy Budget ($\epsilon$):** Measures how much privacy is lost. Lower $\epsilon$ = more privacy = less utility (accuracy).

#### B. Federated Learning (FL)

Instead of sending data to the server, send the *model* to the device.

1. Server sends global model to smartphones.
2. Phone trains locally on user data.
3. Phone sends *only* the weight updates (gradients) back to server.
4. Server averages updates.

*Note:* Gradients can still leak information, so FL is often combined with DP.

### 3. Mathematical Formulation: Differential Privacy

A randomized algorithm $\mathcal{M}$ is $(\epsilon)$-differentially private if for all adjacent datasets $D$ and $D'$ (differing by one person):

$$
P(\mathcal{M}(D) \in S) \leq e^\epsilon \cdot P(\mathcal{M}(D') \in S)
$$

If $\epsilon$ is small (e.g., 0.1), $e^\epsilon \approx 1$. The probabilities are nearly identical. An attacker cannot distinguish if the data came from $D$ or $D'$.

### 4. Relevance to ML Practice

- **Federated Learning:** Used by Apple/Google for predictive keyboards.
- **Trade-offs:** DP significantly hurts model accuracy. It requires training for more epochs and larger batch sizes to converge.

---

## Interview Questions

### Foundational Questions

**Q: Explain the difference between Model Parallelism and Data Parallelism. When would you use one over the other?**

Data Parallelism replicates the model on all devices and splits the data. It is used when the model fits in one device's memory. Model Parallelism splits the model itself across devices. It is used when the model parameters + gradients + optimizer states exceed a single device's memory.

**Q: What is the "straggler problem" in distributed training?**

In synchronous training, the system moves at the speed of the slowest device. If one GPU is slow (straggler), all others wait, reducing cluster utilization.

**Q: Why does mixed precision training (FP16) require loss scaling?**

Gradients in deep networks are often very small. In FP16, values smaller than $\approx 6 \times 10^{-8}$ underflow to zero. Loss scaling multiplies the loss by a constant factor to shift gradients into the representable range of FP16, preventing them from vanishing.

### Mathematical & Technical Questions

**Q: In Ring All-Reduce, what is the communication cost for $N$ GPUs and model size $M$?**

The bandwidth cost is roughly $2M$ per GPU, independent of $N$ (for large $N$). It involves a scatter-reduce pass and an all-gather pass.

**Q: Derive the memory savings of using FP16 vs FP32 for a model with $P$ parameters using Adam optimizer.**

- **FP32:** Parameters (4 bytes) + Gradients (4 bytes) + Adam States (Momentum + Variance, $4+4=8$ bytes) = **16 bytes/param**.
- **Mixed Precision:** Parameters (2 bytes) + Gradients (2 bytes) + Master Weights (FP32, 4 bytes) + Adam States (FP32, 8 bytes) = **16 bytes/param**.
- *Correction:* Mixed Precision usually *retains* a master copy in FP32. The savings come from the activations (stored in FP16 for backprop) and the actual compute speed. The *static* memory consumption for weights/optimizer is similar, but **activation memory** (often the bottleneck) is halved.

**Q: How does Differential Privacy's $\epsilon$ parameter relate to noise?**

$\epsilon$ is inversely proportional to the noise scale. To achieve lower $\epsilon$ (higher privacy), you must add noise with a larger standard deviation (e.g., larger $\sigma$ in Gaussian mechanism), which degrades model accuracy.

### Applied & Design Questions

**Q: Design a serving system for a high-traffic news recommendation engine. Latency must be <100ms.**

- **Architecture:** Two-tower model (User tower, Item tower).
- **Offline:** Pre-compute Item embeddings and index them in an ANN (Approximate Nearest Neighbor) vector database (FAISS/ScaNN).
- **Online:** Compute User embedding on the fly.
- **Caching:** Cache the User embedding in Redis (TTL 10 mins).
- **Scaling:** Use Horizontal Pod Autoscaling (HPA) based on CPU/GPU utilization.
- **Optimization:** Use INT8 quantization for the User tower to speed up inference.

**Q: You notice your distributed training loss is oscillating and not converging. What do you check?**

- **Effective Batch Size:** In Data Parallelism, Global Batch Size = Local Batch Size $\times$ Number of GPUs. If you scaled up GPUs but didn't scale the Learning Rate, the step size is too small relative to the batch noise reduction. Rule of thumb: Linear Scaling Rule (scale LR with Batch Size).
- **Sync Issues:** Ensure gradients are actually being synchronized (check if replicas are diverging).
- **Data Shuffling:** Ensure data is shuffled differently on each GPU (seeds must be different).

### Debugging & Failure Modes

**Q: We switched from FP32 to FP16 and the model's accuracy dropped to random chance immediately. Why?**

Likely **Loss Scaling** was not implemented or the initial scale factor was too high (causing overflow/NaNs) or too low (causing underflow). If the first few gradients are NaN, the weights become NaN forever.

**Q: In a pipeline parallel setup, GPU utilization is only 40%. Why?**

The "Pipeline Bubble." While GPU 4 is processing the last layer of batch 1, GPU 1 is already done and waiting for the backward pass of batch 1. Fix: Use Micro-batching (split the mini-batch into smaller chunks and pipeline those) to fill the bubbles.
