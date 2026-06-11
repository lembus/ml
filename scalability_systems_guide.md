# Scalability & Systems in Machine Learning

## A Comprehensive Interview Preparation Guide

---

# Part I: Distributed Training

---

## 1. Motivation & Intuition

### Why Distributed Training Exists

Imagine you are training a neural network on a dataset of 1 billion images. A single GPU has 80 GB of memory and can process roughly 200 images per second at full precision. At that rate, a single pass through the dataset takes about 58 days. Modern models require hundreds of passes. Training on one GPU is simply not feasible.

Distributed training exists to solve three fundamental bottlenecks:

**Bottleneck 1: Training time.** If one GPU processes 200 samples/second, eight GPUs should give you roughly 1,600 samples/second, reducing weeks to days. This is the throughput problem.

**Bottleneck 2: Model size.** GPT-3 has 175 billion parameters. Storing parameters in FP32 requires 4 bytes each, so the model alone requires 700 GB — far exceeding any single GPU's memory. The model literally does not fit. This is the capacity problem.

**Bottleneck 3: Data size.** Some datasets are so large they cannot fit on a single machine's storage. Distributing data across machines is a prerequisite for even beginning training. This is the storage problem.

Distributed training is the collection of techniques that let you split the work — the data, the model, or both — across multiple devices (GPUs, TPUs, or entire machines) and coordinate them so the result is mathematically equivalent (or very close) to training on a single giant device.

### A Concrete Analogy

Think of grading 10,000 exam papers. One instructor takes 100 hours. Ten instructors each grading 1,000 papers take 10 hours, but they must agree on a grading rubric and periodically synchronize their standards so grades are consistent. This is data parallelism — same rubric (model), different papers (data).

Now imagine a single exam paper that is so complex it requires three experts: one for the math section, one for the essay, and one for the code. Each expert grades their section and passes the paper to the next. This is model parallelism — different parts of the model, same data flowing through in sequence.

---

## 2. Conceptual Foundations

### 2.1 Data Parallelism

#### Core Idea

Every GPU holds a complete copy of the model. The training data is split into chunks, and each GPU processes a different chunk simultaneously. After computing gradients on their local chunk, the GPUs communicate to combine their gradients, and every GPU updates its model copy identically.

#### Step-by-Step Process

1. **Initialization**: Copy the identical model to all $N$ GPUs. Ensure all parameters are identical.
2. **Data partitioning**: Divide the current mini-batch of size $B$ into $N$ micro-batches, each of size $B/N$. GPU $i$ receives micro-batch $i$.
3. **Forward pass (local)**: Each GPU runs its micro-batch through its local model copy, computing predictions and loss.
4. **Backward pass (local)**: Each GPU computes gradients of the loss with respect to all parameters, using only its local micro-batch.
5. **Gradient synchronization**: All GPUs communicate to compute the average gradient across all $N$ GPUs.
6. **Parameter update**: Each GPU applies the averaged gradient to its local model copy using the optimizer (e.g., SGD, Adam).
7. **Repeat** from step 2.

Because every GPU starts with the same parameters and applies the same averaged gradient, all model copies remain identical throughout training. This invariant is critical.

#### Synchronous vs. Asynchronous Updates

**Synchronous (All-Reduce)**

In synchronous data parallelism, all GPUs must finish their gradient computation before any GPU can proceed to the update step. The gradient averaging is performed using an operation called **All-Reduce**.

All-Reduce is a collective communication primitive: every GPU contributes a tensor, the tensors are summed (or averaged), and the result is distributed back to every GPU. After All-Reduce, every GPU has an identical copy of the averaged gradient.

*How All-Reduce works (Ring All-Reduce):*

Suppose you have 4 GPUs arranged in a logical ring: GPU0 → GPU1 → GPU2 → GPU3 → GPU0. Each GPU has a gradient vector $g_i$. The algorithm proceeds in two phases:

Phase 1 — Reduce-Scatter: The gradient vector on each GPU is divided into 4 chunks. Over 3 rounds (N-1 rounds), each GPU sends one chunk to its right neighbor and receives one chunk from its left neighbor, accumulating (summing) the received chunk with its own. After this phase, each GPU holds the complete sum for exactly one chunk.

Phase 2 — All-Gather: Over another 3 rounds, each GPU sends its fully reduced chunk to the right and receives a fully reduced chunk from the left. After this phase, every GPU holds all 4 fully reduced chunks — i.e., the complete averaged gradient.

The key insight is that ring all-reduce requires each GPU to send and receive exactly $2(N-1)/N$ times the gradient size, regardless of $N$. The communication cost grows logarithmically with the gradient size per GPU, not with the number of GPUs. This makes it highly scalable.

*Advantages of synchronous training:*
- Mathematically equivalent to single-GPU training with the global batch size $B$.
- Deterministic and reproducible (given fixed data ordering).
- Easy to reason about convergence.

*Disadvantages:*
- **Straggler problem**: The slowest GPU determines the step time. If one GPU is 10% slower (due to thermal throttling, network congestion, or hardware variability), all GPUs wait.
- **Synchronization barrier**: All GPUs idle after finishing their local computation until the slowest GPU completes.

**Asynchronous Updates**

In asynchronous data parallelism, GPUs do not wait for each other. Typically, a central **parameter server** holds the master copy of the model. Each GPU:

1. Pulls the current parameters from the server.
2. Computes gradients on its local data.
3. Pushes gradients to the server immediately.
4. The server applies the gradient update right away, without waiting for other GPUs.

*The Staleness Problem:*

Suppose GPU A pulls parameters at step $t$, but by the time GPU A pushes its gradient, the server has already received and applied updates from GPUs B, C, and D, moving the parameters to step $t+3$. GPU A's gradient was computed with respect to step-$t$ parameters but is now applied to step-$(t+3)$ parameters. This gradient is "stale" by 3 steps.

Stale gradients point in slightly wrong directions. The effect is similar to adding noise to the optimization. Mild staleness can be tolerated (and may even act as a regularizer), but severe staleness causes divergence or convergence to worse optima.

*Mitigation strategies:*
- **Bounded staleness**: Allow at most $\tau$ steps of staleness. If a worker falls behind by more than $\tau$ steps, it blocks until the gap closes.
- **Gradient compensation**: Scale down stale gradients or apply Taylor-expansion corrections.
- **Learning rate reduction**: Lower the learning rate to dampen the effect of stale updates.

*Advantages of asynchronous training:*
- No synchronization barrier; faster wall-clock time per step.
- Better hardware utilization (no idle waiting).
- Natural fault tolerance (if one GPU fails, others continue).

*Disadvantages:*
- Stale gradients degrade convergence.
- Non-deterministic training; harder to reproduce results.
- Effective learning rate is noisy, which may require careful tuning.
- In practice, synchronous All-Reduce has largely won for dense models due to better convergence and simpler tuning. Asynchronous methods are still used for very large-scale sparse models (e.g., recommendation systems with embedding tables).

### 2.2 Model Parallelism

When the model itself is too large to fit on a single GPU, you must split the model across GPUs. There are two primary strategies.

#### Pipeline Parallelism

**Core idea**: Split the model into sequential stages. Stage 1 (e.g., layers 1–10) lives on GPU 0, stage 2 (layers 11–20) on GPU 1, and so on. Data flows through the pipeline: GPU 0 computes the output of stage 1 and sends it to GPU 1, which computes stage 2 and sends it to GPU 2, etc.

**The Bubble Problem**

Naive pipelining has a severe efficiency problem. Consider 4 stages and one micro-batch:

```
Time →
GPU 0: [F1][ idle ][ idle ][ idle ][B1][ idle ][ idle ][ idle ]
GPU 1: [   ][F2   ][ idle ][ idle ][   ][B2   ][ idle ][ idle ]
GPU 2: [   ][     ][F3   ][ idle ][   ][     ][B3   ][ idle ]
GPU 3: [   ][     ][     ][F4   ][   ][     ][     ][B4   ]
```

F = forward pass, B = backward pass. Each GPU is idle most of the time. The fraction of idle time is called the **pipeline bubble**.

**Micro-batching (GPipe approach)**: To reduce the bubble, split the mini-batch into $M$ micro-batches and inject them into the pipeline one after another. While GPU 0 starts the forward pass on micro-batch 2, GPU 1 is processing micro-batch 1. The pipeline fills up, and idle time shrinks.

The bubble fraction for GPipe is approximately $(P-1) / M$, where $P$ is the number of pipeline stages and $M$ is the number of micro-batches. With 4 stages and 32 micro-batches, the bubble is about 9.4%.

**1F1B Schedule (PipeDream)**: An interleaved schedule where each GPU alternates between forward and backward passes. This reduces peak memory (you don't need to store activations for all $M$ micro-batches simultaneously) and keeps the bubble fraction similar.

```
GPU 0: F1 F2 F3 F4 B1 F5 B2 F6 B3 ...
GPU 1:    F1 F2 F3 B1 F4 B2 F5 B3 ...
```

In 1F1B, each GPU begins backward passes as soon as possible, so it can discard activations earlier, reducing memory usage.

**Assumptions and what breaks:**
- Pipeline stages must be **roughly equal in compute cost**. If stage 1 takes 10 ms and stage 2 takes 100 ms, GPU 0 finishes early and idles. Load balancing across stages is critical.
- Activations at stage boundaries must be communicated between GPUs. This requires high-bandwidth interconnects (NVLink, InfiniBand).
- Gradient updates must be carefully synchronized. In GPipe, gradients are accumulated across micro-batches and applied synchronously. In PipeDream, "weight stashing" stores multiple weight versions to match forward/backward computations correctly.

#### Tensor Parallelism

**Core idea**: Instead of splitting the model by layers (vertically), split individual layers (horizontally). A single matrix multiplication $Y = XW$ is distributed: the weight matrix $W$ is partitioned across GPUs, each GPU computes a partial result, and the results are combined.

*Example — Column-parallel linear layer:*

Suppose $W$ is of shape $(d_{\text{in}}, d_{\text{out}})$ and you have 2 GPUs. Split $W$ along the output dimension:

$$W = [W_1 \;|\; W_2], \quad W_1 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}/2}, \quad W_2 \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}/2}$$

GPU 0 computes $Y_1 = XW_1$, GPU 1 computes $Y_2 = XW_2$. The full output is $Y = [Y_1 \;|\; Y_2]$. Each GPU only stores half the weight matrix.

*Example — Row-parallel linear layer:*

Split $W$ along the input dimension:

$$W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$$

The input $X$ is also split accordingly: GPU 0 gets $X_1$ (first half of features), GPU 1 gets $X_2$. Each GPU computes $Y_i = X_i W_i$. The full output is $Y = Y_1 + Y_2$, requiring an **All-Reduce** to sum the partial results.

**In Transformers (Megatron-LM approach)**:

The self-attention layer and the MLP are each split across GPUs using a column-parallel / row-parallel pair, requiring exactly 2 All-Reduce operations per transformer block (one in the forward pass, one in the backward pass). This design minimizes communication while distributing the largest parameter matrices.

**Assumptions and what breaks:**
- Tensor parallelism requires extremely high-bandwidth communication because partial results are exchanged within a single layer's forward pass. This works within a single node with NVLink but is too slow across nodes over Ethernet.
- It is typically combined with pipeline parallelism across nodes: tensor parallelism within a node, pipeline parallelism across nodes. This hybrid approach is called **3D parallelism** when combined with data parallelism.

### 2.3 Mixed Precision Training

#### The Memory and Speed Problem

Standard training uses FP32 (32-bit floating-point) for all tensors. For a model with $P$ parameters:

- Parameters: $4P$ bytes
- Gradients: $4P$ bytes
- Optimizer states (Adam stores first and second moments): $8P$ bytes
- Total: $16P$ bytes minimum, plus activations

For a 1-billion-parameter model: $16 \times 10^9$ bytes = 16 GB just for parameters, gradients, and optimizer states — leaving little room for activations and data.

**Mixed precision** uses lower-precision formats for most operations while keeping critical computations in FP32. This reduces memory and leverages the specialized hardware (Tensor Cores on NVIDIA GPUs) that can perform FP16/BF16 matrix multiplications at 2–8× the throughput of FP32.

#### Floating-Point Formats

**FP32 (float32)**: 1 sign bit, 8 exponent bits, 23 mantissa bits. Range: $\pm 3.4 \times 10^{38}$. Precision: about 7 decimal digits.

**FP16 (float16)**: 1 sign bit, 5 exponent bits, 10 mantissa bits. Range: $\pm 65,504$. Precision: about 3.3 decimal digits.

**BF16 (bfloat16)**: 1 sign bit, 8 exponent bits, 7 mantissa bits. Range: same as FP32 ($\pm 3.4 \times 10^{38}$). Precision: about 2.4 decimal digits.

The critical difference between FP16 and BF16:

- FP16 has a very small range (max value 65,504). Gradients, activations, or loss values exceeding this range cause **overflow** (become infinity). Conversely, very small gradients below $\approx 6 \times 10^{-8}$ become zero (**underflow**).
- BF16 has the same range as FP32, so overflow is essentially never a problem. However, BF16 has less precision (7 mantissa bits vs. FP16's 10), which means more rounding error in individual operations.

#### The AMP (Automatic Mixed Precision) Recipe

Introduced by Micikevicius et al. (2017), the standard mixed precision recipe has three components:

1. **FP32 master weights**: Maintain a copy of all parameters in FP32. This is the "source of truth." Updates are applied to these FP32 weights.

2. **FP16/BF16 forward and backward passes**: Cast the FP32 weights to FP16/BF16 before the forward pass. Compute activations, loss, and gradients all in FP16/BF16. The forward and backward passes are where most of the computation happens, so this is where the speed gain comes from.

3. **Loss scaling (FP16 only)**: Because FP16 has limited range, many gradient values are small enough to underflow to zero. **Loss scaling** multiplies the loss by a large constant $S$ before the backward pass. By the chain rule, all gradients are also multiplied by $S$, shifting them into FP16's representable range. Before the optimizer step, gradients are divided by $S$ to restore correct magnitudes.

   *Dynamic loss scaling*: Start with a large scale factor (e.g., $2^{16}$). If gradients contain `inf` or `nan` (overflow), skip the optimizer step and halve the scale factor. If several consecutive steps succeed without overflow, double the scale factor. This automatically finds the right scale.

   BF16 typically does not need loss scaling because its exponent range matches FP32.

#### Memory Savings

With mixed precision (FP16 parameters + FP32 master copy + FP32 optimizer states):

- FP16 parameters: $2P$ bytes
- FP16 gradients: $2P$ bytes
- FP32 master weights: $4P$ bytes
- FP32 optimizer states (Adam): $8P$ bytes
- Total: $16P$ bytes

Wait — this is the same total! The savings come from:
- **Activations**: Stored in FP16 during the forward pass, which can be the dominant memory consumer for large batch sizes. Activations often consume 5–20× the parameter memory.
- **Throughput**: Tensor Cores process FP16/BF16 operations 2–8× faster than FP32.
- **Communication**: All-Reduce transmits FP16 gradients (half the bandwidth).

---

## 3. Mathematical Formulation

### 3.1 Gradient Averaging in Data Parallelism

Let the loss function on the full mini-batch $\mathcal{B}$ of size $B$ be:

$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \ell(f_\theta(x_i), y_i)$$

The gradient is:

$$\nabla_\theta \mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(f_\theta(x_i), y_i)$$

Partition $\mathcal{B}$ into $N$ micro-batches $\mathcal{B}_1, \ldots, \mathcal{B}_N$, each of size $B/N$. The local gradient on GPU $k$ is:

$$g_k = \frac{1}{B/N} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell(f_\theta(x_i), y_i)$$

The average of local gradients recovers the global gradient:

$$\frac{1}{N} \sum_{k=1}^{N} g_k = \frac{1}{N} \sum_{k=1}^{N} \frac{N}{B} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell = \nabla_\theta \mathcal{L}(\theta)$$

This proves that synchronous data parallelism with gradient averaging is mathematically equivalent to single-GPU training with the same global batch size.

### 3.2 Communication Cost of Ring All-Reduce

Let each GPU hold a gradient vector of size $D$ elements (bytes). With $N$ GPUs in a ring:

**Reduce-Scatter phase**: Each GPU sends $D/N$ elements in each of $N-1$ rounds. Total data sent per GPU: $(N-1) \cdot D/N$.

**All-Gather phase**: Same — $(N-1) \cdot D/N$ per GPU.

**Total per GPU**: $2(N-1) \cdot D/N \approx 2D$ for large $N$.

This is asymptotically optimal: the result must arrive at every GPU, and each GPU must contribute its entire gradient, so $2D$ is the lower bound. The communication cost per GPU is independent of $N$, which means ring all-reduce scales well.

**Time estimate**: If bandwidth between adjacent GPUs is $\beta$ (bytes/second) and latency is $\alpha$:

$$T_{\text{allreduce}} = 2(N-1)\alpha + \frac{2(N-1)D}{N\beta}$$

The first term is the latency cost (proportional to $N$), and the second is the bandwidth cost (approximately $2D/\beta$, independent of $N$ for large $N$).

### 3.3 Loss Scaling Mathematics

In FP16, the smallest positive normal number is $2^{-14} \approx 6.1 \times 10^{-5}$. Subnormals extend this to $2^{-24} \approx 6 \times 10^{-8}$, but with reduced precision.

Suppose the true gradient of some parameter is $g = 10^{-6}$. In FP16, this is a subnormal with very few significant bits, or it may underflow to zero.

With loss scaling factor $S = 2^{10} = 1024$:
- Scaled gradient: $g' = S \cdot g = 1.024 \times 10^{-3}$
- This is well within FP16's normal range and retains full precision.
- Before the optimizer step: $g = g' / S = 10^{-6}$ (done in FP32).

The invariant is: the optimizer always sees the correct gradient magnitude. Loss scaling only shifts gradients into FP16's representable range during the backward pass.

### 3.4 Pipeline Bubble Fraction

With $P$ pipeline stages and $M$ micro-batches, the total time for one training step in GPipe is:

$$T = (P - 1 + M) \cdot t_f + (P - 1 + M) \cdot t_b$$

where $t_f$ and $t_b$ are the forward and backward times per micro-batch per stage. The ideal time (fully utilized pipeline) would be $M \cdot (t_f + t_b)$ per GPU. The bubble overhead is:

$$\text{Bubble fraction} = \frac{(P-1)(t_f + t_b)}{(P-1+M)(t_f + t_b)} = \frac{P-1}{P-1+M}$$

For $P=4, M=32$: bubble $= 3/35 \approx 8.6\%$. Making $M \gg P$ keeps the bubble small.

---

## 4. Worked Examples

### Example 1: Synchronous Data Parallelism with All-Reduce

**Setup**: 4 GPUs, global batch size $B=256$, model has 2 parameters $\theta = [\theta_1, \theta_2]$.

**Step 1**: Each GPU gets a micro-batch of $256/4 = 64$ samples.

**Step 2**: Each GPU computes its local gradient (simplified):
- GPU 0: $g_0 = [0.1, -0.3]$
- GPU 1: $g_1 = [0.2, -0.1]$
- GPU 2: $g_2 = [-0.1, -0.4]$
- GPU 3: $g_3 = [0.0, -0.2]$

**Step 3**: All-Reduce averages the gradients:

$$\bar{g} = \frac{1}{4}([0.1, -0.3] + [0.2, -0.1] + [-0.1, -0.4] + [0.0, -0.2])$$
$$= \frac{1}{4}[0.2, -1.0] = [0.05, -0.25]$$

**Step 4**: Every GPU now has $\bar{g} = [0.05, -0.25]$ and applies the SGD update with learning rate $\eta = 0.01$:

$$\theta_1 \leftarrow \theta_1 - 0.01 \times 0.05 = \theta_1 - 0.0005$$
$$\theta_2 \leftarrow \theta_2 - 0.01 \times (-0.25) = \theta_2 + 0.0025$$

All four GPUs compute the identical update, so parameters remain synchronized.

### Example 2: Pipeline Parallelism Bubble Calculation

**Setup**: 3-stage pipeline ($P=3$), 6 micro-batches ($M=6$), each micro-batch forward takes 10 ms, backward takes 10 ms per stage.

**Timeline** (simplified, showing GPU 0):

```
Time (ms):  0   10  20  30  40  50  60  70  80  90  100  110  120  130
GPU 0:     F1  F2  F3  F4  F5  F6 [wait]    B6  B5   B4   B3   B2   B1
```

GPU 0 finishes all forwards at 60 ms but must wait for the backward pass to propagate back from GPU 2 through GPU 1. The wait (bubble) is $(P-1) \times t_b = 2 \times 10 = 20$ ms.

Bubble fraction: $\frac{P-1}{P-1+M} = \frac{2}{2+6} = 25\%$.

With $M=32$: $\frac{2}{2+32} = 5.9\%$ — much better.

### Example 3: Dynamic Loss Scaling

**Setup**: FP16 training, initial loss scale $S = 2^{16} = 65536$.

**Step 1**: Loss $= 2.5$. Scaled loss $= 2.5 \times 65536 = 163840$. This is above FP16 max (65504), so it overflows to `inf`. Gradients are `inf`.

**Action**: Skip this step. Set $S \leftarrow 65536 / 2 = 32768$.

**Step 2**: Loss $= 2.5$. Scaled loss $= 2.5 \times 32768 = 81920$. Still overflow.

**Action**: Skip. Set $S \leftarrow 16384$.

**Step 3**: Loss $= 2.5$. Scaled loss $= 2.5 \times 16384 = 40960$. Within FP16 range. Gradients are valid.

**Action**: Unscale gradients by dividing by $S$. Apply optimizer step. After 2000 successful steps, try $S \leftarrow 32768$ again.

---

## 5. Relevance to Machine Learning Practice

### Where It Is Used

**Data parallelism** is the default for training models that fit on a single GPU but need faster training. Virtually every large-scale training run (BERT, GPT, ResNet at scale) uses data parallelism. Frameworks: PyTorch DistributedDataParallel (DDP), Horovod, TensorFlow MirroredStrategy.

**Model parallelism** is used when models exceed single-GPU memory. GPT-3 (175B parameters), PaLM (540B), and similar models use combinations of pipeline and tensor parallelism. Frameworks: Megatron-LM, DeepSpeed, FairScale.

**Mixed precision** is now standard practice. PyTorch `torch.cuda.amp`, TensorFlow mixed precision policy, and NVIDIA Apex all provide easy integration. Nearly all modern training uses BF16 or FP16.

### When to Use What

| Scenario | Strategy |
|----------|----------|
| Model fits on 1 GPU, want faster training | Data parallelism |
| Model fits on 1 GPU, dataset is enormous | Data parallelism |
| Model does not fit on 1 GPU | Model parallelism (pipeline or tensor) |
| Model does not fit on 1 node (8 GPUs) | Pipeline parallelism across nodes, tensor parallelism within node |
| Any training at scale | Mixed precision (BF16 preferred) |
| Sparse models (huge embedding tables) | Asynchronous data parallelism or parameter server |

### Trade-offs

**Data parallelism**:
- (+) Simple, well-understood, near-linear scaling up to ~64 GPUs.
- (+) Mathematically equivalent to single-GPU training.
- (−) Increases effective batch size, which can degrade generalization.
- (−) Communication overhead grows with parameter count.

**Large-batch training considerations**: Scaling from batch 256 to batch 8192 (32× GPUs) often requires learning rate warmup, learning rate scaling (linear or square-root), and LARS/LAMB optimizers to maintain convergence quality.

**Pipeline parallelism**:
- (+) Splits memory across GPUs for large models.
- (−) Pipeline bubble wastes compute.
- (−) Complex scheduling and gradient synchronization.

**Tensor parallelism**:
- (+) No pipeline bubble — computation is truly parallel within each layer.
- (−) Requires extremely high-bandwidth interconnects.
- (−) Synchronization within every layer forward pass.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "More GPUs always means faster training"

Not true. Communication overhead grows, and the effective batch size increases. Beyond a critical batch size, the model's generalization degrades (test accuracy drops even though training loss is the same). There is a point of diminishing returns where doubling GPUs yields much less than 2× speedup.

### Pitfall 2: "Data parallelism changes the optimization"

With synchronous all-reduce, data parallelism with $N$ GPUs and per-GPU batch $b$ is mathematically identical to single-GPU training with batch $Nb$. It does not change the gradient computation — but it does change the effective batch size, which is an optimization hyperparameter that matters.

### Pitfall 3: "FP16 is always safe with loss scaling"

FP16 can produce numerical issues even with loss scaling, particularly in operations that involve reductions over large tensors (e.g., softmax, layer normalization, loss computation). Best practice is to keep these operations in FP32. This is why `torch.cuda.amp` uses an "op-level" cast list: some ops run in FP16, others are forced to FP32.

### Pitfall 4: "Pipeline parallelism has no effect on convergence"

GPipe accumulates gradients across micro-batches, which is mathematically equivalent to a larger batch. PipeDream's weight stashing introduces a subtle inconsistency — the forward and backward passes may use slightly different weight versions. This generally has minimal impact but is not mathematically identical to sequential training.

### Pitfall 5: "BF16 and FP16 are interchangeable"

BF16 has lower precision (7 vs. 10 mantissa bits), which means accumulation errors are larger. For models that are sensitive to numerical precision (e.g., small learning rates, large models, gradient clipping near small thresholds), BF16 can produce different training dynamics than FP16, even though BF16 doesn't need loss scaling.

### Pitfall 6: "Asynchronous training is strictly worse"

For dense models (CNNs, Transformers), synchronous training dominates. But for large-scale recommendation systems (e.g., YouTube, Facebook ads) with massive sparse embedding tables, asynchronous training is standard because the embedding tables are so large they must be sharded across many servers, and synchronous updates would require waiting for the slowest of thousands of workers.

---

# Part II: Distributed Inference

---

## 1. Motivation & Intuition

Training a model is only half the battle. Once trained, the model must serve predictions to users — often millions of them, in real time, at low cost. This is the **inference** or **serving** problem.

Consider a search engine that uses a BERT-based model to re-rank search results. Each query must be processed in under 50 milliseconds. The service handles 100,000 queries per second at peak. A single GPU can process perhaps 500 queries per second. You need at least 200 GPUs, coordinated behind a load balancer, with batching, caching, and fault-tolerance.

Distributed inference solves three problems:

1. **Latency**: Each individual request must be fast. Techniques like model compilation, quantization, and caching reduce per-request latency.
2. **Throughput**: The system must handle high request rates. Batching, load balancing, and horizontal scaling increase aggregate throughput.
3. **Cost**: GPUs are expensive. Efficient batching, caching, and model optimization reduce the number of GPUs required.

---

## 2. Conceptual Foundations

### 2.1 Model Serving Optimizations

#### Batching

**Problem**: A single inference request (e.g., one sentence through BERT) uses only a fraction of the GPU's compute capacity. The GPU has thousands of cores designed for parallel matrix operations, but one small input matrix doesn't saturate them.

**Solution**: Group multiple requests into a batch and process them together. Instead of computing $Y = XW$ for one input row $X \in \mathbb{R}^{1 \times d}$, compute it for a batch $X \in \mathbb{R}^{B \times d}$. The GPU's parallelism is utilized, and throughput increases dramatically.

**Static batching**: Collect requests until you have $B$ of them or until a timeout expires, then process them all at once. Parameters:
- **Maximum batch size** ($B_{\max}$): Upper limit on batch size. Larger batches give higher throughput but use more memory and increase latency for the first request in the batch.
- **Timeout** ($T$): Maximum time to wait for a full batch. If the timeout expires with fewer than $B_{\max}$ requests, process whatever is available.

The tension: larger $B_{\max}$ and $T$ improve throughput but hurt latency. A request that arrives at the start of a batching window waits up to $T$ before processing begins.

**Dynamic / continuous batching (for autoregressive models)**: In models like GPT, each token generation requires a full forward pass. Different requests in a batch may be at different stages of generation (one request may be on token 5 while another is on token 50). Continuous batching (as in vLLM and TGI) allows new requests to join the batch while others are mid-generation, and completed requests to exit immediately. This drastically improves GPU utilization.

**Key mechanism — PagedAttention**: The KV cache for attention grows per token per request. Continuous batching systems use paged memory management (like OS virtual memory) to allocate KV cache blocks on demand, avoiding memory fragmentation and enabling more concurrent requests.

#### Model Compilation

Neural network frameworks like PyTorch execute operations eagerly — each operation is dispatched to the GPU individually, with Python overhead between operations. **Model compilation** optimizes the computation graph ahead of time.

**Operator fusion**: Instead of executing operations sequentially (e.g., MatMul → Add Bias → ReLU as three separate GPU kernel launches), the compiler fuses them into a single kernel. This eliminates intermediate memory reads/writes and kernel launch overhead.

**Memory planning**: The compiler analyzes the entire graph to determine optimal memory allocation, reusing buffers when intermediate results are no longer needed.

**Hardware-specific optimization**: Compilers generate code optimized for the specific GPU architecture, choosing tile sizes, loop orderings, and memory access patterns that match the hardware.

**Common tools:**

- **TensorRT (NVIDIA)**: Converts PyTorch/TensorFlow models to optimized inference engines. Performs layer fusion, precision calibration (INT8/FP16), and kernel auto-tuning. Typical speedup: 2–5× over PyTorch eager.
- **TVM (Apache)**: Open-source compiler stack. Generates optimized code for diverse hardware (GPUs, CPUs, accelerators). Uses auto-tuning to search for the best kernel implementations.
- **torch.compile (PyTorch 2.0+)**: JIT compiler that fuses operations and generates optimized kernels using TorchInductor backend. Lower setup cost than TensorRT but potentially smaller speedups.
- **ONNX Runtime**: Cross-platform runtime with graph optimizations. Models are exported to ONNX format and optimized at load time.

### 2.2 Caching Strategies

#### Prediction Caching

If the same input is sent multiple times, recomputing the prediction wastes GPU time. A prediction cache stores input→output mappings.

**When it works well**: Queries with high repetition rates. For example, product recommendation on an e-commerce site: the top 1000 products get queried millions of times. Caching their embeddings or recommendation scores eliminates redundant computation.

**When it doesn't work**: Unique or highly variable inputs. A conversational AI system where every prompt is different has low cache hit rates.

**Implementation**: Use a hash of the input as the cache key. Store results in a fast key-value store (Redis, Memcached). Set a TTL (time-to-live) to expire stale predictions if the model or data distribution changes.

**Cache invalidation**: When you deploy a new model version, the cache must be invalidated (cleared) because predictions under the new model may differ. Rolling deployments with versioned cache keys can help: cache key = hash(input) + model_version.

#### Feature Caching

Many ML systems compute features from raw data before feeding them to the model. Feature computation can be expensive (e.g., aggregating user click history over 90 days). Caching precomputed features avoids redundant work.

**Online feature stores** (Feast, Tecton) maintain feature tables that are updated asynchronously (e.g., via streaming pipelines). At inference time, the serving system looks up precomputed features by entity key (user_id, item_id) rather than recomputing them.

**KV cache for autoregressive models**: This is a model-internal form of caching. In transformer decoders, the keys and values from previous tokens are cached so they don't need to be recomputed when generating each new token. Without this cache, generating a sequence of $T$ tokens would require $O(T^2)$ total compute; with the cache, it requires $O(T)$.

### 2.3 Load Balancing

When multiple replicas of a model serve predictions, a **load balancer** distributes incoming requests across replicas to maximize throughput and minimize latency.

#### Request Routing Strategies

**Round-robin**: Send request 1 to replica 0, request 2 to replica 1, request 3 to replica 2, then back to replica 0. Simple but ignores differences in request processing time. If some requests take longer (e.g., longer sequences), some replicas may be overloaded.

**Least-connections**: Send each new request to the replica with the fewest active (in-flight) requests. This naturally adapts to heterogeneous request times and replica performance.

**Weighted routing**: Assign weights to replicas based on their capacity. A replica on an A100 GPU gets 2× the traffic of a replica on a V100 GPU.

**Latency-based routing**: Track the recent average latency of each replica and prefer replicas with lower latency. This adapts to transient issues like thermal throttling.

#### Health Checks

**Liveness probes**: Periodic pings to verify the replica process is alive. If it fails, restart the replica.

**Readiness probes**: Verify the replica is ready to serve (model is loaded, GPU memory is allocated). A replica that is still loading the model should not receive traffic.

**Model health checks**: Periodically send known inputs with expected outputs to verify the model is producing correct predictions. This catches corruption from hardware errors (GPU bit flips are rare but real at scale).

---

## 3. Mathematical Formulation

### 3.1 Batching Throughput Model

Let $t_{\text{compute}}(B)$ be the GPU computation time for a batch of size $B$. For matrix multiplications on GPUs, this function is often sublinear in $B$ for small $B$ (the GPU is underutilized) and approximately linear for large $B$ (GPU saturated).

A simplified model:

$$t_{\text{compute}}(B) = t_0 + \alpha B$$

where $t_0$ is the fixed overhead per batch (kernel launch, memory allocation) and $\alpha$ is the per-sample marginal cost.

**Throughput** (samples/second):

$$\text{Throughput}(B) = \frac{B}{t_0 + \alpha B}$$

As $B \to \infty$, throughput approaches $1/\alpha$. As $B \to 0$, throughput approaches $B/t_0$, which is small. This is why batching helps: it amortizes the fixed overhead $t_0$.

**Latency per sample** (for the first sample in the batch):

$$\text{Latency}(B) = T_{\text{wait}} + t_0 + \alpha B$$

where $T_{\text{wait}}$ is the time spent waiting for the batch to fill. The average wait time for a Poisson arrival process with rate $\lambda$ and maximum batch size $B_{\max}$ is approximately $B_{\max}/(2\lambda)$, assuming the batch fills before the timeout.

This reveals the throughput-latency tradeoff: larger batches increase throughput but also increase latency.

### 3.2 Quantization Error Bound

Model quantization maps FP32 weights to lower-precision representations (INT8, INT4). For uniform quantization of a weight $w$ to $b$ bits over range $[w_{\min}, w_{\max}]$:

$$\hat{w} = \text{round}\left(\frac{w - w_{\min}}{s}\right) \cdot s + w_{\min}$$

where the scale factor is:

$$s = \frac{w_{\max} - w_{\min}}{2^b - 1}$$

The maximum quantization error per weight is $s/2$. For INT8 ($b=8$) with weights in $[-1, 1]$: $s = 2/255 \approx 0.0078$, and max error $\approx 0.0039$. For INT4 ($b=4$): $s = 2/15 \approx 0.133$, max error $\approx 0.067$.

The total effect on model output depends on how these per-weight errors accumulate through the network. Empirically, INT8 quantization preserves accuracy within 1% of FP32 for most models. INT4 requires more careful calibration (e.g., GPTQ, AWQ) and may degrade significantly without it.

---

## 4. Worked Examples

### Example 1: Batching Throughput Calculation

**Setup**: Model serving with $t_0 = 5$ ms fixed overhead, $\alpha = 0.5$ ms/sample.

**Batch size 1**:
- Compute time: $5 + 0.5 = 5.5$ ms
- Throughput: $1/5.5 \times 10^{-3} \approx 182$ samples/sec

**Batch size 32**:
- Compute time: $5 + 0.5 \times 32 = 21$ ms
- Throughput: $32/21 \times 10^{-3} \approx 1,524$ samples/sec

**Batch size 128**:
- Compute time: $5 + 0.5 \times 128 = 69$ ms
- Throughput: $128/69 \times 10^{-3} \approx 1,855$ samples/sec

Throughput increases 8.4× going from batch 1 to 32, but only 1.2× going from 32 to 128. The diminishing returns illustrate the transition from overhead-dominated to compute-dominated regimes.

### Example 2: Cache Hit Rate and GPU Savings

**Setup**: 1 million requests/day. Cache hit rate = 60%. Each request costs $10^{-4}$ GPU-seconds.

**Without caching**: $10^6 \times 10^{-4} = 100$ GPU-seconds/day = $\sim$0.028 GPU-hours/day.

**With caching**: Only 40% of requests reach the GPU: $0.4 \times 10^6 \times 10^{-4} = 40$ GPU-seconds/day.

Savings: 60% of GPU cost. At larger scale (billions of requests, expensive models), the savings are substantial.

### Example 3: Quantization

**Setup**: Linear layer with weight matrix $W$ of shape $(1024, 1024)$. FP32 size: $1024 \times 1024 \times 4 = 4$ MB. INT8 size: $1024 \times 1024 \times 1 = 1$ MB (plus 2 scale/zero-point values per channel ≈ negligible).

Memory reduction: 4×.

Suppose weights are uniformly distributed in $[-0.5, 0.5]$.

INT8 scale: $s = 1.0 / 255 \approx 0.00392$.

For input $x = [1.0, 0.5, -0.3, \ldots]$ (1024-dimensional), the output error per element of $y = Wx$ is the accumulation of 1024 quantization errors, each at most $|x_j| \cdot s/2$. By the central limit theorem, the output error standard deviation is approximately:

$$\sigma_{\text{error}} \approx \frac{s}{2\sqrt{3}} \sqrt{\sum_j x_j^2} \approx \frac{0.00392}{2\sqrt{3}} \cdot \|x\|_2$$

For a unit-norm input: $\sigma_{\text{error}} \approx 0.00113$. This is typically small relative to the output magnitude, which is why INT8 quantization usually preserves model accuracy.

---

## 5. Relevance to Machine Learning Practice

### Real-World Serving Architectures

A typical production ML serving stack:

```
User Request
    ↓
[Load Balancer] (NGINX, AWS ALB, Envoy)
    ↓
[API Gateway] (rate limiting, authentication)
    ↓
[Prediction Cache] (Redis) → cache hit → return immediately
    ↓ (cache miss)
[Feature Store] (Feast/Tecton) → retrieve precomputed features
    ↓
[Model Server] (TorchServe, Triton, TFServing)
    ↓
[Batching Layer] → accumulate requests → GPU inference
    ↓
[Post-processing] → format response, log prediction
    ↓
User Response
```

### When to Use Each Optimization

| Technique | When to Use | When NOT to Use |
|-----------|------------|-----------------|
| Static batching | Request rate is high and relatively uniform | Latency-critical applications (real-time gaming) |
| Continuous batching | LLM serving with variable-length generation | Non-autoregressive models |
| TensorRT compilation | Latency-critical production, NVIDIA hardware | Rapid prototyping, frequent model changes |
| Prediction caching | High repetition rate in inputs | Unique inputs (conversational AI) |
| Feature caching | Expensive feature computation | Simple features computable in microseconds |
| INT8 quantization | Model is too large or too slow | Accuracy-critical applications where 1% drop is unacceptable |

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "Larger batches always improve throughput"

Beyond the GPU's memory or compute saturation point, larger batches cause out-of-memory errors or provide no additional throughput. Memory for activations scales linearly with batch size.

### Pitfall 2: "Compilation is always worthwhile"

Model compilation (TensorRT, torch.compile) has a one-time cost of minutes to hours. For models that change frequently (A/B testing many variants), the compilation overhead may not be justified. Also, some dynamic graph patterns (variable-length inputs, conditional computation) may not be fully optimizable.

### Pitfall 3: "Prediction caching means you don't need GPU scaling"

Caching helps only when cache hit rates are high. If inputs are highly diverse (e.g., LLM prompts), the hit rate is near zero, and caching provides no benefit while consuming memory.

### Pitfall 4: "Round-robin load balancing is good enough"

Round-robin assigns equal requests but ignores request heterogeneity. A request with a 512-token sequence takes 10× longer than a 50-token sequence. Least-connections or latency-based routing handles this much better.

### Pitfall 5: "Quantization is lossless"

INT8 quantization is nearly lossless for most vision models but can degrade text generation quality in LLMs, particularly for tail-distribution tokens. INT4 requires careful calibration and can significantly degrade quality without techniques like GPTQ or AWQ. Always measure task-specific accuracy after quantization.

---

# Part III: Resource Management

---

## 1. Motivation & Intuition

GPUs cost $1–$30 per hour. A large training run may require hundreds of GPUs for weeks, costing millions of dollars. A production inference service may run thousands of GPUs continuously. Resource management is the discipline of using these expensive resources efficiently.

There are three dimensions: sharing GPUs across workloads, optimizing memory to fit more work per GPU, and reducing numerical precision to shrink model footprints.

---

## 2. Conceptual Foundations

### 2.1 GPU Sharing

A single GPU may have capacity to serve multiple workloads simultaneously, especially when individual workloads are small or bursty.

**Multi-Process Service (MPS)**: NVIDIA MPS allows multiple CUDA processes to share a single GPU by multiplexing their kernels onto the GPU's streaming multiprocessors. Unlike time-slicing, MPS provides true concurrent execution. However, all processes share GPU memory, and a fault in one process can affect others.

**Time-slicing**: The GPU switches between workloads rapidly. Each workload gets exclusive access for a time slice (e.g., 10 ms). This is simpler but has context-switching overhead and provides no isolation.

**Multi-Instance GPU (MIG)**: Available on NVIDIA A100 and newer, MIG physically partitions a single GPU into up to 7 independent instances, each with its own memory, cache, and streaming multiprocessors. Instances are hardware-isolated: a fault in one instance cannot affect another. MIG provides guaranteed performance but at the cost of flexibility — partitions are coarse-grained.

**Virtual GPU (vGPU)**: NVIDIA's virtualization technology, used mainly in cloud environments. It creates virtual GPU instances that can be assigned to virtual machines, with scheduling and memory isolation managed by a hypervisor.

### 2.2 Memory Optimization

GPU memory is the primary bottleneck for training large models. Several techniques reduce memory consumption:

**Gradient checkpointing (activation recomputation)**: During the forward pass, only a subset of layers' activations are saved. During the backward pass, the non-saved activations are recomputed from the nearest saved checkpoint. This trades compute for memory: memory decreases from $O(L)$ to $O(\sqrt{L})$ for $L$ layers, at the cost of one additional forward pass (about 33% compute overhead).

**Gradient accumulation**: Instead of processing a large batch at once (which requires storing activations for all samples), process smaller micro-batches sequentially, accumulating gradients. After $K$ micro-batches, apply the accumulated gradient. This gives an effective batch size of $K \times \text{micro-batch size}$ while only using memory for one micro-batch's activations.

**CPU offloading (ZeRO-Offload / DeepSpeed)**: Move optimizer states and optionally gradients to CPU memory (which is much larger: 256–1024 GB vs. 40–80 GB on GPU). Computation happens on GPU, but optimizer updates happen on CPU. The trade-off is communication between CPU and GPU, which can become a bottleneck.

**ZeRO (Zero Redundancy Optimizer)**: In data parallelism, every GPU holds a full copy of parameters, gradients, and optimizer states. ZeRO partitions these across GPUs:

- **ZeRO Stage 1**: Partition optimizer states. Each GPU stores optimizer states for only $1/N$th of the parameters. Memory reduction: $\sim 4\times$ for Adam.
- **ZeRO Stage 2**: Partition gradients. Each GPU stores gradients for only its assigned parameter shard.
- **ZeRO Stage 3**: Partition parameters. Each GPU stores only $1/N$th of the parameters and gathers the full parameters on demand for forward/backward passes.

With ZeRO-3 and 64 GPUs, each GPU stores only 1/64th of the model, enabling models that would normally require model parallelism to be trained with data parallelism.

### 2.3 Model Quantization

Quantization reduces the precision of model weights (and optionally activations) to lower-bit representations, reducing memory and accelerating inference.

**Post-Training Quantization (PTQ)**: Quantize a trained FP32 model without additional training. A small calibration dataset is used to determine optimal quantization parameters (scale, zero-point) for each layer. This is fast (minutes) but may lose accuracy, especially at INT4 or lower.

**Quantization-Aware Training (QAT)**: Insert "fake quantization" operations into the training graph. During forward passes, weights and activations are quantized and dequantized; during backward passes, the straight-through estimator approximates gradients through the non-differentiable rounding operation. The model learns to be robust to quantization noise. QAT typically produces better accuracy than PTQ at the same bit width.

**GPTQ / AWQ / SqueezeLLM**: Advanced PTQ methods for large language models:
- **GPTQ**: Quantizes weights layer-by-layer by approximately solving a least-squares problem: for each layer, find INT4 weights that minimize the output error on a calibration set. Uses a Hessian-based method to determine the optimal quantization order and apply compensating adjustments.
- **AWQ (Activation-Aware Weight Quantization)**: Observes that a small fraction of weights are much more important than others (because they multiply large activations). AWQ scales these important weights up before quantization so they retain more precision.

**Quantization granularity**:
- **Per-tensor**: One scale/zero-point for the entire weight matrix. Simplest but least accurate.
- **Per-channel/per-row**: One scale/zero-point per output channel. Better accuracy with minimal overhead. Standard for INT8.
- **Per-group**: One scale/zero-point per group of $g$ weights (e.g., $g=128$). Used for INT4 quantization of LLMs. Balances accuracy and storage overhead.

---

## 3. Mathematical Formulation

### 3.1 Gradient Checkpointing Memory

For a model with $L$ layers, each layer producing activations of size $a$ bytes:

**Without checkpointing**: Memory for activations = $L \cdot a$

**With checkpointing** (checkpoint every $\sqrt{L}$ layers): Memory = $\sqrt{L} \cdot a$ for checkpointed activations, plus at most $\sqrt{L} \cdot a$ for recomputed activations between checkpoints. Total: $2\sqrt{L} \cdot a$.

The compute overhead is one additional forward pass of the non-checkpointed layers, approximately $1/3$ of the total forward+backward compute (since forward ≈ 1/3 of total, backward ≈ 2/3).

### 3.2 ZeRO Memory Analysis

For a model with $P$ parameters and $N$ data-parallel GPUs, using Adam optimizer:

| Component | No ZeRO | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|-----------|---------|--------|--------|--------|
| Parameters | $4P$ | $4P$ | $4P$ | $4P/N$ |
| Gradients | $4P$ | $4P$ | $4P/N$ | $4P/N$ |
| Optimizer states | $8P$ | $8P/N$ | $8P/N$ | $8P/N$ |
| **Total** | **$16P$** | **$8P + 8P/N$** | **$4P + 12P/N$** | **$16P/N$** |

(All values in bytes. $4P$ = FP32 parameters, $8P$ = Adam's two moment estimates.)

With ZeRO-3 and $N=64$: memory per GPU $= 16P/64 = P/4$ bytes. A 10-billion-parameter model requires $10^{10}/4 = 2.5$ GB per GPU (for parameters/gradients/optimizer), making it feasible even on 16 GB GPUs (with space remaining for activations).

### 3.3 Quantization Precision

For symmetric uniform quantization to $b$ bits over range $[-\alpha, \alpha]$:

Scale factor: $s = 2\alpha / (2^b - 1)$

Quantize: $q = \text{clamp}\left(\text{round}(w/s), -2^{b-1}, 2^{b-1}-1\right)$

Dequantize: $\hat{w} = q \cdot s$

The mean squared quantization error (assuming uniform weight distribution over $[-\alpha, \alpha]$) is:

$$\text{MSE} = \frac{s^2}{12} = \frac{\alpha^2}{3(2^b - 1)^2}$$

This decreases quadratically with $2^b$. Doubling the bit width reduces MSE by $\sim 4\times$.

---

## 4. Worked Examples

### Example: ZeRO Stage 3 Memory

**Setup**: 7B parameter model, 8 GPUs, Adam optimizer.

**Without ZeRO**:
- Parameters: $7 \times 10^9 \times 4 = 28$ GB
- Gradients: 28 GB
- Optimizer: $7 \times 10^9 \times 8 = 56$ GB
- Total per GPU: 112 GB → doesn't fit on an 80 GB A100

**With ZeRO-3** ($N=8$):
- Per GPU: $16 \times 7 \times 10^9 / 8 = 14$ GB

Now each GPU only uses 14 GB for parameters/gradients/optimizer, leaving 66 GB for activations and data on an 80 GB GPU. The model trains easily.

---

## 5. Relevance to Machine Learning Practice

### Practical GPU Sharing Guidelines

- **Development/prototyping**: Use MIG or time-slicing to share expensive GPUs among team members.
- **Inference of small models**: Use MPS to serve multiple models on one GPU (e.g., a classifier + an embedding model).
- **Training**: GPU sharing is generally not recommended for training due to performance interference. Dedicate GPUs for training runs.

### Memory Optimization Decision Tree

1. Model doesn't fit on one GPU → Try ZeRO-3 first (simplest). If insufficient, add pipeline parallelism.
2. Model fits but batch size is too small → Use gradient accumulation.
3. Activation memory is the bottleneck → Use gradient checkpointing.
4. Need even more savings → Combine ZeRO-3 + gradient checkpointing + mixed precision.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "ZeRO-3 has no communication overhead"

ZeRO-3 requires an All-Gather of the full parameter tensor before each forward and backward layer computation, plus a Reduce-Scatter for gradients. This can increase communication by 1.5× compared to standard DDP. The trade-off is worth it only when the model genuinely cannot fit otherwise.

### Pitfall 2: "Gradient checkpointing is free"

It reduces memory but increases compute by ~33%. For latency-critical training (tight deadline), this overhead may be unacceptable. The optimal checkpointing interval (which layers to checkpoint) depends on the model architecture.

### Pitfall 3: "INT8 quantization always preserves accuracy"

INT8 works well for vision models and many NLP models, but it can degrade for models with weight distributions that have large outliers (common in large Transformers). Techniques like SmoothQuant handle this by migrating the quantization difficulty from activations to weights.

---

# Part IV: Cost Optimization

---

## 1. Motivation & Intuition

Cloud GPU costs range from $1/hr (T4) to $30+/hr (H100). A large training run on 256 H100s for 30 days costs approximately $256 \times 30 \times 24 \times 30 = \$5.5M$. Inference at scale can cost millions per year. Cost optimization is not an afterthought — it directly affects whether a project is economically viable.

The core tension is between three competing goals: low cost, low latency, and high availability. You can optimize for any two at the expense of the third.

---

## 2. Conceptual Foundations

### 2.1 Spot Instances

Cloud providers (AWS, GCP, Azure) offer unused GPU capacity at 60–90% discounts as **spot instances** (called preemptible VMs on GCP, spot VMs on Azure). The catch: the provider can reclaim the instance with 30 seconds to 2 minutes notice.

**For training**: Spot instances are viable if your training job can tolerate interruptions:
- Checkpoint frequently (every 15–30 minutes).
- On preemption, save the current state (model, optimizer, data loader position, random seed).
- Resume from the latest checkpoint when a new spot instance becomes available.
- Use frameworks with built-in fault tolerance (PyTorch Elastic, Determined AI).

**For inference**: Spot instances are risky because preemption causes downtime. Use spot instances only for non-critical traffic (batch predictions, shadow traffic for A/B testing). For user-facing serving, use on-demand instances.

**Spot bidding strategies**: On AWS, you can set a maximum price. If the spot price exceeds your bid, the instance is terminated. Bidding at the on-demand price guarantees you're never paying more than on-demand while still getting spot discounts most of the time.

### 2.2 Auto-scaling Policies

Inference traffic varies — heavy during business hours, light at night. Auto-scaling adjusts the number of replicas to match demand.

**Reactive scaling**: Monitor a metric (GPU utilization, request queue length, latency). When the metric exceeds a threshold, add replicas; when it drops, remove replicas. Parameters:
- **Scale-up threshold**: e.g., if GPU utilization > 80% for 2 minutes, add 2 replicas.
- **Scale-down threshold**: e.g., if GPU utilization < 30% for 10 minutes, remove 1 replica.
- **Cooldown period**: After a scaling action, wait $C$ minutes before another action to prevent oscillation.

**Predictive scaling**: Use historical traffic patterns (e.g., daily/weekly cycles) to pre-provision replicas before demand increases. This avoids the latency of reactive scaling (spinning up a GPU instance takes 1–5 minutes; loading a model takes another 1–10 minutes).

**Scale-to-zero**: For low-traffic endpoints, scale down to zero replicas and spin up on demand. The first request after scale-to-zero has cold-start latency (potentially minutes). This is acceptable for internal tools but not for user-facing services.

### 2.3 Model Selection Trade-offs

Choosing the right model size is a cost optimization decision. A larger model may be more accurate but costs more to train and serve.

**Scaling laws** (Kaplan et al., Hoffmann et al.) show that model performance improves as a power law with model size, dataset size, and compute budget. Importantly, there are diminishing returns: doubling model size may improve accuracy by only 1–2%, but doubles serving cost.

**Distillation**: Train a small "student" model to mimic a large "teacher" model. The student is cheaper to serve and may retain 90–95% of the teacher's accuracy. This is a direct cost-quality tradeoff.

**Cascade / routing architectures**: Route simple requests to a small, cheap model and only escalate complex requests to a large, expensive model. For example, use a simple classifier to determine query difficulty, then route easy queries to a 1B-parameter model and hard queries to a 70B-parameter model. If 80% of queries are easy, you save 80% of the inference cost of the large model.

---

## 3. Mathematical Formulation

### 3.1 Cost Model for Spot vs. On-Demand

Let $c_{\text{on}}$ be the on-demand cost per GPU-hour, $c_{\text{spot}}$ be the spot cost, $p$ be the probability of preemption per hour, and $t_r$ be the recovery time (hours) after preemption.

Expected cost to complete $T$ hours of computation:

**On-demand**: $C_{\text{on}} = c_{\text{on}} \cdot T$

**Spot**: Training is interrupted on average every $1/p$ hours. Each interruption wastes the work since the last checkpoint plus $t_r$ recovery time. If checkpointing interval is $\Delta$ (hours), the expected wasted work per interruption is $\Delta/2$ (on average, you're halfway between checkpoints). Total expected time:

$$T_{\text{spot}} = T + \frac{p \cdot T}{1} \cdot \left(\frac{\Delta}{2} + t_r\right) = T\left(1 + p\left(\frac{\Delta}{2} + t_r\right)\right)$$

Expected cost:

$$C_{\text{spot}} = c_{\text{spot}} \cdot T_{\text{spot}} = c_{\text{spot}} \cdot T\left(1 + p\left(\frac{\Delta}{2} + t_r\right)\right)$$

Spot is cheaper when $C_{\text{spot}} < C_{\text{on}}$:

$$\frac{c_{\text{spot}}}{c_{\text{on}}} < \frac{1}{1 + p(\Delta/2 + t_r)}$$

**Example**: Spot is 30% of on-demand ($c_{\text{spot}}/c_{\text{on}} = 0.3$), preemption rate $p = 0.1$/hr, checkpoint interval $\Delta = 0.5$ hr, recovery time $t_r = 0.25$ hr.

$$\frac{1}{1 + 0.1(0.25 + 0.25)} = \frac{1}{1.05} = 0.952$$

Since $0.3 < 0.952$, spot is much cheaper in this case: spot costs $0.3 \times 1.05 = 0.315$ relative to on-demand's $1.0$.

### 3.2 Auto-scaling Optimization

Let $\lambda(t)$ be the request arrival rate at time $t$, $\mu$ be the throughput of one replica (requests/second), and $c$ be the cost per replica per second.

At time $t$, we need at least $\lceil \lambda(t)/\mu \rceil$ replicas to meet demand. The optimization problem:

$$\min_R \quad c \cdot R \cdot T \quad \text{subject to:} \quad R \geq \lceil \lambda(t)/\mu \rceil \quad \forall t, \quad \text{latency} \leq L_{\max}$$

The tension between cost and latency: keeping $R$ exactly at $\lceil \lambda/\mu \rceil$ minimizes cost but has zero headroom — any traffic spike causes latency violations. A safety margin of 20–50% extra capacity is typical.

---

## 4. Worked Examples

### Example: Spot Instance Cost Calculation

**Setup**: 100-hour training job. On-demand: $10/GPU-hr. Spot: $3/GPU-hr. Preemption: once every 10 hours on average ($p = 0.1$/hr). Checkpoint every 30 minutes ($\Delta = 0.5$ hr). Recovery: 15 minutes ($t_r = 0.25$ hr).

**On-demand cost**: $100 \times 10 = \$1,000$.

**Spot**: Expected total time = $100 \times (1 + 0.1 \times (0.25 + 0.25)) = 100 \times 1.05 = 105$ hours. Spot cost = $105 \times 3 = \$315$.

Savings: 68.5%. Even with the 5% time overhead from preemptions, spot is dramatically cheaper.

---

## 5. Relevance to Machine Learning Practice

**Spot instances** are standard for research training runs at companies like Google and Meta. Elastic training frameworks (PyTorch Elastic, Ray Train) handle preemptions automatically.

**Auto-scaling** is universal in production ML serving. Kubernetes HPA (Horizontal Pod Autoscaler) is the standard tool, often combined with custom metrics (GPU utilization, queue depth).

**Model selection** is increasingly important as LLM serving costs dominate ML budgets. The trend is toward routing architectures (e.g., Mixtral's sparse mixture-of-experts, FrugalGPT) that match model capacity to request difficulty.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "Spot instances are always cheaper"

If preemption rates are very high (e.g., 50%/hr) and recovery is slow, the total wall-clock time and total cost can exceed on-demand. Always model the expected cost.

### Pitfall 2: "Auto-scaling eliminates over-provisioning"

Scaling up takes time (instance boot + model load = 2–10 minutes). During traffic spikes, there's a window of under-provisioning. Predictive scaling and warm instance pools mitigate this.

### Pitfall 3: "The largest model is always the best choice"

A 70B model that costs 20× more to serve than a 7B model may only be 5% more accurate. For many applications, the smaller model with distillation is the right choice.

---

# Part V: Privacy & Security

---

## 1. Motivation & Intuition

Machine learning models trained on sensitive data (medical records, financial transactions, personal messages) create privacy and security risks that go beyond traditional software.

Consider a language model trained on clinical notes. Even without access to the training data, an adversary could:
- **Extract training data**: Prompt the model to generate verbatim text from its training set, including patient names and diagnoses.
- **Infer membership**: Determine whether a specific patient's record was in the training set, revealing their association with the training hospital.
- **Steal the model**: Replicate the model's behavior by querying it thousands of times and training a copy, stealing millions of dollars of training investment.

These are not theoretical concerns — they have been demonstrated in published research against production systems.

---

## 2. Conceptual Foundations

### 2.1 Model Extraction Attacks

**What it is**: An adversary queries a deployed model (typically via an API) and uses the inputs and outputs to train a "clone" model that approximates the original. This is also called model stealing.

**How it works**:
1. The adversary generates a large set of inputs (which may not even be from the same domain — random noise, synthetic data, or data from a different distribution can work).
2. For each input, the adversary obtains the model's output (predictions, probabilities, or logits).
3. The adversary trains a surrogate model using the input-output pairs, treating the original model's outputs as "soft labels."

**Why it works**: The model's output probabilities contain far more information than hard labels. A model that outputs $[0.1, 0.7, 0.15, 0.05]$ for four classes reveals the relative similarity between classes — essentially transferring the model's learned representation to the attacker.

**Types of information leaked**:
- **Decision boundaries**: The clone model approximates the original's behavior.
- **Confidence calibration**: If the API returns probabilities, the clone can learn calibrated uncertainty.
- **Architecture hints**: The complexity of the learned function can reveal the model's capacity.

**Defenses**:
- **Rate limiting**: Limit the number of API queries per user/time period.
- **Output perturbation**: Add small random noise to output probabilities.
- **Return only top-k predictions or hard labels**: Reduce information per query.
- **Watermarking**: Embed detectable patterns in the model's predictions that survive extraction, enabling legal recourse.
- **Query detection**: Monitor for suspicious query patterns (e.g., many diverse inputs from one account).

### 2.2 Membership Inference Attacks

**What it is**: Given a trained model and a specific data point $(x, y)$, determine whether that data point was in the training set.

**Why it matters**: Knowing that a person's medical record was used to train a cancer prediction model reveals they were a patient at the institution that provided training data. This is a privacy violation even if the model never outputs the patient's data directly.

**How it works**: Models tend to be more confident (assign higher probability) on data points they've seen during training than on unseen data — especially when the model is overfit. The attacker exploits this by:

1. Training "shadow models" on datasets they control, noting which data was in vs. out of training.
2. Training a binary classifier (the "attack model") that, given a model's output on a sample, predicts "member" or "non-member."
3. Applying the attack model to the target model's output on the data point of interest.

**Intuition**: An overfit model essentially memorizes its training data, producing very high confidence on training samples and lower confidence on unseen samples. The gap between these confidence levels is the signal the attacker exploits.

**Defenses**:
- **Regularization**: Weight decay, dropout, and data augmentation reduce memorization, narrowing the confidence gap.
- **Differential privacy**: Provides formal guarantees (see below).
- **Restrict output**: Return only hard labels or top-1 predictions.
- **Limit overfitting**: Early stopping, smaller model capacity.

### 2.3 Differential Privacy (DP)

**The idea**: Add controlled noise during training so that the trained model's behavior is approximately the same whether or not any single individual's data was included. This provides a formal mathematical guarantee of privacy.

**Formal definition**: A randomized mechanism $\mathcal{M}$ satisfies $(\varepsilon, \delta)$-differential privacy if for any two datasets $D$ and $D'$ differing in one record, and for any set of outcomes $S$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^\varepsilon \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

**Interpretation**:
- $\varepsilon$ (epsilon) is the **privacy budget**. Smaller $\varepsilon$ means stronger privacy. $\varepsilon = 0$ means the mechanism's output is independent of any individual's data (perfect privacy but useless model). $\varepsilon = 1$ is considered strong privacy. $\varepsilon = 10$ is weak privacy.
- $\delta$ should be smaller than $1/n$ where $n$ is the dataset size. It represents the probability that the privacy guarantee fails entirely.

**DP-SGD (Differentially Private Stochastic Gradient Descent):**

The standard algorithm for training neural networks with differential privacy:

1. **Per-sample gradient clipping**: For each sample $i$ in the mini-batch, compute the gradient $g_i = \nabla_\theta \ell(f_\theta(x_i), y_i)$. Clip each gradient to a maximum norm $C$:

   $$\bar{g}_i = g_i \cdot \min\left(1, \frac{C}{\|g_i\|_2}\right)$$

   This ensures no single sample's gradient can dominate the update.

2. **Aggregate and add noise**: Average the clipped gradients and add Gaussian noise:

   $$\tilde{g} = \frac{1}{B}\left(\sum_{i=1}^{B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)\right)$$

   The noise standard deviation $\sigma$ controls the privacy-utility tradeoff.

3. **Update parameters**: $\theta \leftarrow \theta - \eta \tilde{g}$

**Privacy accounting**: Each training step consumes some of the privacy budget $\varepsilon$. Over $T$ steps, the total privacy cost grows (sublinearly, due to composition theorems). The Rényi Differential Privacy (RDP) accountant or the Gaussian mechanism's moments accountant tracks the cumulative privacy cost. Training must stop before the budget is exhausted.

**Key parameters and trade-offs**:
- **Clipping norm $C$**: Too small → gradients are aggressively clipped, losing signal. Too large → more noise needed for the same privacy level.
- **Noise multiplier $\sigma$**: Higher → better privacy, worse model accuracy.
- **Batch size $B$**: Larger batches are better for DP-SGD because the noise is added once per batch but the signal (sum of clipped gradients) grows with $B$. The noise-to-signal ratio improves.
- **Training epochs**: More epochs consume more privacy budget. DP-SGD models are typically trained for fewer epochs than non-private models.

### 2.4 Federated Learning

**The idea**: Instead of collecting all data in one place, training happens on-device. Each device (phone, hospital, edge server) trains the model on its local data and sends only model updates (gradients or weight differences) to a central server. The server aggregates updates to improve the global model.

**Federated Averaging (FedAvg) — the standard algorithm**:

1. **Server broadcasts** the current global model $\theta^t$ to a subset of $K$ clients.
2. **Each client $k$** trains the model on its local data for $E$ local epochs, starting from $\theta^t$, producing updated parameters $\theta_k^{t+1}$.
3. **Server aggregates**: $$\theta^{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} \theta_k^{t+1}$$ where $n_k$ is the number of samples on client $k$ and $n = \sum n_k$.
4. Repeat.

**Key challenges**:

- **Non-IID data**: Each client's data may have a very different distribution (e.g., one hospital specializes in pediatrics, another in geriatrics). This violates the IID assumption of standard SGD and can cause the global model to converge slowly or to a poor optimum. Mitigation: use techniques like FedProx (adds a proximal regularizer to keep local updates close to the global model) or SCAFFOLD (uses control variates to correct gradient drift).

- **Communication efficiency**: Sending full model updates is expensive for large models over wireless connections. Mitigation: gradient compression (quantization, sparsification), local training for more epochs (reducing communication rounds), or sending only model differences.

- **Stragglers and dropouts**: Some devices may be slow (old phones, weak connections) or go offline mid-round. The server must handle partial updates and timeouts.

- **Privacy**: While federated learning avoids centralizing raw data, model updates can still leak information. An adversary observing a client's gradient update can potentially reconstruct the client's data. Solution: combine federated learning with differential privacy (add noise to local updates) and/or secure aggregation (an MPC protocol that allows the server to compute the aggregate without seeing individual updates).

---

## 3. Mathematical Formulation

### 3.1 Membership Inference via Confidence Thresholding

A simple membership inference attack uses the model's confidence on the true label:

$$\text{Predict "member"} \iff f_\theta(x)[y] > \tau$$

where $f_\theta(x)[y]$ is the model's predicted probability for the true class $y$, and $\tau$ is a threshold. This works because overfit models assign higher probabilities to training data.

The attack's accuracy depends on the generalization gap:

$$\text{Attack advantage} \approx \text{Train accuracy} - \text{Test accuracy}$$

A model with train accuracy 99% and test accuracy 90% is highly vulnerable. A model with train accuracy 92% and test accuracy 90% is much harder to attack.

### 3.2 DP-SGD Privacy Guarantee

For the Gaussian mechanism with sensitivity $\Delta_2 = C$ (the clipping norm), noise $\sigma$, and $T$ training steps with subsampling rate $q = B/n$ (batch size / dataset size):

Under Rényi Differential Privacy (RDP) of order $\alpha$, a single step provides:

$$\varepsilon_{\text{step}}(\alpha) \leq \frac{1}{\alpha-1} \log\left(1 + q^2 \binom{\alpha}{2} \min\left(4(e^{1/\sigma^2} - 1), \frac{2}{\sigma^2}\right)\right)$$

Over $T$ steps (by RDP composition): $\varepsilon_{\text{total}}(\alpha) = T \cdot \varepsilon_{\text{step}}(\alpha)$

Convert to $(\varepsilon, \delta)$-DP: $\varepsilon = \min_\alpha \varepsilon_{\text{total}}(\alpha) + \frac{\log(1/\delta)}{\alpha - 1}$

In practice, libraries like Opacus (PyTorch) handle this accounting automatically.

### 3.3 FedAvg Convergence

For FedAvg with $K$ clients, $E$ local epochs, learning rate $\eta$, and non-IID data:

The convergence bound (simplified) after $R$ communication rounds is approximately:

$$\mathbb{E}\left[\|\nabla \mathcal{L}(\theta^R)\|^2\right] \leq O\left(\frac{1}{\sqrt{KER}} + \frac{E \sigma_g^2}{K} + E^2 \Gamma^2\right)$$

where $\sigma_g^2$ is the stochastic gradient variance and $\Gamma^2$ measures the data heterogeneity (non-IIDness) across clients.

Key insight from the third term: more local epochs ($E$) allow the local models to drift farther from the global optimum when data is non-IID ($\Gamma > 0$). This creates a tension — more local epochs reduce communication but increase gradient drift.

---

## 4. Worked Examples

### Example 1: DP-SGD

**Setup**: Dataset of $n=10,000$ records. Batch size $B=100$. Clipping norm $C=1.0$. Noise multiplier $\sigma=1.0$. Target $\delta = 10^{-5}$.

Subsampling rate: $q = B/n = 100/10000 = 0.01$.

After $T=1000$ steps (10 epochs), using the RDP accountant, the total $\varepsilon \approx 3.0$ (computed numerically — the exact value depends on the RDP calculation).

This means: for any individual record, the model's distribution changes by at most a factor of $e^3 \approx 20$ whether or not that record is included. This is moderate privacy — strong enough to prevent most membership inference attacks, but weak enough that the model retains reasonable accuracy.

To achieve $\varepsilon = 1.0$ (strong privacy), you would need either more noise ($\sigma \approx 2.0$), fewer steps, or a larger dataset (which reduces $q$).

### Example 2: Membership Inference

**Setup**: Binary classifier on medical data. Training accuracy: 95%. Test accuracy: 78%.

**Attack**: For each candidate data point $(x, y)$, query the model and get $p = f_\theta(x)[y]$.

Training data points have average confidence $p \approx 0.95$.
Test-distribution points have average confidence $p \approx 0.78$.

Set threshold $\tau = 0.87$ (midpoint). The attack correctly identifies members with $\sim$85% precision and $\sim$80% recall.

**Mitigation**: Apply strong regularization. With dropout and weight decay, the gap narrows:
- Training accuracy: 82%. Test accuracy: 78%.
- Confidence gap: small. Attack accuracy drops to near random (50–55%).

### Example 3: Federated Averaging

**Setup**: 5 hospitals. Hospital A has 1000 pediatric records, Hospital B has 500 cardiac records, etc. Global model for disease classification.

**Round 1**:
1. Server sends global model $\theta^0$ to all 5 hospitals.
2. Each hospital trains for $E=3$ local epochs on its data.
3. Hospital A (1000 samples) produces $\theta_A^1$, Hospital B (500 samples) produces $\theta_B^1$, etc.
4. Server aggregates: $\theta^1 = \frac{1000}{3000}\theta_A^1 + \frac{500}{3000}\theta_B^1 + \ldots$ (weighted by dataset size).

**Problem**: Hospital A trains exclusively on pediatric data. Its local model shifts heavily toward pediatric features. Hospital B shifts toward cardiac features. After aggregation, the global model may not be good at either. This is the non-IID problem.

**Solution (FedProx)**: Add a proximal term to each hospital's local loss:

$$\mathcal{L}_k^{\text{local}} = \mathcal{L}_k(\theta) + \frac{\mu}{2}\|\theta - \theta^t\|^2$$

This penalizes the local model for drifting too far from the global model, keeping updates more aligned across heterogeneous clients.

---

## 5. Relevance to Machine Learning Practice

### Where Privacy and Security Techniques Are Used

**Differential privacy**: Apple uses DP for analytics (emoji usage, word frequency). Google uses DP in Chrome's RAPPOR and in GBoard (keyboard prediction). US Census 2020 used DP.

**Federated learning**: Google GBoard (next-word prediction), Apple Siri and QuickType, healthcare consortia (training models across hospitals without sharing patient data).

**Model extraction defenses**: Any API-served model (OpenAI, Google Cloud Vision, AWS Rekognition) must consider extraction risk. Rate limiting and output restriction are standard.

**Membership inference**: Used as a privacy audit tool — organizations test their own models for membership inference vulnerability before deployment.

### Trade-offs

| Technique | Privacy Gain | Cost |
|-----------|-------------|------|
| DP-SGD | Formal guarantee ($\varepsilon$) | Accuracy loss (~2-15%), slower training |
| Federated Learning | Data stays local | Communication cost, non-IID challenges |
| Output perturbation | Reduces extraction risk | Degrades prediction quality |
| Rate limiting | Increases extraction cost | May frustrate legitimate users |
| Quantized outputs | Reduces information leakage | Loses calibration, probability info |

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "Federated learning is private by default"

Federated learning does not centralize raw data, but gradient updates can still leak sensitive information. Deep leakage from gradients (Zhu et al., 2019) showed that images can be reconstructed from gradients. Federated learning must be combined with DP or secure aggregation for real privacy.

### Pitfall 2: "Differential privacy makes the model useless"

Modern DP-SGD with careful hyperparameter tuning achieves reasonable accuracy. Key tricks: use large batch sizes (which improve the noise-to-signal ratio), pretrain on public data (then fine-tune with DP on private data), and use strong data augmentation (which acts as implicit regularization that reduces overfitting even without DP).

### Pitfall 3: "Model extraction requires the same data distribution"

Attackers can use random or synthetic inputs — the model's outputs on any input provide information about its learned function. This makes extraction harder to defend against than one might expect.

### Pitfall 4: "Membership inference only works on overfit models"

While overfitting increases vulnerability, even well-generalized models can leak membership information through subtle differences in output distributions. DP provides the only formal defense.

### Pitfall 5: "Removing data from the training set is sufficient for privacy"

Once a model is trained, removing the data doesn't remove its influence on the model. **Machine unlearning** attempts to address this, but it's an active research area with no production-ready solutions for large models.

---

# Interview Preparation Questions

---

## Section A: Foundational Questions

### Q1: Explain the difference between data parallelism and model parallelism. When would you use each?

**Answer**: Data parallelism replicates the full model on every GPU and partitions the data. Each GPU processes a different subset of the mini-batch, computes gradients locally, and all GPUs synchronize gradients via All-Reduce. Use data parallelism when the model fits on one GPU and you want to increase training throughput.

Model parallelism splits the model itself across GPUs because it doesn't fit on one device. Pipeline parallelism divides the model by layers (vertical split), while tensor parallelism splits individual layers' weight matrices (horizontal split). Use model parallelism when the model exceeds single-GPU memory. In practice, large model training uses 3D parallelism: tensor parallelism within a node (requires high-bandwidth NVLink), pipeline parallelism across nodes, and data parallelism across pipeline replicas.

### Q2: What is the purpose of loss scaling in mixed precision training?

**Answer**: Loss scaling addresses the underflow problem in FP16 arithmetic. FP16 can only represent values down to about $6 \times 10^{-8}$; gradient magnitudes frequently fall below this threshold, especially in early layers. Loss scaling multiplies the loss by a large factor $S$ before the backward pass. Since backpropagation applies the chain rule multiplicatively, all gradients are scaled by $S$, shifting small gradient values into FP16's representable range. Before the optimizer step, gradients are divided by $S$ to restore correct magnitudes. Dynamic loss scaling adjusts $S$ automatically: increasing it when no overflow occurs and halving it when overflow (inf/nan gradients) is detected. BF16 shares FP32's exponent range and typically doesn't need loss scaling.

### Q3: What is differential privacy, and why is it relevant to ML?

**Answer**: Differential privacy is a mathematical framework that bounds the information leakage about any single individual in a dataset. A mechanism is $(\varepsilon, \delta)$-DP if the probability of any output changes by at most a multiplicative factor of $e^\varepsilon$ (plus an additive $\delta$) when one individual's data is added or removed. In ML, DP-SGD achieves this by clipping per-sample gradients to bound their sensitivity and adding calibrated Gaussian noise to the aggregate gradient. The privacy budget $\varepsilon$ accumulates over training steps. DP prevents membership inference attacks and provides a formal privacy guarantee, which is critical for training on sensitive data like medical records or financial transactions. The cost is reduced model accuracy due to the injected noise.

### Q4: What is continuous batching, and why is it important for LLM serving?

**Answer**: In autoregressive LLM serving, different requests generate different numbers of tokens. Under static batching, the batch cannot proceed until the longest sequence finishes, wasting GPU cycles while shorter sequences wait. Continuous batching (also called inflight batching) allows completed requests to exit and new requests to enter the batch at each decoding step. This keeps GPU utilization high because the batch always contains active work. It is implemented using paged KV cache management (PagedAttention), which allocates KV cache memory in fixed-size blocks, avoiding the fragmentation that would occur with contiguous allocation for variable-length sequences. Continuous batching can improve throughput by 2–4× compared to static batching.

### Q5: What are the differences between FP16 and BF16?

**Answer**: Both are 16-bit formats, but they allocate bits differently. FP16 has 5 exponent bits and 10 mantissa bits, giving it moderate precision (about 3.3 decimal digits) but a limited range (max ~65,504). BF16 has 8 exponent bits and 7 mantissa bits, giving it lower precision (about 2.4 decimal digits) but the same range as FP32 (max ~$3.4 \times 10^{38}$). The practical consequence is that FP16 is prone to overflow (values exceeding 65,504 become infinity), necessitating loss scaling during training. BF16 rarely overflows but introduces more rounding error per operation. For training, BF16 is generally preferred because it avoids loss scaling complexity. For inference, FP16 may give slightly more accurate results due to higher precision. Hardware support matters too: BF16 Tensor Cores are available on A100+ and TPUs.

---

## Section B: Mathematical Questions

### Q6: Prove that synchronous data parallelism with gradient averaging is mathematically equivalent to single-GPU training.

**Answer**: The global gradient on mini-batch $\mathcal{B}$ of size $B$ is:

$$\nabla_\theta \mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell_i$$

Partition $\mathcal{B}$ into $N$ equal subsets $\mathcal{B}_1, \ldots, \mathcal{B}_N$. GPU $k$'s local gradient is:

$$g_k = \frac{1}{|\mathcal{B}_k|} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell_i = \frac{N}{B} \sum_{i \in \mathcal{B}_k} \nabla_\theta \ell_i$$

The average across GPUs:

$$\frac{1}{N}\sum_{k=1}^{N} g_k = \frac{1}{N}\sum_{k=1}^{N}\frac{N}{B}\sum_{i \in \mathcal{B}_k}\nabla_\theta \ell_i = \frac{1}{B}\sum_{k=1}^{N}\sum_{i \in \mathcal{B}_k}\nabla_\theta \ell_i = \frac{1}{B}\sum_{i=1}^{B}\nabla_\theta \ell_i = \nabla_\theta \mathcal{L}$$

The last equality holds because $\{\mathcal{B}_k\}$ is a partition of $\mathcal{B}$. Therefore, All-Reduce averaging produces exactly the global gradient.

Note: This equivalence holds for SGD. For optimizers with per-parameter state (Adam), equivalence requires that each GPU maintains identical optimizer states, which is guaranteed because they receive identical averaged gradients at every step (given identical initialization).

### Q7: Derive the communication cost of Ring All-Reduce.

**Answer**: Consider $N$ GPUs in a ring, each with gradient vector of $D$ bytes.

**Reduce-Scatter**: Each GPU's gradient is divided into $N$ chunks of $D/N$ bytes. In round $r$ (for $r = 1, \ldots, N-1$), each GPU sends one chunk to its right neighbor and receives one chunk from its left, performing elementwise addition. After $N-1$ rounds, each GPU holds the fully reduced (summed) version of one chunk.

Per-GPU data sent: $(N-1) \times D/N$ bytes.
Per-GPU data received: $(N-1) \times D/N$ bytes.

**All-Gather**: Over another $N-1$ rounds, each GPU forwards its fully-reduced chunk around the ring. After $N-1$ rounds, every GPU has all $N$ chunks of the fully reduced result.

Per-GPU data sent: $(N-1) \times D/N$ bytes.

**Total per GPU**: $2 \times (N-1) \times D/N$ bytes sent (and same received).

For large $N$: $\approx 2D$ bytes — independent of $N$. This is asymptotically optimal because each GPU must communicate its full gradient ($D$ bytes out) and receive the aggregated result ($D$ bytes in).

### Q8: Explain the privacy guarantee of DP-SGD mathematically. How does the clipping norm C affect the privacy-utility tradeoff?

**Answer**: DP-SGD clips each per-sample gradient to norm $C$ and adds Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$. The clipping bounds the L2 sensitivity of the sum of gradients to $C$ per sample. By the Gaussian mechanism, adding noise with standard deviation $\sigma C$ to a function with L2 sensitivity $C$ provides $(\alpha, \alpha/(2\sigma^2))$-RDP for a single query.

The effect of $C$:

If $C$ is too small, most gradients are clipped, introducing bias (the clipped gradient's direction may differ from the true gradient's, and its magnitude is truncated). This is a utility loss independent of noise.

If $C$ is too large, the noise $\sigma C$ must also be large to maintain the same $\varepsilon$. Since $\varepsilon \propto C/\sigma$ (for fixed Gaussian mechanism), increasing $C$ while keeping $\varepsilon$ fixed requires proportionally increasing $\sigma$, which means more noise relative to the gradient signal.

The optimal $C$ is typically at a percentile (e.g., median) of the gradient norm distribution: large enough that most gradients are not clipped, but small enough that noise remains manageable. In practice, $C$ is a hyperparameter tuned via validation performance under a fixed privacy budget.

### Q9: What is the pipeline bubble fraction, and how can it be minimized?

**Answer**: For a pipeline with $P$ stages and $M$ micro-batches, the GPipe schedule requires $(P-1)$ time slots for the pipeline to fill (ramp-up) and $(P-1)$ time slots for it to drain (ramp-down). The productive work is $M$ time slots.

Bubble fraction = $\frac{(P-1)}{M + (P-1)}$.

This is minimized by making $M \gg P$. For example, with $P = 8$ stages and $M = 64$ micro-batches, the bubble is $7/71 \approx 9.9\%$.

However, increasing $M$ increases memory because GPipe stores activations for all $M$ micro-batches simultaneously (needed for the backward pass). The 1F1B (one-forward-one-backward) schedule of PipeDream limits the peak number of in-flight micro-batches to $P$ (instead of $M$), reducing memory while maintaining a similar bubble fraction.

Interleaved pipeline parallelism (Megatron-LM v3) assigns each GPU multiple non-consecutive stages (e.g., GPU 0 handles stages 1 and 5). This reduces the effective number of pipeline stages per pass through a virtual pipeline, reducing the bubble to $(P-1)/(M \times V + P - 1)$ where $V$ is the number of virtual stages per GPU.

---

## Section C: Applied Questions

### Q10: You're serving a 70B-parameter LLM. Users report high latency (>5s for the first token). How do you diagnose and fix this?

**Answer**: High time-to-first-token (TTFT) for a large LLM is typically caused by the prefill phase (processing the entire input prompt), not token generation. Diagnosis and solutions:

**Diagnosis**: Profile the inference pipeline. Measure time for model loading, tokenization, prefill (processing the full prompt in one forward pass), and first-token decoding. For a 70B model, prefill on a single GPU can take several seconds for long prompts.

**Solutions (ordered by impact)**:

1. **Tensor parallelism**: Split the model across 4–8 GPUs within a node. Prefill compute is divided, reducing TTFT roughly proportionally. This is the single biggest lever.

2. **Quantization**: Apply GPTQ or AWQ to reduce the model to INT4 (70B → ~35 GB, fitting on a single 80 GB GPU or 2 GPUs). This reduces both memory bandwidth and compute, typically 2–3× faster prefill.

3. **Model compilation**: Use TensorRT-LLM or vLLM with optimized CUDA kernels. Flash Attention reduces memory traffic during attention computation. Fused kernels eliminate overhead between operations.

4. **Prompt caching / prefix sharing**: If many requests share a common system prompt, cache the KV cache for the shared prefix. Subsequent requests skip processing the shared prefix entirely.

5. **Speculative decoding**: Use a small draft model (e.g., 7B) to generate candidate tokens, then verify them in parallel with the 70B model. This doesn't reduce TTFT but reduces time-per-output-token, improving overall response latency.

### Q11: You need to train a model on medical data from 10 hospitals without sharing patient data. Describe your approach.

**Answer**: This is a federated learning scenario with privacy requirements. My approach:

**Architecture**: Use FedAvg or FedProx for federated training. Each hospital trains locally and sends model updates (not data) to a central server (which could be operated by a trusted third party or operated in a decentralized manner).

**Privacy layer**: Combine federated learning with DP-SGD at each hospital. Each hospital clips gradients and adds noise locally before sending updates. Additionally, implement secure aggregation so the server only sees the aggregate update, not individual hospital updates. This provides both local and central privacy guarantees.

**Non-IID handling**: Medical data across hospitals will be highly non-IID (different patient demographics, specialties, equipment). Use FedProx (proximal regularization) to reduce client drift. Alternatively, use SCAFFOLD with control variates. Set local epochs $E$ conservatively (1–3) to limit drift.

**Communication efficiency**: Medical models can be large (e.g., 3D imaging CNNs). Use gradient compression (top-k sparsification or random quantization) to reduce communication costs, especially if hospitals have limited bandwidth.

**Validation**: Each hospital holds out a local validation set. The server also maintains a public validation set (non-sensitive data) to evaluate the global model. Monitor for hospital-specific performance degradation (fairness across institutions).

**Regulatory compliance**: Ensure the approach satisfies HIPAA (US) or GDPR (EU) requirements. DP provides a formal privacy guarantee that can satisfy regulatory scrutiny. Document the privacy budget $\varepsilon$ and $\delta$.

### Q12: Your team is spending $500K/month on inference GPUs. How do you reduce costs?

**Answer**: I'd approach this systematically by identifying the largest cost drivers and addressing them in order of impact:

**Step 1: Measure**. Profile the current system: GPU utilization (typically 20–40% for unoptimized serving), request volume patterns, latency SLOs, cache hit rates, model sizes.

**Step 2: Quick wins** (weeks to implement):
- **Auto-scaling**: If utilization is low, reduce replica count during off-peak hours. If currently running 100 GPUs 24/7 but peak utilization is only 60%, scaling down during off-peak (50% of hours) could save 30–40%.
- **Batching optimization**: Increase batch size or implement continuous batching. Moving from batch-1 to batch-32 can provide 5–8× throughput improvement, meaning you need 5–8× fewer GPUs.
- **Prediction caching**: If the workload has repetitive queries, even a 30% cache hit rate saves 30% of GPU cost.

**Step 3: Medium-term** (weeks to months):
- **Model quantization**: INT8 quantization provides 2–4× speedup with negligible accuracy loss. INT4 with GPTQ/AWQ provides another 2×. A 4× speedup means 4× fewer GPUs.
- **Model distillation**: If the serving model is 70B, distill to 7B. Accuracy drops 3–5%, but cost drops 10×.
- **Cascade routing**: Route simple queries to a small model, hard queries to a large model. If 70% of queries are simple, this saves ~60% of large-model cost.

**Step 4: Infrastructure** (months):
- **Spot instances** for non-latency-critical workloads (batch predictions, shadow scoring).
- **Model compilation** (TensorRT) for 2–3× inference speedup.
- **Reserved instances** for the baseline workload (30–60% discount vs. on-demand).

**Expected outcome**: A well-optimized stack typically achieves 3–10× cost reduction, bringing $500K/month to $50K–$150K/month.

---

## Section D: Debugging & Failure-Mode Questions

### Q13: Training on 64 GPUs is slower than expected — 40× speedup instead of the theoretical 64×. What's wrong?

**Answer**: The gap between 40× and 64× (37.5% efficiency loss) is caused by one or more of:

**Communication overhead**: All-Reduce takes time. If the model has many parameters (e.g., hundreds of millions), and the interconnect bandwidth is limited (e.g., 25 Gbps Ethernet instead of 200 Gbps InfiniBand), gradient synchronization becomes the bottleneck. Measure: profile all-reduce time vs. compute time. Fix: overlap communication with computation (backward pass computes gradients layer-by-layer; send each layer's gradient as soon as it's ready, don't wait for the full backward pass). Upgrade to InfiniBand if possible.

**Straggler GPUs**: One slow GPU forces all others to wait. Causes: thermal throttling, ECC memory errors causing retries, uneven data loading. Measure: log per-GPU step times and look for outliers. Fix: identify and replace faulty hardware; ensure uniform data loading speeds.

**Data loading bottleneck**: If data loading is slow (reading from network storage, slow preprocessing), GPUs idle waiting for data. Measure: GPU utilization during training. If it's below 90%, data loading is likely the bottleneck. Fix: pre-fetch data to local SSD, increase the number of data loading workers, use optimized data formats (WebDataset, TFRecord).

**Unbalanced computation**: If using pipeline parallelism and stages are unbalanced, the bubble overhead is larger than expected. If using tensor parallelism, imbalanced layer sizes cause some GPUs to wait for others.

**Small model relative to batch size**: With 64 GPUs, if the model is small, the communication-to-computation ratio is high (more time synchronizing than computing). Fix: increase the micro-batch size per GPU to increase computation time relative to communication.

### Q14: Your DP-SGD trained model has 20% lower accuracy than the non-private version. How do you close the gap?

**Answer**: A 20% accuracy gap is large and suggests several suboptimal settings:

**Increase batch size**: DP-SGD benefits from large batches because noise is added once per batch while signal scales with batch size. The noise-to-signal ratio is $\propto \sigma C / (B \cdot \bar{g})$ where $\bar{g}$ is the average gradient magnitude. Doubling $B$ halves the noise-to-signal ratio. Use gradient accumulation if GPU memory limits $B$.

**Tune the clipping norm $C$**: Monitor the fraction of gradients that are clipped. If $>$90% are clipped, $C$ is too small (too much bias). If $<$10% are clipped, $C$ is too large (noise is unnecessarily amplified). The optimal $C$ is typically at the 50th–75th percentile of gradient norms.

**Pretrain on public data**: Fine-tune the privately-trained model starting from a pretrained checkpoint (trained on non-sensitive public data). The pretrained features already capture useful representations, so the private fine-tuning phase needs fewer steps and less gradient signal, reducing the required privacy budget.

**Reduce training epochs**: More epochs consume more privacy budget. With a pretrained model, 1–3 epochs of private fine-tuning may suffice.

**Use group privacy**: If the privacy requirement is per-user (not per-sample) and each user contributes $k$ samples, the per-sample $\varepsilon$ can be $k$ times smaller for the same per-user guarantee. This allows less noise per step.

**Consider DP-FTRL instead of DP-SGD**: DP-FTRL (Follow The Regularized Leader) can provide better privacy-utility tradeoffs by leveraging the tree aggregation mechanism instead of per-step noise addition.

### Q15: After deploying your model on a quantized INT8 engine, users report that a specific class of inputs produces nonsensical outputs. What's happening?

**Answer**: This is likely an outlier weight or activation problem. Some possibilities:

**Activation outliers**: Certain inputs trigger very large activations in specific layers. INT8 quantization uses a fixed scale factor calibrated on a representative dataset. If the problematic inputs produce activation magnitudes far outside the calibration range, they overflow or are heavily clipped after quantization, corrupting the computation.

Diagnosis: Run the problematic inputs through both the FP32 and INT8 models, comparing intermediate activations layer by layer to identify where they diverge.

Fixes:
- Recalibrate quantization with a dataset that includes the problematic input class.
- Use per-channel or per-token activation quantization (finer granularity adapts to outliers).
- Apply SmoothQuant: multiply weights by a per-channel factor and divide activations by the same factor, migrating the quantization difficulty from activations (which have outliers) to weights (which are more uniform).
- Keep outlier-prone layers in FP16 (mixed-precision quantization).

**Weight outlier channels**: Some transformer layers (particularly in LLMs) have a few weight channels with magnitudes 10–100× larger than others. Per-tensor INT8 quantization allocates equal precision to all channels, wasting precision on the many small-valued channels.

Fix: Use per-channel quantization or GPTQ/AWQ which handle outliers explicitly.

---

## Section E: Follow-Up & Probing Questions

### Q16: "You mentioned ring all-reduce. What are the alternatives, and when would you use them?"

**Answer**: Ring all-reduce is optimal for bandwidth but has latency $O(N)$ due to $2(N-1)$ sequential communication rounds. Alternatives:

**Tree all-reduce**: Arranges GPUs in a binary tree. Reduce phase: leaves send to parents, each parent sums and sends up. Broadcast phase: root sends down. Latency: $O(\log N)$ rounds, but bandwidth cost is suboptimal — the root handles the full gradient while leaves handle only theirs. Good when $N$ is very large and latency dominates (small gradients, many GPUs).

**Recursive halving-doubling**: Combines reduce-scatter via recursive halving with all-gather via recursive doubling. $O(\log N)$ latency and near-optimal bandwidth. Used by some MPI implementations.

**Hierarchical all-reduce**: Two-level ring: first, all-reduce within each node (using NVLink), then all-reduce across nodes (using InfiniBand). This matches the hardware topology: intra-node bandwidth (NVLink, 600+ GB/s) is much higher than inter-node (InfiniBand, 200–400 Gb/s). Reduces inter-node traffic by a factor of the number of GPUs per node.

In practice, NCCL (NVIDIA's library) automatically selects the best algorithm based on message size and topology.

### Q17: "You said federated learning updates can leak data. How exactly?"

**Answer**: The key attack is "deep leakage from gradients" (Zhu et al., 2019). Given the gradient $\nabla_\theta \ell(f_\theta(x), y)$ computed on a single data point $(x, y)$, an attacker initializes a random "dummy" input $x'$ and optimizes it to match the observed gradient:

$$x^* = \arg\min_{x'} \|\nabla_\theta \ell(f_\theta(x'), y') - g_{\text{observed}}\|^2$$

For images, this optimization can recover the original image with high fidelity, especially for small batch sizes.

For batch gradients (averaged over multiple samples), reconstruction is harder but not impossible — follow-up work (Geiping et al., 2020; Yin et al., 2021) showed that batches of up to 100 images can be partially reconstructed for certain architectures.

Defenses beyond DP noise: secure aggregation ensures the server never sees individual gradients (only the aggregate); gradient compression (sparsification) removes information from individual updates; and increasing local batch sizes averages out per-sample information.

### Q18: "What happens when you scale the batch size from 256 to 8192 using data parallelism? What do you need to change?"

**Answer**: Scaling batch size by 32× (using 32× more GPUs) creates the "large-batch training" challenge:

**Learning rate scaling**: The optimal learning rate typically scales with batch size. Linear scaling rule: multiply the learning rate by $k$ when multiplying the batch size by $k$. This works for SGD on convex-like loss surfaces. For Adam, square-root scaling ($\eta \propto \sqrt{k}$) is sometimes more appropriate because Adam already adapts per-parameter.

**Warmup**: Starting with a very large learning rate can destabilize early training when the loss surface is non-smooth. Gradually increase the learning rate from a small value to the target over the first 5–10% of training steps.

**Generalization gap**: Very large batch sizes tend to converge to sharper minima that generalize worse. At some critical batch size, test accuracy degrades noticeably even though training loss reaches the same level. This critical batch size is task-dependent.

**LARS/LAMB optimizers**: For very large batches (32K+), standard SGD with linear scaling diverges. LARS (Layer-wise Adaptive Rate Scaling) and LAMB (Layerwise Adaptive Moments-based) compute per-layer learning rates based on the ratio of parameter norm to gradient norm, stabilizing training at extreme batch sizes.

**Training steps**: With a 32× larger batch, you process 32× more data per step, so you need ~32× fewer steps to see the same amount of data. Total training time decreases, but you must ensure the reduced number of steps is sufficient for convergence. Sometimes more total data (more epochs) is needed to compensate for the generalization gap.

### Q19: "How would you handle a model that's too large for even a single node (8× 80GB GPUs) but you need to train it?"

**Answer**: For models exceeding 640 GB (8×80GB), you need multi-node model parallelism combined with memory optimization:

**Architecture**: 3D parallelism across a cluster:
- Within each node (8 GPUs connected by NVLink at 600+ GB/s): tensor parallelism to split each layer across all 8 GPUs.
- Across nodes (connected by InfiniBand at 200–400 Gb/s): pipeline parallelism to assign different groups of layers to different nodes.
- Across pipeline replicas: data parallelism to increase throughput.

**Memory optimization**:
- ZeRO-3 to partition optimizer states, gradients, and parameters.
- Mixed precision (BF16) to halve activation memory.
- Gradient checkpointing to reduce activation memory from $O(L)$ to $O(\sqrt{L})$.
- Activation offloading (move checkpointed activations to CPU memory between forward and backward).

**Example sizing**: A 175B-parameter model in BF16 requires $175 \times 10^9 \times 2 = 350$ GB for parameters alone. With 8-way tensor parallelism across one node: 350/8 = 43.75 GB per GPU for parameters. Adding gradients and optimizer states (even with ZeRO-1): need multiple nodes. Typical configuration for GPT-3 scale: 64–128 GPUs across 8–16 nodes.

**Communication topology**: Design the parallelism to match hardware topology. Never place tensor-parallel groups across nodes (too slow). Pipeline stage boundaries should align with node boundaries where possible.

### Q20: "Compare model distillation vs. quantization for reducing serving cost. When would you prefer each?"

**Answer**:

**Distillation** trains a new, smaller model to mimic the large model. It changes the model architecture (fewer layers, smaller hidden dimensions). Results: a fundamentally simpler model that is faster, uses less memory, and can be further quantized. The student model is trained from scratch or fine-tuned, which requires compute and training data. Accuracy loss is typically 2–5% but depends on task complexity and student size.

**Quantization** keeps the same architecture but reduces numerical precision (FP32 → INT8 → INT4). No retraining needed for PTQ (minutes), QAT requires a few epochs. INT8 provides ~4× speedup with <1% accuracy loss for most models. INT4 provides ~8× memory reduction but may lose 2–5% accuracy without careful calibration.

**When to prefer distillation**:
- When you need a >4× cost reduction and can tolerate retraining.
- When the large model is significantly over-parameterized for the task.
- When you want to change the architecture (e.g., from Transformer to a lighter architecture for edge deployment).
- When you'll deploy across diverse hardware that may not support INT4/INT8 well.

**When to prefer quantization**:
- When you need fast deployment without retraining.
- When accuracy preservation is critical (INT8 is nearly lossless).
- When the architecture must remain the same (compatibility with existing serving infrastructure).
- When combined with other optimizations (quantized model + batching + compilation).

**Best practice**: Combine both. Distill to a smaller model, then quantize the distilled model. A 70B model → distill to 7B → quantize to INT4: total ~40× cost reduction.
