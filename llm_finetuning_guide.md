# LLM Fine-Tuning, Alignment & Evaluation: A Complete Reference Guide

---

# Part I: Supervised Fine-Tuning (SFT)

---

## 1. Motivation & Intuition

### Why Does Supervised Fine-Tuning Exist?

Imagine you hire a brilliant polymath who has read the entire internet. They can write poetry, explain quantum mechanics, and discuss medieval history. But ask them to "be a helpful customer service agent for Acme Corp" and they ramble, hallucinate product names, or respond in the wrong format. The problem is not a lack of knowledge—it is a lack of *behavioral alignment* with a specific task.

This is exactly the situation with pretrained large language models (LLMs). During pretraining, a model like GPT or LLaMA learns to predict the next token on trillions of tokens of text scraped from the web. The resulting model understands language at a deep statistical level, but it has no concept of "being helpful," "following instructions," or "answering questions." If you give a pretrained model the prompt "What is the capital of France?", it might continue with "What is the capital of Germany? What is the capital of Spain?..." because on the internet, lists of quiz questions are common. It is completing text, not answering questions.

**Supervised Fine-Tuning (SFT)** is the process of taking this pretrained model and training it on a curated dataset of (input, desired output) pairs so that it learns to *behave* in a particular way. The "supervised" part means we have explicit ground-truth examples of what the model should produce.

### A Concrete Example Before Any Math

Suppose you want a model that, given a customer email, drafts a polite response. You would:

1. Collect 10,000 real customer emails.
2. Have human experts write ideal responses for each.
3. Format these into pairs: (customer email → ideal response).
4. Continue training the pretrained model on these pairs.

After SFT, the model has learned the *distribution* of responses your experts would write—it knows the tone, the format, the typical content. It has been behaviorally shaped.

### Connection to Real ML Systems

In the modern LLM pipeline, SFT sits between pretraining and alignment:

```
Pretraining (next-token prediction on web text)
    → SFT (learn to follow instructions from curated examples)
        → RLHF / DPO (learn human preferences for quality)
```

SFT is what transforms a raw text-completion engine into something that *feels* like an assistant. Without it, subsequent alignment stages (RLHF, DPO) have nothing coherent to refine.

---

## 2. Conceptual Foundations

### 2.1 Instruction Tuning: Formatting Datasets into (Prompt, Response) Pairs

The core data structure for SFT is the **(prompt, response)** pair, sometimes called an **(instruction, completion)** pair. The model learns: "Given this input, produce this output."

**Key terminology:**

- **Prompt (Instruction):** The input the user provides. This can be a bare question ("What is photosynthesis?"), an instruction with context ("Summarize the following article: [article text]"), or a multi-turn conversation history.
- **Response (Completion):** The desired output the model should generate. This is written by a human expert or a stronger model (distillation).
- **System prompt:** An optional prefix that sets the model's persona or behavioral constraints ("You are a helpful medical assistant. Always recommend consulting a doctor.").

**Dataset formats in practice:**

The most common format is the *chat template*, which structures data as a sequence of messages with roles:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a Python function to reverse a string."},
    {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}
  ]
}
```

The model is trained to predict only the **assistant** tokens. The system and user tokens are provided as context but their prediction loss is masked (set to zero). This is critical: if the model also learned to predict user messages, it would learn to *generate* user queries, not answer them.

**How the loss mask works:**

Given a sequence of tokens $[x_1, x_2, \ldots, x_n]$ where tokens $x_i$ through $x_j$ are the assistant's response, the training loss is:

$$\mathcal{L} = -\sum_{t=i}^{j} \log p_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

Tokens outside the range $[i, j]$ contribute zero to the gradient. The model still *sees* them (they flow through the forward pass and influence the hidden states), but it is only *trained to produce* the assistant portion.

**Dataset quality matters enormously.** A small dataset of 1,000 high-quality, diverse, expert-written examples often outperforms 100,000 noisy or templated examples. The seminal LIMA paper (2023) demonstrated that just 1,000 carefully curated examples could produce a strong instruction-following model, suggesting that SFT is primarily about teaching *format and style*, not injecting new knowledge.

### 2.2 Full Parameter Fine-Tuning vs. Memory Constraints

**Full parameter fine-tuning** means updating every single weight in the model during training. For a 7-billion-parameter model in float16 (2 bytes per parameter), just storing the model weights requires 14 GB of GPU memory. But training requires far more:

1. **Model weights:** 14 GB (for 7B params at fp16)
2. **Gradients:** 14 GB (one gradient value per parameter, same precision)
3. **Optimizer states:** For Adam, you need the first moment (mean) and second moment (variance) per parameter, stored in fp32. That is $7B \times 4 \text{ bytes} \times 2 = 56 \text{ GB}$.
4. **Activations:** These are the intermediate values saved during the forward pass for use in backpropagation. For long sequences and large batch sizes, this can be 10–50+ GB.

**Total for a 7B model with Adam:** roughly 14 + 14 + 56 + ~20 = **~104 GB**. A single A100 GPU has 80 GB. So even a 7B model does not fit on one GPU for full fine-tuning with Adam.

For a 70B model, multiply everything by 10. You need a cluster of 8+ GPUs with model parallelism.

**Strategies to reduce memory:**

- **Mixed-precision training:** Keep a master copy of weights in fp32, but do forward/backward passes in fp16/bf16. Reduces activation memory.
- **Gradient checkpointing:** Instead of storing all intermediate activations, recompute them during the backward pass. Trades ~30% more compute for ~60% less activation memory.
- **DeepSpeed ZeRO stages:** Partition optimizer states (ZeRO-1), gradients (ZeRO-2), or parameters (ZeRO-3) across multiple GPUs. ZeRO-3 means each GPU only stores 1/N of everything.
- **Parameter-Efficient Fine-Tuning (PEFT):** Do not update all parameters. Only train a small subset or a low-rank perturbation. (Covered in Part II.)

### 2.3 Catastrophic Forgetting

**What is it?**

Catastrophic forgetting occurs when fine-tuning a model on new data causes it to lose capabilities it had before fine-tuning. For example, you fine-tune a general-purpose LLM on medical question-answering data. After training, it answers medical questions well, but it can no longer write poetry, do math, or hold a general conversation. The new gradient updates have overwritten the weights that encoded those prior abilities.

**Why does it happen?**

Neural networks store knowledge *distributedly* across their weights. There is no clean separation between "the weights for poetry" and "the weights for medicine." When you aggressively update weights to minimize loss on medical data, you inevitably perturb the weight configurations that enabled other skills.

Formally, let $\theta_0$ be the pretrained weights and $\theta^*$ be the weights after fine-tuning on dataset $D_{\text{new}}$. If $\theta^*$ is far from $\theta_0$ in parameter space, then the model's behavior on tasks represented in the pretraining data but not in $D_{\text{new}}$ can degrade severely.

**Mitigation strategies:**

1. **Low learning rate:** Use a learning rate 10–100x smaller than pretraining (e.g., $1 \times 10^{-5}$ to $5 \times 10^{-5}$). This keeps $\theta^*$ close to $\theta_0$.

2. **Short training duration:** Fine-tune for only 1–3 epochs. More epochs increase the risk of overfitting to the fine-tuning distribution and forgetting the pretraining distribution.

3. **Data mixing:** Mix a fraction of general-purpose data into the fine-tuning dataset. For example, use 80% domain-specific data and 20% general instruction data. This provides a regularizing signal that preserves general capabilities.

4. **Regularization toward pretrained weights:** Add a penalty term:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda \|\theta - \theta_0\|_2^2$$
This elastic weight consolidation (EWC) variant keeps the fine-tuned weights close to the pretrained weights. More sophisticated versions weight each parameter by its Fisher information (how important it is for pretraining tasks).

5. **Parameter-efficient methods (LoRA, adapters):** By only modifying a small number of parameters (or a low-rank perturbation), the bulk of the pretrained weights remain untouched. This is one of the strongest practical defenses against catastrophic forgetting.

6. **Model merging:** After fine-tuning, interpolate between the pretrained and fine-tuned weights: $\theta_{\text{merged}} = \alpha \theta_0 + (1 - \alpha) \theta^*$. This trades some fine-tuning performance for better retention of general capabilities.

---

## 3. Mathematical Formulation

### The SFT Objective

Let a training example consist of a prompt $\mathbf{x} = (x_1, \ldots, x_m)$ and a response $\mathbf{y} = (y_1, \ldots, y_n)$. The SFT loss for this example is the negative log-likelihood of the response tokens, conditioned on the prompt and all preceding response tokens:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{n} \log p_\theta(y_t \mid x_1, \ldots, x_m, y_1, \ldots, y_{t-1})$$

Across a dataset $\mathcal{D} = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{n_i} \log p_\theta(y_t^{(i)} \mid \mathbf{x}^{(i)}, y_1^{(i)}, \ldots, y_{t-1}^{(i)})$$

This is identical to the pretraining objective (cross-entropy next-token prediction) except:
- The data distribution has changed from raw web text to curated (prompt, response) pairs.
- The loss is computed only over response tokens (the prompt tokens are masked).

**Why masking the prompt matters:**

If we include prompt tokens in the loss, the model receives gradient signal to predict the prompt itself. Since prompts come from users (who may write in many styles, ask varied questions, etc.), this signal is noisy and counterproductive. We only want the model to learn the mapping: prompt → response.

### Learning Rate Schedule

SFT typically uses a cosine decay schedule with warmup:

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \cdot \pi\right)\right) & \text{otherwise} \end{cases}$$

Typical values: $\eta_{\max} \in [1 \times 10^{-5}, 5 \times 10^{-5}]$, warmup for 3–10% of total steps.

---

## 4. Worked Example

### End-to-End SFT of a Small Model

**Setup:** We have a pretrained 1.3B parameter model and want to fine-tune it to be a helpful coding assistant.

**Step 1: Dataset preparation**

We collect 5,000 examples formatted as:
```
<|system|>You are a helpful coding assistant.<|end|>
<|user|>Write a function to check if a number is prime.<|end|>
<|assistant|>def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True<|end|>
```

**Step 2: Tokenization and masking**

The tokenizer converts this to token IDs. We create a loss mask:
- Tokens for `<|system|>...<|end|><|user|>...<|end|><|assistant|>` → mask = 0 (no loss)
- Tokens for `def is_prime(n):...True<|end|>` → mask = 1 (compute loss here)

**Step 3: Training configuration**
- Learning rate: $2 \times 10^{-5}$
- Batch size: 32 (with gradient accumulation if needed)
- Epochs: 3
- Optimizer: AdamW with weight decay 0.01
- Warmup: 100 steps
- Precision: bf16 mixed precision

**Step 4: Training loop (conceptual)**

For each batch:
1. Forward pass: compute logits for all tokens.
2. Compute cross-entropy loss only on masked positions.
3. Backward pass: compute gradients.
4. Optimizer step: update weights.
5. Log training loss and periodically evaluate on held-out validation set.

**Step 5: Evaluation**

After training, we evaluate on held-out coding problems (e.g., HumanEval benchmark). We measure pass@1 (does the model's first attempt solve the problem?) and compare against the base model.

**Expected results:** The base model might achieve pass@1 of 10% on HumanEval (it knows Python syntax but does not understand "solve this problem" as an instruction). After SFT, pass@1 might jump to 25–35% because the model now understands the task format.

---

## 5. Relevance to Machine Learning Practice

### Where SFT Fits in Modern LLM Pipelines

SFT is the bridge between pretraining and alignment. Every commercially deployed LLM (ChatGPT, Claude, Gemini) undergoes SFT as an essential stage.

**When to use SFT:**
- You want a model to follow instructions in a specific format (e.g., JSON output, specific tone).
- You are building a domain-specific assistant (medical, legal, customer service).
- You are distilling a larger model's capabilities into a smaller one (train the small model on outputs from the large model).

**When NOT to use SFT (or use alternatives):**
- You only need to change the model's output format slightly → prompt engineering or few-shot examples may suffice.
- You want the model to prefer certain responses over others → this is a *preference* problem better suited to RLHF/DPO (Part III).
- Your domain requires real-time knowledge updates → retrieval-augmented generation (RAG) is more appropriate than retraining.

**Trade-offs:**

| Consideration | Full SFT | PEFT (LoRA) | Prompt Engineering |
|---|---|---|---|
| Compute cost | Very high | Low–moderate | Near zero |
| Forgetting risk | High | Low | None |
| Expressiveness | Highest | High | Limited |
| Data requirement | 1K–100K examples | Same | 0–50 examples |
| Iteration speed | Hours–days | Minutes–hours | Seconds |

---

## 6. Common Pitfalls & Misconceptions

**Pitfall 1: "More data is always better."**
Noisy, low-quality, or overly templated data can *hurt* performance. A model trained on 100K examples generated by a weak LLM often underperforms one trained on 1K expert-curated examples. Quality dominates quantity for SFT.

**Pitfall 2: "SFT teaches the model new knowledge."**
SFT primarily teaches *behavior and format*, not facts. The factual knowledge comes from pretraining. If the pretrained model does not know something, SFT will not reliably teach it—instead, the model may learn to hallucinate confidently in the fine-tuning domain's style.

**Pitfall 3: "I should train for many epochs until the loss is very low."**
Overfitting to the SFT dataset is dangerous. The model memorizes specific phrasings rather than learning generalizable behaviors. Training loss near zero on a small dataset is a red flag. Typically 1–3 epochs is sufficient.

**Pitfall 4: "I forgot to mask the prompt tokens."**
If prompt tokens are included in the loss, the model receives a noisy training signal. In practice this degrades instruction-following quality and can cause the model to generate user-like text instead of assistant-like text.

**Pitfall 5: "My fine-tuned model is worse than the base model on general tasks."**
This is catastrophic forgetting. Solutions: lower the learning rate, train fewer epochs, mix in general data, or use PEFT methods.

---

## Interview Preparation: Supervised Fine-Tuning

### Foundational Questions

**Q1: What is the difference between pretraining and supervised fine-tuning?**

Pretraining trains a model on a massive, uncurated corpus (trillions of tokens from the web) using a self-supervised next-token prediction objective. The model learns general language understanding and world knowledge. SFT takes this pretrained model and continues training it on a curated dataset of (instruction, response) pairs, teaching the model to follow instructions and produce helpful outputs in a specific format. The key distinction is that pretraining learns *what language is*, while SFT learns *how to use language helpfully*.

**Q2: Why do we mask prompt tokens during SFT?**

The prompt represents user input—varied, unpredictable, and not something we want the model to generate. Including prompt tokens in the loss gives the model a gradient signal to predict what users will say, which is both noisy (users ask diverse questions) and counterproductive (we want the model to *respond* to prompts, not *generate* them). Masking ensures the model only receives learning signal for producing the desired assistant outputs.

**Q3: What is instruction tuning and how does it differ from task-specific fine-tuning?**

Task-specific fine-tuning trains a model on a single task (e.g., sentiment classification, translation). The model becomes a specialist. Instruction tuning trains on a *diverse mixture* of tasks, all formatted as natural language instructions. The model learns to generalize: given a new instruction it has never seen, it can often follow it correctly because it has learned the meta-skill of instruction-following. Instruction tuning produces a generalist; task-specific fine-tuning produces a specialist.

### Mathematical Questions

**Q4: Derive the gradient of the SFT loss with respect to the logits for a single token position.**

At position $t$, the model outputs logits $\mathbf{z}_t \in \mathbb{R}^{V}$ (where $V$ is vocabulary size). The softmax gives probabilities $p(v) = \frac{e^{z_v}}{\sum_{v'} e^{z_{v'}}}$. The loss for the correct token $y_t$ is $\ell_t = -\log p(y_t)$.

The gradient with respect to logit $z_v$ is:

$$\frac{\partial \ell_t}{\partial z_v} = p(v) - \mathbb{1}[v = y_t]$$

This is the softmax probability minus the one-hot target. Intuitively, the gradient pushes down the probability of all incorrect tokens (by an amount proportional to their current probability) and pushes up the probability of the correct token. Tokens the model already assigns high probability to receive small gradients (learning is "done" for those), while confident mistakes receive large corrective gradients.

**Q5: How does the effective batch size interact with the learning rate in SFT, and what is the linear scaling rule?**

When you increase the batch size by a factor $k$ (e.g., via gradient accumulation), each gradient estimate averages over $k$ times more examples, reducing its variance by a factor of $k$. This means you can take larger steps without increasing noise, so the learning rate can be scaled linearly: $\eta_{\text{new}} = k \cdot \eta_{\text{base}}$. This is the *linear scaling rule*.

However, this rule breaks down for very large batch sizes (when the gradient noise is already low, further variance reduction does not help) and at the very start of training (hence the need for warmup). In SFT specifically, batch sizes are typically moderate (32–128 effective examples), and the learning rate is already very low ($10^{-5}$), so the scaling is applied conservatively.

### Applied Questions

**Q6: You are fine-tuning a 70B model for customer service. You have 8 A100 (80GB) GPUs. Walk through your setup decisions.**

First, I would assess memory. A 70B model in bf16 is 140 GB for weights alone. With Adam optimizer states in fp32, that is another 560 GB. Total: ~700 GB minimum for full fine-tuning.

With 8 × 80 GB = 640 GB total GPU memory, full fine-tuning is borderline impossible. My options:

1. **Use PEFT (LoRA):** Train only ~0.1% of parameters. Optimizer states drop from 560 GB to ~0.5 GB. The frozen model fits in bf16 across 2 GPUs with pipeline parallelism, and LoRA adapters are tiny. This is the pragmatic choice.

2. **Use DeepSpeed ZeRO-3:** Shard everything across 8 GPUs. Each GPU holds 1/8 of weights, gradients, and optimizer states. This enables full fine-tuning but with communication overhead. Memory per GPU: ~700/8 ≈ 88 GB, tight but feasible with gradient checkpointing reducing activation memory.

3. **Quantize the base model (QLoRA):** Load the base model in 4-bit (NF4 quantization), reducing model memory from 140 GB to ~35 GB. Train LoRA adapters in bf16. Everything fits on 1–2 GPUs.

I would choose QLoRA or LoRA with DeepSpeed ZeRO-2 as the practical starting point, reserving full fine-tuning for experiments where PEFT quality is insufficient.

**Q7: You fine-tuned a model on 50K medical Q&A pairs. It now answers medical questions well but has lost the ability to do basic math. Diagnose and fix.**

This is classic catastrophic forgetting. The 50K medical examples contained no math content, so the model's math-relevant weights were overwritten by medical-domain gradients.

Diagnosis steps: (a) Compare perplexity on a math benchmark (e.g., GSM8K) before and after fine-tuning—expect a large increase. (b) Check if the learning rate was too high (above $5 \times 10^{-5}$) or training ran too many epochs.

Fixes, in order of least to most effort:
1. Reduce learning rate to $1 \times 10^{-5}$ and train for only 1 epoch.
2. Mix 20–30% general instruction-following data (including math) into the medical dataset.
3. Switch to LoRA, which preserves the frozen base model's math capabilities.
4. Apply EWC regularization: add a penalty term $\lambda \sum_i F_i (\theta_i - \theta_{0,i})^2$ weighted by Fisher information from the pretrained model.
5. After fine-tuning, merge: $\theta_{\text{final}} = 0.7 \cdot \theta_{\text{pretrained}} + 0.3 \cdot \theta_{\text{fine-tuned}}$ using linear interpolation (or SLERP/TIES merging).

### Debugging & Failure-Mode Questions

**Q8: Your SFT training loss is decreasing, but validation loss starts increasing after epoch 1. What is happening?**

This is overfitting. The model is memorizing the training examples rather than learning generalizable instruction-following behavior. Likely causes: the dataset is too small relative to the model, the learning rate is too high, or the data lacks diversity.

Actions: (a) Stop training at epoch 1 (early stopping). (b) Increase dataset size or diversity. (c) Add weight decay or dropout. (d) Reduce the learning rate. (e) Consider switching to PEFT, which has fewer trainable parameters and thus overfits more slowly.

**Q9: After SFT, the model generates correct content but in the wrong format (e.g., it includes the system prompt in its response, or it generates user turns). What went wrong?**

The chat template is likely incorrect. Common issues: (a) Special tokens (e.g., `<|assistant|>`, `<|end|>`) were not added to the tokenizer, so they were tokenized as individual characters instead of single tokens—the model cannot learn to start/stop at these boundaries. (b) The loss mask is not aligned with the actual assistant response boundaries. (c) The EOS (end-of-sequence) token is missing or misplaced, so the model does not learn when to stop.

Debug by: inspecting a few tokenized training examples and verifying that the mask is exactly 1 for assistant tokens and 0 for everything else. Also check that special tokens are in the tokenizer vocabulary and have consistent IDs.

### Follow-Up Probing Questions

**Q10: "You mentioned that SFT teaches format, not knowledge. Can you think of any exceptions?"**

Yes, there are partial exceptions. If the SFT dataset contains factual information presented in a consistent format that reinforces retrieval (e.g., "The capital of X is Y" formatted as Q&A), the model can strengthen its recall of those facts. Additionally, *chain-of-thought* SFT can teach reasoning *strategies* that unlock latent knowledge—the model "knew" the answer but could not reliably surface it without step-by-step reasoning patterns learned during SFT. However, truly novel facts not present in the pretraining corpus cannot be reliably injected via SFT; the model may memorize them for the training distribution but will not generalize.

---

# Part II: Parameter-Efficient Fine-Tuning (PEFT)

---

## 1. Motivation & Intuition

### The Problem PEFT Solves

Recall from Part I that full fine-tuning a 7B model requires ~100 GB of GPU memory. For a 70B model, it requires a cluster of high-end GPUs. This makes fine-tuning inaccessible to most practitioners and organizations.

But there is a deeper question: **do we actually need to change all 7 billion parameters?** Consider an analogy. If you hire a skilled writer and ask them to start writing in British English instead of American English, they do not need to relearn the entire English language. They only need to adjust a small number of habits: spelling conventions (color → colour), vocabulary choices (truck → lorry), and a few grammatical patterns. The vast majority of their writing ability remains unchanged.

The same intuition applies to neural networks. Research has shown that fine-tuning operates in a **low-dimensional subspace**: the meaningful changes to model behavior can be captured by modifying far fewer degrees of freedom than the total parameter count. This is the **intrinsic dimensionality** hypothesis.

**PEFT methods** exploit this by training only a small number of parameters (or a structured low-rank perturbation) while keeping the vast majority of the pretrained weights frozen. The benefits are:

1. **Memory efficiency:** Optimizer states are only maintained for the trainable parameters. If you train 0.1% of parameters, optimizer memory drops by 1000x.
2. **Storage efficiency:** You store one base model and many small adapter files (one per task). A LoRA adapter for a 7B model might be 10–50 MB, versus 14 GB for a full fine-tuned checkpoint.
3. **Reduced forgetting:** Since 99.9% of weights are frozen, the model's general capabilities are preserved by construction.
4. **Multi-task serving:** At inference time, you can hot-swap different LoRA adapters for different users/tasks on the same base model, enabling efficient multi-tenant serving.

---

## 2. LoRA (Low-Rank Adaptation)

### 2.1 Conceptual Foundations

**The key insight:** When you fine-tune a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ (e.g., a query projection in a transformer attention layer), the learned update $\Delta W = W_{\text{fine-tuned}} - W_0$ has low rank. That is, $\Delta W$ can be well-approximated by the product of two much smaller matrices.

Think of it this way: fine-tuning a model for a specific task makes coordinated, structured changes to the weights—not random perturbations in all directions. These coordinated changes lie in a low-dimensional subspace of the full parameter space.

**LoRA's approach:** Instead of learning $\Delta W$ directly (which has $d \times k$ parameters), decompose it as:

$$\Delta W = B \cdot A$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

The number of trainable parameters drops from $d \times k$ to $r \times (d + k)$. For a typical transformer layer with $d = k = 4096$ and $r = 8$:

- Full fine-tuning: $4096 \times 4096 = 16.7M$ parameters per matrix
- LoRA: $8 \times (4096 + 4096) = 65.5K$ parameters per matrix (a **255x** reduction)

### 2.2 How LoRA Works Step by Step

During the forward pass, the computation for a linear layer changes from:

$$h = W_0 x$$

to:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

The base weights $W_0$ are **frozen** (no gradients computed). Only $A$ and $B$ receive gradients and get updated by the optimizer.

**Initialization:**

- $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
- $B$ is initialized to **zero**.

This means that at the start of training, $\Delta W = BA = 0$, so the model begins with exactly the same behavior as the pretrained model. Training then gradually learns the necessary perturbation. This zero-initialization is crucial: it ensures a smooth, stable starting point.

**Scaling factor:** LoRA applies a scaling factor $\alpha / r$ to the output:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

where $\alpha$ is a hyperparameter (typically set to $r$, $2r$, or a fixed value like 16 or 32). This scaling stabilizes training across different rank choices: if you double $r$, the individual entries of $BA$ contribute half as much each, but the scaling compensates. In practice, $\alpha$ controls the effective learning rate of the LoRA parameters.

### 2.3 Which Layers to Adapt

In a transformer, each attention block has four projection matrices ($W_Q$, $W_K$, $W_V$, $W_O$) and each feedforward block has two ($W_{\text{up}}$, $W_{\text{down}}$). LoRA can be applied to any or all of these.

**Common practice:**
- **Minimum:** Apply to $W_Q$ and $W_V$ only. This is the configuration used in the original LoRA paper and works surprisingly well.
- **Recommended:** Apply to all attention projections ($W_Q$, $W_K$, $W_V$, $W_O$). Empirically better than Q+V alone.
- **Maximum:** Apply to all linear layers including feedforward. This gives the most expressivity but increases trainable parameter count.

### 2.4 Mathematical Formulation

**Forward pass with LoRA:**

For a single linear layer with input $x \in \mathbb{R}^k$:

$$h = W_0 x + \frac{\alpha}{r} B A x$$

where $W_0 \in \mathbb{R}^{d \times k}$ (frozen), $A \in \mathbb{R}^{r \times k}$ (trainable), $B \in \mathbb{R}^{d \times r}$ (trainable).

**Gradient computation for $A$:**

Let $\mathcal{L}$ be the loss. By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} B^\top \frac{\partial \mathcal{L}}{\partial h} x^\top$$

The gradient for $A$ depends on $B$ (and vice versa), creating an implicit coupling. Since $B$ is initialized to zero, early gradients for $A$ are zero—but $B$ receives nonzero gradients through $W_0$'s influence on the loss, which then bootstraps $A$'s learning in subsequent steps.

**Merging at inference time:**

After training, you can compute $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$ and replace the original weight matrix. The model then runs at exactly the same speed as the original—no additional latency. This is a key advantage of LoRA: the adapter adds zero inference cost once merged.

### 2.5 Worked Example

**Setup:** A projection matrix $W_Q \in \mathbb{R}^{4096 \times 4096}$ in a transformer attention layer. We apply LoRA with rank $r = 4$.

**Trainable parameters:**
- $A \in \mathbb{R}^{4 \times 4096}$: 16,384 parameters
- $B \in \mathbb{R}^{4096 \times 4}$: 16,384 parameters
- Total: 32,768 parameters (vs. 16.7M for the full matrix, a 512x reduction)

**Initialization:**
- $A_{ij} \sim \mathcal{N}(0, 0.01)$
- $B = 0_{4096 \times 4}$

**Forward pass for input $x \in \mathbb{R}^{4096}$:**
1. Compute frozen part: $h_0 = W_Q x$ (standard matrix-vector multiply)
2. Compute LoRA part: $z = Ax$ (gives $z \in \mathbb{R}^4$), then $\delta h = Bz$ (gives $\delta h \in \mathbb{R}^{4096}$)
3. Scale and add: $h = h_0 + \frac{\alpha}{r} \delta h$

The bottleneck dimension of 4 means the LoRA update can only modify $h$ within a 4-dimensional subspace. This is sufficient for many fine-tuning tasks, but if the task requires more diverse modifications (e.g., adapting to a very different domain), increasing $r$ to 16, 32, or 64 may be necessary.

---

## 3. QLoRA (Quantized LoRA)

### 3.1 Motivation

LoRA reduces trainable parameter memory, but the **frozen base model** still occupies significant GPU memory. A 70B model in bf16 requires 140 GB just for the frozen weights. QLoRA addresses this by quantizing the frozen base model to 4-bit precision, reducing memory by ~4x.

### 3.2 Conceptual Foundations

QLoRA combines three innovations:

**Innovation 1: 4-bit NormalFloat (NF4) Quantization**

Pretrained model weights are approximately normally distributed (empirically verified). Standard 4-bit quantization uses uniformly spaced levels (e.g., -7, -6, ..., 0, ..., 6, 7), which wastes representational capacity on the tails where few weights exist.

NF4 uses *information-theoretically optimal* quantization levels for a normal distribution. The levels are chosen such that each quantization bin contains an equal fraction of the probability mass under a standard normal. This means more levels are concentrated near zero (where most weights are) and fewer in the tails.

Concretely, the 16 quantization levels (for 4-bit, we have $2^4 = 16$ levels) are the quantiles of $\mathcal{N}(0, 1)$ at equal probability intervals:

$$q_i = \Phi^{-1}\left(\frac{i + 0.5}{16}\right) \quad \text{for } i = 0, 1, \ldots, 15$$

where $\Phi^{-1}$ is the inverse CDF of the standard normal. Before quantization, each weight block is normalized by its absolute maximum, mapping weights to approximately $[-1, 1]$, and then each weight is rounded to the nearest NF4 level.

**Innovation 2: Double Quantization**

Each block of weights (typically 64 weights) has a quantization constant (the scaling factor used to normalize the block). With millions of blocks, these constants themselves consume significant memory—about 0.5 bits per parameter at 64-block size.

Double quantization quantizes these scaling factors *again*, using standard 8-bit quantization, grouped into larger blocks of 256. This reduces the overhead from 0.5 bits/param to ~0.127 bits/param.

**Innovation 3: Paged Optimizers**

During training, GPU memory usage can spike during gradient computation. If the spike exceeds available GPU memory, training crashes with an out-of-memory (OOM) error. Paged optimizers use NVIDIA's unified memory feature to automatically page optimizer states between GPU and CPU memory. When GPU memory is full, optimizer states for some parameters are temporarily moved to CPU RAM, then brought back when needed. This introduces some overhead but prevents OOM crashes.

### 3.3 QLoRA Training Pipeline

1. Load the pretrained model in NF4 quantization (4 bits per parameter).
2. Attach LoRA adapters ($A$ and $B$ matrices) in bf16/fp16 to selected layers.
3. During the forward pass: dequantize frozen weights to bf16 on-the-fly, compute the layer output, add the LoRA contribution.
4. During the backward pass: compute gradients only for the LoRA parameters (the frozen NF4 weights receive no gradients).
5. Update LoRA parameters using the optimizer (with paging if needed).

**Memory comparison for a 65B model:**

| Component | Full FT (bf16) | LoRA (bf16) | QLoRA (NF4 + bf16 adapters) |
|---|---|---|---|
| Base model | 130 GB | 130 GB | ~33 GB |
| Trainable params | — | ~0.1 GB | ~0.1 GB |
| Optimizer states | ~520 GB | ~0.4 GB | ~0.4 GB |
| **Total (approx.)** | **~700 GB** | **~135 GB** | **~38 GB** |

QLoRA enables fine-tuning a 65B model on a single 48GB GPU.

### 3.4 Does Quantization Hurt Quality?

The key finding from the QLoRA paper is that **NF4 quantization introduces negligible degradation**. The information-theoretically optimal level placement means the quantization error is nearly as low as possible for 4 bits. The LoRA adapters, trained in full precision (bf16), are expressive enough to compensate for any small quantization artifacts.

In practice, QLoRA models match or come within 1-2% of full-precision LoRA models on standard benchmarks (MMLU, HumanEval), while using 3–4x less memory.

---

## 4. Prefix Tuning & Prompt Tuning

### 4.1 Conceptual Foundations

**Prefix Tuning** and **Prompt Tuning** take a different approach from LoRA. Instead of modifying the model's weights, they modify the model's *input*.

**Prompt Tuning (the simpler version):**

Add a sequence of $k$ trainable "virtual tokens" (continuous embedding vectors) to the beginning of every input. These vectors are not constrained to correspond to any real token in the vocabulary—they are free-floating vectors in embedding space that the optimizer learns to set to values that steer the model's behavior.

Concretely, let the normal input embeddings be $[e_1, e_2, \ldots, e_n] \in \mathbb{R}^{n \times d}$. Prompt tuning prepends $k$ trainable vectors:

$$[\underbrace{p_1, p_2, \ldots, p_k}_{\text{learnable prefix}}, e_1, e_2, \ldots, e_n]$$

The model processes this extended sequence normally. Only $\{p_1, \ldots, p_k\}$ are trainable; all model weights are frozen.

**Number of trainable parameters:** $k \times d$. For $k = 20$ virtual tokens and $d = 4096$, that is only 81,920 parameters—orders of magnitude fewer than even LoRA.

**Prefix Tuning (the more powerful version):**

Instead of only prepending to the input embedding layer, prefix tuning adds trainable key-value pairs to *every attention layer*. In each layer $l$, the attention computation becomes:

$$\text{Attention}(Q, [P_K^{(l)}; K], [P_V^{(l)}; V])$$

where $P_K^{(l)}, P_V^{(l)} \in \mathbb{R}^{k \times d_{\text{head}}}$ are trainable prefix keys and values for layer $l$.

This is more expressive than prompt tuning because it directly injects information into every layer's attention computation, rather than relying on the information propagating from the input layer through the network.

### 4.2 Assumptions and Limitations

**Assumption:** The model's behavior can be sufficiently steered by modifying the context it attends to, without changing how it processes that context. This is a strong assumption and holds well for stylistic or format changes but can break for tasks requiring fundamentally different computations.

**When it breaks:**
- Tasks requiring the model to process information differently (not just attend to different things). For example, if the pretrained model never learned a specific algorithm, no amount of prefix tuning will teach it.
- Prefix tuning uses some of the model's finite context window for virtual tokens, reducing the space available for actual input. For context-limited models, this trade-off matters.

### 4.3 Comparison with LoRA

| Aspect | LoRA | Prefix Tuning | Prompt Tuning |
|---|---|---|---|
| What changes | Weight matrices | Attention context | Input embeddings |
| Inference overhead | Zero (after merge) | Small (extra KV pairs) | Negligible |
| Expressiveness | High | Moderate | Lower |
| Context cost | None | $k$ tokens per layer | $k$ tokens total |
| Typical param count | ~0.1% of model | ~0.01% | ~0.001% |

---

## 5. Adapters

### 5.1 Conceptual Foundations

**Adapters** insert small, trainable bottleneck layers between existing transformer blocks. The original model weights are frozen; only the adapter parameters are trained.

An adapter module typically consists of:
1. A **down-projection**: $W_{\text{down}} \in \mathbb{R}^{m \times d}$ (projects from model dimension $d$ to bottleneck dimension $m$, where $m \ll d$)
2. A **nonlinearity**: typically ReLU or GELU
3. An **up-projection**: $W_{\text{up}} \in \mathbb{R}^{d \times m}$
4. A **residual connection**: the output is the adapter's output added to the input

Mathematically:
$$h = x + W_{\text{up}} \cdot \sigma(W_{\text{down}} \cdot x)$$

where $\sigma$ is the activation function.

**Placement:** Adapters are typically inserted after the feedforward sublayer in each transformer block (in series with the residual stream). Some variants also add adapters after the attention sublayer.

**Trainable parameters per adapter:** $2 \times m \times d + m + d$ (the two projection matrices plus biases). With $m = 64$ and $d = 4096$, each adapter has about 525K parameters.

### 5.2 Adapter vs. LoRA: Key Differences

| Aspect | Adapters | LoRA |
|---|---|---|
| Architecture | Adds new layers (sequential bottleneck) | Modifies existing layers (parallel low-rank) |
| Inference cost | Small overhead (extra computation) | Zero (after weight merging) |
| Nonlinearity | Yes (ReLU/GELU in bottleneck) | No (purely linear) |
| Merging | Cannot merge into base weights | Can merge into base weights |

The nonlinearity in adapters gives them more expressive power per parameter than LoRA (which is purely linear). However, the inability to merge adapters into the base weights means they always add latency at inference time. In production settings with strict latency requirements, LoRA's mergeability is a significant advantage.

---

## 6. PEFT: Relevance to ML Practice

### When to Use Which PEFT Method

**LoRA** is the default recommendation for most practical fine-tuning tasks. It offers the best trade-off between expressiveness, memory savings, and inference efficiency (zero added latency after merging). Use it when:
- You need to fine-tune models of 7B+ parameters.
- You want to maintain multiple task-specific models cheaply (store base model once + multiple small LoRA files).
- Inference latency is critical (merge LoRA into base weights).

**QLoRA** extends LoRA for extreme memory constraints. Use it when:
- You want to fine-tune a 30B–70B model on a single consumer/professional GPU.
- Training throughput is less important than accessibility (QLoRA is ~30% slower than LoRA due to dequantization overhead).

**Prefix/Prompt Tuning** is suitable when:
- Parameter budget is extremely tight.
- The task primarily involves stylistic or format changes (not deep behavioral shifts).
- You are working in a multi-tenant inference setting and want to dynamically switch tasks by swapping prefix vectors (no weight reloading needed).

**Adapters** are historically significant but have been largely supplanted by LoRA in practice. They may still be preferable when:
- The adaptation requires nonlinear transformations (rare).
- You are working in a research context exploring the expressiveness-efficiency frontier.

---

## 7. PEFT: Common Pitfalls & Misconceptions

**Pitfall 1: "LoRA rank should be as high as possible for best quality."**
Higher rank increases expressiveness but also increases overfitting risk, memory usage, and training time. For most tasks, $r = 8$ to $r = 32$ is sufficient. Rank 64+ should only be used with large, diverse datasets.

**Pitfall 2: "I applied LoRA only to the attention layers. My fine-tuned model underperforms."**
For tasks requiring significant behavioral change (e.g., domain adaptation, learning a new output format), applying LoRA to feedforward layers as well can substantially improve results. The feedforward layers store a large portion of the model's factual and procedural knowledge.

**Pitfall 3: "QLoRA quality is always identical to full-precision LoRA."**
While NF4 quantization is remarkably good, some information is lost. For tasks requiring extreme precision (e.g., highly numerical or scientific domains), full-precision LoRA may yield measurably better results. Always benchmark.

**Pitfall 4: "Prompt tuning will work as well as LoRA for my complex task."**
Prompt tuning has strictly less expressiveness than LoRA. For simple tasks (classification, format control), it works well. For complex tasks (multi-step reasoning, domain adaptation), it typically underperforms LoRA significantly.

**Pitfall 5: "I forgot to set LoRA alpha correctly."**
If $\alpha$ is too low, the LoRA contribution is negligible and fine-tuning has little effect. If too high, training is unstable. A common starting point is $\alpha = 2r$. The effective LoRA learning rate is proportional to $\alpha / r$.

---

## Interview Preparation: PEFT

### Foundational Questions

**Q1: Explain the intrinsic dimensionality hypothesis and how it motivates LoRA.**

The intrinsic dimensionality hypothesis states that, for a given fine-tuning task, the meaningful weight updates lie in a low-dimensional subspace of the full parameter space. In other words, although a model has billions of parameters, the actual "directions" that matter for adapting it to a new task can be captured by a much smaller number of free variables. LoRA operationalizes this by constraining weight updates to be low-rank matrices $\Delta W = BA$ where $r \ll \min(d, k)$. The rank $r$ is a proxy for the intrinsic dimensionality. Empirical evidence supports this: LoRA with $r = 8$ often matches full fine-tuning on standard benchmarks, suggesting the intrinsic dimensionality for many tasks is indeed very low.

**Q2: How does LoRA maintain zero additional inference latency?**

After training, the LoRA weight update $\frac{\alpha}{r} BA$ can be computed once and added to the original weight matrix: $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$. The merged weight matrix has the same shape as $W_0$, so inference uses the exact same architecture and computation as the original model. There are no extra layers, no extra multiplications—just a different weight matrix. This contrasts with adapters, which add sequential bottleneck layers that must be computed during every forward pass.

**Q3: What is NF4 quantization and why is it better than standard 4-bit quantization for LLM weights?**

Standard 4-bit quantization uses uniformly spaced levels across the range of weight values. Since LLM weight distributions are approximately Gaussian (concentrated around zero with thin tails), uniform spacing wastes levels on low-density tail regions. NF4 places quantization levels at equal-probability quantiles of the standard normal distribution, concentrating levels where the density is highest (near zero). This minimizes expected quantization error for normally-distributed values, achieving information-theoretically optimal quantization for 4-bit precision under the Gaussian assumption.

### Mathematical Questions

**Q4: For a weight matrix $W_0 \in \mathbb{R}^{4096 \times 4096}$ with LoRA rank $r=16$ and $\alpha=32$, compute the total trainable parameters, the compression ratio, and describe the gradient flow.**

Trainable parameters:
- $A \in \mathbb{R}^{16 \times 4096}$: $16 \times 4096 = 65{,}536$ parameters
- $B \in \mathbb{R}^{4096 \times 16}$: $65{,}536$ parameters
- Total: $131{,}072$ trainable parameters

Compression ratio: $\frac{4096 \times 4096}{131{,}072} = \frac{16{,}777{,}216}{131{,}072} = 128\times$

Gradient flow: The forward pass computes $h = W_0 x + \frac{32}{16} BAx = W_0 x + 2BAx$. The gradient of the loss $\mathcal{L}$ with respect to $B$ is $\frac{\partial \mathcal{L}}{\partial B} = 2 \frac{\partial \mathcal{L}}{\partial h} (Ax)^\top$ and with respect to $A$ is $\frac{\partial \mathcal{L}}{\partial A} = 2 B^\top \frac{\partial \mathcal{L}}{\partial h} x^\top$. The gradients for $A$ and $B$ are coupled—each depends on the other's current value. This is analogous to the gradient structure in matrix factorization problems.

**Q5: Prove that the rank of $\Delta W = BA$ is at most $r$.**

By the rank inequality for matrix products: $\text{rank}(BA) \leq \min(\text{rank}(B), \text{rank}(A))$. Since $B \in \mathbb{R}^{d \times r}$, $\text{rank}(B) \leq r$. Since $A \in \mathbb{R}^{r \times k}$, $\text{rank}(A) \leq r$. Therefore $\text{rank}(\Delta W) = \text{rank}(BA) \leq r$. The update $\Delta W$ lies in at most an $r$-dimensional subspace of $\mathbb{R}^{d \times k}$, precisely encoding the low-intrinsic-dimensionality hypothesis.

### Applied Questions

**Q6: You are serving a 70B model to 100 different enterprise clients, each needing a different customization. How do you architect this?**

Store one base model in 4-bit quantization (~35 GB). Each client has their own LoRA adapter (~50 MB each, totaling ~5 GB for 100 clients). At inference time:

Option A (batch-homogeneous): When a request arrives, load the client's LoRA adapter, merge it into the base weights, and serve. Adapter hot-swapping takes <1 second.

Option B (batch-heterogeneous, using S-LoRA or Punica): Serve multiple clients simultaneously by fusing different LoRA computations in a single batched kernel. The base model forward pass is shared, and each request gets its own $BAx$ added. This enables high throughput with heterogeneous LoRA adapters in a single batch.

The key insight is that the base model (the expensive part) is shared, and only the small adapters differentiate clients. This is dramatically more efficient than storing 100 separate fine-tuned models (100 × 140 GB = 14 TB).

**Q7: You trained a LoRA adapter with $r=4$ and the model's fine-tuned performance is poor. How do you diagnose whether to increase rank vs. look elsewhere?**

Diagnostic steps:

1. **Check training loss:** If training loss is high and not converging, the model may lack expressiveness → increase $r$.
2. **Check validation loss:** If training loss is low but validation loss is high, you are overfitting the current rank → increasing $r$ will make it worse. Instead, add more data or increase regularization.
3. **Compare with $r=16$ and $r=64$:** If increasing rank substantially improves validation performance, the task genuinely needs higher intrinsic dimensionality.
4. **Check which layers have LoRA:** If only Q and V, try adding K, O, and feedforward layers before increasing rank.
5. **Check data quality:** Poor results often stem from noisy or insufficient training data, not insufficient rank.

In my experience, rank is rarely the bottleneck below $r=16$. Data quality, learning rate, and target layer selection are more common culprits.

### Debugging & Failure-Mode Questions

**Q8: After merging LoRA weights into the base model, the merged model produces garbage outputs. The un-merged model with LoRA applied dynamically works fine. What went wrong?**

Common cause: the scaling factor was applied incorrectly during merging. The merge should compute $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$, but if the scaling was omitted or doubled (e.g., applied both during training and during merging), the weights are corrupted.

Debugging: (a) Check that the merging code divides by $r$ and multiplies by $\alpha$. (b) Verify that the LoRA library's training code does not already embed the scaling into $A$ or $B$. (c) Test by manually computing $\frac{\alpha}{r} BA$ for one layer and comparing against the merge tool's output.

Another possible cause: data type mismatch. If $W_0$ is in bf16 and $BA$ is computed in fp32, the addition may introduce precision issues, especially if the library silently truncates.

---

# Part III: Preference Alignment (RLHF & Beyond)

---

## 1. Motivation & Intuition

### Why SFT Is Not Enough

After SFT, a model can follow instructions and produce coherent outputs. But SFT has a fundamental limitation: it treats all correct answers as equally good. In reality, there is a spectrum of quality.

Consider asking a model to explain quantum entanglement. Many explanations are factually correct, but some are clearer, more engaging, better structured, or more appropriate for the audience. SFT can teach the model to *produce* explanations, but it cannot teach it to distinguish a *good* explanation from a *great* one—unless you have a separate example for every quality level, which is impractical.

**The core problem:** How do you teach a model *preferences*? How do you make it produce outputs that humans would choose over alternatives?

Here is a concrete analogy. Imagine training a chef (SFT: teaching them to follow recipes) versus teaching them *taste* (alignment: teaching them to create dishes that diners prefer). You cannot write a recipe for "taste"—it is a holistic judgment that emerges from comparing options.

**Preference alignment** methods solve this by using **comparative feedback**: "Response A is better than Response B for this prompt." This relative signal is much easier for humans to provide than absolute quality scores, and it captures nuanced preferences that are hard to specify explicitly.

The standard pipeline after SFT:

```
SFT Model (follows instructions)
    → Collect comparison data (human judges rank responses)
        → Train a reward model (learns to predict which response humans prefer)
            → Optimize the SFT model to produce high-reward responses (RLHF)
```

Or, more recently:

```
SFT Model → Collect comparison data → Directly optimize preferences (DPO)
```

---

## 2. Reward Modeling

### 2.1 Conceptual Foundations

A **reward model** is a neural network that takes a (prompt, response) pair and outputs a scalar score indicating how "good" the response is. It learns this from human comparison data.

**The Bradley-Terry Model:**

Given a prompt $x$ and two responses $y_w$ (the one humans preferred, the "winner") and $y_l$ (the one they rejected, the "loser"), the Bradley-Terry model defines the probability that $y_w$ is preferred:

$$P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

where $r_\phi$ is the reward model (a neural network with parameters $\phi$) and $\sigma$ is the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.

**Why this form?** The Bradley-Terry model is a classical model from psychometrics for paired comparisons. It assumes each option has a "strength" (the reward), and the probability of one being preferred is a function of the strength *difference*. The sigmoid maps the real-valued difference to a probability in $(0, 1)$.

**Training objective:** Given a dataset $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$ of comparisons, the reward model is trained to maximize:

$$\mathcal{L}_{\text{RM}}(\phi) = -\frac{1}{N}\sum_{i=1}^{N} \log \sigma(r_\phi(x^{(i)}, y_w^{(i)}) - r_\phi(x^{(i)}, y_l^{(i)}))$$

This is binary cross-entropy loss: the reward model is trained so that preferred responses get higher rewards than rejected ones.

**Architecture:** Typically, the reward model is initialized from the SFT model. The language modeling head (which outputs logits over the vocabulary) is replaced with a single linear layer that outputs a scalar. The reward for a response is the scalar output at the last token position.

### 2.2 Assumptions and What Breaks

**Assumption 1: Transitivity.** If $A \succ B$ and $B \succ C$, then $A \succ C$. The Bradley-Terry model bakes this in (since rewards are real numbers, they are totally ordered). But human preferences can be intransitive: a clear, short response might be preferred over a detailed one, and the detailed one over a humorous one, but the humorous one over the clear, short one. This creates noise that the reward model cannot perfectly capture.

**Assumption 2: Context independence.** The reward for a response depends only on the prompt and response, not on what other responses were available. In practice, humans' quality judgments can be influenced by the comparison set (anchoring effects).

**Assumption 3: Human agreement.** The model assumes there is a single underlying preference ordering. In reality, different annotators may have genuinely different preferences (e.g., conciseness vs. thoroughness). This inter-annotator disagreement becomes noise in the training signal.

**When these break:** The reward model's accuracy degrades on the margins—it captures the broad strokes of human preferences well but can be unreliable for subtle quality distinctions. This is why reward models are used as optimization signals (to push the model in a generally good direction), not as absolute quality judges.

### 2.3 Reward Hacking

A critical failure mode: the policy model finds inputs that score highly according to the reward model but are not genuinely good. This is **reward hacking** (or reward overoptimization).

Example: the reward model might learn that longer responses score higher (because annotators often preferred more detailed answers). The policy model then learns to generate excessively verbose responses that score high on the reward model but annoy actual users.

This is why RLHF includes a KL penalty (discussed next) to prevent the policy from deviating too far from the SFT model.

---

## 3. PPO (Proximal Policy Optimization) for RLHF

### 3.1 Conceptual Foundations

RLHF frames language generation as a reinforcement learning problem:
- **State:** The prompt $x$ and all tokens generated so far.
- **Action:** The next token to generate.
- **Policy:** The language model $\pi_\theta(y \mid x)$—the probability distribution over responses given a prompt.
- **Reward:** The reward model score $r_\phi(x, y)$ for the complete response $y$, minus a KL penalty.

PPO is an RL algorithm that updates the policy to produce higher-reward responses while staying close to its previous behavior.

### 3.2 The RLHF Objective

The objective is to maximize the expected reward while penalizing divergence from a reference policy $\pi_{\text{ref}}$ (typically the SFT model):

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \text{KL}(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]$$

**Breaking this down:**

- $r_\phi(x, y)$: The reward model's score for the response. We want to maximize this (produce responses the reward model rates highly).
- $\beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$: A penalty for how much the policy $\pi_\theta$ has diverged from the reference policy $\pi_{\text{ref}}$. This prevents reward hacking and catastrophic forgetting.
- $\beta$: Controls the trade-off. High $\beta$ → policy stays close to SFT (conservative). Low $\beta$ → policy optimizes reward aggressively (risk of hacking).

**Why the KL penalty is critical:**

Without the KL term, the policy will eventually find and exploit artifacts in the reward model. The KL term acts as a regularizer: it says "you can improve, but not by abandoning everything you learned in SFT." In practice, $\beta$ is often dynamically adjusted: if the KL divergence grows too large, $\beta$ is increased; if it stays too small, $\beta$ is decreased.

### 3.3 The PPO Algorithm in RLHF

PPO requires two models during training:

1. **Policy model** $\pi_\theta$: The model being optimized (starts from SFT checkpoint).
2. **Value model** (critic) $V_\psi(x, y_{1:t})$: Estimates the expected future reward from a given state (prompt + partial response). Used to compute advantages.

Additionally, two models are held frozen:
3. **Reference policy** $\pi_{\text{ref}}$: A frozen copy of the SFT model, used to compute KL divergence.
4. **Reward model** $r_\phi$: The trained reward model from the previous stage.

**Total memory:** 4 full-size models must be loaded simultaneously. For a 7B parameter model, this means ~56 GB for model weights alone (at bf16), plus optimizer states for the policy and value models. This is why RLHF is extremely memory-intensive.

**The PPO training loop (one iteration):**

1. **Sample prompts** from the dataset.
2. **Generate responses** using the current policy $\pi_\theta$.
3. **Score responses** using the reward model $r_\phi(x, y)$.
4. **Compute KL penalty:** For each token, compute $\log \pi_\theta(y_t | \cdot) - \log \pi_{\text{ref}}(y_t | \cdot)$.
5. **Compute advantages** using the value model (how much better this action was than expected).
6. **Compute the PPO clipped objective:**

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \; \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$

where $\hat{A}_t$ is the estimated advantage at token $t$ and $\epsilon$ (typically 0.2) is the clipping parameter.

**What clipping does:** The ratio $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ measures how much the policy has changed from the previous iteration. If this ratio gets too large (the policy has changed a lot), the clip function caps it, preventing excessively large updates that could destabilize training. This is the "proximal" in PPO—updates are kept close to the current policy.

7. **Update the value model** to better predict future rewards (standard value function regression).
8. **Update the policy** using the PPO objective.

### 3.4 Practical Challenges

**Instability:** PPO training for LLMs is notoriously unstable. The optimization involves a complex interplay between the policy, value model, reward model, and reference policy. Small hyperparameter changes can cause divergence or collapse.

**Compute cost:** Four models in memory, plus generation at every iteration (which is autoregressive and slow), plus multiple forward/backward passes per iteration. RLHF with PPO is roughly 10–20x more expensive per training example than SFT.

**Hyperparameter sensitivity:** $\beta$ (KL coefficient), $\epsilon$ (clip range), learning rate, batch size, number of PPO epochs per batch, and generation temperature all interact in complex ways.

These challenges motivated simpler alternatives like DPO.

---

## 4. DPO (Direct Preference Optimization)

### 4.1 Motivation

DPO asks: **Can we skip the reward model and the RL training loop entirely?** The answer is yes, by deriving a closed-form relationship between the optimal policy and the preference data.

### 4.2 The Key Mathematical Insight

Start from the RLHF objective:

$$\max_\theta \mathbb{E}_{x, y \sim \pi_\theta} \left[ r(x, y) - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

It can be shown that the **optimal policy** for this objective has a closed-form solution:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ is the partition function (normalization constant).

Rearranging to solve for the reward:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

Now substitute this into the Bradley-Terry preference model. For a preferred response $y_w$ and rejected response $y_l$:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Substituting the expression for $r$:

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

Notice that $\beta \log Z(x)$ cancels out (it appears in both terms and is subtracted away). This is crucial: the intractable partition function disappears.

### 4.3 The DPO Loss

Replace $\pi^*$ with our parameterized policy $\pi_\theta$ and maximize the log-likelihood of the observed preferences:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right) \right]$$

**What this loss does intuitively:**

Define the "implicit reward margin" as:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

This is the log-probability ratio between the current policy and the reference, scaled by $\beta$. The DPO loss pushes this margin to be positive for preferred responses and negative for rejected responses.

**In plain English:** DPO increases the probability of preferred responses (relative to the reference policy) and decreases the probability of rejected responses. The $\beta$ parameter controls how aggressively it does this—higher $\beta$ means more conservative updates (the policy stays closer to the reference).

### 4.4 DPO vs. PPO: Practical Comparison

| Aspect | PPO (RLHF) | DPO |
|---|---|---|
| Models needed | 4 (policy, value, reward, reference) | 2 (policy, reference) |
| Training complexity | RL loop with generation | Standard supervised training |
| Memory | ~4x model size | ~2x model size |
| Stability | Notoriously unstable | Relatively stable |
| Compute | 10–20x SFT cost | 2–3x SFT cost |
| Reward model needed | Yes | No |
| Proven at massive scale | Yes (GPT-4, Claude) | Growing evidence |

**DPO's limitation:** Because there is no explicit reward model, you cannot use the reward for purposes other than training (e.g., rejection sampling at inference, reward-guided search, or monitoring reward trends). PPO's explicit reward model is a versatile asset.

### 4.5 Worked Example

**Preference data:**
```
Prompt: "Explain gravity to a 5-year-old."
Preferred (y_w): "Gravity is like an invisible magnet in the Earth that pulls 
                  everything toward the ground. That's why when you throw a ball 
                  up, it comes back down!"
Rejected (y_l):  "Gravity is a fundamental force described by Einstein's general 
                  theory of relativity, involving the curvature of spacetime by 
                  massive objects."
```

**DPO training step:**

1. Compute $\log \pi_\theta(y_w | x)$: Run the policy model on the prompt + preferred response, sum the log-probabilities of each response token. Say this equals $-15.3$.

2. Compute $\log \pi_{\text{ref}}(y_w | x)$: Same computation with the frozen reference model. Say this equals $-16.1$.

3. Compute $\log \pi_\theta(y_l | x)$: Policy model on prompt + rejected response. Say this equals $-22.7$.

4. Compute $\log \pi_{\text{ref}}(y_l | x)$: Reference model on rejected response. Say this equals $-21.5$.

5. Compute the margin:
$$\Delta = \beta \left[ (\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)) - (\log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x)) \right]$$
$$= \beta \left[ (-15.3 - (-16.1)) - (-22.7 - (-21.5)) \right]$$
$$= \beta \left[ 0.8 - (-1.2) \right] = \beta \cdot 2.0$$

With $\beta = 0.1$: $\Delta = 0.2$

6. Loss contribution: $-\log \sigma(0.2) = -\log(0.5498) = 0.599$

The gradient will push the policy to increase $\pi_\theta(y_w|x)$ and decrease $\pi_\theta(y_l|x)$, making the margin larger until the sigmoid saturates near 1 (meaning the model strongly prefers the preferred response).

---

## 5. KTO & ORPO

### 5.1 KTO (Kahneman-Tversky Optimization)

**Motivation:** Both RLHF and DPO require *paired* preference data: for each prompt, you need both a preferred and rejected response. In practice, it is often easier to collect *unpaired* signals: individual responses labeled as "good" (thumbs up) or "bad" (thumbs down), without requiring that good and bad responses be generated for the same prompt.

KTO draws on Kahneman and Tversky's Prospect Theory from behavioral economics. Prospect Theory says that humans evaluate outcomes relative to a reference point, and losses loom larger than equivalent gains. KTO applies this to alignment:

**The KTO loss:**

For a "desirable" response $y_d$ (thumbs up):
$$\mathcal{L}_{\text{KTO}}^{+} = 1 - \sigma\left(\beta \log \frac{\pi_\theta(y_d | x)}{\pi_{\text{ref}}(y_d | x)} - z_{\text{ref}}\right)$$

For an "undesirable" response $y_u$ (thumbs down):
$$\mathcal{L}_{\text{KTO}}^{-} = 1 - \sigma\left(z_{\text{ref}} - \beta \log \frac{\pi_\theta(y_u | x)}{\pi_{\text{ref}}(y_u | x)}\right)$$

where $z_{\text{ref}} = \mathbb{E}_{x'}\left[\beta \text{KL}(\pi_\theta(\cdot|x') \| \pi_{\text{ref}}(\cdot|x'))\right]$ is the expected KL divergence across the dataset, acting as the "reference point."

**Intuition:** For good responses, the loss decreases when the policy assigns higher probability (relative to reference) than the average KL divergence. For bad responses, the loss decreases when the policy assigns lower probability. The asymmetry in the sigmoid arguments reflects loss aversion: the gradient for avoiding bad responses is stronger than for selecting good responses.

**Key advantage:** KTO only needs binary feedback per response, not paired comparisons. This dramatically simplifies data collection.

### 5.2 ORPO (Odds Ratio Preference Optimization)

**Motivation:** All methods above require a reference model $\pi_{\text{ref}}$, which must be loaded into memory alongside the policy. Can we eliminate this requirement?

ORPO combines SFT and preference alignment into a single training objective, requiring no reference model and no separate alignment stage.

**The ORPO loss:**

$$\mathcal{L}_{\text{ORPO}} = \underbrace{\mathcal{L}_{\text{SFT}}(y_w)}_{\text{learn to generate preferred response}} + \lambda \cdot \underbrace{\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)}_{\text{prefer } y_w \text{ over } y_l}$$

where $\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$ and $P_\theta(y|x)$ is the average token probability of the response.

**Intuition:** The first term (SFT loss) ensures the model learns to generate preferred responses. The second term (odds ratio) ensures the model assigns a higher odds ratio to preferred responses compared to rejected ones. The log odds ratio naturally separates good from bad responses without needing a reference model as the baseline—the comparison is between the two responses directly.

**Key advantages:** (a) No reference model → halves memory. (b) Single-stage training → simpler pipeline. (c) The SFT term provides a stabilizing gradient signal that prevents the model from collapsing.

### 5.3 Comparison of Alignment Methods

| Method | Paired Data? | Reference Model? | Reward Model? | RL? | Memory (models loaded) |
|---|---|---|---|---|---|
| RLHF (PPO) | Yes | Yes | Yes | Yes | 4 |
| DPO | Yes | Yes | No | No | 2 |
| KTO | No | Yes | No | No | 2 |
| ORPO | Yes | No | No | No | 1 |

---

## 6. Preference Alignment: Common Pitfalls & Misconceptions

**Pitfall 1: "DPO is strictly better than RLHF."**
DPO is simpler and cheaper, but RLHF with PPO has been validated at the largest scales (GPT-4, Claude). DPO can suffer from overfitting to the preference dataset because it lacks the online exploration that PPO provides. In PPO, the policy generates new responses each iteration, creating fresh training signal. DPO trains on a fixed offline dataset, which limits diversity.

**Pitfall 2: "The reward model accuracy is 85%, so my RLHF will work well."**
Reward model accuracy on a held-out test set does not guarantee that it is a good optimization objective. The policy will find and exploit systematic biases in the reward model (reward hacking). Even a 95%-accurate reward model can produce pathological behavior when optimized against.

**Pitfall 3: "I can use DPO/KTO without a good SFT model."**
All alignment methods assume the model already knows *how* to generate coherent, instruction-following responses. The alignment stage teaches *which* responses to prefer. If the SFT model is poor, the alignment stage has nothing to work with—it cannot create capabilities that do not exist.

**Pitfall 4: "Higher β means better alignment."**
Higher $\beta$ means more conservative updates (policy stays closer to reference). Too high → the model barely changes from SFT (alignment has no effect). Too low → the model deviates excessively (reward hacking or collapse). Typical range: $\beta \in [0.1, 0.5]$ for DPO.

**Pitfall 5: "KTO does not need any pairing, so I can use random data."**
KTO still needs high-quality, accurately labeled good/bad responses. Mislabeled data (calling good responses bad or vice versa) degrades alignment just as it would for DPO. The advantage is logistical (no pairing needed), not that it is more robust to noise.

---

## Interview Preparation: Preference Alignment

### Foundational Questions

**Q1: Explain the RLHF pipeline end-to-end to a non-ML stakeholder.**

We start with a model that can follow instructions (from SFT). Then we collect human feedback: we show annotators multiple responses to the same question and ask which they prefer. We use this data to train a "reward model"—a separate AI that predicts how much humans would like a response. Finally, we use reinforcement learning to adjust the original model so it generates responses the reward model rates highly, while staying close to its original behavior to prevent gaming the system. The result is a model that not only follows instructions but produces responses humans actually prefer.

**Q2: What is the Bradley-Terry model and why is it used for reward modeling?**

The Bradley-Terry model is a probabilistic model for pairwise comparisons. It assigns each item a "strength" parameter and models the probability of one item being preferred over another as a sigmoid of the strength difference: $P(A \succ B) = \sigma(\text{strength}_A - \text{strength}_B)$. In RLHF, each response's "strength" is its reward model score. We use Bradley-Terry because: (a) it naturally handles the pairwise comparison format of human annotation data; (b) its loss is differentiable and easy to optimize; (c) it produces a scalar reward for each response that can be used in downstream RL training; (d) it satisfies transitivity (if A > B and B > C, then A > C), which is a reasonable approximation for quality judgments.

**Q3: How does DPO eliminate the reward model?**

DPO derives an analytical relationship between the optimal reward function and the optimal policy. Starting from the KL-regularized reward maximization objective, the optimal policy has the form $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$. Inverting this gives the reward as a function of policy probabilities: $r(x,y) = \beta \log(\pi^*/\pi_{\text{ref}}) + C(x)$. Substituting this into the Bradley-Terry model, the prompt-dependent constant $C(x)$ cancels in the preference probability (since it appears identically for both preferred and rejected responses). This yields a loss function that depends only on the policy's log-probabilities and the reference model's log-probabilities—no reward model needed.

### Mathematical Questions

**Q4: Derive the DPO gradient and explain what it does intuitively.**

The DPO loss is:
$$\mathcal{L} = -\log \sigma(\beta(\hat{r}_w - \hat{r}_l))$$
where $\hat{r}_y = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.

Let $u = \beta(\hat{r}_w - \hat{r}_l)$. Then:
$$\frac{\partial \mathcal{L}}{\partial \theta} = -\frac{\sigma'(u)}{\sigma(u)} \cdot \beta \cdot \frac{\partial(\hat{r}_w - \hat{r}_l)}{\partial \theta}$$

Since $\frac{\sigma'(u)}{\sigma(u)} = 1 - \sigma(u) = \sigma(-u)$:

$$\frac{\partial \mathcal{L}}{\partial \theta} = -\beta \sigma(-u) \left[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right]$$

**Intuition:** The gradient has two parts:
- $\nabla_\theta \log \pi_\theta(y_w|x)$: Increases the probability of the preferred response.
- $-\nabla_\theta \log \pi_\theta(y_l|x)$: Decreases the probability of the rejected response.
- $\sigma(-u)$: A weighting factor that is large when the model currently *disagrees* with the human preference (assigns higher implicit reward to the rejected response). As the model aligns, $u$ grows, $\sigma(-u) \to 0$, and the gradient vanishes—the model has already learned this preference.

This self-weighting property means DPO naturally focuses learning on examples where the model's preferences disagree with human preferences, and ignores examples it has already learned.

**Q5: Show that the KL-regularized reward maximization has a closed-form optimal policy.**

We want to solve:
$$\max_\pi \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x,y)] - \beta \text{KL}(\pi \| \pi_{\text{ref}})$$

Expand the KL:
$$= \mathbb{E}_{y \sim \pi} \left[ r(x,y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} \right]$$

$$= \mathbb{E}_{y \sim \pi} \left[ r(x,y) + \beta \log \pi_{\text{ref}}(y|x) - \beta \log \pi(y|x) \right]$$

$$= \sum_y \pi(y|x) \left[ r(x,y) + \beta \log \pi_{\text{ref}}(y|x) \right] + \beta H(\pi)$$

where $H(\pi) = -\sum_y \pi(y|x) \log \pi(y|x)$ is the entropy. This is maximized by the Gibbs distribution (a standard result from statistical mechanics / variational inference):

$$\pi^*(y|x) = \frac{1}{Z(x)} \exp\left(\frac{1}{\beta}\left[r(x,y) + \beta \log \pi_{\text{ref}}(y|x)\right]\right) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

This can also be derived by taking the functional derivative of the objective with respect to $\pi$, adding a Lagrange multiplier for the normalization constraint, setting to zero, and solving. The solution is the expression above with $Z(x)$ serving as the normalization constant.

### Applied Questions

**Q6: You have a DPO-trained model that produces very safe but extremely boring responses. Diagnose and fix.**

This is likely over-alignment, often caused by:

1. **$\beta$ too low:** The policy is deviating too far from the SFT reference, collapsing toward whatever style the preference data rewards most (safety). Fix: increase $\beta$ to keep the policy closer to the more capable SFT model.

2. **Biased preference data:** If annotators systematically preferred safe/conservative responses, the model learns "safe = good." Fix: rebalance the dataset to include preferences for helpfulness and engagement, not just safety.

3. **Length bias:** If preferred responses tended to be shorter (because they were "safer"), the model learns to be terse. Fix: length-normalize the loss or filter the dataset for length parity.

Resolution strategy: (a) Inspect the preference data for systematic biases. (b) Try $\beta$ in the range $[0.2, 0.5]$. (c) Mix in preference pairs where the more helpful (but safe) response is preferred over the overly cautious one. (d) Consider RLHF with a multi-objective reward model that explicitly scores both safety and helpfulness.

**Q7: You are choosing between PPO-based RLHF and DPO for aligning a 13B model. What factors would drive your decision?**

I would consider:

**Choose DPO when:**
- Compute budget is limited (DPO needs ~2x SFT cost vs. ~20x for PPO).
- Engineering resources are limited (DPO is standard supervised training; PPO requires RL infrastructure).
- You already have a high-quality preference dataset.
- The model is relatively small (13B is manageable for DPO's 2-model memory requirement).

**Choose PPO when:**
- You need the reward model for other purposes (rejection sampling, monitoring, RLHF-based search).
- You want online exploration: PPO generates new responses each iteration, providing fresh training signal and potentially discovering better behaviors than what is in the fixed preference dataset.
- You are at massive scale where online methods have been more thoroughly validated.
- Your preference dataset is small—PPO's online generation effectively augments the data.

For a 13B model with a good preference dataset and limited compute, I would start with DPO. If results plateau, I would consider PPO with a reward model trained on the same preferences.

### Debugging & Failure-Mode Questions

**Q8: Your DPO-trained model's performance suddenly degrades after 2,000 training steps, even though the training loss continues to decrease. What happened?**

This is a classic overfitting pattern in DPO. The model has memorized the preference dataset: for the specific (prompt, $y_w$, $y_l$) triples it trained on, it strongly prefers $y_w$. But this does not generalize to new prompts or new response pairs.

Symptoms: training loss decreases, but evaluation metrics (human ratings, benchmark scores) plateau or decline. The model may also become more repetitive (generating responses very similar to $y_w$ in the training set).

Diagnosis: (a) Plot evaluation metrics vs. training steps (not just training loss). (b) Check if the implicit reward margin $\hat{r}_w - \hat{r}_l$ is growing unboundedly (the sigmoid is saturated). (c) Measure diversity of generated responses (a drop indicates mode collapse).

Fixes: (a) Use early stopping based on validation metrics. (b) Increase $\beta$ to slow optimization. (c) Add more diverse preference data. (d) Apply label smoothing to the DPO objective. (e) Reduce learning rate.

**Q9: Your reward model gives high scores to verbose, hedging responses ("I think maybe it could possibly be the case that..."). How does this propagate through the RLHF pipeline?**

This is a reward model bias. If the RM training data was annotated by humans who associated hedging with thoughtfulness, the RM learns that pattern. During PPO, the policy discovers that hedging phrases increase reward. As training progresses, the policy generates increasingly verbose, uncertain-sounding responses because the RM rewards them.

This is reward hacking—the policy is optimizing a *proxy* of quality (hedging language) rather than actual quality. The KL penalty slows this but does not prevent it.

Fixes: (a) Retrain the RM with de-biased data (include examples where a confident, concise response is preferred over a hedging one). (b) Add a length penalty to the reward. (c) Use an ensemble of reward models—if they disagree, the signal is likely due to bias. (d) Periodically refresh the RM with feedback on the policy's actual outputs (iterative RLHF).

### Follow-Up Probing Questions

**Q10: "You said the partition function Z(x) cancels in DPO. What would happen if it did not cancel?"**

If $Z(x)$ did not cancel, we would need to compute $\sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ over all possible responses $y$—an intractable sum over an exponentially large space (all possible token sequences). This is exactly why reward-based approaches (RLHF) are hard: they implicitly involve such normalizations. The mathematical elegance of DPO is precisely that the paired comparison structure of Bradley-Terry causes $Z(x)$ to cancel, turning an intractable RL problem into a tractable supervised one. If we used a different preference model (e.g., one that depends on absolute reward values rather than differences), $Z(x)$ would not cancel and we would be back to needing either RL or approximate inference.

---

# Part IV: Evaluation Frameworks

---

## 1. Motivation & Intuition

### Why LLM Evaluation Is Hard

Evaluating a language model is fundamentally different from evaluating a classifier. A classifier maps inputs to a small set of labels, and accuracy is straightforward. A language model generates *open-ended text*, and there are often many valid responses to a given prompt—some better than others, but none uniquely "correct."

Consider asking an LLM: "Explain why the sky is blue." A good answer could be one paragraph or three paragraphs. It could use an analogy or go straight to Rayleigh scattering. It could be aimed at a child or a physicist. How do you score this automatically?

This has led to a multi-pronged evaluation strategy:

1. **Benchmarks:** Standardized datasets with (usually) single correct answers, measuring specific capabilities like knowledge, reasoning, or coding. These are automated, reproducible, and comparable across models.

2. **LLM-as-a-Judge:** Using a strong LLM (like GPT-4 or Claude) to grade open-ended responses. This bridges the gap between automated metrics (cheap but limited) and human evaluation (rich but expensive).

3. **Human evaluation:** The gold standard but expensive, slow, and subject to inter-annotator variability.

No single evaluation method is sufficient. The field relies on all three in combination.

---

## 2. Benchmarks

### 2.1 MMLU (Massive Multitask Language Understanding)

**What it measures:** Broad world knowledge and language understanding across 57 academic subjects (from elementary mathematics to professional medicine, law, and ethics).

**Format:** Multiple-choice questions with 4 options. Example:

```
Question: What is the capital of Australia?
A) Sydney  B) Melbourne  C) Canberra  D) Perth
Answer: C
```

**How it is scored:** Accuracy (fraction of questions answered correctly). The model is typically given a few-shot prompt (5 examples of the same format from the same subject) and must select the correct letter.

**Why it matters:** MMLU tests whether the model has *retained* broad knowledge from pretraining. A model that scores well on MMLU has a broad knowledge base, which is a prerequisite for being a useful assistant.

**Strengths:**
- Covers 57 diverse subjects, providing a holistic view of knowledge.
- Easy to compute (multiple-choice, no generation needed—just compare logit probabilities).
- Widely adopted, enabling model-to-model comparison.

**Weaknesses:**
- Multiple choice is artificial—real users do not give four options.
- Subject coverage is weighted toward US academic curricula.
- Can be gamed: models can learn to recognize answer patterns without true understanding.
- Does not test generation quality, reasoning depth, or instruction-following.

**Typical scores (as of 2024–2025):** Frontier models (GPT-4, Claude 3.5, Gemini Ultra) score 85–90%. Open models (LLaMA 70B) score 75–82%. Random guessing is 25%.

### 2.2 GSM8K (Grade School Math 8K)

**What it measures:** Multi-step mathematical reasoning. Despite the "grade school" label, solving these problems requires the model to decompose a word problem into steps, perform arithmetic, and track intermediate results—a genuine reasoning challenge for LLMs.

**Format:** Word problems requiring 2–8 reasoning steps. Example:

```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins with four every day. She sells the remainder 
for $2 each. How much money does she make per day?
Answer: 16 - 3 - 4 = 9 eggs remaining. 9 × $2 = $18 per day.
```

**How it is scored:** Exact match on the final numerical answer. The model generates a chain-of-thought solution, and the last number in its response is extracted and compared to the gold answer.

**Why it matters:** Math reasoning is one of the most robust tests of whether a model can *think* step-by-step rather than just pattern-match. It is also practically important—users frequently ask LLMs to help with quantitative problems.

**Strengths:**
- Tests genuine multi-step reasoning, not just knowledge recall.
- Problems are unambiguous with a single correct answer.
- Chain-of-thought evaluation reveals *how* the model reasons, not just *what* it answers.

**Weaknesses:**
- Limited to arithmetic and simple algebra; does not test higher math.
- Models can memorize solutions if GSM8K-like problems leak into training data (contamination).
- Only evaluates the final answer; a model could get the right answer for the wrong reasons.

### 2.3 HumanEval (Coding)

**What it measures:** Code generation ability. The model is given a function signature and docstring and must generate the function body.

**Format:** Python function completion. Example:

```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers closer to each 
    other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**How it is scored:** The key metric is **pass@k**: the probability that at least one of $k$ generated solutions passes all test cases.

$$\text{pass}@k = \mathbb{E}_{\text{problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]$$

where $n$ is the total number of generated samples and $c$ is the number that pass. pass@1 (single attempt) is the most commonly reported.

**Why it matters:** Code generation is one of the most commercially valuable LLM applications. It is also a clean evaluation setting—code either passes tests or does not, removing subjectivity.

**Strengths:**
- Objectively verifiable (tests either pass or fail).
- Tests both language understanding (parsing docstrings) and logical reasoning (writing correct code).
- Practical relevance for real-world applications.

**Weaknesses:**
- Only 164 problems (small benchmark).
- Python-only in the original version (EvalPlus and MultiPL-E extend to other languages).
- Test cases may not cover all edge cases, allowing incorrect solutions to pass.
- Contamination risk: these problems are on the internet and may be in training data.

### 2.4 Benchmark Contamination

A pervasive concern: if benchmark questions appear in the model's training data, the model can memorize answers rather than demonstrating genuine capability. This is **benchmark contamination** (also called data contamination).

**Detection methods:**
- **Canary strings:** Embed unique strings in benchmarks; check if the model can reproduce them.
- **N-gram overlap:** Measure overlap between benchmark questions and training data (requires training data access).
- **Performance on perturbed versions:** If the model's accuracy drops significantly when questions are paraphrased but the task is unchanged, memorization is likely.

**Mitigation:**
- Use held-out, continuously updated benchmarks (e.g., LiveBench, LMSYS Chatbot Arena).
- Report performance on both public and private benchmarks.
- Time-stamp training data and benchmark release dates.

---

## 3. LLM-as-a-Judge

### 3.1 Conceptual Foundations

**What it is:** Use a strong LLM (GPT-4, Claude) to evaluate the outputs of other models. The judge model is given a prompt, one or more candidate responses, and evaluation criteria, and it produces a score or ranking.

**Why it is needed:** Many LLM outputs (summaries, creative writing, advice, explanations) cannot be evaluated with simple metrics like accuracy. Human evaluation is the gold standard but costs $5–20 per judgment and takes days. LLM-as-a-Judge provides 80–90% of the signal at 1/100th the cost.

**How it works:**

```
System: You are an expert evaluator. Score the following response on a scale 
of 1-5 for helpfulness, accuracy, and clarity.

Prompt: [original user question]
Response: [model's response to evaluate]

Provide scores and brief justifications.
```

The judge model outputs structured scores that can be aggregated across many examples.

### 3.2 Formats

**Pointwise scoring:** The judge rates a single response on a scale (1–5 or 1–10). Simple but prone to miscalibration (judges may use different parts of the scale inconsistently).

**Pairwise comparison:** The judge is shown two responses and asked which is better. More reliable than pointwise scoring (relative judgments are easier than absolute ones). This is the format used in Chatbot Arena and many alignment datasets.

**Reference-guided evaluation:** The judge is given a "gold" reference answer and asked how well the candidate response matches it. Useful for factual tasks.

### 3.3 Challenges and Biases

**Position bias:** LLM judges tend to prefer the response presented first (or last, depending on the model). Mitigation: evaluate each pair twice, swapping the order, and only count agreements.

**Verbosity bias:** LLM judges tend to prefer longer responses, even when shorter ones are better. Mitigation: explicitly instruct the judge to not favor length, or length-normalize evaluations.

**Self-preference bias:** A model tends to rate its own outputs higher than those of other models. Mitigation: do not use a model as a judge of its own outputs.

**Ceiling effect:** If the model being evaluated is nearly as capable as the judge, the judge may not reliably distinguish quality differences. The judge must be substantially stronger than the models it evaluates.

### 3.4 When to Use LLM-as-a-Judge vs. Alternatives

| Evaluation Need | Best Method | Why |
|---|---|---|
| Factual accuracy | Benchmarks (MMLU) | Single correct answer |
| Math/code correctness | Benchmarks (GSM8K, HumanEval) | Verifiable output |
| Response quality (helpfulness, clarity) | LLM-as-a-Judge | Subjective, needs nuanced judgment |
| Safety/toxicity | Specialized classifiers + human review | High-stakes, cannot trust LLM alone |
| Final production evaluation | Human evaluation | Gold standard for deployment decisions |

---

## 4. Evaluation: Common Pitfalls & Misconceptions

**Pitfall 1: "Our model scores 90% on MMLU, so it is production-ready."**
Benchmarks measure narrow capabilities. A model scoring 90% on MMLU might still generate unsafe content, hallucinate confidently, refuse reasonable requests, or produce poorly formatted outputs. Benchmarks are necessary but not sufficient.

**Pitfall 2: "We use GPT-4 as our judge, so our evaluation is objective."**
LLM judges have systematic biases (position, verbosity, self-preference). They also cannot catch factual errors if they do not know the ground truth. LLM-as-a-Judge is a useful signal, not ground truth.

**Pitfall 3: "Higher benchmark scores always mean a better model."**
Different benchmarks measure different things. A model might score higher on MMLU but lower on HumanEval, reflecting a trade-off between knowledge and coding ability. The "best" model depends on the application. Additionally, benchmark contamination can inflate scores artificially.

**Pitfall 4: "We evaluated on one benchmark and it looks good."**
One benchmark captures one dimension. You need a *suite* of evaluations: knowledge (MMLU), reasoning (GSM8K, ARC), coding (HumanEval), instruction-following (IFEval, MT-Bench), safety (ToxiGen, BBQ), and open-ended quality (Chatbot Arena, LLM-as-a-Judge).

**Pitfall 5: "Our benchmark numbers went up after fine-tuning, so the model improved."**
If the fine-tuning data contains benchmark questions (or very similar ones), the improvement is contamination, not genuine capability gain. Always evaluate on held-out benchmarks and monitor for suspiciously large jumps on specific benchmarks.

---

## Interview Preparation: Evaluation Frameworks

### Foundational Questions

**Q1: Why can you not just use BLEU/ROUGE to evaluate modern LLMs?**

BLEU and ROUGE measure n-gram overlap with reference texts. They were designed for tasks with limited output variation (machine translation, summarization). For open-ended generation, there are many valid responses with completely different wording. A model producing a correct, clear, but differently-worded answer would score poorly on BLEU/ROUGE. These metrics penalize diversity and creativity, and they cannot detect factual errors, logical mistakes, or safety issues. Modern LLM evaluation requires either task-specific metrics (accuracy for QA, pass@k for code) or more sophisticated approaches (LLM-as-a-Judge, human evaluation).

**Q2: What is the Chatbot Arena and why is it considered a gold standard for LLM evaluation?**

Chatbot Arena (LMSYS) is a crowdsourced evaluation platform where users chat with two anonymous models side by side and vote for the better response. Models are ranked using Elo ratings (borrowed from chess), which are calibrated through thousands of pairwise comparisons. It is considered a gold standard because: (a) evaluators are real users with genuine queries, not paid annotators with synthetic prompts; (b) the blind, pairwise format reduces bias; (c) the large number of votes (millions) provides statistical robustness; (d) it measures overall user satisfaction, not narrow capabilities; (e) it is difficult to game because the query distribution is organic and unpredictable.

**Q3: Explain the difference between pass@1 and pass@k for code evaluation.**

pass@1 measures whether the model's first generated solution passes all test cases. It reflects the model's performance in a "one-shot" usage scenario. pass@k measures whether *at least one* of $k$ independently generated solutions passes. It captures the model's coverage—even if individual attempts sometimes fail, the model might reliably produce at least one correct solution in $k$ tries. For practical applications, pass@1 matters for conversational settings (the user sees one response), while pass@k matters for automated pipelines where multiple candidates can be generated and tested (e.g., with best-of-$k$ sampling or unit test filtering).

### Applied Questions

**Q4: You are building an evaluation suite for a medical LLM. What benchmarks and evaluation methods would you use?**

I would design a multi-layered evaluation:

**Layer 1: Automated benchmarks**
- MedQA (USMLE-style questions) for clinical knowledge.
- PubMedQA for biomedical research comprehension.
- A custom benchmark of institution-specific clinical scenarios.

**Layer 2: LLM-as-a-Judge**
- Use a strong model to evaluate the quality, completeness, and appropriateness of open-ended medical responses.
- Include rubrics for: clinical accuracy, appropriate caveats/disclaimers, referral to professionals when needed, reading level appropriateness.

**Layer 3: Safety-specific evaluation**
- Test for dangerous medical advice (e.g., drug interactions, contraindicated treatments).
- Red-team with adversarial prompts designed to elicit harmful medical guidance.
- Evaluate hallucination rates on rare conditions.

**Layer 4: Expert human evaluation**
- Board-certified physicians review a sample of model outputs for clinical correctness and safety.
- Use structured rubrics with inter-rater reliability metrics.

**Layer 5: Real-world metrics (post-deployment)**
- User satisfaction surveys, time-to-resolution, escalation rates.
- Monitor for safety incidents.

The key principle is that no single evaluation is sufficient. The stakes in medicine require overlapping, redundant evaluation layers.

**Q5: A product manager asks you: "Model A scores higher on MMLU but users prefer Model B in A/B tests. Which model should we deploy?"**

Deploy Model B. User preference in A/B tests is the most direct measure of the metric that matters: user satisfaction. MMLU measures factual knowledge, which is only one component of overall quality. Model B likely excels in aspects that MMLU does not capture: tone, helpfulness, format, conciseness, engagement, instruction-following, and safety. These attributes are often more important for user satisfaction than raw knowledge.

That said, I would investigate *why* the scores diverge. If Model A has specific knowledge strengths (e.g., better at technical domains) that matter for the product, we might use Model A for those specific use cases and Model B as the default. I would also check if Model B's user preference is driven by superficial factors (being more verbose, more agreeable) vs. genuine quality differences.

### Debugging & Failure-Mode Questions

**Q6: Your LLM-as-a-Judge evaluation shows Model A wins 70% of comparisons against Model B. But when you swap the order of presentation, Model A only wins 45%. What is happening?**

This is **position bias**. The judge model is preferring whichever response appears first (or second, depending on the specific pattern). The true preference is somewhere between 70% and 45%.

To get an unbiased estimate, evaluate each pair in both orders and use the average: $(70\% + 45\%) / 2 = 57.5\%$ for Model A. Alternatively, only count pairs where the judge agrees across both orderings (consistent wins). This discards noisy judgments and gives a more reliable signal, at the cost of smaller effective sample size.

Long-term fix: fine-tune the judge model to reduce position bias, use multiple judge models and aggregate, or switch to a judge model known to have low position bias.

**Q7: Your model's GSM8K accuracy dropped from 85% to 60% after fine-tuning it for customer service. Should you be concerned?**

Yes, this is a significant regression indicating catastrophic forgetting of mathematical reasoning. However, whether it matters depends on context:

**If the customer service application never involves math:** It is still concerning because (a) math reasoning is correlated with general reasoning ability, so other capabilities may have degraded too; (b) users may occasionally ask math-related questions.

**Diagnosis:** (a) Check other reasoning benchmarks (ARC, BBQ, MMLU reasoning subsets). If they also dropped, general reasoning has degraded. (b) Compare the fine-tuning learning rate and epochs against recommendations for catastrophic forgetting prevention.

**Fix:** (a) Mix in 10–20% general instruction data (including math). (b) Switch to LoRA to preserve base model capabilities. (c) Reduce fine-tuning epochs. (d) If full parameter fine-tuning is necessary, use a lower learning rate or EWC regularization.

### Follow-Up Probing Questions

**Q8: "You mentioned benchmark contamination. How would you design a contamination-resistant benchmark?"**

Several strategies:

1. **Private, continuously refreshed test sets:** Keep the questions secret and regularly add new ones. This is what LiveBench does—it draws from recent events and publications post-dating model training.

2. **Procedurally generated problems:** For math and code, generate new problems algorithmically so the specific instances cannot be memorized. Parameterize problem templates and generate fresh instances for each evaluation.

3. **Adversarial perturbation:** Take existing benchmark problems and transform them in meaning-preserving ways (rephrase, change names/numbers, translate to different contexts). If a model can solve the original but not the perturbed version, contamination is likely.

4. **Dynamic, human-in-the-loop evaluation:** Chatbot Arena's approach—use real user queries that are unpredictable and continuously generated.

5. **Membership inference testing:** Before using a benchmark, test whether the model has memorized it by checking if it can reproduce questions verbatim or with suspiciously high confidence.

The fundamental tension: the more widely used a benchmark becomes, the more likely it is to be contaminated. The best benchmarks are either private or continuously evolving.

---

**[← Previous Chapter: RL (Reinforcement Learning)](rl_guide.md) | [Table of Contents](index.md) | [Next Chapter: Multimodal Methods →](multimodal_methods_guide.md)**
