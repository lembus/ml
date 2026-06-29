# Transformers: A Comprehensive Reference Guide

A standalone, textbook-quality reference covering the Transformer architecture, its components, efficient variants, and the major pretrained model families. Each topic follows the same six-section structure (Motivation, Conceptual Foundations, Mathematical Formulation, Worked Examples, ML Practice, Pitfalls) and concludes with a tiered interview question bank.

---

## Table of Contents

1. [Transformers — The Big Picture](#1-transformers--the-big-picture)
2. [Self-Attention Mechanism](#2-self-attention-mechanism)
3. [Positional Encoding](#3-positional-encoding)
4. [The Transformer Block](#4-the-transformer-block)
5. [Encoder–Decoder Architecture and Cross-Attention](#5-encoderdecoder-architecture-and-cross-attention)
6. [Efficient Attention Variants](#6-efficient-attention-variants)
7. [Pretrained Models](#7-pretrained-models)
   - 7.1 [BERT family](#71-bert-family)
   - 7.2 [GPT family](#72-gpt-family)
   - 7.3 [T5: Text-to-Text Transfer Transformer](#73-t5-text-to-text-transfer-transformer)
   - 7.4 [Vision Transformers (ViT)](#74-vision-transformers-vit)

---

# 1. Transformers — The Big Picture

## 1.1 Motivation & Intuition

### Why this architecture exists

Before 2017, the dominant tools for sequence modeling were **recurrent neural networks (RNNs)** and their gated cousins **LSTMs** and **GRUs**. These models read a sequence one token at a time, maintaining a hidden state that summarizes everything seen so far. To translate a sentence, an RNN encoder consumed the source word-by-word into a final hidden vector; an RNN decoder then unrolled that vector into the target sentence.

This worked, but it had three painful problems:

1. **Sequentiality kills parallelism.** Step *t* of an RNN cannot be computed until step *t-1* is finished. On modern GPUs that thrive on massive parallel matrix multiplication, this forces hardware to sit idle. A 100-word sentence requires 100 sequential steps.

2. **Long-range dependencies fade.** Information from word 1 has to survive being repeatedly multiplied through 99 nonlinear transformations to influence the prediction at word 100. Gradients vanish, signals get overwritten, and the network forgets. Attention mechanisms (Bahdanau, 2014) were bolted onto RNNs to mitigate this — but the RNN was still the spine.

3. **The bottleneck of a single hidden state.** In encoder–decoder RNNs, the entire source sentence had to be compressed into one vector before decoding began. For long inputs this was a severe information bottleneck.

The 2017 paper "Attention Is All You Need" (Vaswani et al.) proposed a radical idea: **throw out recurrence entirely** and build the whole model from attention layers. Every position can directly look at every other position in a single layer, all in parallel.

### A concrete illustration

Suppose you want to translate the English sentence *"The animal didn't cross the street because it was too tired"* into French. To translate the word *it*, the model must figure out what *it* refers to — *the animal*, not *the street*. An RNN would have to carry that resolution through every intermediate word in its hidden state. A Transformer, by contrast, lets the representation of *it* directly attend to *the animal* in a single layer, computing a similarity-weighted average of the two candidates' representations.

This idea — that each token's representation should be a **content-based weighted combination** of all other tokens' representations — is the entire philosophical core of the Transformer.

### Connection to ML systems

The Transformer is the architectural backbone of essentially every state-of-the-art system in NLP, vision, speech, code, biology, and increasingly RL. The reasons are practical:

- **Trains in parallel** → effective use of TPUs and large GPU clusters.
- **Scales predictably** → Kaplan/Chinchilla scaling laws were derived primarily from Transformer language models. Doubling parameters and data yields predictable loss reductions.
- **Transfers across modalities** → the same architecture handles text (BERT, GPT), images (ViT), audio (Whisper, AudioLM), proteins (AlphaFold's Evoformer), and multimodal inputs (CLIP, Flamingo, Gemini).
- **Pretrain once, fine-tune many** → the dominant paradigm for production ML in 2025 is "take a pretrained Transformer and adapt it." This is only possible because the architecture generalizes so well.

## 1.2 Conceptual Foundations

A Transformer model is built from a small number of repeating components. Once you understand them, the rest is composition.

### Key terms

- **Token**: an atomic unit of input. For text, this is typically a subword (BPE, WordPiece, SentencePiece). For images, a patch. For audio, a spectrogram frame.
- **Embedding**: a learned lookup that maps each token id to a dense vector in $\mathbb{R}^d$.
- **Position encoding**: added information so the model knows the order of tokens (since attention is permutation-equivariant by default).
- **Attention layer**: computes a weighted average of value vectors, where weights come from query–key similarities.
- **Feed-forward network (FFN)**: a position-wise two-layer MLP that processes each token independently.
- **Residual connection**: the input to a sublayer is added to its output ($y = x + \text{Sublayer}(x)$).
- **Layer normalization**: per-token normalization that stabilizes training.
- **Transformer block**: the canonical unit = attention sublayer + FFN sublayer, each wrapped in residual + LayerNorm.
- **Encoder**: a stack of blocks where every position can attend to every other position (bidirectional).
- **Decoder**: a stack of blocks where positions can only attend to earlier positions (causal/autoregressive). May also include cross-attention to encoder output.

### How the components interact

For an encoder-only model processing a sequence of $n$ tokens:

1. Tokens → embedding lookup → matrix $X \in \mathbb{R}^{n \times d}$.
2. Add positional encodings → $X' = X + P$.
3. Pass $X'$ through $L$ identical Transformer blocks. Each block:
   - Applies multi-head self-attention (every position mixes with every other).
   - Adds residual, normalizes.
   - Applies a position-wise FFN.
   - Adds residual, normalizes.
4. Final output: an $n \times d$ matrix where each row is a contextualized representation of the corresponding token.

For an encoder–decoder model (machine translation, T5):

1. Encoder produces context $H_{\text{enc}} \in \mathbb{R}^{n \times d}$ as above.
2. Decoder consumes target tokens (shifted right) through:
   - Masked (causal) self-attention over decoder positions.
   - Cross-attention where queries come from the decoder and keys/values come from $H_{\text{enc}}$.
   - FFN.
3. Output projected to vocabulary logits via a linear head + softmax.

### Underlying assumptions

- **Inputs are sequences (or sets) of vectors.** Anything you can tokenize, you can feed to a Transformer.
- **Position must be supplied externally.** The architecture is otherwise permutation-equivariant.
- **The number of tokens fits in memory.** Standard attention is $O(n^2)$ in time and memory.
- **There is enough data and compute to learn meaningful relationships from scratch.** Transformers are notoriously data-hungry compared to CNNs or RNNs with strong inductive biases.

### What breaks when assumptions fail

- **Very long sequences**: standard attention becomes infeasible (we'll discuss efficient variants in §6).
- **Tiny datasets**: vanilla ViT underperforms a CNN of comparable size on small image datasets because it lacks translation-equivariance and locality biases.
- **Order-sensitive tasks without proper positional info**: the model literally cannot tell *"dog bites man"* from *"man bites dog"*.

## 1.3 Mathematical Formulation

Let $X \in \mathbb{R}^{n \times d}$ be a sequence of $n$ token embeddings of dimension $d$. A single Transformer encoder block computes:

$$
Z = \text{LayerNorm}(X + \text{MultiHeadAttn}(X))
$$

$$
Y = \text{LayerNorm}(Z + \text{FFN}(Z))
$$

(This is the *post-norm* variant from the original paper. Modern implementations usually use *pre-norm*, discussed in §4.)

We will fully unpack each piece in subsequent sections. For now, treat them as black boxes with the following signatures:

- $\text{MultiHeadAttn}: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ — mixes information across positions.
- $\text{FFN}: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ — transforms each position independently.

The full model is a composition of $L$ such blocks.

## 1.4 Worked Example

Consider a tiny model with vocabulary size 5, embedding dimension $d = 4$, and a single block. Input the 3-token sequence "cat sat mat".

1. **Tokenize and look up embeddings.** Suppose:
   - cat → $[0.1, 0.2, 0.3, 0.4]$
   - sat → $[0.5, 0.1, 0.0, 0.2]$
   - mat → $[0.2, 0.3, 0.1, 0.5]$

   Stack: $X = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.1 & 0.0 & 0.2 \\ 0.2 & 0.3 & 0.1 & 0.5 \end{bmatrix}$.

2. **Add positional encoding** (dummy values for illustration):
   $$X' = X + \begin{bmatrix} 0.00 & 1.00 & 0.00 & 1.00 \\ 0.84 & 0.54 & 0.01 & 1.00 \\ 0.91 & -0.42 & 0.02 & 1.00 \end{bmatrix}$$

3. **Self-attention**. Each row of $X'$ is projected to a query, key, value triple. After attention, each row becomes a weighted average of all three value rows. Concretely, suppose attention weights end up as:
   - "cat" attends 70% to itself, 20% to "sat", 10% to "mat".
   - "sat" attends 30% to "cat", 50% to itself, 20% to "mat".
   - "mat" attends 30% to "cat", 30% to "sat", 40% to itself.

   Then the new representation of "cat" is $0.7 \cdot v_{\text{cat}} + 0.2 \cdot v_{\text{sat}} + 0.1 \cdot v_{\text{mat}}$.

4. **Residual + LayerNorm**: $Z = \text{LN}(X' + \text{Attn}(X'))$.

5. **FFN**: each row independently passes through a 2-layer MLP. The same MLP is applied to all three positions; there is no cross-position interaction in this sublayer.

6. **Residual + LayerNorm** again, producing final output $Y$.

Stack $L = 12$ such blocks (BERT-base size), and you have a contextualized representation for each input token. For language modeling, project these to vocabulary logits and apply softmax.

## 1.5 Relevance to Machine Learning Practice

Transformers are essentially universal in modern ML:

- **Pretraining**: BERT, GPT, T5, LLaMA, Claude, Gemini, etc. — all Transformer-based.
- **Fine-tuning / parameter-efficient adaptation**: LoRA, adapters, and prefix tuning all attach to Transformer layers.
- **Inference at scale**: KV-caching, FlashAttention, paged attention, speculative decoding, and quantization are all Transformer-specific systems engineering.
- **Evaluation**: standard benchmarks (GLUE, SuperGLUE, MMLU, HumanEval, BIG-Bench) are designed around Transformer LM capabilities.
- **Monitoring**: drift in attention patterns, perplexity, token-level loss, and KV-cache hit rates are common production signals.

**When to use a Transformer:** when you have sufficient data and compute, when long-range dependencies matter, when you want to leverage a pretrained backbone, or when you need to fuse modalities.

**When not to:** very small datasets without pretraining (CNNs/MLPs may win), strict latency budgets on tiny inputs (specialized architectures may be faster), or when strong domain inductive biases (locality, equivariance) make alternatives more sample-efficient.

**Trade-offs**:
- Bias–variance: low inductive bias → high variance, requires lots of data.
- Interpretability: attention weights are *not* faithful explanations (Jain & Wallace 2019), but they are useful diagnostic signals.
- Robustness: vulnerable to adversarial prompts, distributional shift, and prompt-injection attacks.
- Compute: $O(n^2 d)$ attention is the dominant cost for long sequences.

## 1.6 Common Pitfalls & Misconceptions

- **"Attention weights tell you what the model is thinking."** They don't, fully. They are one signal among many, and gradient-based attribution often disagrees.
- **"Transformers process input in parallel, so they're always faster than RNNs."** True at training time. At autoregressive inference, decoding is still sequential token-by-token.
- **"Bigger context = better."** Up to a point. Many models exhibit "lost in the middle" behavior, where information in the middle of a long context is poorly used.
- **"The same Transformer block is used everywhere."** Modern variants differ substantially — pre-norm vs post-norm, RMSNorm vs LayerNorm, GLU activations, RoPE, sliding-window attention, MoE replacements for FFNs.
- **"Self-attention is a single mechanism."** It's actually a family. Encoder self-attention is bidirectional; decoder self-attention is causal-masked; cross-attention is asymmetric.

---

# 2. Self-Attention Mechanism

## 2.1 Motivation & Intuition

### The retrieval analogy

Imagine you have a small in-memory database of (key, value) pairs. You issue a query and want to retrieve the value whose key best matches. Hard retrieval picks exactly one. **Soft retrieval** computes a similarity between your query and every key, normalizes those similarities into a probability distribution, and returns a weighted average of all values.

Self-attention is exactly this — except every token plays all three roles. Each token emits a **query** describing what information it is looking for, a **key** describing what information it offers, and a **value** describing the actual content it would contribute. The query of token $i$ is compared against every key, producing weights, and token $i$'s new representation is the weighted sum of all values.

### Why "self"?

Because the queries, keys, and values all come from the same input sequence. (In cross-attention, queries come from one sequence and keys/values from another.)

### A linguistic intuition

Take *"The trophy didn't fit in the brown suitcase because it was too small."* The word *it* is ambiguous. To resolve it, we need to compare *it* against *trophy* and *suitcase*. Self-attention lets *it*'s query vector match the keys of both candidates and form a context-weighted blend, ideally putting most weight on *suitcase*.

### Why content-based mixing rather than fixed mixing?

Convolutions mix nearby positions with fixed weights. Recurrent networks mix sequentially through a state. Self-attention mixes positions *based on what they contain*. The mixing pattern is data-dependent and varies by input — which is enormously expressive but expensive ($O(n^2)$).

## 2.2 Conceptual Foundations

### Key terms

- **Query ($q$)**: a vector representing "what I am looking for."
- **Key ($k$)**: a vector representing "what I offer / how I can be matched."
- **Value ($v$)**: a vector representing "what I will contribute if matched."
- **Attention score**: raw similarity between a query and a key, typically a dot product.
- **Attention weight**: the score after softmax normalization, a number in $[0,1]$ that sums to 1 across keys for a given query.
- **Attention output**: weighted sum of value vectors using the attention weights.
- **Head**: one independent set of (Q, K, V) projections producing one attention output. Multi-head attention runs several in parallel.
- **Mask**: a matrix that blocks attention to certain positions (e.g., future tokens in causal LM, padding tokens, or task-specific structure).

### Step-by-step: scaled dot-product attention

Given input matrix $X \in \mathbb{R}^{n \times d}$:

1. Compute three projections:
   $$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$
   where $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$.

2. Compute pairwise scores:
   $$S = QK^\top \in \mathbb{R}^{n \times n}$$

3. Scale by $\sqrt{d_k}$:
   $$S' = \frac{S}{\sqrt{d_k}}$$

4. Apply mask if needed (set masked entries to $-\infty$).

5. Row-wise softmax:
   $$A = \text{softmax}(S')$$
   Each row sums to 1.

6. Weighted sum of values:
   $$\text{Attn}(Q, K, V) = AV \in \mathbb{R}^{n \times d_v}$$

### Multi-head attention

A single attention layer learns one mixing pattern. Multi-head attention learns several in parallel, each with smaller dimensions, then concatenates the results.

If we want $h$ heads with the same total compute, set $d_k = d_v = d / h$. Each head $i$ has its own $(W^Q_i, W^K_i, W^V_i)$. Outputs are concatenated and projected back to dimension $d$:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where $W^O \in \mathbb{R}^{(h d_v) \times d}$.

### Underlying assumptions

- **Dot-product similarity is meaningful in the projected Q/K space.** The model learns this via $W^Q, W^K$.
- **Softmax produces a useful distribution.** This implies queries should produce a few sharp matches rather than uniform attention; if attention is always uniform, the layer collapses to "average all values."
- **The variance of dot products grows with $d_k$.** Without scaling, softmax saturates and gradients vanish. The $\sqrt{d_k}$ scaling assumes Q and K entries have roughly unit variance.
- **All-to-all access is appropriate.** For local-structure tasks like raw pixel images, this assumption is wasteful (motivating ViT's patch embedding and efficient variants).

### What breaks

- **No scaling**: For large $d_k$, dot products have variance $d_k$, pushing softmax into saturation. One element dominates with weight ≈1, others ≈0, gradients vanish.
- **Wrong mask**: forgetting to mask padding tokens causes the model to attend to garbage; forgetting causal mask in a decoder causes information leakage from future tokens.
- **Numerical issues**: very large logits cause overflow; modern implementations subtract the row max before exponentiating.
- **Quadratic memory**: storing $A \in \mathbb{R}^{n \times n}$ becomes infeasible past ~16k tokens on a single GPU without specialized kernels (FlashAttention).

## 2.3 Mathematical Formulation

### Derivation of the $\sqrt{d_k}$ scaling

Suppose entries of $q, k \in \mathbb{R}^{d_k}$ are i.i.d. with mean 0 and variance 1 (a reasonable assumption after careful initialization). Their dot product is

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

with mean

$$
\mathbb{E}[q \cdot k] = \sum_i \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
$$

and, by independence,

$$
\text{Var}(q \cdot k) = \sum_i \text{Var}(q_i k_i) = \sum_i \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k
$$

So the standard deviation of $q \cdot k$ is $\sqrt{d_k}$. As $d_k$ grows, dot products grow, softmax peaks become extremely sharp, and gradients $\partial \text{softmax}_i / \partial s_j = \text{softmax}_i (\delta_{ij} - \text{softmax}_j)$ vanish (the softmax saturates near a one-hot vector).

Dividing by $\sqrt{d_k}$ rescales the variance back to 1, keeping softmax in its useful, gradient-friendly regime.

### Softmax as a smooth argmax

For a row of scores $s \in \mathbb{R}^n$, softmax is

$$
\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}}
$$

Properties:
- Outputs are positive and sum to 1 — a probability distribution over keys.
- Translation-invariant: $\text{softmax}(s + c) = \text{softmax}(s)$, so subtracting the max for numerical stability is exact.
- As scores become more extreme, softmax approaches a one-hot vector at the argmax.

### The attention output as a derivative-friendly retrieval

For query $q_i$:

$$
\text{Attn}_i = \sum_{j=1}^n a_{ij} v_j, \quad a_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}
$$

The whole operation is differentiable with respect to $W^Q, W^K, W^V$, so attention patterns are learned end-to-end.

### Multi-head attention formally

For head $i \in \{1, \ldots, h\}$ with projections $W^Q_i, W^K_i \in \mathbb{R}^{d \times d_k}, W^V_i \in \mathbb{R}^{d \times d_v}$:

$$
\text{head}_i = \text{Attn}(X W^Q_i, X W^K_i, X W^V_i)
$$

$$
\text{MultiHead}(X) = [\text{head}_1; \ldots; \text{head}_h] W^O
$$

with $W^O \in \mathbb{R}^{h d_v \times d}$. Typically $d_k = d_v = d/h$, so total parameters of multi-head attention are $4 d^2$ (for $W^Q, W^K, W^V, W^O$ combined across heads), independent of $h$ given fixed total dimension.

### Computational complexity

- **Time**: $O(n^2 d)$ for the $QK^\top$ multiplication and $O(n^2 d)$ for $AV$.
- **Memory**: $O(n^2 + nd)$ — the attention matrix dominates for long sequences.
- **Parallelism**: all $n$ outputs computed simultaneously; no sequential dependency across positions.

Compare to RNNs: $O(nd^2)$ time, $O(nd)$ memory, but **inherently sequential**.

## 2.4 Worked Example

Let $n = 3$, $d_k = d_v = 2$. Suppose:

$$
Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}
$$

**Step 1**: Scores $S = QK^\top$:

$$
S = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}
$$

(Row 1: query $(1,0)$ · keys $(1,0), (0,1), (1,1)$ = $1, 0, 1$.)

**Step 2**: Scale by $\sqrt{d_k} = \sqrt{2} \approx 1.414$:

$$
S' \approx \begin{bmatrix} 0.707 & 0 & 0.707 \\ 0 & 0.707 & 0.707 \\ 0.707 & 0.707 & 1.414 \end{bmatrix}
$$

**Step 3**: Row-wise softmax. For row 1, $e^{0.707} \approx 2.028, e^{0} = 1, e^{0.707} \approx 2.028$, sum $\approx 5.056$:

Row 1 ≈ $(0.401, 0.198, 0.401)$.

For row 2 by symmetry: $(0.198, 0.401, 0.401)$.

For row 3: $e^{0.707} \approx 2.028$ twice and $e^{1.414} \approx 4.113$, sum $\approx 8.169$:

Row 3 ≈ $(0.248, 0.248, 0.504)$.

**Step 4**: Output $A V$:

Row 1 of output: $0.401 \cdot (1,2) + 0.198 \cdot (3,4) + 0.401 \cdot (5,6) \approx (3.0, 4.0)$.

Row 2 of output: $0.198 \cdot (1,2) + 0.401 \cdot (3,4) + 0.401 \cdot (5,6) \approx (3.4, 4.4)$.

Row 3 of output: $0.248 \cdot (1,2) + 0.248 \cdot (3,4) + 0.504 \cdot (5,6) \approx (3.5, 4.5)$.

Notice token 3 (whose query matched its own key most strongly) ends up closest to its own value $(5, 6)$ — but blended with the others.

### Multi-head extension

If we used 2 heads with $d_k = 1$ each, we would have two separate $3 \times 3$ attention matrices, two separate output rows of dimension 1, then concatenate to dimension 2 and apply $W^O$. Each head could specialize — one might learn to attend to syntactic neighbors, another to semantic associates.

## 2.5 Relevance to Machine Learning Practice

### Where it appears
- Every modern Transformer-based model uses scaled dot-product attention as its core mixing operation.
- Used identically in encoders, decoders (with a causal mask), and cross-attention (with Q from one source, K/V from another).
- Underlies vision (ViT), speech (Whisper), code (Codex), proteins (AlphaFold's pair representation), and multimodal (CLIP, Flamingo) systems.

### When to use multi-head
- Almost always. Empirically, multiple smaller heads outperform one big head at fixed parameter count. They allow the model to attend to different types of relationships simultaneously.
- Typical choice: 8, 12, 16, 32, or 64 heads with $d_{\text{head}} = 64$ to $128$.

### Alternatives
- **Convolutions**: cheaper, locality-biased, less expressive.
- **State-space models (Mamba, S4)**: $O(n)$ in sequence length, competitive on long-range tasks; gaining traction.
- **Linear attention** (§6): trades expressiveness for speed.
- **MLP-mixers**: replace attention with token-mixing MLPs; works for vision but lacks variable-length flexibility.

### Trade-offs
- **Bias–variance**: low inductive bias means data-hungry but flexible.
- **Interpretability**: attention weights are inspectable but not faithful.
- **Robustness**: sensitive to extreme attention sinks (a few tokens absorb most attention mass).
- **Cost**: $O(n^2)$ — the central engineering challenge motivating §6.

### Production considerations
- **FlashAttention** computes attention without materializing the $n \times n$ matrix in HBM, achieving large speedups and memory savings while being numerically exact.
- **KV cache**: at autoregressive inference time, K and V from previous tokens are cached so only the new token's Q needs to be computed against all prior K/V.
- **Grouped-query attention (GQA)** and **multi-query attention (MQA)**: share K/V across multiple Q heads to shrink the KV cache, critical for fast inference of large models.

## 2.6 Common Pitfalls & Misconceptions

- **"Attention weights are explanations."** Not reliably. A token can have low attention weight but high gradient influence via the residual stream.
- **"More heads is always better."** Diminishing returns, and very small per-head dimension hurts capacity.
- **"Causal masking is optional."** For language modeling, omitting it leaks future tokens and the model trivially achieves zero loss by copying.
- **"Softmax attention is the only option."** Linear attention, kernel-based attention, and sigmoid attention exist; softmax dominates for empirical reasons but is not unique.
- **"$\sqrt{d_k}$ scaling is a minor implementation detail."** It is essential. Without it, deep Transformers don't train at all.
- **"Self-attention is permutation-invariant."** It is permutation-*equivariant*: permuting inputs permutes outputs. Without positional encoding it cannot distinguish word order, which is why §3 exists.

## 2.7 Interview Questions: Self-Attention

### Foundational

**Q1. In one sentence, what does self-attention compute?**

A weighted sum of value vectors, where the weights for each query position come from a softmax over the dot products of that query with all key vectors, all derived from the same input sequence.

**Q2. Why are queries, keys, and values projected into separate spaces rather than using the input directly?**

Three reasons. (1) Different roles need different geometry — what makes a good "search query" is not necessarily what makes a good "advertisement of content." (2) Separate projections give the model parameters to learn what relationships to attend to. (3) The dimensionality $d_k$ can be tuned independently of $d_v$ and $d$, controlling computational cost and head count.

**Q3. Why is multi-head attention used instead of a single bigger head?**

A single head can only learn one weighted-mixing pattern. Multiple heads learn diverse patterns in parallel — one might specialize in syntactic relations, another in long-range semantics, another in positional adjacency. Empirically, multi-head outperforms single-head at fixed parameter count, though there are diminishing returns past 8–32 heads.

### Mathematical

**Q4. Derive why dot products are scaled by $\sqrt{d_k}$.**

Assume $q, k \in \mathbb{R}^{d_k}$ have i.i.d. zero-mean unit-variance entries. Then $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has mean 0 and variance $\sum_i \text{Var}(q_i k_i) = \sum_i \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k$. Standard deviation is $\sqrt{d_k}$. Without scaling, softmax inputs grow with $d_k$, pushing the softmax toward a one-hot distribution where gradients $\partial_{s_j} \text{softmax}_i = \text{softmax}_i(\delta_{ij} - \text{softmax}_j)$ vanish. Dividing by $\sqrt{d_k}$ restores unit variance and keeps softmax in a gradient-friendly regime.

**Q5. What is the time and memory complexity of self-attention as a function of sequence length $n$ and dimension $d$? Compare to an RNN.**

Self-attention: $O(n^2 d)$ time, $O(n^2 + nd)$ memory, fully parallelizable across positions.
RNN: $O(n d^2)$ time, $O(nd)$ memory, but each step depends on the previous, so wall-clock time scales with $n$ even on parallel hardware.

For $n \ll d$, the RNN may be cheaper in absolute FLOPs; for long sequences, attention's $n^2$ dominates and motivates efficient variants.

**Q6. Show that self-attention is permutation-equivariant. Why is this both useful and a problem?**

If $P$ is a permutation matrix and $X' = PX$, then projections satisfy $Q' = PQ$, $K' = PK$, $V' = PV$. Scores: $S' = Q'K'^\top = PQK^\top P^\top = P S P^\top$. Softmax of $P S P^\top$ row-wise is $P A P^\top$. Output: $P A P^\top \cdot P V = P A V = P \cdot \text{Attn}(X)$. So permuting inputs permutes outputs identically — equivariance.

This is useful because the model treats sequences as sets and learns relationships from content, not position. It is a problem because for inherently ordered data (text, time series), the model cannot distinguish word orders without supplementary positional information.

**Q7. Derive the gradient of softmax-based attention w.r.t. a query vector.**

Let $a_{ij} = \text{softmax}(q_i \cdot k_j / \sqrt{d_k})$ over $j$, output $o_i = \sum_j a_{ij} v_j$. Let $L$ be a downstream loss. Then

$$
\frac{\partial L}{\partial q_i} = \sum_j \frac{\partial L}{\partial o_i} \cdot \frac{\partial o_i}{\partial a_{ij}} \cdot \frac{\partial a_{ij}}{\partial q_i}
$$

The Jacobian of softmax is $\partial a_{ij} / \partial s_{ik} = a_{ij}(\delta_{jk} - a_{ik})$, where $s_{ij} = q_i \cdot k_j / \sqrt{d_k}$, so $\partial s_{ij} / \partial q_i = k_j / \sqrt{d_k}$. Putting it together,

$$
\frac{\partial L}{\partial q_i} = \frac{1}{\sqrt{d_k}} \sum_j (\nabla_{o_i} L)^\top v_j \cdot \sum_k a_{ij}(\delta_{jk} - a_{ik}) k_k
$$

The key observation: when one $a_{ij}$ is near 1 and others near 0, $a_{ij}(\delta_{jk} - a_{ik}) \approx 0$ for all $j, k$, so the gradient vanishes. This motivates careful scaling and initialization.

### Applied

**Q8. Your Transformer training loss plateaus at a high value early on. The first thing you check is the attention. What do you look for?**

Several diagnostics:
- Are attention distributions degenerate (always uniform, or always one-hot on the same token)? Uniform suggests Q/K projections are too small; one-hot suggests scale issues.
- Are gradients vanishing through the softmax? Check $\sqrt{d_k}$ scaling is correct.
- Is the causal mask applied correctly? Off-by-one bugs are common.
- Are padding tokens being attended to? They should be masked.
- Is layer norm placement correct (pre-norm vs post-norm)?

**Q9. Why is multi-query attention (MQA) used at inference time?**

MQA uses one shared K and V across all query heads, instead of separate K/V per head. The motivation is the **KV cache**: at autoregressive decoding, every previously generated token's K and V are cached. With $h$ heads of dimension $d_k$, the cache is $h \cdot d_k$ per token. With MQA, it shrinks to just $d_k$ per token — an $h$-fold reduction. This is critical for serving long-context LLMs where the KV cache, not weights, dominates memory. The trade-off is some quality loss; **grouped-query attention** (GQA) is a middle ground that shares K/V across groups of query heads.

**Q10. You serve a chatbot with 8k context. Latency is dominated by prefill. Explain why and what you'd do.**

Prefill is the initial pass over the user's prompt where K and V for every prompt token must be computed and cached. This is an $O(n^2)$ operation for $n$ prompt tokens — the full attention matrix is computed, unlike decoding which is $O(n)$ per step thanks to the KV cache. Mitigations: FlashAttention to cut HBM traffic, prefix caching to reuse system prompts across users, sliding-window attention for very long contexts, speculative decoding (more relevant for the decoding phase), or smaller models/distillation if quality permits.

### Debugging & failure modes

**Q11. Your model trains fine but at inference time produces nonsense after the first few tokens. What's likely wrong?**

Several causes worth checking:
- **Train/test mismatch in masking**: if you trained with bidirectional attention but decode causally, or vice versa.
- **KV cache off-by-one**: appending the wrong K/V or shifting positions incorrectly.
- **Position encoding mismatch**: positions 0..n-1 at training but starting elsewhere at inference.
- **Tokenizer mismatch**: special tokens added or removed differently.
- **Sampling temperature or top-k bug**: producing degenerate outputs.

**Q12. Attention weights collapse so that one token (often [BOS] or [SEP]) absorbs most attention mass. Why might this happen and is it a bug?**

This is the well-documented **attention sink** phenomenon. It often arises because softmax forces attention weights to sum to 1, so when a head has nothing useful to do, it parks attention mass on a "safe" anchor token. This is not necessarily a bug — it can act as a no-op for the head — but in some cases it indicates over-pruning or instability. Mitigations: introduce a learnable null/sink token explicitly (StreamingLLM), use modifications like softmax-1 that allow sums less than 1, or audit whether removing the head changes downstream loss.

### Probing follow-ups

**Q13. Could you replace softmax attention with sigmoid attention? What changes?**

Yes, and recent work explores this. Sigmoid attention drops the constraint that weights sum to 1, which removes the "competition" between keys for attention mass. Pros: no softmax, easier to parallelize across keys, no normalization sink. Cons: needs different scaling and initialization, output magnitudes vary across positions, and quality has historically lagged softmax. Some recent results (e.g., Sigmoid Attention in 2024) suggest it can be competitive with proper engineering.

**Q14. Self-attention is $O(n^2)$. Why is the FFN cost typically larger in practice?**

For typical Transformer hyperparameters ($d \approx 4096$, $n \approx 2048$, FFN hidden dim $4d$), the FFN's per-layer FLOPs are roughly $2 n \cdot d \cdot 4d = 8 n d^2$, while attention is roughly $4 n^2 d + 2 n d^2$. When $n < 4d$ (often the case for moderate context lengths), the FFN dominates total compute. As context grows, attention eventually overtakes — which is why long-context inference is bottlenecked by attention, but training cost is FFN-dominated for shorter sequences.

**Q15. Could you build a Transformer with no value projection — i.e., $V = X$ directly?**

You could, but you would be tying values to the input space and losing the ability to transform "what is contributed" separately from "what is matched against" and "what searches." Empirically, removing the value projection (or tying $V = K$) hurts quality. The three projections give the model orthogonal degrees of freedom for the three roles.

---

# 3. Positional Encoding

## 3.1 Motivation & Intuition

### The order problem

Self-attention treats its input as a **set** of vectors. As we proved in §2.7 Q6, permuting the inputs permutes the outputs identically. But language is sequential: *"dog bites man"* and *"man bites dog"* contain the same words and would produce identical contextualized representations under a vanilla Transformer. We must inject **positional information** so that the model can break this permutation symmetry.

### What does "position" mean?

It is information about *where in the sequence* a token sits. The simplest possibility — appending the integer index $i$ to each token — has obvious problems: integers grow without bound, the model has never seen position 5000 if trained on length 512, and integers do not generalize naturally to "position 3.5."

Better solutions encode position in a smooth, bounded, learnable, or structured way. The literature offers four main families:

1. **Sinusoidal absolute** (original Transformer): hand-designed continuous functions of position.
2. **Learned absolute**: a learned embedding for each position.
3. **Relative**: encodings of the *distance between* positions, not absolute positions.
4. **Rotary (RoPE)** and other mixed schemes: encode position via rotations of Q and K vectors.

### A concrete intuition

Imagine each position has a unique "fingerprint" added to the token embedding. When attention computes $q_i \cdot k_j$, the fingerprints introduce a position-dependent term, allowing the model to differentiate *"the cat at position 3"* from *"the cat at position 17."* Sinusoidal encodings build these fingerprints from sines and cosines of varying frequencies — the same trick a Fourier basis uses to represent any signal.

### Connection to ML problems

- **Length generalization**: can a model trained on 2k tokens reason over 100k? Position encoding choice is the single biggest factor.
- **Transfer between tasks**: relative encodings are more robust when fine-tuning on different sequence lengths.
- **Efficient inference**: some encodings (RoPE, ALiBi) play nicer with KV caching and long-context extrapolation.
- **Vision / graphs / proteins**: position must be redefined — 2D for images, distance-based for graphs.

## 3.2 Conceptual Foundations

### Key terms

- **Absolute position**: the integer index of a token in the sequence (0, 1, 2, ...).
- **Relative position**: the signed distance $j - i$ between two positions.
- **Position embedding/encoding**: a vector representation of position. "Embedding" usually implies learned; "encoding" can mean either.
- **Permutation equivariance**: outputs permute identically with inputs; broken by positional encoding.
- **Length extrapolation**: the model's ability to handle sequences longer than seen during training.

### Sinusoidal encoding (Vaswani et al., 2017)

Define for position $\text{pos}$ and dimension index $i$:

$$
PE_{(\text{pos}, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad PE_{(\text{pos}, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

For each dimension pair $(2i, 2i+1)$, you have a sine/cosine pair at a unique frequency. Lower indices oscillate quickly (capturing fine-grained position); higher indices oscillate slowly (capturing coarse position).

The vector $PE_{\text{pos}} \in \mathbb{R}^d$ is added directly to the token embedding before the first Transformer block.

### Learned absolute encoding

Instantiate a parameter matrix $E_{\text{pos}} \in \mathbb{R}^{n_{\max} \times d}$ where $n_{\max}$ is the maximum supported sequence length. Look up row $\text{pos}$ and add it to the token embedding. Used by BERT and GPT-2.

### Relative position encoding

Instead of giving each token its own positional vector, modify attention so that the score depends on $j - i$:

- **Shaw et al. (2018)**: add a learned bias to keys based on relative distance, with distances clipped to a maximum.
- **T5 (Raffel et al., 2020)**: add a scalar bias $b_{i,j}$ to attention logits, where $b_{i,j}$ depends only on $j - i$ via learned bucketed distances.
- **Transformer-XL**: decomposes the attention score into content-content, content-position, position-content, and position-position terms.

### Rotary position embedding (RoPE)

Apply a position-dependent rotation to Q and K vectors so that the dot product $Q_i \cdot K_j$ depends on $i - j$. This elegantly merges absolute and relative encoding: each Q/K is positioned absolutely via rotation, but their inner product depends on relative offset. Used by LLaMA, GPT-NeoX, PaLM, and most modern LLMs.

### ALiBi (Attention with Linear Biases)

Add a fixed (non-learned) linear penalty $-m \cdot |i - j|$ to attention scores, where $m$ varies by head. No position embeddings, no parameters, and surprisingly strong length extrapolation.

### Underlying assumptions

- **Sinusoidal**: positions can be expressed as Fourier features; the model can learn to use them without explicit supervision.
- **Learned absolute**: enough training data exists at every position to learn meaningful embeddings.
- **Relative**: what matters is the distance between tokens, not absolute index. Holds for translation-invariant tasks (most language).
- **RoPE/ALiBi**: position effects can be encoded as smooth functions of distance.

### What breaks

- **Sinusoidal at very long lengths**: model has not seen those frequencies during training and may not extrapolate well in practice.
- **Learned absolute beyond max position**: undefined behavior; most implementations clip or reuse the last embedding.
- **Relative encodings with overly aggressive bucketing**: tokens that are far apart all get bucketed together, losing distance resolution.
- **ALiBi**: works less well when fine-grained order distinctions matter (e.g., counting tasks).
- **All schemes**: if position is fundamentally 2D or graph-based, 1D schemes are inadequate.

## 3.3 Mathematical Formulation

### Sinusoidal encoding details

For embedding dimension $d$ and position $\text{pos} \in \{0, 1, \ldots\}$, define

$$
\omega_i = \frac{1}{10000^{2i/d}}, \quad i \in \{0, 1, \ldots, d/2 - 1\}
$$

Then $PE_{\text{pos}}$ has alternating entries $\sin(\omega_i \text{pos}), \cos(\omega_i \text{pos})$.

**Why these specific functions?**

1. **Bounded**: $|PE| \leq 1$ for every entry, so adding to embeddings doesn't dominate.
2. **Unique fingerprint per position**: with $d/2$ different frequencies, the multidimensional fingerprint is unique up to extremely long ranges.
3. **Linear shift property**: $PE_{\text{pos} + k}$ can be expressed as a linear function (rotation) of $PE_{\text{pos}}$ and $PE_k$:

$$
\begin{bmatrix} \sin(\omega_i (\text{pos} + k)) \\ \cos(\omega_i (\text{pos} + k)) \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} \sin(\omega_i \text{pos}) \\ \cos(\omega_i \text{pos}) \end{bmatrix}
$$

This is a 2D rotation matrix — meaning the model can in principle learn to extract relative position $k$ as a linear operation on absolute encodings.

### Learned absolute (formal)

Let $E_{\text{pos}} \in \mathbb{R}^{n_{\max} \times d}$ be a parameter. Token at position $i$ with embedding $e_i$ is fed into the first block as $e_i + E_{\text{pos}}[i]$. No structure imposed — purely learned.

### Relative encoding (Shaw)

Modify attention as

$$
e_{ij} = \frac{(x_i W^Q)(x_j W^K + a^K_{i-j})^\top}{\sqrt{d_k}}
$$

where $a^K_{i-j} \in \mathbb{R}^{d_k}$ is a learned vector keyed by clipped relative position. Values get an analogous term $a^V_{i-j}$.

### T5 relative bias

Skip vector additions and just add a scalar bias to attention logits:

$$
e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{i-j}
$$

where $b_{\cdot}$ comes from a learned bucketed function of $i - j$. T5 uses log-spaced buckets so distant tokens share buckets while nearby tokens have unique ones.

### RoPE derivation

Define a rotation matrix in 2D for position $m$:

$$
R(m, \theta) = \begin{bmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{bmatrix}
$$

For a query/key vector in $\mathbb{R}^{d_k}$, split into $d_k / 2$ pairs and apply a 2D rotation to each pair using a unique $\theta_i = 10000^{-2i/d_k}$. So the rotated query at position $m$ is

$$
\tilde{q}_m = R_m \, q
$$

where $R_m$ is block-diagonal of 2D rotations. Similarly for $\tilde{k}_n$. Then

$$
\tilde{q}_m \cdot \tilde{k}_n = q^\top R_m^\top R_n k = q^\top R_{n - m} k
$$

The dot product depends only on the relative offset $n - m$. Beautiful: absolute rotation per token, relative effect on attention scores.

### ALiBi formal

$$
e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + m_h \cdot (j - i)
$$

where $m_h$ is a head-specific slope (geometrically spaced across heads). For causal LM, only $j \leq i$ matters, so the bias is $-m_h (i - j) \leq 0$, penalizing distant tokens.

## 3.4 Worked Example

Let $d = 4$, position $\text{pos} \in \{0, 1, 2\}$.

### Sinusoidal

Frequencies: $\omega_0 = 10000^0 = 1$, $\omega_1 = 10000^{-2/4} = 10000^{-0.5} = 0.01$.

| pos | $\sin(\omega_0 \cdot \text{pos})$ | $\cos(\omega_0 \cdot \text{pos})$ | $\sin(\omega_1 \cdot \text{pos})$ | $\cos(\omega_1 \cdot \text{pos})$ |
|-----|--------------------|--------------------|--------------------|--------------------|
| 0   | 0.000              | 1.000              | 0.000              | 1.000              |
| 1   | 0.841              | 0.540              | 0.010              | 1.000              |
| 2   | 0.909              | -0.416             | 0.020              | 1.000              |

Each row is a 4-dim positional encoding. Notice the high-frequency dims (0,1) change quickly with position; the low-frequency dims (2,3) barely move over 3 positions.

### RoPE on a 4-dim query

Suppose query $q = (1, 0, 0, 1)$ and we apply RoPE at position $m = 2$. With the same $\theta_0 = 1, \theta_1 = 0.01$:

Rotation of pair $(q_0, q_1) = (1, 0)$ by $m\theta_0 = 2$ rad:

$$
\begin{pmatrix} \cos 2 & -\sin 2 \\ \sin 2 & \cos 2 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} -0.416 \\ 0.909 \end{pmatrix}
$$

Rotation of pair $(q_2, q_3) = (0, 1)$ by $m\theta_1 = 0.02$ rad:

$$
\begin{pmatrix} \cos 0.02 & -\sin 0.02 \\ \sin 0.02 & \cos 0.02 \end{pmatrix} \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} -0.020 \\ 1.000 \end{pmatrix}
$$

So $\tilde{q}_2 = (-0.416, 0.909, -0.020, 1.000)$. Compare positions: at $m=0$ the rotation is identity, so $\tilde{q}_0 = (1, 0, 0, 1)$. At larger $m$, the high-frequency pair (first two dims) rotates substantially while the low-frequency pair barely moves.

### Sinusoidal as a "clock with multiple hands"

Think of each frequency as one hand on a clock. The fastest hand wraps around every $2\pi$ time steps; the slowest takes $2\pi \cdot 10000$ steps. Reading all hands together gives a unique time, and the relative difference between two times is recoverable from the hand positions — which is what the linear-shift property formalizes.

## 3.5 Relevance to Machine Learning Practice

### Where each scheme is used

- **Sinusoidal**: original Transformer, Whisper.
- **Learned absolute**: BERT, GPT-2, ViT.
- **Relative (Shaw)**: Music Transformer, some vision Transformers.
- **T5 relative bias**: T5, mT5, FLAN-T5.
- **RoPE**: LLaMA 1/2/3, GPT-NeoX, PaLM, most open-source LLMs from 2022 onward.
- **ALiBi**: BLOOM, MPT, original release of MosaicML models.

### Choosing between them

| Scheme | Length extrapolation | Compute overhead | Empirical strength |
|---|---|---|---|
| Sinusoidal | OK | Free | Baseline |
| Learned absolute | Poor | Tiny | Strong in-distribution |
| T5 relative | Good | Small | Strong |
| RoPE | Good (with tricks) | Small | Strong, dominant in 2024–25 LLMs |
| ALiBi | Excellent | Free | Strong, especially for long context |

### Length extrapolation tricks

For RoPE specifically:
- **Position interpolation**: divide positions by a factor $s > 1$ to compress longer contexts into the trained range.
- **YaRN, NTK-aware scaling**: smarter ways to adjust $\theta_i$ to extrapolate while preserving high-frequency precision.
- **Continued pretraining on longer sequences**: still required for best results.

### When to use what

- **Building a new LLM**: RoPE is the default in 2025.
- **Fine-tuning an existing model**: stick with whatever it was pretrained with — switching position encodings mid-training rarely works.
- **Very long context (>100k tokens)**: RoPE with scaling, ALiBi, or specialized attention (e.g., sliding window) become necessary.
- **Vision (2D positions)**: 2D sinusoidal or learned 2D embeddings; ViT typically uses 1D learned embeddings over flattened patches.

## 3.6 Common Pitfalls & Misconceptions

- **"Sinusoidal encodings extrapolate perfectly to any length."** Mathematically the function is defined for any pos, but the model's *use* of these features may not generalize. In practice, long-context extrapolation usually requires retraining or special schemes.
- **"Learned absolute embeddings are strictly worse than relative ones."** Not always — at fixed length, learned absolute can be competitive. The asymmetry shows up in length generalization.
- **"RoPE is just a fancier sinusoidal."** RoPE *applies* the encoding multiplicatively (rotation) inside attention, whereas sinusoidal *adds* it once at the embedding. The mechanism is meaningfully different and gives RoPE its relative-encoding property.
- **"You should drop position encoding entirely if your data is bag-of-words."** True for unordered data, but most "bag" tasks still benefit from position encoding because the surrounding sequence has structure.
- **"Adding two different position encodings will make the model more robust."** Usually destructive — they conflict, and one wins arbitrarily.
- **"ALiBi can't model bidirectional attention well."** It can, but the bias must be made symmetric in $|i - j|$.

## 3.7 Interview Questions: Positional Encoding

### Foundational

**Q1. Why does a Transformer need positional encoding?**

Self-attention is permutation-equivariant: permuting the input permutes the output identically. Without positional information, the model cannot distinguish "dog bites man" from "man bites dog." Positional encoding injects order information so the attention mechanism can break this symmetry.

**Q2. Compare absolute and relative position encodings in one or two sentences each.**

Absolute encodings give each position $i$ a unique vector that is added to the token embedding once, before the first block. Relative encodings give each *pair* of positions a representation of their offset $j - i$, which directly modifies attention scores; this localizes positional reasoning to the attention computation and tends to generalize better across sequence lengths.

**Q3. What is the intuition behind sinusoidal encoding's choice of multiple frequencies?**

The encoding uses sines and cosines at geometrically decreasing frequencies, like the hands of a multi-frequency clock. High frequencies distinguish nearby positions sharply; low frequencies distinguish coarse position over long ranges. Together they form a unique, smooth fingerprint per position.

### Mathematical

**Q4. Prove that sinusoidal positional encodings allow $PE_{\text{pos} + k}$ to be expressed as a linear function of $PE_{\text{pos}}$.**

For a single (sin, cos) pair at frequency $\omega$:

$$
\begin{bmatrix} \sin(\omega(\text{pos} + k)) \\ \cos(\omega(\text{pos} + k)) \end{bmatrix} = \begin{bmatrix} \sin\omega\text{pos}\cos\omega k + \cos\omega\text{pos}\sin\omega k \\ \cos\omega\text{pos}\cos\omega k - \sin\omega\text{pos}\sin\omega k \end{bmatrix}
$$

$$
= \begin{bmatrix} \cos\omega k & \sin\omega k \\ -\sin\omega k & \cos\omega k \end{bmatrix} \begin{bmatrix} \sin\omega\text{pos} \\ \cos\omega\text{pos} \end{bmatrix}
$$

This is a rotation by angle $\omega k$ — a fixed linear map depending only on $k$. So $PE_{\text{pos} + k} = R(k) \cdot PE_{\text{pos}}$ where $R(k)$ is block-diagonal of 2D rotations. This means the model could in principle learn to compute relative offsets by linear projections of absolute encodings.

**Q5. Derive why RoPE makes the attention dot product depend only on relative position.**

Let $R_m$ be the block-diagonal rotation matrix for position $m$ (block $i$ rotates by $m\theta_i$). Apply $R_m$ to the query: $\tilde{q}_m = R_m q$, and $R_n$ to the key: $\tilde{k}_n = R_n k$. Then

$$
\tilde{q}_m^\top \tilde{k}_n = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

since rotations are orthogonal ($R_m^\top = R_{-m}$) and compose additively in their rotation angles ($R_a R_b = R_{a+b}$). The result depends only on $n - m$ — relative position — even though each token was rotated by its absolute position.

**Q6. Why might learned absolute position embeddings fail to generalize beyond max training length, while sinusoidal might?**

Learned embeddings are parameters: positions outside the trained range simply have no defined vector. Even with extrapolation tricks (mirror, repeat), the model has never observed the resulting vectors and can't be expected to interpret them.

Sinusoidal embeddings are functions of position — they extend mathematically to any value. However, the *downstream weights* that consume these features were trained only on values within the training range, so the model's behavior on extrapolated positions is empirically unreliable. RoPE and ALiBi mitigate this with smoother distance-dependent biases.

### Applied

**Q7. You're fine-tuning a 4k-context model on a task requiring 16k context. The performance drops dramatically. What's the likely cause and what would you try?**

The model was pretrained with positional encodings (likely RoPE or learned) tuned for 4k. At 16k, RoPE positions exceed the trained range, causing distribution shift in attention scores. Options:

- **Position interpolation** (Chen et al., 2023): scale positions by 4 to compress 16k into the 4k range, then continue training briefly.
- **NTK-aware scaling / YaRN**: more sophisticated frequency adjustments preserving high-frequency precision.
- **Continued pretraining on long sequences**: even short training (a few hundred steps) on 16k examples helps significantly.
- **Switch to a model with ALiBi or trained-for-long-context** if available.

If positions are learned absolute, you'd need to extend the embedding table and continue training — usually less effective than RoPE-based approaches.

**Q8. For a music generation model where exact rhythmic position matters (e.g., 4/4 time signature with downbeats every 4 steps), which positional encoding would you choose and why?**

Relative encodings tend to win for rhythmic tasks because they make periodicity explicit (the same offset $-4$ tokens always means "previous downbeat"). Shaw-style learned relative encodings or T5-style relative biases would be strong choices. Pure absolute encodings would also work but require the model to learn modular arithmetic implicitly.

### Debugging & failure modes

**Q9. A model performs well on 512-token inputs but produces gibberish on 2k-token inputs at inference time, despite being "trained for" 2k. What might be wrong?**

Check the data distribution: if 95% of training examples were ≤512 tokens, the model may have effectively never learned positions 512–2047. Solutions: rebalance training data, use length-stratified sampling, or introduce curriculum that progressively lengthens. Also verify that positional encodings are correctly computed at inference (off-by-one errors in batched RoPE are common).

**Q10. After switching from learned absolute to RoPE, eval loss goes up. What could be happening?**

Several possibilities:
- The RoPE base frequency ($\theta$) is mismatched to the data's typical sequence length.
- Attention heads were tuned to specific learned-position patterns; with RoPE, those heads behave differently and need retraining.
- RoPE was applied to the wrong dimensions or only to some heads.
- The model needs longer training to adapt; switching position encoding mid-training is generally not a hot-swap operation.

### Probing follow-ups

**Q11. Could a Transformer learn position implicitly without any positional encoding?**

In principle, with causal attention, a decoder can learn to count tokens (e.g., the first token is always the only one that attends to nothing besides itself). Studies have shown causal-only Transformers without positional encoding can achieve competitive performance on language modeling — the asymmetry of the causal mask provides positional information. Encoder-only (bidirectional) models cannot, since their attention is symmetric without positional cues.

**Q12. How does positional encoding interact with multi-head attention?**

In the original Transformer, position is added once and shared across all heads. Different heads then learn different *uses* of this shared positional information. In RoPE, the rotation is applied identically across heads but each head can learn to rely on or ignore different frequencies. In ALiBi, each head has its own slope $m_h$, deliberately diversifying head behavior — some heads focus locally, others globally.

**Q13. ALiBi has zero learned positional parameters. Why does it sometimes outperform learned schemes?**

Three reasons. (1) Inductive bias: the linear penalty matches a real prior — distant tokens are less relevant on average. (2) Length extrapolation: the bias function is defined for any distance, with no parameters that fail to generalize. (3) Simplicity reduces overfitting on finite training data. The trade-off: ALiBi imposes that bias structure even when it's wrong (e.g., long-range dependencies in code).

**Q14. Can position encodings be combined with content-dependent gating?**

Yes — schemes like Transformer-XL decompose attention into content and positional terms. Some recent variants (e.g., contextualized positional encodings) make positional contributions depend on the content, allowing position-aware retrieval that adjusts to what's being looked up. This adds flexibility but also parameters and complexity.

---

# 4. The Transformer Block

## 4.1 Motivation & Intuition

### What the block has to do

A single Transformer block must transform a sequence of token representations into another sequence of equal length and dimension, where each output captures both its own token's information *and* useful context from other tokens. Two ingredients are needed:

1. **Cross-position mixing**: information from other tokens must flow in. This is what self-attention does.
2. **Per-position computation**: once a token has gathered information, it needs to *do something with it* — apply nonlinear transformations, store/retrieve facts, project into useful subspaces. This is what the feed-forward network does.

Plus two engineering ingredients to make the deep stack trainable:

3. **Residual connections**: $y = x + f(x)$, so that gradients flow easily and the model can learn to modify rather than replace the input.
4. **Layer normalization**: stabilizes activations across a deep stack, especially important without a recurrent gating mechanism.

### Why this design works so well

Each block alternates between "look at others" (attention) and "think alone" (FFN). Residual connections create a "**residual stream**" that carries the running representation through the network; each block reads from and writes to this stream. Layer normalization keeps the magnitudes consistent so that no block dominates the next via amplitude alone.

Empirically, this block design has proven extraordinarily robust — variants exist, but the core attention-then-FFN pattern with residuals and norms has dominated since 2017.

### Concrete intuition: the FFN as a key-value memory

A useful interpretation (Geva et al., 2021): the FFN acts as a *key-value memory*. The first linear layer maps the input to similarities with thousands of "memory keys" (rows of $W_1$). The activation function gates which memories activate. The second linear layer reads out the corresponding "memory values" (columns of $W_2$), summing them. So the FFN looks up facts and rewrites the residual stream with retrieved information.

## 4.2 Conceptual Foundations

### Key terms

- **Sublayer**: a building block within a Transformer block, either attention or FFN.
- **Residual connection (skip connection)**: $y = x + \text{Sublayer}(x)$.
- **Layer normalization (LayerNorm)**: per-token normalization that subtracts mean and divides by std across the feature dimension, then applies learned affine transformation.
- **Pre-norm**: LayerNorm applied *before* the sublayer: $y = x + \text{Sublayer}(\text{LN}(x))$.
- **Post-norm**: LayerNorm applied *after* the residual sum: $y = \text{LN}(x + \text{Sublayer}(x))$.
- **Feed-forward network (FFN)**: a 2-layer MLP applied independently to each position.
- **GLU / SwiGLU / GeGLU**: gated variants of the FFN with elementwise multiplications.
- **Residual stream**: the running representation passed forward through residual additions; a useful mental model from interpretability research.

### How the components interact

Single block, post-norm (original):

```
x → MultiHeadAttn(x) → +x → LayerNorm → z
z → FFN(z) → +z → LayerNorm → y
```

Single block, pre-norm (modern):

```
x → LayerNorm → MultiHeadAttn → +x → z
z → LayerNorm → FFN → +z → y
```

The output $y$ has the same shape as $x$ and is fed to the next block.

### FFN structure

The standard FFN is:

$$
\text{FFN}(x) = W_2 \, \sigma(W_1 x + b_1) + b_2
$$

with $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d}$ and $W_2 \in \mathbb{R}^{d \times d_{\text{ff}}}$. Typically $d_{\text{ff}} = 4d$. $\sigma$ is ReLU (original), GELU (BERT/GPT-2), or SwiGLU (LLaMA).

**SwiGLU** variant:

$$
\text{SwiGLU}(x) = (W_1 x \odot \text{Swish}(W_g x)) W_2
$$

introduces a gate $W_g$ for elementwise multiplicative control. Empirically improves quality at fixed parameter count, and is used by LLaMA, PaLM, and most modern LLMs.

### LayerNorm details

For input vector $x \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2
$$

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

with learned $\gamma, \beta \in \mathbb{R}^d$. Importantly, normalization is **per token** (across the feature axis), independent of batch size or sequence length.

**RMSNorm** drops the mean subtraction and the additive bias:

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}
$$

Used by LLaMA. Cheaper, often as good as LayerNorm.

### Residual connections: what they do

For a sublayer $f$, $y = x + f(x)$ has two important effects:

1. **Identity initialization / skip path**: if $f$ is initialized near zero (or normalized to small magnitude), $y \approx x$, so deep stacks behave like shallow ones at the start of training.
2. **Gradient flow**: $\partial y / \partial x = I + \partial f / \partial x$. Even if $\partial f / \partial x$ is tiny, the identity term prevents gradient vanishing through depth.

### Underlying assumptions

- **Normalization is needed for stability.** True for deep networks (>10 layers); shallow networks can sometimes train without.
- **Residuals enable depth.** Empirically yes — vanilla pre-residual networks fail to train past ~10 layers.
- **FFN expansion ratio of 4× is appropriate.** Established empirically; recent work (chinchilla-optimal, gated activations) revisits this.
- **Per-token normalization is the right granularity.** True for sequence models with variable lengths. Batch normalization fails because batch statistics shift wildly with sequence length and batching.

### What breaks when assumptions fail

- **No residuals**: gradients vanish, training fails past a few layers.
- **No normalization**: activation magnitudes drift, training diverges.
- **Wrong norm placement (post-norm at scale)**: requires careful warmup and learning rate scheduling; GPT-3 and later moved to pre-norm to avoid this fragility.
- **Tiny FFN**: model is parameter-starved for memorization; quality drops.
- **BatchNorm in a Transformer**: gives noisy normalization due to variable lengths and small effective batch (per token); usually fails.

## 4.3 Mathematical Formulation

### Full block (pre-norm, modern)

For input $x_l \in \mathbb{R}^{n \times d}$ at layer $l$:

$$
\tilde{x}_l = x_l + \text{MultiHeadAttn}(\text{LN}(x_l))
$$

$$
x_{l+1} = \tilde{x}_l + \text{FFN}(\text{LN}(\tilde{x}_l))
$$

The residual stream $x_l$ flows additively from layer 0 (token embeddings + positional encoding) to layer $L$ (final representation).

### Why pre-norm beats post-norm at scale

In post-norm:

$$
x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))
$$

The norm is *outside* the residual, so the residual stream itself gets renormalized at every layer. This causes the magnitude of the "shortcut" path to be re-anchored, making early-training dynamics highly sensitive to initialization.

In pre-norm:

$$
x_{l+1} = x_l + \text{Sublayer}(\text{LN}(x_l))
$$

The residual is unmodified; only the *input to the sublayer* is normalized. The skip path is a clean identity, and the network behaves at initialization like a much shallower model. This makes pre-norm dramatically easier to train at depth $> 50$.

### Parameter count of a block

Assuming pre-norm, multi-head attention with $h$ heads and $d_k = d_v = d/h$:

- Attention: $W^Q, W^K, W^V, W^O$ each $d \times d$ → $4 d^2$ parameters.
- FFN: $W_1: d \times 4d$, $W_2: 4d \times d$ → $8 d^2$ parameters.
- Norms: 4d parameters per LayerNorm × 2 ≈ negligible.

Total per block: $\approx 12 d^2$. For a model with $L$ blocks, total parameters $\approx 12 L d^2 + V d$ (last term is the embedding/unembedding, $V$ = vocab size).

For BERT-base ($L=12, d=768$): $12 \cdot 12 \cdot 768^2 \approx 85M$ parameters in the blocks, plus embeddings.

For LLaMA-7B ($L=32, d=4096$): $12 \cdot 32 \cdot 4096^2 \approx 6.4B$ — dominated by the blocks.

### FLOPs per token per block

Attention contributes roughly $4 n d^2$ (projections) plus $4 n^2 d$ (the $QK^\top$ and $AV$ matmuls). FFN contributes roughly $16 n d^2$ (two matmuls of $n d \times 4d$ and $n \cdot 4d \times d$). For $n \ll 4d$, FFN dominates; for $n \gg 4d$, attention dominates.

## 4.4 Worked Example

Let's trace a single token through one block. Set $d = 4$, $d_{\text{ff}} = 16$, single attention head, pre-norm.

Input to block (one position): $x = [1.0, 2.0, -1.0, 0.5]$.

**Step 1: LayerNorm.**

$\mu = 0.625$, $\sigma^2 = ((0.375)^2 + (1.375)^2 + (-1.625)^2 + (-0.125)^2)/4 = (0.141 + 1.891 + 2.641 + 0.016)/4 = 1.172$.

$\sigma \approx 1.082$. With $\gamma = 1, \beta = 0$:

$\text{LN}(x) \approx [(1.0 - 0.625)/1.082, (2.0 - 0.625)/1.082, (-1.0 - 0.625)/1.082, (0.5 - 0.625)/1.082]$
$\approx [0.347, 1.271, -1.502, -0.116]$.

**Step 2: Self-attention (single head).** Suppose attention returns (for illustration) $[0.2, -0.1, 0.0, 0.5]$.

**Step 3: Residual.** $\tilde{x} = x + \text{attn output} = [1.2, 1.9, -1.0, 1.0]$.

**Step 4: LayerNorm of $\tilde{x}$.** Computed similarly; suppose result is $[0.20, 1.06, -1.65, 0.39]$.

**Step 5: FFN.** $W_1 \in \mathbb{R}^{16 \times 4}$, $W_2 \in \mathbb{R}^{4 \times 16}$, GELU activation.

Suppose $W_1 \cdot \text{LN}(\tilde{x}) + b_1$ gives some 16-dim hidden. Apply GELU elementwise. Apply $W_2$ to project back to 4-dim. Suppose the result is $[-0.1, 0.3, 0.2, -0.4]$.

**Step 6: Residual.** $x_{\text{next}} = \tilde{x} + \text{FFN output} = [1.1, 2.2, -0.8, 0.6]$.

This $x_{\text{next}}$ is the input to the next block. Notice the residual stream's overall magnitude stays similar to the input — the block has *modified*, not *replaced*, the representation.

### Position-wise nature of the FFN

If we had a sequence of 3 tokens, the same $W_1, W_2$ would be applied independently to each. There is no information mixing in the FFN — that's exclusively the attention sublayer's job.

## 4.5 Relevance to Machine Learning Practice

### Where it shows up

- **Every Transformer**: this block is the unit of repetition. A 12-layer BERT, a 96-layer GPT-3, or a 32-layer LLaMA are all stacks of this same block (with minor variations).
- **Mixture-of-experts (MoE)**: replaces the FFN with a gated mixture of expert FFNs (Switch Transformer, Mixtral, GShard). Attention is unchanged.
- **Adapter / LoRA**: typically inserted as small bottleneck modules around or inside the FFN and attention sublayers.

### Pre-norm vs post-norm in practice

- **Pre-norm**: easier to train at scale, the dominant choice in 2025. Slightly weaker final quality at small/moderate scales (some papers report).
- **Post-norm**: original choice, better quality if you can get it to train; needs careful warmup. Used by original BERT.
- **Hybrid (sandwich norm, deepnet)**: extra normalization before *and* after sublayers; helps very deep networks.

### Activation choices

- **ReLU**: original, fast, but "dying ReLU" risk.
- **GELU**: smoothed ReLU; standard for BERT, GPT-2/3.
- **SwiGLU / GeGLU**: gated variants; introduce a third weight matrix (so the FFN has $\sim 12 d^2$ parameters instead of $8 d^2$), but improve quality enough that the parameter trade-off favors them. Standard in LLaMA, PaLM.

### When to use what

- **Always use residuals and norm** in production Transformers.
- **Pre-norm** for any deep model (>10 layers).
- **SwiGLU / GeGLU** for new LLM training; ReLU/GELU when matching legacy architectures.
- **MoE FFN** when you want to scale parameter count without scaling FLOPs (Mixtral, Switch).

### Trade-offs

- **FFN expansion ratio**: 4× is conventional. Smaller ratios save compute but hurt capacity; larger ratios are expensive.
- **Pre-norm**: easier training, marginally weaker quality at small scale.
- **Layer norm vs RMSNorm**: RMSNorm is ~10–20% cheaper, comparable quality; standard in modern LLMs.

## 4.6 Common Pitfalls & Misconceptions

- **"FFN is the unimportant part of a Transformer."** False — it's roughly 2/3 of parameters and a comparable share of compute. Many Transformer "facts" live in FFN weights.
- **"BatchNorm and LayerNorm are interchangeable."** No — BatchNorm depends on batch statistics, which are noisy with variable-length sequences and break at small batch sizes. LayerNorm normalizes per token.
- **"Pre-norm and post-norm are mathematically identical."** They differ in where the normalization sits relative to the residual; the gradient and training dynamics differ substantially.
- **"You can scale model depth indefinitely with pre-norm."** Pre-norm helps but doesn't fully solve very-deep training; methods like DeepNorm, normalization scaling, or layer dropping become necessary past 100+ layers.
- **"Removing the FFN gives a faster but slightly worse model."** Removing the FFN typically destroys quality. Attention alone cannot do the per-position computation needed for memorization and reasoning.
- **"Layer normalization removes signal magnitude entirely."** It removes per-token magnitude variation but preserves direction; the learned $\gamma$ allows magnitude back per dimension.

## 4.7 Interview Questions: The Transformer Block

### Foundational

**Q1. What two operations does a Transformer block alternate between, and why both?**

Self-attention (cross-position mixing) and a position-wise FFN (per-position nonlinear transformation). Attention gathers relevant information from across the sequence; the FFN then processes that information independently at each position. Without attention, the model can't share context across tokens; without the FFN, it has no way to apply rich nonlinear transformations.

**Q2. What is a residual connection and what problem does it solve?**

A residual connection adds the input of a sublayer to its output: $y = x + f(x)$. It solves two problems: (1) gradient vanishing in deep networks, since $\partial y / \partial x = I + \partial f / \partial x$ has an identity shortcut; (2) optimization difficulty, since the network can default to identity (when $f \approx 0$) and learn modifications incrementally.

**Q3. Why is LayerNorm used instead of BatchNorm in Transformers?**

LayerNorm normalizes across the feature dimension within a single token, so it works regardless of batch size and sequence length. BatchNorm normalizes across the batch dimension, which is unstable for variable-length sequences (different positions get different effective batches) and fails at small batch sizes (common with large models due to memory limits). LayerNorm also doesn't introduce train/inference statistics mismatch.

### Mathematical

**Q4. Derive the parameter count of one Transformer block as a function of $d$ (model dim) and $d_{\text{ff}}$ (FFN hidden dim), assuming standard architecture.**

- Multi-head attention: 4 weight matrices ($W^Q, W^K, W^V, W^O$), each $d \times d$ → $4d^2$.
- FFN: $W_1: d \to d_{\text{ff}}$ has $d \cdot d_{\text{ff}}$, $W_2: d_{\text{ff}} \to d$ has $d_{\text{ff}} \cdot d$. Total $2 d \cdot d_{\text{ff}}$.
- Norms: $2 \times 2d$ (gamma + beta per LayerNorm × 2 LayerNorms) — negligible.
- Biases: $\sim 2d + 2 d_{\text{ff}}$ — negligible.

Total per block: $\approx 4 d^2 + 2 d \cdot d_{\text{ff}}$. With $d_{\text{ff}} = 4d$: $\approx 12 d^2$.

**Q5. Show why pre-norm enables stable training at large depth where post-norm struggles.**

In pre-norm, $x_{l+1} = x_l + g(x_l)$ where $g(x) = \text{Sublayer}(\text{LN}(x))$. The residual stream $x_l$ accumulates additively without renormalization: $x_L = x_0 + \sum_{l=0}^{L-1} g_l(x_l)$. The norm of $x_L$ grows roughly as $\sqrt{L}$ (sum of independent perturbations), but the gradient through the skip path is the identity, so signals propagate freely.

In post-norm, $x_{l+1} = \text{LN}(x_l + \text{Sublayer}(x_l))$. The renormalization at each layer means that the magnitude of the sublayer's contribution must not be much smaller than the residual, or it gets washed out; conversely, if it's much larger, the residual is washed out. Achieving balance requires careful warmup (the LR is ramped up slowly from near zero), and at large depth even warmup is insufficient.

**Q6. The FFN typically uses $d_{\text{ff}} = 4d$. Why not $1d$ or $16d$?**

It's an empirical sweet spot. $d_{\text{ff}}$ controls the FFN's capacity to act as a key-value memory — too small and there are too few "memory slots" to store useful patterns; too large and parameters and FLOPs explode without proportional gains. The 4× ratio was chosen in the original Transformer based on validation performance and has held up empirically. Some recent gated FFN variants (SwiGLU) use $d_{\text{ff}} \approx 2.67 d$ to keep parameter count comparable, since the gate adds a third matrix.

### Applied

**Q7. You're training a 100-layer Transformer and it diverges in the first 1000 steps. What architectural choices might fix this?**

Several options, in order of typical impact:
- **Switch from post-norm to pre-norm**: the most common fix.
- **Use DeepNorm** (Wang et al., 2022): scales residual contributions to maintain stable activations to arbitrary depth.
- **Add LR warmup**: necessary even with pre-norm at extreme depths.
- **Reduce initialization magnitude** (e.g., scale residuals at init by $1/\sqrt{2L}$): keeps initial activations bounded.
- **Gradient clipping** at a tighter norm.
- **Reduce learning rate or use learning rate schedules with longer warmup.**

**Q8. For inference latency, where would you focus optimization in the Transformer block?**

Depends on regime:
- **Long-context inference** (large $n$): attention dominates. FlashAttention, paged attention, KV cache compression, sliding-window attention.
- **Short-context inference, large model**: FFN dominates. Quantization (int8, int4), MoE (only some experts active per token), tensor parallelism.
- **Small batch decoding**: memory-bandwidth bound. Quantization is most impactful.
- **Large batch prefill**: compute bound. Tensor cores via mixed precision.

### Debugging & failure modes

**Q9. After replacing LayerNorm with RMSNorm, your model trains fine but final perplexity is slightly worse. What might explain this and would you accept the trade-off?**

RMSNorm removes the mean subtraction and the additive bias, slightly reducing model capacity. The expressivity loss is small but real. Whether to accept the trade-off depends on:
- The magnitude of perplexity gap (small: accept for compute savings).
- Inference latency requirements (RMSNorm is faster; matters for production).
- Whether downstream tasks are bottlenecked by perplexity or by other factors.

In modern LLM training, RMSNorm is preferred because the compute savings compound across millions of steps and the quality gap is usually negligible.

**Q10. Activations in the residual stream blow up to magnitudes of 10^4 by layer 30. What are likely causes?**

The residual stream grows as the sum of sublayer contributions. If each contribution has magnitude ~1, after 30 layers the sum has magnitude ~$\sqrt{30}$ ≈ 5.5 (assuming roughly orthogonal contributions). Magnitudes of $10^4$ suggest:
- Sublayer outputs are large (poor initialization, missing scaling factor).
- Particular heads or FFN units are producing extreme outputs (worth inspecting per-layer activation norms).
- Pre-norm with no scaling at residuals; consider DeepNorm-style scaling.
- Numerical precision issues in fp16 (use bf16 or normalization tricks).

### Probing follow-ups

**Q11. The "residual stream" interpretability framing treats each block as reading from and writing to a shared communication channel. What predictions does this framing make?**

Predictions include: (1) different blocks specialize in different operations, since they all share the same channel; (2) information added by block $l$ persists until block $l'>l$ deliberately erases or modifies it; (3) intervening on the residual stream at a specific layer should produce predictable downstream effects (verified by activation patching); (4) you can decompose the unembedding logits as a linear sum of contributions from each layer (logit lens). These predictions have been broadly confirmed in interpretability research (Elhage et al., 2021; Anthropic's transformer-circuits work).

**Q12. Could you replace the FFN with another attention layer? What would change?**

Yes, this gives an "all-attention" Transformer. Empirically, it underperforms the standard Attn+FFN block at fixed parameter count. Why: attention is good at routing information, while FFN's per-position MLP is good at applying complex nonlinear transformations and storing facts. The two are complementary. Variants like "attention-only" Transformers exist mostly in interpretability research where simpler architectures are easier to analyze.

**Q13. SwiGLU adds a third weight matrix compared to a standard FFN. Why is it still considered a parameter-efficient improvement?**

SwiGLU is $\text{SwiGLU}(x) = (W_1 x \odot \text{Swish}(W_g x)) W_2$. Three matrices instead of two. To match the parameter count of a standard $4d$ FFN, you typically reduce $d_{\text{ff}}$ to $\approx 2.67d$ ($\approx 8/3 d$). Even with the smaller hidden dim, SwiGLU outperforms the standard FFN at equal parameter count, because the multiplicative gating provides richer expressivity than a simple ReLU/GELU activation. So: same parameters, better performance — hence "parameter-efficient improvement."

**Q14. In a Mixture-of-Experts model, the FFN is replaced with a router + many expert FFNs. What is preserved from the standard block?**

Self-attention, residual connections, and LayerNorm all remain identical. Only the FFN sublayer is replaced. The router selects the top-$k$ experts (typically $k = 1$ or $2$) per token, and only those experts compute their outputs, which are weighted-summed. This means total parameters scale with the number of experts, but compute per token scales with $k$ — so MoE models can have hundreds of billions of parameters while running at the cost of a much smaller dense model. The challenge is load balancing (preventing all tokens from routing to the same expert) and communication overhead in distributed training.

---

# 5. Encoder–Decoder Architecture and Cross-Attention

## 5.1 Motivation & Intuition

### Two distinct problems

So far we've described "encoder-only" (BERT-like) and implicitly "decoder-only" (GPT-like) Transformers. The original Transformer was actually neither: it was an **encoder–decoder**, designed for sequence-to-sequence (seq2seq) tasks like machine translation, where:

- The **input sequence** (source: English sentence) needs to be fully understood.
- The **output sequence** (target: French sentence) needs to be generated one token at a time, conditioned on the input.

These two needs differ. For understanding the input, **bidirectional** attention is best — every word can see every other word. For generating the output, **causal** (left-to-right) attention is required — token $t$ can only attend to tokens $1, \ldots, t-1$, otherwise the model trivially cheats by reading the answer.

Encoder–decoder cleanly separates these: an encoder processes the input bidirectionally, then a decoder generates output autoregressively while attending back to the encoded input via **cross-attention**.

### Concrete example: translating "I love you" to "Je t'aime"

1. Encoder reads "I love you" and produces three context-rich vectors $h_1, h_2, h_3$, each integrating bidirectional context.
2. Decoder starts with the special token `<BOS>`. Its self-attention computes a representation looking only at `<BOS>`. Cross-attention compares this to encoder outputs and learns to focus on "I" → produces "Je."
3. Decoder receives `<BOS> Je`. Self-attention sees both. Cross-attention probably attends to "love" → produces "t'."
4. Continues: `<BOS> Je t'` → "aime." Then `<BOS> Je t' aime` → `<EOS>`. Done.

The cross-attention mechanism is the "alignment" between input and output — analogous to alignment models in classical statistical MT.

### Why not just use a decoder-only model for everything?

You can — and modern LLMs (GPT, LLaMA, Claude) do. Decoder-only models concatenate input and output ("prompt + completion") and treat the whole thing as a single sequence. This is simpler architecturally and trains well at scale. Encoder–decoder models retain advantages:

- **More efficient for fixed-length input + variable output**: the encoder runs once; only the decoder is autoregressive. No re-processing the input every step.
- **Better inductive bias for I/O separation**: T5, the largest pure encoder–decoder LLM, achieves strong performance on classification, translation, and summarization with a single architecture.
- **Cleaner separation for interpretability**: cross-attention weights visualize input-output alignment.

### Connection to ML systems

- **Translation**: Google Translate, DeepL.
- **Summarization**: PEGASUS, BART, T5.
- **Text-to-speech, speech-to-text**: Whisper uses encoder–decoder (audio encoder, text decoder).
- **Image captioning**: CNN/ViT encoder + Transformer decoder.
- **Multimodal generation**: cross-attention is how a text decoder consumes image features (BLIP, Flamingo, LLaVA).
- **Diffusion models**: cross-attention is how text prompts condition image generation in Stable Diffusion.

## 5.2 Conceptual Foundations

### Key terms

- **Encoder**: a stack of Transformer blocks with bidirectional self-attention.
- **Decoder**: a stack of Transformer blocks with causal self-attention plus cross-attention sublayers.
- **Cross-attention**: attention where queries come from one sequence (decoder) and keys/values come from another (encoder).
- **Causal mask**: an upper-triangular mask added to attention logits, setting future positions to $-\infty$.
- **Teacher forcing**: during training, the decoder receives the ground-truth previous tokens as input rather than its own predictions.
- **Autoregressive decoding**: at inference, generating one token at a time, feeding each output back as input.
- **Beam search**: a search procedure that maintains $k$ candidate sequences at each step.
- **KV cache**: at inference, caching the K and V vectors of previously processed tokens to avoid recomputation.

### Encoder block

Standard pre-norm Transformer block with bidirectional self-attention:
1. LN → MultiHeadSelfAttn (no mask) → +residual.
2. LN → FFN → +residual.

Output: contextualized representation per input token.

### Decoder block

Three sublayers per block:
1. **Masked (causal) self-attention**: LN → MultiHeadSelfAttn (causal mask) → +residual.
2. **Cross-attention**: LN → MultiHeadAttn (Q from decoder, K/V from encoder output) → +residual.
3. **FFN**: LN → FFN → +residual.

The encoder is run once over the input. Its output is then attended to by every decoder block via cross-attention.

### Causal masking

For a sequence of length $n$, the causal mask is an $n \times n$ upper-triangular matrix of $-\infty$ above the diagonal:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

Add to attention logits before softmax. After softmax, masked entries become 0, ensuring no information from future positions reaches token $i$.

### Cross-attention specifics

In cross-attention:
- $Q = X_{\text{dec}} W^Q$, where $X_{\text{dec}} \in \mathbb{R}^{m \times d}$ is the decoder's current sequence (length $m$).
- $K = X_{\text{enc}} W^K$, $V = X_{\text{enc}} W^V$, where $X_{\text{enc}} \in \mathbb{R}^{n \times d}$ is the encoder output (length $n$).

The attention matrix $A \in \mathbb{R}^{m \times n}$ aligns each decoder position to encoder positions. No causal mask in cross-attention (the entire encoded input is "past" from the decoder's perspective).

### Underlying assumptions

- **Input and output have meaningful alignment.** Translation, summarization, captioning all satisfy this.
- **The encoder's output captures all needed input info.** Cross-attention re-reads via Q-K matching but cannot recover info the encoder lost.
- **Teacher forcing is reasonable.** During training, conditioning on ground-truth past tokens differs from conditioning on model-generated past tokens at inference — the **exposure bias** problem.

### What breaks

- **Wrong mask in decoder self-attention**: future-token leakage; train loss collapses to zero, but inference fails entirely.
- **Decoder length mismatch**: cross-attention works regardless because $Q$ and $K$ can have different lengths, but length generalization at inference can fail if training data was distributionally narrow.
- **Encoder–decoder dimension mismatch**: $d$ must match between encoder and decoder, or projections are needed.
- **Beam search collapse**: very long beams produce repeated, low-perplexity but degenerate outputs (the "neural text degeneration" problem).

## 5.3 Mathematical Formulation

### Cross-attention

For decoder hidden states $X_{\text{dec}} \in \mathbb{R}^{m \times d}$ and encoder outputs $H_{\text{enc}} \in \mathbb{R}^{n \times d}$:

$$
Q = X_{\text{dec}} W^Q, \quad K = H_{\text{enc}} W^K, \quad V = H_{\text{enc}} W^V
$$

$$
\text{CrossAttn} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

The output has shape $m \times d$ — same as decoder input. Note the asymmetry: even though $m \neq n$, the attention computes correctly because the matrix multiplication $QK^\top$ produces $m \times n$.

### Decoder block (pre-norm)

$$
z_l = x_l + \text{MaskedSelfAttn}(\text{LN}(x_l))
$$

$$
y_l = z_l + \text{CrossAttn}(\text{LN}(z_l), H_{\text{enc}})
$$

$$
x_{l+1} = y_l + \text{FFN}(\text{LN}(y_l))
$$

### Training: teacher forcing

For target sequence $y_1, y_2, \ldots, y_T$, decoder input is $\langle\text{BOS}\rangle, y_1, \ldots, y_{T-1}$ and decoder target is $y_1, \ldots, y_T$. Causal masking ensures position $t$ only sees positions $\leq t$. Loss is the average per-token cross-entropy:

$$
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log p(y_t | y_{<t}, X_{\text{src}})
$$

All $T$ predictions are computed in parallel due to causal masking.

### Inference: autoregressive decoding

At inference, ground-truth $y_t$ is unknown. Generate one token at a time:

1. Run encoder once on input → $H_{\text{enc}}$.
2. Initialize decoder with $\langle\text{BOS}\rangle$.
3. Run decoder, get logits at last position, sample/greedy/beam-select next token $\hat{y}_1$.
4. Append $\hat{y}_1$ to decoder input. Repeat until $\langle\text{EOS}\rangle$ or max length.

Without KV cache, step $t$ recomputes attention over all $1, \ldots, t$ tokens — $O(t^2)$. With KV cache, only the new token's K, V are computed and appended; step cost is $O(t)$. Total decoding cost: $O(T^2)$ vs $O(T^3)$.

### Beam search formalization

Maintain $k$ candidate sequences ("beams") with scores. At each step, expand each beam by every vocabulary token, score the resulting sequences, and keep the top $k$. Score is typically log probability:

$$
s(y_{1:t}) = \sum_{i=1}^t \log p(y_i | y_{<i}, X_{\text{src}})
$$

Often normalized by length to avoid bias toward short sequences:

$$
\bar{s}(y_{1:T}) = \frac{1}{T^\alpha} s(y_{1:T})
$$

with $\alpha \in [0.5, 1]$.

## 5.4 Worked Example

### Translation walkthrough

Translate "I love you" → "Je t'aime."

**Encoder:**
- Tokenize: ["I", "love", "you"]. Embed and add positional encoding.
- Pass through 6 encoder blocks. Output: $H_{\text{enc}} \in \mathbb{R}^{3 \times d}$.

**Decoder, training (teacher forcing):**
- Decoder input: ["<BOS>", "Je", "t'", "aime"]. Decoder target: ["Je", "t'", "aime", "<EOS>"].
- Pass through 6 decoder blocks. At each:
  - Masked self-attention: position 0 sees only itself; position 1 sees 0–1; etc.
  - Cross-attention: each position attends to all 3 encoder outputs.
  - FFN.
- Final output: $4 \times d$ matrix → linear projection to vocab logits → softmax.
- Loss: average cross-entropy of predicted distribution vs target tokens.

All 4 predictions made simultaneously in one forward pass. This is why training scales well.

**Decoder, inference (autoregressive):**
- Input: ["<BOS>"]. Forward pass. Predict "Je."
- Input: ["<BOS>", "Je"]. Forward pass. Predict "t'."
- Continue: predict "aime", then "<EOS>". Stop.

With KV cache, each step processes only the new token; previous K, V are reused.

### Cross-attention pattern visualization

Suppose at decoding step 2 (just predicted "Je", now predicting "t'"), the cross-attention weights from this position to encoder positions are:
- "I" → 0.05
- "love" → 0.85
- "you" → 0.10

The model has correctly identified that the next French token should align with the English verb "love." Visualizing these weights across all decoder positions produces a soft alignment matrix similar to classical word alignment in statistical machine translation.

## 5.5 Relevance to Machine Learning Practice

### Where encoder–decoder is used

- **Translation**: Google Translate (encoder–decoder Transformer for years), DeepL.
- **Summarization**: BART, T5, PEGASUS.
- **Speech-to-text**: Whisper (audio encoder + text decoder).
- **Code completion (some models)**: CodeT5, AlphaCode.
- **Image captioning**: ViT encoder + Transformer decoder; BLIP variants.
- **Diffusion models**: cross-attention from text encoder (CLIP/T5) into the U-Net.

### Where decoder-only dominates

- **General-purpose LLMs**: GPT-3/4, LLaMA, Claude, Mistral, Gemini.
- **Chat models, code assistants**: prompt + completion as a single sequence.
- **In-context learning**: arbitrarily restructured tasks via natural language prompting.

### Encoder-only

- **Embeddings, classification, retrieval**: BERT, RoBERTa, DeBERTa, modern sentence Transformers.

### When to use which

| Task type | Recommended architecture |
|---|---|
| Generate variable-length text from a fixed-length input (translation, summarization) | Encoder–decoder |
| Open-ended generation, chat, code completion, instruction-following | Decoder-only |
| Classification, embedding, retrieval (no generation) | Encoder-only |
| Multimodal: image/audio in, text out | Cross-attention from a modality encoder into a decoder |

### Trade-offs

- **Encoder–decoder**: extra parameters (separate encoder and decoder stacks), but cleaner I/O separation and possibly better for translation.
- **Decoder-only**: simpler, scales better with data, supports few-shot/in-context learning. Wastes compute re-attending to the input every decode step (mitigated by KV cache).
- **Encoder-only**: fastest for non-generation tasks, but cannot generate.

### Production considerations

- **KV cache for decoder self-attention**: standard at inference.
- **Encoder runs once**: cache its output; never recompute during decoding.
- **Beam search vs sampling**: beam search for translation (high precision tasks); sampling for open-ended generation (higher diversity); nucleus / top-p sampling is the default for chat.
- **Length normalization in beam search**: critical to avoid bias toward short outputs.
- **Length penalties at inference**: Google's $((5 + |y|)/(5 + 1))^\alpha$ formulation is a common heuristic.

## 5.6 Common Pitfalls & Misconceptions

- **"Decoder-only models can't do translation."** They can, and do — by concatenating "Translate to French: I love you. → Je t'aime." into one sequence.
- **"Encoder–decoder is obsolete."** Wrong for many use cases. T5, FLAN-T5, Whisper, and Stable Diffusion all use encoder–decoder structures and are state-of-the-art in their domains.
- **"Cross-attention sees the entire input including padding."** Only if you forget to mask padding tokens — which is a common bug. Padding tokens should be masked in both self-attention and cross-attention.
- **"Beam search always improves quality."** Larger beams can degrade quality due to length and repetition issues (Murray & Chiang, 2018), and beams beyond ~5 often plateau.
- **"Teacher forcing produces models that perform identically at inference."** No — at inference the model conditions on its own past predictions, which may diverge from the training distribution (exposure bias). Mitigations: scheduled sampling, RL-based fine-tuning, or simply scaling up.
- **"All cross-attention layers attend the same way."** Different decoder layers learn different alignment patterns — early layers tend to attend broadly, later layers more focused.

## 5.7 Interview Questions: Encoder–Decoder

### Foundational

**Q1. What's the difference between self-attention and cross-attention?**

In self-attention, queries, keys, and values all come from the same sequence — each token attends to other tokens in its own sequence. In cross-attention, queries come from one sequence (e.g., the decoder) and keys/values from another (e.g., the encoder output). Cross-attention is how one sequence "reads" information from another.

**Q2. Why does the decoder need a causal mask but the encoder does not?**

The encoder processes the input, which is fully known — every token can see every other token, leveraging bidirectional context. The decoder generates the output autoregressively; at training time, with teacher forcing, the model has access to the ground-truth target sequence, but it must predict each token using only past tokens. The causal mask prevents the model from "cheating" by looking at the answer.

**Q3. Why is teacher forcing used during training but not at inference?**

Teacher forcing feeds the ground-truth previous tokens to the decoder, allowing parallel computation across the entire target sequence and stable gradients. At inference there is no ground truth — the model must generate one token, feed it back as input, and continue. This creates a train/test distribution mismatch ("exposure bias") because the model is trained conditioning on perfect history but must condition on its own (potentially erroneous) outputs at inference.

### Mathematical

**Q4. The decoder has 3 sublayers per block. List them and write the pre-norm equations.**

The three sublayers are masked (causal) self-attention, cross-attention to encoder outputs, and FFN:

$$
z = x + \text{MaskedSelfAttn}(\text{LN}(x))
$$

$$
w = z + \text{CrossAttn}(\text{LN}(z), H_{\text{enc}})
$$

$$
y = w + \text{FFN}(\text{LN}(w))
$$

The cross-attention is the only sublayer where the decoder reads from the encoder.

**Q5. In cross-attention, the decoder has length $m$ and the encoder output has length $n$. Show the shapes of all matrices and the output.**

- $X_{\text{dec}} \in \mathbb{R}^{m \times d}$, $H_{\text{enc}} \in \mathbb{R}^{n \times d}$.
- $Q = X_{\text{dec}} W^Q \in \mathbb{R}^{m \times d_k}$.
- $K = H_{\text{enc}} W^K \in \mathbb{R}^{n \times d_k}$, $V = H_{\text{enc}} W^V \in \mathbb{R}^{n \times d_v}$.
- $QK^\top \in \mathbb{R}^{m \times n}$.
- $\text{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{m \times n}$.
- Output: $\text{softmax}(\cdot) V \in \mathbb{R}^{m \times d_v}$.

The decoder length $m$ flows through unchanged; encoder length $n$ disappears in the matrix product.

**Q6. Without a KV cache, what is the asymptotic cost of generating $T$ tokens with a decoder of dimension $d$?**

At step $t$, the model recomputes attention over all $t$ tokens. Cost per step: $O(t^2 d + t d^2)$. Summed over $T$ steps: $O(T^3 d / 3 + T^2 d^2)$. With KV cache, K and V from previous tokens are stored; per-step cost becomes $O(t d + d^2)$, summing to $O(T^2 d + T d^2)$.

For $T = 1000$ and $d = 4096$, the cache saves a factor of ~$T/3 \approx 333$ in the $T^3$ term — huge.

### Applied

**Q7. You're training an encoder–decoder for summarization. Validation loss is decreasing but the generated summaries are repetitive ("the the the..."). What's likely wrong?**

Several likely causes:
- **No coverage / repetition penalty**: the model has no incentive to avoid repeating tokens. Add a repetition penalty at decoding time, or use coverage attention during training.
- **Greedy/beam decoding with too narrow a beam** can fall into repetition loops. Try sampling-based decoding (top-p, nucleus).
- **Length normalization in beam search** missing or wrong, biasing toward short repetitive outputs.
- **Training data has repetitive summaries**: data-quality issue.
- **Exposure bias amplified by short-context training**: scheduled sampling or RL fine-tuning may help.

**Q8. Whisper is an encoder–decoder model for speech-to-text. The encoder takes a mel-spectrogram and the decoder produces text. Why this architecture instead of decoder-only?**

Speech and text are fundamentally different modalities with different sequence lengths and structure: mel-spectrograms are dense continuous features, often longer than the corresponding text. Encoder–decoder cleanly separates the modalities — the encoder handles audio with appropriate inductive biases (bidirectional attention over spectrogram frames), the decoder generates text causally with cross-attention to attend back to relevant audio regions. Cross-attention also serves as a soft alignment between audio frames and output tokens, which is useful for word-level timestamps.

### Debugging & failure modes

**Q9. Train loss for an encoder–decoder model is dropping but the generated translations at inference are nonsense. What might be happening?**

Most likely:
- **Decoder is leaking information from future tokens at training time**: causal mask is missing or off-by-one. The model has trivially learned to copy the next token.
- **Train/inference tokenizer mismatch**: different vocabulary or special tokens.
- **Encoder not running at inference**, or running on wrong inputs.
- **KV cache bug**: past states are being shifted incorrectly.
- **EOS token not being predicted**: model never stops generating, producing rambling outputs.

Verify by manually running both train and inference paths on the same example and checking they match (sans the labels).

**Q10. After increasing beam size from 5 to 50, BLEU score *drops*. Why?**

The well-known **beam search curse** (Murray & Chiang, 2018; Stahlberg & Byrne, 2019). Possible causes:
- Larger beams find shorter outputs (since shorter sequences accumulate fewer negative log probs); without length normalization, beam search prefers brevity.
- Larger beams find degenerate, repetitive sequences that have artificially high probability under the model.
- Search exposes the model's inherent preferences for low-perplexity but uninformative outputs (a calibration failure of the model itself).

Fixes: length normalization (Wu et al., 2016), coverage penalties, or simply use a moderate beam size (4–8 is often optimal).

### Probing follow-ups

**Q11. Could you replace cross-attention with something simpler like concatenating encoder output to the decoder input?**

Yes, this is essentially what decoder-only models do — concatenate input and output into one sequence. The trade-off: concatenation forces the decoder to re-process the input through self-attention every block, while cross-attention dedicates a specialized sublayer to encoder reads. Concatenation works well at scale (decoder-only LLMs prove this) but is wasteful for fixed-length inputs that don't change during generation. Cross-attention is more parameter- and compute-efficient when input is fixed.

**Q12. In multimodal models, an image is encoded by a ViT and the decoder is a language model. How is the image fed in?**

Two main approaches:
- **Cross-attention**: the language decoder has additional cross-attention sublayers that attend to ViT outputs (Flamingo, BLIP-2 with Q-Former).
- **Prefix tokens**: ViT outputs are projected and prepended to the text token sequence, then the decoder treats them as ordinary input tokens (LLaVA, InstructBLIP). This requires no architectural change to the LM.

The prefix approach is simpler and benefits from existing pretrained LMs without modification; cross-attention can be more parameter-efficient and isolates modality-specific computation.

**Q13. What happens if you train an encoder–decoder model where the encoder and decoder are tied (share weights)?**

Some models do this for efficiency (e.g., MASS, ALBERT-style sharing). The encoder and decoder must have compatible structures — typically the decoder needs extra parameters for cross-attention. Tying reduces parameters and can act as a regularizer, but limits the model's ability to specialize the encoder for understanding and the decoder for generation. Empirically, untied generally performs better at scale.

**Q14. Why doesn't beam search work well for open-ended generation (e.g., story completion)?**

Beam search optimizes for the highest-probability sequence, but high-probability sequences in open-ended domains are bland, repetitive, or formulaic ("the the the," "and so on and so on"). Diverse, interesting text has individually less likely tokens. For open-ended generation, sampling-based methods (top-k, nucleus/top-p) or temperature-tuned softmax produce better outputs. Beam search remains preferred for tasks with a "right answer" — translation, summarization with reference outputs.

---

# 6. Efficient Attention Variants

## 6.1 Motivation & Intuition

### The quadratic problem

Standard self-attention computes an $n \times n$ matrix of attention scores for sequence length $n$. Time and memory are $O(n^2)$. For $n = 1000$, this is fine; for $n = 100{,}000$ (long documents, codebases, books, video), it becomes prohibitive — $10^{10}$ operations and ~$40$ GB of memory just to store the attention matrix at fp32.

This bottleneck has motivated an entire subfield of "efficient Transformer" research. The goal: maintain quality while reducing complexity to $O(n \log n)$, $O(n \sqrt{n})$, or even $O(n)$.

### Why is attention quadratic, fundamentally?

Because every position must compare against every other. The information-theoretic minimum for "compute a similarity score between all pairs" is $n^2$.

But the question is: *do we actually need all pairs?* If most attention weights are near zero (sparse), we could skip them. If attention can be approximated by a low-rank decomposition, we could factorize. If attention can be expressed as a kernel, we could use random feature approximations. These are the four families of efficient attention:

1. **Sparse**: only compute attention for a subset of (query, key) pairs.
2. **Low-rank / linear**: approximate the attention matrix as $\Phi(Q) \Phi(K)^\top$, reducing to $O(n d)$ via associativity.
3. **Hashing / clustering**: bucket similar queries and keys, compute attention within buckets.
4. **State-space alternatives**: replace attention with linear-time recurrences (Mamba, S4) — not strictly attention variants, but worth knowing.

### Real-world drivers

- **Long-context LLMs**: handling 100k–1M+ tokens (full books, codebases, conversation histories). Claude 3, Gemini 1.5, GPT-4 Turbo all push context lengths.
- **Genomics, video, audio**: naturally long sequences.
- **Edge inference**: phones can't run quadratic attention for long inputs.
- **Cost reduction**: even at moderate lengths, halving attention cost halves serving cost for many production systems.

### A spectrum of approaches

- **Approximate, lossy** (linear attention, Performer): accept some quality loss for asymptotic speedup.
- **Exact, IO-aware** (FlashAttention): same $O(n^2)$ asymptotic complexity but dramatically faster in practice via better memory access patterns.
- **Structured sparsity** (sliding window, BigBird): designed sparsity patterns that work for many tasks.
- **Data-dependent sparsity** (Reformer): hash-based dynamic clustering.

## 6.2 Conceptual Foundations

### Key terms

- **Sparse attention**: attention restricted to a fixed pattern (e.g., local window) or chosen subset.
- **Linear attention**: attention reformulated to scale linearly with $n$ via kernel feature maps or low-rank approximations.
- **Sliding window attention**: each position attends to a fixed window around itself.
- **Global tokens**: a small number of special tokens that attend to / are attended by everyone (mixed pattern).
- **Locality-sensitive hashing (LSH)**: approximate nearest-neighbor scheme used in Reformer to bucket similar Q/K vectors.
- **Random features**: approximation of softmax via random projections (FAVOR+ in Performer).
- **FlashAttention**: an exact attention algorithm that fuses computation to avoid materializing the $n \times n$ matrix in HBM.
- **Chunked attention**: divide sequence into chunks; attend within chunks plus between chunks via summary tokens.

### Sparse attention families

**Sparse Transformer (Child et al., 2019)**: factorize attention into multiple sparse patterns — strided + local. Each head sees a different sparse pattern, and the union covers all positions.

**Longformer (Beltagy et al., 2020)**: combines sliding-window local attention with a few "global" tokens that attend to everyone. Most positions only see a window of $w$ neighbors; complexity $O(n w)$.

**BigBird (Zaheer et al., 2020)**: combines local + global + random attention. Theoretically proven to be a "universal approximator of sequence functions" despite sparsity.

**Mistral / Llama with SWA**: production LLMs use sliding-window attention with a window of e.g. 4096 tokens, plus full attention in some layers.

### Linear attention

**Linear Transformer (Katharopoulos et al., 2020)**: replace softmax with a kernel function $\phi$:

$$
\text{Attn}(Q, K, V) \approx \phi(Q) (\phi(K)^\top V)
$$

Crucially, $\phi(K)^\top V \in \mathbb{R}^{d_k \times d_v}$ is a small matrix; the order of operations matters. Total complexity: $O(n d^2)$ — linear in $n$.

**Performer (Choromanski et al., 2020)**: uses random feature maps to *unbiasedly approximate* softmax attention. The approximation $\phi(\cdot)$ is constructed so that $\mathbb{E}[\phi(q) \cdot \phi(k)] = \exp(q \cdot k / \sqrt{d_k})$.

**Linformer (Wang et al., 2020)**: project K and V to a fixed length $r$ via learned linear projections, reducing $n^2$ to $n r$. Effectively assumes attention is low-rank.

### LSH attention (Reformer)

**Reformer (Kitaev et al., 2020)**: hash queries and keys; queries only attend to keys in the same hash bucket. Uses LSH (random hyperplane projections) to ensure similar Q/K end up in the same bucket with high probability. Complexity: $O(n \log n)$.

Reformer also uses **reversible layers** to drop activations from memory and recompute them in the backward pass — orthogonal to attention efficiency, but stacked together.

### FlashAttention (Dao et al., 2022)

Not asymptotically faster, but dramatically faster in practice. Key insight: attention is **memory-bound** on modern GPUs, not compute-bound. Most time is spent moving the $n \times n$ attention matrix between HBM and SRAM. FlashAttention:

- Tiles the computation so that the attention matrix is computed block-by-block in SRAM, never written to HBM.
- Uses online softmax (processing the matrix in chunks while maintaining running normalization statistics).
- Result: 2–4× wall-clock speedup, 5–20× memory reduction. Exact, not approximate.

FlashAttention 2 and 3 further optimize parallelism, GPU utilization, and FP8 support.

### Underlying assumptions

- **Sparse**: attention patterns are predictable; locality helps for many tasks.
- **Linear/kernel-based**: the attention matrix is approximately low-rank, or softmax can be approximated by random features.
- **LSH**: similar queries genuinely seek similar keys (true for content-based attention).
- **FlashAttention**: GPU memory hierarchy is the bottleneck (true for current hardware).

### What breaks

- **Sliding window** misses long-range dependencies — BigBird's random attention helps, but tasks needing precise long-range retrieval (passkey, multi-hop QA) suffer.
- **Linear attention** loses the sharpness of softmax; on tasks needing precise selection (induction heads, copying), quality drops noticeably.
- **LSH** can have bad luck with hashing — important Q/K pairs may end up in different buckets.
- **FlashAttention** requires custom CUDA kernels; not all hardware supports it.

## 6.3 Mathematical Formulation

### Sparse attention pattern

Define a sparsity mask $M \in \{0, 1\}^{n \times n}$. Attention becomes

$$
A_{ij} = \begin{cases} \text{softmax}_j(q_i \cdot k_j / \sqrt{d_k}) & \text{if } M_{ij} = 1 \\ 0 & \text{otherwise} \end{cases}
$$

with softmax normalized over only the unmasked positions. Memory is $O(\text{nnz}(M))$.

For sliding window of size $w$: $M_{ij} = 1$ iff $|i - j| \leq w/2$. Cost: $O(n w)$.

### Linear attention: associativity trick

Standard: $\text{softmax}(QK^\top) V$. Computing $QK^\top$ first costs $n^2 d$. If we could swap the order — compute $K^\top V$ first ($O(d^2 n)$) then multiply by $Q$ ($O(n d^2)$) — we'd be linear in $n$. But softmax doesn't allow this rearrangement.

Replacing softmax with a feature map:

$$
\text{softmax}(q_i \cdot k_j) \to \phi(q_i) \cdot \phi(k_j)
$$

where $\phi: \mathbb{R}^d \to \mathbb{R}^r$ is a non-negative feature map (positivity ensures the resulting "attention" is non-negative). Then

$$
\text{Attn}_i = \frac{\sum_j (\phi(q_i) \cdot \phi(k_j)) v_j}{\sum_j \phi(q_i) \cdot \phi(k_j)} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
$$

The terms $\sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{r \times d_v}$ and $\sum_j \phi(k_j) \in \mathbb{R}^r$ are computed *once* over all keys, then reused for every query. Total cost: $O(n r d_v)$ — linear in $n$.

Common choices: $\phi(x) = \text{elu}(x) + 1$ (Linear Transformer), or the Performer's random feature map.

### Performer's FAVOR+

Approximate $\exp(q \cdot k / \sqrt{d_k}) \approx \mathbb{E}_\omega[\phi(q) \phi(k)]$ where

$$
\phi(x) = \frac{\exp(-\|x\|^2 / 2)}{\sqrt{m}} [\exp(\omega_1 \cdot x), \ldots, \exp(\omega_m \cdot x)]
$$

with $\omega_i \sim \mathcal{N}(0, I)$. The expectation gives an unbiased estimator of softmax; with $m = \Theta(d \log d)$ random features, the approximation has bounded error.

### LSH attention (Reformer)

For a query $q$, hash it via random hyperplane projection: $h(q) = \arg\max_i (R_i \cdot q)$ where $R$ is a random matrix. Same for keys. Queries only attend to keys with the same hash. With $b$ buckets of expected size $n/b$, cost is $O(n^2 / b)$. Choosing $b = O(n / \log n)$ gives $O(n \log n)$.

Reformer ties Q and K (uses the same projection) so that they hash consistently, and runs multiple hash rounds to reduce collision risk.

### FlashAttention (sketch)

Block sizes $B_q, B_k$ such that one block of $Q, K, V$ fits in SRAM. Algorithm:

1. Initialize output $O = 0$, normalization $\ell = 0$, running max $m = -\infty$ (per row).
2. For each $Q$ block $Q_i$:
   - For each $K, V$ block $K_j, V_j$:
     - Compute $S_{ij} = Q_i K_j^\top / \sqrt{d_k}$ in SRAM.
     - Update running max $m$ and renormalize previous output.
     - Add $\exp(S_{ij} - m) V_j$ to $O$.
3. Final normalize $O = O / \ell$.

Memory: $O(n d)$ instead of $O(n^2)$. Runtime: same FLOPs but with much less HBM traffic, yielding 2–4× speedup.

## 6.4 Worked Examples

### Sliding window cost analysis

For a context length of 32k tokens, $d = 4096$, window $w = 4096$:
- Standard attention: $32{,}000^2 \cdot 4096 \approx 4.2 \times 10^{12}$ FLOPs per head.
- Sliding window: $32{,}000 \cdot 4096 \cdot 4096 \approx 5.4 \times 10^{11}$ FLOPs per head — 8× cheaper.

Memory scales similarly. For $n = 1{,}000{,}000$, the savings become essential — full attention is 250× more expensive than sliding window.

### Linear attention numerical example

Take $n = 4$, $d = 2$, $r = 2$. Suppose queries, keys, values:

$Q = K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ -1 & 0 \end{bmatrix}, V = \begin{bmatrix} 1 \\ 2 \\ 3 \\ 4 \end{bmatrix}$

(Values are 1D for simplicity.)

Use $\phi(x) = \text{ReLU}(x) + 1$ elementwise (non-negative).

Then:
- $\phi(Q) = \begin{bmatrix} 2 & 1 \\ 1 & 2 \\ 2 & 2 \\ 1 & 1 \end{bmatrix}$
- $\phi(K)$ same.

Compute $\sum_j \phi(k_j) v_j^\top$:
- $\phi(k_1) v_1 = (2, 1) \cdot 1 = (2, 1)$
- $\phi(k_2) v_2 = (1, 2) \cdot 2 = (2, 4)$
- $\phi(k_3) v_3 = (2, 2) \cdot 3 = (6, 6)$
- $\phi(k_4) v_4 = (1, 1) \cdot 4 = (4, 4)$
- Sum: $(14, 15)$.

Compute $\sum_j \phi(k_j) = (6, 6)$.

For query 1, $\phi(q_1) = (2, 1)$:
- Numerator: $\phi(q_1)^\top (14, 15) = 28 + 15 = 43$.
- Denominator: $\phi(q_1)^\top (6, 6) = 12 + 6 = 18$.
- Output: $43 / 18 \approx 2.39$.

Notice we never computed the $4 \times 4$ attention matrix. For $n = 1{,}000{,}000$, we'd never compute a million-by-million matrix — savings are vast.

### Reformer's LSH bucketing

Suppose $n = 8$ queries with hashes $[0, 1, 0, 2, 1, 0, 2, 1]$. Bucketing:
- Bucket 0: positions {0, 2, 5}.
- Bucket 1: positions {1, 4, 7}.
- Bucket 2: positions {3, 6}.

Each query attends only within its bucket. Worst-case bucket has 3 elements → 9 pairs per bucket, vs 64 in full attention. With more buckets, fewer pairs.

To reduce variance, run multiple hash functions and union the resulting bucket pairs.

## 6.5 Relevance to Machine Learning Practice

### What's used in production

- **FlashAttention** (and v2, v3): nearly universal in modern Transformer training and inference. Exact, fast, drop-in.
- **Sliding window attention**: Mistral (initial release), some long-context LLaMA variants. Often combined with full attention in some layers.
- **GQA / MQA**: not strictly an attention algorithm change but a structural choice that reduces KV cache; standard in LLaMA-2/3, Gemini, Claude.
- **Paged attention** (vLLM): KV-cache management for serving — not a math change but a systems optimization.

### What's mostly research

- Linear attention, Performer, Reformer: extensively studied, occasionally used in specialized contexts (genomics, very long sequences) but rarely in flagship LLMs.
- BigBird, Longformer: used in document understanding (long-form QA, summarization).

### Why approximations haven't won

- Modern LLM training emphasizes quality, not just speed. Approximate methods often lag in benchmarks.
- FlashAttention removed much of the urgency: it made exact attention fast enough for many use cases.
- Architectures like Mamba (state-space) compete differently — not as attention variants, but as full replacements.

### When to use what

| Context length | Primary technique |
|---|---|
| < 8k | Standard attention with FlashAttention |
| 8k – 32k | FlashAttention, possibly GQA for inference |
| 32k – 128k | Sliding window in some layers, RoPE scaling, GQA |
| 128k – 1M+ | Specialized: hierarchical attention, retrieval-augmentation, Mamba-style alternatives |

### Trade-offs

- **FlashAttention**: pure win on supported hardware. No accuracy trade-off.
- **Sliding window**: misses long-range info; combine with global tokens or full layers.
- **Linear attention**: simpler complexity but quality gap on selective tasks.
- **Sparse patterns**: pattern-specific; works well when the right pattern matches the data.
- **State-space models (Mamba)**: $O(n)$ inference, competitive on long-range, but less mature ecosystem.

## 6.6 Common Pitfalls & Misconceptions

- **"Linear attention is always faster than quadratic."** Only asymptotically. Constants matter — for short sequences, the overhead of feature maps and bookkeeping can make linear attention slower than well-implemented standard attention.
- **"FlashAttention reduces FLOPs."** No — it has the same FLOPs as standard attention. It reduces HBM traffic, which is the actual bottleneck on GPUs.
- **"Sparse attention preserves all important information."** Only the information that fits the chosen sparsity pattern. Tasks requiring precise long-range retrieval (e.g., needle-in-a-haystack) often fail with naive sparse patterns.
- **"Reformer's LSH is exact."** It is approximate — there's always a chance important Q/K pairs land in different buckets. Multiple hash rounds reduce but don't eliminate this risk.
- **"All efficient Transformers are interchangeable."** They embody very different design choices. Sliding window is great for local-bias tasks; LSH for content-based retrieval; linear attention for very long but content-uniform sequences.
- **"Mamba/SSM will replace Transformers."** Premature. SSMs are competitive on some benchmarks and inferior on others (notably copying/in-context retrieval). Hybrid models (Jamba) combine both.

## 6.7 Interview Questions: Efficient Attention

### Foundational

**Q1. Why is standard self-attention $O(n^2)$, and why is this a problem?**

The $QK^\top$ product produces an $n \times n$ matrix, requiring $O(n^2 d)$ FLOPs and $O(n^2)$ memory. For long sequences (10k, 100k, 1M tokens), this is intractable on current hardware. Memory typically becomes the binding constraint before FLOPs.

**Q2. Briefly characterize three families of efficient attention.**

- **Sparse**: only compute attention for a subset of (query, key) pairs (e.g., sliding window, BigBird). Reduces complexity to $O(n w)$ or $O(n \sqrt{n})$.
- **Linear/kernel**: replace softmax with a feature map $\phi$ so that attention factors as $\phi(Q)(\phi(K)^\top V)$, avoiding the $n \times n$ matrix. Complexity $O(n d^2)$.
- **Hashing/clustering** (Reformer): bucket similar Q/K via LSH and only attend within buckets. Complexity $O(n \log n)$.

**Q3. What's the difference between FlashAttention and linear attention?**

FlashAttention is *exact* standard attention with a much faster implementation — it tiles the computation to keep the attention matrix in SRAM and never materializes it in HBM. Same FLOPs and same outputs as standard attention.

Linear attention is *approximate* — it replaces softmax with a kernel approximation, achieving genuine $O(n)$ asymptotic complexity but at some quality cost.

### Mathematical

**Q4. Show that linear attention with non-negative feature map $\phi$ achieves $O(n)$ complexity.**

The attention output for query $i$ is

$$
o_i = \frac{\sum_j (\phi(q_i)^\top \phi(k_j)) v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
$$

The terms $S = \sum_j \phi(k_j) v_j^\top \in \mathbb{R}^{r \times d_v}$ and $z = \sum_j \phi(k_j) \in \mathbb{R}^r$ are computed once in $O(n r d_v)$ and $O(n r)$ respectively. For each query, computing $\phi(q_i)^\top S$ and $\phi(q_i)^\top z$ takes $O(r d_v)$. Total: $O(n r d_v)$ — linear in $n$ for fixed $r, d_v$.

Compare to standard attention's $O(n^2 d)$: linear attention wins when $n > d$ (long sequences relative to model dim).

**Q5. Why does FlashAttention preserve exact softmax despite computing the matrix in chunks?**

It uses an **online softmax** algorithm. For row $i$, maintain running maximum $m$ and running denominator $\ell$. When processing a new chunk of keys with scores $S_j$:
- New max $m' = \max(m, \max_j S_j)$.
- Rescale previous $\ell$: $\ell \cdot e^{m - m'}$.
- Add new contribution: $\ell' = \ell \cdot e^{m - m'} + \sum_j e^{S_j - m'}$.
- Similarly rescale and accumulate the output values.

This is mathematically equivalent to computing softmax over all keys at once, but processes them block by block without storing the full matrix.

**Q6. Why does the Performer's random feature approximation give an unbiased estimator of softmax?**

The Performer constructs $\phi(x) = h(x) \cdot \exp(\omega^\top x)$ for random $\omega \sim \mathcal{N}(0, I)$. Using the identity $\mathbb{E}_\omega[\exp(\omega^\top a) \exp(\omega^\top b)] = \exp((\|a\|^2 + \|b\|^2)/2 + a^\top b)$, with appropriate normalization $h(x) = \exp(-\|x\|^2 / 2)$, one obtains $\mathbb{E}_\omega[\phi(q)^\top \phi(k)] = \exp(q^\top k)$. Averaging over $m$ random samples gives an unbiased low-variance estimator. With $m$ random features, the approximation error is $O(1/\sqrt{m})$.

### Applied

**Q7. You need to fine-tune a 7B model on documents averaging 50k tokens. Memory is the bottleneck. What techniques would you apply?**

- **FlashAttention 2 or 3**: 5–20× memory savings on attention; this alone may make training feasible.
- **Gradient checkpointing**: recompute activations in the backward pass instead of storing them; trades compute for memory.
- **GQA/MQA**: reduces KV cache size at inference (less impact at training but still helpful).
- **Mixed precision (bf16)**: halves activation and weight memory.
- **Sliding window attention** if quality permits: drops memory from $O(n^2)$ to $O(nw)$.
- **DeepSpeed ZeRO or FSDP**: shard optimizer states across GPUs.
- **LoRA / parameter-efficient fine-tuning**: drop trainable parameters by 100–1000×.

Practical order: first try FlashAttention + bf16 + LoRA + FSDP. If still infeasible, add gradient checkpointing or shorten sequences.

**Q8. A user reports that their long-context model performs poorly on the "needle in a haystack" test (retrieving a specific fact from a 100k-token context). Which attention design choices might be implicated?**

- **Sliding window attention** without global tokens: the needle may be outside every position's window.
- **RoPE without proper extrapolation**: positions far from the trained range cause attention scores to drift.
- **Sparse patterns** missing the needle's position.
- **Lost-in-the-middle effect**: even with full attention, models trained on shorter contexts often attend poorly to middle-of-context info.
- **Attention sinks**: heads parking attention on early tokens, missing the needle.

Mitigations: full attention layers interleaved with sparse, position-aware training data with synthetic needles, and RoPE base frequency tuning.

### Debugging & failure modes

**Q9. After switching from standard attention to linear attention, your model trains but in-context learning ability collapses. Why?**

Linear attention loses the *sharpness* of softmax — softmax can produce near-one-hot weights when needed (precise lookup), while linear attention with smooth kernels produces blurry weights. In-context learning relies on "induction heads" — heads that copy specific tokens by precise content matching. Without sharp attention, the copy operation fails. This is a well-documented limitation; hybrid architectures (full attention in some layers) often recover the ability.

**Q10. You implement Reformer's LSH attention and find that performance is worse than standard attention even at long sequences. What could be wrong?**

Several possibilities:
- **Hash collisions**: critical Q/K pairs falling in different buckets. Try more hash rounds or wider buckets.
- **Q/K not tied**: Reformer requires shared Q and K projections so that their hashes align; otherwise queries and their target keys land in different buckets.
- **Sequence too short**: LSH adds overhead; for $n < $ ~4096, standard attention is faster and more accurate.
- **Insufficient training**: Reformer needs more training to learn under approximate attention; you may not have trained long enough.

### Probing follow-ups

**Q11. FlashAttention is asymptotically the same complexity as standard attention. Why is it considered a major advance?**

Because *real-world* performance on GPUs is dominated by memory access patterns, not raw FLOPs. Standard attention writes the $n \times n$ matrix to HBM (slow) and reads it back. FlashAttention computes everything in fast SRAM and writes only the final output. The 2–4× wall-clock speedup and 5–20× memory reduction enable training and inference at scales that were previously infeasible — even though FLOPs are unchanged. It is a textbook case of algorithm-hardware co-design.

**Q12. State-space models (Mamba) achieve $O(n)$ inference. Why aren't they replacing Transformers?**

They are competitive on many benchmarks and superior on some long-context tasks (since they don't have $O(n^2)$ attention). But they underperform on copying-style and in-context retrieval tasks, where Transformers' content-based attention shines. Hybrid architectures (Jamba, Zamba) interleave Mamba and Transformer layers and may be the future. The ecosystem (libraries, tooling, pretrained checkpoints) is also vastly more mature for Transformers, slowing adoption.

**Q13. BigBird claims to be a "universal approximator" of sequence functions despite sparse attention. What does this mean and is it useful?**

The claim (Zaheer et al., 2020) is that BigBird's combination of local + global + random attention can in principle express any function that full attention can, given enough layers. It's a theoretical guarantee similar to those for MLPs being universal approximators. In practice it's reassuring but not predictive — the constants and depth requirements may be very large, and learnability is not the same as expressibility. Empirically, BigBird performs well on long-document tasks but is not a free lunch.

**Q14. How does grouped-query attention (GQA) interact with FlashAttention?**

GQA shares K and V across groups of query heads. FlashAttention can be implemented to exploit this sharing: when reading K/V for one group, all queries in that group can use the same loaded tile. This further reduces HBM traffic and provides additional speedup beyond raw FlashAttention. Modern inference engines (vLLM, TensorRT-LLM) implement GQA-aware FlashAttention kernels.

---

# 7. Pretrained Models

This section covers the four most influential families of pretrained Transformers: **BERT** (encoder-only, masked language modeling), **GPT** (decoder-only, autoregressive language modeling), **T5** (encoder–decoder, text-to-text framing), and **Vision Transformers (ViT)** (encoder over image patches). Each family illustrates a different design choice in a deeply consequential way.

---

## 7.1 BERT family

### 7.1.1 Motivation & Intuition

#### What problem BERT solves

Before BERT (Devlin et al., 2018), most NLP systems were trained from scratch for each task: a sentiment classifier, a named-entity recognizer, a question-answering model — each had its own architecture and was trained on its own (often small) labeled dataset.

BERT introduced **bidirectional pretraining**: train one large Transformer on a massive corpus of unlabeled text using a self-supervised objective (predicting masked words from context), then fine-tune the same model with minimal modification for any downstream task. This shifted the paradigm: large pretraining once, cheap fine-tuning everywhere.

The key insight: if a model can learn rich contextual representations of words from raw text, those representations transfer to almost any task that needs to understand text.

#### Why "bidirectional"?

The previous era's pretraining (ELMo, GPT-1) used left-to-right or shallow bidirectional context. BERT used a **deep bidirectional** Transformer, where every token can attend to all others — both left and right. This better mirrors how humans understand a sentence: by looking at the whole context.

The challenge: how do you train a bidirectional model with language modeling? Standard left-to-right LM lets the model see token $t$ when predicting $t$ via attention. Solution: **mask** some tokens and predict them from the rest.

#### Concrete example

Sentence: "The cat sat on the [MASK]."

The model must predict "mat" (or "rug," "floor," etc.) using *both* the left context ("The cat sat on the") and any right context that exists. With the longer "The cat sat on the [MASK] and slept," the right context "and slept" further constrains the prediction.

#### Connection to ML systems

- **Search and retrieval**: BERT-style models power semantic search, document ranking, and embedding generation (Sentence-BERT, modern embedding models).
- **Classification at scale**: spam filtering, sentiment analysis, topic classification.
- **Named entity recognition, question answering, token tagging**: any task that maps tokens to labels.
- **Foundation for downstream embeddings**: most production "text embedding" APIs descend from BERT-style architectures.

### 7.1.2 Conceptual Foundations

#### Key terms

- **Masked Language Modeling (MLM)**: training objective where ~15% of input tokens are replaced and the model predicts the originals.
- **Next Sentence Prediction (NSP)**: training objective where two sentences are concatenated and the model predicts whether B follows A.
- **[CLS]**: special token prepended to every input; its final hidden state is used for sequence-level classification.
- **[SEP]**: separator token between sentence pairs.
- **WordPiece**: subword tokenization scheme used by BERT (vocab size ~30k).
- **Segment embedding**: an additional embedding indicating which sentence (A or B) each token belongs to.
- **Fine-tuning**: training the entire pretrained model on a downstream task with a task-specific head added.
- **Feature-based use**: extracting fixed embeddings from BERT and feeding them to a separate downstream model (less common now).

#### Architecture

BERT is a **stack of Transformer encoder blocks**:
- BERT-base: 12 layers, $d = 768$, 12 heads, ~110M parameters.
- BERT-large: 24 layers, $d = 1024$, 16 heads, ~340M parameters.

No decoder, no causal masking, no autoregressive generation. Pure bidirectional encoder.

Input format:

```
[CLS] sentence A tokens [SEP] sentence B tokens [SEP]
```

Each token's input embedding is the sum of:
1. Token embedding (WordPiece).
2. Segment embedding (A or B).
3. Position embedding (learned absolute, max length 512).

#### Pretraining objectives

**Masked Language Modeling (MLM):**
- Randomly choose 15% of tokens.
- Of those: 80% are replaced with `[MASK]`, 10% with a random token, 10% kept as the original.
- Model predicts the original token at every chosen position.
- Loss: cross-entropy over the masked positions only.

The 80/10/10 split addresses a train/test mismatch: at fine-tuning time, no `[MASK]` tokens exist, so always masking would create a distribution shift.

**Next Sentence Prediction (NSP):**
- 50% of training pairs: B is the actual next sentence.
- 50%: B is a random sentence from the corpus.
- Model uses the `[CLS]` token's final representation to predict NSP via a linear head.
- Loss: binary cross-entropy.

NSP was meant to teach inter-sentence relationships, but later work (RoBERTa, Liu et al., 2019) found NSP did not help; modern variants drop it.

#### BERT family

- **BERT** (Devlin et al., 2018): the original.
- **RoBERTa** (Liu et al., 2019): more data, longer training, larger batches, no NSP, dynamic masking. Substantially better than BERT at the same architecture.
- **ALBERT** (Lan et al., 2019): parameter-shared layers and factorized embedding for efficiency.
- **DistilBERT** (Sanh et al., 2019): distilled smaller version, ~40% fewer parameters, ~60% faster, ~97% of BERT's performance.
- **DeBERTa** (He et al., 2020): disentangled attention separating content and position; strong on benchmarks.
- **ELECTRA** (Clark et al., 2020): replaces MLM with a discriminator that predicts whether each token was replaced — far more sample-efficient.
- **mBERT, XLM-R**: multilingual variants.

#### Underlying assumptions

- **Mask prediction is a meaningful signal.** Pretraining on this objective produces representations useful for downstream tasks.
- **The same model can be fine-tuned for diverse tasks.** Holds remarkably well for text classification, NER, QA, semantic similarity.
- **15% masking is the right rate.** Empirically near-optimal (some recent work suggests 40% can also work).
- **Static positional embeddings up to 512 tokens.** Most BERT variants are limited to 512 — a constraint of the original choice.

#### What breaks

- **Long documents** (>512 tokens): need chunking or specialized models (Longformer).
- **Generation tasks**: BERT cannot generate text autoregressively; you need a different model.
- **Domain shift**: BERT pretrained on Wikipedia + books may underperform on biomedical or legal text; domain-specific variants (BioBERT, LegalBERT) help.
- **Small downstream datasets**: fine-tuning the entire model can overfit; freeze most layers or use adapters.

### 7.1.3 Mathematical Formulation

#### MLM objective

Let $X = (x_1, \ldots, x_n)$ be the input. Define a mask set $M \subset \{1, \ldots, n\}$ with $|M| \approx 0.15 n$. Construct $\tilde{X}$ by replacing tokens at positions in $M$ according to the 80/10/10 rule. The model produces hidden states $H = \text{BERT}(\tilde{X})$ and logits via a linear head: $z_i = W_{\text{LM}} h_i + b$.

The MLM loss is

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log p(x_i | \tilde{X}) = -\sum_{i \in M} \log \text{softmax}(z_i)_{x_i}
$$

Only masked positions contribute. Unmasked positions are still part of the input but produce no loss.

#### NSP objective

Let $h_{[\text{CLS}]}$ be the final hidden state of the `[CLS]` token. Predict NSP via a linear+softmax head:

$$
p_{\text{NSP}} = \text{softmax}(W_{\text{NSP}} h_{[\text{CLS}]})
$$

Loss: $\mathcal{L}_{\text{NSP}} = -\log p_{\text{NSP}}(y)$ where $y \in \{0, 1\}$.

Total pretraining loss: $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$.

#### Fine-tuning

For sequence classification: take $h_{[\text{CLS}]}$, apply a linear classifier $W_c h_{[\text{CLS}]} + b_c$, train on cross-entropy over the task labels. All BERT parameters are updated.

For token classification (NER): apply a linear classifier per token: $W_c h_i + b_c$.

For span prediction (SQuAD): predict start and end positions of the answer span via two linear heads producing scores per position.

### 7.1.4 Worked Example

Suppose input sentence: "The cat sat on the mat."

After tokenization with WordPiece: `[CLS] the cat sat on the mat . [SEP]` (9 tokens).

**Masking step:**
- Choose 15% of tokens to mask. 9 × 0.15 ≈ 1.35 → mask 1–2 tokens. Suppose we mask "mat":
  - 80% chance → `[MASK]`
  - 10% chance → random token (say "dog")
  - 10% chance → kept as "mat"

Suppose the actual replacement is `[MASK]`. Input becomes: `[CLS] the cat sat on the [MASK] . [SEP]`.

**Forward pass:** Each token gets token + segment + positional embeddings, then 12 layers of bidirectional attention. Final hidden state at the `[MASK]` position is $h_{6}$.

**Loss:** Project $h_6$ to vocab logits via $W_{\text{LM}}$. Compute cross-entropy with the original token ("mat"):

$$
\mathcal{L} = -\log \text{softmax}(W_{\text{LM}} h_6)_{\text{"mat"}}
$$

If the model predicts "mat" with probability 0.4, loss ≈ 0.92. If 0.95, loss ≈ 0.05.

**Fine-tuning for sentiment classification:**

Replace the LM head with a 2-class head ($W_c \in \mathbb{R}^{2 \times 768}$). Take a labeled sentiment dataset; for each example, run BERT, take $h_{[\text{CLS}]}$, classify, compute cross-entropy. Backprop through all of BERT (typically with smaller LR than pretraining, e.g., $2 \times 10^{-5}$). After 2–4 epochs on a few thousand examples, you have a strong sentiment classifier.

### 7.1.5 Relevance to ML Practice

#### Where used

- **Embedding models**: Sentence-BERT and modern OpenAI/Cohere/Voyage embedding APIs are descendants.
- **Search and reranking**: BERT-based bi-encoders and cross-encoders for retrieval ranking.
- **Production NLP**: classification, NER, intent detection in customer service, content moderation.
- **Domain-specific BERTs**: BioBERT (biomedical), SciBERT (scientific), FinBERT (finance), LegalBERT.

#### When to use BERT vs alternatives

- **Use BERT-family for**: classification, embeddings, retrieval, span extraction. When you have labeled data and don't need generation.
- **Use a decoder-only LLM for**: open-ended text generation, instruction following, few-shot learning, anything requiring "in-context learning."
- **Use an encoder–decoder (T5/BART)** for: summarization, translation, structured generation tasks where input and output are both text.

#### Trade-offs

- **Bias–variance**: large pretrained models have low bias on many downstream tasks; fine-tuning small datasets risks overfitting (use early stopping, regularization).
- **Interpretability**: bidirectional contextual embeddings are not human-interpretable; attention attribution methods provide partial visibility.
- **Robustness**: BERT is sensitive to distribution shift (domain, dialect, adversarial perturbations).
- **Compute**: 110M–340M parameters; runs cheaply on a single GPU at inference.

#### Production tips

- For retrieval, use **bi-encoders** (encode query and document separately, use dot product similarity) for speed and **cross-encoders** (encode query and candidate jointly) for reranking the top-K.
- **DistilBERT** or quantized BERT are common for latency-sensitive deployments.
- **Domain-adaptive pretraining** (Gururangan et al., 2020): continue pretraining on in-domain text before fine-tuning. Often produces large gains.

### 7.1.6 Common Pitfalls & Misconceptions

- **"BERT can generate text."** It can fill in masks, but it has no causal mask and was not trained for autoregressive generation. Sampling tokens left-to-right with BERT works poorly.
- **"NSP is essential."** RoBERTa removed it and did better. Modern BERT variants typically drop NSP.
- **"BERT's [CLS] token is a great sentence embedding."** Not without fine-tuning. Raw `[CLS]` embeddings underperform mean-pooling and are far worse than purpose-trained sentence encoders (Sentence-BERT).
- **"You should always fine-tune the entire BERT model."** With small datasets, freezing early layers, using adapters, or LoRA can prevent overfitting.
- **"BERT understands language."** It captures statistical regularities and produces useful representations. Whether that constitutes "understanding" is a philosophical claim, not an empirical one.
- **"BERT and GPT are interchangeable."** They are pretrained for different things — BERT for filling in masks (encoding), GPT for predicting next tokens (generation). Choose based on your downstream need.

### 7.1.7 Interview Questions: BERT family

#### Foundational

**Q1. Explain the MLM objective in one paragraph and state why a non-trivial fraction of "masked" tokens are replaced with the original or a random token.**

MLM randomly selects ~15% of input tokens and replaces them: 80% with `[MASK]`, 10% with a random token, 10% kept as the original. The model predicts the original token at every selected position from bidirectional context. The 80/10/10 split addresses a train/test mismatch: downstream tasks never include `[MASK]` tokens, so always masking would teach the model to ignore non-`[MASK]` positions — keeping or randomly replacing some forces the model to make every position useful and prevents a hard distribution shift at fine-tuning time.

**Q2. Why is BERT bidirectional but GPT is not?**

BERT is trained with MLM, where the model predicts a masked token from both left and right context — bidirectional attention is required and useful. GPT is trained with autoregressive language modeling, predicting each token from prior tokens; allowing future context would let the model "cheat" by reading the answer, so causal masking is required, making it strictly left-to-right.

**Q3. What is the role of the `[CLS]` token?**

`[CLS]` is a special token prepended to every input. Its final hidden state is treated as a summary of the entire input and used as the input to a classification head for sequence-level tasks. During pretraining, it's used for the NSP objective; during fine-tuning, it's the input to task-specific classifiers.

#### Mathematical

**Q4. Write the MLM loss formally.**

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log p(x_i | \tilde{X}) = -\sum_{i \in M} \log \text{softmax}(W_{\text{LM}} h_i)_{x_i}
$$

where $M$ is the set of masked positions, $\tilde{X}$ is the input with masking applied, $h_i$ is BERT's hidden state at position $i$, $W_{\text{LM}}$ is the LM head (often tied to the input embedding matrix), and $x_i$ is the original token.

**Q5. Why is the MLM gradient signal weaker than the autoregressive LM gradient signal, per token of training data?**

In MLM, only the ~15% masked positions contribute to the loss; the other 85% contribute nothing. In autoregressive LM, every token (except the first) is a prediction target, so 100% of positions contribute. This means BERT-style models learn from less signal per pass over the data and typically need more pretraining tokens for equivalent quality. Variants like ELECTRA, where every token contributes via a discriminator, get more efficient signal.

**Q6. Tying the LM head weights to the input embedding matrix is common. Why?**

Two reasons. (1) Parameter efficiency: the embedding matrix is $V \times d$ (e.g., 30k × 768 ≈ 23M parameters); tying saves these parameters in the LM head. (2) Inductive bias: the embedding $e_w$ for word $w$ should be similar to the row of the LM head that produces a high logit for $w$ — both encode "what does $w$ look like in the model's representation space?" Tying has been shown to improve performance and reduce overfitting (Press & Wolf, 2017).

#### Applied

**Q7. You're building a sentiment classifier with 5,000 labeled examples. Should you fine-tune BERT, train BERT from scratch, or use BERT as a feature extractor?**

Fine-tuning BERT is the right choice. Training from scratch would massively overfit on 5,000 examples — pretraining provides the inductive bias the model needs. Using BERT as a frozen feature extractor leaves quality on the table; fine-tuning adapts the representations to your task and typically yields meaningfully better results. With small datasets, use a small learning rate (1e-5 to 5e-5), few epochs (2–4), early stopping on a validation set, and consider data augmentation. If overfitting persists, try LoRA or freezing early layers.

**Q8. Why do practitioners often prefer Sentence-BERT or specialized embedding models over `[CLS]` embeddings for retrieval?**

Raw `[CLS]` embeddings from off-the-shelf BERT are not optimized for similarity — BERT was trained for masked-token prediction, not for sentence matching. The geometry of `[CLS]` representations is not well-aligned with cosine similarity. Sentence-BERT explicitly fine-tunes BERT on sentence-pair similarity tasks (SNLI, STS) using a siamese architecture, producing embeddings whose dot products meaningfully reflect semantic similarity. Modern embedding models (E5, BGE, GTE) extend this with contrastive learning at scale.

#### Debugging & failure modes

**Q9. After fine-tuning BERT on your task, the model achieves 99% training accuracy and 60% validation accuracy. What's wrong and what would you try?**

Classic overfitting. With BERT's 110M+ parameters and likely a small dataset, the model has the capacity to memorize. Mitigations:
- Reduce epochs or use early stopping.
- Increase dropout.
- Use a smaller learning rate.
- Add weight decay.
- Freeze early layers of BERT.
- Use parameter-efficient fine-tuning (LoRA, adapters).
- Augment training data (back-translation, synonym replacement).
- Use a smaller model (DistilBERT).
- Verify that train and validation sets don't overlap (data leakage).

**Q10. You're reproducing BERT pretraining and notice the loss plateaus at a high value. What might be wrong?**

Several common issues:
- **Tokenizer mismatch**: training with a different tokenizer than expected.
- **Data quality**: corrupted or empty examples.
- **Masking implementation bug**: not actually masking tokens, or masking before/after attention computation incorrectly.
- **Loss only over masked tokens**: forgetting to mask out unmasked positions when computing the loss.
- **Learning rate**: too high (divergence) or too low (slow progress).
- **Batch size**: BERT was originally trained with very large batches; small batches need different LR scheduling.
- **Numerical issues**: fp16 underflow without loss scaling.

#### Probing follow-ups

**Q11. ELECTRA replaces MLM with a discriminator that predicts whether each token was replaced. Why is this more sample-efficient?**

In MLM, only ~15% of positions contribute to the loss. In ELECTRA, every token gets a binary label ("real" or "replaced"), so 100% of positions contribute gradient signal. Additionally, the task is harder — distinguishing real from generated tokens requires the discriminator to learn fine-grained linguistic properties. Empirically, ELECTRA achieves comparable performance to BERT with substantially fewer pretraining steps.

**Q12. Why does BERT use WordPiece tokenization rather than word-level or character-level?**

Word-level requires a huge vocabulary (millions of words for multilingual coverage) and handles out-of-vocabulary words poorly. Character-level produces very long sequences and loses lexical structure. Subword tokenization (WordPiece, BPE, SentencePiece) is a sweet spot: a vocabulary of 30–50k subword units covers any text (rare words split into pieces), keeps sequences reasonable in length, and shares parameters across morphologically related words ("running" = "run" + "##ning").

**Q13. Some papers (e.g., RoBERTa) drop NSP entirely. Why might NSP hurt or not help?**

NSP was designed to teach inter-sentence relationships, but the task is too easy: distinguishing two sentences from the same document vs. two from different documents is largely a topical/lexical signal that can be learned trivially. The model may rely on coarse cues (matching keywords) rather than learning nuanced inter-sentence structure. Removing NSP and using more MLM data (longer documents) gives the model richer signal. RoBERTa, ALBERT, and most modern variants drop NSP.

**Q14. BERT is "bidirectional" but GPT-3 can also do classification — sometimes well. What's the difference in approach?**

BERT is bidirectional in attention: every token sees every other directly. For classification, you fine-tune with a head on `[CLS]` and the gradient updates the full model — purpose-built for the task.

GPT-3 does classification via in-context learning: you prompt with examples and ask it to classify. The model is unidirectional but enormously larger. For some tasks GPT-3 matches or exceeds fine-tuned BERT zero-shot; for others, fine-tuned BERT wins. The trade-off is data efficiency vs. scale: BERT needs thousands of labels but is small; GPT-3 needs few or zero labels but is enormous.

---

## 7.2 GPT family

### 7.2.1 Motivation & Intuition

#### What GPT solves

The GPT (Generative Pretrained Transformer) family takes a different approach from BERT: instead of training to fill in masks, train to **predict the next token** given prior tokens. This is the classical autoregressive language modeling objective, applied at unprecedented scale.

The motivation: autoregressive language modeling has long been the canonical task in NLP (it underlies n-gram models, recurrent LMs, and statistical MT). It has the property that *any* text generation, classification, or reasoning task can in principle be cast as continuation: "Translate to French: I love you. Answer:" The model's job is to continue the prompt with a sensible continuation.

#### Why this paradigm scaled so well

Autoregressive LM provides 100% of positions as training signal (every token predicts the next). Combined with massive scale (parameters, data, compute), this leads to a striking emergent property: **in-context learning**. The model can solve tasks it was never explicitly trained on by being shown examples in the prompt.

GPT-2 (1.5B parameters, Radford et al., 2019) hinted at this. GPT-3 (175B, Brown et al., 2020) cemented it. Modern GPT-4, Claude, Gemini, LLaMA, and many others follow the same recipe: decoder-only Transformer + autoregressive LM + scale + RLHF/instruction tuning.

#### Concrete example

Prompt: "Translate English to French: sea otter → loutre de mer; cheese → ____"

Even though the model was never explicitly trained to translate (it's just predicting next tokens of internet text), it learns to complete the pattern: "fromage." This is *few-shot in-context learning* — solving tasks via examples in the prompt, no gradient updates.

#### Connection to ML systems

- **Chatbots, assistants, copilots**: ChatGPT, Claude, Gemini, Copilot, Cursor.
- **Code generation**: Codex, CodeLLaMA, StarCoder, AlphaCode.
- **Foundation models**: serve as the backbone for instruction tuning, RLHF, agent systems, retrieval-augmented generation.
- **Embeddings**: GPT-style models produce strong embeddings (text-embedding-3, etc.).
- **Multimodal**: GPT-4V, Claude 3, Gemini extend the recipe to vision/audio.

### 7.2.2 Conceptual Foundations

#### Key terms

- **Autoregressive language modeling**: predict each token from all preceding tokens.
- **Causal mask**: prevents attention from looking at future tokens during training.
- **Decoder-only**: no encoder, no cross-attention; just stacked decoder blocks (with masked self-attention and FFN, no cross-attention sublayer since there's no encoder).
- **In-context learning (ICL)**: solving a task by showing examples in the prompt without gradient updates.
- **Few-shot, zero-shot, one-shot**: prompting paradigms with N examples.
- **Chain-of-thought (CoT)**: prompting the model to reason step-by-step before answering.
- **Instruction tuning**: fine-tuning on (instruction, response) pairs to make the model follow instructions.
- **RLHF (Reinforcement Learning from Human Feedback)**: aligning the model to human preferences via reward modeling and PPO.
- **System prompt / user prompt**: structured input for chat models.

#### Architecture

GPT models are **decoder-only** Transformers — stacks of Transformer blocks with masked (causal) self-attention and FFN. No cross-attention, since there's no encoder.

Some sizes (illustrative):
- GPT-2 small: 12 layers, $d = 768$, 12 heads, 117M params.
- GPT-2 XL: 48 layers, $d = 1600$, 25 heads, 1.5B params.
- GPT-3: 96 layers, $d = 12288$, 96 heads, 175B params.
- LLaMA-3 70B: 80 layers, $d = 8192$, 64 heads, GQA, RoPE, SwiGLU.

Modern variants commonly use:
- **Pre-norm** with RMSNorm.
- **RoPE** positional encoding.
- **SwiGLU** FFN.
- **GQA / MQA** for inference efficiency.
- **Tied input/output embeddings** (sometimes).

#### Pretraining objective

Autoregressive LM:

$$
\mathcal{L} = -\sum_{t=1}^T \log p(x_t | x_{<t})
$$

The model produces a probability distribution over the vocabulary at every position; loss is the cross-entropy of the next true token. Every position contributes signal, unlike MLM where only ~15% do.

#### Sampling and decoding

At inference, given a prompt, the model produces logits at each step. Decoding strategies:
- **Greedy**: pick the highest-probability token. Deterministic, often boring.
- **Beam search**: maintain $k$ candidates. Used for translation and structured tasks.
- **Temperature sampling**: divide logits by $T$ before softmax. $T \to 0$ → greedy; $T \to \infty$ → uniform.
- **Top-k sampling**: restrict to top-$k$ tokens, sample from their distribution.
- **Nucleus / top-p sampling**: restrict to smallest set of tokens whose cumulative probability exceeds $p$ (e.g., 0.9). Adapts to the distribution shape.
- **Min-p, typical, mirostat, etc.**: further refinements.

#### Post-training paradigm

Modern LLMs aren't just pretrained — they go through several post-training stages:
1. **Pretraining**: massive autoregressive LM on web/text corpus.
2. **Supervised fine-tuning (SFT)**: train on curated (instruction, response) pairs.
3. **Preference learning**: reward modeling from human preferences, then RL (PPO, DPO, KTO) to align.
4. **Safety training**: red-teaming, constitutional AI, additional SFT/RLHF for refusals.

#### Underlying assumptions

- **Next-token prediction subsumes most language tasks.** Empirically, with enough scale, this is largely true.
- **Larger models are better.** Scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022 "Chinchilla") quantify this: doubling parameters and data yields predictable loss reductions.
- **Quality emerges from scale, not architectural cleverness.** The basic Transformer block + autoregressive LM scales remarkably well.
- **In-context learning is a learned capability.** Not designed, but emergent from large-scale autoregressive training.

#### What breaks

- **Long-tail facts**: models hallucinate when they don't actually know.
- **Complex multi-step reasoning**: improves with chain-of-thought but unreliable without it.
- **Up-to-date information**: limited by training cutoff; mitigated with retrieval.
- **Out-of-distribution prompts**: prompt injection, jailbreaks exploit distributional gaps.
- **Numerical precision**: arithmetic, exact lookups, structured outputs without tools.

### 7.2.3 Mathematical Formulation

#### Autoregressive LM loss

For sequence $x = (x_1, \ldots, x_T)$:

$$
p(x) = \prod_{t=1}^T p(x_t | x_1, \ldots, x_{t-1})
$$

$$
\mathcal{L}(\theta) = -\sum_{t=1}^T \log p_\theta(x_t | x_{<t})
$$

The decoder produces logits $z_t = W_{\text{LM}} h_t$ at every position (computed in parallel during training thanks to causal masking), and $p_\theta(x_t | x_{<t}) = \text{softmax}(z_{t-1})_{x_t}$ (the model at position $t-1$ predicts the token at $t$).

#### Causal mask

For sequence length $n$:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

Added to attention logits. After softmax, masked entries are zero, ensuring no future-token leakage.

#### Perplexity

A common evaluation metric:

$$
\text{PPL}(x) = \exp\!\left(\frac{1}{T} \sum_{t=1}^T -\log p_\theta(x_t | x_{<t})\right) = \exp(\bar{\mathcal{L}})
$$

Lower is better. Intuitively, perplexity is the "effective branching factor" — a model with PPL = 20 is on average choosing among 20 plausible next tokens.

#### Scaling laws (Chinchilla, Hoffmann et al., 2022)

For compute-optimal training, parameter count $N$ and training tokens $D$ should scale roughly equally:

$$
N \propto C^{0.5}, \quad D \propto C^{0.5}
$$

where $C$ is total compute. Earlier (Kaplan) scaling laws over-emphasized parameters; Chinchilla showed undertrained large models are wasteful — for fixed compute, a smaller, more-trained model often outperforms.

For a model of $N$ parameters trained on $D$ tokens, the compute cost is approximately $C \approx 6 N D$ FLOPs.

#### KV cache size

For inference, the KV cache holds K and V vectors for each previous token:

$$
\text{KV cache size} = 2 \times L \times n \times h_{\text{kv}} \times d_k \times \text{bytes per value}
$$

where $L$ = layers, $n$ = sequence length, $h_{\text{kv}}$ = number of K/V heads (1 in MQA, $h$ in MHA, intermediate in GQA), $d_k$ = per-head dim. For a 70B model with $L = 80$, $h_{\text{kv}} = 8$ (GQA), $d_k = 128$, and $n = 8000$ tokens at fp16 (2 bytes): $2 \cdot 80 \cdot 8000 \cdot 8 \cdot 128 \cdot 2 \approx 2.6$ GB. For long contexts (128k), KV cache can exceed model weights.

### 7.2.4 Worked Example

#### Training step

Suppose the input batch contains the sequence "The cat sat on the mat .", tokenized into $T = 7$ tokens. With teacher forcing:

- Input to model: `[BOS] The cat sat on the mat`
- Target: `The cat sat on the mat [EOS]`

Forward pass: the model produces 7 distributions over the vocabulary, one per position. Each distribution at position $t$ predicts the token at position $t+1$ (i.e., the $t$-th target token).

Loss is the average cross-entropy:

$$
\mathcal{L} = \frac{1}{7} \sum_{t=1}^7 -\log p(\text{target}_t | x_{1:t})
$$

Suppose the model assigns probability 0.5 to "cat" given "[BOS] The", 0.3 to "sat" given "[BOS] The cat", etc. The loss accumulates $-\log 0.5, -\log 0.3, \ldots$ and averages.

All 7 predictions are computed in parallel due to causal masking — no recurrence needed.

#### Inference (autoregressive)

Prompt: "The cat sat on the"

1. Tokenize → 5 tokens.
2. Forward pass → 5 hidden states, take the last → logits over vocab.
3. Sample (e.g., temperature 0.8, top-p 0.9) → "mat".
4. Append "mat" to input, recompute K/V only for the new token (KV cache makes this $O(1)$ per new token in attention).
5. Forward pass → predict next token.
6. Repeat until `[EOS]` or max length.

With KV cache: each new token requires ~$O(n d^2 + n d \cdot d)$ FLOPs (FFN dominates per step at moderate $n$). Without KV cache: $O(n^2 d)$ per step.

#### In-context learning example

Prompt:
```
Q: What is 2+2?
A: 4

Q: What is 7+5?
A: 12

Q: What is 13+8?
A: 
```

The model "learns" from the two examples to continue the pattern, producing "21." This requires no parameter updates — the model has learned during pretraining to recognize and complete patterns. The mechanism is partially understood: induction heads (Olsson et al., 2022) discover that "when a token X is followed by Y, and X appears again later, predict Y."

### 7.2.5 Relevance to ML Practice

#### Where used

- **General-purpose assistants**: ChatGPT, Claude, Gemini, Copilot.
- **Code assistants**: GitHub Copilot, Cursor, CodeLLaMA.
- **Agent systems**: tool use, browsing, multi-step reasoning.
- **RAG pipelines**: GPT-style models as the generator on top of retrieved context.
- **Domain-specific LLMs**: BloombergGPT (finance), Med-PaLM (medical).
- **Multimodal**: GPT-4V, Gemini, Claude 3 — extend autoregressive prediction to images and audio.

#### When to use a decoder-only LLM

- Open-ended generation (writing, summarization, explanation).
- Chat / instruction-following.
- Few-shot learning where labeled data is scarce.
- Tasks that benefit from reasoning chains.

#### When not

- Classification with abundant labels: a fine-tuned BERT may be cheaper and just as accurate.
- High-throughput retrieval: dedicated embedding models are faster.
- Strict structured outputs without tools: prefer constrained decoding or task-specific models.
- Latency-sensitive inference at scale: careful cost-benefit analysis required.

#### Trade-offs

- **Bias–variance**: low bias (very expressive), but variance can be high (sensitive to prompt phrasing).
- **Interpretability**: poor; we have only partial mechanistic understanding (interpretability research is active).
- **Robustness**: prompt injection, jailbreaks, adversarial inputs are real risks.
- **Compute**: training cost can be tens of millions of dollars; inference cost dominates many production budgets.

#### Production tips

- **KV cache**: essential for any autoregressive serving.
- **GQA / MQA**: shrink KV cache, enabling longer contexts and larger batches.
- **Quantization**: int8 or int4 weights, sometimes activations. Marginal quality loss for big memory savings.
- **Speculative decoding**: a small "draft" model proposes tokens, the large model verifies them in parallel. Speeds up decoding 2–3× without quality loss.
- **Continuous batching** (vLLM): pack requests of varying lengths into batches that grow and shrink dynamically.
- **Prefix caching**: reuse KV cache for common prompt prefixes (system prompts, few-shot examples).
- **Sampling parameters**: tune temperature and top-p per task — low for code/math, higher for creative writing.

### 7.2.6 Common Pitfalls & Misconceptions

- **"GPT models 'understand' what they read."** They learn statistical patterns; whether this constitutes understanding is unsettled. They do *not* maintain explicit world models or beliefs.
- **"Bigger model = better at everything."** Not always — small specialized models can beat huge general ones on narrow tasks. The Chinchilla paper showed many large models were undertrained.
- **"In-context learning is a kind of fine-tuning."** It's not — no weights change. It's pattern completion from the prompt.
- **"Sampling temperature 0 always gives the best output."** Greedy decoding can produce repetitive or low-diversity text and even worse quality on some open-ended tasks. Temperature 0.7–1.0 with top-p 0.9 is a common starting point for chat.
- **"Chain-of-thought always helps."** Helps with multi-step reasoning, hurts on simple tasks (more tokens = more chances for mistakes), and provides no benefit if the model can't reason at all.
- **"RLHF makes models honest."** RLHF makes models *appear* to follow instructions. It can also amplify sycophancy and produce confident-sounding hallucinations.
- **"Long context = good memory."** Large context windows do not mean perfect retrieval from those contexts; "lost in the middle" effects are well-documented.

### 7.2.7 Interview Questions: GPT family

#### Foundational

**Q1. Why is GPT decoder-only and what does that mean architecturally?**

GPT uses only Transformer decoder blocks: stacked masked self-attention + FFN. There is no encoder and no cross-attention sublayer (which would require an encoder to attend to). The "decoder-only" name refers to the use of causal masking — the same property that decoders in encoder–decoder architectures have. This design is well-suited to autoregressive language modeling, the pretraining objective.

**Q2. What is in-context learning, and why is it surprising?**

In-context learning is the ability of a pretrained model to solve a new task by being given examples in the prompt, without any parameter updates. It's surprising because the model was trained only to predict next tokens — it was never explicitly trained to "learn from examples." Yet at sufficient scale, the model exhibits this capability. Mechanistic analysis (Olsson et al., 2022) suggests this emerges from "induction heads" that learn to recognize and complete patterns observed in context.

**Q3. What's the difference between zero-shot, one-shot, and few-shot prompting?**

- **Zero-shot**: the model is given only the task description, no examples. E.g., "Translate to French: hello → ?"
- **One-shot**: the prompt includes one example. E.g., "Translate: dog → chien; hello → ?"
- **Few-shot**: the prompt includes several examples (typically 2–32). E.g., five (English, French) pairs followed by an English word.

#### Mathematical

**Q4. Write the autoregressive LM objective and explain why it is decomposable.**

$$
\mathcal{L} = -\sum_{t=1}^T \log p_\theta(x_t | x_{<t})
$$

By the chain rule of probability, the joint $p(x_1, \ldots, x_T) = \prod_t p(x_t | x_{<t})$, so log-likelihood decomposes into a sum. Causal masking ensures $p_\theta(x_t | x_{<t})$ depends only on positions $1, \ldots, t-1$, so all $T$ predictions can be computed in a single forward pass — making training efficient.

**Q5. State the Chinchilla scaling law conclusion in one sentence and explain its implication.**

For compute-optimal training, model parameters $N$ and training tokens $D$ should grow proportionally (each $\propto C^{0.5}$ for compute $C$). Implication: many large models from the GPT-3 era were undertrained — for the same compute, a smaller model trained on more tokens would outperform. This insight drove the design of LLaMA, Chinchilla itself, and subsequent open-source LLMs.

**Q6. Compute the KV cache size for a 70B model: 80 layers, 64 query heads, 8 KV heads (GQA), per-head dim 128, fp16 (2 bytes), at sequence length 32k.**

$$
\text{KV cache} = 2 \times L \times n \times h_{\text{kv}} \times d_k \times \text{bytes}
$$

$$
= 2 \times 80 \times 32000 \times 8 \times 128 \times 2 = 10{,}485{,}760{,}000 \text{ bytes} \approx 10.5 \text{ GB}
$$

For a single user. With multiple concurrent users, KV cache can quickly exceed model weights (~140 GB at fp16 for the 70B weights themselves, but only the cache scales with user count).

#### Applied

**Q7. You're deploying a GPT-style model that needs to handle 1000 concurrent requests with average prompt length 2k and average completion 500. What inference optimizations would you prioritize?**

- **GQA or MQA**: shrink the per-user KV cache, which is the dominant memory cost at scale.
- **Continuous batching** (vLLM, TensorRT-LLM): pack varying-length requests dynamically.
- **Paged attention**: KV cache is allocated in pages, eliminating fragmentation.
- **Prefix caching**: if requests share a common system prompt, cache its KV.
- **FlashAttention 2/3**: reduce attention compute and memory.
- **Quantization**: int8 weights for less memory and faster GEMMs; possibly int4 if quality permits.
- **Speculative decoding**: a small draft model accelerates the large model's generation 2–3× without quality loss.
- **Pipeline / tensor parallelism**: shard the model across multiple GPUs if it doesn't fit on one.

**Q8. The product team wants the model to "always follow the system prompt and never produce harmful content." Walk through how you'd build this.**

1. **Pretraining**: gives general capability.
2. **Supervised fine-tuning (SFT)**: on (system prompt + user query, helpful safe response) pairs. Curate a dataset that includes diverse harmless examples and explicit refusals for harmful requests.
3. **Reward modeling**: collect human preferences on pairs of model outputs (which is more helpful, which is safer). Train a reward model.
4. **RLHF (PPO or DPO)**: optimize the policy to maximize the reward model while staying close to the SFT model (KL penalty).
5. **Constitutional AI (Anthropic)**: use the model itself to critique and revise outputs based on a set of principles, reducing dependence on human labelers.
6. **Red-teaming**: adversarially probe the model; collect failures and add to training.
7. **Inference safeguards**: input/output classifiers, structured generation, system prompt enforcement.
8. **Monitoring**: log production interactions, sample for review, deploy regression tests.

This is iterative — alignment and capability training continue throughout the model's deployment lifecycle.

#### Debugging & failure modes

**Q9. Your fine-tuned LLM produces correct answers in evaluation but starts repeating itself or going off-topic in production. What might be wrong?**

Several possibilities:
- **Sampling parameters mismatched**: eval uses greedy/low temp, production uses high temp.
- **Distribution shift**: production prompts differ from eval distribution (different topics, lengths, formats).
- **Prompt template mismatch**: production wraps prompts differently than training.
- **Long context degradation**: model trained mostly on short examples but production has long inputs.
- **System prompt drift**: long conversations push the system prompt out of the model's effective attention.
- **KV cache issues** in serving infrastructure.
- **Quantization** introduced subtle quality drops not caught in eval.
- **Repetition penalty** is too low in production.

**Q10. After RLHF, your model becomes more confident but is more often wrong. What's happening?**

This is a known RLHF failure mode — the reward model rewards confident, fluent responses, and the policy learns to *sound* more authoritative even when the underlying facts are uncertain or wrong. Mitigations: (1) train reward models that explicitly reward calibrated uncertainty, (2) include calibration and abstention examples in SFT, (3) constitutional AI revisions targeting overconfidence, (4) measure calibration (e.g., Brier score) as a training metric, (5) reduce KL penalty so the policy doesn't drift too far from the SFT model, (6) introduce verification or self-consistency at inference.

#### Probing follow-ups

**Q11. Could you train a GPT-style model with bidirectional attention?**

Not for autoregressive LM — without causal masking, the model sees the answer at each position and trivially achieves zero training loss but cannot generate. You could train a different objective (MLM) with bidirectional attention — that would be BERT, which can't generate. Some hybrid approaches exist (UniLM uses different masks for different objectives in the same model), but they are essentially multi-task variants.

**Q12. Why does chain-of-thought prompting work?**

Several proposed reasons. (1) The model has more compute per problem — each generated token allows another forward pass that can refine the answer. (2) Pretraining data includes worked solutions that follow step-by-step structure, so generating in that format leverages more of the model's learned distribution. (3) Intermediate steps can decompose a problem into easier subproblems. (4) Mechanistically, the residual stream accumulates intermediate computations that help downstream prediction. CoT helps most on tasks with multi-step structure (math, logic) and least on tasks where the answer is essentially a lookup.

**Q13. What is "speculative decoding" and why does it speed up generation without sacrificing quality?**

A small "draft" model generates several candidate tokens. The large "target" model verifies them in parallel with one forward pass — comparing its own predictions to the drafts. Tokens where the target agrees are accepted; the first disagreement triggers a rejection and resampling using the target model's distribution. The key insight: many tokens are easy and both models agree, so the large model's expensive compute is amortized over multiple tokens per step. Quality is preserved because the verification uses the target model's true distribution. Speedups of 2–3× are common.

**Q14. RLHF uses a KL penalty between the policy and the reference model. Why?**

To prevent the policy from drifting too far from the original (SFT) distribution — without this penalty, the policy could exploit reward model errors and produce outputs that maximize reward but are nonsensical or off-distribution. The KL term anchors the policy to a reasonable baseline:

$$
\mathcal{L} = -\mathbb{E}[r(x, y)] + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

The coefficient $\beta$ trades off reward maximization against staying close to the reference. Too small: reward hacking and degenerate outputs. Too large: little policy improvement. DPO reformulates this entirely, deriving a closed-form optimal policy from preference data and skipping the explicit RL step.

**Q15. Why are decoder-only models more popular than encoder–decoder for general-purpose LLMs?**

Several reasons. (1) Simplicity: one stack, one objective, no need to distinguish encoder/decoder roles. (2) Scaling: empirically, decoder-only scales as well or better than encoder–decoder at large parameter counts. (3) Versatility: prompt + completion as one sequence handles classification, generation, dialogue uniformly. (4) In-context learning emerges naturally from autoregressive training. (5) KV cache works cleanly. Encoder–decoder retains advantages for fixed-input/variable-output tasks (translation, summarization) and is still used in production (T5, Whisper), but for general-purpose chat and instruction following, decoder-only has won.

---

## 7.3 T5: Text-to-Text Transfer Transformer

### 7.3.1 Motivation & Intuition

#### What T5 solves

By 2019 there were many pretrained Transformers — BERT, GPT-2, XLNet, RoBERTa — each with different architectures, objectives, and downstream interfaces. T5 (Raffel et al., 2019, "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer") asked: can we unify everything by treating *every* task as text-to-text?

The framing: regardless of the task — translation, summarization, classification, regression — input is text, output is text. Then one model architecture (encoder–decoder Transformer), one pretraining objective, and one fine-tuning interface suffice for everything.

#### Concrete examples of "text-to-text"

- Translation: input "translate English to German: That is good." → output "Das ist gut."
- Sentiment classification: input "sentiment: This movie was amazing." → output "positive."
- Regression (similarity score 1–5): input "stsb sentence1: ... sentence2: ..." → output "3.4" (as a string).
- Summarization: input "summarize: <article>" → output "<summary>".
- Question answering: input "question: ... context: ..." → output "<answer>".

The model is the same for all tasks; only the input prefix changes.

#### Why this framing matters

- **Unified pipeline**: one tokenizer, one model, one training/inference loop.
- **Multi-task learning by default**: a single T5 fine-tuned on many tasks transfers between them.
- **Exploration of design space**: T5 ran a large empirical study comparing architectures (encoder-only vs decoder-only vs encoder–decoder), objectives (MLM, span corruption, prefix LM), and data — a foundational reference.
- **Foundation for instruction tuning**: FLAN-T5 demonstrated that fine-tuning T5 on many instructions makes it a strong instruction-following model.

#### Connection to ML systems

- **Multi-task NLP services**: when a single backend handles classification, extraction, and generation.
- **Production summarization and translation**: T5 and its variants (BART, mT5) are still used.
- **Instruction-tuned variants**: FLAN-T5 is widely used in research and as a strong baseline.
- **Multimodal extensions**: ViT5, T5X for various modalities.

### 7.3.2 Conceptual Foundations

#### Key terms

- **Span corruption**: pretraining objective where consecutive spans of tokens are masked and the model must produce them.
- **Sentinel tokens**: special tokens (`<extra_id_0>`, `<extra_id_1>`, ...) that mark masked spans.
- **Encoder–decoder**: same as in §5; T5 is one of the largest pure encoder–decoder LLMs.
- **C4 (Colossal Clean Crawled Corpus)**: the cleaned web corpus T5 was pretrained on.
- **Task prefix**: a string prepended to the input indicating the task ("translate English to German:", "summarize:").
- **FLAN (Fine-tuned Language Net)**: a paradigm where a pretrained model is fine-tuned on a large mixture of instruction-formatted tasks.

#### Architecture

Standard encoder–decoder Transformer with a few specifics:
- **Pre-norm with simplified LayerNorm** (no bias, no mean subtraction in the original — similar to RMSNorm).
- **T5 relative position bias**: scalar attention biases based on bucketed relative distance, shared across layers within a stack.
- **No bias in attention or FFN linear layers** in some variants.
- Sizes range from T5-small (60M) to T5-XXL (11B).

#### Pretraining objective: span corruption

Given an input like "The cat sat on the mat":
- Randomly select spans (default: 15% of tokens, mean span length 3).
- Replace each span with a unique sentinel token: `<extra_id_0>`, `<extra_id_1>`, ...
- The encoder receives the corrupted input.
- The decoder outputs the original spans, separated by sentinels.

Example:
- Original: "The cat sat on the mat."
- Encoder input: "The `<extra_id_0>` on the `<extra_id_1>` ."
- Decoder target: "`<extra_id_0>` cat sat `<extra_id_1>` mat `<extra_id_2>`"

The decoder is trained autoregressively to predict the masked spans. This is a denoising autoencoder objective: encoder sees corrupted text, decoder reconstructs.

#### Why span corruption rather than MLM?

- **Encoder–decoder fit**: the decoder's autoregressive output is a natural target for span generation.
- **Multi-token reasoning**: each masked span requires the model to generate a coherent multi-token sequence, exercising more of the model's capacity than single-token MLM.
- **Empirically better**: T5's ablation showed span corruption outperformed several alternatives.

#### Fine-tuning

For any downstream task: format input as text with a task prefix, format output as text. Fine-tune both encoder and decoder end-to-end. No new heads needed — the LM head outputs strings.

For classification, the model learns to output specific strings ("positive," "negative"). At inference, generate a few tokens and parse.

#### Variants

- **mT5**: multilingual T5, trained on 101 languages.
- **ByT5**: byte-level T5, no tokenizer.
- **T5.1.1**: improvements to architecture (gated linear units, no parameter sharing of embeddings).
- **FLAN-T5**: T5 fine-tuned on a massive mixture of instruction tasks; very strong zero-shot and few-shot.
- **UL2** (Tay et al., 2022): unifies multiple denoising objectives (R-denoising, S-denoising, X-denoising) for richer pretraining.

#### Underlying assumptions

- **Every task can be cast as text-to-text.** Holds for most NLP tasks; gets awkward for regression (string outputs) and structured prediction.
- **Encoder–decoder is well-suited to this framing.** Empirically yes; the input-output asymmetry mirrors translation.
- **Span corruption captures most useful patterns.** Pretraining on this objective yields strong general representations.

#### What breaks

- **Pure regression**: predicting a continuous number as a string is awkward and error-prone.
- **Very long outputs**: T5 inherits encoder–decoder's autoregressive cost.
- **Tasks requiring tool use**: T5 has no native interface for calling external tools.
- **Token-level structured outputs without strong format priors**: can violate format mid-output.

### 7.3.3 Mathematical Formulation

#### Span corruption objective

Let $X = (x_1, \ldots, x_n)$. Sample a set of disjoint spans $\{S_1, \ldots, S_k\}$ where $S_i = [a_i, b_i]$. Replace each $S_i$ with a unique sentinel $\langle s_i \rangle$ to get the encoder input $\tilde{X}$. The decoder target is $\langle s_1 \rangle X[S_1] \langle s_2 \rangle X[S_2] \ldots \langle s_k \rangle$.

Loss: standard autoregressive LM loss over the decoder target:

$$
\mathcal{L} = -\sum_{t=1}^{|Y|} \log p(y_t | y_{<t}, \tilde{X})
$$

where $Y$ is the target sequence.

#### T5 relative position bias

T5 doesn't add positional embeddings to input. Instead, attention scores get a learned scalar bias per relative position bucket:

$$
e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{\text{bucket}(j - i)}
$$

Buckets are log-spaced: nearby positions have unique buckets, distant ones share buckets. Biases are head-specific but shared across layers within a stack. This is parameter-efficient and generalizes reasonably to longer sequences than seen in training.

#### Training objective is autoregressive on the decoder side

Even though the encoder sees a "denoising" input, the decoder learns standard next-token prediction over the target. This is why T5's decoder has the same KV-cache mechanics as GPT.

### 7.3.4 Worked Example

Input: "The cat sat on the mat ."

**Span sampling:**
- Choose 2 spans: positions 2–3 ("cat sat") and position 6 ("mat").

**Encoder input** (replacing spans with sentinels):

```
The <extra_id_0> on the <extra_id_1> .
```

**Decoder target** (predict the spans):

```
<extra_id_0> cat sat <extra_id_1> mat <extra_id_2>
```

The trailing `<extra_id_2>` marks the end.

**Training step:**
- Encoder produces hidden states for the 6-token corrupted input.
- Decoder is teacher-forced with `[BOS] <extra_id_0> cat sat <extra_id_1> mat` and predicts `<extra_id_0> cat sat <extra_id_1> mat <extra_id_2>`.
- Decoder cross-attends to encoder outputs at every block, allowing it to read the surrounding context to recover the masked spans.
- Loss: average cross-entropy over the 6 decoder predictions.

**Fine-tuning for sentiment:**

Format input as: `sentiment: The movie was great .`
Format target as: `positive`

Fine-tune end-to-end. At inference, generate ≤ 1 token (or few tokens) and read the result. Accuracy is computed by string match against the gold label.

### 7.3.5 Relevance to ML Practice

#### Where used

- **Production summarization, translation, paraphrasing**: T5 and BART variants remain strong choices.
- **Multilingual NLP**: mT5 is widely used.
- **Instruction-tuned reasoning**: FLAN-T5 is a strong open-source baseline.
- **Embeddings**: T5 encoder-only models can be used as embedders (sentence-T5).
- **Domain-specific T5**: SciT5, BioT5 for science and biology.

#### When to use T5

- Fixed-input → variable-output tasks (translation, summarization, structured generation).
- Multi-task model where a unified text-to-text interface helps.
- When you want a moderately sized open model with strong fine-tuning behavior.

#### When not to

- Open-ended chat / instruction following at the frontier: decoder-only LLMs (Llama, Claude, GPT) are stronger.
- Very long inputs: T5 has its own context-length limits (typically 512 or 1024 in pretraining; long-context variants exist).
- Real-time generation at very low latency: encoder–decoder has higher overhead than decoder-only because the encoder must run before any generation begins.

#### Trade-offs

- **Compute**: encoder–decoder has roughly 2× the parameters of a comparable decoder-only at the same hidden dim, but cheaper inference for long inputs (encoder runs once).
- **Interpretability**: cross-attention provides input-output alignment, helpful for debugging.
- **Format consistency**: text-to-text framing requires the model to produce well-formed outputs (e.g., "positive" not "positiv"). Strong fine-tuning usually handles this.

### 7.3.6 Common Pitfalls & Misconceptions

- **"T5 is obsolete because of GPT-4."** It's still widely used and competitive in its niche (summarization, translation, classification). FLAN-T5 in particular is a strong open-source instruction-following model.
- **"Text-to-text is always the right abstraction."** For pure classification at scale, a small classifier head on a BERT encoder is simpler and faster than generating tokens.
- **"Span corruption and MLM are equivalent."** They're related but produce different distributions over masked content; span corruption forces multi-token coherence and is better suited to encoder–decoder.
- **"T5 uses standard sinusoidal positional encoding."** It uses a relative position bias scheme, not sinusoidal or learned absolute embeddings.
- **"All T5 versions use the same architecture."** T5.1.1, mT5, and UL2 all introduce changes (gated activations, no parameter tying, multiple denoising objectives).

### 7.3.7 Interview Questions: T5

#### Foundational

**Q1. What is the "text-to-text" framing in T5 and why is it useful?**

Every NLP task — translation, classification, regression, QA — is cast as inputting text and producing text. The input includes a task prefix ("translate English to German:") followed by the input data; the output is the answer as a string. This unifies all tasks under one architecture, one objective, one tokenizer, and one training pipeline, simplifying engineering and enabling multi-task learning by default.

**Q2. How does T5's pretraining objective differ from BERT's MLM?**

BERT masks individual tokens (15% per token) and predicts them via the encoder's output at each masked position. T5 masks **spans** of consecutive tokens, replaces each span with a unique sentinel, and the **decoder** generates the masked spans autoregressively, separated by sentinels. Span corruption fits encoder–decoder architectures naturally and exercises multi-token coherence.

**Q3. Why did T5's authors choose encoder–decoder over decoder-only?**

T5 explicitly compared all three architectures (encoder-only, decoder-only, encoder–decoder) in their ablation study. They found encoder–decoder was best for transfer learning across the diverse text-to-text tasks they considered, particularly because input and output have asymmetric roles (input is fully observable, output is generated). For pure language modeling, decoder-only is often preferred.

#### Mathematical

**Q4. Describe the span-corruption process formally.**

Given input tokens $x_1, \ldots, x_n$, sample disjoint spans $S_1, \ldots, S_k$ with mean length 3, covering ~15% of tokens. Replace each span $S_i$ with a unique sentinel token $\langle s_i \rangle$ to form encoder input $\tilde{X}$. The decoder target is the concatenation $\langle s_1 \rangle X[S_1] \langle s_2 \rangle X[S_2] \cdots \langle s_k \rangle$ followed by a final sentinel $\langle s_{k+1} \rangle$ marking termination. Loss is standard autoregressive cross-entropy on the decoder target.

**Q5. Explain T5's relative position bias scheme.**

T5 doesn't add positional embeddings to inputs. Instead, attention logits get an additive scalar bias depending on the relative position $j - i$:

$$
e_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{\text{bucket}(j - i)}
$$

The bucket function maps $j - i$ to one of ~32 log-spaced buckets (close positions get unique buckets, far positions share). The scalars $b$ are learned per bucket per head and shared across all layers within a stack (encoder or decoder). This is parameter-efficient and provides reasonable length generalization.

**Q6. What's the parameter cost of T5 relative biases vs learned absolute position embeddings?**

T5 relative biases: $h \times B$ parameters per stack, where $h$ = number of heads, $B$ = number of buckets (32). For T5-base ($h = 12$, two stacks), that's ~$2 \cdot 12 \cdot 32 = 768$ parameters total — negligible.

Learned absolute embeddings (BERT-style): $n_{\max} \times d$ parameters. For BERT-base ($n_{\max} = 512, d = 768$): ~$393{,}000$ parameters. Three orders of magnitude more.

The relative scheme also generalizes better to sequence lengths beyond training.

#### Applied

**Q7. You need a model that can do summarization, translation, and classification with a single deployed artifact. T5 or a decoder-only LLM?**

Both are viable. **T5** advantages: smaller model can suffice (FLAN-T5-large is ~770M parameters and strong on these tasks); encoder–decoder is well-suited to fixed-input/variable-output; the encoder runs once per input regardless of output length. **Decoder-only LLM** advantages: state-of-the-art quality on diverse tasks; no need for task-specific prefixes (just prompt naturally); easier to extend to new tasks without retraining.

For cost-sensitive production with these specific tasks, a fine-tuned T5 (or FLAN-T5) is usually cheaper. For a wider range of tasks or for reasoning, a decoder-only LLM is more general.

**Q8. You want T5 to do classification with 10 labels. How would you set it up?**

Format input as `classify: <text>` and output as one of the 10 label strings. Choose label strings that are short and tokenize cleanly (avoid weird tokenization edge cases). Fine-tune end-to-end. At inference, generate up to a fixed max length and parse the output string against the label set. To improve robustness, you can use constrained decoding (only allow tokens that begin valid label strings) or compare the per-label likelihood of each label as a continuation, picking the argmax. The latter is more robust to sampling noise.

#### Debugging & failure modes

**Q9. Your fine-tuned T5 sometimes generates labels not in your label set ("possitive" instead of "positive"). What might be wrong and how to fix?**

Causes:
- **Insufficient fine-tuning**: model hasn't fully memorized the label vocabulary.
- **Tokenization quirks**: a label tokenizes oddly, making it less likely.
- **Sampling temperature too high**: at $T > 0$, low-probability typos can sneak in.
- **Distribution shift**: test inputs differ from training.

Fixes:
- Use **constrained decoding** to restrict outputs to valid label tokens.
- Use **likelihood scoring**: compute $p(\text{label} | \text{input})$ for each label and pick the argmax — never generates anything outside the label set.
- Greedy decoding instead of sampling.
- Train for more steps with strong supervision.

**Q10. Span corruption fine-tuning works fine, but task fine-tuning catastrophically forgets the pretraining behavior. What's happening?**

This is **catastrophic forgetting** — the task fine-tuning overwrites the pretrained representations. Mitigations:
- **Mixture training**: mix task data with span-corruption examples during fine-tuning.
- **Lower learning rate** for fine-tuning, especially for early layers.
- **Parameter-efficient fine-tuning** (LoRA, adapters): preserve original weights.
- **Replay-based methods**: occasionally interleave pretraining batches.
- **EWC, regularization to original weights**: penalize movement away from pretrained values.

If your goal is single-task deployment, forgetting may be acceptable; if you want a multi-task model, mixture training is essential.

#### Probing follow-ups

**Q11. Why does T5 use sentinel tokens like `<extra_id_0>` rather than just `[MASK]`?**

Multiple spans can be masked at once. Using distinct sentinels lets the decoder unambiguously refer to which span it's currently producing. The decoder output is structured as `<extra_id_0> [span 0 tokens] <extra_id_1> [span 1 tokens] ... <extra_id_k>`, where each sentinel signals "now producing span $i$." A single `[MASK]` token would create ambiguity.

**Q12. Span corruption with mean span length 3 — what happens if you increase the span length to 50?**

The task becomes much harder — the decoder must generate a 50-token coherent span from limited context. This forces the model to do more "planning" and learn longer-range dependencies, which is the rationale for **UL2**'s X-denoising objective. Empirically, longer spans help downstream generation tasks like summarization but may hurt simpler tasks. UL2 mixes short, medium, and long span objectives for the best of both.

**Q13. T5's encoder runs once per input. Could you cache encoder outputs across inference requests?**

Yes — and this is done in practice. If many requests share the same input (e.g., a fixed prompt with varying continuations, or batched re-ranking of the same document), the encoder output can be cached and reused for the cross-attention K/V in the decoder. This saves substantial encoder compute, similar to KV-cache prefix caching for decoder-only LLMs.

**Q14. How does FLAN-T5 differ from T5?**

FLAN-T5 is T5 fine-tuned on a large mixture of NLP tasks formatted as instructions ("Answer this question:", "Translate to French:", etc.). The fine-tuning teaches the model to follow instructions across diverse tasks, dramatically improving zero-shot and few-shot performance compared to base T5. FLAN-T5 is one of the strongest open-source instruction-following models and is often used as a baseline in research. Architecture and pretraining are the same as T5; only the post-training is different.

---

## 7.4 Vision Transformers (ViT)

### 7.4.1 Motivation & Intuition

#### Why apply Transformers to images?

For decades, **convolutional neural networks (CNNs)** dominated computer vision. They have built-in inductive biases that match natural images:
- **Translation equivariance**: detecting a cat in the top-left should be the same operation as detecting it in the bottom-right.
- **Locality**: nearby pixels are more related than distant pixels.
- **Hierarchical structure**: edges → textures → parts → objects.

CNNs are sample-efficient because they bake in these priors. So why try Transformers?

The answer: at scale, **Transformers learn the right priors from data**, often surpassing CNN performance — and they unlock the same benefits we saw for language: massive pretraining, flexible transfer, multimodal fusion.

The challenge: a standard image (224 × 224 × 3 = 150{,}528 numbers) is far too large to feed token-by-token to a Transformer. ViT (Dosovitskiy et al., 2020, "An Image is Worth 16x16 Words") solved this by:

1. Splitting the image into non-overlapping **patches** (e.g., 16×16).
2. Treating each patch as a "token" via linear projection.
3. Adding positional encodings.
4. Running a standard Transformer encoder.
5. Using a `[CLS]` token (or pooling) for classification.

#### Concrete example

A 224×224 RGB image, with 16×16 patches, gives $14 \times 14 = 196$ patches. Each patch is $16 \times 16 \times 3 = 768$ numbers, projected by a learned linear layer to 768-dim "patch embeddings." Add a `[CLS]` token, add positional embeddings, and feed the 197 tokens into a Transformer. The final `[CLS]` embedding is classified.

It's exactly BERT for images.

#### Why this works (and when it doesn't)

ViT *requires scale*. On ImageNet-1k (1.28M images) trained from scratch, ViT underperforms ResNet because it lacks the locality bias and there isn't enough data to learn it. But pretrained on JFT-300M (300M images) and fine-tuned on ImageNet, ViT outperforms the best CNNs.

The lesson: **inductive bias and data are interchangeable**. CNNs win when data is scarce; Transformers win when data is abundant.

#### Connection to ML systems

- **Image classification, retrieval, segmentation** with pretrained backbones.
- **Multimodal models**: ViT encoders feed text decoders in CLIP, BLIP, Flamingo, GPT-4V, Gemini, Claude.
- **Image generation**: ViT-style encoders are used in diffusion models for text/image conditioning.
- **Medical imaging, satellite imagery, microscopy**: ViT variants now standard.
- **Self-supervised pretraining**: DINO, MAE, SimMIM use ViT as the backbone.

### 7.4.2 Conceptual Foundations

#### Key terms

- **Patch**: a square region of the image (e.g., 16×16 pixels) treated as one token.
- **Patch embedding**: a learned linear projection from a flattened patch to a $d$-dim vector.
- **`[CLS]` token**: a learnable token prepended to the patch sequence, whose final hidden state is used for classification.
- **Positional embedding**: 1D learned absolute embeddings (in original ViT) added to patch embeddings to preserve spatial information.
- **Patch size**: hyperparameter; smaller patches → more tokens → quadratic attention cost grows.
- **Self-supervised pretraining (MAE, DINO)**: train ViT on unlabeled images by masking patches or matching augmented views.

#### Architecture

**Patch embedding:**
- Input image $I \in \mathbb{R}^{H \times W \times 3}$ (e.g., 224×224×3).
- Split into $N = HW/P^2$ patches of size $P \times P$.
- Flatten each patch to a vector of size $3 P^2$.
- Linear projection to $d$-dim: $E \in \mathbb{R}^{3P^2 \times d}$.

Equivalently and more commonly implemented: a single conv layer with kernel size $P$, stride $P$, and $d$ output channels.

**Sequence construction:**
- Prepend a learnable `[CLS]` token: $z_0 = [x_{\text{cls}}; x_1 E; x_2 E; \ldots; x_N E] + E_{\text{pos}}$.
- $z_0 \in \mathbb{R}^{(N+1) \times d}$.

**Transformer encoder:**
- Standard pre-norm Transformer blocks, $L$ of them.
- Multi-head self-attention + FFN, residual + LayerNorm.

**Classification head:**
- Take final `[CLS]` representation $z_L^{[\text{CLS}]}$.
- Linear head to $C$ classes.

**Sizes (original ViT):**
- ViT-B/16: 12 layers, $d = 768$, 12 heads, 86M params, patch size 16.
- ViT-L/16: 24 layers, $d = 1024$, 16 heads, 307M params, patch size 16.
- ViT-H/14: 32 layers, $d = 1280$, 16 heads, 632M params, patch size 14.

#### Pretraining

- **Supervised**: original ViT was supervised on JFT-300M (large internal Google dataset).
- **Self-supervised (MAE, He et al. 2021)**: mask 75% of patches, train an asymmetric encoder–decoder to reconstruct pixels. Decoder is small. Hugely sample-efficient.
- **Self-supervised (DINO)**: distill ViT into itself via teacher-student matching of augmented views. Produces strong features.
- **Contrastive multimodal (CLIP)**: train ViT to match text descriptions; produces a joint vision-language embedding space.

#### Variants

- **DeiT** (Touvron et al., 2020): data-efficient ViT, distillation from a CNN teacher; competitive on ImageNet without massive pretraining.
- **Swin Transformer**: hierarchical with shifted windows; restores some CNN-like locality and supports dense prediction (segmentation, detection) better.
- **MAE**: ViT pretrained via masked image modeling.
- **DINO / DINOv2**: self-supervised ViT producing strong features for detection, segmentation.
- **CLIP / SigLIP**: ViT image encoders aligned with text via contrastive learning.

#### Underlying assumptions

- **Patches are a meaningful tokenization.** Generally yes for natural images; less clear for very high-resolution or sparse data.
- **Position can be encoded with 1D learned embeddings.** Surprisingly works, despite images being 2D; the model learns the 2D structure implicitly. 2D position embeddings help marginally.
- **Enough data is available.** ViT needs >10M images for from-scratch training; pretrained checkpoints are essential for smaller datasets.
- **Patches are self-contained units.** True for moderate resolutions; for fine-grained details (small objects, textures), small patches help but blow up sequence length.

#### What breaks

- **Small datasets, no pretraining**: ViT underperforms CNNs.
- **High resolution**: $n^2$ attention scales poorly; Swin and other hierarchical variants address this.
- **Dense prediction (segmentation, detection)**: vanilla ViT lacks multi-scale features; specialized variants needed.
- **Translation invariance**: ViT must learn it from data; CNNs have it for free.

### 7.4.3 Mathematical Formulation

#### Patch embedding

Image $I \in \mathbb{R}^{H \times W \times C}$. Reshape into $N = (H/P)(W/P)$ patches $x_i \in \mathbb{R}^{P^2 C}$. Project:

$$
z_i = x_i E + e_i^{\text{pos}}, \quad i = 1, \ldots, N
$$

where $E \in \mathbb{R}^{P^2 C \times d}$ is the patch projection and $e_i^{\text{pos}} \in \mathbb{R}^d$ is the position embedding for patch $i$.

Prepend `[CLS]`:

$$
z_0 = [x_{\text{cls}}; z_1; \ldots; z_N] + e^{\text{pos}}
$$

(Position embedding for `[CLS]` is also learned.)

#### Transformer encoder

Standard blocks:

$$
z_l' = \text{MHSA}(\text{LN}(z_{l-1})) + z_{l-1}
$$

$$
z_l = \text{FFN}(\text{LN}(z_l')) + z_l'
$$

for $l = 1, \ldots, L$.

#### Classification

$$
y = \text{LN}(z_L)^{[\text{CLS}]}
$$

$$
\hat{p} = \text{softmax}(W_c y + b_c)
$$

Loss: cross-entropy with the true class.

#### Token count and complexity

For $H = W = 224, P = 16$: $N = 14 \times 14 = 196$, total sequence length $197$. Attention complexity: $O(197^2 \cdot d) \approx 38{,}800 \cdot d$.

For $P = 8$: $N = 28 \times 28 = 784$, complexity $\approx 615{,}000 \cdot d$ — 16× more.

For 1024×1024 image, $P = 16$: $N = 4{,}096$, complexity $\approx 16M \cdot d$ — 400× more. Hierarchical variants like Swin handle this by restricting attention to local windows.

#### MAE objective

Mask 75% of patches randomly. Encoder processes only the visible 25%. A small decoder takes the encoded visible patches plus learnable mask tokens (with position embeddings) at masked positions and reconstructs the masked patches' pixels:

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| \hat{x}_i - x_i \|^2
$$

(Loss is mean squared error on masked patches only, often after per-patch normalization.) The asymmetry is key: encoder runs on 25% of patches, decoder is small — much cheaper than processing all patches.

### 7.4.4 Worked Example

**Image classification setup:**
- Input: 224×224×3 RGB image of a cat.
- Patch size: 16×16. → $14 \times 14 = 196$ patches.
- Each patch flattened: $16 \cdot 16 \cdot 3 = 768$ numbers.
- Linear projection to $d = 768$: each patch becomes a 768-dim vector.
- Prepend `[CLS]` token: 197 tokens total.
- Add learned positional embeddings.
- Run through 12 Transformer blocks (ViT-B).
- Take final `[CLS]` representation.
- Linear classifier to 1000 classes (ImageNet).

If the classifier outputs probability 0.85 for "cat" and the true class is "cat", loss ≈ 0.16.

**Worked sequence reshape:**

Input image of dimensions $224 \times 224 \times 3$ has 150{,}528 values. Reshape into $14 \times 14$ grid of patches, each $16 \times 16 \times 3$. Stack as a tensor of shape $(196, 768)$. Apply patch projection (just a $768 \times 768$ matrix multiplication or, equivalently, a Conv2d with kernel size 16 and stride 16). Prepend `[CLS]` (a learnable $768$-dim vector). Add positional embeddings (197 vectors of size 768).

Total input to first attention layer: $(197, 768)$. Standard Transformer math applies from here.

**MAE pretraining example:**

Take the same image. Randomly mask 75% of patches → 49 visible patches. Encoder processes a $(49, 768)$ sequence (no `[CLS]` in MAE). Output: $(49, 768)$.

Insert learned mask tokens at masked positions, with positional embeddings → $(196, 768)$ (no `[CLS]`). Pass through small decoder (e.g., 8 lightweight Transformer blocks). Output: $(196, 768)$. Linear projection to $(196, 768)$ pixel values for each patch. Loss is MSE against the original masked patches.

After pretraining, discard the decoder. The encoder is now a strong feature extractor for downstream tasks (classification, detection, segmentation).

### 7.4.5 Relevance to ML Practice

#### Where used

- **Image classification**: ViT and variants are state-of-the-art on ImageNet.
- **Multimodal vision-language**: CLIP, SigLIP, BLIP, LLaVA, Flamingo, GPT-4V, Claude, Gemini all use ViT-style image encoders.
- **Self-supervised pretraining**: DINO, MAE, BEiT — all ViT-based.
- **Detection and segmentation**: Swin, DETR, Mask2Former — Transformer-based.
- **Medical imaging**: ViTs adapted for radiology, pathology, etc.
- **Video understanding**: extensions like TimeSformer, ViViT add temporal attention.

#### When to use ViT

- Large-scale image tasks with pretrained models available (always start from a pretrained checkpoint).
- Multimodal pipelines where a Transformer encoder integrates cleanly with text decoders.
- Self-supervised learning from unlabeled images (MAE, DINO).
- Transfer learning across domains.

#### When not to

- Small datasets without strong pretraining: a CNN (ResNet) may be more sample-efficient.
- Very high-resolution dense prediction (large segmentation maps): use hierarchical variants like Swin.
- Latency-critical edge inference: CNNs or efficient hybrid architectures may be faster.

#### Trade-offs

- **Bias–variance**: ViT has weaker inductive bias than CNNs → needs more data, but more flexible at scale.
- **Compute**: $O(N^2 d)$ attention scales poorly with image size; hierarchical variants address this.
- **Sample efficiency**: pretraining + fine-tuning is the default. Self-supervised methods (MAE, DINO) make pretraining cheap.
- **Interpretability**: attention maps over patches give some intuition (though, as in NLP, attention is not a faithful explanation).

#### Production tips

- **Always start from pretrained weights** (DINOv2, MAE, CLIP, ImageNet-22k).
- **Adjust patch size for resolution**: small patches (8×8) for fine detail, large (32×32) for cheap inference.
- **For dense prediction**, use Swin or hybrid CNN-Transformer architectures (ConvNeXt, MaxViT).
- **For multimodal**, CLIP-style ViT encoders are the standard backbone for the image side.
- **Quantization, pruning, distillation** all apply to ViT; DeiT showed strong results from CNN→ViT distillation.

### 7.4.6 Common Pitfalls & Misconceptions

- **"ViT replaces CNNs."** Only at scale and with pretraining. For small datasets without pretraining, CNNs still win.
- **"Patch size doesn't matter much."** It matters a lot — small patches give better quality but quadratic cost; large patches lose detail. 16×16 is a common compromise.
- **"You need 2D positional embeddings for images."** 1D learned embeddings work surprisingly well — the model learns 2D structure implicitly. 2D embeddings give marginal gains.
- **"`[CLS]` token is necessary."** Mean pooling over patch tokens is also commonly used (e.g., in some MAE downstream protocols) and works well.
- **"ViT can't do segmentation/detection."** Plain ViT is suboptimal for dense tasks because it lacks multi-scale features. Hierarchical variants (Swin) and detection-specific designs (DETR, Mask2Former) handle these tasks well.
- **"Self-supervised ViT pretraining is just MLM for images."** It's similar in spirit but different in execution — MAE's asymmetric encoder–decoder design is critical for efficiency, and reconstruction is in pixel space, not over a discrete vocabulary.

### 7.4.7 Interview Questions: ViT

#### Foundational

**Q1. Explain at a high level how a Vision Transformer processes an image.**

The image is split into a grid of non-overlapping patches (e.g., 16×16 pixels). Each patch is flattened and linearly projected to a $d$-dimensional vector — these are the "tokens." A learnable `[CLS]` token is prepended, positional embeddings are added, and the resulting sequence is fed through a stack of standard Transformer encoder blocks. The final `[CLS]` representation is passed through a linear head for classification.

**Q2. Why does ViT need a lot of training data, and how is this addressed in practice?**

ViT lacks the locality and translation-equivariance inductive biases of CNNs, so it must learn these properties from data. With small datasets it underperforms CNNs. The standard fix is large-scale pretraining (supervised on JFT-300M or ImageNet-22k, or self-supervised via MAE / DINO / CLIP) followed by fine-tuning on the smaller target dataset. Pretrained ViT checkpoints are essentially always used in practice.

**Q3. What are some advantages of ViT over CNNs?**

(1) Global receptive field from layer 1 (every patch attends to every other), whereas CNNs need many layers to reach large receptive fields. (2) Flexible architecture — easy to combine with text decoders for multimodal tasks. (3) Scales well with data and parameters. (4) Same architecture as language models — unified backbone for multimodal systems. (5) Strong self-supervised pretraining methods (MAE, DINO).

#### Mathematical

**Q4. For a 224×224 image with patch size 16, how many tokens does ViT process?**

$N = (224/16)^2 = 14^2 = 196$ patches, plus 1 `[CLS]` token = **197 tokens**.

**Q5. Compute the FLOPs of self-attention for a ViT-B/16 model. Use $d = 768$, single layer, single head, 197 tokens.**

Attention has two main matmuls:
- $QK^\top$: $(197 \times 768) \times (768 \times 197) = 197^2 \cdot 768 \approx 30$M FLOPs.
- $AV$: $(197 \times 197) \times (197 \times 768) = 197^2 \cdot 768 \approx 30$M FLOPs.

Plus Q/K/V projections: $3 \cdot 197 \cdot 768 \cdot 768 \approx 350$M FLOPs.

Output projection: $197 \cdot 768 \cdot 768 \approx 116$M FLOPs.

Total per layer per head: ~530M FLOPs (single head). Multiplying by 12 heads is misleading because the per-head dim shrinks to $768/12 = 64$; the projections aggregate to roughly the same total. With 12 layers, total ViT-B attention FLOPs are roughly 6–8 GFLOPs per image — fast on modern hardware.

**Q6. Compare ViT-B/16's parameter count to a comparable ResNet.**

ViT-B/16: ~86M parameters. ResNet-50: ~25M; ResNet-101: ~45M; ResNet-152: ~60M. ViT-B has more parameters but achieves higher accuracy with sufficient pretraining. Per-FLOP, ResNet is more efficient at small scale; ViT becomes competitive at scale.

#### Applied

**Q7. You're building a model for medical image classification with 5,000 labeled X-rays. Should you use ViT or a CNN?**

Almost certainly a pretrained model — training from scratch on 5,000 images won't work for either architecture. Options:
- **Pretrained ViT (ImageNet-22k or DINOv2)** + fine-tuning: typically very strong.
- **Pretrained ResNet/EfficientNet**: also strong, often competitive with ViT at this scale.
- **Domain-specific pretraining**: if available (e.g., RadImageNet pretrained ViT), prefer it.

In practice, fine-tune both, evaluate on a held-out set, and pick the winner. A pretrained ViT on a large medical corpus often wins; a pretrained ImageNet ResNet may win if no medical-pretrained ViT is available.

**Q8. CLIP uses a ViT for the image encoder. Why is ViT a good fit for contrastive multimodal training?**

CLIP requires a global image embedding aligned with a text embedding. ViT's `[CLS]` token (or pooled patch tokens) provides exactly that — a single $d$-dim vector summarizing the image. The architecture also naturally produces interpretable patch-level features useful for downstream tasks (segmentation, detection). Plus, ViT's parameter efficiency at scale matches well with the massive image-text datasets CLIP uses (~400M pairs).

#### Debugging & failure modes

**Q9. You train ViT-B from scratch on ImageNet-1k and achieve 75% accuracy, while a ResNet-50 trained from scratch achieves 76%. Is this expected?**

Yes — this is the classic ViT-vs-CNN result on ImageNet-1k from the original paper. ViT requires substantially more data than ImageNet-1k provides to overcome its weaker inductive biases. With pretraining on ImageNet-22k or JFT-300M followed by fine-tuning on ImageNet-1k, ViT-B reaches 84%+, surpassing ResNet-50. So: from scratch on small data, CNN wins; with pretraining, ViT wins.

**Q10. Your fine-tuned ViT performs poorly on test images that are slightly off-center compared to training. What's the issue?**

ViT lacks translation equivariance — the model has not learned that a centered cat and an off-center cat should be classified the same. This is a known limitation. Mitigations:
- **Heavier data augmentation** (random crops, translations) during training.
- **Use a hybrid CNN-Transformer or Swin Transformer** with stronger spatial inductive biases.
- **More pretraining data** that includes diverse positions.
- **Test-time augmentation**: average predictions over multiple translated crops.

#### Probing follow-ups

**Q11. Why does Masked Autoencoding (MAE) use an asymmetric encoder–decoder design?**

The encoder processes only visible patches (25% of total), so it's much cheaper than processing all patches. The decoder is small (e.g., 8 layers vs the encoder's 12+) and processes all positions (visible + masked). This asymmetry yields: (1) faster pretraining (encoder cost drops 4×); (2) better representations (encoder isn't burdened with reconstruction; decoder handles that); (3) at downstream time, only the encoder is used. The asymmetry is essential to MAE's efficiency.

**Q12. ViT uses 1D positional embeddings even though images are 2D. Why does this work?**

The model can learn 2D structure implicitly: patches at positions $i$ and $i + 14$ (one row down in a 14×14 grid) consistently appear with the same relative offset, and the learned embeddings reflect this. The model figures out the grid structure during training. 2D positional embeddings (separate row and column embeddings) help marginally but the gain is small enough that 1D is the common choice. Some recent variants use 2D RoPE for better extrapolation to different image sizes.

**Q13. Why does Swin Transformer outperform vanilla ViT on segmentation and detection?**

Two main reasons. (1) **Hierarchical features**: Swin progressively merges patches across stages, producing multi-scale features (like a CNN) that are essential for dense prediction. Vanilla ViT only has a single resolution. (2) **Local window attention**: attention is restricted to local windows, with shifted windows between layers to enable cross-window communication. This restores some locality bias and is much more compute-efficient at high resolution. The combination makes Swin a drop-in replacement for CNN backbones in detection and segmentation pipelines.

**Q14. What's special about DINOv2's features compared to supervised ViT features?**

DINOv2 uses self-supervised distillation: a student ViT learns to match a teacher (an EMA of the student) on different augmented views of the same image. This produces features that are: (1) strong without any labels; (2) particularly good for dense prediction tasks (per-patch features cluster meaningfully without supervision); (3) robust across domains. Empirically, DINOv2 features rival or exceed supervised features for many downstream tasks, including ones the supervised model was trained for.

**Q15. ViT is now the standard image encoder in multimodal LLMs (GPT-4V, Claude, Gemini). What practical considerations matter for this integration?**

(1) **Token count**: image patches add many tokens to the LM's context. A 224×224 image at 14×14 patches = 196 tokens; a 1024×1024 image is ~4000+. Image tokens can dominate. (2) **Resolution flexibility**: production systems must handle varying image sizes; tile-based strategies (cut large images into multiple ViT inputs and concatenate features) are common. (3) **Modality alignment**: a projection layer (often an MLP or Q-former) maps ViT outputs to the LM's embedding space. (4) **Training paradigm**: typically the vision encoder is frozen or LoRA-tuned, and only the projector + LM are fine-tuned. (5) **Inference cost**: vision encoder runs once per image, but for high-resolution or multi-image inputs the cost can rival the LM forward pass.

---

# Final Notes

This guide has covered the Transformer architecture from first principles through pretrained model families. A few synthesizing observations:

**The architecture is small.** A handful of components — multi-head attention, residual connections, layer normalization, a feed-forward network, positional encoding — compose into the most influential neural architecture of the past decade. Most "Transformer variants" are minor modifications of these primitives.

**Scale matters more than cleverness.** From BERT to GPT-3 to GPT-4 and Claude, the major leaps have come from scaling data, parameters, and compute. Architecture changes (RoPE, GQA, SwiGLU, FlashAttention) are individually modest but compound to enable scale.

**The same architecture serves all modalities.** Text, vision, audio, code, proteins — Transformers handle them all with appropriate tokenization. The unification suggests something deep about why this architecture works.

**Quadratic attention is the central engineering challenge.** Long contexts, edge inference, and cost reduction all push against the $n^2$ wall. FlashAttention solved much of the practical problem; structural alternatives (Mamba, hierarchical attention) may eventually replace standard attention for very long sequences.

**The interpretability story is incomplete.** We have a much better mechanistic understanding of Transformers in 2025 than 2020 (induction heads, residual stream, circuits), but most of what large models do remains opaque. This is an active research frontier.

For interview preparation: focus on (1) being able to derive the math from scratch, (2) understanding the trade-offs (especially around attention efficiency and architecture choices), (3) being able to debug practical problems (training instability, inference latency, quality regressions), and (4) connecting architectural choices to system-level concerns (KV cache, batching, throughput).

