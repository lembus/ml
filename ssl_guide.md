# Self-Supervised Learning: A Comprehensive Reference Guide

> A standalone, textbook-quality reference covering self-supervised learning, contrastive methods (SimCLR, MoCo, BYOL), pretext tasks, masked modeling (BERT, MAE), and applications. Written for someone with no prior exposure to SSL but comfortable with basic ML and probability.

---

## Table of Contents

1. [Self-Supervised Learning: The Big Picture](#part-i-self-supervised-learning-the-big-picture)
2. [Contrastive Learning: Foundations](#part-ii-contrastive-learning-foundations)
3. [SimCLR: A Simple Framework for Contrastive Learning](#part-iii-simclr)
4. [MoCo: Momentum Contrast](#part-iv-moco)
5. [BYOL: Bootstrap Your Own Latent](#part-v-byol)
6. [Pretext Tasks: Hand-Designed Self-Supervision](#part-vi-pretext-tasks)
7. [Masked Modeling: BERT and MAE](#part-vii-masked-modeling)
8. [Applications and Evaluation of Representations](#part-viii-applications)
9. [Interview Preparation](#part-ix-interview-preparation)

---

# PART I: Self-Supervised Learning — The Big Picture

## 1. Motivation & Intuition

### The labeling bottleneck

Modern deep learning's most embarrassing dependency is the human-labeled dataset. ImageNet, the benchmark that ignited the deep-learning revolution, took years and a small army of crowdworkers to label about 14 million images across 22,000 categories. For specialized domains the cost is far worse: a chest X-ray needs a board-certified radiologist; an audio segment of bird calls requires an ornithologist; a histology slide demands a pathologist. The **labels** are the bottleneck, not the data.

Meanwhile, **unlabeled data is essentially free**. The internet contains hundreds of billions of images, trillions of words of text, and countless hours of video. Every smartphone photographs the world. Every dashcam records driving. Every microphone records audio. The supply of pixels and tokens vastly exceeds the supply of humans willing to annotate them.

### The core idea

**Self-supervised learning (SSL)** asks a deceptively simple question: *Can we design a learning task where the labels come from the data itself, rather than from a human?*

If we can, we can train enormous models on enormous unlabeled corpora — and the representations they learn will hopefully transfer to any downstream task we care about, where we may have only a few labeled examples.

A few intuitive examples make this concrete:

- **Predict the next word.** Given "The cat sat on the ___", the answer "mat" requires no human label — it's just the next token in some scraped web page. A model that does this well has learned grammar, world knowledge, and reasoning patterns. This is GPT.
- **Predict a masked word.** Given "The cat sat on the [MASK]" with bidirectional context, predict the masked word. This is BERT.
- **Predict whether two image crops came from the same image.** Crop two random patches from the same photo, vs from different photos. A model that can tell them apart has learned that the two crops share semantic content. This is the seed of contrastive learning.
- **Predict the rotation applied to an image.** Rotate an image by 0°, 90°, 180°, or 270° and predict the angle. Doing well requires understanding object orientation, which requires understanding what the object is.
- **Reconstruct a masked patch of an image.** Hide 75% of an image's patches and reconstruct them. To do this, the model must learn what objects look like.

In every case: **no human supplied a label**. The "label" was extracted from the data using a deterministic procedure. The model is forced to learn structure in the world to do well.

### Why this matters for ML systems

Self-supervised learning has reshaped almost every applied ML pipeline:

1. **Foundation models.** GPT, BERT, CLIP, DINOv2, MAE, Whisper, SAM — all trained primarily by self-supervision. Downstream tasks ride on these foundations.
2. **Few-shot and zero-shot learning.** A model pretrained on billions of unlabeled examples can often classify new categories with just a handful of labels (or none).
3. **Domain adaptation.** Pretrain on the abundant target-domain unlabeled data, then fine-tune on the small labeled dataset.
4. **Data-efficient training.** The same downstream accuracy can be achieved with 10× or 100× fewer labels by starting from a self-supervised checkpoint.
5. **Multimodal alignment.** SSL extends naturally to image-text pairs (CLIP), video-audio pairs, and other "free" co-occurrences.

The strategic shift is profound: instead of "collect labels, then train", the new pipeline is "**pretrain on unlabeled data → adapt with few labels**".

### The conceptual leap

Self-supervised learning represents a shift away from the supervised paradigm in which a model learns *what we tell it to learn* (`x → y` mappings). Instead, the model learns *general-purpose representations of the input distribution* `p(x)`, with the hope that these representations contain everything needed for many downstream `p(y|x)` tasks.

This is closer to how humans learn. A child does not see a million cat-labeled images. A child sees the world, builds an internal model of objects, surfaces, agents, and language, and *then* learns the word "cat" from a few examples.

---

## 2. Conceptual Foundations

### Key terms

| Term | Definition |
|---|---|
| **Self-supervised learning (SSL)** | A learning paradigm where supervisory signals are derived automatically from the input data, with no human labels. |
| **Pretext task** | A surrogate task whose labels are extracted from the data itself. Solving the pretext task is a means to an end, not the goal. |
| **Downstream task** | The task we actually care about (e.g., classification, detection, segmentation). |
| **Encoder** | The neural network `f_θ` that maps inputs to representations. This is what we want to keep after pretraining. |
| **Representation** (or **embedding**, or **feature**) | The vector `f_θ(x) ∈ ℝ^d` produced by the encoder. |
| **Pretraining** | The phase where `f_θ` is trained on the pretext task using unlabeled data. |
| **Fine-tuning** | The phase where `f_θ` (and possibly a new task head) is trained on the downstream task using labeled data. |
| **Linear probe** | A linear classifier trained on top of frozen `f_θ` features; used to evaluate representation quality. |
| **kNN probe** | A k-nearest-neighbors classifier in feature space; another evaluation of representation quality. |
| **Projection head** | A small MLP `g_φ` placed after the encoder, used during pretraining and discarded for downstream tasks. |
| **Augmentation** | A stochastic transformation `t ~ T` applied to an input to produce different "views" while preserving semantic content. |
| **View** | An augmented version of an input. Two views of the same image are called a **positive pair**. |

### How the components interact

The general SSL pipeline has four stages:

1. **Define a pretext task.** Choose a transformation that produces (input, label) pairs from raw data. Example: BERT's masking transforms `x → (x_masked, x_unmasked)`.
2. **Train the encoder on the pretext task.** Use a large unlabeled corpus. Optimize the pretext loss.
3. **Discard the pretext-task-specific machinery.** Throw away the projection head, the masking decoder, the next-word classifier — anything specific to the pretext task. Keep only the encoder.
4. **Adapt to downstream tasks.** Either freeze the encoder and train a new head (linear probe), or fine-tune end-to-end on labeled downstream data.

The key insight: **the pretext task is a means to an end**. We don't care if the model can predict masked words — we care if learning to do so produces a good representation.

### Three families of SSL methods

Most SSL methods fall into one of three broad categories:

#### (a) Generative / reconstruction methods

The model learns to reconstruct the input (or part of it) from a corrupted version. Examples: autoencoders, BERT (predict masked tokens), MAE (reconstruct masked patches), GPT (predict next token).

- **Pro**: Forces the model to capture detail; natural for generative downstream tasks.
- **Con**: May waste capacity on irrelevant pixel-level detail (high-frequency texture in images).

#### (b) Contrastive methods

The model learns to pull representations of "similar" inputs together and push "dissimilar" inputs apart. Examples: SimCLR, MoCo, CLIP.

- **Pro**: Produces representations that are invariant to nuisance variation; works well for discriminative downstream tasks.
- **Con**: Requires careful design of positive/negative pairs and augmentations; needs many negatives.

#### (c) Self-distillation / non-contrastive methods

The model learns by predicting one view's representation from another, without explicit negatives. Examples: BYOL, SimSiam, DINO, VICReg, Barlow Twins.

- **Pro**: No need for large batches of negatives; simpler implementation.
- **Con**: Risk of "representational collapse" (everything maps to the same vector); requires architectural tricks.

### Underlying assumptions

All SSL methods rely on a small set of assumptions. Knowing them tells you when SSL will fail.

1. **Distributional assumption.** The unlabeled data shares structure with the downstream task. If you pretrain on natural photos and fine-tune on satellite imagery, transfer will be limited.
2. **Invariance assumption.** Whatever the pretext task is invariant to, it teaches the model to be invariant to. If you augment with random color jitter, you teach the model "color doesn't matter" — bad for downstream bird species classification.
3. **Sufficient capacity.** SSL benefits enormously from scale. Small encoders cannot extract general-purpose representations from billions of examples.
4. **Stationarity.** The augmentation distribution `T` and data distribution `p(x)` are assumed roughly stationary during training.
5. **No collapse.** The model's representation space `f_θ(x)` does not collapse to a constant or a low-dimensional subspace. Different methods enforce this differently.

### What breaks when assumptions are violated

- **Distribution shift between pretraining and downstream**: representations may transfer poorly. Solution: domain-specific pretraining (BioBERT, ClinicalBERT, satellite-image SSL).
- **Augmentations destroy task-relevant information**: e.g., color jitter for downstream tasks where color matters. Solution: customize augmentations.
- **Insufficient model capacity or data**: SSL underperforms supervised learning. Solution: scale up, or use supervised pretraining when feasible.
- **Collapse**: representations become constant or low-rank. Detect via feature standard deviation, rank, or alignment metrics. Solution: add negatives, use stop-gradient, predictor networks, or feature decorrelation losses.

---

## 3. Mathematical Formulation (General SSL)

### Notation

- `x ∈ 𝒳`: an input (image, text token sequence, audio waveform).
- `f_θ : 𝒳 → ℝ^d`: the encoder with parameters `θ`. Produces representation `h = f_θ(x)`.
- `g_φ : ℝ^d → ℝ^k`: the projection head with parameters `φ`. Produces `z = g_φ(h)`.
- `t ∈ 𝒯`: a stochastic transformation drawn from an augmentation distribution `T`.
- `ℒ_SSL(θ, φ)`: the self-supervised loss, defined per pretext task.
- `D_u = {x_i}_{i=1}^{N_u}`: large unlabeled dataset.
- `D_l = {(x_j, y_j)}_{j=1}^{N_l}`: small labeled dataset, with `N_l ≪ N_u`.

### General SSL objective

```
θ*, φ* = argmin_{θ, φ}  𝔼_{x ~ p(x), t ~ T}  [ℒ_SSL(θ, φ; t(x))]
```

After pretraining, we discard `g_φ*` and use `h = f_{θ*}(x)` as input to a downstream head `c_ψ`:

```
ψ* = argmin_ψ  𝔼_{(x, y) ~ p(x, y)}  [ℒ_downstream(c_ψ(f_{θ*}(x)), y)]
```

In **linear probing**, `c_ψ` is restricted to a linear function and `θ*` is frozen. In **fine-tuning**, both `θ` and `ψ` are updated.

### Generative (reconstruction) loss

For a corruption process `x̃ = c(x)`, the loss is:

```
ℒ_recon(θ) = 𝔼_x [ ‖x - d_θ(c(x))‖² ]
```

where `d_θ` is a decoder (typically `d_θ = e_φ ∘ f_θ` for some decoder head `e_φ`). For categorical data (tokens), MSE is replaced by cross-entropy on the corrupted positions.

### Contrastive loss (general InfoNCE)

Let `(x_i, x_i^+)` be a positive pair (e.g., two augmentations of the same input) and `{x_i^-_k}_{k=1}^K` be `K` negatives. With `z_i = g_φ(f_θ(x_i))`:

```
ℒ_InfoNCE = - 𝔼  [ log  exp(s(z_i, z_i^+) / τ)  /  ( exp(s(z_i, z_i^+) / τ) + Σ_k exp(s(z_i, z_i^-_k) / τ) ) ]
```

where `s(·, ·)` is a similarity function (typically cosine similarity) and `τ > 0` is a temperature.

### Predictive (non-contrastive) loss

Two views `v_1 = t_1(x)`, `v_2 = t_2(x)`. Online network produces `q_θ(z_θ(v_1))`; target network (often EMA of online) produces `z_ξ(v_2)`:

```
ℒ_BYOL = ‖ q̂_θ(z_θ(v_1)) - ẑ_ξ(v_2) ‖²
```

where `^` denotes ℓ₂-normalization, and `z_ξ` is detached from the gradient (stop-gradient).

### Why this works (mutual information view)

Many SSL losses can be interpreted as lower bounds on the mutual information between two views:

```
I(V_1; V_2) ≥ log K + 𝔼 [ ℒ_InfoNCE ]
```

Maximizing the contrastive objective (or equivalently minimizing the negative loss) increases a lower bound on `I(V_1; V_2)`. Two views of the same image share semantic content; mutual information between them is dominated by that semantic content (since nuisance variation differs across views by design). So maximizing `I(V_1; V_2)` produces representations that capture semantic invariants.

This MI-bound interpretation has caveats — the bound can be loose, and some losses (BYOL, BarlowTwins) don't fit the MI framework cleanly. But it's a useful intuition.

---

## 4. Worked Example: A Tiny SSL Pipeline

Let's walk through a concrete, simplified example to make the abstractions land. We'll do a toy contrastive learning setup on 4 images.

### Setup

Suppose our dataset is 4 images: a cat, a dog, a car, a tree. We'll do contrastive learning with a batch of 2 images at a time, using random horizontal flip and random crop as augmentations.

### Step 1: Sample a batch

Sample images `x_1` (cat) and `x_2` (dog).

### Step 2: Apply two random augmentations to each

```
v_1^(1) = t_1(x_1)  # cat, cropped top-left, flipped
v_1^(2) = t_2(x_1)  # cat, cropped center, no flip
v_2^(1) = t_3(x_2)  # dog, cropped right, flipped
v_2^(2) = t_4(x_2)  # dog, full image, no flip
```

We now have 4 views.

### Step 3: Encode

Pass each through encoder + projection head:

```
z_1^(1) = g_φ(f_θ(v_1^(1)))     # cat view 1, e.g., ℝ^128
z_1^(2) = g_φ(f_θ(v_1^(2)))     # cat view 2
z_2^(1) = g_φ(f_θ(v_2^(1)))     # dog view 1
z_2^(2) = g_φ(f_θ(v_2^(2)))     # dog view 2
```

### Step 4: Define positives and negatives

For view `z_1^(1)`:
- **Positive**: `z_1^(2)` (the other view of the same cat)
- **Negatives**: `z_2^(1)`, `z_2^(2)` (the dog views)

### Step 5: Compute pairwise cosine similarities

Suppose (toy numbers):
```
sim(z_1^(1), z_1^(2)) = 0.85   # high — model already pulls cat views together
sim(z_1^(1), z_2^(1)) = 0.30   # low — different content
sim(z_1^(1), z_2^(2)) = 0.20
```

### Step 6: Compute NT-Xent loss for view `z_1^(1)`

With temperature `τ = 0.1`:

```
numerator   = exp(0.85 / 0.1) = exp(8.5)  ≈ 4914.8
denominator = exp(8.5) + exp(3.0) + exp(2.0) ≈ 4914.8 + 20.09 + 7.39 ≈ 4942.3

loss = -log(4914.8 / 4942.3) = -log(0.9944) ≈ 0.00558
```

The loss is small because the positive similarity (0.85) is much higher than negatives. If the model were untrained, similarities would be roughly equal:

```
sim(z_1^(1), z_1^(2)) ≈ 0.10
sim(z_1^(1), z_2^(1)) ≈ 0.10
sim(z_1^(1), z_2^(2)) ≈ 0.10

numerator = exp(1.0) ≈ 2.718
denominator = 3 · exp(1.0) ≈ 8.155

loss = -log(2.718 / 8.155) = -log(1/3) ≈ 1.0986
```

So the untrained model has loss ≈ log(K+1) = log(3) ≈ 1.0986, where `K` is the number of negatives.

### Step 7: Sum loss over all 4 views, backpropagate

The total loss averages over all 4 view-anchor losses (each view treats its augmented sibling as positive). One gradient step updates `θ` and `φ` to make the representation more invariant to augmentations and more discriminative across instances.

### Step 8: After training, evaluate

Discard `g_φ`. Take the trained encoder `f_θ`. Freeze it. Train a linear classifier on top using the 4 labels (cat, dog, car, tree). Measure accuracy.

This toy example contains every essential ingredient of contrastive SSL: augmentation, encoding, similarity computation, contrastive loss, and downstream evaluation. Real systems just scale this to millions of images and 4096+ batch sizes.

---

## 5. Relevance to ML Practice

### Where SSL is used

1. **Vision foundation models.** SimCLR/MoCo/BYOL/DINO/MAE-pretrained ViTs and ResNets are the starting point for image classification, object detection, segmentation, depth estimation, medical imaging, satellite imagery.
2. **NLP foundation models.** BERT, RoBERTa, GPT, T5 — all trained by self-supervised next-token or masked-token prediction, then adapted to classification, QA, summarization, chat.
3. **Speech.** Wav2vec 2.0, HuBERT, Whisper use SSL on raw audio.
4. **Multimodal.** CLIP and ALIGN learn joint image-text representations from billions of image-caption pairs (a form of "natural" supervision that's still essentially self-supervised).
5. **Robotics and RL.** SSL on state observations (e.g., curiosity, prediction-based exploration).
6. **Tabular and time-series.** SSL approaches like SCARF for tabular data.

### When to use SSL

- **You have abundant unlabeled data and scarce labeled data.** This is the prototypical case.
- **Downstream task is unknown at pretraining time.** SSL gives a general-purpose representation.
- **Multiple downstream tasks.** Pretrain once, fine-tune many times.
- **Domain shift exists between pretraining and deployment.** SSL on in-domain unlabeled data helps bridge the gap.

### When NOT to use SSL

- **You have large labeled data and care only about one task.** End-to-end supervised training may match or beat SSL pretraining (though the cost difference may still favor SSL for ease of iteration).
- **The downstream signal is weakly aligned with semantic content.** Example: estimating image quality scores; the relevant signal may not be what SSL pretraining captures.
- **Compute budget is tiny.** SSL pretraining is expensive.

### Common alternatives

- **Supervised pretraining** on a related labeled dataset (e.g., ImageNet pretraining).
- **Semi-supervised learning** (consistency regularization, pseudo-labeling) when some labels exist.
- **Multi-task learning** when multiple tasks have labels.
- **Transfer learning** from publicly available checkpoints.

### Trade-offs

| Axis | Trade-off |
|---|---|
| **Bias–variance** | SSL biases the encoder towards augmentation-invariant features. If those align with the task, low downstream variance. If not, persistent bias. |
| **Interpretability** | SSL representations are abstract, often harder to interpret than supervised features tied to known classes. |
| **Robustness** | SSL features are often more robust to distribution shift than supervised features (they don't overfit to a specific label set). |
| **Compute cost** | SSL is expensive at pretraining (often more compute than supervised on the same data) but cheap to adapt. |
| **Data efficiency at downstream** | Excellent — often 10× or more reduction in needed labels. |
| **Hyperparameter sensitivity** | High. Augmentation choices, temperature, batch size, LR schedules all matter. |

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "SSL means no labels are needed at all."

False. SSL produces representations. To deploy a classifier, you still need *some* labeled data for the downstream task (linear probe, few-shot adaptation, or fine-tuning). The win is the *amount* of labeled data, not its existence.

### Pitfall 2: "More augmentation is always better."

False. Augmentations define the invariances you teach the model. Aggressive color augmentation makes the model invariant to color — which destroys information for tasks where color matters (medical imaging, fine-grained bird classification, fashion). Choose augmentations to preserve task-relevant information and discard nuisance variation.

### Pitfall 3: "The pretext loss going to zero means we're done."

Misleading. The pretext loss can saturate at a low value while the representation is still poor. Always evaluate via linear probe, kNN, or downstream fine-tuning.

### Pitfall 4: "Use the projection head output as the feature."

A common error. The projection head `g_φ` is designed to be discarded. SimCLR's authors showed that the encoder output `h` is significantly better than the projection output `z` for downstream tasks. Why: the projection head absorbs augmentation-specific information, leaving the encoder representation more general.

### Pitfall 5: Representational collapse.

In non-contrastive methods (BYOL, SimSiam), the model can trivially minimize the loss by mapping every input to the same vector — `f_θ(x) = c` for all `x` — because then `q(f(v_1)) = f(v_2)` is satisfied. Detect by monitoring the standard deviation of representations across the batch; if it crashes, the model has collapsed. Mitigations: predictor network, stop-gradient, EMA target, normalization.

### Pitfall 6: Batch size matters more than expected.

Contrastive methods like SimCLR rely on within-batch negatives. With batch size 256, you have only 510 negatives — too few. SimCLR famously needs batch size 4096+ for best results. MoCo solves this with a queue.

### Pitfall 7: Confusing self-supervised with unsupervised learning.

Strictly, **unsupervised learning** is the broader family (clustering, density estimation, dimensionality reduction). **Self-supervised learning** is a specific subset that uses *supervised* training with *automatically generated* labels. SSL borrows the engineering of supervised learning (loss functions, gradients, classifiers) but eliminates the human in the labeling loop.

### Pitfall 8: Treating SSL evaluation as a single number.

Linear probe accuracy on ImageNet is one signal. But SSL models can be very different along other axes: out-of-distribution robustness, calibration, fine-tuning vs probing, transfer to dense prediction (segmentation), few-shot learning. A model that wins linear probing may lose under fine-tuning, or transfer poorly to detection.

### Pitfall 9: "BatchNorm causes BYOL to work."

A widely-cited initial hypothesis (Tian et al., "Understanding self-supervised learning dynamics without contrastive pairs") was that BatchNorm provides implicit negatives by mixing features across the batch. Subsequent work (Richemond et al., "BYOL works *even* without batch statistics") showed that GroupNorm + careful initialization also works. The real reasons BYOL avoids collapse are more subtle: predictor network + EMA target + asymmetry. Don't oversimplify.

### Pitfall 10: Ignoring the role of architecture.

A ResNet-50 SSL feature is not interchangeable with a ViT-B SSL feature. ViTs benefit from masked modeling (MAE) more than from contrastive methods; CNNs benefit more from contrastive learning. Match the SSL method to the architecture.


---

# PART II: Contrastive Learning — Foundations

## 1. Motivation & Intuition

### The contrastive principle

The contrastive principle is intuitive: **representations of "similar things" should be close; representations of "dissimilar things" should be far apart**. This is essentially the same idea behind metric learning and Siamese networks, recast for the SSL setting where "similar" and "dissimilar" come from data automatically.

A useful mental image: imagine projecting all your images onto the surface of a unit sphere. Contrastive learning organizes the sphere so that:

- Augmented views of the same image cluster tightly together (a "blob").
- Different images occupy different regions, spread out across the sphere.

After training, semantically similar images naturally end up nearby (because they share more invariants), even though we never told the model what semantic similarity is.

### Why this works without labels

The crucial trick: we synthesize "similar" pairs using augmentation. Two random crops of the same image, with different color jitter and blur, are by construction similar — they came from the same scene. Two random crops of *different* images are presumed dissimilar.

This presumption isn't perfect. Two photos of cats may be more similar to each other than two crops of the same urban scene. But in expectation across millions of images, the augmentation-generated positive pairs are reliably more similar than random pairs, and the contrastive loss exploits this signal.

### A real-world analogy

Suppose you want to train a librarian to organize books without telling her any categories. You hand her a book, then hand her another copy of the same book with a different cover (positive). You hand her a third book chosen at random (negative). You ask her to put copies of the same book on the same shelf, and different books on different shelves.

If you do this enough times, with enough books, she will inevitably start grouping similar books together — physics texts near math texts, novels near memoirs, cookbooks together — because books that share content tend to be more similar to each other than to random books. Without ever defining "physics" or "fiction", she has built a categorical structure.

That's contrastive learning.

### Connection to ML systems

Contrastive learning powers:

- **Image embeddings** for visual search, recommendation, deduplication.
- **Face recognition** (FaceNet's triplet loss is the spiritual ancestor).
- **CLIP**: contrastive learning between images and their captions, enabling zero-shot classification.
- **Sentence embeddings** (SimCSE, Contriever).
- **Recommendation systems** (item embeddings learned by contrasting co-occurring items).
- **Audio representation** (wav2vec uses InfoNCE between predicted future and actual future audio).

---

## 2. Conceptual Foundations

### Anatomy of a contrastive learning system

A contrastive system has the following components:

1. **Augmentation module** `T`: a stochastic pipeline that produces views of an input.
2. **Encoder** `f_θ`: maps views to representations.
3. **Projection head** `g_φ`: maps representations to a smaller, normalized space where the contrastive loss is applied.
4. **Similarity function** `s(·, ·)`: measures how close two embeddings are. Typically cosine similarity.
5. **Loss function**: typically a softmax-style loss (InfoNCE / NT-Xent) that contrasts a positive against negatives.

### Positive and negative pairs

- **Positive pair** `(x, x⁺)`: two views of the same instance. Their representations should be close.
- **Negative pair** `(x, x⁻)`: views of different instances. Their representations should be far.

How negatives are obtained distinguishes the methods:
- **SimCLR**: in-batch negatives — every other sample in the same batch.
- **MoCo**: queue of past samples — a memory bank.
- **CLIP**: in-batch negatives across the image-text matrix.
- **BYOL/SimSiam/DINO**: no explicit negatives at all.

### The role of the projection head

The projection head was a small architectural choice in SimCLR that turned out to matter enormously. The encoder `f_θ` produces a feature `h ∈ ℝ^{d_h}` (e.g., d_h = 2048 for ResNet-50). The projection head `g_φ` (typically a 2-3 layer MLP) reduces this to `z ∈ ℝ^{d_z}` (e.g., d_z = 128) and ℓ₂-normalizes it.

**Why use it?** Empirically, applying the contrastive loss directly to `h` produces worse representations than applying it to `g_φ(h)`. The reason: the contrastive loss requires augmentation invariance, which forces it to discard augmentation-specific information. The projection head absorbs this discarding, leaving `h` richer. After pretraining, we use `h` (not `z`) for downstream tasks.

A useful analogy: the projection head is like the Olympic athlete's training shoes — useful for the workout, but you wear different shoes for the race.

### Similarity functions

By far the most common choice is **cosine similarity**:

```
s(u, v) = (u · v) / (‖u‖ ‖v‖)
```

After ℓ₂-normalization, the cosine similarity equals the dot product, which is computationally cheap. Cosine similarity is preferred over raw dot product because it's bounded in [-1, 1] and removes magnitude as a confound.

Less common alternatives:
- Euclidean distance (turns into "smaller is more similar")
- Bilinear similarity `u^T W v` for some learned `W` (used in CPC).

### Temperature

In the contrastive loss, we divide similarity by a temperature `τ`:

```
score_k = s(z_anchor, z_k) / τ
```

The temperature controls the "sharpness" of the softmax over candidates:

- **Low τ** (e.g., 0.05): very peaky distribution — focuses gradient on the hardest negatives. Risk: overconfidence, instability.
- **High τ** (e.g., 1.0): nearly uniform — all negatives matter equally. Risk: weak gradient, slow learning.

SimCLR uses τ = 0.5 for cosine similarity (which is bounded in [-1, 1]). MoCo uses τ = 0.07 (also for cosine similarity). The "right" temperature depends on the similarity range and the desired hardness focus.

Wang & Liu (2021, "Understanding the Behaviour of Contrastive Loss") showed that the temperature controls a tradeoff between **uniformity** (spreading representations across the sphere) and **tolerance** (allowing similar instances to cluster). Very low τ punishes any negative similarity harshly, even for semantically similar items, which can hurt downstream performance.

### Underlying assumptions

1. **Augmentation invariance preserves semantics.** Two augmented views of the same image share the semantic content. If your augmentations destroy semantics (e.g., extreme color jitter on bird species), the assumption breaks.
2. **Negatives are mostly true negatives.** When we treat other batch elements as negatives, we assume they're semantically different from the anchor. In a batch of 4096 ImageNet images, this is mostly true — but you might pull two cat images apart, slightly hurting the model. This is the **false negative problem**.
3. **Batch is representative of the data distribution.** With small batches, the negatives are not diverse enough.
4. **The encoder has capacity to learn invariances.** Large enough to encode augmentation invariances without losing other content.

---

## 3. Mathematical Formulation (Contrastive)

### InfoNCE loss

The InfoNCE loss (van den Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018) is the workhorse contrastive loss. For an anchor `z`, a positive `z⁺`, and `K` negatives `{z_k⁻}`:

```
ℒ_InfoNCE = - log [ exp(s(z, z⁺) / τ)  /  ( exp(s(z, z⁺) / τ) + Σ_{k=1}^K exp(s(z, z_k⁻) / τ) ) ]
```

This is exactly a `(K+1)`-way softmax classification problem: given the anchor, classify which of `K+1` candidates is the positive. The cross-entropy loss is the negative log-likelihood of choosing the correct (positive) candidate.

### Connection to mutual information

InfoNCE is a lower bound on mutual information between paired views. Specifically (Poole et al., 2019):

```
I(X; Y) ≥ log K - ℒ_InfoNCE
```

where `K` is the number of negatives. So **maximizing InfoNCE → maximizing a lower bound on mutual information**. This connects contrastive learning to information theory:

- Two views of the same image share semantic content; nuisance variation (lighting, crop, color) differs.
- Maximizing `I(V_1; V_2)` forces the encoder to retain what's shared (semantics) and discard what differs (nuisance).
- The bound is tight only when `K → ∞`. With finite K, there's a gap, which is one reason large batch sizes help.

### Alignment and uniformity

Wang & Isola (2020, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere") showed that contrastive learning optimizes two objectives simultaneously:

1. **Alignment** — positive pairs map to nearby points:
```
ℒ_align = 𝔼_{(x, x⁺)} [ ‖f(x) - f(x⁺)‖² ]
```

2. **Uniformity** — features are spread uniformly on the unit sphere:
```
ℒ_uniform = log 𝔼_{x, y ~ p_data} [ exp(-t · ‖f(x) - f(y)‖²) ]
```

InfoNCE in the limit of infinite negatives equals (a weighted sum of) these two terms. Alignment ensures invariance; uniformity ensures the representation uses the full capacity of the sphere (no collapse). This decomposition is useful for diagnosing failures: if alignment is poor, augmentations are too aggressive; if uniformity is poor, the model is collapsing.

### Hard negative mining (implicit and explicit)

The InfoNCE gradient, when you work it out, naturally weights the hardest negatives most heavily. This is because the softmax denominator is dominated by the highest-scoring negatives.

Specifically, the gradient of the loss with respect to a negative similarity `s_k = s(z, z_k⁻)` is:

```
∂ℒ / ∂s_k = (1/τ) · p_k
```

where `p_k = exp(s_k / τ) / Σ_j exp(s_j / τ)` is the softmax probability assigned to negative `k`. Hard negatives (high `s_k`) get larger `p_k`, hence larger gradient. This implicit hard negative mining is one reason contrastive learning works without explicit mining strategies — but it's also why temperature matters: too low and you focus exclusively on the hardest negatives, often false negatives.

### The full SimCLR objective for a batch

Given a batch of `N` original images, generate `2N` augmented views. Index them `i = 1, ..., 2N` such that views `2k-1` and `2k` are augmentations of the same image (a positive pair).

For each view `i`, the loss is:

```
ℓ_i = - log [ exp(s(z_i, z_{j(i)}) / τ)  /  Σ_{k ≠ i} exp(s(z_i, z_k) / τ) ]
```

where `j(i)` is the index of `i`'s positive partner.

The total loss is averaged:

```
ℒ = (1 / 2N) · Σ_{i=1}^{2N} ℓ_i
```

Note that the denominator excludes only the anchor itself — even the positive's similarity is in the denominator. This differs from some formulations and matters for gradients.

### Derivation: gradient of InfoNCE

Let `s_+ = s(z, z⁺) / τ` and `s_k = s(z, z_k⁻) / τ`. The loss:

```
ℒ = - s_+ + log [ exp(s_+) + Σ_k exp(s_k) ]
```

Let `Z = exp(s_+) + Σ_k exp(s_k)`.

Gradient with respect to `s_+`:

```
∂ℒ / ∂s_+ = -1 + exp(s_+) / Z = -1 + p_+ = -(1 - p_+)
```

where `p_+ = exp(s_+) / Z` is the probability the model assigns to the positive. The gradient pushes `s_+` higher whenever `p_+ < 1`, i.e., almost always.

Gradient with respect to a negative `s_k`:

```
∂ℒ / ∂s_k = exp(s_k) / Z = p_k
```

So the gradient pushes `s_k` lower with strength proportional to `p_k`. The hardest negatives (largest `p_k`) get the strongest push. This is the implicit hard negative mining.

Backprop through `s_k = (z · z_k⁻) / τ` (with both vectors ℓ₂-normalized) gives:

```
∂s_k / ∂z = z_k⁻ / τ
```

So the encoder is updated to move `z` *away from* the highest-scoring negatives' embeddings.

### Map back to the conceptual story

- The positive term in the numerator drives **alignment** — pulls `z` towards `z⁺`.
- The negatives in the denominator drive **uniformity** — pushes `z` away from `z_k⁻`, with the hardest negatives getting the most push.
- Temperature `τ` controls how concentrated the gradient is on hard negatives.
- Batch size `N` controls the number of negatives, which controls the tightness of the MI bound.

Every architectural choice in SimCLR/MoCo/BYOL traces back to managing these terms.

---

## 4. Worked Example: Mini-Batch Contrastive Learning, Step by Step

Let's do a single training step of SimCLR on a batch of `N = 3` images.

### Setup

Three images: `x_1`, `x_2`, `x_3` (cat, dog, plane).

Augmentation pipeline `T`: random crop + horizontal flip + color jitter.

Encoder `f_θ`: ResNet-18 producing 512-dim features.

Projection head `g_φ`: MLP `512 → 512 → 128` with ReLU, output ℓ₂-normalized.

Temperature `τ = 0.5`.

### Step 1: Augment

Apply two random augmentations to each image, giving `2N = 6` views:

```
v_1 = t_a(x_1) (cat, view A)        v_2 = t_b(x_1) (cat, view B)
v_3 = t_c(x_2) (dog, view A)        v_4 = t_d(x_2) (dog, view B)
v_5 = t_e(x_3) (plane, view A)      v_6 = t_f(x_3) (plane, view B)
```

Positive pairs: (1, 2), (3, 4), (5, 6).

### Step 2: Encode and project

Run each through `g_φ ∘ f_θ` to get unit vectors `z_1, ..., z_6 ∈ ℝ^{128}`.

For toy purposes, say (after some training) the cosine similarities form this matrix:

```
       z_1   z_2   z_3   z_4   z_5   z_6
z_1   1.00  0.80  0.10  0.15  0.05  0.08
z_2   0.80  1.00  0.12  0.18  0.06  0.10
z_3   0.10  0.12  1.00  0.85  0.20  0.18
z_4   0.15  0.18  0.85  1.00  0.22  0.20
z_5   0.05  0.06  0.20  0.22  1.00  0.90
z_6   0.08  0.10  0.18  0.20  0.90  1.00
```

Same-image pairs have similarity ~0.85 (cat-cat, dog-dog, plane-plane).
Cross-image pairs are much lower.

### Step 3: Compute per-view loss

For view `z_1`, positive is `z_2`. Negatives are `z_3, z_4, z_5, z_6`.

Scaled similarities (`s / τ = s / 0.5 = 2s`):

```
s(z_1, z_2) / τ = 1.60
s(z_1, z_3) / τ = 0.20
s(z_1, z_4) / τ = 0.30
s(z_1, z_5) / τ = 0.10
s(z_1, z_6) / τ = 0.16
```

(Note: the SimCLR convention excludes only the anchor itself from the denominator. So we sum over all 5 candidates including the positive.)

Exponentials:
```
exp(1.60) = 4.953
exp(0.20) = 1.221
exp(0.30) = 1.350
exp(0.10) = 1.105
exp(0.16) = 1.174

Z = 4.953 + 1.221 + 1.350 + 1.105 + 1.174 = 9.803
```

Loss:
```
ℓ_1 = -log(4.953 / 9.803) = -log(0.5053) ≈ 0.683
```

By symmetry (since cat-cat similarity is the same in both directions), `ℓ_2 ≈ 0.683`.

For view `z_3` (dog A), positive is `z_4` (dog B):
```
s(z_3, z_4) / τ = 1.70
s(z_3, z_1) / τ = 0.20
s(z_3, z_2) / τ = 0.24
s(z_3, z_5) / τ = 0.40
s(z_3, z_6) / τ = 0.36

exp values: 5.474, 1.221, 1.271, 1.492, 1.433
Z = 10.891

ℓ_3 = -log(5.474 / 10.891) = -log(0.5026) ≈ 0.688
```

For view `z_5` (plane A), positive is `z_6`:
```
s(z_5, z_6) / τ = 1.80
s(z_5, z_1) / τ = 0.10
s(z_5, z_2) / τ = 0.12
s(z_5, z_3) / τ = 0.40
s(z_5, z_4) / τ = 0.44

exp: 6.050, 1.105, 1.127, 1.492, 1.553
Z = 11.327

ℓ_5 = -log(6.050 / 11.327) = -log(0.5341) ≈ 0.627
```

By symmetry `ℓ_4 ≈ 0.688`, `ℓ_6 ≈ 0.627`.

### Step 4: Average

```
ℒ = (1 / 6) · (0.683 + 0.683 + 0.688 + 0.688 + 0.627 + 0.627) = 0.666
```

### Step 5: Compare to chance

If the model were random, all similarities would be ~0, all exp values ~1, and the loss for each view would be `-log(1 / 5) = log 5 ≈ 1.609`. So we've improved from 1.609 to 0.666 — the model has clearly learned positive pairs are similar.

### Step 6: Backpropagate

The gradient updates `θ` and `φ` to:

- Pull positive pairs even closer (increase `s(z_1, z_2)`, etc.).
- Push the highest-scoring negatives further away. For `z_1`, the highest negative is `z_4` (similarity 0.15 → gradient pushes them apart).

After many such steps, the representations on the unit sphere become organized: cats cluster, dogs cluster, planes cluster, and the clusters spread out across the sphere.

### Step 7: Evaluate

After training, freeze `f_θ`. Take the 512-dim representation `h = f_θ(x)` for each image. Train a linear classifier `W h → y` on a small labeled dataset. The classifier should achieve high accuracy because the representation has organized images by semantic similarity — cat/dog/plane are linearly separable.

This is the entire SSL pipeline in microcosm.

---

## 5. Relevance to ML Practice

### Where contrastive learning shines

1. **Visual pretraining at scale.** SimCLR, MoCo, BYOL, DINO, all variants of contrastive (or contrastive-like) learning, dominate ImageNet linear probe leaderboards.
2. **Image-text alignment.** CLIP and ALIGN use contrastive learning between images and captions, enabling zero-shot classification.
3. **Information retrieval.** Sentence embeddings (SimCSE, Contriever) are trained contrastively for retrieval tasks.
4. **Recommender systems.** Item embeddings learned by contrasting co-clicked or co-purchased items.
5. **Speech.** wav2vec 2.0 uses InfoNCE between predicted and actual future audio embeddings.
6. **Anomaly detection.** Contrastive features can isolate normal from anomalous via density in representation space.

### When to use contrastive over alternatives

| Choose contrastive when... | Choose generative (MAE/BERT) when... |
|---|---|
| You need a *discriminative* representation (classification, retrieval). | You need a *generative* representation (reconstruction, dense prediction). |
| You're working with images and standard CNN/ViT backbones. | You're using ViTs and want the simplest, most scalable recipe. |
| You can afford large batch sizes or queues. | Batch size is constrained. |
| Augmentation pipeline is well-understood for your domain. | Augmentations are hard to design (medical, satellite). |

### Trade-offs

- **Compute**: contrastive needs many comparisons per anchor (forward passes for all negatives). Large batches are expensive.
- **Memory**: large batches are memory-hungry. MoCo's queue mitigates this; SimCLR doesn't.
- **Hyperparameter sensitivity**: temperature, augmentations, learning rate, and projection head depth all matter.
- **False negatives**: with batch size 4096 on ImageNet, expect a small fraction of "negatives" to be semantically identical (e.g., two cat images). This is a soft ceiling on contrastive performance.
- **Augmentation bias**: the model learns whatever invariances the augmentations encode. Color jitter teaches color-invariance — bad for downstream tasks where color matters.

### Choosing augmentations: the most important hyperparameter

The SimCLR paper showed that the *combination* of augmentations matters more than any individual one. For ImageNet, the best combination is:

1. Random resized crop (most important — provides spatial diversity).
2. Color distortion (jitter brightness, contrast, saturation, hue + random grayscale).
3. Gaussian blur.

Without color distortion, SimCLR would learn to use color histograms as a shortcut (since random crops of the same image share color statistics). Color distortion forces the model to use shape and texture instead.

The augmentation pipeline encodes prior knowledge about what *should not matter* for downstream tasks. This is a critical design choice.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "Just use cross-entropy directly on the embeddings."

The InfoNCE loss is cross-entropy applied to a *softmax over similarities to a candidate set*. It's not the same as supervised classification — there's no fixed class set. The "classes" change every batch (they're the other batch elements). This dynamic class set is what allows the model to learn from any data, with no labels.

### Pitfall 2: "Use the projection head output for downstream tasks."

The projection head is for the contrastive loss. The encoder representation is for downstream tasks. Using the projection head output can drop downstream accuracy by 10+ points.

### Pitfall 3: "More negatives are always better."

Up to a point — adding negatives gives a tighter MI bound. But beyond ~4K negatives in vision, returns diminish. And false negatives become a real problem at scale. Some methods (debiased contrastive, hard negative sampling) try to address this.

### Pitfall 4: "Cosine similarity is just a normalization trick."

Cosine similarity has a deeper effect than just normalization: it constrains representations to the unit sphere. This bounded geometry stabilizes training and prevents representation magnitude from being a confound. Methods that don't normalize tend to be unstable.

### Pitfall 5: "Augmenting harder always helps."

Augmentation should be aggressive enough to make the task non-trivial, but mild enough to preserve semantics. The optimal augmentation strength depends on the dataset and downstream task. A good rule: pick the strongest augmentation that doesn't change a human's perception of the object class.

### Pitfall 6: "False negatives don't matter."

In a batch of 4096 ImageNet images, the probability of two same-class images is ~4096/1000 ≈ 4 per batch (rough). These false negatives generate gradients that *push apart semantically similar images*. This is a small but real effect, contributing to the gap between contrastive SSL and supervised pretraining for some tasks. Methods like supervised contrastive (SupCon) and debiased contrastive try to address this.

### Pitfall 7: "Higher batch size always helps SimCLR."

Yes, up to ~8192. Beyond that, marginal returns drop and optimization becomes harder (large batches need careful LR scaling, warmup, LARS optimizer). It's not free.

### Pitfall 8: "Contrastive learning has nothing to do with metric learning."

It has *everything* to do with metric learning. Contrastive learning is metric learning in the limit where positives are augmentations of the same instance and negatives are everything else. The historical descendants of triplet loss → N-pair loss → InfoNCE form a continuous lineage.

### Pitfall 9: Forgetting to ℓ₂-normalize.

Every modern contrastive method ℓ₂-normalizes embeddings before computing similarity. Forgetting this leads to instability — the model can satisfy the loss by inflating embedding norms instead of changing directions.

### Pitfall 10: "Contrastive losses always give better representations."

Not always. For dense prediction (segmentation, depth), masked modeling (MAE, BEiT) often produces better features. For classification and retrieval, contrastive methods often win. Match the SSL method to the downstream task family.


---

# PART III: SimCLR — A Simple Framework for Contrastive Learning of Visual Representations

> Chen, Kornblith, Norouzi, Hinton. ICML 2020.

## 1. Motivation & Intuition

### What problem SimCLR solved

By 2020, contrastive learning ideas had been around for years (Hadsell et al. 2006, Wu et al. "Instance Discrimination" 2018, CPC), but no method had matched supervised pretraining on ImageNet. SimCLR's contribution was *not* a fundamentally new idea. It was a careful, controlled study of what makes contrastive learning work, distilled into a remarkably simple recipe:

1. Take a batch of images.
2. Augment each one twice.
3. Use a ResNet encoder.
4. Add an MLP projection head.
5. Use NT-Xent (a normalized softmax contrastive loss).
6. Use big batches and strong augmentations.

That recipe matched supervised pretraining on linear probe — a watershed moment for SSL.

### The core intuition

Imagine you want to teach a network "this cat photo and this cropped, color-jittered, blurred version of the same cat photo are the same thing — but they're different from these 4094 other random images in the batch". If the network can learn to do this consistently across millions of images, it must have learned a representation that captures what's invariant under your augmentations (semantic content) and discards what's variable (pose, lighting, color, framing).

The 4094 "other images in the batch" are crucial. The more comparisons, the harder the discrimination task, the more refined the representation must be. This is why batch size matters so much.

### Why "Simple" is in the name

SimCLR removed components that previous methods relied on:
- No memory bank (unlike Wu et al.).
- No specialized architectures.
- No clever pretext tasks beyond instance discrimination.
- No multi-stage training.

Just augmentation + encoder + projection head + softmax loss. This simplicity made SimCLR a clear baseline that subsequent methods (MoCo v2, BYOL, SimSiam) all measure themselves against.

---

## 2. Conceptual Foundations

### The four pillars of SimCLR

1. **Composition of data augmentations**. The right *combination* of augmentations is more important than any single one.
2. **Learnable nonlinear projection**. A 2-layer MLP between the encoder and the contrastive loss substantially improves the encoder's representation.
3. **Normalized temperature-scaled cross entropy (NT-Xent) loss**. A specific contrastive loss with cosine similarity, temperature, and a softmax over the batch.
4. **Large batch sizes and longer training**. SimCLR benefits more than supervised learning from scaling these up.

### Augmentation composition

The SimCLR ablation showed that no single augmentation is sufficient; the *composition* is critical. Random crop alone is weak. Color distortion alone is weak. But random crop + color distortion together is dramatically better than either.

Why? Without color distortion, two crops of the same image have very similar color histograms — the model learns to use color statistics as a shortcut. Without random crop, two color-jittered versions of the same image have the same spatial layout — the model learns position. Together, the two augmentations destroy each other's shortcuts, forcing the model to learn semantic features.

The full SimCLR augmentation pipeline (in order):
1. Random resized crop (`scale = (0.08, 1.0)`, `ratio = (3/4, 4/3)`) → resize to 224×224.
2. Random horizontal flip (p = 0.5).
3. Color jitter (`brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2`), applied with p = 0.8.
4. Random grayscale (p = 0.2).
5. Gaussian blur (kernel 23×23, σ ∈ [0.1, 2.0]), applied with p = 0.5.
6. Normalize (ImageNet mean/std).

### Architecture

- **Encoder** `f_θ`: ResNet-50 (most experiments). Output: 2048-dim global-average-pooled feature `h`.
- **Projection head** `g_φ`: MLP of structure `Linear(2048, 2048) → ReLU → Linear(2048, 128)`. Output: 128-dim `z`. This is ℓ₂-normalized before the loss.

### The NT-Xent loss

NT-Xent stands for *Normalized Temperature-scaled Cross Entropy*. For a batch of `N` images, augment to `2N` views. For each pair `(i, j)` of views from the same image:

```
ℓ(i, j) = - log [ exp(sim(z_i, z_j) / τ)  /  Σ_{k=1}^{2N} 𝟙[k ≠ i] · exp(sim(z_i, z_k) / τ) ]
```

where `sim(u, v) = u^T v / (‖u‖ · ‖v‖)` is cosine similarity.

The total loss is:

```
ℒ = (1 / 2N) · Σ_{k=1}^{N} [ ℓ(2k-1, 2k) + ℓ(2k, 2k-1) ]
```

Each pair contributes twice (asymmetric: `(i, j)` and `(j, i)`).

### Why temperature?

Temperature `τ` scales the similarity scores before softmax. Smaller `τ` means a sharper distribution — the softmax assigns more probability to the highest-scoring negatives. SimCLR uses `τ = 0.5`.

A useful way to think about it: `1/τ` is the "inverse temperature" — at `τ = 0.1`, similarities are inflated 10×, so the softmax behaves as if differences were 10× larger, putting nearly all probability mass on the top candidate.

### Underlying assumptions

1. **Augmentation invariance is the right objective.** Two augmented views of the same image should be invariant under all augmentation transformations.
2. **In-batch negatives are sufficient.** With a batch of 4096 images, you have 8190 negatives per anchor — enough to span a representative chunk of the data distribution.
3. **The encoder benefits from a separate projection space.** The contrastive task is solved in `z`-space; the representation is taken from `h`-space.
4. **Large batches train stably with appropriate optimization.** Use LARS optimizer or specific learning rate schedules.

### What breaks

- **Augmentations that destroy task-relevant information.** Color jitter for bird species. Solution: customize.
- **Small batches.** ResNet-50 SimCLR with batch 256 underperforms supervised by ~5+ points on linear probe; batch 8192 closes the gap.
- **Insufficient training.** SimCLR benefits more than supervised from longer training (1000 epochs vs 100). Why: each "epoch" the model sees a different augmented view; longer training = more views.
- **Removing the projection head.** Loss spikes on linear probe.

---

## 3. Mathematical Formulation

### Notation summary

- `N`: batch size (number of original images).
- `2N`: number of views (each image augmented twice).
- `i, j`: indices into the view set, `i, j ∈ {1, ..., 2N}`.
- `(2k-1, 2k)`: positive pair for image `k`.
- `z_i = g_φ(f_θ(t_i(x_i)))`, ℓ₂-normalized so `‖z_i‖ = 1`.
- `s_{i,j} = z_i^T z_j` (cosine similarity, since vectors are unit-norm).
- `τ`: temperature, typically 0.5 for SimCLR.

### NT-Xent for a single view

For view `i` with positive partner `j(i)`:

```
ℓ_i = - log [ exp(s_{i, j(i)} / τ)  /  Σ_{k ≠ i} exp(s_{i, k} / τ) ]
```

Equivalently:

```
ℓ_i = - s_{i, j(i)} / τ + log Σ_{k ≠ i} exp(s_{i, k} / τ)
```

The first term is the "alignment" term (pulls positives together). The second term is the log-sum-exp over candidates (pushes negatives apart).

### Total batch loss

```
ℒ = (1 / 2N) · Σ_{i=1}^{2N} ℓ_i
```

Note: the loss treats all `2N` views as anchors, each with a positive partner. So each positive pair `(2k-1, 2k)` contributes two terms: `ℓ_{2k-1}` (with positive `2k`) and `ℓ_{2k}` (with positive `2k-1`).

### Derivation: gradient through cosine similarity

Let `z_i = ẑ_i / ‖ẑ_i‖` where `ẑ_i = g_φ(f_θ(t_i(x_i)))`. Cosine similarity:

```
s_{i,k} = z_i^T z_k = (ẑ_i · ẑ_k) / (‖ẑ_i‖ · ‖ẑ_k‖)
```

Partial derivative w.r.t. `ẑ_i`:

```
∂s_{i,k} / ∂ẑ_i = (1 / ‖ẑ_i‖) · (z_k - s_{i,k} · z_i)
```

This is the projection of `z_k` onto the tangent space of the unit sphere at `z_i`, scaled by `1 / ‖ẑ_i‖`.

Why this matters: the gradient that flows back to the encoder is along the unit sphere's tangent — it cannot change `‖ẑ_i‖` (because `s_{i,k}` doesn't depend on it). The loss is invariant to the magnitude of `ẑ_i`. This stabilizes training.

### Gradient of the loss with respect to `s_{i, k}`

For view `i` with positive `j(i)`:

```
∂ℓ_i / ∂s_{i, j(i)} = - (1 / τ) · (1 - p_{i, j(i)})
∂ℓ_i / ∂s_{i, k}     = + (1 / τ) · p_{i, k}      (for k ≠ j(i))
```

where:

```
p_{i, k} = exp(s_{i, k} / τ) / Σ_{m ≠ i} exp(s_{i, m} / τ)
```

is the softmax probability assigned to candidate `k` given anchor `i`.

The gradient pushes `s_{i, j(i)}` higher (with strength proportional to `1 - p_{i, j(i)}`) and pushes each `s_{i, k}` lower (with strength proportional to `p_{i, k}`). The hardest negatives — those with highest similarity to the anchor — receive the largest gradients. This is implicit hard negative mining.

### Connection to InfoNCE and mutual information

NT-Xent is essentially InfoNCE with:
- Cosine similarity (instead of bilinear).
- ℓ₂-normalized embeddings.
- In-batch negatives (no separate negative sampling).
- Symmetric treatment of the two views.

The MI bound applies:

```
I(V_1; V_2) ≥ log(2N - 1) - 𝔼[ℒ_NT-Xent]
```

Larger batch size `N` raises the upper bound on the MI lower bound — i.e., loosens the constraint. This is the formal reason batch size helps.

### Why projection head?

This was an empirical finding, but here's an interpretation. The contrastive loss requires:

```
g_φ(f_θ(t(x))) ≈ g_φ(f_θ(t'(x)))   for any t, t' ∈ T
```

i.e., `g_φ ∘ f_θ` must be invariant to all transformations in `T`. If `g_φ = identity`, then `f_θ` itself must be invariant to all transformations — including color, blur, crop, etc. This forces `f_θ` to discard color information, blur information, position information.

But many of these are useful for downstream tasks! Color matters for distinguishing red apples from green apples; position matters for object detection. The projection head `g_φ` allows `f_θ` to *retain* this information while `g_φ` does the discarding work for the contrastive loss. The encoder representation `h` is therefore richer than `z`.

This is consistent with empirical findings: linear probe accuracy on `h` is ~10 points higher than on `z` for SimCLR (on ImageNet).

### The role of the temperature

Wang & Liu (2021) gave a beautiful analysis: the gradient on negative `k` from anchor `i` is:

```
∂ℒ_i / ∂s_{i, k} = (1 / τ) · p_{i, k}
```

where `p_{i, k} = exp(s_{i, k}/τ) / Z_i` is the softmax probability. The relative weight on hard vs easy negatives is:

```
p_{i, hard} / p_{i, easy} = exp((s_{i, hard} - s_{i, easy}) / τ)
```

For `τ → 0`, this ratio diverges — only the very hardest negatives matter. For `τ → ∞`, all negatives matter equally.

The "right" temperature balances:
- **Hard negative mining** (smaller τ): focuses learning on informative examples.
- **False negative tolerance** (larger τ): doesn't punish the model for failing to separate semantically similar items.

For ImageNet SimCLR, `τ = 0.5` is the sweet spot. Other domains may need different values.

---

## 4. Worked Example: SimCLR on a Tiny Batch

### Setup

Batch of `N = 2` images: `x_1` (cat photo), `x_2` (dog photo).

Apply two augmentations to each: views `1, 2, 3, 4` where:
- View 1, 2 = augmented cat
- View 3, 4 = augmented dog

After encoder + projection + ℓ₂-normalize, suppose:
```
z_1 = (0.6, 0.8)            # ‖z_1‖ = 1
z_2 = (0.65, 0.76)          # ‖z_2‖ ≈ 1
z_3 = (-0.7, 0.71)          # ‖z_3‖ ≈ 1
z_4 = (-0.75, 0.66)         # ‖z_4‖ ≈ 1
```

(2-D for visualization. Real SimCLR uses 128-D.)

### Step 1: Compute pairwise similarities

```
s_{12} = (0.6)(0.65) + (0.8)(0.76) = 0.39 + 0.608 = 0.998
s_{13} = (0.6)(-0.7) + (0.8)(0.71) = -0.42 + 0.568 = 0.148
s_{14} = (0.6)(-0.75) + (0.8)(0.66) = -0.45 + 0.528 = 0.078
s_{23} = (0.65)(-0.7) + (0.76)(0.71) = -0.455 + 0.5396 = 0.085
s_{24} = (0.65)(-0.75) + (0.76)(0.66) = -0.4875 + 0.5016 = 0.014
s_{34} = (-0.7)(-0.75) + (0.71)(0.66) = 0.525 + 0.469 = 0.994
```

So:
- Cat-cat: 0.998 (high — encoder has learned cat invariance)
- Dog-dog: 0.994 (high)
- Cat-dog: ~0.01 to 0.15 (low)

### Step 2: Apply temperature `τ = 0.5`

```
s_{12}/τ = 1.996, s_{13}/τ = 0.296, s_{14}/τ = 0.156
s_{21}/τ = 1.996, s_{23}/τ = 0.170, s_{24}/τ = 0.028
s_{31}/τ = 0.296, s_{32}/τ = 0.170, s_{34}/τ = 1.988
s_{41}/τ = 0.156, s_{42}/τ = 0.028, s_{43}/τ = 1.988
```

### Step 3: Compute softmax denominators

For anchor `i = 1`, candidates are `{2, 3, 4}` (exclude self):
```
Z_1 = exp(1.996) + exp(0.296) + exp(0.156)
    = 7.358 + 1.345 + 1.169 = 9.872
```

For anchor `i = 2`:
```
Z_2 = exp(1.996) + exp(0.170) + exp(0.028)
    = 7.358 + 1.185 + 1.028 = 9.571
```

For `i = 3`:
```
Z_3 = exp(0.296) + exp(0.170) + exp(1.988)
    = 1.345 + 1.185 + 7.300 = 9.830
```

For `i = 4`:
```
Z_4 = exp(0.156) + exp(0.028) + exp(1.988)
    = 1.169 + 1.028 + 7.300 = 9.497
```

### Step 4: Compute per-view losses

```
ℓ_1 = - log( exp(s_{12}/τ) / Z_1 ) = - log(7.358 / 9.872) = - log(0.7454) = 0.2939
ℓ_2 = - log( exp(s_{21}/τ) / Z_2 ) = - log(7.358 / 9.571) = - log(0.7688) = 0.2630
ℓ_3 = - log( exp(s_{34}/τ) / Z_3 ) = - log(7.300 / 9.830) = - log(0.7426) = 0.2976
ℓ_4 = - log( exp(s_{43}/τ) / Z_4 ) = - log(7.300 / 9.497) = - log(0.7687) = 0.2632
```

### Step 5: Total loss

```
ℒ = (1/4) · (0.2939 + 0.2630 + 0.2976 + 0.2632) = 0.2794
```

### Step 6: Compare to chance

If the model were random, all similarities ≈ 0, so all `exp(s/τ) ≈ 1`, and:
```
ℓ_i = - log(1 / 3) = log 3 ≈ 1.099
```

Our loss 0.279 is far below 1.099, indicating significant learning.

### Step 7: Compute softmax probabilities for each anchor

For anchor 1:
```
p_{1, 2} = 7.358 / 9.872 = 0.7454   ← positive
p_{1, 3} = 1.345 / 9.872 = 0.1362
p_{1, 4} = 1.169 / 9.872 = 0.1184
```

The model assigns 74.5% probability to the correct positive. The remaining 25.5% is split between the two negatives, with `z_3` (dog A) slightly more "confused for cat" than `z_4` (dog B).

### Step 8: Inspect gradients

Gradient of `ℓ_1` w.r.t. `s_{1, 2}`:
```
∂ℓ_1 / ∂s_{12} = - (1 / τ) · (1 - p_{1,2}) = - 2 · (1 - 0.7454) = - 0.5092
```

This pushes `s_{12}` higher (the negative gradient is the descent direction; the actual update is `s_{12} ← s_{12} + lr · 0.5092 / 2`, scaled by the learning rate and dividing out the chain rule with `1/τ`).

Gradient w.r.t. `s_{1, 3}`:
```
∂ℓ_1 / ∂s_{13} = + (1 / τ) · p_{1,3} = 2 · 0.1362 = 0.2724
```

Pushes `s_{13}` lower.

Gradient w.r.t. `s_{1, 4}`:
```
∂ℓ_1 / ∂s_{14} = 2 · 0.1184 = 0.2368
```

Slightly lower than for `s_{1, 3}` because `z_3` is more confused. The hardest negative (`z_3`) receives the largest negative-pushing gradient — this is implicit hard negative mining at work.

### Step 9: Backpropagate through the encoder

The `s_{i, k}` gradients propagate back through cosine similarity, projection head, and encoder. The encoder weights are updated to:

- Make cat representations more invariant under augmentation (move `z_1` and `z_2` closer).
- Make cat and dog representations more distinct (move `z_1` away from `z_3, z_4`, and similarly for `z_2, z_3, z_4`).

After many such steps with millions of images, the encoder learns a representation in which all cats live in one region of the sphere, all dogs in another, all planes in another, etc. — without ever being told that cats, dogs, and planes exist.

---

## 5. Relevance to ML Practice

### When to use SimCLR

- You have a large unlabeled image dataset.
- You can afford large batch sizes (4096+) and 100s of GPU days.
- Your downstream tasks are primarily classification or retrieval (where invariance helps).
- You want a simple, well-understood baseline.

### When SimCLR is suboptimal

- Tiny batch sizes (use MoCo instead).
- ViT backbones (use BYOL or DINO or MAE — SimCLR with ViTs has stability issues).
- Dense prediction tasks where pixel-level detail matters (use MAE).
- Domains where strong augmentation is unavailable (use masked modeling).

### Practical hyperparameters (ResNet-50, ImageNet)

| Hyperparameter | Value |
|---|---|
| Batch size | 4096 (linear scaling: 256 per GPU × 16 GPUs) |
| Optimizer | LARS |
| Learning rate | 0.3 × batch_size / 256 = 4.8 |
| Weight decay | 1e-6 |
| LR schedule | Cosine decay with 10-epoch warmup |
| Epochs | 1000 (yes, 1000 — much more than supervised) |
| Temperature | 0.5 |
| Projection head | 2048 → 2048 → 128, ReLU |
| Augmentations | Crop + flip + color jitter + grayscale + blur |

These hyperparameters are unusually expensive — SimCLR pretraining on 8 V100s would take months. Most practitioners use checkpoints or smaller-scale variants.

### Trade-offs

- **Compute cost**: very high. A single SimCLR pretraining run can cost hundreds of thousands of dollars at cloud rates.
- **Conceptual simplicity**: very high. The paper is famously easy to read; the recipe is easy to implement.
- **Robustness**: SimCLR features generalize well across datasets, often better than supervised.
- **Fine-tuning vs probing gap**: SimCLR has a noticeable gap between linear probe (~76% top-1 on ImageNet) and full fine-tuning (~78%), indicating not all information is linearly accessible.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: Treating projection head output as the representation.

The most common mistake. After SimCLR pretraining, the encoder feature `h` is what you use, not the projection output `z`. Using `z` typically loses 5-10 points of linear probe accuracy.

### Pitfall 2: "We can use a small batch and just train longer."

Partially true but misleading. Small batches give a worse MI bound and worse hard negative mining. Longer training with a small batch underperforms shorter training with a large batch in published benchmarks. If you can't afford large batches, use MoCo.

### Pitfall 3: Using the wrong augmentation pipeline.

The exact augmentations matter. Removing color jitter drops accuracy by ~5 points. Removing crop drops accuracy by ~10 points. Don't assume a different augmentation pipeline (e.g., from supervised classification) will work.

### Pitfall 4: Setting temperature to "whatever the SimCSE / MoCo paper used".

Temperature interacts with similarity range, batch size, and dataset. SimCLR's `τ = 0.5` was tuned for ImageNet ResNet-50. Different setups need different `τ`. A common starting point: `τ = 0.1` for MoCo-style, `τ = 0.5` for SimCLR-style, then sweep ±50%.

### Pitfall 5: Forgetting that SimCLR is brittle to LR.

LARS optimizer is essentially required for batch sizes ≥ 4096. Adam at high learning rate diverges. SGD with momentum can work but needs careful warmup. If you see "loss diverges in the first 1000 steps", it's almost always LR or warmup.

### Pitfall 6: Comparing SimCLR to supervised with mismatched epochs.

Supervised ImageNet training typically uses 90-100 epochs. SimCLR uses 1000 epochs. Apples-to-apples comparisons must control for compute. The fair comparison: SimCLR at 100 epochs vs supervised at 100 epochs (SimCLR underperforms substantially), or both at equal compute (SimCLR catches up but doesn't always exceed).

### Pitfall 7: "Adding more augmentations always helps."

The SimCLR ablation showed that adding *some* augmentations (e.g., Sobel filter) actually hurts. The pipeline was carefully tuned. Adding random rotation (90°) hurts because ImageNet has natural orientation. Domain-specific tuning is required.

### Pitfall 8: Believing the projection head must be linear.

The 2-layer MLP (with ReLU) is critical. Replacing with a single linear layer drops accuracy by 3+ points. The nonlinearity matters. Some later methods (DINO) use even deeper projection heads.


---

# PART IV: MoCo — Momentum Contrast for Unsupervised Visual Representation Learning

> He, Fan, Wu, Xie, Girshick. CVPR 2020.

## 1. Motivation & Intuition

### The problem MoCo solved

SimCLR works beautifully — but only if you can run batch size 4096 on 16+ GPUs. Most labs and most practitioners cannot. With batch size 256, SimCLR has only 510 negatives per anchor, which gives a weak MI bound and noisy training.

The natural workaround: keep a *memory bank* of features from past batches. With a memory bank of 65,536 features, every anchor has 65,536 negatives — far more than even SimCLR's biggest batch. And the batch size itself can stay small.

But naive memory banks fail. As the encoder's parameters change between batches, features in the memory bank become *stale* — they were computed by a slightly different encoder. Treating stale features as negatives generates noisy gradients.

MoCo's contribution: a **momentum encoder** that updates slowly via exponential moving average (EMA). The key encoder changes so slowly that features in the queue remain *consistent* — close enough to the current key encoder's outputs to provide useful negatives. This single architectural insight — "use a slowly-updating encoder for the keys" — turned out to be transformative.

### The intuition

Think of the contrastive task as training a query agent to find a matching key in a haystack of candidates. To do well, the haystack should be:

1. **Large** — many candidates, providing fine discrimination.
2. **Consistent** — all keys produced by similar enough encoders that the query's notion of similarity is stable.

SimCLR achieves both via large batches (every key in the batch was produced by the *current* encoder — perfectly consistent — but the haystack is small).

A memory bank without momentum gives you a large haystack, but inconsistent (keys come from many different past encoder states).

MoCo achieves both via:
- **Queue** for size: 65,536 keys instead of 4096.
- **Momentum encoder** for consistency: keys are produced by the EMA-smoothed key encoder, which changes slowly over thousands of steps.

### Real-world analogy

Consider a teacher who walks around a classroom asking each student a different question, then comparing their answers to a list of "correct answers" that the teacher has memorized. If the teacher constantly changes their mind about what the correct answer is, the students get inconsistent feedback. If the teacher's notion of "correct" evolves slowly, the students learn coherently.

In MoCo, the query encoder is the student (updated by gradient). The key encoder is the teacher (updated by EMA). The queue is the teacher's evolving memory of past answers. The momentum keeps the teacher's notion of "correct" stable across many student iterations.

### Why this matters for ML systems

MoCo enabled SSL pretraining on commodity hardware. With MoCo and a single 8-GPU node, you can pretrain a ResNet-50 on ImageNet to near-SimCLR quality — democratizing SSL research. MoCo also opened the door to SSL on enormous datasets that don't fit in a single batch (Instagram billion-image dataset, video datasets).

---

## 2. Conceptual Foundations

### Components

1. **Query encoder** `f_q`: produces query representations from one augmented view. Updated by gradient descent.
2. **Key encoder** `f_k`: produces key representations from another augmented view. Updated by EMA of `f_q`'s parameters.
3. **Queue** `Q`: a FIFO queue of past key embeddings, of size `K` (e.g., 65,536). Acts as the negative bank.
4. **InfoNCE loss**: contrastive loss with one positive (the matching key from the current batch) and `K` negatives (the queue).

### How they interact, step by step

For each minibatch:

1. Sample a batch of `N` images.
2. Generate two augmented views per image: `v_q` (queries) and `v_k` (keys).
3. Compute query embeddings `q = f_q(v_q)`. (Note: this view is the "query" view.)
4. Compute key embeddings `k = f_k(v_k)`, **with no gradient flow into `f_k`**.
5. For each query `q_i`, the positive is `k_i` (the key from the same image). The negatives are all the embeddings currently in the queue `Q`.
6. Compute InfoNCE loss; backpropagate through `f_q` only.
7. Update `f_k` parameters via EMA: `θ_k ← m · θ_k + (1 - m) · θ_q` (typically `m = 0.999`).
8. Enqueue the current batch of keys `{k_i}` into `Q`. Dequeue the oldest batch (FIFO).

### Why EMA momentum

The key encoder must produce *consistent* representations across the entire queue lifetime. A representation enqueued 100 steps ago should still be a meaningful negative now. If `f_k` changed significantly in 100 steps, the old representations would be from a "different" encoder and become unreliable.

EMA with high momentum (`m = 0.999`) ensures `f_k` changes by at most ~0.1% per step. Over 65,536 steps (the queue lifetime), `f_k` evolves smoothly but slowly enough that all queue entries are mutually consistent.

A useful number: with `m = 0.999`, the EMA "half-life" — number of steps for old parameters' contribution to drop by half — is `log(0.5) / log(0.999) ≈ 693` steps. With `m = 0.9999`, half-life is 6,931 steps. MoCo uses `m = 0.999`; MoCo v2 also uses 0.999; BYOL uses a higher `m` that depends on training schedule.

### Why FIFO queue (not random replacement)

FIFO ensures the queue always contains the most recent keys. With random replacement, you'd sometimes evict recent (consistent) keys and keep ancient (stale) ones — defeating the purpose.

### MoCo v1 vs v2 vs v3

- **MoCo v1** (2019): just queue + momentum encoder. No projection head. Strong augmentation. Achieves ~60% linear probe on ImageNet.
- **MoCo v2** (2020): adds a SimCLR-style 2-layer MLP projection head + stronger augmentation (color jitter, blur). Achieves ~71% linear probe.
- **MoCo v3** (2021): replaces ResNet with ViT, removes the queue (uses in-batch negatives like SimCLR — the queue becomes less useful for ViTs because ViT batches are smaller but still large enough). Adds a predictor head similar to BYOL. Achieves ~76% linear probe with ViT-B.

### Underlying assumptions

1. **Slow-changing key encoder produces consistent representations.** True for high momentum but breaks down for low momentum (`m = 0.9` is too low).
2. **Queue keys are not too stale.** Achievable with sufficient momentum and reasonable queue size.
3. **Negatives in the queue are mostly true negatives.** With queue of 65k vs ImageNet 1k classes, expect ~65 false negatives per anchor (small but nonzero).
4. **The query and key encoders converge to similar functions.** The EMA target follows the gradient-trained query encoder, so they don't diverge.

### What breaks

- **Momentum too low** (e.g., `m = 0.9`): queue keys become inconsistent; loss is noisy; performance drops sharply (the MoCo paper has a beautiful ablation showing exactly this).
- **Momentum too high** (e.g., `m = 0.99999`): key encoder doesn't update enough; falls behind the query encoder; representations diverge.
- **Queue too small**: not enough negatives; degrades to small-batch SimCLR.
- **Queue too large**: keys become too stale (some entries are from millions of steps ago).
- **Momentum encoder gets gradients**: this single bug would cause representational collapse — the system would trivially align query and key encoders at the cost of representation quality.

---

## 3. Mathematical Formulation

### Notation

- `f_q`, `f_k`: query and key encoders with parameters `θ_q`, `θ_k`.
- `g_q`, `g_k`: projection heads (in MoCo v2+).
- `q_i = g_q(f_q(t_q(x_i)))`: query for image `i`.
- `k_i = g_k(f_k(t_k(x_i)))`: key for image `i` (no gradient flow).
- `Q = [k^{(1)}, k^{(2)}, ..., k^{(K)}]`: the queue of past keys.
- `m`: momentum coefficient (e.g., 0.999).
- `τ`: temperature (typically 0.07 in MoCo, 0.2 in MoCo v2).

### MoCo InfoNCE loss

For a single query `q_i` with positive `k_i` and queue `Q = {k^{(1)}, ..., k^{(K)}}`:

```
ℒ_q_i = - log [ exp(q_i^T k_i / τ)  /  ( exp(q_i^T k_i / τ) + Σ_{j=1}^{K} exp(q_i^T k^{(j)} / τ) ) ]
```

All embeddings are ℓ₂-normalized so that `q^T k = cos_sim(q, k)`.

For a batch of `N` queries:

```
ℒ = (1 / N) · Σ_{i=1}^{N} ℒ_q_i
```

### Momentum update

After each gradient step on `θ_q`:

```
θ_k ← m · θ_k + (1 - m) · θ_q
```

This is an exponential moving average. Equivalently, after `T` steps:

```
θ_k^(T) = m^T · θ_k^(0) + (1 - m) · Σ_{t=0}^{T-1} m^{T-1-t} · θ_q^(t)
```

For `m = 0.999`:
- Initial conditions decay: contribution of `θ_k^(0)` drops by factor `m^T`. After 1000 steps: `0.999^1000 ≈ 0.368`. After 10000 steps: `~ 4.5 × 10⁻⁵` (essentially zero).
- The EMA approximates the average of the last `~ 1/(1-m) = 1000` steps of `θ_q`.

### Queue update

After each batch, enqueue current batch of keys, dequeue the oldest batch:

```
Q ← concat(Q[N:], k_batch)
```

The queue is treated as a list of fixed-length feature vectors; gradients do not flow into the queue.

### Gradient analysis

The loss only flows backward through `q_i` (the gradient through the queue and through `k_i` is detached). Specifically:

```
∂ℒ_q_i / ∂q_i = - (1 - p_+) · k_i / τ + Σ_{j=1}^{K} p_j · k^{(j)} / τ
```

where `p_+` and `p_j` are the softmax probabilities for the positive and the j-th negative.

This pulls `q_i` toward `k_i` (positive term) and away from the high-probability queue keys (negative terms).

The gradient does **not** flow into `θ_k` directly. `θ_k` is updated only through the EMA. This breaks the symmetry that would otherwise allow trivial collapse — if both encoders received gradient, they could align by both moving to a constant function.

### Connection to InfoNCE bound

MoCo's InfoNCE bound:

```
I(Q; K) ≥ log(K + 1) - 𝔼[ℒ]
```

Larger queue size `K` → tighter MI bound. MoCo uses `K = 65536`, vs SimCLR's effective `K = 2N - 1 = 8191` at batch size 4096. So MoCo's bound is ~3 nats tighter than SimCLR's.

### Why momentum prevents collapse

Suppose we removed the momentum and made `θ_k = θ_q` at all times (i.e., shared encoder). Then for any input `x` with views `v_q, v_k`:

```
q = f(t_q(x)),    k = f(t_k(x))
```

Both `q` and `k` are computed by the *same* encoder. The loss requires `q · k > q · k^{(j)}` for all queue keys.

If the encoder collapses to `f(x) = c` for some constant vector `c`, then `q = k = c` for any input, and `q · k = ‖c‖² = 1` (since normalized). Meanwhile `q · k^{(j)} = ‖c‖² = 1` for queue keys too. The loss becomes `-log(1 / (K+1)) = log(K+1)`, which is the chance level — *not* zero. So shared-encoder collapse doesn't actually minimize this contrastive loss.

But it's worse than chance for typical losses with negatives — because all candidates equally win, the model doesn't perform any discrimination. The contrastive loss penalizes this.

Shared-encoder shouldn't collapse, but in practice it can be unstable. The momentum encoder provides additional stability and decoupling.

### MoCo v3 (ViT) modifications

For ViT backbones, MoCo v3 made several changes:
- Removed the queue (in-batch negatives like SimCLR).
- Added a 3-layer predictor MLP after the query projection (BYOL-inspired).
- Used the symmetric loss (loss on both view orderings).
- Found that fixing the patch projection (random init, no training) stabilized ViT contrastive training.

The architectural simplification (no queue) was possible because ViTs work with reasonably large batches, and the queue's main benefit (cheap negatives) became less important than its complication (managing the FIFO).

---

## 4. Worked Example: One MoCo Step

### Setup

- Batch size `N = 4`.
- Queue size `K = 8` (toy; real MoCo uses 65,536).
- Embedding dim `d = 4` (toy).
- Temperature `τ = 0.07`.
- Momentum `m = 0.999`.

### Step 1: Initialize

Random init `θ_q`. Set `θ_k = θ_q` (paper says to init key encoder identically).
Initialize queue `Q` with `K = 8` random unit vectors.

### Step 2: Sample batch and augment

4 images. Apply 2 augmentations per image to get 4 query views and 4 key views.

### Step 3: Forward pass

```
q = f_q(query_views)   # (4, 4) — 4 queries of dim 4
k = f_k(key_views)     # (4, 4) — 4 keys of dim 4, NO GRADIENT
```

After ℓ₂-normalization:

```
q = [[ 0.5,  0.5,  0.5,  0.5],
     [ 0.7,  0.7,  0.0,  0.14],
     [-0.6, -0.6,  0.4,  0.36],
     [ 0.3, -0.3,  0.7,  0.58]]

k = [[ 0.45,  0.55,  0.5,  0.5 ],
     [ 0.7,   0.65,  0.1,  0.27],
     [-0.65, -0.55,  0.45, 0.27],
     [ 0.32, -0.28,  0.72, 0.55]]
```

(Approximately matched query-key pairs — i.e., the model has learned roughly.)

### Step 4: Compute positive similarities

```
sim(q_1, k_1) = 0.5·0.45 + 0.5·0.55 + 0.5·0.5 + 0.5·0.5 = 0.225+0.275+0.25+0.25 = 1.000   # nearly aligned
sim(q_2, k_2) = 0.7·0.7 + 0.7·0.65 + 0.0·0.1 + 0.14·0.27 ≈ 0.49+0.455+0+0.038 = 0.983
sim(q_3, k_3) = (-0.6)(-0.65) + (-0.6)(-0.55) + 0.4·0.45 + 0.36·0.27 = 0.39+0.33+0.18+0.097 = 0.997
sim(q_4, k_4) = 0.3·0.32 + (-0.3)(-0.28) + 0.7·0.72 + 0.58·0.55 = 0.096+0.084+0.504+0.319 = 1.003
```

(Slight numerical noise; in practice these would be ≤ 1 after normalization. Treat as ≈ 1.0 for the example.)

### Step 5: Compute negative similarities (queue)

Suppose for query `q_1`, similarities to the 8 queue entries are:
```
[0.10, -0.05, 0.20, 0.15, -0.10, 0.08, 0.25, 0.30]
```

(All low, since queue contains random/different images.)

### Step 6: Compute loss for `q_1`

Scaled similarities (`/ τ = / 0.07`):

```
positive: 1.00 / 0.07 = 14.29
negatives: [1.43, -0.71, 2.86, 2.14, -1.43, 1.14, 3.57, 4.29]
```

Exponentials:
```
exp(14.29) ≈ 1.6 × 10⁶
exp(1.43) ≈ 4.18
exp(-0.71) ≈ 0.49
exp(2.86) ≈ 17.46
exp(2.14) ≈ 8.50
exp(-1.43) ≈ 0.24
exp(1.14) ≈ 3.13
exp(3.57) ≈ 35.52
exp(4.29) ≈ 73.0

sum_neg = 4.18 + 0.49 + 17.46 + 8.50 + 0.24 + 3.13 + 35.52 + 73.0 = 142.52
total = 1.6e6 + 142.52 ≈ 1.6e6 (positive dominates)

loss_1 = - log(1.6e6 / 1.6e6) ≈ - log(0.99991) ≈ 8.9 × 10⁻⁵
```

The loss is essentially zero because the positive dominates. The model is well-trained on this query.

If the model were random, positive sim ≈ 0:
```
loss_1 ≈ - log(1 / (1 + 8)) = log 9 ≈ 2.197
```

### Step 7: Aggregate over batch

Suppose all four queries have similar low losses. Total batch loss: small.

### Step 8: Backpropagate

The gradient updates `θ_q` only. `θ_k` does not receive gradient.

### Step 9: EMA update

```
θ_k ← 0.999 · θ_k + 0.001 · θ_q
```

After this step, `θ_k` has moved 0.1% of the way toward the new `θ_q`.

### Step 10: Update queue

Take the 4 key vectors `k = [k_1, k_2, k_3, k_4]` and enqueue them. Dequeue the oldest 4 entries (positions 0-3 in the queue). New queue: positions [old 4, old 5, old 6, old 7, k_1, k_2, k_3, k_4].

### Step 11: Next iteration

Repeat. With each batch, the queue refreshes by `N` entries (FIFO), maintaining a sliding window of the most recent `K` keys, all produced by very-recent versions of `f_k`.

### Why this works (intuitively)

- The query encoder `f_q` learns to separate semantically distinct images by being asked to discriminate the current key from 65k other (mostly different) keys.
- The key encoder `f_k` follows `f_q` slowly, so the queue stays consistent.
- Because the queue is large, the contrastive task is hard (the model can't memorize), forcing the encoder to learn rich semantic features.

After ~200 epochs of this on ImageNet, the encoder learns features that match SimCLR's quality with 16× less GPU memory.

---

## 5. Relevance to ML Practice

### When to use MoCo

- **Limited GPU memory**: can't fit large batches. MoCo's queue gives many negatives at small batch.
- **Streaming or online learning**: the queue naturally accumulates representative negatives over time.
- **Multi-modal contrastive learning**: MoCo-style queues are useful for video, audio, where samples are large.
- **Domain-specific SSL**: MoCo-style training scales easily to billion-image datasets like Instagram.

### When to prefer SimCLR or BYOL

- **You have large-batch hardware**: SimCLR's simplicity is appealing if you can afford batch 4096.
- **You want no negatives at all**: BYOL avoids the false-negative problem and the momentum/queue management.
- **You're using ViTs**: MoCo v3 is good, but BYOL/DINO are widely adopted for ViTs.

### Practical hyperparameters (MoCo v2, ResNet-50, ImageNet)

| Hyperparameter | Value |
|---|---|
| Batch size | 256 |
| Queue size `K` | 65,536 |
| Momentum `m` | 0.999 |
| Temperature `τ` | 0.2 |
| Optimizer | SGD with momentum 0.9 |
| Learning rate | 0.03 (cosine schedule) |
| Weight decay | 1e-4 |
| Epochs | 200-800 |
| Augmentations | Crop + flip + strong color jitter + Gaussian blur |

These run on 8 V100s in a few days — far cheaper than SimCLR.

### Trade-offs

- **Compute**: much cheaper than SimCLR (memory-wise).
- **Implementation complexity**: more than SimCLR. The queue, momentum encoder, and "shuffle BN" trick (to prevent batch-norm leakage between query/key encoders) all need careful implementation.
- **Hyperparameter sensitivity**: momentum coefficient is critical. Queue size has a sweet spot.
- **Stability**: MoCo is more stable than SimCLR at small batch sizes but less stable than BYOL at very small batches.

### Shuffle BN: a subtle but critical detail

If query and key encoders share the same batch (which they do — same images, different augmentations), and both use BatchNorm, the BN statistics of the keys leak information into the query path. The model can use BN statistics as a shortcut to identify which key matches which query. This causes representational collapse.

**Solution**: shuffle the batch before passing through the key encoder (across multiple GPUs in distributed training). The key encoder sees a different ordering / partition than the query encoder, so BN statistics are uncorrelated. After encoding, unshuffle.

This trick is MoCo-specific and easy to forget. Without it, MoCo is broken.

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: Forgetting to detach the key encoder gradient.

If gradients flow through `f_k`, the model can collapse trivially. Always detach: `k = f_k(x).detach()` (in PyTorch).

### Pitfall 2: Setting momentum too low.

`m = 0.99` (a "reasonable" sounding value) actually performs much worse than `m = 0.999`. The MoCo paper has an ablation showing accuracy drops by ~2 points going from 0.999 to 0.99, and by ~4 points going to 0.9.

### Pitfall 3: Queue too small or too large.

Sweet spot is `K = 65536` for ImageNet. Smaller (`K = 4096`) loses the MoCo advantage. Larger (`K = 1M`) introduces too-stale keys.

### Pitfall 4: Not using shuffle BN.

Without shuffle BN, MoCo collapses or trains poorly. This is a frequent bug in re-implementations.

### Pitfall 5: Initializing key encoder differently from query.

The original MoCo initializes `θ_k = θ_q`. Random independent init causes early instability.

### Pitfall 6: "MoCo's queue stores raw images."

No — the queue stores *encoded features* (output of `g_k(f_k(·))`). Storing images would be memory-prohibitive.

### Pitfall 7: Updating `θ_k` before computing gradients on `θ_q`.

Order matters: compute loss → backprop to `θ_q` → step `θ_q` → EMA update `θ_k`. If you EMA-update before backprop, the keys and queue used for the loss correspond to a stale `θ_k`.

### Pitfall 8: Using MoCo v1 augmentations with v2 architecture.

MoCo v2's gains over v1 came mostly from SimCLR-style augmentations. Don't mix and match.

### Pitfall 9: Confusing MoCo's momentum with optimizer momentum.

Two different things. SGD's momentum is for parameter updates. MoCo's momentum is for the EMA between encoders. They're independent.

### Pitfall 10: Treating MoCo as "obviously better than SimCLR".

MoCo is more memory-efficient. SimCLR is conceptually simpler. At equal compute, they reach similar accuracy. The choice depends on your hardware, your codebase, and your stomach for queue management.


---

# PART V: BYOL — Bootstrap Your Own Latent

> Grill, Strub, Altché, Tallec, Richemond et al. NeurIPS 2020.

## 1. Motivation & Intuition

### The shocking result

In 2020, BYOL ("Bootstrap Your Own Latent") published a result that seemed impossible: **contrastive learning without negatives**. No queue. No big batches. No "push apart" loss term. Just predict one view's representation from another. And it worked — BYOL matched or exceeded SimCLR and MoCo on ImageNet.

This was deeply puzzling. Conventional wisdom said: without negatives, the model would collapse — every input maps to the same vector, satisfying any prediction loss trivially. How does BYOL avoid this?

The answer is subtle, involves careful asymmetric architecture, and has been the subject of a small academic literature (Tian et al. 2021; Richemond et al. 2020; Grill et al. 2020 ablations). The short version: **predictor network + EMA target + asymmetry create a slow, stable dynamics that escapes collapse**.

### The intuition (simplified)

Suppose you want a network to learn good representations of images. You take two augmented views of the same image. You make one network ("online") try to *predict* what the other network ("target") would output for the second view. The target network is a slowly-updating EMA of the online network.

If both networks were the same, prediction would be easy — output a constant. But the target network is *not* updated by gradient. It changes only via EMA. So it lags behind. Meanwhile, the online network's predictor is trying to "catch up" with where the target *was*, while the target itself is slowly drifting toward where the online *is*.

This creates a curious dynamics: the online tries to match the target; the target is a slow average of the online; if both collapse to a constant, the system would be stable but useless. But the *predictor network* (a small MLP between the online encoder and the loss) prevents this — it absorbs trivial constant solutions and creates a tension that pushes the encoder toward more informative representations.

It's not a clean story. But it works empirically, and ablations confirm: remove the predictor → collapse. Remove the EMA → collapse. Remove augmentations → collapse. The combination is what saves it.

### Why this matters

- **No negatives** = no false-negative problem. SimCLR and MoCo can pull apart semantically similar images by treating them as negatives. BYOL doesn't have this issue.
- **Smaller batches work**. BYOL doesn't depend on batch size for negatives, so it works at batch size 256, 512, etc. — comparable to MoCo.
- **Simpler hyperparameters**. No temperature. No queue size. No negative sampling strategy.
- **Inspired a family**: SimSiam (no EMA — uses just stop-gradient), DINO (centering + sharpening), VICReg (variance-invariance-covariance), Barlow Twins (cross-correlation). All avoid negatives.

### Real-world analogy

Imagine two students studying together. Student A reads notes from yesterday's lecture and tries to predict what Student B would say about today's lecture. Student B's "current understanding" is a slowly-updated synthesis of his past understandings (EMA of his learning).

If both students just gave the same answer to everything, this game would be too easy — they'd "win" but learn nothing. But Student A has to use a translator (the predictor MLP) that adds a layer of inference. The translator can't just pass through; it has to reason about what Student B *would* say given a new perspective. This forces both students to develop richer internal models to keep the game working.

After many days, both students have learned to articulate the lecture's structure precisely — without ever being told "this is right" or "this is wrong" by an outside grader.

---

## 2. Conceptual Foundations

### The architecture

BYOL has two networks:

**Online network** (gradient-trained):
- Encoder `f_θ`
- Projector `g_θ` (MLP)
- Predictor `q_θ` (MLP)

**Target network** (EMA of online):
- Encoder `f_ξ`
- Projector `g_ξ` (MLP)
- (No predictor — the target stops at the projection.)

### How they interact

For a single image `x`, two augmented views `v = t(x)`, `v' = t'(x)`:

1. Online path:
   - `y = f_θ(v)` (encoder representation)
   - `z = g_θ(y)` (projection)
   - `p = q_θ(z)` (prediction)

2. Target path:
   - `y' = f_ξ(v')` (encoder representation, no gradient)
   - `z' = g_ξ(y')` (projection, no gradient)

3. Loss: minimize the squared L2 distance between (normalized) `p` and `z'`:
```
ℒ = ‖ p̂ - ẑ' ‖²    where  p̂ = p / ‖p‖, ẑ' = z' / ‖z'‖
```

This equals `2 - 2 · cos_sim(p, z')` — minimizing it maximizes cosine similarity.

4. Symmetrize: also compute the loss in the other direction (online sees `v'`, target sees `v`), and add.

5. Backprop through online network only.

6. EMA update target: `ξ ← τ · ξ + (1 - τ) · θ` where `τ` is the EMA coefficient (often called `m` in BYOL papers; we use `τ` here for consistency, even though it conflicts with temperature notation).

### Key components and why each matters

1. **The predictor `q_θ`** is the secret sauce. It's a small MLP (e.g., 256 → 4096 → 256 with BN + ReLU) that maps the online projection to a "prediction" of the target's projection. Without the predictor (i.e., comparing `z` directly to `z'`), BYOL collapses immediately. The predictor breaks the symmetry between online and target, and absorbs collapse pressures.

2. **EMA target**. The target slowly tracks the online. This provides a "moving target" that is consistent across batches but slow enough that the online has to actually represent the data to match it.

3. **Stop-gradient on target**. Crucial. If gradient flowed back through the target, the system could trivially align both networks (gradient finds a way to fold both into a constant function).

4. **Augmentations**. The two views differ by augmentation. Without augmentation, the prediction task would be too trivial.

5. **No negatives** (the namesake feature). BYOL never compares to other images.

### Underlying assumptions

1. **The predictor cannot collapse.** BYOL's stability depends on the predictor's failure to map to a constant. Empirically, this holds for properly initialized predictors with batch normalization.
2. **The EMA target stays close to but lagged behind the online.** This requires a sufficiently high EMA coefficient (τ ≈ 0.99-0.999).
3. **Augmentations preserve semantics.** Same as for SimCLR.
4. **Random initialization breaks symmetry.** With identical online and target init, BYOL works because of the predictor and the dynamics. With trivial init (all zeros), it can't escape.

### What breaks

- **Remove predictor** → immediate collapse.
- **Remove EMA target** (online predicts itself) → immediate collapse.
- **Remove stop-gradient** → collapse.
- **Set EMA coefficient too low** → instability, sometimes collapse.
- **Remove augmentation** → trivial solution (identity prediction).
- **Single view (`v = v'`)** → identity prediction is easy, no learning.

### Three families of "non-contrastive" methods

BYOL spawned a small zoo of related methods:

- **BYOL**: predictor + EMA target.
- **SimSiam** (Chen & He 2021): predictor + stop-gradient (no EMA — just stop gradient at one branch). Surprisingly, this works too. Implication: EMA isn't strictly necessary; stop-gradient is what matters.
- **DINO** (Caron et al. 2021): student-teacher with EMA, but the loss is cross-entropy with a centering operation on the teacher output to prevent collapse. Works exceptionally well with ViTs.
- **VICReg** (Bardes et al. 2021): explicit variance, invariance, covariance terms in the loss. No negatives, no predictor; collapse prevented by the variance term.
- **Barlow Twins** (Zbontar et al. 2021): minimize off-diagonal cross-correlation between batch embeddings; no predictor or negatives.

The collective takeaway: there are *many* ways to avoid negatives, all relying on some form of asymmetry, regularization, or feature decorrelation.

---

## 3. Mathematical Formulation

### Notation

- `θ`: online network parameters; `ξ`: target network parameters.
- `f_θ, g_θ, q_θ`: online encoder, projector, predictor.
- `f_ξ, g_ξ`: target encoder, projector (no predictor).
- `τ ∈ (0, 1)`: EMA coefficient (often denoted `m` in code).
- `t, t'`: augmentations sampled from `T`.
- `v = t(x), v' = t'(x)`: two views.

### Forward pass

Online:
```
y_θ = f_θ(v)
z_θ = g_θ(y_θ)
p_θ = q_θ(z_θ)
```

Target:
```
y_ξ' = f_ξ(v')
z_ξ' = g_ξ(y_ξ')   (no gradient w.r.t. ξ here)
```

### Loss

ℓ₂-normalize predictions and targets:
```
p̂_θ = p_θ / ‖p_θ‖_2
ẑ_ξ' = z_ξ' / ‖z_ξ'‖_2
```

BYOL loss for one direction:
```
ℒ_θ,ξ = ‖ p̂_θ - ẑ_ξ' ‖²_2
      = 2 - 2 · ⟨p̂_θ, ẑ_ξ'⟩
      = 2 - 2 · cos_sim(p_θ, z_ξ')
```

Symmetric loss (swap views):
```
ℒ_total = ℒ_θ,ξ(v, v') + ℒ_θ,ξ(v', v)
```

The gradient updates only `θ` (through the online network):
```
θ ← θ - η · ∇_θ ℒ_total
```

EMA update:
```
ξ ← τ · ξ + (1 - τ) · θ
```

The EMA coefficient `τ` is often scheduled (BYOL uses base `τ = 0.996` and increases toward 1 over training).

### Why `2 - 2·cos_sim`?

The squared L2 distance between unit vectors:
```
‖p̂ - ẑ'‖² = ‖p̂‖² - 2 p̂^T ẑ' + ‖ẑ'‖² = 1 - 2 cos_sim + 1 = 2 - 2 cos_sim
```

So minimizing the squared distance = maximizing cosine similarity. Practically, cosine similarity loss `1 - cos_sim` is equivalent up to a constant.

### Why does this avoid collapse?

Suppose the online network collapses to a constant: `f_θ(x) = c, g_θ(c) = c'`. Then `q_θ(c') = q_θ(c') = c''`, a constant. Meanwhile, the target also tends to a constant via EMA: `f_ξ(x) → c, g_ξ(c) → c'`.

The loss becomes `2 - 2 · cos_sim(c'', c') = 2 - 2 · cos_sim(c'', c')`. If `c'' = c'`, loss is 0. So the constant solution is a global minimum — yet BYOL doesn't find it. Why?

The proposed mechanisms (Tian et al. 2021, others):

1. **The predictor's eigenvalue dynamics**. The predictor `q_θ` is initialized randomly, so it doesn't immediately map to a constant. As training proceeds, the predictor's optimal weight matrix is the *correlation* between online and target features. This sets up a feedback loop where good representations are reinforced and constant solutions are unstable.

2. **EMA's lag**. The target lags behind the online. Even if the online "tries" to collapse, the target retains memory of less-collapsed past states. This creates pressure away from collapse on a short timescale — but the predictor needs to handle the long timescale.

3. **BatchNorm interaction**. BN's per-dimension variance constraint provides implicit pressure for diverse outputs. This was thought (early on) to be the *whole* explanation, but Richemond et al. showed BYOL works without BN if other tricks are used.

4. **Initialization**. BYOL doesn't start in a collapsed state. The optimization may simply not find collapse from the random init basin.

The empirical conclusion: collapse is *theoretically* a global minimum, but the dynamics of gradient descent + EMA + predictor avoid it. This is unusual — most ML training succeeds because we engineered the loss landscape; BYOL succeeds despite the loss landscape having a trivial solution.

### A formal analysis (sketch)

Tian et al. ("Understanding self-supervised learning dynamics without contrastive pairs", 2021) analyzed a linear BYOL: linear encoder `W_θ`, linear projector, linear predictor `W_p`, EMA target `W_ξ`. They showed:

- The optimal predictor is `W_p* = E[(W_θ x)(W_ξ x)^T] · (E[(W_θ x)(W_θ x)^T])^{-1}`, i.e., the Wiener filter.
- With this optimal predictor, the loss reduces to a function of `W_θ` and `W_ξ` that has *non-trivial* minima even in linear case.
- The EMA dynamics drive `W_θ` toward the eigenspace of the data covariance — a Hebbian-like update — which is meaningful representation learning.

For nonlinear BYOL, the analysis is harder, but the same flavor of dynamics is conjectured: the predictor encodes the relationship between online and target outputs, and the EMA + gradient dynamics extract data structure.

---

## 4. Worked Example: One BYOL Step

### Setup

- Image `x` (single image for simplicity).
- Two augmented views `v` and `v'`.
- Online: encoder + projector + predictor, all 1D for visualization (real BYOL is 256-D projection).
- Target: encoder + projector, EMA of online.
- EMA coefficient `τ = 0.996`.

### Step 1: Forward online

`y_θ = f_θ(v) = 0.7` (encoder output, 1D toy).
`z_θ = g_θ(y_θ) = 0.6` (projection).
`p_θ = q_θ(z_θ) = 0.5` (prediction).

Normalize: `p̂_θ = sign(p_θ) = +1` (in 1D, normalization trivializes; in real BYOL this is in 256-D).

For pedagogy, let's switch to 2D.

`p_θ = (0.5, 0.5)`, `‖p_θ‖ = 0.707`, `p̂_θ = (0.707, 0.707)`.

### Step 2: Forward target

`y_ξ' = f_ξ(v') = (0.6, 0.4)` (target encoder, no gradient).
`z_ξ' = g_ξ(y_ξ') = (0.5, 0.4)`.
Normalize: `‖z_ξ'‖ = sqrt(0.25 + 0.16) = sqrt(0.41) = 0.640`.
`ẑ_ξ' = (0.781, 0.625)`.

### Step 3: Compute loss

```
ℒ = ‖p̂_θ - ẑ_ξ'‖² 
  = (0.707 - 0.781)² + (0.707 - 0.625)²
  = (-0.074)² + (0.082)²
  = 0.00548 + 0.00672
  = 0.0122

cos_sim(p̂_θ, ẑ_ξ') = (0.707)(0.781) + (0.707)(0.625) = 0.552 + 0.442 = 0.994
2 - 2·(0.994) = 0.012  ✓ (matches)
```

The loss is small — the prediction is close to the target. The model has learned reasonable representations.

### Step 4: Symmetric loss

Recompute with views swapped:
- Online sees `v'`, target sees `v`.
- Suppose `p̂_θ' = (0.620, 0.785)`, `ẑ_ξ = (0.640, 0.768)`.
```
ℒ' = (0.620 - 0.640)² + (0.785 - 0.768)² = 0.0004 + 0.000289 = 0.00069
```

Total: `ℒ_total = ℒ + ℒ' = 0.0122 + 0.00069 = 0.01289`.

### Step 5: Backpropagate

Gradient flows back through the online predictor, projector, and encoder. The target receives no gradient.

The gradient w.r.t. `p̂_θ`:
```
∂ℒ / ∂p̂_θ = 2 · (p̂_θ - ẑ_ξ')
           = 2 · (0.707 - 0.781, 0.707 - 0.625)
           = 2 · (-0.074, 0.082)
           = (-0.148, 0.164)
```

This gradient (after backprop through normalization, predictor, projection, encoder) updates `θ`.

### Step 6: EMA update

After applying the gradient update to `θ`:
```
ξ ← 0.996 · ξ + 0.004 · θ
```

The target moves 0.4% toward the new online parameters.

### Step 7: Diagnostic: detect collapse

Compute the standard deviation of `z_θ` across the batch (in real training; not meaningful for 1 example). If the std crashes (e.g., below 0.01), the model is collapsing. Healthy BYOL should maintain `std ≈ 1/sqrt(d)` for d-dimensional features (after normalization).

### Step 8: After many steps

Over thousands of steps, the online and target encoders both learn to encode meaningful semantic content. The EMA averages out batch-level noise. The predictor learns to map `z_θ` to its corresponding `z_ξ'`, which on average is the conditional expectation `E[z_ξ' | v]` — a meaningful prediction problem.

After 1000 epochs of this on ImageNet, BYOL's encoder produces representations that achieve ~74% linear probe accuracy — comparable to or better than SimCLR.

---

## 5. Relevance to ML Practice

### When to use BYOL

- **Small or moderate batches**: BYOL works at batch 256, 512, etc.
- **No false-negative concerns**: domains where many "negative" pairs would actually be semantically similar (e.g., a dataset with many duplicates or near-duplicates).
- **You want simplicity**: no temperature, no queue size, no negative count to tune.
- **ViT pretraining**: BYOL-style and DINO-style methods work well with ViTs.

### When BYOL might be suboptimal

- **You need explicit information-theoretic guarantees**: contrastive methods have MI-bound interpretations; BYOL's theory is murkier.
- **You suspect collapse**: BYOL is more fragile to hyperparameters than contrastive methods. If collapse is happening, you have to debug subtle architectural choices.
- **Compatibility with downstream methods that expect a contrastive structure**: some methods (e.g., supervised contrastive fine-tuning) assume contrastive features.

### Practical hyperparameters (ResNet-50, ImageNet)

| Hyperparameter | Value |
|---|---|
| Batch size | 4096 (but works at 256-2048) |
| Optimizer | LARS |
| Learning rate | 0.2 × batch_size / 256 = 3.2 (cosine schedule, 10-epoch warmup) |
| Weight decay | 1.5e-6 |
| EMA coefficient `τ` | 0.996, increased to 1.0 over training |
| Projector | 4096 → 256 |
| Predictor | 256 → 4096 → 256 with BN + ReLU |
| Augmentations | Same as SimCLR |
| Epochs | 1000 |

### Trade-offs

- **Stability**: more sensitive to architectural details than SimCLR. Bugs cause collapse.
- **Compute**: similar to SimCLR.
- **Theoretical clarity**: less than contrastive methods.
- **Empirical performance**: matches or beats contrastive baselines on most benchmarks.
- **Implementation complexity**: lower than MoCo (no queue), comparable to SimCLR (plus EMA target and predictor).

---

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: "BYOL works because of BatchNorm."

This was an early hypothesis (Tian et al. 2020 first version) but was later disproven. Richemond et al. ("BYOL works *even* without batch statistics") showed BYOL works with GroupNorm + careful initialization. BatchNorm helps but isn't necessary. The real reasons are predictor + EMA + asymmetry.

### Pitfall 2: Removing the predictor "for simplicity".

Don't. The predictor is critical. Removing it → immediate collapse. SimSiam variants that drop the EMA can keep working, but only if they keep the predictor + stop-gradient.

### Pitfall 3: Setting EMA coefficient too low.

`τ = 0.99` (a common "default") makes the target drift too fast, leading to instability. BYOL uses `τ = 0.996` as the base, scheduled to 1.0. Setting it too high (e.g., `τ = 0.9999`) makes the target lag too much, also harmful.

### Pitfall 4: Confusing BYOL with BYOL-style fine-tuning.

BYOL is a pretraining method. The target network is discarded after pretraining; you only keep the online encoder. The predictor is also discarded. Only the encoder representation `y_θ = f_θ(x)` is used for downstream tasks.

### Pitfall 5: Using BYOL features by passing through the projector.

Like SimCLR, the encoder features `y_θ` are richer than the projection `z_θ`. Don't use the projection.

### Pitfall 6: Forgetting symmetric loss.

Single-direction loss (online only ever predicts target on the second view) underperforms symmetric loss by a small but consistent margin. Both BYOL and SimSiam use symmetric formulations.

### Pitfall 7: "BYOL has no contrastive structure."

Implicit contrast still exists. The EMA target is computed across the *whole batch* (BatchNorm statistics couple batch members). When BatchNorm is replaced (as in the no-BN version), the predictor's optimal solution still depends on the data distribution — which is implicitly contrastive across the dataset.

### Pitfall 8: Believing BYOL is "easier" than SimCLR/MoCo.

BYOL is harder to debug. When it works, it works simply. When it collapses, you have to inspect predictor weights, EMA dynamics, BN statistics, and feature variance simultaneously. Contrastive methods give you a clear loss signal that diagnoses progress.

### Pitfall 9: Not monitoring feature variance.

Always log `std(z_θ)` or `std(y_θ)` across the batch during BYOL training. Collapse manifests as variance crashing toward zero. Catch it early.

### Pitfall 10: Using BYOL directly on text or other modalities without adaptation.

BYOL was designed for images. For text, BERT-style masking is dominant; BYOL-style methods exist (e.g., DeCLUTR for sentences) but require careful augmentation design (paraphrasing, masking) suitable for text.


---

# PART VI: Pretext Tasks — Hand-Designed Self-Supervision

## 1. Motivation & Intuition

### Pretext tasks: SSL before contrastive learning

Before contrastive learning swept the field around 2019-2020, self-supervised learning in vision was dominated by *pretext tasks*: hand-designed prediction problems where the supervisory signal comes from a known transformation of the input. The recipe:

1. Take an image.
2. Apply a known transformation `T` (rotation, masking, shuffling).
3. Train a model to recover the transformation parameters or to invert the transformation.
4. The model, in solving this surrogate problem, must learn to understand image structure.

These methods are *intuitive* and *human-designable* — you can articulate exactly what the model has to learn. They produced strong results in 2016-2018 (often beating fully-random initialization for downstream tasks) but were eventually surpassed by contrastive and masked modeling methods around 2019-2020.

Even though pretext tasks are no longer the state of the art, understanding them is important for two reasons: (a) they teach the principles of self-supervision in their cleanest form; (b) some pretext-task ideas (especially masked modeling, which we treat separately in Part VII) remain dominant.

### The intuition

Consider a child playing with a jigsaw puzzle. To assemble the puzzle, the child must understand which colors typically appear next to each other, which textures continue across edges, what shapes objects in the scene typically have. Solving the puzzle teaches general perception even though "puzzle solving" isn't a downstream goal.

Analogously: train a network to predict image rotations (0°, 90°, 180°, 270°). To classify rotation accurately, the network must learn what objects are upright. To learn what's upright, it must learn what objects look like. The pretext task forces semantic understanding *as a side effect*.

The art of pretext-task design is:
- The task should be **non-trivial** — not solvable by superficial cues.
- The task should require **semantic understanding** — solvable only by learning meaningful structure.
- The task should have **a clear inverse** — labels can be generated automatically.

### Real-world relevance

Pretext tasks shaped early SSL and inspired several enduring ideas:
- **Rotation prediction** showed that simple transformations could yield strong representations.
- **Inpainting / Context Encoders** anticipated MAE.
- **Colorization** anticipated diffusion / generative pretraining for downstream tasks.
- **Relative position prediction** (Doersch et al.) demonstrated that local spatial relationships carry enough information for global representation learning.

In domains where contrastive learning is hard (e.g., medical imaging where augmentations destroy signal), pretext tasks can still be the right choice.

---

## 2. Conceptual Foundations

### A taxonomy of pretext tasks

#### A. Context-based tasks
Learn from spatial relationships within an image.
- **Relative patch prediction**: given two patches, predict their relative position.
- **Jigsaw puzzles**: shuffle 9 patches, predict the permutation.
- **Inpainting / Context Encoders**: mask a region, reconstruct it.

#### B. Transformation-classification tasks
Apply a known transformation, predict the transformation parameter.
- **Rotation prediction**: rotate by 0°/90°/180°/270°, classify the angle.
- **Geometric transformation prediction**: more general transformations.

#### C. Generative / reconstruction tasks
Reconstruct the input or part of it from a corrupted version.
- **Colorization**: predict color from grayscale.
- **Splitbrain autoencoders**: reconstruct one half of the channels from the other.
- **Denoising autoencoders**: reconstruct clean from noisy.
- **MAE** (we treat this separately in Part VII).

#### D. Clustering-based tasks
Use clustering of features to assign pseudo-labels, then train classifier.
- **DeepCluster**: alternates k-means clustering and supervised training on the cluster assignments.
- **SwAV**: contrastive learning over cluster assignments rather than instances.

### The four canonical pretext tasks (per the topic list)

We focus on:
1. **Rotation prediction** (Gidaris et al., 2018).
2. **Jigsaw puzzles** (Noroozi & Favaro, 2016).
3. **Colorization** (Zhang et al., 2016).
4. **Relative patch prediction** (Doersch et al., 2015).

### Underlying assumptions

1. **The pretext task signal is correlated with semantic structure.** Solving the task requires features useful for downstream classification. If the task is solvable by a shortcut (e.g., chromatic aberration patterns near image edges in patch prediction), the model learns the shortcut, not semantics.
2. **The transformation is invertible / well-defined.** Rotation can be undone (and the angle is unambiguous); but if you applied a transformation that destroys content (e.g., very heavy noise), the pretext task is impossible.
3. **The encoder is the part that matters.** The downstream pipeline uses the encoder, not the classification head specific to the pretext task.

### What breaks

- **Shortcuts.** Patch prediction was famously beaten by chromatic aberration (relative patch position is easier to predict from per-patch color statistics than from semantic content). Doersch et al. had to add color jitter and other tricks to suppress shortcuts.
- **Insufficient task difficulty.** A pretext task that's too easy doesn't push the encoder to learn rich features.
- **Mismatch with downstream task.** Rotation prediction works well on natural images (which have a canonical orientation) but not on images that lack a canonical orientation (satellite imagery, microscopy).

---

## 3. Pretext Task #1: Rotation Prediction

### The idea

Take an image. Rotate it by one of {0°, 90°, 180°, 270°}. Train a 4-way classifier to predict the rotation angle.

### Why it works

To predict the rotation correctly, the model must recognize what objects are "upright". This requires:
- Understanding object semantics (cars have wheels at the bottom).
- Understanding scene structure (sky is at the top).
- Understanding texture orientations (tree bark is roughly vertical).

These are exactly the features useful for downstream classification.

### Mathematical formulation

For input image `x` and rotation angle `r ∈ {0°, 90°, 180°, 270°}`:
- Apply rotation: `x_r = R_r(x)`.
- Encoder: `h = f_θ(x_r)`.
- Classifier: `ŷ = c_φ(h)`, a 4-way softmax.
- Loss: cross-entropy between `ŷ` and the one-hot label of `r`.

```
ℒ_rot(θ, φ) = - 𝔼_{x ~ p(x), r ~ Uniform({0°, 90°, 180°, 270°})}  
              [ log P(r | x_r; θ, φ) ]
```

### Worked example

Image: a photo of a cat (right-side up).

Step 1: Sample rotation `r = 90°`.

Step 2: Rotate. The cat is now lying on its side.

Step 3: Encode `f_θ(rotated_cat) = h ∈ ℝ^{2048}`.

Step 4: Classify. The 4-way softmax outputs probabilities over {0°, 90°, 180°, 270°}.

Step 5: True label is `90°`. Cross-entropy loss penalizes if the model puts low probability on `90°`.

To do well, the encoder must recognize that the cat is lying on its side — which requires recognizing that a cat exists, and that cats are normally upright.

### Practical notes

- Trained on ImageNet without labels, achieves ~50-60% linear probe accuracy. Below contrastive methods (~70%+) but well above random init (~10%).
- Best for natural images with strong orientation priors. Fails on rotation-invariant data (microscopy, satellite, astronomy).
- Used as a strong baseline through 2019; less common today.

### Common pitfalls

- **Padding artifacts.** If rotation creates black borders, the model can use border patterns to predict rotation. Solution: crop after rotation to remove borders.
- **Center bias.** If images are always centered, rotation creates predictable corner patterns. Solution: random crops before rotation.
- **Symmetric objects.** A perfectly symmetric object (e.g., a centered ball) has no preferred orientation. The pretext task is impossible for these, providing no learning signal.

---

## 4. Pretext Task #2: Jigsaw Puzzles

### The idea

Divide the image into a 3×3 grid of patches. Shuffle the patches according to one of a fixed set of permutations. Train the network to predict which permutation was applied.

### Why it works

To solve a jigsaw puzzle, the model must understand:
- What shapes typically appear adjacent to each other.
- How edges should align.
- The global structure of the image.

This forces representations that capture spatial relationships and object structure.

### Implementation details

- The 9 patches are *permuted*, not just shuffled randomly. Why: a 9-patch random permutation has 9! = 362,880 possibilities — too many for a classification head. So a subset of "good" permutations is selected (typically 100-1000 with high mutual Hamming distance).
- The model is a Siamese-style network that processes each patch through a shared encoder, concatenates the features, and passes through a permutation classifier.
- Permutations are chosen to maximize the difficulty (i.e., far from identity) and discriminability.

### Mathematical formulation

Let `{p_1, ..., p_9}` be the 9 patches of an image. Let `σ ∈ Σ` be a permutation drawn from a fixed set of `|Σ|` permutations (e.g., 100). Apply: `(p_{σ(1)}, ..., p_{σ(9)})`.

Encode each: `h_i = f_θ(p_{σ(i)})`.

Concatenate: `H = [h_1; ...; h_9]`.

Predict permutation: `ŷ = c_φ(H)`, a `|Σ|`-way softmax over permutations.

Loss:
```
ℒ_jigsaw(θ, φ) = - 𝔼_{x, σ}  [ log P(σ | (p_{σ(1)}, ..., p_{σ(9)}); θ, φ) ]
```

### Worked example

Image: a photo of a dog standing in grass.

Step 1: Divide into 3×3 patches. Indices 1-9 in row-major order.

Step 2: Sample permutation `σ = (3, 1, 2, 6, 4, 5, 9, 7, 8)` (i.e., position 1 gets original patch 3, position 2 gets original patch 1, etc.).

Step 3: Encode each shuffled patch through shared CNN.

Step 4: Concatenate features and classify which permutation was used.

To do well: the model must recognize that, e.g., the dog's head should be near the body, not isolated; the grass should form a continuous region. This requires understanding the dog and the scene.

### Pitfalls

- **Chromatic aberration / lens distortion** can let the model identify patches' original positions from low-level visual cues. Solution: random crop within each patch (so patches don't have predictable boundaries).
- **Edge continuity shortcuts**. If patches are aligned, the model can match edge pixels. Solution: apply a small gap between patches or random spatial jitter.
- **Permutation set design**. Too few permutations → easy task. Too many → unfocused gradient. ~100 well-chosen permutations is a sweet spot.

### Comparison to other tasks

- Jigsaw is harder to scale than rotation (requires multi-patch processing).
- Performs comparably to rotation on linear probe (~50-55% on ImageNet).
- More sensitive to shortcuts than rotation.

---

## 5. Pretext Task #3: Colorization

### The idea

Convert the image to grayscale. Train a network to predict the original colors (or color components, like the `a` and `b` channels in `Lab` color space).

### Why it works

Predicting the color of a region requires knowing what's in the region:
- Sky is blue.
- Grass is green.
- Skin tones are within a narrow range.
- Bananas are yellow.
- Apples are red or green.

A model that predicts colors well must have semantic content built into its representation.

### Implementation details

- Convert image to `Lab` color space. Use `L` (lightness) as input. Predict `a` (red-green) and `b` (yellow-blue) channels.
- The output is per-pixel — colorization is a dense prediction task.
- Use a fully-convolutional encoder-decoder. The encoder can be repurposed for downstream classification.
- Loss can be regression (MSE on `a, b`) or classification (discretize the color space and predict a class per pixel — this works better empirically because color is multimodal: a car can be red, blue, or black, all valid).

### Mathematical formulation

Image in Lab: `x = (L, a, b)`. Input: `L`. Target: `(a, b)`.

Per-pixel classification approach:
- Discretize the `(a, b)` plane into `Q` color bins (e.g., 313 bins covering the visible color gamut).
- For each pixel, target is the bin containing the true `(a, b)`.
- Loss: pixel-wise cross-entropy.

```
ℒ_color(θ) = - Σ_{pixels (i, j)}  Σ_{q=1}^{Q}  𝟙[bin(a_{ij}, b_{ij}) = q]  ·  log P(q | L; θ)
```

Class-rebalancing weights are applied because most pixels have desaturated colors (skewing the loss); rare colors (saturated reds, blues) need upweighting to avoid trivial gray predictions.

### Worked example

Image: a photo of a banana on a wooden table.

Step 1: Convert to Lab. Discard `(a, b)`. Keep `L` (grayscale).

Step 2: Pass `L` through encoder-decoder.

Step 3: Predict per-pixel color distribution over 313 bins.

Step 4: For each pixel, the target bin is the true color of that pixel.

Step 5: Compute cross-entropy loss.

To do well: the model must know "this region is a banana → predict yellow"; "this region is wood → predict brown"; etc. The encoder learns to identify objects.

### Practical notes

- Linear probe accuracy on ImageNet: ~40-50%. Below other pretext tasks but historically important.
- Most useful when downstream tasks require dense prediction (segmentation, depth) — colorization pretrains a dense feature extractor naturally.
- Has resurfaced in various forms (e.g., generative models pretrained for many tasks).

### Pitfalls

- **Modal collapse to gray**. Without rebalancing, the loss is minimized by predicting the most common color (gray) for everything. Class rebalancing is critical.
- **Texture without structure**. The model can predict colors using local texture (e.g., sky-blue from cloud texture) without semantic understanding. Best results require diverse images with varied colorings of similar objects.
- **Lab vs RGB**. Lab separates lightness from chroma cleanly, making the task well-posed. Doing this in RGB couples lightness prediction with the supposedly missing color, complicating the loss.

---

## 6. Pretext Task #4: Relative Patch Prediction

### The idea

Sample two patches from an image: a center patch and a neighbor patch from one of 8 surrounding positions (N, NE, E, SE, S, SW, W, NW). Predict which of the 8 positions the neighbor came from.

### Why it works

To predict relative position, the model must understand:
- That objects span across patches in coherent ways.
- What semantic content lives "above", "below", "left", "right" of given content.
- The global structure of scenes.

A model that knows "if the center patch is a car wheel, the patch above is likely the car body" can predict the relative position. This requires semantic understanding.

### Implementation details

- Choose a center patch `p_c` and one neighbor patch `p_n` from 8 positions.
- A small gap separates patches to avoid chromatic continuity shortcuts.
- Both patches go through a shared CNN.
- Concatenate features and classify into 8 positions.

### Mathematical formulation

Image `x`. Sample center patch `p_c` and one of 8 neighbors `p_n^{(k)}` for `k ∈ {1, ..., 8}`.

Encode: `h_c = f_θ(p_c)`, `h_n = f_θ(p_n^{(k)})`.

Predict: `ŷ = c_φ(h_c, h_n)`, an 8-way softmax over positions.

Loss:
```
ℒ_relpos(θ, φ) = - 𝔼_{x, k}  [ log P(k | p_c, p_n^{(k)}; θ, φ) ]
```

### Worked example

Image: a photo of a dog in a park.

Step 1: Sample center patch covering the dog's body. Sample neighbor patch position `k = N` (north / above). The actual neighbor patch covers the dog's head.

Step 2: Encode both patches through shared CNN.

Step 3: Predict the relative position. The correct answer is `N`.

To predict correctly, the model must recognize that "the head is above the body" — implying recognition of dog parts. Other positions (e.g., the center is the body, neighbor is sky) require understanding scene structure.

### Pitfalls (the famous shortcut story)

The original Doersch et al. paper revealed a profound failure mode: **chromatic aberration**. Camera lenses bend light slightly differently for different wavelengths, producing per-pixel color shifts that vary smoothly across the image (e.g., the red channel may be shifted leftward relative to blue near the edges). This shift depends on absolute position in the image — so the model could predict relative position by detecting the chromatic aberration pattern at each patch, without any semantic understanding.

The fix: random color jitter, random color channel dropping, projecting to grayscale randomly. After these tricks, the model was forced to use semantic features.

This story has been retold many times to illustrate the danger of shortcuts in self-supervised learning. *Whatever cue solves the task most cheaply is what the model will learn.* If your pretext task can be solved by a degenerate shortcut, the encoder learns the shortcut, not the meaningful structure you intended.

### Practical notes

- Linear probe accuracy: ~50% on ImageNet (with shortcut suppression).
- Spawned a body of work on patch-based SSL.
- Less popular today, but the lessons (especially shortcut-suppression) carry over to all SSL methods.

---

## 7. Relevance to ML Practice (Pretext Tasks)

### When to use pretext tasks

- **You can't design good augmentations.** In domains where augmentations don't preserve semantics well (medical, scientific), a hand-designed pretext task can outperform contrastive methods.
- **Your downstream task aligns with the pretext.** If the downstream task is "predict object orientation in an X-ray", rotation prediction is a natural pretext.
- **You want interpretable pretraining.** Pretext tasks have a clear narrative — easier to explain to non-ML stakeholders.
- **You're operating with limited compute.** Pretext tasks are often cheaper than contrastive methods.

### When NOT to use pretext tasks

- **Modern benchmarks**: pretext tasks are dominated by contrastive and masked modeling on standard image benchmarks. Use those instead.
- **Risk of shortcuts**: any pretext task can be defeated by an unanticipated shortcut. Validate that the model isn't using degenerate cues.

### Trade-offs vs contrastive / masked modeling

| Pretext tasks | Contrastive | Masked modeling |
|---|---|---|
| Single classification head, simple loss | Big batches, careful augmentations | High masking ratio, asymmetric encoder |
| Risk of shortcuts | Risk of false negatives | Risk of focusing on low-frequency texture |
| Lower SOTA performance | Strong on classification | Strong on dense prediction |
| Easy to implement | Moderate complexity | Moderate complexity |

---

## 8. Common Pitfalls & Misconceptions

### Pitfall 1: Trusting that the pretext task captures semantics.

Always verify with a downstream evaluation (linear probe, kNN). The pretext loss can drop to zero with no semantic learning if the model exploits a shortcut.

### Pitfall 2: Combining many pretext tasks naively.

Adding "rotation + jigsaw + colorization" doesn't always help. Tasks can interfere — training signals can pull the encoder in incompatible directions. Best to pick one strong task or use a unified framework.

### Pitfall 3: Using rotation prediction on rotation-invariant data.

Satellite imagery, microscopy, abstract textures — all lack canonical orientation. Rotation prediction is then literally impossible (every angle is equally likely), so no useful gradient.

### Pitfall 4: Forgetting to suppress shortcuts.

Color jitter, random cropping, random grayscale — these tricks aren't decoration; they're necessary to force the model to use semantic features. Drop them and the model reverts to shortcuts.

### Pitfall 5: Using the pretext-task classifier head for downstream tasks.

The classifier head is task-specific. Throw it away. Use only the encoder.

### Pitfall 6: Treating "the pretext task is too hard / too easy" as binary.

Difficulty is a continuous knob. A pretext task that's too hard doesn't train (gradient is uninformative). Too easy doesn't push the encoder. Tune the difficulty (number of permutations, masking ratio, transformation set) for best representation quality.

### Pitfall 7: "More data should always help."

For pretext tasks, beyond a certain dataset size (usually ImageNet-scale or larger), returns diminish faster than for contrastive methods. Pretext tasks have lower ceilings.


---

# Part VII: Masked Modeling — BERT-Style Masking and Masked Autoencoders (MAE)

Masked modeling is the most successful self-supervised paradigm in modern AI. It powers BERT for language and MAE (and its descendants) for vision. The core idea is disarmingly simple: hide part of the input, then ask the model to predict what was hidden. Yet this simple recipe scales to billions of parameters and produces representations that transfer to virtually every downstream task in NLP and (increasingly) computer vision.

This part dissects two landmark methods — BERT (Devlin et al., 2018) and MAE (He et al., 2021) — and the design decisions that make them work.

## 1. Motivation & Intuition

### 1.1 The cloze test as a learning signal.

Educational psychologists have long used "cloze tests" — passages with words removed that the student must fill in — to measure comprehension. To fill in "The cat sat on the ___," you need to understand syntax (it's a noun), semantics (something a cat could sit on), and world knowledge (mat, chair, couch are common; helicopter is not). A test-taker who fills these in correctly demonstrates language understanding. The genius of BERT was to flip this around: if filling in blanks measures understanding, then *training* a model to fill in blanks should *create* understanding.

For images, the analog is hiding patches of a picture and asking the model to reconstruct them. To reconstruct a missing patch of a dog's eye, the model must know it's looking at a dog, where the eye should be, and what dog eyes look like — semantic, spatial, and visual knowledge.

### 1.2 Why this works (the deep reason).

The information-theoretic story: if a model can reconstruct missing information from observed information, it must have captured the joint distribution of observed and missing parts. Capturing the joint distribution is, in essence, understanding the data. Compare to autoregressive models (GPT-style), which predict the next token from previous tokens — also a form of masked prediction, just with a specific masking pattern (mask everything to the right). Masked modeling generalizes this to arbitrary masking patterns and bidirectional context.

### 1.3 The contrast with contrastive learning.

Contrastive methods like SimCLR/MoCo/BYOL learn by *comparison* — pulling positives together, pushing negatives apart (or, in BYOL, just predicting one view from another). This requires careful design of augmentations, large batches or queues, and is sensitive to the augmentation policy.

Masked modeling learns by *reconstruction* — fill in what's missing. This requires no augmentations, no negatives, no momentum encoders. Just masking and a reconstruction loss. The simplicity is a major reason why masked modeling has become dominant: it scales cleanly, has fewer hyperparameters, and is conceptually clean.

### 1.4 Real-world ML impact.

BERT (2018) revolutionized NLP. Within a year, every NLP benchmark was being topped by BERT or its variants (RoBERTa, ALBERT, ELECTRA, DeBERTa). The "pretrain on masked language modeling, then fine-tune" pipeline became the default. Modern LLMs (GPT-4, Claude, Llama) use causal language modeling rather than masked, but masked modeling remains dominant for encoder-style models used in retrieval, classification, and embedding tasks.

MAE (2021) and its successors brought the same paradigm to vision. MAE achieves higher downstream accuracy than contrastive methods like MoCo v3 and DINO on ImageNet fine-tuning while being simpler and more efficient to train. MAE-style pretraining underpins many modern vision foundation models.

### 1.5 Intuitive analogy: the redacted document.

Imagine you receive a document where 15% of the words have been blacked out, and you must guess them. To do this well, you must understand the language, the topic, the writer's style, and the surrounding context. Now imagine you receive an image where 75% of the patches have been blacked out, and you must reconstruct the image. To do this, you must understand what kind of scene it is, what objects are present, and how visual features are spatially arranged. The model that succeeds at this task has, by necessity, built rich representations of language or visual structure.

## 2. Conceptual Foundations

### 2.1 The masked modeling framework.

All masked modeling methods share a common structure:

1. **Masking.** A subset of the input units (tokens, patches, pixels) is selected for masking. Each masked unit is replaced with a special placeholder (e.g., a `[MASK]` token, or zeroed-out pixels, or a learnable mask embedding).

2. **Encoding.** A neural network (typically a Transformer) processes the corrupted input and produces representations for each unit.

3. **Decoding/Prediction.** A decoder (could be a single linear layer or a multi-layer Transformer) produces predictions for the masked units.

4. **Loss.** A reconstruction loss (cross-entropy for discrete tokens, MSE for continuous pixels) is computed on the masked positions only.

### 2.2 Key terminology.

| Term | Definition |
|------|------------|
| Token | A discrete input unit (a word, subword, or BPE piece in language; a patch in vision) |
| Masking ratio | Fraction of tokens that are masked |
| `[MASK]` token | A special learnable embedding used to indicate a masked position |
| Visible / unmasked tokens | Tokens that the model can see |
| Encoder | Network that produces contextualized representations from (possibly partially-masked) input |
| Decoder | Network that maps encoded representations to predictions for masked positions |
| MLM | Masked Language Modeling — BERT's training objective |
| MIM | Masked Image Modeling — vision analog (MAE, BEiT, SimMIM, iBOT) |
| Reconstruction target | What the model must predict (raw pixels, normalized pixels, discrete codes, features) |

### 2.3 BERT in detail.

BERT (Bidirectional Encoder Representations from Transformers) is a Transformer encoder pretrained with two objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Subsequent work showed NSP is largely useless, so we focus on MLM.

**MLM procedure:**
1. Take a sequence of tokens (e.g., "The cat sat on the mat").
2. Randomly select 15% of tokens.
3. For each selected token, with 80% probability replace it with `[MASK]`, with 10% probability replace it with a random token, and with 10% probability leave it unchanged.
4. Feed the corrupted sequence through the Transformer encoder.
5. For each selected position, predict the original token using a softmax over the vocabulary.

**Why the 80/10/10 split?** During pretraining, the model sees `[MASK]` tokens. During fine-tuning on downstream tasks, the input has no `[MASK]` tokens. This train/test mismatch could hurt the model's ability to encode unmasked tokens well. By sometimes leaving tokens unchanged or replacing with random tokens, the model learns to encode every token usefully (because it can never be sure which positions are "selected"). The 80/10/10 ratio was chosen empirically.

**Why bidirectional context?** Unlike GPT (which only sees tokens to the left), BERT sees the entire sequence. This bidirectional context is crucial for tasks like classification, NER, and question answering where understanding requires both leftward and rightward information.

### 2.4 MAE in detail.

MAE (Masked Autoencoders Are Scalable Vision Learners) adapts masked modeling to images with several key innovations:

**MAE procedure:**
1. Divide an image into non-overlapping patches (e.g., 14×14 patches of 16×16 pixels each = 196 patches for a 224×224 image).
2. Randomly mask 75% of patches (147 patches; only 49 visible).
3. Feed only the visible patches through a deep ViT encoder.
4. Concatenate the encoder output with learnable mask tokens at the masked positions, and add positional embeddings.
5. Feed this full sequence through a lightweight Transformer decoder.
6. Predict the pixel values of the masked patches using MSE loss (computed only on masked patches).
7. Discard the decoder; use only the encoder for downstream tasks.

**Key innovations:**
- **High masking ratio (75%).** Far higher than BERT's 15%. This is because images are highly redundant — adjacent patches share a lot of information. Low masking ratios produce trivially easy tasks.
- **Asymmetric encoder-decoder.** The encoder only sees visible patches (saving 4× compute on the encoder, which is the heavy network). The decoder is lightweight (~10% of encoder params) and processes the full sequence.
- **Pixel reconstruction as target.** Despite concerns that pixels are too "low-level," reconstructing pixels works well, especially with per-patch normalization.
- **No augmentations.** Unlike contrastive methods, MAE uses minimal augmentation (just random crop and flip).

### 2.5 Assumptions.

- **Local-to-global predictability.** Visible context contains enough information to (probabilistically) predict the masked content. If masked content is unpredictable from context (e.g., truly random noise), the loss is uninformative.
- **The reconstruction target is meaningful.** Pixels work for natural images; they wouldn't work for, say, randomly colored confetti where pixel values are arbitrary.
- **The masking strategy uncovers structure.** Random masking works well for both BERT and MAE; structured masking (block masking, span masking) works for some tasks.
- **The encoder has sufficient capacity.** Masked modeling benefits from large encoders (BERT-Large >> BERT-Base; ViT-Huge >> ViT-Base).

### 2.6 What breaks when assumptions fail.

- **Low redundancy data + high mask ratio.** If you mask 75% of an EHR record (electronic health record) where each field is independent, reconstruction is impossible. The 75% ratio is a feature of natural images' redundancy, not a universal recipe.
- **Inappropriate target.** Predicting raw RGB pixels for medical images (where intensity is calibrated to physical units) might lead the model to focus on calibration noise rather than anatomy. Using normalized pixels or learned features as target might work better.
- **Tiny models.** A small encoder cannot capture the joint distribution well; performance is poor.
- **No positional information.** Without positional embeddings, the model cannot place masked tokens correctly. MAE is acutely sensitive to this — masked positions need positional info.

### 2.7 BERT vs MAE: a comparison table.

| Aspect | BERT (text) | MAE (images) |
|--------|-------------|--------------|
| Input units | Tokens (subwords) | Patches |
| Mask ratio | 15% | 75% |
| Mask token use | 80/10/10 split (mask/random/keep) | Always mask |
| Encoder sees | Full sequence with `[MASK]` tokens | Only visible patches |
| Decoder | Same as encoder (just an output head) | Separate, lightweight Transformer |
| Reconstruction target | Token IDs (categorical) | Pixel values (continuous) |
| Loss | Cross-entropy | MSE (mean squared error) |
| Augmentation | None | Minimal (crop + flip) |
| Architecture | Transformer encoder | ViT (Vision Transformer) |

The differences arise from the differences between text and images: text is discrete and information-dense (each token carries semantic meaning), while images are continuous and redundant (adjacent patches share information).

## 3. Mathematical Formulation

### 3.1 BERT's MLM objective.

Let `x = (x_1, x_2, ..., x_N)` be a sequence of N token IDs. Let `M ⊂ {1, 2, ..., N}` be the set of masked positions (typically |M| ≈ 0.15·N). Let `x̃` be the corrupted sequence: `x̃_i = [MASK]` for i ∈ M (with probability 0.8), random token (probability 0.1), or `x_i` (probability 0.1).

The model `f_θ` maps the corrupted sequence to per-position logit vectors `f_θ(x̃) ∈ ℝ^(N × V)` where V is vocabulary size. The MLM loss is:

```
L_MLM(θ) = -E_{x, M} [ Σ_{i ∈ M} log p_θ(x_i | x̃) ]
```

where 

```
p_θ(x_i = v | x̃) = softmax(f_θ(x̃)_i)_v = exp(f_θ(x̃)_{i,v}) / Σ_{v'} exp(f_θ(x̃)_{i,v'})
```

The loss is a sum of cross-entropies over masked positions only (positions i ∉ M contribute zero). The expectation is over data and over random mask choices.

### 3.2 Connection to maximum likelihood.

MLM can be viewed as maximizing a *pseudo-likelihood* of the data. The true likelihood of a sequence under a joint distribution would be `p(x_1, ..., x_N)`. Pseudo-likelihood approximates this as `Π_i p(x_i | x_{-i})` where `x_{-i}` denotes all tokens except i. MLM trains to model `p(x_M | x_{-M})` for random subsets M, which is a generalization of pseudo-likelihood. As masking patterns become richer, MLM approaches modeling the full joint distribution (in the limit, modeling each unmasked subset gives complete distributional information).

### 3.3 MAE's reconstruction objective.

Let `x ∈ ℝ^(H × W × 3)` be an image, divided into N non-overlapping patches `x = (p_1, ..., p_N)`, where each patch `p_i ∈ ℝ^(P × P × 3)` is a flattened vector of dimension `P² · 3` (e.g., 16·16·3 = 768 for 16×16 patches).

A random subset `M ⊂ {1, ..., N}` of size `|M| = ρ · N` (where ρ = 0.75) is masked. Let `V = {1, ..., N} \ M` be the visible set.

**Encoder.** The encoder `f_E` takes only the visible patches plus their positional embeddings:

```
z_V = f_E({(p_i + pos_i) : i ∈ V})
```

producing token representations `z_V ∈ ℝ^(|V| × d)`.

**Mask tokens and decoder input.** A learnable mask token embedding `[m] ∈ ℝ^d` is repeated for each masked position. The decoder input is the concatenation of visible token representations and mask tokens, ordered by original position (with positional embeddings re-added):

```
z_full = arrange( {z_V} ∪ {[m] for i ∈ M}, order=position ) + positional_embeddings
```

**Decoder.** A lightweight Transformer decoder `f_D` processes this:

```
ẑ = f_D(z_full)
```

**Prediction head.** A linear layer maps decoder outputs to flattened pixel predictions:

```
p̂_i = W · ẑ_i + b   for i ∈ M
```

**Loss.** MSE on masked patches:

```
L_MAE(θ) = (1 / |M|) Σ_{i ∈ M} ||p̂_i - p_i||²
```

(Per-patch normalization: in practice, `p_i` is normalized to zero mean, unit variance per patch before computing the loss. This significantly improves results.)

### 3.4 Per-patch normalization explained.

Without normalization, the loss is dominated by patches with high variance (textured regions) and ignores patches with low variance (smooth regions). Per-patch normalization rescales each patch so all contribute comparably, focusing learning on relative patterns rather than absolute brightness. The normalization is:

```
p_i_normalized = (p_i - mean(p_i)) / sqrt(var(p_i) + ε)
```

The model predicts these normalized values, and at evaluation/visualization time, the predictions can be denormalized using the per-patch stats (which would not be available at inference for true generation, but for representation learning we don't generate at inference — we just use the encoder).

### 3.5 Why high masking works for images: the redundancy argument.

Mathematically: consider a Gaussian random field where adjacent patches are correlated with coefficient ρ ∈ (0, 1). The information about a masked patch contained in K nearest visible neighbors grows as `(1 - ρ^(2K))` (approximately). For natural images, ρ between adjacent patches is typically > 0.9, so even a single visible neighbor provides ~80% of the information needed. To make the task non-trivial, you need to mask enough so that masked patches are rarely adjacent to visible ones, requiring high masking ratios.

For text, tokens are much less redundant (each carries distinct meaning), so 15% masking already leaves enough information to make the task meaningful but not trivial.

### 3.6 Computational efficiency of MAE.

The encoder is the expensive part of a ViT (because of attention's `O(n²)` complexity in sequence length). MAE's asymmetric design feeds only `(1 - ρ) · N = 0.25 · N` patches to the encoder, reducing encoder compute by `(1 - ρ)² = 0.0625` (16× speedup for attention layers).

If `N = 196` patches and ρ = 0.75, the encoder sees 49 patches (instead of 196). Attention complexity drops from `O(196²) = O(38416)` to `O(49²) = O(2401)`, a ~16× reduction.

Concrete numbers: training MAE on ImageNet with ViT-Large takes ~64 hours on 64 TPUs at 75% masking, vs. estimated ~200 hours without the asymmetric design. The savings enable scaling to ViT-Huge (~635M parameters) where contrastive approaches struggle.

### 3.7 Decoder design.

The MAE decoder has two roles:

1. **Process visible context to predict masked content** — needs enough capacity to do nontrivial reconstruction.
2. **Be discarded after pretraining** — should be lightweight to not waste compute.

The MAE paper found that an 8-layer Transformer with 512-dim embeddings (vs encoder's 16-32 layers and 768-1280-dim) works well. Going lighter hurts representations; going heavier doesn't help much.

The decoder's depth matters because the encoder is now relieved of low-level reconstruction duties — those can be handled by the decoder. The encoder is free to focus on higher-level features. This is a subtle and important point: the asymmetric design implicitly *separates representation learning (encoder's job) from reconstruction (decoder's job)*.

### 3.8 The relationship to denoising autoencoders.

Vincent et al.'s denoising autoencoders (2008) trained networks to reconstruct clean inputs from corrupted versions. BERT and MAE are direct descendants: corruption is now structured (masking specific positions), the architecture is a Transformer, and the scale is much larger. The theoretical underpinning is the same: learning to denoise / reconstruct teaches the model the data manifold.

## 4. Worked Examples

### 4.1 Worked Example 1: BERT's MLM on a toy sentence.

Input sentence: "the quick brown fox jumps"

Tokenized (using a toy vocabulary): `[the, quick, brown, fox, jumps]` → token IDs `[101, 2451, 2829, 4419, 14523]`.

**Step 1: Sample mask.** Suppose 15% of 5 tokens = 0.75, so we mask one token. Pick position 2 (the word "brown") with the random selection.

**Step 2: Apply 80/10/10.** Roll a die (uniform [0,1]).
- If < 0.8: replace with `[MASK]` → corrupted = `[the, quick, [MASK], fox, jumps]` → IDs `[101, 2451, 103, 4419, 14523]`
- If 0.8–0.9: random token replacement
- If 0.9–1.0: leave unchanged

Suppose the die gave 0.5 → use `[MASK]`.

**Step 3: Forward pass.** The Transformer encoder processes the corrupted sequence. For each position, it produces a 768-dim hidden state (for BERT-Base). At position 2 (the masked one), the hidden state captures contextual information from "the quick ___ fox jumps."

**Step 4: Output projection.** A final linear layer (sometimes tied to the input embedding matrix) maps the 768-dim hidden state to a vector of size V (vocab size, ~30k for BERT). After softmax, this gives `p(token | context)` at position 2.

**Step 5: Loss.** Suppose the model assigns probability `p(brown | context) = 0.3`. The loss contribution from this position is `-log(0.3) ≈ 1.20`. Cross-entropy.

**Step 6: Backprop.** The loss at position 2 generates gradients that flow back through the entire network, updating all parameters to make "brown" more probable next time.

If the model has learned well, after training it should assign high probability to plausible fillers — "brown" (correct), "red," "small," "lazy" — and low probability to implausible ones — "the," "quickly," "Toronto."

### 4.2 Worked Example 2: MAE on a tiny image.

Consider a tiny 8×8 image divided into 4 patches of 4×4 (so N = 4 patches; ridiculously small, but illustrative).

**Patches** (each is 16 pixels grayscale, values 0–1):
- p_1 (top-left): mostly dark, mean = 0.2
- p_2 (top-right): bright, mean = 0.8
- p_3 (bottom-left): mid, mean = 0.5
- p_4 (bottom-right): mid, mean = 0.5

**Step 1: Mask.** 75% of 4 = 3. Mask patches 1, 3, 4. Visible = {p_2}.

**Step 2: Encode visible.** The encoder takes p_2 (with positional embedding pos_2) and produces a representation z_2 ∈ ℝ^d.

**Step 3: Decoder input.** Construct full sequence: `[m, z_2, m, m]` (mask token at positions 1, 3, 4 and z_2 at position 2). Add positional embeddings to all positions:
```
decoder_input = [m + pos_1, z_2 + pos_2, m + pos_3, m + pos_4]
```
Note: positional embeddings are re-added at the decoder; the encoder used them too, but the decoder needs them to know which mask token is which spatial position.

**Step 4: Decode.** Pass through the lightweight decoder Transformer. Output: `[ẑ_1, ẑ_2, ẑ_3, ẑ_4]` ∈ ℝ^(4 × d).

**Step 5: Prediction.** A linear layer maps each ẑ_i to 16-dim vector (4×4 pixels). Get predictions p̂_1, p̂_3, p̂_4.

**Step 6: Loss.** Compute MSE only on masked positions:
```
L = (1/3) [ ||p̂_1 - p_1||² + ||p̂_3 - p_3||² + ||p̂_4 - p_4||² ]
```

(In practice, with per-patch normalization: each p_i is first normalized to mean 0, std 1 within its 16 pixels, and the model predicts these normalized values.)

The gradient updates the encoder, decoder, and all embeddings to make better predictions next time. Notice: the loss does *not* include p_2 (the visible patch); the model gets no credit for "predicting" what it can already see.

### 4.3 Worked Example 3: Why per-patch normalization helps.

Consider two patches: a smooth patch with all pixels at value 0.5 (variance 0), and a textured patch with pixels ranging 0.0 to 1.0 (variance 0.08).

**Without normalization:** A baseline that always predicts the mean intensity would have:
- MSE on smooth patch: 0 (perfect)
- MSE on textured patch: ~0.08 (variance)
- Total loss dominated by textured patches.

To reduce loss, the model focuses on getting textured regions roughly right. Smooth regions are "free."

**With normalization:** Both patches have mean 0, variance 1 after normalization.
- Smooth patch: still flat (mean 0), predicting anything other than 0 gives MSE = prediction². Trivially perfect = predict 0.
- Textured patch: must predict the actual normalized pattern, MSE = ||prediction - true_pattern||².

Wait, the smooth patch becomes degenerate after normalization (var = 0, divide by 0). The MAE paper handles this by adding ε to the variance. Effectively, the smooth patch becomes uninformative (predicting all zeros gives near-zero loss), and the model focuses on patches where there's actually signal.

The key effect of normalization is that the *relative pattern* within each patch becomes the prediction target, not the absolute brightness. This makes the model learn structural / textural / semantic content rather than overall illumination.

## 5. Relevance to Machine Learning Practice

### 5.1 Where masked modeling is used.

**Language (BERT and descendants):**
- BERT, RoBERTa: foundational pretraining.
- ALBERT, DistilBERT, ELECTRA: efficiency variants.
- T5: encoder-decoder, masked spans (different masking strategy).
- DeBERTa: improved positional encoding.
- Used as base encoders for: classification, NER, QA, retrieval (sentence embeddings), entailment.

**Vision (MAE and descendants):**
- MAE (He et al., 2021): the foundational work.
- BEiT (Bao et al., 2021): predicts discrete VQ-VAE codes instead of pixels.
- SimMIM (Xie et al., 2022): predicts pixels with simpler design.
- iBOT (Zhou et al., 2021): combines masked image modeling with self-distillation (like DINO).
- Used as base encoders for: classification, detection (DETR variants), segmentation, depth estimation.

**Multimodal:**
- BERT-style masking on image tokens + text tokens jointly: VL-BERT, ViLBERT, FLAVA.

**Audio and speech:**
- wav2vec 2.0, HuBERT: masked modeling of audio frames.

### 5.2 When to use masked modeling.

Use it when:
- You have abundant unlabeled data and limited labeled data.
- The data has strong sequential/spatial structure (text, images, audio).
- You can afford to train a Transformer at scale.
- You want a foundation model usable across many downstream tasks.

### 5.3 When NOT to use it.

- When data lacks redundancy/structure that masking can exploit. Tabular data with mostly independent columns isn't well-suited.
- When labeled data is plentiful and the task is narrow. Direct supervised training may suffice.
- When you cannot afford large models. Masked modeling shines at scale; small models often underperform contrastive or supervised approaches.
- When inference latency requires very small models. Better to distill from a masked-modeling pretrained model than to pretrain a tiny model directly.

### 5.4 Common alternatives.

- **Causal language modeling (GPT-style):** Predict next token given previous tokens. Better for generative tasks. Worse for tasks needing bidirectional context (though in-context learning has narrowed the gap).
- **Contrastive methods (SimCLR, MoCo, BYOL):** Better for tasks emphasizing instance-level discrimination (retrieval, ReID). MAE often wins for dense prediction tasks (segmentation) at scale.
- **Supervised pretraining:** If you have a large labeled dataset (ImageNet-21k), supervised pretraining is competitive. Masked modeling is more flexible and scales better.

### 5.5 Trade-offs.

- **Computational cost.** MAE's asymmetric design makes it cheaper than many alternatives, but Transformer training is still expensive. BERT-Base pretraining on Wikipedia + BookCorpus took several days on 16 TPUs (in 2018).
- **Sample efficiency.** Masked modeling typically requires more pretraining steps than contrastive methods, but each step is cheaper (no momentum encoder, no large batch needed).
- **Interpretability.** MAE produces visualizations of reconstruction, which can be inspected. The model's "guesses" for masked patches give intuition about what it has learned.
- **Robustness.** Models pretrained with masked modeling are often more robust to occlusion (since they're trained to handle missing patches).
- **Fine-tuning vs linear probing gap.** Masked-modeling encoders often have a large gap: fine-tuning works well, but linear probing (frozen features) underperforms contrastive methods. This is because masked modeling learns features that are useful when combined non-linearly through the network, but aren't as immediately linearly separable.

### 5.6 Typical hyperparameters (BERT-Base / MAE).

| Hyperparameter | BERT-Base | MAE (ViT-L) |
|----------------|-----------|-------------|
| Architecture | 12-layer Transformer, 768-dim, 12 heads | 24-layer ViT, 1024-dim, 16 heads |
| Parameters | 110M | 304M |
| Pretrain data | Wiki + BookCorpus (~3.3B tokens) | ImageNet-1k (1.28M images) |
| Mask ratio | 15% | 75% |
| Mask token | `[MASK]` (80%), random (10%), keep (10%) | Always `[m]` |
| Optimizer | Adam | AdamW |
| Learning rate | 1e-4, linear warmup + decay | 1.5e-4, cosine schedule |
| Batch size | 256 sequences (length 512) | 4096 images |
| Pretrain steps | 1M | ~800 epochs (~250k steps) |
| Hardware | 16 TPU v3 chips, ~4 days | 64 TPU v3 chips, ~64 hours |

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: Using BERT's mask ratio (15%) for images.

A common mistake when porting masked modeling to vision is to use BERT's 15%. This makes the task trivial — the model can interpolate from neighbors. Use 75% for images (per MAE), or experiment with 60–80% for new modalities based on redundancy.

### Pitfall 2: Adding `[MASK]` tokens to the encoder in MAE.

The MAE design feeds only visible patches to the encoder, with mask tokens appearing only in the decoder. A naive implementation might pass masked patches through the encoder too. This wastes compute and, more importantly, causes train/test mismatch (at inference, no mask tokens). Stick with the asymmetric design.

### Pitfall 3: Forgetting positional embeddings in the decoder.

Mask tokens are identical (same learnable embedding `[m]`). Without positional embeddings, the decoder cannot distinguish mask token at position 1 from mask token at position 5 — and would predict the same thing for both. Always add positional embeddings to the decoder input.

### Pitfall 4: Using the decoder for downstream tasks.

The MAE decoder is trained for one job: pixel reconstruction. It is not useful for representation. Throw it away after pretraining and use only the encoder. Same for BERT: the MLM head is for pretraining; for downstream tasks, you typically replace it with a task-specific head.

### Pitfall 5: Skipping per-patch normalization in MAE.

Without per-patch normalization, the loss is dominated by high-variance patches. Per-patch normalization in MAE improves linear probe accuracy by ~2–3% on ImageNet. It's nearly free and should always be included.

### Pitfall 6: Treating masked modeling as a generative model.

BERT and MAE are not good generative models. They're trained to fill in small fractions of input given context. They don't model the joint distribution well enough to sample realistic full inputs from scratch. For generation, use causal models (GPT) or diffusion models.

### Pitfall 7: Believing higher mask ratio is always better.

There's a sweet spot. MAE found 75% optimal for ImageNet. Going higher (e.g., 90%) makes the task too hard — context is too sparse to predict anything meaningful, gradients become noisy. Going lower (e.g., 25%) makes it too easy — interpolation suffices. Don't blindly tune up.

### Pitfall 8: Confusing MLM with autoregressive LM.

MLM (BERT) predicts masked tokens given bidirectional context. Autoregressive LM (GPT) predicts the next token given only previous tokens. They produce different representations and are good at different things. MLM is better for understanding (classification, NER); AR is better for generation.

### Pitfall 9: Underestimating the value of large encoders.

Masked modeling scales remarkably well with model size. Going from BERT-Base (110M) to BERT-Large (340M) gives substantial gains; further scaling continues to help (RoBERTa, DeBERTa, etc., have demonstrated this). MAE's results are best with ViT-Large or ViT-Huge. If you're seeing modest gains, consider scaling the encoder before tweaking other hyperparameters.

### Pitfall 10: Over-relying on default reconstruction targets.

Pixels (MAE) and tokens (BERT) are popular but not the only choices. BEiT predicts discrete VQ-VAE codes; iBOT predicts features from a teacher network; data2vec predicts contextualized representations. The choice of target matters. For specialized domains (e.g., medical imaging), think about what target captures the structure you care about.


---

# Part VIII: Applications — Pretraining for Downstream Tasks and Representation Quality Evaluation

The point of self-supervised learning is *not* the pretext task itself — it's the representation produced by the encoder. This part covers how to use SSL representations in downstream tasks and how to measure their quality.

## 1. Motivation & Intuition

### 1.1 The two-stage pipeline.

Modern ML often follows a two-stage pipeline:

1. **Pretrain** an encoder on a large unlabeled corpus using a self-supervised objective.
2. **Adapt** the encoder to a specific downstream task using (typically smaller) labeled data.

The first stage is expensive but reusable. The second stage is cheap and task-specific. This is analogous to how humans first develop general perception/language abilities, then learn specialized skills (medicine, law, programming) on top of that foundation. The pretrained encoder is the "general intelligence"; the adapted model is the "specialist."

### 1.2 Why this works.

The pretraining task forces the encoder to learn rich, general-purpose features. These features capture broad regularities of the data (visual structure, linguistic patterns) that are useful for many downstream tasks, not just the pretext task.

When labeled data is scarce (medical images, legal documents, niche languages), training from scratch produces poor results — there isn't enough signal to learn good features. Pretraining gives the model a head start: it already knows what edges, textures, words, and syntactic structures look like, so the small labeled dataset only needs to teach the task-specific decisions.

### 1.3 Real-world impact.

This pipeline has become the dominant approach in modern AI. BERT-style pretraining + fine-tuning replaced custom architectures for nearly every NLP task by 2020. SSL pretraining + fine-tuning is increasingly dominant in computer vision, especially for medical imaging, satellite imagery, and other specialized domains where labels are expensive.

The concrete impact: on a downstream task with 1,000 labeled examples, fine-tuning a pretrained encoder typically beats training from scratch by 10–30 percentage points of accuracy. With 100 labeled examples, the gap can exceed 40 points.

## 2. Conceptual Foundations

### 2.1 Adaptation strategies.

There are several ways to adapt a pretrained encoder to a downstream task, ordered roughly by computational cost and flexibility:

1. **Linear probing.** Freeze the encoder. Add a single linear layer (logistic regression) on top of the frozen features. Train only the linear layer. Cheapest; tests linear separability of the features.

2. **Multilayer probe.** Like linear probing but with a small MLP head. Tests whether features are nonlinearly useful.

3. **kNN probing.** No training at all. For each test point, find k nearest neighbors in the training set (using encoder features), and predict the majority label. Tests whether semantically similar inputs are close in feature space.

4. **Fine-tuning.** Unfreeze the encoder. Train end-to-end on the downstream task with a small learning rate for the encoder and a larger one for the head. Most flexible; usually highest accuracy when labels are sufficient.

5. **Partial fine-tuning.** Unfreeze only the last few layers (or use techniques like LoRA, adapters). Compromise between linear probing and full fine-tuning.

6. **Prompting / In-context learning.** Don't update parameters at all. Frame the task as a continuation of the pretraining objective. Common for large language models.

### 2.2 Representation quality metrics.

How do you know if SSL is working? You can't just look at the pretext loss — a low loss might mean the model exploited a shortcut. Real evaluation requires measuring how useful the representations are.

Common evaluation protocols:

- **ImageNet linear probe.** Pretrain SSL on ImageNet (no labels). Train a linear classifier on top of frozen features using ImageNet labels. Report top-1 accuracy. The de facto standard for SSL methods.
- **Fine-tuning on downstream tasks.** Pretrain on a large corpus, then fine-tune on smaller datasets (CIFAR-100, Pascal VOC, COCO). Compare to from-scratch training and to other pretrained models.
- **Few-shot / low-shot evaluation.** Fine-tune with very few labels (e.g., 1%, 10% of full labels). Tests whether the representation is good enough to learn from minimal supervision.
- **Transfer to new tasks.** Object detection, segmentation, depth estimation — tasks structurally different from the pretraining task. Tests true generality.
- **Geometric properties.** Alignment (positives close), uniformity (representations spread out), feature rank (effective dimensionality) — direct measurements of feature space.

### 2.3 The linear probe vs fine-tuning gap.

Different SSL methods have different trade-offs between linear probe accuracy and fine-tuning accuracy:

- Contrastive methods (SimCLR, MoCo): high linear probe (features are immediately linearly separable), moderate fine-tuning.
- Masked modeling (MAE): lower linear probe, very high fine-tuning. The features need the network's nonlinear processing to be useful.
- Non-contrastive (BYOL, DINO): in between.

Why? Contrastive losses explicitly shape the feature space to separate semantic classes — this directly produces linear separability. Masked modeling produces representations optimized for reconstruction, which encode information broadly but not necessarily linearly.

### 2.4 Domain-specific pretraining.

When the downstream domain differs significantly from the pretraining domain, pretraining on in-domain unlabeled data often beats pretraining on a generic large corpus. Examples:

- **Medical imaging.** Pretraining on ImageNet then fine-tuning on chest X-rays often underperforms pretraining on a large corpus of medical images.
- **Code.** Pretraining BERT on text then fine-tuning on code is worse than pretraining on code (e.g., CodeBERT, GraphCodeBERT).
- **Scientific text.** PubMedBERT (pretrained on PubMed abstracts) outperforms general BERT on biomedical NLP tasks.

The lesson: in-domain unlabeled data, even if smaller than a generic corpus, often gives better representations for in-domain tasks.

## 3. Mathematical Formulation

### 3.1 Linear probing.

Given a pretrained encoder `f_θ : X → ℝ^d` and a downstream classification dataset `D = {(x_i, y_i)}_{i=1}^N` with labels `y_i ∈ {1, ..., C}`:

Train a linear classifier `g : ℝ^d → ℝ^C` defined by `g(z) = Wz + b` where `W ∈ ℝ^(C × d)`, `b ∈ ℝ^C`. The loss is:

```
L_probe(W, b) = (1/N) Σ_{i} CE(softmax(W f_θ(x_i) + b), y_i)
```

with θ frozen. Only W, b are updated. Typical setup: train for ~100 epochs with SGD, learning rate ~0.1 (linear models tolerate higher LRs than full networks), weight decay 0.

### 3.2 Fine-tuning.

Same dataset, but now train both the encoder and a head:

```
L_finetune(θ, W, b) = (1/N) Σ_{i} CE(softmax(W f_θ(x_i) + b), y_i)
```

with θ unfrozen. Typical setup: smaller LR for encoder (~1e-4), larger for head (~1e-3); shorter schedule (~50 epochs); strong augmentation; small weight decay.

### 3.3 kNN classifier.

Compute features `z_i = f_θ(x_i)` for all training examples. For a test point `x*`, compute `z* = f_θ(x*)`, find k nearest neighbors `{(z_{i_1}, y_{i_1}), ..., (z_{i_k}, y_{i_k})}` under cosine similarity, and predict:

```
ŷ* = argmax_c Σ_{j=1}^k 1[y_{i_j} = c]
```

(Or weighted by similarity: `Σ_j cos(z*, z_{i_j}) · 1[y_{i_j} = c]`.)

### 3.4 Alignment and uniformity.

From Wang & Isola (2020), used to characterize contrastive feature spaces:

```
L_align = E_{(x, x+)} [ ||f(x) - f(x+)||² ]   (assuming unit-norm features)
L_uniform = log E_{x, y i.i.d.} [ exp(-t · ||f(x) - f(y)||²) ]
```

Lower L_align means positives are closer; lower L_uniform means features are more uniformly distributed on the hypersphere. Good representations have both low.

### 3.5 The effective rank.

For a feature matrix `Z ∈ ℝ^(N × d)` (rows are features for N examples), compute SVD: `Z = UΣV^T`. The effective rank is:

```
EffectiveRank = exp(H(σ))
```

where `H(σ) = -Σ_i (σ_i / Σ_j σ_j) log(σ_i / Σ_j σ_j)` is the entropy of the normalized singular values. A high effective rank means features span the space well; a low effective rank means features are concentrated in a low-dimensional subspace (sign of partial collapse).

### 3.6 Quantifying transfer with the relative error.

To compare pretrained vs from-scratch, define:

```
RelativeError = (Error_scratch - Error_pretrained) / Error_scratch
```

A relative error of 0.5 means pretraining halves the error rate. This metric scales out task difficulty and makes comparisons across tasks meaningful.

## 4. Worked Examples

### 4.1 Worked Example: Evaluating a SimCLR-pretrained encoder.

Suppose we've pretrained an encoder on ImageNet using SimCLR. Total training: 1000 epochs on unlabeled ImageNet, batch size 4096, LARS optimizer.

**Linear probe protocol:**
1. Freeze the encoder. Discard the projection head.
2. Add a linear layer mapping 2048-dim features to 1000 ImageNet classes.
3. Train for 90 epochs on ImageNet labels with SGD, LR=0.1 (cosine schedule, no warmup needed for linear), weight decay=0, momentum=0.9, batch size=4096.
4. Augmentation: only random crop + horizontal flip (much weaker than during pretraining).
5. Evaluate top-1 accuracy on ImageNet val set.

Expected result: ~70% top-1 accuracy (depending on encoder size, training length).

**Fine-tuning protocol:**
1. Unfreeze the encoder. Initialize linear head from scratch.
2. Train for 100 epochs on ImageNet labels with AdamW, encoder LR=1e-4, head LR=1e-3, weight decay=0.05.
3. Augmentation: standard supervised training augmentation (RandAugment, mixup, cutmix).
4. Expected: ~76% top-1 accuracy (better than from-scratch by 2–4 points).

**kNN protocol:**
1. Compute features for all 1.28M training images (frozen encoder).
2. For each val image, find 200 nearest neighbors by cosine similarity.
3. Predict majority vote weighted by similarity: ŷ = argmax_c Σ_j cos(z, z_j) · 1[y_j = c].
4. Expected: ~67% top-1 (slightly below linear probe).

**Few-shot:** Take 1% of training labels (~12k images, ~12 per class). Fine-tune a linear head on these. Expected: ~52% (vs. ~25% from-scratch with 1%).

### 4.2 Worked Example: Diagnosing collapse via uniformity.

Pretrain BYOL. Periodically (every epoch), compute uniformity loss on a held-out set.

**Healthy training:**
- Epoch 0: L_uniform ≈ 0 (random init produces near-uniform features)
- Epoch 50: L_uniform ≈ -3.0 (features become more concentrated as semantic structure forms; this is normal)
- Epoch 100: L_uniform ≈ -3.5 (stable)
- Epoch 200: L_uniform ≈ -3.5 (stable, converged)

**Collapsed training:**
- Epoch 0: L_uniform ≈ 0
- Epoch 5: L_uniform ≈ -10 (sharply decreasing — features rapidly concentrating)
- Epoch 10: L_uniform → -∞ (all features identical)

If you see L_uniform plummet to very large negative values quickly, you have collapse. Diagnose by checking your stop-gradient placement, learning rates (too high?), and predictor architecture.

### 4.3 Worked Example: Domain-specific pretraining for a medical task.

Task: classify chest X-rays for 14 pathologies (NIH ChestX-ray14 dataset).

**Approach 1: From scratch.** Train a ResNet-50 from random init on the 112k labeled chest X-rays. Result: ~78% mean AUC.

**Approach 2: ImageNet supervised pretraining.** Initialize from ResNet-50 trained on ImageNet labels. Fine-tune on chest X-rays. Result: ~82% mean AUC. (Pretraining helps despite domain gap.)

**Approach 3: SSL pretraining on ImageNet.** Use SimCLR weights from ImageNet (no labels). Fine-tune on chest X-rays. Result: ~83% mean AUC.

**Approach 4: SSL pretraining on chest X-rays.** Run SimCLR on a large unlabeled chest X-ray corpus (e.g., MIMIC-CXR, ~377k images, no labels). Then fine-tune on the 112k labeled set. Result: ~85% mean AUC.

**Approach 5: SSL on X-rays + fine-tune.** Combination: pretrain SSL on a large medical imaging corpus, then fine-tune on the small labeled task. Best of both worlds.

The key insight: when the unlabeled in-domain corpus is large enough, in-domain SSL pretraining beats generic ImageNet pretraining (whether supervised or SSL). When in-domain unlabeled data is small, ImageNet pretraining still helps as a starting point.

## 5. Relevance to Machine Learning Practice

### 5.1 Industry workflows.

- **Web companies (Google, Meta, etc.):** Pretrain massive foundation models (BERT, ViT, MAE) on web-scale data. Distill or fine-tune for specific products (search ranking, content moderation, recommendation).
- **Medical AI startups:** Pretrain on hospital partner data (no patient labels needed for pretraining), fine-tune on small expert-annotated subsets.
- **E-commerce:** Pretrain image encoders on product photos, fine-tune for visual search, recommendation, attribute extraction.
- **Robotics:** Pretrain on large video datasets, fine-tune for manipulation tasks with limited labeled trajectories.

### 5.2 When to use which evaluation method.

- **Linear probe:** When comparing SSL methods. Standard benchmark. Shows immediate feature quality.
- **kNN:** Sanity check during pretraining (no extra training needed). Quick diagnostics.
- **Fine-tuning:** When deploying a model. Reflects actual downstream performance.
- **Few-shot:** When labels are expensive. Tests the most important real-world property.
- **Transfer to multiple tasks:** For evaluating foundation models / general-purpose pretraining.

### 5.3 Trade-offs in adaptation choice.

| Method | Compute | Storage | Performance | When to use |
|--------|---------|---------|-------------|-------------|
| Linear probe | Very low | One W per task | Modest | Many tasks, quick deployment |
| kNN | Zero training | Stored features for entire training set | Modest | Zero-training prototyping |
| Fine-tuning | High | Full model copy per task | High | Few tasks, max performance |
| LoRA / adapters | Medium | Small per task (~MBs) | High (close to fine-tune) | Many tasks, parameter-efficient |
| Prompting | Zero | Zero per task | Variable | LLMs with strong base capabilities |

### 5.4 Monitoring SSL training.

A common mistake is to monitor only the pretext loss. Better practice:

1. Track pretext loss (decreasing → training is at least functioning).
2. Periodically (every N epochs) compute kNN accuracy on a held-out labeled set. If it's increasing, representations are improving.
3. Monitor feature uniformity (avoid collapse).
4. Monitor gradient norms and weight norms (avoid divergence).
5. Periodically run a quick linear probe (every ~50 epochs) for stronger evaluation.

If pretext loss is decreasing but kNN accuracy isn't, you're learning a shortcut. Investigate.

## 6. Common Pitfalls & Misconceptions

### Pitfall 1: Using the projection head for downstream tasks.

The projection head (in SimCLR, MoCo, BYOL) is for the contrastive objective. It removes information that might be useful downstream. Always use the encoder output (before the projection head) for downstream tasks.

### Pitfall 2: Comparing methods using different evaluation protocols.

Method A: linear probe with 90 epochs, LR 0.1, no augmentation.
Method B: linear probe with 30 epochs, LR 0.01, with augmentation.
These are not comparable. Use the standard protocol (e.g., as defined in SimCLR or MoCo papers) for fair comparison.

### Pitfall 3: Believing pretraining always helps.

For very large labeled datasets (e.g., 100M labeled images), pretraining adds little — the supervised signal alone is strong enough. Pretraining's value is greatest for small labeled datasets.

### Pitfall 4: Overlooking the encoder size.

Pretrained encoder size matters. A SimCLR-pretrained ResNet-18 is often worse than a supervised ResNet-50 for downstream tasks. Compare apples to apples (same architecture).

### Pitfall 5: Catastrophic forgetting during fine-tuning.

Aggressive fine-tuning (high LR, long schedule) can erase pretrained features, especially in earlier layers. Use lower LR for early layers, layer-wise LR decay, or freeze early layers initially.

### Pitfall 6: Mismatched normalization.

If pretrained features were computed with one normalization (e.g., ImageNet mean/std), and downstream uses a different one (e.g., dataset-specific), the encoder sees out-of-distribution inputs. Match the input pipeline to the pretraining pipeline.

### Pitfall 7: Ignoring the train/test gap during pretraining.

If you pretrain at resolution 224 but fine-tune at 384, you might see performance drops or surprising behaviors. Match resolution, or use methods designed for resolution flexibility (FixRes, EfficientNet).

### Pitfall 8: Using kNN with un-normalized features.

kNN with cosine similarity requires comparing unit-norm features. Always L2-normalize before computing cosine. With L2-normalized features, dot product = cosine similarity.


---

# Part IX: Interview Preparation

This section provides a comprehensive set of interview questions across all topics covered in this guide, organized by difficulty and type. Answers are written at the level expected of a candidate for a machine learning or applied scientist role.

---

## Foundational Questions

### Q1: What is self-supervised learning, and how does it differ from supervised and unsupervised learning?

Self-supervised learning (SSL) is a paradigm where the supervisory signal is derived from the data itself, without human-provided labels. The model is trained on a "pretext task" — a task whose labels are constructed automatically from the inputs (e.g., predicting the next word, filling in a masked patch, distinguishing different views of the same image).

The distinction:
- **Supervised learning** uses human-provided labels `(x, y)` and trains to predict y from x.
- **Unsupervised learning** has no labels and aims to discover structure (clustering, density estimation, dimensionality reduction).
- **Self-supervised learning** has no human labels but constructs labels from the input itself, then uses supervised techniques on the constructed labels. SSL sits inside unsupervised learning historically but is distinguished by its supervised-style training mechanics.

The key practical advantage: SSL can leverage virtually unlimited unlabeled data, which is abundant on the web, in medical archives, etc., while supervised learning is bottlenecked by the cost of labeling.

### Q2: What is contrastive learning, in one sentence?

Contrastive learning trains a representation by pulling together representations of "positive pairs" (typically two augmented views of the same input) and pushing apart representations of "negative pairs" (different inputs).

### Q3: What's the difference between SimCLR, MoCo, and BYOL?

All three learn representations by comparing augmented views of images, but differ in how they manage negatives:

- **SimCLR** uses the other elements of the same large batch as negatives. Requires very large batches (typically 4096+).
- **MoCo** maintains a queue of past representations as negatives, decoupling negative count from batch size. Uses a momentum-updated encoder for the negatives to keep them consistent across recent steps.
- **BYOL** uses no negatives at all. It uses an asymmetric architecture: a target network (EMA of online network) produces target embeddings, and an online network with a predictor learns to predict them. A stop-gradient on the target prevents collapse.

Empirically, all three achieve similar quality on ImageNet linear probe (~70%+); they differ in computational and memory profiles, and in robustness to design choices.

### Q4: What is a pretext task? Give three examples.

A pretext task is an artificial supervised task constructed from unlabeled data, designed to force the model to learn useful representations as a byproduct. Examples:

- **Rotation prediction:** Rotate an image by 0°, 90°, 180°, or 270°, train the model to predict which rotation was applied. To predict, the model must recognize canonical orientations of objects.
- **Jigsaw puzzles:** Split an image into 9 patches, shuffle, train the model to predict the permutation. Forces understanding of spatial relationships.
- **Colorization:** Given a grayscale image, predict the color. Forces semantic understanding (skies are blue, grass is green) since color is correlated with object identity.

Pretext tasks have largely been superseded by contrastive learning and masked modeling for general-purpose pretraining, but remain useful for specific scenarios.

### Q5: Why does masked modeling use such different mask ratios for text (15%) vs images (75%)?

Because text is information-dense (each token carries distinct meaning) while images are information-redundant (adjacent patches share much information). 

For text, masking 15% leaves enough remaining tokens to make the task well-posed (filling in a missing word from rich surrounding context) and challenging (the model can't trivially guess from immediate neighbors).

For images at 15% masking, adjacent patches almost always remain visible, making the masked patch easy to interpolate. The model would not need to learn semantic features. At 75%, masked patches are usually surrounded by other masked patches, forcing the model to use global, semantic context to make predictions.

### Q6: What is the purpose of the projection head in SimCLR?

The projection head (a small MLP) sits on top of the encoder during training, and the contrastive loss is applied to its output. After training, the projection head is discarded and only the encoder is used for downstream tasks.

Its role: the contrastive loss encourages the projection head to be invariant to augmentations (color jitter, cropping, etc.). If the loss were applied directly to encoder features, the encoder would be forced to also be invariant, throwing away color and spatial information. The projection head absorbs this invariance, freeing the encoder to retain richer features. Empirically, removing the projection head drops downstream accuracy by ~10 percentage points.

### Q7: Why does BYOL not collapse despite having no negatives?

BYOL uses three mechanisms to prevent collapse:

1. **Asymmetric architecture:** The online network has a predictor on top; the target network does not. The predictor is forced to predict the target, but the target doesn't try to predict the online output.
2. **Target is an EMA of the online network:** This creates a moving target — the target slowly tracks the online network. The predictor must constantly chase a moving target, which provides a learning signal even without negatives.
3. **Stop-gradient on the target:** Gradients flow only through the online network. This breaks the symmetry that would otherwise lead the system to a trivial fixed point.

A constant predictor (e.g., always output zero) would minimize the loss only if the target output were also zero. But the target is not trained to output zero — it slowly tracks the online network, which is being pushed away from constant outputs by the stop-gradient/predictor dynamics. This creates a stable equilibrium where features are nontrivial.

### Q8: What is "linear probing"?

Linear probing is an evaluation protocol: freeze a pretrained encoder, train a single linear layer (logistic regression) on top of its features for a downstream classification task, and measure accuracy. It tests whether features are *linearly separable* into semantic classes — a strong indicator of representation quality. It's the de facto standard for comparing SSL methods on ImageNet.

### Q9: What does "momentum encoder" mean in MoCo?

In MoCo, there are two encoders: a query encoder (regular, updated by gradient descent) and a key encoder (momentum, updated by EMA of the query encoder). The momentum update is `θ_k ← m·θ_k + (1-m)·θ_q` with m typically 0.999.

The momentum encoder produces consistent representations for negatives in the queue (which were computed at slightly older parameter values). Without momentum (i.e., copying weights every step), representations in the queue would be inconsistent across recent steps, hurting learning. Momentum balances the need for slowly-evolving negatives (consistent) with eventual updating (still tracking the query encoder over time).

### Q10: What is MAE, and what makes it efficient?

Masked Autoencoders (MAE) are a vision SSL method that:
1. Splits an image into patches.
2. Masks 75% of patches randomly.
3. Encodes only the visible 25% with a deep ViT encoder.
4. Reconstructs the masked patches using a lightweight decoder.
5. Computes MSE loss only on masked patches.

Its efficiency comes from the asymmetric design: the heavy encoder processes only 25% of patches, reducing attention complexity by 16× (since attention is quadratic in sequence length). The decoder is lightweight and discarded after pretraining.

---

## Mathematical Questions

### Q11: Derive the gradient of NT-Xent loss with respect to a positive pair.

Let `z_i` and `z_j` be a positive pair (both unit-normalized; we'll write `s_{ij} = z_i · z_j / τ`). The NT-Xent loss for example i is:

```
L_i = -log(exp(s_{ij}) / Σ_{k ≠ i} exp(s_{ik}))
    = -s_{ij} + log Σ_{k ≠ i} exp(s_{ik})
```

(where the sum runs over all other elements in the batch including j).

Take the gradient with respect to `z_i`:

```
∂L_i / ∂z_i = -∂s_{ij}/∂z_i + Σ_{k ≠ i} (exp(s_{ik}) / Z) · ∂s_{ik}/∂z_i
```

where `Z = Σ_{k ≠ i} exp(s_{ik})`.

Each `∂s_{ik}/∂z_i = z_k / τ` (gradient of dot product is the other vector, divided by temperature; ignoring the unit-norm constraint for clarity, which adds a projection term in the actual implementation).

So:
```
∂L_i / ∂z_i = -(z_j / τ) + (1/τ) Σ_{k ≠ i} p_{ik} z_k
            = (1/τ) [ -z_j + Σ_{k ≠ i} p_{ik} z_k ]
            = (1/τ) [ (p_{ij} - 1) z_j + Σ_{k ∉ {i,j}} p_{ik} z_k ]
```

where `p_{ik} = exp(s_{ik}) / Z` is the softmax probability over candidate matches.

**Interpretation:**
- The term `(p_{ij} - 1) z_j`: since `p_{ij} < 1`, this is negative-coefficient z_j — pulling `z_i` toward `z_j` (the positive). The pull is strongest when `p_{ij}` is small (the positive isn't yet ranked well).
- The terms `p_{ik} z_k` for negatives: positive-coefficient — pushing `z_i` away from each negative, weighted by how much "weight" the softmax assigns to that negative. Hard negatives (high `p_{ik}`) get pushed harder. This is the implicit hard-negative mining property.

### Q12: Show that InfoNCE provides a lower bound on mutual information.

Setup: We have two views `v_1, v_2` of the same data drawn from joint `p(v_1, v_2)`. Let `f(v_1, v_2)` be a learned scoring function (often `exp(z_1·z_2/τ)`). Given a positive pair `(v_1, v_2^+)` and K-1 negatives `v_2^- ~ p(v_2)` (marginal), the InfoNCE loss is:

```
L_InfoNCE = -E [ log( f(v_1, v_2^+) / (f(v_1, v_2^+) + Σ_{k=1}^{K-1} f(v_1, v_2^{-,k})) ) ]
          = -E [ log( f(v_1, v_2^+) / Σ_{k=1}^K f(v_1, v_2^{(k)}) ) ]
```

(Treating the positive as one of K total candidates, indexed.)

The optimal `f*(v_1, v_2)` minimizing this loss is proportional to `p(v_2 | v_1) / p(v_2)` — the ratio of conditional to marginal density.

Plugging in `f* = p(v_2 | v_1) / p(v_2)`:

```
L_InfoNCE = -E[ log( (p(v_2^+ | v_1) / p(v_2^+)) / Σ_k (p(v_2^{(k)} | v_1) / p(v_2^{(k)})) ) ]
```

Now, `Σ_k (p(v_2^{(k)} | v_1) / p(v_2^{(k)}))` is approximately `K · E_{v ~ p(v_2)}[ p(v | v_1) / p(v) ]`. The expectation under the marginal of the density ratio is 1 (since `E[p(v|v_1)/p(v)] = ∫ p(v|v_1) dv = 1`).

So `L_InfoNCE ≈ -E[ log(p(v_2|v_1)/p(v_2)) ] + log(K) = -I(v_1; v_2) + log(K)`.

Therefore:
```
I(v_1; v_2) ≥ log(K) - L_InfoNCE
```

Maximizing the negative loss (= minimizing the loss) maximizes a lower bound on mutual information. The bound improves with K (more negatives = tighter bound).

Note: in practice, the bound is loose, and the connection between MI maximization and downstream utility is debated. Tschannen et al. (2019) showed that better MI bounds don't necessarily yield better representations.

### Q13: Explain the role of temperature τ in NT-Xent and its effect on gradients.

Temperature τ scales the similarities before softmax. Smaller τ means sharper softmax (more concentrated on the highest similarity); larger τ means softer (more uniform).

**Effect on gradients:**
- Small τ: the softmax assigns most weight to a few hardest negatives. Gradient magnitudes scale as 1/τ — gradients are larger overall. Hard-negative mining is aggressive.
- Large τ: softmax is more uniform; many negatives contribute. Gradients are smaller in magnitude. Less aggressive hard-negative emphasis.

Typical value: τ = 0.5 in SimCLR (with cosine similarity, range [-1, 1], so similarities post-temperature range [-2, 2]). MoCo uses τ = 0.07 (with similarities in same range, post-temperature [-14, 14] — effectively much sharper). The differences arise because MoCo has many more negatives (queue), so it can afford sharper softmax without numerical issues.

Both too-small and too-large temperatures hurt: too small leads to instability and over-emphasis on possibly-noisy hard negatives; too large makes the loss insensitive, slowing learning.

### Q14: Why does cosine similarity (with L2-normalized features) work well in contrastive learning, vs raw dot product?

Cosine similarity (or equivalently, dot product on unit-norm vectors) constrains all features to lie on the unit hypersphere. This has several consequences:

1. **No "magnitude shortcut."** With unnormalized dot products, the model could trivially increase similarity by scaling vectors larger. Cosine similarity removes this degree of freedom.
2. **Bounded similarity.** Cosine ∈ [-1, 1], leading to bounded loss. Numerical stability.
3. **Geometric interpretation.** Distance on the unit sphere is well-defined and scale-invariant.
4. **Connection to angular distance.** Often more meaningful for semantics: the angle between vectors captures direction (concept) rather than magnitude.

Empirically, cosine similarity outperforms unnormalized dot product in contrastive learning. The Wang & Isola alignment-uniformity decomposition is also defined on the unit sphere.

### Q15: Why does BYOL's loss `2 - 2·cos_sim` arise from minimizing squared L2 between unit-norm vectors?

BYOL minimizes:
```
L = ||q_θ(z_θ) - z'_ξ||²₂
```

where `q_θ(z_θ)` is the prediction (normalized to unit norm) and `z'_ξ` is the target (also normalized).

Expanding:
```
||q - z'||² = ||q||² + ||z'||² - 2·q·z' = 1 + 1 - 2·cos(q, z') = 2 - 2·cos(q, z')
```

So minimizing L2 between unit-norm vectors is equivalent to maximizing cosine similarity. BYOL's loss can equivalently be written as `1 - cos_sim` (up to a constant factor), but the L2 form is convenient for symmetrization (averaging over both orderings of views).

### Q16: Derive the EMA half-life. If m = 0.999, after how many steps does the contribution of an old update halve?

The EMA update is `θ_t = m·θ_{t-1} + (1-m)·g_t` where `g_t` is the new value. Solving recursively, the contribution of a past gradient update from k steps ago is `(1-m)·m^k`.

Half-life: find k such that `m^k = 0.5`.

```
k = log(0.5) / log(m) = log(0.5) / log(0.999) ≈ -0.693 / -0.001 ≈ 693
```

So with m = 0.999, the influence of an update halves every ~693 steps. This is why the "queue" or "momentum" representations stay relatively consistent — they evolve slowly.

For m = 0.99: half-life ≈ 69 steps.
For m = 0.9999: half-life ≈ 6932 steps.

### Q17: Why is stop-gradient critical in BYOL? What happens if you remove it?

Stop-gradient on the target side ensures gradients only flow through the online network. Without it, the system would try to make both branches output the same thing, and the easiest way to do that is to make them both output a constant (collapse).

With stop-gradient, the target network is updated only via EMA, not via gradient descent. This breaks the symmetry: the online network adjusts to predict the target, but the target doesn't adjust to be predictable. The target is constrained to be a slowly-evolving copy of the online network (via EMA), and the online network's only way to reduce loss is to actually make meaningful predictions.

If you remove stop-gradient: the network rapidly collapses. Loss goes to zero, all features become identical or very similar, downstream accuracy is near random. This was empirically demonstrated by Grill et al. (the BYOL paper).

The deeper theoretical understanding (Tian et al. 2021) shows that the linearized BYOL dynamics with stop-gradient have a stable subspace where features are nontrivial; without stop-gradient, this subspace becomes unstable and the system collapses.

### Q18: Show that the optimal classifier `f` for the InfoNCE loss is `f*(v_1, v_2) ∝ p(v_2|v_1) / p(v_2)`.

Treating `f` as an arbitrary positive function, the loss for a single positive pair plus K-1 negatives is:

```
L = -E [ log( f(v_1, v_2^+) / Σ_k f(v_1, v_2^{(k)}) ) ]
```

For the optimal f, take a functional derivative. Conditional on v_1 and the K-1 negatives drawn from p(v_2), the inner expression's expectation over `v_2^+ ~ p(v_2 | v_1)` is:

```
E_{v_2^+} [ log f(v_1, v_2^+) - log( f(v_1, v_2^+) + Σ_k f(v_1, v_2^{(k)}) ) ]
```

Setting the derivative with respect to `f(v_1, v)` to zero (Euler-Lagrange-style) gives that optimal `f` satisfies `f(v_1, v) ∝ p(v | v_1) / p(v)`.

Intuition: the optimal classifier scores a candidate based on how much more likely it is under the conditional than the marginal — i.e., how uniquely predictable it is from `v_1`. Scoring based on the density ratio is the Bayes-optimal way to discriminate the positive (drawn from conditional) from negatives (drawn from marginal).

### Q19: For MAE, why do we mask 75% of patches, and not 50% or 90%?

This is empirically determined; the MAE paper sweeps mask ratios and finds 75% optimal for ImageNet linear probe accuracy. The interpretation:

- **Too low (e.g., 25–50%):** Too easy. The model can interpolate from many visible neighbors. The pretext task doesn't force learning of high-level features.
- **Just right (~75%):** Hard but solvable. Most masked patches have few or no visible neighbors, requiring the model to use global semantic context. Linear probe accuracy peaks.
- **Too high (e.g., 90%):** Too hard. Even global context isn't sufficient to predict patches. Loss is uninformative; gradients are noisy. Performance degrades.

The "right" ratio depends on the data. For text (BERT), 15% is right because text is information-dense. For images, 75% is right because images are information-redundant.

### Q20: Explain why per-patch normalization improves MAE.

Without per-patch normalization, the MSE loss is dominated by patches with high pixel variance (textured regions, edges). Smooth patches (sky, blank walls) contribute little to the loss because their variance is small. The model spends capacity getting absolute pixel values right in textured patches, which is mostly about getting the overall brightness/color, not the structure.

With per-patch normalization, each patch is rescaled to mean 0, variance 1. The loss now measures whether the model can reproduce the *relative pattern* within each patch, regardless of absolute brightness. This forces the model to learn structural features (edges, textures, semantic content) rather than absolute illumination.

Empirically, per-patch normalization improves linear probe accuracy by ~2–3% on ImageNet — substantial for a single design choice.

---

## Applied Questions

### Q21: You're working on a medical imaging project. You have 500k unlabeled chest X-rays and 5k labeled ones (for pneumonia detection). What pretraining strategy do you recommend?

This is a textbook scenario for SSL. Recommended approach:

1. **In-domain SSL pretraining.** Pretrain a ViT or ResNet on the 500k unlabeled chest X-rays using a strong SSL method. Both MAE and DINO/BYOL would be reasonable; MAE may be simpler to implement and tends to scale well.
2. **Domain-appropriate augmentations.** For chest X-rays, avoid color jitter (X-rays are grayscale) and aggressive cropping (may crop out the lungs). Use rotations, flips, and intensity adjustments.
3. **Fine-tune on the 5k labels.** Use the pretrained encoder, add a classification head, fine-tune with appropriate LR (smaller for encoder, larger for head) and strong regularization.

Compare against:
- ImageNet pretrained (supervised or SSL).
- From-scratch on 5k labels (likely poor).
- Few-shot evaluation: how does performance scale with 100, 500, 1000, 5000 labels?

In-domain SSL pretraining should win, especially on the lower-shot regime. Report results across multiple labeled set sizes.

### Q22: You're choosing between SimCLR, MoCo, BYOL, and MAE for pretraining a foundation model for retail product images. What are the key considerations?

Considerations:

- **Compute budget.** MAE is the most compute-efficient (asymmetric design, no momentum encoder, no queue). SimCLR needs huge batches (4096+). MoCo and BYOL are intermediate.
- **Memory budget.** SimCLR needs lots of memory for large batches. MoCo's queue is memory-efficient. BYOL has two networks but no large batch requirement.
- **Architecture preference.** MAE is most natural with ViT; contrastive methods work with CNNs and ViTs.
- **Augmentation design.** Contrastive methods are sensitive to augmentation choices; you'd need to tune for retail images (don't want crops that remove the product). MAE doesn't depend on augmentations beyond crop+flip.
- **Downstream tasks.** If primarily classification → contrastive methods give good linear probe. If primarily detection/segmentation (pixel-level tasks) → MAE typically wins.
- **Robustness.** MAE has fewer hyperparameters; less risk of misconfiguration.

For a fresh project, I'd lean toward MAE (or DINO if you want strong linear probe) due to simpler design and good scaling properties. SimCLR is well-understood but requires expensive batches.

### Q23: Your team trained MoCo on satellite imagery. The contrastive loss is decreasing nicely, but linear probe accuracy is poor. What might be wrong?

Possible issues, in rough order of likelihood:

1. **Augmentations don't match the domain.** Satellite imagery may be rotation-invariant (no canonical "up"), so rotation augmentation is fine, but you should *not* use horizontal flip in some cases (e.g., text overlays would be unrealistic). Color jitter may also be inappropriate (specific spectral signatures matter for satellite). The model is learning invariance to features that should be retained.
2. **The pretext is exploiting a shortcut.** Satellite images have constant features (sun angle, cloud patterns) that the model might use to "match" positives without learning semantic content. Diagnose by trying different augmentation settings.
3. **Crop strategy is wrong.** If the crop is too aggressive, the two views might share too little overlap; too gentle, and they're trivially similar. Tune the crop ratio for the data scale.
4. **Momentum is wrong.** Too high (m=0.9999) means encoder updates too slowly; too low (m=0.99) means representations in the queue are stale. Default 0.999 is usually fine; deviate only with reason.
5. **Queue size is inappropriate.** Too small (e.g., 4096) is fine for ImageNet but may be too few for the diversity of satellite imagery. Try larger queues.
6. **Forgot shuffle BN.** If using BatchNorm with multi-GPU training, you must use shuffle BN (per MoCo paper) or BN statistics leak between query and key, providing a shortcut.

Fix: First inspect what the model is learning (visualize nearest neighbors for some images). Then try domain-appropriate augmentations.

### Q24: A junior engineer pretrained BERT-style on a small dataset (1M sentences) and is disappointed with results. What advice do you give?

Several things to discuss:

1. **Scale matters for masked modeling.** BERT's original pretraining used 3.3B tokens. 1M sentences (~30M tokens) is ~100× too small. Masked modeling shines at scale; with limited data, contrastive methods or smaller architectures may work better.
2. **Consider initializing from existing pretrained weights.** Take a public BERT/RoBERTa checkpoint, then continue pretraining on the in-domain data ("continued pretraining" or "domain adaptive pretraining"). This combines large-scale general knowledge with in-domain specialization.
3. **Reduce model size if scaling data isn't possible.** A 30M parameter model trained on 30M tokens is more reasonable than a 110M parameter model trained on 30M tokens.
4. **Use a more sample-efficient objective.** ELECTRA's replaced-token-detection objective is more sample-efficient than vanilla MLM. T5's span masking can also help.
5. **Verify the data is high-quality.** Garbage in, garbage out — even at small scale, well-curated data beats noisy web scrapes.

Most importantly: align expectations. SSL pretraining at 1M sentences won't match BERT's quality; the benefit will be marginal compared to a well-chosen public model fine-tuned on the same data.

### Q25: You're deploying an SSL pretrained encoder for image retrieval in production. What's the right adaptation strategy?

For retrieval, you typically want:
- High linear separability (similar images have similar features).
- L2-normalized features (so cosine similarity = dot product, fast indexing).
- Fixed feature dimension matching your retrieval index.

Strategy:
1. **Use the encoder backbone, not the projection head.** The projection head is for the contrastive objective during training and may discard useful retrieval information.
2. **Add an L2 normalization layer.** Final features should be unit norm for cosine similarity / FAISS indexing.
3. **Optional: learned projection to retrieval space.** Train a small MLP on labeled retrieval pairs (if available) to optimize for retrieval-specific objectives like triplet loss.
4. **Index with FAISS or similar.** ANN (approximate nearest neighbor) for fast retrieval at scale.
5. **Calibrate the threshold.** Determine the cosine similarity threshold for "relevant" matches via held-out evaluation.

Alternative: train the encoder with a retrieval-specific contrastive objective where positives are known retrieval pairs (if available). This is closer to metric learning / supervised contrastive than pure SSL.

### Q26: You're given a budget to pretrain a vision foundation model. You can either: (a) pretrain MAE on ImageNet for 4 weeks, or (b) pretrain MAE on a 100× larger uncurated web dataset for 1 week. Which do you choose and why?

Generally favor (b), but with caveats. Reasoning:

- SSL benefits substantially from data diversity. ImageNet, while large by traditional standards, is curated and biased toward certain object categories. Web-scale data is more diverse.
- MAE is data-efficient per epoch but benefits from more data — research shows scaling data helps even if you reduce epochs.
- One week of compute on 100× more data means each example is seen many fewer times (1/400 the per-example training), but the diversity helps generalization.

Caveats favoring (a):
- ImageNet is curated; web data is noisy. Junk data can hurt. You'd need to pre-filter (e.g., CLIP scores).
- If your downstream tasks are ImageNet-like (object classification on similar distribution), in-domain training matters.
- Reproducibility: ImageNet is a fixed benchmark; web-scale data is harder to reason about.

Decision: lean toward (b) if data quality can be ensured (filter), but include a final "fine-tune on ImageNet" stage to specialize for downstream evaluation. This combines diversity with curation.

### Q27: How would you design augmentations for SSL pretraining on satellite imagery?

Satellite imagery has unique properties:
- **Rotation invariance:** No canonical orientation. Rotation augmentation up to 360° is appropriate.
- **Mirror-asymmetry:** Some features are not mirror-symmetric (e.g., text, road markings). Be careful with horizontal flip.
- **Spectral importance:** Specific wavelength ratios matter (NDVI for vegetation, water indices). Don't aggressively color-jitter.
- **Scale variability:** Same scene at different zoom levels reveals different features. Multi-scale crops are useful.
- **Clouds / atmospheric noise:** Could be considered "augmentation" naturally — sometimes use cloud overlay augmentation to encourage robustness.

Recommended augmentation pipeline:
- Random crop (with appropriate scale range, e.g., 30–100% of image area).
- Random rotation (0–360°).
- Random horizontal flip (acceptable in most cases).
- Mild brightness/contrast adjustments (avoid full color jitter).
- Optionally: random erasing (simulates partial cloud cover).

Avoid:
- Aggressive color jitter (changes spectral signature).
- Channel dropping (loses important spectral information).

The choice of augmentations should be tested empirically: vary one at a time and measure linear probe accuracy on a held-out labeled subset.

### Q28: A team reports their SSL method outperforms SimCLR on ImageNet linear probe by 2 points. What questions do you ask before believing the result?

1. **Same encoder?** ResNet-50 is standard; many "improvements" come from secretly using ResNet-50 with more channels or ResNet-101.
2. **Same training compute?** Number of epochs, batch size, optimizer, schedule. SimCLR was trained for 1000 epochs in the original paper; many comparisons use shorter training.
3. **Same evaluation protocol?** Linear probe configuration matters: number of epochs, augmentation during probe, learning rate. Compare apples to apples.
4. **Same augmentation policy?** If they're using stronger augmentation in addition to their method, the method's contribution is unclear.
5. **Statistical significance?** A 2-point difference can be within run-to-run variation. Need multiple seeds.
6. **Fine-tuning results?** Linear probe alone can be misleading; check fine-tuning performance too.
7. **Transfer learning?** Does the improvement transfer to other datasets/tasks, or is it specific to ImageNet?
8. **Ablations.** Is the improvement from one specific change, or a combination?

A 2-point improvement is meaningful but worth scrutinizing. The SSL field has had many "improvements" that didn't replicate or didn't transfer.

---

## Debugging & Failure-Mode Questions

### Q29: You're training BYOL. Loss converges to near-zero in the first 100 steps. What's wrong?

This is collapse: the network is producing constant or near-constant features, making the predictor's job trivial. Likely causes:

1. **Stop-gradient missing.** Verify that gradients do not flow through the target network. This is the most common cause of collapse.
2. **Predictor missing.** BYOL requires a predictor on the online side. Without it, both networks become symmetric and collapse.
3. **EMA not applied.** If the target network is updated with gradients (or copied directly from online without EMA), collapse is likely.
4. **Learning rate too high.** Even with correct setup, very high LRs can drive the system into collapse.
5. **Predictor too weak / wrong architecture.** If the predictor lacks BatchNorm or has too few hidden units, collapse is more likely.

Diagnose:
- Compute std of features along batch dimension. Should be > 0; if it's near 0, features have collapsed.
- Check the actual feature values: if all examples produce similar vectors, you have collapse.
- Inspect kNN accuracy on a held-out set; near-random means collapse.

Fix: First, audit the implementation (stop-gradient, EMA, predictor). Then check hyperparameters. BYOL is more fragile than its name suggests.

### Q30: SimCLR training loss decreases nicely, but downstream accuracy is much worse than reported in the paper. What might be wrong?

Several possible causes:

1. **Augmentation policy is wrong.** SimCLR's augmentations are critical. Verify: random resized crop (scale 0.08–1.0), random horizontal flip, color jitter (brightness/contrast/saturation 0.4, hue 0.1), random grayscale (p=0.2), Gaussian blur (kernel 23, applied to one of the two views). If you skip any of these, especially color jitter or blur, performance drops.
2. **Batch size too small.** SimCLR uses batch size 4096. With batch 256, you have far fewer negatives, hurting the loss's discriminative power. Use a queue (MoCo-style) if you can't use large batches.
3. **Insufficient training duration.** SimCLR was trained for 1000 epochs. If you trained 100, you're far from convergence.
4. **Wrong projection head.** SimCLR's projection head is a 2-layer MLP with 2048→2048→128 output. Variations matter.
5. **Wrong temperature.** τ = 0.5 is standard. 0.1 or 1.0 will give different (usually worse) results.
6. **Used projection head for downstream.** A common bug — use the encoder output, not the projection head output, for downstream tasks.
7. **Wrong evaluation protocol.** Linear probe protocol matters (LR, schedule, augmentation).

Diagnose by trying to replicate published numbers exactly (same code, same hyperparameters) before innovating.

### Q31: MoCo pretraining is producing features that all look the same after 50 epochs. The contrastive loss is still high though. What's happening?

If features are similar but the loss is still high, this is unusual — usually collapse implies low loss (positives match easily). Possible explanations:

1. **The "similarity" you're measuring is misleading.** Maybe features are spread across a low-dim subspace but vary within it. Run PCA on the feature matrix and check the effective rank.
2. **Numerical instability / NaN.** If gradients have exploded, features may be all zero or all nan, and loss is high but uninformative. Check feature norms and gradient norms.
3. **Augmentations too weak.** If positive pairs are nearly identical inputs, there's no learning signal. Verify augmentation pipeline.
4. **Shuffle BN missing.** Without shuffle BN (in multi-GPU setups), BN statistics leak between query and key, allowing the model to "cheat" by detecting the BN signature. Loss decreases but features don't learn semantics.
5. **Encoder bug.** Maybe the encoder isn't producing distinct features at all (a coding error like always returning the same embedding).

Diagnose: Compute pairwise feature distances on a small set of distinct inputs. If they're all small, the encoder isn't producing distinct features. Investigate the implementation.

### Q32: You trained MAE successfully (loss decreasing, reconstructions look reasonable), but linear probe accuracy is much lower than reported. What's wrong?

1. **Linear probe protocol differs.** MAE's linear probe protocol uses BatchNorm (no affine) on the features before the linear layer. This is unusual but matters. Without it, accuracy can be 5–10 points lower.
2. **Mask ratio at inference.** For linear probe, you don't mask anything — the encoder sees all patches. Verify your evaluation pipeline doesn't accidentally mask during probing.
3. **Wrong layer for features.** MAE features are typically taken from the encoder's last layer. Some implementations average across layers or use a specific block.
4. **Insufficient pretraining.** MAE benefits from long pretraining (800+ epochs). At 100 epochs, you're nowhere near convergence.
5. **Wrong encoder size.** MAE results are reported for ViT-Large. ViT-Base will be lower.
6. **Per-patch normalization disabled.** Without per-patch normalization, performance drops.

Note: even with everything correct, MAE has a notably *lower* linear probe accuracy than BYOL/SimCLR (e.g., MAE's ~68% vs BYOL's ~74%). MAE shines at fine-tuning and downstream tasks, not linear probing. Make sure you're comparing to the right baseline.

### Q33: You're seeing very different SimCLR results across runs (with different random seeds). Why, and how to reduce variance?

Variance sources:
1. **Negative sampling stochasticity.** Different batches give different negative sets each iteration. Larger batches reduce variance.
2. **Augmentation randomness.** Each forward pass uses different augmentations.
3. **Optimizer randomness.** SGD with momentum has stochastic dynamics.
4. **Initialization.** Different random initializations lead to different solutions.

Reduce variance:
- **Average over multiple seeds.** Report mean ± std over 3–5 runs.
- **Larger batches.** Reduce stochasticity per step.
- **Longer training.** Convergence reduces seed dependence.
- **Use deterministic algorithms where possible.** PyTorch's `torch.use_deterministic_algorithms(True)`.
- **Same-seed protocol.** Use the same seed for the linear probe across SSL methods to isolate the SSL training variance.

In SSL papers, ±0.5 to ±1.0% on ImageNet is typical run-to-run variance for established methods. Differences smaller than this should not be claimed.

### Q34: Your team's MAE shows great loss curves but the model fails on a downstream segmentation task — performance is below random initialization. Possible causes?

This is suspicious; MAE typically transfers well to segmentation. Possible causes:

1. **Pretraining/downstream mismatch.** Was pretraining done at a different resolution than the downstream task? Resolution mismatch can severely hurt ViT.
2. **Wrong feature extraction.** For segmentation (per-pixel prediction), you typically use multi-scale features from multiple ViT layers, not just the last layer. Make sure you're extracting features at the right granularity.
3. **Catastrophic forgetting during fine-tuning.** Fine-tuning with too high a learning rate can destroy pretrained features. Use lower LR for the encoder or freeze early layers.
4. **Decoder misuse.** If you accidentally used the MAE pretraining decoder for segmentation, you'd get nonsense — that decoder was trained for pixel reconstruction, not segmentation.
5. **Domain mismatch.** If pretraining was on ImageNet (natural images) and downstream is medical imaging, the features may be inappropriate. Try in-domain SSL.
6. **Bug in the segmentation head.** Verify the segmentation head's setup. The encoder output → segmentation head pipeline can have subtle errors.
7. **Mask tokens leaked.** If at inference time the model sees mask tokens (when it shouldn't), behavior is undefined.

Diagnose: First, verify linear probe / classification works (basic sanity check). If classification is fine, the issue is in the segmentation pipeline. If even classification fails, the encoder itself is broken or the features are being used wrong.

### Q35: You're observing that the contrastive loss decreases but feature uniformity stays near zero (features highly concentrated). How is this possible, and what does it mean?

Loss can decrease while features remain concentrated if:
1. **Positives are particularly tightly clustered.** Loss drops because positives are close, but uniformity loss isn't decreasing because features as a whole are not spread.
2. **Hard negatives are being mined effectively.** Even concentrated features can have well-separated "regions" that distinguish hard negatives.
3. **The loss is not strongly enforcing uniformity.** This is actually expected behavior if you're pulling positives together hard but only weakly pushing negatives. Investigate the magnitude of the negative term in the loss.

What it means:
- Some semantic structure is being learned (positives are distinguishable from negatives).
- But the feature space is suboptimal — collapsed onto a low-dimensional manifold.
- Downstream performance will likely be mediocre.

Mitigation:
- Add an explicit uniformity term to the loss.
- Use a temperature that emphasizes negatives more.
- Increase batch size or queue size.
- Examine the augmentation policy — maybe positives are too similar (not enough augmentation diversity).

---

## Follow-up & Probing Questions (depth-of-understanding)

### Q36: People often say "SSL works because it learns to compress information." Is this true? Discuss.

It's a partial truth, but oversimplified. The relationship between SSL and information theory is subtle:

- **InfoNCE bounds mutual information.** Minimizing InfoNCE maximizes a lower bound on `I(v_1; v_2)`. This is a form of "preserving information about the data" rather than compressing.
- **But MI maximization isn't the whole story.** Tschannen et al. (2019) showed that better MI bounds don't necessarily yield better downstream representations. The structure of the encoder and the choice of negatives matter independently.
- **Compression interpretation:** Some SSL methods (especially those with bottlenecks like the MAE encoder seeing only 25% of patches) implicitly compress. But this isn't a universal SSL property.
- **Predictive coding view:** SSL learns features that are predictive across views or across time. Predictability is a kind of "useful information" for the downstream task.

A more accurate framing: SSL learns features that capture invariances and predictive structure relevant to the data distribution. Information theory provides one lens, but the success of SSL also depends on inductive biases of the architecture, the augmentation/masking policy (which encodes domain knowledge), and the loss function (which shapes the geometry of the feature space).

### Q37: How do contrastive learning and masked modeling relate? Are they fundamentally different, or instances of a common framework?

Both can be viewed as instances of a common framework: **predicting some part of the data from another part**.

- **Contrastive:** Predict the identity of one view given another view. The "prediction" is implicit through the discriminative loss (rank the true positive above negatives).
- **Masked modeling:** Predict the value of masked tokens given visible tokens. The prediction is explicit (regression or classification).

In a sense, masked modeling is a "harder" task — it requires committing to specific reconstructions. Contrastive learning is "easier" in that it only requires distinguishing positives from negatives, not generating them.

Both impose different inductive biases:
- Contrastive: features must be invariant to augmentations and discriminative across instances.
- Masked: features must contain enough information to reconstruct missing parts.

They're complementary in some sense. iBOT, DINO v2, and other recent methods combine both objectives, getting better results than either alone. The framework of "self-prediction" unifies them, with different methods choosing different ways to construct the prediction problem.

### Q38: Why don't we just train SSL with a classification objective using "image instance" as the class (i.e., each image is its own class)?

This is essentially what early SSL methods like instance discrimination (Wu et al., 2018, the precursor to MoCo) did. The challenges:

1. **Scaling.** ImageNet has 1.28M images. A 1.28M-class softmax is computationally expensive. Methods like NCE (Noise Contrastive Estimation) and InfoNCE arose to make this tractable by sampling negatives.
2. **Inefficiency without augmentation.** Without augmentation, "predict the image's identity from itself" is trivial. With augmentation, you're really learning invariance to augmentation — which becomes contrastive learning.
3. **The class memory is impractical.** Each instance's representation needs to be tracked. Wu et al.'s memory bank addressed this but had staleness issues. MoCo's queue is the modern solution.

So contrastive learning *is* essentially instance classification, but with engineering for scale (sampling negatives instead of full softmax) and augmentation (to create positive pairs).

### Q39: What's the connection between SSL and curriculum learning?

Curriculum learning trains models on easier examples first, then progressively harder ones. SSL can be viewed through this lens:

- **Implicit curriculum in masked modeling:** Early in training, the model learns to reconstruct using simple statistics (mean color, common patterns). As training progresses, it learns more sophisticated reconstructions requiring semantic understanding.
- **Hard negative mining as curriculum:** In contrastive learning with a hard negative emphasis, the model implicitly faces a curriculum — easy negatives are quickly dismissed, hard negatives drive learning. The temperature τ controls the curriculum's difficulty.
- **Mask ratio as curriculum:** Some methods (e.g., curriculum MAE variants) start with low mask ratios and gradually increase, enabling the model to bootstrap.

Conversely, SSL itself can serve as a "curriculum" for downstream training: pretrain on a self-supervised task (rich signal), then fine-tune on the target task. This curriculum is one of the major reasons SSL works — it provides easier-to-learn intermediate objectives.

### Q40: How does SSL relate to data augmentation in supervised learning?

Both SSL and augmented supervised learning use augmentation to encode invariances, but use them differently:

- **Supervised + augmentation:** Augmentations are applied to inputs while keeping labels fixed. The model learns to be invariant because the loss penalizes label misclassification across augmented versions. This indirect signal works but is weak.
- **Contrastive SSL:** The contrastive loss *directly* enforces invariance — augmented versions of the same image must have nearby representations. This is a stronger and more efficient invariance signal.

You can think of contrastive SSL as a more direct way to inject invariance priors into a model. This is why SSL pretraining followed by supervised fine-tuning often beats supervised training alone: SSL gives a strong invariance prior; supervised learning then adapts the invariant features to the task.

### Q41: Compare BYOL with knowledge distillation. Are they the same thing?

They share structure but have different goals:

- **Knowledge distillation:** A small "student" model learns to match the predictions of a fixed, pretrained "teacher" model. Goal: compress a large model into a small one, transferring knowledge. Teacher is pretrained, student starts from scratch.
- **BYOL:** Online network learns to predict the target network's representation. Target network is an EMA of the online network. No external pretrained teacher. Goal: learn representations from scratch.

BYOL can be viewed as "self-distillation": the model distills knowledge from a slowly-updated version of itself. This is a sub-genre of methods including DINO, MoCo v3 (which has similar self-distillation aspects), and others.

A key difference: in classical distillation, the teacher is fixed. In BYOL, the target is moving (slowly). This moving target is what makes BYOL interesting as a learning algorithm — the model "lifts itself by its bootstraps."

### Q42: What happens to SSL's effectiveness at very large scales (billions of images, billions of parameters)? Are there scaling laws?

Empirical observations:
- SSL benefits substantially from scaling both data and model size, similar to (and possibly more than) supervised learning.
- DINO v2, OpenCLIP, EVA-02, and other recent works have demonstrated strong scaling properties for SSL pretraining with billions of images and billion-parameter models.
- Scaling laws for SSL are an active research area. Early results suggest the effective scaling exponent is comparable to language model scaling laws.

Caveats:
- At very large scales, data quality matters more than quantity. Filtering web data is crucial.
- The choice of method matters: methods that scale well with batch size or data (MAE, DINO, contrastive with queues) are preferred at scale.
- Compute requirements grow rapidly. State-of-the-art SSL models now require thousands of GPU-days.
- Diminishing returns: each doubling of data/compute provides smaller absolute gains, even if the trend is favorable.

Practical implication: SSL is now the default for foundation models, and scaling SSL pretraining is the dominant axis of progress. The ceiling has not yet been reached.

### Q43: Why do contrastive methods care about "hard negatives" while masked modeling doesn't have an analog?

Contrastive methods rank a positive against many negatives. Most negatives are "easy" — clearly different from the positive — and provide little learning signal. The few "hard" negatives (semantically similar but not the same) drive learning. The temperature parameter τ implicitly weights toward hard negatives via the softmax.

Masked modeling has no negatives. The model predicts a target value (token, pixel) directly. Every prediction provides signal (via the loss); there's no concept of "easy" vs "hard" examples in the same sense.

That said, masked modeling has analogous concepts:
- "Hard masks": some masked positions are easier (low-information, high-redundancy regions); others are harder (semantically critical regions). The mask ratio implicitly controls difficulty.
- Curriculum: progressive masking schedules (start low, ramp up) can make learning more efficient.
- Loss reweighting: weighting harder examples (those with higher loss) more — a direct analog to hard-negative mining.

So while there's no direct analog of "hard negatives" in masked modeling, the underlying issue (concentrating learning on informative examples) appears in different forms.

### Q44: Why does SSL sometimes outperform supervised pretraining, even when labels are available?

This is a striking empirical finding: SSL pretraining can outperform supervised pretraining even when the labeled dataset is the same size. Reasons:

1. **Labels are noisy.** Supervised training optimizes the noisy labels, which can hurt feature quality. SSL ignores labels and learns from the data itself.
2. **Labels are limiting.** Supervised training optimizes for the specific label set (e.g., 1000 ImageNet classes). Features are tuned to discriminate those classes; out-of-class concepts may be suppressed. SSL learns more general features.
3. **Labels are a coarse signal.** A class label is one bit per example (relative to vocabulary log_2(C)). The SSL signal can be much richer (e.g., reconstructing all pixels gives ~10^4 bits per example).
4. **Implicit ensembling.** SSL augmentations produce many "examples" per real image, providing more effective training signal.
5. **Avoidance of label bias.** Supervised models pick up on label-correlated artifacts (e.g., watermarks, photographer biases). SSL is more robust to these.

When does supervised pretraining win? When:
- Labels are extremely high-quality and the label space is rich (e.g., dense annotation).
- The downstream task closely mirrors the pretraining label structure.
- The pretrained model is at a smaller scale than the SSL ceiling.

In practice, SSL has become the default for foundation models because it scales better with data and avoids dependency on label availability.

### Q45: Are SSL-pretrained models more or less interpretable than supervised models? Discuss.

This is a nuanced question:

**Arguments for SSL being more interpretable:**
- SSL features tend to be more disentangled, with different dimensions capturing different concepts (e.g., DINO features show object boundaries clearly).
- SSL doesn't impose a label taxonomy, so features represent the natural structure of the data rather than a human-imposed categorization.
- Reconstruction-based SSL (MAE) can be directly inspected via reconstructions.
- Attention maps from SSL ViTs (like DINO) cluster nicely on object boundaries, providing interpretable visualizations.

**Arguments against:**
- Without labels, there's no clear semantic anchor for features. Attribution to specific concepts requires probing.
- Some SSL features are "weird" — they may represent textures or low-level statistics that don't align with human intuitions.
- Linear probing can be misleading (a feature is "useful" if a probe can extract it, but the feature itself may be entangled).

In practice, SSL models like DINO have produced some of the most striking interpretability results in vision (clear segmentation maps from attention). But interpretation requires effort — SSL features don't come pre-annotated with semantics.

### Q46: How might SSL evolve in the next 5 years?

Speculative but informed:

1. **Multimodal SSL becomes the norm.** Pretraining on (image, text), (image, audio), (video, text) etc. jointly — like CLIP scaled up. Foundation models will be increasingly multimodal.
2. **Better unification of SSL paradigms.** Methods that combine contrastive, masked modeling, and self-distillation (like iBOT, DINO v2) will become more common.
3. **Domain-specific foundation models.** SSL pretraining on medical images, satellite data, scientific data, code, etc. — each producing a specialized foundation model.
4. **Compute-efficient SSL.** As models scale, methods that achieve good results with less compute (efficient masking, sparse architectures) will be valuable.
5. **SSL for video and 4D data.** Currently, video SSL is less mature than image SSL. Methods for learning from video (with temporal redundancy) will improve.
6. **Theoretical understanding catches up.** Why does BYOL work? What's the right objective? Theoretical work (like Tian et al., HaoChen et al.) will mature.
7. **Integration with reinforcement learning.** SSL representations as input to RL agents, providing a foundation for embodied AI.
8. **Better evaluation frameworks.** Beyond ImageNet linear probe — more comprehensive, real-world evaluations.

The general trend: SSL is moving from a research curiosity to the foundational layer of most ML systems. Specific methods will evolve, but the paradigm of learning from unlabeled data with self-constructed objectives is here to stay.

---

## Closing Notes

This guide has covered self-supervised learning from foundational concepts through specific methods (SimCLR, MoCo, BYOL, MAE) to applications and evaluation. The field is moving fast, but the fundamentals — augmentation-based invariance, contrastive comparison, masked prediction, and the asymmetric architectures that prevent collapse — provide a stable conceptual base.

Key takeaways:
1. SSL eliminates the labeling bottleneck, enabling training on virtually unlimited data.
2. Augmentations encode domain-specific invariances; the choice is critical.
3. Contrastive methods (SimCLR, MoCo) excel at linear probing; masked modeling (MAE) excels at fine-tuning.
4. BYOL's success without negatives reveals that careful asymmetric design can prevent collapse.
5. The pretrain-then-adapt pipeline is now the dominant paradigm for most ML applications.

For deeper study, the key papers (in order):
- Doersch et al. 2015 (relative patch prediction)
- Gidaris et al. 2018 (rotation prediction)
- Devlin et al. 2018 (BERT)
- Wu et al. 2018 (instance discrimination)
- He et al. 2019 (MoCo)
- Chen et al. 2020 (SimCLR)
- Grill et al. 2020 (BYOL)
- Wang & Isola 2020 (alignment-uniformity)
- Caron et al. 2021 (DINO)
- He et al. 2021 (MAE)
- Tian et al. 2021 (BYOL theory)
- Oquab et al. 2023 (DINO v2)

Each paper is well worth reading in full for the depth and the empirical analyses they provide.

