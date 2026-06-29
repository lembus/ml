# Comprehensive Machine Learning & Applied Science Guide

## Module 1: Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT)

### 1. Motivation & Intuition

Imagine you have a student who has read every book in a massive library. They are incredibly knowledgeable but have no idea how to take an exam or answer a specific question. If you ask them "How do I bake a cake?", they might just start reciting a history of wheat.

**Supervised Fine-Tuning (SFT)** is the process of teaching that student how to behave. In Machine Learning, we take a **Pre-trained Model** (which has learned the statistical patterns of language) and turn it into an **Assistant** (which follows instructions).

* **Instruction Tuning:** We provide the model with a "cheat sheet" of questions and ideal answers. This bridges the gap between *predicting the next word* and *completing a task*.
* **The Problem of Scale:** A modern Large Language Model (LLM) might have 70 billion parameters. Updating all of them (Full Fine-Tuning) requires massive amounts of GPU memory. **PEFT (Parameter-Efficient Fine-Tuning)** was invented to allow us to "specialize" these models using only a fraction of the hardware.

---

### 2. Conceptual Foundations

#### Key Terms
* **Base Model:** A model trained on raw text to predict the next token (e.g., Llama-3).
* **Instruction Dataset:** A collection of (Prompt, Response) pairs.
* **Catastrophic Forgetting:** When a model learns a new task (e.g., medical coding) but loses its ability to perform general tasks (e.g., writing poetry).
* **Low-Rank Adaptation (LoRA):** A method that keeps the original model weights frozen and only trains a tiny "plugin" layer.
* **QLoRA:** An extension of LoRA that quantizes the base model to 4-bit precision to further reduce memory usage.
* **Prefix Tuning & Prompt Tuning:** Techniques that prepend trainable continuous vector representations (virtual tokens) to the input sequence or hidden states while keeping the base transformer frozen.
* **Adapters:** Small bottleneck feed-forward layers inserted sequentially or in parallel within transformer blocks.

#### How Components Interact Step-by-Step
1. **Data Formatting:** Raw task data is structured into a structured prompt template: `### Instruction: {prompt} \n ### Response: {response}`.
2. **Tokenization & Embedding:** The formatted string is converted into token IDs and projected into dense vector spaces.
3. **Forward Pass:** The pre-trained weights (and adapter layers, if using PEFT) process the sequence to predict the response token by token.
4. **Loss Calculation & Backpropagation:** The model is evaluated against the ground truth response tokens. Gradients are computed and backpropagated—either updating all weights (Full SFT) or only the PEFT parameters (e.g., LoRA matrices $A$ and $B$).

#### Assumptions & Violations
* **Underlying Assumption:** The pre-trained base model already possesses the requisite factual knowledge and reasoning capabilities; SFT merely aligns the output *format* and *style* to user expectations.
* **Violation Consequences:** If a model is fine-tuned on domains or facts entirely absent from its pre-training corpus, it will frequently "hallucinate" (generate confident fabrications). It attempts to satisfy the structural format of the prompt without the grounding latent representations.

---

### 3. Mathematical Formulation

#### The SFT Objective
We minimize the negative log-likelihood (cross-entropy loss) of the target tokens $y$ given the prompt context $x$:

$$
\mathcal{L}_{SFT}(\theta) = -\sum_{t=1}^{|y|} \log P(y_t \mid x, y_{<t}; \theta)
$$

Where $\theta$ represents the trainable parameters of the model, and $y_{<t}$ denotes the preceding autoregressively generated tokens.

#### LoRA: Low-Rank Adaptation
Standard fine-tuning modifies the dense weight matrix $W_0 \in \mathbb{R}^{d \times k}$ by adding a dense update matrix $\Delta W$:

$$
h = W_0 x + \Delta W x
$$

LoRA hypothesizes that the weight updates have a **low intrinsic dimension**. We decompose $\Delta W$ into two low-rank matrices, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where the rank $r \ll \min(d, k)$:

$$
\Delta W = \frac{\alpha}{r} (B A)
$$

* **Matrix $A$:** Projects the high-dimensional input $x$ down to a low-dimensional bottleneck of size $r$.
* **Matrix $B$:** Projects the low-dimensional representation back up to the original dimension $d$.
* **Scaling Factor $\frac{\alpha}{r}$:** $\alpha$ is a constant hyperparameter. Scaling by $\frac{\alpha}{r}$ reduces the need to retune hyperparameters when varying the rank $r$.

#### QLoRA Innovations
QLoRA introduces three mathematical and architectural optimizations:
1. **4-bit NormalFloat (NF4):** An information-theoretically optimal quantile quantization scheme for normally distributed weights.
2. **Double Quantization:** Quantizes the quantization constants themselves, saving an additional $\approx 0.37$ bits per parameter.
3. **Paged Optimizers:** Uses NVIDIA unified memory to evict optimizer states to CPU RAM during GPU memory spikes, preventing out-of-memory (OOM) faults.

---

### 4. Worked Examples

#### Concrete LoRA Parameter Calculation
Suppose we have a transformer linear projection layer with a weight matrix $W_0$ of dimensions $d = 4096$ and $k = 4096$ (total parameters = $16,777,216$).

1. **Full Fine-Tuning:**
   * Gradients and Adam optimizer states (momentum and variance) must be allocated for all **16.78M** parameters.
   * Memory footprint per layer $\approx 16.78\text{M} \times (2\text{ bytes weights} + 2\text{ bytes grad} + 8\text{ bytes optimizer}) = 201.33\text{ MB}$.

2. **LoRA Decomposition (Rank $r = 8$):**
   * Matrix $A$ dimension: $8 \times 4096 \implies 32,768$ parameters.
   * Matrix $B$ dimension: $4096 \times 8 \implies 32,768$ parameters.
   * **Total Trainable Parameters:** $65,536$ (a **99.61% reduction**).
   * Optimizer memory footprint drops from $\approx 134.2\text{ MB}$ to just $\approx 0.52\text{ MB}$ per layer.

3. **Inference Execution:**
   During deployment, the frozen base weights and trainable adapters are merged explicitly:

   $$W_{\text{merged}} = W_0 + \frac{\alpha}{r} (B A)$$

   This results in **zero inference latency overhead** compared to the un-fine-tuned base model.

---

### 5. Relevance to Machine Learning Practice

#### Lifecycle Usage
* **Training/Adaptation:** SFT is universally applied post-pre-training to transform foundational base models into dialogue agents.
* **Multi-Tenancy Inference:** PEFT allows serving hundreds of specialized domain assistants from a single GPU by swapping tiny LoRA adapters ($A$ and $B$ matrices) in VRAM while sharing the frozen multi-gigabyte base model $W_0$.

#### Trade-off Analysis
| Dimension | Full Parameter Fine-Tuning | LoRA / QLoRA | Prompt / Prefix Tuning |
| :--- | :--- | :--- | :--- |
| **Computational Cost** | Extremely High (Multi-node clusters) | Low (Single consumer/server GPU) | Very Low |
| **Expressive Capacity**| Maximum ceiling | High (Approaches Full FT at $r \ge 16$) | Moderate (Struggles on complex tasks)|
| **Catastrophic Forgetting**| High Risk | Moderate Risk (Protected by frozen $W_0$) | Low Risk |
| **Storage per Task** | $\approx 10\text{ GB} - 100\text{+ GB}$ | $\approx 50\text{ MB} - 200\text{ MB}$ | $\approx \text{Few KB}$ |

---

### 6. Common Pitfalls & Misconceptions

* **Data Leakage:** Unintentionally including target label tokens or benchmark evaluation sets within the instruction prompt context during training.
* **Superficial Alignment (Style Over Substance):** Over-training on small, homogeneous SFT datasets causes models to adopt repetitive conversational tics ("As an AI language model...") without gaining true domain competency.
* **Template Mismatch:** Training a model using one system prompt formatting convention (e.g., ChatML) but executing inference with another (e.g., Alpaca or Vicuna formatting). This destroys token-level conditional probability distributions.

---

### Interview Preparation: SFT & PEFT

#### Foundational
**Q: What is the primary difference between Pre-training and Instruction Fine-tuning?** **A:** Pre-training is self-supervised learning on massive, diverse, unstructured text corpora designed to learn general language mechanics, syntax, and world representations. Instruction fine-tuning is supervised learning on curated tasks designed to teach the model how to interpret user intent and adhere to conversational dialogue interfaces.

#### Mathematical
**Q: Why are LoRA matrices initialized with Gaussian noise for matrix $A$ and zeros for matrix $B$?** **A:** If both matrices were initialized with random noise, their product $BA$ would produce arbitrary non-zero modifications to the hidden states at step zero, destabilizing the pre-trained representations. By setting $B=0$, the initial product $\Delta W = BA = 0$. This ensures the training starts exactly at the pre-trained base model function and smoothly drifts as gradients accumulate.

#### Applied
**Q: You fine-tuned an LLM on proprietary legal contracts, but it now fails at basic summarization and general Python coding. How do you resolve this?** **A:** This is severe catastrophic forgetting. Resolutions include:
1. Switch from Full FT to **LoRA**, keeping base generalist representations intact.
2. Implement **Data Replay Buffer:** Construct a heterogeneous training mix containing 80% domain-specific legal data and 20% general instruction/coding datasets (e.g., FLAN or SlimOrca) to preserve base capabilities.
3. Apply **KL-Divergence Regularization** against the base model predictions during training.

---
---

## Module 2: Preference Alignment (RLHF, DPO, & Beyond)

### 1. Motivation & Intuition

SFT teaches a model how to follow instructions, but it fails to instill safety, ethics, and subtle human preference nuances. If an SFT model is asked "How do I hotwire a car?", and its training data contained crime thriller novels, it will gladly provide step-by-step instructions.

**Preference Alignment** bridges the gap between *coherence* and *human utility*. Instead of specifying "predict the next statistical word," we provide comparative supervision: "Between Response A and Response B, Response A is safer, more accurate, and more helpful."

---

### 2. Conceptual Foundations

#### RLHF Architecture (Reinforcement Learning from Human Feedback)
1. **SFT Base Model ($\pi^{\text{SFT}}$):** The baseline instruction-following model.
2. **Reward Model ($r_\phi$):** A scalar model initialized from the SFT checkpoint. The final language modeling head is replaced with a linear regression head that outputs a single numerical quality score for a given `(Prompt, Response)` pair.
3. **Active Policy ($\pi_\theta$):** The model being actively optimized to maximize the reward score.
4. **Reference Policy ($\pi_{\text{ref}}$):** A frozen copy of the initial SFT model used to compute behavioral drift constraints.

#### Direct Preference Optimization (DPO) Paradigm
DPO eliminates the fragile, computationally expensive RL loop (actor-critic networks, rollout generation). It leverages an exact analytical re-parameterization of the reward function to optimize the policy directly on preference rankings using standard cross-entropy loss.

#### KTO & ORPO Alternatives
* **KTO (Kahneman-Tversky Optimization):** Alignment framework utilizing unpaired binary signals ($\text{True}/\text{False}$ thumbs up/down) grounded in behavioral economics prospects.
* **ORPO (Odds Ratio Preference Optimization):** Combines SFT and alignment into a single training stage by penalizing rejected generations via odds-ratio log-likelihood penalties directly during instruction tuning.

---

### 3. Mathematical Formulation

#### The Bradley-Terry Preference Model
To train the Reward Model $r_\phi(x, y)$, we assume the human preference distribution over two responses follows the Bradley-Terry model. Given prompt $x$, winning response $y_w$, and losing response $y_l$:

$$
P(y_w \succ y_l \mid x) = \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right)
$$

Where $\sigma(z) = \frac{1}{1 + e^{-z}}$. The reward model is trained by minimizing the negative log-likelihood:

$$
\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) \right]
$$

#### RLHF (PPO) Objective Function
The active policy $\pi_\theta$ is optimized to maximize expected reward while penalized by the Kullback-Leibler (KL) divergence from the reference policy $\pi_{\text{ref}}$:

$$
\max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y \mid x)} \left[ r_\phi(x, y) \right] - \beta \mathbb{D}_{KL} \left( \pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x) \right)
$$

#### Direct Preference Optimization (DPO) Derivation
By deriving the optimal policy $\pi^*(y \mid x)$ analytically from the RL objective, we can isolate and express the underlying reward in terms of policy log-ratios:

$$
r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Substituting this identity into the Bradley-Terry log-likelihood objective cancels the intractable partition function $Z(x)$, yielding the closed-form DPO loss:

$$
\mathcal{L}_{DPO}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

* **Intuition:** The gradient increases the conditional likelihood of preferred completions $y_w$ and decreases rejected completions $y_l$, dynamically scaled by how strongly the implicit reward model misclassifies the preference pair. $\beta$ controls the strength of the reference prior constraint.

---

### 4. Worked Examples

#### Conceptual DPO Gradient Dynamics
Consider the prompt $x$: *"Explain quantum computing."*
* $y_w$: Clear, accurate, accessible summary.
* $y_l$: Confusing, highly jargon-laden text containing factual errors.

1. **Forward Pass Evaluation:**
   Compute log probabilities under active policy $\pi_\theta$ and frozen reference $\pi_{\text{ref}}$:
   * Active policy implicit reward: $\hat{r}_\theta(x, y_w) = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}$
   * Active policy implicit reward: $\hat{r}_\theta(x, y_l) = \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$

2. **Error Signal Dynamics:**
   If $\hat{r}_\theta(x, y_l) > \hat{r}_\theta(x, y_w)$ (model prefers the bad response), the argument inside the sigmoid becomes negative. The error signal approaches maximum magnitude, driving aggressive weight updates to suppress $\pi_\theta(y_l \mid x)$.

3. **Convergence:**
   Once $\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$, the sigmoid saturates near $1$, gradient updates diminish near zero, and stability is achieved without explicit value functions.

---

### 5. Relevance to Machine Learning Practice

#### System Architecture Selection
* **PPO Frameworks:** High engineering complexity. Requires coordinating 4 concurrent models in GPU cluster memory (Actor, Critic/Value, Reward Model, Reference Policy). Highly susceptible to reward hacking and training instability.
* **DPO Frameworks:** Industry standard for post-training. Requires only 2 models (Active Policy and Reference Policy). Executes like standard supervised classification, utilizing standard memory-efficient optimization paradigms (DeepSpeed ZeRO, FSDP).

---

### 6. Common Pitfalls & Misconceptions

* **Length Exploitation (Verbosity Bias):** Reward models frequently correlate sequence length with intelligence. PPO and DPO optimization loops quickly learn to generate long-winded, repetitive summaries to artificially inflate rewards.
* **Distributional Collapse:** Setting the KL constraint parameter $\beta$ too low causes the model to concentrate all probability mass onto a narrow set of "safe" phrases, destroying creative generation capabilities and vocabulary diversity.

---

### Interview Preparation: Preference Alignment

#### Foundational
**Q: Why can we not simply train models on "good" responses using SFT instead of introducing RLHF or DPO?** **A:** SFT is fundamentally imitation learning—it minimizes cross-entropy loss over a static distribution. It cannot distinguish *why* a demonstration is good, nor can it extrapolate beyond the demonstration quality. Preference alignment operates on comparative signals, enabling models to optimize generalized latent concepts of usefulness, truthfulness, and safety.

#### Mathematical
**Q: Explain the exact mechanistic function of the KL-divergence penalty $\mathbb{D}_{KL}(\pi_\theta \parallel \pi_{\text{ref}})$ in alignment objectives.** **A:** The KL penalty serves as a continuous information-theoretic tether. Imperfect reward models (or static DPO preference datasets) contain out-of-distribution blind spots. Without regularization, the optimizer will find degenerate adversarial sequences ("reward hacking") that achieve infinite reward scores. The KL penalty forces the active policy to remain within the trusted manifold of natural language established during pre-training.

#### Applied
**Q: During DPO training, your validation loss is decreasing, but human evaluators report the generation quality has degraded into repetitive boilerplate. What is happening?** **A:** The model is overfitting to the preference margins. In DPO, minimizing loss can occur by driving the probability of rejected responses $\pi_\theta(y_l \mid x) \to 0$ faster than increasing $\pi_\theta(y_w \mid x)$. To fix this:
1. Increase the $\beta$ regularization parameter to tighten reference constraints.
2. Add a positive log-likelihood SFT loss term directly to the winning completion $y_w$ (NLL regularization).

---
---

## Module 3: Evaluation Frameworks

### 1. Motivation & Intuition

Building an LLM without structured evaluation is equivalent to engineering a jet engine and declaring it airworthy based on engine roar alone. Because generation tasks are open-ended and highly non-deterministic, exact-match metrics (like BLEU or ROUGE) correlate poorly with human judgment. Comprehensive evaluation frameworks establish objective verification standards across distinct cognitive domains.

---

### 2. Conceptual Foundations

#### Evaluation Taxonomy
1. **Static Academic Benchmarks:** Curated datasets measuring deterministic knowledge retention and multi-step deductive logic.
2. **LLM-as-a-Judge:** Utilizing frontier neural models (e.g., GPT-4, Claude 3.5 Sonnet) to dynamically score open-ended generations via pairwise comparisons or single-point rubrics.
3. **Human Preference Studies:** Blind Elo-rating arenas (e.g., Chatbot Arena) capturing statistical consensus from diverse human cohorts.

#### Critical Benchmark Profiles
* **MMLU (Massive Multitask Language Understanding):** 57 STEM, humanities, and social science subject areas testing zero-shot and few-shot knowledge retrieval.
* **GSM8K (Grade School Math 8K):** Multi-step arithmetic word problems requiring explicit chain-of-thought mathematical reasoning.
* **HumanEval:** Docstring-to-code generation benchmark evaluated via sandboxed execution of unit tests ($Pass@k$ metric).

---

### 3. Mathematical Formulation

#### The $Pass@k$ Metric (HumanEval)
To evaluate functional code generation correctness without statistical bias from finite sampling trials, we define the unbiased estimator $Pass@k$. Given $n$ total generated code samples per problem, where $c$ samples pass all unit tests:

$$
Pass@k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

* **Intuition:** Calculates the true probability that at least one functional program is executed within $k$ generated attempts, correcting for variance when $k < n$.

#### Inter-Annotator Agreement: Cohen's Kappa ($\kappa$)
When validating LLM-as-a-Judge alignment against human grading panels, we measure categorical reliability beyond chance agreement:

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Where $p_o$ is the observed relative observed agreement among raters, and $p_e$ is the hypothetical probability of chance agreement.

---

### 4. Worked Examples

#### Calculating $Pass@1$ and $Pass@2$
An LLM generates $n = 5$ distinct Python functions for a sorting algorithm. Running unit tests reveals $c = 2$ functions are completely correct, and $3$ fail edge cases.

1. **Calculate $Pass@1$:**

   $$Pass@1 = 1 - \frac{\binom{5 - 2}{1}}{\binom{5}{1}} = 1 - \frac{\binom{3}{1}}{5} = 1 - \frac{3}{5} = 0.40 \text{ (40% accuracy)}$$

2. **Calculate $Pass@2$:**

   $$Pass@2 = 1 - \frac{\binom{5 - 2}{2}}{\binom{5}{2}} = 1 - \frac{\binom{3}{2}}{10} = 1 - \frac{3}{10} = 0.70 \text{ (70% accuracy)}$$

---

### 5. Relevance to Machine Learning Practice

#### Automated CI/CD Pipelines
Modern enterprise deployment stacks integrate automated LLM evaluation gates. Before a fine-tuned checkpoint (e.g., LoRA weights) is pushed to production, automated suites execute GSM8K and domain-specific regression tests. If performance drifts $>2\%$ below baseline metrics, deployment is blocked.

---

### 6. Common Pitfalls & Misconceptions

* **Benchmark Contamination (Data Leakage):** Frontier models pre-trained on web-scale scrapes frequently ingest public benchmark test sets. High scores reflect memorization rather than generalization.
* **Self-Enhancement Bias:** LLM judges systematically assign higher scores to generations produced by their own architectural family (e.g., GPT-4 judging GPT-4 outputs favorably over Claude outputs).
* **Position Bias:** In pairwise comparison setups (`Response A` vs. `Response B`), automated LLM judges exhibit statistical preference for whichever response is presented first in the prompt context.

---

### Interview Preparation: Evaluation Frameworks

#### Foundational
**Q: Why do traditional NLP metrics like BLEU and ROUGE fail when evaluating modern Large Language Models?** **A:** BLEU and ROUGE rely on exact n-gram token overlap between candidate generations and reference texts. Modern LLMs generate structurally diverse, paraphrased, and highly nuanced completions that are semantically correct but lexically divergent from static references. They penalize synonyms and creative phrasing.

#### Mathematical
**Q: Derive why naive empirical calculation of $Pass@k$ (counting successful runs over total runs) is statistically biased.** **A:** If we naively generate $k$ samples and check if any pass, the expected value of that success rate varies depending on the random seed across small sample sizes. The hypergeometrical formulation $\left(1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right)$ calculates the exact mathematical expectation across all possible combinations of drawing $k$ samples from a larger generation pool $n$, yielding an invariant, unbiased estimator.

#### Debugging
**Q: Your newly aligned chat model scores 88% on GSM8K but fails simple multi-step deduction queries from internal beta testers. Identify the failure mode.** **A:** This is benchmark overfitting via stylistic contamination. The model has internalized the specific structural syntax of GSM8K word problems during training. When presented with unstructured, noisy human prompts that deviate from the benchmark distribution, the fragile reasoning chains collapse. Resolution requires evaluating via **LLM-as-a-Judge** on held-out, adversarial prompt sets.