# Natural Language Processing — A Rigorous Learning Guide

*A standalone reference covering preprocessing, representations, sequence labeling, classification, generation, information extraction, QA, summarization, and machine translation. Each topic is structured as: Motivation & Intuition → Conceptual Foundations → Mathematical Formulation → Worked Examples → Relevance to ML Practice → Common Pitfalls, followed by a tiered interview question bank.*

---

## Table of Contents

1. [Text Preprocessing: Tokenization, Stemming, Lemmatization](#1-text-preprocessing)
2. [Static Word Representations: Word2Vec, GloVe, FastText](#2-static-word-representations)
3. [Contextual Representations: ELMo, BERT, GPT, Sentence Transformers](#3-contextual-word-representations)
4. [Sequence Labeling: NER, POS, Chunking (CRF, BiLSTM-CRF)](#4-sequence-labeling)
5. [Text Classification: Hierarchical Attention, Transformer Fine-tuning](#5-text-classification)
6. [Text Generation: Decoding Strategies, Controllable Generation](#6-text-generation)
7. [Information Extraction: Relation, Event, Open IE](#7-information-extraction)
8. [Question Answering: Extractive, Generative, Reading Comprehension](#8-question-answering)
9. [Summarization: Extractive (TextRank), Abstractive (Seq2Seq)](#9-summarization)
10. [Machine Translation: Attention, Back-translation, Multilingual Models](#10-machine-translation)

---

# 1. Text Preprocessing

*Tokenization (WordPiece, SentencePiece, BPE), stemming, lemmatization.*

## 1.1 Motivation & Intuition

Text is not a native data type for neural networks. A model cannot multiply the string `"machine learning"` by a weight matrix. Before any learning happens, we must convert text into a sequence of integer IDs, each of which indexes into a learned embedding vector. **Tokenization** is the process of splitting a string into those indexable units, and **normalization** (stemming, lemmatization, lowercasing, Unicode normalization) is the process of collapsing surface variations so that related forms share parameters.

Consider a naive approach: use whole words as tokens. Immediately you face three problems:

1. **Vocabulary explosion.** English has ~170,000 common words; add inflections, proper nouns, numbers, URLs, typos, hashtags, and code snippets, and you easily reach tens of millions of distinct strings. Each one needs its own embedding row, which is expensive in memory and starves rare words of training signal.

2. **The out-of-vocabulary (OOV) problem.** At inference time, a user types `"transformerize"`. If that exact string was never in training data, a word-level model has no embedding for it and must fall back to a generic `<UNK>` token, losing all information.

3. **Morphological waste.** The words `run`, `runs`, `running`, `ran` share a core meaning, yet word-level tokenization treats them as four independent symbols with no shared parameters.

Subword tokenization — WordPiece, BPE, SentencePiece — solves all three. It splits rare or unseen words into frequent subword pieces (`transformer` + `##ize`), keeping vocabulary at a manageable ~30k–250k tokens while guaranteeing that *any* string can be encoded as a sequence of known pieces. This is the reason modern LLMs generalize to words they have never seen as whole units: they have seen the pieces.

**Real-world ML impact:** Tokenization choices determine sequence length (which drives quadratic attention cost), vocabulary size (which drives embedding-table memory), cross-lingual transfer (a shared multilingual BPE vocabulary is the backbone of mBERT and XLM-R), and downstream task performance (a poor tokenizer can make a numeric reasoning benchmark impossible because it splits `1234567` inconsistently).

## 1.2 Conceptual Foundations

### Key terms

- **Token**: an atomic unit of text fed to the model. Can be a character, subword, word, or whole sentence depending on scheme.
- **Vocabulary ($V$)**: the finite set of tokens the model knows. Every input must be expressible as a sequence of elements of $V$.
- **Corpus**: the text collection used to *learn* the tokenizer (for data-driven schemes like BPE).
- **Encoding**: the function $\text{enc}: \text{string} \to \mathbb{Z}^n$ mapping a string to a sequence of token IDs.
- **Decoding**: the inverse $\text{dec}: \mathbb{Z}^n \to \text{string}$. Must be lossless for a well-designed tokenizer.
- **Normalization**: pre-tokenization transformations — lowercasing, Unicode NFKC normalization, accent stripping, whitespace collapsing.
- **Pre-tokenization**: a coarse split (usually on whitespace and punctuation) that constrains what the subword algorithm can merge.
- **Stemming**: crude suffix stripping (`running` → `run`, `flies` → `fli`) using rule lists. Fast, lossy, language-specific.
- **Lemmatization**: mapping a word to its dictionary form (`running` → `run`, `was` → `be`) using a morphological analyzer and often part-of-speech information. Slower, more accurate.

### How subword tokenization works

All modern subword algorithms follow the same template:

1. Start with a base vocabulary (characters or bytes).
2. Iteratively grow the vocabulary by merging or adding the subword that maximizes some objective on the training corpus.
3. Stop when the vocabulary reaches a target size.
4. At inference, encode each word by greedily or optimally segmenting it using the learned vocabulary.

The three dominant variants differ in the merge criterion and segmentation rule:

| Scheme | Merge rule | Segmentation | Used by |
|---|---|---|---|
| **BPE** (Byte-Pair Encoding) | Merge the most frequent adjacent pair | Greedy left-to-right, applying merges in the order learned | GPT-2/3/4, RoBERTa |
| **WordPiece** | Merge the pair whose merge maximizes corpus likelihood (equivalently, maximizes pointwise mutual information) | Greedy longest-match | BERT, DistilBERT |
| **Unigram LM** | Start large, iteratively remove tokens whose removal least hurts corpus likelihood | Probabilistic — Viterbi over all segmentations | T5, XLNet (via SentencePiece) |

**SentencePiece** is not a separate algorithm but a *framework* that runs BPE or Unigram LM directly on raw Unicode text without requiring pre-tokenization. It treats whitespace as a regular character (rendered as `▁`), which makes it language-agnostic — crucial for Chinese, Japanese, and Thai where words are not whitespace-separated.

### Stemming vs. lemmatization

- **Porter stemmer** applies ~60 rules like "if word ends in `sses`, replace with `ss`". Operates without dictionary, context, or POS.
- **Lemmatizer** (e.g., spaCy, WordNet) looks up the word in a morphological database, optionally using POS: `saw` as a noun lemmatizes to `saw`, as a verb to `see`.

### Assumptions and failure modes

**Assumption 1: The training corpus is representative of inference-time text.**
Violated when you train BPE on English Wikipedia and deploy on Twitter. Tokens like `lol`, `smh`, or emoji will be over-fragmented, ballooning sequence length and hurting accuracy.

**Assumption 2: Whitespace or a pre-tokenizer provides meaningful boundaries.**
Violated for Chinese, Japanese, Thai, Khmer — no spaces exist. Use SentencePiece (which treats whitespace as a learnable token) or a language-specific segmenter.

**Assumption 3: The vocabulary is large enough to cover domain terminology.**
Violated in biomedical or legal domains: `pneumonoultramicroscopicsilicovolcanoconiosis` will be split into a dozen pieces, harming both speed and the model's ability to recognize it as a unit. Domain-adapted tokenizers (SciBERT, BioBERT) retrain the tokenizer on domain text.

**Assumption 4: Normalization is reversible or lossless for the task.**
Violated when you lowercase an NER corpus: you destroy the strongest signal for proper-noun detection (`Apple` vs. `apple`). Violated when you stem a search index: `university` and `universal` both stem to `univers`, collapsing distinct concepts.

## 1.3 Mathematical Formulation

### BPE — formal algorithm

Let $\mathcal{C}$ be a corpus of words, each represented as a sequence of characters terminated by an end-of-word symbol `</w>`. Let $V_0$ be the set of all characters appearing in $\mathcal{C}$. We build the vocabulary iteratively:

$$
V_{t+1} = V_t \cup \{(a, b)^*\}, \quad (a, b)^* = \arg\max_{(a,b) \in V_t \times V_t} \text{freq}(a, b; \mathcal{C}_t)
$$

where $\text{freq}(a, b; \mathcal{C}_t)$ is the count of adjacent occurrences of tokens $a$ and $b$ in the corpus *after* previous merges have been applied, and $(a,b)^*$ denotes the new merged token. We also record the merge rule $a + b \to ab$.

Stop when $|V_t|$ reaches the target vocabulary size. To tokenize a new word, split it into characters and repeatedly apply merge rules in the order they were learned.

### WordPiece — likelihood-based merges

WordPiece chooses the merge that maximizes the *corpus log-likelihood* under a unigram language model. For candidate merge $(a, b)$:

$$
\text{score}(a, b) = \log \frac{p(ab)}{p(a)\, p(b)} \;\propto\; \log \frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}
$$

This is pointwise mutual information. It prefers merges of genuinely associated pairs over merely frequent pairs — e.g., it will not merge `the` and `cat` just because `the cat` occurs often, because `the` is already very common.

### Unigram LM — probabilistic segmentation

Treat each segmentation $\mathbf{x} = (x_1, \dots, x_n)$ of a word as a sample from a unigram model:

$$
p(\mathbf{x}) = \prod_{i=1}^n p(x_i), \quad x_i \in V
$$

Training maximizes the marginal likelihood over all segmentations using EM:

$$
\mathcal{L} = \sum_{w \in \mathcal{C}} \log \sum_{\mathbf{x} \in S(w)} p(\mathbf{x})
$$

where $S(w)$ is the set of valid segmentations of word $w$. After training, the best segmentation is found by Viterbi:

$$
\mathbf{x}^* = \arg\max_{\mathbf{x} \in S(w)} \sum_{i=1}^n \log p(x_i)
$$

Unigram LM enables **subword regularization**: at training time you sample segmentations from $p(\mathbf{x} \mid w)$ rather than always taking the best one, which acts like dropout on the input side and improves robustness.

### Each math term, intuitively

- $\text{freq}(a,b)$: "how often do these two pieces sit next to each other" — BPE is greedy about compression.
- $\log p(ab)/(p(a)p(b))$: "how surprised would we be by this pair if the pieces were independent" — high when pieces genuinely co-occur.
- Marginal likelihood $\sum_{\mathbf{x}} p(\mathbf{x})$: "a word's probability summed over all ways to split it" — lets unigram LM handle ambiguous segmentations gracefully.

## 1.4 Worked Example

**Toy BPE on a corpus of five words.**

Corpus (with counts):

```
low    : 5
lower  : 2
newest : 6
widest : 3
```

Initialize each word as a space-separated character sequence with `</w>`:

```
l o w </w>      : 5
l o w e r </w>  : 2
n e w e s t </w>: 6
w i d e s t </w>: 3
```

Initial vocabulary $V_0 = \{l, o, w, e, r, n, s, t, i, d, \langle/w\rangle\}$, size 11.

**Iteration 1.** Count adjacent pairs across the corpus, weighted by word frequency:

| Pair | Count |
|---|---|
| (e, s) | $6 + 3 = 9$ |
| (s, t) | $6 + 3 = 9$ |
| (t, `</w>`) | $6 + 3 = 9$ |
| (l, o) | $5 + 2 = 7$ |
| (o, w) | $5 + 2 = 7$ |
| ... | ... |

Tie among three pairs at count 9. Break ties alphabetically; merge `(e, s) → es`. Add `es` to vocabulary.

```
l o w </w>      : 5
l o w e r </w>  : 2
n e w es t </w> : 6
w i d es t </w> : 3
```

**Iteration 2.** Recount:

| Pair | Count |
|---|---|
| (es, t) | $6 + 3 = 9$ |
| (t, `</w>`) | $9$ |
| (l, o) | $7$ |

Merge `(es, t) → est`.

```
l o w </w>      : 5
l o w e r </w>  : 2
n e w est </w>  : 6
w i d est </w>  : 3
```

**Iteration 3.** Top pair is `(est, </w>) → est</w>` (count 9). Continue for as many merges as desired.

**Inference.** Encode the new word `newer`:
- Start: `n e w e r </w>`
- Apply learned merges in order. We have `es → est → est</w>`, but no pair in `newer` matches — none of our merges apply because `newer` contains no `es`.
- Further iterations would have learned `(l, o) → lo` and `(lo, w) → low`, at which point `lower` would become `low e r </w>`.

This illustrates how BPE produces compact, data-driven segmentations and how unseen words decompose into known pieces.

## 1.5 Relevance to ML Practice

**Where it appears in the pipeline.**

- **Training.** The tokenizer is chosen *before* pretraining and is essentially frozen. Changing vocabulary after pretraining invalidates the embedding table and requires retraining or careful embedding-transfer tricks.
- **Inference.** Input text is tokenized, mapped to IDs, fed through the model. For generation, the model outputs token IDs which are detokenized back to text.
- **Evaluation.** Token-level metrics like BLEU or perplexity are tokenizer-dependent. Comparing two models with different tokenizers on raw perplexity is meaningless; you must use a neutral evaluation tokenizer or a shared metric like character-level bits-per-byte.
- **Monitoring.** A spike in average tokens-per-input at inference time is a strong signal of distribution shift — your tokenizer is encountering unfamiliar text (new jargon, a new language, corrupted encoding).

**When to use which tokenizer.**

- **BPE** — default choice for English-centric or majority-English multilingual models. Fast, deterministic, well-understood.
- **WordPiece** — when you want slightly better compression via PMI-based merges and are willing to use greedy longest-match segmentation (BERT-family convention).
- **SentencePiece + Unigram LM** — best for multilingual models, languages without whitespace (Japanese, Chinese, Thai), and when you want subword regularization at training time.
- **Byte-level BPE** — when you cannot afford OOV at all (any byte is a valid token) and are willing to pay longer sequence lengths. GPT-2/3 use this.
- **Character-level** — for morphologically very rich languages, noisy user-generated text, or when vocabulary size is extremely constrained.

**When *not* to use subword tokenization.**

- For strict symbolic tasks (arithmetic, code execution), character-level or digit-level tokenization often outperforms BPE, which can split `12345` inconsistently depending on neighboring context.
- For phoneme- or byte-level tasks (speech, low-level code), use raw bytes/phonemes.

**Stemming/lemmatization in modern pipelines.**

With contextual embeddings, stemming and lemmatization are mostly obsolete for deep-learning pipelines — the model learns that `run`/`running`/`ran` are related through distributional co-occurrence. They remain valuable for:
- Classical IR (BM25, TF-IDF) where surface form matters.
- Rule-based systems and small models where vocabulary reduction is essential.
- Linguistic analysis and feature engineering for interpretable models.

**Trade-offs.**

| Axis | Character | Subword | Word |
|---|---|---|---|
| Vocabulary size | Tiny (~100) | Moderate (~30k) | Large (~500k+) |
| Sequence length | Longest | Medium | Shortest |
| OOV handling | Perfect | Excellent | Poor |
| Semantic unit | None | Partial | Full |
| Compute cost | Highest | Medium | Lowest |

## 1.6 Common Pitfalls

1. **Training tokenizer on one domain, deploying on another.** Tokens are silently over-fragmented and sequence lengths explode. Always measure tokens/word on a holdout of the target distribution.

2. **Forgetting Unicode normalization.** `café` can be encoded as `cafe + ́` (NFD) or `café` (NFC). Without NFKC normalization you will have two distinct tokens for the same visible string.

3. **Stripping case for NER or proper-noun-heavy tasks.** `US` the country vs. `us` the pronoun become indistinguishable. Use a cased tokenizer.

4. **Comparing perplexity across tokenizers.** A model with a larger vocabulary generally gets lower perplexity not because it is better but because each token carries more information. Always use bits-per-character or bits-per-byte for cross-tokenizer comparison.

5. **Confusing stemming with lemmatization.** Stemming produces non-words (`univers`, `fli`); lemmatization produces valid dictionary forms. Stemming is faster but lossy.

6. **Numeric tokenization.** Default BPE splits numbers unpredictably (`1234` might be one token, `1235` three). For arithmetic-heavy tasks, force digit-level tokenization or use a tokenizer trained with explicit number-splitting rules.

7. **Leaking test-set tokens into tokenizer training.** Train the tokenizer only on the training split — otherwise you bias evaluation.

8. **Assuming the tokenizer is fast.** For very long inputs, tokenization itself can become a bottleneck. Use the HuggingFace `tokenizers` Rust library, not pure Python.

---

## 1.7 Interview Questions: Text Preprocessing

### Foundational

**Q1. What is a token, and why can't we just use whole words?**
A token is an atomic unit the model operates on — a character, subword, or word. Whole-word tokenization gives an enormous vocabulary (hundreds of thousands of words, exploding with proper nouns, numbers, and typos), cannot handle out-of-vocabulary words at inference, and fails to share parameters across morphological variants. Subword schemes solve all three by using a fixed-size vocabulary of frequent pieces that can compose into any word.

**Q2. Explain the difference between stemming and lemmatization.**
Stemming is a crude rule-based suffix stripper (Porter, Snowball) that often produces non-words (`studies → studi`). Lemmatization uses a dictionary and often POS tags to map a word to its canonical form (`studies → study`, `was → be`). Lemmatization is slower, more accurate, and language-specific; stemming is fast and language-specific but lossy. In modern neural pipelines with contextual embeddings, neither is commonly used — the model learns inflectional relationships implicitly. They persist in classical IR and rule-based systems.

**Q3. What is the difference between BPE, WordPiece, and SentencePiece?**
BPE and WordPiece are algorithms; SentencePiece is a framework that implements BPE or Unigram LM directly on raw bytes without requiring whitespace pre-tokenization. BPE merges the *most frequent* adjacent pair. WordPiece merges the pair with highest *pointwise mutual information* (highest likelihood gain under a unigram LM). Unigram LM (inside SentencePiece) goes the other direction: it starts with a large vocabulary and *prunes* tokens whose removal least hurts likelihood, then segments probabilistically via Viterbi.

**Q4. Why does BERT use `##` prefixes?**
WordPiece marks non-initial subwords with `##` so the detokenizer knows whether to prepend a space. `un ##friend ##ly` reconstructs to `unfriendly`, while `un friendly` (no `##`) reconstructs to `un friendly` with a space. This preserves round-trip fidelity.

### Mathematical

**Q5. Derive the WordPiece merge score.**
Under a unigram LM, the corpus likelihood is $\prod_i p(x_i)$. Merging tokens $a, b$ into a new token $ab$ changes the likelihood of sequences containing the pair $a, b$. The log-likelihood gain per occurrence is $\log p(ab) - \log p(a) - \log p(b)$, which is exactly pointwise mutual information. WordPiece picks the merge maximizing total log-likelihood gain, equivalent to picking the merge with highest $\log p(ab) / (p(a)p(b))$.

**Q6. Given a unigram LM tokenizer with probabilities $p(\text{un}) = 0.01$, $p(\text{happy}) = 0.005$, $p(\text{unhappy}) = 0.003$, which segmentation of "unhappy" does Viterbi choose?**
Compare $\log p(\text{unhappy}) = \log 0.003 \approx -5.81$ against $\log p(\text{un}) + \log p(\text{happy}) = \log 0.01 + \log 0.005 \approx -4.61 - 5.30 = -9.91$. The single-token segmentation wins: higher log-probability. This illustrates why Unigram LM prefers longer tokens when they are frequent enough.

**Q7. What is subword regularization and how does it relate to dropout?**
At training time, rather than using the single best segmentation, sample from $p(\mathbf{x} \mid w) \propto \prod_i p(x_i)$ over all segmentations of word $w$. This exposes the model to multiple valid segmentations of the same word, acting as a regularizer analogous to dropout on input. It is specific to Unigram LM (BPE-dropout is the BPE analog: randomly skip merges with some probability). Empirically improves robustness and low-resource NMT.

**Q8. Prove that BPE tokenization is deterministic given the merge table.**
Merges are applied in the fixed learned order. At each step, the leftmost applicable merge of highest priority (earliest learned) is applied. The merge table is a total order; given a starting character sequence, applying all possible merges in order until none remains yields a unique segmentation. Hence the encoding is a deterministic function of the input string and merge table. (Contrast with Unigram LM, which is only deterministic if you take argmax.)

### Applied

**Q9. Your multilingual model performs poorly on Japanese despite a large shared vocabulary. Debug.**
Likely causes: (a) the tokenizer was trained with Latin-script pre-tokenization that strips Japanese into characters, ballooning sequence length; (b) the shared vocabulary allocates few tokens to Japanese pieces because training data is English-heavy (known as "the curse of multilinguality"). Fixes: use SentencePiece (no whitespace assumption), upsample Japanese in tokenizer training, or use a language-specific subword budget (mBERT's vocabulary allocation by language frequency).

**Q10. Why might forcing digit-level tokenization improve arithmetic performance?**
Default BPE might tokenize `12345` as `[123, 45]` and `12346` as `[12, 346]` depending on frequency. The model has no way to know these integers are "close". Splitting every digit into its own token gives a consistent positional representation — `[1,2,3,4,5]` vs `[1,2,3,4,6]` — and allows the model to learn digit-wise arithmetic. Models like Llama-3 and Qwen use this trick.

**Q11. You train a tokenizer on clean Wikipedia text and deploy the model on noisy user chat. What do you observe and how do you fix it?**
Observation: average tokens per message explodes; rare symbols, slang, emoji, repeated punctuation each fragment into many pieces; inference latency increases; model quality drops because context windows are consumed by tokenizer inefficiency. Fixes: retrain tokenizer on representative data; fine-tune with a domain-adapted vocabulary; or perform continued pretraining with an expanded vocabulary (keeping existing rows, appending new ones).

### Debugging & Failure Modes

**Q12. After continued pretraining on medical text, you add 5,000 new tokens to the vocabulary. Why do outputs now contain gibberish?**
Newly added embedding rows are initialized randomly and have *never been trained*. The model produces them with near-random logits. Fix: initialize each new row as the mean (or more cleverly, the weighted average) of the subwords it replaces, then do several epochs of continued pretraining before using it. Additionally, freeze the rest of the model initially so the new rows catch up.

**Q13. Two engineers compare perplexity on the same test set. Engineer A uses a 50k-vocab BPE tokenizer, Engineer B uses a 250k-vocab word tokenizer. B reports lower perplexity. Is B's model better?**
Not necessarily. Perplexity is defined over tokens. A larger vocabulary means each token carries more information, so the per-token probability distribution is sharper. To compare fairly, normalize to bits-per-character or bits-per-byte: $\text{bpc} = \frac{1}{|\text{chars}|} \sum -\log_2 p(\text{token}_i)$. This is tokenizer-agnostic and is the standard for cross-model comparison.

**Q14. Your NER model fine-tuned on a BERT tokenizer fails on a company name like `Acme-Tech`. Explain and fix.**
BERT's WordPiece likely splits `Acme-Tech` into multiple pieces: `A ##c ##me - Te ##ch`. If the NER labels are assigned at the *word* level but predictions are made per token, label alignment can be ambiguous. Also, the hyphen is often a standalone token with no entity context. Fix: assign the entity label to the *first* subword of each word, ignore subsequent subwords in the loss (standard HuggingFace `-100` convention), and ensure the tokenizer's `is_split_into_words=True` API is used for alignment.

### Follow-up Probes

**Q15. If you had infinite compute, would you prefer character-level or byte-level models?**
Byte-level is strictly more general — every possible string is representable, including non-Unicode garbage. Character-level requires a Unicode code-point assumption and is slightly shorter on European languages but longer on Asian scripts. With infinite compute, byte-level wins for maximum robustness (ByT5, Charformer). With finite compute, subword is the practical sweet spot.

**Q16. Why is BPE-dropout less common than Unigram LM subword regularization?**
BPE-dropout (skip a merge with probability $p$ during training) still produces deterministic segmentations given a fixed random seed per example, and the distribution it samples from is less principled than Unigram LM's explicit probabilistic model. Unigram LM gives you a proper probability distribution over segmentations, making ablations and theory cleaner. Empirically, both help, but Unigram LM is used more widely (T5, NLLB).

**Q17. Can the same tokenizer serve both a text model and a code model? Why do Copilot-style models often use distinct tokenizers?**
Code has very different statistics: long runs of whitespace, specific punctuation (`=>`, `::`, `->`), and meaningful indentation. A tokenizer trained on natural language will fragment `    ` (four-space indent) into multiple tokens, wasting context. Code tokenizers either include multi-space tokens explicitly or use a code-weighted corpus during BPE training. Modern LLMs (GPT-4, Claude) use tokenizers trained on a *mixture* of code and text, with explicit multi-whitespace tokens added.

---

# 2. Static Word Representations

*Word2Vec (CBOW, Skip-gram), GloVe, FastText.*

## 2.1 Motivation & Intuition

One-hot vectors treat every word as maximally distinct: $\cos(\text{cat}, \text{kitten}) = 0$ exactly as $\cos(\text{cat}, \text{economy}) = 0$. This is absurd — we know cats and kittens share almost everything. The insight of distributional semantics, going back to Firth (1957) — *"you shall know a word by the company it keeps"* — is that if two words appear in similar contexts, they probably mean similar things.

If we encode each word as a dense vector such that vectors of semantically similar words are geometrically close, we get three enormous benefits:

1. **Generalization.** A classifier trained on the phrase "buy a dog" can transfer to "purchase a puppy" because the vectors are close.
2. **Low-dimensional efficiency.** Replace $|V| = 100{,}000$-dimensional sparse vectors with $d = 300$ dense ones: 333× more compact and continuous.
3. **Algebraic structure.** Remarkably, these vectors support *arithmetic*: $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$. This was a shocking empirical finding of Word2Vec.

Static embeddings give every word a single vector regardless of context. This is their defining limitation — `bank` has one vector whether it means a river bank or a financial bank — and the motivation for contextual embeddings covered in §3. But they remain excellent as initialization, for low-compute deployments, and as building blocks for classical pipelines.

**Real-world ML impact.** Before Transformers dominated, static embeddings were the default first layer of nearly every NLP model. They still power word-similarity search, keyword expansion, bilingual lexicon induction, and many embedding-based retrieval systems with tight latency budgets.

## 2.2 Conceptual Foundations

### Key concepts

- **Distributional hypothesis**: a word's meaning is characterized by the distribution of contexts in which it appears.
- **Context window**: for a target word $w_t$, the window is the set of surrounding words $\{w_{t-c}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+c}\}$ for window size $c$.
- **Center/target word** vs. **context words**: which one the model predicts determines CBOW vs. Skip-gram.
- **Co-occurrence matrix**: an $|V| \times |V|$ matrix $X$ where $X_{ij}$ counts how often word $j$ appears in the context of word $i$. GloVe operates on this matrix directly.
- **Negative sampling**: instead of computing a full softmax over $|V|$ words, sample a small number of "negative" words per positive example and train a binary classifier. Drastically reduces training cost.
- **Subword embeddings** (FastText): represent a word as the sum of its character n-gram embeddings. Gives meaningful vectors to OOV words.

### Word2Vec variants

**Continuous Bag-of-Words (CBOW).** Predict the center word given the (averaged) context words. Fast, good for small datasets, tends to smooth over rare words.

**Skip-gram.** Predict each context word given the center word. Slower (many more training pairs), but produces better representations for rare words because each rare word generates many training examples (one per context word).

### GloVe (Global Vectors)

Word2Vec is *local*: each training example is a tiny window. GloVe is *global*: it fits embeddings directly to the full corpus-wide co-occurrence matrix. The key insight is that *ratios* of co-occurrence probabilities encode meaning. For words $i$ and $j$ and a probe word $k$:

- If $P(k \mid i)/P(k \mid j)$ is large, $k$ is more related to $i$ than $j$.
- If the ratio is ~1, $k$ is equally related (or unrelated) to both.

GloVe trains embeddings so dot products reproduce these log-ratios.

### FastText

FastText extends Word2Vec by representing each word as the set of its character n-grams. The word `where` with n-gram sizes 3–6 decomposes into `<wh`, `whe`, `her`, `ere`, `re>`, and longer grams. The word's embedding is the sum of its n-gram embeddings. Benefits:

- Morphologically related words (`play`, `playing`, `player`) share n-grams and therefore share representation.
- Any OOV word has a vector — just sum its n-gram embeddings.
- Works especially well for morphologically rich languages (Turkish, Finnish, Arabic).

### Assumptions and failure modes

**Assumption 1: One sense per word.**
Violated for polysemous words. `bank`, `apple`, `mouse` get a single vector that is a weighted average over senses — potentially meaningless. Contextual embeddings solve this.

**Assumption 2: Context window captures meaning.**
Violated for long-range dependencies (`not good` — the negation is local) or topic-level meaning (`apple` in a tech article vs. a cooking article). Window-based methods see only $\pm c$ words.

**Assumption 3: Co-occurrence statistics are rich enough.**
Violated for rare words with few occurrences — their vectors are unreliable. FastText partially addresses this via shared n-grams.

**Assumption 4: Training corpus is semantically unbiased.**
Violated everywhere. Embeddings encode societal biases: `man : doctor :: woman : nurse` is a classic demonstration.

## 2.3 Mathematical Formulation

### Skip-gram with full softmax

Given a corpus as a sequence $w_1, \dots, w_T$, maximize:

$$
\mathcal{L} = \frac{1}{T} \sum_{t=1}^T \sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p(w_{t+j} \mid w_t)
$$

with softmax parameterization:

$$
p(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} {}^\top \mathbf{v}_{w_I})}{\sum_{w=1}^{|V|} \exp(\mathbf{v}'_w {}^\top \mathbf{v}_{w_I})}
$$

where $\mathbf{v}_w$ is the "input" embedding (center word) and $\mathbf{v}'_w$ is the "output" embedding (context word). Every word has two vectors; after training, typically $\mathbf{v}_w$ is used as the final embedding.

### Skip-gram with negative sampling (SGNS)

The full softmax is $O(|V|)$ per example — infeasible for $|V| = 10^6$. Negative sampling replaces it with a binary classification: for each true (center, context) pair, sample $k$ random words as fake contexts and train the model to distinguish them.

$$
\log \sigma(\mathbf{v}'_{w_O} {}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}'_{w_i} {}^\top \mathbf{v}_{w_I})]
$$

The noise distribution $P_n(w) \propto U(w)^{3/4}$ where $U(w)$ is the unigram frequency — the $3/4$ exponent downweights very frequent words.

**Derivation intuition.** Skip-gram with negative sampling implicitly factorizes a matrix whose $(i,j)$ entry is the shifted pointwise mutual information:

$$
\mathbf{v}_i^\top \mathbf{v}'_j \approx \text{PMI}(i, j) - \log k
$$

This was shown formally by Levy and Goldberg (2014). So SGNS is essentially a low-rank factorization of a PMI-like matrix — connecting it directly to GloVe and classical distributional semantics.

### GloVe objective

Let $X_{ij}$ be the count of word $j$ in the context of word $i$. Define the conditional probability $P_{ij} = X_{ij} / X_i$ where $X_i = \sum_k X_{ik}$.

GloVe wants $\mathbf{v}_i^\top \mathbf{v}'_j + b_i + b'_j = \log X_{ij}$. The derivation from co-occurrence ratios is:

$$
F(\mathbf{v}_i - \mathbf{v}_j, \mathbf{v}'_k) = \frac{P_{ik}}{P_{jk}}
$$

Choosing $F(\mathbf{x}) = \exp(\mathbf{x})$ and adding biases yields the symmetric form above. The loss is weighted least squares:

$$
\mathcal{L} = \sum_{i,j=1}^{|V|} f(X_{ij}) \left(\mathbf{v}_i^\top \mathbf{v}'_j + b_i + b'_j - \log X_{ij}\right)^2
$$

The weighting function $f(x) = \min(1, (x/x_{\max})^{\alpha})$ with $\alpha = 3/4$, $x_{\max} = 100$ downweights very frequent pairs (which otherwise dominate the loss) and zeros out pairs never observed ($f(0) = 0$).

### FastText

Each word $w$ is represented as the sum over its character n-grams $G_w$:

$$
\mathbf{v}_w = \sum_{g \in G_w} \mathbf{z}_g
$$

Training objective is identical to Skip-gram negative sampling, but with $\mathbf{v}_w$ computed as above. Gradients propagate back to each $\mathbf{z}_g$.

### Mapping math back to intuition

- $\mathbf{v}_i^\top \mathbf{v}'_j$: similarity between word $i$'s input vector and word $j$'s output vector. High when they co-occur.
- $\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i)P(j)}$: how much more often they co-occur than chance. Word2Vec approximates this.
- $\log X_{ij}$ in GloVe: raw log-count target. Weighted regression avoids the infinity at $X_{ij} = 0$.
- N-gram sum in FastText: compositional word representation. Shared n-grams are the source of morphological transfer.

## 2.4 Worked Example

**Skip-gram with negative sampling on a tiny corpus.**

Corpus: `"the cat sat on the mat"`. Window size $c = 2$, embedding dim $d = 3$. Vocabulary: `{the, cat, sat, on, mat}`, indices 0–4.

**Step 1. Generate training pairs** (center, context):
- center `cat`: contexts `the, sat, on` → pairs `(cat, the), (cat, sat), (cat, on)`
- center `sat`: `(sat, the), (sat, cat), (sat, on), (sat, the)`
- ...and so on.

**Step 2. Initialize.** Every word gets two random 3-d vectors $\mathbf{v}_w, \mathbf{v}'_w \sim \mathcal{N}(0, 0.01)$.

**Step 3. For one training pair** `(cat, sat)`: sample $k=2$ negatives, say `the` and `mat`. The loss for this example is

$$
-\log \sigma(\mathbf{v}'_{\text{sat}} \cdot \mathbf{v}_{\text{cat}}) - \log \sigma(-\mathbf{v}'_{\text{the}} \cdot \mathbf{v}_{\text{cat}}) - \log \sigma(-\mathbf{v}'_{\text{mat}} \cdot \mathbf{v}_{\text{cat}})
$$

Compute gradients:
- Push $\mathbf{v}'_{\text{sat}}$ and $\mathbf{v}_{\text{cat}}$ *together* (positive).
- Push $\mathbf{v}'_{\text{the}}, \mathbf{v}'_{\text{mat}}$ *away* from $\mathbf{v}_{\text{cat}}$.

**Step 4. Iterate** over all pairs for many epochs. After training, $\mathbf{v}_{\text{cat}}$ and $\mathbf{v}_{\text{mat}}$ end up close because they share contexts like `the _ sat` and `on the _`; they are pushed away from `the` because `the` is an effective negative sample.

**Step 5. Query.** Cosine similarity:
$$
\cos(\mathbf{v}_{\text{cat}}, \mathbf{v}_{\text{mat}}) = \frac{\mathbf{v}_{\text{cat}} \cdot \mathbf{v}_{\text{mat}}}{\|\mathbf{v}_{\text{cat}}\| \|\mathbf{v}_{\text{mat}}\|}
$$
should be higher than $\cos(\mathbf{v}_{\text{cat}}, \mathbf{v}_{\text{the}})$.

**Analogy example.** On a real-world-trained 300d Word2Vec, the vector $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}}$ has `queen` as its nearest neighbor in cosine distance. The reason (informally): training forces $\vec{\text{king}} - \vec{\text{queen}} \approx \vec{\text{man}} - \vec{\text{woman}}$ because both differences capture the gender dimension extracted from co-occurrence patterns.

## 2.5 Relevance to ML Practice

**Where they appear.**

- **Initialization for neural models.** Pre-Transformer era, almost every NLP model initialized its embedding layer from Word2Vec or GloVe, then fine-tuned.
- **Feature engineering for classical ML.** Average or sum a document's word vectors to get a fixed-size document representation for logistic regression, random forests, or SVMs. Simple but surprisingly strong baseline.
- **Semantic search and retrieval** before dense sentence encoders. Embed query and documents as bag-of-vectors, use cosine similarity.
- **Bilingual lexicon induction.** Train monolingual embeddings in two languages, then find a linear mapping between the spaces (Mikolov 2013) for zero-shot translation of word lists.
- **Cold-start systems.** When you have no labeled data, static embeddings + nearest neighbor can bootstrap.

**When to use them.**

- Low-latency, low-memory systems where a BERT pass is too expensive.
- Tasks that truly are "bag of words" (topic modeling, simple classification).
- Initialization for small models on small data.
- Linguistic analysis and similarity tasks.

**When NOT to use them.**

- Any task requiring disambiguation of word senses (QA, NLI, sarcasm detection).
- Syntactic tasks requiring sensitivity to word order.
- Low-resource OOV-heavy settings — prefer FastText or, better, subword-based contextual models.

**Common alternatives.**

- **Contextual embeddings** (BERT, ELMo): every occurrence of a word gets its own vector. Discussed in §3.
- **Sentence embeddings** (Sentence-BERT, SimCSE): skip word-level entirely, embed whole sentences.
- **Learned from scratch** inside the task model — sometimes better than pretrained for specialized domains with enough data.

**Trade-offs.**

| Aspect | Word2Vec SGNS | GloVe | FastText |
|---|---|---|---|
| Training objective | Local prediction | Global matrix factorization | Local prediction + char n-grams |
| OOV handling | `<UNK>` | `<UNK>` | Composes from n-grams |
| Memory | $|V| \times d$ | $|V| \times d$ | $(|V| + |N|) \times d$ |
| Training speed | Fast (SGNS) | Faster (closed form per example) | Slower (n-gram lookups) |
| Morphology | None | None | Strong |
| Rare words | Weak | Weak | Strong |
| Syntactic tasks | Medium | Medium | Best |

## 2.6 Common Pitfalls

1. **Ignoring OOV.** Evaluating a Word2Vec model on a test set with many OOV tokens and blaming the model. OOV is a tokenizer/vocabulary issue; fix with FastText or subword embeddings.

2. **Assuming one vector captures all senses.** `java` the language, `java` the island, `java` the coffee — a static embedding averages them all into one weighted centroid. Use sense-disambiguated embeddings or contextual models.

3. **Measuring similarity with Euclidean distance.** Dot products and cosine similarity are the natural metrics (training objective uses them); Euclidean distance mixes norm and direction and is rarely what you want.

4. **Using unnormalized word frequency for negative sampling.** The $3/4$ power is empirical and critical; using raw unigram frequency makes `the`, `of` too likely as negatives, making training trivial and embeddings poor.

5. **Confusing GloVe and Word2Vec's "global" vs "local" dichotomy.** GloVe is global in the sense that it fits to aggregated counts, but the counts are still derived from the same local windowed co-occurrence. The difference is objective, not context.

6. **Bias amplification.** Embeddings trained on real text encode and often amplify societal biases. If you deploy them in a hiring system, you may see discriminatory recommendations. Debiasing methods (Bolukbasi 2016) exist but don't remove bias, only obscure it.

7. **Expecting analogies to always work.** $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ is real but cherry-picked; most analogies fail. Don't use analogy tests as the primary evaluation.

8. **Anisotropic embedding space.** Static embeddings (especially Skip-gram) concentrate in a narrow cone of the vector space; cosine similarities are all uniformly high. You may need to center or whiten them before downstream use.

---

## 2.7 Interview Questions: Static Word Representations

### Foundational

**Q1. Why do word embeddings work? What assumption do they rely on?**
They rely on the distributional hypothesis: words appearing in similar contexts have similar meanings. By forcing co-occurring words to have similar vectors, gradient descent discovers a geometry in which semantic similarity corresponds to vector similarity. This works because natural language exhibits strong regularities in word co-occurrence patterns that correlate with semantic relationships.

**Q2. Contrast CBOW and Skip-gram.**
CBOW predicts the center word from the average of context words — one training example per window, fast, smooths rare words. Skip-gram predicts each context word from the center — $2c$ training examples per window, slower, better for rare words (each rare word generates many training pairs). Skip-gram typically wins on downstream tasks; CBOW wins on speed.

**Q3. Why does FastText handle OOV better than Word2Vec?**
Word2Vec has no representation for words not in its vocabulary. FastText represents every word as the sum of its character n-gram embeddings, so any string — even one never seen in training — gets a vector by summing its n-grams' embeddings. Morphologically related words share n-grams, so `running` and `runner` end up close even if one was rare in training.

**Q4. What is the difference between input and output embeddings in Word2Vec?**
Each word has two vectors: $\mathbf{v}$ (when it appears as center/target) and $\mathbf{v}'$ (when it appears as context). The softmax/dot-product score is $\mathbf{v}'_{\text{context}} \cdot \mathbf{v}_{\text{center}}$. After training, typically $\mathbf{v}$ is used, but averaging them or concatenating sometimes helps. The two-matrix setup breaks the symmetry of the problem and prevents the trivial solution $\mathbf{v}_i = \mathbf{v}_j$ for all co-occurring words.

### Mathematical

**Q5. Derive the gradient of the Skip-gram negative sampling loss.**
For a single positive pair $(w_I, w_O)$ and negatives $\{w_1, \dots, w_k\}$, the per-example loss is:

$$
L = -\log\sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) - \sum_{i=1}^k \log\sigma(-\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I})
$$

Gradient w.r.t. $\mathbf{v}_{w_I}$:

$$
\frac{\partial L}{\partial \mathbf{v}_{w_I}} = (\sigma(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I}) - 1)\mathbf{v}'_{w_O} + \sum_i \sigma(\mathbf{v}'_{w_i} \cdot \mathbf{v}_{w_I}) \mathbf{v}'_{w_i}
$$

The first term pulls $\mathbf{v}_{w_I}$ toward $\mathbf{v}'_{w_O}$; the second pushes it away from each negative $\mathbf{v}'_{w_i}$. Magnitude is modulated by how confidently the model already classifies each pair correctly.

**Q6. Show that SGNS implicitly factorizes a shifted PMI matrix.**
At the optimum, setting the gradient to zero and assuming sufficient dimensionality, one can show (Levy & Goldberg, 2014):

$$
\mathbf{v}_i \cdot \mathbf{v}'_j = \log\frac{P(i,j)}{P(i)P(j)} - \log k = \text{PMI}(i,j) - \log k
$$

where $k$ is the number of negative samples. Proof sketch: at the stationary point, the expected gradient is zero, which for each pair gives $\sigma(\mathbf{v}_i \cdot \mathbf{v}'_j) = \frac{\#(i,j)}{\#(i,j) + k \cdot P_n(j) \cdot \#i}$. Solving and inverting the sigmoid yields the PMI relationship. This connects SGNS to decades of distributional semantics.

**Q7. Derive GloVe's objective from the co-occurrence ratio postulate.**
Postulate: $F(\mathbf{v}_i - \mathbf{v}_j, \mathbf{v}'_k) = P_{ik}/P_{jk}$. Require $F$ to be a homomorphism from $(\mathbb{R}^d, +)$ to $(\mathbb{R}, \times)$: $F(\mathbf{a} + \mathbf{b}) = F(\mathbf{a})F(\mathbf{b})$, giving $F = \exp$. Also require $F((\mathbf{v}_i - \mathbf{v}_j) \cdot \mathbf{v}'_k)$, so $F((\mathbf{v}_i - \mathbf{v}_j) \cdot \mathbf{v}'_k) = \exp(\mathbf{v}_i \cdot \mathbf{v}'_k)/\exp(\mathbf{v}_j \cdot \mathbf{v}'_k)$. Setting $\exp(\mathbf{v}_i \cdot \mathbf{v}'_k) = P_{ik} = X_{ik}/X_i$ and taking logs: $\mathbf{v}_i \cdot \mathbf{v}'_k + \log X_i = \log X_{ik}$. The $\log X_i$ is absorbed into a bias $b_i$, and symmetric bias $b'_k$ gives final form $\mathbf{v}_i \cdot \mathbf{v}'_k + b_i + b'_k = \log X_{ik}$.

**Q8. Why is the weighting function $f(x) = \min(1, (x/x_{\max})^{3/4})$ used in GloVe?**
Two reasons. First, $f(0) = 0$ removes the $-\infty$ target of unobserved pairs (which would be $\log 0$). Second, a naive unweighted loss is dominated by the very large number of high-frequency pairs like `the–of`, which carry little signal. Downweighting these with a sub-linear function gives the model room to learn from mid-frequency, content-bearing pairs. The $3/4$ exponent and $x_{\max} = 100$ are empirical defaults.

### Applied

**Q9. You are building a product-review sentiment classifier. Do you use GloVe, FastText, or a BERT embedding?**
Depends on constraints. If you have ample labeled data and a GPU budget: fine-tune a BERT-family model — contextual embeddings handle negation ("not good") and sarcasm better than static ones. If latency is tight or the device is constrained: averaged FastText embeddings + logistic regression is often within a few points of BERT on simple sentiment, with 100× lower latency. If the domain has heavy morphology or misspellings (e.g., user-generated content): prefer FastText over GloVe for OOV robustness.

**Q10. You have embeddings for 1M words and need fast nearest-neighbor search. How?**
Normalize to unit length so cosine similarity equals dot product. Use an approximate nearest-neighbor index: FAISS (IVF+PQ or HNSW), ScaNN, or Annoy. These achieve sub-millisecond lookup at 10M scale with >95% recall. Exact search is $O(|V| \cdot d)$ per query — fine for small vocabularies, infeasible at scale.

**Q11. How would you detect and mitigate gender bias in your embeddings?**
*Detection:* compute a "gender direction" as $\vec{\text{he}} - \vec{\text{she}}$ (or average across multiple pairs). Measure each profession word's projection onto this direction — large positive projection means male-associated, large negative means female-associated. *Mitigation:* Bolukbasi (2016) proposes projecting every non-gender-definitional word onto the hyperplane orthogonal to the gender direction (hard debiasing), or penalizing projections during training (soft debiasing). Caveat: Gonen & Goldberg (2019) showed these methods hide but don't remove bias — downstream classifiers can still recover it.

**Q12. Design an embedding-based search system that handles typos.**
Use FastText: its character n-gram model gives similar vectors to `laptop` and `laptob`. Alternative: train a dedicated typo-tolerant embedding on (correct, typo) pairs. For production, combine edit-distance filtering (candidate generation) with FastText reranking. For very high-recall systems, use byte-level models or fuzzy symmetric-delete indexes.

### Debugging & Failure Modes

**Q13. Your Word2Vec model trained on 10M sentences produces nearly identical vectors for all words. Diagnose.**
Likely causes:
- *Learning rate too high* — vectors collapse to a degenerate solution.
- *No negative sampling or wrong noise distribution* — without separation pressure, cosine pulls all vectors together.
- *Window too large* — every word ends up "related" to every other.
- *Sub-sampling of frequent words disabled* — `the` co-occurs with everything and pulls the whole space.
Check: measure $\mathbb{E}[\cos(\mathbf{v}_i, \mathbf{v}_j)]$ over random pairs; should be near 0 for well-trained embeddings. If it's near 1, you have a degenerate solution.

**Q14. You average word embeddings to classify documents but can't beat TF-IDF. Why?**
Averaging discards word order and gives equal weight to function words (`the`, `is`) and content words (`excellent`, `terrible`). TF-IDF weights uncommon informative words more. Fixes: (a) use IDF-weighted averaging, (b) use SIF embeddings (Arora 2017 — weight words by $a/(a + p(w))$ and remove first principal component), or (c) use a dedicated sentence encoder.

**Q15. After fine-tuning GloVe vectors on a small domain corpus, model performance *drops*. Why?**
GloVe needs billions of tokens to produce stable statistics. Continued training on a small corpus (~1M tokens) overfits to domain-specific biases and destroys the global structure learned during pretraining. Better: freeze embeddings and train downstream layers only; or use a small learning rate and early stopping; or switch to FastText which is more robust to small-corpus continued training.

### Follow-up Probes

**Q16. Why does $\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$?**
Mikolov's analogy structure emerges because the objective forces certain consistent differences. Across many sentences the pattern `{man, woman}` appearing in parallel syntactic positions (e.g., "the [man/woman] walked"), combined with the pattern `{king, queen}` in parallel royal contexts, produces approximately parallelogram structure: $\vec{\text{king}} - \vec{\text{queen}} \approx \vec{\text{man}} - \vec{\text{woman}}$ because both differences align with the "gender" axis learned from co-occurrence. Note: this is *approximate* and not a theorem — it was an empirical discovery and depends on training hyperparameters.

**Q17. Why are static embeddings fundamentally limited?**
Four reasons: (1) single vector per word cannot represent polysemy, (2) no sensitivity to syntax — word order is thrown away, (3) no long-range context — windows are local, (4) no composition — averaging vectors is a very weak sentence representation. These motivated contextual embeddings and Transformers.

**Q18. Under what conditions might a well-trained static embedding beat BERT?**
- When latency/memory is extremely tight (mobile, edge).
- When the downstream task is genuinely bag-of-words (topic modeling, spam detection).
- When the training data is so small that BERT overfits or cannot be fine-tuned.
- When interpretability via linear probing is required.
- Short queries in information retrieval where BERT adds little over tuned BM25 + GloVe.

**Q19. What is anisotropy in embedding space and why does it matter?**
Embeddings produced by deep models — and to a lesser extent SGNS — tend to occupy a narrow cone in the high-dimensional space rather than filling it isotropically. This means random pairs have high cosine similarity (0.4–0.8 for BERT), which washes out true similarity signals. Fixes: subtract the mean (centering), whiten the space, or use methods like BERT-whitening or SimCSE. GloVe is more isotropic than Word2Vec because the explicit matrix factorization objective encourages symmetric structure.

---

# 3. Contextual Word Representations

*ELMo, BERT, GPT embeddings, sentence transformers.*

## 3.1 Motivation & Intuition

Static embeddings have a fatal flaw: every word gets the same vector regardless of context. The sentence "I deposited money at the **bank**" and "I sat on the **bank** of the river" use `bank` in completely different senses, but Word2Vec returns the same vector for both. Worse, subtle context matters too: `light` is an adjective in "light color" but a noun in "turn on the light". A good representation should reflect this.

Contextual embeddings produce a vector for each *occurrence* of a word, computed as a function of its sentence. The same word in different sentences gets different vectors. This is the single largest conceptual jump in NLP of the 2010s.

The practical consequences are enormous:

- Question answering, natural language inference, and coreference resolution — all of which require reasoning about word meaning in context — became tractable.
- A single pretrained model (BERT) replaced dozens of task-specific architectures.
- The pretrain-then-finetune paradigm enabled transfer learning at NLP scale.
- Eventually, scaling this idea produced GPT-3, GPT-4, Claude, and the foundation-model era.

**Real-world ML impact.** Essentially every production NLP system since 2019 uses contextual embeddings either as features, as the backbone for fine-tuning, or as the base for in-context learning. Search (Google uses BERT in ranking), dialogue, classification, translation, code completion — all built on contextual representations.

## 3.2 Conceptual Foundations

### Core concepts

- **Pretraining**: train a large model on a self-supervised objective over massive unlabeled text. The model learns general linguistic competence.
- **Fine-tuning**: continue training on a smaller task-specific labeled dataset, adapting the pretrained weights to the task.
- **Language modeling objective**: predict next token (causal/left-to-right) or predict masked tokens (masked LM).
- **Bidirectional context**: the embedding for a word depends on both its left and right context. True for BERT and ELMo; not true for autoregressive GPT.
- **Layerwise representations**: deep models produce different embeddings at different layers. Lower layers encode syntax, middle layers morphology, higher layers semantics. You can pool or concatenate multiple layers.

### ELMo (Embeddings from Language Models, 2018)

Two independent LSTMs: one left-to-right, one right-to-left. Each word's representation is a learned weighted sum of:
- The character-CNN base embedding
- The first LSTM layer's output (both directions, concatenated)
- The second LSTM layer's output (both directions, concatenated)

These three are combined per-task with learned scalar weights, then fed as features to a downstream model. ELMo is *feature-based* — the LSTM parameters are frozen.

### BERT (Bidirectional Encoder Representations from Transformers, 2018)

A Transformer *encoder* stack (typically 12 or 24 layers). Pretrained on two objectives:

1. **Masked Language Modeling (MLM).** Randomly replace 15% of input tokens with `[MASK]` and train the model to predict the originals. Because self-attention sees all positions, each word's representation integrates full bidirectional context.
2. **Next Sentence Prediction (NSP).** Given two sentences, predict whether sentence B immediately follows sentence A. Later shown largely unnecessary (RoBERTa removed it); replaced by better pretraining recipes.

BERT is used by *fine-tuning*: add a thin task head, train end-to-end on labeled data.

### GPT (Generative Pre-trained Transformer, 2018+)

A Transformer *decoder* stack with causal (left-to-right) masking. Pretrained on next-token prediction:
$$
\mathcal{L} = -\sum_t \log p(x_t \mid x_1, \dots, x_{t-1})
$$
Later versions (GPT-2, 3, 4) scaled to billions/trillions of parameters and introduced in-context learning — performing new tasks from a few examples in the prompt, without weight updates.

**GPT embeddings** can be obtained by running the model forward and extracting hidden states; often the final layer's representation of the final token is used to summarize a prompt.

### Sentence Transformers (SBERT, 2019)

BERT produces token-level embeddings; averaging them to get a sentence embedding is shockingly bad for semantic similarity (worse than averaging GloVe vectors!). Sentence-BERT fine-tunes BERT with a **siamese** architecture and a contrastive or triplet objective so that semantically similar sentences end up close in embedding space. Modern variants (SimCSE, E5, BGE) dominate semantic search leaderboards.

### Transformer architecture — the common substrate

All of BERT, GPT, and SBERT share the Transformer block:
- **Self-attention**: each token computes a weighted combination of all other tokens' representations, where weights depend on content.
- **Feed-forward network (MLP)** applied to each position.
- **Residual connections and LayerNorm** for stable training.

Self-attention is the mechanism that enables contextualization: the embedding of `bank` in one sentence directly depends on the embeddings of `river` or `money` through attention weights.

### Assumptions and failure modes

**Assumption 1: Pretraining distribution roughly matches downstream distribution.**
Violated when pretraining on Wikipedia and deploying on legal contracts, clinical notes, or code — performance drops. Domain-adaptive pretraining (BioBERT, LegalBERT, CodeBERT) addresses this.

**Assumption 2: The masking / LM objective captures what you need.**
Violated when your downstream task requires sentence-level semantics — BERT's [CLS] token is *not* a good sentence embedding without fine-tuning (SBERT's core insight).

**Assumption 3: Fine-tuning is stable.**
Violated in practice — fine-tuning BERT with too-high learning rates or too few examples is notoriously unstable across seeds. Solutions: smaller LR ($2 \times 10^{-5}$), warmup, and techniques like layer-wise LR decay or SWA.

**Assumption 4: Deeper layers are better.**
Violated depending on task. For syntactic probes, layers 2–5 of BERT-base are best. For semantic tasks, layers 8–12. Concat-of-last-4 often beats using only the final layer.

## 3.3 Mathematical Formulation

### Self-attention — the core mechanism

Given input representations $\mathbf{X} \in \mathbb{R}^{n \times d}$ (sequence of $n$ tokens, dimension $d$), project into three matrices:

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}^V
$$

Attention is:

$$
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}
$$

The $\sqrt{d_k}$ scaling prevents dot products from growing large with dimension, which would saturate the softmax. The softmax row for position $i$ gives the weights token $i$ places on every other token; these weights are content-dependent.

**Multi-head attention** runs $h$ attention operations in parallel with different projections, then concatenates:

$$
\text{MultiHead}(\mathbf{X}) = [\text{head}_1; \dots; \text{head}_h] \mathbf{W}^O
$$

Each head can attend to different linguistic relations (syntactic dependencies, coreference, position).

**Causal masking** (GPT): set attention weights from position $i$ to $j > i$ to $-\infty$ before softmax so no token attends to the future.

### MLM objective (BERT)

For a sequence $\mathbf{x} = (x_1, \dots, x_n)$, randomly mask a subset $M \subset \{1, \dots, n\}$ (typically $|M| \approx 0.15 n$). The model predicts the masked tokens given the full masked sequence:

$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log p(x_i \mid \mathbf{x}_{\setminus M})
$$

BERT applies the 80/10/10 rule: 80% of masked positions become `[MASK]`, 10% are replaced with a random token, 10% are left unchanged. This reduces pretrain-finetune mismatch (fine-tuning data has no `[MASK]` tokens).

### Causal LM objective (GPT)

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^n \log p(x_t \mid x_1, \dots, x_{t-1})
$$

Every token is a training signal, in contrast to MLM's 15%. This is why decoder-only models train more sample-efficiently per token — but MLM can use bidirectional context, so the trade-off is genuine.

### Sentence-BERT training

Given a pair of sentences $(s_a, s_b)$ with label $y \in \{0, 1\}$ (entailment/not, or semantic similar/not), pass each through the *same* BERT (siamese), pool token representations to get $\mathbf{u}, \mathbf{v}$, and optimize:

*Classification objective (NLI)*:
$$
\mathbf{o} = \mathbf{W}\, [\mathbf{u}; \mathbf{v}; |\mathbf{u} - \mathbf{v}|]
$$
with cross-entropy against the label.

*Triplet objective* for retrieval:
$$
\mathcal{L} = \max\!\left(0, \|\mathbf{u} - \mathbf{v}^+\|_2 - \|\mathbf{u} - \mathbf{v}^-\|_2 + \epsilon\right)
$$
pulls the anchor $\mathbf{u}$ closer to a positive $\mathbf{v}^+$ than to a negative $\mathbf{v}^-$ by margin $\epsilon$.

### Mapping math to intuition

- $\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k}$: how much token $i$ "wants information from" token $j$, content-dependent.
- Softmax row: probability distribution over where to look.
- Weighted sum of $\mathbf{V}$: a fresh representation of $i$ built from relevant context.
- MLM reconstruction: forces each position to aggregate enough context to predict itself from its surroundings.
- Siamese training: pulls semantic neighbors together and pushes non-neighbors apart — the fundamental recipe of contrastive learning.

## 3.4 Worked Example

**Obtaining BERT embeddings for `"the bank of the river"` vs `"the bank approved the loan"`.**

Step 1. Tokenize both sentences with BERT's WordPiece:
- S1: `[CLS] the bank of the river [SEP]`
- S2: `[CLS] the bank approved the loan [SEP]`

Step 2. Forward pass through all 12 Transformer layers. For each token position, each layer produces a 768-dim hidden state (for BERT-base).

Step 3. Extract the hidden state at position `bank` from the final layer (or average the last 4 layers — a common choice). Call these $\mathbf{h}_{\text{bank}}^{(1)}$ and $\mathbf{h}_{\text{bank}}^{(2)}$.

Step 4. Compare: $\cos(\mathbf{h}_{\text{bank}}^{(1)}, \mathbf{h}_{\text{bank}}^{(2)})$. Empirically this is around $0.4$ — much lower than $\cos(\mathbf{h}_{\text{bank}}^{(1)}, \mathbf{h}_{\text{river}}^{(1)})$ or $\cos(\mathbf{h}_{\text{bank}}^{(2)}, \mathbf{h}_{\text{loan}}^{(2)})$. Contextualization in action.

**Fine-tuning BERT for sentiment classification.**

- Add a linear layer on top of `[CLS]`: $p(y) = \text{softmax}(\mathbf{W} \mathbf{h}_{[\text{CLS}]} + \mathbf{b})$.
- Loss: cross-entropy on labeled data.
- Optimizer: AdamW with LR $2 \times 10^{-5}$, linear warmup over 10% of steps, then linear decay.
- Typical recipe: 3–5 epochs, batch size 16–32, max length 128 or 256.

After fine-tuning, the `[CLS]` vector becomes a task-specific sentence representation tuned for sentiment.

**Sentence-BERT for semantic search.**

Step 1. Pretrain a BERT-style base model.
Step 2. Fine-tune on NLI (SNLI + MultiNLI) with a siamese classification objective for a few epochs.
Step 3. To index documents, compute $\mathbf{v}_{\text{doc}} = \text{MeanPool}(\text{BERT}(\text{doc}))$ for each document. Store in a vector DB.
Step 4. At query time, compute $\mathbf{v}_{\text{query}}$ similarly, find nearest neighbors by cosine similarity.

Contrast with using vanilla BERT: its raw `[CLS]` or mean-pool vectors give worse retrieval quality than GloVe averaging, because BERT is not trained to make *sentence* representations.

## 3.5 Relevance to ML Practice

**Where they appear.**

- **Training time**: pretraining + fine-tuning paradigm is ubiquitous; prompt tuning and adapters (LoRA) are lightweight alternatives.
- **Inference**: single-sentence classification, token tagging, pair classification (NLI, QA), and retrieval (SBERT + vector DB).
- **Evaluation**: models like BERTScore use BERT embeddings to compare generated text to references, capturing semantic similarity better than BLEU.
- **Monitoring**: embedding drift detection — compute embeddings over production traffic and compare to training distribution via MMD or KL over cluster assignments.

**When to use each.**

- **BERT (encoder-only)**: classification, NER, extractive QA, NLI, retrieval with fine-tuning. Not for open-ended generation (it wasn't trained for it).
- **GPT (decoder-only)**: generation, dialogue, few-shot / zero-shot reasoning, code synthesis. Also surprisingly strong for classification if prompted well.
- **ELMo**: rarely used anymore — superseded by BERT. Historical value: showed that contextual features could improve many tasks.
- **Sentence transformers**: retrieval, clustering, semantic search, deduplication, re-ranking. Cheap to serve at inference time.
- **T5 (encoder-decoder)**: when both understanding and generation are needed (summarization, translation, QA with generation).

**Trade-offs.**

| Model family | Best at | Cost | Example |
|---|---|---|---|
| Encoder-only | Understanding tasks | Low at inference | BERT, RoBERTa, DeBERTa |
| Decoder-only | Generation, in-context | Very high at scale | GPT, Llama, Claude |
| Encoder-decoder | Seq2seq tasks | Medium | T5, BART, mT5 |
| Sentence encoders | Retrieval | Very low | SBERT, E5, BGE |

**Scaling considerations.**

- BERT-base (110M params): fits on a single GPU, fine-tunes in hours.
- BERT-large (340M): 2–4× compute, ~2% better on most benchmarks.
- Llama-3 405B: cannot fine-tune on a single machine. Use LoRA, QLoRA, or inference-only with in-context learning.

**Alternatives and when to prefer them.**

- If you need the smallest fast model: DistilBERT, MiniLM, or TinyBERT.
- If you need long context: Longformer, BigBird, Mistral (sliding window), Claude (200k+ tokens).
- If your task is clearly tabular or not-really-NLP: static embeddings or TF-IDF + linear model often suffices.

## 3.6 Common Pitfalls

1. **Using BERT's `[CLS]` token for semantic similarity without fine-tuning.** The `[CLS]` was trained for NSP; it is a poor sentence embedding. Cosine on raw BERT `[CLS]` vectors is worse than averaged GloVe. Always fine-tune (SBERT) for retrieval/similarity.

2. **Ignoring anisotropy in contextual embedding space.** BERT embeddings concentrate in a narrow cone — random pairs have cosine 0.5–0.8. Applying cosine naively gives unreliable rankings. Use whitening, BERT-flow, or SimCSE.

3. **Catastrophic forgetting during fine-tuning.** Fine-tuning on a small domain dataset can destroy general linguistic competence. Mitigations: smaller LR, fewer epochs, layer-wise LR decay, or continued pretraining before task fine-tuning.

4. **Seed sensitivity.** Fine-tuning small BERT on small datasets varies by several F1 points across random seeds. Always run 3+ seeds and report median or mean with variance. "Mixout", re-init of top layers, or stochastic weight averaging help.

5. **Truncating context silently.** BERT has a 512-token limit. If you feed a 2000-word document, the last 75% is discarded. For long inputs, use long-context models or hierarchical architectures.

6. **Mismatched tokenizers between base and fine-tune.** The tokenizer must match the pretrained model exactly. Using a "similar" tokenizer produces garbage embeddings.

7. **Assuming bigger is always better.** For many enterprise NLP tasks — ticket classification, entity tagging — a fine-tuned BERT-base outperforms a zero-shot GPT-4 at 1000× lower cost and 100× lower latency.

8. **Forgetting special tokens when constructing input.** For sentence pairs, you need `[CLS] sent1 [SEP] sent2 [SEP]` with proper segment IDs. Forgetting the SEP or getting segment embeddings wrong silently hurts performance.

9. **Pooling inconsistently at training vs. inference.** If you fine-tuned SBERT with mean pooling, you must use mean pooling at inference. Mixing `[CLS]` vs mean-pool vs max-pool changes the embedding distribution entirely.

---

## 3.7 Interview Questions: Contextual Representations

### Foundational

**Q1. What makes BERT "bidirectional" and why does this matter?**
BERT's self-attention is not causally masked: every token at every layer attends to every other token. The MLM objective (masking a token and predicting it from surrounding context) forces each position's representation to integrate information from both left and right. This matters because many NLP tasks need full context — "the trophy didn't fit in the suitcase because **it** was too small" requires reasoning that the pronoun refers to the suitcase, which only full context can disambiguate.

**Q2. Why does GPT use causal masking if bidirectional context is helpful?**
Causal masking enforces left-to-right generation, which is what generative tasks require: at inference you have only the past. The training objective (next-token prediction) naturally respects this. Bidirectional models like BERT cannot generate text left-to-right without additional machinery because their representations depend on the future. BERT is for understanding; GPT is for generation.

**Q3. Why did Sentence-BERT need to exist — why can't we just average BERT token embeddings?**
BERT is pretrained to produce contextualized *token* representations and (weakly) a next-sentence-predictor via `[CLS]`. It is not trained to put semantically similar *sentences* close in embedding space. Empirically, average-pooled BERT embeddings perform worse than average-pooled GloVe on semantic textual similarity tasks. SBERT adds a siamese fine-tuning step on NLI data with a classification or triplet loss that explicitly shapes the sentence-vector space — pulling semantically similar sentences together.

**Q4. Compare ELMo, BERT, and GPT in one sentence each.**
- *ELMo*: two LSTMs (forward and backward) whose layer-wise outputs are combined as features for downstream tasks.
- *BERT*: Transformer encoder pretrained on MLM + NSP, used by fine-tuning for understanding tasks.
- *GPT*: Transformer decoder pretrained on causal LM, used for generation and increasingly for general-purpose tasks via prompting.

### Mathematical

**Q5. Derive scaled dot-product attention and explain each component.**
Given $\mathbf{X} \in \mathbb{R}^{n \times d}$, project $\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}^K$, $\mathbf{V} = \mathbf{X}\mathbf{W}^V$. Attention output:

$$
\text{Attn} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

- $\mathbf{Q}\mathbf{K}^\top$: all pairwise similarities (query-key). Row $i$ gives how much token $i$ wants information from each other token.
- Scaling by $\sqrt{d_k}$: as $d_k$ grows, dot products have variance $d_k$ (sum of $d_k$ iid unit-variance terms). Large values saturate softmax, killing gradients. Dividing by $\sqrt{d_k}$ normalizes variance to 1.
- Softmax: turns raw scores into a probability distribution over attention targets.
- Multiplication by $\mathbf{V}$: weighted sum of "values" gives the output representation.

**Q6. Explain why BERT's 80/10/10 masking rule exists.**
The raw MLM objective would always replace masked tokens with `[MASK]`. But at fine-tuning and inference, no `[MASK]` tokens appear. This creates a distributional mismatch — the model never learned what to do when it sees the original word. To reduce this gap: 80% of chosen positions get `[MASK]`, 10% get a random token (so the model must robustly predict every position regardless of the token there), and 10% get the original token unchanged (so the model learns to "handle" real tokens like targets). Without this trick, fine-tuning performance degrades.

**Q7. How many floating-point operations does self-attention require, and what is the bottleneck?**
For sequence length $n$ and hidden dim $d$:
- Projections $\mathbf{Q}, \mathbf{K}, \mathbf{V}$: $3 n d^2$ FLOPs.
- Attention matrix $\mathbf{Q}\mathbf{K}^\top$: $n^2 d$ FLOPs, producing an $n \times n$ matrix.
- Softmax $\cdot \mathbf{V}$: $n^2 d$ FLOPs.
- Output projection: $n d^2$ FLOPs.
Total: $O(n^2 d + n d^2)$. For long sequences ($n \gg d$), the quadratic-in-$n$ terms dominate. This is why long-context requires either sparse attention (Longformer, BigBird), linearized attention (Performer), or sliding-window attention (Mistral).

**Q8. Derive the gradient of the MLM loss with respect to a masked token's embedding.**
Let $\mathbf{h}_i$ be the final-layer hidden state at masked position $i$, and $\mathbf{W}_{\text{mlm}}$ the output projection to vocabulary. The logit for vocab word $v$ is $z_v = \mathbf{W}_{v,:} \mathbf{h}_i$. Loss is $-\log p(y_i) = -z_{y_i} + \log \sum_v e^{z_v}$. Gradient:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_i} = \sum_v (p_v - \mathbb{1}[v = y_i]) \mathbf{W}_{v,:}
$$

This pushes $\mathbf{h}_i$ toward $\mathbf{W}_{y_i,:}$ (the correct row) and away from high-probability incorrect rows. The gradient then backpropagates through the Transformer, updating the attention and embedding parameters so that future representations at this position better predict the correct token.

### Applied

**Q9. You want to deploy BERT for 10k queries/sec. What do you do?**
Options:
- *Distillation*: train a smaller student (DistilBERT, MiniLM) to mimic BERT. 2–4× speedup for modest accuracy loss.
- *Quantization*: INT8 quantization with calibration. 2–4× speedup and memory reduction.
- *ONNX/TensorRT*: graph optimization and kernel fusion can give 2× on GPU.
- *Batching*: dynamic batching with a short queue.
- *Caching*: if queries repeat, cache embeddings.
- *For retrieval*: precompute document embeddings offline; only encode the query at runtime.
Combined, these routinely give 10–50× speedup.

**Q10. You fine-tuned BERT on a multi-class classifier; accuracy plateaus below a reasonable baseline. Debug.**
Check (in order): (1) label leakage — duplicates across train/test; (2) learning rate too high (common: use $2 \times 10^{-5}$, not $10^{-3}$); (3) insufficient warmup; (4) tokenizer mismatch with checkpoint; (5) truncation destroying content (check `max_length`); (6) class imbalance — use weighted loss or focal loss; (7) only 1–3 epochs (usually enough) vs 10+ (usually too many, overfits); (8) seed variance — try 3 seeds.

**Q11. Design an SBERT-based semantic search pipeline for a corpus of 10M documents.**
- *Model choice*: SBERT variant appropriate to domain, e.g., E5-large or BGE-large.
- *Indexing*: encode all documents offline, normalize to unit L2 norm, store in a vector DB (Qdrant, Milvus, or FAISS).
- *Approximate search*: use HNSW or IVF-PQ index with 10–50ms p99 latency at 10M scale.
- *Query*: encode query at runtime, retrieve top-100 candidates.
- *Re-rank*: pass top-100 through a cross-encoder (which jointly encodes query + doc) for precise ranking. This is the classic bi-encoder / cross-encoder two-stage setup.
- *Evaluation*: NDCG@10 and Recall@100 on a held-out labeled set.
- *Update strategy*: periodic re-indexing; for streaming updates, Qdrant/Milvus support live inserts.

**Q12. Should you use zero-shot GPT-4 or fine-tuned BERT for a binary classification task with 10k labeled examples?**
Likely fine-tuned BERT: cheaper ($100× lower per-call cost), faster (10–100× lower latency), comparable or better quality on in-distribution data with 10k labels. GPT-4 wins when you have <100 labels, the task requires reasoning, or you need few-shot flexibility across many related tasks. With 10k labels, a specialist beats a generalist on the specific task — while being dramatically cheaper to serve.

### Debugging & Failure Modes

**Q13. After fine-tuning, your BERT model has high training accuracy but random test accuracy. What happened?**
Severe overfitting. Most likely causes and fixes:
- Too many epochs → use early stopping with dev set.
- Learning rate too high → reduce to $2 \times 10^{-5}$ or lower.
- Too little regularization → add dropout, weight decay ($0.01$), or label smoothing.
- Training set too small for BERT's capacity → augment data, use data-efficient methods (LoRA, prompt tuning).
- Data leakage between train and dev inflates dev accuracy too; check if dev accuracy is also inflated or if test is a different distribution.

**Q14. Mean-pooled BERT embeddings give cosine similarities clustered around 0.9 for all document pairs. What's wrong, and how do you fix it?**
Anisotropy: BERT's embedding space concentrates in a narrow cone. Every pair has high cosine regardless of actual similarity. Fixes in increasing effectiveness:
- Subtract the mean of the training set embeddings (centering).
- Whitening: apply $\mathbf{z} = (\mathbf{x} - \mu) \Sigma^{-1/2}$ using training-set mean and covariance.
- Use BERT-flow or BERT-whitening (both introduce a normalizing flow).
- Best: use SimCSE/SBERT — a fine-tuning procedure that explicitly shapes the similarity geometry via contrastive learning.

**Q15. You're getting different fine-tuned accuracies on the same data with different seeds — variance of 3 F1 points. Why and what do you do?**
BERT fine-tuning is famously unstable with small datasets and short training runs. Causes: (a) optimizer state initialized from scratch while weights come from pretraining — a gradient-direction mismatch; (b) top layers (close to task head) get more gradient and can be destabilized; (c) `[CLS]`-based classifiers are sensitive to the arbitrary pretraining signal on `[CLS]`. Fixes: re-initialize the top $k$ encoder layers, use Mixout, use a longer warmup, or average multiple fine-tuning runs (checkpoint averaging / SWA). Mosbach et al. (2021) found that longer training + smaller LR largely resolves the instability.

### Follow-up Probes

**Q16. Why does the Transformer's positional encoding exist, and what happens without it?**
Self-attention is permutation-equivariant: shuffle the input tokens and the output is shuffled the same way. Without positional information, the model cannot distinguish "dog bites man" from "man bites dog". Positional encodings (sinusoidal in the original paper, learned in BERT, rotary/RoPE in modern LLMs, ALiBi for extrapolation) inject order information. Different schemes trade off between training efficiency and length extrapolation.

**Q17. What does "probing" tell us about what BERT learns?**
Probing trains a small classifier on frozen BERT features to predict some linguistic property (POS, dependency label, semantic role). Results: lower layers encode surface and morphological features, middle layers encode syntax, higher layers encode semantics. This "BERTology" literature (Tenney et al., Rogers et al.) established that BERT implicitly learns a classical NLP pipeline without being told to. Caveat: probes can succeed by exploiting the probe's own capacity, not only the representation's — use information-theoretic probes or control tasks to isolate what the representation itself encodes.

**Q18. Why is BERT pretrained on both MLM and NSP, but modern models drop NSP?**
NSP was meant to teach sentence-level relationships. Later work (RoBERTa, Liu et al. 2019) showed NSP was too easy: the negative pairs came from different documents, which is also a topic change, so NSP reduces to topic classification and doesn't force deep sentence-level reasoning. Replacements: ALBERT uses Sentence Order Prediction (same document, swapped order) which is genuinely harder; most modern pretraining simply uses MLM on longer chunks and drops NSP entirely.

**Q19. What is in-context learning, and how does it emerge in GPT-style models?**
In-context learning: the model performs a new task from a prompt that contains task descriptions and/or examples, with no weight updates. The weights are fixed; the "learning" happens implicitly in the forward pass. Emergence: empirically appears around 1B+ parameters for simple tasks and scales smoothly with model size on harder tasks. Mechanistically (active research area): the attention patterns in certain heads implement something like gradient descent on a linearized meta-objective — Akyürek et al. 2022 and others show this for linear regression tasks. For general language tasks the mechanism is less clear but involves induction heads (heads that copy patterns from context) and function-composition across layers.

**Q20. When would you prefer LoRA/PEFT over full fine-tuning?**
Prefer LoRA when:
- The base model is large (>1B) and full fine-tuning is infeasible on your hardware.
- You need to serve many task-specialized variants — LoRA adapters are small (~10MB) and swappable.
- Your training data is small enough that full fine-tuning would overfit.
- You want to avoid catastrophic forgetting — LoRA keeps base weights frozen.
Prefer full fine-tuning when:
- You have abundant data and compute.
- Your task requires changing core model behavior (e.g., heavy domain shift).
- Maximum performance is required and a few extra points matter.

---

# 4. Sequence Labeling

*NER, POS tagging, chunking; CRF, BiLSTM-CRF.*

## 4.1 Motivation & Intuition

Classification assigns a single label to a whole sentence. Sequence labeling assigns a label to *each token*. Applications:

- **POS tagging**: tag each word with its part of speech (`the/DT cat/NN sat/VBD`).
- **Named entity recognition (NER)**: identify spans like `[PERSON Barack Obama]` or `[ORG Google]`.
- **Chunking (shallow parsing)**: group words into non-overlapping phrases (`[NP the quick brown fox]`).
- **Slot filling**: for a user utterance "book a flight from Boston to NYC tomorrow", tag Boston as FROM-CITY, NYC as TO-CITY, tomorrow as DATE.

The naive approach — classify each token independently — ignores that *labels are correlated*. In NER, `B-PER` (beginning of person) is likely followed by `I-PER` (inside person) and never by `I-LOC` (inside location). POS has strong bigram structure: a determiner is almost always followed by a noun or adjective. Token-independent classification routinely produces illegal label sequences like `B-PER I-LOC`.

Sequence labeling models exploit these correlations. The classical solution is a **Conditional Random Field (CRF)** layer stacked on top of per-token features (HMM features historically, BiLSTM embeddings in the 2015-2020 era, Transformer outputs today). CRFs model the *joint* probability of the entire label sequence conditional on the input, capturing label-to-label transitions.

**Real-world ML impact.** Sequence labeling powers voice assistants (slot filling), clinical NLP (extracting diagnoses and medications from notes), financial compliance (detecting PII), search (query understanding), and any pipeline that needs structured facts from unstructured text.

## 4.2 Conceptual Foundations

### Core terms

- **Tag scheme (IOB/BIO/BIOES)**: encoding for span labels.
  - *IOB*: `B-X` = beginning of span of type X, `I-X` = inside, `O` = outside.
  - *BIO* = IOB but `B-X` is mandatory for the first token of every span.
  - *BIOES/BILOU*: adds `E-X` (end) and `S-X` (single-token span), providing stronger signal at span boundaries.
- **Emission score**: how well a token matches a tag based on its features alone.
- **Transition score**: how compatible two adjacent tags are.
- **Viterbi decoding**: dynamic programming to find the highest-scoring tag sequence.
- **Forward–backward**: DP to compute marginal probabilities and partition function for training.
- **CRF**: a globally normalized sequence model where the score of a label sequence $\mathbf{y}$ given input $\mathbf{x}$ is $\exp(\text{emit} + \text{transit}) / Z(\mathbf{x})$, with $Z$ summing over all possible sequences.

### Architectural evolution

1. **Hand-crafted features + HMM or linear-chain CRF** (2000s). Features: word shape, capitalization, surrounding words, gazetteers.
2. **BiLSTM + softmax** (2015). Each token's hidden state feeds a per-token softmax. Local, no transitions.
3. **BiLSTM-CRF** (Huang 2015, Lample 2016). Add a CRF layer on top of BiLSTM emissions. Near-SOTA for years, still competitive.
4. **Transformer (BERT) + token classification** (2018+). Often outperforms BiLSTM-CRF without the CRF — but BERT-CRF (adding a CRF on top of BERT) sometimes gives another ~0.5 F1 point, especially on small datasets with strong label grammar.
5. **Generative / span-based** (2020+). Frame NER as span prediction or as text-to-text generation (T5, GPT).

### BIO tagging in detail

For an NER example: "Barack Obama was born in Hawaii":

| Token | Label |
|---|---|
| Barack | B-PER |
| Obama | I-PER |
| was | O |
| born | O |
| in | O |
| Hawaii | B-LOC |

**Why not just mark the inside?** Without `B-` and `I-` distinctions, two adjacent person names like "Alice Bob-Smith Carol" would become one span. The `B-` tag marks span boundaries explicitly.

### Assumptions and failure modes

**Assumption 1: Labels form a linear chain.**
CRFs assume transitions only matter between adjacent tokens. Violated for long-distance label dependencies (coreference-like patterns). Higher-order CRFs or graph CRFs can help but are expensive.

**Assumption 2: Tokenization matches annotation.**
Violated frequently. NER is usually annotated at the word level, but models use subword tokenization. Need alignment: typically, label the first subword and mask the rest (convention: `-100` label in PyTorch).

**Assumption 3: Training and test entity distributions match.**
Violated when deploying a Wikipedia-trained NER model on clinical notes — it will not find `Metformin` as a drug because it never saw medication names.

**Assumption 4: Spans are non-overlapping.**
Violated for "Bank of America": is it an ORG or can it contain LOC? BIO cannot represent overlapping or nested entities. For nested NER, use span-based models (biaffine) or layered tagging.

## 4.3 Mathematical Formulation

### Linear-chain CRF

Given input $\mathbf{x} = (x_1, \dots, x_n)$ and label sequence $\mathbf{y} = (y_1, \dots, y_n)$:

$$
p(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\!\left(\sum_{t=1}^n \psi(y_t, x_t, t) + \sum_{t=2}^n \phi(y_{t-1}, y_t)\right)
$$

- $\psi(y_t, x_t, t)$: **emission** score for assigning label $y_t$ to token $x_t$. In BiLSTM-CRF or BERT-CRF this is $\mathbf{W}_{y_t} \mathbf{h}_t + b_{y_t}$ where $\mathbf{h}_t$ is the contextualized hidden state.
- $\phi(y_{t-1}, y_t)$: **transition** score from $y_{t-1}$ to $y_t$, typically a learned $|Y| \times |Y|$ matrix $\mathbf{T}$.
- $Z(\mathbf{x}) = \sum_{\mathbf{y}'} \exp(\text{score}(\mathbf{y}', \mathbf{x}))$: partition function, ensures $p$ sums to 1 over all $|Y|^n$ label sequences.

### Computing $Z$ with the forward algorithm

Naive sum over $|Y|^n$ is intractable. Exploit the Markov structure: define

$$
\alpha_t(y) = \sum_{\mathbf{y}_{1:t-1}} \exp\!\left(\sum_{s=1}^t \psi(y_s, x_s) + \sum_{s=2}^t \phi(y_{s-1}, y_s)\right)\bigg|_{y_t = y}
$$

the unnormalized score of all partial sequences ending in tag $y$ at position $t$. Recurrence:

$$
\alpha_t(y) = \exp(\psi(y, x_t)) \sum_{y'} \alpha_{t-1}(y') \exp(\phi(y', y))
$$

Base case $\alpha_1(y) = \exp(\psi(y, x_1))$. Final: $Z(\mathbf{x}) = \sum_y \alpha_n(y)$. Complexity: $O(n |Y|^2)$.

In log-space to avoid overflow:
$$
\log \alpha_t(y) = \psi(y, x_t) + \text{logsumexp}_{y'}\!\left(\log \alpha_{t-1}(y') + \phi(y', y)\right)
$$

### Training

Negative log-likelihood:
$$
-\log p(\mathbf{y}^* \mid \mathbf{x}) = -\text{score}(\mathbf{y}^*, \mathbf{x}) + \log Z(\mathbf{x})
$$
Gradient: the first term is easy (just the score of the gold sequence). The second is $\nabla \log Z(\mathbf{x}) = \mathbb{E}_{p(\mathbf{y} \mid \mathbf{x})}[\nabla \text{score}(\mathbf{y}, \mathbf{x})]$, which is the expected feature count under the model and can be computed by forward-backward. Fortunately modern frameworks (PyTorch) give this automatically via autograd on the log $Z$ computation.

### Viterbi decoding

At inference, find $\mathbf{y}^* = \arg\max_{\mathbf{y}} p(\mathbf{y} \mid \mathbf{x})$. Dynamic programming:
$$
\delta_t(y) = \max_{y'} \left[\delta_{t-1}(y') + \phi(y', y)\right] + \psi(y, x_t)
$$
with back-pointer $\beta_t(y) = \arg\max_{y'}[\delta_{t-1}(y') + \phi(y', y)]$. Trace back from $y_n^* = \arg\max_y \delta_n(y)$. Complexity: $O(n |Y|^2)$.

### BiLSTM-CRF architecture

$$
\mathbf{h}_t = \text{BiLSTM}(x_1, \dots, x_n)_t, \quad \psi(y, x_t) = (\mathbf{W} \mathbf{h}_t)_y
$$

$$
\text{score}(\mathbf{y}, \mathbf{x}) = \sum_t (\mathbf{W}\mathbf{h}_t)_{y_t} + \sum_t \mathbf{T}_{y_{t-1}, y_t}
$$

The transition matrix $\mathbf{T}$ is learned end-to-end with the BiLSTM. Crucial: add start and end states or boundary handling.

### Mapping math back to intuition

- $\psi$: "how well does this word fit this label on its own?"
- $\phi$: "how natural is this label-to-label transition?"
- $Z$: "total mass across all label sequences" — gives the normalization so probabilities are valid.
- Forward algorithm: "accumulate probability from left to right, sharing computation across all paths that end in each tag".
- Viterbi: "same recursion but with max instead of sum — pick the single best path".

## 4.4 Worked Example

**BiLSTM-CRF NER on "Apple sold iPhones".**

Label set: $\{O, \text{B-ORG}, \text{I-ORG}, \text{B-PROD}, \text{I-PROD}\}$.

**Step 1. BiLSTM gives us emission scores** (hypothetical, arbitrary units):

| Token | O | B-ORG | I-ORG | B-PROD | I-PROD |
|---|---|---|---|---|---|
| Apple | 1.0 | 3.5 | 0.1 | 1.2 | 0.0 |
| sold | 4.0 | 0.5 | 0.2 | 0.3 | 0.1 |
| iPhones | 1.5 | 1.0 | 0.8 | 3.2 | 0.5 |

**Step 2. Transition matrix** $\mathbf{T}$ (learned): high for valid transitions, low for invalid.
- $\mathbf{T}_{O, B\text{-}ORG} = 0.5$ (valid; entity may start after O)
- $\mathbf{T}_{O, I\text{-}ORG} = -5$ (invalid; cannot start inside an entity with `I`)
- $\mathbf{T}_{B\text{-}ORG, I\text{-}ORG} = 1.0$ (natural continuation)
- $\mathbf{T}_{B\text{-}ORG, I\text{-}PROD} = -5$ (type mismatch)
- $\mathbf{T}_{B\text{-}ORG, O} = 0.3$ (entity ends)

**Step 3. Viterbi for the best sequence.**

Initialize $\delta_1(y) = \psi(y, x_1)$ plus a start transition (omitted for brevity):

$\delta_1 = [\text{Apple:O}=1.0, \text{B-ORG}=3.5, \text{B-PROD}=1.2]$ etc.

For position 2 (`sold`), consider all predecessors. For tag `O` at position 2:
$\delta_2(O) = \max_{y'} [\delta_1(y') + \mathbf{T}_{y', O}] + \psi(O, \text{sold})$

Compute the max over $y'$:
- From O: $1.0 + 0.3 + 4.0 = 5.3$
- From B-ORG: $3.5 + 0.3 + 4.0 = 7.8$
- From B-PROD: $1.2 + 0.3 + 4.0 = 5.5$

So $\delta_2(O) = 7.8$ with backpointer B-ORG.

Similarly compute all $\delta_2$ entries; invalid transitions (like `B-ORG → I-PROD`) get very low scores from the $-5$ penalty.

Repeat for position 3 (`iPhones`). Best final tag: `B-PROD` is highly favored by emission (3.2) and transition from `O` (which was best at position 2) is clean.

Trace back: `Apple → B-ORG`, `sold → O`, `iPhones → B-PROD`. Without the CRF layer, a softmax might have given `iPhones` the label `I-ORG` (spurious) or `I-PROD` (illegal from `O`). The transition scores forbid these.

**Contrast: soft softmax baseline** might produce `Apple → B-ORG, sold → O, iPhones → I-ORG`, which is syntactically invalid (cannot go O → I-ORG). CRF decoding guarantees legal sequences.

## 4.5 Relevance to ML Practice

**Where it appears.**

- **Training**: final layer of NER/POS/chunking models. In modern BERT pipelines, a per-token classification head often suffices, but a CRF on top gives small gains on small-data regimes.
- **Inference**: Viterbi decoding is cheap ($O(n|Y|^2)$) and parallelizable.
- **Evaluation**: token-level accuracy is easy; span-level F1 (per the CoNLL evaluation) is the standard and requires exact boundary match.
- **Monitoring**: watch for tag-grammar violations (e.g., `I-X` without preceding `B-X`) — they indicate the model has drifted or decoding is off.

**When to use.**

- CRF when: training data is small, label grammar is strong (NER with BIO), interpretability of transitions matters, or you need hard constraints (disallow illegal transitions with $-\infty$).
- Token-level softmax when: training data is large (>50k sentences), you're using a strong encoder (BERT-large), and task is simple (POS tagging).
- Span-based models when: nested or overlapping entities exist.
- Generative models (T5, LLMs) when: label set is very large, or you want few-shot / zero-shot generalization.

**When NOT to use CRF.**

- When modern LLMs already give near-perfect results with prompting and the label grammar is trivially enforceable by post-processing.
- For very long sequences with small $|Y|$: transitions may not be the bottleneck.
- When overlapping or hierarchical structure is needed — linear-chain CRF cannot represent it.

**Trade-offs.**

| Approach | Pros | Cons |
|---|---|---|
| Softmax per token | Simple, fast, no dependencies | Ignores label grammar, can produce illegal sequences |
| Linear-chain CRF | Models transitions, enforces grammar | Training is slightly slower; only local dependencies |
| BiLSTM-CRF | Strong features + transitions | Dated compared to Transformers |
| BERT + softmax | Best features, huge pretraining benefit | May produce illegal sequences |
| BERT-CRF | Best of both worlds | CRF layer is marginal gain on big data |
| Span-based (biaffine) | Handles nested entities | More complex, slower |
| LLM prompting | Zero-shot, flexible | Expensive, brittle, prompt-sensitive |

## 4.6 Common Pitfalls

1. **Using token-level accuracy instead of entity F1.** Token accuracy is deceptive because most tokens are `O` — a model that predicts `O` for everything might get 90% token accuracy but 0% entity F1.

2. **Not enforcing BIO constraints at decode time.** Without a CRF, naive argmax decoding often produces `O → I-PER` (illegal). Either use a CRF, post-process, or mask invalid transitions.

3. **Mis-aligning subword tokens with word labels.** BERT tokenizes "playing" as `play ##ing`. If `playing` is labeled `B-VERB`, you must decide whether to (a) label only `play` and set `##ing` to `-100` (ignore in loss), (b) label both with the same tag, or (c) use BIOES where `##ing` gets `E-VERB`. Option (a) is standard.

4. **Evaluating at the word level when annotations are at the sentence level.** Ensure consistent granularity.

5. **Hand-labeling inconsistencies.** IAA (inter-annotator agreement) is often <85% for NER in specialized domains. Your test set noise floor is set by annotator disagreement.

6. **Class imbalance.** `O` dominates. Weighted cross-entropy, focal loss, or over-sampling entity-bearing sentences can help.

7. **Over-relying on capitalization.** English NER models heavily exploit case; on lowercased text (tweets, ASR output) they fail. Train with cased + uncased augmentation.

8. **Nested entities overlooked.** "Bank of America Tower" contains ORG `Bank of America` and FAC `Bank of America Tower`. BIO cannot represent both. Use nested NER models.

9. **CRF on top of BERT with the wrong learning rate.** BERT wants LR $\approx 2 \times 10^{-5}$; the CRF layer often wants $10^{-3}$. Use layer-wise learning rates.

10. **Assuming strict IOB = IOB2.** IOB1 uses `B-X` only at a boundary between two same-type adjacent entities; IOB2 (BIO) uses `B-X` at every entity start. Mixing conventions silently changes the label count.

---

## 4.7 Interview Questions: Sequence Labeling

### Foundational

**Q1. What is sequence labeling and how does it differ from classification?**
Classification assigns one label to a whole input. Sequence labeling assigns a label to each element of a sequence — usually each token of a sentence. Examples: POS tagging, NER, chunking, slot filling. The key challenge is that labels are not independent: transitions between adjacent tags carry information and constrain valid sequences.

**Q2. Explain BIO tagging and why it exists.**
BIO (or BIO2) marks each token as `B-X` (beginning of span of type X), `I-X` (inside span of type X), or `O` (outside any span). It exists because simple binary "in-entity / not" annotation cannot distinguish two adjacent entities: "Alice Bob went home" would be ambiguous as one span or two. `B-` tags explicitly mark boundaries. Variants (BIOES, BILOU) add `E-` (end) and `S-` (single-token) tags for stronger boundary signals.

**Q3. What is a CRF and why do we add it on top of BiLSTM or BERT?**
A CRF is a globally normalized probabilistic model over label sequences: $p(\mathbf{y} \mid \mathbf{x}) \propto \exp(\text{emissions} + \text{transitions})$. On top of BiLSTM/BERT, it replaces the independent per-token softmax with a joint distribution over all possible tag sequences. This lets the model (a) learn transition scores like "B-ORG is often followed by I-ORG, rarely by I-LOC" and (b) produce only valid sequences at decode time via Viterbi.

**Q4. What's the difference between an HMM and a CRF?**
HMMs are *generative*: they model $p(\mathbf{x}, \mathbf{y})$ with independence assumptions (output $x_t$ depends only on state $y_t$). CRFs are *discriminative*: they model $p(\mathbf{y} \mid \mathbf{x})$ directly and can use arbitrary features of the full input $\mathbf{x}$ without violating the model. For sequence labeling where we always have $\mathbf{x}$, discriminative is almost always better.

### Mathematical

**Q5. Write the linear-chain CRF probability and explain each term.**

$$
p(\mathbf{y} \mid \mathbf{x}) = \frac{1}{Z(\mathbf{x})} \exp\left(\sum_{t=1}^n \psi(y_t, \mathbf{x}, t) + \sum_{t=2}^n \phi(y_{t-1}, y_t)\right)
$$

- $\psi$: emission score from encoder (BiLSTM/BERT hidden state projected to label scores).
- $\phi$: transition score, a learned $|Y| \times |Y|$ matrix.
- $Z(\mathbf{x}) = \sum_{\mathbf{y}'} \exp(\dots)$: partition function summed over all $|Y|^n$ sequences.

**Q6. Derive the forward algorithm recursion.**
Let $\alpha_t(y)$ be the sum of unnormalized scores of all partial sequences ending in tag $y$ at position $t$:

$$
\alpha_t(y) = \sum_{\mathbf{y}_{1:t-1}} \exp\left(\sum_{s=1}^t \psi(y_s, \mathbf{x}, s) + \sum_{s=2}^t \phi(y_{s-1}, y_s)\right)\bigg|_{y_t=y}
$$

Factor out position $t$:

$$
\alpha_t(y) = \exp(\psi(y, \mathbf{x}, t)) \sum_{y'} \alpha_{t-1}(y') \exp(\phi(y', y))
$$

Base case: $\alpha_1(y) = \exp(\psi(y, \mathbf{x}, 1))$. Final: $Z(\mathbf{x}) = \sum_y \alpha_n(y)$. Complexity $O(n|Y|^2)$.

**Q7. Derive Viterbi and compare to forward.**
Viterbi replaces the sum in forward with a max:

$$
\delta_t(y) = \max_{y'} [\delta_{t-1}(y') + \phi(y', y)] + \psi(y, \mathbf{x}, t)
$$

in log-space. Keep backpointers $\beta_t(y) = \arg\max_{y'}[\dots]$. Final: $y_n^* = \arg\max \delta_n$; trace backpointers. Both are $O(n|Y|^2)$. Forward computes probabilistic quantities (partition function, marginals for training); Viterbi computes the single best sequence for decoding.

**Q8. Show how to compute the gradient of the CRF loss.**
Loss: $\mathcal{L} = -\log p(\mathbf{y}^* \mid \mathbf{x}) = -\text{score}(\mathbf{y}^*, \mathbf{x}) + \log Z(\mathbf{x})$.
- Gradient of first term w.r.t. parameters is direct (just sum features of the gold sequence).
- Gradient of $\log Z$ is $\mathbb{E}_{\mathbf{y} \sim p(\cdot \mid \mathbf{x})}[\nabla \text{score}]$, the expected features under the model. Compute via forward-backward:
  - $\alpha_t(y)$ from forward, $\beta_t(y)$ from backward.
  - Node marginal: $p(y_t = y \mid \mathbf{x}) \propto \alpha_t(y)\beta_t(y)$.
  - Edge marginal: $p(y_{t-1}=y', y_t=y \mid \mathbf{x}) \propto \alpha_{t-1}(y')\exp(\phi(y', y) + \psi(y, \mathbf{x}, t))\beta_t(y)$.
The expected feature counts are computed from these marginals; gradient follows. Modern autograd in PyTorch handles this automatically via $\log Z$ computation.

### Applied

**Q9. Design an NER system for extracting drugs, diseases, and lab values from clinical notes.**
- *Label schema*: BIO with types DRUG, DISEASE, LAB. Clinician-annotated guidelines.
- *Tokenizer*: clinical-domain BPE (ClinicalBERT, SciBERT). Must handle dosage strings like `500mg`.
- *Architecture*: ClinicalBERT encoder + per-token classifier, optionally with CRF for decode-time constraints.
- *Training*: class weighting for imbalance; 3-seed average.
- *Evaluation*: span-level F1, macro across types.
- *Post-processing*: gazetteer of known drugs to catch misses; rule-based normalization to UMLS/RxNorm.
- *Deployment*: batched inference; flag low-confidence spans for human review.

**Q10. Your NER model has 90% token accuracy but 60% entity F1. Explain.**
The dataset is dominated by `O` tokens (~85–90%). A model that labels everything `O` already achieves 85% token accuracy without extracting a single entity. Entity F1 measures span-level precision and recall on *non-O* spans, which is the actual task. Always report entity F1 for NER.

**Q11. How would you handle an NER task with nested entities (e.g., "Johns Hopkins University Hospital" where the inner span "Johns Hopkins" is an ORG and the whole span is a FAC)?**
BIO cannot represent both. Options:
- *Multiple layered taggers*: one per entity type, each with its own BIO.
- *Span-based biaffine models*: enumerate all spans up to length $L$ and classify each with types (including a null class). Used in state-of-the-art nested NER.
- *Seq2seq*: emit entity spans as structured output, e.g., `(Johns Hopkins, ORG) (Johns Hopkins University Hospital, FAC)`.

**Q12. You're serving POS tagging at 100k requests/sec. BERT is too slow. What's your plan?**
- Switch to a distilled model (DistilBERT, MiniLM) or a BiLSTM-CRF with 100d embeddings.
- POS tagging is a particularly easy task — classical feature-based CRFs achieve 97%+ on standard English. Consider Stanford CoreNLP's POS tagger if latency dominates.
- Cache on high-frequency inputs if applicable.
- For truly extreme QPS, use a lookup table for the top-N most common words; fall back to the model for the rest.

### Debugging & Failure Modes

**Q13. Your BiLSTM-CRF model produces many `O → I-X` sequences. What's wrong?**
If you're using softmax decoding ignoring transitions, those invalid sequences occur naturally. If you're using Viterbi with a CRF, there are two culprits: (a) you initialized transitions to all zeros and they haven't been trained enough — force `T[O, I-X] = -inf` as a hard constraint or train longer; (b) you forgot to use Viterbi decode and fell back to argmax — fix by using the CRF's decode method.

**Q14. Your NER model works great on Wikipedia but fails on tweets. Diagnose and fix.**
Causes: casing (tweets use inconsistent case), slang / shortened forms, hashtags, OOV names, absence of punctuation. Fix:
- Re-annotate or fine-tune on tweet NER datasets (WNUT).
- Train with lowercasing augmentation so the model doesn't rely on case.
- Add character-level features or use a byte-level model.
- Preprocess hashtags: `#CovidVaccine` → `covid vaccine`.

**Q15. You added a CRF on top of BERT and performance dropped. Why?**
Likely causes: (1) wrong learning rate for CRF parameters vs BERT. BERT wants $2\text{e}{-5}$; CRF transition matrix wants $10^{-3}$. Use parameter groups. (2) Initialization bugs: uninitialized transition matrix can give $-\infty$ to all transitions. (3) On large datasets with strong BERT, a CRF adds little and can even hurt due to added variance — consider it optional.

### Follow-up Probes

**Q16. What's the difference between the structured perceptron and a CRF?**
Both are discriminative sequence models. Structured perceptron (Collins 2002) optimizes a 0/1 error via hinge-like updates without computing a partition function — only needs Viterbi, not forward-backward. Faster training, no probabilities. CRF is the probabilistic version: computes full $p(\mathbf{y} \mid \mathbf{x})$, gives calibrated probabilities, allows for marginal inference. For production with strict calibration needs, prefer CRF.

**Q17. Why is $O(n|Y|^2)$ the natural complexity of linear-chain inference?**
Forward and Viterbi both iterate over $n$ positions, and at each position compute a $|Y| \times |Y|$ matrix-vector product (all pairs of previous-current tags). Higher-order CRFs (bigram of labels, not unigram) would be $O(n|Y|^3)$. Semi-Markov CRFs over segments become $O(n^2 |Y|^2)$. These become expensive quickly, which is why linear-chain dominated in practice.

**Q18. Can you use a CRF with a Transformer encoder? What do you gain?**
Yes — this is BERT-CRF. Gains: (a) hard constraint enforcement (no `O → I-X`), (b) small F1 gains on small data, (c) confidence calibration via CRF marginals. Losses: extra parameters ($|Y|^2$ transitions), slightly slower decoding. In large-data / high-resource settings, Transformer + softmax often catches up.

**Q19. How would you handle 1000+ fine-grained entity types?**
Linear-chain CRF's $O(n|Y|^2)$ explodes when $|Y| = 1000$. Options:
- *Hierarchical tagging*: first predict coarse type, then fine type given coarse. Reduces to $O(n|Y_\text{coarse}|^2 + n|Y_\text{fine}|^2)$ with conditioning.
- *Span-based models*: enumerate spans, classify each into types using a softmax — $O(L n)$ spans, each with a $|Y|$-way classifier. Also handles nested.
- *Retrieval-based NER*: encode each span and retrieve the nearest class prototype in a learned embedding space.

**Q20. How do you deal with extremely rare entity types where the training set has only a few dozen examples?**
- *Data augmentation*: substitute entities from gazetteers into sentence templates.
- *Transfer from similar types*: joint training where common types help rare ones via shared encoder.
- *Few-shot / prompting*: frame NER as QA ("What is the DISEASE in the following?") and use LLMs for zero-shot extraction, then use LLM outputs as training data.
- *Active learning*: ask annotators for only the hardest examples to label.
- *Class-balanced loss*: reweight the rare class heavily.

---

# 5. Text Classification

*Hierarchical attention, transformer fine-tuning.*

## 5.1 Motivation & Intuition

Text classification is the most common NLP task in production: spam vs. not-spam, positive vs. negative sentiment, toxic vs. safe, intent routing for chatbots, topic categorization for news feeds, priority triage for support tickets. The goal: assign one label (or several, if multi-label) to an entire input.

A *naive bag-of-words + logistic regression* pipeline gets you surprisingly far. Add TF-IDF weighting and you have the 2010-era baseline. It is fast, interpretable, and often within a few points of neural models on easy tasks.

But bag-of-words has three core limitations:

1. **No word order.** "The movie wasn't good" and "Was the movie good" and "Good movie, wasn't it?" all have identical bag-of-words representations but very different meanings.
2. **No compositionality.** The model must re-learn from scratch that `not good` ≈ `bad` in every context.
3. **No generalization to unseen phrases.** Every new construction is a new feature.

Neural classifiers — originally CNNs and LSTMs, now Transformers — address all three by learning distributed representations that capture word order and composition. For long documents where flat attention becomes expensive or attention dilutes across thousands of tokens, **hierarchical attention** builds up the representation in stages: words → sentences → document. This mirrors how humans read (sentence by sentence, then aggregating).

**Real-world ML impact.** Classification powers content moderation (Meta, YouTube), support ticket routing (Zendesk, Salesforce), medical triage, legal document categorization, email filtering, and intent classification in every voice assistant. Many production systems layer multiple classifiers (toxicity, PII, sentiment, topic) and make decisions based on their combined output.

## 5.2 Conceptual Foundations

### Key concepts

- **Binary vs. multi-class vs. multi-label**: binary has 2 mutually exclusive classes; multi-class has $K > 2$ mutually exclusive; multi-label allows simultaneous memberships (one email can be both "work" and "urgent").
- **Hierarchical classification**: labels are organized in a taxonomy (e.g., news → sports → football). Predict each level conditional on the previous.
- **Class imbalance**: real-world data is often skewed (99% ham, 1% spam). Requires special handling.
- **Sentence-level vs. document-level**: short (tweet), medium (review), long (research paper, legal contract). Architecture choice depends on length.
- **Hierarchical attention**: two-level model — word-level attention within each sentence, then sentence-level attention to form the document representation.
- **Fine-tuning**: update all (or some) pretrained weights on task-specific labeled data.
- **Prompt-based / zero-shot**: use an LLM with a prompt template, optionally with examples; no gradient updates.

### Architectural landscape

1. **Linear on bag-of-features (TF-IDF + logistic regression, SVM)** — baseline for any classification task. Fast, interpretable.
2. **FastText classifier** — averaged word embeddings + linear classifier, trained end-to-end. Competitive with neural models on many tasks, blazing fast.
3. **CNN for text (Kim 2014)** — convolutions over word embeddings with filters of sizes 3/4/5, then max-pool and classify. Great for sentiment on short texts.
4. **LSTM / BiLSTM** — sequential reading, last hidden state (or pooled) for classification. Supplanted by Transformers but still a useful baseline.
5. **Hierarchical Attention Network (HAN, Yang 2016)** — specifically for long documents; described in detail below.
6. **BERT / RoBERTa fine-tuning** — the default 2020s approach. Add a linear head on `[CLS]`, fine-tune.
7. **Large LM prompting (GPT-4, Claude)** — zero-shot or few-shot classification via instructions.

### Hierarchical Attention Network (HAN) in depth

Given a document with $S$ sentences and each sentence $s$ having $T_s$ words:

- **Word-level encoder**: apply BiGRU to word embeddings of sentence $s$, get hidden states $\{\mathbf{h}_{st}\}$.
- **Word-level attention**: compute attention weights $\alpha_{st}$ across words in sentence $s$; form sentence vector $\mathbf{s}_s = \sum_t \alpha_{st} \mathbf{h}_{st}$.
- **Sentence-level encoder**: apply BiGRU over sentence vectors $\{\mathbf{s}_s\}$.
- **Sentence-level attention**: compute attention weights $\beta_s$; form document vector $\mathbf{v} = \sum_s \beta_s \mathbf{s}_s$.
- **Classification head**: $p(y) = \text{softmax}(\mathbf{W} \mathbf{v} + \mathbf{b})$.

The benefits: (a) the model learns which *words* matter within each sentence and which *sentences* matter within the document, giving interpretability via attention weights; (b) the hierarchical structure reduces sequence length at each level, avoiding the quadratic cost of flat attention over long documents.

### Fine-tuning Transformers for classification

Standard recipe:
1. Load a pretrained model (BERT, RoBERTa, DeBERTa).
2. Add a small head on top of the `[CLS]` (or pooled) output — usually one or two linear layers with dropout.
3. Fine-tune end-to-end with AdamW, LR $\approx 2 \times 10^{-5}$, small batch (16–32), 3–5 epochs.
4. Use a scheduler: linear warmup then linear decay.

This gives near-SOTA results on almost any classification benchmark with modest data.

### Assumptions and failure modes

**Assumption 1: Labels are mutually exclusive (for multi-class).**
Violated when an email can be both spam and phishing. Use multi-label classification (sigmoid per class, not softmax).

**Assumption 2: Training label distribution matches production.**
Violated when prevalence shifts over time (spam tactics evolve). Requires monitoring and regular retraining.

**Assumption 3: The `[CLS]` token is a good classifier input.**
Mostly true for BERT-family fine-tuning, but the pre-fine-tune `[CLS]` without a trained head is a poor general-purpose sentence embedding.

**Assumption 4: Document fits in the context window.**
Violated for long legal, scientific, or financial documents. Need hierarchical models, long-context Transformers, or chunking strategies.

## 5.3 Mathematical Formulation

### Logistic regression baseline

$$
p(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \text{tfidf}(\mathbf{x}) + b)
$$
Trained by minimizing binary cross-entropy with L2 regularization.

### Softmax for multi-class

For $K$ classes:
$$
p(y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^\top \boldsymbol{\phi}(\mathbf{x}) + b_k)}{\sum_{k'=1}^K \exp(\mathbf{w}_{k'}^\top \boldsymbol{\phi}(\mathbf{x}) + b_{k'})}
$$
where $\boldsymbol{\phi}(\mathbf{x})$ is the feature representation — TF-IDF, averaged embeddings, or `[CLS]`.

### Hierarchical attention (HAN)

**Word-level attention** for sentence $s$:
$$
\mathbf{u}_{st} = \tanh(\mathbf{W}_w \mathbf{h}_{st} + \mathbf{b}_w), \quad \alpha_{st} = \frac{\exp(\mathbf{u}_{st}^\top \mathbf{u}_w)}{\sum_{t'} \exp(\mathbf{u}_{st'}^\top \mathbf{u}_w)}
$$
$$
\mathbf{s}_s = \sum_t \alpha_{st} \mathbf{h}_{st}
$$
$\mathbf{u}_w$ is a learnable "word-context vector" — think of it as the query "what words matter for classification?" Each hidden state $\mathbf{h}_{st}$ is transformed (via a nonlinearity) and scored by inner product with this query.

**Sentence-level attention**: identical formula with sentence encoder outputs and a sentence-context vector $\mathbf{u}_s$:
$$
\mathbf{v} = \sum_s \beta_s \mathbf{s}_s^{\text{enc}}
$$

**Loss**: cross-entropy.

### Transformer fine-tuning

Input: `[CLS] x_1 ... x_n [SEP]`. Output: hidden state $\mathbf{h}_{[\text{CLS}]}$ from final layer. Classification:
$$
p(y) = \text{softmax}(\mathbf{W}_{\text{cls}} \text{Dropout}(\tanh(\mathbf{W}_{\text{pool}} \mathbf{h}_{[\text{CLS}]})) + \mathbf{b})
$$
(for BERT; some variants simplify). Loss: cross-entropy. All parameters trained, typically with LR $2 \times 10^{-5}$.

### Multi-label

Replace softmax with per-class sigmoid:
$$
p(y_k = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}_k^\top \boldsymbol{\phi}(\mathbf{x}) + b_k)
$$
Loss: sum of binary cross-entropies. At inference, threshold (default 0.5, optimized on dev).

### Class-weighted and focal loss

For imbalance:
$$
\mathcal{L}_{\text{weighted}} = -\sum_k w_k y_k \log p_k, \quad w_k = \frac{N}{K \cdot N_k}
$$
where $N_k$ is the number of examples in class $k$.

Focal loss (Lin et al., for imbalance and hard examples):
$$
\mathcal{L}_{\text{focal}} = -\sum_k (1 - p_k)^\gamma y_k \log p_k
$$
The $(1 - p_k)^\gamma$ factor downweights easy examples (high $p_k$ for correct class), focusing training on hard ones.

### Mapping math to intuition

- Attention weights $\alpha_{st}$: "which words are important in sentence $s$"
- Sentence vector $\mathbf{s}_s$: "the concentrated meaning of sentence $s$"
- Sentence-level attention $\beta_s$: "which sentences are important for the document's label"
- `[CLS]` embedding: "a learned summary of the entire input, sculpted by fine-tuning"
- Focal loss: "don't waste gradient on examples you already get right"

## 5.4 Worked Example

**Binary sentiment: "The movie wasn't bad, actually pretty good."**

### Baseline (TF-IDF + logistic regression)

Vocabulary: `{the, movie, was, n't, bad, actually, pretty, good}`. TF-IDF vector will have high weight on `bad` (often negative) and `good` (positive), but ignore that `wasn't` negates `bad`. Prediction: ambiguous or wrong.

### CNN for text (Kim)

Filters of size 3 convolve over word embeddings:
- `the movie wasn't` → representation $\mathbf{c}_1$
- `movie wasn't bad` → $\mathbf{c}_2$ (captures `wasn't bad`!)
- `wasn't bad actually` → $\mathbf{c}_3$
- etc.

Max-pool each filter, concatenate, feed to classifier. The filter that triggers on `wasn't bad` provides positive evidence.

### BERT fine-tuning

Tokenize: `[CLS] the movie was ##n ' t bad , actually pretty good . [SEP]`. Forward through BERT. Final-layer `[CLS]` integrates the full sentence via self-attention, which has seen `wasn't` modify `bad`, and `actually pretty good` provide positive evidence. A trained classifier head outputs $p(\text{positive}) \approx 0.95$.

### Hierarchical Attention Network for a long review

Imagine a 50-sentence film review. HAN process:
1. Encode each sentence with BiGRU + word-attention, producing 50 sentence vectors.
2. Sentence vector for "The plot was incoherent" gets negative-sentiment signal; for "But the acting was superb" gets positive.
3. BiGRU over sentence vectors contextualizes them.
4. Sentence-level attention might give high weight to summary sentences like "Overall, I enjoyed it despite flaws".
5. Weighted sum → document vector → classifier.

Visualizing attention: word-level highlights reveal which words drove each sentence's representation; sentence-level highlights reveal which sentences drove the prediction. Useful for debugging and user-facing explanations.

### Fine-tuning BERT — concrete hyperparameters

- Dataset: IMDb (25k train, 25k test, balanced).
- Model: `bert-base-uncased`.
- Max length: 256 tokens (truncate longer reviews from the end, or use sliding windows).
- Batch: 16.
- Optimizer: AdamW, LR $2 \times 10^{-5}$, weight decay 0.01.
- Schedule: linear warmup over 10% of steps, linear decay.
- Epochs: 3.
- Dropout: 0.1.
- Expected accuracy: ~92–93% test. RoBERTa-large pushes to ~96%.

## 5.5 Relevance to ML Practice

**Where it appears.**

- **Content moderation**: toxicity, harassment, misinformation classifiers at massive scale.
- **Intent detection** in dialogue systems: given user utterance, predict one of dozens of intents.
- **Email/spam**: billions of decisions per day; latency and cost dominant.
- **Topic categorization**: news, research papers, customer support tickets.
- **Safety / compliance**: PII detection, regulated content.
- **Monitoring**: drift detection over predicted class distributions.

**When to use what.**

- *Short texts, simple task, tight latency*: TF-IDF + logistic regression, or FastText.
- *Medium-length texts, moderate resources*: BERT fine-tuning.
- *Long documents*: hierarchical models (HAN), Longformer/BigBird, or chunked BERT with majority vote / sentence-level aggregation.
- *Very few labels*: few-shot LLM prompting; SetFit (contrastive + classifier).
- *No labels, rough taxonomy*: zero-shot NLI (use entailment scores as class scores) or LLM.
- *Heavy class imbalance or long-tail labels*: focal loss, hierarchical classification, or retrieval-based classifiers.

**When NOT to use a heavy neural model.**

- If TF-IDF + LR matches neural within 1% — production cost/latency wins.
- If labels are trivially determinable by rules (regex on keywords for certain PII).
- If the label set is enormous (millions of fine categories) — retrieval or hierarchical schemes beat flat softmax.

**Trade-offs.**

| Approach | Accuracy | Latency | Cost | Interpretability |
|---|---|---|---|---|
| Rules / regex | Low-medium | Nanoseconds | None | High |
| TF-IDF + LR | Medium | Microseconds | Low | High (coefficients) |
| FastText | Medium-high | Microseconds | Low | Medium |
| CNN/LSTM | High | Milliseconds | Low | Low |
| HAN | High (long docs) | Milliseconds | Low | Medium (attention) |
| BERT fine-tune | Very high | 10–50ms | Medium | Low |
| BERT distilled | High | 1–5ms | Low | Low |
| LLM zero-shot | Varies | 100–1000ms | High | Medium (rationale) |
| LLM few-shot | Very high | 100–1000ms | Very high | Medium |

## 5.6 Common Pitfalls

1. **Data leakage via duplicates.** Near-duplicate examples across train and test inflate accuracy. Deduplicate with hashing or min-hash LSH before splitting.

2. **Evaluating only accuracy on imbalanced data.** On a 99/1 split, predicting the majority class gets 99% accuracy and is useless. Use F1, PR-AUC, or the confusion matrix.

3. **Wrong loss for multi-label.** Using softmax (exclusive) when labels can co-occur. Use sigmoid + BCE per class.

4. **Using a fixed 0.5 threshold** for multi-label or calibrated binary classification. Tune threshold on dev to optimize F1 or business-relevant metric.

5. **Not stratifying splits.** For rare classes, a random split can leave none in dev or test. Use stratified splits or time-based splits for temporal data.

6. **Truncating long documents from the end.** Important information may be at the start or the end. Try head+tail truncation: keep first and last $n/2$ tokens.

7. **Overfitting on small classes.** Massive class weights can cause the model to obsess over a single rare class. Combine weighting with focal loss or oversampling carefully.

8. **Dependence on dataset-specific artifacts.** Models pick up spurious cues (length, presence of certain words, annotator bias). Use adversarial evaluation sets (HANS, Checklist) to test robustness.

9. **Multiplying latency by batching at inference.** Dynamic batching helps throughput but increases tail latency. Measure P99, not just mean.

10. **Fine-tuning instability.** BERT on small datasets can give variable results across seeds (seen in §3). Run 3+ seeds.

11. **Not calibrating probabilities.** Even when accurate, neural classifiers often output miscalibrated confidences (Guo et al. 2017). Use temperature scaling or Platt scaling post-hoc.

---

## 5.7 Interview Questions: Text Classification

### Foundational

**Q1. When would you choose logistic regression with TF-IDF over a fine-tuned BERT?**
When latency and cost matter more than a few F1 points; when the task is simple (spam, clear sentiment); when you need interpretability (feature weights); when you have a small labeled dataset where BERT would overfit; when you have millions of requests per second and deploying BERT is cost-prohibitive.

**Q2. What's the difference between multi-class and multi-label classification?**
Multi-class: each example belongs to exactly one of $K$ classes. Output is a softmax; predict $\arg\max$. Multi-label: each example can belong to multiple classes simultaneously (a movie can be "comedy" + "romance"). Output is $K$ independent sigmoids; predict each with a threshold. Using softmax for multi-label is a common bug.

**Q3. Explain hierarchical attention in two sentences.**
Encode each sentence by attending over its words, then encode the document by attending over its sentences. Each level has a learnable context vector acting as a "what matters?" query, giving interpretable attention weights and letting the model focus on salient content.

**Q4. Why does BERT's `[CLS]` work as a classifier input?**
During pretraining, `[CLS]` was used as the input to the Next Sentence Prediction head, giving it a signal to encode sentence-level information. During fine-tuning on the classification task, the model further optimizes `[CLS]` to carry exactly the information needed for the labels. After fine-tuning, `[CLS]` is a task-specific sentence embedding.

### Mathematical

**Q5. Derive the gradient of binary cross-entropy w.r.t. logit.**
Loss $L = -y \log \sigma(z) - (1-y) \log(1 - \sigma(z))$. Using $\frac{d}{dz}\sigma(z) = \sigma(z)(1-\sigma(z))$:

$$
\frac{\partial L}{\partial z} = -y(1 - \sigma(z)) + (1-y)\sigma(z) = \sigma(z) - y
$$

Clean form: gradient is prediction minus target. This is why cross-entropy pairs naturally with sigmoid and why it has better gradients than squared loss for classification.

**Q6. Why use softmax + cross-entropy and not MSE for classification?**
Softmax produces a valid probability distribution; MSE does not. Cross-entropy's gradient is $\mathbf{p} - \mathbf{y}$, proportional to the error, which provides strong gradients even when predictions are very wrong. MSE's gradient vanishes as predictions saturate (softmax outputs near 0 or 1), causing slow learning. Information-theoretically, cross-entropy is the KL divergence to the true label distribution, which is the natural measure.

**Q7. Focal loss: explain the terms and why it helps with imbalance.**
$\mathcal{L} = -(1 - p_t)^\gamma \log p_t$, where $p_t$ is the model's probability on the correct class. $(1-p_t)^\gamma$ is a modulating factor: when the model is confident and correct ($p_t \to 1$), this factor $\to 0$, so loss is small. When the model is wrong or unconfident ($p_t$ small), factor $\approx 1$ and full cross-entropy applies. Effect: easy examples contribute less, hard examples more. Helpful when the dataset has many easy negatives (e.g., object detection, highly imbalanced classification).

**Q8. Derive the HAN word-level attention weight.**
Given BiGRU hidden states $\{\mathbf{h}_{st}\}$ and learnable parameters $\mathbf{W}_w, \mathbf{b}_w, \mathbf{u}_w$:
$\mathbf{u}_{st} = \tanh(\mathbf{W}_w \mathbf{h}_{st} + \mathbf{b}_w)$ (transform each hidden state into an attention space),
$\alpha_{st} = \frac{\exp(\mathbf{u}_{st}^\top \mathbf{u}_w)}{\sum_{t'} \exp(\mathbf{u}_{st'}^\top \mathbf{u}_w)}$ (softmax over scores from a learned "query" vector $\mathbf{u}_w$).
Sentence vector $\mathbf{s}_s = \sum_t \alpha_{st} \mathbf{h}_{st}$.

### Applied

**Q9. You need to classify 10M support tickets into 200 categories. Design the pipeline.**
- *Data*: split stratified 80/10/10. Check for near-duplicates and leakage.
- *Baseline*: TF-IDF + one-vs-rest logistic regression or a linear SVM per class; establish a lower bound.
- *Next step*: FastText classifier — fast, competitive, multi-class native.
- *Production model*: DistilBERT or MiniLM fine-tuned — best accuracy-latency tradeoff.
- *Class imbalance*: check per-class frequency; use class weights for tail classes; consider hierarchical classification if taxonomy exists.
- *Evaluation*: macro-F1 (prevents majority-class domination) + top-K accuracy if the downstream step allows multiple candidates.
- *Monitoring*: track per-class F1 over time; alert on drift.

**Q10. Your binary spam classifier has 99% accuracy but 40% F1 on the spam class. What's happening?**
Class imbalance. Most emails are not spam, so predicting "not spam" always gives 99% accuracy. F1 reveals that actual spam detection is poor. Fixes: class-weighted loss, oversample spam, SMOTE (less effective for text), threshold tuning (lower threshold favors recall), or use PR-AUC instead of ROC-AUC.

**Q11. For a document classifier where the relevant signal appears *anywhere* in a 20-page document, how do you handle BERT's 512-token limit?**
Options:
- *Chunk and aggregate*: split into 512-token chunks, classify each, aggregate via max-pool, mean-pool, or a learned aggregator.
- *Hierarchical*: HAN or a sentence-Transformer encoding each sentence + a sentence-level Transformer aggregating.
- *Long-context models*: Longformer (4k-16k tokens with sliding window + global attention), BigBird, or modern LLMs with long contexts.
- *Key-sentence extraction first*: use a lightweight extractor to select the top-K most important sentences, then classify the concatenation.

**Q12. When would you use zero-shot LLM classification vs. fine-tuned BERT?**
Zero-shot LLM: (a) no labels available, (b) label set is dynamic (new classes added), (c) need to classify across many related tasks without retraining, (d) production can tolerate higher latency and cost. Fine-tuned BERT: (a) you have labeled data, (b) you need low cost and latency, (c) you need stable, reproducible predictions, (d) the task is well-defined and unlikely to change.

### Debugging & Failure Modes

**Q13. Your model achieves 95% train accuracy but only 70% test. Your dev accuracy is 88%. Diagnose.**
Signs of overfitting. The gap between dev and test suggests distribution shift between dev and test splits, possibly non-random splitting. Fixes: (a) verify splits are stratified and not temporally broken; (b) add regularization — dropout, weight decay, early stopping; (c) reduce epochs; (d) check for duplicates across train/dev — if dev has near-duplicates of train, dev accuracy is inflated and test is the "real" performance.

**Q14. Your classifier makes great predictions on clean text but fails on text with typos. Mitigate.**
- Use a subword or FastText-based model (character n-grams).
- Train with data augmentation: synthetic typos (swap/delete/insert/replace characters), keyboard-neighbor substitutions.
- Use a byte-level model (ByT5, Charformer) for maximum robustness.
- Add a spell-correction pre-processing stage if latency permits.

**Q15. A deployed classifier's accuracy silently dropped 5% over 6 months. How do you detect and fix?**
Detect:
- Monitor confidence distribution over production traffic; a shift may indicate concept drift.
- Sample and re-label periodically; compute current accuracy on recent data.
- Monitor class prevalence distributions (predicted vs. expected).
- Use drift-detection tests (KS test on features, MMD on embeddings).

Fix:
- Retrain on fresh data.
- Continual learning with replay buffer of old data to prevent catastrophic forgetting.
- Consider ensemble of old + new model, transitioning weight over time.

### Follow-up Probes

**Q16. What is label smoothing and when does it help?**
Label smoothing replaces hard one-hot targets with a mixture: $y_k = (1 - \epsilon)$ for the correct class and $\epsilon/(K-1)$ for others, with $\epsilon$ typically 0.1. It prevents the network from becoming over-confident, improves calibration, and often gives small accuracy gains. It can hurt when the task is a true one-hot ground truth (some adversarial settings) or when model size is very small.

**Q17. Explain why attention weights are *not* always interpretable.**
Attention weights indicate where the model "looked", but:
- Multiple attention patterns can yield the same output (non-identifiability).
- High attention ≠ causal importance; ablation studies sometimes show that zeroing high-attention tokens doesn't change predictions (Jain & Wallace, 2019).
- In deep models, attention at one layer is combined with subsequent layers, so interpreting any one layer in isolation is misleading.
Use attention as a rough visualization tool, not a rigorous explanation. Prefer attribution methods like integrated gradients or Shapley for genuine interpretability.

**Q18. How does active learning help classification with limited labels?**
Instead of labeling randomly, pick examples where the model is *uncertain* (entropy, margin) or *disagreement-prone* (across a model ensemble). These examples, once labeled, give the most information per label. Typical active learning loop: train → score unlabeled pool → pick top-K uncertain → label → retrain. Can reduce labeling cost by 2–10× to reach target accuracy. Caveats: can bias the dataset toward ambiguous examples; combine with diversity sampling for robustness.

**Q19. What is "prompt tuning" for classification, and when is it preferable to fine-tuning?**
Prompt tuning: freeze the LM; learn a small set of soft (continuous) prompt vectors prepended to the input. Only a few thousand parameters are trained. Preferable when: (a) the LM is huge and full fine-tuning is infeasible; (b) you have many tasks and want a small per-task parameter count; (c) you want to preserve the base model's general capabilities and avoid catastrophic forgetting. Close in quality to fine-tuning for large base models; worse for small base models.

**Q20. How would you build a classifier where the label taxonomy changes monthly (new categories added)?**
Options:
- *Retrieval-based*: embed examples with a frozen encoder; embed class prototypes from a few labeled examples per class; classify via nearest prototype. Adding a new class is trivial — just add its prototype.
- *Zero-shot NLI*: frame classification as "does this document entail the hypothesis 'This document is about [class X]?'". Add a class by adding a hypothesis.
- *LLM prompting with class descriptions*: include definitions of current classes in the prompt. Adding a class updates the prompt.
- *Few-shot*: a frozen sentence encoder + a small classifier head that can be quickly retrained with a few examples (SetFit).

---

# 6. Text Generation

*Decoding strategies (greedy, beam, nucleus); controllable generation.*

## 6.1 Motivation & Intuition

Given a language model that assigns probabilities to token sequences, how do we actually *produce* text from it? This is the problem of **decoding**. The model provides, at each step, a probability distribution $p(x_t \mid x_{1:t-1})$ over the vocabulary. Decoding is the algorithm that turns this sequence of distributions into an output sequence.

Decoding matters far more than most people expect. Two models of identical quality can produce very different outputs under different decoders: one writes fluently, the other repeats itself endlessly. A powerful GPT-4 used with bad decoding produces garbage; a smaller model with well-tuned sampling produces engaging text.

The tension in decoding is:

- **Exact maximum likelihood is misleading.** The highest-probability sequence is often short, generic, and repetitive ("I I I I..."). Language models have "mass" in boring regions.
- **Pure random sampling is chaotic.** Low-probability tokens sneak in and derail coherence.

Modern decoding methods (top-k, nucleus/top-p, temperature, repetition penalties) navigate between these extremes.

**Controllable generation** goes further: can we guide the model to produce text with specific attributes (sentiment, style, topic, factuality)? Techniques range from prompt engineering to control codes (CTRL), conditional training (T5), classifier-guided decoding (PPLM, FUDGE), and RLHF/DPO which reshape the probability distribution itself.

**Real-world ML impact.** Every LLM product — ChatGPT, Claude, Gemini, Copilot — exposes decoding parameters (temperature, top_p) and sometimes controllability via system prompts. Production systems carefully tune decoders for each application: factual QA uses low temperature; creative writing uses nucleus sampling with moderate temperature; code completion uses beam search or constrained decoding.

## 6.2 Conceptual Foundations

### Autoregressive generation

Language models factorize sequence probability via the chain rule:
$$
p(\mathbf{x}) = \prod_{t=1}^n p(x_t \mid x_1, \dots, x_{t-1})
$$
At each step, sample or select a token, append to context, repeat until end-of-sequence or max length.

### Decoding strategies

**Greedy decoding.** At each step, pick the argmax.
$$
x_t = \arg\max_v p(v \mid x_{1:t-1})
$$
Fast, deterministic, but often falls into repetition loops (high-probability continuations often lead to more of the same).

**Beam search.** Maintain the top-$B$ partial sequences (beams), extending each by every token, keeping the top-$B$ of the expanded set at each step. Used heavily in machine translation and summarization.

**Temperature sampling.** Rescale logits by $1/T$ before softmax:
$$
p_T(v) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}
$$
$T = 1$ is the original distribution. $T \to 0$ approaches greedy. $T \to \infty$ approaches uniform. Higher $T$ = more diverse, less coherent.

**Top-k sampling.** At each step, restrict the distribution to the top $k$ most probable tokens, renormalize, sample.

**Top-p (nucleus) sampling** (Holtzman et al., 2019). Dynamically pick the smallest set of tokens whose cumulative probability exceeds $p$, renormalize, sample. Adaptive to the distribution's sharpness: in confident regions (next token obvious), the nucleus is small; in uncertain regions, it's larger.

**Repetition penalty.** Penalize logits of tokens that recently appeared to avoid loops:
$$
z_v \leftarrow z_v / r \quad \text{if } v \in \text{recent tokens}
$$
with $r > 1$.

**Contrastive search** (Su et al., 2022). Pick the token that maximizes a trade-off between probability and diversity from previous context.

**Constrained decoding.** Force outputs to obey a grammar, regex, or a set of allowed tokens (e.g., only valid JSON, only from a schema).

### Controllable generation

- **Prompt engineering.** Condition on instructions: "Write in a formal tone about X."
- **Control codes.** Prepend tokens like `<pos>` or `<formal>` that the model was trained to condition on (CTRL).
- **Classifier-guided decoding.** A separate classifier scores desirability; blend with LM logits to steer generation (PPLM, FUDGE).
- **Conditional training.** Train the model with explicit condition inputs (style tokens, source document).
- **RLHF / DPO.** Shape the model's distribution post hoc via preference-based training so it produces desired outputs naturally without explicit constraints.

### Assumptions and failure modes

**Assumption 1: High-probability sequences are the "best" outputs.**
Violated! Famous finding of Holtzman et al.: the mode of a modern LM's distribution is a short, dull, repetitive sequence. Good generations live in the *middle* of the probability mass.

**Assumption 2: The model's distribution is calibrated.**
Large pretrained LMs are fairly well calibrated at the token level, but modern RLHF-tuned models become *miscalibrated*, over-confident on the "preferred" response.

**Assumption 3: Decoding is independent of training.**
Violated. Models trained with exposure bias (teacher forcing) behave differently at inference when they see their own outputs. Scheduled sampling, minimum-risk training, and RL address this.

**Assumption 4: Length is controllable via max_length.**
Partially violated: models have learned biases about sequence length. Controlling for length often requires explicit length tokens or reward shaping.

## 6.3 Mathematical Formulation

### Beam search

At step $t$, maintain a beam $\mathcal{B}_t$ of size $B$ containing partial sequences and their scores. For each $(\mathbf{y}, s) \in \mathcal{B}_{t-1}$, expand:
$$
\mathcal{C}_t = \{(\mathbf{y} \cdot v, s + \log p(v \mid \mathbf{y})) : v \in V\}
$$
Keep top $B$ by score: $\mathcal{B}_t = \text{top-}B\,\mathcal{C}_t$.

A pure log-probability objective favors short sequences. Length normalization:
$$
\text{score}(\mathbf{y}) = \frac{1}{|\mathbf{y}|^\alpha} \sum_{t=1}^{|\mathbf{y}|} \log p(y_t \mid y_{<t})
$$
with $\alpha \in [0.6, 1.0]$ typical for NMT.

### Temperature and entropy

Temperature $T$ controls the entropy of the sampling distribution. $T = 1$: original entropy. $T \to 0$: deterministic. $T \to \infty$: uniform.

### Top-k

Given logits $\mathbf{z}$, set $z_v = -\infty$ for all but the top $k$, then softmax.

### Top-p (nucleus)

Sort tokens by probability descending. Let $V_p$ be the smallest set such that $\sum_{v \in V_p} p(v) \ge p$. Set $z_v = -\infty$ for $v \notin V_p$, renormalize.

### Classifier-guided generation (FUDGE, Yang & Klein 2021)

Want to generate a sequence satisfying attribute $a = 1$. By Bayes:
$$
p(x_t \mid x_{<t}, a=1) \propto p(x_t \mid x_{<t}) \cdot p(a=1 \mid x_{\le t})
$$
The first factor is the base LM; the second is an attribute classifier. Train a lightweight classifier; use its scores to reweight logits at each step.

### RLHF — mathematical sketch

1. Train reward model $r(\mathbf{x})$ from human preference pairs.
2. Fine-tune LM with PPO to maximize $\mathbb{E}[r(\mathbf{x})] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$.

The KL term keeps the fine-tuned policy close to the pretrained reference, preventing reward hacking. DPO (Direct Preference Optimization) skips RL by turning preferences into a classification loss on pairs of responses, directly fine-tuning the policy.

### Mapping math to intuition

- $\arg\max p(v)$: greedy — "always pick what you're most sure of". Short and repetitive.
- Beam search: "carry forward the top $B$ hypotheses to avoid early mistakes". Expensive but higher-quality for MT.
- Temperature: "how much do I trust the model's confidence?"
- Top-p nucleus: "take the smallest set of sensible tokens". Adapts to context.
- RLHF: "reshape the whole distribution so the model naturally produces preferred outputs".

## 6.4 Worked Example

**Decoding comparison on a 5-token continuation.**

Suppose at the first step the model outputs the distribution:

| Token | p |
|---|---|
| the | 0.25 |
| a | 0.18 |
| my | 0.12 |
| this | 0.10 |
| some | 0.08 |
| (other 50k tokens) | small |

**Greedy**: pick `the`. Deterministic, may lead to generic continuation.

**Top-k with k=5**: restrict to {the, a, my, this, some}, renormalize, sample.

**Top-p with p=0.5**: cumulative sum over the sorted list: 0.25, 0.43, 0.55 — the cumulative crosses 0.5 at `my`. Nucleus = {the, a, my}.

**Temperature T=2** rescales: for token with original logit $z$, new probability $\propto \exp(z/2)$. The distribution flattens.

**Concrete repetition issue.** A small model generating with greedy:
> "The cat sat on the mat. The cat sat on the mat. The cat sat on..."

With nucleus sampling $p = 0.95$:
> "The cat sat on the mat and closed its eyes, purring contentedly while the rain drummed against the window."

**Constrained decoding for JSON.** Build a finite-state automaton that tracks what tokens are legal at each position. At each decode step, mask out logits of tokens that would violate the schema. Guaranteed valid outputs regardless of LM quality.

## 6.5 Relevance to ML Practice

**When to use which decoder.**

- *Factual QA, math, code*: greedy or beam (B=4–8). Low or zero temperature.
- *Machine translation*: beam search (B=4–12) with length normalization.
- *Summarization*: beam with length penalty; sometimes nucleus for abstractive diversity.
- *Creative writing, dialogue*: nucleus (p=0.9–0.95) with temperature 0.7–1.0.
- *Structured output (JSON, SQL)*: constrained decoding + low temperature.

**When NOT to use beam search.** Open-ended generation: beam collapses to the mode, producing dull, generic text.

**Controllability trade-offs.**

| Method | Training cost | Inference cost | Flexibility |
|---|---|---|---|
| Prompting | None | Minimal | High |
| Control codes | Yes (re-train) | Minimal | Moderate |
| FUDGE / classifier-guided | Small (classifier) | Moderate | High |
| RLHF / DPO | Expensive | None (baked in) | Moderate |
| Constrained decoding | None | Small | Hard constraints only |

## 6.6 Common Pitfalls

1. **Using beam search for open-ended generation.** The result is repetitive, generic, mode-collapsed text. Use sampling.
2. **Using greedy and being surprised by loops.** "The the the..." is almost guaranteed without repetition penalty.
3. **Very high temperature.** T=2+ produces gibberish. T rarely exceeds 1.2 in practice.
4. **Nucleus with very small p.** p < 0.5 reduces to near-greedy.
5. **Length bias in beam search.** Without length normalization, short outputs always win.
6. **Repetition penalty too aggressive.** Can prevent legitimate repeated words (names, technical terms).
7. **Ignoring prompt format at inference.** Chat-tuned models have specific formatting; wrong format degrades quality drastically.
8. **Confusing calibration with output quality.** A model can be well-calibrated yet produce poor text.
9. **Assuming RLHF = aligned.** RLHF-tuned models are optimized for the preference distribution they saw.
10. **Decoding hyperparameter coupling.** temperature × top-p × top-k all interact.

---

## 6.7 Interview Questions: Text Generation

### Foundational

**Q1. Why is greedy decoding often bad?**
Greedy is myopic: at each step, it takes the locally best token, which can lead into low-quality regions globally. Combined with exposure bias (models trained via teacher forcing behave poorly on their own outputs), greedy often produces repetition. The highest-probability full sequence under autoregressive LMs is often short and generic.

**Q2. Explain top-k vs. top-p.**
Top-k always samples from the $k$ most probable tokens. Top-p samples from the smallest set whose cumulative probability exceeds $p$. Top-p adapts: when the distribution is sharp, the nucleus is small; when flat, larger. Top-k is simpler but can include improbable tokens when sharp or exclude probable ones when flat.

**Q3. What does temperature = 0 do?**
$T \to 0$ concentrates all mass on the max logit, equivalent to greedy decoding.

**Q4. Why does beam search work well for translation but poorly for open-ended generation?**
Translation has a relatively peaked posterior — there's usually a "right" translation. Open-ended generation has diffuse posteriors with many valid continuations; beam collapses to the mode.

### Mathematical

**Q5. Derive length-normalized beam score.**
A partial sequence of length $t$ has score $s = \sum_{i=1}^t \log p(y_i \mid y_{<i})$. Each term is negative, so longer sequences have lower scores. Normalization: $s / t^\alpha$ for $\alpha \in [0.6, 1.0]$. With $\alpha = 1$ the score becomes mean log-prob.

**Q6. Compute $H(p_T)$ in terms of $T$.**
$p_T(v) = \frac{\exp(z_v/T)}{Z_T}$ with $Z_T = \sum \exp(z_v/T)$. Entropy:

$$
H(p_T) = \log Z_T - \frac{1}{T}\mathbb{E}_{p_T}[z]
$$

$H$ is monotonically increasing in $T$: 0 at $T \to 0$, $\log |V|$ at $T \to \infty$.

**Q7. Why the KL penalty in RLHF?**
Objective: $\max_\pi \mathbb{E}_{\mathbf{x} \sim \pi}[r(\mathbf{x})] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})$. Prevents policy drift — stops reward hacking, preserves fluency, prevents mode collapse. $\beta$ trades off: too small → hacking; too large → no improvement.

**Q8. Nucleus sampling algorithm.**
```
1. Apply temperature: z ← z / T
2. Sort indices by z descending
3. Compute softmax probabilities
4. Compute cumulative probabilities along sort order
5. Find smallest prefix whose cumulative exceeds p
6. Set z_v ← -inf for all v outside the nucleus
7. Resample softmax; draw one token
```

### Applied

**Q9. Customer-service chatbot — choose decoding.**
- Factual intents: greedy/$T=0$. Deterministic.
- Open-ended help: nucleus $p=0.9$, $T=0.7$.
- Empathy: $T=0.8$, $p=0.9$.
- Repetition penalty $r=1.1$.
- Constrained decoding for structured fields.

**Q10. Reliable JSON outputs.**
Option 1: constrained decoding with FSM over schema. Option 2: prompt + parse + retry. Option 3: tool use. Combine: constrained decoding + semantic validation.

**Q11. Evaluating generated text.**
- *Translation*: BLEU, chrF, COMET.
- *Summarization*: ROUGE, BERTScore, human faithfulness.
- *Dialogue*: pairwise human eval, LLM-as-judge.
- *Open-ended*: distinct-n, MAUVE, human eval.

**Q12. "More creative" outputs — what changes?**
Increase temperature (0.8–1.0), top-p (0.92–0.95), decrease repetition penalty, generate multiple samples and show diverse ones. Trade-off: creativity ↑ = hallucinations ↑.

### Debugging & Failure Modes

**Q13. Model outputs `...................`. Why?**
Mode collapse — high-probability attractor. Causes: no repetition penalty, low temperature, degenerate fine-tuning, tokenizer bug. Fixes: add repetition penalty ($r=1.1$–$1.3$), increase temperature, inspect last-token distribution.

**Q14. Nucleus sampling with $p=0.95$ still shows repetition.**
If distribution is peaked (context makes one token 98% likely), nucleus contains just that token — equivalent to greedy. Fix: add repetition penalty on top of nucleus; use contrastive search or typical sampling.

**Q15. RLHF model hallucinates more than SFT.**
Reward hacking: raters reward confident, fluent responses; model learns to produce confident-sounding outputs even when unsure. Mitigations: increase KL penalty, improve reward-model coverage, explicitly reward hedging and uncertainty, use factuality-based rewards.

### Follow-up Probes

**Q16. What is exposure bias?**
During training, the model sees ground-truth previous tokens (teacher forcing). At inference, it sees its own outputs. Small errors compound. Mitigations: scheduled sampling, minimum-risk training, RL.

**Q17. Why does beam search "collapse" for open-ended tasks?**
Beams all converge to similar high-probability sequences. Diverse beam search penalizes similarity between beams to force diversity.

**Q18. Contrastive search intuition.**
Pick token $x_t$ maximizing $(1-\alpha) \cdot p(x_t) - \alpha \cdot \max_{s < t} \cos(h_{x_t}, h_s)$. Trades off likelihood and degeneration avoidance (low cosine similarity to recent hidden states).

**Q19. Inference-time speedups.**
- KV caching (save $\mathbf{K}, \mathbf{V}$ from previous steps; standard).
- Speculative decoding (small model drafts tokens, large model verifies in parallel).
- Medusa heads (model predicts multiple tokens in parallel, verifies).
- Quantization (INT8, INT4).
- Continuous batching.

**Q20. DPO vs. PPO RLHF.**
PPO: separate reward model, PPO updates policy to max reward with KL constraint. DPO: derives closed-form that replaces RL with a classification loss on preference pairs. DPO is simpler, more stable, no reward model needed. PPO can reach higher quality when reward model is very accurate and compute is abundant.

---

# 7. Information Extraction

*Relation extraction, event extraction, open information extraction.*

## 7.1 Motivation & Intuition

Named Entity Recognition finds *what* entities exist in text. Information Extraction (IE) goes further: it extracts *structured facts* about those entities — relations (who works where), events (who did what to whom, when), and attributes (age, title). The output is typically a knowledge graph or a tabular database built from unstructured text.

Consider the sentence: *"Satya Nadella was appointed CEO of Microsoft in 2014."*

- NER extracts `[PERSON Satya Nadella]`, `[ORG Microsoft]`, `[DATE 2014]`, `[TITLE CEO]`.
- Relation extraction outputs `employed_by(Satya Nadella, Microsoft)`, `has_title(Satya Nadella, CEO)`.
- Event extraction outputs `Event(type=APPOINTMENT, person=Satya Nadella, org=Microsoft, role=CEO, time=2014)`.
- Open IE (no fixed schema) outputs `(Satya Nadella; was appointed CEO of; Microsoft)` and `(Satya Nadella; was appointed CEO of Microsoft; in 2014)`.

The goal: transform the world's unstructured text into structured data that can be queried, reasoned over, aggregated, and integrated with databases.

**Real-world ML impact.** IE powers knowledge graphs (Google Knowledge Graph, Wikidata), financial intelligence (extracting M&A events from news), biomedical discovery (extracting drug-disease interactions from research papers), cybersecurity (extracting indicators of compromise from threat reports), legal analytics (extracting clauses and obligations from contracts), and scientific literature mining.

## 7.2 Conceptual Foundations

### Core tasks

**Relation Extraction (RE).** Given a sentence and two entity spans, classify their relation from a predefined schema (e.g., `FreeBase` relations, `TACRED` 41 types). Types:
- *Binary*: `(subject, relation, object)` triples.
- *N-ary*: relations with more than two arguments (rare).
- *Cross-sentence*: relation spans multiple sentences (document-level RE).

**Event Extraction (EE).** Given a text, identify:
1. *Trigger*: the word/phrase indicating the event (`fired`, `acquired`, `appointed`).
2. *Event type*: from a schema (ACE2005: 33 types; specialized schemas for domains).
3. *Arguments*: participants playing roles (Agent, Patient, Time, Place, Instrument...).

**Open Information Extraction (OpenIE).** Extract `(subject, predicate, object)` triples *without a predefined schema*. The predicate is a span from the sentence, often a verb phrase. Scales to new domains with no retraining; noisy and unnormalized.

### Approaches

**Classical (pre-2015).** Hand-crafted features on dependency paths, part-of-speech patterns, named-entity types, WordNet semantics. Feed to SVM or logistic regression.

**Neural RE (2015–2018).**
- *CNN over sentence with positional features*: encode each word's distance to subject and object entities as extra embedding dimensions.
- *BiLSTM over dependency path* between two entities: strong signal for relations.
- *Attention-based models*: learn which parts of the sentence matter for each entity pair.

**Transformer RE (2018+).** Encode the sentence with BERT; use a head on the hidden states of the subject and object (or a special marker) to predict the relation.

**Typed entity markers.** Insert special tokens around entities: `[E1-PER] Satya Nadella [/E1-PER] was appointed CEO of [E2-ORG] Microsoft [/E2-ORG]`. This lets BERT learn relation-specific representations directly.

**Event extraction pipelines.**
1. Trigger detection: sequence labeling (every token → event type or O).
2. Argument detection: per event, classify each candidate entity as playing a specific role.
- *Joint models*: predict triggers + arguments together (reduces cascading errors).

**OpenIE.** Classical: rule-based on dependency parses (Stanford OpenIE, ReVerb). Neural: seq2seq — encode sentence, generate triples as text.

**Distant supervision.** Expensive to hand-label RE data. Trick (Mintz et al., 2009): if a knowledge base says `employed_by(Nadella, Microsoft)`, assume *every* sentence mentioning both is evidence for this relation. Produces noisy labels at scale; address with multi-instance learning or partial-label learning.

### Assumptions and failure modes

**Assumption 1: Relations are sentence-scoped.**
Violated frequently: "John founded Apple. He left in 1985." The relation `left_company(John, Apple)` requires resolving `He → John` and seeing both sentences. Document-level RE and cross-sentence RE address this.

**Assumption 2: The schema covers what matters.**
Violated for emerging domains (new drug mechanisms, new crime types). OpenIE or dynamic schemas (zero-shot RE) are alternatives.

**Assumption 3: Entity boundaries are known.**
If NER is wrong, RE is wrong. End-to-end models that jointly predict entities and relations reduce cascading errors.

**Assumption 4: Distant-supervision assumption.**
Wrong often: a sentence mentioning (Nadella, Microsoft) may talk about Nadella *visiting* Microsoft, not being employed by it. Multi-instance learning treats this as label noise at the bag level rather than instance.

## 7.3 Mathematical Formulation

### Relation classification with BERT

Given input sentence tokenized and wrapped with entity markers:
`[CLS] ... [E1] subj [/E1] ... [E2] obj [/E2] ... [SEP]`

Encode with BERT to get hidden states $\{\mathbf{h}_i\}$. Form a relation representation:
$$
\mathbf{r} = [\mathbf{h}_{[\text{CLS}]}; \mathbf{h}_{[\text{E1}]}; \mathbf{h}_{[\text{E2}]}]
$$
(concatenation). Classify:
$$
p(r \mid s, e_1, e_2) = \text{softmax}(\mathbf{W}_r \mathbf{r} + \mathbf{b}_r)
$$
Loss: cross-entropy. Often the relation set includes a `no_relation` class.

### Multi-instance learning for distant supervision

Given a bag $\mathcal{B}_{ij}$ of sentences each mentioning entity pair $(e_i, e_j)$, and the KB's label $r_{ij}$, we want to train without knowing which sentence actually expresses the relation. Two common objectives:

*At-least-one (Mintz heuristic)*: assume at least one sentence expresses $r_{ij}$:
$$
p(r_{ij} \mid \mathcal{B}_{ij}) = \max_{s \in \mathcal{B}_{ij}} p(r_{ij} \mid s)
$$

*Attention over sentences* (Lin et al., 2016):
$$
\mathbf{b} = \sum_{s \in \mathcal{B}_{ij}} \alpha_s \mathbf{h}_s, \quad \alpha_s = \text{softmax}(\mathbf{h}_s^\top \mathbf{W} \mathbf{r})
$$
where $\mathbf{r}$ is the relation query vector. The attention allows the model to focus on sentences that truly express the relation.

### Event extraction — joint trigger and argument

For each token $t$, predict trigger label (including None). For each (trigger, entity) pair in the sentence, predict argument role (including None).

Joint loss:
$$
\mathcal{L} = \mathcal{L}_{\text{trigger}} + \lambda \mathcal{L}_{\text{argument}}
$$

Constraint: argument prediction conditions on predicted trigger type — a role like "Victim" is only valid for certain event types.

### OpenIE as sequence generation

Encode sentence; decode structured triples as text:
Input: `Satya Nadella was appointed CEO of Microsoft in 2014.`
Output: `<subj>Satya Nadella</subj><rel>was appointed CEO of</rel><obj>Microsoft</obj>; <subj>Satya Nadella</subj><rel>was appointed CEO of Microsoft in</rel><obj>2014</obj>`.

Train with maximum likelihood over human-written triples.

### Mapping math to intuition

- `[E1]` and `[E2]` markers: "tell the model where the entities are"
- $\mathbf{h}_{[\text{E1}]}$: "Subject embedding after integrating context; knows what the subject 'is' in this sentence"
- Bag attention: "trust the sentence that most resembles what the relation looks like"
- Joint trigger/argument loss: "predicting together so argument predictions are consistent with trigger predictions"

## 7.4 Worked Example

**Relation extraction on TACRED.**

Schema: 41 relation types including `per:employee_of`, `org:founded_by`, `no_relation`.

Sentence: "Steve Jobs co-founded Apple in 1976."
- Subject: `Steve Jobs` (PER)
- Object: `Apple` (ORG)

Desired label: `org:founded_by` (or symmetric `per:founder_of`, depending on convention).

Pipeline:
1. Tokenize with entity markers: `[CLS] [SUBJ-PER] Steve Jobs [/SUBJ-PER] co-founded [OBJ-ORG] Apple [/OBJ-ORG] in 1976 . [SEP]`
2. Forward through BERT; get hidden states.
3. Relation representation: concatenate `[CLS]` and the two subject/object marker tokens' hidden states.
4. Linear + softmax → 42-way prediction (41 relations + `no_relation`).
5. Train on ~100k labeled instances; F1 on dev around 70% for BERT-large, higher for modern RoBERTa and SpanBERT.

**Event extraction on ACE2005.**

Sentence: "The company fired the CEO yesterday over accounting fraud."

- Trigger: `fired` → event type `END-POSITION`.
- Arguments: `The company` → Entity (role: Entity), `the CEO` → Person (role: Person), `yesterday` → Time (role: Time-within), `accounting fraud` (more complex — actually a reason; for argument schema may or may not be captured).

Pipeline:
1. BiLSTM-CRF or BERT-based sequence labeling for triggers. "Fired" → B-END-POSITION.
2. For each trigger, iterate over named entities in sentence; classify each as a role under that event type.
3. Result: Event(type=END-POSITION, person=the CEO, entity=The company, time=yesterday).

**Open IE on the same sentence.**

Stanford OpenIE (rule-based over dependency parse) yields:
- `(The company; fired; the CEO)`
- `(The company; fired the CEO; yesterday)`
- `(The company; fired the CEO over; accounting fraud)`

Neural OpenIE (seq2seq) trained to mimic such outputs. More fluent extractions, sometimes lower precision.

## 7.5 Relevance to ML Practice

**Where it appears.**

- **Knowledge graph construction**: extract triples from the web to populate Wikidata, commercial KGs.
- **Financial/news monitoring**: real-time extraction of M&A, leadership changes, product launches.
- **Biomedical IE**: BioCreative competitions extract gene-disease, drug-drug interactions.
- **Legal tech**: clause extraction, party identification, obligation tracking.
- **Security intelligence**: extracting IOCs (IP, hash, CVE) and attack steps from threat reports.
- **Structured search**: enables queries like "list all companies founded in Germany after 2010".

**When to use which approach.**

- *Schema is fixed, labeled data available*: supervised RE with BERT or RoBERTa.
- *Schema is fixed, labels are scarce*: distant supervision + multi-instance learning; few-shot with LLMs.
- *Schema is open or unclear*: OpenIE.
- *Events with complex structure*: joint neural models or LLM prompting with structured output.
- *Very high precision needed (medical, legal)*: classical models + human review, or LLMs with verification step.

**When NOT to use neural IE.**

- When a well-maintained regex or rule set suffices (extracting dates, phone numbers).
- When the downstream system requires deterministic, auditable decisions.
- When training data is essentially zero and no LLM is available.

**Trade-offs.**

| Approach | Precision | Recall | Scalability | Labels needed |
|---|---|---|---|---|
| Rules / patterns | High | Low | Easy | None |
| Supervised RE (BERT) | High | Moderate | Moderate | Many |
| Distant supervision | Moderate | High | High | KB only |
| OpenIE (rules) | Moderate | High | Very high | None |
| Neural OpenIE | High | High | Moderate | Moderate |
| LLM zero-shot | Moderate-high | High | High | None |
| LLM few-shot | High | High | Moderate | Few |

## 7.6 Common Pitfalls

1. **Cascading errors from NER → RE.** A miss in NER is a miss in RE. Mitigate with joint models or confidence-based post-processing.

2. **Assuming all mentions co-occur in one sentence.** Many real relations span documents. Use cross-sentence or document-level RE.

3. **Distant supervision noise.** A sentence mentioning two entities doesn't necessarily express their KB relation. Use multi-instance learning and bag-level evaluation.

4. **Class imbalance with `no_relation`.** In TACRED ~80% of pairs are `no_relation`. A naive classifier predicts `no_relation` always → inflated accuracy, useless. Use per-class F1, excluding `no_relation` from the metric.

5. **Ignoring coreference.** "Apple was founded by Jobs. He also co-founded NeXT." Without coreference resolution, the second sentence's `He` is untethered.

6. **OpenIE triples are not canonicalized.** `fired`, `dismissed`, `terminated`, `laid off` all mean similar things but appear as distinct predicates. Entity linking and predicate normalization (paraphrase clustering) are needed for downstream use.

7. **Hallucinations with LLM extraction.** LLMs sometimes invent entities or relations not in the source. Use constrained extraction, verification prompts, or evidence-grounded methods.

8. **Evaluation leakage.** Splitting by sentence rather than by document (or entity) leaks information: the same entity pair can appear in both train and test. Split by document or by entity pair.

9. **Nested events.** `"The meeting about Alice's resignation was postponed"` contains a resignation event inside a postponement event. Flat models miss the hierarchy.

10. **Ambiguity in OpenIE.** "She saw the man with the telescope" — is `(she; saw; the man)` or `(she; saw; the man with the telescope)`? OpenIE systems often generate both.

---

## 7.7 Interview Questions: Information Extraction

### Foundational

**Q1. Difference between NER, relation extraction, and event extraction.**
NER identifies entity spans and types (PERSON, ORG, LOC). RE, given two entities, classifies their relation from a fixed schema. EE identifies events (trigger + type) and the participants playing various roles. IE chains these together to build structured knowledge.

**Q2. What is distant supervision?**
A weak-supervision strategy: use a knowledge base to automatically label sentences. Assumption: every sentence mentioning two entities with a KB relation expresses that relation. Label is noisy but plentiful. Multi-instance learning reduces the noise impact.

**Q3. Explain Open IE and contrast with schema-based RE.**
OpenIE extracts (subject, predicate, object) triples where the predicate is a phrase from text, not a fixed relation. Pros: works on any domain, no schema needed, captures long tail. Cons: outputs are noisy, un-normalized, and require linking for downstream use. Schema-based RE is cleaner but limited to known relations.

**Q4. Why are entity markers effective in BERT-based RE?**
Markers (`[E1] ... [/E1]`) tell the model exactly which spans are the subject and object of the relation query. BERT learns to attend specifically to these markers' positions, producing relation-aware representations. Empirically, marker-based approaches beat "average of entity tokens" by 2–5 F1 points.

### Mathematical

**Q5. Write the loss for attention-based multi-instance learning.**
For a bag $\mathcal{B}$ of sentences with label $r$ from the KB, compute sentence representations $\{\mathbf{h}_s\}$. Attention:

$$
\alpha_s = \frac{\exp(\mathbf{h}_s^\top \mathbf{W} \mathbf{r})}{\sum_{s'} \exp(\mathbf{h}_{s'}^\top \mathbf{W} \mathbf{r})}, \quad \mathbf{b} = \sum_s \alpha_s \mathbf{h}_s
$$

Then $p(r \mid \mathcal{B}) = \text{softmax}(\mathbf{W}_o \mathbf{b})$; cross-entropy against $r$. Attention focuses on sentences consistent with the relation, reducing the impact of noisy instances.

**Q6. Derive the joint event-extraction loss.**

$$
\mathcal{L} = \sum_t \mathcal{L}_{\text{trig}}(y^t_t, \hat{y}^t_t) + \lambda \sum_{(t, e)} \mathcal{L}_{\text{arg}}(y^a_{t,e}, \hat{y}^a_{t,e})
$$

where $y^t$ are trigger labels per token, $y^a$ are role labels per (trigger, entity) pair. $\lambda$ balances the two losses. Often further constrained: argument role labels are conditioned on the predicted trigger type (a "Defendant" role is invalid for a "Meeting" event).

**Q7. How does the at-least-one (MIL) objective differ from standard cross-entropy?**
Standard CE: $-\log p(r \mid s)$ per sentence. At-least-one: $-\log \max_{s \in \mathcal{B}} p(r \mid s)$ per bag. The max is not differentiable — typically replaced by softmax approximation or attention. Effect: only the best-matching sentence drives the gradient, avoiding penalty from noisy bag members.

**Q8. Express document-level RE objective.**
For a document with $n$ entities, there are up to $n(n-1)$ entity pairs. For each pair $(e_i, e_j)$, predict a set of relations (multi-label). Loss:

$$
\mathcal{L} = -\sum_{i \ne j} \sum_{r \in R} [y_{ij}^r \log \hat{p}_{ij}^r + (1 - y_{ij}^r) \log (1 - \hat{p}_{ij}^r)]
$$

with challenges: imbalance (most pairs have no relation), long-range dependencies (relation evidence scattered across the document), and coreference.

### Applied

**Q9. Build a pipeline to extract drug-disease associations from 30M PubMed abstracts.**
- *Preprocessing*: sentence split, tokenize with biomedical BPE.
- *NER*: fine-tune a BioBERT-based model to identify DRUG and DISEASE mentions (use BC5CDR dataset).
- *Entity linking*: map mentions to UMLS CUIs for canonicalization.
- *Relation extraction*: per sentence or window, classify pairs of DRUG-DISEASE mentions into TREATS / CAUSES / NONE. Use a ClinicalBERT/BioBERT model fine-tuned on e.g. the ChemProt or BioCreative CDR corpus.
- *Aggregation*: for each canonical (drug, disease) pair, count supporting sentences; produce a confidence.
- *Human review*: high-confidence predictions go to a curated KB; low-confidence flagged for biocurator review.
- *Evaluation*: precision/recall against a curated gold standard (e.g., DrugBank, CTD).

**Q10. You need to extract clauses and obligations from legal contracts. Approach?**
- *Taxonomy definition*: collaborate with legal SMEs to define clause types (termination, indemnification, non-compete, limitation of liability, etc.).
- *Annotation*: a corpus of contracts annotated for clauses. 500–1000 contracts typically suffice with active learning.
- *Model*: LegalBERT or RoBERTa-large fine-tuned for span extraction (like extractive QA) or sequence tagging.
- *Obligation extraction*: within identified clauses, extract subject (who), action (what), condition (when), modality (must/may).
- *Output*: structured JSON with clause type, text span, extracted arguments.
- *LLMs*: increasingly, a single Claude/GPT-4 call with a detailed schema gives competitive results with far less engineering.

**Q11. Event extraction from noisy news streams: design it.**
- *Real-time component*: fast event detection via fine-tuned DistilBERT — low-latency trigger classifier.
- *Batch component*: hourly job that extracts full event structures with a larger model.
- *Deduplication*: events from different news sources often describe the same incident. Cluster by entity overlap + time window + event type.
- *Canonicalization*: link entities to a KG; normalize dates and amounts.
- *Confidence*: ensemble of models; human review for high-impact events.
- *Alerting*: downstream rules route events to analysts.

**Q12. LLM zero-shot IE vs. fine-tuned BERT: when to choose which?**
- *Fine-tuned BERT*: labeled data available, production cost/latency matters, consistent deterministic output needed, very high volume (millions of docs).
- *LLM zero/few-shot*: no or very little data, schema is complex, need flexibility in relation definitions, smaller volume. Cost and latency are 10–100× higher but setup time is 100× shorter.
- *Hybrid*: use LLM to label thousands of examples, then distill to a fine-tuned small model for production.

### Debugging & Failure Modes

**Q13. Your RE model has 85% accuracy overall but F1 on non-`no_relation` classes is 20%. Explain.**
Class imbalance. `no_relation` is ~80% of TACRED. Predicting `no_relation` always gets 80% accuracy but 0% F1 on real relations. Report per-class F1 and micro-F1 excluding `no_relation`. Mitigate: class weighting, focal loss, oversample real-relation instances, or use a two-stage model (binary "has relation?" then multi-class among real relations).

**Q14. A BERT RE model works well on TACRED but fails on your company's internal text. Why?**
Domain shift. TACRED is news-like (Reuters, WSJ); internal text may be emails, tickets, chats. Mismatch in:
- Vocabulary and jargon.
- Entity types and relations (your schema may differ).
- Style (formal news vs. informal chat).
Fixes: annotate in-domain data, continued pretraining on domain text, or use a generalist LLM with in-domain prompts.

**Q15. Your OpenIE system produces many duplicate near-identical triples. Fix.**
OpenIE over-generates by design (extracts multiple levels of specificity). Post-process:
- Canonicalize predicates (lemmatize, synonym clustering).
- Deduplicate via entity linking (map "Apple Inc.", "Apple", "AAPL" to one canonical).
- Keep only maximally specific or maximally precise triples per entity pair.
- For downstream KG ingestion, use confidence thresholds and manual verification.

### Follow-up Probes

**Q16. Why is document-level RE fundamentally harder than sentence-level?**
- *Longer context*: evidence may be dispersed across many sentences.
- *Coreference*: the subject/object may be referred to by pronouns or aliases.
- *Multiple candidate pairs*: $n^2$ pairs for $n$ entities, mostly `no_relation`.
- *Interaction among relations*: one relation may imply another; joint inference helps.
- Datasets: DocRED; models like SSAN, ATLOP handle this with evidence-aware scoring.

**Q17. Explain "mention-level" vs. "entity-level" evaluation.**
*Mention-level*: each occurrence of an entity is scored separately. Easier but counts repeated errors multiply.
*Entity-level*: group all mentions of the same entity; one prediction per entity. Harder, more faithful to downstream use (a KG stores entity-level facts).
For RE, entity-level evaluation is the norm for KB construction tasks.

**Q18. What is "entity linking" and how does it interact with IE?**
Entity linking maps extracted mentions to canonical IDs in a knowledge base (Wikipedia, Wikidata, UMLS). Enables merging duplicate mentions ("J.K. Rowling" and "Joanne Rowling"), grounding in prior knowledge, and answering queries across documents. Without linking, extracted triples are orphaned strings.

**Q19. Zero-shot relation extraction — how?**
Frame RE as NLI or QA:
- *NLI*: Premise = sentence; Hypothesis = "Subject X employer_of Object Y"; entailment → relation holds.
- *QA*: Context = sentence; Question = "Who does X work for?"; Answer span = object.
- *LLM prompting*: list candidate relations with definitions; ask which applies.
Advantage: can classify into relations never seen at training. Limitation: precision lower than fine-tuned models on fixed schemas.

**Q20. How does coreference resolution help IE?**
Coreference identifies all mentions referring to the same entity. Without it, "Jobs founded Apple. He was its first CEO." would miss the relation between "He" (i.e. Jobs) and "CEO". With coreference-resolved mentions, IE sees "Jobs founded Apple. Jobs was Apple's first CEO." — enabling the extraction. Joint coreference + IE models are state of the art for document-level extraction.

---

# 8. Question Answering

*Extractive QA, generative QA, reading comprehension.*

## 8.1 Motivation & Intuition

Question answering is the archetypal NLP task: given a question and possibly a context, produce an answer. Its different flavors capture different real-world needs:

- **Reading comprehension (extractive QA)**: given a passage and a question, find the answer as a span of the passage. Think SQuAD. Practical uses: reading comprehension in education, document search with passage highlighting.
- **Generative QA**: given a question (possibly with context), generate the answer as free text. Applies when the answer isn't a verbatim span — requires paraphrasing or synthesis.
- **Open-domain QA**: no context given. The system must first *retrieve* relevant documents from a corpus, then extract or generate an answer. This is how Google, Bing, and ChatGPT answer factual questions.
- **Conversational QA / dialogue QA**: multi-turn — subsequent questions reference prior turns ("What about Obama?" after asking about presidents).
- **Closed-book QA**: the model must answer from its parameters alone (no retrieval). Tests what LLMs "know".

QA has been a battleground for NLP progress. BERT (2018) jumped SQuAD F1 from ~75 to ~90; RoBERTa, DeBERTa pushed further. Retrieval-augmented generation (RAG) and LLMs extended QA to open-domain with strong performance.

**Real-world ML impact.** QA powers search features (Google's featured snippets), customer support ("search our help center"), enterprise knowledge bases, document-grounded chatbots, and educational tutors. RAG is now the default architecture for enterprise LLM deployments requiring factual grounding.

## 8.2 Conceptual Foundations

### Task formulations

**Extractive QA.** Given passage $P$ and question $Q$:
- Output: start and end positions in $P$ indicating the answer span.
- Assumes answer is a contiguous span of $P$.
- Datasets: SQuAD (100k Q-A pairs), SQuAD 2.0 (adds "unanswerable"), NaturalQuestions, TriviaQA.

**Generative (abstractive) QA.**
- Output: free text.
- Handles questions requiring synthesis, paraphrasing, or reasoning.
- Datasets: MS MARCO, ELI5, NarrativeQA.

**Open-domain QA.**
- Given only $Q$, retrieve relevant docs from a corpus (Wikipedia, proprietary DB) and answer.
- Architecture: retriever + reader.
- Datasets: OpenNQ, TriviaQA-open.

**Multi-hop QA.**
- Requires chaining evidence across multiple documents or facts.
- E.g., "What is the capital of the country where Sachin Tendulkar was born?" → need Tendulkar-born-in-India → India-capital-Delhi.
- Datasets: HotpotQA, MuSiQue.

### Architectures

**BERT for extractive QA.** Input: `[CLS] Q [SEP] P [SEP]`. Two heads predict start and end positions over passage tokens.

$$
p_{\text{start}}(i) = \frac{\exp(\mathbf{w}_s^\top \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_s^\top \mathbf{h}_j)}, \quad p_{\text{end}}(i) = \frac{\exp(\mathbf{w}_e^\top \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_e^\top \mathbf{h}_j)}
$$

At inference: pick $(i, j)$ maximizing $p_{\text{start}}(i) p_{\text{end}}(j)$ subject to $i \le j$ and $j - i \le L$.

**SQuAD 2.0 with unanswerable questions.** Add a null-answer option: if $p_{\text{start}}([\text{CLS}]) \cdot p_{\text{end}}([\text{CLS}])$ exceeds the best span score by a threshold, predict "no answer".

**Generative QA with seq2seq.** T5 / BART / Flan-T5 take `question: Q context: P` and generate the answer. Scales to abstractive answers.

**Retriever-reader (DPR + Reader, 2020).**
1. *Retriever*: Dense Passage Retriever encodes queries and passages into vectors; retrieve top-$k$ via approximate nearest neighbor.
2. *Reader*: extract or generate answer from top-$k$ passages.

**Retrieval-Augmented Generation (RAG, 2020).** Retrieve passages; feed them plus the question into a generator (BART or LLM). Generator learns to condition on retrieved content.

**LLMs as QA systems.** Large models (GPT-4, Claude) answer questions either:
- *Closed-book*: from parametric memory. Strong for well-covered facts but hallucination-prone for long tail.
- *Open-book with RAG*: retrieve, include in context, generate. Default production architecture.

### Dense retrieval (DPR)

Train a bi-encoder: question encoder $E_Q$ and passage encoder $E_P$. Similarity $s(q, p) = E_Q(q)^\top E_P(p)$. Train with contrastive loss using positive (relevant) and negative (irrelevant) passages:

$$
\mathcal{L} = -\log \frac{\exp(s(q, p^+))}{\exp(s(q, p^+)) + \sum_{p^- \in \mathcal{N}} \exp(s(q, p^-))}
$$

Hard negatives (e.g., BM25 top-ranked but wrong passages) are critical for quality.

### Assumptions and failure modes

**Assumption 1: The answer is in the context (extractive).**
Violated when the passage doesn't contain it. SQuAD 2.0 adds unanswerable questions precisely to test this.

**Assumption 2: A single passage contains the full answer.**
Violated for multi-hop questions. Requires retrieval across multiple hops or chain-of-thought reasoning.

**Assumption 3: Questions are well-formed and context-independent.**
Violated in conversational QA where questions reference prior turns.

**Assumption 4: The retriever returns relevant passages.**
Retrieval failures are the bottleneck for most open-domain systems. If the retriever misses, the reader can't recover.

## 8.3 Mathematical Formulation

### Extractive QA loss

Given gold start $s^*$ and end $e^*$ positions:
$$
\mathcal{L} = -\log p_{\text{start}}(s^*) - \log p_{\text{end}}(e^*)
$$
The start and end are trained independently; the model learns to jointly predict them by sharing the encoder.

### Span scoring at inference

Best span $(i, j)$ maximizes $\log p_{\text{start}}(i) + \log p_{\text{end}}(j)$ with constraints $i \le j$, $j - i < L$. Exhaustive: $O(n^2)$ pairs; usually limited to $L_{\max} = 30$ tokens.

### Generative QA loss

Standard seq2seq cross-entropy:
$$
\mathcal{L} = -\sum_t \log p(y_t \mid y_{<t}, Q, P)
$$

### Retriever loss (in-batch contrastive)

For batch of $B$ (question, positive passage) pairs, treat other passages in the batch as negatives. For pair $i$:
$$
\mathcal{L}_i = -\log \frac{\exp(s(q_i, p_i))}{\sum_{j=1}^B \exp(s(q_i, p_j))}
$$

### RAG joint distribution

$$
p(y \mid x) = \sum_{p \in \text{top-}k} p(p \mid x) \cdot p(y \mid x, p)
$$

Retriever gives $p(p \mid x)$; generator gives $p(y \mid x, p)$. In practice, either marginalize over retrieved docs (RAG-Sequence) or concatenate them into one context for the generator (Fusion-in-Decoder, FiD).

### Mapping math to intuition

- Two heads (start, end): "point to the beginning and ending of the answer"
- Softmax over passage tokens: "probability distribution over 'where does the answer start/end'"
- Retriever bi-encoder: "encode everything once; cheap cosine lookups at query time"
- RAG marginalization: "average the answer distribution over the top retrieved docs, weighted by retriever confidence"

## 8.4 Worked Example

**SQuAD extractive QA with BERT.**

Passage: "Albert Einstein was a theoretical physicist born in Ulm in 1879. He developed the theory of relativity."

Question: "Where was Einstein born?"

Expected span: "Ulm".

Pipeline:
1. Tokenize: `[CLS] where was einstein born ? [SEP] albert einstein was a theoretical physicist born in ulm in 1879 . he developed ... [SEP]`
2. Forward through BERT.
3. Start logits: token "ulm" gets highest score (say 5.0); others low.
4. End logits: token "ulm" also highest (3.8).
5. Span scoring: $(i, j)$ pair with $i = j = $ position of "ulm" wins. Answer: "Ulm".

**SQuAD 2.0 with unanswerable question.**

Passage: same as above.
Question: "What is Einstein's spouse's name?" (not in passage).

Expected: no answer.

Pipeline:
- Best span score over tokens: say 2.1.
- `[CLS][CLS]` null score: 3.0.
- Null wins → predict "no answer".

**Open-domain QA with DPR + FiD reader.**

Question: "When was Einstein born?"

1. Encode question with $E_Q$.
2. Nearest neighbor over 21M Wikipedia passage embeddings. Top-10 include Einstein's Wikipedia bio passage.
3. FiD reader: encode each of top-10 passages independently with T5 encoder; concat the encoder outputs; decode answer.
4. Generated: "March 14, 1879".

**Multi-hop with HotpotQA.**

Question: "Which magazine was founded first, Arthur's Magazine or First for Women?"

- Hop 1 retrieve: find "Arthur's Magazine" article → founded 1844.
- Hop 2 retrieve: find "First for Women" article → founded 1989.
- Answer: Arthur's Magazine.

Requires either iterative retrieval (retrieve → read → re-retrieve) or a retriever trained to find both passages jointly.

## 8.5 Relevance to ML Practice

**Where it appears.**

- **Search engines**: Google featured snippets, Bing answers. Usually extractive QA over top search results.
- **Enterprise search**: document QA over internal wikis, ticket systems.
- **Customer support bots**: RAG over FAQ and knowledge base.
- **Education**: tutoring, homework help, test question generation.
- **Medical/legal/financial**: grounded QA over proprietary document collections.

**When to use which architecture.**

- *Fixed passage, answer must be verbatim*: extractive BERT. High precision, easy to audit.
- *Fixed passage, answer may be synthesized*: generative T5 / LLM over the passage.
- *Large corpus, scalable*: retriever-reader. DPR + FiD or RAG.
- *Conversational, multi-turn*: LLM with chat memory + RAG.
- *Closed-book with strong general knowledge*: LLM alone; accept hallucination risk.
- *Tight factual grounding needed*: RAG with citation requirements; verify retrieval quality.

**Trade-offs.**

| Architecture | Precision | Recall | Latency | Cost | Hallucination risk |
|---|---|---|---|---|---|
| Extractive BERT | Very high | Low (span must match) | Low | Low | Zero (extracted) |
| Generative T5 | High | Medium | Low | Low | Low |
| DPR + Reader | High | High | Medium | Medium | Low |
| RAG + LLM | Medium-high | Very high | High | High | Medium |
| Closed-book LLM | Medium | Medium | Low | Medium | High |

## 8.6 Common Pitfalls

1. **Training extractive QA without negative/unanswerable examples.** Model never learns to say "I don't know" — hallucinates spans for unanswerable questions.

2. **Ignoring retrieval quality.** Most open-domain QA failures are retriever failures. Always measure retrieval recall separately.

3. **Passage truncation.** BERT's 512 limit can cut off the answer. Use sliding windows with stride, deduplicate overlapping predictions.

4. **Wrong evaluation metric.** SQuAD uses EM (exact match) and F1 (token overlap). Generative QA should use ROUGE, BLEU, or human eval. Using EM on generative output is unfairly harsh.

5. **Context leakage in training.** Sometimes the question contains words only present in the correct passage, making retrieval artificially easy. Deduplicate or test on held-out questions.

6. **Hallucinated answers from LLMs.** Without retrieval or grounding, LLMs confidently invent facts. Always ground with retrieval for factual QA; verify outputs with secondary checks.

7. **Retriever-reader mismatch.** Retriever trained on different data than reader leads to distribution mismatch. Train jointly or use the same backbone.

8. **Multi-hop "cheating".** Models can answer HotpotQA questions via shortcuts (entity overlap) without true multi-hop reasoning. Check with counterfactual evaluation.

9. **Long-context confusion.** Feeding all top-100 retrieved passages to an LLM causes lost-in-the-middle effects — middle-of-context info is ignored. Carefully rank and truncate.

10. **Stale retrieval index.** RAG systems fail when the knowledge base updates and the index lags. Rebuild index regularly.

---

## 8.7 Interview Questions: Question Answering

### Foundational

**Q1. Extractive vs. generative QA: trade-offs.**
Extractive answers are spans of the context, so they cannot hallucinate within the passage — high factual grounding. Limited to answers literally present. Generative synthesizes answers — handles paraphrase, aggregation, abstraction — but can hallucinate. Extractive is preferred for legal/medical; generative for conversational, aggregative, or multi-source answers.

**Q2. What is retrieval-augmented generation (RAG)?**
A hybrid where a retriever finds relevant documents from a corpus, and a generator produces an answer conditioned on both the question and retrieved context. Solves the parametric-memory limitation of LLMs, enables citation, and supports knowledge updates without retraining.

**Q3. Why is multi-hop QA harder than single-hop?**
The answer requires combining evidence from two or more passages. Challenges: (1) retrieval must find all relevant passages (often from different topics); (2) reasoning must chain facts; (3) models can cheat via shortcuts (entity overlap); (4) error compounds across hops.

**Q4. What is the difference between SQuAD and SQuAD 2.0?**
SQuAD 2.0 adds unanswerable questions — queries for which no span in the passage is the correct answer. Requires models to output "no answer" when appropriate. Harder because models must know what they *don't* know; pure span-prediction models tend to always predict some span.

### Mathematical

**Q5. Derive the loss for BERT extractive QA and explain the two-head design.**
Two independent softmaxes: $p_{\text{start}}(i) = \text{softmax}(\mathbf{w}_s^\top \mathbf{h})$, $p_{\text{end}}(i) = \text{softmax}(\mathbf{w}_e^\top \mathbf{h})$. Loss: $\mathcal{L} = -\log p_{\text{start}}(s^*) - \log p_{\text{end}}(e^*)$. Independent heads are simpler than joint span modeling and work well because the encoder shares context. Joint span models (predicting $(i,j)$ directly) are more expressive but require $O(n^2)$ output space — rarely worth it for SQuAD-scale.

**Q6. How does DPR handle negative passages for training?**
In-batch negatives: for a batch of $B$ (question, positive-passage) pairs, the positive passage of pair $i$ is contrasted against all other pairs' positive passages as in-batch negatives (essentially free compute). Hard negatives are added: passages retrieved by BM25 for question $i$ but not actually relevant. Hard negatives teach the model to distinguish lexically similar but irrelevant passages from truly relevant ones.

**Q7. Derive the RAG objective.**

$$
p(y \mid x) = \sum_{z \in \text{top-}k} p_\eta(z \mid x) \cdot p_\theta(y \mid x, z)
$$

where $z$ is a retrieved document, $p_\eta$ the retriever, $p_\theta$ the generator. Log-likelihood:

$$
\log p(y \mid x) = \log \sum_z p_\eta(z \mid x) p_\theta(y \mid x, z)
$$

Marginalization is exact over top-$k$; gradient flows into both retriever and generator. Simpler variants (FiD) concatenate retrieved docs in the encoder and skip marginalization.

**Q8. For SQuAD 2.0, how do you decide "no answer"?**
Compare the best span score $\max_{i,j} s(i, j)$ to the null score $s(\text{null}) = \text{logit}_{\text{start}}([\text{CLS}]) + \text{logit}_{\text{end}}([\text{CLS}])$. Predict null if $s(\text{null}) > s(\text{best span}) + \tau$ where $\tau$ is a tuned threshold on dev. Intuition: `[CLS]` represents a "no answer" option; if it outscores all spans, the question is unanswerable.

### Applied

**Q9. Build a production QA system over 10M internal documents.**
- *Indexing*: chunk documents into 300–500 token passages with ~50 token overlap. Encode with a dense retriever (E5, BGE, or fine-tuned DPR). Build a FAISS/HNSW index.
- *Retrieval*: hybrid BM25 + dense retrieval for robustness. Rerank top-100 with a cross-encoder (ms-marco-MiniLM). Top-5 passes to reader.
- *Reader*: RAG with a small LLM (Llama-3 70B or Claude Haiku) conditioned on top-5. Return answer + cited passages.
- *Latency*: sub-second for retrieval, 1–2s for reading. Cache frequent queries.
- *Evaluation*: annotated gold set of (question, answer, relevant-passages). Metrics: retrieval recall@5, EM, F1, human faithfulness.
- *Monitoring*: track answer confidence, retrieval score distribution, user feedback.

**Q10. Your QA system confidently answers questions whose answer is not in any retrieved doc.**
Classic RAG hallucination. Mitigations:
- Prompt engineering: instruct the model to say "I don't know" if the answer isn't supported.
- Verifier model: second model checks if the answer is supported by retrieved context; reject if not.
- Faithfulness metrics: train or use a classifier (SummaC, factuality checkers).
- Constrained decoding over retrieved text (extractive fallback).
- Confidence calibration: abstain below a threshold.

**Q11. How would you handle multi-turn conversational QA?**
- *Query rewriting*: rewrite the follow-up question into a self-contained query using dialogue history ("What about his wife?" → "Who is Albert Einstein's wife?"). Standard technique in CoQA, QuAC.
- *Retrieval*: use rewritten query.
- *Context*: pass rewritten query + current dialogue history to the reader.
- *Coreference*: handled implicitly by query rewriting.
- *State*: maintain a compact dialogue state (key facts, last topic) to avoid context explosion.

**Q12. When is closed-book LLM QA acceptable?**
- Well-covered common knowledge (capitals, historical facts).
- Creative or opinion-based answers where grounding isn't required.
- Prototyping when retrieval infrastructure doesn't exist yet.
- Applications where slight hallucination is tolerable.
Never for: medical, legal, financial advice; anything requiring auditable sources.

### Debugging & Failure Modes

**Q13. BERT QA model gives correct-looking answers but F1 is low. Investigate.**
Check:
- *Exact match vs. F1 mismatch*: predictions may be right but tokenization off. E.g., "March 14, 1879" vs. "March 14th, 1879".
- *Boundary errors*: off-by-one-token; common for subword tokenization. Use character-level evaluation.
- *Multiple valid spans*: passage repeats info; gold is one occurrence but model predicts another. Use F1 with union of gold spans.
- *Answer normalization*: "USA" vs "United States" — normalize before scoring.

**Q14. Retrieval recall@5 is 90% but end-to-end QA F1 is 40%. Why?**
Reader is the bottleneck. Possibilities:
- Retrieved passages contain the answer but reader misses it — reader too small or undertrained.
- Passages contain distractors that confuse the reader (common when retrieving multiple candidates, only one of which has the answer).
- Training/inference mismatch: reader trained on single gold passage, seeing 5 at inference. Retrain with multiple passages.

**Q15. Your retriever works on SQuAD but fails on your domain. Diagnose.**
Likely domain shift. DPR trained on Wikipedia+NQ struggles on legal, medical, or code corpora. Fixes:
- Fine-tune retriever on domain questions with BM25-retrieved positives as weak supervision.
- Use a strong general retriever (E5-large, BGE-large) which pretrains on diverse data.
- Domain-adaptive retraining with contrastive learning on your corpus.
- Hybrid BM25 + dense; BM25 works out-of-the-box for specialized terminology.

### Follow-up Probes

**Q16. "Lost in the middle" — what is it and how to mitigate?**
LLMs with long contexts tend to attend strongly to the beginning and end of context, missing middle content. Liu et al. 2023 showed 10–20% accuracy drops for info in the middle of 10k-token contexts. Mitigations: careful reranking (put most relevant first), prompt-level repetition of key facts, or models specifically trained for uniform long-context attention (e.g., using random permutations during training).

**Q17. What is FiD (Fusion-in-Decoder)?**
FiD (Izacard & Grave 2020): for open-domain QA, encode each retrieved passage independently with the T5 encoder, concatenate all encoder outputs, then let the decoder cross-attend to the concatenated sequence. Scales to 100+ passages because encoder cost is linear (not quadratic on total length). Dominant architecture for NaturalQuestions SOTA models.

**Q18. Reader without retriever vs. retriever without reader.**
*Reader without retriever (closed-book)*: the model is the knowledge base. Works for small, common knowledge; poor for long tail; hallucinates.
*Retriever without reader (extractive span from top passage)*: ranks passages, extracts the top span. No synthesis, fragile if retriever miss-ranks. Okay for factoid questions with clear spans.
The combination (retriever + reader) dominates in practice.

**Q19. Zero-shot QA on a new language.**
- Multilingual retriever (mE5, LaBSE) → retrieves across languages.
- Multilingual reader (mT5, XLM-R, multilingual BERT) → reads and extracts.
- Cross-lingual transfer: train on English, test on target language. Works surprisingly well for high-resource targets, worse for low-resource.
- Translation-based: translate question to English, retrieve/read in English, translate answer back.

**Q20. How do you evaluate RAG quality?**
- *Retrieval*: recall@k on (query, relevant-passages) gold data.
- *Reader*: EM / F1 on extractive or ROUGE / BERTScore for generative.
- *End-to-end*: joint EM/F1 computed from the full pipeline.
- *Faithfulness*: is the answer supported by retrieved content? Use factuality classifiers or human eval.
- *Attribution*: does the system correctly cite its sources?
- *Hallucination rate*: answers without supporting evidence.
- *Latency* and *cost*: production KPIs.
Composite scoring: many benchmarks (BEIR, RGB, RAGAS) combine these.

---

# 9. Summarization

*Extractive (TextRank), abstractive (seq2seq).*

## 9.1 Motivation & Intuition

Summarization compresses a long document into a short version preserving the essential information. Two fundamentally different approaches:

- **Extractive**: select and stitch together sentences from the source. Like highlighting. Guaranteed factual — every word came from the source. But can be choppy, miss synthesis, or repeat.
- **Abstractive**: generate new sentences that paraphrase, aggregate, or synthesize the source. More fluent, handles compression better. Can hallucinate — say things not in the source.

Consider summarizing a news article:

*Original (200 words)*: "The city council announced on Thursday that it has approved a new initiative to expand public transit in the downtown area. The plan, which passed by a 7-2 vote, allocates $50 million over five years for new bus routes, bike lanes, and pedestrian infrastructure..."

*Extractive summary (3 sentences)*: "The city council announced on Thursday that it has approved a new initiative to expand public transit in the downtown area. The plan, which passed by a 7-2 vote, allocates $50 million over five years. Council members cited environmental and economic benefits."

*Abstractive summary*: "The city council voted 7-2 to approve a $50 million, 5-year plan to expand downtown public transit, including new bus routes and bike lanes."

The abstractive version is shorter, fluent, and non-verbatim. The extractive is guaranteed accurate but awkwardly assembled.

**Real-world ML impact.** News aggregation (Yahoo, Apple News), search result snippets, meeting summarization (Zoom, Microsoft Copilot), document briefings (legal, medical, financial), email summarization (Gmail), research paper summaries, and chat history compression for LLMs all use summarization.

## 9.2 Conceptual Foundations

### Extractive summarization

Pick the "best" sentences from the source:

**Rule-based and statistical**:
- *Lead-N*: take the first N sentences. Surprisingly strong baseline for news (journalists front-load content).
- *TF-IDF centroid*: score sentences by similarity to document centroid.

**TextRank (Mihalcea & Tarau, 2004)**: graph-based unsupervised method. Nodes = sentences; edges = similarity. Run PageRank; top-scoring sentences form the summary. Simple, effective, no training needed.

**Supervised extractive**: train a classifier (sequence labeler or sentence binary classifier) to decide whether each sentence belongs in the summary. Requires sentence-level labels (often created by aligning gold summaries to source sentences).

**BERT for extraction (BERTSum)**: encode each sentence with a special token; classify each to include/exclude. SOTA extractive for years.

### Abstractive summarization

Seq2seq generation from source to summary:

**Pointer-generator networks (See et al., 2017)**: hybrid that can either generate from vocabulary or copy tokens from source. Addresses OOV and factuality issues of pure generation.

**Transformer seq2seq (BART, T5, Pegasus)**: pretrained encoder-decoders fine-tuned on summarization. BART uses denoising pretraining (mask spans, reconstruct); Pegasus uses Gap Sentence Generation (mask whole important sentences, generate them). These are the modern workhorses.

**LLMs for summarization**: GPT-4, Claude, Gemini with instructions. Strong zero-shot and with few-shot examples.

### Summarization objectives

- **Informativeness**: summary retains the key content of the source.
- **Faithfulness** (factual consistency): summary does not contradict or invent facts.
- **Coherence and fluency**: summary reads as well-formed prose.
- **Conciseness**: summary is substantially shorter than source.

These can conflict: more abstractive = more fluent but higher hallucination risk; more extractive = safer but less fluent.

### Evaluation

**ROUGE**: word n-gram overlap with reference summary. ROUGE-1 (unigram), ROUGE-2 (bigram), ROUGE-L (longest common subsequence). The de facto standard but has limits — rewards lexical overlap, doesn't measure faithfulness.

**BERTScore**: semantic similarity via BERT embeddings. Better correlation with human judgment.

**Faithfulness metrics**: FactCC, DAE, SummaC — check whether summary claims are entailed by source.

**Human evaluation**: likert scales on informativeness, faithfulness, fluency. Expensive but gold standard.

### Assumptions and failure modes

**Assumption 1: A "gold" summary exists.**
Often there are many valid summaries. ROUGE penalizes legitimate paraphrases and alternative content selection. Multi-reference evaluation helps.

**Assumption 2: The source document is self-contained.**
Violated for news articles referencing prior context, or scientific papers assuming background knowledge.

**Assumption 3: The lead-N baseline is weak.**
For news, lead-3 is hard to beat! News-trained models often implicitly learn position-based selection.

**Assumption 4: Abstractive models are faithful.**
Violated. Abstractive models routinely invent facts (hallucinations), especially for long-tail content. 20–30% of outputs from strong models contain at least one factual error.

## 9.3 Mathematical Formulation

### TextRank

Build a graph $G = (V, E)$ where $V$ is sentences and edges weighted by similarity:
$$
w(s_i, s_j) = \frac{|\{w : w \in s_i \cap s_j\}|}{\log|s_i| + \log|s_j|}
$$
(original formulation; many variants use cosine similarity on TF-IDF or embeddings).

PageRank-style iterative score:
$$
\text{score}(s_i) = (1 - d) + d \sum_{s_j \in N(s_i)} \frac{w(s_i, s_j)}{\sum_{s_k \in N(s_j)} w(s_j, s_k)} \cdot \text{score}(s_j)
$$
where $d \approx 0.85$ is the damping factor. Iterate until convergence.

Select top-$k$ sentences by score; order them as they appear in the original document.

### Supervised extractive with BERT (BERTSum)

Prepend each sentence with `[CLS]`: `[CLS] s_1 [SEP] [CLS] s_2 [SEP] ...`. Encode the full document. For each `[CLS]` token, classify: include (1) or exclude (0).

$$
p_i = \sigma(\mathbf{w}^\top \mathbf{h}_{[\text{CLS}_i]})
$$

Loss: binary cross-entropy per sentence.

### Pointer-generator

At decoder step $t$, generate from vocabulary with probability $p_{\text{gen}}$ or copy from source with probability $1 - p_{\text{gen}}$:
$$
p(w) = p_{\text{gen}} \cdot p_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: x_i = w} \alpha_{t,i}
$$
where $\alpha_{t,i}$ is the attention weight over source positions. The model learns when to copy (rare entities, numbers) vs. generate.

$p_{\text{gen}}$ is itself computed from decoder state, context vector, and current input:
$$
p_{\text{gen}} = \sigma(\mathbf{w}_c^\top \mathbf{c}_t + \mathbf{w}_s^\top \mathbf{s}_t + \mathbf{w}_x^\top \mathbf{x}_t + b)
$$

### Pegasus Gap Sentence Generation

Pretraining: for each document, mask entire sentences deemed "important" (by ROUGE-F1 against the rest of the document). Train the model to generate these masked sentences. This pretraining task closely mirrors summarization.

### ROUGE

ROUGE-N: n-gram recall of candidate summary against reference.
$$
\text{ROUGE-N} = \frac{\sum_{S \in \{\text{Ref}\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \{\text{Ref}\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}
$$

ROUGE-L: LCS-based, normalizes by reference length.

### Mapping math to intuition

- TextRank similarity edge weights: "sentences that share content support each other's importance"
- PageRank iteration: "a sentence is important if it's similar to many other important sentences"
- Copy mechanism: "for rare words or numbers, don't try to generate — just point back to where it appeared in the source"
- Pegasus GSG: "pretraining on the actual task shape gives transfer"

## 9.4 Worked Example

**TextRank on a 5-sentence document.**

Document:
1. The city approved a new transit plan.
2. The council voted 7-2 on Thursday.
3. The plan includes new bus routes and bike lanes.
4. $50 million will be spent over five years.
5. Critics argued the cost is too high.

Step 1. Compute pairwise similarity (e.g., word overlap / cosine on TF-IDF):

|   | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 1 | - | 0.2 | 0.3 | 0.15 | 0.1 |
| 2 | 0.2 | - | 0.1 | 0.2 | 0.15 |
| 3 | 0.3 | 0.1 | - | 0.25 | 0.15 |
| 4 | 0.15 | 0.2 | 0.25 | - | 0.3 |
| 5 | 0.1 | 0.15 | 0.15 | 0.3 | - |

Step 2. Initialize $\text{score}_i = 1$. Iterate PageRank:
Round 1: $\text{score}_1 \approx 0.15 + 0.85 \cdot (\text{contributions from 2,3,4,5})$. Compute through convergence. Sentences 1, 3, 4 typically score highest due to their centrality.

Step 3. Pick top-3 sentences and reorder by position: 1, 3, 4.

Summary: "The city approved a new transit plan. The plan includes new bus routes and bike lanes. $50 million will be spent over five years."

**Abstractive with BART.**

Input: full article, tokenized. Output (generated): "City council approves $50M, 5-year plan for new bus routes and bike lanes, passing 7-2."

Pipeline:
1. Load `facebook/bart-large-cnn` (pretrained on CNN/DailyMail).
2. Tokenize input with max_length=1024, truncate.
3. `model.generate(..., num_beams=4, length_penalty=2.0, max_length=142, min_length=56)` — beam search with length penalty tuned for CNN/DM.
4. Decode output tokens.

Expected ROUGE-1 on CNN/DM: ~44. Pegasus and larger models exceed 47.

**Comparing extractive and abstractive outputs.**

| Method | Summary | Faithfulness | Fluency |
|---|---|---|---|
| Lead-3 | "The city approved a new transit plan. The council voted 7-2 on Thursday. The plan includes new bus routes and bike lanes." | Perfect (verbatim) | Good (news lead is fluent) |
| TextRank | As above, same sentences by coincidence | Perfect | Good |
| BART fine-tuned | Abstractive as shown above | Usually good; occasional errors | Very good |
| Claude/GPT-4 zero-shot | Highly fluent, may add interpretation | Risk of hallucination in long tail | Excellent |

## 9.5 Relevance to ML Practice

**Where it appears.**

- **News summarization**: headlines, snippets, mobile-friendly briefings.
- **Meeting notes**: automated summaries from Zoom/Meet transcripts.
- **Email**: inbox summaries, long-thread digests.
- **Scientific literature**: abstract generation, review summaries.
- **Legal**: contract summaries, deposition highlights.
- **Medical**: patient record summaries for clinicians.
- **Chat compression**: LLM systems summarize dialogue history to stay within context windows.

**When to use what.**

- *High stakes, must be factual*: extractive (or abstractive with verification). Medical, legal, financial.
- *General-purpose summaries of news, meetings*: fine-tuned BART/Pegasus/T5 or LLM.
- *No labeled data, new domain*: LLM zero-shot with careful prompting. Measure faithfulness.
- *Budget constraints, large scale*: distilled or smaller abstractive model; extractive baselines are cheap.
- *Multi-document summarization* (news events, papers on a topic): LLMs excel; classical multi-doc ML-Sum type models exist.

**Trade-offs.**

| Method | Faithfulness | Fluency | Compression | Cost |
|---|---|---|---|---|
| Lead-N | Perfect | Source-dependent | Fixed length | Zero |
| TextRank | Perfect | Moderate | Flexible | Low |
| Supervised extractive | Perfect | Moderate | Tunable | Medium |
| Pointer-generator | Good | Good | Good | Medium |
| BART/Pegasus | Good | Very good | Very good | Medium |
| LLM zero-shot | Variable | Excellent | Excellent | High |

## 9.6 Common Pitfalls

1. **Evaluating only with ROUGE.** ROUGE rewards lexical overlap, not faithfulness. A summary can have high ROUGE and contain hallucinated facts. Always complement with faithfulness metrics or human eval for high-stakes applications.

2. **Not catching position bias.** News-trained models heavily weight the lead. Test on non-news (reviews, papers) to see if the model truly summarizes or just copies the beginning.

3. **Hallucinated numbers and entities.** Abstractive models often corrupt numbers (off-by-one, wrong units, swapped dates). Use pointer mechanisms, or post-hoc verification.

4. **Exposure bias in seq2seq.** During teacher-forced training, models see ground truth; at inference, they see their own outputs. Errors cascade. Mitigate with scheduled sampling or minimum-risk training.

5. **Length control failures.** Asking for "3 sentences" can yield 2 or 7. Solutions: length-control tokens, constrained generation, or post-hoc truncation with care.

6. **Bad tokenization for long docs.** Truncating at 1024 tokens (BART's limit) may lose the content being summarized. Use long-context models (LED, Longformer-Encoder-Decoder) or hierarchical approaches.

7. **Over-fitting to reference style.** Summaries trained on CNN/DM inherit the "three bullet" style; applying this model to scientific papers produces inappropriate summaries.

8. **Trusting zero-shot LLMs for factual content.** LLMs occasionally invent plausible facts. Always verify for production.

9. **Abstractive for non-English or low-resource domains.** Pretrained models are English-heavy. Multilingual models (mBART, mT5) help but are weaker than English counterparts.

10. **Ignoring summary diversity for multiple generations.** Sampling multiple summaries with the same decoder produces very similar outputs. Use diverse beam search or different temperatures for user-facing variety.

---

## 9.7 Interview Questions: Summarization

### Foundational

**Q1. Extractive vs. abstractive — when and why?**
Extractive (select source sentences): guaranteed no hallucination, simpler, best for high-stakes factual summaries. Abstractive (generate new text): more fluent, better compression, handles paraphrase and synthesis. Most modern systems are abstractive but with faithfulness safeguards.

**Q2. Why is the lead-3 baseline so strong on news?**
Journalists write in the inverted pyramid style: most important info first. Lead-3 (first 3 sentences) often captures the essential 5 W's (who/what/where/when/why). News-trained models implicitly learn this position bias. This is why non-news domains (papers, reviews) are much harder summarization targets.

**Q3. What is ROUGE and what are its limits?**
ROUGE measures n-gram overlap with reference summaries. Variants: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence). Limits: rewards lexical overlap over semantic accuracy; penalizes valid paraphrases; doesn't measure faithfulness or coherence. Use with BERTScore and faithfulness metrics.

**Q4. What is TextRank?**
An unsupervised graph-based extractive summarizer. Build a graph of sentences with edges weighted by similarity (word overlap, cosine). Run PageRank: sentences similar to many others score high. Top-$k$ sentences form the summary. Inspired by Google's PageRank; no training required.

### Mathematical

**Q5. Derive the TextRank iteration.**
Let $\text{score}_i$ be the importance of sentence $i$. The update:

$$
\text{score}_i = (1 - d) + d \sum_{j \in N(i)} \frac{w_{ji}}{\sum_{k \in N(j)} w_{jk}} \text{score}_j
$$

- $(1 - d)$ is the teleport probability (damping).
- Sum: weighted contributions from neighbors, normalized so each sentence "distributes" its score proportionally to edge weights.
- Iterate until $|\Delta \text{score}| < \epsilon$. Fixed point exists (Perron-Frobenius on an irreducible stochastic matrix).

**Q6. Pointer-generator equations.**
$p_{\text{gen}} = \sigma(\mathbf{w}_h^\top \mathbf{h}_t^* + \mathbf{w}_s^\top \mathbf{s}_t + \mathbf{w}_x^\top \mathbf{x}_t + b_{\text{ptr}})$

$p(w) = p_{\text{gen}} p_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i : x_i = w} \alpha_{t, i}$

where $\mathbf{h}_t^*$ is the context vector, $\mathbf{s}_t$ is the decoder state, $\mathbf{x}_t$ the decoder input, and $\alpha_{t, i}$ is attention over source. The second term lets the model "copy" OOV tokens by routing probability through attention.

**Q7. Pegasus's gap-sentence objective.**
Given a document $D$, select "important" sentences using:

$$
\text{ROUGE-F1}(s_i, D \setminus s_i)
$$

Mask the top-$k$ sentences with `[MASK_SENT]`. Train encoder-decoder to generate the masked sentences. Rationale: mimics summarization at pretraining; pretraining data becomes pseudo-supervised. Results: Pegasus achieved SOTA ROUGE on many summarization benchmarks.

**Q8. Why does teacher forcing cause exposure bias in summarization, and what's the fix?**
During training with teacher forcing, the decoder always conditions on gold previous tokens. At inference, it conditions on its own (possibly wrong) predictions. Errors accumulate. Fix options:
- *Scheduled sampling*: during training, probabilistically feed the model's own prediction instead of gold.
- *Minimum-risk training*: directly optimize ROUGE (non-differentiable; use REINFORCE or contrastive methods).
- *DPO on human preferences* (modern approach).

### Applied

**Q9. Design a meeting-summarization system.**
- *ASR*: speech to text with speaker diarization (who said what).
- *Preprocessing*: filter filler, restore punctuation, chunk into topics.
- *Summarization*: hierarchical — per-speaker turn summaries → overall summary. Use a long-context model (Longformer-Encoder-Decoder or Claude).
- *Action item extraction*: structured output — list of (action, owner, due-date). Separate head or constrained decoding.
- *Evaluation*: human ratings on informativeness, faithfulness, action-item F1.
- *UX*: highlight source transcript segments; let users click to see the original.

**Q10. Your summarization model hallucinates numbers. Fix.**
- *Pointer-generator*: encourages copying rare tokens.
- *Constrained decoding*: restrict numeric tokens to those in source.
- *Post-hoc verification*: for each number in summary, check if it appears in source; replace or flag if not.
- *Training*: augment with counterfactual examples (swap numbers) and train with factuality loss.
- *Use larger models*: GPT-4 / Claude hallucinate numbers less than smaller models.

**Q11. Summarize 500-page legal contracts.**
- *Chunking*: split into sections (often explicit in contracts).
- *Hierarchical summarization*: summarize each section, then summarize the summaries.
- *Clause-type-aware*: classify each section (definition, obligation, termination), use different prompts per type.
- *Citation*: every claim in summary links to source section — crucial for legal review.
- *Human-in-the-loop*: lawyers verify summaries; feedback loops improve future models.

**Q12. How would you evaluate summarization without reference summaries?**
- *Reference-free metrics*: SummaC, QuestEval (QA-based; ask questions whose answers are in the summary, check answers in source).
- *LLM-as-judge*: prompt GPT-4/Claude to score summaries on informativeness, faithfulness, conciseness.
- *Proxy tasks*: entailment between source and summary (NLI models); information coverage (named entity recall).
- *Human preference*: pairwise comparisons.

### Debugging & Failure Modes

**Q13. Your BART model produces the first 3 sentences nearly verbatim. Why?**
It has learned the news position bias (lead-3 is a strong heuristic). Fixes:
- Train on non-news data (scientific, narrative).
- Augment with shuffled-position training where the "important" info is moved to later positions.
- Use regularization that penalizes high extractive overlap with the lead.
- Consider: is this actually a problem? If the domain is news, lead-3 is often optimal.

**Q14. Your abstractive model's outputs are fluent but frequently incorrect. Metrics?**
Add faithfulness metrics:
- *FactCC / DAE*: train entailment models to score fact consistency.
- *QA-based*: generate questions from summary; answer from source; check agreement.
- *Human eval*: ask annotators to mark unfaithful spans.
- *Ablation*: generate summaries with retrieval augmentation vs. closed-book — see which is more faithful.

**Q15. Your model outputs summaries longer than instructed.**
- *Training*: include length tokens (<short>, <long>) and train conditionally.
- *Inference*: use `min_length` / `max_length` with `length_penalty` in beam search.
- *Post-hoc truncation*: risky, may cut mid-sentence.
- *Prompt engineering* (LLMs): specify word count explicitly, follow with "brief" style instructions.

### Follow-up Probes

**Q16. Multi-document summarization — what's different?**
Multi-doc: summarize a *cluster* of related documents (e.g., multiple news articles about the same event). Challenges:
- *Redundancy*: repeated facts across docs; need deduplication.
- *Contradictions*: different sources may disagree; how to surface?
- *Source diversity*: ensure summary covers multiple viewpoints.
- Approaches: Maximum Marginal Relevance for extraction (select diverse sentences); hierarchical encoding; LLMs with all docs in context.

**Q17. What's a coverage mechanism in seq2seq summarization?**
See et al.'s coverage penalty (2017): track cumulative attention $c_t = \sum_{t' < t} \alpha_{t'}$ over source positions. Penalize attending to already-attended positions to reduce repetition:
$\text{loss}_{\text{cov}} = \sum_i \min(\alpha_{t, i}, c_{t, i})$
Added to standard loss. Prevents the decoder from looping on the same source span, a common pre-Transformer summarization failure.

**Q18. Faithfulness vs. informativeness trade-off.**
More extractive = more faithful but less informative (can only say what the source says, verbatim). More abstractive = more informative (synthesis, inference) but higher hallucination risk. The frontier is pushed by pretraining (reduces hallucination while allowing synthesis) and grounding (retrieval, citations) which gives both.

**Q19. How do LLMs change summarization?**
- *Zero-shot* quality now approaches fine-tuned models on news.
- *Controllability*: natural-language prompts specify length, style, focus.
- *Multi-doc* and *long docs* handled by long-context LLMs.
- *But*: higher cost, less deterministic, harder to audit.
- Hybrid systems: use LLM for flexibility + extractive or grounding for faithfulness.

**Q20. What is "extractive-abstractive" or "two-stage" summarization?**
First, extract the most salient sentences/spans from the source (content selection). Then, feed only these to an abstractive model to generate the final summary (content realization). Reduces long-input burden, focuses the abstractive model on key content, and improves faithfulness (the abstractive step has less raw material to hallucinate from). Common in long-document summarization.

---

# 10. Machine Translation

*Attention mechanisms, back-translation, multilingual models.*

## 10.1 Motivation & Intuition

Machine translation (MT) converts text from a source language to a target language. It is the oldest NLP application (Warren Weaver, 1949) and has driven much of modern NLP: the encoder-decoder architecture came from MT (Sutskever 2014); attention was invented for MT (Bahdanau 2015); the Transformer was introduced in "Attention Is All You Need" (Vaswani 2017), a paper about translation.

The core challenge: map a variable-length sequence in one language to a variable-length sequence in another, preserving meaning. Early approaches (rule-based, statistical MT) relied on heavy linguistic engineering and phrase tables. Neural MT (NMT) replaced all of this with learned encoder-decoder models.

**Why MT is so hard**:
1. **Word order differs across languages.** English SVO vs. Japanese SOV; Arabic VSO.
2. **Morphology**. Turkish can pack meanings into single words that English needs a sentence for.
3. **Idioms don't translate literally**. "It's raining cats and dogs" → literal translations are nonsense.
4. **Ambiguity resolved differently**. "Bank" in English (river/financial) may require different words.
5. **Unknown words** (names, terms, slang) proliferate at scale.
6. **Low-resource languages** have little parallel data.

**Real-world ML impact.** Google Translate (140+ languages), DeepL, Meta's NLLB, cross-border e-commerce, humanitarian response (Rapid Response for low-resource languages), content localization for streaming services, and real-time subtitle generation. LLMs have added zero-shot translation quality approaching commercial MT for high-resource pairs.

## 10.2 Conceptual Foundations

### Architectures through time

**Statistical MT (SMT, pre-2014)**: phrase-based; learn phrase translation tables from parallel corpora; combine with a language model; decode with beam search. Moses was the dominant open-source toolkit.

**RNN encoder-decoder (2014)**: encode source with RNN → single fixed vector → decode target. Worked for short sentences; failed catastrophically on long inputs (information bottleneck).

**Attention mechanism (Bahdanau 2015)**: at each decoder step, attend over encoder hidden states. Dissolved the information bottleneck — quality jumped dramatically.

**Transformer (Vaswani 2017)**: pure attention, no recurrence. Faster training (parallelizable), better quality, now universal.

**Back-translation** (Sennrich 2016): augment parallel data by translating monolingual target text to source with a reverse model. Hugely effective for low-resource.

**Multilingual models**: a single model trained on many languages (mBART, mT5, M2M-100, NLLB). Enables transfer from high-resource to low-resource languages; supports zero-shot translation (language pair never seen).

**LLMs** (GPT-4, Claude, Gemini): zero-shot or few-shot translation, competitive with dedicated MT for high-resource pairs; strong at style, context, and formal register adaptation.

### Attention for translation — the key mechanism

The decoder at step $t$ attends to encoder hidden states $\mathbf{h}_1, \dots, \mathbf{h}_n$ (source positions):
$$
\alpha_{t,i} = \frac{\exp(\text{score}(\mathbf{s}_t, \mathbf{h}_i))}{\sum_j \exp(\text{score}(\mathbf{s}_t, \mathbf{h}_j))}, \quad \mathbf{c}_t = \sum_i \alpha_{t,i} \mathbf{h}_i
$$
The context vector $\mathbf{c}_t$ is a soft alignment: "which source word am I translating now?" The decoder produces the next token conditioned on $\mathbf{s}_t$ and $\mathbf{c}_t$.

Different scoring functions:
- *Additive* (Bahdanau): $\text{score} = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_t + \mathbf{W}_h \mathbf{h}_i)$
- *Dot-product* (Luong): $\text{score} = \mathbf{s}_t^\top \mathbf{h}_i$
- *Scaled dot-product* (Transformer): $\text{score} = \mathbf{s}_t^\top \mathbf{h}_i / \sqrt{d}$

### Back-translation

Given a small parallel corpus $\{(x_i, y_i)\}$ and a large monolingual target corpus $\{y_j\}$:
1. Train an initial target-to-source model on the parallel data.
2. Use it to translate all monolingual target data back to source, producing pseudo-parallel data $\{(\hat{x}_j, y_j)\}$.
3. Train the final source-to-target model on the union of real and pseudo-parallel data.

Effect: dramatically more training data, especially useful for low-resource. The pseudo-data is noisy on the source side, but the target side (monolingual) is clean — and the model is trained to generate clean targets.

### Multilingual NMT

**Many-to-many** (M2M-100, NLLB): a single model translates among $N$ languages via learned language tokens. Input format:
`<src_lang> source text </s> <tgt_lang>`

The model learns:
- Language-agnostic representations in middle layers.
- Language-specific input and output behavior via language tokens.
- Zero-shot translation via the shared representation (e.g., translate Swahili → Thai without ever seeing Swahili-Thai parallel data).

**Challenges**:
- *Curse of multilinguality*: as $N$ grows, per-language capacity shrinks, hurting each pair.
- *Language imbalance*: English dominates training data, biasing toward English-like constructions.
- *Zero-shot quality*: substantially worse than supervised; often goes through English as a pivot.

### Evaluation

**BLEU**: n-gram precision against reference, with brevity penalty. De facto but flawed.
$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_n w_n \log p_n\right)
$$

**chrF, chrF++**: character n-gram F-score. More robust for morphologically rich languages.

**COMET, BLEURT**: learned metrics trained on human judgments. Strong correlation with human quality.

**Human evaluation**: direct assessment (score 0–100), MQM (multidimensional), side-by-side preferences.

### Assumptions and failure modes

**Assumption 1: Parallel data exists.**
Violated for most language pairs. Back-translation, multilingual transfer, and unsupervised MT address this.

**Assumption 2: Translations are one-to-one.**
Violated — many valid translations exist per sentence. BLEU with single references unfairly penalizes valid alternatives. Use multi-reference or human/learned metrics.

**Assumption 3: Meaning is preserved.**
Violated by idioms, cultural references, wordplay. Sometimes literal translation is wrong even if grammatical.

**Assumption 4: Sentence-by-sentence translation suffices.**
Violated for document-level phenomena: pronoun resolution, tense consistency, discourse connectors. Document-level NMT is an active research area.

## 10.3 Mathematical Formulation

### Encoder-decoder with attention (Bahdanau-style)

Encoder: BiRNN produces $\mathbf{h}_1, \dots, \mathbf{h}_n$ from source $x_1, \dots, x_n$.

Decoder at step $t$:
- Previous state $\mathbf{s}_{t-1}$, previous output $y_{t-1}$.
- Attention: $\alpha_{t, i} = \frac{\exp(\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_i))}{\sum_j \exp(\text{score}(\mathbf{s}_{t-1}, \mathbf{h}_j))}$
- Context: $\mathbf{c}_t = \sum_i \alpha_{t, i} \mathbf{h}_i$
- State update: $\mathbf{s}_t = \text{RNN}(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$
- Output: $p(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o [\mathbf{s}_t; \mathbf{c}_t] + \mathbf{b}_o)$

Loss: cross-entropy over target tokens.

### Transformer NMT

Encoder stack: $N$ layers of self-attention + FFN. Decoder stack: $N$ layers of masked self-attention + cross-attention (to encoder) + FFN. Cross-attention is exactly the "attention" mechanism above, now between decoder queries and encoder keys/values.

### Label smoothing

In MT it is standard:
$$
y_k^{\text{smooth}} = (1 - \epsilon) \mathbb{1}[k = y^*] + \epsilon / K
$$
with $\epsilon = 0.1$. Prevents overconfidence, improves BLEU ~0.5 on average.

### Back-translation formal

Target-to-source model $p_{\theta_{t \to s}}(x \mid y)$. Given monolingual target $\mathcal{M}_{\text{tgt}}$:
$$
\tilde{\mathcal{D}} = \{(\hat{x}, y) : y \in \mathcal{M}_{\text{tgt}}, \hat{x} \sim p_{\theta_{t \to s}}(\cdot \mid y)\}
$$
Train final source-to-target model on $\mathcal{D} \cup \tilde{\mathcal{D}}$. Variants use sampling (noisier, more diverse) or beam (cleaner, less diverse) for back-translation; sampling tends to help more.

### BLEU

$p_n$ = clipped n-gram precision (each reference n-gram counted at most the number of times it appears).
$$
\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{otherwise} \end{cases}
$$
where $c$ is candidate length, $r$ is reference length.
$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)
$$
typically $N = 4$, $w_n = 1/4$ each.

### Multilingual objective

$$
\mathcal{L} = \sum_{(l_s, l_t) \in \mathcal{L}} \mathbb{E}_{(x, y) \sim \mathcal{D}_{l_s, l_t}}[-\log p(y \mid x, l_s, l_t)]
$$
summed over all language pairs. Temperature sampling balances high- and low-resource pairs during training:
$$
p(l_s, l_t) \propto |\mathcal{D}_{l_s, l_t}|^\alpha
$$
with $\alpha \in [0.3, 0.7]$ to upsample low-resource.

### Mapping math to intuition

- Attention weights: "soft word alignment — which source token am I producing this target token from?"
- Context vector: "look up the relevant source info for the current target step"
- Back-translation: "free extra data by inverting the task with a reverse model"
- Multilingual training: "share a model across many languages; let them transfer"
- Label smoothing: "don't let the model be overconfident — there are often multiple valid translations"

## 10.4 Worked Example

**Translate English "The cat sat on the mat" to French "Le chat s'est assis sur le tapis".**

**Encoder-decoder with attention (conceptual)**:

1. Tokenize source: `[the, cat, sat, on, the, mat]`. BPE might split differently but keep simple here.
2. Encoder gives 6 hidden states $\mathbf{h}_1, \dots, \mathbf{h}_6$.
3. Decoder starts with `<s>` (start token), state $\mathbf{s}_0$.
4. Step 1 (generate "Le"):
   - Attention: compute $\alpha_{1, i}$ over source. "Le" is a determiner; highest attention on `the` (position 1). $\alpha \approx [0.8, 0.05, 0.02, 0.03, 0.08, 0.02]$.
   - Context: mostly $\mathbf{h}_1$.
   - Output distribution: highest probability on "Le".
5. Step 2 (generate "chat"): attention peaks at `cat` (position 2). Output "chat".
6. Step 3 (generate "s'est"): English "sat" + reflexive auxiliary. Attention primarily on `sat` but some on surrounding (French reflexive requires context).
7. Continue until `</s>`.

**Visualizing attention** in a trained NMT model: a matrix of $\alpha_{t, i}$ often looks close to a diagonal for same-order languages (English-French) but has dramatic re-orderings for different orders (English-Japanese).

**Back-translation example**:

Setup: 100k English-Hindi parallel; 10M Hindi monolingual.
1. Train initial Hindi→English model on 100k parallel.
2. Translate all 10M Hindi sentences to English via this model. Pseudo-parallel 10M set.
3. Train final English→Hindi model on 100k real + 10M pseudo = 10.1M total. Final model is much better than the 100k-only baseline.

**Multilingual model example**:

NLLB-200 (Meta, 2022) is trained on 200 languages. To translate Swahili to Thai:
- Input: `<swa_Latn> Nakupenda </s> <tha_Thai>`
- Output: Thai translation.
- The model has seen Swahili-English and English-Thai abundantly; it has seen very little Swahili-Thai directly. Yet it can zero-shot translate via its language-agnostic internal representation.

## 10.5 Relevance to ML Practice

**Where it appears.**

- **Product translation**: Google Translate, DeepL, Bing.
- **Content localization**: Netflix subtitles, Amazon product listings.
- **Cross-lingual search and retrieval**: query in one language, find documents in another.
- **Multilingual chatbots and support**: serve users in their language.
- **Humanitarian and governmental**: crisis translation (NLLB), legal/medical interpreter support.
- **LLM preprocessing**: translate non-English training data to English (or vice versa) to augment monolingual model training.

**When to use what.**

- *High-resource pairs (en-fr, en-de, en-zh)*: fine-tuned Transformer NMT with back-translation. Near-human on news translation.
- *Low-resource pairs*: multilingual transfer (NLLB, mBART) + back-translation.
- *Specialized domains* (legal, medical, code): fine-tune on in-domain parallel data.
- *Zero/one-shot*: LLMs (Claude, GPT-4) with prompting.
- *Very fast production*: distilled/quantized Transformer or even an LSTM seq2seq.
- *Document-level or style-aware*: LLMs or dedicated document-level NMT.

**Trade-offs.**

| Approach | Quality (high-resource) | Quality (low-resource) | Cost | Controllability |
|---|---|---|---|---|
| Phrase-based SMT | Moderate | Low | Low | High (rules) |
| RNN + attention | High | Low-moderate | Moderate | Low |
| Transformer NMT | Very high | Moderate | High | Moderate |
| Multilingual | High | High | Very high | Low |
| LLMs (zero-shot) | Very high | Moderate-high | Very high | Very high (prompts) |

## 10.6 Common Pitfalls

1. **Evaluating only BLEU.** BLEU is lexical; modern paraphrastic translations can score low BLEU but be equally good. Use COMET or chrF alongside.

2. **No back-translation for low-resource.** Not using monolingual target data is leaving huge gains on the table.

3. **Forgetting language tokens for multilingual models.** NLLB and mBART require specific tokens; omitting them gives garbage output.

4. **Over-truncation at inference.** Default max_length may cut off long translations. Set generously based on source length.

5. **Wrong beam size.** B = 1 is greedy; B = 4–12 typical for NMT. Very large (50+) causes length bias and quality drop.

6. **Length bias without normalization.** Beam search without length penalty produces too-short outputs. Use length penalty $\alpha \approx 1.0$ for NMT.

7. **Domain mismatch.** News-trained models fail on tech manuals, medical notes. Fine-tune on in-domain.

8. **Catastrophic forgetting in multilingual fine-tuning.** Fine-tuning NLLB on one language pair destroys others. Use LoRA or joint fine-tuning.

9. **Ignoring direction asymmetry.** English→Chinese and Chinese→English have different difficulty; train and evaluate in both directions.

10. **Hallucination in low-resource or out-of-distribution inputs.** NMT models can produce fluent but wrong output. LLMs too. Always evaluate with human review for high-stakes uses.

11. **Not using noisy channel at decoding.** For hard inputs, $p(x \mid y) \cdot p(y)$ (reverse model + target LM) sometimes beats $p(y \mid x)$ alone.

12. **Neglecting tokenization for target language.** Japanese and Chinese need specific segmenters; training on wrong segmentation corrupts BLEU.

---

## 10.7 Interview Questions: Machine Translation

### Foundational

**Q1. What was the attention mechanism's contribution to NMT?**
Before attention, encoder-decoder RNNs compressed the entire source into a fixed vector, creating an information bottleneck for long sentences. Attention (Bahdanau 2015) lets the decoder "look back" at all source hidden states at every step, removing the bottleneck. Quality on long sentences improved dramatically; this paradigm eventually culminated in the Transformer.

**Q2. Explain back-translation.**
Train a target-to-source model; use it to translate monolingual target text back to source, producing synthetic parallel data. Combine with real parallel data and train the final source-to-target model. Effect: often several BLEU points of improvement, especially for low-resource pairs. Target side is clean monolingual; source side is noisy but still informative.

**Q3. How does a multilingual NMT model translate between a pair it never saw (zero-shot)?**
The model learns a shared, language-agnostic representation in its middle layers. Given a source and language tokens, the encoder maps source text into this shared space; the decoder reads it conditioned on the target language. Because pairs like Swahili-English and English-Thai are both trained, the model can zero-shot Swahili-Thai by passing through this shared space, though with degraded quality compared to supervised pairs.

**Q4. Why is BLEU flawed and what are alternatives?**
BLEU rewards lexical overlap; it penalizes valid paraphrases and ignores meaning. For morphologically rich languages, unigram matching is especially harsh. Alternatives: chrF (character n-gram F-score; better for morphology), COMET/BLEURT (learned metrics trained on human judgments; stronger correlation), and human evaluation for final validation.

### Mathematical

**Q5. Derive Bahdanau attention.**
Given encoder states $\mathbf{h}_1, \dots, \mathbf{h}_n$ and decoder state $\mathbf{s}_{t-1}$:
- Compute energies: $e_{t, i} = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i)$.
- Softmax: $\alpha_{t, i} = \exp(e_{t, i}) / \sum_j \exp(e_{t, j})$.
- Context: $\mathbf{c}_t = \sum_i \alpha_{t, i} \mathbf{h}_i$.
- Update $\mathbf{s}_t$ using $\mathbf{s}_{t-1}$, $y_{t-1}$, $\mathbf{c}_t$.
Intuition: the feed-forward network learns a soft alignment score between target state and each source position.

**Q6. BLEU's brevity penalty — why?**
Without BP, a model could game BLEU by emitting only high-precision n-grams (short outputs). BP penalizes candidate length $c$ shorter than reference length $r$:
$\text{BP} = \min(1, \exp(1 - r/c))$
If $c < r$, BP < 1; if $c \ge r$, BP = 1. This discourages too-short outputs while not rewarding artificial length inflation. Combined with precision (which penalizes too-long with over-generation), BLEU roughly balances precision and recall.

**Q7. Derive the training loss for a multilingual NMT model.**
For a set of language pairs $\mathcal{L}$ with data $\mathcal{D}_{l_s, l_t}$:

$$
\mathcal{L} = -\sum_{(l_s, l_t) \in \mathcal{L}} \mathbb{E}_{(x, y) \sim \mathcal{D}_{l_s, l_t}}\left[\sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x, l_s, l_t)\right]
$$

Sampling pairs proportionally to $|\mathcal{D}_{l_s, l_t}|^\alpha$ with $\alpha < 1$ upsamples low-resource pairs. The language tokens $l_s, l_t$ are inputs that steer the model.

**Q8. Why is label smoothing standard in NMT?**
Cross-entropy forces $p(y_t^* \mid \cdot) \to 1$, driving logits to $\infty$. In reality, multiple target tokens are often valid (synonyms, paraphrases). Label smoothing ($\epsilon \approx 0.1$) distributes probability mass: gold token gets $1 - \epsilon$, all others share $\epsilon/(K-1)$. This:
- Prevents overconfidence.
- Improves BLEU by ~0.5 on average.
- Acts as a regularizer.
- Compatible with beam search — sharper models can have worse beam behavior.

### Applied

**Q9. You have 500k English-Igbo parallel sentences and 50M Igbo monolingual. Build the best MT system.**
- *Baseline*: Transformer base fine-tuned on parallel. BLEU ~X.
- *Back-translation*: train Igbo→English; translate Igbo monolingual to English; augment training set to 50.5M pairs. Expect +3–5 BLEU.
- *Multilingual starting point*: fine-tune NLLB-200 which includes Igbo; leverage cross-lingual transfer from similar African languages. Expect further gains for low-resource.
- *Iterative back-translation*: after training with BT, re-train the reverse model with the improved forward model, then re-generate BT data. Marginal but consistent gains.
- *Evaluation*: human evaluation in addition to BLEU/chrF/COMET — human judgments are essential for low-resource where metrics are noisy.

**Q10. Your Transformer NMT model produces hallucinations on out-of-domain text. Mitigate.**
- *Domain adaptation*: fine-tune on in-domain parallel data (even small amounts help).
- *Source-conditioned constraints*: penalize output tokens without evidence in source (copy mechanism for names, numbers).
- *Ensemble decoding*: combine multiple models; disagreement signals uncertainty.
- *Noisy channel decoding*: combine $p(y \mid x)$ with $p(x \mid y) p(y)$ for robustness.
- *Detection*: flag outputs whose source attention is too uniform or whose target-side LM probability is unusually low.

**Q11. Design a production translation service for e-commerce (millions of product descriptions).**
- *Models*: separate models per language pair (best quality) vs. one multilingual (operational simplicity). Usually: top-10 pairs served with specialized models; long tail with multilingual NLLB.
- *Inference*: GPU batched inference with dynamic batching; quantization (INT8); caching frequent strings.
- *Quality*: human post-editing queue for high-value listings.
- *Glossary enforcement*: brand names, product categories must translate consistently. Use constrained decoding or terminology injection.
- *Monitoring*: BLEU drift, user click-through rates, customer-reported mistranslations.
- *A/B testing*: new models deployed to small traffic slice first.

**Q12. When does a general LLM (Claude, GPT-4) beat a dedicated NMT model?**
- *Zero-shot low-resource pairs*: LLMs often match or beat dedicated MT where parallel data is scarce.
- *Style and register control*: LLMs can follow instructions like "translate formally" or "use Brazilian Portuguese".
- *Document context*: LLMs handle pronouns, tense consistency, discourse markers better at document level.
- *Creative content* (marketing copy, poetry): LLMs preserve nuance better.
- *Code or technical content*: LLMs trained on code preserve symbols and structure.
Dedicated NMT still wins on: speed, cost, and stability at massive scale for high-resource pairs.

### Debugging & Failure Modes

**Q13. Your NMT model omits rare words (names, numbers).**
Several causes:
- *OOV / UNK issue*: rare tokens get mapped to UNK in output; use BPE throughout to eliminate UNK.
- *Model prefers common words*: cross-entropy training doesn't strongly reward producing rare tokens. Add copy mechanism (pointer network), or constrained decoding for entities.
- *Back-translation didn't cover rare names*: augment with synthetic data containing names.
- *Terminology injection*: for known names, force their translation via decoding constraints.

**Q14. BLEU = 30 but human eval says translations are unnatural. What's happening?**
BLEU measures n-gram overlap; a model can achieve decent BLEU by producing grammatically broken outputs that happen to contain correct n-grams. Causes:
- *Beam search artifacts*: incorrect word order within fluent phrases.
- *Exposure bias*: model produces hallucinations during free generation not seen during teacher-forced training.
- *Train/test mismatch*: training domain differs from test domain.
Fixes: evaluate with COMET/BLEURT + human; retrain with scheduled sampling or MRT; add more in-domain data.

**Q15. After fine-tuning NLLB on your specific pair, zero-shot quality on other pairs collapsed.**
Catastrophic forgetting. Fixes:
- *LoRA / adapter-based fine-tuning*: only train small task-specific parameters; base model frozen.
- *Replay buffer*: include small samples of original pairs during fine-tuning.
- *Lower learning rate*: $10^{-5}$ instead of $10^{-4}$.
- *Fewer epochs*: use early stopping on the target pair's dev.
- *Multi-task fine-tuning*: include the original multilingual data alongside the target pair.

### Follow-up Probes

**Q16. What is "unsupervised NMT"?**
Training translation with *no* parallel data, only monolingual in each language. Key components:
- *Cross-lingual word embeddings*: map source and target embeddings into a shared space (via adversarial training or Procrustes).
- *Denoising autoencoder*: train models to reconstruct each language.
- *Back-translation*: iteratively translate and re-train.
Quality was surprising at launch (Lample, Artetxe 2018) but still lags supervised; modern practice prefers multilingual pretraining + minimal parallel data.

**Q17. Why does beam search with length penalty beat greedy in NMT?**
NMT loss is token-level; the best sequence globally may have slightly lower-probability tokens early that enable better tokens later. Greedy misses this. Beam (B=4–12) keeps multiple hypotheses; length penalty prevents too-short outputs. Trade-off: very large B hurts because exact MAP sequences are often short and dull (similar to open-ended generation). Typical sweet spot: B=4–12, length penalty 0.6–1.0.

**Q18. Explain noisy-channel decoding and when to use it.**
Noisy channel: $p(y \mid x) \propto p(x \mid y) p(y)$ by Bayes. Train a reverse model $p(x \mid y)$ and a target LM $p(y)$. At decode, score candidate translations by $\log p(x \mid y) + \lambda \log p(y) + \mu |y|$. Used in domain-mismatched or noisy settings: the reverse model provides a "faithfulness" check and the LM provides fluency. Common in low-resource MT.

**Q19. What is document-level NMT, and why is it harder?**
Sentence-level NMT translates each sentence independently, losing context: pronouns may not resolve correctly, tenses may be inconsistent, discourse connectors may be wrong. Document-level NMT conditions on surrounding sentences. Challenges: longer inputs (context window), discourse data is scarce, evaluation is harder (BLEU doesn't capture discourse phenomena). Current SOTA uses long-context Transformers or LLMs with full document context.

**Q20. How does MT quality relate to LLM pretraining data?**
LLMs trained on multilingual web data acquire translation capability as a side effect of language modeling. Quality scales with: (a) amount of each language's data in pretraining; (b) amount of parallel text (including Wikipedia mixed-language edits, translations on blogs); (c) model size; (d) instruction tuning with translation examples. For high-resource pairs, GPT-4 / Claude approach dedicated SOTA NMT; for low-resource, dedicated models with back-translation still lead, but the gap is narrowing.

---

## Closing Notes

This guide has traversed the modern NLP stack from lexical preprocessing through full-system applications. A few themes recur:

- **Representation is everything.** Static embeddings → contextual embeddings → multilingual embeddings. Each jump unlocked new capabilities.
- **Scale plus pretraining beats task-specific engineering.** The Transformer + massive pretraining replaced hand-crafted features, dependency parses, and task-specific architectures.
- **Decoding matters as much as the model.** The same LM produces wildly different outputs under different decoders.
- **Grounding beats parametric memory.** For factual tasks, retrieval + generation is more reliable than generation alone.
- **Evaluation is harder than modeling.** Metrics like BLEU and ROUGE have carried us far but underestimate modern models; human evaluation remains gold.
- **Low-resource and multilingual generalization** is the frontier — techniques like back-translation, multilingual transfer, and cross-lingual embedding spaces make progress where parallel data is scarce.

A strong practitioner combines:
- **Theoretical grounding** (probability, linear algebra, optimization) to reason about why a method works.
- **Architectural fluency** (encoders, decoders, attention, retrieval) to design systems.
- **Experimental discipline** (controlled ablations, proper splits, multiple seeds, calibrated evaluation).
- **Pragmatism** (simpler models often win in production; evaluate latency and cost alongside quality).

For interviews and real work, the ability to go from a messy real-world problem — noisy text, scarce labels, latency budgets, multilingual users — to a coherent architectural proposal with justified trade-offs is worth more than any one model or metric.

---

**[← Previous Chapter: Scalability & Systems](scalability_systems_guide.md) | [Table of Contents](../README.md) | [Next Chapter: CV (Computer Vision) →](Computer_Vision_Comprehensive_Guide.md)**
