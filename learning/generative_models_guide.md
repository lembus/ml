# Generative Models: A Comprehensive Reference Guide

> A standalone, textbook-quality reference covering Variational Autoencoders, Generative Adversarial Networks, Diffusion Models, Normalizing Flows, and Autoregressive Models — including motivation, conceptual foundations, mathematical derivations, worked examples, ML practice, common pitfalls, and interview preparation.

---

## Table of Contents

0. [Preamble: The Landscape of Generative Modeling](#0-preamble)
1. [Variational Autoencoders (VAEs)](#1-vaes)
2. [Generative Adversarial Networks (GANs)](#2-gans)
3. [Diffusion Models](#3-diffusion)
4. [Normalizing Flows](#4-flows)
5. [Autoregressive Models](#5-autoregressive)
6. [Cross-Cutting Comparisons](#6-comparisons)

---

<a name="0-preamble"></a>
# 0. Preamble: The Landscape of Generative Modeling

## 0.1 The Generative Modeling Problem

Given i.i.d. samples $x^{(1)}, \dots, x^{(N)} \sim p_{\text{data}}(x)$ from some unknown distribution over a high-dimensional space $\mathcal{X}$ (e.g., images, text, audio), we want to:

1. **Learn** a model $p_\theta(x)$ approximating $p_{\text{data}}$.
2. **Sample** from $p_\theta$ to generate new examples.
3. (Optionally) **Evaluate** $p_\theta(x)$ for any $x$ — i.e., compute densities or likelihoods.

This contrasts with discriminative modeling, where we learn $p(y \mid x)$. Generative modeling is harder because $p(x)$ lives in a far richer space than typical labels.

## 0.2 Why Is This Hard?

For a 256×256 RGB image, $\mathcal{X} = \mathbb{R}^{196{,}608}$. The data manifold is a tiny, highly structured subset of this space. We need architectures that exploit structure (locality, hierarchy, compositionality) and tractable objective functions.

## 0.3 Two Philosophies

**Likelihood-based models** parameterize $p_\theta(x)$ explicitly:

- **Autoregressive**: factorize $p(x) = \prod_i p(x_i \mid x_{<i})$. Exact likelihood, slow sampling.
- **Flows**: $x = f_\theta(z)$ where $f$ is invertible. Use change-of-variables for exact likelihood.
- **VAEs**: introduce latent $z$, optimize a *lower bound* on $\log p(x)$.
- **Diffusion**: hierarchy of latents tied to a fixed noising process; optimize a variational bound.

**Implicit models** define $p_\theta(x)$ only through a sampling procedure:

- **GANs**: train a generator adversarially against a discriminator.

## 0.4 Trade-Off Cheat Sheet

| Family         | Likelihood   | Sample Quality    | Sample Speed     | Training Stability | Mode Coverage |
|----------------|--------------|-------------------|------------------|--------------------|----------------|
| Autoregressive | Exact        | High              | Very slow        | Stable             | Strong         |
| Flows          | Exact        | Medium            | Fast             | Stable             | Strong         |
| VAEs           | Lower bound  | Often blurry      | Fast             | Stable             | Strong         |
| Diffusion      | Lower bound  | State-of-the-art  | Slow (improving) | Very stable        | Strong         |
| GANs           | None         | High              | Very fast        | Unstable           | Often poor     |

## 0.5 Recurring Mathematical Tools

- **KL divergence**: $D_{\text{KL}}(p \,\|\, q) = \mathbb{E}_p[\log p/q]$. Asymmetric; non-negative; zero iff $p = q$ a.e.
- **Jensen's inequality**: for concave $f$, $f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]$. The engine behind ELBO.
- **Change of variables**: if $y = f(x)$ is a diffeomorphism, $p_Y(y) = p_X(x) \left| \det \partial f/\partial x \right|^{-1}$.
- **Reparameterization trick**: rewrite $z \sim q_\phi(z)$ as $z = g_\phi(\epsilon)$ with $\epsilon$ from a fixed distribution.
- **Score function**: $\nabla_x \log p(x)$ — the vector field pointing toward higher density.

---

<a name="1-vaes"></a>
# 1. Variational Autoencoders (VAEs)

## 1.1 Motivation & Intuition

### 1.1.1 Why Probabilistic Latent Variables?

You have 60{,}000 images of handwritten digits and want to generate new ones. Each digit image is complex, but it can be "explained" by a small number of underlying *factors* — which digit, slant, stroke thickness, position. The generative recipe:

1. Sample latent factors $z$ from some simple prior (e.g., Gaussian).
2. Decode $z$ into a pixel image $x$.

The challenge: how do we *learn* the decoder $p_\theta(x \mid z)$ from data alone, without ever observing $z$?

### 1.1.2 Why Not Plain Autoencoders?

A plain autoencoder learns encoder $f: x \mapsto z$ and decoder $g: z \mapsto x$ minimizing $\|x - g(f(x))\|^2$. It compresses, but:

- The latent space has **no probabilistic structure**. Random $z$ decodes to garbage.
- Interpolating between two latents often passes through low-density regions.
- There's no principled way to estimate $p(x)$ or generate diverse new samples.

A VAE *forces the latent space to behave like a known distribution* (typically $\mathcal{N}(0, I)$). Then sampling from that distribution and decoding generates plausible new data.

### 1.1.3 The Real-World ML Connection

VAEs are used for: image generation (early baselines, modern variants like VQ-VAE-2 still competitive), representation learning (semi-supervised learning, clustering), anomaly detection, drug/molecular discovery, controllable synthesis, and as the perceptual compressor inside latent diffusion models like Stable Diffusion.

## 1.2 Conceptual Foundations

### 1.2.1 Components

A VAE is a triple of distributions:

- **Prior** $p(z)$: simple, fixed. Almost always $\mathcal{N}(0, I_d)$.
- **Likelihood / decoder** $p_\theta(x \mid z)$: a neural network outputting parameters of a distribution over $x$ given $z$. For continuous $x$, often Gaussian; for binary $x$, Bernoulli.
- **Approximate posterior / encoder** $q_\phi(z \mid x)$: a neural network outputting parameters of a distribution over $z$ given $x$. Usually a diagonal Gaussian $\mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x)))$.

Greek letters: $\theta$ for decoder (generative model parameters), $\phi$ for encoder (variational parameters).

### 1.2.2 The Generative Story

To generate, the model says: nature first samples $z \sim p(z)$, then $x \sim p_\theta(x \mid z)$. The encoder is a learned trick to make training tractable; it is *not* part of the generative model. At generation time, you only need $p(z)$ and the decoder.

### 1.2.3 Why an Encoder At All?

For maximum likelihood we'd want $\log p_\theta(x) = \log \int p_\theta(x \mid z) p(z) \, dz$, which is intractable. Importance sampling needs a proposal that concentrates near the *posterior* $p_\theta(z \mid x)$. The encoder $q_\phi(z \mid x)$ is exactly that learned proposal. Variational inference reframes the problem: optimize a *lower bound* using $q_\phi$.

### 1.2.4 Assumptions

| Assumption | What it says | What breaks if violated |
|---|---|---|
| Prior is $\mathcal{N}(0, I)$ | Latent space is unimodal Gaussian | Mismatch between aggregate posterior and prior → "holes" → poor samples |
| $q_\phi(z \mid x)$ is diagonal Gaussian | Posterior is unimodal, axis-aligned | True posterior may be multimodal; bound becomes loose |
| Decoder likelihood form | E.g., Gaussian for images | Gaussian + MSE → blurry; Bernoulli for non-binary $x$ → wrong likelihoods |
| Conditional independence in $p_\theta(x \mid z)$ | Pixels independent given $z$ | Decoder cannot model fine-scale texture without help |
| Latent dim $d$ sufficient | Enough capacity | Too small: under-fitting; too large: posterior collapse |

### 1.2.5 Posterior Collapse

If the decoder is so powerful that it can ignore $z$ and still model the data well (e.g., autoregressive PixelCNN decoder), the encoder will set $q_\phi(z \mid x) \approx p(z)$ for all $x$. The KL term goes to zero, latent code carries no information, and you've trained a fancy decoder with a useless encoder.

## 1.3 Mathematical Formulation

### 1.3.1 Setup

Data $x \in \mathcal{X}$, latent $z \in \mathbb{R}^d$. Goal: maximize $\sum_n \log p_\theta(x^{(n)})$.

### 1.3.2 Deriving the ELBO

$$
\log p_\theta(x) = \log \int p_\theta(x, z) \, dz = \log \mathbb{E}_{q_\phi}\!\left[\frac{p_\theta(x,z)}{q_\phi(z \mid x)}\right].
$$

Apply Jensen's inequality:

$$
\log p_\theta(x) \ge \mathbb{E}_{q_\phi(z \mid x)}\!\left[\log \frac{p_\theta(x,z)}{q_\phi(z \mid x)}\right] =: \mathcal{L}(\theta, \phi; x).
$$

This is the **Evidence Lower Bound (ELBO)**.

### 1.3.3 Two Useful Decompositions

Using $p_\theta(x, z) = p_\theta(x \mid z) p(z)$:

$$
\boxed{ \mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]}_{\text{reconstruction}} \;-\; \underbrace{D_{\text{KL}}\!\big(q_\phi(z \mid x) \,\|\, p(z)\big)}_{\text{regularizer}} }
$$

Equivalently, using $p_\theta(x, z) = p_\theta(z \mid x) p_\theta(x)$:

$$
\mathcal{L}(\theta, \phi; x) = \log p_\theta(x) - D_{\text{KL}}\!\big(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)\big).
$$

This shows **the ELBO is tight iff $q_\phi(z\mid x) = p_\theta(z\mid x)$**.

### 1.3.4 Closed-Form KL for Gaussians

If $q_\phi(z \mid x) = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ and $p(z) = \mathcal{N}(0, I)$:

$$
D_{\text{KL}}\big(\mathcal{N}(\mu, \mathrm{diag}(\sigma^2)) \,\|\, \mathcal{N}(0, I)\big) = \frac{1}{2}\sum_{j=1}^d \big( \mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2 \big).
$$

**Derivation.** For 1D Gaussians $q = \mathcal{N}(\mu, \sigma^2)$, $p = \mathcal{N}(0, 1)$:

$$
D_{\text{KL}}(q \| p) = \int q(z)[\log q(z) - \log p(z)] dz.
$$

Use $\log q(z) = -\tfrac{1}{2}\log(2\pi\sigma^2) - \tfrac{(z-\mu)^2}{2\sigma^2}$, $\log p(z) = -\tfrac{1}{2}\log(2\pi) - \tfrac{z^2}{2}$. Take expectation under $q$ using $\mathbb{E}_q[(z-\mu)^2] = \sigma^2$, $\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$. Simplify to $\tfrac12(\mu^2 + \sigma^2 - 1 - \log \sigma^2)$. Sum over independent dimensions.

### 1.3.5 The Reparameterization Trick

To compute $\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[f(z)]$, the naive REINFORCE estimator is unbiased but extremely high variance. **Reparameterization** rewrites

$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
$$

Now sampling $z$ is a deterministic, differentiable function of $\phi$ and parameter-free noise $\epsilon$:

$$
\mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)] = \mathbb{E}_{\epsilon}[\log p_\theta(x \mid \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon)],
$$

and we differentiate inside with low variance. The trick generalizes to any *location-scale* distribution. It does **not** work for discrete latents — use Gumbel-Softmax or REINFORCE.

### 1.3.6 The Final Training Loss

Per data point, with one Monte Carlo sample of $\epsilon$:

$$
\mathcal{L}(\theta, \phi; x) \approx \log p_\theta(x \mid z) - \frac{1}{2}\sum_j (\mu_j^2 + \sigma_j^2 - 1 - \log \sigma_j^2),
$$

where $z = \mu + \sigma \odot \epsilon$. Encoders typically output $\log \sigma^2$ (logvar) for stability.

## 1.4 Worked Example

### 1.4.1 MNIST End-to-End

**Setup.** $x \in \{0, 1\}^{784}$ (binarized MNIST). Latent $z \in \mathbb{R}^{20}$. Encoder: MLP $784 \to 400 \to (\mu, \log\sigma^2) \in \mathbb{R}^{40}$. Decoder: MLP $20 \to 400 \to 784$ logits, with $p_\theta(x \mid z) = \prod_i \mathrm{Bern}(\sigma(\ell_i))$.

**Per-sample loss.** Reconstruction is binary cross-entropy summed over pixels. KL is the closed-form Gaussian expression. Total: BCE + KL.

**One forward pass.** Encoder outputs $\mu = (0.2, -0.1, 0, \dots)$, $\log\sigma^2 = (-2, -2, \dots)$, so $\sigma \approx 0.37$. KL contribution from one dimension: $\tfrac12(0.04 + 0.135 - 1 - (-2)) = 0.588$. Sum across 20 dims for total KL.

Sample $\epsilon \sim \mathcal{N}(0, I_{20})$, compute $z = \mu + \sigma \odot \epsilon$, decode, compute BCE. Backprop through the chain.

**After training.** Sample $z \sim \mathcal{N}(0, I_{20})$, decode → plausible (slightly blurry) MNIST digits. The 2D projection of test latents typically clusters by digit class, even without label supervision.

### 1.4.2 Why VAE Samples Are Blurry

For continuous $x$ with Gaussian likelihood $p_\theta(x \mid z) = \mathcal{N}(g_\theta(z), \sigma^2 I)$, the optimal $g_\theta(z)$ predicts the *mean* of $p_\theta(x \mid z)$. If multiple $z$'s explain similar $x$'s, the decoder hedges by averaging — and averaging high-frequency content blurs it.

Fixes: more expressive likelihoods (autoregressive PixelCNN decoder, mixture of logistics); discrete latents (VQ-VAE) with autoregressive priors; adversarial losses on top (VAE-GAN); or use the VAE only as a perceptual compressor for diffusion (Stable Diffusion).

## 1.5 Conditional VAEs and β-VAE

### 1.5.1 Conditional VAE (cVAE)

For controllable generation given conditioning $y$:

$$
\text{prior } p(z \mid y), \quad \text{decoder } p_\theta(x \mid z, y), \quad \text{encoder } q_\phi(z \mid x, y).
$$

ELBO:

$$
\mathcal{L} = \mathbb{E}_{q_\phi}[\log p_\theta(x \mid z, y)] - D_{\text{KL}}(q_\phi(z \mid x, y) \,\|\, p(z \mid y)).
$$

Often $p(z \mid y) = \mathcal{N}(0, I)$ (conditioning enters only through encoder/decoder).

### 1.5.2 β-VAE (Higgins et al., 2017)

$$
\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \beta \cdot D_{\text{KL}}(q_\phi(z \mid x) \,\|\, p(z)).
$$

- $\beta = 1$: standard VAE.
- $\beta > 1$: stronger pressure on $q_\phi$ to match the factorized prior. Encourages "disentangled" latent dimensions. Cost: worse reconstruction.
- $\beta < 1$: looser regularization, sharper reconstructions.

**Caveat.** Locatello et al. (2019) showed that fully unsupervised disentanglement is, strictly speaking, impossible without inductive biases. β-VAE produces *some* disentanglement on certain datasets but is not a guarantee.

## 1.6 Relevance to ML Practice

**Where VAEs live.** Representation learning pretext (encoder as feature extractor); anomaly detection (low ELBO → anomaly); hybrid pipelines (VQ-VAE + PixelCNN; **Stable Diffusion** uses a VAE as perceptual compressor with diffusion in the VAE's latent space); data augmentation; drug/materials design.

**When to use a VAE.** You want a smooth probabilistic latent space; you need lower-bound likelihoods; you prefer training stability over sample fidelity; the latent is the product (not the samples).

**When not to use a VAE.** Photorealistic samples are the goal — use diffusion. You need exact likelihoods — use flows or autoregressive.

**Trade-offs.**
- *Bias–variance.* Diagonal-Gaussian posteriors are biased (cannot capture correlations). Reparameterization controls gradient variance.
- *Interpretability.* Sometimes interpretable (β-VAE); not by default.
- *Robustness.* Stable training, forgiving hyperparameters.
- *Compute.* Cheap. Single forward + backward; sampling is one decoder pass.

## 1.7 Common Pitfalls & Misconceptions

1. **Posterior collapse.** Symptoms: KL ≈ 0, decoder ignores $z$. Fixes: KL annealing (ramp KL weight from 0 to 1); free bits (don't penalize per-dim KL below $\lambda$); weaker decoder; β < 1.
2. **Treating the encoder as part of the generative model.** It isn't. At test time, sample from the prior and decode.
3. **Confusing reconstruction with $\log p(x)$.** ELBO = reconstruction − KL. Reconstruction alone underestimates loss; ELBO underestimates true log-likelihood.
4. **MSE on images expecting sharp samples.** MSE = Gaussian likelihood with fixed variance. Sharp samples need richer likelihoods.
5. **ELBO as a meaningful absolute number.** Depends on normalization, likelihood form, reduction. Compare bits/dim with proper dequantization for continuous data.
6. **Forgetting dequantization.** Discrete pixels under continuous density can yield arbitrarily high "likelihoods." Add uniform noise.
7. **Latent dimension mis-set.** Too small → bad reconstruction; too large → many dimensions collapse. Sweep.
8. **Reparameterization for non-reparameterizable distributions.** Discrete categorical latents need Gumbel-Softmax or REINFORCE.
9. **β-VAE as a guaranteed disentangler.** It isn't. Use as a regularization knob.
10. **Aggregate posterior ≠ prior gap.** Even with low per-data KL, $q(z) = \mathbb{E}_{p_{\text{data}}}[q_\phi(z \mid x)]$ may differ from $p(z)$, leading to "holes." Mitigations: better priors (VampPrior, flow priors); two-stage training.

## 1.8 Interview Preparation: VAEs

### Foundational

**Q1. What problem do VAEs solve that plain autoencoders do not?**

A plain autoencoder learns a deterministic compressive mapping; its latent space has no probabilistic structure: random latents decode to garbage, interpolation can pass through low-density regions, you cannot estimate $p(x)$ or sample principedly. VAEs impose a probabilistic prior on the latent space and train both networks so (i) the aggregate posterior matches the prior, allowing principled sampling, and (ii) decoded samples are plausible.

**Q2. Walk me through the components and training objective.**

Prior $p(z)$ (typically $\mathcal{N}(0, I)$), decoder $p_\theta(x \mid z)$, encoder $q_\phi(z \mid x)$ approximating the posterior. Maximize the ELBO:

$$
\mathcal{L} = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \,\|\, p(z)).
$$

With diagonal-Gaussian $q_\phi$ and $\mathcal{N}(0, I)$ prior the KL has closed form; reconstruction is one Monte Carlo sample using reparameterization.

**Q3. What is the reparameterization trick and why is it needed?**

To compute $\nabla_\phi \mathbb{E}_{q_\phi(z \mid x)}[f(z)]$ we need gradients to flow through sampling. For Gaussian $\mathcal{N}(\mu_\phi, \mathrm{diag}(\sigma_\phi^2))$, write $z = \mu_\phi + \sigma_\phi \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$. Now $\nabla_\phi$ flows directly through $\mu_\phi, \sigma_\phi$, giving low-variance unbiased estimates. The alternative — REINFORCE — is unbiased but very high variance.

### Mathematical

**Q4. Derive the ELBO from $\log p(x)$.**

$$
\log p_\theta(x) = \log \int q_\phi(z \mid x) \frac{p_\theta(x, z)}{q_\phi(z \mid x)} dz.
$$

By Jensen, $\log p_\theta(x) \ge \mathbb{E}_{q_\phi}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \,\|\, p(z))$. Equivalently, $\log p_\theta(x) - \mathcal{L} = D_{\text{KL}}(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x)) \ge 0$, so the ELBO is tight iff $q_\phi$ equals the true posterior.

**Q5. Derive the closed-form KL for $\mathcal{N}(\mu, \sigma^2)$ and $\mathcal{N}(0, 1)$.**

$D_{\text{KL}}(q\|p) = \int q(z)[\log q - \log p] dz$. Substituting Gaussian log-densities and taking expectations using $\mathbb{E}_q[(z - \mu)^2] = \sigma^2$, $\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$, the result simplifies to $\tfrac12(\mu^2 + \sigma^2 - 1 - \log \sigma^2)$.

**Q6. Why is the score-function gradient noisier than reparameterization?**

REINFORCE relies on $\nabla_\phi \log q_\phi(z)$, which can take large values when $z$ is in the tail of $q$, multiplying $f(z)$ by a large noise term. Reparameterization moves randomness outside, so the gradient depends on $\nabla_\phi$ of a smooth $g_\phi(\epsilon)$ — variance is $O(1)$ rather than scaling with $f$.

**Q7. Hoffman & Johnson decomposition: rewrite the KL as MI plus an aggregate-posterior term.**

With $q(x, z) = p_{\text{data}}(x) q_\phi(z \mid x)$ and $q(z) = \mathbb{E}_{p_{\text{data}}}[q_\phi(z \mid x)]$:

$$
\mathbb{E}_{p_{\text{data}}}[D_{\text{KL}}(q_\phi(z \mid x) \,\|\, p(z))] = I_q(x; z) + D_{\text{KL}}(q(z) \,\|\, p(z)).
$$

Minimizing the KL term penalizes both mutual information (driving toward posterior collapse) and the gap between aggregate posterior and prior. β-VAE multiplies both penalties by β, explaining why high β kills information in $z$.

### Applied

**Q8. VAE samples on faces are blurry. Why and what would you try?**

Gaussian likelihoods + uncertain posteriors → optimal decoder predicts a per-pixel mean over plausible reconstructions, averaging away high-frequency detail. Mitigations: (i) richer likelihood (autoregressive PixelCNN decoder, mixture of discretized logistics); (ii) perceptual loss based on pretrained features; (iii) hybrid like VQ-VAE + autoregressive prior, or VAE as compressor for a diffusion model; (iv) adversarial loss term (VAE-GAN).

**Q9. You add a powerful PixelCNN decoder and find $D_{\text{KL}}(q_\phi \| p) \to 0$. Diagnosis and fix?**

**Posterior collapse**. The autoregressive decoder models $p(x)$ on its own, so the model achieves high likelihood without using $z$. Fixes: KL annealing; free bits ($\max(\lambda, \text{KL}_d)$ per dimension); weaker decoder; explicit information constraints.

**Q10. How would you do conditional generation (e.g., generate a "5")?**

Conditional VAE: feed label $y$ to encoder and decoder, train $\mathbb{E}_{q_\phi(z \mid x, y)}[\log p_\theta(x \mid z, y)] - D_{\text{KL}}(q_\phi \,\|\, p(z))$. At inference, sample $z \sim p(z)$, decode with desired $y$.

**Q11. Why might ELBO improve while sample quality degrades?**

Reconstruction-KL trade-off. Very low KL (near posterior collapse) gives high apparent ELBO but useless latents and poor sample diversity. Always evaluate sample quality (FID), reconstruction, and KL separately; the single-number ELBO hides the diagnosis.

### Debugging & failure-mode

**Q12. Training KL goes to nearly zero in the first epochs.**

Posterior collapse, almost certainly. Anneal KL weight, use free bits, weaken decoder. Confirm by sampling encoder twice for the same $x$ — collapse implies near-deterministic $\mu = 0, \sigma = 1$ regardless of $x$.

**Q13. Reconstructions perfect, prior samples terrible.**

**Aggregate posterior / prior mismatch**: encoder maps each $x$ to a tight cluster, but the union of clusters does not cover the prior, so prior samples land in "holes." Fixes: more flexible prior (Mixture of Gaussians, VampPrior, flow prior); two-stage training (fit a separate density model on latents).

**Q14. Loss explodes / NaNs.**

Likely: (i) `logvar` not bounded — clip to e.g., $[-10, 10]$; (ii) Decoder predicts probabilities at exactly 0 or 1 → $\log 0$; use logits + numerically stable BCE; (iii) Input not normalized as expected (e.g., $[0, 255]$ to a Bernoulli decoder).

**Q15. β-VAE samples are smooth but very low fidelity.**

β too high. KL penalty crushed information in latent. Reduce β, or use β-TCVAE / FactorVAE which target only the *between-dimension* correlation.

### Follow-ups & probes

**Q16. Variational gap vs. amortization gap?**

**Variational gap**: $D_{\text{KL}}(q^\star_\phi(z \mid x) \,\|\, p_\theta(z \mid x))$ where $q^\star_\phi$ is the *best* member of the variational family for that specific $x$. **Amortization gap**: additional slack from a single shared network rather than per-data-point optimization. IWAE and semi-amortized inference reduce amortization gap; flow posteriors reduce variational gap.

**Q17. Tell me about IWAE.**

**Importance-Weighted Autoencoder**. Use $K$ samples and a tighter bound:

$$
\mathcal{L}_K = \mathbb{E}_{z_{1:K} \sim q_\phi}\!\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p_\theta(x, z_k)}{q_\phi(z_k \mid x)}\right].
$$

$\mathcal{L}_K$ is monotonically tighter in $K$, converging to $\log p(x)$. Downside: encoder gradients suffer ("SNR collapse") at large $K$ unless using doubly-reparameterized gradients (Tucker et al., 2019).

**Q18. How do VAEs compare to diffusion conceptually?**

A diffusion model is a *hierarchical* VAE with $T$ latents where the encoders $q(x_t \mid x_{t-1})$ are *fixed* (Gaussian noise, not learned) and only the decoders $p_\theta(x_{t-1} \mid x_t)$ are learned. The ELBO decomposes into a sum of denoising terms. The fixed encoder eliminates posterior collapse and the amortization gap; small denoising steps are easy.

---

<a name="2-gans"></a>
# 2. Generative Adversarial Networks (GANs)

## 2.1 Motivation & Intuition

### 2.1.1 The Core Idea: Generation as a Game

Imagine learning to forge paintings. You produce candidate forgeries; an art critic compares them to real paintings and tells you which is which. You improve to fool the critic; the critic improves to catch you. At equilibrium, your forgeries are indistinguishable from real paintings.

A **generator** $G$ produces samples; a **discriminator** $D$ classifies samples as real or fake. They train against each other. If $D$ converges to optimum and $G$ adapts, $G$'s sample distribution approaches the data distribution.

### 2.1.2 Why Not Just Maximize Likelihood?

Likelihood-based models require either tractable density (autoregressive, flows — restricting architectures) or a variational lower bound (VAEs — gaps, blurry samples). GANs sidestep both: they define an *implicit* model with no requirement for tractable density. The cost: no likelihood, hard to evaluate, training notoriously unstable, and pathologies like mode collapse.

### 2.1.3 The Real-World ML Connection

GANs powered an era of generative ML: photorealistic image synthesis (StyleGAN3, BigGAN), image-to-image translation (Pix2Pix, CycleGAN), super-resolution (SRGAN, ESRGAN), domain adaptation, data augmentation, speech synthesis (WaveGAN, MelGAN). Diffusion has overtaken GANs for top-quality unconditional generation, but GANs remain competitive for fast inference (one forward pass) and controllable synthesis (StyleGAN).

## 2.2 Conceptual Foundations

### 2.2.1 Components

- **Generator** $G_\theta: \mathcal{Z} \to \mathcal{X}$: maps noise $z \sim p_z$ (e.g., $\mathcal{N}(0, I)$) to a sample. Defines an implicit distribution $p_g(x)$.
- **Discriminator** $D_\phi: \mathcal{X} \to [0, 1]$: outputs the probability that $x$ is real. (In WGAN: a real-valued *critic*.)

### 2.2.2 The Training Loop

Each iteration: sample real $x \sim p_{\text{data}}$ and fake $G(z)$; update $D$ to increase real/fake classification accuracy; update $G$ to fool $D$. Repeat.

### 2.2.3 Assumptions

| Assumption | What it says | What breaks if violated |
|---|---|---|
| $G$ and $D$ have sufficient capacity | Both representable | Theoretical guarantees evaporate |
| Optimization reaches Nash equilibrium | Game-theoretic result | Almost always violated; oscillation, divergence |
| $D$ near optimum given $G$ | Justifies JSD interpretation | If $D$ trains too slowly, $G$ optimizes against a stale critic |
| Overlapping supports | KL/JSD gradients informative | High-dim images live on disjoint manifolds → vanishing gradients → motivates Wasserstein |

### 2.2.4 The Failure Modes

- **Mode collapse**: $G$ produces only a subset of plausible samples.
- **Vanishing gradient**: $D$ too good → $\log(1 - D(G(z)))$ saturates → no signal to $G$.
- **Non-convergence / oscillation**: min-max dynamics can cycle indefinitely.

## 2.3 Mathematical Formulation

### 2.3.1 Original GAN Objective (Goodfellow et al., 2014)

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].
$$

### 2.3.2 The Optimal Discriminator

For fixed $G$ inducing $p_g$, pointwise the inner objective is $a \log D + b \log(1-D)$ with $a = p_{\text{data}}(x)$, $b = p_g(x)$. Set derivative to zero:

$$
D^\star(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}.
$$

### 2.3.3 Implicit Loss: Jensen-Shannon Divergence

Plug $D^\star$ back into $V$:

$$
V(D^\star, G) = -\log 4 + 2 \cdot \mathrm{JSD}(p_{\text{data}} \,\|\, p_g).
$$

**Derivation.** Substituting $D^\star$:

$$
V(D^\star, G) = \mathbb{E}_{p_{\text{data}}}[\log \tfrac{p_{\text{data}}}{p_{\text{data}} + p_g}] + \mathbb{E}_{p_g}[\log \tfrac{p_g}{p_{\text{data}} + p_g}].
$$

Insert $-\log 2$ into each $\log$: $\log \tfrac{p_{\text{data}}}{p_{\text{data}} + p_g} = \log \tfrac{p_{\text{data}}}{(p_{\text{data}} + p_g)/2} - \log 2$. Define $m = (p_{\text{data}} + p_g)/2$. Then
$V(D^\star, G) = -2\log 2 + D_{\text{KL}}(p_{\text{data}} \| m) + D_{\text{KL}}(p_g \| m) = -\log 4 + 2 \mathrm{JSD}(p_{\text{data}} \| p_g)$. ∎

So minimizing over $G$ minimizes JSD. JSD is symmetric, bounded in $[0, \log 2]$, zero iff $p_g = p_{\text{data}}$.

### 2.3.4 Saturation and the Non-Saturating Loss

Early on, $D(G(z)) \approx 0$, so $\log(1 - D(G(z))) \approx 0$ — gradient vanishes. **Goodfellow's fix**: instead of minimizing $\mathbb{E}[\log(1 - D(G(z)))]$, *maximize* $\mathbb{E}[\log D(G(z))]$. Same fixed points; gradient is large when $G$ is bad.

### 2.3.5 Wasserstein GAN (Arjovsky et al., 2017)

JSD/KL fail when supports are disjoint (common in high dimensions). The **Wasserstein-1** distance,

$$
W(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|],
$$

remains informative even with disjoint supports. By Kantorovich-Rubinstein duality:

$$
W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \le 1} \mathbb{E}_{p_{\text{data}}}[f(x)] - \mathbb{E}_{p_g}[f(x)].
$$

WGAN replaces $D$ with a **critic** $f_\phi$:

$$
\min_G \max_{\|f\|_L \le 1} \mathbb{E}_{p_{\text{data}}}[f(x)] - \mathbb{E}_{p_z}[f(G(z))].
$$

**Enforcing Lipschitz.** Original WGAN: clip critic weights to $[-c, c]$ (crude). **WGAN-GP**: add penalty $\lambda \mathbb{E}_{\hat x}[(\|\nabla_{\hat x} f(\hat x)\|_2 - 1)^2]$ on samples $\hat x$ on lines between real and fake. **Spectral normalization**: divide each layer's weight matrix by its largest singular value.

WGAN benefits: smoother loss landscape (critic loss tracks $W$, useful for monitoring); better stability; no saturation.

## 2.4 Worked Example

### 2.4.1 Toy 1D Mixture

Data: mixture of three Gaussians at $-4, 0, 4$. $G$: small MLP $\mathbb{R}^1 \to \mathbb{R}^1$. $D$: MLP $\mathbb{R}^1 \to (0, 1)$.

**Iteration 0.** $G$ outputs values near zero; $D$ easily distinguishes; updating $G$ pushes outputs toward where $D$ thinks "real."

**Iteration 1{,}000.** Often: $G$ has learned to produce a single mode — **mode collapse**.

**Fix.** Switch to WGAN-GP. Critic gives smooth gradient even with non-overlapping supports; $G$ progressively covers all three modes.

### 2.4.2 DCGAN on MNIST (Radford et al., 2015)

The DCGAN recipe — strided convolutions, batch norm, ReLU/LeakyReLU, `tanh` output — was the first stable convolutional GAN. Architecture:
- $G$: $z \in \mathbb{R}^{100}$ → FC → reshape $4 \times 4 \times 1024$ → 4 transposed-conv (stride 2) → $64 \times 64 \times 3$ tanh.
- $D$: $64 \times 64 \times 3$ → 4 strided-conv → flatten → sigmoid.

Training: Adam with $\beta_1 = 0.5$, lr $2 \times 10^{-4}$. Update $D$ once per $G$ update. Tens of thousands of iterations to converge.

### 2.4.3 Progression of Image GANs

- **DCGAN (2015)**: convolutional recipe; $64 \times 64$.
- **Progressive GAN (2017)**: grow $G$ and $D$ from low to high resolution; stable at $1024 \times 1024$.
- **StyleGAN / 2 / 3 (2019–2021)**: style-based generator with mapping network $z \to w$, modulated convolutions, disentangled style spaces; SOTA controllable face synthesis.
- **BigGAN (2018)**: class-conditional, very large batch (~2048), self-attention; SOTA ImageNet at the time. Introduced **truncation trick** (sample from truncated prior; trades diversity for fidelity).
- **GigaGAN (2023)**: text-to-image GAN at scale; competitive with diffusion when scaled carefully.

## 2.5 Variants You Should Know

### 2.5.1 Conditional GAN (cGAN) / AC-GAN

Condition $G(z, y)$ and $D(x, y)$ on $y$ (label, image, text). Powers class-conditional generation, image translation (Pix2Pix paired; CycleGAN unpaired with cycle-consistency).

### 2.5.2 WGAN / WGAN-GP / SNGAN

Address instability via Wasserstein distance + Lipschitz enforcement (clipping → gradient penalty → spectral norm).

### 2.5.3 StyleGAN

- **Mapping network**: 8-layer MLP transforms $z \sim \mathcal{N}(0, I)$ to intermediate $w$; $\mathcal{W}$ disentangles factors better than $\mathcal{Z}$.
- **Style modulation**: at each block, $w$ projects to per-channel scale/bias (AdaIN in v1; weight modulation in v2).
- **Stochastic noise**: per-pixel noise added to features controls fine details.
- **Path length regularization, lazy regularization, equalized LR**.

StyleGAN3 fixes "texture sticking" with anti-aliased operations.

### 2.5.4 BigGAN

Very large batch, class-conditional batch norm, self-attention, hinge loss. **Truncation trick** at inference: smaller truncation $\tau$ → higher fidelity, less diversity.

## 2.6 Relevance to ML Practice

**Where GANs live.** StyleGAN-family for face/avatar generation, virtual try-on, dataset augmentation; image translation (super-res, style transfer, satellite-to-map); anomaly detection (BiGAN, AnoGAN); simulation enhancement; domain adaptation.

**When to use a GAN.** You need single-pass inference (~ms latency); you want best-in-class controllability (StyleGAN); you can tolerate training instability for fast inference.

**When not to use a GAN.** You need likelihoods. Strong mode coverage matters more than stability — diffusion is usually better in 2025+.

**Trade-offs.**
- *Quality vs. diversity.* Truncation slides along this curve; mode collapse is chronic.
- *Stability vs. capacity.* Larger architectures destabilize; spectral norm/GP help.
- *Compute.* Sampling: $O(1)$ forward pass — very cheap. Training: expensive, careful tuning.
- *Evaluation.* No likelihood. Use FID, IS, Precision/Recall — all imperfect.

## 2.7 Common Pitfalls & Misconceptions

1. **"Train D to optimum then G."** Practically, alternate $k_D$ steps of $D$ and 1 step of $G$ ($k_D = 1$ vanilla; $k_D = 5$ for WGAN-GP).
2. **Batch norm in $D$.** Causes batch-dependent leakage; prefer instance norm, layer norm, or none.
3. **Comparing GANs by loss curves.** GAN losses are not monotonic quality indicators; use FID/IS.
4. **Mode collapse vs. mode dropping.** Collapse: one mode. Dropping: misses some modes. Both fixed by WGAN-GP, MBD, unrolled GANs, PacGAN.
5. **Saturating loss.** Always use non-saturating $-\log D(G(z))$.
6. **Sigmoid in WGAN critic.** WGAN critic is unbounded real; sigmoid breaks it.
7. **No Lipschitz constraint in WGAN.** Critic supremum is $+\infty$; gradients diverge. Use clipping, GP, or SN.
8. **Too few D updates.** $D$ should be roughly aligned with current $G$.
9. **Visual cherry-picking.** Misleading. Use FID over ≥10k samples; report Precision/Recall.
10. **Truncation not reported.** Different truncation values give very different FID/IS. Always report.
11. **Conditioning by concatenation.** Often worse than conditional batch norm or projection discriminator.
12. **Believing GANs estimate $p(x)$.** They don't — implicit model.

## 2.8 Interview Preparation: GANs

### Foundational

**Q1. Explain the GAN training objective in two sentences.**

A generator $G$ maps noise to samples; a discriminator $D$ classifies samples as real or fake; they play a min-max game where $D$ maximizes accuracy and $G$ minimizes it by producing samples $D$ judges as real. At the optimum (in the original formulation), $G$'s sample distribution equals $p_{\text{data}}$ and $D$ outputs $0.5$ everywhere.

**Q2. Why do GANs avoid the blurriness of VAEs?**

VAEs train with a pixel-level likelihood under a posterior averaging over plausible reconstructions, pushing the decoder toward per-pixel means (which blur high-frequency content). GANs have no per-pixel reconstruction loss; the discriminator is sensitive to high-frequency content, rewarding $G$ for sharp, locally consistent images. The cost: no likelihood and unstable training.

**Q3. What are the three classic failure modes?**

(i) **Mode collapse** — $G$ produces only a small subset of modes. (ii) **Vanishing gradient** — $D$ becomes too accurate, saturating $\log(1 - D(G(z)))$ and starving $G$. (iii) **Non-convergence / oscillation** — the min-max game cycles indefinitely without settling.

### Mathematical

**Q4. Show $D^\star(x) = p_{\text{data}}(x)/(p_{\text{data}}(x) + p_g(x))$ and that minimizing $V(D^\star, G)$ minimizes $2 \mathrm{JSD} - \log 4$.**

Pointwise objective: $a \log D + b \log(1 - D)$, $a = p_{\text{data}}, b = p_g$. Derivative zero gives $D^\star = a/(a+b)$. Substitute: $V(D^\star, G) = \mathbb{E}_{p_{\text{data}}}[\log \tfrac{p_{\text{data}}}{p_{\text{data}} + p_g}] + \mathbb{E}_{p_g}[\log \tfrac{p_g}{p_{\text{data}} + p_g}]$. Insert $-\log 2$ in each log; identify $m = (p_{\text{data}} + p_g)/2$:

$$
V(D^\star, G) = -2\log 2 + D_{\text{KL}}(p_{\text{data}} \| m) + D_{\text{KL}}(p_g \| m) = -\log 4 + 2 \mathrm{JSD}.
$$

**Q5. Why is JSD problematic with disjoint supports?**

If supports are disjoint, $p_{\text{data}}(x) p_g(x) = 0$ everywhere; the integrals collapse to $\log 2$, so JSD = $\log 2$ regardless of geometric "distance." The gradient w.r.t. $G$'s parameters is zero. In high dimensions, the data manifold is low-dimensional; $G$'s manifold and $p_{\text{data}}$'s typically intersect on a measure-zero set.

**Q6. State the WGAN objective and why it helps.**

$$
\min_G \max_{\|f\|_L \le 1} \mathbb{E}_{p_{\text{data}}}[f(x)] - \mathbb{E}_{p_g}[f(x)].
$$

This is the K-R dual of $W_1$. Wasserstein remains finite and informative with disjoint supports; its gradient corresponds to optimal transport directions. The critic provides smooth gradients; Lipschitz keeps $|f|$ bounded so no saturation.

**Q7. Why is weight clipping problematic, and what's better?**

Weight clipping pushes weights to extremes, reducing capacity and creating pathological functions. **Gradient penalty** softly enforces the unit-norm gradient constraint along lines between real and fake. **Spectral normalization** divides each layer's weight by its largest singular value (computed via one power-method iteration), yielding a globally 1-Lipschitz network at modest cost.

**Q8. Derive why the non-saturating loss $-\log D(G(z))$ has better gradients early.**

Let $u = D(G(z))$. Original $G$ loss: $\log(1 - u)$, gradient $-\nabla u/(1 - u)$. Early in training $u \approx 0$ with $D$ confidently saying "fake"; the gradient is close to $-\nabla u$ but $D$'s confidence makes $u$ extremely small and the curvature of the loss flat at this region — signal is weak in practice. Non-saturating: $-\log u$, gradient $\nabla u/u$ which *blows up* as $u \to 0$, giving $G$ a strong push when it most needs one.

### Applied

**Q9. Samples look great for one digit but not others — diagnosis and fix?**

Mode collapse. Fixes (rough order): WGAN-GP or hinge loss; spectral normalization; minibatch discrimination (let $D$ see batches); unrolled GANs; PacGAN; ensure $D$ isn't overpowering $G$ (lower $D$ updates per $G$ update).

**Q10. StyleGAN vs. diffusion for face generation in production?**

StyleGAN: single-shot inference (~ms); excellent controllable latent (W space, style mixing); mature face-editing pipeline; no likelihood; can be unstable to retrain. Diffusion: SOTA quality and diversity; natural text conditioning via CFG; iterative sampling means latency scales with steps (ms→seconds without distillation); easier to retrain stably; less mature for fine-grained latent editing. If real-time UX matters and you don't need text conditioning, StyleGAN is reasonable; otherwise default to diffusion.

**Q11. How do you evaluate a GAN?**

Don't use the discriminator's score (co-adapted, not calibrated). Standard metrics: **Inception Score** ($\exp(\mathbb{E}[D_{\text{KL}}(p(y\mid x) \| p(y))])$ — rewards confident diverse classifications, but flawed); **FID** (Fréchet distance between Inception-feature Gaussians of real vs. fake — de facto standard); **Precision/Recall** (Kynkäänniemi et al. — decomposes into quality and diversity, diagnoses mode collapse); human evaluation for high-stakes deployment.

**Q12. Conditional generation approaches?**

Concatenation cGAN (feed $y$ to both $G$ and $D$); conditional batch normalization (BigGAN); projection discriminator $D(x, y) = f(x) + y^\top V \phi(x)$ (often best); AC-GAN ($D$ also predicts class). For text: cross-attention with text embeddings.

### Debugging & failure-mode

**Q13. $D$ loss → 0, $G$ loss explodes. What now?**

$D$ winning too easily; $G$ saturating. Use non-saturating $G$ loss; check WGAN regularization (SN/GP); reduce $D$ steps per $G$ step; reduce $D$ capacity, increase $G$'s; one-sided label smoothing (target 0.9 for real); switch to WGAN-GP.

**Q14. FID improving but samples look worse.**

FID is approximate. Possible: (i) FID rewards diversity; weird new samples may match feature statistics. (ii) Sample count too low; need ≥10k, ideally 50k. (iii) Inception features mismatched to your domain; consider domain-specific FID, LPIPS, CLIP-FID.

**Q15. Mode hopping — different modes dominate at different times.**

Classic non-convergence. Fixes: WGAN-GP, spectral normalization, two-time-scale update rule (Heusel et al. — different LRs for $D$ and $G$, $D$ slightly faster), EMA of generator weights for sampling, larger batches.

**Q16. CycleGAN outputs preserve content but with bizarre artifacts.**

Cycle consistency keeps content; adversarial loss responsible for plausibility — artifacts mean the discriminator is being fooled by them. Stronger discriminator; perceptual losses; replace patch GAN if too local; replace transposed convs with upsample-then-conv to avoid checkerboard artifacts.

### Follow-ups & probes

**Q17. Relationship between GANs and Bayes-optimal classifiers?**

$D^\star(x) = p_{\text{data}}/(p_{\text{data}} + p_g)$ is exactly the Bayes-optimal classifier between two classes (real vs. fake) under uniform priors. Training $D$ amounts to fitting this Bayes classifier given current $p_g$; $G$ then plays a gradient game against it.

**Q18. How does spectral normalization enforce Lipschitz, and the cost?**

For linear layer $W$, Lipschitz constant equals $\sigma(W)$ (largest singular value). SN divides by $\sigma(W)$ → 1-Lipschitz. For a network: product of layer Lipschitz constants (assuming 1-Lipschitz nonlinearities like ReLU). Computing $\sigma(W)$ exactly is expensive; SN uses one power-method iteration per forward pass, amortized — small cost, large benefit.

**Q19. What is GAN inversion?**

Given a trained $G$, find $z^\star$ (or $w^\star$ in StyleGAN) such that $G(z^\star) \approx x$. Methods: gradient-based optimization in $\mathcal{Z}/\mathcal{W}$, learned encoders, hybrids. Useful for image editing (perturb $z^\star$ along learned semantic directions), restoration, treating $G$ as a generative prior.

**Q20. TTUR briefly?**

**Two-Time-Scale Update Rule** (Heusel et al., 2017): use different LRs for $D$ and $G$, typically $D$ slightly larger. Provably converges to a local Nash equilibrium under mild conditions. In practice (e.g., $\eta_D = 4 \times 10^{-4}, \eta_G = 1 \times 10^{-4}$) often stabilizes training.

---

<a name="3-diffusion"></a>
# 3. Diffusion Models

## 3.1 Motivation & Intuition

### 3.1.1 The Big Idea

Slowly add Gaussian noise to an image until it becomes pure static — that's easy. Now reverse: start from pure static and gradually denoise until you get a clean image. Each denoising step is a small, *learnable* prediction: "given a slightly noisy image, what was it just before the last noise?" Train a network to do each denoising step, and you can sample by running the reverse process from random noise.

Why does this work where GANs and VAEs struggle?
- The **forward process** is fixed and analytically tractable. No learning needed.
- Each **reverse step** denoises only a tiny amount — a much easier subproblem than "generate a full image from scratch."
- The objective decomposes into a sum of denoising losses, each well-posed and stable.
- The model implicitly learns the **score function** $\nabla \log p_t(x)$, connecting to score matching and Langevin dynamics.

### 3.1.2 An Everyday Analogy

Sculpting from a fog of marble dust: each step removes a little dust to reveal more form. Early steps shape the silhouette; later steps carve detail. A diffusion model inverts this for data — starting from pure noise, progressively revealing coherent structure.

### 3.1.3 Real-World ML Connection

Diffusion powers most current SOTA generation: **images** (DALL·E 2/3, Imagen, Stable Diffusion, Midjourney, FLUX); **video** (Sora, Stable Video Diffusion); **audio** (AudioLDM); **3D** (DreamFusion); **molecules and proteins** (RFdiffusion, Chroma); **robotics** (diffusion policies); **inverse problems** (super-resolution, inpainting, MRI/CT reconstruction).

## 3.2 Conceptual Foundations

### 3.2.1 Forward (Noising) Process

Define $T$ timesteps with noise schedule $\beta_1 < \dots < \beta_T \in (0, 1)$. Markov chain:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\big).
$$

Start with clean $x_0 \sim p_{\text{data}}$; end with $x_T \approx \mathcal{N}(0, I)$.

### 3.2.2 Reverse (Denoising) Process

The true reverse $q(x_{t-1} \mid x_t)$ is intractable. Parameterize:

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t)\big).
$$

Two equivalent parameterization families:
- **DDPM (Ho et al., 2020)**: predict the noise $\epsilon$ that was added.
- **Score-based (Song & Ermon, 2019)**: predict $\nabla_x \log p_t(x)$.

### 3.2.3 Components

- Noise schedule $\{\beta_t\}$.
- Network $\epsilon_\theta(x, t)$ — typically a U-Net for images, with sinusoidal time embedding and cross-attention for text/class conditioning.
- Sampler — algorithm to run the reverse process (DDPM ancestral, DDIM, DPM-Solver).

### 3.2.4 Assumptions

| Assumption | What it says | What breaks if violated |
|---|---|---|
| Reverse $q(x_{t-1} \mid x_t)$ ≈ Gaussian | Holds when $\beta_t$ small | Coarse schedules → bad Gaussian approximation |
| $x_T \sim \mathcal{N}(0, I)$ | Forward chain runs long enough | Too few steps → prior at sampling differs from actual marginal |
| Network expressive at all noise levels | One U-Net handles all $t$ | One network must denoise vastly different problems; time conditioning matters |
| Data lies on smooth manifold | Gaussian noise can bridge | Discrete data may need specialized noise (categorical diffusion) |

## 3.3 Mathematical Formulation

### 3.3.1 Closed-Form Marginals

Define $\alpha_t = 1 - \beta_t$, $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. Then

$$
\boxed{\,q(x_t \mid x_0) = \mathcal{N}\!\big(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\; (1 - \bar\alpha_t) I\big).\,}
$$

Equivalently:

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

**Derivation by induction.** Base: $q(x_1 \mid x_0)$ is Gaussian by construction. Inductive step: assume $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$. Then $x_{t+1} = \sqrt{\alpha_{t+1}} x_t + \sqrt{\beta_{t+1}} z$ for indep $z \sim \mathcal{N}(0, I)$. Plugging $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$:

$$
x_{t+1} = \sqrt{\alpha_{t+1} \bar\alpha_t} x_0 + \sqrt{\alpha_{t+1}(1 - \bar\alpha_t)} \epsilon + \sqrt{\beta_{t+1}} z.
$$

Two indep Gaussians: variances sum to $\alpha_{t+1}(1 - \bar\alpha_t) + \beta_{t+1} = 1 - \alpha_{t+1}\bar\alpha_t = 1 - \bar\alpha_{t+1}$. Mean coefficient $\sqrt{\alpha_{t+1}\bar\alpha_t} = \sqrt{\bar\alpha_{t+1}}$. ∎

### 3.3.2 Tractable Posterior $q(x_{t-1} \mid x_t, x_0)$

Given $x_0$, the conditional reverse is Gaussian:

$$
q(x_{t-1} \mid x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0),\; \tilde\beta_t I),
$$

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}} \beta_t}{1 - \bar\alpha_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t} x_t, \quad \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t.
$$

### 3.3.3 The Variational Bound

$$
-\log p_\theta(x_0) \le L_T + \sum_{t=2}^T L_{t-1} + L_0,
$$

where $L_T = D_{\text{KL}}(q(x_T \mid x_0) \,\|\, p(x_T))$ (no parameters), $L_{t-1} = \mathbb{E}_q D_{\text{KL}}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))$ (KL between two Gaussians — closed form), $L_0 = -\mathbb{E}_q \log p_\theta(x_0 \mid x_1)$.

### 3.3.4 The DDPM Simplification

Parameterize $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \tilde\beta_t I)$ via noise prediction:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left( x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(x_t, t) \right).
$$

Then

$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon}\!\left[\frac{\beta_t^2}{2 \tilde\beta_t \alpha_t (1 - \bar\alpha_t)} \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon, t)\|^2 \right].
$$

DDPM drops the prefactor:

$$
\boxed{\, L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \!\left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon, t) \|^2 \right]. \,}
$$

This is denoising score matching in disguise: predict the noise added at level $t$.

### 3.3.5 Score-Based View

Define $p_t(x_t) = \int q(x_t \mid x_0) p_{\text{data}}(x_0) dx_0$. By Vincent (2011) / Tweedie:

$$
\nabla_{x_t} \log p_t(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}.
$$

Predicting $\epsilon$ ≡ estimating the score, up to known scaling. This connects diffusion to score matching (training), Langevin dynamics (sampling), and SDEs (Song et al., 2021).

### 3.3.6 Sampling Algorithms

**DDPM ancestral.** From $x_T \sim \mathcal{N}(0, I)$:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z,\quad z \sim \mathcal{N}(0, I).
$$

**DDIM** (Song et al., 2020). Deterministic (or nearly so); same $\epsilon_\theta$, different update rule. Drastically fewer steps (10–50 typical).

**DPM-Solver / DPM-Solver++** (Lu et al., 2022). High-order ODE solvers for the probability-flow ODE; excellent samples in 10–20 steps.

**Distillation / Consistency Models** (Salimans & Ho, 2022; Song et al., 2023). Train a small/fast student to match many-step output; one-step inference becomes possible.

## 3.4 Worked Example

### 3.4.1 Training Loop

For each minibatch:
1. Sample $t \sim \mathrm{Unif}\{1, T\}$ per example.
2. Sample noise $\epsilon \sim \mathcal{N}(0, I)$.
3. Compute $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t} \epsilon$.
4. Forward $\epsilon_\theta(x_t, t)$ through U-Net (with sinusoidal time embedding).
5. Loss: $\|\epsilon - \epsilon_\theta\|^2$.
6. Backprop, optimizer step.

No two-network adversarial dance; no posterior collapse; just denoising.

### 3.4.2 Tiny Numerical Pass

$T = 4$, $\beta_t = (0.1, 0.2, 0.3, 0.4)$. Then $\alpha_t = (0.9, 0.8, 0.7, 0.6)$, $\bar\alpha_t = (0.9, 0.72, 0.504, 0.3024)$. Take $x_0 = 1.0$, $\epsilon = 0.5$, $t = 3$:
- $\sqrt{\bar\alpha_3} = \sqrt{0.504} \approx 0.710$, $\sqrt{1 - \bar\alpha_3} \approx 0.704$.
- $x_3 = 0.710 \cdot 1.0 + 0.704 \cdot 0.5 = 1.062$.
- Network outputs (say) $\hat\epsilon = 0.4$. Loss: $(0.5 - 0.4)^2 = 0.01$.

### 3.4.3 What $\epsilon_\theta$ Looks Like for Images

Standard architecture: **U-Net** with downsampling residual blocks → bottleneck with self-attention → upsampling blocks with skip connections. **Time conditioning**: sinusoidal embedding of $t$ → MLP → added (FiLM-style) to features. **Class/text conditioning**: cross-attention over embeddings.

### 3.4.4 Latent Diffusion (Stable Diffusion)

Diffusion in pixel space at $1024 \times 1024$ is expensive. **Latent Diffusion** (Rombach et al., 2022):
1. Train VAE/VQ-VAE: encoder $\mathcal{E}(x) \to z$, decoder $\mathcal{D}(z) \to x$. E.g., $64 \times 64 \times 4$ latent for $512 \times 512$ image — 64× fewer pixels.
2. Train diffusion on latents.
3. Sample latent via diffusion, decode with $\mathcal{D}$.

Orders-of-magnitude speedup with quality nearly indistinguishable from pixel-space.

### 3.4.5 Guided Generation

**Classifier guidance** (Dhariwal & Nichol, 2021). External classifier $p_\phi(y \mid x_t)$ on noisy data. Modify score:

$$
\hat\nabla \log p(x_t \mid y) = \nabla \log p(x_t) + s \cdot \nabla \log p_\phi(y \mid x_t).
$$

**Classifier-free guidance** (Ho & Salimans, 2022). One network conditioned on $y$; randomly drop $y$ during training (~10%) so the network learns both $\epsilon_\theta(x_t, t, y)$ and $\epsilon_\theta(x_t, t, \emptyset)$. Sample with:

$$
\hat\epsilon = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset)).
$$

Dominant technique today.

## 3.5 Variants & Extensions

- **DDPM** (Ho et al., 2020): the simple noise-prediction loss.
- **Improved DDPM** (Nichol & Dhariwal, 2021): cosine schedule, learned variances.
- **DDIM** (Song et al., 2020): deterministic sampler.
- **Score SDE** (Song et al., 2021): unifies DDPM, NCSN under SDEs.
- **Latent Diffusion** (Rombach et al., 2022): Stable Diffusion's foundation.
- **Classifier-Free Guidance**: de facto conditioning.
- **DPM-Solver / DPM-Solver++**: high-order ODE solvers.
- **Consistency Models**: one-step sampling via direct mapping.
- **Rectified Flow / Flow Matching**: straight-line paths between distributions; underpin SD3, FLUX.
- **EDM** (Karras et al., 2022): redesign of design space.
- **Cascaded Diffusion** (Imagen): low-res then super-resolution stages.
- **DiT** (Peebles & Xie, 2023): transformer denoiser; scales beautifully (Sora, SD3).

## 3.6 Relevance to ML Practice

**Where diffusion lives.** Effectively *everywhere* in cutting-edge generation: text-to-image, text-to-video, text-to-audio, text-to-3D, protein design, motion synthesis, robotic policies, image restoration, scientific imaging.

**When to use diffusion.** Top-tier sample quality, mode coverage, principled conditioning, stable training. When you can amortize sampling cost.

**When not to use it.** Hard real-time constraints with no chance of distillation. Need exact likelihoods. Highly discrete data without an established noise process.

**Trade-offs.**
- *Quality vs. speed.* Iterative sampling means latency is N forward passes. Many-step ancestral DDPM (best quality), few-step DDIM/DPM-Solver, or one-step distilled (fast, slight quality loss).
- *Pixel vs. latent space.* Latent saves compute but ties you to encoder/decoder; VAE artifacts leak.
- *Guidance scale.* Larger $s$ trades diversity for prompt fidelity; mode collapse at very large scales.
- *Memory.* U-Net activations dominate; gradient checkpointing and mixed precision essential.

## 3.7 Common Pitfalls & Misconceptions

1. **Treating $T$ as a hyperparameter to keep small.** $T = 1000$ isn't "too many" if your sampler is efficient. Variational bound improves with finer schedules.
2. **Confusing the trained network with the sampler.** Network predicts $\epsilon$; sampler decides how to step. Train once, use many samplers.
3. **Forgetting dequantization.** Train and evaluate with $x_0 + \mathcal{U}(-1/255, 1/255)$ for image likelihoods.
4. **Mismatched scaling.** Normalize images to $[-1, 1]$ during training; ensure sampler matches.
5. **CFG scale too high.** Causes oversaturation, mode collapse, "burned" look. Start $s = 7.5$ for text-to-image; tune.
6. **Confusing classifier guidance with CFG.** First uses external classifier; second uses jointly trained conditional/unconditional model.
7. **Mis-applying guidance to non-CFG methods.** Some samplers need their own conditioning machinery.
8. **Schedule mismatched to data.** Latents typically need different schedules than pixels (cosine, sigmoid, learned shift).
9. **Evaluating latent diffusion likelihoods as image likelihoods.** They're not.
10. **Underestimating EMA.** Use exponential moving average of model weights for sampling — materially improves quality.

## 3.8 Interview Preparation: Diffusion Models

### Foundational

**Q1. Explain diffusion training in plain language.**

Take an image, add a little Gaussian noise, train a network to predict the noise. Repeat at all noise levels. Single denoising network handles all noise scales. Sampling: start from pure noise, repeatedly subtract predicted noise, eventually reach a clean image. The math: a variational bound on $\log p(x)$, but with much nicer training dynamics than VAEs or GANs.

**Q2. Why are diffusion models more stable than GANs?**

(i) The objective is a simple regression (predict noise), not a min-max game; loss decreases monotonically. (ii) The problem at each step (denoise a small amount) is *easy*; the network never has to model the full distribution in one shot. Variance of the gradient estimator is well-behaved.

**Q3. What does the network predict?**

Most commonly the noise $\epsilon$. Equivalent parameterizations: predict $x_0$ directly, predict the score $\nabla \log p_t$, predict velocity $v$ in flow-matching frameworks. All related by linear transforms; choice affects loss weighting and conditioning.

### Mathematical

**Q4. Derive $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$.**

By induction. Base: $q(x_1 \mid x_0) = \mathcal{N}(\sqrt{\alpha_1} x_0, \beta_1 I)$ matches with $\bar\alpha_1 = \alpha_1, 1 - \bar\alpha_1 = \beta_1$. Step: $x_{t+1} = \sqrt{\alpha_{t+1}} x_t + \sqrt{\beta_{t+1}} z$. Plugging $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\epsilon$ gives $x_{t+1} = \sqrt{\bar\alpha_{t+1}} x_0 + \text{(indep Gaussian with variance } 1 - \bar\alpha_{t+1})$.

**Q5. Show the per-step KL reduces to noise-prediction MSE.**

Reverse parameterization $\mu_\theta = \tfrac{1}{\sqrt{\alpha_t}}(x_t - \tfrac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta)$ matches $\tilde\mu_t$ when $\epsilon_\theta = \epsilon$. Substituting into the closed-form Gaussian KL with equal variance:

$$
L_{t-1} = \tfrac{1}{2 \tilde\beta_t} \|\mu_\theta - \tilde\mu_t\|^2 = \frac{\beta_t^2}{2 \tilde\beta_t \alpha_t (1 - \bar\alpha_t)} \|\epsilon - \epsilon_\theta\|^2.
$$

DDPM drops the prefactor.

**Q6. Relationship between noise prediction and score?**

Since $x_t \mid x_0 \sim \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) I)$, by Tweedie:

$$
\nabla_{x_t} \log p_t(x_t) = -\frac{x_t - \sqrt{\bar\alpha_t} \mathbb{E}[x_0 \mid x_t]}{1 - \bar\alpha_t} \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}.
$$

Noise prediction = score estimation up to known scaling.

**Q7. Why does DDIM allow much faster sampling than DDPM?**

DDPM ancestral discretizes a stochastic SDE; reducing $T$ degrades quality because each step is a larger jump in a stochastic process. DDIM discretizes the deterministic probability-flow ODE implied by the same network. Deterministic ODEs admit larger step sizes for the same accuracy; 50 DDIM steps often match 1000 DDPM steps.

**Q8. Derive classifier-free guidance from Bayes' rule.**

$\nabla_x \log p(x \mid y) = \nabla_x \log p(x) + \nabla_x \log p(y \mid x)$. Classifier guidance amplifies the second term by $s$. CFG avoids the explicit classifier:

$$
\nabla_x \log p(y \mid x) = \nabla_x \log p(x \mid y) - \nabla_x \log p(x).
$$

Substitute and amplify:

$$
\hat\nabla \log p_s(x \mid y) = \nabla \log p(x) + s [\nabla \log p(x \mid y) - \nabla \log p(x)],
$$

giving in noise form: $\hat\epsilon = \epsilon_\theta(x_t, t, \emptyset) + s [\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset)]$.

### Applied

**Q9. Picking the noise schedule for a new dataset?**

Two main families: linear $\beta_t \in [10^{-4}, 0.02]$ (DDPM original); cosine schedule (Improved DDPM) — better for high-res images, SNR decays more gracefully. For latent diffusion, schedules with shifted SNR (rectified flow, EDM) often outperform cosine. Practical: start cosine, tune by FID curves.

**Q10. Compare CFG scales 1, 5, 15.**

$s = 1$: no guidance; high diversity, often poor prompt fidelity. $s = 5$ – $7.5$: sweet spot for text-to-image. $s = 15$: oversaturated, "burned" look, low diversity, sometimes mode collapse.

**Q11. Halve text-to-image latency. What do you try first?**

Order: (1) higher-order ODE solver (DPM-Solver++ at 20–25 steps often matches 50-step Heun); (2) reduce step count, validate FID; (3) distillation (consistency, progressive); (4) smaller DiT base; (5) move to latent space; (6) FlashAttention + torch.compile; (7) one-step models (consistency, SD-Turbo, SDXL-Lightning).

**Q12. Inpainting with diffusion — how?**

Several approaches. **Naive (RePaint)**: at each step, replace known region with $q(x_t \mid x_0^{\text{known}})$. **Trained inpainting**: condition on masked image and mask via channel concatenation. **Score-based posterior sampling**: combine unconditional score with $\nabla \log p(\text{observed} \mid x_t)$. Trained variant gives cleanest results.

### Debugging & failure-mode

**Q13. Text-to-image samples coherent but ignore the prompt.**

Causes: (i) CFG scale too low — raise to 5–10. (ii) CFG dropout rate during training off — standard 10%. (iii) Conditioning pathway broken — verify text embeddings reach cross-attention. (iv) Distribution shift in prompts.

**Q14. Strange grid-like artifact in samples.**

Common with low-quality VAE decoder in latent diffusion. Diagnose: pass a clean encoded latent through decoder; if artifact appears, it's the VAE. Fix: better VAE or fine-tune. Also check for transposed-conv checkerboard artifacts in U-Net upsampling.

**Q15. Loss low but samples are pure noise.**

Probably a $1/\sqrt{\alpha_t}$ scaling missing in sampler, or off-by-one indexing. Reconstruct $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t} \hat\epsilon)/\sqrt{\bar\alpha_t}$ and inspect — if reasonable, the bug is in the sampler update, not the network.

**Q16. Samples great early, degrade later.**

EMA decay too low; overfitting on small datasets; LR too high causing weight oscillation. Use EMA decay ≥ 0.999, plot validation loss, cosine LR decay.

### Follow-ups & probes

**Q17. Compare DDPM SDE with the probability-flow ODE.**

Forward noising as SDE: $dx = f(x, t) dt + g(t) dW$. Anderson's reverse-time SDE: $dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar W$. Replacing score with network gives DDPM ancestral. Probability-flow ODE has same marginals: $dx = [f(x, t) - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x)] dt$ — no noise. ODE is deterministic, amenable to high-order solvers; most fast samplers discretize it.

**Q18. Flow matching, briefly?**

Flow matching (Lipman et al., 2022) trains a velocity field $v_\theta(x, t)$ to push samples along a designed path from noise to data — typically straight lines (Rectified Flow, Liu et al., 2022). Loss again MSE: $\|v_\theta(x_t, t) - v_t\|^2$. Diffusion is one special case (curved paths from noise schedules). Straight-path flows enable few-step sampling without distillation. SD3, FLUX use flow matching.

**Q19. U-Net vs. transformer (DiT) for the denoiser?**

U-Nets historically chosen for image inductive bias (locality + multi-scale skips). Transformers (DiT) win at scale: enough compute and parameters, transformers learn the same multi-scale structure and scale better with model size and data, like NLP. Modern SOTA image/video models are mostly transformer-based.

**Q20. Consistency model briefly?**

Trained so for any pair on a probability-flow ODE trajectory, the model maps both to the same endpoint. One forward pass at any noise level produces a clean sample. Trained by distillation from a pretrained diffusion model (consistency distillation) or from scratch (consistency training). Trades a little quality for orders-of-magnitude speedup.

---

<a name="4-flows"></a>
# 4. Normalizing Flows

## 4.1 Motivation & Intuition

### 4.1.1 The Problem They Solve

We want **exact density** *and* **efficient sampling**. Autoregressive gives exact density (slow sampling); VAEs give a lower bound (loose, blurry); GANs give neither. Flows give both.

The idea: build $x = f_\theta(z)$ with $z \sim p_Z$ simple (Gaussian) and $f_\theta$ *invertible* with *tractable Jacobian determinant*. The change-of-variables formula gives exact density:

$$
p_X(x) = p_Z(f_\theta^{-1}(x)) \cdot \left| \det \frac{\partial f_\theta^{-1}}{\partial x} \right|.
$$

Sampling: forward through $f_\theta$. Density: run $f_\theta^{-1}$. Both one network pass each. Train by max likelihood. Clean.

### 4.1.2 The Catch

Two constraints make flows tricky:
- $f_\theta$ must be **invertible** — restricts architecture.
- The Jacobian determinant must be **cheap** — general $D \times D$ determinants are $O(D^3)$.

The art of flows is designing rich, expressive transformations under these constraints.

### 4.1.3 Real-World ML Connection

Flows shine in: **density estimation** (anomaly/OOD detection); **variational posteriors** (replace simple $q_\phi$ in VAEs with a flow); **scientific ML** — dominant likelihood model in lattice QCD, cosmology, particle physics (likelihoods matter); **speech synthesis** (Parallel WaveNet's student is a flow); **image generation** (Glow, Flow++, RealNVP — quality lags diffusion now).

## 4.2 Conceptual Foundations

### 4.2.1 The Recipe

A flow is $K$ invertible transformations:

$$
x = f_K \circ f_{K-1} \circ \dots \circ f_1(z).
$$

Each $f_k$ has inverse and tractable $\log |\det \partial f_k/\partial \cdot|$. By chain rule:

$$
\log p_X(x) = \log p_Z(z) - \sum_{k=1}^K \log \left| \det \frac{\partial f_k}{\partial f_{k-1}(z)} \right|.
$$

### 4.2.2 Architectural Building Blocks

- **Coupling layers** (NICE, RealNVP): split input into halves; transform one conditioned on the other. Triangular Jacobian.
- **Autoregressive flows** (MAF, IAF): each output depends only on previous. Triangular Jacobian. MAF: parallel forward, serial inverse. IAF: reversed.
- **1×1 convolutions** (Glow): learnable channel-mixing; LU decomposition for cheap log-det.
- **Continuous flows / FFJORD**: parameterize $\partial x/\partial t = f_\theta(x, t)$; integrate; log-det via Hutchinson's trace estimator.

### 4.2.3 Assumptions

| Assumption | What it says | What breaks if violated |
|---|---|---|
| $f$ is bijective | Densities transform cleanly | Excludes dim reduction, ReLU, etc. |
| $\dim(z) = \dim(x)$ | Required by invertibility | Can't natively model low-dim manifold; capacity wasted |
| Tractable Jacobian | Linear-time log-det | Generic dense layers $O(D^3)$ — intractable for images |
| Data well-modeled by smooth bijection from Gaussian | Continuous, smooth | Discrete or topologically complex distributions stress the architecture |

### 4.2.4 The "Manifold Mismatch"

Real data lives on a low-dimensional manifold in high-dimensional ambient space. Bijections cannot reduce dimension; the flow spends capacity stretching probability orthogonally to the manifold. This is why pure flows haven't beaten diffusion on image quality — they fight an architectural constraint.

## 4.3 Mathematical Formulation

### 4.3.1 Change of Variables

For diffeomorphism $f: \mathcal{X} \to \mathcal{Z}$ with $z = f(x)$:

$$
p_X(x) = p_Z(z) \cdot \left| \det \frac{\partial z}{\partial x} \right|.
$$

**Intuition.** $|\det J|$ is the local volume change factor. $f$ stretching → density drops; $f$ compressing → density rises. Total probability preserved.

For stacked flows:

$$
\log p_X(x) = \log p_Z(z) + \sum_{k=1}^K \log \left| \det \frac{\partial f_k^{-1}}{\partial h_k} \right|.
$$

### 4.3.2 Coupling Layers (RealNVP)

Split $x = (x_a, x_b)$, $x_a \in \mathbb{R}^d$, $x_b \in \mathbb{R}^{D-d}$:

$$
y_a = x_a, \quad y_b = x_b \odot \exp(s_\theta(x_a)) + t_\theta(x_a),
$$

where $s, t$ are arbitrary networks (don't need to be invertible). Inverse:

$$
x_a = y_a, \quad x_b = (y_b - t_\theta(y_a)) \odot \exp(-s_\theta(y_a)).
$$

Jacobian:

$$
\frac{\partial y}{\partial x} = \begin{pmatrix} I_d & 0 \\ \frac{\partial y_b}{\partial x_a} & \mathrm{diag}(\exp s_\theta(x_a)) \end{pmatrix},
$$

triangular, so $|\det J| = \exp(\sum_i s_\theta(x_a)_i)$.

Stack many layers, alternating active half (or applying permutations between).

### 4.3.3 Autoregressive Flows (MAF, IAF)

Write $z_i = (x_i - \mu_i(x_{<i})) \cdot \exp(-\alpha_i(x_{<i}))$. Triangular Jacobian, log-det $\sum_i \alpha_i$.

- **MAF**: forward $z = f^{-1}(x)$ parallel (one masked-MLP pass); inverse $x = f(z)$ serial. Good for density estimation.
- **IAF**: forward parallel; inverse serial. Good for fast sampling (used as variational posterior).

You cannot get parallel forward AND parallel inverse in a single autoregressive layer.

### 4.3.4 Glow's 1×1 Convolutions

Glow (Kingma & Dhariwal, 2018) introduced learnable invertible 1×1 conv per channel-mixing layer. Weight $W \in \mathbb{R}^{C \times C}$, $|\det J| = |\det W|^{HW}$ for $H \times W$ feature map. Parameterize $W = PL(U + \mathrm{diag}(s))$ (LU decomposition), so $\log|\det W| = \sum_i \log|s_i|$.

### 4.3.5 Neural Spline Flows

Replace the affine $y_b = x_b \exp(s) + t$ with a piecewise rational-quadratic spline (Durkan et al., 2019). Monotonic, learnable, tractable inverse. Significantly more expressive; near-SOTA for tabular density estimation.

## 4.4 Worked Example

### 4.4.1 2D Toy: Gaussian → Two Moons

$z \sim \mathcal{N}(0, I_2)$, $x = $ two-moons. 8 coupling layers alternating active dim, small MLPs for $s, t$. Train by max likelihood:

$$
\hat\theta = \arg\max_\theta \sum_n \log p_Z(f_\theta^{-1}(x^{(n)})) + \log |\det J_{f_\theta^{-1}}(x^{(n)})|.
$$

After training, sample $z$, push through $f_\theta$ — out come moons.

**Numeric step.** $z = (0.5, -0.3)$, layer 1: $x_a = z_1 = 0.5$, $s(0.5) = 0.2$, $t(0.5) = 0.1$, so $x_b = -0.3 \cdot e^{0.2} + 0.1 = -0.266$. Continue through subsequent layers.

### 4.4.2 Toy Numeric: Single Coupling Step

$x = (1.0, 2.0)$, $s_\theta(x_1) = 0.3$, $t_\theta(x_1) = -0.5$. Inverse:
- $z_1 = x_1 = 1.0$.
- $z_2 = (2.0 - (-0.5)) e^{-0.3} = 2.5 \cdot 0.741 = 1.852$.
- Log-det of inverse: $-\sum s = -0.3$.

Contribution to $\log p_X(x)$: $\log p_Z((1.0, 1.852)) - 0.3$.

### 4.4.3 RealNVP on Images

Image flows partition by checkerboard or channel pattern, apply convolutional $s, t$, use multi-scale architecture (factor out half the dimensions at intermediate resolutions). Train maximizes log-likelihood, reported as **bits per dimension**:

$$
\text{bpd} = -\frac{1}{D \log 2} \log p_X(x).
$$

Lower better. Glow on CelebA-HQ achieves around 1 bpd.

## 4.5 Relevance to ML Practice

**Where flows live.** Scientific likelihood modeling (physics, cosmology); tabular density estimation (neural spline flows competitive); variational posteriors in VAEs (sharply reduces variational gap); RL (flow-based policies for continuous actions); speech synthesis (Parallel WaveNet); OOD detection (with care).

**When to use flows.** Need exact, fast likelihoods. Need exact sampling. Moderate-dim data or good multi-scale architecture. Domain has track record (physics, audio).

**When not to use.** SOTA image quality (use diffusion). Low-dim manifold in high-dim space (manifold mismatch). Discrete data (autoregressive cleaner).

**Trade-offs.**
- *Expressiveness vs. tractability.* Coupling expressive but limited; spline better; CNFs more flexible but expensive.
- *Forward vs. inverse speed.* MAF: fast density, slow sampling. IAF: reversed. Coupling: equally fast both ways.
- *Memory.* More than VAEs/diffusion at equivalent expressiveness (no compression).
- *Likelihood interpretability.* Best-in-class — exact bpd, useful for OOD and lossless compression.

## 4.6 Common Pitfalls & Misconceptions

1. **High likelihood ≠ in-distribution.** Nalisnick et al. (2019): flows trained on CIFAR-10 assign higher likelihood to SVHN. Use likelihood ratios, typicality tests for OOD.
2. **Forgetting dequantization.** Pixel data is integer; without uniform/variational dequantization → meaningless infinite likelihoods.
3. **Assuming all bijections easily invertible.** Coupling and affine autoregressive: yes. General invertible MLPs: iterative inversion, slow.
4. **Conflating MAF and IAF.** Same final density, reversed cost. Pick by whether you need more density evaluation or sampling.
5. **Too few coupling layers.** With $K = 2$ you can't even mix all dimensions. Practical Glow: dozens.
6. **Confusing CNF training time with sampling time.** FFJORD slow at both — each step is an ODE solve.
7. **Numerical stability.** Exponentials in coupling can overflow; clamp $s_\theta$ output ($\tanh$-scaled) or use gated exponential.
8. **Standard batch norm in flows.** Not a true bijection per-sample (depends on batch). Use act-norm (data-dependent init with learned scale/bias) as in Glow.
9. **Underestimating manifold-mismatch penalty.** Bijection $\mathbb{R}^D \to \mathbb{R}^D$ cannot represent a $d$-dim manifold's true density; spreads probability orthogonally, hurting likelihood and quality.

## 4.7 Interview Preparation: Normalizing Flows

### Foundational

**Q1. What is a normalizing flow?**

A generative model that maps a simple base distribution (Gaussian) to data through invertible transformations. Change of variables gives exact density $p_X(x) = p_Z(f^{-1}(x)) |\det \partial f^{-1}/\partial x|$. Unlike VAEs (lower bound) and GANs (no density), flows give exact likelihood and exact sampling. Cost: architectural restriction (invertible + cheap log-det).

**Q2. What's a coupling layer?**

Split $x = (x_a, x_b)$, leave $x_a$ unchanged, transform $x_b$ via affine map whose parameters are functions of $x_a$: $y_b = x_b \cdot \exp(s_\theta(x_a)) + t_\theta(x_a)$. Jacobian triangular with diagonal $\exp(s_\theta(x_a))$, so log-det = $\sum_i s_\theta(x_a)_i$. Stack many, alternating active half, for expressivity.

**Q3. Why can't a flow reduce dimensionality?**

A bijection requires equal input and output dimension. Real data on a low-dim manifold in high-dim ambient space: flow allocates capacity orthogonally to the manifold (cannot collapse it). Limits expressivity and inflates likelihoods slightly with off-manifold mass.

### Mathematical

**Q4. State and prove the change-of-variables formula.**

For diffeomorphism $f: \mathcal{X} \to \mathcal{Z}$ with $z = f(x)$, conservation of probability requires $|p_X(x) dx| = |p_Z(z) dz|$. Jacobian links infinitesimal volumes: $dz = |\det \partial f/\partial x| dx$. Substituting: $p_X(x) = p_Z(f(x)) |\det \partial f/\partial x|$.

**Q5. Show coupling layer Jacobian is triangular.**

$$
\frac{\partial y}{\partial x} = \begin{pmatrix} \partial y_a / \partial x_a & \partial y_a / \partial x_b \\ \partial y_b / \partial x_a & \partial y_b / \partial x_b \end{pmatrix} = \begin{pmatrix} I & 0 \\ \star & \mathrm{diag}(\exp s_\theta(x_a)) \end{pmatrix}.
$$

Off-diagonal $\partial y_a/\partial x_b = 0$ because $y_a = x_a$ doesn't depend on $x_b$. Block-triangular, so $\det = \prod_i \exp(s_i)$.

**Q6. How does an autoregressive flow differ from coupling?**

In autoregressive, every output $z_i$ depends on all previous inputs $x_{<i}$. Jacobian fully triangular. MAF: parallel forward, serial inverse. IAF: reversed. Coupling = autoregressive where outputs depend on a partition (not all previous), limiting expressivity per layer but enabling parallel both ways.

**Q7. How do CNFs work, and what's the log-det trick?**

Parameterize as ODE: $dx/dt = f_\theta(x, t)$, $x(0) = z, x(1) = x$. Instantaneous change of variables:

$$
\frac{d \log p(x(t))}{dt} = -\mathrm{tr}\!\left(\frac{\partial f_\theta}{\partial x}\right).
$$

So $\log p_X(x) = \log p_Z(z) - \int_0^1 \mathrm{tr}(\partial f_\theta/\partial x(t)) dt$. Trace via Hutchinson: $\mathrm{tr}(A) = \mathbb{E}_v[v^\top A v]$ for $v$ with $\mathbb{E}[vv^\top] = I$ — one VJP per timestep instead of $D$ JVPs. This is FFJORD.

**Q8. Why must we dequantize image data?**

Pixels are integers. Treating as samples from continuous density allows arbitrary delta-spike densities → meaningless infinite likelihoods. Dequantization adds noise (uniform $\mathcal{U}(0, 1)$ per pixel). Likelihoods reported as bpd after corrections. **Variational dequantization** uses a learned noise distribution for tighter bounds.

### Applied

**Q9. Use a flow as variational posterior in a VAE — walk through.**

Replace $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi, \mathrm{diag}(\sigma_\phi^2))$ with: sample $z_0 \sim \mathcal{N}(\mu_\phi(x), \mathrm{diag}(\sigma_\phi^2(x)))$, then $z_K = f_K \circ \dots \circ f_1(z_0)$ via $K$ flow layers conditioned on $x$. ELBO becomes $\mathbb{E}_{z_K}[\log p_\theta(x \mid z_K)] + \log p(z_K) - \log q_0(z_0) + \sum_k \log|\det J_{f_k}|$. Use IAF for fast sampling (parallel forward).

**Q10. Flows vs. diffusion for tabular density estimation?**

Flows, especially neural spline flows. Reasons: tabular isn't a low-dim manifold in high-dim space (no manifold mismatch); exact likelihood matters (calibrated uncertainty, OOD); dimensionality small enough that $D$-dim bijection is fine; diffusion has no edge for low-dim while requiring more inference compute.

**Q11. Flow assigns high likelihood to OOD inputs. What's happening?**

Documented failure (Nalisnick et al., 2019). Flows trained on natural images may assign higher likelihood to constant or simple OOD images because such images have low pixel-space complexity. Mitigations: **likelihood ratios** to background distribution; **typicality** (does input lie near typical set); **complexity correction** (subtract generic compressor's bits); representation-space methods.

**Q12. MAF vs. IAF for two cases?**

Tabular density: MAF — fast forward (density evaluation); sampling rare. Variational posterior: IAF — sample $z$ many times per step; need density only on the generated sample (cache intermediates).

### Debugging & failure-mode

**Q13. Loss decreases but samples look like noise.**

Inverse $f^{-1}$ buggy or numerically unstable. Verify by round-tripping known input. Or exponential blow-ups in $s_\theta$ causing density spikes. Clamp $s_\theta$ outputs; add diagnostic round-trip checks.

**Q14. Likelihoods nonsensically high vs. competitors.**

Almost certainly a dequantization bug. Adding uniform noise to integer data? Converting "log p of dequantized continuous data" to "log p of discrete pixel" correctly (subtract $D \log 256$ for 8-bit)? Bpd conversion using $\log 2$?

**Q15. Sampling produces redundant similar samples.**

Flow converged to a narrow region of prior. Check posterior of $z$ for training data — if tightly clustered, flow barely uses prior space. Need more layers, capacity, or higher LR. For multi-scale flows, check factored-out variables; may be discarding important variation.

**Q16. Adding many coupling layers makes training unstable.**

Initialization matters. Act-norm (Glow) is data-dependent (mean 0, unit variance after first batch). Without it, deep flows diverge. Use act-norm or careful init (small initial $s_\theta$ outputs near 0). Gradient clipping helps. LR warmup.

### Follow-ups & probes

**Q17. Why is exact log-det cheap in coupling/autoregressive but not general?**

$D \times D$ determinant is $O(D^3)$ general. Triangular: product of diagonal — $O(D)$. Coupling and autoregressive layers designed to have triangular Jacobians by construction. General invertible MLPs lack this structure.

**Q18. Glow vs. RealNVP — key innovation?**

Glow's main contribution: invertible 1×1 convolution replacing fixed permutations between coupling layers. Learnable channel-mixing improves expressivity substantially. LU-decomposed weights for efficient log-det. Also act-norm replaces batch norm.

**Q19. How does the flow framework connect to score matching and diffusion?**

A continuous normalizing flow is the deterministic limit of an SDE — exactly the **probability-flow ODE** of diffusion. Diffusion's $\epsilon_\theta$ learns a score $\nabla \log p_t(x)$, which is the velocity field of a CNF whose path is constrained by the noising process. **Flow matching** explicitly trains a velocity field to match a designed coupling between distributions, generalizing both. Modern systems (rectified flow, FLUX) make this connection explicit.

**Q20. Could you build a flow on discrete data?**

Not directly with the standard formulation — change of variables requires continuous diffeomorphisms. Approaches: (i) **Dequantize** to continuous and apply continuous flow with discrete-likelihood corrections. (ii) **Categorical normalizing flows** with discrete invertible operations (permutations, integer transforms). (iii) Use autoregressive models for discrete data (native exact discrete likelihoods). Most discrete-data applications: autoregressive cleaner.

---

<a name="5-autoregressive"></a>
# 5. Autoregressive Models

## 5.1 Motivation & Intuition

### 5.1.1 The Core Idea

For a sequence $x = (x_1, \dots, x_n)$, the joint distribution factorizes by the chain rule:

$$
p(x) = \prod_{i=1}^n p(x_i \mid x_{<i}).
$$

This factorization is **always exact**. The modeling choice is how to parameterize each conditional. If a single network $p_\theta(x_i \mid x_{<i})$ models all conditionals, training reduces to next-token prediction — plain classification or regression. Maximum likelihood = per-step cross-entropy. This simplicity, combined with massive scaling, is why autoregressive transformers (GPTs) dominate.

### 5.1.2 Strengths and Weaknesses

**Strengths.** Exact likelihood; stable training (cross-entropy per step); strong density modeling at scale; easy conditioning (prepend tokens).

**Weaknesses.** Sequential sampling ($n$ forward passes for $n$ tokens); order dependence for non-sequential data; **exposure bias** (training uses ground-truth context, inference uses sampled — errors compound).

### 5.1.3 Real-World ML Connection

**Language models** (GPT, Llama, Claude, Gemini, Mistral); **speech/audio** (WaveNet, neural codec models like SoundStream, EnCodec); **image generation** (PixelCNN, ImageGPT, Parti, MUSE); **multimodal** (Chameleon, GPT-4o-style — tokenize all modalities); **code generation** (Codex, AlphaCode, Code Llama); **agents and reasoning**.

## 5.2 Conceptual Foundations

### 5.2.1 Setup

Pick an ordering on variables. For each $i$, model $p_\theta(x_i \mid x_{<i})$ with a network respecting the **causal mask** — output $i$ depends only on inputs $< i$.

Architectures:
- **RNN/LSTM**: hidden state evolves left-to-right (mostly historical now).
- **Causal CNN**: dilated causal convolutions (WaveNet, PixelCNN).
- **Causal Transformer**: self-attention with causal mask (GPT family — dominant today).

### 5.2.2 Training: Teacher Forcing

For sequence $x = (x_1, \dots, x_n)$, run all positions in **parallel** with the causal mask, predict each $x_i$ from $x_{<i}$:

$$
\mathcal{L}(\theta) = -\sum_{i=1}^n \log p_\theta(x_i \mid x_{<i}).
$$

Crucially, this uses *ground-truth* previous tokens at every position. Fast to train but creates a mismatch with sampling.

### 5.2.3 Sampling

Sequential: $x_1 \sim p_\theta(x_1)$, then $x_2 \sim p_\theta(x_2 \mid x_1)$, etc. Each step is one forward pass. Strategies:
- **Greedy / argmax**: pick most likely token. Repetitive output.
- **Temperature sampling**: $\mathrm{softmax}(\ell / \tau)$ for temperature $\tau$. Lower $\tau$ → sharper.
- **Top-$k$**: restrict to top $k$ logits.
- **Top-$p$ (nucleus)**: smallest set with cumulative prob ≥ $p$.
- **Beam search**: maintain $k$ candidates by joint log-prob. Common in MT.
- **Speculative decoding**: small "draft" model proposes tokens, big model verifies in one pass — speedup with no quality loss.

### 5.2.4 Assumptions

| Assumption | What it says | What breaks if violated |
|---|---|---|
| Chosen ordering reasonable | Some orderings make conditionals easier | Unnatural orders (random pixel) hurt density |
| Past captures relevant conditional info | Future doesn't affect past | Bidirectional info (BERT) excluded — fine for generation, suboptimal for some understanding |
| Teacher forcing approximates inference | Distribution at test matches | Exposure bias — small per-step errors compound |
| Tokenization appropriate | Right granularity | Bad tokenization hurts efficiency (numeric reasoning, multilingual) |

## 5.3 Mathematical Formulation

### 5.3.1 The Loss

Per sequence:

$$
\mathcal{L}(\theta; x) = -\sum_{i=1}^n \log p_\theta(x_i \mid x_{<i}).
$$

For LMs with softmax over vocabulary $V$:

$$
p_\theta(x_i = v \mid x_{<i}) = \frac{\exp(\ell_v^{(i)})}{\sum_{v'} \exp(\ell_{v'}^{(i)})}, \quad \ell^{(i)} = W_{\text{out}} h^{(i)}.
$$

**Perplexity**: $\mathrm{PPL} = \exp(\tfrac{1}{n} \sum_i -\log p_\theta(x_i \mid x_{<i}))$.

### 5.3.2 Causal Self-Attention

$$
h^{(i)} = \sum_{j=1}^i \alpha_{ij} V^{(j)}, \quad \alpha_{ij} = \mathrm{softmax}_j\!\left(\frac{Q^{(i)} \cdot K^{(j)}}{\sqrt{d_k}} + M_{ij}\right),
$$

where $M_{ij} = 0$ if $j \le i$ else $-\infty$ (causal mask). All $h^{(i)}$ computed in parallel during training.

### 5.3.3 KV-Caching

At inference, $K^{(j)}, V^{(j)}$ for past positions can be cached and reused. Per-step cost $O(\text{position})$ rather than $O(\text{position}^2)$ — critical for long-context generation.

### 5.3.4 PixelCNN: Masked Convolutions

For raster order, $p(x_i \mid x_{<i})$ depends on pixels above and to the left. **Masked convolutions** zero out filter weights to enforce causality:
- **Mask A** (first layer): masks center *and* right/below.
- **Mask B** (subsequent): allows center (since center hidden representation already only depends on past).

PixelCNN++ adds gated activations, dilations, discretized mixture-of-logistics output.

### 5.3.5 WaveNet: Dilated Causal Convolutions

For raw audio (16 kHz), receptive field must be large. WaveNet: causal convolutions with exponentially increasing dilations. Layer $k$ has dilation $2^k$; 10 layers reach $2^{10} = 1024$ samples. Stacked blocks extend further. Output: softmax over 256 quantized levels (μ-law) or discretized mixture of logistics.

### 5.3.6 GPT Recipe

- Token + positional embeddings.
- $L$ blocks of: causal multi-head self-attention + MLP, residual + layer norm.
- Final layer norm + linear projection to vocab logits.

Train: causal LM loss on web text. Scale: parameters ($10^9$ to $10^{12}$+), data ($10^{12}$ tokens+), context (4k → 128k → 1M+). Modern variants: rotary position embeddings (RoPE), grouped-query attention, mixture of experts, RLHF / DPO post-training.

## 5.4 Worked Example

### 5.4.1 Tiny Character-Level LM

Vocab: 27 tokens (lowercase + space). 2-layer transformer, $d = 128$, 4 heads, context 64. Train on Shakespeare. After training:
- Step $i$ logits $\ell = (5.0, 2.0, 1.0, \dots)$ for $a, b, c, \dots$.
- Temperature 1: probs $\approx (0.92, 0.05, 0.02, \dots)$ — nearly always $a$.
- Temperature 2: probs $\approx (0.78, 0.20, 0.13, \dots)$ — more variety.
- Top-$p = 0.9$: keep just $a$.

### 5.4.2 PixelCNN Generation Trace

Generating $32 \times 32$ image: 1024 sequential forward passes. Slow even on modern hardware (tens of seconds vs. milliseconds for GAN). Sample quality (2017 vintage) respectable but behind GANs of the era.

### 5.4.3 GPT-Style Inference with KV-Cache

Prompt: "The cat sat on the". Tokenize → $(t_1, \dots, t_5)$. Forward pass: logits at every position; use logits at position 5. Sample $t_6$. Append.

With KV-cache: only compute Q at the new position; reuse K, V for previous. Per-step cost: $O(L \cdot d^2 + L \cdot d \cdot \text{ctx})$. Continue until EOS or max length.

## 5.5 Variants You Should Know

### 5.5.1 PixelCNN / PixelCNN++ / PixelSNAIL

Image AR baselines. PixelCNN++: discretized mixture of logistics output (better likelihoods on continuous-valued pixels), down/up-sampling streams, dropout. Largely superseded for sample quality.

### 5.5.2 WaveNet and Successors

WaveNet (2016) made high-quality neural speech synthesis viable. Successors:
- **Parallel WaveNet** (2017): distill autoregressive teacher into an IAF student via probability density distillation.
- **WaveRNN**: more efficient autoregressive.
- **Neural audio codecs** (SoundStream, EnCodec): tokenize audio into discrete codes; AR transformer over codes (modern recipe).

### 5.5.3 GPT Family

- **GPT** (2018): 117M params, transformer LM.
- **GPT-2** (2019): up to 1.5B; zero-shot task transfer.
- **GPT-3** (2020): 175B; in-context learning at scale.
- **GPT-4** (2023): multimodal, instruction-tuned, RLHF.
- **GPT-4o, o1+**: continued scaling, reasoning fine-tuning.

Recognizable lineage: causal decoder transformers with progressively more parameters, more data, better tricks.

### 5.5.4 Image Token Models

Tokenize images via VQ-VAE/VQGAN, model token sequence autoregressively. **ImageGPT** (raw pixels reordered), **Parti** (text-to-image AR), **MUSE** (masked-token parallel decoding for speed), **Chameleon** (interleaved image and text tokens for multimodal).

### 5.5.5 Non-Autoregressive Variants

For parallel generation: **MaskGIT, MUSE** (masked-token models that iteratively unmask in parallel); discrete diffusion over token sequences. Trade quality for speed.

## 5.6 Relevance to ML Practice

**Where AR lives.** Foundation language models (dominant); code generation; multimodal foundation models; sequence-to-sequence (translation, summarization, dialogue); anomaly detection on sequences; compression (arithmetic coding); embedded speech synthesis (with distillation).

**When to use AR.** Sequential data. Exact likelihood matters. Stability and simplicity priorities. Data has natural causal/temporal order.

**When not to use.** Data has no natural order and quality is paramount — diffusion wins. Ultra-low latency without distillation possible — diffusion with consistency or one-step models may be faster.

**Trade-offs.**
- *Quality vs. inference speed.* Sequential — sample length × per-step cost. Mitigations: KV-cache, speculative decoding, distillation.
- *Training compute.* Heavily parallel across positions and batches; AR training is among the most compute-efficient.
- *Likelihood vs. sample quality.* AR likelihoods can be excellent without samples being great.
- *Exposure bias.* Real, but large-scale AR tolerates it surprisingly well; scheduled sampling, sequence-level training, RLHF address it.

## 5.7 Common Pitfalls & Misconceptions

1. **Confusing teacher forcing with sampling.** Training: ground truth context. Inference: sampled context. Mismatch = **exposure bias**. Mitigations: scheduled sampling, sequence-level RL, RLHF.
2. **Ignoring KV cache.** Without it, generation step is $O(\text{ctx}^2)$; with it, $O(\text{ctx})$. Critical for long-context.
3. **Bad ordering for non-sequential data.** Raster order biases image generation. Random orderings or specialized architectures (XLNet, MaskGIT) help, with trade-offs.
4. **Greedy for open-ended generation.** Repetitive, low-quality. Use temperature + top-$k$ / top-$p$.
5. **Assuming PPL = sample quality.** Low PPL ≠ great samples. Models with similar PPL can produce wildly different outputs (chat, code). Evaluate on the task.
6. **Forgetting special tokens.** EOS, BOS, padding. Mishandling → never-ending generation, batched evaluation leaks.
7. **Tokenization without thought.** Affects rare-word performance, multilingual capability, numeric reasoning. Some apps need character/byte tokenization.
8. **Beam search for everything.** Over-optimizes likelihood, produces generic outputs. Sampling usually wins for open-ended.
9. **Misinterpreting attention as explanation.** Attention weights correlate with but don't faithfully cause model decisions.
10. **Underestimating context-length effects.** Many models degrade past nominal context window, even with extrapolation tricks (positional interpolation, RoPE scaling). Validate on long inputs.

## 5.8 Interview Preparation: Autoregressive Models

### Foundational

**Q1. Explain autoregressive modeling in one paragraph.**

Exploit chain rule: $p(x) = \prod_i p(x_i \mid x_{<i})$. Model each conditional with one network; train by max likelihood = per-token cross-entropy. Architecture must be causal (no future inputs); causal transformers dominate. Training is parallel (teacher forcing); sampling is sequential (one forward pass per token). Result: exact likelihood, stable training, sequential inference latency.

**Q2. What's teacher forcing and its downside?**

During training, feed *true* previous tokens at every position when predicting the next. Allows parallel cross-entropy training. Downside: **exposure bias**. At inference we use *sampled* previous tokens; any error feeds back. Modern large LMs tolerate it due to scale, but it remains an issue for long generations and tasks like translation.

**Q3. Why is KV caching important?**

Causal transformer at position $i$ needs $K^{(j)}, V^{(j)}$ for $j \le i$. During parallel training, one pass. During sequential generation, naively recomputing $K, V$ for past positions every step is wasteful. KV cache stores once, reuses. Per-step cost $O(\text{ctx}^2) \to O(\text{ctx})$.

### Mathematical

**Q4. Show the chain-rule factorization is exact.**

For any joint and any ordering, the chain rule of probability gives $p(x_1, \dots, x_n) = p(x_1) p(x_2 \mid x_1) \cdots p(x_n \mid x_1, \dots, x_{n-1})$. A tautology of conditional probability — no assumptions, no approximation. Modeling work is in parameterizing each conditional.

**Q5. Derive cross-entropy loss for an AR model.**

Max likelihood: $\theta^\star = \arg\max_\theta \sum_n \log p_\theta(x^{(n)})$. AR factorization: $\log p_\theta(x) = \sum_i \log p_\theta(x_i \mid x_{<i})$. For categorical $x_i$ with softmax $p_\theta(x_i = v) = \exp(\ell_v)/\sum_u \exp(\ell_u)$:

$$
-\log p_\theta(x_i \mid x_{<i}) = -\ell_{x_i} + \log \sum_u \exp(\ell_u),
$$

which is cross-entropy between predicted distribution and one-hot $x_i$.

**Q6. Relate perplexity and bits-per-token.**

$\mathrm{PPL} = \exp(\mathrm{NLL}/n)$ in nats. Bits/token $= \log_2 \mathrm{PPL} = \mathrm{NLL}/(n \log 2)$. Monotone transforms.

**Q7. Receptive field of $L$ stacked dilated causal convolutions, kernel size 2, dilations $1, 2, 4, \dots, 2^{L-1}$?**

Each layer at dilation $2^k$ contributes $2^k$. Total: $1 + 1 + 2 + 4 + \dots + 2^{L-1} = 2^L$. So 10 layers reach 1024 samples — WaveNet's receptive field.

**Q8. State Tweedie's formula and its connection to score-based denoising.**

For $y = x + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$: $\mathbb{E}[x \mid y] = y + \sigma^2 \nabla_y \log p(y)$. So an optimal denoiser implicitly estimates the score of the noisy data. Bridges denoising and score-based generative modeling — central to diffusion's foundations.

### Applied

**Q9. Choose between greedy, beam search, and nucleus sampling for: machine translation; open-ended dialogue.**

**MT**: beam search (e.g., width 4–10) — output space is constrained, you want high-likelihood translations. **Open-ended dialogue**: nucleus sampling (top-$p \approx 0.9$) with moderate temperature — beam produces generic, repetitive responses; sampling preserves diversity and naturalness. Greedy almost never optimal for open-ended.

**Q10. Speculative decoding — how does it work?**

A small "draft" model proposes $K$ tokens autoregressively (cheap). The big "target" model evaluates all $K$ tokens in one parallel forward pass, computing $p_{\text{target}}(t_i \mid t_{<i})$ for each. Accept tokens with probability $\min(1, p_{\text{target}}/p_{\text{draft}})$ greedily until first rejection; resample the rejected token from a corrected distribution. Mathematically equivalent to sampling from the target model — no quality loss. Speedup: typically 2–3× for modern LM serving.

**Q11. PixelCNN vs. diffusion for image generation today?**

Diffusion. PixelCNN: exact likelihood (advantage); slow sequential sampling per pixel; raster order biases samples; quality consistently behind diffusion. Diffusion: bound (not exact) likelihood; iterative sampling but parallel within each step; better quality and conditioning. The use case for AR image models survives in tokenized-latent settings (Parti, MUSE) where AR has compositional advantages.

**Q12. Why is RLHF used on top of an AR LM?**

Maximum-likelihood pretraining produces a model that imitates training data distribution — including its undesirable patterns. RLHF (or DPO) optimizes a learned reward model approximating human preferences. The model learns to produce outputs preferred by humans (helpful, harmless, on-task) rather than just plausible-looking. Mathematically: post-training KL-constrained policy optimization where the policy is the LM and the reward comes from human comparisons.

### Debugging & failure-mode

**Q13. Generation produces repetitive loops.**

Common with greedy or low-temperature sampling, especially in models with strong unigram biases. Fixes: increase temperature; use nucleus sampling; **repetition penalty** (down-weight tokens recently emitted); **frequency/presence penalties**; check for tokenizer bugs that cause "stuck" tokens.

**Q14. Validation loss great, generation quality poor.**

Possible causes: distribution shift between training prompts and test prompts; bad sampling settings; exposure bias making errors compound on long generations; tokenizer mismatch in evaluation pipeline. Check by comparing teacher-forced loss to free-running quality, examining failure cases in detail, and ensuring tokenization is identical.

**Q15. Long-context performance degrades sharply past N tokens.**

Position-encoding extrapolation issue. RoPE with linear scaling, NTK-aware scaling, YaRN, or PI (positional interpolation) extends usable context, but quality often drops. Best fix: train (or fine-tune) on long contexts directly. Also check for attention sink behavior, "lost in the middle" issues with relevant info placement.

**Q16. Inference fast on CPU but slow on GPU.**

Likely small batch + KV-cache making memory-bound; you're not utilizing GPU compute. Solutions: batch multiple sequences (continuous batching); use FlashAttention for memory-efficient attention; quantize weights (int8, int4); use serving frameworks (vLLM, TensorRT-LLM) with paged KV cache.

### Follow-ups & probes

**Q17. Why does scaling work so well for AR LMs?**

Empirically, neural scaling laws (Kaplan et al., 2020; Chinchilla, Hoffmann et al., 2022) show test loss decreases predictably as a power law in model size, dataset size, and compute. Compute-optimal allocation balances all three (Chinchilla: ~20 tokens per parameter). Causal LM is uniquely well-suited to scaling: stable optimization, simple objective, abundant data, parallel training. The same recipe scaled across orders of magnitude continues to improve.

**Q18. Mixture of Experts (MoE) — what and why?**

Replace dense FFN layers with a sparse mixture of $E$ "expert" FFNs. A gating network routes each token to top-$k$ experts (typically $k = 2$). Total parameters $\approx E \times$ dense; FLOPs ≈ dense (only $k$ experts active per token). Larger model capacity at fixed inference cost. Training challenges: load balancing (experts shouldn't collapse to a few), routing gradient flow (often estimated via noisy top-$k$). Used in Mixtral, GPT-4 (rumored), Gemini.

**Q19. RoPE in one paragraph.**

**Rotary Position Embedding** (Su et al., 2021). Rotate the query and key vectors at position $i$ by angle $i \theta_d$ for each dimension pair $d$, with $\theta_d$ from a geometric series. The dot product $Q^{(i)} \cdot K^{(j)}$ becomes a function of relative position $i - j$, encoding position implicitly. Better long-context behavior than learned absolute embeddings; supports linear extrapolation and various scaling tricks. Ubiquitous in modern LLMs.

**Q20. How would you do continual learning / fine-tuning on a pretrained AR LM without catastrophic forgetting?**

Several approaches: (i) **Low learning rate + replay** of pretraining data mixed with new data. (ii) **LoRA** (Low-Rank Adaptation) — freeze base weights, train low-rank update matrices. (iii) **Adapters** — small MLPs inserted between layers. (iv) **Elastic Weight Consolidation** — penalize movement of important weights (Fisher-weighted). (v) **Mixture-of-experts** for new domains. For most practical purposes, LoRA + replay is the workhorse — cheap, effective, easy to deploy multiple specialized variants.

---

<a name="6-comparisons"></a>
# 6. Cross-Cutting Comparisons

## 6.1 Choosing a Model Family

A practitioner's decision tree:

1. **Need exact likelihood?** → Autoregressive or Flows. AR for sequential / discrete data; Flows for continuous data with moderate dimension.
2. **Need top sample quality?** → Diffusion (default in 2025+) or GANs (StyleGAN-family for fast inference + controllability).
3. **Need a smooth probabilistic latent space?** → VAE or VQ-VAE (often as a perceptual compressor inside a diffusion or AR pipeline).
4. **Need ultra-fast (one-shot) sampling?** → GANs or distilled / consistency-model diffusion or normalizing flows.
5. **Need controllable generation from text/labels?** → Diffusion with classifier-free guidance, or AR with conditioning prefixes (chat/image-token models).

## 6.2 Conceptual Unifications

Several unifications across the families are worth holding in mind:

- **Diffusion is a hierarchical VAE** with $T$ latents and *fixed* (not learned) encoders. The ELBO decomposes into per-step denoising losses.
- **Diffusion = score-based generative modeling**: predicting noise ≡ estimating $\nabla \log p_t(x)$ up to known scaling.
- **Diffusion sampling is a discretization of the probability-flow ODE** — same ODE underpins continuous normalizing flows. Flow matching makes this connection explicit.
- **GANs minimize JSD** (or Wasserstein); other families minimize a likelihood-based objective. The implicit-vs-explicit distinction is the main divide.
- **Tokenization + AR is a universal recipe**: any modality compressed into tokens (BPE for text, VQ for images, neural codecs for audio) can be modeled by a causal transformer.

## 6.3 Composite Architectures You'll See in the Wild

- **Stable Diffusion**: VAE encoder + diffusion in latent space + VAE decoder + cross-attention text conditioning + CFG.
- **DALL·E 2 / Imagen**: text encoder → diffusion prior → diffusion decoder (sometimes cascaded).
- **Parti / MUSE / Chameleon**: VQ-tokenizer + AR transformer (or masked-token model) over tokens.
- **VAE-GAN, VQGAN**: VAE/VQ-VAE with adversarial loss on the decoder for sharper reconstructions.
- **Parallel WaveNet**: AR teacher + IAF flow student via probability density distillation.

## 6.4 Evaluation Across Families

| Metric | Use case | Caveats |
|---|---|---|
| Bits per dimension (bpd) | Likelihood-based models on continuous data | Requires correct dequantization |
| Perplexity | Language models | Sensitive to tokenization; not a sample-quality measure |
| FID | Image generators | Inception-feature-based; needs ≥10k samples |
| Inception Score | Image generators (legacy) | Many flaws; FID preferred |
| Precision / Recall (Kynkäänniemi) | Image generators | Decomposes quality vs. diversity; diagnoses mode collapse |
| CLIP score | Text-to-image | Measures prompt alignment |
| Human eval | Anything high-stakes | Gold standard but expensive |
| Downstream task performance | Representation models | Most meaningful when the model is a means to an end |

## 6.5 What I'd Tell a Junior Researcher Today

- **Start with diffusion** for any new continuous-data generative problem; it's the most forgiving paradigm and the SOTA in most domains.
- **Use AR transformers** for sequential or tokenized data; the toolchain (HuggingFace, vLLM, FlashAttention) is mature.
- **Use VAEs as components** rather than as standalone generative models for high-fidelity data — they shine as perceptual compressors and as representation learners.
- **Reach for GANs** when StyleGAN-style controllability or single-pass inference latency is non-negotiable.
- **Reach for flows** in scientific domains where exact likelihoods matter and dimension is moderate.
- **Don't reinvent**: most production systems are composites (e.g., VAE + diffusion + text encoder + CFG); learn the recipe for each piece.

