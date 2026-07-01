# Reinforcement Learning (RL)

## 1. Motivation & Intuition

### Why Reinforcement Learning?

Most Machine Learning is **Supervised Learning** (SL), where a model learns from a dataset of correct answers (labels). If you want to classify images of cats, you show the model thousands of images labeled "cat."

But what if you are training a robot to walk? There is no dataset of "correct muscle movements" for every millisecond of walking on uneven terrain. The robot must attempt to move, fall over, adjust, and try again.

**Reinforcement Learning** is the branch of ML concerned with **decision-making over time**. It solves problems where:

1. **There is no supervisor:** No one tells the agent the "correct" action immediately.
2. **Feedback is delayed:** A move made now might only pay off (or cause a crash) 100 steps later.
3. **Data is non-iid:** The agent's actions determine the next data point it sees (if I turn left, I see a different corridor than if I turn right).

### Real-World Intuition

Think of training a dog. You don't physically move the dog's legs to teach it to "sit." Instead:

* **Command:** You say "Sit."
* **Action:** The dog jumps. (Result: Nothing happens/Negative feedback).
* **Action:** The dog sits.
* **Reward:** You give the dog a treat.
* **Learning:** The dog associates the state (hearing "Sit") and the action (lowering hind legs) with a positive outcome (Treat).

In system design, this maps to:

* **Recommender Systems:** The agent (Netflix) recommends a movie (Action). The user watches it (Reward). The goal is to maximize watch time over the user's lifetime (Long-term return).
* **Datacenter Cooling:** Google uses RL to control fans and windows in data centers. The agent tweaks settings to minimize energy usage while keeping servers safe.

---

## 2. Conceptual Foundations

### The RL Loop

The standard RL model involves two main entities:

1. **The Agent:** The learner and decision-maker.
2. **The Environment:** Everything outside the agent.

They interact in a discrete time loop ($t=0, 1, 2, ...$):

1. The Agent observes the current **State** ($S_t$) of the environment.
2. The Agent selects an **Action** ($A_t$).
3. The Environment transitions to a new state ($S_{t+1}$) and emits a scalar **Reward** ($R_{t+1}$).
4. The Agent updates its knowledge to maximize future rewards.

### The Markov Decision Process (MDP)

To solve RL problems mathematically, we formalize the environment as an MDP. An MDP is defined by the tuple $\langle S, A, P, R, \gamma \rangle$:

1. **State Space ($S$):** The set of all possible situations (e.g., board positions in Chess).
2. **Action Space ($A$):** The set of all things the agent can do (e.g., move Knight to F3).
3. **Transition Probability ($P$):** The dynamics of the world.

$$
P(s' | s, a) = \mathbb{P}[S_{t+1} = s' \mid S_t = s, A_t = a]
$$

   *Assumption:* The **Markov Property**. The future depends *only* on the current state, not the history of how we got there.

4. **Reward Function ($R$):** The immediate signal.

$$
R(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]
$$

5. **Discount Factor ($\gamma \in [0, 1]$):** Determines how much we care about the future.
   * $\gamma = 0$: Myopic (greedy). Only cares about immediate reward.
   * $\gamma \to 1$: Far-sighted. Values long-term consequences.

### Key Components

* **Policy ($\pi$):** The agent's brain. A mapping from state to action.
  * Deterministic: $a = \pi(s)$
  * Stochastic: $\pi(a|s) = \mathbb{P}[A_t=a | S_t=s]$
* **Return ($G_t$):** The cumulative discounted reward. This is what we want to maximize.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

---

## 3. Mathematical Formulation

The core of solving an MDP is finding the **Optimal Policy** ($\pi^*$) that maximizes the expected return. To do this, we use **Value Functions**.

### Value Functions

1. **State-Value Function ($V(s)$):** How good is it to be in state $s$?

$$
V_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
$$

2. **Action-Value Function ($Q(s, a)$):** How good is it to take action $a$ in state $s$?

$$
Q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
$$

The relationship between $V$ and $Q$:

$$
V_\pi(s) = \sum_{a} \pi(a|s) Q_\pi(s, a)
$$

### The Bellman Equations

The recursive definition that powers RL. It states that the value of the current state is the immediate reward plus the discounted value of the next state.

**Bellman Expectation Equation (for a specific policy $\pi$):**

$$
V_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s'|s,a) \left[ r + \gamma V_\pi(s') \right]
$$

**Bellman Optimality Equation (for the optimal policy $\pi^*$):**

$$
V_*(s) = \max_{a} \sum_{s', r} P(s'|s,a) \left[ r + \gamma V_*(s') \right]
$$

$$
Q_*(s, a) = \sum_{s', r} P(s'|s,a) \left[ r + \gamma \max_{a'} Q_*(s', a') \right]
$$

---

## 4. Algorithms & Worked Examples

We categorize algorithms based on what information they have and how they store knowledge.

### A. Tabular Methods (Small State Spaces)

When $S$ and $A$ are small enough to fit in a table (e.g., Tic-Tac-Toe, Grid World).

#### 1. Dynamic Programming (DP)

* **Assumption:** We know the perfect model of the world (we know $P$ and $R$).
* **Method:** We iteratively update the table using the Bellman equation until it converges.
* **Policy Iteration:**
  1. *Evaluation:* Calculate $V_\pi$ for current $\pi$.
  2. *Improvement:* Update $\pi$ to be greedy with respect to $V_\pi$.
* **Value Iteration:** Combine steps. Iterate $V(s) \leftarrow \max_a \sum P(s'|s,a)[r + \gamma V(s')]$.

#### 2. Monte Carlo (MC)

* **Assumption:** Model-free (we don't know $P$). We must learn from experience.
* **Method:** Play entire episodes ($s_0, a_0, r_1, ..., s_T$). Calculate the actual return $G_t$ observed and average it.

$$
V(s) \approx \text{Average of all returns observed after visiting } s
$$

* **First-visit vs. Every-visit:** Do we count a state multiple times if we loop back to it in one episode? Usually, first-visit is unbiased but higher variance.

#### 3. Temporal Difference (TD) Learning

The "Holy Grail" combining DP and MC.

* **Idea:** Don't wait until the end of the episode (like MC). Update your estimate based on your *next* estimate (bootstrapping).
* **Update Rule:**

$$
V(s) \leftarrow V(s) + \alpha [ \underbrace{r + \gamma V(s')}_{\text{Target}} - V(s) ]
$$

  * $\alpha$: Learning rate.
  * $r + \gamma V(s')$: The TD Target.

**Q-Learning (Off-policy TD):**
Learns the value of the *optimal* policy regardless of the *current* policy (often $\epsilon$-greedy).

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [ r + \gamma \max_{a'} Q(s', a') - Q(s, a) ]
$$

**SARSA (On-policy TD):**
Learns the value of the policy *currently being followed*.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [ r + \gamma Q(s', a') - Q(s, a) ]
$$

---

### B. Function Approximation (Deep RL)

When $S$ is too large (e.g., images, continuous physics), we cannot use a table. We approximate $Q(s, a) \approx Q_\theta(s, a)$ using a Neural Network with parameters $\theta$.

#### 1. Deep Q-Networks (DQN)

Adapts Q-Learning to Neural Networks.

* **Challenge:** Naive Q-learning with NNs is unstable because data is correlated (sequential frames).
* **Solution 1: Experience Replay Buffer:** Store transitions $(s, a, r, s')$ in a buffer. Train on random mini-batches to break correlation.
* **Solution 2: Target Network:** The target $r + \gamma \max Q(s', a')$ keeps changing as we update the network, leading to "chasing your own tail." DQN uses a frozen copy of the network for the target, updated every $C$ steps.

#### 2. Policy Gradients (PG)

Instead of learning values ($Q$), learn the policy $\pi_\theta(a|s)$ directly.

* **Objective:** Maximize $J(\theta) = \mathbb{E}[G_t]$.
* **REINFORCE Algorithm:**

$$
\nabla_\theta J(\theta) = \mathbb{E} [ \nabla_\theta \ln \pi_\theta(a|s) \cdot G_t ]
$$

  *Intuition:* If an action $a$ led to a high return $G_t$, increase the probability (log-likelihood) of taking action $a$ in state $s$.

#### 3. Advanced Algorithms

* **Actor-Critic:** Hybrids. The **Actor** learns the policy $\pi$. The **Critic** learns the value function $V$ to reduce the variance of the Actor's updates.
* **PPO (Proximal Policy Optimization):** Limits how much the policy changes in one step using a "clipped" objective function. This prevents the "collapse" where a bad update destroys the policy.
* **Soft Actor-Critic (SAC):** Maximizes reward *plus* the entropy (randomness) of the policy. Encourages exploration and robustness.

---

### C. Exploration Strategies

How do we balance doing what we know is good (**Exploitation**) vs. trying new things (**Exploration**)?

1. **$\epsilon$-greedy:** With probability $\epsilon$, take a random action. Otherwise, take the best action.
2. **Upper Confidence Bound (UCB):** Add a "bonus" to actions we haven't tried much:

$$
A_t = \text{argmax} \left[ Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]
$$

3. **Thompson Sampling:** Maintain a probability distribution for the value of each action. Sample from these distributions and pick the highest.
4. **Curiosity-Driven:** The agent gets an intrinsic reward if it encounters a state that surprises its internal model (prediction error).

---

## 5. Worked Example: The "Cliff Walking" Grid

**Scenario:** A $4 \times 12$ grid.

* **Start:** Bottom-left (3, 0).
* **Goal:** Bottom-right (3, 11).
* **The Cliff:** The bottom row between Start and Goal (3, 1 to 3, 10). Stepping here yields Reward -100 and sends you back to Start.
* **Step Reward:** -1 for all other steps.
* **Discount ($\gamma$):** 1.0 (undiscounted).

**Walkthrough with Q-Learning (Off-policy):**

1. **Init:** $Q(s, a) = 0$ for all states.
2. **Step 1:** Agent is at Start. $\epsilon$-greedy chooses "Right".
3. **Outcome:** Agent lands in the Cliff! $r = -100$. Reset to Start.
4. **Update:**

$$
Q(\text{Start}, \text{Right}) \leftarrow 0 + \alpha [-100 + 1 \cdot \max Q(\text{Start}, :) - 0]
$$

   $Q(\text{Start}, \text{Right})$ becomes negative (e.g., -10).

5. **Step 2:** Agent is back at Start. "Right" now looks bad. It chooses "Up". $r = -1$. New state (2, 0).

$$
Q(\text{Start}, \text{Up}) \leftarrow 0 + \alpha [-1 + 1 \cdot \max(0) - 0] = -\alpha
$$

**Result:**

* **Q-Learning:** Finds the **Optimal Path**. It walks right along the edge of the cliff because it assumes it will take the optimal action (not fall) next time.
* **SARSA:** Finds the **Safe Path**. It realizes that because the policy is $\epsilon$-greedy, there is a chance it might accidentally step into the cliff. It learns a path further away from the edge (upper rows).

---

## 6. Relevance to Machine Learning Practice

### When to use RL?

* **Use RL when:** The sequence matters, feedback is delayed, and you can simulate the environment (or afford mistakes).
* **Do NOT use RL when:** You have a labeled dataset (use Supervised Learning) or mistakes are catastrophic and simulators don't exist (e.g., learning to fly a real plane from scratch).

### Trade-offs

* **Sample Efficiency:** RL is notoriously data-hungry. It might take millions of frames to learn Atari games.
* **Stability:** RL training (especially Deep RL) is highly sensitive to hyperparameters (learning rate, discount factor, reward scaling).
* **Reward Hacking:** The agent will exploit loopholes. (e.g., In a boat racing game, an agent found it got points for collecting powerups, so it spun in circles collecting powerups indefinitely instead of finishing the race).

---

## 7. Interview Preparation

### Foundational Questions

#### Q1: What is the difference between On-policy and Off-policy learning?

**Answer:**

* **On-policy (e.g., SARSA, PPO):** The agent learns the value of the policy it is *currently* executing. If the agent explores (takes random actions), the learned value reflects that exploration.
* **Off-policy (e.g., Q-Learning, DQN):** The agent learns the value of the *optimal* policy, regardless of the policy it uses to generate data. It assumes that eventually, it will stop exploring and act greedily.

#### Q2: Explain the Markov Property.

**Answer:** A state $S_t$ is Markov if it contains all necessary information to predict the future. Formally:

$$
P(S_{t+1} \mid S_t) = P(S_{t+1} \mid S_t, S_{t-1}, \dots)
$$

If a state is not Markov (e.g., a single frame of a moving ball doesn't show velocity), we often stack frames or use RNNs to approximate the Markov state.

### Mathematical Questions

#### Q3: Derive the Bellman Expectation Equation for $V_\pi(s)$.

**Answer:**

$$
V_\pi(s) = \mathbb{E}[G_t \mid S_t=s]
$$

$$
= \mathbb{E}[R_{t+1} + \gamma G_{t+1} \mid S_t=s]
$$

We expand the expectation over actions $a$ taken by $\pi$ and transitions to $s'$:

$$
= \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \mathbb{E}[G_{t+1} \mid S_{t+1}=s'] \right]
$$

Since $\mathbb{E}[G_{t+1} \mid S_{t+1}=s'] = V_\pi(s')$:

$$
V_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} P(s'|s,a) [ r + \gamma V_\pi(s') ]
$$

#### Q4: Why do we need a discount factor $\gamma$? What happens if $\gamma=1$ in a continuous task?

**Answer:**

1. **Mathematical convergence:** It ensures the infinite sum of rewards $G_t$ is finite (geometric series).
2. **Uncertainty:** We trust immediate rewards more than far-future ones.
3. If $\gamma=1$ in a task without a terminal state (continuous), the return $G_t$ could go to infinity, causing value iteration to diverge.

### Applied & System Design Questions

#### Q5: You are designing an RL system to control traffic lights to minimize wait time. How do you define the State, Action, and Reward?

**Answer:**

* **State:** The number of cars in each lane (queue length), current light phase, time of day. (Maybe image inputs from cameras).
* **Action:** Switch light to Green/Red, or Extend Green by 5 seconds.
* **Reward:** Negative sum of squared queue lengths (penalizes long queues heavily) or Negative total wait time per step.
* **Constraint:** Safety (cannot turn all green).

#### Q6: What is "Experience Replay" in DQN and why is it necessary?

**Answer:** In standard online learning, data arrives sequentially ($s_t, s_{t+1}, ...$). These are highly correlated. Gradient descent assumes data is independent and identically distributed (i.i.d). Training on correlated data causes the network to overfit to the local situation and "forget" earlier experiences (Catastrophic Forgetting). Experience Replay stores transitions in a buffer and samples *random* batches, restoring the i.i.d property and stabilizing training.

### Debugging & Failure Modes

#### Q7: Your RL agent is not learning. The reward curve is flat. What do you check?

**Answer:**

1. **Sparse Rewards:** Is the agent ever seeing a positive reward? If not, it can't learn. (Fix: Reward shaping or better exploration).
2. **Input Scaling:** Are inputs normalized? (e.g., one input is 0-1, another is 0-1000). This kills gradients.
3. **Hyperparameters:** Is the learning rate too high (divergence) or too low (slow)? Is $\epsilon$ decaying too fast?
4. **Bug in the environment:** Is the `step()` function actually returning the correct next state?
5. **Observation clipping:** Is the agent seeing the relevant information?

#### Q8: The agent finds a policy that achieves high reward but behaves undesirably (e.g., spinning in circles). What is this called and how do you fix it?

**Answer:** This is **Reward Hacking** (or Specification Gaming). The agent optimized the *proxy* reward you gave it, not the *intent*.
* **Fix:** Adjust the reward function to penalize the unwanted behavior or include a "progress" term. Use **Inverse RL** to learn the reward from human demonstration rather than hand-coding it.

### Advanced/Probing Questions

#### Q9: Explain the difference between Model-Based and Model-Free RL.

**Answer:**

* **Model-Free (DQN, PPO):** Learns a mapping from state to action/value directly. Does not "know" physics. It just knows "If I do X, I get 10 points."
* **Model-Based (AlphaZero, Dyna):** Learns a model of the environment ($S_{t+1} = f(S_t, A_t)$). It can "plan" by simulating future steps in its "head" before acting in the real world.
* **Trade-off:** Model-based is much more sample efficient (needs less data) but suffers from **model bias** (if the learned model is wrong, the plan is wrong).

#### Q10: What is the "Deadly Triad" in RL?

**Answer:** Instability and divergence occur when you combine three elements:

1. **Function Approximation:** Using NNs (not tables).
2. **Bootstrapping:** Updating estimates based on other estimates (TD learning).
3. **Off-Policy Training:** Training on data generated by a different policy.

*DQN manages this using Target Networks (dampens bootstrapping issues) and Experience Replay.*

---

## 8. The Policy Gradient Theorem & Variance Reduction

Section 4 introduced REINFORCE, but it works "by magic" — we never justified *why* $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \, G_t]$. The **Policy Gradient Theorem** is the foundation of every modern policy-based method (A2C, PPO, SAC), so it is worth stating precisely.

### The Theorem
For an objective $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$, the gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^\pi,\, a \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \, Q^{\pi}(s, a) \right]
$$

The remarkable part: the gradient does **not** require differentiating the environment dynamics $P(s'|s,a)$. We only differentiate the **policy**, and the effect of the environment enters through the (non-differentiated) return $Q^\pi$. This is what makes RL work on black-box, non-differentiable simulators.

* **The log-derivative trick** is the mechanism: $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$. It converts a gradient of a probability into an expectation we can estimate from samples.

### The Variance Problem and Baselines
REINFORCE is unbiased but **extremely high variance**, because $G_t$ is a noisy Monte Carlo sample of the return. The key insight: we can subtract any **baseline** $b(s)$ that does not depend on the action without introducing bias:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a \mid s) \, \big( Q^{\pi}(s,a) - b(s) \big) \right]
$$

This is unbiased because $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \, b(s)] = b(s) \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \nabla_\theta 1 = 0$. The optimal, most-variance-reducing baseline is the **state value** $V^\pi(s)$, which turns $Q - b$ into the **advantage function**:

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
$$

The advantage answers "how much better is action $a$ than the policy's *average* action in state $s$?" — a naturally low-variance, zero-centered signal.

### Generalized Advantage Estimation (GAE)
How do we estimate $A_t$? There is a bias–variance spectrum. The one-step TD residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is low-variance but biased (it trusts the value estimate); the full Monte Carlo return is unbiased but high-variance. **GAE** interpolates with a decay parameter $\lambda$:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

* $\lambda = 0$: $\hat{A}_t = \delta_t$ — pure one-step TD (low variance, high bias).
* $\lambda = 1$: $\hat{A}_t = \sum_l \gamma^l r_{t+l} - V(s_t)$ — Monte Carlo (high variance, unbiased).
* $\lambda \approx 0.95$ in practice — the workhorse setting used by PPO.

---

## 9. Actor-Critic, TRPO, and PPO

### The Actor-Critic Architecture
Actor-critic makes the baseline learnable and estimated online:
* **Actor** $\pi_\theta(a|s)$ — the policy, updated via the policy gradient using the advantage.
* **Critic** $V_\phi(s)$ — a value network trained by regression on TD targets, supplying the advantage estimate $\hat{A}_t$ to the actor.

**A2C/A3C** are the canonical synchronous/asynchronous implementations. They collect short rollouts, compute GAE advantages from the critic, and take one gradient step. The problem: a plain policy-gradient step can be **too large**, collapsing the policy — RL has no fixed dataset, so a bad update poisons all future data collection.

### TRPO — the Trust Region
**Trust Region Policy Optimization** bounds how far the policy can move per update by constraining the KL divergence between the new and old policies:

$$
\max_\theta \; \mathbb{E}_t\!\left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{subject to} \quad \mathbb{E}_t\big[ D_{\text{KL}}(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta) \big] \le \delta
$$

This guarantees monotonic-ish improvement but requires expensive second-order (conjugate-gradient) machinery.

### PPO — the Clipped Objective (the modern default)
**Proximal Policy Optimization** achieves TRPO's stability with only first-order SGD. Define the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$. PPO optimizes:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[ \min\Big( r_t(\theta) \hat{A}_t, \;\; \text{clip}\big(r_t(\theta),\, 1 - \epsilon,\, 1 + \epsilon\big) \hat{A}_t \Big) \right]
$$

* **Intuition:** the clip (typically $\epsilon = 0.2$) removes the incentive to push the ratio far from 1. When an action was good ($\hat{A}_t > 0$), the objective stops rewarding further probability increases past $1+\epsilon$; when bad, it stops past $1-\epsilon$. The `min` makes the bound **pessimistic**, so the policy never over-commits to a single batch.
* PPO is on-policy, remarkably robust to hyperparameters, and is the algorithm behind most large-scale RL — including RLHF for LLMs (§13).

---

## 10. Continuous Control: DDPG, TD3, SAC

DQN's $\max_{a'} Q(s',a')$ requires enumerating actions — impossible for continuous action spaces (robot joint torques, steering angles). The following methods handle continuous $\mathcal{A}$.

### DDPG — Deterministic Policy Gradient
DDPG learns a **deterministic** actor $\mu_\theta(s)$ and an off-policy critic $Q_\phi$. The **Deterministic Policy Gradient theorem** replaces the $\max$ with a differentiable chain through the critic:

$$
\nabla_\theta J = \mathbb{E}_{s \sim D}\!\left[ \nabla_a Q_\phi(s, a)\big|_{a = \mu_\theta(s)} \; \nabla_\theta \mu_\theta(s) \right]
$$

i.e. "move the actor in the direction that the critic says increases $Q$." It uses replay + target networks like DQN, and adds exploration noise to the deterministic action. DDPG is powerful but notoriously brittle — it **overestimates** $Q$ and can diverge.

### TD3 — Fixing DDPG's Overestimation
**Twin Delayed DDPG** adds three fixes:
1. **Clipped double-Q:** learn two critics and use the **minimum** for the target, curbing overestimation: $y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde{a})$.
2. **Delayed policy updates:** update the actor (and targets) less frequently than the critics, so the actor chases a more stable value estimate.
3. **Target policy smoothing:** add clipped noise to the target action $\tilde{a}$, regularizing sharp, exploitable peaks in $Q$.

### SAC — Maximum-Entropy RL
**Soft Actor-Critic** augments the reward with a policy-**entropy** bonus, so the agent maximizes reward *and* stays as stochastic (exploratory) as possible:

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t)}\!\left[ r(s_t, a_t) + \alpha \, \mathcal{H}\big(\pi(\cdot \mid s_t)\big) \right]
$$

* The temperature $\alpha$ trades reward vs. exploration (and can be auto-tuned to hit a target entropy).
* SAC is **off-policy** (sample-efficient), **stochastic** (robust), and currently the default for continuous-control benchmarks and real-robot learning.

| Algorithm | Policy | On/Off-policy | Action space | Key idea |
| :--- | :--- | :--- | :--- | :--- |
| **PPO** | Stochastic | On-policy | Both | Clipped trust region |
| **DDPG** | Deterministic | Off-policy | Continuous | Differentiate through critic |
| **TD3** | Deterministic | Off-policy | Continuous | Twin critics + delays |
| **SAC** | Stochastic | Off-policy | Continuous | Max-entropy, auto-temperature |

---

## 11. Model-Based RL

Everything above is **model-free** — it learns values/policies from experience without ever modeling the environment. **Model-based RL** learns (or is given) a dynamics model $\hat{P}(s'|s,a)$ and reward model, then *plans*.

* **Dyna:** interleave real experience with **imagined** rollouts from the learned model to augment the replay buffer — dramatically improving sample efficiency.
* **MuZero:** learns a *latent* dynamics model sufficient to predict reward, value, and policy, and plans with **Monte Carlo Tree Search** in that latent space — mastering Go, chess, and Atari **without being told the rules**.
* **Dreamer / World Models:** learn a latent world model and train the actor-critic entirely inside "imagined" latent trajectories, enabling learning from very little real interaction.

**Trade-off:** model-based methods are far more **sample-efficient** (crucial when real data is expensive, e.g. robotics) but suffer **model bias** — planning against an inaccurate model compounds errors, so a wrong model yields a confidently wrong plan.

---

## 12. Offline (Batch) Reinforcement Learning

Standard RL requires online interaction — unacceptable in healthcare, autonomous driving, or recommendation, where exploratory mistakes are costly or dangerous. **Offline RL** learns a policy from a *fixed* dataset $D$ of previously logged transitions, with **no further environment interaction**.

### The Core Problem: Distributional Shift
The killer issue is **extrapolation error on out-of-distribution actions**. When the Bellman target computes $\max_{a'} Q(s', a')$, the max may select an action **never seen** in the data. The critic has no grounding there, typically **overestimates** it, and — because bootstrapping propagates that error — the policy learns to prefer fantasy actions that look great but were never validated. Online RL self-corrects (it tries the action and gets punished); offline RL cannot.

### Solutions — Stay Close to the Data
* **Policy constraint (BCQ, BEAR):** restrict the learned policy to actions near the behavior policy that generated the data.
* **Conservative Q-Learning (CQL):** add a regularizer that **pushes down** Q-values for OOD actions while keeping in-data actions high, yielding a lower bound on the true value:

$$
\min_Q \; \alpha \, \mathbb{E}_{s \sim D}\!\left[ \log \sum_a \exp Q(s,a) \; - \; \mathbb{E}_{a \sim \hat{\pi}_\beta}[Q(s,a)] \right] \; + \; \tfrac{1}{2}\,\mathbb{E}_{D}\big[(Q - \hat{\mathcal{B}} Q)^2\big]
$$

The first term suppresses over-optimistic OOD Q-values; the second is the usual TD error. This deliberate pessimism is the defining theme of offline RL.

---

## 13. RLHF — RL from Human Feedback

The most consequential modern application of RL is **aligning large language models**. The reward is not given by an environment — it is *learned from human preferences*. The pipeline (also detailed in the NLP guide) is:

1. **SFT:** supervised fine-tune a pretrained LM on demonstrations.
2. **Reward model:** collect human rankings of pairs of responses and train a reward model $R_\phi(x, y)$ with a Bradley–Terry loss $-\mathbb{E}[\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))]$.
3. **PPO optimization:** treat the LM as a policy $\pi_\theta(y|x)$ and optimize the learned reward, with a **KL penalty** anchoring it to the reference model so it does not "reward-hack" into gibberish:

$$
\max_\theta \; \mathbb{E}_{x, \, y \sim \pi_\theta}\big[ R_\phi(x, y) \big] - \beta \, D_{\text{KL}}\!\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

Here the RL concepts connect directly: the token-generating LM is the **policy**, the KL term is a **trust region** (as in PPO/TRPO), and reward hacking is exactly the **specification gaming** from §6. (Direct Preference Optimization, **DPO**, achieves the same alignment *without* an explicit RL loop by reparameterizing the reward into a classification loss — see the NLP guide.)

---

## 14. Additional Interview Questions

### Advanced Policy Methods

**Q11: Why subtract a baseline in policy gradients, and why doesn't it introduce bias?**
> **Answer:** REINFORCE is unbiased but very high variance because $G_t$ is a noisy sample. Subtracting a state-dependent baseline $b(s)$ reduces variance while leaving the gradient unbiased, because $\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s)\, b(s)] = b(s)\,\nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\,\nabla_\theta 1 = 0$. The variance-minimizing baseline is $V^\pi(s)$, which converts the return into the zero-centered **advantage** $A(s,a) = Q(s,a) - V(s)$.

**Q12: Explain PPO's clipped objective. What failure does the clip prevent?**
> **Answer:** With ratio $r_t(\theta) = \pi_\theta/\pi_{\theta_{\text{old}}}$, PPO maximizes $\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)$. The clip removes any gain from moving the policy ratio beyond $[1-\epsilon, 1+\epsilon]$, and the `min` makes the surrogate a pessimistic lower bound. This prevents a single large, destructive update — critical in RL because the policy generates its own future training data, so an over-large step can irreversibly collapse learning.

**Q13: You must learn a controller for a physical robot arm where each real trajectory is expensive and random exploration could damage hardware. Which algorithm family, and why?**
> **Answer:** Prefer **off-policy, sample-efficient** methods (**SAC** or **TD3**) over on-policy PPO, since they reuse a replay buffer and need far less data. If a decent logged dataset already exists and online exploration is unsafe, use **offline RL** (CQL) to learn purely from logs, and/or a **model-based** approach to plan in a learned dynamics model, minimizing costly real interactions. SAC's entropy bonus also gives smoother, safer exploration than DDPG's raw action noise.

### Continuous Control & Offline

**Q14: Why does DDPG overestimate Q-values, and how does TD3 fix it?**
> **Answer:** DDPG's actor is trained to maximize the critic, so any positive bias/noise in $Q$ is actively exploited and then bootstrapped, creating a self-reinforcing overestimation loop. **TD3** fixes this with (1) **clipped double-Q** — two critics, take the minimum for the target; (2) **delayed** actor/target updates so the policy chases a more stable value; and (3) **target policy smoothing** — noise on the target action to flatten exploitable Q peaks.

**Q15: What is the fundamental difficulty of offline RL that online RL does not face?**
> **Answer:** **Distributional shift / extrapolation error.** The Bellman max can query actions absent from the dataset; the critic overestimates these OOD actions and bootstrapping spreads the error, so the policy prefers unvalidated "fantasy" actions. Online RL self-corrects by actually trying them; offline RL cannot. Remedies enforce **pessimism/proximity**: constrain the policy to in-data actions (BCQ) or explicitly suppress OOD Q-values (CQL).

**Q16: How does RLHF map onto standard RL vocabulary?**
> **Answer:** The language model is the **policy** $\pi_\theta(y|x)$; the **reward** is a model $R_\phi$ learned from human preference rankings rather than supplied by an environment; PPO optimizes expected reward; and the **KL penalty** to the reference policy is a **trust region** preventing the policy from drifting into high-reward gibberish (**reward hacking / specification gaming**). It is ordinary policy-gradient RL with a learned, non-stationary-ish reward.