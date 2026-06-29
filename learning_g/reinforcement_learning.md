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