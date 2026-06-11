# Reinforcement Learning: A Comprehensive Learning Guide

## From First Principles to Advanced Algorithms

---

# Part I: MDP Fundamentals — States, Actions, Transitions, Rewards, and the Discount Factor

---

## 1. Motivation & Intuition

### Why Does Reinforcement Learning Exist?

Imagine you are teaching a dog to sit. You do not hand the dog a textbook on canine biomechanics. Instead, you wait for the dog to sit, then give it a treat. Over time, the dog learns that sitting leads to treats. This is the essence of reinforcement learning (RL): an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards.

Most of machine learning falls into two paradigms that do not capture this kind of learning:

- **Supervised learning** requires labeled input-output pairs. A teacher tells you the correct answer for every example. But in many real-world problems — playing a game of chess, driving a car, managing a supply chain — no one hands you a dataset of "correct moves." You must figure out what works by trying things.

- **Unsupervised learning** finds patterns in data without labels, but it has no notion of "goals" or "outcomes." It can cluster data or learn representations, but it cannot learn to *act* in a world.

Reinforcement learning fills this gap. It is the study of how an agent should take actions in an environment to maximize some notion of cumulative reward over time. The agent is not told which actions to take; it must discover which actions yield the most reward by trial and error.

**Real-world problems that are naturally RL problems:**

- A robot learning to walk. Nobody writes down the "correct" joint torques at each millisecond. The robot tries movements, falls, and gradually learns gaits that keep it upright and moving forward.
- An algorithm deciding which ad to show a user. Each ad shown is an action; the user's click (or lack thereof) is the reward. The algorithm must balance showing ads it knows work (exploitation) with trying new ads to learn about them (exploration).
- A thermostat controlling a building's HVAC system. At each time step, it decides whether to heat, cool, or do nothing. The reward reflects energy savings and occupant comfort. The effect of each decision unfolds over hours.

### Why Not Just Optimize Greedily?

A critical insight that separates RL from simpler optimization is the concept of **delayed reward**. The best action right now may not be the action that leads to the best long-term outcome. A chess player sacrifices a pawn (a locally bad outcome) to set up a checkmate in five moves (a globally excellent outcome). An RL agent must reason about the long-term consequences of its decisions.

This brings us to the mathematical framework that makes RL rigorous: the **Markov Decision Process (MDP)**.

---

## 2. Conceptual Foundations

### The Five Components of an MDP

An MDP is defined by a tuple **(S, A, P, R, γ)**. Let us build each piece from scratch.

#### States (S)

A **state** is a complete description of the situation the agent finds itself in at a given point in time. "Complete" means that the state contains all the information the agent needs to make a decision — no hidden variables matter.

- In a board game, the state is the configuration of all pieces on the board.
- In a robot, the state might be the positions and velocities of all joints, plus the positions of obstacles.
- In an inventory management system, the state is the current stock levels of all products, pending orders, and supplier lead times.

The set of all possible states is denoted **S**. This set can be finite (a grid world with 100 cells has |S| = 100) or infinite (a robot's joint angles are continuous).

#### Actions (A)

An **action** is a choice the agent can make. The set of all possible actions is denoted **A**. In some MDPs, the actions available depend on the current state, written A(s).

- In a grid world: {up, down, left, right}.
- In a stock trading system: {buy, sell, hold} for each stock, possibly with a quantity.
- In a game of Go: placing a stone on any empty intersection.

#### Transition Function (P)

The **transition function** (also called the dynamics or model) specifies what happens when the agent takes an action. Formally, P(s' | s, a) is the probability of ending up in state s' given that the agent is currently in state s and takes action a.

This is where the word "Markov" comes in. The **Markov property** states:

> The probability of transitioning to s' depends only on the current state s and action a, not on any previous states or actions.

In other words, the current state contains all relevant information about the past. This is a strong assumption. If it does not hold — for instance, if the outcome depends on the last three states, not just the current one — then you must either augment the state to include the relevant history, or use a different framework (like a Partially Observable MDP, or POMDP).

**Example:** In a grid world, if the agent is in cell (2, 3) and takes action "right," the transition function might say: with probability 0.8, the agent moves to (3, 3); with probability 0.1, it slips to (2, 4); with probability 0.1, it slips to (2, 2).

#### Reward Function (R)

The **reward function** R(s, a, s') gives a scalar signal after each transition. Sometimes written as R(s, a) or R(s) when the reward depends only on the state or state-action pair. The reward is the agent's sole source of feedback about how well it is doing.

Designing the reward function is one of the most critical and difficult parts of applying RL. The reward must incentivize the behavior you actually want. If a cleaning robot gets +1 for every piece of trash it picks up, it might learn to dump trash and pick it up again.

#### Discount Factor (γ)

The **discount factor** γ (gamma) is a number in [0, 1] that controls how much the agent cares about future rewards relative to immediate rewards.

- γ = 0: The agent is completely myopic. It only cares about the immediate reward.
- γ = 1: The agent weights all future rewards equally. This can cause problems in infinite-horizon settings because the sum of rewards may diverge to infinity.
- γ = 0.99: A common practical choice. The agent cares about future rewards, but more distant rewards are discounted exponentially. A reward received k steps in the future is worth γ^k times its face value today.

**Intuition for discounting:** Think of it like interest rates in finance. A dollar today is worth more than a dollar tomorrow because you could invest it. Similarly, a reward now is "worth more" than the same reward later, partly because the future is uncertain (the agent might not survive to collect it) and partly for mathematical convenience (it ensures sums converge).

### Policies, Value Functions, and the Agent's Goal

#### Policy (π)

A **policy** π tells the agent what to do in each state. It can be:

- **Deterministic:** π(s) = a. In state s, always take action a.
- **Stochastic:** π(a | s) = probability of taking action a in state s.

The agent's goal is to find an **optimal policy** π* that maximizes the expected cumulative discounted reward:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

This quantity G_t is called the **return** from time step t.

#### State-Value Function V^π(s)

The **state-value function** under policy π gives the expected return starting from state s and following π thereafter:

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

It answers: "How good is it to be in state s if I follow policy π?"

#### Action-Value Function Q^π(s, a)

The **action-value function** (or Q-function) gives the expected return starting from state s, taking action a, and following π thereafter:

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

It answers: "How good is it to take action a in state s, then follow policy π?"

### What Breaks When Assumptions Are Violated

| Assumption | What Breaks | Remedy |
|---|---|---|
| Markov property | The agent cannot predict transitions because relevant information is hidden. Q-values become unreliable. | Augment the state with history (stack frames, use recurrent nets), or use POMDPs. |
| Stationary dynamics | Transition probabilities change over time. Learned values become stale. | Use non-stationary RL methods, meta-learning, or continual learning. |
| Scalar reward | When objectives are multi-dimensional (safety AND efficiency AND fairness), collapsing them into a scalar can cause unexpected trade-offs. | Multi-objective RL, constrained MDPs, reward shaping with care. |
| Full observability | The agent sees a corrupted or partial view of the state. | POMDPs, belief states, recurrent architectures. |

---

## 3. Mathematical Formulation

### The Bellman Equation

The core recursive relationship in RL is the **Bellman equation**. It says that the value of a state equals the immediate reward plus the discounted value of the next state.

**Derivation for V^π:**

Starting from the definition:

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

Expand the return G_t:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s]$$

By linearity of expectation and the law of total expectation (conditioning on the action taken and the next state):

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

**Reading this equation:**

- For each action a the policy might take (weighted by π(a|s))...
- For each state s' we might land in (weighted by P(s'|s,a))...
- The value is the immediate reward R(s,a,s') plus the discounted future value γV^π(s').

**Bellman equation for Q^π:**

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a') \right]$$

### The Bellman Optimality Equation

For the optimal policy π*, the agent always picks the action that maximizes the Q-value:

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]$$

The optimal policy is then:

$$\pi^*(s) = \arg\max_a Q^*(s,a)$$

---

## 4. Worked Example

### A Simple Grid World

Consider a 3×3 grid. The agent starts at cell (0,0) and wants to reach the goal at (2,2). Cells are numbered 0 through 8 (row-major). Actions: {up, down, left, right}. Moving off the grid keeps the agent in place. Transitions are deterministic (no stochasticity).

**Rewards:** −1 for each step (to encourage reaching the goal quickly), +10 upon reaching (2,2), then the episode ends.

**Discount factor:** γ = 0.9.

**Computing V* for the goal state:** V*(goal) = 0 (terminal state; no future rewards).

**One step away from the goal:** Consider cell (2,1) (directly left of the goal). The optimal action is "right."

V*(2,1) = R + γ · V*(goal) = −1 + 0.9 · 10 = −1 + 9 = 8.0

Wait — we need to be careful. The +10 reward is received upon *arriving* at the goal. So:

V*(2,1) = R(2,1, right, goal) + γ · V*(goal) = (+10 − 1) + 0.9 · 0 = 9.0

Actually, let us be precise. If R = −1 for every step and an additional +10 for reaching the goal:

R(s, a, s') = −1 + 10 · 𝟙[s' = goal]

For cell (2,1), action right: R = −1 + 10 = 9, next state is terminal with V* = 0.

V*(2,1) = 9 + 0.9 · 0 = 9.

For cell (1,1), best action goes toward goal. Say action "right" leads to (2,1):

V*(1,1) = −1 + 0.9 · V*(2,1) = −1 + 0.9 · 9 = −1 + 8.1 = 7.1

For cell (0,0), the shortest path is 4 steps (right, right, down, down), but through the Bellman recursion:

V*(0,0) = −1 + 0.9 · V*(next) and so on. We can compute this iteratively using value iteration (covered next).

---

## 5. Relevance to Machine Learning Practice

### Where MDPs Appear in ML

**Recommendation systems** can be modeled as MDPs where states represent user profiles/sessions, actions are items to recommend, transitions model how user preferences evolve after seeing a recommendation, and rewards are clicks, purchases, or engagement metrics. The MDP framing captures the fact that recommendations affect future user behavior — showing too many similar items may bore the user.

**Neural architecture search (NAS):** The state is the partially constructed architecture, actions add layers or connections, and the reward is the validation accuracy of the trained architecture. Google used RL-based NAS to discover the architectures behind EfficientNet.

**Compiler optimization:** States represent intermediate representations of code, actions are optimization passes (loop unrolling, inlining, etc.), and rewards measure runtime or binary size. MLIR and LLVM have explored RL-based phase ordering.

**Dialogue systems:** States are conversation histories, actions are possible responses, and rewards come from user satisfaction scores or task completion indicators.

### When NOT to Use MDPs/RL

- When you have large amounts of labeled data and the problem is a standard prediction task (classification, regression) — use supervised learning.
- When the environment cannot be interacted with (only a static dataset) — consider offline RL or contextual bandits, but be cautious.
- When the state space is so large and the reward so sparse that exploration is infeasible.
- When the cost of exploration is too high (a self-driving car cannot "explore" by crashing into walls).

---

## 6. Common Pitfalls & Misconceptions

**Confusing states with observations.** In many practical settings, the agent does not observe the full state. Camera images from a self-driving car are observations, not states. The true state includes the positions and velocities of all nearby vehicles, pedestrian intentions, road conditions, and more. Treating observations as states violates the Markov property and can cause learning to fail. Mitigation: use frame stacking (concatenate several recent observations) or recurrent networks.

**Reward hacking.** The agent optimizes the reward you give it, not the reward you intend. A famous example: a boat-racing agent learned to go in circles collecting small bonuses rather than finishing the race. Always test learned policies qualitatively and consider reward shaping carefully.

**Ignoring the discount factor's effect on behavior.** A low γ (e.g., 0.5) makes the agent very short-sighted; it may avoid long detours even if they lead to much larger rewards. A high γ (e.g., 0.999) makes learning slower because the agent must propagate reward information over many steps. Choose γ based on the effective planning horizon of your problem.

**Assuming deterministic transitions.** Even in seemingly deterministic environments (like board games against an opponent), the opponent's actions introduce stochasticity from the agent's perspective. Always model transitions probabilistically when there is any uncertainty.

**Conflating return with reward.** The reward is the immediate signal; the return is the cumulative discounted sum. Optimizing for immediate reward (a greedy strategy) is not the same as optimizing for return. This distinction is the entire reason RL is interesting.

---

# Part II: Tabular Methods

---

# II-A: Dynamic Programming — Policy Iteration and Value Iteration

---

## 1. Motivation & Intuition

### When You Know Everything About the World

Imagine you have a complete map of a building, including every room, every door, and the probability that each door is locked. You want to find the fastest route from the entrance to the exit. Because you know the full layout, you can compute the optimal path without ever stepping inside the building.

**Dynamic programming (DP)** methods in RL occupy this same privileged position. They assume you have complete knowledge of the MDP — that is, you know the full transition function P(s' | s, a) and the reward function R(s, a, s'). Given this knowledge, DP computes optimal policies by systematically solving the Bellman equations.

This may seem unrealistic — and in many practical RL problems, it is. But DP is foundational for two reasons:

1. **It provides the theoretical benchmark.** DP gives the provably optimal solution. All other RL methods (Monte Carlo, TD learning, deep RL) are approximations that try to get close to what DP would give if the model were known.
2. **It introduces the key algorithms.** The ideas of iteratively improving value estimates and greedily improving policies appear in every RL method, even those that do not assume a known model.

### The Two Big Questions

DP methods answer two related questions:

- **Policy Evaluation:** Given a fixed policy π, what is V^π(s) for every state? (How good is this policy?)
- **Policy Improvement:** Given V^π, can we find a better policy? (Can we do better?)

Combining these two steps gives us two complete algorithms: **Policy Iteration** and **Value Iteration**.

---

## 2. Conceptual Foundations

### Policy Evaluation (Prediction)

Given a policy π, we want to compute V^π(s) for all s. The Bellman equation for V^π is:

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

This is a system of |S| linear equations in |S| unknowns. For small state spaces, you could solve it directly by matrix inversion. But for large state spaces, **iterative policy evaluation** is more practical.

**Algorithm: Iterative Policy Evaluation**

1. Initialize V(s) = 0 for all s (or any arbitrary values; terminal states always have V = 0).
2. Repeat until convergence:
   - For each state s:
     - V_new(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V_old(s')]
   - Check if max_s |V_new(s) − V_old(s)| < θ for some small threshold θ.
   - Set V_old = V_new.

This is a contraction mapping (because γ < 1), so it is guaranteed to converge to V^π.

### Policy Improvement

Once we have V^π, we can ask: "Is there a better policy?" The **policy improvement theorem** says that if we construct a new policy π' by acting greedily with respect to V^π:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

then π' is at least as good as π in every state. If π' ≠ π, then π' is strictly better in at least one state.

**Intuition:** If we can find even one state where a different action would lead to a higher expected return (according to the current value function), then switching to that action improves the policy.

### Policy Iteration

Policy iteration alternates between evaluation and improvement:

1. **Initialize** π arbitrarily.
2. **Policy Evaluation:** Compute V^π (by iterating the Bellman equation until convergence).
3. **Policy Improvement:** Construct π' by acting greedily w.r.t. V^π.
4. If π' = π, stop — we have found the optimal policy. Otherwise, set π = π' and go to step 2.

**Convergence guarantee:** The number of deterministic policies is finite (|A|^|S|), and each iteration produces a strictly better policy (unless we've reached the optimum). Therefore, policy iteration terminates in a finite number of steps. In practice, it converges very quickly — often in a handful of iterations.

### Value Iteration

Value iteration is a streamlined approach that combines evaluation and improvement into a single update:

1. **Initialize** V(s) = 0 for all s.
2. Repeat until convergence:
   - For each state s:
     - V_new(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
3. Extract the optimal policy: π*(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V*(s')]

**Key insight:** Value iteration is equivalent to doing only *one sweep* of policy evaluation before policy improvement at each iteration, and it directly iterates the Bellman optimality equation.

**Why does this work?** The Bellman optimality operator is also a contraction mapping with fixed point V*. Each application of the operator brings V closer to V*.

---

## 3. Mathematical Formulation

### Policy Evaluation as a Linear System

Write the Bellman equation for all states simultaneously:

$$\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma \mathbf{P}^\pi \mathbf{v}^\pi$$

where:
- **v**^π is the vector of state values [V^π(s₁), V^π(s₂), ...]ᵀ
- **r**^π is the vector of expected immediate rewards under π
- **P**^π is the state-transition matrix under π, where P^π_{ss'} = Σ_a π(a|s) P(s'|s,a)

Solving directly:

$$\mathbf{v}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{r}^\pi$$

This costs O(|S|³) due to matrix inversion, which is feasible for |S| up to a few thousand but impractical for larger problems.

### Contraction Mapping Argument

Define the Bellman operator T^π:

$$(T^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]$$

**Claim:** T^π is a γ-contraction in the sup-norm: ||T^π V₁ − T^π V₂||_∞ ≤ γ ||V₁ − V₂||_∞.

**Proof sketch:**

|(T^π V₁)(s) − (T^π V₂)(s)| = |Σ_a π(a|s) Σ_{s'} P(s'|s,a) γ [V₁(s') − V₂(s')]|
≤ Σ_a π(a|s) Σ_{s'} P(s'|s,a) γ |V₁(s') − V₂(s')|
≤ γ ||V₁ − V₂||_∞ · Σ_a π(a|s) Σ_{s'} P(s'|s,a)
= γ ||V₁ − V₂||_∞

The last step uses the fact that the probabilities sum to 1. Taking the max over s gives the result. By the Banach fixed-point theorem, T^π has a unique fixed point (V^π) and iterative application converges to it.

### Value Iteration Convergence Rate

After k iterations of value iteration:

$$||V_k - V^*||_\infty \leq \gamma^k ||V_0 - V^*||_\infty$$

This means the error decreases geometrically. For γ = 0.99, you need about 460 iterations to reduce the error by a factor of 100 (since 0.99^460 ≈ 0.01). For γ = 0.9, you need only 44 iterations for the same factor.

---

## 4. Worked Example

### Policy Iteration on a Small Grid

**Setup:** 4×1 grid. States: {s₀, s₁, s₂, s₃}. s₃ is the terminal/goal state. Actions: {left, right}. Transitions are deterministic. Moving left from s₀ stays at s₀. R = −1 for every step, R = +5 for reaching s₃. γ = 0.9.

**Step 1: Initialize π(s) = left for all non-terminal states.** (A bad policy: the agent always moves left.)

**Step 2: Policy Evaluation.**

Under this policy, s₁ → s₀ → s₀ → s₀ → ... (stuck forever).

V^π(s₃) = 0 (terminal)
V^π(s₀): Agent stays at s₀ forever, getting −1 each step.
V^π(s₀) = −1 + 0.9·(−1) + 0.81·(−1) + ... = −1/(1−0.9) = −10

V^π(s₁): Agent moves to s₀, then stays there.
V^π(s₁) = −1 + 0.9·V^π(s₀) = −1 + 0.9·(−10) = −10

V^π(s₂): Agent moves to s₁, then s₀, then stuck.
V^π(s₂) = −1 + 0.9·V^π(s₁) = −1 + 0.9·(−10) = −10

**Step 3: Policy Improvement.** For each state, check which action gives higher value.

For s₀:
- Left: −1 + 0.9·V^π(s₀) = −1 + 0.9·(−10) = −10 (same)
- Right: −1 + 0.9·V^π(s₁) = −1 + 0.9·(−10) = −10 (same)

For s₁:
- Left: −1 + 0.9·V^π(s₀) = −10
- Right: −1 + 0.9·V^π(s₂) = −1 + 0.9·(−10) = −10

For s₂:
- Left: −1 + 0.9·V^π(s₁) = −10
- Right: −1 + 5 + 0.9·0 = 4  ← **Better!**

New policy: π'(s₂) = right. Now re-evaluate.

**Step 4: Policy Evaluation with updated π'.**

V^π'(s₂) = (−1 + 5) + 0.9·0 = 4
V^π'(s₁): Now check — with old policy, π'(s₁) = left, so V^π'(s₁) = −1 + 0.9·V^π'(s₀) = −10 still.

**Step 5: Policy Improvement again.**

For s₁:
- Left: −1 + 0.9·(−10) = −10
- Right: −1 + 0.9·4 = −1 + 3.6 = 2.6 ← **Better!**

Update: π''(s₁) = right.

Continue iterating. After a few rounds, the optimal policy becomes π*(s₀) = right, π*(s₁) = right, π*(s₂) = right, with:

V*(s₂) = 4, V*(s₁) = −1 + 0.9·4 = 2.6, V*(s₀) = −1 + 0.9·2.6 = 1.34.

---

## 5. Relevance to Machine Learning Practice

### When DP Is Used in Practice

**Planning in known environments:** When you have a simulator (e.g., a game engine, a physics simulator, a traffic model), you effectively know the transition function. DP or DP-like methods can be applied within the simulator. AlphaGo used a form of planning (Monte Carlo Tree Search) that is closely related to DP.

**As a subroutine in model-based RL:** In model-based RL, the agent first learns a model of the environment (estimates P and R from data), then applies DP-like planning within the learned model. Dyna-Q, for example, alternates between learning from real experience and planning with a learned model.

**Approximate dynamic programming (ADP):** In operations research and control, ADP methods use function approximation to apply DP ideas to large or continuous state spaces. This bridges the gap between tabular DP and deep RL.

### Policy Iteration vs. Value Iteration: Trade-offs

| Aspect | Policy Iteration | Value Iteration |
|---|---|---|
| Per-iteration cost | High (full policy evaluation) | Low (one Bellman backup per state) |
| Number of iterations | Few (often 3–10) | Many (depends on γ) |
| Total cost | Often lower for small/medium MDPs | Often lower for large MDPs with high γ |
| Memory | Stores V and π | Stores only V |

---

## 6. Common Pitfalls & Misconceptions

**Assuming DP applies without a model.** DP requires the full transition and reward functions. If you do not know P(s'|s,a), you cannot perform the Bellman backup. In model-free settings, use Monte Carlo or TD methods instead.

**Confusing policy evaluation with value iteration.** Policy evaluation computes V for a *fixed* policy. Value iteration computes V* by taking the max over actions. They are different algorithms solving different equations.

**Not initializing terminal state values to zero.** If terminal states are not properly handled, the Bellman updates will propagate incorrect values. Always set V(terminal) = 0 and never update terminal states.

**Using γ = 1 in infinite-horizon problems.** With γ = 1 and infinite horizons, the return can diverge. DP convergence guarantees rely on γ < 1 (or episodic tasks with guaranteed termination).

**Synchronous vs. asynchronous updates.** The standard presentation uses synchronous updates (update all states, then move to the next iteration). In practice, asynchronous updates (update states one at a time, using the latest values) often converge faster and are sometimes called "in-place" DP. Be aware of which version you are implementing.

---

# II-B: Monte Carlo Methods — First-Visit and Every-Visit MC

---

## 1. Motivation & Intuition

### Learning from Complete Experiences

Suppose you want to learn how to play blackjack. You do not know the exact probabilities of drawing each card (the "transition function"). Instead, you simply play many hands and observe the outcomes.

After each hand, you know:
- The sequence of states you visited (your hand total, the dealer's showing card).
- The actions you took (hit or stand).
- Whether you won or lost (the reward).

By averaging the outcomes of hands where you were in a particular situation, you can estimate how good that situation is. This is the essence of **Monte Carlo (MC) methods**: learn value functions by averaging the returns observed after visiting states across many complete episodes.

**Key distinction from DP:** MC methods do not require a model of the environment. They learn directly from experience (episodes of interaction). However, they do require episodes to terminate — you must reach the end of an episode before you can compute the return and update your estimates.

### Why "Monte Carlo"?

The name comes from the famous casino in Monaco. Monte Carlo methods, broadly, refer to any technique that uses random sampling to estimate a quantity. Here, we randomly sample episodes (by following a policy and letting the environment respond) and use the observed returns to estimate value functions.

---

## 2. Conceptual Foundations

### Episodes and Returns

An **episode** is a sequence of states, actions, and rewards that terminates:

s₀, a₀, r₁, s₁, a₁, r₂, ..., s_{T-1}, a_{T-1}, r_T, s_T (terminal)

The **return** from time step t is:

G_t = r_{t+1} + γ r_{t+2} + ... + γ^{T-t-1} r_T

MC methods estimate V^π(s) or Q^π(s,a) by averaging these returns.

### First-Visit MC vs. Every-Visit MC

A state s might be visited multiple times within a single episode. The two variants handle this differently:

**First-Visit MC:** Only the first time s is visited in an episode counts. Compute the return from that first visit, and average it with returns from first visits in other episodes.

**Every-Visit MC:** Every time s is visited in an episode contributes a return estimate. If s is visited at time steps 3, 7, and 12, all three returns (G₃, G₇, G₁₂) are averaged.

**Which is better?** Both converge to V^π(s) as the number of episodes goes to infinity. First-visit MC produces unbiased estimates with independent samples. Every-visit MC uses more data per episode (which can reduce variance) but its samples within an episode are correlated (which complicates the analysis, though it is still consistent). In practice, they perform similarly.

### The Exploration Problem

MC methods estimate Q^π(s,a) by observing what happens after taking action a in state s. But if the policy π never takes action a in state s, the method has no data for Q(s,a). This is the **exploration problem**.

**Solution 1: Exploring Starts.** Guarantee that every state-action pair has a nonzero probability of being the starting point of an episode. This works in simulation but is impractical in most real-world scenarios (you cannot force a self-driving car to start in an arbitrary dangerous state).

**Solution 2: ε-soft policies.** Use a policy that assigns at least ε/|A| probability to every action. The ε-greedy policy is the most common: with probability 1−ε, take the greedy action; with probability ε, take a random action.

### On-Policy vs. Off-Policy MC

**On-policy:** Learn about the policy you are currently following. Use ε-greedy exploration during learning; the learned value function is for the ε-greedy policy, not the optimal greedy policy.

**Off-policy:** Learn about a different policy (the "target" policy) while following another policy (the "behavior" policy). This is more powerful but requires **importance sampling** to correct for the difference between the behavior and target policies.

**Importance sampling ratio:** If we follow behavior policy b but want to estimate values under target policy π:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

The return G_t is then weighted by this ratio:

$$V^\pi(s) \approx \frac{\sum_{\text{episodes}} \rho \cdot G_t}{\sum_{\text{episodes}} \rho}$$ (weighted importance sampling)

or

$$V^\pi(s) \approx \frac{1}{N} \sum_{\text{episodes}} \rho \cdot G_t$$ (ordinary importance sampling)

Weighted importance sampling is biased but has lower variance and is generally preferred.

---

## 3. Mathematical Formulation

### First-Visit MC Prediction

Let N(s) be a counter and S_total(s) be a running sum.

For each episode:
1. Generate episode following π.
2. Compute returns G_t for each t.
3. For each state s appearing in the episode:
   - If this is the first visit to s in this episode:
     - N(s) ← N(s) + 1
     - S_total(s) ← S_total(s) + G_t
4. V(s) = S_total(s) / N(s)

**Incremental update form** (avoids storing all returns):

$$V(s) \leftarrow V(s) + \frac{1}{N(s)} [G_t - V(s)]$$

This has the form: new_estimate = old_estimate + step_size × (target − old_estimate).

More generally, we can use a constant step size α:

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

A constant α gives more weight to recent episodes, which is useful in non-stationary environments.

### MC Control (Learning Optimal Policies)

To learn not just V^π but the optimal policy, we use MC to estimate Q(s,a) and then improve the policy greedily.

**On-policy MC Control with ε-greedy:**

1. Initialize Q(s,a) arbitrarily, set π to be ε-greedy w.r.t. Q.
2. For each episode:
   a. Generate episode following π.
   b. For each (s,a) pair in the episode:
      - Compute G_t.
      - N(s,a) ← N(s,a) + 1.
      - Q(s,a) ← Q(s,a) + (1/N(s,a))[G_t − Q(s,a)].
   c. Update π to be ε-greedy w.r.t. updated Q.

**Convergence:** Under mild conditions (all state-action pairs visited infinitely often, step size conditions), on-policy MC control converges to the optimal ε-soft policy.

### Variance of MC Estimates

The variance of the return G_t can be high because it depends on the entire trajectory (all actions and transitions for the rest of the episode). A single unlucky transition can dramatically change G_t. This high variance is MC's main weakness compared to TD methods.

For an episode of length T with reward variance σ² at each step:

Var(G_t) ≈ σ² · (1 + γ² + γ⁴ + ... + γ^{2(T-t-1)}) = σ² · (1 − γ^{2(T-t)}) / (1 − γ²)

Longer episodes and higher γ both increase variance.

---

## 4. Worked Example

### Blackjack (Simplified)

**Setup:** The player's state is (hand_sum, dealer_showing, usable_ace). The player can "hit" (draw a card) or "stand" (stop). The player wins if their sum is closer to 21 than the dealer's without exceeding 21.

Consider a simple policy π: stand if hand_sum ≥ 20, else hit.

**Episode 1:** State (14, 10, no_ace), action hit → State (22, 10, no_ace), bust. Reward = −1. Return G = −1.

**Episode 2:** State (14, 10, no_ace), action hit → State (18, 10, no_ace), action stand → Dealer busts → Reward = +1. Return from state (14, 10, no_ace) = 0 + γ·(+1) = 0.9.

**Episode 3:** State (14, 10, no_ace), action hit → State (21, 10, no_ace), action stand → Win. Return from (14, 10, no_ace) = 0 + γ·(+1) = 0.9.

After 3 episodes, V((14, 10, no_ace)) ≈ (−1 + 0.9 + 0.9)/3 = 0.267.

After 10,000 episodes, this estimate will be much more accurate and will converge to the true V^π.

---

## 5. Relevance to Machine Learning Practice

### Strengths of MC Methods

**No model required.** MC methods need only the ability to generate episodes. This makes them applicable to any environment where you can simulate or interact, even if the dynamics are unknown.

**Unbiased estimates.** First-visit MC provides unbiased estimates of V^π(s). Unlike TD methods (covered next), MC does not bootstrap — it uses actual observed returns, not estimates of future values.

**Natural for episodic tasks.** MC is well-suited for problems with clear episode boundaries: games, clinical trials, advertising campaigns, batch manufacturing processes.

### Weaknesses

**High variance.** Returns depend on many stochastic events. You may need millions of episodes for accurate estimates.

**Requires complete episodes.** MC cannot learn from incomplete episodes. If episodes are very long (or infinite), MC is impractical. You cannot apply MC to continuing (non-episodic) tasks without modification.

**Slow credit assignment.** If a critical decision occurs early in a long episode, its effect is buried in a long sequence of subsequent events, making it hard to isolate.

### MC in Modern ML

Monte Carlo estimation is used inside many advanced algorithms:

- **REINFORCE** (a policy gradient method) uses MC returns to estimate the gradient.
- **Monte Carlo Tree Search (MCTS)** in AlphaGo uses random rollouts to estimate the value of game positions.
- **MC evaluation** is used to benchmark TD-based and model-based methods.

---

## 6. Common Pitfalls & Misconceptions

**Forgetting that MC only works for episodic tasks.** If episodes never end, you never get a return to average. For continuing tasks, use TD methods.

**Not exploring enough.** If your policy is deterministic, many state-action pairs will never be visited, and Q(s,a) for those pairs will remain at their arbitrary initial values. Always use ε-soft exploration.

**Confusing the first-visit and every-visit distinction.** When implementing first-visit MC, you must track which states have already been visited in the current episode. A common bug is to accidentally treat every visit as a first visit (or vice versa).

**High variance with importance sampling.** Off-policy MC with ordinary importance sampling can have extremely high (even infinite) variance due to products of importance weights. Weighted importance sampling helps, but the variance can still be problematic with long episodes. Consider per-decision importance sampling or truncation.

**Not using enough episodes.** MC estimates converge slowly. Convergence goes as O(1/√N), so reducing the error by 10x requires 100x more episodes.

---

# II-C: Temporal Difference Learning — SARSA and Q-Learning

---

## 1. Motivation & Intuition

### Learning Without Waiting for the End

Imagine you are navigating a maze. With Monte Carlo methods, you would have to reach the exit (or a dead end) before updating your estimates of how good each position in the maze is. If the maze is enormous, you might wander for hours before getting any learning signal.

What if, instead, you could update your estimates *at every single step*? After moving from one room to the next, you immediately revise your estimate of how good the previous room was, based on the reward you just received and your current estimate of how good the new room is.

This is the key idea behind **Temporal Difference (TD) learning:** update estimates based on other estimates, without waiting for the final outcome. This is called **bootstrapping**.

**TD combines the best of DP and MC:**

| Feature | DP | MC | TD |
|---|---|---|---|
| Requires model? | Yes | No | No |
| Learns from incomplete episodes? | N/A | No | Yes |
| Bootstraps? | Yes | No | Yes |
| Guaranteed unbiased? | Yes (exact) | Yes | No (biased by bootstrap) |

### The Fundamental TD Idea

After taking action a in state s, receiving reward r, and transitioning to state s', the TD update is:

$$V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]$$

Compare with the MC update:

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

MC uses the actual return G_t (which requires the complete episode). TD uses the **TD target** r + γV(s'), which uses the current estimate V(s') as a stand-in for the future return. The quantity δ = r + γV(s') − V(s) is called the **TD error**.

---

## 2. Conceptual Foundations

### TD(0): The Simplest TD Method

TD(0) updates the value function after every single step. The "0" refers to looking ahead only one step (as opposed to TD(λ), which looks ahead multiple steps, discussed later).

**Algorithm:**

```
Initialize V(s) for all s
For each episode:
    Initialize s
    For each step:
        Choose action a using policy π
        Take action a, observe reward r and next state s'
        V(s) ← V(s) + α [r + γ V(s') − V(s)]
        s ← s'
    Until s is terminal
```

### SARSA: On-Policy TD Control

**SARSA** (State-Action-Reward-State-Action) is the TD control method for learning Q(s,a). The name comes from the quintuple of values used in each update: (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}).

**Update rule:**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Key point:** The next action A_{t+1} is chosen by the *same policy* being evaluated (e.g., ε-greedy). This makes SARSA an **on-policy** method — it learns about the policy it is currently following.

**Algorithm:**

```
Initialize Q(s,a) for all s, a
For each episode:
    Initialize s
    Choose a from s using policy derived from Q (e.g., ε-greedy)
    For each step:
        Take action a, observe r, s'
        Choose a' from s' using policy derived from Q (e.g., ε-greedy)
        Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') − Q(s,a)]
        s ← s', a ← a'
    Until s is terminal
```

**Behavior:** Because SARSA learns about the ε-greedy policy (which sometimes takes random actions), it tends to learn safer, more conservative policies in environments with penalties. It accounts for the fact that it will occasionally make mistakes due to exploration.

### Q-Learning: Off-Policy TD Control

**Q-learning** modifies the SARSA update to learn the *optimal* Q-function directly, regardless of the exploration policy.

**Update rule:**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]$$

**Key difference from SARSA:** The update uses max_{a'} Q(s', a') instead of Q(s', a') where a' is the action actually taken. This means Q-learning always aims for the optimal policy, even while following an exploratory policy.

**Algorithm:**

```
Initialize Q(s,a) for all s, a
For each episode:
    Initialize s
    For each step:
        Choose a from s using policy derived from Q (e.g., ε-greedy)
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
        s ← s'
    Until s is terminal
```

### SARSA vs. Q-Learning: The Cliff Walking Example

Consider a grid world with a cliff along the bottom row. The agent starts at the bottom-left and must reach the bottom-right. Stepping into the cliff gives −100 reward and sends the agent back to the start. Each step costs −1.

**Q-learning** finds the optimal path: walk along the bottom, right next to the cliff. This is the shortest path, and the optimal policy (which is deterministic) never falls off.

**SARSA** learns a longer but safer path along the top of the grid, because it accounts for the fact that ε-greedy exploration occasionally takes random actions. Near the cliff, a random action could be catastrophic, so SARSA learns to avoid the cliff edge.

**Which is better?** It depends on the context:
- If you will *deploy* the greedy policy (with ε = 0), Q-learning learns the right thing.
- If you must continue exploring during deployment, SARSA gives a better policy for the actual behavior.

---

## 3. Mathematical Formulation

### Convergence of Q-Learning

**Theorem (Watkins & Dayan, 1992):** Q-learning converges to Q* with probability 1, provided:

1. All state-action pairs are visited infinitely often.
2. The learning rate satisfies: Σ_t α_t = ∞ and Σ_t α_t² < ∞.
3. The rewards are bounded.

The standard schedule α_t = 1/N_t(s,a) (where N_t counts visits) satisfies these conditions.

**Proof sketch:** Q-learning can be viewed as stochastic approximation of the Bellman optimality operator. The Bellman operator is a contraction, and the stochastic updates are noisy but unbiased (w.r.t. the contraction). The Robbins-Monro conditions on the step size ensure that the noise averages out over time while the signal accumulates.

### TD Error and Its Properties

The TD error at time t is:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Key property:** Under the true value function V^π, the expected TD error is zero:

$$\mathbb{E}_\pi[\delta_t | S_t = s] = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')] - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0$$

This is exactly the Bellman equation. So V^π is a fixed point of the TD update in expectation.

### Bias-Variance Trade-off: MC vs. TD

**MC target:** G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
- Unbiased (E[G_t | S_t = s] = V^π(s)).
- High variance (depends on many random variables).

**TD(0) target:** R_{t+1} + γV(S_{t+1})
- Biased (because V(S_{t+1}) is an estimate, not the true value).
- Low variance (depends on only one step of randomness).

The bias decreases as V approaches V^π. In practice, the lower variance of TD often leads to faster learning, despite the bias.

### Expected SARSA

A useful intermediate between SARSA and Q-learning is **Expected SARSA:**

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \sum_{a'} \pi(a'|s') Q(s',a') - Q(s,a)]$$

Instead of sampling the next action (SARSA) or taking the max (Q-learning), Expected SARSA takes the expectation over the next action under the current policy. This reduces variance compared to SARSA while maintaining the on-policy character. With a greedy target policy, Expected SARSA becomes Q-learning.

---

## 4. Worked Example

### Q-Learning on a 3-State MDP

**States:** {s₀, s₁, s₂ (terminal)}. **Actions:** {left, right}. **Transitions:**
- From s₀: right → s₁ (always), left → s₀ (stay).
- From s₁: right → s₂ (always), left → s₀ (always).

**Rewards:** R = 0 everywhere except R = +1 when transitioning to s₂. **γ = 0.9, α = 0.1.**

**Initialize Q(s,a) = 0 for all pairs.**

**Episode 1:**

Step 1: s = s₀, ε-greedy picks a = right (random). Observe r = 0, s' = s₁.
Q(s₀, right) ← 0 + 0.1 · [0 + 0.9 · max(Q(s₁, left), Q(s₁, right)) − 0]
= 0 + 0.1 · [0 + 0.9 · 0 − 0] = 0

Step 2: s = s₁, ε-greedy picks a = right. Observe r = 1, s' = s₂ (terminal).
Q(s₁, right) ← 0 + 0.1 · [1 + 0.9 · 0 − 0] = 0.1

**Episode 2:**

Step 1: s = s₀, a = right. r = 0, s' = s₁.
Q(s₀, right) ← 0 + 0.1 · [0 + 0.9 · max(0, 0.1) − 0] = 0 + 0.1 · 0.09 = 0.009

Step 2: s = s₁, a = right. r = 1, s' = s₂.
Q(s₁, right) ← 0.1 + 0.1 · [1 + 0 − 0.1] = 0.1 + 0.09 = 0.19

**After many episodes:** Q(s₁, right) → 1.0, Q(s₀, right) → 0.9 (the true Q*).

---

## 5. Relevance to Machine Learning Practice

### TD Learning in Practice

**Temporal difference methods are the workhorse of RL.** Most practical RL algorithms — including DQN, A3C, PPO, SAC — use some form of TD learning.

**Where TD shines:**
- Continuing (non-episodic) tasks where MC cannot be applied.
- Tasks with long episodes where MC would have high variance.
- Online learning where you want to update parameters after every step.
- When combined with function approximation (neural networks), leading to deep RL.

**Real applications:**
- TD-Gammon (Tesauro, 1995) used TD(λ) to learn a world-class backgammon player.
- Recommendation systems use TD-based methods to model long-term user engagement.
- Robotic control often uses Q-learning or its variants with function approximation.

### SARSA vs. Q-Learning in Practice

Use **SARSA** when:
- The exploration policy will also be used during deployment (safety-critical applications).
- You want the agent to account for its own exploration noise.
- The environment has dangerous states near optimal paths.

Use **Q-learning** when:
- You will deploy a greedy (deterministic) policy after training.
- You want to separate the exploration strategy from the learned policy.
- Off-policy learning is desirable (e.g., learning from demonstration data).

---

## 6. Common Pitfalls & Misconceptions

**Choosing the learning rate α incorrectly.** Too large and the estimates oscillate; too small and learning is glacially slow. A constant α (e.g., 0.1) works well in practice for non-stationary environments but violates the theoretical convergence conditions. A decaying α (e.g., 1/N(s,a)) converges but adapts slowly. Practical advice: start with α ∈ [0.01, 0.5] and tune.

**Assuming Q-learning is always better than SARSA.** Q-learning learns the optimal policy but can overestimate Q-values (a problem called the **maximization bias**). SARSA does not suffer from this. Double Q-learning addresses the maximization bias in Q-learning.

**Forgetting that TD(0) is biased.** Early in training, V(s') is a poor estimate, so the TD target r + γV(s') is inaccurate. This bias can persist if the function approximator (in the deep RL setting) cannot represent V^π well. MC avoids this bias but at the cost of higher variance.

**Conflating "off-policy" with "uses a replay buffer."** Off-policy means the policy being learned (target policy) differs from the policy generating data (behavior policy). A replay buffer is one way to achieve off-policy learning, but the term "off-policy" is about the policy mismatch, not the presence of a buffer.

**Not handling terminal states correctly.** When s' is terminal, the TD target is just r (no future value). A common bug is to include γV(s') for terminal states, which adds spurious value.

---

# Part III: Function Approximation

---

# III-A: Deep Q-Networks — Experience Replay and Target Networks

---

## 1. Motivation & Intuition

### Why Tables Are Not Enough

Everything we have discussed so far uses tables: one entry for V(s) per state, or one entry for Q(s,a) per state-action pair. This works beautifully when the state space is small (a few hundred or thousand states). But consider these real problems:

- **Atari games:** The state is a 210 × 160 pixel image with 128 possible colors per pixel. The number of possible states is approximately 128^(210×160) — a number so large it dwarfs the number of atoms in the universe.
- **Robotics:** Joint angles and velocities are continuous. The state space is infinite.
- **Go:** There are approximately 2.08 × 10^170 legal board positions.

For these problems, we cannot allocate a table entry for every state. Instead, we need **function approximation**: represent V(s) or Q(s,a) as a parameterized function (like a neural network) that generalizes across similar states.

### The Deep Q-Network (DQN) Breakthrough

In 2013-2015, DeepMind demonstrated that a neural network could learn to play Atari games at superhuman levels directly from pixel inputs. The algorithm, **DQN (Deep Q-Network)**, combined Q-learning with two critical innovations that stabilized training:

1. **Experience replay**: Store past transitions in a buffer and train on random minibatches, breaking the correlation between consecutive training samples.
2. **Target networks**: Use a slowly-updated copy of the network to compute TD targets, preventing the "moving target" problem.

Before these innovations, combining neural networks with Q-learning was notoriously unstable. DQN made it work reliably and reproducibly.

---

## 2. Conceptual Foundations

### Q-Learning with Function Approximation

Instead of a table Q(s,a), we use a neural network Q(s, a; θ) parameterized by weights θ. Given a state s (e.g., a stack of game frames), the network outputs Q-values for all actions simultaneously.

**The loss function:** For a transition (s, a, r, s'), the TD target is:

y = r + γ max_{a'} Q(s', a'; θ)

We want Q(s, a; θ) to match this target. The loss is:

L(θ) = (y − Q(s, a; θ))²

We minimize this loss by gradient descent:

θ ← θ − α ∇_θ L(θ) = θ + α (y − Q(s, a; θ)) ∇_θ Q(s, a; θ)

### Problem 1: Correlated Samples

In online Q-learning, the agent collects data sequentially: s₁, a₁, r₂, s₂, a₂, r₃, s₃, ... Consecutive transitions are highly correlated (s₂ is related to s₁ because the agent was just there). Neural networks trained on correlated sequential data tend to:

- Overfit to recent experiences, forgetting what they learned earlier.
- Get trapped in local optima specific to the current part of the state space.
- Exhibit oscillating or diverging parameters.

Standard supervised learning assumes that training examples are independent and identically distributed (i.i.d.). Sequential RL data violates this assumption badly.

**Solution: Experience Replay**

Maintain a **replay buffer** D of capacity N, storing transitions (s, a, r, s', done). At each step:

1. Store the latest transition in D (overwriting the oldest if full).
2. Sample a random minibatch of k transitions from D.
3. Compute the loss and update θ using the minibatch.

**Why this works:**
- Random sampling breaks the temporal correlation between training samples, approximating the i.i.d. assumption.
- Each transition can be reused many times, improving data efficiency.
- The agent does not forget earlier experiences because old transitions persist in the buffer.

**Trade-off:** The buffer stores old transitions generated by earlier (and different) policies. This makes experience replay an **off-policy** method. Q-learning is off-policy by nature, so this is compatible. On-policy methods like SARSA cannot use a replay buffer directly.

### Problem 2: Moving Targets

In standard supervised learning, the target y does not depend on the model parameters θ. In Q-learning, the target *does* depend on θ:

y = r + γ max_{a'} Q(s', a'; θ)

So when we update θ to make Q(s,a;θ) closer to y, we also *change* y. We are chasing a moving target. This creates a feedback loop that can cause oscillations or divergence.

**Solution: Target Network**

Maintain a separate **target network** Q(s, a; θ⁻) with frozen parameters θ⁻. The target becomes:

y = r + γ max_{a'} Q(s', a'; θ⁻)

Now y does not change with every gradient step. The target network is updated periodically:

- **Hard update:** Every C steps, set θ⁻ ← θ (copy the current weights).
- **Soft update (Polyak averaging):** At every step, θ⁻ ← τθ + (1−τ)θ⁻ for some small τ (e.g., 0.005). This smoothly blends the target network toward the current network.

**Why this works:** By fixing the target for many gradient steps, we approximate the supervised learning setting where the target is independent of the model. The target still changes (at each hard update or continuously via Polyak), but slowly enough to allow stable convergence.

### The Full DQN Algorithm

```
Initialize replay buffer D with capacity N
Initialize action-value network Q with random weights θ
Initialize target network Q⁻ with weights θ⁻ = θ

For each episode:
    Initialize state s (e.g., stack of 4 game frames)
    For each step:
        With probability ε, select random action a
        Otherwise, select a = argmax_a Q(s, a; θ)
        
        Execute a, observe reward r and next state s'
        Store (s, a, r, s', done) in D
        
        Sample random minibatch of transitions from D
        For each (sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ) in minibatch:
            If doneⱼ:
                yⱼ = rⱼ
            Else:
                yⱼ = rⱼ + γ max_{a'} Q(s'ⱼ, a'; θ⁻)
        
        Perform gradient descent on Σⱼ (yⱼ − Q(sⱼ, aⱼ; θ))²
        
        Every C steps: θ⁻ ← θ
```

---

## 3. Mathematical Formulation

### The DQN Loss Function

Given a minibatch B = {(sⱼ, aⱼ, rⱼ, s'ⱼ)} of size m:

$$L(\theta) = \frac{1}{m} \sum_{j=1}^{m} \left( y_j - Q(s_j, a_j; \theta) \right)^2$$

where:

$$y_j = r_j + \gamma (1 - \text{done}_j) \max_{a'} Q(s'_j, a'; \theta^-)$$

The gradient:

$$\nabla_\theta L(\theta) = -\frac{2}{m} \sum_{j=1}^{m} (y_j - Q(s_j, a_j; \theta)) \nabla_\theta Q(s_j, a_j; \theta)$$

Note: y_j is treated as a constant (no gradient flows through the target network).

### Maximization Bias and Double DQN

Q-learning uses max_{a'} Q(s', a') to estimate the value of the next state. This introduces a positive bias: even if all true Q-values are zero, the max of a set of noisy estimates is positive on average.

**Proof of the bias:** Let Q(s', aᵢ) = Q*(s', aᵢ) + εᵢ where εᵢ are i.i.d. zero-mean noise. Then:

E[max_i (Q* + εᵢ)] ≥ max_i Q*(s', aᵢ) + E[max_i εᵢ] ≥ max_i Q*(s', aᵢ)

The inequality is strict whenever there are multiple actions and the noise is non-degenerate. This means Q-learning systematically overestimates action values.

**Double DQN** addresses this by decoupling action selection from action evaluation:

$$y_j = r_j + \gamma Q(s'_j, \arg\max_{a'} Q(s'_j, a'; \theta); \theta^-)$$

The current network θ selects the best action, but the target network θ⁻ evaluates its value. This dramatically reduces overestimation.

### Prioritized Experience Replay

Instead of sampling transitions uniformly from the buffer, **prioritized experience replay (PER)** samples transitions with probability proportional to their TD error:

$$p_j \propto |\delta_j|^\alpha + \epsilon$$

where δ_j is the TD error of transition j, α controls how much prioritization is used (α=0 is uniform, α=1 is fully prioritized), and ε is a small constant to ensure all transitions have nonzero probability.

**Why this helps:** Transitions with large TD errors are "surprising" — the agent's current Q-function is particularly wrong about them. Prioritizing these transitions focuses learning on the most informative data.

**Importance sampling correction:** Because we no longer sample uniformly, the expected gradient is biased. We correct this with importance sampling weights:

$$w_j = \left( \frac{1}{N \cdot P(j)} \right)^\beta$$

where β anneals from some initial value to 1 over the course of training. This correction ensures unbiased gradient updates.

### Dueling DQN

The **dueling architecture** decomposes Q(s,a) into:

$$Q(s, a; \theta) = V(s; \theta_v) + A(s, a; \theta_a) - \frac{1}{|A|} \sum_{a'} A(s, a'; \theta_a)$$

where V is the state-value stream and A is the advantage stream. Both share a common feature extraction trunk.

**Intuition:** In many states, the value does not depend much on the action (e.g., in an Atari game, when no enemies are nearby, all actions have similar value). The dueling architecture learns V(s) more efficiently because it does not need to learn identical Q-values for every action in such states.

---

## 4. Worked Example

### DQN for CartPole

**Environment:** A pole is attached to a cart that moves along a track. The state is (cart_position, cart_velocity, pole_angle, pole_angular_velocity) — a 4-dimensional continuous vector. Actions: {push_left, push_right}. Reward: +1 for every step the pole stays upright. Episode ends when the pole falls or the cart leaves the track.

**Network architecture:** Input: 4 neurons. Hidden layers: 128 → 128 (ReLU). Output: 2 neurons (Q-values for left and right).

**Training (conceptual):**

1. Fill replay buffer with some random transitions.
2. At step 100: ε = 0.9 (mostly random). Sample minibatch, compute targets using target network, do gradient step.
   - Transition: s = (0.01, 0.02, 0.03, -0.01), a = right, r = 1, s' = (0.02, 0.01, 0.02, 0.03)
   - Target: y = 1 + 0.99 · max(Q(s', left; θ⁻), Q(s', right; θ⁻))
   - If Q(s', left; θ⁻) = 1.2 and Q(s', right; θ⁻) = 1.5: y = 1 + 0.99 · 1.5 = 2.485
   - Loss: (2.485 − Q(s, right; θ))²
3. Over thousands of steps, Q-values converge, ε decays, and the agent learns to balance the pole for 500 steps (the maximum episode length).

---

## 5. Relevance to Machine Learning Practice

### Where DQN and Its Variants Are Used

**Game playing:** DQN's original domain. Atari 2600, board games, and card games.

**Discrete action control:** Any problem with a finite action set — network routing, resource allocation, scheduling.

**Recommendation systems:** Actions are items to recommend; DQN-like methods learn long-term user engagement strategies.

**Not suitable for:** Continuous action spaces (you cannot compute max_{a'} Q(s',a') when a' is continuous). For continuous actions, use policy gradient methods, DDPG, TD3, or SAC.

### Hyperparameter Sensitivity

| Hyperparameter | Typical Range | Effect |
|---|---|---|
| Replay buffer size | 10^4 to 10^6 | Too small: forgets old experiences. Too large: wastes memory, dilutes recent data. |
| Target update frequency C | 100 to 10000 | Too small: targets change too fast. Too large: targets are stale. |
| Minibatch size | 32 to 256 | Larger batches reduce variance but cost more compute. |
| ε decay schedule | Linear or exponential | Must balance exploration and exploitation. |
| Learning rate α | 10^{-4} to 10^{-3} | Standard neural network tuning applies. |
| Discount γ | 0.95 to 0.999 | Task-dependent. |

---

## 6. Common Pitfalls & Misconceptions

**Not using a target network.** Without it, training is unstable and may diverge, especially with neural networks.

**Setting the replay buffer too small.** The buffer should be large enough to store diverse transitions. A buffer of size 1000 for a complex Atari game is far too small; 10^5 to 10^6 is typical.

**Applying DQN to continuous action spaces.** DQN requires computing max_a Q(s, a), which is only tractable for discrete actions. For continuous actions, use actor-critic methods.

**Using the same network for targets.** A very common implementation bug: forgetting to detach the target computation from the gradient graph. The gradient should not flow through the target network.

**Decaying ε too fast.** If exploration drops too quickly, the agent may converge to a suboptimal policy because it has not explored enough. A common approach is to decay ε linearly from 1.0 to 0.01 over the first million steps.

---

# III-B: Policy Gradient Methods — REINFORCE and Actor-Critic

---

## 1. Motivation & Intuition

### The Limitations of Value-Based Methods

All methods so far (DP, MC, TD, DQN) learn a value function (V or Q) and derive the policy from it: π(s) = argmax_a Q(s,a). This approach has several fundamental limitations:

1. **Discrete actions only.** Computing argmax over a discrete set is trivial. But for continuous actions (torque on a robotic joint, acceleration of a car), the argmax requires solving an optimization problem at every step, which is expensive or intractable.

2. **Deterministic policies.** Value-based methods produce deterministic policies. But some problems require stochastic policies (e.g., rock-paper-scissors, where any deterministic strategy is exploitable).

3. **Small changes in Q can cause large policy changes.** If Q(s, a₁) = 5.01 and Q(s, a₂) = 5.00, the greedy policy selects a₁. But if the values shift slightly to Q(s, a₁) = 4.99, the policy suddenly switches to a₂. This discontinuity makes learning unstable.

**Policy gradient methods** take a fundamentally different approach: they parameterize the policy directly as π(a|s; θ) and optimize θ by gradient ascent on the expected return.

### The Key Idea

Instead of learning Q-values and deriving a policy, we learn a policy directly. The policy is a probability distribution over actions, parameterized by θ. For continuous actions, this might be a Gaussian: π(a|s; θ) = N(μ_θ(s), σ_θ(s)²). For discrete actions, a softmax over action scores.

We define the objective:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R_t \right]$$

and optimize it by computing ∇_θ J(θ) and performing gradient ascent.

---

## 2. Conceptual Foundations

### The Policy Gradient Theorem

The policy gradient theorem (Sutton et al., 1999) provides the gradient of J(θ) without requiring the derivative of the transition function P(s'|s,a):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t | S_t) \cdot G_t \right]$$

**Reading this equation:**

- ∇_θ log π_θ(a|s) is the **score function** — the direction that makes action a more likely in state s.
- G_t is the return from time t.
- The gradient says: "Make actions that led to high returns more likely, and actions that led to low returns less likely, in proportion to how high or low the return was."

### REINFORCE

REINFORCE is the simplest policy gradient algorithm. It uses Monte Carlo returns (complete episodes) to estimate the gradient.

**Algorithm:**

```
Initialize policy parameters θ
For each episode:
    Generate episode: s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T
    For t = 0, 1, ..., T-1:
        Compute return G_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k+1}
        θ ← θ + α · γ^t · ∇_θ log π_θ(aₜ|sₜ) · G_t
```

**Derivation of the policy gradient:**

Start with J(θ) = E[G_0]. A trajectory τ = (s₀, a₀, r₁, ..., s_T) has probability:

P(τ|θ) = P(s₀) · Π_{t=0}^{T-1} [π_θ(a_t|s_t) · P(s_{t+1}|s_t, a_t)]

$$\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau|\theta) R(\tau) d\tau$$

Using the log-derivative trick: ∇_θ P(τ|θ) = P(τ|θ) ∇_θ log P(τ|θ):

$$= \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) \cdot R(\tau) \, d\tau$$

Now, ∇_θ log P(τ|θ) = ∇_θ [log P(s₀) + Σ_t log π_θ(a_t|s_t) + Σ_t log P(s_{t+1}|s_t,a_t)].

The P(s₀) and P(s_{t+1}|s_t,a_t) terms do not depend on θ, so they vanish:

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

Therefore:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) R(\tau) \right]$$

This can be refined to associate each action with only the future return (since past rewards cannot be influenced by future actions), yielding the version with G_t shown earlier.

### Baselines: Reducing Variance

REINFORCE has very high variance because the return G_t fluctuates widely across episodes. A simple and powerful fix is to subtract a **baseline** b(s) from the return:

$$\nabla_\theta J(\theta) = \mathbb{E} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t)) \right]$$

This does not introduce bias (because E[∇_θ log π_θ(a|s) · b(s)] = 0 for any b that does not depend on a), but it can dramatically reduce variance.

The optimal baseline that minimizes variance is approximately V^π(s). When G_t > V^π(s_t), the action was better than average — make it more likely. When G_t < V^π(s_t), the action was worse than average — make it less likely. The quantity G_t − V^π(s_t) is called the **advantage** A^π(s_t, a_t).

### Actor-Critic Methods

An **actor-critic** architecture combines:

- **Actor:** The policy π_θ(a|s) (learns which actions to take).
- **Critic:** A value function V_w(s) (estimates how good the current state is).

The critic provides the baseline for the actor's gradient, and is updated using TD learning:

**Critic update:** w ← w + α_w · δ · ∇_w V_w(s)
where δ = r + γV_w(s') − V_w(s) (TD error)

**Actor update:** θ ← θ + α_θ · δ · ∇_θ log π_θ(a|s)

Notice the actor update uses the TD error δ as the advantage estimate. This replaces the MC return G_t with a one-step bootstrapped estimate. The trade-off: lower variance (because we bootstrap) but introducing bias (because V_w is an approximation).

**Advantage Actor-Critic (A2C):**

Uses the advantage function explicitly:

A(s, a) ≈ r + γV_w(s') − V_w(s) = δ

**Asynchronous Advantage Actor-Critic (A3C):**

Runs multiple copies of the environment in parallel, each with its own agent. Each agent collects data and computes gradients independently, then asynchronously updates shared parameters. This provides:
- Diverse experience (different agents explore different parts of the state space).
- Stability (the effective batch is large because many agents contribute).
- Speed (parallel data collection).

---

## 3. Mathematical Formulation

### Generalized Advantage Estimation (GAE)

GAE (Schulman et al., 2015) provides a family of advantage estimators that interpolate between the bias-variance extremes:

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where δ_t = r_t + γV(s_{t+1}) − V(s_t).

- λ = 0: A_t = δ_t (1-step TD error). Low variance, high bias.
- λ = 1: A_t = G_t − V(s_t) (MC advantage). High variance, low bias (zero bias if V is exact).

λ ∈ (0, 1) trades off between these extremes. λ = 0.95 is a common practical choice.

**Derivation:**

The k-step advantage estimator is:

A_t^(k) = -V(s_t) + r_t + γr_{t+1} + ... + γ^{k-1}r_{t+k-1} + γ^k V(s_{t+k})

GAE is the exponentially-weighted average of these k-step estimators:

$$\hat{A}_t^{GAE} = (1-\lambda) \sum_{k=1}^{\infty} \lambda^{k-1} A_t^{(k)}$$

After algebraic simplification (telescoping), this yields the sum of discounted TD errors shown above.

### Entropy Regularization

To encourage exploration, a common technique is to add an entropy bonus to the objective:

$$J(\theta) = \mathbb{E}[\sum_t \gamma^t r_t] + \beta H(\pi_\theta(\cdot|s))$$

where H is the entropy of the policy:

$$H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

High entropy means the policy is spread out across actions (more exploration). The coefficient β controls the trade-off between reward maximization and exploration.

---

## 4. Worked Example

### REINFORCE on a Simple Bandit

**Setup:** Two actions {a₁, a₂}. No states (single-state problem). Rewards: a₁ → N(1, 1), a₂ → N(2, 1). Policy: π_θ(a₁) = σ(θ), π_θ(a₂) = 1 − σ(θ), where σ is the sigmoid function. Initialize θ = 0, so π(a₁) = π(a₂) = 0.5.

**Episode 1:** Sample a₁ (probability 0.5). Receive r = 0.5. No baseline.

∇_θ log π_θ(a₁) = ∇_θ log σ(θ) = 1 − σ(θ) = 0.5 (at θ = 0).

θ ← 0 + 0.01 · 0.5 · 0.5 = 0.0025

**Episode 2:** Sample a₂ (probability ~0.5). Receive r = 2.3.

∇_θ log π_θ(a₂) = ∇_θ log(1 − σ(θ)) = −σ(θ) ≈ −0.5.

θ ← 0.0025 + 0.01 · (−0.5) · 2.3 = 0.0025 − 0.0115 = −0.009

θ becomes negative, increasing π(a₂). Over many episodes, θ converges to a large negative value, making π(a₂) ≈ 1 (the optimal policy).

---

## 5. Relevance to Machine Learning Practice

### When to Use Policy Gradients vs. Value-Based Methods

| Criterion | Value-Based (DQN) | Policy Gradient |
|---|---|---|
| Action space | Discrete only | Discrete or continuous |
| Policy type | Deterministic (greedy) | Stochastic (explicitly parameterized) |
| Sample efficiency | Generally higher (off-policy) | Generally lower (on-policy) |
| Stability | Can diverge with function approx | Smoother optimization landscape |
| Best for | Discrete control, games | Robotics, continuous control |

### Policy Gradients in Production

**Robotics:** Policy gradient methods (especially PPO) are the standard for learning continuous control policies for robotic manipulation and locomotion.

**Natural Language Processing:** RLHF (Reinforcement Learning from Human Feedback) used to fine-tune large language models (like ChatGPT) uses PPO to optimize a reward model trained on human preferences. The policy is the language model itself.

**Recommendation systems:** Policy gradients can optimize long-term engagement metrics rather than immediate click-through rates.

---

## 6. Common Pitfalls & Misconceptions

**Forgetting to normalize returns.** Raw returns can vary enormously in scale. Normalizing returns (or advantages) to have zero mean and unit variance within a batch dramatically improves stability.

**Not using a baseline.** Vanilla REINFORCE without a baseline has extremely high variance and learns very slowly. Always use at least a simple value function baseline.

**Confusing the score function with the log probability.** The score function ∇_θ log π_θ(a|s) is a gradient — it is a vector in parameter space, not a scalar. Multiplying it by the scalar advantage gives the policy gradient.

**Assuming policy gradient methods are always on-policy.** While vanilla REINFORCE and A2C are on-policy, importance sampling can make them off-policy (at the cost of high variance with stale data). PPO and TRPO handle this with clipping or trust regions.

---

# III-C: Advanced Algorithms — PPO, TRPO, SAC, TD3

---

## 1. Motivation & Intuition

### Why Vanilla Policy Gradients Are Not Enough

REINFORCE and basic actor-critic methods work in principle but struggle in practice due to two interrelated problems:

1. **Step size sensitivity.** If the learning rate is too large, a single bad gradient step can drastically change the policy, collapsing it to a degenerate distribution (always choosing one action) and making recovery impossible. If too small, learning is glacially slow.

2. **Sample efficiency.** On-policy methods throw away all collected data after each gradient update (because the data came from the old policy). With expensive-to-collect data (e.g., real robot interactions), this waste is unacceptable.

Advanced algorithms address these problems through clever constraints on policy updates (TRPO, PPO) and off-policy learning with stable actor-critic architectures (SAC, TD3).

---

## 2. Conceptual Foundations

### Trust Region Policy Optimization (TRPO)

**Core idea:** Instead of taking arbitrary-sized gradient steps, constrain each update so that the new policy is "close" to the old policy. "Close" is measured by the KL divergence.

**Objective:**

$$\max_\theta \; \mathbb{E}_{s,a \sim \pi_{\theta_{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \hat{A}(s,a) \right]$$

subject to:

$$\mathbb{E}_s \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta$$

The ratio π_θ / π_{θ_old} is the importance sampling ratio. The constraint ensures the policy does not change too much in a single step.

**Why KL divergence?** The KL divergence between two policies measures how different they are in terms of action probabilities. A small KL divergence means the new policy is similar to the old one, so the importance sampling approximation remains valid and the performance estimate is reliable.

**Implementation:** TRPO uses conjugate gradient methods to approximately solve this constrained optimization, then performs a line search to ensure the constraint is satisfied. This is mathematically elegant but computationally expensive and tricky to implement.

### Proximal Policy Optimization (PPO)

**Core idea:** PPO achieves similar stability to TRPO but with a much simpler implementation. Instead of a hard KL constraint, PPO clips the importance sampling ratio to prevent large policy changes.

**Clipped objective:**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) and ε is typically 0.1 or 0.2.

**How clipping works:**

- If A_t > 0 (the action was good), the objective wants to increase r_t. But the clip limits r_t to 1+ε, preventing too large an increase.
- If A_t < 0 (the action was bad), the objective wants to decrease r_t. The clip limits r_t to 1−ε.
- The min ensures we take the more conservative (pessimistic) bound.

**Why PPO is popular:**

- Simple to implement (no conjugate gradients, no line search).
- Works well across a wide range of tasks.
- Used in RLHF for language model alignment, robotics, game playing.
- Can reuse data for multiple gradient steps per batch (typically 3-10 epochs per batch), improving sample efficiency over vanilla on-policy methods.

### Twin Delayed DDPG (TD3)

TD3 is designed for **continuous action spaces** and addresses three critical issues with DDPG (Deep Deterministic Policy Gradient):

1. **Overestimation bias (Twin critics):** Use two independent Q-networks Q₁ and Q₂. When computing the target, use the minimum of the two:

$$y = r + \gamma \min(Q_1(s', \tilde{a}'; \theta_1^-), Q_2(s', \tilde{a}'; \theta_2^-))$$

This combats the maximization bias that a single Q-network exhibits.

2. **Target policy smoothing (Noise):** Add noise to the target action:

$$\tilde{a}' = \pi_\phi(s') + \text{clip}(\epsilon, -c, c), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

This regularization prevents the policy from exploiting sharp peaks in the Q-function.

3. **Delayed policy updates:** Update the policy (actor) less frequently than the value function (critics). Typically, the actor is updated every 2 critic updates. This gives the critics time to converge before the policy changes.

### Soft Actor-Critic (SAC)

SAC introduces **maximum entropy RL**: the agent maximizes not just the expected return, but also the entropy of its policy.

**Objective:**

$$J(\pi) = \sum_t \mathbb{E} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

where α is the temperature parameter controlling the entropy-reward trade-off.

**Why entropy maximization?**

- **Exploration:** A high-entropy policy explores broadly, discovering multiple strategies.
- **Robustness:** A stochastic policy trained with entropy regularization is more robust to perturbations.
- **Multi-modality:** The policy can capture multiple near-optimal strategies rather than collapsing to one.
- **Better optimization landscape:** The entropy term smooths the objective, making optimization easier.

**Key components:**

- **Stochastic policy:** π_θ(a|s) = tanh(μ_θ(s) + σ_θ(s) · ε), where ε ~ N(0, I). The tanh squashes actions to a bounded range.
- **Twin Q-networks:** Like TD3, to combat overestimation.
- **Automatic temperature tuning:** α is learned by optimizing a dual objective to maintain a target entropy level.
- **Off-policy:** SAC stores transitions in a replay buffer, like DQN, achieving high sample efficiency.

**Soft Bellman equation:**

$$Q(s,a) = r + \gamma \mathbb{E}_{s'} \left[ V(s') \right]$$

where:

$$V(s) = \mathbb{E}_{a \sim \pi} \left[ Q(s,a) - \alpha \log \pi(a|s) \right]$$

---

## 3. Mathematical Formulation

### TRPO: The Surrogate Objective and Its Guarantee

TRPO relies on a theoretical result: the true performance difference between π_new and π_old can be expressed as:

$$J(\pi_{new}) - J(\pi_{old}) = \mathbb{E}_{s \sim d^{\pi_{new}}} \left[ \sum_a \pi_{new}(a|s) A^{\pi_{old}}(s,a) \right]$$

where d^{π_new} is the state distribution under π_new. The problem is that this depends on d^{π_new} (we do not know the state distribution under the new policy before we try it).

TRPO approximates this using the old state distribution and shows that the approximation error is bounded by the KL divergence:

$$J(\pi_{new}) \geq L_{\pi_{old}}(\pi_{new}) - C \cdot D_{KL}^{max}(\pi_{old} || \pi_{new})$$

where L is the surrogate objective (using old state distribution) and C is a constant. This gives a guaranteed monotonic improvement as long as the KL constraint is satisfied.

### SAC: Policy Evaluation and Improvement

**Soft Policy Evaluation:** For fixed π, the soft Q-function is the fixed point of:

$$Q(s,a) \leftarrow r + \gamma \mathbb{E}_{s'} \left[ \mathbb{E}_{a' \sim \pi}[Q(s',a') - \alpha \log \pi(a'|s')] \right]$$

**Soft Policy Improvement:** Given Q, the improved policy is:

$$\pi_{new} = \arg\min_{\pi'} D_{KL}\left( \pi'(\cdot|s) \;\bigg\|\; \frac{\exp(Q(s,\cdot)/\alpha)}{Z(s)} \right)$$

In practice, this is done by minimizing:

$$J_\pi(\theta) = \mathbb{E}_{s \sim D, \, a \sim \pi_\theta} \left[ \alpha \log \pi_\theta(a|s) - Q(s,a) \right]$$

The reparameterization trick is used to compute gradients through the stochastic sampling.

### Comparison Table

| Algorithm | Action Space | On/Off Policy | Key Innovation | Sample Eff. |
|---|---|---|---|---|
| TRPO | Discrete/Continuous | On-policy | KL trust region | Low-medium |
| PPO | Discrete/Continuous | On-policy (reuses for few epochs) | Clipped surrogate | Medium |
| TD3 | Continuous | Off-policy | Twin critics, delayed updates, target smoothing | High |
| SAC | Continuous | Off-policy | Max-entropy, automatic temp | High |

---

## 4. Worked Example

### PPO Update Step (Conceptual)

**Batch of data:** Collected under π_{θ_old}. For each transition (s, a, r, s'), we have:
- Old log probability: log π_{θ_old}(a|s) = −1.2
- Advantage estimate: A = +3.5

**Computing the update at epoch 1:**
- New log probability: log π_θ(a|s) = −0.8 (policy has shifted to make this action more likely)
- Ratio: r(θ) = exp(−0.8 − (−1.2)) = exp(0.4) ≈ 1.49
- Since A > 0, we want to increase the probability, so clip applies at 1 + ε = 1.2 (with ε = 0.2).
- Clipped value: min(1.49 × 3.5, 1.2 × 3.5) = min(5.22, 4.2) = 4.2

The objective is 4.2 instead of 5.22 — the clip has prevented the policy from changing too aggressively.

**At epoch 5:** If the ratio has moved to 1.3:
- min(1.3 × 3.5, 1.2 × 3.5) = min(4.55, 4.2) = 4.2 (still clipped)

The gradient is zero for this sample because it is in the clipped region. The policy has moved "enough" for this action.

---

## 5. Relevance to Machine Learning Practice

### PPO

**The most widely used RL algorithm.** Used in OpenAI Five (Dota 2), RLHF for ChatGPT, robotics (sim-to-real transfer), and many research benchmarks. Its simplicity and robustness make it the default choice for most new RL projects.

### SAC

**The default for continuous control.** Dominates MuJoCo benchmarks and is widely used in robotics. Its entropy regularization provides natural exploration without manual ε-scheduling.

### TD3

**A strong alternative to SAC.** Sometimes preferred when deterministic policies are desired (e.g., when the task has a clear single optimal action in each state). TD3 is simpler than SAC but lacks the automatic exploration benefit of entropy maximization.

### TRPO

**Mostly superseded by PPO in practice** but remains important theoretically. TRPO's guarantees inform why PPO works, and TRPO-like ideas appear in constrained RL settings.

---

## 6. Common Pitfalls & Misconceptions

**Thinking PPO is strictly on-policy.** PPO reuses data for multiple epochs, which is slightly off-policy. The clipping prevents this from causing problems, but using too many epochs (e.g., >10) can degrade performance because the data becomes too stale.

**Ignoring the entropy term in SAC.** The temperature α is crucial. Too high and the agent is random; too low and it collapses to a deterministic policy and stops exploring. Automatic temperature tuning (the default in modern SAC implementations) removes this from the list of hyperparameters.

**Applying TD3 to discrete action spaces.** TD3 is designed for continuous actions. For discrete actions, use DQN variants or PPO.

**Not normalizing advantages in PPO.** Unnormalized advantages can cause the clipped objective to behave poorly. Always normalize advantages to zero mean and unit variance within each minibatch.

**Confusing PPO's clipping with reward clipping.** PPO clips the *objective function* (specifically the importance ratio), not the rewards. Reward clipping is a separate technique (used in DQN, for example).

---

# Part IV: Exploration Strategies

---

## 1. Motivation & Intuition

### The Exploration-Exploitation Dilemma

An RL agent faces a fundamental tension at every step:

- **Exploitation:** Do what it currently believes is best, based on learned Q-values or policy.
- **Exploration:** Try something new to potentially discover better strategies.

Too much exploitation and the agent gets stuck in a suboptimal policy (because it never tried the better alternative). Too much exploration and the agent wastes time on random actions when it should be capitalizing on what it has learned.

This is not just a theoretical concern. In practice, exploration strategy is often the difference between an algorithm that learns a good policy and one that completely fails.

**Concrete example:** A restaurant recommendation system has served Italian food 100 times and Thai food once (by accident). Italian got an average rating of 3.5/5. Thai got 4.5/5 (but from just one observation). Should the system keep recommending Italian (exploitation) or try Thai more to get a better estimate of its true quality (exploration)?

---

## 2. Conceptual Foundations

### ε-Greedy Exploration

The simplest exploration strategy. With probability 1−ε, take the action with the highest Q-value. With probability ε, take a uniformly random action.

**Pros:** Simple, easy to implement, guarantees all actions are tried (eventually).
**Cons:** Exploration is undirected (random actions are not chosen intelligently). Every exploratory action is equally likely, even obviously bad ones. Requires tuning the ε schedule.

**Common schedules:**
- Constant ε (e.g., 0.1 throughout training).
- Linearly decaying ε (e.g., from 1.0 to 0.01 over training).
- Exponentially decaying ε.

### Softmax (Boltzmann) Exploration

Instead of choosing uniformly at random, softmax exploration chooses actions with probability proportional to their estimated Q-values:

$$\pi(a|s) = \frac{\exp(Q(s,a) / \tau)}{\sum_{a'} \exp(Q(s,a') / \tau)}$$

where τ is the **temperature** parameter:
- τ → ∞: Uniform random (all actions equally likely).
- τ → 0: Greedy (highest Q-value always chosen).

**Advantage over ε-greedy:** Actions with higher Q-values are explored more often. The agent does not waste time on clearly bad actions.

**Disadvantage:** Sensitive to the scale of Q-values. If Q-values are very close, softmax is nearly uniform; if they are far apart, it is nearly greedy. The temperature must be tuned.

### Upper Confidence Bound (UCB)

**Idea:** Choose the action that has the highest **upper confidence bound** on its Q-value. Actions that have been tried fewer times have wider confidence intervals, so they are more likely to be selected.

$$A_t = \arg\max_a \left[ Q(s,a) + c \sqrt{\frac{\ln t}{N(s,a)}} \right]$$

where N(s,a) is the number of times action a has been taken in state s, t is the total number of actions taken, and c controls the exploration-exploitation balance.

**Intuition:** The √(ln t / N(s,a)) term is large when N(s,a) is small (action a has not been tried much). This bonus shrinks as the action is tried more, at rate 1/√N, which matches the rate at which our estimate of Q improves.

**Provable guarantee:** UCB achieves logarithmic regret in multi-armed bandits, which is optimal up to constant factors.

**Limitation in RL:** UCB was designed for bandits (no states). Extending it to MDPs requires counting (s,a) visits, which is straightforward in tabular settings but difficult with function approximation.

### Thompson Sampling

**Idea:** Maintain a posterior distribution over Q-values (or reward parameters). At each step, sample Q-values from the posterior and act greedily with respect to the sample.

**Mechanism:**
1. Start with a prior distribution over the Q-values (e.g., Beta(1,1) for Bernoulli rewards).
2. At each step, sample Q̃(s,a) from the current posterior for each action.
3. Take action a = argmax Q̃(s,a).
4. Observe the reward and update the posterior using Bayes' rule.

**Why it works:** When the agent is uncertain about an action (wide posterior), it will sometimes sample high values and try it. When it has strong evidence that an action is good or bad (narrow posterior), it will behave accordingly. The exploration is naturally directed by uncertainty.

**Advantages:** Often the best-performing exploration strategy in practice. Automatically adapts exploration intensity to the level of uncertainty. Has theoretical guarantees competitive with UCB.

**Challenges in deep RL:** Maintaining a full posterior over neural network weights is intractable. Approximations include:
- Ensemble methods (train multiple Q-networks, sample one at random).
- Dropout as approximate Bayesian inference.
- Bootstrapped DQN.

### Curiosity-Driven Exploration

**Problem addressed:** In environments with **sparse rewards** (e.g., the agent receives zero reward for thousands of steps until it accidentally solves a puzzle), ε-greedy and softmax exploration are nearly useless because random actions almost never lead to the goal.

**Idea:** Give the agent an **intrinsic reward** for visiting "surprising" or "novel" states. The agent is incentivized to explore states it does not understand well.

**Prediction-error curiosity (ICM — Intrinsic Curiosity Module):**

1. Train a forward model f that predicts the next state representation: f(s_t, a_t) ≈ φ(s_{t+1}).
2. The intrinsic reward is the prediction error: r_int = ||f(s_t, a_t) − φ(s_{t+1})||².
3. Total reward: r_total = r_ext + β · r_int.

**Intuition:** If the agent can predict the next state well, it is not surprised — low curiosity. If it cannot predict the next state (because it has not explored this part of the environment), the curiosity reward drives it to explore.

**The "noisy TV" problem:** If there is a stochastic element the agent can never predict (like a TV showing random static), the prediction error remains high forever, and the agent gets trapped staring at the noise. Solutions include:
- Learning in a feature space (φ) that ignores irrelevant stochasticity (the ICM approach).
- Random Network Distillation (RND): measuring novelty by comparing a fixed random network's output to a trained predictor network's output.
- Count-based exploration: track how often states are visited and bonus for rare states.

---

## 3. Mathematical Formulation

### Regret Analysis for Bandits

In a K-armed bandit, the **regret** after T rounds is:

$$\text{Regret}(T) = T \mu^* - \sum_{t=1}^{T} \mathbb{E}[R_t]$$

where μ* is the mean reward of the best arm.

**UCB regret bound:**

$$\text{Regret}(T) \leq 8 \sum_{i: \mu_i < \mu^*} \frac{\ln T}{\Delta_i} + (1 + \frac{\pi^2}{3}) \sum_{i=1}^{K} \Delta_i$$

where Δ_i = μ* − μ_i is the suboptimality gap of arm i. This is O(K ln T), which is optimal (matching the Lai-Robbins lower bound up to constants).

**Thompson Sampling regret bound (Agrawal & Goyal, 2012):** For Beta-Bernoulli bandits, Thompson sampling also achieves O(K ln T) regret, matching UCB.

### Count-Based Exploration Bonuses

For tabular MDPs, a simple exploration bonus is:

$$r^+(s,a) = \frac{\beta}{\sqrt{N(s,a)}}$$

where N(s,a) is the visit count. The total reward becomes r(s,a) + r+(s,a).

**Why 1/√N?** By the central limit theorem, the standard error of the mean reward estimate decreases as 1/√N. The bonus matches this rate, ensuring that under-explored actions receive bonuses commensurate with our uncertainty about their true values.

For function approximation, pseudo-counts and density models approximate N(s,a).

---

## 4. Worked Example

### Thompson Sampling for a 3-Armed Bandit

**Setup:** Three arms with true reward probabilities p₁ = 0.3, p₂ = 0.5, p₃ = 0.7. We use Beta priors.

**Initialize:** Prior for each arm: Beta(1, 1) (uniform).

**Round 1:** Sample θ₁ ~ Beta(1,1) = 0.62, θ₂ ~ Beta(1,1) = 0.31, θ₃ ~ Beta(1,1) = 0.44.
Best sample: arm 1. Pull arm 1. Reward: 0 (failure). Update: arm 1 posterior → Beta(1, 2).

**Round 2:** Sample θ₁ ~ Beta(1,2) = 0.18, θ₂ ~ Beta(1,1) = 0.75, θ₃ ~ Beta(1,1) = 0.58.
Best sample: arm 2. Pull arm 2. Reward: 1 (success). Update: arm 2 posterior → Beta(2, 1).

**Round 3:** Sample θ₁ ~ Beta(1,2) = 0.25, θ₂ ~ Beta(2,1) = 0.89, θ₃ ~ Beta(1,1) = 0.91.
Best sample: arm 3. Pull arm 3. Reward: 1. Update: arm 3 posterior → Beta(2, 1).

After many rounds, arm 3's posterior concentrates near 0.7, and it is pulled most often.

---

## 5. Relevance to Machine Learning Practice

### Which Exploration Strategy to Use

| Strategy | Best When | Avoid When |
|---|---|---|
| ε-greedy | Simple problems, good baseline | Complex environments, sparse rewards |
| Softmax | Q-values are meaningful and well-scaled | Q-values change rapidly |
| UCB | Tabular settings, theoretical guarantees needed | Deep RL (hard to define counts) |
| Thompson sampling | Bayesian framework available, moderate action spaces | Exact posterior is intractable (approximations needed) |
| Curiosity-driven | Sparse rewards, large state spaces | Noisy environments (noisy TV problem) |

### Exploration in Production

In production systems (recommendations, ads, clinical trials), exploration has real costs. Showing a bad ad wastes money. Assigning a patient to a suboptimal treatment has ethical implications.

**Safe exploration** techniques include:
- Constraining exploration to actions that are "close" to a known-safe policy.
- Using pessimistic Q-value estimates (lower confidence bounds) to avoid catastrophic actions.
- Human-in-the-loop exploration where uncertain decisions are escalated to humans.

---

## 6. Common Pitfalls & Misconceptions

**Using ε-greedy with ε too high for too long.** If ε remains at 0.3 throughout training, 30% of actions are random. The agent's online performance is dragged down. Decay ε aggressively once the Q-function is reasonably accurate.

**Assuming exploration is unnecessary after enough training.** In non-stationary environments, the optimal policy may change. Continued exploration (with small ε or entropy regularization) is needed to detect and adapt to changes.

**Ignoring the difference between directed and undirected exploration.** ε-greedy explores uniformly at random. UCB and Thompson sampling explore intelligently, focusing on uncertain actions. The difference in performance can be enormous in complex environments.

**Applying curiosity bonuses without careful normalization.** Intrinsic rewards can dominate extrinsic rewards if not properly scaled, causing the agent to explore endlessly rather than exploiting what it has learned.

---

# Part V: Multi-Agent Reinforcement Learning

---

## 1. Motivation & Intuition

### When One Agent Is Not Enough

Many real-world problems involve multiple decision-makers interacting simultaneously. Traffic at an intersection involves multiple drivers. A market involves buyers and sellers. A team sport involves coordinated players. In these settings, the "environment" that each agent faces includes the behavior of other agents, which is itself changing as those agents learn.

**Why single-agent RL fails here:** In single-agent RL, the environment is (approximately) stationary — it does not change as the agent learns. But in multi-agent settings, the environment for agent A includes agent B, whose policy is changing. From A's perspective, the transition function P(s'|s,a) is non-stationary, violating the MDP assumption. This makes learning fundamentally harder.

### Two Key Settings

**Competitive (zero-sum):** One agent's gain is another's loss. Examples: board games (chess, Go), adversarial security, some financial trading. The solution concept is the **Nash equilibrium** — a set of strategies where no agent can improve by unilaterally changing its strategy.

**Cooperative:** All agents share a common goal. Examples: robot teams in a warehouse, cooperative communication, multi-drone coordination. The challenge is **credit assignment** (which agent's action led to the team's success or failure) and **coordination** (avoiding conflicting actions).

---

## 2. Conceptual Foundations

### Stochastic Games (Markov Games)

The multi-agent generalization of an MDP is a **stochastic game** defined by:

- **S:** Set of states (the global state visible to all agents, or partially observed).
- **A₁, A₂, ..., Aₙ:** Action sets for each of n agents.
- **P(s' | s, a₁, ..., aₙ):** Transition function depends on all agents' actions.
- **R_i(s, a₁, ..., aₙ):** Reward function for agent i.
- **γ:** Discount factor.

For zero-sum games with two agents: R₁ = −R₂.

### Competitive RL

**Self-play:** Train an agent against copies of itself. As the agent improves, its opponent improves too, creating an automatic curriculum. This is how AlphaGo and AlphaZero were trained.

**Minimax:** In two-player zero-sum games, the optimal strategy maximizes the minimum payoff (assuming the opponent plays optimally):

$$V^*(s) = \max_{a_1} \min_{a_2} \left[ R(s, a_1, a_2) + \gamma \sum_{s'} P(s'|s,a_1,a_2) V^*(s') \right]$$

**Nash equilibrium:** A strategy profile (π₁*, ..., πₙ*) such that no agent can improve its expected return by unilaterally changing its strategy:

$$J_i(\pi_i^*, \pi_{-i}^*) \geq J_i(\pi_i, \pi_{-i}^*) \quad \forall \pi_i, \forall i$$

For two-player zero-sum games, a Nash equilibrium always exists (in mixed strategies) and can be computed by linear programming for matrix games. For general-sum games, computing Nash equilibria is PPAD-complete (intractable in the worst case).

### Cooperative RL

**Centralized Training, Decentralized Execution (CTDE):** During training, agents can share information (observations, actions, rewards). During execution (deployment), each agent acts based only on its own observations.

**QMIX:** A popular CTDE algorithm that learns a centralized action-value function Q_tot that decomposes (monotonically) into individual agent Q-values:

$$Q_{tot}(s, \mathbf{a}) = f(Q_1(o_1, a_1), Q_2(o_2, a_2), ..., Q_n(o_n, a_n))$$

where f is a monotonic mixing function (ensuring that argmax of Q_tot over all agents' actions equals the product of individual argmaxes).

**Independent Learners:** Each agent runs its own RL algorithm (e.g., DQN) independently, treating other agents as part of the environment. Simple but suffers from non-stationarity. Surprisingly effective in some settings when combined with techniques like parameter sharing and communication channels.

### Communication in Multi-Agent RL

Agents can learn to communicate by sending messages through a differentiable communication channel. Approaches like **CommNet** and **TarMAC** allow agents to learn what information to share and how to interpret messages from teammates.

---

## 3. Mathematical Formulation

### Nash Equilibrium in Matrix Games

For a 2-player game with payoff matrices A (for player 1) and B (for player 2), a Nash equilibrium (p*, q*) satisfies:

p*ᵀ A q* ≥ pᵀ A q*  for all p (player 1's mixed strategy)
p*ᵀ B q* ≥ p*ᵀ B q  for all q (player 2's mixed strategy)

For zero-sum games (B = −A), this reduces to the minimax theorem:

$$\max_p \min_q p^T A q = \min_q \max_p p^T A q = v^*$$

where v* is the **value of the game**.

### Multi-Agent Policy Gradient

For agent i in a cooperative setting, the policy gradient is:

$$\nabla_{\theta_i} J_i = \mathbb{E} \left[ \nabla_{\theta_i} \log \pi_{\theta_i}(a_i|o_i) \cdot A_i(s, \mathbf{a}) \right]$$

The challenge is estimating A_i accurately when the return depends on all agents' actions. CTDE methods use a centralized critic that conditions on the global state and all agents' actions:

$$A_i(s, \mathbf{a}) \approx Q_i(s, a_1, ..., a_n) - V_i(s)$$

---

## 4. Worked Example

### Rock-Paper-Scissors (Nash Equilibrium)

**Payoff matrix for Player 1 (zero-sum):**

|  | Rock | Paper | Scissors |
|---|---|---|---|
| Rock | 0 | −1 | +1 |
| Paper | +1 | 0 | −1 |
| Scissors | −1 | +1 | 0 |

The unique Nash equilibrium is the uniform mixed strategy: p* = q* = (1/3, 1/3, 1/3).

**Verification:** If Player 2 plays (1/3, 1/3, 1/3), then Player 1's expected payoff for any pure strategy is:

E[Rock] = 0·(1/3) + (−1)·(1/3) + 1·(1/3) = 0
E[Paper] = 1·(1/3) + 0·(1/3) + (−1)·(1/3) = 0
E[Scissors] = (−1)·(1/3) + 1·(1/3) + 0·(1/3) = 0

All equal, so Player 1 has no incentive to deviate. By symmetry, neither does Player 2. The game value is v* = 0.

**Why a deterministic strategy fails:** If Player 1 always plays Rock, Player 2 plays Paper and wins every time. Any deterministic strategy is exploitable.

---

## 5. Relevance to Machine Learning Practice

### Applications

**Game AI:** AlphaGo/AlphaZero (competitive self-play), OpenAI Five for Dota 2 (cooperative team play), Pluribus for multiplayer poker.

**Autonomous driving:** Each car is an agent. Must cooperate (don't crash) while competing (each wants to reach its destination quickly).

**Communication networks:** Multiple base stations manage bandwidth allocation cooperatively, or multiple users compete for bandwidth.

**Multi-robot systems:** Warehouse robots (Amazon), drone swarms for search and rescue, multi-robot assembly.

### Challenges in Practice

**Scalability:** The joint action space grows exponentially with the number of agents. With n agents and k actions each, the joint action space has k^n elements.

**Non-stationarity:** As each agent learns, the environment for every other agent changes. This can cause oscillating or diverging learning dynamics.

**Credit assignment:** In cooperative settings, the team gets a single reward. Determining which agent deserves credit is difficult (did the team win because of Agent 3's brilliant move or despite Agent 5's mistake?).

---

## 6. Common Pitfalls & Misconceptions

**Assuming independent learning always works.** Independent learners treat other agents as part of the environment, but the "environment" is non-stationary (other agents are learning). This can prevent convergence.

**Assuming Nash equilibria are unique or easy to find.** Most multi-player games have multiple Nash equilibria, and finding even one is computationally hard in general.

**Applying single-agent algorithms without modification.** Many convergence guarantees for single-agent RL (e.g., Q-learning converging to Q*) do not hold in multi-agent settings. Joint-action learners and CTDE methods are needed.

**Neglecting the coordination problem.** Even if agents learn individually optimal strategies, they may not coordinate. Two robots heading for the same doorway simultaneously is suboptimal even if each robot's individual policy is locally optimal.

---

# Part VI: Inverse Reinforcement Learning

---

## 1. Motivation & Intuition

### The Reward Function Is the Hardest Part

Designing a reward function for RL is notoriously difficult. What reward function would you give a self-driving car? "+1 for reaching the destination" leads to reckless driving. "−1 for each second of travel time" incentivizes speeding. Adding "−100 for hitting a pedestrian" still does not capture the nuance of human driving (maintaining comfortable distances, following social norms, being courteous).

**Inverse RL (IRL)** flips the problem: instead of designing a reward function by hand, learn it from demonstrations of desired behavior.

**Key insight:** An expert's behavior implicitly encodes the reward function they are optimizing. A skilled driver's demonstrations reveal that they value safety, efficiency, comfort, and social norms — even though these are never explicitly specified.

### The IRL Pipeline

1. Observe an expert performing a task (collect trajectories).
2. Infer the reward function that makes the expert's behavior (approximately) optimal.
3. Use the inferred reward function to train a new RL agent, potentially in different conditions.

---

## 2. Conceptual Foundations

### Why Not Just Imitate?

**Behavioral cloning** (supervised learning: state → action) is the simplest approach but suffers from **compounding errors**: small mistakes at each step push the agent into states the expert never visited, where the cloned policy has no training data and makes larger mistakes, leading to a cascade of failures.

IRL avoids this because it learns a reward function, not a policy. The RL agent trained on this reward function can recover from mistakes (it knows what states are good and bad, not just what actions to take in the states the expert visited).

### The Ambiguity Problem

IRL is fundamentally **ill-posed**: many reward functions can explain the same behavior. The constant reward R(s,a) = 0 makes every policy optimal, including the expert's. Trivially, R(s,a) = 0 always works. More subtly, adding a constant or multiplying by a positive scalar does not change the optimal policy.

**Resolutions:**
- **Maximum margin:** Find the reward function that makes the expert's policy maximally better than alternatives (Abbeel & Ng, 2004).
- **Maximum entropy IRL (MaxEnt IRL):** Assume the expert follows a Boltzmann-rational policy: π(τ) ∝ exp(R(τ)). Among all policies consistent with the observed feature expectations, choose the one with maximum entropy. This gives a unique, well-defined distribution.
- **Bayesian IRL:** Place a prior over reward functions and compute the posterior given demonstrations.

### Maximum Entropy IRL (Ziebart et al., 2008)

**Assumption:** The expert's trajectory distribution is:

$$P(\tau) = \frac{1}{Z} \exp\left(\sum_t R(s_t, a_t)\right)$$

where Z is the partition function. This says trajectories with higher cumulative reward are exponentially more likely, but the expert is not perfectly optimal (they occasionally make suboptimal choices, consistent with bounded rationality).

**Feature matching:** Assume R(s,a) = θᵀ φ(s,a), where φ are hand-crafted features (e.g., distance to lane center, speed relative to limit, proximity to other cars). The goal is to find θ such that the expected feature counts under the model match the empirical feature counts from demonstrations:

$$\mathbb{E}_\pi[\phi] = \hat{\mathbb{E}}_D[\phi]$$

This is a moment-matching condition. Maximum entropy IRL finds the θ that satisfies this condition while maximizing the entropy of the trajectory distribution.

---

## 3. Mathematical Formulation

### Maximum Entropy IRL Objective

$$\max_\theta \sum_{\tau \in D} \log P(\tau | \theta) = \sum_{\tau \in D} \left[ \theta^T \sum_t \phi(s_t, a_t) - \log Z(\theta) \right]$$

The gradient:

$$\nabla_\theta \mathcal{L} = \hat{\mathbb{E}}_D[\phi] - \mathbb{E}_{\pi_\theta}[\phi]$$

This says: increase the reward for features that the expert exhibits more than the current policy, and decrease it for features the current policy exhibits more than the expert.

**Algorithm:**

1. Initialize θ.
2. Solve the forward RL problem: find π_θ (the optimal policy for R = θᵀφ).
3. Compute expected feature counts under π_θ.
4. Update θ ← θ + α (feature_counts_expert − feature_counts_policy).
5. Repeat until convergence.

### Generative Adversarial Imitation Learning (GAIL)

GAIL (Ho & Ermon, 2016) casts IRL as a GAN-like problem:

- **Generator:** The policy π_θ generates trajectories.
- **Discriminator:** D_w(s,a) classifies whether a state-action pair comes from the expert or the policy.

**Objective:**

$$\min_\theta \max_w \mathbb{E}_{\pi_E}[\log D_w(s,a)] + \mathbb{E}_{\pi_\theta}[\log(1 - D_w(s,a))]$$

The discriminator's output log(D_w(s,a)) acts as a learned reward function. The policy is trained with any standard RL algorithm (e.g., TRPO, PPO) using this reward.

**Advantage:** No need to hand-design features φ. The discriminator learns to distinguish expert from agent behavior in whatever way is most informative.

---

## 4. Worked Example

### Simple Feature-Based IRL

**Setup:** A 3×3 grid world. The expert always takes the shortest path from (0,0) to (2,2), preferring routes that stay near the left wall.

**Features:** φ₁(s) = distance to goal, φ₂(s) = distance to left wall.

**Expert demonstrations:**

τ₁: (0,0)→(0,1)→(0,2)→(1,2)→(2,2) (goes down along left wall, then right)
τ₂: (0,0)→(0,1)→(0,2)→(1,2)→(2,2) (same path)

**Expert feature counts:** E_D[φ₁] = average distance to goal along path ≈ 2.0. E_D[φ₂] = average distance to left wall ≈ 0.4.

**Initialize θ = (0, 0).** Current policy is random. Its expected features will differ: E_π[φ₁] ≈ 2.5 (less efficient), E_π[φ₂] ≈ 1.0 (not near left wall).

**Gradient:** (2.0 − 2.5, 0.4 − 1.0) = (−0.5, −0.6). Update θ: R = −0.5·φ₁ − 0.6·φ₂. This means: reward getting closer to the goal (negative reward for being far away) and reward being near the left wall.

After several iterations, the policy learns to mimic the expert's path preference.

---

## 5. Relevance to Machine Learning Practice

### Applications of IRL

**Autonomous driving:** Learn driving styles from human demonstrations. Captures nuanced behavior (how aggressively to merge, how much space to leave) that is impossible to specify manually.

**Robot manipulation:** Learn assembly or cooking tasks from human demonstrations. The reward function transfers to different robot embodiments or environments.

**RLHF for language models:** The reward model trained on human preferences in RLHF is conceptually an IRL-derived reward function. Humans compare two model outputs (the "demonstrations" of preference), and the reward model learns what makes a response "good."

**Animation and game design:** Learn natural-looking character movement from motion capture data, rather than hand-crafting animation rules.

### IRL vs. Behavioral Cloning vs. DAgger

| Method | Learns | Compounding Error? | Needs RL? |
|---|---|---|---|
| Behavioral Cloning | Policy (supervised) | Yes (severe) | No |
| DAgger | Policy (interactive supervised) | Mitigated (queries expert) | No |
| IRL | Reward function | No (trains RL agent) | Yes |
| GAIL | Policy (via adversarial reward) | No | Yes |

---

## 6. Common Pitfalls & Misconceptions

**Assuming IRL gives a unique reward function.** The reward function is always ambiguous up to shaping, scaling, and other transformations. The learned reward should be evaluated by the quality of the policy it produces, not by comparing it to some "ground truth" reward.

**Ignoring the cost of the inner RL loop.** IRL requires solving a full RL problem at each iteration of the outer loop (to find the optimal policy for the current reward estimate). This can be extremely expensive. GAIL avoids this by combining the two loops.

**Using too few demonstrations.** IRL needs enough demonstrations to cover the relevant state space. If the expert only demonstrates one scenario, the inferred reward function may not generalize.

**Confusing IRL with imitation learning.** IRL learns a reward function; imitation learning (behavioral cloning, DAgger) learns a policy directly. They solve related but distinct problems.

---

# Part VII: Applications of RL

---

## 1. Robotics

### Challenges

**Continuous, high-dimensional state and action spaces:** A humanoid robot has 20+ degrees of freedom, each with position, velocity, and torque dimensions.

**Sample efficiency:** Real robot interactions are slow and expensive. An Atari agent can play thousands of games per hour; a real robot might manage a few dozen trials per hour.

**Sim-to-real transfer:** Train in simulation (cheap, fast, safe) and transfer to the real world. The challenge is the **reality gap** — differences between the simulator and reality (friction, sensor noise, visual appearance). **Domain randomization** addresses this by randomizing simulator parameters during training, making the policy robust to variation.

**Safety:** A robot cannot "explore" dangerous actions (dropping objects, colliding with humans). Constrained RL and safe exploration techniques are essential.

### Success Stories

**Dexterous manipulation:** OpenAI trained a robotic hand (Shadow Dexterous Hand) to solve a Rubik's Cube using PPO with domain randomization. The policy transferred from simulation to a real robot hand.

**Locomotion:** Quadruped robots (like those from Agility Robotics and Boston Dynamics) use RL-trained policies for walking, running, and navigating rough terrain.

**Surgical robotics:** RL has been applied to autonomous suturing and needle insertion, learning fine motor policies from demonstrations and simulation.

---

## 2. Game Playing

### Milestones

**TD-Gammon (1995):** TD(λ) learned backgammon at a near-expert level.

**Atari (2013-2015):** DQN played Atari games from raw pixels at superhuman levels.

**AlphaGo (2016) and AlphaZero (2017):** Defeated the world champion in Go using a combination of deep neural networks, MCTS, and self-play RL. AlphaZero mastered chess, shogi, and Go from scratch with no human knowledge beyond the rules.

**OpenAI Five (2019):** Cooperative multi-agent RL team beat world champions in Dota 2 (a complex, real-time strategy game with imperfect information).

**Pluribus (2019):** Trained via self-play with search, achieved superhuman performance in 6-player no-limit Texas Hold'em poker.

---

## 3. Resource Management

**Data center cooling (Google, 2016):** RL reduced energy consumption for cooling by 40%, saving millions of dollars annually. The agent controlled fans, cooling systems, and window openings based on temperature, humidity, and workload.

**Network traffic optimization:** RL agents manage packet routing in communication networks, adapting to changing traffic patterns in real time.

**Supply chain optimization:** RL handles inventory management, logistics, and demand forecasting under uncertainty, outperforming traditional operations research methods in complex, dynamic environments.

---

## 4. Autonomous Systems

**Self-driving cars:** RL for lane changing, merging, and intersection navigation. Often combined with model predictive control and imitation learning for safety.

**Drone navigation:** RL for obstacle avoidance, path planning, and formation control in GPS-denied environments.

**Financial trading:** RL agents learn trading strategies that adapt to market conditions. Challenges include non-stationarity, partial observability, and transaction costs.

---

# Part VIII: Interview Preparation

---

## Foundational Questions

**Q1: What is the difference between reinforcement learning, supervised learning, and unsupervised learning?**

Supervised learning maps inputs to known outputs using labeled data. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning learns to make sequential decisions by interacting with an environment and receiving reward signals, without explicit correct-action labels. The key distinguishing feature of RL is the temporal nature of the problem: actions affect not just immediate rewards but also future states and future rewards. RL also involves the exploration-exploitation trade-off, which is absent in supervised and unsupervised learning.

**Q2: What is the Markov property and why is it important?**

The Markov property states that the future is conditionally independent of the past given the present state. Formally: P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t). This is important because it means the current state contains all information necessary for decision-making. Without it, the Bellman equations do not hold, and the entire MDP framework breaks down. In practice, when the Markov property is violated (partial observability), we either augment the state (e.g., stack recent observations) or use recurrent architectures that implicitly maintain a memory of past observations.

**Q3: Explain the difference between the state-value function V(s) and the action-value function Q(s,a).**

V^π(s) is the expected return starting from state s and following policy π. Q^π(s,a) is the expected return starting from state s, taking action a, and then following π. The relationship is: V^π(s) = Σ_a π(a|s) Q^π(s,a). Q-values are more useful for control because they directly tell you which action is best: π*(s) = argmax_a Q*(s,a). V-values alone are insufficient for action selection without a model of the environment (you need to know which state each action leads to).

**Q4: What is the discount factor γ and what happens if you set it too high or too low?**

γ ∈ [0,1] controls the relative importance of future vs. immediate rewards. G_t = Σ_{k=0}^∞ γ^k R_{t+k+1}. With γ = 0, the agent is completely myopic and only optimizes for immediate reward. With γ → 1, the agent values long-term rewards almost equally to immediate ones, but learning becomes slower because reward signals must propagate over more steps, and in infinite-horizon tasks, the return may diverge. In practice, γ is chosen based on the effective time horizon: γ = 0.99 corresponds to an effective horizon of ~100 steps (since 0.99^100 ≈ 0.37).

**Q5: What is the Bellman equation and why does it matter?**

The Bellman equation is a recursive relationship that decomposes the value of a state into the immediate reward plus the discounted value of the next state: V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V^π(s')]. It matters because it transforms the problem of estimating a long-run expected sum into a one-step relationship. This enables iterative algorithms (policy evaluation, value iteration, TD learning) that converge to the true value function by repeatedly applying this relationship.

---

## Mathematical Questions

**Q6: Derive the policy gradient theorem. Why does it not require differentiating the transition dynamics?**

Starting from J(θ) = E_{τ~π_θ}[R(τ)], we write:

∇_θ J = ∇_θ ∫ P(τ|θ) R(τ) dτ = ∫ P(τ|θ) ∇_θ log P(τ|θ) R(τ) dτ

Now, log P(τ|θ) = log p(s₀) + Σ_t [log π_θ(a_t|s_t) + log P(s_{t+1}|s_t,a_t)].

Taking the gradient with respect to θ, the terms p(s₀) and P(s_{t+1}|s_t,a_t) do not depend on θ and vanish:

∇_θ log P(τ|θ) = Σ_t ∇_θ log π_θ(a_t|s_t)

Therefore: ∇_θ J = E_τ[(Σ_t ∇_θ log π_θ(a_t|s_t)) R(τ)]

The transition dynamics P(s'|s,a) cancel because log P(τ|θ) includes them but they are constant with respect to θ. This is the key insight: we can optimize the policy without knowing or differentiating through the environment dynamics.

**Q7: Prove that the Bellman operator is a contraction mapping.**

Define (TV)(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')].

For any two value functions V₁, V₂:

|(TV₁)(s) − (TV₂)(s)| = |Σ_a π(a|s) Σ_{s'} P(s'|s,a) γ [V₁(s') − V₂(s')]|
≤ Σ_a π(a|s) Σ_{s'} P(s'|s,a) γ |V₁(s') − V₂(s')|
≤ γ ||V₁ − V₂||_∞ · Σ_a π(a|s) Σ_{s'} P(s'|s,a) = γ ||V₁ − V₂||_∞

Taking max over s: ||TV₁ − TV₂||_∞ ≤ γ ||V₁ − V₂||_∞. Since γ < 1, this is a contraction. By the Banach fixed-point theorem, there exists a unique fixed point V^π, and iterative application of T converges to it at rate γ^k.

**Q8: Explain the maximization bias in Q-learning and how Double Q-learning fixes it.**

In Q-learning, the target uses max_a Q(s',a). If Q-values have estimation errors ε_a ~ N(0, σ²), then E[max_a (Q* + ε_a)] ≥ max_a Q*, with the gap growing with the number of actions and variance. This is because the max operator selects the action with the highest *noise*, not necessarily the highest true value.

Double Q-learning decouples selection and evaluation: it uses one Q-network to select the action (a* = argmax_a Q(s',a; θ)) and a different Q-network to evaluate it (Q(s',a*; θ⁻)). Since the selection noise and evaluation noise are independent, the bias is eliminated:

y = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ⁻)

In DQN, the current network and target network serve as the two Q-networks.

**Q9: Derive the GAE (Generalized Advantage Estimation) formula.**

The k-step advantage is: A_t^(k) = −V(s_t) + Σ_{l=0}^{k-1} γ^l r_{t+l} + γ^k V(s_{t+k}).

Define δ_t = r_t + γV(s_{t+1}) − V(s_t).

Note: A_t^(1) = δ_t, A_t^(2) = δ_t + γδ_{t+1}, and in general A_t^(k) = Σ_{l=0}^{k-1} γ^l δ_{t+l}.

GAE is the exponentially-weighted average: A_t^GAE = (1−λ) Σ_{k=1}^∞ λ^{k-1} A_t^(k).

Substituting: A_t^GAE = (1−λ) Σ_{k=1}^∞ λ^{k-1} Σ_{l=0}^{k-1} γ^l δ_{t+l}

Swapping summation order and computing the geometric sum yields: A_t^GAE = Σ_{l=0}^∞ (γλ)^l δ_{t+l}.

At λ=0: A_t = δ_t (one-step TD, low variance, high bias). At λ=1: A_t = Σ γ^l δ_{t+l} = G_t − V(s_t) (MC, zero bias if V correct, high variance).

---

## Applied Questions

**Q10: You are building an RL agent for a robotic arm that must pick up objects of varying shapes and sizes. What algorithm would you use and why? What are the key design decisions?**

**Algorithm choice:** SAC or TD3 for continuous action control, trained in simulation with domain randomization.

**Key decisions:**

*State representation:* Joint angles/velocities + object pose from vision (raw images or extracted features). Consider using a learned visual encoder (pre-trained or jointly trained).

*Action space:* Joint torques or position deltas. Position deltas are easier to learn but less precise; torques give full control but increase the difficulty.

*Reward design:* Dense reward (reward for reducing distance to object, grasping force, lift height) rather than sparse reward (+1 for successful grasp only). Shaped reward dramatically accelerates learning.

*Sim-to-real:* Domain randomization over object textures, masses, friction coefficients, camera positions, and lighting conditions. Maybe system identification to calibrate the simulator.

*Safety:* Joint limits, torque limits, collision avoidance constraints. Possibly constrained MDP formulation.

*Sample efficiency:* Off-policy methods (SAC) for data efficiency. Pre-train with demonstrations (from kinesthetic teaching or teleoperation) using behavioral cloning, then fine-tune with RL.

**Q11: Your Q-learning agent's performance suddenly degrades after initially improving. What might be going wrong?**

Several possibilities:

*Catastrophic forgetting:* The replay buffer has been overwritten with recent (different) experiences. If the agent's policy has shifted to a different part of the state space, it may forget how to handle earlier regions. Solution: increase buffer size, use prioritized replay.

*Target network divergence:* If target updates are too frequent, the moving target can cause oscillations. Solution: increase the target update period or use Polyak averaging.

*Exploration collapse:* ε has decayed too quickly, and the agent has settled on a suboptimal policy. Solution: slow down ε decay or use entropy regularization.

*Overestimation bias:* Q-values have inflated to unrealistic levels, causing the policy to take bad actions that appear good on paper. Solution: use Double DQN.

*Non-stationarity:* If the environment is changing (e.g., an opponent adapting), the agent's policy may become obsolete. Solution: maintain exploration, use faster learning rates, or meta-learning.

*Learning rate issues:* If α is too high, the agent overshoots the optimum and oscillates. If it was reduced by a schedule, check that the schedule is appropriate.

**Q12: How would you apply RL to optimize a recommendation system for long-term user engagement?**

*State:* User profile features (demographics, past interactions, session features like time of day, device).

*Action:* Which item (or slate of items) to recommend.

*Reward:* Immediate: click, dwell time, purchase. Long-term: retention (does the user come back tomorrow?), subscription renewal, total lifetime value.

*Algorithm:* Off-policy methods (DQN or SAC) are preferred because you can learn from logged interaction data. Batch (offline) RL methods are particularly relevant since live exploration is risky.

*Key challenges:*
- Huge discrete action space (millions of items): use action embedding or hierarchical action selection.
- Delayed rewards: γ must be set appropriately (e.g., one step = one session; γ = 0.95 corresponds to ~20 sessions of planning horizon).
- Off-policy evaluation: before deploying a new policy, evaluate it offline using importance sampling or doubly robust estimators.
- Filter bubbles: without exploration, the system shows only familiar content. Add exploration bonuses or entropy regularization.
- Confounders: users choose to visit the platform for exogenous reasons. Careful causal modeling is needed.

**Q13: Compare on-policy and off-policy methods. When would you choose one over the other?**

*On-policy (SARSA, A2C, PPO):* Learn about the policy currently being followed. Data must be generated by the current policy; old data is discarded after each update. This is sample-inefficient but stable, because the data distribution matches the policy being evaluated.

*Off-policy (Q-learning, DQN, SAC, TD3):* Learn about a different (typically optimal) policy while following an exploratory policy. Data can be reused from a replay buffer, making these methods more sample-efficient. But the distribution mismatch (data from old policies) can cause instability.

Choose on-policy when: safety is critical (the agent's actual behavior matches the policy being optimized), the environment is fast to simulate (sample efficiency is less important), or stability is paramount. Choose off-policy when: data is expensive (robotics, real-world systems), you have access to historical interaction logs (offline RL), or you need to separate exploration from the target policy.

---

## Debugging & Failure-Mode Questions

**Q14: Your PPO agent's policy entropy collapses to near zero early in training. What does this mean and how do you fix it?**

Entropy collapse means the policy has become nearly deterministic — it assigns almost all probability to one action in each state. This typically happens because:

*Learning rate too high:* Large policy updates push the policy to an extreme, and the clipping mechanism cannot prevent it if the initial update is already past the clip boundary.

*Insufficient entropy regularization:* The coefficient β on the entropy bonus is too small. Increase it.

*Poor initialization:* If the policy network's initial outputs are skewed (e.g., due to weight initialization), the softmax may already be near-deterministic.

*Reward scale mismatch:* If rewards are very large, the advantage estimates dominate the entropy bonus, driving the policy toward determinism.

**Fixes:** Increase the entropy coefficient. Reduce the learning rate. Use a learning rate warmup. Clip the initial advantages to a reasonable range. Check that the policy network initialization produces roughly uniform action probabilities.

**Q15: You trained a DQN agent in simulation and it works well, but performance drops dramatically when deployed on a real robot. What went wrong?**

This is the **sim-to-real gap**. Possible causes:

*Visual differences:* The simulator's rendering does not match real camera images (different lighting, textures, colors). Fix: domain randomization over visual parameters, or use a learned domain adaptation module.

*Dynamics mismatch:* Simulated physics differ from reality (friction, mass, joint flexibility, actuator delay). Fix: system identification (measure real physics and calibrate the simulator), or train with randomized physics parameters.

*Sensor noise:* Real sensors have noise and latency that the simulator did not model. Fix: add realistic noise models to the simulator.

*Action execution differences:* In simulation, actions execute instantly and perfectly. Real motors have response time, backlash, and control frequency limitations. Fix: model actuator dynamics in simulation.

*State representation:* If the policy relies on simulator-specific state information (e.g., exact object positions from the physics engine), these may not be available in the real world. Fix: use only sensor-based state representations (images, joint encoders) during training.

**Q16: Your multi-agent cooperative RL system shows training instability — the team reward oscillates wildly. Diagnose and propose solutions.**

*Likely causes:*

*Non-stationarity:* Each agent's policy change affects all other agents' optimal policies. This creates a "moving target" problem where no agent can converge because the others keep changing.

*Credit assignment failure:* With a shared team reward, individual agents cannot determine whether their own actions helped or hurt. Some agents may learn to "free-ride" while others do the work.

*Communication breakdown:* If agents need to coordinate (e.g., avoid collisions or cover different areas), but the communication protocol has not been learned, the team alternates between good and bad coordination randomly.

*Solutions:*
- Use CTDE methods (QMIX, MAPPO) that train with centralized critics but decentralized actors.
- Introduce individual reward shaping (bonus for agent-specific contributions) alongside team reward.
- Stabilize training with smaller learning rates, target networks for the centralized critic, and larger batch sizes.
- Use parameter sharing across agents (when they are homogeneous) to reduce the search space.
- Add communication channels (CommNet, TarMAC) to enable coordination.

---

## Follow-Up and Probing Questions

**Q17: "You said PPO uses clipping. But what if the advantage estimate is wrong? How does PPO handle bad advantage estimates?"**

If the advantage estimate is wrong (high bias from a poor value function), PPO's clipping does not help directly — it prevents large policy changes but cannot fix the direction of the change. A bad advantage may push the policy in the wrong direction, just by a limited amount.

To mitigate: improve the value function (more training, better architecture, larger batch sizes). Use GAE with appropriate λ to control the bias-variance trade-off of the advantage estimator. Monitor the explained variance of the value function (correlation between predicted values and actual returns). If it is low, the value function is unreliable and the advantages are noisy.

**Q18: "Can you use Q-learning with continuous actions? If not, what are the alternatives?"**

Standard Q-learning requires computing max_a Q(s,a), which is intractable for continuous a. Alternatives:

- *Discretize the action space:* Simple but scales exponentially with dimensions. A 6-DOF robot with 10 bins per dimension has 10^6 actions.
- *DDPG / TD3:* Learn a deterministic policy μ(s) that approximates the argmax. The critic evaluates Q(s, μ(s)).
- *SAC:* Uses a stochastic policy with reparameterization.
- *NAF (Normalized Advantage Functions):* Parameterize Q(s,a) as a quadratic in a, so the max has a closed-form solution.
- *CEM or CMA-ES:* Use evolutionary optimization to find argmax Q(s,a) at each step.

**Q19: "Explain the 'deadly triad' in RL. Why is it dangerous?"**

The deadly triad refers to the combination of: (1) function approximation, (2) bootstrapping, and (3) off-policy learning. When all three are present simultaneously, Q-learning can diverge — the Q-values grow without bound.

- *Function approximation* introduces generalization errors that can propagate.
- *Bootstrapping* (using V(s') in the target) means errors in one state's estimate affect other states' targets.
- *Off-policy* means the data distribution does not match the target policy, so errors are not corrected by revisiting states under the true distribution.

DQN overcomes this (in practice, if not in theory) through target networks (which limit how fast the bootstrap target changes) and experience replay (which provides a more diverse data distribution). But convergence is not guaranteed, and instability remains a practical concern.

**Q20: "How does RLHF for language models relate to the RL concepts we've discussed?"**

RLHF uses RL to fine-tune a language model (the policy) based on human preferences:

1. A reward model is trained from human comparisons (which of two responses is better). This is inverse RL: the reward function is learned from demonstrations of human preferences.

2. The language model is fine-tuned using PPO with the reward model providing the reward signal. The state is the prompt + generated text so far. The action is the next token. The policy is the language model's next-token distribution.

3. A KL penalty between the fine-tuned model and the original model prevents the policy from diverging too far (analogous to the trust region in TRPO).

This connects to: policy gradient methods (the PPO update), reward shaping (the KL penalty acts as an implicit reward that keeps the model fluent), and inverse RL (the reward model captures human preferences without explicit specification).

**Q21: "Walk me through how you would debug an RL system that is not learning at all — zero improvement over random."**

Systematic debugging checklist:

1. *Check the reward signal:* Print rewards at each step. Is the agent receiving any reward? If rewards are sparse, the agent may never stumble upon the reward by random exploration. Add shaped rewards or use curiosity-driven exploration.

2. *Check the environment:* Is the environment implemented correctly? Are states and actions passed in the right format? Are episodes terminating properly? A common bug: the "done" flag is not set, so the agent never sees a terminal state.

3. *Check the value/Q-function:* Print Q-values over training. Are they changing? If Q-values are constant, the gradient may be zero (check for dead ReLUs, vanishing gradients). If Q-values are exploding, the learning rate is too high or the target network is not being used.

4. *Simplify the environment:* Test on a trivial environment (e.g., a 4-state grid world) where you know the optimal policy. If the algorithm fails on a simple environment, the bug is in the algorithm implementation, not the problem complexity.

5. *Check the exploration:* Is ε > 0? Is the policy stochastic? Are all actions being tried? Print action histograms.

6. *Check the hyperparameters:* Learning rate, batch size, discount factor, replay buffer size, target update frequency. Try the hyperparameters from a known-working reference implementation.

7. *Check for implementation bugs:* Is the gradient computation correct? Is the loss being minimized (not maximized)? Are gradients flowing through the right parts of the computation graph (not through the target network)?

**Q22: "What is the difference between model-based and model-free RL? When would you use each?"**

*Model-free RL* (Q-learning, SARSA, policy gradient, SAC, PPO) learns a policy or value function directly from interaction data without building an explicit model of the environment dynamics.

*Model-based RL* learns a model of the environment's transition function P(s'|s,a) and reward function R(s,a), then uses this model for planning (DP, MCTS) or generating synthetic data (Dyna, dreamer).

Model-based is more sample-efficient (you learn a model and can "imagine" thousands of transitions without interacting with the real environment). But the model is an approximation, and errors in the model can compound over long planning horizons (model error compounding).

Use model-based when: data is very expensive (real-world robotics), the environment has structure that is easy to model (physics), or you need long-horizon planning. Use model-free when: the environment is complex and hard to model accurately (Atari games with complex visual dynamics), computation is cheap but interaction is fast (simulated environments), or you want simplicity and robustness.

Hybrid approaches (Dyna, MuZero, Dreamer) learn a model and use it to augment model-free learning, combining the strengths of both.

**Q23: "Explain how experience replay can introduce bias and how prioritized experience replay addresses sample efficiency but creates its own issues."**

Standard uniform experience replay introduces no statistical bias in Q-learning because Q-learning is off-policy and each transition (s,a,r,s') provides a valid Bellman update regardless of when or under what policy it was collected.

However, uniform sampling is wasteful: many transitions in the buffer are already well-learned (small TD error) and provide little gradient signal. Prioritized experience replay (PER) samples transitions with probability proportional to |δ|^α, focusing learning on "surprising" transitions.

**PER creates bias** because the non-uniform sampling changes the expected gradient. If we sample high-error transitions more often, the gradient is dominated by these transitions, which may not represent the true data distribution. This is corrected with importance sampling weights w_j = (1/(N·P(j)))^β, which down-weight over-sampled transitions.

**Additional PER issues:**
- Stale priorities: a transition's TD error may have been large when it was stored, but the Q-function has since improved. The priority is outdated until the transition is sampled again.
- Computational overhead: maintaining a sorted or heap-based priority queue adds O(log N) per insertion and sampling.
- Loss of diversity: if α is too large, the buffer becomes effectively much smaller (only high-error transitions are sampled), reducing the diversity benefit of replay.

**Q24: "In the context of safe RL, how would you ensure an autonomous vehicle does not take catastrophic actions during training?"**

Several approaches:

*Constrained MDP:* Add explicit constraints (e.g., collision probability < 0.001) to the optimization problem. Algorithms like Constrained Policy Optimization (CPO) handle this.

*Conservative policy initialization:* Start with a policy pre-trained via behavioral cloning on safe human demonstrations. RL fine-tuning uses the safe policy as a prior, with a KL penalty preventing large deviations.

*Safety layer:* Add a hard constraint layer on top of the policy that vetoes unsafe actions (e.g., actions that would violate a physics-based collision model). The policy proposes actions, and the safety layer projects them to the nearest safe action.

*Simulation-first:* Do all exploration in simulation. Transfer the trained policy to the real world with no further exploration (or with only conservative fine-tuning).

*Reward shaping for safety:* Add large negative rewards for unsafe states, but be careful — this can cause the agent to avoid even approaching useful but risky states.

*Human oversight:* Keep a human operator in the loop during training who can override dangerous actions and terminate episodes.

**Q25: "What are the key differences between TRPO and PPO, and why has PPO largely replaced TRPO in practice?"**

Both solve the same problem: preventing destructively large policy updates. TRPO enforces a hard KL divergence constraint using constrained optimization (conjugate gradient + line search). PPO approximates this with a clipped surrogate objective.

Key differences:
- *Implementation complexity:* TRPO requires computing the Fisher information matrix (or its inverse via conjugate gradient) and performing a line search. PPO uses standard first-order optimization (Adam) with a simple clip operation.
- *Computation per update:* TRPO is significantly more expensive per update due to the second-order optimization.
- *Tuning:* PPO has one key hyperparameter (clip range ε ≈ 0.2). TRPO has the KL constraint δ, which is harder to set because the relationship between KL divergence and policy change is problem-dependent.
- *Empirical performance:* PPO matches or exceeds TRPO's performance on most benchmarks while being simpler and faster.

PPO replaced TRPO because it achieves similar stability with dramatically simpler implementation and lower computational cost. The theoretical guarantees of TRPO (monotonic improvement) are stronger, but in practice, PPO's heuristic clipping works just as well.

---

**[← Previous Chapter: Recommender Systems](recommender_systems.md) | [Table of Contents](index.md) | [Next Chapter: Fine-Tuning & Alignment →](llm_finetuning_guide.md)**
