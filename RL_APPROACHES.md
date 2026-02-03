# Technical Deep Dive: RL Approaches in IntelliSense

This document details the three Reinforcement Learning strategies implemented in the IntelliSense trading agent. All approaches frame the trading problem as a **Markov Decision Process (MDP)** but differ significantly in how they represent the state space and derive the optimal policy $\pi^*$.

---

## 1. Fixed Q-Table (Heuristic Baseline)
**Implementation**: `MarkovRLAgent` (initialized with priors, learning disabled)

This is a **Rule-Based System** disguised as an RL agent. It uses a static look-up table where Q-values are hardcoded based on financial domain expertise ("Priors").

### State Space ($\mathcal{S}$): Discrete
The continuous world is discretized into a small, finite set of states:
- **Narrative State ($N$)**: $\{0: \text{Bullish}, 1: \text{Neutral}, 2: \text{Bearish}\}$ (derived from semantic embedding clusters)
- **Financial State ($F$)**: $\{0: \text{Healthy}, 1: \text{Weak}\}$ (derived from profit margins and earnings surprises)
- **Total States**: $3 \times 2 = 6$ unique states.

### Policy ($\pi$)
The policy is deterministic and static.
$$ a_t = \arg\max_{a} Q_{fixed}(s_t, a) $$

**Example Rule**:
- If $s_t = (\text{Bullish}, \text{Healthy})$, then $Q(\cdot) = \{\text{BUY}: 0.9, \text{HOLD}: 0.1, \text{SELL}: -0.8\}$. Decision: **BUY**.

### Theoretical Context
This approach has **High Bias** and **Zero Variance**. It cannot learn or adapt, but it is immune to overfitting. In low-data regimes (like quarterly trading), this often outperforms learning algorithms because it relies on "General Intelligence" (human expertise) rather than data-driven patterns.

---

## 2. Tabular Q-Learning
**Implementation**: `MarkovRLAgent` (learning enabled)

This is a **Model-Free Reinforcement Learning** algorithm. It starts with the same discrete state space as the Fixed Q-Table but updates its Q-values iteratively based on experience.

### State Space ($\mathcal{S}$): Discrete
Same as above ($6$ discrete states). Crucially, this means it **cannot distinguish** between "Slightly Bullish" and "Very Bullish"—both map to the same discrete bucket.

### Bellman Update Rule
After executing action $a_t$, transitioning to state $s_{t+1}$, and receiving reward $r_t$ (portfolio % return), the table is updated:

$$ Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right] $$

- $\alpha$ (Learning Rate): How fast old knowledge is overwritten.
- $\gamma$ (Discount Factor): Importance of future rewards.

### Exploration
Uses **$\epsilon$-greedy**:
- With probability $\epsilon$, choose a random action (Explore).
- With probability $1-\epsilon$, choose $\arg\max Q(s_t, a)$ (Exploit).

### Limitations
**State Aliasing**: Because it buckets complex continuous data into just 6 states, it loses massive amounts of information. The "Technical Context" (RSI, Moving Averages) allows for slight heuristic adjustments, but the core learning is limited to the coarse state definition.

---

## 3. Deep Q-Network (DQN)
**Implementation**: `dqn_agent.py`

This uses **Function Approximation** to handle a high-dimensional, continuous state space. Instead of a table, a Neural Network ($\theta$) estimates the Q-values: $Q(s, a; \theta) \approx Q^*(s, a)$.

### State Space ($\mathcal{S}$): Continuous & High-Dimensional
The state $s_t$ is a dense vector containing:
1.  **Narrative Embedding**: 384-dimensional vector from `all-MiniLM-L6-v2`. capturing semantic nuance.
2.  **Technical Indicators**: Continuous values for RSI, Momentum, Volatility, Trend (Normalized).
3.  **Financial State**: One-hot encoded vector.

$$ s_t \in \mathbb{R}^{384 + 7 + 2} \approx \mathbb{R}^{393} $$

### Architecture
- **Network**: Multi-Layer Perceptron (MLP).
  - Input: State Vector (393 dims)
  - Hidden: [256, 128, 64] with ReLU activation and Dropout.
  - Output: 3 Q-values (BUY, HOLD, SELL).
- **Target Network**: A copy of the main network ($\theta^-$) frozen for $N$ steps to stabilize the learning target $y_t$.

### Loss Function
Minimizes the Huber Loss between predicted Q-values and target Q-values:

$$ L(\theta) = \mathbb{E} \left[ \left( \overbrace{r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)}^{\text{Target}} - \underbrace{Q(s_t, a_t; \theta)}_{\text{Prediction}} \right)^2 \right] $$

### Key Mechanisms
1.  **Experience Replay**: Stores transitions $(s, a, r, s', d)$ in a buffer. Training samples random batches to break temporal correlations (which would otherwise destabilize the gradient descent).
2.  **Double Q-Learning (Implicit)**: By using a target network, it reduces the "Maximization Bias" where regular Q-learning tends to overestimate Q-values.

### Trade-offs
- **Pros**: Can learn subtle, non-linear patterns (e.g., "Buy only if Sentiment is bullish AND RSI < 30"). No information loss via discretization.
- **Cons**: Extremely **sample inefficient**. Neural networks typically need $10^4$ to $10^6$ samples to converge. With quarterly data, we have very few samples, leading to high variance and instability.

---

## Summary Comparison

| Feature | Fixed Q-Table | Tabular Q-Learning | Deep Q-Network (DQN) |
| :--- | :--- | :--- | :--- |
| **State Space** | Discrete (6 states) | Discrete (6 states) | Continuous (~400 dims) |
| **Policy** | Static / Expert Rules | Dynamic / Learned Table | Dynamic / Neural Network |
| **Data Efficiency** | High (Needs 0 data) | Medium (Needs repeated visits to states) | Low (Needs massive data) |
| **Nuance** | Low (Bucketed) | Low (Bucketed) | High (Raw Vectors) |
| **Best For** | Low-data regimes, establishing baselines | Simple environments with clear states | Rich environments with massive history |
