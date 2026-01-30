# 🤖 Deep Q-Network (DQN) Implementation Guide

## What is DQN?

**Deep Q-Network (DQN)** replaces the Q-table with a **neural network** that can:
- Handle continuous state spaces
- Learn complex non-linear patterns
- Scale to high-dimensional inputs
- Generalize across similar states

---

## 🏗️ Architecture Comparison

### Q-Learning (Tabular)
```
State: (Narrative=0, Financial=0)
    ↓
Q-Table Lookup
    ↓
{BUY: 0.9, HOLD: 0.1, SELL: -0.8}
```
**Limitations:**
- Only 6 discrete states
- Cannot handle nuances
- Manual feature engineering

### DQN (Deep Learning)
```
State Vector (396 dims):
  • Narrative embedding (384 dims from BERT)
  • Technical indicators (7 dims: momentum, RSI, etc.)
  • Narrative state (3 dims one-hot)
  • Financial state (2 dims one-hot)
    ↓
Neural Network (142,979 parameters):
  Input → Dense(256) → ReLU → Dropout
       → Dense(128) → ReLU → Dropout
       → Dense(64)  → ReLU → Dropout
       → Dense(3)   [BUY, HOLD, SELL]
    ↓
{BUY: 0.87, HOLD: 0.23, SELL: -0.45}
```
**Advantages:**
- Continuous state space
- Learns non-linear relationships
- Automatic feature learning
- Better generalization

---

## 🔑 Key Components

### 1. Neural Network Q-Function
```python
class DQNNetwork(nn.Module):
    def __init__(self, state_dim=396, action_dim=3):
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim)
        )
```

### 2. Experience Replay Buffer
```python
# Store experiences: (state, action, reward, next_state)
replay_buffer = deque(maxlen=10000)

# Sample random batch for training
batch = random.sample(replay_buffer, 32)
```
**Why?** Breaks correlation between consecutive samples, improves learning stability.

### 3. Target Network
```python
q_network = DQNNetwork()         # Main network (updated frequently)
target_network = DQNNetwork()    # Target network (updated slowly)

# Update target every N steps
if steps % 10 == 0:
    target_network.load_state_dict(q_network.state_dict())
```
**Why?** Provides stable targets for TD-learning, reduces oscillations.

### 4. Training Loop
```python
# For each trade:
1. Observe state
2. Select action (ε-greedy)
3. Execute trade → get reward
4. Store (state, action, reward, next_state) in buffer
5. Sample batch from buffer
6. Compute loss: MSE(Q(s,a), r + γ*max(Q(s',a')))
7. Backpropagate and update network
```

---

## 📊 State Vector Construction

**Total: 396 dimensions**

| Component | Dimensions | Source |
|-----------|------------|--------|
| **Narrative Embedding** | 384 | Sentence-BERT encoding of LLM analysis |
| **Technical Indicators** | 7 | Momentum, volatility, trend, RSI, price changes (1d, 5d, 20d) |
| **Narrative State (one-hot)** | 3 | [Bullish, Neutral, Bearish] |
| **Financial State (one-hot)** | 2 | [Good, Bad] |

```python
state_vector = np.concatenate([
    narrative_embedding,      # 384 dims
    [momentum, volatility, trend, rsi/100, 
     price_1d/100, price_5d/100, price_20d/100],  # 7 dims
    [1, 0, 0],  # Bullish (one-hot)
    [1, 0]      # Good financials (one-hot)
])
# Total: 396 dims
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Learning Rate (α)** | 0.0001 | How fast network learns (lower = more stable) |
| **Discount Factor (γ)** | 0.95 | Value of future rewards |
| **Epsilon (ε)** | 0.15 → 0.01 | Exploration rate (decays over time) |
| **Batch Size** | 32 | Samples per training step |
| **Buffer Size** | 10,000 | Max experiences stored |
| **Target Update Freq** | 10 steps | How often to update target network |

---

## 🚀 Usage

### Training from Scratch:
```python
from dqn_agent import DQNAgent

# Create DQN agent
dqn = DQNAgent(
    state_dim=396,
    learning_rate=0.0001,
    epsilon=0.2,  # High exploration during training
    enable_learning=True
)

# Run backtest (agent learns as it trades)
backtester = Backtester(ticker="NVDA", enable_learning=True)
backtester.rl_agent = dqn
metrics = backtester.run_backtest(financial_data)

# Save learned model
dqn.save_model('dqn_trained.pth')
```

### Using Pre-trained Model:
```python
# Load pre-trained DQN
dqn = DQNAgent(state_dim=396, enable_learning=False)
dqn.load_model('dqn_trained.pth')

# Deploy (no exploration)
dqn.epsilon = 0.0
backtester = Backtester(ticker="AAPL")
backtester.rl_agent = dqn
metrics = backtester.run_backtest(financial_data)
```

---

## 📈 Expected Performance

### Comparison (NVDA 2021-2026):

| Method | Final Value | Return | Sharpe | Max DD | Complexity |
|--------|-------------|--------|--------|--------|------------|
| **Fixed Q-table** | $161,825 | +1518% | 3.17 | 54.61% | 6 states |
| **Q-learning** | $131,016 | +1210% | 3.58 | 23.30% | 6 states |
| **DQN** | **TBD** | **TBD** | **TBD** | **TBD** | **142,979 params** |

---

## 🎯 Advantages of DQN

1. ✅ **Continuous State Space**: No manual discretization
2. ✅ **Automatic Feature Learning**: Network learns relevant patterns
3. ✅ **Non-Linear Relationships**: Can model complex interactions
4. ✅ **Generalization**: Similar states get similar Q-values
5. ✅ **Scalability**: Can handle high-dimensional inputs
6. ✅ **Transfer Learning**: Pre-train on one stock, deploy on another

---

## ⚠️ Challenges

1. **Data Hungry**: Needs more experiences than Q-learning
2. **Hyperparameter Sensitive**: Many knobs to tune
3. **Training Instability**: Can diverge if not careful
4. **Computational Cost**: Slower than Q-table lookup
5. **Black Box**: Harder to interpret than Q-table
6. **Overfitting Risk**: Can memorize training data

---

## 🔧 Troubleshooting

### Problem: DQN underperforms Q-learning
**Solutions:**
- Increase buffer size (more experiences)
- Lower learning rate (0.00001)
- Increase batch size (64 or 128)
- Pre-train on more historical data
- Reduce epsilon faster (more exploitation)

### Problem: Training is unstable
**Solutions:**
- Use gradient clipping (already implemented)
- Increase target network update frequency
- Add more dropout (reduce overfitting)
- Use Huber loss instead of MSE (already implemented)
- Normalize state inputs

### Problem: Too conservative (only HOLD)
**Solutions:**
- Adjust reward function (penalize HOLD less)
- Start with pre-trained Q-table values
- Use reward shaping
- Increase exploration (higher ε)

---

## 🚀 Future Enhancements

### 1. **Double DQN**
Reduces overestimation bias:
```python
# Standard DQN
target = r + γ * max(Q_target(s'))

# Double DQN
action = argmax(Q(s'))
target = r + γ * Q_target(s', action)
```

### 2. **Dueling DQN**
Separate value and advantage streams:
```python
V(s) = value network
A(s,a) = advantage network
Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
```

### 3. **Prioritized Experience Replay**
Sample important experiences more often:
```python
priority = |TD-error| + ε
p_sample = priority^α / Σ priority^α
```

### 4. **Multi-Stock Training**
Train on portfolio of stocks:
```python
for stock in ['NVDA', 'AAPL', 'MSFT', 'TSLA']:
    experiences = collect_experiences(stock)
    replay_buffer.extend(experiences)
train_on_batch(replay_buffer.sample(256))
```

---

## 📚 References

1. **DQN Paper**: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
2. **Double DQN**: van Hasselt et al. (2016)
3. **Dueling DQN**: Wang et al. (2016)
4. **Prioritized Experience Replay**: Schaul et al. (2016)

---

## 💡 Key Takeaway

**DQN trades Q-table simplicity for neural network power:**
- ✅ Better for complex, continuous state spaces
- ✅ Can learn subtle patterns Q-learning misses
- ⚠️ Needs more data and tuning
- ⚠️ More computationally expensive

**Best Practice**: Start with Q-learning for MVP, upgrade to DQN when you hit Q-table limitations!
