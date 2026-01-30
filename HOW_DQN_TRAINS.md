# 🧠 How DQN Training Works

## 🎯 Training Method: **ONLINE LEARNING**

The DQN is **NOT pre-trained**. It learns **during the backtest** by trading and receiving feedback.

---

## 📊 Training Flow (Step by Step)

### For Each Trading Day:

```
Day 1: May 3, 2021
├─ 1. GET CURRENT STATE
│  ├─ LLM embedding: [0.23, -0.45, 0.67, ...] (384 dims)
│  ├─ Technical: [momentum=5.2%, volatility=2.1%, RSI=62.5, ...]
│  ├─ Narrative: Bullish (one-hot: [1, 0, 0])
│  └─ Financial: Good (one-hot: [1, 0])
│  → State vector: 396 dimensions
│
├─ 2. SELECT ACTION (ε-greedy)
│  ├─ Forward pass through neural network
│  │  State → Network → [Q_BUY=0.12, Q_HOLD=-0.05, Q_SELL=-0.23]
│  ├─ 15% chance: Random action (explore)
│  └─ 85% chance: BUY (best Q-value)
│
├─ 3. EXECUTE TRADE
│  ├─ BUY 674 shares @ $14.80
│  ├─ Portfolio before: $10,000.00
│  └─ Portfolio after: $9,990.02
│
├─ 4. CALCULATE REWARD
│  └─ Reward = (9990.02 - 10000) / 10000 * 100 = -0.10%
│
├─ 5. STORE EXPERIENCE IN REPLAY BUFFER
│  └─ Buffer: [(state, action=BUY, reward=-0.10, next_state)]
│
└─ 6. TRAIN NEURAL NETWORK
   ├─ IF buffer has >= 32 experiences:
   │  ├─ Sample 32 random experiences from buffer
   │  ├─ Compute target: r + γ * max(Q_target(next_state))
   │  ├─ Compute loss: MSE(Q(state, action), target)
   │  ├─ Backpropagation
   │  └─ Update network weights
   │
   └─ Update target network every 10 steps

Day 2: May 10, 2021
└─ Repeat steps 1-6 with new state...
```

---

## 🔄 Detailed Training Process

### 1. **State Construction** (Line 290-292 in `dqn_agent.py`)
```python
state = get_state_vector(
    narrative_vector,  # 384 dims from LLM
    stock_features,    # 7 technical indicators
    narrative_state,   # 3 dims (Bullish/Neutral/Bearish)
    financial_state    # 2 dims (Good/Bad)
)
# Result: 396-dimensional vector
```

### 2. **Action Selection** (Lines 195-210)
```python
if random() < epsilon:
    action = random_choice([BUY, HOLD, SELL])  # Explore
else:
    q_values = neural_network(state)
    action = argmax(q_values)  # Exploit
```

### 3. **Experience Storage** (Lines 337-344)
```python
replay_buffer.push(
    state=current_state,
    action=action_idx,
    reward=portfolio_return,
    next_state=next_state,
    done=False
)
```

### 4. **Neural Network Training** (Lines 218-269)
```python
def train_step():
    # Sample random batch (breaks correlation)
    batch = replay_buffer.sample(32)
    
    # Forward pass: current Q-values
    current_Q = q_network(states)[actions]
    
    # Target Q-values (using TARGET network)
    with torch.no_grad():
        next_Q_max = target_network(next_states).max()
        target_Q = rewards + gamma * next_Q_max
    
    # Compute loss
    loss = MSE(current_Q, target_Q)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    # Update target network every 10 steps
    if steps % 10 == 0:
        target_network.copy(q_network)
```

---

## 📈 Training Timeline

```
Backtest Start (2021-01-15)
├─ Neural Network: Random weights (142,979 parameters)
├─ Replay Buffer: Empty (0 experiences)
└─ Epsilon: 0.15 (15% exploration)

Trade #1 (Day 1)
├─ Action: Random (exploring)
├─ Reward: -0.5%
└─ Buffer: 1 experience
   → NOT TRAINING YET (need 32 experiences)

Trade #10 (Day 10)
├─ Buffer: 10 experiences
└─ NOT TRAINING YET (need 32)

Trade #32 (Day 32) ← TRAINING STARTS!
├─ Buffer: 32 experiences
├─ Sample batch of 32
├─ Train network → Loss: 3.45
├─ Network weights updated!
└─ Epsilon: 0.149 (decaying)

Trade #50
├─ Buffer: 50 experiences
├─ Sample batch of 32 (random from all 50)
├─ Train network → Loss: 2.87 (improving!)
└─ Epsilon: 0.142

Trade #100
├─ Buffer: 100 experiences
├─ Sample batch of 32
├─ Train network → Loss: 1.92 (getting better!)
└─ Epsilon: 0.125

Trade #262 (End)
├─ Buffer: 262 experiences
├─ Total training steps: ~230
├─ Network: Learned optimal Q-function
├─ Epsilon: 0.08 (mostly exploiting now)
└─ Final loss: ~1.2
```

---

## 🎓 What DQN Learns From

### Training Data:
- **262 trading decisions** (weekly trades over 5 years)
- Each experience: `(state, action, reward, next_state)`

### Examples of Experiences:

**Experience #1:**
```python
state = [LLM_embedding_2021Q1, momentum=5.2%, RSI=62, Bullish, Good]
action = BUY
reward = +2.3%  (portfolio grew)
next_state = [LLM_embedding_2021Q1, momentum=4.8%, RSI=65, Bullish, Good]
```
**Network learns:** "In bullish states with good fundamentals, BUY is good!"

**Experience #47:**
```python
state = [LLM_embedding_2021Q4, momentum=-8.3%, RSI=35, Bearish, Bad]
action = SELL
reward = +1.2%  (avoided loss)
next_state = [LLM_embedding_2022Q1, momentum=-9.1%, RSI=32, Bearish, Bad]
```
**Network learns:** "In bearish states with bad fundamentals, SELL works!"

---

## 🔄 Why Experience Replay?

### Without Replay (Sequential Learning):
```
Trade #1 → Train on Trade #1
Trade #2 → Train on Trade #2  
Trade #3 → Train on Trade #3
```
**Problem:** Consecutive trades are highly correlated → unstable learning!

### With Replay (DQN Approach):
```
Trade #1 → Store in buffer
Trade #2 → Store in buffer
Trade #3 → Store in buffer
...
Trade #50 → Sample RANDOM batch [#3, #17, #42, #9, #31, ...]
         → Train on diverse experiences
```
**Benefit:** Breaks correlation → stable learning!

---

## 🎯 Training vs Pre-training

### Current Approach (Online Learning):
```
Start backtest (2021)
    ↓
Trade + Learn + Trade + Learn + Trade + Learn...
    ↓
End backtest (2026)
    ↓
Network is trained!
```

**Pros:**
- ✅ Simple implementation
- ✅ Learns on actual trading environment

**Cons:**
- ❌ Early trades are random (cold start)
- ❌ Exploration penalty during test period
- ❌ Can't evaluate true performance (training = testing)

### With Pre-training (Better Approach):
```
PHASE 1: TRAIN (2015-2020 data)
├─ Trade on old data
├─ Learn optimal policy
├─ Save trained network
└─ Result: Smart DQN

PHASE 2: TEST (2021-2026 data)
├─ Load trained network
├─ epsilon = 0 (no exploration!)
├─ Trade using learned policy
└─ Result: True performance evaluation
```

**Pros:**
- ✅ Network is smart from day 1
- ✅ No exploration penalty
- ✅ Proper train/test split
- ✅ Can evaluate generalization

---

## 📊 What DQN Learns

The neural network discovers:

1. **Non-linear patterns:**
   - "When RSI < 30 AND momentum > 5% AND LLM is bullish → BUY"
   - Q-learning can't learn this (only has discrete states)

2. **Feature interactions:**
   - "High volatility + bearish sentiment + good financials = ?"
   - Network learns complex relationships

3. **Generalization:**
   - Similar states get similar Q-values
   - Network interpolates between seen states

4. **Temporal patterns:**
   - Through reward propagation: `Q(s,a) = r + γ*max(Q(s',a'))`
   - Learns long-term consequences

---

## 🔢 Training Data Requirements

### Q-Learning (Tabular):
- **Minimum**: ~50-100 experiences per state
- **6 states** × 100 = 600 experiences
- ✅ Can work with 262 trades

### DQN (Neural Network):
- **Minimum**: ~1,000-10,000 experiences
- **Rule of thumb**: 10× more data than parameters
- 142,979 params → need ~1.4M experiences ideally
- ⚠️ **262 trades is NOT enough for optimal DQN!**

**This is why DQN might underperform Q-learning on limited data!**

---

## 💡 How to Fix Data Limitation

### Option 1: Train on Multiple Stocks
```python
for ticker in ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMD', 'INTC']:
    experiences = backtest(ticker)
    replay_buffer.extend(experiences)
# Now you have 262 × 6 = 1,572 experiences!
```

### Option 2: Data Augmentation
```python
# Add noise to historical states
for original_state in replay_buffer:
    for _ in range(5):
        noisy_state = original_state + noise(0.01)
        augmented_buffer.push(noisy_state)
# 262 × 5 = 1,310 experiences
```

### Option 3: Use Simpler Network
```python
# Reduce from 142,979 params to ~10,000 params
DQNNetwork(state_dim=396, hidden_dims=[64, 32])
# Needs less data!
```

### Option 4: Use More Historical Data
```python
# Get 20 years instead of 5
price_data = fetch_stock_price_history('NVDA', years=20)
# 262 × 4 = 1,048 trades
```

---

## 🎯 Bottom Line

**Current DQN Training:**
- ✅ Trains online during backtest
- ✅ Uses experience replay for stability
- ✅ Updates target network every 10 steps
- ⚠️ Only 262 experiences (too few for 142K params!)
- ⚠️ No train/test split (overfitting risk)

**Best Practice:**
1. Collect 1,000-10,000 experiences from multiple stocks/periods
2. Pre-train DQN on training set
3. Test on hold-out set with epsilon=0
4. Compare to Q-learning and fixed baselines

**For 262 trades, Q-learning is likely better than DQN!**

---

## 📁 Files to Check

- `dqn_agent.py` lines 218-269: Training logic
- `dqn_agent.py` lines 320-360: Update method (called after each trade)
- `backtest.py` lines 460-470: Integration with backtest loop
