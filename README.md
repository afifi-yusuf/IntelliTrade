# 🧠 IntelliSense - Context Enriched Agentic RAG Trading System

> Combines Large Language Models, Knowledge Graphs (Neo4j), and Reinforcement Learning for intelligent stock trading

## 🎯 Overview

IntelliSense is an AI-powered trading system that:
1. **Collects** comprehensive financial data (fundamentals, technicals, earnings transcripts, macroeconomics)
2. **Analyzes** using LLMs (Azure OpenAI) with Neo4j knowledge graphs for historical context
3. **Learns** trading strategies using Reinforcement Learning (Q-learning or Deep Q-Networks)
4. **Backtests** on historical data to evaluate performance

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA COLLECTION                             │
│  Alpha Vantage API → CSV Files (fundamentals, technicals, etc.) │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   KNOWLEDGE GRAPH (Neo4j)                        │
│  • Store quarterly contexts                                      │
│  • Build historical context chains                               │
│  • Cache LLM analyses per quarter                                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    LLM ANALYSIS (Azure OpenAI)                   │
│  Input: Comprehensive Context                                    │
│  • Company financials (revenue, EPS, margins)                    │
│  • Earnings transcripts (quarterly)                              │
│  • Technical indicators (RSI, MACD, moving averages)             │
│  • Macroeconomic data (CPI, GDP, unemployment)                   │
│  • Historical context from Neo4j                                 │
│                                                                   │
│  Output: Strategic Analysis (2-3 paragraphs)                     │
│  • Sentiment, growth outlook, risks, catalysts                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    VECTORIZATION                                 │
│  • Embedding: Hugging Face (all-MiniLM-L6-v2, 384 dims)         │
│  • Sentiment: Zero-shot classification (Bullish/Neutral/Bearish) │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│              REINFORCEMENT LEARNING AGENT                        │
│  State: [LLM Embedding (384) + Stock Features (7) +             │
│          Narrative State (3) + Financial State (2)]              │
│                                                                   │
│  Actions: BUY, HOLD, SELL                                        │
│                                                                   │
│  Methods:                                                        │
│  1. Fixed Q-table (baseline, no learning)                        │
│  2. Q-learning (tabular RL, learns Q-values)                     │
│  3. Deep Q-Network (neural network approximation)                │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     BACKTESTING                                  │
│  • Simulate trades on historical data (5 years)                  │
│  • Calculate metrics (return, Sharpe ratio, max drawdown)        │
│  • Online learning: RL agent learns during backtest              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repo
git clone <your-repo-url>
cd IntelliSense

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```bash
# Azure OpenAI (for LLM analysis)
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Neo4j (for knowledge graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Alpha Vantage (for data collection)
ALPHA_VANTAGE_API_KEY=your_api_key
```

### 3. Run!

```bash
# Compare all methods (Fixed Q-table, Q-learning, DQN)
./venv/bin/python run.py compare --ticker NVDA

# Or run individual backtest
./venv/bin/python run.py backtest --method qlearning --ticker NVDA

# Or collect new data first
./venv/bin/python run.py collect --ticker AAPL
```

---

## 📊 Results (NVIDIA, 2021-2026)

| Method | Final Value | Total Return | Sharpe Ratio | Win Rate | Complexity |
|--------|-------------|--------------|--------------|----------|------------|
| **Fixed Q-table** | $161,825 | +1,518% | 3.17 | 54.61% | 6 states |
| **Q-learning** | $50,747 | +407% | 2.78 | 45.46% | 6 states |
| **DQN** | $26,027 | +160% | 1.72 | 31.92% | 142,979 weights |
| **Buy & Hold** | ~$200,000 | +1,900% | ~3.5 | N/A | Baseline |

### Insights:
- ✅ **Fixed Q-table won** - but it's "lucky" (tuned for NVDA's bull run)
- ✅ **Q-learning works well** with limited data (262 trades)
- ❌ **DQN overfits** - 142K parameters but only 262 experiences (needs 10× more data)

**Recommendation:** Use Q-learning for single-stock, limited data scenarios.

---

## 🛠️ Usage

### Single Command Interface

All operations go through `run.py`:

```bash
# Show help
python run.py --help

# Available commands:
#   collect   - Collect data from Alpha Vantage
#   rag       - Run RAG pipeline (LLM analysis only)
#   backtest  - Backtest a single method
#   compare   - Compare all three methods
#   quickstart - Collect data + compare (all-in-one)
```

### Examples

#### 1. Collect Data

```bash
python run.py collect --ticker NVDA --api-key YOUR_KEY
```

This downloads:
- Company overview
- Income statements, balance sheets, cash flows
- Earnings transcripts (quarterly)
- Daily stock prices (5 years)
- Technical indicators (RSI, MACD, SMA, EMA, etc.)
- Macroeconomic data (CPI, GDP, unemployment, etc.)

#### 2. Run Backtest

```bash
# Q-learning (recommended)
python run.py backtest --method qlearning --ticker NVDA

# DQN (experimental)
python run.py backtest --method dqn --ticker NVDA --learning-rate 0.001

# Fixed Q-table (baseline)
python run.py backtest --method fixed --ticker NVDA
```

#### 3. Compare All Methods

```bash
python run.py compare --ticker NVDA --capital 10000
```

Output:
```
Method               | Final Value     | Return       | Sharpe   | Win Rate | Complexity
---------------------|-----------------|--------------|----------|----------|---------------
Fixed Q-table        | $    161,825.14 |     1518.25% |     3.17 |   54.61% | 6 states
Q-learning           | $     50,747.09 |      407.47% |     2.78 |   45.46% | 6 states
DQN                  | $     26,026.77 |      160.27% |     1.72 |   31.92% | 142,979 weights

🏆 BEST METHOD (by Sharpe Ratio): Fixed Q-table
```

#### 4. Adjust Hyperparameters

```bash
# Higher exploration for Q-learning
python run.py backtest --method qlearning --epsilon 0.2 --learning-rate 0.15

# Lower learning rate for DQN
python run.py backtest --method dqn --learning-rate 0.0005 --epsilon 0.05
```

---

## 📁 Project Structure

```
IntelliSense/
├── run.py                 # 🎯 MAIN ENTRY POINT (use this!)
├── README.md              # This file
├── requirements.txt       # Python dependencies
│
├── Core Modules:
│   ├── main.py            # LLM RAG pipeline
│   ├── backtest.py        # Backtesting framework
│   ├── rl_agent.py        # Q-learning agent
│   ├── dqn_agent.py       # Deep Q-Network agent
│   ├── vectorizer.py      # Text embedding & sentiment
│   ├── graph_manager.py   # Neo4j knowledge graph
│   ├── llm_parser.py      # Enhanced state extraction (NEW)
│   └── data.py            # Alpha Vantage data collection
│
├── Data:
│   ├── data/              # CSV files (stock data, financials, etc.)
│   └── backtest_results/  # Saved models, Q-tables, trade logs
│
└── Old Scripts (deprecated):
    ├── compare_all_methods.py
    ├── run_backtest_with_learning.py
    └── run_dqn_comparison.py
```

**Use only `run.py` going forward!**

---

## 🧠 Reinforcement Learning Methods

### 1. Fixed Q-table (Baseline)

**How it works:**
- Hardcoded Q-values based on heuristics
- No learning during backtest
- Fast but inflexible

**State Space:** 6 states
- Narrative: Bullish/Neutral/Bearish (3)
- Financial: Strong/Weak (2)

**Performance:** Good on NVDA (lucky), but won't generalize

---

### 2. Q-learning (Recommended)

**How it works:**
- Tabular RL: Learns Q-values for each (state, action) pair
- Updates after each trade using reward signal
- Epsilon-greedy exploration

**Update Rule:**
```python
Q(s, a) = Q(s, a) + α * [reward + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `α` = learning rate (0.1)
- `γ` = discount factor (0.95)
- `reward` = profit/loss from trade

**State Space:** 6 states (same as fixed)

**Performance:** Works well with limited data (262 trades)

**Hyperparameters:**
- `learning_rate`: 0.05-0.2 (higher = faster learning but less stable)
- `epsilon`: 0.05-0.2 (higher = more exploration)
- `discount_factor`: 0.9-0.99 (higher = more long-term focus)

---

### 3. Deep Q-Network (DQN)

**How it works:**
- Neural network approximates Q-function
- Experience replay buffer (10,000 experiences)
- Target network (updated every 10 steps)
- Handles continuous state spaces

**Network Architecture:**
```
Input (396 dims) → [256] → [128] → [64] → Output (3 actions)
Total parameters: 142,979
```

**State Vector:**
- LLM Embedding: 384 dims
- Stock Features: 7 dims (momentum, volatility, RSI, etc.)
- Narrative State: 3 dims (one-hot)
- Financial State: 2 dims (one-hot)

**Performance:** Underperforms due to overfitting (too many params, not enough data)

**Improvements Needed:**
- Reduce network size: `[64, 32]` → ~5K params
- Train on multiple stocks (6× more data)
- Use enhanced state (27 dims instead of 396)

---

## 🎛️ Configuration Options

### Hyperparameters

| Parameter | Q-learning | DQN | Description |
|-----------|------------|-----|-------------|
| `learning_rate` | 0.1 | 0.001 | How fast to update Q-values |
| `discount_factor` | 0.95 | 0.95 | Future reward weighting |
| `epsilon` | 0.1 | 0.1 | Exploration rate (0=greedy, 1=random) |
| `epsilon_decay` | N/A | 0.995 | Decay exploration over time |
| `batch_size` | N/A | 64 | DQN training batch size |
| `buffer_size` | N/A | 10000 | Experience replay capacity |

### Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | $10,000 | Starting cash |
| `transaction_cost` | 0.1% | Per-trade fee |
| `position_size` | 100% | All-in/all-out (future: variable) |

---

## 📈 Performance Metrics

The system calculates:

- **Total Return**: `(final_value - initial_capital) / initial_capital`
- **Annualized Return**: `(final_value / initial_capital)^(252/days) - 1`
- **Sharpe Ratio**: `mean(daily_returns) / std(daily_returns) * sqrt(252)`
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: `winning_trades / total_trades`
- **Average Trade**: Mean profit/loss per trade

---

## 🔧 Advanced: Enhanced State Representation

**Problem:** Current state uses simple cosine similarity, collapsing multi-dimensional LLM analysis into a single "Bullish/Neutral/Bearish" bucket.

**Solution:** Extract structured features from LLM output (see `llm_parser.py`)

### Current State (Oversimplified):
```
LLM: "Strong growth BUT expensive, bearish ST, bullish LT"
→ Cosine Similarity: "Bullish"
→ State: 1 dimension ❌
```

### Enhanced State (Multi-Dimensional):
```
LLM: (same text)
→ Structured Parsing:
   • Sentiment: bullish (confidence: 0.85)
   • Growth Outlook: 0.75
   • Valuation Concern: 0.80 (HIGH!)
   • Short-term: bearish
   • Long-term: bullish
   • Competition Risk: 0.70
   • Risk/Reward: -0.05 (negative!)
→ State: 18 dimensions ✅
```

**Expected Impact:**
- Q-learning: +10-20% better returns
- DQN: +30-50% better returns (less overfitting)

**Status:** Implemented in `llm_parser.py`, not yet integrated into backtest.

---

## 🐛 Known Issues & Limitations

### Current Limitations:
1. **DQN overfits** - 142K params for 262 experiences
2. **All-in/all-out trades** - no position sizing
3. **Single stock only** - no portfolio diversification
4. **No stop-loss** - can have large drawdowns
5. **Simplified state** - losing LLM nuance (cosine similarity)

### Future Improvements:
1. ✅ **Enhanced state** (18 dims) - implemented, needs integration
2. 📝 **Position sizing** based on confidence
3. 📝 **Multi-stock training** for DQN
4. 📝 **Risk management** (stop-loss, take-profit)
5. 📝 **Train/test split** for proper evaluation

---

## 🔬 Experiments & Analysis

### Test: Impact of Exploration Rate

```bash
# Low exploration (ε=0.05) - more greedy
python run.py backtest --method qlearning --epsilon 0.05

# High exploration (ε=0.2) - more random
python run.py backtest --method qlearning --epsilon 0.2
```

**Hypothesis:** Higher epsilon → more exploration → better long-term learning

### Test: Learning Rate Sensitivity

```bash
# Conservative (α=0.05)
python run.py backtest --method qlearning --learning-rate 0.05

# Aggressive (α=0.2)
python run.py backtest --method qlearning --learning-rate 0.2
```

**Hypothesis:** Higher α → faster learning but less stable

---

## 📚 Technical Details

### LLM Context Construction

For each quarter, the system builds a comprehensive context:

```python
context = f"""
## COMPANY OVERVIEW
{company_name} ({ticker})
Sector: {sector} | Industry: {industry}

## FINANCIAL PERFORMANCE
Revenue (TTM): ${revenue}M ({revenue_growth}% YoY)
Net Income: ${net_income}M (Margin: {margin}%)
EPS: ${eps} (Surprise: {surprise}%)

## TECHNICAL INDICATORS
RSI: {rsi} | MACD: {macd}
Price: ${price} (20d: +{change_20d}%)

## MACROECONOMIC ENVIRONMENT
GDP Growth: {gdp}% | Unemployment: {unemployment}%
CPI: {cpi}% | Fed Funds Rate: {fed_rate}%

## EARNINGS TRANSCRIPT (Q{quarter})
{transcript_text}

## HISTORICAL CONTEXT
{previous_quarter_analysis_from_neo4j}
"""
```

This is sent to Azure OpenAI (GPT-4) which returns a 2-3 paragraph strategic analysis.

### Reward Function

```python
def calculate_reward(prev_value, curr_value, action):
    # Percentage return
    reward = (curr_value - prev_value) / prev_value * 100
    
    # Penalize transaction costs
    if action in ['BUY', 'SELL']:
        reward -= 0.1  # 0.1% fee
    
    # Clip to prevent extreme values
    reward = np.clip(reward, -10, 10)
    
    return reward
```

### Q-Learning Update

```python
# After each trade:
reward = calculate_reward(prev_portfolio, curr_portfolio, action)
next_max_q = max(q_values[next_state])

# Bellman equation
q_values[state][action] += learning_rate * (
    reward + discount_factor * next_max_q - q_values[state][action]
)
```

---

## 🤝 Contributing

This is a research project. Key areas for contribution:

1. **Enhanced state integration** - Use `llm_parser.py` in backtest
2. **Multi-stock portfolio** - Extend to multiple tickers
3. **Better DQN architecture** - Reduce params, avoid overfitting
4. **Position sizing** - Variable allocation based on confidence
5. **Risk management** - Stop-loss, take-profit logic

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- **Alpha Vantage** - Financial data API
- **Azure OpenAI** - GPT-4 for strategic analysis
- **Neo4j** - Knowledge graph for historical context
- **Hugging Face** - Sentence transformers for embeddings

---

## 📞 Support

For questions or issues:
1. Check this README
2. Review code comments in `run.py`, `backtest.py`, `rl_agent.py`
3. Open an issue on GitHub

---

**Built with ❤️ using LLMs, Knowledge Graphs, and Reinforcement Learning**
