# IntelliSense Project Design and Implementation

## 1) Objective

This project is a **context-enriched RL trading system**:

- ingest market, fundamental, transcript, and macro data,
- build a **Causal-Temporal Knowledge Graph (CTKG)** in Neo4j,
- run a **multi-agent LLM pipeline** to produce narrative + structured signals,
- embed that context and combine it with price/technical features,
- train a DQN policy on multi-ticker historical data,
- freeze the model and run inference from `2024-01-01` onward.

Current production comparison in `run.py` is:

- `dqn` (LLM + CTKG + price features)
- `dqn_nollm` (price-only DQN)


## 2) Universe and Time Setup

Ticker universe from `data.py`:

- `SPY, NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, ASML, TSM`

Backtest/training simulation behavior:

- price data is loaded as daily (fallback weekly),
- execution frequency defaults to **weekly** (first trading day of each ISO week),
- only information available as-of trade date is used (anti look-ahead filtering),
- quarterly context uses reporting-lag assumptions (about one month after quarter end),
- training set is pre-cutoff; inference in compare mode starts at `2024-01-01`.


## 3) Data Collection (Alpha Vantage) and Why

Data ingestion script: `data.py`

### 3.1 Price and market series

- Daily adjusted OHLCV (5y slice)
- Weekly adjusted OHLCV
- Monthly adjusted OHLCV
- Global quote snapshot

Why: price/return/volatility structure for RL technical features and execution.

### 3.2 Fundamental and earnings

- Company overview
- Income statement (annual + quarterly)
- Balance sheet (annual + quarterly)
- Cash flow (annual + quarterly)
- Shares outstanding
- Earnings annual + quarterly (incl. surprise fields)
- Earnings estimates
- Earnings calendar

Why: derive coarse financial health state and contextualize narrative signals.

### 3.3 Textual and event context

- Earnings call transcripts (`EARNINGS_CALL_TRANSCRIPT`, up to ~20 quarters)
- News sentiment
- Insider transactions

Why: primary source for LLM fact extraction and sentiment/context signals.

### 3.4 Technical indicators

- SMA/EMA, RSI/MACD/STOCH/CCI, BBANDS/ATR, ADX/AROON, OBV/AD, VWAP

Why: expose momentum, trend, and volatility proxies beyond raw prices.

### 3.5 Macro indicators

- Fed funds, 10Y treasury, GDP, GDP per capita, CPI, inflation, unemployment, nonfarm payroll, retail sales, durables

Why: macro regime context for transcript and earnings interpretation.


## 4) CTKG (Neo4j) Context Engineering

Graph manager: `graph_manager.py`

### 4.1 Core schema

Main nodes:

- `Ticker`, `Quarter`, `Fact`, `Metric`, `Value`, `Reason`
- `Transcript`, `EconomicContext`, `FinancialMetrics`, `Summary`, `Sector`

Important edges:

- `(:Ticker)-[:REPORTS_IN]->(:Quarter)`
- `(:Quarter)-[:HAS_FACT]->(:Fact)-[:FOR_FIRM]->(:Ticker)`
- `(:Fact)-[:HAS_METRIC]->(:Metric)`
- `(:Fact)-[:HAS_VALUE]->(:Value)`
- `(:Fact)-[:EXPLAINED_BY]->(:Reason)`
- `(:Ticker)-[:HAS_TRANSCRIPT]->(:Transcript)-[:IN_QUARTER]->(:Quarter)`
- `(:Quarter)-[:HAS_ECONOMIC_CONTEXT]->(:EconomicContext)`
- `(:Ticker)-[:HAS_METRICS]->(:FinancialMetrics)-[:IN_QUARTER]->(:Quarter)`
- `(:Ticker)-[:HAS_SUMMARY]->(:Summary)-[:IN_QUARTER]->(:Quarter)`
- `(:Ticker)-[:IN_SECTOR]->(:Sector)`
- `(:Quarter)-[:NEXT_QUARTER]->(:Quarter)`
- optional `(:Fact)-[:QoQ_COMPARES]->(:Fact)`

### 4.2 Quarterly structure

Quarter IDs are normalized like `YYYY-Qn` (example: `2024-Q3`).

Per quarter, the system can store:

- transcript text,
- macro snapshot,
- financial snapshot blob,
- extracted facts + embeddings,
- synthesized summary + embedding + structured signals + reliability metadata.

### 4.3 Vector index

Neo4j vector index on `Fact.embedding`:

- index name: `fact_embedding_index`
- similarity: cosine
- configured dimension: `384` (default embedding backend in this project setup)

Used for:

- peer fact retrieval (cross-company),
- historical fact retrieval (same company, prior quarters).


## 5) Multi-Agent LLM Pipeline

Pipeline orchestration: `llm.py` (`generate_agentic_analysis`)

Agents (folder: `agents`):

### 5.1 FactExtractorAgent

File: `agents/fact_extractor.py`

Prompt objective:

- extract discrete, explicitly stated financial facts from transcript.

Output schema (JSON array):

- `metric, value, change, sentiment, context, category`

Category set:

- `revenue, profitability, growth, risk, guidance, operational`

### 5.2 PeerAnalystAgent

File: `agents/peer_analyst.py`

Prompt objective:

- compare target quarter vs peers using graph-retrieved peer facts.

Output:

- analysis text + `PEER_SCORE: <float in [-1,1]>`

### 5.3 HistoricalAnalystAgent

File: `agents/historical_analyst.py`

Prompt objective:

- evaluate multi-quarter trajectory from graph trends + historical semantic retrieval.

Output:

- analysis text + `TRENDS: revenue=<float> margin=<float> momentum=<float>` in `[-1,1]`.

### 5.4 SynthesizerAgent

File: `agents/synthesizer.py`

Prompt objective:

- combine facts + peer + historical + macro into final strategic narrative.

Output:

- narrative (200-300 words),
- last line `SIGNALS:` with six structured signals:
  - `revenue_trend`
  - `margin_trend`
  - `peer_relative`
  - `historical_momentum`
  - `risk_level` (bounded [0,1] in code)
  - `overall_sentiment`

### 5.5 Reliability and delegation metadata

The orchestrator also builds:

- `delegation` map (fact category -> specialist routes),
- `reliability` bundle:
  - evidence count/strength,
  - agent agreement,
  - parse success,
  - per-agent confidence proxies.

These reliability features are passed to DQN as context quality signals.


## 6) Embedding and RL Feature Pipeline

Vectorizer: `vectorizer.py`

Backend priority:

1. SentenceTransformers (`all-MiniLM-L6-v2`, default, **384-dim**)
2. Azure embedding fallback (`1536-dim`)
3. TF-IDF fallback (`100-dim`)

DQN state size is `dqn_state_dim(embedding_dim)` in `vectorizer.py` (embedding + fixed scalar block `DQN_STATE_SCALAR_DIM = 30`).

- With MiniLM (**384**), `state_dim = 414`.

State composition (ordered):

1. Narrative embedding (`embedding_dim`)
2. Technical features (7)
3. Narrative one-hot state (3)
4. Financial one-hot state (2)
5. Position ratio (1)
6. Structured signals from LLM agents (6)
7. Signal deltas (6)
8. Context reliability meta (5)

Why weekly execution:

- quarterly/earnings context changes slowly,
- weekly cadence reduces turnover/noise vs daily,
- better aligns action timing with medium-horizon narrative signals.


## 7) Markov States and Transitions (Tabular RL Agent)

Agent: `rl_agent.py`

Discrete Markov state:

- `narrative_state ∈ {0,1,2}` = bullish/neutral/bearish
- `financial_state ∈ {0,1}` = good/weak
- combined key: `(narrative_state, financial_state)` -> 6 states.

Action set:

- `BUY`, `HOLD`, `SELL`

Transitions and learning:

- after each step, portfolio-value change produces reward,
- Bellman update:
  - `Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]`
- state transition is implicit through next `(narrative, financial)` pair.


## 8) DQN Design

Agent/network: `dqn_agent.py`

### 8.1 Action space and current position logic

- Action indices: `0=BUY, 1=HOLD, 2=SELL`
- Current implementation enforces **atomic long-only transitions**:
  - when `FLAT`: valid actions are `BUY/HOLD`
  - when `LONG`: valid actions are `HOLD/SELL`
- backtester applies an additional hard guard to enforce same rule.

### 8.2 Network architecture

`DQNNetwork` has two branches + gating:

1. **Price branch**
   - inputs: technical (7) + financial one-hot (2) + position ratio (1)
2. **Context branch**
   - inputs: narrative embedding + narrative one-hot + attended structured signals + context meta
3. **Signal attention**
   - MLP + softmax over 6 structured signals to learn per-signal weights
4. **Context gate**
   - `alpha = sigmoid(g(context_meta))`
   - fusion: `fused = price_latent + alpha * context_latent`
5. **Q head**
   - outputs Q-values for BUY/HOLD/SELL.

### 8.3 Confidence

For reporting/sizing compatibility, confidence is computed as:

- softmax probability of the chosen action from Q-values.


## 9) Training Protocol (50 Epochs, All Stocks)

Training entry: `train.py`

Configuration:

- one shared DQN across all tickers (not one model per ticker),
- default: `epochs=50`,
- time-series split via cutoff (default `2024-01-01`),
- each epoch shuffles tickers and runs a per-ticker backtest on **pre-cutoff** data only,
- replay buffer + target network + epsilon decay inside DQN.

Graph/cache behavior during training:

- When LLM is on: epoch 1 may write Neo4j cache; later epochs use `read_only` so reused summaries do not re-hit the graph writers.
- In-memory `summary_cache` persists by ticker across epochs.

Typical command:

```bash
python3 run.py train --method dqn --epochs 50 --cutoff 2024-01-01
```


## 10) Frozen Inference from 2024-01-01 to Present

Inference mode:

- load trained model checkpoint,
- disable learning (`--no-learning` or compare path default),
- evaluate from `start_date=2024-01-01` to end of available local price data.

Single-method inference:

```bash
python3 run.py backtest \
  --method dqn \
  --ticker NVDA \
  --no-learning \
  --load-model backtest_results/dqn_model_walkforward_final.pth \
  --start-date 2024-01-01
```

Comparison (LLM DQN vs no-LLM DQN):

```bash
python3 run.py compare --ticker NVDA
```


## 11) End-to-End Operational Flow

1. Collect/update raw data from Alpha Vantage into `data/` (or use existing CSVs).
2. Ensure Neo4j is running and reachable via env vars.
3. (Optional but recommended) warm context cache by running LLM-mode backtests.
4. Train `dqn` for 50 epochs with cutoff split.
5. Freeze model and run post-`2024-01-01` inference.
6. Compare against `dqn_nollm` to isolate LLM/CTKG value-add.

Exploratory plots and seeded multi-ticker summaries: Jupyter notebooks under `notebooks/` (see root `README.md`).


## 12) What Is Currently Implemented vs Not

Implemented now:

- CTKG schema + vector search integration,
- multi-agent LLM analysis with structured 6-signal output,
- signal-attention DQN with context gating,
- weekly backtest with point-in-time safeguards,
- `dqn` and `dqn_nollm` training/inference paths.

Not currently implemented in active code path:

- production short-selling state machine in DQN/backtest (current path is long-only atomic transitions).
