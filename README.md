# IntelliSense — Context‑Enriched RAG Trading

Research prototype that combines LLM analysis, a Neo4j knowledge graph, and RL‑based trading signals. It collects fundamentals, transcripts, technicals, and macro data; generates a quarterly narrative; embeds it; and runs a backtest using fixed Q‑table, Q‑learning, or DQN.

## Core Workflow
1. **Data ingestion** → CSVs (prices, financials, transcripts, macro).  
2. **Context building** → quarterly context + historical graph context.  
3. **LLM analysis** → 1–2 paragraph strategic summary.  
4. **Embedding** → `all-MiniLM-L6-v2` (384 dims).  
5. **RL policy** → BUY / HOLD / SELL.  
6. **Backtest** → portfolio simulation + metrics.

## Research Notes
- Look‑ahead controls: price history filtered by `current_date`; quarterly data gated by reporting lag; graph context filtered by `max_quarter`; cache validated by generation date.
- Economic CSVs are filtered to `date <= current_date` but lack release‑calendar metadata.
- Results are **in‑sample** unless you add a walk‑forward split.

## Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Fetching
To collect data for specific tickers, edit the `TICKERS` list in `data.py` and run:
```bash
python data.py
```
This will fetch daily/weekly prices, financials, and transcripts for all configured tickers (e.g., NVDA, AAPL, MSFT, ...).
**Note:** This requires an Alpha Vantage Premium key.

Create `.env`:
```bash
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_OPENAI_API_VERSION=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
ALPHA_VANTAGE_API_KEY=...
```

## Usage
```bash
# Compare all methods
python run.py compare --ticker NVDA

# Single method
python run.py backtest --method qlearning --ticker NVDA

# RAG only
python run.py rag --ticker NVDA
```

## Outputs
Backtests write to `backtest_results/`:
- `trade_log.csv`, `equity_curve.csv`, `performance_summary.txt`
- saved Q‑tables or DQN checkpoints

## Methods
- **Fixed Q‑table**: heuristic baseline.  
- **Q‑learning**: tabular RL, coarse state.  
- **DQN**: neural approximation, higher variance and data‑hungry.

## Limitations (for research rigor)
- No walk‑forward validation or multi‑asset splits.  
- Macro data lacks release calendars and revisions.  
- **DQN**: neural approximation, sensitive to data volume.

## SOTA DQN Workflow (Walk-Forward)
To train a robust agent without data leakage:

1. **Train on Past (2020-2023)**:
```bash
python train_dqn.py --epochs 50 --cutoff 2024-01-01
```
This saves the "brain" to `backtest_results/dqn_model_walkforward_final.pth`.

2. **Test on Future (2024-Present)**:
```bash
python run.py backtest --method dqn --ticker NVDA --no-learning --load-model backtest_results/dqn_model_walkforward_final.pth --start-date 2024-01-01
```

## Limitations (for research rigor)

## Project Map
```
run.py            # CLI entry point
backtest.py       # simulation engine
main.py           # RAG pipeline
graph_manager.py  # Neo4j context/cache
rl_agent.py       # Q-learning
dqn_agent.py      # DQN
vectorizer.py     # embeddings/sentiment
data.py           # Alpha Vantage collection
data/             # CSVs
```

### Customizing Tickers
To run the RAG pipeline (`main.py`) for different tickers, update the `tickers` list in `main.py`:
```python
tickers = ["NVDA", "AAPL", "MSFT", ...]
```
Ensure you have fetched data for these tickers first using `data.py`.
