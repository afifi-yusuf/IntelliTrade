# IntelliSense DQN Workflow (Neo4j Empty -> Populate -> Train -> Inference)

This README is a practical runbook for the **DQN (LLM-enabled)** path only.

## 1. Setup

```bash
cd /Users/yusufafifi/Desktop/IntelliSense
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in project root:

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

## 2. (Optional) Fetch / Refresh Market Data

If needed, update tickers in `data.py` and run:

```bash
venv/bin/python data.py
```

## 3. Start From Empty Neo4j

If your DB is empty, that is fine.  
You can verify counts in Neo4j Browser:

```cypher
MATCH (s:Summary) RETURN count(s);
MATCH (f:Fact) RETURN count(f);
```

## 4. Populate Neo4j Cache (DQN, No Learning)

Run one populate pass so quarterly summaries/facts are written to Neo4j.

```bash
cd /Users/yusufafifi/Desktop/IntelliSense
for t in MSFT AMZN TSLA TSM ASML NVDA GOOGL META AAPL; do
  venv/bin/python run.py backtest --method dqn --ticker "$t" --no-learning --no-save
done
```

What this does:
- Uses DQN with LLM/context path enabled.
- Writes graph cache (`Summary`, `Fact`, etc.) because backtest runs with write mode by default.
- Does **not** train or save model weights (`--no-learning --no-save`).

## 5. Train DQN (Walk-Forward Style)

Train on data before cutoff:

```bash
cd /Users/yusufafifi/Desktop/IntelliSense
venv/bin/python run.py train --method dqn --epochs 50 --cutoff 2024-01-01
```

Expected output checkpoint:

`/Users/yusufafifi/Desktop/IntelliSense/backtest_results/dqn_model_walkforward_final.pth`

## 6. Inference / Backtest DQN (No Learning)

Run inference from 2024 onward with trained model:

```bash
cd /Users/yusufafifi/Desktop/IntelliSense
venv/bin/python run.py backtest \
  --method dqn \
  --ticker NVDA \
  --no-learning \
  --load-model /Users/yusufafifi/Desktop/IntelliSense/backtest_results/dqn_model_walkforward_final.pth \
  --start-date 2024-01-01
```

## 7. Where Results Are Saved

Backtest artifacts are written under:

`/Users/yusufafifi/Desktop/IntelliSense/backtest_results/`

Comparison/test history files are written under:

`/Users/yusufafifi/Desktop/IntelliSense/test_results/`

## Notes

- `read_only=True` mode is used for safe evaluation/compare paths; it reads cache but does not write Neo4j.
- For first-time population on an empty DB, use `run.py backtest --method dqn ... --no-learning --no-save` as shown above.
- If you change prompt/schema/agent logic significantly, repopulate Neo4j before retraining.
