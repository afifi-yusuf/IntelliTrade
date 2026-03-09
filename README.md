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

## 7. Seeded Benchmark — DQN vs DQN No-LLM (reproducible comparison)

Runs 5 seeded training runs for both `dqn` and `dqn_nollm`, then evaluates all tickers on 2024-present data. This is how the architecture overhaul results are measured.

```bash
cd /Users/yusufafifi/Desktop/IntelliSense
venv/bin/python seeded_weekly_compare.py \
  --cutoff       2024-01-01 \
  --start-date   2024-01-01 \
  --epochs       50 \
  --seeds        11,22,33,44,55 \
  --stop-loss-pct 0.10 \
  --output-dir   test_results/seeded_weekly_full
```

What this does:
- Trains `dqn` (LLM structured signals) × 5 seeds, then `dqn_nollm` (price-only) × 5 seeds — 10 training runs total.
- Backtests every ticker on 2024-present (read-only, 10% stop-loss) after each run.
- Writes results to `test_results/seeded_weekly_full/`:
  - `seeded_weekly_runs.csv` — raw per-seed, per-ticker results
  - `seeded_weekly_summary.csv` — mean ± std per ticker
  - `seeded_weekly_delta.csv` — LLM vs no-LLM return delta per ticker
- **Auto-generates `backtest_analysis_seeded_weekly_compare.ipynb`** — open it in Jupyter to see the bar chart comparison (mean ± std by ticker).

> **This notebook shows the seeded bar chart only.** It is different from `backtest_analysis.ipynb`.

## 8. Rich Equity Curve Analysis (backtest_analysis.ipynb)

`backtest_analysis.ipynb` shows equity curves, trade markers, drawdown, alpha vs S&P 500, and Sharpe per ticker. It reads from `test_results/*.json` trade history files.

> **`seeded_weekly_compare.py` does NOT generate these files.** They must be created separately with the steps below.

To generate those files, run the compare command once per ticker:

```bash
cd /Users/yusufafifi/Desktop/IntelliSense
for TICKER in NVDA AAPL MSFT AMZN GOOGL META TSLA ASML TSM; do
  venv/bin/python run.py compare --ticker $TICKER --stop-loss-pct 0.10
done
```

This loads the trained models from `backtest_results/` and runs DQN vs DQN No-LLM for each ticker, saving a JSON trade history per method to `test_results/`. Then open `backtest_analysis.ipynb` in Jupyter.

## 9. Where Results Are Saved

| Path | Contents |
|------|----------|
| `backtest_results/` | Trained model checkpoints (`.pth`) |
| `backtest_results/seed_runs/` | Per-seed seeded benchmark models |
| `test_results/` | Backtest JSON histories + seeded CSVs |
| `backtest_analysis_seeded_weekly_compare.ipynb` | Auto-generated bar chart (seeded benchmark) |
| `backtest_analysis.ipynb` | Rich equity curve analysis (manual, needs JSON histories) |

## Notes

- `read_only=True` mode is used for safe evaluation/compare paths; it reads cache but does not write Neo4j.
- For first-time population on an empty DB, use `run.py backtest --method dqn ... --no-learning --no-save` as shown above.
- If you change prompt/schema/agent logic significantly, repopulate Neo4j before retraining.
- Old `.pth` checkpoints trained with the 414-dim architecture will not load into the current 30-dim network — re-train from scratch.
