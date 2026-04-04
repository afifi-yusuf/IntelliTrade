# IntelliSense

Context-enriched RL trading: local CSVs + **Neo4j** (causal-temporal graph) + multi-agent **LLM** signals, fused with price features into **DQN** policies. Two trained variants: **`dqn`** (full context) and **`dqn_nollm`** (price-only).

Full architecture, schema, and training details: **[`PROJECT_DESIGN_IMPLEMENTATION.md`](PROJECT_DESIGN_IMPLEMENTATION.md)**.

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create **`.env`** at the repo root. Typical names: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_DEPLOYMENT` (see `llm.py`), `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `ALPHA_VANTAGE_API_KEY` (see `data.py`).

---

## Data

Tickers live in **`data.py`**. Refresh CSVs under **`data/`** when needed:

```bash
python data.py
```

---

## CLI (`run.py`)

All commands assume the **repository root** as the working directory.

| Command | Purpose |
|--------|---------|
| `python run.py train --method dqn --epochs 50 --cutoff 2024-01-01` | Walk-forward-style training (shared DQN across tickers) |
| `python run.py train --method dqn_nollm ...` | Price-only DQN |
| `python run.py backtest --method dqn --ticker NVDA --no-learning --load-model backtest_results/dqn_model_walkforward_final.pth --start-date 2024-01-01` | Frozen inference |
| `python run.py compare --ticker NVDA` | Compare `dqn` vs `dqn_nollm` |

**Neo4j:** first-time graph population often uses LLM-mode backtest **without** saving a model, e.g. `python run.py backtest --method dqn --ticker AAPL --no-learning --no-save` (repeat per ticker if needed). Evaluation paths use read-only graph access where configured.

---

## Analysis notebooks

Jupyter notebooks live in **`notebooks/`**:

- `backtest_analysis.ipynb` — multi-method comparisons  
- `backtest_analysis_daily.ipynb`, `backtest_analysis_daily_dqn_nollm.ipynb` — daily cadence variants  
- `backtest_analysis_dqn_nollm.ipynb` — no-LLM analysis  
- `backtest_analysis_seeded_weekly_compare.ipynb` — seeded weekly DQN vs no-LLM vs buy-and-hold (uses **`seeded_weekly_report.py`** + CSV under **`test_results/seeded_weekly_full/`**)

The **first code cell** in each notebook switches to the repo root when the kernel’s working directory is `notebooks/`, so paths like `data/` and `test_results/` keep working.

---

## Seeded weekly benchmark (optional)

Multi-seed train + backtest, aggregate CSVs, and a minimal generated notebook:

```bash
python seeded_weekly_compare.py --output-dir test_results/seeded_weekly_full
```

Rich tables and plots: open **`notebooks/backtest_analysis_seeded_weekly_compare.ipynb`**. Editable summary numbers: **`test_results/seeded_weekly_full/seeded_weekly_async_by_ticker.csv`**.

---

## Outputs

| Location | Contents |
|----------|-----------|
| **`backtest_results/`** | Checkpoints (`*_walkforward_final.pth`), Q-tables, seed copies |
| **`test_results/`** | Run logs, histories, seeded summaries, generated `seeded_weekly_generated.ipynb` |

---

## Shared Python utilities

- **`reproducibility.py`** — `set_random_seed` for training/benchmarks  
- **`vectorizer.py`** — `dqn_state_dim()` / `DQN_STATE_SCALAR_DIM` (single source for DQN input size)  
- **`seeded_weekly_report.py`** — helpers for the seeded weekly notebook  
