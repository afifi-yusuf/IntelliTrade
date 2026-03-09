# DQN Architecture Overhaul

## Summary

The 384-dim sentence embedding was removed from the DQN state vector and the network was simplified to a single-branch MLP. Double DQN and signal staleness decay were added. The result is a 14× smaller state (414 → 30 dims) with cleaner, more learnable features — while the full LLM multi-agent pipeline remains in place and its outputs continue to feed the agent as compact 6-dim structured signals.

---

## Motivation

The original DQN with LLM produced returns nearly identical to the no-LLM variant (avg 47.8% vs 48.2% across 9 tickers, 5 seeded runs each). Three architectural problems explained this:

1. **Embedding dominated the state.** The 384-dim `all-MiniLM-L6-v2` embedding made up 93% of the 414-dim state vector. It is a general-purpose sentence encoder trained for semantic similarity, not financial direction — so the 12 LLM-derived structured signal features were buried in noise.

2. **Temporal frequency mismatch.** The embedding updates quarterly (on new earnings). Between earnings, all 384+12 LLM dims are frozen constants. Only the 7 technical features changed week-to-week, meaning the "context" branch contributed a static bias rather than dynamic signal.

3. **Context gate couldn't save it.** The gate was conditioned on `context_meta` (5 dims), which was all-zeros in no-LLM mode — giving a fixed 50% blend regardless of staleness.

---

## Changes

### `dqn_agent.py`

#### State vector — 414 → 30 dims

| Component | Old | New | Notes |
|-----------|-----|-----|-------|
| Narrative embedding | 384 | **removed** | Was `all-MiniLM-L6-v2` output |
| Technical features | 7 | 7 | unchanged |
| Narrative one-hot | 3 | 3 | unchanged |
| Financial one-hot | 2 | 2 | unchanged |
| Position ratio | 1 | 1 | unchanged |
| Structured signals | 6 | 6 | LLM synthesizer outputs (with staleness decay) |
| Signal deltas | 6 | 6 | Quarter-over-quarter changes (with staleness decay) |
| Context meta | 5 | 5 | days_since_report, is_new_quarter, agreement, evidence_strength, parse_success |
| **Total** | **414** | **30** | |

`narrative_vector` is kept as a silently-ignored keyword argument on `get_state_vector()` and `get_signal()` for call-site backward compatibility.

#### Network architecture

Replaced the two-branch (price + context) architecture with a **single-branch MLP**:

```
Old:  price_branch(10) + context_branch(404) + signal_attention + context_gate → q_head  (~50K params)
New:  Linear(30,128) → ReLU → Dropout(0.1) → Linear(128,128) → ReLU → Dropout(0.1) → Linear(128,64) → ReLU → Linear(64,3)  (28,931 params)
```

Removed: `signal_attention`, `price_branch`, `context_branch`, `context_gate`, all state-slicing logic in `forward()`.

#### Double DQN

`train_step()` was upgraded from vanilla DQN to Double DQN:

```python
# Before (vanilla DQN — overestimates Q-values):
next_q_values = self.target_network(next_states).max(1)[0]

# After (Double DQN — online net selects, target net evaluates):
best_actions = self.q_network(next_states).argmax(1)
next_q_values = self.target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
```

#### Constructor

`embedding_dim` removed from `__init__` (kept as ignored kwarg). `state_dim` is now always `STATE_DIM = 30`.

---

### `backtest.py`

#### Signal staleness decay

Structured signals and their deltas are multiplied by an exponential decay factor before being placed in the state vector. This naturally transitions the agent toward price-only decisions as earnings age:

```python
days_since_report = max(0.0, (date - report_date).days)
staleness_decay = exp(-days_since_report / 60.0)   # τ = 60 days, half-life ≈ 42 days
structured_signals      = structured_signals      * staleness_decay
structured_signal_deltas = structured_signal_deltas * staleness_decay
```

| Days after earnings | Decay factor |
|---------------------|-------------|
| 0 (fresh) | 1.00 |
| 14 | 0.79 |
| 42 | 0.50 |
| 90 | 0.22 |
| 120 | 0.14 |

#### Embedding no longer passed to the agent

`narrative_vector=embedding` removed from `signal_kwargs` in both the LLM and no-LLM paths. The embedding is still generated and stored in Neo4j for RAG retrieval — it is just no longer part of the RL state.

---

### `train.py`

```python
# Before:
state_dim = vectorizer.embedding_dim + 7 + 3 + 2 + 1 + 6 + 6 + 5  # 484 or 414 depending on model

# After:
state_dim = 7 + 3 + 2 + 1 + 6 + 6 + 5  # = 30
```

`embedding_dim=vectorizer.embedding_dim` removed from the `DQNAgent()` constructor call.

### `run.py`

Same `state_dim` and constructor fix as `train.py`.

---

## What did NOT change

- The full LLM multi-agent pipeline (`llm.py`, `agents/`) — still runs, still produces structured signals
- Neo4j graph (`graph_manager.py`) — still caches embeddings and facts for RAG retrieval
- The vectorizer — still used for narrative-state classification (`get_narrative_state()`) and Neo4j storage
- The backtest engine, portfolio logic, transaction costs, stop-loss, reward shaping — all unchanged
- The seeded weekly comparison framework (`seeded_weekly_compare.py`) — unchanged

---

## Training & Evaluation

The commands are **identical** to the previous DQN runs. The architecture change is internal — method names, CLI flags, and file paths are all the same.

### Option A — Full seeded benchmark (matches the existing comparison)

This reproduces the same 5-seed, train-2020-to-2024, test-2024-to-present setup:

```bash
./venv/bin/python3 seeded_weekly_compare.py \
  --cutoff    2024-01-01 \
  --start-date 2024-01-01 \
  --epochs    50 \
  --seeds     11,22,33,44,55 \
  --stop-loss-pct 0.10 \
  --output-dir test_results/seeded_weekly_full
```

This trains both `dqn` and `dqn_nollm` for all 5 seeds automatically, then runs inference on all tickers and writes:

- `test_results/seeded_weekly_full/seeded_weekly_runs.csv` — raw per-seed results
- `test_results/seeded_weekly_full/seeded_weekly_summary.csv` — mean ± std per ticker
- `test_results/seeded_weekly_full/seeded_weekly_delta.csv` — LLM vs no-LLM delta
- `backtest_analysis_seeded_weekly_compare.ipynb` — auto-generated comparison notebook

### Option B — Train only (no evaluation)

```bash
# DQN with LLM structured signals
./venv/bin/python3 run.py train \
  --method dqn \
  --epochs 50 \
  --cutoff 2024-01-01 \
  --rebalance-frequency weekly

# DQN price-only (no LLM)
./venv/bin/python3 run.py train \
  --method dqn_nollm \
  --epochs 50 \
  --cutoff 2024-01-01 \
  --rebalance-frequency weekly
```

Or with a specific seed for reproducibility:

```bash
./venv/bin/python3 train.py --method dqn --epochs 50 --cutoff 2024-01-01 --seed 42
./venv/bin/python3 train.py --method dqn_nollm --epochs 50 --cutoff 2024-01-01 --seed 42
```

### Option C — Single-ticker backtest (inference only)

After training, test on a single ticker over the 2024–present period:

```bash
./venv/bin/python3 run.py backtest \
  --ticker NVDA \
  --method dqn \
  --no-learning \
  --load-model backtest_results/dqn_model_walkforward_final.pth \
  --start-date 2024-01-01 \
  --stop-loss-pct 0.10
```

### Option D — Side-by-side comparison on one ticker

```bash
./venv/bin/python3 run.py compare \
  --ticker NVDA \
  --stop-loss-pct 0.10
```

---

## Expected model paths after training

| Method | Path |
|--------|------|
| DQN (LLM) | `backtest_results/dqn_model_walkforward_final.pth` |
| DQN No-LLM | `backtest_results/dqn_nollm_model_walkforward_final.pth` |
| Seeded models | `backtest_results/seed_runs/{method}_weekly_seed_{N}.pth` |

> **Note:** Old checkpoint files trained with the 414-dim architecture will not load cleanly into the new 30-dim network. The existing `_partial_load` fallback in `load_model()` will skip mismatched weight tensors and log a warning — effectively initialising a fresh agent. Re-train from scratch to get correct weights.
