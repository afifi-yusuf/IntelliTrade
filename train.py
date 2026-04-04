import argparse
import contextlib
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np

from backtest import Backtester
from data import TICKERS
from dqn_agent import DQNAgent
from llm import fetch_financial_metrics
from paths import BACKTEST_RESULTS_DIR
from reproducibility import set_random_seed
from rl_agent import MarkovRLAgent
from vectorizer import NarrativeVectorizer, dqn_state_dim

logging.basicConfig(level=logging.INFO)
logging.getLogger("backtest").setLevel(logging.WARNING)
logging.getLogger("graph_manager").setLevel(logging.WARNING)


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def train_agent(
    method="dqn",
    epochs=50,
    save_freq=5,
    train_cutoff_date="2024-01-01",
    rebalance_frequency: str = "weekly",
    seed: int = None,
):
    """Train DQN / Q-learning on pre-cutoff data; Neo4j writes only on epoch 1 when LLM is on."""
    print(f"Training {method.upper()} (walk-forward)")
    print("=" * 72)
    print(f"  Tickers: {', '.join(TICKERS)}")
    print(f"  Cutoff: {train_cutoff_date}")
    print(f"  Epochs: {epochs}")
    print(f"  Rebalance: {rebalance_frequency}")
    if seed is not None:
        print(f"  Seed: {seed}")
    print("=" * 72)

    if seed is not None:
        set_random_seed(seed)

    print("Initializing vectorizer...")
    vectorizer = NarrativeVectorizer()
    state_dim = dqn_state_dim(vectorizer.embedding_dim)
    print(f"  Embedding dim: {vectorizer.embedding_dim}")

    is_dqn = method in ("dqn", "dqn_nollm")
    use_llm_signals = method != "dqn_nollm"
    model_prefix = "dqn_nollm" if method == "dqn_nollm" else "dqn"
    model_suffix = "" if rebalance_frequency == "weekly" else f"_{rebalance_frequency}"
    model_prefix_with_freq = f"{model_prefix}{model_suffix}"

    agent = None
    if is_dqn:
        agent = DQNAgent(
            state_dim=state_dim,
            embedding_dim=vectorizer.embedding_dim,
            learning_rate=0.0005,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.05,
            buffer_size=50000,
            batch_size=64,
            enable_learning=True,
        )
    elif method == "qlearning":
        agent = MarkovRLAgent(
            narrative_dim=vectorizer.embedding_dim,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            enable_learning=True,
        )

    print(f"  {method.upper()} agent ready.")

    print("Loading financial data cache...")
    financial_data_cache = {}
    results_dir = BACKTEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    cutoff_dt = datetime.strptime(train_cutoff_date, "%Y-%m-%d")

    for ticker in TICKERS:
        if not use_llm_signals:
            financial_data_cache[ticker] = {}
            print(f"  {ticker}: (price-only)")
            continue
        try:
            with suppress_stdout():
                financial_data_cache[ticker] = fetch_financial_metrics(ticker)
            print(f"  {ticker}: ok")
        except Exception as e:
            print(f"  {ticker}: failed — {e}")

    valid_tickers = list(financial_data_cache.keys())
    shared_summary_cache_by_ticker = {ticker: {} for ticker in valid_tickers}
    print(f"Training on {len(valid_tickers)} tickers.")
    print("=" * 72)

    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_q = (epsilon_end / epsilon_start) ** (1 / epochs)
    current_epsilon = epsilon_start
    bar_len = 30

    for epoch in range(1, epochs + 1):
        if method == "qlearning":
            agent.epsilon = current_epsilon
            display_epsilon = current_epsilon
        else:
            display_epsilon = agent.epsilon if hasattr(agent, "epsilon") else 0.0
        epoch_read_only = not (use_llm_signals and epoch == 1)

        print(f"\nEpoch {epoch}/{epochs} (epsilon={display_epsilon:.3f})")
        if use_llm_signals:
            mode = "read-only" if epoch_read_only else "cache warmup (writes)"
            print(f"  Neo4j: {mode}")

        random.shuffle(valid_tickers)
        epoch_rewards = []

        for i, ticker in enumerate(valid_tickers):
            progress = i / max(len(valid_tickers), 1)
            filled = int(progress * bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {ticker:<6}", end="", flush=True)

            try:
                with suppress_stdout():
                    backtester = Backtester(
                        ticker=ticker,
                        initial_capital=10000.0,
                        rebalance_frequency=rebalance_frequency,
                        enable_learning=True,
                        epsilon=display_epsilon,
                        use_llm_signals=use_llm_signals,
                        read_only=epoch_read_only,
                        summary_cache=shared_summary_cache_by_ticker[ticker],
                    )
                    backtester.rl_agent = agent
                    backtester.vectorizer = vectorizer
                    backtester.price_data = backtester.price_data[
                        backtester.price_data["timestamp"] < cutoff_dt
                    ]
                    if len(backtester.price_data) < 50:
                        continue
                    metrics = backtester.run_backtest(financial_data_cache[ticker])
                    epoch_rewards.append(metrics.total_return_pct)
            except Exception as e:
                logging.getLogger(__name__).debug("skip %s: %s", ticker, e)

        if method == "qlearning":
            current_epsilon *= epsilon_decay_q
            current_epsilon = max(current_epsilon, epsilon_end)

        stats = agent.get_learning_stats()
        avg_return = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        print(f"\r  [{'#' * bar_len}] done")
        if is_dqn:
            print(
                f"  loss={stats['avg_loss']:.4f} eps={stats['exploration_rate']:.3f} "
                f"buffer={stats['buffer_size']} avg_return={avg_return:6.2f}%"
            )
        else:
            print(
                f"  q_table={stats['q_table_size']} eps={display_epsilon:.3f} avg_return={avg_return:6.2f}%"
            )

        if epoch % save_freq == 0:
            if is_dqn:
                path = f"{results_dir}/{model_prefix_with_freq}_model_walkforward_epoch_{epoch}.pth"
                agent.save_model(path)
            else:
                if rebalance_frequency == "weekly":
                    path = f"{results_dir}/q_table_walkforward_epoch_{epoch}.json"
                else:
                    path = f"{results_dir}/q_table_walkforward_{rebalance_frequency}_epoch_{epoch}.json"
                agent.save_q_table(path)

    if is_dqn:
        final_path = f"{results_dir}/{model_prefix_with_freq}_model_walkforward_final.pth"
        agent.save_model(final_path)
        print(f"\nTraining finished. Model: {final_path}")
    else:
        if rebalance_frequency == "weekly":
            final_path = f"{results_dir}/q_table_walkforward_final.json"
        else:
            final_path = f"{results_dir}/q_table_walkforward_{rebalance_frequency}_final.json"
        agent.save_q_table(final_path)
        print(f"\nTraining finished. Q-table: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--method", choices=["dqn", "dqn_nollm", "qlearning"], default="dqn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cutoff", type=str, default="2024-01-01")
    parser.add_argument(
        "--rebalance-frequency", choices=["daily", "weekly", "monthly"], default="weekly"
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    train_agent(
        method=args.method,
        epochs=args.epochs,
        train_cutoff_date=args.cutoff,
        rebalance_frequency=args.rebalance_frequency,
        seed=args.seed,
    )
