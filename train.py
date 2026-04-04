
import os
import sys
import logging
import random
import numpy as np
import contextlib
import argparse
from datetime import datetime
from data import TICKERS
from llm import fetch_financial_metrics
from backtest import Backtester
from dqn_agent import DQNAgent
from paths import BACKTEST_RESULTS_DIR
from reproducibility import set_random_seed
from rl_agent import MarkovRLAgent
from vectorizer import NarrativeVectorizer, dqn_state_dim

logging.basicConfig(level=logging.INFO)
logging.getLogger("backtest").setLevel(logging.WARNING)
logging.getLogger("graph_manager").setLevel(logging.WARNING)

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def train_agent(
    method='dqn',
    epochs=50,
    save_freq=5,
    train_cutoff_date="2024-01-01",
    rebalance_frequency: str = "weekly",
    seed: int = None
):
    """Train DQN / Q-learning on pre-cutoff data; Neo4j writes only on epoch 1 when LLM is on."""
    print(f"🚀 INITIALIZING {method.upper()} PRE-TRAINING (WALK-FORWARD)")
    print(f"================================================================================")
    print(f"   Target Tickers: {', '.join(TICKERS)}")
    print(f"   Method: Strict Time-Series Split")
    print(f"   TRAINING END DATE: {train_cutoff_date}")
    print(f"   Epochs: {epochs}")
    print(f"   Rebalance Frequency: {rebalance_frequency}")
    if seed is not None:
        print(f"   Seed: {seed}")
    print(f"================================================================================")

    if seed is not None:
        set_random_seed(seed)

    # 1. Initialize Shared Resources
    print("⏳ Initializing Vectorizer (this takes a moment)...")
    vectorizer = NarrativeVectorizer()
    state_dim = dqn_state_dim(vectorizer.embedding_dim)
    print(f"   ✅ Vectorizer initialized. Dim: {vectorizer.embedding_dim}")

    is_dqn = method in ('dqn', 'dqn_nollm')
    use_llm_signals = method != 'dqn_nollm'
    model_prefix = 'dqn_nollm' if method == 'dqn_nollm' else 'dqn'
    model_suffix = "" if rebalance_frequency == "weekly" else f"_{rebalance_frequency}"
    model_prefix_with_freq = f"{model_prefix}{model_suffix}"

    # 2. Initialize Agent
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
            enable_learning=True
        )
    elif method == 'qlearning':
        agent = MarkovRLAgent(
            narrative_dim=vectorizer.embedding_dim,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0, 
            enable_learning=True
        )
    
    print(f"   ✅ {method.upper()} Agent initialized.")

    # 3. Pre-load Financial Data
    print("⏳ Pre-loading financial data & filtering dates...")
    financial_data_cache = {}
    results_dir = BACKTEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    cutoff_dt = datetime.strptime(train_cutoff_date, "%Y-%m-%d")

    for ticker in TICKERS:
        if not use_llm_signals:
            financial_data_cache[ticker] = {}
            print(f"   ✅ Loaded {ticker} (price-only mode)")
            continue

        try:
            with suppress_stdout():
                financial_data_cache[ticker] = fetch_financial_metrics(ticker)
            print(f"   ✅ Loaded {ticker}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {ticker}: {e}")

    valid_tickers = list(financial_data_cache.keys())
    shared_summary_cache_by_ticker = {ticker: {} for ticker in valid_tickers}
    print(f"   Ready to train on {len(valid_tickers)} tickers.")
    print(f"================================================================================")

    # 4. Training Loop
    # For Q-learning manual epsilon decay
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_q = (epsilon_end / epsilon_start) ** (1 / epochs)
    current_epsilon = epsilon_start

    for epoch in range(1, epochs + 1):
        # Handle Q-learning epsilon manually
        if method == 'qlearning':
            agent.epsilon = current_epsilon
            display_epsilon = current_epsilon
        else:
            display_epsilon = agent.epsilon if hasattr(agent, 'epsilon') else 0.0
        epoch_read_only = not (use_llm_signals and epoch == 1)

        print(f"\n⚡ EPOCH {epoch}/{epochs} (ε={display_epsilon:.3f})")
        if use_llm_signals:
            print(
                f"   🗄️ Neo4j: {'READ-ONLY' if epoch_read_only else 'CACHE WARMUP (writes)'}"
            )
        random.shuffle(valid_tickers) 
        
        epoch_rewards = []
        progress_bar_length = 30
        
        for i, ticker in enumerate(valid_tickers):
            # Progress bar
            progress = (i / len(valid_tickers))
            bar = "█" * int(progress * progress_bar_length) + "-" * (progress_bar_length - int(progress * progress_bar_length))
            print(f"\r   Processing: [{bar}] {ticker:<5}", end="")
            
            try:
                # Create fresh backtester with READ_ONLY=True
                with suppress_stdout():
                    backtester = Backtester(
                        ticker=ticker,
                        initial_capital=10000.0,
                        rebalance_frequency=rebalance_frequency,
                        enable_learning=True,
                        epsilon=display_epsilon, # Passed to backtester but agent might override
                        use_llm_signals=use_llm_signals,
                        read_only=epoch_read_only,
                        summary_cache=shared_summary_cache_by_ticker[ticker]
                    )
                    
                    # INJECT SHARED RESOURCES
                    backtester.rl_agent = agent
                    backtester.vectorizer = vectorizer
                    
                    original_len = len(backtester.price_data)
                    backtester.price_data = backtester.price_data[backtester.price_data['timestamp'] < cutoff_dt]
                    filtered_len = len(backtester.price_data)
                    
                    if filtered_len < 50:
                        continue
                        
                    # Run backtest on the PAST data (Training Phase)
                    metrics = backtester.run_backtest(financial_data_cache[ticker])
                    epoch_rewards.append(metrics.total_return_pct)
                    
            except Exception as e:
                pass

        # Decay epsilon for Q-learning
        if method == 'qlearning':
            current_epsilon *= epsilon_decay_q
            current_epsilon = max(current_epsilon, epsilon_end)

        # End of epoch stats
        stats = agent.get_learning_stats()
        avg_return = np.mean(epoch_rewards) if epoch_rewards else 0.0
        
        print(f"\r   Processing: [{'█'*progress_bar_length}] DONE!")
        
        if is_dqn:
            print(f"   📊 Stats | Loss: {stats['avg_loss']:.4f} | Eps: {stats['exploration_rate']:.3f} | Buffer: {stats['buffer_size']} | Avg Ret: {avg_return:>6.2f}%")
        else:
            print(f"   📊 Stats | Q-Table Size: {stats['q_table_size']} | Eps: {display_epsilon:.3f} | Avg Ret: {avg_return:>6.2f}%")
        
        # Save Checkpoint
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

    # Final Save
    if is_dqn:
        final_path = f"{results_dir}/{model_prefix_with_freq}_model_walkforward_final.pth"
        agent.save_model(final_path)
        print(f"\n✅ TRAINING COMPLETE! Model saved to {final_path}")
    else:
        if rebalance_frequency == "weekly":
            final_path = f"{results_dir}/q_table_walkforward_final.json"
        else:
            final_path = f"{results_dir}/q_table_walkforward_{rebalance_frequency}_final.json"
        agent.save_q_table(final_path)
        print(f"\n✅ TRAINING COMPLETE! Q-Table saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL Agent')
    parser.add_argument('--method', choices=['dqn', 'dqn_nollm', 'qlearning'], default='dqn', help='RL Method')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--cutoff', type=str, default="2024-01-01", help='Training Data Cutoff Date')
    parser.add_argument('--rebalance-frequency', choices=['daily', 'weekly', 'monthly'], default='weekly', help='Trading decision frequency')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible training')
    args = parser.parse_args()
    
    train_agent(
        method=args.method,
        epochs=args.epochs,
        train_cutoff_date=args.cutoff,
        rebalance_frequency=args.rebalance_frequency,
        seed=args.seed
    )
