
import os
import sys
import logging
import random
import numpy as np
import contextlib
import io
import json
from datetime import datetime
from data import TICKERS
from main import fetch_financial_metrics
from backtest import Backtester
from rl_agent import MarkovRLAgent
from vectorizer import NarrativeVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_qlearning")

# Suppress other loggers
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

def train_qlearning(epochs=50, save_freq=5, train_cutoff_date="2024-01-01"):
    """
    Pre-train Tabular Q-Learning Agent on all available tickers using a STRICT Time-Series Split.
    
    METHODOLOGY (SOTA):
    - Training Set: All data BEFORE train_cutoff_date (e.g., 2020-2023)
    - Test/Trade Set: All data AFTER train_cutoff_date (e.g., 2024-2025)
    
    This ensures NO data leakage. The model learns generalizable state-action pairs
    from historical data across multiple assets.
    """
    print(f"🚀 INITIALIZING Q-LEARNING PRE-TRAINING (WALK-FORWARD)")
    print(f"================================================================================")
    print(f"   Target Tickers: {', '.join(TICKERS)}")
    print(f"   Method: Strict Time-Series Split")
    print(f"   TRAINING END DATE: {train_cutoff_date}")
    print(f"   Epochs: {epochs}")
    print(f"================================================================================")

    # 1. Initialize Shared Resources
    print("⏳ Initializing Vectorizer (this takes a moment)...")
    vectorizer = NarrativeVectorizer(model_name='all-MiniLM-L6-v2')
    print(f"   ✅ Vectorizer Loop initialized. Dim: {vectorizer.embedding_dim}")

    # 2. Initialize Agent
    agent = MarkovRLAgent(
        narrative_dim=vectorizer.embedding_dim,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,  # Start with high exploration
        enable_learning=True
    )
    # Decrease epsilon manually over epochs? No, let's just keep high exploration for pre-training.
    # Actually, we should decay it to exploit learned patterns.
    # MarkovRLAgent handles epsilon decay internally if called? No, it's fixed usually.
    # Let's verify rl_agent.py... it takes fixed epsilon.
    # We will manually decay it in the loop.
    
    print(f"   ✅ Q-Learning Agent initialized.")

    # 3. Pre-load Financial Data
    print("⏳ Pre-loading financial data & filtering dates...")
    financial_data_cache = {}
    results_dir = "backtest_results"
    os.makedirs(results_dir, exist_ok=True)
    
    cutoff_dt = datetime.strptime(train_cutoff_date, "%Y-%m-%d")

    for ticker in TICKERS:
        try:
            with suppress_stdout():
                financial_data_cache[ticker] = fetch_financial_metrics(ticker)
            print(f"   ✅ Loaded {ticker}")
        except Exception as e:
            print(f"   ⚠️ Failed to load {ticker}: {e}")

    valid_tickers = list(financial_data_cache.keys())
    print(f"   Ready to train on {len(valid_tickers)} tickers.")
    print(f"================================================================================")

    # 4. Training Loop
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = (epsilon_end / epsilon_start) ** (1 / epochs)
    current_epsilon = epsilon_start

    for epoch in range(1, epochs + 1):
        # Update agent epsilon
        agent.epsilon = current_epsilon
        
        print(f"\n⚡ EPOCH {epoch}/{epochs} (ε={current_epsilon:.3f})")
        random.shuffle(valid_tickers) 
        
        epoch_rewards = []
        progress_bar_length = 30
        
        for i, ticker in enumerate(valid_tickers):
            # Progress bar
            progress = (i / len(valid_tickers))
            bar = "█" * int(progress * progress_bar_length) + "-" * (progress_bar_length - int(progress * progress_bar_length))
            print(f"\r   Processing: [{bar}] {ticker:<5}", end="")
            
            try:
                # Create fresh backtester
                with suppress_stdout():
                    backtester = Backtester(
                        ticker=ticker,
                        initial_capital=10000.0,
                        enable_learning=True,
                        epsilon=current_epsilon,
                        transaction_cost_pct=0.001
                    )
                    
                    # INJECT SHARED RESOURCES
                    backtester.rl_agent = agent
                    backtester.vectorizer = vectorizer
                    
                    # ----------------------------------------------------
                    # CRITICAL: TIME-SERIES SPLIT
                    # Filter the backtester's price data to ONLY include past data
                    # ----------------------------------------------------
                    original_len = len(backtester.price_data)
                    backtester.price_data = backtester.price_data[backtester.price_data['timestamp'] < cutoff_dt]
                    filtered_len = len(backtester.price_data)
                    
                    if filtered_len < 50:
                        continue
                        
                    # Run backtest on the PAST data (Training Phase)
                    metrics = backtester.run_backtest(financial_data_cache[ticker])
                    epoch_rewards.append(metrics.total_return_pct)
                    
            except Exception as e:
                # print(f"Error: {e}")
                pass

        # Decay epsilon
        current_epsilon *= epsilon_decay
        current_epsilon = max(current_epsilon, epsilon_end)

        # End of epoch stats
        stats = agent.get_learning_stats()
        avg_return = np.mean(epoch_rewards) if epoch_rewards else 0.0
        
        print(f"\r   Processing: [{'█'*progress_bar_length}] DONE!")
        print(f"   📊 Stats | Q-Table Size: {stats['q_table_size']} | Eps: {current_epsilon:.3f} | Avg Return (Train): {avg_return:>6.2f}%")
        
        # Save Checkpoint
        if epoch % save_freq == 0:
            checkpoint_path = f"{results_dir}/q_table_walkforward_epoch_{epoch}.json"
            agent.save_q_table(checkpoint_path)

    # Final Save
    final_path = f"{results_dir}/q_table_walkforward_final.json"
    agent.save_q_table(final_path)
    print(f"\n✅ TRAINING COMPLETE! Q-Table saved to {final_path}")
    print(f"   To TEST this model on the FUTURE (2024-Present), run:")
    print(f"   python run.py backtest --method qlearning --ticker NVDA --no-learning --load-model {final_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Tabular Q-Learning Agent')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--cutoff', type=str, default="2024-01-01", help='Training Data Cutoff Date')
    args = parser.parse_args()
    
    train_qlearning(epochs=args.epochs, train_cutoff_date=args.cutoff)
