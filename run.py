#!/usr/bin/env python3
"""CLI: data collection, RAG, training, backtest, method comparison."""

import argparse
import json
import os
import sys
from typing import Optional

from backtest import Backtester
from llm import fetch_financial_metrics, process_ticker
from paths import BACKTEST_RESULTS_DIR, TEST_RESULTS_DIR
from train import train_agent


def _walkforward_model_path(model_prefix: str, rebalance_frequency: str = "weekly") -> str:
    b = BACKTEST_RESULTS_DIR
    if rebalance_frequency == "weekly":
        return f"{b}/{model_prefix}_walkforward_final.pth"
    if model_prefix.endswith("_model"):
        base = model_prefix[:-6]
        return f"{b}/{base}_{rebalance_frequency}_model_walkforward_final.pth"
    return f"{b}/{model_prefix}_{rebalance_frequency}_walkforward_final.pth"


def collect_data(ticker: str, api_key: Optional[str] = None):
    print(f"\n{'='*80}\nDATA COLLECTION\n{'='*80}\n")
    print(f"To collect data for {ticker}:")
    print("1. Ensure the ticker is in data.py `TICKERS` (and add symbol-specific fetches if needed)")
    print("2. Run: python data.py")
    print("\nOr run individual fetch functions in data.py as needed.")
    print("\nNote: set ALPHA_VANTAGE_API_KEY in .env")


def run_backtest(
    ticker: str = "NVDA",
    method: str = "qlearning",
    initial_capital: float = 10000.0,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    epsilon: float = 0.1,
    enable_learning: bool = True,
    save_model: bool = True,
    use_confidence_sizing: bool = True,
    stop_loss_pct: float = 0.0,
    model_path: Optional[str] = None,
    start_date: Optional[str] = None,
    rebalance_frequency: str = "weekly",
    read_only: bool = False,
):
    print(f"\n{'='*80}\nBACKTEST: {method.upper()}\n{'='*80}\n")
    print("Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Method: {method}")
    print(f"  Rebalance Frequency: {rebalance_frequency}")
    print(
        f"  Position Sizing: {'Confidence-weighted (10-100%)' if use_confidence_sizing else 'All-in/all-out (100%)'}"
    )
    if stop_loss_pct > 0:
        print(f"  Stop-Loss: {stop_loss_pct * 100:.1f}% (when learning is disabled)")
    if method in ["qlearning", "dqn", "dqn_nollm"]:
        print(f"  Learning Enabled: {enable_learning}")
    use_llm_signals = method != "dqn_nollm"

    backtester = Backtester(
        ticker=ticker,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        enable_learning=enable_learning if method != "fixed" else False,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        use_confidence_sizing=use_confidence_sizing,
        use_llm_signals=use_llm_signals,
        stop_loss_pct=stop_loss_pct,
        start_date=start_date,
        read_only=read_only,
    )

    if method in ("dqn", "dqn_nollm"):
        from dqn_agent import DQNAgent
        from vectorizer import NarrativeVectorizer, dqn_state_dim

        vectorizer = NarrativeVectorizer()
        state_dim = dqn_state_dim(vectorizer.embedding_dim)

        backtester.rl_agent = DQNAgent(
            state_dim=state_dim,
            embedding_dim=vectorizer.embedding_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            enable_learning=enable_learning,
        )
        if model_path:
            try:
                backtester.rl_agent.load_model(model_path)
                print(f"Loaded DQN weights from {model_path}")
            except Exception:
                print(f"Could not load model from {model_path}; using fresh agent")
        else:
            print(f"Initialized fresh DQN agent (state_dim={state_dim})")

    elif method == "qlearning":
        if model_path:
            from rl_agent import MarkovRLAgent
            from vectorizer import NarrativeVectorizer

            print("Initializing vectorizer for Q-learning...")
            vectorizer = NarrativeVectorizer()
            backtester.vectorizer = vectorizer

            backtester.rl_agent = MarkovRLAgent(
                narrative_dim=vectorizer.embedding_dim,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                enable_learning=enable_learning,
            )
            try:
                if hasattr(backtester.rl_agent, "load_q_table"):
                    backtester.rl_agent.load_q_table(model_path)
                    print(f"Loaded Q-table from {model_path}")
                else:
                    print("Agent does not support load_q_table")
            except Exception as e:
                print(f"Failed to load Q-table: {e}")

    if method == "dqn_nollm":
        print(f"Price-only mode for {ticker} (no LLM context).")
        financial_data = {}
    else:
        print(f"Loading financial metrics for {ticker}...")
        financial_data = fetch_financial_metrics(ticker)

    print("\nRunning backtest...\n")
    metrics = backtester.run_backtest(financial_data)

    print(f"\n{'='*80}\nRESULTS\n{'='*80}\n")
    print(f"Final Portfolio Value: ${metrics.final_portfolio_value:,.2f}")
    print(f"Total Return: {metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")

    if save_model and enable_learning:
        if method == "qlearning" and hasattr(backtester.rl_agent, "save_q_table"):
            filepath = f"{BACKTEST_RESULTS_DIR}/learned_q_table_{ticker}.json"
            backtester.rl_agent.save_q_table(filepath)
            print(f"\nQ-table saved to {filepath}")

        elif method in ("dqn", "dqn_nollm") and hasattr(backtester.rl_agent, "save_model"):
            prefix = "dqn_nollm_model" if method == "dqn_nollm" else "dqn_model"
            filepath = f"{BACKTEST_RESULTS_DIR}/{prefix}_{ticker}.pth"
            backtester.rl_agent.save_model(filepath)
            print(f"\nDQN model saved to {filepath}")

    print(f"\n{'='*80}\n")
    return metrics


def run_training(method: str, epochs: int, cutoff: str, rebalance_frequency: str = "weekly", seed: int = None):
    train_agent(
        method=method,
        epochs=epochs,
        train_cutoff_date=cutoff,
        rebalance_frequency=rebalance_frequency,
        seed=seed,
    )


def compare_methods(
    ticker: str = "NVDA",
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 0.0,
    rebalance_frequency: str = "weekly",
):
    print(f"\n{'='*80}\nCOMPARE DQN vs DQN (no LLM): {ticker}\n{'='*80}\n")

    results = {}

    print("\n" + "-" * 80 + "\n1/2: DQN (LLM)\n" + "-" * 80)
    dqn_model = _walkforward_model_path("dqn_model", rebalance_frequency)
    if not os.path.exists(dqn_model):
        print("No default DQN checkpoint found; using untrained agent.")
        dqn_model = None
    else:
        print(f"Loading {dqn_model}")

    try:
        results["DQN"] = run_backtest(
            ticker=ticker,
            method="dqn",
            initial_capital=initial_capital,
            model_path=dqn_model,
            enable_learning=False,
            save_model=False,
            use_confidence_sizing=True,
            stop_loss_pct=stop_loss_pct,
            start_date="2024-01-01",
            rebalance_frequency=rebalance_frequency,
            read_only=True,
        )
    except Exception as e:
        print(f"Failed: {e}")

    print("\n" + "-" * 80 + "\n2/2: DQN (price-only)\n" + "-" * 80)
    dqn_nollm_model = _walkforward_model_path("dqn_nollm_model", rebalance_frequency)
    if not os.path.exists(dqn_nollm_model):
        print("No default DQN no-LLM checkpoint found; using untrained agent.")
        dqn_nollm_model = None
    else:
        print(f"Loading {dqn_nollm_model}")

    try:
        results["DQN Price Only"] = run_backtest(
            ticker=ticker,
            method="dqn_nollm",
            initial_capital=initial_capital,
            model_path=dqn_nollm_model,
            enable_learning=False,
            save_model=False,
            use_confidence_sizing=True,
            stop_loss_pct=stop_loss_pct,
            start_date="2024-01-01",
            rebalance_frequency=rebalance_frequency,
            read_only=True,
        )
    except Exception as e:
        print(f"Failed: {e}")

    print(f"\n{'='*80}\nCOMPARISON (from 2024-01-01)\n{'='*80}\n")
    print(
        f"{'Method':<20} | {'Final Value':<15} | {'Return':<12} | {'Sharpe':<8} | {'Win Rate':<8} | {'Max DD':<10}"
    )
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for name, metrics in results.items():
        if metrics:
            print(
                f"{name:<20} | ${metrics.final_portfolio_value:>14,.2f} | "
                f"{metrics.total_return_pct:>11.2f}% | {metrics.sharpe_ratio:>8.2f} | "
                f"{metrics.win_rate:>7.2f}% | {metrics.max_drawdown_pct:>9.2f}%"
            )

    print(f"\n{'='*80}\n")

    if results:
        best_method = max(results.items(), key=lambda x: x[1].sharpe_ratio if x[1] else -999)
        if best_method[1]:
            print(f"Best by Sharpe: {best_method[0]}")
            print(f"  Return: {best_method[1].total_return_pct:.2f}%")
            print(f"  Sharpe: {best_method[1].sharpe_ratio:.2f}")

    print(f"\n{'='*80}\nSaving history JSON\n{'='*80}")
    results_dir = TEST_RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    for name, metrics in results.items():
        if metrics and metrics.history:
            safe_name = name.replace(" ", "_").replace("-", "_")
            filename = f"{ticker}_{safe_name}_history.json"
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "w") as f:
                    json.dump(metrics.history, f, indent=2)
                print(f"Saved {filepath}")
            except Exception as e:
                print(f"Failed to save history for {name}: {e}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="IntelliSense — RAG + RL trading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py collect --ticker NVDA
  python run.py train --method dqn --epochs 50
  python run.py compare --ticker NVDA
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    parser_collect = subparsers.add_parser("collect", help="Print instructions for data collection")
    parser_collect.add_argument("--ticker", default="NVDA", help="Stock ticker")
    parser_collect.add_argument("--api-key", help="(unused) Alpha Vantage key; use .env")

    parser_rag = subparsers.add_parser("rag", help="Run RAG / LLM pipeline for one ticker")
    parser_rag.add_argument("--ticker", default="NVDA", help="Stock ticker")

    parser_train = subparsers.add_parser("train", help="Train RL agent")
    parser_train.add_argument("--method", choices=["dqn", "dqn_nollm", "qlearning"], default="dqn")
    parser_train.add_argument("--epochs", type=int, default=50)
    parser_train.add_argument("--cutoff", type=str, default="2024-01-01")
    parser_train.add_argument(
        "--rebalance-frequency", choices=["daily", "weekly", "monthly"], default="weekly"
    )
    parser_train.add_argument("--seed", type=int, default=None)

    parser_backtest = subparsers.add_parser("backtest", help="Run backtest")
    parser_backtest.add_argument("--ticker", default="NVDA")
    parser_backtest.add_argument("--method", choices=["fixed", "qlearning", "dqn", "dqn_nollm"], default="qlearning")
    parser_backtest.add_argument("--capital", type=float, default=10000.0)
    parser_backtest.add_argument("--learning-rate", type=float, default=0.1)
    parser_backtest.add_argument("--epsilon", type=float, default=0.1)
    parser_backtest.add_argument("--no-learning", action="store_true")
    parser_backtest.add_argument("--no-save", action="store_true")
    parser_backtest.add_argument("--no-confidence-sizing", action="store_true")
    parser_backtest.add_argument("--stop-loss-pct", type=float, default=0.0)
    parser_backtest.add_argument("--load-model", help="Checkpoint path")
    parser_backtest.add_argument("--start-date", help="YYYY-MM-DD")
    parser_backtest.add_argument(
        "--rebalance-frequency", choices=["daily", "weekly", "monthly"], default="weekly"
    )

    parser_compare = subparsers.add_parser("compare", help="Compare DQN vs DQN no-LLM")
    parser_compare.add_argument("--ticker", default="NVDA")
    parser_compare.add_argument("--capital", type=float, default=10000.0)
    parser_compare.add_argument("--stop-loss-pct", type=float, default=0.0)
    parser_compare.add_argument(
        "--rebalance-frequency", choices=["daily", "weekly", "monthly"], default="weekly"
    )

    parser_quickstart = subparsers.add_parser("quickstart", help="Same as compare")
    parser_quickstart.add_argument("--ticker", default="NVDA")
    parser_quickstart.add_argument("--stop-loss-pct", type=float, default=0.0)
    parser_quickstart.add_argument(
        "--rebalance-frequency", choices=["daily", "weekly", "monthly"], default="weekly"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "collect":
        collect_data(args.ticker, args.api_key)
    elif args.command == "rag":
        process_ticker(args.ticker)
    elif args.command == "train":
        run_training(args.method, args.epochs, args.cutoff, args.rebalance_frequency, args.seed)
    elif args.command == "backtest":
        run_backtest(
            ticker=args.ticker,
            method=args.method,
            initial_capital=args.capital,
            learning_rate=args.learning_rate,
            epsilon=args.epsilon,
            enable_learning=not args.no_learning,
            save_model=not args.no_save,
            use_confidence_sizing=not args.no_confidence_sizing,
            stop_loss_pct=args.stop_loss_pct,
            model_path=args.load_model,
            start_date=args.start_date,
            rebalance_frequency=args.rebalance_frequency,
        )
    elif args.command == "compare" or args.command == "quickstart":
        compare_methods(args.ticker, args.capital, args.stop_loss_pct, args.rebalance_frequency)


if __name__ == "__main__":
    main()
