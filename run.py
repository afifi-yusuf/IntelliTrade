#!/usr/bin/env python3
"""CLI: data collection, RAG, training, backtest, method comparison."""

import argparse
import sys
import os
from typing import Optional

# Core imports
from backtest import Backtester
from llm import fetch_financial_metrics
from llm import main as run_main_agent
from train import train_agent

def _walkforward_model_path(model_prefix: str, rebalance_frequency: str = 'weekly') -> str:
    """Resolve default walk-forward model path for a frequency."""
    if rebalance_frequency == 'weekly':
        return f'backtest_results/{model_prefix}_walkforward_final.pth'
    if model_prefix.endswith('_model'):
        base = model_prefix[:-6]
        return f'backtest_results/{base}_{rebalance_frequency}_model_walkforward_final.pth'
    return f'backtest_results/{model_prefix}_{rebalance_frequency}_walkforward_final.pth'


def collect_data(ticker: str, api_key: Optional[str] = None):
    """Collect financial data from Alpha Vantage"""
    print(f"\n{'='*80}")
    print(f"📊 DATA COLLECTION")
    print(f"{'='*80}\n")
    
    print(f"To collect data for {ticker}:")
    print(f"1. Edit data.py to set TICKER = '{ticker}'")
    print(f"2. Run: ./venv/bin/python data.py")
    print(f"\nOr manually run the individual fetch functions in data.py")
    print(f"\n⚠️  Note: You need an Alpha Vantage API key (set in .env)")

def run_backtest(
    ticker: str = 'NVDA',
    method: str = 'qlearning',
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
    rebalance_frequency: str = 'weekly',
    read_only: bool = False
):
    """Run backtest with specified method"""
    
    print(f"\n{'='*80}")
    print(f"🚀 BACKTESTING: {method.upper()}")
    print(f"{'='*80}\n")
    
    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Method: {method}")
    print(f"  Rebalance Frequency: {rebalance_frequency}")
    print(f"  Position Sizing: {'Confidence-weighted (10-100%)' if use_confidence_sizing else 'All-in/all-out (100%)'}")
    if stop_loss_pct > 0:
        print(f"  Stop-Loss: {stop_loss_pct * 100:.1f}% (applies only when learning is disabled)")
    
    if method in ['qlearning', 'dqn', 'dqn_nollm']:
        print(f"  Learning Enabled: {enable_learning}")
    use_llm_signals = method != 'dqn_nollm'
    
    # Initialize backtester
    backtester = Backtester(
        ticker=ticker,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        enable_learning=enable_learning if method != 'fixed' else False,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        use_confidence_sizing=use_confidence_sizing,
        use_llm_signals=use_llm_signals,
        stop_loss_pct=stop_loss_pct,
        start_date=start_date,
        read_only=read_only
    )
    
    # For DQN, replace the RL agent
    if method in ('dqn', 'dqn_nollm'):
        from dqn_agent import DQNAgent
        from vectorizer import NarrativeVectorizer, dqn_state_dim

        vectorizer = NarrativeVectorizer(model_name='all-MiniLM-L6-v2')
        state_dim = dqn_state_dim(vectorizer.embedding_dim)

        backtester.rl_agent = DQNAgent(
            state_dim=state_dim,
            embedding_dim=vectorizer.embedding_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            enable_learning=enable_learning
        )
        if model_path:
            try:
                backtester.rl_agent.load_model(model_path)
                print(f"🧠 Loaded DQN Agent from {model_path}")
            except Exception:
                print(f"⚠️ Could not load model from {model_path}, using fresh agent")
        else:
            print(f"🧠 Initialized Fresh DQN Agent (state_dim={state_dim})")
            
    elif method == 'qlearning':
        if model_path:
            from rl_agent import MarkovRLAgent
            from vectorizer import NarrativeVectorizer
            print("⏳ Initializing Vectorizer for Q-Learning...")
            vectorizer = NarrativeVectorizer(model_name='all-MiniLM-L6-v2')
            backtester.vectorizer = vectorizer
            
            backtester.rl_agent = MarkovRLAgent(
                narrative_dim=vectorizer.embedding_dim,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                enable_learning=enable_learning
            )
            try:
                # Basic JSON load or specific method
                if hasattr(backtester.rl_agent, 'load_q_table'):
                    backtester.rl_agent.load_q_table(model_path)
                    print(f"🧠 Loaded Q-Table form {model_path}")
                else:
                    print(f"⚠️ Agent does not support loading Q-Table")
            except Exception as e:
                print(f"⚠️ Failed to load Q-table: {e}")
    
    # Fetch financial data
    if method == 'dqn_nollm':
        print(f"📈 Price-only mode active for {ticker} (skipping LLM/earnings/financial context).")
        financial_data = {}
    else:
        print(f"📈 Fetching financial metrics for {ticker}...")
        financial_data = fetch_financial_metrics(ticker)
    
    # Run backtest
    print(f"\n🔄 Running backtest...\n")
    metrics = backtester.run_backtest(financial_data)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"📊 BACKTEST RESULTS")
    print(f"{'='*80}\n")
    
    print(f"Final Portfolio Value: ${metrics.final_portfolio_value:,.2f}")
    print(f"Total Return: {metrics.total_return_pct:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    
    # Save model if requested
    if save_model and enable_learning:
        if method == 'qlearning' and hasattr(backtester.rl_agent, 'save_q_table'):
            filepath = f'backtest_results/learned_q_table_{ticker}.json'
            backtester.rl_agent.save_q_table(filepath)
            print(f"\n💾 Q-table saved to: {filepath}")
        
        elif method in ('dqn', 'dqn_nollm') and hasattr(backtester.rl_agent, 'save_model'):
            prefix = 'dqn_nollm_model' if method == 'dqn_nollm' else 'dqn_model'
            filepath = f'backtest_results/{prefix}_{ticker}.pth'
            backtester.rl_agent.save_model(filepath)
            print(f"\n💾 DQN model saved to: {filepath}")
    
    print(f"\n{'='*80}\n")
    
    return metrics

def run_training(method: str, epochs: int, cutoff: str, rebalance_frequency: str = 'weekly', seed: int = None):
    """Run training for specified method"""
    train_agent(
        method=method,
        epochs=epochs,
        train_cutoff_date=cutoff,
        rebalance_frequency=rebalance_frequency,
        seed=seed
    )

def compare_methods(
    ticker: str = 'NVDA',
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 0.0,
    rebalance_frequency: str = 'weekly'
):
    """Compare DQN variants only: DQN and DQN price-only."""
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPARING DQN VARIANTS FOR {ticker}")
    print(f"{'='*80}\n")
    
    results = {}

    # 1. DQN
    print(f"\n{'─'*80}")
    print("1/2: Deep Q-Network (DQN) (Pre-trained)")
    print(f"{'─'*80}")
    dqn_model = _walkforward_model_path('dqn_model', rebalance_frequency)
    if not os.path.exists(dqn_model):
        print("⚠️ No pre-trained DQN model found at default path. Using untrained agent.")
        dqn_model = None
    else:
        print(f"📂 Loading: {dqn_model}")
        
    try:
        results['DQN'] = run_backtest(
            ticker=ticker,
            method='dqn',
            initial_capital=initial_capital,
            model_path=dqn_model,
            enable_learning=False, # Testing mode
            save_model=False,
            use_confidence_sizing=True,
            stop_loss_pct=stop_loss_pct,
            start_date='2024-01-01',
            rebalance_frequency=rebalance_frequency,
            read_only=True
        )
    except Exception as e:
        print(f"❌ Failed: {e}")

    # 2. DQN (Price-Only / No LLM)
    print(f"\n{'─'*80}")
    print("2/2: Deep Q-Network (Price-Only, No LLM) (Pre-trained)")
    print(f"{'─'*80}")
    dqn_nollm_model = _walkforward_model_path('dqn_nollm_model', rebalance_frequency)
    if not os.path.exists(dqn_nollm_model):
        print("⚠️ No pre-trained DQN No-LLM model found at default path. Using untrained agent.")
        dqn_nollm_model = None
    else:
        print(f"📂 Loading: {dqn_nollm_model}")

    try:
        results['DQN Price Only'] = run_backtest(
            ticker=ticker,
            method='dqn_nollm',
            initial_capital=initial_capital,
            model_path=dqn_nollm_model,
            enable_learning=False,  # Testing mode
            save_model=False,
            use_confidence_sizing=True,
            stop_loss_pct=stop_loss_pct,
            start_date='2024-01-01',
            rebalance_frequency=rebalance_frequency,
            read_only=True
        )
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY (2024-Present)")
    print(f"{'='*80}\n")
    
    print(f"{'Method':<20} | {'Final Value':<15} | {'Return':<12} | {'Sharpe':<8} | {'Win Rate':<8} | {'Max DD':<10}")
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
    
    for name, metrics in results.items():
        if metrics:
            print(f"{name:<20} | ${metrics.final_portfolio_value:>14,.2f} | {metrics.total_return_pct:>11.2f}% | {metrics.sharpe_ratio:>8.2f} | {metrics.win_rate:>7.2f}% | {metrics.max_drawdown_pct:>9.2f}%")
    
    print(f"\n{'='*80}\n")
    
    # Winner
    if results:
        best_method = max(results.items(), key=lambda x: x[1].sharpe_ratio if x[1] else -999)
        if best_method[1]:
            print(f"🏆 BEST METHOD (by Sharpe Ratio): {best_method[0]}")
            print(f"   Return: {best_method[1].total_return_pct:.2f}%")
            print(f"   Sharpe: {best_method[1].sharpe_ratio:.2f}")

    # Save detailed history for graphing
    print(f"\n{'='*80}")
    print("💾 SAVING DETAILED HISTORY")
    print(f"{'='*80}")
    
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    import json
    
    for name, metrics in results.items():
        if metrics and metrics.history:
            # Clean filename: "Fixed Q-table" -> "Fixed_Q_table"
            safe_name = name.replace(" ", "_").replace("-", "_")
            filename = f"{ticker}_{safe_name}_history.json"
            filepath = os.path.join(results_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(metrics.history, f, indent=2)
                print(f"✅ Saved history to {filepath}")
            except Exception as e:
                print(f"❌ Failed to save history for {name}: {e}")
    print(f"{'='*80}\n")


def main():
    """Main entry point with CLI"""
    
    parser = argparse.ArgumentParser(
        description='IntelliSense - Context Enriched Agentic RAG Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data
  python run.py collect --ticker NVDA
  
  # Train Agents
  python run.py train --method dqn --epochs 50
  python run.py train --method dqn_nollm --epochs 50
  
  # Compare all methods
  python run.py compare --ticker NVDA
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect data
    parser_collect = subparsers.add_parser('collect', help='Collect financial data')
    parser_collect.add_argument('--ticker', default='NVDA', help='Stock ticker (default: NVDA)')
    parser_collect.add_argument('--api-key', help='Alpha Vantage API key')
    
    # RAG pipeline
    parser_rag = subparsers.add_parser('rag', help='Run RAG pipeline (LLM analysis)')
    parser_rag.add_argument('--ticker', default='NVDA', help='Stock ticker (default: NVDA)')
    
    # Train
    parser_train = subparsers.add_parser('train', help='Train RL Agent')
    parser_train.add_argument('--method', choices=['dqn', 'dqn_nollm', 'qlearning'], default='dqn', help='RL Method')
    parser_train.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser_train.add_argument('--cutoff', type=str, default="2024-01-01", help='Training Cutoff Date')
    parser_train.add_argument('--rebalance-frequency', choices=['daily', 'weekly', 'monthly'], default='weekly', help='Trading decision frequency')
    parser_train.add_argument('--seed', type=int, default=None, help='Random seed for reproducible training')

    # Backtest
    parser_backtest = subparsers.add_parser('backtest', help='Run backtest')
    parser_backtest.add_argument('--ticker', default='NVDA', help='Stock ticker (default: NVDA)')
    parser_backtest.add_argument('--method', choices=['fixed', 'qlearning', 'dqn', 'dqn_nollm'], default='qlearning')
    parser_backtest.add_argument('--capital', type=float, default=10000.0)
    parser_backtest.add_argument('--learning-rate', type=float, default=0.1)
    parser_backtest.add_argument('--epsilon', type=float, default=0.1)
    parser_backtest.add_argument('--no-learning', action='store_true')
    parser_backtest.add_argument('--no-save', action='store_true')
    parser_backtest.add_argument('--no-confidence-sizing', action='store_true')
    parser_backtest.add_argument('--stop-loss-pct', type=float, default=0.0, help='Inference-only stop-loss as decimal (e.g., 0.10)')
    parser_backtest.add_argument('--load-model', help='Path to pre-trained model')
    parser_backtest.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser_backtest.add_argument('--rebalance-frequency', choices=['daily', 'weekly', 'monthly'], default='weekly', help='Trading decision frequency')
    
    # Compare methods
    parser_compare = subparsers.add_parser('compare', help='Compare all methods')
    parser_compare.add_argument('--ticker', default='NVDA', help='Stock ticker')
    parser_compare.add_argument('--capital', type=float, default=10000.0)
    parser_compare.add_argument('--stop-loss-pct', type=float, default=0.0, help='Inference-only stop-loss as decimal (e.g., 0.10)')
    parser_compare.add_argument('--rebalance-frequency', choices=['daily', 'weekly', 'monthly'], default='weekly', help='Trading decision frequency')
    
    # Quickstart
    parser_quickstart = subparsers.add_parser('quickstart', help='Compare methods')
    parser_quickstart.add_argument('--ticker', default='NVDA')
    parser_quickstart.add_argument('--stop-loss-pct', type=float, default=0.0, help='Inference-only stop-loss as decimal (e.g., 0.10)')
    parser_quickstart.add_argument('--rebalance-frequency', choices=['daily', 'weekly', 'monthly'], default='weekly', help='Trading decision frequency')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'collect':
        collect_data(args.ticker, args.api_key)
    
    elif args.command == 'rag':
        run_rag_pipeline(args.ticker)
        
    elif args.command == 'train':
        run_training(args.method, args.epochs, args.cutoff, args.rebalance_frequency, args.seed)
    
    elif args.command == 'backtest':
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
            rebalance_frequency=args.rebalance_frequency
        )
    
    elif args.command == 'compare' or args.command == 'quickstart':
        compare_methods(args.ticker, args.capital, args.stop_loss_pct, args.rebalance_frequency)

if __name__ == "__main__":
    main()
