#!/usr/bin/env python3
"""
IntelliSense - Context Enriched Agentic RAG Trading System

Single entry point for all operations:
- Collect data from Alpha Vantage
- Run RAG pipeline (LLM analysis with Neo4j)
- Backtest trading strategies (Fixed Q-table, Q-learning, DQN)
- Compare methods
"""

import argparse
import sys
from typing import Optional

# Core imports
from backtest import Backtester
from main import fetch_financial_metrics


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
    use_confidence_sizing: bool = True
):
    """Run backtest with specified method"""
    
    print(f"\n{'='*80}")
    print(f"🚀 BACKTESTING: {method.upper()}")
    print(f"{'='*80}\n")
    
    print(f"Configuration:")
    print(f"  Ticker: {ticker}")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Method: {method}")
    print(f"  Position Sizing: {'Confidence-weighted (10-100%)' if use_confidence_sizing else 'All-in/all-out (100%)'}")
    
    if method in ['qlearning', 'dqn']:
        print(f"  Learning Enabled: {enable_learning}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Discount Factor: {discount_factor}")
        print(f"  Exploration (ε): {epsilon}")
    
    print()
    
    # Initialize backtester
    backtester = Backtester(
        ticker=ticker,
        initial_capital=initial_capital,
        enable_learning=enable_learning if method != 'fixed' else False,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        use_confidence_sizing=use_confidence_sizing
    )
    
    # For DQN, replace the RL agent
    if method == 'dqn':
        from dqn_agent import DQNAgent
        
        # Get state dimension from first signal to properly initialize DQN
        # For now, use default state dim (384 embedding + 7 stock + 3 narrative + 2 financial)
        state_dim = 396  # This matches current implementation
        
        backtester.rl_agent = DQNAgent(
            state_dim=state_dim,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            enable_learning=enable_learning
        )
        print(f"🧠 Using DQN Agent (state_dim={state_dim})")
    
    # Fetch financial data
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
    print(f"Annualized Return: {metrics.annualized_return_pct:.2f}%")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
    print(f"Win Rate: {metrics.win_rate:.2f}%")
    print(f"Total Trades: {metrics.total_trades}")
    
    if method == 'qlearning' and hasattr(backtester.rl_agent, 'get_learning_stats'):
        stats = backtester.rl_agent.get_learning_stats()
        print(f"\n🧠 Q-Learning Statistics:")
        print(f"   Total Updates: {stats['total_updates']}")
        print(f"   Q-Table Size: {stats['q_table_size']} states")
        print(f"   Final Exploration Rate: {stats['exploration_rate']:.3f}")
    
    elif method == 'dqn' and hasattr(backtester.rl_agent, 'get_learning_stats'):
        stats = backtester.rl_agent.get_learning_stats()
        print(f"\n🧠 DQN Statistics:")
        print(f"   Training Steps: {stats['training_steps']}")
        print(f"   Network Parameters: {stats['network_params']:,}")
        print(f"   Buffer Size: {stats['buffer_size']}")
        print(f"   Final Exploration Rate: {stats['exploration_rate']:.3f}")
        print(f"   Average Loss: {stats['avg_loss']:.4f}")
    
    # Save model if requested
    if save_model and enable_learning:
        if method == 'qlearning' and hasattr(backtester.rl_agent, 'save_q_table'):
            filepath = f'backtest_results/learned_q_table_{ticker}.json'
            backtester.rl_agent.save_q_table(filepath)
            print(f"\n💾 Q-table saved to: {filepath}")
        
        elif method == 'dqn' and hasattr(backtester.rl_agent, 'save_model'):
            filepath = f'backtest_results/dqn_model_{ticker}.pth'
            backtester.rl_agent.save_model(filepath)
            print(f"\n💾 DQN model saved to: {filepath}")
    
    print(f"\n{'='*80}\n")
    
    return metrics


def compare_methods(
    ticker: str = 'NVDA',
    initial_capital: float = 10000.0,
    learning_rate: float = 0.1,
    epsilon: float = 0.1
):
    """Compare all three methods: Fixed Q-table, Q-learning, DQN"""
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPARING ALL METHODS")
    print(f"{'='*80}\n")
    
    results = {}
    
    # 1. Fixed Q-table (baseline)
    print(f"\n{'─'*80}")
    print("1/3: Fixed Q-table (Baseline)")
    print(f"{'─'*80}")
    metrics_fixed = run_backtest(
        ticker=ticker,
        method='fixed',
        initial_capital=initial_capital,
        enable_learning=False,
        save_model=False
    )
    results['Fixed Q-table'] = metrics_fixed
    
    # 2. Q-learning
    print(f"\n{'─'*80}")
    print("2/3: Q-learning")
    print(f"{'─'*80}")
    metrics_qlearning = run_backtest(
        ticker=ticker,
        method='qlearning',
        initial_capital=initial_capital,
        learning_rate=learning_rate,
        epsilon=epsilon,
        enable_learning=True,
        save_model=True
    )
    results['Q-learning'] = metrics_qlearning
    
    # 3. DQN
    print(f"\n{'─'*80}")
    print("3/3: Deep Q-Network (DQN)")
    print(f"{'─'*80}")
    metrics_dqn = run_backtest(
        ticker=ticker,
        method='dqn',
        initial_capital=initial_capital,
        learning_rate=0.001,  # DQN typically needs smaller LR
        epsilon=epsilon,
        enable_learning=True,
        save_model=True
    )
    results['DQN'] = metrics_dqn
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("📊 COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Method':<20} | {'Final Value':<15} | {'Return':<12} | {'Sharpe':<8} | {'Win Rate':<8} | {'Complexity':<15}")
    print(f"{'-'*20}-+-{'-'*15}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*15}")
    
    for name, metrics in results.items():
        complexity = "6 states" if name == "Fixed Q-table" else \
                    f"{results['Q-learning'].total_trades} states" if name == "Q-learning" and hasattr(results['Q-learning'], 'total_trades') else \
                    "142,979 weights"
        
        print(f"{name:<20} | ${metrics.final_portfolio_value:>14,.2f} | {metrics.total_return_pct:>11.2f}% | {metrics.sharpe_ratio:>8.2f} | {metrics.win_rate:>7.2f}% | {complexity:<15}")
    
    print(f"\n{'='*80}\n")
    
    # Winner
    best_method = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"🏆 BEST METHOD (by Sharpe Ratio): {best_method[0]}")
    print(f"   Return: {best_method[1].total_return_pct:.2f}%")
    print(f"   Sharpe: {best_method[1].sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {best_method[1].max_drawdown_pct:.2f}%")
    
    print(f"\n{'='*80}\n")


def run_rag_pipeline(ticker: str = 'NVDA'):
    """Run the RAG pipeline to generate LLM analysis"""
    print(f"\n{'='*80}")
    print(f"🤖 RUNNING RAG PIPELINE FOR {ticker}")
    print(f"{'='*80}\n")
    
    from main import main as run_main
    run_main()
    
    print(f"\n✅ RAG pipeline complete!")


def main():
    """Main entry point with CLI"""
    
    parser = argparse.ArgumentParser(
        description='IntelliSense - Context Enriched Agentic RAG Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data for a stock
  python run.py collect --ticker NVDA
  
  # Run RAG pipeline (LLM analysis)
  python run.py rag --ticker NVDA
  
  # Backtest with Q-learning
  python run.py backtest --method qlearning --ticker NVDA
  
  # Backtest with DQN
  python run.py backtest --method dqn --ticker NVDA --learning-rate 0.001
  
  # Compare all methods
  python run.py compare --ticker NVDA
  
  # Quick start (collect data + compare all methods)
  python run.py quickstart --ticker NVDA
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
    
    # Backtest
    parser_backtest = subparsers.add_parser('backtest', help='Run backtest')
    parser_backtest.add_argument('--ticker', default='NVDA', help='Stock ticker (default: NVDA)')
    parser_backtest.add_argument('--method', choices=['fixed', 'qlearning', 'dqn'], default='qlearning',
                                help='RL method (default: qlearning)')
    parser_backtest.add_argument('--capital', type=float, default=10000.0,
                                help='Initial capital (default: 10000)')
    parser_backtest.add_argument('--learning-rate', type=float, default=0.1,
                                help='Learning rate (default: 0.1)')
    parser_backtest.add_argument('--epsilon', type=float, default=0.1,
                                help='Exploration rate (default: 0.1)')
    parser_backtest.add_argument('--no-learning', action='store_true',
                                help='Disable learning (evaluation mode)')
    parser_backtest.add_argument('--no-save', action='store_true',
                                help='Do not save trained model')
    parser_backtest.add_argument('--no-confidence-sizing', action='store_true',
                                help='Disable confidence-weighted position sizing (use all-in/all-out)')
    
    # Compare methods
    parser_compare = subparsers.add_parser('compare', help='Compare all methods')
    parser_compare.add_argument('--ticker', default='NVDA', help='Stock ticker (default: NVDA)')
    parser_compare.add_argument('--capital', type=float, default=10000.0,
                               help='Initial capital (default: 10000)')
    parser_compare.add_argument('--learning-rate', type=float, default=0.1,
                               help='Learning rate for Q-learning (default: 0.1)')
    parser_compare.add_argument('--epsilon', type=float, default=0.1,
                               help='Exploration rate (default: 0.1)')
    
    # Quickstart
    parser_quickstart = subparsers.add_parser('quickstart', help='Collect data + compare methods')
    parser_quickstart.add_argument('--ticker', default='NVDA', help='Stock ticker (default: NVDA)')
    parser_quickstart.add_argument('--api-key', help='Alpha Vantage API key')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'collect':
        collect_data(args.ticker, args.api_key)
    
    elif args.command == 'rag':
        run_rag_pipeline(args.ticker)
    
    elif args.command == 'backtest':
        run_backtest(
            ticker=args.ticker,
            method=args.method,
            initial_capital=args.capital,
            learning_rate=args.learning_rate,
            epsilon=args.epsilon,
            enable_learning=not args.no_learning,
            save_model=not args.no_save,
            use_confidence_sizing=not args.no_confidence_sizing
        )
    
    elif args.command == 'compare':
        compare_methods(
            ticker=args.ticker,
            initial_capital=args.capital,
            learning_rate=args.learning_rate,
            epsilon=args.epsilon
        )
    
    elif args.command == 'quickstart':
        print("\n🚀 QUICKSTART MODE\n")
        print("Assuming data already exists in data/ folder...")
        print("(If not, run: ./venv/bin/python data.py first)\n")
        
        # Compare all methods
        compare_methods(ticker=args.ticker)


if __name__ == "__main__":
    main()
