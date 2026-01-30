"""
Run backtest with Q-learning enabled

This script runs the backtest with REAL reinforcement learning where the agent
learns from each trade and updates its Q-values based on rewards.
"""

import sys
from backtest import Backtester
from main import fetch_financial_metrics

def run_backtest_comparison():
    """
    Compare three scenarios:
    1. Fixed Q-table (no learning)
    2. Q-learning with exploration (ε=0.1)
    3. Q-learning with more exploration (ε=0.2)
    """
    
    print("\n" + "=" * 100)
    print("🧪 COMPARING LEARNING STRATEGIES")
    print("=" * 100)
    
    # Fetch financial data once
    print("\n📊 Loading financial data...")
    financial_data = fetch_financial_metrics("NVDA")
    
    results = []
    
    # Scenario 1: Fixed Q-table (no learning)
    print("\n" + "=" * 100)
    print("📋 SCENARIO 1: FIXED Q-TABLE (NO LEARNING)")
    print("=" * 100)
    
    backtester1 = Backtester(
        ticker="NVDA",
        initial_capital=10000.0,
        transaction_cost_pct=0.001,
        rebalance_frequency="weekly",
        enable_learning=False  # No learning
    )
    
    metrics1 = backtester1.run_backtest(financial_data)
    results.append(("Fixed Q-table", metrics1))
    
    # Scenario 2: Q-learning with low exploration
    print("\n" + "=" * 100)
    print("🧠 SCENARIO 2: Q-LEARNING (ε=0.1, α=0.1, γ=0.95)")
    print("=" * 100)
    
    backtester2 = Backtester(
        ticker="NVDA",
        initial_capital=10000.0,
        transaction_cost_pct=0.001,
        rebalance_frequency="weekly",
        enable_learning=True,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.1
    )
    
    metrics2 = backtester2.run_backtest(financial_data)
    results.append(("Q-learning (ε=0.1)", metrics2))
    
    # Scenario 3: Q-learning with more exploration
    print("\n" + "=" * 100)
    print("🔬 SCENARIO 3: Q-LEARNING WITH MORE EXPLORATION (ε=0.2, α=0.15, γ=0.95)")
    print("=" * 100)
    
    backtester3 = Backtester(
        ticker="NVDA",
        initial_capital=10000.0,
        transaction_cost_pct=0.001,
        rebalance_frequency="weekly",
        enable_learning=True,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.2
    )
    
    metrics3 = backtester3.run_backtest(financial_data)
    results.append(("Q-learning (ε=0.2)", metrics3))
    
    # Print comparison
    print("\n" + "=" * 100)
    print("📊 FINAL COMPARISON")
    print("=" * 100)
    print(f"{'Strategy':<25} | {'Final Value':>15} | {'Total Return':>12} | {'Sharpe':>8} | {'Max DD':>8}")
    print("-" * 100)
    
    for name, metrics in results:
        print(
            f"{name:<25} | "
            f"${metrics.final_portfolio_value:>14,.2f} | "
            f"{metrics.total_return_pct:>11.2f}% | "
            f"{metrics.sharpe_ratio:>8.2f} | "
            f"{metrics.max_drawdown_pct:>7.2f}%"
        )
    
    print("=" * 100)
    print()
    print("📝 Key Insights:")
    print("  • Fixed Q-table uses hardcoded heuristics (no adaptation)")
    print("  • Q-learning adapts to market conditions over time")
    print("  • Higher ε = more exploration (tries random actions)")
    print("  • Check learned_q_table_NVDA.json to see what it learned!")
    print()

if __name__ == "__main__":
    run_backtest_comparison()
