#!/usr/bin/env python3
import argparse
import sys
from run import run_backtest

def compare_strategies(ticker='NVDA'):
    """
    Run a side-by-side comparison of 3 Trading Strategies on Out-of-Sample Data (2024-Present).
    
    1. Hardcoded Priors (Baseline): No training, just fixed logical rules.
    2. Generalist Q-Learning: Tabular agent pre-trained on 2020-2023 data.
    3. Generalist DQN: Neural agent pre-trained on 2020-2023 data.
    """
    print(f"\n🚀 SOTA STRATEGY SHOWDOWN: {ticker}")
    print("================================================================================")
    print("Conditions:")
    print("  - Start Date: 2024-01-01 (Strict Out-of-Sample)")
    print("  - Learning:   DISABLED (Testing Pre-trained Knowledge)")
    print("================================================================================\n")

    results = {}

    # 1. Fixed Q (Baseline)
    print("\n🥊 ROUND 1: Fixed Q (Baseline)")
    print("--------------------------------------------------------------------------------")
    try:
        results['Fixed Q'] = run_backtest(
            ticker=ticker,
            method='fixed',
            start_date='2024-01-01',
            enable_learning=False,
            save_model=False,
            use_confidence_sizing=True
        )
    except Exception as e:
        print(f"❌ Failed: {e}")

    # 2. Generalist Q-Learning (Tabular)
    print("\n🥊 ROUND 2: Generalist Q-Learning (Tabular)")
    print("--------------------------------------------------------------------------------")
    try:
        results['Q-Learning'] = run_backtest(
            ticker=ticker,
            method='qlearning',
            model_path='backtest_results/q_table_walkforward_final.json',
            start_date='2024-01-01',
            enable_learning=False,
            save_model=False,
            use_confidence_sizing=True
        )
    except Exception as e:
        print(f"❌ Failed: {e}")

    # 3. Generalist DQN (Neural)
    print("\n🥊 ROUND 3: Generalist DQN (Neural)")
    print("--------------------------------------------------------------------------------")
    try:
        results['DQN'] = run_backtest(
            ticker=ticker,
            method='dqn',
            model_path='backtest_results/dqn_model_walkforward_final.pth',
            start_date='2024-01-01',
            enable_learning=False,
            save_model=False,
            use_confidence_sizing=True
        )
    except Exception as e:
        print(f"❌ Failed: {e}")

    # Comparison Table
    print(f"\n{'='*80}")
    print("🏆 FINAL SCOREBOARD")
    print(f"{'='*80}\n")
    
    print(f"{'Strategy':<20} | {'Return':<10} | {'Sharpe':<8} | {'Win Rate':<8} | {'Max DD':<10} | {'Trades':<6}")
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

    for name, m in results.items():
        if m:
            print(f"{name:<20} | {m.total_return_pct:>9.2f}% | {m.sharpe_ratio:>8.2f} | {m.win_rate:>7.2f}% | {m.max_drawdown_pct:>9.2f}% | {m.total_trades:>6}")

    print(f"\n{'='*80}\n")
    
    # Identify Winner
    if results:
        best_sharpe = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        best_return = max(results.items(), key=lambda x: x[1].total_return_pct)
        
        print(f"🥇 Winner (Sharpe): {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
        print(f"🥇 Winner (Return): {best_return[0]} ({best_return[1].total_return_pct:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SOTA Strategy Comparison")
    parser.add_argument('--ticker', default='NVDA', help="Stock Ticker (e.g. NVDA, TSM)")
    args = parser.parse_args()
    
    compare_strategies(args.ticker)
