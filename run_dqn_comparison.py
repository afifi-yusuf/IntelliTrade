"""
Compare Fixed Q-table, Q-learning, and DQN
"""

from pathlib import Path
from backtest import Backtester
from main import fetch_financial_metrics
from dqn_agent import DQNAgent

print("\n" + "=" * 100)
print("🔬 COMPARING: FIXED Q-TABLE vs Q-LEARNING vs DQN")
print("=" * 100)
print()

# Load financial data
print("📊 Loading financial data...")
financial_data = fetch_financial_metrics('NVDA')
print("✅ Data loaded\n")

results = []

# ============================================================================
# 1. FIXED Q-TABLE (Baseline)
# ============================================================================
print("=" * 100)
print("📋 METHOD 1: FIXED Q-TABLE")
print("=" * 100)

backtester1 = Backtester(
    ticker="NVDA",
    initial_capital=10000.0,
    enable_learning=False
)

metrics1 = backtester1.run_backtest(financial_data)
results.append(("Fixed Q-table", metrics1, "6 states"))

# ============================================================================
# 2. Q-LEARNING
# ============================================================================
print("\n" + "=" * 100)
print("🧠 METHOD 2: Q-LEARNING")
print("=" * 100)

backtester2 = Backtester(
    ticker="NVDA",
    initial_capital=10000.0,
    enable_learning=True,
    learning_rate=0.1,
    epsilon=0.1
)

metrics2 = backtester2.run_backtest(financial_data)
results.append(("Q-learning", metrics2, f"{len(backtester2.rl_agent.q_table)} states"))

# ============================================================================
# 3. DQN (DEEP Q-NETWORK)
# ============================================================================
print("\n" + "=" * 100)
print("🤖 METHOD 3: DEEP Q-NETWORK (DQN)")
print("=" * 100)

# Create DQN agent
state_dim = 396  # 384 + 7 + 3 + 2
dqn_agent = DQNAgent(
    state_dim=state_dim,
    learning_rate=0.0001,
    discount_factor=0.95,
    epsilon=0.15,
    batch_size=32,
    enable_learning=True
)

# Create backtester with DQN
backtester3 = Backtester(
    ticker="NVDA",
    initial_capital=10000.0,
    enable_learning=True  # Enable for update() calls
)

# Replace RL agent with DQN agent
backtester3.rl_agent = dqn_agent

# Run backtest
metrics3 = backtester3.run_backtest(financial_data)
results.append(("DQN", metrics3, f"{dqn_agent.get_learning_stats()['network_params']:,} params"))

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("📊 FINAL COMPARISON")
print("=" * 100)

print(f"{'Method':<20} | {'Final Value':>15} | {'Return':>10} | {'Sharpe':>7} | {'Max DD':>7} | {'Complexity':>15}")
print("-" * 100)

for name, metrics, complexity in results:
    print(
        f"{name:<20} | "
        f"${metrics.final_portfolio_value:>14,.2f} | "
        f"{metrics.total_return_pct:>9.2f}% | "
        f"{metrics.sharpe_ratio:>7.2f} | "
        f"{metrics.max_drawdown_pct:>6.2f}% | "
        f"{complexity:>15}"
    )

print("=" * 100)
print()

# Save models
results_dir = Path("backtest_results")
results_dir.mkdir(exist_ok=True)

if hasattr(backtester2.rl_agent, 'save_q_table'):
    backtester2.rl_agent.save_q_table(str(results_dir / "qlearning_final.json"))
    print(f"💾 Saved Q-learning table: {results_dir / 'qlearning_final.json'}")

if hasattr(dqn_agent, 'save_model'):
    dqn_agent.save_model(str(results_dir / "dqn_model_final.pth"))
    print(f"💾 Saved DQN model: {results_dir / 'dqn_model_final.pth'}")

print()
print("✅ Comparison complete!")
