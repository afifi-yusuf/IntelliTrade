"""
Backtesting Engine for IntelliSense Trading System

This module simulates historical trading performance using the Context Enriched
Agentic RAG system with RL-based trading signals.

Key Features:
- Realistic portfolio simulation (cash, positions, value tracking)
- Transaction costs (optional)
- Performance metrics (returns, Sharpe, drawdown, win rate)
- Trade log and equity curve generation
- Avoids look-ahead bias (only uses data available at each point in time)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

from llm import (
    fetch_stock_price_history,
    get_economic_data_for_quarter,
    load_earnings_transcript,
    build_comprehensive_context,
    generate_llm_analysis
)
from vectorizer import NarrativeVectorizer
from rl_agent import MarkovRLAgent
from graph_manager import Neo4jHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade execution"""
    date: datetime
    action: str  # BUY, SELL, HOLD
    shares: float
    price: float
    value: float
    cash_after: float
    portfolio_value: float
    confidence: float
    narrative_state: int
    financial_state: int


@dataclass
class PerformanceMetrics:
    """Summary of backtest performance"""
    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    buy_trades: int
    sell_trades: int
    hold_days: int
    avg_trade_size: float
    history: List[Dict] = None


class Portfolio:
    """Manages cash, positions, and portfolio value"""
    
    def __init__(self, initial_capital: float = 10000.0, transaction_cost_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0.0
        self.transaction_cost_pct = transaction_cost_pct
        self.trade_history: List[Trade] = []
        
    def get_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        return self.cash + (self.shares * current_price)
    
    def execute_trade(
        self,
        action: str,
        current_price: float,
        date: datetime,
        confidence: float,
        narrative_state: int,
        financial_state: int,
        position_size_pct: float = 1.0
    ) -> bool:
        """
        Execute a trade based on signal
        
        Args:
            action: BUY, SELL, or HOLD
            current_price: Current stock price
            date: Trade date
            confidence: Signal confidence (0-1)
            narrative_state: Narrative state (0=Bullish, 1=Neutral, 2=Bearish)
            financial_state: Financial state (0=Good, 1=Weak)
            position_size_pct: Percentage of available capital to use (0-1)
        
        Returns:
            True if trade executed, False otherwise
        """
        executed = False
        shares_traded = 0.0
        trade_value = 0.0
        
        portfolio_value_before = self.get_value(current_price)
        
        if action == "BUY" and self.cash > current_price:
            # Buy as many shares as possible with available cash
            max_shares = (self.cash * position_size_pct) / (current_price * (1 + self.transaction_cost_pct))
            shares_traded = np.floor(max_shares)  # Can only buy whole shares
            
            if shares_traded > 0:
                cost = shares_traded * current_price
                transaction_cost = cost * self.transaction_cost_pct
                total_cost = cost + transaction_cost
                
                self.cash -= total_cost
                self.shares += shares_traded
                trade_value = cost
                executed = True
                
        elif action == "SELL" and self.shares > 0:
            # Sell shares (respecting position_size_pct for partial sells)
            shares_to_sell = np.floor(self.shares * position_size_pct)
            
            if shares_to_sell > 0:
                shares_traded = shares_to_sell
                proceeds = shares_traded * current_price
                transaction_cost = proceeds * self.transaction_cost_pct
                net_proceeds = proceeds - transaction_cost
                
                self.cash += net_proceeds
                self.shares -= shares_traded
                trade_value = proceeds
                executed = True
        
        # Record trade (including HOLD)
        trade = Trade(
            date=date,
            action=action,
            shares=shares_traded,
            price=current_price,
            value=trade_value,
            cash_after=self.cash,
            portfolio_value=self.get_value(current_price),
            confidence=confidence,
            narrative_state=narrative_state,
            financial_state=financial_state
        )
        self.trade_history.append(trade)
        
        return executed


class Backtester:
    """
    Backtesting engine for the IntelliSense trading system
    """
    
    def __init__(
        self,
        ticker: str = "NVDA",
        initial_capital: float = 10000.0,
        transaction_cost_pct: float = 0.001,
        rebalance_frequency: str = "weekly",  # daily, weekly, monthly
        enable_learning: bool = True,  # Enable Q-learning
        learning_rate: float = 0.1,  # α
        discount_factor: float = 0.95,  # γ
        epsilon: float = 0.1,  # ε exploration rate

        use_confidence_sizing: bool = True,  # Weight position size by confidence
        start_date: Optional[str] = None,  # Start date for backtest (YYYY-MM-DD)
        read_only: bool = False # Prevent writing to graph
    ):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.rebalance_frequency = rebalance_frequency
        self.enable_learning = enable_learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.use_confidence_sizing = use_confidence_sizing
        self.start_date = start_date
        self.read_only = read_only
        
        self.portfolio = Portfolio(initial_capital, transaction_cost_pct)
        self.vectorizer: Optional[NarrativeVectorizer] = None
        self.rl_agent: Optional[MarkovRLAgent] = None
        
        # Initialize Neo4j handler
        try:
            self.graph_db = Neo4jHandler()
        except Exception as e:
            logger.warning(f"Neo4j initialization failed: {e}. Continuing without cache.")
            self.graph_db = None
        
        # Load full historical data
        self.price_data = fetch_stock_price_history(ticker)
        if self.price_data is None or self.price_data.empty:
            raise ValueError(f"No price data available for {ticker}")
        
        # Ensure timestamp column exists and is datetime
        if 'timestamp' not in self.price_data.columns:
            # Check if there's a date column or an unnamed index column
            if 'date' in self.price_data.columns:
                self.price_data.rename(columns={'date': 'timestamp'}, inplace=True)
            elif self.price_data.columns[0] == '' or 'Unnamed' in str(self.price_data.columns[0]):
                # First column is likely the date
                self.price_data.rename(columns={self.price_data.columns[0]: 'timestamp'}, inplace=True)
            else:
                raise ValueError("Could not find date/timestamp column in price data")
        
        self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
        self.price_data = self.price_data.sort_values('timestamp').reset_index(drop=True)
        
        # Filter by start_date if provided
        if self.start_date:
            try:
                cutoff = pd.to_datetime(self.start_date)
                self.price_data = self.price_data[self.price_data['timestamp'] >= cutoff]
                if self.price_data.empty:
                    raise ValueError(f"No price data available after {self.start_date}")
                logger.info(f"🗓️  Filtered backtest to start from: {self.start_date}")
            except Exception as e:
                logger.error(f"Failed to filter by start_date: {e}")
                
        logger.info(f"📊 Loaded {len(self.price_data)} days of price data")
        logger.info(f"   Date range: {self.price_data['timestamp'].min()} to {self.price_data['timestamp'].max()}")
        
    def get_trading_dates(self) -> List[datetime]:
        """Generate list of trading dates based on rebalance frequency"""
        all_dates = self.price_data['timestamp'].tolist()
        
        if self.rebalance_frequency == "daily":
            return all_dates
        elif self.rebalance_frequency == "weekly":
            # Trade every Monday (or first available day of week)
            weekly_dates = []
            current_week = None
            for date in all_dates:
                week_num = date.isocalendar()[1]
                if week_num != current_week:
                    weekly_dates.append(date)
                    current_week = week_num
            return weekly_dates
        elif self.rebalance_frequency == "monthly":
            # Trade on first day of each month
            monthly_dates = []
            current_month = None
            for date in all_dates:
                month = (date.year, date.month)
                if month != current_month:
                    monthly_dates.append(date)
                    current_month = month
            return monthly_dates
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.rebalance_frequency}")
    
    def get_price_history_up_to_date(self, current_date: datetime) -> pd.DataFrame:
        """Get price history available up to (but not including) current date"""
        return self.price_data[self.price_data['timestamp'] < current_date].copy()
    
    def get_current_quarter(self, date: datetime) -> str:
        """Determine which quarter a date falls into"""
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}-Q{quarter}"
    
    def get_available_quarters(self, current_date: datetime) -> List[str]:
        """Get list of quarters with data available before current date"""
        quarters = []
        start_year = 2021
        end_year = current_date.year
        
        for year in range(start_year, end_year + 1):
            for q in range(1, 5):
                quarter_id = f"{year}-Q{q}"
                # Check if this quarter's data would be available
                # Assume earnings are reported ~1 month after quarter end
                quarter_end_month = q * 3
                if quarter_end_month == 12:
                    reporting_date = datetime(year + 1, 1, 31)
                else:
                    reporting_date = datetime(year, quarter_end_month + 1, 30)
                
                if reporting_date < current_date:
                    quarters.append(quarter_id)
        
        return quarters[-3:] if len(quarters) >= 3 else quarters  # Use last 3 quarters
    
    def generate_signal_for_date(
        self,
        date: datetime,
        price_history: pd.DataFrame,
        financial_data: Dict
    ) -> Dict:
        """
        Generate trading signal for a specific date using only data available at that time
        """
        try:
            # Get available quarters (no look-ahead bias)
            available_quarters = self.get_available_quarters(date)
            
            if not available_quarters:
                logger.debug(f"No quarterly data available for {date.date()}")
                return {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "narrative_state": 1,
                    "financial_state": 1,
                    "reason": "Insufficient historical data"
                }
            
            # Initialize models if not already done (only once)
            if self.vectorizer is None:
                logger.info("🔢 Initializing vectorizer (one-time setup)...")
                self.vectorizer = NarrativeVectorizer(model_name='all-MiniLM-L6-v2')
                logger.info(f"   ✅ Vectorizer ready ({self.vectorizer.embedding_dim} dimensions)")
            
            if self.rl_agent is None:
                logger.info("🤖 Initializing RL agent (one-time setup)...")
                self.rl_agent = MarkovRLAgent(
                    narrative_dim=self.vectorizer.embedding_dim,
                    learning_rate=self.learning_rate,
                    discount_factor=self.discount_factor,
                    epsilon=self.epsilon,
                    enable_learning=self.enable_learning
                )
                if self.enable_learning:
                    logger.info(f"   ✅ RL agent ready (Learning ENABLED: α={self.learning_rate}, γ={self.discount_factor}, ε={self.epsilon})")
                else:
                    logger.info("   ✅ RL agent ready (Learning DISABLED - using fixed Q-table)")
            
            # Get most recent quarter
            recent_quarter = available_quarters[-1]
            
            # Try to get cached data first
            cached_summary = None
            if self.graph_db:
                cached_summary = self.graph_db.retrieve_quarterly_summary(self.ticker, recent_quarter)
            
            if cached_summary and 'embedding' in cached_summary:
                cache_max_quarter = cached_summary.get("max_quarter")
                cache_generated_for = cached_summary.get("generated_for_date")
                cache_valid = cache_max_quarter == recent_quarter
                if cache_valid and cache_generated_for:
                    try:
                        cache_generated_dt = datetime.fromisoformat(str(cache_generated_for))
                        if cache_generated_dt > date:
                            cache_valid = False
                    except Exception:
                        cache_valid = False
                if not cache_valid:
                    cached_summary = None
            
            if cached_summary and 'embedding' in cached_summary:
                # Cache hit - reuse existing analysis
                llm_analysis_text = cached_summary['text']
                embedding = np.array(cached_summary['embedding'], dtype=np.float64)
            else:
                # Cache miss - generate ONCE and cache it
                print(f"   🔄 Generating LLM analysis for {recent_quarter} (first time)...")
                
                # Build context for this quarter
                economic_data = get_economic_data_for_quarter(recent_quarter, current_date=date)
                transcript = load_earnings_transcript(self.ticker, recent_quarter, current_date=date)
                
                if not transcript or len(transcript) < 100:
                    # No transcript available - use neutral embedding and text
                    llm_analysis_text = "No data available for this quarter."
                    embedding = np.zeros(self.vectorizer.embedding_dim, dtype=np.float64)
                else:
                    context, _, _ = build_comprehensive_context(
                        self.ticker, recent_quarter, financial_data, current_date=date
                    )
                    
                    # Retrieve historical context
                    graph_context = ""
                    if self.graph_db:
                        graph_context = self.graph_db.retrieve_all_context(self.ticker, max_quarter=recent_quarter)
                    
                    # Generate LLM analysis (this is the slow part, but only happens once per quarter)
                    llm_analysis_text = generate_llm_analysis(context, graph_context)
                    
                    # Generate embedding
                    embedding = self.vectorizer.get_embedding(llm_analysis_text)
                    
                    # Cache it for future trades in this quarter
                    if self.graph_db:
                        self.graph_db.ingest_quarterly_summary(
                            self.ticker,
                            recent_quarter,
                            llm_analysis_text,
                            embedding.tolist(),
                            max_quarter=recent_quarter,
                            generated_for_date=date.date().isoformat()
                        )
                        print(f"   ✅ Cached analysis for {recent_quarter}")
            
            # Get narrative state (uses keyword analysis on text)
            narrative_state = self.vectorizer.get_narrative_state(text=llm_analysis_text)
            
            # Debug: Print sentiment analysis occasionally
            if np.random.random() < 0.1:  # 10% of trades
                print(f"   🔍 DEBUG | Quarter: {recent_quarter} | State: {narrative_state} (0=Bull, 1=Neut, 2=Bear) | Text preview: {llm_analysis_text[:80]}...")
            
            # Generate RL signal
            signal = self.rl_agent.get_signal(
                narrative_vector=embedding,
                price_history=price_history,
                financial_data=financial_data,
                narrative_state=narrative_state
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Error generating signal for {date.date()}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "narrative_state": 1,
                "financial_state": 1,
                "reason": f"Error: {str(e)}"
            }
    
    def run_backtest(self, financial_data: Dict) -> PerformanceMetrics:
        """
        Run the backtest simulation
        """
        logger.info("=" * 80)
        logger.info("🔄 STARTING BACKTEST")
        logger.info("=" * 80)
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"📊 Ticker: {self.ticker}")
        logger.info(f"📅 Rebalance Frequency: {self.rebalance_frequency}")
        logger.info(f"💸 Transaction Cost: {self.transaction_cost_pct * 100:.3f}%")
        logger.info("")
        
        trading_dates = self.get_trading_dates()
        logger.info(f"📅 Total trading dates: {len(trading_dates)}")
        logger.info(f"   First: {trading_dates[0].date()}")
        logger.info(f"   Last: {trading_dates[-1].date()}")
        logger.info("")
        
        # Print header for trade log
        print("\n" + "="*120)
        print(f"{'Date':<12} | {'Action':<4} | {'Price':>10} | {'Shares':>8} | {'Cash':>14} | {'Portfolio Value':>16} | Progress")
        print("="*120)
        
        # Run simulation
        for i, date in enumerate(trading_dates):
            # Get price for this date
            price_row = self.price_data[self.price_data['timestamp'] == date]
            if price_row.empty:
                continue
            
            # Use adjusted_close to handle stock splits properly
            if 'adjusted_close' in price_row.columns:
                current_price = price_row.iloc[0]['adjusted_close']
            else:
                current_price = price_row.iloc[0]['close']
            
            # Get historical price data available up to this date
            price_history = self.get_price_history_up_to_date(date)
            
            if len(price_history) < 30:  # Need minimum history
                # Just hold for now
                self.portfolio.execute_trade(
                    "HOLD", current_price, date, 0.0, 1, 1
                )
                continue
            
            # Store portfolio value before trade (for RL reward calculation)
            portfolio_value_before = self.portfolio.get_value(current_price)
            
            # Generate trading signal
            signal = self.generate_signal_for_date(date, price_history, financial_data)
            
            # Calculate position size based on confidence
            if self.use_confidence_sizing:
                # Confidence-weighted: Use confidence (0-1) as position size
                # Ensure minimum 10% and maximum 100%
                confidence = signal['confidence']
                position_size = np.clip(confidence, 0.1, 1.0) if confidence > 0 else 0.5
            else:
                # All-in/all-out: Always use 100% of capital
                position_size = 1.0
            
            # Execute trade
            executed = self.portfolio.execute_trade(
                action=signal['action'],
                current_price=current_price,
                date=date,
                confidence=signal['confidence'],
                narrative_state=signal.get('narrative_state', 1),
                financial_state=signal.get('financial_state', 1),
                position_size_pct=position_size
            )
            
            # Get portfolio value after trade
            portfolio_value = self.portfolio.get_value(current_price)
            
            # Update RL agent with reward (Q-learning)
            if self.rl_agent and self.rl_agent.enable_learning:
                next_state_key = (
                    signal.get('narrative_state', 1),
                    signal.get('financial_state', 1)
                )
                next_q_values = signal.get('q_values', {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0})
                self.rl_agent.update(portfolio_value, next_state_key, next_q_values)
            
            # Log progress for EVERY trade (not just every 10%)
            pct_complete = (i + 1) / len(trading_dates) * 100
            
            # Show different symbol based on action
            action_symbol = "🟢" if signal['action'] == "BUY" else "🔴" if signal['action'] == "SELL" else "⚪"
            
            print(
                f"{action_symbol} {date.date()} | "
                f"{signal['action']:4s} | "
                f"Price: ${current_price:7.2f} | "
                f"Shares: {self.portfolio.shares:6.1f} | "
                f"Cash: ${self.portfolio.cash:10,.2f} | "
                f"Portfolio: ${portfolio_value:12,.2f} | "
                f"({pct_complete:5.1f}%)",
                flush=True
            )
        
        # Calculate final metrics
        final_price = self.price_data.iloc[-1]['close']
        final_value = self.portfolio.get_value(final_price)
        
        metrics = self._calculate_metrics(final_value)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 BACKTEST COMPLETE")
        logger.info("=" * 80)
        
        # Diagnostic summary
        if self.portfolio.trade_history:
            trades = self.portfolio.trade_history
            total_steps = len(trades)
            action_counts = {
                "BUY": sum(1 for t in trades if t.action == "BUY"),
                "SELL": sum(1 for t in trades if t.action == "SELL"),
                "HOLD": sum(1 for t in trades if t.action == "HOLD")
            }
            market_days = sum(1 for t in trades if t.shares > 0)
            time_in_market_pct = (market_days / total_steps) * 100 if total_steps > 0 else 0.0
            buy_sizes = [
                (t.value / t.portfolio_value) for t in trades
                if t.action == "BUY" and t.value > 0 and t.portfolio_value > 0
            ]
            avg_buy_size_pct = (np.mean(buy_sizes) * 100) if buy_sizes else 0.0
            
            logger.info("📌 DIAGNOSTICS")
            logger.info(f"Action Counts: BUY={action_counts['BUY']} | SELL={action_counts['SELL']} | HOLD={action_counts['HOLD']}")
            logger.info(f"Time in Market: {time_in_market_pct:.2f}%")
            logger.info(f"Avg Buy Size: {avg_buy_size_pct:.2f}% of portfolio value")
        
        # Print RL learning statistics
        if self.rl_agent and self.rl_agent.enable_learning:
            logger.info("")
            logger.info("🧠 Q-LEARNING STATISTICS")
            logger.info("=" * 80)
            stats = self.rl_agent.get_learning_stats()
            logger.info(f"Total Q-value updates: {stats['total_updates']}")
            logger.info(f"Average reward: {stats.get('avg_reward', 0.0):.4f}")
            logger.info(f"Reward std dev: {stats.get('std_reward', 0.0):.4f}")
            logger.info(f"Avg Q-value change: {stats.get('avg_q_change', 0.0):.4f}")
            logger.info(f"Q-table size (states): {stats.get('q_table_size', 0)}")
            logger.info(f"Exploration rate (ε): {stats.get('exploration_rate', 0.0):.2f}")
            logger.info("")
            
            # Save learned model (Q-table or DQN)
            results_path = Path("backtest_results")
            results_path.mkdir(exist_ok=True)
            
            # Check which type of agent and call appropriate save method
            if hasattr(self.rl_agent, 'save_q_table'):
                # Q-learning agent
                q_table_path = results_path / f"learned_q_table_{self.ticker}.json"
                self.rl_agent.save_q_table(str(q_table_path))
            elif hasattr(self.rl_agent, 'save_model'):
                # DQN agent
                model_path = results_path / f"dqn_model_{self.ticker}.pth"
                self.rl_agent.save_model(str(model_path))
        
        return metrics
    
    def _calculate_metrics(self, final_value: float) -> PerformanceMetrics:
        """Calculate performance metrics from trade history"""
        
        # Basic returns
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate annualized return
        if len(self.portfolio.trade_history) > 0:
            first_date = self.portfolio.trade_history[0].date
            last_date = self.portfolio.trade_history[-1].date
            years = (last_date - first_date).days / 365.25
            annualized_return_pct = (((final_value / self.initial_capital) ** (1 / years)) - 1) * 100 if years > 0 else 0
        else:
            annualized_return_pct = 0
        
        # Sharpe ratio (simplified - using daily returns)
        portfolio_values = [t.portfolio_value for t in self.portfolio.trade_history]
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = self.initial_capital
        max_drawdown_pct = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = ((peak - value) / peak) * 100
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown
        
        # Trade statistics
        trades = self.portfolio.trade_history
        buy_trades = sum(1 for t in trades if t.action == "BUY" and t.shares > 0)
        sell_trades = sum(1 for t in trades if t.action == "SELL" and t.shares > 0)
        hold_days = sum(1 for t in trades if t.action == "HOLD")
        total_trades = buy_trades + sell_trades
        
        # Win rate (simplified - count trades where portfolio increased)
        winning_trades = 0
        for i in range(1, len(trades)):
            if trades[i].portfolio_value > trades[i-1].portfolio_value:
                winning_trades += 1
        win_rate = (winning_trades / len(trades) * 100) if len(trades) > 0 else 0
        # Average trade size
        trade_values = [t.value for t in trades if t.value > 0]
        avg_trade_size = np.mean(trade_values) if trade_values else 0
        
        # Prepare history for serialization
        history = []
        for t in self.portfolio.trade_history:
            history.append({
                "date": t.date.isoformat(),
                "action": t.action,
                "shares": t.shares,
                "price": t.price,
                "value": t.value,
                "cash": t.cash_after,
                "portfolio_value": t.portfolio_value,
                "confidence": t.confidence,
                "narrative_state": t.narrative_state,
                "financial_state": t.financial_state
            })

        return PerformanceMetrics(
            initial_capital=self.initial_capital,
            final_portfolio_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            hold_days=hold_days,
            avg_trade_size=avg_trade_size,
            history=history
        )
    
    def export_results(self, metrics: PerformanceMetrics, output_dir: str = "backtest_results"):
        """Export backtest results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Trade log CSV
        trades_df = pd.DataFrame([
            {
                'date': t.date,
                'action': t.action,
                'shares': t.shares,
                'price': t.price,
                'value': t.value,
                'cash_after': t.cash_after,
                'portfolio_value': t.portfolio_value,
                'confidence': t.confidence,
                'narrative_state': t.narrative_state,
                'financial_state': t.financial_state
            }
            for t in self.portfolio.trade_history
        ])
        trades_df.to_csv(output_path / "trade_log.csv", index=False)
        
        # 2. Equity curve CSV
        equity_curve = pd.DataFrame({
            'date': [t.date for t in self.portfolio.trade_history],
            'portfolio_value': [t.portfolio_value for t in self.portfolio.trade_history]
        })
        equity_curve.to_csv(output_path / "equity_curve.csv", index=False)
        
        # 3. Performance summary
        with open(output_path / "performance_summary.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("BACKTEST PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Initial Capital:        ${metrics.initial_capital:,.2f}\n")
            f.write(f"Final Portfolio Value:  ${metrics.final_portfolio_value:,.2f}\n")
            f.write(f"Total Return:           {metrics.total_return_pct:+.2f}%\n")
            f.write(f"Annualized Return:      {metrics.annualized_return_pct:+.2f}%\n")
            f.write(f"Sharpe Ratio:           {metrics.sharpe_ratio:.2f}\n")
            f.write(f"Max Drawdown:           {metrics.max_drawdown_pct:.2f}%\n")
            f.write(f"Win Rate:               {metrics.win_rate:.2f}%\n")
            f.write(f"\n")
            f.write(f"Total Trades:           {metrics.total_trades}\n")
            f.write(f"  - Buy Trades:         {metrics.buy_trades}\n")
            f.write(f"  - Sell Trades:        {metrics.sell_trades}\n")
            f.write(f"  - Hold Days:          {metrics.hold_days}\n")
            f.write(f"Average Trade Size:     ${metrics.avg_trade_size:,.2f}\n")
        
        logger.info(f"📁 Results exported to: {output_path.absolute()}")


def print_performance_summary(metrics: PerformanceMetrics):
    """Print formatted performance summary"""
    print("\n" + "=" * 80)
    print("📊 BACKTEST PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"💰 Initial Capital:        ${metrics.initial_capital:,.2f}")
    print(f"💵 Final Portfolio Value:  ${metrics.final_portfolio_value:,.2f}")
    print(f"📈 Total Return:           {metrics.total_return_pct:+.2f}%")
    print(f"📊 Annualized Return:      {metrics.annualized_return_pct:+.2f}%")
    print(f"📉 Sharpe Ratio:           {metrics.sharpe_ratio:.2f}")
    print(f"⚠️  Max Drawdown:           {metrics.max_drawdown_pct:.2f}%")
    print(f"🎯 Win Rate:               {metrics.win_rate:.2f}%")
    print("")
    print(f"🔄 Total Trades:           {metrics.total_trades}")
    print(f"   • Buy Trades:          {metrics.buy_trades}")
    print(f"   • Sell Trades:         {metrics.sell_trades}")
    print(f"   • Hold Days:           {metrics.hold_days}")
    print(f"💵 Average Trade Size:     ${metrics.avg_trade_size:,.2f}")
    print("=" * 80)


if __name__ == "__main__":
    # Load financial data
    from llm import fetch_financial_metrics
    
    ticker = "NVDA"
    financial_data = fetch_financial_metrics(ticker)
    
    print("\n💡 INFO: Backtest uses lazy caching - LLM analyses are generated once per quarter")
    print("         and cached in Neo4j. Subsequent trades reuse cached analyses.\n")
    
    # Run backtest (now FAST - no LLM calls!)
    backtester = Backtester(
        ticker=ticker,
        initial_capital=10000.0,
        transaction_cost_pct=0.001,  # 0.1% per trade
        rebalance_frequency="weekly"  # Trade weekly
    )
    
    metrics = backtester.run_backtest(financial_data)
    
    # Print results
    print_performance_summary(metrics)
    
    # Export results
    backtester.export_results(metrics)
