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
    fetch_financial_metrics,
    build_comprehensive_context,
    generate_llm_analysis,
    generate_agentic_analysis
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
    rationale: str = ""
    llm_analysis: str = ""


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
        rationale: str = "",
        llm_analysis: str = "",
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

            financial_state=financial_state,
            rationale=rationale,
            llm_analysis=llm_analysis
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
        use_llm_signals: bool = True,  # If False, run price-only (no LLM/earnings/financial context)
        start_date: Optional[str] = None,  # Start date for backtest (YYYY-MM-DD)
        read_only: bool = False, # Prevent writing to graph
        summary_cache: Optional[Dict[str, Dict]] = None  # Optional shared cache (e.g., across training epochs)
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
        self.use_llm_signals = use_llm_signals
        self.start_date = start_date
        self.read_only = read_only
        
        self.portfolio = Portfolio(initial_capital, transaction_cost_pct)
        self.vectorizer: Optional[NarrativeVectorizer] = None
        self.rl_agent: Optional[MarkovRLAgent] = None
        self._summary_cache: Dict[str, Dict] = summary_cache if summary_cache is not None else {}
        
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

    def _filter_financial_data_for_date(self, financial_data: Dict, current_date: datetime) -> Dict:
        """
        Build point-in-time financial input by filtering dated surprise records.
        Prevents leaking future quarterly surprises into past timesteps.
        """
        if not financial_data:
            return {}

        filtered = {
            "financials": financial_data.get("financials", {}),
            "surprise": []
        }

        surprises = financial_data.get("surprise", []) or []
        parsed = []
        for row in surprises:
            if not isinstance(row, dict):
                continue
            date_raw = row.get("reportedDate") or row.get("fiscalDateEnding") or row.get("date")
            try:
                row_dt = pd.to_datetime(date_raw) if date_raw else None
            except Exception:
                row_dt = None

            if row_dt is None or pd.isna(row_dt):
                continue
            if row_dt <= pd.Timestamp(current_date):
                parsed.append((row_dt, row))

        # Keep newest-first ordering expected by assess_financial_state.
        parsed.sort(key=lambda x: x[0], reverse=True)
        filtered["surprise"] = [row for _, row in parsed]
        return filtered

    def _quarter_reporting_date(self, quarter_id: str) -> datetime:
        """Approximate date when quarter information is first available."""
        try:
            year_str, quarter_str = quarter_id.split("-")
            year = int(year_str)
            q = int(quarter_str.replace("Q", ""))
        except Exception:
            # Defensive fallback if quarter format is malformed.
            return datetime.now()
        quarter_end_month = q * 3
        if quarter_end_month == 12:
            return datetime(year + 1, 1, 31)
        return datetime(year, quarter_end_month + 1, 30)

    def _build_context_meta(
        self,
        current_date: datetime,
        quarter_id: str,
        reliability: Dict,
        is_new_quarter: bool
    ) -> np.ndarray:
        """
        Build compact context quality features for RL gating.
        Order:
        [days_since_report_norm, is_new_quarter, agent_agreement, evidence_strength, parse_success]
        """
        report_date = self._quarter_reporting_date(quarter_id)
        days_since_report = max(0.0, (current_date - report_date).days)
        days_since_report_norm = float(np.clip(days_since_report / 120.0, 0.0, 1.0))
        return np.array([
            days_since_report_norm,
            1.0 if is_new_quarter else 0.0,
            float(np.clip(reliability.get("agent_agreement", 0.5), 0.0, 1.0)),
            float(np.clip(reliability.get("evidence_strength", 0.0), 0.0, 1.0)),
            float(np.clip(reliability.get("parse_success", 0.0), 0.0, 1.0)),
        ], dtype=np.float32)
    
    def generate_signal_for_date(
        self,
        date: datetime,
        price_history: pd.DataFrame,
        financial_data: Dict,
        position_ratio: float = 0.0,
        position_state: str = "FLAT"
    ) -> Dict:
        """
        Generate trading signal for a specific date using only data available at that time
        """
        try:
            # Time-aware financial view to avoid future surprise leakage.
            financial_data_pti = self._filter_financial_data_for_date(financial_data, date)

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

            if not self.use_llm_signals:
                # Price-only mode: no LLM narrative, no earnings/transcript/economic context,
                # and no structured multi-agent signals.
                embedding = np.zeros(self.vectorizer.embedding_dim, dtype=np.float64)
                structured_signals = np.zeros(6, dtype=np.float32)
                context_meta = np.zeros(5, dtype=np.float32)
                narrative_state = 1
                financial_state = 0
                llm_analysis_text = "Price-only DQN mode (LLM/context disabled)."

                from dqn_agent import DQNAgent
                signal_kwargs = dict(
                    narrative_vector=embedding,
                    price_history=price_history,
                    financial_data={},
                    narrative_state=narrative_state,
                    position_ratio=position_ratio,
                    structured_signals=structured_signals,
                )
                if isinstance(self.rl_agent, DQNAgent):
                    signal_kwargs["financial_state"] = financial_state
                    signal_kwargs["context_meta"] = context_meta
                    signal_kwargs['explore'] = self.enable_learning
                    signal_kwargs['position_state'] = position_state

                signal = self.rl_agent.get_signal(**signal_kwargs)
                signal['llm_analysis'] = llm_analysis_text
                return signal

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
            
            # Get most recent quarter
            recent_quarter = available_quarters[-1]
            is_new_quarter = False
            reliability = {
                "evidence_count": 0,
                "evidence_strength": 0.0,
                "agent_agreement": 0.5,
                "parse_success": 0.0,
                "agent_confidences": {}
            }
            
            # Try to get cached data first
            cached_summary = self._summary_cache.get(recent_quarter)
            if cached_summary is None and self.graph_db:
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
                    self._summary_cache.pop(recent_quarter, None)
                else:
                    self._summary_cache[recent_quarter] = cached_summary
            
            try:
                if cached_summary and 'embedding' in cached_summary:
                    # Cache hit - reuse existing analysis
                    llm_analysis_text = cached_summary['text']
                    embedding = np.array(cached_summary['embedding'], dtype=np.float64)
                    reliability = cached_summary.get("reliability", reliability)
                    cached_signals = cached_summary.get("structured_signals", {})
                    if isinstance(cached_signals, dict):
                        structured_signals = np.array([
                            float(cached_signals.get("revenue_trend", 0.0)),
                            float(cached_signals.get("margin_trend", 0.0)),
                            float(cached_signals.get("peer_relative", 0.0)),
                            float(cached_signals.get("historical_momentum", 0.0)),
                            float(cached_signals.get("risk_level", 0.5)),
                            float(cached_signals.get("overall_sentiment", 0.0))
                        ], dtype=np.float32)
                    elif isinstance(cached_signals, (list, tuple, np.ndarray)) and len(cached_signals) >= 6:
                        structured_signals = np.array(cached_signals[:6], dtype=np.float32)
                    else:
                        structured_signals = np.zeros(6, dtype=np.float32)
                else:
                    # Cache miss - use MULTI-AGENT pipeline
                    is_new_quarter = True
                    print(f"   🔄 Running Multi-Agent Analysis for {recent_quarter} (first time)...")
                    
                    # Build context for this quarter
                    economic_data = get_economic_data_for_quarter(recent_quarter, current_date=date)
                    transcript = load_earnings_transcript(self.ticker, recent_quarter, current_date=date)
                    
                    if not transcript or len(transcript) < 100:
                        # No transcript available - use neutral embedding and text
                        llm_analysis_text = "No data available for this quarter."
                        embedding = np.zeros(self.vectorizer.embedding_dim, dtype=np.float64)
                        structured_signals = np.zeros(6, dtype=np.float32)
                        signals = {
                            "revenue_trend": 0.0,
                            "margin_trend": 0.0,
                            "peer_relative": 0.0,
                            "historical_momentum": 0.0,
                            "risk_level": 0.5,
                            "overall_sentiment": 0.0
                        }
                        reliability = {
                            "evidence_count": 0,
                            "evidence_strength": 0.0,
                            "agent_agreement": 0.5,
                            "parse_success": 0.0,
                            "agent_confidences": {}
                        }
                        self._summary_cache[recent_quarter] = {
                            "text": llm_analysis_text,
                            "embedding": embedding.tolist(),
                            "max_quarter": recent_quarter,
                            "generated_for_date": date.date().isoformat(),
                            "structured_signals": signals,
                            "reliability": reliability,
                            "source_quarter": recent_quarter
                        }
                    else:
                        # Run the multi-agent agentic RAG pipeline
                        agentic_result = generate_agentic_analysis(
                            ticker=self.ticker,
                            quarter=recent_quarter,
                            transcript=transcript,
                            economic_data=economic_data,
                            financial_metrics=financial_data_pti,
                            graph_db_instance=self.graph_db,
                            vectorizer=self.vectorizer,
                            read_only=self.read_only
                        )
                        
                        llm_analysis_text = agentic_result.get("narrative", "")
                        if not llm_analysis_text:
                            llm_analysis_text = "Analysis generation failed."
                        
                        # Generate embedding from the full narrative
                        embedding = self.vectorizer.get_embedding(llm_analysis_text)
                        
                        # Extract structured signals for RL
                        signals = agentic_result.get("signals", {})
                        reliability = agentic_result.get("reliability", reliability)
                        structured_signals = np.array([
                            signals.get("revenue_trend", 0.0),
                            signals.get("margin_trend", 0.0),
                            signals.get("peer_relative", 0.0),
                            signals.get("historical_momentum", 0.0),
                            signals.get("risk_level", 0.5),
                            signals.get("overall_sentiment", 0.0)
                        ], dtype=np.float32)
                        
                        # Cache the analysis for future trades in this quarter
                        self._summary_cache[recent_quarter] = {
                            "text": llm_analysis_text,
                            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                            "max_quarter": recent_quarter,
                            "generated_for_date": date.date().isoformat(),
                            "structured_signals": signals,
                            "reliability": reliability,
                            "source_quarter": agentic_result.get("source_quarter", recent_quarter)
                        }

                        if self.graph_db and not self.read_only:
                            self.graph_db.ingest_quarterly_summary(
                                self.ticker,
                                recent_quarter,
                                llm_analysis_text,
                                embedding.tolist(),
                                max_quarter=recent_quarter,
                                generated_for_date=date.date().isoformat(),
                                structured_signals=signals,
                                reliability=reliability,
                                source_quarter=agentic_result.get("source_quarter", recent_quarter)
                            )
                            print(f"   ✅ Cached multi-agent analysis for {recent_quarter}")
            except Exception as e:
                logger.warning(f"⚠️ Multi-Agent pipeline unavailable. Falling back. Error: {e}")
                llm_analysis_text = "Technical Analysis Only (LLM Unavailable)"
                embedding = np.zeros(self.vectorizer.embedding_dim, dtype=np.float64)
                structured_signals = np.zeros(6, dtype=np.float32)
                reliability = {
                    "evidence_count": 0,
                    "evidence_strength": 0.0,
                    "agent_agreement": 0.5,
                    "parse_success": 0.0,
                    "agent_confidences": {}
                }

            context_meta = self._build_context_meta(
                current_date=date,
                quarter_id=recent_quarter,
                reliability=reliability,
                is_new_quarter=is_new_quarter
            )
                
            # Get narrative state (uses keyword analysis on text)
            narrative_state = self.vectorizer.get_narrative_state(text=llm_analysis_text)
            
            # Debug: Print sentiment analysis occasionally
            if np.random.random() < 0.1:  # 10% of trades
                print(f"   🔍 DEBUG | Quarter: {recent_quarter} | State: {narrative_state} (0=Bull, 1=Neut, 2=Bear) | Signals: {structured_signals}")
            
            # Generate RL signal
            signal_kwargs = dict(
                narrative_vector=embedding,
                price_history=price_history,
                financial_data=financial_data_pti,
                narrative_state=narrative_state,
                position_ratio=position_ratio,
                structured_signals=structured_signals
            )
            
            # For DQN, also pass explore flag
            from dqn_agent import DQNAgent
            if isinstance(self.rl_agent, DQNAgent):
                signal_kwargs["context_meta"] = context_meta
                signal_kwargs['explore'] = self.enable_learning
                signal_kwargs['position_state'] = position_state
            
            signal = self.rl_agent.get_signal(**signal_kwargs)
            
            # Inject LLM analysis text into signal for logging
            signal['llm_analysis'] = llm_analysis_text
            
            # Add fallback flag if applicable
            if "Technical Analysis Only" in llm_analysis_text:
                signal['fallback_active'] = True
            
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
        from dqn_agent import DQNAgent
        logger.info(f"📅 Total trading dates: {len(trading_dates)}")
        logger.info(f"   First: {trading_dates[0].date()}")
        logger.info(f"   Last: {trading_dates[-1].date()}")
        logger.info("")
        
        # Print header for trade log
        print("\n" + "="*120)
        print(f"{'Date':<12} | {'Action':<15} | {'Price':>10} | {'Shares':>8} | {'Cash':>14} | {'Portfolio Value':>16} | Progress")
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
                    "HOLD", current_price, date, 0.0, 1, 1, rationale="Insufficient history"
                )
                continue
            
            # Store portfolio value before trade (for RL reward calculation)
            portfolio_value_before = self.portfolio.get_value(current_price)
            
            # Compute position ratio (fraction of portfolio in stock)
            stock_value = self.portfolio.shares * current_price
            total_value = portfolio_value_before if portfolio_value_before > 0 else 1.0
            position_ratio = stock_value / total_value
            position_state = "LONG" if self.portfolio.shares > 0.001 else "FLAT"
            
            # Generate trading signal
            signal = self.generate_signal_for_date(
                date,
                price_history,
                financial_data,
                position_ratio=position_ratio,
                position_state=position_state
            )

            # Hard guard for atomic long-only transitions, regardless of model output:
            # FLAT: BUY/HOLD only, LONG: SELL/HOLD only.
            if position_state == "FLAT" and signal.get('action') == "SELL":
                signal['action'] = "HOLD"
            elif position_state == "LONG" and signal.get('action') == "BUY":
                signal['action'] = "HOLD"
            
            # Calculate position size based on confidence
            if isinstance(self.rl_agent, DQNAgent):
                # Atomic all-in/all-out for DQN.
                position_size = 1.0
            elif self.use_confidence_sizing:
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
                rationale=signal.get('rationale', ""),
                llm_analysis=signal.get('llm_analysis', ""),
                position_size_pct=position_size
            )
            
            # Get portfolio value after trade
            portfolio_value = self.portfolio.get_value(current_price)
            
            # Update RL agent with reward (Q-learning)
            if self.rl_agent and self.rl_agent.enable_learning:
                if signal.get('fallback_active', False):
                    # SKIP UPDATE if fallback was active (to avoid training on missing narrative)
                    logger.debug(f"⚠️ Training update skipped for {date.date()} due to missing narrative data.")
                else:
                    next_state_key = (
                        signal.get('narrative_state', 1),
                        signal.get('financial_state', 1)
                    )
                    next_q_values = signal.get('q_values', {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0})
                    self.rl_agent.update(portfolio_value, next_state_key, next_q_values)
            
            # Log progress for EVERY trade (not just every 10%)
            pct_complete = (i + 1) / len(trading_dates) * 100
            
            # Display logic
            action_display = signal['action']
            action_symbol = "⚪"
            
            if signal['action'] == "BUY":
                action_symbol = "🟢"
            elif signal['action'] == "SELL":
                if self.portfolio.shares <= 0.001 and not executed:
                     action_symbol = "⚪"
                     action_display = "SKIP (Empty)"
                else:
                     action_symbol = "🔴"
            
            print(
                f"{action_symbol} {date.date()} | "
                f"{action_display:15s} | "
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
            if years > 0 and final_value > 0 and self.initial_capital > 0:
                annualized_return_pct = (((final_value / self.initial_capital) ** (1 / years)) - 1) * 100
            else:
                annualized_return_pct = 0
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
                "financial_state": t.financial_state,
                "rationale": t.rationale,
                "llm_analysis": t.llm_analysis
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
                'financial_state': t.financial_state,
                'rationale': t.rationale,
                'llm_analysis': t.llm_analysis
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
