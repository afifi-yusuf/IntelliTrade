import numpy as np
import pandas as pd

class MarkovRLAgent:
    """
    A Reinforcement Learning Agent that uses both LLM vectorized output and stock price history.
    
    State Space (S): (Narrative_Vector, Stock_Features, Financial_State)
    Action Space (A): BUY (1), HOLD (0), SELL (-1)
    
    The agent now considers:
    - Narrative_Vector: TF-IDF vectorized LLM summary (continuous features)
    - Stock_Features: Technical indicators from price history (momentum, volatility, trend)
    - Financial_State: Discretized fundamental metrics
    
    **NOW WITH REAL Q-LEARNING:**
    - Updates Q-values after each trade based on rewards
    - Explores vs exploits using epsilon-greedy
    - Learns optimal policy over time
    """
    def __init__(self, narrative_dim=100, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, enable_learning=True):
        """
        Args:
            narrative_dim: Dimension of narrative embedding
            learning_rate (α): How much to update Q-values (0-1)
            discount_factor (γ): How much to value future rewards (0-1)
            epsilon (ε): Exploration rate for epsilon-greedy (0-1)
            enable_learning: If True, updates Q-values. If False, uses fixed priors (for comparison)
        """
        # Q-Table: Maps (Narrative_State, Financial_State) -> {Action: Expected_Reward}
        self.q_table = {}
        self.narrative_dim = narrative_dim
        
        # Learning parameters
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.enable_learning = enable_learning
        
        # Track learning progress
        self.learning_history = []
        self.total_updates = 0
        
        # Track state/action transitions for learning
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None
        self.current_state = None
        self.current_action = None
        
        self._initialize_priors()

    def _initialize_priors(self):
        """
        Initialize Q-Values with common sense financial logic.
        States:
          Narrative: 0 (Bullish), 1 (Neutral), 2 (Bearish)
          Financial: 0 (Good), 1 (Bad)
        """
        # Format: (Narrative, Financial) -> {Action: Score}
        
        # 1. Bullish Narrative + Good Financials -> BUY (but not overwhelming)
        self.q_table[(0, 0)] = {
            "BUY": 0.5, "HOLD": 0.3, "SELL": -0.1
        }
        
        # 2. Bullish Narrative + Bad Financials -> Caution: SELL or HOLD
        self.q_table[(0, 1)] = {
            "BUY": 0.1, "HOLD": 0.4, "SELL": 0.3
        }
        
        # 3. Bearish Narrative + Good Financials -> SELL (protect gains)
        self.q_table[(2, 0)] = {
            "BUY": 0.1, "HOLD": 0.3, "SELL": 0.4
        }
        
        # 4. Bearish Narrative + Bad Financials -> Strong SELL
        self.q_table[(2, 1)] = {
            "BUY": -0.6, "HOLD": 0.1, "SELL": 0.8
        }
        
        # 5. Neutral + Good Financials -> Slight BUY bias
        self.q_table[(1, 0)] = {"BUY": 0.3, "HOLD": 0.4, "SELL": 0.1}
        
        # 6. Neutral + Bad Financials -> SELL bias
        self.q_table[(1, 1)] = {"BUY": -0.3, "HOLD": 0.3, "SELL": 0.6}

    def extract_stock_features(self, price_history: pd.DataFrame) -> dict:
        """
        Extract technical indicators from stock price history.
        Returns a dictionary of features for the RL agent.
        """
        if price_history is None or len(price_history) < 20:
            return {
                "momentum": 0.0,
                "volatility": 0.0,
                "trend": 0.0,
                "rsi": 50.0,
                "price_change_1d": 0.0,
                "price_change_5d": 0.0,
                "price_change_20d": 0.0
            }
        
        # Ensure we have a 'close' column (try different column name formats)
        close_col = None
        for col in ['close', 'Close', '5. adjusted close', 'adjusted_close']:
            if col in price_history.columns:
                close_col = col
                break
        
        if close_col is None:
            return {"momentum": 0.0, "volatility": 0.0, "trend": 0.0, "rsi": 50.0,
                    "price_change_1d": 0.0, "price_change_5d": 0.0, "price_change_20d": 0.0}
        
        prices = price_history[close_col].values
        
        # 1. Price Changes (momentum indicators)
        price_change_1d = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0.0
        price_change_5d = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) > 5 else 0.0
        price_change_20d = (prices[-1] - prices[-20]) / prices[-20] * 100 if len(prices) > 20 else 0.0
        
        # 2. Volatility (standard deviation of returns)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:]) * 100 if len(returns) > 20 else 0.0
        
        # 3. Trend (Simple Moving Average comparison)
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
        trend = (sma_20 - sma_50) / sma_50 * 100 if sma_50 > 0 else 0.0
        
        # 4. RSI (Relative Strength Index)
        rsi = self._calculate_rsi(prices, period=14)
        
        # 5. Overall momentum score
        momentum = (price_change_5d + price_change_20d) / 2
        
        return {
            "momentum": momentum,
            "volatility": volatility,
            "trend": trend,
            "rsi": rsi,
            "price_change_1d": price_change_1d,
            "price_change_5d": price_change_5d,
            "price_change_20d": price_change_20d
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def assess_financial_state(self, metrics: dict) -> int:
        """
        Discretize financial metrics into a state.
        0 = Good/Healthy
        1 = Bad/Weak
        """
        # Simple heuristic rule based on Growth and Surprise
        # In a real RL system, this would be a learned policy.
        
        score = 0
        
        # 1. Revenue Growth (if available)
        # Using a dummy check for now as we might not have calculated growth explicitly
        # We'll look for positive sentiment in surprises
        surprises = metrics.get("surprise", [])
        if surprises and len(surprises) > 0:
            last_surprise = surprises[0].get("surprisePercent", 0)
            if last_surprise > 0:
                score += 1
            else:
                score -= 1
                
        # 2. Profit Margin check (from basic financials)
        financials = metrics.get("financials", {}).get("metric", {})
        margin = financials.get("netProfitMarginTTM")
        
        if margin is None:
            # Fallback for CSV data which might use 'ProfitMargin'
            margin = financials.get("ProfitMargin", 0)
            # Normalize: If margin is a small ratio (e.g. 0.04), convert to percentage (4.0)
            if margin < 1.0 and margin != 0:
                margin *= 100
        
        # Lower threshold slightly to be more inclusive (e.g. 10% instead of 15%)
        if margin > 10: 
            score += 1
        elif margin < 0:
            score -= 1
            
        # Return discrete state
        return 0 if score >= 0 else 1

    def get_signal(self, narrative_vector: np.ndarray, price_history: pd.DataFrame, 
                   financial_data: dict, narrative_state: int = None,
                   position_ratio: float = 0.0,
                   structured_signals: np.ndarray = None):
        """
        Get the optimal action for the current state using enriched features.
        
        Args:
            narrative_vector: TF-IDF vectorized LLM output (shape: narrative_dim)
            price_history: DataFrame with stock price history
            financial_data: Dictionary with financial metrics
            narrative_state: Optional discrete narrative state (for backward compatibility)
            position_ratio: Fraction of portfolio currently invested (0.0=all cash, 1.0=fully invested)
        """
        # Extract stock features
        stock_features = self.extract_stock_features(price_history)
        
        # Determine Financial State
        fin_state = self.assess_financial_state(financial_data)
        
        # If narrative_state not provided, infer from vector
        if narrative_state is None:
            # Default to Neutral — np.mean(embedding) is unreliable for sentiment
            # The vectorizer's cosine-similarity method (used in backtest) is preferred
            narrative_state = 1  # Neutral
        
        # Lookup Base Q-Values from table
        state_key = (narrative_state, fin_state)
        base_q_values = self.q_table.get(state_key, {"BUY": 0.0, "HOLD": 0.5, "SELL": 0.0})
        
        # Adjust Q-Values based on stock features
        adjusted_q_values = self._adjust_q_values_with_features(base_q_values, stock_features)
        
        # Position-aware adjustments
        momentum = stock_features.get("momentum", 0)
        trend = stock_features.get("trend", 0)
        
        if position_ratio > 0.5:
            # Heavily invested — boost SELL, penalize BUY
            adjusted_q_values["SELL"] += 0.3 * position_ratio
            adjusted_q_values["BUY"] -= 0.5 * position_ratio
        elif position_ratio < 0.1:
            # No position — can't sell what we don't own
            adjusted_q_values["SELL"] -= 1.0
            
            # Only encourage buying if technicals support it
            if momentum > 2 or trend > 1:
                adjusted_q_values["BUY"] += 0.4
                adjusted_q_values["HOLD"] -= 0.2  # Penalize sitting on sidelines during uptrend
            elif momentum < -3 or trend < -2:
                # Technicals are bearish — sitting out IS the right move
                adjusted_q_values["HOLD"] += 0.1
            else:
                # Neutral technicals — slight nudge toward BUY
                adjusted_q_values["BUY"] += 0.15
        
        # Dampen extreme HOLD dominance from the Q-table (softer compression)
        hold_q = adjusted_q_values["HOLD"]
        max_action_q = max(adjusted_q_values["BUY"], adjusted_q_values["SELL"])
        hold_gap = hold_q - max_action_q
        if hold_gap > 0.8:
            # Compress only the excess — HOLD stays preferred but not insurmountable
            adjusted_q_values["HOLD"] -= (hold_gap - 0.8) * 0.3
        
        # Structured signals from multi-agent analysis
        if structured_signals is not None and len(structured_signals) >= 6:
            revenue_trend = float(structured_signals[0])
            margin_trend = float(structured_signals[1])
            peer_relative = float(structured_signals[2])
            hist_momentum = float(structured_signals[3])
            risk_level = float(structured_signals[4])
            overall_sentiment = float(structured_signals[5])
            
            # Sentiment-driven adjustments (strongest signal)
            if overall_sentiment > 0.3:
                adjusted_q_values["BUY"] += 0.25 * overall_sentiment
                adjusted_q_values["SELL"] -= 0.15 * overall_sentiment
            elif overall_sentiment < -0.3:
                adjusted_q_values["SELL"] += 0.25 * abs(overall_sentiment)
                adjusted_q_values["BUY"] -= 0.15 * abs(overall_sentiment)
            
            # Peer comparison: outperforming = hold/buy, underperforming = sell
            if peer_relative > 0.3:
                adjusted_q_values["BUY"] += 0.15 * peer_relative
                adjusted_q_values["HOLD"] += 0.1
            elif peer_relative < -0.3:
                adjusted_q_values["SELL"] += 0.15 * abs(peer_relative)
                adjusted_q_values["BUY"] -= 0.1
            
            # Historical momentum
            if hist_momentum > 0.3:
                adjusted_q_values["BUY"] += 0.1 * hist_momentum
            elif hist_momentum < -0.3:
                adjusted_q_values["SELL"] += 0.1 * abs(hist_momentum)
            
            # Risk level: high risk = more cautious
            if risk_level > 0.7:
                adjusted_q_values["HOLD"] += 0.15
                adjusted_q_values["BUY"] -= 0.1
        
        # Select Action using EPSILON-GREEDY (exploration vs exploitation)
        if self.enable_learning and np.random.random() < self.epsilon:
            # Explore: Choose random action
            best_action = np.random.choice(["BUY", "HOLD", "SELL"])
            confidence = adjusted_q_values.get(best_action, 0.5)
        else:
            # Exploit: Choose best action
            best_action = max(adjusted_q_values, key=adjusted_q_values.get)
            confidence = adjusted_q_values[best_action]
        
        # Normalize confidence for display
        confidence = max(0.1, min(1.0, abs(confidence)))

        # Enforce minimum sell confidence
        if best_action == "SELL" and confidence < 0.5:
             confidence = 0.5
        
        # Build detailed rationale
        rationale = self._build_rationale(narrative_state, fin_state, stock_features, best_action)
        
        # Store current state/action for learning (used on next update)
        self.current_state = state_key
        self.current_action = best_action
        
        return {
            "action": best_action,
            "confidence": confidence,
            "rationale": rationale,
            "narrative_state": narrative_state,
            "financial_state": fin_state,
            "stock_features": stock_features,
            "q_values": adjusted_q_values  # For debugging
        }
    
    def _adjust_q_values_with_features(self, base_q_values: dict, stock_features: dict) -> dict:
        """
        Adjust Q-Values based on technical indicators from stock price history.
        """
        adjusted = base_q_values.copy()
        
        momentum = stock_features["momentum"]
        trend = stock_features["trend"]
        rsi = stock_features["rsi"]
        volatility = stock_features["volatility"]
        
        # Momentum adjustments (stronger to overcome base priors)
        if momentum > 5:  # Strong positive momentum
            adjusted["BUY"] += 0.3
            adjusted["SELL"] -= 0.2
        elif momentum < -3:  # Negative momentum (lower threshold)
            adjusted["SELL"] += 0.4
            adjusted["BUY"] -= 0.3
        
        # Trend adjustments
        if trend > 2:  # Uptrend (lower threshold)
            adjusted["BUY"] += 0.2
            adjusted["SELL"] -= 0.15
        elif trend < -2:  # Downtrend (lower threshold)
            adjusted["SELL"] += 0.4
            adjusted["BUY"] -= 0.35
        
        # RSI adjustments (overbought/oversold)
        if rsi > 70:  # Overbought — strong SELL signal
            adjusted["SELL"] += 0.5
            adjusted["BUY"] -= 0.5
        elif rsi > 60:  # Moderately overbought
            adjusted["SELL"] += 0.2
            adjusted["BUY"] -= 0.15
        elif rsi < 30:  # Oversold — strong BUY signal
            adjusted["BUY"] += 0.4
            adjusted["SELL"] -= 0.3
        elif rsi < 40:  # Moderately oversold
            adjusted["BUY"] += 0.15
            adjusted["SELL"] -= 0.1
        
        # Volatility adjustments (high volatility = more caution)
        if volatility > 3:
            adjusted["HOLD"] += 0.15
            # In high vol, reduce extreme positions
            adjusted["BUY"] -= 0.05
            adjusted["SELL"] -= 0.05
        
        return adjusted
    
    def _build_rationale(self, narrative_state: int, fin_state: int, 
                        stock_features: dict, action: str) -> str:
        """Build a detailed rationale for the trading decision."""
        narrative_desc = ["Bullish", "Neutral", "Bearish"][narrative_state]
        fin_desc = ["Good", "Weak"][fin_state]
        
        rationale = f"State: Narrative={narrative_desc}, Financials={fin_desc}. "
        rationale += f"Stock Metrics: Momentum={stock_features['momentum']:.2f}%, "
        rationale += f"Trend={stock_features['trend']:.2f}%, RSI={stock_features['rsi']:.1f}. "
        rationale += f"Policy suggests {action}."
        
        return rationale
    
    # ==================== Q-LEARNING METHODS ====================
    
    def calculate_reward(self, previous_portfolio_value: float, 
                        current_portfolio_value: float, 
                        action: str) -> float:
        """
        Calculate reward for the action taken.
        
        Reward function considers:
        - Portfolio value change (profit/loss)
        - Penalize excessive risk
        - Small penalty for transaction costs
        
        Args:
            previous_portfolio_value: Portfolio value before trade
            current_portfolio_value: Portfolio value after trade execution
            action: Action taken (BUY, HOLD, SELL)
        
        Returns:
            Reward value (positive = good, negative = bad)
        """
        if previous_portfolio_value <= 0:
            return 0.0
        
        # Calculate percentage return
        pct_return = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value * 100
        
        # Base reward is the percentage return
        reward = pct_return
        
        # Small penalty for trading (transaction costs)
        if action in ["BUY", "SELL"]:
            reward -= 0.1  # 0.1% transaction cost penalty
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -10, 10)
        
        return reward
    
    def update(self, current_portfolio_value: float, 
               next_state_key: tuple, 
               next_q_values: dict):
        """
        Update Q-value using Q-learning (Bellman equation).
        
        Q(s,a) = Q(s,a) + α * [reward + γ * max(Q(s',a')) - Q(s,a)]
        
        Args:
            current_portfolio_value: Portfolio value after trade
            next_state_key: The state transitioned to (narrative, financial)
            next_q_values: Q-values for next state
        """
        if not self.enable_learning:
            return
        
        if self.current_state is None or self.current_action is None:
            return

        if self.last_state is None or self.last_action is None or self.last_portfolio_value is None:
            # First trade - initialize previous state/action
            self.last_state = self.current_state
            self.last_action = self.current_action
            self.last_portfolio_value = current_portfolio_value
            return
        
        # Calculate reward
        reward = self.calculate_reward(
            self.last_portfolio_value, 
            current_portfolio_value, 
            self.last_action
        )
        
        # Get current Q-value
        if self.last_state not in self.q_table:
            self.q_table[self.last_state] = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
        
        current_q = self.q_table[self.last_state][self.last_action]
        
        # Get max Q-value for next state
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # Update Q-table
        self.q_table[self.last_state][self.last_action] = new_q
        
        # Track learning progress
        self.learning_history.append({
            'state': self.last_state,
            'action': self.last_action,
            'reward': reward,
            'old_q': current_q,
            'new_q': new_q,
            'portfolio_value': current_portfolio_value
        })
        self.total_updates += 1
        
        # Update previous state/action for next iteration
        self.last_state = self.current_state
        self.last_action = self.current_action
        self.last_portfolio_value = current_portfolio_value
        
        # Optional: Print learning progress every 50 updates
        if self.total_updates % 50 == 0:
            avg_reward = np.mean([h['reward'] for h in self.learning_history[-50:]])
            print(f"   🧠 RL Update #{self.total_updates} | Avg Reward (last 50): {avg_reward:.3f} | Q-change: {new_q - current_q:.4f}")
    
    def get_learning_stats(self) -> dict:
        """Get statistics about the learning progress."""
        if len(self.learning_history) == 0:
            return {
                'total_updates': 0,
                'avg_reward': 0.0,
                'q_table_size': len(self.q_table),
                'exploration_rate': self.epsilon
            }
        
        rewards = [h['reward'] for h in self.learning_history]
        q_changes = [abs(h['new_q'] - h['old_q']) for h in self.learning_history]
        
        return {
            'total_updates': self.total_updates,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_q_change': np.mean(q_changes),
            'q_table_size': len(self.q_table),
            'exploration_rate': self.epsilon
        }
    
    def save_q_table(self, filepath: str):
        """Save learned Q-table to file."""
        import json
        
        # Convert tuple keys to strings for JSON serialization
        serializable_q_table = {
            f"{k[0]},{k[1]}": v for k, v in self.q_table.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump({
                'q_table': serializable_q_table,
                'learning_stats': self.get_learning_stats(),
                'params': {
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon
                }
            }, f, indent=2)
        
        print(f"💾 Q-table saved to {filepath}")
    
    def load_q_table(self, filepath: str):
        """Load learned Q-table from file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table = {
            tuple(map(int, k.split(','))): v 
            for k, v in data['q_table'].items()
        }
        
        print(f"📂 Q-table loaded from {filepath}")
        print(f"   Learned states: {len(self.q_table)}")
        print(f"   Total updates: {data['learning_stats']['total_updates']}")
