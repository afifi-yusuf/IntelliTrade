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
    """
    def __init__(self, narrative_dim=100):
        # Q-Table: Maps (Narrative_State, Financial_State) -> {Action: Expected_Reward}
        # Initialized with some heuristic priors for the demo
        self.q_table = {}
        self.narrative_dim = narrative_dim
        self._initialize_priors()

    def _initialize_priors(self):
        """
        Initialize Q-Values with common sense financial logic.
        States:
          Narrative: 0 (Bullish), 1 (Neutral), 2 (Bearish)
          Financial: 0 (Good), 1 (Bad)
        """
        # Format: (Narrative, Financial) -> {Action: Score}
        
        # 1. Bullish Narrative + Good Financials -> Strong BUY
        self.q_table[(0, 0)] = {
            "BUY": 0.9, "HOLD": 0.1, "SELL": -0.8
        }
        
        # 2. Bullish Narrative + Bad Financials -> Speculative (HOLD/BUY)
        self.q_table[(0, 1)] = {
            "BUY": 0.4, "HOLD": 0.5, "SELL": -0.2
        }
        
        # 3. Bearish Narrative + Good Financials -> Value Trap? (HOLD)
        self.q_table[(2, 0)] = {
            "BUY": -0.1, "HOLD": 0.8, "SELL": 0.2
        }
        
        # 4. Bearish Narrative + Bad Financials -> SELL
        self.q_table[(2, 1)] = {
            "BUY": -0.9, "HOLD": -0.2, "SELL": 0.9
        }
        
        # Defaults for others (Neutral)
        self.q_table[(1, 0)] = {"BUY": 0.6, "HOLD": 0.3, "SELL": -0.1}
        self.q_table[(1, 1)] = {"BUY": -0.2, "HOLD": 0.6, "SELL": 0.4}

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
        margin = financials.get("netProfitMarginTTM", 0)
        if margin > 15: # Healthy Tech Margin
            score += 1
        elif margin < 0:
            score -= 1
            
        # Return discrete state
        return 0 if score >= 0 else 1

    def get_signal(self, narrative_vector: np.ndarray, price_history: pd.DataFrame, 
                   financial_data: dict, narrative_state: int = None):
        """
        Get the optimal action for the current state using enriched features.
        
        Args:
            narrative_vector: TF-IDF vectorized LLM output (shape: narrative_dim)
            price_history: DataFrame with stock price history
            financial_data: Dictionary with financial metrics
            narrative_state: Optional discrete narrative state (for backward compatibility)
        """
        # Extract stock features
        stock_features = self.extract_stock_features(price_history)
        
        # Determine Financial State
        fin_state = self.assess_financial_state(financial_data)
        
        # If narrative_state not provided, infer from vector
        if narrative_state is None:
            # Simple clustering: compute weighted sum to infer bullish/bearish
            if narrative_vector is not None and len(narrative_vector) > 0:
                vector_score = np.mean(narrative_vector)
                if vector_score > 0.05:
                    narrative_state = 0  # Bullish
                elif vector_score < 0.02:
                    narrative_state = 2  # Bearish
                else:
                    narrative_state = 1  # Neutral
            else:
                narrative_state = 1
        
        # Lookup Base Q-Values from table
        state_key = (narrative_state, fin_state)
        base_q_values = self.q_table.get(state_key, {"BUY": 0.0, "HOLD": 1.0, "SELL": 0.0})
        
        # Adjust Q-Values based on stock features
        adjusted_q_values = self._adjust_q_values_with_features(base_q_values, stock_features)
        
        # Select Action with Max Q-Value
        best_action = max(adjusted_q_values, key=adjusted_q_values.get)
        confidence = adjusted_q_values[best_action]
        
        # Normalize confidence for display
        confidence = max(0.1, min(1.0, abs(confidence)))
        
        # Build detailed rationale
        rationale = self._build_rationale(narrative_state, fin_state, stock_features, best_action)
        
        return {
            "action": best_action,
            "confidence": confidence,
            "rationale": rationale,
            "stock_features": stock_features
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
        
        # Momentum adjustments
        if momentum > 5:  # Strong positive momentum
            adjusted["BUY"] += 0.2
            adjusted["SELL"] -= 0.2
        elif momentum < -5:  # Strong negative momentum
            adjusted["SELL"] += 0.2
            adjusted["BUY"] -= 0.2
        
        # Trend adjustments
        if trend > 3:  # Uptrend
            adjusted["BUY"] += 0.15
            adjusted["SELL"] -= 0.1
        elif trend < -3:  # Downtrend
            adjusted["SELL"] += 0.15
            adjusted["BUY"] -= 0.1
        
        # RSI adjustments (overbought/oversold)
        if rsi > 70:  # Overbought
            adjusted["SELL"] += 0.1
            adjusted["BUY"] -= 0.15
        elif rsi < 30:  # Oversold
            adjusted["BUY"] += 0.1
            adjusted["SELL"] -= 0.15
        
        # Volatility adjustments (high volatility = more caution)
        if volatility > 3:
            adjusted["HOLD"] += 0.1
        
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
