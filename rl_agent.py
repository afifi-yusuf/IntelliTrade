import numpy as np
import pandas as pd

class MarkovRLAgent:
    """
    A simple Reinforcement Learning Agent based on a Markov Decision Process (MDP).
    
    State Space (S): (Narrative_State, Financial_State)
    Action Space (A): BUY (1), HOLD (0), SELL (-1)
    
    Since we don't have a live simulation environment to 'train' episodically,
    we use a 'Q-Table Loopup' approach based on historical probabilities.
    """
    def __init__(self):
        # Q-Table: Maps (Narrative_State, Financial_State) -> {Action: Expected_Reward}
        # Initialized with some heuristic priors for the demo
        self.q_table = {}
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

    def get_signal(self, narrative_state: int, financial_data: dict):
        """
        Get the optimal action for the current state.
        """
        # Determine Financial State
        fin_state = self.assess_financial_state(financial_data)
        
        # Lookup Q-Values
        state_key = (narrative_state, fin_state)
        q_values = self.q_table.get(state_key, {"BUY": 0.0, "HOLD": 1.0, "SELL": 0.0})
        
        # Select Action with Max Q-Value
        best_action = max(q_values, key=q_values.get)
        confidence = q_values[best_action]
        
        # Normalize confidence for display (simple logic)
        confidence = max(0.1, abs(confidence))
        
        return {
            "action": best_action,
            "confidence": confidence,
            "rationale": f"State Detected: Narrative={narrative_state} (0=Bull, 2=Bear), Financials={fin_state} (0=Good). Historical Q-Policy suggests {best_action}."
        }
