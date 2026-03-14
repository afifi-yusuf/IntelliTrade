"""
Synthesizer Agent
=================
Combines outputs from all specialist agents into a final analysis with structured signals.
The structured signals feed directly into the RL agent's state vector.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are a senior quantitative investment strategist. Your structured signals \
feed directly into a reinforcement learning trading agent — numerical precision and full range \
utilisation matter MORE than qualitative hedging.

You will receive:
- EXTRACTED FACTS: Key financial metrics from the latest earnings call
- PEER COMPARISON: How the company performs relative to sector peers
- HISTORICAL TRENDS: Multi-quarter trajectory and momentum
- MACRO ENVIRONMENT: Current economic conditions

Provide a CONCISE strategic synthesis (150-200 words) focusing on the most actionable findings, \
then output calibrated signals.

SIGNAL CALIBRATION RULES — read carefully:
1. USE THE FULL RANGE. If revenue is accelerating and margins are expanding, score them near +0.8 or \
+0.9, not a cautious +0.2. If fundamentals are genuinely deteriorating, use -0.6 to -0.9.
2. ANCHOR AGAINST HISTORY. A company growing revenue 4% YoY when it grew 25% last year is a \
negative revenue_trend (-0.4 to -0.6), not neutral.
3. BE SPECIFIC. "Revenue beat by 8% and accelerated to 22% YoY vs 14% last quarter" → revenue_trend=+0.75. \
"Revenue missed and decelerated from 18% to 9% YoY" → revenue_trend=-0.55.
4. risk_level: 0.5 is average/neutral risk. Only use values below 0.3 for clearly low-risk situations \
or above 0.7 for genuine elevated risk (guidance cut, regulatory threat, leverage concern).
5. peer_relative: Only output 0.0 if peer data is genuinely absent — otherwise quantify the gap.

Signal definitions and calibration anchors:
- revenue_trend:        -0.9=revenue collapsing, -0.5=decelerating, 0=flat/in-line, +0.5=solid beat, +0.9=blowout growth
- margin_trend:         -0.9=severe compression (-5pp+), -0.4=mild compression, 0=stable, +0.4=modest expansion, +0.9=strong expansion
- peer_relative:        -0.9=clear sector laggard, -0.4=slight underperform, 0=in-line, +0.4=slight outperform, +0.9=sector leader
- historical_momentum:  -0.9=multi-quarter deterioration, -0.4=slowing, 0=stable, +0.4=improving, +0.9=strong multi-quarter acceleration
- risk_level:           0.1=very low risk, 0.3=below average, 0.5=neutral, 0.7=elevated, 0.9=high risk
- overall_sentiment:    -0.9=strong sell, -0.5=bearish, 0=neutral, +0.5=bullish, +0.9=strong buy

CRITICAL: Output structured signals on the LAST line in this exact format:
SIGNALS: revenue_trend=<float> margin_trend=<float> peer_relative=<float> historical_momentum=<float> risk_level=<float> overall_sentiment=<float>"""


class SynthesizerAgent:
    """
    Combines all specialist agent outputs into a final analysis with structured signals.
    The structured signals provide interpretable features for the RL agent.
    """
    
    def __init__(self, azure_client: Optional[AzureOpenAI] = None):
        if azure_client:
            self.client = azure_client
        else:
            self.client = AzureOpenAI()
        self.model = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
    
    def synthesize(self, ticker: str, quarter: str,
                   facts: List[Dict],
                   peer_result: Dict[str, Any],
                   historical_result: Dict[str, Any],
                   economic_data: Dict = None) -> Dict[str, Any]:
        """
        Synthesize all agent outputs into final analysis + structured signals.
        
        Args:
            ticker: Company ticker
            quarter: Current quarter
            facts: Extracted facts from FactExtractorAgent
            peer_result: Output from PeerAnalystAgent
            historical_result: Output from HistoricalAnalystAgent
            economic_data: Economic indicators dict
            
        Returns:
            {
                "narrative": str,          # Full synthesis text
                "signals": {               # Structured signals for RL
                    "revenue_trend": float,
                    "margin_trend": float,
                    "peer_relative": float,
                    "historical_momentum": float,
                    "risk_level": float,
                    "overall_sentiment": float
                }
            }
        """
        # Build context sections
        facts_section = self._format_facts(facts, ticker, quarter)
        peer_section = self._format_peer(peer_result)
        historical_section = self._format_historical(historical_result)
        macro_section = self._format_macro(economic_data)
        
        prompt = f"""COMPANY: {ticker} | QUARTER: {quarter}

=== EXTRACTED FACTS ===
{facts_section}

=== PEER COMPARISON (from Knowledge Graph) ===
{peer_section}

=== HISTORICAL TRENDS (from Knowledge Graph) ===
{historical_section}

=== MACRO ENVIRONMENT ===
{macro_section}

Synthesize all inputs into a strategic assessment:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYNTHESIS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_completion_tokens=800
            )
            
            raw = response.choices[0].message.content.strip()
            narrative, signals = self._parse_response(raw)
            
            # Merge with signals from specialist agents if available
            signals = self._merge_signals(signals, peer_result, historical_result)
            
            return {
                "narrative": narrative,
                "signals": signals
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed for {ticker}: {e}")
            return self._fallback_result(facts, peer_result, historical_result)
    
    def _format_facts(self, facts: List[Dict], ticker: str, quarter: str) -> str:
        """Format extracted facts."""
        if not facts:
            return "No facts extracted for this quarter."
        
        lines = []
        for f in facts:
            line = f"• {f['metric']}: {f['value']}"
            if f.get('change'):
                line += f" ({f['change']})"
            if f.get('sentiment'):
                line += f" [{f['sentiment']}]"
            lines.append(line)
        return "\n".join(lines)
    
    def _format_peer(self, peer_result: Dict) -> str:
        """Format peer analysis."""
        analysis = peer_result.get("analysis", "No peer data available.")
        score = peer_result.get("peer_score", 0.0)
        return f"{analysis}\n(Peer Relative Score: {score:+.2f})"
    
    def _format_historical(self, historical_result: Dict) -> str:
        """Format historical analysis."""
        analysis = historical_result.get("analysis", "No historical data available.")
        rev = historical_result.get("revenue_trend", 0.0)
        margin = historical_result.get("margin_trend", 0.0)
        momentum = historical_result.get("momentum", 0.0)
        return f"{analysis}\n(Revenue Trend: {rev:+.2f}, Margin Trend: {margin:+.2f}, Momentum: {momentum:+.2f})"
    
    def _format_macro(self, economic_data: Dict) -> str:
        """Format economic data."""
        if not economic_data:
            return "No economic data available."
        
        lines = []
        for key, value in economic_data.items():
            if value is not None:
                display_key = key.replace("_", " ").title()
                lines.append(f"• {display_key}: {value}")
        return "\n".join(lines) if lines else "No economic data available."
    
    def _parse_response(self, raw: str) -> tuple:
        """Parse synthesis response to extract narrative and signals."""
        default_signals = {
            "revenue_trend": 0.0,
            "margin_trend": 0.0,
            "peer_relative": 0.0,
            "historical_momentum": 0.0,
            "risk_level": 0.5,
            "overall_sentiment": 0.0
        }
        
        lines = raw.strip().split("\n")
        narrative_lines = []
        signals = default_signals.copy()
        
        for line in lines:
            if line.strip().startswith("SIGNALS:"):
                try:
                    parts = line.split(":", 1)[1].strip()
                    for kv in parts.split():
                        if "=" in kv:
                            k, v = kv.split("=")
                            k = k.strip().lower()
                            v = float(v.strip())
                            v = max(-1.0, min(1.0, v))
                            if k in signals:
                                signals[k] = v
                except (ValueError, IndexError):
                    pass
            else:
                narrative_lines.append(line)
        
        return "\n".join(narrative_lines).strip(), signals
    
    def _merge_signals(self, synth_signals: Dict, peer_result: Dict, 
                       historical_result: Dict) -> Dict[str, float]:
        """
        Merge synthesizer's signals with those from specialist agents.
        Uses averaging when both have values, otherwise takes whichever is non-zero.
        """
        # Use peer agent's score if synthesizer didn't provide one
        if synth_signals["peer_relative"] == 0.0:
            synth_signals["peer_relative"] = peer_result.get("peer_score", 0.0)
        else:
            # Average the two
            peer_score = peer_result.get("peer_score", 0.0)
            if peer_score != 0.0:
                synth_signals["peer_relative"] = (synth_signals["peer_relative"] + peer_score) / 2
        
        # Use historical agent's trends if available
        hist_rev = historical_result.get("revenue_trend", 0.0)
        hist_margin = historical_result.get("margin_trend", 0.0)
        hist_mom = historical_result.get("momentum", 0.0)
        
        if synth_signals["revenue_trend"] == 0.0:
            synth_signals["revenue_trend"] = hist_rev
        elif hist_rev != 0.0:
            synth_signals["revenue_trend"] = (synth_signals["revenue_trend"] + hist_rev) / 2
        
        if synth_signals["margin_trend"] == 0.0:
            synth_signals["margin_trend"] = hist_margin
        elif hist_margin != 0.0:
            synth_signals["margin_trend"] = (synth_signals["margin_trend"] + hist_margin) / 2
        
        if synth_signals["historical_momentum"] == 0.0:
            synth_signals["historical_momentum"] = hist_mom
        elif hist_mom != 0.0:
            synth_signals["historical_momentum"] = (synth_signals["historical_momentum"] + hist_mom) / 2
        
        return synth_signals
    
    def _fallback_result(self, facts: List[Dict], peer_result: Dict, 
                         historical_result: Dict) -> Dict[str, Any]:
        """Fallback when synthesis LLM call fails."""
        # Construct basic narrative from available data
        narrative_parts = []
        if facts:
            fact_strs = [f"{f['metric']}: {f['value']}" for f in facts[:3]]
            narrative_parts.append(f"Key facts: {', '.join(fact_strs)}")
        
        peer_analysis = peer_result.get("analysis", "")
        if peer_analysis:
            narrative_parts.append(f"Peer context: {peer_analysis[:200]}")
        
        hist_analysis = historical_result.get("analysis", "")
        if hist_analysis:
            narrative_parts.append(f"Trends: {hist_analysis[:200]}")
        
        # Construct signals from specialist agents
        signals = {
            "revenue_trend": historical_result.get("revenue_trend", 0.0),
            "margin_trend": historical_result.get("margin_trend", 0.0),
            "peer_relative": peer_result.get("peer_score", 0.0),
            "historical_momentum": historical_result.get("momentum", 0.0),
            "risk_level": 0.5,
            "overall_sentiment": 0.0
        }
        
        return {
            "narrative": " | ".join(narrative_parts) if narrative_parts else "Synthesis unavailable.",
            "signals": signals
        }
