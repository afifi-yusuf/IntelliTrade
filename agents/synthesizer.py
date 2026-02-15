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

SYNTHESIS_PROMPT = """You are a senior investment strategist synthesizing analyses from multiple 
specialist agents into a final investment assessment.

You will receive:
- EXTRACTED FACTS: Key financial metrics from the latest earnings call
- PEER COMPARISON: How the company performs relative to sector peers
- HISTORICAL TRENDS: Multi-quarter trajectory and momentum
- MACRO ENVIRONMENT: Current economic conditions

Provide a CONCISE strategic synthesis (200-300 words) that:
1. Integrates peer comparison with historical context
2. Identifies the most important signals for trading decisions
3. Weighs bullish vs bearish factors

CRITICAL: On the LAST line, provide structured signals as floats between -1.0 and 1.0:
SIGNALS: revenue_trend=<float> margin_trend=<float> peer_relative=<float> historical_momentum=<float> risk_level=<float> overall_sentiment=<float>

Where:
- revenue_trend: Revenue growth direction (-1=declining, 1=accelerating)
- margin_trend: Margin trajectory (-1=compressing, 1=expanding)  
- peer_relative: Performance vs peers (-1=underperforming, 1=outperforming)
- historical_momentum: Multi-quarter momentum (-1=decelerating, 1=accelerating)
- risk_level: Risk assessment (0=low risk, 1=high risk)
- overall_sentiment: Net bullish/bearish assessment (-1=very bearish, 1=very bullish)"""


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
                temperature=0.3,
                max_tokens=800
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
