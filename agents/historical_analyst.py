"""
Historical Analyst Agent
========================
Analyzes company's own historical patterns by traversing the Neo4j knowledge graph.
Combines vector similarity search with graph traversal (QoQ_COMPARES edges).
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

HISTORICAL_ANALYSIS_PROMPT = """You are a financial trend analyst specializing in quarter-over-quarter 
pattern recognition. Analyze the company's current facts in the context of its own historical data.

Focus on:
1. Multi-quarter TRENDS (accelerating/decelerating revenue, margin expansion/compression)
2. Seasonal PATTERNS (if visible)
3. How current performance compares to the company's own trajectory
4. Whether management guidance has been reliable historically

Be specific — reference actual numbers and quarters. Keep to 150-200 words.

On the LAST line, output two scores as floats between -1.0 and 1.0:
- revenue_trend: revenue momentum direction
- margin_trend: margin trajectory direction 
- momentum: overall multi-quarter momentum

Format: TRENDS: revenue=<float> margin=<float> momentum=<float>"""


class HistoricalAnalystAgent:
    """
    Analyzes company's own historical patterns using Neo4j graph traversal.
    Combines QoQ_COMPARES edge traversal with vector similarity search.
    """
    
    def __init__(self, graph_db=None, azure_client: Optional[AzureOpenAI] = None):
        self.graph_db = graph_db
        if azure_client:
            self.client = azure_client
        else:
            self.client = AzureOpenAI()
        self.model = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
    
    def analyze(self, facts: List[Dict], ticker: str, quarter: str,
                embedder=None) -> Dict[str, Any]:
        """
        Analyze current facts against company's historical data.
        
        Args:
            facts: Extracted facts for the current quarter
            ticker: Company ticker
            quarter: Current quarter
            embedder: Vectorizer for embedding creation
            
        Returns:
            {"analysis": str, "revenue_trend": float, "margin_trend": float, 
             "momentum": float, "historical_facts": List}
        """
        historical_facts = []
        metric_trends = {}
        
        # Method 1: Graph traversal for QoQ trends
        if self.graph_db:
            metric_trends = self._get_metric_trends(ticker, quarter)
        
        # Method 2: Vector similarity for semantically similar past facts
        if self.graph_db and embedder and facts:
            historical_facts = self._search_historical_facts(facts, ticker, quarter, embedder)
        
        # Method 3: Fallback to retrieve_all_context
        if not historical_facts and not metric_trends and self.graph_db:
            historical_facts = self._fallback_historical(ticker, quarter)
        
        # Build LLM context
        current_text = self._format_current_facts(facts, ticker, quarter)
        history_text = self._format_historical(historical_facts, metric_trends)
        
        if not history_text.strip() or history_text.strip() == "No historical data available.":
            return {
                "analysis": "Insufficient historical data for trend analysis.",
                "revenue_trend": 0.0,
                "margin_trend": 0.0,
                "momentum": 0.0,
                "historical_facts": []
            }
        
        prompt = f"""CURRENT QUARTER: {ticker} {quarter}
{current_text}

HISTORICAL DATA FROM KNOWLEDGE GRAPH:
{history_text}

Analyze the historical trends and current trajectory:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": HISTORICAL_ANALYSIS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            raw = response.choices[0].message.content.strip()
            analysis, trends = self._parse_response(raw)
            
            return {
                "analysis": analysis,
                "revenue_trend": trends.get("revenue", 0.0),
                "margin_trend": trends.get("margin", 0.0),
                "momentum": trends.get("momentum", 0.0),
                "historical_facts": historical_facts
            }
            
        except Exception as e:
            logger.error(f"Historical analysis failed for {ticker}: {e}")
            return {
                "analysis": f"Historical analysis error: {e}",
                "revenue_trend": 0.0,
                "margin_trend": 0.0,
                "momentum": 0.0,
                "historical_facts": []
            }
    
    def _get_metric_trends(self, ticker: str, quarter: str) -> Dict[str, List[Dict]]:
        """Get QoQ metric trends via graph traversal."""
        trends = {}
        
        key_metrics = ["Revenue", "Net Income", "Operating Margin", "EPS", "Gross Margin"]
        
        for metric in key_metrics:
            try:
                if hasattr(self.graph_db, 'get_metric_trend'):
                    trend = self.graph_db.get_metric_trend(ticker, metric, num_quarters=4)
                    if trend:
                        trends[metric] = trend
            except Exception as e:
                logger.debug(f"Could not get trend for {metric}: {e}")
        
        return trends
    
    def _search_historical_facts(self, facts: List[Dict], ticker: str,
                                  quarter: str, embedder) -> List[Dict]:
        """Find semantically similar past facts using vector search."""
        if not hasattr(self.graph_db, 'search_historical_facts'):
            return []
        
        # Create query from key current facts
        fact_text = " | ".join([f"{f['metric']}: {f['value']}" for f in facts[:5]])
        
        try:
            query_embedding = embedder.get_embedding(fact_text)
            if query_embedding is not None:
                results = self.graph_db.search_historical_facts(
                    query_embedding=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                    ticker=ticker,
                    max_quarter=quarter,
                    top_k=10
                )
                return results
        except Exception as e:
            logger.warning(f"Historical vector search failed: {e}")
        
        return []
    
    def _fallback_historical(self, ticker: str, quarter: str) -> List[Dict]:
        """Fallback: use existing retrieve_all_context as text."""
        try:
            context = self.graph_db.retrieve_all_context(ticker, max_quarter=quarter)
            if context and len(context) > 50:
                return [{"type": "text_context", "content": context[:3000]}]
        except Exception as e:
            logger.debug(f"Fallback historical retrieval failed: {e}")
        return []
    
    def _format_current_facts(self, facts: List[Dict], ticker: str, quarter: str) -> str:
        """Format current quarter facts."""
        lines = [f"Facts for {ticker} ({quarter}):"]
        for f in facts:
            line = f"  • {f['metric']}: {f['value']}"
            if f.get('change'):
                line += f" ({f['change']})"
            lines.append(line)
        return "\n".join(lines)
    
    def _format_historical(self, historical_facts: List[Dict], 
                           metric_trends: Dict[str, List[Dict]]) -> str:
        """Format historical data for LLM context."""
        lines = []
        
        # Format metric trends (from graph traversal)
        if metric_trends:
            lines.append("METRIC TRENDS (QoQ from Knowledge Graph):")
            for metric, trend_data in metric_trends.items():
                trend_str = " → ".join([
                    f"{t.get('quarter', '?')}: {t.get('value', '?')}"
                    for t in trend_data
                ])
                lines.append(f"  {metric}: {trend_str}")
        
        # Format similar historical facts (from vector search)
        if historical_facts:
            lines.append("\nSIMILAR PAST FACTS (Semantic Search):")
            for f in historical_facts:
                if f.get("type") == "text_context":
                    lines.append(f.get("content", ""))
                else:
                    metric = f.get("metric", f.get("metric_name", "?"))
                    value = f.get("value", "?")
                    quarter = f.get("quarter", "?")
                    lines.append(f"  • {quarter} — {metric}: {value}")
        
        return "\n".join(lines) if lines else "No historical data available."
    
    def _parse_response(self, raw: str) -> tuple:
        """Parse LLM response to extract analysis and trend scores."""
        lines = raw.strip().split("\n")
        trends = {"revenue": 0.0, "margin": 0.0, "momentum": 0.0}
        analysis_lines = []
        
        for line in lines:
            if line.strip().startswith("TRENDS:"):
                try:
                    parts = line.split(":", 1)[1].strip()
                    for kv in parts.split():
                        if "=" in kv:
                            k, v = kv.split("=")
                            k = k.strip().lower()
                            v = float(v.strip())
                            v = max(-1.0, min(1.0, v))
                            if k in trends:
                                trends[k] = v
                except (ValueError, IndexError):
                    pass
            else:
                analysis_lines.append(line)
        
        return "\n".join(analysis_lines).strip(), trends
